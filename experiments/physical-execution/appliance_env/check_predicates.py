"""P1 predicate check: appliance_env's re-implementation of RoboCasa's appliance predicates.

1. Threshold cases set directly on the joints (RoboCasa's own rules, quoted in appliance_env):
   burner on iff 0.35 <= angle mod 2pi <= 2pi - 0.35; door fraction (q - lo) / (hi - lo), flipped
   when lo < 0, open at >= 0.90, closed at <= 0.005; preheated at temperature >= 0.25.
2. Physical microwave button presses through the controller: closed fingers pushed into the start
   button turn the microwave on, the stop button turns it off, opening the door turns it off;
   `*_far` reads False while the hand is at the button and True 0.15 m away.

    MUJOCO_GL=osmesa python -I check_predicates.py --assets ~/arb-data/robocasa-assets/extracted/fixtures
"""
import argparse
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appliance_env as E  # noqa: E402

FAILS = []
MW_MOUNT = {"front_x": 0.55, "robot_z": 0.65}


def check(label, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + label + ("  " + detail if detail else ""), flush=True)
    if not cond:
        FAILS.append(label)


def rotvec(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(9))
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def goto(env, target, approach, grip, steps=120, tol=0.004):
    """Drive the gripper site to `target` with its z axis along `approach` (privileged scripted motion)."""
    z = approach / np.linalg.norm(approach)
    y = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rd = np.column_stack([x, y, z])
    for _ in range(steps):
        d = env.data
        p = d.site_xpos[env.site]
        R = d.site_xmat[env.site].reshape(3, 3)
        dp = target - p
        dr = rotvec(Rd @ R.T)
        env.step(list(dp) + list(dr) + [grip, 0])
        if np.linalg.norm(target - env.data.site_xpos[env.site]) < tol and np.linalg.norm(dr) < 0.02:
            break
    return np.linalg.norm(target - env.data.site_xpos[env.site])


def thresholds(assets):
    env = E.ApplianceEnv(E.unit_spec("stoves/Stove026"), assets, render=False, jitter_scale=(0, 0, 0))
    env.reset(0)
    loc = next(iter(env.unit.knobs))
    for val, want in ((0.0, False), (0.349, False), (0.351, True), (3.0, True), (2 * math.pi - 0.36, True),
                      (-1e-6, False)):
        env.set_joint("knob_%s_joint" % loc, val)
        check("burner %s at %.4f rad -> on=%s" % (loc, val, want), env.unit_state()["burner_on_" + loc] == want)
    qa, lo, hi, _ = env.unit.door
    for frac, op, cl in ((0.0, False, True), (0.004, False, True), (0.5, False, False), (0.9, True, False)):
        env.set_joint("door_joint", lo + frac * (hi - lo))
        s = env.unit_state()
        check("range door at fraction %.3f -> open=%s closed=%s" % (frac, op, cl),
              s["door_open"] == op and s["door_closed"] == cl, str(s["door_fraction"]))
    tq, tlo, thi = env.unit.temp_knob
    for frac, want in ((0.24, False), (0.26, True)):
        env.set_joint("knob_temp_joint", tlo + frac * (thi - tlo))
        check("temperature %.2f -> preheated=%s" % (frac, want), env.unit_state()["preheated"] == want)
    env = E.ApplianceEnv(E.unit_spec("microwaves/Microwave011"), assets, render=False, jitter_scale=(0, 0, 0))
    env.reset(0)
    qa, lo, hi, _ = env.unit.door                     # range (-pi/2, 0): fraction flips (RoboCasa)
    for q, frac in ((0.0, 0.0), (lo, 1.0), (lo * 0.9, 0.9)):
        env.set_joint("microjoint", q)
        check("microwave door q=%.3f -> fraction %.2f" % (q, frac),
              abs(env.unit_state()["door_fraction"] - frac) < 1e-6, str(env.unit_state()["door_fraction"]))


def press(env, button, approach):
    gid = env.unit.buttons[button]
    p = env.data.geom_xpos[gid].copy()
    goto(env, p - 0.10 * approach, approach, 1.0)
    at = goto(env, p + 0.012 * approach, approach, 1.0, steps=60)     # push 12 mm past the face
    far_at = env.unit_state()[button + "_far"]
    goto(env, p - 0.20 * approach, approach, 1.0)
    return at, far_at


def microwave(assets):
    env = E.ApplianceEnv(E.unit_spec("microwaves/Microwave011"), assets, render=False, jitter_scale=(0, 0, 0),
                         mount=MW_MOUNT)
    env.reset(0)
    yaw = env.layout["unit_yaw"]
    into = np.array([math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2), 0.0])
    # Tilted 35 degrees down: this model's buttons sit 3 cm above the counter, and a level hand meets
    # the counter top with its lower edge before the fingertips reach the panel.
    side = np.cross([0.0, 0.0, 1.0], into)          # rotating about up x into by +35 deg tilts it down
    app = E.axis_angle_mat(side / np.linalg.norm(side) * math.radians(35)) @ into
    check("microwave starts off", env.unit_state()["turned_on"] is False)
    err, far_at = press(env, "start_button", app)
    s = env.unit_state()
    check("press start -> on", s["turned_on"] is True, "hand-to-target %.3f m" % err)
    check("start_button_far False at the button", far_at is False)
    check("start_button_far True after backing off", s["start_button_far"] is True)
    err, _ = press(env, "stop_button", app)
    check("press stop -> off", env.unit_state()["turned_on"] is False, "hand-to-target %.3f m" % err)
    press(env, "start_button", app)
    check("press start again -> on", env.unit_state()["turned_on"] is True)
    env.set_joint("microjoint", env.unit.door[1])    # door fully open
    env.step([0, 0, 0, 0, 0, 0, 1, 0])
    check("door open -> off", env.unit_state()["turned_on"] is False)
    env.set_joint("microjoint", 0.0)
    press(env, "start_button", app)
    check("start after closing the door -> on", env.unit_state()["turned_on"] is True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    a = ap.parse_args()
    assets = os.path.expanduser(a.assets)
    thresholds(assets)
    microwave(assets)
    print("FAILED: %s" % FAILS if FAILS else "ALL PASS")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
