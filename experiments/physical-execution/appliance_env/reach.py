"""P1 reach study: can the fixed Panda reach every control of every unit, with a sensible approach?

Kinematic only (no physics, no collision check): damped least-squares IK on the gripper site, from
the home pose, for a target point and an approach direction (the gripper's z axis, the direction
the fingers point). A target is reachable if the IK converges within 1 cm and 10 degrees inside
the joint limits. Targets per unit:

  * a burner knob (each): its joint anchor; approach along the knob axis into the panel (range,
    front panel) or straight down (cooktop, knobs on top);
  * the temperature knob (oven, range): the same;
  * a microwave button: its geom centre, approach into the front face;
  * the door handle (`door_handle_main`) closed and part-open: 0 and 0.5 for a microwave's side-hinged
    door; 0 and 0.3 for a drop-down oven door, which falls open under its own weight past a few
    degrees (pilot finding), so the hand never needs to follow it down; approach into the front.

Each target is also checked 8 cm before it along the approach (the pre-contact point). An approach
may also be tilted 30 degrees down or 30 degrees to either side of the nominal one (a pinch or a
handle grasp works from those too); the target counts as reachable if any of them works.

    python -I reach.py --assets ~/arb-data/robocasa-assets/extracted/fixtures --out ../outputs/p1/reach.json
"""
import argparse
import json
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appliance_env as E  # noqa: E402

KINDS = ("stovetops", "microwaves", "ovens", "stoves")
POS_TOL = 0.01
ANG_TOL = math.radians(10)
PRE = 0.08
RESTARTS = 9                       # the home pose, then 8 fixed pseudo-random joint configurations
RNG_STARTS = np.random.default_rng(20261112).uniform(0.15, 0.85, (RESTARTS - 1, 7))


def ik(m, d, site, jq, jv, lo, hi, target, approach, iters=300):
    """Position + approach-axis IK from the current d.qpos (modified in place). Returns (pos_err, ang_err)."""
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    approach = approach / np.linalg.norm(approach)
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = d.site_xpos[site]
        z = d.site_xmat[site].reshape(3, 3)[:, 2]
        ep = target - p
        er = np.cross(z, approach)
        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
        mujoco.mj_jacSite(m, d, jp, jr, site)
        J = np.vstack([jp[:, jv], 0.5 * jr[:, jv]])
        e = np.concatenate([ep, 0.5 * er])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        dq = np.clip(dq, -0.2, 0.2)
        d.qpos[jq] = np.clip(d.qpos[jq] + dq, lo, hi)
    mujoco.mj_kinematics(m, d)
    p = d.site_xpos[site]
    z = d.site_xmat[site].reshape(3, 3)[:, 2]
    ang = math.acos(max(-1.0, min(1.0, float(z @ approach))))
    return float(np.linalg.norm(target - p)), ang


def targets(env):
    """(label, point, approach) in world coordinates, for the unit as built."""
    m, d = env.model, env.data
    u = env.unit
    out = []
    mujoco.mj_forward(m, d)
    yaw = env.layout["unit_yaw"]
    into_front = np.array([math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2), 0.0])   # model +y, turned
    for loc, (qa, _dof) in u.knobs.items():
        j = m.joint("u_knob_%s_joint" % loc).id
        axis = d.xaxis[j].copy()
        if abs(axis[2]) > 0.7:
            app = np.array([0.0, 0.0, -1.0])
        else:
            app = into_front
        out.append(("knob_" + loc, d.xanchor[j].copy(), app))
    if u.temp_knob is not None:
        j = m.joint("u_knob_temp_joint").id
        out.append(("knob_temp", d.xanchor[j].copy(), into_front))
    for b, gid in u.buttons.items():
        out.append((b, d.geom_xpos[gid].copy(), into_front))
    if u.door is not None and u.handle is not None:
        qa, lo, hi, _dof = u.door
        q0 = d.qpos[qa]
        for frac in ((0.0, 0.5) if u.kind == "microwave" else (0.0, 0.3)):
            q = lo + frac * (hi - lo) if lo >= 0 else hi - frac * (hi - lo)
            d.qpos[qa] = q
            mujoco.mj_forward(m, d)
            out.append(("handle_%.1f" % frac, d.geom_xpos[u.handle].copy(), into_front))
        d.qpos[qa] = q0
        mujoco.mj_forward(m, d)
    return out


def study(env):
    m = env.model
    lo = m.jnt_range[[m.joint("r_joint%d" % i).id for i in range(1, 8)], 0]
    hi = m.jnt_range[[m.joint("r_joint%d" % i).id for i in range(1, 8)], 1]
    res = []
    for label, point, app0 in targets(env):
        row = None
        for tilt in tilts(app0):
            r = one_target(env, m, lo, hi, label, point, tilt)
            if row is None or r["ok"]:
                row = r
            if r["ok"]:
                break
        res.append(row)
    return res


def tilts(app):
    """The nominal approach, then 30 degrees down, then 30 degrees to either side."""
    app = app / np.linalg.norm(app)
    out = [app]
    side = np.cross([0.0, 0.0, 1.0], app)          # rotating about up x app by +30 deg tilts it down
    if np.linalg.norm(side) < 1e-6:                 # a top-down approach: tilt towards and sideways
        for v in ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]):
            out.append(E.axis_angle_mat(np.cross(app, v) / np.linalg.norm(np.cross(app, v)) * math.radians(30)) @ app)
        return out
    side /= np.linalg.norm(side)
    out.append(E.axis_angle_mat(side * math.radians(30)) @ app)
    out.append(E.axis_angle_mat(np.array([0.0, 0.0, 1.0]) * math.radians(30)) @ app)
    out.append(E.axis_angle_mat(np.array([0.0, 0.0, 1.0]) * math.radians(-30)) @ app)
    return out


def one_target(env, m, lo, hi, label, point, app):
    if True:
        row = {"target": label, "point": np.round(point, 3).tolist(), "approach": np.round(app, 3).tolist()}
        for name, pt in (("pre", point - PRE * app), ("at", point)):
            best = None
            for k in range(RESTARTS):
                dd = mujoco.MjData(m)
                dd.qpos[:] = env.data.qpos
                if k:
                    dd.qpos[env.jq] = RNG_STARTS[k - 1] * (hi - lo) + lo
                pe, ae = ik(m, dd, env.site, env.jq, env.jv, lo, hi, pt, app)
                if best is None or pe + 0.05 * ae < best[0] + 0.05 * best[1]:
                    best = (pe, ae)
                if pe < POS_TOL and ae < ANG_TOL:
                    break
            pe, ae = best
            row[name] = {"pos_err": round(pe, 4), "ang_err_deg": round(math.degrees(ae), 1),
                         "ok": bool(pe < POS_TOL and ae < ANG_TOL)}
        row["ok"] = row["pre"]["ok"] and row["at"]["ok"]
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--front-x", type=float, default=None, help="override every kind's front_x")
    ap.add_argument("--robot-z", type=float, default=None, help="override every kind's robot_z")
    a = ap.parse_args()
    mount = {}
    if a.front_x is not None:
        mount["front_x"] = a.front_x
    if a.robot_z is not None:
        mount["robot_z"] = a.robot_z
    assets = os.path.expanduser(a.assets)
    models = a.models or [k + "/" + n for k in KINDS for n in sorted(os.listdir(os.path.join(assets, k)))]
    result = {}
    for model in models:
        env = E.ApplianceEnv(E.unit_spec(model), assets, render=False, mount=mount or None,
                             jitter_scale=(0, 0, 0))
        try:
            env.reset(0)
            rows = study(env)
        except Exception as exc:
            result[model] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
            print(model, "ERROR", result[model]["error"], flush=True)
            continue
        ok = sum(r["ok"] for r in rows)
        result[model] = {"targets": rows, "ok": ok, "n": len(rows),
                         "unit_size": np.round(env.layout["bbox_hi"] - env.layout["bbox_lo"], 3).tolist()}
        bad = [r["target"] for r in rows if not r["ok"]]
        print("%-24s %2d/%2d reachable  %s" % (model, ok, len(rows), " ".join(bad)), flush=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
