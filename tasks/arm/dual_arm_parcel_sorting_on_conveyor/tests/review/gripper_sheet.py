"""The gripper's data sheet, measured in the cell (a maintainer's tool; the grader never runs it).

    python3 -B tests/review/gripper_sheet.py --out DIR [--write-measured]

On the cell's own world (the grader's configuration, the robot at its start posture, the left gripper):

* closing and opening times on nothing: grip 0 -> 1 and 1 -> 0, until the opening (the state's ``opening_m``: the
  distance between the inner faces of the pads) is within 0.5 mm of where it settles;
* the openings fully open (at rest, and at the shift's start) and closed on nothing, and the opening over the
  command (steady state, 0 .. 1);
* the squeeze on a carton block held at command 1 (50 mm across the pads): the summed normal force of the robot's
  contacts on it (the quantity of the damage rule) and the drive's ``force`` reading;
* the delay of the ``object_detected`` status after the fingers close on the block (the cell's ``GripperStatus``);
* the heaviest block such a pinch holds against gravity for 3 s (the block is closed on with gravity off, then
  gravity is switched on; a construction, physics otherwise the cell's).

``--write-measured`` writes the block "gripper" of ``docs/measured.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import common  # noqa: E402

MEASURED = os.path.join(HERE, "docs", "measured.json")
BLOCK_W = 0.05


def _world(cfg, spare=1):
    from parcelcell.config import cell_config
    from parcelcell.stream import Shift
    from parcelcell.world import ShiftWorld
    return ShiftWorld(Shift(belt_speed=((0.0, 0.0),), arrivals=(), duration=600.0, seed=0), cell_config(cfg),
                      spare_slots=spare)


def opening(world, side="Left"):
    """The state's opening_m (server._openings): distance between the pads' inner faces."""
    import mujoco
    import numpy as np
    m, d = world.model, world.data
    p = side[0]
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{p}_base")
    R = d.xmat[bid].reshape(3, 3)
    gaps = []
    for k in ("1", "2"):
        a, b = world._gid(f"{p}_left_pad{k}"), world._gid(f"{p}_right_pad{k}")
        gaps.append(abs(float((R.T @ (d.geom_xpos[a] - d.geom_xpos[b]))[1])) - m.geom_size[a][1] - m.geom_size[b][1])
    return max(0.0, min(gaps))


def settle_time(series, tol=0.0005):
    final = series[-1][1]
    for t, v in series:
        if all(abs(x - final) <= tol for _, x in series if _ >= t):
            return t, final
    return series[-1][0], final


def close_open(cfg):
    w = _world(cfg, 0)
    q = w.home.copy()
    start = round(opening(w), 4)                 # what the state shows at t = 0
    for _ in range(25):
        w.step(q, (0.0, 0.0))
    ser = []
    t0 = w.time
    for _ in range(150):
        w.step(q, (1.0, 0.0))
        ser.append((round(w.time - t0, 3), opening(w)))
    tc, closed = settle_time(ser)
    ser2 = []
    t0 = w.time
    for _ in range(150):
        w.step(q, (0.0, 0.0))
        ser2.append((round(w.time - t0, 3), opening(w)))
    to, opened = settle_time(ser2)
    curve = []
    for c in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        for _ in range(60):
            w.step(q, (c, 0.0))
        curve.append((c, round(opening(w), 4)))
    w.close()
    return {"close_s": tc, "open_s": to, "opening_closed_m": round(closed, 4), "opening_open_m": round(opened, 4),
            "opening_start_m": start, "opening_over_command": curve}


def hold(cfg, mass, hold_s=3.0):
    import mujoco
    import numpy as np
    from parcelcell.stream import Parcel
    w = _world(cfg, 1)
    m, d = w.model, w.data
    q = w.home.copy()
    for _ in range(25):
        w.step(q, (0.0, 0.0))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "L_base")
    R = d.xmat[bid].reshape(3, 3)
    pos, _ = w.tcp_pose("Left")
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.reshape(9))
    p = Parcel(kind="rigid", size=(0.04, BLOCK_W, 0.04), mass=mass, friction=0.55, label_face=0)
    g0 = m.opt.gravity.copy()
    m.opt.gravity[:] = 0.0
    i = w.inject(p, pos, quat)
    from parcelcell.server import GripperStatus
    gs = GripperStatus(cfg["world"]["gripper_object"])
    stop_t = det_t = None
    prev = None
    for _ in range(60):
        w.step(q, (1.0, 0.0))
        op = [opening(w, "Left"), opening(w, "Right")]
        det = gs.update(op, [1.0, 0.0])
        if prev is not None and stop_t is None and abs(op[0] - prev) < 0.0002:
            stop_t = w.time
        prev = op[0]
        if det[0] and det_t is None:
            det_t = w.time
    m.opt.gravity[:] = g0
    sq, fr = [], []
    start = d.qpos[w._slot_qadr[i]:w._slot_qadr[i] + 3].copy()
    t_end = w.time + hold_s
    while w.time < t_end - 1e-9:
        w.step(q, (1.0, 0.0))
        cs = [c for c in w.truth.contacts(kinds=("robot-parcel",)) if c.slot == i]
        sq.append(sum(c.force_normal for c in cs))
        fr.append(float(w.gripper_state().force[0]))
    end = d.qpos[w._slot_qadr[i]:w._slot_qadr[i] + 3].copy()
    tcp, _ = w.tcp_pose("Left")
    held = bool(w.slots[i].active and np.linalg.norm(end - tcp) < 0.01 and np.linalg.norm(end - start) < 0.01)
    n = len(sq) // 2
    out = {"mass": mass, "held": held, "squeeze_N": round(float(np.mean(sq[n:])), 1),
           "squeeze_peak_N": round(float(np.max(sq)), 1), "force_reading": round(float(np.mean(fr[n:])), 3),
           "drift_m": round(float(np.linalg.norm(end - start)), 4),
           "detect_ms": None if (det_t is None or stop_t is None) else round((det_t - stop_t) * 1000.0)}
    w.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-measured", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cfg = common.load_cfg()
    co = close_open(cfg)
    print("close / open:", co)
    trials = []
    # from the lightest block up to the population's heaviest parcel (the configuration's mass range), in 0.3 kg steps
    m_hi = float(cfg["shifts"]["mass_kg"][1])
    masses = [round(0.2 + 0.3 * i, 3) for i in range(int((m_hi - 0.2) / 0.3 + 1e-9) + 1)]
    if masses[-1] < m_hi - 1e-9:
        masses.append(m_hi)
    for mass in masses:
        r = hold(cfg, mass)
        trials.append(r)
        print("hold", r, flush=True)
    held = [t["mass"] for t in trials if t["held"]]
    first = trials[0]
    sheet = dict(co, block_mm=BLOCK_W * 1000, pad_force_N=first["squeeze_N"], force_reading=first["force_reading"],
                 detect_ms=first["detect_ms"],
                 hold_kg=max(held) if held else 0.0, hold_trials=trials,
                 tool="tests/review/gripper_sheet.py")
    common.dump_json(os.path.join(a.out, "gripper_sheet.json"), sheet)
    if a.write_measured:
        mm = json.load(open(MEASURED)) if os.path.exists(MEASURED) else {}
        mm["gripper"] = {k: sheet[k] for k in ("close_s", "open_s", "opening_open_m", "opening_start_m",
                                               "opening_closed_m", "block_mm",
                                               "pad_force_N", "detect_ms", "hold_kg", "opening_over_command",
                                               "tool")}
        with open(MEASURED, "w") as fh:
            json.dump(mm, fh, indent=1)
            fh.write("\n")
        print("wrote", os.path.relpath(MEASURED, common.TASK))
    return 0


if __name__ == "__main__":
    sys.exit(main())
