"""Lab sample handling, feasibility probe 1: can a Panda insert a sample tube into a rack or rotor hole in
this simulation, and does a placement error make the insertion fail?

Question: the task's manipulation core is moving capped 15 mL conical tubes between holes (a rack, a centrifuge
rotor whose holes may be tilted). That is a real skill only if a privileged controller inserts reliably at a realistic clearance while an error of a few
millimetres or degrees jams the tube or misses the hole.

World: box transport's table, Panda, gripper and operational-space controller (`box_transport_env.py`),
with the stiff contact model (condim 4, torsional friction, no-slip). A tube built from primitives (a capsule body, a cap cylinder) stands in a source hole; a target hole,
upright or tilted, stands beside it. Holes are rings of box segments with a floor, built from primitives.

A trial (privileged: it reads the tube's pose): grasp the tube below its cap, lift it out, bring it above the
target hole along the hole's axis with an injected error (position across the axis, angle), push it in along
the axis, release, retreat. Reported: inserted (depth reached, inside, aligned), jammed (the push stalled or the
wrist force passed a limit), the largest wrist force while pushing, the final depth, radial offset and angle.

    python -I tube_insert_probe.py --clearance 0.001 --tilt 45 --pos-err 0.002 --ang-err 2
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "box_env"))
import box_transport_env as B
import mujoco
import numpy as np

from lab_sim import (  # noqa: F401
    NSEG,
    SOURCE_CLEARANCE,
    TABLE_Z,
    WALL,
    Probe,
    add_ring,
    add_tube,
    base_world,
    rot_y,
    rotvec,
    set_tube_mass,
    stiff_pads,
)

SOURCE_XY = np.array([0.50, 0.16])
TARGET_XY = np.array([0.50, -0.12])
SOURCE_DEPTH = 0.050
GRASP_BELOW_TOP = 0.024      # m, pad centre below the tube's top (below the cap)
LIFT_CLEAR = 0.04            # m, tube bottom above the hole's entrance before the push
FORCE_LIMIT = 25.0           # N, wrist force (beyond the tube's weight) that ends the push as a jam
G = 9.81


def build(a):
    w, wb = base_world()
    r_t = a.tube_d / 2
    # holes: walls and floor are world geoms in a frame at the hole's entrance
    holes = {}
    for name, xy, clearance, depth, tilt in (("src", SOURCE_XY, SOURCE_CLEARANCE, SOURCE_DEPTH, 0.0),
                                             ("tgt", TARGET_XY, a.clearance, a.depth, math.radians(a.tilt))):
        R = rot_y(tilt)                          # the axis leans away from the robot (+x) by `tilt`
        top = np.array([xy[0], xy[1], TABLE_Z + depth * math.cos(tilt) + 0.015 + (WALL + r_t) * math.sin(tilt)])
        holes[name] = add_ring(wb, name, top, R, r_t + clearance, depth)
    # the tube starts seated in the source hole
    src = holes["src"]
    add_tube(wb, "tube", src["top"] - src["axis"] * (src["depth"] - 0.0005), a)
    m = w.compile()
    set_tube_mass(m, "tube", a)
    stiff_pads(m)
    mujoco.mj_setConst(m, mujoco.MjData(m))
    return m, holes


def trial(a):
    t0 = time.time()
    m, holes = build(a)
    p = Probe(m)
    rng = np.random.default_rng(a.seed)
    out = {"clearance": a.clearance, "tilt": a.tilt, "pos_err": a.pos_err, "ang_err": a.ang_err, "mass": a.mass,
           "friction": a.friction, "depth": a.depth, "speed": a.speed, "seed": a.seed}
    try:
        for _ in range(10):
            p.step([0, 0, 0], [0, 0, 0], -1)
        top = p.tube_pos() + p.tube_R() @ np.array([0, 0, a.tube_len])
        gp = top - np.array([0, 0, GRASP_BELOW_TOP])
        R_down = p.down(math.pi / 2)            # fingers close along world y: the tilt is about world y
        if not (p.servo(gp + [0, 0, 0.10], R_down, -1, 0.025, max_steps=200)
                and p.servo(gp, R_down, -1, 0.01)):
            raise RuntimeError("approach")
        for _ in range(8):
            p.step([0, 0, 0], [0, 0, 0], 1)
        if not p.grasped():
            raise RuntimeError("no grasp")
        src = holes["src"]
        p.servo(p.ee() + [0, 0, src["depth"] + 0.06], R_down, 1, 0.01, max_steps=200)
        # the tube in the gripper frame, as grasped (privileged)
        Re, pe = p.ee_R(), p.ee()
        G = Re.T @ p.tube_R()
        x = Re.T @ (p.tube_pos() - pe)
        tgt = holes["tgt"]
        ax = tgt["axis"]
        # injected errors: a position offset across the axis, a tilt about a random axis across it
        u = np.cross(ax, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(ax, u)
        phi = rng.uniform(0, 2 * math.pi)
        off = a.pos_err * (math.cos(phi) * u + math.sin(phi) * v)
        psi = rng.uniform(0, 2 * math.pi)
        tilt_axis = math.cos(psi) * u + math.sin(psi) * v
        R_err = B.axis_angle_mat(math.radians(a.ang_err) * tilt_axis)
        # desired tube rotation: its axis along the (erroneous) hole axis, keeping its yaw
        R_td = R_err @ tgt["R"]                    # a round tube: its yaw about the axis does not matter
        R_ee_des = R_td @ G.T
        ax_err = R_err @ ax

        def ee_for(bottom):
            return bottom - R_ee_des @ x

        pre = tgt["top"] + off + ax_err * LIFT_CLEAR
        # travel high, then down to the pre-insertion pose
        hi = ee_for(pre) + [0, 0, 0.06]
        if not p.servo(hi, R_ee_des, 1, 0.02, tol=0.005, max_steps=300):
            raise RuntimeError("transfer")
        if not p.servo(ee_for(pre), R_ee_des, 1, 0.006, tol=0.0015, max_steps=200):
            raise RuntimeError("pre-insert")
        # push along the (erroneous) axis to the floor
        goal = tgt["top"] + off - ax_err * (tgt["depth"] - 0.001)
        p.force_ref = p.wrench_force()
        p.track_force = True
        stalled, jammed, last = 0, False, None
        cmd = ee_for(pre)
        travel = float(np.dot(pre - goal, ax_err))
        n_steps = math.ceil(travel / a.speed)
        for k in range(n_steps + 20):
            s = min(1.0, (k + 1) / n_steps)
            cmd = ee_for(pre + (goal - pre) * s)
            e = cmd - p.ee()
            p.integral = np.clip(p.integral + 0.15 * e, -0.02, 0.02)
            p.step(e + p.integral, np.clip(rotvec(R_ee_des @ p.ee_R().T), -0.15, 0.15), 1)
            f = float(np.linalg.norm(p.wrench_force() - p.force_ref))
            if f > FORCE_LIMIT:
                jammed = True
                break
            prog = float(np.dot(tgt["top"] - (p.tube_pos()), ax))
            if last is not None and s >= 1.0 and abs(prog - last) < 1e-4:
                stalled += 1
                if stalled > 5:
                    break
            last = prog
        p.track_force = False
        out["max_force_N"] = round(p.max_force, 2)
        out["jammed"] = jammed
        for _ in range(8):
            p.step([0, 0, 0], [0, 0, 0], -1)
        p.servo(p.ee() - p.ee_R()[:, 2] * 0.10, p.ee_R(), -1, 0.01, max_steps=60)
        for _ in range(20):
            p.step([0, 0, 0], [0, 0, 0], -1)
        # where the tube ended up, against the true hole
        bottom = p.tube_pos()
        rel = bottom - tgt["top"]
        depth_in = float(-np.dot(rel, ax))
        radial = float(np.linalg.norm(rel + depth_in * ax))
        tube_ax = p.tube_R()[:, 2]
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(tube_ax, ax))))))
        out.update(depth_in=round(depth_in, 4), radial=round(radial, 4), angle_deg=round(ang, 2))
        out["inserted"] = bool(depth_in > tgt["depth"] - 0.006 and radial < tgt["radius"] and ang < 5.0)
        out["status"] = "ok"
    except FloatingPointError:
        out["status"] = "unstable"
    except RuntimeError as exc:
        out["status"] = str(exc)
    out["steps"] = p.t
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clearance", type=float, default=0.001, help="m, radial, target hole")
    ap.add_argument("--depth", type=float, default=0.06, help="m, target hole depth")
    ap.add_argument("--tilt", type=float, default=0.0, help="deg, target hole axis from vertical")
    ap.add_argument("--pos-err", type=float, default=0.0, help="m, placement error across the axis")
    ap.add_argument("--ang-err", type=float, default=0.0, help="deg, axis error")
    ap.add_argument("--speed", type=float, default=0.003, help="m per control step, push")
    ap.add_argument("--tube-d", type=float, default=0.017, help="m, 15 mL conical tube")
    ap.add_argument("--tube-len", type=float, default=0.120)
    ap.add_argument("--mass", type=float, default=0.015, help="kg, tube + contents")
    ap.add_argument("--com-z", type=float, default=0.035, help="m, centre of mass above the tip")
    ap.add_argument("--friction", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", help="JSON {arg: [values]}: every combination in parallel, one line each to --out")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=48)
    a = ap.parse_args()
    if not a.grid:
        print(json.dumps(trial(a)))
        return
    import itertools
    from concurrent.futures import ProcessPoolExecutor
    grid = json.loads(a.grid)
    keys = list(grid)
    jobs = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        b = argparse.Namespace(**vars(a))
        for k, val in zip(keys, combo):
            setattr(b, k.replace("-", "_"), val)
        jobs.append(b)
    with ProcessPoolExecutor(a.jobs) as ex, open(a.out, "a") as f:
        f.writelines(json.dumps(row) + "\n" for row in ex.map(trial, jobs))
    print(f"{len(jobs)} trials -> {a.out}")


if __name__ == "__main__":
    main()
