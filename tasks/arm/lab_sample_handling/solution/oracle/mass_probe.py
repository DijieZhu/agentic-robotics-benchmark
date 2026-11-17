"""Lab sample handling, feasibility probe 4b: how well can a tube's mass be measured, against a balance tolerance?

A balanced centrifuge load pairs tubes of matching mass (tube + contents; 1 mL of a watery sample is about 1 g).
Two ways an agent could measure it here, each with an estimator of the kind an agent could write:

- `--mode fill`: read the liquid column in a close camera image. A tube with a red liquid column (centrifuge
  probe's render mode) stands in an open rack; a camera with the wrist camera's intrinsics (75 deg, 256 x 192)
  looks at it from 0.12-0.18 m, from the robot's side, with the environment's pixel noise. The estimator finds the
  column's top edge and the cap's bottom edge in the image, intersects their pixel rays with the tube's axis
  (its position known to `--axis-err` from the rack's layout) and turns the gap into millilitres. The camera is a
  free camera with the wrist camera's intrinsics placed where the hand could put it; whether the arm reaches
  each such pose is not tested here.
- `--mode weigh`: the wrist force/torque sensor. The arm hovers, records the force, grasps a tube, lifts it and
  holds still, records again; the difference along gravity is the tube's weight. Reported noise-free (the
  controller's own bias) and with the environment's sensor noise (box transport's WRENCH_NOISE per reading).

    python -I mass_probe.py --mode fill --grid '{"seed": [0, 1, 2]}' --out ../outputs/lab/fill.jsonl
    python -I mass_probe.py --mode weigh --fill 8
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cv2
import mujoco
import numpy as np

import centrifuge_probe as C
import tube_insert_probe as TP
from localize_probe import cam_pose, intrinsics, render
from tube_insert_probe import B

ML = 1e-6                       # m^3
L_BODY_OFF = 0.012              # the cap's height (tube_insert_probe.add_tube: the body is tube_len - 0.012)
LIQ_BASE = 0.002                # the column starts this far above the tip (centrifuge_probe.build)


def defaults(a):
    for k, v in {"rotor_model": "solid", "no_lid": False, "lid_side": "right",
                 "lid_open": C.LID_OPEN, "lid_start": C.LID_OPEN, "lid_mass": 0.4, "lid_damping": 0.3,
                 "lid_friction": 0.5, "rotor_damping": 0.05, "rotor_friction": 0.3, "rotor_jitter": 0.0,
                 "rotor_start": 0.0, "clearance": 0.001, "tube_d": 0.017, "tube_len": 0.120, "mass": 0.015,
                 "com_z": 0.035, "friction": 0.6, "yaw_choice": 0, "yaw_plan": None, "held_depth": 0.035}.items():
        if not hasattr(a, k):
            setattr(a, k, v)


def column_rows(img, col_px, half_w):
    """Image rows of the liquid's top edge and the cap's bottom edge in a band of columns around the tube's axis:
    red (the sample) and blue (the cap) pixels, the topmost red row and the lowest blue row."""
    w = img.shape[1]
    x0, x1 = int(max(0, col_px - half_w)), int(min(w, col_px + half_w + 1))
    hsv = cv2.cvtColor(img[:, x0:x1], cv2.COLOR_RGB2HSV)
    red = cv2.inRange(hsv, (0, 90, 50), (8, 255, 255)) | cv2.inRange(hsv, (170, 90, 50), (180, 255, 255))
    blue = cv2.inRange(hsv, (100, 90, 40), (130, 255, 255))
    min_px = max(2, (x1 - x0) // 3)            # a row counts when a third of the band is that colour
    red_rows = np.nonzero((red > 0).sum(1) >= min_px)[0]
    blue_rows = np.nonzero((blue > 0).sum(1) >= min_px)[0]
    if len(red_rows) == 0 or len(blue_rows) == 0:
        return None
    # the cap sits above the column: take the lowest blue row above the topmost red row
    top_red = int(red_rows.min())
    above = blue_rows[blue_rows < top_red]
    if len(above) == 0:
        return None
    return top_red - 0.5, int(above.max()) + 0.5   # pixel edges: the top of a row, the bottom of a row


def ray_axis_z(K, R, p, u, v, axis_xy):
    """Height where the pixel's ray passes closest to the vertical line through axis_xy."""
    ray = R @ np.linalg.solve(K, np.array([u, v, 1.0]))
    q = np.array([axis_xy[0], axis_xy[1], 0.0])
    # closest points between p + s ray and q + t z
    z = np.array([0.0, 0.0, 1.0])
    w0 = p - q
    a_, b_, c_ = ray @ ray, ray @ z, 1.0
    d_, e_ = ray @ w0, z @ w0
    den = a_ * c_ - b_ * b_
    t = (a_ * e_ - b_ * d_) / den
    return float(t)


def fill_trial(a):
    t0 = time.time()
    defaults(a)
    rng = np.random.default_rng([a.seed, 11])
    fills = [float(x) for x in rng.uniform(a.fill_min, a.fill_max, 2)] + [5.0]
    a.tubes = 3
    C.SRC_XY[2:2] = [np.array([0.36, 0.38])]              # the third tube out of the views of the first two

    def extra(wb):
        cam = wb.add_camera()
        cam.name = "probe"
        cam.fovy = B.WRIST["fovy"]

    m, _src, names = C.build(a, fills=fills, extra=extra)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cid = m.camera("probe").id
    ww, wh = B.WRIST["width"], B.WRIST["height"]
    K = intrinsics(B.WRIST["fovy"], ww, wh)
    rows = []
    for i in range(2):
        tip = d.xpos[m.body(names[i]).id].copy()
        mid = tip + np.array([0, 0, 0.06])
        for v in range(a.views):
            dist = rng.uniform(0.12, 0.18)
            az = math.radians(180 + rng.uniform(-30, 30))            # from the robot's side (-x)
            el = math.radians(rng.uniform(-10, 20))
            look = mid + rng.uniform(-0.01, 0.01, 3)
            pos = look + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            m.cam_pos[cid] = pos
            m.cam_quat[cid] = B.quat_from_mat(B.look_at(pos, look))
            mujoco.mj_forward(m, d)
            img = render(m, d, "probe", ww, wh, rng)
            R, pc = cam_pose(d, cid)
            # the tube's axis as the estimator knows it: the rack position with an error
            axis_xy = tip[:2] + rng.uniform(-a.axis_err, a.axis_err, 2)
            u_axis = (K @ (R.T @ (np.array([axis_xy[0], axis_xy[1], mid[2]]) - pc)))
            col = u_axis[0] / u_axis[2]
            row = {"seed": a.seed, "tube": i, "view": v, "fill_ml": round(fills[i], 3), "dist_m": round(dist, 3),
                   "el_deg": round(math.degrees(el), 1)}
            got = column_rows(img, col, a.band)
            if got is None:
                row["found"] = False
            else:
                v_liq, v_cap = got
                z_liq = ray_axis_z(K, R, pc, col, v_liq, axis_xy)
                z_cap = ray_axis_z(K, R, pc, col, v_cap, axis_xy)
                # the cap's bottom is at tip + tube_len - 0.012, the column's top at tip + 0.002 + h
                h = (z_liq - z_cap) + (a.tube_len - L_BODY_OFF) - LIQ_BASE
                r_in = a.tube_d / 2 - 0.0012
                est = h * math.pi * r_in ** 2 / ML
                row.update(found=True, est_ml=round(est, 3), err_ml=round(est - fills[i], 3))
            if a.save_dir and a.seed == 0:
                os.makedirs(a.save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(a.save_dir, f"fill_t{i}_v{v}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            rows.append(row)
    for r in rows:
        r["wall_s"] = round(time.time() - t0, 1)
    return rows


def weigh_trial(a):
    """Hover, record the force; grasp the tube, lift, hold, record; the difference along gravity."""
    t0 = time.time()
    defaults(a)
    a.tubes = 2
    m, _src, names = C.build(a)
    tube_mass = a.mass + a.fill * 1e-3               # 1 mL of a watery sample = 1 g
    TP.set_tube_mass(m, names[0], a, mass=tube_mass)
    mujoco.mj_setConst(m, mujoco.MjData(m))
    p = C.World(m, a)
    rng = np.random.default_rng([a.seed, 13])
    p.use(names[0])
    for _ in range(10):
        p.step([0, 0, 0], [0, 0, 0], -1)
    top = p.tube_pos() + p.tube_R() @ np.array([0, 0, a.tube_len])
    gp = top - np.array([0, 0, TP.GRASP_BELOW_TOP])
    R_down = p.down(math.pi / 2)
    hold = gp + [0, 0, 0.12]

    def record(n):
        fz = []
        for _ in range(n):
            p.step(hold - p.ee(), C.rotvec(R_down @ p.ee_R().T), p.grip_now)
            fz.append(float(p.wrench_force()[2]))
        return np.array(fz)

    p.grip_now = -1
    p.servo(hold, R_down, -1, 0.02, tol=0.002, max_steps=200)
    for _ in range(20):
        p.step(hold - p.ee(), C.rotvec(R_down @ p.ee_R().T), -1)
    empty = record(a.hold_steps)
    p.servo(gp, R_down, -1, 0.01)
    for _ in range(8):
        p.step([0, 0, 0], [0, 0, 0], 1)
    ok = p.grasped()
    p.grip_now = 1
    p.servo(hold, R_down, 1, 0.01, tol=0.002, max_steps=300)
    for _ in range(20):
        p.step(hold - p.ee(), C.rotvec(R_down @ p.ee_R().T), 1)
    held = record(a.hold_steps)
    true_g = tube_mass * 1000
    # the sensor reads the load on the hand: a held mass adds m g along world z (this sensor frame's sign)
    est_clean = (held.mean() - empty.mean()) / TP.G * 1000
    noisy_e = empty + rng.normal(0, B.WRENCH_NOISE, empty.shape)
    noisy_h = held + rng.normal(0, B.WRENCH_NOISE, held.shape)
    est_noisy = (noisy_h.mean() - noisy_e.mean()) / TP.G * 1000
    return {"seed": a.seed, "fill_ml": a.fill, "grasped": bool(ok), "hold_steps": a.hold_steps, "true_g": round(true_g, 2),
            "est_clean_g": round(est_clean, 2), "err_clean_g": round(est_clean - true_g, 2),
            "est_noisy_g": round(est_noisy, 2), "err_noisy_g": round(est_noisy - true_g, 2),
            "clean_sd_N": round(float(held.std()), 4), "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("fill", "weigh"), default="fill")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fill-min", type=float, default=3.0, help="mL (lower columns hide in the rack's collar)")
    ap.add_argument("--fill-max", type=float, default=13.0)
    ap.add_argument("--views", type=int, default=3, help="camera poses per tube")
    ap.add_argument("--axis-err", type=float, default=0.002, help="m, the tube axis' position error (each axis)")
    ap.add_argument("--band", type=int, default=2, help="px, half width of the column band read")
    ap.add_argument("--fill", type=float, default=8.0, help="mL, weigh mode")
    ap.add_argument("--hold-steps", type=int, default=50, help="control steps (10 Hz) per force record")
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--grid")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    fn = fill_trial if a.mode == "fill" else weigh_trial
    if not a.grid:
        res = fn(a)
        for r in res if isinstance(res, list) else [res]:
            print(json.dumps(r))
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
        for res in ex.map(fn, jobs):
            f.writelines(json.dumps(r) + "\n" for r in (res if isinstance(res, list) else [res]))
    print(f"{len(jobs)} trials -> {a.out}")


if __name__ == "__main__":
    main()
