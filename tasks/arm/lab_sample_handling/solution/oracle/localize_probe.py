"""Lab sample handling, feasibility probe 3: how well do the cameras locate a hole, against the insertion's
capture range of about 4 mm (probe 1)?

World: centrifuge_probe's (the Panda, the two fixed cameras and the wrist camera of box transport, the
centrifuge with its 45 deg rotor, the source rack) plus an empty 3 x 3 rack. Rack and rotor rings are coloured
(orange), as lab racks usually are. Each seed shifts the empty rack (+-20 mm) and turns the rotor (+-10 deg).

A non-privileged estimator, of the kind an agent could write, per target hole:
1. Fixed stereo: in the left and right images (320 x 240, with the environment's pixel noise), find the ring's
   inner contour (the opening) near a prior position (the true projection offset by up to `--prior-px`
   pixels, standing for the layout a task card gives), fit an ellipse, take its centre in both images and
   triangulate with the cameras' published intrinsics and poses.
2. Wrist: put the wrist camera `--wrist-dist` above a coarse prior (the truth offset by up to `--prior-mm` across
   the hole's axis and 5 mm along it), its optical axis straight down (the site pose follows from the camera's
   mount); find the opening in the wrist image (256 x 192), gated by its expected size, and meet the ray through its
   centre with the entrance plane through the prior. Then move the camera `--wrist-baseline` sideways, find the
   opening again and triangulate the two rays (no entrance plane needed).
On the solid rotor (`--rotor-model solid`, the default; probe 4) rotor openings are dark holes in a light surface;
`--rotor-model rings` is probe 3's rotor of separate rings, recoloured orange.
Errors are against the true entrance centre: the full 3-D error and the error across the hole's axis (the one
the capture range is about).

    python -I localize_probe.py --seed 0
    python -I localize_probe.py --grid '{"seed": [0, 1, 2]}' --out ../outputs/lab/localize.jsonl
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cv2
import numpy as np

import centrifuge_probe as C
import tube_insert_probe as TP
from lab_sim import RING_RGBA, cam_pose, intrinsics, project, render
from tube_insert_probe import B

RACK_XY = np.array([0.64, 0.26])
RACK_PITCH = 0.045
ROTOR_TARGETS = tuple(range(C.NHOLES))


def rack_holes(seed, a):
    rng = np.random.default_rng([seed, 3])
    shift = rng.uniform(-0.02, 0.02, 2)
    out = {}
    for i in range(3):
        for j in range(3):
            xy = RACK_XY + shift + RACK_PITCH * np.array([i - 1, j - 1])
            top = np.array([xy[0], xy[1], B.TABLE_Z + 0.03 + 0.015])
            out[f"r{i}{j}"] = {"top": top, "R": np.eye(3), "axis": np.array([0, 0, 1.0])}
    return out


def build(a):
    holes = rack_holes(a.seed, a)

    def extra(wb):
        for name, h in holes.items():
            TP.add_ring(wb, name, h["top"], h["R"], a.tube_d / 2 + a.clearance, 0.03, rgba=RING_RGBA)

    m, _src, _names = C.build(a, extra=extra)
    rings = ("r",) if a.rotor_model == "solid" else ("h", "r")   # the solid rotor's bores stay dark
    for g in range(m.ngeom):
        nm = m.geom(g).name
        if nm.startswith(rings) and "_w" in nm:           # rotor rings h<k>_w*, rack rings r<ij>_w*
            m.geom_rgba[g] = RING_RGBA
    m.site_rgba[:, 3] = 0.0                                # sites are not part of the scene
    return m, holes


def find_opening(img, prior, win, size_px=None):
    """Centre (px) of the ring opening nearest `prior`: the ring's orange pixels, the holes in their contours,
    an ellipse fitted to the chosen hole's contour (its centroid when it is too small to fit)."""
    h, w = img.shape[:2]
    x0, y0 = int(max(0, prior[0] - win)), int(max(0, prior[1] - win))
    x1, y1 = int(min(w, prior[0] + win)), int(min(h, prior[1] + win))
    crop = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (5, 100, 60), (35, 255, 255))     # orange to its yellow highlights
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return None
    best, best_d = None, 1e9
    for c, hh in zip(contours, hier[0]):
        if hh[3] < 0:                       # an outer contour; the openings are the inner ones
            continue
        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) >= 5:
            (cx, cy), axes, _ang = cv2.fitEllipse(c)
            if size_px is not None and not (0.6 * size_px <= max(axes) <= 1.5 * size_px
                                             and min(axes) >= 0.5 * max(axes)):
                continue                    # not an opening of the expected size (a gap between rings)
        else:
            if size_px is not None:
                continue
            cx, cy = pts.mean(0)
        ctr = np.array([cx + x0, cy + y0])
        dd = float(np.linalg.norm(ctr - prior))
        if dd < best_d:
            best, best_d = ctr, dd
    if best is None and size_px is not None:
        best = find_dark_opening(crop, mask, prior - [x0, y0], size_px)
        if best is not None:
            best = best + [x0, y0]
    return best


def find_dark_opening(crop, ring_mask, prior, size_px):
    """Fallback for an opening the ring does not enclose in the image (a tilted hole seen from above): the dark
    interior next to ring pixels, an ellipse fitted to it, the one nearest the prior."""
    v = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)[:, :, 2]
    dark = (v < 70).astype(np.uint8) * 255
    near_ring = cv2.dilate(ring_mask, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(dark)
    best, best_d = None, 1e9
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 6 or not near_ring[ys, xs].any():
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (cx, cy), axes, _ang = cv2.fitEllipse(pts)
        if not (0.4 * size_px <= max(axes) <= 1.5 * size_px):
            continue
        dd = float(np.hypot(cx - prior[0], cy - prior[1]))
        if dd < best_d:
            best, best_d = np.array([cx, cy]), dd
    return best


def find_bore(img, prior, win, size_px):
    """Centre (px) of the dark opening nearest `prior` in a light surface (the solid rotor's bored holes): dark
    pixels next to light ones, an ellipse fitted to each blob, gated by the opening's expected size."""
    h, w = img.shape[:2]
    x0, y0 = int(max(0, prior[0] - win)), int(max(0, prior[1] - win))
    x1, y1 = int(min(w, prior[0] + win)), int(min(h, prior[1] + win))
    crop = img[y0:y1, x0:x1]
    v = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)[:, :, 2]
    light = (v > 120).astype(np.uint8) * 255
    best = find_dark_opening(crop, light, prior - [x0, y0], size_px)
    return None if best is None else best + [x0, y0]


def triangulate(Ks, Rs, ps, uvs):
    """Least-squares point closest to the rays through the pixel centres."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for K, R, p, uv in zip(Ks, Rs, ps, uvs):
        ray = R @ np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
        ray /= np.linalg.norm(ray)
        P = np.eye(3) - np.outer(ray, ray)
        A += P
        b += P @ p
    return np.linalg.solve(A, b)


def across(err, axis):
    return float(np.linalg.norm(err - np.dot(err, axis) * axis))


def trial(a):
    t0 = time.time()
    a.rotor_jitter = 10.0
    m, rack = build(a)
    p = C.World(m, a)
    rng = np.random.default_rng([a.seed, 9])
    for _ in range(10):
        p.step([0, 0, 0], [0, 0, 0], -1)
    d = p.data
    targets = dict(rack)
    for k in ROTOR_TARGETS:
        h = p.hole(k)
        targets[f"h{k}"] = {"top": h["top"], "R": h["R"], "axis": h["axis"]}
    # fixed stereo
    cams = {}
    for cam in ("left", "right"):
        _p, _t, fovy, w, h = B.CAMERAS[cam]
        K = intrinsics(fovy, w, h)
        R, pc = cam_pose(d, m.camera(cam).id)
        cams[cam] = (K, R, pc, render(m, d, cam, w, h, rng))
    rows = []
    wfov, ww, wh = B.WRIST["fovy"], B.WRIST["width"], B.WRIST["height"]
    Kw = intrinsics(wfov, ww, wh)
    wcam = m.camera("r_eye_in_hand").id
    for name, t in targets.items():
        row = {"seed": a.seed, "target": name, "kind": "rotor" if name.startswith("h") else "rack"}
        uvs, Ks, Rs, ps = [], [], [], []
        for cam, (K, R, pc, img) in cams.items():
            truth_px = project(K, R, pc, t["top"])
            prior = truth_px + rng.uniform(-a.prior_px, a.prior_px, 2)
            if row["kind"] == "rotor" and a.rotor_model == "solid":
                dist = float(np.linalg.norm(pc - t["top"]))
                uv = find_bore(img, prior, a.win, (a.tube_d + 2 * a.clearance) * K[0, 0] / dist)
            else:
                uv = find_opening(img, prior, a.win)
            row[f"{cam}_px_err"] = None if uv is None else round(float(np.linalg.norm(uv - truth_px)), 2)
            if uv is not None:
                uvs.append(uv)
                Ks.append(K)
                Rs.append(R)
                ps.append(pc)
        row["stereo_found"] = len(uvs) == 2
        if len(uvs) == 2:
            est = triangulate(Ks, Rs, ps, uvs)
            e = est - t["top"]
            row["stereo_err_mm"] = round(float(np.linalg.norm(e)) * 1000, 2)
            row["stereo_across_mm"] = round(across(e, t["axis"]) * 1000, 2)
        # wrist refinement from a coarse prior (the layout, or the rack's or rotor's pose from the fixed cameras):
        # the truth offset by up to prior_mm across the axis and 5 mm along it; the hand points down the axis,
        # the camera about wrist_dist above the prior
        ax = t["axis"]
        u = np.cross(ax, [1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.cross(ax, [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(ax, u)
        off = rng.uniform(-1, 1, 2) * a.prior_mm / 1000
        est = t["top"] + off[0] * u + off[1] * v + ax * rng.uniform(-0.005, 0.005)
        # the wrist camera straight above the prior, its optical axis pointing down (a tilted opening shows as
        # an ellipse): the site pose follows from the camera's fixed mount on the hand
        Rs_now = d.site_xmat[p.site].reshape(3, 3)
        C_s = Rs_now.T @ d.cam_xmat[wcam].reshape(3, 3)
        o_s = Rs_now.T @ (d.cam_xpos[wcam] - d.site_xpos[p.site])
        # the optical axis: straight down, or down the hole's axis (`--wrist-view axis`: an error of the prior
        # along the axis then barely moves the ray's meeting point with the entrance plane)
        z_c = np.array([0.0, 0.0, -1.0]) if a.wrist_view == "down" else -ax
        xc = d.cam_xmat[wcam].reshape(3, 3)[:, 0]
        x_c = xc - np.dot(xc, z_c) * z_c                # keep the camera's current roll as far as possible
        x_c /= np.linalg.norm(x_c)
        R_cv = np.column_stack([x_c, np.cross(z_c, x_c), z_c])
        R_cam_mj = R_cv @ np.diag([1.0, -1.0, -1.0])
        R_goal = R_cam_mj @ C_s.T
        site_goal = est - z_c * a.wrist_dist - R_goal @ o_s
        reached = p.servo(site_goal, R_goal, -1, 0.025, tol=0.004, max_steps=250)
        for _ in range(5):
            p.step([0, 0, 0], [0, 0, 0], -1)
        Rw, pw = cam_pose(d, wcam)
        img = render(m, d, "r_eye_in_hand", ww, wh, rng)
        prior_w = project(Kw, Rw, pw, est)              # where the stereo estimate projects (not the truth)
        size = (a.tube_d + 2 * a.clearance) * Kw[0, 0] / max(0.05, float(np.dot(pw - est, Rw[:, 2]) * -1))
        inside = 0 <= prior_w[0] < ww and 0 <= prior_w[1] < wh
        if not inside:
            uv = None
        elif row["kind"] == "rotor" and a.rotor_model == "solid":
            uv = find_bore(img, prior_w, a.win_wrist, size)
        else:
            uv = find_opening(img, prior_w, a.win_wrist, size_px=size)
        row["wrist_reached"] = bool(reached)
        row["wrist_dist_m"] = round(float(np.linalg.norm(pw - t["top"])), 3)
        if uv is not None:
            ray = Rw @ np.linalg.solve(Kw, np.array([uv[0], uv[1], 1.0]))
            s = float(np.dot(est - pw, ax) / np.dot(ray, ax))
            west = pw + s * ray
            e2 = west - t["top"]
            row["wrist_err_mm"] = round(float(np.linalg.norm(e2)) * 1000, 2)
            row["wrist_across_mm"] = round(across(e2, ax) * 1000, 2)
            row["wrist_px_err"] = round(float(np.linalg.norm(uv - project(Kw, Rw, pw, t["top"]))), 2)
            # a second wrist view, the camera moved sideways by --wrist-baseline: the two rays triangulated
            # (no assumption about where the entrance plane is)
            shift = Rw[:, 0] * a.wrist_baseline
            reached2 = p.servo(site_goal + shift, R_goal, -1, 0.025, tol=0.004, max_steps=150)
            for _ in range(5):
                p.step([0, 0, 0], [0, 0, 0], -1)
            Rw2, pw2 = cam_pose(d, wcam)
            img2 = render(m, d, "r_eye_in_hand", ww, wh, rng)
            prior2 = project(Kw, Rw2, pw2, west)
            if row["kind"] == "rotor" and a.rotor_model == "solid":
                uv2 = find_bore(img2, prior2, a.win_wrist, size)
            else:
                uv2 = find_opening(img2, prior2, a.win_wrist, size_px=size)
            row["wrist2_reached"] = bool(reached2)
            if uv2 is not None:
                e3 = triangulate([Kw, Kw], [Rw, Rw2], [pw, pw2], [uv, uv2]) - t["top"]
                row["wrist2_err_mm"] = round(float(np.linalg.norm(e3)) * 1000, 2)
                row["wrist2_across_mm"] = round(across(e3, ax) * 1000, 2)
        if a.save_dir and a.seed == 0:
            os.makedirs(a.save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(a.save_dir, f"wrist_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append(row)
    if a.save_dir and a.seed == 0:
        for cam, (_K, _R, _p, img) in cams.items():
            cv2.imwrite(os.path.join(a.save_dir, f"{cam}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    for r in rows:
        r["wall_s"] = round(time.time() - t0, 1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior-px", type=float, default=4.0, help="px, the prior's offset from the truth (each axis)")
    ap.add_argument("--win", type=int, default=14, help="px, half window around the prior (fixed cameras)")
    ap.add_argument("--win-wrist", type=int, default=40, help="px, half window around the prior (wrist)")
    ap.add_argument("--wrist-dist", type=float, default=0.15, help="m, wrist camera above the estimate")
    ap.add_argument("--prior-mm", type=float, default=15.0, help="mm, the wrist prior's error across the axis")
    ap.add_argument("--save-dir", default="../outputs/lab/localize")
    ap.add_argument("--clearance", type=float, default=0.001)
    ap.add_argument("--rotor-model", choices=("rings", "solid"), default="solid")
    ap.add_argument("--wrist-baseline", type=float, default=0.06, help="m, the second wrist view's sideways move")
    ap.add_argument("--wrist-view", choices=("down", "axis"), default="down",
                    help="the wrist camera's optical axis: straight down, or down the hole's axis")
    ap.add_argument("--grid", help="JSON {arg: [values]}")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    # the centrifuge world's defaults
    for k, v in {"mode": "position", "hole": 4, "yaw_choice": 0, "yaw_plan": None, "held_depth": 0.035, "rotor_damping": 0.05,
                     "rotor_friction": 0.3, "rotor_jitter": 10.0, "lid_open": C.LID_OPEN, "lid_start": C.LID_OPEN,
                     "lid_mass": 0.4, "lid_side": "right", "no_lid": False, "lid_damping": 0.3, "lid_friction": 0.5,
                     "lid_steps": 120, "tubes": 2, "rotor_start": 0.0, "turn": 45.0, "second": 2, "tube_d": 0.017, "tube_len": 0.120, "mass": 0.015, "com_z": 0.035,
                     "friction": 0.6}.items():
        setattr(a, k, v)
    if not a.grid:
        for r in trial(a):
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
        for rows in ex.map(trial, jobs):
            f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{len(jobs)} trials -> {a.out}")


if __name__ == "__main__":
    main()
