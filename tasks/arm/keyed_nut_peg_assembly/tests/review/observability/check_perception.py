"""Can a controller see what it needs? Camera-only perception (perceive.py) against the simulator's state, on maintainer
development kits only (a maintainer's evidence; the grader never runs it, and no graded kit is used).

Per kit, from the episode's first observation (its three free images): the shaft tops in the wrist camera's view from
the start pose (peg positions), the bands of every peg in the fixed cameras (which peg is which), each nut's rough
position in the fixed cameras. Then, as a controller would, the arm moves the wrist camera over each nut's rough
position (plain actions, the gripper pointing down) and takes one close-up: the ring's centre and the handle's
direction. Every estimate is compared with the simulator's state.

    docker run --rm --network none -v <task>/tests:/stage:ro --entrypoint /opt/venv/bin/python3 <task image> \\
        -I -B /stage/review/observability/check_perception.py --first 1000 --n 30 > perception_dev_kits.json

Kits: keyed_env.sample_kit(numpy.random.default_rng(k)), layout seed k + 4000, k = first .. first + n - 1.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # tests/ (keyed_env)
sys.path.insert(0, HERE)
import keyed_env as E  # noqa: E402
import perceive as P  # noqa: E402


def down_rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.diag([1.0, -1.0, -1.0])


def rotvec(Rerr):
    c = max(-1.0, min(1.0, (np.trace(Rerr) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([Rerr[2, 1] - Rerr[1, 2], Rerr[0, 2] - Rerr[2, 0], Rerr[1, 0] - Rerr[0, 1]])
    return v * th / (2 * math.sin(th)) if th < math.pi - 1e-6 else v


def close_up(env, obs, xy, cam_off, cam_dz, steps=45, turn=0.0):
    """Move the wrist camera to 0.24 m above `xy` with plain actions (turning the hand by `turn` rad about the
    vertical on the way), then return the wrist image and its pose."""
    R = E.mat_from_quat(obs["ee_quat"])
    left = float(turn)                       # the turn still to make
    for k in range(steps + int(abs(turn) / 0.12)):
        R = E.mat_from_quat(obs["ee_quat"])
        ee = np.asarray(obs["ee_pos"], float)
        off = R[:2, :2] @ cam_off
        tgt = np.array([xy[0] - off[0], xy[1] - off[1], E.TABLE_Z + 0.24 - cam_dz])
        yaw = math.atan2(R[1, 0], R[0, 0])
        step = max(-0.12, min(0.12, left))
        dp = tgt - ee
        n = np.linalg.norm(dp)
        dp = dp * (0.02 / n) if n > 0.02 else dp
        arrived = n < 0.004 and k > 2 and abs(left) < 1e-9
        a = list(dp) + list(rotvec(down_rot(yaw + step) @ R.T)) + [-1.0, 4 if arrived else 0]
        yaw_before = yaw
        obs, _done, _info = env.step(a)
        R2 = E.mat_from_quat(obs["ee_quat"])
        left -= E.wrap_angle(math.atan2(R2[1, 0], R2[0, 0]) - yaw_before)     # what the hand really turned
        if abs(left) < 0.01:
            left = 0.0
        if arrived:
            break
    if obs["wrist"].shape[0] == 0:
        obs, _d, _i = env.step([0, 0, 0, 0, 0, 0, -1.0, 4])
    return obs, obs["wrist"], np.asarray(obs["T_world_wrist"], float)


def run_kit(k):
    kit = E.sample_kit(np.random.default_rng(k))
    env = E.NutEnv(kit["nuts"], early_end=False)
    obs = env.reset(k + 4000, kit["order"])
    cols = [o["colour"] for o in env.work_order()]
    Kw = P.intrinsics(*P.WRIST)
    Tw = np.asarray(obs["T_world_wrist"], float)
    R0 = E.mat_from_quat(obs["ee_quat"])
    ee0 = np.asarray(obs["ee_pos"], float)
    cam_off = R0.T[:2, :2] @ (Tw[:2, 3] - ee0[:2])
    cam_dz = Tw[2, 3] - ee0[2]
    out = {"kit": k, "layout_seed": k + 4000, "pegs": [], "nuts": []}
    pegs = P.find_pegs(obs["wrist"], Kw, Tw, exclude_rows_from=140)
    first = dict(obs)                        # the episode's first observation (its three free images)
    e_hand = None                            # the wrist camera's apparent offset in the hand frame, once measured
    for j, (px, py, _yaw) in enumerate(env.peg_poses):
        d = [math.hypot(p[0] - px, p[1] - py) for p in pegs]
        est = pegs[int(np.argmin(d))] if d else None
        res = []
        for cam in ("left", "right"):
            Kc, Tc = P.fixed_camera(cam)
            res.append(P.count_bands(first[cam], Kc, Tc, est if est is not None else (px, py)))
        good = [r for r in res if r[0] is not None]
        bands = max(good, key=lambda r: r[1])[0] if good else None
        rec = {"bands": j + 1, "found": est is not None, "err_mm": round(1000 * min(d), 2) if d else None,
               "bands_seen": bands, "bands_left": res[0][0], "bands_right": res[1][0],
               "closeup_err_mm": None, "two_view_err_mm": None, "two_view_steps": None, "corrected_err_mm": None}
        if est is not None:
            # two wrist close-ups of the shaft top, the hand turned half a turn between them: the camera's mounting
            # and principal-point errors move the two estimates in opposite directions, so their mean is free of them
            t0 = int(obs["t"])
            obs, img, Tc = close_up(env, obs, est, cam_off, cam_dz)
            a = P.refine_peg_top(img, Kw, Tc, est)
            Ra = E.mat_from_quat(obs["ee_quat"])
            yaw_a = math.atan2(Ra[1, 0], Ra[0, 0])
            if a is not None and e_hand is not None:
                # one close-up, corrected by the offset measured once on an earlier peg of this episode
                ca, sa = math.cos(yaw_a), math.sin(yaw_a)
                fixed = np.asarray(a) - np.array([ca * e_hand[0] - sa * e_hand[1], sa * e_hand[0] + ca * e_hand[1]])
                rec["corrected_err_mm"] = round(1000 * math.hypot(fixed[0] - px, fixed[1] - py), 2)
            half = math.pi if float(obs["q"][6]) > 0 else -math.pi      # away from joint 7's nearer limit
            t1 = int(obs["t"])
            obs, img, Tc = close_up(env, obs, a if a is not None else est, cam_off, cam_dz, turn=half)
            b = P.refine_peg_top(img, Kw, Tc, a if a is not None else est)
            if a is not None:
                rec["closeup_err_mm"] = round(1000 * math.hypot(a[0] - px, a[1] - py), 2)
            if a is not None and b is not None:
                mid = (np.asarray(a) + np.asarray(b)) / 2.0
                rec["two_view_err_mm"] = round(1000 * math.hypot(mid[0] - px, mid[1] - py), 2)
                rec["two_view_steps"] = int(obs["t"]) - t1
                if e_hand is None:
                    off = np.asarray(a) - mid                # view A's apparent offset, turned into the hand frame
                    ca, sa = math.cos(-yaw_a), math.sin(-yaw_a)
                    e_hand = np.array([ca * off[0] - sa * off[1], sa * off[0] + ca * off[1]])
            del t0
        out["pegs"].append(rec)
    imgs = {"left": first["left"], "right": first["right"]}
    coarse = {c: P.coarse_nut(imgs, c, cols) for c in cols}
    for i, col in enumerate(cols):
        p, R = env.nut_state(i)
        yaw = math.atan2(R[1, 0], R[0, 0])
        rec = {"colour": col, "coarse_err_mm": None, "err_mm": None, "yaw_err_deg": None}
        if coarse[col] is not None:
            rec["coarse_err_mm"] = round(1000 * math.hypot(coarse[col][0] - p[0], coarse[col][1] - p[1]), 2)
            obs, img, Tc = close_up(env, obs, coarse[col], cam_off, cam_dz)
            est = P.find_nuts(img, Kw, Tc, [col], kit_colours=cols, refine=True)[col]
            if est is not None:
                rec["err_mm"] = round(1000 * math.hypot(est[0][0] - p[0], est[0][1] - p[1]), 2)
                rec["yaw_err_deg"] = round(math.degrees(E.wrap_angle(est[1] - yaw)), 2)
        out["nuts"].append(rec)
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1000)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    kits = [run_kit(k) for k in range(a.first, a.first + a.n)]
    pegs = [p for r in kits for p in r["pegs"]]
    nuts = [n for r in kits for n in r["nuts"]]
    ok = lambda v: v is not None  # noqa: E731
    pe = [p["err_mm"] for p in pegs if ok(p["err_mm"])]
    ne = [n["err_mm"] for n in nuts if ok(n["err_mm"])]
    ye = [abs(n["yaw_err_deg"]) for n in nuts if ok(n["yaw_err_deg"])]
    ce = [n["coarse_err_mm"] for n in nuts if ok(n["coarse_err_mm"])]
    pct = lambda v: [round(float(x), 2) for x in np.percentile(v, [50, 90, 100])] if v else None  # noqa: E731
    cu = [p["closeup_err_mm"] for p in pegs if ok(p["closeup_err_mm"])]
    tv = [p["two_view_err_mm"] for p in pegs if ok(p["two_view_err_mm"])]
    ts = [p["two_view_steps"] for p in pegs if ok(p["two_view_steps"])]
    co = [p["corrected_err_mm"] for p in pegs if ok(p["corrected_err_mm"])]
    summary = {"kits": len(kits),
               "pegs": len(pegs), "pegs_found": sum(1 for p in pegs if p["found"]), "peg_err_mm_p50_p90_max": pct(pe),
               "peg_closeup_err_mm_p50_p90_max": pct(cu), "peg_two_view_n": len(tv),
               "peg_two_view_err_mm_p50_p90_max": pct(tv), "peg_two_view_steps_p50_max": [float(np.median(ts)), float(max(ts))] if ts else None,
               "peg_corrected_n": len(co), "peg_corrected_err_mm_p50_p90_max": pct(co),
               "bands_right": sum(1 for p in pegs if p["bands_seen"] == p["bands"]),
               "bands_wrong": sum(1 for p in pegs if p["bands_seen"] not in (None, p["bands"])),
               "bands_undecided": sum(1 for p in pegs if p["bands_seen"] is None),
               "nuts": len(nuts), "nuts_found": len(ne), "coarse_err_mm_p50_p90_max": pct(ce),
               "closeup_err_mm_p50_p90_max": pct(ne), "handle_yaw_err_deg_p50_p90_max": pct(ye),
               "handle_flipped": sum(1 for y in ye if y > 90)}
    print(json.dumps({"summary": summary, "kits": kits}, indent=1))


if __name__ == "__main__":
    main()
