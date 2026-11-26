"""Maintainers' diagnostic (privileged; never run by the grader, not a solution): the signal of observability.py,
read from the recordings' own pictures.

For every scored fabric and each of the two clip kinds it takes the fabric's best clip and camera (from
observability.json), simulates the clip's settled state on the real model with the real material and with the
quantity stepped (friction x 1.05; thickness x 1.08), draws both states as the recordings are drawn (the same
renderer, bundle, look, arm pose and camera, before the camera effects) and compares the two pictures: how many
pixels differ by more than 24 levels (summed over the channels) and by how much on average. The whole fabric moves
with its weave, so the count is larger than the scored edge's shift times its length (the ratio is recorded, it is
not a shift); what it shows is whether the step is in the pictures at a contrast far above the camera's noise
(2 to 4 levels).

    CLOTH_RENDER_DIR=<bundle> python3 -I -B tests/review/observability_pictures.py --workers 10      (one process per pair)
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

TASK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(TASK, "environment", "gen"))
import cphys as W  # noqa: E402
import creal as R  # noqa: E402
import render_clips as RC  # noqa: E402
import csheet as CS  # noqa: E402

STEP = {"drag_release": ("mu", 1.05), "pull_over_edge": ("thickness", 1.08)}


def pair(a):
    import mujoco
    import carm
    import crender
    import cscene
    fid, ci, kind, cam_i, pose, C, m0, mass = a
    key, f = STEP[kind]
    cams = RC.cameras_json()["cameras"]
    cam = cams[cam_i]
    m = carm.arm_model()
    d = mujoco.MjData(m)
    carm.set_arm(m, d, carm.first_pose(m, d, C[-1]))
    mujoco.mj_forward(m, d)
    clip = crender.open_clip(m, cams, cscene.fabric_looks(RC.FABRIC_IDS, 12)[fid])
    imgs, edges = [], []
    for mat in (m0, dict(m0, **{key: m0[key] * f})):
        sim = R.Sim(mat, mass)
        sim.place_flat(pose)
        fr = sim.run(C, record_every=10, start=len(C) - 100)
        if sim.unstable or not len(fr):
            return dict(fabric=fid, clip=ci, kind=kind, failed=True)
        P = fr.mean(0)
        sh = CS.shell(P, (R.NX, R.NY), R.RADIUS, RC.DRAW_THICKNESS, RC.TABLE_BOX, C[-1], W.JAW_W / 2)
        imgs.append(clip.draw(cam["name"], sh, d, C[-1], thickness=RC.DRAW_THICKNESS).astype(np.int32))
        edges.append(P.reshape(R.NX, R.NY, 3)[-1])
    K, Rw, t = np.array(cam["K"]), np.array(cam["R_w2c"]), np.array(cam["t_w2c"])
    uv = (K @ (Rw @ edges[0].T + t[:, None])).T
    uv = uv[:, :2] / uv[:, 2:3]
    length = float(np.linalg.norm(np.diff(uv, axis=0), axis=1).sum())
    diff = np.abs(imgs[0] - imgs[1]).sum(2)
    n = int((diff > 24).sum())
    return dict(fabric=fid, clip=ci, kind=kind, camera=cam["name"], differing_pixels=n, edge_length_px=length,
                changed_pixels_per_edge_pixel=n / max(length, 1.0), mean_difference_levels=float(diff[diff > 24].mean()) if n else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(TASK, "tests", "review", "observability_pictures.json"))
    a = ap.parse_args()
    fab = json.load(open(os.path.join(TASK, "environment", "gen", "fabrics.json")))["fabrics"]
    geo = json.load(open(os.path.join(TASK, "tests", "review", "observability.json")))["rows"]
    jobs, ref = [], {}
    for fid in "ABCDEFGHIJ":
        progs = RC.clip_programs(np.random.default_rng(1000 + ord(fid)))
        for kind in STEP:
            rows = [r for r in geo if r["fabric"] == fid and r["kind"] == kind and not r.get("failed")]
            best = max(rows, key=lambda r: max(r["shift_px"]))
            cam_i = int(np.argmax(best["shift_px"]))
            _k, pose, C = progs[best["clip"]]
            jobs.append((fid, best["clip"], kind, cam_i, pose, C, fab[fid]["material"], fab[fid]["mass_kg"]))
            ref[(fid, kind)] = max(best["shift_px"])
    with ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as ex:
        rows = list(ex.map(pair, jobs))
    for r in rows:
        r["geometric_shift_px"] = ref[(r["fabric"], r["kind"])]
        print(json.dumps(r), flush=True)
    json.dump(dict(rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
