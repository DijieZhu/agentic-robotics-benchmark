"""Maintainers' diagnostic (privileged; never run by the grader, not a solution): do the recordings carry the two
quantities at the precision the tolerances ask for?

For every scored fabric it replays the fabric's own drag_release and pull_over_edge clips on the real fabric's model
twice: with the real material, and with the quantity the skill depends on changed by the step the score's
sensitivity table names (friction x 1.05 for the drags; thickness x 1.08, which is bending length x 1.08, for the
pulls). It reports how far the thing a camera can measure moves (the stop position of the fabric's leading point along
the drag; the settled height of the pulled edge), in millimetres and in pixels of each of the three cameras. If that
shift is several times the localisation error of an edge in a 640 x 480 image (a fraction of a pixel on a static
frame), the clips constrain the quantity; if it is below a pixel, the tolerance is not supported by the recordings.

The recordings do not draw the model's vertices but the shell that csheet lays on them (clear of the table, a drawn
thickness). So the same two quantities are also read from the drawn shell's middle surface, and the difference from
the model's value (what the grader scores) is reported: its bias and spread must be well inside the tolerances.

    python3 -I -B tests/review/observability.py --workers 40 --out tests/review/observability.json   (from the task folder)
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


def measure(a):
    kind, pose, C, material, mass = a
    sim = R.Sim(material, mass)
    sim.place_flat(pose)
    fr = sim.run(C, record_every=10, start=len(C) - 100)
    if sim.unstable or not len(fr):
        return None
    P = fr.mean(0)
    # what is drawn (csheet, the renderer's fabric): the middle surface of the drawn shell
    sh = CS.shell(P, (R.NX, R.NY), R.RADIUS, RC.DRAW_THICKNESS, RC.TABLE_BOX, C[-1], W.JAW_W / 2)
    D = 0.5 * (sh["bot"] + sh["top"])
    if kind == "drag_release":
        d = np.array([np.cos(pose[2]), np.sin(pose[2]), 0.0])
        i = int(np.argmax(P @ d))
        return dict(value=float(P[i] @ d), point=P[i].tolist(), dir=d.tolist(), drawn=float((D.reshape(-1, 3) @ d).max()))
    edge = P.reshape(R.NX, R.NY, 3)[-1]
    return dict(value=float(edge[:, 2].mean()), point=edge.mean(0).tolist(), dir=[0.0, 0.0, 1.0], drawn=float(D[-1, :, 2].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(TASK, "tests", "review", "observability.json"))
    a = ap.parse_args()
    fab = json.load(open(os.path.join(TASK, "environment", "gen", "fabrics.json")))["fabrics"]
    cams = RC.cameras_json()["cameras"]
    jobs, meta = [], []
    for fid in "ABCDEFGHIJ":
        for ci, (kind, pose, C) in enumerate(RC.clip_programs(np.random.default_rng(1000 + ord(fid)))):
            if kind not in STEP:
                continue
            key, f = STEP[kind]
            m0 = fab[fid]["material"]
            jobs += [(kind, pose, C, m0, fab[fid]["mass_kg"]), (kind, pose, C, dict(m0, **{key: m0[key] * f}), fab[fid]["mass_kg"])]
            meta.append((fid, ci, kind))
    with ProcessPoolExecutor(a.workers) as pool:
        res = list(pool.map(measure, jobs, chunksize=1))
    rows = []
    for j, (fid, ci, kind) in enumerate(meta):
        r0, r1 = res[2 * j], res[2 * j + 1]
        if r0 is None or r1 is None:
            rows.append(dict(fabric=fid, clip=ci, kind=kind, failed=True))
            continue
        shift = r1["value"] - r0["value"]
        X0 = np.array(r0["point"]); X1 = X0 + shift * np.array(r0["dir"])
        px = []
        for c in cams:
            K, Rw, t = np.array(c["K"]), np.array(c["R_w2c"]), np.array(c["t_w2c"])
            uv = [(K @ (Rw @ X + t)) for X in (X0, X1)]
            uv = [u[:2] / u[2] for u in uv]
            px.append(float(np.linalg.norm(uv[1] - uv[0])))
        rows.append(dict(fabric=fid, clip=ci, kind=kind, step="%s x %.2f" % STEP[kind], shift_mm=1000 * shift, shift_px=px,
                         drawn_minus_model_mm=[1000 * (r0["drawn"] - r0["value"]), 1000 * (r1["drawn"] - r1["value"])]))
    out = {}
    for kind in STEP:
        rr = [r for r in rows if r["kind"] == kind and not r.get("failed")]
        mm = np.abs([r["shift_mm"] for r in rr]); best = np.array([max(r["shift_px"]) for r in rr])
        out[kind] = dict(step="%s x %.2f" % STEP[kind], clips=len(rr), failed=sum(1 for r in rows if r["kind"] == kind and r.get("failed")),
                         shift_mm=dict(min=float(mm.min()), median=float(np.median(mm)), max=float(mm.max())),
                         shift_px_best_camera=dict(min=float(best.min()), median=float(np.median(best)), max=float(best.max())),
                         drawn_minus_model_mm=(lambda b: dict(min=float(b.min()), mean=float(b.mean()), max=float(b.max()), spread=float(b.std())))(
                             np.array([v for r in rr for v in r["drawn_minus_model_mm"]])))
        print(kind, json.dumps(out[kind]), flush=True)
    json.dump(dict(summary=out, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
