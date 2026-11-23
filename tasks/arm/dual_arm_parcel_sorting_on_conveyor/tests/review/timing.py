"""Timing of the cell, for the verify timeout (a maintainer's tool; the grader never runs it).

    MUJOCO_GL=egl|osmesa python3 -B tests/review/timing.py --out DIR [--shifts 3] [--frames 40] [--tag NAME]

Measured through the server's own call path (``ParcelCellServer.handle``: the world, the safety watch, the record,
the camera ring and the tool-pose level):

* the pace: wall seconds per simulated second for a whole shift, the robot holding the start posture and the robot
  moving (sinusoidal joint targets on both arms), with the cell's own split (``tick_wall_s``: inside the world's
  step) and the server's overhead around it; the cell build time;
* the cost of a frame per camera through ``get_camera`` (render, noise, PNG, base64), colour only and colour with
  depth, at fresh exposures (median and p95), the first frame of a cell, and a repeat;
* the verify-timeout arithmetic (the usual template) with these inputs, for a graded shift (the fixed shift
  duration, ``docs/facts.py``): T = (shifts + 1 re-run) x [build + first frame + duration x pace + program
  budget + frame budget x p95 frame + 60 + 30] + 250, then x 1.35.

Run it on the grading hardware in the task image for the numbers that go into task.toml; on another machine the
numbers are indicative only. Output: ``DIR/timing_<tag>.json`` and a printed summary.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import common  # noqa: E402

CAMERAS = ("infeed", "head", "wrist_left", "wrist_right")


def pace(package, moving):
    import numpy as np
    srv = common.make_server()
    t0 = time.time()
    info = srv.reset(package)
    build = time.time() - t0
    q0 = np.asarray(info["robot"]["start_posture"], float)
    t0 = time.time()
    k = 0
    st = srv.handle("step", {"q_target": q0.tolist(), "grip": [0.0, 0.0], "n": 1})
    while not st["shift_over"]:
        if moving:
            q = q0.copy()
            q[6] = 0.6 * math.sin(0.15 * k)
            q[13] = -0.6 * math.sin(0.15 * k)
            q[2] = 0.3 * math.sin(0.05 * k)
            st = srv.handle("step", {"q_target": q.tolist(), "n": 5})
        else:
            st = srv.handle("step", {"n": 50})
        k += 1
    wall = time.time() - t0
    rec = srv.finalize()
    out = {"moving": moving, "sim_s": rec["t_end"], "wall_s": round(wall, 3),
           "wall_per_sim_s": round(wall / max(rec["t_end"], 1e-9), 5),
           "tick_wall_per_sim_s": round(rec["tick_wall_s"] / max(rec["t_end"], 1e-9), 5),
           "build_s": round(build, 3), "n_parcels": rec["shift"]["n_parcels"], "stops": rec["safety"]["protective_stop_count"]}
    srv.close()
    return out


def frames(package, n):
    import numpy as np
    srv = common.make_server()
    srv.reset(package)
    srv.handle("step", {"n": 50 * 20})            # 20 s into the shift: parcels on the belt
    res = {c: {"rgb": [], "rgbd": []} for c in CAMERAS}
    t0 = time.time()
    srv.handle("get_camera", {"name": "head"})
    first = time.time() - t0
    t0 = time.time()
    r = srv.handle("get_camera", {"name": "head"})
    repeat = time.time() - t0
    sizes = {}
    for i in range(n):
        srv.handle("step", {"n": 2})
        for c in CAMERAS:
            depth = bool(i % 2)
            t0 = time.time()
            r = srv.handle("get_camera", {"name": c, "depth": depth})
            dt = time.time() - t0
            if "error" in r:
                continue
            res[c]["rgbd" if depth else "rgb"].append(dt)
            sizes.setdefault(c + ("_rgbd" if depth else "_rgb"), len(r.get("rgb_png_b64", "")) +
                             len(r.get("depth_png_b64", "")))
    srv.close()

    def q(xs, p):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(math.ceil(p * len(xs))) - 1)], 4) if xs else None
    out = {c: {k: {"median_s": q(v, 0.5), "p95_s": q(v, 0.95), "n": len(v)} for k, v in d.items()}
           for c, d in res.items()}
    return {"per_camera": out, "first_frame_s": round(first, 3), "repeat_s": round(repeat, 5), "reply_bytes": sizes}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shifts", type=int, default=3)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--tag", default=os.environ.get("MUJOCO_GL", "default"))
    ap.add_argument("--write-measured", action="store_true",
                    help="write the largest camera reply into docs/measured.json (block 'timing')")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    shifts = common.review_shifts(n=a.shifts)
    paces = []
    for k, seed, pkg in shifts:
        for moving in (False, True):
            p = pace(pkg, moving)
            p.update(k=k, belt_class=pkg["meta"]["belt_class"])
            paces.append(p)
            print("pace k=%d %-6s moving=%s: %.4f wall s per sim s (world %.4f), build %.2f s" % (
                k, p["belt_class"], moving, p["wall_per_sim_s"], p["tick_wall_per_sim_s"], p["build_s"]), flush=True)
    fr = frames(shifts[0][2], a.frames)
    for c, d in fr["per_camera"].items():
        print("frame %-11s rgb median %s p95 %s | rgb+depth median %s p95 %s" % (
            c, d["rgb"]["median_s"], d["rgb"]["p95_s"], d["rgbd"]["median_s"], d["rgbd"]["p95_s"]))
    print("first frame %.3f s, repeat %.5f s" % (fr["first_frame_s"], fr["repeat_s"]))
    # verify-timeout arithmetic (the usual template), for the population's longest shift
    sys.path.insert(0, os.path.join(HERE, "docs"))
    import facts as FACTS
    F, ctx = FACTS.collect()
    cfg, run = ctx["cfg"], ctx["runner"]
    dur = float(F["shift_s"])                        # every graded shift lasts the same fixed time
    pace_max = max(p["wall_per_sim_s"] for p in paces)
    build = max(p["build_s"] for p in paces)
    p95 = max(d["rgbd"]["p95_s"] or 0 for d in fr["per_camera"].values())
    b = cfg["budgets"]
    prog = float(b["program_time_factor"]) * dur + float(b["program_time_extra_s"])
    nframes = math.floor(float(b["frames_per_shift_s"]) * dur)
    per = build + fr["first_frame_s"] + dur * pace_max + prog + nframes * p95 + 60 + 30
    n = int(run["shifts_graded"])
    total = (n + 1) * per + 250
    arith = {"duration_s": dur, "pace_wall_per_sim_s": pace_max, "build_s": build, "first_frame_s": fr["first_frame_s"],
             "program_budget_s": prog, "frames": nframes, "frame_p95_s": p95, "per_shift_s": round(per, 1),
             "shifts_plus_rerun": n + 1, "fixed_s": 250, "total_s": round(total, 1),
             "with_35pct_margin_s": round(total * 1.35, 1)}
    print("verify timeout arithmetic (indicative): per shift %.0f s, total %.0f s, with margin %.0f s" % (
        per, total, total * 1.35))
    if a.write_measured:
        import json
        mp = os.path.join(HERE, "docs", "measured.json")
        mm = json.load(open(mp)) if os.path.exists(mp) else {}
        mm.setdefault("timing", {})["reply_mb_max"] = round(max(fr["reply_bytes"].values()) / 1e6, 2)
        mm["timing"]["tool"] = "tests/review/timing.py"
        with open(mp, "w") as fh:
            json.dump(mm, fh, indent=1)
            fh.write("\n")
    common.dump_json(os.path.join(a.out, "timing_%s.json" % a.tag),
                     {"gl": os.environ.get("MUJOCO_GL"), "lp_threads": os.environ.get("LP_NUM_THREADS"),
                      "paces": paces, "frames": fr, "verify_timeout_arithmetic": arith,
                      "host_note": "measured on the build host, not in the task image"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
