"""The venue generator's venues in the packages that are already built.

    python -B -m sketch.venue.integration practice --world W.json [--seconds 60] [--speed 1.5] --out OUT.json
    python -B -m sketch.venue.integration oracle   --world W.json --out OUT.json
    python -B -m sketch.venue.integration oracle-batch --worlds DIR --workers N --out DIR      # every <id>.json of DIR

practice  the cell (`Venue(world, config, providers="auto")`: the drive model's car, the sensor model, this package's build_spec with
          the duct model's pushable ducts, the obstacle model) opens the practice session and a scripted driver runs it for `seconds`: pure
          pursuit on the centreline from the cell's privileged truth state (`Venue.truth_state()`, the in-process
          maintainer view; no wire), at `speed` m/s, below the mapping phase's 2.0 m/s. Reported: errors in any
          reply, laps and lap times, touches, crashes, warnings, the cell's timing split.
oracle    the reference program (`sketch.oracle.run_inprocess`: the program through the cell's in-process transport,
          its token in memory only) on the world, all four sessions. Reported: the venue result's laps per
          session, clean laps, T_best, N_best, the two avoidance phases, touches and crashes.
"""

import argparse
import copy
import json
import math
import sys
import time

from sketch.examples import check
from sketch.examples import constants as K


def _pursuit(tr, pose, lookahead=0.8):
    x, y, yaw = pose
    rx, ry = x + K.REF_POINT_C[0] * math.cos(yaw), y + K.REF_POINT_C[0] * math.sin(yaw)
    s, _ = tr.project(rx, ry)
    tx, ty, _, _, _ = tr.at(s + lookahead)
    alpha = math.atan2(ty - y, tx - x) - yaw
    ld = max(0.3, math.hypot(tx - x, ty - y))
    return max(-K.STEER_MAX_RAD, min(K.STEER_MAX_RAD, math.atan2(2.0 * K.WHEELBASE_M * math.sin(alpha), ld)))


def practice(world, seconds=60.0, speed=1.5):
    from sketch.referee import Venue, load_config
    cfg = load_config()
    t0 = time.time()
    v = Venue(copy.deepcopy(world), cfg, providers="auto")
    sources = dict(v.prov.sources)
    tr = check.Track(world)
    out = v.open_session("practice")
    errors = []
    if "error" in out:
        errors.append(out)
    ticks = int(round(seconds / cfg["sim"]["tick_s"]))
    events = []
    steer = 0.0
    for n in range(ticks):
        r = v.step(steer, speed)
        if "error" in r:
            errors.append(r)
            break
        for e in r["events"]:
            events.append(dict(e, tick=r["tick"]))
        st = v.truth_state()
        steer = _pursuit(tr, st["car"]["pose"])
    c = v.close_session()
    if "error" in c:
        errors.append(c)
    res = v.result()
    ses = res["sessions"][0]
    return {"world_id": world["world_id"], "sources": sources, "ticks": ses["ticks"], "errors": errors,
            "laps": [{"n": x["n"], "lap_s": x["lap_s"], "clean": x["clean"]} for x in ses["laps"]],
            "touches": ses["touches"], "crashes": ses["crashes"], "warnings": ses["warnings"],
            "out_of_bounds": ses["out_of_bounds"], "events": events[:40], "end_reason": res["end_reason"],
            "check_result": check.check_result(res, cfg), "wall_s": round(time.time() - t0, 1),
            "timing": {k: round(float(x), 3) for k, x in getattr(v, "timing", {}).items()}}


def oracle(world):
    from sketch.oracle import run_inprocess
    from sketch.referee import load_config
    t0 = time.time()
    res, venue, stats = run_inprocess(copy.deepcopy(world))
    sess = {}
    for s in res["sessions"]:
        sess[s["name"]] = {"laps": len(s["laps"]), "clean": sum(1 for x in s["laps"] if x["clean"]),
                           "lap_s": [round(x["lap_s"], 3) for x in s["laps"] if x["lap_s"] is not None][:40],
                           "touches": s["touches"], "crashes": s["crashes"], "pickups": s["pickups"],
                           "out_of_bounds": s["out_of_bounds"], "end": s["end"]}
    return {"world_id": world["world_id"], "end_reason": res["end_reason"], "time_trial": res["time_trial"],
            "avoid": res["avoid"], "sessions": sess, "stats": res["stats"], "program_stats": stats,
            "check_result": check.check_result(res, load_config()), "wall_s": round(time.time() - t0, 1)}


def _oracle_job(job):
    path, out_dir = job
    import os
    with open(path, "r", encoding="utf-8") as f:
        w = json.load(f)
    try:
        out = oracle(w)
    except Exception as e:                                   # reported, never hidden
        out = {"world_id": w["world_id"], "error": "%s: %s" % (type(e).__name__, e)}
    with open(os.path.join(out_dir, "oracle_%s.json" % w["world_id"]), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    return out


def oracle_batch(world_dir, out_dir, workers=1):
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(os.path.join(world_dir, f) for f in os.listdir(world_dir)
                   if f.endswith(".json") and not f.endswith(".venue_sheet.json") and not f.endswith(".truth.json"))
    jobs = [(p, out_dir) for p in paths]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            rows = pool.map(_oracle_job, jobs, chunksize=1)
    else:
        rows = [_oracle_job(j) for j in jobs]
    table = []
    for r in rows:
        if "error" in r:
            table.append({"world_id": r["world_id"], "error": r["error"]})
            continue
        ses = r["sessions"]
        table.append({"world_id": r["world_id"], "end_reason": r["end_reason"], "T_best_s": r["time_trial"]["T_best_s"],
                      "N_best": r["time_trial"]["N_best"], "clean_laps_tt1": _run(r, "tt1"),
                      "T_static_s": r["avoid"]["static"]["T_clean_s"], "T_dynamic_s": r["avoid"]["dynamic"]["T_clean_s"],
                      "touches": r["stats"]["touches"], "crashes": r["stats"]["crashes"],
                      "laps": {k: v["laps"] for k, v in ses.items()}, "check_result": len(r["check_result"]),
                      "wall_s": r["wall_s"]})
    with open(os.path.join(out_dir, "oracle_table.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, indent=1)
    return table


def _run(r, name):
    return r["sessions"][name]["clean"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("practice", "oracle", "oracle-batch"))
    ap.add_argument("--world")
    ap.add_argument("--worlds")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.what == "oracle-batch":
        table = oracle_batch(a.worlds, a.out, a.workers)
        print(json.dumps(table, indent=1))
        return 0
    with open(a.world, "r", encoding="utf-8") as f:
        w = json.load(f)
    out = practice(w, a.seconds, a.speed) if a.what == "practice" else oracle(w)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: out[k] for k in out if k not in ("events", "program_stats")}, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
