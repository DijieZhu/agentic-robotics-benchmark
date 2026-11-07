"""Completability evidence: the relative pose can be recovered from the docking camera's frames (a maintainer's tool;
the grader never runs it).

    # 1. the privileged reference's 48 fixed-set flights, in process (no rendering): every real request's frame record
    python3 -I -B check_perception.py record --out <scratch>/flights.json
    # 2. in containers of the task image (the camera bundle is root's there): render and estimate, one worker each
    python3 -I -B check_perception.py frames --flights <scratch>/flights.json --worker K --workers 4 --out <scratch>
    # 3. the table (aggregates only)
    python3 -I -B check_perception.py table --scratch <scratch> --out <aggregates.json>

Frames are rendered at every fresh stamp of the reference's flights in two attitudes, as flown (the reference's yaw)
and aimed (the same positions, the boresight through the drum's centre), with the grader's own camera and noise
(camera.Camera, the set's key), exactly as the grader would deliver them. The estimator (estimate_pose.py) uses only
the frames, the star tracker's readings and the agent's documents. Nothing of a fixed-set episode leaves the scratch
folder: its frames are never written, and the per-frame records stay there; only the aggregates (`table`) are kept.

The pre-registered rows (spec 2026-11-07, section 6.2):
  E1 the craft's position relative to the drum's centre, all 5 DT LEDs in frame and the camera within 1.2 m of the DT:
     RMS <= 0.010 m, p99 <= 0.030 m; E2 the target's phase, the same frames: RMS <= 0.004 rad, p99 <= 0.012 rad;
  E3 the port's radius from the BT, >= 2 BT LEDs in frame: RMS <= 0.010 m;
  E4 the craft's position in the far field, aimed, both drum limbs in frame, 1.2-6.2 m: RMS <= 0.05 m;
  O1 all 5 DT LED centres in frame in every fresh frame of the 30 docked steps and the 17 before, every episode;
  O2 the whole drum width in frame, aimed, at every reference position 1.2 m or more out;
  O3 per episode, fresh frames after the re-key's restart with E3's condition and |r_port - 1.14| >= 3 x E3's RMS;
  O4 at t = 0: the drum's centre in frame, part of the drum in frame, the largest off-axis angle.
"""

import argparse
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import oracle_batch  # noqa: E402  (tests/ and solution/privileged/ on the path)
import verify  # noqa: E402
import camera  # noqa: E402

DRUM_R = 0.82


def set_spec():
    with open(os.path.join(TESTS, "hidden", "episodes.json")) as f:
        return json.load(f)


class RecSensors(verify.Sensors):
    log = []

    def observe(self, env, obs16):
        obs, rec = super(RecSensors, self).observe(env, obs16)
        RecSensors.log.append({"t": env.steps, "rec": list(rec) if rec else None, "hold": env.hold_steps,
                               "restart": verify.restart_active(env), "rekey": env.rekey_active})
        return obs, rec


def record(a):
    spec = set_spec()
    out = []
    verify.Sensors = RecSensors
    t0 = time.time()
    for k, e in enumerate(spec["episodes"]):
        RecSensors.log = []
        r = oracle_batch.run_one(e["seed"], spec["probe_key"])
        out.append({"episode": k, "seed": e["seed"], "success": r["success"], "steps": r["steps"],
                    "quality": r["quality"], "log": list(RecSensors.log)})
    verify.Sensors = RecSensors.__bases__[0]
    with open(a.out, "w") as f:
        json.dump({"key": spec["probe_key"], "flights": out}, f)
    print("{} flights, {} real requests, {:.0f} s".format(len(out), sum(len(x["log"]) for x in out), time.time() - t0))


def geometry(x, y, th, ph, rp):
    """What is in the frame of a craft at (x, y, th), the target at phase ph with its port at rp (truth only)."""
    fid = camera.fiducials_world(ph, rp)
    pos = camera.camera_pose(x, y, th)[0]
    inside = {}
    for name, p in fid.items():
        q = camera.project(p, x, y, th)
        normal = (math.cos(ph), math.sin(ph))
        facing = (pos[0] - p[0]) * normal[0] + (pos[1] - p[1]) * normal[1] > 0
        inside[name] = bool(q and 0 <= q[0] <= camera.WIDTH - 1 and 0 <= q[1] <= camera.HEIGHT - 1 and facing)
    dt_x = camera.DT_LEDS[0][1]
    dt_rng = math.hypot(pos[0] - dt_x * math.cos(ph), pos[1] - dt_x * math.sin(ph))   # (the DT is at the lens's height)
    d = math.hypot(pos[0], pos[1])                        # the lens to the drum's axis
    b = th + math.pi                                       # the boresight's yaw
    bearing = camera.wrap(math.atan2(-pos[1], -pos[0]) - b)   # the drum's centre, left of the boresight positive
    half = math.asin(min(1.0, DRUM_R / d))
    hfov = math.atan((camera.WIDTH / 2.0) / camera.FX)
    return {"dt_all": all(inside[n] for n in ("DT1", "DT2", "DT3", "DT4", "DTS")),
            "bt_n": sum(inside[n] for n in ("BT1", "BT2", "BT3")),
            "dt_range": dt_rng, "range": math.hypot(x, y), "lens_range": d,
            "centre_in": abs(bearing) <= hfov, "part_in": abs(bearing) - half <= hfov,
            "limbs_in": abs(bearing) + half <= hfov, "off_axis_deg": math.degrees(abs(bearing))}


def frames(a):
    import numpy as np
    import estimate_pose as EP
    data = json.load(open(a.flights))
    key = int(data["key"], 16)
    mine = [f for f in data["flights"] if f["episode"] % a.workers == a.worker]
    cam = camera.Camera(threads=a.threads)
    path = os.path.join(a.out, "frames_{}.jsonl".format(a.worker))
    t0, n = time.time(), 0
    with open(path, "w") as out:
        for fl in mine:
            seed = fl["seed"]
            log = fl["log"]
            last_t = log[-1]["t"] if fl["success"] else None   # (the 30 docked steps end a docked episode)
            restart_end = next((e["t"] for e in log if e["rekey"] and not e["restart"]), None)
            done = set()
            for e in log:
                rec = e["rec"]
                if rec is None or rec[0] in done:
                    continue
                done.add(rec[0])
                stamp, x, y, th, gx, gy, ph = rec
                rp = math.hypot(gx, gy)
                for att in ("flown", "aimed"):
                    yaw = th if att == "flown" else camera.wrap(math.atan2(y, x))
                    img = cam.frame(key, seed, stamp, x, y, yaw, ph, rp)
                    st = camera.star_tracker(yaw, key, seed, stamp)
                    g = geometry(x, y, yaw, ph, rp)
                    bl = EP.blobs(img)
                    row = {"ep": fl["episode"], "stamp": stamp, "t": e["t"], "att": att, "rp": rp, "geo": g,
                           "hold_window": bool(last_t is not None and e["t"] >= last_t - 46),
                           "after_restart": bool(restart_end is not None and e["t"] >= restart_end),
                           "hold_steps": e["hold"]}
                    dt = EP.find_dt(bl)
                    if dt is not None:
                        pose, resid = EP.pose_from_dt(dt)
                        if pose is not None:
                            p, phi = EP.craft_from_camera(pose, st)
                            row["dt"] = {"pos_err": float(math.hypot(p[0] - x, p[1] - y)),
                                         "phi_err": abs(camera.wrap(phi - ph)), "resid_px": resid,
                                         "accepted": bool(EP.accept_dt(pose, resid))}
                            rp_est, used, bres = EP.port_radius_from_bt(pose, [b for b in bl], ())
                            if rp_est is not None:
                                row["bt"] = {"rp_err": abs(rp_est - rp), "used": used, "resid_px": bres,
                                             "accepted": bool(row["dt"]["accepted"] and bres < EP.GATE_BT_PX)}
                    bt = EP.find_bt(bl)
                    if bt is not None:
                        pose, resid = EP.pose_from_bt(bt, rp)     # (the encoder's radius, its polarity calibrated)
                        if pose is not None:
                            p, phi = EP.craft_from_camera(pose, st)
                            row["btpose"] = {"pos_err": float(math.hypot(p[0] - x, p[1] - y)),
                                             "phi_err": abs(camera.wrap(phi - ph)), "resid_px": resid,
                                             "accepted": bool(resid < EP.GATE_BT_PX and abs(camera.wrap(pose[2] - math.pi)) < 1.4)}
                    out.write(json.dumps(row) + "\n")
                    n += 1
            out.flush()
            print("worker {} episode {} done, {} frames, {:.0f} s".format(a.worker, fl["episode"], n, time.time() - t0),
                  flush=True)
    cam.close()


def _stats(v):
    import numpy as np
    v = np.asarray(v, float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "rms": float(np.sqrt(np.mean(v ** 2))), "p50": float(np.median(v)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def table(a):
    rows = []
    for p in sorted(glob.glob(os.path.join(a.scratch, "frames_*.jsonl"))):
        with open(p) as f:
            rows.extend(json.loads(line) for line in f)
    flights = {f["episode"]: f for f in json.load(open(os.path.join(a.scratch, "flights.json")))["flights"]}
    out = {"frames": len(rows), "episodes": len({r["ep"] for r in rows})}
    def ok_dt(r):
        return "dt" in r and r["dt"]["accepted"]

    def ok_bt(r):
        return "btpose" in r and r["btpose"]["accepted"]
    e1 = [r for r in rows if r["att"] == "flown" and r["geo"]["dt_all"] and r["geo"]["dt_range"] <= 1.2]
    det = [r for r in e1 if ok_dt(r)]
    out["E1"] = dict(_stats([r["dt"]["pos_err"] for r in det]), condition=len(e1), accepted=len(det),
                     detected=sum(1 for r in e1 if "dt" in r))
    out["E2"] = dict(_stats([r["dt"]["phi_err"] for r in det]), condition=len(e1), accepted=len(det))
    bins = {}
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.2)):
        b = [r for r in det if lo <= r["geo"]["dt_range"] < hi]
        bins["{:.1f}-{:.1f}".format(lo, hi)] = {"pos": _stats([r["dt"]["pos_err"] for r in b]),
                                                "phi": _stats([r["dt"]["phi_err"] for r in b])}
    out["E1_E2_by_range"] = bins
    hold = [r for r in det if r["hold_window"]]
    out["E1_E2_hold_window"] = {"pos": _stats([r["dt"]["pos_err"] for r in hold]),
                                "phi": _stats([r["dt"]["phi_err"] for r in hold])}
    e3 = [r for r in rows if r["att"] == "flown" and r["geo"]["bt_n"] >= 2]
    bt = [r for r in e3 if "bt" in r and r["bt"]["accepted"]]
    out["E3"] = dict(_stats([r["bt"]["rp_err"] for r in bt]), condition=len(e3), estimated=len(bt))
    e4 = [r for r in rows if r["att"] == "aimed" and r["geo"]["limbs_in"] and 1.2 <= r["geo"]["range"] <= 6.2]
    best = []
    src = {"dt": 0, "bt": 0, "none": 0}
    for r in e4:
        if ok_dt(r):
            best.append(r["dt"]["pos_err"])
            src["dt"] += 1
        elif ok_bt(r):
            best.append(r["btpose"]["pos_err"])
            src["bt"] += 1
        else:
            src["none"] += 1
    vis = [r for r in e4 if r["geo"]["bt_n"] == 3 or r["geo"]["dt_all"]]
    out["E4"] = dict(_stats(best), condition=len(e4), sources=src,
                     fiducials_in_frame=len(vis), estimated_of_those=sum(1 for r in vis if ok_dt(r) or ok_bt(r)),
                     dt_only=_stats([r["dt"]["pos_err"] for r in e4 if ok_dt(r)]),
                     bt_only=_stats([r["btpose"]["pos_err"] for r in e4 if ok_bt(r)]))
    flown_far = [r for r in rows if r["att"] == "flown" and 1.2 <= r["geo"]["range"] <= 6.2 and ok_bt(r)]
    out["E4_flown_bt"] = _stats([r["btpose"]["pos_err"] for r in flown_far])
    o1 = {}
    for r in rows:
        if r["att"] == "flown" and r["hold_window"]:
            o1.setdefault(r["ep"], []).append(r["geo"]["dt_all"])
    docked = [k for k, f in flights.items() if f["success"]]
    out["O1"] = {"episodes_docked": len(docked), "episodes_all_in_frame": sum(1 for k in docked if all(o1.get(k, [False]))),
                 "frames": sum(len(v) for v in o1.values()), "frames_not_in": sum(v.count(False) for v in o1.values())}
    o2 = [r for r in rows if r["att"] == "aimed" and r["geo"]["range"] >= 1.2]
    out["O2"] = {"frames": len(o2), "drum_width_in_frame": sum(1 for r in o2 if r["geo"]["limbs_in"])}
    e3_rms = out["E3"].get("rms", 0.01) or 0.01
    o3 = {}
    for r in rows:
        if r["att"] == "flown" and r["after_restart"] and r["geo"]["bt_n"] >= 2 and abs(r["rp"] - 1.14) >= 3 * e3_rms:
            o3[r["ep"]] = o3.get(r["ep"], 0) + 1
    counts = [o3.get(k, 0) for k in sorted(flights)]
    out["O3"] = {"per_episode_min": min(counts), "per_episode_median": sorted(counts)[len(counts) // 2],
                 "episodes_with_5_or_more": sum(1 for c in counts if c >= 5), "threshold_m": 3 * e3_rms}
    starts = [r for r in rows if r["att"] == "flown" and r["stamp"] == 0]
    out["O4"] = {"starts": len(starts), "centre_in_frame": sum(1 for r in starts if r["geo"]["centre_in"]),
                 "part_in_frame": sum(1 for r in starts if r["geo"]["part_in"]),
                 "max_off_axis_deg": max(r["geo"]["off_axis_deg"] for r in starts),
                 "fix_at_start": sum(1 for r in starts if ok_dt(r) or ok_bt(r))}
    far_flown = [r for r in rows if r["att"] == "flown" and r["geo"]["range"] > 1.47]
    out["far_flown_fix"] = {"frames": len(far_flown), "dt": sum(1 for r in far_flown if ok_dt(r)),
                            "dt_or_bt": sum(1 for r in far_flown if ok_dt(r) or ok_bt(r))}
    out["gates_px"] = {"dt": __import__("estimate_pose").GATE_DT_PX, "bt": __import__("estimate_pose").GATE_BT_PX}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("record", "frames", "table"))
    ap.add_argument("--out", default="")
    ap.add_argument("--flights", default="")
    ap.add_argument("--scratch", default="")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    {"record": record, "frames": frames, "table": table}[a.mode](a)


if __name__ == "__main__":
    main()
