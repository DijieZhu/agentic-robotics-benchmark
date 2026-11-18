"""The gates of the sensor model, as functions (the tests call them) and a command line:

    python -B -m sketch.sensors.gates {analytic,distortion,timing,determinism,all} [--out DIR] [--ticks N]

  analytic     ranges in a known scene (the example venue, several level, pitched and rolled poses, both variants,
               a stub slow car) against the independent analytic caster `reference.py`: exact without noise, and the
               residuals with the default noise. Since the first revision the residuals are judged by their
               distribution: the counts of |residual| >= 21 mm and >= 31 mm (2 and 3 sigma) each lie inside the
               two-sided 99.9 % binomial interval around the count expected for Gaussian noise of `noise_sigma_m`
               on integer-mm readings (`tail_expected`); the largest residual is reported, not judged
  distortion   the car moves at 5 m/s; per sub-scan the sensor offset that best explains the scan is fitted against
               the analytic scene; the travel inside one scan and from one revolution to the next
  timing       the cost of one whole scan (5 sub-scans + the beam rules) on 902 duct pieces, and the raw cost of one
               1081-beam mj_multiRay, by layout
  determinism  sha256 of the observations of a fixed physics run; equal on reruns and across machines
Each prints JSON and, with --out, writes <gate>.json there.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time

import mujoco
import numpy as np

from sketch.examples import check
from sketch.examples import constants as K

from . import reference, scenes
from .params import load_params, noiseless
from .suite import SensorSuite

DISTORTION_SPEED_MPS = 5.0
TAIL_MM = (21, 31)          # |residual| >= these (2 and 3 sigma of 10 mm on integer mm readings)
TAIL_Z = 3.2905             # two-sided 99.9 % of the normal approximation of the binomial count


def tail_probability(k_mm, sigma_mm):
    """P(|rint(x + e) - rint(x)| >= k) for e ~ N(0, sigma) and the fraction of x uniform: the expected tail of the
    integer-mm residuals of the analytic gate (quadrature over the fraction, 2001 points)."""
    f = np.linspace(-0.5, 0.5, 2001)
    up = 0.5 * np.array([math.erfc((k_mm - 0.5 - v) / (sigma_mm * math.sqrt(2.0))) for v in f])
    lo = 0.5 * np.array([math.erfc((k_mm - 0.5 + v) / (sigma_mm * math.sqrt(2.0))) for v in f])
    return float(np.trapezoid(up + lo, f))


def tail_verdict(counts, n, sigma_mm):
    """{k: {"count", "expected", "lo", "hi", "ok"}} of the distribution test."""
    out = {}
    for k in TAIL_MM:
        p = tail_probability(k, sigma_mm)
        mu, sd = n * p, math.sqrt(n * p * (1.0 - p))
        c = int(counts[str(k)])
        out[str(k)] = {"count": c, "expected": mu, "lo": mu - TAIL_Z * sd, "hi": mu + TAIL_Z * sd,
                       "ok": bool(mu - TAIL_Z * sd <= c <= mu + TAIL_Z * sd)}
    return out


# ------------------------------------------------------------------------------------------------ analytic
def geom_item(name):
    """(kind, ident) of a hit geom, in the terms of `reference.Primitives.items`."""
    parts = name.split("/")
    if parts[0] == "duct":
        return ("duct", (parts[1], int(parts[2])))
    if parts[0] in ("clutter", "box"):
        return (parts[0], parts[1])
    if parts[0] in ("floor", "zone"):
        return ("floor", "floor")
    return ("car", name)


def analytic_case(world, pose, variant="track", cars=None, params=None, ticks=1, roll=0.0, pitch=0.0, z=0.0, seed=7):
    """Scan the static scene from a fixed pose; compare with the analytic caster. Returns per-case numbers."""
    allcars = {K.EGO_PREFIX: list(pose[:3])}
    allcars.update(cars or {})
    model, data = scenes.model_of(world, variant=variant, cars=allcars)
    p = load_params(params)
    suite = SensorSuite(model, data, world, params=p, seed=seed)
    scenes.set_pose(model, data, pose[0], pose[1], pose[2], z=z, roll=roll, pitch=pitch)
    mujoco.mj_forward(model, data)
    pos, rot = suite.lidar.sensor_pose(data)
    dirs = suite.lidar.local @ rot.T
    prims = reference.scene_primitives(world, variant=variant, cars=cars)
    dist, which = reference.cast(prims, pos, dirs)
    gap = reference.second_gap(prims, pos, dirs)
    noisy = p["lidar"]["noise_sigma_m"] > 0.0 or p["lidar"]["range_fade_frac"] > 0.0
    exact_p = load_params(noiseless()) if noisy else p
    expect = reference.expected_mm(prims, dist, which, exact_p,
                                   reference.floor_return_p(world, exact_p, pos, dirs, dist, which, prims))
    kin = scenes.Kinematic(model, data, suite, scenes.straight(pose[0], pose[1], pose[2], 0.0, z=z, roll=roll, pitch=pitch))
    scans = np.array([kin.tick()["scan"] for _ in range(ticks)])
    true_m = suite.lidar.last["true_m"]
    hit_items = [geom_item(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))) if g >= 0 else None
                 for g in suite.lidar.last["geomid"]]
    ana_items = [prims.items[w][:2] if w >= 0 else None for w in which]
    kinds = [prims.items[w][0] if w >= 0 else None for w in which]
    # beams whose wire value is decided by the fade band are compared statistically elsewhere
    fade = p["lidar"]["range_fade_frac"]
    band = np.zeros(K.N_BEAMS, bool)
    for i, w in enumerate(which):
        if w >= 0 and prims.items[w][0] != "floor" and np.isfinite(dist[i]):
            r = p["lidar"]["range_by_class_m"][prims.items[w][2]]
            band[i] = r < dist[i] <= r * (1.0 + fade) if fade > 0 else False
    finite = np.isfinite(dist) & (which >= 0)
    out = {"pose": [round(float(v), 6) for v in pose[:3]], "roll_rad": roll, "pitch_rad": pitch, "variant": variant,
           "cars": sorted((cars or {}).keys()), "ticks": ticks,
           "beams_hit_analytic": int(finite.sum()), "beams_expected_return": int((expect > 0).sum()),
           "beams_floor": int(sum(1 for k in kinds if k == "floor")),
           "beams_black_beyond_4m": int(sum(1 for i, w in enumerate(which) if w >= 0 and prims.items[w][2] in ("black_plain", "dark") and dist[i] > 4.0)),
           "true_range_max_abs_err_m": float(np.max(np.abs(np.where(finite & (true_m >= 0), true_m - dist, 0.0)))),
           "min_second_surface_gap_m": float(np.min(gap[finite])) if finite.any() else None,
           "item_mismatch_beams": int(sum(1 for i in range(K.N_BEAMS) if hit_items[i] != ana_items[i] and gap[i] > 1e-6))}
    keep = ~band
    if not noisy:
        out["mismatch_beams"] = int((scans[:, keep] != expect[keep]).any(axis=0).sum())
        out["max_abs_diff_mm"] = int(np.abs(scans[:, keep] - expect[keep]).max())
    else:
        both = keep & (expect > 0)
        same_set = (scans[:, keep] > 0) == (expect[keep] > 0)
        res = (scans[:, both] - expect[both]).astype(float).ravel()
        out.update({"return_set_mismatch_beams": int((~same_set).any(axis=0).sum()), "residual_n": int(res.size),
                    "residual_mean_mm": float(res.mean()), "residual_std_mm": float(res.std()),
                    "residual_max_abs_mm": float(np.abs(res).max()), "fade_band_beams": int(band.sum()),
                    "residual_tail_counts": {str(k): int((np.abs(res) >= k).sum()) for k in TAIL_MM}})
    return out


def analytic_poses(world):
    """Fixed poses for the analytic gate: the start pose and stations along the example track, with tilts."""
    tr = check.Track(world)
    out = [(world["truth"]["start_pose"]["pose"], 0.0, 0.0, 0.0)]
    for s, d, roll, pitch in ((3.0, 0.2, 0.0, 0.0), (7.5, -0.3, 0.0, 0.03), (11.0, 0.0, -0.05, 0.0), (15.5, 0.4, 0.02, -0.02),
                              (19.0, -0.2, 0.0, 0.05), (23.0, 0.1, 0.08, 0.0), (26.0, -0.4, -0.03, 0.04)):
        x, y, hd, _, _ = tr.at(s)
        out.append(([x - d * math.sin(hd), y + d * math.cos(hd), hd], roll, pitch, 0.0))
    return out


def run_analytic(world=None, ticks=20):
    world = world or scenes.min_world()
    cases = []
    slow = world["truth"]["start_pose"]["pose"]
    ahead = [slow[0] + 2.0, slow[1] + 0.1, slow[2]]
    for params, label, nt in ((noiseless(), "noiseless", 1),
                              ({"lidar": {"range_fade_frac": 0.0}}, "noise_no_fade", ticks)):
        for j, (pose, roll, pitch, z) in enumerate(analytic_poses(world)):
            c = analytic_case(world, pose, params=params, ticks=nt, roll=roll, pitch=pitch, z=z, seed=100 + j)
            c["params"] = label
            cases.append(c)
        c = analytic_case(world, slow, variant="avoid", cars={K.slow_prefix(0): ahead}, params=params, ticks=nt, seed=199)
        c["params"] = label
        cases.append(c)
    exact = [c for c in cases if c["params"] == "noiseless"]
    noisy = [c for c in cases if c["params"] != "noiseless"]
    res_n = sum(c["residual_n"] for c in noisy)
    mean = sum(c["residual_mean_mm"] * c["residual_n"] for c in noisy) / res_n
    var = sum((c["residual_std_mm"] ** 2 + c["residual_mean_mm"] ** 2) * c["residual_n"] for c in noisy) / res_n - mean ** 2
    counts = {str(k): sum(c["residual_tail_counts"][str(k)] for c in noisy) for k in TAIL_MM}
    sigma_mm = 1000.0 * float(load_params()["lidar"]["noise_sigma_m"])
    tails = tail_verdict(counts, res_n, sigma_mm)
    return {"gate": "analytic", "cases": cases,
            "summary": {"exact_cases": len(exact), "exact_mismatch_beams": sum(c["mismatch_beams"] for c in exact),
                        "exact_max_abs_diff_mm": max(c["max_abs_diff_mm"] for c in exact),
                        "true_range_max_abs_err_m": max(c["true_range_max_abs_err_m"] for c in cases),
                        "noisy_cases": len(noisy), "noisy_ticks_per_case": ticks, "residual_n": res_n,
                        "residual_mean_mm": mean, "residual_std_mm": math.sqrt(var),
                        "residual_max_abs_mm": max(c["residual_max_abs_mm"] for c in noisy),
                        "residual_tails": tails, "residual_distribution_ok": all(v["ok"] for v in tails.values()),
                        "return_set_mismatch_beams": sum(c["return_set_mismatch_beams"] for c in noisy),
                        "beams_floor": sum(c["beams_floor"] for c in exact),
                        "beams_black_beyond_4m": sum(c["beams_black_beyond_4m"] for c in exact)}}


# ------------------------------------------------------------------------------------------------ distortion
GATE_M = (0.20, 0.10, 0.05)     # residual gates of the first fit iterations, then the last one stays


def fit_offsets(prims, scan_mm, origins, rot, local, iters=10, h=1e-4):
    """Per sub-scan k: the planar offset (dx, dy) from `origins[k]` of the sensor position that best explains the
    sub-scan's returned ranges against the analytic scene (attitude given). Gauss-Newton on range residuals; a beam
    takes part in an iteration only if its residual is below the gate of that iteration (0.20, 0.10, then 0.05 m), so
    beams that change surface at an edge (a pillar's silhouette) do not pull the fit. Returns (5, 2) offsets and the
    number of beams used in the last iteration of each sub-scan."""
    out, used = [], []
    scan = np.asarray(scan_mm)
    for k, (a, b) in enumerate(K.SUBSCAN_SLICES):
        sel = scan[a:b] > 0
        dirs = local[a:b][sel] @ rot.T
        meas = scan[a:b][sel] / 1000.0
        delta = np.zeros(2)
        n_used = 0
        for it in range(iters):
            o = np.asarray(origins[k], dtype=float) + np.array([delta[0], delta[1], 0.0])
            r0, w0 = reference.cast(prims, o, dirs)
            jac = []
            ok = np.isfinite(r0)
            with np.errstate(invalid="ignore"):
                ok &= np.abs(meas - r0) < GATE_M[min(it, len(GATE_M) - 1)]
                for e in (np.array([h, 0.0, 0.0]), np.array([0.0, h, 0.0])):
                    r1, w1 = reference.cast(prims, o + e, dirs)
                    ok &= np.isfinite(r1) & (w1 == w0)
                    jac.append((r1 - r0) / h)
            jac = np.stack(jac, axis=1)
            step = np.linalg.lstsq(jac[ok], (meas - r0)[ok], rcond=None)[0]
            delta += step
            n_used = int(ok.sum())
            if it >= len(GATE_M) and np.abs(step).max() < 1e-7:
                break
        out.append([float(delta[0]), float(delta[1])])
        used.append(n_used)
    return np.array(out), used


def run_distortion(ticks=6, speed=DISTORTION_SPEED_MPS, dynamic=False, params=None, seed=11):
    """Drive at `speed` along +x through the round room; fit the five sub-scan positions of every scan."""
    world = scenes.round_room_world()
    x0, y0, yaw = -0.9, -0.35, 0.0
    model, data = scenes.model_of(world, ego_pose=[x0, y0, yaw])
    p = load_params(params)
    suite = SensorSuite(model, data, world, params=p, seed=seed)
    prims = reference.scene_primitives(world)
    h = float(model.opt.timestep)
    scenes.set_state(model, data, [x0, y0, 0.0], scenes.quat_rpy(0, 0, yaw), [speed, 0.0, 0.0])
    if dynamic:
        model.opt.gravity[:] = 0.0
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        step = lambda: mujoco.mj_step(model, data)
    else:
        kin = scenes.Kinematic(model, data, suite, scenes.straight(x0, y0, yaw, speed))
    mujoco.mj_forward(model, data)
    rows = []
    for n in range(ticks):
        start = suite.lidar.sensor_pose(data)[0]
        t0 = n * suite.n_steps * h
        obs = kin.tick() if not dynamic else suite.tick(step)
        rot = suite.lidar.sensor_pose(data)[1]
        off, used = fit_offsets(prims, obs["scan"], [start] * 5, rot, suite.lidar.local)
        cast_pos = suite.lidar.last["origin"]
        rows.append({"tick": n + 1, "t0_s": t0, "fitted_dx_m": off[:, 0].tolist(), "fitted_dy_m": off[:, 1].tolist(),
                     "beams_used": used, "cast_dx_m": (cast_pos[:, 0] - start[0]).tolist(), "start_x_m": float(start[0])})
    fx = np.array([r["fitted_dx_m"] for r in rows])
    abs_x = fx + np.array([[r["start_x_m"]] for r in rows])
    within = fx[:, 4] - fx[:, 0]
    per_rev = (abs_x[1:] - abs_x[:-1]).ravel()
    slope = np.array([np.polyfit(np.arange(5), fx[i], 1)[0] for i in range(len(rows))])
    t_k = np.array([K.subscan_step(k, suite.n_steps) * h for k in range(5)])
    naive = speed * (suite.n_steps * h - t_k)
    return {"gate": "distortion", "speed_mps": speed, "dynamic": dynamic, "ticks": ticks, "rows": rows,
            "subscan_times_s": t_k.tolist(),
            "summary": {"within_scan_first_to_last_m": {"mean": float(within.mean()), "min": float(within.min()), "max": float(within.max())},
                        "per_revolution_m": {"mean": float(per_rev.mean()), "min": float(per_rev.min()), "max": float(per_rev.max())},
                        "rate_per_revolution_m": {"mean": float(5 * slope.mean()), "min": float(5 * slope.min()), "max": float(5 * slope.max())},
                        "fitted_dy_max_abs_m": float(np.abs(np.array([r["fitted_dy_m"] for r in rows])).max()),
                        "fit_vs_cast_max_abs_m": float(np.abs(fx - np.array([r["cast_dx_m"] for r in rows])).max()),
                        "snapshot_error_by_subscan_m": naive.tolist(),
                        "expected": {"within_scan_m": speed * (t_k[4] - t_k[0]), "per_revolution_m": speed * K.TICK_S}}}


# ------------------------------------------------------------------------------------------------ timing
def time_scans(world, ticks=300, warm=30, speed=5.0, cutoff=None, duct_bodies=None):
    """Median cost of one whole scan of the suite's LiDAR (5 sub-scans + finish) while the car moves at `speed`,
    and of one plain 1081-beam mj_multiRay from the same poses. Physics is not stepped (static world).

    Two clocks: wall time (perf_counter) and the thread's own CPU time (CLOCK_THREAD_CPUTIME_ID), which leaves out
    the time the thread waits for a core; on a loaded machine the second estimates the cost on an unloaded core
    (an upper bound when a busy SMT sibling slows the core)."""
    model, data = scenes.model_of(world, duct_bodies=duct_bodies)
    suite = SensorSuite(model, data, world)
    lid = suite.lidar
    if cutoff is not None:
        lid.cutoff = float(cutoff)
    x, y, yaw = world.get("car_pose") or world["truth"]["start_pose"]["pose"]
    kin = scenes.Kinematic(model, data, suite, scenes.straight(x, y, yaw, speed))
    gid, dist = np.zeros(K.N_BEAMS, np.int32), np.zeros(K.N_BEAMS)
    mask = np.array(K.LIDAR_FLOOR_GROUPMASK, np.uint8)
    scan, fin, raw, imu, scan_cpu, raw_cpu = [], [], [], [], [], []
    wall, cpu = time.perf_counter, time.thread_time
    for it in range(ticks + warm):
        suite.begin_tick()
        ts = ti = tc = 0.0
        for i in range(1, suite.n_steps + 1):
            kin.t += kin.h
            kin.apply()
            k = suite.subscan_at.get(i)
            c0, t0 = cpu(), wall()
            if k is not None:
                lid.cast(k, data)
            t1, c1 = wall(), cpu()
            suite.imu.after_step(i, data)
            t2 = wall()
            suite.i = i
            ts += t1 - t0
            tc += c1 - c0
            ti += t2 - t1
        c0, t0 = cpu(), wall()
        lid.finish(suite.rng_lidar)
        t1, c1 = wall(), cpu()
        suite.imu.finish()
        suite.motor.read(data)
        suite.i = None
        pos, rot = lid.sensor_pose(data)
        vec = np.ascontiguousarray((lid.local @ rot.T).reshape(-1))
        c2, t2 = cpu(), wall()
        mujoco.mj_multiRay(model, data, pos, vec, mask, 1, lid.chassis.body, gid, dist, None, K.N_BEAMS, lid.cutoff)
        t3, c3 = wall(), cpu()
        if it >= warm:
            scan.append(ts + (t1 - t0))
            scan_cpu.append(tc + (c1 - c0))
            fin.append(t1 - t0)
            raw.append(t3 - t2)
            raw_cpu.append(c3 - c2)
            imu.append(ti)

    def stats(v):
        v = 1e3 * np.asarray(v)
        return {"median_ms": round(float(np.median(v)), 4), "p25_ms": round(float(np.percentile(v, 25)), 4),
                "p90_ms": round(float(np.percentile(v, 90)), 4), "min_ms": round(float(v.min()), 4)}
    n_pieces = sum(len(d["pieces"]) for d in world["truth"]["ducts"])
    lidar_geoms = int(sum(1 for g in range(model.ngeom) if int(model.geom_group[g]) in (K.GROUP_LIDAR, K.GROUP_CLUTTER, K.GROUP_FLOOR)))
    return {"world": world["world_id"], "duct_bodies": duct_bodies or "static", "nbody": int(model.nbody),
            "duct_pieces": n_pieces, "lidar_geoms": lidar_geoms, "ngeom": int(model.ngeom),
            "cutoff_m": lid.cutoff, "ticks": ticks, "scan": stats(scan), "scan_thread_cpu": stats(scan_cpu),
            "finish": stats(fin), "raw_multiray_1081": stats(raw), "raw_multiray_1081_thread_cpu": stats(raw_cpu),
            "imu_per_tick": stats(imu)}


def run_timing(ticks=300):
    rings, rows = scenes.duct_field_world(902, "rings"), scenes.duct_field_world(902, "rows")
    cases = [time_scans(rings, ticks=ticks),
             time_scans(rings, ticks=ticks, duct_bodies="piece"),
             time_scans(rings, ticks=ticks, duct_bodies="duct"),
             time_scans(rows, ticks=ticks),
             time_scans(rows, ticks=ticks, duct_bodies="piece"),
             time_scans(rows, ticks=ticks, cutoff=30.0),
             time_scans(scenes.min_world(), ticks=ticks),
             time_scans(scenes.min_world(), ticks=ticks, duct_bodies="piece")]
    return {"gate": "timing", "cases": cases, "machine": machine_info()}


# ------------------------------------------------------------------------------------------------ determinism
def determinism_digest(ticks=40, seed=None, timestep=K.PHYSICS_DT_S):
    """sha256 of the observations of a fixed run: the stub car dropped onto the example venue's start pose with a
    push and a turn (mj_step physics, contacts and gravity on), observations in wire form."""
    world = scenes.min_world()
    model, data = scenes.model_of(world, timestep=timestep)
    suite = SensorSuite(model, data, world, seed=seed)
    qa, va = scenes.root_adr(model)
    data.qpos[qa + 2] = 0.004
    data.qvel[va:va + 6] = [3.0, 0.2, 0.0, 0.0, 0.0, 0.8]
    mujoco.mj_forward(model, data)
    hsh = hashlib.sha256()
    step = lambda: mujoco.mj_step(model, data)
    for _ in range(ticks):
        obs = suite.tick(step)
        hsh.update(json.dumps(obs, sort_keys=True, separators=(",", ":")).encode("ascii"))
    return hsh.hexdigest()


def run_determinism(ticks=40):
    a = determinism_digest(ticks)
    b = determinism_digest(ticks)
    c = determinism_digest(ticks, seed=12345)
    return {"gate": "determinism", "ticks": ticks, "digest": a, "rerun_digest": b, "rerun_equal": a == b,
            "other_seed_digest": c, "other_seed_differs": c != a,
            "fallback_step_digest": determinism_digest(ticks, timestep=K.PHYSICS_DT_FALLBACK_S), "machine": machine_info()}


def machine_info():
    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), "")
    except OSError:
        pass
    return {"host": platform.node(), "cpu": cpu, "load1": round(os.getloadavg()[0], 1), "mujoco": mujoco.__version__,
            "numpy": np.__version__, "python": platform.python_version(),
            "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES")},
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gate", choices=("analytic", "distortion", "timing", "determinism", "all"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ticks", type=int, default=None)
    a = ap.parse_args(argv)
    todo = ("analytic", "distortion", "timing", "determinism") if a.gate == "all" else (a.gate,)
    for g in todo:
        t0 = time.time()
        if g == "analytic":
            r = run_analytic(ticks=a.ticks or 20)
        elif g == "distortion":
            r = {"kinematic": run_distortion(ticks=a.ticks or 6), "dynamic": run_distortion(ticks=a.ticks or 6, dynamic=True)}
        elif g == "timing":
            r = run_timing(ticks=a.ticks or 300)
        else:
            r = run_determinism(ticks=a.ticks or 40)
        r = dict(r, wall_s=round(time.time() - t0, 1), machine=machine_info())
        text = json.dumps(r, indent=1, sort_keys=True)
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, g + ".json"), "w") as f:
                f.write(text + "\n")
        summary = {k: r[k] for k in ("summary",) if k in r} or {k: v for k, v in r.items() if k not in ("rows", "cases")}
        if g == "distortion":
            summary = {m: r[m]["summary"] for m in ("kinematic", "dynamic")}
        if g == "timing":
            summary = [{k: c[k] for k in ("world", "duct_bodies", "duct_pieces", "cutoff_m", "scan", "scan_thread_cpu", "raw_multiray_1081")}
                       for c in r["cases"]]
        print(g, json.dumps(summary, sort_keys=True)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
