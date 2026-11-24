#!/usr/bin/env python3
"""The test that must fail: clock offset and drift estimated from the logs' stamps alone (no image, point or IMU
content): (1) drift from the stamps' spacing; (2) offset and drift from where each sequence's first and last stamps
fall against the sequence's reference times (imu.csv's first and last stamp). Compared with the estimate "zero".

    <python with numpy, pyyaml> -I -B stamp_leak.py --root <rigs root> --sets practice[,hidden] [--out json]
Pass (exit 0): for each sensor and each of drift and offset, the best stamp-only estimate's RMS error over the rigs
is at least 0.8 x the RMS error of "zero"."""
import argparse, csv, json, os, sys
import numpy as np, yaml
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--sets", required=True); ap.add_argument("--out", default="")
a = ap.parse_args()
def col(path, k):
    with open(path) as fh:
        return np.array([float(r[k]) for r in csv.DictReader(fh)])
rows = []
for s in a.sets.split(","):
    for r in sorted(os.listdir(os.path.join(a.root, s))):
        rd, td = os.path.join(a.root, s, r), os.path.join(a.root, s + "_truth", r)
        gt = yaml.safe_load(open(os.path.join(td, "ground_truth.yaml")))
        seqs = sorted(d for d in os.listdir(rd) if d.startswith("seq_"))
        for sensor in ("camera", "lidar"):
            o, d = gt["time"][sensor]["offset_s"], gt["time"][sensor]["drift_ppm"]
            sl, T0s, first, last, T1s = [], [], [], [], []
            for q in seqs:
                st = col(os.path.join(rd, q, sensor, "timestamps.csv"), "stamp_s"); im = col(os.path.join(rd, q, "imu.csv"), "stamp_s")
                n = np.arange(st.size); A = np.stack([n, np.ones_like(n)], 1)
                # robust to a camera_time_jump stretch: median spacing
                sl.append((np.median(np.diff(st)) / 0.1 - 1.0) * 1e6)
                T0s.append(im[0]); T1s.append(im[-1]); first.append(st[0]); last.append(st[-1])
            T0s, T1s, first, last = map(np.array, (T0s, T1s, first, last))
            # the log starts 0.2..1.0 s (+ up to one period) after the sequence's start and ends as long before its end
            y = np.concatenate([first - T0s - 0.65, last - T1s + 0.65]); t = np.concatenate([T0s, T1s])
            A = np.stack([t, np.ones_like(t)], 1); (dd, oo), *_ = np.linalg.lstsq(A, y, rcond=None)
            rows.append({"rig": s + "/" + r, "sensor": sensor, "offset": o, "drift_ppm": d, "drift_spacing_ppm": float(np.mean(sl)),
                         "drift_bounds_ppm": float(dd * 1e6), "offset_bounds": float(oo), "offset_bounds_mean": float(np.mean(y))})
ok = True; rep = {}
for sensor in ("camera", "lidar"):
    R = [x for x in rows if x["sensor"] == sensor]
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    z_d = rms([x["drift_ppm"] for x in R]); z_o = rms([x["offset"] for x in R])
    e = {"drift, spacing": rms([x["drift_spacing_ppm"] - x["drift_ppm"] for x in R]) / z_d,
         "drift, sequence bounds": rms([x["drift_bounds_ppm"] - x["drift_ppm"] for x in R]) / z_d,
         "offset, sequence bounds (line)": rms([x["offset_bounds"] - x["offset"] for x in R]) / z_o,
         "offset, sequence bounds (mean)": rms([x["offset_bounds_mean"] - x["offset"] for x in R]) / z_o}
    rep[sensor] = {"rigs": len(R), "rms_zero_drift_ppm": z_d, "rms_zero_offset_s": z_o, "error_over_zero": e}
    for k, v in e.items():
        print("%-7s %-34s RMS error / RMS of zero = %.2f %s" % (sensor, k, v, "" if v >= 0.8 else "  <-- LEAK"))
        ok &= v >= 0.8
if a.out:
    json.dump({"rows": rows, "summary": rep, "pass": bool(ok)}, open(a.out, "w"), indent=1)
print("PASS (the stamps do not give the clocks)" if ok else "FAIL: a stamp-only estimate beats the prior")
sys.exit(0 if ok else 1)
