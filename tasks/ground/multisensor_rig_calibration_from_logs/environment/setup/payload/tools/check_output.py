"""Is an output folder well-formed? It checks the two files' FORMAT only; it knows nothing about their values.

    python3 ~/tools/check_output.py <out_dir>
"""

import json
import math
import os
import sys

import numpy as np
import yaml

SENSORS = ("camera", "lidar", "imu", "wheel_odom")
TYPES = ("camera_frozen", "camera_time_jump", "lidar_degraded", "imu_axis_swap", "odom_glitch", "camera_bracket_shift")


def matrix(obj, name, problems):
    try:
        T = np.array(obj, dtype=float)
    except (TypeError, ValueError):
        problems.append("%s is not a 4x4 numeric matrix" % name)
        return
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        problems.append("%s is not a finite 4x4 matrix" % name)
        return
    if np.max(np.abs(T[3] - np.array([0.0, 0.0, 0.0, 1.0]))) > 1e-6:
        problems.append("%s: last row must be [0, 0, 0, 1]" % name)
    R = T[:3, :3]
    if np.max(np.abs(R.T @ R - np.eye(3))) > 1e-3 or np.linalg.det(R) < 0.0:
        problems.append("%s: rotation block is not a rotation (orthonormal, det +1, to 1e-3)" % name)


def number(obj, name, problems):
    if isinstance(obj, bool) or not isinstance(obj, (int, float)) or not math.isfinite(float(obj)):
        problems.append("%s is not a finite number" % name)
        return False
    return True


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out, problems = sys.argv[1], []
    try:
        with open(os.path.join(out, "calibration.yaml")) as fh:
            doc = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        doc = None
        problems.append("calibration.yaml: %s" % str(exc).splitlines()[0])
    if doc is not None and not isinstance(doc, dict):
        problems.append("calibration.yaml is not a mapping")
        doc = None
    if doc is not None:
        for key in ("T_lidar_camera", "T_lidar_imu"):
            matrix(doc.get(key), key, problems)
        cam = doc.get("camera") if isinstance(doc.get("camera"), dict) else {}
        for key in ("fx", "fy", "cx", "cy", "readout_s"):
            number(cam.get(key), "camera." + key, problems)
        d = cam.get("distortion")
        if not isinstance(d, list) or len(d) != 5 or not all(number(x, "camera.distortion[%d]" % i, problems) for i, x in enumerate(d)):
            problems.append("camera.distortion is not a list of 5 numbers [k1, k2, p1, p2, k3]")
        if number(cam.get("readout_s"), "camera.readout_s", []) and not 0.0 <= cam["readout_s"] <= 0.1:
            problems.append("camera.readout_s must be between 0 and 0.1 s")
        time = doc.get("time") if isinstance(doc.get("time"), dict) else {}
        for sensor in ("camera", "lidar"):
            e = time.get(sensor) if isinstance(time.get(sensor), dict) else {}
            number(e.get("offset_s"), "time.%s.offset_s" % sensor, problems)
            number(e.get("drift_ppm"), "time.%s.drift_ppm" % sensor, problems)
        imu = doc.get("imu") if isinstance(doc.get("imu"), dict) else {}
        for key in ("gyro_bias_rps", "accel_bias_mps2"):
            v = imu.get(key)
            if not isinstance(v, list) or len(v) != 3 or not all(number(x, "imu.%s[%d]" % (key, i), problems) for i, x in enumerate(v)):
                problems.append("imu.%s is not a list of 3 numbers" % key)
        ch = doc.get("changes")
        if ch is not None:
            if not isinstance(ch, list) or len(ch) > 1:
                problems.append("changes must be a list of at most one entry")
            else:
                for c in ch:
                    if not isinstance(c, dict) or c.get("type") != "camera_bracket_shift":
                        problems.append("changes[0].type must be camera_bracket_shift")
                    else:
                        matrix(c.get("T_lidar_camera_before"), "changes[0].T_lidar_camera_before", problems)
    try:
        with open(os.path.join(out, "fault_report.json")) as fh:
            rep = json.load(fh)
    except (OSError, ValueError) as exc:
        rep = None
        problems.append("fault_report.json: %s" % exc)
    if rep is not None:
        if not isinstance(rep, dict) or not isinstance(rep.get("faults"), list):
            problems.append("fault_report.json must be an object with a 'faults' list")
        elif len(rep["faults"]) > 50:
            problems.append("fault_report.json lists more than 50 faults")
        else:
            for i, f in enumerate(rep["faults"]):
                if not isinstance(f, dict) or not isinstance(f.get("sequence"), str) or f.get("sensor") not in SENSORS or f.get("type") not in TYPES:
                    problems.append("faults[%d] needs a sequence name, a sensor in %s and a type in %s" % (i, "/".join(SENSORS), "/".join(TYPES)))
                    continue
                if number(f.get("start_s"), "faults[%d].start_s" % i, problems) and number(f.get("end_s"), "faults[%d].end_s" % i, problems) and f["end_s"] < f["start_s"]:
                    problems.append("faults[%d] ends before it starts" % i)
    if problems:
        print("NOT well-formed:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print("well-formed (format only; nothing here says whether the values are right)")


if __name__ == "__main__":
    main()
