"""cio: boilerplate I/O for cloth_material_id_from_video (no perception, tracking or identification code).

    import cio
    cams = cio.cameras()                         # list of dicts: name, K, R_w2c, t_w2c, width, height
    fab = cio.fabric_info("A")                   # size_m, grid, mass_kg, n_clips
    clip = cio.clip_info("A", 0)                 # kind, fps, n_frames, clamp_500hz (M, 8), arm_q (N, 7); no start pose
    frames = cio.load_video("A", 0, "cam1")      # (N, H, W, 3) uint8 RGB
    uv = cio.project(cams[0], X_world)           # (n, 2) pixels of world points (n, 3)
    cio.write_params({"A": {...}, ..., "J": {...}})
    cio.check_params()                           # the format check of ~/submission/params.json (python3 cio.py runs it)
"""
import json
import math
import os

import numpy as np

DATA = os.path.join(os.path.expanduser("~"), "data")
SUBMISSION = os.path.join(os.path.expanduser("~"), "submission", "params.json")
KEYS = ("young", "poisson", "thickness", "damping", "mu")


def cameras():
    with open(os.path.join(DATA, "cameras.json")) as f:
        cams = json.load(f)["cameras"]
    for c in cams:
        c["K"] = np.array(c["K"])
        c["R_w2c"] = np.array(c["R_w2c"])
        c["t_w2c"] = np.array(c["t_w2c"])
    return cams


def fabric_dir(fid):
    return os.path.join(DATA, "fabrics", fid)


def fabric_info(fid):
    with open(os.path.join(fabric_dir(fid), "fabric.json")) as f:
        return json.load(f)


def clip_info(fid, i):
    d = os.path.join(fabric_dir(fid), "clip%02d" % i)
    with open(os.path.join(d, "clip.json")) as f:
        info = json.load(f)
    info["clamp_500hz"] = np.load(os.path.join(d, "clamp_500hz.npy"), allow_pickle=False)
    info["arm_q"] = np.load(os.path.join(d, "arm_q.npy"), allow_pickle=False)
    return info


def load_video(fid, i, cam):
    import cv2
    cap = cv2.VideoCapture(os.path.join(fabric_dir(fid), "clip%02d" % i, cam + ".mp4"))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f[:, :, ::-1])
    cap.release()
    return np.stack(out)


def project(cam, X):
    X = np.atleast_2d(np.asarray(X, float))
    Xc = X @ cam["R_w2c"].T + cam["t_w2c"]
    uv = Xc @ cam["K"].T
    return uv[:, :2] / uv[:, 2:3]


def write_params(fabrics):
    """fabrics: {"A": {"young", "poisson", "thickness", "damping", "mu"}, ...} -> ~/submission/params.json.
    Checks that every value is a finite number (the grader clamps each to its admissible range)."""
    out = {}
    for k, e in fabrics.items():
        row = {}
        for key in KEYS:
            v = float(e[key])
            if not math.isfinite(v):
                raise ValueError("%s.%s is not finite" % (k, key))
            row[key] = v
        out[k] = row
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    with open(SUBMISSION, "w") as f:
        json.dump({"fabrics": out}, f, indent=1)
    return SUBMISSION


FABRICS = tuple("ABCDEFGHIJ")


def check_params(path=SUBMISSION):
    """Format check of the submission, by the grader's own rules (it says nothing about how good the materials are).
    Returns a list of problems; an empty list means the file will be read and every fabric scored."""
    import cphys
    problems = []
    try:
        if os.path.islink(path) or not os.path.isfile(path):
            return ["%s is not a regular file" % path]
        if os.path.getsize(path) > 64 * 1024:
            return ["larger than 64 kB"]
        with open(path) as f:
            doc = json.load(f, parse_constant=lambda name: (_ for _ in ()).throw(ValueError("non-finite number %s" % name)))
    except (OSError, ValueError) as exc:
        return ["not readable as JSON: %s" % exc]
    if not isinstance(doc, dict) or not isinstance(doc.get("fabrics"), dict):
        return ["the top level must be an object with a 'fabrics' object"]
    for k in FABRICS:
        e = doc["fabrics"].get(k)
        if not isinstance(e, dict):
            problems.append("fabric %s is missing (it scores 0)" % k)
            continue
        for key in KEYS:
            v = e.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                problems.append("fabric %s: %s must be a finite number (the fabric scores 0)" % (k, key))
            else:
                lo, hi = cphys.RANGES[key]
                if not lo <= float(v) <= hi:
                    problems.append("fabric %s: %s = %g is outside [%g, %g] and will be clamped" % (k, key, v, lo, hi))
    return problems


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    bad = check_params()
    print("\n".join(bad) if bad else "params.json: format OK (ten fabrics, five numbers each)")
