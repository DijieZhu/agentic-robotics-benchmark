"""Numerics fingerprint of a mission program's arithmetic (verify / maintainers only; verify.py runs it before the
first mission, as the runner account, in the program's own environment).

    python3 -I numerics_check.py --math       # prints one JSON line: {part: hash, ...}

A program's results must not depend on the grading machine's CPU: NumPy picks SIMD loops and OpenBLAS its kernels by
CPU, a threaded product is split by the core count, and OpenCV's IPP picks its code by CPU. The run regime pins all of
it (setup_payload/training/submission_checks.py `numerics_pin`). This file hashes a fixed set of NumPy, SciPy and
OpenCV floating-point results on data built from integers; verify.py compares the hashes with tests/numerics.json
(written on the reference machine) and refuses to grade (a task error, never the agent's zero) on a machine that
computes any part differently. The report names the part that differs.

    python3 -I numerics_check.py --write numerics.json     # a maintainer, under the pin, after a change of the image
"""

import hashlib
import json
import sys

sys.dont_write_bytecode = True


def math_parts():
    import numpy as np
    out = {}

    def put(name, *xs):
        h = hashlib.sha256()
        for x in xs:
            a = np.ascontiguousarray(x)
            h.update(str(a.dtype).encode())
            h.update(a.tobytes())
        out[name] = h.hexdigest()[:16]
    i = np.arange(1, 4097, dtype=np.int64)
    x = ((i * 2654435761) % 1000003).astype(np.float64) / 1000003.0
    y = ((i * 40503 + 977) % 999983).astype(np.float64) / 999983.0
    s = 6.0 * x - 3.0
    t = 6.0 * y - 3.0
    for dt in (np.float64, np.float32):
        a, b, u, v = s.astype(dt), t.astype(dt), x.astype(dt), y.astype(dt)
        tag = np.dtype(dt).name
        put("trig_" + tag, np.sin(a), np.cos(a), np.tan(a), np.arctan(a), np.arctan2(a, b), np.arcsin(2 * u - 1), np.arccos(2 * v - 1))
        put("exp_" + tag, np.exp(a), np.expm1(a), np.log(u + dt(1e-3)), np.log1p(u), np.log2(u + dt(1e-3)), np.log10(u + dt(1e-3)),
            np.power(u + dt(0.1), b), np.tanh(a), np.sinh(a), np.cosh(a), np.cbrt(a), np.sqrt(u), np.hypot(a, b))
        for n in (7, 64, 781, 4096):
            put("red_%s_%d" % (tag, n), a[:n].sum(), a[:n].mean(), a[:n].std(), np.dot(a[:n], b[:n]), np.linalg.norm(a[:n]),
                np.cumsum(a[:n]), np.median(a[:n]))
    for n in (3, 40, 781, 2500):
        p = np.column_stack([s[:n], t[:n]]) * 0.06 + np.array([0.47, -0.13])
        A = np.column_stack([2 * p[:, 0], 2 * p[:, 1], np.ones(n)])
        bb = (p ** 2).sum(1)
        mu = p.mean(0)
        C = np.cov((p - mu).T) if n > 3 else np.eye(2)
        K = np.array([[415.7, 0, 320.0], [0, 415.7, 240.0], [0, 0, 1.0]])
        d = np.linalg.solve(K, np.column_stack([p * 500, np.ones(n)]).T)
        put("lin_%d" % n, np.linalg.lstsq(A, bb, rcond=None)[0] if n > 3 else 0.0, d, K @ d, np.linalg.eigh(C)[0],
            np.abs(np.linalg.eigh(C)[1]), (p - mu) @ np.array([0.6, 0.8]), np.linalg.norm(p - mu, axis=1),
            np.linalg.svd(A, compute_uv=False), np.linalg.pinv(A[:50]) if n > 3 else 0.0)
    M = np.outer(s[:400], t[:400]) + np.eye(400) * 5
    put("big", M @ M, np.linalg.inv(M[:64, :64] + np.eye(64) * 50), np.linalg.solve(M[:300, :300] + np.eye(300) * 60, t[:300]),
        M.astype(np.float32) @ M.astype(np.float32))
    # an occupancy-grid and point-cloud flavour: what a mapping program does with lidar points
    ang = x * 6.283185307179586
    rng_ = 0.3 + 24.0 * y
    pts = np.column_stack([rng_ * np.cos(ang), rng_ * np.sin(ang), 0.2 * s]).astype(np.float32)
    Rz = np.array([[np.cos(0.37), -np.sin(0.37), 0.0], [np.sin(0.37), np.cos(0.37), 0.0], [0.0, 0.0, 1.0]])
    put("cloud", pts @ Rz.T.astype(np.float32), (pts.astype(np.float64) @ Rz.T + np.array([1.5, -2.25, 0.1])),
        np.floor((pts[:, :2] + 30.0) / 0.05).astype(np.int32), np.histogram2d(pts[:, 0], pts[:, 1], bins=64)[0],
        np.arctan2(pts[:, 1], pts[:, 0]), np.linalg.norm(pts, axis=1))
    try:
        import scipy.linalg as sl
        import scipy.ndimage as nd
        import scipy.optimize as so
        from scipy.spatial import cKDTree
        p = np.column_stack([s[:781], t[:781]])
        A = np.column_stack([p, np.ones(781)])
        r = so.least_squares(lambda q: np.hypot(p[:, 0] - q[0], p[:, 1] - q[1]) - q[2], [0.1, 0.1, 1.0])
        dd, ii = cKDTree(p).query(p[:200] + 0.01, k=3)
        put("scipy", sl.lstsq(A, s[781:1562])[0], sl.eigh(A.T @ A)[0], sl.svd(A, compute_uv=False), r.x,
            nd.gaussian_filter(M[:64, :64], 1.5), sl.expm(M[:6, :6] / 50), dd, ii.astype(np.int64),
            nd.distance_transform_edt(M[:64, :64] > 5.0), nd.binary_dilation(M[:64, :64] > 5.0, iterations=2))
    except ImportError:
        out["scipy"] = "absent"
    try:
        import cv2
        im = ((np.add.outer(np.arange(240) * 7, np.arange(320) * 13)[..., None] * np.array([3, 5, 11])) % 256).astype(np.uint8)
        f = im.astype(np.float32) / 255.0
        pt = (np.column_stack([np.cos(x[:200] * 6.283) * 80 + 160 + y[:200], np.sin(x[:200] * 6.283) * 50 + 120])).astype(np.float32)
        Mx = cv2.getRotationMatrix2D((160, 120), 17.0, 1.1)
        (cx, cy), rr = cv2.minEnclosingCircle(pt)
        el = cv2.fitEllipse(pt)
        put("cv2", cv2.cvtColor(im, cv2.COLOR_RGB2HSV), cv2.cvtColor(f, cv2.COLOR_RGB2HSV), cv2.cvtColor(im, cv2.COLOR_RGB2LAB),
            cv2.GaussianBlur(im, (7, 7), 1.7), cv2.GaussianBlur(f, (0, 0), 2.3),
            cv2.resize(im, (213, 161), interpolation=cv2.INTER_LINEAR), cv2.resize(f, (640, 480), interpolation=cv2.INTER_CUBIC),
            cv2.warpAffine(im, Mx, (320, 240)), cv2.warpAffine(f, Mx, (320, 240)), cv2.Canny(im[..., 0], 50, 150),
            cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.medianBlur(im, 5), cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
            np.array([cx, cy, rr]), np.array(el[0] + el[1] + (el[2],)), cv2.moments(im[..., 0])["mu20"],
            cv2.matchTemplate(im[..., 1], im[40:60, 50:80, 1], cv2.TM_CCOEFF_NORMED),
            cv2.solve(M[:9, :9].copy() + np.eye(9) * 20, t[:9].reshape(9, 1).copy())[1], cv2.Rodrigues(np.array([0.3, -0.2, 0.9]))[0],
            cv2.connectedComponentsWithStats((im[..., 0] > 128).astype(np.uint8))[2].astype(np.int64),
            cv2.imdecode(cv2.imencode(".png", im)[1], cv2.IMREAD_COLOR))
    except ImportError:
        out["cv2"] = "absent"
    return out


def main():
    import warnings
    warnings.simplefilter("ignore")
    parts = math_parts()
    if len(sys.argv) == 3 and sys.argv[1] == "--write":
        import numpy
        with open(sys.argv[2], "w") as fh:
            json.dump({"math": parts, "numpy": numpy.__version__,
                       "what": "hashes of NumPy, SciPy and OpenCV results from a Python started as a mission program is "
                               "(tests/numerics_check.py --math under the run regime's numerics pin); verify.py refuses to "
                               "grade on a machine that computes a part differently"}, fh, indent=1, sort_keys=True)
            fh.write("\n")
    print(json.dumps(parts, sort_keys=True))


if __name__ == "__main__":
    main()
