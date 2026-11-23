"""Numerics fingerprint of the simulation (verify / maintainers only; run by verify.py before anything is graded).

    /opt/venv/bin/python3 -I numerics_check.py        # prints one JSON line: {"fingerprint": sha256, ...}

The operational-space controller inside the simulation inverts two small matrices with LAPACK in every physics substep
(appliance_env.ApplianceEnv._osc_torque), and NumPy's OpenBLAS picks its kernels by CPU: on another kernel the
simulation differs from the first step (measured on this task: a fixed 400-step script on practice unit p0 left the
knob at 2.965647, 2.968151 or 2.965510 rad under the SkylakeX, Haswell or Sandybridge kernels, and the reference's
practice episodes ended in different states). The image pins the kernels (OPENBLAS_CORETYPE=Haswell with
OPENBLAS_L2_SIZE=256, environment/Dockerfile). This check runs STEPS control steps of one episode that is not graded
(the first unit of tests/units.json at a seed of its own, no image drawn) under a fixed script of actions that needs
no math library, and hashes every state.

A controller's own arithmetic must repeat too, and the kernels alone do not pin it (np.arctan2 and others round
differently where NumPy's AVX-512 loops are on; a threaded product sums in another order; OpenCV's IPP picks its code
by CPU): the image's .pth takes those out for every /opt/venv Python. So the check also hashes a set of NumPy, SciPy
and OpenCV results (math_parts), computed by a Python that is started as the grader starts a controller's run.sh:
with PATH, HOME and LANG only, so that the pins come from the image's .pth alone.

The fingerprint is the sha256 of both hashes. verify.py compares it with tests/numerics.json and refuses to grade (a
task error) on a machine that does not reproduce it; the report names the part that differs.

    /opt/venv/bin/python3 -I numerics_check.py --write numerics.json     # a maintainer, after a change of the simulation
"""

import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess  # noqa: E402

import numpy as np  # noqa: E402

CONTROLLER_ENV = {"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8"}
MATH_TIMEOUT_S = 120.0


def math_parts():
    """Hashes of NumPy, SciPy and OpenCV floating-point results on fixed data built from integers: the functions a
    controller's own arithmetic is made of (elementary functions in both float types, sums, linear algebra small and
    large, filters, fits). A part that differs between two machines names the library that computes differently."""
    out = {}
    def put(name, *xs):
        h = hashlib.sha256()
        for x in xs:
            a = np.ascontiguousarray(x)
            h.update(str(a.dtype).encode()); h.update(a.tobytes())
        out[name] = h.hexdigest()[:16]
    i = np.arange(1, 4097, dtype=np.int64)
    x = ((i * 2654435761) % 1000003).astype(np.float64) / 1000003.0          # (0, 1)
    y = ((i * 40503 + 977) % 999983).astype(np.float64) / 999983.0
    s = 6.0 * x - 3.0; t = 6.0 * y - 3.0
    for dt in (np.float64, np.float32):
        a, b, u, v = s.astype(dt), t.astype(dt), x.astype(dt), y.astype(dt)
        tag = np.dtype(dt).name
        put("trig_" + tag, np.sin(a), np.cos(a), np.tan(a), np.arctan(a), np.arctan2(a, b), np.arcsin(2 * u - 1), np.arccos(2 * v - 1))
        put("exp_" + tag, np.exp(a), np.expm1(a), np.log(u + dt(1e-3)), np.log1p(u), np.log2(u + dt(1e-3)), np.log10(u + dt(1e-3)), np.power(u + dt(0.1), b),
            np.tanh(a), np.sinh(a), np.cosh(a), np.cbrt(a), np.sqrt(u), np.hypot(a, b))
        for n in (7, 64, 781, 4096):
            put("red_%s_%d" % (tag, n), a[:n].sum(), a[:n].mean(), a[:n].std(), np.dot(a[:n], b[:n]), np.linalg.norm(a[:n]), np.cumsum(a[:n]), np.median(a[:n]))
    for n in (3, 40, 781, 2500):
        p = np.column_stack([s[:n], t[:n]]) * 0.06 + np.array([0.47, -0.13])
        A = np.column_stack([2 * p[:, 0], 2 * p[:, 1], np.ones(n)]); bb = (p ** 2).sum(1)
        mu = p.mean(0); C = np.cov((p - mu).T) if n > 3 else np.eye(2)
        K = np.array([[415.7, 0, 320.0], [0, 415.7, 240.0], [0, 0, 1.0]])
        d = np.linalg.solve(K, np.column_stack([p * 500, np.ones(n)]).T)
        put("lin_%d" % n, np.linalg.lstsq(A, bb, rcond=None)[0] if n > 3 else 0.0, d, K @ d, np.linalg.eigh(C)[0], np.abs(np.linalg.eigh(C)[1]),
            (p - mu) @ np.array([0.6, 0.8]), np.linalg.norm(p - mu, axis=1), np.linalg.svd(A, compute_uv=False), np.linalg.pinv(A[:50]) if n > 3 else 0.0)
    M = np.outer(s[:400], t[:400]) + np.eye(400) * 5
    put("big", M @ M, np.linalg.inv(M[:64, :64] + np.eye(64) * 50), np.linalg.solve(M[:300, :300] + np.eye(300) * 60, t[:300]), M.astype(np.float32) @ M.astype(np.float32))
    try:
        import scipy.linalg as sl, scipy.optimize as so, scipy.ndimage as nd
        p = np.column_stack([s[:781], t[:781]])
        A = np.column_stack([p, np.ones(781)])
        r = so.least_squares(lambda q: np.hypot(p[:, 0] - q[0], p[:, 1] - q[1]) - q[2], [0.1, 0.1, 1.0])
        put("scipy", sl.lstsq(A, s[781:1562])[0], sl.eigh(A.T @ A)[0], sl.svd(A, compute_uv=False), r.x, nd.gaussian_filter(M[:64, :64], 1.5), sl.expm(M[:6, :6] / 50))
    except ImportError:
        out["scipy"] = "absent"
    try:
        import cv2
        im = ((np.add.outer(np.arange(240) * 7, np.arange(320) * 13)[..., None] * np.array([3, 5, 11])) % 256).astype(np.uint8)
        f = im.astype(np.float32) / 255.0
        pts = (np.column_stack([np.cos(x[:200] * 6.283) * 80 + 160 + y[:200], np.sin(x[:200] * 6.283) * 50 + 120])).astype(np.float32)
        Mx = cv2.getRotationMatrix2D((160, 120), 17.0, 1.1)
        (cx, cy), rr = cv2.minEnclosingCircle(pts); el = cv2.fitEllipse(pts)
        put("cv2", cv2.cvtColor(im, cv2.COLOR_RGB2HSV), cv2.cvtColor(f, cv2.COLOR_RGB2HSV), cv2.cvtColor(im, cv2.COLOR_RGB2LAB), cv2.GaussianBlur(im, (7, 7), 1.7), cv2.GaussianBlur(f, (0, 0), 2.3),
            cv2.resize(im, (213, 161), interpolation=cv2.INTER_LINEAR), cv2.resize(f, (640, 480), interpolation=cv2.INTER_CUBIC), cv2.warpAffine(im, Mx, (320, 240)), cv2.warpAffine(f, Mx, (320, 240)),
            cv2.Canny(im[..., 0], 50, 150), cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.medianBlur(im, 5), cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
            np.array([cx, cy, rr]), np.array(el[0] + el[1] + (el[2],)), cv2.moments(im[..., 0])["mu20"], cv2.matchTemplate(im[..., 1], im[40:60, 50:80, 1], cv2.TM_CCOEFF_NORMED),
            cv2.solve(M[:9, :9].copy() + np.eye(9) * 20, t[:9].reshape(9, 1).copy())[1], cv2.Rodrigues(np.array([0.3, -0.2, 0.9]))[0],
            cv2.undistortPoints(pts.reshape(-1, 1, 2), np.array([[415.7, 0, 320.0], [0, 415.7, 240.0], [0, 0, 1.0]]), np.array([0.1, -0.05, 0.001, 0.002, 0.0])))
    except ImportError:
        out["cv2"] = "absent"
    return out


if __name__ == "__main__" and sys.argv[1:] == ["--math"]:
    import warnings
    warnings.simplefilter("ignore")
    print(json.dumps(math_parts(), sort_keys=True))
    sys.exit(0)

import appliance_env as E  # noqa: E402

SEED = 20261124
STEPS = 200


def script():
    """STEPS actions from integer arithmetic only: the hand moves, turns, opens and closes; no image is asked for."""
    out = []
    for i in range(STEPS):
        a = ((i * 7) % 11 - 5) / 400.0
        b = ((i * 5) % 13 - 6) / 500.0
        c = ((i * 3) % 7 - 3) / 60.0
        grip = -1.0 if i % 40 < 20 else 1.0
        if i < 80:
            out.append([0.004 + a, b, -0.003, c, 0.0, -c, grip, 0])
        elif i < 140:
            out.append([-b, a, 0.002, 0.0, c, c, -grip, 0])
        else:
            out.append([a, -a, b, c, -c, 0.0, grip, 0])
    return out


def fingerprint():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "units.json")) as fh:
        units = json.load(fh)
    uid = sorted(units)[0]
    c = units[uid]
    spec = E.unit_spec(c["model"], files=E.read_unit_files(os.path.join(here, c["dir"])), quirks=c.get("quirks"))
    env = E.ApplianceEnv(spec, None, render=False)
    env.reset(SEED)
    h = hashlib.sha256()
    d = env.data
    h.update(np.ascontiguousarray(d.qpos, dtype="<f8").tobytes())
    for action in script():
        env.step(action)
        h.update(np.ascontiguousarray(d.qpos, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(d.qvel, dtype="<f8").tobytes())
    if env.unstable():
        raise RuntimeError("the numerics script made the simulation unstable")
    env.close()
    return h.hexdigest()


def controller_math():
    """math_parts() from a fresh interpreter with a controller's environment (the grader starts run.sh with these
    three variables and nothing else)."""
    proc = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--math"], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, timeout=MATH_TIMEOUT_S, env=dict(CONTROLLER_ENV), check=True)
    return json.loads(proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1])


def combine(simulation, parts):
    h = hashlib.sha256()
    h.update(simulation.encode())
    h.update(json.dumps(parts, sort_keys=True).encode())
    return h.hexdigest()


def main():
    simulation, parts = fingerprint(), controller_math()
    report = {"fingerprint": combine(simulation, parts), "simulation": simulation, "math": parts, "steps": STEPS,
              "seed": SEED, "openblas_coretype": os.environ.get("OPENBLAS_CORETYPE"), "numpy": np.__version__}
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "numerics.json")
    if os.path.isfile(here):
        with open(here) as fh:
            ref = json.load(fh)
        report["differs"] = sorted((["simulation"] if ref.get("simulation") != simulation else [])
                                   + ["math." + k for k in sorted(set(parts) | set(ref.get("math", {})))
                                      if ref.get("math", {}).get(k) != parts.get(k)])
    if len(sys.argv) == 3 and sys.argv[1] == "--write":
        with open(sys.argv[2], "w") as fh:
            json.dump({"fingerprint": report["fingerprint"], "simulation": simulation, "math": parts, "steps": STEPS,
                       "seed": SEED,
                       "what": "fingerprint: sha256 of `simulation` (the sha256 of qpos and qvel after each of the "
                               "script's control steps, OPENBLAS_CORETYPE=Haswell, OPENBLAS_L2_SIZE=256) and of `math` (hashes of NumPy, "
                               "SciPy and OpenCV results from a Python started as a controller is: "
                               "tests/numerics_check.py)"}, fh, indent=1, sort_keys=True)
            fh.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
