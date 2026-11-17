#!/usr/bin/env python3
"""Check that matrix products on /opt/grasp come out right in this process's environment.

    /opt/grasp/bin/python -I numerics_check.py

verify.py runs it as root before the hidden frames are fetched, once in the environment of the solver processes and
once in the scoring's. It never imports submitted code. It imports what those processes import (numpy, and
graspnetAPI.utils.eval_utils, which pulls in scipy, cv2 and cvxopt), then:

  * asks every OpenBLAS copy mapped into the process which kernel family it selected. numpy 1.23.5 bundles OpenBLAS
    0.3.20, which picks its kernels by CPU when it loads. On Intel CPUs with AVX512_BF16 (Cooper Lake, Sapphire Rapids
    and later) it picks its Cooperlake kernels, and with those DGEMM returns wrong products (errors of order 1 to 10,
    not rounding), the one in graspnetAPI's transform_points among them. With OPENBLAS_CORETYPE=Haswell every copy
    uses its Haswell kernels, which are correct and run on any x86-64 CPU with AVX2 and FMA. On x86-64 every copy must
    report Haswell; on other architectures the family is reported but not required;
  * compares BLAS-backed products with a BLAS-free evaluation of the same sums (np.einsum) on fixed-seed random data,
    in the layouts the evaluation uses, for 1 to 100000 points. A relative error above MAX_REL_ERROR fails, on every
    architecture. The wrong kernels get some sizes right (and with several threads some sizes that fail on one
    thread pass), so many sizes are tried.

Prints exactly one JSON line on stdout, {"machine", "cores", "max_rel_error", "ok", "problems"}, and exits 0 when ok,
1 otherwise; whatever the libraries print goes to stderr. Importing this module needs only the standard library, so
that decide() can be tested anywhere.
"""
import ctypes
import json
import math
import os
import platform
import re
import sys

REQUIRED_CORE = "Haswell"
MAX_REL_ERROR = 1e-9
X86_64 = ("x86_64", "amd64")
# Point counts, from one point to more than a scene's table (93750 points: create_table_points at 0.008 m, as the
# evaluation builds it). Measured with the Cooperlake kernels of OpenBLAS 0.3.20, the transposed product was wrong
# from N = 193 up on one thread, and on many threads only up to about 10000, where the work is split.
SIZES = (1, 2, 3, 4, 7, 16, 64, 100, 127, 128, 129, 191, 192, 193, 254, 255, 256, 257, 300, 400, 512, 1000, 1024,
         2000, 4096, 10000, 20000, 31250, 50000, 93750, 100000)
# The stacked product of collision_detection, (G, N, 3) @ (G, 3, 3): G grasps, at most STACK_POINTS points in all.
STACK_GRASPS, STACK_POINTS = 8, 800000
# The core-name query of an OpenBLAS build, by the symbol prefix and suffix the build gives its exports.
CORENAME_SYMBOLS = ("openblas_get_corename", "openblas_get_corename64_", "scipy_openblas_get_corename",
                    "scipy_openblas_get_corename64_", "scipy_openblas_get_corename64")
SEED = 20261114


def decide(machine, cores, errors):
    """The problems of one run (an empty list: its matrix products can be trusted). Pure: no numpy, no I/O.

    machine  platform.machine()
    cores    {OpenBLAS copy: the kernel family it reports, or None when it could not be asked}
    errors   {layout: {point count: largest relative error}}; inf or NaN where a product was not finite
    """
    problems = []
    if str(machine).lower() in X86_64:
        if not cores:
            problems.append("no OpenBLAS copy found in the process")
        for name in sorted(cores):
            core = cores[name]
            if core is None:
                problems.append("%s: its kernel family could not be queried" % name)
            elif str(core).lower() != REQUIRED_CORE.lower():
                problems.append("%s uses its %s kernels, not its %s ones" % (name, core, REQUIRED_CORE))
    if not any(errors.values()):
        problems.append("the matrix-product self-test did not run")
    for layout in sorted(errors):
        bad = sorted(n for n, err in errors[layout].items() if not err <= MAX_REL_ERROR)    # NaN fails too
        if bad:
            worst = max(errors[layout][n] for n in bad)
            problems.append("%s: relative error up to %.3g (limit %g) for N = %s"
                            % (layout, worst, MAX_REL_ERROR, ", ".join(str(n) for n in bad)))
    return problems


def openblas_cores(maps="/proc/self/maps"):
    """{copy: kernel family} for every OpenBLAS shared object mapped into this process; None where it cannot be asked.
    A copy is named by its folder and file name (numpy.libs/libopenblas64_p-r0-742d56dc.3.20.so)."""
    paths = set()
    with open(maps) as handle:
        for line in handle:
            fields = line.split(None, 5)
            if len(fields) == 6:
                path = fields[5].strip()
                if path.startswith("/") and re.search(r"openblas.*\.so", os.path.basename(path)):
                    paths.add(path)
    cores = {}
    for path in sorted(paths):
        name = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
        cores[name] = _corename(path)
    return cores


def _corename(path):
    try:
        lib = ctypes.CDLL(path)                      # already loaded: the same handle, nothing is initialised again
    except OSError:
        return None
    for symbol in CORENAME_SYMBOLS:
        fn = getattr(lib, symbol, None)
        if fn is not None:
            fn.restype, fn.argtypes = ctypes.c_char_p, []
            name = fn()
            return name.decode(errors="replace").strip() if name else None
    return None


def load_stack():
    """Import what a solver or scoring process imports; returns numpy and graspnetAPI's transform_points."""
    import warnings
    warnings.filterwarnings("ignore")
    import numpy
    from graspnetAPI.utils.eval_utils import transform_points
    import cv2, cvxopt.blas, scipy.linalg                                 # noqa: E401,F401 (eval_utils loads them)
    return numpy, transform_points


def _rel(np, got, ref):
    got = np.asarray(got, dtype=np.float64)
    if got.shape != ref.shape or not np.all(np.isfinite(got)):
        return float("inf")
    return float(np.abs(got - ref).max()) / max(float(np.abs(ref).max()), 1e-300)


def _rotations(np, rng, count):
    """`count` random rotation matrices, from unit quaternions, with elementwise arithmetic only (no BLAS)."""
    q = rng.normal(size=(count, 4))
    q = q / np.sqrt((q * q).sum(axis=1, keepdims=True))
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1).reshape(count, 3, 3)


def self_test(np, transform_points):
    """{layout: {N: largest relative error}} of the BLAS-backed products against np.einsum (its own loops, no BLAS)."""
    rng = np.random.default_rng(SEED)
    errors = {"transform_points": {}, "matmul(T, X.T)": {}, "matmul(T, contiguous X.T)": {},
              "matmul((G,N,3), (G,3,3))": {}}
    for n in SIZES:
        # graspnetAPI's transform_points: points (N, 3), a rigid 4x4 transform
        trans = np.eye(4)
        trans[:3, :3] = _rotations(np, rng, 1)[0]
        trans[:3, 3] = rng.uniform(-1.0, 1.0, size=3)
        points = rng.uniform(-0.5, 0.5, size=(n, 3))
        homogeneous = np.concatenate([points, np.ones((n, 1))], axis=-1)
        ref = np.einsum("ij,nj->ni", trans[:3], homogeneous)
        errors["transform_points"][n] = _rel(np, transform_points(points, trans), ref)
        # the product inside it, T (4, 4) @ X.T with X (N, 4): a transposed view, then a contiguous copy
        t4, x4 = rng.normal(size=(4, 4)), rng.normal(size=(n, 4))
        ref = np.einsum("ij,nj->in", t4, x4)
        errors["matmul(T, X.T)"][n] = _rel(np, np.matmul(t4, x4.T), ref)
        errors["matmul(T, contiguous X.T)"][n] = _rel(np, np.matmul(t4, np.ascontiguousarray(x4.T)), ref)
        # collision_detection: scene points in each grasp's frame, (G, N, 3) @ (G, 3, 3)
        g = max(1, min(STACK_GRASPS, STACK_POINTS // n))
        target, rot = rng.uniform(-0.2, 0.2, size=(g, n, 3)), _rotations(np, rng, g)
        ref = np.einsum("gnk,gkj->gnj", target, rot)
        errors["matmul((G,N,3), (G,3,3))"][n] = _rel(np, np.matmul(target, rot), ref)
    return errors


def main():
    report_fd = os.dup(1)
    os.dup2(2, 1)                       # stdout carries the report only; anything else printed goes to stderr
    machine, cores, errors, problems = platform.machine(), {}, {}, []
    try:
        np, transform_points = load_stack()
    except Exception as exc:
        problems.append("import failed: %s: %s" % (type(exc).__name__, str(exc)[:200]))
    else:
        try:
            errors = self_test(np, transform_points)
        except Exception as exc:
            problems.append("self-test failed: %s: %s" % (type(exc).__name__, str(exc)[:200]))
    try:
        cores = openblas_cores()
    except OSError as exc:
        problems.append("could not read /proc/self/maps: %s" % exc)
    problems += decide(machine, cores, errors)
    worst = [e for layout in errors.values() for e in layout.values()]
    max_err = None
    if worst:
        max_err = float("inf") if any(math.isnan(e) for e in worst) else max(worst)
    report = {"machine": machine, "cores": cores, "max_rel_error": max_err, "ok": not problems,
              "problems": problems}
    os.write(report_fd, (json.dumps(report) + "\n").encode())
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
