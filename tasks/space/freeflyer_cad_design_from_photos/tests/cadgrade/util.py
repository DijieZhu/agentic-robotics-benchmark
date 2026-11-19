"""Small helpers of the grader: the bands, rigid transforms in a fixed arithmetic order, the error kinds.

Rigid transforms are applied elementwise (x' = ((r00 x + r01 y) + r02 z) + t0), never through a matrix product, so a
point set maps to the same bits on every CPU (the reader of the maker files does the same).
"""
import hashlib
import math
import os

import numpy as np


class TaskError(Exception):
    """The grader's own inputs (the standard, its pins, the seen tables, the configuration) are unusable, or the
    extraction is not one the reader wrote: a broken task, never a score of 0."""


def item_error(stats, diags, what, exc):
    """A grading item that raised: it scores 0; counted (grader_item_errors) and reported."""
    stats["grader_item_errors"] = stats.get("grader_item_errors", 0) + 1
    if diags is not None and sum(1 for d in diags if d.startswith("Item error:")) < 20:
        diags.append("Item error: %s could not be evaluated (%s: %s); it scores 0" % (
            what, type(exc).__name__, str(exc)[:160]))


# ---------------------------------------------------------------------------------------------------- bands
def band(value, b):
    """1 when |value| <= b[0], 0 when |value| >= b[1], linear between; non-finite -> 0."""
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    full, zero = float(b[0]), float(b[1])
    if v <= full:
        return 1.0
    if v >= zero:
        return 0.0
    return (zero - v) / (zero - full)


def rising(value, b):
    """0 at value <= b[0], 1 at value >= b[1], linear between; non-finite -> 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    lo, hi = float(b[0]), float(b[1])
    return float(min(1.0, max(0.0, (v - lo) / (hi - lo))))


def finite(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


# ---------------------------------------------------------------------------------------------------- transforms
def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(t, float)
    return T


def apply(T, P):
    """Points P (n, 3) or (3,) mapped by the 4 x 4 transform T, elementwise in a fixed order."""
    T = np.asarray(T, float)
    P = np.asarray(P, float)
    one = P.ndim == 1
    Q = P.reshape(1, 3) if one else P
    out = np.empty(Q.shape, float)
    x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
    for i in range(3):
        out[:, i] = ((T[i, 0] * x + T[i, 1] * y) + T[i, 2] * z) + T[i, 3]
    return out[0] if one else out


def rotate(T, V):
    """Directions V (n, 3) or (3,) turned by the rotation part of T, elementwise."""
    T = np.asarray(T, float)
    V = np.asarray(V, float)
    one = V.ndim == 1
    Q = V.reshape(1, 3) if one else V
    out = np.empty(Q.shape, float)
    x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
    for i in range(3):
        out[:, i] = (T[i, 0] * x + T[i, 1] * y) + T[i, 2] * z
    return out[0] if one else out


def compose(A, B):
    """A . B for 4 x 4 transforms, in Python floats with a fixed summation order."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    out = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            out[i, j] = ((A[i, 0] * B[0, j] + A[i, 1] * B[1, j]) + A[i, 2] * B[2, j]) + A[i, 3] * B[3, j]
    return out


def inv(T):
    """The inverse of a rigid transform."""
    T = np.asarray(T, float)
    R = T[:3, :3].T.copy()
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = -np.array([(R[i, 0] * t[0] + R[i, 1] * t[1]) + R[i, 2] * t[2] for i in range(3)])
    return out


def orthonormal(R):
    """The proper rotation nearest R (SVD; a reflection is turned into a rotation)."""
    U, _s, Vt = np.linalg.svd(np.asarray(R, float)[:3, :3])
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def rot_angle_deg(R):
    R = np.asarray(R, float)[:3, :3]
    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return math.degrees(math.atan2(float(np.linalg.norm(v)), 0.5 * (float(np.trace(R)) - 1.0)))


def axis_angle_R(axis, ang):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def unit(v):
    v = np.asarray(v, float)
    n = math.sqrt(float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
    return v / n if n > 0 else v


def point_line_distance(p, a, d):
    """Distance of point p from the line through a with direction d."""
    d = unit(d)
    w = np.asarray(p, float) - np.asarray(a, float)
    along = float(w[0] * d[0] + w[1] * d[1] + w[2] * d[2])
    r = w - along * d
    return math.sqrt(float(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]))


def line_angle_deg(d1, d2):
    """Angle between two undirected lines, 0 to 90 degrees."""
    a, b = unit(d1), unit(d2)
    c = abs(float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    return math.degrees(math.acos(min(1.0, c)))


# ---------------------------------------------------------------------------------------------------- misc
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def limit_cpus(n):
    """Pin this process to n CPUs of its affinity set: OCC's mesher and manifold3d's thread pool size themselves to the
    host, whatever OMP_NUM_THREADS says."""
    try:
        cpus = sorted(os.sched_getaffinity(0))[:max(1, int(n))]
        os.sched_setaffinity(0, cpus)
        return cpus
    except (AttributeError, OSError, ValueError):
        return []
