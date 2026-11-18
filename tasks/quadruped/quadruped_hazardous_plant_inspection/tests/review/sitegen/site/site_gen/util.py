"""Small shared helpers: seeded random streams, rotations, rectangles, JSON rounding."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable, Sequence

import numpy as np


# ----------------------------------------------------------------------------------------------- random streams
class Rng:
    """Named, independent random streams derived from (seed, attempt, name): adding draws to one stream never
    shifts another, which keeps sites stable while the generator evolves."""

    def __init__(self, seed: int, attempt: int = 0):
        self.seed = int(seed)
        self.attempt = int(attempt)
        self._streams: dict[str, np.random.Generator] = {}

    def __call__(self, name: str) -> np.random.Generator:
        g = self._streams.get(name)
        if g is None:
            h = hashlib.sha256(("%d/%d/%s" % (self.seed, self.attempt, name)).encode()).digest()
            g = np.random.Generator(np.random.PCG64(int.from_bytes(h[:16], "little")))
            self._streams[name] = g
        return g


def choice(g: np.random.Generator, seq: Sequence, p=None):
    if p is not None:
        p = np.asarray(p, float)
        p = p / p.sum()
    i = int(g.choice(len(seq), p=p))
    return seq[i]


def weighted_key(g: np.random.Generator, weights: dict):
    keys = sorted(weights.keys())
    return choice(g, keys, [float(weights[k]) for k in keys])


def uniform(g: np.random.Generator, lo: float, hi: float) -> float:
    return float(g.uniform(lo, hi))


def lattice_uniform(g: np.random.Generator, lo: float, hi: float, step: float = 0.5) -> float:
    n0 = int(math.ceil(lo / step - 1e-9))
    n1 = int(math.floor(hi / step + 1e-9))
    if n1 < n0:
        return round(lo / step) * step
    return float(g.integers(n0, n1 + 1)) * step


# ----------------------------------------------------------------------------------------------- rotations
def quat_axis_angle(axis, angle) -> list:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2.0)
    return [math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s]


def quat_mul(a, b) -> list:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def quat_yaw(yaw: float) -> list:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def mat_from_quat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_from_mat(R) -> list:
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q, float)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def quat_z_to(direction) -> list:
    """Quaternion turning local +z onto `direction` (for cylinders / capsules)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c > 1 - 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    if c < -1 + 1e-12:
        return [0.0, 1.0, 0.0, 0.0]
    ax = np.cross(z, d)
    ax /= np.linalg.norm(ax)
    return quat_axis_angle(ax, math.acos(c))


def face_quat(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> list:
    """Face frame (bricks convention: +x out of the readable face toward the viewer, +z up) with its normal along
    azimuth `yaw` (rad, ccw from +x) tilted up by `pitch` (rad)."""
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    cp, sp = math.cos(-pitch), math.sin(-pitch)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    cr, sr = math.cos(roll), math.sin(roll)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return quat_from_mat(Rz @ Ry @ Rx)


# ----------------------------------------------------------------------------------------------- rectangles
def rect_w(r) -> float:
    return r[2] - r[0]


def rect_h(r) -> float:
    return r[3] - r[1]


def rect_center(r) -> tuple:
    return (0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3]))


def rect_overlap(a, b, pad: float = 0.0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def rect_contains(r, x, y, pad: float = 0.0) -> bool:
    return r[0] - pad <= x <= r[2] + pad and r[1] - pad <= y <= r[3] + pad


def shared_edge(a, b, tol: float = 1e-6):
    """The common boundary segment of two touching axis-aligned rectangles: (axis, coord, lo, hi) where axis 'x'
    means a vertical edge at x = coord spanning y in [lo, hi]; None when they do not share an edge."""
    if abs(a[2] - b[0]) < tol or abs(b[2] - a[0]) < tol:
        x = a[2] if abs(a[2] - b[0]) < tol else a[0]
        lo, hi = max(a[1], b[1]), min(a[3], b[3])
        if hi - lo > tol:
            return ("x", x, lo, hi)
    if abs(a[3] - b[1]) < tol or abs(b[3] - a[1]) < tol:
        y = a[3] if abs(a[3] - b[1]) < tol else a[1]
        lo, hi = max(a[0], b[0]), min(a[2], b[2])
        if hi - lo > tol:
            return ("y", y, lo, hi)
    return None


def poly_rect(r) -> list:
    return [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]


# ----------------------------------------------------------------------------------------------- JSON
def rnd(x, nd: int = 4):
    """Round floats recursively (stable, byte-identical JSON across runs)."""
    if isinstance(x, float):
        v = round(x, nd)
        return 0.0 if v == 0 else v
    if isinstance(x, (np.floating,)):
        return rnd(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), nd)
    if isinstance(x, dict):
        return {str(k): rnd(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, nd) for v in x]
    return x


def dump_json(obj, path: str, nd: int = 4):
    with open(path, "w") as f:
        json.dump(rnd(obj, nd), f, indent=1, sort_keys=False)
        f.write("\n")


def runs_from_mask(mask: np.ndarray) -> list:
    """Row run-length encoding of a boolean grid: [[iy, ix0, ix1], ...] with ix1 inclusive."""
    out = []
    for iy in range(mask.shape[0]):
        row = mask[iy]
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts = np.nonzero(d == 1)[0]
        ends = np.nonzero(d == -1)[0] - 1
        for s, e in zip(starts, ends):
            out.append([int(iy), int(s), int(e)])
    return out


def mask_from_runs(runs: Iterable, shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    for iy, a, b in runs:
        m[iy, a:b + 1] = True
    return m


def angle_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
