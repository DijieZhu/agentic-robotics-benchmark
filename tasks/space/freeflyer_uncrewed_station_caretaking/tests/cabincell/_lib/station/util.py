"""Small geometry helpers of the station scene (quaternions, frames, MJCF number formatting)."""
from __future__ import annotations

import hashlib
import math

import numpy as np


def fmt(x, k: int = 7) -> str:
    """MJCF number list."""
    return " ".join(f"{float(v):.{k}g}" for v in np.ravel(x))


def quat_from_mat(R) -> list:
    """(w, x, y, z) unit quaternion of a rotation matrix, w >= 0 (copy of the station library's station_lib.quat_from_mat)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def frame_axes(normal, t1):
    """(n, a, b): n into the cabin, a = t1 projected into the wall plane, b = n x a."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    a = np.asarray(t1, float) - (np.asarray(t1, float) @ n) * n
    a /= np.linalg.norm(a)
    return n, a, np.cross(n, a)


def unit_axes(normal, up):
    """(n, a, b) of a stowage unit on a rack face: n into the cabin, b = ``up`` projected into the face plane, a = b x n
    (so that n x a = b)."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    b = np.asarray(up, float) - (np.asarray(up, float) @ n) * n
    b /= np.linalg.norm(b)
    a = np.cross(b, n)
    return n, a, b


def obb_corners(c, R, h) -> np.ndarray:
    import itertools
    return np.array([c + R @ (np.array(s) * h) for s in itertools.product((-1, 1), repeat=3)])


def obb_overlap(c1, R1, h1, c2, R2, h2) -> bool:
    """Separating-axis test of two oriented boxes."""
    axes = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)]
    axes += [np.cross(R1[:, i], R2[:, j]) for i in range(3) for j in range(3)]
    d = np.asarray(c2, float) - np.asarray(c1, float)
    for ax in axes:
        nrm = np.linalg.norm(ax)
        if nrm < 1e-9:
            continue
        ax = ax / nrm
        r1 = sum(h1[i] * abs(R1[:, i] @ ax) for i in range(3))
        r2 = sum(h2[i] * abs(R2[:, i] @ ax) for i in range(3))
        if abs(d @ ax) > r1 + r2:
            return False
    return True


def state_hash(data) -> str:
    """sha256 of qpos and qvel."""
    return hashlib.sha256(np.ascontiguousarray(data.qpos).tobytes() + np.ascontiguousarray(data.qvel).tobytes()).hexdigest()


def seed_of(*parts) -> int:
    """Stable 32-bit seed of a label."""
    h = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "little")


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval of k successes in n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))
