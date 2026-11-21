"""Quaternion and rotation helpers (pure numpy).

Convention everywhere in spacekit: quaternions are (w, x, y, z), Hamilton product, and a pose
quaternion ``q_wb`` rotates body-frame vectors into the world frame (same as MuJoCo ``xquat``).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "quat_mul", "quat_conj", "quat_normalize", "quat_to_mat", "mat_to_quat", "quat_from_rotvec",
    "rotvec_from_quat", "quat_from_axis_angle", "quat_angle", "quat_rotate", "quat_integrate_body",
    "skew",
]


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product a * b of (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q) -> np.ndarray:
    """Conjugate (inverse for unit quaternions)."""
    q = np.asarray(q, float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q, positive_w: bool = False) -> np.ndarray:
    """Unit quaternion; with ``positive_w`` the sign is chosen so that w >= 0."""
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    q = q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    if positive_w and q[0] < 0:
        q = -q
    return q


def quat_to_mat(q) -> np.ndarray:
    """3x3 rotation matrix of a unit (w, x, y, z) quaternion."""
    w, x, y, z = quat_normalize(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R) -> np.ndarray:
    """Unit quaternion (w >= 0) of a rotation matrix (Shepperd's method)."""
    R = np.asarray(R, float).reshape(3, 3)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return quat_normalize(q, positive_w=True)


def quat_from_rotvec(v) -> np.ndarray:
    """Quaternion of the rotation vector v (axis * angle, rad)."""
    v = np.asarray(v, float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        q = np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]])
        return q / np.linalg.norm(q)
    a = v / th
    return np.array([np.cos(th / 2), *(np.sin(th / 2) * a)])


def rotvec_from_quat(q) -> np.ndarray:
    """Rotation vector (axis * angle, angle in [0, pi]) of a quaternion."""
    q = quat_normalize(q, positive_w=True)
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return 2.0 * q[1:]
    ang = 2.0 * np.arctan2(s, q[0])
    return q[1:] / s * ang


def quat_from_axis_angle(axis, angle) -> np.ndarray:
    """Quaternion of a rotation by ``angle`` (rad) about ``axis``."""
    a = np.asarray(axis, float)
    return quat_from_rotvec(a / np.linalg.norm(a) * float(angle))


def quat_angle(qa, qb) -> float:
    """Angle (rad, in [0, pi]) of the rotation between two orientations."""
    d = quat_mul(quat_conj(qa), qb)
    return float(2.0 * np.arctan2(np.linalg.norm(d[1:]), abs(d[0])))


def quat_rotate(q, v) -> np.ndarray:
    """Rotate vector(s) v (..., 3) by q."""
    return np.asarray(v, float) @ quat_to_mat(q).T


def quat_integrate_body(q, omega_body, dt) -> np.ndarray:
    """q * exp(omega_body * dt / 2): constant body-frame rate over dt."""
    return quat_normalize(quat_mul(q, quat_from_rotvec(np.asarray(omega_body, float) * dt)))


def skew(v) -> np.ndarray:
    """Cross-product matrix [v]x."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
