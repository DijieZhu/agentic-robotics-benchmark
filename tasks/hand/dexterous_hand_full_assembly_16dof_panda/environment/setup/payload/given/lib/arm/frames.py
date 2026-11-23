"""Rigid transforms with the chain's conventions (CHAIN_PLAN.md section 2.1).

`rpy` is fixed-axis XYZ: R = Rz(yaw) Ry(pitch) Rx(roll) (URDF); a pose is a 4x4 homogeneous
matrix; quaternions are `wxyz` (MuJoCo order). The functions reproduce
`arm_links_around_servo_cadquery/setup/payload/arm_design_check.py:39-99` so that the CAD
grader, the adapters and the identity checks agree on every frame to the last digit.
"""

import math

import numpy as np

__all__ = [
    "rpy_to_matrix", "matrix_to_rpy", "pose", "pose_from_dict", "pose_to_dict", "rot_axis", "inv",
    "apply", "angle_between", "quat_from_matrix", "matrix_from_quat", "rotation_angle_deg",
    "frame_with_z", "round_list",
]


def rpy_to_matrix(rpy):
    r, p, y = [float(v) for v in rpy]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(R):
    R = np.asarray(R, dtype=float)
    sy = max(-1.0, min(1.0, -R[2, 0]))
    p = math.asin(sy)
    if abs(math.cos(p)) > 1e-9:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:
        r = 0.0
        y = math.atan2(-R[0, 1], R[1, 1])
    return [r, p, y]


def pose(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def pose_from_dict(d):
    return pose(d["xyz"], d["rpy"])


def pose_to_dict(T, scale=1.0, ndigits_xyz=9, ndigits_rpy=9):
    """`{xyz, rpy}` of a 4x4 pose; `scale` converts the translation (1e-3 for mm -> m)."""
    return {"xyz": round_list(np.asarray(T)[:3, 3] * scale, ndigits_xyz),
            "rpy": round_list(matrix_to_rpy(np.asarray(T)[:3, :3]), ndigits_rpy)}


def rot_axis(axis, q):
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        raise ValueError("zero joint axis")
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def inv(T):
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def apply(T, p):
    p = np.asarray(p, dtype=float)
    return T[:3, :3] @ p + T[:3, 3]


def angle_between(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.acos(max(-1.0, min(1.0, c)))


def rotation_angle_deg(Ra, Rb):
    """Geodesic angle between two rotation matrices, degrees."""
    cosang = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))


def quat_from_matrix(R):
    """`wxyz` quaternion (the CAD grader's `_quat_from_matrix`, arm_design_check.py:860)."""
    m = np.asarray(R, float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], float)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def matrix_from_quat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def frame_with_z(axis, x_hint=None):
    """A right-handed rotation whose third column is `axis` (deterministic choice of x).

    Used for joint frames whose only defined direction is the axis: x is the projection of
    `x_hint` (default: the world axis least aligned with `axis`) onto the plane normal to it.
    """
    z = np.asarray(axis, float)
    z = z / np.linalg.norm(z)
    if x_hint is None:
        x_hint = np.eye(3)[int(np.argmin(np.abs(z)))]
    x = np.asarray(x_hint, float)
    x = x - (x @ z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def round_list(values, ndigits=9):
    out = []
    for v in np.asarray(values, float).ravel().tolist():
        r = round(float(v), ndigits)
        out.append(0.0 if r == 0 else r)   # no "-0.0" in emitted files (byte-identical CI rule)
    return out
