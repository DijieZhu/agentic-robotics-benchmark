"""Frame conventions and small rotation helpers shared by every uwkit module.

Conventions (project-wide, see UWKIT_DYNAMICS.md section 1):
  World W : right-handed, z up, free surface at z = 0, depth = -z [m].
  Body  B : FLU (x forward towards the dome, y left / port, z up); origin = CAD origin of the
            official whole-vehicle STEP (on the 4-inch electronics-tube axis).
  Quaternions are MuJoCo order (w, x, y, z) and rotate B -> W:  p_W = R(q) p_B.
  Euler angles (roll, pitch, yaw) are the intrinsic Z-Y-X sequence of the FLU/ENU-style
  convention: R = Rz(yaw) Ry(pitch) Rx(roll).  yaw is measured counter-clockwise about +z_W
  from +x_W (NOT a compass heading; see heading_from_yaw).

Only numpy is used.  Every function accepts plain sequences and returns new float arrays.
"""
from __future__ import annotations

import math

import numpy as np

# FRD <-> FLU (same origin): flip y and z.  Diagonal Fossen coefficients are invariant.
T_FRD_FLU = np.diag([1.0, -1.0, -1.0])
T6_FRD_FLU = np.diag([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])


def frd_to_flu(v):
    """Point or free vector expressed in FRD -> FLU (same origin)."""
    v = np.asarray(v, float)
    return v * np.array([1.0, -1.0, -1.0])


flu_to_frd = frd_to_flu  # the map is an involution


def inertia_frd_to_flu(I):
    """3x3 inertia tensor FRD -> FLU (T I T with T = diag(1, -1, -1))."""
    I = np.asarray(I, float)
    return T_FRD_FLU @ I @ T_FRD_FLU


def wrap_pi(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a, float) + math.pi) % (2.0 * math.pi) - math.pi


def quat_normalize(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    """Hamilton product a*b, (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q):
    q = np.asarray(q, float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q):
    """Rotation matrix R (3x3) of a unit quaternion (w, x, y, z); p_W = R p_B."""
    w, x, y, z = quat_normalize(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    """Unit quaternion (w, x, y, z) with w >= 0 from a rotation matrix (Shepperd)."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def euler_to_quat(roll, pitch, yaw):
    """(w, x, y, z) of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def quat_to_euler(q):
    """(roll, pitch, yaw) [rad] of a (w, x, y, z) quaternion, Z-Y-X intrinsic."""
    w, x, y, z = quat_normalize(q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def mat_to_euler(R):
    R = np.asarray(R, float)
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def quat_from_x_axis(d, up_hint=(0.0, 0.0, 1.0)):
    """Quaternion of a frame whose +x axis is the unit vector d (and +z as close as possible to up_hint)."""
    x = np.asarray(d, float)
    x = x / np.linalg.norm(x)
    up = np.asarray(up_hint, float)
    if abs(float(x @ up)) > 0.95:          # d (anti)parallel to the hint: use body +y as the hint
        up = np.array([0.0, 1.0, 0.0]) if abs(x[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
        y = np.cross(up, x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
    else:
        z = up - (up @ x) * x
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
    return mat_to_quat(np.column_stack([x, y, z]))


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def cross3(a, b):
    """Fast cross product of two 3-vectors (np.cross has ~20 us of Python overhead)."""
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def crossk(a, b):
    """Row-wise cross product of (k, 3) arrays (either may be a single 3-vector)."""
    a = np.asarray(a)
    b = np.asarray(b)
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def heading_from_yaw(yaw, world_x_is_east=True):
    """Compass heading (clockwise from north, [0, 2pi)) of a yaw angle.

    Only meaningful when the task fixes the world axes; with x = east, y = north (ENU) the
    heading is pi/2 - yaw.  uwkit itself never uses compass headings internally."""
    if not world_x_is_east:
        raise ValueError("only the ENU world layout is defined")
    return float((math.pi / 2 - yaw) % (2 * math.pi))


def yaw_from_heading(heading, world_x_is_east=True):
    if not world_x_is_east:
        raise ValueError("only the ENU world layout is defined")
    return float(wrap_pi(math.pi / 2 - heading))
