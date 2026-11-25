"""Small rigid-body math (numpy only): 4 x 4 transforms, quaternions w x y z, rotation vectors."""
import math

import numpy as np


def quat_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    R = np.asarray(R, float)
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
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
    q = np.array(q, float)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def T_of(pos, quat=None, R=None):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(quat) if R is None else np.asarray(R, float)
    T[:3, 3] = np.asarray(pos, float)
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def rotvec(R):
    """the rotation vector of R (axis x angle, angle in [0, pi])"""
    R = np.asarray(R, float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-12:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(v)
    if n < 1e-9:                                     # angle near pi
        w, V = np.linalg.eigh(0.5 * (R + R.T))
        ax = V[:, int(np.argmax(w))]
        return ax / np.linalg.norm(ax) * ang
    return v / n * ang


def exp_rot(v):
    v = np.asarray(v, float)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def angle(R):
    return float(np.linalg.norm(rotvec(R)))


def wrap(a):
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def seat_errors(E, pitch):
    """(axial, lateral, tilt, clock, k_near, residual) of E = the part's interface frame in the seat frame: the
    coupling's own reading (cell manual: axial along z, lateral across z, tilt between the z axes, clock = the twist
    about z by the swing-twist decomposition, the nearest bolt position and the residual phase)"""
    t = E[:3, 3]
    axial = float(t[2])
    lateral = float(math.hypot(t[0], t[1]))
    tilt = float(math.atan2(math.hypot(E[0, 2], E[1, 2]), E[2, 2]))
    q = mat_to_quat(E[:3, :3])
    clock = wrap(2.0 * math.atan2(float(q[3]), float(q[0])))
    k = int(round(clock / pitch))
    return axial, lateral, tilt, clock, k, wrap(clock - k * pitch)


def quintic(s):
    """the minimum-jerk profile on [0, 1]: position, and the peak speed / acceleration factors 1.875 and 5.7735"""
    s = min(1.0, max(0.0, float(s)))
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


QUINTIC_V = 1.875
QUINTIC_A = 5.7735
