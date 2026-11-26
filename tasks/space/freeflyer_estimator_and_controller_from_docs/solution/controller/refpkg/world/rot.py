# Generated file (the task's solution): do not edit; regenerate it from the maintained source.
import math
import numpy as np

def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])

def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def qnormalize(q):
    q = np.asarray(q, float)
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return q / n

def qmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def qexp(rv):
    th = math.sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2])
    if th < 1e-12:
        return qnormalize(np.array([0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2], 1.0]))
    s = math.sin(0.5 * th) / th
    return np.array([rv[0] * s, rv[1] * s, rv[2] * s, math.cos(0.5 * th)])

def qlog(q):
    q = np.asarray(q, float)
    if q[3] < 0:
        q = -q
    v = q[:3]
    s = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if s < 1e-12:
        return 2.0 * v
    th = 2.0 * math.atan2(s, q[3])
    return v * (th / s)

def qangle(a, b):
    d = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return 2.0 * math.acos(min(1.0, d))

def qrot(q, v):
    return qmat(q) @ np.asarray(v, float)

def qfrom_mat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = qnormalize(np.array(q))
    return q if q[3] >= 0 else -q

def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

def rotz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
