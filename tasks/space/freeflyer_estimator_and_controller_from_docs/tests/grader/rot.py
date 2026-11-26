"""Quaternion helpers of the grader (order x, y, z, w; q maps body vectors into the cabin frame, the printed
convention). The grader's own copy: the scoring code imports nothing of the world package."""
import math

import numpy as np


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qnormalize(q):
    q = np.asarray(q, float)
    n = math.sqrt(float(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]))
    if not (n > 0.0) or not math.isfinite(n):
        raise ValueError("a quaternion of norm %r" % n)
    return q / n


def qmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def qmats(Q):
    """rotation matrices of an array of quaternions (n, 4) -> (n, 3, 3)."""
    Q = np.asarray(Q, float)
    x, y, z, w = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    R = np.empty((len(Q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def qlog(q):
    """unit quaternion -> rotation vector (shortest)"""
    q = np.asarray(q, float)
    if q[3] < 0:
        q = -q
    v = q[:3]
    s = math.sqrt(float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
    if s < 1e-12:
        return 2.0 * v
    th = 2.0 * math.atan2(s, float(q[3]))
    return v * (th / s)


def qangle(a, b):
    d = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return 2.0 * math.acos(min(1.0, float(d)))


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


def body_goal(job, mount):
    """the body pose that meets a job's goal: a leg's or slew's goal pose; for a view the pose whose imaging camera
    (jobs.camera_mount_m) sits at the goal centre looking along the goal axis with the goal's up (the world's own
    construction of the pose a view's goal names)."""
    if job["kind"] == "view" or job.get("request_shape") == "view":
        g = job["goal"]
        a = np.asarray(g["axis"], float)
        u = np.asarray(g["up"], float)
        ez = -u
        ey = np.cross(ez, a)
        R = np.stack([a, ey, ez], 1)
        q = qfrom_mat(R)
        return np.asarray(g["centre"], float) - R @ np.asarray(mount, float), q
    return np.asarray(job["goal"]["position"], float), qnormalize(job["goal"]["attitude"])
