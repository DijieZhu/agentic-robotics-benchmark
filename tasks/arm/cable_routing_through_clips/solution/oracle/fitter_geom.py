"""Small geometry helpers of the reference (fitter.py): arc lengths along the cable's points, rotations."""
import math

import numpy as np


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def arc_of(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def at_arc(P, s):
    a = arc_of(P)
    s = min(max(s, 0.0), a[-1] - 1e-6)
    i = int(np.clip(np.searchsorted(a, s) - 1, 0, len(P) - 2))
    f = (s - a[i]) / max(a[i + 1] - a[i], 1e-9)
    return P[i] + f * (P[i + 1] - P[i]), unit(P[i + 1, :2] - P[i, :2])


def nearest_arc(P, xy):
    """Arc length of the point of the polyline nearest to xy (continuous along the segments)."""
    a = arc_of(P)
    best, s_best = 1e9, 0.0
    for j in range(len(P) - 1):
        p, q = P[j, :2], P[j + 1, :2]
        r = q - p
        t = float(np.clip(np.dot(xy - p, r) / max(np.dot(r, r), 1e-12), 0.0, 1.0))
        d = float(np.linalg.norm(p + t * r - xy))
        if d < best:
            best, s_best = d, a[j] + t * (a[j + 1] - a[j])
    return s_best


def lateral_at(P, c, e, s_lo, s_hi):
    """Signed distance (along the left normal of e) from c to where the polyline passes c's normal plane, among
    passings with arc length in [s_lo, s_hi]; None when it does not pass there."""
    a = arc_of(P)
    n = np.array([-e[1], e[0]])
    best = None
    for j in range(len(P) - 1):
        u0, u1 = float(np.dot(P[j, :2] - c, e)), float(np.dot(P[j + 1, :2] - c, e))
        if (u0 <= 0.0 <= u1) or (u1 <= 0.0 <= u0):
            t = 0.0 if abs(u1 - u0) < 1e-12 else -u0 / (u1 - u0)
            sa = a[j] + t * (a[j + 1] - a[j])
            if s_lo - 0.02 <= sa <= s_hi + 0.02:
                pt = P[j] + t * (P[j + 1] - P[j])
                lat = float(np.dot(pt[:2] - c, n))
                if best is None or abs(lat) < abs(best[0]):
                    best = (lat, float(pt[2]))
    return best


def rot_down(psi):
    """Grip site frame with the tool pointing down and the fingers closing along direction psi."""
    return np.array([[math.cos(psi), math.sin(psi), 0.0], [math.sin(psi), -math.cos(psi), 0.0], [0.0, 0.0, -1.0]])


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotvec(R):
    c = (np.trace(R) - 1) / 2
    ang = math.acos(max(-1.0, min(1.0, c)))
    if ang < 1e-9:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return ax * ang
