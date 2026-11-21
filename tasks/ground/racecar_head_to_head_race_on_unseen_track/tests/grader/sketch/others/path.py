"""Path geometry of the slow cars and the reference speed of a path.

A slow car's path (`truth.avoidance.slow_cars[].path.samples`, rows [x, y, reference speed]) is the trajectory of
the car's reference point (the wheelbase midpoint), closed, in driving order. `Path` gives stations,
interpolation and projections on it; `reference_speed` is the quasi-steady speed profile that defines the column
`reference speed`; `outline` gives the footprint points that the follower's detection band uses.
"""

import bisect
import math

import numpy as np

from sketch.examples import constants as K

G = 9.81
REF_X = K.REF_POINT_C[0]


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Path(object):
    """A closed polyline with a value per sample (the reference speed)."""

    def __init__(self, samples):
        a = np.asarray(samples, dtype=float)
        if a.ndim != 2 or a.shape[1] < 2 or len(a) < 3:
            raise ValueError("a path needs at least 3 rows [x, y, ...]")
        self.p = np.ascontiguousarray(a[:, :2])
        self.v = a[:, 2].copy() if a.shape[1] > 2 else np.zeros(len(a))
        self.n = len(a)
        self.seg = np.roll(self.p, -1, axis=0) - self.p
        self.seg_len = np.linalg.norm(self.seg, axis=1)
        if float(self.seg_len.min()) <= 1e-9:
            raise ValueError("a path has coincident samples")
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(self.seg_len.sum())
        self.useg = self.seg / self.seg_len[:, None]
        t = np.roll(self.p, -1, axis=0) - np.roll(self.p, 1, axis=0)
        self.tan = t / np.linalg.norm(t, axis=1, keepdims=True)
        self.step = float(np.median(self.seg_len))
        # plain-Python copies for the per-step lookups (numpy's call overhead dominates on 2-vectors)
        self._s = self.s.tolist()
        self._p = self.p.tolist()
        self._seg = self.seg.tolist()
        self._len = self.seg_len.tolist()
        self._tan = self.tan.tolist()
        self._v = self.v.tolist()

    # ------------------------------------------------------------------ lookup
    def locate(self, s):
        s = s % self.length
        i = bisect.bisect_right(self._s, s) - 1
        i = min(max(i, 0), self.n - 1)
        return i, (i + 1) % self.n, (s - self._s[i]) / self._len[i]

    def at(self, s):
        """(x, y, heading, reference speed) at station s; the heading interpolates the sample tangents."""
        i, j, u = self.locate(s)
        p, d = self._p[i], self._seg[i]
        ti, tj = self._tan[i], self._tan[j]
        return (p[0] + u * d[0], p[1] + u * d[1], math.atan2((1.0 - u) * ti[1] + u * tj[1], (1.0 - u) * ti[0] + u * tj[0]),
                (1.0 - u) * self._v[i] + u * self._v[j])

    def speed(self, s):
        i, j, u = self.locate(s)
        return (1.0 - u) * self._v[i] + u * self._v[j]

    def gap(self, s_from, s_to):
        """Station distance from s_from forward to s_to, in [0, length)."""
        return (s_to - s_from) % self.length

    # ------------------------------------------------------------------ projections
    def _window(self, i_hint, back, ahead):
        return np.arange(i_hint - back, i_hint + ahead + 1) % self.n

    def project(self, x, y, s_hint=None, back_m=1.0, ahead_m=3.0):
        """(s, lateral offset (left +), squared distance) of point (x, y): on the segments near `s_hint` when it is
        given, else on the whole path."""
        q = np.array([x, y], dtype=float)
        if s_hint is None:
            idx = np.arange(self.n)
        else:
            i, _, _ = self.locate(s_hint)
            idx = self._window(i, int(back_m / self.step) + 2, int(ahead_m / self.step) + 2)
        a, d, ln = self.p[idx], self.seg[idx], self.seg_len[idx]
        u = np.clip(((q - a) * d).sum(axis=1) / (ln * ln), 0.0, 1.0)
        foot = a + d * u[:, None]
        dist2 = ((foot - q) ** 2).sum(axis=1)
        k = int(np.argmin(dist2))
        i = int(idx[k])
        lat = float(self.useg[i, 0] * (q[1] - foot[k, 1]) - self.useg[i, 1] * (q[0] - foot[k, 0]))
        return float((self.s[i] + u[k] * ln[k]) % self.length), lat, float(dist2[k])

    def project_many(self, pts, s0, ahead_m, back_m=0.5):
        """Stations relative to s0 (in [-back_m, ahead_m + ...]) and lateral offsets of many points, projected on the
        segments from s0 - back_m to s0 + ahead_m only. Returns (rel_s, lat, dist2)."""
        i, _, _ = self.locate(s0)
        idx = self._window(i, int(back_m / self.step) + 1, int(ahead_m / self.step) + 2)
        a, d, ln = self.p[idx], self.seg[idx], self.seg_len[idx]           # (m, 2)
        q = np.asarray(pts, dtype=float)                                     # (k, 2)
        rel = q[:, None, :] - a[None, :, :]                                   # (k, m, 2)
        u = np.clip((rel * d[None]).sum(axis=2) / (ln * ln)[None], 0.0, 1.0)
        foot = a[None] + d[None] * u[..., None]
        diff = q[:, None, :] - foot
        dist2 = (diff ** 2).sum(axis=2)
        k = np.argmin(dist2, axis=1)
        rows = np.arange(len(q))
        seg_i = idx[k]
        s_abs = self.s[seg_i] + u[rows, k] * ln[k]
        rel_s = (s_abs - s0 + back_m) % self.length - back_m
        ud = self.useg[seg_i]
        lat = ud[:, 0] * diff[rows, k, 1] - ud[:, 1] * diff[rows, k, 0]
        return rel_s, lat, dist2[rows, k]

    def target_at_distance(self, x, y, s_from, dist):
        """The path point ahead of station s_from whose straight distance from (x, y) is `dist` (pure pursuit)."""
        s = s_from + dist
        for _ in range(4):
            px, py, _, _ = self.at(s)
            e = dist - math.hypot(px - x, py - y)
            if abs(e) < 1e-3:
                break
            s += e
        px, py, _, _ = self.at(s)
        return px, py, s

    def curvature(self, w=2):
        """Signed curvature at every sample from the circle through samples i - w, i, i + w."""
        a, b, c = np.roll(self.p, w, axis=0), self.p, np.roll(self.p, -w, axis=0)
        return signed_curvature(a, b, c)


def signed_curvature(a, b, c):
    """Signed curvature of the circle through a, b, c (rows): positive when it turns left."""
    ab, bc, ca = b - a, c - b, a - c
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    den = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ca, axis=1)
    return np.where(den > 1e-12, 2.0 * cross / np.maximum(den, 1e-300), 0.0)


def reference_speed(points, grip, params, top_speed_mps):
    """The reference speed of a closed path: the quasi-steady-state speed of this car on this line
    with every grip-dependent capability scaled by f^2, f = `speed.ref_speed_factor` (0.93, a factor on SPEED, as the
    reference program's heat 2):
        v <= f sqrt(c_lat mu g / |kappa|) = sqrt(c_lat f^2 mu g / |kappa|)   (c_lat = lateral_frac_of_grip_g, 0.975)
        v <= top_speed_frac x top speed,
        forward pass a <= min(accel_max, c_trac f^2 mu g)                    (c_trac = traction_frac_of_grip_g, 0.95)
        backward pass a <= c_brk f^2 mu g                                    (c_brk = brake_frac_of_grip_g, 1.0)
    around the closed loop. The lateral limit is the one of the reference program's `qss_speed` (sqrt(c_lat f^2 mu g / kappa)).
    `grip` is the friction under each sample (already the minimum over a window, see placement)."""
    sp = params["speed"]
    p = np.asarray(points, dtype=float)
    mu = np.asarray(grip, dtype=float)
    f2 = float(sp["ref_speed_factor"]) ** 2
    c_lat, c_trac, c_brk = (float(sp["lateral_frac_of_grip_g"]), float(sp["traction_frac_of_grip_g"]),
                            float(sp["brake_frac_of_grip_g"]))
    n = len(p)
    w = int(sp["curvature_window_samples"])
    kap = np.abs(signed_curvature(np.roll(p, w, axis=0), p, np.roll(p, -w, axis=0)))
    v_top = float(sp["top_speed_frac"]) * float(top_speed_mps)
    v = np.minimum(np.sqrt(c_lat * f2 * mu * G / np.maximum(kap, 1e-9)), v_top)
    ds = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    a_acc = np.minimum(float(sp["accel_max_mps2"]), c_trac * f2 * mu * G)
    a_brk = c_brk * f2 * mu * G
    for _ in range(2):
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], math.sqrt(v[i] * v[i] + 2.0 * a_acc[i] * ds[i]))
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] * v[j] + 2.0 * a_brk[i] * ds[i]))
    return v


def outline(x0, x1, half_w, spacing):
    """Points on the perimeter of the rectangle x0..x1, -half_w..half_w (car frame), at most `spacing` apart."""
    out = []
    for (ax, ay), (bx, by) in (((x0, -half_w), (x1, -half_w)), ((x1, -half_w), (x1, half_w)),
                               ((x1, half_w), (x0, half_w)), ((x0, half_w), (x0, -half_w))):
        n = max(1, int(math.ceil(math.hypot(bx - ax, by - ay) / spacing)))
        for k in range(n):
            u = k / n
            out.append((ax + u * (bx - ax), ay + u * (by - ay)))
    return np.asarray(out, dtype=float)


def car_extent(tail_box_centre=None):
    """(rear x, front x, half width) of a car's footprint in its frame C; with a tail box its rear face counts."""
    x0 = K.X_REAR_M
    if tail_box_centre is not None:
        x0 = min(x0, float(tail_box_centre[0]) - 0.5 * K.TAIL_BOX_SIZE_M[0])
    return x0, K.X_FRONT_M, 0.5 * K.ENVELOPE_M[1]


def transform(local, pose):
    """World points of car-frame points `local` (k, 2) at pose [x, y, yaw]."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    out = np.empty_like(local)
    out[:, 0] = pose[0] + c * local[:, 0] - s * local[:, 1]
    out[:, 1] = pose[1] + s * local[:, 0] + c * local[:, 1]
    return out
