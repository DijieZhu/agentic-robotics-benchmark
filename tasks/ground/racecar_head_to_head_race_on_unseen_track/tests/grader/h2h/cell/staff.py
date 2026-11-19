"""Where the staff set a stopped car down in a heat.

The search of the shared relocation is kept as it is (the same candidates in the same order, the same three error
draws per relocation); a candidate is valid in a heat when it is valid for the shared rule and the car's envelope
stays `clearance_m` clear of the band of every field car's line (the line +- `half_width_m`). When no candidate of
the search satisfies the band condition, the first candidate that satisfies the shared rule is used and the placement
is a fallback; when not even that exists, the shared rule's own fallback (the centreline) is used, also a fallback.

    rel = BandRelocator(track, lines, config, params, seed, half_width_m, clearance_m)
    rel.set_bands([path, ...])              # the field lines of the heat (paths with .p, the points of the line)
    pose, info = rel.place_clear(s, d, obstacles)       # info["placement"] = "clear" or "fallback"
"""

import math

import numpy as np

from sketch.referee.geometry import car_corners
from sketch.referee.staff import Relocator


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    u = 0.0 if l2 <= 0.0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    qx, qy = ax + u * dx, ay + u * dy
    return math.hypot(px - qx, py - qy)


def _cross(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _segments_cross(a, b, c, d):
    d1 = _cross(c[0], c[1], d[0], d[1], a[0], a[1])
    d2 = _cross(c[0], c[1], d[0], d[1], b[0], b[1])
    d3 = _cross(a[0], a[1], b[0], b[1], c[0], c[1])
    d4 = _cross(a[0], a[1], b[0], b[1], d[0], d[1])
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _inside(poly, x, y):
    n = len(poly)
    sign = None
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        c = _cross(ax, ay, bx, by, x, y)
        if abs(c) < 1e-15:
            continue
        if sign is None:
            sign = c > 0
        elif (c > 0) != sign:
            return False
    return True


def polygon_polyline_distance(poly, pts, closed=True):
    """Smallest distance between a convex polygon (list of (x, y)) and a polyline (k x 2 points); 0 when they meet."""
    n = len(pts)
    segs = range(n if closed else n - 1)
    best = math.inf
    edges = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    for i in segs:
        a = (float(pts[i][0]), float(pts[i][1]))
        b = (float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1]))
        if _inside(poly, a[0], a[1]) or _inside(poly, b[0], b[1]):
            return 0.0
        for p, q in edges:
            if _segments_cross(a, b, p, q):
                return 0.0
            best = min(best, _seg_dist(a[0], a[1], p[0], p[1], q[0], q[1]), _seg_dist(b[0], b[1], p[0], p[1], q[0], q[1]),
                       _seg_dist(p[0], p[1], a[0], a[1], b[0], b[1]))
    return best


class BandRelocator(Relocator):
    """The shared relocation with the band condition of a heat (see the module text)."""

    def __init__(self, track, lines, config, params, seed, half_width_m, clearance_m):
        super().__init__(track, lines, config, params, seed)
        self.band = float(half_width_m) + float(clearance_m)
        self.bands = []
        self.first_base = None
        self.checking = False

    def set_bands(self, paths):
        """The field lines of the open heat (objects with `.p`, the points of a closed line); [] outside a heat."""
        self.bands = [np.asarray(p.p, dtype=float) for p in paths]

    def band_clear(self, pose):
        env = car_corners(pose)
        cx = sum(p[0] for p in env) / 4.0
        cy = sum(p[1] for p in env) / 4.0
        reach = self.band + 0.5 * math.hypot(env[0][0] - env[2][0], env[0][1] - env[2][1]) + 0.5
        for pts in self.bands:
            near = np.nonzero(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) <= reach)[0]
            if not len(near):
                continue
            n = len(pts)
            idx = sorted(set(int(i) for i in near) | set(int(i - 1) % n for i in near))
            for i in idx:
                seg = [pts[i], pts[(i + 1) % n]]
                if polygon_polyline_distance(env, seg, closed=False) < self.band - 1e-12:
                    return False
        return True

    def valid(self, pose, obstacles=()):
        ok = super().valid(pose, obstacles)
        if not self.checking or not ok:
            return ok
        if self.first_base is None:
            self.first_base = [float(v) for v in pose]
        return self.band_clear(pose)

    def place_clear(self, s_clean, d_clean, obstacles=()):
        """(pose, info): the shared search with the band condition; info["placement"] is "clear" or "fallback"."""
        self.first_base = None
        self.checking = bool(self.bands)
        try:
            pose, info = self.place(s_clean, d_clean, obstacles)
        finally:
            self.checking = False
        info = dict(info)
        if not info["fallback"]:
            info["placement"] = "clear"
            return pose, info
        info["placement"] = "fallback"
        if self.first_base is not None:
            pose = list(self.first_base)
            info["side"] = "first_valid"
        return pose, info


__all__ = ["BandRelocator", "polygon_polyline_distance"]
