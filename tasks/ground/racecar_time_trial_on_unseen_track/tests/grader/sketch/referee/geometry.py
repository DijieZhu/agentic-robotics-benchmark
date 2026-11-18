"""Geometry the referee evaluates after every physics step: the track corridor, the timing line and checkpoints, the
tape lines of open walls, and the floor lookup.

`FastTrack` gives the same station `s`, lateral offset `d` and inside test as the contract's reference
`sketch.examples.check.Track` (the contract: the station of the NEAREST centreline point) at a fraction of the cost
(an exact candidate set per 5 cm cell, see the class). Agreement with `check.Track` is a test.
"""

import math

import numpy as np

from sketch.examples import check


class FastTrack(object):
    """check.Track's projection, exact, with a per-cell candidate cache.

    The plane is cut into square cells of `cell_m`. For a cell with centre c and half diagonal r, every segment that
    can be the nearest one for SOME point of the cell is within D(c) + 2 r of c (D(c) = distance from c to its
    nearest segment): the nearest segment j* of a point q of the cell has dist(c, j*) <= dist(q, j*) + r <=
    dist(q, j_c) + r <= D(c) + 2 r. The candidates of a cell are computed once, on its first query, by a global
    numpy search; a query is then a pure-Python evaluation of those few segments, with the lowest index winning a
    tie (the rule of numpy's argmin in check.Track). So the answers are check.Track's, and they do not depend on
    which cells happen to be cached."""

    def __init__(self, world, cell_m=0.05):
        tr = check.Track(world)
        self.ref = tr
        self.p, self.seg, self.seg_len, self.s, self.length = tr.p, tr.seg, tr.seg_len, tr.s, tr.length
        self.l2 = self.seg_len ** 2
        self.left, self.right = tr.left, tr.right
        self.n = tr.n
        self.h = float(cell_m)
        self._r2 = 2.0 * math.sqrt(0.5) * self.h + 1e-9
        self._P = [(float(a), float(b)) for a, b in self.p]
        self._SEG = [(float(a), float(b)) for a, b in self.seg]
        self._L2 = [float(v) for v in self.l2]
        self._LEN = [float(v) for v in self.seg_len]
        self._S = [float(v) for v in self.s]
        self._LEFT = [float(v) for v in self.left]
        self._RIGHT = [float(v) for v in self.right]
        self._cells = {}
        self.stats = {"cells": 0, "queries": 0}

    def _candidates(self, key):
        cx, cy = (key[0] + 0.5) * self.h, (key[1] + 0.5) * self.h
        q = np.array([cx, cy])
        u = np.clip(((q - self.p) * self.seg).sum(axis=1) / self.l2, 0.0, 1.0)
        dist = np.sqrt(((self.p + self.seg * u[:, None] - q) ** 2).sum(axis=1))
        cand = tuple(int(j) for j in np.nonzero(dist <= float(dist.min()) + self._r2)[0])
        self._cells[key] = cand
        self.stats["cells"] += 1
        return cand

    def project1(self, x, y):
        """(s, d, seg, u) of one point: check.Track.project plus the segment and its parameter."""
        self.stats["queries"] += 1
        key = (math.floor(x / self.h), math.floor(y / self.h))
        cand = self._cells.get(key)
        if cand is None:
            cand = self._candidates(key)
        best = None
        for j in cand:
            px, py = self._P[j]
            sx, sy = self._SEG[j]
            u = ((x - px) * sx + (y - py) * sy) / self._L2[j]
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            fx, fy = px + sx * u, py + sy * u
            dist = math.sqrt((fx - x) * (fx - x) + (fy - y) * (fy - y))
            if best is None or dist < best[0]:
                best = (dist, j, u, fx, fy)
        dist, j, u, fx, fy = best
        sx, sy = self._SEG[j]
        d = dist if sx * (y - fy) - sy * (x - fx) >= 0.0 else -dist
        return (self._S[j] + u * self._LEN[j]) % self.length, d, j, u

    def widths1(self, j, u):
        k = j + 1 if j + 1 < self.n else 0
        return (1.0 - u) * self._LEFT[j] + u * self._LEFT[k], (1.0 - u) * self._RIGHT[j] + u * self._RIGHT[k]

    def inside1(self, x, y, margin=0.0):
        """(inside, s, d) of one point: check.Track.inside."""
        s, d, j, u = self.project1(x, y)
        wl, wr = self.widths1(j, u)
        return (-wr + margin <= d <= wl - margin), s, d

    def project(self, pts):
        """Vectorised form for many points: arrays (s, d)."""
        out = [self.project1(float(x), float(y)) for x, y in np.asarray(pts, dtype=float).reshape(-1, 2)]
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])

    # ------------------------------------------------------------------ positions on the track
    def at(self, s):
        """(x, y, heading, left, right) at station s (check.Track.at)."""
        return self.ref.at(s)

    def point(self, s, d):
        """World point at station s and lateral offset d (left positive), and the heading there."""
        x, y, h, _, _ = self.ref.at(s)
        return x - d * math.sin(h), y + d * math.cos(h), h

    def ahead(self, s_from, s_to):
        """Station distance from s_from forward to s_to, in [0, length)."""
        return (s_to - s_from) % self.length

    def signed_gap(self, s_a, s_b):
        """s_a - s_b wrapped into (-length / 2, length / 2]."""
        g = (s_a - s_b) % self.length
        return g - self.length if g > 0.5 * self.length else g


class Line(object):
    """The timing line or a checkpoint: segment a (left boundary) -> b (right boundary) with the driving direction.
    Plain floats: `crossing` runs after every physics step."""

    def __init__(self, line, name, index):
        self.name = name
        self.index = index                       # -1 for the timing line, else the checkpoint's order
        self.s_m = float(line["s_m"])
        self.ax, self.ay = float(line["a"][0]), float(line["a"][1])
        self.bx, self.by = float(line["b"][0]), float(line["b"][1])
        fx, fy = float(line["forward"][0]), float(line["forward"][1])
        r = math.hypot(fx, fy)
        self.fx, self.fy = fx / r, fy / r
        self.abx, self.aby = self.bx - self.ax, self.by - self.ay
        self.ab2 = self.abx * self.abx + self.aby * self.aby
        pad = 1.0
        self.box = (min(self.ax, self.bx) - pad, max(self.ax, self.bx) + pad, min(self.ay, self.by) - pad, max(self.ay, self.by) + pad)

    def side(self, p):
        return (p[0] - self.ax) * self.fx + (p[1] - self.ay) * self.fy

    def crossing(self, p0, p1):
        """None, or (direction +1 / -1, fraction of the motion p0 -> p1 at the line) when the motion crosses the
        segment: forward when side goes from < 0 to >= 0, backward from >= 0 to < 0."""
        x0, x1, y0, y1 = self.box
        if not (x0 <= p1[0] <= x1 and y0 <= p1[1] <= y1):
            return None
        s0 = (p0[0] - self.ax) * self.fx + (p0[1] - self.ay) * self.fy
        s1 = (p1[0] - self.ax) * self.fx + (p1[1] - self.ay) * self.fy
        if s0 < 0.0 <= s1:
            direction = 1
        elif s1 < 0.0 <= s0:
            direction = -1
        else:
            return None
        frac = s0 / (s0 - s1)
        x = p0[0] + frac * (p1[0] - p0[0])
        y = p0[1] + frac * (p1[1] - p0[1])
        u = ((x - self.ax) * self.abx + (y - self.ay) * self.aby) / self.ab2
        if not 0.0 <= u <= 1.0:
            return None
        return direction, frac


class Tape(object):
    """The tape of an open wall: a point p is beyond it when
    0 < (p - tape midpoint) . outward <= TAPE_DEPTH_M (1.5 m) and its projection falls between the tape's ends.

    `near(p)` is a conservative pre-filter on the reference point: True whenever some point within one car length
    (`ENVELOPE_M[0]`, 0.388 m; a wheel point is at most 0.151 m from the reference point) of p could be beyond the
    tape, i.e. p inside the tape-by-depth rectangle inflated by a car length. It never drops a wheel point that the
    definition counts (1.0's L1 radius around the midpoint did: a 0.6 m opening with the reference point 1.6 m out and
    0.3 m aside had a rear wheel point 1.47 m out that counted by the definition)."""

    def __init__(self, open_wall, depth=None, inflate=None):
        from sketch.examples import constants as K
        self.id = open_wall["id"]
        (self.x0, self.y0), (self.x1, self.y1) = [(float(a), float(b)) for a, b in open_wall["tape"]]
        self.mx, self.my = 0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1)
        self.ox, self.oy = float(open_wall["outward"][0]), float(open_wall["outward"][1])
        self.tx, self.ty = self.x1 - self.x0, self.y1 - self.y0
        self.t2 = self.tx * self.tx + self.ty * self.ty
        self.tlen = math.sqrt(self.t2)
        self.depth = float(K.TAPE_DEPTH_M if depth is None else depth)
        self.inflate = float(K.ENVELOPE_M[0] if inflate is None else inflate)

    def near(self, p):
        v = (p[0] - self.mx) * self.ox + (p[1] - self.my) * self.oy
        if v < -self.inflate or v > self.depth + self.inflate:
            return False
        a = ((p[0] - self.x0) * self.tx + (p[1] - self.y0) * self.ty) / self.tlen
        return -self.inflate <= a <= self.tlen + self.inflate

    def beyond1(self, x, y):
        v = (x - self.mx) * self.ox + (y - self.my) * self.oy
        if v <= 0.0 or v > self.depth:
            return False
        u = ((x - self.x0) * self.tx + (y - self.y0) * self.ty) / self.t2
        return 0.0 <= u <= 1.0

    def beyond(self, pts):
        """Boolean array: which of the points (m x 2) are beyond the tape."""
        return np.array([self.beyond1(float(x), float(y)) for x, y in np.asarray(pts, dtype=float).reshape(-1, 2)], dtype=bool)

    def count(self, pts):
        return sum(1 for x, y in pts if self.beyond1(x, y))


class FastFloor(object):
    """check.FloorMap with a bounding-box prefilter (same answers)."""

    def __init__(self, world):
        self.ref = check.FloorMap(world)
        self.base_grip = self.ref.base_grip
        self.patches = []
        for p in world["truth"]["floor"]["patches"]:
            v = np.asarray(p["polygon"], dtype=float)
            self.patches.append((p["id"], float(p["grip"]), p["polygon"], float(v[:, 0].min()), float(v[:, 0].max()),
                                 float(v[:, 1].min()), float(v[:, 1].max())))
        self.patches.reverse()

    def zone_at(self, x, y):
        for pid, _, poly, x0, x1, y0, y1 in self.patches:
            if x0 <= x <= x1 and y0 <= y <= y1 and check.point_in_polygon(x, y, poly):
                return pid
        return None

    def grip_at(self, x, y):
        for _, grip, poly, x0, x1, y0, y1 in self.patches:
            if x0 <= x <= x1 and y0 <= y <= y1 and check.point_in_polygon(x, y, poly):
                return grip
        return self.base_grip


def car_corners(pose, length_front=None, length_rear=None, half_width=None):
    """The four corners of the car's envelope at a pose (car frame C: front at x = X_FRONT_M, rear at X_REAR_M)."""
    from sketch.examples import constants as K
    xf = K.X_FRONT_M if length_front is None else length_front
    xr = K.X_REAR_M if length_rear is None else length_rear
    hw = 0.5 * K.ENVELOPE_M[1] if half_width is None else half_width
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * a - s * b, y + s * a + c * b) for a, b in ((xf, hw), (xr, hw), (xr, -hw), (xf, -hw))]


def polygons_overlap(pa, pb):
    """Separating axis test of two convex polygons (lists of (x, y))."""
    for poly in (pa, pb):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            nx, ny = y1 - y2, x2 - x1
            a = [nx * x + ny * y for x, y in pa]
            b = [nx * x + ny * y for x, y in pb]
            if max(a) < min(b) or max(b) < min(a):
                return False
    return True
