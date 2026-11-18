"""Plane geometry of the reference program: closed loops, projections, the as-built walls and ray casts.

numpy only (the program runs in its own process with nothing but carkit on its path).

    loop = Loop(points, ds=0.1)               # a closed polyline resampled at a uniform spacing
    s, d, i = loop.project(x, y, hint=i)      # station, signed lateral offset (left +), segment index
    walls = Walls.from_world(world)           # duct capsules (as built, moved pieces applied) and tape segments
    left, right = walls.free_along_normals(loop.p, loop.nl, reach)    # free distance to the left / right
"""

import math

import numpy as np

DUCT_RADIUS_M = 0.165


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def resample_closed(pts, n, src=False):
    """n points at uniform arc length along the closed polyline `pts` (first point kept). With `src`, also the
    index of the source segment of every new point and the fraction along it."""
    pts = np.asarray(pts, dtype=float)
    seg = np.roll(pts, -1, axis=0) - pts
    ln = np.hypot(seg[:, 0], seg[:, 1])
    keep = ln > 1e-9
    orig = np.nonzero(keep)[0]
    pts, seg, ln = pts[keep], seg[keep], ln[keep]
    cum = np.concatenate([[0.0], np.cumsum(ln)])
    total = float(cum[-1])
    t = np.arange(n) * (total / n)
    i = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(ln) - 1)
    u = (t - cum[i]) / ln[i]
    out = pts[i] + seg[i] * u[:, None]
    if src:
        return out, total, orig[i], u
    return out, total


def periodic_smooth(a, half_window):
    """Moving average over 2 * half_window + 1 samples of a periodic array (no smoothing for half_window < 1)."""
    k = int(half_window)
    if k < 1:
        return np.array(a, dtype=float)
    a = np.asarray(a, dtype=float)
    c = np.concatenate([a[-k:], a, a[:k]])
    cs = np.concatenate([[0.0], np.cumsum(c)])
    return (cs[2 * k + 1:] - cs[:-2 * k - 1]) / (2 * k + 1)


def periodic_min(a, half_window):
    k = int(half_window)
    a = np.asarray(a, dtype=float)
    if k < 1:
        return a.copy()
    out = a.copy()
    for j in range(1, k + 1):
        out = np.minimum(out, np.minimum(np.roll(a, j), np.roll(a, -j)))
    return out


class Loop(object):
    """A closed polyline at uniform spacing: p (n, 2), stations s, tangents tan, left normals nl, heading, curvature
    kappa (signed, left turns positive; three-point circle over +-`kappa_half` samples, then smoothed)."""

    def __init__(self, pts, ds=0.1, kappa_half_m=0.15, kappa_smooth_m=0.15, values=None):
        pts = np.asarray(pts, dtype=float)
        seg = np.roll(pts, -1, axis=0) - pts
        total = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        n = max(16, int(round(total / ds)))
        self.p, self.length, self.src_i, self.src_u = resample_closed(pts, n, src=True)
        self.n = n
        self.ds = self.length / n
        self.s = np.arange(n) * self.ds
        t = np.roll(self.p, -1, axis=0) - np.roll(self.p, 1, axis=0)
        self.tan = t / np.hypot(t[:, 0], t[:, 1])[:, None]
        self.nl = np.stack([-self.tan[:, 1], self.tan[:, 0]], axis=1)
        self.heading = np.arctan2(self.tan[:, 1], self.tan[:, 0])
        w = max(1, int(round(kappa_half_m / self.ds)))
        self.kappa = periodic_smooth(signed_curvature(np.roll(self.p, w, axis=0), self.p, np.roll(self.p, -w, axis=0)),
                                     int(round(kappa_smooth_m / self.ds)))
        self.seg = np.roll(self.p, -1, axis=0) - self.p
        self.seg_len2 = np.einsum("ij,ij->i", self.seg, self.seg)
        self.values = values

    def wrap_s(self, s):
        return s % self.length

    def index(self, s):
        return int(math.floor((s % self.length) / self.ds)) % self.n

    def interp(self, arr, s):
        """Linear interpolation of a per-sample periodic array at station s (scalar)."""
        x = (s % self.length) / self.ds
        i = int(math.floor(x)) % self.n
        u = x - math.floor(x)
        return float((1.0 - u) * arr[i] + u * arr[(i + 1) % self.n])

    def point(self, s):
        x = (s % self.length) / self.ds
        i = int(math.floor(x)) % self.n
        u = x - math.floor(x)
        q = self.p[i] + u * self.seg[i]
        return float(q[0]), float(q[1])

    def heading_at(self, s):
        x = (s % self.length) / self.ds
        i = int(math.floor(x)) % self.n
        u = x - math.floor(x)
        h0, h1 = self.heading[i], self.heading[(i + 1) % self.n]
        return float(h0 + u * wrap(h1 - h0))

    def project(self, x, y, hint=None, window_m=3.0):
        """(s, d, i): station of the nearest point, signed offset (left +), segment index. With a hint index the
        search is local (+- window_m); without, global."""
        if hint is None:
            idx = np.arange(self.n)
        else:
            k = int(window_m / self.ds) + 1
            idx = (int(hint) + np.arange(-k, k + 1)) % self.n
        p, sg, l2 = self.p[idx], self.seg[idx], self.seg_len2[idx]
        qx, qy = x - p[:, 0], y - p[:, 1]
        u = np.clip((qx * sg[:, 0] + qy * sg[:, 1]) / l2, 0.0, 1.0)
        fx, fy = qx - u * sg[:, 0], qy - u * sg[:, 1]
        dist2 = fx * fx + fy * fy
        j = int(np.argmin(dist2))
        i = int(idx[j])
        cross = sg[j, 0] * fy[j] - sg[j, 1] * fx[j]
        d = math.sqrt(float(dist2[j]))
        return float((self.s[i] + float(u[j]) * self.ds) % self.length), (d if cross >= 0.0 else -d), i

    def gap(self, s_a, s_b):
        """s_a - s_b wrapped to (-L/2, L/2]."""
        g = (s_a - s_b) % self.length
        return g - self.length if g > 0.5 * self.length else g


def signed_curvature(a, b, c):
    """Signed curvature of the circle through three points (arrays (n, 2)); left turns positive."""
    ab, bc, ac = b - a, c - b, c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    la, lb, lc = np.hypot(ab[:, 0], ab[:, 1]), np.hypot(bc[:, 0], bc[:, 1]), np.hypot(ac[:, 0], ac[:, 1])
    return 2.0 * cross / np.maximum(la * lb * lc, 1e-12)


def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def points_in_polygon(pts, poly):
    """Vectorised even-odd test of points (n, 2) against one polygon."""
    pts = np.asarray(pts, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    v = np.asarray(poly, dtype=float)
    for i in range(len(v)):
        x1, y1 = v[i]
        x2, y2 = v[(i + 1) % len(v)]
        if y1 == y2:
            continue
        cond = (y1 > y) != (y2 > y)
        xc = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= cond & (x < xc)
    return inside


class GripMap(object):
    """The floor of the world file: grip of the last listed patch containing a point, else the base grip."""

    def __init__(self, world):
        floor = world["truth"]["floor"]
        self.base = float(floor["base_grip"])
        self.patches = [(np.asarray(p["polygon"], dtype=float), float(p["grip"]), p["id"], float(p.get("edge_height_m", 0.0)))
                        for p in floor["patches"]]

    def grip_at(self, x, y):
        for poly, g, _pid, _h in reversed(self.patches):
            if point_in_polygon(x, y, poly):
                return g
        return self.base

    def grip_points(self, pts):
        pts = np.asarray(pts, dtype=float)
        out = np.full(len(pts), self.base)
        done = np.zeros(len(pts), dtype=bool)
        for poly, g, _pid, _h in reversed(self.patches):
            m = (~done) & points_in_polygon(pts, poly)
            out[m] = g
            done |= m
        return out

    def near_raised_edge(self, pts, zone):
        """True where a point lies within `zone` of the boundary of a raised patch (a carpet tile's edge)."""
        pts = np.asarray(pts, dtype=float)
        out = np.zeros(len(pts), dtype=bool)
        for poly, _g, _pid, h in self.patches:
            if h <= 0.0:
                continue
            m = len(poly)
            A = poly
            B = np.roll(poly, -1, axis=0)
            for k in range(m):
                ex, ey = B[k, 0] - A[k, 0], B[k, 1] - A[k, 1]
                wx, wy = pts[:, 0] - A[k, 0], pts[:, 1] - A[k, 1]
                u = np.clip((wx * ex + wy * ey) / max(ex * ex + ey * ey, 1e-18), 0.0, 1.0)
                dx, dy = wx - u * ex, wy - u * ey
                out |= (dx * dx + dy * dy) <= zone * zone
        return out

    def raised_points(self, pts):
        """True where a point lies on a raised patch (a carpet tile with an edge)."""
        pts = np.asarray(pts, dtype=float)
        out = np.zeros(len(pts), dtype=bool)
        for poly, _g, _pid, h in self.patches:
            if h > 0.0:
                out |= points_in_polygon(pts, poly)
        return out


def piece_ends(piece, length):
    x, y, yaw = piece
    dx, dy = 0.5 * length * math.cos(yaw), 0.5 * length * math.sin(yaw)
    return (x - dx, y - dy), (x + dx, y + dy)


class Walls(object):
    """What the car must not touch, as built: capsules on the axes of the duct pieces (radius 0.165 m) and tape
    segments of the open walls (radius 0). Boxes are added as convex polygons."""

    def __init__(self, a, b, r, polys=()):
        self.a = np.asarray(a, dtype=float).reshape(-1, 2)
        self.b = np.asarray(b, dtype=float).reshape(-1, 2)
        self.r = np.asarray(r, dtype=float).reshape(-1)
        self.polys = [np.asarray(p, dtype=float) for p in polys]

    @classmethod
    def from_world(cls, world, moved=None, tapes=True, boxes=None, box_edges=False):
        """`moved`: [[duct id, piece, x, y, yaw], ...] (priv_state's moved_pieces) overrides the world file's poses;
        `boxes`: list of (pose, size) to add as polygons."""
        t = world["truth"]
        over = {}
        for row in moved or []:
            over[(str(row[0]), int(row[1]))] = (float(row[2]), float(row[3]), float(row[4]))
        a, b, r = [], [], []
        for d in t["ducts"]:
            c = float(d["piece_length_m"])
            for k, piece in enumerate(d["pieces"]):
                pc = over.get((d["id"], k), piece)
                ta, hb = piece_ends(pc, c)
                a.append(ta)
                b.append(hb)
                r.append(DUCT_RADIUS_M)
        if tapes:
            for o in t["open_walls"]:
                a.append(tuple(o["tape"][0]))
                b.append(tuple(o["tape"][1]))
                r.append(0.0)
        polys = []
        for pose, size in boxes or []:
            cs = box_corners(pose, size[0], size[1])
            polys.append(cs)
            if box_edges:
                for k in range(4):
                    a.append(cs[k])
                    b.append(cs[(k + 1) % 4])
                    r.append(0.0)
        return cls(a, b, r, polys)

    def free_along_normals(self, p, nl, reach):
        """Free distance from each point p[i] along +nl[i] (left) and -nl[i] (right) to the first capsule surface,
        tape segment or polygon edge, capped at `reach` (scalar or per point)."""
        p = np.asarray(p, dtype=float)
        nl = np.asarray(nl, dtype=float)
        n = len(p)
        reach = np.broadcast_to(np.asarray(reach, dtype=float), (n,)).copy()
        left = reach.copy()
        right = reach.copy()
        if len(self.a):
            mid = 0.5 * (self.a + self.b)
            half = 0.5 * np.hypot(*(self.b - self.a).T) + self.r
            chunk = 256
            for i0 in range(0, n, chunk):
                sl = slice(i0, min(n, i0 + chunk))
                for sign, out in ((1.0, left), (-1.0, right)):
                    t = _ray_capsules(p[sl], sign * nl[sl], self.a, self.b, self.r, mid, half, reach[sl])
                    out[sl] = np.minimum(out[sl], t)
        for poly in self.polys:
            for sign, out in ((1.0, left), (-1.0, right)):
                t = _ray_polygon(p, sign * nl, poly)
                np.minimum(out, t, out=out)
        return left, right

    def build_grid(self, cell=1.0):
        """A uniform grid of the segments' bounding boxes for the per-tick clearance query."""
        self.cell = float(cell)
        self.grid = {}
        lo = np.minimum(self.a, self.b) - self.r[:, None]
        hi = np.maximum(self.a, self.b) + self.r[:, None]
        for k in range(len(self.a)):
            for gx in range(int(math.floor(lo[k, 0] / cell)), int(math.floor(hi[k, 0] / cell)) + 1):
                for gy in range(int(math.floor(lo[k, 1] / cell)), int(math.floor(hi[k, 1] / cell)) + 1):
                    self.grid.setdefault((gx, gy), []).append(k)

    def clearance(self, poly, reach=0.6):
        """Smallest distance between a convex polygon (the car's envelope, corners in order) and the capsules and
        tape segments near it (negative = overlap); reach caps the search. Boxes are not included."""
        if not hasattr(self, "grid"):
            self.build_grid()
        q = np.asarray(poly, dtype=float)
        cx, cy = float(q[:, 0].mean()), float(q[:, 1].mean())
        rad = float(np.max(np.hypot(q[:, 0] - cx, q[:, 1] - cy))) + reach
        cell = self.cell
        ks = set()
        for gx in range(int(math.floor((cx - rad) / cell)), int(math.floor((cx + rad) / cell)) + 1):
            for gy in range(int(math.floor((cy - rad) / cell)), int(math.floor((cy + rad) / cell)) + 1):
                ks.update(self.grid.get((gx, gy), ()))
        if not ks:
            return reach
        idx = np.fromiter(sorted(ks), dtype=int)
        A, B, R = self.a[idx], self.b[idx], self.r[idx]
        best = poly_segment_distance(q, A, B) - R
        return float(min(reach, float(np.min(best))))


def _point_seg(px, py, A, B):
    """Distances from one point to segments A[i]-B[i]."""
    ex, ey = B[:, 0] - A[:, 0], B[:, 1] - A[:, 1]
    wx, wy = px - A[:, 0], py - A[:, 1]
    l2 = np.maximum(ex * ex + ey * ey, 1e-18)
    u = np.clip((wx * ex + wy * ey) / l2, 0.0, 1.0)
    dx, dy = wx - u * ex, wy - u * ey
    return np.sqrt(dx * dx + dy * dy)


def poly_segment_distance(q, A, B):
    """Distance between a convex polygon q (m, 2) and each segment A[i]-B[i] (0 when they intersect)."""
    m = len(q)
    d = np.full(len(A), np.inf)
    for i in range(m):                                   # polygon corners to the segments
        d = np.minimum(d, _point_seg(q[i, 0], q[i, 1], A, B))
    for i in range(m):                                   # segment ends to the polygon edges
        P0, P1 = q[i], q[(i + 1) % m]
        for E in (A, B):
            ex, ey = P1[0] - P0[0], P1[1] - P0[1]
            wx, wy = E[:, 0] - P0[0], E[:, 1] - P0[1]
            u = np.clip((wx * ex + wy * ey) / max(ex * ex + ey * ey, 1e-18), 0.0, 1.0)
            dx, dy = wx - u * ex, wy - u * ey
            d = np.minimum(d, np.sqrt(dx * dx + dy * dy))
    # a segment end inside the polygon, or a segment crossing it: distance 0
    inside = points_in_polygon(A, q) | points_in_polygon(B, q)
    cross = np.zeros(len(A), dtype=bool)
    for i in range(m):
        P0, P1 = q[i], q[(i + 1) % m]
        cross |= _segments_cross(P0, P1, A, B)
    d[inside | cross] = 0.0
    return d


def _segments_cross(P0, P1, A, B):
    def orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    o1 = orient(P0[0], P0[1], P1[0], P1[1], A[:, 0], A[:, 1])
    o2 = orient(P0[0], P0[1], P1[0], P1[1], B[:, 0], B[:, 1])
    o3 = orient(A[:, 0], A[:, 1], B[:, 0], B[:, 1], P0[0], P0[1])
    o4 = orient(A[:, 0], A[:, 1], B[:, 0], B[:, 1], P1[0], P1[1])
    return (o1 * o2 < 0.0) & (o3 * o4 < 0.0)


def car_envelope(pose, x_front=0.33446, x_rear=-0.0533, half_width=0.09246, grow=0.0):
    """Corners of the car's envelope at pose [x, y, yaw] of C (counter-clockwise), grown by `grow`."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    xf, xr, hw = x_front + grow, x_rear - grow, half_width + grow
    return [(x + c * a - s * b, y + s * a + c * b) for a, b in ((xf, hw), (xr, hw), (xr, -hw), (xf, -hw))]


def box_corners(pose, lx, ly):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in ((0.5 * lx, 0.5 * ly), (-0.5 * lx, 0.5 * ly), (-0.5 * lx, -0.5 * ly), (0.5 * lx, -0.5 * ly))]


def _ray_capsules(o, dirs, a, b, r, mid, half, reach):
    """For rays o[i] + t dirs[i] (unit dirs), the smallest t in [0, reach[i]] that enters any capsule (segment a-b,
    radius r); reach where none. Only capsules whose bounding circle can meet the ray segment are tested."""
    out = reach.copy()
    # candidate capsules per ray: distance of the capsule's centre from the ray segment's midpoint
    rm = o + dirs * (0.5 * reach)[:, None]
    dx = mid[None, :, 0] - rm[:, None, 0]
    dy = mid[None, :, 1] - rm[:, None, 1]
    cand = (dx * dx + dy * dy) <= (half[None, :] + 0.5 * reach[:, None]) ** 2
    rows, cols = np.nonzero(cand)
    if not len(rows):
        return out
    Org, D = o[rows], dirs[rows]
    A, B, R = a[cols], b[cols], r[cols]
    t = np.full(len(rows), np.inf)
    # end spheres (or points): |O + tD - C|^2 = R^2
    for C in (A, B):
        w = Org - C
        bq = np.einsum("ij,ij->i", w, D)
        cq = np.einsum("ij,ij->i", w, w) - R * R
        disc = bq * bq - cq
        ok = disc >= 0.0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t1 = -bq - sq
        t2 = -bq + sq
        inside = cq <= 0.0                      # the ray starts inside: t = 0
        cand_t = np.where(inside, 0.0, np.where(t1 >= 0.0, t1, np.where(t2 >= 0.0, t2, np.inf)))
        t = np.minimum(t, np.where(ok | inside, cand_t, np.inf))
    # cylinder (or the segment itself for R = 0)
    e = B - A
    L = np.hypot(e[:, 0], e[:, 1])
    e = e / np.maximum(L, 1e-12)[:, None]
    w = Org - A
    # components perpendicular to e
    De = np.einsum("ij,ij->i", D, e)
    we = np.einsum("ij,ij->i", w, e)
    Dp = D - De[:, None] * e
    wp = w - we[:, None] * e
    qa = np.einsum("ij,ij->i", Dp, Dp)
    qb = np.einsum("ij,ij->i", Dp, wp)
    qc = np.einsum("ij,ij->i", wp, wp) - R * R
    with np.errstate(invalid="ignore", divide="ignore"):
        disc = qb * qb - qa * qc
        ok = (disc >= 0.0) & (qa > 1e-14)
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t1 = (-qb - sq) / np.where(qa > 1e-14, qa, 1.0)
        t2 = (-qb + sq) / np.where(qa > 1e-14, qa, 1.0)
    for tc in (t1, t2):
        u = we + tc * De
        good = ok & (tc >= 0.0) & (u >= 0.0) & (u <= L)
        t = np.minimum(t, np.where(good, tc, np.inf))
    inside_cyl = (qc <= 0.0) & (we >= 0.0) & (we <= L)
    t = np.where(inside_cyl, 0.0, t)
    # R = 0: a segment crossing (the cylinder above degenerates); solve the 2-D line intersection
    seg0 = R <= 0.0
    if np.any(seg0):
        Os, Ds, As, Bs = Org[seg0], D[seg0], A[seg0], B[seg0]
        E = Bs - As
        den = Ds[:, 0] * E[:, 1] - Ds[:, 1] * E[:, 0]
        W = As - Os
        with np.errstate(invalid="ignore", divide="ignore"):
            tt = (W[:, 0] * E[:, 1] - W[:, 1] * E[:, 0]) / den
            uu = (W[:, 0] * Ds[:, 1] - W[:, 1] * Ds[:, 0]) / den
        good = (np.abs(den) > 1e-12) & (tt >= 0.0) & (uu >= 0.0) & (uu <= 1.0)
        t0 = t[seg0]
        t[seg0] = np.minimum(t0, np.where(good, tt, np.inf))
    np.minimum.at(out, rows, np.minimum(t, reach[rows]))
    return out


def _ray_polygon(o, dirs, poly):
    """Smallest t >= 0 at which rays o + t dirs cross an edge of the polygon (inf when none)."""
    v = np.asarray(poly, dtype=float)
    t = np.full(len(o), np.inf)
    inside = points_in_polygon(o, v)
    for i in range(len(v)):
        A, B = v[i], v[(i + 1) % len(v)]
        E = B - A
        den = dirs[:, 0] * E[1] - dirs[:, 1] * E[0]
        W = A[None, :] - o
        with np.errstate(invalid="ignore", divide="ignore"):
            tt = (W[:, 0] * E[1] - W[:, 1] * E[0]) / den
            uu = (W[:, 0] * dirs[:, 1] - W[:, 1] * dirs[:, 0]) / den
        good = (np.abs(den) > 1e-12) & (tt >= 0.0) & (uu >= 0.0) & (uu <= 1.0)
        t = np.minimum(t, np.where(good, tt, np.inf))
    t[inside] = 0.0
    return t
