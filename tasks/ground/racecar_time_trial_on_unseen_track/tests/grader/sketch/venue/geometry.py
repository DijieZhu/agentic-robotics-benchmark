"""Plane geometry of the venue generator: the skeleton polygon, its fillets, the dense centreline,
chord walks along polylines and a few distance helpers. Pure numpy / math, no randomness.

    poly = [Vtx(x, y, tag=None, radius=None), ...]            # a simple polygon, either orientation
    prims = fillet(poly, radii)                                # straights and arcs (tangent-continuous)
    cl = sample(prims, spacing)                                 # dense centreline samples with heading, curvature
"""

import math

import numpy as np


class Vtx(object):
    """A vertex of the skeleton polygon. `tag` marks vertices whose fillet radius is fixed by a feature
    ('spine_top', 'spine_tip'); `radius` is that fixed radius; `nid` the feature (notch) id."""

    __slots__ = ("x", "y", "tag", "radius", "nid")

    def __init__(self, x, y, tag=None, radius=None, nid=None):
        self.x, self.y, self.tag, self.radius, self.nid = float(x), float(y), tag, radius, nid

    def p(self):
        return np.array([self.x, self.y])

    def moved(self, x, y):
        return Vtx(x, y, self.tag, self.radius, self.nid)


def unit(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n)


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def turn(u1, u2):
    """Signed angle from direction u1 to direction u2 (counter-clockwise positive)."""
    return math.atan2(cross2(u1, u2), u1[0] * u2[0] + u1[1] * u2[1])


def signed_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def vtx_area(poly):
    return signed_area([(v.x, v.y) for v in poly])


def _seg_intersect(a, b, c, d, eps=1e-9):
    """True when segments ab and cd intersect (touching counts)."""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p, q, r):
        return min(p[0], q[0]) - eps <= r[0] <= max(p[0], q[0]) + eps and min(p[1], q[1]) - eps <= r[1] <= max(p[1], q[1]) + eps

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    if abs(o1) <= eps and on_seg(a, b, c):
        return True
    if abs(o2) <= eps and on_seg(a, b, d):
        return True
    if abs(o3) <= eps and on_seg(c, d, a):
        return True
    if abs(o4) <= eps and on_seg(c, d, b):
        return True
    return False


def seg_seg_dist(a, b, c, d):
    """Distance between segments ab and cd (0 when they intersect)."""
    if _seg_intersect(a, b, c, d, eps=0.0):
        return 0.0
    return min(point_seg_dist(a, c, d), point_seg_dist(b, c, d), point_seg_dist(c, a, b), point_seg_dist(d, a, b))


def point_seg_dist(p, a, b):
    ab = (b[0] - a[0], b[1] - a[1])
    L2 = ab[0] * ab[0] + ab[1] * ab[1]
    if L2 <= 1e-18:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / L2))
    return math.hypot(p[0] - a[0] - t * ab[0], p[1] - a[1] - t * ab[1])


def points_segs_dist(pts, a, b):
    """Distances (len(pts), len(a)) from points to segments a[k] -> b[k] (numpy, element-wise only)."""
    pts = np.asarray(pts, dtype=float)
    ab = b - a
    L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18)
    rel = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("pkj,kj->pk", rel, ab) / L2[None, :], 0.0, 1.0)
    foot = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.sqrt(np.sum((pts[:, None, :] - foot) ** 2, axis=2))


def polygon_is_simple(poly, min_gap=0.0):
    """True when no two non-adjacent edges intersect, every edge has length > 0, and (min_gap > 0) non-adjacent
    edges keep at least min_gap from each other."""
    pts = [(v.x, v.y) for v in poly]
    n = len(pts)
    if n < 3:
        return False
    for i in range(n):
        if math.dist(pts[i], pts[(i + 1) % n]) < 1e-6:
            return False
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if min_gap > 0.0:
                if seg_seg_dist(a, b, c, d) < min_gap:
                    return False
            elif _seg_intersect(a, b, c, d):
                return False
    return True


def drop_collinear(poly, tol=1e-9):
    """The polygon without vertices whose turning angle is zero (tags of dropped vertices are lost)."""
    out = list(poly)
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            u1, u2 = (b.x - a.x, b.y - a.y), (c.x - b.x, c.y - b.y)
            if abs(cross2(unit(u1), unit(u2))) < tol and u1[0] * u2[0] + u1[1] * u2[1] > 0 and b.tag is None:
                del out[i]
                changed = True
                break
    return out


def ray_hit(poly, p, d, skip=()):
    """Smallest positive distance from point p along unit direction d to an edge of the polygon (edges in `skip`,
    given by their start index, are ignored); inf when nothing is hit."""
    best = math.inf
    n = len(poly)
    for i in range(n):
        if i in skip:
            continue
        a, b = (poly[i].x, poly[i].y), (poly[(i + 1) % n].x, poly[(i + 1) % n].y)
        e = (b[0] - a[0], b[1] - a[1])
        den = cross2(d, e)
        if abs(den) < 1e-12:
            continue
        w = (a[0] - p[0], a[1] - p[1])
        t = cross2(w, e) / den
        u = cross2(w, d) / den
        if t > 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
            best = min(best, t)
    return best


# ------------------------------------------------------------------------------------------------ fillets
class Prim(object):
    """A primitive of the centreline: a straight (kind 'line': start point p0, unit direction u, length) or an arc
    (kind 'arc': centre c, radius r, start angle a0, signed sweep, length r |sweep|). `vertex` is the polygon vertex
    index of an arc, `edge` the edge index of a straight."""

    __slots__ = ("kind", "p0", "u", "length", "c", "r", "a0", "sweep", "vertex", "edge", "tag", "nid")

    def __init__(self, kind, length, **kw):
        self.kind, self.length = kind, float(length)
        self.p0 = self.u = self.c = None
        self.r = self.a0 = self.sweep = 0.0
        self.vertex = self.edge = None
        self.tag = self.nid = None
        for k, v in kw.items():
            setattr(self, k, v)

    def at(self, t):
        """(x, y, heading, curvature) at arc length t from the primitive's start."""
        if self.kind == "line":
            return self.p0[0] + t * self.u[0], self.p0[1] + t * self.u[1], math.atan2(self.u[1], self.u[0]), 0.0
        sgn = 1.0 if self.sweep > 0 else -1.0
        ang = self.a0 + sgn * t / self.r
        x, y = self.c[0] + self.r * math.cos(ang), self.c[1] + self.r * math.sin(ang)
        return x, y, ang + sgn * 0.5 * math.pi, sgn / self.r


def tangent_lengths(poly, radii):
    n = len(poly)
    out = []
    for k in range(n):
        a, b, c = poly[k - 1], poly[k], poly[(k + 1) % n]
        th = turn(unit((b.x - a.x, b.y - a.y)), unit((c.x - b.x, c.y - b.y)))
        out.append(radii[k] * math.tan(0.5 * abs(th)))
    return out


def turns(poly):
    n = len(poly)
    out = []
    for k in range(n):
        a, b, c = poly[k - 1], poly[k], poly[(k + 1) % n]
        out.append(turn(unit((b.x - a.x, b.y - a.y)), unit((c.x - b.x, c.y - b.y))))
    return out


def fit_radii(poly, wanted, fixed, iters=200):
    """Scale the wanted fillet radii down until the two tangent lengths on every edge fit into it. Fixed radii
    (fixed[k] True) are kept. Returns the radii or None when a fixed pair alone overfills an edge."""
    n = len(poly)
    th = turns(poly)
    tn = [math.tan(0.5 * abs(t)) for t in th]
    r = list(wanted)
    edge = [math.hypot(poly[(k + 1) % n].x - poly[k].x, poly[(k + 1) % n].y - poly[k].y) for k in range(n)]
    for _ in range(iters):
        ok = True
        for k in range(n):
            j = (k + 1) % n
            ta, tb = r[k] * tn[k], r[j] * tn[j]
            if ta + tb <= edge[k] * (1.0 + 1e-12):
                continue
            ok = False
            if fixed[k] and fixed[j]:
                return None
            if fixed[k]:
                r[j] = max(0.0, edge[k] - ta) / tn[j]
            elif fixed[j]:
                r[k] = max(0.0, edge[k] - tb) / tn[k]
            else:
                # take the excess from the larger tangent first, then from both equally
                ex = ta + tb - edge[k]
                if ta >= tb:
                    d1 = min(ex, ta - tb)
                    ta -= d1
                else:
                    d1 = min(ex, tb - ta)
                    tb -= d1
                ex -= d1
                ta, tb = max(0.0, ta - 0.5 * ex), max(0.0, tb - 0.5 * ex)
                r[k], r[j] = ta / tn[k], tb / tn[j]
        if ok:
            return r
    return None


def fillet(poly, radii):
    """Straights and arcs of the polygon with the given fillet radius at every vertex (tangent lengths must fit)."""
    n = len(poly)
    tl = tangent_lengths(poly, radii)
    th = turns(poly)
    prims = []
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        u = unit((b.x - a.x, b.y - a.y))
        p0 = (a.x + tl[k] * u[0], a.y + tl[k] * u[1])
        L = math.hypot(b.x - a.x, b.y - a.y) - tl[k] - tl[(k + 1) % n]
        prims.append(("line", k, p0, u, max(0.0, L)))
        # arc at vertex k+1
        j = (k + 1) % n
        v = poly[j]
        w = poly[(j + 1) % n]
        u2 = unit((w.x - v.x, w.y - v.y))
        s = (v.x - tl[j] * u[0], v.y - tl[j] * u[1])
        sgn = 1.0 if th[j] > 0 else -1.0
        nl = (-u[1], u[0])
        c = (s[0] + sgn * radii[j] * nl[0], s[1] + sgn * radii[j] * nl[1])
        a0 = math.atan2(s[1] - c[1], s[0] - c[0])
        prims.append(("arc", j, c, radii[j], a0, th[j], u2))
    out = []
    for item in prims:
        if item[0] == "line":
            _, k, p0, u, L = item
            out.append(Prim("line", L, p0=p0, u=u, edge=k))
        else:
            _, j, c, r, a0, sweep, _u2 = item
            out.append(Prim("arc", r * abs(sweep), c=c, r=r, a0=a0, sweep=sweep, vertex=j, tag=poly[j].tag, nid=poly[j].nid))
    # rotate so that the list starts with the straight of edge 0
    return out


class Centreline(object):
    """Dense samples of the filleted polygon: x, y, heading, curvature, station, primitive index and the
    primitive's local parameter. Consecutive samples are at most `spacing` apart."""

    def __init__(self, prims, spacing):
        xs, ys, hs, ks, ss, pi, pt = [], [], [], [], [], [], []
        s = 0.0
        self.prim_start = []
        for idx, p in enumerate(prims):
            self.prim_start.append(s)
            if p.length <= 1e-9:
                continue
            m = max(1, int(math.ceil(p.length / spacing - 1e-9)))
            for q in range(m):
                t = p.length * q / m
                x, y, h, k = p.at(t)
                xs.append(x)
                ys.append(y)
                hs.append(h)
                ks.append(k)
                ss.append(s + t)
                pi.append(idx)
                pt.append(t)
            s += p.length
        self.prims = prims
        self.length = s
        self.x, self.y = np.array(xs), np.array(ys)
        self.h, self.k, self.s = np.array(hs), np.array(ks), np.array(ss)
        self.prim, self.t = np.array(pi, dtype=int), np.array(pt)
        self.n = len(xs)
        self.tx, self.ty = np.cos(self.h), np.sin(self.h)
        self.nx, self.ny = -self.ty, self.tx                     # left normals

    def xy(self):
        return np.stack([self.x, self.y], axis=1)


def cyclic_ds(s, L):
    """Matrix of cyclic station distances between all samples."""
    d = np.abs(s[:, None] - s[None, :])
    return np.minimum(d, L - d)


# ------------------------------------------------------------------------------------------------ polylines
class Polyline(object):
    """A polyline with cumulative arc length; chord walks forward along it (contract-style piece laying)."""

    def __init__(self, pts):
        pts = np.asarray(pts, dtype=float)
        keep = [0]
        for i in range(1, len(pts)):
            if math.hypot(pts[i, 0] - pts[keep[-1], 0], pts[i, 1] - pts[keep[-1], 1]) > 1e-9:
                keep.append(i)
        self.p = pts[keep]
        seg = np.diff(self.p, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.u = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.u[-1])
        self.src = np.asarray(keep)

    def point(self, u):
        u = min(max(u, 0.0), self.length)
        i = int(np.searchsorted(self.u, u, side="right") - 1)
        i = min(max(i, 0), len(self.seg_len) - 1)
        f = (u - self.u[i]) / self.seg_len[i] if self.seg_len[i] > 0 else 0.0
        a, b = self.p[i], self.p[i + 1]
        return (float(a[0] + f * (b[0] - a[0])), float(a[1] + f * (b[1] - a[1])))

    def advance(self, u, chord):
        """Smallest u2 > u with |P(u2) - P(u)| = chord (None when the polyline ends first)."""
        if chord <= 0.0:
            return u
        p0 = self.point(u)
        i = int(np.searchsorted(self.u, u, side="right") - 1)
        i = min(max(i, 0), len(self.seg_len) - 1)
        c2 = chord * chord
        while i < len(self.seg_len):
            a = self.p[i]
            b = self.p[i + 1]
            if float((b[0] - p0[0]) ** 2 + (b[1] - p0[1]) ** 2) >= c2:
                # solve |a + t (b - a) - p0|^2 = chord^2 on this segment, the root after u
                dx, dy = b[0] - a[0], b[1] - a[1]
                wx, wy = a[0] - p0[0], a[1] - p0[1]
                A = dx * dx + dy * dy
                B = 2.0 * (wx * dx + wy * dy)
                C = wx * wx + wy * wy - c2
                disc = max(0.0, B * B - 4.0 * A * C)
                t = (-B + math.sqrt(disc)) / (2.0 * A)
                t = min(max(t, 0.0), 1.0)
                u2 = float(self.u[i] + t * self.seg_len[i])
                return max(u2, u)
            i += 1
        return None

    def retreat(self, u, chord):
        """Largest u2 < u with |P(u2) - P(u)| = chord (None when the polyline starts first)."""
        if chord <= 0.0:
            return u
        p0 = self.point(u)
        i = int(np.searchsorted(self.u, u, side="left") - 1)
        i = min(max(i, 0), len(self.seg_len) - 1)
        c2 = chord * chord
        while i >= 0:
            a = self.p[i + 1]
            b = self.p[i]
            if float((b[0] - p0[0]) ** 2 + (b[1] - p0[1]) ** 2) >= c2:
                dx, dy = b[0] - a[0], b[1] - a[1]
                wx, wy = a[0] - p0[0], a[1] - p0[1]
                A = dx * dx + dy * dy
                B = 2.0 * (wx * dx + wy * dy)
                C = wx * wx + wy * wy - c2
                disc = max(0.0, B * B - 4.0 * A * C)
                t = (-B + math.sqrt(disc)) / (2.0 * A)
                t = min(max(t, 0.0), 1.0)
                u2 = float(self.u[i + 1] - t * self.seg_len[i])
                return min(u2, u)
            i -= 1
        return None


def rect_corners(cx, cy, yaw, lx, ly):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + c * dx - s * dy, cy + s * dx + c * dy)
            for dx, dy in ((0.5 * lx, 0.5 * ly), (-0.5 * lx, 0.5 * ly), (-0.5 * lx, -0.5 * ly), (0.5 * lx, -0.5 * ly))]


def wrap(a):
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if a == -math.pi else a
