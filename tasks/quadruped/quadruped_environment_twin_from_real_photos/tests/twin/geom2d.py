"""Plane geometry of the site frame's xy: polygons (region, zones) and footprint boxes. numpy only.

    points_in_polygon(xy, poly) -> bool (n,)        crossing-number test; a point exactly on an edge is undefined
    polygon_distance(xy, poly) -> (n,)              distance to the polygon's boundary (0 inside is NOT implied)
    polygon_area(poly) -> float                     absolute area
    convex_hull_2d(xy) -> (k, 2)                    counter-clockwise hull (monotone chain), duplicates removed
    min_area_rect(xy) -> dict                       the minimum-area enclosing rectangle (rotating calipers):
                                                    {"centre" (2,), "axes" (2, 2) columns, "half" (2,), "corners" (4, 2)}
"""
import numpy as np


def _as_xy(xy):
    return np.atleast_2d(np.asarray(xy, float))[:, :2]


def points_in_polygon(xy, poly):
    P = _as_xy(xy)
    V = np.asarray(poly, float).reshape(-1, 2)
    inside = np.zeros(len(P), bool)
    n = len(V)
    if n < 3:
        return inside
    x, y = P[:, 0], P[:, 1]
    for i in range(n):
        x1, y1 = V[i]
        x2, y2 = V[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < xc)
    return inside


def polygon_distance(xy, poly):
    P = _as_xy(xy)
    V = np.asarray(poly, float).reshape(-1, 2)
    best = np.full(len(P), np.inf)
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0) if L2 > 0 else np.zeros(len(P))
        d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best


def polygon_area(poly):
    V = np.asarray(poly, float).reshape(-1, 2)
    x, y = V[:, 0], V[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def convex_hull_2d(xy):
    P = np.unique(_as_xy(xy), axis=0)              # sorted by x, then y: deterministic
    if len(P) < 3:
        return P
    if len(P) > 64:
        from scipy.spatial import ConvexHull, QhullError
        try:
            h = ConvexHull(P)
            return P[h.vertices]                   # counter-clockwise in 2-D
        except QhullError:
            pass                                   # degenerate (collinear): the monotone chain below

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], float)


def min_area_rect(xy):
    """The minimum-area rectangle enclosing the points (over the hull's edge directions; ties: the first edge)."""
    Hh = convex_hull_2d(xy)
    if len(Hh) == 0:
        raise ValueError("no points")
    if len(Hh) < 3:
        c = Hh.mean(axis=0)
        d = Hh[-1] - Hh[0]
        L = float(np.linalg.norm(d))
        u = d / L if L > 0 else np.array([1.0, 0.0])
        A = np.c_[u, [-u[1], u[0]]]
        half = np.array([L / 2.0, 0.0])
        return {"centre": c, "axes": A, "half": half, "corners": _corners(c, A, half)}
    if len(Hh) > 2000:                             # thin a dense round hull (an inner bound off by < 1e-5 m)
        Hh = Hh[::int(np.ceil(len(Hh) / 2000.0))]
    best = None
    E = np.roll(Hh, -1, axis=0) - Hh
    L = np.linalg.norm(E, axis=1)
    ok = np.nonzero(L > 0)[0]
    # one candidate per 0.05 deg bin of the edge angle modulo 90 deg (the first edge of each bin, exactly): a round
    # footprint has thousands of hull edges, and the rectangle cannot change by more than the bin within a bin
    ang = np.degrees(np.arctan2(E[ok, 1], E[ok, 0])) % 90.0
    _b, first = np.unique(np.floor(ang / 0.05).astype(np.int64), return_index=True)
    for i in ok[np.sort(first)]:
        u = E[i] / L[i]
        A = np.c_[u, [-u[1], u[0]]]
        Q = Hh @ A
        lo, hi = Q.min(axis=0), Q.max(axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0] - 1e-12:
            best = (area, A, lo, hi)
    _area, A, lo, hi = best
    half = 0.5 * (hi - lo)
    c = A @ (0.5 * (hi + lo))
    return {"centre": c, "axes": A, "half": half, "corners": _corners(c, A, half)}


def _corners(c, A, half):
    s = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    return c + (s * half) @ A.T
