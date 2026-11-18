"""Criterion G, functional geometry (weight 0.10): what the strut works by, measured on the submission's
own geometry as relations between its parts.

Principles:
  1. Measured on the submission. Each reference primitive (tests/reference/features.json functional.primitives, in
     the design file's frame) is carried onto the submitted part by that part's own fit (inv(T_fit)); the part's own
     primitive of the same kind is then searched near it, within the capture range (axis or centre within
     functional.capture_mm = 3 mm of the carried one, radius within functional.capture_radius_rel = 50 %). Every
     surface kind (analytic, B-spline, faceted) goes the same way: least squares on the part's dense surface samples
     (one code path for every kind of submission, rather than reading analytic
     faces directly). A winch's drum comes from its radial profile about its own bore axis. A
     cylinder whose last search window is not round as a whole (a flat, a key or another surface beside the wall
     within the window: a D-shaped or keyed bore) is fitted on its own wall, the points whose normals are radial and
     that lie on one circle, when they are the larger part of the window (cylinder_wall).
  2. Relative quantities only: sphere centre to sphere centre, axis to axis, turn about the rod relative to the
     electronics datum, all in the submission's own frame. The whole-strut alignment is used only to know which
     submitted part plays which reference instance (and, when neither electronics half is fitted, which end is +z).
  3. A quantity whose primitive is not found scores 0 for that quantity only.

The rod axis is the recognised rod's axis; without a rod, the line through the two end-cap bore axes (their midpoints).
The strut's +z sense is the one the fitted electronics halves (P01, P02) carry; the electronics datum is the common
azimuth (mod 180 deg) of their radial-hole direction (features.json F-EA / F-EB electronics_datum_axis), from one of
them when only one is fitted; a half whose chosen pose's shape score is below functional.datum_shape_min (0.2, as
align.shape_min) gives neither; without either, the four datum quantities score 0.

Measurement conventions are those of the reference values:
azimuths about the rod axis; the post pattern's phase is the circular mean of the four posts modulo 90 deg, folded to
(-45, 45]; the drum is the contiguous band of the radial profile within drum_delta_mm (0.01 mm) of the median outer
radius over the middle 40 % of the winch's axial range; axial distances are absolute values along the rod axis; a
motor enclosure's middle is the middle of its extent along its own rod bore. The radial-profile and drum-band code
below is copied from the reference tools (radial_profile, edge_samples, drum_band, fit_circle_2d, fit_sphere,
perp_basis) so that the submission side and the reference values are measured alike.

Items (grader_config.json functional.items; weight = the item's share of G over all its instances, split equally;
G1 0.20, G2 0.15, G3 0.15, G4 0.15, G5 0.35, the five G5 angle sub-items 0.05 each):
  node_sphere_centre_distance (P12, P13), node_sphere_radius (each cover), winch_axis_offset_from_rod_axis (each winch),
  drum_mid_to_node_centre_axial (each winch with its end's cover), drum_diameter (each winch),
  post_mid_to_node_centre_axial (each of the 8 posts with its end's cover), post_axis_offset_from_cap_bore_axis (each
  post), winch_azimuth_minus_electronics_datum_mod180 (each winch), twist_between_winches,
  post_pattern_phase_minus_electronics_datum_mod90 (each end cap), motor_encl_A_mid_to_rod_mid_axial (each P04 / P05).
"""
import math

import numpy as np

from .geom import band, inv
from .primitives import HOLE_RULE
from .recognise import submission_geo

NAME = "functional"


# ============================================================================ small geometry (as in the reference tools)
def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def perp_basis(d):
    d = d / np.linalg.norm(d)
    h = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, h)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return e1, e2


def fit_circle_2d(x, y):
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    res = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx, cy, r, res


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P * P).sum(1), rcond=None)
    c = sol[:3]
    r2 = sol[3] + c @ c
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    res = np.abs(np.linalg.norm(P - c, axis=1) - r)
    return {"centre": c, "radius": r, "res": res}


def edge_samples(V, F, step_mm):
    """deterministic points along every mesh edge, at most step_mm apart"""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    a, b = V[e[:, 0]], V[e[:, 1]]
    n = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / step_mm).astype(np.int64))
    idx = np.repeat(np.arange(len(e)), n + 1)
    k = np.concatenate([np.arange(m + 1) for m in n.tolist()])
    t = (k / np.repeat(n, n + 1))[:, None]
    return a[idx] * (1 - t) + b[idx] * t


def radial_profile(V, F, point, d, bin_mm=0.01):
    """outer radius r(t) about an axis: max distance from the axis of edge points in each axial bin of width bin_mm"""
    d = np.asarray(d, float) / np.linalg.norm(d)
    Q = edge_samples(V, F, 0.5 * bin_mm) - point
    t = Q @ d
    rho = np.linalg.norm(Q - np.outer(t, d), axis=1)
    lo = math.floor(t.min() / bin_mm) * bin_mm
    nb = int(math.ceil((t.max() - lo) / bin_mm)) + 1
    k = np.clip(((t - lo) / bin_mm).astype(np.int64), 0, nb - 1)
    r = np.full(nb, -1.0)
    np.maximum.at(r, k, rho)
    tc = lo + (np.arange(nb) + 0.5) * bin_mm
    ok = r >= 0
    return tc[ok], r[ok]


def drum_band(t, r, delta_mm=0.01, middle_share=0.4):
    """drum radius = median outer radius over the middle middle_share of the axial range; the band = the contiguous run
    around the middle where r <= drum radius + delta_mm, its ends interpolated. Returns (drum radius, t_lo, t_hi)."""
    tmid = 0.5 * (t.min() + t.max())
    half = 0.5 * middle_share * (t.max() - t.min())
    core = np.abs(t - tmid) <= half
    rd = float(np.median(r[core]))
    lim = rd + delta_mm
    i0 = int(np.argmin(np.abs(t - tmid)))

    def walk(step):
        i = i0
        while 0 <= i + step < len(t) and r[i + step] <= lim:
            i += step
        j = i + step
        if not (0 <= j < len(t)):
            return float(t[i])
        f = (lim - r[i]) / (r[j] - r[i]) if r[j] != r[i] else 0.0
        return float(t[i] + f * (t[j] - t[i]))

    return rd, walk(-1), walk(1)


def line_point_distance(p, d, q):
    v = np.asarray(q, float) - np.asarray(p, float)
    return float(np.linalg.norm(v - (v @ d) * d))


def wrap(a, period):
    v = math.fmod(a, period)
    return v + period if v < 0 else v


def ang_diff(a, b, period):
    d = wrap(a - b, period)
    return min(d, period - d)


def fold(a, period):
    """into (-period / 2, period / 2]"""
    v = wrap(a, period)
    return v - period if v > 0.5 * period else v


# ============================================================================ primitive search on a submitted part
def find_sphere(P, N, c0, R0, F):
    """The part's own convex sphere near (c0, R0): selection by radius window and outward radial normal, algebraic
    least squares, the window narrowed in functional.windows_mm. None when not found (module docstring)."""
    cap, rel = float(F["capture_mm"]), float(F["capture_radius_rel"])
    pre = np.abs(np.linalg.norm(P - c0, axis=1) - R0) <= rel * R0 + cap
    P1, N1 = P[pre], N[pre]
    c, R = np.asarray(c0, float), float(R0)
    sel = None
    for win in [rel * R0] + [float(w) for w in F["windows_mm"]]:
        v = P1 - c
        r = np.linalg.norm(v, axis=1)
        u = v / np.maximum(r, 1e-12)[:, None]
        cosn = np.einsum("ij,ij->i", N1, u)
        sel = (np.abs(r - R) <= win) & (cosn >= float(F["normal_cos"]))
        if int(sel.sum()) < int(F["min_points"]):
            return None
        s = fit_sphere(P1[sel])
        if s is None:
            return None
        c, R = s["centre"], float(s["radius"])
        if float(np.linalg.norm(c - c0)) > cap or abs(R - R0) > rel * R0:
            return None
    res = np.abs(np.linalg.norm(P1[sel] - c, axis=1) - R)
    rms = float(np.sqrt(np.mean(res ** 2)))
    ev = np.linalg.eigvalsh(N1[sel].T @ N1[sel] / float(sel.sum()))
    if rms > max(float(F["rms_mm"]), float(F["rms_rel"]) * R) or ev[0] < float(F["sphere_spread"][0]) or \
            ev[1] < float(F["sphere_spread"][1]):
        return None
    return {"centre": c, "radius": R, "rms_mm": rms, "n": int(sel.sum())}


def cylinder_wall(Ps, Ns, p, d, p0, sgn, F):
    """The points of a cylinder's own wall among the last window's points (Ps, Ns) when the window as a whole is not
    round (the bore of a D-shaped, keyed or flatted hole): the points whose normal is within half of the
    holes criterion's smoothness angle (primitives.HOLE_RULE smooth_deg: two neighbouring facets of one smooth bore
    turn by less than that) of the radial direction about the axis (p, d), the circle refitted on them in the plane
    across d, then of those the points within the roundness limit (max(functional.rms_mm, functional.rms_rel x R)) of
    that circle, refitted once more. They stand for the cylinder only when they are more than half of the window's
    points and at least functional.min_points: beside a flat or a key the wall is the larger part of the window; on a
    hole of a few flat sides, a square or a slot only the narrow strips where a flat is nearly tangent to a circle
    pass, the smaller part. Returns (the wall's points, the axis point nearest p0, the radius) or None."""
    cmin = math.cos(math.radians(0.5 * float(HOLE_RULE["smooth_deg"])))
    e1, e2 = perp_basis(d)

    def radial(c):
        v = Ps - c
        rad = v - np.outer(v @ d, d)
        r = np.linalg.norm(rad, axis=1)
        return r, sgn * np.einsum("ij,ij->i", Ns, rad / np.maximum(r, 1e-12)[:, None])

    def circle(keep):
        if int(keep.sum()) < int(F["min_points"]):
            return None
        got = fit_circle_2d(Ps[keep] @ e1, Ps[keep] @ e2)
        if got is None:
            return None
        centre = got[0] * e1 + got[1] * e2
        return centre + d * float((p0 - centre) @ d), float(got[2])

    keep = radial(p)[1] >= cmin
    got = circle(keep)
    if got is None:
        return None
    c, R = got
    keep &= np.abs(radial(c)[0] - R) <= max(float(F["rms_mm"]), float(F["rms_rel"]) * R)
    if not 2 * int(keep.sum()) > len(Ps):
        return None
    got = circle(keep)
    if got is None:
        return None
    return Ps[keep], got[0], got[1]


def find_cylinder(P, N, p0, d0, R0, half, concave, F):
    """The part's own concave (a bore) or convex (a post) cylinder near the carried axis (p0, d0) of radius R0, within
    +-half of p0 along the axis: selection by radius window, normal toward (concave) or away from (convex) the axis and
    normal across the axis; the axis direction from the normals, the centre and radius from a circle fit in the plane
    across it; the window narrowed in functional.windows_mm. When the last window's points are not round as a whole
    (rms above the roundness limit), the cylinder's own wall among them (cylinder_wall) is fitted instead, or the
    cylinder is not found. Returns the axis point nearest p0, the direction (sign of d0), the radius, the
    axial extent (t_lo, t_hi about that point) and the arc it covers; None when not found."""
    cap, rel = float(F["capture_mm"]), float(F["capture_radius_rel"])
    sgn = -1.0 if concave else 1.0
    p0, d0 = np.asarray(p0, float), _unit(d0)
    v = P - p0
    t = v @ d0
    rr = np.linalg.norm(v - np.outer(t, d0), axis=1)
    pre = (np.abs(t) <= half + cap) & (rr <= cap + (1.0 + rel) * R0)
    P1, N1 = P[pre], N[pre]
    p, d, R = p0.copy(), d0.copy(), float(R0)
    sin_max = math.sin(math.radians(float(F["across_deg"])))
    sel = None
    for win in [rel * R0] + [float(w) for w in F["windows_mm"]]:
        v = P1 - p
        t = v @ d
        rad = v - np.outer(t, d)
        r = np.linalg.norm(rad, axis=1)
        u = rad / np.maximum(r, 1e-12)[:, None]
        cosn = sgn * np.einsum("ij,ij->i", N1, u)
        tc = float((p0 - p) @ d)
        sel = (np.abs(t - tc) <= half) & (np.abs(r - R) <= win) & (cosn >= float(F["normal_cos"])) & \
              (np.abs(N1 @ d) <= sin_max)
        if int(sel.sum()) < int(F["min_points"]):
            return None
        Ns = N1[sel]
        _ev, E = np.linalg.eigh(Ns.T @ Ns)
        dn = E[:, 0]
        if float(dn @ d0) < 0:
            dn = -dn
        if math.degrees(math.acos(min(1.0, abs(float(dn @ d0))))) > float(F["max_tilt_deg"]):
            return None
        e1, e2 = perp_basis(dn)
        Ps = P1[sel]
        got = fit_circle_2d(Ps @ e1, Ps @ e2)
        if got is None:
            return None
        cx, cy, R, _res = got
        centre = cx * e1 + cy * e2
        d = dn
        p = centre + d * float((p0 - centre) @ d)
        if line_point_distance(p, d, p0) > cap or abs(R - R0) > rel * R0:
            return None
    Ps = P1[sel]
    v = Ps - p
    t = v @ d
    rad = v - np.outer(t, d)
    res = np.abs(np.linalg.norm(rad, axis=1) - R)
    rms = float(np.sqrt(np.mean(res ** 2)))
    if rms > max(float(F["rms_mm"]), float(F["rms_rel"]) * R):
        # not round as a whole: a flat, key or other surface beside the wall inside the window (cylinder_wall)
        got = cylinder_wall(Ps, N1[sel], p, d, p0, sgn, F)
        if got is None:
            return None
        Ps, p, R = got
        if line_point_distance(p, d, p0) > cap or abs(R - R0) > rel * R0:
            return None
        v = Ps - p
        t = v @ d
        rad = v - np.outer(t, d)
        res = np.abs(np.linalg.norm(rad, axis=1) - R)
        rms = float(np.sqrt(np.mean(res ** 2)))
    e1, e2 = perp_basis(d)
    az = np.degrees(np.arctan2(rad @ e2, rad @ e1)) % 360.0
    nb = int(round(360.0 / float(F["arc_bin_deg"])))
    arc = float(len(np.unique(np.minimum((az / float(F["arc_bin_deg"])).astype(int), nb - 1)))) * float(F["arc_bin_deg"])
    if rms > max(float(F["rms_mm"]), float(F["rms_rel"]) * R) or arc < float(F["min_arc_deg"]):
        return None
    t_lo, t_hi = axial_run(t, float((p0 - p) @ d), float(F["extent_bin_mm"]), float(F["extent_share"]))
    return {"point": p, "dir": d, "radius": float(R), "rms_mm": rms, "n": int(len(Ps)), "t_lo": t_lo,
            "t_hi": t_hi, "arc_deg": arc}


def axial_run(t, tc, b, share):
    """The axial extent of a cylinder's inliers: the contiguous run of axial bins (width b) around the bin nearest tc
    whose counts are at least share x the largest count (partial arcs of neighbouring surfaces that happen to lie on
    the same radius are thinner and stay out), then the smallest and largest inlier position inside that run."""
    lo = math.floor(float(t.min()) / b) * b
    k = np.floor((t - lo) / b).astype(np.int64)
    cnt = np.bincount(k)
    ok = cnt >= share * float(cnt.max())
    good = np.nonzero(ok)[0]
    i0 = int(np.clip(math.floor((tc - lo) / b), 0, len(cnt) - 1))
    i0 = int(good[np.argmin(np.abs(good - i0))])
    a = c = i0
    while a - 1 >= 0 and ok[a - 1]:
        a -= 1
    while c + 1 < len(cnt) and ok[c + 1]:
        c += 1
    inside = (k >= a) & (k <= c)
    return float(t[inside].min()), float(t[inside].max())


# ============================================================================ the item
class _Measure(object):
    """Primitives of the submitted parts (cached per solid and primitive) and the strut's datum quantities."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.F = ctx.cfg["functional"]
        self.prims = (ctx.ref.features or {}).get("functional", {}).get("primitives", {})
        self.al = ctx.alignment or {}
        self.asg = self.al.get("assignment", {})
        self.cache = ctx.cache.setdefault(NAME, {})
        self.notes = []

    # ------------------------------------------------------------------ parts and their fits
    def sid(self, tag):
        return self.asg.get(tag)

    def to_sub(self, tag):
        """design frame -> submission frame of the part playing reference instance `tag` (inv of its chosen fit)."""
        sid = self.sid(tag)
        if sid is None or sid not in self.al.get("T_fit", {}):
            return None
        return inv(self.al["T_fit"][sid])

    def prim(self, tag, name):
        fam = self.ctx.ref.family_of[tag]
        for p in self.prims.get(fam, []):
            if p["name"] == name:
                return p
        raise KeyError("no primitive %s in design %s" % (name, fam))

    def samples(self, tag):
        g = submission_geo(self.ctx, self.ctx.solid(self.sid(tag)))
        if g is None:
            return None
        P, N, _tree = g.dense()
        return P, N

    def found(self, tag, name):
        """The submitted part's own primitive `name` (cached), or None."""
        key = (tag, self.sid(tag), name)
        if key in self.cache:
            return self.cache[key]
        out = None
        M = self.to_sub(tag)
        if M is not None:
            pr = self.prim(tag, name)
            S = self.samples(tag)
            if S is not None:
                P, N = S
                R3, t3 = M[:3, :3], M[:3, 3]
                if pr["kind"] == "sphere":
                    out = find_sphere(P, N, R3 @ np.asarray(pr["centre_mm"], float) + t3, float(pr["radius_mm"]), self.F)
                elif pr["kind"] in ("concave_cylinder", "convex_cylinder"):
                    out = find_cylinder(P, N, R3 @ np.asarray(pr["axis_point_mm"], float) + t3,
                                        R3 @ np.asarray(pr["axis_dir"], float), float(pr["radius_mm"]),
                                        self.half_window(pr), pr["kind"] == "concave_cylinder", self.F)
        if out is None:
            self.notes.append("%s: %s not found" % (tag, name))
        self.cache[key] = out
        return out

    def half_window(self, pr):
        """Half the axial search window of a cylinder primitive about its carried axis point: half its reference
        length (posts), half its axial extent (the motor enclosure bore), else functional.half_window_mm[name]."""
        m = float(self.F["axial_margin_mm"])
        if "length_mm" in pr:
            return 0.5 * float(pr["length_mm"]) + m
        if "axial_extent_mm" in pr:
            return 0.5 * float(pr["axial_extent_mm"][1] - pr["axial_extent_mm"][0]) + m
        return float(self.F["half_window_mm"][pr["name"]]) + m

    def drum(self, tag):
        """(drum radius, drum mid point on the winch axis) from the radial profile about the winch's own bore, or None."""
        key = (tag, self.sid(tag), "drum")
        if key in self.cache:
            return self.cache[key]
        out = None
        bore = self.found(tag, "winch_bore_axis")
        M = self.to_sub(tag)
        if bore is not None and M is not None:
            g = submission_geo(self.ctx, self.ctx.solid(self.sid(tag)))
            pr = self.prim(tag, "winch_drum")
            p, d = bore["point"], bore["dir"]
            t, r = radial_profile(g.V, g.F, p, d, float(self.F["profile_bin_mm"]))
            rd, tlo, thi = drum_band(t, r, float(self.F["drum_delta_mm"]), float(self.F["drum_middle_share"]))
            mid = p + d * 0.5 * (tlo + thi)
            pred = M[:3, :3] @ np.asarray(pr["axis_point_mm"], float) + M[:3, 3]
            R0 = float(pr["drum_radius_mm"])
            if thi > tlo and abs(rd - R0) <= float(self.F["capture_radius_rel"]) * R0 and \
                    abs(float((mid - pred) @ d)) <= float(self.F["capture_mm"]):
                out = {"radius": rd, "mid": mid, "width": thi - tlo}
        if out is None:
            self.notes.append("%s: drum not found" % tag)
        self.cache[key] = out
        return out

    # ------------------------------------------------------------------ the strut's frame of reference
    def datum_fit(self, tag):
        """inv(T_fit) of an electronics half that may give the datum: fitted, and its chosen pose's shape score at least
        functional.datum_shape_min (a part reduced to its centroid in the alignment, e.g. a convex hull, has no say in
        the turn either: align.shape_min); else None."""
        M = self.to_sub(tag)
        if M is None:
            return None
        sid = self.sid(tag)
        c = self.ctx.fits[sid]["candidates"][self.al["chosen"][sid]]
        return M if float(c["shape"]) >= float(self.F["datum_shape_min"]) else None

    def plus_z(self):
        """The strut's +z direction as the fitted electronics halves carry it (else the alignment's)."""
        z = np.zeros(3)
        for tag in ("P01", "P02"):
            M = self.datum_fit(tag)
            if M is not None:
                W = self.ctx.ref.instances[tag]["W"]
                z += M[:3, :3] @ (W[:3, :3].T @ np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(z) < 1e-9:
            z = inv(self.al["A"])[:3, :3] @ np.array([0.0, 0.0, 1.0])
        return _unit(z)

    def rod(self):
        """(point, unit direction toward +z, midpoint or None) of the rod axis in the submission frame, or None."""
        if "rod" in self.cache:
            return self.cache["rod"]
        out = None
        zp = self.plus_z()
        pa = self.al.get("purchased_assignment", {})
        for pid in sorted(self.ctx.ref.placements):
            pl = self.ctx.ref.placements[pid]
            if pl["type"] != "ROD" or pa.get(pid) is None:
                continue
            it = self.ctx.instances[pa[pid]]
            sym = self.ctx.ref.types["ROD"]["symmetry"]
            W = it["W"]
            p = W[:3, :3] @ sym["axis_point"] + W[:3, 3]
            d = _unit(W[:3, :3] @ sym["axis_dir"])
            out = (p, d if float(d @ zp) >= 0 else -d, p, "rod")
            break
        if out is None:
            a = self.found("P10", "cap_bore_axis") if self.sid("P10") is not None else None
            b = self.found("P11", "cap_bore_axis") if self.sid("P11") is not None else None
            if a is not None and b is not None:
                pa_, pb_ = self._mid_on_axis(a), self._mid_on_axis(b)
                d = _unit(pa_ - pb_)
                out = (0.5 * (pa_ + pb_), d if float(d @ zp) >= 0 else -d, 0.5 * (pa_ + pb_), "end-cap bores")
            elif a is not None or b is not None:
                c = a if a is not None else b
                d = c["dir"]
                out = (c["point"], d if float(d @ zp) >= 0 else -d, None, "one end-cap bore")
        if out is None:
            self.notes.append("no rod axis: neither a recognised rod nor an end-cap bore")
        self.cache["rod"] = out
        return out

    @staticmethod
    def _mid_on_axis(c):
        return c["point"] + c["dir"] * 0.5 * (c["t_lo"] + c["t_hi"])

    def azimuth(self, v):
        """azimuth (deg, [0, 360)) of a vector about the rod axis (submission frame; the zero is arbitrary, only
        differences are used)."""
        rod = self.rod()
        d = rod[1]
        e1, e2 = perp_basis(d)
        w = np.asarray(v, float) - (np.asarray(v, float) @ d) * d
        return wrap(math.degrees(math.atan2(float(w @ e2), float(w @ e1))), 360.0)

    def datum(self):
        """The electronics datum azimuth (mod 180) about the rod axis: the circular mean of the fitted halves'."""
        if "datum" in self.cache:
            return self.cache["datum"]
        z, n = 0j, 0
        for tag, key in (("P01", "F-EA"), ("P02", "F-EB")):
            M = self.datum_fit(tag)
            if M is None:
                continue
            pr = [p for p in self.prims.get(key, []) if p["kind"] == "direction"][0]
            a = self.azimuth(M[:3, :3] @ np.asarray(pr["axis_dir"], float))
            z += complex(math.cos(math.radians(2 * a)), math.sin(math.radians(2 * a)))
            n += 1
        out = wrap(math.degrees(math.atan2(z.imag, z.real)) / 2.0, 180.0) if n else None
        if out is None:
            self.notes.append("no electronics datum: neither P01 nor P02 is fitted")
        self.cache["datum"] = out
        self.cache["datum_parts"] = n
        return out


def _measure_item(m, name, inst, k):
    """The measured value of instance `inst` (its tags, from features.json) of item `name`, or None."""
    rod = m.rod()
    if name == "node_sphere_centre_distance":
        a, b = (m.found(t, "cover_outer_sphere") if m.sid(t) is not None else None for t in inst)
        return None if a is None or b is None else float(np.linalg.norm(a["centre"] - b["centre"]))
    if name == "node_sphere_radius":
        a = m.found(inst[0], "cover_outer_sphere") if m.sid(inst[0]) is not None else None
        return None if a is None else a["radius"]
    if rod is None:
        return None
    if name == "winch_axis_offset_from_rod_axis":
        w = m.found(inst[0], "winch_bore_axis") if m.sid(inst[0]) is not None else None
        return None if w is None else line_point_distance(rod[0], rod[1], m._mid_on_axis(w))
    if name == "drum_mid_to_node_centre_axial":
        dr = m.drum(inst[0]) if m.sid(inst[0]) is not None else None
        nd = m.found(inst[1], "cover_outer_sphere") if m.sid(inst[1]) is not None else None
        return None if dr is None or nd is None else abs(float((dr["mid"] - nd["centre"]) @ rod[1]))
    if name == "drum_diameter":
        dr = m.drum(inst[0]) if m.sid(inst[0]) is not None else None
        return None if dr is None else 2.0 * dr["radius"]
    if name in ("post_mid_to_node_centre_axial", "post_axis_offset_from_cap_bore_axis"):
        post = "post_%d" % (k % 4 + 1)
        ps = m.found(inst[0], post) if m.sid(inst[0]) is not None else None
        if ps is None:
            return None
        if name == "post_mid_to_node_centre_axial":
            nd = m.found(inst[1], "cover_outer_sphere") if m.sid(inst[1]) is not None else None
            return None if nd is None else abs(float((m._mid_on_axis(ps) - nd["centre"]) @ rod[1]))
        bore = m.found(inst[0], "cap_bore_axis")
        return None if bore is None else line_point_distance(bore["point"], bore["dir"], m._mid_on_axis(ps))
    if name == "winch_azimuth_minus_electronics_datum_mod180":
        w = m.found(inst[0], "winch_bore_axis") if m.sid(inst[0]) is not None else None
        dt = m.datum()
        if w is None or dt is None:
            return None
        return wrap(m.azimuth(m._mid_on_axis(w) - rod[0]) - dt, 180.0)
    if name == "twist_between_winches":
        a, b = (m.found(t, "winch_bore_axis") if m.sid(t) is not None else None for t in inst)
        if a is None or b is None:
            return None
        return wrap(m.azimuth(m._mid_on_axis(a) - rod[0]) - m.azimuth(m._mid_on_axis(b) - rod[0]), 360.0)
    if name == "post_pattern_phase_minus_electronics_datum_mod90":
        dt = m.datum()
        if dt is None or m.sid(inst[0]) is None:
            return None
        z = 0j
        n = 0
        for j in range(4):
            ps = m.found(inst[0], "post_%d" % (j + 1))
            if ps is None:
                continue
            a = 4.0 * (m.azimuth(m._mid_on_axis(ps) - rod[0]) - dt)
            z += complex(math.cos(math.radians(a)), math.sin(math.radians(a)))
            n += 1
        return None if n == 0 else fold(math.degrees(math.atan2(z.imag, z.real)) / 4.0, 90.0)
    if name == "motor_encl_A_mid_to_rod_mid_axial":
        if rod[2] is None or m.sid(inst[0]) is None:
            return None
        b = m.found(inst[0], "motor_encl_A_rod_bore_axis")
        if b is None:
            return None
        g = submission_geo(m.ctx, m.ctx.solid(m.sid(inst[0])))
        t = (g.V - b["point"]) @ b["dir"]
        mid = b["point"] + b["dir"] * 0.5 * (float(t.min()) + float(t.max()))
        return abs(float((mid - rod[2]) @ rod[1]))
    raise KeyError("unknown functional item %s" % name)


PERIODS = {"winch_azimuth_minus_electronics_datum_mod180": 180.0, "twist_between_winches": 360.0,
           "post_pattern_phase_minus_electronics_datum_mod90": 90.0}


def item(ctx):
    """Criterion G (module docstring): {"score", "per_part", "diagnostics"}; per_part[tag] lists the sub-items the part
    takes part in (measured, reference, error, score)."""
    F = ctx.cfg["functional"]
    m = _Measure(ctx)
    rows = {r["item"]: r for r in ctx.ref.features["functional"]["values"]}
    total = sum(float(it["weight"]) for it in F["items"])
    score = 0.0
    per_part = {}
    table = []
    for it in F["items"]:
        row = rows[it["name"]]
        n = len(row["instances"])
        sub = []
        for k, (inst, refv) in enumerate(zip(row["instances"], row["reference"])):
            try:
                val = _measure_item(m, it["name"], inst, k)
            except Exception as exc:  # noqa: BLE001 - one quantity that cannot be measured costs only itself
                from . import item_error
                item_error(ctx, "the functional quantity %s of %s" % (it["name"], "/".join(inst)), exc)
                val = None
            if val is None:
                err, sc = None, 0.0
            else:
                err = ang_diff(val, float(refv), PERIODS[it["name"]]) if it["name"] in PERIODS else abs(val - float(refv))
                sc = band(err, it["band"])
            sub.append(sc)
            rec = {"item": it["name"], "instance": list(inst), "measured": None if val is None else float(val),
                   "reference": float(refv), "error": None if err is None else float(err), "score": float(sc)}
            if sum(1 for x in row["instances"] if list(x) == list(inst)) > 1:
                rec["k"] = k
            table.append(rec)
            for t in inst:
                per_part.setdefault(t, []).append({k2: rec[k2] for k2 in ("item", "measured", "error", "score")})
        score += float(it["weight"]) * (sum(sub) / float(n) if n else 0.0)
    score /= total if total > 0 else 1.0
    ctx.stats["functional_found_rod_axis"] = 0.0 if m.rod() is None else 1.0
    ctx.stats["functional_datum_parts"] = float(m.cache.get("datum_parts", 0))
    ctx.stats["functional_zero_quantities"] = float(sum(1 for r in table if r["score"] <= 0.0))
    per_part["_table"] = table
    diags = []
    worst = sorted([r for r in table if r["score"] < 1.0], key=lambda r: r["score"])[:3]
    for r in worst:
        diags.append("Functional %s (%s): %s" % (r["item"], "/".join(r["instance"]),
                                                 "not found" if r["measured"] is None else
                                                 "off by %.3g %s" % (r["error"], "deg" if r["item"] in PERIODS else "mm")))
    if m.rod() is not None and m.rod()[3] != "rod":
        diags.append("Functional: rod axis taken from %s" % m.rod()[3])
    return {"score": float(score), "per_part": per_part, "diagnostics": diags}
