"""Shape and widths of a venue: the skeleton polygon, its fillets, the half widths and the
geometric validation of the corridor. Everything here is a function of the given numpy Generators.

Shape. A rectangle (the class extent times U(base_scale)) whose corners are kept, chamfered or stepped, into which
notches are cut from the sides until the estimated centreline length reaches a target drawn inside the class range.
A notch is a bight of the track: the track runs into it on one side, turns around at its end and runs out on the
other side. A Featured venue gets one spine notch first: a notch exactly 2 r_s wide whose two legs share the single
row of ducts down its middle (a hairpin around the free end of a shared wall, as in the IROS 2026 tube layout and
the ICRA 2026 Classic map); its fillets are fixed (r_s at the tip, r_top at both top corners). The polygon is then
mirrored and its direction drawn; the centreline is the polygon with a circular fillet at every vertex.

Widths. Every straight draws a total width (U(1.5, 2.2) with probability 0.82, else a narrower one) and a small
asymmetry; an arc interpolates between the straights before and after it. Caps: on the inside of an arc the half
width leaves at least `duct_bend_min_m` of duct axis radius; between two different parts of the lap (more than
`far_station_m` of station apart) each facing half width is at most (D - 2 x 0.33 - clearance) / 2 (two rows of
ducts and free floor between them; this also keeps every corridor inside its own Voronoi cell);
the spine's shared side is fixed at r_s - 0.165 (so that both legs' duct axes are the same line). A Featured venue
narrows one straight to U(1.0, 1.06) m. Half widths then change by at most `slope` per metre (lower envelope).
"""

import math

import numpy as np

from sketch.examples import constants as K

from . import geometry as G

R = K.DUCT_RADIUS_M


class LayoutError(RuntimeError):
    """This attempt cannot give a valid layout; the generator draws another one."""


# ------------------------------------------------------------------------------------------------ shape
def _choice(rng, probs):
    keys = list(probs)
    u = float(rng.random())
    acc = 0.0
    for k in keys:
        acc += float(probs[k])
        if u < acc:
            return k
    return keys[-1]


def _corner_edits(rng, poly, S):
    out = []
    n = len(poly)
    for i, v in enumerate(poly):
        a, c = poly[i - 1], poly[(i + 1) % n]
        u_in = G.unit((v.x - a.x, v.y - a.y))
        u_out = G.unit((c.x - v.x, c.y - v.y))
        e_in, e_out = math.hypot(v.x - a.x, v.y - a.y), math.hypot(c.x - v.x, c.y - v.y)
        kind = _choice(rng, S["corner_edit_probs"])
        ca = float(rng.uniform(*S["chamfer_m"]))
        sa, sb = float(rng.uniform(*S["step_m"])), float(rng.uniform(*S["step_m"]))
        if kind == "chamfer":
            a_ = min(ca, S["chamfer_max_frac"] * e_in, S["chamfer_max_frac"] * e_out)
            out += [G.Vtx(v.x - a_ * u_in[0], v.y - a_ * u_in[1]), G.Vtx(v.x + a_ * u_out[0], v.y + a_ * u_out[1])]
        elif kind == "step":
            a_ = min(sa, S["step_max_frac"] * e_in)
            b_ = min(sb, S["step_max_frac"] * e_out)
            p1 = (v.x - a_ * u_in[0], v.y - a_ * u_in[1])
            out += [G.Vtx(*p1), G.Vtx(p1[0] + b_ * u_out[0], p1[1] + b_ * u_out[1]),
                    G.Vtx(v.x + b_ * u_out[0], v.y + b_ * u_out[1])]
        else:
            out.append(v)
    return out


def _edge_info(poly, i):
    a, b = poly[i], poly[(i + 1) % len(poly)]
    L = math.hypot(b.x - a.x, b.y - a.y)
    u = G.unit((b.x - a.x, b.y - a.y))
    return a, b, L, u


def _insert_notch(poly, i, t0, d, h, tags=None):
    """The polygon with a rectangular notch cut into edge i (from its start point + t0 along the edge, width d,
    depth h toward the interior of a counter-clockwise polygon). tags: (tag, radius, nid) for A, A', B', B."""
    a, b, L, u = _edge_info(poly, i)
    nin = (-u[1], u[0])
    A = (a.x + t0 * u[0], a.y + t0 * u[1])
    B = (A[0] + d * u[0], A[1] + d * u[1])
    A2 = (A[0] + h * nin[0], A[1] + h * nin[1])
    B2 = (B[0] + h * nin[0], B[1] + h * nin[1])
    tg = tags or [(None, None, None)] * 4
    new = [G.Vtx(A[0], A[1], *tg[0]), G.Vtx(A2[0], A2[1], *tg[1]), G.Vtx(B2[0], B2[1], *tg[2]), G.Vtx(B[0], B[1], *tg[3])]
    return poly[:i + 1] + new + poly[i + 1:]


def _edge_blocked(poly, i):
    """Edges that must not get another notch: the edges of a notch (its legs and its end)."""
    a, b = poly[i], poly[(i + 1) % len(poly)]
    return a.tag in ("spine_tip", "notch_tip") or b.tag in ("spine_tip", "notch_tip")


def _inward_room(poly, i, t0, d):
    a, b, L, u = _edge_info(poly, i)
    nin = (-u[1], u[0])
    room = math.inf
    for f in (0.0, 0.5, 1.0):
        p = (a.x + (t0 + f * d) * u[0] + 1e-6 * nin[0], a.y + (t0 + f * d) * u[1] + 1e-6 * nin[1])
        room = min(room, G.ray_hit(poly, p, nin, skip=(i,)))
    return room


def estimate_length(poly, r_nom=1.8):
    n = len(poly)
    per = sum(math.hypot(poly[(k + 1) % n].x - poly[k].x, poly[(k + 1) % n].y - poly[k].y) for k in range(n))
    th = G.turns(poly)
    save = 0.0
    for k in range(n):
        r = poly[k].radius if poly[k].radius is not None else r_nom
        save += 2.0 * r * math.tan(0.5 * abs(th[k])) - r * abs(th[k])
    return per - save


def make_polygon(rng, P, cls):
    """The skeleton polygon of one attempt (counter-clockwise, before mirroring): returns (polygon, info) or raises
    LayoutError."""
    C, S = P["classes"][cls], P["shape"]
    ex, ey = C["centre_extent_m"]
    W = ex * float(rng.uniform(*C["base_scale"]))
    H = ey * float(rng.uniform(*C["base_scale"]))
    lo, hi = C["length_m"]
    m = S["length_margin_m"]
    target = float(rng.uniform(lo + m, hi - m))
    poly = [G.Vtx(0.0, 0.0), G.Vtx(W, 0.0), G.Vtx(W, H), G.Vtx(0.0, H)]
    poly = _corner_edits(rng, poly, S)
    if not G.polygon_is_simple(poly, S["min_edge_gap_m"]):
        raise LayoutError("corner edits made a bad polygon")
    info = {"W": W, "H": H, "target_length_m": target, "notches": [], "spine": None}
    end_m = S["notch_end_margin_m"]
    if C["spine_notches"]:
        r_s = float(rng.uniform(*S["spine_radius_m"]))
        r_top = float(rng.uniform(*S["spine_top_radius_m"]))
        L_sp = float(rng.uniform(*S["spine_length_m"]))
        d, h = 2.0 * r_s, r_top + r_s + L_sp
        done = False
        for _ in range(S["notch_tries"]):
            i = int(rng.integers(0, len(poly)))
            u_t = float(rng.random())
            a, b, L, u = _edge_info(poly, i)
            if L < d + 2.0 * end_m:
                continue
            t0 = end_m + u_t * (L - d - 2.0 * end_m)
            room = _inward_room(poly, i, t0, d)
            if room - h < S["interior_min_m"]:
                continue
            tags = [("spine_top", r_top, 0), ("spine_tip", r_s, 0), ("spine_tip", r_s, 0), ("spine_top", r_top, 0)]
            cand = _insert_notch(poly, i, t0, d, h, tags)
            if G.polygon_is_simple(cand, S["min_edge_gap_m"]):
                poly = cand
                done = True
                break
        if not done:
            raise LayoutError("no room for the spine notch")
        info["spine"] = {"r_s": r_s, "r_top": r_top, "spine_length_m": L_sp, "depth_m": h}
    n_max = int(C["normal_notches"][1])                 # notches are cut until the target length is reached
    tries = 0
    while len(info["notches"]) < n_max and tries < S["notch_tries"] and estimate_length(poly) < target - 1.0:
        tries += 1
        i = int(rng.integers(0, len(poly)))
        d = float(rng.uniform(*S["notch_width_m"]))
        u_t = float(rng.random())
        h_u = float(rng.random())
        a, b, L, u = _edge_info(poly, i)
        if _edge_blocked(poly, i) or L < d + 2.0 * end_m:
            continue
        t0 = end_m + u_t * (L - d - 2.0 * end_m)
        room = _inward_room(poly, i, t0, d)
        h_hi = min(S["notch_depth_m"][1], room - S["interior_min_m"])
        need = 0.5 * (target - estimate_length(poly)) + 1.5           # a notch adds about 2 h - 3 m
        h_lo = max(S["notch_depth_m"][0], 0.5 * d + S["min_radius_m"] + 0.2)      # room for the top corner's fillet
        if h_hi < h_lo:
            continue
        h = h_lo + h_u * (h_hi - h_lo)
        h = min(h, max(h_lo, need))
        nid = 1 + len(info["notches"])
        tags = [(None, None, None), ("notch_tip", 0.5 * d, nid), ("notch_tip", 0.5 * d, nid), (None, None, None)]
        cand = _insert_notch(poly, i, t0, d, h, tags)
        if G.polygon_is_simple(cand, S["min_edge_gap_m"]):
            poly = cand
            info["notches"].append({"width_m": d, "depth_m": h})
    info["estimated_length_m"] = estimate_length(poly)
    return poly, info


def transform_polygon(rng, poly, cls):
    """Random mirror images, (Classic) a random transposition, and the driving direction. Returns (polygon,
    direction) with the vertex order in driving order."""
    mx, my, tr = bool(rng.random() < 0.5), bool(rng.random() < 0.5), bool(rng.random() < 0.5)
    ccw = bool(rng.random() < 0.5)
    out = []
    for v in poly:
        x, y = v.x, v.y
        if cls == "classic" and tr:
            x, y = y, x
        if mx:
            x = -x
        if my:
            y = -y
        out.append(v.moved(x, y))
    if (G.vtx_area(out) > 0) != ccw:
        out = out[::-1]
    return out, ("ccw" if ccw else "cw")


def rotate_polygon(poly):
    return [v.moved(-v.y, v.x) for v in poly]


def translate_polygon(poly, dx, dy):
    return [v.moved(v.x + dx, v.y + dy) for v in poly]


def fillet_radii(rng, poly, S):
    th = G.turns(poly)
    wanted, fixed = [], []
    for v, t in zip(poly, th):
        a = abs(t)
        u = float(rng.random())
        if v.radius is not None:
            wanted.append(float(v.radius))
            fixed.append(True)
            continue
        if abs(a - 0.5 * math.pi) < 0.05:
            lo, hi = S["radius_90_m"]
        elif a < 0.9:
            lo, hi = S["radius_45_m"]
        else:
            lo, hi = S["radius_other_m"]
        wanted.append(lo + u * (hi - lo))
        fixed.append(False)
    r = G.fit_radii(poly, wanted, fixed)
    if r is None:
        raise LayoutError("fixed fillets do not fit")
    for k in range(len(poly)):
        if not fixed[k] and r[k] < S["min_radius_m"] - 1e-9:
            raise LayoutError("a fillet radius below %.2f m" % S["min_radius_m"])
    return r


# ------------------------------------------------------------------------------------------------ layout
class Layout(object):
    """Centreline, half widths and the features that the widths know about (spine zone, narrow section)."""

    def __init__(self, poly, radii, direction, P, cls, rng):
        self.P, self.cls, self.direction = P, cls, direction
        S, Wp = P["shape"], P["width"]
        self.poly, self.radii = poly, radii
        self.prims = G.fillet(poly, radii)
        self.cl = G.Centreline(self.prims, S["sample_spacing_m"])
        cl = self.cl
        n = cl.n
        self.length = cl.length
        lo, hi = P["classes"][cls]["length_m"]
        if not lo + 0.3 <= self.length <= hi - 0.3:
            raise LayoutError("centreline %.1f m outside the class range" % self.length)
        self._spine_zone()
        wl, wr = self._desired(rng)
        fixed_l = np.full(n, np.nan)
        fixed_r = np.full(n, np.nan)
        if self.spine is not None:
            z = self.spine["zone"]
            fixed = self.spine["r_s"] - R
            if self.spine["side"] == "left":
                fixed_l[z] = fixed
            else:
                fixed_r[z] = fixed
        # inner side of the arcs
        k = cl.k
        rho = Wp["duct_bend_min_m"]
        with np.errstate(divide="ignore"):
            inner_cap = np.where(np.abs(k) > 1e-9, 1.0 / np.maximum(np.abs(k), 1e-12) - R - rho, np.inf)
        free_l, free_r = np.isnan(fixed_l), np.isnan(fixed_r)          # the spine's shared side is not capped here
        inner_l = np.where((k > 1e-9) & free_l, inner_cap, np.inf)
        inner_r = np.where((k < -1e-9) & free_r, inner_cap, np.inf)
        # the cap of an arc also holds `bend_pad_m` before and after it, so that the half width has reached its
        # value before the bend starts (a ramp inside the bend kinks the inner duct axis)
        pad = Wp["bend_pad_m"]
        L = cl.length
        for i, s0, p in self.bends():
            cap = p.r - R - rho
            rel = (cl.s - (s0 - pad)) % L
            m = rel <= p.length + 2.0 * pad
            if p.sweep > 0:
                inner_l = np.where(m & free_l, np.minimum(inner_l, cap), inner_l)
            else:
                inner_r = np.where(m & free_r, np.minimum(inner_r, cap), inner_r)
        # what a bend takes from its inner side it gives to its outer side (the outer wall bulges out)
        red_l = np.maximum(0.0, wl - (inner_l - Wp["smooth_margin_m"]))
        red_r = np.maximum(0.0, wr - (inner_r - Wp["smooth_margin_m"]))
        wl = np.where(free_l, np.minimum(Wp["half_max_m"], wl + red_r), wl)
        wr = np.where(free_r, np.minimum(Wp["half_max_m"], wr + red_l), wr)
        wl, wr = np.minimum(wl, inner_l), np.minimum(wr, inner_r)

        # narrow section
        self.narrow = None
        if P["classes"][cls]["narrow_section"]:
            wl, wr = self._narrow(rng, wl, wr)
        # separation between different parts of the lap
        capl, capr = self._separation_caps(wl, wr)
        capl, capr = np.minimum(capl, Wp["half_max_m"]), np.minimum(capr, Wp["half_max_m"])
        for w, cap, f in ((wl, capl, fixed_l), (wr, capr, fixed_r)):
            m = ~np.isnan(f)
            if np.any(cap[m] < f[m] - 1e-9):
                raise LayoutError("a cap is tighter than the spine's shared half width")
            w[:] = np.minimum(w, cap)
            w[m] = f[m]
        margin = Wp["smooth_margin_m"]
        free = [np.isnan(fixed_l), np.isnan(fixed_r)]
        caps = [np.where(free[0], np.minimum(capl, inner_l), np.inf), np.where(free[1], np.minimum(capr, inner_r), np.inf)]
        out = []
        for w, f, cap, fr in ((wl, fixed_l, caps[0], free[0]), (wr, fixed_r, caps[1], free[1])):
            env = self._lipschitz(np.where(fr, np.minimum(w, cap - margin), w), f)
            sm = self._smooth(env, fr)
            out.append(np.where(fr, np.minimum(sm, cap), f))
        wl, wr = out
        self.wl, self.wr = wl, wr
        self._validate()

    # ---------------------------------------------------------------- features
    def _spine_zone(self):
        self.spine = None
        cl, prims = self.cl, self.prims
        tips = [i for i, p in enumerate(prims) if p.kind == "arc" and p.tag == "spine_tip"]
        if not tips:
            return
        m = len(prims)
        # in driving order: top arc, leg in, tip arc, (zero straight), tip arc, leg out, top arc
        t1, t2 = tips[0], tips[1]
        if (t2 - t1) % m != 2:
            t1, t2 = t2, t1
        if (t2 - t1) % m != 2:
            raise LayoutError("spine tip arcs are not consecutive")
        leg_in, leg_out = (t1 - 1) % m, (t2 + 1) % m
        top_in, top_out = (t1 - 2) % m, (t2 + 2) % m
        if prims[top_in].tag != "spine_top" or prims[top_out].tag != "spine_top":
            raise LayoutError("spine top arcs not found")
        ids = [top_in, leg_in, t1, (t1 + 1) % m, t2, leg_out, top_out]
        zone = np.isin(cl.prim, ids)
        side = "left" if prims[t1].sweep > 0 else "right"
        r_s = prims[t1].r
        st = cl.prim_start
        self.spine = {"zone": zone, "side": side, "r_s": r_s, "prims": ids,
                      "leg_in": leg_in, "leg_out": leg_out, "tip": (t1, t2), "top": (top_in, top_out),
                      "s_T_in": st[leg_in], "s_E_in": st[leg_in] + prims[leg_in].length,
                      "s_E_out": st[leg_out], "s_T_out": st[leg_out] + prims[leg_out].length,
                      "spine_length_m": prims[leg_in].length}
        if prims[leg_in].length < 1.0:
            raise LayoutError("spine shorter than 1 m")

    def _desired(self, rng):
        cl, prims, Wp = self.cl, self.prims, self.P["width"]
        m = len(prims)
        cls = self.cls
        line_w = {}
        for i, p in enumerate(prims):
            if p.kind != "line":
                continue
            u = float(rng.random())
            a = float(rng.uniform(*Wp["wide_m"]))
            b = float(rng.uniform(*Wp["narrow_m"][cls]))
            asym = float(rng.uniform(-Wp["asym_m"], Wp["asym_m"]))
            W = a if u < Wp["wide_prob"] else b
            line_w[i] = (0.5 * W + asym, 0.5 * W - asym)
        if self.spine is not None:
            # the straights next to the spine's top arcs keep at least the shared half width on the spine side
            f = self.spine["r_s"] - R
            j = 0 if self.spine["side"] == "left" else 1
            for top in self.spine["top"]:
                for nb in ((top - 1) % m, (top + 1) % m):
                    if prims[nb].kind == "line" and nb not in (self.spine["leg_in"], self.spine["leg_out"]):
                        w = list(line_w[nb])
                        w[j] = max(w[j], f)
                        line_w[nb] = tuple(w)
        wl, wr = np.zeros(cl.n), np.zeros(cl.n)
        for q in range(cl.n):
            i = int(cl.prim[q])
            p = prims[i]
            if p.kind == "line":
                wl[q], wr[q] = line_w[i]
            else:
                a, b = line_w[(i - 1) % m], line_w[(i + 1) % m]
                f = cl.t[q] / p.length if p.length > 0 else 0.0
                wl[q] = (1.0 - f) * a[0] + f * b[0]
                wr[q] = (1.0 - f) * a[1] + f * b[1]
        return wl, wr

    def _narrow(self, rng, wl, wr):
        cl, prims, Wp = self.cl, self.prims, self.P["width"]
        zone = set(self.spine["prims"]) if self.spine else set()
        cands = [i for i, p in enumerate(prims) if p.kind == "line" and p.length >= Wp["narrow_section_straight_min_m"]
                 and i not in zone]
        u = float(rng.random())
        Wn = float(rng.uniform(*Wp["narrow_section_width_m"]))
        Ln = float(rng.uniform(*Wp["narrow_section_length_m"]))
        if not cands:
            raise LayoutError("no straight for the narrow section")
        i = cands[min(int(u * len(cands)), len(cands) - 1)]
        p = prims[i]
        s0 = cl.prim_start[i] + 0.5 * p.length
        Ln = min(Ln, p.length - 2.0)
        m = (cl.prim == i) & (np.abs(cl.s - s0) <= 0.5 * Ln)
        tot = wl + wr
        scale = np.where(m, np.minimum(1.0, Wn / tot), 1.0)
        self.narrow = {"prim": i, "s_m": float(s0), "length_m": float(Ln), "width_m": Wn}
        return wl * scale, wr * scale

    def _far_pairs(self):
        cl = self.cl
        far = G.cyclic_ds(cl.s, cl.length) > self.P["width"]["far_station_m"]
        if self.spine is not None:
            z = self.spine["zone"]
            far &= ~(z[:, None] & z[None, :])
        return far

    def _separation_caps(self, wl, wr):
        cl, Wp = self.cl, self.P["width"]
        xy = cl.xy()
        dx = xy[None, :, 0] - xy[:, None, 0]
        dy = xy[None, :, 1] - xy[:, None, 1]
        D = np.hypot(dx, dy)
        dot = dx * cl.nx[:, None] + dy * cl.ny[:, None]
        far = self._far_pairs()
        cap = np.where(far, 0.5 * (D - 4.0 * R - Wp["wall_clearance_m"]), np.inf)
        capl = np.min(np.where(dot > 0, cap, np.inf), axis=1)
        capr = np.min(np.where(dot <= 0, cap, np.inf), axis=1)
        self._D, self._dot, self._far = D, dot, far
        return capl, capr

    def _lipschitz(self, w, fixed):
        cl, k = self.cl, self.P["width"]["slope"]
        n = cl.n
        w = w.copy()
        s, L = cl.s, cl.length
        for _ in range(2):
            for q in range(1, 2 * n):
                i, j = q % n, (q - 1) % n
                ds = (s[i] - s[j]) % L
                w[i] = min(w[i], w[j] + k * ds)
            for q in range(2 * n - 2, -1, -1):
                i, j = q % n, (q + 1) % n
                ds = (s[j] - s[i]) % L
                w[i] = min(w[i], w[j] + k * ds)
        m = ~np.isnan(fixed)
        if np.any(w[m] < fixed[m] - 1e-9):
            raise LayoutError("the width ramps cut into the spine's shared half width")
        return w

    def _smooth(self, w, free):
        """Two passes of a cyclic moving average over +-`smooth_half_window` samples on the free samples (pinned
        samples keep their values but feed their neighbours)."""
        h = int(self.P["width"]["smooth_half_window"])
        n = len(w)
        out = w.copy()
        for _ in range(2):
            pad = np.concatenate([out[-h:], out, out[:h]])
            ker = np.ones(2 * h + 1) / (2 * h + 1)
            avg = np.convolve(pad, ker, mode="valid")[:n]
            out = np.where(free, avg, out)
        return out

    # ---------------------------------------------------------------- validation
    def _validate(self):
        cl, Wp, P = self.cl, self.P["width"], self.P
        wl, wr = self.wl, self.wr
        if np.any(wl < Wp["half_min_m"] - 1e-9) or np.any(wr < Wp["half_min_m"] - 1e-9):
            raise LayoutError("a half width below %.2f m" % Wp["half_min_m"])
        W = wl + wr
        wmin = P["classes"][self.cls]["min_width_m"]
        if float(W.min()) < wmin - 1e-9:
            raise LayoutError("width %.2f m below the class minimum" % float(W.min()))
        ds = np.diff(np.concatenate([cl.s, [cl.length]]))
        frac = float(np.sum(ds[W >= 1.5 - 1e-9]) / cl.length)
        if frac < Wp["frac_ge_1p5_min"]:
            raise LayoutError("only %.2f of the length at 1.5 m or more" % frac)
        if self.narrow is not None and float(W.min()) >= K.NARROW_SECTION_M:
            raise LayoutError("the narrow section is not narrower than 1.2 m")
        # facing corridors of different parts of the lap: two rows of ducts and the clearance between them
        D, dot, far = self._D, self._dot, self._far
        wi = np.where(dot > 0, wl[:, None], wr[:, None])
        dotj = dot.T                                         # dotj[i, j] = (c_i - c_j) . n_j: the side of j facing i
        wj = np.where(dotj > 0, wl[None, :], wr[None, :])
        short = far & (D < wi + wj + 4.0 * R + Wp["wall_clearance_m"] - 1e-6)
        if np.any(short):
            raise LayoutError("two parts of the lap are too close")
        # duct axis curves
        AL = np.stack([cl.x + (wl + R) * cl.nx, cl.y + (wl + R) * cl.ny], axis=1)
        AR = np.stack([cl.x - (wr + R) * cl.nx, cl.y - (wr + R) * cl.ny], axis=1)
        self.axis = {"left": AL, "right": AR}
        for name, A in (("left", AL), ("right", AR)):
            d = np.diff(np.vstack([A, A[:1]]), axis=0)
            fwd = d[:, 0] * cl.tx + d[:, 1] * cl.ty
            bad = fwd < -1e-9
            if self.spine is not None and self.spine["side"] == name:
                bad &= ~self.spine["zone"]
            if np.any(bad):
                raise LayoutError("the %s duct axis folds back" % name)
            q = 3
            a_, b_, c_ = np.roll(A, q, axis=0), A, np.roll(A, -q, axis=0)
            la = np.hypot(*(b_ - c_).T)
            lb = np.hypot(*(a_ - c_).T)
            lc = np.hypot(*(a_ - b_).T)
            ar2 = np.abs((b_[:, 0] - a_[:, 0]) * (c_[:, 1] - a_[:, 1]) - (b_[:, 1] - a_[:, 1]) * (c_[:, 0] - a_[:, 0]))
            with np.errstate(divide="ignore", invalid="ignore"):
                rad = np.where(ar2 > 1e-12, la * lb * lc / (2.0 * np.maximum(ar2, 1e-300)), np.inf)
            tight = (rad < Wp["duct_bend_check_m"]) & (lb > 0.05)
            if self.spine is not None and self.spine["side"] == name:
                tight &= ~self.spine["zone"]
            if np.any(tight):
                raise LayoutError("the %s duct axis bends tighter than %.2f m" % (name, Wp["duct_bend_check_m"]))
        both = np.vstack([AL, AR])
        sidx = np.concatenate([np.arange(cl.n), np.arange(cl.n)])
        n2 = len(both)
        zone2 = np.concatenate([self.spine["zone"], self.spine["zone"]]) if self.spine is not None else np.zeros(n2, bool)
        for a0 in range(0, n2, 400):
            blk = both[a0:a0 + 400]
            dd = np.hypot(blk[:, None, 0] - both[None, :, 0], blk[:, None, 1] - both[None, :, 1])
            dsb = np.abs(cl.s[sidx[a0:a0 + 400]][:, None] - cl.s[sidx][None, :])
            dsb = np.minimum(dsb, cl.length - dsb)
            ex = zone2[a0:a0 + 400][:, None] & zone2[None, :]
            bad = (dsb > Wp["far_station_m"]) & ~ex & (dd < 2.0 * R - 1e-6)
            if np.any(bad):
                raise LayoutError("ducts of two parts of the lap overlap")
        # the class footprint
        ext = np.vstack([AL, AR])
        fx = float(ext[:, 0].max() - ext[:, 0].min()) + 2.0 * R
        fy = float(ext[:, 1].max() - ext[:, 1].min()) + 2.0 * R
        bx, by = P["classes"][self.cls]["footprint_m"]
        if max(fx, fy) > max(bx, by) - 0.05 or min(fx, fy) > min(bx, by) - 0.05:
            raise LayoutError("footprint %.2f x %.2f m over the class box" % (fx, fy))
        self.frac_ge_1p5 = frac

    # ---------------------------------------------------------------- helpers
    def samples(self):
        """[x, y, left, right] rows in driving order."""
        cl = self.cl
        return np.stack([cl.x, cl.y, self.wl, self.wr], axis=1)

    def straights(self, min_len=0.0):
        return [(i, self.cl.prim_start[i], p.length) for i, p in enumerate(self.prims)
                if p.kind == "line" and p.length >= min_len]

    def bends(self):
        return [(i, self.cl.prim_start[i], p) for i, p in enumerate(self.prims) if p.kind == "arc" and p.length > 1e-9]
