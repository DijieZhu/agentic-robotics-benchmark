"""Suction of a return inlet on loose items that cover part of it: a lumped duct-and-fan network
(numpy only, no scipy).

Network (one cabin fan, N return inlets in parallel -> common duct -> fan):
    fan:    dp_fan(Q, w) = curve(Q / (s w)) * (s w)^2       published cabin-fan curve, affinity scaled; w = speed
                                                             fraction 0..1 (spin-down state), s = size scale
    duct:   K_sys Q^2                                       calibrated so the unblocked flow is Q_design
    inlet i (face area A, open area A_o = A - sealed area):
            dp_i = K_face rho / (2 A_o^2) Q_i^2 + r_face / A_o Q_i
            K_face = quadratic loss of grille + face screen (on the face velocity),
            r_face = linear resistance of a filter medium right behind the face [Pa per m/s of face velocity].
The inlet pressure difference P_c (cabin minus duct) is common to all open inlets.  An item lying on inlet i
carries P_c over its sealed area (behind it the medium / duct passes no flow, so there is no drop across it).
Sealing decays with the item's gap g to the face: s(g) = 1 / (1 + (g / g0)^2).

Inlet builds (``INLET_BUILDS``): four (bare grille, grille + 20-mesh screen, + coarse prefilter mat, filter element
as the inlet face) plus two intermediate mats.  The filter resistance is given per (m^3/s) at a 0.737 x 0.102 m
filter face and converted once to r_face [Pa s/m] (physical, transfers to other face areas), so the values hold
exactly at that geometry.

Fan spin-down / spin-up: first order, tau_fan (3 s, an estimate).

Items: ``SuctionFaces`` samples the two large faces (local z = +-half_z) of each item body in cells with
sub-points; the suction of a cell acts at the cell centre, so lifting an edge loses that edge's suction (the
peel mechanics appears by itself).  It is a wrench source (``accumulate(model, data, W)``); feed the sealed
areas back to the networks with ``update_networks()`` at the network rate and advance the fans with
``FanNetwork.advance(dt)``.  No global state; deterministic.
"""
from __future__ import annotations

import math

import numpy as np

from ._fast import cross_rows

__all__ = [
    "INCH", "CFM", "INH2O", "AIR_DENSITY", "FILTER_FACE_REF", "FAN_CURVE_Q", "FAN_CURVE_DP", "FAN_DESIGN_POINT",
    "INLET_BUILDS", "MAIN_BUILDS", "brent", "Inlet", "fan_dp", "FanNetwork", "SuctionFaces", "holding_curve",
]

INCH = 0.0254
CFM = 4.71947e-4             # m^3/s per ft^3/min
INH2O = 249.089              # Pa per inch of water
AIR_DENSITY = 1.2            # kg/m^3 (the propulsion model's 1.2)

# Station filter element face 29 x 4 in, rated 70 cfm (the reference inlet)  [R filter test-protocol report]
FILTER_FACE_REF = (29 * INCH, 4 * INCH)
FILTER_FACE_REF_AREA = FILTER_FACE_REF[0] * FILTER_FACE_REF[1]
FILTER_RATED_FLOW = 70 * CFM

# Quiet cabin-ventilation fan, 12 000 rpm: free 220.4 cfm @ 0.52 inH2O, design 150.6 cfm @ 3.48 inH2O, peak
# 4.0 inH2O @ 120.7 cfm [R quiet cabin-fan measurement report]; the points below 120 cfm (stall region) are estimates [G].
FAN_CURVE_Q = np.array([0.0, 60.0, 100.0, 120.7, 150.6, 190.0, 220.4]) * CFM
FAN_CURVE_DP = np.array([3.0, 3.4, 3.85, 4.0, 3.48, 1.9, 0.52]) * INH2O
FAN_DESIGN_POINT = (150.6 * CFM, 3.48 * INH2O)

# K_face [-] on the face velocity; r_face [Pa per m/s of face velocity].  Values: r_filter 200 / 500 / 1200 /
# 5300 Pa per (m^3/s) at the reference face -> r_face = r_filter * A_ref.  All [G] (order of magnitude of
# a perforated grille, a 20-mesh screen, a coarse prefilter mat, a filter-faced inlet).
_A = FILTER_FACE_REF_AREA
INLET_BUILDS = {
    "grille": dict(K_face=2.0, r_face=0.0),               # bare perforated / louvred grille
    "screen": dict(K_face=8.0, r_face=0.0),               # grille + 20-mesh debris screen
    "low": dict(K_face=8.0, r_face=200.0 * _A),           # + thin dust mat (removal-sweep intermediate)
    "mid": dict(K_face=8.0, r_face=500.0 * _A),           # + light prefilter (removal-sweep intermediate)
    "prefilter": dict(K_face=8.0, r_face=1200.0 * _A),    # + coarse prefilter mat (~40 Pa at the ref. face, 70 cfm)
    "filter_face": dict(K_face=8.0, r_face=5300.0 * _A),  # filter element as the inlet face (~175 Pa)
    "prefilter_double": dict(K_face=8.0, r_face=2400.0 * _A),  # two coarse mats (the v0.2 smoke demo's inlet build)  [G]
    # v0.3: a light prefilter mat on the screen, chosen so that at the task station's 40 cfm per inlet a
    # 0.30 x 0.30 m soft item covering 0.036 m^2 of a 0.40 x 0.12 m inlet is held with 1.16 N (window 0.85-1.45 N)  [G]
    "prefilter_light": dict(K_face=8.0, r_face=64.0),
}
MAIN_BUILDS = ("grille", "screen", "prefilter", "filter_face")


def brent(f, a: float, b: float, xtol: float = 1e-12, rtol: float = 4.4e-16, maxiter: int = 200) -> float:
    """Root of f in [a, b] (f(a), f(b) of opposite sign), Brent's method (as scipy.optimize.brentq)."""
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("root not bracketed")
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxiter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2.0 * rtol * abs(b) + 0.5 * xtol
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return b
        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p, q = 2.0 * m * s, 1.0 - s
            else:
                q, r = fa / fc, fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = m
                e = d
        else:
            d = m
            e = d
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0 else -tol)
        fb = f(b)
    return b


class Inlet:
    """Rectangular return inlet (or supply opening) on a wall.  ``normal`` points into the cabin; ``t1`` is the
    in-plane long axis; ``half`` (2,) the half sizes along t1, t2 = normal x t1.  ``flow`` is the current volume
    flow [m^3/s] (written by the network); ``flow0`` the design flow."""

    def __init__(self, name: str, center, normal, t1, half, kind: str = "return", flow: float = FILTER_RATED_FLOW,
                 n_sinks: int = 7, reg: float | None = None):
        self.name = name
        self.c = np.asarray(center, float)
        self.n = np.asarray(normal, float) / np.linalg.norm(normal)
        t1 = np.asarray(t1, float)
        t1 = t1 - (t1 @ self.n) * self.n
        self.t1 = t1 / np.linalg.norm(t1)
        self.t2 = np.cross(self.n, self.t1)
        self.half = np.asarray(half, float)
        self.kind = kind
        self.flow0 = float(flow)
        self.flow = float(flow)
        k = int(n_sinks)
        s = (np.arange(k) + 0.5) / k * 2 - 1
        self.pts = self.c[None, :] + s[:, None] * self.half[0] * self.t1[None, :]
        self.reg = reg if reg is not None else max(self.half[1], 0.6 * self.half[0] / k)

    @property
    def area(self) -> float:
        """Face area [m^2]."""
        return 4.0 * self.half[0] * self.half[1]

    def velocity(self, x) -> np.ndarray:
        """Near-field velocity [m/s] of the inlet as a line of regularised half-space sinks (sources for supply)
        at points x (N, 3); zero more than 2 cm behind the wall plane.  The total flux equals ``flow``."""
        x = np.atleast_2d(np.asarray(x, float))
        sgn = -1.0 if self.kind == "return" else 1.0
        q = self.flow / len(self.pts)
        u = np.zeros_like(x)
        a2 = self.reg ** 2
        for p in self.pts:
            r = x - p
            r2 = np.einsum("ij,ij->i", r, r)
            front = (r @ self.n) > -0.02
            u += (sgn * q / (2 * np.pi)) * r / ((r2 + a2) ** 1.5)[:, None] * front[:, None]
        return u

    def local(self, x):
        """(gap along the normal, in-plane s1, s2) of points x."""
        d = np.asarray(x, float) - self.c
        return d @ self.n, d @ self.t1, d @ self.t2


def fan_dp(Q: float, omega_frac: float = 1.0, size_scale: float = 1.0) -> float:
    """Fan pressure rise [Pa] at flow Q [m^3/s]: published curve, affinity-scaled by speed fraction and size scale
    (Q ~ s w, dp ~ (s w)^2).  0 when stopped; 0 beyond the free-delivery end of the curve."""
    sw = omega_frac * size_scale
    if sw <= 1e-6:
        return 0.0
    q = Q / sw
    return float(np.interp(q, FAN_CURVE_Q, FAN_CURVE_DP, right=0.0)) * sw * sw


class FanNetwork:
    """One cabin fan with N parallel return inlets (see module doc).

    builds: one build name for all inlets or a dict inlet name -> build name (or a dict of K_face / r_face).
    Q_design: unblocked design flow (default: sum of the inlets' flow0).  fan_scale: 'published' (the curve at its
    test speed) or 'design_point' (size-scaled so the curve's design point delivers Q_design: keeps a
    small module fan out of the stall region) or a float.  tau_fan: first-order spin time constant [s].
    State: ``omega`` (speed fraction), ``omega_cmd``, ``Pc`` (inlet pressure difference [Pa]), ``Q`` (total flow),
    ``blocked`` {inlet: sealed area}.  ``solve()`` is skipped when nothing changed (key = speed and blockages)."""

    def __init__(self, inlets, build="prefilter", Q_design: float | None = None, tau_fan: float = 3.0,
                 rho: float = AIR_DENSITY, fan_scale="published", name: str = "fan"):
        self.name = name
        self.inlets = list(inlets)
        self.by_name = {i.name: i for i in self.inlets}

        def resolve(b):
            return dict(INLET_BUILDS[b]) if isinstance(b, str) else dict(b)
        if isinstance(build, (str, dict)) and not (isinstance(build, dict) and set(build) & set(self.by_name)):
            self.build = {i.name: resolve(build) for i in self.inlets}
        else:
            self.build = {i.name: resolve(build[i.name]) for i in self.inlets}
        self.rho = rho
        self.Qd = float(Q_design) if Q_design is not None else float(sum(i.flow0 for i in self.inlets))
        if fan_scale == "published":
            self.size = 1.0
        elif fan_scale == "design_point":
            self.size = self.Qd / FAN_DESIGN_POINT[0]
        else:
            self.size = float(fan_scale)
        self.omega = 1.0
        self.omega_cmd = 1.0
        self.tau = float(tau_fan)
        self.blocked = {i.name: 0.0 for i in self.inlets}
        self.history = []                # (time, 'on' / 'off') of fan requests, written by set_fan(t=...)
        Pc0 = self._inlet_dp_total(self.Qd)
        dp_fan = fan_dp(self.Qd, 1.0, self.size)
        if dp_fan < Pc0:
            raise ValueError(f"fan '{name}' cannot deliver {self.Qd:.4f} m^3/s through the inlets")
        self.K_sys = (dp_fan - Pc0) / self.Qd ** 2
        self.Pc, self.Q = Pc0, self.Qd
        self._key = None
        self.solve()

    def _coeffs(self, inl):
        b = self.build[inl.name]
        A = inl.area
        A_open = max(A - self.blocked[inl.name], 1e-5)
        return b["K_face"] * self.rho / (2.0 * A_open ** 2), b["r_face"] / A_open

    def _kr(self):
        k, r = zip(*(self._coeffs(inl) for inl in self.inlets))
        return np.array(k), np.array(r)

    def _q_of_p(self, Pc: float, kr=None) -> np.ndarray:
        k, r = self._kr() if kr is None else kr
        if Pc <= 0:
            return np.zeros(len(k))
        q = np.where(k > 0, (-r + np.sqrt(r * r + 4.0 * k * Pc)) / (2.0 * np.where(k > 0, k, 1.0)), Pc / np.maximum(r, 1e-30))
        return q

    def _inlet_dp_total(self, Qtot: float, kr=None) -> float:
        kr = self._kr() if kr is None else kr
        if Qtot <= 0:
            return 0.0
        hi = 1.0
        while self._q_of_p(hi, kr).sum() < Qtot:
            hi *= 4.0
        return brent(lambda p: float(self._q_of_p(p, kr).sum()) - Qtot, 0.0, hi, xtol=1e-6, rtol=1e-10)

    def solve(self) -> None:
        """Operating point dp_fan(Q) = K_sys Q^2 + P_c(Q); writes Pc, Q and every inlet's flow."""
        w = self.omega
        key = (round(w, 6),) + tuple(round(self.blocked[i.name], 7) for i in self.inlets)
        if self._key == key:
            return
        self._key = key
        if w < 1e-4:
            self.Pc, self.Q = 0.0, 0.0
            for inl in self.inlets:
                inl.flow = 0.0
            return
        kr = self._kr()

        def f(Q):
            return fan_dp(Q, w, self.size) - self.K_sys * Q * Q - self._inlet_dp_total(Q, kr)
        hi = FAN_CURVE_Q[-1] * w * self.size * 0.999999
        self.Q = brent(f, 1e-9, hi, xtol=1e-9) if f(hi) < 0 else hi
        self.Pc = self._inlet_dp_total(self.Q, kr)
        for inl, q in zip(self.inlets, self._q_of_p(self.Pc, kr)):
            inl.flow = float(q)

    def advance(self, dt: float) -> None:
        """First-order spin toward the commanded speed fraction over dt seconds."""
        self.omega += (self.omega_cmd - self.omega) * (1.0 - math.exp(-dt / self.tau))
        if self.omega_cmd == 0.0 and self.omega < 1e-3:
            self.omega = 0.0
        if self.omega_cmd == 1.0 and self.omega > 1.0 - 1e-6:
            self.omega = 1.0

    def set_fan(self, on: bool, t: float | None = None) -> None:
        """Command the fan on (speed fraction 1) or off (0)."""
        self.omega_cmd = 1.0 if on else 0.0
        if t is not None:
            self.history.append((float(t), "on" if on else "off"))

    def set_blocked(self, name: str, area: float) -> None:
        """Sealed area of one inlet (clipped below the face area)."""
        self.blocked[name] = min(max(float(area), 0.0), 0.999 * self.by_name[name].area)

    def open_fraction(self, name: str) -> float:
        """Open area of an inlet relative to its face area."""
        return 1.0 - self.blocked[name] / self.by_name[name].area


class SuctionFaces:
    """Suction on the two large faces of item bodies (wrench source).

    items: list of (body_id, half_sizes (3,), grid (nx, ny)); each face (local z = +-half_z) is split into nx x ny
    cells with ``sub`` x ``sub`` sub-points for the coverage and gap test; the cell's force acts at its centre.
    networks: FanNetwork or list.  g0: seal decay length [m]; g_max: no seal beyond this gap; cos_min: the face must
    oppose the inlet normal at least this much.  ``covered`` {inlet: sealed area}, ``item_force`` {body: normal
    suction force [N]} after each evaluation.
    occlusion (v0.4; False = v0.3 and the first v0.4): only the layer that directly faces an inlet (or a
    holding-screen face) is sucked: a sub-point whose straight path to the face (along the inlet normal) crosses the
    large face of another item that has sub-points in the seal box -- an item lying between it and the face -- gets no
    suction and adds no sealed area (behind a sealing item there is cabin pressure, and nothing flows around a layer
    that lies on the face).  ``occl_tol``: that item's face must be at least this much nearer the face [m].  With a
    single item at an inlet, or items that do not overlap along the normal, the result is bit-identical to v0.3.
    """

    def __init__(self, model, items, networks, g0: float = 0.0015, g_max: float = 0.012, cos_min: float = 0.5,
                 sub: int = 4, occlusion: bool = True, occl_tol: float = 1e-4):
        self.nets = list(networks) if isinstance(networks, (list, tuple)) else [networks]
        self.g0, self.gmax, self.cmin = float(g0), float(g_max), float(cos_min)
        self.occlusion, self.occl_tol = bool(occlusion), float(occl_tol)
        self._half = {int(bid): tuple(float(v) for v in half) for bid, half, _ in items}
        self._half_arr = np.zeros((max(self._half, default=-1) + 1, 3))
        for bid_, h_ in self._half.items():
            self._half_arr[bid_] = h_
        self.shadowed = {}
        b, rl, nl, A, offs = [], [], [], [], []
        for bid, half, (nx, ny) in items:
            hx, hy, hz = (float(v) for v in half)
            cx, cy = 2 * hx / nx, 2 * hy / ny
            xs = ((np.arange(nx) + 0.5) / nx * 2 - 1) * hx
            ys = ((np.arange(ny) + 0.5) / ny * 2 - 1) * hy
            u = (np.arange(sub) + 0.5) / sub - 0.5
            o = np.stack(np.meshgrid(u * cx, u * cy, indexing="ij"), -1).reshape(-1, 2)
            o3 = np.column_stack([o, np.zeros(len(o))])
            for sz in (1.0, -1.0):
                for x in xs:
                    for y in ys:
                        b.append(int(bid))
                        rl.append((x, y, sz * hz))
                        nl.append((0.0, 0.0, sz))
                        A.append(cx * cy)
                        offs.append(o3)
        self.b = np.array(b, int)
        self.rl = np.array(rl, float)
        self.nl = np.array(nl, float)
        self.A = np.array(A, float)
        self.offs = np.array(offs, float)
        self.ubodies = np.unique(self.b)
        self._rows = {int(bb): np.nonzero(self.b == bb)[0] for bb in self.ubodies}
        self.brad = np.array([float(np.max(np.linalg.norm(self.rl[self.b == bb], axis=1))) for bb in self.ubodies]) + 0.05
        self.inlets = [inl for net in self.nets for inl in net.inlets]
        self._C = np.array([i.c for i in self.inlets])
        self._N = np.array([i.n for i in self.inlets])
        self._T1 = np.array([i.t1 for i in self.inlets])
        self._T2 = np.array([i.t2 for i in self.inlets])
        self._H = np.array([i.half for i in self.inlets])
        self.covered = {i.name: 0.0 for i in self.inlets}
        self.item_force = {}
        self.enabled = True
        # v0.4 (bit-identical speed-up): the network of each inlet, the cell rows of each candidate-body set, the radius
        # of each cell's sub-point pattern (a cell whose centre is farther than this outside the seal box cannot
        # contribute: its sub-points are within that radius of the centre)
        self._inlet_net = [net for net in self.nets for _ in net.inlets]
        self._rows_cache = {}
        self._rsub = np.linalg.norm(self.offs, axis=2).max(axis=1) + 1e-9

    def _cand_rows(self, mask):
        """(cell rows, their bodies, body-frame centres, body-frame normals) of the candidate bodies ``mask`` (cached)."""
        key = mask.tobytes()
        c = self._rows_cache.get(key)
        if c is None:
            rows = np.concatenate([self._rows[int(bb)] for bb in self.ubodies[mask]])
            c = (rows, self.b[rows], self.rl[rows], self.nl[rows], self._rsub[rows])
            self._rows_cache[key] = c
        return c

    def seal(self, g):
        """Seal fraction at gap g [m]."""
        return 1.0 / (1.0 + (np.maximum(g, 0.0) / self.g0) ** 2)

    def _shadowed(self, data, sp, gs, ins, cb, act, inl, a1, a2):
        """(cells, sub) bool or None (the layering rule): sub-points in the seal box whose path to the inlet face along -n
        crosses the large face of another item in ``act`` that faces the inlet (that item lies between them and the
        face); None when there are none.  Necessary conditions, from the faces' corners, are tested first -- the whole
        set, then per pair of items: an item can only shadow a sub-point if its facing face reaches at least
        ``occl_tol`` nearer the inlet plane than the sub-point and its footprint on the inlet plane covers it -- and
        the exact ray test runs only on the sub-points that pass."""
        n, tol = inl.n, self.occl_tol
        R = data.xmat[act].reshape(-1, 3, 3)
        Rn = n @ R                                                         # (x . n, y . n, z . n) of every item
        half = self._half_arr[act]
        s = np.where(Rn[:, 2] > 0.0, -1.0, 1.0)                            # the large face whose outward normal opposes n
        # lowest gap of each facing face over its rectangle (exact: centre - the half extents' projections)
        cgmin = ((data.xpos[act] - inl.c) @ n + s * half[:, 2] * Rn[:, 2] - half[:, 0] * np.abs(Rn[:, 0])
                 - half[:, 1] * np.abs(Rn[:, 1]))
        if float(gs[ins].max()) < float(cgmin.min()) + tol:
            return None                                                    # nothing lies above another item's face
        z = R[:, :, 2]
        sg = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
        corners = (data.xpos[act][:, None, :] + (half[:, None, 0:1] * sg[None, :, 0:1]) * R[:, None, :, 0]
                   + (half[:, None, 1:2] * sg[None, :, 1:2]) * R[:, None, :, 1] + (half[:, 2] * s)[:, None, None] * z[:, None, :])
        dc = corners - inl.c
        c1, c2 = dc @ inl.t1, dc @ inl.t2
        out = None
        for k in range(len(act)):                                          # occluder k
            sel = ins & (cb != act[k])[:, None] & (gs >= cgmin[k] + tol)
            if not sel.any():
                continue
            sel &= (a1 >= c1[k].min()) & (a1 <= c1[k].max()) & (a2 >= c2[k].min()) & (a2 <= c2[k].max())
            if not sel.any():
                continue
            nc = z[k] * s[k]
            den = float(n @ nc)
            if abs(den) < 1e-9:
                continue
            Q = sp[sel]
            tt = ((Q - (data.xpos[act[k]] + nc * half[k, 2])) @ nc) / den              # distance along -n to that face
            X = Q - tt[:, None] * n[None, :]
            u = (X - data.xpos[act[k]]) @ R[k]
            hit = (tt >= tol) & (tt <= gs[sel] + 0.01) & (np.abs(u[:, 0]) <= half[k, 0]) & (np.abs(u[:, 1]) <= half[k, 1])
            if hit.any():
                if out is None:
                    out = np.zeros(ins.shape, bool)
                out[sel] |= hit
        return out

    def accumulate(self, model, data, W) -> None:
        """Add the suction forces (world axes at the body CoMs) into W."""
        self.item_force = {}
        self.shadowed = {}
        cov = {i.name: 0.0 for i in self.inlets}
        self.covered = cov
        if not self.enabled or len(self.inlets) == 0:
            return
        dd = data.xpos[self.ubodies][:, None, :] - self._C[None, :, :]
        rr = self.brad[:, None]
        ok = ((np.abs(np.einsum("bik,ik->bi", dd, self._N)) < rr)
              & (np.abs(np.einsum("bik,ik->bi", dd, self._T1)) < self._H[None, :, 0] + rr)
              & (np.abs(np.einsum("bik,ik->bi", dd, self._T2)) < self._H[None, :, 1] + rr))
        hit = np.nonzero(ok.any(axis=0))[0]                  # inlets with a candidate body near them
        if len(hit) == 0:
            return
        for ki in hit:
            inl = self.inlets[int(ki)]
            Pc = self._inlet_net[int(ki)].Pc
            idx_all, b_all, rl_all, nl_all, rs = self._cand_rows(ok[:, ki])
            R = data.xmat[b_all].reshape(-1, 3, 3)
            p = data.xpos[b_all] + np.einsum("bij,bj->bi", R, rl_all)
            nrm = np.einsum("bij,bj->bi", R, nl_all)
            g, s1, s2 = inl.local(p)
            near = ((np.abs(s1) <= inl.half[0] + 0.3) & (np.abs(s2) <= inl.half[1] + 0.3) & (g <= self.gmax + 0.3)
                    & (g > -0.02))
            # exact early-out (v0.4): cells that cannot have a sub-point inside the seal box (no effect on the result)
            near &= ((np.abs(s1) <= inl.half[0] + rs) & (np.abs(s2) <= inl.half[1] + rs) & (g <= self.gmax + rs)
                     & (g > -0.01 - rs))
            m = near & ((nrm @ inl.n) < -self.cmin)
            if not m.any():
                continue
            loc = np.nonzero(m)[0]
            idx = idx_all[loc]
            sp = p[loc, None, :] + np.einsum("cij,ckj->cki", R[loc], self.offs[idx])
            d = sp - inl.c
            gs = d @ inl.n
            a1 = d @ inl.t1
            a2 = d @ inl.t2
            ins = (np.abs(a1) <= inl.half[0]) & (np.abs(a2) <= inl.half[1]) & (gs <= self.gmax) & (gs > -0.01)
            if self.occlusion:
                # only the layer directly facing the face is sucked (no change unless items overlap along n)
                cb = self.b[idx]
                cbi = cb[ins.any(axis=1)]                                  # rows are grouped by body, ascending
                act = np.unique(cbi) if len(cbi) and cbi[0] != cbi[-1] else ()
                if len(act) > 1:
                    sh = self._shadowed(data, sp, gs, ins, cb, act, inl, a1, a2)
                    if sh is not None and sh.any():
                        ins = ins & ~sh
                        for bb_ in np.unique(cb[sh.any(axis=1)]).tolist():
                            self.shadowed[(inl.name, bb_)] = int(sh[cb == bb_].sum())    # shadowed sub-points
            sl = (ins * self.seal(gs)).mean(axis=1)
            keep = sl > 0
            if not keep.any():
                continue
            idx, sl, loc = idx[keep], sl[keep], loc[keep]
            cov[inl.name] += float(np.sum(self.A[idx] * sl))
            f = (-Pc * self.A[idx] * sl)[:, None] * inl.n[None, :]
            bb = self.b[idx]
            # rows are grouped by body in ascending body order (candidate rows are concatenated that way): per body the
            # same rows in the same order as the v0.3 boolean selection.  Per body run, the wrench row is updated by a
            # sequential running sum starting from W's row (np.add.accumulate), i.e. the same additions in the same
            # order as v0.3's np.add.at; force and torque columns are independent.
            blk = np.empty((len(bb) + 1, 6))
            blk[1:, 0:3] = f
            blk[1:, 3:6] = cross_rows(p[loc] - data.xipos[bb], f)                # = np.cross, bit for bit
            cut = np.flatnonzero(bb[1:] != bb[:-1]) + 1
            for r0, r1 in zip(np.concatenate([[0], cut]).tolist(), np.concatenate([cut, [len(bb)]]).tolist()):
                k = int(bb[r0])
                blk[r0] = W[k]                                                     # row r0 is the slot before row r0 + 1
                W[k] = np.add.accumulate(blk[r0:r1 + 1], axis=0)[-1]
                self.item_force[k] = self.item_force.get(k, 0.0) + float(-(f[r0:r1] @ inl.n).sum())

    def wrench(self, model, data) -> np.ndarray:
        """New (nbody, 6) array of the suction wrench."""
        W = np.zeros((model.nbody, 6))
        self.accumulate(model, data, W)
        return W

    def update_networks(self) -> None:
        """Feed the sealed areas back into the networks and re-solve them (call at the network rate)."""
        for net in self.nets:
            for inl in net.inlets:
                net.set_blocked(inl.name, self.covered.get(inl.name, 0.0))
            net.solve()

    def hold_force(self, bodies=None) -> float:
        """Total normal suction force [N] on the given bodies (all if None) at the last evaluation."""
        if bodies is None:
            return float(sum(self.item_force.values()))
        return float(sum(self.item_force.get(int(b), 0.0) for b in np.atleast_1d(bodies)))


def holding_curve(build, inlet_area: float, n_inlets: int = 1, Q_design: float | None = None,
                  phis=np.linspace(0, 0.98, 50), fan_scale="published") -> np.ndarray:
    """Rows (phi, P_c [Pa], Q_total, Q of the blocked inlet) when ONE of n identical square inlets is blocked by
    the fraction phi (the others open)."""
    s = math.sqrt(inlet_area) / 2
    qd = Q_design if Q_design is not None else FILTER_RATED_FLOW * n_inlets
    inls = [Inlet(f"g{i}", (0, 0, 0), (0, 0, 1), (1, 0, 0), (s, s), flow=qd / n_inlets) for i in range(n_inlets)]
    net = FanNetwork(inls, build=build, Q_design=qd, fan_scale=fan_scale)
    out = []
    for phi in phis:
        net.set_blocked("g0", phi * inlet_area)
        net.solve()
        out.append((float(phi), float(net.Pc), float(net.Q), float(inls[0].flow)))
    return np.array(out)
