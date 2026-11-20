"""Cabin air inside a station module: drag relative to a ventilation velocity field, the exhaust jets of
the robot's own nozzles acting on nearby objects, and suction of objects covering a return-air grille.

Everything is a wrench source (``accumulate(model, data, W)``, world axes at body CoMs) or a velocity
field (``field(x) -> (N, 3)``, x in world coordinates, m/s).

Drag:  F = -1/2 rho Cd A(v_hat) |v_r| v_r,  v_r = v_com - u_air(x_cp),  torque = r_cp x F, with A(v_hat)
the projected area of the body's shape along the relative flow (box / sphere / cylinder / capsule /
fixed) and r_cp an optional centre-of-pressure offset; optional rotational drag -c |w| w.

Fields: ``UniformFlow``, ``DuctJet`` (analytic), ``GridFlowField`` (regular grid, trilinear), and
``FlowFieldBuilder`` which builds a divergence-free field inside a box from wall inlets / outlets, interior
sources / sinks and inlet jets: potential flow with the given wall fluxes (exact DCT Poisson solve on a
staggered grid), plus jet profiles projected back to zero divergence; the mass balance is checked.

Jets (``NozzleJets``): each nozzle is a round turbulent jet with momentum flux J = its current thrust,
Gaussian velocity profile u = u_c(x) exp(-r^2 / b^2), b = b0 + k_b x, k_b = 0.094 / sqrt(ln 2) (half-velocity
spreading rate 0.094, Pope 2000 Table 5.1), u_c from momentum conservation J = rho u_c^2 pi b^2 / 2.
The force on a target is the quadratic drag of the jet velocity integrated over the target's silhouette.

Suction (``GrilleSuction``): pressure behind a return grille p_b from a quadratic fan curve and the
grille's open area (closed form); an object whose surface is within ``gap_max`` of the grille plane is
pulled with p_b * w(gap) * dA at every covered sample point (w smooth, 1 in contact, 0 at gap_max).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ._fast import cross, cross_rows
from .frames import body_com_velocity

__all__ = [
    "AIR_DENSITY", "DRAG_COEFF", "DragShape", "shape_from_body", "projected_area", "UniformFlow", "DuctJet",
    "GridFlowField", "FlowFieldBuilder", "ScaledFlow", "SumFlow", "CabinAir", "hold_force", "RoundJet",
    "NozzleJets", "duct_operating_point", "GrilleSuction",
]

AIR_DENSITY = 1.225   # kg/m^3, cabin air (the firmware plant uses 1.2 inside the propulsion model)
# Quadratic drag coefficients at Re ~ 1e3-1e5 (Hoerner 1965; standard tables): sphere 0.47, cube face-on
# 1.05, cylinder in cross flow 1.0 (finite L/D), cylinder axial 0.82 (long) - 1.15 (short), flat plate 1.17.
DRAG_COEFF = {"sphere": 0.47, "box": 1.05, "cylinder_cross": 1.0, "cylinder_axial": 0.9, "plate": 1.17,
              "capsule_cross": 0.9}


class DragShape:
    """Drag description of one body (body frame).

    kind: 'box' (size = half extents (3,)), 'sphere' (size = (r,)), 'cylinder' / 'capsule' (size =
    (r, half_length), axis = body z), 'fixed' (size = (area,), constant reference area).
    cd: drag coefficient (default from DRAG_COEFF; for cylinders / capsules a (cross, axial) pair).
    center: body-frame point where the flow is sampled and the force acts (centre of pressure) relative
    to the body origin; default the body CoM.  rot_coeff: rotational drag tau = -rot_coeff |w| w.
    """

    def __init__(self, kind: str, size, cd=None, center=None, rot_coeff: float = 0.0):
        self.kind = kind
        self.size = np.atleast_1d(np.asarray(size, float))
        if cd is None:
            cd = {"box": DRAG_COEFF["box"], "sphere": DRAG_COEFF["sphere"],
                  "cylinder": (DRAG_COEFF["cylinder_cross"], DRAG_COEFF["cylinder_axial"]),
                  "capsule": (DRAG_COEFF["capsule_cross"], DRAG_COEFF["sphere"]), "fixed": DRAG_COEFF["box"]}[kind]
        self.cd = cd
        self.center = None if center is None else np.asarray(center, float)
        self.rot_coeff = float(rot_coeff)

    def cd_area(self, u_body) -> float:
        """Cd * A [m^2] for a unit relative-flow direction u_body (body frame)."""
        return projected_area(self.kind, self.size, u_body, self.cd)


def projected_area(kind: str, size, u, cd=None) -> float:
    """Projected area along unit direction u (body frame), weighted by cd if given (returns Cd*A)."""
    u = np.asarray(u, float)
    n = float(np.linalg.norm(u))
    u = u / n if n > 0 else np.array([1.0, 0.0, 0.0])
    s = np.asarray(size, float)
    if kind == "box":
        a, b, c = s
        A = 4.0 * (b * c * abs(u[0]) + a * c * abs(u[1]) + a * b * abs(u[2]))
        return A * (1.0 if cd is None else float(cd))
    if kind == "sphere":
        return math.pi * s[0] ** 2 * (1.0 if cd is None else float(cd))
    if kind in ("cylinder", "capsule"):
        r, h = s[0], s[1]
        uz = abs(u[2])
        sin = math.sqrt(max(0.0, 1.0 - uz * uz))
        cross, axial = (1.0, 1.0) if cd is None else (float(cd[0]), float(cd[1]))
        if kind == "cylinder":
            return cross * 4.0 * r * h * sin + axial * math.pi * r * r * uz
        return cross * 4.0 * r * h * sin + axial * math.pi * r * r
    if kind == "fixed":
        return float(s[0]) * (1.0 if cd is None else float(cd))
    raise ValueError(kind)


def shape_from_body(model, body: int, cd=None, rot_coeff: float = 0.0) -> DragShape:
    """Box DragShape spanning the body's collision geoms (body-frame AABB; single sphere -> sphere)."""
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
    coll = [g for g in gids if model.geom_contype[g] or model.geom_conaffinity[g]] or gids
    if len(coll) == 1 and model.geom_type[coll[0]] == mujoco.mjtGeom.mjGEOM_SPHERE:
        return DragShape("sphere", [model.geom_size[coll[0]][0]], cd=cd,
                         center=model.geom_pos[coll[0]].copy(), rot_coeff=rot_coeff)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in coll:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        R = R.reshape(3, 3)
        ext = _geom_half_extents(model, g)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * ext
        pts = model.geom_pos[g] + corners @ R.T
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    return DragShape("box", 0.5 * (hi - lo), cd=cd, center=0.5 * (hi + lo), rot_coeff=rot_coeff)


def _geom_half_extents(model, g) -> np.ndarray:
    t = model.geom_type[g]
    s = model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return s.copy()
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.full(3, s[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return np.array([s[0], s[0], s[1] + s[0]])
    if t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return np.array([s[0], s[0] if t == mujoco.mjtGeom.mjGEOM_CYLINDER else s[1], s[1] if t == mujoco.mjtGeom.mjGEOM_CYLINDER else s[2]])
    return np.full(3, model.geom_rbound[g])


# ------------------------------------------------------------------------------------------------ fields
class UniformFlow:
    """Constant air velocity u [m/s] everywhere."""

    def __init__(self, u=(0.0, 0.0, 0.0)):
        self.u = np.asarray(u, float)

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        return np.broadcast_to(self.u, x.shape).copy()


class DuctJet:
    """Analytic ventilation jet along +axis from a diffuser at ``origin`` (Gaussian cross profile,
    width r0 + spread * s, centre speed u0 r0 / width), floored at ``background`` along the axis."""

    def __init__(self, origin, axis=(1, 0, 0), u0=0.2, r0=0.15, spread=0.1, background=0.02):
        self.o = np.asarray(origin, float)
        self.a = np.asarray(axis, float) / np.linalg.norm(axis)
        self.u0, self.r0, self.k, self.bg = u0, r0, spread, background

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        d = x - self.o
        s = np.clip(d @ self.a, 0.0, None)
        radial = np.linalg.norm(d - s[:, None] * self.a, axis=1)
        width = self.r0 + self.k * s
        speed = self.u0 * self.r0 / width * np.exp(-(radial / width) ** 2)
        return np.maximum(speed, self.bg)[:, None] * self.a


class ScaledFlow:
    """field(x) * scale, with ``scale`` a float attribute that callers may change (e.g. fan state)."""

    def __init__(self, field, scale: float = 1.0):
        self.field, self.scale = field, float(scale)

    def __call__(self, x) -> np.ndarray:
        return self.scale * self.field(x)


class SumFlow:
    """Sum of several fields."""

    def __init__(self, *fields):
        self.fields = fields

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        out = np.zeros_like(x)
        for f in self.fields:
            out += f(x)
        return out


class GridFlowField:
    """Velocity field sampled on a regular grid, trilinear interpolation.

    origin: world position of sample [0, 0, 0]; spacing: (3,) or scalar [m]; values: (nx, ny, nz, 3).
    outside: 'clamp' (nearest boundary sample, default), 'zero', or 'background' (constant ``background``).
    """

    def __init__(self, origin, spacing, values, outside: str = "clamp", background=(0.0, 0.0, 0.0)):
        self.origin = np.asarray(origin, float)
        self.h = np.broadcast_to(np.asarray(spacing, float), (3,)).copy()
        self.values = np.asarray(values, float)
        if self.values.ndim != 4 or self.values.shape[3] != 3:
            raise ValueError("values must have shape (nx, ny, nz, 3)")
        self.n = np.array(self.values.shape[:3])
        self.outside = outside
        self.background = np.asarray(background, float)
        self.hi = self.origin + (self.n - 1) * self.h

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        f = (x - self.origin) / self.h
        inside = np.all((f >= -1e-9) & (f <= self.n - 1 + 1e-9), axis=1)
        f = np.clip(f, 0.0, self.n - 1)
        i0 = np.minimum(np.floor(f).astype(int), np.maximum(self.n - 2, 0))
        t = f - i0
        i1 = np.minimum(i0 + 1, self.n - 1)
        V = self.values
        out = np.zeros((len(x), 3))
        for cx in (0, 1):
            ix = i1[:, 0] if cx else i0[:, 0]
            wx = t[:, 0] if cx else 1.0 - t[:, 0]
            for cy in (0, 1):
                iy = i1[:, 1] if cy else i0[:, 1]
                wy = t[:, 1] if cy else 1.0 - t[:, 1]
                for cz in (0, 1):
                    iz = i1[:, 2] if cz else i0[:, 2]
                    wz = t[:, 2] if cz else 1.0 - t[:, 2]
                    out += (wx * wy * wz)[:, None] * V[ix, iy, iz]
        if self.outside == "zero":
            out[~inside] = 0.0
        elif self.outside == "background":
            out[~inside] = self.background
        return out


def _dct2(x, axis):
    """Unnormalised DCT-II along axis via FFT of the even extension: X_k = sum x_n cos(pi k (n+1/2)/N)."""
    x = np.moveaxis(x, axis, -1)
    N = x.shape[-1]
    y = np.concatenate([x, x[..., ::-1]], axis=-1)
    Y = np.fft.fft(y, axis=-1)[..., :N]
    k = np.arange(N)
    X = np.real(Y * np.exp(-1j * np.pi * k / (2 * N))) / 2.0
    return np.moveaxis(X, -1, axis)


def _idct2(X, axis):
    """Inverse of _dct2."""
    X = np.moveaxis(X, axis, -1)
    N = X.shape[-1]
    k = np.arange(N)
    Yh = 2.0 * np.exp(1j * np.pi * k / (2 * N)) * X
    Y = np.zeros(X.shape[:-1] + (2 * N,), complex)
    Y[..., :N] = Yh
    Y[..., N + 1:] = np.conj(Yh[..., 1:][..., ::-1])
    y = np.real(np.fft.ifft(Y, axis=-1))[..., :N]
    return np.moveaxis(y, -1, axis)


def _neumann_poisson(rhs):
    """Solve L phi = rhs, L = sum over interior cell faces (phi_nb - phi_c) (Neumann Laplacian, unit
    spacing), mean(phi) = 0.  rhs must sum to ~0.  Exact (DCT diagonalisation)."""
    nx, ny, nz = rhs.shape
    R = _dct2(_dct2(_dct2(rhs, 0), 1), 2)
    lam = [2.0 * np.cos(np.pi * np.arange(n) / n) - 2.0 for n in (nx, ny, nz)]
    L = lam[0][:, None, None] + lam[1][None, :, None] + lam[2][None, None, :]
    L[0, 0, 0] = 1.0
    P = R / L
    P[0, 0, 0] = 0.0
    return _idct2(_idct2(_idct2(P, 0), 1), 2)


class FlowFieldBuilder:
    """Simple ventilation field inside an axis-aligned box [lo, hi] (cell size h).

    add_inlet / add_outlet: rectangular wall patches with a volume flow (m^3/s); an inlet may also be a
    jet (its momentum is added as a round-jet profile, then projected to zero divergence).
    add_source: interior point source (Q > 0) or sink (Q < 0).  build() checks the mass balance
    (sum inflow = sum outflow within rtol, else ValueError) and returns (GridFlowField, report).
    """

    _AX = {"-x": (0, 0), "+x": (0, 1), "-y": (1, 0), "+y": (1, 1), "-z": (2, 0), "+z": (2, 1)}

    def __init__(self, lo, hi, h: float = 0.05):
        self.lo, self.hi, self.h = np.asarray(lo, float), np.asarray(hi, float), float(h)
        self.n = np.maximum(1, np.round((self.hi - self.lo) / self.h).astype(int))
        self.hi = self.lo + self.n * self.h
        self.patches = []
        self.sources = []
        self.jets = []

    def add_inlet(self, wall: str, center, half_size, flow: float, jet: dict | None = None):
        """Inlet on wall '-x'...'+z': rectangle at ``center`` (the two in-plane coordinates are used),
        ``half_size`` (2,) in the in-plane axes (sorted), volume flow ``flow`` > 0 into the box.
        jet: optional dict(speed=..., spread=0.094) to add a round-jet momentum profile."""
        self.patches.append((wall, np.asarray(center, float), np.asarray(half_size, float), -abs(flow)))
        if jet is not None:
            self.jets.append((wall, np.asarray(center, float), np.asarray(half_size, float), abs(flow), dict(jet)))

    def add_outlet(self, wall: str, center, half_size, flow: float):
        """Outlet (return grille) on a wall; ``flow`` > 0 leaves the box."""
        self.patches.append((wall, np.asarray(center, float), np.asarray(half_size, float), abs(flow)))

    def add_source(self, pos, flow: float):
        """Interior point source (flow > 0) or sink (flow < 0), m^3/s."""
        self.sources.append((np.asarray(pos, float), float(flow)))

    def _wall_flux(self):
        """Outward normal velocity on every boundary face: dict wall -> (n1, n2) array [m/s]."""
        h = self.h
        out = {}
        Q_in = Q_out = 0.0
        for wall, (ax, side) in self._AX.items():
            others = [a for a in range(3) if a != ax]
            g = np.zeros((self.n[others[0]], self.n[others[1]]))
            for w, c, hs, q in self.patches:
                if w != wall:
                    continue
                c1 = self.lo[others[0]] + (np.arange(self.n[others[0]]) + 0.5) * h
                c2 = self.lo[others[1]] + (np.arange(self.n[others[1]]) + 0.5) * h
                m1 = np.abs(c1 - c[others[0]]) <= hs[0] + 1e-12
                m2 = np.abs(c2 - c[others[1]]) <= hs[1] + 1e-12
                mask = m1[:, None] & m2[None, :]
                nf = int(mask.sum())
                if nf == 0:
                    raise ValueError(f"patch on {wall} at {c} covers no boundary face (h = {h})")
                g[mask] += q / (nf * h * h)
                if q < 0:
                    Q_in += -q
                else:
                    Q_out += q
            out[wall] = g
        return out, Q_in, Q_out

    def build(self, rtol: float = 1e-3, project_jets: bool = True):
        """Return (GridFlowField on cell centres, report dict) for the configured patches / sources."""
        h, n = self.h, self.n
        g, Q_in, Q_out = self._wall_flux()
        src = np.zeros(tuple(n))
        for p, q in self.sources:
            idx = np.clip(((p - self.lo) / h).astype(int), 0, n - 1)
            src[tuple(idx)] += q
            if q > 0:
                Q_in += q
            else:
                Q_out += -q
        net = Q_in - Q_out
        if abs(net) > rtol * max(Q_in, Q_out, 1e-12):
            raise ValueError(f"mass balance violated: inflow {Q_in:.6g} m^3/s, outflow {Q_out:.6g} m^3/s")
        # cell rhs: L phi = (s_c - sum_boundary g h^2) / h  (see module notes)
        rhs = src.copy()
        rhs[0, :, :] -= g["-x"] * h * h
        rhs[-1, :, :] -= g["+x"] * h * h
        rhs[:, 0, :] -= g["-y"] * h * h
        rhs[:, -1, :] -= g["+y"] * h * h
        rhs[:, :, 0] -= g["-z"] * h * h
        rhs[:, :, -1] -= g["+z"] * h * h
        rhs /= h
        rhs -= rhs.mean()                      # remove the round-off part of the (checked) imbalance
        phi = _neumann_poisson(rhs)
        # staggered face velocities (u on x-faces etc.), boundary faces prescribed
        ux = np.zeros((n[0] + 1, n[1], n[2]))
        uy = np.zeros((n[0], n[1] + 1, n[2]))
        uz = np.zeros((n[0], n[1], n[2] + 1))
        ux[1:-1] = (phi[1:] - phi[:-1]) / h
        uy[:, 1:-1] = (phi[:, 1:] - phi[:, :-1]) / h
        uz[:, :, 1:-1] = (phi[:, :, 1:] - phi[:, :, :-1]) / h
        ux[0], ux[-1] = -g["-x"], g["+x"]
        uy[:, 0], uy[:, -1] = -g["-y"], g["+y"]
        uz[:, :, 0], uz[:, :, -1] = -g["-z"], g["+z"]
        if self.jets:
            jx, jy, jz = self._jet_faces()
            ux[1:-1] += jx[1:-1]
            uy[:, 1:-1] += jy[:, 1:-1]
            uz[:, :, 1:-1] += jz[:, :, 1:-1]
            if project_jets:
                div = ((ux[1:] - ux[:-1]) + (uy[:, 1:] - uy[:, :-1]) + (uz[:, :, 1:] - uz[:, :, :-1])) * h * h - src
                psi = _neumann_poisson((div - div.mean()) / h)
                ux[1:-1] -= (psi[1:] - psi[:-1]) / h
                uy[:, 1:-1] -= (psi[:, 1:] - psi[:, :-1]) / h
                uz[:, :, 1:-1] -= (psi[:, :, 1:] - psi[:, :, :-1]) / h
        div = ((ux[1:] - ux[:-1]) + (uy[:, 1:] - uy[:, :-1]) + (uz[:, :, 1:] - uz[:, :, :-1])) * h * h - src
        wall_net = (ux[-1].sum() - ux[0].sum() + uy[:, -1].sum() - uy[:, 0].sum()
                    + uz[:, :, -1].sum() - uz[:, :, 0].sum()) * h * h
        cell = np.stack([0.5 * (ux[1:] + ux[:-1]), 0.5 * (uy[:, 1:] + uy[:, :-1]), 0.5 * (uz[:, :, 1:] + uz[:, :, :-1])], -1)
        field = GridFlowField(self.lo + 0.5 * h, h, cell)
        report = dict(inflow_m3s=Q_in, outflow_m3s=Q_out, net_m3s=net, wall_net_outflow_m3s=float(wall_net),
                      max_cell_divergence_m3s=float(np.max(np.abs(div))), shape=tuple(int(v) for v in n),
                      max_speed_ms=float(np.max(np.linalg.norm(cell, axis=-1))))
        self.faces = (ux, uy, uz)
        return field, report

    def _jet_faces(self):
        """Round-jet velocity components on the staggered faces for every jet inlet."""
        h, n, lo = self.h, self.n, self.lo
        xs = [lo[a] + np.arange(n[a] + 1) * h for a in range(3)]
        cs = [lo[a] + (np.arange(n[a]) + 0.5) * h for a in range(3)]
        comps = []
        for ax in range(3):
            grids = [cs[0], cs[1], cs[2]]
            grids[ax] = xs[ax]
            X, Y, Z = np.meshgrid(*grids, indexing="ij")
            P = np.stack([X, Y, Z], -1).reshape(-1, 3)
            u = np.zeros(len(P))
            for wall, c, hs, q, opt in self.jets:
                a, side = self._AX[wall]
                normal = np.zeros(3)
                normal[a] = -1.0 if side == 1 else 1.0        # into the box
                area = 4.0 * hs[0] * hs[1]
                speed = opt.get("speed", q / area)
                jet = RoundJet(c, normal, momentum_flux=AIR_DENSITY * speed * speed * area, exit_area=area,
                               spread=opt.get("spread", 0.094))
                u += jet.velocity(P)[:, ax]
            comps.append(u.reshape(X.shape))
        return comps


class CabinAir:
    """Quadratic drag of selected bodies relative to the air velocity field (wrench source).

    bodies: body ids; shapes: list of DragShape (one per body) or None for ``shape_from_body``;
    flow: callable field (default still air); rho: air density.
    """

    def __init__(self, model, bodies, shapes=None, flow=None, rho: float = AIR_DENSITY):
        self.bodies = np.asarray(bodies, int)
        self.shapes = [shape_from_body(model, int(b)) for b in self.bodies] if shapes is None else list(shapes)
        if len(self.shapes) != len(self.bodies):
            raise ValueError("one DragShape per body")
        self.flow = flow if flow is not None else UniformFlow()
        self.rho = rho
        self.enabled = True

    def accumulate(self, model, data, W) -> None:
        """Add drag forces / torques into W."""
        if not self.enabled or len(self.bodies) == 0:
            return
        b = self.bodies
        vcom = body_com_velocity(model, data)[b]
        R = data.xmat[b].reshape(-1, 3, 3)
        pts = data.xipos[b].copy()
        offs = np.zeros((len(b), 3))
        for k, s in enumerate(self.shapes):
            if s.center is not None:
                offs[k] = R[k] @ (s.center - model.body_ipos[b[k]])
        pts += offs
        w = data.cvel[b, 0:3]
        v = vcom + cross_rows(w, offs)
        u = self.flow(pts)
        vr = v - u
        for k, s in enumerate(self.shapes):
            sp = float(np.linalg.norm(vr[k]))
            if sp > 0:
                cda = s.cd_area(R[k].T @ (vr[k] / sp))
                F = -0.5 * self.rho * cda * sp * vr[k]
                W[b[k], 0:3] += F
                W[b[k], 3:6] += cross(offs[k], F)
            if s.rot_coeff:
                W[b[k], 3:6] += -s.rot_coeff * np.linalg.norm(w[k]) * w[k]


def hold_force(u: float, cd_area: float = 1.05 * 0.092903, rho: float = AIR_DENSITY) -> float:
    """Thrust [N] needed to hold station in a steady flow of speed u for a body with Cd*A."""
    return 0.5 * rho * cd_area * u * u


# -------------------------------------------------------------------------------------------- jets
class RoundJet:
    """Momentum-conserving round turbulent jet from ``origin`` along unit ``direction``.

    momentum_flux J [N] (= thrust of the nozzle), exit_area A_e [m^2] (sets the start width b0 =
    sqrt(2 A_e / pi)), spread = half-velocity spreading rate (0.094), rho air density.
    Velocity u(x, r) = u_c(x) exp(-r^2 / b(x)^2), b = b0 + spread / sqrt(ln 2) * x,
    u_c = sqrt(2 J / (rho pi b^2)); zero behind the exit plane (x < 0).
    """

    def __init__(self, origin, direction, momentum_flux: float, exit_area: float, spread: float = 0.094,
                 rho: float = AIR_DENSITY):
        self.o = np.asarray(origin, float)
        d = np.asarray(direction, float)
        self.d = d / np.linalg.norm(d)
        self.J, self.rho = float(momentum_flux), rho
        self.b0 = math.sqrt(2.0 * float(exit_area) / math.pi)
        self.kb = spread / math.sqrt(math.log(2.0))

    def width(self, x):
        """1/e velocity radius b(x) [m]."""
        return self.b0 + self.kb * np.maximum(np.asarray(x, float), 0.0)

    def centre_speed(self, x):
        """Centreline speed u_c(x) [m/s]."""
        b = self.width(x)
        return np.sqrt(2.0 * max(self.J, 0.0) / (self.rho * math.pi * b * b))

    def velocity(self, P) -> np.ndarray:
        """(N, 3) jet velocity at points P (N, 3)."""
        P = np.atleast_2d(np.asarray(P, float))
        rel = P - self.o
        x = rel @ self.d
        r2 = np.einsum("ij,ij->i", rel, rel) - x * x
        b = self.width(x)
        u = self.centre_speed(x) * np.exp(-np.maximum(r2, 0.0) / (b * b))
        u = np.where(x > 0.0, u, 0.0)
        return u[:, None] * self.d[None, :]


class NozzleJets:
    """Wrench source: the robot's nozzle exhaust blowing on target bodies.

    robot_body: body id carrying the nozzles; nozzle_pos / nozzle_dir: (12, 3) body-frame exit positions
    and exhaust directions; thrust_fn(): returns the current (12,) thrusts (e.g. FanPropulsion.thrust);
    exit_area_fn(): current (12,) open areas (default constant 0.002 m^2); targets: body ids with their
    DragShape (default shape_from_body); n_samples: silhouette grid per side (default 5 -> 25 points; the
    projected area is spread uniformly over the silhouette's bounding rectangle); occlusion: cast one
    ray nozzle -> target and skip blocked pairs; max_range [m]; ambient: the ventilation field, so that
    only the increment over the ambient-air drag is applied (no double counting with CabinAir).
    """

    def __init__(self, model, robot_body: int, nozzle_pos, nozzle_dir, thrust_fn, targets, shapes=None,
                 exit_area_fn=None, n_samples: int = 5, occlusion: bool = True, max_range: float = 3.0,
                 spread: float = 0.094, rho: float = AIR_DENSITY, min_thrust: float = 1e-4, ambient=None):
        self.robot = int(robot_body)
        self.pos = np.asarray(nozzle_pos, float)
        self.dir = np.asarray(nozzle_dir, float) / np.linalg.norm(nozzle_dir, axis=1, keepdims=True)
        self.thrust_fn = thrust_fn
        self.exit_area_fn = exit_area_fn
        self.targets = np.asarray(targets, int)
        self.shapes = [shape_from_body(model, int(b)) for b in self.targets] if shapes is None else list(shapes)
        self.ns = int(n_samples)
        self.occlusion = occlusion
        self.max_range = max_range
        self.spread, self.rho, self.min_thrust = spread, rho, min_thrust
        self.ambient = ambient
        self.enabled = True
        self.last_forces = np.zeros((len(self.targets), 3))
        self._geomid = np.zeros(1, np.int32)

    def _silhouette(self, shape, R, center, d):
        """Sample points (N, 3) and the area per point on the target's silhouette seen along d."""
        e1 = cross(d, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = cross(d, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = cross(d, e1)
        d_body = R.T @ d
        area = projected_area(shape.kind, shape.size, d_body)
        if shape.kind == "box":
            axes = R @ np.diag(shape.size)             # columns: half-axis vectors in world
            r1 = float(np.sum(np.abs(e1 @ axes)))
            r2 = float(np.sum(np.abs(e2 @ axes)))
        else:
            r1 = r2 = math.sqrt(area / math.pi)
        g = (np.arange(self.ns) + 0.5) / self.ns * 2.0 - 1.0
        A, Bm = np.meshgrid(g * r1, g * r2, indexing="ij")
        pts = center + A.reshape(-1, 1) * e1 + Bm.reshape(-1, 1) * e2
        return pts, area / (self.ns * self.ns)

    def accumulate(self, model, data, W) -> None:
        """Add jet forces on the targets into W."""
        self.last_forces[:] = 0.0
        if not self.enabled or len(self.targets) == 0:
            return
        thrust = np.asarray(self.thrust_fn(), float)
        if not np.any(thrust > self.min_thrust):
            return
        areas = (np.full(len(thrust), 0.002) if self.exit_area_fn is None else np.asarray(self.exit_area_fn(), float))
        Rr = data.xmat[self.robot].reshape(3, 3)
        P = data.xpos[self.robot] + self.pos @ Rr.T
        D = self.dir @ Rr.T
        vcom = body_com_velocity(model, data)
        for k, b in enumerate(self.targets):
            s = self.shapes[k]
            Rb = data.xmat[b].reshape(3, 3)
            c = data.xpos[b] + Rb @ (s.center if s.center is not None else model.body_ipos[b])
            for i in np.nonzero(thrust > self.min_thrust)[0]:
                rel = c - P[i]
                x = float(rel @ D[i])
                if x <= 0.0 or np.linalg.norm(rel) > self.max_range:
                    continue
                jet = RoundJet(P[i], D[i], thrust[i], max(areas[i], 1e-6), self.spread, self.rho)
                if math.sqrt(max(float(rel @ rel) - x * x, 0.0)) > 3.0 * float(jet.width(x)) + np.max(s.size):
                    continue
                if self.occlusion:
                    dist = np.linalg.norm(rel)
                    hit = mujoco.mj_ray(model, data, P[i] + D[i] * 1e-3, rel / dist, None, 1, self.robot, self._geomid)
                    if hit >= 0 and hit < dist - np.max(s.size) - 1e-3 and model.geom_bodyid[self._geomid[0]] != b:
                        continue
                pts, dA = self._silhouette(s, Rb, c, D[i])
                u = jet.velocity(pts)
                w0 = vcom[b] - (self.ambient(c[None, :])[0] if self.ambient is not None else 0.0)
                vr = w0[None, :] - u
                sp = np.linalg.norm(vr, axis=1)
                cd = s.cd if np.ndim(s.cd) == 0 else s.cd[0]
                # increment over the drag in the ambient air (that part belongs to CabinAir)
                F = -0.5 * self.rho * float(cd) * (sp[:, None] * vr - np.linalg.norm(w0) * w0[None, :]) * dA
                Ftot = F.sum(0)
                W[b, 0:3] += Ftot
                W[b, 3:6] += cross_rows(pts - data.xipos[b], F).sum(0)
                self.last_forces[k] += Ftot


# ------------------------------------------------------------------------------------------ suction
def duct_operating_point(open_area: float, p_shutoff: float, q_free: float, cd: float = 0.6,
                         rho: float = AIR_DENSITY):
    """Return (Q [m^3/s], p_behind [Pa]) of a return duct: fan curve p = p0 (1 - (Q/Q0)^2) in series with
    the grille orifice Q = cd A_open sqrt(2 p / rho).  Closed form; A_open -> 0 gives (0, p0)."""
    k = 2.0 * p_shutoff / rho * (cd * max(open_area, 0.0)) ** 2
    q2 = k / (1.0 + k / (q_free * q_free))
    return math.sqrt(q2), p_shutoff * (1.0 - q2 / (q_free * q_free))


class GrilleSuction:
    """Suction on objects covering part of a return-air grille (wrench source).

    center: grille centre (world); normal: unit normal pointing out of the grille into the cabin;
    half_size: (2,) half extents along the in-plane axes u (= ``u_axis``) and v = normal x u;
    porosity: free-area ratio of the grille face; p_shutoff, q_free, cd: duct model
    (duct_operating_point).  targets: body ids that can be sucked; samples: grid points per side;
    gap_max: distance over which the suction fades (w = (1 - gap/gap_max)^2, gap <= 0 -> 1); friction: None
    (rely on MuJoCo contact friction with the grille geom) or mu for a regularised model
    F_t = -mu N tanh(|v_t| / v_slip) v_t / |v_t| acting with the suction.  Rays start ``back`` behind the
    grille plane (so touching or slightly penetrating objects are seen) and ignore ``exclude_body`` (the
    static station, default world body 0).
    """

    def __init__(self, model, center, normal, u_axis, half_size, targets, porosity: float = 0.7,
                 p_shutoff: float = 150.0, q_free: float = 0.12, cd: float = 0.6, samples: int = 16,
                 gap_max: float = 0.01, friction: float | None = None, v_slip: float = 0.005,
                 rho: float = AIR_DENSITY, geomgroup=None, exclude_body: int = 0, back: float = 0.005):
        self.c = np.asarray(center, float)
        n = np.asarray(normal, float)
        self.n = n / np.linalg.norm(n)
        u = np.asarray(u_axis, float)
        u = u - (u @ self.n) * self.n
        self.u = u / np.linalg.norm(u)
        self.v = np.cross(self.n, self.u)
        self.hs = np.asarray(half_size, float)
        self.targets = np.asarray(targets, int)
        self.porosity, self.p0, self.q0, self.cd, self.rho = porosity, p_shutoff, q_free, cd, rho
        self.ns = int(samples)
        g = (np.arange(self.ns) + 0.5) / self.ns * 2.0 - 1.0
        A, B = np.meshgrid(g * self.hs[0], g * self.hs[1], indexing="ij")
        self.pts = self.c + A.reshape(-1, 1) * self.u + B.reshape(-1, 1) * self.v
        self.dA = 4.0 * self.hs[0] * self.hs[1] / (self.ns * self.ns)
        self.gap_max = gap_max
        self.friction, self.v_slip = friction, v_slip
        self.enabled = True
        self._geomid = np.zeros(1, np.int32)
        self.state = dict(covered_area=0.0, open_area=self.porosity * self.area, flow=0.0, p_behind=0.0)
        self.last_forces = np.zeros((len(self.targets), 3))
        self.geomgroup = None if geomgroup is None else np.asarray(geomgroup, np.uint8)
        self.exclude_body, self.back = int(exclude_body), float(back)

    @property
    def area(self) -> float:
        """Gross grille area [m^2]."""
        return 4.0 * self.hs[0] * self.hs[1]

    def coverage(self, model, data):
        """(weights (n_targets, n_samples), hit points (n_targets, n_samples, 3)) of the grille samples."""
        nt, npnt = len(self.targets), len(self.pts)
        Wt = np.zeros((nt, npnt))
        H = np.zeros((nt, npnt, 3))
        start = self.pts - self.n * self.back
        for j in range(npnt):
            d = mujoco.mj_ray(model, data, start[j], self.n, self.geomgroup, 1, self.exclude_body, self._geomid)
            if d < 0:
                continue
            gap = d - self.back
            if gap > self.gap_max:
                continue
            bid = model.geom_bodyid[self._geomid[0]]
            hit = np.nonzero(self.targets == bid)[0]
            if len(hit):
                k = hit[0]
                Wt[k, j] = 1.0 if gap <= 0 else (1.0 - gap / self.gap_max) ** 2
                H[k, j] = start[j] + self.n * d
        return Wt, H

    def accumulate(self, model, data, W) -> None:
        """Add suction (and optional regularised friction) on the covering targets into W."""
        self.last_forces[:] = 0.0
        if not self.enabled or len(self.targets) == 0:
            return
        Wt, H = self.coverage(model, data)
        covered = float(np.sum(Wt) * self.dA)
        open_area = self.porosity * max(self.area - covered, 0.0)
        q, pb = duct_operating_point(open_area, self.p0, self.q0, self.cd, self.rho)
        self.state = dict(covered_area=covered, open_area=open_area, flow=q, p_behind=pb)
        if covered <= 0.0:
            return
        vcom = body_com_velocity(model, data)
        for k, b in enumerate(self.targets):
            wk = Wt[k]
            if not np.any(wk > 0):
                continue
            dF = -(pb * wk * self.dA)[:, None] * self.n[None, :]
            Ftot = dF.sum(0)
            tau = cross_rows(H[k][wk > 0] - data.xipos[b], dF[wk > 0]).sum(0)
            if self.friction is not None:
                N = float(np.linalg.norm(Ftot))
                vt = vcom[b] - (vcom[b] @ self.n) * self.n
                s = float(np.linalg.norm(vt))
                if s > 0:
                    Ftot = Ftot - self.friction * N * math.tanh(s / self.v_slip) * vt / s
            W[b, 0:3] += Ftot
            W[b, 3:6] += tau
            self.last_forces[k] = Ftot

    def flow_scale(self, q_nominal: float | None = None) -> float:
        """Current duct flow relative to the uncovered grille (for ScaledFlow of the ventilation field)."""
        q_open, _ = duct_operating_point(self.porosity * self.area, self.p0, self.q0, self.cd, self.rho)
        return self.state["flow"] / (q_nominal if q_nominal else q_open)
