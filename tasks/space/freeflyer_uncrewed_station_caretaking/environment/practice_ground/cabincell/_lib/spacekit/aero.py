"""Aerodynamic forces on loose items in cabin air, and the free-flyer's nozzle exhaust as a velocity field.

Item drag (``ItemAero``): every item is described by one or more ``ItemShape`` (box / plate, cylinder, capsule,
sphere) in its body frame; each shape is represented by sample points that carry a share of its projected areas.
At every point the relative velocity v_r = v_point - u_air(x) (u_air = ventilation field + optional jet field) gives
    box:      F_j = -1/2 rho Cd A_j |v_rj| v_rj        per local axis j (A_j = face area normal to axis j / n_points)
    cylinder: F_perp = -1/2 rho Cd_cross (2 r L / n) |v_perp| v_perp,  F_ax = -1/2 rho Cd_ax (pi r^2 / n) |v_ax| v_ax
    sphere:   F = -1/2 rho Cd pi r^2 |v_r| v_r
summed as a wrench about the body CoM (flow gradients and spin give torques by themselves).  Point velocities use
MuJoCo's ``cvel`` (linear part at the subtree CoM of the tree root), valid after ``mj_step1``.

Exhaust jets (``jet_velocity``, ``NozzleJetField``): a round turbulent jet with the momentum flux of the nozzle
thrust T and exit velocity v0: exit area A = T / (rho v0^2), d0 = sqrt(4 A / pi); core u = v0 for s < 6.2 d0 (1/e
radius d0 / sqrt 2), then u_c = v0 6.2 d0 / s with a Gaussian profile exp(-(r / 0.114 s)^2) (B c = 1/sqrt 2
conserves the momentum flux).  The exit velocity follows the plenum pressure: v0 = cd_exit sqrt(2 dp / rho).
The jet field is added to the ambient air at the item points, so there is no double counting with item drag.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["AIR_DENSITY", "ITEM_CD", "ItemShape", "ItemAero", "jet_velocity", "NozzleJetField", "StillAir",
           "UniformAir", "settle_time_closed_form", "Shelters"]

AIR_DENSITY = 1.2       # kg/m^3 (spacekit.cabin uses 1.225 for the robot's own drag)
ITEM_CD = {"box": 1.2, "plate": 1.2, "cube": 1.05, "cylinder_cross": 1.0, "cylinder_axial": 0.9, "sphere": 0.47}
JET_B = 6.2             # potential-core length / far-field decay constant (u_c = v0 B d0 / s)
JET_C = 0.114           # 1/e radius growth (r_e = c s); B c = 1/sqrt(2) conserves momentum flux


class StillAir:
    """Zero velocity field."""

    def __call__(self, x) -> np.ndarray:
        return np.zeros_like(np.atleast_2d(np.asarray(x, float)))


class UniformAir:
    """Uniform velocity field u (world, m/s)."""

    def __init__(self, u):
        self.u = np.asarray(u, float)

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        return np.broadcast_to(self.u, x.shape).copy()


class ItemShape:
    """Drag description of one shape of an item body (body frame).

    kind: 'box' (size = half extents (3,), grid (nx, ny, nz) sample points), 'cylinder' / 'capsule' (size = (r,
    half length), axis = local z, n points along the axis), 'sphere' (size = (r,)).  offset / quat_wxyz: shape
    frame relative to the body frame.  cd: drag coefficient (box: scalar; cylinder: (cross, axial)).
    """

    def __init__(self, body: int, kind: str, size, grid=None, cd=None, offset=(0.0, 0.0, 0.0), quat_wxyz=(1, 0, 0, 0)):
        self.body = int(body)
        self.kind = kind
        self.size = np.atleast_1d(np.asarray(size, float))
        self.offset = np.asarray(offset, float)
        w, x, y, z = (float(v) for v in quat_wxyz)
        self.R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                           [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                           [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        if kind == "box":
            self.grid = tuple(grid) if grid is not None else (2, 2, 1)
            self.cd = float(cd) if cd is not None else ITEM_CD["box"]
        elif kind in ("cylinder", "capsule"):
            self.grid = (int(grid) if grid is not None else 4,)
            c = cd if cd is not None else (ITEM_CD["cylinder_cross"], ITEM_CD["cylinder_axial"])
            self.cd = tuple(float(v) for v in np.broadcast_to(np.asarray(c, float), (2,)))
        elif kind == "sphere":
            self.grid = (1,)
            self.cd = float(cd) if cd is not None else ITEM_CD["sphere"]
        else:
            raise ValueError(f"unknown shape kind {kind}")

    def samples(self):
        """(points (n, 3) in the body frame, axis areas*cd (n, 3) in the shape frame, model id per point)."""
        if self.kind == "box":
            hx, hy, hz = self.size
            g = [((np.arange(n) + 0.5) / n * 2 - 1) * h for n, h in zip(self.grid, (hx, hy, hz))]
            pts = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
            area = np.array([4 * hy * hz, 4 * hx * hz, 4 * hx * hy]) / len(pts)
            ca = np.broadcast_to(area * self.cd, pts.shape).copy()
            kind = np.zeros(len(pts), int)
        elif self.kind in ("cylinder", "capsule"):
            r, h = self.size[0], self.size[1]
            n = self.grid[0]
            z = ((np.arange(n) + 0.5) / n * 2 - 1) * h
            pts = np.column_stack([np.zeros(n), np.zeros(n), z])
            L = 2 * h + (2 * r if self.kind == "capsule" else 0.0)
            ca = np.tile([2 * r * L / n * self.cd[0], 2 * r * L / n * self.cd[0], math.pi * r * r / n * self.cd[1]], (n, 1))
            kind = np.ones(n, int)
        else:
            r = self.size[0]
            pts = np.zeros((1, 3))
            ca = np.full((1, 3), math.pi * r * r * self.cd)
            kind = np.full(1, 2, int)
        return pts @ self.R.T + self.offset, ca, kind


class Shelters:
    """Closed volumes the cabin air and the robot's jets do not reach (stowage pockets): oriented boxes
    (centre (3,), axes R (3, 3) with the box axes as columns, half sizes (3,)).  ``mask(p)`` -> bool per point."""

    def __init__(self, boxes=()):
        boxes = list(boxes)
        self.c = np.array([np.asarray(b[0], float) for b in boxes]).reshape(-1, 3)
        self.R = np.array([np.asarray(b[1], float) for b in boxes]).reshape(-1, 3, 3)
        self.h = np.array([np.asarray(b[2], float) for b in boxes]).reshape(-1, 3)

    def mask(self, p) -> np.ndarray:
        p = np.atleast_2d(np.asarray(p, float))
        if len(self.c) == 0:
            return np.zeros(len(p), bool)
        loc = np.einsum("nbi,bij->nbj", p[:, None, :] - self.c[None, :, :], self.R)
        return np.any(np.all(np.abs(loc) <= self.h[None, :, :], axis=2), axis=1)


class ItemAero:
    """Wrench source: quadratic drag of item sample points relative to the air (ventilation + jets).

    shapes: list of ItemShape; flow: velocity field callable (default still air); jets: optional velocity field
    of the robot's exhaust (e.g. NozzleJetField); rho: air density; shelters: optional Shelters (points inside see
    still air: an item in a stowage pocket is not blown about by the cabin flow the field model knows nothing of).
    After each evaluation ``last_points``, ``last_air`` (with jets) and ``last_force`` {body: world force} are kept.
    """

    def __init__(self, model, shapes, flow=None, jets=None, rho: float = AIR_DENSITY, shelters=None):
        self.flow = flow if flow is not None else StillAir()
        self.jets = jets
        self.shelters = shelters
        self.rho = float(rho)
        b, pl, ca, kd, Rs = [], [], [], [], []
        for s in shapes:
            p, a, k = s.samples()
            b.append(np.full(len(p), s.body))
            pl.append(p)
            ca.append(a)
            kd.append(k)
            Rs.append(np.broadcast_to(s.R, (len(p), 3, 3)))
        self.b = np.concatenate(b) if b else np.zeros(0, int)
        self.pl = np.concatenate(pl) if pl else np.zeros((0, 3))
        self.ca = np.concatenate(ca) if ca else np.zeros((0, 3))
        self.kind = np.concatenate(kd) if kd else np.zeros(0, int)
        self.Rs = np.concatenate(Rs) if Rs else np.zeros((0, 3, 3))
        self.bodies = np.unique(self.b)
        self.enabled = True
        self.last_points = None
        self.last_air = None
        self.last_force = {}

    def points(self, data):
        """World positions of the sample points and the shape-to-world rotations."""
        R = data.xmat[self.b].reshape(-1, 3, 3)
        return data.xpos[self.b] + np.einsum("bij,bj->bi", R, self.pl), np.einsum("bij,bjk->bik", R, self.Rs)

    def accumulate(self, model, data, W) -> None:
        """Add the drag wrench of all item points into W (world axes at the body CoMs)."""
        self.last_force = {}
        if not self.enabled or len(self.b) == 0:
            return
        p, Rw = self.points(data)
        root = model.body_rootid[self.b]
        w = data.cvel[self.b, 0:3]
        v = data.cvel[self.b, 3:6] + np.cross(w, p - data.subtree_com[root])
        air = self.flow(p)
        if self.jets is not None:
            air = air + self.jets(p)
        if self.shelters is not None:
            air = np.where(self.shelters.mask(p)[:, None], 0.0, air)
        self.last_points, self.last_air = p, air
        vr = v - air
        vl = np.einsum("bji,bj->bi", Rw, vr)                         # shape-frame components
        Fl = np.zeros_like(vl)
        box = self.kind == 0
        if box.any():
            Fl[box] = -0.5 * self.rho * self.ca[box] * np.abs(vl[box]) * vl[box]
        cyl = self.kind == 1
        if cyl.any():
            vp = vl[cyl][:, 0:2]
            sp = np.linalg.norm(vp, axis=1, keepdims=True)
            Fl[cyl, 0:2] = -0.5 * self.rho * self.ca[cyl][:, 0:1] * sp * vp
            Fl[cyl, 2] = -0.5 * self.rho * self.ca[cyl][:, 2] * np.abs(vl[cyl][:, 2]) * vl[cyl][:, 2]
        sph = self.kind == 2
        if sph.any():
            sp = np.linalg.norm(vl[sph], axis=1, keepdims=True)
            Fl[sph] = -0.5 * self.rho * self.ca[sph][:, 0:1] * sp * vl[sph]
        F = np.einsum("bij,bj->bi", Rw, Fl)
        tau = np.cross(p - data.xipos[self.b], F)
        np.add.at(W[:, 0:3], self.b, F)
        np.add.at(W[:, 3:6], self.b, tau)
        for bb in self.bodies:
            self.last_force[int(bb)] = F[self.b == bb].sum(0)

    def wrench(self, model, data) -> np.ndarray:
        """New (nbody, 6) array."""
        W = np.zeros((model.nbody, 6))
        self.accumulate(model, data, W)
        return W


def jet_velocity(x, noz_pos, noz_dir, thrust, v0, rho: float = AIR_DENSITY, tmin: float = 1e-4) -> np.ndarray:
    """Superposed exhaust jets at points x (N, 3).  noz_pos / noz_dir (k, 3) world exit points and exhaust
    directions, thrust (k,) [N], v0 exit velocity [m/s] (scalar or (k,))."""
    x = np.atleast_2d(np.asarray(x, float))
    u = np.zeros_like(x)
    v0 = np.broadcast_to(np.asarray(v0, float), (len(thrust),))
    for p, e, T, ve in zip(noz_pos, noz_dir, thrust, v0):
        if T < tmin or ve <= 0:
            continue
        A = T / (rho * ve * ve)
        d0 = math.sqrt(4.0 * A / math.pi)
        r = x - p
        s = r @ e
        ok = s > 0
        if not ok.any():
            continue
        rad = np.linalg.norm(r - s[:, None] * e[None, :], axis=1)
        s_core = JET_B * d0
        sc = np.maximum(s, 1e-9)
        uc = np.where(s < s_core, ve, ve * s_core / sc)
        re = np.where(s < s_core, d0 / math.sqrt(2.0), JET_C * sc)
        mag = uc * np.exp(-(rad / re) ** 2) * ok
        u += mag[:, None] * e[None, :]
    return u


class NozzleJetField:
    """Velocity field of the free-flyer's 12 nozzle jets, read from a live plant.

    body: robot body id; nozzle_pos / nozzle_dir (12, 3) body-frame exit points and exhaust directions;
    thrust_fn() -> (12,) current thrusts [N]; dp_fn() -> (2,) plenum pressures [Pa] of the two modules (nozzles
    0-5 module 0, 6-11 module 1); cd_exit (0.9); data: the MjData whose body
    pose is used (set by ``bind``).  Callable on points (N, 3) -> (N, 3).
    """

    def __init__(self, body: int, nozzle_pos, nozzle_dir, thrust_fn, dp_fn, cd_exit: float = 0.9,
                 rho: float = AIR_DENSITY, max_range: float = 4.0):
        self.body = int(body)
        self.pos = np.asarray(nozzle_pos, float)
        self.dir = np.asarray(nozzle_dir, float) / np.linalg.norm(nozzle_dir, axis=1, keepdims=True)
        self.thrust_fn, self.dp_fn = thrust_fn, dp_fn
        self.cd_exit, self.rho, self.max_range = float(cd_exit), float(rho), float(max_range)
        self.data = None
        self.enabled = True

    def bind(self, data) -> "NozzleJetField":
        """Use this MjData for the robot pose."""
        self.data = data
        return self

    def exit_velocity(self) -> np.ndarray:
        """(12,) exit velocities [m/s] from the plenum pressures."""
        dp = np.asarray(self.dp_fn(), float)
        v = self.cd_exit * np.sqrt(2.0 * np.maximum(dp, 0.0) / self.rho)
        return np.repeat(v, 6)

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        if not self.enabled or self.data is None:
            return np.zeros_like(x)
        d = self.data
        R = d.xmat[self.body].reshape(3, 3)
        P = d.xpos[self.body] + self.pos @ R.T
        D = self.dir @ R.T
        if self.max_range is not None and len(x):
            far = np.min(np.linalg.norm(x[:, None, :] - P[None, :, :], axis=2), axis=1) > self.max_range
            if far.all():
                return np.zeros_like(x)
        return jet_velocity(x, P, D, np.asarray(self.thrust_fn(), float), self.exit_velocity(), self.rho)


def settle_time_closed_form(mass: float, cd_area: float, u: float, rho: float = AIR_DENSITY) -> float:
    """Time for a body at rest to reach half the air speed u under quadratic drag (m dv/dt = k (u - v)^2,
    k = 1/2 rho cd A): t = m / (k u)."""
    k = 0.5 * rho * cd_area
    return mass / (k * u)
