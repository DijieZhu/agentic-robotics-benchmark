"""World frames for work inside an orbiting station module, as per-body external wrenches.

``RotatingStationFrame`` (use this for intra-vehicular work): MuJoCo ``world`` is fixed to the station
module, which holds a constant attitude in the local-vertical/local-horizontal (LVLH) frame of a
circular orbit and therefore rotates at the orbit rate ``n`` about the orbit normal.  Static station
geometry lives in ``worldbody``.  Per simulated body (force at the body CoM, world axes):

    F   = m [ tidal(rho) - 2 Om x v - Om x (Om x rho) ]                 (tidal, Coriolis, centrifugal)
    tau = tau_gg - Om x J Om - Om x J w - w x J Om + J (w x Om)          (gravity gradient, rotational)

with ``rho`` the CoM position relative to the orbiting point (the station centre of mass), ``v`` and
``w`` the CoM velocity and angular velocity relative to the station, ``J`` the world-axes inertia about
the CoM, ``Om`` the frame rate in world axes.  The rotational fictitious torque is required for the
attitude of free objects to be right (without it, 34-149 deg error after 20 min).

``InertialChiefFrame`` (cross-check / extra-vehicular work): world = chief-centred axes fixed in
inertial space (the chief's LVLH axes at t = 0); only the exact (Encke) differential gravity and the
gravity-gradient torque act.  Both formulations agree to 1e-10 m over 1200 s.

Every frame object is a wrench source: ``accumulate(model, data, W)`` adds its (nbody, 6) wrench to W.
"""
from __future__ import annotations

import numpy as np

from ._fast import cross_rows

__all__ = [
    "MU_EARTH", "R_EARTH", "J2_EARTH", "DEFAULT_ORBIT_RATE", "mean_motion", "radius_for_rate",
    "encke_rel_accel", "j2_accel", "cw_propagate", "CircularChief", "body_com_velocity", "world_inertia",
    "free_bodies", "RotatingStationFrame", "InertialChiefFrame", "rotating_to_inertial",
]

MU_EARTH = 3.986004418e14   # m^3 s^-2
R_EARTH = 6378137.0         # m
J2_EARTH = 1.08262668e-3
DEFAULT_ORBIT_RATE = float(np.sqrt(MU_EARTH / (R_EARTH + 400e3) ** 3))  # 1.1314e-3 rad/s (400 km)


def mean_motion(a: float, mu: float = MU_EARTH) -> float:
    """Orbit rate n = sqrt(mu / a^3) [rad/s] of a circular orbit of radius a [m]."""
    return float(np.sqrt(mu / a ** 3))


def radius_for_rate(n: float, mu: float = MU_EARTH) -> float:
    """Circular-orbit radius [m] with orbit rate n [rad/s]."""
    return float((mu / n ** 2) ** (1.0 / 3.0))


def encke_rel_accel(rho, R, mu: float = MU_EARTH) -> np.ndarray:
    """Exact g(R + rho) - g(R) for point-mass gravity, without cancellation (Battin's f(q)).

    rho: (..., 3) offsets from the chief [m]; R: (3,) chief position from the attracting centre [m].
    """
    rho = np.asarray(rho, float)
    R = np.asarray(R, float)
    rc2 = float(R @ R)
    q = (2.0 * (rho @ R) + np.einsum("...i,...i->...", rho, rho)) / rc2
    s = (1.0 + q) ** 1.5
    f = q * (3.0 + 3.0 * q + q * q) / ((1.0 + s) * s)
    return -mu / rc2 ** 1.5 * (rho - f[..., None] * (R + rho))


def j2_accel(r, mu: float = MU_EARTH, j2: float = J2_EARTH, re: float = R_EARTH) -> np.ndarray:
    """J2 acceleration in an Earth-centred frame whose z axis is the spin axis."""
    r = np.asarray(r, float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    rn = np.linalg.norm(r, axis=-1)
    k = 1.5 * j2 * mu * re ** 2 / rn ** 5
    zr2 = (z / rn) ** 2
    return np.stack([k * x * (5 * zr2 - 1), k * y * (5 * zr2 - 1), k * z * (5 * zr2 - 3)], axis=-1)


def cw_propagate(x0, v0, n: float, t: float):
    """Clohessy-Wiltshire closed form in LVLH (x radial out, y along-track, z orbit normal).

    Returns (position, velocity) at time t for initial position x0 and velocity v0 (LVLH-relative).
    """
    x0 = np.asarray(x0, float)
    v0 = np.asarray(v0, float)
    nt = n * t
    c, s = np.cos(nt), np.sin(nt)
    x = (4 - 3 * c) * x0[0] + s / n * v0[0] + 2 / n * (1 - c) * v0[1]
    y = 6 * (s - nt) * x0[0] + x0[1] - 2 / n * (1 - c) * v0[0] + (4 * s - 3 * nt) / n * v0[1]
    z = x0[2] * c + v0[2] / n * s
    vx = 3 * n * s * x0[0] + c * v0[0] + 2 * s * v0[1]
    vy = 6 * n * (c - 1) * x0[0] - 2 * s * v0[0] + (4 * c - 3) * v0[1]
    vz = -x0[2] * n * s + v0[2] * c
    return np.array([x, y, z]), np.array([vx, vy, vz])


class CircularChief:
    """Analytic circular reference orbit expressed in the inertial world axes (= LVLH axes at t = 0)."""

    def __init__(self, orbit_rate: float = DEFAULT_ORBIT_RATE, mu: float = MU_EARTH):
        self.n = float(orbit_rate)
        self.mu = mu
        self.a = radius_for_rate(self.n, mu)

    def state(self, t: float):
        """(R, V): chief position and velocity relative to the attracting centre, world axes."""
        c, s = np.cos(self.n * t), np.sin(self.n * t)
        return self.a * np.array([c, s, 0.0]), self.a * self.n * np.array([-s, c, 0.0])

    def lvlh_basis(self, t: float) -> np.ndarray:
        """Rows: LVLH unit axes at time t expressed in world axes (C_lvlh_from_world)."""
        c, s = np.cos(self.n * t), np.sin(self.n * t)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def body_com_velocity(model, data) -> np.ndarray:
    """(nbody, 3) linear velocity of every body's centre of mass, world axes.

    MuJoCo's ``cvel`` is a spatial velocity referenced to the subtree CoM of the body's root; shift it to
    the body CoM (exact for articulated bodies; using cvel[:, 3:] directly is exact only for roots).
    Valid after mj_step1 / mj_forward (needs xipos, subtree_com, cvel).
    """
    r = data.xipos - data.subtree_com[model.body_rootid]
    return data.cvel[:, 3:6] + cross_rows(data.cvel[:, 0:3], r)


def world_inertia(model, data, bodies=None) -> np.ndarray:
    """(nbody or len(bodies), 3, 3) inertia tensors about each body CoM in world axes."""
    idx = slice(None) if bodies is None else np.asarray(bodies, int)
    R = data.ximat[idx].reshape(-1, 3, 3)
    return np.einsum("bij,bj,bkj->bik", R, model.body_inertia[idx], R)


def free_bodies(model) -> np.ndarray:
    """Ids of all bodies that carry mass and move (not welded to the world, not mocap)."""
    out = [b for b in range(1, model.nbody)
           if model.body_mass[b] > 0 and model.body_weldid[b] != 0 and model.body_mocapid[b] < 0]
    return np.asarray(out, int)


def _gg_torque(J, rhat, r_mag, mu):
    """Gravity-gradient torque 3 mu / r^3 * rhat x (J rhat) per body (world axes)."""
    Jr = np.einsum("bij,bj->bi", J, rhat)
    return 3.0 * mu / r_mag[:, None] ** 3 * cross_rows(rhat, Jr)


class RotatingStationFrame:
    """Fictitious and tidal loads for a world frame fixed to a station module (see module docstring).

    orbit_rate: n [rad/s] (default 400 km circular orbit).  C_lvlh_from_world: rows = LVLH axes in world
    (station) axes, i.e. the module's constant attitude in LVLH.  origin: world position of the orbiting
    point (station centre of mass).  bodies: ids that receive loads (default ``free_bodies``).
    enabled=False turns all loads off; the per-term switches allow ablations.  linear_tidal uses the
    linearised gravity difference n^2 (2x, -y, -z) (LVLH) instead of the exact Encke difference; together
    with the centrifugal term n^2 (x, y, 0) this is the Hill/CW field n^2 (3x, 0, -z).
    """

    def __init__(self, model, orbit_rate: float = DEFAULT_ORBIT_RATE, C_lvlh_from_world=np.eye(3),
                 origin=(0.0, 0.0, 0.0), bodies=None, enabled: bool = True, tidal: bool = True,
                 coriolis: bool = True, centrifugal: bool = True, rotational: bool = True,
                 gravity_gradient: bool = True, linear_tidal: bool = False, mu: float = MU_EARTH):
        self.n = float(orbit_rate)
        self.mu = mu
        self.a = radius_for_rate(self.n, mu)
        self.C = np.asarray(C_lvlh_from_world, float)
        self.Om = self.C.T @ np.array([0.0, 0.0, self.n])           # frame rate, world axes
        self.R_world = self.C.T @ np.array([self.a, 0.0, 0.0])        # orbiting point from Earth centre
        self.origin = np.asarray(origin, float)
        self.bodies = free_bodies(model) if bodies is None else np.asarray(bodies, int)
        self.enabled = enabled
        self.tidal, self.coriolis, self.centrifugal = tidal, coriolis, centrifugal
        self.rotational, self.gravity_gradient, self.linear_tidal = rotational, gravity_gradient, linear_tidal

    def accumulate(self, model, data, W) -> None:
        """Add this frame's wrench (world axes, at body CoMs) into W (nbody, 6)."""
        if not self.enabled or len(self.bodies) == 0:
            return
        b = self.bodies
        rho = data.xipos[b] - self.origin
        m = model.body_mass[b]
        Om = self.Om
        a = np.zeros((len(b), 3))
        if self.tidal:
            if self.linear_tidal:   # linearised gravity difference n^2 (2x, -y, -z) in LVLH (centrifugal separate)
                rl = rho @ self.C.T
                a += (self.n ** 2 * np.stack([2 * rl[:, 0], -rl[:, 1], -rl[:, 2]], axis=1)) @ self.C
            else:
                a += encke_rel_accel(rho, self.R_world, self.mu)
        if self.coriolis:
            v = body_com_velocity(model, data)[b]
            a -= 2.0 * cross_rows(Om, v)
        if self.centrifugal:
            a -= cross_rows(Om, cross_rows(Om, rho))
        W[b, 0:3] += m[:, None] * a
        if self.gravity_gradient or self.rotational:
            J = world_inertia(model, data, b)
            if self.gravity_gradient:
                r = self.R_world[None, :] + rho
                r_mag = np.linalg.norm(r, axis=1)
                W[b, 3:6] += _gg_torque(J, r / r_mag[:, None], r_mag, self.mu)
            if self.rotational:
                w = data.cvel[b, 0:3]
                JOm = J @ Om
                Jw = np.einsum("bij,bj->bi", J, w)
                W[b, 3:6] += (-cross_rows(Om, JOm) - cross_rows(Om, Jw) - cross_rows(w, JOm)
                              + np.einsum("bij,bj->bi", J, cross_rows(w, Om)))

    def wrench(self, model, data) -> np.ndarray:
        """Return this frame's (nbody, 6) wrench as a new array."""
        W = np.zeros((model.nbody, 6))
        self.accumulate(model, data, W)
        return W


class InertialChiefFrame:
    """World = chief-centred, inertially fixed axes (the chief's LVLH axes at t = 0).

    Loads: exact differential gravity m [g(R + rho) - g(R)] and the gravity-gradient torque, with the
    chief state taken at ``data.time`` (inside a passive callback this is the RK4 stage time).
    ``extra_accel(t, rho)`` may add further accelerations (N, 3).
    """

    def __init__(self, model, chief: CircularChief | None = None, bodies=None, gravity_gradient: bool = True,
                 extra_accel=None, enabled: bool = True):
        self.chief = chief if chief is not None else CircularChief()
        self.bodies = free_bodies(model) if bodies is None else np.asarray(bodies, int)
        self.gravity_gradient = gravity_gradient
        self.extra_accel = extra_accel
        self.enabled = enabled

    def accumulate(self, model, data, W) -> None:
        """Add the wrench at time data.time into W (nbody, 6)."""
        if not self.enabled or len(self.bodies) == 0:
            return
        R, _ = self.chief.state(data.time)
        b = self.bodies
        rho = data.xipos[b]
        a = encke_rel_accel(rho, R, self.chief.mu)
        if self.extra_accel is not None:
            a = a + self.extra_accel(data.time, rho)
        W[b, 0:3] += model.body_mass[b][:, None] * a
        if self.gravity_gradient:
            r = R[None, :] + rho
            r_mag = np.linalg.norm(r, axis=1)
            W[b, 3:6] += _gg_torque(world_inertia(model, data, b), r / r_mag[:, None], r_mag, self.chief.mu)

    def wrench(self, model, data) -> np.ndarray:
        """Return the (nbody, 6) wrench as a new array."""
        W = np.zeros((model.nbody, 6))
        self.accumulate(model, data, W)
        return W


def rotating_to_inertial(p, v, w, orbit_rate: float = DEFAULT_ORBIT_RATE, C_lvlh_from_world=np.eye(3)):
    """Convert an initial state relative to the rotating station frame (at t = 0, axes coincide) to the
    inertial chief-centred frame: returns (p, v + Om x p, w + Om)."""
    Om = np.asarray(C_lvlh_from_world, float).T @ np.array([0.0, 0.0, orbit_rate])
    p = np.asarray(p, float)
    return p.copy(), np.asarray(v, float) + np.cross(Om, p), np.asarray(w, float) + Om
