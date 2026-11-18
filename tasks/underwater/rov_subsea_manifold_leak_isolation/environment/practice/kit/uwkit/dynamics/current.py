"""Water-current fields.  Interface: field(pos_w, t) -> velocity, pos_w shape (n, 3) [m], t [s],
velocity shape (n, 3) [m/s] in the world frame W (z up, surface z = 0, depth = -z).

Building blocks (combine with `+` or wrap):
  Uniform(v)                         constant everywhere
  Layered(depths, velocities)        piecewise-linear profile in depth
  Tidal(...)                         rectilinear or rotary tidal ellipse, slack and turning of the tide
  BoundaryLayer(field, seabed_depth) power-law reduction near the seabed, zero below it
  GaussMarkov(sigma, tau_c, rng)     first-order Gauss-Markov turbulence (spatially uniform)
  Wake(field, cylinders)             potential flow + wake deficit + vortex-shedding sway behind
                                     vertical cylinders (piles, risers, legs)
  Scaled(field, factor_fn)           any multiplicative envelope

All randomness comes from an explicit numpy Generator; nothing reads the wall clock.  The
Gauss-Markov process is generated sequentially on a fixed time grid, so the samples do not
depend on the order or the times at which the field is queried.
"""
from __future__ import annotations

import math

import numpy as np


def _as_pos(pos):
    p = np.asarray(pos, float)
    return p.reshape(1, 3) if p.ndim == 1 else p


class CurrentField:
    needs_rng = False

    def velocity(self, pos, t):
        raise NotImplementedError

    def __call__(self, pos, t):
        return self.velocity(_as_pos(pos), float(t))

    def __add__(self, other):
        return Sum([self, other])

    def bind_rng(self, rng):
        """Hook for fields that need randomness but were created without a Generator."""
        return self

    def children(self):
        return []


class Zero(CurrentField):
    def velocity(self, pos, t):
        return np.zeros((_as_pos(pos).shape[0], 3))


class Uniform(CurrentField):
    def __init__(self, v):
        self.v = np.asarray(v, float).reshape(3)

    def velocity(self, pos, t):
        out = np.empty((1 if np.ndim(pos) == 1 else len(pos), 3))
        out[:] = self.v
        return out


class Sum(CurrentField):
    def __init__(self, fields):
        self.fields = []
        for f in fields:
            self.fields.extend(f.fields if isinstance(f, Sum) else [f])

    def velocity(self, pos, t):
        out = self.fields[0].velocity(pos, t)
        for f in self.fields[1:]:
            out = out + f.velocity(pos, t)
        return out

    def bind_rng(self, rng):
        for f in self.fields:
            f.bind_rng(rng)
        return self

    def children(self):
        return list(self.fields)

    @property
    def needs_rng(self):
        return any(f.needs_rng for f in self.fields)


class Layered(CurrentField):
    """Velocity profile v(depth): linear interpolation, constant beyond the table ends."""

    def __init__(self, depths, velocities):
        self.d = np.asarray(depths, float)
        self.v = np.asarray(velocities, float).reshape(len(self.d), 3)
        if np.any(np.diff(self.d) <= 0):
            raise ValueError("depths must be strictly increasing")

    def velocity(self, pos, t):
        depth = -_as_pos(pos)[:, 2]
        return np.stack([np.interp(depth, self.d, self.v[:, k]) for k in range(3)], axis=1)


class Tidal(CurrentField):
    """Tidal ellipse.  Major-axis speed u(t) = amp * cos(2 pi (t - t_peak) / period); minor-axis
    speed = minor_ratio * amp * sin(...) (0 = rectilinear: slack, then the flow turns 180 deg).
    direction = flood direction (unit vector in the horizontal plane of W); `mean` adds a residual
    current; `profile` is an optional (depths, factors) table scaling the tide with depth."""

    def __init__(self, amp, direction=(1.0, 0.0), period_s=44712.0, t_peak_s=0.0, minor_ratio=0.0,
                 mean=(0.0, 0.0, 0.0), profile=None):
        self.amp = float(amp)
        e = np.asarray(direction, float).reshape(2)
        self.e = e / np.linalg.norm(e)
        self.n = np.array([-self.e[1], self.e[0]])
        self.period = float(period_s)
        self.t_peak = float(t_peak_s)
        self.minor = float(minor_ratio)
        self.mean = np.asarray(mean, float).reshape(3)
        self.profile = None if profile is None else (np.asarray(profile[0], float), np.asarray(profile[1], float))

    def horizontal(self, t):
        ph = 2 * math.pi * (t - self.t_peak) / self.period
        return self.amp * (math.cos(ph) * self.e + self.minor * math.sin(ph) * self.n)

    def velocity(self, pos, t):
        p = _as_pos(pos)
        h = self.horizontal(t)
        v = np.zeros((p.shape[0], 3))
        v[:, 0], v[:, 1] = h[0], h[1]
        if self.profile is not None:
            v *= np.interp(-p[:, 2], self.profile[0], self.profile[1])[:, None]
        return v + self.mean


class BoundaryLayer(CurrentField):
    """Power-law boundary layer over the seabed: the wrapped field is multiplied by

        f(h) = (min(h, delta) / delta)^exponent / (min(z_ref, delta) / delta)^exponent

    for heights h above the seabed (zero inside / below it; seabed_depth may be a callable (x, y) -> depth).
    delta = boundary-layer thickness (constant above it).  z_ref = height at which the wrapped field's amplitude is the
    given one (f(z_ref) = 1); None = delta (0.2.0 behaviour: the amplitude holds above delta, the layer only slows the
    flow below it).  Example: a current "given at 2 m above the bottom" with a 20 m layer is
    BoundaryLayer(field, seabed, delta=20.0, z_ref=2.0)."""

    def __init__(self, field: CurrentField, seabed_depth, delta=2.0, exponent=1.0 / 7.0, z_ref=None):
        self.field, self.seabed, self.delta, self.k = field, seabed_depth, float(delta), float(exponent)
        self.z_ref = None if z_ref is None else float(z_ref)
        if self.z_ref is not None and self.z_ref <= 0.0:
            raise ValueError("z_ref must be > 0")
        self._norm = 1.0 if self.z_ref is None else (min(self.z_ref, self.delta) / self.delta) ** self.k

    def velocity(self, pos, t):
        p = _as_pos(pos)
        sb = self.seabed(p[:, 0], p[:, 1]) if callable(self.seabed) else self.seabed
        h = np.clip(np.asarray(sb, float) + p[:, 2], 0.0, None)     # height above seabed
        f = np.where(h < self.delta, (h / self.delta) ** self.k, 1.0)
        if self.z_ref is not None:
            f = f / self._norm
        return self.field.velocity(p, t) * f[:, None]

    def bind_rng(self, rng):
        self.field.bind_rng(rng)
        return self

    def children(self):
        return [self.field]

    @property
    def needs_rng(self):
        return self.field.needs_rng


class Scaled(CurrentField):
    def __init__(self, field: CurrentField, factor_fn):
        self.field, self.fn = field, factor_fn

    def velocity(self, pos, t):
        p = _as_pos(pos)
        return self.field.velocity(p, t) * np.asarray(self.fn(p, t), float).reshape(-1, 1)

    def bind_rng(self, rng):
        self.field.bind_rng(rng)
        return self

    def children(self):
        return [self.field]

    @property
    def needs_rng(self):
        return self.field.needs_rng


class GaussMarkov(CurrentField):
    """Spatially uniform first-order Gauss-Markov turbulence, per world axis:
    exact AR(1) on a grid of dt_sample:  x_{k+1} = phi x_k + sigma sqrt(1 - phi^2) xi,
    phi = exp(-dt_sample / tau_c); stationary std = sigma; linear interpolation in between
    (continuous velocity, bounded derivative, so the unsteady-current force stays finite)."""

    def __init__(self, sigma=(0.05, 0.05, 0.01), tau_c=(20.0, 20.0, 10.0), rng=None,
                 dt_sample=0.1, block=2048):
        self.sigma = np.asarray(sigma, float).reshape(3)
        self.tau = np.asarray(tau_c, float).reshape(3)
        self.dt = float(dt_sample)
        self.block = int(block)
        self.phi = np.exp(-self.dt / self.tau)
        self.rng = rng
        self._x = None

    @property
    def needs_rng(self):
        return self.rng is None

    def bind_rng(self, rng):
        if self.rng is None:
            self.rng = rng
        return self

    def _extend(self, k_needed):
        if self.rng is None:
            raise RuntimeError("GaussMarkov needs a numpy Generator (pass rng= or let UnderwaterSim bind one)")
        if self._x is None:
            self._x = (self.sigma * self.rng.standard_normal(3))[None, :]
        while self._x.shape[0] <= k_needed:
            xi = self.rng.standard_normal((self.block, 3))
            out = np.empty((self.block, 3))
            x = self._x[-1]
            s = self.sigma * np.sqrt(1.0 - self.phi ** 2)
            for k in range(self.block):
                x = self.phi * x + s * xi[k]
                out[k] = x
            self._x = np.vstack([self._x, out])

    def sample(self, t):
        if t < 0:
            t = 0.0
        k = int(t // self.dt)
        if self._x is None or self._x.shape[0] <= k + 1:
            self._extend(k + 1)
        a = t / self.dt - k
        return (1 - a) * self._x[k] + a * self._x[k + 1]

    def velocity(self, pos, t):
        n = 1 if np.ndim(pos) == 1 else len(pos)
        out = np.empty((n, 3))
        out[:] = self.sample(t)
        return out


class Wake(CurrentField):
    """Ambient field modified around vertical circular cylinders.

    cylinders: list of dicts {x, y, diameter, z_top (default 0), z_bottom (default -1e9)}.
    In the cylinder's flow-aligned frame (s downstream, y' across, R = D/2, U = |ambient_xy|):
      potential flow outside the cylinder: u = U (1 - R^2/r^2 cos 2theta), v = -U R^2/r^2 sin 2theta
      wake (s > 0):  deficit  U * c_def * sqrt(D / (s + D)) * exp(-(y'/b)^2),  b = 0.5 D sqrt(1 + 4 s / D)
      shedding sway: U * a_shed * sqrt(D / (s + D)) * exp(-(y'/b)^2) * sin(2 pi f (t - s / (0.85 U)) + phase),
                     f = St U / D
      inside the cylinder: velocity = 0.
    The wake onset is smoothed over 0.5 D downstream and over 0.5 D at the vertical ends.
    """

    def __init__(self, field: CurrentField, cylinders, c_def=0.8, a_shed=0.3, strouhal=0.2):
        self.field = field
        self.cyl = [dict(x=float(c["x"]), y=float(c["y"]), D=float(c["diameter"]),
                         z_top=float(c.get("z_top", 0.0)), z_bottom=float(c.get("z_bottom", -1e9)),
                         phase=float(c.get("phase", 0.0))) for c in cylinders]
        self.c_def, self.a_shed, self.St = float(c_def), float(a_shed), float(strouhal)

    def bind_rng(self, rng):
        self.field.bind_rng(rng)
        return self

    def children(self):
        return [self.field]

    @property
    def needs_rng(self):
        return self.field.needs_rng

    @staticmethod
    def _smooth(x):
        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        return x * x * (3 - 2 * x)

    def velocity(self, pos, t):
        """Scalar loop over the query points (the simulator asks for a handful of points per step;
        a Python loop is ~10x cheaper than ~50 small numpy operations)."""
        p = _as_pos(pos)
        amb = self.field.velocity(p, t)
        out = amb.copy()
        sm = self._smooth
        P = p.tolist()
        A = amb.tolist()
        two_pi_t = 2 * math.pi * t
        for c in self.cyl:
            cx, cy, D, zt, zb, ph = c["x"], c["y"], c["D"], c["z_top"], c["z_bottom"], c["phase"]
            R2 = 0.25 * D * D
            for k in range(len(P)):
                ux, uy = A[k][0], A[k][1]
                U = math.sqrt(ux * ux + uy * uy)
                if U <= 1e-6:
                    continue
                ex, ey = ux / U, uy / U
                x, y, z = P[k]
                dx, dy = x - cx, y - cy
                s_ = dx * ex + dy * ey
                yp = dy * ex - dx * ey
                r2 = s_ * s_ + yp * yp
                zf = sm((zt - z) / (0.5 * D) + 1.0) * sm((z - zb) / (0.5 * D) + 1.0)
                if r2 < R2 and zf > 0.5:
                    out[k] = 0.0
                    continue
                if zf == 0.0:
                    continue
                r2s = r2 if r2 > R2 else R2
                kk = U * R2 / (r2s * r2s)
                du = -kk * (s_ * s_ - yp * yp)
                dv = -kk * (2.0 * s_ * yp)
                sp = s_ if s_ > 0.0 else 0.0
                if s_ > 0.0:
                    g = math.sqrt(D / (sp + D)) * sm(s_ / (0.5 * D)) * \
                        math.exp(-(yp * yp) / (R2 * (1.0 + 4.0 * sp / D)))
                    du -= U * self.c_def * g
                    f = self.St * U / D
                    dv += U * self.a_shed * g * math.sin(f * (two_pi_t - 2 * math.pi * sp / (0.85 * U)) + ph)
                du *= zf
                dv *= zf
                out[k, 0] += du * ex - dv * ey
                out[k, 1] += du * ey + dv * ex
        return out


def make_current(spec, rng=None) -> CurrentField:
    """Build a field from a small JSON-able dict, e.g.
    {"type": "sum", "fields": [{"type": "uniform", "v": [0.3, 0, 0]},
                               {"type": "gauss_markov", "sigma": [0.05, 0.05, 0.01], "tau_c": [20, 20, 10]}]}"""
    if spec is None:
        return Zero()
    if isinstance(spec, CurrentField):
        return spec
    t = spec["type"]
    if t == "zero":
        return Zero()
    if t == "uniform":
        return Uniform(spec["v"])
    if t == "layered":
        return Layered(spec["depths"], spec["velocities"])
    if t == "tidal":
        kw = {k: v for k, v in spec.items() if k != "type"}
        return Tidal(**kw)
    if t == "gauss_markov":
        kw = {k: v for k, v in spec.items() if k != "type"}
        return GaussMarkov(rng=rng, **kw)
    if t == "boundary_layer":
        return BoundaryLayer(make_current(spec["field"], rng), spec["seabed_depth"],
                             spec.get("delta", 2.0), spec.get("exponent", 1.0 / 7.0), spec.get("z_ref"))
    if t == "wake":
        kw = {k: v for k, v in spec.items() if k not in ("type", "field", "cylinders")}
        return Wake(make_current(spec["field"], rng), spec["cylinders"], **kw)
    if t == "sum":
        return Sum([make_current(f, rng) for f in spec["fields"]])
    raise ValueError(f"unknown current type {t}")
