"""Imaging sonars: a mechanically scanned single-beam imaging sonar (the reference vehicle's main sonar) and an
optional forward-looking multibeam imaging sonar.  CPU only: mj_multiRay over the beam's solid angle.

Per ray k (direction inside the beam, solid angle dOmega_k), first hit at true range r with incidence i on geom g:

    I_k = B_tx(az, el) B_rx(az, el) * sigma_g(i) * dOmega_k / max(cos i, 0.05)
          * 10^(-2 (alpha_true - alpha_cfg) r / 10)                     (residual after the device's TVG)
    sigma_g(i) = mu_g cos^2 i + s_g exp(-i^2 / 2 w_g^2)                  (materials.MaterialMap)

The device TVG is taken as 20 log r + 2 alpha_cfg r (area-scattering TVG): sea-bed returns are range-independent and
small targets fall as 1/r^2.  Each ray's energy is spread over the range extent of its footprint,
dr = r * dtheta * tan(i), so oblique sea bed renders as a continuous return, not as stripes.  Shadows are the natural
first-hit occlusion.  Ranges are converted with the *configured* speed of sound (r_meas = r_true * c_cfg / c_true).
Then: range-resolution blur -> fully developed speckle (exponential intensity) -> additive noise that grows with the
TVG -> transducer ring-down near zero range -> log compression to 8 bit with a gain setting.

Mechanical scan: one ping per step (default 1 gradian = 0.9 deg, 400 steps per turn), each ping uses the vehicle
pose at the moment it is generated (motion smear appears naturally).  Ping period = listening window 2 R / c_cfg plus a
fixed overhead fitted to the vendor's two published scan times (1 m: 4.17 s and 50 m: 33 s per 360 deg).

Angle convention (mechanical sonar): site `sonar_head`, rotation axis = site +z; angle 0 = site +x; angle increases
CLOCKWISE when viewed from +z (gradians 0..399, like the reference device's protocol).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import mujoco

from ._util import RayCaster, RateScheduler, site_id, site_pose, SpecItem
from .materials import MaterialMap
from .acoustics import WaterColumn, SEAWATER_TEMPERATE, absorption_db_per_m

GRAD = 2.0 * math.pi / 400.0


# ===================================================================================================== shared core

@dataclass
class SonarHidden:
    sidelobe_db: tuple = (-13.3, -17.8)   # one-way side-lobe levels of a uniform line aperture
    noise_db: float = -92.0               # additive noise floor (dB re the intensity scale) at r_ref
    noise_ref_m: float = 10.0
    ring_db: float = -30.0                # transducer ring-down level at r = 0
    ring_len_m: float = 0.05
    floor_db: float = -88.0               # 8-bit mapping: value 0 at floor_db (normal gain)
    dyn_db: float = 60.0                  #               value 255 at floor_db + dyn_db
    gain_step_db: float = 6.0             # each gain index step shifts the floor
    speckle_looks: float = 1.0
    alpha_cfg_db_m: Optional[float] = None  # device's assumed absorption; None -> nominal for 10 degC / 35 PSU


def _beam_dirs(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """Directions in the sonar frame (x fwd, y left, z up) for azimuth (CCW about +z, rad) and elevation (rad)."""
    ce = np.cos(el)
    return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], -1)


def footprint_caps(m: mujoco.MjModel) -> np.ndarray:
    """Upper bound of a ray footprint's range extent per geom: flat-faced geoms (planes, height fields, boxes) keep
    the tangent-plane extrapolation; curved primitives are capped by their radius (the extrapolation is wrong near
    their silhouette); meshes by their bounding radius."""
    T = mujoco.mjtGeom
    cap = np.full(m.ngeom, np.inf)
    for g in range(m.ngeom):
        t = int(m.geom_type[g])
        sz = m.geom_size[g]
        if t in (int(T.mjGEOM_SPHERE), int(T.mjGEOM_CAPSULE), int(T.mjGEOM_CYLINDER)):
            cap[g] = float(sz[0])
        elif t == int(T.mjGEOM_ELLIPSOID):
            cap[g] = float(np.max(sz[:3]))
        elif t == int(T.mjGEOM_BOX):
            cap[g] = 2.0 * float(np.max(sz[:3]))
        elif t == int(T.mjGEOM_MESH):
            cap[g] = float(m.geom_rbound[g]) if m.geom_rbound[g] > 0 else np.inf
    return cap


def _beam_tangents(az: np.ndarray, el: np.ndarray):
    """Unit tangents of _beam_dirs along elevation and azimuth."""
    se, ce = np.sin(el), np.cos(el)
    sa, ca = np.sin(az), np.cos(az)
    t_el = np.stack([-se * ca, -se * sa, ce], -1)
    t_az = np.stack([-sa, ca, np.zeros_like(az)], -1)
    return t_el, t_az


def _deposit(prof: np.ndarray, r_meas: np.ndarray, dr: np.ndarray, val: np.ndarray, rmax: float) -> None:
    """Add each ray's energy uniformly over [r - dr/2, r + dr/2] into the range profile (n,), via a difference
    array (O(rays + samples))."""
    n = prof.shape[0]
    scale = n / rmax
    a = (r_meas - 0.5 * dr) * scale
    b = (r_meas + 0.5 * dr) * scale
    ok = (b > 0) & (a < n) & (val > 0)
    a, b, val = a[ok], b[ok], val[ok]
    a = np.clip(a, 0, n)
    b = np.clip(b, 0, n)
    width = np.maximum(b - a, 1e-6)
    dens = val / width
    diff = np.zeros(n + 2)
    ia = np.floor(a).astype(np.int64)
    ib = np.floor(b).astype(np.int64)
    # fractional coverage of the edge samples
    same = ia == ib
    np.add.at(diff, ia[same], val[same])
    np.add.at(diff, ia[same] + 1, -val[same])
    o = ~same
    fa = (ia[o] + 1 - a[o]) * dens[o]
    fb = (b[o] - ib[o]) * dens[o]
    np.add.at(diff, ia[o], fa)
    np.add.at(diff, ia[o] + 1, -fa)
    np.add.at(diff, ia[o] + 1, dens[o])          # full samples ia+1 .. ib-1
    np.add.at(diff, ib[o], -dens[o])
    np.add.at(diff, ib[o], fb)
    np.add.at(diff, ib[o] + 1, -fb)
    prof += np.cumsum(diff)[:n]


def _gauss_kernel(sigma: float) -> np.ndarray:
    if sigma < 0.3:
        return np.ones(1)
    r = int(math.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


class _SonarCore:
    def __init__(self, m: mujoco.MjModel, site: str, materials: MaterialMap, water: WaterColumn,
                 rng: np.random.Generator, hidden: SonarHidden, frequency_hz: float,
                 geomgroup: Optional[Sequence[int]], exclude_body: Optional[int]):
        self.m = m
        self.sid = site_id(m, site)
        self.mat = materials
        self.water = water
        self.rng = rng
        self.hid = hidden
        self.f = float(frequency_hz)
        root = int(m.body_rootid[m.site_bodyid[self.sid]]) if exclude_body is None else int(exclude_body)
        self.caster = RayCaster(m, geomgroup=geomgroup, bodyexclude=root)
        self.dr_cap = footprint_caps(m)

    def alpha_true(self, depth: float) -> float:
        return self.water.absorption_db_per_m(self.f, depth)

    def alpha_cfg(self) -> float:
        a = self.hid.alpha_cfg_db_m
        return absorption_db_per_m(self.f, 10.0, 35.0, 0.0) if a is None else a

    def returns(self, d: mujoco.MjData, origin: np.ndarray, dirs_w: np.ndarray, gain2: np.ndarray,
                domega: np.ndarray, t_el_w: np.ndarray, t_az_w: np.ndarray, d_el: np.ndarray, d_az: np.ndarray,
                rmax_true: float):
        """Cast rays and return (r_true, dr, I) for hits.

        Footprint range extent of a ray: dr = r (d_el |n.t_el| + d_az |n.t_az|) / |n.u|, with t_el / t_az the unit
        tangents of the ray direction u along elevation / azimuth and d_el / d_az the angular spacing the ray stands
        for.  On a grazing sea bed this is r d_el tan(i); on a vertical pole the elevation term vanishes, so thin
        objects keep their true range extent."""
        gid, dist, nrm = self.caster.cast(d, origin, dirs_w, rmax_true)
        hit = (gid >= 0) & (dist > 0) & (dist <= rmax_true)
        self.last_first_hit = np.where(hit, dist, np.inf)     # per ray, true range (inf = no hit within range)
        if not np.any(hit):
            return np.zeros(0), np.zeros(0), np.zeros(0)
        r = dist[hit]
        n = nrm[hit]
        cosi = np.abs(np.einsum("ij,ij->i", dirs_w[hit], n))
        sig = self.mat.sigma(gid[hit], cosi)
        depth = max(-float(origin[2]), 0.0)
        resid = 10.0 ** (-2.0 * (self.alpha_true(depth) - self.alpha_cfg()) * r / 10.0)
        I = gain2[hit] * sig * domega[hit] / np.maximum(cosi, 0.05) * resid
        ne = np.abs(np.einsum("ij,ij->i", t_el_w[hit], n))
        na = np.abs(np.einsum("ij,ij->i", t_az_w[hit], n))
        dr = np.minimum(np.clip(r * (d_el[hit] * ne + d_az[hit] * na) / np.maximum(cosi, 0.05), 0.0, 0.5 * r),
                        self.dr_cap[gid[hit]])
        return r, dr, I

    def finish_profile(self, prof: np.ndarray, rmax_meas: float, res_m: float, gain_index: int) -> np.ndarray:
        """Range blur, speckle, noise, ring-down, log compression -> uint8."""
        hid = self.hid
        n = prof.shape[0]
        dr_s = rmax_meas / n
        prof = prof / dr_s                                  # intensity per metre of range (sampling-independent)
        k = _gauss_kernel(max(res_m / dr_s / 2.355, 0.0))
        if k.size > 1:
            prof = np.convolve(prof, k, mode="same")
        L = hid.speckle_looks
        prof = prof * self.rng.gamma(L, 1.0 / L, n)
        r = (np.arange(n) + 0.5) * dr_s
        a = self.alpha_cfg()
        n0 = 10.0 ** (hid.noise_db / 10.0) * (np.maximum(r, 0.3) / hid.noise_ref_m) ** 2 \
            * 10.0 ** (2.0 * a * (r - hid.noise_ref_m) / 10.0)
        prof = prof + n0 * self.rng.exponential(1.0, n)
        prof = prof + 10.0 ** (hid.ring_db / 10.0) * np.exp(-r / hid.ring_len_m)
        db = 10.0 * np.log10(np.maximum(prof, 1e-30))
        floor = hid.floor_db - (gain_index - 1) * hid.gain_step_db
        v = (db - floor) / hid.dyn_db * 255.0
        return np.clip(v + 0.5, 0, 255).astype(np.uint8)


@dataclass
class SonarPingContext:
    """What a ScanningSonar extension hook gets for one ping (all world frame, true ranges unless noted).

    hook(ctx, prof): called after the scene's returns are deposited and before range blur / speckle / noise / log
    compression; add task-specific scatterers with ctx.deposit(r_true, dr_true, intensity) (same intensity units as
    the scene returns: two-way gain x sigma x solid angle x residual absorption, see the module docstring)."""
    t: float
    angle_grad: int
    theta_rad: float                       # head angle (clockwise from site +x)
    origin: np.ndarray                     # (3,) transducer position
    R: np.ndarray                          # (3, 3) site frame -> world
    dirs_w: np.ndarray                     # (N, 3) ray directions
    gain2: np.ndarray                      # (N,) two-way beam pattern of each ray
    domega: np.ndarray                     # (N,) solid angle each ray stands for [sr]
    first_hit_m: np.ndarray                # (N,) first scene hit along each ray, true range (inf = none)
    rmax_true: float                       # listening window in true range [m]
    scale: float                           # r_meas / r_true (configured / true speed of sound)
    alpha_resid_db_m: float                # true - configured absorption [dB/m] (two-way factor 10^(-2 a r / 10))
    depth_m: float
    range_m: float                         # configured range (measured units)
    n_samples: int
    prof: np.ndarray = field(repr=False, default=None)

    def residual(self, r_true) -> np.ndarray:
        """Two-way residual absorption factor after the device TVG at true range r."""
        return 10.0 ** (-2.0 * self.alpha_resid_db_m * np.asarray(r_true, float) / 10.0)

    def deposit(self, r_true, dr_true, intensity) -> None:
        """Spread each return uniformly over [r - dr/2, r + dr/2] (true range) into this ping's profile."""
        r = np.atleast_1d(np.asarray(r_true, float)) * self.scale
        dr = np.broadcast_to(np.asarray(dr_true, float), r.shape) * self.scale
        I = np.broadcast_to(np.asarray(intensity, float), r.shape)
        deposit_profile(self.prof, r, np.asarray(dr, float), np.asarray(I, float), self.range_m)


def deposit_profile(prof: np.ndarray, r_meas: np.ndarray, dr_meas: np.ndarray, intensity: np.ndarray,
                    range_m: float) -> None:
    """Public form of the range-profile deposit (measured ranges, see SonarPingContext.deposit)."""
    _deposit(prof, np.asarray(r_meas, float), np.asarray(dr_meas, float), np.asarray(intensity, float), float(range_m))


# ===================================================================================================== mechanical

@dataclass
class ScanningSonarConfig:
    frequency_hz: float = 750e3
    beam_h_deg: float = 2.0
    beam_v_deg: float = 25.0
    range_m: float = 20.0
    n_samples: int = 1200
    min_range_m: float = 0.75
    max_range_m: float = 50.0
    max_samples: int = 1200
    step_grad: int = 1
    sector_grad: tuple = (0, 400)          # [start, stop) in gradians; (0, 400) = continuous 360 deg
    gain_index: int = 1                    # 0 low, 1 normal, 2 high
    speed_of_sound_cfg: float = 1500.0
    transmit: bool = True
    n_el_rays: int = 96                    # elevation bins; rays per ping = n_el_rays * n_az_rays + side lobes; cost
                                           # ~ linear in rays (96 -> 48 saves ~40 % of the ray casting, coarser
                                           # elevation sampling: check thin / small targets before lowering it)
    n_az_rays: int = 5                     # main-lobe rays per elevation bin (random azimuth from the pattern)
    el_extent_deg: float = 20.0            # rays cover +-el_extent (beyond the -3 dB half width of 12.5 deg)
    overhead_s: float = 0.00897            # fitted: t_ping = 2R/c + overhead + per_m * R
    overhead_per_m_s: float = 0.000137
    range_resolution_frac: float = 0.0008  # vendor: 0.08 % of range


@dataclass
class SonarPing:
    t: float
    valid: bool
    angle_grad: int
    angle_rad: float                       # clockwise from site +x, viewed from site +z
    range_m: float                         # configured range (measured with the configured sound speed)
    data: np.ndarray                       # (n_samples,) uint8 intensity
    sample_period_s: float
    speed_of_sound_cfg: float
    gain_index: int


def scanning_sonar_public_spec(cfg: ScanningSonarConfig) -> list[SpecItem]:
    return [
        SpecItem("type", "mechanically scanned single-beam imaging sonar"),
        SpecItem("frequency", cfg.frequency_hz / 1e3, "kHz"),
        SpecItem("beam width (H x V, -3 dB)", f"{cfg.beam_h_deg} x {cfg.beam_v_deg}", "deg"),
        SpecItem("mechanical step", f"{cfg.step_grad} gradian ({0.9 * cfg.step_grad:.1f} deg); 400 per turn"),
        SpecItem("range", f"{cfg.min_range_m} .. {cfg.max_range_m}", "m", "user setting"),
        SpecItem("samples per ping", f"up to {cfg.max_samples}"),
        SpecItem("range resolution", "0.08 % of range"),
        SpecItem("360 deg scan time", "about 4 s at 1 m range, about 33 s at 50 m range"),
        SpecItem("output", "8-bit echo intensity per range sample; gain low/normal/high; speed of sound user setting"),
    ]


class ScanningSonar:
    """Mechanically scanned imaging sonar.  Call `update(t, d)` every physics step (or as often as possible); it
    generates every ping whose scheduled time has passed, using the current pose."""

    def __init__(self, m: mujoco.MjModel, materials: MaterialMap, rng: np.random.Generator,
                 water: WaterColumn = SEAWATER_TEMPERATE, site: str = "sonar_head",
                 config: Optional[ScanningSonarConfig] = None, hidden: Optional[SonarHidden] = None,
                 geomgroup: Optional[Sequence[int]] = None, exclude_body: Optional[int] = None, t0: float = 0.0,
                 extra=None):
        self.cfg = config or ScanningSonarConfig()
        self.hid = hidden or SonarHidden()
        self.core = _SonarCore(m, site, materials, water, rng, self.hid, self.cfg.frequency_hz, geomgroup,
                               exclude_body)
        self.rng = rng
        self.angle = 0
        self.direction = +1
        self.configure(sector_grad=self.cfg.sector_grad)          # normalise (start, stop) and set the start angle
        self.next_t = float(t0)
        self.buffer: list[SonarPing] = []
        self.extra = extra                 # optional hook(ctx: SonarPingContext, prof) per ping (see SonarPingContext)

    # ---------------------------------------------------------------- controls (as the device exposes them)
    def configure(self, range_m: Optional[float] = None, n_samples: Optional[int] = None,
                  sector_grad: Optional[tuple] = None, step_grad: Optional[int] = None,
                  gain_index: Optional[int] = None, speed_of_sound_cfg: Optional[float] = None,
                  transmit: Optional[bool] = None) -> None:
        c = self.cfg
        if range_m is not None:
            c.range_m = float(min(max(range_m, c.min_range_m), c.max_range_m))
        if n_samples is not None:
            c.n_samples = int(min(max(n_samples, 200), c.max_samples))
        if sector_grad is not None:
            # (start, stop) in gradians, clockwise from start to stop; equal or 400 apart = full circle
            a = int(sector_grad[0]) % 400
            span = (int(sector_grad[1]) - int(sector_grad[0])) % 400
            c.sector_grad = (a, a + (span if span else 400))
            self.angle = a
            self.direction = +1
        if step_grad is not None:
            c.step_grad = int(min(max(step_grad, 1), 10))
        if gain_index is not None:
            c.gain_index = int(min(max(gain_index, 0), 2))
        if speed_of_sound_cfg is not None:
            c.speed_of_sound_cfg = float(min(max(speed_of_sound_cfg, 1300.0), 1700.0))
        if transmit is not None:
            c.transmit = bool(transmit)

    def ping_period(self) -> float:
        c = self.cfg
        return 2.0 * c.range_m / c.speed_of_sound_cfg + c.overhead_s + c.overhead_per_m_s * c.range_m

    def scan_time_360(self) -> float:
        return 400.0 / self.cfg.step_grad * self.ping_period()

    # ---------------------------------------------------------------- simulation
    def update(self, t: float, d: mujoco.MjData) -> list[SonarPing]:
        out = []
        if not self.cfg.transmit:
            self.next_t = max(self.next_t, t)
            return out
        while self.next_t <= t + 1e-12:
            p = self._ping(self.next_t, d)
            out.append(p)
            self.next_t += self.ping_period()
            self._advance()
        self.buffer.extend(out)
        return out

    def pop(self) -> list[SonarPing]:
        b, self.buffer = self.buffer, []
        return b

    def _advance(self):
        c = self.cfg
        a0, a1 = c.sector_grad
        span = a1 - a0
        if span >= 400:
            self.angle = (self.angle + c.step_grad) % 400
            return
        pos = (self.angle - a0) % 400 + self.direction * c.step_grad
        if pos > span:
            pos = 2 * span - pos
            self.direction = -1
        elif pos < 0:
            pos = -pos
            self.direction = +1
        self.angle = int((a0 + pos) % 400)

    def _sample_beam(self, theta: float):
        """Monte-Carlo sampling of the two-way beam pattern.  Main lobe: elevation stratified-uniform over
        +-el_extent (weight = two-way elevation pattern), azimuth drawn from the two-way Gaussian main lobe
        (importance sampling, equal weights) -> thin targets are hit in proportion to their angular width.
        Side lobes: fixed azimuths at the sinc^2 side-lobe peaks with a coarser elevation grid."""
        c = self.cfg
        nel = c.n_el_rays
        naz = c.n_az_rays
        ext = math.radians(c.el_extent_deg)
        del_ = 2 * ext / nel
        el = -ext + del_ * (np.arange(nel)[:, None] + self.rng.random((nel, naz)))
        s1 = math.radians(c.beam_h_deg) / 2.355                 # one-way sigma
        az = (s1 / math.sqrt(2.0)) * self.rng.standard_normal((nel, naz))
        s_v = math.radians(c.beam_v_deg) / 2.355
        w_az = s1 * math.sqrt(math.pi) / naz                    # integral of exp(-az^2/s1^2) over az, per sample
        g_main = np.exp(-(el / s_v) ** 2)
        dom_main = np.full(el.shape, w_az * del_) * np.cos(el)
        # side lobes
        nsl = max(4, nel // 4)
        dsl = 2 * ext / nsl
        el_s = -ext + dsl * (np.arange(nsl) + self.rng.random(nsl))
        null = math.radians(c.beam_h_deg) / 0.886
        az_s, g_s = [], []
        for j, lev in enumerate(self.hid.sidelobe_db):
            a = (1.43 if j == 0 else 2.46) * null
            for sgn in (-1.0, 1.0):
                az_s.append(np.full(nsl, sgn * a))
                g_s.append(np.full(nsl, 10.0 ** (2.0 * lev / 10.0)))
        az_s = np.concatenate(az_s)
        g_s = np.concatenate(g_s) * np.exp(-(np.tile(el_s, len(az_s) // nsl) / s_v) ** 2)
        el_s2 = np.tile(el_s, len(az_s) // nsl)
        dom_s = np.full(az_s.shape, 0.5 * null * dsl) * np.cos(el_s2)
        AZ = np.concatenate([az.reshape(-1), az_s]) - theta     # clockwise head angle -> CCW azimuth
        EL = np.concatenate([el.reshape(-1), el_s2])
        g2 = np.concatenate([g_main.reshape(-1), g_s])
        dom = np.concatenate([dom_main.reshape(-1), dom_s])
        d_el = np.concatenate([np.full(el.size, del_), np.full(az_s.size, dsl)])
        # azimuth is sampled densely at random, so only the elevation spacing needs footprint spreading (spreading
        # along azimuth by tangent-plane extrapolation smears curved objects far behind their silhouette)
        d_az = np.zeros(el.size + az_s.size)
        t_el, t_az = _beam_tangents(AZ, EL)
        return _beam_dirs(AZ, EL), g2, dom, t_el, t_az, d_el, d_az

    def _ping(self, t: float, d: mujoco.MjData) -> SonarPing:
        c, core = self.cfg, self.core
        origin, R = site_pose(d, core.sid)
        depth = max(-float(origin[2]), 0.0)
        c_true = core.water.sound_speed(depth)
        scale = c.speed_of_sound_cfg / c_true                 # r_meas = r_true * scale
        rmax_true = c.range_m / scale
        theta = self.angle * GRAD
        dirs, g2, dom, t_el, t_az, d_el, d_az = self._sample_beam(theta)
        dirs_w = dirs @ R.T
        r, dr, I = core.returns(d, origin, dirs_w, g2, dom, t_el @ R.T, t_az @ R.T, d_el, d_az, rmax_true)
        prof = np.zeros(c.n_samples)
        if r.size:
            _deposit(prof, r * scale, dr * scale, I, c.range_m)
        if self.extra is not None:
            ctx = SonarPingContext(t=t, angle_grad=int(self.angle), theta_rad=theta, origin=origin, R=R, dirs_w=dirs_w,
                                   gain2=g2, domega=dom, first_hit_m=core.last_first_hit, rmax_true=rmax_true,
                                   scale=scale, alpha_resid_db_m=core.alpha_true(depth) - core.alpha_cfg(), depth_m=depth,
                                   range_m=c.range_m, n_samples=c.n_samples, prof=prof)
            self.extra(ctx, prof)
        res = max(c.range_resolution_frac * c.range_m, 0.5 * c.range_m / c.n_samples)
        data = core.finish_profile(prof, c.range_m, res, c.gain_index)
        valid = bool(origin[2] < 0.0)
        if not valid:
            data[:] = 0
        sp = 2.0 * c.range_m / c.speed_of_sound_cfg / c.n_samples
        return SonarPing(t=t, valid=valid, angle_grad=int(self.angle), angle_rad=theta, range_m=c.range_m, data=data,
                         sample_period_s=sp, speed_of_sound_cfg=c.speed_of_sound_cfg, gain_index=c.gain_index)


class ScanBuffer:
    """Latest ping per gradian, for display or for an agent-side mapper."""

    def __init__(self, n_samples: int = 1200):
        self.n = n_samples
        self.data = np.zeros((400, n_samples), np.uint8)
        self.range_m = np.zeros(400)
        self.t = np.full(400, -np.inf)

    def add(self, pings: Sequence[SonarPing]) -> None:
        for p in pings:
            row = p.data
            if row.shape[0] != self.n:
                x = np.linspace(0, row.shape[0] - 1, self.n)
                row = np.interp(x, np.arange(row.shape[0]), row.astype(float)).astype(np.uint8)
            self.data[p.angle_grad] = row
            self.range_m[p.angle_grad] = p.range_m
            self.t[p.angle_grad] = p.t

    def to_image(self, size: int = 600, range_m: Optional[float] = None) -> np.ndarray:
        """Plan-position (PPI) raster: image up = sonar angle 0, clockwise angles to the right.  uint8 (size, size)."""
        rm = float(range_m if range_m is not None else (self.range_m.max() or 1.0))
        yy, xx = np.mgrid[0:size, 0:size]
        X = (xx + 0.5 - size / 2) / (size / 2) * rm
        Y = (size / 2 - yy - 0.5) / (size / 2) * rm
        rr = np.hypot(X, Y)
        ang = (np.arctan2(X, Y) % (2 * np.pi)) / GRAD          # clockwise from up, in gradians
        gi = np.floor(ang + 0.5).astype(np.int64) % 400
        rng_ok = self.range_m[gi] > 0
        si = np.floor(rr / np.where(rng_ok, self.range_m[gi], 1.0) * self.n).astype(np.int64)
        ok = rng_ok & (si < self.n) & (rr <= rm)
        img = np.zeros((size, size), np.uint8)
        img[ok] = self.data[gi[ok], si[ok]]
        return img


# ===================================================================================================== multibeam

@dataclass
class MultibeamSonarConfig:
    frequency_hz: float = 750e3
    fov_h_deg: float = 130.0
    fov_v_deg: float = 20.0
    n_beams: int = 256
    beam_h_deg: float = 1.0                # angular resolution (-3 dB beam width, one-way)
    range_m: float = 30.0
    min_range_m: float = 0.1
    max_range_m: float = 120.0
    n_ranges: int = 512
    range_resolution_m: float = 0.004
    max_rate_hz: float = 40.0
    overhead_s: float = 0.005
    gain_index: int = 1
    speed_of_sound_cfg: float = 1500.0
    n_el_rays: int = 16
    az_oversample: int = 2                 # rays per beam in azimuth


@dataclass
class MultibeamFrame:
    t: float
    valid: bool
    bearings_rad: np.ndarray               # (n_beams,) CCW positive (port), 0 = site +x
    range_m: float
    image: np.ndarray                      # (n_ranges, n_beams) uint8, row 0 = nearest range
    speed_of_sound_cfg: float
    gain_index: int


def multibeam_public_spec(cfg: MultibeamSonarConfig) -> list[SpecItem]:
    return [
        SpecItem("type", "forward-looking multibeam imaging sonar"),
        SpecItem("frequency", cfg.frequency_hz / 1e3, "kHz"),
        SpecItem("field of view (H x V)", f"{cfg.fov_h_deg} x {cfg.fov_v_deg}", "deg"),
        SpecItem("beams", cfg.n_beams),
        SpecItem("angular resolution", cfg.beam_h_deg, "deg"),
        SpecItem("range", f"{cfg.min_range_m} .. {cfg.max_range_m}", "m"),
        SpecItem("range resolution", cfg.range_resolution_m * 1000, "mm"),
        SpecItem("max update rate", cfg.max_rate_hz, "Hz", "limited by 2 R / c"),
    ]


class MultibeamSonar:
    def __init__(self, m: mujoco.MjModel, materials: MaterialMap, rng: np.random.Generator,
                 water: WaterColumn = SEAWATER_TEMPERATE, site: str = "sonar_head",
                 config: Optional[MultibeamSonarConfig] = None, hidden: Optional[SonarHidden] = None,
                 geomgroup: Optional[Sequence[int]] = None, exclude_body: Optional[int] = None, t0: float = 0.0):
        self.cfg = config or MultibeamSonarConfig()
        self.hid = hidden or SonarHidden(floor_db=-80.0, noise_db=-98.0)
        self.core = _SonarCore(m, site, materials, water, rng, self.hid, self.cfg.frequency_hz, geomgroup,
                               exclude_body)
        self.rng = rng
        self.sched = RateScheduler(1.0 / self.period(), t0)

    def period(self) -> float:
        c = self.cfg
        return max(1.0 / c.max_rate_hz, 2.0 * c.range_m / c.speed_of_sound_cfg + c.overhead_s)

    def configure(self, range_m: Optional[float] = None, gain_index: Optional[int] = None,
                  speed_of_sound_cfg: Optional[float] = None) -> None:
        c = self.cfg
        if range_m is not None:
            c.range_m = float(min(max(range_m, 1.0), c.max_range_m))
        if gain_index is not None:
            c.gain_index = int(min(max(gain_index, 0), 2))
        if speed_of_sound_cfg is not None:
            c.speed_of_sound_cfg = float(speed_of_sound_cfg)
        self.sched.set_rate(1.0 / self.period())

    def update(self, t: float, d: mujoco.MjData) -> Optional[MultibeamFrame]:
        if not self.sched.due(t):
            return None
        self.sched.fire(t)      # no backlog: a late call produces one frame now
        return self.capture(t, d)

    def capture(self, t: float, d: mujoco.MjData) -> MultibeamFrame:
        c, core = self.cfg, self.core
        origin, R = site_pose(d, core.sid)
        depth = max(-float(origin[2]), 0.0)
        scale = c.speed_of_sound_cfg / core.water.sound_speed(depth)
        rmax_true = c.range_m / scale
        h = math.radians(c.fov_h_deg) / 2.0
        nb = c.n_beams
        nray_az = nb * c.az_oversample
        daz = 2 * h / nray_az
        az = -h + daz * (np.arange(nray_az) + self.rng.random(nray_az))
        v = math.radians(c.fov_v_deg) / 2.0
        ext = v * 1.4
        nel = c.n_el_rays
        del_ = 2 * ext / nel
        AZ = np.repeat(az[:, None], nel, axis=1)
        EL = -ext + del_ * (np.arange(nel)[None, :] + self.rng.random((nray_az, nel)))   # per-ray jitter
        s_v = (2 * v) / 2.355
        g2 = np.exp(-(EL / s_v) ** 2).reshape(-1)
        dom = (daz * del_ * np.cos(EL)).reshape(-1)
        dirs = _beam_dirs(AZ.reshape(-1), EL.reshape(-1)) @ R.T
        gid, dist, nrm = core.caster.cast(d, origin, dirs, rmax_true)
        hit = (gid >= 0) & (dist > 0)
        img = np.zeros((c.n_ranges, nray_az))
        if np.any(hit):
            r = dist[hit]
            cosi = np.abs(np.einsum("ij,ij->i", dirs[hit], nrm[hit]))
            sig = core.mat.sigma(gid[hit], cosi)
            resid = 10.0 ** (-2.0 * (core.alpha_true(depth) - core.alpha_cfg()) * r / 10.0)
            I = g2[hit] * sig * dom[hit] / np.maximum(cosi, 0.05) * resid
            col = np.repeat(np.arange(nray_az), nel)[hit]
            t_el, t_az = _beam_tangents(AZ.reshape(-1)[hit], EL.reshape(-1)[hit])
            ne = np.abs(np.einsum("ij,ij->i", t_el @ R.T, nrm[hit]))
            dr = np.minimum(np.clip(r * (del_ * ne) / np.maximum(cosi, 0.05), 0, 0.5 * r), core.dr_cap[gid[hit]]) * scale
            rm = r * scale
            # deposit per column with the difference-array trick (vectorised over columns)
            n = c.n_ranges
            sc = n / c.range_m
            a = np.clip((rm - 0.5 * dr) * sc, 0, n - 1e-6)
            b = np.clip((rm + 0.5 * dr) * sc, 0, n - 1e-6)
            ia = a.astype(np.int64)
            ib = b.astype(np.int64)
            span = ib - ia + 1
            dens = I / span
            diff = np.zeros((n + 1, nray_az))
            np.add.at(diff, (ia, col), dens)
            np.add.at(diff, (ib + 1, col), -dens)
            img = np.cumsum(diff, axis=0)[:n]
        # azimuth beam pattern: main lobe + side lobes as a convolution across ray columns
        bw = math.radians(c.beam_h_deg)
        s = (bw / 2.355) / math.sqrt(2) / daz                # two-way sigma in ray columns
        rr = int(math.ceil(4 * max(s, 1)))
        null = bw / 0.886 / daz
        kx = np.arange(-int(3 * null), int(3 * null) + 1)
        ker = np.exp(-0.5 * (kx / max(s, 0.3)) ** 2)
        for j, lev in enumerate(self.hid.sidelobe_db):
            pos = (1.43 if j == 0 else 2.46) * null
            ker += 10 ** (2 * lev / 10) * (np.exp(-0.5 * ((kx - pos) / (0.3 * null)) ** 2)
                                            + np.exp(-0.5 * ((kx + pos) / (0.3 * null)) ** 2))
        ker /= ker.sum()
        img = np.apply_along_axis(lambda row: np.convolve(row, ker, mode="same"), 1, img) if img.any() else img
        img = img.reshape(c.n_ranges, nb, c.az_oversample).sum(axis=2)
        img = img[:, ::-1]                                    # columns: port (left) -> starboard (right)
        dr_s = c.range_m / c.n_ranges
        img = img / dr_s
        img = img * self.rng.gamma(self.hid.speckle_looks, 1.0 / self.hid.speckle_looks, img.shape)
        rr_ax = (np.arange(c.n_ranges) + 0.5) * dr_s
        a = core.alpha_cfg()
        n0 = 10.0 ** (self.hid.noise_db / 10.0) * (np.maximum(rr_ax, 0.3) / self.hid.noise_ref_m) ** 2 \
            * 10.0 ** (2.0 * a * (rr_ax - self.hid.noise_ref_m) / 10.0)
        img = img + n0[:, None] * self.rng.exponential(1.0, img.shape)
        db = 10.0 * np.log10(np.maximum(img, 1e-30))
        floor = self.hid.floor_db - (c.gain_index - 1) * self.hid.gain_step_db
        out = np.clip((db - floor) / self.hid.dyn_db * 255.0 + 0.5, 0, 255).astype(np.uint8)
        b = np.linspace(math.radians(c.fov_h_deg) / 2.0, -math.radians(c.fov_h_deg) / 2.0, nb)
        valid = bool(origin[2] < 0)
        return MultibeamFrame(t=t, valid=valid, bearings_rad=b, range_m=c.range_m, image=out,
                              speed_of_sound_cfg=c.speed_of_sound_cfg, gain_index=c.gain_index)


def fan_image(frame: MultibeamFrame, size: int = 512) -> np.ndarray:
    """Cartesian fan view of a multibeam frame (apex at the bottom centre, port on the left).  uint8 (size, size)."""
    nr, nb = frame.image.shape
    b = frame.bearings_rad
    h = max(abs(b[0]), abs(b[-1]))
    W = size
    Hh = size
    yy, xx = np.mgrid[0:Hh, 0:W]
    span = math.sin(h) if h < math.pi / 2 else 1.0
    X = (xx + 0.5 - W / 2) / (W / 2) * frame.range_m * span
    Y = (Hh - yy - 0.5) / Hh * frame.range_m
    rr = np.hypot(X, Y)
    th = np.arctan2(-X, Y)                                  # CCW positive (port) like bearings
    ri = np.floor(rr / frame.range_m * nr).astype(np.int64)
    bi = np.floor((b[0] - th) / (b[0] - b[-1]) * (nb - 1) + 0.5).astype(np.int64)
    ok = (ri < nr) & (bi >= 0) & (bi < nb)
    out = np.zeros((Hh, W), np.uint8)
    out[ok] = frame.image[ri[ok], bi[ok]]
    return out
