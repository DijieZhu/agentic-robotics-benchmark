"""The plume, volume reverberation and drifting clutter in the scanning sonar, and the plume's bubbles in the camera
(cell side; uwkit is not modified: both couplings are subclasses that use uwkit 0.3.0's public extension hooks).

Sonar (`PlumeSonar`, a uwkit ScanningSonar with `extra` = the hook below).  uwkit casts its rays, deposits the scene's
returns and calls the hook before range blur / speckle / noise / log compression; the hook adds, in the same intensity
units as the scene returns (two-way beam pattern x cross-section x solid angle x residual absorption after the device's
20 log r + 2 alpha r TVG):
  * point scatterers (plume particles, clutter clump points), cross-section sigma [m^2]:
        I = sigma * B^2(az, el) / r^2 * 10^(-2 (alpha_true - alpha_cfg) r / 10),  B^2 = exp(-(az/s_h)^2 - (el/s_v)^2)
    one point is hidden when its elevation stratum's first scene hit is nearer (median over the stratum's rays);
    the particles are taken from the plume's horizontal index: only the cells whose box can intersect the beam wedge
    (separating-plane test against the wedge's six faces) are projected (`binned=False` projects all, same result);
  * volume reverberation, s_v(h) = s_v0 (1 + (10^(bed_excess / 10) - 1) exp(-h / H)), h = height above the sea bed:
        dI = sum_strata B^2 dOmega s_v(h(r)) dr   for r before the stratum's first hit.
  Values (DESIGN 2.3.5, [placeholder] as used by G3): particle TS -55 dB; s_v0 = -64 dB + 10 lg(particle load) (c 0.42
  -> -64, c 1.2 -> -59), +6 dB at the bed with a 3 m e-folding height.
Camera (`PlumeCamera`, a uwkit UnderwaterCamera with `extra_radiance` = the bubble hook): every particle within 6 m of
the camera is imaged by uwkit's `render_points` (the marine-snow optics: lamp cones, inverse square, two-way
attenuation, depth test against the frame's range, defocus), cross-section of a 2-6 mm disc (log-uniform per particle
id), snow phase function 0.06 / 0.30 1/sr, gain 1.0 [placeholder, not calibrated].

Known artefact (uwkit, documented there): along bars thinner than about a pixel the lamp-lit edge shows a chain of
one-pixel bright dots fixed in the world; a single frame cannot tell them from bubbles, a two-frame motion test can.

The sentinel `UWKIT_SOURCE_PINS` records the sha256 of the uwkit sources these couplings rely on (`uwkit_source_hashes`
recomputes them; tests/test_plume.py fails when uwkit changes under them).
"""
from __future__ import annotations

import hashlib
import inspect
import math
import time
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from uwkit.sensors.sonar import ScanningSonar, SonarPingContext
from uwkit.sensors.camera import UnderwaterCamera

from .plume import hash3

# ------------------------------------------------------------------------------------------------ sentinel
UWKIT_VERSION_PINNED = "0.3.0"
UWKIT_SOURCE_PINS = {          # sha256 of inspect.getsource(), uwkit 0.3.0 as installed 2026-11-15 (records/P2_P06.md)
    "sensors.sonar.ScanningSonar._ping": "396325b299992eec03103b6b64b8f40f3a564dd37839f432bf87640dd78bd546",
    "sensors.camera.UnderwaterCamera._marine_snow": "c7de860e4c39526d034d06a3c779287a343b0bcff7f1463638568cef818ee60d",
    "sensors.sonar.ScanningSonar._sample_beam": "6bcadd73000ad3990e32a76c1a428d682762498f9838019ea03b46e217873464",
    "sensors.sonar._SonarCore.returns": "3c81983614db167d056071043e20c2748563f57f3c52fd360595e2491f4f23a3",
    "sensors.sonar.SonarPingContext": "ceda0d4dee25fb76b75fce77359ba8ca28341f08fb3bf0a29b53f601f26b0a01",
    "sensors.sonar.deposit_profile": "78a42eda1434fc6731c98333176ebb906f42c4264ebd786225f928b7398b56d2",
    "sensors.sonar._deposit": "3593e8a8872e350379802696e6be0c87dc3ab95f7c25e9db4d872515abbe0780",
    "sensors.camera.UnderwaterCamera._image_formation": "7f42039857e19f597a53ef836bd7dfad402745417ea267c3563b4d71dd03f80b",
    "sensors.camera.UnderwaterCamera.render_points": "12dfa1423a3043eaf7e38cdec7720f7f54897ee3952d7e423602d231ae0fcea2",
    "sensors.camera.UnderwaterCamera._image_points": "7476f2ce2b0ec0a336aa66a6882a15e18075c1dfb732bd7de8a438c12e3151b2",
    "dynamics.current.Tidal": "db76b634b1c3bcc61644c1fab1cdec297930e581ce60513b3cff986f1e197425",
    "dynamics.current.BoundaryLayer": "04366429b9475a04c86af6c58aa3b62ed92acb1881f7a187b29f0a325b70e9ac",
    "dynamics.current.GaussMarkov": "7e0cc2583c9143aebcc075450047831738a7a8f7022f49e554e91c44e9cd1376",
    "dynamics.current.Scaled": "683de1df098de29d500993f94d217336af3bd0e0ec753013945d0e2ba9d43db6",
}


def _uwkit_objects():
    import uwkit.sensors.sonar as S
    import uwkit.sensors.camera as Cm
    import uwkit.dynamics.current as CU
    return {
        "sensors.sonar.ScanningSonar._ping": S.ScanningSonar._ping,
        "sensors.camera.UnderwaterCamera._marine_snow": Cm.UnderwaterCamera._marine_snow,
        "sensors.sonar.ScanningSonar._sample_beam": S.ScanningSonar._sample_beam,
        "sensors.sonar._SonarCore.returns": S._SonarCore.returns,
        "sensors.sonar.SonarPingContext": S.SonarPingContext,
        "sensors.sonar.deposit_profile": S.deposit_profile,
        "sensors.sonar._deposit": S._deposit,
        "sensors.camera.UnderwaterCamera._image_formation": Cm.UnderwaterCamera._image_formation,
        "sensors.camera.UnderwaterCamera.render_points": Cm.UnderwaterCamera.render_points,
        "sensors.camera.UnderwaterCamera._image_points": Cm.UnderwaterCamera._image_points,
        "dynamics.current.Tidal": CU.Tidal,
        "dynamics.current.BoundaryLayer": CU.BoundaryLayer,
        "dynamics.current.GaussMarkov": CU.GaussMarkov,
        "dynamics.current.Scaled": CU.Scaled,
    }


def uwkit_source_hashes() -> dict:
    """sha256 of inspect.getsource() of every uwkit object the couplings depend on."""
    return {k: hashlib.sha256(inspect.getsource(o).encode()).hexdigest() for k, o in _uwkit_objects().items()}


# ------------------------------------------------------------------------------------------------ acoustics
def particle_load(c550: float) -> float:
    """Particle-scattering factor of the site water relative to uwkit's clear water, from the beam attenuation at
    550 nm (the factor uwkit's WaterOptics.scaled_particles takes)."""
    from uwkit.sensors import water as W
    base = W.CLEAR
    return float((float(c550) - base.a_[1] - base.b_water[1]) / base.b_particle[1])


def site_water(c550: float):
    from uwkit.sensors import water as W
    return W.CLEAR.scaled_particles(particle_load(c550), name="site")


@dataclass
class AcousticSpec:
    sv_db: float = -64.0          # volume backscattering strength far above the bed [dB re 1/m] (clear water)
    bed_excess_db: float = 6.0    # extra at the sea bed
    bed_scale_m: float = 3.0      # e-folding height of the excess
    particle_ts_db: float = -55.0  # one plume particle (a cluster of bubbles) [dB re 1 m^2]

    @classmethod
    def for_water(cls, c550: float, **kw) -> "AcousticSpec":
        """s_v0 = -64 dB + 10 lg(particle load) [placeholder, G3]."""
        base = cls(**kw)
        base.sv_db = base.sv_db + 10.0 * math.log10(max(particle_load(c550), 1e-3))
        return base

    @classmethod
    def from_site(cls, site: dict) -> "AcousticSpec":
        a = site.get("acoustic")
        if a:
            names = {f.name for f in fields(cls)}
            return cls(**{k: float(v) for k, v in a.items() if k in names})
        return cls.for_water(site["water_c550"])


class PlumeSonar(ScanningSonar):
    """uwkit ScanningSonar + plume / clutter point scatterers + volume reverberation (see the module docstring).

    plume: plume.Plume (or None: reverberation only); seabed_depth_m: default plume.flow.seabed_depth.
    Timing (thread CPU, the hook only; the plume stepping is timed by the plume): c_extra / n_extra, and the per-ping
    list `extra_cpu_s` when record_timing."""

    def __init__(self, m, materials, rng, plume=None, acoustic: Optional[AcousticSpec] = None,
                 seabed_depth_m: Optional[float] = None, binned: bool = True, index_bin_m: float = 0.5,
                 record_timing: bool = True, **kw):
        if "extra" in kw:
            raise ValueError("PlumeSonar owns the extra hook")
        super().__init__(m, materials, rng, extra=self._plume_extra, **kw)
        self.plume = plume
        self.ac = acoustic or AcousticSpec()
        if seabed_depth_m is None:
            if plume is None:
                raise ValueError("seabed_depth_m is needed without a plume")
            seabed_depth_m = plume.flow.seabed_depth
        self.seabed_depth = float(seabed_depth_m)
        self.binned = bool(binned)
        self.index_bin_m = float(index_bin_m)
        self.record_timing = bool(record_timing)
        self.t_extra = 0.0
        self.c_extra = 0.0
        self.n_extra = 0
        self.extra_cpu_s: list = []
        self.last_meta = None
        self._rg_cache = {}
        self._beam = None
        self._att_k = 0.2 * math.log(10.0)                                  # 10^(-2 a r / 10) = exp(-att_k a r)
        self.set_acoustic(self.ac)

    def set_acoustic(self, acoustic: AcousticSpec) -> None:
        """Replace the acoustic spec (and the constants derived from it)."""
        self.ac = acoustic
        self._sig_p = 10.0 ** (acoustic.particle_ts_db / 10.0)
        self._inv_H = -1.0 / acoustic.bed_scale_m
        self._bed_A = 10.0 ** (acoustic.bed_excess_db / 10.0) - 1.0
        self._sv_lin_log = acoustic.sv_db / 10.0 * math.log(10.0)

    def ping_now(self, t: float, d):
        """One ping at time t with the current pose, then step the head (no scheduling)."""
        p = self._ping(t, d)
        self._advance()
        return p

    # ---------------------------------------------------------------- the hook
    def _plume_extra(self, ctx: SonarPingContext, prof: np.ndarray) -> None:
        pl = self.plume
        hi = None
        if pl is not None:
            pl.advance_to(ctx.t)                       # timed by the plume (c_step_s)
            if self.binned and pl.n > 0:
                hi = pl.horizontal_index(self.index_bin_m)  # once per plume step, timed by the plume (c_index_s)
        t0 = time.perf_counter()
        c0 = time.thread_time()
        meta = self._add_extras(ctx, prof, hi)
        c1 = time.thread_time()
        self.t_extra += time.perf_counter() - t0
        self.c_extra += c1 - c0
        self.n_extra += 1
        if self.record_timing:
            self.extra_cpu_s.append(c1 - c0)
        self.last_meta = meta

    def _beam_consts(self) -> dict:
        """Per-configuration constants of the beam (recomputed only when the fields that define them change)."""
        c = self.cfg
        key = (c.n_el_rays, c.n_az_rays, c.el_extent_deg, c.beam_h_deg, c.beam_v_deg)
        if self._beam is not None and self._beam[0] == key:
            return self._beam[1]
        nel, naz = c.n_el_rays, c.n_az_rays
        ext = math.radians(c.el_extent_deg)
        del_ = 2 * ext / nel
        s1 = math.radians(c.beam_h_deg) / 2.355                      # one-way sigma, as uwkit's _sample_beam
        s_v = math.radians(c.beam_v_deg) / 2.355
        ta, te = math.tan(3.5 * s1), math.tan(ext) * 1.05
        el_c = -ext + del_ * (np.arange(nel) + 0.5)                   # stratum centres
        # reverberation weight of a stratum: integral of the two-way pattern over its solid angle (the main-lobe
        # azimuth integral s1 sqrt(pi), the stratum's elevation width, cos(el)) - the ray sample average, without
        # its Monte-Carlo noise
        W = s1 * math.sqrt(math.pi) * del_ * np.cos(el_c) * np.exp(-(el_c / s_v) ** 2)
        K = np.array([[-1.0, 1.0, -ta, -ta, -te, -te],               # the beam wedge's six face normals in beam
                      [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],               # coordinates (axis, +azimuth, +elevation)
                      [0.0, 0.0, 0.0, 0.0, 1.0, -1.0]])
        g = dict(nel=nel, naz=naz, nmain=nel * naz, ext=ext, del_=del_, inv_del=1.0 / del_, s1=s1, s_v=s_v,
                 is1=1.0 / s1, isv=1.0 / s_v, ta=ta, te=te, cos_el=np.cos(el_c), sin_el=np.sin(el_c), W=W, KT=K.T)
        self._beam = (key, g)
        return g

    def _reverb_axis(self, ctx: SonarPingContext):
        key = (ctx.rmax_true, ctx.n_samples, ctx.range_m, ctx.scale)
        rgc = self._rg_cache.get(key)
        if rgc is None:
            rg = np.linspace(0.0, ctx.rmax_true, 49)[1:]
            dr_s = ctx.range_m / ctx.n_samples
            rs = (np.arange(ctx.n_samples) + 0.5) * dr_s / ctx.scale
            rgc = (rg, rs, dr_s / ctx.scale)
            if len(self._rg_cache) > 16:
                self._rg_cache.clear()
            self._rg_cache[key] = rgc
        return rgc

    def _reverb(self, ctx: SonarPingContext, g: dict, B: np.ndarray, shadow_r: np.ndarray) -> np.ndarray:
        """Volume reverberation profile (n_samples,): sum over the main-lobe elevation strata (centre directions) of
        B^2 dOmega s_v(h(r)) dr for r before the stratum's first scene hit (48 range nodes, linear interpolation)."""
        rg, rs, dr_true = self._reverb_axis(ctx)
        uz = g["cos_el"] * B[2, 0] + g["sin_el"] * B[2, 2]          # vertical component of each stratum's direction
        h = np.multiply.outer(uz, rg)
        h += ctx.origin[2] + self.seabed_depth                         # height above the bed along each stratum
        M = np.exp(h * self._inv_H)
        M *= self._bed_A
        M += 1.0
        vis = rg[None, :] < shadow_r[:, None]
        vis &= h > 0.0
        M *= vis
        rev = g["W"] @ M
        rev *= np.exp((self._sv_lin_log - self._att_k * ctx.alpha_resid_db_m * rg))
        return np.interp(rs, rg, rev) * dr_true

    def _add_extras(self, ctx: SonarPingContext, prof: np.ndarray, hi=None) -> dict:
        g = self._beam_consts()
        nel, naz, nmain = g["nel"], g["naz"], g["nmain"]
        ct, st = math.cos(ctx.theta_rad), math.sin(ctx.theta_rad)
        B = ctx.R @ np.array([[ct, st, 0.0], [-st, ct, 0.0], [0.0, 0.0, 1.0]])   # columns: axis, +azimuth, +elevation
        hitd = ctx.first_hit_m[:nmain].reshape(nel, naz)
        if naz % 2:
            shadow_r = np.partition(hitd, naz // 2, axis=1)[:, naz // 2]   # median of an odd count
        else:
            shadow_r = np.median(hitd, axis=1)
        o = ctx.origin
        rmax = ctx.rmax_true
        a_res = ctx.alpha_resid_db_m
        # ---- volume reverberation (main-lobe strata)
        prof += self._reverb(ctx, g, B, shadow_r)
        meta = {"n_particles_in_beam": 0, "n_clutter_in_beam": 0, "n_candidates": 0}
        # ---- point scatterers: plume particles (index cells that can meet the beam, or all) + clutter clump points
        pl = self.plume
        if pl is None:
            return meta
        n = prof.shape[0]
        to_bin = ctx.scale * n / ctx.range_m
        if pl.n > 0:
            Pc = pl.P[self._candidates(hi, g, B, o, rmax)] if hi is not None else pl.P
            meta["n_candidates"] = int(Pc.shape[0])
            if Pc.shape[0]:
                keep, r3, val = self._project(Pc, None, self._sig_p, g, B, o, rmax, shadow_r, a_res)
                if r3.size:
                    # each point into the range sample that contains it (its own 2 mm extent is below one sample)
                    i = (r3 * to_bin).astype(np.int64)
                    ok = i < n
                    prof += np.bincount(i[ok], weights=val[ok], minlength=n)
                    meta["n_particles_in_beam"] = int(r3.size)
        if pl.cP.shape[0]:
            reach = rmax + max(pl.cs.radius) + 1e-6
            dc = pl.cP - o
            if float(np.min(np.einsum("ij,ij->i", dc, dc))) <= reach * reach:
                cp, cs = pl.clutter_points()
                keep, r3, val = self._project(cp, cs, None, g, B, o, rmax, shadow_r, a_res)
                if r3.size:
                    i = (r3 * to_bin).astype(np.int64)
                    ok = i < n
                    prof += np.bincount(i[ok], weights=val[ok], minlength=n)
                    meta["n_clutter_in_beam"] = int(r3.size)
        return meta

    def _candidates(self, hi, g, B, o, rmax) -> np.ndarray:
        """Particle indices (ascending) of the index cells whose box is not separated from the beam wedge by one of
        its six faces (al >= 0.3, al <= rmax, |acr| <= al tan_a, |q_el| <= al tan_e): a box meets the half-space
        n.(x - o) <= c iff n.(centre - o) - (hb |n_x| + hb |n_y| + hz |n_z|) <= c."""
        if hi.cx.size == 0:
            return np.zeros(0, np.int64)
        NT = g["KT"] @ B.T                                       # (6, 3) face normals in W
        Nab = np.abs(NT)
        D = NT @ (hi.centre - o[:, None])                        # (6, M)
        D -= np.multiply.outer(Nab[:, 2], hi.hz + 1e-6)
        D -= ((0.5 * hi.bin + 1e-6) * (Nab[:, 0] + Nab[:, 1]) + np.array([-0.3, rmax, 0.0, 0.0, 0.0, 0.0])
              + 1e-9)[:, None]
        return hi.gather((D <= 0.0).all(axis=0))

    @staticmethod
    def _project(Pw, sig_each, sig_scalar, g, B, o, rmax, shadow_r, a_res):
        """Exact beam test and intensity of point scatterers Pw (n, 3).  Returns (indices kept, r_true, intensity)."""
        Q = (Pw - o) @ B                                    # (axis, +azimuth, +elevation) coordinates
        al = Q[:, 0]
        m = np.abs(Q[:, 1]) < al * g["ta"]
        m &= np.abs(Q[:, 2]) < al * g["te"]
        m &= al > 0.3
        m &= al < rmax
        idx = np.flatnonzero(m)
        if not idx.size:
            return idx, np.zeros(0), np.zeros(0)
        Q = Q[idx]
        q2 = Q * Q
        rho = np.sqrt(q2[:, 0] + q2[:, 1])
        r3 = np.sqrt(rho * rho + q2[:, 2])
        el = np.arctan2(Q[:, 2], rho)
        ext = g["ext"]
        istr = ((el + ext) * g["inv_del"]).astype(np.int64)
        np.clip(istr, 0, g["nel"] - 1, out=istr)
        ok = (r3 < shadow_r[istr]) & (r3 < rmax) & (np.abs(el) < ext)
        if not ok.all():
            idx, Q, el, r3 = idx[ok], Q[ok], el[ok], r3[ok]
            if not idx.size:
                return idx, np.zeros(0), np.zeros(0)
        az = np.arctan2(Q[:, 1], Q[:, 0])
        sg = sig_scalar if sig_each is None else sig_each[idx]
        val = sg * np.exp(-(az * g["is1"]) ** 2 - (el * g["isv"]) ** 2 - (0.2 * math.log(10.0) * a_res) * r3) / (r3 * r3)
        return idx, r3, val


# ------------------------------------------------------------------------------------------------ camera
@dataclass
class BubbleOptics:
    diam_mm: tuple = (2.0, 6.0)   # apparent diameter of one particle's highlight (log-uniform per particle id)
    phase_back: float = 0.06      # 1/sr, like uwkit's snow flake; bubbles are brighter forward
    phase_fwd: float = 0.30
    gain: float = 1.0             # brightness multiplier (not calibrated)
    max_dist_m: float = 6.0
    seed: int = 77


class PlumeCamera(UnderwaterCamera):
    """uwkit UnderwaterCamera + the plume's bubbles as world-anchored bright points (extra_radiance hook).

    record_truth: keep the image positions of the bubbles that reached the image in `last_bubbles` (grader side:
    u, v [px], dist [m], ids) - the same projection and depth test as uwkit's point imaging."""

    def __init__(self, m, water, rng, plume=None, bubbles: Optional[BubbleOptics] = None, record_truth: bool = False,
                 **kw):
        super().__init__(m, water, rng, **kw)
        self.plume = plume
        self.bub = bubbles or BubbleOptics()
        self.record_truth = bool(record_truth)
        self.t_extra = 0.0
        self.c_extra = 0.0
        self.n_extra = 0
        self.last_bubbles = None
        self.last_n_imaged = 0
        self.extra_radiance = self._bubble_hook

    def _bubble_hook(self, img, rng_r, p, C, lampgeo, t):
        if self.plume is None:
            return
        self.plume.advance_to(t)
        t0 = time.perf_counter()
        c0 = time.thread_time()
        self._bubbles(img, rng_r, p, C, lampgeo)
        self.t_extra += time.perf_counter() - t0
        self.c_extra += time.thread_time() - c0
        self.n_extra += 1

    def _bubbles(self, img, rng_r, p, C, lampgeo):
        bo = self.bub
        Pw = self.plume.P
        self.last_n_imaged = 0
        self.last_bubbles = None
        if Pw.shape[0] == 0:
            return
        d = Pw - p
        near = np.einsum("ij,ij->i", d, d) < bo.max_dist_m ** 2
        if not np.any(near):
            return
        Pn = Pw[near]
        ids = self.plume.ids[near]
        h1, _, _ = hash3(bo.seed, ids, 0, salt=5)
        dmin, dmax = bo.diam_mm
        diam = dmin * (dmax / dmin) ** h1 * 1e-3
        sigma = math.pi * (0.5 * diam) ** 2 * bo.gain
        self.last_n_imaged = self.render_points(img, rng_r, p, C, lampgeo, Pn, sigma,
                                                phase=(bo.phase_back, bo.phase_fwd), max_dist_m=bo.max_dist_m)
        if self.record_truth:
            self.last_bubbles = self._visible(rng_r, p, C, Pn, ids)

    def _visible(self, rng_r, p, C, Pw, ids) -> dict:
        """Image positions of the points that pass uwkit's point-imaging tests (in front, in the frame, not behind
        the scene surface)."""
        cfg = self.cfg
        H, W = cfg.height, cfg.width
        hit = np.isfinite(rng_r)
        r = np.where(hit, rng_r, 0.0)
        Pc = (Pw - p) @ C
        z = -Pc[:, 2]
        ok = (z > 0.05) & (Pw[:, 2] < 0.0)
        Pc, z, ids = Pc[ok], z[ok], ids[ok]
        dist = np.linalg.norm(Pc, axis=1)
        ok = dist < self.bub.max_dist_m
        Pc, z, ids, dist = Pc[ok], z[ok], ids[ok], dist[ok]
        if self.remap is None:
            u = W / 2.0 + self.f_render * Pc[:, 0] / z
            v = H / 2.0 - self.f_render * Pc[:, 1] / z
        else:
            rho_w = np.hypot(Pc[:, 0], Pc[:, 1]) / z
            beta = np.arctan(rho_w)
            alpha = np.arcsin(np.clip(np.sin(beta) * cfg.n_water, -1, 1))
            f_air = (W / 2.0) / math.tan(math.radians(cfg.hfov_air_deg) / 2.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                sc = np.where(rho_w > 1e-12, np.tan(alpha) / rho_w, cfg.n_water)
            u = W / 2.0 + f_air * sc * Pc[:, 0] / z
            v = H / 2.0 - f_air * sc * Pc[:, 1] / z
        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui, vi, u, v, dist, ids = ui[ok], vi[ok], u[ok], v[ok], dist[ok], ids[ok]
        vis = (~hit[vi, ui]) | (r[vi, ui] > dist)
        return {"u": u[vis], "v": v[vis], "dist": dist[vis], "ids": ids[vis]}
