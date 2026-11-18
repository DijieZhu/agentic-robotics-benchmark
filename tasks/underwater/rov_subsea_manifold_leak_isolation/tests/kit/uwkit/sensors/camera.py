"""Underwater colour camera: MuJoCo classic render (albedo + metric depth [+ segmentation]) followed by a numpy
underwater image-formation model.

Image formation (per channel k in R, G, B; all quantities linear, photometric units: lux, cd, cd/m^2):

  surface radiance      J   = albedo / pi * ( sum_lamps E_lamp + E_ambient )
     E_lamp  = I0 * P(theta_beam) * max(0, n.l) / d_L^2 * exp(-c_k d_L)           (lamp -> surface, inverse square)
     E_amb   = E_d(z) * [ f_dir(z) * max(0, n.s) * caustics + (1 - f_dir) (1 + n_z)/2 + R_u (1 - n_z)/2 ]
     E_d(z)  = E_0 * exp(-kd_k * depth)
  direct signal         D   = J * exp(-c_k r)
  forward scatter       F   = PSF * { J * [exp(-(c_k - gamma b_k) r) - exp(-c_k r)] }  (two-lobe Gaussian PSF)
  lamp backscatter      B_L = int_0^r beta_k(theta_s) I0 P / d_L^2 exp(-c_k (t + d_L)) dt      (single scattering)
                              computed once per (tilt, lamp geometry, water) as a cumulative table over 1/4-res ray
                              directions and looked up at each pixel's own range r (sharp silhouettes, no per-frame march).
                              Cache key: the tilt is quantised to bs_tilt_quantum_deg (0.5 deg) and the table is built
                              for the lamp geometry AT the quantised tilt (rounded to 0.1 mm), so a servo that settles
                              a little differently each time reuses the table; LRU cache of bs_cache_size tables
  ambient veiling light V   = S_0 / (c_k - kd_k mu) * (1 - exp(-(c_k - kd_k mu) r)),  S_0 = E_d(z_cam) beta_k(theta_sun)
  marine snow               world-anchored particles (hash of cell index and a seed drawn from the Generator), lit by
                              the lamps and the ambient field, depth-tested, defocused near the lens
  sensor                    auto exposure (time then gain) -> shot + read noise -> saturation -> vignetting -> sRGB 8 bit

Camera mount: site `cam_front` (+x = optical axis at zero tilt, +z = image up).  Tilt rotates about the site +y axis,
positive = looking up, slewed at a finite rate.  Port: 'dome' (entrance pupil at the dome centre, in-water FOV equals
in-air FOV) or 'flat' (refraction at a flat window: FOV shrinks and the image gets pincushion distortion; modelled
exactly for a thin window by remapping the rendered rectilinear image).

Lamps: sites `light_1..4` (+x = beam axis).  All lamps share one dimming level (one PWM channel on the reference
vehicle).

Not modelled (documented limitations): lamp shadows (no shadow test from the lamp positions), specular highlights,
shadowing of the lamp beams inside the backscatter integral, motion blur, rolling shutter, compression artefacts,
multiple scattering beyond the forward PSF, spatially varying water optics.

Known artefact (sub-pixel structures): the colour pass is multisample-blended but the depth pass keeps ONE surface per
pixel.  Along a bar thinner than about a pixel (a far grating, a thin rod) the pixels whose depth sample hits the bar are
lit at the bar's range, their neighbours (partly covered, depth = background) at the background's range, so the bar
shows as a chain of one-pixel bright dots that stay fixed in the world (a single frame cannot tell them from bubbles or
snow; a two-frame motion test can).  Normals come from one-sided depth differences on the side with the smaller jump.

Extension hooks (for task worlds that add their own optics without copying this module):
  extra_radiance(img, rng_r, p, C, lampgeo, t)   called at the end of the image formation (linear cd/m^2, in place)
  render_points(img, rng_r, p, C, lampgeo, points_w, cross_section_m2, ...)   lamp + ambient lit point scatterers
                                                 (depth-tested, defocused); marine snow uses it
GL safety: one mujoco.Renderer per camera; close_renderer() frees a Renderer without touching another one's GL objects
(mujoco 3.14's Renderer.close frees its MjrContext AFTER dropping its own GL context, i.e. in whatever context is current
-> a second Renderer that is closed or garbage-collected blanks the first).  Every capture checks that the camera's
framebuffer still exists and raises RuntimeError otherwise (CameraConfig.check_gl).
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence, Union

import numpy as np
import mujoco

from ._util import RateScheduler, hash_uniform, site_id, site_pose, SpecItem
from .water import WaterOptics, AmbientLight, hg_phase, rayleigh_phase, DARK

# --------------------------------------------------------------------------------------------------------- config


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    hfov_air_deg: float = 80.0            # horizontal FOV of the lens in air (square pixels)
    port: str = "dome"                    # 'dome' | 'flat'
    n_water: float = 1.34
    max_fps: float = 30.0
    tilt_min_deg: float = -90.0
    tilt_max_deg: float = 90.0
    tilt_rate_deg_s: float = 60.0
    exposure_min_s: float = 1.0 / 10000.0
    exposure_max_s: float = 1.0 / 30.0
    gain_max_db: float = 30.0
    ae_target: float = 0.18               # metered mean linear luminance after exposure
    ae_speed: float = 0.6                 # fraction of the log-exposure error corrected per frame
    near_m: float = 0.03
    port_clearance_m: float = 0.055       # lens -> outer port surface (dome radius ~0.052 m); no water inside
    far_m: float = 150.0
    hide_geomgroups: tuple = ()           # geom groups not rendered (e.g. the vehicle's own parts)
    bs_div: int = 4                       # backscatter table direction resolution: 1/bs_div of the image
    bs_steps: int = 32
    snow: bool = True
    snow_radius_m: float = 5.0
    snow_cell_m: float = 1.0
    # --- 0.3.0
    bs_tilt_quantum_deg: float = 0.5      # backscatter cache key: tilt quantum (0 = key on the exact geometry, as 0.2.0)
    bs_cache_size: int = 24               # backscatter tables kept (LRU); one 640x480 table = 7.6 MB, 320x240 = 1.9 MB
    show_geomgroups: tuple = ()           # extra geom groups to render (MuJoCo shows 0-2 by default; e.g. (4, 5))
    show_sitegroups: tuple = ()           # site groups drawn as spheres (default none; 0.2.0 drew groups 0-2)
    check_gl: bool = True                 # raise if this camera's framebuffer was freed by another Renderer


@dataclass
class CameraHidden:
    """Sensor internals (not in the public datasheet)."""
    full_well_e: float = 6000.0
    read_noise_e: float = 2.5
    eta_e_per_lux_s: float = 11368.0      # electrons per (cd/m^2 * s) per pixel: mid-grey 0.18 at 100 lux, 1/60 s, 0 dB
    vignetting: float = 0.25              # relative fall-off at the image corner
    focus_m: float = 1.5                  # focus distance (for defocus of near marine snow)
    defocus_px_m: float = 1.2             # blur-spot radius [px] per (1/m) of defocus at 640 px width
    snow_phase_back: float = 0.06         # phase function of a snow flake (1/sr), backward / forward
    snow_phase_fwd: float = 0.30
    wb_gains: tuple = (1.0, 1.0, 1.0)


@dataclass
class Lamp:
    """Dimmable LED lamp, intensity I(theta) = I0 cos^m(theta) inside a housing cut-off.

    The datasheet "beam angle 135 deg in water" is read as the field angle where the intensity has dropped to
    `field_level` (10 %) of the peak (an interpretation; the vendor does not define it).  That gives m = 2.40 and a
    half-intensity full width of 83 deg.  Reading it as FWHM instead (m = 0.72) lights the water right in front of the
    dome much more and roughly doubles the lamp backscatter at short range.
    """
    site: str
    flux_lm: float = 1500.0
    field_angle_deg: float = 135.0        # in water, full angle at `field_level` of peak intensity
    field_level: float = 0.10
    cutoff_deg: float = 85.0              # housing cut-off half angle
    rgb: tuple = (1.0, 1.0, 1.0)

    @property
    def exponent(self) -> float:
        return math.log(self.field_level) / math.log(math.cos(math.radians(self.field_angle_deg / 2.0)))

    @property
    def fwhm_deg(self) -> float:
        return 2.0 * math.degrees(math.acos(0.5 ** (1.0 / self.exponent)))

    @property
    def peak_cd(self) -> float:
        """I(theta) = I0 cos^m(theta) for theta < cutoff; flux = 2 pi I0 (1 - cos^(m+1)(cutoff)) / (m + 1)."""
        mexp = self.exponent
        frac = 1.0 - math.cos(math.radians(self.cutoff_deg)) ** (mexp + 1.0)
        return self.flux_lm * (mexp + 1.0) / (2.0 * math.pi * frac)


DEFAULT_LAMPS = [Lamp(f"light_{i}") for i in (1, 2, 3, 4)]


@dataclass
class CameraFrame:
    t: float
    valid: bool
    rgb: np.ndarray                       # (H, W, 3) uint8, sRGB
    exposure_s: float
    gain_db: float
    tilt_rad: float
    lamp_level: float
    # grader-side extras (None unless requested); never hand these to the agent
    range_m: Optional[np.ndarray] = None  # (H, W) float32, inf where no geometry
    seg_geom: Optional[np.ndarray] = None  # (H, W) int32 geom id, -1 background
    radiance: Optional[np.ndarray] = None  # (H, W, 3) float32 linear cd/m^2 before the sensor


def public_spec(cfg: CameraConfig) -> list[SpecItem]:
    hf = in_water_hfov_deg(cfg)
    vf = math.degrees(2 * math.atan(math.tan(math.radians(hf) / 2) * cfg.height / cfg.width))
    return [
        SpecItem("type", "colour CMOS camera behind a %s port, on a tilt servo" % cfg.port),
        SpecItem("resolution", f"{cfg.width} x {cfg.height}", "px"),
        SpecItem("field of view in water (H x V)", f"{hf:.1f} x {vf:.1f}", "deg",
                 "flat port: edge FOV, image has pincushion distortion" if cfg.port == "flat" else ""),
        SpecItem("max frame rate", cfg.max_fps, "Hz"),
        SpecItem("tilt range", f"{cfg.tilt_min_deg:.0f} .. {cfg.tilt_max_deg:.0f}", "deg", "positive = up"),
        SpecItem("tilt rate", cfg.tilt_rate_deg_s, "deg/s"),
        SpecItem("exposure", f"auto, {cfg.exposure_min_s*1e3:.2f} .. {cfg.exposure_max_s*1e3:.1f} ms, gain 0..{cfg.gain_max_db:.0f} dB"),
        SpecItem("lamps", "4 x 1500 lm, 135 deg beam in water, common dimming 0..1"),
    ]


def in_water_hfov_deg(cfg: CameraConfig) -> float:
    if cfg.port == "flat":
        return math.degrees(2 * math.asin(math.sin(math.radians(cfg.hfov_air_deg) / 2) / cfg.n_water))
    return cfg.hfov_air_deg


# --------------------------------------------------------------------------------------------------------- helpers

_SRGB_LUT = None


def srgb_to_linear_lut() -> np.ndarray:
    global _SRGB_LUT
    if _SRGB_LUT is None:
        x = np.arange(256, dtype=np.float64) / 255.0
        _SRGB_LUT = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)
    return _SRGB_LUT


def linear_to_srgb_u8(x: np.ndarray) -> np.ndarray:
    """x linear in [0, 1] -> sRGB uint8 via a 4096-entry table."""
    idx = np.clip(x * 4095.0 + 0.5, 0, 4095).astype(np.int32)
    return _SRGB_ENC[idx]


_t = np.arange(4096, dtype=np.float64) / 4095.0
_SRGB_ENC = (np.where(_t <= 0.0031308, 12.92 * _t, 1.055 * _t ** (1 / 2.4) - 0.055) * 255.0 + 0.5).astype(np.uint8)
del _t


def caustics(x: np.ndarray, y: np.ndarray, t: float, depth_m: np.ndarray, fade_m: float) -> np.ndarray:
    """Procedural, deterministic, tileable-ish caustic pattern (mean ~1), contrast fading with depth."""
    f = 0.0
    for k, (kx, ky, w) in enumerate(((1.7, 0.9, 1.1), (-1.1, 1.9, 0.8), (2.3, -1.4, 1.4))):
        f = f + np.sin(kx * x + ky * y + w * t + 0.7 * np.sin(0.8 * y - 1.3 * t + k))
    pat = np.exp(3.0 * (np.abs(f) / 3.0 - 0.6))
    pat = pat / 1.25  # long-run mean of the pattern is ~1.25
    fade = np.exp(-depth_m / fade_m)
    return (1.0 + fade * (pat - 1.0)).astype(np.float32)


def box_blur(img: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Running-mean blur (window 2r+1) along an axis with edge clamping, via cumulative sums."""
    if radius <= 0:
        return img
    a = np.moveaxis(img, axis, 0)
    w = 2 * radius + 1
    p = np.concatenate([np.repeat(a[:1], radius + 1, 0), a, np.repeat(a[-1:], radius, 0)], 0)
    cs = np.cumsum(p, axis=0, dtype=np.float32)
    out = (cs[w:] - cs[:-w]) * np.float32(1.0 / w)
    return np.moveaxis(out, 0, axis)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about a unit axis (right hand)."""
    x, y, z = (float(v) for v in axis)
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _upsample_linear(small: np.ndarray, f: int, H: int, W: int) -> np.ndarray:
    hs, ws = small.shape[:2]
    y = (np.arange(H) + 0.5) / f - 0.5
    y0 = np.clip(np.floor(y).astype(np.int64), 0, hs - 1)
    y1 = np.minimum(y0 + 1, hs - 1)
    wy = np.clip(y - y0, 0.0, 1.0).astype(np.float32)[:, None, None]
    rows = small[y0] * (1.0 - wy) + small[y1] * wy
    x = (np.arange(W) + 0.5) / f - 0.5
    x0 = np.clip(np.floor(x).astype(np.int64), 0, ws - 1)
    x1 = np.minimum(x0 + 1, ws - 1)
    wx = np.clip(x - x0, 0.0, 1.0).astype(np.float32)[None, :, None]
    return rows[:, x0] * (1.0 - wx) + rows[:, x1] * wx


def downsample(img: np.ndarray, f: int) -> np.ndarray:
    H, W = img.shape[:2]
    Hc, Wc = H // f * f, W // f * f
    return img[:Hc, :Wc].reshape(Hc // f, f, Wc // f, f, -1).mean(axis=(1, 3), dtype=np.float32)


def gauss_like_blur(img: np.ndarray, sigma_px: float, small: Optional[dict] = None) -> np.ndarray:
    """Approximate Gaussian blur of an (H, W, C) image: down-sample by f (sigma/f <= 2 px), two box passes per axis,
    bilinear up-sampling.  `small` caches down-sampled copies between calls on the same image."""
    if sigma_px < 0.4:
        return img
    H, W = img.shape[:2]
    f = 1
    while sigma_px / f > 2.0 and min(H, W) // (2 * f) >= 8:
        f *= 2
    if f > 1:
        key = f
        if small is not None and key in small:
            sm = small[key]
        else:
            sm = downsample(img, f)
            if small is not None:
                small[key] = sm
    else:
        sm = img
    s = sigma_px / f
    if f > 1:
        s = math.sqrt(max(s * s - (f * f - 1) / (12.0 * f * f) - 1.0 / 6.0, 0.09))   # pooling + bilinear add blur
    r = max(1, int(round(math.sqrt(3.0 * s * s + 0.25) - 0.5)))   # 2 passes of width 2r+1 -> var = (w^2-1)/6
    for ax in (0, 1):
        sm = box_blur(box_blur(sm, r, ax), r, ax)
    if f == 1:
        return sm
    return _upsample_linear(sm, f, H, W)


# --------------------------------------------------------------------------------------------------------- GL safety

def close_renderer(r: Optional[mujoco.Renderer]) -> None:
    """Free a mujoco.Renderer without corrupting any other Renderer of the process.

    mujoco 3.14's Renderer.close() drops its own GL context first and then frees its MjrContext in whatever GL context
    happens to be current -- usually another live Renderer's -- which deletes that Renderer's framebuffer and textures
    (its frames come out black or garbage, no error).  Here the Renderer's own context is made current before its
    MjrContext is freed.  Safe to call twice."""
    if r is None:
        return
    gl, mjr = getattr(r, "_gl_context", None), getattr(r, "_mjr_context", None)
    if gl is not None and mjr is not None:
        gl.make_current()
        mjr.free()
        r._mjr_context = None
        gl.free()
        r._gl_context = None
    else:
        r.close()


def patch_renderer_close() -> bool:
    """Opt-in, process-wide: make every mujoco.Renderer.close() (and therefore its __del__) free its GL objects in its
    own context (see close_renderer).  Returns True if the patch was installed (idempotent)."""
    cls = mujoco.Renderer
    if getattr(cls.close, "_uwkit_safe", False):
        return False
    orig = cls.close

    def close(self):
        gl, mjr = getattr(self, "_gl_context", None), getattr(self, "_mjr_context", None)
        if gl is not None and mjr is not None:
            gl.make_current()
            mjr.free()
            self._mjr_context = None
        orig(self)
    close._uwkit_safe = True
    cls.close = close
    return True


def _framebuffer_alive(r: mujoco.Renderer) -> Optional[bool]:
    """True / False: the Renderer's offscreen framebuffer object still exists in its own GL context; None: cannot tell
    (no GL context object, no PyOpenGL)."""
    gl, mjr = getattr(r, "_gl_context", None), getattr(r, "_mjr_context", None)
    if gl is None or mjr is None:
        return None
    try:
        from OpenGL import GL
    except Exception:                                   # pragma: no cover - PyOpenGL ships with mujoco's GL backends
        return None
    gl.make_current()
    try:
        for _ in range(16):                             # clear stale error flags (they would make the query raise)
            if GL.glGetError() == GL.GL_NO_ERROR:
                break
        fbo = int(getattr(mjr, "offFBO", 0))
        return True if fbo == 0 else bool(GL.glIsFramebuffer(fbo))
    except Exception:
        return False


# --------------------------------------------------------------------------------------------------------- camera


class UnderwaterCamera:
    """See module docstring.  One instance owns one mujoco.Renderer (one per process is the safe rule)."""

    def __init__(self, m: mujoco.MjModel, water: WaterOptics, rng: np.random.Generator,
                 ambient: AmbientLight = DARK, site: str = "cam_front",
                 lamps: Optional[Sequence[Lamp]] = None, config: Optional[CameraConfig] = None,
                 hidden: Optional[CameraHidden] = None, water_velocity=(0.0, 0.0, 0.0),
                 snow_sink_m_s: float = 0.001, renderer: Optional[mujoco.Renderer] = None,
                 tilt_joint: Optional[str] = "cam_tilt"):
        self.m = m
        self.cfg = config or CameraConfig()
        self.hid = hidden or CameraHidden()
        self.water = water
        self.ambient = ambient
        self.rng = rng
        self.snow_seed = int(rng.integers(0, 2 ** 62))      # world-anchored particle field seed
        self.sid = site_id(m, site)
        self.lamps = list(DEFAULT_LAMPS if lamps is None else lamps)
        self.lamp_sids = [site_id(m, L.site) for L in self.lamps]
        self.lamp_level = 1.0
        self.tilt = 0.0
        self.tilt_cmd = 0.0
        self._tilt_t = None
        # Tilt source: if the site rides on a body driven by a hinge joint named `tilt_joint` (the vehicle builder's
        # physical tilt servo), the site pose already contains the tilt -> no internal rotation; the reported tilt
        # (up-positive) is tilt_sign * qpos, tilt_sign from the hinge axis (about -y_site: +1; about +y_site: -1).
        # Otherwise the tilt is simulated here as a rate-limited rotation about the site +y axis.
        self.tilt_jid = -1
        self.tilt_sign = -1.0
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, tilt_joint) if tilt_joint else -1
        if jid >= 0:
            b = int(m.site_bodyid[self.sid])
            while b > 0:
                if int(m.jnt_bodyid[jid]) == b:
                    self.tilt_jid = jid
                    if b == int(m.site_bodyid[self.sid]):
                        ys = np.empty(9)
                        mujoco.mju_quat2Mat(ys, m.site_quat[self.sid])
                        y_site = ys.reshape(3, 3)[:, 1]
                        self.tilt_sign = -1.0 if float(m.jnt_axis[jid] @ y_site) > 0 else 1.0
                    break
                b = int(m.body_parentid[b])
        self.water_velocity = np.asarray(water_velocity, dtype=np.float64)
        self.snow_sink = float(snow_sink_m_s)
        self.sched = RateScheduler(self.cfg.max_fps)
        W, H = self.cfg.width, self.cfg.height
        self._owns_renderer = renderer is None
        self.renderer = renderer or mujoco.Renderer(m, H, W)
        self.opt = mujoco.MjvOption()
        for g in self.cfg.show_geomgroups:
            self.opt.geomgroup[int(g)] = 1
        for g in self.cfg.hide_geomgroups:
            self.opt.geomgroup[int(g)] = 0
        self.opt.sitegroup[:] = 0                       # sites are not physical: never in a sensor image
        for g in self.cfg.show_sitegroups:
            self.opt.sitegroup[int(g)] = 1
        self._free = mujoco.MjvCamera()
        self._free.type = mujoco.mjtCamera.mjCAMERA_FREE
        mujoco.mjv_defaultFreeCamera(m, self._free)
        self._setup_projection()
        self._bs_cache: "OrderedDict" = OrderedDict()
        self.bs_stats = dict(hits=0, builds=0)
        self._log_ev = None     # AE state: log(exposure_time * gain)
        self.last_exposure = (self.cfg.exposure_max_s, 0.0)
        # optional hook: extra_radiance(img, rng_r, p, C, lampgeo, t) adds task-specific light (linear cd/m^2, in place)
        self.extra_radiance = None
        self._tilt_body = int(m.jnt_bodyid[self.tilt_jid]) if self.tilt_jid >= 0 else -1
        self._lamp_on_tilt = [self._rides_tilt(int(m.site_bodyid[s])) for s in self.lamp_sids]

    def _rides_tilt(self, body: int) -> bool:
        b = body
        while b > 0:
            if b == self._tilt_body:
                return True
            b = int(self.m.body_parentid[b])
        return False

    # ------------------------------------------------------------------ lifecycle
    def reseed(self, rng: np.random.Generator) -> None:
        """New noise stream and a new marine-snow field (drawn from rng exactly as in the constructor)."""
        self.rng = rng
        self.snow_seed = int(rng.integers(0, 2 ** 62))

    def close(self) -> None:
        """Free the renderer if this camera created it (safe for other Renderers of the process)."""
        if getattr(self, "_owns_renderer", False) and getattr(self, "renderer", None) is not None:
            close_renderer(self.renderer)
            self.renderer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ geometry
    def _setup_projection(self):
        cfg = self.cfg
        W, H = cfg.width, cfg.height
        u = (np.arange(W, dtype=np.float64) + 0.5)
        v = (np.arange(H, dtype=np.float64) + 0.5)
        uu, vv = np.meshgrid(u, v)
        if cfg.port == "dome":
            self.f_render = (W / 2.0) / math.tan(math.radians(cfg.hfov_air_deg) / 2.0)
            self.render_vfov = 2 * math.atan((H / 2.0) / self.f_render)
            x = (uu - W / 2.0) / self.f_render
            y = -(vv - H / 2.0) / self.f_render
            d = np.stack([x, y, -np.ones_like(x)], -1)
            self.range_factor = np.linalg.norm(d, axis=-1).astype(np.float32)   # range = depth * factor
            self.dirs = (d / self.range_factor[..., None]).astype(np.float32)   # GL camera frame (x right, y up, -z fwd)
            self.remap = None
        else:
            # flat port: air pinhole with f_air; water direction by Snell at a thin window
            f_air = (W / 2.0) / math.tan(math.radians(cfg.hfov_air_deg) / 2.0)
            xa = (uu - W / 2.0) / f_air
            ya = -(vv - H / 2.0) / f_air
            rho = np.hypot(xa, ya)
            alpha = np.arctan(rho)
            beta = np.arcsin(np.sin(alpha) / cfg.n_water)
            with np.errstate(invalid="ignore", divide="ignore"):
                s = np.where(rho > 1e-12, np.tan(beta) / rho, 1.0 / cfg.n_water)
            xw, yw = xa * s, ya * s                       # rectilinear coords of the water ray
            d = np.stack([xw, yw, -np.ones_like(xw)], -1)
            nrm = np.linalg.norm(d, axis=-1)
            self.dirs = (d / nrm[..., None]).astype(np.float32)
            # render a rectilinear image just covering the corner water ray
            xmax = float(np.abs(xw).max()) * 1.01
            self.f_render = (W / 2.0) / xmax
            self.render_vfov = 2 * math.atan((H / 2.0) / self.f_render)
            ur = np.clip(np.floor(xw * self.f_render + W / 2.0), 0, W - 1).astype(np.int64)
            vr = np.clip(np.floor(-yw * self.f_render + H / 2.0), 0, H - 1).astype(np.int64)
            self.remap = (vr * W + ur).reshape(-1)
            # render-pixel range factor
            x = (uu - W / 2.0) / self.f_render
            y = -(vv - H / 2.0) / self.f_render
            self.range_factor = np.sqrt(x * x + y * y + 1.0).astype(np.float32)
        self._dir_planes = tuple(np.ascontiguousarray(self.dirs[..., k]) for k in range(3))
        self.px_per_rad = W / math.radians(in_water_hfov_deg(cfg))
        # vignetting map
        cx = (uu - W / 2.0) / (W / 2.0)
        cy = (vv - H / 2.0) / (W / 2.0)
        r2 = (cx * cx + cy * cy) / (1.0 + (H / W) ** 2)
        self.vignette = (1.0 - self.hid.vignetting * r2).astype(np.float32)[..., None]
        # coarse direction grid for the backscatter table
        dv = self.cfg.bs_div
        self.bs_dirs = self.dirs[dv // 2::dv, dv // 2::dv].reshape(-1, 3).astype(np.float64)
        Hc, Wc = self.dirs[dv // 2::dv, dv // 2::dv].shape[:2]
        jj = np.minimum(np.arange(H) // dv, Hc - 1)[:, None] * Wc + np.minimum(np.arange(W) // dv, Wc - 1)[None, :]
        self.bs_index = jj.astype(np.int64)

    # ------------------------------------------------------------------ controls
    def set_tilt(self, tilt_rad: float) -> None:
        lo, hi = math.radians(self.cfg.tilt_min_deg), math.radians(self.cfg.tilt_max_deg)
        self.tilt_cmd = min(max(float(tilt_rad), lo), hi)

    def set_tilt_now(self, tilt_rad: float) -> None:
        """Kinematic tilt (no slew): command and current tilt jump to tilt_rad (clamped).  Only for the internal tilt
        model; when the site rides a physical tilt joint, move the joint (UnderwaterSim.set_camera_tilt or qpos)."""
        if self.tilt_jid >= 0:
            raise RuntimeError("the camera tilt is a physical joint of the model; set the joint (UnderwaterSim."
                               "set_camera_tilt / data.qpos) instead of set_tilt_now")
        self.set_tilt(tilt_rad)
        self.tilt = self.tilt_cmd
        self._tilt_t = None

    def set_lamp_level(self, level: float) -> None:
        self.lamp_level = min(max(float(level), 0.0), 1.0)

    def set_water(self, water: WaterOptics) -> None:
        self.water = water

    def set_ambient(self, ambient: AmbientLight) -> None:
        self.ambient = ambient

    def reset_auto_exposure(self) -> None:
        self._log_ev = None

    def tilt_ctrl_value(self) -> float:
        """Joint-mode helper: the actuator target for the vehicle's tilt joint that realises tilt_cmd (up-positive)."""
        return self.tilt_sign * self.tilt_cmd

    def _advance_tilt(self, t: float, d: Optional[mujoco.MjData] = None) -> None:
        if self.tilt_jid >= 0:
            if d is not None:
                self.tilt = self.tilt_sign * float(d.qpos[self.m.jnt_qposadr[self.tilt_jid]])
            return
        if self._tilt_t is None:
            self._tilt_t = t
        dt = max(0.0, t - self._tilt_t)
        self._tilt_t = t
        step = math.radians(self.cfg.tilt_rate_deg_s) * dt
        e = self.tilt_cmd - self.tilt
        self.tilt += max(-step, min(step, e))

    def camera_pose(self, d: mujoco.MjData, tilt: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (position, C) where C's columns are the GL camera axes (right, up, back) in W.  tilt: override of
        the internal tilt (internal tilt model only)."""
        p, Rs = site_pose(d, self.sid)
        th = 0.0 if self.tilt_jid >= 0 else (self.tilt if tilt is None else float(tilt))
        fwd = Rs @ np.array([math.cos(th), 0.0, math.sin(th)])
        up = Rs @ np.array([-math.sin(th), 0.0, math.cos(th)])
        right = np.cross(fwd, up)
        return p, np.stack([right, up, -fwd], axis=1)

    # ------------------------------------------------------------------ render passes
    def _render_buffers(self, d: mujoco.MjData, p: np.ndarray, C: np.ndarray, want_seg: bool):
        m, r = self.m, self.renderer
        if r is None:
            raise RuntimeError("camera closed")
        if self.cfg.check_gl and _framebuffer_alive(r) is False:
            raise RuntimeError("this camera's OpenGL framebuffer no longer exists: another mujoco.Renderer of the process "
                               "was closed or garbage-collected while this one was alive (mujoco 3.14 frees it in the "
                               "wrong GL context). Free extra renderers with uwkit.sensors.camera.close_renderer(r), call "
                               "patch_renderer_close() once at start-up, or keep one Renderer per process.")
        hl = m.vis.headlight
        saved = (hl.active, hl.ambient.copy(), hl.diffuse.copy(), hl.specular.copy(), m.vis.map.znear, m.vis.map.zfar)
        try:
            hl.active = 1
            hl.ambient[:] = 1.0
            hl.diffuse[:] = 0.0
            hl.specular[:] = 0.0
            ext = m.stat.extent
            m.vis.map.znear = self.cfg.near_m / ext
            m.vis.map.zfar = self.cfg.far_m / ext
            r.update_scene(d, camera=self._free, scene_option=self.opt)
            scn = r.scene
            for f in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_SKYBOX, mujoco.mjtRndFlag.mjRND_FOG,
                      mujoco.mjtRndFlag.mjRND_HAZE, mujoco.mjtRndFlag.mjRND_REFLECTION):
                scn.flags[f] = 0
            scn.nlight = 1
            fwd = -C[:, 2]
            up = C[:, 1]
            for i in range(2):
                gc = scn.camera[i]
                gc.pos[:] = p
                gc.forward[:] = fwd
                gc.up[:] = up
                n = gc.frustum_near
                half = n * math.tan(self.render_vfov / 2.0)
                gc.frustum_center = 0.0
                gc.frustum_bottom = -half
                gc.frustum_top = half
                gc.frustum_width = 0.0
            rgb = r.render()
            r.enable_depth_rendering()
            depth = r.render()
            r.disable_depth_rendering()
            seg = None
            if want_seg:
                r.enable_segmentation_rendering()
                sg = r.render()
                r.disable_segmentation_rendering()
                seg = np.where(sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), sg[..., 0], -1).astype(np.int32)
        finally:
            hl.active, hl.ambient[:], hl.diffuse[:], hl.specular[:] = saved[0], saved[1], saved[2], saved[3]
            m.vis.map.znear, m.vis.map.zfar = saved[4], saved[5]
        return rgb, depth, seg

    # ------------------------------------------------------------------ backscatter table
    def _lamp_geometry(self, d, p, C):
        Ls, As, I0s, rgbs, mexp, ccut = [], [], [], [], [], []
        for L, sid in zip(self.lamps, self.lamp_sids):
            lp, lR = site_pose(d, sid)
            Ls.append(C.T @ (lp - p))
            As.append(C.T @ lR[:, 0])
            I0s.append(L.peak_cd)
            rgbs.append(np.asarray(L.rgb, dtype=np.float64))
            mexp.append(L.exponent)
            ccut.append(math.cos(math.radians(L.cutoff_deg)))
        return Ls, As, I0s, rgbs, mexp, ccut

    @staticmethod
    def _pattern(cosb, mexp, ccut):
        soft = 0.05
        cb = np.clip(cosb, 1e-6, 1.0)
        return np.exp(mexp * np.log(cb)) * np.clip((cosb - ccut) / soft, 0.0, 1.0)

    def bs_lamp_geometry(self, d, p, C, lampgeo):
        """Lamp geometry the backscatter table is built for: the geometry at the tilt quantised to
        cfg.bs_tilt_quantum_deg (lamps that ride the tilt body keep theirs), rounded to 0.1 mm / 1e-4.  Pure function
        of the current state, so the table does not depend on which frame first filled the cache.
        Returns (lampgeo for the table, key part) - with bs_tilt_quantum_deg = 0 the exact geometry (0.2.0 behaviour)."""
        Ls, As = lampgeo[0], lampgeo[1]
        q = float(self.cfg.bs_tilt_quantum_deg or 0.0)
        if q <= 0.0:
            return lampgeo, tuple(np.round(np.concatenate([np.ravel(Ls), np.ravel(As)]), 4))
        qr = math.radians(q)
        dth = round(self.tilt / qr) * qr - self.tilt
        if dth != 0.0:
            if self.tilt_jid >= 0:
                # rotate the camera about the tilt hinge (world axis through its anchor) to the quantised angle
                ax = np.asarray(d.xaxis[self.tilt_jid], float)
                an = np.asarray(d.xanchor[self.tilt_jid], float)
                R = _axis_angle(ax / np.linalg.norm(ax), dth * self.tilt_sign)
                p2, C2 = an + R @ (p - an), R @ C
            else:
                p2, C2 = self.camera_pose(d, tilt=self.tilt + dth)
            Ls, As = list(Ls), list(As)
            for i, sid in enumerate(self.lamp_sids):
                if self._lamp_on_tilt[i]:
                    continue                                  # rides with the camera: tilt-invariant
                lp, lR = site_pose(d, sid)
                Ls[i] = C2.T @ (lp - p2)
                As[i] = C2.T @ lR[:, 0]
        Lr = [np.round(np.asarray(x, float), 4) for x in Ls]
        Ar = [np.round(np.asarray(x, float), 4) for x in As]
        geo = (Lr, Ar) + tuple(lampgeo[2:])
        return geo, tuple(np.concatenate([np.ravel(Lr), np.ravel(Ar)]))

    def _bs_table(self, lampgeo):
        """Backscatter table for a lamp geometry (camera frame); key = the geometry rounded to 1e-4 (+ water, lamps,
        image geometry).  _image_formation passes bs_lamp_geometry(...)[0], which is already rounded, so the key and
        the geometry the table is built from are the same numbers."""
        Ls, As, I0s, rgbs, mexp, ccut = lampgeo
        geokey = tuple(np.round(np.concatenate([np.ravel(Ls), np.ravel(As)]), 4))
        key = (self.water, geokey, tuple(I0s), self.cfg.port, self.cfg.width, self.cfg.height, self.cfg.hfov_air_deg)
        tab = self._bs_cache.get(key)
        if tab is not None:
            self._bs_cache.move_to_end(key)
            self.bs_stats["hits"] += 1
            return tab
        self.bs_stats["builds"] += 1
        w = self.water
        c = w.c
        bp, bw = w.b_particle, w.b_water
        K = self.cfg.bs_steps
        t0 = self.cfg.port_clearance_m          # water starts outside the dome / window
        tmax = min(self.cfg.far_m, 8.0 / float(np.min(c)))
        edges = t0 * (tmax / t0) ** (np.arange(K + 1) / K)
        tm = 0.5 * (edges[1:] + edges[:-1])
        dt = np.diff(edges)
        D = self.bs_dirs                                   # (J, 3)
        X = D[:, None, :] * tm[None, :, None]              # (J, K, 3)
        acc = np.zeros((D.shape[0], K, 3))
        att_t = np.exp(-np.outer(tm, c))                   # (K, 3)
        for Lc, Ac, I0, rgb, me, cc in zip(Ls, As, I0s, rgbs, mexp, ccut):
            v = X - Lc[None, None, :]
            dl = np.sqrt(np.einsum("jki,jki->jk", v, v))
            u = v / dl[..., None]
            cosb = np.einsum("jki,i->jk", u, Ac)
            pat = self._pattern(cosb, me, cc)
            cos_s = -np.einsum("jki,ji->jk", u, D)          # scattering angle: propagation u vs. towards camera -D
            beta = (hg_phase(cos_s, w.g)[..., None] * bp[None, None, :]
                    + rayleigh_phase(cos_s)[..., None] * bw[None, None, :])
            g = (I0 * pat / np.maximum(dl, 0.02) ** 2 * dt[None, :])[..., None]
            acc += g * beta * np.exp(-dl[..., None] * c[None, None, :]) * att_t[None, :, :] * rgb[None, None, :]
        tab = np.zeros((D.shape[0], K + 1, 3), np.float32)
        tab[:, 1:] = np.cumsum(acc, axis=1)
        out = (tab.reshape(-1, 3), math.log(t0), math.log(tmax / t0) / K, K)
        while len(self._bs_cache) >= max(1, int(self.cfg.bs_cache_size)):
            self._bs_cache.popitem(last=False)                # least recently used
        self._bs_cache[key] = out
        return out

    # ------------------------------------------------------------------ main
    def due(self, t: float) -> bool:
        return self.sched.due(t)

    def capture(self, d: mujoco.MjData, t: float, want_extras: bool = False, force: bool = False) -> Optional[CameraFrame]:
        """Render one frame at sim time t (None if called faster than max_fps unless force)."""
        if not force and not self.sched.due(t):
            return None
        self.sched.fire(t)
        self._advance_tilt(t, d)
        p, C = self.camera_pose(d)
        rgb8, depth, seg = self._render_buffers(d, p, C, want_extras)
        cfg = self.cfg
        H, W = cfg.height, cfg.width
        lut = srgb_to_linear_lut()
        albedo = lut[rgb8]                                  # (H, W, 3) float32
        rng_r = depth * self.range_factor
        far_hit = depth >= 0.995 * cfg.far_m
        if self.remap is not None:
            albedo = albedo.reshape(-1, 3)[self.remap].reshape(H, W, 3)
            rng_r = rng_r.reshape(-1)[self.remap].reshape(H, W)
            far_hit = far_hit.reshape(-1)[self.remap].reshape(H, W)
            if seg is not None:
                seg = seg.reshape(-1)[self.remap].reshape(H, W)
        rng_r = np.where(far_hit, np.float32(np.inf), rng_r).astype(np.float32)
        L = self._image_formation(albedo, rng_r, p, C, d, t)
        valid = bool(p[2] < 0.0)
        img8, exp_s, gain_db = self._sensor(L)
        return CameraFrame(t=t, valid=valid, rgb=img8, exposure_s=exp_s, gain_db=gain_db, tilt_rad=self.tilt,
                           lamp_level=self.lamp_level,
                           range_m=rng_r if want_extras else None, seg_geom=seg if want_extras else None,
                           radiance=L if want_extras else None)

    def _image_formation(self, albedo, rng_r, p, C, d, t):
        """Linear radiance image (H, W, 3) float32 [cd/m^2].  Per-pixel work is done on (H, W) planes."""
        w = self.water
        f32 = np.float32
        c = w.c.astype(np.float32)
        a_fwd = (w.c - w.forward_frac * w.b_).astype(np.float32)
        H, W = rng_r.shape
        hit = np.isfinite(rng_r)
        r = np.where(hit, rng_r, f32(0.0))
        dx_, dy_, dz_ = self._dir_planes
        Px, Py, Pz = dx_ * r, dy_ * r, dz_ * r                       # camera frame points
        # normals from depth: one-sided differences, choosing the side with the smaller range jump
        def diffs(A, axis):
            f = np.zeros_like(A)
            if axis == 1:
                f[:, :-1] = A[:, 1:] - A[:, :-1]
            else:
                f[:-1] = A[1:] - A[:-1]
            b = np.zeros_like(A)
            if axis == 1:
                b[:, 1:] = f[:, :-1]
            else:
                b[1:] = f[:-1]
            return f, b
        rf, rb = diffs(r, 1)
        uf, ub = diffs(r, 0)
        selx = np.abs(rf) <= np.abs(rb)
        selx[:, -1] = False
        selx[:, 0] = True
        sely = np.abs(uf) <= np.abs(ub)
        sely[-1] = False
        sely[0] = True
        ax_ = []
        ay_ = []
        for A in (Px, Py, Pz):
            fx, bx = diffs(A, 1)
            fy, by = diffs(A, 0)
            ax_.append(np.where(selx, fx, bx))
            ay_.append(np.where(sely, fy, by))
        Nx = ax_[1] * ay_[2] - ax_[2] * ay_[1]
        Ny = ax_[2] * ay_[0] - ax_[0] * ay_[2]
        Nz = ax_[0] * ay_[1] - ax_[1] * ay_[0]
        inv = f32(1.0) / (np.sqrt(Nx * Nx + Ny * Ny + Nz * Nz) + f32(1e-12))
        sgn = np.where(Nx * dx_ + Ny * dy_ + Nz * dz_ > 0, f32(-1.0), f32(1.0)) * inv
        Nx *= sgn; Ny *= sgn; Nz *= sgn
        E = [np.zeros((H, W), np.float32) for _ in range(3)]
        lampgeo = self._lamp_geometry(d, p, C)
        lvl = self.lamp_level
        if lvl > 0:
            for Lc, Ac, I0, rgb, me, cc in zip(*lampgeo):
                vx = f32(Lc[0]) - Px
                vy = f32(Lc[1]) - Py
                vz = f32(Lc[2]) - Pz
                dl2 = vx * vx + vy * vy + vz * vz
                dl = np.sqrt(dl2)
                idl = f32(1.0) / np.maximum(dl, f32(1e-3))
                cos_s = np.clip((Nx * vx + Ny * vy + Nz * vz) * idl, 0.0, 1.0)
                cosb = -(vx * f32(Ac[0]) + vy * f32(Ac[1]) + vz * f32(Ac[2])) * idl
                pat = self._pattern(cosb, me, cc).astype(np.float32)
                g = f32(I0 * lvl) * pat * cos_s / np.maximum(dl2, f32(1e-4))
                for k in range(3):
                    E[k] += g * np.exp(-c[k] * dl) * f32(rgb[k])
        amb = self.ambient
        V = None
        if amb.active:
            kd = w.kd_.astype(np.float32)
            Cz = C[2].astype(np.float32)
            zw = f32(p[2]) + Px * Cz[0] + Py * Cz[1] + Pz * Cz[2]
            dep = np.maximum(-zw, f32(0.0))
            nz = Nx * Cz[0] + Ny * Cz[1] + Nz * Cz[2]
            fdir = f32(amb.direct_frac_surface) * np.exp(-dep / f32(amb.direct_scale_m))
            sun = np.asarray(amb.sun_dir_world, np.float64)
            sun = sun / np.linalg.norm(sun)
            sc = (C.T @ sun).astype(np.float32)
            ns = np.clip(-(Nx * sc[0] + Ny * sc[1] + Nz * sc[2]), 0.0, 1.0)
            if amb.caustics and amb.direct_frac_surface > 0:
                Cx, Cy = C[0].astype(np.float32), C[1].astype(np.float32)
                xw = f32(p[0]) + Px * Cx[0] + Py * Cx[1] + Pz * Cx[2]
                yw = f32(p[1]) + Px * Cy[0] + Py * Cy[1] + Pz * Cy[2]
                ns = ns * caustics(xw, yw, t, dep, amb.caustic_fade_m)
            fac = fdir * ns + (f32(1.0) - fdir) * f32(0.5) * (f32(1.0) + nz) \
                + f32(amb.upwelling_ratio * 0.5) * (f32(1.0) - nz)
            col = np.asarray(amb.color, np.float32)
            for k in range(3):
                E[k] += f32(amb.surface_lux * col[k]) * np.exp(-dep * kd[k]) * fac
            # ambient veiling light along each line of sight
            mu = self._view_z(C)
            Ed0 = np.float32(amb.surface_lux) * np.exp(-np.float32(max(-p[2], 0.0)) * kd) * col
            cos_t = -(self._view_dot(C, sun))
            ph_p = hg_phase(cos_t, w.g_ambient).astype(np.float32)
            ph_w = rayleigh_phase(cos_t).astype(np.float32)
            bp, bw = w.b_particle.astype(np.float32), w.b_water.astype(np.float32)
            V = []
            for k in range(3):
                S0 = Ed0[k] * (ph_p * bp[k] + ph_w * bw[k])
                ce = np.maximum(c[k] - kd[k] * mu, f32(1e-3))
                V.append(S0 / ce * np.where(hit, f32(1.0) - np.exp(-ce * r), f32(1.0)))
        img = np.empty((H, W, 3), np.float32)
        Fsrc = np.empty((H, W, 3), np.float32)
        inv_pi = f32(1.0 / math.pi)
        for k in range(3):
            J = albedo[..., k] * E[k] * inv_pi
            T = np.where(hit, np.exp(-c[k] * r), f32(0.0))
            img[..., k] = J * T
            Fsrc[..., k] = J * (np.where(hit, np.exp(-a_fwd[k] * r), f32(0.0)) - T)
        if w.forward_frac > 0 and float(np.max(w.b_)) > 1e-6:
            sig_n = math.radians(w.psf_narrow_deg) * self.px_per_rad
            sig_w = math.radians(w.psf_wide_deg) * self.px_per_rad
            cache: dict = {}
            img += f32(w.psf_narrow_weight) * gauss_like_blur(Fsrc, sig_n, cache)
            img += f32(1.0 - w.psf_narrow_weight) * gauss_like_blur(Fsrc, sig_w, cache)
        if lvl > 0:
            tab, lt0, dlog, K = self._bs_table(self.bs_lamp_geometry(d, p, C, lampgeo)[0])
            rr = np.where(hit, r, f32(1e9))
            s_ = (np.log(np.maximum(rr, f32(1e-6))) - f32(lt0)) * f32(1.0 / dlog)
            s_ = np.clip(s_, 0.0, K - 1e-4)
            k0 = s_.astype(np.int64)
            fr = (s_ - k0).astype(np.float32)[..., None]
            base = self.bs_index * (K + 1) + k0
            img += f32(lvl) * (tab[base] * (f32(1.0) - fr) + tab[base + 1] * fr)
        if V is not None:
            for k in range(3):
                img[..., k] += V[k]
        if self.cfg.snow and w.snow_density > 0:
            self._marine_snow(img, r, hit, p, C, lampgeo, t)
        if self.extra_radiance is not None:
            self.extra_radiance(img, rng_r, p, C, lampgeo, t)
        return img

    def _view_z(self, C):
        Cz = C[2].astype(np.float32)
        dx_, dy_, dz_ = self._dir_planes
        return dx_ * Cz[0] + dy_ * Cz[1] + dz_ * Cz[2]

    def _view_dot(self, C, vec_world):
        vc = (C.T @ np.asarray(vec_world, float)).astype(np.float32)
        dx_, dy_, dz_ = self._dir_planes
        return dx_ * vc[0] + dy_ * vc[1] + dz_ * vc[2]

    def _marine_snow(self, img, r, hit, p, C, lampgeo, t):
        w = self.water
        c = w.c
        R = min(self.cfg.snow_radius_m, 4.0 / float(np.min(c)))
        cell = self.cfg.snow_cell_m
        n_per = int(round(w.snow_density * cell ** 3))
        if n_per <= 0:
            return
        drift = self.water_velocity * t + np.array([0.0, 0.0, -self.snow_sink * t])
        q = p - drift
        lo = np.floor((q - R) / cell).astype(np.int64)
        hi = np.floor((q + R) / cell).astype(np.int64)
        gi = [np.arange(lo[k], hi[k] + 1) for k in range(3)]
        I, Jc, Kc = np.meshgrid(*gi, indexing="ij")
        I, Jc, Kc = I.reshape(-1, 1), Jc.reshape(-1, 1), Kc.reshape(-1, 1)
        n = np.arange(n_per).reshape(1, -1)
        s = self.snow_seed
        ux = hash_uniform(s, I, Jc, Kc, 4 * n)
        uy = hash_uniform(s, I, Jc, Kc, 4 * n + 1)
        uz = hash_uniform(s, I, Jc, Kc, 4 * n + 2)
        us = hash_uniform(s, I, Jc, Kc, 4 * n + 3)
        Pw = np.stack([(I + ux) * cell, (Jc + uy) * cell, (Kc + uz) * cell], -1).reshape(-1, 3) + drift
        size = us.reshape(-1)
        dmin, dmax = w.snow_diam_mm

        def sigma_of(sz):
            diam = dmin * (dmax / dmin) ** sz * 1e-3
            return math.pi * (0.5 * diam) ** 2
        self._image_points(img, r, hit, p, C, lampgeo, Pw, size, sigma_of,
                           (self.hid.snow_phase_back, self.hid.snow_phase_fwd), None, R)

    def render_points(self, img: np.ndarray, rng_r: np.ndarray, p: np.ndarray, C: np.ndarray, lampgeo,
                      points_w: np.ndarray, cross_section_m2, phase=(0.06, 0.30), ambient_phase: Optional[float] = None,
                      max_dist_m: Optional[float] = None) -> int:
        """Add lamp- and ambient-lit point scatterers (bubbles, particles, ...) to a linear radiance image, the way
        marine snow is imaged: irradiance from every lamp (beam pattern, inverse square, attenuation lamp -> point)
        and the ambient field, times the cross-section and a phase function, attenuated to the camera, depth-tested
        against rng_r (per-pixel range, inf = open water), defocus-blurred near the lens.  For use inside an
        extra_radiance hook: render_points(img, rng_r, p, C, lampgeo, ...) with the hook's own arguments.

        points_w: (N, 3) world positions; cross_section_m2: scalar or (N,) [m^2]; phase: (back, forward) [1/sr] of the
        snow form back + (fwd - back) ((1 + cos)/2)^2, or a callable f(cos_scatter) -> (N,) [1/sr];
        ambient_phase: phase factor for the ambient field (default: mean of back / forward, or 1 / (4 pi) for a
        callable); max_dist_m: ignore points farther than this.  Returns the number of points that reached the image."""
        Pw = np.asarray(points_w, float).reshape(-1, 3)
        sig = np.broadcast_to(np.asarray(cross_section_m2, float), (Pw.shape[0],)).copy()
        r = np.where(np.isfinite(rng_r), rng_r, np.float32(0.0))
        return self._image_points(img, r, np.isfinite(rng_r), p, C, lampgeo, Pw, sig, None, phase, ambient_phase,
                                  max_dist_m)

    def _image_points(self, img, r, hit, p, C, lampgeo, Pw, carry, sigma_of, phase, ambient_phase, max_dist):
        """Shared point imaging (marine snow, render_points).  carry: per-point array filtered with the points;
        sigma_of(carry) -> cross-sections (None: carry already is the cross-section)."""
        w = self.water
        c = w.c
        Pc = (Pw - p) @ C                                   # camera frame (x right, y up, z back)
        z = -Pc[:, 2]
        ok = (z > 0.05) & (Pw[:, 2] < 0.0)
        Pc, z, carry = Pc[ok], z[ok], carry[ok]
        dist = np.linalg.norm(Pc, axis=1)
        if max_dist is not None:
            ok = dist < max_dist
            Pc, z, carry, dist = Pc[ok], z[ok], carry[ok], dist[ok]
        if not len(z):
            return 0
        cfg = self.cfg
        H, W = cfg.height, cfg.width
        dirc = Pc / dist[:, None]
        if self.remap is None:
            u = (cfg.width / 2.0 + self.f_render * Pc[:, 0] / z)
            v = (cfg.height / 2.0 - self.f_render * Pc[:, 1] / z)
        else:  # flat port: invert Snell for the projection
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
        ui, vi, Pc, dist, carry, dirc = ui[ok], vi[ok], Pc[ok], dist[ok], carry[ok], dirc[ok]
        vis = (~hit[vi, ui]) | (r[vi, ui] > dist)
        ui, vi, Pc, dist, carry, dirc = ui[vis], vi[vis], Pc[vis], dist[vis], carry[vis], dirc[vis]
        if not len(ui):
            return 0
        sigma = carry if sigma_of is None else sigma_of(carry)
        if callable(phase):
            ph_fn = phase
            ph_amb = 1.0 / (4.0 * math.pi) if ambient_phase is None else float(ambient_phase)
        else:
            pb, pf = float(phase[0]), float(phase[1])
            ph_fn = None
            ph_amb = 0.5 * (pb + pf) if ambient_phase is None else float(ambient_phase)
        Ip = np.zeros((len(ui), 3))
        hid = self.hid
        if self.lamp_level > 0:
            for Lc, Ac, I0, rgb, me, cc in zip(*lampgeo):
                v_ = Pc - Lc
                dl = np.linalg.norm(v_, axis=1)
                lv = v_ / dl[:, None]
                pat = self._pattern(lv @ Ac, me, cc)
                cos_sc = -(lv * dirc).sum(1)
                ph = ph_fn(cos_sc) if ph_fn is not None else pb + (pf - pb) * (0.5 * (1 + cos_sc)) ** 2
                Ip += (I0 * self.lamp_level * pat / np.maximum(dl, 0.02) ** 2 * ph)[:, None] * np.exp(-np.outer(dl, c)) * rgb
        amb = self.ambient
        if amb.active:
            zw = p[2] + Pc @ C[2]
            Ed = amb.surface_lux * np.exp(-np.outer(np.maximum(-zw, 0), w.kd_)) * np.asarray(amb.color)
            Ip += Ed * ph_amb
        Ip *= sigma[:, None]
        Lpx = Ip * np.exp(-np.outer(dist, c)) * (self.f_render ** 2) / (dist ** 2)[:, None]
        rad = hid.defocus_px_m * (cfg.width / 640.0) * np.abs(1.0 / dist - 1.0 / hid.focus_m)
        rad = np.maximum(rad, 0.5)
        flat = img.reshape(-1, 3)
        for lo_r, hi_r, k in ((0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 3.5, 2), (3.5, 1e9, 3)):
            sel = (rad >= lo_r) & (rad < hi_r)
            if not np.any(sel):
                continue
            offs = np.arange(-k, k + 1)
            if k == 0:
                ker = np.ones((1, 1))
            else:   # soft disc (defocus blur spot)
                yy_, xx_ = np.meshgrid(offs, offs, indexing="ij")
                ker = np.exp(-0.5 * (yy_ ** 2 + xx_ ** 2) / (0.6 * k) ** 2)
                ker /= ker.sum()
            uu = ui[sel]
            vv = vi[sel]
            val = Lpx[sel].astype(np.float32)
            for iy, oy in enumerate(offs):
                for ix, ox in enumerate(offs):
                    u2 = np.clip(uu + ox, 0, W - 1)
                    v2 = np.clip(vv + oy, 0, H - 1)
                    np.add.at(flat, v2 * W + u2, val * np.float32(ker[iy, ix]))
        return int(len(ui))

    # ------------------------------------------------------------------ sensor
    def _sensor(self, L: np.ndarray):
        cfg, hid = self.cfg, self.hid
        Y = L[::4, ::4] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        H4, W4 = Y.shape
        yy, xx = np.mgrid[0:H4, 0:W4]
        wgt = np.exp(-(((yy - H4 / 2) / (0.5 * H4)) ** 2 + ((xx - W4 / 2) / (0.5 * W4)) ** 2))
        meter = float((Y * wgt).sum() / wgt.sum())
        # normalised value v = L * t * G * eta / full_well
        k = hid.eta_e_per_lux_s / hid.full_well_e
        need = cfg.ae_target / max(meter * k, 1e-12)        # required t*G
        ev_min = cfg.exposure_min_s
        ev_max = cfg.exposure_max_s * 10.0 ** (cfg.gain_max_db / 20.0)
        log_need = math.log(min(max(need, ev_min), ev_max))
        if self._log_ev is None:
            self._log_ev = log_need
        else:
            self._log_ev += cfg.ae_speed * (log_need - self._log_ev)
        ev = math.exp(self._log_ev)
        t_exp = min(max(ev, cfg.exposure_min_s), cfg.exposure_max_s)
        gain = max(1.0, ev / t_exp)
        gain_db = 20.0 * math.log10(gain)
        e = L * np.float32(t_exp * hid.eta_e_per_lux_s) * self.vignette
        noise = self.rng.standard_normal(e.shape, dtype=np.float32)
        e = e + noise * np.sqrt(np.maximum(e, 0.0) + np.float32(hid.read_noise_e ** 2))
        v = e * np.float32(gain / hid.full_well_e) * np.asarray(hid.wb_gains, np.float32)
        v = np.clip(np.nan_to_num(v, nan=0.0, posinf=1.0), 0.0, 1.0)
        self.last_exposure = (t_exp, gain_db)
        return linear_to_srgb_u8(v), t_exp, gain_db
