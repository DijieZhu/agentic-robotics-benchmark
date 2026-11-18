"""Water optical properties for the camera model (three RGB channels).

Channel wavelengths: R = 650 nm, G = 550 nm, B = 450 nm (the convention used by most RGB underwater image models).
A real camera's red band is broad (about 580-680 nm), so red attenuation is somewhat over-stated at 650 nm; this is a
documented simplification.

Inherent optical properties (IOPs) per channel, all in 1/m:
  a  absorption coefficient
  b  scattering coefficient (water molecules + particles)
  c  = a + b beam attenuation
  kd diffuse attenuation of the downwelling irradiance (Jerlov classifies water by kd, not by c)
Particle phase function: Henyey-Greenstein with asymmetry g (0.924 gives a backscatter fraction b_bp/b_p = 0.017, close to
the Petzold average 0.018); molecular scattering is Rayleigh-like.

Presets (see notes/dev/UWKIT_SENSORS.md for provenance):
  Jerlov II, 1C, 3C, 7C, 9C at 650 / 550 / 450 nm, values as transcribed from Solonenko & Mobley (2015, Appl. Opt.
  54(17)) in a third-party design-tool repository; cross-checked at 550 nm against a second, independent transcription
  (a rendering simulator's lookup table), which agrees for II, 1C, 5C, 9C to 3 significant digits.  Type I and 5C are
  excluded (the transcription has a pure-water-only b for I and a duplicated tail for 5C); III is excluded because its kd
  list is misaligned.  The original paper was not obtained, so these are "checked transcriptions", not primary data.
  Snow densities are self-defined (visual tuning, not particle counts).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

LAMBDA_NM = np.array([650.0, 550.0, 450.0])


def seawater_scattering(lam_nm=LAMBDA_NM) -> np.ndarray:
    """Molecular scattering of sea water, b_sw = 0.00288 (500/lambda)^4.32 1/m (Morel 1974 magnitude, S = 35)."""
    return 0.00288 * (500.0 / np.asarray(lam_nm, dtype=float)) ** 4.32


def hg_phase(cos_t, g):
    """Henyey-Greenstein phase function [1/sr], normalised over the sphere."""
    g = float(g)
    return (1.0 - g * g) / (4.0 * math.pi * (1.0 + g * g - 2.0 * g * np.asarray(cos_t)) ** 1.5)


def rayleigh_phase(cos_t):
    """Molecular (Rayleigh-like, depolarisation ignored) phase function [1/sr]."""
    return (3.0 / (16.0 * math.pi)) * (1.0 + np.asarray(cos_t) ** 2)


def hg_backscatter_fraction(g: float) -> float:
    g = float(g)
    return (1 - g) / (2 * g) * ((1 + g) / math.sqrt(1 + g * g) - 1)


@dataclass(frozen=True)
class WaterOptics:
    name: str
    a: tuple                      # absorption (R, G, B) 1/m
    b: tuple                      # total scattering (R, G, B) 1/m
    kd: Optional[tuple] = None    # diffuse attenuation of downwelling light (R, G, B) 1/m; None -> approximation
    g: float = 0.924              # HG asymmetry of the particle phase function
    g_ambient: float = 0.70       # smoothed asymmetry used for in-scatter of the (diffuse-ish) downwelling field
    forward_frac: float = 0.5     # fraction of scattering that stays inside the forward-scatter PSF (rest is lost)
    psf_narrow_weight: float = 0.35   # share of that forward light in the narrow PSF lobe (rest in the wide lobe)
    psf_narrow_deg: float = 0.25  # sigma of the narrow PSF lobe (deg)
    psf_wide_deg: float = 2.5     # sigma of the wide PSF lobe (deg)
    snow_density: float = 10.0    # resolvable marine-snow particles per m^3 (visual parameter)
    snow_diam_mm: tuple = (0.5, 3.0)  # log-uniform particle diameter range
    note: str = ""

    # ------------------------------------------------------------------ derived
    @property
    def a_(self) -> np.ndarray:
        return np.asarray(self.a, dtype=np.float64)

    @property
    def b_(self) -> np.ndarray:
        return np.asarray(self.b, dtype=np.float64)

    @property
    def c(self) -> np.ndarray:
        return self.a_ + self.b_

    @property
    def b_water(self) -> np.ndarray:
        return np.minimum(seawater_scattering(), self.b_)

    @property
    def b_particle(self) -> np.ndarray:
        return np.maximum(self.b_ - self.b_water, 0.0)

    @property
    def bb(self) -> np.ndarray:
        """Backscattering coefficient: 0.5 b_w + B_p(g) b_p."""
        return 0.5 * self.b_water + hg_backscatter_fraction(self.g) * self.b_particle

    @property
    def kd_(self) -> np.ndarray:
        return np.asarray(self.kd, dtype=float) if self.kd is not None else self.kd_approx(0.87)

    def kd_approx(self, mu_d: float = 0.87) -> np.ndarray:
        """Gordon-type K_d = (a/mu_d) sqrt(1 + (b/a) G), G per the pi-SUB paper (arXiv 2608.10589 eq. 4-5)."""
        a, b = self.a_, self.b_
        eta = self.b_water / np.maximum(b, 1e-9)
        G = mu_d * (1.3433 * eta - 0.01414) + (0.1304 + 0.272 * eta)
        return (a / mu_d) * np.sqrt(1.0 + (b / np.maximum(a, 1e-9)) * G)

    def visibility_black_target_m(self) -> float:
        """Horizontal visibility of a black target, y ~ 4.8 / c(550) (empirical; coefficient not verified at source)."""
        return 4.8 / float(self.c[1])

    def contrast_limit_range_m(self, eps: float = 0.01) -> float:
        """Range where direct signal falls to eps in the clearest channel: -ln(eps) / min(c)."""
        return -math.log(eps) / float(np.min(self.c))

    def consistency(self) -> dict:
        """Physical sanity checks: a <= kd <= c and kd(table) vs kd(approx)."""
        kd = self.kd_
        return {
            "a<=kd": bool(np.all(self.a_ <= kd + 1e-9)),
            "kd<=c": bool(np.all(kd <= self.c + 1e-9)),
            "kd_table/kd_approx": (kd / self.kd_approx(0.87)).round(3).tolist(),
        }

    def scaled_particles(self, factor: float, name: Optional[str] = None) -> "WaterOptics":
        """Scale the particle scattering (and snow density) by `factor` (a turbidity knob of the site).  Absorption is
        left unchanged (dissolved matter is not particles)."""
        b = self.b_water + self.b_particle * float(factor)
        kd = None
        if self.kd is not None:
            # keep the table/approx ratio of the base water
            ratio = self.kd_ / self.kd_approx(0.87)
            kd = tuple((replace(self, b=tuple(b), kd=None).kd_approx(0.87) * ratio).tolist())
        return replace(self, name=name or f"{self.name}*{factor:g}", b=tuple(b.tolist()), kd=kd,
                       snow_density=self.snow_density * float(factor))

    def summary(self) -> dict:
        return {
            "name": self.name, "a": list(self.a), "b": list(self.b), "c": self.c.round(4).tolist(),
            "kd": self.kd_.round(4).tolist(), "bb": self.bb.round(5).tolist(),
            "visibility_black_target_m": round(self.visibility_black_target_m(), 2),
            "contrast_1pct_range_m": round(self.contrast_limit_range_m(), 2),
            "snow_density_per_m3": self.snow_density,
        }


# Values at 650 / 550 / 450 nm.  Source: transcription of Solonenko & Mobley 2015 (see module docstring).
JERLOV = {
    "II": WaterOptics("jerlov_II", a=(0.334, 0.0622, 0.0241), b=(0.270, 0.358, 0.504), kd=(0.393, 0.0998, 0.0619),
                      snow_density=3.0, note="open ocean, Chl ~0.5 mg/m3"),
    "1C": WaterOptics("jerlov_1C", a=(0.344, 0.076, 0.105), b=(0.274, 0.365, 0.514), kd=(0.431, 0.145, 0.174),
                      snow_density=8.0, note="clearest coastal"),
    "3C": WaterOptics("jerlov_3C", a=(0.346, 0.082, 0.154), b=(0.800, 1.06, 1.50), kd=(0.478, 0.211, 0.288),
                      snow_density=20.0, note="coastal"),
    "7C": WaterOptics("jerlov_7C", a=(0.403, 0.195, 0.542), b=(1.77, 2.35, 3.30), kd=(0.635, 0.462, 0.868),
                      snow_density=60.0, note="turbid coastal / harbour"),
    "9C": WaterOptics("jerlov_9C", a=(0.456, 0.348, 0.943), b=(2.35, 3.12, 4.39), kd=(0.755, 0.635, 1.56),
                      snow_density=100.0, note="very turbid coastal"),
}

# The three named presets used by the tasks.
CLEAR = replace(JERLOV["II"], name="clear")
COASTAL = replace(JERLOV["3C"], name="coastal")
HARBOUR = replace(JERLOV["7C"], name="harbour_turbid")
# Self-defined extra: Jerlov II absorption with 30 % of its particle scattering (black-target visibility ~ 28 m),
# for "very clear" offshore sites.  Not a Jerlov type.
VERY_CLEAR = replace(JERLOV["II"].scaled_particles(0.3), name="very_clear_selfdefined", snow_density=1.0,
                     note="self-defined: Jerlov II with b_p x 0.3")
PRESETS = {"clear": CLEAR, "coastal": COASTAL, "harbour_turbid": HARBOUR, "very_clear": VERY_CLEAR}


def preset(name: str) -> WaterOptics:
    if name in PRESETS:
        return PRESETS[name]
    if name in JERLOV:
        return JERLOV[name]
    raise KeyError(f"unknown water preset {name!r}; known: {sorted(PRESETS) + sorted(JERLOV)}")


@dataclass(frozen=True)
class AmbientLight:
    """Down-welling daylight.  surface_lux = illuminance just below the surface (0 at night).

    E_d(z) = surface_lux * exp(-kd * z) per channel (z = depth of the point).  At a few hundred metres (or at night)
    this is negligible, i.e. "deep water has no ambient light" falls out of the exponential.
    direct_frac_surface: share of the down-welling irradiance that is direct (sun) just below the surface; it decays as
    exp(-z / direct_scale_m) and only the direct part carries caustics.
    """
    surface_lux: float = 0.0
    sun_dir_world: tuple = (0.0, 0.0, -1.0)   # propagation direction of the refracted sun beam (points down)
    direct_frac_surface: float = 0.6
    direct_scale_m: float = 8.0
    upwelling_ratio: float = 0.03            # E_u / E_d
    caustics: bool = True
    caustic_fade_m: float = 6.0
    color: tuple = (1.0, 1.0, 1.0)           # relative RGB of daylight after white balance

    @property
    def active(self) -> bool:
        return self.surface_lux > 0.0


DAYLIGHT_SUNNY = AmbientLight(surface_lux=60000.0)
DAYLIGHT_OVERCAST = AmbientLight(surface_lux=12000.0, direct_frac_surface=0.15)
DARK = AmbientLight(surface_lux=0.0)
