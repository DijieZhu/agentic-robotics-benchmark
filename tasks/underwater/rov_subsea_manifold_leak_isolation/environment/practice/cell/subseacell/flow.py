"""Site current field of the manifold task (cell side, grader / service only; never shipped to the agent).

    flow = StationFlow.from_site(site)
    flow.field(pos_w, t)            # uwkit CurrentField: drives the vehicle, the free objects and the plume
    flow.v2m(t)                     # mean horizontal current 2 m above the sea bed, outside the structure (the set speed)
    flow.velocity_fast(P, t)        # the same field for many points (plume particles), vertical-table fast path

Model (DESIGN 2.1, the "tide" row; ruling L15: the gust model stays as it is):

    Scaled( BoundaryLayer( Tidal + GaussMarkov, seabed, delta = 20 m, 1/7, z_ref = 2 m ), structure factor )

  * Tidal: uwkit tidal ellipse.  `U2` is the major-axis amplitude AT 2 m above the sea bed (the station's set speed),
    slack at `t_slack` (flow along +theta after it), minor-axis ratio `minor_ratio`, M2 period.
  * BoundaryLayer: 1/7 power law with a 20 m layer, normalised at z_ref = 2 m (uwkit 0.3.0 `z_ref`): the speed at 2 m
    above the bed equals the set speed and keeps growing as (h / 2 m)^(1/7) up to 20 m (the G3 construction, which
    multiplied the amplitude by (20 / 2)^(1/7) with the 0.2.0 layer).
  * GaussMarkov: spatially uniform first-order gusts, horizontal std sigma = gm_coef * |v_2m(t_meas)| + 0.015 m/s at
    2 m, time scale gm_tau; vertical component 0.25 sigma and gm_tau / 2.  The process is generated on uwkit's fixed
    0.1 s grid from FLOW_T_ORIGIN (mission time -600 s), so the plume spin-up before the start of the run sees a real
    gust history and the samples do not depend on the order of the queries.
  * Structure factor: f_inside (0.4-0.7) inside the frame envelope (mudmat bottom .. roof grating, lx x ly), 1 outside,
    linear 0.3 m transition (ramp centred on the envelope faces).

Times are MISSION time (s since the start of the run).  The sea-bed depth comes from the site (`seabed_depth_m`; if
absent it is derived from the structure origin, which sits on the mudmat top).  Frames: W (z up, surface z = 0);
S = structure frame (origin on the mudmat top at the structure centre, x along lx, z up), pose (pose_o, pose_psi).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from uwkit.dynamics import current as CUR

# ------------------------------------------------------------------------------------------------ model constants
Z_REF_M = 2.0                  # the set speed is defined at this height above the sea bed (DESIGN 2.1)
BL_DELTA_M = 20.0              # boundary-layer thickness of the 1/7 law (G3)
BL_EXPONENT = 1.0 / 7.0
TIDE_PERIOD_S = 12.42 * 3600.0  # M2
GUST_SIGMA_OFFSET_M_S = 0.015
GUST_VERTICAL_SIGMA_FRAC = 0.25
GUST_VERTICAL_TAU_FRAC = 0.5
GUST_DT_S = 0.1                # uwkit GaussMarkov sample grid
FLOW_T_ORIGIN_S = -600.0       # mission time at which the gust process starts (covers the plume spin-up)
PROFILE_DZ_M = 0.025           # vertical table spacing of the fast path
PROFILE_SPAN_M = 24.0          # table from the bed up to this height (constant above)
NEAR_BED_EXACT_M = 1.0         # below this height above the bed the fast path evaluates the power law exactly


# ------------------------------------------------------------------------------------------------ frames and geometry
class SitePose:
    """As-built pose of the structure frame S in W: origin (mudmat top, structure centre) and heading psi [rad]."""

    def __init__(self, origin_w, psi: float):
        self.o = np.asarray(origin_w, float).reshape(3)
        self.psi = float(psi)
        c, s = math.cos(self.psi), math.sin(self.psi)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @classmethod
    def from_site(cls, site: dict) -> "SitePose":
        return cls(site["pose_o"], site["pose_psi"])

    def to_w(self, p_s):
        p = np.asarray(p_s, float)
        if p.shape[-1] == 2:
            p = np.concatenate([p, np.zeros(p.shape[:-1] + (1,))], -1)
        return p @ self.R.T + self.o

    def to_s(self, p_w):
        return (np.asarray(p_w, float) - self.o) @ self.R

    def vec_to_w(self, v_s):
        v = np.asarray(v_s, float)
        if v.shape[-1] == 2:
            v = np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], -1)
        return v @ self.R.T

    def vec_to_s(self, v_w):
        return np.asarray(v_w, float) @ self.R


@dataclass
class StructureEnvelope:
    """The box in which the current is reduced (frame footprint, mudmat bottom .. roof grating top), in S.
    Defaults = the stand-in manifold of G3 / G6 [placeholder until the purchased asset is measured, A01 / A02]."""
    lx: float = 12.0
    ly: float = 8.0
    roof_z: float = 3.4            # roof grating top above the mudmat top
    mudmat_thk: float = 0.3
    ramp_m: float = 0.3            # linear transition, centred on the faces

    @classmethod
    def from_site(cls, site: dict) -> "StructureEnvelope":
        s = dict(site.get("structure") or {})
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in s.items() if k in names})

    def factor_fn(self, pose: SitePose, f_inside: float):
        """Envelope factor 1 - (1 - f) * inside(p), inside in [0, 1] with the ramp; callable (p_w (n, 3), t) -> (n,)."""
        hx, hy, ramp = self.lx / 2.0, self.ly / 2.0, self.ramp_m
        top, bot = self.roof_z, -self.mudmat_thk
        o, R = pose.o, pose.R
        f = float(f_inside)

        def fn(p, t=None):
            p = np.asarray(p, float).reshape(-1, 3)
            q = (p - o) @ R
            ax = np.clip((hx - np.abs(q[:, 0])) / ramp + 0.5, 0.0, 1.0)
            ay = np.clip((hy - np.abs(q[:, 1])) / ramp + 0.5, 0.0, 1.0)
            az = np.clip((top - q[:, 2]) / ramp + 0.5, 0.0, 1.0) * np.clip((q[:, 2] - bot) / ramp + 0.5, 0.0, 1.0)
            return 1.0 - (1.0 - f) * (ax * ay * az)
        return fn


def seabed_depth_of(site: dict, env: Optional[StructureEnvelope] = None) -> float:
    """Water depth at the site [m].  Explicit `seabed_depth_m`, checked against the structure origin (mudmat top =
    sea bed + mudmat thickness); derived from the origin when absent."""
    env = env or StructureEnvelope.from_site(site)
    derived = -(float(site["pose_o"][2]) - env.mudmat_thk)
    if "seabed_depth_m" in site and site["seabed_depth_m"] is not None:
        d = float(site["seabed_depth_m"])
        if abs(d - derived) > 1e-6:
            raise ValueError(f"site seabed_depth_m {d} disagrees with the structure origin (mudmat top) -> {derived}")
        return d
    return derived


# ------------------------------------------------------------------------------------------------ uwkit helpers
class TimeShift(CUR.CurrentField):
    """inner(pos, t + shift): starts a fixed-grid process (GaussMarkov) at mission time -shift."""

    def __init__(self, inner: CUR.CurrentField, shift_s: float):
        self.inner, self.shift = inner, float(shift_s)

    def velocity(self, pos, t):
        return self.inner.velocity(pos, float(t) + self.shift)

    def bind_rng(self, rng):
        self.inner.bind_rng(rng)
        return self

    def children(self):
        return [self.inner]

    @property
    def needs_rng(self):
        return self.inner.needs_rng


# ------------------------------------------------------------------------------------------------ the station flow
@dataclass
class FlowSpec:
    U2: float                  # major-axis tidal amplitude at 2 m above the bed [m/s]
    theta: float               # flood (major-axis) direction in W [rad]
    t_slack: float             # slack time [mission s]; the flow runs along +theta after it
    minor_ratio: float
    f_inside: float            # in-structure reduction
    gm_coef: float             # gust std = gm_coef * |v_2m(t_meas)| + 0.015 m/s
    gm_tau: float              # gust time scale [s]
    gm_seed: int
    t_meas: float = 240.0      # the time whose 2 m speed sets the gust std (inside the marker window)

    @classmethod
    def from_site(cls, site: dict) -> "FlowSpec":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(site["flow"]).items() if k in names})

    def tidal(self) -> CUR.Tidal:
        e = (math.cos(self.theta), math.sin(self.theta))
        return CUR.Tidal(self.U2, direction=e, period_s=TIDE_PERIOD_S, t_peak_s=self.t_slack + TIDE_PERIOD_S / 4.0,
                         minor_ratio=self.minor_ratio)

    def v2m(self, t: float) -> np.ndarray:
        """Mean (no gust) horizontal current 2 m above the bed outside the structure [m/s] (2,)."""
        return self.tidal().horizontal(float(t))

    def gust_sigma(self) -> float:
        return self.gm_coef * float(np.linalg.norm(self.v2m(self.t_meas))) + GUST_SIGMA_OFFSET_M_S


def profile_factor(h_above_bed, delta: float = BL_DELTA_M, z_ref: float = Z_REF_M):
    """Mean speed at height h relative to the set speed at z_ref: (min(h, delta) / z_ref)^(1/7), 0 below the bed."""
    h = np.clip(np.asarray(h_above_bed, float), 0.0, delta)
    return (h / z_ref) ** BL_EXPONENT


class StationFlow:
    """The station's current: the uwkit field (`field`, `mean_field`) and the fast vertical-table path of the plume.

    Tidal, GaussMarkov and BoundaryLayer are horizontally uniform, so for many points the field is
    prof(z) * V(t) * structure_factor(p), with V(t) the free stream above the layer (one uwkit call per time) and prof
    the layer factor tabulated every 0.025 m (linear interpolation, < 1e-5 m/s above 1 m from the bed; exact below)."""

    def __init__(self, fs: FlowSpec, pose: SitePose, seabed_depth_m: float,
                 envelope: Optional[StructureEnvelope] = None):
        self.fs = fs
        self.pose = pose
        self.seabed_depth = float(seabed_depth_m)
        self.env = envelope or StructureEnvelope()
        sg = fs.gust_sigma()
        gm = CUR.GaussMarkov(sigma=(sg, sg, GUST_VERTICAL_SIGMA_FRAC * sg),
                             tau_c=(fs.gm_tau, fs.gm_tau, GUST_VERTICAL_TAU_FRAC * fs.gm_tau),
                             rng=np.random.default_rng(int(fs.gm_seed)), dt_sample=GUST_DT_S)
        self.gust = TimeShift(gm, -FLOW_T_ORIGIN_S)
        self.tide = fs.tidal()
        self.free = CUR.BoundaryLayer(CUR.Sum([self.tide, self.gust]), self.seabed_depth, delta=BL_DELTA_M,
                                      exponent=BL_EXPONENT, z_ref=Z_REF_M)
        self.sfac = self.env.factor_fn(pose, fs.f_inside)
        self.field = CUR.Scaled(self.free, self.sfac)           # vehicle, free objects, checks
        self.mean_field = CUR.Scaled(CUR.BoundaryLayer(self.tide, self.seabed_depth, delta=BL_DELTA_M,
                                                       exponent=BL_EXPONENT, z_ref=Z_REF_M), self.sfac)
        # fast path tables
        self.z_bed = -self.seabed_depth
        self.zcol = self.z_bed + np.arange(0.0, PROFILE_SPAN_M + 1e-9, PROFILE_DZ_M)
        h = np.clip(self.zcol - self.z_bed, 0.0, BL_DELTA_M)
        self.prof_tab = (h / BL_DELTA_M) ** BL_EXPONENT            # relative to the free stream above the layer
        self.prof_slope = np.append(np.diff(self.prof_tab), 0.0)
        self._inv_dz = 1.0 / PROFILE_DZ_M
        self._ptop = np.array([[pose.o[0] + 50.0, pose.o[1], self.z_bed + BL_DELTA_M + 1.0]])
        # the structure factor is exactly 1 above roof + ramp / 2 (and outside the footprint + ramp / 2)
        self.z_fac_top = pose.o[2] + self.env.roof_z + 0.5 * self.env.ramp_m + 1e-9

    @classmethod
    def from_site(cls, site: dict) -> "StationFlow":
        env = StructureEnvelope.from_site(site)
        return cls(FlowSpec.from_site(site), SitePose.from_site(site), seabed_depth_of(site, env), env)

    # ---------------------------------------------------------------- set speed and mean profile
    def v2m(self, t: float) -> np.ndarray:
        return self.fs.v2m(t)

    def mean_speed_at_height(self, h_above_bed: float, t: float) -> float:
        """|mean current| at height h above the bed outside the structure (no gusts) [m/s]."""
        return float(np.linalg.norm(self.fs.v2m(t)) * profile_factor(h_above_bed))

    def gust_sample(self, t: float) -> np.ndarray:
        """The gust vector of the free stream at 2 m above the bed [m/s] (3,)."""
        return self.gust.velocity(np.zeros((1, 3)), t)[0]

    # ---------------------------------------------------------------- fast path
    def V(self, t: float) -> np.ndarray:
        """Free stream above the layer (Tidal + gust, x (delta / z_ref)^(1/7)), via the uwkit field."""
        return self.free.velocity(self._ptop, t)[0]

    def prof(self, z: np.ndarray) -> np.ndarray:
        x = (np.asarray(z, float) - self.zcol[0]) * self._inv_dz
        i0 = x.astype(np.int64)
        np.clip(i0, 0, self.zcol.shape[0] - 2, out=i0)
        return self.prof_tab[i0] + self.prof_slope[i0] * np.clip(x - i0, 0.0, 1.0)

    def factor_fast(self, P: np.ndarray) -> np.ndarray:
        """prof(z) * structure factor for points P (n, 3).  Within 1 m of the bed (where the 1/7 law is too curved for
        the table) the profile is evaluated exactly; those points are a subset of the low ones."""
        pr = self.prof(P[:, 2])
        low = P[:, 2] < self.z_fac_top
        if np.any(low):
            Pl = P[low]
            f = self.sfac(Pl)
            hb = Pl[:, 2] - self.z_bed
            nb = hb < NEAR_BED_EXACT_M
            if np.any(nb):
                pl = pr[low]
                pl[nb] = (np.clip(hb[nb], 0.0, None) / BL_DELTA_M) ** BL_EXPONENT
                pr[low] = pl * f
            else:
                pr[low] *= f
        return pr

    def velocity_fast(self, P: np.ndarray, t: float, V: Optional[np.ndarray] = None) -> np.ndarray:
        V = self.V(t) if V is None else V
        return self.factor_fast(P)[:, None] * V[None, :]

    def check_fast_path(self, P: np.ndarray, t: float) -> float:
        return float(np.max(np.abs(self.velocity_fast(P, t) - self.field.velocity(P, t))))
