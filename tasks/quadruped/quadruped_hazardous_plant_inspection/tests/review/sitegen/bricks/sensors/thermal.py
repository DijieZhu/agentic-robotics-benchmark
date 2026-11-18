"""Radiometric thermal camera (LWIR microbolometer model) from a segmentation render and a per-geom temperature table.

Pipeline of `ThermalCamera.capture` (output float32 deg C, H x W):
 1. segmentation render at `supersample` x the resolution (which geom each sub-pixel sees; background = no hit);
 2. per sub-pixel apparent radiance L = eps * sigma T^4 + (1 - eps) * sigma T_refl^4 (grey body, Kelvin; the
    reflected temperature is the ambient; background pixels at `background_c`);
 3. pixel integration (box average of the sub-pixels) and the optics' blur (Gaussian, `blur_sigma_px`) in radiance;
 4. back to temperature, plus fixed-pattern noise (per camera instance) and temporal noise with sd = NETD.
Glass and thin plastic are opaque in LWIR, as the segmentation render assumes. The T^4 law stands in for the
8-14 um band radiance (close enough near room temperature for this model). Models: THERMAL_LO 160 x 120 and
THERMAL_HI 320 x 240, both 50 deg horizontal FOV, NETD 0.1 K.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import mujoco
import numpy as np

from .camera import RenderPool, camera_pose_optical, default_scene_option, fovy_from_hfov, intrinsics, segmentation

__all__ = ["ThermalModel", "THERMAL_LO", "THERMAL_HI", "TemperatureTable", "ThermalCamera", "hottest_region",
           "emissivity_correct", "SIGMA"]

SIGMA = 5.670374419e-8
K0 = 273.15


@dataclass(frozen=True)
class ThermalModel:
    name: str = "thermal"
    width: int = 160
    height: int = 120
    hfov_deg: float = 50.0
    netd_k: float = 0.1
    fpn_k: float = 0.08
    blur_sigma_px: float = 0.6
    supersample: int = 3
    background_c: float = 15.0

    @property
    def fovy_deg(self) -> float:
        return fovy_from_hfov(self.hfov_deg, self.width, self.height)

    @property
    def K(self):
        return intrinsics(self.fovy_deg, self.width, self.height)

    def resized(self, w, h):
        return replace(self, width=int(w), height=int(h))


THERMAL_LO = ThermalModel("thermal", 160, 120)
THERMAL_HI = ThermalModel("thermal_hi", 320, 240, supersample=2)


class TemperatureTable:
    """Temperature (deg C) and emissivity per geom of a compiled model; everything not set is at `ambient_c`
    (plus a small fixed per-geom spread, `spread_k`, so that the background is not perfectly flat)."""

    def __init__(self, model, ambient_c: float = 22.0, emissivity: float = 0.93, spread_k: float = 0.4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.model = model
        self.ambient_c = float(ambient_c)
        self.T = (ambient_c + rng.normal(0.0, spread_k, model.ngeom)).astype(np.float64)
        self.eps = np.full(model.ngeom, float(emissivity))

    def set(self, temps: dict, emissivity: dict | None = None):
        for nm, t in temps.items():
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if gid < 0:
                raise KeyError(nm)
            self.T[gid] = float(t)
        for nm, e in (emissivity or {}).items():
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            self.eps[gid] = float(e)

    def set_targets(self, names_list):
        """Load the tables of thermal_source TargetNames."""
        for nm in names_list:
            self.set(nm.state["temps"], nm.state.get("emissivity"))

    def radiance(self) -> np.ndarray:
        Tk = self.T + K0
        Tr = self.ambient_c + K0
        return self.eps * SIGMA * Tk ** 4 + (1.0 - self.eps) * SIGMA * Tr ** 4


def _rad_to_c(L):
    return (np.maximum(L, 1e-9) / SIGMA) ** 0.25 - K0


def emissivity_correct(t_meas_c: float, eps: float, t_refl_c: float) -> float:
    """Object temperature from a measured (black-body equivalent) temperature for a grey body of emissivity eps."""
    Lm = SIGMA * (t_meas_c + K0) ** 4
    Lr = SIGMA * (t_refl_c + K0) ** 4
    Lo = (Lm - (1.0 - eps) * Lr) / max(eps, 1e-3)
    return float(_rad_to_c(Lo))


class ThermalCamera:
    def __init__(self, model, cam_name: str, tm: ThermalModel, table: TemperatureTable, pool: RenderPool | None = None,
                 rng: np.random.Generator | None = None, noise: bool = True, scene_option=None):
        self.model = model
        self.tm = tm
        self.table = table
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self.cam_id < 0:
            raise KeyError(cam_name)
        model.cam_fovy[self.cam_id] = tm.fovy_deg
        self.pool = pool or RenderPool(model)
        self.rng = rng or np.random.default_rng(0)
        self.noise = noise
        self.opt = scene_option or default_scene_option()
        frng = np.random.default_rng(12345)
        self.fpn = frng.normal(0.0, tm.fpn_k, (tm.height, tm.width)).astype(np.float32)

    @property
    def K(self):
        return self.tm.K

    def pose(self, data):
        return camera_pose_optical(self.model, data, self.cam_id)

    def capture(self, data, return_seg: bool = False):
        tm = self.tm
        ss = int(tm.supersample)
        ids, typ = segmentation(self.pool, data, self.cam_id, tm.width * ss, tm.height * ss, self.opt)
        Lg = self.table.radiance()
        Lbg = SIGMA * (tm.background_c + K0) ** 4
        geom = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0) & (ids < self.model.ngeom)
        L = np.full(ids.shape, Lbg, np.float64)
        L[geom] = Lg[ids[geom]]
        L = L.reshape(tm.height, ss, tm.width, ss).mean(axis=(1, 3))
        if tm.blur_sigma_px > 0:
            L = cv2.GaussianBlur(L, (0, 0), tm.blur_sigma_px, borderType=cv2.BORDER_REPLICATE)
        T = _rad_to_c(L).astype(np.float32)
        if self.noise:
            T = T + self.fpn + self.rng.normal(0.0, tm.netd_k, T.shape).astype(np.float32)
        if return_seg:
            return T, ids, typ
        return T


def hottest_region(T: np.ndarray, delta_k: float | None = None, min_excess_k: float = 3.0):
    """The connected region around the hottest pixel: pixels within `delta_k` of the peak (default: half way between
    the background median and the peak). Returns {'u', 'v' (peak pixel), 't_peak', 't_mean', 'n_px', 'bbox'
    (x0, y0, x1, y1), 'centroid'} or None when nothing is `min_excess_k` above the median."""
    Ts = cv2.GaussianBlur(T.astype(np.float32), (3, 3), 0)
    v, u = np.unravel_index(int(np.argmax(Ts)), Ts.shape)
    t_peak = float(T[max(0, v - 1):v + 2, max(0, u - 1):u + 2].max())
    med = float(np.median(T))
    if t_peak - med < min_excess_k:
        return None
    thr = t_peak - (delta_k if delta_k is not None else 0.5 * (t_peak - med))
    m = (T >= thr).astype(np.uint8)
    n, lab = cv2.connectedComponents(m, connectivity=8)
    reg = lab == lab[v, u]
    ys, xs = np.nonzero(reg)
    return {"u": int(u), "v": int(v), "t_peak": t_peak, "t_mean": float(T[reg].mean()), "n_px": int(reg.sum()),
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            "centroid": (float(xs.mean()), float(ys.mean())), "t_median_scene": med}
