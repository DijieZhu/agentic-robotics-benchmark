"""Spinning lidar from mujoco.mj_multiRay: 2D (one channel) and 16-channel 3D (+-15 deg).

All beams of a sweep start at the lidar's optical centre (a site) at one instant (no motion distortion during the
revolution). Measurement model: range noise sd = a + b * r, a fraction `dropout` of the returns is lost at random,
returns outside [r_min, r_max] are invalid (NaN in the range image). Points are given in the lidar frame (the site's
axes: x forward, y left, z up). Ray casting sees every geom in the selected geom groups (default 0-2, like the
cameras), including glass (a real lidar partly sees through glass; idealised).

Self-exclusion (both back ends, since 2026-11-13): with `exclude_root` (the robot's root body), **a beam
whose first hit is the robot's own body gives no return** (NaN), as on a real lidar whose own legs block the view and
whose driver drops those points. The geoms of the body that carries the lidar (the sensor housing the beams leave
from) never block. The robot's blocking geoms are those of the `exclude_root` subtree in `self_geomgroup` (default
groups 0-2, the robot as the cameras draw it) or in the lidar's own `geomgroup`; they block whether or not the scene
pass sees their group. Earlier versions re-cast blocked beams from behind the robot (the robot was transparent for
some beams and blocked others); that mix is gone.

`DepthLidar` is a faster approximate backend for the multi-channel lidar: four 90 deg depth renders from cameras at
the lidar origin (`add_lidar_cameras` at build time), each beam reads the depth pixel nearest to its direction
(angular quantisation about 90 deg / width). With `exclude_root` it applies the same self-exclusion: one extra
`mj_multiRay` pass, restricted to the beams that cross a blocking geom's bounding box and to dynamic geoms (the
static plant is skipped; about 1-8 ms for 28,800 beams), gives each beam's distance to the robot, and a beam whose
robot hit is not behind its depth range gives no return. Hide the robot from the depth renders by geom group (and
list that group in `self_geomgroup`) for an exact result; with the robot drawn, about 1 beam in 10^4 that grazes the
robot's outline reads the robot's pixel (quantisation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from ..common import body_subtree_ids

__all__ = ["LidarModel", "LIDAR_2D_360", "LIDAR_2D_720", "LIDAR_3D16", "Lidar", "LidarScan", "DepthLidar",
           "add_lidar_cameras"]


@dataclass(frozen=True)
class LidarModel:
    name: str
    channels: int = 1
    beams: int = 720                 # per revolution
    el_min_deg: float = 0.0
    el_max_deg: float = 0.0
    r_min: float = 0.1
    r_max: float = 20.0
    noise_a: float = 0.01
    noise_b: float = 0.0005
    dropout: float = 0.01
    rate_hz: float = 10.0

    def directions(self) -> np.ndarray:
        """(channels, beams, 3) unit vectors in the lidar frame; azimuth from +x counter-clockwise."""
        az = np.arange(self.beams) * (2 * math.pi / self.beams)
        if self.channels == 1:
            el = np.array([math.radians(0.5 * (self.el_min_deg + self.el_max_deg))])
        else:
            el = np.radians(np.linspace(self.el_min_deg, self.el_max_deg, self.channels))
        E, A = np.meshgrid(el, az, indexing="ij")
        return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)


LIDAR_2D_360 = LidarModel("lidar2d_360", 1, 360)
LIDAR_2D_720 = LidarModel("lidar2d_720", 1, 720)
LIDAR_3D16 = LidarModel("lidar16", 16, 1800, -15.0, 15.0, r_min=0.1, r_max=20.0, noise_a=0.02, noise_b=0.0)


@dataclass
class LidarScan:
    ranges: np.ndarray        # (channels, beams) float32, NaN = no return
    points: np.ndarray        # (N, 3) float32 valid returns in the lidar frame
    ring: np.ndarray          # (N,) channel index
    azimuth: np.ndarray       # (N,) radians
    T_world_lidar: np.ndarray  # 4x4
    # (channels, beams) hit geom ids (ground truth, debugging only): -1 no hit within r_max, SELF_BLOCKED (-2) the
    # beam's first hit is the robot's own body (no return)
    geom: np.ndarray | None = None


SELF_BLOCKED = -2


class _SelfBlock:
    """The robot's blocking geoms and a ray pass against them (shared by Lidar and DepthLidar)."""

    def __init__(self, model, exclude_root, own_body: int, geomgroup, self_geomgroup):
        self.model = model
        self.own_body = int(own_body)
        self.excl = np.zeros(model.ngeom, bool)          # every geom of the robot's subtree (a hit on it is "self")
        self.block = np.zeros(model.ngeom, bool)         # the subset that blocks beams (the drawn robot)
        self.groups = None
        self.need_pass = False
        if exclude_root is None:
            return
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, exclude_root)
        if bid < 0:
            raise KeyError(exclude_root)
        bodies = body_subtree_ids(model, bid)
        self.excl[np.isin(model.geom_bodyid, bodies)] = True
        self.excl[model.geom_bodyid == self.own_body] = False      # the sensor housing never blocks
        gg = np.ones(6, bool) if geomgroup is None else np.asarray(geomgroup, bool)
        sg = np.ones(6, bool) if self_geomgroup is None else np.asarray(self_geomgroup, bool)
        grp = np.clip(np.asarray(model.geom_group), 0, 5)
        self.block = self.excl & (sg[grp] | gg[grp])
        if self.block.any():
            mask = np.zeros(6, np.uint8)
            mask[np.unique(grp[self.block])] = 1
            self.groups = mask
            # robot geoms the scene pass does not see need their own pass
            self.need_pass = bool((self.block & ~gg[grp]).any())
        # a robot welded to the world (no joint) is "static" for mj_ray: the pass must include static geoms then
        self.flg_static = 1 if int(model.body_weldid[bid]) == 0 else 0

    def distances(self, data, p, vec, cutoff):
        """Distance along each unit direction (N, 3) from p to the first robot geom (inf if the first geom hit in the
        robot's groups is not a blocking robot geom, or nothing is hit). Only beams inside the bounding-sphere cone
        of some blocking geom are cast (mj_multiRay's cost grows with rays x geoms)."""
        n = len(vec)
        p = np.asarray(p, np.float64)
        vec = np.ascontiguousarray(vec, np.float64)
        out = np.full(n, np.inf)
        # candidates: beams that cross the local bounding box (geom_aabb) of some blocking geom (slab test)
        cand = np.zeros(n, bool)
        for g in np.nonzero(self.block)[0]:
            R = np.asarray(data.geom_xmat[g], np.float64).reshape(3, 3)
            o = R.T @ (p - np.asarray(data.geom_xpos[g], np.float64))
            w = vec @ R                                              # directions in the geom frame
            c = np.asarray(self.model.geom_aabb[g][:3], np.float64)
            h = np.asarray(self.model.geom_aabb[g][3:], np.float64) * 1.01 + 1e-3
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (c - h - o) / w
                t2 = (c + h - o) / w
            lo = np.where(np.abs(w) < 1e-12, np.where(np.abs(o - c) <= h, -np.inf, np.inf), np.minimum(t1, t2))
            hi = np.where(np.abs(w) < 1e-12, np.where(np.abs(o - c) <= h, np.inf, -np.inf), np.maximum(t1, t2))
            tmin, tmax = lo.max(axis=1), hi.min(axis=1)
            cand |= (tmax >= np.maximum(tmin, 0.0)) & (tmin <= cutoff)
        idx = np.nonzero(cand)[0]
        if len(idx) == 0:
            return out
        sub = np.ascontiguousarray(vec[idx])
        gid = np.zeros(len(idx), np.int32)
        dist = np.zeros(len(idx), np.float64)
        mujoco.mj_multiRay(self.model, data, p, sub.ravel(), self.groups, self.flg_static, self.own_body, gid, dist,
                           None, len(idx), cutoff)
        ok = (gid >= 0) & (dist >= 0)
        hit = np.zeros(len(idx), bool)
        hit[ok] = self.block[gid[ok]]
        out[idx[hit]] = dist[hit]
        return out


class Lidar:
    def __init__(self, model, site_name: str, lm: LidarModel, exclude_root: str | None = None,
                 rng: np.random.Generator | None = None, geomgroup=(1, 1, 1, 0, 0, 0), noise: bool = True,
                 self_geomgroup=(1, 1, 1, 0, 0, 0)):
        self.model = model
        self.lm = lm
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site < 0:
            raise KeyError(site_name)
        self.own_body = int(model.site_bodyid[self.site])
        self.dirs = lm.directions()
        self.flat = self.dirs.reshape(-1, 3)
        self.rng = rng or np.random.default_rng(0)
        self.noise = noise
        self.geomgroup = None if geomgroup is None else np.asarray(geomgroup, np.uint8)
        self.selfblock = _SelfBlock(model, exclude_root, self.own_body, geomgroup, self_geomgroup)
        self.excl = self.selfblock.excl
        n = len(self.flat)
        self._gid = np.zeros(n, np.int32)
        self._dist = np.zeros(n, np.float64)

    def scan(self, data, return_geom: bool = False) -> LidarScan:
        m, lm = self.model, self.lm
        p = data.site_xpos[self.site].copy()
        R = data.site_xmat[self.site].reshape(3, 3)
        vec = (self.flat @ R.T).astype(np.float64)
        n = len(vec)
        mujoco.mj_multiRay(m, data, p, vec.ravel(), self.geomgroup, 1, self.own_body, self._gid, self._dist, None, n,
                           lm.r_max)
        dist = self._dist.copy()
        gid = self._gid.copy()
        sb = self.selfblock
        if sb.excl.any():
            # first hit on the robot's own body: no return
            blocked = (gid >= 0) & sb.excl[np.maximum(gid, 0)]
            if sb.need_pass:
                d_self = sb.distances(data, p, vec, lm.r_max)
                d_scene = np.where((gid >= 0) & (dist >= 0), dist, np.inf)
                blocked |= np.isfinite(d_self) & (d_self <= d_scene)
            gid[blocked] = SELF_BLOCKED
            dist[blocked] = -1.0
        r = dist.astype(np.float32)
        r[(gid < 0) | (dist < 0)] = np.nan
        if self.noise:
            ok = np.isfinite(r)
            r[ok] += (self.rng.standard_normal(int(ok.sum())) * (lm.noise_a + lm.noise_b * r[ok])).astype(np.float32)
            r[self.rng.random(n) < lm.dropout] = np.nan
        r[(r < lm.r_min) | (r > lm.r_max)] = np.nan
        ranges = r.reshape(lm.channels, lm.beams)
        ok = np.isfinite(r)
        pts = (self.flat[ok] * r[ok, None]).astype(np.float32)
        ring = np.repeat(np.arange(lm.channels), lm.beams)[ok]
        az = np.tile(np.arange(lm.beams) * (2 * math.pi / lm.beams), lm.channels)[ok]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return LidarScan(ranges, pts, ring, az, T, gid.reshape(lm.channels, lm.beams) if return_geom else None)


def add_lidar_cameras(spec, body, prefix: str, pos=(0.0, 0.0, 0.0), width: int = 512, el_max_deg: float = 16.0):
    """Four depth cameras (azimuth 0, 90, 180, 270 deg about the body's z) at `pos` on `body` for DepthLidar.
    Returns their names; the image size for DepthLidar is (width, height) with height covering +-el_max_deg."""
    from ..common import quat_from_mat
    h = int(math.ceil(width * math.tan(math.radians(el_max_deg)) / math.tan(math.radians(45.0)))) * 2 // 2
    fovy = math.degrees(2 * math.atan(math.tan(math.radians(45.0)) * h / float(width)))
    names = []
    for k in range(4):
        a = k * math.pi / 2
        f = np.array([math.cos(a), math.sin(a), 0.0])
        left = np.array([-math.sin(a), math.cos(a), 0.0])
        # camera axes: x = right = -left, y = up, z = -forward
        R = np.column_stack([-left, [0.0, 0.0, 1.0], -f])
        nm = "%s/lidarcam%d" % (prefix, k)
        body.add_camera(name=nm, pos=list(pos), quat=list(quat_from_mat(R)), fovy=fovy)
        names.append(nm)
    return names, (width, h)


class DepthLidar:
    """Multi-channel spinning lidar from four depth renders (see add_lidar_cameras)."""

    def __init__(self, model, cam_names, size, lm: LidarModel, pool=None, rng=None, geomgroup=(1, 1, 1, 0, 0, 0),
                 noise: bool = True, exclude_root: str | None = None, self_geomgroup=(1, 1, 1, 0, 0, 0)):
        from .camera import RenderPool, intrinsics
        self.model = model
        self.lm = lm
        self.cams = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in cam_names]
        # self-exclusion as in Lidar: the cameras' body is the sensor housing; the scene pass is the depth render
        self.selfblock = _SelfBlock(model, exclude_root, int(model.cam_bodyid[self.cams[0]]), geomgroup,
                                    self_geomgroup)
        self.w, self.h = size
        self.pool = pool or RenderPool(model)
        self.rng = rng or np.random.default_rng(0)
        self.noise = noise
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0
        for g in range(6):
            self.opt.geomgroup[g] = int(geomgroup[g])
        K = intrinsics(float(model.cam_fovy[self.cams[0]]), self.w, self.h)
        d = lm.directions()                                  # (C, B, 3) in the lidar frame
        az = np.arctan2(d[..., 1], d[..., 0]) % (2 * math.pi)
        sector = np.floor(((az + math.pi / 4) % (2 * math.pi)) / (math.pi / 2)).astype(int) % 4
        rel = az - sector * (math.pi / 2)
        rel = (rel + math.pi) % (2 * math.pi) - math.pi
        el = np.arcsin(np.clip(d[..., 2], -1, 1))
        x = -np.tan(rel)                                       # camera x (right) per unit forward
        y = np.tan(el) / np.cos(rel)                          # camera y (up) per unit forward
        self.u = np.clip(np.round(K[0, 2] + K[0, 0] * x).astype(int), 0, self.w - 1)
        self.v = np.clip(np.round(K[1, 2] - K[1, 1] * y).astype(int), 0, self.h - 1)
        self.scale = np.sqrt(1.0 + x * x + y * y)            # z-depth -> range
        self.sector = sector
        self.dirs = d
        self.site_body = None

    def scan(self, data, T_world_lidar=None) -> LidarScan:
        lm = self.lm
        r_ = self.pool.get(self.w, self.h)
        r_.enable_depth_rendering()
        Z = np.zeros((4, self.h, self.w), np.float32)
        try:
            for k, cid in enumerate(self.cams):
                r_.update_scene(data, camera=cid, scene_option=self.opt)
                Z[k] = r_.render()
        finally:
            r_.disable_depth_rendering()
        z = Z[self.sector, self.v, self.u]
        r = (z * self.scale).astype(np.float32)
        r[~np.isfinite(r) | (z >= 0.99 * float(self.model.vis.map.zfar * self.model.stat.extent))] = np.nan
        if T_world_lidar is None:
            cid = self.cams[0]
            T_world_lidar = np.eye(4)
            T_world_lidar[:3, 3] = data.cam_xpos[cid]
            # lidar frame = camera 0's forward as x, z up
            Rc = data.cam_xmat[cid].reshape(3, 3)
            fwd = -Rc[:, 2]
            up = Rc[:, 1]
            T_world_lidar[:3, :3] = np.column_stack([fwd, np.cross(up, fwd), up])
        geom = None
        sb = self.selfblock
        if sb.block.any():
            # beams whose first hit is the robot (drawn in the depth renders or not): no return. The robot hit counts
            # unless the depth range is clearly shorter (depth quantisation: 3 cm + 3 %)
            Rw = T_world_lidar[:3, :3]
            vec = self.dirs.reshape(-1, 3) @ Rw.T
            d_self = sb.distances(data, T_world_lidar[:3, 3], vec, lm.r_max).reshape(r.shape)
            d_scene = np.where(np.isfinite(r), r, np.inf)
            blocked = np.isfinite(d_self) & (d_self <= d_scene + 0.03 + 0.03 * np.minimum(d_self, 100.0))
            r[blocked] = np.nan
            geom = np.where(blocked, SELF_BLOCKED, -1).astype(np.int32)
        if self.noise:
            ok = np.isfinite(r)
            r[ok] += (self.rng.standard_normal(int(ok.sum())) * (lm.noise_a + lm.noise_b * r[ok])).astype(np.float32)
            r[self.rng.random(r.shape) < lm.dropout] = np.nan
        r[(r < lm.r_min) | (r > lm.r_max)] = np.nan
        ok = np.isfinite(r)
        pts = (self.dirs[ok] * r[ok][:, None]).astype(np.float32)
        ring = np.broadcast_to(np.arange(lm.channels)[:, None], r.shape)[ok]
        az = np.broadcast_to(np.arange(lm.beams)[None, :] * (2 * math.pi / lm.beams), r.shape)[ok]
        return LidarScan(r, pts, ring, az, T_world_lidar, geom)
