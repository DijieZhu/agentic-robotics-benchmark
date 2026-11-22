"""The crawler's sensors (cell internals): RGB-D cameras, the 16-channel lidar, the thermal camera, the odometry and
the radio link, all computed from the simulation the way the real devices measure.

Rendering: the colour images come from look.py (MuJoCo's Filament renderer on the scene's render twin); depth, lidar,
segmentation and thermal from MuJoCo's classic renderer (OSMesa in the sandbox) on the sensor twin (the physics geometry
plus the drawn-only decor, whose ids follow the physics model's), both synced to the physics state before each frame.
The physics model is never rendered. One classic renderer per image size is kept for the episode. With
RUBBLECELL_LOOK=classic (development) the colour comes from the classic renderer too.
* Cameras: colour + depth (z-depth, metres) with the device's noise, dropouts and range limits.
* Lidar: a 16-channel spinning lidar on the mast top (elevations -15 .. +15 deg, 0.5 deg azimuth steps, 0.3-25 m),
  sampled from four depth renders (90 deg sectors) at the lidar's origin; points in the lidar frame (axes of the
  chassis, origin at the lidar's optical centre).
* Thermal: a 160 x 120 radiometric thermal camera co-mounted with the mast RGB-D camera: a segmentation render maps
  every pixel to the surface it sees, each surface has its temperature (survivor's skin and clothes, a still-warm
  boiler, sun-heated metal and asphalt, everything else at the ambient temperature with a small spread), plus the
  camera's optics blur and noise; output in centi-kelvin (uint16), as radiometric cores report.
* Odometry: the robot's lidar-inertial odometry estimate in the MAP frame (the robot's pose at deployment: origin at
  its start position on the ground, x along its initial heading, z up), with a slowly drifting scale and heading bias.
* Radio link: path loss from the team's base station (at the deployment point) through the walls and slabs between.
"""

import math

import mujoco
import numpy as np

__all__ = ["Sensors", "LIDAR", "THERMAL", "CAMERAS"]

CAMERAS = {
    # name: (width, height, depth?, depth range m, depth noise a + b z^2)
    "mast": {"cam": "mast_rgbd", "w": 640, "h": 480, "depth": True, "range": (0.2, 10.0), "noise": (0.002, 0.0025)},
    "front": {"cam": "front", "w": 320, "h": 240, "depth": False},
    "rear": {"cam": "rear", "w": 320, "h": 240, "depth": False},
    "gripper": {"cam": "gripper", "w": 320, "h": 240, "depth": True, "range": (0.07, 3.0), "noise": (0.001, 0.003)},
}
LIDAR = {"channels": 16, "el_min": -15.0, "el_max": 15.0, "az_step": 0.5, "range": (0.3, 25.0), "noise": (0.01, 0.002),
         "dropout": 0.02, "img_w": 256, "img_h": 64, "fovy": 44.0}
THERMAL = {"cam": "thermal", "w": 160, "h": 120, "netd_k": 0.05, "sky_c": -5.0}
SEG_W, SEG_H = 160, 120


class Sensors(object):
    def __init__(self, scene, rng, render=True, look=None):
        self.sc = scene
        self.look = look
        self.phys_d = scene.data
        tw = look.twins if look is not None else None
        # what the classic renderer draws: the sensor twin when there is one, else the physics model itself
        self.m = tw.depth_model if tw is not None else scene.model
        self.d = tw.depth_data if tw is not None else scene.data
        self.rng = rng
        self.render_ok = render
        self._r = {}
        self.cam_ids = {}
        for k, c in CAMERAS.items():
            self.cam_ids[k] = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, c["cam"])
        self.lidar_cams = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_%d" % k) for k in range(4)]
        self.thermal_cam = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, THERMAL["cam"])
        m = self.m
        m.vis.map.znear = 0.02 / max(1.0, m.stat.extent)
        m.vis.map.zfar = 120.0 / max(1.0, m.stat.extent)
        self._temps = None
        self._survivor_geoms = None
        # what the devices see: geom groups 0-2, no sites, no frames
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0
        for g in range(3, 6):
            self.opt.geomgroup[g] = 0

    # ------------------------------------------------------------------ renderers
    def _sync(self):
        if self.look is not None:
            self.look.twins.sync(self.phys_d)

    def _renderer(self, w, h):
        key = (w, h)
        if key not in self._r:
            self._r[key] = mujoco.Renderer(self.m, h, w)
        return self._r[key]

    def close(self):
        for r in self._r.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        self._r = {}
        if self.look is not None:
            self.look.close()
            self.look = None

    def _seg(self, cam_id, w, h):
        """(object id, object type) per pixel of an id-colour render; pixels whose colour is no geom's id (the edge
        of an antialiased silhouette, the background) read (-1, -1). Fog, haze, sky, shadows and reflections off."""
        r = self._renderer(w, h)
        r.update_scene(self.d, camera=cam_id, scene_option=self.opt)
        sc = r.scene
        flags0 = sc.flags.copy()
        F = mujoco.mjtRndFlag
        sc.flags[F.mjRND_SEGMENT] = 1
        sc.flags[F.mjRND_IDCOLOR] = 1
        for f in (F.mjRND_FOG, F.mjRND_HAZE, F.mjRND_SKYBOX, F.mjRND_SHADOW, F.mjRND_REFLECTION):
            sc.flags[f] = 0
        try:
            if r._gl_context:
                r._gl_context.make_current()
            mujoco.mjr_render(r._rect, sc, r._mjr_context)
            out = np.empty((h, w, 3), np.uint8)
            mujoco.mjr_readPixels(out, None, r._rect, r._mjr_context)
        finally:
            np.copyto(sc.flags, flags0)
        if r._gl_context:
            out = np.flipud(out)
        img = out.astype(np.int64)
        segimg = img[..., 0] + img[..., 1] * 256 + img[..., 2] * 65536
        ng = sc.ngeom
        lut = np.full((ng + 2, 2), -1, np.int32)
        for k in range(ng):
            g = sc.geoms[k]
            if g.segid != -1 and g.segid + 1 <= ng:
                lut[g.segid + 1, 0] = g.objid
                lut[g.segid + 1, 1] = g.objtype
        segimg = np.where((segimg >= 0) & (segimg <= ng), segimg, 0)
        return lut[segimg]

    def _render(self, cam_id, w, h, mode):
        if mode == "seg":
            return self._seg(cam_id, w, h)
        r = self._renderer(w, h)
        if mode == "depth":
            r.enable_depth_rendering()
        elif mode == "seg":
            r.enable_segmentation_rendering()
        try:
            r.update_scene(self.d, camera=cam_id, scene_option=self.opt)
            # fog would tint the far pixels of an id-colour (segmentation) render into ids that do not exist
            fog = 0 if mode in ("seg", "depth") else 1
            r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = fog
            r.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = fog
            # the skybox gets an id one past the last geom in an id-colour render: no sky there (it is background)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = fog
            out = r.render()
        finally:
            if mode == "depth":
                r.disable_depth_rendering()
            elif mode == "seg":
                r.disable_segmentation_rendering()
        return out.copy()

    # ------------------------------------------------------------------ cameras
    def camera(self, name, depth=True):
        """(rgb uint8, depth float32 metres with NaN where invalid or None, intrinsics, T_chassis_cam 4x4)."""
        c = CAMERAS[name]
        cid = self.cam_ids[name]
        self._sync()
        if self.look is not None:
            rgb = self.look.rgb(self.phys_d, c["cam"], c["w"], c["h"])
        else:
            rgb = self._render(cid, c["w"], c["h"], "rgb")
        sig = 2.0
        rgb = np.clip(rgb.astype(np.float32) + self.rng.normal(0.0, sig, rgb.shape), 0, 255).astype(np.uint8)
        dep = None
        if depth and c.get("depth"):
            z = self._render(cid, c["w"], c["h"], "depth").astype(np.float32)
            a, b = c["noise"]
            z = z + self.rng.normal(0.0, 1.0, z.shape).astype(np.float32) * (a + b * z * z)
            lo, hi = c["range"]
            bad = ~np.isfinite(z) | (z < lo) | (z > hi) | (self.rng.random(z.shape) < 0.01)
            z[bad] = np.nan
            dep = z
        return rgb, dep, self.intrinsics(name), self.cam_extrinsics(cid)

    def intrinsics(self, name):
        c = CAMERAS[name] if name in CAMERAS else {"w": THERMAL["w"], "h": THERMAL["h"]}
        cid = self.cam_ids.get(name, self.thermal_cam)
        fovy = float(self.m.cam_fovy[cid])
        fy = (c["h"] / 2.0) / math.tan(math.radians(fovy) / 2.0)
        return {"width": c["w"], "height": c["h"], "fx": fy, "fy": fy, "cx": (c["w"] - 1) / 2.0, "cy": (c["h"] - 1) / 2.0,
                "fovy_deg": fovy, "model": "pinhole; optical frame x right, y down, z forward"}

    def cam_extrinsics(self, cid):
        """4x4 pose of the camera's OPTICAL frame (x right, y down, z forward) in the chassis frame."""
        d = self.phys_d
        ch = self.sc.info["chassis_body"]
        Rw_ch = d.xmat[ch].reshape(3, 3)
        pw_ch = d.xpos[ch]
        Rw_c = d.cam_xmat[cid].reshape(3, 3)
        pw_c = d.cam_xpos[cid]
        # MuJoCo camera frame: x right, y up, z backward -> optical: x right, y down, z forward
        R_opt = Rw_c @ np.diag([1.0, -1.0, -1.0])
        T = np.eye(4)
        T[:3, :3] = Rw_ch.T @ R_opt
        T[:3, 3] = Rw_ch.T @ (pw_c - pw_ch)
        return T

    # ------------------------------------------------------------------ lidar
    def lidar(self):
        """(N, 4) float32: x, y, z in the lidar frame (axes of the chassis) and the channel index; plus the lidar's
        pose in the chassis frame."""
        L = LIDAR
        els = np.radians(np.linspace(L["el_min"], L["el_max"], L["channels"]))
        azs = np.radians(np.arange(0.0, 360.0, L["az_step"]))
        w, h = L["img_w"], L["img_h"]
        fy = (h / 2.0) / math.tan(math.radians(L["fovy"]) / 2.0)
        fx = fy
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        pts = []
        self._sync()
        for k, cid in enumerate(self.lidar_cams):
            z = self._render(cid, w, h, "depth").astype(np.float64)
            lo = math.radians(k * 90.0 - 45.0)
            hi = math.radians(k * 90.0 + 45.0)
            sel = azs[(azs >= lo % (2 * math.pi)) & (azs < hi % (2 * math.pi))] if k > 0 else azs[(azs >= 2 * math.pi + lo) | (azs < hi)]
            rel = (sel - math.radians(k * 90.0) + math.pi) % (2 * math.pi) - math.pi
            A, E = np.meshgrid(rel, els)
            xc = -np.tan(A)
            yc = np.tan(E) / np.cos(A)
            u = np.clip(np.round(cx + fx * xc).astype(int), 0, w - 1)
            v = np.clip(np.round(cy - fy * yc).astype(int), 0, h - 1)
            depth = z[v, u]
            rng_ = depth * np.sqrt(1.0 + xc * xc + yc * yc)
            AZ = A + math.radians(k * 90.0)
            dirs = np.stack([np.cos(E) * np.cos(AZ), np.cos(E) * np.sin(AZ), np.sin(E)], -1)
            ch = np.broadcast_to(np.arange(L["channels"])[:, None], A.shape)
            pts.append((rng_, dirs, ch))
        rng_ = np.concatenate([p[0].ravel() for p in pts])
        dirs = np.concatenate([p[1].reshape(-1, 3) for p in pts])
        ch = np.concatenate([p[2].ravel() for p in pts])
        a, b = L["noise"]
        rng_ = rng_ + self.rng.normal(0.0, 1.0, rng_.shape) * (a + b * rng_)
        lo, hi = L["range"]
        ok = np.isfinite(rng_) & (rng_ >= lo) & (rng_ <= hi) & (self.rng.random(rng_.shape) >= L["dropout"])
        P = dirs[ok] * rng_[ok, None]
        out = np.concatenate([P, ch[ok, None]], 1).astype(np.float32)
        return out, self.lidar_pose()

    def lidar_pose(self):
        """4x4 pose of the lidar frame (chassis axes, origin at the optical centre) in the chassis frame."""
        d = self.phys_d
        ch = self.sc.info["chassis_body"]
        cid = self.lidar_cams[0]
        Rw_ch = d.xmat[ch].reshape(3, 3)
        T = np.eye(4)
        T[:3, 3] = Rw_ch.T @ (d.cam_xpos[cid] - d.xpos[ch])
        return T

    # ------------------------------------------------------------------ thermal
    def set_temperatures(self, temps, survivor_geoms, outdoor_c=None):
        """Per-geom temperatures of the physics model; the sensor twin's drawn-only decor (ids past the physics
        model's, all of it outdoors) at `outdoor_c`."""
        temps = np.asarray(temps, np.float32)
        if self.m.ngeom > len(temps):
            pad = float(outdoor_c) if outdoor_c is not None else float(np.median(temps))
            temps = np.concatenate([temps, np.full(self.m.ngeom - len(temps), pad, np.float32)])
        self._temps = temps
        self._survivor_geoms = np.asarray(sorted(survivor_geoms), np.int32)

    def thermal(self):
        """(centi-kelvin uint16 H x W, survivor pixel count) of the thermal camera."""
        self._sync()
        seg = self._render(self.thermal_cam, THERMAL["w"], THERMAL["h"], "seg")
        ids = seg[..., 0].astype(np.int64)
        typ = seg[..., 1].astype(np.int64)
        T = np.full(ids.shape, THERMAL["sky_c"], np.float32)
        geom = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        T[geom] = self._temps[ids[geom]]
        # optics: a 3x3 blur, then the sensor's noise (NETD) and a faint vignette
        Tp = np.pad(T, 1, mode="edge")
        T = sum(Tp[1 + dy:1 + dy + T.shape[0], 1 + dx:1 + dx + T.shape[1]] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
        T = T + self.rng.normal(0.0, THERMAL["netd_k"], T.shape).astype(np.float32)
        yy, xx = np.mgrid[0:T.shape[0], 0:T.shape[1]]
        r2 = ((xx - T.shape[1] / 2) / T.shape[1]) ** 2 + ((yy - T.shape[0] / 2) / T.shape[0]) ** 2
        T = T - 0.6 * r2.astype(np.float32)
        ck = np.clip(np.round((T + 273.15) * 100.0), 0, 65535).astype(np.uint16)
        n_surv = int(np.isin(ids[geom], self._survivor_geoms).sum()) if self._survivor_geoms is not None else 0
        return ck, n_surv

    def survivor_pixels(self, name):
        """How many pixels of the survivor a low-resolution segmentation of camera `name` shows (the visibility record)."""
        if self._survivor_geoms is None:
            return 0
        self._sync()
        seg = self._render(self.cam_ids[name], SEG_W, SEG_H, "seg")
        ids = seg[..., 0].astype(np.int64)
        typ = seg[..., 1].astype(np.int64)
        geom = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        return int(np.isin(ids[geom], self._survivor_geoms).sum())


class Odometry(object):
    """Lidar-inertial odometry in the map frame: integrates the true planar motion with a scale error and a heading
    bias that drift slowly (about 1 % of the distance travelled, a few degrees over a mission)."""

    def __init__(self, rng, start_xy, start_yaw):
        self.rng = rng
        self.x0, self.y0, self.yaw0 = float(start_xy[0]), float(start_xy[1]), float(start_yaw)
        self.est = np.zeros(3)                          # x, y, yaw in the map frame
        self.scale = 1.0 + float(rng.normal(0.0, 0.006))
        self.bias = float(rng.normal(0.0, math.radians(0.01)))     # rad per second
        self.prev = None

    def to_map(self, xw, yw, yaww):
        c, s = math.cos(-self.yaw0), math.sin(-self.yaw0)
        dx, dy = xw - self.x0, yw - self.y0
        return c * dx - s * dy, s * dx + c * dy, yaww - self.yaw0

    def to_world(self, xm, ym):
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        return self.x0 + c * xm - s * ym, self.y0 + s * xm + c * ym

    def update(self, xw, yw, yaww, dt):
        cur = np.array(self.to_map(xw, yw, yaww))
        if self.prev is None:
            self.prev = cur
            self.est = cur.copy()
            return
        d = cur - self.prev
        self.prev = cur
        dyaw = (d[2] + math.pi) % (2 * math.pi) - math.pi
        # motion in the true heading's frame, re-applied along the estimated heading with the scale error
        c, s = math.cos(cur[2] - dyaw), math.sin(cur[2] - dyaw)
        fwd = c * d[0] + s * d[1]
        lat = -s * d[0] + c * d[1]
        self.scale += float(self.rng.normal(0.0, 0.0004)) * math.sqrt(abs(fwd) + 1e-9)
        yaw_e = self.est[2]
        ce, se = math.cos(yaw_e), math.sin(yaw_e)
        self.est[0] += self.scale * (ce * fwd - se * lat)
        self.est[1] += self.scale * (se * fwd + ce * lat)
        self.est[2] += dyaw * (1.0 + 0.004) + self.bias * dt + float(self.rng.normal(0.0, 1e-5))
