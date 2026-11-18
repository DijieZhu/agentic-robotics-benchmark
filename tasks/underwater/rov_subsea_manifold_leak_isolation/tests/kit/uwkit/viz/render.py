"""Replay rendering of a SimLog on a VisualTwin: cameras, rotor display modes, motion blur, wake, overlays, MP4.

Rotor display modes (see rotors.py for why):
  'limited'    display angle from the speed-limited map (<= 40 deg / frame at 30 fps), with a short fading trail behind
               each blade over the last frame's rotation (shows the direction); recommended for people
  'true'       true angle sampled at the frame time, no blur (shows the stroboscopic aliasing a 30 fps camera would see)
  'true_blur'  true angle + motion blur: the blades are drawn K times over the angle swept during the shutter
               (shutter = fraction of the frame interval), each copy translucent; slow rotors stay crisp
  'accumulate' true angle + temporal super-sampling: the whole frame is rendered at `subsamples` instants inside the
               shutter and averaged (reference for 'true_blur'; costs `subsamples` renders per frame)
One mujoco.Renderer per process (OSMesa); several videos of the same size can share it (render_jobs).
"""
from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco
from PIL import Image

from . import overlay as OV
from .rotors import DisplaySpeed, integrate_angles, rpm_reference
from .wake import Wake, WakeStyle

_GEOM_FIELDS_ARR = ("size", "pos", "mat", "rgba", "texrepeat")
_GEOM_FIELDS = ("type", "dataid", "objtype", "objid", "category", "matid", "texcoord", "segid", "emission", "specular",
                "shininess", "reflectance", "transparent", "texid", "texuniform", "modelrbound", "camdist", "label")
# 'label' matters: scene slots beyond the ones mjv_updateScene wrote hold whatever the heap held, and a non-empty label
# is drawn as text by mjr_render (seen as rare, history-dependent differences between identical replays)


def rodrigues(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def _clone(src, dst):
    for f in _GEOM_FIELDS:
        if hasattr(src, f):
            try:
                setattr(dst, f, getattr(src, f))
            except Exception:
                pass
    for f in _GEOM_FIELDS_ARR:
        if hasattr(src, f):
            np.copyto(getattr(dst, f), getattr(src, f))


# ---------------------------------------------------------------------------------------------------- cameras
@dataclass
class FollowCamera:
    """Free camera following the vehicle: lookat = low-passed hull position (+ offset in B), azimuth = low-passed
    heading + azimuth offset (camera position direction = azimuth + 180 deg), fixed elevation, world up."""
    distance: float = 1.35
    azimuth: float = 215.0            # deg, relative to the vehicle heading (215 = camera at the front-left)
    elevation: float = -16.0
    yaw_tau: float = 2.0              # s; large = the camera keeps its direction while the vehicle turns
    pos_tau: float = 0.25
    lookat_B: tuple = (0.03, 0.0, -0.03)
    name: str = "follow"

    def reset(self):
        self._yaw = None
        self._p = None

    def update(self, dt, hull_pos, hull_R):
        yaw = math.atan2(hull_R[1, 0], hull_R[0, 0]) if abs(hull_R[2, 0]) < 0.95 else (self._yaw or 0.0)
        target = np.asarray(hull_pos, float) + hull_R @ np.asarray(self.lookat_B, float)
        if getattr(self, "_yaw", None) is None:
            self._yaw, self._p = yaw, target.copy()
        else:
            a = 1.0 - math.exp(-dt / max(self.yaw_tau, 1e-6))
            self._yaw += a * math.atan2(math.sin(yaw - self._yaw), math.cos(yaw - self._yaw))
            b = 1.0 - math.exp(-dt / max(self.pos_tau, 1e-6))
            self._p = self._p + b * (target - self._p)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self._p
        cam.distance = self.distance
        cam.azimuth = math.degrees(self._yaw) + self.azimuth
        cam.elevation = self.elevation
        return cam


@dataclass
class FixedCamera:
    """A camera of the twin model (e.g. 'viz_cam_thr_3')."""
    name: str

    def reset(self):
        pass

    def update(self, dt, hull_pos, hull_R, model=None):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = self._id
        return cam

    def bind(self, model):
        self._id = model.camera(self.name).id
        return self


# ---------------------------------------------------------------------------------------------------- frame state
@dataclass
class FrameState:
    row: int
    phi: np.ndarray                     # displayed rotor angles
    sweep: np.ndarray                   # signed blur / trail span per rotor (rad), 0 = crisp
    weights: str                        # 'uniform' (motion blur) | 'trail' (fading trail)


class AngleSource:
    """Display angles and blur spans for every frame row of a log."""

    def __init__(self, log, spin, fps=30.0, mode="limited", shutter=0.5, display: Optional[DisplaySpeed] = None,
                 trail=0.85):
        self.log, self.mode, self.fps = log, mode, float(fps)
        self.rows_per_frame = max(1, int(round(log.log_hz / fps)))
        if display is None:
            # frames sit on the nearest log row; with 500 Hz rows and 30 fps a frame interval spans up to 17 rows
            # (34 ms) instead of 16.67 -> shrink the cap by 16.67 / 17 so that no frame step exceeds 40 deg
            k = log.log_hz / float(fps)
            display = DisplaySpeed(fps=fps, rpm_ref=rpm_reference(), max_deg_per_frame=40.0 * k / math.ceil(k - 1e-9))
        self.display = display
        dt = 1.0 / log.log_hz
        self.phi_true = log.phi_true
        self.phi_lim = integrate_angles(log.rpm, dt, spin, self.display)
        self.shutter_rows = max(1, int(round(shutter * self.rows_per_frame)))
        self.shutter = shutter
        self.trail = trail

    def at(self, row):
        if self.mode == "limited":
            p = self.phi_lim
            j = max(0, row - self.rows_per_frame)
            return FrameState(row, p[row], self.trail * (p[row] - p[j]), "trail")
        if self.mode == "true":
            return FrameState(row, self.phi_true[row], np.zeros(self.phi_true.shape[1]), "uniform")
        if self.mode in ("true_blur", "accumulate"):
            j = max(0, row - self.shutter_rows)
            return FrameState(row, self.phi_true[row], self.phi_true[row] - self.phi_true[j], "uniform")
        raise ValueError(self.mode)


# ---------------------------------------------------------------------------------------------------- renderer
class TwinRenderer:
    def __init__(self, twin, width=1920, height=1080, max_geom=20000, fog=True, blur_step_deg=7.0, blur_max_copies=36,
                 blur_coverage=0.8, extra_geomgroups=()):
        """extra_geomgroups: geom groups to show besides MuJoCo's default 0-2 (e.g. task scenery in groups 4 / 5);
        group 3 (vehicle collision) stays hidden unless listed."""
        self.twin = twin
        self.w, self.h = int(width), int(height)
        self.r = mujoco.Renderer(twin.model, self.h, self.w, max_geom=max_geom)
        self.opt = mujoco.MjvOption()
        for k in range(6):
            self.opt.sitegroup[k] = 0
        self.opt.geomgroup[3] = 0
        for k in extra_geomgroups:
            self.opt.geomgroup[int(k)] = 1
        self.fog = fog
        self.blur_step = math.radians(blur_step_deg)
        self.blur_max = int(blur_max_copies)
        self.blur_cov = float(blur_coverage)
        self.timing = {}

    def close(self):
        from ..sensors.camera import close_renderer
        close_renderer(self.r)               # frees the GL objects in this renderer's own context

    def _t(self, key, t0):
        self.timing.setdefault(key, []).append(time.perf_counter() - t0)

    def scene(self, camera):
        r = self.r
        r.update_scene(self.twin.data, camera=camera, scene_option=self.opt)
        scn = r.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1 if self.fog else 0
        return scn

    def _scene_index(self, scn):
        idx = {}
        G = int(mujoco.mjtObj.mjOBJ_GEOM)
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype == G:
                idx[int(g.objid)] = i
        return idx

    def inject_rotor_copies(self, scn, fs: FrameState, idx=None, only=None):
        """Blur (uniform weights) or trail (fading weights) copies of the blades and marker of every rotor."""
        tw = self.twin
        idx = idx if idx is not None else self._scene_index(scn)
        axes = tw.rotor_axes_world()
        ctrs = tw.rotor_centres_world()
        added = 0
        for k, gids in enumerate(tw.rotor_geom_ids):
            if only is not None and k not in only:
                continue
            sweep = float(fs.sweep[k])
            if abs(sweep) < 0.5 * self.blur_step:
                continue
            K = int(min(self.blur_max, math.ceil(abs(sweep) / self.blur_step) + 1))
            if fs.weights == "uniform":
                a = 1.0 - (1.0 - self.blur_cov) ** (1.0 / K)
                alphas = np.full(K, a)
            else:
                alphas = np.linspace(1.0, 0.12, K) ** 1.5
                alphas[0] = 1.0
            for gid in gids[1:]:                                    # blades, marker (the hub is axisymmetric)
                i = idx.get(gid)
                if i is None:
                    continue
                src = scn.geoms[i]
                p0 = np.array(src.pos, float)
                M0 = np.array(src.mat, float).reshape(3, 3)
                base_rgba = np.array(src.rgba, float)
                src.rgba[3] = base_rgba[3] * alphas[0]
                if alphas[0] < 1:
                    src.transparent = 1
                for j in range(1, K):
                    if scn.ngeom >= scn.maxgeom:
                        break
                    R = rodrigues(axes[k], -sweep * j / (K - 1))
                    g = scn.geoms[scn.ngeom]
                    _clone(src, g)
                    g.pos[:] = ctrs[k] + R @ (p0 - ctrs[k])
                    g.mat[:] = (R @ M0).astype(np.float32)
                    g.rgba[3] = base_rgba[3] * alphas[j]
                    g.transparent = 1
                    scn.ngeom += 1
                    added += 1
        return added

    def ghost(self, scn, keep_ids, alpha=0.10, idx=None):
        """Make every vehicle geom translucent except keep_ids (close-ups through the frame)."""
        keep = set(int(x) for x in keep_ids)
        G = int(mujoco.mjtObj.mjOBJ_GEOM)
        veh = self.twin._veh_set if hasattr(self.twin, "_veh_set") else set(int(x) for x in self.twin.vehicle_geom_ids)
        self.twin._veh_set = veh
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype == G and int(g.objid) in veh and int(g.objid) not in keep:
                g.rgba[3] = min(float(g.rgba[3]), alpha)
                g.transparent = 1

    @staticmethod
    def compact(scn, keep_objids):
        """Keep only the scene geoms of the given model geoms (and their injected copies, which share the objid)."""
        keep = set(int(x) for x in keep_objids)
        G = int(mujoco.mjtObj.mjOBJ_GEOM)
        j = 0
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype == G and int(g.objid) in keep:
                if i != j:
                    _clone(g, scn.geoms[j])
                j += 1
        scn.ngeom = j

    @staticmethod
    def _close_up_light(scn, ambient=0.42, diffuse=0.75):
        """Brighter headlight for close-ups (the nozzle and propeller are black plastic)."""
        if scn.nlight > 0 and scn.lights[0].headlight:
            scn.lights[0].ambient[:] = ambient
            scn.lights[0].diffuse[:] = diffuse

    def _render_rect(self, w, h):
        """Render the current scene into the lower-left w x h corner of the (shared) offscreen buffer.  Uses the
        Renderer's GL / mjr contexts, so one Renderer per process still serves several image sizes."""
        r = self.r
        if r._gl_context:
            r._gl_context.make_current()
        rect = mujoco.MjrRect(0, 0, int(w), int(h))
        mujoco.mjr_render(rect, r.scene, r._mjr_context)
        out = np.empty((int(h), int(w), 3), np.uint8)
        mujoco.mjr_readPixels(out, None, rect, r._mjr_context)
        return np.flipud(out) if r._gl_context else out

    def render_isolated(self, k, fs: FrameState, w, h, camera=None):
        """Close-up of thruster k alone (stator + rotor) from its nose-side camera, w x h pixels.
        Call after render() / forward() of the same frame (the twin is already posed)."""
        tw = self.twin
        t0 = time.perf_counter()
        tw.set_rotor_angles(fs.phi)
        tw.forward()
        if camera is None:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = tw.cameras[tw.rotors.sites[k]]
        scn = self.scene(camera)
        idx = self._scene_index(scn)
        self.inject_rotor_copies(scn, fs, idx, only=[k])
        self.compact(scn, list(tw.rotor_geom_ids[k]) + [tw.stator_geom_ids[k]])
        self._close_up_light(scn)
        img = self._render_rect(w, h)
        self._t("inset", t0)
        return img

    def render(self, camera, fs: FrameState, wake: Optional[Wake] = None, water_w=(0, 0, 0), ghost_keep=None,
               wake_look=None):
        t0 = time.perf_counter()
        self.twin.set_rotor_angles(fs.phi)
        self.twin.forward()
        scn = self.scene(camera)
        idx = self._scene_index(scn)
        if ghost_keep is not None:
            self.ghost(scn, ghost_keep, idx=idx)
            self._close_up_light(scn)
        self.inject_rotor_copies(scn, fs, idx)
        if wake is not None:
            wake.draw(scn, water_w, **(wake_look or {}))
        self._t("scene", t0)
        t1 = time.perf_counter()
        img = self.r.render()
        self._t("render", t1)
        return img


# ---------------------------------------------------------------------------------------------------- video output
class VideoWriter:
    """H.264 MP4 through ffmpeg's stdin (no temporary frames on disk)."""

    def __init__(self, path, width, height, fps=30, crf=20, preset="medium"):
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = "ffmpeg"
        self.path = path
        self.p = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                   "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                   path], stdin=subprocess.PIPE)
        self.n = 0

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img, np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}")


# ---------------------------------------------------------------------------------------------------- log helpers
def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def hull_pose_from_log(log, row):
    t, a, nq = log.meta["joints"][log.meta["hull_joint"]]
    q = log.qpos[row, a:a + 7]
    return q[:3].copy(), quat_to_R(q[3:7] / np.linalg.norm(q[3:7]))


def rpy_from_R(R):
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def default_caption(log, row, extra=""):
    p, R = hull_pose_from_log(log, row)
    v_b = R.T @ log.qvel6[row, :3]
    roll, pitch, yaw = (math.degrees(x) for x in rpy_from_R(R))
    lines = [f"{log.t[row]:5.1f} s   {log.labels_at(row)}",
             f"深度 {-p[2]:.2f} m   u {v_b[0]:+.2f}  v {v_b[1]:+.2f}  w {v_b[2]:+.2f} m/s   "
             f"艏向 {yaw:+.1f}°  横滚 {roll:+.1f}°  纵倾 {pitch:+.1f}°"]
    if extra:
        lines.append(extra)
    return lines


MODE_NOTE = {
    "limited": "桨的显示转速按比例压到 ≤ 40°/帧（方向与快慢顺序不变）",
    "true": "桨按真实转角显示（30 fps 下会出现倒转 / 静止错觉）",
    "true_blur": "桨按真实转角 + 运动模糊（快门 1/2 帧）",
    "accumulate": "桨按真实转角 + 时间超采样模糊",
}


# ---------------------------------------------------------------------------------------------------- main loop
def render_video(log, twin, path, camera=None, mode="limited", wake=False, width=1920, height=1080, fps=30, insets=0,
                 overlay=True, **kw):
    """One video from a log (convenience wrapper of render_jobs)."""
    job = VideoJob(path, camera or FollowCamera(), mode, wake, overlay, insets=insets)
    return render_jobs(log, twin, [job], width, height, fps, **kw)


@dataclass
class VideoJob:
    path: str
    camera: object
    mode: str = "limited"
    wake: bool = False
    overlay: bool = True
    keyframes: tuple = ()                 # times [s] collected for a contact sheet
    sheet_path: Optional[str] = None
    sheet_title: Optional[str] = None
    caption_extra: str = ""
    insets: int = 0                       # > 0: tile size [px] of the 8 per-thruster close-ups at the right edge
    wake_look: dict = field(default_factory=dict)   # Wake.draw look overrides for this view (size_scale, rgb, ...)
    collect: bool = False                 # keep every frame in memory (tests); path=None writes no video
    frames: list = field(default_factory=list)
    frames_all: list = field(default_factory=list)


def render_jobs(log, twin, jobs, width=1920, height=1080, fps=30, shutter=0.5, display=None, wake_seed=0,
                wake_style=None, floor_z=None, t_range=None, subsamples=8, renderer=None, progress_every=150,
                crf=20, log_prefix=""):
    """Render several videos (same size) from one log with one Renderer.  Returns a timing summary."""
    own = renderer is None
    rend = renderer or TwinRenderer(twin, width, height)
    rows_all = log.rows_for_fps(fps)
    # frames outside t_range are not rendered, but the cameras' filters and the wake still run through them, so that
    # chunks rendered in separate processes join seamlessly (concatenate the MP4s)
    active = np.ones(len(rows_all), bool) if t_range is None else \
        (log.t[rows_all] >= t_range[0] - 1e-9) & (log.t[rows_all] <= t_range[1] + 1e-9)
    rows = rows_all[active]
    spin = twin.rotors.spin
    srcs = {m: AngleSource(log, spin, fps, m, shutter, display) for m in {j.mode for j in jobs}}
    rpm_ref = (srcs[next(iter(srcs))].display.rpm_ref)
    need_wake = any(j.wake for j in jobs)
    wake = Wake(twin.rotors, seed=wake_seed, style=wake_style, floor_z=floor_z) if need_wake else None
    for j in jobs:
        j.camera.reset()
        if isinstance(j.camera, FixedCamera):
            j.camera.bind(twin.model)
    writers = [VideoWriter(j.path, rend.w, rend.h, fps, crf) if j.path else None for j in jobs]
    keyrows = {}
    for jb in jobs:
        for tk in jb.keyframes:
            keyrows.setdefault(int(np.argmin(np.abs(log.t[rows] - tk))), []).append(jb)
    joints = log.meta["joints"]
    dt_row = 1.0 / log.log_hz
    t_start = time.perf_counter()
    wake_row = 0
    tm = {"state": [], "overlay": [], "encode": [], "wake": []}
    fi = -1
    for ai, row in enumerate(rows_all):
        if not active[ai]:
            p, R = hull_pose_from_log(log, row)
            for jb in jobs:
                jb.camera.update(1.0 / fps, p, R)
            continue
        fi += 1
        t0 = time.perf_counter()
        if wake is not None:                                   # integrate the wake row by row up to this frame
            while wake_row < row:
                p, R = hull_pose_from_log(log, wake_row)
                q6 = log.qvel6[wake_row]
                wake.advance(dt_row, p, R, q6[:3], q6[3:], log.thrust[wake_row], log.water_w[wake_row])
                wake_row += 1
        tm["wake"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        twin.set_physics_qpos(log.qpos[row], joints)
        twin.set_lamps(log.lights[row])
        p, R = hull_pose_from_log(log, row)
        tm["state"].append(time.perf_counter() - t0)
        for jb, wr in zip(jobs, writers):
            cam = jb.camera.update(1.0 / fps, p, R)
            fs = srcs[jb.mode].at(row)
            if jb.mode == "accumulate":
                img = _accumulate(rend, log, twin, cam, row, srcs[jb.mode], subsamples, joints,
                                  wake if jb.wake else None)
            else:
                img = rend.render(cam, fs, wake if jb.wake else None, log.water_w[row], wake_look=jb.wake_look)
            ins = [rend.render_isolated(k, fs, jb.insets, jb.insets) for k in range(twin.rotors.n)] if jb.insets else None
            t1 = time.perf_counter()
            if jb.overlay:
                img = draw_overlay(img, log, row, twin, jb, rpm_ref, ins)
            tm["overlay"].append(time.perf_counter() - t1)
            t1 = time.perf_counter()
            if wr is not None:
                wr.write(img)
            if jb.collect:
                jb.frames_all.append(np.array(img))
            tm["encode"].append(time.perf_counter() - t1)
            if fi in keyrows and jb in keyrows[fi]:
                jb.frames.append((float(log.t[row]), log.labels_at(row), np.array(img)))
        if progress_every and fi % progress_every == 0:
            el = time.perf_counter() - t_start
            print(f"{log_prefix}frame {fi + 1}/{len(rows)}  t {log.t[row]:.1f} s  {el / (fi + 1):.2f} s/frame "
                  f"(all jobs)", flush=True)
    for wr in writers:
        if wr is not None:
            wr.close()
    for jb in jobs:
        if jb.sheet_path and jb.frames:
            sh = OV.contact_sheet([f[2] for f in jb.frames], [f"{f[0]:.1f} s  {f[1]}" for f in jb.frames], cols=3,
                                  scale=0.5, title=jb.sheet_title)
            sh.save(jb.sheet_path)
    if own:
        rend.close()
    T = {k: float(np.mean(v)) * 1e3 for k, v in list(rend.timing.items()) + list(tm.items()) if len(v)}
    T["frames"] = int(len(rows))
    T["jobs"] = len(jobs)
    T["wall_s"] = time.perf_counter() - t_start
    T["per_frame_all_jobs_s"] = T["wall_s"] / max(1, len(rows))
    return T


def _accumulate(rend, log, twin, cam, row, src, n, joints, wake):
    """Temporal super-sampling: n renders inside the shutter window (vehicle pose and rotor angle both move)."""
    j0 = max(0, row - src.shutter_rows)
    acc = None
    for s in range(n):
        x = (s + 0.5) / n
        rr = j0 + x * (row - j0)
        i0 = int(math.floor(rr))
        i1 = min(i0 + 1, row)
        a = rr - i0
        q = (1 - a) * log.qpos[i0] + a * log.qpos[i1]
        twin.set_physics_qpos(q, joints)
        phi = (1 - a) * src.phi_true[i0] + a * src.phi_true[i1]
        fs = FrameState(row, phi, np.zeros_like(phi), "uniform")
        img = rend.render(cam, fs, wake, log.water_w[row]).astype(np.float32)
        acc = img if acc is None else acc + img
    twin.set_physics_qpos(log.qpos[row], joints)
    return np.clip(acc / n + 0.5, 0, 255).astype(np.uint8)


def draw_insets(im, insets, twin, log, row, mode, x0=None, y0=None, tile=200, cols=2, title=None):
    """Paste the per-thruster close-ups (seen from the nose cone) with the model's expected spin arrow."""
    from PIL import ImageDraw
    n = len(insets)
    rows_ = int(math.ceil(n / cols))
    W, H = im.size
    pad, head = 6, 34
    pw = cols * (tile + pad) + pad
    ph = head + rows_ * (tile + pad) + pad
    x0 = W - pw - 16 if x0 is None else x0
    y0 = 16 if y0 is None else y0
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=8, fill=OV.PANEL_RGBA)
    d.text((x0 + 10, y0 + 8), title or "8 个桨特写（从鼻锥一侧看；箭头 = 模型给出的应有转向）", font=OV.font(15),
           fill=OV.DIM_RGB)
    for k, tim in enumerate(insets):
        cx, cy = x0 + pad + (k % cols) * (tile + pad), y0 + head + (k // cols) * (tile + pad)
        im.paste(Image.fromarray(tim), (cx, cy))
        r = twin.rotors.rotors[k]
        rpm = float(log.rpm[row, k])
        sense = int(np.sign(r.spin_sign_forward * rpm)) if abs(rpm) > 1e-6 else 0
        col = OV.FWD_RGB if rpm > 0 else (OV.REV_RGB if rpm < 0 else OV.DIM_RGB)
        d.rectangle([cx, cy, cx + tile, cy + 24], fill=(0, 0, 0, 120))
        d.text((cx + 6, cy + 2), f"{r.site[4:]} {'H' if abs(r.axis_B[2]) < 0.7 else 'V'} {r.propeller}  {rpm:+5.0f}",
               font=OV.font(16), fill=OV.TXT_RGB)
        OV.spin_glyph(d, (cx + tile - 22, cy + tile - 22), 13, sense, col, width=3)
    return pw, ph


def draw_overlay(img, log, row, twin, job, rpm_ref, insets=None):
    im = Image.fromarray(img)
    W, H = im.size
    sc = H / 1080.0
    if insets:
        draw_insets(im, insets, twin, log, row, job.mode, tile=insets[0].shape[1])
    extra = MODE_NOTE.get(job.mode, "") + ("；尾流可视化：开（仅演示）" if job.wake else "")
    OV.caption(im, default_caption(log, row, extra + (("；" + job.caption_extra) if job.caption_extra else "")),
               xy=(int(24 * sc), int(20 * sc)), sizes=(max(14, int(30 * sc)), max(11, int(19 * sc))))
    h = 40 + twin.rotors.n * 26 + 10
    OV.rpm_bars(im, log.rpm[row], twin.rotors, (int(20 * sc), H - h - int(20 * sc)), rpm_ref=rpm_ref)
    OV.thrust_diagram(im, log.thrust[row], twin.rotors, (int(20 * sc) + 540, H - 290 - int(20 * sc)), size=260)
    return np.asarray(im)


def render_thruster_tiles(log, twin, path, mode="limited", tile=(480, 540), cols=4, fps=30, shutter=0.5, display=None,
                          t_range=None, keyframes=(), sheet_path=None, sheet_title=None, progress_every=150,
                          caption=None, crf=20, log_prefix="", renderer=None, isolate=False, subsamples=8):
    """Close-up video: one tile per thruster, each from its body-fixed camera on the nose-cone side (rest of the vehicle
    translucent), labelled with the command, rotor speed, thrust and the expected spin seen from the nose."""
    own = renderer is None
    rend = renderer or TwinRenderer(twin, tile[0], tile[1])
    rows = log.rows_for_fps(fps)
    if t_range is not None:
        rows = rows[(log.t[rows] >= t_range[0] - 1e-9) & (log.t[rows] <= t_range[1] + 1e-9)]
    src = AngleSource(log, twin.rotors.spin, fps, mode, shutter, display)
    n = twin.rotors.n
    nrow = int(math.ceil(n / cols))
    W, H = cols * tile[0], nrow * tile[1]
    head = 0
    wr = VideoWriter(path, W, H, fps, crf)
    cams = [FixedCamera(f"viz_cam_{s}").bind(twin.model) for s in twin.rotors.sites]
    keep = [twin.rotor_geom_ids[k] + [twin.stator_geom_ids[k]] for k in range(n)]
    joints = log.meta["joints"]
    frames = []
    keyset = {int(np.argmin(np.abs(log.t[rows] - tk))) for tk in keyframes}
    t_start = time.perf_counter()
    note = {"limited": "（显示限速）", "true": "（真实转角）", "true_blur": "（真实 + 模糊）", "accumulate": "（真实 + 超采样）"}[mode]
    for fi, row in enumerate(rows):
        twin.set_physics_qpos(log.qpos[row], joints)
        twin.set_lamps(log.lights[row])
        fs = src.at(row)
        canvas = Image.new("RGB", (W, H))
        for k in range(n):
            if mode == "accumulate":
                img = _accumulate_tile(rend, log, twin, cams[k], row, src, subsamples, joints, keep[k], isolate, k,
                                       tile)
            elif isolate:
                img = rend.render_isolated(k, fs, tile[0], tile[1])
            else:
                img = rend.render(cams[k].update(0, None, None), fs, None, log.water_w[row], ghost_keep=keep[k])
            im = Image.fromarray(img)
            OV.tile_label(im, twin.rotors.rotors[k], float(log.rpm[row, k]), float(log.thrust[row, k]),
                          float(log.u_cmd[row, k]), note)
            canvas.paste(im, ((k % cols) * tile[0], head + (k // cols) * tile[1]))
        if caption:
            OV.caption(canvas, [f"{log.t[row]:5.1f} s   {log.labels_at(row)}", caption], xy=(W // 2 - 260, H // 2 - 34),
                       sizes=(24, 16))
        arr = np.asarray(canvas)
        wr.write(arr)
        if fi in keyset:
            frames.append((float(log.t[row]), log.labels_at(row), arr.copy()))
        if progress_every and fi % progress_every == 0:
            el = time.perf_counter() - t_start
            print(f"{log_prefix}tile frame {fi + 1}/{len(rows)}  {el / (fi + 1):.2f} s/frame", flush=True)
    wr.close()
    if sheet_path and frames:
        OV.contact_sheet([f[2] for f in frames], [f"{f[0]:.1f} s  {f[1]}" for f in frames], cols=2, scale=0.5,
                         title=sheet_title).save(sheet_path)
    if own:
        rend.close()
    return dict(frames=int(len(rows)), wall_s=time.perf_counter() - t_start,
                per_frame_s=(time.perf_counter() - t_start) / max(1, len(rows)),
                render_ms=float(np.mean(rend.timing.get("render", [0]))) * 1e3)


def _accumulate_tile(rend, log, twin, cam, row, src, n, joints, keep, isolate=False, k=0, tile=(480, 540)):
    j0 = max(0, row - src.shutter_rows)
    acc = None
    for s in range(n):
        x = (s + 0.5) / n
        rr = j0 + x * (row - j0)
        i0 = int(math.floor(rr))
        i1 = min(i0 + 1, row)
        a = rr - i0
        twin.set_physics_qpos((1 - a) * log.qpos[i0] + a * log.qpos[i1], joints)
        phi = (1 - a) * src.phi_true[i0] + a * src.phi_true[i1]
        fs = FrameState(row, phi, np.zeros_like(phi), "uniform")
        if isolate:
            img = rend.render_isolated(k, fs, tile[0], tile[1]).astype(np.float32)
        else:
            img = rend.render(cam.update(0, None, None), fs, None, log.water_w[row], ghost_keep=keep).astype(np.float32)
        acc = img if acc is None else acc + img
    twin.set_physics_qpos(log.qpos[row], joints)
    return np.clip(acc / n + 0.5, 0, 255).astype(np.uint8)
