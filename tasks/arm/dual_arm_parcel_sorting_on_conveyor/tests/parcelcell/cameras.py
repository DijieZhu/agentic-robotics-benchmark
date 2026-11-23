"""The station's cameras as the program sees them: exposure in simulated
time, a fixed frame age, a ring of render states, lazy rendering, the pooled
frame budget, the sensor models.

Timing. Every camera exposes once every ``exposure_ticks`` ticks (2: 40 ms)
at the tick counts n = 0, 2, 4, ... (n = ticks completed since the shift's
t = 0; the settle before t = 0 exposes too, at n = -2, -4, ...). A frame
exposed at n is delivered ``delay_ticks`` ticks later (3: 60 ms): at tick
count N, ``get_camera`` returns the frame with the largest even n <= N - 3,
whose ``stamp`` is n x 20 ms. A frame therefore shows the world as it was at
its stamp, 60-80 ms before the reply.

The ring. At each exposure the cell keeps the render state (the full
position vector: robot, grippers, parcels, belt) in a ring of
``ring_states`` entries; nothing is drawn then. A request draws the
delivered state once (render twin data: positions copied, kinematics run;
the physics data is never touched). Asking again for the same frame of the
same camera is free and returns the same bytes; a new frame costs one unit of
the shift's pooled budget (all the cameras together). A frame the renderer
fails to draw is not charged.

Sensor models (datasheet level). RGB: additive Gaussian noise, sigma
``rgb_noise_sigma`` grey levels per channel. Depth (planar, metres -> 16-bit
millimetres, 0 = no reading): Gaussian noise of ``depth_noise_rel`` times the
distance, ``depth_dropout`` of the pixels without a reading, no reading
where the measured distance lies outside the camera's depth range. Every random draw comes from a generator
seeded by the shift's digest, the camera and the frame's exposure index only:
the noise of a frame is the same in any process, whatever was asked before.
The classic renderer's images are byte-identical across processes. The
filament renderer's dithering is switched off (or the renderer fails
closed), but its frames are not promised to be identical across processes: a
few pixels at object edges can differ (filament_engine.py).
"""

from __future__ import annotations

import base64
import math
import time

import mujoco
import numpy as np

from .look import CameraView, RenderError, make_depth, make_look

__all__ = ["CameraSystem", "encode_png", "NoFramesLeft", "NoFrameYet"]


class NoFramesLeft(Exception):
    pass


class NoFrameYet(Exception):
    pass


def encode_png(arr: np.ndarray) -> bytes:
    """PNG bytes of an 8-bit RGB image or a 16-bit single-channel image."""
    a = np.ascontiguousarray(arr)
    try:
        import cv2
        if a.ndim == 3:
            a = np.ascontiguousarray(a[..., ::-1])            # cv2 writes BGR
        ok, buf = cv2.imencode(".png", a, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        if not ok:
            raise RuntimeError("PNG encoding failed")
        return buf.tobytes()
    except ImportError:
        import io
        from PIL import Image
        b = io.BytesIO()
        Image.fromarray(a).save(b, format="PNG", compress_level=1)
        return b.getvalue()


def _T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])


class CameraSystem:
    """``world``: the ShiftWorld; ``cfg``: the whole configuration; ``digest``:
    a 64-bit integer naming the shift (noise seeds)."""

    def __init__(self, world, cfg: dict, digest: int, look: str = "classic",
                 render: bool = True, log=None):
        self.world = world
        self.model = world.model
        cc = cfg["cameras"]
        self.exposure = int(cc["exposure_ticks"])
        self.delay = int(cc["delay_ticks"])
        self.ring_len = int(cc["ring_states"])
        self.specs = {}
        self.order = sorted(cc["list"])
        for name in self.order:
            c = cc["list"][name]
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, c["model_camera"])
            if cid < 0:
                raise ValueError("no camera %r in the model" % c["model_camera"])
            self.specs[name] = dict(c, cid=cid, view=CameraView(c["model_camera"],
                                                                int(c["size"][0]), int(c["size"][1])))
            self._install(world, name, cid, c, ((cfg.get("installation") or {}).get("cameras") or {}).get(name))
        w = max(s["view"].width for s in self.specs.values())
        h = max(s["view"].height for s in self.specs.values())
        self.model.vis.global_.offwidth = w          # the offscreen buffer: the largest camera
        self.model.vis.global_.offheight = h
        tol = dict(cc.get("installation_tolerance") or {})
        fr = tol.pop("focal_rel", None)
        self.tolerance = {}
        for name in self.order:
            t = tol.get("wrist" if "flange_site" in self.specs[name] else name)
            if t:
                self.tolerance[name] = {"position_m": float(t["position_m"]), "angle_deg": float(t["angle_deg"]),
                                        "frame": str(t["frame"])}
                if fr is not None:
                    self.tolerance[name]["focal_length_rel"] = float(fr)
        sn = cfg["sensors"]
        self.rgb_sigma = float(sn["rgb_noise_sigma"])
        self.depth_rel = float(sn["depth_noise_rel"])
        self.dropout = float(sn["depth_dropout"])
        self.digest = int(digest) & ((1 << 64) - 1)
        self.look_name = look
        self.look_options = dict(cfg.get("look") or {})
        self.render_enabled = bool(render)
        self.log = log or (lambda s: None)
        self._look = None
        self._depth = None
        self._rdata = mujoco.MjData(self.model)
        self.budget = 0
        self.damage_fn = None
        """A callable giving the safety watch's {slot: damage record} (set by the task cell): a parcel damaged by an
        exposure's tick is drawn with its crushed look in that exposure and every later one (parcelkit.crush)."""
        self.reset_counters()

    # ------------------------------------------------------------ installation
    def _install(self, world, name, cid, c, inst) -> None:
        """The camera as installed. The model's camera holds the installation drawing's pose (in its mount's body:
        the world for ``infeed``, the robot's base for ``head``, the gripper for a palm camera) and the data sheet's
        field of view; the installed camera differs from it by a fixed error (``installation.cameras.<name>``:
        ``d_pos_m``, a translation in the mount's frame; ``d_rotvec_rad``, a rotation about the camera's own centre,
        axis and angle in the mount's frame; ``focal_scale``, the true focal length over the data sheet's). The images
        (colour and depth) are drawn from the installed camera; the public calibration (``describe()``, a palm frame's
        ``T_world_cam``) is the drawing's and the data sheet's. Idempotent per model (the drawing is kept on the world)."""
        m = self.model
        draw = getattr(world, "_camera_drawing", None)
        if draw is None:
            draw = world._camera_drawing = {}
        if cid not in draw:
            if int(m.cam_mode[cid]) != 0:
                raise ValueError("camera %r: only fixed-mode cameras can carry an installation error" % name)
            draw[cid] = (m.cam_pos[cid].copy(), m.cam_quat[cid].copy())
        p0, q0 = draw[cid]
        fovy = float(c["fovy_deg"])
        pos, quat = p0.copy(), q0.copy()
        if inst:
            rv = np.asarray(inst.get("d_rotvec_rad", (0.0, 0.0, 0.0)), float)
            ang = float(np.linalg.norm(rv))
            dq = np.array([1.0, 0.0, 0.0, 0.0])
            if ang > 0:
                dq = np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * rv / ang])
            mujoco.mju_mulQuat(quat, dq, q0)
            quat /= np.linalg.norm(quat)
            pos = p0 + np.asarray(inst.get("d_pos_m", (0.0, 0.0, 0.0)), float)
            fs = float(inst.get("focal_scale", 1.0))
            fovy = math.degrees(2.0 * math.atan(math.tan(math.radians(fovy) / 2.0) / fs))
        m.cam_pos[cid] = pos
        m.cam_quat[cid] = quat
        m.cam_fovy[cid] = fovy
        self.specs[name]["drawing"] = (p0, q0)

    def _T_world_cam_drawing(self, data, name: str) -> np.ndarray:
        """The installation drawing's camera pose (OpenCV axes) at the kinematic state ``data``."""
        cid = self.specs[name]["cid"]
        p0, q0 = self.specs[name]["drawing"]
        b = int(self.model.cam_bodyid[cid])
        Rb = data.xmat[b].reshape(3, 3)
        Rc = np.zeros(9)
        mujoco.mju_quat2Mat(Rc, q0)
        return _T(Rb @ Rc.reshape(3, 3) @ CV_FROM_MJ, data.xpos[b] + Rb @ p0)

    # ------------------------------------------------------------ ring
    def reset_counters(self, budget: int | None = None) -> None:
        self.ring = {}                       # exposure index -> qpos copy
        self.ring_damaged = {}               # exposure index -> the slots damaged by then
        self.crushed = {}                    # slot -> {"exposure", "switch_s"}: the crushed looks drawn
        self.cache = {}                      # camera -> (exposure, reply dict, arrays)
        self.used = 0
        self.by_camera = {n: 0 for n in self.order}
        self.repeats = 0
        self.render_wall = 0.0
        self.render_failures = 0
        self.frames_lost = 0
        self.render_times = {n: [] for n in self.order}
        if budget is not None:
            self.budget = int(budget)

    def _keep(self, n: int) -> None:
        self.ring[n] = self.world.data.qpos.copy()
        dmg = self.damage_fn() if callable(self.damage_fn) else None
        self.ring_damaged[n] = frozenset(int(s) for s in (dmg or {}))
        for k in sorted(self.ring)[:-self.ring_len]:
            del self.ring[k]
            self.ring_damaged.pop(k, None)

    def crush_seed(self, slot: int) -> int:
        return (self.digest * 1000003 + 7919 * (int(slot) + 1)) & ((1 << 63) - 1)

    def _apply_damage(self, e: int) -> None:
        """Draw every parcel damaged by exposure e with its crushed look (once per parcel: a damage is for good, and
        the frames are rendered in time order: the newest delivered exposure)."""
        new = sorted(set(self.ring_damaged.get(e, ())) - set(self.crushed))
        if not new or self._look is None or not hasattr(self._look, "crush"):
            return
        for slot in new:
            t0 = time.time()
            arr = self.world.build_shift.arrivals
            kind = arr[slot].parcel.kind if 0 <= slot < len(arr) else "rigid"
            self._look.crush(slot, self.crush_seed(slot), kind)
            self.crushed[slot] = {"exposure": int(e), "switch_s": round(time.time() - t0, 4)}

    def on_settle(self, world, k: int) -> None:
        if k % self.exposure == 0:
            self._keep(k)

    def on_tick(self, world) -> None:
        n = int(world.n_ticks)
        if n % self.exposure == 0:
            self._keep(n)

    def delivered_index(self, now_tick: int | None = None) -> int:
        """Exposure index of the newest frame delivered at tick count now."""
        N = int(self.world.n_ticks if now_tick is None else now_tick)
        e = N - self.delay
        return e - (e % self.exposure)

    @property
    def frames_left(self) -> int:
        return max(0, self.budget - self.used)

    # ------------------------------------------------------------ calibration
    def intrinsics(self, name: str) -> dict:
        v = self.specs[name]["view"]
        fovy = math.radians(float(self.specs[name]["fovy_deg"]))
        f = 0.5 * v.height / math.tan(0.5 * fovy)
        return {"fx": f, "fy": f, "cx": 0.5 * v.width - 0.5, "cy": 0.5 * v.height - 0.5,
                "width": v.width, "height": v.height}

    def T_world_cam(self, data, name: str) -> np.ndarray:
        """The published pose: the installation drawing's (a palm camera's: the drawing's mount on the flange
        composed with the flange's pose from the measured joints)."""
        return self._T_world_cam_drawing(data, name)

    def T_world_cam_installed(self, data, name: str) -> np.ndarray:
        """The camera's true pose as installed (the grader's side only; never on the wire)."""
        cid = self.specs[name]["cid"]
        return _T(data.cam_xmat[cid].reshape(3, 3) @ CV_FROM_MJ, data.cam_xpos[cid])

    def T_flange_cam(self, data, name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.specs[name]["flange_site"])
        Tf = _T(data.site_xmat[sid].reshape(3, 3), data.site_xpos[sid])
        return np.linalg.inv(Tf) @ self._T_world_cam_drawing(data, name)

    def describe(self, data) -> dict:
        """The public camera table (``data``: any kinematic state; fixed cameras
        and the wrist mounts do not depend on it): the data sheet's optics and
        the installation drawing's poses with their tolerances."""
        out = {}
        for name in self.order:
            s = self.specs[name]
            k = self.intrinsics(name)
            e = {"width": k["width"], "height": k["height"], "fx": k["fx"], "fy": k["fy"],
                 "cx": k["cx"], "cy": k["cy"],
                 "K": [[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]],
                 "fovy_deg": float(s["fovy_deg"]),
                 "depth_range_m": [float(v) for v in s["depth_range_m"]],
                 "depth": "planar distance along the optical axis",
                 "convention": "OpenCV: x right, y down, z forward; pixel (u, v) = (column, row), "
                               "row 0 at the top; a pixel centre's ray is ((u - cx) / fx, (v - cy) / fy, 1)",
                 "exposure_period_s": self.exposure * 0.02,
                 "frame_age_s": self.delay * 0.02}
            tol = self.tolerance.get(name)
            if tol:
                e["installation"] = dict(tol, note="poses and intrinsics are the installation drawing's and the "
                                                   "data sheet's; the installed camera differs within these "
                                                   "tolerances: calibrate it on site")
            if "flange_site" in s:
                e["mount"] = "wrist"
                e["flange_site"] = s["flange_site"]
                e["T_flange_cam"] = np.round(self.T_flange_cam(data, name), 9).tolist()
                e["note"] = ("T_world_cam = T_world_flange @ T_flange_cam; each frame carries "
                             "its T_world_cam at exposure")
            else:
                e["mount"] = "fixed"
                e["T_world_cam"] = np.round(self.T_world_cam(data, name), 9).tolist()
            out[name] = e
        return out

    # ------------------------------------------------------------ frames
    def _renderers(self):
        if self._look is None:
            self._look = make_look(self.look_name, self.model, self.specs, world=self.world,
                                   options=self.look_options)
            self._depth = make_depth(self.model, self._look)

    def _rng(self, name: str, e: int, channel: int) -> np.random.Generator:
        lo, hi = self.digest & 0xFFFFFFFF, self.digest >> 32
        return np.random.default_rng([lo, hi, self.order.index(name), e + (1 << 20), channel])

    def _state(self, e: int):
        q = self.ring.get(e)
        if q is None:
            raise RenderError("no render state for exposure %d" % e)
        d = self._rdata
        d.qpos[:] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_camlight(self.model, d)
        return d

    def _rgb(self, name: str, e: int) -> np.ndarray:
        self._apply_damage(e)
        d = self._state(e)
        img = self._look.rgb(d, self.specs[name]["view"])
        if self.rgb_sigma > 0:
            noise = self._rng(name, e, 0).normal(0.0, self.rgb_sigma, img.shape)
            img = np.clip(np.rint(img.astype(np.float64) + noise), 0, 255).astype(np.uint8)
        return img

    def _depth_mm(self, name: str, e: int) -> np.ndarray:
        d = self._state(e)
        z = self._depth.depth(d, self.specs[name]["view"]).astype(np.float64)
        rng = self._rng(name, e, 1)
        zn = z + rng.normal(0.0, 1.0, z.shape) * (self.depth_rel * z)
        lo, hi = (float(v) for v in self.specs[name]["depth_range_m"])
        drop = rng.random(z.shape) < self.dropout
        bad = (~np.isfinite(zn)) | (zn < lo) | (zn > hi) | drop
        mm = np.clip(np.rint(np.where(bad, 0.0, zn) * 1000.0), 1, 65535).astype(np.uint16)
        mm[bad] = 0
        return mm

    def get(self, name: str, depth: bool = False) -> dict:
        """The newest delivered frame of camera ``name`` (a reply dict).
        Raises NoFramesLeft (budget), NoFrameYet, RenderError (not charged)."""
        e = self.delivered_index()
        cached = self.cache.get(name)
        repeat = cached is not None and cached[0] == e
        if not repeat:
            if e not in self.ring:
                raise NoFrameYet("no frame delivered yet")
            if self.used >= self.budget:
                raise NoFramesLeft("the shift's frame budget is used up")
        if not self.render_enabled:
            raise RenderError("rendering is off in this cell")
        t0 = time.time()
        try:
            self._renderers()
            if repeat:
                reply = cached[1]
                if depth and "depth_png_b64" not in reply:
                    reply["depth_png_b64"] = base64.b64encode(
                        encode_png(self._depth_mm(name, e))).decode("ascii")
                    reply["depth_unit_m"] = 0.001
                    reply["depth_invalid"] = 0
                self.repeats += 1
                out = dict(reply)
                if not depth:
                    for k in ("depth_png_b64", "depth_unit_m", "depth_invalid"):
                        out.pop(k, None)
                out["repeat"] = True
                out["frames_left"] = self.frames_left
                return out
            rgb = self._rgb(name, e)
            v = self.specs[name]["view"]
            reply = {"camera": name, "frame": int(e), "stamp": round(e * 0.02, 6),
                     "width": v.width, "height": v.height, "format": "png",
                     "rgb_png_b64": base64.b64encode(encode_png(rgb)).decode("ascii")}
            if depth:
                reply["depth_png_b64"] = base64.b64encode(
                    encode_png(self._depth_mm(name, e))).decode("ascii")
                reply["depth_unit_m"] = 0.001
                reply["depth_invalid"] = 0
            if "flange_site" in self.specs[name]:
                reply["T_world_cam"] = np.round(self.T_world_cam(self._rdata, name), 9).tolist()
        except RenderError:
            self.render_failures += 1
            self.frames_lost += 1
            raise
        except Exception as ex:  # noqa: BLE001 - a renderer failure never takes the cell down
            self.render_failures += 1
            self.frames_lost += 1
            self.log("[cell] %s frame %d not rendered: %s" % (name, e, type(ex).__name__))
            raise RenderError(str(ex)) from ex
        finally:
            dt = time.time() - t0
            self.render_wall += dt
        self.render_times[name].append(dt)
        self.used += 1
        self.by_camera[name] += 1
        self.cache[name] = (e, reply)
        out = dict(reply)
        out["repeat"] = False
        out["frames_left"] = self.frames_left
        return out

    def close(self) -> None:
        for r in (self._depth, self._look):
            if r is not None:
                try:
                    r.close()
                except Exception:  # noqa: BLE001
                    pass
        self._look = self._depth = None
