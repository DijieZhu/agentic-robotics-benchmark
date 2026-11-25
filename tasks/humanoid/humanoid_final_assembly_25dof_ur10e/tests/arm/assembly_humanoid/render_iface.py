"""The ONE render interface of the humanoid cell's camera call.

New file of this task. The server's get_camera() never renders itself: it asks a backend made by `open_backend(name,
cell, cfg)` for one frame (`frame(camera, depth, index)`), and every backend answers with the same structure, so that
the given render layer (Filament in the baked room, through a worker process) is one more backend and nothing else in
the server changes. Backends here:

  "null"    no picture: every frame raises RenderUnavailable (the server answers {"error": "camera_unavailable"} and
            the frame counts against the budget). The default on a host without GL.
  "mujoco"  MuJoCo's classic renderer (mujoco.Renderer, needs EGL or OSMesa) on the cell's own model; it draws a
            different picture from the given render layer, which the grader uses.
  "blank"   tests only: a uniform grey RGB frame and an all-invalid depth frame of the camera's configured size, encoded
            exactly as a real frame is (PNG rgb, 16-bit PNG depth in mm, 0 = invalid), so that the reply format and the
            budget can be tested without a renderer.
  "given_layer"  the cell's cameras through the given render layer in a worker process; a failed frame is retried once
            on a fresh worker, then a control frame is rendered on another fresh worker (GivenLayerBackend.frame).

A frame = {"rgb": uint8 H x W x 3, "depth_mm": uint16 H x W or None}; `encode_reply(...)` builds the shared reply
(arm/assembly_info/cell_info.py get_camera: "ok", "camera", "frame_index", "width", "height", "format", "rgb_png_b64",
"sensor_depth_u16_mm_b64", "depth_unit_m", "depth_invalid_value").

Camera names: the agent's names (cfg["cameras"]["agent_cameras"], "top" and "wrist") map to the model's cameras:
"wrist" -> the robot's wrist camera (`ur/wrist`, gripper_wide.add_ur_wide), any other name -> the model camera of that
name (the scene data's cameras).
"""
import base64
import io
import json
import os
import select
import subprocess
import tempfile
import time

import numpy as np

__all__ = ["RenderUnavailable", "RenderInfrastructureError", "open_backend", "encode_reply", "BACKENDS", "model_camera",
           "DEADLINE_FLOORS_S"]

# the given layer's waits: the first frame of a worker session (its start, its "open" and that frame) and every later
# frame or close; the grader binds them to the budget (verify.render_deadlines); these are the floors
DEADLINE_FLOORS_S = {"first_frame": 300.0, "frame": 60.0}
MAX_REPLY_BYTES = 1 << 20


class RenderUnavailable(RuntimeError):
    """no frame for this request: the server answers {"error": "camera_unavailable"}"""


class RenderInfrastructureError(RenderUnavailable):
    """the renderer itself failed: a frame, its retry on a fresh worker and a control frame on another fresh worker all
    failed. Never answered to the program: the server lets it through, and the grader ends the verify with a task error"""


def model_camera(cell, name):
    if name == "wrist" and cell.robot:
        return cell.robot["wrist_camera"]
    return name


class NullBackend(object):
    name = "null"

    def __init__(self, cell, cfg):
        self.cell = cell

    def frame(self, camera, depth, index):
        raise RenderUnavailable("no renderer in this cell (backend 'null')")

    def close(self):
        pass


class BlankBackend(object):
    name = "blank"

    def __init__(self, cell, cfg):
        self.sizes = {k: tuple(int(x) for x in v) for k, v in cfg["cameras"]["size"].items()}

    def frame(self, camera, depth, index):
        w, h = self.sizes[camera]
        rgb = np.full((h, w, 3), 128, np.uint8)
        dm = np.zeros((h, w), np.uint16) if depth else None
        return {"rgb": rgb, "depth_mm": dm}

    def close(self):
        pass


class MujocoBackend(object):
    name = "mujoco"

    def __init__(self, cell, cfg):
        import mujoco
        self.mj, self.cell = mujoco, cell
        self.sizes = {k: tuple(int(x) for x in v) for k, v in cfg["cameras"]["size"].items()}
        self.r = {}

    def _renderer(self, camera):
        if camera not in self.r:
            w, h = self.sizes[camera]
            try:
                self.r[camera] = self.mj.Renderer(self.cell.m, h, w)
            except Exception as e:  # noqa: BLE001 - a host without GL
                raise RenderUnavailable("MuJoCo renderer unavailable: %s" % type(e).__name__)
        return self.r[camera]

    def frame(self, camera, depth, index):
        r = self._renderer(camera)
        cam = model_camera(self.cell, camera)
        r.update_scene(self.cell.d, camera=cam)
        rgb = r.render().copy()
        dm = None
        if depth:
            r.enable_depth_rendering()
            r.update_scene(self.cell.d, camera=cam)
            z = r.render().astype(np.float64)
            r.disable_depth_rendering()
            dm = np.where(np.isfinite(z) & (z > 0) & (z < 65.0), np.round(z * 1000.0), 0).astype(np.uint16)
        return {"rgb": rgb, "depth_mm": dm}

    def close(self):
        for r in self.r.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        self.r = {}


CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def given_layer_session(cell, cfg, log=None):
    """the seam of the given render layer: a session object with .frame(camera_dict, depth, index, state=None) ->
    {"rgb", "depth_mm"} and .close(). Without cfg["cameras"]["given_layer"] this raises RenderUnavailable. With it, the
    session is a WORKER PROCESS running the given render layer (in this task tests/render_seam/twin_worker.py:
    the given layer's open_view, the render twin, the room and the look) on the episode's own layout. The worker's file
    descriptor 1 (where the Filament library writes its native log) is a log FILE, its replies come back on a separate
    pipe: nothing it prints can reach the server's process, its wire to the agent's program (that program's own stdin /
    stdout pipes) or the server's own stdout. Every wait for a reply has a deadline (cfg["cameras"]["given_layer"]
    "deadlines_s": "first_frame" from the worker's start to its first frame's reply, "frame" for every later frame and
    for the close; DEADLINE_FLOORS_S when not given): past it the worker is killed and the request fails."""
    gl = (cfg.get("cameras") or {}).get("given_layer")
    if not gl:
        raise RenderUnavailable("the given render layer is not installed in this cell (render_iface.given_layer_session)")
    return _WorkerSession(cell, gl, log)


def joint_state(m, qpos):
    """{joint name: its qpos values} of every joint of the cell's model (the worker's input)"""
    state = {}
    for j in range(m.njnt):
        n = m.joint(j).name
        a = int(m.jnt_qposadr[j])
        k = {0: 7, 1: 4, 2: 1, 3: 1}[int(m.jnt_type[j])]
        state[n] = [float(v) for v in qpos[a:a + k]]
    return state


def _deadlines(gl):
    d = dict(DEADLINE_FLOORS_S)
    d.update({k: float(v) for k, v in (gl.get("deadlines_s") or {}).items() if k in d})
    return d


class _WorkerSession(object):
    def __init__(self, cell, gl, log=None):
        self.cell = cell
        self.log = log or (lambda s: None)
        self.deadlines = _deadlines(gl)
        self.t_start = time.monotonic()
        self.frames = 0
        self.proc = self.logf = None
        self._rfd, self._buf = None, b""
        self.folder = tempfile.mkdtemp(prefix="render_session_", dir=gl.get("scratch"))
        r, w = os.pipe()
        env = dict(os.environ)
        env.update({k: str(v) for k, v in (gl.get("env") or {}).items()})
        try:
            self.logf = open(os.path.join(self.folder, "worker_fd1_fd2.log"), "wb")
            self.proc = subprocess.Popen([gl["python"], "-B", gl["worker"], str(w), self.folder], stdin=subprocess.PIPE,
                                         stdout=self.logf, stderr=subprocess.STDOUT, pass_fds=(w,), env=env, close_fds=True)
        except BaseException:
            os.close(r)
            os.close(w)
            if self.logf is not None:
                self.logf.close()
            raise
        os.close(w)
        self._rfd = r
        try:
            draw = getattr(cell, "episode_draw", None)
            if draw is None:
                raise RenderUnavailable("the cell carries no episode draw (build it with episode.build_episode)")
            lay = {k: (list(v) if isinstance(v, tuple) and k != "top_camera_T" else v) for k, v in draw["layout"].items()}
            if lay.get("top_camera_T") is not None:
                lay["top_camera_T"] = [np.asarray(lay["top_camera_T"][0], float).tolist(), np.asarray(lay["top_camera_T"][1], float).tolist()]
            self.opened = self._ask({"op": "open", "layout": lay}, self.t_start + self.deadlines["first_frame"])
        except BaseException:
            self.kill()
            raise

    def _deadline(self):
        if self.frames == 0:
            return self.t_start + self.deadlines["first_frame"]
        return time.monotonic() + self.deadlines["frame"]

    def _ask(self, req, deadline):
        self.proc.stdin.write((json.dumps(req) + "\n").encode())
        self.proc.stdin.flush()
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise RenderUnavailable("the render worker did not answer %r in time (see its log %s)" % (req.get("op"), self.logf.name))
            ready, _w, _x = select.select([self._rfd], [], [], left)
            if not ready:
                continue
            chunk = os.read(self._rfd, 1 << 16)
            if not chunk:
                raise RenderUnavailable("the render worker ended (see its log %s)" % self.logf.name)
            self._buf += chunk
            if len(self._buf) > MAX_REPLY_BYTES:
                raise RenderUnavailable("the render worker's reply is too long")
        line, _nl, self._buf = self._buf.partition(b"\n")
        rep = json.loads(line.decode("utf-8"))
        if not isinstance(rep, dict) or not rep.get("ok"):
            raise RenderUnavailable("render worker: %s" % (rep.get("error") if isinstance(rep, dict) else rep,))
        return rep

    def frame(self, camera, depth, index, state=None):
        state = joint_state(self.cell.m, self.cell.d.qpos) if state is None else state
        rep = self._ask({"op": "frame", "camera": camera, "state": state, "depth": bool(depth), "index": int(index)},
                        self._deadline())
        z = np.load(rep["npz"])
        dep = z["depth"]
        dm = None
        if depth and dep.size:
            dm = np.where(np.isfinite(dep) & (dep > 0) & (dep < 65.0), np.round(dep * 1000.0), 0).astype(np.uint16)
        self.last = rep
        self.frames += 1
        return {"rgb": np.asarray(z["rgb"], np.uint8), "depth_mm": dm}

    def kill(self):
        """end the worker now (a failed session): no close request, no wait beyond the kill"""
        if self.proc is not None:
            try:
                self.proc.kill()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                pass
        self._release()

    def _release(self):
        for f in (getattr(self.proc, "stdin", None), self.logf):
            try:
                if f is not None:
                    f.close()
            except Exception:  # noqa: BLE001
                pass
        if self._rfd is not None:
            try:
                os.close(self._rfd)
            except OSError:
                pass
            self._rfd = None

    def close(self):
        deadline = time.monotonic() + self.deadlines["frame"]
        try:
            self.closed = self._ask({"op": "close"}, deadline)
        except Exception:  # noqa: BLE001
            self.closed = None
        try:
            self.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except Exception:  # noqa: BLE001
            self.kill()
            return
        self._release()


class GivenLayerBackend(object):
    """the agent cameras through the given render layer (the seam given_layer_session): the overhead camera with the
    episode's drawn mount and lens (pinhole terms; the drawn distortion is the render version's), the wrist camera at the
    robot's model camera with the configured nominal field.

    A frame that fails in any way (an exception or a deadline of the worker while it makes the frame) is retried ONCE on
    a fresh worker; if the retry fails too, a CONTROL frame (the cell's scene as it was when this backend was made, the
    same camera) is rendered on another fresh worker. The control fails: RenderInfrastructureError (the renderer is
    broken; the grader's task error). The control renders: RenderUnavailable for this request (the program gets
    camera_unavailable), and the control's worker serves the next frames."""
    name = "given_layer"

    def __init__(self, cell, cfg):
        self.cell, self.cfg = cell, cfg
        self.session = None
        self.retries = self.controls = 0
        self.failures = []
        # the control scene: the cell's joint state and each camera as they are now (the episode's start)
        self._control = {"state": None, "cameras": {}}
        try:
            self._control["state"] = joint_state(cell.m, np.array(cell.d.qpos, float))
        except Exception as e:  # noqa: BLE001 - no control scene: a failed frame then ends as an infrastructure failure
            self.failures.append("no control state: %s: %s" % (type(e).__name__, str(e)[:200]))
        for n in ((cfg.get("cameras") or {}).get("size") or {}):
            try:
                self._control["cameras"][n] = self._camera(n)
            except Exception as e:  # noqa: BLE001
                self.failures.append("no control view of %s: %s: %s" % (n, type(e).__name__, str(e)[:200]))

    def _camera(self, camera):
        cell = self.cell
        C = self.cfg["cameras"]
        w, h = [int(v) for v in C["size"][camera]]
        if camera == "top":
            dr = getattr(cell, "episode_draw", None)
            K = dr["cameras"]["top"]["intrinsics"]
            return {"name": "top", "width": w, "height": h, "fx": float(K["fx"]), "fy": float(K["fy"]), "cx": float(K["cx"]),
                    "cy": float(K["cy"]), "world_T_cam": dr["cameras"]["top"]["world_T_cam"]}
        import math
        cid = cell.m.camera(model_camera(cell, camera)).id
        T = np.eye(4)
        T[:3, :3] = cell.d.cam_xmat[cid].reshape(3, 3) @ CV_TO_MJ
        T[:3, 3] = cell.d.cam_xpos[cid]
        f = 0.5 * w / math.tan(math.radians(float(C["mounts"]["wrist"]["hfov_deg"])) / 2.0)
        return {"name": camera, "width": w, "height": h, "fx": f, "fy": f, "cx": (w - 1) / 2.0, "cy": (h - 1) / 2.0,
                "world_T_cam": T.tolist()}

    def _drop(self):
        s, self.session = self.session, None
        if s is not None:
            s.kill()

    def _attempt(self, camera, depth, index, control=False):
        if self.session is None:
            self.session = given_layer_session(self.cell, self.cfg)
        if control:
            return self.session.frame(self._control["cameras"][camera], depth, index, state=self._control["state"])
        return self.session.frame(self._camera(camera), depth, index)

    def _failed(self, what, e):
        self.failures.append("%s: %s: %s" % (what, type(e).__name__, str(e)[:200]))
        self._drop()

    def frame(self, camera, depth, index):
        try:
            return self._attempt(camera, depth, index)
        except Exception as e:  # noqa: BLE001 - every failure of the frame takes the retry path
            self._failed("frame %d" % index, e)
        self.retries += 1
        try:
            return self._attempt(camera, depth, index)
        except Exception as e:  # noqa: BLE001
            self._failed("frame %d, its retry on a fresh worker" % index, e)
        self.controls += 1
        if self._control["state"] is None or camera not in self._control["cameras"]:
            raise RenderInfrastructureError("the %s camera's frame %d failed twice and there is no control scene (%s)" % (
                camera, index, "; ".join(self.failures[-3:])))
        try:
            self._attempt(camera, depth, index, control=True)
        except Exception as e:  # noqa: BLE001
            self._failed("frame %d, the control frame on another fresh worker" % index, e)
            raise RenderInfrastructureError("the %s camera's frame %d, its retry and the control frame all failed (%s)" % (
                camera, index, "; ".join(self.failures[-3:])))
        raise RenderUnavailable("the %s camera's frame %d and its retry failed; the control frame rendered" % (camera, index))

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None


BACKENDS = {"null": NullBackend, "blank": BlankBackend, "mujoco": MujocoBackend, "given_layer": GivenLayerBackend}


def open_backend(name, cell, cfg):
    if name not in BACKENDS:
        raise ValueError("render backend %r (one of %s)" % (name, ", ".join(sorted(BACKENDS))))
    return BACKENDS[name](cell, cfg)


def _png(a):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def encode_reply(camera, index, fr):
    rgb = np.asarray(fr["rgb"], np.uint8)
    out = {"ok": True, "camera": camera, "frame_index": int(index), "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
           "format": "png", "rgb_png_b64": base64.b64encode(_png(rgb)).decode("ascii")}
    if fr.get("depth_mm") is not None:
        out["sensor_depth_u16_mm_b64"] = base64.b64encode(_png(np.asarray(fr["depth_mm"], np.uint16))).decode("ascii")
        out["depth_unit_m"] = 0.001
        out["depth_invalid_value"] = 0
    return out
