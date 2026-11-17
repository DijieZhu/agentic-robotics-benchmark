"""Camera frames of the cell: colour from MuJoCo's Filament renderer (physically based, on Vulkan: Mesa's lavapipe on
the CPU in the sandbox), depth from MuJoCo's classic renderer (OSMesa) on the same model, plus a RealSense-like sensor
model.

RGB: pinhole, no distortion; the decorated apartment lit by the morning sun through the east window, the two ceiling
panels, the pendant over the dining table, the strip under the wall cabinets, the light the sunlit patches of the floor
and the table throw back up, and an image light, all distilled from the path-traced (OVRTX) look of the same apartment
(look.json next to this file: the lights in physical units with the look's white balance, the image light, the
materials' PBR terms and per-material colour gains fitted to OVRTX stills of these cameras); per-episode white-balance
/ brightness draw, per-pixel noise. Depth (metres along the optical axis, registered to the RGB pixels): noise sigma 1.5 mm + 2.5 mm/m^2 z^2, 1 mm
quantisation, invalid (0) below the minimum range, beyond 6 m, on depth discontinuities and on 2 % random pixels.
Intrinsics are exact and published (fx = fy = (H / 2) / tan(fovy / 2), principal point at the image centre). Colour
and depth are drawn from the state the cell hands over (Cell.render_state: the physics state with the robots' legs as
walk_core draws them, stepping while a robot moves).

Filament facts this relies on (MuJoCo 3.16): one context serves one model's resources at a time, so only the model
rendered last is live (rendering another model first frees the live one's Filament objects); the engine is bound to
the thread that created it, so a dedicated thread owns the context and every Filament object and callers post jobs to
it; its job threads would each get a glibc malloc arena, so the arenas are capped at import; a texture's colour space
must be marked sRGB (Filament reads MuJoCo's "auto" as linear data); textured materials keep MuJoCo's legacy shading
(no PBR terms on them).
"""

from __future__ import annotations

import atexit
import base64
import gc
import io
import json
import math
import os
import queue
import threading
import time
import weakref


def _cap_malloc_arenas():
    """The renderer's job threads would each get a glibc malloc arena and the process would grow by gigabytes over
    episodes. Two arenas are plenty (M_ARENA_MAX); glibc honours the cap only before other threads allocate."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").mallopt(-8, 2)
    except (OSError, AttributeError):
        pass


_cap_malloc_arenas()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DEPTH_MIN = {"head": 0.28, "ceiling": 0.40}
DEPTH_MAX = 6.0
HERE = os.path.dirname(os.path.abspath(__file__))
LOOK_FILE = os.path.join(HERE, "look.json")
DECOR_DIR = os.environ.get("ARB_ROOM_DECOR_DIR", "/opt/arb-assets/room_decor")
CAM_NEAR, CAM_FAR = 0.01, 160.0        # m: Filament's clip planes (the street view is 30 to 120 m out)
CALL_TIMEOUT = 300.0                    # s: one job of the renderer's thread


def load_look(path: str = LOOK_FILE) -> dict:
    with open(path) as f:
        return json.load(f)


LOOK = load_look()


def filament_view_keys(look: dict = LOOK) -> tuple[dict, dict]:
    """(numeric, text): the Filament view options MuJoCo reads from the model's custom fields (`filament.*`); the
    cell adds them to its spec (scene.build)."""
    v = look.get("view", {})
    return ({k: float(x) for k, x in v.get("numeric", {}).items()}, {k: str(x) for k, x in v.get("text", {}).items()})


def intrinsics(width: int, height: int, fovy_deg: float) -> dict:
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return {"width": width, "height": height, "fx": f, "fy": f, "cx": (width - 1) / 2.0, "cy": (height - 1) / 2.0,
            "fovy_deg": fovy_deg, "model": "pinhole, no distortion"}


def camera_pose_world(m, d, cam_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Optical frame in the room: origin, rotation whose columns are the image x (right), y (down), z (forward) axes."""
    p = d.cam_xpos[cam_id].copy()
    R = d.cam_xmat[cam_id].reshape(3, 3)
    # MuJoCo cameras look along -z with +y up; the optical frame (OpenCV) is x right, y down, z forward
    return p, np.stack([R[:, 0], -R[:, 1], -R[:, 2]], axis=1)


# ------------------------------------------------------------------------------------------------ the look
def _lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def apply_look(m: mujoco.MjModel, look: dict = LOOK) -> dict:
    """The look on a compiled cell model, in place, on render-only fields (no effect on the physics or on the classic
    renderer's depth): sRGB textures, the untextured materials' PBR terms (or legacy specular), every material's
    calibrated colour gain (per channel, on the linear albedo: the colour of an untextured material, the texture data
    of a textured one; each decor material has a texture of its own) and the lights in physical units (spot lights in
    candela, the sun in lux; a light's own shadow map size, `shadow_map`, is set where the renderer builds its lights).
    Returns a record of what was set."""
    m.tex_colorspace[:] = int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)
    m.vis.headlight.active = 0
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    pbr, gains = look.get("materials", {}), look.get("gains", {})
    rec = {"pbr": 0, "gains": 0, "lights": 0}
    for i in range(m.nmat):
        name = m.mat(i).name
        tex = int(m.mat_texid[i][rgb_role])
        p = pbr.get(name)
        # PBR terms on a textured material break MuJoCo 3.16's Filament: textured materials (and the look's very dark
        # ones) keep MuJoCo's legacy shading, with the look's specular
        if p is not None and tex < 0 and "roughness" in p:
            m.mat_roughness[i] = float(p["roughness"])
            m.mat_metallic[i] = float(p.get("metallic", 0.0))
            rec["pbr"] += 1
        if p is not None:
            for key, field in (("specular", "mat_specular"), ("shininess", "mat_shininess"),
                               ("emission", "mat_emission")):
                if key in p:
                    getattr(m, field)[i] = float(p[key])
        g = gains.get(name)
        if g is None:
            continue
        g = np.broadcast_to(np.asarray(g, np.float64), (3,))
        if tex >= 0:
            a, n = int(m.tex_adr[tex]), int(m.tex_width[tex] * m.tex_height[tex] * m.tex_nchannel[tex])
            px = m.tex_data[a:a + n].reshape(-1, int(m.tex_nchannel[tex]))
            lin = _lin(px[:, :3] / 255.0) * g[None, :]
            px[:, :3] = np.clip(np.round(255.0 * _srgb(lin)), 0, 255).astype(np.uint8)
        else:
            want = _lin(m.mat_rgba[i, :3]) * g
            m.mat_rgba[i, :3] = _srgb(want)
            over = float(want.max())
            if over > 1.0:                     # what a colour at full value cannot take: a little emission
                m.mat_emission[i] = min(0.4, 0.5 * (over - 1.0))
        rec["gains"] += 1
    for i in range(m.nlight):
        name = m.light(i).name
        L = look.get("lights", {}).get(name)
        if L is None:
            m.light_intensity[i] = 0.0
            m.light_castshadow[i] = 0
            continue
        m.light_intensity[i] = float(L["intensity"])
        m.light_diffuse[i] = list(L.get("color", (1.0, 1.0, 1.0)))
        m.light_specular[i] = [0.0, 0.0, 0.0]
        m.light_ambient[i] = [0.0, 0.0, 0.0]
        for key, field in (("bulb_m", "light_bulbradius"), ("cutoff", "light_cutoff"), ("exponent", "light_exponent"),
                           ("range", "light_range")):
            if key in L:
                getattr(m, field)[i] = float(L[key])
        if "shadow" in L:
            m.light_castshadow[i] = int(bool(L["shadow"]))
        rec["lights"] += 1
    return rec


# ------------------------------------------------------------------------------------------------ the renderer
class RenderError(RuntimeError):
    pass


class _View:
    """One cell model in the process's renderer; its Filament objects stay in the renderer's thread."""

    def __init__(self, fil, model):
        self._fil = fil
        self._sid = fil.call(fil.new_session, model)
        self._finalizer = weakref.finalize(self, fil.post, fil.free_session, self._sid)

    def render(self, data, pos, forward, up, fovy, width, height):
        return self._fil.call(self._fil.draw, self._sid, data, pos, forward, up, fovy, width, height)

    def close(self):
        if self._finalizer.detach() is not None:
            self._fil.call(self._fil.free_session, self._sid)


class _Filament:
    """The process's Filament renderer: a dedicated thread owns the context and every Filament object; callers on any
    thread post jobs and wait. A forked child starts its own. Only the model rendered last is live."""

    _inst = None
    _guard = threading.Lock()

    @classmethod
    def instance(cls):
        with cls._guard:
            if cls._inst is None or cls._inst.pid != os.getpid():
                cls._inst = cls()
            return cls._inst

    def __init__(self):
        self.pid = os.getpid()
        self.jobs = queue.Queue()
        self.models = {}         # session id -> its model
        self.active = None       # the session whose Filament objects are live
        self.live = []
        self.targets = {}
        self.next_sid = 0
        self.thread = threading.Thread(target=self._loop, name="house-cell-filament", daemon=True)
        self.thread.start()
        self.call(self._start)

    def open(self, model):
        return _View(self, model)

    # -- any thread
    def call(self, fn, *args):
        done = threading.Event()
        out = []
        self.jobs.put((fn, args, done, out))
        if not done.wait(CALL_TIMEOUT):
            raise RenderError("the renderer did not answer within %.0f s" % CALL_TIMEOUT)
        ok, val = out[0]
        if not ok:
            raise val
        return val

    def post(self, fn, *args):
        if self.pid == os.getpid() and self.thread.is_alive():
            self.jobs.put((fn, args, None, None))

    def stop(self, timeout=30.0):
        if self.pid == os.getpid() and self.thread.is_alive():
            self.jobs.put((None, (), None, None))
            self.thread.join(timeout)

    def _loop(self):
        while True:
            fn, args, done, out = self.jobs.get()
            if fn is None:
                self._shutdown()
                return
            try:
                res = (True, fn(*args))
            except BaseException as exc:  # noqa: BLE001 - handed to the caller without the traceback (its frames
                res = (False, RenderError("renderer: %s: %s" % (type(exc).__name__, exc)))   # hold Filament objects)
            if done is not None:
                out.append(res)
                done.set()

    # -- the renderer's thread only
    def _start(self):
        # offscreen: Filament's resource providers would connect to an X display if one were named
        for k in ("DISPLAY", "WAYLAND_DISPLAY"):
            os.environ.pop(k, None)
        # lavapipe's rasteriser threads: the CPUs this process may use (its default is every CPU of the host)
        try:
            ncpu = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            ncpu = os.cpu_count() or 4
        os.environ.setdefault("LP_NUM_THREADS", str(max(1, min(8, ncpu))))
        import mujoco._render_filament as F
        from mujoco.experimental.studio import window  # noqa: F401  registers Filament's resource providers
        self.F = F
        ibl = LOOK.get("ibl", {})
        path = os.path.join(DECOR_DIR, ibl.get("file", "ibl.ktx"))
        if not os.path.isfile(path):
            raise RenderError("render asset missing: " + path)
        self.ctx = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN))
        self.ctx.set_clear_color([float(c) for c in LOOK.get("clear_color", (0.0, 0.0, 0.0))])
        with open(path, "rb") as fh:
            raw = fh.read()
        tc = F.TextureConfig()
        tc.width, tc.height, tc.format = len(raw), 0, F.PixelFormat.PIXEL_FORMAT_KTX
        tc.sampler_type, tc.color_space = F.SamplerType.TEXTURE_CUBE, F.ColorSpace.COLORSPACE_LINEAR
        self.ibl_tex = self.ctx.create_texture(tc)
        td = F.TextureData()
        td.set_image(raw)
        self.ibl_tex.upload(td)
        self.ibl_intensity = float(ibl.get("intensity", 20000.0))
        self.opt = mujoco.MjvOption()      # what the cameras show: geom groups 0-2 (as the classic renderer), no sites
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[:3] = 1
        self.opt.sitegroup[:] = 0

    def new_session(self, model):
        sid = self.next_sid
        self.next_sid += 1
        self.models[sid] = model
        return sid

    def free_session(self, sid):
        self.models.pop(sid, None)
        if self.active == sid:
            self._evict()

    def _evict(self):
        while self.live:
            self.live.pop(0)   # renderables and lights, the image light, the scene, then the model resources
        gc.collect()
        try:                   # hand the freed model resources back to the system
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
        self.active = None

    def _activate(self, sid):
        if self.active == sid:
            return
        model = self.models[sid]
        self._evict()
        F = self.F
        res = F.ModelResources(self.ctx, model)
        scn = self.ctx.create_scene(F.SceneParams())
        scn.configure_from_model(model)
        lights = F.ModelLights(scn, res)
        for name, L in LOOK.get("lights", {}).items():     # a light's own shadow map size (else the view's map_size)
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name)
            if i >= 0 and "shadow_map" in L:
                lights.get_light(i).set_shadow_map_size(int(L["shadow_map"]))
        rends = F.ModelRenderables(scn, res)
        rends.set_options(self.opt)
        lp = F.LightParams()
        lp.type, lp.texture, lp.color, lp.intensity = F.LightType.LIGHT_IMAGE, self.ibl_tex, \
            [float(c) for c in LOOK.get("ibl", {}).get("color", (1.0, 1.0, 1.0))], self.ibl_intensity
        ibl = self.ctx.create_light(lp)
        scn.add_light(ibl)
        self.live = [rends, lights, ibl, scn, res]
        self.active = sid

    def _target(self, width, height):
        F = self.F
        key = (width, height)
        if key not in self.targets:     # one target and one read request per size (a request per frame leaks)
            buf = np.zeros((height, width, 3), np.uint8)
            tgt = self.ctx.create_render_target(F.RenderTargetConfig(color_format=F.PixelFormat.PIXEL_FORMAT_RGB8))
            tgt.resize(width, height)
            read = F.ReadRequest()
            read.target = tgt
            read.set_buffer(buf)
            self.targets[key] = (tgt, read, buf)
        return self.targets[key]

    def draw(self, sid, data, pos, forward, up, fovy, width, height):
        """RGB (rows top-down) of the session's scene in the state `data`, from an exact pinhole camera."""
        F = self.F
        self._activate(sid)
        rends, lights, _ibl, scn, _res = self.live
        lights.update(data)
        rends.update(data)
        tgt, read, buf = self._target(width, height)
        top = CAM_NEAR * math.tan(math.radians(fovy) / 2.0)
        cam = F.Camera(pos=np.asarray(pos, np.float32), forward=np.asarray(forward, np.float32),
                       up=np.asarray(up, np.float32), frustum_center=0.0, frustum_width=top * width / height,
                       frustum_bottom=-top, frustum_top=top, frustum_near=CAM_NEAR, frustum_far=CAM_FAR)
        req = F.RenderRequest(camera=cam, draw_mode=F.DrawMode.DRAW_MODE_DEFAULT)
        req.scene, req.target = scn, tgt
        req.viewport.left, req.viewport.bottom, req.viewport.width, req.viewport.height = 0, 0, width, height
        self.ctx.wait_for_frame(self.ctx.render([req], [read]))
        return buf.copy()

    def _shutdown(self):
        """Every Filament object before the context (destroying the context first panics)."""
        self.models.clear()
        self._evict()
        self.targets.clear()
        self.ibl_tex = None
        gc.collect()
        self.ctx = None
        gc.collect()


@atexit.register
def _stop_renderer():
    if _Filament._inst is not None:
        _Filament._inst.stop()


# ------------------------------------------------------------------------------------------------ the cameras
class CameraRig:
    """The cell's four cameras: Filament colour, classic depth, the sensor model. The look is applied to the cell's
    model at the first frame (render-only fields)."""

    def __init__(self, m: mujoco.MjModel, sizes: dict, rng: np.random.Generator):
        self.m = m
        self.sizes = sizes                    # camera name -> (width, height)
        self._renderers = {}
        self._view = None
        self.rng = rng
        self.tint = {name: rng.uniform(0.94, 1.06, size=3) for name in sizes}
        self.gain = {name: float(rng.uniform(0.92, 1.08)) for name in sizes}
        self.look_record = None
        self.rgb_wall = {}                    # camera name -> [frames, seconds] of Filament colour

    def _renderer(self, w, h):
        key = (w, h)
        if key not in self._renderers:
            r = mujoco.Renderer(self.m, h, w)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0       # depth only: no shadow or reflection passes
            r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            self._renderers[key] = r
        return self._renderers[key]

    def _filament(self):
        if self._view is None:
            self.look_record = apply_look(self.m)
            self._view = _Filament.instance().open(self.m)
        return self._view

    def close(self):
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        self._renderers = {}
        if self._view is not None:
            try:
                self._view.close()
            except Exception:  # noqa: BLE001
                pass
            self._view = None

    def rgb_clean(self, d, name: str) -> np.ndarray:
        """The camera's colour image before the sensor model (uint8, rows top-down)."""
        w, h = self.sizes[name]
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        R = d.cam_xmat[cid].reshape(3, 3)
        t = time.perf_counter()
        img = self._filament().render(d, d.cam_xpos[cid], -R[:, 2], R[:, 1], float(self.m.cam_fovy[cid]), w, h)
        st = self.rgb_wall.setdefault(name, [0, 0.0])
        st[0] += 1
        st[1] += time.perf_counter() - t
        return img

    def render(self, d, name: str, depth: bool = True, kind: str = "ceiling"):
        w, h = self.sizes[name]
        rgb = self.rgb_clean(d, name).astype(np.float32) / 255.0
        rgb = np.clip(rgb * self.tint[name] * self.gain[name] + self.rng.normal(0.0, 0.008, size=rgb.shape), 0, 1)
        rgb8 = (rgb * 255.0 + 0.5).astype(np.uint8)
        dep = None
        if depth:
            r = self._renderer(w, h)
            r.enable_depth_rendering()
            r.update_scene(d, camera=name)
            z = r.render().astype(np.float64)
            r.disable_depth_rendering()
            sigma = 0.0015 + 0.0025 * z * z
            zn = z + self.rng.normal(0.0, 1.0, size=z.shape) * sigma
            invalid = (z < DEPTH_MIN[kind]) | (z > DEPTH_MAX) | (self.rng.random(z.shape) < 0.02)
            # discontinuities: relative jump > 3 % to a 4-neighbour
            jump = np.zeros_like(z, dtype=bool)
            for ax in (0, 1):
                dz = np.abs(np.diff(z, axis=ax)) > 0.03 * np.minimum(z.take(range(z.shape[ax] - 1), axis=ax),
                                                                     z.take(range(1, z.shape[ax]), axis=ax))
                if ax == 0:
                    jump[:-1] |= dz
                    jump[1:] |= dz
                else:
                    jump[:, :-1] |= dz
                    jump[:, 1:] |= dz
            invalid |= jump
            mm = np.clip(np.round(zn * 1000.0), 0, 65535).astype(np.uint16)
            mm[invalid] = 0
            dep = mm
        return rgb8, dep


def encode_frame(name: str, index: int, rgb8: np.ndarray, depth_mm, intr: dict) -> dict:
    buf = io.BytesIO()
    Image.fromarray(rgb8).save(buf, format="PNG", compress_level=1)
    out = {"camera": name, "frame_index": index, "width": int(rgb8.shape[1]), "height": int(rgb8.shape[0]),
           "format": "png", "rgb_png_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "intrinsics": intr}
    if depth_mm is not None:
        buf = io.BytesIO()
        Image.fromarray(depth_mm).save(buf, format="PNG", compress_level=1)
        out.update({"sensor_depth_u16_mm_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                    "depth_unit_m": 0.001, "depth_invalid_value": 0})
    return out
