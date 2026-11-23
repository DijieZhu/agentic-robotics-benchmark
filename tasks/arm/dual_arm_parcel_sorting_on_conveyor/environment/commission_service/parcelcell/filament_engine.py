"""The process's Filament renderer (cell internals): MuJoCo's Filament module on Vulkan (Mesa's lavapipe on the CPU in
the task image), in one dedicated thread.

What lives for the process (made once, in the renderer's thread): the context; the image light (the look's KTX cube);
the station's surroundings from the render bundle (``static.npz`` / ``static.json``: the hall near the station, the
conveyor, the station's hardware; one mesh, one material and one renderable per bundle material); the robot and its
grippers (``robot.npz`` / ``robot.json``: one renderable per drawn part, posed at every frame from its body's pose in
the shift's render twin); the spot lamps of the look; one render target and one read request per image size.
What lives per shift (a "session"): the render twin's model resources (the parcels with their own meshes and maps, the
wrist cameras' housings): one session's resources are live at a time; activating another frees the live one first.

Facts this relies on (MuJoCo 3.14, measured): the engine is bound to the thread that created it, so callers on any
thread post jobs and wait (``call``); objects made directly (meshes, materials, renderables, lights) can share a scene
with a model's renderables; a texture marked sRGB is decoded as sRGB (MuJoCo's "auto" would be read as linear);
a material's opacity map is not applied (cut-outs are drawn blended at their coverage); Filament writes its start-up
lines (engine, Vulkan device) and its teardown lines to the process's C standard output, so file descriptor 1 points
at standard error while the engine starts, after every job and while it stops (``_quiet``); the view's temporal
dithering would add a new +-1 noise to every frame (and FXAA turns it into differences of tens of levels along edges),
so it is switched off on the view (``_undither``: Filament's exported View::setDithering, the view found by its state
after a probe frame, searched for again more deeply and with a new probe when not found; the same is done for the
shadow filter, View::setShadowType, and the anti-aliasing). The view is searched for as the renderer starts, before
any camera frame: a start whose search does not find it is taken down and made again from a new context (where the
binding keeps its view depends on its allocations), up to ``START_ATTEMPTS`` (3) starts in the process, and the
renderer fails closed when the last one does not find it either; the number of starts is in the view state (the
shift record's renderer field), and a start that finds the view gives the same frames however many it took. Teardown
frees every Filament object before the context (the context first would crash at exit); its job threads would each
take a glibc malloc arena, so arenas are capped at import and the heap trimmed after a session is freed.
Every failure Python sees is raised as ``RenderError`` (an error of the cell, never of a program); a job that does not
return within the look's timeout marks the renderer broken for the process.

Frames are not promised to be identical across processes: with dithering off (or the renderer failed closed), a few
pixels at object edges can still differ between two processes that draw the same state (seen in some runs: 5 to 7
pixels of a frame, at silhouettes; the cause is not isolated). Within one process a frame asked again is the same
bytes, and the physics never depends on a frame. A re-grade of a program that acts on its camera images is therefore
a sample, not a replay.
"""

import atexit
import ctypes
import gc
import json
import math
import os
import queue
import struct
import sys
import threading
import time
import weakref


def _cap_malloc_arenas():
    try:
        ctypes.CDLL("libc.so.6").mallopt(-8, 2)            # M_ARENA_MAX = 2, before the job threads allocate
    except (OSError, AttributeError):
        pass


_cap_malloc_arenas()

import numpy as np

__all__ = ["Engine", "RenderError", "engine_info"]


START_ATTEMPTS = 3      # starts of the renderer in one process before it fails closed (the view search, ``_undither``)


class RenderError(RuntimeError):
    """The renderer failed: an error of the cell (the grader's or the service's environment), not of a program."""


def _trim_heap():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class _quiet(object):
    """File descriptor 1 points at standard error for the block; C's buffers are flushed before it is given back, so
    nothing Filament wrote meanwhile reaches the process's standard output."""

    def __enter__(self):
        try:
            sys.stdout.flush()
        except (AttributeError, OSError, ValueError):
            pass
        self.libc = ctypes.CDLL(None)
        self.libc.fflush(None)
        try:
            self.saved = os.dup(1)
            os.dup2(2, 1)
        except OSError:
            self.saved = None
        return self

    def __exit__(self, *exc):
        self.libc.fflush(None)
        if self.saved is not None:
            os.dup2(self.saved, 1)
            os.close(self.saved)
        return False


def _lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def gain_srgb8(img, gain):
    """An sRGB 8-bit image times a linear gain per channel (one table per channel; clipped at full value)."""
    x = _lin(np.arange(256) / 255.0)
    g = np.broadcast_to(np.asarray(gain, np.float64), (3,))
    out = np.empty_like(img)
    for c in range(3):
        lut = np.round(_srgb(x * float(g[c])) * 255.0).astype(np.uint8)
        out[..., c] = lut[img[..., c]]
    return out


def gain_rgb(rgb, gain):
    return [float(v) for v in _srgb(_lin(np.asarray(rgb, float)[:3]) * np.asarray(gain, float))]


class Engine(object):
    """The process's renderer. ``Engine.instance(render_dir, look)``; ``call(fn, *args)`` runs a job in the
    renderer's thread and returns its result (or raises RenderError)."""

    _inst = None
    _failed = None
    _last_start = None      # (pid, key, {"attempts", "view_found", "errors"}) of the process's last start
    _guard = threading.Lock()

    @classmethod
    def instance(cls, render_dir, look):
        key = (os.path.abspath(render_dir), json.dumps(look, sort_keys=True))
        with cls._guard:
            pid = os.getpid()
            if cls._failed is not None and cls._failed[0] == pid and cls._failed[1] == key:
                raise RenderError(cls._failed[2])
            inst = cls._inst
            if inst is not None and inst.pid == pid and inst.key != key:
                inst.stop()                                  # another look in the same process: start afresh
                inst = cls._inst = None
            if inst is None or inst.pid != pid:
                cls._inst = None
                errors = []
                for n in range(1, START_ATTEMPTS + 1):
                    try:
                        inst = cls(render_dir, look, key)
                    except ViewNotFound as exc:          # a new context places the view elsewhere: start again
                        errors.append(str(exc))
                        if n < START_ATTEMPTS:
                            continue
                        msg = "%s (%d starts of the renderer)" % (exc, n)
                        cls._failed = (pid, key, msg)
                        cls._last_start = (pid, key, {"attempts": n, "view_found": False, "errors": errors})
                        raise RenderError(msg)
                    except RenderError as exc:
                        cls._failed = (pid, key, str(exc))
                        cls._last_start = (pid, key, {"attempts": n, "view_found": False,
                                                      "errors": errors + [str(exc)]})
                        raise
                    inst.view_state["attempts"] = n
                    cls._last_start = (pid, key, {"attempts": n, "view_found": bool(inst.view_state.get("view_found")),
                                                  "errors": errors})
                    cls._inst = inst
                    break
            return cls._inst

    @classmethod
    def start_state(cls):
        """{"attempts", "view_found", "errors"} of this process's last start of the renderer, or None."""
        ls = cls._last_start
        if ls is None or ls[0] != os.getpid():
            return None
        return dict(ls[2])

    def __init__(self, render_dir, look, key):
        self.pid = os.getpid()
        self.key = key
        self.render_dir = os.path.abspath(render_dir)
        self.look = look
        self.timeout = float(look.get("render", {}).get("timeout_s", 300.0))            # start, a shift's first frame
        self.frame_timeout = float(look.get("render", {}).get("frame_timeout_s", 60.0))  # any later frame
        self.broken = None
        self.jobs = queue.Queue()
        self.sessions = {}
        self.active = None
        self.live = {}
        self.targets = {}
        self.next_sid = 0
        self.view = None
        self.view_state = {}
        self.stats = {"activations": 0, "frames": 0, "targets": 0, "sessions": 0}
        self.thread = threading.Thread(target=self._loop, name="parcelcell-filament", daemon=True)
        self.thread.start()
        try:
            self.call(self._start)
        except RenderError as exc:
            self.post(None)
            if isinstance(exc, ViewNotFound):
                self.thread.join(60.0)                   # its objects freed in its own thread, before another start
            raise

    # ---------------------------------------------------------------- any thread
    def call(self, fn, *args, timeout=None):
        """Run fn(*args) in the renderer's thread and return its result. A job that does not return within
        ``timeout`` (default: the look's timeout_s) marks the renderer broken for the process."""
        if self.broken:
            raise RenderError(self.broken)
        if not self.thread.is_alive():
            raise RenderError("renderer: its thread has ended")
        limit = float(self.timeout if timeout is None else timeout)
        done = threading.Event()
        out = []
        self.jobs.put((fn, args, done, out))
        if not done.wait(limit):
            self.broken = "renderer: a call did not return within %.0f s" % limit
            raise RenderError(self.broken)
        ok, val = out[0]
        if not ok:
            raise val
        return val

    def post(self, fn, *args):
        if self.pid == os.getpid() and self.thread.is_alive():
            self.jobs.put((fn, args, None, None))

    def stop(self, timeout=60.0):
        if self.pid == os.getpid() and self.thread.is_alive() and not self.broken:
            self.jobs.put((None, (), None, None))
            self.thread.join(timeout)

    def _loop(self):
        while True:
            fn, args, done, out = self.jobs.get()
            if fn is None:
                try:
                    with _quiet():
                        self._shutdown()
                except BaseException:  # noqa: BLE001
                    pass
                return
            try:
                with _quiet():
                    res = (True, fn(*args))
            except BaseException as exc:  # noqa: BLE001 - a new exception without the traceback: the traceback's
                # frames hold Filament objects, and freeing them in the caller's thread would abort the process
                res = (False, (type(exc) if isinstance(exc, RenderError) else RenderError)(
                    str(exc) if isinstance(exc, RenderError) else "renderer: %s: %s" % (type(exc).__name__, exc)))
                del exc
            if done is not None:
                out.append(res)
                done.set()

    # ---------------------------------------------------------------- the renderer's thread only
    def _start(self):
        t0 = time.perf_counter()
        for k in ("DISPLAY", "WAYLAND_DISPLAY"):          # offscreen: Filament would connect to a named display
            os.environ.pop(k, None)
        rv = self.look.get("render", {})
        try:
            ncpu = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            ncpu = os.cpu_count() or 4
        lp = int(rv.get("lavapipe_threads", 8))
        os.environ["LP_NUM_THREADS"] = str(max(1, min(lp, ncpu)))
        import mujoco
        import mujoco._render_filament as F
        from mujoco.experimental.studio import window  # registers Filament's resource providers (imported for that)
        self.mujoco, self.F = mujoco, F
        self.ctx = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN))
        self.ctx.set_clear_color([float(x) for x in rv.get("clear_color", [0.0, 0.0, 0.0])])
        self.scn = self.ctx.create_scene(F.SceneParams())
        self._configure()
        with open(os.path.join(self.render_dir, self.look["ibl"]["file"]), "rb") as fh:
            raw = fh.read()
        tc = F.TextureConfig()
        tc.width, tc.height, tc.format = len(raw), 0, F.PixelFormat.PIXEL_FORMAT_KTX
        tc.sampler_type, tc.color_space = F.SamplerType.TEXTURE_CUBE, F.ColorSpace.COLORSPACE_LINEAR
        self.ibl_tex = self.ctx.create_texture(tc)
        self.ibl_td = F.TextureData()
        self.ibl_td.set_image(raw)
        self.ibl_tex.set_data(self.ibl_td)
        lpar = F.LightParams()
        lpar.type, lpar.texture, lpar.color = F.LightType.LIGHT_IMAGE, self.ibl_tex, [1.0, 1.0, 1.0]
        lpar.intensity = float(self.look["ibl"]["intensity"])
        self.ibl = self.ctx.create_light(lpar)
        self.scn.add_light(self.ibl)
        self.lamps = []
        for L in self.look.get("lamps", []):
            p = F.LightParams()
            p.type = F.LightType.LIGHT_SPOT if L.get("type", "spot") == "spot" else F.LightType.LIGHT_POINT
            p.color = [float(x) for x in L.get("color", [1.0, 1.0, 1.0])]
            p.intensity = float(L["candela"])
            p.cast_shadows = bool(L.get("shadow", False))
            if p.cast_shadows:
                p.shadow_map_size = int(L.get("shadow_map", 1024))
            p.spot_cone_angle = float(L.get("cutoff_deg", 80.0))
            p.spot_softness = float(L.get("softness", 0.0))
            p.bulb_radius = float(L.get("bulb_m", 0.02))
            p.range = float(L.get("range_m", 30.0))
            li = self.ctx.create_light(p)
            d = np.asarray(L.get("dir", [0.0, 0.0, -1.0]), np.float64)
            li.set_transform(np.asarray(L["pos"], np.float32), (d / np.linalg.norm(d)).astype(np.float32))
            self.scn.add_light(li)
            self.lamps.append((li, float(L["candela"])))
        self._exposure = None
        self._textures = []
        self.static = self._load_static()
        self.robot = self._load_robot()
        self.view = None
        if rv.get("undither", True):
            # the view search, before any camera frame (a probe frame through a fixed camera; its pixels are dropped)
            self._undither(F.Camera(pos=np.asarray([0.0, 0.0, 3.0], np.float32),
                                    forward=np.asarray([0.0, 0.0, -1.0], np.float32),
                                    up=np.asarray([0.0, 1.0, 0.0], np.float32), frustum_center=0.0,
                                    frustum_width=0.04, frustum_bottom=-0.03, frustum_top=0.03, frustum_near=0.05,
                                    frustum_far=60.0))
        self.start_s = time.perf_counter() - t0

    def _configure(self):
        """The look's view options travel as custom fields of a tiny model (MuJoCo reads filament.* from them)."""
        mujoco = self.mujoco
        spec = mujoco.MjSpec()
        vw = self.look.get("view", {})
        for k, v in sorted(vw.get("numeric", {}).items()):
            spec.add_numeric(name=k, data=[float(x) for x in (v if isinstance(v, (list, tuple)) else [v])])
        for k, v in sorted(vw.get("text", {}).items()):
            spec.add_text(name=k, data=str(v))
        spec.visual.headlight.active = 0
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], pos=[0, 0, -50])
        self.look_model = spec.compile()
        self.scn.configure_from_model(self.look_model)

    def _texture(self, img, srgb=True):
        F = self.F
        h, w = img.shape[:2]
        tc = F.TextureConfig()
        tc.width, tc.height, tc.format = int(w), int(h), F.PixelFormat.PIXEL_FORMAT_RGB8
        tc.sampler_type = F.SamplerType.TEXTURE_2D
        tc.color_space = F.ColorSpace.COLORSPACE_SRGB if srgb else F.ColorSpace.COLORSPACE_LINEAR
        t = self.ctx.create_texture(tc)
        td = F.TextureData()
        td.set_image(np.ascontiguousarray(img, np.uint8).tobytes())
        t.set_data(td)
        self._textures.append((t, td))
        return t

    def _mesh(self, V, N, U, I, Q=None):
        """Q: per-vertex tangent frames (quaternions x, y, z, w) for a normal-mapped material; the binding takes
        either normals or tangent frames, never both."""
        F = self.F
        A, Us, Ty = F.VertexAttribute, F.VertexAttributeUsage, F.VertexAttributeType
        if Q is not None:
            attrs = [A(usage=Us.VERTEX_ATTRIBUTE_USAGE_POSITION, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                     A(usage=Us.VERTEX_ATTRIBUTE_USAGE_TANGENTS, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT4),
                     A(usage=Us.VERTEX_ATTRIBUTE_USAGE_UV, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT2)]
            N = Q
        else:
            attrs = [A(usage=Us.VERTEX_ATTRIBUTE_USAGE_POSITION, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                     A(usage=Us.VERTEX_ATTRIBUTE_USAGE_NORMAL, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                     A(usage=Us.VERTEX_ATTRIBUTE_USAGE_UV, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT2)]
        cfg = F.MeshConfig(max_vertices=int(len(V)), max_indices=int(len(I)), attributes=attrs, interleaved=False,
                           index_type=F.IndexType.INDEX_TYPE_U32,
                           primitive_type=F.MeshPrimitiveType.MESH_PRIMITIVE_TYPE_TRIANGLES)
        mesh = self.ctx.create_mesh(cfg)
        md = F.MeshData(compute_bounds=True)
        md.set_vertices(cfg, np.concatenate([np.asarray(V, np.float32).ravel(), np.asarray(N, np.float32).ravel(),
                                             np.asarray(U, np.float32).ravel()]).tobytes())
        md.set_indices(cfg, np.ascontiguousarray(I, np.uint32).tobytes())
        mesh.set_data(md)
        return mesh, md

    def _bundle_tex(self, z, k, gain, cache):
        key = (int(k), tuple(np.round(np.broadcast_to(np.asarray(gain, float), (3,)), 5)))
        if key not in cache:
            o, w, h = int(z["tex_ofs"][k]), int(z["tex_w"][k]), int(z["tex_h"][k])
            img = z["tex"][o:o + w * h * 3].reshape(h, w, 3)
            if not np.allclose(gain, 1.0):
                img = gain_srgb8(img, gain)
            cache[key] = self._texture(img)
        return cache[key]

    def _bundle_map(self, z, k, cache, channel=None):
        """A linear map (normal or ORM) of the bundle's map list (the near set's surface maps); channel:
        one channel of it replicated to RGB (the binding's ORM input does not read roughness from G, measured; its
        roughness input does)."""
        key = ("map", int(k), channel)
        if key not in cache:
            o, w, h = int(z["mtex_ofs"][k]), int(z["mtex_w"][k]), int(z["mtex_h"][k])
            img = z["mtex"][o:o + w * h * 3].reshape(h, w, 3)
            if channel is not None:
                img = np.repeat(img[..., channel:channel + 1], 3, axis=2)
            cache[key] = self._texture(img, srgb=False)
        return cache[key]

    def _maps(self, z, rec, cache):
        mode = self.look.get("render", {}).get("surface_maps", True)
        if not mode or "mtex" not in z:
            return None, None
        ni, oi = int(rec.get("nrm_tex", -1)), int(rec.get("orm_tex", -1))
        if mode == "orm":
            ni = -1
        elif mode == "normal":
            oi = -1
        return (self._bundle_map(z, ni, cache) if ni >= 0 else None), (self._bundle_map(z, oi, cache, channel=1) if oi >= 0 else None)

    def _material(self, rec, gain, tex, rough_scale=1.0, emissive_scale=1.0, nrm=None, orm=None):
        F = self.F
        g = np.broadcast_to(np.asarray(gain, float), (3,))
        rgba = list(rec["rgba"])
        if tex is None:
            rgba[:3] = gain_rgb(rgba[:3], g)
        alpha = float(rgba[3]) if rec.get("blend") is not None else 1.0   # below 1: Filament's blended path
        kw = dict(color=np.asarray(rgba[:3] + [alpha], np.float32),
                  roughness=float(np.clip(float(rec["roughness"]) * rough_scale, 0.05, 1.0)),
                  metallic=float(rec["metallic"]), specular=float(rec.get("specular", 0.5)))
        em = float(rec.get("emissive", 0.0)) * emissive_scale
        if em > 0:
            kw["emissive"] = em
        if tex is not None:
            kw["color_texture"] = tex
        if nrm is not None:
            kw["normal_texture"] = nrm
        if orm is not None:
            kw["roughness_texture"] = orm          # the roughness channel of the bundle's ORM map; metallic stays constant
            kw["roughness"] = 1.0                  # the binding multiplies the map (linear) by the constant (measured)
        return F.Material(**kw)

    def _load_static(self):
        F = self.F
        meta = json.load(open(os.path.join(self.render_dir, "static.json")))
        z = dict(np.load(os.path.join(self.render_dir, "static.npz")))
        sl = self.look.get("static", {})
        classes = sl.get("classes", {})
        out = []
        cache = {}
        for m in meta["materials"]:
            if m["icnt"] == 0:
                continue
            c = classes.get(m["cls"], {})
            if c.get("hide"):
                continue
            gain = np.broadcast_to(np.asarray(c.get("gain", 1.0), float), (3,)) * float(sl.get("albedo_gain", 1.0))
            tex = self._bundle_tex(z, m["tex"], gain, cache) if m["tex"] >= 0 else None
            nrm, orm = self._maps(z, m, cache)
            mat = self._material(m, gain, tex, rough_scale=float(c.get("roughness_scale", 1.0)),
                                 emissive_scale=float(c.get("emissive_scale", sl.get("emissive_scale", 1.0))), nrm=nrm, orm=orm)
            v0, vc, i0, ic = m["vofs"], m["vcnt"], m["iofs"], m["icnt"]
            mesh, md = self._mesh(z["V"][v0:v0 + vc], z["N"][v0:v0 + vc], z["UV"][v0:v0 + vc],
                                  z["I"][i0:i0 + ic],            # indices are the material's own (from 0)
                                  Q=z["Q"][v0:v0 + vc] if nrm is not None else None)
            blend = m.get("blend") is not None
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadows=bool(c.get("cast_shadows", not blend)),
                                                              receive_shadows=True))
            r.set_mesh(mesh, 0, int(ic))
            r.set_material(mat)
            r.set_transform(np.zeros(3, np.float32), np.eye(3, dtype=np.float32).ravel())
            self.scn.add_renderable(r)
            out.append((r, mat, mesh, md))
        self.static_stats = meta["stats"]
        return out

    def _load_robot(self):
        F = self.F
        meta = json.load(open(os.path.join(self.render_dir, "robot.json")))
        z = dict(np.load(os.path.join(self.render_dir, "robot.npz")))
        rl = self.look.get("robot", {})
        classes = rl.get("classes", {})
        mats, cache = {}, {}
        has_nrm = {}
        parts = []
        for p in meta["parts"]:
            cn = p["cls"]
            if cn not in mats:
                base = dict(meta["classes"][cn])
                c = classes.get(cn, {})
                for k in ("roughness", "metallic"):
                    if k in c:
                        base[k] = c[k]
                gain = np.broadcast_to(np.asarray(c.get("gain", 1.0), float), (3,)) * float(rl.get("albedo_gain", 1.0))
                tex = self._bundle_tex(z, base["tex"], gain, cache) if base.get("tex", -1) >= 0 else None
                nrm, orm = self._maps(z, base, cache)
                mats[cn] = (base, gain, tex, nrm, orm)
                has_nrm[cn] = nrm is not None
            v0, vc, i0, ic = p["vofs"], p["vcnt"], p["iofs"], p["icnt"]
            mesh, md = self._mesh(z["V"][v0:v0 + vc], z["N"][v0:v0 + vc], z["UV"][v0:v0 + vc], z["I"][i0:i0 + ic],
                                  Q=z["Q"][v0:v0 + vc] if has_nrm.get(cn) else None)
            # one material object per drawn part (the class's values)
            base, gain, tex, nrm, orm = mats[cn]
            mat = self._material(base, gain, tex, nrm=nrm, orm=orm)
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadows=True, receive_shadows=True))
            r.set_mesh(mesh, 0, int(ic))
            r.set_material(mat)
            r.set_transform(np.zeros(3, np.float32), np.eye(3, dtype=np.float32).ravel())
            self.scn.add_renderable(r)
            parts.append({"body": p["body"], "r": r, "mesh": mesh, "md": md, "mat": mat})
        self.robot_mats = mats
        self.robot_stats = meta["stats"]
        return parts

    # ---------------------------------------------------------------- sessions (one shift each)
    def new_session(self, model, robot_bodies):
        """model: the shift's render twin (MjModel); robot_bodies: {body name: twin body id} of the robot's parts."""
        sid = self.next_sid
        self.next_sid += 1
        missing = sorted({p["body"] for p in self.robot} - set(robot_bodies))
        if missing:
            raise RenderError("renderer: the twin lacks the robot bodies %s" % ", ".join(missing[:5]))
        self.sessions[sid] = (model, [int(robot_bodies[p["body"]]) for p in self.robot])
        self.stats["sessions"] += 1
        return sid

    def free_session(self, sid):
        self.sessions.pop(sid, None)
        if self.active == sid:
            self._evict()

    def invalidate(self, sid):
        """The session's model changed (a texture): its renderables are made again at its next frame."""
        if self.active == sid:
            self._evict()
        self.stats["invalidations"] = self.stats.get("invalidations", 0) + 1

    def _evict(self):
        L = self.live
        for k in ("rends", "objs"):                      # renderables before the resources they draw
            L.pop(k, None)
        L.clear()
        gc.collect()
        _trim_heap()
        self.active = None

    def _activate(self, sid):
        if self.active == sid:
            return
        F = self.F
        t0 = time.perf_counter()
        model, _ids = self.sessions[sid]
        self._evict()
        objs = F.ModelObjects(self.ctx, model)
        rends = F.ModelRenderables(self.scn, objs)
        self.live = {"objs": objs, "rends": rends}
        self.active = sid
        self.stats["activations"] += 1
        self.last_activation_s = time.perf_counter() - t0

    def _target(self, width, height):
        F = self.F
        key = (int(width), int(height))
        if key not in self.targets:                      # one target and one read request per size
            tgt = self.ctx.create_render_target(F.RenderTargetConfig(color_format=F.PixelFormat.PIXEL_FORMAT_RGB8))
            tgt.resize(key[0], key[1])
            rd = F.ReadPixelsRequest()
            rd.target = tgt
            rd.alloc(key[0] * key[1] * 3)
            self.targets[key] = (tgt, rd)
            self.stats["targets"] += 1
        return self.targets[key]

    def _set_exposure(self, e):
        if self._exposure == e:
            return
        self.ibl.set_intensity(float(self.look["ibl"]["intensity"]) * e)
        for li, cd in self.lamps:
            li.set_intensity(max(1e-6, cd * e))
        self._exposure = e

    def draw(self, sid, data, pos, forward, up, fovy, width, height, exposure, near, far):
        """RGB (rows top-down, uint8) of the session's twin in the state ``data`` (its kinematics computed), through an
        exact pinhole camera: centre ``pos``, axes ``forward`` / ``up``, vertical field of view ``fovy`` (degrees),
        principal point at the image centre."""
        F = self.F
        self._activate(sid)
        _model, ids = self.sessions[sid]
        xpos, xmat = data.xpos, data.xmat
        for p, b in zip(self.robot, ids):
            p["r"].set_transform(np.asarray(xpos[b], np.float32), np.asarray(xmat[b], np.float32))
        self.live["rends"].update(data)
        self._set_exposure(float(exposure))
        tgt, rd = self._target(width, height)
        top = near * math.tan(math.radians(fovy) / 2.0)
        cam = F.Camera(pos=np.asarray(pos, np.float32), forward=np.asarray(forward, np.float32),
                       up=np.asarray(up, np.float32), frustum_center=0.0, frustum_width=top * width / height,
                       frustum_bottom=-top, frustum_top=top, frustum_near=float(near), frustum_far=float(far))
        req = F.RenderRequest(camera=cam, draw_mode=F.DrawMode.DRAW_MODE_DEFAULT)
        req.scene, req.target = self.scn, tgt
        req.viewport.left, req.viewport.bottom, req.viewport.width, req.viewport.height = 0, 0, int(width), int(height)
        if self.view is None and self.look.get("render", {}).get("undither", True):
            self._undither(cam)
        self.ctx.wait_for_frame(self.ctx.render([req], [rd]))
        self.stats["frames"] += 1
        return np.frombuffer(rd.buffer(), np.uint8).reshape(int(height), int(width), 3).copy()

    # Filament's exported View setters, called on the view the binding draws through (found by its state after a
    # probe frame: viewport (0, 0, width, height) at +0x268, anti-aliasing FXAA at +0x2b1, dithering TEMPORAL at
    # +0x2b2, shadowing on at +0x2b3, post-processing on at +0x2b5; offsets read from the exported getters of the
    # Filament linked into MuJoCo 3.14: getViewport, getAntiAliasing, getDithering, isShadowingEnabled,
    # isPostProcessingEnabled). Memory is read through write(2) into a pipe, so a wrong guess cannot crash.
    SYM = {"set_dither": "_ZN8filament4View12setDitheringENS_9DitheringE",
           "get_dither": "_ZNK8filament4View12getDitheringEv",
           "set_aa": "_ZN8filament4View15setAntiAliasingENS_12AntiAliasingE",
           "get_aa": "_ZNK8filament4View15getAntiAliasingEv",
           "set_shadow": "_ZN8filament4View13setShadowTypeENS_10ShadowTypeE",
           "get_shadow": "_ZNK8filament4View13getShadowTypeEv"}
    SHADOW_TYPES = {"pcf": 0, "vsm": 1, "dpcf": 2, "pcss": 3, "pcfd": 4}
    AA_TYPES = {"none": 0, "fxaa": 1}

    _probes = [0]

    def _undither(self, cam):
        """Find the view by a probe frame of a size no other view of the process has had (a renderer started again
        in the same process leaves its freed views' memory behind, and their last viewports are camera sizes), then
        switch its dithering off and set its shadow filter and anti-aliasing. The view is searched for through the
        pointers reachable from the binding's scene, context, render target and read request, two hops deep first and
        three when that finds nothing (where the binding keeps the view depends on its allocations, which differ from
        process to process); up to three probe frames. A view that cannot be found would leave the frames with the
        view's temporal dithering (each frame then depends on how many frames the process drew before it), so the
        renderer fails closed instead: every later frame is a camera failure of the cell, never a dithered frame."""
        F = self.F
        lib = ctypes.CDLL(self.F.__file__)
        try:
            fn = {k: getattr(lib, v) for k, v in self.SYM.items()}
        except AttributeError as exc:
            raise RenderError("renderer: MuJoCo's Filament module exports no View setter: %s" % exc)
        for k in ("set_dither", "set_shadow", "set_aa"):
            fn[k].restype, fn[k].argtypes = None, [ctypes.c_void_p, ctypes.c_uint8]
        for k in ("get_dither", "get_shadow", "get_aa"):
            fn[k].restype, fn[k].argtypes = ctypes.c_uint8, [ctypes.c_void_p]
        libc = ctypes.CDLL(None, use_errno=True)
        libc.write.restype, libc.write.argtypes = ctypes.c_ssize_t, [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        tries = []
        view = None
        for attempt in range(3):
            Engine._probes[0] += 1
            width, height = 29 + 2 * Engine._probes[0], 19 + 2 * Engine._probes[0]
            tgt, rd = self._target(width, height)
            req = F.RenderRequest(camera=cam, draw_mode=F.DrawMode.DRAW_MODE_DEFAULT)
            req.scene, req.target = self.scn, tgt
            req.viewport.left, req.viewport.bottom, req.viewport.width, req.viewport.height = 0, 0, width, height
            self.ctx.wait_for_frame(self.ctx.render([req], [rd]))
            want = struct.pack("<iiII", 0, 0, width, height)
            for max_depth in (3, 4):
                found, seen = set(), set()
                rfd, wfd = os.pipe()

                def read(addr, n):
                    k = libc.write(wfd, addr, n)
                    return os.read(rfd, k) if k > 0 else None

                def walk(addr, n, depth):
                    b = read(addr, n)
                    for (p,) in struct.iter_unpack("<Q", b[:len(b) // 8 * 8] if b else b""):
                        if p in seen or p < 0x10000 or p > 0x7fffffffffff or p % 8:
                            continue
                        seen.add(p)
                        v = None if p % 16 else read(p + 0x268, 0x4e)
                        if v and len(v) == 0x4e and v[:16] == want and v[0x49] == 1 and v[0x4a] == 1 and v[0x4d] == 1:
                            found.add(p)
                        if depth < max_depth:
                            walk(p, 512, depth + 1)
                try:
                    for root in (id(self.scn), id(self.ctx), id(tgt), id(rd), id(req)):
                        walk(root, 64, 1)
                        if found:
                            break
                finally:
                    os.close(rfd)
                    os.close(wfd)
                cand = found.pop() if len(found) == 1 else None
                tries.append({"probe": [width, height], "depth": max_depth, "candidates": len(found) + (cand is not None)})
                if cand is not None and int(fn["get_dither"](cand)) == 1:   # TEMPORAL, as the binding leaves it
                    view = cand
                    break
            self.targets.pop((width, height), None)
            if view is not None:
                break
        if view is None:
            self.view = 0
            self.view_state = {"dithering": None, "view_found": False, "tries": tries}
            self.broken = ("renderer: the view's temporal dithering could not be switched off (the view was not found: "
                           "%d tries)" % len(tries))
            raise ViewNotFound(self.broken)
        self.view = view
        fn["set_dither"](self.view, 0)                   # Dithering::NONE
        st = self.look.get("render", {}).get("shadow_type")
        before = int(fn["get_shadow"](self.view))
        if st is not None:
            fn["set_shadow"](self.view, int(self.SHADOW_TYPES[str(st).lower()]))
        aa = str(self.look.get("render", {}).get("anti_aliasing", "fxaa")).lower()
        if aa in self.AA_TYPES:
            fn["set_aa"](self.view, int(self.AA_TYPES[aa]))
        self.view_state = {"dithering": int(fn["get_dither"](self.view)), "shadow_type_before": before,
                           "shadow_type": int(fn["get_shadow"](self.view)),
                           "anti_aliasing": int(fn["get_aa"](self.view)), "view_found": True, "tries": tries}

    def info(self):
        return {"stats": dict(self.stats), "start_s": round(self.start_s, 3), "static": self.static_stats,
                "robot": self.robot_stats, "view": dict(self.view_state), "targets": sorted(self.targets),
                "lamps": len(self.lamps), "last_activation_s": getattr(self, "last_activation_s", None),
                "lavapipe_threads": os.environ.get("LP_NUM_THREADS")}

    def _shutdown(self):
        """Every Filament object before the context: sessions, the model renderables, the robot and the static set
        out of the scene and freed, the lights, the scene, the targets, the textures, then the context."""
        self.sessions.clear()
        try:
            self._evict()
        except Exception:  # noqa: BLE001
            self.live = {}
        scn = getattr(self, "scn", None)
        for group in ("robot", "static"):
            for item in getattr(self, group, []) or []:
                r = item["r"] if isinstance(item, dict) else item[0]
                try:
                    scn.remove_renderable(r)
                except Exception:  # noqa: BLE001
                    pass
            setattr(self, group, [])
        self.robot_mats = {}
        gc.collect()
        for li, _cd in getattr(self, "lamps", []):
            try:
                scn.remove_light(li)
            except Exception:  # noqa: BLE001
                pass
        self.lamps = []
        if getattr(self, "ibl", None) is not None and scn is not None:
            try:
                scn.remove_light(self.ibl)
            except Exception:  # noqa: BLE001
                pass
        self.ibl = None
        gc.collect()
        self.scn = None
        self.look_model = None
        gc.collect()
        self.targets.clear()
        self._textures = []
        self.ibl_tex = self.ibl_td = None
        gc.collect()
        self.ctx = None
        gc.collect()
        _trim_heap()


class ViewNotFound(RenderError):
    """The view search of a start found no view (``_undither``): the start is made again, up to START_ATTEMPTS."""


def engine_info():
    e = Engine._inst
    if e is None or e.pid != os.getpid():
        return None
    return e.call(e.info)


@atexit.register
def _stop_engine():
    e = Engine._inst
    if e is not None:
        e.stop()
