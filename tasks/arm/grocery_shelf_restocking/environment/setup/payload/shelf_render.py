"""The shelf cell's cameras: MuJoCo 3.16's Filament renderer (physically based) on Vulkan; in the task's image Vulkan
is Mesa's lavapipe on the CPU (no GPU). shelf_env.render draws every image through this module, in the agent's copy
of the simulator and in the grader's alike.

What is drawn: shelf_scene.py's items (the approved look's own meshes: the gondola bay, cart, tray tote, products
with their print and code label, the Panda with soft finger pads, the telescopic lift, the store aisle round the
bay with its stocked runs, floor, ceiling, walls, door and marker) and the render bundle's stock-handling props
(SHELF_RENDER_DIR, default /opt/task-assets/payload/render: props_bay/, props_world/, textures.npz, textures.json,
ibl.ktx). Light: an image light (ibl.ktx: a path-traced probe of the aisle in front of the bay, with the store's
lamps painted in) and ambient occlusion; no lamp casts a shadow. The cameras' exposure is fixed (shelf_look.json).
Cameras: exactly the simulated cameras (mjv_updateCamera), the frustum from the vertical field of view at the
image's aspect ratio. An image is drawn at a multiple of its size and averaged in linear light (render.supersample).

An image is a function of the unit, the episode, the simulator's state and the camera alone: every renderable,
mesh, texture and material of an episode is made once, in one order, when its view is opened, and a process draws
ONE episode (a second open_view in the same process raises RenderError: shelf_env starts a worker process per
episode); Filament's temporal dithering is switched off (_undither). The sensor noise stays shelf_env.render's.
Every failure the renderer sees is raised as RenderError: an error of the environment, never of a controller.
Nothing here changes the simulation.
"""
import atexit
import copy
import ctypes
import gc
import hashlib
import json
import os
import queue
import struct
import sys
import threading
import time
import weakref

import numpy as np


def _cap_malloc_arenas():
    """Filament's and lavapipe's job threads would each get a glibc malloc arena, and the resident size grows by
    gigabytes over episodes. M_ARENA_MAX must be set before those threads allocate: at import."""
    try:
        ctypes.CDLL("libc.so.6").mallopt(-8, 2)
    except (OSError, AttributeError):
        pass


class _StdoutToStderr(object):
    """Filament writes its start-up banner (engine, Vulkan device) to the process's standard output, where a
    program may speak a wire protocol or print its result: while the renderer starts, file descriptor 1 points at
    standard error, and C's buffers are flushed before it is given back."""

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


def _trim_heap():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


_cap_malloc_arenas()
import mujoco  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.environ.get("SHELF_RENDER_DIR", "/opt/task-assets/payload/render")
LOOK_FILE = os.environ.get("SHELF_LOOK_FILE", os.path.join(HERE, "shelf_look.json"))
BUNDLE_FILES = ("textures.npz", "textures.json", "ibl.ktx", "props_bay/room.npz", "props_bay/room.json",
                "props_world/room.npz", "props_world/room.json")
CALL_TIMEOUT = float(os.environ.get("SHELF_RENDER_TIMEOUT", "300"))
CAM_FAR = 40.0
SKU_SCALE = 0.25               # a task SKU's atlas: the look's 8000 px/m map scaled to 2000 px/m
CODE_SCALE = 0.1               # a code label: the look's 60000 px/m map scaled to 6000 px/m


class RenderError(RuntimeError):
    """The renderer failed: an error of the environment (the practice simulator or the grader), not of a
    controller."""


def load_look(path=LOOK_FILE):
    with open(path) as fh:
        return json.load(fh)


def srgb_from_linear(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def linear_from_srgb(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gain_srgb(rgb, gain):
    """An sRGB colour times a linear gain (a number or one per channel)."""
    return srgb_from_linear(linear_from_srgb(np.asarray(rgb, float)[:3]) * np.asarray(gain, float))


def _gain_srgb8(img, gain):
    """An sRGB 8-bit image times a linear per-channel gain (one lookup table per channel)."""
    x = linear_from_srgb(np.arange(256) / 255.0)
    g = np.broadcast_to(np.asarray(gain, float), (3,))
    out = np.empty_like(img)
    for c in range(3):
        lut = np.round(srgb_from_linear(np.clip(x * float(g[c]), 0.0, 1.0)) * 255.0).astype(np.uint8)
        out[..., c] = lut[img[..., c]]
    return out


_LIN_OF_8 = linear_from_srgb(np.arange(256) / 255.0).astype(np.float32)


def _box_down(img, k):
    """An sRGB 8-bit image drawn at k times the camera's size, averaged over k x k blocks in linear light: the same
    pixel centres and the same lens as the camera's own size. MuJoCo's Filament has no multisampling on the CPU
    driver, and without this a stripe two or three pixels wide (a peg's bands in the fixed cameras) breaks up."""
    h, w = img.shape[0] // k, img.shape[1] // k
    lin = _LIN_OF_8[img].reshape(h, k, w, k, 3).mean(axis=(1, 3))
    return np.round(srgb_from_linear(lin) * 255.0).astype(np.uint8)



def gl_camera(m, d, camid, scn, cam, aspect):
    """MuJoCo's own camera for `camid` in the state `d`: (pos, forward, up, [centre, half width, bottom, top, near]
    of its frustum). A camera given by its vertical field of view has no width of its own (MuJoCo's renderer takes
    it from the image's aspect ratio): half the frustum's height times `aspect` (width / height), as there. The
    near plane is the model's."""
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = int(camid)
    mujoco.mjv_updateCamera(m, d, cam, scn)
    a, b = scn.camera[0], scn.camera[1]
    pos = (np.asarray(a.pos, float) + np.asarray(b.pos, float)) / 2.0
    width = float(a.frustum_width)
    if width <= 0.0:
        width = (float(a.frustum_top) - float(a.frustum_bottom)) / 2.0 * float(aspect)
    return (pos, np.asarray(a.forward, float).copy(), np.asarray(a.up, float).copy(),
            [float(a.frustum_center), width, float(a.frustum_bottom), float(a.frustum_top), float(a.frustum_near)])





class Scene(object):
    """Everything the renderer needs of one episode, read in the caller's thread (plain data)."""

    def __init__(self, unit, episode, model, meta, bay, look):
        import shelf_scene as SC
        import shelf_sim as S
        import shelf_tex as TX
        import shelf_world as W
        from PIL import Image
        t0 = time.perf_counter()
        with open(os.path.join(RENDER_DIR, "textures.json")) as fh:
            tj = json.load(fh)
        dress, lays = tj["dress"], dict(tj["layouts"])
        C = SC.consts(S, W)
        self.textures = {}
        names = sorted(list(unit["skus"]) + list(dress))            # the look's atlas seeds: the index in this order
        for name, sku in unit["skus"].items():
            seed = 5000 + names.index(name)
            if sku["kind"] == "box":
                im, _rg, lay = TX.box_atlas(sku, seed)
            else:
                im, _rg, lay = TX.wrap_atlas(sku, seed, bottle=sku["size"][1] > 0.08)
            lays[name] = lay
            im = im.convert("RGB").resize((max(1, int(round(im.size[0] * SKU_SCALE))), max(1, int(round(im.size[1] * SKU_SCALE)))), Image.LANCZOS)
            self.textures[("sku", name)] = np.ascontiguousarray(np.asarray(im)[::-1])
        for code in sorted({int(v["code"]) for v in meta.values()}):
            im = TX.code_label(code, W.CODE_PAPER, W.CODE_INK, C["CODE"]).convert("RGB")
            im = im.resize((max(1, int(round(im.size[0] * CODE_SCALE))), max(1, int(round(im.size[1] * CODE_SCALE)))), Image.LANCZOS)
            self.textures[("code", code)] = np.ascontiguousarray(np.asarray(im)[::-1])
        stocked = SC.dress_for(unit, dress)
        col = SC.Collector(SC.looks_of(unit, dress))
        SC.cell(col, unit, model, meta, bay, C, lays)
        SC.store(col, bay, C, stocked, lays)
        SC.booths(col, unit, lays)
        self.items = col.items()
        self.dress_left_out = sorted(set(dress) - set(stocked))
        self.booth = [(name, SC.booth_origin(i), SC.booth_camera(i, sku)) for i, (name, sku) in enumerate(unit["skus"].items())]
        R, p = S.bay_frame(bay)
        self.bay_pose = (np.asarray(p, np.float32).copy(), np.ascontiguousarray(R, np.float32).ravel().copy())
        self.carriage = model.body("carriage").id
        self.lift_tops = SC.lift_tops
        self.gain = float(look["exposure_scale"])
        self.ibl_intensity = float(look["ibl"]["intensity"]) * self.gain
        self.supersample = max(1, int(look["render"].get("supersample", 1)))
        self.supersample_of = {str(k): max(1, int(v)) for k, v in look["render"].get("supersample_by_camera", {}).items()}
        self.cams = {model.camera(i).name: i for i in range(model.ncam)}
        self.triangles = int(sum(len(it["mesh"][3]) // 3 for it in self.items))
        self.seconds = time.perf_counter() - t0

    def poses(self, d):
        xp, xm = np.asarray(d.xpos, np.float32), np.asarray(d.xmat, np.float32)
        return {"xpos": xp.copy(), "xmat": xm.copy(), "lift": self.lift_tops(float(d.xpos[self.carriage][2]))}


class View(object):
    """One episode's cameras; the Filament objects stay in the renderer's thread."""

    def __init__(self, unit, episode, model, meta, bay):
        self._fil = _Filament.instance()
        self.model = model
        try:
            self.scene = Scene(unit, episode, model, meta, bay, self._fil.look)
            self._scn = mujoco.MjvScene(model, 1)
            self._cam = mujoco.MjvCamera()
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError("renderer: reading the episode's parts: %s: %s" % (type(exc).__name__, exc))
        self.build_s = self.scene.seconds
        self._fil.call(self._fil.load, self.scene)

    def render(self, cam, data, width, height):
        """RGB uint8 (height, width, 3), rows top-down, of the camera named `cam` in the state `data`; no noise."""
        try:
            sc = self.scene
            W, H, k = int(width), int(height), sc.supersample_of.get(cam, sc.supersample)
            pos, fwd, up, fr = gl_camera(self.model, data, sc.cams[cam], self._scn, self._cam, W / float(H))
            args = (sc.poses(data), pos, fwd, up, fr, W * k, H * k)
        except Exception as exc:  # noqa: BLE001
            raise RenderError("renderer: camera %r: %s: %s" % (cam, type(exc).__name__, exc))
        img = self._fil.call(self._fil.draw, *args)
        return img if k == 1 else _box_down(img, k)

    def catalog(self, data, px):
        """One front-view picture per SKU (px x px): the unit alone on a grey ground, lit as the bay is; no noise."""
        out = {}
        try:
            poses = self.scene.poses(data)
        except Exception as exc:  # noqa: BLE001
            raise RenderError("renderer: catalog: %s: %s" % (type(exc).__name__, exc))
        for name, _origin, (pos, fwd, up, fr) in self.scene.booth:
            out[name] = _box_down(self._fil.call(self._fil.draw, poses, pos, fwd, up, fr, 2 * int(px), 2 * int(px)), 2)
        return out

    def close(self):
        pass


def open_view(unit, episode, model, meta, bay):
    """The cameras of one episode. Raises RenderError."""
    return View(unit, episode, model, meta, bay)


class _Filament(object):
    _inst = None
    _failed = None
    _guard = threading.Lock()

    @classmethod
    def instance(cls):
        with cls._guard:
            pid = os.getpid()
            if cls._failed is not None and cls._failed[0] == pid:
                raise RenderError(cls._failed[1])
            if cls._inst is None or cls._inst.pid != pid:
                cls._inst = None
                try:
                    cls._inst = cls()
                except RenderError as exc:
                    cls._failed = (pid, str(exc))
                    raise
            return cls._inst

    def __init__(self):
        self.pid = os.getpid()
        self.broken = None
        try:
            self.look = load_look()
        except (OSError, ValueError) as exc:
            raise RenderError("renderer: cannot read %s: %s" % (LOOK_FILE, exc))
        missing = [f for f in BUNDLE_FILES if not os.path.isfile(os.path.join(RENDER_DIR, f))]
        if missing:
            raise RenderError("renderer: the render bundle in %s lacks %s (SHELF_RENDER_DIR)" % (RENDER_DIR, ", ".join(missing)))
        self.jobs = queue.Queue()
        self.targets = {}
        self.view = None
        self.scene = None
        self.stats = {"frames": 0, "targets_created": 0}
        self.thread = threading.Thread(target=self._loop, name="shelf-filament", daemon=True)
        self.thread.start()
        try:
            self.call(self._start)
        except RenderError:
            self.post(None)
            raise

    def call(self, fn, *args):
        if self.broken:
            raise RenderError(self.broken)
        done = threading.Event()
        out = []
        self.jobs.put((fn, args, done, out))
        if not done.wait(CALL_TIMEOUT):
            self.broken = "renderer: a call did not return within %.0f s (SHELF_RENDER_TIMEOUT)" % CALL_TIMEOUT
            raise RenderError(self.broken)
        ok, val = out[0]
        if not ok:
            raise val
        return val

    def post(self, fn, *args):
        if self.pid == os.getpid() and self.thread.is_alive():
            self.jobs.put((fn, args, None, None))

    def stop(self, timeout=30.0):
        if self.pid == os.getpid() and self.thread.is_alive() and not self.broken:
            self.jobs.put((None, (), None, None))
            self.thread.join(timeout)

    def _loop(self):
        while True:
            fn, args, done, out = self.jobs.get()
            if fn is None:
                try:
                    self._shutdown()
                except BaseException:  # noqa: BLE001
                    pass
                return
            try:
                res = (True, fn(*args))
            except BaseException as exc:  # noqa: BLE001
                res = (False, RenderError("renderer: %s: %s" % (type(exc).__name__, exc)))
            if done is not None:
                out.append(res)
                done.set()

    def _start(self):
        with _StdoutToStderr():
            t0 = time.perf_counter()
            for k in ("DISPLAY", "WAYLAND_DISPLAY"):
                os.environ.pop(k, None)
            os.environ["MESA_SHADER_CACHE_DISABLE"] = "true"
            if self.look["render"].get("lavapipe_threads"):
                os.environ["LP_NUM_THREADS"] = str(int(self.look["render"]["lavapipe_threads"]))
            import mujoco._render_filament as F
            from mujoco.experimental.studio import window  # noqa: F401  registers Filament's resource providers
            self.F = F
            look = self.look
            self.ctx = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN))
            self.ctx.set_clear_color([float(x) for x in look["render"].get("clear_color", [0.0, 0.0, 0.0])])
            self.ibl_tex = self._ktx(look["ibl"]["file"], F.SamplerType.TEXTURE_CUBE)
            self.scn = self.ctx.create_scene(F.SceneParams())
            self._configure()
            lp = F.LightParams()
            lp.type, lp.texture, lp.color, lp.intensity = F.LightType.LIGHT_IMAGE, self.ibl_tex, [1.0, 1.0, 1.0], 1.0
            self.ibl = self.ctx.create_light(lp)
            self.scn.add_light(self.ibl)
            self.start_s = time.perf_counter() - t0

    def _configure(self):
        look = self.look
        spec = mujoco.MjSpec()
        for k, v in sorted(look["filament_numeric"].items()):
            spec.add_numeric(name=k, data=[float(x) for x in (v if isinstance(v, (list, tuple)) else [v])])
        for k, v in sorted(look["filament_text"].items()):
            spec.add_text(name=k, data=str(v))
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], pos=[0, 0, -50])
        self.look_model = spec.compile()
        for _ in range(2):
            self.scn.configure_from_model(self.look_model)

    def _ktx(self, name, sampler):
        F = self.F
        path = os.path.join(RENDER_DIR, name)
        with open(path, "rb") as fh:
            raw = fh.read()
        tc = F.TextureConfig()
        tc.width, tc.height, tc.format = len(raw), 0, F.PixelFormat.PIXEL_FORMAT_KTX
        tc.sampler_type, tc.color_space = sampler, F.ColorSpace.COLORSPACE_LINEAR
        t = self.ctx.create_texture(tc)
        td = F.TextureData()
        td.set_image(raw)
        t.upload(td)
        return t

    def _mesh_config(self, n_vertices, n_indices):
        F = self.F
        A, Us, Ty = F.VertexAttribute, F.VertexAttributeUsage, F.VertexAttributeType
        attrs = [A(usage=Us.VERTEX_ATTRIBUTE_USAGE_POSITION, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                 A(usage=Us.VERTEX_ATTRIBUTE_USAGE_NORMAL, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                 A(usage=Us.VERTEX_ATTRIBUTE_USAGE_UV, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT2)]
        return F.MeshConfig(max_vertices=n_vertices, max_indices=n_indices, attributes=attrs, interleaved=False,
                            index_type=F.IndexType.INDEX_TYPE_U32,
                            primitive_type=F.MeshPrimitiveType.MESH_PRIMITIVE_TYPE_TRIANGLES)

    def _mesh_upload(self, mesh, cfg, V, N, U, I):
        md = self.F.MeshData(compute_bounds=True)
        md.set_vertices(cfg, np.concatenate([np.asarray(V, np.float32).ravel(), np.asarray(N, np.float32).ravel(),
                                             np.asarray(U, np.float32).ravel()]).tobytes())
        md.set_indices(cfg, np.ascontiguousarray(I, np.uint32).tobytes())
        mesh.upload(md)

    def _mesh(self, V, N, U, I):
        cfg = self._mesh_config(len(V), len(I))
        mesh = self.ctx.create_mesh(cfg)
        self._mesh_upload(mesh, cfg, V, N, U, I)
        return mesh


    def _texture(self, img):
        F = self.F
        h, w = img.shape[:2]
        tc = F.TextureConfig()
        tc.width, tc.height, tc.format = w, h, F.PixelFormat.PIXEL_FORMAT_RGB8
        tc.sampler_type, tc.color_space = F.SamplerType.TEXTURE_2D, F.ColorSpace.COLORSPACE_SRGB
        t = self.ctx.create_texture(tc)
        td = F.TextureData()
        td.set_image(np.ascontiguousarray(img, np.uint8).tobytes())
        t.upload(td)
        return t


    def _props(self, group):
        """a prop group of the bundle: (mesh arrays, material terms, texture image or None) per material"""
        d = os.path.join(RENDER_DIR, "props_" + group)
        with open(os.path.join(d, "room.json")) as fh:
            meta = json.load(fh)
        z = np.load(os.path.join(d, "room.npz"))
        V, N, U, I, tex = z["V"], z["N"], z["UV"], z["I"], z["tex"]
        tex_ofs, tw, th = z["tex_ofs"], z["tex_w"], z["tex_h"]
        out = []
        for mm in meta["materials"]:
            if mm["icnt"] == 0:
                continue
            img = None
            if mm["tex"] >= 0:
                k = mm["tex"]
                o, w, h = int(tex_ofs[k]), int(tw[k]), int(th[k])
                img = tex[o:o + w * h * 3].reshape(h, w, 3)
            v0, vc, i0, ic = mm["vofs"], mm["vcnt"], mm["iofs"], mm["icnt"]
            out.append(((V[v0:v0 + vc], N[v0:v0 + vc], U[v0:v0 + vc], I[i0:i0 + ic]),
                        (np.asarray(mm["rgba"][:3], float), float(mm["roughness"]), float(mm["metallic"]), 0.0), img))
        return out

    def load(self, sc):
        """The episode's renderables, made once and in one order: the scene's items, then the bundle's props."""
        if self.scene is not None:
            raise RenderError("renderer: this process has drawn another episode (one episode per process)")
        F = self.F
        rl = self.look["materials"]
        stat = np.load(os.path.join(RENDER_DIR, "textures.npz"))
        texs = {}
        self.rend = []

        def texture(key, img):
            if key not in texs:
                texs[key] = self._texture(img)
            return texs[key]

        def add(mesh, mat, tex, frame, look):
            V, N, U, I = mesh
            rgb, rough, metal, emis = mat
            gain = np.broadcast_to(np.asarray(rl.get("gain_by_look", {}).get(look, rl.get("gain", 1.0)), float), (3,))
            rgba = np.ones(4, np.float32)
            rgba[:3] = srgb_from_linear(linear_from_srgb(np.asarray(rgb, float)[:3]) * gain)
            m = self._mesh(V, N, U, I)
            material = F.Material(color=rgba, roughness=float(np.clip(rough * rl.get("roughness_scale", 1.0), 0.05, 1.0)),
                                  metallic=float(metal) * float(rl.get("metallic_scale", 1.0)), specular=float(rl.get("specular", 0.3)),
                                  emissive=float(emis), color_texture=tex)
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadow=False, receive_shadow=True))
            r.set_mesh(m, 0, len(I))
            r.set_material(material)
            self.scn.add_renderable(r)
            self.rend.append((r, frame, m, material))
            if frame == "bay":
                r.set_pose(*sc.bay_pose)
            elif isinstance(frame, tuple) and frame[0] == "booth":
                r.set_pose(np.asarray(sc.booth[frame[1]][1], np.float32), np.eye(3, dtype=np.float32).ravel())

        for it in sc.items:
            tex = None
            if it["tex"] is not None:
                key = it["tex"]
                img = stat[key[1]] if key[0] == "static" else sc.textures[key]
                tex = texture(key, img)
            add(it["mesh"], it["mat"], tex, it["frame"], it["look"])
        for group in ("bay", "world"):
            for k, (mesh, mat, img) in enumerate(self._props(group)):
                add(mesh, mat, None if img is None else texture(("props", group, k), img), group, "prop")
        self.textures = texs
        self.moving = [(r, f) for r, f, _m, _mt in self.rend if isinstance(f, tuple) and f[0] in ("body", "lift")]
        self.ibl.set_intensity(float(sc.ibl_intensity))
        self.scene = sc
        gc.collect()
        _trim_heap()

    def _target(self, width, height):
        F = self.F
        key = (width, height)
        if key not in self.targets:
            buf = np.zeros((height, width, 3), np.uint8)
            tgt = self.ctx.create_render_target(F.RenderTargetConfig(color_format=F.PixelFormat.PIXEL_FORMAT_RGB8))
            tgt.resize(width, height)
            read = F.ReadRequest()
            read.target = tgt
            read.set_buffer(buf)
            self.targets[key] = (tgt, read, buf)
            self.stats["targets_created"] += 1
        return self.targets[key]

    def draw(self, poses, pos, forward, up, frustum, width, height):
        F = self.F
        xp, xm, lift = poses["xpos"], poses["xmat"], poses["lift"]
        eye = np.eye(3, dtype=np.float32).ravel()
        for r, (kind, k) in self.moving:
            if kind == "body":
                r.set_pose(xp[k], xm[k])
            else:
                r.set_pose(np.array([0.0, 0.0, lift[k]], np.float32), eye)
        tgt, read, buf = self._target(width, height)
        cam = F.Camera(pos=np.asarray(pos, np.float32), forward=np.asarray(forward, np.float32),
                       up=np.asarray(up, np.float32), frustum_center=frustum[0], frustum_width=frustum[1],
                       frustum_bottom=frustum[2], frustum_top=frustum[3], frustum_near=frustum[4], frustum_far=CAM_FAR)
        rv = self.look["render"]
        req = F.RenderRequest(camera=cam, draw_mode=F.DrawMode.DRAW_MODE_DEFAULT, post_processing=bool(rv["post_processing"]),
                              shadows=bool(rv["shadows"]), reflections=bool(rv.get("reflections", False)))
        req.scene, req.target = self.scn, tgt
        req.viewport.left, req.viewport.bottom, req.viewport.width, req.viewport.height = 0, 0, width, height
        self.ctx.wait_for_frame(self.ctx.render([req], [read]))
        self.stats["frames"] += 1
        if self.view is None:
            self._undither(width, height, bool(rv["shadows"]), bool(rv["post_processing"]))
            self.ctx.wait_for_frame(self.ctx.render([req], [read]))
        return buf.copy()

    # Filament's view adds a new noise to every frame before the 8-bit output (temporal dithering, +-1 level; the
    # FXAA pass after it turns that into differences of up to ~100 levels along edges), so without this an image
    # depended on how many frames the process had drawn before. MuJoCo 3.16 has no option for it (no filament.*
    # field, nothing in the binding) and draws every camera through one filament::View; Filament's exported
    # View::setDithering(Dithering::NONE) is called on that view. The view is found from the binding's Scene object
    # (pointers followed, each read by write(2) into a pipe: a wrong guess cannot crash, and a process that is not
    # dumpable may do it) by its state after the first image: viewport (0, 0, width, height) at +0x268,
    # anti-aliasing FXAA, dithering TEMPORAL, shadowing and post-processing as requested at +0x2b1..+0x2b5 (Filament
    # as linked into MuJoCo 3.16). FXAA stays on: on an image without the noise it is a function of the image.
    VIEW_SET_DITHERING = "_ZN8filament4View12setDitheringENS_9DitheringE"
    VIEW_GET_DITHERING = "_ZNK8filament4View12getDitheringEv"

    def _undither(self, width, height, shadows, post):
        lib = ctypes.CDLL(self.F.__file__)
        try:
            setd, getd = getattr(lib, self.VIEW_SET_DITHERING), getattr(lib, self.VIEW_GET_DITHERING)
        except AttributeError as exc:
            raise RenderError("renderer: MuJoCo's Filament module exports no View::setDithering: %s" % exc)
        setd.restype, setd.argtypes = None, [ctypes.c_void_p, ctypes.c_uint8]
        getd.restype, getd.argtypes = ctypes.c_uint8, [ctypes.c_void_p]
        want = (0, 0, width, height, 1, 1, int(shadows), 1, int(post))
        found, seen = set(), set()
        libc = ctypes.CDLL(None, use_errno=True)
        libc.write.restype, libc.write.argtypes = ctypes.c_ssize_t, [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        rfd, wfd = os.pipe()

        def read(addr, n):                             # write(2) from an unreadable address fails (EFAULT)
            k = libc.write(wfd, addr, n)
            return os.read(rfd, k) if k > 0 else None

        # Every pointer within three steps of the Scene object, breadth first: the pointers in the object's first 64
        # bytes, then the ones in the 512 bytes each of those points at, then once more. (A depth-first walk with one
        # set of visited pointers for all depths can miss the view: a block that it meets first at its last level,
        # through some other pointer, is then never read at the level where its own pointers still count. Seen once
        # in about a hundred process starts here, as a RenderError at the process's first image, and in
        # keyed_nut_peg_assembly, where the breadth-first search then found the view in 200 of 200 starts.)
        try:
            level = [(id(self.scn), 64)]
            for _depth in range(3):
                nxt = []
                for addr, n in level:
                    b = read(addr, n)
                    for (p,) in struct.iter_unpack("<Q", b[:len(b) // 8 * 8] if b else b""):
                        if p in seen or p < 0x10000 or p > 0x7fffffffffff or p % 8:
                            continue
                        seen.add(p)
                        v = None if p % 16 else read(p + 0x268, 0x4e)
                        if v and len(v) == 0x4e and struct.unpack_from("<iiII5B", v[:16] + v[0x49:0x4e]) == want:
                            found.add(p)
                        nxt.append((p, 512))
                level = nxt
        finally:
            os.close(rfd)
            os.close(wfd)
        if len(found) != 1:
            raise RenderError("renderer: cannot find the Filament view to switch its dithering off (%d candidates)"
                              % len(found))
        self.view = found.pop()
        setd(self.view, 0)                             # filament::View::Dithering::NONE
        if getd(self.view) != 0:
            raise RenderError("renderer: Filament's dithering did not switch off")



    def info(self):
        return {"stats": dict(self.stats), "start_s": self.start_s, "renderables": len(self.rend) if self.scene else 0,
                "textures": len(self.textures) if self.scene else 0, "triangles": self.scene.triangles if self.scene else 0}

    def _shutdown(self):
        self.targets.clear()
        self.rend, self.moving, self.textures = [], [], {}
        self.ibl = None
        self.scn = None
        self.look_model = None
        self.ibl_tex = None
        gc.collect()
        self.ctx = None
        gc.collect()


def renderer_info():
    f = _Filament.instance()
    return f.call(f.info)


@atexit.register
def _stop_renderer():
    if _Filament._inst is not None:
        _Filament._inst.stop()


# ----------------------------------------------------------------------------------------------------------------
# One process per episode. shelf_env draws through a Worker: a child process started for the episode, which builds
# the episode's model from (unit, episode) exactly as the simulator does, opens its view and draws the states it is
# sent (the joint positions). Nothing of an earlier episode exists in it, so a picture cannot depend on what the
# caller's process has drawn before.
# ----------------------------------------------------------------------------------------------------------------
def _read_exact(fd, n, deadline):
    import select
    buf = bytearray()
    while len(buf) < n:
        left = deadline - time.monotonic()
        if left <= 0 or not select.select([fd], [], [], left)[0]:
            raise RenderError("renderer: the episode's render process did not answer within %.0f s" % CALL_TIMEOUT)
        chunk = os.read(fd, min(1 << 20, n - len(buf)))
        if not chunk:
            raise RenderError("renderer: the episode's render process ended")
        buf += chunk
    return bytes(buf)


def _read_msg(fd, deadline):
    (n,) = struct.unpack("<I", _read_exact(fd, 4, deadline))
    return json.loads(_read_exact(fd, n, deadline).decode("utf-8"))


def _write_all(fd, data):
    view = memoryview(data)
    while len(view):
        view = view[os.write(fd, view):]


def _write_msg(fd, obj, payload=b""):
    head = json.dumps(obj).encode("utf-8")
    _write_all(fd, struct.pack("<I", len(head)) + head + payload)


class Worker(object):
    """The cameras of one episode, drawn in a process of its own."""

    def __init__(self, unit, episode):
        import subprocess
        try:
            self.proc = subprocess.Popen([sys.executable, "-I", "-B", os.path.abspath(__file__), "--worker"],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
            self.w, self.r = self.proc.stdin.fileno(), self.proc.stdout.fileno()
            _write_msg(self.w, {"unit": unit, "episode": episode})
            rep = _read_msg(self.r, time.monotonic() + CALL_TIMEOUT)
        except RenderError:
            self.close()
            raise
        except Exception as exc:  # noqa: BLE001
            self.close()
            raise RenderError("renderer: cannot start the episode's render process: %s: %s" % (type(exc).__name__, exc))
        if not rep.get("ok"):
            self.close()
            raise RenderError(rep.get("error", "renderer: the episode's render process failed to start"))
        self.nq = int(rep["nq"])
        self.info = rep

    def _ask(self, obj, payload=b""):
        try:
            _write_msg(self.w, obj, payload)
            rep = _read_msg(self.r, time.monotonic() + CALL_TIMEOUT)
            if not rep.get("ok"):
                raise RenderError(rep.get("error", "renderer: failed"))
            return rep, _read_exact(self.r, int(rep["bytes"]), time.monotonic() + CALL_TIMEOUT)
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError("renderer: %s: %s" % (type(exc).__name__, exc))

    def render(self, cam, qpos, width, height):
        q = np.ascontiguousarray(qpos, np.float64)
        if q.shape != (self.nq,):
            raise RenderError("renderer: a state of %s joint positions, the episode has %d" % (q.shape, self.nq))
        _rep, raw = self._ask({"op": "render", "cam": cam, "w": int(width), "h": int(height)}, q.tobytes())
        return np.frombuffer(raw, np.uint8).reshape(int(height), int(width), 3).copy()

    def catalog(self, qpos, px):
        q = np.ascontiguousarray(qpos, np.float64)
        rep, raw = self._ask({"op": "catalog", "px": int(px)}, q.tobytes())
        a = np.frombuffer(raw, np.uint8).reshape(len(rep["names"]), int(px), int(px), 3)
        return {n: a[i].copy() for i, n in enumerate(rep["names"])}

    def close(self):
        p = getattr(self, "proc", None)
        self.proc = None
        if p is None:
            return
        for f in (p.stdin, p.stdout):
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            p.wait(timeout=20)
        except Exception:  # noqa: BLE001
            p.kill()
            p.wait()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def _worker_main():
    out = os.dup(1)                    # the wire; whatever the renderer prints goes to standard error
    os.dup2(2, 1)
    inp = 0
    far = time.monotonic() + 10 ** 9
    try:
        sys.path.insert(0, HERE)
        head = _read_msg(inp, far)
        import shelf_world as W
        t0 = time.perf_counter()
        w = W.World(head["unit"], head["episode"])
        m = w.model
        d = mujoco.MjData(m)
        view = open_view(head["unit"], head["episode"], m, w.meta, w.bay)
        _write_msg(out, {"ok": True, "nq": int(m.nq), "build_s": round(time.perf_counter() - t0, 2),
                         "triangles": view.scene.triangles, "dress_left_out": view.scene.dress_left_out})
    except BaseException as exc:  # noqa: BLE001
        _write_msg(out, {"ok": False, "error": "renderer: %s: %s" % (type(exc).__name__, exc)})
        return 1
    nq = int(m.nq)
    while True:
        try:
            req = _read_msg(inp, far)
        except RenderError:
            return 0                   # the caller closed the wire: the episode is over
        try:
            d.qpos[:] = np.frombuffer(_read_exact(inp, nq * 8, far), np.float64)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            if req["op"] == "render":
                img = view.render(req["cam"], d, req["w"], req["h"])
                _write_msg(out, {"ok": True, "bytes": int(img.size)}, np.ascontiguousarray(img).tobytes())
            elif req["op"] == "catalog":
                pics = view.catalog(d, req["px"])
                names = list(pics)
                raw = b"".join(np.ascontiguousarray(pics[n]).tobytes() for n in names)
                _write_msg(out, {"ok": True, "names": names, "bytes": len(raw)}, raw)
            else:
                _write_msg(out, {"ok": False, "error": "renderer: unknown request"})
        except BaseException as exc:  # noqa: BLE001
            _write_msg(out, {"ok": False, "error": "renderer: %s: %s" % (type(exc).__name__, exc)})


if __name__ == "__main__" and "--worker" in sys.argv[1:]:
    code = _worker_main()
    _stop_renderer()
    os._exit(int(code or 0))
