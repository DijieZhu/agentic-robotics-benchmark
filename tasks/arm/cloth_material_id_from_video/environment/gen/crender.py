"""crender: the three cameras of cloth_material_id_from_video, drawn with MuJoCo 3.16's Filament renderer
(physically based) on Vulkan; in the build image Vulkan is Mesa's lavapipe on the CPU (no GPU).

What is drawn: cscene.py's parts (the bench, the robot's pedestal, the Panda's visual meshes, the fabric clamp, the
three cameras on their stands, a stack of swatches), the fabric as csheet.shell lays it (one closed shell: the
printed face with the rim, the under side), and the render bundle's room (CLOTH_RENDER_DIR, default
/opt/task-assets/render: room.npz, room.json, textures.npz, parts.npz, ibl.ktx): a loft work space round the bench.
Light: an image light (ibl.ktx: a path-traced probe of that room over the bench with its lamps painted in) and
ambient occlusion; no lamp casts a shadow. Exposure fixed (clook.json). An image is drawn at a multiple of its size
and averaged in linear light (render.supersample).

    import crender
    clip = crender.open_clip(model, cameras, look=3)          # model: a MuJoCo model that holds carm's arm; cameras: cameras.json's list
    img = clip.draw("cam0", sh, data, clamp_row)              # sh: csheet.shell(...) of the frame; data: the arm posed (mj_kinematics)

An image is a function of the look, the arm's pose, the clamp row and the sheet alone: every renderable, mesh,
texture and material is made once, in one order, when the clip is opened, and a process draws ONE clip (a second
open_clip in the same process raises RenderError: give every clip a fresh process); Filament's temporal dithering
and FXAA are switched off (_undither). The camera's blur, vignetting, white balance, noise and H.264 stay
render_clips.py's. Maintainers' side: never in the image's layers or the agent's home.
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
RENDER_DIR = os.environ.get("CLOTH_RENDER_DIR", "/opt/task-assets/render")
LOOK_FILE = os.environ.get("CLOTH_LOOK_FILE", os.path.join(HERE, "clook.json"))
BUNDLE_FILES = ("textures.npz", "parts.npz", "ibl.ktx", "room.npz", "room.json")
CALL_TIMEOUT = float(os.environ.get("CLOTH_RENDER_TIMEOUT", "300"))
CAM_FAR = 40.0
VIEW_NOT_FOUND = "cannot find the Filament view"
VIEW_RETRIES = 3                 # fresh render processes for an episode whose first one did not find its view


class RenderError(RuntimeError):
    """The renderer failed."""


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



def camera_terms(cam, near=0.05):
    """A camera of cameras.json (OpenCV pinhole: pixel ~ K (R_w2c X + t_w2c), the principal point at the centre) as
    the renderer's camera: (position, forward, up, [frustum centre, half width, bottom, top, near])."""
    Rw, t, K = np.asarray(cam["R_w2c"], float), np.asarray(cam["t_w2c"], float), np.asarray(cam["K"], float)
    w, h = float(cam["width"]), float(cam["height"])
    if abs(K[0, 2] - w / 2) > 1e-6 or abs(K[1, 2] - h / 2) > 1e-6:
        raise RenderError("renderer: camera %r: the principal point is not at the image's centre" % cam.get("name"))
    return (-Rw.T @ t, Rw[2].copy(), -Rw[1].copy(), [0.0, near * (w / 2) / K[0, 0], -near * (h / 2) / K[1, 1], near * (h / 2) / K[1, 1], near])


class Collector(object):
    """Parts merged per frame and look, in the order they first appear (one mesh per frame and look: separate parts of
    one look that touch are drawn in an order that can change from process to process)."""

    def __init__(self):
        self.groups = {}

    def add(self, frame, look, mesh):
        import cscene as SC
        V, N, UV, F = [np.asarray(x) for x in mesh]
        if len(F) == 0:
            return
        L = SC.LOOKS.get(look) or SC.LOOKS["default"]
        UV = UV.astype(np.float64) / float(L["tile"]) if "tile" in L else UV.astype(np.float64)
        g = self.groups.setdefault((frame, look), {"frame": frame, "look": look, "V": [], "N": [], "U": [], "F": [], "nv": 0})
        g["V"].append(V.astype(np.float64))
        g["N"].append(N.astype(np.float64))
        g["U"].append(UV)
        g["F"].append(F.reshape(-1, 3).astype(np.int64) + g["nv"])
        g["nv"] += len(V)

    def items(self):
        import cscene as SC
        out = []
        for g in self.groups.values():
            L = SC.LOOKS.get(g["look"]) or SC.LOOKS["default"]
            N = np.concatenate(g["N"])
            N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
            tex = L.get("col")
            rgb = np.ones(3) if tex else srgb_from_linear(L.get("base", [0.5, 0.5, 0.5]))
            out.append({"frame": g["frame"], "look": g["look"], "tex": tex, "mat": (rgb, float(L.get("rough", 0.5)), float(L.get("metal", 0.0)), 0.0),
                        "mesh": (np.ascontiguousarray(np.concatenate(g["V"]), np.float32), np.ascontiguousarray(N, np.float32),
                                 np.ascontiguousarray(np.concatenate(g["U"]), np.float32),
                                 np.ascontiguousarray(np.concatenate(g["F"]).reshape(-1), np.uint32))})
        return out


class Scene(object):
    """Everything the renderer needs of one clip, read in the caller's thread (plain data)."""

    def __init__(self, model, cameras, look, look_file):
        import carm
        import cphys as W
        import csheet as CS
        import cscene as SC
        t0 = time.perf_counter()
        parts = np.load(os.path.join(RENDER_DIR, "parts.npz"))
        cam_list = [(c["name"], np.asarray(c["position_world"], float), np.asarray(c["position_world"], float) + np.asarray(c["R_w2c"], float)[2])
                    for c in cameras]
        col = Collector()
        world, self.boxes, self.stands = SC.world_parts(W.EDGE_X, W.TABLE_HALF, carm.ROBOT_BASE, cam_list, parts["cam_soup"])
        for _name, mesh, lk in world:
            col.add("world", lk, mesh)
        for b, _name, mesh, lk in SC.arm_parts(model):
            col.add(("body", int(b)), lk, mesh)
        lower, upper, self.clamp_boxes = SC.clamp_parts(W.JAW_W, carm.R_CLAMP_EEF, carm.TOOL_TIP, CS)
        for _name, mesh, lk in lower:
            col.add("clamp", lk, mesh)
        for _name, mesh, lk in upper:
            col.add("pad", lk, mesh)
        self.items = col.items()
        self.look = int(look)
        self.half_w = W.JAW_W / 2 + 0.002
        self.gain = float(look_file["exposure_scale"])
        self.ibl_intensity = float(look_file["ibl"]["intensity"]) * self.gain
        self.supersample = max(1, int(look_file["render"].get("supersample", 1)))
        self.cams = {c["name"]: camera_terms(c) for c in cameras}
        self.size = {c["name"]: (int(c["width"]), int(c["height"])) for c in cameras}
        self.triangles = int(sum(len(it["mesh"][3]) // 3 for it in self.items))
        self.seconds = time.perf_counter() - t0


class Clip(object):
    """One clip's cameras; the Filament objects stay in the renderer's thread."""

    def __init__(self, model, cameras, look):
        self._fil = _Filament.instance()
        try:
            self.scene = Scene(model, cameras, look, self._fil.look)
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError("renderer: reading the clip's parts: %s: %s" % (type(exc).__name__, exc))
        self._fil.call(self._fil.load, self.scene)

    def draw(self, cam, sh, data, clamp_row, thickness=0.0015):
        """RGB uint8 (height, width, 3), rows top-down, of camera `cam`: the fabric as csheet.shell laid it (sh), the
        arm as posed in `data` (xpos, xmat), the clamp at the commanded row; no blur, vignetting or noise."""
        import cscene as SC
        import csheet as CS
        try:
            sc = self.scene
            W, H = sc.size[cam]
            k = sc.supersample
            pos, fwd, up, fr = sc.cams[cam]
            row = np.asarray(clamp_row, float)
            Rq = SC.quat_mat(row[3:7])
            closed = bool(row[7] > 0.5)
            pz = sh["pad_z"] if (closed and sh["pad_z"] is not None) else SC.pad_rest(CS, thickness) + (0.0 if closed else SC.JAW_OPEN)
            poses = {"xpos": np.asarray(data.xpos, np.float32).copy(), "xmat": np.asarray(data.xmat, np.float32).copy(),
                     "clamp": (row[:3].astype(np.float32), Rq.astype(np.float32).ravel()),
                     "pad": ((row[:3] + Rq @ np.array([0.0, 0.0, pz])).astype(np.float32), Rq.astype(np.float32).ravel())}
            sheet = (np.asarray(sh["top"], np.float32).reshape(-1, 3), np.asarray(sh["bot"], np.float32).reshape(-1, 3),
                     np.asarray(sh["n"], np.float32).reshape(-1, 3))
        except Exception as exc:  # noqa: BLE001
            raise RenderError("renderer: camera %r: %s: %s" % (cam, type(exc).__name__, exc))
        img = self._fil.call(self._fil.draw, poses, sheet, pos, fwd, up, fr, W * k, H * k)
        return img if k == 1 else _box_down(img, k)

    def info(self):
        return self._fil.call(self._fil.info)

    def close(self):
        pass


def open_clip(model, cameras, look):
    """The cameras of one clip: model = a MuJoCo model holding carm's arm (bodies named r_...), cameras = the list of
    cameras.json, look = the index of the fabric's look in the bundle's library. Raises RenderError."""
    return Clip(model, cameras, look)


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
            raise RenderError("renderer: the render bundle in %s lacks %s (CLOTH_RENDER_DIR)" % (RENDER_DIR, ", ".join(missing)))
        self.jobs = queue.Queue()
        self.targets = {}
        self.view = None
        self.scene = None
        self.stats = {"frames": 0, "targets_created": 0}
        self.thread = threading.Thread(target=self._loop, name="cloth-filament", daemon=True)
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
            self.broken = "renderer: a call did not return within %.0f s (CLOTH_RENDER_TIMEOUT)" % CALL_TIMEOUT
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


    def _room(self):
        """the bundle's room: (mesh arrays, material terms, texture image or None, class) per material"""
        with open(os.path.join(RENDER_DIR, "room.json")) as fh:
            meta = json.load(fh)
        z = np.load(os.path.join(RENDER_DIR, "room.npz"))
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
                        (np.asarray(mm["rgba"][:3], float), float(mm["roughness"]), float(mm["metallic"]), float(mm.get("emissive", 0.0))),
                        img, "room_" + str(mm.get("cls", ""))))
        return out

    def load(self, sc):
        """The clip's renderables, made once and in one order: the scene's items, the fabric's two meshes, the room."""
        if self.scene is not None:
            raise RenderError("renderer: this process has drawn another clip (one clip per process)")
        import csheet as CS
        import creal as RL
        F = self.F
        rl = self.look["materials"]
        stat = np.load(os.path.join(RENDER_DIR, "textures.npz"))
        texs = {}
        self.rend = []

        def texture(key, img):
            if key not in texs:
                texs[key] = self._texture(img)
            return texs[key]

        mats = {}

        def add(mesh, mat, tex, frame, look, dynamic=False):
            V, N, U, I = mesh
            rgb, rough, metal, emis = mat
            gain = np.broadcast_to(np.asarray(rl.get("gain_by_look", {}).get(look, rl.get("gain", 1.0)), float), (3,))
            rgba = np.ones(4, np.float32)
            rgba[:3] = srgb_from_linear(linear_from_srgb(np.asarray(rgb, float)[:3]) * gain)
            cfg = self._mesh_config(len(V), len(I))
            m = self.ctx.create_mesh(cfg)
            self._mesh_upload(m, cfg, V, N, U, I)
            mkey = (look, id(tex), tuple(np.round(rgba, 6)), float(rough), float(metal), float(emis))
            material = mats.get(mkey)
            if material is None:
                material = mats[mkey] = F.Material(
                    color=rgba, roughness=float(np.clip(rough * rl.get("roughness_scale", 1.0), 0.05, 1.0)),
                    metallic=float(metal) * float(rl.get("metallic_scale", 1.0)), specular=float(rl.get("specular", 0.3)),
                    emissive=float(emis), color_texture=tex)
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadow=False, receive_shadow=True))
            r.set_mesh(m, 0, len(I))
            r.set_material(material)
            self.scn.add_renderable(r)
            self.rend.append((r, frame, m, material))
            return m, cfg, r

        for it in sc.items:
            try:
                tex = None if it["tex"] is None else texture(it["tex"], stat[it["tex"]])
                add(it["mesh"], it["mat"], tex, it["frame"], it["look"])
            except Exception as exc:  # noqa: BLE001
                raise RenderError("renderer: part %r of %r: %s: %s" % (it["look"], it["frame"], type(exc).__name__, exc))
        # the fabric: the printed face with the rim, and the under side; their vertices are set at every draw
        nx, ny = 4 * (RL.NX - 1) + 1, 4 * (RL.NY - 1) + 1
        uv, f_top, ring, inner, f_edge = CS.topology(nx, ny)
        n0, nr = nx * ny, len(ring)
        uv = uv.astype(np.float32)
        i_face = np.concatenate([f_top.reshape(-1), (f_edge + n0).reshape(-1)]).astype(np.uint32)
        i_face_flip = np.concatenate([f_top.reshape(-1), (f_edge[:, ::-1] + n0).reshape(-1)]).astype(np.uint32)
        i_back = np.ascontiguousarray(f_top[:, ::-1].reshape(-1), np.uint32)
        u_face = np.concatenate([uv, uv[ring], uv[ring]])
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")           # a flat sheet far under the floor until the first draw
        flat = np.stack([ii * 0.002, jj * 0.002, 0.0 * ii - 30.0], -1).reshape(-1, 3).astype(np.float32)
        up = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (n0 + 2 * nr, 1))
        z = np.concatenate([flat, flat[ring] + [0, 0, 0.001], flat[ring]]).astype(np.float32)
        look = int(sc.look)
        if "look%02d_face" % look not in stat.files:
            raise RenderError("renderer: the bundle's library has no look %d" % look)
        fm, fc, fr = add((z, up, u_face, i_face), (np.ones(3), float(stat["look_rough"][look]), 0.0, 0.0),
                         texture("look_face", stat["look%02d_face" % look]), "sheet", "fabric_face", dynamic=True)
        bm, bc, br = add((z[:n0], -up[:n0], uv, i_back), (np.ones(3), float(stat["look_rough"][look]), 0.0, 0.0),
                         texture("look_back", stat["look%02d_back" % look]), "sheet", "fabric_back", dynamic=True)
        self.sheet = {"face": (fm, fc, fr, u_face, i_face, i_face_flip), "back": (bm, bc, br, uv, i_back), "ring": ring, "inner": inner,
                      "f_edge": f_edge, "n0": n0, "flip": None}
        for k, (mesh, mat, img, cls) in enumerate(self._room()):
            add(mesh, mat, None if img is None else texture(("room", k), img), "world", cls)
        self.textures = texs
        self.moving = [(r, f) for r, f, _m, _mt in self.rend if f in ("clamp", "pad") or (isinstance(f, tuple) and f[0] == "body")]
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

    def draw(self, poses, sheet, pos, forward, up, frustum, width, height):
        F = self.F
        xp, xm = poses["xpos"], poses["xmat"]
        for r, f in self.moving:
            if isinstance(f, tuple):
                r.set_pose(xp[f[1]], xm[f[1]])
            else:
                r.set_pose(poses[f][0], poses[f][1])
        top, bot, n = sheet
        S_ = self.sheet
        ring, inner = S_["ring"], S_["inner"]
        out = top[ring] - top[inner]
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        rim = np.concatenate([top[ring], bot[ring]])
        if S_["flip"] is None:                       # the rim's triangles face outwards: decided once, at the first frame
            fe = S_["f_edge"]
            fn = np.cross(rim[fe[:, 1]] - rim[fe[:, 0]], rim[fe[:, 2]] - rim[fe[:, 0]])
            S_["flip"] = bool((fn * np.concatenate([out, out])[fe[:, 0]]).sum() < 0)
        fm, fc, fr, u_face, i_face, i_face_flip = S_["face"]
        idx = i_face_flip if S_["flip"] else i_face
        self._mesh_upload(fm, fc, np.concatenate([top, rim]), np.concatenate([n, out, out]), u_face, idx)
        fr.set_mesh(fm, 0, len(idx))                 # the renderable takes the mesh's new bounds (it is culled by them)
        bm, bc, br, uvb, i_back = S_["back"]
        self._mesh_upload(bm, bc, bot, -n, uvb, i_back)
        br.set_mesh(bm, 0, len(i_back))
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
    # as linked into MuJoCo 3.16).
    # FXAA is switched off on the same view (View::setAntiAliasing(AntiAliasing::NONE)): with it, one image in about
    # two hundred differed between two fresh processes in a single pixel (four levels after the downsampling; the
    # same two variants every time, about 7 : 3, whatever lavapipe's thread count, with ambient occlusion on or off);
    # without it the same state gave one image in every process. Edges are smoothed by the supersampling instead
    # (clook.json: supersample).
    VIEW_SET_DITHERING = "_ZN8filament4View12setDitheringENS_9DitheringE"
    VIEW_GET_DITHERING = "_ZNK8filament4View12getDitheringEv"
    VIEW_SET_ANTIALIASING = "_ZN8filament4View15setAntiAliasingENS_12AntiAliasingE"
    VIEW_GET_ANTIALIASING = "_ZNK8filament4View15getAntiAliasingEv"

    def _undither(self, width, height, shadows, post):
        lib = ctypes.CDLL(self.F.__file__)
        try:
            setd, getd = getattr(lib, self.VIEW_SET_DITHERING), getattr(lib, self.VIEW_GET_DITHERING)
        except AttributeError as exc:
            raise RenderError("renderer: MuJoCo's Filament module exports no View::setDithering: %s" % exc)
        try:
            seta, geta = getattr(lib, self.VIEW_SET_ANTIALIASING), getattr(lib, self.VIEW_GET_ANTIALIASING)
        except AttributeError as exc:
            raise RenderError("renderer: MuJoCo's Filament module exports no View::setAntiAliasing: %s" % exc)
        for fn in (setd, seta):
            fn.restype, fn.argtypes = None, [ctypes.c_void_p, ctypes.c_uint8]
        for fn in (getd, geta):
            fn.restype, fn.argtypes = ctypes.c_uint8, [ctypes.c_void_p]
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
            raise RenderError("renderer: %s to switch its dithering off (%d candidates)" % (VIEW_NOT_FOUND, len(found)))
        self.view = found.pop()
        setd(self.view, 0)                             # filament::View::Dithering::NONE
        if getd(self.view) != 0:
            raise RenderError("renderer: Filament's dithering did not switch off")
        if not self.look["render"].get("fxaa", False):
            seta(self.view, 0)                         # filament::View::AntiAliasing::NONE
            if geta(self.view) != 0:
                raise RenderError("renderer: Filament's FXAA did not switch off")



    def info(self):
        return {"stats": dict(self.stats), "start_s": self.start_s, "renderables": len(self.rend) if self.scene else 0,
                "textures": len(self.textures) if self.scene else 0, "triangles": self.scene.triangles if self.scene else 0}

    def _shutdown(self):
        self.targets.clear()
        self.rend, self.moving, self.textures, self.sheet = [], [], {}, None
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
