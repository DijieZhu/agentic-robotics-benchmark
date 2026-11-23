"""The UAV's survey camera and the Go1's mast camera: MuJoCo 3.16's Filament renderer (physically based) on Vulkan; in
the task's image Vulkan is Mesa's lavapipe on the CPU (no GPU).

What is drawn
  * the hall the course stands in (walls, roof, beams, ducts, cable trays, doors, lamp fixtures), its dressing outside
    the course (racks with boxes, a forklift, a reach truck, pallets, a table, carts) and the floor with its wear: one
    mesh, the same at every site (scene.npz of the render bundle, hall_look/ in the task's assets);
  * the course parts of the site, generated for the site by look_parts.py from the site's own numbers: steel barriers
    with orange caps and hazard ends, the duct, the tread-plate deck with its guides, the teal covers, bollards with
    green caps, the blue dock mat, the dock beacon, the magenta pallet in its foot brackets, the charging station, the lane;
  * the Go1 (the simulated model's own link meshes), its equipment (lidar in its guard frame, compute unit,
    line profiler, bracket, cables) and the UAV, each moved as a whole with its body.
Light: an image light (brightest overhead, dimmer towards the horizon, dark below, the same in every compass direction:
the hall's lamp grid) and ambient occlusion; no lamp and no shadow map. The lamp fixtures glow.
Cameras: position, axes and field of view are the simulated cameras' own (the caller passes them). An image is drawn
at twice its size and averaged over 2 x 2 blocks in linear light (integer arithmetic).

An image is a function of the site and the state alone: the same state gives the same bytes however often it is
drawn, whatever was drawn before, in whichever process. Filament's temporal dithering is off; every Filament object
is made once per process, in one order; one process draws one site (Cameras starts a worker of this module and ends
it with the site). Everything is drawn with one material (one atlas); everything that stands still in a scene is one
mesh and one draw call; no mesh is uploaded after the site is built.

    cams = Cameras({"survey": spec, "found": spec})        # starts the worker: python3 -I hall_render.py --worker
    image = cams.image("survey", eye, forward, up, fovy)                    # uint8 (256, 256, 3), rows top-down
    image = cams.image("found", eye, forward, up, fovy, bodies=..., uav=(x, y, z))
    cams.close()

A scene's spec: {"walls": {name: [x0, x1, y0, y1]}, "fittings": [[kind, x, lo, hi, length, clearance, rise]],
"posts": [[x, y]], "goal": [x, y] or None, "beacon": [x, y] or None, "pallet": [x0, x1, y0, y1] or None,
"bound": the north wall's inner face, "lane_y": its outer face (the painted lane) or None, "station_y": where the
charging station stands or None}.

Every failure (a missing file of the bundle, a worker that dies or does not answer within HALL_RENDER_TIMEOUT
seconds) is raised as RenderError: an error of the environment, never of a mission program.
"""
import ctypes
import gc
import json
import os
import select
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.environ.get("HALL_RENDER_DIR") or os.path.join(os.environ.get("VISUAL_ROBOTICS_ASSETS", "/opt/visual_robotics/assets"), "hall_look")
CALL_TIMEOUT = float(os.environ.get("HALL_RENDER_TIMEOUT", "300"))
SIZE = 256
MAGIC = b"HALL"
BODIES = ("trunk", "FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf", "RR_hip", "RR_thigh", "RR_calf",
          "RL_hip", "RL_thigh", "RL_calf", "inspection_equipment")
RECORD = 2 + 11 + 12 * len(BODIES) + 3          # scene, flags, camera, the bodies' poses, the UAV's place
AWAY = (0.0, 0.0, -500.0)                        # where a body that is not in a picture is put


class RenderError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------------------------
# The client: one worker process per site
# ----------------------------------------------------------------------------------------------------------------
class Cameras(object):
    def __init__(self, scenes, render_dir=None):
        self.names = sorted(scenes)
        env = dict(os.environ)
        if render_dir:
            env["HALL_RENDER_DIR"] = render_dir
        self.proc = subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__), "--worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=None if os.environ.get("HALL_RENDER_DEBUG") else subprocess.DEVNULL, env=env, cwd=HERE)
        try:
            self.proc.stdin.write((json.dumps({"scenes": {k: scenes[k] for k in self.names}}) + "\n").encode())
            self.proc.stdin.flush()
            head = self._read(8)
        except (OSError, RenderError) as exc:
            self.close()
            raise RenderError("renderer: the worker did not start: %s" % exc)
        if head[:4] != MAGIC or struct.unpack("<I", head[4:])[0] != RECORD:
            self.close()
            raise RenderError("renderer: the worker answered something else")

    def _read(self, n):
        buf, fd = b"", self.proc.stdout.fileno()
        while len(buf) < n:
            ready, _, _ = select.select([fd], [], [], CALL_TIMEOUT)
            if not ready:
                raise RenderError("renderer: no answer within %g s" % CALL_TIMEOUT)
            chunk = os.read(fd, n - len(buf))
            if not chunk:
                raise RenderError("renderer: the worker ended")
            buf += chunk
        return buf

    def image(self, scene, eye, forward, up, fovy, near=0.02, bodies=None, uav=None):
        """bodies: {name: (position 3, rotation matrix 9)} of the Go1's bodies, or None (no Go1 in the picture);
        uav: the UAV's place, or None (not in the picture)"""
        rec = np.zeros(RECORD, np.float64)
        rec[0] = self.names.index(scene)
        rec[1] = (1 if bodies is not None else 0) + (2 if uav is not None else 0)
        rec[2:5], rec[5:8], rec[8:11], rec[11], rec[12] = eye, forward, up, fovy, near
        if bodies is not None:
            for i, name in enumerate(BODIES):
                pos, mat = bodies[name]
                rec[13 + 12 * i:16 + 12 * i] = pos
                rec[16 + 12 * i:25 + 12 * i] = np.asarray(mat, np.float64).reshape(9)
        if uav is not None:
            rec[-3:] = uav
        try:
            self.proc.stdin.write(rec.tobytes())
            self.proc.stdin.flush()
            raw = self._read(SIZE * SIZE * 3)
        except OSError as exc:
            raise RenderError("renderer: the worker is gone: %s" % exc)
        return np.frombuffer(raw, np.uint8).reshape(SIZE, SIZE, 3).copy()

    def close(self):
        proc, self.proc = getattr(self, "proc", None), None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stdout.close()

    def __del__(self):
        self.close()


# ----------------------------------------------------------------------------------------------------------------
# The site's course parts in the atlas
# ----------------------------------------------------------------------------------------------------------------
def _clip_half(poly, axis, bound, keep_above):
    c = 6 + axis
    d = (poly[:, c] - bound) * (1 if keep_above else -1)
    out, m = [], len(poly)
    for i in range(m):
        a, b, da, db = poly[i], poly[(i + 1) % m], d[i], d[(i + 1) % m]
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            v = a + (da / (da - db)) * (b - a)
            v[c] = bound
            out.append(v)
    return np.array(out) if len(out) >= 3 else None


def cut_tiles(V):
    """triangles (n, 3, 8: place, normal, texture coordinate) -> the same surface with every triangle inside one
    repeat of its picture and its coordinates in [0, 1]"""
    uv = V[:, :, 6:8]
    cell = np.floor(uv.mean(1))
    inside = np.all(uv.min(1) >= cell - 1e-6, 1) & np.all(uv.max(1) <= cell + 1 + 1e-6, 1)
    done = V[inside].copy()
    done[:, :, 6:8] -= cell[inside][:, None, :]
    out = [done]
    for tri in V[~inside]:
        polys = [tri.astype(np.float64)]
        for axis in (0, 1):
            nxt = []
            for p in polys:
                a0, a1 = p[:, 6 + axis].min(), p[:, 6 + axis].max()
                for k in range(int(np.floor(a0 + 1e-9)), int(np.ceil(a1 - 1e-9))):
                    q = _clip_half(p, axis, float(k), True)
                    q = _clip_half(q, axis, float(k + 1), False) if q is not None else None
                    if q is not None:
                        nxt.append(q)
            polys = nxt
        for p in polys:
            p = p.copy()
            p[:, 6:8] -= np.floor(p[:, 6:8].mean(0))
            for j in range(1, len(p) - 1):
                t = np.stack([p[0], p[j], p[j + 1]])
                if np.linalg.norm(np.cross(t[1, :3] - t[0, :3], t[2, :3] - t[0, :3])) > 1e-12:
                    out.append(t[None])
    V2 = np.concatenate(out)
    V2[:, :, 6:8] = np.clip(V2[:, :, 6:8], 0.0, 1.0)
    return V2


def course(spec, looks, pallet_model):
    """(places, normals, atlas coordinates), float32, three rows a triangle: the spec's course parts"""
    import look_parts as LP
    w = LP.Soup()
    LP.looks2(w)
    S = "static"
    bound = float(spec["bound"])
    pal = spec.get("pallet")
    for name, (x0, x1, y0, y1) in spec.get("walls", {}).items():
        rack = name.startswith("rack")
        closed = pal is not None and name.startswith("rack2")          # no hazard end where the pallet stands against it
        lo_open = rack and y0 > -bound + 1e-6 and not (closed and abs(y0 - pal[3]) < 0.2)
        hi_open = rack and y1 < bound - 1e-6 and not (closed and abs(y1 - pal[2]) < 0.2)
        LP.barrier(w, S, name, x0, x1, y0, y1, rack, lo_open, hi_open)
    if spec.get("fittings"):
        LP.fittings(w, S, {"fittings": [list(f) for f in spec["fittings"]]})
    for x, y in spec.get("posts", []):
        LP.bollard(w, S, x, y)
    if spec.get("goal") is not None:
        gx, gy = spec["goal"]
        w.mesh(S, "dock_mat", LP.rslab((gx, gy), .29, .28, .03, 0.0, 0.004, 0.0015, 0.0), "mat_blue")
        w.mesh(S, "dock_nose", LP.box((gx + .261, gy, .005), (.026, .26, .001), 0.0005), "paint_yellow")
        for s in (-1, 1):
            w.mesh(S, "dock_contact", LP.box((gx + .12, gy + s * .09, .0045), (.07, .012, .0006), 0.0003), "brass")
    if spec.get("beacon") is not None:
        LP.beacon(w, S, *spec["beacon"])
    if pal is not None:
        LP.pallet_feet(w, S, pal)
    if spec.get("station_y") is not None:
        LP.charger(w, S, float(spec["station_y"]), bound)
    if spec.get("lane_y") is not None:
        LP.lanes(w, S, float(spec["lane_y"]))
    W, H = looks["atlas"]
    Vs, Ns, Ts = [], [], []
    for _parent, _name, look, T in w.parts:
        x0, y0, pw, ph = looks["looks"][look]["rect"]
        if looks["looks"][look]["tiled"]:
            T = cut_tiles(T)
            uv = T[:, :, 6:8]
            t = np.stack([(x0 + uv[..., 0] * pw) / W, (y0 + (1.0 - uv[..., 1]) * ph) / H], 2)
        else:
            t = np.tile(np.array([(x0 + pw / 2.0) / W, (y0 + ph / 2.0) / H]), (len(T), 3, 1))
        Vs.append(T[:, :, :3]); Ns.append(T[:, :, 3:6]); Ts.append(t)
    out = [np.concatenate(x).reshape(-1, x[0].shape[-1]).astype(np.float32) for x in (Vs, Ns, Ts)]
    if pal is not None:
        M = LP.pallet_matrix(pal)
        pv, pn, pt = pallet_model
        out[0] = np.concatenate([out[0], (pv.astype(np.float64) @ M[:3, :3].T + M[:3, 3]).astype(np.float32)])
        out[1] = np.concatenate([out[1], (pn.astype(np.float64) @ M[:3, :3].T).astype(np.float32)])
        out[2] = np.concatenate([out[2], pt])
    return out


# ----------------------------------------------------------------------------------------------------------------
# The worker: Filament, one site, one process
# ----------------------------------------------------------------------------------------------------------------
def _linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin16():
    f = [int(round(65535.0 * _linear(i / 255.0))) for i in range(256)]
    fwd = np.array(f, np.uint32)
    edges = (fwd[1:] + fwd[:-1] + 1) // 2
    back = np.searchsorted(edges, np.arange(65536, dtype=np.uint32), side="right").astype(np.uint8)
    return fwd, back


def box_down(img, k, tables):
    fwd, back = tables
    h, w = img.shape[0] // k, img.shape[1] // k
    s = fwd[img].reshape(h, k, w, k, 3).sum(axis=(1, 3), dtype=np.uint32)
    return back[(s + (k * k) // 2) // (k * k)]


class Worker(object):
    VIEW_SET_DITHERING = "_ZN8filament4View12setDitheringENS_9DitheringE"
    VIEW_GET_DITHERING = "_ZNK8filament4View12getDitheringEv"

    def __init__(self, scenes, render_dir=None):
        self.dir = render_dir or RENDER_DIR
        try:
            self.look = json.load(open(os.path.join(self.dir, "look.json")))
            self.looks = json.load(open(os.path.join(self.dir, "looks.json")))
        except OSError as exc:
            raise RenderError("renderer: the render bundle is incomplete: %s" % exc)
        for key in ("DISPLAY", "WAYLAND_DISPLAY"):
            os.environ.pop(key, None)
        os.environ["MESA_SHADER_CACHE_DISABLE"] = "true"
        os.environ["LP_NUM_THREADS"] = str(int(self.look["render"].get("lavapipe_threads", 2)))
        import mujoco
        import mujoco._render_filament as F
        from mujoco.experimental.studio import window  # noqa: F401  registers Filament's resource providers
        from PIL import Image
        self.mj, self.F = mujoco, F
        look = self.look
        self.k = max(1, int(look["render"].get("supersample", 2)))
        self.tables = _lin16()
        self.shadows = bool(look.get("lamp", {}).get("shadow"))
        self.ctx = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN))
        self.ctx.set_clear_color([float(x) for x in look["render"].get("clear_color", [0.0, 0.0, 0.0])])
        self.fscn = self.ctx.create_scene(F.SceneParams())
        spec = mujoco.MjSpec()
        for k, v in sorted(look["filament_numeric"].items()):
            spec.add_numeric(name=k, data=[float(x) for x in (v if isinstance(v, (list, tuple)) else [v])])
        for k, v in sorted(look["filament_text"].items()):
            spec.add_text(name=k, data=str(v))
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], pos=[0, 0, -50])
        self.look_model = spec.compile()
        for _ in range(2):                             # MuJoCo 3.16 applies the colour grading one configure late
            self.fscn.configure_from_model(self.look_model)
        try:
            raw = open(os.path.join(self.dir, look["ibl"]["file"]), "rb").read()
            scene = np.load(os.path.join(self.dir, "scene.npz"))
            pictures = [np.ascontiguousarray(np.asarray(Image.open(os.path.join(self.dir, "atlas_%s.png" % n)).convert("RGB")))
                        for n in ("colour", "orm", "glow")]
        except OSError as exc:
            raise RenderError("renderer: the render bundle is incomplete: %s" % exc)
        tc = F.TextureConfig()
        tc.width, tc.height, tc.format = len(raw), 0, F.PixelFormat.PIXEL_FORMAT_KTX
        tc.sampler_type, tc.color_space = F.SamplerType.TEXTURE_CUBE, F.ColorSpace.COLORSPACE_LINEAR
        self.ibl_tex = self.ctx.create_texture(tc)
        td = F.TextureData()
        td.set_image(raw)
        self.ibl_tex.upload(td)
        lp = F.LightParams()
        lp.type, lp.texture = F.LightType.LIGHT_IMAGE, self.ibl_tex
        lp.color, lp.intensity = [float(x) for x in look["ibl"].get("color", [1.0, 1.0, 1.0])], float(look["ibl"]["intensity"])
        self.ibl = self.ctx.create_light(lp)
        self.fscn.add_light(self.ibl)
        self.lamp = None
        lamp = look.get("lamp")
        if lamp and lamp.get("lux", 0) > 0:
            lp = F.LightParams()
            lp.type, lp.color, lp.intensity = F.LightType.LIGHT_DIRECTIONAL, [float(x) for x in lamp.get("color", [1, 1, 1])], float(lamp["lux"])
            lp.cast_shadow = self.shadows
            if lamp.get("shadow_map"):
                lp.shadow_map_size = int(lamp["shadow_map"])
            self.lamp = self.ctx.create_light(lp)
            d = np.asarray(lamp.get("dir", [0, 0, -1]), np.float32)
            self.lamp.set_pose(np.zeros(3, np.float32), d / np.linalg.norm(d))
            self.fscn.add_light(self.lamp)

        def texture(img, space):
            h, w = img.shape[:2]
            tc = F.TextureConfig()
            tc.width, tc.height, tc.format = w, h, F.PixelFormat.PIXEL_FORMAT_RGB8
            tc.sampler_type, tc.color_space = F.SamplerType.TEXTURE_2D, space
            t = self.ctx.create_texture(tc)
            td = F.TextureData()
            td.set_image(img.tobytes())
            t.upload(td)
            return t
        self.atlas = [texture(pictures[0], F.ColorSpace.COLORSPACE_SRGB), texture(pictures[1], F.ColorSpace.COLORSPACE_LINEAR),
                      texture(pictures[2], F.ColorSpace.COLORSPACE_SRGB)]
        m = look["material"]
        self.material = F.Material(color=np.ones(4, np.float32), roughness=1.0, metallic=1.0, specular=float(m["specular"]),
                                   glossiness=0.05, reflectance=0.0, emissive=float(m["glow"]),
                                   color_texture=self.atlas[0], orm_texture=self.atlas[1], emissive_texture=self.atlas[2])
        A, Us, Ty = F.VertexAttribute, F.VertexAttributeUsage, F.VertexAttributeType
        attrs = [A(usage=Us.VERTEX_ATTRIBUTE_USAGE_POSITION, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                 A(usage=Us.VERTEX_ATTRIBUTE_USAGE_NORMAL, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                 A(usage=Us.VERTEX_ATTRIBUTE_USAGE_UV, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT2)]

        def unit(P, Nn, T):
            nv = len(P)
            cfg = F.MeshConfig(max_vertices=nv, max_indices=nv, interleaved=False, index_type=F.IndexType.INDEX_TYPE_U32,
                               primitive_type=F.MeshPrimitiveType.MESH_PRIMITIVE_TYPE_TRIANGLES, attributes=attrs)
            mesh = self.ctx.create_mesh(cfg)
            md = F.MeshData(compute_bounds=True)
            md.set_vertices(cfg, np.concatenate([P.ravel(), Nn.ravel(), T.ravel()]).astype(np.float32).tobytes())
            md.set_indices(cfg, np.arange(nv, dtype=np.uint32).tobytes())
            mesh.upload(md)
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadow=self.shadows, receive_shadow=True))
            r.set_mesh(mesh, 0, nv)
            r.set_material(self.material)
            return mesh, r, nv // 3
        # one standing mesh per scene: the hall and the scene's course parts. Only the scene in the picture is in the
        # Filament scene; the others wait outside it.
        hall = tuple(scene["hall_" + a] for a in ("V", "N", "T"))
        pallet = tuple(scene["pallet_" + a] for a in ("V", "N", "T"))
        self.names = sorted(scenes)
        self.standing, self.triangles = [], {}
        for name in self.names:
            parts = course(scenes[name], self.looks, pallet)
            mesh, r, n = unit(*(np.concatenate([h, p]) for h, p in zip(hall, parts)))
            self.standing.append((mesh, r))
            self.triangles[name] = n
        self.now = None
        self.bodies = []
        for name in BODIES + ("uav",):
            key = name if name == "uav" else "go1_" + name
            mesh, r, n = unit(*(scene[key + "_" + a] for a in ("V", "N", "T")))
            r.set_pose(np.asarray(AWAY, np.float32), np.eye(3, dtype=np.float32).reshape(-1))
            self.fscn.add_renderable(r)
            self.bodies.append((mesh, r))
            self.triangles[name] = n
        size = SIZE * self.k
        self.buf = np.zeros((size, size, 3), np.uint8)
        self.tgt = self.ctx.create_render_target(F.RenderTargetConfig(color_format=F.PixelFormat.PIXEL_FORMAT_RGB8))
        self.tgt.resize(size, size)
        self.read = F.ReadRequest()
        self.read.target = self.tgt
        self.read.set_buffer(self.buf)
        self.view = None

    def draw(self, rec):
        F = self.F
        scene, flags = int(rec[0]), int(rec[1])
        if scene != self.now:
            if self.now is not None:
                self.fscn.remove_renderable(self.standing[self.now][1])
            self.fscn.add_renderable(self.standing[scene][1])
            self.now = scene
        eye3 = np.eye(3, dtype=np.float32).reshape(-1)
        for i in range(len(BODIES)):
            if flags & 1:
                self.bodies[i][1].set_pose(rec[13 + 12 * i:16 + 12 * i].astype(np.float32), rec[16 + 12 * i:25 + 12 * i].astype(np.float32))
            else:
                self.bodies[i][1].set_pose(np.asarray(AWAY, np.float32), eye3)
        self.bodies[-1][1].set_pose((rec[-3:] if flags & 2 else np.asarray(AWAY)).astype(np.float32), eye3)
        near, fovy = float(rec[12]), float(rec[11])
        t = near * np.tan(np.radians(fovy) / 2.0)
        cam = F.Camera(pos=rec[2:5].astype(np.float32), forward=rec[5:8].astype(np.float32), up=rec[8:11].astype(np.float32),
                       frustum_center=0.0, frustum_width=float(t), frustum_bottom=float(-t), frustum_top=float(t),
                       frustum_near=near, frustum_far=float(self.look["render"].get("far", 120.0)))
        size = SIZE * self.k
        post = bool(self.look["render"]["post_processing"])
        req = F.RenderRequest(camera=cam, draw_mode=F.DrawMode.DRAW_MODE_DEFAULT, post_processing=post, shadows=self.shadows, reflections=False)
        req.scene, req.target = self.fscn, self.tgt
        req.viewport.left, req.viewport.bottom, req.viewport.width, req.viewport.height = 0, 0, size, size
        self.ctx.wait_for_frame(self.ctx.render([req], [self.read]))
        if self.view is None:                          # the process's first image: switch the dithering off, again
            self._undither(size, size, self.shadows, post)
            self.ctx.wait_for_frame(self.ctx.render([req], [self.read]))
        return self.buf.copy() if self.k == 1 else box_down(self.buf, self.k, self.tables)

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
        try:
            level = [(id(self.fscn), 64)]
            for _depth in range(3):                    # breadth first: the view is found the same way in every process
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
            raise RenderError("renderer: cannot find the Filament view to switch its dithering off (%d candidates)" % len(found))
        self.view = found.pop()
        setd(self.view, 0)                             # filament::View::Dithering::NONE
        if getd(self.view) != 0:
            raise RenderError("renderer: Filament's dithering did not switch off")

    def close(self):
        """Every Filament object before the context (destroying the context first panics)."""
        if self.now is not None:
            self.fscn.remove_renderable(self.standing[self.now][1])
        for _mesh, r in self.bodies:
            self.fscn.remove_renderable(r)
        self.standing, self.bodies, self.now = [], [], None
        self.material = None
        self.atlas = []
        self.read = self.tgt = None
        self.ibl = self.ibl_tex = self.lamp = None
        self.fscn = None
        self.look_model = None
        gc.collect()
        self.ctx = None
        gc.collect()


def _worker_main():
    try:
        ctypes.CDLL("libc.so.6").mallopt(-8, 2)       # M_ARENA_MAX: before Filament's and lavapipe's threads allocate
    except (OSError, AttributeError):
        pass
    out = os.fdopen(os.dup(1), "wb", buffering=0)     # the protocol's pipe; fd 1 goes to stderr for Filament's banner
    os.dup2(2, 1)
    sys.path.insert(0, HERE)
    inp = sys.stdin.buffer
    head = json.loads(inp.readline().decode())
    worker = Worker(head["scenes"])
    out.write(MAGIC + struct.pack("<I", RECORD))
    try:
        while True:
            raw = inp.read(8 * RECORD)
            if len(raw) < 8 * RECORD:
                break
            out.write(worker.draw(np.frombuffer(raw, np.float64)).tobytes())
    finally:
        worker.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        _worker_main()
    else:
        sys.exit("usage: hall_render.py --worker   (started by Cameras)")
