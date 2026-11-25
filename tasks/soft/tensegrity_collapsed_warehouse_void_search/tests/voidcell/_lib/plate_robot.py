"""The robot layer of the plate backend: MuJoCo's physically based renderer (mujoco._render_filament, MuJoCo 3.14,
Vulkan; Mesa's lavapipe on the CPU in the task image) renders ONLY the robot (the cell's decimated copy of the
authors' strut mesh, five material groups) in a window of the search camera on the 4x film.

    lay = RobotLayer(look, sun_dir, sun_lux_bake, probes, cam4)    # one per episode; at most one per process
    lay.set_qpos(q); lay.use_probe(i)
    out = lay.render((u0, v0, w, h), ("amb", "lit"))               # window in 4x pixels -> {kind: float32 linear}
    lay.close()                                                     # objects, then the context; malloc_trim

Two Filament scenes share the model's objects: "amb" = the robot + the nearest probe's image light (intensity I);
"lit" = the robot + the same probe at I x G_amb / G_sun + the site's sun (the model's directional light, casting the
robot's self-shadow). Nothing but the robot is in either scene: the environment's light is in the plate and its maps.
The sun's colour is the look's robot_layer.sun_rgb_linear (the bake's sun in the
bundle's white-balanced linear space, luminance 1); the light gets that colour scaled to a maximum of 1 and the
intensity times its maximum; a look without the key keeps a white sun.
View options from the look (model custom fields filament.*): tone mapping linear, AO, SSCT, bloom, MSAA, vignette,
fog off, PCF shadows (MuJoCo 3.14 has no dithering or TAA switch). The camera is the cell's exact off-axis pinhole
(ysim ClassicSession) cut to the window: the same frustum arithmetic, near 0.05 m, far 50 m.

Practices kept from earlier MuJoCo Filament cameras: ONE render target
and ONE ReadPixelsRequest per pass, reused (a new request per frame leaks); every mesh without texture coordinates
gets a per-vertex UV0 of zeros (Filament's PBR material declares one; the compiled mesh is unchanged); the context
outlives every scene object and is released when the layer closes (the engine keeps an earlier session's objects
while its context lives: ~2.6 GB per episode on lavapipe, measured); DISPLAY / WAYLAND_DISPLAY are dropped
before Filament starts (its resource providers would connect to an X server that the verify freezer stops).
"""
import atexit
import ctypes
import gc
import os
import time

import numpy as np

from . import plate_look

NEAR, FAR = 0.05, 50.0
KINDS = ("amb", "lit")
_LIVE = set()


class RobotLayerError(RuntimeError):
    pass


def _headless():
    for k in ("DISPLAY", "WAYLAND_DISPLAY"):
        os.environ.pop(k, None)
    if not os.environ.get("VK_ICD_FILENAMES"):               # the CPU driver when the image has it (no GPU in a task)
        for p in ("/usr/share/vulkan/icd.d/lvp_icd.x86_64.json", "/usr/share/vulkan/icd.d/lvp_icd.json"):
            if os.path.isfile(p):
                os.environ["VK_ICD_FILENAMES"] = p
                break


def trim_heap():
    """Hand the freed heap back to the system (glibc malloc_trim); a no-op elsewhere."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def _shutdown():
    for lay in list(_LIVE):
        try:
            lay.close()
        except Exception:  # noqa: BLE001
            pass


atexit.register(_shutdown)


def _srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def _quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _uv0(spec):
    """A per-vertex UV0 of zeros on every mesh without texture coordinates; a file mesh becomes a user mesh of its
    compiled data in the asset frame (asset = R(mesh_quat) v + mesh_pos), so it compiles to the same mesh."""
    m = spec.compile()
    index = {m.mesh(i).name: i for i in range(m.nmesh)}
    for ms in list(spec.meshes):
        i = index.get(ms.name)
        if i is None or int(m.mesh_texcoordnum[i]) > 0:
            continue
        if not ms.file:
            ms.usertexcoord = [0.0] * (2 * (len(ms.uservert) // 3))
            continue
        nv, nf = int(m.mesh_vertnum[i]), int(m.mesh_facenum[i])
        va, fa, na = int(m.mesh_vertadr[i]), int(m.mesh_faceadr[i]), int(m.mesh_normaladr[i])
        R, p = _quat_R(m.mesh_quat[i]), np.asarray(m.mesh_pos[i], np.float64)
        V = np.asarray(m.mesh_vert[va:va + nv], np.float64) @ R.T + p
        N = np.asarray(m.mesh_normal[na:na + int(m.mesh_normalnum[i])], np.float64) @ R.T
        F = np.asarray(m.mesh_face[fa:fa + nf], np.int64)
        FN = np.asarray(m.mesh_facenormal[fa:fa + nf], np.int64)
        ms.file = ""
        ms.scale = [1.0, 1.0, 1.0]
        ms.uservert = V.ravel().tolist()
        ms.usernormal = N.ravel().tolist()
        ms.userface = F.ravel().tolist()
        ms.userfacenormal = FN.ravel().tolist()
        ms.usertexcoord = [0.0] * (2 * nv)
        ms.userfacetexcoord = F.ravel().tolist()


def robot_xml(mesh_dir):
    """MJCF of the robot alone: the vendored simulation base's model (visual, decimated meshes, no lights, no floor),
    the same bodies, joints and geoms as the robot in the cell's render model."""
    from cellcore import world as wmod
    ys = wmod.ysim()
    cfg = ys.model.ModelConfig(visual=True, mesh_res="dec", lights=False, floor=False, assets=mesh_dir,
                               offscreen=(640, 480))
    xml, _info = ys.model.build_xml(cfg)
    return xml


def robot_spec(mesh_dir, look, sun_dir, sun_lux):
    """MjSpec of the robot layer: the robot alone, the look's PBR materials (group gains in linear albedo), collision
    geoms removed, the site's sun as a directional shadow-casting light, the view options as custom fields."""
    import mujoco
    spec = mujoco.MjSpec.from_string(robot_xml(mesh_dir))
    mats = look["materials"]
    gains = look["group_gain"]
    seen = set()
    for m in spec.materials:
        if m.name not in mats:
            continue
        rec = mats[m.name]
        lin = (np.asarray(rec["albedo_linear"], float) if "albedo_linear" in rec
               else _srgb_to_linear(rec["albedo_srgb"]))
        lin = np.clip(lin * float(gains.get(rec["group"], 1.0)), 0.0, 1.0)
        m.rgba = list(_linear_to_srgb(lin)) + [1.0]
        m.roughness, m.metallic = float(rec["roughness"]), float(rec["metallic"])
        m.specular, m.shininess, m.reflectance = 0.5, 0.5, 0.0
        seen.add(m.name)
    if seen != set(mats):
        raise RobotLayerError("the robot model has no material(s) %s of the look" % sorted(set(mats) - seen))
    for g in list(spec.geoms):                         # the visual geoms only (groups 0-2); collision geoms are group 3
        if int(g.group) > 2:
            spec.delete(g)
    k = np.asarray(sun_dir, float)
    k = k / np.linalg.norm(k)
    rl = look["robot_layer"]
    col = [1.0, 1.0, 1.0]
    if rl.get("sun_rgb_linear") is not None:          # the balanced sun colour (max 1; the intensity carries the rest)
        c = np.asarray(rl["sun_rgb_linear"], float)
        cm = float(c.max())
        if not (c.shape == (3,) and cm > 0 and np.all(c > 0)):
            raise RobotLayerError("the look's robot_layer.sun_rgb_linear %r is not a positive colour" % (rl["sun_rgb_linear"],))
        col = [float(q) for q in c / cm]
        sun_lux = float(sun_lux) * cm
    spec.worldbody.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, dir=list(k), pos=[2.2, 2.0, 6.0],
                             castshadow=True, intensity=float(sun_lux), diffuse=col,
                             specular=col, ambient=[0.0, 0.0, 0.0])
    for nm, v in sorted(rl["view_numeric"].items()):
        spec.add_numeric(name=nm, data=[float(v)])
    spec.add_text(name="filament.cg.tone_mapping", data=str(rl["tone_mapping"]))
    _uv0(spec)
    return spec


class RobotLayer:
    def __init__(self, look, mesh_dir, sun_dir, sun_lux_bake, probes, cam4, win_px):
        _headless()
        if _LIVE:
            raise RobotLayerError("one robot layer per process (Filament holds one model per context); close the "
                                  "previous episode's camera first")
        import mujoco
        import mujoco._render_filament as mjrf
        from mujoco.experimental.studio import window  # noqa: F401  registers Filament's resource providers
        t0 = time.perf_counter()
        self.mj, self.F = mujoco, mjrf
        self.cam = cam4
        rl = look["robot_layer"]
        self.g_amb, self.g_sun = float(rl["G_amb"]), float(rl["G_sun"])
        self.sun_lux = float(rl["sun_lux_per_bake_lux"]) * float(sun_lux_bake)
        self.spec = robot_spec(mesh_dir, look, sun_dir, self.sun_lux)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.ctx = mjrf.Context(mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_VULKAN))
        _LIVE.add(self)
        self.objs = mjrf.ModelObjects(self.ctx, self.model)
        self.scn, self.rends = {}, {}
        for kind in KINDS:
            s = self.ctx.create_scene(mjrf.SceneParams())
            s.configure_from_model(self.model)
            self.scn[kind] = s
            self.rends[kind] = mjrf.ModelRenderables(s, self.objs)
        self.lights = mjrf.ModelLights(self.scn["lit"], self.objs)
        I = float(rl["ibl_intensity"])
        self.ibl = []                                     # per probe: (texture, data, {kind: light})
        for p in probes:
            raw = p["ktx"]
            tc = mjrf.TextureConfig()
            tc.width, tc.height, tc.format = len(raw), 0, mjrf.PixelFormat.PIXEL_FORMAT_KTX
            tc.sampler_type, tc.color_space = mjrf.SamplerType.TEXTURE_CUBE, mjrf.ColorSpace.COLORSPACE_LINEAR
            tex = self.ctx.create_texture(tc)
            td = mjrf.TextureData()
            td.set_image(raw)
            tex.set_data(td)
            lights = {}
            for kind, inten in (("amb", I), ("lit", I * self.g_amb / self.g_sun)):
                lp = mjrf.LightParams()
                lp.type, lp.texture, lp.color = mjrf.LightType.LIGHT_IMAGE, tex, [1.0, 1.0, 1.0]
                lp.intensity = float(inten)
                lights[kind] = self.ctx.create_light(lp)
            self.ibl.append((tex, td, lights))
        self.active = None
        w, h = int(win_px[0]), int(win_px[1])
        self.size = (w, h)
        self.targets = {}
        for kind in KINDS:
            t = self.ctx.create_render_target(mjrf.RenderTargetConfig(color_format=mjrf.PixelFormat.PIXEL_FORMAT_RGB8))
            t.resize(w, h)
            rd = mjrf.ReadPixelsRequest()
            rd.target = t
            rd.alloc(w * h * 3)
            self.targets[kind] = (t, rd)
        self.t_open_s = time.perf_counter() - t0
        # warm-up: both passes once (the robot at its model pose, probe 0). Measured: a layer that never rendered
        # segfaults in its teardown when the process also holds classic (OSMesa) sessions -- an episode whose robot is
        # hidden from its first frame on would end in exit 139; it also moves the shaders' first compile into the open
        t1 = time.perf_counter()
        self.use_probe(0)
        self.set_qpos(self.model.qpos0)
        self.render((0, 0, w, h), KINDS)
        self.t_warm_s = time.perf_counter() - t1

    def use_probe(self, i):
        """The image light of probe i in both scenes (the previous one removed)."""
        if self.active == i:
            return
        for kind, s in self.scn.items():
            if self.active is not None:
                s.remove_light(self.ibl[self.active][2][kind])
            s.add_light(self.ibl[i][2][kind])
        self.active = int(i)

    def set_qpos(self, q):
        mj = self.mj
        self.data.qpos[:] = q
        mj.mj_kinematics(self.model, self.data)
        mj.mj_camlight(self.model, self.data)             # light_xdir (mj_kinematics alone leaves it zero)
        self.lights.update(self.data)
        for r in self.rends.values():
            r.update(self.data)

    def _camera(self, win):
        c = self.cam
        u0, v0, w, h = win
        left, right = (u0 - 0.5 - c["cx"]) / c["fx"] * NEAR, (u0 + w - 0.5 - c["cx"]) / c["fx"] * NEAR
        top, bottom = (c["cy"] - v0 + 0.5) / c["fy"] * NEAR, -(v0 + h - 0.5 - c["cy"]) / c["fy"] * NEAR
        R, t = c["R"], c["t"]
        return self.F.Camera(pos=[float(x) for x in t], forward=[float(x) for x in R[:, 2]],
                             up=[float(x) for x in -R[:, 1]], frustum_center=0.5 * (left + right),
                             frustum_width=0.5 * (right - left), frustum_bottom=bottom, frustum_top=top,
                             frustum_near=NEAR, frustum_far=FAR)

    def render(self, win, kinds):
        """{kind: float32 h x w x 3 linear (the sRGB8 output decoded)} of the window (4x pixels) for the given passes
        (one render call per pass: MuJoCo 3.14's Filament reads back one request per call)."""
        F = self.F
        u0, v0, w, h = (int(v) for v in win)
        if (w, h) != self.size:
            raise RobotLayerError("window %s x %s, the layer renders %s" % (w, h, self.size))
        cam = self._camera((u0, v0, w, h))
        out = {}
        for kind in kinds:
            t, rd = self.targets[kind]
            rq = F.RenderRequest(camera=cam, draw_mode=F.DrawMode.DRAW_MODE_DEFAULT)
            rq.scene, rq.target = self.scn[kind], t
            rq.viewport.left, rq.viewport.bottom, rq.viewport.width, rq.viewport.height = 0, 0, w, h
            self.ctx.wait_for_frame(self.ctx.render([rq], [rd]))
            out[kind] = plate_look.srgb_decode_u8(np.frombuffer(rd.buffer(), np.uint8).reshape(h, w, 3))
        return out

    def close(self):
        """Free the Filament objects in dependency order, then the context, then hand the heap back (malloc_trim)."""
        if self.ctx is None:
            return
        for kind, s in self.scn.items():
            if self.active is not None:
                try:
                    s.remove_light(self.ibl[self.active][2][kind])
                except Exception:  # noqa: BLE001
                    pass
        self.active = None
        self.targets.clear()
        self.rends.clear()
        self.lights = None
        self.scn.clear()
        self.ibl = []
        self.objs = None
        gc.collect()
        self.ctx = None
        self.model = self.data = self.spec = None
        _LIVE.discard(self)
        gc.collect()
        trim_heap()
