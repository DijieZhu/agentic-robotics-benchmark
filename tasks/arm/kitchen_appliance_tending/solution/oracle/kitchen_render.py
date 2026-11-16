"""The kitchen's cameras: MuJoCo 3.16's Filament renderer (physically based) on Vulkan; in the task image Vulkan is
Mesa's lavapipe on the CPU (no GPU). appliance_env.ApplianceEnv.render draws every image through this module, in the
dev sim, the lab and the grader alike.

What is drawn
  * the cell's own robot and unit: the episode's compiled model (a copy whose render-only fields are edited: textures
    flagged sRGB, PBR terms on the untextured materials), visual geom group 1 only, so the environment's floor
    plane, support box, back wall and pedestal (group 0) are not drawn;
  * the apartment around the cell (room.npz / room.json of the room bundle in KITCHEN_RENDER_DIR, default
    /opt/task-assets/payload/render), placed per episode at kitchen_scene.room_pose();
  * the dressing of this unit and episode: kitchen_scene.all_parts() (station, fillers, pedestal, shelf), drawn only;
  * the unit's state, read from the simulator before every image: each burner's flame (the flame site's colour,
    visibility and radius, RoboCasa's Stove.update_state as appliance_env.Unit.update sets them) and the indicator
    geom's colour (lit or INDICATOR_OFF); they are lit emitters here.
The room and the dressing are drawn only: nothing of them is in the simulated model, so the robot touches only what
appliance_env.build_model builds (the pedestal, the unit and its support).
Light: an image light (ibl.ktx of the bundle: a path-traced panorama of the kitchen at the station, daylight only;
the same panorama is the sky behind the windows, sky.ktx) and one Filament light per lamp of kitchen_scene.lights()
(spots, points; a linear LED as a row of spots), photometric: candela from lumens.  The episode's light draw
(appliance_env._appearance, read from the model: the scene light's position, aim, diffuse, the headlight) moves and
aims the key lamp, scales every light and the image light by the strength, tints them by the tint, and adds a
headlight (a directional light along the camera's view).  Every number of the look is in kitchen_look.json beside
this file; the bundle holds the room's meshes, textures and materials and the two panoramas.

Nothing here changes the simulation: the renderer reads the model and the data, it writes neither, and it draws no
random numbers (the sensor noise stays the environment's).

Filament runs in one dedicated thread per process (its engine is thread-bound; a forked child starts its own).
Objects that live for the process: the context, the room (meshes, textures, materials, renderables), the image
light, the lamps, the scene, one render target per image size.  Per episode: the copy of the model and its
ModelResources / ModelRenderables, the dressing.  MuJoCo 3.16's Filament module serves one model's resources at a
time per context, so only the episode rendered last is live; rendering another one first frees the live one's
model resources (the room stays).  The first image of a process takes several seconds (lavapipe compiles the
shaders); later episodes start in about 0.1 s.

    view = open_view(env)          # after env.reset(): builds the episode's drawn model (no Filament work yet)
    rgb = view.render("left")      # uint8 (240, 320, 3), rows top-down, no noise
    view.close()                   # or let it be garbage-collected

Every failure of the renderer (a missing bundle file, a Vulkan error, a call that does not return within
KITCHEN_RENDER_TIMEOUT seconds, default 300) is raised as RenderError: an error of the environment, never of a
controller.  A renderer that failed once stays failed in that process.
"""
import atexit
import copy
import ctypes
import gc
import json
import math
import os
import queue
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
    controller speaks its wire protocol and the practice grader prints its JSON: while the renderer starts, file
    descriptor 1 points at standard error, and C's buffers are flushed before it is given back."""

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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import kitchen_scene  # noqa: E402  (imported now: a process that drops privileges later, the lab, has it already)

RENDER_DIR = os.environ.get("KITCHEN_RENDER_DIR", "/opt/task-assets/payload/render")
LOOK_FILE = os.path.join(HERE, "kitchen_look.json")
BUNDLE_FILES = ("room.npz", "room.json", "ibl.ktx", "sky.ktx")
CALL_TIMEOUT = float(os.environ.get("KITCHEN_RENDER_TIMEOUT", "300"))
CAM_NEAR, CAM_FAR = 0.01, 30.0
CAMERA_NAMES = ("left", "right", "wrist")
NOMINAL_LIGHT_POS = np.array([-0.5, 0.0, 2.6])    # appliance_env.build_model's scene light
NOMINAL_LIGHT_DIFFUSE = 0.8
GEOM_GROUP = 1                                     # robot and unit visual geoms; the env's own scenery is group 0


class RenderError(RuntimeError):
    """The renderer failed: an error of the environment (the dev sim, the lab or the grader), not of a controller."""


def load_look(path=LOOK_FILE):
    """The look's numbers (kitchen_look.json beside this file)."""
    with open(path) as fh:
        return json.load(fh)


try:                                               # read now, for the same reason as kitchen_scene
    LOOK, _LOOK_ERROR = load_look(), None
except (OSError, ValueError) as _exc:
    LOOK, _LOOK_ERROR = None, "cannot read %s: %s" % (LOOK_FILE, _exc)


def srgb_from_linear(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def spot_candela(lumens, half_deg):
    return lumens / (2 * math.pi * (1 - math.cos(math.radians(half_deg))))


# ----------------------------------------------------------------------------------------------------------------
# Per episode, in the caller's thread: what is drawn and how it is lit (plain data, no Filament objects)
# ----------------------------------------------------------------------------------------------------------------
def _drawn_model(m, look):
    """A copy of the episode's model with the render-only fields set for Filament (the physics model is untouched)."""
    d = copy.copy(m)
    roles = d.mat_texid
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    for t in range(d.ntex):
        used_as = set(int(r) for r in np.argwhere(roles == t)[:, 1]) if d.nmat else set()
        if not used_as or rgb_role in used_as:
            d.tex_colorspace[t] = int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)   # Filament reads "auto" as linear
    rules = look["model_materials"]
    for i in range(d.nmat):
        if (d.mat_texid[i] >= 0).any():
            # PBR terms on textured materials crash MuJoCo 3.16's Filament renderer: they keep the legacy (Phong)
            # shading, whose specular RoboCasa sets up to 15 (a mirror under the image light): capped
            d.mat_specular[i] = min(float(d.mat_specular[i]), rules["legacy_specular_max"])
            d.mat_shininess[i] = min(float(d.mat_shininess[i]), rules["legacy_shininess_max"])
            continue
        name = d.mat(i).name
        r = rules["default"]
        for prefix, val in rules["by_prefix"]:
            if name.startswith(prefix):
                r = val
                break
        d.mat_roughness[i] = r["roughness"]
        d.mat_metallic[i] = r["metallic"]
    d.light_intensity[:] = 0.0
    d.light_castshadow[:] = 0
    return d


def _appearance(m, layout, look):
    """The episode's light draw as appliance_env._appearance left it in the model (read only)."""
    diffuse = np.asarray(m.light_diffuse[0], float) / NOMINAL_LIGHT_DIFFUSE
    gain = float(np.mean(diffuse))
    tint = diffuse / max(gain, 1e-6)
    pos = np.asarray(m.light_pos[0], float)
    ldir = np.asarray(m.light_dir[0], float)
    lo, hi, mt = layout["bbox_lo"], layout["bbox_hi"], layout["mount"]
    front = np.array([mt["front_x"], 0.0, mt["bottom_z"] + (hi[2] - lo[2]) / 2])
    t = max(float(np.dot(front - pos, ldir)), 0.5)
    aim = pos + t * ldir                               # the point of the light's axis nearest the unit's front
    return {"gain": gain, "tint": tint.tolist(), "offset": (pos - NOMINAL_LIGHT_POS).tolist(), "aim": aim.tolist(),
            "head_diffuse": float(m.vis.headlight.diffuse[0]), "head_ambient": float(m.vis.headlight.ambient[0])}


def _rig(KS, kind, lo, hi, jitter, look, app):
    """Filament lights of this episode: kitchen_scene.lights() in the cell frame, photometric, varied by the draw."""
    L = look["lamps"]
    key_name = L["key"]
    lights = KS.lights_cell(kind, lo, hi, jitter)
    names = [l["name"] for l in lights]
    if key_name not in names:                          # fall back to the spot nearest above the station
        spots = [l for l in lights if l["type"] == "spot"]
        key_name = min(spots, key=lambda l: math.hypot(l["pos"][0] - 0.6, l["pos"][1]))["name"] if spots else None
    g = app["gain"] * look["exposure_scale"]
    tint = np.asarray(app["tint"])
    out = []
    for l in lights:
        if l["type"] in ("sky", "sun") or l.get("demo_only"):
            continue                                   # daylight is in the image light
        rgb = np.asarray(l["rgb"], float) * tint
        lm = float(l.get("lumens", 0.0)) * L["gain"].get(l["name"], L["gain"].get(l["type"], 1.0))
        pos = np.asarray(l["pos"], float)
        d = np.asarray(l.get("dir", [0, 0, -1.0]), float)
        rec = {"name": l["name"], "color": rgb.tolist(), "shadow": False, "bulb": float(l.get("radius", 0.02)),
               "softness": L["softness"]}
        if l["name"] == key_name:
            pos = pos + np.asarray(app["offset"]) * L["key_offset_scale"]
            aim = np.asarray(app["aim"])
            d = (aim - pos) / np.linalg.norm(aim - pos)
            rec.update(shadow=bool(L["key_shadow"]), bulb=L["key_bulb_m"], shadow_map=int(L["key_shadow_map"]))
        if l["type"] == "point":
            rec.update(type="point", pos=pos.tolist(), dir=[0, 0, -1.0], intensity=g * lm / (4 * math.pi), cutoff=180.0)
            out.append(rec)
        elif l["type"] == "spot":
            half = float(l["spot_deg"]) / 2
            rec.update(type="spot", pos=pos.tolist(), dir=d.tolist(), intensity=g * spot_candela(lm, half), cutoff=half)
            out.append(rec)
        elif l["type"] == "area_line":
            # a Lambertian strip: n spots along it, each lm / n, peak candela lm / (pi n), wide cone
            n = int(L["line_spots"])
            size = l.get("size", [1.0, 0.01])
            T = KS.room_pose(kind, lo, hi, jitter)
            axis = T[:3, :3] @ np.array([1.0, 0.0, 0.0])          # a line light runs along the bundle's x
            for k in range(n):
                s = (k + 0.5) / n - 0.5
                r2 = dict(rec, name="%s_%d" % (l["name"], k))
                r2.update(type="spot", pos=(pos + axis * s * float(size[0])).tolist(), dir=d.tolist(),
                          intensity=g * lm / (math.pi * n), cutoff=float(L["line_cutoff_deg"]))
                out.append(r2)
    return out


class Episode(object):
    """Everything the renderer needs of one episode, built in the caller's thread."""

    def __init__(self, env, look):
        t0 = time.perf_counter()
        KS = kitchen_scene
        m, lay = env.model, env.layout
        kind = env.spec["kind"]
        lo, hi = np.asarray(lay["bbox_lo"], float), np.asarray(lay["bbox_hi"], float)
        jit = tuple(float(x) for x in env.jitter)
        self.live = m                                  # the simulator's model: flames and indicator are read from it
        self.model = _drawn_model(m, look)
        self.room_pose = KS.room_pose(kind, lo, hi, jit)
        self.parts = KS.all_parts(kind, lo, hi, jit)
        self.materials = KS.materials()
        self.app = _appearance(m, lay, look)
        self.lights = _rig(KS, kind, lo, hi, jit, look, self.app)
        ibl = look["ibl"]
        self.ibl_intensity = (ibl["intensity"] * look["exposure_scale"] * self.app["gain"]
                              * (1.0 + look["headlight"]["ambient_to_ibl"] * (self.app["head_ambient"] - 0.125)))
        self.ibl_color = np.clip(np.asarray(self.app["tint"]), 0.0, 2.0).tolist()
        self.head_lux = look["headlight"]["lux_at_0.4"] * self.app["head_diffuse"] / 0.4 * look["exposure_scale"]
        u = env.unit
        self.flames = sorted(u.flames.items())         # [(loc, site id)]
        self.indicator = u.indicator                   # geom id or None
        self.indicator_kind = kind
        self.wrist_cam = env.wrist_cam
        self.fixed_cam = {c: m.camera(c).id for c in ("left", "right")}
        self.seconds = time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------------------------
# Geometry helpers for the drawn-only parts (cell frame)
# ----------------------------------------------------------------------------------------------------------------
def _box_mesh(lo, hi, tile, grain):
    """24 vertices (positions, normals, uvs), 12 triangles; planar uvs in metres / tile per face: the texture's v
    axis follows the wood's grain ('v': z on the vertical faces, 'h': the face's horizontal axis); horizontal faces:
    u along y, v along x."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    grain = grain or "v"
    V, N, U, I = [], [], [], []
    for ax in range(3):
        for s in (-1, 1):
            o = [k for k in range(3) if k != ax]
            n = np.zeros(3)
            n[ax] = s
            quad = []
            for c1, c2 in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = lo.copy()
                p[ax] = hi[ax] if s > 0 else lo[ax]
                p[o[0]] = hi[o[0]] if c1 else lo[o[0]]
                p[o[1]] = hi[o[1]] if c2 else lo[o[1]]
                quad.append(p)
            if np.dot(np.cross(quad[1] - quad[0], quad[2] - quad[0]), n) < 0:   # counter-clockwise from outside
                quad = quad[::-1]
            base = len(V)
            for p in quad:
                if ax == 2:
                    uu, vv = p[1], p[0]
                else:
                    horiz = o[0] if o[0] != 2 else o[1]
                    uu, vv = (p[horiz], p[2]) if grain == "v" else (p[2], p[horiz])
                V.append(p)
                N.append(n)
                U.append([uu / tile, vv / tile])
            I += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(V), np.array(N), np.array(U), np.array(I)


def _cyl_mesh(c, r, hl, axis, seg=16):
    k = "xyz".index(axis)
    a1, a2 = [i for i in range(3) if i != k]
    V, N, U, I = [], [], [], []
    for j in range(seg + 1):
        th = 2 * math.pi * j / seg
        for s in (-1, 1):
            p = np.array(c, float)
            p[a1] += r * math.cos(th)
            p[a2] += r * math.sin(th)
            p[k] += s * hl
            n = np.zeros(3)
            n[a1], n[a2] = math.cos(th), math.sin(th)
            V.append(p); N.append(n); U.append([j / seg, (s + 1) / 2])
    for j in range(seg):
        b = 2 * j
        I += [b, b + 2, b + 1, b + 1, b + 2, b + 3]
    for s in (-1, 1):                                   # caps
        base = len(V)
        n = np.zeros(3)
        n[k] = s
        p = np.array(c, float)
        p[k] += s * hl
        V.append(p.copy()); N.append(n); U.append([0.5, 0.5])
        for j in range(seg):
            th = 2 * math.pi * j / seg
            q = p.copy()
            q[a1] += r * math.cos(th)
            q[a2] += r * math.sin(th)
            V.append(q); N.append(n); U.append([0.5, 0.5])
        for j in range(seg):
            i0, i1 = base + 1 + j, base + 1 + (j + 1) % seg
            I += [base, i0, i1] if s > 0 else [base, i1, i0]
    V, N, U, I = np.array(V), np.array(N), np.array(U), np.array(I)
    # fix the side winding to face outwards
    t = I[:3]
    if np.dot(np.cross(V[t[1]] - V[t[0]], V[t[2]] - V[t[0]]), N[t[0]]) < 0:
        side = 6 * seg
        I[:side] = I[:side].reshape(-1, 3)[:, ::-1].ravel()
    return V, N, U, I


# ----------------------------------------------------------------------------------------------------------------
# The renderer (one per process, its own thread)
# ----------------------------------------------------------------------------------------------------------------
class View(object):
    """One episode's cameras; the Filament objects stay in the renderer's thread."""

    def __init__(self, env, renderer=None):
        self._fil = renderer or _Filament.instance()
        self.env_ref = weakref.ref(env)
        try:
            self.episode = Episode(env, self._fil.look)
        except Exception as exc:  # noqa: BLE001 - whatever breaks here is the renderer's, not the caller's
            raise RenderError("renderer: building the episode's drawn model: %s: %s" % (type(exc).__name__, exc))
        self.build_s = self.episode.seconds
        self._sid = self._fil.call(self._fil.new_session, self.episode)
        self._finalizer = weakref.finalize(self, self._fil.post, self._fil.free_session, self._sid)

    def render(self, cam, data=None):
        try:
            env = self.env_ref()
            d = env.data if data is None else data
            m = env.model
            ep = self.episode
            if cam == "wrist":
                cid, W, H = ep.wrist_cam, 256, 192
            else:
                cid = ep.fixed_cam[cam]
                W, H = 320, 240
            R = d.cam_xmat[cid].reshape(3, 3)
            args = (self._sid, d, d.cam_xpos[cid].copy(), (-R[:, 2]).copy(), R[:, 1].copy(), float(m.cam_fovy[cid]),
                    W, H)
        except Exception as exc:  # noqa: BLE001
            raise RenderError("renderer: camera %r: %s: %s" % (cam, type(exc).__name__, exc))
        return self._fil.call(self._fil.draw, *args)

    def close(self):
        if self._finalizer.detach() is not None and not self._fil.broken:
            self._fil.call(self._fil.free_session, self._sid)


def open_view(env):
    """The cameras of `env`'s current episode (after env.reset()). Raises RenderError."""
    return View(env)


class _Filament(object):
    _inst = None
    _failed = None                                     # (pid, message) of a renderer that could not start
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
        self.broken = None                             # why the renderer stopped answering, once it did
        if LOOK is None:
            raise RenderError("renderer: " + _LOOK_ERROR)
        self.look = LOOK
        missing = [f for f in BUNDLE_FILES if not os.path.isfile(os.path.join(RENDER_DIR, f))]
        if missing:
            raise RenderError("renderer: the room bundle in %s lacks %s (KITCHEN_RENDER_DIR)"
                              % (RENDER_DIR, ", ".join(missing)))
        self.jobs = queue.Queue()
        self.sessions = {}
        self.active = None
        self.live = {}
        self.targets = {}
        self.next_sid = 0
        self.stats = {"activations": 0, "frames": 0, "targets_created": 0}
        self.thread = threading.Thread(target=self._loop, name="kitchen-filament", daemon=True)
        self.thread.start()
        try:
            self.call(self._start)
        except RenderError:
            self.post(None)                            # the thread frees what _start made, then ends
            raise

    # -- any thread
    def call(self, fn, *args):
        if self.broken:
            raise RenderError(self.broken)
        done = threading.Event()
        out = []
        self.jobs.put((fn, args, done, out))
        if not done.wait(CALL_TIMEOUT):
            self.broken = "renderer: a call did not return within %.0f s (KITCHEN_RENDER_TIMEOUT)" % CALL_TIMEOUT
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
                except BaseException:  # noqa: BLE001 - after a failed start some objects were never made
                    pass
                return
            try:
                res = (True, fn(*args))
            except BaseException as exc:  # noqa: BLE001 - handed over without its traceback (it holds Filament objects)
                res = (False, RenderError("renderer: %s: %s" % (type(exc).__name__, exc)))
            if done is not None:
                out.append(res)
                done.set()

    # -- the renderer's thread only
    def _start(self):
        with _StdoutToStderr():
            self._start_quiet()

    def _start_quiet(self):
        t0 = time.perf_counter()
        self.room, self.lamps = [], {}
        for k in ("DISPLAY", "WAYLAND_DISPLAY"):       # Filament would connect to a named display (and may hang)
            os.environ.pop(k, None)
        if self.look["render"].get("lavapipe_threads"):  # read by lavapipe when the Vulkan device is created
            os.environ["LP_NUM_THREADS"] = str(int(self.look["render"]["lavapipe_threads"]))
        import mujoco._render_filament as F
        from mujoco.experimental.studio import window  # noqa: F401  registers Filament's resource providers
        self.F = F
        look = self.look
        self.ctx = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN))
        self.ctx.set_clear_color([0.0, 0.0, 0.0])
        self.ibl_tex = self._ktx(look["ibl"]["file"], F.SamplerType.TEXTURE_CUBE)
        self.sky_tex = self._ktx(look["ibl"]["sky_file"], F.SamplerType.TEXTURE_SKYBOX) if look["ibl"].get("sky_file") else None
        # the look's view options travel as custom fields of a tiny model (MuJoCo reads filament.* from them)
        spec = mujoco.MjSpec()
        for k, v in sorted(look["filament_numeric"].items()):
            spec.add_numeric(name=k, data=[float(v)])
        for k, v in sorted(look["filament_text"].items()):
            spec.add_text(name=k, data=str(v))
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], pos=[0, 0, -50])
        self.look_model = spec.compile()
        self.scn = self.ctx.create_scene(F.SceneParams())
        self.scn.configure_from_model(self.look_model)
        lp = F.LightParams()
        lp.type, lp.texture, lp.color, lp.intensity = F.LightType.LIGHT_IMAGE, self.ibl_tex, [1.0, 1.0, 1.0], 1.0
        self.ibl = self.ctx.create_light(lp)
        self.scn.add_light(self.ibl)
        if self.sky_tex is not None:
            self.scn.set_skybox(self.sky_tex)
        if int(look["render"].get("msaa", 0)) > 1:
            self.scn.set_msaa(int(look["render"]["msaa"]))
        lp = F.LightParams()
        lp.type, lp.color, lp.intensity, lp.cast_shadow = F.LightType.LIGHT_DIRECTIONAL, [1.0, 1.0, 1.0], 0.0, False
        self.head = self.ctx.create_light(lp)
        self.scn.add_light(self.head)
        self.lamps = {}
        self._load_room()
        self._unit_meshes()
        self.start_s = time.perf_counter() - t0

    def _ktx(self, name, sampler):
        F = self.F
        path = os.path.join(RENDER_DIR, name)
        if not os.path.isfile(path):
            raise RenderError("render asset missing: " + path)
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

    def _mesh(self, V, N, U, I):
        F = self.F
        A, Us, Ty = F.VertexAttribute, F.VertexAttributeUsage, F.VertexAttributeType
        attrs = [A(usage=Us.VERTEX_ATTRIBUTE_USAGE_POSITION, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                 A(usage=Us.VERTEX_ATTRIBUTE_USAGE_NORMAL, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                 A(usage=Us.VERTEX_ATTRIBUTE_USAGE_UV, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT2)]
        cfg = F.MeshConfig(max_vertices=len(V), max_indices=len(I), attributes=attrs, interleaved=False,
                           index_type=F.IndexType.INDEX_TYPE_U32,
                           primitive_type=F.MeshPrimitiveType.MESH_PRIMITIVE_TYPE_TRIANGLES)
        mesh = self.ctx.create_mesh(cfg)
        md = F.MeshData(compute_bounds=True)
        md.set_vertices(cfg, np.concatenate([np.asarray(V, np.float32).ravel(), np.asarray(N, np.float32).ravel(),
                                             np.asarray(U, np.float32).ravel()]).tobytes())
        md.set_indices(cfg, np.ascontiguousarray(I, np.uint32).tobytes())
        mesh.upload(md)
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

    def _load_room(self):
        F = self.F
        with open(os.path.join(RENDER_DIR, "room.json")) as fh:
            meta = json.load(fh)
        z = np.load(os.path.join(RENDER_DIR, "room.npz"))
        V, N, U, I, tex = z["V"], z["N"], z["UV"], z["I"], z["tex"]
        tex_ofs, tw, th = z["tex_ofs"], z["tex_w"], z["tex_h"]
        texs = [self._texture(tex[o:o + w * h * 3].reshape(h, w, 3)) for o, w, h in zip(tex_ofs, tw, th)]
        self.room_textures = texs
        self.dressing_tex = {k: texs[i] for k, i in meta.get("dressing_textures", {}).items()}
        self.room = []
        rl = self.look["room"]
        for mm in meta["materials"]:
            if mm["icnt"] == 0:
                continue
            v0, vc, i0, ic = mm["vofs"], mm["vcnt"], mm["iofs"], mm["icnt"]
            mesh = self._mesh(V[v0:v0 + vc], N[v0:v0 + vc], U[v0:v0 + vc], I[i0:i0 + ic])
            rgba = np.array(mm["rgba"], np.float32)
            g = rl["albedo_gain"]
            rgba[:3] = srgb_from_linear(np.power(np.clip(rgba[:3], 0, 1), 2.2) * g) if g != 1.0 else rgba[:3]
            mat = F.Material(color=rgba, roughness=float(np.clip(mm["roughness"] * rl["roughness_scale"], 0.05, 1.0)),
                             metallic=float(mm["metallic"]) * rl["metallic_scale"],
                             emissive=float(mm["emissive"]) * rl["emissive"],
                             color_texture=texs[mm["tex"]] if mm["tex"] >= 0 else None)
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadow=bool(rl["cast_shadow"]), receive_shadow=True))
            r.set_mesh(mesh, 0, ic)
            r.set_material(mat)
            self.scn.add_renderable(r)
            self.room.append((mesh, mat, r))
        self.room_stats = meta["stats"]
        self.room_pose = None

    def _unit_meshes(self):
        V, N, U, I = _cyl_mesh([0, 0, 0], 1.0, 1.0, "z", 32)
        self.flame_mesh = (self._mesh(V, N, U, I), len(I))
        V, N, U, I = _box_mesh([-1, -1, -1], [1, 1, 1], 1.0, None)
        self.box_mesh = (self._mesh(V, N, U, I), len(I))

    # -- sessions
    def new_session(self, ep):
        sid = self.next_sid
        self.next_sid += 1
        self.sessions[sid] = ep
        return sid

    def free_session(self, sid):
        self.sessions.pop(sid, None)
        if self.active == sid:
            self._evict()

    def _evict(self):
        L = self.live
        for r in L.get("dressing", []) + [x[0] for x in L.get("flames", []) if x[2]]:
            self.scn.remove_renderable(r[0] if isinstance(r, tuple) else r)
        if L.get("indicator") is not None and L["indicator"][2]:
            self.scn.remove_renderable(L["indicator"][0])
        for k in ("rends", "res"):                      # renderables before the resources they draw
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
        ep = self.sessions[sid]
        self._evict()
        look = self.look
        res = F.ModelResources(self.ctx, ep.model)
        rends = F.ModelRenderables(self.scn, res)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[GEOM_GROUP] = 1
        opt.sitegroup[:] = 0
        rends.set_options(opt)
        L = {"res": res, "rends": rends, "dressing": [], "keep": []}
        # the room at this episode's pose
        T = ep.room_pose
        if self.room_pose is None or not np.allclose(self.room_pose, T):
            p, R = T[:3, 3].astype(np.float32), T[:3, :3].astype(np.float32).reshape(9)
            for _m, _mat, r in self.room:
                r.set_pose(p, R)
            self.room_pose = T.copy()
        # the dressing: one mesh per material
        groups = {}
        for part in ep.parts:
            groups.setdefault(part["material"], []).append(part)
        dl = look["dressing"]
        for mname, parts in sorted(groups.items()):
            mdef = ep.materials.get(mname, {"base": [0.7, 0.7, 0.7], "roughness": 0.5, "metallic": 0.0})
            tile = float(mdef.get("tile_m", 1.0))
            VV, NN, UU, II = [], [], [], []
            n0 = 0
            for part in parts:
                if part["shape"] == "box":
                    c, s = np.asarray(part["center"]), np.asarray(part["size"]) / 2
                    V, N, U, I = _box_mesh(c - s, c + s, tile, part.get("grain"))
                else:
                    V, N, U, I = _cyl_mesh(part["center"], part["size"][0], part["size"][1], part["axis"], 12)
                VV.append(V); NN.append(N); UU.append(U); II.append(I + n0)
                n0 += len(V)
            V, N, U, I = np.concatenate(VV), np.concatenate(NN), np.concatenate(UU), np.concatenate(II)
            mesh = self._mesh(V, N, U, I)
            tex = self.dressing_tex.get(mname)
            base = np.asarray(mdef.get("base", [0.7, 0.7, 0.7]), float) * dl["albedo_gain"]
            rgba = np.concatenate([srgb_from_linear(base), [1.0]]).astype(np.float32)
            emissive = 0.0
            if mdef.get("emission_K"):
                emissive = dl["emitter_emissive"]
            mat = F.Material(color=rgba, roughness=float(mdef.get("roughness", 0.5)), metallic=float(mdef.get("metallic", 0.0)),
                             emissive=emissive, color_texture=tex)
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadow=bool(dl["cast_shadow"]), receive_shadow=True))
            r.set_mesh(mesh, 0, len(I))
            r.set_material(mat)
            r.set_pose(np.zeros(3, np.float32), np.eye(3, dtype=np.float32).reshape(9))
            self.scn.add_renderable(r)
            L["dressing"].append(r)
            L["keep"].append((mesh, mat))
        # the unit's state: flames and indicator (added to the scene while lit)
        L["flames"] = []
        for loc, site in ep.flames:
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadow=False, receive_shadow=False))
            r.set_mesh(self.flame_mesh[0], 0, self.flame_mesh[1])
            L["flames"].append([r, site, False, None, None])        # renderable, site, shown, material, its colour
        L["indicator"] = None
        if ep.indicator is not None:
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadow=False, receive_shadow=True))
            r.set_mesh(self.box_mesh[0], 0, self.box_mesh[1])
            L["indicator"] = [r, ep.indicator, False, None, None]
        # lamps of this episode
        self._lamps(ep)
        self.ibl.set_intensity(float(ep.ibl_intensity))
        self.ibl.set_color(np.asarray(ep.ibl_color, np.float32))
        self.live = L
        self.active = sid
        self.stats["activations"] += 1
        self.last_activation_s = time.perf_counter() - t0

    def _lamps(self, ep):
        F = self.F
        want = {l["name"]: l for l in ep.lights}
        for name in list(self.lamps):
            if name not in want:
                self.scn.remove_light(self.lamps.pop(name)[0])
        for name, l in want.items():
            sig = (l["type"], l["shadow"], l.get("shadow_map", 0))
            if name in self.lamps and self.lamps[name][1] != sig:
                self.scn.remove_light(self.lamps.pop(name)[0])
            if name not in self.lamps:
                lp = F.LightParams()
                lp.type = F.LightType.LIGHT_SPOT if l["type"] == "spot" else F.LightType.LIGHT_POINT
                lp.color, lp.intensity = [1.0, 1.0, 1.0], 0.0
                lp.cast_shadow = bool(l["shadow"])
                if l["shadow"]:
                    lp.shadow_map_size = int(l.get("shadow_map", 512))
                lp.bulb_radius = float(l["bulb"])
                lp.cutoff = float(l["cutoff"])
                lp.softness = float(l["softness"])
                light = self.ctx.create_light(lp)
                self.scn.add_light(light)
                self.lamps[name] = (light, sig)
            light = self.lamps[name][0]
            light.set_pose(np.asarray(l["pos"], np.float32), np.asarray(l["dir"], np.float32))
            light.set_color(np.asarray(l["color"], np.float32))
            light.set_intensity(float(l["intensity"]))
            light.set_cutoff(float(l["cutoff"]))
            light.set_bulb_radius(float(l["bulb"]))

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

    def _state(self, ep, data):
        """Flames and indicator from the simulator's model (colour, visibility, size) and data (pose)."""
        F = self.F
        m = ep.live
        look = self.look
        fl = look["flame"]
        for rec in self.live["flames"]:
            r, sid, shown, _mat, last = rec
            a = float(m.site_rgba[sid, 3])
            if a > 0.0:
                rgb = tuple(np.round(m.site_rgba[sid, :3], 4))
                if rgb != last:                     # the site's own colour (RoboCasa: gas blue, electric red), lit
                    c = np.asarray(rgb, float) / max(float(np.max(rgb)), 1e-6)
                    c = (1.0 - fl.get("white_mix", 0.0)) * c + fl.get("white_mix", 0.0)   # a hotter core
                    mat = F.Material(color=np.array(list(c) + [fl["alpha"]], np.float32), roughness=1.0, metallic=0.0,
                                     emissive=float(fl["emissive"]))
                    r.set_material(mat)
                    rec[3], rec[4] = mat, rgb
                rad, hl = float(m.site_size[sid, 0]), float(m.site_size[sid, 1])
                R = data.site_xmat[sid].reshape(3, 3) @ np.diag([rad, rad, max(hl, 1e-4) * look["flame"]["height_scale"]])
                p = data.site_xpos[sid] + data.site_xmat[sid].reshape(3, 3)[:, 2] * look["flame"]["lift_m"]
                r.set_pose(p.astype(np.float32), R.astype(np.float32).reshape(9))
                if not shown:
                    self.scn.add_renderable(r)
                    rec[2] = True
            elif shown:
                self.scn.remove_renderable(r)
                rec[2] = False
        ind = self.live["indicator"]
        if ind is not None:
            r, gid, shown, last, _mat = ind
            rgba = tuple(np.round(m.geom_rgba[gid], 4))
            if rgba != last:
                il = look["indicator"]
                lit = float(np.max(rgba[:3])) > 0.5
                col = np.array(list(rgba[:3]) + [1.0], np.float32)
                if not lit:
                    col[:3] = il["off_srgb"]
                mat = F.Material(color=col, roughness=il["roughness"], metallic=0.0,
                                 emissive=float(il["emissive_on"] if lit else 0.0))
                r.set_material(mat)
                ind[3], ind[4] = rgba, mat
            s = m.geom_size[gid]
            R = data.geom_xmat[gid].reshape(3, 3) @ np.diag([s[0], max(s[1], look["indicator"]["min_half_depth_m"]), s[2]])
            r.set_pose(data.geom_xpos[gid].astype(np.float32), R.astype(np.float32).reshape(9))
            if not shown:
                self.scn.add_renderable(r)
                ind[2] = True

    def draw(self, sid, data, pos, forward, up, fovy, width, height):
        F = self.F
        self._activate(sid)
        ep = self.sessions[sid]
        L = self.live
        L["rends"].update(data)
        self._state(ep, data)
        if ep.head_lux > 0:
            self.head.set_pose(np.asarray(pos, np.float32), np.asarray(forward, np.float32))
            self.head.set_intensity(float(ep.head_lux))
            self.head.set_color(np.asarray(ep.ibl_color, np.float32))
        tgt, read, buf = self._target(width, height)
        top = CAM_NEAR * math.tan(math.radians(fovy) / 2.0)
        cam = F.Camera(pos=np.asarray(pos, np.float32), forward=np.asarray(forward, np.float32),
                       up=np.asarray(up, np.float32), frustum_center=0.0, frustum_width=top * width / height,
                       frustum_bottom=-top, frustum_top=top, frustum_near=CAM_NEAR, frustum_far=CAM_FAR)
        rv = self.look["render"]
        req = F.RenderRequest(camera=cam, draw_mode=F.DrawMode.DRAW_MODE_DEFAULT, post_processing=bool(rv["post_processing"]),
                              shadows=bool(rv["shadows"]), reflections=bool(rv.get("reflections", False)))
        req.scene, req.target = self.scn, tgt
        req.viewport.left, req.viewport.bottom, req.viewport.width, req.viewport.height = 0, 0, width, height
        self.ctx.wait_for_frame(self.ctx.render([req], [read]))
        self.stats["frames"] += 1
        return buf.copy()

    def info(self):
        return {"stats": dict(self.stats), "start_s": self.start_s, "room": self.room_stats,
                "last_activation_s": getattr(self, "last_activation_s", None), "targets": len(self.targets),
                "lamps": len(self.lamps)}

    def _shutdown(self):
        """Every Filament object before the context (destroying the context first panics); also after a start that
        failed half-way (whatever was made is freed here, in the renderer's thread)."""
        self.sessions.clear()
        if getattr(self, "scn", None) is not None:
            try:
                self._evict()
            except Exception:  # noqa: BLE001
                self.live = {}
            for name in list(getattr(self, "lamps", {})):
                try:
                    self.scn.remove_light(self.lamps.pop(name)[0])
                except Exception:  # noqa: BLE001
                    pass
        self.live = {}
        self.targets.clear()
        self.lamps = {}
        self.room = []
        self.room_textures = []
        self.dressing_tex = {}
        self.flame_mesh = self.box_mesh = None
        self.head = self.ibl = None
        self.scn = None
        self.look_model = None
        self.ibl_tex = self.sky_tex = None
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
