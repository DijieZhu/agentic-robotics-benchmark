"""The bench's cameras: MuJoCo 3.16's Filament renderer (physically based) on Vulkan; in the task's images Vulkan is
Mesa's lavapipe on the CPU (no GPU). lab_sim.render draws every image through this module, in the commissioning
service and in the grader alike.

What is drawn
  * the episode's own bench: a copy of the episode's compiled model whose render-only fields are edited (one material
    per kind of part, in the part's own colour: the centrifuge's cabinet, the yellow shield, the rotor and its hub,
    the bores, the green mark, the knob, the lid, the racks' rings, the tubes and their caps, the scale and its
    holder), in the simulator's state at the moment of the image. Shapes, sizes, places and colours are the model's:
    nothing is added to the bench and nothing is taken from it;
  * the robot: the model's visual meshes of the arm, the hand and the fingers (its collision shapes are not drawn,
    except the fingers' rubber pads) where the simulator's state has them, as renderables that the process keeps
    (robot_model says why). The wrist camera sees the underside of the hand it is mounted in along its lower edge
    and the two fingers, where MuJoCo's own renderer showed the hand's collision box;
  * the room around the bench (room.npz / room.json of the render bundle in LAB_RENDER_DIR, default
    /opt/task-assets/payload/render): a laboratory's wall benches, cabinets and floor, and the bench's own worktop
    and cabinet. Drawn only: nothing of it is in the simulated model. The model's floor plane and table block are
    not drawn: the bundle's floor and worktop stand exactly where they are.
Light: an image light (ibl.ktx of the bundle: a path-traced probe of the laboratory over the bench, the robot and
the bench's parts taken out, the bay's ceiling luminaires painted in) and two spot lights at the bay's two ceiling
luminaires (lab_look.json: a lamp casts a shadow only when the look gives it a shadow map; the task's look gives
none: the luminaires are long, their shadows soft); ambient occlusion shades what the image light cannot (a bore,
the gap under a part). The cameras' exposure (the look's exposure_scale) is fixed.
Cameras: exactly the simulated cameras. Position and axes are the ones MuJoCo computes for the episode's model
(mjv_updateCamera) and the frustum is the camera's vertical field of view at the image's aspect ratio, as MuJoCo's
own renderer drew it: the published intrinsics and poses hold as before. An image is drawn at a multiple of its
size and averaged over the blocks in linear light (the look's render.supersample, or render.supersample_by_camera
for a camera named there).

An image is a function of the model, the simulator's state and the camera alone: the same camera in the same state
gives the same bytes however often it is drawn, whatever was drawn before it, however many other episodes the
process has drawn and in whichever process of the same image (Filament's temporal dithering, a new noise per frame,
is off: _undither; no two drawn boxes share a face's plane and a ring's wall is one mesh, not its crossing box
segments: _drawn_model; the robot's shells are drawn in one order: robot_model). The sensor noise stays
lab_sim.render's.

Filament runs in one dedicated thread per process (its engine is thread-bound; a forked child starts its own).
Objects that live for the process: the context, the room and the robot (meshes, textures, materials, renderables),
the image light, the lamps, the scene, one render target per image size. Per episode: the copy of the model and its
ModelResources / ModelRenderables (the bench's parts). MuJoCo 3.16's Filament module serves one model's resources at a time per
context, so only the episode rendered last is live; rendering another one first frees the live one's model
resources (the room stays). The first image of a process takes several seconds (lavapipe compiles the shaders).

    view = open_view(model)                        # builds the episode's drawn model (no Filament work yet)
    rgb = view.render("left", data, 320, 240)      # uint8 (240, 320, 3), rows top-down, no noise
    view.close()                                   # or let it be garbage-collected

Every failure of the renderer that Python sees (a missing or broken file of the render bundle, an error while an
episode is built or an image drawn, a call that does not return within LAB_RENDER_TIMEOUT seconds, default 300) is
raised as RenderError: an error of the environment, never of a controller. A renderer that failed once stays failed
in that process. A Vulkan backend that cannot start at all (no usable driver) aborts the whole process inside
Filament before Python can catch anything; the task images' builds draw the four cameras through this module for
every account that renders, so each image proves its driver.

Nothing here changes the simulation: the renderer reads the model and the data and writes neither.
"""
import atexit
import copy
import ctypes
import gc
import hashlib
import json
import os
import queue
import re
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
RENDER_DIR = os.environ.get("LAB_RENDER_DIR", "/opt/task-assets/payload/render")
LOOK_FILE = os.path.join(HERE, "lab_look.json")
BUNDLE_FILES = ("room.npz", "room.json", "ibl.ktx")
CALL_TIMEOUT = float(os.environ.get("LAB_RENDER_TIMEOUT", "300"))
CAM_FAR = 40.0                 # the room's far corner; the near plane is the simulated camera's own
GEOM_GROUP = 1                 # what the cameras show of the episode's model, after _drawn_model regroups it
HIDDEN_GROUP = 4
SCENERY = ("table",)             # and every plane: the bundle draws the floor and the bench
ROBOT_PARTS = ("robot", "hand", "finger", "pad")   # drawn by the process's renderables (robot_model), not the episode's


class RenderError(RuntimeError):
    """The renderer failed: an error of the environment (the commissioning service or the grader), not of a
    controller."""


def load_look(path=LOOK_FILE):
    """The look's numbers (lab_look.json beside this file)."""
    with open(path) as fh:
        return json.load(fh)


try:
    LOOK, _LOOK_ERROR = load_look(), None
except (OSError, ValueError) as _exc:
    LOOK, _LOOK_ERROR = None, "cannot read %s: %s" % (LOOK_FILE, _exc)


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


# ----------------------------------------------------------------------------------------------------------------
# Per episode, in the caller's thread: what is drawn and how it is lit (plain data, no Filament objects)
# ----------------------------------------------------------------------------------------------------------------
PARTS = ("cabinet", "shield", "rotor", "hub", "bore", "mark", "knob", "lid", "lid_handle", "ring", "ring_floor",
         "tube", "cap", "liquid", "scale", "holder", "hand", "finger", "pad", "other")
_BORE = re.compile(r"^h\d+_(w\d+|wall|floor)$")
_SEGMENT = re.compile(r"^(.+)_w\d+$")              # a ring's wall segment: not drawn where the ring has its `_wall` mesh


def part_of_geom(m, g):
    """Which drawn part a geom of the episode's model is: one of PARTS (a kind of part that has a look_<part>
    material of its own), "robot" (a visual mesh of the arm, drawn with its own material), or None (not drawn: the
    robot's collision shapes, the model's floor plane and table block, where the bundle's floor and worktop are)."""
    body = m.body(m.geom_bodyid[g]).name
    name = m.geom(g).name or ""
    if body.startswith("r_"):
        if name.endswith("_pad_collision"):
            return "pad"                                # the fingers' rubber pads: what touches a tube
        if int(m.geom_group[g]) != 1:
            return None                                 # a collision shape
        if name == "r_g_hand_visual":
            return "hand"
        if name.startswith("r_g_finger") and name.endswith("_visual"):
            return "finger"
        return "robot"
    if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE) or name in SCENERY:
        return None
    if name == "cabinet" or name.startswith("wall"):
        return "cabinet"
    if name.startswith("shield"):
        return "shield"
    if name == "rotor_shell":
        return "rotor"
    if name in ("hub", "mark", "knob"):
        return name
    if name == "lid_plate":
        return "lid"
    if name.startswith("lid_"):
        return "lid_handle"
    if name == "scale_base":
        return "scale"
    if _BORE.match(name):
        return "bore"
    if name.endswith("_floor"):
        return "ring_floor"                             # the grey floors of the racks' rings and of the scale's holder
    if name.startswith("scale_holder"):
        return "holder"
    if name.endswith("_cap"):
        return "cap"
    if name.endswith("_body"):
        return "tube"
    if name.endswith("_liquid"):
        return "liquid"
    if "_w" in name:
        return "ring"
    return "other"


def _drawn_model(m, look):
    """A copy of the episode's model with the render-only fields set for Filament (the physics model is untouched):
    the bench's drawn geoms in group GEOM_GROUP and everything else in HIDDEN_GROUP (the robot too: robot_model);
    every part of the bench on the material lab_sim.build added for its kind (look_<part>), in the geoms' own colour
    and opacity (the lid's plate stays as see-through as the model has it); PBR terms from the look. All geoms of one
    kind have one colour in a model; a model in which they do not is refused. Boxes are drawn a hair smaller (below)."""
    d = copy.copy(m)
    rules = look["model_materials"]
    for t in range(d.ntex):
        d.tex_colorspace[t] = int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)        # Filament reads "auto" as linear
    part_mat = {}
    walls = {m.geom(g).name[:-5] for g in range(m.ngeom) if (m.geom(g).name or "").endswith("_wall")}
    merged = tuple(m.geom(g).name[:-6] for g in range(m.ngeom) if (m.geom(g).name or "").endswith("_drawn"))
    for g in range(d.ngeom):
        part = part_of_geom(m, g)
        seg = _SEGMENT.match(m.geom(g).name or "")
        box = int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        if part is None or part in ROBOT_PARTS or (seg and seg.group(1) in walls) or (
                box and merged and (m.geom(g).name or "").startswith(merged)):
            # the robot: the process's renderables (robot_model); a ring's wall segments, the shield's segments and
            # the cabinet's walls: one mesh is drawn in their place (lab_sim.add_ring_walls, add_drawn_boxes: the
            # boxes pass through each other)
            d.geom_group[g] = HIDDEN_GROUP
            continue
        d.geom_group[g] = GEOM_GROUP
        r = rules[part]
        own = tuple(float(x) for x in m.geom_rgba[g])
        if part not in part_mat:
            mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "look_" + part)
            if mid < 0:
                raise RenderError("renderer: the model has no material %r (lab_sim.build adds them)" % ("look_" + part))
            d.mat_rgba[mid, :3] = _gain_srgb(r.get("srgb", own), r.get("gain", 1.0))
            d.mat_rgba[mid, 3] = own[3]
            d.mat_roughness[mid] = r["roughness"]
            d.mat_metallic[mid] = r["metallic"]
            d.mat_specular[mid] = r.get("specular", 0.5)
            part_mat[part] = (mid, own)
        elif "srgb" not in r and own != part_mat[part][1]:
            raise RenderError("renderer: two colours among the model's %r geoms" % part)
        d.geom_matid[g] = part_mat[part][0]
        d.geom_rgba[g] = [0.5, 0.5, 0.5, 1.0]           # MuJoCo's default: the material's colour is the geom's
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            # The bench's rings, shield and walls are sets of boxes whose faces lie in one plane where they meet.
            # Which of two such faces a pixel shows is the renderer's draw order, and that differs from process to
            # process (a few dozen pixels by 1 to 5 levels). In the drawn copy every box is 0.02 to 0.26 mm smaller
            # (half sizes), a different amount for neighbours: no two faces share a plane, the picture is the same.
            d.geom_size[g] = m.geom_size[g] - 2e-5 * (1 + g % 13)
    d.light_intensity[:] = 0.0
    d.light_castshadow[:] = 0
    return d


def robot_geoms(m, look):
    """The robot's drawn geoms (the visual meshes of the arm, the hand and the fingers; of the collision shapes only
    the fingers' rubber pads): (their indices in the model, in the model's order; a signature). The signature is the
    same for two models whose robot is the same, geom for geom in that order, wherever in the model it stands (a
    bench with other parts before the robot has it at other indices): every episode of a process must have the
    robot of the first, whose renderables the process keeps (robot_model)."""
    idx, sig = [], hashlib.sha256()
    meshes = {}
    for g in range(m.ngeom):
        part = part_of_geom(m, g)
        if part not in ROBOT_PARTS:
            continue
        idx.append(g)
        mid, j = int(m.geom_matid[g]), int(m.geom_dataid[g])
        if j >= 0 and j not in meshes:
            va, fa = int(m.mesh_vertadr[j]), int(m.mesh_faceadr[j])
            meshes[j] = hashlib.sha256(m.mesh_vert[va:va + int(m.mesh_vertnum[j])].tobytes()
                                       + m.mesh_face[fa:fa + int(m.mesh_facenum[j])].tobytes()).hexdigest()
        sig.update(repr((part, m.body(m.geom_bodyid[g]).name, int(m.geom_type[g]), meshes.get(j), m.geom_size[g].tolist(),
                         m.geom_rgba[g].tolist(), m.geom_pos[g].tolist(), m.geom_quat[g].tolist(),
                         (m.mat_rgba[mid].tolist(), float(m.mat_shininess[mid])) if mid >= 0 else None)).encode())
    sig.update(json.dumps(look["model_materials"], sort_keys=True).encode())
    return np.asarray(idx, np.intp), sig.hexdigest()


def robot_model(m, look):
    """A copy of an episode's model in which only the robot is drawn, with the render-only fields set for Filament
    (the episode's model is untouched): the robot's drawn geoms in group GEOM_GROUP, everything else in
    HIDDEN_GROUP; the hand, the fingers and their pads on the materials lab_sim.build added for them (look_hand,
    look_finger, look_pad); the arm's shells on their own materials, with the look's PBR terms by the material's
    colour (white covers, dark joints, the rest).

    The process keeps the copy of its first episode and MuJoCo's renderables of it for good, and every image moves
    them to the state's poses (_Filament.draw). The reason is the order in which Filament draws: the shells of the
    arm pass through each other at the joints, and at a sample where two of them compute the same depth the one
    drawn first wins. For renderables that MuJoCo made anew per episode that order changed from one loading of a
    model to the next (about one loading in seven; a few dozen pixels by one level along the seams after the
    anti-aliasing and the ambient occlusion); the renderables of a process's first loading are drawn in the same
    order in every process."""
    d = copy.copy(m)
    rules = look["model_materials"]
    for t in range(d.ntex):
        d.tex_colorspace[t] = int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)        # Filament reads "auto" as linear
    part_mat = {}
    for g in range(d.ngeom):
        part = part_of_geom(m, g)
        if part not in ROBOT_PARTS:
            d.geom_group[g] = HIDDEN_GROUP
            continue
        d.geom_group[g] = GEOM_GROUP
        if part == "robot":
            continue
        name = "look_" + part
        if part not in part_mat:
            mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if mid < 0:
                raise RenderError("renderer: the model has no material %r (lab_sim.build adds them)" % name)
            r = rules[part]
            d.mat_rgba[mid, :3] = _gain_srgb(r.get("srgb", m.geom_rgba[g, :3]), r.get("gain", 1.0))
            d.mat_rgba[mid, 3] = 1.0
            d.mat_roughness[mid] = r["roughness"]
            d.mat_metallic[mid] = r["metallic"]
            d.mat_specular[mid] = r.get("specular", 0.5)
            part_mat[part] = mid
        d.geom_matid[g] = part_mat[part]
        d.geom_rgba[g] = [0.5, 0.5, 0.5, 1.0]           # MuJoCo's default: the material's colour is the geom's
    for i in range(d.nmat):
        name = d.mat(i).name
        if name.startswith("look_") or (d.mat_texid[i] >= 0).any():
            continue                                    # (PBR terms on a textured material crash MuJoCo 3.16's Filament)
        r = rules["robot_default"]
        lum = float(np.mean(d.mat_rgba[i, :3]))
        for rule in rules["robot_by_colour"]:
            if rule["lum_min"] <= lum < rule["lum_max"] and float(np.ptp(d.mat_rgba[i, :3])) <= rule.get("chroma_max", 9.0):
                r = rule
                break
        d.mat_roughness[i] = r["roughness"]
        d.mat_metallic[i] = r["metallic"]
        d.mat_specular[i] = r.get("specular", 0.5)
        if "srgb" in r:
            d.mat_rgba[i, :3] = r["srgb"]
        d.mat_rgba[i, :3] = _gain_srgb(d.mat_rgba[i, :3], r.get("gain", 1.0))
    d.light_intensity[:] = 0.0
    d.light_castshadow[:] = 0
    return d


class Episode(object):
    """Everything the renderer needs of one episode's model, built in the caller's thread."""

    def __init__(self, m, look):
        t0 = time.perf_counter()
        self.source = m                                 # the episode's own model (read only: robot_model copies it)
        self.model = _drawn_model(m, look)
        self.robot, self.robot_sig = robot_geoms(m, look)
        if str(look["render"].get("mount_camera", "")):
            raise RenderError("renderer: render.mount_camera is not supported (the hand's shell is drawn for every camera)")
        self.gain = float(look["exposure_scale"])
        self.ibl_intensity = float(look["ibl"]["intensity"]) * self.gain
        self.lamps = [dict(l, intensity=float(l["candela"]) * self.gain) for l in look["lamps"]]
        self.head_lux = float(look["headlight"]["lux"]) * self.gain
        self.supersample = max(1, int(look["render"].get("supersample", 1)))
        self.supersample_of = {str(k): max(1, int(v)) for k, v in look["render"].get("supersample_by_camera", {}).items()}
        self.cams = {m.camera(i).name: i for i in range(m.ncam)}
        self.seconds = time.perf_counter() - t0


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


class View(object):
    """One model's cameras; the Filament objects stay in the renderer's thread."""

    def __init__(self, model, renderer=None):
        self._fil = renderer or _Filament.instance()
        self.model = model
        try:
            self.episode = Episode(model, self._fil.look)
            self._scn = mujoco.MjvScene(model, 1)              # for mjv_updateCamera only: no geoms, no GL context
            self._cam = mujoco.MjvCamera()
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001 - whatever breaks here is the renderer's, not the caller's
            raise RenderError("renderer: building the episode's drawn model: %s: %s" % (type(exc).__name__, exc))
        self.build_s = self.episode.seconds
        self._sid = self._fil.call(self._fil.new_session, self.episode)
        self._finalizer = weakref.finalize(self, self._fil.post, self._fil.free_session, self._sid)

    def render(self, cam, data, width, height):
        """RGB uint8 (height, width, 3), rows top-down, of the camera named `cam` in the state `data`; no noise."""
        try:
            ep = self.episode
            W, H, k = int(width), int(height), ep.supersample_of.get(cam, ep.supersample)
            pos, fwd, up, fr = gl_camera(self.model, data, ep.cams[cam], self._scn, self._cam, W / float(H))
            args = (self._sid, data, pos, fwd, up, fr, W * k, H * k)
        except Exception as exc:  # noqa: BLE001
            raise RenderError("renderer: camera %r: %s: %s" % (cam, type(exc).__name__, exc))
        img = self._fil.call(self._fil.draw, *args)
        return img if k == 1 else _box_down(img, k)

    def close(self):
        if self._finalizer.detach() is not None and not self._fil.broken:
            self._fil.call(self._fil.free_session, self._sid)


def open_view(model):
    """The cameras of the episode's `model`. Raises RenderError."""
    return View(model)


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
            raise RenderError("renderer: the render bundle in %s lacks %s (LAB_RENDER_DIR)"
                              % (RENDER_DIR, ", ".join(missing)))
        self.jobs = queue.Queue()
        self.sessions = {}
        self.active = None
        self.live = {}
        self.targets = {}
        self.next_sid = 0
        self.view = None                               # the filament::View, once its dithering is off (_undither)
        self.stats = {"activations": 0, "frames": 0, "targets_created": 0}
        self.robot = None                              # the robot's model, data and renderables, for the process (_robot)
        self.thread = threading.Thread(target=self._loop, name="lab-filament", daemon=True)
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
            self.broken = "renderer: a call did not return within %.0f s (LAB_RENDER_TIMEOUT)" % CALL_TIMEOUT
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
        self.ctx.set_clear_color([float(x) for x in look["render"].get("clear_color", [0.0, 0.0, 0.0])])
        self.ibl_tex = self._ktx(look["ibl"]["file"], F.SamplerType.TEXTURE_CUBE)
        self.scn = self.ctx.create_scene(F.SceneParams())
        self._configure()
        lp = F.LightParams()
        lp.type, lp.texture, lp.color, lp.intensity = F.LightType.LIGHT_IMAGE, self.ibl_tex, [1.0, 1.0, 1.0], 1.0
        self.ibl = self.ctx.create_light(lp)
        self.scn.add_light(self.ibl)
        lp = F.LightParams()
        lp.type, lp.color, lp.intensity, lp.cast_shadow = F.LightType.LIGHT_DIRECTIONAL, [1.0, 1.0, 1.0], 0.0, False
        self.head = self.ctx.create_light(lp)
        self.scn.add_light(self.head)
        self._load_room()
        self.start_s = time.perf_counter() - t0

    def _configure(self):
        """The look's view options (they travel as custom fields of a tiny model: MuJoCo reads filament.* from
        them)."""
        look = self.look
        spec = mujoco.MjSpec()
        for k, v in sorted(look["filament_numeric"].items()):       # a number or a list (curves: r, g, b)
            spec.add_numeric(name=k, data=[float(x) for x in (v if isinstance(v, (list, tuple)) else [v])])
        for k, v in sorted(look["filament_text"].items()):
            spec.add_text(name=k, data=str(v))
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], pos=[0, 0, -50])
        self.look_model = spec.compile()
        for _ in range(2):                             # MuJoCo 3.16 applies the colour grading one configure late
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
        """The room bundle: one mesh, one material and one renderable per material of room.json, for the process.
        The look's gain of a material's class (linear, a number or one per channel) goes into its texture where it
        has one (a gain above 1 cannot go into a material's colour), else into its colour."""
        F = self.F
        with open(os.path.join(RENDER_DIR, "room.json")) as fh:
            meta = json.load(fh)
        z = np.load(os.path.join(RENDER_DIR, "room.npz"))
        V, N, U, I, tex = z["V"], z["N"], z["UV"], z["I"], z["tex"]
        tex_ofs, tw, th = z["tex_ofs"], z["tex_w"], z["tex_h"]
        rl = self.look["room"]
        classes = rl["classes"]
        texs = {}
        self.room = []
        for mm in meta["materials"]:
            if mm["icnt"] == 0:
                continue
            c = classes.get(mm["cls"], classes["default"])
            if c.get("hide"):
                continue
            gain = np.broadcast_to(np.asarray(c.get("gain", 1.0), float), (3,)) * float(rl["albedo_gain"])
            rgba = np.array(mm["rgba"], np.float32)
            color_tex = None
            if mm["tex"] >= 0:
                key = (mm["tex"], tuple(np.round(gain, 4)))
                if key not in texs:
                    k = mm["tex"]
                    o, w, h = int(tex_ofs[k]), int(tw[k]), int(th[k])
                    img = tex[o:o + w * h * 3].reshape(h, w, 3)
                    texs[key] = self._texture(img if np.allclose(gain, 1.0) else _gain_srgb8(img, gain))
                color_tex = texs[key]
            else:
                rgba[:3] = _gain_srgb(rgba[:3], gain)
            v0, vc, i0, ic = mm["vofs"], mm["vcnt"], mm["iofs"], mm["icnt"]
            mesh = self._mesh(V[v0:v0 + vc], N[v0:v0 + vc], U[v0:v0 + vc], I[i0:i0 + ic])
            if c.get("unlit") is not None:              # an emitter (the lamp's diffuser): its own colour, unlit
                e = np.asarray(c["unlit"], np.float32)
                mat = F.Material(color=np.array([e[0], e[1], e[2], 1.0], np.float32), decor=True)
            else:
                # specular: the strength of a non-metal's mirror reflection (MuJoCo's material term, 0.5 by default).
                # The image light has no occlusion, so a dark part standing in shade (the cabinet under the bench
                # top, the pedestal) would mirror the bright bench and lamps: the look lowers it per class
                mat = F.Material(color=rgba,
                                 roughness=float(np.clip(c.get("roughness", mm["roughness"]) * rl["roughness_scale"], 0.05, 1.0)),
                                 metallic=float(c.get("metallic", mm["metallic"])) * rl["metallic_scale"],
                                 specular=float(c.get("specular", rl.get("specular", 0.5))),
                                 emissive=float(mm["emissive"]) * float(c.get("emissive", rl["emissive"])),
                                 color_texture=color_tex)
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadow=bool(mm.get("shadow") and rl["cast_shadow"]),
                                                              receive_shadow=True))
            r.set_mesh(mesh, 0, ic)
            r.set_material(mat)
            self.scn.add_renderable(r)
            self.room.append((mesh, mat, r))
        self.room_textures = list(texs.values())
        self.room_stats = meta["stats"]

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
        self._robot(ep)                                 # before anything of the episode is made (robot_model)
        res = F.ModelResources(self.ctx, ep.model)
        rends = F.ModelRenderables(self.scn, res)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[GEOM_GROUP] = 1
        opt.sitegroup[:] = 0
        rends.set_options(opt)
        self._lamps(ep)
        self.ibl.set_intensity(float(ep.ibl_intensity))
        self.live = {"res": res, "rends": rends, "opt": opt}
        self.active = sid
        self.stats["activations"] += 1
        self.last_activation_s = time.perf_counter() - t0

    def _robot(self, ep):
        """The robot's renderables: MuJoCo's own, of a robot-only copy of the process's first episode's model
        (robot_model), made once, before anything of an episode; every later episode must draw the same robot
        (robot_geoms)."""
        if self.robot is not None:
            if ep.robot_sig != self.robot["sig"]:
                raise RenderError("renderer: this episode's robot is not the one the process draws")
            return
        F = self.F
        model = robot_model(ep.source, self.look)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        res = F.ModelResources(self.ctx, model)
        rends = F.ModelRenderables(self.scn, res)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[GEOM_GROUP] = 1
        opt.sitegroup[:] = 0
        rends.set_options(opt)
        self.robot = {"sig": ep.robot_sig, "idx": ep.robot.copy(), "model": model, "data": data, "res": res,
                      "rends": rends, "opt": opt}

    def _lamps(self, ep):
        F = self.F
        want = {l["name"]: l for l in ep.lamps}
        for name in list(self.lamps):
            if name not in want:
                self.scn.remove_light(self.lamps.pop(name)[0])
        for name, l in want.items():
            sig = (l["type"], bool(l.get("shadow")), int(l.get("shadow_map", 0)))
            if name in self.lamps and self.lamps[name][1] != sig:
                self.scn.remove_light(self.lamps.pop(name)[0])
            if name not in self.lamps:
                lp = F.LightParams()
                lp.type = F.LightType.LIGHT_SPOT if l["type"] == "spot" else F.LightType.LIGHT_POINT
                lp.color, lp.intensity = [1.0, 1.0, 1.0], 0.0
                lp.cast_shadow = bool(l.get("shadow"))
                if l.get("shadow"):
                    lp.shadow_map_size = int(l.get("shadow_map", 512))
                lp.bulb_radius = float(l.get("bulb", 0.02))
                lp.cutoff = float(l.get("cutoff", 180.0))
                lp.softness = float(l.get("softness", 0.0))
                light = self.ctx.create_light(lp)
                self.scn.add_light(light)
                self.lamps[name] = (light, sig)
            light = self.lamps[name][0]
            d = np.asarray(l.get("dir", [0, 0, -1.0]), float)
            light.set_pose(np.asarray(l["pos"], np.float32), (d / np.linalg.norm(d)).astype(np.float32))
            light.set_color(np.asarray(l.get("color", [1.0, 1.0, 1.0]), np.float32))
            light.set_intensity(float(l["intensity"]))
            light.set_cutoff(float(l.get("cutoff", 180.0)))
            light.set_bulb_radius(float(l.get("bulb", 0.02)))

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

    def draw(self, sid, data, pos, forward, up, frustum, width, height):
        """RGB (rows top-down) of the session's episode in the state `data`, through MuJoCo's own camera."""
        F = self.F
        self._activate(sid)
        ep = self.sessions[sid]
        self.live["rends"].update(data)
        R = self.robot                                  # the robot: MuJoCo's renderables of the process's copy, at
        R["data"].geom_xpos[R["idx"]] = data.geom_xpos[ep.robot]       # this state's poses (geom for geom: this
        R["data"].geom_xmat[R["idx"]] = data.geom_xmat[ep.robot]       # episode's robot may stand at other indices)
        R["rends"].update(R["data"])
        self.head.set_pose(np.asarray(pos, np.float32), np.asarray(forward, np.float32))
        self.head.set_intensity(float(ep.head_lux))
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
        if self.view is None:                          # the process's first image: switch the dithering off, again
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
        R = getattr(self, "robot", None) or {}
        for k in ("rends", "res"):                      # renderables before the resources they draw
            R.pop(k, None)
        self.robot = None
        self.room_textures = []
        self.head = self.ibl = None
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
