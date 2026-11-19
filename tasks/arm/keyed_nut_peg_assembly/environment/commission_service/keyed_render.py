"""The cell's cameras: MuJoCo 3.16's Filament renderer (physically based) on Vulkan; in the task image Vulkan is
Mesa's lavapipe on the CPU (no GPU). keyed_env.NutEnv.render draws every image through this module, in the
commissioning service and in the grader alike.

What is drawn
  * the episode's own robot, pegs and nuts, read from the episode's compiled model (which is not touched): the
    robot's visual meshes (its collision shapes are not drawn, except the fingers' rubber pads), every peg part,
    and every nut as one mesh of its blocks in the unit's own colour, with the look's PBR terms, in the simulator's
    state at the moment of the image; the wrist camera does not see the shell of the hand it is mounted in;
  * the room around the cell (room.npz / room.json of the render bundle in KEYED_RENDER_DIR, default
    /opt/task-assets/payload/render): the laboratory, its shelving and benches, the cell's bench top and cabinet,
    the robot's pedestal, the camera gantry with its lamp. Drawn only: nothing of it is in the simulated model. The
    model's own floor plane, back wall, table block and pedestal block are not drawn: the bundle's bench top and
    pedestal stand exactly where they are.
Light: an image light (ibl.ktx of the bundle: a path-traced probe of the laboratory at the bench, the robot, the
pegs and the nuts taken out) and the gantry's lamp, a bar, as two spot lights with a shadow map each
(keyed_look.json); ambient occlusion shades what the image light cannot (a recess, the gap under a part). The
cell's lamp brightness of the day (keyed_env's tolerance TOL["light"], the factor NutEnv.lamp_gain) multiplies the
image light and every lamp alike; the cameras' exposure (the look's exposure_scale) is fixed.
Cameras: exactly the simulated cameras. Position, axes and the off-centre frustum are the ones MuJoCo computes for
the episode's model (mjv_updateCamera), so each camera's real mount, focal length and principal point of the day
are in the image as they were with MuJoCo's own renderer; image sizes are the cameras' resolutions. An image is
drawn at twice that size and averaged over 2 x 2 blocks in linear light (the look's render.supersample): the
anti-aliasing that keeps a peg's bands apart in the fixed cameras.

An image is a function of the model, the simulator's state and the camera alone: the same camera in the same state
gives the same bytes however often it is drawn, whatever was drawn before it, however many other episodes the
process has drawn and in whichever process of the same image. Two things make it so. Filament's temporal
dithering, a new noise per frame, is off (_undither). And nothing that is drawn is made anew per episode. Where two
surfaces compute the same depth at a sample (blocks of a nut, which overlap by design; shells of the arm, which
pass through each other at the joints), the one Filament draws first wins, and the order in which it draws two
renderables is not a function of the scene: it changed when MuJoCo's ModelRenderables made an episode's
renderables again (a handful of pixels by one or two levels on the nuts, from a process's second episode on), when
a renderable was given a material again, and, for a nut drawn as some four hundred renderables of its own, from
process to process. So (PARTS below): every renderable, material, mesh and texture is made once per process, at
its first episode and in a fixed order; an episode only writes data into them (poses, sizes, a nut's vertices and
colour); and a nut is ONE mesh, whose triangles are drawn in the mesh's own order. The sensor noise stays the
environment's.

Filament runs in one dedicated thread per process (its engine is thread-bound; a forked child starts its own).
Objects that live for the process: the context, the room (meshes, textures, materials, renderables), the robot's,
the pegs' and the nuts' renderables with their materials, meshes and textures, the image light, the lamps, the
scene, one render target per image size. Per episode only data changes: the peg parts' sizes, the nuts' vertices
and colours, the lamps' brightness. The first image of a process takes several seconds (lavapipe compiles the
shaders).

    view = open_view(env)          # after env.reset(): reads the episode's parts (no Filament work yet)
    rgb = view.render("left")      # uint8 (240, 320, 3), rows top-down, no noise
    view.close()                   # or let it be garbage-collected

Every failure of the renderer that Python sees (a missing or broken file of the render bundle, an error while an
episode is built or an image drawn, a call that does not return within KEYED_RENDER_TIMEOUT seconds, default 300)
is raised as RenderError: an error of the environment, never of a controller. A renderer that failed once stays
failed in that process. A Vulkan backend that cannot start at all (no usable driver) aborts the whole process
inside Filament before Python can catch anything; the task image's build draws the three cameras through this
module for every account that renders, so the image proves its driver.

Nothing here changes the simulation: the renderer reads the model and the data and writes neither.
"""
import atexit
import ctypes
import copy
import gc
import hashlib
import json
import math
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
RENDER_DIR = os.environ.get("KEYED_RENDER_DIR", "/opt/task-assets/payload/render")
LOOK_FILE = os.path.join(HERE, "keyed_look.json")
BUNDLE_FILES = ("room.npz", "room.json", "ibl.ktx")
CALL_TIMEOUT = float(os.environ.get("KEYED_RENDER_TIMEOUT", "300"))
CAM_FAR = 40.0                 # the room's far corner; the near plane is the simulated camera's own
# PARTS: what is drawn of the episode's model, by part_of_geom's names. The process makes the renderables of every
# part once, at its first episode, in this order and in the model's geom order within a part; none is made again,
# and none gets another material, mesh or texture object.
ROBOT_PARTS = ("robot", "hand", "finger", "pad")   # MuJoCo's own renderables of a robot-only copy of the first episode's model
PEG_PARTS = ("collar", "shaft", "pin", "band")     # one renderable per geom: MuJoCo's own box / cylinder shape, resized per episode
NUT_BLOCKS_MAX = 24 * 9 * 2 + 1                    # a nut: one renderable; its mesh has room for this many boxes (blocks, handle)
GEOM_GROUP = 1                 # in the robot-only copy: what the cameras show
MOUNT_GROUP = 2                # ... the hand's own shell: the wrist camera sits in it and does not see it
HIDDEN_GROUP = 4               # ... everything else (the pegs and the nuts have renderables of their own)


class RenderError(RuntimeError):
    """The renderer failed: an error of the environment (the commissioning service or the grader), not of a
    controller."""


def load_look(path=LOOK_FILE):
    """The look's numbers (keyed_look.json beside this file)."""
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
def part_of_geom(m, g):
    """Which drawn part a geom of the episode's model is: "nut<i>", "collar", "shaft", "pin", "band" (a peg's
    parts), "hand", "finger", "pad" (the gripper's parts without a material of their own), "robot" (a visual mesh
    of the arm, drawn with its own material), or None (not drawn: the robot's collision shapes, the model's
    scenery)."""
    body = m.body(m.geom_bodyid[g]).name
    if body.startswith("o_nut"):
        return "nut" + body[5:].split("_")[0]
    if body.startswith("o_peg"):
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            return "shaft" if int(m.geom_contype[g]) or int(m.geom_conaffinity[g]) else "band"
        return "collar" if float(m.geom_size[g][0]) > 0.01 else "pin"
    if body.startswith("r_"):
        name = m.geom(g).name or ""
        if name.endswith("_pad_collision"):
            return "pad"                                # the fingers' rubber pads: what touches a nut
        if int(m.geom_group[g]) != 1:
            return None                                 # a collision shape
        if name == "r_g_hand_visual":
            return "hand"
        if name.startswith("r_g_finger") and name.endswith("_visual"):
            return "finger"
        return "robot"
    return None


def _material(m, g, part, look):
    """The material of a peg part or of a nut: (r, g, b, roughness, metallic, specular, glossiness), the
    colour in sRGB: the geom's own colour (a nut's is the unit's shade of the day) times the look's gain, the
    look's PBR terms. specular and glossiness are a MuJoCo material's `specular` and `shininess` (0.5 where the look
    names none) as MuJoCo 3.16's Filament module scales them for its own renderables (x 0.2, x 0.1); every number
    passes through a float32, as it did in the model's material that this stands for."""
    r = look["model_materials"]["nut" if part.startswith("nut") else part]
    rgb = np.asarray(_gain_srgb(r.get("srgb", m.geom_rgba[g, :3]), r.get("gain", 1.0)), np.float32)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(np.float32(r["roughness"])), float(np.float32(r["metallic"])),
            float(np.float32(0.2) * np.float32(r.get("specular", 0.5))), float(np.float32(0.1) * np.float32(0.5)))


def robot_geoms(m, look):
    """The robot's drawn geoms (the visual meshes of the arm, the hand and the fingers; of the collision shapes only
    the fingers' rubber pads): (their indices in the model, a signature). The signature is the same for two models
    whose robot is the same and stands at the same geom indices, which every episode of a process must be: the
    robot's renderables are made once (robot_model)."""
    idx, sig = [], hashlib.sha256()
    meshes = set()
    for g in range(m.ngeom):
        part = part_of_geom(m, g)
        if part not in ROBOT_PARTS:
            continue
        idx.append(g)
        mid, j = int(m.geom_matid[g]), int(m.geom_dataid[g])
        sig.update(repr((g, part, m.body(m.geom_bodyid[g]).name, int(m.geom_type[g]), j, m.geom_size[g].tolist(),
                         m.geom_rgba[g].tolist(), m.geom_pos[g].tolist(), m.geom_quat[g].tolist(),
                         (m.mat_rgba[mid].tolist(), float(m.mat_shininess[mid])) if mid >= 0 else None)).encode())
        if j >= 0 and j not in meshes:
            meshes.add(j)
            va, fa = int(m.mesh_vertadr[j]), int(m.mesh_faceadr[j])
            sig.update(m.mesh_vert[va:va + int(m.mesh_vertnum[j])].tobytes())
            sig.update(m.mesh_face[fa:fa + int(m.mesh_facenum[j])].tobytes())
    sig.update(json.dumps(look["model_materials"], sort_keys=True).encode())
    return np.asarray(idx, np.intp), sig.hexdigest()


def robot_model(m, look):
    """A copy of an episode's model in which only the robot is drawn, with the render-only fields set for Filament
    (the episode's model is untouched): the robot's drawn geoms in group GEOM_GROUP (the hand's shell in
    MOUNT_GROUP), everything else in HIDDEN_GROUP; the gripper's parts on the materials keyed_env.build_model added
    for them (look_hand, look_finger, look_pad); the arm's shells on their own materials, with the look's PBR terms
    by the material's colour (white covers, dark joints, the rest). The process keeps the copy of its first episode
    and MuJoCo's renderables of it for good; every image moves them to the state's poses (_Filament.draw)."""
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
        d.geom_group[g] = MOUNT_GROUP if part == "hand" else GEOM_GROUP
        if part == "robot":
            continue
        name = "look_" + part
        if part not in part_mat:
            mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if mid < 0:
                raise RenderError("renderer: the model has no material %r (keyed_env.build_model adds them)" % name)
            r = rules[part]
            d.mat_rgba[mid, :3] = _gain_srgb(r.get("srgb", m.geom_rgba[g]), r.get("gain", 1.0))
            d.mat_rgba[mid, 3] = 1.0
            d.mat_roughness[mid] = r["roughness"]
            d.mat_metallic[mid] = r["metallic"]
            d.mat_specular[mid] = r.get("specular", 0.5)
            part_mat[part] = mid
        d.geom_matid[g] = part_mat[part]
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


def peg_parts(m, look):
    """The pegs' parts (collars, shafts, pins, bands), in the model's order: the geom's index, kind, type, size of
    the day (a real peg has its own shaft and pin) and material."""
    parts = []
    for g in range(m.ngeom):
        part = part_of_geom(m, g)
        if part not in PEG_PARTS:
            continue
        gt = int(m.geom_type[g])
        if gt not in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            raise RenderError("renderer: a peg part that is neither a box nor a cylinder")
        sz = m.geom_size[g]                             # as MuJoCo scales its own shapes: a cylinder by (r, r, half length)
        size = (sz[0], sz[1], sz[2]) if gt == int(mujoco.mjtGeom.mjGEOM_BOX) else (sz[0], sz[0], sz[1])
        parts.append({"geom": g, "part": part, "type": gt, "size": np.asarray(size, np.float32),
                      "material": _material(m, g, part, look)})
    return parts


def _box_arrays():
    """A box with half sizes 1: 24 vertices, their normals, 12 triangles (counter-clockwise from outside)."""
    V, N, I = [], [], []
    for ax in range(3):
        for sgn in (1.0, -1.0):
            n = np.zeros(3)
            n[ax] = sgn
            u = np.zeros(3)
            u[(ax + 1) % 3] = 1.0
            w = np.cross(n, u)
            k = len(V)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                V.append(n + a * u + b * w)
                N.append(n)
            I += [k, k + 1, k + 2, k, k + 2, k + 3]
    return np.asarray(V, np.float64), np.asarray(N, np.float64), np.asarray(I, np.uint32)


def nut_parts(m, look):
    """The nuts, in their order: the body's index, the nut's material (its colour of the day) and ONE mesh of all
    its blocks and its handle in the body's frame (vertices, normals, indices; the blocks in the model's geom
    order). A nut's blocks overlap by design, and where two of them compute the same depth the one drawn first
    wins: within one mesh that is the mesh's own triangle order, the same in every image of the same state."""
    by_nut = {}
    for g in range(m.ngeom):
        part = part_of_geom(m, g)
        if part is not None and part.startswith("nut"):
            if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise RenderError("renderer: a nut geom that is not a box")
            by_nut.setdefault(int(part[3:]), []).append(g)
    v0, n0, i0 = _box_arrays()
    nuts = []
    for i in sorted(by_nut):
        gs = by_nut[i]
        if len(gs) > NUT_BLOCKS_MAX or len({int(m.geom_bodyid[g]) for g in gs}) != 1:
            raise RenderError("renderer: nut %d is not one body of at most %d blocks" % (i, NUT_BLOCKS_MAX))
        R = np.empty((len(gs), 3, 3))
        for k, g in enumerate(gs):
            mujoco.mju_quat2Mat(R[k].reshape(9), m.geom_quat[g])
        V = np.einsum("kij,kvj->kvi", R, v0[None] * m.geom_size[gs][:, None, :]) + m.geom_pos[gs][:, None, :]
        N = np.einsum("kij,vj->kvi", R, n0)
        I = (i0[None, :] + 24 * np.arange(len(gs), dtype=np.uint32)[:, None]).ravel()
        nuts.append({"body": int(m.geom_bodyid[gs[0]]), "material": _material(m, gs[0], "nut%d" % i, look),
                     "V": np.ascontiguousarray(V.reshape(-1, 3), np.float32),
                     "N": np.ascontiguousarray(N.reshape(-1, 3), np.float32), "I": np.ascontiguousarray(I, np.uint32)})
    return nuts


class Episode(object):
    """Everything the renderer needs of one episode, read from its model in the caller's thread."""

    def __init__(self, env, look):
        t0 = time.perf_counter()
        m = env.model
        self.model = m                                  # read only (copied for the robot at the process's first episode)
        self.robot, self.robot_sig = robot_geoms(m, look)
        self.pegs = peg_parts(m, look)
        self.nuts = nut_parts(m, look)
        self.quality = (int(m.vis.quality.numstacks), int(m.vis.quality.numslices), int(m.vis.quality.numquads))
        self.gain = float(getattr(env, "lamp_gain", 1.0)) * float(look["exposure_scale"])
        self.ibl_intensity = float(look["ibl"]["intensity"]) * self.gain
        self.lamps = []
        for l in look["lamps"]:
            self.lamps.append(dict(l, intensity=float(l["candela"]) * self.gain))
        self.head_lux = float(look["headlight"]["lux"]) * self.gain
        self.supersample = max(1, int(look["render"].get("supersample", 1)))
        self.wrist_cam = env.wrist_cam
        self.cams = {"wrist": int(env.wrist_cam)}
        for c in ("left", "right"):
            self.cams[c] = int(m.camera(c).id)
        self.size = {c: (int(m.cam_resolution[i][0]), int(m.cam_resolution[i][1])) for c, i in self.cams.items()}
        self.seconds = time.perf_counter() - t0


def gl_camera(m, d, camid, scn, cam):
    """MuJoCo's own camera for `camid` in the state `d`: (pos, forward, up, [centre, half width, bottom, top, near]
    of its frustum). The intrinsics of the day (focal length, principal point) are in the frustum; the near plane is
    the model's (what is nearer is not in the image: the hand's own body around the wrist camera)."""
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = int(camid)
    mujoco.mjv_updateCamera(m, d, cam, scn)
    a, b = scn.camera[0], scn.camera[1]
    pos = (np.asarray(a.pos, float) + np.asarray(b.pos, float)) / 2.0
    return (pos, np.asarray(a.forward, float).copy(), np.asarray(a.up, float).copy(),
            [float(a.frustum_center), float(a.frustum_width), float(a.frustum_bottom), float(a.frustum_top),
             float(a.frustum_near)])


class View(object):
    """One episode's cameras; the Filament objects stay in the renderer's thread."""

    def __init__(self, env, renderer=None):
        self._fil = renderer or _Filament.instance()
        self.env_ref = weakref.ref(env)
        try:
            self.episode = Episode(env, self._fil.look)
            self._scn = mujoco.MjvScene(env.model, 1)          # for mjv_updateCamera only: no geoms, no GL context
            self._cam = mujoco.MjvCamera()
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001 - whatever breaks here is the renderer's, not the caller's
            raise RenderError("renderer: reading the episode's parts: %s: %s" % (type(exc).__name__, exc))
        self.build_s = self.episode.seconds
        self._sid = self._fil.call(self._fil.new_session, self.episode)
        self._finalizer = weakref.finalize(self, self._fil.post, self._fil.free_session, self._sid)

    def render(self, cam, data=None):
        try:
            env = self.env_ref()
            d = env.data if data is None else data
            ep = self.episode
            W, H = ep.size[cam]
            k = ep.supersample
            pos, fwd, up, fr = gl_camera(env.model, d, ep.cams[cam], self._scn, self._cam)
            args = (self._sid, d, pos, fwd, up, fr, W * k, H * k, cam != "wrist")
        except Exception as exc:  # noqa: BLE001
            raise RenderError("renderer: camera %r: %s: %s" % (cam, type(exc).__name__, exc))
        img = self._fil.call(self._fil.draw, *args)
        return img if k == 1 else _box_down(img, k)

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
            raise RenderError("renderer: the render bundle in %s lacks %s (KEYED_RENDER_DIR)"
                              % (RENDER_DIR, ", ".join(missing)))
        self.jobs = queue.Queue()
        self.sessions = {}
        self.active = None
        self.parts = None                              # the robot's, the pegs' and the nuts' renderables (_parts)
        self.targets = {}
        self.next_sid = 0
        self.view = None                               # the filament::View, once its dithering is off (_undither)
        self.stats = {"activations": 0, "frames": 0, "targets_created": 0}
        self.thread = threading.Thread(target=self._loop, name="keyed-filament", daemon=True)
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
            self.broken = "renderer: a call did not return within %.0f s (KEYED_RENDER_TIMEOUT)" % CALL_TIMEOUT
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
            self.active = None

    def _material_of(self, mat):
        r, g, b, rough, metal, spec, gloss = mat
        return self.F.Material(color=np.array([r, g, b, 1.0], np.float32), roughness=rough, metallic=metal, specular=spec,
                               reflectance=0.0, glossiness=gloss, emissive=0.0)

    def _renderable(self, mat):
        r = self.ctx.create_renderable(self.F.RenderableParams(cast_shadow=True, receive_shadow=True))
        r.set_material(self._material_of(mat))
        return r

    def _parts(self, ep):
        """The renderables of everything an episode draws of its own, made once: at the process's first episode, in
        the order robot (MuJoCo's own renderables of robot_model, in the model's geom order), peg parts (the
        model's order), nuts (one each). Every later episode must have the same robot at the same geom indices and
        peg parts and nuts of the same kinds and materials; what differs between episodes is data (_activate)."""
        sig = (ep.robot_sig, tuple((p["part"], p["type"], p["material"]) for p in ep.pegs),
               tuple(n["material"][3:] for n in ep.nuts), ep.quality)
        if self.parts is not None:
            if sig != self.parts["sig"]:
                raise RenderError("renderer: this episode's robot, pegs or nuts are not of the kind the process draws")
            return
        F = self.F
        model = robot_model(ep.model, self.look)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        res = F.ModelResources(self.ctx, model)
        rends = F.ModelRenderables(self.scn, res)
        opts = []
        for mount in (False, True):                     # [without, with] the hand's shell
            opt = mujoco.MjvOption()
            opt.geomgroup[:] = 0
            opt.geomgroup[GEOM_GROUP] = 1
            opt.geomgroup[MOUNT_GROUP] = int(mount)
            opt.sitegroup[:] = 0
            opts.append(opt)
        rends.set_options(opts[1])
        robot = {"model": model, "data": data, "res": res, "rends": rends, "opts": opts, "mount": True}
        zero = np.zeros(3, np.float32)
        pegs, nuts = [], []
        for p in ep.pegs:
            r = self._renderable(p["material"])
            r.set_geom_mesh(mujoco.mjtGeom(p["type"]), *ep.quality)
            r.set_size(zero)
            self.scn.add_renderable(r)
            pegs.append(r)
        for n in ep.nuts:
            # the nut's colour of the day is a texture of one texel (8 bits per channel, sRGB), written anew per
            # episode (_activate): the renderable keeps the material it was made with ...
            tex = self._texture(np.full((1, 1, 3), 255, np.uint8))
            _r, _g, _b, rough, metal, spec, gloss = n["material"]
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadow=True, receive_shadow=True))
            r.set_material(F.Material(color=np.ones(4, np.float32), roughness=rough, metallic=metal, specular=spec,
                                      reflectance=0.0, glossiness=gloss, emissive=0.0, color_texture=tex))
            # ... and the mesh it was made with: room for NUT_BLOCKS_MAX boxes, written anew per episode (boxes a
            # nut does not have are triangles of no area at the nut's origin)
            cfg = self._mesh_config(24 * NUT_BLOCKS_MAX, 36 * NUT_BLOCKS_MAX)
            mesh = self.ctx.create_mesh(cfg)
            self._nut_upload(mesh, cfg, n)
            r.set_mesh(mesh, 0, 36 * NUT_BLOCKS_MAX)
            r.set_size(np.ones(3, np.float32))
            self.scn.add_renderable(r)
            nuts.append({"renderable": r, "texture": tex, "mesh": mesh, "config": cfg})
        self.parts = {"sig": sig, "robot": robot, "pegs": pegs, "nuts": nuts}

    def _nut_upload(self, mesh, cfg, n):
        """A nut's blocks into its mesh of fixed size (the rest: zeros)."""
        nv, ni = 24 * NUT_BLOCKS_MAX, 36 * NUT_BLOCKS_MAX
        V, N, I = np.zeros((nv, 3), np.float32), np.zeros((nv, 3), np.float32), np.zeros(ni, np.uint32)
        V[:len(n["V"])], N[:len(n["N"])], I[:len(n["I"])] = n["V"], n["N"], n["I"]
        N[len(n["N"]):, 2] = 1.0
        self._mesh_upload(mesh, cfg, V, N, np.zeros((nv, 2), np.float32), I)

    def _activate(self, sid):
        """Make `sid` the episode that is drawn: the sizes of its peg parts go into the process's renderables, each
        nut's blocks into its mesh and its colour into its texture (no Filament object is made here)."""
        if self.active == sid:
            return
        t0 = time.perf_counter()
        ep = self.sessions[sid]
        self._parts(ep)
        for r, p in zip(self.parts["pegs"], ep.pegs):
            r.set_size(p["size"])
        for slot, n in zip(self.parts["nuts"], ep.nuts):
            td = self.F.TextureData()
            td.set_image(np.round(np.clip(np.asarray(n["material"][:3], float), 0.0, 1.0) * 255.0).astype(np.uint8).tobytes())
            slot["texture"].upload(td)
            self._nut_upload(slot["mesh"], slot["config"], n)
        self._lamps(ep)
        self.ibl.set_intensity(float(ep.ibl_intensity))
        self.active = sid
        self.stats["activations"] += 1
        self.last_activation_s = time.perf_counter() - t0
        gc.collect()
        _trim_heap()

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

    def draw(self, sid, data, pos, forward, up, frustum, width, height, mount=True):
        """RGB (rows top-down) of the session's episode in the state `data`, through MuJoCo's own camera. mount:
        whether the hand's shell is drawn (not for the wrist camera: it sits inside that shell, whose rim the near
        plane would cut into a bright line across the image)."""
        F = self.F
        self._activate(sid)
        ep = self.sessions[sid]
        P = self.parts
        R = P["robot"]
        if R["mount"] != bool(mount):
            R["rends"].set_options(R["opts"][int(bool(mount))])
            R["mount"] = bool(mount)
        R["data"].geom_xpos[ep.robot] = data.geom_xpos[ep.robot]      # the robot: MuJoCo's renderables of the copy,
        R["data"].geom_xmat[ep.robot] = data.geom_xmat[ep.robot]      # at this state's poses
        R["rends"].update(R["data"])
        xpos, xmat = np.asarray(data.geom_xpos, np.float32), np.asarray(data.geom_xmat, np.float32)
        for r, p in zip(P["pegs"], ep.pegs):            # every peg part and every nut where the state has it
            r.set_pose(xpos[p["geom"]], xmat[p["geom"]])
        for slot, n in zip(P["nuts"], ep.nuts):
            slot["renderable"].set_pose(np.asarray(data.xpos[n["body"]], np.float32), np.asarray(data.xmat[n["body"]], np.float32))
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
        # bytes, then the ones in the 512 bytes each of those points at, then once more. (Until amendment 7 this
        # was a depth-first walk with one set of visited pointers for all depths. A block that the walk met first
        # at its last level, through some other pointer, was then never read at the level where its own pointers
        # still counted, and when that block was the one that holds the view, no view was found: seen once in
        # about 250 process starts, as a RenderError at the process's first image.)
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
        P = self.parts or {"pegs": [], "nuts": []}
        return {"stats": dict(self.stats), "start_s": self.start_s, "room": self.room_stats,
                "last_activation_s": getattr(self, "last_activation_s", None), "targets": len(self.targets),
                "lamps": len(self.lamps), "peg_parts": len(P["pegs"]), "nuts": len(P["nuts"])}

    def _shutdown(self):
        """Every Filament object before the context (destroying the context first panics); also after a start that
        failed half-way (whatever was made is freed here, in the renderer's thread)."""
        self.sessions.clear()
        self.active = None
        if getattr(self, "scn", None) is not None:
            for name in list(getattr(self, "lamps", {})):
                try:
                    self.scn.remove_light(self.lamps.pop(name)[0])
                except Exception:  # noqa: BLE001
                    pass
        if self.parts is not None:                      # MuJoCo's renderables before the resources they draw
            for k in ("rends", "res", "data", "model"):
                self.parts["robot"].pop(k, None)
        self.parts = None
        self.targets.clear()
        self.lamps = {}
        self.room = []
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
