"""The agent cameras' colour images (cell internals): MuJoCo's Filament renderer (physically based, on Vulkan: Mesa's
lavapipe on the CPU in the sandbox) on a render twin of the cell, the cell's depth from MuJoCo's classic renderer on a
depth twin with the same geometry.

The physics model is never touched. Per episode two twins are compiled from a copy of the cell's MjSpec, with the same
bodies, joints and cameras, so a frame copies the physics qpos into the twin and runs forward kinematics only; the
instruments' changing parts (the lamps, the analyser's segments, the sight glasses, the level gauges' flap columns, the
drums' liquid) are copied from the physics model's state at every frame:

* the RGB twin: the skid with physically based materials (powder-coated frames, brushed steel pipes and tanks, painted
  valves and pumps, the cell's own tag plates, dials and scales), the lit lamps and segments glowing, the skid's back
  wall carried on as the room's lining wall, the water-treatment room the cell stands in with its props (static, drawn
  only; nothing of it stands within the skid or the robot's rail), the two ceiling panels over the rail as
  shadow-casting spot lights and an image light distilled from path-traced (OVRTX) renders of the same room (the probe
  in front of the skid with the robot hidden; the panels' own light is the spots'). The collision geoms are hidden in
  this twin. The site's light factor (layout "light", 0.8-1.2) scales every light, as it scales the cell's lights.
* the depth twin: the cell exactly as the physics model draws it (groups 0-2, the back wall carried on as in the RGB
  twin) plus the room's and the props' meshes near the skid, so depth stays registered with the colour image; the
  sensor model (noise, dropout, range) is the server's.

The light strengths and per-class colour gains were fitted, in linear units, to OVRTX stills of the two agent cameras
on states of a reference shift (LOOK below). Filament facts this relies on (MuJoCo 3.14): the engine is bound to the
thread that created it, so one thread owns the context and every Filament object and callers post jobs to it; one
model's Filament objects are live at a time; its job threads would each get a glibc malloc arena, so the arenas are
capped at import; textures are marked sRGB (Filament reads MuJoCo's "auto" as linear data); Filament draws every geom
group, so hidden geoms get alpha 0; a primitive's size is fixed when its Filament object is built, so a changing size is
drawn as a scaled transform (geom_xmat times the size ratio); a geom with a material is drawn in the material's colour,
and a material's colour, alpha and emission are read at every frame (so the geoms that switch colour carry their own).

Assets: $SKIDCELL_RENDER_DIR (default /opt/task-assets/render; the task image fetches them, see the Dockerfile):
manifest.json, decor/ (the room and props: arb-decor/1), tex/ (the cell's textures), ibl.ktx. Runtime dependencies:
mujoco (with its Filament module), numpy, Pillow.
"""

import atexit
import gc
import hashlib
import json
import math
import os
import queue
import threading
import time
import weakref


def _cap_malloc_arenas():
    """The renderer's job threads would each get a glibc malloc arena; two are plenty (M_ARENA_MAX = -8), and glibc
    honours the cap only before other threads allocate."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").mallopt(-8, 2)
    except (OSError, AttributeError):
        pass


_cap_malloc_arenas()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["RENDER_DIR", "LOOK", "available", "AgentLook", "RenderError"]

RENDER_DIR = os.environ.get("SKIDCELL_RENDER_DIR", "/opt/task-assets/render")
CAM_NEAR, CAM_FAR = 0.005, 120.0       # m: Filament's clip planes (the room's far walls are ~40 m out)
CALL_TIMEOUT = 300.0                    # s: one job of the renderer's thread
DEPTH_DECOR_M = 8.0                     # decor meshes reaching within this of the skid's centre enter the depth twin
SKID_CENTRE = (2.5, 0.0)                # m: x, y of the skid's centre (the skid is x 0 .. 5 m, its front plane y = 0)
# the skid's back wall (y 1.00 .. 1.04, x -0.2 .. 5.2, 2.3 m high) carried on as the room's lining wall, in both twins
WALL = {"x": (-2.25, 7.25), "z_top": 5.0}

# The look: Filament view options, the lights, the image light, every class's material (base colour sRGB, an optional
# texture and its tile, roughness, metallic) and its fitted colour gain (r, g, b on the linear albedo). Classes with
# "own": True keep the cell's own colour (the geom's or its material's rgba) under the class's gain; "emission_on" is
# the glow of a lamp lens or an analyser segment while lit.
LOOK = {
    "view": {"numeric": {"filament.ao.enabled": 0, "filament.shadows.type": 3, "filament.shadows.map_size": 1024},
             "text": {"filament.cg.tone_mapping": "filmic"}},
    "ibl": {"file": "ibl.ktx", "intensity": 40500.0},
    # supersampling: each frame is drawn at this many times the camera's size and averaged down (in linear light) to
    # it, pixel block by pixel block, so the pixel centres and the intrinsics are the camera's; thin strokes (the
    # analyser's segments, 1.1 mm wide) are covered as by the classic renderer's multisampling
    "ssaa": 2,
    # the two ceiling panels over the rail (1.18 x 0.28 m, 4300 K, 3.0 m up), each split into a shadow-casting part and
    # a shadow-free rest (a panel's soft shadow is about half as deep as a spot's)
    "lamps": [{"name": "panel_w", "pos": [1.3, -0.45, 2.995], "dir": [0.0, 0.0, -1.0], "intensity": 46000.0,
               "color": [1.0, 0.665, 0.452], "cutoff": 85.0, "exponent": 1.0, "bulb_m": 0.5, "range": 30.0,
               "shadow": True},
              {"name": "panel_w_fill", "pos": [1.3, -0.45, 2.995], "dir": [0.0, 0.0, -1.0], "intensity": 57500.0,
               "color": [1.0, 0.665, 0.452], "cutoff": 85.0, "exponent": 1.0, "bulb_m": 0.5, "range": 30.0,
               "shadow": False},
              {"name": "panel_e", "pos": [3.7, -0.45, 2.995], "dir": [0.0, 0.0, -1.0], "intensity": 46000.0,
               "color": [1.0, 0.665, 0.452], "cutoff": 85.0, "exponent": 1.0, "bulb_m": 0.5, "range": 30.0,
               "shadow": True},
              {"name": "panel_e_fill", "pos": [3.7, -0.45, 2.995], "dir": [0.0, 0.0, -1.0], "intensity": 57500.0,
               "color": [1.0, 0.665, 0.452], "cutoff": 85.0, "exponent": 1.0, "bulb_m": 0.5, "range": 30.0,
               "shadow": False}],
    "classes": {
        "panda_white": {"rgb": [0.92, 0.92, 0.91], "roughness": 0.36},
        "panda_offwhite": {"rgb": [0.84, 0.85, 0.85], "roughness": 0.40},
        "panda_black": {"rgb": [0.09, 0.09, 0.10], "roughness": 0.95},
        "panda_led_green": {"rgb": [0.20, 0.90, 0.30], "roughness": 0.30},
        "panda_led_blue": {"rgb": [0.10, 0.50, 0.90], "roughness": 0.30},
        "wall": {"rgb": [0.70, 0.72, 0.70], "roughness": 0.80},
        "skid_base": {"rgb": [0.16, 0.24, 0.38], "tex": "powder_col.png", "tile": 0.4, "roughness": 0.75},
        "rail_track": {"rgb": [0.55, 0.56, 0.57], "tex": "brushed_col.png", "tile": 0.5, "roughness": 0.45,
                       "metallic": 1.0},
        "rail_mast": {"rgb": [0.78, 0.79, 0.80], "tex": "alu_rough_col.png", "tile": 0.5, "roughness": 0.45,
                      "metallic": 1.0},
        "rail_frame": {"rgb": [0.22, 0.24, 0.27], "tex": "powder_col.png", "tile": 0.3, "roughness": 0.80},
        "plate": {"own": True, "roughness": 0.45},
        "steel": {"rgb": [0.70, 0.71, 0.72], "tex": "brushed_col.png", "tile": 0.4, "roughness": 0.45, "metallic": 1.0},
        "frame": {"own": True, "tex": "powder_col.png", "tile": 0.3, "roughness": 0.75},
        "pipe": {"rgb": [0.74, 0.75, 0.76], "tex": "brushed2_col.png", "tile": 0.3, "roughness": 0.45, "metallic": 1.0},
        "valve": {"own": True, "roughness": 0.38},
        "dark": {"rgb": [0.13, 0.13, 0.14], "roughness": 0.60},
        "tank": {"rgb": [0.80, 0.81, 0.82], "tex": "brushed_col.png", "tile": 0.8, "roughness": 0.40, "metallic": 1.0},
        "pump": {"own": True, "roughness": 0.35},
        "motor": {"own": True, "roughness": 0.45},
        "panel": {"rgb": [0.86, 0.86, 0.82], "roughness": 0.45},
        "rubber": {"rgb": [0.06, 0.06, 0.06], "roughness": 0.80},
        "tint": {"own": True, "roughness": 0.42},
        "part_light": {"own": True, "roughness": 0.40},
        "part_red": {"own": True, "roughness": 0.35},
        "part_dark": {"own": True, "roughness": 0.90},
        "lamp": {"own": True, "roughness": 0.20, "emission_on": 0.12},
        "segment": {"own": True, "roughness": 0.90, "emission_on": 0.10},
        "glass": {"own": True, "roughness": 0.10, "blend": True},
        "drum_shell": {"own": True, "roughness": 0.40, "blend": True},
        "liquid": {"own": True, "roughness": 0.20},
        "flaps": {"own": True, "roughness": 0.40},
        "default": {"own": True, "roughness": 0.50},
    },
    # fitted 2026-11-09 (14 states x 2 cameras of the seed-1001 reference shift against OVRTX 0.6, 256 samples,
    # HdrColor; a luminance gain per class, its colour held within 8 % of grey)
    "gains": {
        "dark": [0.4857, 0.4553, 0.4221], "drum_shell": [0.4858, 0.5033, 0.4702], "flaps": [1.1200, 1.1200, 1.1200],
        "frame": [1.0645, 1.1088, 1.1355], "glass": [0.6858, 0.6234, 0.5984], "lamp": [1.0808, 1.0989, 1.1228],
        "motor": [1.3977, 1.4334, 1.4516], "panda_black": [2.1647, 2.1236, 1.9476],
        "panda_offwhite": [0.7327, 0.6982, 0.6820], "panda_white": [1.0307, 1.1049, 1.1303],
        "panel": [1.0078, 1.0410, 1.0519], "part_dark": [1.1787, 1.0820, 1.1471],
        "part_light": [1.0869, 1.1140, 1.1404], "pipe": [1.0085, 0.9989, 0.9900], "plate": [1.3588, 1.5056, 1.5018],
        "pump": [1.5375, 1.3696, 1.3097], "rail_frame": [1.7588, 1.8762, 1.9456],
        "rail_mast": [1.0750, 0.9754, 0.9157], "rail_track": [0.9667, 0.9561, 0.9236],
        "skid_base": [1.6197, 1.6644, 1.6108], "steel": [0.9462, 0.9057, 0.8656], "tank": [0.6127, 0.6018, 0.5896],
        "tint": [0.9347, 1.0338, 1.0972], "valve": [0.6281, 0.7163, 0.6281], "wall": [0.9657, 1.0342, 1.1069]},
    "decor_gains": {
        "floor": [1.6372, 1.8053, 1.9220], "props": [0.6476, 0.5558, 0.5700]},
}
# the classes of the dynamic geoms (their state is copied from the physics model at every frame); those that switch
# colour carry one material each (Filament reads a material's colour, alpha and emission at every frame)
DYNAMIC = ("lamp", "segment", "glass", "liquid", "flaps")
OWN_MATERIAL = ("lamp", "segment", "glass")
MAT_CLASS = {"steel": "steel", "frame": "frame", "pipe": "pipe", "valve": "valve", "dark": "dark", "tank": "tank",
             "pump": "pump", "motor": "motor", "panel": "panel", "rubber": "rubber"}
PANDA_CLASS = {"white": "panda_white", "off_white": "panda_offwhite", "black": "panda_black", "dark_grey": "panda_black",
               "green": "panda_led_green", "light_blue": "panda_led_blue"}


class RenderError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ geom classes
def geom_class(m, g):
    """The look class of a cell geom (None: hidden in the RGB twin; "floor": the checker plane, which the room's floor
    replaces)."""
    name = m.geom(g).name or ""
    if int(m.geom_group[g]) >= 3:
        return None
    mat = m.mat(int(m.geom_matid[g])).name if m.geom_matid[g] >= 0 else ""
    body = m.body(int(m.geom_bodyid[g])).name or ""
    if body.startswith("panda/"):
        return PANDA_CLASS.get(mat.split("/")[-1], "panda_black")
    if name == "floor":
        return "floor"
    if name == "skid/wall":
        return "wall"
    if name == "skid/base":
        return "skid_base"
    if name.startswith("rail/"):
        return {"rail/track": "rail_track", "rail/mast": "rail_mast"}.get(name, "rail_frame")
    if mat.startswith("m/") and m.mat_texid[int(m.geom_matid[g])][1] >= 0:
        return "plate"
    if mat.startswith("mat/"):
        return MAT_CLASS.get(mat.split("/")[1], "default")
    if mat.startswith("m/"):
        return "tint"
    leaf = name.split("/")[-1]
    if leaf == "lens":
        return "lamp"
    if name.startswith("AI-201/d"):
        return "segment"
    if leaf in ("glass", "tube"):
        return "glass"
    if leaf == "shell":
        return "drum_shell"
    if leaf == "liquid":
        return "liquid"
    if leaf == "flaps":
        return "flaps"
    if leaf in ("mark", "pointer", "rail"):
        return "part_light"
    if leaf in ("blade", "bob", "cone"):
        return "part_red"
    if leaf in ("hubcap", "window"):
        return "part_dark"
    return "default"


def _lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


_TO_LIN = _lin(np.arange(256) / 255.0).astype(np.float32)
_TO_SRGB = np.round(255.0 * _srgb(np.arange(4096) / 4095.0)).astype(np.uint8)


def downsample(img, s):
    """uint8 (H*s)x(W*s)x3 -> HxWx3: the mean of each s x s block in linear light."""
    h, w = img.shape[0] // s, img.shape[1] // s
    lin = _TO_LIN[img].reshape(h, s, w, s, 3).mean(axis=(1, 3))
    return _TO_SRGB[np.clip(np.round(lin * 4095.0), 0, 4095).astype(np.int32)]


def look_rgba(rgba, gain):
    """The twin's rgba of a cell colour (sRGB rgba) under a class gain on its linear albedo."""
    lin = _lin(np.asarray(rgba[:3], float)) * np.asarray(gain, float)
    return [float(x) for x in _srgb(lin)] + [float(rgba[3])]


# ------------------------------------------------------------------------------------------------ assets
def _path(*parts):
    return os.path.join(RENDER_DIR, *parts)


def available():
    """(ok, reason): the render assets are present and match their manifest's digest."""
    try:
        _assets()
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, "%s: %s" % (type(e).__name__, e)


_ASSETS = {}


def _assets():
    """The render bundle's manifest, read once and checked file by file (sha256)."""
    if "manifest" in _ASSETS:
        return _ASSETS["manifest"]
    with open(_path("manifest.json")) as f:
        man = json.load(f)
    for rel, digest in man["files"].items():
        h = hashlib.sha256()
        with open(_path(rel), "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != digest:
            raise RenderError("render asset changed: " + rel)
    _ASSETS["manifest"] = man
    return man


_PNG = {}


def _png(rel, size=None):
    """uint8 HxWxC of a bundle PNG (cached per process)."""
    key = (rel, size)
    if key not in _PNG:
        from PIL import Image
        im = Image.open(_path(rel))
        im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") and "rgba" in rel else "RGB")
        if size is not None and max(im.size) > size:
            s = float(size) / max(im.size)
            im = im.resize((max(1, int(round(im.size[0] * s))), max(1, int(round(im.size[1] * s)))), Image.LANCZOS)
        _PNG[key] = np.asarray(im, np.uint8)
    return _PNG[key]


_NPZ = {}


def _npz(rel):
    if rel not in _NPZ:
        with np.load(_path(rel)) as z:
            _NPZ[rel] = {k: z[k] for k in z.files}
    return _NPZ[rel]


def _json(rel):
    if rel not in _ASSETS:
        with open(_path(rel)) as f:
            _ASSETS[rel] = json.load(f)
    return _ASSETS[rel]


def _add_mesh(spec, mats_rgba, body, name, v, n, uv, f, material, group=1):
    me = spec.add_mesh(name="mesh_" + name)
    me.uservert = np.asarray(v, np.float64).ravel().tolist()
    if n is not None:
        me.usernormal = np.asarray(n, np.float64).ravel().tolist()
    if uv is not None:
        me.usertexcoord = np.asarray(uv, np.float64).ravel().tolist()
    me.userface = np.asarray(f, np.int32).ravel().tolist()
    me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=me.name, contype=0, conaffinity=0,
                      density=0.0, group=group)
    if material:
        g.material = material
        g.rgba = mats_rgba.get(material, [1.0, 1.0, 1.0, 1.0])
    return g


def _quat_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _uv_meshes(spec, m):
    """Filament's physically based materials need texture coordinates and the Panda's STL meshes carry none: in the
    twin every such visible mesh is replaced by a copy with zero uv, its vertices and normals those of the compiled
    physics model taken back to the mesh file's frame (so the geoms keep their poses)."""
    users = {}
    for g in spec.geoms:
        if int(g.type) == int(mujoco.mjtGeom.mjGEOM_MESH) and int(g.group) < 3 and g.meshname:
            users.setdefault(g.meshname, []).append(g)
    n = 0
    for name, geoms in sorted(users.items()):
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, name)
        if mid < 0 or int(m.mesh_texcoordnum[mid]) > 0:
            continue
        va, fa, fnum = int(m.mesh_vertadr[mid]), int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        na = int(m.mesh_normaladr[mid])
        f = np.asarray(m.mesh_face[fa:fa + fnum], np.int64)
        fnr = np.asarray(m.mesh_facenormal[fa:fa + fnum], np.int64)
        R = _quat_mat(m.mesh_quat[mid])
        v = np.asarray(m.mesh_vert[va + f.ravel()], np.float64) @ R.T + np.asarray(m.mesh_pos[mid], np.float64)
        nn = np.asarray(m.mesh_normal[na + fnr.ravel()], np.float64) @ R.T
        key = np.concatenate([np.round(v * 1e7), np.round(nn * 1e4)], 1)
        u, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        first = np.zeros(len(u), np.int64)
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        me = spec.add_mesh(name="uv_" + name)
        me.uservert = v[first].ravel().tolist()
        me.usernormal = nn[first].ravel().tolist()
        me.usertexcoord = np.zeros(2 * len(u)).tolist()
        me.userface = inv.astype(np.int32).tolist()
        for g in geoms:
            g.meshname = me.name
        n += 1
    return n


# ------------------------------------------------------------------------------------------------ the twins
def _add_tex(spec, name, img, srgb=True):
    t = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=int(img.shape[1]), height=int(img.shape[0]),
                         nchannel=int(img.shape[2]))
    t.data = np.ascontiguousarray(img, np.uint8).tobytes()
    t.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_SRGB if srgb else mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR
    return name


_TEXMEAN = {}


def _tex_lin_mean(rel, img):
    """The texture's mean linear luminance (a class texture modulates its class colour around 1)."""
    if rel not in _TEXMEAN:
        lin = _lin(img[..., :3])
        _TEXMEAN[rel] = float(max(1e-3, (lin @ np.array([0.2126, 0.7152, 0.0722])).mean()))
    return _TEXMEAN[rel]


def _class_material(spec, mats_rgba, name, c, rgb_srgb, gain, tex_px=256, alpha=None):
    """One material: a colour (the class's, or the cell's own for "own" classes) under the class gain, untextured or
    modulating the class texture (in its data)."""
    lin = _lin(np.asarray(rgb_srgb[:3], float)) * np.asarray(gain, float)
    mat = spec.add_material(name=name)
    mat.roughness = float(c.get("roughness", 0.5))
    mat.metallic = float(c.get("metallic", 0.0))
    mat.specular = 0.5
    mat.reflectance = 0.0
    if alpha is None:
        alpha = 1.0
    alpha = min(0.99, float(alpha)) if c.get("blend") else 1.0   # below 1: Filament's blended path, with this alpha
    if c.get("tex"):
        img = _png("tex/" + c["tex"], tex_px).astype(np.float64) / 255.0
        k = 1.0 / _tex_lin_mean(c["tex"], img)
        data = np.round(255.0 * _srgb(_lin(img[..., :3]) * lin[None, None, :] * k)).astype(np.uint8)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _add_tex(spec, "tex_" + name, data)
        mat.texuniform = True
        mat.texrepeat = [1.0 / float(c["tile"]), 1.0 / float(c["tile"])]
        mat.rgba = [1.0, 1.0, 1.0, alpha]
    else:
        mat.rgba = [float(x) for x in _srgb(lin)] + [alpha]   # (a gain past full value is clipped)
    mats_rgba[mat.name] = [float(x) for x in mat.rgba]
    return mat.name


def _stretch_wall(spec):
    """The skid's back wall carried on as the room's lining wall (same front face y = 1.00)."""
    g = spec.geom("skid/wall")
    x0, x1 = WALL["x"]
    g.size = [0.5 * (x1 - x0), float(g.size[1]), 0.5 * WALL["z_top"]]
    g.pos = [0.5 * (x0 + x1), float(g.pos[1]), 0.5 * WALL["z_top"]]


def _plate_textures(spec, gain):
    """The cell's own plate textures (tags, dials, scales, labels) marked sRGB, with the plate gain in their data."""
    n = 0
    for t in spec.textures:
        if not (t.name or "").startswith("t/"):
            continue
        h, w, c = int(t.height), int(t.width), int(t.nchannel)
        img = np.frombuffer(bytes(t.data), np.uint8).reshape(h, w, c).astype(np.float64) / 255.0
        out = np.round(255.0 * _srgb(_lin(img[..., :3]) * np.asarray(gain, float)[None, None, :])).astype(np.uint8)
        if c == 4:
            out = np.concatenate([out, np.round(img[..., 3:] * 255.0).astype(np.uint8)], 2)
        t.data = np.ascontiguousarray(out).tobytes()
        t.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_SRGB
        n += 1
    return n


def decor_class(me, mat):
    """The look class of a decor mesh: the lamps and the panels' diffusers (emissive), the room's floor, the rest of
    the room, the panels' housings and wires, the props."""
    if mat.get("emission"):
        return "lamps"
    if "Concrete" in mat.get("source", ""):
        return "floor"
    if me["item"] == "Hall":
        return "hall"
    if me["item"].startswith("fixture_"):
        return "fixtures"
    return "props"


def _add_decor(spec, mats_rgba, textures=True, near_m=None):
    """The room and its props (arb-decor/1) as static world meshes (drawn only)."""
    dec = _json("decor/decor.json")
    mats = {}
    nf = 0
    gains = LOOK.get("decor_gains", {})
    cx, cy = SKID_CENTRE
    for me in dec["meshes"]:
        if near_m is not None:
            lo, hi = np.asarray(me["bbox"][0], float), np.asarray(me["bbox"][1], float)
            c = np.array([cx, cy, 0.0])
            gap = np.maximum(0.0, np.maximum(lo - c, c - hi))   # distance of the box from the skid centre per axis
            if float(np.linalg.norm(gap[:2])) > near_m:
                continue
        mk = me["material"]
        if textures and mk not in mats:
            m = dec["materials"][mk]
            mat = spec.add_material(name="decor_" + mk)
            g = np.asarray(gains.get(decor_class(me, m), [1.0, 1.0, 1.0]), float)
            rgba = np.asarray(m["rgba"][:3], float)
            metal = float(m.get("metallic_eff", m.get("metallic", 0.0))) >= 0.5
            rough = float(m.get("roughness_eff", m.get("roughness", 0.5)))
            mat.roughness = float(np.clip(rough if (metal or decor_class(me, m) == "floor") else max(rough, 0.6), 0.05, 1.0))
            mat.metallic = 1.0 if metal else 0.0
            if m.get("cutout"):
                mat.roughness, mat.metallic = 0.6, 1.0
            mat.specular, mat.reflectance = 0.5, 0.0
            if m.get("rgb"):
                img = _png("decor/textures/" + m["rgb"])
                rgb = np.round(255.0 * _srgb(_lin(img[..., :3] / 255.0) * (rgba * g)[None, None, :])).astype(np.uint8)
                cut = bool(m.get("cutout") and img.shape[2] == 4)
                if cut:
                    a = img[..., 3:4].astype(np.float64) / 255.0
                    rgb = np.concatenate([np.round(rgb * a).astype(np.uint8), img[..., 3:4]], 2)
                mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _add_tex(spec, "decor_tex_" + mk, rgb)
                mat.rgba = [1.0, 1.0, 1.0, 0.99 if cut else 1.0]
            else:
                lin = _lin(rgba) * g
                mat.rgba = [float(x) for x in _srgb(lin)] + [1.0]
            if m.get("emission"):
                mat.emission = float(min(4.0, m["emission"]))
            mats_rgba[mat.name] = [float(x) for x in mat.rgba]
            mats[mk] = mat.name
        z = _npz("decor/" + me["file"])
        _add_mesh(spec, mats_rgba, spec.worldbody, me["name"], z["v"], z["n"] if textures else None,
                  z["uv"] if textures else None, z["f"], mats.get(mk), group=2)
        nf += int(len(z["f"]))
    return nf


def _own_rgba(m, g):
    """The cell's own colour of a geom: its material's rgba where the geom carries MuJoCo's default rgba."""
    rgba = np.asarray(m.geom_rgba[g], float)
    if m.geom_matid[g] >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        rgba = np.asarray(m.mat_rgba[int(m.geom_matid[g])], float)
    return rgba


def build_rgb_spec(scene, light=1.0):
    """The RGB twin's MjSpec and a record (per named geom its class; the dynamic geoms' own materials)."""
    spec = scene.spec.copy()
    m = scene.model
    classes = LOOK["classes"]
    gains = LOOK.get("gains", {})
    rec = {"classes": {}, "geom_class": {}, "dyn_mat": {}}
    mats_rgba = {}
    for li in list(spec.lights):
        spec.delete(li)
    spec.visual.headlight.active = 0
    _stretch_wall(spec)
    rec["uv_meshes"] = _uv_meshes(spec, m)
    rec["plate_textures"] = _plate_textures(spec, gains.get("plate", [1.0, 1.0, 1.0]))
    mats = {}
    by_name = {g.name: g for g in spec.geoms if g.name}
    for gi in range(m.ngeom):
        name = m.geom(gi).name
        sg = by_name.get(name) if name else None
        if sg is None:
            continue                             # unnamed: the Panda's meshes and collision hulls (below)
        cls = geom_class(m, gi)
        if cls == "floor":
            spec.delete(sg)                      # the checker plane (Filament draws a plane whatever its alpha)
            continue
        if cls is None:
            sg.rgba = [0.0, 0.0, 0.0, 0.0]
            sg.material = ""
            continue
        c = classes.get(cls, classes["default"])
        own = _own_rgba(m, gi)
        if cls == "plate":
            mn = m.mat(int(m.geom_matid[gi])).name
            sm = spec.material(mn)
            sm.roughness, sm.metallic, sm.specular, sm.reflectance = float(c["roughness"]), 0.0, 0.5, 0.0
            sm.rgba = [1.0, 1.0, 1.0, 1.0]
            sg.rgba = [1.0, 1.0, 1.0, 1.0]
            sg.material = mn
        else:
            if cls in OWN_MATERIAL:
                key = "look_%s_%s" % (cls, name.replace("/", "_"))      # one material per geom (colour, glow, alpha)
            elif c.get("own"):
                key = "look_%s_%s" % (cls, "_".join("%.3f" % x for x in own[:3]))
            else:
                key = "look_" + cls
            if key not in mats:
                mats[key] = _class_material(spec, mats_rgba, key, c, own if c.get("own") else c["rgb"], gains.get(cls, [1, 1, 1]),
                                            alpha=float(own[3]))
            sg.material = mats[key]
            sg.rgba = list(mats_rgba[mats[key]][:3]) + [float(own[3]) if c.get("own") else float(mats_rgba[mats[key]][3])]
            if cls in OWN_MATERIAL:
                rec["dyn_mat"][name] = key
        rec["geom_class"][name] = cls
        rec["classes"][cls] = rec["classes"].get(cls, 0) + 1
    pmat = {}
    for gs in spec.geoms:
        if gs.name:
            continue
        if int(gs.group) >= 3:
            gs.rgba = [0.0, 0.0, 0.0, 0.0]
            gs.material = ""
            continue
        cls = PANDA_CLASS.get((gs.material or "").split("/")[-1])
        if cls is None:
            continue
        if cls not in pmat:
            pmat[cls] = _class_material(spec, mats_rgba, "look_" + cls, classes[cls], classes[cls]["rgb"], gains.get(cls, [1, 1, 1]))
        gs.material = pmat[cls]
        gs.rgba = mats_rgba[pmat[cls]]
        rec["classes"][cls] = rec["classes"].get(cls, 0) + 1
    rec["decor_faces"] = _add_decor(spec, mats_rgba, textures=True)
    for L in LOOK["lamps"]:
        li = spec.worldbody.add_light(name="look_" + L["name"], type=mujoco.mjtLightType.mjLIGHT_SPOT)
        li.pos, li.dir = [float(x) for x in L["pos"]], [float(x) for x in L["dir"]]
        li.cutoff, li.exponent = float(L["cutoff"]), float(L["exponent"])
        li.diffuse = [float(x) for x in L["color"]]
        li.specular, li.ambient = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        li.intensity = max(1e-4, float(L["intensity"]) * float(light))   # (0 would be MuJoCo's legacy intensity)
        li.bulbradius = float(L["bulb_m"])
        li.range = float(L["range"])
        li.castshadow = bool(L["shadow"])
    for k, v in sorted(LOOK["view"]["numeric"].items()):
        spec.add_numeric(name=k, data=[float(v)])
    for k, v in sorted(LOOK["view"]["text"].items()):
        spec.add_text(name=k, data=str(v))
    return spec, rec


def build_depth_spec(scene):
    """The depth twin's MjSpec: the cell as the physics model draws it (the back wall carried on) plus the room's and
    the props' meshes near the skid."""
    spec = scene.spec.copy()
    _stretch_wall(spec)
    _add_decor(spec, {}, textures=False, near_m=DEPTH_DECOR_M)
    return spec


# ------------------------------------------------------------------------------------------------ the renderer thread
class _View:
    """One RGB twin in the process's renderer; its Filament objects stay in the renderer's thread."""

    def __init__(self, fil, model, ibl_intensity):
        self._fil = fil
        self._sid = fil.call(fil.new_session, model, ibl_intensity)
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
        self.models = {}
        self.active = None
        self.live = []
        self.targets = {}
        self.next_sid = 0
        self.thread = threading.Thread(target=self._loop, name="skidcell-filament", daemon=True)
        self.thread.start()
        self.call(self._start)

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
                res = (False, RenderError("renderer: %s: %s" % (type(exc).__name__, exc)))  # hold Filament objects)
            if done is not None:
                out.append(res)
                done.set()

    # -- the renderer's thread only
    def _start(self):
        for k in ("DISPLAY", "WAYLAND_DISPLAY"):       # offscreen: Filament's resource providers would connect to X
            os.environ.pop(k, None)
        try:
            ncpu = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            ncpu = os.cpu_count() or 4
        # lavapipe's rasteriser threads: the CPUs this process may use, at most 8 (the image's LP_NUM_THREADS=2 is the
        # classic renderer's; the program waits for its frame, so the renderer may use them all meanwhile)
        os.environ["LP_NUM_THREADS"] = str(max(1, min(8, ncpu)))
        import mujoco._render_filament as F
        from mujoco.experimental.studio import window  # noqa: F401  registers Filament's resource providers
        self.F = F
        path = _path(LOOK["ibl"]["file"])
        self.ctx = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN))
        self.ctx.set_clear_color([0.0, 0.0, 0.0])
        with open(path, "rb") as fh:
            raw = fh.read()
        tc = F.TextureConfig()
        tc.width, tc.height, tc.format = len(raw), 0, F.PixelFormat.PIXEL_FORMAT_KTX
        tc.sampler_type, tc.color_space = F.SamplerType.TEXTURE_CUBE, F.ColorSpace.COLORSPACE_LINEAR
        self.ibl_tex = self.ctx.create_texture(tc)
        self.ibl_td = F.TextureData()
        self.ibl_td.set_image(raw)
        self.ibl_tex.set_data(self.ibl_td)

    def new_session(self, model, ibl_intensity):
        sid = self.next_sid
        self.next_sid += 1
        self.models[sid] = (model, float(ibl_intensity))
        return sid

    def free_session(self, sid):
        self.models.pop(sid, None)
        if self.active == sid:
            self._evict()

    def _evict(self):
        if self.live:
            rends, lights, ibl, scn, objs = self.live
            try:
                scn.remove_light(ibl)
            except Exception:  # noqa: BLE001
                pass
            self.live = []
            del rends, lights, ibl, scn, objs
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
        model, ibl_intensity = self.models[sid]
        self._evict()
        F = self.F
        objs = F.ModelObjects(self.ctx, model)
        scn = self.ctx.create_scene(F.SceneParams())
        scn.configure_from_model(model)
        lights = F.ModelLights(scn, objs)
        rends = F.ModelRenderables(scn, objs)
        lp = F.LightParams()
        lp.type, lp.texture, lp.color = F.LightType.LIGHT_IMAGE, self.ibl_tex, [1.0, 1.0, 1.0]
        lp.intensity = ibl_intensity
        ibl = self.ctx.create_light(lp)
        scn.add_light(ibl)
        self.live = [rends, lights, ibl, scn, objs]
        self.active = sid

    def _target(self, width, height):
        F = self.F
        key = (width, height)
        if key not in self.targets:     # one target and one read request per size (a request per frame leaks)
            tgt = self.ctx.create_render_target(F.RenderTargetConfig(color_format=F.PixelFormat.PIXEL_FORMAT_RGB8))
            tgt.resize(width, height)
            rd = F.ReadPixelsRequest()
            rd.target = tgt
            rd.alloc(width * height * 3)
            self.targets[key] = (tgt, rd)
        return self.targets[key]

    def draw(self, sid, data, pos, forward, up, fovy, width, height):
        """RGB (rows top-down) of the session's scene in the state `data`, from an exact pinhole camera."""
        F = self.F
        self._activate(sid)
        rends, lights, _ibl, scn, _objs = self.live
        lights.update(data)
        rends.update(data)
        tgt, rd = self._target(width, height)
        top = CAM_NEAR * math.tan(math.radians(fovy) / 2.0)
        cam = F.Camera(pos=[float(x) for x in pos], forward=[float(x) for x in forward], up=[float(x) for x in up],
                       frustum_center=0.0, frustum_width=top * width / height, frustum_bottom=-top, frustum_top=top,
                       frustum_near=CAM_NEAR, frustum_far=CAM_FAR)
        req = F.RenderRequest(camera=cam, draw_mode=F.DrawMode.DRAW_MODE_DEFAULT)
        req.scene, req.target = scn, tgt
        req.viewport.left, req.viewport.bottom, req.viewport.width, req.viewport.height = 0, 0, width, height
        self.ctx.wait_for_frame(self.ctx.render([req], [rd]))
        return np.frombuffer(rd.buffer(), np.uint8).reshape(height, width, 3).copy()

    def _shutdown(self):
        """Every Filament object before the context (destroying the context first panics)."""
        self.models.clear()
        self._evict()
        self.targets.clear()
        self.ibl_tex = None
        self.ibl_td = None
        gc.collect()
        self.ctx = None
        gc.collect()


@atexit.register
def _stop_renderer():
    if _Filament._inst is not None:
        _Filament._inst.stop()


# ------------------------------------------------------------------------------------------------ the cell's look
class AgentLook(object):
    """The cell's agent cameras: Filament colour on the RGB twin, classic depth on the depth twin, both built at the
    episode's first frame; the instruments' changing parts follow the physics model at every frame."""

    def __init__(self, scene, sizes):
        self.scene = scene
        self.sizes = dict(sizes)
        self.view = None
        self.rgb_model = self.rgb_data = None
        self.depth_model = self.depth_data = None
        self._depth_renderers = {}
        self._option = mujoco.MjvOption()
        self._option.geomgroup[4] = 0
        self.record = {}
        self.dyn = []
        self._rgba_cache = {}

    def _light(self):
        try:
            return float(self.scene.L.get("light", 1.0))
        except Exception:  # noqa: BLE001
            return 1.0

    def _on_colours(self):
        """{geom name: its lit rgba} for the lamp lenses and the analyser's segments."""
        dev = self.scene.dev
        out = {}
        for tag, d in dev.items():
            if d.get("type") == "lamp" and d.get("lens"):
                out[d["lens"]] = d["rgba_on"]
        ai = dev.get("AI-201", {})
        for dig in ai.get("segments", []):
            for g in dig.values():
                out[g] = ai["rgba_on"]
        if ai.get("dp"):
            out[ai["dp"]] = ai["rgba_on"]
        return out

    def _ensure(self):
        if self.view is not None:
            return
        _assets()
        t0 = time.time()
        light = self._light()
        spec, rec = build_rgb_spec(self.scene, light)
        self.rgb_model = spec.compile()
        self.rgb_data = mujoco.MjData(self.rgb_model)
        self.depth_model = build_depth_spec(self.scene).compile()
        self.depth_data = mujoco.MjData(self.depth_model)
        pm = self.scene.model
        for tw in (self.rgb_model, self.depth_model):
            if tw.nq != pm.nq or tw.ncam != pm.ncam:
                raise RenderError("render twin out of step with the cell (nq %d / %d)" % (tw.nq, pm.nq))
        # the dynamic geoms: (physics id, rgb twin id, depth twin id, class, lit rgba or None, rgb material id, gain,
        # the rgb twin's compiled size)
        on = self._on_colours()
        classes = LOOK["classes"]
        gains = LOOK.get("gains", {})
        self.dyn = []
        for name, cls in rec["geom_class"].items():
            if cls not in DYNAMIC:
                continue
            pg = pm.geom(name).id
            rg = self.rgb_model.geom(name).id
            dg = self.depth_model.geom(name).id
            mid = int(self.rgb_model.geom_matid[rg])
            self.dyn.append((pg, rg, dg, cls, on.get(name), mid, tuple(gains.get(cls, [1.0, 1.0, 1.0])),
                             self.rgb_model.geom_size[rg].copy(), float(classes[cls].get("emission_on", 0.0)),
                             int(self.rgb_model.geom_type[rg])))
        self.view = _View(_Filament.instance(), self.rgb_model, float(LOOK["ibl"]["intensity"]) * light)
        rec["build_s"] = round(time.time() - t0, 3)
        rec["light"] = light
        self.record = rec

    def _look_rgba(self, rgba, gain):
        key = (rgba, gain)
        v = self._rgba_cache.get(key)
        if v is None:
            v = look_rgba(rgba, gain)
            self._rgba_cache[key] = v
        return v

    def _sync(self, data):
        pm = self.scene.model
        rm, dm = self.rgb_model, self.depth_model
        for pg, rg, dg, cls, on, mid, gain, _s0, emis, _t in self.dyn:
            dm.geom_size[dg] = pm.geom_size[pg]
            dm.geom_pos[dg] = pm.geom_pos[pg]
            dm.geom_rgba[dg] = pm.geom_rgba[pg]
            rm.geom_pos[rg] = pm.geom_pos[pg]
            rgba = tuple(round(float(x), 6) for x in pm.geom_rgba[pg])
            col = self._look_rgba(rgba, gain)
            rm.geom_rgba[rg] = col
            if cls in OWN_MATERIAL and mid >= 0:
                # Filament draws a geom with a material in the material's colour, read at every frame: the geom's own
                # material follows the cell's colour (and, for the glass, its alpha: the blended path's)
                rm.mat_rgba[mid][:3] = col[:3]
                if cls == "glass":
                    rm.mat_rgba[mid][3] = min(0.99, col[3])
            if on is not None and mid >= 0:
                lit = bool(np.allclose(rgba[:3], on[:3], atol=1e-4))
                rm.mat_emission[mid] = emis if lit else 0.0
        for tw, td in ((rm, self.rgb_data), (dm, self.depth_data)):
            td.qpos[:] = data.qpos
            mujoco.mj_kinematics(tw, td)
            mujoco.mj_camlight(tw, td)
        rd = self.rgb_data
        for pg, rg, _dg, cls, _on, _mid, _gain, s0, _e, gtype in self.dyn:
            if cls not in ("flaps", "liquid"):
                continue
            s = pm.geom_size[pg]
            if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                r = np.array([s[0] / s0[0], s[0] / s0[0], s[1] / s0[1]])
            else:
                r = np.asarray(s, float) / np.maximum(np.asarray(s0, float), 1e-9)
            rd.geom_xmat[rg] = (rd.geom_xmat[rg].reshape(3, 3) * r[None, :]).ravel()

    def _depth(self, name):
        w, h = self.sizes[name]
        key = (int(w), int(h))
        r = self._depth_renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.depth_model, height=key[1], width=key[0])
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            r.enable_depth_rendering()
            self._depth_renderers[key] = r
        r.update_scene(self.depth_data, camera=name, scene_option=self._option)
        return r.render().astype(np.float32)

    def rgb(self, data, name):
        """The camera's colour image (uint8 HxWx3, rows top-down) at the physics state `data`, no sensor noise."""
        self._ensure()
        self._sync(data)
        w, h = self.sizes[name]
        cid = self.rgb_model.camera(name).id
        R = self.rgb_data.cam_xmat[cid].reshape(3, 3)
        s = max(1, int(LOOK.get("ssaa", 1)))
        img = self.view.render(self.rgb_data, self.rgb_data.cam_xpos[cid], -R[:, 2], R[:, 1],
                               float(self.rgb_model.cam_fovy[cid]), int(w) * s, int(h) * s)
        return img if s == 1 else downsample(img, s)

    def render(self, data, name):
        """(rgb uint8 HxWx3, depth float32 HxW planar z in metres) of camera `name` at the physics state `data`."""
        rgb = self.rgb(data, name)
        return rgb, self._depth(name)

    def close(self):
        if self.view is not None:
            try:
                self.view.close()
            except Exception:  # noqa: BLE001
                pass
            self.view = None
        self.rgb_model = self.rgb_data = None
        for r in self._depth_renderers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        self._depth_renderers = {}
        self.depth_model = self.depth_data = None
