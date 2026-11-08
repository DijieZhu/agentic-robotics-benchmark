"""The agent cameras' colour images (PUBLIC CODE): MuJoCo's Filament renderer (physically based, on Vulkan: Mesa's
lavapipe on the CPU in the sandbox) on a render twin of the cell, the cell's depth from MuJoCo's classic renderer on a
depth twin with the same geometry.

The physics model is never touched. Per episode (and again when a new pair's colours arrive) two twins are compiled
from a copy of the cell's MjSpec, with the same bodies, joints and cameras, so a frame copies the physics qpos into the
twin and runs forward kinematics only:

* the RGB twin: the cell's parts with physically based materials (the bench's oak top, powder-coated legs, brushed
  steel tray, kraft and white-lined board for the blank and the lid, the Pandas' paint), a sneaker skin dressed onto
  each shoe body (render only: sole, knit and leather upper in the pair's colours, white side wing, lining, metal
  grommets at the task's eyelet rings, stitching; warped onto the batch's shoe so that every ring, the toe cap and the
  heel block keep their place and size), the hall the cell stands in with its props (static, drawn only; nothing of
  it stands on the bench), the cell's LED panel 2.05 m above the bench as a shadow-casting spot light and an image
  light distilled from path-traced (OVRTX) renders of the same hall (the probe at the work area with the cell's live
  objects hidden; the panel's own pixels are the spot's). The shoes' physics shells, the rings and the collision geoms
  are hidden in this twin.
* the depth twin: the cell exactly as the physics model draws it (groups 0-2) plus the hall's and the props' meshes,
  so depth stays registered with the colour image; the sensor model (noise, dropout, range) is the server's.

The light strengths and per-class colour gains were fitted, in linear units, to OVRTX stills of the five agent
cameras on states of a reference episode (LOOK below). Filament facts this relies on (MuJoCo 3.14): the engine is
bound to the thread that created it, so one thread owns the context and every Filament object and callers post jobs
to it; one model's Filament objects are live at a time; its job threads would each get a glibc malloc arena, so the
arenas are capped at import; textures are marked sRGB (Filament reads MuJoCo's "auto" as linear data); Filament draws
every geom group, so hidden geoms get alpha 0.

Assets: $SHOECELL_RENDER_DIR (default /opt/task-assets/render; the task image fetches them, see the Dockerfile):
manifest.json, decor/ (the hall and props: arb-decor/1), skin/ (the skin template), tex/ (the cell's textures),
ibl.ktx. Runtime dependencies: mujoco (with its Filament module), numpy, Pillow.
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

RENDER_DIR = os.environ.get("SHOECELL_RENDER_DIR", "/opt/task-assets/render")
CAM_NEAR, CAM_FAR = 0.005, 120.0       # m: Filament's clip planes (the hall's far wall is ~40 m out)
CALL_TIMEOUT = 300.0                    # s: one job of the renderer's thread
DEPTH_DECOR_M = 4.5                     # decor meshes reaching within this of the bench centre enter the depth twin

# The look: Filament view options, the lights, the image light, every class's material (base colour sRGB, an optional
# texture and its tile, roughness, metallic) and its fitted colour gain (r, g, b on the linear albedo). "tint": "upper",
# "sole", "lace" or "logo" take the pair's colour (the physics model's geom rgba at delivery).
LOOK = {
    "view": {"numeric": {"filament.ao.enabled": 0, "filament.shadows.type": 3, "filament.shadows.map_size": 1024},
             "text": {"filament.cg.tone_mapping": "filmic"}},
    "ibl": {"file": "ibl.ktx", "intensity": 150000.0},
    # the LED panel (1.23 x 0.30 m, 4800 K): 93 000 cd fitted, split into a shadow-casting part (45 %) and a shadow-free
    # rest, as a panel's soft shadow is about half as deep as a spot's
    "lamps": [{"name": "cell_lamp", "pos": [-0.05, 0.0, 2.04], "dir": [0.0, 0.0, -1.0], "intensity": 41850.0,
               "color": [1.0, 0.745, 0.57], "cutoff": 85.0, "exponent": 1.0, "bulb_m": 0.5, "range": 30.0,
               "shadow": True},
              {"name": "cell_lamp_fill", "pos": [-0.05, 0.0, 2.04], "dir": [0.0, 0.0, -1.0], "intensity": 51150.0,
               "color": [1.0, 0.745, 0.57], "cutoff": 85.0, "exponent": 1.0, "bulb_m": 0.5, "range": 30.0,
               "shadow": False}],
    "classes": {
        "panda_white": {"rgb": [0.92, 0.92, 0.91], "roughness": 0.36},
        "panda_offwhite": {"rgb": [0.84, 0.85, 0.85], "roughness": 0.40},
        "panda_black": {"rgb": [0.09, 0.09, 0.10], "roughness": 0.95},
        "panda_led_green": {"rgb": [0.20, 0.90, 0.30], "roughness": 0.30},
        "panda_led_blue": {"rgb": [0.10, 0.50, 0.90], "roughness": 0.30},
        "bench_wood": {"rgb": [1.0, 1.0, 1.0], "tex": "oak_col.png", "tile": 0.8, "neutral": False, "roughness": 0.62},
        "bench_leg": {"rgb": [0.20, 0.21, 0.22], "tex": "powder_col.png", "tile": 0.3, "roughness": 0.95},
        "pedestal": {"rgb": [0.50, 0.51, 0.52], "tex": "powder_col.png", "tile": 0.3, "roughness": 0.90},
        "floor_tape": {"rgb": [0.98, 0.78, 0.05], "roughness": 0.45},
        "park_shelf": {"rgb": [0.70, 0.71, 0.72], "tex": "alu_rough_col.png", "tile": 0.6, "roughness": 0.45,
                       "metallic": 1.0},
        "hook_flange": {"rgb": [0.22, 0.22, 0.23], "roughness": 0.80, "metallic": 1.0},
        "hook_steel": {"rgb": [0.80, 0.80, 0.79], "roughness": 0.22, "metallic": 1.0},
        "hook_tip": {"rgb": [0.95, 0.45, 0.08], "roughness": 0.30, "metallic": 1.0},
        "tray_plate": {"rgb": [0.86, 0.87, 0.88], "tex": "brushed_col.png", "tile": 0.5, "roughness": 0.40,
                       "metallic": 1.0},
        "tray_rim": {"rgb": [0.72, 0.73, 0.74], "tex": "brushed_col.png", "tile": 0.5, "roughness": 0.42,
                     "metallic": 1.0},
        "cradle_plate": {"rgb": [0.40, 0.41, 0.43], "tex": "alu_rough_col.png", "tile": 0.3, "roughness": 0.45,
                         "metallic": 1.0},
        "cradle_bumper": {"rgb": [0.05, 0.05, 0.05], "roughness": 0.95},
        "lace": {"tint": "lace", "roughness": 0.70},
        "aglet": {"rgb": [0.83, 0.83, 0.81], "roughness": 0.20, "metallic": 1.0},
        "carton_outer": {"rgb": [1.0, 1.0, 1.0], "tex": "kraft_paper_col.png", "tile": 0.3, "neutral": False,
                         "roughness": 0.80},
        "carton_inner": {"rgb": [0.93, 0.89, 0.82], "tex": "paper_white_col.png", "tile": 0.3, "neutral": False,
                         "roughness": 0.85},
        "lid_board": {"rgb": [1.0, 1.0, 1.0], "tex": "kraft_paper_col.png", "tile": 0.3, "neutral": False,
                      "roughness": 0.80},
        "lid_logo": {"tint": "logo", "roughness": 0.38},
        "default": {"rgb": [0.60, 0.60, 0.60], "roughness": 0.50},
        # the sneaker skin
        "sole": {"tint": "sole", "tex": "rubber_col.png", "uv": True, "roughness": 0.95},
        "knit": {"tint": "upper", "tex": "knit_col.png", "uv": True, "desat": 0.12, "roughness": 0.95},
        "leather": {"tint": "upper", "tint_scale": 0.93, "tex": "leather_col.png", "uv": True, "desat": 0.12,
                    "roughness": 0.80},
        "wing": {"rgb": [0.93, 0.92, 0.89], "tex": "leather_col.png", "uv": True, "roughness": 0.50},
        "lining": {"rgb": [0.16, 0.16, 0.17], "tex": "knit_col.png", "uv": True, "roughness": 1.0},
        "insole": {"rgb": [0.62, 0.58, 0.52], "roughness": 0.80},
        "thread": {"rgb": [0.90, 0.88, 0.82], "roughness": 0.75},
        "grommet": {"rgb": [0.80, 0.79, 0.76], "roughness": 0.28, "metallic": 1.0},
    },
    # fitted 2026-11-08 (12 states x 5 cameras of the seed-1046 reference episode against OVRTX 0.6, 256 samples,
    # HdrColor; a luminance gain per class, its colour held within 8 % of grey)
    "gains": {
        "aglet": [0.1107, 0.1059, 0.1165], "bench_leg": [0.0425, 0.0485, 0.0499],
        "bench_wood": [0.9890, 0.9914, 0.9693], "carton_inner": [0.8301, 0.8664, 0.8729],
        "carton_outer": [0.1666, 0.1746, 0.1576], "cradle_bumper": [1.4479, 1.6095, 1.6997],
        "cradle_plate": [0.9991, 1.0091, 1.0222], "floor_tape": [1.0808, 0.9961, 0.9231],
        "grommet": [0.5071, 0.5460, 0.5662], "hook_flange": [0.1243, 0.1401, 0.1459],
        "hook_steel": [0.0840, 0.0933, 0.0986], "hook_tip": [0.0782, 0.0671, 0.0671],
        "insole": [0.2475, 0.2228, 0.2108], "knit": [0.2919, 0.2506, 0.2506], "lace": [0.4370, 0.4447, 0.4671],
        "leather": [0.6947, 0.6275, 0.6434], "lid_board": [0.9359, 0.9645, 0.9929],
        "lid_logo": [1.5257, 1.2997, 1.5257], "lining": [0.0200, 0.0200, 0.0200],
        "panda_black": [0.5931, 0.6651, 0.6962], "panda_offwhite": [0.1335, 0.1490, 0.1567],
        "panda_white": [0.4055, 0.4504, 0.4760], "park_shelf": [0.8835, 0.8854, 0.8976],
        "pedestal": [0.3809, 0.4303, 0.4472], "sole": [0.0367, 0.0416, 0.0431], "thread": [0.5676, 0.5756, 0.5781],
        "tray_plate": [0.7020, 0.6690, 0.6481], "tray_rim": [0.1651, 0.1817, 0.1938]},
    "decor_gains": {
        "fence_racks": [0.2500, 0.2500, 0.2500], "floor": [0.5661, 0.5695, 0.5646], "hall": [0.0243, 0.0273, 0.0285],
        "lamps": [1.0000, 1.0000, 1.0000], "props": [2.1074, 2.4258, 2.4258]},
}
TINTS = ("upper", "sole", "lace", "logo")


class RenderError(RuntimeError):
    pass


def _lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


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


# ------------------------------------------------------------------------------------------------ geom classes
def geom_class(m, g):
    """The look class of a cell geom (None: hidden in the RGB twin)."""
    name = m.geom(g).name or ""
    body = m.body(int(m.geom_bodyid[g])).name or ""
    if int(m.geom_group[g]) >= 3:
        return None
    if body.startswith("shoe") and body.endswith("/body"):
        return None                               # the shell, its rings and seams: the skin replaces them
    mat = m.mat(int(m.geom_matid[g])).name if m.geom_matid[g] >= 0 else ""
    if mat:
        k = mat.split("/")[-1]
        if k == "white":
            return "panda_white"
        if k == "off_white":
            return "panda_offwhite"
        if k in ("black", "dark_grey"):
            return "panda_black"
        if k == "green":
            return "panda_led_green"
        if k == "light_blue":
            return "panda_led_blue"
        if k == "wood":
            return "bench_wood"
        if k == "floor":
            return None                           # the checker plane: the hall's floor replaces it
    if name.startswith("bench/leg"):
        return "bench_leg"
    if name.startswith("pedestal_"):
        return "pedestal"
    if name == "outfeed/line":
        return "floor_tape"
    if name == "park/shelf":
        return "park_shelf"
    if name == "right/hook/flange":
        return "hook_flange"
    if name in ("right/hook/shaft", "right/hook/foot"):
        return "hook_steel"
    if name == "right/hook/tip":
        return "hook_tip"
    if name == "infeed/plate":
        return "tray_plate"
    if name.startswith("infeed/rim"):
        return "tray_rim"
    if name == "cradle/plate":
        return "cradle_plate"
    if name.startswith("cradle/"):
        return "cradle_bumper"
    if name.startswith("lace") and "/g" in name:
        return "aglet" if m.geom_size[g][0] > 0.0016 else "lace"
    if name.startswith("box/") and name.endswith("_panel"):
        return "carton_outer"
    if name.startswith("box/") and name.endswith("_inner"):
        return "carton_inner"
    if name == "lid/logo":
        return "lid_logo"
    if name.startswith("lid/"):
        return "lid_board"
    return "default"


def pair_colours(scene):
    """The pair's colours as the physics model holds them (sRGB rgb of the shoe upper / sole, the lace, the logo)."""
    m = scene.model
    out = {}
    for key, geoms in (("upper", scene.shoe_upper_geoms[0]), ("sole", scene.shoe_sole_geoms[0]),
                       ("lace", scene.lace_cord_geoms[0]), ("logo", [scene.lid_logo_geom])):
        g = int(geoms[0]) if len(geoms) else -1
        out[key] = [float(x) for x in m.geom_rgba[g][:3]] if g >= 0 else [0.5, 0.5, 0.5]
    return out


# ------------------------------------------------------------------------------------------------ the skin
SKIN_RING_HALF = 0.0076      # m: unit slope this far round every ring (outer radius 7.37 mm) in x
SKIN_HOLE_HALF = 0.005       # m: ... and round the eyelet line in y (the holes' radius is 4.8 mm)


def _knots(d):
    x0, x1 = float(d["x_rows"][0]), float(d["x_rows"][-1])
    L = float(d["L"])
    kx = [-L / 2, float(d["x_hc"]), x1 - SKIN_RING_HALF, x1 + SKIN_RING_HALF]
    if len(d["x_rows"]) > 1:
        kx += [x0 - SKIN_RING_HALF, x0 + SKIN_RING_HALF]
    kx += [float(d["x_tc"]), L / 2]
    ye = float(d["eyelet_y"])
    ky = [0.0, float(d["g"]) / 2, ye - SKIN_HOLE_HALF, ye + SKIN_HOLE_HALF, float(d["Wc"]) / 2, float(d["W"]) / 2]
    return np.asarray(kx, float), np.asarray(ky, float)


def _pl(x, src, dst):
    """Piecewise-linear map through the knots, unit slope beyond them; (values, slopes)."""
    out = np.interp(x, src, dst)
    seg = np.clip(np.searchsorted(src, x) - 1, 0, len(src) - 2)
    slope = (dst[seg + 1] - dst[seg]) / (src[seg + 1] - src[seg])
    lo, hi = x < src[0], x > src[-1]
    out[lo] = dst[0] + (x[lo] - src[0])
    out[hi] = dst[-1] + (x[hi] - src[-1])
    slope[lo | hi] = 1.0
    return out, slope


def skin_warp(template_dims, dims):
    """f(v, n) -> (v', n'): the template shoe frame onto the batch's (x, |y| piecewise linear; z unchanged)."""
    sx, sy = _knots(template_dims)
    tx, ty = _knots(dims)
    ok = all(np.all(np.diff(k) > 0) for k in (sx, sy, tx, ty))
    if not ok:                      # a degenerate batch: one scale along x and y instead
        sx, tx = np.array([-template_dims["L"] / 2, template_dims["L"] / 2]), np.array([-dims["L"] / 2, dims["L"] / 2])
        sy, ty = np.array([0.0, template_dims["W"] / 2]), np.array([0.0, dims["W"] / 2])

    def f(v, n):
        v = np.asarray(v, np.float64)
        n = np.asarray(n, np.float64)
        x, ax = _pl(v[:, 0], sx, tx)
        ay, ay_s = _pl(np.abs(v[:, 1]), sy, ty)
        out = np.stack([x, np.sign(v[:, 1]) * ay, v[:, 2]], 1)
        nn = np.stack([n[:, 0] / ax, n[:, 1] / ay_s, n[:, 2]], 1)
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        return out, nn
    return f


# ------------------------------------------------------------------------------------------------ the twins
def _class_rgb(c, colours):
    """sRGB base colour of a class with the pair's tint applied."""
    if c.get("tint"):
        rgb = np.asarray(colours[c["tint"]], float) * 1.0
        lin = _lin(rgb) * float(c.get("tint_scale", 1.0))
    else:
        lin = _lin(c["rgb"])
    if c.get("desat"):
        lum = float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])
        lin = lin + (lum - lin) * float(c["desat"])
    return lin                                   # linear


def _add_tex(spec, name, img, srgb=True):
    t = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=int(img.shape[1]), height=int(img.shape[0]),
                         nchannel=int(img.shape[2]))
    t.data = np.ascontiguousarray(img, np.uint8).tobytes()
    t.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_SRGB if srgb else mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR
    return name


MAT_RGBA = {}     # material name -> rgba, for the geoms that use it (a geom rgba other than MuJoCo's default overrides
                  # the material's, so every geom carries its material's rgba)


def _class_material(spec, key, c, colours, gain, tex_px=256):
    """One material per class: an untextured colour, or the class texture tinted (and gained) in its data."""
    lin = _class_rgb(c, colours) * np.asarray(gain, float)
    mat = spec.add_material(name="look_" + key)
    if c.get("legacy"):
        mat.specular, mat.shininess = float(c.get("specular", 0.5)), float(c.get("shininess", 0.5))
    else:
        mat.roughness = float(c.get("roughness", 0.5))
        mat.metallic = float(c.get("metallic", 0.0))
        mat.specular = 0.5
    mat.reflectance = 0.0
    if c.get("tex"):
        img = _png("tex/" + c["tex"], tex_px).astype(np.float64) / 255.0
        k = 1.0 / float(c.get("col_mean", 0.5)) if c.get("neutral", True) else 1.0
        data = np.round(255.0 * _srgb(_lin(img[..., :3]) * lin[None, None, :] * k)).astype(np.uint8)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _add_tex(spec, "look_tex_" + key, data)
        if not c.get("uv"):
            mat.texuniform = True
            mat.texrepeat = [1.0 / float(c["tile"]), 1.0 / float(c["tile"])]
        mat.rgba = [1.0, 1.0, 1.0, 1.0]
    else:
        over = float(lin.max())
        mat.rgba = [float(x) for x in _srgb(lin)] + [1.0]
        if over > 1.0:                           # what a colour at full value cannot take: a little emission
            mat.emission = min(0.4, 0.5 * (over - 1.0))
    MAT_RGBA[mat.name] = [float(x) for x in mat.rgba]
    return mat.name


def _add_mesh(spec, body, name, v, n, uv, f, material, group=1):
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
        g.rgba = MAT_RGBA.get(material, [1.0, 1.0, 1.0, 1.0])
    return g


def _quat_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _uv_meshes(spec, m):
    """Filament's physically based materials need texture coordinates and the Pandas' STL meshes carry none: in the
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


def decor_class(me, mat):
    """The look class of a decor mesh: the lamps (emissive), the hall's floor, the rest of the hall, the wire fence and
    the racks behind it, the other props."""
    if mat.get("emission"):
        return "lamps"
    if "Concrete" in mat.get("source", ""):
        return "floor"
    if me["item"] == "Hall":
        return "hall"
    if me["item"].startswith("fence") or me["item"] == "racks":
        return "fence_racks"
    return "props"


def _add_decor(spec, textures=True, near_m=None):
    """The hall and its props (arb-decor/1) as static world meshes (drawn only)."""
    dec = _json("decor/decor.json")
    mats = {}
    nf = 0
    gains = LOOK.get("decor_gains", {})
    for me in dec["meshes"]:
        if near_m is not None:
            lo, hi = np.asarray(me["bbox"][0]), np.asarray(me["bbox"][1])
            gap = np.maximum(0.0, np.maximum(lo, -hi))   # distance of the box from the bench centre per axis
            if float(np.linalg.norm(gap[:2])) > near_m:
                continue
        mk = me["material"]
        if textures and mk not in mats:
            m = dec["materials"][mk]
            mat = spec.add_material(name="decor_" + mk)
            g = np.asarray(gains.get(decor_class(me, m), [1.0, 1.0, 1.0]), float)
            rgba = np.asarray(m["rgba"][:3], float)
            # the maps' means as scalars, metal or not; a non-metal at least satin (a scalar from a mean of a mask
            # map made cardboard and painted props glossy)
            metal = float(m.get("metallic_eff", m.get("metallic", 0.0))) >= 0.5
            rough = float(m.get("roughness_eff", m.get("roughness", 0.5)))
            mat.roughness = float(np.clip(rough if (metal or decor_class(me, m) == "floor") else max(rough, 0.6), 0.05, 1.0))
            mat.metallic = 1.0 if metal else 0.0
            if m.get("cutout"):          # Filament's blended path keeps the specular of the see-through texels (a
                mat.roughness, mat.metallic = 0.6, 1.0   # veil): as a metal their reflectance is their (premultiplied, black) colour
            mat.specular, mat.reflectance = 0.5, 0.0
            if m.get("rgb"):
                img = _png("decor/textures/" + m["rgb"])
                rgb = np.round(255.0 * _srgb(_lin(img[..., :3] / 255.0) * (rgba * g)[None, None, :])).astype(np.uint8)
                cut = bool(m.get("cutout") and img.shape[2] == 4)
                if cut:
                    a = img[..., 3:4].astype(np.float64) / 255.0
                    rgb = np.concatenate([np.round(rgb * a).astype(np.uint8), img[..., 3:4]], 2)
                mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _add_tex(spec, "decor_tex_" + mk, rgb)
                # a material alpha below 1 puts Filament on its blended path, which honours the texture's cut-out alpha
                # (premultiplied: the colour of a see-through texel is black)
                mat.rgba = [1.0, 1.0, 1.0, 0.99 if cut else 1.0]
            else:
                lin = _lin(rgba) * g
                mat.rgba = [float(x) for x in _srgb(lin)] + [1.0]
            if m.get("emission"):
                mat.emission = float(min(4.0, m["emission"]))
            MAT_RGBA[mat.name] = [float(x) for x in mat.rgba]
            mats[mk] = mat.name
        z = _npz("decor/" + me["file"])
        _add_mesh(spec, spec.worldbody, me["name"], z["v"], z["n"] if textures else None, z["uv"] if textures else None,
                  z["f"], mats.get(mk), group=2)
        nf += int(len(z["f"]))
    return nf


def build_rgb_spec(scene, colours):
    """The RGB twin's MjSpec (and a record): the cell with the look, the skins, the decor, the lamp."""
    spec = scene.spec.copy()
    m = scene.model
    classes = LOOK["classes"]
    gains = LOOK.get("gains", {})
    rec = {"classes": {}}
    # the cell's own lights and headlight: the look's lamp and the image light replace them
    for li in list(spec.lights):
        spec.delete(li)
    spec.visual.headlight.active = 0
    rec["uv_meshes"] = _uv_meshes(spec, m)
    mats = {}
    by_name = {g.name: g for g in spec.geoms if g.name}
    for gi in range(m.ngeom):
        name = m.geom(gi).name
        sg = by_name.get(name) if name else None
        if sg is None:
            # unnamed geoms: the Pandas' meshes (visual groups keep their Menagerie materials) and collision hulls
            continue
        cls = geom_class(m, gi)
        if cls is None:
            if int(m.geom_type[gi]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                spec.delete(sg)                  # the checker floor (Filament draws a plane whatever its alpha)
                continue
            sg.rgba = [0.0, 0.0, 0.0, 0.0]
            sg.material = ""
            continue
        if cls not in mats:
            mats[cls] = _class_material(spec, cls, classes.get(cls, classes["default"]), colours, gains.get(cls, [1, 1, 1]))
        sg.material = mats[cls]
        sg.rgba = MAT_RGBA[mats[cls]]
        rec["classes"][cls] = rec["classes"].get(cls, 0) + 1
    # unnamed geoms: the Pandas (by their material) and every group >= 3 geom (hidden)
    pmat = {}
    for gs in spec.geoms:
        if gs.name:
            continue
        if int(gs.group) >= 3:
            gs.rgba = [0.0, 0.0, 0.0, 0.0]
            gs.material = ""
            continue
        mk = (gs.material or "").split("/")[-1]
        cls = {"white": "panda_white", "off_white": "panda_offwhite", "black": "panda_black", "dark_grey": "panda_black",
               "green": "panda_led_green", "light_blue": "panda_led_blue"}.get(mk)
        if cls is None:
            continue
        if cls not in pmat:
            pmat[cls] = mats.get(cls) or _class_material(spec, cls, classes[cls], colours, gains.get(cls, [1, 1, 1]))
            mats[cls] = pmat[cls]
        gs.material = pmat[cls]
        gs.rgba = MAT_RGBA[pmat[cls]]
        rec["classes"][cls] = rec["classes"].get(cls, 0) + 1
    # the sneaker skins
    sk = _json("skin/skin.json")
    tdims = sk["template_dims"]
    nskin = 0
    for i, sh in enumerate(scene.shoes):
        hand = "L" if i == 0 else "R"
        z = _npz("skin/" + sk["hands"][hand]["file"])
        warp = skin_warp(tdims, sh["dims"])
        body = spec.body(sh["body"])
        for part, pinfo in sk["parts"].items():
            if part == "Grommet":
                continue
            if part + "_v" not in z:
                continue
            cls = pinfo["class"]
            if cls not in mats:
                mats[cls] = _class_material(spec, cls, classes[cls], colours, gains.get(cls, [1, 1, 1]))
            v, n = warp(z[part + "_v"], z[part + "_n"])
            _add_mesh(spec, body, "skin%d/%s" % (i, part), v, n, z[part + "_uv"], z[part + "_f"], mats[cls])
            nskin += int(len(z[part + "_f"]))
        gv, gn, guv, gf = z["Grommet_v"], z["Grommet_n"], z["Grommet_uv"], z["Grommet_f"]
        d = sh["dims"]
        V, N, U, F = [], [], [], []
        for xk in d["x_rows"]:
            for sgn in (1.0, -1.0):
                V.append(gv + np.array([float(xk), sgn * float(d["eyelet_y"]), float(d["z_f"])]))
                N.append(gn)
                U.append(guv)
                F.append(gf + sum(len(x) for x in V[:-1]))
        if "grommet" not in mats:
            mats["grommet"] = _class_material(spec, "grommet", classes["grommet"], colours, gains.get("grommet", [1, 1, 1]))
        _add_mesh(spec, body, "skin%d/grommets" % i, np.concatenate(V), np.concatenate(N), np.concatenate(U),
                  np.concatenate(F), mats["grommet"])
        nskin += int(sum(len(x) for x in F))
    rec["skin_faces"] = nskin
    rec["decor_faces"] = _add_decor(spec, textures=True)
    for L in LOOK["lamps"]:
        li = spec.worldbody.add_light(name=L["name"], type=mujoco.mjtLightType.mjLIGHT_SPOT)
        li.pos, li.dir = [float(x) for x in L["pos"]], [float(x) for x in L["dir"]]
        li.cutoff, li.exponent = float(L["cutoff"]), float(L["exponent"])
        li.diffuse = [float(x) for x in L["color"]]
        li.specular, li.ambient = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        li.intensity = float(L["intensity"])
        li.bulbradius = float(L["bulb_m"])
        li.range = float(L["range"])
        li.castshadow = bool(L["shadow"])
    for k, v in sorted(LOOK["view"]["numeric"].items()):
        spec.add_numeric(name=k, data=[float(v)])
    for k, v in sorted(LOOK["view"]["text"].items()):
        spec.add_text(name=k, data=str(v))
    return spec, rec


def build_depth_spec(scene):
    """The depth twin's MjSpec: the cell as the physics model draws it plus the hall's and the props' meshes."""
    spec = scene.spec.copy()
    _add_decor(spec, textures=False, near_m=DEPTH_DECOR_M)
    return spec


# ------------------------------------------------------------------------------------------------ the renderer thread
class _View:
    """One RGB twin in the process's renderer; its Filament objects stay in the renderer's thread."""

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
        self.models = {}
        self.active = None
        self.live = []
        self.targets = {}
        self.next_sid = 0
        self.thread = threading.Thread(target=self._loop, name="shoecell-filament", daemon=True)
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
        model = self.models[sid]
        self._evict()
        F = self.F
        objs = F.ModelObjects(self.ctx, model)
        scn = self.ctx.create_scene(F.SceneParams())
        scn.configure_from_model(model)
        lights = F.ModelLights(scn, objs)
        rends = F.ModelRenderables(scn, objs)
        lp = F.LightParams()
        lp.type, lp.texture, lp.color = F.LightType.LIGHT_IMAGE, self.ibl_tex, [1.0, 1.0, 1.0]
        lp.intensity = float(LOOK["ibl"]["intensity"])
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
    """The cell's agent cameras: Filament colour on the RGB twin, classic depth on the depth twin. Built at the first
    frame of an episode and again when the pair's colours change."""

    def __init__(self, scene, sizes):
        self.scene = scene
        self.sizes = dict(sizes)
        self.key = None
        self.view = None
        self.rgb_model = self.rgb_data = None
        self.depth_model = self.depth_data = None
        self._depth_renderers = {}
        self._option = mujoco.MjvOption()
        self._option.geomgroup[4] = 0
        self.record = {}
        self._jmap = None

    def _ensure(self):
        _assets()
        colours = pair_colours(self.scene)
        key = json.dumps(colours, sort_keys=True)
        if key == self.key:
            return
        t0 = time.time()
        self._close_rgb()
        spec, rec = build_rgb_spec(self.scene, colours)
        self.rgb_model = spec.compile()
        self.rgb_data = mujoco.MjData(self.rgb_model)
        if self.depth_model is None:
            self.depth_model = build_depth_spec(self.scene).compile()
            self.depth_data = mujoco.MjData(self.depth_model)
        pm = self.scene.model
        for tw in (self.rgb_model, self.depth_model):
            if tw.nq != pm.nq or tw.ncam != pm.ncam:
                raise RenderError("render twin out of step with the cell (nq %d / %d)" % (tw.nq, pm.nq))
        self.view = _View(_Filament.instance(), self.rgb_model)
        self.key = key
        rec["build_s"] = round(time.time() - t0, 3)
        self.record = rec

    def _sync(self, data):
        for tw, td in ((self.rgb_model, self.rgb_data), (self.depth_model, self.depth_data)):
            td.qpos[:] = data.qpos
            mujoco.mj_kinematics(tw, td)
            mujoco.mj_camlight(tw, td)

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
        """The camera's colour image (uint8 HxWx3, rows top-down), no sensor noise."""
        self._ensure()
        self._sync(data)
        w, h = self.sizes[name]
        cid = self.rgb_model.camera(name).id
        R = self.rgb_data.cam_xmat[cid].reshape(3, 3)
        return self.view.render(self.rgb_data, self.rgb_data.cam_xpos[cid], -R[:, 2], R[:, 1],
                                float(self.rgb_model.cam_fovy[cid]), int(w), int(h))

    def render(self, data, name):
        """(rgb uint8 HxWx3, depth float32 HxW planar z in metres) of camera `name` at the physics state `data`."""
        rgb = self.rgb(data, name)
        return rgb, self._depth(name)

    def _close_rgb(self):
        if self.view is not None:
            try:
                self.view.close()
            except Exception:  # noqa: BLE001
                pass
            self.view = None
        self.rgb_model = self.rgb_data = None
        self.key = None

    def close(self):
        self._close_rgb()
        for r in self._depth_renderers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        self._depth_renderers = {}
        self.depth_model = self.depth_data = None
