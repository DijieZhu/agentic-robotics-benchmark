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
  * the unit's state, read from the simulator before every image: each burner's flame site (its colour, visibility
    and radius, RoboCasa's Stove.update_state as appliance_env.Unit.update sets them) and the indicator geom's colour
    (lit or INDICATOR_OFF).  A lit burner is drawn by kitchen_fx.py (beside this file): a gas burner (blue site) as
    a ring of blue flame jets around the burner head, a glass-ceramic zone (red site) as concentric rings glowing
    dark red to orange under the glass, with a clear glossy coat over the glass top; both grow with the setting
    (the site's radius).  The indicator is a lit emitter.
  * grounding, drawn only: soft contact shadows (dark decals that fade out) around the unit's footprint on what it
    stands on, the station's, the fillers' and the pedestal's on the floor and under the shelf, and a soft shadow of
    the hand and forearm straight below them on the unit's top or the support's top (from the robot bodies' poses).
The room and the dressing are drawn only: nothing of them is in the simulated model, so the robot touches only what
appliance_env.build_model builds (the pedestal, the unit and its support).
Light: an image light (ibl.ktx of the bundle: the mean of three path-traced probes of the kitchen at the work area,
all of the task's lights on; its diffuse part also carries the direct light of the lamps that are not drawn, its
reflections do not; sky.ktx, the same probes, is the sky behind the windows) and one Filament light per lamp of
kitchen_scene.lights() that the look draws (lamps.drawn: the work light, the key lamp), photometric: candela from
lumens.  The episode's light draw (appliance_env._appearance, read from the model: the scene light's position, aim,
diffuse, the headlight) moves and aims the key lamp, scales every light and the image light by the strength, tints
them by the tint, and adds a headlight (a directional light along the camera's view); the look's white balance is
one more gain per channel on every light.  The display transform (exposure, tone mapping, colour grading) and every
other number of the look are in kitchen_look.json beside this file, calibrated against path-traced references; the
bundle holds the room's meshes, textures and materials and the two panoramas.

Nothing here changes the simulation: the renderer reads the model and the data, it writes neither, and it draws no
random numbers (the sensor noise stays the environment's).

Filament runs in one dedicated thread per process (its engine is thread-bound; a forked child starts its own).
Objects that live for the process: the context, the room (meshes, textures, materials, renderables), the image
light, the lamps, the scene, one render target per image size, and every mesh made here (MuJoCo 3.16's binding never
releases a Filament mesh: the dressing's and the burners' meshes are built once per unit and cached).  Per episode:
the copy of the model and its ModelResources / ModelRenderables.  MuJoCo 3.16's Filament module serves one
model's resources at a time per context, so only the episode rendered last is live; rendering another one first
frees the live one's model resources (the room stays).  The first image of a process takes several seconds
(lavapipe compiles the shaders); later episodes start in about 0.1 s.

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
import kitchen_fx  # noqa: E402  (imported now: a process that drops privileges later, the lab, has them already)
import kitchen_scene  # noqa: E402

RENDER_DIR = os.environ.get("KITCHEN_RENDER_DIR", "/opt/task-assets/payload/render")
LOOK_FILE = os.path.join(HERE, "kitchen_look.json")
BUNDLE_FILES = ("room.npz", "room.json", "ibl.ktx", "sky.ktx")
CALL_TIMEOUT = float(os.environ.get("KITCHEN_RENDER_TIMEOUT", "300"))
CAM_NEAR, CAM_FAR = 0.01, 30.0
CAMERA_NAMES = ("left", "right", "wrist")
NOMINAL_LIGHT_POS = np.array([-0.5, 0.0, 2.6])    # appliance_env.build_model's scene light
NOMINAL_LIGHT_DIFFUSE = 0.8
GEOM_GROUP = 1                                     # robot and unit visual geoms; the env's own scenery is group 0
LUMA = np.array([0.2126, 0.7152, 0.0722])
ACESCCT_STOPS = 17.52                                  # filament.cg.offset o = a linear gain of 2 ** (17.52 o)


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


def linear_from_srgb(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _tint_for(name, table):
    """The look's gain for a room material: the entry whose key the material's name starts with."""
    for prefix, t in table.items():
        if name.startswith(prefix):
            return np.asarray(t, float)
    return None


def _gain_srgb8(img, gain):
    """An sRGB 8-bit image times a linear per-channel gain (one lookup table per channel)."""
    x = linear_from_srgb(np.arange(256) / 255.0)
    out = np.empty_like(img)
    for c in range(3):
        lut = np.round(srgb_from_linear(np.clip(x * float(gain[c]), 0.0, 1.0)) * 255.0).astype(np.uint8)
        out[..., c] = lut[img[..., c]]
    return out


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
    for i in range(d.nmat):                            # the look's colour gains (linear) per material name prefix
        name = d.mat(i).name
        for prefix, t in rules.get("tint", []):
            if name.startswith(prefix):
                d.mat_rgba[i, :3] = srgb_from_linear(linear_from_srgb(d.mat_rgba[i, :3]) * np.asarray(t, float))
                break
    d.light_intensity[:] = 0.0
    d.light_castshadow[:] = 0
    return d


def _appearance(m, layout, look):
    """The episode's light draw as appliance_env._appearance left it in the model (read only)."""
    diffuse = np.asarray(m.light_diffuse[0], float) / NOMINAL_LIGHT_DIFFUSE
    gain = float(np.mean(diffuse))
    # the look's white balance: a gain per channel on the lamps and the headlight (the image light has it baked in:
    # MuJoCo 3.16's binding ignores an image light's colour).  The episode's tint goes into the colour grading as a
    # per-channel gain on everything drawn (look["episode_tint_in_grading"]), so it reaches the image light too;
    # otherwise it multiplies the lamps and the headlight only.
    raw = diffuse / max(gain, 1e-6)
    wb = np.asarray(look.get("white_balance_rgb", [1.0, 1.0, 1.0]), float)
    tint = wb if look.get("episode_tint_in_grading") else raw * wb
    pos = np.asarray(m.light_pos[0], float)
    ldir = np.asarray(m.light_dir[0], float)
    lo, hi, mt = layout["bbox_lo"], layout["bbox_hi"], layout["mount"]
    front = np.array([mt["front_x"], 0.0, mt["bottom_z"] + (hi[2] - lo[2]) / 2])
    t = max(float(np.dot(front - pos, ldir)), 0.5)
    aim = pos + t * ldir                               # the point of the light's axis nearest the unit's front
    return {"gain": gain, "tint": tint.tolist(), "episode_tint": raw.tolist(), "offset": (pos - NOMINAL_LIGHT_POS).tolist(), "aim": aim.tolist(),
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
        if l["type"] in ("sky", "sun") or l.get("demo_only") or l["name"] not in L["drawn"]:
            continue                                   # daylight and the other lamps are in the image light
        rgb = np.asarray(l["rgb"], float)
        rgb = rgb / float(np.dot(rgb, LUMA)) * tint    # a lamp's colour keeps its luminance (lumens as given)
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
        self.dressing = _dressing_plan(KS, kind, lo, hi, jit)
        self.decals = _contact_decals(KS.all_parts(kind, lo, hi, jit), lo, hi, m, look)
        self.hand_shadow = _hand_shadow_plan(m, lo, hi, look)
        self.materials = KS.materials()
        self.app = _appearance(m, lay, look)
        self.lights = _rig(KS, kind, lo, hi, jit, look, self.app)
        ibl = look["ibl"]
        self.ibl_intensity = (ibl["intensity"] * look["exposure_scale"] * self.app["gain"]
                              * (1.0 + look["headlight"]["ambient_to_ibl"] * (self.app["head_ambient"] - 0.125)))
        self.ibl_color = np.clip(np.asarray(self.app["tint"]), 0.0, 8.0).tolist()
        self.head_lux = look["headlight"]["lux_at_0.4"] * self.app["head_diffuse"] / 0.4 * look["exposure_scale"]
        self.grade_offset = None
        if look.get("episode_tint_in_grading"):
            t = np.clip(np.asarray(self.app["episode_tint"], float), 0.05, 20.0)
            self.grade_offset = tuple(round(float(x), 6) for x in np.log2(t) / ACESCCT_STOPS)
        u = env.unit
        self.flames = sorted(u.flames.items())         # [(loc, site id)]
        self.fx = _fx_episode(env, m, lo, hi)          # burner visuals: base sizes, kinds, the unit's box
        self.indicator = u.indicator                   # geom id or None
        self.indicator_kind = kind
        self.wrist_cam = env.wrist_cam
        self.fixed_cam = {c: m.camera(c).id for c in ("left", "right")}
        self.seconds = time.perf_counter() - t0


RIGID_GROUPS = (("station", "st_"), ("shelf", "shelf_"), ("pedestal", "ped_"))


def _dressing_plan(KS, kind, lo, hi, jit):
    """The dressing of this episode as what the renderer caches for the process.  The station, the shelf and the
    pedestal move rigidly with the unit's placement (their parts are the placement-free parts shifted): such a group
    is drawn from merged meshes built once per unit (key) and placed by the shift.  The filler cabinets' widths
    follow the placement: each of their parts is a single shape (a shared unit box or cylinder, scaled)."""
    parts = KS.all_parts(kind, lo, hi, jit)
    parts0 = {p["name"]: p for p in KS.all_parts(kind, lo, hi, (0.0, 0.0, 0.0))}
    key = (kind,) + tuple(np.round(np.concatenate([lo, hi]), 6))
    rigid, single = [], []
    for group, prefix in RIGID_GROUPS:
        mine = [p for p in parts if p["name"].startswith(prefix)]
        if not mine:
            continue
        base = [parts0.get(p["name"]) for p in mine]
        shift = None
        if all(b is not None and b["shape"] == p["shape"] and np.allclose(b["size"], p["size"], atol=1e-9)
               for b, p in zip(base, mine)) and len(base) == sum(1 for n in parts0 if n.startswith(prefix)):
            d = np.array([np.subtract(p["center"], b["center"]) for b, p in zip(base, mine)])
            if np.allclose(d, d[0], atol=1e-9):
                shift = d[0]
        if shift is None:
            single += mine
        else:
            rigid.append({"key": (group,) + key, "parts": base, "shift": shift.tolist()})
    single += [p for p in parts if not p["name"].startswith(tuple(pr for _g, pr in RIGID_GROUPS))]
    return {"rigid": rigid, "single": single}


def _contact_decals(parts, lo, hi, m, look):
    """Static contact shadows (soft dark decals, shaped after the path-traced references' soft shadows): around the
    unit's footprint on what it stands on, around the station's, the fillers' and the pedestal's footprints on the
    floor, and on the wall under the shelf.  Each: a frame (rotation, origin; the unit's moves with its body), the
    inner half-size, the falloff margin and the darkness; drawn as a 9-slice quad (_decal_mesh)."""
    D = look.get("contact_shadows")
    if not D or not D.get("enabled"):
        return []
    out = []
    Rz = np.eye(3)

    def box_xy(ps):
        b = [KR_part_bounds(p) for p in ps]
        a = np.min([x[0] for x in b], 0)
        c = np.max([x[1] for x in b], 0)
        return a, c
    # the unit on its support (frame: the unit's body; its box bottom is what touches)
    u = D["unit"]
    c = (np.asarray(lo, float) + np.asarray(hi, float)) / 2
    h = (np.asarray(hi, float) - np.asarray(lo, float)) / 2
    # (the unit's model frame: its front is model -y); a low unit (a cooktop) casts a narrower shadow than a tall one
    k = float(np.clip(2 * h[2] / float(u.get("full_height_m", 0.3)), float(u.get("min_scale", 0.3)), 1.0))
    out.append({"body": int(m.body("u_object").id), "R": Rz, "p": [c[0], c[1], lo[2] + u["lift_m"]],
                "half": [h[0] * u["inset"], h[1] * u["inset"]],
                "margin": [k * u["margin_m"], k * u["margin_m"], k * u.get("front_margin_m", u["margin_m"]), k * u["margin_m"]],
                "alpha": u["alpha"]})
    f = D["floor"]
    groups = {}
    for p in parts:
        if p["name"].startswith("st_") and p.get("role") == "body":
            groups.setdefault("station", []).append(p)
        elif p.get("role") == "filler" and p["name"].endswith(("_plinth", "_carcass")):
            groups.setdefault(p["name"].rsplit("_", 1)[0], []).append(p)
        elif p["name"] == "ped_base":
            groups.setdefault("pedestal", []).append(p)
    for name, ps in sorted(groups.items()):
        a, b = box_xy(ps)
        g = f.get(name.split("_")[0] if name.startswith("fill") else name, f["default"])
        out.append({"body": None, "R": Rz, "p": [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, f["lift_m"]],
                    "half": [(b[0] - a[0]) / 2, (b[1] - a[1]) / 2], "margin": [g["margin_m"]] * 4, "alpha": g["alpha"]})
    sh = D.get("shelf")
    board = [p for p in parts if p["name"] == "shelf_board"]
    if sh and board:
        a, b = KR_part_bounds(board[0])
        # on the wall (the board's back face, x = b[0]) just under the board; facing the room (-x)
        R = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out.append({"body": None, "R": R, "p": [b[0] - sh["lift_m"], (a[1] + b[1]) / 2, a[2] + sh["below_m"]],
                    "half": [(b[1] - a[1]) / 2, sh["half_height_m"]],
                    "margin": [sh["margin_m"], sh["margin_m"], sh.get("below_margin_m", sh["margin_m"]), 0.0],
                    "alpha": sh["alpha"]})
    for d in out:
        d["key"] = ("decal",) + tuple(np.round(list(d["half"]) + list(d["margin"]), 3))
    return out


def KR_part_bounds(p):
    c = np.asarray(p["center"], float)
    if p["shape"] == "box":
        hh = np.asarray(p["size"], float) / 2
    else:
        r, hl = p["size"][0], p["size"][1]
        hh = np.array([r, r, r])
        hh["xyz".index(p["axis"])] = hl
    return c - hh, c + hh


def _hand_shadow_plan(m, lo, hi, look):
    """The soft shadow of the hand and forearm (look["hand_shadow"]): casters = robot bodies (name, radius), drawn
    straight down onto the first known plane below each: the unit's top (its model box, in the unit body's frame)
    or the top of the support it stands on (the simulator's "support" geom).  None when off."""
    H = look.get("hand_shadow")
    if not H or not H.get("enabled"):
        return None
    casters = []
    for name, r in H["casters"]:
        try:
            casters.append((int(m.body(name).id), float(r)))
        except KeyError:
            pass
    sup = None
    try:
        g = m.geom("support")
        sup = (np.asarray(m.geom_pos[g.id], float).copy(), np.asarray(m.geom_size[g.id], float).copy())
    except KeyError:
        pass
    return {"casters": casters, "unit_body": int(m.body("u_object").id), "lo": np.asarray(lo, float),
            "hi": np.asarray(hi, float), "support": sup}


def _hand_shadow_poses(plan, data, H):
    """Per caster: (position of the blob on its plane, radius, alpha bucket) or None (too high, nothing below)."""
    out = []
    b = plan["unit_body"]
    Rb = data.xmat[b].reshape(3, 3)
    pb = data.xpos[b]
    lo, hi = plan["lo"], plan["hi"]
    nb = int(H["alpha_buckets"])
    for body, r in plan["casters"]:
        c = data.xpos[body]
        q = Rb.T @ (c - pb)                            # the caster in the unit's model frame
        z = None
        if lo[0] <= q[0] <= hi[0] and lo[1] <= q[1] <= hi[1] and q[2] > hi[2]:
            z = float(pb[2] + Rb[2] @ np.array([q[0], q[1], hi[2]]))
            edge = min(q[0] - lo[0], hi[0] - q[0], q[1] - lo[1], hi[1] - q[1])
        elif plan["support"] is not None:
            sp, ss = plan["support"]
            if abs(c[0] - sp[0]) <= ss[0] and abs(c[1] - sp[1]) <= ss[1] and c[2] > sp[2] + ss[2]:
                z = float(sp[2] + ss[2])
                edge = min(ss[0] - abs(c[0] - sp[0]), ss[1] - abs(c[1] - sp[1]))
        if z is None:
            out.append(None)
            continue
        h = max(float(c[2]) - z - r, 0.0)
        a = float(H["alpha"]) * max(0.0, 1.0 - h / float(H["fade_height_m"])) ** float(H["fade_power"])
        k = int(round(a / float(H["alpha"]) * (nb - 1)))
        if k <= 0:
            out.append(None)
            continue
        rad = min(r * float(H["radius_scale"]) + h * float(H["spread"]), edge * 1.1)   # not past the plane's edge
        if rad < 0.01:
            out.append(None)
            continue
        out.append((np.array([c[0], c[1], z + float(H["lift_m"])], np.float32), rad, k))
    return out


def _decal_mesh(hx, hy, m):
    """A 9-slice quad in its frame's xy plane (normal +z): the inner rectangle +-hx, +-hy maps to the texture's
    centre square (0.25 .. 0.75), the margins m = (-x, +x, -y, +y) around it to the texture's border, so the falloff
    keeps its width in metres whatever the size."""
    xs = [-hx - m[0], -hx, hx, hx + m[1]]
    ys = [-hy - m[2], -hy, hy, hy + m[3]]
    uv = [0.0, 0.25, 0.75, 1.0]
    V, N, U, I = [], [], [], []
    for j in range(4):
        for i in range(4):
            V.append([xs[i], ys[j], 0.0]); N.append([0.0, 0.0, 1.0]); U.append([uv[i], uv[j]])
    for j in range(3):
        for i in range(3):
            a = j * 4 + i
            I += [a, a + 1, a + 5, a, a + 5, a + 4]
    return np.array(V), np.array(N), np.array(U), np.array(I)


def _decal_alpha(n=64, power=1.6):
    """The decal texture's alpha: 1 in the centre square, falling smoothly to 0 at the border (rounded corners)."""
    t = (np.arange(n) + 0.5) / n
    d = np.maximum(np.abs(t - 0.5) - 0.25, 0.0) / 0.25
    r = np.clip(np.sqrt(d[:, None] ** 2 + d[None, :] ** 2), 0.0, 1.0)
    a = (1.0 - r * r * (3.0 - 2.0 * r)) ** power
    return a


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
# Burner visuals (kitchen_fx): what of the episode they need, in the caller's thread
# ----------------------------------------------------------------------------------------------------------------
def _fx_episode(env, m, lo, hi):
    """Per burner: its kind (from the site's colour), base radius and half-height (Unit.flame_size, the sizes before
    any setting scaled them), key; the unit's body and box (for the glass coat of glass-ceramic tops)."""
    FX = kitchen_fx
    u = env.unit
    burners = {}
    for loc, sid in sorted(u.flames.items()):
        r0, hl = (float(x) for x in u.flame_size[loc][:2])
        burners[loc] = {"site": int(sid), "kind": FX.burner_kind(m.site_rgba[sid, :3]), "r0": r0, "hl": hl,
                        "key": FX.burner_key(loc)}
    return {"burners": burners, "body": int(m.body("u_object").id), "lo": np.asarray(lo, float).copy(),
            "hi": np.asarray(hi, float).copy()}


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
        self.scn = self.ctx.create_scene(F.SceneParams())
        self.grade = "unset"
        self._configure(None)
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

    def _configure(self, offset):
        """The look's view options (they travel as custom fields of a tiny model: MuJoCo reads filament.* from them),
        with the colour grading's offset = the episode's tint (None: no offset)."""
        if offset == self.grade:
            return
        look = self.look
        num = dict(look["filament_numeric"])
        if offset is not None:
            num["filament.cg.offset"] = list(offset)
        spec = mujoco.MjSpec()
        for k, v in sorted(num.items()):                # a number or a list (curves, offset: r, g, b)
            spec.add_numeric(name=k, data=[float(x) for x in (v if isinstance(v, (list, tuple)) else [v])])
        for k, v in sorted(look["filament_text"].items()):
            spec.add_text(name=k, data=str(v))
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], pos=[0, 0, -50])
        self.look_model = spec.compile()
        for _ in range(2):                             # MuJoCo 3.16 applies the colour grading one configure late
            self.scn.configure_from_model(self.look_model)
        self.grade = offset
        self.stats["regrades"] = self.stats.get("regrades", 0) + 1

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
        # the look's per-material colour gains (linear, per channel) go into the texture where a material has one
        # (a gain above 1 cannot go into a material's colour); a texture is one material's
        gains = {}
        for mm in meta["materials"]:
            t = _tint_for(mm["name"], self.look["room"].get("tint", {}))
            if t is not None and mm["tex"] >= 0:
                gains[mm["tex"]] = t
        for name, i in meta.get("dressing_textures", {}).items():
            t = self.look["dressing"].get("tint", {}).get(name)
            if t is not None:
                gains[i] = np.asarray(t, float)
        texs = []
        for k, (o, w, h) in enumerate(zip(tex_ofs, tw, th)):
            img = tex[o:o + w * h * 3].reshape(h, w, 3)
            if k in gains:
                img = _gain_srgb8(img, gains[k])
            texs.append(self._texture(img))
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
            t = _tint_for(mm["name"], rl.get("tint", {})) if mm["tex"] < 0 else None
            if g != 1.0 or t is not None:
                rgba[:3] = srgb_from_linear(np.power(np.clip(rgba[:3], 0, 1), 2.2) * g * (1.0 if t is None else t))
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
        self.dress_cache, self.dress_mats = {}, {}
        self._fx_start()
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
        self._fx_evict(L)
        for r in L.get("dressing", []):
            self.scn.remove_renderable(r[0] if isinstance(r, tuple) else r)
        for r, _dc in L.get("decals", []):
            self.scn.remove_renderable(r)
        for r, _k in L.get("hand", []):
            self.scn.remove_renderable(r)
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
        L = {"res": res, "rends": rends, "dressing": []}
        # the room at this episode's pose
        T = ep.room_pose
        if self.room_pose is None or not np.allclose(self.room_pose, T):
            p, R = T[:3, 3].astype(np.float32), T[:3, :3].astype(np.float32).reshape(9)
            for _m, _mat, r in self.room:
                r.set_pose(p, R)
            self.room_pose = T.copy()
        # the dressing, from meshes and materials that live for the process (see _dressing_mesh)
        for r in self._dressing(ep):
            self.scn.add_renderable(r)
            L["dressing"].append(r)
        # contact shadows: process-cached meshes (by size), one shared material per darkness
        L["decals"] = []
        for dc in ep.decals:
            r = self._decal(dc)
            if dc["body"] is None:
                r.set_pose(np.asarray(dc["p"], np.float32), np.asarray(dc["R"], np.float32).reshape(9))
            self.scn.add_renderable(r)
            L["decals"].append((r, dc))
        L["hand"] = []
        if ep.hand_shadow is not None:
            for _c in ep.hand_shadow["casters"]:
                r = self._hand_blob()
                r.set_pose(np.array([0.0, 0.0, -20.0], np.float32), np.eye(3, dtype=np.float32).reshape(9))
                self.scn.add_renderable(r)
                L["hand"].append([r, 0])
        # the unit's state: flames and indicator (added to the scene while lit)
        L["flames"] = self._fx_activate(ep)
        L["indicator"] = None
        if ep.indicator is not None:
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadow=False, receive_shadow=True))
            r.set_mesh(self.box_mesh[0], 0, self.box_mesh[1])
            L["indicator"] = [r, ep.indicator, False, None, None]
        # lamps of this episode
        self._lamps(ep)
        self.ibl.set_intensity(float(ep.ibl_intensity))
        self.ibl.set_color(np.asarray(ep.ibl_color, np.float32))   # (ignored by MuJoCo 3.16's binding: see _appearance)
        if ep.grade_offset is not None:
            self._configure(ep.grade_offset)
        self.live = L
        self.active = sid
        self.stats["activations"] += 1
        self.last_activation_s = time.perf_counter() - t0

    # -- the dressing: MuJoCo 3.16's binding never releases a Filament mesh (nor the data uploaded into one), so
    # no mesh is made per episode: a rigid group (station, shelf, pedestal; _dressing_plan) is one merged mesh per
    # material, built the first time its unit is drawn and placed by the episode's shift; a single part (the filler
    # cabinets) is a shared unit box (untextured) or unit cylinder scaled to it.  Bounded by the units a process
    # serves; materials are made once per material.
    def _dressing_material(self, mname, mdef):
        if mname not in self.dress_mats:
            F = self.F
            dl = self.look["dressing"]
            tint = np.asarray(dl.get("tint", {}).get(mname, [1.0, 1.0, 1.0]), float)
            if self.dressing_tex.get(mname) is not None:
                tint = np.ones(3)                       # a textured material's gain is in its texture (_load_room)
            base = np.asarray(mdef.get("base", [0.7, 0.7, 0.7]), float) * dl["albedo_gain"] * tint
            rgba = np.concatenate([srgb_from_linear(base), [1.0]]).astype(np.float32)
            emissive = float(dl["emitter_emissive"]) if mdef.get("emission_K") else 0.0
            self.dress_mats[mname] = F.Material(color=rgba, roughness=float(mdef.get("roughness", 0.5)),
                                                metallic=float(mdef.get("metallic", 0.0)), emissive=emissive,
                                                color_texture=self.dressing_tex.get(mname))
        return self.dress_mats[mname]

    def _dressing_renderable(self, mesh, n, mat, p, R):
        F = self.F
        r = self.ctx.create_renderable(F.RenderableParams(cast_shadow=bool(self.look["dressing"]["cast_shadow"]),
                                                          receive_shadow=True))
        r.set_mesh(mesh, 0, n)
        r.set_material(mat)
        r.set_pose(np.asarray(p, np.float32), np.asarray(R, np.float32).reshape(9))
        return r

    def _hand_blob(self):
        """A renderable of the hand's shadow: a unit disc (a quad with a radial falloff texture), process-cached mesh,
        texture and per-bucket materials."""
        F = self.F
        H = self.look["hand_shadow"]
        if "hand_blob" not in self.dress_cache:
            V = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
            N = np.array([[0, 0, 1.0]] * 4)
            U = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
            self.dress_cache["hand_blob"] = (self._mesh(V, N, U, np.array([0, 1, 2, 0, 2, 3])), 6)
            n = 64
            t = (np.arange(n) + 0.5) / n * 2 - 1
            rr = np.clip(np.sqrt(t[:, None] ** 2 + t[None, :] ** 2), 0, 1)
            a = (1.0 - rr * rr * (3.0 - 2.0 * rr)) ** float(H["profile_power"])
            rgba = np.zeros((n, n, 4), np.uint8)
            rgba[..., 3] = np.round(a * 255).astype(np.uint8)
            tex = self._fx_texture(rgba, F.PixelFormat.PIXEL_FORMAT_RGBA8)
            self.dress_mats["hand_blob_tex"] = tex             # a material does not keep its texture alive
            nb = int(H["alpha_buckets"])
            self.dress_mats["hand_blob"] = [None] + [
                F.Material(color=np.array([0.0, 0.0, 0.0, float(H["alpha"]) * k / (nb - 1)], np.float32), roughness=1.0,
                           metallic=1.0, reflectance=0.0, color_texture=tex) for k in range(1, nb)]
        mesh, n = self.dress_cache["hand_blob"]
        rp = F.RenderableParams(cast_shadow=False, receive_shadow=False)
        # after the glass coat (3): a blended layer writes depth, so a blob drawn before the coat would cut the coat
        # out over its whole quad; the flames stand in front of it in depth and stay undarkened
        rp.blend_order = int(H.get("blend_order", 4))
        r = self.ctx.create_renderable(rp)
        r.set_mesh(mesh, 0, n)
        r.set_material(self.dress_mats["hand_blob"][1])
        return r

    def _decal(self, dc):
        F = self.F
        if dc["key"] not in self.dress_cache:
            V, N, U, I = _decal_mesh(dc["half"][0], dc["half"][1], dc["margin"])
            self.dress_cache[dc["key"]] = (self._mesh(V, N, U, I), len(I))
        mk = ("decal_mat", round(float(dc["alpha"]), 3))
        if mk not in self.dress_mats:
            if "decal_tex" not in self.dress_mats:
                rgba = np.zeros((64, 64, 4), np.uint8)
                rgba[..., 3] = np.round(_decal_alpha(power=float(self.look["contact_shadows"].get("power", 1.6)))
                                        * 255).astype(np.uint8)
                self.dress_mats["decal_tex"] = self._fx_texture(rgba, F.PixelFormat.PIXEL_FORMAT_RGBA8)
            # transparent PBR is premultiplied and its specular is not faded by alpha: a black metal has no diffuse
            # and no specular (f0 = f90 = 0), so the layer only darkens what is behind it, by alpha
            self.dress_mats[mk] = F.Material(color=np.array([0.0, 0.0, 0.0, float(dc["alpha"])], np.float32),
                                             roughness=1.0, metallic=1.0, reflectance=0.0,
                                             color_texture=self.dress_mats["decal_tex"])
        mesh, n = self.dress_cache[dc["key"]]
        rp = F.RenderableParams(cast_shadow=False, receive_shadow=False)
        rp.blend_order = 0
        r = self.ctx.create_renderable(rp)
        r.set_mesh(mesh, 0, n)
        r.set_material(self.dress_mats[mk])
        return r

    def _dressing(self, ep):
        mats = ep.materials
        dflt = {"base": [0.7, 0.7, 0.7], "roughness": 0.5, "metallic": 0.0}
        out = []
        for g in ep.dressing["rigid"]:
            if g["key"] not in self.dress_cache:
                groups = {}
                for part in g["parts"]:
                    groups.setdefault(part["material"], []).append(part)
                built = []
                for mname, parts in sorted(groups.items()):
                    tile = float(mats.get(mname, dflt).get("tile_m", 1.0))
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
                    I = np.concatenate(II)
                    built.append((mname, self._mesh(np.concatenate(VV), np.concatenate(NN), np.concatenate(UU), I),
                                  len(I)))
                self.dress_cache[g["key"]] = built
            for mname, mesh, n in self.dress_cache[g["key"]]:
                out.append(self._dressing_renderable(mesh, n, self._dressing_material(mname, mats.get(mname, dflt)),
                                                     g["shift"], np.eye(3)))
        for part in ep.dressing["single"]:
            mname = part["material"]
            mdef = mats.get(mname, dflt)
            c = np.asarray(part["center"], float)
            if part["shape"] == "box":
                size = np.asarray(part["size"], float)
                if self.dressing_tex.get(mname) is None:
                    k, S = ("unit_box",), np.diag(size)
                else:                                   # a textured box keeps its uvs in metres: its own size
                    k, S = ("box", mname, part.get("grain")) + tuple(np.round(size, 4)), np.eye(3)
                if k not in self.dress_cache:
                    if k[0] == "unit_box":
                        V, N, U, I = _box_mesh([-0.5] * 3, [0.5] * 3, 1.0, None)
                    else:
                        h = np.asarray(k[3:], float) / 2
                        V, N, U, I = _box_mesh(-h, h, float(mdef.get("tile_m", 1.0)), part.get("grain"))
                    self.dress_cache[k] = (self._mesh(V, N, U, I), len(I))
            else:
                k = ("unit_cyl", part["axis"])
                ax = "xyz".index(part["axis"])
                sc = np.full(3, float(part["size"][0]))
                sc[ax] = float(part["size"][1])
                S = np.diag(sc)
                if k not in self.dress_cache:
                    V, N, U, I = _cyl_mesh([0.0, 0.0, 0.0], 1.0, 1.0, part["axis"], 12)
                    self.dress_cache[k] = (self._mesh(V, N, U, I), len(I))
            mesh, n = self.dress_cache[k]
            out.append(self._dressing_renderable(mesh, n, self._dressing_material(mname, mdef), c, S))
        return out

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
        self._fx_state(ep, data)
        for r, dc in self.live.get("decals", []):
            if dc["body"] is not None:                 # on the unit's body (it sits still; read like the flames)
                b = dc["body"]
                Rb = data.xmat[b].reshape(3, 3)
                r.set_pose((data.xpos[b] + Rb @ np.asarray(dc["p"])).astype(np.float32),
                           (Rb @ dc["R"]).astype(np.float32).reshape(9))
        if self.live.get("hand"):
            H = look["hand_shadow"]
            for slot, pose in zip(self.live["hand"], _hand_shadow_poses(ep.hand_shadow, data, H)):
                r = slot[0]
                if pose is None:
                    r.set_pose(np.array([0.0, 0.0, -20.0], np.float32), np.eye(3, dtype=np.float32).reshape(9))
                    continue
                p, rad, k = pose
                if k != slot[1]:
                    r.set_material(self.dress_mats["hand_blob"][k])
                    slot[1] = k
                r.set_pose(p, (np.eye(3) * rad).astype(np.float32).reshape(9))
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
                elif il.get("lit_saturation", 1.0) != 1.0:  # ahead of the display's desaturation (its saturation)
                    c = linear_from_srgb(col[:3])
                    y = float(np.dot(c, LUMA))
                    col[:3] = srgb_from_linear(np.clip(y + (c - y) * float(il["lit_saturation"]), 0.0, 1.0))
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

    # -- burner visuals (kitchen_fx) ----------------------------------------------------------------------------------
    # Gas burners: three renderables per burner (inner cones: MuJoCo's unlit decor material, opaque; sheaths and haze:
    # transparent PBR, black base colour, alpha from an RGBA texture, emission from an emission texture).  Their
    # meshes: for each level bucket the burner reaches, one Filament mesh holding every flicker phase
    # (kitchen_fx.FLICKER_PHASES) of every part, built the first time the bucket is drawn; a frame only points the
    # renderables at the (bucket, phase) index ranges.  Glass-ceramic zones: one glow disk per zone (one material per
    # level bucket, shared by the zones), scaled to the site's radius; one clear glossy coat over the glass top.
    # Textures, materials and meshes live for the process: MuJoCo 3.16's binding does not release a Filament mesh when
    # its Python object is dropped (measured: 13 MB per episode with one bucket, 75 MB with nine), so the meshes are
    # cached by what they depend on (a burner's base size, key and level bucket; a glass top's box), bounded by the
    # units a process serves.  Images depend only on the model, the data and data.time (no clock, no random numbers).

    def _fx_texture(self, img, fmt):
        F = self.F
        h, w = img.shape[:2]
        tc = F.TextureConfig()
        tc.width, tc.height, tc.format = w, h, fmt
        tc.sampler_type, tc.color_space = F.SamplerType.TEXTURE_2D, F.ColorSpace.COLORSPACE_SRGB
        t = self.ctx.create_texture(tc)
        td = F.TextureData()
        td.set_image(np.ascontiguousarray(img, np.uint8).tobytes())
        t.upload(td)
        return t

    def _fx_start(self):
        F = self.F
        FX = kitchen_fx
        self.FX = FX
        fxl = self.look["burner_fx"]
        self.fx_look = fxl
        tex, mats = FX.textures(), FX.materials()
        res = {"tex": [], "mat": {}}
        gain = float(fxl["emissive_gain"])

        def textured(name, emission, roughness, gain=gain):
            t = tex[name]
            rgba = np.zeros(t["alpha"].shape + (4,), np.uint8)
            rgba[..., 3] = t["alpha"]                        # black, premultiplied: only the emission shows
            ta = self._fx_texture(rgba, F.PixelFormat.PIXEL_FORMAT_RGBA8)
            te = self._fx_texture(t["emission"], F.PixelFormat.PIXEL_FORMAT_RGB8)
            res["tex"] += [ta, te]
            return F.Material(color=np.array([1.0, 1.0, 1.0, 0.999], np.float32), roughness=float(roughness),
                              metallic=0.0, emissive=float(emission) * gain, color_texture=ta, emissive_texture=te)
        g = mats["gas_inner"]
        res["mat"]["gas_inner"] = F.Material(
            color=np.array(list(np.asarray(g["color"], float) * fxl["gas_inner_gain"]) + [1.0], np.float32), decor=True)
        for part in ("gas_sheath", "gas_haze"):               # the gas flames' own gain (default: emissive_gain)
            res["mat"][part] = textured(part, mats[part]["emission"], mats[part]["roughness"],
                                        float(fxl.get("gas_emissive_gain", gain)))
        rg = mats["radiant_glow"]
        res["mat"]["radiant"] = [textured("radiant_%d" % b, rg["emission"], rg["roughness"]) for b in range(FX.RADIANT_LEVELS)]
        c = mats["glass_coat"]
        res["mat"]["glass_coat"] = F.Material(color=np.array(list(c["color"]) + [c["alpha"]], np.float32),
                                              roughness=float(c["roughness"]), metallic=float(c["metallic"]))
        V, N, U, I = FX.radiant_mesh()["radiant_glow"]
        res["radiant_mesh"] = (self._mesh(V, N, U, I), len(I))
        res["meshes"] = {}                                   # process-wide mesh cache (see above)
        self.fx_res = res

    def _fx_renderable(self, blend_order):
        rp = self.F.RenderableParams(cast_shadow=False, receive_shadow=False)
        rp.blend_order = int(blend_order)
        return self.ctx.create_renderable(rp)

    def _fx_activate(self, ep):
        res = self.fx_res
        out = {"burners": [], "coat": None}
        for loc, b in sorted(ep.fx["burners"].items()):
            rec = {"loc": loc, "b": b, "shown": False, "last": None, "parts": {}}
            if b["kind"] == "gas":
                for part, bo in (("gas_inner", 0), ("gas_sheath", 1), ("gas_haze", 2)):
                    r = self._fx_renderable(bo)
                    r.set_material(res["mat"][part])
                    rec["parts"][part] = r
            else:
                r = self._fx_renderable(1)
                r.set_mesh(res["radiant_mesh"][0], 0, res["radiant_mesh"][1])
                r.set_material(res["mat"]["radiant"][0])
                rec["parts"]["radiant_glow"] = r
                rec["last"] = 0
            out["burners"].append(rec)
        if self.fx_look["coat"] and any(b["kind"] == "radiant" for b in ep.fx["burners"].values()):
            out["coat"] = {"shown": False, "rend": None, "mesh": None}
        return out

    def _fx_gas_mesh(self, b, level):
        """One mesh with every flicker phase of every part at this level; index ranges per phase and part."""
        V, N, U, I, ranges = [], [], [], [], []
        nv = ni = 0
        for var in self.FX.gas_variants(b["r0"], b["hl"], level, b["key"]):
            rr = {}
            for part in ("gas_inner", "gas_sheath", "gas_haze"):
                v, n, u, i = var[part]
                V.append(v); N.append(n); U.append(u); I.append(i + nv)
                rr[part] = (ni, len(i))
                nv += len(v)
                ni += len(i)
            ranges.append(rr)
        return self._mesh(np.concatenate(V), np.concatenate(N), np.concatenate(U), np.concatenate(I)), ranges

    def _fx_coat(self, ep, data, coat):
        """The clear coat over the glass top: the unit's box (x / y, less a margin) at the zones' plane."""
        FX = self.FX
        bid = ep.fx["body"]
        pb, Rb = data.xpos[bid], data.xmat[bid].reshape(3, 3)
        zs = [float(Rb[:, 2] @ (data.site_xpos[b["site"]] - pb)) for b in ep.fx["burners"].values() if b["kind"] == "radiant"]
        mg = float(self.fx_look["coat_margin_m"])
        lo, hi = ep.fx["lo"][:2] + mg, ep.fx["hi"][:2] - mg
        z = float(np.mean(zs)) + FX.COAT_LIFT
        mk = ("coat",) + tuple(np.round(np.concatenate([lo, hi, [z]]), 5))
        if mk not in self.fx_res["meshes"]:
            V, N, U, I = FX.glass_coat_mesh(lo, hi, z=z)["glass_coat"]
            self.fx_res["meshes"][mk] = (self._mesh(V, N, U, I), len(I))
        coat["mesh"], n = self.fx_res["meshes"][mk]
        r = self._fx_renderable(3)
        r.set_mesh(coat["mesh"], 0, n)
        r.set_material(self.fx_res["mat"]["glass_coat"])
        coat["rend"] = r

    def _fx_state(self, ep, data):
        FX = self.FX
        m = ep.live
        fl = self.live["flames"]
        for rec in fl["burners"]:
            b = rec["b"]
            sid = b["site"]
            if float(m.site_rgba[sid, 3]) <= 0.0:
                if rec["shown"]:
                    for r in rec["parts"].values():
                        self.scn.remove_renderable(r)
                    rec["shown"] = False
                continue
            level = FX.burner_level(m.site_size[sid, 0], b["r0"])
            p = data.site_xpos[sid].astype(np.float32)
            R = data.site_xmat[sid].reshape(3, 3)
            if b["kind"] == "gas":
                nb = int(self.fx_look["gas_buckets"])
                key = (FX.gas_bucket(level, nb), FX.gas_phase(data.time))
                if key != rec["last"]:
                    mk = ("gas", round(b["r0"], 6), round(b["hl"], 6), b["key"], nb, key[0])
                    if mk not in self.fx_res["meshes"]:
                        self.fx_res["meshes"][mk] = self._fx_gas_mesh(b, key[0] / (nb - 1.0))
                    mesh, ranges = self.fx_res["meshes"][mk]
                    for part, r in rec["parts"].items():
                        off, cnt = ranges[key[1]][part]
                        r.set_mesh(mesh, off, cnt)
                    rec["last"] = key
                Rf = R.astype(np.float32).reshape(9)
                for r in rec["parts"].values():
                    r.set_pose(p, Rf)
            else:
                bucket = FX.radiant_bucket(level)
                r = rec["parts"]["radiant_glow"]
                if bucket != rec["last"]:
                    r.set_material(self.fx_res["mat"]["radiant"][bucket])
                    rec["last"] = bucket
                rad = float(m.site_size[sid, 0]) * FX.RADIANT_HALO
                r.set_pose((data.site_xpos[sid] + R[:, 2] * FX.RADIANT_LIFT).astype(np.float32),
                           (R @ np.diag([rad, rad, 1.0])).astype(np.float32).reshape(9))
            if not rec["shown"]:
                for r in rec["parts"].values():
                    self.scn.add_renderable(r)
                rec["shown"] = True
        coat = fl["coat"]
        if coat is not None:
            if coat["rend"] is None:
                self._fx_coat(ep, data, coat)
            bid = ep.fx["body"]
            coat["rend"].set_pose(data.xpos[bid].astype(np.float32), data.xmat[bid].astype(np.float32).reshape(9))
            if not coat["shown"]:
                self.scn.add_renderable(coat["rend"])
                coat["shown"] = True

    def _fx_evict(self, L):
        fl = L.get("flames")
        if not fl:
            return
        for rec in fl["burners"]:
            if rec["shown"]:
                for r in rec["parts"].values():
                    self.scn.remove_renderable(r)
                rec["shown"] = False
        if fl.get("coat") and fl["coat"]["shown"]:
            self.scn.remove_renderable(fl["coat"]["rend"])
            fl["coat"]["shown"] = False
    # -- end of the burner visuals -------------------------------------------------------------------------------------

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
        self.fx_res = None
        self.dress_cache = self.dress_mats = None
        self.box_mesh = None
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
