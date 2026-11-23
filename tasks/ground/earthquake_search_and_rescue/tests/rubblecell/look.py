"""The colour images of the rubble site (cell internals): MuJoCo's Filament renderer (physically based, Vulkan: Mesa's
lavapipe on the CPU in the sandbox, a GPU's driver where there is one) on a render twin of the scene.

The physics model is never touched. The twins are compiled from a copy of the scene's MjSpec with the same bodies,
joints and cameras, so a frame copies the physics qpos into a twin and runs forward kinematics only:

* the RGB twin: every surface with a physically based material (pbr.py: albedo, normal and ORM maps), the robot's
  paint, rubber and metal, the people's skin and clothes; the set dressing of decor.py (the street's continuation,
  the neighbouring and opposite buildings, overhead wires, details on the upper storeys: all where no part of the robot
  can reach, see decor.py); lit by the sun (a shadow-casting directional light), the sky's image light and the robot's
  two lamps; the sky of sky.py as the backdrop; Filament's ambient occlusion, fog and filmic tone mapping.
* the sensor twin (`depth_model`): the physics geometry plus decor.py's free-standing decor, so the depth images, the
  lidar, the thermal image and the visibility record see what the colour image shows. Decor lives in one static body
  appended after every other body, so the geom and body ids of the physics model are the sensor twin's first ids.

Filament facts this relies on (MuJoCo 3.14, as found for the pump task's look.py): the engine is bound to the thread
that created it, so one thread owns the context and every Filament object, and callers post jobs to it; one model's
Filament objects are live at a time; textures must be marked sRGB (Filament reads MuJoCo's "auto" as linear data); a
geom with a material is drawn in the material's colour.
"""

import atexit
import gc
import math
import os
import queue
import threading
import time


def _cap_malloc_arenas():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").mallopt(-8, 2)
    except (OSError, AttributeError):
        pass


_cap_malloc_arenas()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from . import pbr  # noqa: E402

__all__ = ["LOOK", "available", "look_mode", "build_twins", "Twins", "Renderer", "SceneLook", "RenderError"]

CAM_NEAR, CAM_FAR = 0.01, 400.0
CALL_TIMEOUT = 300.0

# Shadows: type 0 (percentage-closer filtering). The other types (1 variance, 2 and 3 with a blocker search for soft
# edges) gave a handful of different pixels along shadow edges in two fresh processes of the same frame (measured on
# the test lanes, three processes per type); type 0 gave the same bytes every time.
LOOK = {
    "version": "look-3",
    # Filament view options (read from the twin model's custom numeric / text fields)
    "view": {"numeric": {"filament.ao.enabled": 1, "filament.ao.quality": 2, "filament.shadows.type": 0,
                         "filament.shadows.map_size": 2048, "filament.bloom.enabled": 1,
                         "filament.bloom.strength": 0.06, "filament.msaa.enabled": 0,
                         "filament.cg.contrast": 1.10, "filament.cg.saturation": 0.86,
                         "filament.fog.enabled": 1, "filament.fog.density": 0.0016, "filament.fog.distance": 25.0,
                         "filament.fog.height": 0.0, "filament.fog.color": [0.60, 0.585, 0.56],
                         "filament.fallback.head_light_intensity": 0.0,
                         "filament.fallback.scene_light_intensity": 0.0,
                         "filament.fallback.environment_light_intensity": 0.0},
             "text": {"filament.cg.tone_mapping": "filmic"}},
    "ssaa": 2,
    "sun_lux": 95000.0,
    "ibl_intensity": 9000.0,
    "lamp_candela": 2200.0,
    # exposure compensation (EV) of a camera by the light level where it stands: a camera's automatic exposure
    "exposure_ev": {"ext": 0.0, "open": 0.8, "upper": 1.0, "int": 1.9, "void": 2.8},
    # physics material -> (texture set, tint rgb (sRGB), metres one repeat covers or None for the set's own). The sets
    # are photographs (pbr.SHIPPED): the tint is 1 except where one photograph serves several colours.
    "materials": {
        "concrete": ("concrete", (1.0, 1.0, 1.0), 2.71), "concrete_dark": ("concrete_dark", (1.0, 1.0, 1.0), 3.5),
        "slab": ("slab", (1.0, 1.0, 1.0), 3.0), "rubble": ("rubble", (1.0, 1.0, 1.0), 2.0),
        "brick": ("brick", (1.0, 1.0, 1.0), 1.5),
        "plaster_white": ("plaster", (0.98, 0.97, 0.95), 4.6), "plaster_cream": ("plaster", (0.98, 0.91, 0.78), 4.6),
        "plaster_blue": ("plaster", (0.74, 0.83, 0.90), 4.6), "plaster_green": ("plaster", (0.78, 0.88, 0.76), 4.6),
        "plaster_pink": ("plaster", (0.96, 0.80, 0.78), 4.6), "tiles": ("tiles", (1.0, 1.0, 1.0), 2.0),
        "wood": ("wood", (1.0, 1.0, 1.0), 2.0), "wood_dark": ("wood_dark", (1.0, 1.0, 1.0), 2.0),
        "asphalt": ("asphalt", (1.0, 1.0, 1.0), 4.0), "paving": ("paving", (1.0, 1.0, 1.0), 1.8),
        "fabric_red": ("fabric", (0.75, 0.19, 0.16), 0.6), "fabric_blue": ("fabric", (0.25, 0.36, 0.62), 0.6),
        "fabric_grey": ("fabric", (0.56, 0.56, 0.57), 0.6), "fabric_green": ("fabric", (0.32, 0.45, 0.30), 0.6),
        "metal_white": ("paint_metal", (1.0, 1.0, 0.98), 1.0), "metal_red": ("paint_metal", (0.77, 0.12, 0.10), 1.0),
        "metal_blue": ("paint_metal", (0.17, 0.32, 0.62), 1.0), "metal_grey": ("paint_metal", (0.57, 0.58, 0.60), 1.0),
        "rust": ("rust", (1.0, 1.0, 1.0), 1.1), "rebar": ("rust", (0.70, 0.62, 0.58), 0.25), "dust": ("dust", (1.0, 1.0, 1.0), None),
        "tarp_orange": ("fabric", (1.0, 0.52, 0.10), 0.6), "kit": ("bag", (1.0, 1.0, 1.0), 0.26),
        "yellow": ("paint_metal", (1.0, 0.90, 0.10), 1.0), "white_paint": ("road_paint", (1.0, 1.0, 1.0), None),
    },
    # the robot's and the kit's drawn parts (the look pack's robot.npz: meshes per body, inside the task's geoms):
    # look -> (rgb (sRGB), roughness, metallic, emission)
    "robot_looks": {
        "robot_paint": ((0.80, 0.36, 0.10), 0.55, 0.0, 0.0), "robot_dark": ((0.085, 0.09, 0.095), 0.50, 0.0, 0.0),
        "robot_shell": ((0.11, 0.115, 0.12), 0.52, 0.0, 0.0), "robot_alu": ((0.70, 0.71, 0.72), 0.34, 1.0, 0.0),
        "robot_steel": ((0.55, 0.55, 0.56), 0.32, 1.0, 0.0), "robot_rubber": ((0.045, 0.045, 0.047), 0.78, 0.0, 0.0),
        "robot_rubber_worn": ((0.075, 0.072, 0.068), 0.85, 0.0, 0.0), "robot_glass": ((0.015, 0.015, 0.02), 0.05, 0.0, 0.0),
        "robot_lidar_window": ((0.02, 0.02, 0.025), 0.08, 0.0, 0.0), "robot_lens_thermal": ((0.10, 0.05, 0.02), 0.12, 0.6, 0.0),
        "robot_lamp": ((1.0, 0.97, 0.88), 0.15, 0.0, 4.0), "robot_pad": ((0.07, 0.07, 0.075), 0.70, 0.0, 0.0),
        "robot_white": ((0.90, 0.90, 0.88), 0.40, 0.0, 0.0),
        "kit_bag": ((0.72, 0.10, 0.06), 0.75, 0.0, 0.0), "kit_dark": ((0.05, 0.05, 0.055), 0.70, 0.0, 0.0),
        "kit_strap": ((0.93, 0.93, 0.90), 0.55, 0.0, 0.0), "kit_cross": ((0.80, 0.06, 0.05), 0.50, 0.0, 0.0),
        "kit_buckle": ((0.16, 0.16, 0.17), 0.35, 0.0, 0.0), "kit_reflect": ((0.86, 0.88, 0.86), 0.25, 0.6, 0.0),
    },
    # a room's plaster colour -> the wall it has inside (a wallpaper or a painted wall, one picture a storey high);
    # the street front and the stubs around a breach keep broken plaster in the room's colour
    "room_walls": {"plaster_cream": "wall_cream", "plaster_green": "wall_green", "plaster_pink": "wall_pink",
                   "plaster_blue": "wall_blue", "plaster_white": "wall_white"},
    # loose chunks by the material the site gave them: (set, repeat)
    "chunks": {"rubble": ("slab_broken", 1.0), "brick": ("chunk_brick", 1.8), "concrete": ("chunk_plaster", 1.0)},
    # plain materials: rgba (sRGB), roughness, metallic, reflectance, emission
    "plain": {
        "glass": ((0.55, 0.62, 0.66, 0.30), 0.05, 0.0, 0.6, 0.0),
        "window": ((0.05, 0.06, 0.07, 1.0), 0.08, 0.0, 0.7, 0.0),
        "black": ((0.06, 0.06, 0.07, 1.0), 0.55, 0.0, 0.5, 0.0),
        "plastic_skin": ((0.90, 0.86, 0.80, 1.0), 0.35, 0.0, 0.5, 0.0),
    },
}

# the robot's and the people's colours (geoms without a material): rgb -> (set, tint, roughness override or None)
_ROBOT = {(0.93, 0.47, 0.08): ("robot_paint", (0.93, 0.40, 0.05)), (0.18, 0.19, 0.21): ("robot_dark", (0.17, 0.18, 0.19)),
          (0.10, 0.10, 0.10): ("track", (1.0, 1.0, 1.0)), (0.55, 0.57, 0.60): ("alu", (0.82, 0.83, 0.85)),
          (0.12, 0.12, 0.13): ("robot_dark", (0.11, 0.11, 0.12))}


class RenderError(RuntimeError):
    pass


def available():
    try:
        import mujoco._render_filament  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------------------------- the twins
def _key(rgba):
    return tuple(round(float(x), 2) for x in rgba[:3])


def _add_tex(spec, name, img, srgb):
    t = spec.add_texture()
    t.name = name
    t.type = mujoco.mjtTexture.mjTEXTURE_2D
    t.width, t.height = int(img.shape[1]), int(img.shape[0])
    t.nchannel = 3
    t.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_SRGB if srgb else mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR
    t.data = np.ascontiguousarray(img[::-1]).tobytes()      # MuJoCo's row 0 is the image's bottom row
    return name


def _strip_lights(spec):
    for lt in list(spec.lights):
        spec.delete(lt)


# the people's parts by kind: (texture set or None, roughness of a plain one)
_PEOPLE = {"skin": "skin", "hair": "hair", "shirt": "fabric", "trousers": "fabric", "cloth": "fabric"}


class _Bank(object):
    """The render twin's materials, made on demand: a texture set (or plain parameters) at a light level."""

    def __init__(self, spec, sets, levels=None):
        from . import twin as _twin
        self.spec, self.sets, self.levels = spec, sets, dict(levels or _twin.LEVELS)
        self.tex, self.mats = {}, {}
        self.robot_mats = []
        self.figure_tint = {}       # who -> class -> tint (site.figure_tint)
        self.paints = {}            # a real vehicle's paint geom -> the site's paint material

    def _texture(self, setname, kind, level="ext"):
        key = (setname, kind, level)
        if key not in self.tex:
            img = self.sets[setname][kind]
            if kind == "orm" and level != "ext":
                img = img[::2, ::2].copy()
                img[..., 0] = (img[..., 0].astype(np.float32) * self.levels[level] + 0.5).astype(np.uint8)
            self.tex[key] = _add_tex(self.spec, "lk_%s_%s_%s" % (setname, kind, level), img, kind == "albedo")
        return self.tex[key]

    def _orm(self, rough, metal, level):
        key = ("orm", round(rough, 2), round(metal, 2), level)
        if key not in self.tex:
            img = np.zeros((4, 4, 3), np.uint8)
            img[..., 0] = int(255 * self.levels[level] + 0.5)
            img[..., 1] = int(255 * rough + 0.5)
            img[..., 2] = int(255 * metal + 0.5)
            self.tex[key] = _add_tex(self.spec, "lk_orm_%d" % len(self.tex), img, False)
        return self.tex[key]

    def _new(self, key):
        mt = self.spec.add_material()
        mt.name = "lk_m%d" % len(self.mats)
        for role in range(int(mujoco.mjtTextureRole.mjNTEXROLE)):
            mt.textures[role] = ""
        self.mats[key] = mt.name
        return mt

    def textured(self, setname, tint, level, alpha=1.0, robot=False, tile=None):
        tile = float(tile or pbr.TILE_M[setname])
        key = ("t", setname, _key(tint), level, round(float(alpha), 2), round(tile, 3))
        if key in self.mats:
            return self.mats[key]
        mt = self._new(key)
        mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = self._texture(setname, "albedo")
        mt.textures[mujoco.mjtTextureRole.mjTEXROLE_NORMAL] = self._texture(setname, "normal")
        mt.textures[mujoco.mjtTextureRole.mjTEXROLE_ORM] = self._texture(setname, "orm", level)
        mt.texuniform = False
        rep = 1.0 / tile
        mt.texrepeat = [rep, rep]
        mt.rgba = [float(tint[0]), float(tint[1]), float(tint[2]), float(alpha)]
        mt.roughness, mt.metallic, mt.specular, mt.reflectance = 1.0, 1.0, 0.5, 0.0
        if robot:
            self.robot_mats.append(mt.name)
        return mt.name

    def plain(self, rgba, rough, metal, level, refl=0.0, emission=0.0, robot=False):
        key = ("p", tuple(round(float(x), 3) for x in rgba), round(rough, 2), round(metal, 2), level, round(refl, 2),
               round(emission, 2))
        if key in self.mats:
            return self.mats[key]
        mt = self._new(key)
        mt.textures[mujoco.mjtTextureRole.mjTEXROLE_ORM] = self._orm(rough, metal, level)
        mt.rgba = [float(x) for x in rgba]
        # reflectance stays 0: MuJoCo's Filament renderer draws a planar mirror pass of the whole scene for every
        # reflective material (40-80 % of a CPU frame each); glossy surfaces reflect the sky's image light anyway
        mt.roughness, mt.metallic, mt.reflectance, mt.emission = 1.0, 1.0, 0.0, float(emission)
        mt.specular = 0.5
        if robot:
            self.robot_mats.append(mt.name)
        return mt.name

    def hidden(self):
        return self.plain((0.0, 0.0, 0.0, 0.0), 1.0, 0.0, "ext")

    def avatar(self, avatar, part, level, tint=(1.0, 1.0, 1.0)):
        """A posed body's material: its own colour map (the look pack's avatars/<avatar>_<part>_albedo.png, mapped by
        the mesh's own texture coordinates), cloth-rough, at a light level."""
        key = ("a", avatar, part, level, tuple(tint))
        if key in self.mats:
            return self.mats[key]
        tk = ("avatar", avatar, part)
        if tk not in self.tex:
            from PIL import Image
            img = np.asarray(Image.open(os.path.join(pbr.cache_dir(), "avatars", "%s_%s_albedo.png" % (avatar, part))).convert("RGB"))
            self.tex[tk] = _add_tex(self.spec, "lk_avatar_%s_%s" % (avatar, part), img, True)
        mt = self._new(key)
        mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = self.tex[tk]
        mt.textures[mujoco.mjtTextureRole.mjTEXROLE_ORM] = self._orm(0.85, 0.0, level)
        mt.texuniform = False
        mt.texrepeat = [1.0, 1.0]
        mt.rgba = [float(tint[0]), float(tint[1]), float(tint[2]), 1.0]
        mt.roughness, mt.metallic, mt.specular, mt.reflectance = 1.0, 1.0, 0.5, 0.0
        return mt.name

    def prop(self, key, level, robot=False):
        """A look-pack model's material (props.material): its colour map by the model's own texture coordinates, or
        plain parameters; glass is a dark mirror of the sky's image light."""
        from . import props
        d = props.material(key)
        if "tex" not in d:
            if d.get("glass"):
                return self.plain((0.42, 0.47, 0.52, 1.0), 0.05, 1.0, "ext", 0.0, 0.0, robot)
            c = d["rgb"]
            return self.plain((c[0], c[1], c[2], 1.0), float(d.get("rough", 0.6)), float(d.get("metal", 0.0)), level, 0.0,
                              float(d.get("emission", 0.0)), robot)
        mk = ("pr", key, level)
        if mk in self.mats:
            return self.mats[mk]
        tk = ("prop", key)
        if tk not in self.tex:
            from PIL import Image
            img = np.asarray(Image.open(props.texture_path(key)).convert("RGB"))
            self.tex[tk] = _add_tex(self.spec, "lk_prop_%s" % key, img, True)
        mt = self._new(mk)
        mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = self.tex[tk]
        mt.textures[mujoco.mjtTextureRole.mjTEXROLE_ORM] = self._orm(0.75, 0.0, level)
        mt.texuniform = False
        mt.texrepeat = [1.0, 1.0]
        mt.rgba = [1.0, 1.0, 1.0, 1.0]
        mt.roughness, mt.metallic, mt.specular, mt.reflectance = 1.0, 1.0, 0.5, 0.0
        return mt.name

    def assign(self, g, level, robot, what):
        """Set geom g's material; True when it is textured (a box of it is then drawn as a textured mesh)."""
        name = g.material
        if (g.name or "").startswith("veh_") and g.type == mujoco.mjtGeom.mjGEOM_MESH:   # a real vehicle (scene._vehicle_meshes)
            cls = g.name.split("_")[-1]
            if cls == "paint":
                c = LOOK["materials"].get(self.paints.get(g.name, ""), (None, (0.5, 0.5, 0.52)))[1]
                c = tuple(0.72 * x + 0.28 * d for x, d in zip(c, (0.62, 0.59, 0.54)))             # dusted
                g.material = self.plain((c[0], c[1], c[2], 1.0), 0.45, 0.0, level, 0.0, 0.0, robot)
            else:
                rgba, rough = {"glass": ((0.42, 0.47, 0.52, 1.0), 0.05), "dark": ((0.05, 0.05, 0.05, 1.0), 0.8),
                               "lamp": ((0.75, 0.72, 0.68, 1.0), 0.3), "other": ((0.33, 0.33, 0.34, 1.0), 0.6)}[cls]
                g.material = self.plain(rgba, rough, 1.0 if cls == "glass" else 0.0, level, 0.0, 0.0, robot)
            return False
        if (g.name or "").startswith("prop_") and g.type == mujoco.mjtGeom.mjGEOM_MESH:   # a look-pack model
            from . import props
            _p, _k, i, model = g.name.split("_", 3)
            g.material = self.prop(props.parts(model)[int(i)][0], level, robot)
            return False
        if "_body_" in (g.name or "") and g.type == mujoco.mjtGeom.mjGEOM_MESH:      # a posed body (scene._figure_meshes)
            who, _b, part, cls = g.name.split("_")
            if cls == "plastic":
                g.material = self.plain((0.86, 0.82, 0.76, 1.0), 0.32, 0.0, level, 0.5, 0.0, robot)
            else:
                from .human import figures
                g.material = self.avatar(figures()["side"]["avatar"], part, level,
                                         self.figure_tint.get(who, {}).get(cls, (1.0, 1.0, 1.0)))
            return False
        if name in LOOK["materials"]:
            setname, tint, tile = LOOK["materials"][name]
            gname, bname = g.name or "", (g.parent.name or "") if g.parent is not None else ""
            if gname.startswith("chunk") and name in LOOK["chunks"]:
                setname, tile = LOOK["chunks"][name]
            elif name in LOOK["room_walls"] and level != "ext":
                if max(float(v) for v in g.size[:3]) >= 0.65:          # a wall of a room (1.3 m or longer)
                    setname, tint, tile = LOOK["room_walls"][name], (1.0, 1.0, 1.0), 3.0
                else:
                    tile = 1.85
            elif name in LOOK["room_walls"] and level == "ext":         # a street front: the same wall, calmer
                setname, tile = "plaster_ext", 4.6
            elif gname.startswith("legslab"):                        # the broken plate over the survivor's legs
                setname, tile = "slab_broken", 1.2
            elif bname.startswith("propslab"):
                setname, tile = "slab_broken", 1.6
            elif bname.startswith("prop_") and setname == "wood":
                tint, tile = (0.95, 0.88, 0.74), 0.9
            elif bname.startswith("debris") and setname in ("wood", "wood_dark"):
                setname, tint, tile = "beam", ((1.0, 0.94, 0.87) if setname == "wood" else (0.61, 0.55, 0.51)), 0.5
            g.material = self.textured(setname, tint, level, 1.0, robot, tile)
            return True
        if name in LOOK["plain"]:
            rgba, rough, metal, refl, emis = LOOK["plain"][name]
            g.material = self.plain(rgba, rough, metal, level, refl, emis, robot)
            return False
        rgba = [float(x) for x in g.rgba]
        k = _key(rgba)
        if name:                                   # a material this look does not know: its own colour
            g.material = self.plain((0.6, 0.6, 0.6, 1.0), 0.7, 0.0, level, robot=robot)
            return False
        if k in _ROBOT:
            setname, tint = _ROBOT[k]
            g.material = self.textured(setname, tint, level, 1.0, robot)
            return True
        if k == (1.0, 0.98, 0.85):                # the lamps' lenses
            g.material = self.plain((1.0, 0.97, 0.88, 1.0), 0.15, 0.0, "ext", 0.5, 4.0, robot)
            return False
        if what in _PEOPLE:
            g.material = self.textured(_PEOPLE[what], rgba[:3], level, 1.0, robot)
            return True
        if what == "plastic":
            g.material = self.plain((0.86, 0.82, 0.76, 1.0), 0.32, 0.0, level, 0.5, 0.0, robot)
            return False
        g.material = self.plain(rgba, 0.7, 0.0, level, 0.35, 0.0, robot)
        return False


# the replays' cut-away views open the storeys over the robot: brighter interiors than the robot's own cameras see
VIDEO_LEVELS = {"ext": 1.0, "open": 0.8, "upper": 0.6, "int": 0.55, "void": 0.38}


def make_rgb_spec(scene, sets, decor_fn=None, batch=True, levels=None):
    """The RGB twin's spec: a copy of the scene's spec, re-dressed (the physics spec is not modified)."""
    from . import twin as _twin
    spec = scene.info["spec"].copy()
    _strip_lights(spec)
    bank = _Bank(spec, sets, levels)
    bank.paints = dict(getattr(scene.site, "vehicle_paint", {}))
    bank.figure_tint = dict(getattr(scene.site, "figure_tint", {}))
    body = spec.worldbody.add_body()
    body.name = "lk_decor"
    levels = _twin.dress(spec, scene, bank, body)
    _twin.add_terrain(spec, body, scene, bank)
    if decor_fn is not None:
        decor_fn(spec, body, bank)
    for g in list(spec.worldbody.geoms):                      # the heightfield: the terrain meshes draw it
        if g.type == mujoco.mjtGeom.mjGEOM_HFIELD:
            spec.delete(g)
    for g in list(spec.worldbody.geoms):                      # the capsules inside the posed bodies: contact only
        if int(g.group) == 3:
            spec.delete(g)
    bank.batched = _twin.batch_statics(spec, [spec.worldbody, body]) if batch else (0, 0)
    bank.robot_drawn = _twin.draw_robot(spec, bank, LOOK["robot_looks"])
    bank.robot_batched = _twin.batch_robot(spec, bank) if batch else (0, 0)
    for k, v in sorted(LOOK["view"]["numeric"].items()):
        spec.add_numeric(name=k, data=[float(x) for x in (v if isinstance(v, (list, tuple)) else [v])])
    for k, v in sorted(LOOK["view"]["text"].items()):
        spec.add_text(name=k, data=str(v))
    return spec, bank, levels


def make_depth_spec(scene, decor_fn=None):
    spec = scene.info["spec"].copy()
    if decor_fn is not None:
        body = spec.worldbody.add_body()
        body.name = "lk_decor"
        decor_fn(spec, body, None)
    return spec


class Twins(object):
    """The scene's render twin (Filament colour) and sensor twin (classic depth, lidar, segmentation)."""

    def __init__(self, scene, sets=None, decor_fn=None, sensor_twin=True, batch=True, levels=None):
        t0 = time.time()
        self.scene = scene
        sets = sets if sets is not None else pbr.texture_sets()
        self.rgb_spec, self.bank, self.levels = make_rgb_spec(scene, sets, decor_fn, batch=batch, levels=levels)
        self.rgb_model = self.rgb_spec.compile()
        from . import twin as _twin
        gid = lambda n: mujoco.mj_name2id(self.rgb_model, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        self.robot_copies = {lv: np.array([gid(n) for n in names], int)
                             for lv, names in getattr(self.bank, "robot_copies", {}).items()}
        self.robot_level = None
        self.rgb_data = mujoco.MjData(self.rgb_model)
        if sensor_twin:
            self.depth_spec = make_depth_spec(scene, decor_fn)
            self.depth_model = self.depth_spec.compile()
            self.depth_data = mujoco.MjData(self.depth_model)
        else:
            self.depth_spec = self.depth_model = self.depth_data = None
        m = scene.model
        for tw in (self.rgb_model, self.depth_model):
            if tw is not None and (tw.nq != m.nq or tw.nbody < m.nbody or tw.ncam != m.ncam):
                raise RenderError("twin layout differs from the physics model")
        self.build_s = time.time() - t0

    def robot_light_level(self, data):
        """Where the robot stands, as the light level of its drawn version (its chassis' centre)."""
        from . import twin as _twin
        m = self.scene.model
        ch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        p = np.array(data.xpos[ch]) + np.array([0.0, 0.0, 0.15])
        S = self.scene.site
        return _twin.ROBOT_LEVEL_OF[_twin.classify_point(S.plan, S.W, S.D, p)]

    def sync(self, data):
        for tm, td in ((self.rgb_model, self.rgb_data), (self.depth_model, self.depth_data)):
            if tm is None:
                continue
            td.qpos[:] = data.qpos
            if tm.nmocap:
                td.mocap_pos[:] = data.mocap_pos
                td.mocap_quat[:] = data.mocap_quat
            mujoco.mj_kinematics(tm, td)
            mujoco.mj_camlight(tm, td)
        if self.robot_copies:                       # one version of the robot drawn: the others at zero size
            self.robot_level = self.robot_light_level(data)
            for lv, ids in self.robot_copies.items():
                if lv != self.robot_level and len(ids):
                    self.rgb_data.geom_xmat[ids] = 0.0


# --------------------------------------------------------------------------------------------- the renderer
class _Filament(object):
    """One thread owns Filament's context and every Filament object; callers post jobs."""

    _inst = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._inst is None or not cls._inst.thread.is_alive():
                cls._inst = _Filament()
            return cls._inst

    def __init__(self):
        self.jobs = queue.Queue()
        self.ready = threading.Event()
        self.err = None
        self.live = None          # (sid, objs, scn, lights, rends, extra lights)
        self.sessions = {}
        self.targets = {}
        self.next_sid = 1
        self.thread = threading.Thread(target=self._run, name="rubblecell-filament", daemon=True)
        self.thread.start()
        self.ready.wait(120)
        if self.err is not None:
            raise RenderError("Filament did not start: %s" % self.err)

    def call(self, fn, *args):
        done = threading.Event()
        out = []
        self.jobs.put((fn, args, out, done))
        if not done.wait(CALL_TIMEOUT):
            raise RenderError("the renderer did not answer in %.0f s" % CALL_TIMEOUT)
        if out and isinstance(out[0], BaseException):
            raise RenderError("%s: %s" % (type(out[0]).__name__, out[0]))
        return out[0] if out else None

    def stop(self):
        try:
            self.jobs.put(None)
            self.thread.join(10)
        except Exception:  # noqa: BLE001
            pass

    def _run(self):
        try:
            self._start()
        except BaseException as e:  # noqa: BLE001
            self.err = e
            self.ready.set()
            return
        self.ready.set()
        while True:
            job = self.jobs.get()
            if job is None:
                self._shutdown()
                return
            fn, args, out, done = job
            try:
                out.append(fn(self, *args))
            except BaseException as e:  # noqa: BLE001
                out.append(e)
            done.set()

    def _start(self):
        for k in ("DISPLAY", "WAYLAND_DISPLAY"):
            os.environ.pop(k, None)
        try:
            ncpu = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            ncpu = os.cpu_count() or 4
        # lavapipe's rasteriser threads: the CPUs this process may use, at most 8 (the image's LP_NUM_THREADS=2 is the
        # classic renderer's default; the program waits for its frame, so the renderer may use them all meanwhile)
        os.environ["LP_NUM_THREADS"] = str(max(1, min(8, ncpu)))
        import mujoco._render_filament as F
        from mujoco.experimental.studio import window  # noqa: F401  registers Filament's resource providers
        self.F = F
        self.ctx = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN))
        self.ctx.set_clear_color([0.0, 0.0, 0.0])

    # -- sessions (the renderer's thread only)
    def j_new(self, model, sky):
        sid = self.next_sid
        self.next_sid += 1
        self.sessions[sid] = {"model": model, "sky": sky}
        return sid

    def j_free(self, sid):
        self.sessions.pop(sid, None)
        if self.live and self.live["sid"] == sid:
            self._evict()

    def _evict(self):
        if self.live:
            for lt in self.live["all_lights"]:
                try:
                    self.live["scn"].remove_light(lt)
                except Exception:  # noqa: BLE001
                    pass
            self.live = None
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

    def _texture(self, raw, sampler):
        F = self.F
        tc = F.TextureConfig()
        tc.width, tc.height, tc.format = len(raw), 0, F.PixelFormat.PIXEL_FORMAT_KTX
        tc.sampler_type, tc.color_space = sampler, F.ColorSpace.COLORSPACE_LINEAR
        tex = self.ctx.create_texture(tc)
        td = F.TextureData()
        td.set_image(raw)
        tex.set_data(td)
        return tex, td

    def _activate(self, sid):
        if self.live and self.live["sid"] == sid:
            return self.live
        self._evict()
        F = self.F
        ses = self.sessions[sid]
        model, sky = ses["model"], ses["sky"]
        objs = F.ModelObjects(self.ctx, model)
        scn = self.ctx.create_scene(F.SceneParams())
        scn.configure_from_model(model)
        lights = F.ModelLights(scn, objs)
        rends = F.ModelRenderables(scn, objs)
        keep, all_lights = [], []
        ibl_tex, ibl_td = self._texture(sky["ibl_ktx"], F.SamplerType.TEXTURE_CUBE)
        keep += [ibl_tex, ibl_td]
        ibl = self.ctx.create_light(F.LightParams(type=F.LightType.LIGHT_IMAGE, texture=ibl_tex,
                                                  color=np.array([1.0, 1.0, 1.0], np.float32),
                                                  intensity=float(sky["ibl_intensity"])))
        scn.add_light(ibl)
        all_lights.append(ibl)
        if sky.get("skybox_ktx"):
            sb_tex, sb_td = self._texture(sky["skybox_ktx"], F.SamplerType.TEXTURE_SKYBOX)
            keep += [sb_tex, sb_td]
            scn.set_skybox(sb_tex)
        sun = self.ctx.create_light(F.LightParams(type=F.LightType.LIGHT_DIRECTIONAL,
                                                  color=np.asarray(sky["sun_color"], np.float32),
                                                  intensity=float(sky["sun_lux"]), cast_shadows=True,
                                                  shadow_map_size=int(sky.get("shadow_map", 2048))))
        sun.set_transform(np.zeros(3, np.float32), np.asarray(sky["sun_dir"], np.float32))
        scn.add_light(sun)
        all_lights.append(sun)
        lamps = []
        for lp in sky.get("lamps", []):
            L = self.ctx.create_light(F.LightParams(type=F.LightType.LIGHT_SPOT, color=np.asarray(lp["color"], np.float32),
                                                    intensity=float(lp["candela"]), cast_shadows=bool(lp.get("shadow", True)),
                                                    spot_cone_angle=float(lp["cone_deg"]), spot_softness=float(lp.get("soft", 0.6)),
                                                    range=float(lp.get("range", 25.0)), shadow_map_size=1024,
                                                    bulb_radius=float(lp.get("bulb", 0.02))))
            scn.add_light(L)
            all_lights.append(L)
            lamps.append(L)
        points = []
        for pl in sky.get("points", []):
            L = self.ctx.create_light(F.LightParams(type=F.LightType.LIGHT_POINT, color=np.asarray(pl["color"], np.float32),
                                                    intensity=float(pl["candela"]), cast_shadows=False,
                                                    range=float(pl.get("range", 15.0)), bulb_radius=0.05))
            L.set_transform(np.asarray(pl["pos"], np.float32), np.array([0.0, 0.0, -1.0], np.float32))
            scn.add_light(L)
            all_lights.append(L)
            points.append(L)
        self.live = {"sid": sid, "objs": objs, "scn": scn, "lights": lights, "rends": rends, "keep": keep,
                     "lamps": lamps, "points": points, "all_lights": all_lights}
        return self.live

    def _target(self, w, h):
        F = self.F
        key = (w, h)
        if key not in self.targets:
            tgt = self.ctx.create_render_target(F.RenderTargetConfig(color_format=F.PixelFormat.PIXEL_FORMAT_RGB8))
            tgt.resize(w, h)
            rd = F.ReadPixelsRequest()
            rd.target = tgt
            rd.alloc(w * h * 3)
            self.targets[key] = (tgt, rd)
        return self.targets[key]

    def j_draw(self, sid, data, lamp_poses, pos, forward, up, fovy, w, h, near, far, gain=1.0):
        F = self.F
        live = self._activate(sid)
        live["lights"].update(data)
        live["rends"].update(data)
        for L, (p, d) in zip(live["lamps"], lamp_poses):
            L.set_transform(np.asarray(p, np.float32), np.asarray(d, np.float32))
        sky = self.sessions[sid]["sky"]
        if live.get("gain") != gain:                         # the camera's exposure: every light scaled alike
            live["all_lights"][0].set_intensity(float(sky["ibl_intensity"]) * gain)
            live["all_lights"][1].set_intensity(float(sky["sun_lux"]) * gain)
            for L, lp in zip(live["lamps"], sky.get("lamps", [])):
                L.set_intensity(float(lp["candela"]) * gain)
            for L, pl in zip(live.get("points", []), sky.get("points", [])):
                L.set_intensity(float(pl["candela"]) * gain)
            live["gain"] = gain
        tgt, rd = self._target(w, h)
        top = near * math.tan(math.radians(fovy) / 2.0)
        cam = F.Camera(pos=[float(x) for x in pos], forward=[float(x) for x in forward], up=[float(x) for x in up],
                       frustum_center=0.0, frustum_width=top * w / h, frustum_bottom=-top, frustum_top=top,
                       frustum_near=near, frustum_far=far)
        req = F.RenderRequest(camera=cam, draw_mode=F.DrawMode.DRAW_MODE_DEFAULT)
        req.enable_reflections = False      # no planar mirror passes (see _Bank.plain)
        req.scene, req.target = live["scn"], tgt
        req.viewport.left, req.viewport.bottom, req.viewport.width, req.viewport.height = 0, 0, w, h
        self.ctx.wait_for_frame(self.ctx.render([req], [rd]))
        return np.frombuffer(rd.buffer(), np.uint8).reshape(h, w, 3).copy()

    def _shutdown(self):
        self.sessions.clear()
        self._evict()
        self.targets.clear()
        gc.collect()
        self.ctx = None
        gc.collect()


@atexit.register
def _stop_renderer():
    if _Filament._inst is not None:
        _Filament._inst.stop()


_SRGB_LUT = None


def _downsample(img, s):
    """Average s x s pixel blocks in linear light (sRGB in and out)."""
    global _SRGB_LUT
    if s <= 1:
        return img
    if _SRGB_LUT is None:
        c = np.arange(256) / 255.0
        _SRGB_LUT = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)
    lin = _SRGB_LUT[img]
    h, w = img.shape[0] // s, img.shape[1] // s
    lin = lin[:h * s, :w * s].reshape(h, s, w, s, 3).mean(axis=(1, 3))
    out = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.clip(lin, 0, None), 1 / 2.4) - 0.055)
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


class Renderer(object):
    """Colour frames of a scene's RGB twin: from the twin's cameras or any pose."""

    def __init__(self, twins, sky, ssaa=None):
        self.tw = twins
        self.sky = sky
        self.ssaa = int(ssaa or LOOK["ssaa"])
        self.fil = _Filament.get()
        self.sid = self.fil.call(_Filament.j_new, twins.rgb_model, sky)
        m = twins.scene.model
        self.lamp_ids = [i for i in range(m.nlight) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, i) or "").startswith("lamp")]

    def close(self):
        if self.sid is not None:
            try:
                self.fil.call(_Filament.j_free, self.sid)
            except Exception:  # noqa: BLE001
                pass
            self.sid = None

    def _lamps(self, data):
        """The robot's lamp poses from the physics data (their lights are the physics model's)."""
        return [(np.array(data.light_xpos[i]), np.array(data.light_xdir[i])) for i in self.lamp_ids]

    def camera(self, data, cam, width, height, near=CAM_NEAR, far=CAM_FAR):
        tm, td = self.tw.rgb_model, self.tw.rgb_data
        cid = cam if isinstance(cam, int) else mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        R = td.cam_xmat[cid].reshape(3, 3)
        return self.view(data, td.cam_xpos[cid], -R[:, 2], R[:, 1], float(tm.cam_fovy[cid]), width, height, near, far)

    def exposure_gain(self, pos, forward):
        """A camera's automatic exposure by the light level where it stands and where it looks (2^EV)."""
        from . import twin as _twin
        S = self.tw.scene.site
        ev = LOOK["exposure_ev"]
        p = np.asarray(pos, float)
        f = np.asarray(forward, float)
        pts = [p, p + 0.6 * f, p + 1.2 * f]
        vals = [ev[_twin.classify_point(S.plan, S.W, S.D, q)] for q in pts]
        return float(2.0 ** (0.5 * vals[0] + 0.3 * vals[1] + 0.2 * vals[2]))

    def view(self, data, pos, forward, up, fovy, width, height, near=CAM_NEAR, far=CAM_FAR, ssaa=None, gain=None):
        s = int(ssaa or self.ssaa)
        if gain is None:
            gain = self.exposure_gain(pos, forward)
        img = self.fil.call(_Filament.j_draw, self.sid, self.tw.rgb_data, self._lamps(data), pos, forward, up, fovy,
                            int(width * s), int(height * s), near, far, gain)
        return _downsample(img, s)


def look_mode():
    """'filament' (the default) or 'classic' (RUBBLECELL_LOOK=classic: MuJoCo's classic colour, for development)."""
    return "classic" if os.environ.get("RUBBLECELL_LOOK", "").strip().lower() == "classic" else "filament"


class SceneLook(object):
    """A scene's look for one episode: the twins (render and sensor), the sky, the lights, the Filament renderer.

    quality "agent": what the robot's cameras get (a 512 px skybox, a 64 px image light, 2x supersampling);
    "video": the replays' cinematic quality (a 1024 px skybox, a 128 px image light)."""

    def __init__(self, scene, decor=True, quality="agent", sets=None):
        from . import decor as _decor
        from . import sky as _sky
        t0 = time.time()
        self.scene = scene
        self.decor = _decor.decor_fn(scene) if decor else None
        video = quality == "video"
        self.twins = Twins(scene, sets=sets, decor_fn=self.decor, sensor_twin=not video, batch=not video,
                           levels=VIDEO_LEVELS if video else None)
        plan = scene.site.plan
        seed = int(plan.get("seed", 0)) % 1000003
        sizes = {"agent": (64, 512), "video": (128, 1024)}[quality]
        self.sky = _sky.make_sky(plan["env"], seed, ibl_size=sizes[0], sky_size=sizes[1], sun_lux=LOOK["sun_lux"],
                                 ibl_intensity=LOOK["ibl_intensity"])
        lamp = {"color": [1.0, 0.97, 0.92], "candela": LOOK["lamp_candela"], "cone_deg": 0.62, "soft": 0.5}
        self.sky["lamps"] = [dict(lamp), dict(lamp)]
        self.sky["points"] = _decor.lights(self.decor) if self.decor is not None else []
        self.renderer = Renderer(self.twins, self.sky)
        self.build_s = time.time() - t0

    def rgb(self, data, cam, width, height):
        """The colour image of camera `cam` (the twins synced to `data` by the caller)."""
        return self.renderer.camera(data, cam, width, height)

    def close(self):
        try:
            self.renderer.close()
        except Exception:  # noqa: BLE001
            pass


def build_twins(scene, decor=True, sensor_twin=True):
    fn = None
    if decor:
        from . import decor as _decor
        fn = _decor.decor_fn(scene)
    return Twins(scene, decor_fn=fn, sensor_twin=sensor_twin)
