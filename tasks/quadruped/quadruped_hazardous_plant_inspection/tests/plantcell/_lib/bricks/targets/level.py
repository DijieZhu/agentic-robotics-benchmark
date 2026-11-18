"""Tubular level gauge on a tank: a glass tube between two fittings, a coloured liquid column, a white backing strip
behind the tube and a graduated scale plate (0 .. 100 %) to its right.

Face frame (x out, y right, z up): origin at the centre of the visible glass window on the tube axis' plane; the tube
axis is at y = 0, the window spans z in [-L/2, +L/2], the scale plate's 0 % and 100 % marks are exactly at the window
ends. Ground truth `level` in [0, 1] = liquid height / window length. State change: the liquid column geom
'<name>/liquid' is resized and re-posed in the model (geom_size / geom_pos / geom_rbound); it has no contacts.

Reader: find the tall bright scale plate, rectify it with its corners (homography), read its tick rows (0 % at the
lowest, 100 % at the highest major tick), then scan the tube strip to the plate's left for the transition between the
coloured column and the empty glass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ..common import Pose, TargetNames, color_material, fit_font, save_png, texture_material, visual_only
from . import tag as tagmod

# (reader names in __all__ resolve lazily through __getattr__ below)
__all__ = ["LevelGaugeSpec", "LIQUIDS", "scale_image", "add_to_spec", "set_state", "read", "public_hint"]

LIQUIDS = {"amber": (0.85, 0.45, 0.05, 1.0), "water": (0.20, 0.45, 0.80, 1.0), "green": (0.10, 0.60, 0.25, 1.0),
           "oil": (0.35, 0.18, 0.05, 1.0)}
TUBE_R = 0.012
PLATE_W = 0.05
PLATE_GAP = 0.016                # gap between the tube's right edge and the plate (clears the fittings)
FIT_GAP = 0.03                   # the fittings start this far beyond the window ends (beyond the plate's 25 mm margin)


@dataclass
class LevelGaugeSpec:
    name: str = "LG-1101"
    window_m: float = 0.5
    level: float = 0.5
    liquid: str = "amber"
    tank_gap: float = 0.10           # from the tube axis back to the tank wall (nozzle length)
    tag: tagmod.TagSpec | None = None
    tag_offset: tuple | None = None
    texture_px_per_m: float = 2400.0

    @property
    def plate_y(self) -> float:
        return TUBE_R + PLATE_GAP + PLATE_W / 2


def scale_image(ls: LevelGaugeSpec) -> np.ndarray:
    """Plate image: height = window + 2 x 25 mm margins; ticks every 5 %, labels every 20 %."""
    margin = 0.025
    H = int((ls.window_m + 2 * margin) * ls.texture_px_per_m)
    W = int(PLATE_W * ls.texture_px_per_m)
    img = Image.new("RGB", (W, H), (244, 244, 238))
    d = ImageDraw.Draw(img)
    ppm = H / (ls.window_m + 2 * margin)
    y0 = H - margin * ppm
    y1 = margin * ppm
    for k in range(21):
        y = y0 + (y1 - y0) * k / 20.0
        major = k % 2 == 0
        d.line([(0, y), (W * (0.45 if major else 0.28), y)], fill=(15, 15, 15),
               width=max(2, int(W * (0.035 if major else 0.02))))
    for k in range(0, 21, 4):
        y = y0 + (y1 - y0) * k / 20.0
        lab = "%d" % (k * 5)
        d.text((W * 0.74, y), lab, font=fit_font(d, lab, W * 0.46, ppm * 0.03, ppm * 0.028), fill=(15, 15, 15),
               anchor="mm")
    d.text((W * 0.74, y1 - margin * ppm * 0.55), "%", font=fit_font(d, "%", W * 0.4, ppm * 0.02, ppm * 0.02),
           fill=(40, 40, 40), anchor="mm")
    return np.asarray(img, dtype=np.uint8)


def add_to_spec(spec, parent, pose: Pose, ls: LevelGaugeSpec, assets_dir: str) -> TargetNames:
    n = ls.name
    G = mujoco.mjtGeom
    L = ls.window_m
    body = parent.add_body(name=n, pos=list(pose.pos), quat=list(pose.quat))
    names = TargetNames(name=n, kind="level_gauge", body=n, bodies=[n])
    steel = color_material(spec, "bricks/steel", (0.60, 0.61, 0.62, 1), specular=0.5, shininess=0.5)
    glass = color_material(spec, "bricks/level_glass", (0.85, 0.90, 0.92, 0.22), specular=0.6, shininess=0.8)
    back = color_material(spec, "bricks/level_backing", (0.93, 0.93, 0.90, 1), specular=0.1)
    liq = color_material(spec, n + "/liquid_mat", LIQUIDS.get(ls.liquid, LIQUIDS["amber"]), specular=0.4,
                         shininess=0.6)
    plate = texture_material(spec, n + "/scale", save_png(scale_image(ls), assets_dir, "level_scale_%d" % int(L * 1000)),
                             specular=0.12)
    names.materials += [liq, plate]

    def geom(nm, **kw):
        body.add_geom(name=nm, **kw)
        names.geoms.append(nm)

    zf = L / 2 + FIT_GAP + 0.03            # fitting centres: the fittings start beyond the scale plate's ends
    geom(n + "/backing", type=G.mjGEOM_BOX, size=[0.002, 0.016, zf - 0.03], pos=[-TUBE_R - 0.004, 0, 0],
         material=back)
    geom(n + "/glass", type=G.mjGEOM_CYLINDER, size=[TUBE_R, zf - 0.03, 0], material=glass, **visual_only())
    # liquid: sized for the initial level; set_state resizes it
    lv = float(np.clip(ls.level, 0.0, 1.0))
    hh = max(1e-4, lv * L / 2)
    geom(n + "/liquid", type=G.mjGEOM_CYLINDER, size=[TUBE_R * 0.78, hh, 0], pos=[0, 0, -L / 2 + hh],
         material=liq, **visual_only())
    for s, nm in ((-1, "bottom"), (1, "top")):
        geom(n + "/fitting_" + nm, type=G.mjGEOM_BOX, size=[0.026, 0.024, 0.03], pos=[-0.004, 0, s * zf],
             material=steel)
        geom(n + "/nozzle_" + nm, type=G.mjGEOM_CYLINDER, size=[0.013, ls.tank_gap / 2, 0],
             pos=[-ls.tank_gap / 2 - 0.01, 0, s * zf], quat=[0.7071068, 0, 0.7071068, 0], material=steel)
        geom(n + "/valve_" + nm, type=G.mjGEOM_BOX, size=[0.012, 0.018, 0.012], pos=[0.022, 0, s * zf],
             material=steel)
    geom(n + "/drain", type=G.mjGEOM_CYLINDER, size=[0.007, 0.02, 0], pos=[-0.004, 0, -(zf + 0.05)],
         material=steel)
    py = ls.plate_y
    ph = L / 2 + 0.025
    geom(n + "/plate", type=G.mjGEOM_BOX, size=[0.0015, PLATE_W / 2, ph], pos=[-0.0015, py, 0], material=plate)
    geom(n + "/plate_arm", type=G.mjGEOM_BOX, size=[0.003, 0.02, 0.01], pos=[-0.006, py - 0.02, zf],
         material=steel)
    names.face_geoms = [n + "/glass", n + "/plate"]
    names.state = {"liquid": n + "/liquid", "spec": ls}
    if ls.tag is not None:
        tw, th = ls.tag.plate_size_m()
        ty, tz = ls.tag_offset if ls.tag_offset is not None else (py + PLATE_W / 2 + 0.02 + tw / 2, -L / 2 + th / 2)
        names.tag = tagmod.add_to_spec(spec, body, Pose((0.0, ty, tz)), ls.tag, assets_dir, name=n + "/tag")
    return names


def set_state(model, data, names: TargetNames, level: float, forward: bool = True):
    ls = names.state["spec"]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, names.state["liquid"])
    L = ls.window_m
    lv = float(np.clip(level, 0.0, 1.0))
    hh = max(1e-4, lv * L / 2)
    model.geom_size[gid, 1] = hh
    model.geom_pos[gid] = [0.0, 0.0, -L / 2 + hh]
    model.geom_rbound[gid] = math.hypot(model.geom_size[gid, 0], hh)
    ls.level = float(level)
    if forward:
        mujoco.mj_kinematics(model, data)


def public_hint(ls: LevelGaugeSpec) -> dict:
    return {"type": "level_gauge", "range": (0.0, 100.0), "unit": "%", "window_m": ls.window_m,
            "scale_side": "right", "plate_size_m": (PLATE_W, ls.window_m + 0.05),
            "tube_offset_m": TUBE_R + PLATE_GAP, "tube_od_m": 2 * TUBE_R}


# ----------------------------------------------------------------------------------------------- reader
# The reference reader is in bricks.readers.level since 2026-11-14 (the cell imports this module for its
# builder and set_state; the readers never ship with the cell). The old names still work: they resolve lazily, through
# this module's __getattr__, to the reader module, which is imported only when a reader name is used.
_READER_MODULE = ".readers.level"
_READER_NAMES = frozenset(("N_TICKS", "MAJOR_EVERY", "TICK_BAND", "MAJOR_BAND", "LEVEL_HI_FRACS", "V_MARGIN",
                           "SPAN_RATIO_LIMITS", "MIN_COMB_MATCHED", "_plate_comb", "_span_ratio", "_rect_homography",
                           "read"))


def __getattr__(name):
    if name in _READER_NAMES:
        import importlib
        return getattr(importlib.import_module(_READER_MODULE, __package__.rpartition(".")[0]), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
