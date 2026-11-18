"""Local panel with 3 to 6 indicator lamps (22 mm pilot lights: red / amber / green / white) and a legend strip.

Face frame (x out, y right, z up): origin at the centre of the lamp row on the panel's front face; lamps are spaced
`pitch` along +y, the legend strip is below them, the enclosure lies behind x = 0. A lamp's lens geom
'<name>/lens<k>' switches between an ON material (emissive, light) and an OFF material (dull, dark tint of the same
colour) through model.geom_matid, so set_state needs no recompile. Ground truth: list of booleans, left to right as
the viewer sees the panel.

Reader: find the round black bezels (a dark ring, or a dark disc when the lamp is off) of similar size on one line,
take the n nearest the image centre, and call a lamp ON when its lens is much brighter than the panel around it or
saturated.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ..common import (QUAT_Z_TO_X, Pose, TargetNames, add_annulus_mesh, color_material, fit_font, save_png,
                      texture_material, visual_only)
from . import tag as tagmod

# (reader names in __all__ resolve lazily through __getattr__ below)
__all__ = ["LampPanelSpec", "LAMP_COLORS", "add_to_spec", "set_state", "read", "public_hint", "legend_image"]

LAMP_COLORS = {"red": (0.92, 0.10, 0.08), "amber": (1.0, 0.62, 0.05), "green": (0.10, 0.82, 0.25),
               "white": (0.95, 0.95, 0.92)}
LENS_R = 0.0115
BEZEL_R = 0.0165


@dataclass
class LampPanelSpec:
    name: str = "LP-0101"
    colors: tuple = ("green", "amber", "red", "white")
    states: tuple = (True, False, False, True)
    labels: tuple = ("RUN", "WARN", "TRIP", "POWER")
    pitch: float = 0.065
    panel_rgb: tuple = (0.76, 0.77, 0.75, 1.0)     # RAL 7035 light grey
    panel_h: float = 0.20
    panel_d: float = 0.12
    tag: tagmod.TagSpec | None = None
    tag_offset: tuple | None = None

    @property
    def n(self) -> int:
        return len(self.colors)

    @property
    def panel_w(self) -> float:
        return self.n * self.pitch + 0.05


def legend_image(ps: LampPanelSpec, px_per_m=4000) -> np.ndarray:
    W, H = int((ps.n * ps.pitch) * px_per_m), int(0.028 * px_per_m)
    img = Image.new("RGB", (W, H), (245, 245, 240))
    d = ImageDraw.Draw(img)
    for k, lab in enumerate(ps.labels):
        cx = (k + 0.5) * ps.pitch * px_per_m
        d.text((cx, H / 2), lab, font=fit_font(d, lab, ps.pitch * px_per_m * 0.85, H * 0.7, H * 0.6), fill=(15, 15, 15),
               anchor="mm")
    d.rectangle([0, 0, W - 1, H - 1], outline=(40, 40, 40), width=3)
    return np.asarray(img, dtype=np.uint8)


def _lamp_materials(spec, color):
    c = np.array(LAMP_COLORS[color])
    on = color_material(spec, "bricks/lamp_%s_on" % color, tuple(0.65 * c + 0.35) + (1.0,), specular=0.2,
                        shininess=0.5, emission=1.0)
    off = color_material(spec, "bricks/lamp_%s_off" % color, tuple(0.42 * c + 0.04) + (1.0,), specular=0.55,
                         shininess=0.7)
    return on, off


def add_to_spec(spec, parent, pose: Pose, ps: LampPanelSpec, assets_dir: str) -> TargetNames:
    n = ps.name
    G = mujoco.mjtGeom
    body = parent.add_body(name=n, pos=list(pose.pos), quat=list(pose.quat))
    names = TargetNames(name=n, kind="lamp_panel", body=n, bodies=[n])
    panel = color_material(spec, n + "/panel_mat", ps.panel_rgb, specular=0.15, shininess=0.2)
    black = color_material(spec, "bricks/bezel_black", (0.05, 0.05, 0.05, 1), specular=0.3)
    leg = texture_material(spec, n + "/legend", save_png(legend_image(ps), assets_dir, "legend"), specular=0.1)

    def geom(nm, **kw):
        body.add_geom(name=nm, **kw)
        names.geoms.append(nm)

    zl = 0.025
    geom(n + "/enclosure", type=G.mjGEOM_BOX, size=[ps.panel_d / 2, ps.panel_w / 2, ps.panel_h / 2],
         pos=[-ps.panel_d / 2, 0, zl - 0.03], material=panel)
    geom(n + "/legend", type=G.mjGEOM_BOX, size=[0.001, ps.n * ps.pitch / 2, 0.014], pos=[0.001, 0, -0.012],
         material=leg)
    bm = add_annulus_mesh(spec, "bricks/lamp_bezel", LENS_R - 0.0005, BEZEL_R, 0.011, 0.0)
    mats = []
    for k, col in enumerate(ps.colors):
        y = (k - (ps.n - 1) / 2.0) * ps.pitch
        on, off = _lamp_materials(spec, col)
        mats.append((on, off))
        geom(n + "/bezel%d" % k, type=G.mjGEOM_MESH, meshname=bm, pos=[0, y, zl], material=black, **visual_only())
        geom(n + "/lens%d" % k, type=G.mjGEOM_CYLINDER, size=[LENS_R, 0.004, 0], pos=[0.0055, y, zl],
             quat=list(QUAT_Z_TO_X), material=on if ps.states[k] else off, **visual_only())
        geom(n + "/dome%d" % k, type=G.mjGEOM_ELLIPSOID, size=[0.003, LENS_R * 0.95, LENS_R * 0.95],
             pos=[0.0095, y, zl], material=on if ps.states[k] else off, **visual_only())
    names.face_geoms = [n + "/lens%d" % k for k in range(ps.n)]
    names.state = {"spec": ps, "lenses": [[n + "/lens%d" % k, n + "/dome%d" % k] for k in range(ps.n)],
                   "materials": mats}
    if ps.tag is not None:
        tw, th = ps.tag.plate_size_m()
        ty, tz = ps.tag_offset if ps.tag_offset is not None else (0.0, zl - 0.03 - ps.panel_h / 2 - 0.012 - th / 2)
        if tz < zl - 0.03 - ps.panel_h / 2:
            geom(n + "/tag_hanger", type=G.mjGEOM_BOX, size=[0.0015, 0.008, 0.012],
                 pos=[0.0, ty, zl - 0.03 - ps.panel_h / 2 - 0.004], material=black)
        names.tag = tagmod.add_to_spec(spec, body, Pose((0.003, ty, tz)), ps.tag, assets_dir, name=n + "/tag")
    return names


def set_state(model, data, names: TargetNames, states, forward: bool = False):
    ps = names.state["spec"]
    for k, (geoms, (on, off)) in enumerate(zip(names.state["lenses"], names.state["materials"])):
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, on if states[k] else off)
        for g in geoms:
            model.geom_matid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)] = mid
    ps.states = tuple(bool(s) for s in states)
    if forward:
        mujoco.mj_kinematics(model, data)


def public_hint(ps: LampPanelSpec) -> dict:
    return {"type": "lamp_panel", "n_lamps": ps.n, "colors": tuple(ps.colors), "labels": tuple(ps.labels)}


# ----------------------------------------------------------------------------------------------- reader
# The reference reader is in bricks.readers.lamps since 2026-11-14 (the cell imports this module for its
# builder and set_state; the readers never ship with the cell). The old names still work: they resolve lazily, through
# this module's __getattr__, to the reader module, which is imported only when a reader name is used.
_READER_MODULE = ".readers.lamps"
_READER_NAMES = frozenset(('_dark_circles', '_pick_row', 'read'))


def __getattr__(name):
    if name in _READER_NAMES:
        import importlib
        return getattr(importlib.import_module(_READER_MODULE, __package__.rpartition(".")[0]), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
