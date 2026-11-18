"""Round analog dial gauge (EN 837 style): case, thin bezel, white printed dial, black needle on a hinge.

Dial convention (public, like every real dial): the scale sweeps 270 degrees clockwise, from the low end at the 7:30
position to the high end at 4:30; the gap of 90 degrees is at the bottom (6 o'clock). In the face frame (x out, y
right, z up) the needle is a child body '<name>/needle' with a hinge '<name>/needle' about +x; joint angle 0 points
the needle straight up (mid scale) and a positive angle turns it counter-clockwise as the viewer sees it:
    q(value) = radians(135 - 270 * u),  u = (value - lo) / (hi - lo)  (clipped to the stop pegs at -1.5 % / 101.5 %).
The needle body has gravcomp = 1, damping and no contacts, so it stays where set_state put it when the world steps.

Sizes: nominal case diameter 63 / 100 / 160 mm; the printed dial is 0.9 of it, the bezel lip stands 4 % of the
diameter in front of the dial (it hides the dial's edge at grazing angles, as real bezels do).

Reader: find the bright elliptical dial, rectify it affinely, find the needle as the darkest ray between 18 % and
55 % of the dial radius, find the scale's end ticks around the 90 degree gap, and interpolate the needle angle between
them (no knowledge of the render: only the hint's range and the 270 degree convention as a sanity check).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ..common import (QUAT_Z_TO_X, Pose, TargetNames, add_annulus_mesh, add_disc_mesh, color_material, font,
                      hinge_range, save_png, texture_material, visual_only)
from . import tag as tagmod

# (reader names in __all__ resolve lazily through __getattr__ below)
__all__ = ["GaugeSpec", "SCALES", "dial_image", "add_to_spec", "set_state", "read", "needle_angle", "public_hint",
           "value_from_qpos", "random_spec"]

SWEEP_DEG = 270.0
START_DEG = 225.0          # low end, counter-clockwise from 3 o'clock as the viewer sees it
PEG_FRAC = 0.015

# key: (lo, hi, unit, major, minor)
SCALES = {
    "0-6bar": (0.0, 6.0, "bar", 1.0, 0.2),
    "0-10bar": (0.0, 10.0, "bar", 1.0, 0.2),
    "0-16bar": (0.0, 16.0, "bar", 2.0, 0.5),
    "0-25bar": (0.0, 25.0, "bar", 5.0, 1.0),
    "0-120C": (0.0, 120.0, "°C", 20.0, 5.0),
    "0-250C": (0.0, 250.0, "°C", 50.0, 10.0),
    "-1-3bar": (-1.0, 3.0, "bar", 1.0, 0.2),
}


@dataclass
class GaugeSpec:
    name: str = "PI-2041"
    size_mm: float = 100.0
    scale: str = "0-10bar"
    value: float = 5.0
    band: tuple = ()                       # ((v0, v1, (r, g, b)), ...) printed arcs, e.g. a red over-range band
    case: str = "black"                    # "black" | "stainless"
    needle_rgb: tuple = (18, 18, 18)
    brand: str = ""                        # small text above the centre (optional)
    mount: str = "bracket"                 # "bracket" (plate + arm to a wall behind) | "stub" (pipe from below) | "none"
    standoff: float = 0.06                 # bracket: case back to wall (m); stub: stub length below the socket
    tag: tagmod.TagSpec | None = None
    tag_offset: tuple | None = None        # (y, z) of the tag plate centre in the face frame; None = default
    glass: bool = False
    texture_px: int = 1024

    @property
    def range(self):
        lo, hi = SCALES[self.scale][:2]
        return lo, hi

    @property
    def D(self) -> float:
        return self.size_mm / 1000.0


def needle_angle(gs: GaugeSpec, value: float) -> float:
    """Hinge angle (rad) of the needle showing `value`."""
    lo, hi = gs.range
    u = float(np.clip((value - lo) / (hi - lo), -PEG_FRAC, 1 + PEG_FRAC))
    return math.radians(135.0 - SWEEP_DEG * u)


def value_from_qpos(gs: GaugeSpec, q: float) -> float:
    lo, hi = gs.range
    u = (135.0 - math.degrees(q)) / SWEEP_DEG
    return lo + u * (hi - lo)


def _fmt(v, step):
    if abs(step - round(step)) < 1e-9:
        return "%d" % int(round(v))
    return "%.1f" % v if abs(v - round(v)) > 1e-9 else "%d" % int(round(v))


def dial_image(gs: GaugeSpec) -> np.ndarray:
    """Square RGB image whose inscribed circle is the printed dial."""
    lo, hi, unit, major, minor = SCALES[gs.scale]
    px = int(gs.texture_px)
    img = Image.new("RGB", (px, px), (246, 246, 242))
    d = ImageDraw.Draw(img)
    c = px / 2.0
    R = px / 2.0

    def pt(theta, r):
        return c + r * math.cos(theta), c - r * math.sin(theta)

    def ang(v):
        return math.radians(START_DEG - SWEEP_DEG * (v - lo) / (hi - lo))

    # printed bands just outside the ticks
    for v0, v1, rgb in gs.band:
        a0, a1 = math.degrees(ang(v1)), math.degrees(ang(v0))
        r0, r1 = 0.93 * R, 0.985 * R
        # PIL angles are clockwise from 3 o'clock
        d.pieslice([c - r1, c - r1, c + r1, c + r1], -a1, -a0, fill=tuple(rgb))
        d.pieslice([c - r0, c - r0, c + r0, c + r0], -a1 - 1, -a0 + 1, fill=(246, 246, 242))
    n_minor = int(round((hi - lo) / minor))
    for k in range(n_minor + 1):
        v = lo + k * minor
        is_major = abs((v - lo) / major - round((v - lo) / major)) < 1e-6
        r_in = 0.76 * R if is_major else 0.84 * R
        wdt = max(2, int(round(px * (0.009 if is_major else 0.0045))))
        d.line([pt(ang(v), r_in), pt(ang(v), 0.93 * R)], fill=(12, 12, 12), width=wdt)
    # the arc joining the tick roots
    d.arc([c - 0.93 * R, c - 0.93 * R, c + 0.93 * R, c + 0.93 * R], -START_DEG + 0.0, -START_DEG + SWEEP_DEG,
          fill=(12, 12, 12), width=max(1, px // 400))
    f = font(px * 0.085)
    n_major = int(round((hi - lo) / major))
    for k in range(n_major + 1):
        v = lo + k * major
        x, y = pt(ang(v), 0.61 * R)
        d.text((x, y), _fmt(v, major), font=f, fill=(12, 12, 12), anchor="mm")
    d.text((c, c + 0.42 * R), unit, font=font(px * 0.085), fill=(12, 12, 12), anchor="mm")
    d.text((c, c + 0.60 * R), "Cl. 1.0", font=font(px * 0.04), fill=(60, 60, 60), anchor="mm")
    if gs.brand:
        d.text((c, c - 0.33 * R), gs.brand, font=font(px * 0.045), fill=(70, 70, 70), anchor="mm")
    return np.asarray(img, dtype=np.uint8)


def _case_rgba(case):
    return (0.12, 0.12, 0.13, 1.0) if case == "black" else (0.62, 0.63, 0.64, 1.0)


def add_to_spec(spec, parent, pose: Pose, gs: GaugeSpec, assets_dir: str) -> TargetNames:
    """Add the gauge with its face frame at `pose` (origin at the dial centre on the dial surface)."""
    n = gs.name
    D = gs.D
    Rd = 0.45 * D
    depth = 0.34 * D
    body = parent.add_body(name=n, pos=list(pose.pos), quat=list(pose.quat))
    names = TargetNames(name=n, kind="gauge", body=n, bodies=[n])
    case_mat = color_material(spec, "bricks/gauge_case_%s" % gs.case, _case_rgba(gs.case), specular=0.5,
                              shininess=0.5)
    dial_mat = texture_material(spec, n + "/dial", save_png(dial_image(gs), assets_dir, "dial_" + gs.scale),
                                specular=0.12, shininess=0.15)
    needle_mat = color_material(spec, n + "/needle_mat", tuple(v / 255.0 for v in gs.needle_rgb) + (1.0,),
                                specular=0.3, shininess=0.4)
    dark = color_material(spec, "bricks/dark_steel", (0.22, 0.22, 0.23, 1), specular=0.4)
    steel = color_material(spec, "bricks/steel", (0.60, 0.61, 0.62, 1), specular=0.5, shininess=0.5)
    names.materials += [case_mat, dial_mat, needle_mat]

    def geom(nm, **kw):
        body.add_geom(name=nm, **kw)
        names.geoms.append(nm)
        return nm

    G = mujoco.mjtGeom
    # case (cylinder along x), behind the dial
    geom(n + "/case", type=G.mjGEOM_CYLINDER, size=[0.5 * D, depth / 2, 0], pos=[-depth / 2 - 0.001, 0, 0],
         quat=list(QUAT_Z_TO_X), material=case_mat)
    # dial disc (textured mesh, front face at x = 0)
    dm = add_disc_mesh(spec, "bricks/dial_disc_%d" % int(round(gs.size_mm)), Rd, 0.001)
    geom(n + "/dial", type=G.mjGEOM_MESH, meshname=dm, material=dial_mat, **visual_only())
    # bezel ring with a front lip
    bm = add_annulus_mesh(spec, "bricks/bezel_%d" % int(round(gs.size_mm)), 0.435 * D, 0.505 * D, 0.04 * D,
                          -0.02 * D)
    geom(n + "/bezel", type=G.mjGEOM_MESH, meshname=bm, material=case_mat, **visual_only())
    # needle: child body on a hinge about +x, drawn pointing up at q = 0
    nb = body.add_body(name=n + "/needle", pos=[0.022 * D, 0, 0], gravcomp=1.0)
    nb.explicitinertial = True
    nb.mass = 0.002
    nb.inertia = [1e-7, 1e-7, 1e-7]
    nb.ipos = [0, 0, 0]
    nb.iquat = [1, 0, 0, 0]
    nb.add_joint(name=n + "/needle", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], damping=0.02,
                 armature=1e-5, range=hinge_range(spec, -140.0, 140.0),
                 limited=mujoco.mjtLimited.mjLIMITED_TRUE)
    t = max(0.0004, 0.005 * D)
    L1, L2, Lt = 0.50 * Rd, 0.90 * Rd, 0.20 * Rd
    w1, w2 = 0.040 * Rd, 0.020 * Rd
    for nm, kw in ((n + "/needle_base", dict(type=G.mjGEOM_BOX, size=[t / 2, w1 / 2, L1 / 2], pos=[0, 0, L1 / 2])),
                   (n + "/needle_tip", dict(type=G.mjGEOM_BOX, size=[t / 2, w2 / 2, (L2 - L1) / 2],
                                            pos=[0, 0, (L1 + L2) / 2])),
                   (n + "/needle_tail", dict(type=G.mjGEOM_BOX, size=[t / 2, w1 * 0.8, Lt / 2], pos=[0, 0, -Lt / 2])),
                   (n + "/needle_hub", dict(type=G.mjGEOM_CYLINDER, size=[0.065 * Rd, t, 0], pos=[0, 0, 0],
                                            quat=list(QUAT_Z_TO_X)))):
        nb.add_geom(name=nm, material=needle_mat, **kw, **visual_only())
        names.geoms.append(nm)
    names.bodies.append(n + "/needle")
    names.joints.append(n + "/needle")
    if gs.glass:
        gm = color_material(spec, "bricks/glass", (0.9, 0.95, 1.0, 0.12), specular=0.9, shininess=0.9)
        geom(n + "/glass", type=G.mjGEOM_CYLINDER, size=[0.44 * D, 0.0005, 0], pos=[0.035 * D, 0, 0],
             quat=list(QUAT_Z_TO_X), material=gm, **visual_only())
    # tag placement (face frame) and mounting hardware
    tag_pos = None
    if gs.tag is not None:
        tw, th = gs.tag.plate_size_m()
        if gs.tag_offset is not None:
            ty, tz = gs.tag_offset
        elif gs.mount == "stub":
            ty, tz = 0.5 * D + 0.015 + tw / 2, -0.5 * D - 0.03 - th / 2
        else:
            ty, tz = 0.0, -0.5 * D - 0.015 - th / 2
        tag_pos = (ty, tz, tw, th)
    if gs.mount == "bracket":
        # a mounting plate behind the case (extended down behind the tag) and an arm back to the wall
        y0, y1, z0, z1 = -0.3 * D, 0.3 * D, -0.3 * D, 0.3 * D
        if tag_pos is not None:
            ty, tz, tw, th = tag_pos
            y0, y1 = min(y0, ty - tw / 2 - 0.008), max(y1, ty + tw / 2 + 0.008)
            z0, z1 = min(z0, tz - th / 2 - 0.008), max(z1, tz + th / 2 + 0.008)
        geom(n + "/plate", type=G.mjGEOM_BOX, size=[0.002, (y1 - y0) / 2, (z1 - z0) / 2],
             pos=[-depth - 0.003, (y0 + y1) / 2, (z0 + z1) / 2], material=dark)
        arm_len = max(0.005, gs.standoff)
        geom(n + "/arm", type=G.mjGEOM_BOX, size=[arm_len / 2, 0.012, 0.0025],
             pos=[-depth - 0.005 - arm_len / 2, 0, 0], material=steel)
        names.state["wall_x"] = -depth - 0.005 - arm_len
    elif gs.mount == "stub":
        geom(n + "/socket", type=G.mjGEOM_BOX, size=[0.011, 0.011, 0.012], pos=[-depth / 2, 0, -0.5 * D - 0.012],
             material=steel)
        sl = max(0.02, gs.standoff)
        geom(n + "/stub", type=G.mjGEOM_CYLINDER, size=[0.0107, sl / 2, 0],
             pos=[-depth / 2, 0, -0.5 * D - 0.024 - sl / 2], material=steel)
        names.state["stub_bottom_z"] = -0.5 * D - 0.024 - sl
    names.face_geoms = [n + "/dial"]
    names.state.update({"joint": n + "/needle", "spec": gs})
    if tag_pos is not None:
        ty, tz, tw, th = tag_pos
        tx = -depth + 0.001 if gs.mount == "bracket" else -depth / 2 + 0.015
        names.tag = tagmod.add_to_spec(spec, body, Pose((tx, ty, tz)), gs.tag, assets_dir, name=n + "/tag")
    return names


def set_state(model, data, names: TargetNames, value: float, forward: bool = True):
    """Point the needle at `value` (in the scale's unit)."""
    gs = names.state["spec"]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names.state["joint"])
    data.qpos[model.jnt_qposadr[jid]] = needle_angle(gs, value)
    data.qvel[model.jnt_dofadr[jid]] = 0.0
    gs.value = float(value)
    if forward:
        mujoco.mj_kinematics(model, data)


def public_hint(gs: GaugeSpec) -> dict:
    """What a work order says about the gauge (no state)."""
    lo, hi, unit = SCALES[gs.scale][:3]
    return {"type": "dial_gauge", "range": (lo, hi), "unit": unit, "size_mm": gs.size_mm}


def random_spec(rng, name="PI-0000", size_mm=None, scale=None, **kw) -> GaugeSpec:
    scale = scale or rng.choice(list(SCALES))
    lo, hi = SCALES[scale][:2]
    band = ()
    if rng.random() < 0.4:
        band = ((lo + 0.8 * (hi - lo), hi, (200, 30, 30)),)
        if rng.random() < 0.5:
            band = band + ((lo + 0.3 * (hi - lo), lo + 0.6 * (hi - lo), (40, 150, 60)),)
    return GaugeSpec(name=name, size_mm=size_mm or float(rng.choice([63, 100, 160])), scale=scale,
                     value=float(rng.uniform(lo, hi)), band=band, case=str(rng.choice(["black", "stainless"])),
                     brand=str(rng.choice(["", "", "WIKA-like", "BRICKS"])), **kw)


# ----------------------------------------------------------------------------------------------- reader
# The reference reader is in bricks.readers.gauge since 2026-11-14 (the cell imports this module for its
# builder and set_state; the readers never ship with the cell). The old names still work: they resolve lazily, through
# this module's __getattr__, to the reader module, which is imported only when a reader name is used.
_READER_MODULE = ".readers.gauge"
_READER_NAMES = frozenset(('_needle_line', '_rectify', '_read_on_ellipse', 'read'))


def __getattr__(name):
    if name in _READER_NAMES:
        import importlib
        return getattr(importlib.import_module(_READER_MODULE, __package__.rpartition(".")[0]), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
