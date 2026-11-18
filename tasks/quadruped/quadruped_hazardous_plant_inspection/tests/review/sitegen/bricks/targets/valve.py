"""Valves: (a) quarter-turn lever (ball) valve, (b) handwheel gate valve with a rising stem and a position indicator.

(a) Lever valve, face frame (x towards the viewer, y right, z up): the pipe runs along +y (or +z with
pipe_axis='z') through the origin; the stem points along +x; the lever turns in a plane parallel to the face on a
hinge '<name>/lever' about +x. Ground truth `angle_deg` in [0, 90]: 0 = lever parallel to the pipe = OPEN, 90 =
across the pipe = CLOSED; joint angle q = closed_sign * radians(angle_deg).
Reader: segment the saturated lever, fit its line; with the face orientation in the camera frame (hint
'R_cam_face', e.g. from the tag plate's PnP; the tag is mounted parallel to the face) the image line is lifted to
the 3D direction in the lever plane (exact under perspective), otherwise the image angle between the lever and the
pipe's edges is used (foreshortened at oblique views).

(b) Gate valve, face frame origin at the centre of the indicator scale plate (front face); the pipe runs along +y
below it, the stem rises along +z at (x, y) = STEM_XY, the handwheel turns about z at the yoke top. A slide joint
'<name>/stem' (along +z, 0 .. stroke) moves the stem and its pointer; ground truth `opening` in [0, 1].
Reader: rectify the indicator plate (a bright tall quadrilateral), find its end ticks (0 % and 100 %) and the
pointer (the saturated arrow), and interpolate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ..common import (QUAT_Z_TO_X, QUAT_Z_TO_Y, Pose, TargetNames, add_torus_mesh, color_material, fit_font,
                      hinge_range, save_png, texture_material, visual_only)
from . import tag as tagmod

# (reader names in __all__ resolve lazily through __getattr__ below)
__all__ = ["LeverValveSpec", "GateValveSpec", "add_lever_valve", "add_gate_valve", "set_lever", "set_gate",
           "read_lever", "read_gate", "LEVER_COLORS", "indicator_image", "lever_hint", "gate_hint", "add_to_spec",
           "set_state", "read"]

LEVER_COLORS = {"red": (0.80, 0.08, 0.06), "yellow": (0.95, 0.75, 0.05), "blue": (0.08, 0.25, 0.75),
                "orange": (0.95, 0.42, 0.05), "green": (0.10, 0.55, 0.20)}


# ======================================================================================= (a) lever valve
@dataclass
class LeverValveSpec:
    name: str = "HV-3101"
    angle_deg: float = 0.0             # 0 open .. 90 closed
    pipe_od: float = 0.048             # DN40
    pipe_axis: str = "y"               # "y" horizontal across the view, "z" vertical
    pipe_len: float = 0.6              # drawn pipe spool length (0: none, the level designer draws the line)
    lever_len: float = 0.18
    lever_color: str = "red"
    closed_sign: int = 1               # +1: closing turns the lever counter-clockwise as seen (towards +z for a y pipe)
    body_color: tuple = (0.30, 0.31, 0.33, 1.0)
    tag: tagmod.TagSpec | None = None
    tag_offset: tuple | None = None    # (y, z) of the tag centre in the face frame

    @property
    def lever_x(self) -> float:
        """x of the lever plane (in front of the pipe axis)."""
        return 0.62 * self.pipe_od + 0.035


def add_lever_valve(spec, parent, pose: Pose, vs: LeverValveSpec, assets_dir: str) -> TargetNames:
    n = vs.name
    G = mujoco.mjtGeom
    rp = vs.pipe_od / 2
    body = parent.add_body(name=n, pos=list(pose.pos), quat=list(pose.quat))
    names = TargetNames(name=n, kind="lever_valve", body=n, bodies=[n])
    pipe_m = color_material(spec, "bricks/pipe_grey", (0.58, 0.60, 0.60, 1), specular=0.35, shininess=0.4)
    body_m = color_material(spec, n + "/body_mat", vs.body_color, specular=0.3)
    steel = color_material(spec, "bricks/steel", (0.60, 0.61, 0.62, 1), specular=0.5, shininess=0.5)
    lever_rgb = LEVER_COLORS.get(vs.lever_color, LEVER_COLORS["red"])
    lev_m = color_material(spec, n + "/lever_mat", tuple(lever_rgb) + (1.0,), specular=0.25, shininess=0.3)
    names.materials += [body_m, lev_m]
    qa = QUAT_Z_TO_Y if vs.pipe_axis == "y" else [1.0, 0.0, 0.0, 0.0]
    ax = np.array([0, 1, 0]) if vs.pipe_axis == "y" else np.array([0, 0, 1])

    def geom(nm, **kw):
        body.add_geom(name=nm, **kw)
        names.geoms.append(nm)

    if vs.pipe_len > 0:
        geom(n + "/pipe", type=G.mjGEOM_CYLINDER, size=[rp, vs.pipe_len / 2, 0], quat=list(qa), material=pipe_m)
    bl = 2.6 * rp + 0.04
    for s in (-1, 1):
        geom(n + "/flange%+d" % s, type=G.mjGEOM_CYLINDER, size=[1.75 * rp, 0.009, 0],
             pos=list(ax * s * (bl / 2)), quat=list(qa), material=body_m)
    geom(n + "/body", type=G.mjGEOM_SPHERE, size=[1.3 * rp, 0, 0], material=body_m)
    geom(n + "/body_ends", type=G.mjGEOM_CYLINDER, size=[1.1 * rp, bl / 2 - 0.006, 0], quat=list(qa), material=body_m)
    neck0, neck1 = 1.1 * rp, vs.lever_x - 0.006
    geom(n + "/neck", type=G.mjGEOM_CYLINDER, size=[0.45 * rp, (neck1 - neck0) / 2, 0],
         pos=[(neck0 + neck1) / 2, 0, 0], quat=list(QUAT_Z_TO_X), material=body_m)
    geom(n + "/stem", type=G.mjGEOM_CYLINDER, size=[0.007, 0.012, 0], pos=[vs.lever_x - 0.002, 0, 0],
         quat=list(QUAT_Z_TO_X), material=steel)
    lb = body.add_body(name=n + "/lever", pos=[vs.lever_x, 0, 0], gravcomp=1.0)
    lb.explicitinertial = True
    lb.mass = 0.2
    lb.inertia = [2e-4, 2e-4, 2e-4]
    lb.ipos = [0, 0, 0]
    lb.iquat = [1, 0, 0, 0]
    lb.add_joint(name=n + "/lever", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], damping=0.5, armature=1e-3,
                 range=hinge_range(spec, -100.0, 100.0), limited=mujoco.mjtLimited.mjLIMITED_TRUE)
    L = vs.lever_len
    # lever drawn along +y (open, parallel to a y pipe); for a z pipe the lever's open direction is +z
    lq = [1.0, 0.0, 0.0, 0.0] if vs.pipe_axis == "y" else list(
        np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]))
    # the lever geoms get their mass from their volume: this child body needs gravity compensation too, otherwise the
    # lever sags under gravity while the world steps
    ll = lb.add_body(name=n + "/lever_geom", quat=lq, gravcomp=1.0)
    for nm, kw in ((n + "/lever_bar", dict(type=G.mjGEOM_BOX, size=[0.003, L * 0.45, 0.009], pos=[0, L * 0.45, 0],
                                           material=steel)),
                   (n + "/lever_grip", dict(type=G.mjGEOM_BOX, size=[0.0065, L * 0.30, 0.0135],
                                            pos=[0.002, L * 0.66, 0], material=lev_m)),
                   (n + "/lever_hub", dict(type=G.mjGEOM_BOX, size=[0.006, 0.018, 0.013], pos=[0, 0.004, 0],
                                           material=lev_m)),
                   (n + "/lever_nut", dict(type=G.mjGEOM_CYLINDER, size=[0.0075, 0.004, 0], pos=[0.006, 0, 0],
                                           quat=list(QUAT_Z_TO_X), material=steel))):
        ll.add_geom(name=nm, **kw, **visual_only())
        names.geoms.append(nm)
    names.bodies += [n + "/lever", n + "/lever_geom"]
    names.joints.append(n + "/lever")
    names.face_geoms = [n + "/lever_grip", n + "/lever_bar"]
    names.state = {"joint": n + "/lever", "spec": vs}
    if vs.tag is not None:
        tw, th = vs.tag.plate_size_m()
        if vs.tag_offset is not None:
            ty, tz = vs.tag_offset
        elif vs.pipe_axis == "y":
            # below the stem when the lever closes upwards (closed_sign +1), above it otherwise
            ty, tz = 0.0, -vs.closed_sign * (1.75 * rp + 0.02 + th / 2)
        else:
            # beside the pipe on the closing side (marker next to the stem), but below the quadrant the lever sweeps
            # (from +z open to closed_sign * y closed): centred at z = 0 the closed lever lay across the marker
            ty, tz = vs.closed_sign * (1.75 * rp + 0.03 + tw / 2), -(0.0135 + 0.01 + th / 2)
        tx = 1.3 * rp
        if vs.pipe_axis == "y" and abs(tz) > th / 2:
            # a short strap from the valve body to the plate
            edge = tz - math.copysign(th / 2, tz)
            geom(n + "/tag_strap", type=G.mjGEOM_BOX, size=[0.0015, 0.006, abs(edge) / 2 + 0.002],
                 pos=[tx - 0.003, ty, edge / 2], material=steel)
        elif vs.pipe_axis == "z" and abs(ty) > tw / 2 + rp:
            # a short arm from the pipe to the plate
            edge = ty - math.copysign(tw / 2, ty)
            y0 = math.copysign(rp, ty)
            geom(n + "/tag_strap", type=G.mjGEOM_BOX, size=[0.0015, abs(edge - y0) / 2 + 0.002, 0.006],
                 pos=[tx - 0.003, (edge + y0) / 2, tz], material=steel)
        names.tag = tagmod.add_to_spec(spec, body, Pose((tx, ty, tz)), vs.tag, assets_dir, name=n + "/tag")
    return names


def set_lever(model, data, names: TargetNames, angle_deg: float, forward: bool = True):
    vs = names.state["spec"]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names.state["joint"])
    q = vs.closed_sign * math.radians(float(angle_deg))
    if vs.pipe_axis == "z":
        q = -q
    data.qpos[model.jnt_qposadr[jid]] = q
    data.qvel[model.jnt_dofadr[jid]] = 0.0
    vs.angle_deg = float(angle_deg)
    if forward:
        mujoco.mj_kinematics(model, data)


def lever_hint(vs: LeverValveSpec) -> dict:
    return {"type": "lever_valve", "pipe_axis": vs.pipe_axis, "lever_len_m": vs.lever_len, "pipe_od_m": vs.pipe_od}


# ======================================================================================= (b) gate valve
STEM_XY = (-0.035, -0.075)       # stem axis (x, y) relative to the indicator plate centre
PLATE_W, PLATE_H = 0.036, 0.13   # indicator plate (y, z) size
SCALE_SPAN = 0.10                # 0 .. 100 % over this z span, centred on the plate


@dataclass
class GateValveSpec:
    name: str = "HV-3201"
    opening: float = 0.0             # 0 closed .. 1 fully open
    pipe_od: float = 0.089           # DN80
    stroke: float = SCALE_SPAN
    wheel_d: float = 0.25
    wheel_color: tuple = (0.75, 0.10, 0.08, 1.0)
    pointer_color: tuple = (0.95, 0.45, 0.02, 1.0)
    body_color: tuple = (0.22, 0.24, 0.26, 1.0)
    pipe_drop: float = 0.42          # pipe centre below the plate centre
    tag: tagmod.TagSpec | None = None
    tag_offset: tuple | None = None
    texture_px_per_m: float = 6000.0


def indicator_image(gs: GateValveSpec) -> np.ndarray:
    W, H = int(PLATE_W * gs.texture_px_per_m), int(PLATE_H * gs.texture_px_per_m)
    img = Image.new("RGB", (W, H), (242, 240, 228))
    d = ImageDraw.Draw(img)
    ppm = H / PLATE_H
    z0 = H / 2 + SCALE_SPAN / 2 * ppm       # 0 % row (bottom)
    z1 = H / 2 - SCALE_SPAN / 2 * ppm       # 100 % row (top)
    for k in range(0, 21):
        y = z0 + (z1 - z0) * k / 20.0
        major = k % 5 == 0
        x0 = 0
        x1 = W * (0.62 if major else 0.38)
        d.line([(x0, y), (x1, y)], fill=(15, 15, 15), width=max(2, int(H * (0.008 if major else 0.004))))
    d.line([(1, z0), (1, z1)], fill=(15, 15, 15), width=max(2, int(H * 0.005)))
    for k, lab in ((0, "0"), (10, "50"), (20, "100")):
        y = z0 + (z1 - z0) * k / 20.0
        d.text((W * 0.83, y), lab, font=fit_font(d, lab, W * 0.32, H * 0.05, H * 0.045), fill=(15, 15, 15),
               anchor="mm")
    # legends on the right half only (the tick roots on the left stay clean for the reader)
    for y, lab in ((H * 0.035, "OPEN"), (H * 0.965, "SHUT")):
        d.text((W * 0.74, y), lab, font=fit_font(d, lab, W * 0.46, H * 0.04, H * 0.035), fill=(40, 40, 40),
               anchor="mm")
    return np.asarray(img, dtype=np.uint8)


def add_gate_valve(spec, parent, pose: Pose, gs: GateValveSpec, assets_dir: str) -> TargetNames:
    n = gs.name
    G = mujoco.mjtGeom
    sx, sy = STEM_XY
    rp = gs.pipe_od / 2
    body = parent.add_body(name=n, pos=list(pose.pos), quat=list(pose.quat))
    names = TargetNames(name=n, kind="gate_valve", body=n, bodies=[n])
    pipe_m = color_material(spec, "bricks/pipe_grey", (0.58, 0.60, 0.60, 1), specular=0.35, shininess=0.4)
    body_m = color_material(spec, n + "/body_mat", gs.body_color, specular=0.3)
    wheel_m = color_material(spec, n + "/wheel_mat", gs.wheel_color, specular=0.3)
    ptr_m = color_material(spec, n + "/pointer_mat", gs.pointer_color, specular=0.2)
    steel = color_material(spec, "bricks/steel", (0.60, 0.61, 0.62, 1), specular=0.5, shininess=0.5)
    stem_m = color_material(spec, "bricks/stem", (0.70, 0.70, 0.66, 1), specular=0.6, shininess=0.6)
    plate_m = texture_material(spec, n + "/indicator", save_png(indicator_image(gs), assets_dir, "gate_indicator"),
                               specular=0.15)
    names.materials += [body_m, wheel_m, ptr_m, plate_m]

    def geom(nm, b=body, **kw):
        b.add_geom(name=nm, **kw)
        names.geoms.append(nm)

    zp = -gs.pipe_drop
    geom(n + "/pipe", type=G.mjGEOM_CYLINDER, size=[rp, 0.45, 0], pos=[sx, sy, zp], quat=list(QUAT_Z_TO_Y),
         material=pipe_m)
    for s in (-1, 1):
        geom(n + "/flange%+d" % s, type=G.mjGEOM_CYLINDER, size=[1.7 * rp, 0.012, 0], pos=[sx, sy + s * 0.13, zp],
             quat=list(QUAT_Z_TO_Y), material=body_m)
    geom(n + "/body", type=G.mjGEOM_BOX, size=[1.15 * rp, 0.125, 1.25 * rp], pos=[sx, sy, zp + 0.1 * rp],
         material=body_m)
    zb = zp + 1.35 * rp
    geom(n + "/bonnet", type=G.mjGEOM_CYLINDER, size=[0.75 * rp, 0.05, 0], pos=[sx, sy, zb + 0.05], material=body_m)
    zy0, zy1 = zb + 0.10, PLATE_H / 2 + 0.06          # yoke from the bonnet top to above the plate
    for s in (-1, 1):
        geom(n + "/yoke%+d" % s, type=G.mjGEOM_BOX, size=[0.012, 0.010, (zy1 - zy0) / 2],
             pos=[sx, sy + s * 0.055, (zy0 + zy1) / 2], material=body_m)
    geom(n + "/yoke_top", type=G.mjGEOM_BOX, size=[0.025, 0.075, 0.015], pos=[sx, sy, zy1 + 0.015], material=body_m)
    zw = zy1 + 0.045
    tm = add_torus_mesh(spec, "bricks/handwheel_%d" % int(gs.wheel_d * 1000), gs.wheel_d / 2 - 0.012, 0.012)
    geom(n + "/wheel_rim", type=G.mjGEOM_MESH, meshname=tm, pos=[sx, sy, zw], material=wheel_m, **visual_only())
    for k in range(3):
        a = k * 2 * math.pi / 3
        geom(n + "/wheel_spoke%d" % k, type=G.mjGEOM_CAPSULE, size=[0.007, 0, 0],
             fromto=[sx, sy, zw, sx + (gs.wheel_d / 2 - 0.012) * math.cos(a), sy + (gs.wheel_d / 2 - 0.012) * math.sin(a), zw],
             material=wheel_m, **visual_only())
    geom(n + "/wheel_hub", type=G.mjGEOM_CYLINDER, size=[0.022, 0.018, 0], pos=[sx, sy, zw], material=wheel_m)
    # the plate on a bracket from the right yoke arm
    geom(n + "/plate", type=G.mjGEOM_BOX, size=[0.0015, PLATE_W / 2, PLATE_H / 2], pos=[-0.0015, 0, 0],
         material=plate_m)
    geom(n + "/plate_bracket", type=G.mjGEOM_BOX, size=[0.012, 0.012, 0.03], pos=[-0.015, sy + 0.055 + 0.02, 0],
         material=body_m)
    # moving stem with its pointer (slide along +z); at opening 0 the pointer is at the 0 % mark
    st = body.add_body(name=n + "/stem", pos=[0, 0, 0], gravcomp=1.0)
    st.explicitinertial = True
    st.mass = 1.0
    st.inertia = [1e-3, 1e-3, 1e-4]
    st.ipos = [0, 0, 0]
    st.iquat = [1, 0, 0, 0]
    st.add_joint(name=n + "/stem", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], damping=5.0, armature=0.01,
                 range=[-0.005, gs.stroke + 0.005], limited=mujoco.mjtLimited.mjLIMITED_TRUE)
    z_ptr0 = -SCALE_SPAN / 2
    s_lo, s_hi = zb + 0.02, zw + 0.03                   # stem extent at closed
    geom(n + "/stem_rod", b=st, type=G.mjGEOM_CYLINDER, size=[0.011, (s_hi - s_lo) / 2, 0],
         pos=[sx, sy, (s_lo + s_hi) / 2], material=stem_m, **visual_only())
    geom(n + "/stem_cap", b=st, type=G.mjGEOM_CYLINDER, size=[0.013, 0.006, 0], pos=[sx, sy, s_hi + 0.006],
         material=steel, **visual_only())
    geom(n + "/pointer_clamp", b=st, type=G.mjGEOM_BOX, size=[0.016, 0.016, 0.008], pos=[sx, sy, z_ptr0],
         material=steel, **visual_only())
    ya = sy + 0.014
    yb = -PLATE_W / 2 + 0.010
    geom(n + "/pointer_arm", b=st, type=G.mjGEOM_BOX, size=[0.003, (yb - ya) / 2, 0.004],
         pos=[0.0035, (ya + yb) / 2, z_ptr0], material=steel, **visual_only())
    geom(n + "/pointer_link", b=st, type=G.mjGEOM_BOX, size=[(0.0035 - sx) / 2, 0.004, 0.004],
         pos=[(0.0035 + sx) / 2, ya, z_ptr0], material=steel, **visual_only())
    geom(n + "/pointer", b=st, type=G.mjGEOM_BOX, size=[0.0012, 0.009, 0.0035], pos=[0.0015, yb, z_ptr0],
         material=ptr_m, **visual_only())
    names.bodies.append(n + "/stem")
    names.joints.append(n + "/stem")
    names.face_geoms = [n + "/plate", n + "/pointer"]
    names.state = {"joint": n + "/stem", "spec": gs}
    if gs.tag is not None:
        tw, th = gs.tag.plate_size_m()
        ty, tz = gs.tag_offset if gs.tag_offset is not None else (PLATE_W / 2 + 0.02 + tw / 2, 0.0)
        geom(n + "/tag_back", type=G.mjGEOM_BOX, size=[0.0015, tw / 2 + 0.006, th / 2 + 0.006],
             pos=[-0.0035, ty, tz], material=body_m)
        geom(n + "/tag_link", type=G.mjGEOM_BOX, size=[0.0015, (ty - tw / 2 - PLATE_W / 2) / 2 + 0.004, 0.01],
             pos=[-0.0035, (ty - tw / 2 + PLATE_W / 2) / 2, tz], material=body_m)
        names.tag = tagmod.add_to_spec(spec, body, Pose((0.0, ty, tz)), gs.tag, assets_dir, name=n + "/tag")
    return names


def set_gate(model, data, names: TargetNames, opening: float, forward: bool = True):
    gs = names.state["spec"]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names.state["joint"])
    data.qpos[model.jnt_qposadr[jid]] = float(np.clip(opening, 0, 1)) * gs.stroke
    data.qvel[model.jnt_dofadr[jid]] = 0.0
    gs.opening = float(opening)
    if forward:
        mujoco.mj_kinematics(model, data)


def gate_hint(gs: GateValveSpec) -> dict:
    return {"type": "gate_valve", "indicator": "vertical scale 0-100 %, 0 at the bottom", "plate_size_m": (PLATE_W, PLATE_H)}


# ----------------------------------------------------------------------------------------------- uniform API
def add_to_spec(spec, parent, pose, target_spec, assets_dir):
    if isinstance(target_spec, LeverValveSpec):
        return add_lever_valve(spec, parent, pose, target_spec, assets_dir)
    return add_gate_valve(spec, parent, pose, target_spec, assets_dir)


def set_state(model, data, names, value, forward=True):
    if names.kind == "lever_valve":
        return set_lever(model, data, names, value, forward)
    return set_gate(model, data, names, value, forward)


# ----------------------------------------------------------------------------------------------- reader
# The reference reader is in bricks.readers.valve since 2026-11-14 (the cell imports this module for its
# builder and set_state; the readers never ship with the cell). The old names still work: they resolve lazily, through
# this module's __getattr__, to the reader module, which is imported only when a reader name is used.
_READER_MODULE = ".readers.valve"
_READER_NAMES = frozenset(("_lever_mask", "_pipe_direction", "LEVER_UNVERIFIED_CONF", "LEVER_OBLIQUE_CONF",
                           "LEVER_MAX_PIPE_TILT", "LEVER_IMAGE_MAX_CONF", "_hub_pivot", "_pipe_both_sides",
                           "read_lever", "GATE_HI_FRACS", "GATE_MAX_QUADS", "GATE_RETRY_CONF", "_plate_ticks",
                           "GATE_N_TICKS", "GATE_MAJOR_EVERY", "GATE_TICK_BAND", "GATE_MAJOR_BAND", "GATE_ALL_FRACS",
                           "GATE_V_MARGIN", "GATE_SPAN_RATIO_LIMITS", "GATE_MIN_MATCHED", "GATE_PTR_ROWS",
                           "GATE_MAX_GAP", "GATE_END_SLACK", "read_gate", "_gate_from_quads", "read"))


def __getattr__(name):
    if name in _READER_NAMES:
        import importlib
        return getattr(importlib.import_module(_READER_MODULE, __package__.rpartition(".")[0]), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
