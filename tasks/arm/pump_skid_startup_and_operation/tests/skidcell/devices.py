"""MjSpec builders of the skid's devices (cell internals).

Frame: the skid's front plane is y = 0, everything an operator touches or reads faces -y (towards the robot's rail);
x runs along the skid, z is up, the floor is z = 0. Every builder takes `(b, tag, pos, ...)` where `b` is a `Builder`
(the spec, its materials and its texture cache) and returns a plain dict descriptor; names start with `<tag>/`.

Operating conventions (cell_manual.md):
* a hinge of an operable device turns about the axis (0, -1, 0), the axis pointing at the operator, so a positive
  angle is a COUNTER-CLOCKWISE turn as the operator sees it; valves open counter-clockwise ("lefty loosey"); the
  joint angle 0 is closed (handwheels, levers, needle valves) / OFF (the selector) / chamber A (the changeover lever);
* a lever points along its pipe when the valve is open and across it when closed (quarter-turn ball valves);
* buttons travel 8 mm along +y into the panel; a press registers past 4 mm.

Handles: each grippable device lists its handles in its moving body's frame: `pos`, `pinch` (the axis the gripper's
fingers must close along, or None for a round knob: any axis perpendicular to the stem), `approach` (the stem axis the
gripper's approach must lie along: +y, into the device) and `width` (the pinch width). `physics.Scene` turns them into
world poses for the grip rule.

Runtime dependencies: mujoco + numpy (+ Pillow through textures).
"""

import math

import mujoco
import numpy as np

from . import textures as tx

__all__ = ["Builder", "add_pipe", "add_handwheel_valve", "add_lever_valve", "add_knob_valve", "add_selector",
           "add_button", "add_lamp", "add_dial_gauge", "add_rotameter", "add_level_gauge", "add_sight_glass",
           "add_display", "add_tank", "add_drum", "add_pump", "add_dosing_pump", "add_plate", "add_tag", "SEGMENTS"]

GEOM = mujoco.mjtGeom
Y_AXIS_OUT = [0.0, -1.0, 0.0]      # the operable hinges' axis (towards the operator)

# 7-segment layout: segment -> (centre x, centre z, horizontal?) in units of the digit (w = 1, h = 2)
SEGMENTS = {"a": (0.0, 1.0, True), "b": (0.5, 0.5, False), "c": (0.5, -0.5, False), "d": (0.0, -1.0, True),
            "e": (-0.5, -0.5, False), "f": (-0.5, 0.5, False), "g": (0.0, 0.0, True)}
DIGIT_SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg", "6": "acdefg",
                  "7": "abc", "8": "abcdefg", "9": "abcdfg", "-": "g", " ": ""}


def _q_axis_angle(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = math.sin(ang / 2.0)
    return [math.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s]


def _q_x_to(direction_xz_angle):
    """Quaternion that turns local +x into the face-plane direction (cos a, 0, sin a) (a counter-clockwise from +x as the
    operator sees it): a rotation about +y by -a."""
    return _q_axis_angle([0.0, 1.0, 0.0], -float(direction_xz_angle))


class Builder(object):
    """The spec plus shared materials and in-memory textures."""

    def __init__(self, spec):
        self.spec = spec
        self.w = spec.worldbody
        self._tex = 0
        self.mats = {}
        for name, rgba, spec_ in (("steel", (0.62, 0.64, 0.66, 1), 0.4), ("frame", (0.25, 0.30, 0.38, 1), 0.2),
                                  ("pipe", (0.72, 0.73, 0.74, 1), 0.5), ("valve", (0.30, 0.42, 0.33, 1), 0.3),
                                  ("dark", (0.16, 0.17, 0.18, 1), 0.2), ("tank", (0.78, 0.80, 0.82, 1), 0.3),
                                  ("pump", (0.18, 0.34, 0.62, 1), 0.3), ("motor", (0.25, 0.27, 0.30, 1), 0.3),
                                  ("panel", (0.86, 0.86, 0.82, 1), 0.1), ("rubber", (0.08, 0.08, 0.08, 1), 0.05)):
            spec.add_material(name="mat/" + name, rgba=list(rgba), specular=spec_, shininess=0.3)
            self.mats[name] = "mat/" + name

    def tex_material(self, img, name=None):
        """A 2D texture from an RGB uint8 array plus a material that shows it; returns the material name."""
        self._tex += 1
        name = name or "tex%03d" % self._tex
        h, w = int(img.shape[0]), int(img.shape[1])
        t = self.spec.add_texture(name="t/" + name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=w, height=h, nchannel=3)
        t.data = np.ascontiguousarray(img, dtype=np.uint8).tobytes()
        m = self.spec.add_material(name="m/" + name, specular=0.1, shininess=0.1)
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "t/" + name
        return "m/" + name

    def material(self, rgba, name=None):
        self._tex += 1
        name = name or "c%03d" % self._tex
        self.spec.add_material(name="m/" + name, rgba=list(rgba), specular=0.2, shininess=0.3)
        return "m/" + name


def _static(geom_kw=None, visual=False):
    kw = dict(geom_kw or {})
    if visual:
        kw.update(contype=0, conaffinity=0, group=1)
    return kw


def add_plate(b, parent, name, centre, half_w, half_h, img=None, material=None, thick=0.0015, visual=True):
    """A thin box facing -y (its -y face at y = centre[1] - thick) showing `img` (or a plain material)."""
    mat = material if material is not None else b.tex_material(img)
    parent.add_geom(name=name, type=GEOM.mjGEOM_BOX, size=[half_w, thick, half_h], pos=list(centre), material=mat,
                    **_static(visual=visual))
    return name


def add_tag(b, parent, tag, centre, text=None, half_w=0.034, half_h=0.013):
    """The device's tag plate (a stainless plate with its site tag) facing -y: about 68 x 26 mm, legible from close up
    (the wrist camera), not from across the cell."""
    return add_plate(b, parent, "%s/tag" % tag, centre, half_w, half_h, tx.tag_plate(text or tag, w_px=272, h_px=104))


def add_pipe(b, parent, name, p0, p1, r=0.024, material=None, visual=False):
    parent.add_geom(name=name, type=GEOM.mjGEOM_CYLINDER, fromto=list(p0) + list(p1), size=[r, 0, 0],
                    material=material or b.mats["pipe"], **_static(visual=visual))
    return name


# ----------------------------------------------------------------------------------------------- valves
def _valve_body(b, tag, pos, stem_y, body_half=(0.045, 0.04, 0.05), bonnet_r=0.016, flange_r=0.042, axis="x"):
    """Static valve body at the pipe point `pos` and its bonnet out to y = stem_y (the stem axis is y)."""
    w = b.w
    x, y, z = pos
    names = []
    w.add_geom(name=tag + "/body", type=GEOM.mjGEOM_BOX, size=list(body_half), pos=[x, y, z], material=b.mats["valve"])
    names.append(tag + "/body")
    if axis == "x":
        for s in (-1, 1):
            w.add_geom(name="%s/flange%d" % (tag, (s + 1) // 2), type=GEOM.mjGEOM_CYLINDER,
                       fromto=[x + s * body_half[0], y, z, x + s * (body_half[0] + 0.012), y, z], size=[flange_r, 0, 0],
                       material=b.mats["valve"])
            names.append("%s/flange%d" % (tag, (s + 1) // 2))
    else:
        for s in (-1, 1):
            w.add_geom(name="%s/flange%d" % (tag, (s + 1) // 2), type=GEOM.mjGEOM_CYLINDER,
                       fromto=[x, y, z + s * body_half[2], x, y, z + s * (body_half[2] + 0.012)], size=[flange_r, 0, 0],
                       material=b.mats["valve"])
            names.append("%s/flange%d" % (tag, (s + 1) // 2))
    w.add_geom(name=tag + "/bonnet", type=GEOM.mjGEOM_CYLINDER, fromto=[x, y - body_half[1], z, x, stem_y + 0.022, z],
               size=[bonnet_r, 0, 0], material=b.mats["valve"])
    names.append(tag + "/bonnet")
    return names


def add_handwheel_valve(b, tag, pos, wheel_d=0.18, spokes=4, turns=2.5, stem_len=0.15, friction=0.5, damping=0.4,
                        wheel_rgba=(0.70, 0.08, 0.06, 1), pipe_axis="x", tag_side=1, rim_r=0.009, wheel_phase=0.0, label=None):
    """A multi-turn gate / globe valve with a handwheel in the plane y = pos.y - stem_len and a rising stem (it moves
    out through the hub as the valve opens: 0 mm closed, 34 mm fully open). Joint `<tag>/stem`: 0 (closed) ..
    turns * 2 pi (fully open), counter-clockwise as the operator sees it."""
    x, y, z = [float(v) for v in pos]
    yw = y - stem_len
    R = wheel_d / 2.0
    static = _valve_body(b, tag, pos, yw, axis=pipe_axis)
    wheel = b.w.add_body(name=tag + "/wheel", pos=[x, yw, z])
    rng = turns * 2.0 * math.pi
    wheel.add_joint(name=tag + "/stem", type=mujoco.mjtJoint.mjJNT_HINGE, axis=Y_AXIS_OUT, range=[0.0, rng],
                    frictionloss=float(friction), damping=float(damping), armature=0.01)
    mat = b.material(wheel_rgba)
    geoms = []
    nseg = 20
    for i in range(nseg):
        a0 = 2 * math.pi * i / nseg
        a1 = 2 * math.pi * (i + 1) / nseg
        g = "%s/rim%02d" % (tag, i)
        wheel.add_geom(name=g, type=GEOM.mjGEOM_CAPSULE, fromto=[R * math.cos(a0), 0, R * math.sin(a0), R * math.cos(a1), 0,
                                                                  R * math.sin(a1)], size=[rim_r, 0, 0], material=mat, mass=0.03)
        geoms.append(g)
    r_hub = 0.022
    for k in range(spokes):
        a = wheel_phase + 2 * math.pi * k / spokes
        g = "%s/spoke%d" % (tag, k)
        wheel.add_geom(name=g, type=GEOM.mjGEOM_CAPSULE, fromto=[r_hub * math.cos(a), 0, r_hub * math.sin(a),
                                                                  (R - rim_r) * math.cos(a), 0, (R - rim_r) * math.sin(a)],
                       size=[0.0065, 0, 0], material=mat, mass=0.03)
        geoms.append(g)
    wheel.add_geom(name=tag + "/hub", type=GEOM.mjGEOM_CYLINDER, fromto=[0, -0.016, 0, 0, 0.016, 0], size=[r_hub, 0, 0],
                   material=mat, mass=0.12)
    geoms.append(tag + "/hub")
    # the rising stem: a visual rod on a slide joint, set from the wheel angle (physics.Scene.update_visuals)
    stem = b.w.add_body(name=tag + "/stemrod", pos=[x, yw, z])
    stem.add_joint(name=tag + "/rise", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=Y_AXIS_OUT, range=[0.0, 0.05], damping=5.0)
    stem.add_geom(name=tag + "/rod", type=GEOM.mjGEOM_CYLINDER, fromto=[0, 0.012, 0, 0, -0.015, 0], size=[0.0075, 0, 0],
                  material=b.mats["steel"], contype=0, conaffinity=0, mass=0.01, group=1)
    handles = []
    for k in range(spokes):
        a = wheel_phase + 2 * math.pi * k / spokes
        rm = 0.5 * (r_hub + R - rim_r)
        handles.append({"name": "%s/spoke%d" % (tag, k), "pos": [rm * math.cos(a), 0.0, rm * math.sin(a)],
                        "pinch": [-math.sin(a), 0.0, math.cos(a)], "approach": [0.0, 1.0, 0.0], "width": 0.013})
    for j in range(spokes):
        a = wheel_phase + 2 * math.pi * (j + 0.5) / spokes
        handles.append({"name": "%s/rim%d" % (tag, j), "pos": [R * math.cos(a), 0.0, R * math.sin(a)],
                        "pinch": [math.cos(a), 0.0, math.sin(a)], "approach": [0.0, 1.0, 0.0], "width": 2 * rim_r})
    tagc = (x, yw + 0.03, z + tag_side * (R + 0.035))
    tagname = add_tag(b, b.w, tag, tagc, text=label)
    return {"tag": tag, "type": "handwheel", "body": tag + "/wheel", "joint": tag + "/stem", "range": [0.0, rng], "turns": turns,
            "rise_joint": tag + "/rise", "rise_max": 0.034, "geoms": geoms, "static_geoms": static, "handles": handles,
            "wheel_d": wheel_d, "spokes": spokes, "stem_y": yw, "centre": [x, yw, z], "friction": friction, "tag_plate": tagname,
            "tag_centre": list(tagc)}


def add_lever_valve(b, tag, pos, lever_len=0.15, closed_dir=math.pi / 2, range_rad=math.pi / 2, stem_len=0.08, friction=1.0,
                    damping=0.4, grip_rgba=(0.85, 0.12, 0.08, 1), pipe_axis="x", tag_offset=(0.0, -0.075), small=False, label=None,
                    end_labels=None):
    """A quarter-turn ball valve (or the filter's changeover cock) with a flat lever in the plane y = pos.y - stem_len.
    `closed_dir`: the lever's direction at joint angle 0 (counter-clockwise from +x as the operator sees it). Joint
    `<tag>/stem`: 0 .. range_rad, counter-clockwise."""
    x, y, z = [float(v) for v in pos]
    yl = y - stem_len
    body_half = (0.03, 0.03, 0.032) if small else (0.04, 0.035, 0.042)
    static = _valve_body(b, tag, pos, yl, body_half=body_half, bonnet_r=0.012 if small else 0.015,
                         flange_r=0.028 if small else 0.038, axis=pipe_axis)
    lev = b.w.add_body(name=tag + "/lever", pos=[x, yl, z])
    lev.add_joint(name=tag + "/stem", type=mujoco.mjtJoint.mjJNT_HINGE, axis=Y_AXIS_OUT, range=[0.0, float(range_rad)],
                  frictionloss=float(friction), damping=float(damping), armature=0.01)
    a = float(closed_dir)
    q = _q_x_to(a)
    d = np.array([math.cos(a), 0.0, math.sin(a)])
    L = float(lever_len)
    geoms = []
    lev.add_geom(name=tag + "/nut", type=GEOM.mjGEOM_CYLINDER, fromto=[0, -0.012, 0, 0, 0.012, 0], size=[0.016, 0, 0],
                 material=b.mats["steel"], mass=0.05)
    geoms.append(tag + "/nut")
    bar_w = 0.010 if small else 0.011
    lev.add_geom(name=tag + "/bar", type=GEOM.mjGEOM_BOX, size=[L / 2, 0.004, bar_w], pos=list(d * L / 2), quat=q,
                 material=b.mats["steel"], mass=0.12)
    geoms.append(tag + "/bar")
    sl = 0.45 * L
    lev.add_geom(name=tag + "/grip", type=GEOM.mjGEOM_BOX, size=[sl / 2, 0.007, bar_w + 0.002], pos=list(d * (L - sl / 2)), quat=q,
                 material=b.material(grip_rgba), mass=0.05)
    geoms.append(tag + "/grip")
    perp = [-math.sin(a), 0.0, math.cos(a)]
    handles = []
    for s in (0.35, 0.55, 0.72, 0.9):
        handles.append({"name": "%s/lever%02d" % (tag, int(s * 100)), "pos": list(d * (s * L)), "pinch": perp,
                        "approach": [0.0, 1.0, 0.0], "width": 2 * (bar_w + (0.002 if s * L > L - sl else 0.0))})
    tagc = (x + tag_offset[0], yl + 0.035, z + tag_offset[1])
    tagname = add_tag(b, b.w, tag, tagc, text=label, half_w=0.03 if small else 0.034, half_h=0.0115 if small else 0.013)
    for k, (txt, ang) in enumerate(sorted((end_labels or {}).items())):
        r = L + 0.03
        add_plate(b, b.w, "%s/end%d" % (tag, k), (x + r * math.cos(ang), yl + 0.03, z + r * math.sin(ang)), 0.011, 0.011,
                  tx.tag_plate(txt, w_px=64, h_px=64))
    return {"tag": tag, "type": "lever", "body": tag + "/lever", "joint": tag + "/stem", "range": [0.0, float(range_rad)],
            "closed_dir": a, "lever_len": L, "geoms": geoms, "static_geoms": static, "handles": handles, "centre": [x, yl, z],
            "friction": friction, "tag_plate": tagname, "tag_centre": list(tagc)}


def add_knob_valve(b, tag, pos, turns=3.0, stem_len=0.10, friction=0.15, knob_r=0.019, tag_offset=(0.0, -0.06), pipe_axis="x",
                   label=None):
    """A needle valve with a round knurled knob (an index mark shows its angle). Joint `<tag>/stem`: 0 (closed) ..
    turns * 2 pi, counter-clockwise."""
    x, y, z = [float(v) for v in pos]
    yk = y - stem_len
    static = _valve_body(b, tag, pos, yk, body_half=(0.025, 0.025, 0.025), bonnet_r=0.011, flange_r=0.022, axis=pipe_axis)
    kb = b.w.add_body(name=tag + "/knob", pos=[x, yk, z])
    rng = turns * 2.0 * math.pi
    kb.add_joint(name=tag + "/stem", type=mujoco.mjtJoint.mjJNT_HINGE, axis=Y_AXIS_OUT, range=[0.0, rng],
                 frictionloss=float(friction), damping=0.05, armature=0.002)
    kb.add_geom(name=tag + "/cap", type=GEOM.mjGEOM_CYLINDER, fromto=[0, -0.014, 0, 0, 0.014, 0], size=[knob_r, 0, 0],
                material=b.mats["dark"], mass=0.04)
    kb.add_geom(name=tag + "/mark", type=GEOM.mjGEOM_BOX, size=[0.003, 0.0008, 0.007], pos=[0.0, -0.0145, knob_r - 0.008],
                rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0, mass=0.001)
    handles = [{"name": tag + "/knob", "pos": [0.0, 0.0, 0.0], "pinch": None, "approach": [0.0, 1.0, 0.0], "width": 2 * knob_r}]
    tagc = (x + tag_offset[0], yk + 0.03, z + tag_offset[1])
    tagname = add_tag(b, b.w, tag, tagc, text=label, half_w=0.03, half_h=0.0115)
    return {"tag": tag, "type": "knob", "body": tag + "/knob", "joint": tag + "/stem", "range": [0.0, rng], "turns": turns,
            "geoms": [tag + "/cap"], "static_geoms": static, "handles": handles, "centre": [x, yk, z], "friction": friction,
            "tag_plate": tagname, "tag_centre": list(tagc)}


def add_selector(b, parent, tag, centre, positions=("HAND", "OFF", "AUTO"), step=math.pi / 4):
    """A three-position selector switch on a panel face (centre = the knob's axis point on the face): a wing knob that
    points up at OFF, HAND = +45 degrees (counter-clockwise), AUTO = -45 degrees. Joint `<tag>/sel` in [-step, step]
    with detents (physics.Scene applies the detent torque)."""
    x, y, z = [float(v) for v in centre]
    plate_img = tx.label_plate(["%s     %s     %s" % positions], w_px=360, h_px=60)
    add_plate(b, parent, tag + "/legend", (x, y - 0.0005, z + 0.042), 0.045, 0.0075, plate_img)
    kb = parent.add_body(name=tag + "/knob", pos=[x, y - 0.012, z])
    kb.add_joint(name=tag + "/sel", type=mujoco.mjtJoint.mjJNT_HINGE, axis=Y_AXIS_OUT, range=[-step, step], frictionloss=0.05,
                 damping=0.05, armature=0.002)
    kb.add_geom(name=tag + "/base", type=GEOM.mjGEOM_CYLINDER, fromto=[0, -0.004, 0, 0, 0.010, 0], size=[0.022, 0, 0],
                material=b.mats["dark"], mass=0.03)
    kb.add_geom(name=tag + "/wing", type=GEOM.mjGEOM_BOX, size=[0.0075, 0.012, 0.024], pos=[0.0, -0.016, 0.0],
                material=b.mats["dark"], mass=0.02)
    kb.add_geom(name=tag + "/pointer", type=GEOM.mjGEOM_BOX, size=[0.0025, 0.0006, 0.008], pos=[0.0, -0.0285, 0.014],
                rgba=[0.97, 0.97, 0.97, 1], contype=0, conaffinity=0, mass=0.001)
    handles = [{"name": tag + "/wing", "pos": [0.0, -0.016, 0.0], "pinch": [1.0, 0.0, 0.0], "approach": [0.0, 1.0, 0.0],
                "width": 0.015}]
    return {"tag": tag, "type": "selector", "body": tag + "/knob", "joint": tag + "/sel", "range": [-step, step],
            "positions": {positions[0]: step, positions[1]: 0.0, positions[2]: -step}, "geoms": [tag + "/base", tag + "/wing"],
            "handles": handles, "centre": [x, y - 0.012, z]}


def add_button(b, parent, tag, centre, rgba, label, mushroom=False):
    """A momentary push button (mushroom head for the emergency stop) on a panel face: the cap travels 8 mm along +y."""
    x, y, z = [float(v) for v in centre]
    r = 0.026 if mushroom else 0.0155
    parent.add_geom(name=tag + "/bezel", type=GEOM.mjGEOM_CYLINDER, fromto=[x, y + 0.002, z, x, y - 0.004, z],
                    size=[r + 0.005, 0, 0], material=b.mats["dark"])
    cb = parent.add_body(name=tag + "/cap", pos=[x, y, z])
    cb.add_joint(name=tag + "/press", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0.0, 1.0, 0.0], range=[0.0, 0.008],
                 stiffness=1100.0, springref=-0.004, damping=4.0, frictionloss=0.4, armature=0.02)
    mat = b.material(rgba)
    if mushroom:
        cb.add_geom(name=tag + "/stem", type=GEOM.mjGEOM_CYLINDER, fromto=[0, -0.004, 0, 0, -0.016, 0], size=[0.012, 0, 0],
                    material=mat, mass=0.02)
        cb.add_geom(name=tag + "/head", type=GEOM.mjGEOM_CYLINDER, fromto=[0, -0.016, 0, 0, -0.03, 0], size=[r, 0, 0],
                    material=mat, mass=0.03)
        geoms = [tag + "/stem", tag + "/head"]
        face_y = -0.03
    else:
        cb.add_geom(name=tag + "/head", type=GEOM.mjGEOM_CYLINDER, fromto=[0, -0.004, 0, 0, -0.018, 0], size=[r, 0, 0],
                    material=mat, mass=0.02)
        geoms = [tag + "/head"]
        face_y = -0.018
    lab = tx.label_plate(label, w_px=240, h_px=60)
    add_plate(b, parent, tag + "/label", (x, y - 0.0005, z - r - 0.016), 0.03, 0.0075, lab)
    return {"tag": tag, "type": "button", "body": tag + "/cap", "joint": tag + "/press", "range": [0.0, 0.008],
            "press_m": 0.004, "geoms": geoms, "centre": [x, y + face_y, z], "radius": r, "label": label}


def add_lamp(b, parent, tag, centre, on_rgba, label):
    """A pilot lamp: the lens geom's colour is switched between dim and lit (physics.Scene.update_visuals)."""
    x, y, z = [float(v) for v in centre]
    parent.add_geom(name=tag + "/bezel", type=GEOM.mjGEOM_CYLINDER, fromto=[x, y + 0.002, z, x, y - 0.005, z], size=[0.016, 0, 0],
                    material=b.mats["dark"], **_static(visual=True))
    on = list(on_rgba)
    off = [0.25 * on[0] + 0.12, 0.25 * on[1] + 0.12, 0.25 * on[2] + 0.12, 1.0]
    parent.add_geom(name=tag + "/lens", type=GEOM.mjGEOM_CYLINDER, fromto=[x, y - 0.005, z, x, y - 0.013, z], size=[0.0115, 0, 0],
                    rgba=off, contype=0, conaffinity=0, group=1)
    lab = tx.label_plate(label, w_px=240, h_px=60)
    add_plate(b, parent, tag + "/label", (x, y - 0.0005, z - 0.03), 0.03, 0.0075, lab)
    return {"tag": tag, "type": "lamp", "lens": tag + "/lens", "rgba_on": on, "rgba_off": off, "centre": [x, y - 0.013, z],
            "label": label}


# ----------------------------------------------------------------------------------------------- instruments
def add_dial_gauge(b, tag, pos, kind, dial_d=0.10, stand_off=0.06, riser_to=None, title="", label=None):
    """A dial gauge whose face (centre at pos, facing -y) shows GAUGE_KINDS[kind]; the needle is a body on a hinge set
    from the process value (physics.Scene.update_visuals). riser_to: the pipe point the gauge's impulse line goes to."""
    x, y, z = [float(v) for v in pos]
    w = b.w
    R = dial_d / 2.0
    yc = y + 0.016
    w.add_geom(name=tag + "/case", type=GEOM.mjGEOM_CYLINDER, fromto=[x, yc + 0.016, z, x, yc - 0.016, z], size=[R, 0, 0],
               material=b.mats["dark"])
    half = 0.36 * dial_d
    face = tx.dial_face(kind, title=title)
    add_plate(b, w, tag + "/face", (x, yc - 0.0172, z), half, half, face, thick=0.0006)
    nb = w.add_body(name=tag + "/needle", pos=[x, yc - 0.0195, z])
    nb.add_joint(name=tag + "/pointer", type=mujoco.mjtJoint.mjJNT_HINGE, axis=Y_AXIS_OUT, range=[-10.0, 10.0], damping=0.5)
    L = 0.31 * dial_d
    nb.add_geom(name=tag + "/blade", type=GEOM.mjGEOM_BOX, size=[L / 2, 0.0006, 0.0018], pos=[L / 2 - 0.012, 0, 0],
                rgba=[0.75, 0.05, 0.05, 1], contype=0, conaffinity=0, mass=0.002, group=1)
    nb.add_geom(name=tag + "/hubcap", type=GEOM.mjGEOM_CYLINDER, fromto=[0, 0.0008, 0, 0, -0.0012, 0], size=[0.0055, 0, 0],
                rgba=[0.1, 0.1, 0.1, 1], contype=0, conaffinity=0, mass=0.001, group=1)
    static = [tag + "/case"]
    if riser_to is not None:
        rx, ry, rz = riser_to
        add_pipe(b, w, tag + "/riser", (x, yc + 0.016, z), (x, ry, z), r=0.006, visual=True)
        add_pipe(b, w, tag + "/riser2", (x, ry, z), (rx, ry, rz), r=0.006, visual=True)
    tagc = (x, yc - 0.012, z - R - 0.025)
    tagname = add_tag(b, w, tag, tagc, text=label, half_w=0.032, half_h=0.012)
    return {"tag": tag, "type": "dial", "kind": kind, "joint": tag + "/pointer", "static_geoms": static, "centre": [x, yc - 0.0172, z],
            "dial_d": dial_d, "tag_plate": tagname, "tag_centre": list(tagc)}


def add_rotameter(b, tag, pos, kind, height=0.36, label=None):
    """A variable-area flow meter: a clear vertical tube (centre at pos), a float on a slide joint and a scale plate
    behind the tube. The float's top edge reads the flow on the scale."""
    x, y, z = [float(v) for v in pos]
    w = b.w
    h2 = height / 2.0
    w.add_geom(name=tag + "/tube", type=GEOM.mjGEOM_CYLINDER, fromto=[x, y, z - h2, x, y, z + h2], size=[0.021, 0, 0],
               rgba=[0.85, 0.93, 1.0, 0.22], contype=0, conaffinity=0, group=1)
    for s, nm in ((-1, "foot"), (1, "head")):
        w.add_geom(name="%s/%s" % (tag, nm), type=GEOM.mjGEOM_CYLINDER, fromto=[x, y, z + s * h2, x, y, z + s * (h2 + 0.035)],
                   size=[0.03, 0, 0], material=b.mats["steel"])
    lo_f, hi_f = tx.scale_span()
    img = tx.scale_strip(kind, w_px=128, h_px=768)
    add_plate(b, w, tag + "/scale", (x + 0.052, y + 0.005, z), 0.028, h2, img)
    fb = w.add_body(name=tag + "/float", pos=[x, y, z - h2])
    fb.add_joint(name=tag + "/lift", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0.0, 0.0, 1.0], range=[0.0, height], damping=1.0)
    fb.add_geom(name=tag + "/bob", type=GEOM.mjGEOM_CYLINDER, fromto=[0, 0, -0.016, 0, 0, 0.0], size=[0.017, 0, 0],
                rgba=[0.9, 0.12, 0.06, 1], contype=0, conaffinity=0, mass=0.005, group=1)
    fb.add_geom(name=tag + "/cone", type=GEOM.mjGEOM_CAPSULE, fromto=[0, 0, -0.03, 0, 0, -0.016], size=[0.006, 0, 0],
                rgba=[0.9, 0.12, 0.06, 1], contype=0, conaffinity=0, mass=0.001, group=1)
    tagc = (x + 0.052, y - 0.002, z - h2 - 0.065)
    tagname = add_tag(b, w, tag, tagc, text=label, half_w=0.032, half_h=0.012)
    return {"tag": tag, "type": "rotameter", "kind": kind, "joint": tag + "/lift", "z_lo": z - h2 + lo_f * height,
            "z_hi": z - h2 + hi_f * height, "z_bottom": z - h2, "centre": [x, y, z], "static_geoms": [tag + "/foot", tag + "/head"],
            "tag_plate": tagname, "tag_centre": list(tagc)}


def add_level_gauge(b, tag, pos, height=1.1, label=None):
    """A magnetic level indicator: a chamber pipe with a column of orange flaps showing the liquid height (the flap
    column's geom is resized from the process level) and a 0-100 % scale plate beside it. pos = the bottom of the
    scale (x, y of the indicator face)."""
    x, y, z0 = [float(v) for v in pos]
    w = b.w
    w.add_geom(name=tag + "/chamber", type=GEOM.mjGEOM_CYLINDER, fromto=[x, y + 0.035, z0 - 0.05, x, y + 0.035, z0 + height + 0.05],
               size=[0.024, 0, 0], material=b.mats["steel"])
    w.add_geom(name=tag + "/rail", type=GEOM.mjGEOM_BOX, size=[0.012, 0.004, height / 2 + 0.02], pos=[x, y + 0.007, z0 + height / 2],
               rgba=[0.95, 0.95, 0.93, 1], **_static(visual=True))
    w.add_geom(name=tag + "/flaps", type=GEOM.mjGEOM_BOX, size=[0.010, 0.001, 0.01], pos=[x, y + 0.002, z0 + 0.01],
               rgba=[0.95, 0.42, 0.05, 1], contype=0, conaffinity=0, group=1)
    img = tx.scale_strip("l_0_100_pct", w_px=128, h_px=1024)
    lo_f, hi_f = tx.scale_span()
    # the plate is sized so that 0 % / 100 % of its printed scale fall at z0 / z0 + height
    plate_h = height / (hi_f - lo_f)
    plate_bottom = z0 - lo_f * plate_h
    add_plate(b, w, tag + "/scale", (x + 0.043, y + 0.006, plate_bottom + plate_h / 2), 0.026, plate_h / 2, img)
    tagc = (x + 0.04, y - 0.002, z0 - 0.085)
    tagname = add_tag(b, w, tag, tagc, text=label, half_w=0.032, half_h=0.012)
    return {"tag": tag, "type": "level", "flaps": tag + "/flaps", "z0": z0, "height": height, "x": x, "y": y + 0.002,
            "static_geoms": [tag + "/chamber"], "tag_plate": tagname, "tag_centre": list(tagc)}


def add_sight_glass(b, tag, pos, horizontal=False):
    """A small sight glass on a vent line: clear with air in it, blue once liquid stands in it."""
    x, y, z = [float(v) for v in pos]
    w = b.w
    if horizontal:
        p0, p1 = [x - 0.025, y, z], [x + 0.025, y, z]
    else:
        p0, p1 = [x, y, z - 0.025], [x, y, z + 0.025]
    w.add_geom(name=tag + "/glass", type=GEOM.mjGEOM_CYLINDER, fromto=p0 + p1, size=[0.011, 0, 0], rgba=[0.9, 0.93, 0.97, 0.45],
               contype=0, conaffinity=0, group=1)
    for i, p in enumerate((p0, p1)):
        w.add_geom(name="%s/cap%d" % (tag, i), type=GEOM.mjGEOM_SPHERE, size=[0.013, 0, 0], pos=p, material=b.mats["steel"],
                   **_static(visual=True))
    return {"tag": tag, "type": "sight", "glass": tag + "/glass", "centre": [x, y, z], "rgba_air": [0.9, 0.93, 0.97, 0.45],
            "rgba_liquid": [0.12, 0.33, 0.82, 0.92]}


def add_display(b, tag, pos, unit="mg/L", n_digits=3, dp_after=1, title=None, digit_h=0.016):
    """A panel analyser with a red 7-segment readout `d.dd` (n_digits digits, the decimal point after `dp_after`);
    digit_h: the digits' height (16 mm: about 4 px in the overview frame, legible there in an enlarged crop)."""
    x, y, z = [float(v) for v in pos]
    w = b.w
    chh = digit_h / 2.0                     # digit half-height of the segment frame
    cw = 0.94 * chh                         # digit half-width
    pitch = 2.0 * cw + 0.0045
    total = pitch * n_digits
    w.add_geom(name=tag + "/housing", type=GEOM.mjGEOM_BOX, size=[total / 2 + 0.03, 0.03, 0.055], pos=[x, y + 0.03, z],
               material=b.mats["panel"])
    w.add_geom(name=tag + "/window", type=GEOM.mjGEOM_BOX, size=[total / 2 + 0.008, 0.001, chh + 0.012], pos=[x, y - 0.0005, z + 0.01],
               rgba=[0.04, 0.04, 0.04, 1], contype=0, conaffinity=0, group=1)
    segs = []
    for k in range(n_digits):
        cx = x - total / 2 + pitch * (k + 0.5)
        dsegs = {}
        for sname, (sx, sz, horiz) in SEGMENTS.items():
            gname = "%s/d%d%s" % (tag, k, sname)
            sw = max(0.0011, 0.13 * chh)
            if horiz:
                size = [cw * 0.78, 0.0008, sw]
            else:
                size = [sw, 0.0008, chh * 0.42]
            w.add_geom(name=gname, type=GEOM.mjGEOM_BOX, size=size, pos=[cx + sx * 2 * cw * 0.95, y - 0.0025, z + 0.01 + sz * chh],
                       rgba=[0.28, 0.04, 0.03, 1], contype=0, conaffinity=0, group=1)
            dsegs[sname] = gname
        segs.append(dsegs)
    dpx = x - total / 2 + pitch * dp_after - 0.0005
    w.add_geom(name=tag + "/dp", type=GEOM.mjGEOM_BOX, size=[0.14 * chh, 0.0008, 0.14 * chh], pos=[dpx, y - 0.0025, z + 0.01 - chh],
               rgba=[0.28, 0.04, 0.03, 1], contype=0, conaffinity=0, group=1)
    add_plate(b, w, tag + "/unit", (x + 0.03, y - 0.0005, z - chh - 0.018), 0.022, 0.0065,
              tx.label_plate(unit, w_px=220, h_px=60, bg=(220, 220, 214)))
    tagc = (x - 0.03, y - 0.0005, z - chh - 0.018)
    tagname = add_plate(b, w, tag + "/tag", tagc, 0.026, 0.0065, tx.label_plate(title or tag, w_px=220, h_px=60, bg=(220, 220, 214)))
    return {"tag": tag, "type": "display", "segments": segs, "dp": tag + "/dp", "n_digits": n_digits, "dp_after": dp_after,
            "rgba_on": [1.0, 0.16, 0.08, 1], "rgba_off": [0.28, 0.04, 0.03, 1], "centre": [x, y - 0.0025, z + 0.01],
            "static_geoms": [tag + "/housing"], "tag_plate": tagname, "tag_centre": list(tagc), "unit": unit}


# ----------------------------------------------------------------------------------------------- vessels and machines
def add_tank(b, tag, centre_xy, radius, z0, height, label=None):
    """A vertical tank standing on the floor (static) with its tag painted on a plate at its front."""
    x, y = [float(v) for v in centre_xy]
    w = b.w
    w.add_geom(name=tag + "/shell", type=GEOM.mjGEOM_CYLINDER, fromto=[x, y, z0, x, y, z0 + height], size=[radius, 0, 0],
               material=b.mats["tank"])
    w.add_geom(name=tag + "/roof", type=GEOM.mjGEOM_ELLIPSOID, size=[radius, radius, 0.06], pos=[x, y, z0 + height],
               material=b.mats["tank"], **_static(visual=True))
    tagc = (x, y - radius - 0.002, z0 + height - 0.18)
    add_plate(b, w, tag + "/name", tagc, 0.09, 0.035, tx.tag_plate(label or tag, w_px=320, h_px=124), thick=0.002)
    return {"tag": tag, "type": "tank", "static_geoms": [tag + "/shell"], "centre": [x, y, z0 + height / 2], "radius": radius,
            "z0": z0, "height": height}


def add_pump(b, tag, casing_xyz, direction=1, label=None):
    """An end-suction centrifugal pump: a volute casing facing -y, the motor along +x (or -x with direction -1), a base
    plate; static."""
    x, y, z = [float(v) for v in casing_xyz]
    w = b.w
    sgn = 1 if direction >= 0 else -1
    w.add_geom(name=tag + "/base", type=GEOM.mjGEOM_BOX, size=[0.30, 0.16, 0.03], pos=[x + sgn * 0.20, y + 0.04, 0.03], material=b.mats["frame"])
    w.add_geom(name=tag + "/pedestal", type=GEOM.mjGEOM_BOX, size=[0.06, 0.08, (z - 0.06) / 2], pos=[x, y + 0.05, 0.06 + (z - 0.06) / 2],
               material=b.mats["pump"])
    w.add_geom(name=tag + "/casing", type=GEOM.mjGEOM_CYLINDER, fromto=[x, y + 0.06, z, x, y - 0.07, z], size=[0.11, 0, 0], material=b.mats["pump"])
    w.add_geom(name=tag + "/motor", type=GEOM.mjGEOM_CYLINDER, fromto=[x + sgn * 0.13, y + 0.04, z, x + sgn * 0.45, y + 0.04, z],
               size=[0.095, 0, 0], material=b.mats["motor"])
    w.add_geom(name=tag + "/guard", type=GEOM.mjGEOM_BOX, size=[0.04, 0.06, 0.06], pos=[x + sgn * 0.09, y + 0.04, z], material=b.mats["frame"])
    tagc = (x + sgn * 0.29, y + 0.04 - 0.097, z + 0.0)
    add_plate(b, w, tag + "/name", tagc, 0.04, 0.015, tx.tag_plate(label or tag, w_px=256, h_px=96), thick=0.0015)
    return {"tag": tag, "type": "pump", "static_geoms": [tag + "/base", tag + "/pedestal", tag + "/casing", tag + "/motor", tag + "/guard"],
            "casing": [x, y, z]}


def add_dosing_pump(b, tag, pos, direction=1, label=None):
    """A small diaphragm metering pump (static); its head faces upstream (-x, or +x with direction -1)."""
    x, y, z = [float(v) for v in pos]
    w = b.w
    sgn = 1 if direction >= 0 else -1
    w.add_geom(name=tag + "/body", type=GEOM.mjGEOM_BOX, size=[0.07, 0.06, 0.06], pos=[x, y + 0.04, z], material=b.mats["pump"])
    w.add_geom(name=tag + "/head", type=GEOM.mjGEOM_CYLINDER, fromto=[x - sgn * 0.07, y + 0.04, z, x - sgn * 0.11, y + 0.04, z],
               size=[0.045, 0, 0], material=b.mats["steel"])
    w.add_geom(name=tag + "/stand", type=GEOM.mjGEOM_BOX, size=[0.07, 0.06, (z - 0.06) / 2], pos=[x, y + 0.04, (z - 0.06) / 2],
               material=b.mats["frame"])
    tagc = (x, y - 0.022, z + 0.0)
    add_plate(b, w, tag + "/name", tagc, 0.034, 0.013, tx.tag_plate(label or tag, w_px=272, h_px=104), thick=0.0015)
    return {"tag": tag, "type": "pump", "static_geoms": [tag + "/body", tag + "/head", tag + "/stand"], "casing": [x, y, z]}


def add_drum(b, tag, centre_xy, radius, z0, height, label=None, letter=None):
    """A translucent chemical drum (static) whose liquid shows through its wall: the liquid column's geom is resized
    from the drum's contents (physics.Scene.update_visuals). A name plate (the site tag) and a drum letter at its front."""
    x, y = [float(v) for v in centre_xy]
    w = b.w
    w.add_geom(name=tag + "/shell", type=GEOM.mjGEOM_CYLINDER, fromto=[x, y, z0, x, y, z0 + height], size=[radius, 0, 0],
               rgba=[0.93, 0.93, 0.90, 0.38], group=0)
    w.add_geom(name=tag + "/liquid", type=GEOM.mjGEOM_CYLINDER, size=[radius - 0.012, 0.01, 0], pos=[x, y, z0 + 0.01],
               rgba=[0.78, 0.80, 0.20, 1.0], contype=0, conaffinity=0, group=1)
    w.add_geom(name=tag + "/cap", type=GEOM.mjGEOM_CYLINDER, fromto=[x, y, z0 + height, x, y, z0 + height + 0.02],
               size=[0.03, 0, 0], material=b.mats["dark"], **_static(visual=True))
    tagc = (x, y - radius - 0.004, z0 + height - 0.06)
    add_plate(b, w, tag + "/name", tagc, 0.04, 0.015, tx.tag_plate(label or tag, w_px=256, h_px=96), thick=0.0015)
    if letter:
        add_plate(b, w, tag + "/letter", (x, y - radius - 0.004, z0 + height - 0.13), 0.022, 0.022,
                  tx.tag_plate(letter, w_px=96, h_px=96), thick=0.0015)
    return {"tag": tag, "type": "drum", "static_geoms": [tag + "/shell"], "liquid": tag + "/liquid", "centre": [x, y, z0 + height / 2],
            "radius": radius, "z0": z0, "height": height}
