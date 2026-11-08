"""The dual-Panda work cell: bench, two Franka Panda arms (left: Menagerie parallel-jaw hand,
right: a lacing-hook tool), grasp/attach welds, lights and cameras, built with mujoco.MjSpec.

Conventions (shared by every block of the task): metres / radians / seconds / kg, wxyz quaternions,
bench top plane z = 0, x along the bench length, y across it, z up.  Every builder takes
``(spec, parent_body, name, **params)``, names everything it adds with the prefix ``'<name>/'``
(the Panda sub-models with ``'left/'`` / ``'right/'``) and returns a plain dict descriptor.

Robot model: MuJoCo Menagerie ``franka_emika_panda`` (Apache-2.0), copied into
``<this dir>/menagerie_panda/`` (panda.xml, panda_nohand.xml, assets/).  Sub-models are loaded with
``MjSpec.from_file`` and attached with ``spec.attach(child, prefix=..., frame=...)``; the compile
resolves the meshes through the child's own ``modelfiledir`` + ``meshdir``.  :func:`save_cell_xml`
writes a stand-alone XML whose ``meshdir`` points at ``menagerie_panda/assets`` relative to the XML.

Runtime dependencies: mujoco + numpy only.
"""

import math
import os
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .control import mat_to_quat, quat_to_mat, rotz

__all__ = [
    "PANDA_DIR", "BENCH", "ARM_BASES", "HOME_Q", "HOME_TCP", "PHYSICS",
    "build_bench", "build_pedestal", "build_hook_tool", "add_panda", "build_welds", "add_lights",
    "build_cell", "compile_cell", "reset_home", "save_cell_xml",
]

HERE = os.path.dirname(os.path.abspath(__file__))
PANDA_DIR = os.environ.get("CELL_PANDA_DIR", os.path.join(HERE, "menagerie_panda"))

# ----------------------------------------------------------------------------------- layout
BENCH = {
    "half": (0.80, 0.50, 0.02),      # top surface z = 0, 0.04 thick
    "top_z": 0.0,
    "floor_z": -0.75,
    "rgba": (0.62, 0.45, 0.27, 1.0),  # wood-like
}
ARM_BASES = {
    "left": {"pos": (-0.42, 0.55, 0.0), "yaw": -math.pi / 2},   # base x axis -> world -y (faces the bench)
    "right": {"pos": (0.42, 0.55, 0.0), "yaw": -math.pi / 2},
}
# home posture (elbow up, tool pointing down ~0.35 m above the bench, ~0.42 m in front of the base);
# found with ik_solve from the Menagerie 'home' key and rounded (see REPORT.md)
# the right tool frame (Menagerie 'attachment', +135 deg about link7 z) is rotated 180 deg about z relative
# to the hand frame (-45 deg), so the same arm pose needs joint7 + pi for the same tool orientation
HOME_Q = {
    "left": np.array([0.0, -0.40, 0.0, -2.43, 0.0, 2.03, -0.785]),
    "right": np.array([0.0, -0.40, 0.0, -2.43, 0.0, 2.03, 2.356]),
}
HOME_TCP = {"height": 0.35, "reach": 0.42}
TOOL_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # tool +z = world -z, tool +x = world +x

PHYSICS = {
    "timestep": 0.001,
    "integrator": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    "cone": mujoco.mjtCone.mjCONE_ELLIPTIC,
    "jacobian": mujoco.mjtJacobian.mjJAC_DENSE,
    "solver": mujoco.mjtSolver.mjSOL_NEWTON,
    "iterations": 50,
    "ls_iterations": 50,
    "noslip_iterations": 0,
    "impratio": 10.0,
}
GRIPPER = {"kp": 2500.0, "kv": 50.0, "max_force": 60.0}   # N/m on the tendon (half width), Ns/m, N
PAD_FRICTION = (1.5, 0.005, 0.0001)
WELD_SOLREF = (0.002, 1.0)
HOOK = {"shaft_len": 0.12, "shaft_r": 0.0025, "foot_len": 0.014, "tip_len": 0.012, "foot_r": 0.002,
        "flange_r": 0.03, "flange_h": 0.016, "mass": 0.12}


def yaw_quat(yaw):
    """wxyz quaternion of a rotation about world z."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# ----------------------------------------------------------------------------------- builders
def build_bench(spec, parent, name="bench", half=BENCH["half"], top_z=BENCH["top_z"], floor_z=BENCH["floor_z"],
                rgba=BENCH["rgba"], with_floor=True):
    """Static bench top (box, top surface at top_z) plus an optional floor plane.  Returns a descriptor."""
    hx, hy, hz = half
    spec.add_material(name=name + "/wood", rgba=list(rgba), specular=0.2, shininess=0.3)
    parent.add_geom(name=name + "/top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
                    pos=[0.0, 0.0, top_z - hz], material=name + "/wood", condim=3, friction=[1.0, 0.005, 0.0001])
    # four legs (visual only, group 2) so the front camera sees a bench, not a floating slab
    legs = []
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        gname = "%s/leg%d" % (name, i)
        parent.add_geom(name=gname, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, (top_z - 2 * hz - floor_z) / 2],
                        pos=[sx * (hx - 0.05), sy * (hy - 0.05), (top_z - 2 * hz + floor_z) / 2],
                        rgba=[0.3, 0.3, 0.32, 1], contype=0, conaffinity=0, group=2)
        legs.append(gname)
    if with_floor:
        spec.add_texture(name=name + "/floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.35, 0.37, 0.40], rgb2=[0.28, 0.30, 0.33], width=256, height=256, mark=mujoco.mjtMark.mjMARK_EDGE,
                         markrgb=[0.5, 0.5, 0.5])
        mat = spec.add_material(name=name + "/floor", texrepeat=[8, 8], texuniform=True, reflectance=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name + "/floor_tex"
        parent.add_geom(name=name + "/floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3.0, 3.0, 0.1], pos=[0, 0, floor_z],
                        material=name + "/floor")
    return {"name": name, "top_geom": name + "/top", "legs": legs, "half": tuple(half), "top_z": top_z,
            "x_range": (-hx, hx), "y_range": (-hy, hy), "floor_z": floor_z}


def build_pedestal(spec, parent, name, pos_xy, half_xy=0.09, top_z=0.0, floor_z=BENCH["floor_z"]):
    """Static column under an arm base, top flush with the bench top.  Returns a descriptor."""
    h = (top_z - floor_z) / 2
    parent.add_geom(name=name + "/column", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half_xy, half_xy, h],
                    pos=[pos_xy[0], pos_xy[1], top_z - h], rgba=[0.45, 0.45, 0.48, 1])
    return {"name": name, "geom": name + "/column", "pos": (pos_xy[0], pos_xy[1], top_z)}


def build_hook_tool(spec, parent, name, site_prefix, shaft_len=HOOK["shaft_len"], shaft_r=HOOK["shaft_r"],
                    foot_len=HOOK["foot_len"], tip_len=HOOK["tip_len"], foot_r=HOOK["foot_r"],
                    flange_r=HOOK["flange_r"], flange_h=HOOK["flange_h"], mass=HOOK["mass"]):
    """Lacing hook tool body under ``parent`` (the flange): shaft along tool +z, L-foot along +x, tip back along -z.

    Sites: ``<site_prefix>tcp`` at the shaft end (tool frame) and ``<site_prefix>hook_tip`` at the end of the
    tip segment with +z along the tip direction (= tool -z).  Collision geoms are the capsules themselves
    (condim 3, contype 1 / conaffinity 0: see grasp.attach for why).  Returns a descriptor.
    """
    body = parent.add_body(name=name, pos=[0, 0, 0])
    kw = dict(condim=3, contype=1, conaffinity=0, friction=[0.8, 0.005, 0.0001])
    m_flange = mass * 0.6
    m_rest = (mass - m_flange) / 3.0
    body.add_geom(name=name + "/flange", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[flange_r, flange_h / 2, 0],
                  pos=[0, 0, flange_h / 2], rgba=[0.25, 0.25, 0.27, 1], mass=m_flange, **kw)
    body.add_geom(name=name + "/shaft", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, flange_h, 0, 0, shaft_len],
                  size=[shaft_r, 0, 0], rgba=[0.75, 0.75, 0.78, 1], mass=m_rest, **kw)
    body.add_geom(name=name + "/foot", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, shaft_len, foot_len, 0, shaft_len],
                  size=[foot_r, 0, 0], rgba=[0.75, 0.75, 0.78, 1], mass=m_rest, **kw)
    body.add_geom(name=name + "/tip", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=[foot_len, 0, shaft_len, foot_len, 0, shaft_len - tip_len],
                  size=[foot_r, 0, 0], rgba=[0.9, 0.55, 0.15, 1], mass=m_rest, **kw)
    body.add_site(name=site_prefix + "tcp", pos=[0, 0, shaft_len], size=[0.003, 0.003, 0.003], rgba=[0.2, 0.9, 0.2, 0.5], group=4)
    # hook_tip: +z along the tip direction (tool -z): rotate 180 deg about x
    body.add_site(name=site_prefix + "hook_tip", pos=[foot_len, 0, shaft_len - tip_len], quat=[0, 1, 0, 0],
                  size=[0.002, 0.002, 0.002], rgba=[0.9, 0.2, 0.2, 0.5], group=4)
    return {"name": name, "body": name, "geoms": [name + "/flange", name + "/shaft", name + "/foot", name + "/tip"],
            "tcp_site": site_prefix + "tcp", "tip_site": site_prefix + "hook_tip",
            "tcp_local": (0.0, 0.0, shaft_len), "tip_local": (foot_len, 0.0, shaft_len - tip_len),
            "tip_dir_local": (0.0, 0.0, -1.0), "shaft_len": shaft_len, "foot_len": foot_len, "tip_len": tip_len,
            "foot_r": foot_r, "shaft_r": shaft_r}


def add_panda(spec, parent, name, pos, yaw, hand=True, panda_dir=PANDA_DIR, gravcomp=1.0):
    """Attach a Menagerie Panda (with the hand when hand=True, else arm + hook tool) at (pos, yaw).

    name is the prefix without the slash ('left' / 'right').  Returns a descriptor with the body,
    joint, actuator and site names.
    """
    prefix = name + "/"
    child = mujoco.MjSpec.from_file(os.path.join(panda_dir, "panda.xml" if hand else "panda_nohand.xml"))
    for k in list(child.keys):
        child.delete(k)
    for l in list(child.lights):
        child.delete(l)
    frame = parent.add_frame(name=prefix + "base_frame", pos=list(pos), quat=yaw_quat(yaw).tolist())
    spec.attach(child, prefix=prefix, frame=frame)
    for b in spec.bodies:
        if b.name.startswith(prefix) and b.name != prefix + "link0":
            b.gravcomp = float(gravcomp)
    desc = {
        "name": name, "prefix": prefix, "base_pos": tuple(pos), "base_yaw": yaw,
        "base_body": prefix + "link0", "flange_body": prefix + ("hand" if hand else "attachment"),
        "joints": [prefix + "joint%d" % (i + 1) for i in range(7)],
        "actuators": [prefix + "actuator%d" % (i + 1) for i in range(7)],
        "hand": hand,
    }
    if hand:
        hand_body = spec.body(prefix + "hand")
        # tcp between the finger pads: 0.1034 m from the hand origin along +z (approach), +y = closing axis
        hand_body.add_site(name=prefix + "tcp", pos=[0.0, 0.0, 0.1034], size=[0.003, 0.003, 0.003], rgba=[0.2, 0.9, 0.2, 0.5], group=4)
        pads = []
        for g in spec.geoms:
            b = g.parent
            if b is not None and b.name in (prefix + "left_finger", prefix + "right_finger") and g.type == mujoco.mjtGeom.mjGEOM_BOX:
                side = "left" if "left" in b.name else "right"
                g.name = "%spad_%s_%d" % (prefix, side, len([p for p in pads if side in p]))
                g.friction = list(PAD_FRICTION)
                g.condim = 4
                pads.append(g.name)
        for a in spec.actuators:
            if a.name == prefix + "actuator8":
                # keep the Menagerie ctrl range (0..255 <-> half width 0..0.04) but stiffen: force =
                # kp (target - half width) - kv v, capped
                scale = 0.04 / 255.0
                a.gainprm = [GRIPPER["kp"] * scale] + [0.0] * 9
                a.biasprm = [0.0, -GRIPPER["kp"], -GRIPPER["kv"]] + [0.0] * 7
                a.forcerange = [-GRIPPER["max_force"], GRIPPER["max_force"]]
        desc.update({
            "tool_body": prefix + "hand", "tcp_site": prefix + "tcp", "tcp_local": (0.0, 0.0, 0.1034),
            "gripper_actuator": prefix + "actuator8", "finger_joints": [prefix + "finger_joint1", prefix + "finger_joint2"],
            "finger_bodies": [prefix + "left_finger", prefix + "right_finger"], "pad_geoms": pads,
            "max_width": 0.08,
        })
    else:
        flange = spec.body(prefix + "attachment")
        tool = build_hook_tool(spec, flange, prefix + "hook", site_prefix=prefix)
        tool_body = spec.body(prefix + "hook")
        tool_body.gravcomp = float(gravcomp)
        desc.update({"tool_body": prefix + "hook", "tcp_site": prefix + "tcp", "tip_site": prefix + "hook_tip", "hook": tool,
                     "tcp_local": tool["tcp_local"]})
    return desc


def build_welds(spec, left_tool_body, right_tool_body, solref=WELD_SOLREF):
    """Two inactive weld equalities ('left/grasp', 'right/hook'), body2 = world as a placeholder."""
    data = [0.0] * 11
    data[10] = 1.0   # pivot 0, relpose 0 (filled at attach), torquescale 1
    names = {}
    for eq_name, body in (("left/grasp", left_tool_body), ("right/hook", right_tool_body)):
        spec.add_equality(name=eq_name, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=body, name2="world", active=False, solref=list(solref), data=data)
        names[eq_name] = body
    return {"left": "left/grasp", "right": "right/hook", "body1": names}


def add_lights(spec, parent):
    """Three lights, no shadows (speed)."""
    parent.add_light(name="light/top", pos=[0.0, 0.0, 2.0], dir=[0, 0, -1], diffuse=[0.55, 0.55, 0.55], specular=[0.1, 0.1, 0.1],
                     castshadow=False)
    parent.add_light(name="light/front_left", pos=[-1.0, -1.2, 1.5], dir=[0.5, 0.6, -0.7], diffuse=[0.45, 0.45, 0.45],
                     specular=[0.05, 0.05, 0.05], castshadow=False)
    parent.add_light(name="light/front_right", pos=[1.0, -1.2, 1.5], dir=[-0.5, 0.6, -0.7], diffuse=[0.45, 0.45, 0.45],
                     specular=[0.05, 0.05, 0.05], castshadow=False)
    return ["light/top", "light/front_left", "light/front_right"]


# ----------------------------------------------------------------------------------- the cell
def apply_physics(spec, **overrides):
    """Write the PHYSICS options (plus overrides) into spec.option."""
    p = dict(PHYSICS)
    p.update(overrides)
    o = spec.option
    o.timestep = p["timestep"]
    o.integrator = p["integrator"]
    o.cone = p["cone"]
    o.jacobian = p["jacobian"]
    o.solver = p["solver"]
    o.iterations = p["iterations"]
    o.ls_iterations = p["ls_iterations"]
    o.noslip_iterations = p["noslip_iterations"]
    o.impratio = p["impratio"]
    o.gravity = [0.0, 0.0, -9.81]
    return p


def build_cell(panda_dir=PANDA_DIR, with_cameras=True, physics=None, offscreen=(640, 480)):
    """Build the whole cell spec.  Returns (spec, info) where info holds every descriptor by name."""
    spec = mujoco.MjSpec()
    spec.modelname = "dual_panda_cell"
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    phys = apply_physics(spec, **(physics or {}))
    spec.visual.global_.offwidth = int(offscreen[0])
    spec.visual.global_.offheight = int(offscreen[1])
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.map.znear = 0.005
    spec.visual.map.zfar = 30.0
    spec.stat.extent = 1.0
    spec.stat.center = [0.0, 0.0, 0.2]
    gd = spec.default.geom
    gd.condim = 3
    gd.friction = [1.0, 0.005, 0.0001]
    w = spec.worldbody
    info = {"physics": phys}
    info["bench"] = build_bench(spec, w, "bench")
    info["pedestals"] = {}
    info["arms"] = {}
    for side in ("left", "right"):
        base = ARM_BASES[side]
        info["pedestals"][side] = build_pedestal(spec, w, "pedestal_" + side, base["pos"][:2], top_z=base["pos"][2])
        info["arms"][side] = add_panda(spec, w, side, base["pos"], base["yaw"], hand=(side == "left"), panda_dir=panda_dir)
    info["welds"] = build_welds(spec, info["arms"]["left"]["tool_body"], info["arms"]["right"]["tool_body"])
    info["lights"] = add_lights(spec, w)
    if with_cameras:
        from cameras import build_cameras
        info["cameras"] = build_cameras(spec, info)
    # home keyframe (only the arm joints and fingers; free bodies added later start at their spec pose)
    info["home_q"] = {k: v.copy() for k, v in HOME_Q.items()}
    return spec, info


def compile_cell(spec, info):
    """Compile, create data, put the arms at home with matching ctrl and run mj_forward."""
    model = spec.compile()
    data = mujoco.MjData(model)
    reset_home(model, data, info)
    return model, data


def reset_home(model, data, info):
    """Set both arms to HOME_Q (fingers open), ctrl to match, and mj_forward."""
    mujoco.mj_resetData(model, data)
    for side, arm in info["arms"].items():
        q = info["home_q"][side]
        for i, jn in enumerate(arm["joints"]):
            data.qpos[model.jnt_qposadr[model.joint(jn).id]] = q[i]
            data.ctrl[model.actuator(arm["actuators"][i]).id] = q[i]
        if arm["hand"]:
            for jn in arm["finger_joints"]:
                data.qpos[model.jnt_qposadr[model.joint(jn).id]] = 0.04
            data.ctrl[model.actuator(arm["gripper_actuator"]).id] = model.actuator_ctrlrange[model.actuator(arm["gripper_actuator"]).id][1]
    mujoco.mj_forward(model, data)


def save_cell_xml(spec, path, panda_dir=PANDA_DIR, assets_rel="menagerie_panda/assets"):
    """Write spec.to_xml() to path with compiler meshdir = assets_rel (relative to the XML's directory).

    The Menagerie meshes are expected at <dir of path>/<assets_rel>; they are copied there when missing
    (so a Docker image only needs the XML's directory).  Returns the path.
    """
    xml = spec.to_xml()
    root = ET.fromstring(xml)
    comp = root.find("compiler")
    if comp is None:
        comp = ET.SubElement(root, "compiler")
    comp.set("meshdir", assets_rel)
    comp.set("angle", "radian")
    out_dir = os.path.dirname(os.path.abspath(path))
    dst_assets = os.path.join(out_dir, assets_rel)
    os.makedirs(dst_assets, exist_ok=True)
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if not f:
            continue
        base = os.path.basename(f)
        mesh.set("file", base)
        dst = os.path.join(dst_assets, base)
        if not os.path.isfile(dst):
            src = os.path.join(panda_dir, "assets", base)
            if os.path.isfile(src):
                shutil.copyfile(src, dst)
    ET.indent(root, space="  ")
    with open(path, "w") as fh:
        fh.write(ET.tostring(root, encoding="unicode") + "\n")
    return path
