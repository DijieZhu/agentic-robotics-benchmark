"""The UR arm of the quadruped assembly cell: the given MuJoCo UR10e (or UR5e) model with the given 2F-85 on its real tool
flange, attached into the cell's MjSpec by `add_ur(spec, cfg)`.

What is attached, and why (the numbers carry their source):
* the arm: the given `universal_robots_ur10e` model as shipped: visual meshes, capsule collision geoms,
  joint armature 0.1 kg m^2 (the harmonic drives' reflected rotor inertia), damping 10 / 10 / 5 / 2 / 2 / 2 N m s/rad,
  ranges +-2 pi (elbow +-pi). Its six position servos (`general`, kp 5000, kv 500) are DELETED: under Euler they chatter
  and their stiffness is not the robot's. The arm is driven by computed torque
  (`ur_controller.URController`) through six `motor` actuators whose ctrl is the joint torque, limited to the e-Series
  joint ratings (UR10e 330 / 330 / 150 / 56 / 56 / 56 N m = sizes 4 / 4 / 3 / 2 / 2 / 2, the given model's forceranges;
  UR5e 150 / 150 / 150 / 28 / 28 / 28 N m). No gravcomp: gravity is in the computed torque's bias term.
* the flange: UR's DH d6 along wrist_3's +y (UR10e 0.11655 m, UR5e 0.0996 m). The given model's `attachment_site` sits at
  0.100 m, 16.55 mm inside the UR10e's real flange face (its wrist3 mesh ends at 0.11678 m), so a tool on the site would
  sit inside the wrist. The flange frame keeps the site's axes (z out of the flange; the 2F-85 closes along its x).
* the gripper: the given `robotiq_2f85` model (v1) patched in code:
    - both coupler joints deleted: the fingertip link rigid on the driver at its stop = the real gripper's parallel
      (pinch) mode. The free coupler (a 0.05 N m/rad spring) folds the pads 9-12 deg onto their tips at 10 N per pad;
      locked, the pads keep the 4-bar's own attitude, the faces converging 0-1.8 deg towards the tips
      (measured, gripper_2f85). The power / encompassing grasp is not available;
    - pad boxes condim 4 (torsional friction 0.005 m resists a part turning about the closing axis), solref 0.002
      (stiffer pad contact: the held part's creep under a steady load is a soft-contact effect), priority 2 (above
      the parts' 0 / 1, so the pads' friction and solref govern every pad contact);
    - the tendon servo's force range +-9 (pad normal force ~8.9 x the tendon force: about 80 N per pad; the shipped +-5
      gives ~45 N; the real 2F-85 grips with 20-235 N).
  The 2F-85 keeps its own servo (kp 100, kv 10 on the driver angle, ctrl 0-255) and its linkage equalities.
* options: `MjSpec.attach` keeps the PARENT's options and drops the gripper's `cone="elliptic" impratio="10"` (in that
  raw state a held 150 g box slid 80 mm / 62 deg in 10 s). `add_ur` sets cone = elliptic and impratio (default 100: the
  steady creep of a held part fell from 11 to 2.4 um/s) on the parent. Timestep, integrator and noslip are the cell's
  (1 ms, implicitfast, noslip_iterations 5). The children's options are aligned with the parent's
  before each attach so MuJoCo does not warn about the dropped values.
* sites and sensors (names below with the default prefix "ur/"):
    - `ur/tcp`: the 2F-85's pinch point (the given model's `pinch` site, 0.145 m along the gripper base's z) = 155.8 mm
      from the flange face; axes: z = approach, y = closing axis (the right pad on +y), x across the pads (the Panda
      cell's convention). The pads' face centre lies 1.1-3.6 mm short of it (towards the flange) for 10-45 mm
      grasps, 14.2 mm when fully open; the fingertips reach 160.4 (open) to 173.7 mm from the flange
      (`gripper_2f85.py`). Robotiq's own documented tcp (174 mm with its coupling) is NOT this point;
    - `ur/flange_ft`: the flange face's centre, flange axes, on the 2F-85's coupling body (welded to wrist_3).
      `ur/flange_force` / `ur/flange_torque` read the wrench between the tool and the wrist in that frame (MuJoCo:
      the interaction force between the coupling body and its parent): the e-Series' built-in flange sensor. Sim
      bias-free; it includes the tool's own weight (the given 2F-85's 1.053 kg) and inertial forces;
    - `ur/wrist_cam_mount`: a body on the coupling whose frame is a MuJoCo camera frame (looks along its -z, image
      x along the closing axis, up = +y): the lens 58 mm off the tool axis on the flange's -y side (= the tcp's +x
      side, so the view runs between the fingers, not into one), 36 mm out from the flange, aimed at the tool axis
      240 mm out (70 mm beyond the closed fingertips: a 16 deg tilt, the Panda cell's wrist camera had 16.9 deg). The
      fingers and a held part fill the lower third of the frame, the scene the fingertips approach the rest.
      From a lens near the flange the 2F-85's housing (a 37.5 mm radius
      cylinder to 50 mm out) hides the pinch point for any offset under ~48 mm. A housing box (36 x 24 x 35 mm,
      visual + collision, massless by default: the real Robotiq wrist camera weighs 160 g) sits on the coupling
      under it. `add_ur` puts a camera `ur/wrist` (fovy 70) on the mount unless cfg["wrist_camera"]["camera"] is
      None; the cell may set its resolution / intrinsics.
* numeric `ur/joint_speed_limit_rad_s` (UR10e base / shoulder 120 deg/s, elbow and wrists 180 deg/s; UR5e 180 deg/s)
  and text `ur/robot` (the model name), so a controller on the compiled model knows the robot.
"""
import copy
import math
import os

import numpy as np

__all__ = ["UR_MODELS", "JOINT_NAMES", "ACTUATOR_NAMES", "PAD_GEOMS", "GRIPPER_JOINTS", "DEFAULT_CFG", "add_ur",
           "robot_of", "gripper_spec", "quat_wxyz", "rotz", "mat_from_quat"]

# the folder holding universal_robots_ur10e/ (ur5e) and robotiq_2f85/: $ARB_ROBOT_MODELS, else the task's given robot
# models ($ARB_GIVEN_DIR/robot: the grader's root-owned copy at verify, ~/given/robot in the agent's home)
ROBOT_MODELS = os.environ.get("ARB_ROBOT_MODELS") or os.path.join(os.environ.get("ARB_GIVEN_DIR") or os.path.join(os.path.expanduser("~"), "given"), "robot")
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
               "wrist_3_joint"]
ACTUATOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
PAD_GEOMS = ["right_pad1", "right_pad2", "left_pad1", "left_pad2"]
GRIPPER_JOINTS = ["right_driver_joint", "right_spring_link_joint", "right_follower_joint", "left_driver_joint",
                  "left_spring_link_joint", "left_follower_joint"]

# e-Series data (UR product sheets): DH d6 (flange), joint torque ratings (= the given model's forceranges), joint speed limits
UR_MODELS = {
    "ur10e": {"dir": "universal_robots_ur10e", "xml": "ur10e.xml", "d6_m": 0.11655,
              "torque_limits_Nm": [330.0, 330.0, 150.0, 56.0, 56.0, 56.0],
              "speed_limits_deg_s": [120.0, 120.0, 180.0, 180.0, 180.0, 180.0]},
    "ur5e": {"dir": "universal_robots_ur5e", "xml": "ur5e.xml", "d6_m": 0.0996,
             "torque_limits_Nm": [150.0, 150.0, 150.0, 28.0, 28.0, 28.0],
             "speed_limits_deg_s": [180.0, 180.0, 180.0, 180.0, 180.0, 180.0]},
}
# the 2F-85 base_mount body sits 7 mm out from the attach frame (2f85.xml), the base 3.8 mm further, the pinch 145 mm
GRIP_MOUNT_Z = 0.007
PINCH_IN_BASE = [0.0, 0.0, 0.145]

DEFAULT_CFG = {
    "model": "ur10e",
    "pos": [0.0, 0.0, 0.0],             # the base body's frame origin in the parent's world (the mounting face)
    "yaw_deg": 0.0,                     # turn of that frame about z; at 0 the given model's base (quat 0 0 0 -1) is kept,
                                        # so q1 = 0 points the arm along world -x and q1 = pi along +x
    "prefix": "ur/",
    "gripper_prefix": "g/",
    "models_dir": ROBOT_MODELS,
    "gripper_xml": None,                # a ready 2F-85 MJCF instead of the patched given file (no patches applied)
    "impratio": 100.0,
    "lock_coupler": True,
    "pad": {"condim": 4, "solref": [0.002, 1.0], "priority": 2, "friction": None},   # friction None: the given model's 0.7 / 0.6
    "grip_forcerange": 9.0,
    "tcp_name": "tcp",
    "ft_site": "flange_ft",
    "force_sensor": "flange_force",
    "torque_sensor": "flange_torque",
    "wrist_camera": {"mount": "wrist_cam_mount", "site": "wrist_cam", "camera": "wrist", "fovy_deg": 70.0,
                     "lens_in_flange_m": [0.0, -0.058, 0.036], "aim_in_flange_m": [0.0, 0.0, 0.240],
                     "housing_centre_in_flange_m": [0.0, -0.056, 0.0185], "housing_half_m": [0.018, 0.012, 0.0175],
                     "housing_mass_kg": 0.0},
}


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def robot_of(model, prefix="ur/"):
    """the UR model name ("ur10e" / "ur5e") that `add_ur` stored in the compiled model (text `<prefix>robot`)"""
    import mujoco
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, prefix + "robot")
    if i < 0:
        return None
    a, n = int(model.text_adr[i]), int(model.text_size[i])
    return bytes(model.text_data[a:a + n]).split(b"\x00")[0].decode()


def quat_wxyz(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def mat_from_quat(q):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float) / np.linalg.norm(q))
    return R.reshape(3, 3)


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _align_options(child, parent):
    """copy the parent's solver options into a child spec before attaching it: MjSpec.attach keeps the parent's values
    anyway, and with equal values it does not warn about the child's authored ones (the 2F-85's cone / impratio, the
    UR's integrator)"""
    for k in ("timestep", "integrator", "cone", "impratio", "noslip_iterations", "iterations", "ls_iterations",
              "tolerance", "solver", "jacobian"):
        setattr(child.option, k, getattr(parent.option, k))


def _absolute_meshes(spec, xml_path):
    """make every mesh / texture file of a loaded spec absolute (its compiler meshdir / texturedir joined): an attached
    spec keeps its files relative to its own folder, which a saved cell.xml cannot resolve"""
    base = os.path.dirname(os.path.abspath(xml_path))
    mdir = os.path.join(base, spec.compiler.meshdir or "")
    tdir = os.path.join(base, spec.compiler.texturedir or "")
    for mesh in spec.meshes:
        if mesh.file and not os.path.isabs(mesh.file):
            mesh.file = os.path.join(mdir, mesh.file)
    for tex in spec.textures:
        if tex.file and not os.path.isabs(tex.file):
            tex.file = os.path.join(tdir, tex.file)
    spec.compiler.meshdir = ""
    spec.compiler.texturedir = ""
    return spec


def gripper_spec(cfg):
    """The 2F-85 spec: the given file with the recommended patches (or cfg["gripper_xml"] as is)."""
    import mujoco
    if cfg.get("gripper_xml"):
        return _absolute_meshes(mujoco.MjSpec.from_file(cfg["gripper_xml"]), cfg["gripper_xml"])
    gx = os.path.join(cfg["models_dir"], "robotiq_2f85", "2f85.xml")
    g = _absolute_meshes(mujoco.MjSpec.from_file(gx), gx)
    pad = cfg["pad"]
    for geom in g.geoms:
        if geom.name in PAD_GEOMS:
            geom.condim = int(pad["condim"])
            geom.solref = list(pad["solref"])
            geom.priority = int(pad["priority"])
            if pad.get("friction") is not None:
                f = list(geom.friction)
                f[0] = float(pad["friction"])
                geom.friction = f
    if cfg.get("lock_coupler", True):
        for n in ("right_coupler_joint", "left_coupler_joint"):
            g.delete(g.joint(n))
    act = g.actuators[0]
    fr = float(cfg["grip_forcerange"])
    act.forcerange = [-fr, fr]
    return g


def _camera_frame(lens, aim):
    """rotation (columns = camera x, y, z in the flange frame) of a MuJoCo camera at `lens` looking at `aim`, its image
    x along the flange's x (the closing axis: the two fingers left and right in the image)"""
    lens, aim = np.asarray(lens, float), np.asarray(aim, float)
    cz = lens - aim
    cz /= np.linalg.norm(cz)
    cx = np.array([1.0, 0.0, 0.0]) - cz * cz[0]
    cx /= np.linalg.norm(cx)
    cy = np.cross(cz, cx)
    return np.column_stack([cx, cy, cz])


def add_ur(spec, cfg=None):
    """Attach a UR arm with the 2F-85 into `spec` (an MjSpec). Returns a dict of the element names the cell, the
    controller and the IK need (see the end of this function)."""
    import mujoco
    cfg = _merge(DEFAULT_CFG, cfg)
    rob = UR_MODELS[cfg["model"]]
    P, G = cfg["prefix"], cfg["gripper_prefix"]
    ax = os.path.join(cfg["models_dir"], rob["dir"], rob["xml"])
    arm = _absolute_meshes(mujoco.MjSpec.from_file(ax), ax)
    grip = gripper_spec(cfg)
    for k in list(arm.keys):
        arm.delete(k)
    for a in list(arm.actuators):
        arm.delete(a)
    lim = rob["torque_limits_Nm"]
    for an, jn, tl in zip(ACTUATOR_NAMES, JOINT_NAMES, lim):
        a = arm.add_actuator(name=an, target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_motor()
        a.ctrlrange = [-tl, tl]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-tl, tl]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE

    # the gripper on the real flange face: wrist_3 +y at d6, the attachment site's axes (z out of the flange)
    site = arm.site("attachment_site")
    w3 = arm.body("wrist_3_link")
    flange = w3.add_frame(pos=[0.0, float(rob["d6_m"]), 0.0], quat=list(site.quat))
    _align_options(grip, arm)
    flange.attach_body(grip.body("base_mount"), G, "")
    mount = arm.body(G + "base_mount")
    base = arm.body(G + "base")
    base.add_site(name=cfg["tcp_name"], pos=list(PINCH_IN_BASE), size=[0.004, 0.004, 0.004], group=4,
                  rgba=[0.2, 0.9, 0.2, 0.6])
    mount.add_site(name=cfg["ft_site"], pos=[0.0, 0.0, -GRIP_MOUNT_Z], size=[0.003, 0.003, 0.003], group=4)
    wc = cfg["wrist_camera"]
    off = np.array([0.0, 0.0, -GRIP_MOUNT_Z])           # flange frame -> coupling (base_mount) frame
    hc = np.asarray(wc["housing_centre_in_flange_m"], float) + off
    mount.add_geom(name="wrist_cam_housing", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(wc["housing_half_m"]),
                   pos=hc.tolist(), rgba=[0.12, 0.12, 0.13, 1.0], mass=float(wc["housing_mass_kg"]), group=2)
    Rc = _camera_frame(wc["lens_in_flange_m"], wc["aim_in_flange_m"])
    cam_body = mount.add_body(name=wc["mount"], pos=(np.asarray(wc["lens_in_flange_m"], float) + off).tolist(),
                              quat=quat_wxyz(Rc).tolist())
    cam_body.add_site(name=wc["site"], size=[0.003, 0.003, 0.003], group=4, rgba=[0.9, 0.6, 0.1, 0.6])
    if wc.get("camera"):
        cam_body.add_camera(name=wc["camera"], fovy=float(wc["fovy_deg"]))

    # the arm into the parent: a frame at the base pose, the given model's base body (its own 180 deg quat) under it
    frame = spec.worldbody.add_frame(pos=[float(v) for v in cfg["pos"]],
                                     quat=quat_wxyz(rotz(math.radians(float(cfg["yaw_deg"])))).tolist())
    _align_options(arm, spec)
    frame.attach_body(arm.body("base"), P, "")
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = float(cfg["impratio"])
    spec.add_sensor(name=P + cfg["force_sensor"], type=mujoco.mjtSensor.mjSENS_FORCE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname=P + cfg["ft_site"])
    spec.add_sensor(name=P + cfg["torque_sensor"], type=mujoco.mjtSensor.mjSENS_TORQUE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname=P + cfg["ft_site"])
    spd = [math.radians(v) for v in rob["speed_limits_deg_s"]]
    spec.add_numeric(name=P + "joint_speed_limit_rad_s", data=spd)
    spec.add_text(name=P + "robot", data=cfg["model"])

    gb = [P + G + n for n in ("base_mount", "base", "right_driver", "right_coupler", "right_spring_link",
                              "right_follower", "right_pad", "right_silicone_pad", "left_driver", "left_coupler",
                              "left_spring_link", "left_follower", "left_pad", "left_silicone_pad")]
    return {
        "model": cfg["model"], "prefix": P, "gripper_prefix": P + G,
        "joints": [P + j for j in JOINT_NAMES],
        "actuators": [P + a for a in ACTUATOR_NAMES],
        "torque_limits_Nm": list(lim),
        "speed_limits_rad_s": spd,
        "base_body": P + "base",
        "arm_bodies": [P + b for b in ("base", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link",
                                       "wrist_2_link", "wrist_3_link")],
        "flange_body": P + "wrist_3_link",
        "tool_body": P + G + "base_mount",
        "gripper_bodies": gb + [P + wc["mount"]],
        "finger_bodies": {"right": [P + G + n for n in ("right_driver", "right_coupler", "right_spring_link",
                                                        "right_follower", "right_pad", "right_silicone_pad")],
                          "left": [P + G + n for n in ("left_driver", "left_coupler", "left_spring_link",
                                                       "left_follower", "left_pad", "left_silicone_pad")]},
        "pad_bodies": {"right": P + G + "right_pad", "left": P + G + "left_pad"},
        "pad_geoms": [P + G + n for n in PAD_GEOMS],
        "gripper_joints": [P + G + n for n in GRIPPER_JOINTS],
        "driver_joints": [P + G + "right_driver_joint", P + G + "left_driver_joint"],
        "gripper_actuator": P + G + "fingers_actuator",
        "tcp_site": P + cfg["tcp_name"],
        "ft_site": P + cfg["ft_site"],
        "force_sensor": P + cfg["force_sensor"],
        "torque_sensor": P + cfg["torque_sensor"],
        "wrist_camera_body": P + wc["mount"],
        "wrist_camera_site": P + wc["site"],
        "wrist_camera": (P + wc["camera"]) if wc.get("camera") else None,
        "wrist_camera_housing": P + "wrist_cam_housing",
        "d6_m": float(rob["d6_m"]),
        "tcp_from_flange_m": GRIP_MOUNT_Z + 0.0038 + PINCH_IN_BASE[2],
    }
