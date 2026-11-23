"""The agent's model files, generated from the cell (a maintainer's tool; the grader never runs it).

    python3 -B tests/review/docs/make_models.py --write | --check

given/robot/
    dual_arm_with_grippers.xml   the robot with the maker's data: the cell's kinematics, collision shapes and torque
                                 limits exactly, the dynamics nominal (masses, inertias, joint damping, friction loss,
                                 armature = the cell's x the hidden factors of docs/nominal_dynamics.json), torque
                                 inputs instead of the position servos (no gain), the pads' friction the middle of the
                                 data sheet's range, the cameras' drawing poses and data-sheet fields of view, a
                                 keyframe "start" (the start posture); every mesh under meshes/
    dual_arm.xml, meshes/*.STL   the arm alone (the cell's arm model with the same nominal dynamics; the meshes are
                                 byte copies)
    gripper_2f85/                the gripper alone (2f85.xml with the same nominal dynamics; assets/*.stl, LICENSE
                                 byte copies)
    meshes/*.stl                 the gripper meshes again, for dual_arm_with_grippers.xml
given/station/
    station.xml                  the station's geometry only: the collision boxes of describe()["station"], the belt
                                 surface as a fixed box, the infeed camera, marker sites for the lines; default
                                 contact settings, no drive, no parcel, no light
    station_with_robot.xml       station.xml with the robot attached at the origin (no parcel)

Nothing of a shift is in these files (no parcel, no hidden value, no true dynamic value). ``check_models()`` compiles
every file and compares it with the cell: joint names, ranges and axes, the torque inputs at the drives' limits, every
mass, inertia, damping, friction loss and armature a nominal value (inside its tolerance, never the true one), the
tool centre point and the cameras (the drawing's).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.dirname(HERE)
TESTS = os.path.dirname(REVIEW)
TASK = os.path.dirname(TESTS)
sys.path.insert(0, TESTS)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

ASSETS = os.path.join(TESTS, "parcelcell", "assets")
GIVEN = os.path.join(TASK, "environment", "setup", "payload", "given")
ROBOT_OUT = os.path.join(GIVEN, "robot")
STATION_OUT = os.path.join(GIVEN, "station")

ROBOT_README = """# The robot's model files

MuJoCo models (MJCF) of the station's dual-arm robot with the maker's data, for your own kinematics, collision checks
and simulation. They hold no station and no parcel (the station's geometry: `../station/`).

| file | content |
|---|---|
| `dual_arm_with_grippers.xml` | the robot: the arm model with a gripper attached at each wrist flange site (its coupler joints locked: each finger is one rigid linkage driven by its driver joint), the tool centre point sites, the finger pads, the palm cameras flush in the grippers' palms (no housing, no collision shape), the head camera's housing, the cameras' drawing poses and data-sheet fields of view, and a keyframe `start` (the start posture of `describe()["robot"]["start_posture"]`) |
| `dual_arm.xml`, `meshes/*.STL` | the arm alone (column, head, two arms of seven joints; no gripper) |
| `gripper_2f85/` | the gripper alone: the MuJoCo Menagerie model of the Robotiq 2F-85 (`2f85.xml`, `assets/`), BSD-2-Clause (`gripper_2f85/LICENSE`, copyright 2013 ROS-Industrial): its geometry and kinematics as published, its dynamic values the station's nominal data (below) |

What the files are exact in, and what they are not:

* Exact: the kinematics (bodies, joint axes and ranges, the flange and tool centre point sites), the collision shapes,
  the torque limits (`forcerange`) and the meshes.
* Nominal: every link's mass and inertia and every joint's damping, friction loss and armature are the maker's
  nominal data; the station's robot differs from them within the tolerances of `cell_manual.md` section 4.
* Not in the files: the drives' servo gains. Each arm actuator here is a torque input (newton metres, limited to the
  drive's torque limit), not the station's position servo (`cell_manual.md` section 4 describes the servos).
* The finger pads' friction coefficient here is the middle of the data sheet's range (`cell_manual.md` section 5).
* The cameras' poses are the installation drawing's and their fields of view the data sheet's; the installed cameras
  differ within the tolerances of `cell_manual.md` section 6.

Frames and names:

* World frame: the column's base at the origin, z up, the robot facing +x (the belt runs along y in front of it).
* Joints `Left_<j>_Joint` and `Right_<j>_Joint` with `<j>` = `Shoulder_Pitch`, `Shoulder_Roll`, `Shoulder_Yaw`,
  `Elbow_Pitch`, `Wrist_Yaw`, `Wrist_Pitch`, `Wrist_Roll`: the order of every 14-vector on the wire is left arm then
  right arm, each in this order. The actuators carry the joints' names.
* The right arm is a mirror image of the left about the x-z plane: a mirror-symmetric posture has the same
  `Elbow_Pitch` angle on both arms and the opposite angle at every other joint (the ranges are not symmetric).
* Sites `Left_flange` and `Right_flange`: the wrist flanges, z axis = the tool axis. The grippers are attached there,
  their bodies, joints and geoms prefixed `L_` and `R_`. Sites `Left_tcp` and `Right_tcp`: the tool centre points
  between the finger pads (`describe()["robot"]["T_flange_tcp"]`).
* Cameras: `head_cam` (the `head` camera), `Left_wrist_cam` and `Right_wrist_cam` (`wrist_left`, `wrist_right`: palm
  cameras, flush in each gripper base's palm between the finger roots, looking along the tool axis; no housing and no
  collision shape of their own); the head camera's housing `head_camera_housing` on the column's body. MuJoCo cameras
  look along their -z axis with +y up; the cell publishes every camera pose in OpenCV axes (`describe()["cameras"]`).
* Geom groups: 1 robot visuals, 2 gripper visuals (no contact), 3 collision shapes, 4 sites.
* Units: metres, radians, kilograms. The timestep is the cell's physics step.

The `<option>` block is the arm model's own with the cell's `impratio` (its timestep is the cell's physics step); the
cell's simulation of the station and the parcels (the belt's drive, the station's and the parcels' contact
parameters) is not in these files.
"""

STATION_README = """# The station's geometry

`station.xml` is the station's geometry as a MuJoCo model (MJCF): the conveyor's collision boxes exactly as the cell
has them (`describe()["station"]["hardware"]` and `cell_manual.md` section 2), the belt surface as a fixed box, the
floor, the `infeed` camera (`inspect_cam`, at the installation drawing's pose with the data sheet's field of view and
image size; the installed camera differs within the tolerances of `cell_manual.md` section 6) and marker sites for the lines
(`entry_line`: the upstream end of the set-down range, `scan_line`, `belt_end`). `station_with_robot.xml` is the
same with the robot of
`../robot/dual_arm_with_grippers.xml` attached at the origin.

It is geometry only: the friction, contact and drive settings in these files are MuJoCo's defaults, not the cell's;
the belt does not move; there is no parcel and no light. World frame and units as in `cell_manual.md` section 1.
"""


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _cell_cfg():
    sys.path.insert(0, REVIEW)
    from common import load_cfg
    return load_cfg()


NOMINAL_FILE = os.path.join(HERE, "nominal_dynamics.json")
# the friction coefficients the data sheet gives as ranges (the cell's true values lie inside: test_docs.py); the
# robot model carries the range's middle for the pads
PAD_FRICTION_RANGE = (1.5, 2.3)
BELT_FRICTION_RANGE = (0.75, 1.15)


def nominal_factors():
    """The hidden factors between the maker's nominal data and the cell's robot (draw_installed.py dynamics)."""
    import json
    with open(NOMINAL_FILE) as fh:
        return json.load(fh)


def _default_classes(spec):
    xml = spec.to_xml()
    return re.findall(r'<default class="([^"]+)">', xml)


def nominalize(spec, true_model, factors, prefix_map=None):
    """The spec's dynamics as the maker's nominal data: every body's mass and inertia (explicit, in its principal
    frame) and every joint's damping, friction loss and armature = the compiled true model's x the hidden factor of
    that quantity; the default classes carry none of these values; every arm actuator becomes a torque input limited
    to the drive's torque limit (the servos' gains are not part of the model). ``prefix_map``: names in ``spec`` ->
    names in the factor table (a standalone gripper takes the left gripper's factors)."""
    import mujoco
    import numpy as np
    from parcelcell.robot import JOINT_NAMES
    pm = prefix_map or (lambda n: n)
    m = true_model
    for cls in _default_classes(spec):
        d = spec.find_default(cls)
        if d is not None:
            d.joint.damping = np.zeros_like(np.asarray(d.joint.damping, float))
            d.joint.armature = 0.0
            d.joint.frictionloss = 0.0
    for b in spec.bodies:
        if not b.name:
            continue
        f = factors["bodies"].get(pm(b.name))
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, pm(b.name) if prefix_map else b.name)
        if f is None or bid < 0:
            continue
        b.explicitinertial = True
        b.mass = float(m.body_mass[bid]) * float(f["mass"])
        b.ipos = [float(v) for v in m.body_ipos[bid]]
        b.iquat = [float(v) for v in m.body_iquat[bid]]
        b.inertia = [float(v) * float(f["inertia"]) for v in m.body_inertia[bid]]
        b.fullinertia = [float("nan")] * 6
    for j in spec.joints:
        f = factors["joints"].get(pm(j.name))
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, pm(j.name) if prefix_map else j.name)
        if f is None or jid < 0:
            continue
        dof = int(m.jnt_dofadr[jid])
        dmp = np.zeros_like(np.asarray(j.damping, float))
        dmp[0] = float(m.dof_damping[dof]) * float(f["damping"])
        j.damping = dmp
        j.frictionloss = float(m.dof_frictionloss[dof]) * float(f["frictionloss"])
        j.armature = float(m.dof_armature[dof]) * float(f["armature"])
    for n in JOINT_NAMES:
        a = spec.actuator(n)
        if a is None:
            continue
        lim = [float(v) for v in a.forcerange]
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        g = np.zeros_like(np.asarray(a.gainprm, float))
        g[0] = 1.0
        a.gainprm = g
        a.biastype = mujoco.mjtBias.mjBIAS_NONE
        a.biasprm = np.zeros_like(np.asarray(a.biasprm, float))
        a.ctrlrange = lim
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    return spec


def _true_robot_model(cfg):
    """The cell's robot as compiled (the truth the factors apply to)."""
    from parcelcell.robot import build_robot_spec
    wc = cfg["world"]
    return build_robot_spec(bool(wc.get("lock_gripper_couplers", True)), wc.get("robot_mounts") or None).compile()


def arm_xml(cfg) -> str:
    """The arm alone (the cell's arm model) with the maker's nominal dynamics."""
    import mujoco
    spec = mujoco.MjSpec.from_file(os.path.join(ASSETS, "robot", "dual_arm.xml"))
    nominalize(spec, _true_robot_model(cfg), nominal_factors())
    xml = spec.to_xml()
    return re.sub(r'file="([^"]+)"', lambda mo: 'file="%s"' % os.path.basename(mo.group(1)), xml)


def gripper_xml(cfg) -> str:
    """The gripper alone (the Menagerie model the cell attaches) with the nominal dynamics of the station's units."""
    import mujoco
    spec = mujoco.MjSpec.from_file(os.path.join(ASSETS, "gripper_2f85", "2f85.xml"))
    nominalize(spec, _true_robot_model(cfg), nominal_factors(), prefix_map=lambda n: "L_" + n)
    return spec.to_xml()


def robot_xml(cfg) -> str:
    """The combined robot model as the cell builds it (robot.build_robot_spec, the soft pads of
    the cell's configuration, the cameras' fields of view), meshes flattened into meshes/."""
    import mujoco
    import numpy as np
    from parcelcell.robot import GRIPPER, SIDES, build_robot_spec
    wc = cfg["world"]
    spec = build_robot_spec(bool(wc.get("lock_gripper_couplers", True)), wc.get("robot_mounts") or None)
    spec.modelname = "dual_arm_with_grippers"
    fp = wc["finger_pads"]
    if fp.get("soft", True):
        for s in SIDES:
            for gname in GRIPPER.pad_geoms:
                g = spec.geom(f"{s[0]}_{gname}")
                g.solref = list(fp["solref"])
                g.condim = int(fp.get("condim", 4))
                fr = list(g.friction)
                fr[0] = round(0.5 * (PAD_FRICTION_RANGE[0] + PAD_FRICTION_RANGE[1]), 4)     # the data sheet's middle
                g.friction = fr
    spec.option.impratio = float(wc["solver"].get("impratio", 100.0))
    for name, c in cfg["cameras"]["list"].items():
        cam = spec.camera(c["model_camera"])
        if cam is not None:
            cam.fovy = float(c["fovy_deg"])
            cam.resolution = [int(c["size"][0]), int(c["size"][1])]
    nominalize(spec, _true_robot_model(cfg), nominal_factors())
    m = spec.compile()
    q = np.array(m.qpos0).copy()
    from parcelcell.robot import JOINT_NAMES
    for i, n in enumerate(JOINT_NAMES):
        q[m.jnt_qposadr[m.joint(n).id]] = float(cfg["world"]["start_posture"][i])
    key = spec.add_key()
    key.name = "start"
    key.qpos = q.tolist()
    key.ctrl = np.zeros(m.nu).tolist()          # torque inputs: zero (the grippers open)
    xml = spec.to_xml()
    xml = re.sub(r'file="([^"]+)"', lambda mo: 'file="%s"' % os.path.basename(mo.group(1)), xml)
    xml = re.sub(r'meshdir="[^"]*"', 'meshdir="meshes"', xml)
    if 'meshdir="meshes"' not in xml:
        xml = xml.replace("<compiler ", '<compiler meshdir="meshes" ', 1)
    return xml


ROBOT_MOUNTED = ("head_camera_housing",)


def _box(name, centre, half, rgba="0.5 0.5 0.5 1"):
    return ('    <geom name="%s" type="box" pos="%s" size="%s" group="3" rgba="%s"/>\n'
            % (name, " ".join("%.6g" % v for v in centre), " ".join("%.6g" % v for v in half), rgba))


def station_xml(cfg, describe) -> str:
    from parcelcell.station import SLAB_HALF_LEN, SLAB_HZ
    st = describe["station"]
    b, lines = st["belt"], st["lines"]
    cam = cfg["cameras"]["list"]["infeed"]
    T = describe["cameras"]["infeed"]["T_world_cam"]
    import numpy as np
    Tm = np.asarray(T)
    R = Tm[:3, :3] @ np.diag([1.0, -1.0, -1.0])          # OpenCV -> MuJoCo camera axes
    xy = list(R[:, 0]) + list(R[:, 1])
    out = io.StringIO()
    out.write('<mujoco model="parcel_station_geometry">\n')
    out.write('  <!-- The label-facing station: geometry only (no drive, no parcel, default contact settings). '
              'World frame: the robot column at the origin, z up, the robot facing +x, the belt along +y. -->\n')
    out.write('  <compiler angle="radian"/>\n')
    out.write('  <option timestep="%g"/>\n' % describe["physics_step_s"])
    out.write('  <worldbody>\n')
    out.write(_box("floor", [0, 0, -0.05], [4, 4, 0.05], "0.42 0.45 0.48 1"))
    out.write(_box("belt_surface", [b["centre_x"], 0.0, b["top_z"] - SLAB_HZ], [b["width"] / 2, SLAB_HALF_LEN, SLAB_HZ],
                   "0.1 0.1 0.1 1"))
    for name in sorted(st["hardware"]):
        if name in ROBOT_MOUNTED:
            continue                                     # in the robot's own model
        h = st["hardware"][name]
        out.write(_box(name, h["centre"], h["half_size"], "0.55 0.57 0.6 1"))
    w = b["width"] / 2 + 0.03
    for sname, y in (("entry_line", lines["entry_y"]), ("scan_line", lines["scan_y"]), ("belt_end", lines["exit_y"])):
        out.write('    <site name="%s" type="box" pos="%.6g %.6g %.6g" size="%.6g 0.002 0.0005" group="4" '
                  'rgba="0.9 0.6 0.1 1"/>\n' % (sname, b["centre_x"], y, b["top_z"] + 0.0005, w))
    out.write('    <camera name="%s" pos="%s" xyaxes="%s" fovy="%.6g" resolution="%d %d"/>\n'
              % (cam["model_camera"], " ".join("%.6g" % v for v in Tm[:3, 3]), " ".join("%.6g" % v for v in xy),
                 cam["fovy_deg"], cam["size"][0], cam["size"][1]))
    out.write('  </worldbody>\n</mujoco>\n')
    return out.getvalue()


def station_with_robot_xml(station: str, robot_xml_text: str) -> str:
    """The robot model with the station's bodies inserted into its world body (text level, so that
    the robot's numbers stay as written); meshes from ../robot/meshes."""
    body = station.split("<worldbody>", 1)[1].split("</worldbody>", 1)[0]
    xml = robot_xml_text.replace('model="dual_arm_with_grippers"', 'model="parcel_station_with_robot"', 1)
    if "<worldbody>" not in xml:
        raise ValueError("robot model without a world body")
    xml = xml.replace("<worldbody>", "<worldbody>\n    <!-- the station (geometry only, ../station/station.xml) -->" +
                      body.rstrip() + "\n    <!-- the robot -->", 1)
    xml = xml.replace('meshdir="meshes"', 'meshdir="../robot/meshes"', 1)
    return xml


def build(cfg=None):
    """{path relative to given/: bytes}."""
    from facts import _empty_cell
    cfg = cfg or _cell_cfg()
    srv, info = _empty_cell(cfg)
    srv.close()
    out = {}
    rx = robot_xml(cfg)
    out["robot/dual_arm_with_grippers.xml"] = rx.encode()
    out["robot/dual_arm.xml"] = arm_xml(cfg).encode()
    for f in sorted(os.listdir(os.path.join(ASSETS, "robot", "meshes"))):
        out["robot/meshes/" + f] = open(os.path.join(ASSETS, "robot", "meshes", f), "rb").read()
    g = os.path.join(ASSETS, "gripper_2f85")
    out["robot/gripper_2f85/2f85.xml"] = gripper_xml(cfg).encode()
    out["robot/gripper_2f85/LICENSE"] = open(os.path.join(g, "LICENSE"), "rb").read()
    for f in sorted(os.listdir(os.path.join(g, "assets"))):
        data = open(os.path.join(g, "assets", f), "rb").read()
        out["robot/gripper_2f85/assets/" + f] = data
        out["robot/meshes/" + f] = data
    out["robot/README.md"] = ROBOT_README.encode()
    sx = station_xml(cfg, info)
    out["station/station.xml"] = sx.encode()
    out["station/station_with_robot.xml"] = station_with_robot_xml(sx, rx).encode()
    out["station/README.md"] = STATION_README.encode()
    return out, info


def write(files):
    tmp = os.path.join(GIVEN, ".models_tmp")
    if os.path.exists(tmp):
        raise SystemExit("leftover %s: remove it by hand" % tmp)
    for rel, data in files.items():
        p = os.path.join(GIVEN, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p) or open(p, "rb").read() != data:
            with open(p, "wb") as fh:
                fh.write(data)
            print("wrote given/%s" % rel)


def _nominal_ok(nominal, true, tol):
    """A nominal value: within ``tol`` of the true one and, unless both are zero, at least a quarter of ``tol`` off it
    (never the true value)."""
    nominal, true = float(nominal), float(true)
    if true == 0.0:
        return nominal == 0.0
    r = nominal / true
    return (1.0 - tol - 1e-9) <= r <= (1.0 + tol + 1e-9) and abs(r - 1.0) >= tol / 4.0 - 1e-6


def check_models(info=None):
    """Compile every given model and compare it with the cell. Returns a list of problems."""
    import mujoco
    import numpy as np
    from facts import _empty_cell
    from parcelcell.robot import JOINT_NAMES
    cfg = _cell_cfg()
    srv, info = _empty_cell(cfg)
    cm = srv.world.model
    probs = []
    fac = nominal_factors()
    MASS_TOL, JOINT_TOL = float(fac["mass_tolerance"]), float(fac["joint_tolerance"])
    for rel in ("robot/dual_arm_with_grippers.xml", "station/station_with_robot.xml"):
        p = os.path.join(GIVEN, rel)
        try:
            m = mujoco.MjModel.from_xml_path(p)
        except Exception as e:  # noqa: BLE001
            probs.append("%s does not compile: %s" % (rel, e))
            continue
        for n in JOINT_NAMES:
            j, cj = m.joint(n), cm.joint(n)
            if not (np.allclose(j.range, cj.range) and np.allclose(m.jnt_axis[j.id], cm.jnt_axis[cj.id])):
                probs.append("%s: joint %s differs" % (rel, n))
            a, ca = m.actuator(n), cm.actuator(n)
            # a torque input (gain 1, no bias) limited to the drive's torque limit: no servo gain in the model
            if not (abs(a.gainprm[0] - 1.0) < 1e-12 and not np.any(a.gainprm[1:]) and not np.any(a.biasprm)
                    and np.allclose(a.forcerange, ca.forcerange) and np.allclose(a.ctrlrange, ca.forcerange)):
                probs.append("%s: actuator %s is not a torque input at the drive's limit" % (rel, n))
            jd, cd = int(m.jnt_dofadr[j.id]), int(cm.jnt_dofadr[cj.id])
            for fld, tol in (("dof_damping", JOINT_TOL), ("dof_frictionloss", JOINT_TOL), ("dof_armature", JOINT_TOL)):
                if not _nominal_ok(getattr(m, fld)[jd], getattr(cm, fld)[cd], tol):
                    probs.append("%s: joint %s %s is not a nominal value (%g against %g)"
                                 % (rel, n, fld[4:], getattr(m, fld)[jd], getattr(cm, fld)[cd]))
        for b in range(cm.nbody):
            name = cm.body(b).name
            if name.startswith(("Left_", "Right_", "L_", "R_", "Base_Link")):
                mb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
                if mb < 0 or not _nominal_ok(m.body_mass[mb], cm.body_mass[b], MASS_TOL) or \
                        not all(_nominal_ok(x, y, MASS_TOL) for x, y in zip(m.body_inertia[mb], cm.body_inertia[b])):
                    probs.append("%s: body %s mass or inertia is not a nominal value, or is missing" % (rel, name))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("start").id)
        mujoco.mj_forward(m, d)
        dc = mujoco.MjData(cm)
        dc.qpos[:] = srv.world.data.qpos
        dc.qpos[srv.world._arm_qadr] = np.asarray(cfg["world"]["start_posture"], float)
        mujoco.mj_kinematics(cm, dc)
        for s in ("Left_tcp", "Right_tcp"):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
            p0 = dc.site_xpos[mujoco.mj_name2id(cm, mujoco.mjtObj.mjOBJ_SITE, s)]
            if sid < 0 or np.linalg.norm(d.site_xpos[sid] - p0) > 2e-5:
                probs.append("%s: %s at the start keyframe differs from the cell's start posture" % (rel, s))
        for name, c in cfg["cameras"]["list"].items():
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c["model_camera"])
            if name == "infeed" and rel.startswith("robot/"):
                continue
            if cid < 0 or abs(m.cam_fovy[cid] - float(c["fovy_deg"])) > 1e-9:
                probs.append("%s: camera %s missing or its field of view differs" % (rel, c["model_camera"]))
            elif "flange_site" in c:
                fs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, c["flange_site"])
                Tf = np.eye(4)
                Tf[:3, :3], Tf[:3, 3] = d.site_xmat[fs].reshape(3, 3), d.site_xpos[fs]
                Tc = np.eye(4)
                Tc[:3, :3], Tc[:3, 3] = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1]), d.cam_xpos[cid]
                if np.abs(np.linalg.inv(Tf) @ Tc - np.asarray(info["cameras"][name]["T_flange_cam"])).max() > 2e-5:
                    probs.append("%s: camera %s mount differs from describe()" % (rel, c["model_camera"]))
            elif name == "infeed" or "flange_site" not in c:
                T = np.asarray(info["cameras"][name]["T_world_cam"])
                if np.linalg.norm(d.cam_xpos[cid] - T[:3, 3]) > 2e-5 or \
                        np.abs(d.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1]) - T[:3, :3]).max() > 2e-5:
                    probs.append("%s: camera %s pose differs from describe()" % (rel, c["model_camera"]))
    # station geometry = describe()
    ms = mujoco.MjModel.from_xml_path(os.path.join(GIVEN, "station", "station.xml"))
    for name, h in info["station"]["hardware"].items():
        if name in ROBOT_MOUNTED:
            mr = mujoco.MjModel.from_xml_path(os.path.join(GIVEN, "robot", "dual_arm_with_grippers.xml"))
            g = mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_GEOM, name)
            cg = srv.world._gid(name)
            if g < 0 or not (np.allclose(mr.geom_pos[g], cm.geom_pos[cg]) and np.allclose(mr.geom_size[g], cm.geom_size[cg])):
                probs.append("dual_arm_with_grippers.xml: %s differs from the cell" % name)
            continue
        g = mujoco.mj_name2id(ms, mujoco.mjtObj.mjOBJ_GEOM, name)
        cg = srv.world._gid(name)
        if g < 0 or not (np.allclose(ms.geom_pos[g], cm.geom_pos[cg]) and np.allclose(ms.geom_size[g], cm.geom_size[cg])):
            probs.append("station.xml: %s differs from the cell" % name)
    bs = mujoco.mj_name2id(ms, mujoco.mjtObj.mjOBJ_GEOM, "belt_surface")
    if bs < 0 or not np.allclose(ms.geom_size[bs], cm.geom_size[srv.world._belt_surface]):
        probs.append("station.xml: belt surface differs")
    srv.close()
    return probs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    files, info = build()
    rc = 0
    if a.write:
        write(files)
    for rel, data in files.items():
        p = os.path.join(GIVEN, rel)
        if not os.path.exists(p) or open(p, "rb").read() != data:
            print("differs from its build: given/%s" % rel)
            rc = 1
    probs = check_models()
    for p in probs:
        print("model check:", p)
    return 1 if (rc or probs) else 0


if __name__ == "__main__":
    sys.exit(main())
