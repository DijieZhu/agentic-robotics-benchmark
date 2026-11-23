"""The dual-arm platform: a fixed column with two 7-DOF arms and two parallel
grippers (Robotiq 2F-85 from MuJoCo Menagerie) on the wrist flanges.

World frame: the column stands at the origin, the robot faces +x, z is up.

Joint order
-----------
There is one joint order in this package, ``JOINT_NAMES``: left arm then right
arm, each in kinematic-chain order

    Shoulder_Pitch, Shoulder_Roll, Shoulder_Yaw, Elbow_Pitch,
    Wrist_Yaw, Wrist_Pitch, Wrist_Roll

Everything inside the package addresses joints and actuators by NAME; a
14-vector is always in ``JOINT_NAMES`` order. Do not zip a vector in another
order (for instance a logged-data order with the last two wrist joints
swapped) against this list by position: wrist roll and pitch would be swapped
silently, and reads and writes through the same wrong map stay
self-consistent, so closed-loop tracking hides the error.

Mirrored right arm
------------------
The right arm is a mirror image of the left about the x-z plane, but its joint
axes are not all mirrored the same way: a mirror-symmetric posture has the same
Elbow_Pitch angle on both arms and the opposite angle at every other joint
(checked on the model: right = (-q0, -q1, -q2, q3, -q4, -q5, -q6) of the left
gives the left tool centre point with y negated; the same angles do not). The
start posture is such a pair. The joint ranges are not symmetric (see
``joint_ranges``).

Drives
------
Position actuators (the robot's position mode). Shoulder pitch and roll use
the large drive (torque limit 80 N m, kp 900, kv 90); the other five joints
the small drive (36 N m, kp 700, kv 60). All drives: speed limit 10 rad/s in
the manufacturer's actuator parameter file (not enforced by the simulator; see
``ParcelCell`` for the optional target slew limit), MIT-mode gain ranges
kp <= 1000, kd <= 500. Timestep 2 ms (the 500 Hz bus), control tick 20 ms.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

__all__ = [
    "ARM_JOINTS", "JOINT_NAMES", "HOME_QPOS_RAW", "GRIPPER", "GripperInfo",
    "DRIVE_LIMITS", "JOINT_DRIVE", "build_robot_spec", "home_qpos",
    "ROBOT_XML", "GRIPPER_XML", "SIDES", "joint_ranges",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_XML = os.path.join(_HERE, "assets", "robot", "dual_arm.xml")
GRIPPER_XML = os.path.join(_HERE, "assets", "gripper_2f85", "2f85.xml")

SIDES = ("Left", "Right")
_CHAIN = ("Shoulder_Pitch", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch",
          "Wrist_Yaw", "Wrist_Pitch", "Wrist_Roll")
ARM_JOINTS = {side: tuple(f"{side}_{j}_Joint" for j in _CHAIN) for side in SIDES}
JOINT_NAMES: tuple[str, ...] = ARM_JOINTS["Left"] + ARM_JOINTS["Right"]

# Drive data from the manufacturer's actuator parameter file (per drive class)
# and the generated MJCF (gains, armature, damping, friction loss).
DRIVE_LIMITS = {
    "drive_80nm": dict(torque_nm=80.0, speed_rad_s=10.0, kp=900.0, kv=90.0,
                       armature=0.05, damping=2.0, frictionloss=0.3,
                       pos_cmd_range_rad=(-10.0, 10.0),
                       mit_kp_max=1000.0, mit_kd_max=500.0),
    "drive_36nm": dict(torque_nm=36.0, speed_rad_s=10.0, kp=700.0, kv=60.0,
                       armature=0.03, damping=1.0, frictionloss=0.2,
                       pos_cmd_range_rad=(-10.0, 10.0),
                       mit_kp_max=1000.0, mit_kd_max=500.0),
}
JOINT_DRIVE = {
    f"{side}_{j}_Joint": ("drive_80nm" if j in ("Shoulder_Pitch", "Shoulder_Roll")
                          else "drive_36nm")
    for side in SIDES for j in _CHAIN
}

# Source home posture. The elbow value -1.8 lies outside the corrected elbow
# range [-0.3, 2.5] (the vendor URDF had the elbow direction reversed); in the
# source it was written raw and pulled back to the limit during the settle.
# ``home_qpos`` clips it to the joint range, which is where the source's arms
# end up after the settle.
HOME_QPOS_RAW = {
    "Left_Shoulder_Pitch_Joint": -0.5, "Left_Shoulder_Roll_Joint": 0.35,
    "Left_Elbow_Pitch_Joint": -1.8, "Left_Wrist_Pitch_Joint": 0.3,
    "Right_Shoulder_Pitch_Joint": 0.5, "Right_Shoulder_Roll_Joint": -0.35,
    "Right_Elbow_Pitch_Joint": -1.8, "Right_Wrist_Pitch_Joint": -0.3,
}


@dataclass(frozen=True)
class GripperInfo:
    """How the gripper is driven (relative to its MJCF root body)."""
    xml: str
    actuator: str            # finger actuator name (after attach: L_/R_ prefix)
    driver_joint: str        # one driver joint, its angle measures closure
    driver_range: tuple      # (open, closed) driver angle (rad)
    ctrl_open: float
    ctrl_close: float
    pinch_pos: tuple         # grasp centre (TCP) in the gripper root body
    max_width: float         # max opening (m)
    pad_geoms: tuple         # finger pad collision boxes (names before prefix)


GRIPPER = GripperInfo(
    xml=GRIPPER_XML, actuator="fingers_actuator",
    driver_joint="right_driver_joint", driver_range=(0.0, 0.8),
    ctrl_open=0.0, ctrl_close=255.0,
    # base_mount (0.007) + base (0.0038) + pinch site (0.145)
    pinch_pos=(0.0, 0.0, 0.1558), max_width=0.085,
    pad_geoms=("right_pad1", "right_pad2", "left_pad1", "left_pad2"),
)


LOCK_COUPLER = True
"""Delete the two coupler joints of each 2F-85 (the coupler becomes rigid on
its driver): the four-bar is then a 1-DOF linkage and the fingertip pads stay
on the parallel-grip path under tip and side loads, as on the real gripper in
a fingertip contact. Without it the Menagerie model's passive spring-link DOF
(0.05 N m/rad) lets the fingers fold under ~10-40 N."""


def build_robot_spec(lock_coupler: bool | None = None,
                     mounts: dict | None = None) -> mujoco.MjSpec:
    """Arm MJCF + two grippers attached at ``Left_flange`` / ``Right_flange``
    (flange site z axis = tool axis). Adds ``Left_tcp`` / ``Right_tcp`` sites
    at the grasp centre. ``lock_coupler``: None = ``LOCK_COUPLER``.
    ``mounts``: optional robot-mounted hardware (``add_mounts``), added after
    everything else. Returns an editable MjSpec."""
    if lock_coupler is None:
        lock_coupler = LOCK_COUPLER
    spec = mujoco.MjSpec.from_file(ROBOT_XML)
    for side in SIDES:
        child = mujoco.MjSpec.from_file(GRIPPER.xml)
        for key in list(child.keys):        # gripper keyframes do not fit the
            key.delete()                    # combined model
        if lock_coupler:
            for n in ("right_coupler_joint", "left_coupler_joint"):
                child.delete(child.joint(n))
        site = spec.site(f"{side}_flange")
        spec.attach(child, prefix=f"{side[0]}_", site=site)
        root = _gripper_root_body(spec, side[0])
        root.add_site(name=f"{side}_tcp", pos=list(GRIPPER.pinch_pos),
                      size=[0.005, 0.005, 0.005], group=4)
    if mounts:
        add_mounts(spec, mounts)
    return spec


def _collision_box(body, name, pos, size):
    """A robot collision box (group 3, massless: the body keeps its explicit
    inertia)."""
    return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                         pos=[float(v) for v in pos],
                         size=[float(v) for v in size], mass=0, group=3,
                         rgba=[0.15, 0.15, 0.17, 1])


def wrist_camera_axes(side_sign: float, pitch_rad: float) -> list:
    """MuJoCo ``xyaxes`` of a camera on a gripper's root body that looks
    along the tool axis (+z), offset to ``side_sign`` x y of it and pitched
    ``pitch_rad`` towards it: image up points away from the tool axis (the
    fingertips appear at the bottom edge), image right along -side_sign x."""
    c, s = float(np.cos(pitch_rad)), float(np.sin(pitch_rad))
    sg = 1.0 if side_sign >= 0 else -1.0
    return [-sg, 0.0, 0.0, 0.0, sg * c, s]


def add_mounts(spec: mujoco.MjSpec, mounts: dict) -> None:
    """Robot-mounted hardware.

    ``head_camera_housing``: {"body", "centre", "half_size"}: a collision box
    on that body (centre in the body frame; the column's body frame is the
    world frame at the start).
    ``wrist_cameras``: {"Left": {...}, "Right": {...}} with "pos" (camera
    position in the gripper root body, which is the flange frame shifted
    7 mm along the tool axis), "side_sign", "pitch_deg", "fovy_deg" and,
    optionally, "housing" {"centre", "half_size"} (root body frame): the arm
    model's wrist camera ``<Side>_wrist_cam`` is moved onto the gripper root
    body; with a housing it gets a collision box ``<Side>_wrist_cam_housing``
    (the palm cameras of the task's cell have none: they sit flush in the
    palm)."""
    hh = mounts.get("head_camera_housing")
    if hh:
        _collision_box(spec.body(hh.get("body", "Base_Link")),
                       "head_camera_housing", hh["centre"], hh["half_size"])
    wc = mounts.get("wrist_cameras")
    if wc:
        for side in SIDES:
            c = wc.get(side)
            if not c:
                continue
            name = f"{side}_wrist_cam"
            old = spec.camera(name)
            fovy = float(c.get("fovy_deg", old.fovy if old is not None else 58.0))
            if old is not None:
                spec.delete(old)
            root = _gripper_root_body(spec, side[0])
            root.add_camera(name=name, pos=[float(v) for v in c["pos"]],
                            xyaxes=wrist_camera_axes(float(c["side_sign"]),
                                                     np.radians(float(c["pitch_deg"]))),
                            fovy=fovy)
            h = c.get("housing")
            if h:
                _collision_box(root, f"{side}_wrist_cam_housing", h["centre"],
                               h["half_size"])


def _gripper_root_body(spec: mujoco.MjSpec, prefix: str):
    for body in spec.bodies:
        if body.name.startswith(f"{prefix}_") and body.parent is not None \
                and body.parent.name.endswith("Wrist_Roll_Link"):
            return body
    raise RuntimeError(f"gripper root body with prefix {prefix}_ not found")


def joint_ranges(model: mujoco.MjModel) -> dict[str, tuple[float, float]]:
    out = {}
    for n in JOINT_NAMES:
        j = model.joint(n)
        out[n] = (float(j.range[0]), float(j.range[1]))
    return out


def home_qpos(model: mujoco.MjModel | None = None) -> np.ndarray:
    """Home posture as a 14-vector in ``JOINT_NAMES`` order, clipped to the
    joint ranges (needs ``model`` for the ranges; without it the ranges of the
    shipped MJCF are used)."""
    q = np.array([HOME_QPOS_RAW.get(n, 0.0) for n in JOINT_NAMES])
    if model is None:
        model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    rng = joint_ranges(model)
    lo = np.array([rng[n][0] for n in JOINT_NAMES])
    hi = np.array([rng[n][1] for n in JOINT_NAMES])
    return np.clip(q, lo, hi)
