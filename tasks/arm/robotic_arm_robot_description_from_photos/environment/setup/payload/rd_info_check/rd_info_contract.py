"""rd_info.contract: the public contract of the robot-description stage (payload-safe, stdlib only).

Names, conventions and bands that the instruction / robot_contract.md state and the grader enforces. Nothing here
depends on any instance (no GT number): the same file ships in the task payload and with the grader.

Conventions:
  world    the frame of the published base (origin on the table top, the mounting plane, nominally on the base-yaw
           axis), z up, +y = the base's forward direction at
           design yaw 0, +x = the pitch-axis direction at yaw 0; SI units (m, rad, kg).
  q        design angle = the qpos of robot.xml; q = 0 is the straight-up configuration; no `ref` on any joint.
  bus      b = (ticks - 2048) x 2 pi / 4096 (the servo reading in radians); calibration.json per joint:
           b = sign x (q - zero_offset_rad)  <=>  q = sign x b + zero_offset_rad.
  tcp      site `tcp`, rigid on `gripper`: on the fixed finger's gripping face, on its mid-line, 10.0 mm back (along the
           finger) from the fingertip; +z = approach (along the finger, out of the gripper), +x = the face normal towards
           the moving jaw. `jaw_tip`: the same construction on the moving jaw's gripping face (rigid on `jaw`; +x towards
           the fixed finger).
"""

CONTRACT_VERSION = "rd-info-contract/1.0.0"

JOINTS = ["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll", "jaw"]
ARM_JOINTS = JOINTS[:5]
BODIES = ["base", "shoulder", "upper_arm", "forearm", "wrist", "gripper", "jaw"]
PARENT = {"base": "world", "shoulder": "base", "upper_arm": "shoulder", "forearm": "upper_arm", "wrist": "forearm",
          "gripper": "wrist", "jaw": "gripper"}
JOINT_OF_BODY = {"shoulder": "base_yaw", "upper_arm": "shoulder_pitch", "forearm": "elbow_pitch", "wrist": "wrist_pitch",
                 "gripper": "wrist_roll", "jaw": "jaw"}
BODY_OF_JOINT = {j: b for b, j in JOINT_OF_BODY.items()}
SITES = {"tcp": "gripper", "jaw_tip": "jaw"}
ACTUATORS = {"act_" + j: j for j in JOINTS}
SERVO_IDS = {"base_yaw": 1, "shoulder_pitch": 2, "elbow_pitch": 3, "wrist_pitch": 4, "wrist_roll": 5, "jaw": 6}

TICKS_PER_REV = 4096
TICK_ZERO = 2048
STALL_NM = 2.94               # BUS-SERVO-12V-A at 12 V (datasheet)

# caps of the gate
CAPS = {
    "max_submission_bytes": 200 * 1024 * 1024,
    "max_mesh_faces": 400000,
    "max_collision_geoms_total": 320,
    "max_collision_geoms_per_body": 48,
    "max_collision_mesh_faces": 50000,
    "criterion_timeout_s": 300.0,
}

# actuator / joint bands of the tier servo class (per joint: one servo drives each joint). Published in
# robot_contract.md; the rule set of `limits_actuators` (the actuator half).
ACTUATOR_BANDS = {
    "kp": [250.0, 2500.0],            # N m / rad (position servo)
    "kv": [0.5, 8.0],                 # N m s / rad
    "armature": [0.005, 0.1],         # kg m^2 (reflected rotor inertia)
    "damping": [0.1, 3.0],            # N m s / rad
    "frictionloss": [0.01, 0.25],     # N m
    "forcerange_fraction_of_stall": [0.5, 1.0],
    "ctrlrange_tol_rad": 1e-3,
}


def bus_from_ticks(t):
    import math
    return (float(t) - TICK_ZERO) * 2.0 * math.pi / TICKS_PER_REV


def ticks_from_bus(b):
    import math
    return int(round(float(b) * TICKS_PER_REV / (2.0 * math.pi))) + TICK_ZERO


def q_from_bus(b, cal_joint):
    return float(cal_joint["sign"]) * float(b) + float(cal_joint["zero_offset_rad"])


def bus_from_q(q, cal_joint):
    return float(cal_joint["sign"]) * (float(q) - float(cal_joint["zero_offset_rad"]))
