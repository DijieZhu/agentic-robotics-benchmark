"""Constants of the sketch contract `sketch-contract/1.2` (racecar_time_trial_on_unseen_track, as amended).

The machine-readable twin of the tables in the contract. Every package may import it
(`from sketch.examples import constants as K`, import root = the folder that holds `sketch/`); a test of the maintainers checks
that the numbers here and the numbers printed in the contract agree. Standard library only, no side effects.

Sources: the car's shared specification values (v2) and the owner model's digest for the car; the LiDAR axis
position is the contract's own measurement on the exported mesh; everything marked "starting value" is
calibrated by the part of the model that owns it; later rulings made them final.
"""

import math
import re

CONTRACT_VERSION = "sketch-contract/1.2"
WORLD_SCHEMA = "racecar-venue-world/1"
WIRE_PROTOCOL = "racecar-wire/1"
CONFIG_SCHEMA = "racecar-grader-config/1"
RESULT_SCHEMA = "racecar-venue-result/1"
REFERENCE_SCHEMA = "racecar-venue-reference/1"
SCORE_SCHEMA = "racecar-score/1"
LAB_LOG_SCHEMA = "racecar-lab-log/1"

# ---------------------------------------------------------------- time
TICK_S = 0.025                       # one `step` = one LiDAR revolution = 40 Hz
PHYSICS_DT_S = 0.0025                # 10 physics steps per tick
PHYSICS_STEPS_PER_TICK = 10
PHYSICS_DT_FALLBACK_S = 0.00125      # the drive model's fallback if 2.5 ms is not stable: 20 physics steps per tick
PHYSICS_STEPS_PER_TICK_FALLBACK = 20
SUBSCANS_PER_TICK = 5
IMU_SAMPLES_PER_TICK = 5
IMU_RATE_HZ = 200
CMD_DELAY_TICKS = 1                  # starting value
SESSIONS = ("practice", "tt1", "tt2", "avoid")


def subscan_step(k, steps_per_tick=PHYSICS_STEPS_PER_TICK):
    """1-based index of the physics step of a tick after which sub-scan k (0..4) is cast: t = (2k + 1) * 2.5 ms."""
    if steps_per_tick % 10:
        raise ValueError("physics steps per tick must be a multiple of 10")
    return (2 * k + 1) * (steps_per_tick // 10)


def imu_step(j, steps_per_tick=PHYSICS_STEPS_PER_TICK):
    """1-based index of the physics step of a tick after which IMU sample j (0..4) is complete: t = 5 (j + 1) ms."""
    if steps_per_tick % 10:
        raise ValueError("physics steps per tick must be a multiple of 10")
    return 2 * (j + 1) * (steps_per_tick // 10)


# ---------------------------------------------------------------- LiDAR
N_BEAMS = 1081
ANGLE_MIN_RAD = -0.75 * math.pi      # beam 0: -135 deg (right rear); beam 540: straight ahead; beam 1080: +135 deg
ANGLE_INC_RAD = math.pi / 720.0      # 0.25 deg
RANGE_MIN_M = 0.06
RANGE_MAX_M = 10.0
NO_RETURN = 0                        # wire value of a beam without a return (integer mm)
SUBSCAN_SLICES = ((0, 216), (216, 432), (432, 648), (648, 864), (864, 1081))


def beam_angle(i):
    """Angle of beam i in the LiDAR frame (rad, counter-clockwise from the car's forward axis)."""
    return ANGLE_MIN_RAD + i * ANGLE_INC_RAD


# ---------------------------------------------------------------- the car, car frame C, metres
WHEELBASE_M = 0.25784
TRACK_FRONT_M = 0.15591
TRACK_REAR_M = 0.15595
TYRE_RADIUS_M = 0.03290
TYRE_WIDTH_M = 0.02719
MASS_KG = 1.8359
COM_C = (0.12495, -0.00263, 0.05014)
ENVELOPE_M = (0.38776, 0.18492, 0.15252)        # length, width, height
X_FRONT_M = 0.33446                             # foremost point (foam bumper)
X_REAR_M = -0.05330                             # rearmost point (rear gear case)
GROUND_CLEARANCE_M = 0.01469
STEER_MAX_RAD = 0.4189
STEER_RATE_MAX_RADPS = 3.2
FINAL_DRIVE_RATIO = 8.3474
MIN_TURN_RADIUS_M = 0.5791
LIDAR_POS_C = (0.21315, 0.0, 0.1099)            # scan origin: optical axis (measured), scan plane 109.9 mm
IMU_POS_C = (0.0603, -0.0051, 0.0780)           # centre of the motor controller board
REF_POINT_C = (0.12892, 0.0, 0.0)               # timing / checkpoint reference point: wheelbase midpoint on the floor
WHEELS = ("fl", "fr", "rl", "rr")
WHEEL_GROUND_C = {                              # wheel ground points at the static pose
    "fl": (0.25784, 0.07796, 0.0), "fr": (0.25784, -0.07796, 0.0),
    "rl": (0.0, 0.07798, 0.0), "rr": (0.0, -0.07798, 0.0),
}
TAIL_BOX_SIZE_M = (0.12, 0.12, 0.20)            # slow cars only
TAIL_BOX_CENTRE_C = (-0.034, 0.0, 0.1702)       # the obstacle model's measurement: bottom face 70.2 mm above the floor
TAIL_BOX_COL_BOTTOM_Z_M = 0.080                 # tail_box/col is the part of the box above this height
TAIL_BOX_MASS_KG = 0.05                         # on the child body slow<i>/tail_box/mass

# ---------------------------------------------------------------- ducts
DUCT_DIAMETER_M = 0.33
DUCT_RADIUS_M = 0.165
DUCT_LENGTH_M = 7.62                            # one 25 ft duct
DUCT_PIECES_FULL = 15
PIECE_LENGTH_NOMINAL_M = 0.508                  # 7.62 / 15
PIECE_LENGTH_RANGE_M = (0.35, 0.65)
GAP_MAX_M = 0.15
OPEN_WALL_RANGE_M = (0.6, 1.2)
TAPE_DEPTH_M = 1.5                              # a wheel point is beyond a tape when 0 < (p - mid) . outward <= this
OPEN_WALL_BEHIND_CLEAR_M = 3.0                  # no other stretch of the track 0.05 .. this far behind a tape
OPEN_WALL_BEHIND_NEAR_M = 0.05
WALL_CLASSES = ("orange_ribbed", "yellow_ribbed", "black_plain")

# ---------------------------------------------------------------- geom groups
GROUP_VISUAL = 0       # appearance only; no contact; not a LiDAR target
GROUP_COLLISION = 1    # takes part in contact; not a LiDAR target
GROUP_LIDAR = 2        # LiDAR targets of the track and of cars; no contact
GROUP_FLOOR = 3        # the floor plane and floor zone slabs; contact; the LiDAR's floor rule
GROUP_CLUTTER = 4      # out-of-track clutter proxies; no contact; LiDAR targets
GROUP_HIDDEN = 5       # staged objects and debug markers; no contact; never a LiDAR target
N_GROUPS = 6
LIDAR_GROUPMASK = (0, 0, 1, 0, 1, 0)            # what a beam can return from
LIDAR_FLOOR_GROUPMASK = (0, 0, 1, 1, 1, 0)      # the same plus the floor (the sensor model's shallow-angle floor rule)

# ---------------------------------------------------------------- contact bits
BIT_FLOOR, BIT_CAR, BIT_DUCT, BIT_BOX = 1, 2, 4, 8
CONTACT = {                                      # category -> (contype, conaffinity)
    "floor": (BIT_FLOOR, BIT_CAR | BIT_BOX),
    "car": (BIT_CAR, BIT_FLOOR | BIT_CAR | BIT_DUCT | BIT_BOX),
    "duct": (BIT_DUCT, BIT_CAR | BIT_BOX),
    "box": (BIT_BOX, BIT_FLOOR | BIT_CAR | BIT_DUCT | BIT_BOX),
    "none": (0, 0),
}
DUCT_DUCT_CONAFFINITY = BIT_CAR | BIT_BOX | BIT_DUCT   # the duct model may switch duct-duct contact on with this value


def collide(cat_a, cat_b):
    """MuJoCo's rule on the contract's categories: (contype_a & conaffinity_b) or (contype_b & conaffinity_a)."""
    ta, aa = CONTACT[cat_a]
    tb, ab = CONTACT[cat_b]
    return bool((ta & ab) or (tb & aa))


# ---------------------------------------------------------------- MJCF names
EGO_PREFIX = "car/"
ID_RE = r"[A-Za-z][A-Za-z0-9_]{0,23}"


def slow_prefix(i):
    return "slow%d/" % i


def duct_body(duct_id, k):
    return "duct/%s/%d" % (duct_id, k)


def duct_geom(duct_id, k, role):
    """role: 'col' (collision, group 1) or 'lidar' (LiDAR target, group 2)."""
    return "duct/%s/%d/%s" % (duct_id, k, role)


def duct_root_joint(duct_id, axis):
    """axis: 'x', 'y' or 'yaw' (the planar root of a pushable duct)."""
    return "duct/%s/%s" % (duct_id, axis)


def duct_bend_joint(duct_id, k):
    """The hinge between piece k - 1 and piece k (k >= 1)."""
    return "duct/%s/%d/bend" % (duct_id, k)


def box_body(box_id):
    return "box/%s" % box_id


def box_geom(box_id, role):
    return "box/%s/%s" % (box_id, role)


def clutter_geom(clutter_id):
    return "clutter/%s/lidar" % clutter_id


def zone_geom(patch_id, role):
    """role: 'slab' (group 3, patches with an edge height) or 'decal' (group 0)."""
    return "zone/%s/%s" % (patch_id, role)


CAR_BODIES = ("chassis",) + tuple("%s/wheel" % w for w in WHEELS)
CAR_JOINTS = ("root", "steer") + tuple("%s/spin" % w for w in WHEELS)
CAR_SITES = ("lidar", "imu", "ref")
CAR_TYRE_GEOMS = tuple("%s/tyre" % w for w in WHEELS)
CAR_LIDAR_GEOMS = ("lidar_head/lidar", "hull/lidar")          # slow cars add "tail_box/lidar"

# geom name pattern -> (group, contact category); a geom of a graded model must match exactly one pattern
_P = r"(?:car/|slow[0-9]/)"
GEOM_RULES = (
    (r"floor", GROUP_FLOOR, "floor"),
    (r"zone/%s/slab" % ID_RE, GROUP_FLOOR, "floor"),
    (r"zone/%s/decal" % ID_RE, GROUP_VISUAL, "none"),
    (r"tape/%s" % ID_RE, GROUP_VISUAL, "none"),
    (r"timing_line", GROUP_VISUAL, "none"),
    (r"checkpoint/%s" % ID_RE, GROUP_HIDDEN, "none"),
    (r"clutter/%s/lidar" % ID_RE, GROUP_CLUTTER, "none"),
    (r"duct/%s/[0-9]+/col" % ID_RE, GROUP_COLLISION, "duct"),
    (r"duct/%s/[0-9]+/lidar" % ID_RE, GROUP_LIDAR, "none"),
    (r"duct/%s/[0-9]+/vis" % ID_RE, GROUP_VISUAL, "none"),
    (r"box/%s/col" % ID_RE, GROUP_COLLISION, "box"),
    (r"box/%s/lidar" % ID_RE, GROUP_LIDAR, "none"),
    (r"box/%s/vis" % ID_RE, GROUP_VISUAL, "none"),
    (_P + r"(?:fl|fr|rl|rr)/tyre", GROUP_COLLISION, "car"),
    (_P + r"chassis/col[0-9]+", GROUP_COLLISION, "car"),
    (_P + r"tail_box/col", GROUP_COLLISION, "car"),
    (_P + r"(?:lidar_head|hull|tail_box)/lidar", GROUP_LIDAR, "none"),
    (_P + r"vis/[A-Za-z0-9_./-]+", GROUP_VISUAL, "none"),
)
GEOM_RULES_RE = tuple((re.compile("^" + p + "$"), g, c) for p, g, c in GEOM_RULES)


def geom_rule(name):
    """(group, contact category) the contract assigns to a geom name, or None if the name is not in the contract."""
    for rx, group, cat in GEOM_RULES_RE:
        if rx.match(name):
            return group, cat
    return None


# ---------------------------------------------------------------- world file numbers
WORLD_FLOAT_DECIMALS = 6            # every float of a world file is rounded to this when written
FLOOR_KINDS = ("painted_concrete", "sealed_concrete", "polished_concrete", "exhibition_carpet")
PATCH_KINDS = ("dust", "metal_cover", "carpet_tile")
EDGE_HEIGHT_MAX_M = 0.007
CLUTTER_MIN_CLEARANCE_M = 0.5
TRACK_WIDTH_RANGE_M = (1.0, 2.2)
MIN_RADIUS_M = 0.9
BOX_MAX_SIZE_M = (0.35, 0.32, 0.30)
BOX_MASS_RANGE_KG = (0.3, 0.6)
BOX_FREE_PASSAGE_M = 0.8
SLOW_SPEED_FACTOR_RANGE = (0.35, 0.50)
SLOW_GAP_KEEP_M = 0.8
FEATURES = ("duct_gaps", "open_wall", "floor_patch_dust", "floor_patch_metal_cover", "floor_patch_carpet_tile",
            "black_ducts", "narrow_section", "shared_wall", "outside_clutter")
NARROW_SECTION_M = 1.2              # a venue has the feature `narrow_section` when its minimum width is below this
