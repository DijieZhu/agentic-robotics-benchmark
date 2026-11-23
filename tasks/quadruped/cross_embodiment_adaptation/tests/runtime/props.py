"""Per-site physical properties of the loaded Go1 and the two fittings.

Every site (and every commissioning-pad episode) draws its own values from the ranges below, which the
instruction and the datasheet state. Nothing here is sent to a program: a program learns a site's values
from what its robot measures there (joint angles under load, gravity, odometry, the two cameras).
"""
from dataclasses import dataclass

import numpy as np

RANGES = {
    "payload_kg": (2.50, 4.50),         # total mass carried: the 1.77 kg of equipment plus the instrument pack
    "ballast_x_m": (.05, .28),          # fore-aft place of the instrument pack, trunk frame
    "servo_gain": (.85, 1.15),          # per joint: stiffness relative to the datasheet's 100 N m/rad
    "torque_scale": (.85, 1.00),        # per site: every joint's torque limit relative to the datasheet
    "friction": (.45, .90),             # sliding friction of every walkable surface
    "duct_clearance_m": (.410, .445),   # underside of the suspended duct above the floor
    "guide_rise_m": (.070, .112),       # crest of the deck's guides above the deck
    "odom_yaw_drift_deg_s": (-.80, .80),  # leg odometry: a constant yaw-rate error
    "odom_scale": (.94, 1.06),          # leg odometry: reported distance over true distance
    "uav_scale": (.97, 1.03),           # survey: reported UAV displacement over true displacement
}
# Sensor noise (one standard deviation per reading) and per-site constant biases (uniform within +/- the bound).
ENCODER_NOISE_RAD, ENCODER_BIAS_RAD = .002, .008
GYRO_NOISE_RAD_S, GYRO_BIAS_RAD_S = .01, .015
GRAVITY_NOISE = .01
SADDLE_KG = 1.77          # the fixed equipment: lidar head, mast, compute unit, guard frame, profiler and bracket


@dataclass(frozen=True)
class Props:
    payload_kg: float
    ballast_x_m: float
    servo_gain: tuple
    torque_scale: float
    friction: float
    duct_clearance_m: float
    guide_rise_m: float
    odom_yaw_drift_deg_s: float
    odom_scale: float
    uav_scale: float

    def public(self):
        return {**self.__dict__, "servo_gain": list(self.servo_gain)}


def draw(seed, stream=3703):
    """A site's properties: an independent stream of the site's seed bits (see articulated.layout)."""
    rng = np.random.default_rng([int(seed) & ((1 << 63)-1), stream])
    u = lambda key: float(rng.uniform(*RANGES[key]))  # noqa: E731
    return Props(payload_kg=u("payload_kg"), ballast_x_m=u("ballast_x_m"),
                 servo_gain=tuple(float(v) for v in rng.uniform(*RANGES["servo_gain"], size=12)),
                 torque_scale=u("torque_scale"), friction=u("friction"),
                 duct_clearance_m=u("duct_clearance_m"), guide_rise_m=u("guide_rise_m"),
                 odom_yaw_drift_deg_s=u("odom_yaw_drift_deg_s"), odom_scale=u("odom_scale"),
                 uav_scale=u("uav_scale"))
