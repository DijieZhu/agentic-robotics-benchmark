"""Hidden per-episode layout of the assembly cell (CHAIN_PLAN 4.3 "Hidden": tray slot assignment,
+-3 mm / +-10 deg jitter, fixture +-10 mm, friction; rungs widen the family through `config`).

Pure function of (seed, config, part names): the same seed always produces the same layout, so the
grader's reference run and the agent's episode see the same world. Poses are
the INITIAL poses before the grader's settle phase (parts are dropped 3 mm and settled 0.6 s).

Resting orientations (rung 1 "pre-oriented"): servos upright (horn up, cable towards -y at yaw 0),
shoulder on its horn plate (z up), upper arm on its +x face (beam on top, length along the tray's x).
"""

import math

import numpy as np

__all__ = ["DEFAULT_LAYOUT_CONFIG", "hidden_layout", "quat_from_rotmat", "rotz", "REST_ROTATIONS"]

DEFAULT_LAYOUT_CONFIG = {
    "fixture_nominal_xy_m": [0.10, 0.0],
    "fixture_jitter_mm": 10.0,
    "fixture_top_z_m": 0.02,
    "tray_centres_m": {"L": [0.08, 0.32], "R": [0.08, -0.32]},
    "tray_size_m": [0.30, 0.22, 0.01],
    "socket_oversize_mm": 2.0,
    "socket_wall_mm": 2.0,
    "socket_height_mm": 8.0,
    "jitter_pos_mm": 3.0,
    "jitter_yaw_deg": 10.0,
    "friction_range": [0.8, 0.8],
    "panda_joint_noise_deg": 0.0,
    "servo_slot_x_rel_m": [-0.09, 0.0, 0.09],
    # rung-1 pre-orientation of the servos: upright (horn up) for vertical pockets, lying on the -x face in a
    # cradle for horizontal pockets (a servo grasped top-down keeps its axis along the fingers, so a servo
    # for a horizontal pocket must lie down); by default a servo lies down
    # a horizontal pocket's seated servo has its x axis up (pocket A of a dual joint) or down (pocket B);
    # a top-down grasp keeps the servo's x along the fingers, so the tray holds one of each
    "servo_rest": {"S1": "upright", "S2A": "side", "S2B": "side_flipped", "S3A": "side", "S3B": "side_flipped"},
    "servo_rest_default": "side",
    "cradle_height_mm": 7.3,
    "link_slots_rel_m": {"upper_arm": [0.0, -0.038], "shoulder": [0.0, 0.075], "forearm": [0.0, -0.038],
                         "wrist": [0.09, 0.075], "gripper_carrier": [-0.09, 0.075], "jaw_fixed": [0.11, -0.06], "jaw_moving": [-0.11, -0.06]},
    "drop_height_m": 0.003,
}

# part-frame axes expressed in the world at yaw 0 (columns = world directions of part x, y, z)
REST_ROTATIONS = {
    "servo": np.eye(3),                                                     # horn up
    "servo_side": np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),   # -x face down (x up), axis along +x world
    "servo_side_flipped": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),   # +x face down (x down), axis along +x world
    "shoulder": np.eye(3),                                                  # plate down
    # +x face down, length (part y) along -x world: the arm points back towards the robot, which is the
    # direction it is mounted in (horizontal, hand vertical, no yaw of the hand on the way)
    "upper_arm": np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]),
    "forearm": np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]),
    "wrist": np.eye(3),
    "gripper_carrier": np.eye(3),
    "jaw_fixed": np.eye(3),
    "jaw_moving": np.eye(3),
}
# world-z height of the part frame origin above the tray top in the rest orientation (m) and the
# in-plane offset (part frame, m) that puts the footprint centre on the slot centre
REST_OFFSETS = {
    "servo": {"z": 0.0, "centre_part": [0.0, -0.002, 0.0], "footprint_half": [0.0123, 0.0175]},
    "servo_side": {"z": 0.019, "centre_part": [0.0, -0.002, 0.023], "footprint_half": [0.023, 0.0175]},
    "servo_side_flipped": {"z": 0.019, "centre_part": [0.0, -0.002, 0.023], "footprint_half": [0.023, 0.0175]},
    "shoulder": {"z": 0.0, "centre_part": [0.0, -0.003, 0.0], "footprint_half": [0.065, 0.022]},
    "upper_arm": {"z": 0.019, "centre_part": [0.0, 0.105, 0.0], "footprint_half": [0.1265, 0.067]},
    "forearm": {"z": 0.019, "centre_part": [0.0, 0.082, 0.0], "footprint_half": [0.101, 0.046]},
    "wrist": {"z": 0.0598, "centre_part": [0.0, -0.0086, 0.0], "footprint_half": [0.025, 0.030]},
    "gripper_carrier": {"z": 0.0, "centre_part": [0.0175, 0.0, 0.0], "footprint_half": [0.0305, 0.021]},
    "jaw_fixed": {"z": 0.0, "centre_part": [0.0, 0.0088, 0.0], "footprint_half": [0.0195, 0.0285]},
    "jaw_moving": {"z": 0.003, "centre_part": [0.0, 0.0083, 0.0], "footprint_half": [0.0135, 0.022]},
}


def rotz(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_from_rotmat(R):
    """wxyz quaternion of a rotation matrix (robust branch selection)."""
    m = np.asarray(R, float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, float)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def _kind(name, manifest_parts, cfg=None):
    p = manifest_parts.get(name, {})
    if p.get("kind") == "servo":
        cfg = cfg or DEFAULT_LAYOUT_CONFIG
        rest = cfg.get("servo_rest", {}).get(name, cfg.get("servo_rest_default", "side"))
        return {"upright": "servo", "side": "servo_side", "side_flipped": "servo_side_flipped"}[rest]
    return name


def hidden_layout(seed, config, part_names, manifest_parts):
    """Layout for one episode.

    `part_names`: the loose parts in the cell (the fixed base excluded); `manifest_parts`: name ->
    manifest entry. Returns {"seed", "fixture": {"pos"}, "friction", "trays": {"L": centre, "R": centre},
    "parts": {name: {"pos", "quat", "yaw", "tray", "socket": {"centre_xy", "half_xy", "yaw"}}},
    "panda_joint_noise_rad": [7]}.
    """
    cfg = dict(DEFAULT_LAYOUT_CONFIG)
    cfg.update(config or {})
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    fj = cfg["fixture_jitter_mm"] * 1e-3
    fixture_xy = np.asarray(cfg["fixture_nominal_xy_m"], float) + rng.uniform(-fj, fj, 2)
    fr = cfg["friction_range"]
    friction = float(rng.uniform(fr[0], fr[1]))
    servo_tray = "R" if rng.integers(2) == 0 else "L"
    link_tray = "L" if servo_tray == "R" else "R"
    tray_top = float(cfg["tray_size_m"][2])
    servos = [n for n in part_names if manifest_parts[n]["kind"] == "servo"]
    links = [n for n in part_names if manifest_parts[n]["kind"] != "servo"]
    slot_x = list(cfg["servo_slot_x_rel_m"])
    if len(servos) > len(slot_x):
        slot_x = [(-0.12 + 0.24 * i / max(1, len(servos) - 1)) for i in range(len(servos))]
    perm = rng.permutation(len(servos))
    parts = {}
    jp = cfg["jitter_pos_mm"] * 1e-3
    jy = math.radians(cfg["jitter_yaw_deg"])

    def place(name, tray, rel_xy):
        kind = _kind(name, manifest_parts, cfg)
        R0 = REST_ROTATIONS[kind]
        off = REST_OFFSETS[kind]
        yaw = float(rng.uniform(-jy, jy))
        R = rotz(yaw) @ R0
        centre = np.asarray(cfg["tray_centres_m"][tray], float) + np.asarray(rel_xy, float) + rng.uniform(-jp, jp, 2)
        c_part = np.asarray(off["centre_part"], float)
        origin = np.array([centre[0], centre[1], tray_top + off["z"] + cfg["drop_height_m"]]) - R @ c_part
        origin[2] = tray_top + off["z"] + cfg["drop_height_m"]
        cradle = None
        if kind in ("servo_side", "servo_side_flipped"):
            # block under the body (servo z 2..32, y +-10) so the body face and the ears rest together
            cradle = {"centre_part": [0.0, 0.0, 0.017], "half_xy_part": [0.015, 0.010], "height_m": cfg["cradle_height_mm"] * 1e-3}
        parts[name] = {
            "pos": [float(v) for v in origin], "quat": quat_from_rotmat(R), "yaw": yaw, "tray": tray,
            "rest": kind, "cradle": cradle,
            "socket": {"centre_xy": [float(centre[0]), float(centre[1])],
                       "half_xy": [float(v) for v in off["footprint_half"]], "yaw": yaw},
        }

    for i, name in enumerate(servos):
        place(name, servo_tray, [slot_x[int(perm[i])], 0.0])
    for name in links:
        rel = cfg["link_slots_rel_m"].get(name, [0.0, 0.0])
        place(name, link_tray, rel)
    noise = math.radians(cfg["panda_joint_noise_deg"])
    pj = rng.uniform(-noise, noise, 7) if noise > 0 else np.zeros(7)
    return {
        "seed": int(seed), "fixture": {"pos": [float(fixture_xy[0]), float(fixture_xy[1]), 0.0]},
        "fixture_top_z_m": float(cfg["fixture_top_z_m"]),
        "friction": friction, "trays": {"L": list(cfg["tray_centres_m"]["L"]), "R": list(cfg["tray_centres_m"]["R"])},
        "servo_tray": servo_tray, "parts": parts,
        "panda_joint_noise_rad": [float(v) for v in pj],
        "config": {k: cfg[k] for k in ("jitter_pos_mm", "jitter_yaw_deg", "fixture_jitter_mm", "socket_oversize_mm", "friction_range")},
    }
