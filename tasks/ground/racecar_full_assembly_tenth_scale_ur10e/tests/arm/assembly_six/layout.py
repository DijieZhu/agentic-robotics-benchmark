"""Hidden per-episode layout of the six-servo arm's cell (CHAIN_PLAN 4.3 "Hidden": tray slot assignment, +-3 mm /
+-10 deg jitter, fixture +-10 mm, friction; the reference benchmark's own randomisation maps onto the same knobs:
its graded episodes differ by the spawn perturbation of the free parts, ours by the tray jitter).

Pure function of (seed, config, part names, rest poses): the live anchor and the agent see the same world for one
seed. Rest poses (`rest_poses`): every loose part (or pre-assembled compound) stands in a socketed tray in the
orientation it will have at its mate, up to the yaw jitter, so a top-down / side grasp needs no re-orientation in
hand (rung 1 "pre-oriented"; rung 3 draws any resting face):

    servo (rung 1)     standing on its connector end (servo -x face down): the servo's long axis is vertical, the
                       horn axis horizontal; the pocket it goes into faces up
    distal compound    fork down (forearm +x down), the elbow axis horizontal along the servo's horn axis: the
                       fork is lowered over the seated servo
    a tray socket of `socket_wall_mm` x `socket_height_mm` (a taller `stand_height_mm` for the compound) with
    `socket_oversize_mm` of play holds every part; the fixture carries the pre-assembled proximal half
"""

import math

import numpy as np

__all__ = ["DEFAULT_LAYOUT_CONFIG", "hidden_layout", "quat_from_rotmat", "rotz", "rest_pose_for"]

DEFAULT_LAYOUT_CONFIG = {
    "fixture_nominal_xy_m": [0.10, 0.0],
    "fixture_jitter_mm": 10.0,
    "fixture_top_z_m": 0.02,
    "base_yaw_on_fixture_rad": -1.5707963,   # yaw of the base part on the fixture: -pi/2 points the horizontal upper arm sideways (+y), where the Panda holds the fork compound most comfortably (IK margins measured, DESIGN.md)
    "tray_centres_m": {"L": [0.08, 0.32], "R": [0.08, -0.32]},
    "tray_size_m": [0.30, 0.22, 0.01],
    "socket_oversize_mm": 2.0,
    "socket_wall_mm": 2.0,
    "socket_height_mm": 8.0,
    "stand_height_mm": 40.0,                  # the pre-assembled distal compound stands fork-down in a taller socket
    "jitter_pos_mm": 3.0,
    "jitter_yaw_deg": 10.0,
    "friction_range": [0.8, 0.8],
    "panda_joint_noise_deg": 0.0,
    "servo_slot_x_rel_m": [-0.09, 0.0, 0.09],
    "compound_slot_rel_m": [0.0, 0.0],
    "drop_height_m": 0.003,
    "servo_rest": "on_connector_end",         # rung 1; rung 3: "random_face"
}


def rotz(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_from_rotmat(R):
    """wxyz quaternion of a rotation matrix (robust branch selection; same as the shared cell's)."""
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


def _frame_from_down_and_side(down_part, side_part, side_world=(0.0, 1.0, 0.0)):
    """Rotation (part -> world) that maps `down_part` (unit, part frame) to world -z and `side_part` (unit, part frame,
    orthogonal to it) to the horizontal world direction `side_world`."""
    d = np.asarray(down_part, float)
    s = np.asarray(side_part, float)
    sw = np.asarray(side_world, float)
    sw = sw - sw[2] * np.array([0.0, 0.0, 1.0])
    sw /= np.linalg.norm(sw)
    P = np.column_stack([d, s, np.cross(d, s)])
    W = np.column_stack([[0.0, 0.0, -1.0], sw, np.cross([0.0, 0.0, -1.0], sw)])
    return W @ P.T


def rest_pose_for(kind, points_part, down_part, side_part, side_world=(0.0, 1.0, 0.0)):
    """Rest description of a part: `kind` label, the rotation (down_part -> world -z, side_part -> `side_world`), the height
    of the part origin above the tray top (the lowest sampled point touches the tray), the footprint half-extents in the
    yawed tray frame and the footprint centre expressed in the part frame. `points_part`: vertices of the part (or
    compound) in the part frame (m)."""
    R = _frame_from_down_and_side(down_part, side_part, side_world)
    P = (R @ np.asarray(points_part, float).T).T
    z = -float(P[:, 2].min())
    lo, hi = P[:, :2].min(axis=0), P[:, :2].max(axis=0)
    centre_w = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])
    # the footprint half-extents are given in the frame whose +y is `side_world`: the socket is yawed with the part
    sw = _frame_from_down_and_side(down_part, side_part, side_world) @ np.asarray(side_part, float)
    yaw0 = math.atan2(sw[1], sw[0]) - math.pi / 2          # rotation that carries world +y onto side_world
    c, s_ = math.cos(-yaw0), math.sin(-yaw0)
    Q = (np.array([[c, -s_], [s_, c]]) @ P[:, :2].T).T      # footprint in the un-yawed frame
    lo2, hi2 = Q.min(axis=0), Q.max(axis=0)
    return {"kind": kind, "R": R, "z": z, "yaw0": yaw0, "footprint_half": [float((hi2[0] - lo2[0]) / 2), float((hi2[1] - lo2[1]) / 2)],
            "centre_part": (R.T @ centre_w).tolist()}


def hidden_layout(seed, config, part_names, rest_poses, servo_names=()):
    """Layout for one episode.

    `part_names`: the loose bodies of the cell (servos and compound roots); `rest_poses`: name -> rest description
    (`rest_pose_for`). Returns {"seed", "fixture": {"pos"}, "base_yaw_rad", "friction", "trays", "servo_tray",
    "parts": {name: {"pos", "quat", "yaw", "tray", "rest", "socket": {"centre_xy", "half_xy", "yaw", "height_m"}}},
    "panda_joint_noise_rad"}."""
    cfg = dict(DEFAULT_LAYOUT_CONFIG)
    cfg.update(config or {})
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    fj = cfg["fixture_jitter_mm"] * 1e-3
    fixture_xy = np.asarray(cfg["fixture_nominal_xy_m"], float) + rng.uniform(-fj, fj, 2)
    fr = cfg["friction_range"]
    friction = float(rng.uniform(fr[0], fr[1]))
    servo_tray = "R" if rng.integers(2) == 0 else "L"
    other_tray = "L" if servo_tray == "R" else "R"
    tray_top = float(cfg["tray_size_m"][2])
    servos = [n for n in part_names if n in set(servo_names)]
    others = [n for n in part_names if n not in set(servo_names)]
    slot_x = list(cfg["servo_slot_x_rel_m"])
    if len(servos) > len(slot_x):
        slot_x = [(-0.12 + 0.24 * i / max(1, len(servos) - 1)) for i in range(len(servos))]
    perm = rng.permutation(max(1, len(servos)))
    jp = cfg["jitter_pos_mm"] * 1e-3
    jy = math.radians(cfg["jitter_yaw_deg"])
    parts = {}

    def place(name, tray, rel_xy, height_m):
        rest = rest_poses[name]
        R0 = np.asarray(rest["R"], float)
        yaw = float(rng.uniform(-jy, jy))
        if rest["kind"].startswith("compound") and cfg["tray_centres_m"][tray][1] < 0.0:
            # a compound in the tray on the robot's -y side stands turned by 180 deg about the vertical, so that the side it is
            # approached from faces the robot's centreline on either tray (the horn mate accepts the yaw modulo 90 deg)
            yaw += math.pi
        R = rotz(yaw) @ R0
        centre = np.asarray(cfg["tray_centres_m"][tray], float) + np.asarray(rel_xy, float) + rng.uniform(-jp, jp, 2)
        c_part = np.asarray(rest["centre_part"], float)
        origin = np.array([centre[0], centre[1], 0.0]) - R @ c_part
        origin[2] = tray_top + rest["z"] + cfg["drop_height_m"]
        parts[name] = {
            "pos": [float(v) for v in origin], "quat": quat_from_rotmat(R), "yaw": yaw, "tray": tray, "rest": rest["kind"],
            "socket": {"centre_xy": [float(centre[0]), float(centre[1])], "half_xy": [float(v) for v in rest["footprint_half"]],
                       "yaw": yaw + float(rest.get("yaw0", 0.0)), "height_m": float(height_m)},
        }

    for i, name in enumerate(servos):
        place(name, servo_tray, [slot_x[int(perm[i])], 0.0], cfg["socket_height_mm"] * 1e-3)
    for name in others:
        place(name, other_tray, cfg["compound_slot_rel_m"], cfg["stand_height_mm"] * 1e-3)
    noise = math.radians(cfg["panda_joint_noise_deg"])
    pj = rng.uniform(-noise, noise, 7) if noise > 0 else np.zeros(7)
    return {
        "seed": int(seed), "fixture": {"pos": [float(fixture_xy[0]), float(fixture_xy[1]), 0.0]},
        "fixture_top_z_m": float(cfg["fixture_top_z_m"]), "base_yaw_rad": float(cfg["base_yaw_on_fixture_rad"]),
        "friction": friction, "trays": {"L": list(cfg["tray_centres_m"]["L"]), "R": list(cfg["tray_centres_m"]["R"])},
        "servo_tray": servo_tray, "parts": parts, "tray_size_m": list(cfg["tray_size_m"]),
        "panda_joint_noise_rad": [float(v) for v in pj],
        "config": {k: cfg[k] for k in ("jitter_pos_mm", "jitter_yaw_deg", "fixture_jitter_mm", "socket_oversize_mm", "friction_range", "socket_wall_mm")},
    }
