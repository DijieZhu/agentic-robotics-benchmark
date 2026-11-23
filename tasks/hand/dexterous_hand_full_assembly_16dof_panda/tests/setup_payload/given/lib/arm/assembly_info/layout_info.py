"""Per-episode draws of the information-tier assembly cell `assembly_info/cell_v1` (assembly.md 2.6, 4, 7.1; E-ASM-15).

PUBLIC CODE (shipped with the practice cell): every draw is a pure function of (episode seed, config, rig, kit). The
public code carries NO camera mount: a cell's nominal mounts come from its configuration (the hidden cell: the grader's
verify-only config) or, in a practice cell, from the practice kit's own rig file (`cell_rigs.json`: nominal mounts drawn
per practice seed, away from the hidden mounts and with no shared eye / target convention). The camera classes (size,
nominal field of view, depth) and the practice cell's per-episode spreads are public.

Per episode:
  * the loose-part layout of the six-servo cell (tray assignment, +-3 mm / +-10 deg part jitter, fixture +-10 mm): the servo
    tray holds three slots: the elbow servo WITH horn, one horn-less spare (catalogue id BUS-SERVO-12V-A-NH, a seat with it
    never latches) and one empty slot, in a random order; the elbow servo stands on its connector end with its horn facing
    one of the two sides (p 0.5 each); the forearm-to-gripper compound stands fork-down in its 40 mm stand on the other tray
  * the proximal posture: the base yawed on the fixture by the nominal +- 10 deg, the upper-arm beam elevation (shoulder
    pitch) nominal +- 6 deg (the nominal puts the elbow pocket's mouth straight up)
  * the cameras: each fixed camera's mount jittered around its cell's nominal mount, intrinsics (focal length, principal
    point, radial / tangential distortion) drawn around the camera class; the wrist camera jittered around its nominal
    mount on the hand; the depth units' seeds (spreads: `camera_draw` of the cell's configuration)
  * light intensities +-20 %, the F/T sensor bias (+-1 N / +-0.05 N m), the hand plate's mount (+-1 mm / +-0.5 deg)
"""

import copy
import math

import numpy as np

from ..assembly_six.layout import DEFAULT_LAYOUT_CONFIG as SIX_LAYOUT, quat_from_rotmat, rest_pose_for, rotz

__all__ = ["DEFAULT_INFO_LAYOUT", "draw_episode", "place_parts", "rot_about", "look_at_T", "CAMERA_NAMES", "MissingMount", "merge_config"]

CAMERA_NAMES = ("overview", "top", "oblique", "wrist", "inspect")

DEFAULT_INFO_LAYOUT = {
    "servo_slot_x_rel_m": [-0.09, 0.0, 0.09],
    "spare": True,
    "horn_flip_p": 0.5,
    "proximal_jitter_deg": {"base_yaw": 10.0, "shoulder_pitch": 6.0},
    # the camera CLASSES (public); a cell's nominal mounts are not part of the public code: `cameras` of the cell's config
    # (fixed cameras: eye / target / up in the cell frame; wrist: offset_m / z_m / tilt_deg on the hand) or the practice
    # kit's rig file supply them
    "cameras": {
        "overview": {"size": [848, 480], "hfov_deg": 70.0, "depth": True},
        "top": {"size": [848, 480], "hfov_deg": 70.0, "depth": True},
        "oblique": {"size": [848, 480], "hfov_deg": 70.0, "depth": True},
        # the inspection camera stands ON the bench: its height varies by its base only (2026-09-28 fix: with the fixed
        # cameras' +-30 mm z jitter about 30 % of the draws put it inside the bench top)
        "inspect": {"size": [640, 480], "hfov_deg": 60.0, "depth": True, "mount_dz_m": 0.002},
        "wrist": {"size": [640, 480], "hfov_deg": 80.0, "depth": True},
    },
    # per-episode spreads of the PRACTICE cell (the hidden cell's own spreads are part of its configuration)
    "camera_draw": {"mount_pos_m": 0.040, "mount_ang_deg": 4.0, "f_rel": 0.04, "c_rel": 0.025, "k1": [-0.10, 0.04], "k2": [-0.03, 0.03],
                    "p": 0.0015, "wrist_pos_m": 0.003, "wrist_ang_deg": 1.5},
    "lights": {"key": {"pos": [0.9, -0.8, 2.4], "target": [0.1, 0.0, 0.0], "diffuse": 0.62},
               "fill": {"pos": [-0.6, 1.4, 1.8], "target": [0.1, 0.0, 0.0], "diffuse": 0.30},
               "dome": 0.22, "jitter_rel": 0.20},
    "wrench_noise": {"sigma_f_N": 0.3, "sigma_t_Nm": 0.02, "bias_f_N": 1.0, "bias_t_Nm": 0.05},
    "hand_plate": {"jitter_mm": 1.0, "jitter_deg": 0.5},
}


class MissingMount(ValueError):
    """A camera without a nominal mount (neither the cell's config nor a practice rig supplies it)."""


def _rng(seed, label):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, sum(ord(c) * (i + 1) for i, c in enumerate(label)) & 0xFFFFFFFF])


def rot_about(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def look_at_T(eye, target, up=(0.0, 0.0, 1.0)):
    """world_T_cam in the OpenCV convention (+z forward, +y down)."""
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    up = np.asarray(up, float)
    if abs(float(z @ (up / np.linalg.norm(up)))) > 0.999:
        up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def _small_rot(r, ang_deg):
    ax = r.normal(0.0, 1.0, 3)
    ax /= max(1e-12, np.linalg.norm(ax))
    return rot_about(ax, math.radians(float(r.uniform(-ang_deg, ang_deg))))


def lens_invertible(w, h, fx, fy, cx, cy, k1, k2, p1, p2, max_margin_frac=0.25):
    """True when the lens model is invertible over the image: the fixed-point undistortion of every border pixel converges
    and the undistorted canvas stays within `max_margin_frac` of the image size on each side (version 2 defect fix: a
    practice draw k1 -0.085 / k2 -0.024 at the wrist camera's 80 deg made the undistortion diverge, the render layer asked for
    a 68 744 x 66 785 px canvas and OpenGL refused the framebuffer)."""
    n = 64
    us = np.concatenate([np.linspace(-0.5, w - 0.5, n), np.full(n, w - 0.5), np.linspace(-0.5, w - 0.5, n), np.full(n, -0.5)])
    vs = np.concatenate([np.full(n, -0.5), np.linspace(-0.5, h - 0.5, n), np.full(n, h - 0.5), np.linspace(-0.5, h - 0.5, n)])
    xd, yd = (us - cx) / fx, (vs - cy) / fy
    x, y = xd.copy(), yd.copy()
    for _ in range(60):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        if np.any(radial <= 0.2):
            return False
        xn = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / radial
        yn = (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / radial
        step = max(float(np.max(np.abs(xn - x))), float(np.max(np.abs(yn - y))))
        x, y = xn, yn
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))) or step > 1e-6:
        return False
    uu, vv = fx * x + cx, fy * y + cy
    return (-uu.min() < max_margin_frac * w and uu.max() - w < max_margin_frac * w and -vv.min() < max_margin_frac * h
            and vv.max() - h < max_margin_frac * h)


def _intrinsics(r, size, hfov_deg, d):
    w, h = int(size[0]), int(size[1])
    f0 = 0.5 * w / math.tan(math.radians(hfov_deg) / 2.0)
    fx = f0 * (1.0 + r.uniform(-d["f_rel"], d["f_rel"]))
    fy = f0 * (1.0 + r.uniform(-d["f_rel"], d["f_rel"]))
    cx = (w - 1) / 2.0 + r.uniform(-d["c_rel"], d["c_rel"]) * w
    cy = (h - 1) / 2.0 + r.uniform(-d["c_rel"], d["c_rel"]) * h
    k1 = r.uniform(*d["k1"])
    k2 = r.uniform(*d["k2"])
    p1 = r.uniform(-d["p"], d["p"])
    p2 = r.uniform(-d["p"], d["p"])
    for _ in range(200):                    # a lens that is not invertible over the image is redrawn (same stream)
        if lens_invertible(w, h, fx, fy, cx, cy, k1, k2, p1, p2):
            break
        k1 = r.uniform(*d["k1"])
        k2 = r.uniform(*d["k2"])
    return {"width": w, "height": h, "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy), "k1": float(k1), "k2": float(k2),
            "p1": float(p1), "p2": float(p2)}


def draw_cameras(seed, cfg, strict=True):
    """{name: {"intrinsics": {...}, "world_T_cam": 4x4 (fixed cameras) | "hand_T_cam": 4x4 (wrist), "depth": bool}}. A camera
    without a nominal mount raises MissingMount (strict) or is None (a cell without cameras, e.g. an agent's own twin)."""
    d = cfg["camera_draw"]
    out = {}
    for name in CAMERA_NAMES:
        c = cfg["cameras"][name]
        need = ("offset_m", "z_m", "tilt_deg") if name == "wrist" else ("eye", "target")
        if any(k not in c for k in need):
            if strict:
                raise MissingMount("camera %s has no nominal mount (the cell's config or the practice kit's cell_rigs.json supplies it)" % name)
            out[name] = None
            continue
        r = _rng(seed, "camera/" + name)
        intr = _intrinsics(r, c["size"], c["hfov_deg"], d)
        if name == "wrist":
            off, zc, tilt = float(c["offset_m"]), float(c["z_m"]), math.radians(float(c["tilt_deg"]))
            # optical axis in the hand frame: from the mount towards the flange axis, tilted by `tilt` from the approach (+z)
            zdir = np.array([-math.sin(tilt), 0.0, math.cos(tilt)])
            xdir = np.array([0.0, 1.0, 0.0])                 # image x along the closing direction
            ydir = np.cross(zdir, xdir)
            T = np.eye(4)
            T[:3, 0], T[:3, 1], T[:3, 2] = xdir, ydir, zdir
            T[:3, 3] = [off, 0.0, zc]
            J = np.eye(4)
            J[:3, :3] = _small_rot(r, d["wrist_ang_deg"])
            J[:3, 3] = r.uniform(-d["wrist_pos_m"], d["wrist_pos_m"], 3)
            out[name] = {"intrinsics": intr, "hand_T_cam": (J @ T).tolist(), "depth": bool(c.get("depth", True))}
        else:
            T = look_at_T(c["eye"], c["target"], c.get("up", (0.0, 0.0, 1.0)))
            J = np.eye(4)
            J[:3, :3] = _small_rot(r, d["mount_ang_deg"])
            T2 = T.copy()
            T2[:3, :3] = J[:3, :3] @ T[:3, :3]
            T2[:3, 3] = T[:3, 3] + r.uniform(-d["mount_pos_m"], d["mount_pos_m"], 3)
            if c.get("mount_dz_m") is not None:            # a camera standing on the bench: the z jitter is its own
                dz = float(c["mount_dz_m"])
                T2[2, 3] = T[2, 3] + (r.uniform(-dz, dz) if dz > 0 else 0.0)
            out[name] = {"intrinsics": intr, "world_T_cam": T2.tolist(), "depth": bool(c.get("depth", True))}
    return out


def merge_config(cfg=None, rig=None):
    """DEFAULT_INFO_LAYOUT updated by `cfg` (the cell's info_layout) and then by `rig` ({"cameras": {name: mount}, "camera_draw"?}):
    camera entries are merged per camera (a mount adds to the camera's class)."""
    c = copy.deepcopy(DEFAULT_INFO_LAYOUT)
    for src in (cfg or {}, rig or {}):
        for k, v in src.items():
            if k == "cameras" and isinstance(v, dict):
                for name, cam in v.items():
                    c["cameras"].setdefault(name, {}).update(copy.deepcopy(cam))
            elif isinstance(v, dict) and isinstance(c.get(k), dict):
                c[k].update(copy.deepcopy(v))
            else:
                c[k] = copy.deepcopy(v)
    return c


def draw_episode(seed, cfg=None, rig=None, strict=True):
    """The non-layout draws of one episode (cameras, lights, noise, proximal jitter, horn side, slot order). `rig` = a practice
    kit's rig for this seed (its nominal camera mounts); strict=False lets cameras without a mount be None."""
    c = merge_config(cfg, rig)
    r = _rng(seed, "episode")
    pj = c["proximal_jitter_deg"]
    prox = {"base_yaw_deg": float(r.uniform(-pj["base_yaw"], pj["base_yaw"])), "shoulder_pitch_deg": float(r.uniform(-pj["shoulder_pitch"], pj["shoulder_pitch"]))}
    horn_flip = bool(r.uniform() < float(c["horn_flip_p"]))
    slot_order = [int(v) for v in r.permutation(3)]          # slots of (elbow servo, spare, empty)
    spare_flip = bool(r.uniform() < 0.5)
    lj = float(c["lights"]["jitter_rel"])
    lights = {k: float(c["lights"][k]["diffuse"] if isinstance(c["lights"][k], dict) else c["lights"][k]) * float(r.uniform(1 - lj, 1 + lj))
              for k in ("key", "fill", "dome")}
    wn = c["wrench_noise"]
    bias = list(r.uniform(-wn["bias_f_N"], wn["bias_f_N"], 3)) + list(r.uniform(-wn["bias_t_Nm"], wn["bias_t_Nm"], 3))
    hp = c["hand_plate"]
    J = np.eye(4)
    J[:3, :3] = _small_rot(r, hp["jitter_deg"])
    J[:3, 3] = r.uniform(-hp["jitter_mm"], hp["jitter_mm"], 3) * 1e-3
    return {"seed": int(seed), "config": c, "proximal_jitter": prox, "horn_flip": horn_flip, "slot_order": slot_order, "spare_flip": spare_flip,
            "lights": lights, "wrench_bias": [float(v) for v in bias], "hand_plate_jitter": J.tolist(), "cameras": draw_cameras(seed, c, strict=strict)}


def place_parts(seed, cfg_layout, loose, rests, servo_names, episode):
    """The six-servo layout (`assembly_six.layout.hidden_layout` numbers and stream) with the info-tier servo tray: every
    servo of `servo_names` gets a slot from `episode["slot_order"]` (elbow servo first, then the spare; the remaining
    slot stays empty); a flipped servo stands turned 180 deg about the vertical (its horn facing the other side)."""
    cfg = dict(SIX_LAYOUT)
    cfg.update(cfg_layout or {})
    seed = int(seed)
    # the six-servo layout stream for seeds < 2**31 (practice seeds, recorded reference values); the hidden grader draws 63-bit
    # seeds, whose high bits enter the stream too
    rng = np.random.default_rng(seed) if 0 <= seed < (1 << 31) else np.random.default_rng([seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, 0x5A17])
    fj = cfg["fixture_jitter_mm"] * 1e-3
    fixture_xy = np.asarray(cfg["fixture_nominal_xy_m"], float) + rng.uniform(-fj, fj, 2)
    fr = cfg["friction_range"]
    friction = float(rng.uniform(fr[0], fr[1]))
    servo_tray = "R" if rng.integers(2) == 0 else "L"
    other_tray = "L" if servo_tray == "R" else "R"
    tray_top = float(cfg["tray_size_m"][2])
    jp = cfg["jitter_pos_mm"] * 1e-3
    jy = math.radians(cfg["jitter_yaw_deg"])
    slot_x = list(cfg.get("servo_slot_x_rel_m", DEFAULT_INFO_LAYOUT["servo_slot_x_rel_m"]))
    parts = {}

    def place(name, tray, rel_xy, height_m, flip=False):
        rest = rests[name]
        R0 = np.asarray(rest["R"], float)
        yaw = float(rng.uniform(-jy, jy))
        if rest["kind"].startswith("compound") and cfg["tray_centres_m"][tray][1] < 0.0:
            yaw += math.pi                   # six-servo rule: the compound on the robot's -y side stands turned 180 deg
        if flip:
            yaw += math.pi
        R = rotz(yaw) @ R0
        centre = np.asarray(cfg["tray_centres_m"][tray], float) + np.asarray(rel_xy, float) + rng.uniform(-jp, jp, 2)
        origin = np.array([centre[0], centre[1], 0.0]) - R @ np.asarray(rest["centre_part"], float)
        origin[2] = tray_top + rest["z"] + cfg["drop_height_m"]
        parts[name] = {"pos": [float(v) for v in origin], "quat": quat_from_rotmat(R), "yaw": yaw, "tray": tray, "rest": rest["kind"],
                       "flipped": bool(flip),
                       "socket": {"centre_xy": [float(centre[0]), float(centre[1])], "half_xy": [float(v) for v in rest["footprint_half"]],
                                  "yaw": yaw + float(rest.get("yaw0", 0.0)), "height_m": float(height_m)}}

    servos = [n for n in servo_names if n in loose or n in rests]
    for i, name in enumerate(servos):
        flip = episode["horn_flip"] if i == 0 else episode["spare_flip"]
        place(name, servo_tray, [slot_x[episode["slot_order"][i]], 0.0], cfg["socket_height_mm"] * 1e-3, flip=flip)
    for name in [n for n in loose if n not in servos]:
        place(name, other_tray, cfg["compound_slot_rel_m"], cfg["stand_height_mm"] * 1e-3)
    empty = [slot_x[episode["slot_order"][k]] for k in range(len(servos), 3)]
    return {
        "seed": int(seed), "fixture": {"pos": [float(fixture_xy[0]), float(fixture_xy[1]), 0.0]},
        "fixture_top_z_m": float(cfg["fixture_top_z_m"]), "friction": friction,
        "trays": {"L": list(cfg["tray_centres_m"]["L"]), "R": list(cfg["tray_centres_m"]["R"])}, "servo_tray": servo_tray, "parts": parts,
        "tray_size_m": list(cfg["tray_size_m"]), "panda_joint_noise_rad": [0.0] * 7, "empty_slots_rel_x_m": empty,
        "config": {k: cfg[k] for k in ("jitter_pos_mm", "jitter_yaw_deg", "fixture_jitter_mm", "socket_oversize_mm", "friction_range", "socket_wall_mm")},
    }


def rest_of(kind, corners_part, down_part, side_part, side_world):
    return rest_pose_for(kind, corners_part, down_part=down_part, side_part=side_part, side_world=side_world)
