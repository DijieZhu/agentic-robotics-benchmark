"""The cell's configuration: one JSON document (``tests/grader_config.json``)
deep-merged over ``DEFAULTS``. Every number of the world, the sensors, the
budgets, the safety rule and the score lives here, so a value can change
without touching code.

Blocks:
  world    physics and station: solver options, finger pads, parcel contact
           values, the belt's geometry (station: belt, lip, near rails,
           legs, solid frame), station hardware collision boxes, robot mounts (head-camera
           housing, the palm cameras), start posture, settle time, command
           speed limit, the tool-pose level, lines, entry clearance, gripper
           object detection
  cameras  exposure period, frame age, ring length, the four cameras
  sensors  RGB / depth noise, the motor torque readings' noise
  installation  the cameras as installed (grader-only: their errors against
           the drawing)
  budgets  program time, silence, frames, ticks per step, after-over caps
  safety   protective stop, lock-out, parcel damage
  score    the "readable" thresholds, the station's duty (the kinds and
           the label positions it turns)
  line     the line setting (grader-only): its duty (``line.score``,
           over ``score``), the indexing zone, its generator (``line.shifts``)
  shifts   the shift generator (belt-speed classes, the mailers' share,
           population ranges, the arrival gap, duration rule)
Keys outside these blocks belong to the grader's runner (verify).
"""

from __future__ import annotations

import copy
import json
import math

__all__ = ["DEFAULTS", "load_config", "deep_update", "cell_config", "look_at_xyaxes", "score_config",
           "START_POSTURE", "CAMERA_NAMES", "GRADER_ONLY_KEYS", "LABEL_POSITIONS"]

# the start posture ("park"): both palms 0.26 m above the belt top, 0.22 m
# either side of the column's plane, at x = 0.08 m, tool axes tilted 25
# degrees from straight down; every link stays behind the belt's near edge,
# so no parcel can touch the robot while it rides past. The right arm is the
# left arm's mirror image (same elbow angle, opposite angles elsewhere).
START_POSTURE = [-0.7842, 2.1067, 0.7271, 1.946, -1.9143, -1.54, -1.1825,
                 0.7842, -2.1067, -0.7271, 1.946, 1.9143, 1.54, 1.1825]

# station hardware near the arms as collision boxes (name, centre, half
# sizes; world frame). Not drawn by the cell's own renderer (the look draws
# the hardware). Fitted to the art layer's hardware; to be corrected from the
# re-fitted art when it changes.
HARDWARE_BOXES = [
    # the infeed camera: its body (down to its front face), its mounting plate and its drop rod
    ["infeed_camera_body", [0.38, -0.75, 1.7823], [0.045, 0.014, 0.0321]],
    ["infeed_camera_bracket", [0.38, -0.738, 1.82], [0.022, 0.022, 0.0025]],
    ["infeed_camera_rod", [0.38, -0.715, 2.125], [0.015, 0.015, 0.295]],
    # the belt's drive under the downstream end: the drum between its side plates and snub rollers, the motor's
    # guard (over the gearbox), the motor with its fins and fan cover, its terminal box
    ["belt_drive_unit", [0.38, 0.98, 0.7625], [0.217, 0.126, 0.0785]],
    ["belt_drive_motor_cover", [0.684, 0.975, 0.75], [0.062, 0.085, 0.074]],
    ["belt_drive_motor", [0.674, 1.18, 0.75], [0.064, 0.12, 0.0632]],
    ["belt_drive_motor_terminal", [0.674, 1.13, 0.825], [0.04, 0.045, 0.022]],
]

CAMERA_NAMES = ("infeed", "head", "wrist_left", "wrist_right")

# label positions at arrival (the world direction of the label face's normal
# when the parcel lands): the names of the duty sheet
LABEL_POSITIONS = ("top", "underneath", "downstream", "upstream", "near_side", "far_side")

DEFAULTS = {
    "world": {
        "solver": {"integrator": "implicitfast", "cone": "elliptic",
                   "impratio": 100.0, "iterations": 100, "ls_iterations": 50,
                   "noslip_iterations": 0, "tolerance": 1e-8},
        "lock_gripper_couplers": True,
        "finger_pads": {"soft": True, "solref": [0.005, 1.0], "condim": 4,
                        "friction": 1.8},
        "parcel_contact": {
            "rigid": {"solimp": [0.95, 0.99, 0.001, 0.5, 2.0], "solref": [0.005, 1.0]},
            "soft_geom": {"solimp": [0.6, 0.95, 0.008, 0.5, 2.0], "solref": [0.02, 0.6]},
            "soft_pad": {"solimp": [0.5, 0.9, 0.01, 0.5, 2.0], "solref": [0.03, 0.5]},
        },
        "station": {"belt_height": 0.90, "belt_center_x": 0.38, "belt_width": 0.44,
                    "visible_half_len": 1.3, "work_window_half": 0.42, "belt_friction": 0.9,
                    "lip_full_window": True, "lip_height": 0.03,
                    # the near rails upstream and downstream of the work window:
                    # their tops above the belt
                    "near_guide_up_height": 0.09, "near_guide_down_height": 0.09,
                    # the legs as drawn (two posts, a cap, a brace per leg)
                    "leg_style": "posts"},
        "hardware_boxes": [list(h) for h in HARDWARE_BOXES],
        "robot_mounts": {
            "head_camera_housing": {"body": "Base_Link", "centre": [0.078, 0.0, 1.3045],
                                    "half_size": [0.0866, 0.044, 0.0425]},
            # each wrist camera is a palm camera: flush in its gripper base's
            # palm between the finger roots (the gripper root body's frame is
            # the flange frame shifted 7 mm along the tool axis; the palm's top
            # is at z = 0.0918 m on the tool axis, |y| <= 0.010 m), its optical
            # centre 0.5 mm under the palm's surface and 4.2 mm off the tool
            # axis (the left on its root's +y side, the right on -y), looking
            # along the tool axis between the fingers; no collision geometry,
            # nothing above the palm (the look draws a small dark window there)
            "wrist_cameras": {
                "Left": {"pos": [0.0, 0.0042, 0.0913], "side_sign": 1.0, "pitch_deg": 0.0},
                "Right": {"pos": [0.0, -0.0042, 0.0913], "side_sign": -1.0, "pitch_deg": 0.0},
            },
        },
        "start_posture": list(START_POSTURE),
        "settle_s": 0.3,
        "target_speed_limit_rad_s": 6.0,
        # the tool-pose command level (parcelcell.toolpose): per arm, a target
        # pose of the tool centre point; the station controller solves the
        # arm's kinematics each tick
        "tool_pose": {"iterations": 6, "damping": 1.0e-4, "residual_damping": 0.1, "rot_length_m": 2.0,
                      "max_step_error_m": 0.5, "joint_margin_rad": 0.01, "speed_share": 0.95,
                      "bound_approach": 0.3,
                      "reached_m": 0.001, "reached_deg": 0.5, "stall_rel": 0.002, "stall_ticks": 5,
                      "unlatch_m": 0.005, "unlatch_rel": 0.05,
                      "gravity_compensation": True},
        "belt_speed_compensation": True,
        "solid_frame": True,
        "entry_y": -1.15,
        "scan_y": 0.95,
        "exit_y": 1.32,
        "entry_clearance_m": 0.06,
        # object detection (the real gripper's "object detected" status): the
        # fingers stopped short of the free opening at the command by more
        # than short_m and moved less than still_m over hold_s; the free
        # opening per command (0, 0.05, .., 1) is measured on this model
        "gripper_object": {"hold_s": 0.2, "short_m": 0.003, "still_m": 0.0005,
                           "free_opening_m": [0.0851, 0.0815, 0.0777, 0.0738, 0.0698, 0.0657, 0.0615, 0.0572, 0.0529, 0.0485, 0.044, 0.0395, 0.0349, 0.0303, 0.0256, 0.0209, 0.0162, 0.0115, 0.0067, 0.002, 0.0004]},
    },
    "cameras": {
        "exposure_ticks": 2,
        "delay_ticks": 3,
        "ring_states": 4,
        "list": {
            "infeed": {"model_camera": "inspect_cam", "size": [640, 480],
                       "fovy_deg": 62.0, "depth_range_m": [0.2, 4.0]},
            "head": {"model_camera": "head_cam", "size": [640, 400],
                     "fovy_deg": 66.0, "depth_range_m": [0.2, 4.0]},
            "wrist_left": {"model_camera": "Left_wrist_cam", "size": [424, 240],
                           "fovy_deg": 58.0, "depth_range_m": [0.07, 0.6],
                           "flange_site": "Left_flange"},
            "wrist_right": {"model_camera": "Right_wrist_cam", "size": [424, 240],
                            "fovy_deg": 58.0, "depth_range_m": [0.07, 0.6],
                            "flange_site": "Right_flange"},
        },
        # the installation's tolerances against the drawing (public): position (m) and angle (deg) of each camera in
        # its mount's frame, and the focal length against the data sheet's (relative)
        "installation_tolerance": {
            "infeed": {"position_m": 0.015, "angle_deg": 1.5, "frame": "world"},
            "head": {"position_m": 0.010, "angle_deg": 1.5, "frame": "the robot's base"},
            "wrist": {"position_m": 0.003, "angle_deg": 1.0, "frame": "its gripper"},
            "focal_rel": 0.01,
        },
    },
    # the motor torque readings (tau of the public state): white noise of tau_noise_rel x the joint's torque limit
    # (one sigma, a new draw every tick) and a slowly wandering offset of at most tau_offset_rel x the limit (periods
    # tau_offset_period_s), seeded by the shift; the privileged state and the safety watch read the true torques
    "sensors": {
        "rgb_noise_sigma": 2.0,
        "depth_noise_rel": 0.005,
        "depth_dropout": 0.01,
        "tau_noise_rel": 0.015,
        "tau_offset_rel": 0.01,
        "tau_offset_period_s": [40.0, 240.0],
    },
    # grader-only: the cameras as installed (each camera's fixed error against the drawing: d_pos_m, d_rotvec_rad in
    # its mount's frame, focal_scale); empty = as drawn. Never published (describe() gives the drawing).
    "installation": {"cameras": {}},
    # the renderer of the program's cameras, and the parcels' looks: "kit" =
    # the parcel kit (each parcel's own mesh, 512 px maps and waybill; the
    # graded and commissioning cells), "plain" = plain boxes (tooling).
    # backend: "classic" (MuJoCo's classic renderer) or "filament" (the
    # station's look: look_filament.py); look_file: the filament look's
    # numbers (relative to the package folder); render_dir: its render bundle
    # (null: $PARCELCELL_RENDER_DIR, else /opt/task-assets/render)
    # depth_twin: depth images also show the render bundle's near meshes of
    # the station's surroundings (look_depth.py; "auto": with the filament
    # backend, whose colour images draw them); kit_cache_dir: the shift
    # looks' cache (kitcache.py; null: $PARCELCELL_KIT_CACHE, else none)
    "look": {"backend": "filament", "parcel_visuals": "kit",
             "look_file": "look_config/look.json", "render_dir": None,
             "depth_twin": "auto", "kit_cache_dir": None},
    "budgets": {
        "program_time_factor": 1.0,
        "program_time_extra_s": 60.0,
        "silence_s": 20.0,
        "frames_per_shift_s": 4.0,
        "max_ticks_per_step": 50,
        "after_over_steps": 50,
        "after_over_requests": 200,
        "after_over_s": 60.0,
        "request_line_bytes": 4 * 1024 * 1024,
        "privileged_program_time_factor": 10.0,
        "start_s": 120.0,
    },
    "route_gate": {
        "forbidden_calls": ["torch.optim", ".backward(", "jax.grad"],
        "forbidden_extensions": [".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".h5", ".pb",
                                 ".pkl", ".npz"],
        "max_submission_mb": 60,
    },
    # the safety watch's final thresholds, each held for press_ticks ticks
    # (squeeze for squeeze_ticks): robot on structure, arm on arm or on its
    # own column, any robot part on the belt, the lock-out (max_stops stops
    # or one contact at damage_N) and the parcel's damage by squeeze
    "safety": {
        "structure_N": 60.0,
        "self_N": 60.0,
        "belt_N": 100.0,
        "press_ticks": 3,
        "stop_lock_s": 5.0,
        "damage_N": 400.0,
        "max_stops": 5,
        "squeeze_N": 300.0,
        "squeeze_ticks": 3,
    },
    "score": {
        "label_tilt_deg": 25.0,
        "max_angular_speed_rad_s": 2.0,
        "max_rel_speed_m_s": 0.10,
        # "rests on the belt" at the scan line: belt contact in at least one
        # tick of the last rest_window_s (the crossing tick included) and the
        # parcel's lowest point at most rest_clearance_max_m above the belt
        # surface; a soft mailer's belt contact flickers for a few ticks at a
        # time while it rides untouched, so one tick's flag never decides
        "rest_window_s": 0.5,
        "rest_clearance_max_m": 0.01,
        # the station's duty: a parcel whose kind is in ``kinds`` and whose
        # label position at arrival is in ``label_positions`` is the station's
        # work (class W); any other parcel whose label is not up passes
        # untouched to the exception lane (class X); any parcel that arrives
        # label up is class U
        "duty": {"kinds": ["rigid"], "label_positions": ["underneath", "downstream"]},
    },
    "shifts": {
        "task_id": "dual_arm_parcel_sorting_on_conveyor",
        "version": "v1",
        "belt_speed_classes": {"slow": [0.05, 0.13], "medium": [0.13, 0.21],
                               "fast": [0.21, 0.28]},
        # the share of soft mailers (one class: one parcel in five)
        "mix_classes": {"standard": 0.20},
        # parcels per full shift: within ``parcels_by_class`` for the shift's
        # belt-speed class (a faster belt brings more parcels in the same
        # time); ``parcels`` is the range over every class
        "parcels": [15, 25],
        "parcels_by_class": {"slow": [15, 21], "medium": [17, 24], "fast": [18, 25]},
        "bench_parcels": [1, 3],
        "first_arrival_s": [1.0, 3.0],
        "gap_margin_m": 0.10,
        # the arrival gap as BELT DISTANCE between two consecutive arrivals,
        # one rule for every pair whatever their classes: half the sum of the
        # two footprint diagonals + the entry's clearance + gap_margin_m + a
        # random part (exponential with mean gap_extra_m, capped at
        # gap_extra_cap_m; bench shifts: uniform in bench_gap_extra_m) + the
        # minimum arrival_gap_min (t_s seconds of belt travel + d_m metres);
        # the time between the arrivals is that distance over the belt speed
        "gap_extra": "exponential",
        "gap_extra_m": 0.6,
        "gap_extra_cap_m": 2.4,
        "bench_gap_extra_m": [1.5, 2.5],
        "arrival_gap_min": {"t_s": 10.0, "d_m": 0.20},
        "carton_size_m": [[0.16, 0.30], [0.16, 0.30], [0.11, 0.18]],
        "mailer_size_m": [[0.16, 0.30], [0.14, 0.26], [0.05, 0.12]],
        "mass_kg": [0.3, 3.0],
        "friction": [0.4, 0.75],
        "com_offset_max_m": 0.0,
        "label_uv_max": 0.55,
        "mailer_label_faces": [0, 1],
        "near_face_gap_m": [0.0, 0.01],
        "yaw_noise_deg": 8.0,
        "label_face_weights": "area",
        # every full shift lasts duration_s, every bench shift
        # bench_duration_s, whatever its belt speed and arrivals (nothing on
        # the wire bounds the last arrival or the parcel count); the
        # generator keeps only draws whose natural end (last arrival + the
        # belt's travel from the entry to the scan line + tail_s) is at most
        # that, and draws again otherwise
        "tail_s": 40.0,
        "duration_s": 600.0,
        "bench_duration_s": 300.0,
    },
}

# keys a public copy of the configuration (cell_config.json) leaves out
GRADER_ONLY_KEYS = ("shifts", "line", "look", "installation", "runner_account", "socket_sample_s",
                    "privileged_program_time_factor", "privileged_program_time_s")


def deep_update(base: dict, upd: dict | None) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(source=None) -> dict:
    """DEFAULTS deep-merged with ``source``: None, a dict, or a JSON path."""
    cfg = copy.deepcopy(DEFAULTS)
    if source is None:
        return cfg
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        with open(source) as fh:
            source = json.load(fh)
    deep_update(cfg, source)
    _check(cfg)
    return cfg


def _check(cfg: dict) -> None:
    w = cfg["world"]
    if len(w["start_posture"]) != 14:
        raise ValueError("world.start_posture needs 14 values")
    for k in ("settle_s", "target_speed_limit_rad_s"):
        if not (isinstance(w[k], (int, float)) and math.isfinite(w[k]) and w[k] >= 0):
            raise ValueError("world.%s must be a finite number >= 0" % k)
    c = cfg["cameras"]
    if int(c["exposure_ticks"]) < 1 or int(c["delay_ticks"]) < 0:
        raise ValueError("cameras: exposure_ticks >= 1, delay_ticks >= 0")
    need = (int(c["delay_ticks"]) + int(c["exposure_ticks"]) - 1) // int(c["exposure_ticks"]) + 2
    if int(c["ring_states"]) < need:
        raise ValueError("cameras.ring_states must be at least %d" % need)
    if set(c["list"]) != set(CAMERA_NAMES):
        raise ValueError("cameras.list must name %s" % ", ".join(CAMERA_NAMES))
    duties = [("score.duty", cfg["score"].get("duty") or {})]
    if isinstance(cfg.get("line"), dict):
        duties.append(("line.score.duty", (cfg["line"].get("score") or {}).get("duty") or {}))
        z = cfg["line"].get("indexing_zone_y")
        if not (isinstance(z, list) and len(z) == 2 and float(z[0]) < float(z[1])):
            raise ValueError("line.indexing_zone_y must be [lo, hi] with lo < hi")
    for name, duty in duties:
        for pos in duty.get("label_positions", ()):
            if pos not in LABEL_POSITIONS:
                raise ValueError("%s.label_positions: unknown position %r (%s)" % (name, pos, ", ".join(LABEL_POSITIONS)))
        for k in duty.get("kinds", ()):
            if k not in ("rigid", "soft"):
                raise ValueError("%s.kinds: rigid and / or soft" % name)


def score_config(cfg: dict, shift) -> dict:
    """The score block a shift is judged by: ``score``, with the line setting's overrides (``line.score``) for a
    shift fed on settle (``Shift.setting``)."""
    sc = copy.deepcopy(cfg["score"])
    if getattr(shift, "setting", None):
        if not isinstance(cfg.get("line"), dict):
            raise ValueError("a line shift needs the configuration's line block")
        deep_update(sc, cfg["line"].get("score") or {})
    return sc


def look_at_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> list:
    """MuJoCo ``xyaxes`` of a camera at ``pos`` looking at ``target`` (its image's up as near ``up`` as the view
    allows): x = the image's right, y = the image's up; the camera looks along -z."""
    import numpy as np
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return [round(float(v), 9) for v in list(r) + list(u)]


def cell_config(cfg: dict):
    """The simulator's ``CellConfig`` for this configuration."""
    from .cell import CellConfig
    from .sensors import CameraSpec
    from .station import StationConfig
    w = cfg["world"]
    g = w["station"]
    hw = tuple((str(n), tuple(float(v) for v in p), tuple(float(v) for v in hs))
               for n, p, hs in (w.get("hardware_boxes") or ()))
    st = StationConfig(belt_height=float(g["belt_height"]), belt_center_x=float(g["belt_center_x"]),
                       belt_width=float(g["belt_width"]), visible_half_len=float(g["visible_half_len"]),
                       work_window_half=float(g["work_window_half"]),
                       belt_friction=float(g["belt_friction"]),
                       entry_y=float(w["entry_y"]), exit_y=float(w["exit_y"]),
                       lip_full_window=bool(g.get("lip_full_window", False)),
                       lip_height=float(g.get("lip_height", 0.025)),
                       near_guide_up_height=float(g.get("near_guide_up_height", 0.09)),
                       near_guide_down_height=float(g.get("near_guide_down_height", 0.09)),
                       leg_style=str(g.get("leg_style", "block")),
                       solid_frame=bool(w["solid_frame"]),
                       frame_probes=not bool(w["solid_frame"]),
                       hardware=hw)
    cams = tuple(CameraSpec(c["model_camera"], int(c["size"][0]), int(c["size"][1]),
                            float(c["fovy_deg"]), tuple(float(v) for v in c["depth_range_m"]))
                 for _, c in sorted(cfg["cameras"]["list"].items()))
    fp = w["finger_pads"]
    return CellConfig(
        station=st, n_slots=0,
        lines=(("scan", float(w["scan_y"])),),
        cameras=cams,
        target_slew_rad_s=float(w["target_speed_limit_rad_s"]) or None,
        soft_finger_pads=bool(fp.get("soft", True)),
        pad_solref=float(fp["solref"][0]), pad_condim=int(fp.get("condim", 4)),
        pad_friction=float(fp["friction"]),
        finger_pad={"solref": [float(v) for v in fp["solref"]], "condim": int(fp.get("condim", 4)),
                    "friction": float(fp["friction"])},
        impratio=float(w["solver"].get("impratio", 100.0)),
        noslip_iterations=int(w["solver"].get("noslip_iterations", 0)),
        lock_couplers=bool(w.get("lock_gripper_couplers", True)),
        robot_mounts=json.loads(json.dumps(w.get("robot_mounts") or {})),
        belt_speed_compensation=bool(w["belt_speed_compensation"]),
        home=tuple(float(v) for v in w["start_posture"]),
        settle_s=float(w["settle_s"]),
        entry_clearance=float(w["entry_clearance_m"]),
        solver=dict(w["solver"]),
        parcel_contact=json.loads(json.dumps(w["parcel_contact"])),
    )
