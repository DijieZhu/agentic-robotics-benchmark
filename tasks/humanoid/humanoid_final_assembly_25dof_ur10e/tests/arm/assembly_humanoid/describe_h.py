"""describe()'s public description and its FIELD TABLE.

New file of this task. describe() gives what the design gives (the robot, the control, the cameras' nominal mounts and
their draw bounds, the fastening rules and the measured windows by seat class, how each module is presented and how far
its fixture's place is drawn, the seven datum boards, the budgets, the safety numbers, the scoring) and withholds what
the design hides (the mating graph and every design pose and phase, the collision pieces, the bench plan, the
episode's draws: the layout, the clamp repeat and the camera mount and lenses, the module poses, the coupling's states,
the reference program and the token). EVERY value here comes from the configuration (cell_config.json), the fixture
design of record (fixtures.json: nominal places, boards) or the coupling's capture table: nothing depends on the
episode's draw, so describe() is the same for every seed (the test proves it).

The agent-visible SENTENCES are not written here (the texts' templates hold them): this module gives
structured fields; the few strings it carries are the cell's own (the server's robot strings, the coupling's seat-class words
of coupling_h.CLASS_WORDS, the fixture design's own draw texts) and are listed as such in FIELD_TABLE.

the camera terms that never reach a frame are not given: neither camera's
distortion ranges (k1, k2, the tangential terms) and neither the wrist camera's mount tolerance nor its lens tolerance.
The render path (render_iface.GivenLayerBackend._camera, the only camera input of the given layer's worker) takes the
overhead camera's drawn pose and its four pinhole terms fx, fy, cx, cy, and the wrist camera at the model camera of the
hand with the nominal field of view (no draw); the worker's camera dict carries no distortion term and the library
renders raw pinhole views. The overhead camera's mount draw and its focal and centre draws do reach its frames: kept.

FIELD_TABLE: {key path: {"unit", "frame", "source", "design": "gives" | "gives (wording: the texts)"}}; WITHHELD: what the
design withholds and how the tests prove it is absent; OBSERVATION_TABLE and EVENT_TABLE: get_state()'s fields and
every event string step() can return.
"""
import copy
import math

import numpy as np

__all__ = ["public_description", "FIELD_TABLE", "WITHHELD", "OBSERVATION_TABLE", "EVENT_TABLE", "EVENT_PATTERNS",
           "key_paths", "PRESENTED"]

# how each unit is presented (fixtures.json): unit -> the fixture that carries it
PRESENTED = {"base": "stand", "leg_R": "leg_station_R", "leg_L": "leg_station_L", "arm_R": "arm_rack_R",
             "arm_L": "arm_rack_L", "head": "head_nest"}
# the layout draw of each fixture: (configuration key of the xy amplitude, of the yaw amplitude)
DRAW_KEYS = {"stand": ("stand_mm", "stand_yaw_deg"), "leg_station_R": ("leg_rack_mm", "leg_rack_yaw_deg"),
             "leg_station_L": ("leg_rack_mm", "leg_rack_yaw_deg"), "arm_rack_R": ("arm_racks_mm", "arm_racks_yaw_deg"),
             "arm_rack_L": ("arm_racks_mm", "arm_racks_yaw_deg"), "head_nest": ("head_nest_mm", "head_nest_yaw_deg")}
BOARD_HOSTS = {"stand_board": "stand", "leg_rack_board": "leg_bench", "arm_rack_R_board": "arm_rack_R",
               "arm_rack_L_board": "arm_rack_L", "head_nest_board": "head_nest", "tool_board": "rack_table",
               "floor_board": "floor_board"}
FIXED_HOSTS = ("rack_table", "floor_board")     # not drawn (fixtures_h: the rack table and the floor board are not drawn)


def _r(v, nd=6):
    return round(float(v), nd)


def _yaw_deg(T):
    T = np.asarray(T, float)
    return math.degrees(math.atan2(T[1, 0], T[0, 0]))


def _cameras(cfg, robot_wrist):
    C = cfg["cameras"]
    D = C["camera_draw"]
    out = {}
    for n in C["agent_cameras"]:
        w, h = [int(v) for v in C["size"][n]]
        mt = C["mounts"][n]
        if n == "top":
            vfov = float(mt["fovy_deg"])
            hfov = 2.0 * math.degrees(math.atan(math.tan(math.radians(vfov) / 2.0) * w / h))
            mount = {"nominal_lens_front_cell_m": [_r(v) for v in mt["pos"]], "looking": "straight down",
                     "image_top_towards_cell": [_r(v) for v in mt["up"]],
                     "mount_draw": {"position_mm_per_axis": _r(1e3 * float(D["mount_pos_m"]), 3),
                                    "rotation_deg_about_any_axis": _r(D["mount_ang_deg"], 3)}}
        else:
            hfov = float(mt["hfov_deg"])
            vfov = float(mt["fovy_deg"])
            mount = {"on": "the hand", "lens_in_flange_m": [_r(v) for v in robot_wrist["lens_in_flange_m"]],
                     "aim_point_in_flange_m": [_r(v) for v in robot_wrist["aim_in_flange_m"]]}
            # no mount draw for the wrist camera (it is rendered at the model camera of the hand)
        # no depth range is listed: the configuration's range is applied to no depth
        # image (render_iface.py: every finite reading above 0 and below 65.0 m, in 1 mm steps; 0 for no reading)
        out[n] = {"resolution": [w, h], "rgb": True, "depth": bool(C["depth"][n]),
                  "nominal_hfov_deg": _r(hfov, 4), "nominal_vfov_deg": _r(vfov, 4),
                  "depth_unit_m": 0.001, "depth_invalid_value": 0,
                  "mount": mount}
        if n == "top":      # the overhead camera's focal and centre draws reach its frames; no distortion term does
            out[n]["intrinsics_draw"] = {"focal_rel": _r(D["f_rel"]), "centre_rel": _r(D["c_rel"])}
    return out


def _fastening(cfg):
    from . import coupling_h as CH
    ct = CH.capture_table(cfg)
    C = cfg["coupling"]
    return {"seat_classes": copy.deepcopy(ct["rows"]), "table_units": copy.deepcopy(ct["units"]),
            "interface_class_of_unit": dict(C["interfaces"]), "weights": {k: float(v) for k, v in C["weights"].items()},
            "total_weight": float(sum(float(v) for v in C["weights"].values()))}


def _items(cfg, des):
    """how each unit is presented: its fixture, the fixture's place ROUNDED to 0.01 m (the precedents' 'at about (x, y)
    m'; the exact rest pose of a module in its fixture is the hidden bench plan), and the draw's bounds"""
    J = cfg["fixtures"]["layout_jitter"]
    out = []
    for unit, fid in PRESENTED.items():
        fx = des["_by_id"][fid]
        kx, ky = DRAW_KEYS[fid]
        if fid == "stand":
            piv = np.asarray(des["station_transform_nominal_world_m"], float) @ np.r_[np.asarray(fx["draw_pivot_body_mm"], float) * 1e-3, 1.0]
        elif fid.startswith("leg_station_"):
            piv = np.asarray(des["_by_id"]["leg_bench"]["T_nominal_world_m"], float)[:, 3]
        else:
            piv = np.asarray(fx["T_nominal_world_m"], float) @ np.r_[np.asarray(fx["draw_pivot_body_mm"], float) * 1e-3, 1.0]
        rec = {"unit": unit, "fixture": fid, "fixture_role": fx["role"],
               "about_xy_m": [round(float(piv[0]), 2), round(float(piv[1]), 2)],
               "draw": {"xy_mm_per_axis": float(J[kx]), "yaw_deg": float(J[ky]), "distribution": J["distribution"],
                        "pivot": fx["draw"].get("pivot"), "drawn_with": fx["draw"].get("what")}}
        out.append(rec)
    return out


def _datums(des):
    """the seven boards' print (squares, square size, border, board size), the fixture each is on and whether it moves
    with it; for the two boards on fixtures that are never drawn (the rack table's tool board, the floor board) the
    inner corners in the cell frame (exact: the cell's fixed datums). Where a board sits on a drawn fixture is in the
    drawings, not here (the module's rest pose in its fixture is the hidden bench plan)."""
    out = {}
    for f in des["fixtures"]:
        for p in f["parts"]:
            if "board" not in p:
                continue
            b = p["board"]
            host = f["id"]
            rec = {"host_fixture": host, "squares": [int(v) for v in b["squares"]], "inner_corners": [int(v) for v in b["inner_corners"]],
                   "square_mm": float(b["square_mm"]), "border_mm": float(b["border_mm"]), "board_mm": [float(v) for v in b["board_mm"]],
                   "moves_with_its_host": host not in FIXED_HOSTS}
            if host in FIXED_HOSTS:
                T = np.asarray(f["T_nominal_world_m"], float)
                lo, hi = np.asarray(p["bbox_mm"][0], float), np.asarray(p["bbox_mm"][1], float)
                sq, bd = float(b["square_mm"]), float(b["border_mm"])
                nx, ny = [int(v) for v in b["squares"]]
                pts = []
                for j in range(1, ny):
                    for i in range(1, nx):
                        x = lo[0] + bd + i * sq
                        y = lo[1] + bd + j * sq
                        w = T @ np.array([x * 1e-3, y * 1e-3, hi[2] * 1e-3, 1.0])
                        pts.append([_r(w[0]), _r(w[1]), _r(w[2])])
                rec["inner_corners_cell_m"] = pts
            out[p["part"]] = rec
    return out


def public_description(cfg, des, robot_wrist_cfg):
    """the draw-independent part of describe(): cameras, fastening, items, datums, scoring, carry limits, observation"""
    S = cfg["scoring"]
    C = cfg["coupling"]
    return {"cameras": _cameras(cfg, robot_wrist_cfg), "fastening": _fastening(cfg), "items": _items(cfg, des),
            "datums": _datums(des),
            "scoring": {"weights": {k: float(v) for k, v in C["weights"].items()},
                        "total_weight": float(sum(float(v) for v in C["weights"].values())),
                        "episodes": int(S["episodes"]), "safety_violation_factor": float(S["safety_violation_factor"]),
                        "counts": "a mate fastened at the bolt position that counts, its module not scrapped"},
            "carry_limits": {"tcp_speed_m_s": float(cfg["robot"]["carry_tcp_m_s"]), "tcp_acc_m_s2": float(cfg["robot"]["carry_acc_m_s2"]),
                             "enforced": False},
            "observation": {"wrench_noise": {k: float(v) for k, v in cfg["observation"]["wrench_noise"].items()}}}


def key_paths(d, p=""):
    """every key path of a nested dict ('a.b.c'; list items as 'a[]'): the shape describe() has to match the table"""
    out = set()
    if isinstance(d, dict):
        for k, v in d.items():
            q = p + "." + k if p else k
            out.add(q)
            out |= key_paths(v, q)
    elif isinstance(d, list):
        for v in d:
            if isinstance(v, (dict, list)):
                out |= key_paths(v, p + "[]")
    return out


_G = "gives"
_GW = "gives (wording: the texts)"
# describe()'s FIELD TABLE: key path -> (unit, frame, source, design). '[]' = every item of a list.
_ROWS = [
    ("cell_version", "-", "-", "server.CELL_VERSION", _G),
    ("units", "-", "-", "server.describe (m, rad, wxyz, N)", _G),
    ("units.length", "-", "-", "server.describe", _G), ("units.angle", "-", "-", "server.describe", _G),
    ("units.quaternion", "-", "-", "server.describe", _G), ("units.force", "-", "-", "server.describe", _G),
    ("robot", "-", "-", "the task's design (UR10e-class arm, house wide gripper)", _G),
    ("robot.model", "text", "-", "the server's string", _GW),
    ("robot.base_pos", "m", "cell", "cell_config station.ur_base_pos_m", _G),
    ("robot.joints", "names", "-", "the robot model (gripper_wide.add_ur_wide)", _G),
    ("robot.joint_range", "rad", "joint", "the compiled model's jnt_range", _G),
    ("robot.home_q", "rad", "joint", "cell_config robot.home_q", _G),
    ("robot.torque_limits_Nm", "N m", "joint", "ur_robot.UR_MODELS torque limits", _G),
    ("robot.tcp_offset_from_flange_m", "m", "flange", "gripper_wide tcp_from_flange_m", _G),
    ("robot.tcp_axes", "text", "tcp", "the server's string", _GW),
    ("robot.gripper", "-", "-", "gripper_wide / cell_config gripper", _G),
    ("robot.gripper.max_width_m", "m", "-", "WideController.max_width", _G),
    ("robot.gripper.rate_m_s", "m/s", "-", "ur_controller.GRIP_RATE_M_S", _G),
    ("robot.gripper.force_N", "N per pad", "-", "cell_config gripper.pad_force_N", _G),
    ("robot.gripper.default_force_N", "N per pad", "-", "cell_config gripper.force_N_default", _G),
    ("robot.gripper.width", "text", "-", "the server's string", _GW),
    ("control", "-", "-", "the task's design (the control calls)", _G),
    ("control.tick_s", "s", "-", "cell_config physics (timestep x tick_steps)", _G),
    ("control.joint_rate_limit_rad_s", "rad/s", "joint", "the controller's rate limit", _G),
    ("control.cartesian_speed_limit_m_s", "m/s", "tcp", "ur_controller.CART_V_MAX", _G),
    ("control.cartesian_angular_limit_rad_s", "rad/s", "tcp", "ur_controller.CART_W_MAX", _G),
    ("control.stiffness_scale_range", "-", "-", "ur_controller.STIFFNESS_RANGE", _G),
    ("control.modes", "names", "-", "server.set_control", _G),
    ("control.carry_limits", "-", "-", "cell_config robot.carry_*: a handling procedure, not enforced", _G),
    ("control.carry_limits.tcp_speed_m_s", "m/s", "tcp", "cell_config robot.carry_tcp_m_s", _G),
    ("control.carry_limits.tcp_acc_m_s2", "m/s2", "tcp", "cell_config robot.carry_acc_m_s2", _G),
    ("control.carry_limits.enforced", "bool", "-", "the cell does not enforce it", _G),
    ("cameras", "-", "-", "cell_config cameras", _G),
]
for _n in ("top", "wrist"):
    _ROWS += [
        ("cameras.%s" % _n, "-", "-", "cell_config cameras", _G),
        ("cameras.%s.resolution" % _n, "px", "image", "cell_config cameras.size", _G),
        ("cameras.%s.rgb" % _n, "bool", "-", "-", _G), ("cameras.%s.depth" % _n, "bool", "-", "cell_config cameras.depth", _G),
        ("cameras.%s.nominal_hfov_deg" % _n, "deg", "image", "cell_config cameras.mounts", _G),
        ("cameras.%s.nominal_vfov_deg" % _n, "deg", "image", "cell_config cameras.mounts", _G),
        ("cameras.%s.depth_unit_m" % _n, "m", "-", "render_iface.encode_reply", _G),
        ("cameras.%s.depth_invalid_value" % _n, "-", "-", "render_iface.encode_reply", _G),
        ("cameras.%s.mount" % _n, "-", "-", "the NOMINAL mount (the drawn one is withheld)", _G)]
# the overhead camera only: its mount draw and its focal / centre draws (they reach its frames)
_ROWS += [
    ("cameras.top.mount.mount_draw", "-", "-", "cell_config cameras.camera_draw (the bounds)", _G),
    ("cameras.top.mount.mount_draw.position_mm_per_axis", "mm", "-", "cell_config cameras.camera_draw", _G),
    ("cameras.top.mount.mount_draw.rotation_deg_about_any_axis", "deg", "-", "cell_config cameras.camera_draw", _G),
    ("cameras.top.intrinsics_draw", "-", "-", "cell_config cameras.camera_draw (the bounds)", _G),
    ("cameras.top.intrinsics_draw.focal_rel", "-", "-", "camera_draw.f_rel", _G),
    ("cameras.top.intrinsics_draw.centre_rel", "-", "-", "camera_draw.c_rel", _G)]
_ROWS += [
    ("cameras.top.mount.nominal_lens_front_cell_m", "m", "cell", "cell_config cameras.mounts.top.pos", _G),
    ("cameras.top.mount.looking", "text", "-", "cell_config cameras.mounts.top (straight down)", _GW),
    ("cameras.top.mount.image_top_towards_cell", "unit vector", "cell", "cell_config cameras.mounts.top.up", _G),
    ("cameras.wrist.mount.on", "text", "-", "cell_config cameras.mounts.wrist.on", _GW),
    ("cameras.wrist.mount.lens_in_flange_m", "m", "flange", "gripper_wide DEFAULT_CFG wrist_camera", _G),
    ("cameras.wrist.mount.aim_point_in_flange_m", "m", "flange", "gripper_wide DEFAULT_CFG wrist_camera", _G),
    ("fastening", "-", "-", "coupling_h.capture_table (the measured windows)", _G),
    ("fastening.seat_classes", "-", "-", "coupling_h.capture_table(cfg) rows (no design pose, no phase, no mate name); "
     "SUBTREE: its fields are the capture table's own (coupling_h.capture_table; the measured part from cell_config "
     "coupling.measured)", "gives (subtree: coupling_h.capture_table)"),
    ("fastening.table_units", "-", "-", "coupling_h.capture_table units", _G),
    ("fastening.interface_class_of_unit", "-", "-", "cell_config coupling.interfaces", _G),
    ("fastening.weights", "-", "-", "cell_config coupling.weights", _G),
    ("fastening.total_weight", "-", "-", "sum of the weights (7.0)", _G),
    ("items", "-", "-", "'where each module is': the fixture, its place at about 0.01 m, the draw's bounds", _G),
    ("items[].unit", "id", "-", "kit unit id", _G), ("items[].fixture", "id", "-", "fixtures.json", _G),
    ("items[].fixture_role", "-", "-", "fixtures.json", _G),
    ("items[].about_xy_m", "m (0.01)", "cell", "the fixture's draw pivot at the nominal layout, rounded to 0.01 m (fixtures.json)", _G),
    ("items[].draw", "-", "-", "cell_config fixtures.layout_jitter (the bounds, never the draw)", _G),
    ("items[].draw.xy_mm_per_axis", "mm", "cell", "cell_config fixtures.layout_jitter", _G),
    ("items[].draw.yaw_deg", "deg", "about z", "cell_config fixtures.layout_jitter", _G),
    ("items[].draw.distribution", "text", "-", "cell_config fixtures.layout_jitter.distribution", _GW),
    ("items[].draw.pivot", "text", "-", "fixtures.json draw.pivot", _GW),
    ("items[].draw.drawn_with", "text", "-", "fixtures.json draw.what", _GW),
    ("datums", "-", "-", "the seven boards (fixtures.json parts with a board)", _G),
]
for _b in BOARD_HOSTS:
    _ROWS += [("datums.%s" % _b, "-", "-", "fixtures.json part %s" % _b, _G)]
    for _k, _u in (("host_fixture", "id"), ("squares", "count"), ("inner_corners", "count"), ("square_mm", "mm"),
                   ("border_mm", "mm"), ("board_mm", "mm"), ("moves_with_its_host", "bool")):
        _ROWS.append(("datums.%s.%s" % (_b, _k), _u, "-", "fixtures.json board", _G))
    if BOARD_HOSTS[_b] in FIXED_HOSTS:
        _ROWS.append(("datums.%s.inner_corners_cell_m" % _b, "m", "cell", "fixtures.json board + its fixed host's T_nominal_world_m", _G))
_ROWS += [
    ("budgets", "-", "-", "cell_config budgets", _G),
    ("budgets.sim_time_cap_s", "s", "-", "cell_config budgets.sim_cap_s", _G),
    ("budgets.wall_clock_cap_s", "s", "-", "cell_config budgets.wall_cap_s (the grader overrides it with the budget of record)", _G),
    ("budgets.silence_abort_s", "s", "-", "cell_config budgets.silence_s", _G),
    ("budgets.max_ticks_per_step", "ticks", "-", "cell_config budgets.max_ticks_per_step", _G),
    ("budgets.camera_frames", "frames", "-", "cell_config budgets.camera_frames", _G),
    ("safety", "-", "-", "cell_config safety", _G),
    ("safety.robot_force_N", "N", "-", "R1", _G), ("safety.part_force_N", "N", "-", "R2", _G),
    ("safety.part_force_window_s", "s", "-", "R2: part_ticks x tick", _G), ("safety.part_speed_m_s", "m/s", "-", "R4", _G),
    ("safety.floor_contact", "text", "-", "R5", _GW), ("safety.violation_score_factor", "-", "-", "cell_config scoring", _G),
    ("scoring", "-", "-", "cell_config scoring and coupling.weights", _G),
    ("scoring.weights", "-", "-", "cell_config coupling.weights", _G), ("scoring.total_weight", "-", "-", "7.0", _G),
    ("scoring.episodes", "count", "-", "cell_config scoring.episodes", _G),
    ("scoring.safety_violation_factor", "-", "-", "cell_config scoring", _G),
    ("scoring.counts", "text", "-", "the cell's scoring rule", _GW),
    ("observation", "-", "-", "cell_config observation (get_state)", _G),
    ("observation.wrench_noise", "-", "-", "cell_config observation.wrench_noise", _G),
    ("observation.wrench_noise.sigma_f_N", "N", "flange", "cell_config", _G),
    ("observation.wrench_noise.sigma_t_Nm", "N m", "flange", "cell_config", _G),
    ("observation.wrench_noise.bias_f_N", "N", "flange", "cell_config", _G),
    ("observation.wrench_noise.bias_t_Nm", "N m", "flange", "cell_config", _G),
]
for _u in ("leg_R", "leg_L", "arm_L", "arm_R", "head"):
    _ROWS += [("fastening.weights.%s" % _u, "-", "-", "cell_config coupling.weights", _G),
              ("scoring.weights.%s" % _u, "-", "-", "cell_config coupling.weights", _G),
              ("fastening.interface_class_of_unit.%s" % _u, "class", "-", "cell_config coupling.interfaces", _G)]
for _k in ("length", "angle", "time", "speed"):
    _ROWS.append(("fastening.table_units.%s" % _k, "-", "-", "coupling_h.capture_table units", _G))
FIELD_TABLE = {k: {"unit": u, "frame": f, "source": s, "design": g} for k, u, f, s, g in _ROWS}

# what the design withholds, and the test that shows it is absent
WITHHELD = {
    "mating graph, design poses, design phases (bolt positions that count)": "no key 'interfaces' / 'design' / 'phase' / 'k'; the "
        "graph's seat frames' translations do not occur among describe()'s numbers",
    "module poses (ground truth)": "no module root position (get_state_privileged) occurs among describe()'s numbers (1e-6 m)",
    "the episode's draws: the layout (stand, clamp repeat, racks, nest)": "describe() is identical for two seeds and the nominal layout",
    "the camera mount draw and the drawn lenses": "describe() identical across seeds; the drawn lens front is not among its numbers",
    "the collision pieces and the bench plan": "no key 'collision' / 'pieces' / 'plan'",
    "the coupling's states, the lock events, the scrap record": "no key 'state' / 'states' / 'locked'; step() events are only "
        "'safety:' and 'episode_over' (EVENT_TABLE)",
    "the reference program and the oracle token": "privileged calls answer 'forbidden' without the token; describe() names none",
}
OBSERVATION_TABLE = {
    "sim_time": ("s", "-", "MjData.time"), "done": ("bool", "-", "the server's episode end"),
    "robot": ("-", "-", "the arm and the gripper only (no module, no fixture)"),
    "robot.q": ("rad", "joint", "the six arm joints"), "robot.dq": ("rad/s", "joint", "the six arm joints"),
    "robot.cmd_q": ("rad", "joint", "the controller's commanded configuration"),
    "robot.gripper_width": ("m", "-", "the pads' gap"), "robot.force_cmd": ("N per pad", "-", "the force register"),
    "robot.tcp_pos": ("m", "cell", "the tcp site"), "robot.tcp_quat": ("wxyz", "cell", "the tcp site"),
    "robot.wrench": ("N, N m", "flange", "the F/T sensor + the episode's bias + white noise (cell_config observation)"),
    "robot.mode": ("-", "-", "joint | cartesian"), "robot.stiffness_scale": ("-", "-", "the controller"),
}
EVENT_TABLE = {
    "safety:robot_contact:<robot body>:<other body>:<N>N": "R1 (monitor.py): a robot link above robot_force_N on a body that is not the robot (the pads exempt on a module only)",
    "safety:part_force:<module>:<body pressed on>:<N>N": "R2: a module above part_force_N for part_ticks ticks on one body",
    "safety:part_speed:<module>:<v>m/s": "R4: a module's centre-of-mass speed above part_speed_m_s",
    "safety:floor_contact:<module>:<module body>": "R5: a module body on the floor (the module is scrapped)",
    "episode_over": "the episode ended (sim cap, finish)",
}
EVENT_PATTERNS = [r"^safety:robot_contact:ur/[^:]+:[^:]+:\d+N$", r"^safety:part_force:[A-Za-z_]+:[^:]+:\d+N$",
                  r"^safety:part_speed:[A-Za-z_]+:\d+\.\d\dm/s$", r"^safety:floor_contact:[A-Za-z_]+:[^:]+$", r"^episode_over$"]
