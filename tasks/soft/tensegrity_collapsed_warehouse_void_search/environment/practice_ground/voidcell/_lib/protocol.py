"""The station's calls on the wire (given/station_api.md): names, argument kinds, the reply-key whitelist, the error codes.

Every reply of a public call is a JSON object whose keys are a subset of REPLY_KEYS[method] (nested objects:
NESTED_KEYS), or an error {"error": code[, "detail": text]} with code in ERROR_CODES. No reply carries geometry of
the site: the outcome of a command is only what the off-board PID knows (done flags, timeouts); `pose` answers only
when the task's pose_source is "cell". The QC (the tests and tests/review/qc) imports this module, the server enforces it on every
reply (`check_reply`), so a reply outside the whitelist is the cell's own failure (internal_error), never data.
"""

PUBLIC_METHODS = ("mission", "get_frame", "telemetry", "status", "run_primitive", "set_targets", "stop",
                  "report_point", "step", "finish", "pose")
PRIVILEGED_METHODS = ("reference_run",)
PRACTICE_METHODS = ("reset",)
# calls that change the episode's state (refused with episode_over once the episode has ended)
STATE_METHODS = ("run_primitive", "set_targets", "stop", "report_point", "step")
# the cell's own work, excluded from the program's time (cellkit.wire: program time = wall - work_wall)
WORK_METHODS = ("step", "get_frame", "telemetry", "pose")

# the fixed enumeration of the station's codes (given/station_api.md, Errors), plus the three wire-level codes of the cell kit's
# convention: forbidden (an unknown, privileged or practice-only call), episode_over (a state call after the end),
# internal_error (the cell's own failure; the grader re-runs such an episode)
STATION_ERRORS = ("busy", "invalid_name", "invalid_bottom", "out_of_range", "budget_exhausted", "not_available",
                  "bad_argument")
WIRE_ERRORS = ("forbidden", "episode_over", "internal_error")
ERROR_CODES = STATION_ERRORS + WIRE_ERRORS

PRIMITIVE_NAMES = ("roll_100_100", "roll_120_120", "roll_140_140", "roll_100_120", "roll_120_100", "roll_100_140",
                   "roll_140_100", "roll_120_140", "roll_140_120", "ccw", "cw")
ROLLABLE_FACES = ((0, 2, 5), (0, 3, 5), (1, 2, 4), (1, 2, 5), (0, 3, 4), (1, 3, 4))

# argument names of each public call (anything else: bad_argument, nothing runs)
ARGS = {"mission": (), "get_frame": (), "telemetry": (), "status": (), "stop": (), "finish": (), "pose": (),
        "run_primitive": ("name", "bottom_nodes"), "set_targets": ("lengths_mm",), "report_point": ("i",),
        "step": ("dt",)}

REPLY_KEYS = {
    "mission": ("site", "pad", "search_points", "dwell_s", "credit_radius_m", "still_tolerance_m", "camera",
                "limits", "pose_source"),
    "get_frame": ("index", "t", "width", "height", "encoding", "rgb_b64", "depth_b64", "depth_unit_m", "frames_left"),
    "telemetry": ("n", "dropped", "columns", "t", "sensor_length_mm", "sensor_capacitance_pF", "motor_encoder_counts",
                  "motor_encoder_length_mm", "motor_target_mm", "motor_done", "motor_duty", "imu_accel_g",
                  "imu_gyro_dps"),
    "status": ("t", "busy", "kind", "state_index", "done", "timeouts"),
    "run_primitive": ("accepted", "error"),
    "set_targets": ("accepted", "error"),
    "stop": (),
    "report_point": ("accepted", "reports_left", "error"),
    "step": ("t", "done"),
    "finish": ("done",),
    "pose": ("t", "index", "nodes_xyz", "visible_caps"),
}
NESTED_KEYS = {
    ("mission", "site"): ("frame", "size_m"),
    ("mission", "pad"): ("xy", "retrieval_radius_m", "setdown_error_m"),
    ("mission", "search_points"): ("id", "xy", "weight"),
    ("mission", "camera"): ("width", "height", "K", "depth_unit_m", "depth_range_m", "rate_hz", "T_world_camera"),
    ("mission", "limits"): ("time_limit_s", "frames", "reports_per_point", "cable_length_mm", "program_time_s",
                            "silence_s", "cpu_s"),
}


def is_error(res):
    return isinstance(res, dict) and "error" in res and set(res) <= {"error", "detail"} and \
        res.get("error") in ERROR_CODES


def check_reply(method, res):
    """None when the reply is inside the whitelist, else a short reason (the server turns it into internal_error)."""
    if not isinstance(res, dict):
        return "reply is not an object"
    if "error" in res and set(res) <= {"error", "detail"}:
        return None if res["error"] in ERROR_CODES else "error code %r not in the enumeration" % (res["error"],)
    allowed = REPLY_KEYS.get(method)
    if allowed is None:
        return None                       # privileged / practice calls: not part of the public contract
    extra = set(res) - set(allowed)
    if extra:
        return "keys %s not in the whitelist of %s" % (sorted(extra), method)
    if res.get("error") is not None and res["error"] not in ERROR_CODES:
        return "error code %r not in the enumeration" % (res["error"],)
    for (m, k), sub in NESTED_KEYS.items():
        if m != method or k not in res:
            continue
        vals = res[k] if isinstance(res[k], list) else [res[k]]
        for v in vals:
            if not isinstance(v, dict) or set(v) - set(sub):
                return "keys of %s.%s outside the whitelist" % (method, k)
    return None
