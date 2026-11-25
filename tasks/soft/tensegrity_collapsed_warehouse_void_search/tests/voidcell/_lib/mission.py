"""mission(): what the rescue team is given for one episode. Built key by key from a whitelist: the site
rectangle and its frame, the pad, the three search points with their weights, the scoring constants every team
knows, the camera's intrinsics and its calibrated (given) extrinsic, the budgets, the pose source. Nothing of the
layout (boxes, openings, classes, the void, patches, chips, friction, the true extrinsic, seeds) is copied: the
values are taken from named fields only, never by copying a dict of the site package.
"""

from . import protocol

SITE_FRAME = "x east, y north, z up, origin = south-west floor corner"
DWELL_S = 20.0
CREDIT_RADIUS_M = [0.20, 0.40]
STILL_TOLERANCE_M = 0.02
RETRIEVAL_RADIUS_M = 0.60
SETDOWN_ERROR_M = 0.10


def _r(x, n=6):
    return round(float(x), n)


def build(site, limits, pose_source):
    cm = site["camera"]["model"]
    given = site["camera"]["given"]["T_world_camera"]
    out = {
        "site": {"frame": SITE_FRAME, "size_m": [_r(v) for v in site["size_m"]]},
        "pad": {"xy": [_r(v) for v in site["pad"]["xy"]], "retrieval_radius_m": RETRIEVAL_RADIUS_M,
                "setdown_error_m": SETDOWN_ERROR_M},
        "search_points": [{"id": int(p["id"]), "xy": [_r(v) for v in p["xy"]], "weight": _r(p["weight"], 4)}
                          for p in sorted(site["search_points"], key=lambda p: int(p["id"]))],
        "dwell_s": DWELL_S,
        "credit_radius_m": list(CREDIT_RADIUS_M),
        "still_tolerance_m": STILL_TOLERANCE_M,
        "camera": {"width": int(cm["width"]), "height": int(cm["height"]),
                   "K": [_r(cm["fx"], 3), _r(cm["fy"], 3), _r(cm["cx"], 3), _r(cm["cy"], 3)],
                   "depth_unit_m": float(cm["depth_unit_m"]), "depth_range_m": [float(v) for v in cm["depth_range_m"]],
                   "rate_hz": float(cm["rate_hz"]),
                   "T_world_camera": [[_r(v) for v in row] for row in given]},
        "limits": {"time_limit_s": float(limits["time_limit_s"]), "frames": int(limits["frames"]),
                   "reports_per_point": int(limits["reports_per_point"]),
                   "cable_length_mm": [float(v) for v in limits["cable_length_mm"]],
                   "program_time_s": float(limits["program_time_s"]), "silence_s": float(limits["silence_s"]),
                   "cpu_s": float(limits["cpu_s"])},
        "pose_source": str(pose_source),
    }
    why = protocol.check_reply("mission", out)
    if why:
        raise RuntimeError("mission(): " + why)
    return out
