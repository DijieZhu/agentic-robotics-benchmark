"""The robot's 17 public calls, in process, on top of the mission world.

    describe, mission, get_state, get_camera, get_depth, get_thermal, get_lidar, walk, set_mode, set_body, set_head,
    set_lamp, step, report, report_mark, report_source, finish

This is the implementation the grader's cell (tests/plantcell/server.py) puts on the wire: the server calls
these methods, encodes their numpy results (PNG, little-endian arrays) and turns a CallError into an error reply.
A program never holds this object: it talks to the server through the client proxy (plantcell.client), over a pipe,
a socket, or (tests, the privileged reference) an in-process handler with the same JSON round trip.

Rules:
  * nothing returned contains ground truth, credit, a true pose, an instrument's name in an event, or a grader-side
    record (the evidence records, the true camera and base poses stay in the world's frame log);
  * a call past its budget, after the end of the mission, or with bad arguments raises CallError and changes nothing;
  * time passes only in step().

Units and frames: metres, radians, seconds; the site frame has its origin at the deployment pad centre, x into the
building, y left, z up. Camera poses are the OpenCV optical frame (x right, y down, z forward) in the robot's body
frame (x forward, y left, z up, origin at the body centre). Dose rates in Gy/h, doses in Gy.
"""

from __future__ import annotations

import copy
import json
import math
import os

import numpy as np

from . import paths
from .catalog import GRID_MARKER_M, public_catalog

PUBLIC_METHODS = ("describe", "mission", "get_state", "get_camera", "get_depth", "get_thermal", "get_lidar", "walk",
                  "set_mode", "set_body", "set_head", "set_lamp", "step", "report", "report_mark", "report_source",
                  "finish")
# the events a program sees (step() replies); each is {"t", "type"} (+ "report_id" for report_sent)
PUBLIC_EVENTS = ("blocked", "protective_sit", "link_up", "link_down", "report_sent", "robot_lost", "dose_limit",
                 "time_limit")
MODES = ("stand", "walk", "high_step", "crouch_walk", "sit")
SIT_REST_HEIGHT_M = 0.035       # body centre above the floor when seated: the body rests on the floor (rated_modes)
CAMERAS = ("nav", "inspect")
MAX_CITED = 3

# round-sheet fields (ladder prior.json) a program needs on the work-order item of the same tag (draft prior format)
ROUND_SHEET_FIELDS = ("type_text", "kind", "scale", "case_mm", "surface", "component", "window_m",
                      "n_lamps", "colors", "last_2009")
# version 2: nothing that identifies a plate but its tag; removed from every work-order and round-sheet item
# whatever a package holds (version 1 packages carry them)
CODE_FIELDS = ("label_code", "marker_id")
# what of a site's prior.json is the public document (drawing, rooms, zoning, link, work order, round sheet, ...)
PUBLIC_PRIOR_KEYS = ("frame", "datum", "drawing", "building", "grid", "rooms", "walls", "doors", "equipment",
                     "grid_markers", "zoning", "link", "work_order", "round_sheet", "modification_log",
                     "facility_facts")
DRAFT_PRIOR_SCHEMAS = ("site_gen/prior/0.2-ladder",)


class CallError(Exception):
    """A refused call: `code` (bad_params, budget, mission_over) and a human-readable `detail`."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__("%s: %s" % (code, detail) if detail else code)
        self.code = str(code)
        self.detail = str(detail)


def _finite(name, v, lo=None, hi=None):
    # (the cell's server checks every argument before it reaches here: plantcell/server.py check_args; this is the
    # same rule for in-process callers: any failure to convert is the caller's bad_params, never the cell's error)
    try:
        x = float(v)
    except (TypeError, ValueError, OverflowError):
        raise CallError("bad_params", "%s must be a number" % name) from None
    if not math.isfinite(x):
        raise CallError("bad_params", "%s must be finite" % name)
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def _check_value(v, depth=0):
    """A report value: None, a finite number, a string, a bool, a list of those, or a dict of those (one level)."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        if len(v) > 64:
            raise CallError("bad_params", "a reported word has at most 64 characters")
        return v
    if isinstance(v, (int, float, np.integer, np.floating)):
        try:
            x = float(v)
        except OverflowError:
            raise CallError("bad_params", "a reported number is at most 1e15 in magnitude") from None
        if not math.isfinite(x):
            raise CallError("bad_params", "a reported value must be finite")
        if abs(x) > 1e15:
            raise CallError("bad_params", "a reported number is at most 1e15 in magnitude")
        return v.item() if isinstance(v, (np.integer, np.floating)) else v
    if isinstance(v, np.ndarray):
        return _check_value(v.tolist(), depth)
    if depth < 2 and isinstance(v, (list, tuple)):
        if len(v) > 64:
            raise CallError("bad_params", "a reported list has at most 64 entries")
        return [_check_value(x, depth + 1) for x in v]
    if depth < 2 and isinstance(v, dict):
        if len(v) > 8:
            raise CallError("bad_params", "a reported dict has at most 8 entries")
        return {str(k)[:64]: _check_value(x, depth + 1) for k, x in v.items()}
    raise CallError("bad_params", "a reported value is a number, a string, a list or a small dict")


class Robot:
    """The robot of one mission (in process). Construct with a MissionWorld."""

    def __init__(self, world):
        self._w = world

    # ------------------------------------------------------------------ helpers
    def _live(self):
        if self._w.done:
            raise CallError("mission_over", "the mission has ended (%s)" % (self._w.done_reason,))

    def _budget(self, key):
        """Refuse when the frame budget `key` is used up; the frame counts only once it was taken (_used)."""
        w = self._w
        lim = int(w.cfg["budgets"]["frames"][key])
        if w.n_frames[key] >= lim:
            raise CallError("budget", "%s frame budget used (%d)" % (key, lim))

    def _used(self, key):
        self._w.n_frames[key] += 1

    def _T_body(self, T_world):
        Tb = self._w.T_world_base()
        return np.linalg.inv(Tb) @ np.asarray(T_world)

    # ------------------------------------------------------------------ information
    def describe(self) -> dict:
        """The robot's datasheet: dimensions, modes, cameras, lidar, head, time, budgets, dosimeter, odometry,
        link, tag plates, frames. No generator band, no physics setting."""
        w = self._w
        cfg = w.cfg
        cams = {}
        for k in ("nav", "inspect", "thermal"):
            cm = w.cam_models[k]
            cams[k] = {"width": int(cm.width), "height": int(cm.height), "hfov_deg": float(cm.hfov_deg),
                       "K": np.asarray(cm.K).tolist()}
        cams["nav"]["depth_range_m"] = [float(v) for v in w.cam_models["nav"].depth_range]
        cams["nav"]["depth"] = "metres along the optical axis; NaN where the camera has no reading"
        cams["thermal"]["unit"] = "degrees C (apparent temperature, emissivity 1)"
        Tbl = np.eye(4)
        Tbl[:3, 3] = w.lidar_offset_body()
        D = cfg["dose"]
        out = {
            "robot": {"model": "Q12 inspection quadruped: 12-actuator quadruped inspection robot with pan-tilt sensor "
                               "head and planar lidar",
                      "mass_kg": round(float(w.model.body_subtreemass[w.base_id]), 2), "length_m": 0.55,
                      "width_m": 0.33, "height_m": 0.32, "top_above_body_origin_m": 0.081,
                      "camera_ahead_of_body_centre_m": 0.18},
            "modes": rated_modes(w),
            "cameras": cams,
            "lidar": {"beams": int(w.lidar_model.beams), "range_m": [float(w.lidar_model.r_min),
                                                                     float(w.lidar_model.r_max)],
                      "T_body_lidar": Tbl.tolist(),
                      "angles": "beam k at k x 360 / beams degrees, counter-clockwise from the body's x axis",
                      "plane": "horizontal through the lidar, moves with the body"},
            "head": {"limits_rad": {k: [float(v) for v in lim] for k, lim in cfg["sensors"]["head_limits"].items()},
                     "pan_axis_body": [0.165, 0.0, 0.043], "tilt_axis_above_pan_m": 0.012,
                     "tilt_positive": "nose down", "lamp_range_m": 3.0},
            "time": {"tick_s": float(cfg["time"]["tick_s"]), "max_ticks_per_step": int(cfg["time"]["max_ticks_per_step"]),
                     "sim_time_limit_s": float(w.sim_time_limit)},
            "budgets": {"frames": {k: int(v) for k, v in cfg["budgets"]["frames"].items()},
                        "reports": {k: int(v) for k, v in cfg["budgets"]["reports"].items()},
                        "max_cited_frames": MAX_CITED},
            "dosimeter": {"unit_rate": "Gy/h", "unit_dose": "Gy", "sensitivity_cps_per_gyh": float(w.sens),
                          "rating_gy": round(float(w.rating_gy), 4), "position": "body centre",
                          "reading": "get_state()['dosimeter']: the counts since the previous get_state() call and "
                                     "the time they were counted over; rate = counts / (dwell_s x sensitivity)",
                          "accumulated_dose": "get_state()['dosimeter']['dose_gy'] is the dosimeter's own estimate, "
                                              "integrated from its counts",
                          "sample_s": float(D["sample_s"]),
                          "rating_rule": "the mission ends when the dose the sensor head has received reaches the "
                                         "rating (the dosimeter's estimate follows it within its counting statistics)"},
            "odometry": {"drift": "on level floor about 2 % of the distance travelled, and the heading drifts up to "
                                  "about 1 degree per minute; on ramps and slopes much more: typically 5 to 15 % of "
                                  "the distance walked on them, at worst about a third",
                         "start": "starts at the deployment pad pose of the drawing (mission()['datum']); the robot "
                                  "is set down within 5 cm and 2 degrees of it"},
            "link": {"base_station_range_m": float(cfg["link"]["base_range_m"]),
                     "repeater_range_m": float(cfg["link"]["repeater_range_m"]),
                     "rule": "a report is delivered when the robot has line of sight to a working link node in range; "
                             "otherwise it is queued and sent when the link comes up; queued reports are lost when "
                             "the mission ends"},
            # version 2: the grid marker plates only (instrument tag plates carry text: mission()["catalog"]["tags"])
            "tags": {"grid_marker_m": GRID_MARKER_M, "dictionary": "DICT_APRILTAG_36h11"},
            "events": list(PUBLIC_EVENTS),
            "blocked_hold_s": float(w.hold_s),
            "frames": "site frame: origin at the pad centre, x into the building, y left, z up; camera poses: OpenCV "
                      "optical frame (x right, y down, z forward) in the body frame (x forward, y left, z up)",
        }
        return out

    def mission(self) -> dict:
        """The documents of this site: the drawing (rooms, walls, doors, equipment, grid and its marker plates,
        zoning, link nodes), the work order, the round sheet (2009 values), the modification record and the
        facility facts; plus the plant catalogue (key 'catalog') and the enforced budgets (key 'budgets').
        Version 2: an instrument is identified by its tag alone (no label code, no marker id)."""
        w = self._w
        out = public_prior(w.prior, w.cfg, w.rating_gy, w.sim_time_limit)
        out["catalog"] = public_catalog()
        return out

    def get_state(self) -> dict:
        w = self._w
        st = w.st
        x, y, yaw = w.drift.pose(float(st["odom_yaw"]))
        counts, dwell = w.take_dosimeter()
        roll, pitch, _ = st["imu_rpy"]
        hd = st["head"]
        rssi = None
        if w.link_up and w.link_dist is not None:
            rssi = round(-40.0 - 20.0 * math.log10(max(w.link_dist, 0.5)), 1)
        return {
            "sim_time": round(w.time, 4), "done": bool(w.done), "done_reason": w.done_reason,
            # during the hold after a `blocked` the status stays `blocked` whatever the gait says (a mode change
            # clears the gait's own report at once)
            "status": "blocked" if w.hold_active() else str(st.get("status", "ok")),
            "odometry": {"x": float(x), "y": float(y), "yaw": float(yaw),
                         "distance": max(0.0, float(st["odom_distance"]) - w.odom_distance0) * (1.0 + w.drift.k)},
            "imu": {"roll": float(roll), "pitch": float(pitch), "gyro": [float(v) for v in st["imu_gyro"]]},
            "joint_pos": [float(v) for v in st["joint_pos"]],
            "foot_contact": [bool(v) for v in st["foot_contact"]],
            "mode": str(st["mode"]),
            "body": {"height": float(st["body_height"]), "pitch_cmd": float(w.body_pitch)},
            "head": {"pan": float(hd[0]), "tilt": float(hd[1])},
            "lamp": bool(w.lamp_on),
            "dosimeter": {"counts": int(counts), "dwell_s": round(float(dwell), 4), "dose_gy": float(w.dose_meas_gy),
                          "dose_fraction": float(w.dose_meas_gy / w.rating_gy)},
            "link": {"up": bool(w.link_up), "rssi_dbm": rssi},
            "reports_pending": int(sum(1 for r in w.reports if r["status"] == "queued")),
            "frames_used": {k: int(v) for k, v in w.n_frames.items()},
            "reports_used": {k: int(v) for k, v in w.n_reports.items()},
        }

    # ------------------------------------------------------------------ sensors
    def get_camera(self, name: str = "nav", depth: bool = True) -> dict:
        w = self._w
        if str(name) not in CAMERAS:
            raise CallError("bad_params", "unknown camera %r (nav, inspect)" % (name,))
        name = str(name)
        self._live()
        self._budget(name)
        want_depth = bool(depth) and name == "nav"
        fid, fr, T = w.capture(name, depth=want_depth)
        self._used(name)
        out = {"frame_id": fid, "sensor": name, "t": round(w.time, 4), "image": fr.rgb,
               "K": np.asarray(fr.K, float).copy(), "T_body_cam": self._T_body(T), "gain": float(fr.gain)}
        if want_depth:
            out["depth"] = fr.depth
        return out

    def get_depth(self) -> dict:
        w = self._w
        self._live()
        self._budget("depth")
        fid, z, K, T = w.capture_depth()
        self._used("depth")
        return {"frame_id": fid, "sensor": "depth", "t": round(w.time, 4), "depth": z,
                "K": np.asarray(K, float).copy(), "T_body_cam": self._T_body(T)}

    def get_thermal(self) -> dict:
        w = self._w
        self._live()
        self._budget("thermal")
        fid, img, K, T = w.capture_thermal()
        self._used("thermal")
        return {"frame_id": fid, "sensor": "thermal", "t": round(w.time, 4), "image_c": img,
                "K": np.asarray(K, float).copy(), "T_body_cam": self._T_body(T)}

    def get_lidar(self) -> dict:
        w = self._w
        self._live()
        self._budget("lidar")
        sc = w.scan_lidar()
        self._used("lidar")
        n = w.lidar_model.beams
        return {"t": round(w.time, 4), "ranges": np.asarray(sc.ranges[0], np.float32).copy(),
                "angles": (np.arange(n) * (2 * math.pi / n)).astype(np.float32),
                "T_body_lidar": self._T_body(sc.T_world_lidar)}

    # ------------------------------------------------------------------ motion
    def walk(self, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0) -> dict:
        """The walking command. For hold_s after a `blocked` report it is not taken: the reply says so
        (ok false, held true, the simulated time the hold ends) and nothing changes."""
        self._live()
        a = (_finite("vx", vx), _finite("vy", vy), _finite("yaw_rate", yaw_rate))
        w = self._w
        if not w.set_velocity(*a):
            return {"ok": False, "held": True, "hold_until": round(float(w.hold_until), 4),
                    "detail": "no walking command is taken for %.1f s after a blocked report (until t = %.2f s)"
                              % (w.hold_s, float(w.hold_until))}
        return {"ok": True}

    def set_mode(self, m: str = "stand") -> dict:
        if str(m) not in MODES:
            raise CallError("bad_params", "unknown mode %r (%s)" % (m, ", ".join(MODES)))
        self._live()
        self._w.set_mode(str(m))
        return {"ok": True}

    def set_body(self, height: float | None = None, pitch: float | None = None) -> dict:
        h = None if height is None else _finite("height", height)
        p = None if pitch is None else _finite("pitch", pitch)
        self._live()
        self._w.set_body(h, p)
        return {"ok": True}

    def set_head(self, pan: float | None = None, tilt: float | None = None) -> dict:
        p = None if pan is None else _finite("pan", pan)
        t = None if tilt is None else _finite("tilt", tilt)
        self._live()
        pp, tt = self._w.set_head(p, t)
        return {"ok": True, "pan": float(pp), "tilt": float(tt)}

    def set_lamp(self, on: bool = False) -> dict:
        self._live()
        self._w.set_lamp(bool(on))
        return {"ok": True}

    def step(self, n_ticks: int = 1) -> dict:
        w = self._w
        try:
            n = int(n_ticks)
        except (TypeError, ValueError, OverflowError):
            raise CallError("bad_params", "n_ticks must be an integer") from None
        if n < 1:
            raise CallError("bad_params", "n_ticks must be >= 1")
        evs = w.step(n)
        return {"sim_time": round(w.time, 4), "done": bool(w.done),
                "events": [public_event(e) for e in evs if e.get("type") in PUBLIC_EVENTS]}

    # ------------------------------------------------------------------ reports
    def report(self, tag: str, value=None, frame_ids=()) -> dict:
        w = self._w
        if isinstance(frame_ids, str):
            frame_ids = [frame_ids]
        try:
            fids = [str(f)[:16] for f in list(frame_ids or [])]
        except TypeError:
            raise CallError("bad_params", "frame_ids is a list of frame ids") from None
        if len(fids) > MAX_CITED:
            raise CallError("bad_params", "at most %d frame ids" % MAX_CITED)
        tag = str(tag)[:64]
        if w.report_kind_of(tag) != "instrument":
            raise CallError("bad_params", "use report_mark / report_source for this tag")
        value = _check_value(value)
        return w.report("instrument", tag, value, fids)

    def report_mark(self, tag: str, dose_rate: float | None = None) -> dict:
        w = self._w
        tag = str(tag)[:64]
        if w.report_kind_of(tag) != "mark":
            raise CallError("bad_params", "not a survey mark of the work order")
        v = None if dose_rate is None else _finite("dose_rate", dose_rate)
        return w.report("mark", tag, v, [])

    def report_source(self, x: float, y: float) -> dict:
        w = self._w
        xy = [_finite("x", x), _finite("y", y)]
        tags = [it["tag"] for it in w.prior["work_order"]["items"] if it["type"] == "hotspot"]
        return w.report("source", tags[0] if tags else "SOURCE", xy, [])

    def finish(self) -> dict:
        w = self._w
        w.end("finish")
        return {"ok": True, "sim_time": round(w.time, 4)}


# ============================================================================================== documents
def public_event(e: dict) -> dict:
    out = {"t": float(e["t"]), "type": str(e["type"])}
    if e["type"] == "report_sent":
        out["report_id"] = int(e.get("report_id"))
    return out


def public_prior(prior: dict, cfg: dict, rating_gy: float, sim_time_limit: float) -> dict:
    """The public document of a site's prior.json: an allow-list of its top-level keys, the enforced budgets, and
    (while the site packages are in the draft format) the draft adapter."""
    from .scoring import item_weights
    out = {k: copy.deepcopy(prior[k]) for k in PUBLIC_PRIOR_KEYS if k in prior}
    if prior.get("schema") in DRAFT_PRIOR_SCHEMAS:
        _draft_adapter(out)
    wts = item_weights(out, cfg["weights"])
    for it in out["work_order"]["items"]:
        if it["tag"] in wts:
            it["weight"] = round(wts[it["tag"]], 4)
    for it in list(out["work_order"]["items"]) + list((out.get("round_sheet") or {}).get("items", [])):
        for k in CODE_FIELDS:
            it.pop(k, None)
    out["work_order"].pop("tag_codes", None)
    out["budgets"] = {"sim_time_limit_s": float(sim_time_limit), "dose_rating_gy": round(float(rating_gy), 4),
                      "frames": {k: int(v) for k, v in cfg["budgets"]["frames"].items()},
                      "reports": {k: int(v) for k, v in cfg["budgets"]["reports"].items()},
                      "max_cited_frames": MAX_CITED,
                      "program_time_s": float(cfg["budgets"]["program_time_s"]),
                      "silence_s": float(cfg["budgets"]["silence_s"])}
    return out


def _draft_adapter(out: dict) -> None:
    """The draft ladder format (site_gen/prior/0.2-ladder) keeps half of each item on the round sheet and prints
    the dose in the generator's units: merge the round sheet's fields into the work-order items, write the dose in
    Gy/h and drop the generator's own suggestions. Removed when the site packages carry the final format."""
    rs = {it.get("tag"): it for it in (out.get("round_sheet") or {}).get("items", [])}
    for it in out["work_order"]["items"]:
        r = rs.get(it["tag"])
        if r is not None:
            for k in ROUND_SHEET_FIELDS:
                if k in r and k not in it:
                    it[k] = copy.deepcopy(r[k])
        if it.get("type") == "dose_point":
            it["report"] = "dose rate (Gy/h) at the survey mark"
            it.pop("dwell_s", None)
        if it.get("type") == "hotspot":
            it["report"] = "position (x, y in the site frame) of the unlisted source"
    for it in (out.get("round_sheet") or {}).get("items", []):
        if it.get("type") == "dose_point":
            it["report"] = "dose rate (Gy/h) at the survey mark"
            it.pop("dwell_s", None)
            last = it.get("last_2009") or {}
            if "unit" in last:
                last["unit"] = "Gy/h"
        if it.get("type") == "hotspot":
            it["report"] = "position (x, y in the site frame) of the unlisted source"
        it.pop("evidence", None)


def rated_modes(w) -> dict:
    """The mode table of the datasheet: per mode the firmware's own limits (walking, body height range and default,
    speed limits; the same keys always), and under "tables" the gait's measured manual tables
    (manual_tables.json: speed tracking, stopping, posture, and pass / blocked / fall rates per obstacle, slope,
    surface and space) when the file exists."""
    tables = None
    p = paths.MANUAL_TABLES_JSON
    if p and os.path.exists(p):
        try:
            with open(p) as f:
                tables = json.load(f)
        except (OSError, ValueError):
            tables = None
    out = {"source": "firmware limits per mode; the measured tables under 'tables'" if tables is not None else
           "firmware limits per mode (the measured mode tables are in the robot manual)"}
    fw = getattr(w.ctl, "modes", None) or {}
    for name in MODES:
        md = fw.get(name)
        if md is None:
            out[name] = {}
            continue
        row = {"walking": bool(md.walking), "body_height_m": [float(md.height_range[0]), float(md.height_range[1])],
               "default_body_height_m": float(md.height)}
        if md.walking:
            row.update({"max_vx_mps": float(md.v_max[0]), "max_vy_mps": float(md.v_max[1]),
                        "max_yaw_rate_rps": float(md.v_max[2])})
        out[name] = row
    out["stand"]["pitch_rad"] = [-math.radians(20.0), math.radians(20.0)]
    if out.get("sit"):
        # what the body does, not the firmware's posture target (0.06-0.10 m, 0.075): sitting, the body rests on the
        # floor and set_body does not move it (measured on the training
        # hall's pad: true base height 0.0348 m after 6 s, 0.0337 / 0.0332 m after set_body(0.06) / (0.10))
        out["sit"]["body_height_m"] = [SIT_REST_HEIGHT_M, SIT_REST_HEIGHT_M]
        out["sit"]["default_body_height_m"] = SIT_REST_HEIGHT_M
    if tables is not None:
        out["tables"] = tables
    return out
