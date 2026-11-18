"""PlantCellServer: the grader-owned simulation of one inspection mission and the robot's API on the wire.

One episode = one stored site package: the robot is set down on the airlock's deployment pad, the program inspects the
building with the robot's own sensors (eight tagged instruments, a dose-rate survey mark, one unlisted source) and
brings the robot back. The server owns the MuJoCo model (mission.world.MissionWorld: the site, the robot, its gait
firmware, the sensors, the dose field, the link), steps it only inside `step`, renders the sensors on request, and
scores the mission from its own state (mission.scoring, mission.evidence). Nothing the program sends is a scoring
input except its reports and what the robot physically does.

The calls (mission.facade.PUBLIC_METHODS; robot_api.md is the agent-facing contract):
  describe  mission  get_state  get_camera  get_depth  get_thermal  get_lidar  walk  set_mode  set_body  set_head
  set_lamp  step  report  report_mark  report_source  finish
privileged (the oracle token only): get_state_privileged, reference_run; practice only (allow_reset): reset.

Surface used by the grader (tests/verify.py) and the wire (cellkit.wire):
  server = PlantCellServer(config=grader_config, mode="hidden", privileged=False, log=log, reference_dir=None)
  server.reset(package, seed=None, kind="hidden")     package: a site archive (<name>.tar.xz) or an unpacked folder
  server.handle(method, params) -> result dict | {"error": code[, "detail": text]}      (never raises)
  server.done, server.done_reason, server._end(reason), server.work_wall, server.wall_start, server.program_used,
  server.wall_used, server.steps_after_done
  server.finalize() -> the mission's record (JSON-able; "episode_score" in [0, 1])
  server.trajectory() -> float32 (n, k) state samples (columns in finalize()["trajectory_columns"])
  server.debrief() -> the practice ground's per-item debrief (no total)
  server.close()
Wire encodings: colour frames base64 PNG (RGB, 8 bit); depth base64 16-bit PNG in millimetres, 0 = no reading; thermal
base64 16-bit PNG in centi-kelvin; lidar ranges base64 little-endian float32, NaN = no return. Every other value is
plain JSON. Program time = the wall clock minus the cell's own work (`work_wall`: step and the sensor calls,
encoding included), as in cellkit.wire.
"""

import base64
import hashlib
import importlib.util
import inspect
import json
import math
import os
import time
import traceback

import numpy as np

from . import CELL_VERSION, DATA_DIR  # noqa: F401  (the package puts the vendored library on sys.path)

from mission import paths  # noqa: E402
from mission.config import load_cfg  # noqa: E402
from mission.facade import PUBLIC_METHODS, CallError, Robot as Facade  # noqa: E402
from mission.scoring import INSTRUMENT_TYPES, score as score_mission  # noqa: E402
from mission.world import MissionWorld  # noqa: E402
from site_gen.loader import open_site  # noqa: E402

__all__ = ["PlantCellServer", "PUBLIC_METHODS", "PRIVILEGED_METHODS", "PRACTICE_METHODS", "site_sha256",
           "noise_seed_for"]

PRIVILEGED_METHODS = ("get_state_privileged", "reference_run")
PRACTICE_METHODS = ("reset",)
# calls that change the mission's state: an internal error in one of them ends the mission (the grader re-runs it)
STATE_METHODS = ("walk", "set_mode", "set_body", "set_head", "set_lamp", "step", "report", "report_mark",
                 "report_source", "finish", "reference_run")
SENSOR_METHODS = ("get_camera", "get_depth", "get_thermal", "get_lidar")
# the cell's own work, excluded from the program's time (cellkit.wire)
WORK_METHODS = ("step",) + SENSOR_METHODS
KNOWN_METHODS = frozenset(PUBLIC_METHODS + PRIVILEGED_METHODS + PRACTICE_METHODS)
# grader_config.json keys deep-merged over the cell's own configuration (run settings only)
CONFIG_KEYS = ("time", "budgets")
# the boundary: every argument of a public call is checked here, before anything runs, and a call
# whose arguments fail answers bad_params and changes nothing. Kinds: num (a JSON number, not a boolean, finite, at most
# NUM_MAX in magnitude), num? (num or null), bool (true / false), word (a string of at most WORD_MAX characters),
# ticks (a number from 1 to TICKS_MAX; the cell runs at most max_ticks_per_step of them), value (a report value: null,
# true / false, a number of at most VALUE_NUM_MAX in magnitude, a word, a list of at most 64 or a dict of at most 8 of
# those, nested at most twice), frame_ids (a frame id or a list of at most 3; a frame id is a word of at most 16).
NUM_MAX = 1.0e6
VALUE_NUM_MAX = 1.0e15
TICKS_MAX = 1.0e6
WORD_MAX = 64
ARG_KINDS = {
    "describe": {}, "mission": {}, "get_state": {}, "finish": {}, "get_depth": {}, "get_thermal": {}, "get_lidar": {},
    "get_camera": {"name": "word", "depth": "bool"},
    "walk": {"vx": "num", "vy": "num", "yaw_rate": "num"},
    "set_mode": {"m": "word"},
    "set_body": {"height": "num?", "pitch": "num?"},
    "set_head": {"pan": "num?", "tilt": "num?"},
    "set_lamp": {"on": "bool"},
    "step": {"n_ticks": "ticks"},
    "report": {"tag": "word", "value": "value", "frame_ids": "frame_ids"},
    "report_mark": {"tag": "word", "dose_rate": "num?"},
    "report_source": {"x": "num", "y": "num"},
}
TRAJ_HZ = 4.0
MAX_EVENTS_LOGGED = 3000
_PNG = None


# ============================================================================================== site packages
def site_sha256(path, pkg=None):
    """sha256 of a site archive's bytes; for a site folder, of the members the loader reads from it (each member's
    name and sha256, sorted), so that only the package's content counts."""
    if os.path.isfile(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    members = getattr(pkg, "_m", None) if pkg is not None else None
    if members is None:
        from site_gen.loader import open_site as _open
        members = _open(path)._m
    h = hashlib.sha256()
    for name in sorted(members):
        h.update(("%s %s\n" % (name, hashlib.sha256(members[name]).hexdigest())).encode())
    return h.hexdigest()


def noise_seed_for(sha):
    """The mission's noise seed when none is given: from the site package's sha256 (a 62-bit integer)."""
    return int(sha[:16], 16) >> 2


# ============================================================================================== encodings
def _png_b64(arr):
    """base64 PNG of an 8-bit RGB image or a 16-bit single-channel image (fast zlib level)."""
    import cv2
    a = np.ascontiguousarray(arr)
    if a.ndim == 3:
        a = np.ascontiguousarray(a[..., ::-1])            # cv2 writes BGR
    ok, buf = cv2.imencode(".png", a, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _depth_mm(z):
    z = np.asarray(z, np.float32)
    ok = np.isfinite(z)
    mm = np.where(ok, np.clip(np.round(np.where(ok, z, 0.0) * 1000.0), 1, 65535), 0)
    return mm.astype(np.uint16)


def _thermal_ck(t_c):
    t = np.asarray(t_c, np.float64)
    ok = np.isfinite(t)
    ck = np.where(ok, np.clip(np.round((np.where(ok, t, 0.0) + 273.15) * 100.0), 1, 65535), 0)
    return ck.astype(np.uint16)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def encode(method, res):
    """A facade result as the wire's JSON object."""
    if method in ("get_camera", "get_depth", "get_thermal"):
        out = {k: v for k, v in res.items() if k not in ("image", "depth", "image_c")}
        if "image" in res:
            img = res["image"]
            out["width"], out["height"] = int(img.shape[1]), int(img.shape[0])
            out["image_png_b64"] = _png_b64(img)
        if res.get("depth") is not None:
            d = res["depth"]
            out["width"], out["height"] = int(d.shape[1]), int(d.shape[0])
            out["depth_png_b64"] = _png_b64(_depth_mm(d))
            out["depth_unit_m"] = 0.001
            out["depth_invalid"] = 0
        if "image_c" in res:
            t = res["image_c"]
            out["width"], out["height"] = int(t.shape[1]), int(t.shape[0])
            out["thermal_ck_png_b64"] = _png_b64(_thermal_ck(t))
            out["thermal_unit_k"] = 0.01
        return _jsonable(out)
    if method == "get_lidar":
        r = np.asarray(res["ranges"], "<f4")
        n = int(r.shape[0])
        return _jsonable({"t": res["t"], "beams": n, "ranges_f32_b64": base64.b64encode(r.tobytes()).decode("ascii"),
                          "angle_step_rad": 2.0 * math.pi / max(n, 1), "T_body_lidar": res["T_body_lidar"]})
    return _jsonable(res)


class _Bad(Exception):
    """An argument that fails the boundary check (its text is the reply's detail)."""


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num(name, v, lim):
    if not _is_num(v):
        raise _Bad("%s must be a number" % name)
    if isinstance(v, int):
        if not -lim <= v <= lim:                  # exact for any size of integer (no conversion)
            raise _Bad("%s must be at most %g in magnitude" % (name, lim))
        return v
    if not math.isfinite(v):
        raise _Bad("%s must be finite" % name)
    if abs(v) > lim:
        raise _Bad("%s must be at most %g in magnitude" % (name, lim))
    return v


def _word(name, v, n=WORD_MAX):
    if not isinstance(v, str):
        raise _Bad("%s must be a string" % name)
    if len(v) > n:
        raise _Bad("%s has at most %d characters" % (name, n))
    return v


def _value(v, depth=0):
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        return _word("a reported word", v)
    if _is_num(v):
        return _num("a reported number", v, VALUE_NUM_MAX)
    if isinstance(v, list) and depth < 2:
        if len(v) > 64:
            raise _Bad("a reported list has at most 64 entries")
        return [_value(x, depth + 1) for x in v]
    if isinstance(v, dict) and depth < 2:
        if len(v) > 8:
            raise _Bad("a reported dict has at most 8 entries")
        return {_word("a reported dict key", k): _value(x, depth + 1) for k, x in v.items()}
    raise _Bad("a reported value is a number, a word, true / false, null, a list of at most 64 or a dict of at most 8 "
               "of those (nested at most twice)")


def check_args(method, params):
    """The arguments of a public call, checked (ARG_KINDS); raises _Bad. Unknown names are refused here too."""
    kinds = ARG_KINDS[method]
    for name in params:
        if name not in kinds:
            raise _Bad("%s has no argument %r" % (method, str(name)[:WORD_MAX]))
    out = {}
    for name, v in params.items():
        k = kinds[name]
        if k == "num":
            out[name] = _num(name, v, NUM_MAX)
        elif k == "num?":
            out[name] = None if v is None else _num(name, v, NUM_MAX)
        elif k == "bool":
            if not isinstance(v, bool):
                raise _Bad("%s must be true or false" % name)
            out[name] = v
        elif k == "word":
            out[name] = _word(name, v)
        elif k == "ticks":
            n = _num(name, v, TICKS_MAX)
            if n < 1:
                raise _Bad("%s must be >= 1" % name)
            out[name] = int(n)
        elif k == "value":
            out[name] = _value(v)
        elif k == "frame_ids":
            if v is None:
                out[name] = []
            elif isinstance(v, str):
                out[name] = [_word("a frame id", v, 16)]
            elif isinstance(v, list):
                if len(v) > 3:
                    raise _Bad("at most 3 frame ids")
                out[name] = [_word("a frame id", f, 16) for f in v]
            else:
                raise _Bad("frame_ids is a frame id or a list of at most 3")
    return out


def _cell_overrides(config):
    """The grader's run settings that the cell applies (time, budgets), from grader_config.json."""
    out = {}
    for k in CONFIG_KEYS:
        if isinstance((config or {}).get(k), dict):
            out[k] = json.loads(json.dumps(config[k]))
    b = out.setdefault("budgets", {})
    if "program_time_cap_s" in (config or {}):
        b["program_time_s"] = float(config["program_time_cap_s"])
    if "silence_s" in (config or {}):
        b["silence_s"] = float(config["silence_s"])
    b.pop("max_cited_frames", None)                 # fixed by the API (facade.MAX_CITED), stated for the documents
    return out


# ============================================================================================== the server
class PlantCellServer(object):
    def __init__(self, config=None, mode="hidden", privileged=False, allow_reset=False, render=True, log=None,
                 reference_dir=None, workdir=None):
        """config: the grader's run settings (grader_config.json); 'time' and 'budgets' are deep-merged over the
        cell's configuration. mode: hidden | practice. privileged: the oracle token's calls. reference_dir: the
        folder of the privileged reference program (tests/reference). workdir: kept for the surface (archives are
        read in memory by the site loader)."""
        self.config = dict(config or {})
        self.mode = mode
        self.privileged = bool(privileged)
        self.allow_reset = bool(allow_reset)
        self.render_enabled = bool(render)
        self.log = log or (lambda m: None)
        self.reference_dir = reference_dir
        self.workdir = workdir
        self.cfg = load_cfg(overrides=_cell_overrides(self.config))
        self.world = None
        self.robot = None
        self.done = True
        self.done_reason = None
        self.wall_start = time.time()
        self.wall_used = 0.0
        self.work_wall = 0.0
        self.program_used = 0.0
        self.steps_after_done = 0
        self.calls = {}
        self.step_calls = 0
        self.illegal_calls = 0
        self.internal_errors = 0
        self.internal_errors_live = 0
        self.render_failures = 0
        self.frames_lost = 0
        self.build_wall_s = 0.0
        self.site_name = None
        self.site_sha = None
        self.kind = None
        self.seed = None
        self.reference_result = None

    # ------------------------------------------------------------------ episode
    def reset(self, site, seed=None, kind="hidden", sim_time_limit=None, name=None):
        """Load a site package and start its mission: the robot on the pad, standing. seed: the mission's noise seed
        (None: from the package's sha256). kind: hidden | practice | hall (a label for the record).
        sim_time_limit: None = the cell's (the grader's 'time' setting)."""
        t0 = time.time()
        self.close()
        path = os.fspath(site)
        pkg = open_site(path, shared_dir=paths.SHARED_ASSETS)       # archives are read in memory, checked
        self.site_sha = site_sha256(path, pkg)
        self.site_name = name or os.path.basename(path.rstrip("/")).split(".")[0]
        self.kind = str(kind)
        self.seed = int(seed) if seed is not None else noise_seed_for(self.site_sha)
        self.world = MissionWorld(pkg, seed=self.seed, cfg=self.cfg, sim_time_limit=sim_time_limit)
        self.robot = Facade(self.world)
        self.done = False
        self.done_reason = None
        self.calls = {}
        self.step_calls = 0
        self.illegal_calls = 0
        self.internal_errors = 0
        self.internal_errors_live = 0
        self.render_failures = 0
        self.frames_lost = 0
        self.wall_start = time.time()
        self.wall_used = 0.0
        self.work_wall = 0.0
        self.program_used = 0.0
        self.steps_after_done = 0
        self.reference_result = None
        self.build_wall_s = time.time() - t0
        return {"ok": True}

    def close(self):
        """Release the renderers (each in its own GL context: mission.render_guard) and the unpacked package (safe to
        call twice)."""
        w = self.world
        if w is not None:
            for obj in (w, getattr(w, "pool", None), getattr(getattr(w, "thermal", None), "pool", None)):
                fn = getattr(obj, "close", None)
                try:
                    if callable(fn):
                        fn()
                except Exception:  # noqa: BLE001
                    pass

    def _end(self, reason):
        if self.world is not None and not self.world.done:
            self.world.end(str(reason))
        self.done = True
        if self.done_reason is None:
            self.done_reason = self.world.done_reason if self.world is not None else str(reason)

    # ------------------------------------------------------------------ dispatch
    def handle(self, method, params):
        method = str(method)
        key = method if method in KNOWN_METHODS else "_unknown"
        self.calls[key] = self.calls.get(key, 0) + 1
        t0 = time.time()
        live = not self.done
        try:
            return self._dispatch(method, params)
        except Exception:  # noqa: BLE001 - nothing a request does may crash the wire
            # past the boundary check (check_args) an exception is the cell's own failure; the grader re-runs a mission
            # for it only when the mission was still live (internal_errors_live; tests/verify.py _cell_failed)
            self.internal_errors += 1
            if live:
                self.internal_errors_live += 1
            self.log("[cell] internal error in %s\n%s" % (method, traceback.format_exc()))
            if method in STATE_METHODS:
                self._end("internal_error")
            return {"error": "internal_error"}
        finally:
            if method in WORK_METHODS:
                self.work_wall += time.time() - t0
            if self.world is not None and self.world.done and not self.done:
                self.done = True
                self.done_reason = self.world.done_reason

    def _dispatch(self, method, params):
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return {"error": "bad_params", "detail": "params must be an object"}
        if method in PRACTICE_METHODS:
            if not (self.mode == "practice" and self.allow_reset):
                self.illegal_calls += 1
                return {"error": "forbidden"}
            p = dict(params)
            return self.reset(p.pop("site"), **p)
        if method in PRIVILEGED_METHODS:
            if not self.privileged:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            return getattr(self, method)()
        if method not in PUBLIC_METHODS:
            self.illegal_calls += 1
            return {"error": "forbidden"}
        if self.robot is None:
            return {"error": "no_mission", "detail": "the mission has not started"}
        fn = getattr(self.robot, method)
        try:
            params = check_args(method, params)       # the boundary: nothing runs for a call that fails it
        except _Bad as e:
            return {"error": "bad_params", "detail": str(e)[:200]}
        try:
            inspect.signature(fn).bind(**params)
        except TypeError as e:
            return {"error": "bad_params", "detail": str(e)[:200]}
        if method == "step":
            self.step_calls += 1
            # (the steps after the end are counted by the wire loop that applies the limit, cellkit.wire or the
            # practice ground's serve; counting them here as well halved the documented 50)
        try:
            res = fn(**params)
        except CallError as e:
            return {"error": e.code, "detail": e.detail} if e.detail else {"error": e.code}
        except Exception:
            if method in SENSOR_METHODS:
                self.render_failures += 1
                self.frames_lost += 1
                self.log("[cell] %s not rendered\n%s" % (method, traceback.format_exc()))
                return {"error": "sensor_unavailable"}
            raise
        return encode(method, res)

    # ------------------------------------------------------------------ privileged
    def get_state_privileged(self):
        w = self.world
        if w is None:
            return {"error": "no_mission"}
        p = w.true_pose()
        return _jsonable({"sim_time": round(w.time, 4), "pose_true": p, "dose_rate_gyh": float(w.rate_now),
                          "dose_gy": float(w.dose_gy), "rating_gy": float(w.rating_gy)})

    def reference_run(self):
        """The privileged reference program inside the cell (reference_dir/reference.py, run(server))."""
        if not self.reference_dir:
            return {"error": "no_reference", "detail": "this cell has no reference program"}
        path = os.path.join(self.reference_dir, "reference.py")
        spec = importlib.util.spec_from_file_location("plantcell_reference", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t0 = time.time()
        result = mod.run(self)
        self.reference_result = result
        return _jsonable({"ok": True, "wall_s": round(time.time() - t0, 2), "sim_time": round(self.world.time, 3),
                          "summary": (result or {}).get("summary") if isinstance(result, dict) else None})

    # ------------------------------------------------------------------ the record
    def _scored(self):
        w = self.world
        end = w.end_state()
        sc = score_mission(w.site, w.prior, w.reports, w.frames, w.dose_log, end, w.cfg)
        return sc, end

    def finalize(self):
        """The mission's record (root-only verify output; also the practice debrief's source)."""
        if self.world is None:
            return {"cell_version": CELL_VERSION, "episode_score": 0.0, "step_calls": 0, "calls": dict(self.calls),
                    "illegal_call_count": int(self.illegal_calls), "internal_errors": int(self.internal_errors),
            "internal_errors_live": int(self.internal_errors_live),
                    "done_reason": "no_mission"}
        if not self.done:
            self._end("solve_returned")
        w = self.world
        sc, end = self._scored()
        S = float(sc["S"])
        if not (math.isfinite(S) and 0.0 <= S <= 1.0 + 1e-9):
            self.log("[cell] score %r out of range: recorded as 0" % (S,))
            S = 0.0
        parts = {}
        items = []
        for r in sc["items"]:
            parts[r.get("slot") or r["tag"]] = float(r["credit"])
            row = {k: r.get(k) for k in ("tag", "type", "slot", "weight", "reports", "delivered", "lost", "E", "V",
                                         "A", "error", "tolerance", "credit", "dwell_s", "max_3s_reading_gyh",
                                         "max_3s_reading_near_gyh", "closest_to_reported_m", "approach_failed")
                   if k in r}
            ev = r.get("evidence") or {}
            if ev:
                row["evidence_ok"] = bool(ev.get("ok"))
                row["evidence_frame"] = ev.get("frame")
                row["evidence_why"] = list(ev.get("why", []))[:6]
            items.append(row)
        n_made = len(w.reports)
        events = [[round(float(e["t"]), 2), str(e["type"])] for e in w.events
                  if e.get("type") not in ("frame", "mark_dwell_start", "mark_dwell_end", "controller_status")]
        kinds = {}
        for e in w.events:
            if e.get("type") in ("blocked", "protective_sit", "robot_lost", "dose_limit", "time_limit",
                                 "physics_fault"):
                kinds[e["type"]] = kinds.get(e["type"], 0) + 1
        frames = {k: int(v) for k, v in w.n_frames.items()}
        summary = {
            "cell_version": CELL_VERSION, "site": self.site_name, "site_sha256": self.site_sha, "kind": self.kind,
            "noise_seed": str(self.seed), "done_reason": w.done_reason or self.done_reason,
            "sim_time_s": round(float(w.time), 3), "sim_time_limit_s": float(w.sim_time_limit), "ticks": int(w.tick),
            "episode_score": round(S, 6), "raw_score": float(sc["sum_weighted"]), "R": float(sc["R"]), "parts": parts,
            "items": items, "end": sc["end"], "lost": bool(w.done_reason == "robot_lost"),
            "dose_gy": round(float(w.dose_gy), 6), "dose_fraction": round(float(w.dose_gy / w.rating_gy), 6),
            "distance_m": round(float(w.distance_true), 3),
            "reports_made": n_made, "reports_delivered": sum(1 for r in w.reports if r["status"] == "delivered"),
            "reports_lost": sum(1 for r in w.reports if r["status"] == "lost"),
            "reports": {k: int(v) for k, v in w.n_reports.items()},
            "frames": frames, "camera_frames_total": frames.get("nav", 0) + frames.get("inspect", 0),
            "lidar_sweeps": frames.get("lidar", 0), "thermal_frames": frames.get("thermal", 0),
            "depth_frames": frames.get("depth", 0), "event_counts": kinds,
            # version 2: the holds after `blocked` and the render bundle the world was built with (mission.bundle)
            "blocked_holds": int(getattr(w, "n_holds", 0)), "blocked_hold_s": float(getattr(w, "hold_s", 0.0)),
            "render_bundle": getattr(w, "render_bundle", None),
            "illegal_call_count": int(self.illegal_calls), "internal_errors": int(self.internal_errors),
            "internal_errors_live": int(self.internal_errors_live),
            "calls": dict(self.calls), "step_calls": int(self.step_calls), "steps_after_done": int(self.steps_after_done),
            "program_time_s": float(self.program_used), "cell_work_s": float(self.work_wall),
            "wall_used_s": float(self.wall_used), "render_wall_s": float(sum(w.render_wall.values())),
            "render_wall_by_sensor_s": {k: round(float(v), 3) for k, v in w.render_wall.items()},
            "physics_wall_s": float(w.physics_wall), "render_failures": int(self.render_failures),
            "frames_lost": int(self.frames_lost),
            "render_retries": sum(1 for x in w.render_incidents() if x.get("what") == "segmentation_rerendered"),
            "render_incidents": w.render_incidents(), "build_wall_s": float(self.build_wall_s),
            "start_pose_true": [round(float(v), 4) for v in w.start_pose],
            "final_state_sha256": w.state_digest(), "model_hash": w.model_hash,
            "events_log": events[:MAX_EVENTS_LOGGED],
            "trajectory_columns": self.trajectory_columns(),
        }
        if isinstance(self.reference_result, dict) and self.reference_result.get("summary") is not None:
            summary["reference"] = self.reference_result.get("summary")
        return _jsonable(summary)

    def trajectory_columns(self):
        nq = int(self.world.model.nq) if self.world is not None else 0
        return (["t"] + ["qpos%d" % i for i in range(nq)] + ["mode", "head_pan", "head_tilt", "lamp", "link",
                                                             "true_x", "true_y", "true_z", "true_yaw", "true_roll",
                                                             "true_pitch", "odom_x", "odom_y", "odom_yaw",
                                                             "dose_rate_gyh", "dose_gy"])

    def trajectory(self, hz=TRAJ_HZ):
        """The mission's state samples (for the maintainers' replay), float32 (n, k), about `hz` per second."""
        w = self.world
        if w is None or not w.traj["t"]:
            return np.zeros((0, 0), np.float32)
        T = w.traj
        t = np.asarray(T["t"], np.float64)
        keep = [0]
        nxt = t[0] + 1.0 / hz
        for i in range(1, len(t)):
            if t[i] >= nxt - 1e-9 or i == len(t) - 1:
                keep.append(i)
                nxt = t[i] + 1.0 / hz
        keep = np.asarray(sorted(set(keep)), int)
        cols = [t[keep, None], np.asarray(T["qpos"], np.float64)[keep],
                np.asarray(T["mode"], np.float64)[keep, None], np.asarray(T["head"], np.float64)[keep],
                np.asarray(T["lamp"], np.float64)[keep, None], np.asarray(T["link"], np.float64)[keep, None],
                np.asarray(T["true"], np.float64)[keep], np.asarray(T["odom"], np.float64)[keep],
                np.asarray(T["dose_rate"], np.float64)[keep, None], np.asarray(T["dose"], np.float64)[keep, None]]
        return np.hstack(cols).astype(np.float32)

    def _rule_point(self, tag):
        """An instrument's evidence reference point (mission.evidence EvidenceModel.face_frame), rounded to 1 cm."""
        try:
            ev = self.world.evidence
            rec = ev.items.get(tag)
            if rec is None:
                return None
            c, _ = ev.face_frame(rec)
            return [round(float(v), 2) for v in c[:3]]
        except Exception:  # noqa: BLE001 - a debrief field, never a scoring input
            return None

    # ------------------------------------------------------------------ the practice debrief
    def debrief(self, mission=None, true_pose_hz=None, snapshot=False):
        """What the practice ground's trainers tell the team after a mission: per item delivered or
        not, the evidence accepted or which conditions failed, the value's error, the item's true position; the
        mark's and the source's errors; events; budget use; in the training hall also the true pose at 2 Hz. No
        total, no as-built drawing. snapshot: the same while the mission runs on (it is not ended; the practice
        ground keeps the latest one, sent should the mission's process fail)."""
        if self.world is None:
            return {"error": "no mission"}
        if not self.done and not snapshot:
            self._end("closed")
        w = self.world
        sc, end = self._scored()
        ips = {ip["tag"]: ip for ip in w.site["ips"]}
        rows = []
        for r in sc["items"]:
            ip = ips.get(r["tag"], {})
            row = {"tag": r["tag"], "type": r["type"], "reports_made": int(r["reports"]),
                   "delivered": bool(r["delivered"]), "reports_lost_at_the_end": int(r.get("lost", 0))}
            pos = (ip.get("truth") or {}).get("pos") if r["type"] == "hotspot" else (ip.get("pose") or {}).get("pos")
            if pos is not None:
                row["true_position"] = [round(float(v), 2) for v in pos[:3]]
            if r["type"] in INSTRUMENT_TYPES:
                rp = self._rule_point(r["tag"])
                if rp is not None:
                    # the point the distance, angle and size rules are measured to
                    row["rule_point"] = rp
            if r["type"] in INSTRUMENT_TYPES:
                row["true_value"] = _jsonable(r.get("truth"))
                if r["delivered"]:
                    ev = r.get("evidence") or {}
                    row["evidence"] = ("accepted (frame %s)" % ev.get("frame")) if ev.get("ok") else \
                        {"accepted": False, "conditions_failed": list(ev.get("why", []))[:9]}
                    row["reported_value"] = _jsonable(r.get("value"))
                    row["error"] = r.get("error")
                    if r.get("tolerance") is not None:
                        row["tolerance"] = r.get("tolerance")
            elif r["type"] == "dose_point":
                row["true_dose_rate_gyh_where_the_robot_stood"] = r.get("truth")
                if r["delivered"]:
                    row["reported_gyh"] = r.get("value")
                    row["dwell_s_within_0_30_m"] = r.get("dwell_s")
                    row["dwell_ok"] = bool(r.get("E"))
                    row["relative_error"] = r.get("error")
            elif r["type"] == "hotspot":
                if r["delivered"]:
                    row["reported_position"] = r.get("value")
                    row["error_m"] = r.get("error")
                    # the approach: 3 consecutive seconds with the own dosimeter's mean >= 2.0 Gy/h while
                    # the body centre stayed within 3.0 m of the reported point; which half failed, if one did
                    row["approach_ok"] = bool(r.get("E"))
                    row["own_dosimeter_peak_3s_gyh"] = r.get("max_3s_reading_gyh")
                    row["own_dosimeter_peak_3s_within_3m_of_reported_gyh"] = r.get("max_3s_reading_near_gyh")
                    row["closest_to_reported_m"] = r.get("closest_to_reported_m")
                    dz = w.cfg["dose"]
                    q = {"g": float(dz["e_x_threshold_gyh"]), "m": float(dz.get("e_x_within_m", 3.0)),
                         "s": float(((w.cfg.get("evidence") or {}).get("source") or {}).get(
                             "ex_window_s", dz.get("e_x_window_s", 3.0)))}          # (scoring's own lookup)
                    row["approach_failed"] = {
                        None: None, "rate": "the own dosimeter's %(s)g s mean never reached %(g).1f Gy/h" % q,
                        "near": "the %(s)g s mean reached %(g).1f Gy/h only farther than %(m).1f m from the reported "
                                "point" % q,
                        "rate_and_near": "the robot never stayed %(s)g s within %(m).1f m of the reported point (and "
                                         "the %(s)g s mean never reached %(g).1f Gy/h)" % q}.get(
                        r.get("approach_failed"), r.get("approach_failed"))
            rows.append(_jsonable(row))
        kinds = {}
        for e in w.events:
            if e.get("type") in ("blocked", "protective_sit", "robot_lost", "dose_limit", "time_limit",
                                 "link_up", "link_down"):
                kinds[e["type"]] = kinds.get(e["type"], 0) + 1
        out = {"mission": mission, "site": self.site_name, "ended": w.done_reason or self.done_reason,
               "items": rows,
               "end": {"upright": bool(end["upright"]), "distance_from_pad_centre_m": round(float(end["pad_distance"]), 2)},
               "events": kinds,
               "budget_use": {"sim_time_s": round(float(w.time), 1), "sim_time_limit_s": float(w.sim_time_limit),
                              "dose_gy": round(float(w.dose_gy), 4), "dose_rating_gy": float(w.rating_gy),
                              "frames": {k: int(v) for k, v in w.n_frames.items()},
                              "frame_limits": {k: int(v) for k, v in w.cfg["budgets"]["frames"].items()},
                              "reports": {k: int(v) for k, v in w.n_reports.items()},
                              "program_time_s": round(float(self.program_used), 1)}}
        if true_pose_hz:
            T = w.traj
            t = np.asarray(T["t"], float)
            tr = np.asarray(T["true"], float)
            step = max(1, int(round((1.0 / float(true_pose_hz)) * float(w.cfg["log"]["traj_hz"]))))
            out["true_pose"] = {"hz": float(true_pose_hz), "columns": ["t", "x", "y", "yaw"],
                                "samples": [[round(float(t[i]), 2), round(float(tr[i, 0]), 3),
                                             round(float(tr[i, 1]), 3), round(float(tr[i, 3]), 4)]
                                            for i in range(0, len(t), step)]}
        if snapshot:
            out["ended"] = None
            out["snapshot"] = {"sim_time_s": round(float(w.time), 1),
                               "note": "taken while the mission ran: a report still queued counts as not delivered"}
        return _jsonable(out)
