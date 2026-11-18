"""SubseaCellServer: the grader-owned cell of one run and the vehicle's API on the wire (cell side; DESIGN v1.2
sections 1.1, 1.4-1.6, 3.2, 5; decisions K4, K5, U3, U6, M12; package P09).

One run = one site package (subseacell.site): the vehicle powers up seated in its deployment cage with the marker block
in its jaws, the program does the intervention with the vehicle's own sensors, the cell owns everything else - the
physics model, the render twin, the acoustic model, the firmware, the sensors, the process graph and the plume
(subseacell.world.SiteWorld) - and writes the grader's event log (subseacell.events).  Simulated time passes only in
`step` and in the calls that take simulated time (`sonar_scan`: each ping advances the clock by the ping period).
Nothing the program sends is a scoring input; it only moves the vehicle.

The calls (`vehicle_api.md` is the agent-facing contract; client.py is the program-side proxy):
  PUBLIC (17)      mission clock step finish state set_nav_fix idle set_attitude set_velocity set_hold set_wrench
                   camera_frame set_camera_tilt set_lights sonar_scan gripper_open gripper_close
  PRIVILEGED       get_state_privileged (true vehicle / structure poses), reference_run (the reference program inside
                   the cell) - only when the server was built privileged (the run's token matched, `token_matches`)
  PRACTICE         reset (start this connection's run on a named practice site), instruments (the training tank's
                   instruments) - only in practice mode
Every other name - and a privileged / practice call the server does not grant - answers exactly {"error":
"forbidden"} (no detail; the same bytes on every site and in every state) and is counted (illegal_calls).

Replies: a result dict, or {"error": code[, "detail": text]} with code in ERROR_CODES.  `handle()` never raises:
malformed parameters are bad_params (the detail names only the parameter), anything else is logged and answered
internal_error (inside a call that changes the run it also ends the run, whose state is then scored as it stands; the
grader re-runs a station whose cell failed).  Every reply passes a whitelist (REPLY_KEYS, value vocabularies): no
object identity, valve angle, segment pressure, plume or score field can reach the program; a reply that does not
pass is not sent (internal_error).  Refusal texts depend on public facts only (the parameter, the public budgets, the
run's end reason).

Budgets (subseacell.budget, DESIGN 1.5): simulated time, camera frame units, sonar pings, protective stops are charged
here atomically; program time and silence are the wire's (cellkit.wire.serve_episode reads `work_wall`, the cell's own
time inside step / camera_frame / sonar_scan, and sets wall_start / wall_used / program_used).

Surface used by the grader (P24's verify) and the wire:
  S = SubseaCellServer(values=None, config=None, mode="hidden", privileged=False, render=True, log=None,
                       reference_dir=None, world_factory=None, practice_sites=None, site_guard=None)
  S.reset(site, seed=None, kind="hidden", name=None)       site: a package path / bytes / subseacell.site.Site
  S.handle(method, params) -> result | {"error", "detail"}  (never raises; thread-safe: one call at a time)
  S.done, S.done_reason, S._end(reason), S.work_wall, S.wall_start, S.wall_used, S.program_used, S.steps_after_done
  S.finalize() -> the run's record (JSON-able, no score: subseacell.scoring reads S.event_log() and S.world.truth())
  S.event_log() -> subseacell.events.EventLog;  S.close()
Wire encodings: camera frames base64 PNG (8-bit RGB, filter 0, zlib level 1); sonar intensities base64 of the
(n_pings, n_samples) uint8 array, row-major.  Everything else is plain JSON without NaN.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import importlib.util
import inspect
import json
import math
import numbers
import os
import struct
import threading
import time
import traceback
import zlib

import numpy as np

from . import CELL_VERSION
from . import budget as BU
from . import events as EV
from . import firmware as FW
from . import site as SITE

__all__ = ["SubseaCellServer", "PUBLIC_METHODS", "PRIVILEGED_METHODS", "PRACTICE_METHODS", "ERROR_CODES",
           "REPLY_KEYS", "MISSION_KEYS", "token_matches", "site_sha256", "png_rgb", "ReplyRejected"]

PUBLIC_METHODS = ("mission", "clock", "step", "finish", "state", "set_nav_fix", "idle", "set_attitude", "set_velocity",
                  "set_hold", "set_wrench", "camera_frame", "set_camera_tilt", "set_lights", "sonar_scan",
                  "gripper_open", "gripper_close")
PRIVILEGED_METHODS = ("get_state_privileged", "reference_run")
PRACTICE_METHODS = ("reset", "instruments")
KNOWN_METHODS = frozenset(PUBLIC_METHODS + PRIVILEGED_METHODS + PRACTICE_METHODS)
READ_METHODS = frozenset(("mission", "clock", "state"))                 # never change the run
WORK_METHODS = frozenset(("step", "camera_frame", "sonar_scan", "reference_run"))   # the cell's own time
MOTION_METHODS = FW.MOTION_CALLS
STATE_CHANGING = frozenset(PUBLIC_METHODS) - READ_METHODS | {"reference_run", "reset"}
STATE_PATCH_KEYS = ("mode", "degraded", "gripper", "budget")
ERROR_CODES = ("forbidden", "bad_params", "no_mission", "episode_over", "budget", "sensor_unavailable",
               "internal_error")
FORBIDDEN = {"error": "forbidden"}

# The API's own numbers come from the task values (P24a, decision P21-1: one source, spec/task_values.json):
#   STEP_MAX_S        api.step_max_s         one step() call advances at most this much simulated time
#   SONAR_STEP_DEG    api.sonar_step_deg     the sonar head's mechanical step (one gradian: 400 per turn)
#   NUMBER_MAX        api.number_max         any number inside a call's parameters (larger: bad_params)
#   STATE_RESOLUTION  api.state_resolution   state() values on the wire: decimals per key (0.1 mm, 0.1 mm/s, 1e-7 rad,
#                                            1e-5 rad/s, 1 mV, 1 mA, ...: below every sensor's noise and resolution);
#                                            nav_cov_m2 keeps 4 significant digits
# A server sets them from its own task values when it is built (set_api_values); before that, reading one of these
# module attributes (subseacell.server.STEP_MAX_S, ...) loads the default task values (budget.default_values_path).
API_VALUE_KEYS = {"STEP_MAX_S": "api.step_max_s", "SONAR_STEP_DEG": "api.sonar_step_deg",
                  "NUMBER_MAX": "api.number_max", "STATE_RESOLUTION": "api.state_resolution"}
_API = {}


def _api_from_values(values: dict) -> dict:
    out = {}
    for name, key in API_VALUE_KEYS.items():
        if key not in values:
            raise ValueError(f"the task values have no {key}")
        v = values[key]
        out[name] = {str(k): int(x) for k, x in dict(v).items()} if name == "STATE_RESOLUTION" else float(v)
    turn = 360.0 / out["SONAR_STEP_DEG"]
    if abs(turn - round(turn)) > 1e-9:
        raise ValueError("api.sonar_step_deg must divide a full turn")
    out["PINGS_PER_TURN"] = int(round(turn))
    return out


def set_api_values(values: dict) -> dict:
    """Put the API numbers of these task values in force in this process (one set per process)."""
    new = _api_from_values(values)
    if _API and _API != new:
        raise ValueError("these task values' api numbers differ from the ones already in force in this process")
    _API.update(new)
    globals().update(new)
    return dict(_API)


def _api(name):
    if not _API:
        set_api_values(BU.load_task_values(BU.default_values_path()))
    return _API[name]


def __getattr__(name):
    if name in API_VALUE_KEYS or name == "PINGS_PER_TURN":
        return _api(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


END_REASONS = ("finish", "sim_cap", "disabled", "out_of_bounds", "undervoltage", "eof", "silence", "stalled",
               "protocol", "program_time_cap", "internal_error", "physics_fault", "solve_returned", "closed", "done")

# the work order (mission()); the site package's public "mission" block may hold these keys (P12 writes it)
MISSION_KEYS = ("chart", "cage", "panel_layout", "state_table", "tide_forecast", "telemetry", "debris_report",
                "notes", "budget")
# the scored sites' work order (P24a, decision P22-7): the cell assembles it from the site package's "work_order" block
# (chart, state_table, tide_forecast), the world's cage, the as-designed panel layout and these fixed texts; the keys
# are those of the practice sites' work orders (gen/practice_sites.py: chart, cage, panel_layout, state_table,
# tide_forecast, telemetry, debris_report, notes).  Nothing in it depends on the site's hidden state.
WORK_ORDER_KEYS = ("chart", "state_table", "tide_forecast", "cage")     # what a site package's work_order may hold
CHART_NOTE = "nominal: the manifold as built is within a few metres and degrees of the charted position and heading"
# amendment 02 (package P38, decision AM2-7 (1)): the chart's as-laid lines (the site package's work_order.chart.lines,
# written by the site generator; the practice sister site's mission block carries the same fields)
CHART_LINE_KEYS = ("name", "tie_in", "outer_diameter_m", "pipe", "route_xy_m", "node_spacing_m", "mattresses_xy_m",
                   "mattress_size_m", "continues")
LINES_NOTE = ("as-laid survey of the lines at the tie-ins, in the chart's frame: the same survey as the manifold's "
              "charted position, so the same offset and heading error; route nodes every 5 m from the tie-in; each line "
              "continues beyond its last node (the edge of the survey area); mattress positions approximate")
LAYOUT_NOTE = "as designed (panel standard, sheet F-2); the tag plates on the panel govern"
TELEMETRY_TEXT = "manifold pressure falling; bubbles reported at the surface over the drilling centre"
DEBRIS_REPORT_TEXT = ("an object stands on the mud mat in front of the panel's port row; it can be any of the "
                      "register's classes D1 to D4")
NOTES_FACILITY = "drilling centre: six-slot production manifold"
# keys that never appear anywhere inside the work order (the site's true state; checked recursively at reset)
MISSION_DENY = frozenset(("leak_branch", "flange", "src_S", "src_W", "theta0_deg", "breakaway_Nm", "running_Nm",
                          "hydro", "plume", "clutter", "flow", "leak_p0_bar", "leak_tau_s", "as_built", "code_fouled",
                          "fouling", "sensor_seed", "water_c550", "acoustic", "angle_deg", "pressure_bar",
                          "contact_window", "gm_seed", "seed", "score", "isolation_set"))

_MOTION_KEYS = ("accepted", "mode", "reason", "protect_phase", "remaining_s")
_ACK_KEYS = ("accepted", "reason")
REPLY_KEYS = {
    "mission": MISSION_KEYS,
    "clock": ("t", "sim_time_limit_s", "program_time_used_s", "program_time_limit_s"),
    "step": ("t", "dt", "done", "end_reason"),
    "finish": ("t", "done", "end_reason"),
    "state": FW.STATE_KEYS,
    "set_nav_fix": _ACK_KEYS,
    "idle": _MOTION_KEYS, "set_attitude": _MOTION_KEYS, "set_velocity": _MOTION_KEYS, "set_hold": _MOTION_KEYS,
    "set_wrench": _MOTION_KEYS,
    "camera_frame": ("t", "frame_index", "width", "height", "image_png_b64"),
    "set_camera_tilt": _ACK_KEYS,
    "set_lights": ("accepted", "level", "reason"),
    "sonar_scan": ("t_start", "t_end", "n_pings", "n_samples", "range_m", "gain", "sample_period_s",
                   "speed_of_sound_m_s", "angle_deg", "t", "nav_position_m", "depth_m", "heading_rad", "quat_wxyz",
                   "valid", "intensity_u8_b64", "done"),
    "gripper_open": _ACK_KEYS, "gripper_close": _ACK_KEYS,
}
_VOCAB = {"mode": FW.MODES, "reason": ("protective_stop", "episode_over", "disabled", "out_of_bounds", "undervoltage"),
          "protect_phase": (None, "stop", "hold", "surfacing", "disabled"), "nav_status": FW.NAV_STATUS,
          "gripper": FW.GRIPPER_STATES, "end_reason": (None,) + END_REASONS}


class ReplyRejected(RuntimeError):
    """A reply that does not pass the whitelist (never sent; the call answers internal_error)."""


class _BadParams(ValueError):
    pass


# ============================================================================================================ helpers
def token_matches(token_text, sha256_hex) -> bool:
    """The privileged run's test: sha256 of the stripped token text equals the stored hash (constant-time compare).
    False for a missing token or hash."""
    if not token_text or not sha256_hex:
        return False
    t = token_text.decode("utf-8", "replace") if isinstance(token_text, (bytes, bytearray)) else str(token_text)
    h = hashlib.sha256(t.strip().encode("utf-8")).hexdigest()
    return hmac.compare_digest(h, str(sha256_hex).strip().lower())


def site_sha256(site) -> str:
    """sha256 of a site package's bytes (path or bytes)."""
    if isinstance(site, (bytes, bytearray)):
        return hashlib.sha256(bytes(site)).hexdigest()
    h = hashlib.sha256()
    with open(os.fspath(site), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def png_rgb(img, level: int = 1) -> bytes:
    """8-bit RGB PNG: no interlace, filter 0 on every row, one IDAT, no ancillary chunk (deterministic bytes)."""
    a = np.ascontiguousarray(img, np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("png_rgb wants (H, W, 3) uint8")
    h, w, _ = a.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), a.reshape(h, w * 3)], axis=1).tobytes()
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(raw, int(level))) + _chunk(b"IEND", b""))


def _is_real(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def _float(x, name):
    try:
        return float(x)
    except (OverflowError, ValueError, TypeError):
        raise _BadParams(f"{name} must be a finite number") from None


def _real(x, name):
    """A JSON number (not a boolean), finite, |x| <= NUMBER_MAX."""
    if not _is_real(x):
        raise _BadParams(f"{name} must be a number")
    v = _float(x, name)
    lim = _api("NUMBER_MAX")
    if not math.isfinite(v) or abs(v) > lim:
        raise _BadParams(f"{name} must be finite and at most {lim:g} in magnitude")
    return v


def _opt_real(x, name):
    return None if x is None else _real(x, name)


def _vec(x, n, name):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)) or len(x) != n:
        raise _BadParams(f"{name} must be {n} numbers")
    return [_real(v, name) for v in x]


def _opt_vec(x, n, name):
    return None if x is None else _vec(x, n, name)


def _free(x):
    """`free`: six booleans (or 0 / 1), a list of axis names, or one axis name."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, str):
        x = [x]
    if not isinstance(x, (list, tuple)) or not x or len(x) > 6:
        raise _BadParams("free must be six booleans (Fx, Fy, Fz, Mx, My, Mz) or a list of axis names")
    if all(isinstance(f, str) for f in x):
        if any(f not in FW.AXES for f in x):
            raise _BadParams("free: unknown axis name; use %s" % ", ".join(FW.AXES))
        return list(x)
    if len(x) == 6 and all(isinstance(f, (bool, np.bool_)) or (_is_real(f) and f in (0, 1)) for f in x):
        return [bool(f) for f in x]
    raise _BadParams("free must be six booleans (Fx, Fy, Fz, Mx, My, Mz) or a list of axis names")


def _intval(x, name):
    if not _is_real(x) or not math.isfinite(_float(x, name)) or _float(x, name) != int(x):
        raise _BadParams(f"{name} must be an integer")
    return int(x)


def _jsonable(x):
    return EV._jsonable(x)


def _q(x, d):
    if x is None:
        return None
    if isinstance(x, float):
        return round(x, d)
    if isinstance(x, list):
        return [_q(v, d) for v in x]
    if isinstance(x, dict):
        return {k: _q(v, d) for k, v in x.items()}
    return x


def quantize_state(st: dict) -> dict:
    """The firmware's state() reply at the wire resolutions (STATE_RESOLUTION); same keys, same order."""
    out = {}
    res = _api("STATE_RESOLUTION")
    for k, v in st.items():
        d = res.get(k)
        if d is not None:
            out[k] = _q(v, d)
        elif k == "nav_cov_m2" and v is not None:
            out[k] = [[None if c is None else float("%.4g" % c) for c in r] for r in v]
        else:
            out[k] = v
    return out


def _scan_deny(obj, path="mission"):
    """True-state keys anywhere in a work-order tree -> list of paths."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in MISSION_DENY:
                out.append(f"{path}.{k}")
            out += _scan_deny(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += _scan_deny(v, f"{path}[{i}]")
    return out


def _chart_lines(lines) -> list:
    """The work order's as-laid lines (amendment 02): a list of objects with exactly the keys CHART_LINE_KEYS."""
    if not isinstance(lines, list) or not all(isinstance(x, dict) and set(x) == set(CHART_LINE_KEYS) for x in lines):
        raise SITE.SitePackageError("the work order's chart.lines must be objects with the keys %s" % (CHART_LINE_KEYS,))
    return copy.deepcopy(lines)


def _required(fn) -> frozenset:
    sig = inspect.signature(fn)
    return frozenset(n for n, p in sig.parameters.items()
                     if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))


# ============================================================================================================ server
class SubseaCellServer(object):
    def __init__(self, values=None, config=None, mode="hidden", privileged=False, allow_reset=None, render=True,
                 log=None, reference_dir=None, world_factory=None, practice_sites=None, site_guard=None,
                 mesh_root=None, workdir=None):
        """values: the flat task values ({key: value}, spec/task_values.json; None = load from
        budget.default_values_path()).  config: the grader's run settings (grader_config.json): a "budgets" block
        overrides the budget values.  mode: hidden | practice.  privileged: the run's token matched (token_matches).
        world_factory(site, values, sensor_seed, render) -> a world with SiteWorld's surface (practice facilities, P20);
        None = subseacell.world.SiteWorld.  practice_sites: {name: package path} a practice connection may reset to;
        site_guard(site_obj) -> bool: last check before a practice site is loaded (P20's label / path guard)."""
        if values is None:
            values = BU.load_task_values(BU.default_values_path())
        self.values = dict(values)
        if tuple(self.values.get("api.public_calls", PUBLIC_METHODS)) != PUBLIC_METHODS:
            raise ValueError("api.public_calls of the task values differs from the server's PUBLIC_METHODS")
        set_api_values(self.values)
        self.config = dict(config or {})
        self.limits = BU.Limits.from_values(self.values).override(self.config.get("budgets"))
        self.camera_sizes = tuple(tuple(int(v) for v in s) for s in self.values.get("api.camera_sizes",
                                                                                   ((640, 480), (320, 240))))
        self.mode = str(mode)
        self.privileged = bool(privileged)
        self.allow_reset = (self.mode == "practice") if allow_reset is None else bool(allow_reset)
        self.render_enabled = bool(render)
        self.log = log or (lambda m: None)
        self.reference_dir = reference_dir
        self.world_factory = world_factory
        self.practice_sites = dict(practice_sites or {})
        self.site_guard = site_guard
        self.mesh_root = mesh_root
        self.workdir = workdir
        self._lock = threading.RLock()
        self._api = {m: getattr(self, "_api_" + m) for m in PUBLIC_METHODS}
        self._req = {m: _required(fn) for m, fn in self._api.items()}
        self._sig = {m: inspect.signature(fn) for m, fn in self._api.items()}
        self.world = None
        self.budget = None
        self.elog = None
        self._clear_run()
        self.calls = {}
        self.illegal_calls = 0
        self.internal_errors = 0

    # ------------------------------------------------------------------------------------------ run state
    def _clear_run(self):
        self.done = self.world is None
        self.done_reason = None
        self.wall_start = time.time()
        self.wall_used = 0.0
        self.program_used = 0.0
        self.steps_after_done = 0
        self.render_failures = 0
        self.frames_lost = 0
        self.frame_index = 0
        self.reference_result = None
        self.cell_fault = None
        self._mut = 0
        self._state_cache = None
        self._nav_dirty = False
        self._last_end = None

    @property
    def work_wall(self) -> float:
        return 0.0 if self.budget is None else self.budget.work_wall

    @work_wall.setter
    def work_wall(self, value):
        if self.budget is not None:
            self.budget.reset_work(float(value))

    def reset(self, site, seed=None, kind="hidden", name=None):
        """Load a site package and start its run.  seed: the sensors' noise seed (None: the site's own)."""
        t0 = time.perf_counter()
        with self._lock:
            self.close()
            if isinstance(site, SITE.Site):
                obj, sha = site, hashlib.sha256(site.to_package()).hexdigest()
            else:
                if isinstance(site, (bytes, bytearray)):
                    blob = bytes(site)
                else:
                    with open(os.fspath(site), "rb") as f:
                        blob = f.read()
                obj, sha = SITE.load(blob), hashlib.sha256(blob).hexdigest()
            self.site_sha = sha
            sd = obj.dict()
            self.site_label = sd.get("label")
            self.site_name = name or self.site_label
            self.kind = str(kind)
            if self.world_factory is not None:
                w = self.world_factory(obj, self.values, seed, self.render_enabled)
            else:
                from .world import SiteWorld
                w = SiteWorld(obj, config=self.values, cameras=self.render_enabled, mesh_root=self.mesh_root,
                              sensor_seed=seed, camera_sizes=self.camera_sizes)
            self.world = w
            self.seed = int(w.sensor_seed)
            if int(w.protect.cfg.max_stops) != int(self.limits.protective_stops):
                raise ValueError("the firmware's stop limit differs from the protective-stop budget")
            self.phys_dt = float(w.sim.dt)
            self.ctrl_dt = float(w.fw.dt)
            self.k_ctrl = int(w.k_ctrl)
            self.budget = BU.Budget(self.limits, self.phys_dt)
            self.elog = EV.EventLog(w, self.values, judge=self.render_enabled)
            self._mission = self._build_mission(sd)
            self._clear_run()
            self.done = False
            self.build_wall_s = time.perf_counter() - t0
            return {"ok": True}

    def close(self):
        w = self.world
        if w is not None:
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass

    def _end(self, reason):
        if not self.done:
            self.done = True
            self.done_reason = str(reason)
            if self.elog is not None:
                self.elog.cell("end", reason=str(reason))

    # ------------------------------------------------------------------------------------------ dispatch
    def handle(self, method, params):
        try:
            method = method if isinstance(method, str) else str(method)
        except Exception:  # noqa: BLE001
            method = "?"
        with self._lock:
            t0 = time.perf_counter()
            key = method if method in KNOWN_METHODS else "_unknown"
            self.calls[key] = self.calls.get(key, 0) + 1
            try:
                return self._dispatch(method, params)
            except Exception:  # noqa: BLE001 - nothing a request does may crash the wire
                self.internal_errors += 1
                self.log("[cell] internal error in %s\n%s" % (method[:64], traceback.format_exc()))
                if method in STATE_CHANGING and self.world is not None:
                    self._end("internal_error")
                return {"error": "internal_error"}
            finally:
                if method in WORK_METHODS and self.budget is not None:
                    self.budget.add_work(time.perf_counter() - t0)
                w = self.world
                if w is not None and getattr(w, "ended", False) and not self.done:
                    self._end(w.end_reason)

    def _forbid(self, method):
        self.illegal_calls += 1
        if self.elog is not None:
            self.elog.api("forbidden", method=method[:64])
        return dict(FORBIDDEN)

    def _dispatch(self, method, params):
        if method in PRACTICE_METHODS:
            if not (self.mode == "practice" and self.allow_reset):
                return self._forbid(method)
            return self._practice(method, params)
        if method in PRIVILEGED_METHODS:
            if not self.privileged:
                return self._forbid(method)
            if self.world is None:
                return {"error": "no_mission", "detail": "the run has not started"}
            if not (params is None or (isinstance(params, dict) and not params)):
                return {"error": "bad_params", "detail": "this call takes no parameters"}
            return getattr(self, method)()
        if method not in self._api:
            return self._forbid(method)
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return self._bad(method, "params must be an object")
        if self.world is None:
            return {"error": "no_mission", "detail": "the run has not started"}
        if params:
            params = {str(k): v for k, v in params.items()}
            try:
                self._sig[method].bind(**params)
            except TypeError as e:
                return self._bad(method, str(e)[:200])
        elif self._req[method]:
            return self._bad(method, "missing parameter(s): %s" % ", ".join(sorted(self._req[method])))
        if method not in READ_METHODS:
            self._mut += 1
        try:
            res = self._api[method](**params)
        except (_BadParams, FW.CallError) as e:
            return self._bad(method, str(e)[:200])
        if not (isinstance(res, dict) and "error" in res):
            self._check_reply(method, res)
        return res

    def _bad(self, method, detail):
        if self.elog is not None:
            self.elog.api("bad_params", method=method[:64])
        return {"error": "bad_params", "detail": detail}

    def _check_reply(self, method, res):
        allowed = REPLY_KEYS[method]
        if method == "state":
            if tuple(res) != FW.STATE_KEYS:
                raise ReplyRejected("state() keys differ from the key table")
            if set(res["budget"]) - set(BU.SNAPSHOT_KEYS):
                raise ReplyRejected("state() budget keys")
            for k in ("mode", "protect_phase", "nav_status", "gripper"):
                if res[k] not in _VOCAB[k]:
                    raise ReplyRejected(f"state() {k} value")
            return
        extra = set(res) - set(allowed)
        if extra:
            raise ReplyRejected(f"{method}() reply keys {sorted(extra)}")
        for k in ("mode", "reason", "protect_phase", "end_reason"):
            if k in res and res[k] not in _VOCAB[k]:
                raise ReplyRejected(f"{method}() {k} value")

    # ------------------------------------------------------------------------------------------ time
    def _t(self) -> float:
        return float(self.world.sim.time)

    def _advance_steps(self, n: int) -> int:
        """Run up to n physics steps within the simulated-time budget; ends the run at the cap or when the world ends."""
        W, B = self.world, self.budget
        g = B.grant("sim_steps", int(n))
        ran = 0
        if g > 0:
            try:
                ran = W.advance(g * self.phys_dt)
            except FloatingPointError:
                ran = W.n_steps - (B.used["sim_steps"] - g)
                B.refund("sim_steps", g - ran)
                self.cell_fault = "physics_fault"
                self._end("physics_fault")
                raise
            if ran < g:
                B.refund("sim_steps", g - ran)
        self._state_cache = None
        if getattr(W, "ended", False):
            self._end(W.end_reason)
        elif B.left("sim_steps") <= 0:
            self._end("sim_cap")
        return ran

    # ------------------------------------------------------------------------------------------ the 17 calls
    def _api_mission(self):
        return self._mission

    def _api_clock(self):
        used = max(0.0, (time.time() - self.wall_start) - self.work_wall)
        return {"t": self._t(), "sim_time_limit_s": float(self.limits.sim_time_s),
                "program_time_used_s": round(used, 3), "program_time_limit_s": float(self.limits.program_time_s)}

    def _api_step(self, dt):
        dt = _real(dt, "dt")
        smax = _api("STEP_MAX_S")
        if not (0.0 < dt <= smax):
            raise _BadParams(f"dt must be in (0, {smax:g}] s")
        if self.done:
            return {"t": self._t(), "dt": 0.0, "done": True, "end_reason": self.done_reason}
        n_ctrl = max(1, int(round(dt / self.ctrl_dt)))
        ran = self._advance_steps(n_ctrl * self.k_ctrl)
        self._mut += 1
        # the state() reply of the new time, built now (cell work, not program time): the program's next state() is a
        # lookup (records/P2_P09.md: the round trip after a step drops by the 60-80 us of the build)
        self._api_state()
        return {"t": self._t(), "dt": round(ran * self.phys_dt, 9), "done": bool(self.done),
                "end_reason": self.done_reason}

    def _api_finish(self):
        self._end("finish")
        return {"t": self._t(), "done": True, "end_reason": self.done_reason}

    def _api_state(self):
        """The firmware's state() reply.  Built in full when simulated time has passed (at the end of step() /
        sonar_scan(), inside the cell's work) or after set_nav_fix; after the other calls without a step only the keys
        such a call can change (STATE_PATCH_KEYS: firmware mode, degraded flag, gripper, budget) are refreshed on the
        last full reply - everything else in it is a function of the simulated state, which did not move.
        tests/test_server.py checks the shortcut against a full rebuild after every kind of call."""
        W = self.world
        c = self._state_cache
        if c is not None and c[0] == W.n_steps:
            if c[1] == self._mut:
                return c[2]
            if not self._nav_dirty:
                st = dict(c[2])
                fw = W.fw
                st["mode"] = fw.mode
                st["degraded"] = bool(fw.degraded)
                st["gripper"] = W.gripper_public()
                self.budget.set_used("protective_stops", int(W.protect.count))
                st["budget"] = self.budget.snapshot()
                self._check_reply("state", st)
                self._state_cache = (W.n_steps, self._mut, st)
                return st
        st = self.state_full()
        self._check_reply("state", st)
        self._state_cache = (W.n_steps, self._mut, st)
        self._nav_dirty = False
        return st

    def state_full(self) -> dict:
        """A full rebuild of the state() reply (no shortcut)."""
        W = self.world
        self.budget.set_used("protective_stops", int(W.protect.count))
        return quantize_state(W.state(budget=self.budget.snapshot()))

    def _refused(self):
        return {"accepted": False, "reason": "episode_over"}

    def _api_set_nav_fix(self, x, y):
        x, y = _real(x, "x"), _real(y, "y")
        if self.done:
            return self._refused()
        r = self.world.fw.set_nav_fix(x, y)
        self._nav_dirty = True
        self.elog.api("set_nav_fix", x=x, y=y)
        return {"accepted": bool(r.get("accepted", True))}

    def _motion(self, name, **kw):
        if self.done:
            r = self._refused()
        else:
            r = getattr(self.world.fw, name)(**kw)
        self.elog.motion(name, r)
        return r

    def _api_idle(self):
        return self._motion("idle")

    def _api_set_attitude(self, roll=0.0, pitch=0.0, heading=None, depth=None, thrust=(0.0, 0.0, 0.0), quat=None):
        return self._motion("set_attitude", roll=_real(roll, "roll"), pitch=_real(pitch, "pitch"),
                            heading=_opt_real(heading, "heading"), depth=_opt_real(depth, "depth"),
                            thrust=_vec(thrust, 3, "thrust"), quat=_opt_vec(quat, 4, "quat"))

    def _api_set_velocity(self, u, v, depth, heading):
        return self._motion("set_velocity", u=_real(u, "u"), v=_real(v, "v"), depth=_real(depth, "depth"),
                            heading=_real(heading, "heading"))

    def _api_set_hold(self, x, y, depth, heading):
        return self._motion("set_hold", x=_real(x, "x"), y=_real(y, "y"), depth=_real(depth, "depth"),
                            heading=_real(heading, "heading"))

    def _api_set_wrench(self, force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0), free=FW.ALL, frame="body", quat=None,
                        depth=None):
        if not isinstance(frame, str) or frame not in ("body", "world"):
            raise _BadParams("frame must be 'body' or 'world'")
        return self._motion("set_wrench", force=_vec(force, 3, "force"), torque=_vec(torque, 3, "torque"),
                            free=_free(free), frame=frame, quat=_opt_vec(quat, 4, "quat"),
                            depth=_opt_real(depth, "depth"))

    def _size(self, size):
        if isinstance(size, str):
            parts = size.lower().split("x")
            if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
                raise _BadParams("size must be [width, height] or 'WIDTHxHEIGHT'")
            sz = (int(parts[0]), int(parts[1]))
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            sz = (_intval(size[0], "size"), _intval(size[1], "size"))
        else:
            raise _BadParams("size must be [width, height]")
        if sz not in self.camera_sizes:
            raise _BadParams("size must be one of %s" % ", ".join("%dx%d" % s for s in self.camera_sizes))
        return sz

    def _api_camera_frame(self, size=(640, 480)):
        sz = self._size(size)
        if self.done:
            return {"error": "episode_over", "detail": "the run is over (%s)" % self.done_reason}
        if not self.render_enabled:
            return {"error": "sensor_unavailable", "detail": "this cell has no camera"}
        cost = self.budget.frame_cost(sz)
        if not self.budget.take("frame_units", cost):
            self.elog.api("budget", method="camera_frame", size=list(sz))
            return {"error": "budget", "detail": "the camera frame budget (%g units) does not cover this frame"
                    % self.limits.camera_frame_units}
        try:
            fr = self.world.frame(sz)
            png = png_rgb(fr.rgb)
        except Exception:  # noqa: BLE001 - a renderer failure is the cell's, not charged to the program
            self.budget.refund("frame_units", cost)
            self.render_failures += 1
            self.frames_lost += 1
            self.log("[cell] frame not rendered\n%s" % traceback.format_exc())
            return {"error": "sensor_unavailable", "detail": "the frame could not be taken"}
        self.frame_index += 1
        return {"t": self._t(), "frame_index": int(self.frame_index), "width": int(fr.rgb.shape[1]),
                "height": int(fr.rgb.shape[0]), "image_png_b64": base64.b64encode(png).decode("ascii")}

    def _api_set_camera_tilt(self, deg):
        v = _real(deg, "deg")
        if self.done:
            return self._refused()
        self.world.set_camera_tilt(math.radians(v))
        self.elog.api("set_camera_tilt", deg=v)
        return {"accepted": True}

    def _api_set_lights(self, level):
        v = min(max(_real(level, "level"), 0.0), 1.0)
        if self.done:
            return self._refused()
        self.world.set_lights(v)
        self.elog.api("set_lights", level=v)
        return {"accepted": True, "level": v}

    def _api_sonar_scan(self, start_deg, stop_deg, range_m, gain=1):
        a, b = _real(start_deg, "start_deg"), _real(stop_deg, "stop_deg")
        r = _real(range_m, "range_m")
        W = self.world
        cfg = W.sonar.cfg
        if not (cfg.min_range_m <= r <= cfg.max_range_m):
            raise _BadParams(f"range_m must be in [{cfg.min_range_m:g}, {cfg.max_range_m:g}] m")
        g = _intval(gain, "gain")
        if g not in (0, 1, 2):
            raise _BadParams("gain must be 0 (low), 1 (normal) or 2 (high)")
        if self.done:
            return {"error": "episode_over", "detail": "the run is over (%s)" % self.done_reason}
        step, turn = _api("SONAR_STEP_DEG"), _api("PINGS_PER_TURN")
        g0 = int(round(a / step)) % turn
        g1 = int(round(b / step)) % turn
        span = (g1 - g0) % turn
        n, sector = (turn, (g0, g0 + turn)) if span == 0 else (span + 1, (g0, g1))
        if not self.budget.take("sonar_pings", n):
            self.elog.api("budget", method="sonar_scan", n=n)
            return {"error": "budget", "detail": "the sonar ping budget (%d) does not cover this scan (%d pings)"
                    % (self.limits.sonar_pings, n)}
        W.sonar_configure(range_m=r, sector_grad=sector, gain_index=g, step_grad=1)
        period = float(W.ping_period())             # the head's timing model; the clock follows it exactly on average
        ns = int(cfg.n_samples)
        data = np.zeros((n, ns), np.uint8)
        ang, ts, pos, dep, hdg, quat, valid = [], [], [], [], [], [], []
        t_start = self._t()
        k = 0
        sp = None
        gp = W.gripper_public()
        try:
            while k < n and not self.done:
                st = W.fw.state(gripper=gp, budget={})
                p = W.ping()
                data[k, :] = np.asarray(p.data, np.uint8)[:ns]
                ang.append(round(int(p.angle_grad) * step, 6))
                ts.append(float(p.t))
                pos.append(_q(st["nav_position_m"], 6))
                dep.append(_q(st["depth_m"], 6))
                hdg.append(_q(st["heading_rad"], 9))
                quat.append(_q(st["quat_wxyz"], 9))
                valid.append(bool(p.valid))
                sp = float(p.sample_period_s)
                k += 1
                due = int(round(k * period / self.phys_dt)) - int(round((k - 1) * period / self.phys_dt))
                self._advance_steps(max(1, due))
        finally:
            if k < n:
                self.budget.refund("sonar_pings", n - k)
        self.elog.api("sonar_scan", start_grad=g0, sector=list(sector), n=n, done_pings=k, range_m=r, gain=g)
        self._mut += 1
        self._api_state()
        return {"t_start": t_start, "t_end": self._t(), "n_pings": k, "n_samples": ns, "range_m": float(cfg.range_m),
                "gain": g, "sample_period_s": sp, "speed_of_sound_m_s": float(cfg.speed_of_sound_cfg),
                "angle_deg": ang, "t": ts, "nav_position_m": pos, "depth_m": dep, "heading_rad": hdg,
                "quat_wxyz": quat, "valid": valid, "intensity_u8_b64": base64.b64encode(data[:k].tobytes()).decode("ascii"),
                "done": bool(self.done)}

    def _api_gripper_open(self):
        if self.done:
            return self._refused()
        ok = bool(self.world.gripper_open())
        self.elog.api("gripper_open", accepted=ok)
        return {"accepted": ok}

    def _api_gripper_close(self):
        if self.done:
            return self._refused()
        ok = bool(self.world.gripper_close())
        self.elog.api("gripper_close", accepted=ok)
        return {"accepted": ok}

    # ------------------------------------------------------------------------------------------ the work order
    def _cage_block(self) -> dict:
        cg = self.world.cage
        return {"position_m": [float(cg["pos_w"][0]), float(cg["pos_w"][1])], "heading_rad": float(cg["yaw"]),
                "deck_z_m": float(cg["deck_top_w"]), "deck_above_seabed_m": float(cg["deck_top_m"])}

    def _as_designed_layout(self) -> dict:
        """The work order's panel_layout: the AS-DESIGNED panel of the panel standard (column i from the viewer's left
        serves branch i, rows BV, PV, TV bottom up, the ports in drawing order), from the skid's design parameters -
        never the as-built mapping of the site (the tag plates on the panel govern).  Same keys as the practice
        sites' layout (gen/practice_sites.py::_layout)."""
        P = self.world.site.skid["params"]
        n = int(P["columns"])
        return {"rows_bottom_up": [[f"{r}{j + 1}" for j in range(n)] for r in P["rows"]],
                "row_heights_m": [float(x) for x in P["row_heights"]], "column_pitch_m": float(P["pitch"]),
                "ports_left_to_right": [str(x) for x in P["ports"]], "note": LAYOUT_NOTE}

    def _build_mission(self, sd: dict) -> dict:
        """The work order (mission()).  A site package's public "mission" block (practice facilities) is passed as it
        is.  A scored site (P12's package, decision P22-7) carries a "work_order" block (chart, state_table,
        tide_forecast; its cage is the world's): the cell assembles the work order from it, the world's cage, the
        as-designed panel layout and the fixed telemetry, dropped-object report and notes - the same keys as the
        practice sites' work orders.  Amendment 02: the chart's as-laid lines (chart.lines, with chart.lines_note) when
        the package's work_order.chart has them.  A package with neither (old stand-in sites) gets chart, state table and cage.
        Plus the run's budgets."""
        m = sd.get("mission")
        if m is not None:
            if not isinstance(m, dict):
                raise SITE.SitePackageError("the site's mission block must be an object")
            bad = set(m) - set(MISSION_KEYS)
            if bad or "budget" in m:
                raise SITE.SitePackageError(f"mission block: keys not in the work order: {sorted(bad | ({'budget'} & set(m)))}")
            out = copy.deepcopy(m)
        elif sd.get("work_order") is not None:
            wo = sd["work_order"]
            if not isinstance(wo, dict) or set(wo) - set(WORK_ORDER_KEYS) or not {"chart", "state_table",
                                                                                   "tide_forecast"} <= set(wo):
                raise SITE.SitePackageError("the site's work_order block must hold chart, state_table, tide_forecast")
            chart = {k: copy.deepcopy(wo["chart"][k]) for k in ("origin_xy", "psi")}
            chart["note"] = CHART_NOTE
            if wo["chart"].get("lines") is not None:
                chart["lines"] = _chart_lines(wo["chart"]["lines"])
                chart["lines_note"] = LINES_NOTE
            out = {"chart": chart, "cage": self._cage_block(), "panel_layout": self._as_designed_layout(),
                   "state_table": copy.deepcopy(wo["state_table"]), "tide_forecast": copy.deepcopy(wo["tide_forecast"]),
                   "telemetry": TELEMETRY_TEXT, "debris_report": DEBRIS_REPORT_TEXT,
                   "notes": {"facility": NOTES_FACILITY, "water_depth_m": round(float(sd["seabed_depth_m"]), 1)}}
        else:
            out = {}
            if "chart" in sd:
                out["chart"] = copy.deepcopy(sd["chart"])
            if "state_table" in sd:
                out["state_table"] = copy.deepcopy(sd["state_table"])
            out["cage"] = self._cage_block()
        out["budget"] = BU.Budget(self.limits, self.phys_dt).public_limits()
        bad = _scan_deny(out)
        if bad:
            raise SITE.SitePackageError(f"the work order carries true-state keys: {bad[:5]}")
        return _jsonable(out)

    # ------------------------------------------------------------------------------------------ privileged
    def get_state_privileged(self):
        """True poses for the reference program: the vehicle (and its tool point), the structure, the panel frame, the
        cage, the mudline.  No valve angle, pressure, plume or leak."""
        W = self.world
        d = W.d
        st = W.sim.state()
        p_tcp, R_tcp = W.tcp_w()
        cg = W.cage
        return _jsonable({"t": self._t(),
                          "vehicle": {"pos_w": st.pos_w, "quat_wxyz": st.quat_wb, "vel_b": st.vel_b,
                                      "omega_b": st.omega_b},
                          "tcp": {"pos_w": p_tcp, "R_w": R_tcp},
                          "structure": {"origin_w": W.pose.o, "R_w": W.pose.R},
                          "panel": {"origin_w": W.P_o, "R_w": W.P_R},
                          "cage": {"pos_w": cg["pos_w"], "R_w": cg["R_w"], "deck_top_w": cg["deck_top_w"]},
                          "mudline_z": W.z_bed, "physics_dt": self.phys_dt, "steps": int(W.n_steps),
                          "d_time": float(d.time)})

    def reference_run(self):
        """The privileged reference program inside the cell (reference_dir/reference.py: run(server))."""
        if not self.reference_dir:
            return {"error": "bad_params", "detail": "this cell has no reference program"}
        path = os.path.join(self.reference_dir, "reference.py")
        spec = importlib.util.spec_from_file_location("subseacell_reference", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t0 = time.perf_counter()
        result = mod.run(self)
        self.reference_result = result
        return _jsonable({"ok": True, "wall_s": round(time.perf_counter() - t0, 2), "sim_time": self._t(),
                          "summary": (result or {}).get("summary") if isinstance(result, dict) else None})

    # ------------------------------------------------------------------------------------------ practice
    def _practice(self, method, params):
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return {"error": "bad_params", "detail": "params must be an object"}
        if method == "reset":
            if self.world is not None:
                return {"error": "bad_params", "detail": "this connection has its run already"}
            name = params.get("site")
            if set(params) - {"site", "seed"} or not isinstance(name, str) or name not in self.practice_sites:
                return {"error": "bad_params", "detail": "site must be one of the practice sites by name"}
            path = self.practice_sites[name]
            if self.site_guard is not None and not self.site_guard(SITE.load(open(path, "rb").read())):
                return self._forbid(method)
            seed = params.get("seed")
            if seed is not None:
                seed = _intval(seed, "seed")
            self.reset(path, seed=seed, kind="practice", name=name)
            return {"ok": True, "site": name}
        if self.world is None:
            return {"error": "no_mission", "detail": "the run has not started"}
        fn = getattr(self.world, "instruments", None)
        if fn is None:
            return self._forbid(method)
        return _jsonable(fn())

    # ------------------------------------------------------------------------------------------ the record
    def event_log(self):
        return self.elog

    def finalize(self) -> dict:
        """The run's record (root-only verify output): what ended it, the budgets' books, calls, the event log's
        digest, the world's determinism fingerprint and its terminal truth.  No score (subseacell.scoring)."""
        with self._lock:
            if self.world is None:
                return _jsonable({"cell_version": CELL_VERSION, "done_reason": "no_mission", "calls": dict(self.calls),
                                  "illegal_call_count": int(self.illegal_calls),
                                  "internal_errors": int(self.internal_errors)})
            if not self.done:
                self._end("solve_returned")
            W = self.world
            return _jsonable({
                "cell_version": CELL_VERSION, "site": self.site_name, "site_label": self.site_label,
                "site_sha256": self.site_sha, "kind": self.kind, "sensor_seed": str(self.seed),
                "done_reason": self.done_reason, "cell_fault": self.cell_fault, "sim_time_s": self._t(),
                "physics_steps": int(W.n_steps), "budget": self.budget.books(), "budget_limits": self.limits.as_dict(),
                "calls": dict(self.calls), "illegal_call_count": int(self.illegal_calls),
                "internal_errors": int(self.internal_errors), "render_failures": int(self.render_failures),
                "frames_lost": int(self.frames_lost), "frames": {"%dx%d" % k: int(v) for k, v in W.n_frames.items()},
                "pings": int(W.n_pings), "program_time_s": float(self.program_used), "cell_work_s": float(self.work_wall),
                "wall_used_s": float(self.wall_used), "steps_after_done": int(self.steps_after_done),
                "build_wall_s": float(getattr(self, "build_wall_s", 0.0)), "events_sha256": self.elog.digest(),
                "fingerprint": W.fingerprint(), "truth": W.truth(),
                "reference": (self.reference_result or {}).get("summary") if isinstance(self.reference_result, dict)
                else None})
