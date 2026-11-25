"""VoidCellServer: the grader-owned simulation of one void-search episode and the station's calls on the wire.

One episode = one site package (W2 family): the robot is set down on the pad (rest shape, a seeded rollable face
down, seeded yaw and position error, settled; cellcore.world.set_down), the program sees the site through the fixed
RGB-D search camera and the robot's telemetry, commands primitives and cable targets, reports the three search
points and brings the robot back. The server owns the physics (the vendored simulation base `ysim`, the cell core's
executor), steps it only inside `step`, renders the camera on request, samples the telemetry and keeps the truth log
the scoring reads (tests/voidscore). Nothing the program sends is a scoring input except its reports and what
the robot physically does.

The calls (voidcell._lib.protocol; the agent-facing contract is station_api.md):
  mission  get_frame  telemetry  status  run_primitive  set_targets  stop  report_point  step  finish  pose
privileged (the oracle token only): reference_run; practice only (allow_reset): reset.

Surface used by the grader (tests/verify.py) and the wire (cellkit.wire):
  server = VoidCellServer(config=grader_config, mode="hidden", privileged=False, log=log, reference_dir=None)
  server.reset(package, seed=None, kind="hidden")      package: a site folder or archive (<name>.tar.xz)
  server.handle(method, params) -> result dict | {"error": code[, "detail": text]}       (never raises)
  server.done, server.done_reason, server._end(reason), server.work_wall, server.wall_start, server.program_used,
  server.wall_used, server.steps_after_done
  server.finalize() -> the episode's record (JSON-able; "episode_score" from the scorer, None without one)
  server.trajectory() -> float32 (n, k) 10 Hz truth samples (columns in finalize()["trajectory_columns"])
  server.close()
Time: the simulated time advances only in `step(dt)`; the program's time = the wall clock minus the cell's own work
(`work_wall`: step, get_frame, telemetry, pose; encoding included), as in cellkit.wire.
"""

import hashlib
import importlib.util
import json
import math
import os
import time
import traceback

import numpy as np

from . import CELL_VERSION, MESH_DIR, check_vendored

from cellcore import executor as exmod  # noqa: E402  (vendored, on sys.path through the package)
from cellcore import truth as trmod  # noqa: E402
from cellcore import world as wmod  # noqa: E402
from voidcell._lib import budgets as bmod  # noqa: E402
from voidcell._lib import camera as cmod  # noqa: E402
from voidcell._lib import mission as mmod  # noqa: E402
from voidcell._lib import pose_source as pmod  # noqa: E402
from voidcell._lib import protocol as proto  # noqa: E402
from voidcell._lib import sitepkg  # noqa: E402
from voidcell._lib import telemetry as tmod  # noqa: E402

__all__ = ["VoidCellServer", "noise_seed_for", "derive_seed"]

LOG_HZ = 10.0
NUM_MAX = 1.0e6
CONFIG_KEYS = ("limits", "pose_source", "render_backend", "msaa_rgb", "plate_root")
_MASK63 = (1 << 63) - 1


def noise_seed_for(sha):
    """The episode's noise seed when none is given: from the site package's sha256 (a 62-bit integer)."""
    return int(sha[:16], 16) >> 2


def derive_seed(seed, label):
    """An independent sub-seed of the episode seed (set-down, camera, telemetry, pose) by label."""
    h = hashlib.sha256(("%d/%s" % (int(seed), label)).encode()).hexdigest()
    return int(h[:16], 16) >> 2


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


class _Bad(Exception):
    """An argument that fails the boundary check (its text is the reply's detail)."""


def _num(name, v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _Bad("%s must be a number" % name)
    if isinstance(v, float) and not math.isfinite(v):
        raise _Bad("%s must be finite" % name)
    if abs(v) > NUM_MAX:
        raise _Bad("%s must be at most %g in magnitude" % (name, NUM_MAX))
    return v


def check_args(method, params):
    """The boundary check of a public call (protocol.ARGS): unknown names and wrong JSON types are refused here,
    before anything runs; the command calls' own refusals (busy, invalid_name, ...) come from the executor."""
    names = proto.ARGS[method]
    for k in params:
        if k not in names:
            raise _Bad("%s has no argument %r" % (method, str(k)[:64]))
    for k in names:
        if k not in params:
            raise _Bad("%s needs the argument %r" % (method, k))
    out = dict(params)
    if method == "run_primitive":
        if not isinstance(params["name"], str) or len(params["name"]) > 64:
            raise _Bad("name must be a string of at most 64 characters")
        b = params["bottom_nodes"]
        if not isinstance(b, list) or len(b) != 3:
            raise _Bad("bottom_nodes must be a list of three node numbers")
        for x in b:
            _num("a bottom node", x)
    elif method == "set_targets":
        L = params["lengths_mm"]
        if not isinstance(L, list) or len(L) != 6:
            raise _Bad("lengths_mm must be a list of six numbers")
        for x in L:
            _num("a target length", x)
    elif method == "report_point":
        i = params["i"]
        if isinstance(i, bool) or not isinstance(i, int) or i not in (1, 2, 3):
            raise _Bad("i must be 1, 2 or 3")
    elif method == "step":
        _num("dt", params["dt"])
    return out


class VoidCellServer(object):
    def __init__(self, config=None, mode="hidden", privileged=False, allow_reset=False, render=True, log=None,
                 reference_dir=None, workdir=None, scorer=None):
        """config: the grader's run settings (grader_config.json): 'limits' (budgets.DEFAULTS keys), 'pose_source'
        ('estimated' | 'cell'), 'render_backend' (default 'classic'), 'msaa_rgb' (default true), 'plate_root' (the
        folder of the plate bundles, absolute; read by the 'plate_filament' backend only). mode: hidden |
        practice. privileged: the oracle token's calls. reference_dir: the folder of the privileged reference
        (tests/reference). scorer: callable(record) -> {"S": ...} (tests/voidscore; None = no score in the record)."""
        self.config = dict(config or {})
        self.limits = bmod.merged(self.config.get("limits"))
        ps = self.config.get("pose_source", pmod.DEFAULT)
        if ps not in pmod.VALUES:
            raise ValueError("pose_source must be one of %s, not %r" % (pmod.VALUES, ps))
        self.pose_source = ps
        self.render_backend = self.config.get("render_backend", "classic")
        self.msaa_rgb = bool(self.config.get("msaa_rgb", True))
        self.plate_root = self.config.get("plate_root")
        self.mode = mode
        self.privileged = bool(privileged)
        self.allow_reset = bool(allow_reset)
        self.render_enabled = bool(render)
        self.log = log or (lambda m: None)
        self.reference_dir = reference_dir
        self.workdir = workdir
        self.scorer = scorer
        self.ysim_version = check_vendored()
        self.S = None
        self.X = None
        self.cam = None
        self.tel = None
        self.pose = None
        self.done = True
        self.done_reason = None
        self.wall_start = time.time()
        self.wall_used = 0.0
        self.work_wall = 0.0
        self.program_used = 0.0
        self.steps_after_done = 0
        self.calls = {}
        self.illegal_calls = 0
        self.internal_errors = 0
        self.internal_errors_live = 0
        self.render_failures = 0
        self.reference_result = None

    # ------------------------------------------------------------------ episode
    def reset(self, site, seed=None, kind="hidden", sim_time_limit=None, name=None, setdown_seed=None):
        """Load a site package and start its episode: the robot set down on the pad, settled; the episode clock at
        0. seed: the episode's noise seed (None: from the package's sha256); the set-down and every noise stream are
        derived from it (setdown_seed overrides the set-down's). sim_time_limit: None = limits.time_limit_s."""
        t0 = time.time()
        self.close()
        path = os.fspath(site)
        self.site, xml, self.prov = sitepkg.open_site(path)
        self.site_xml = xml
        self.site_sha = self.prov["sha256"]
        self.site_name = name or os.path.basename(path.rstrip("/")).split(".")[0]
        self.kind = str(kind)
        self.seed = int(seed) if seed is not None else noise_seed_for(self.site_sha)
        self.setdown_seed = int(setdown_seed) if setdown_seed is not None else derive_seed(self.seed, "setdown")
        self.time_limit = float(sim_time_limit) if sim_time_limit is not None else self.limits["time_limit_s"]
        self.budgets = bmod.Budgets(dict(self.limits, time_limit_s=self.time_limit))
        tb = time.time()
        S, self.setdown = wmod.set_down(lambda: wmod.new_simulator(xml), self.site["pad"]["xy"], self.setdown_seed)
        self.setdown_wall_s = time.time() - tb
        self.S = S
        self.X = exmod.Executor(S)
        self.robot_truth = trmod.RobotTruth(S)
        self.site_truth = trmod.SiteTruth(self.site)
        self.tel = tmod.Telemetry(S, derive_seed(self.seed, "telemetry"))
        self.clock = cmod.ExposureClock(float(self.site["camera"]["model"]["rate_hz"]))
        self.cam = None
        if self.render_enabled:
            bkw = {"msaa_rgb": self.msaa_rgb}
            if self.render_backend != "classic":     # the plate backend picks the site's bundle by these
                bkw.update(site_sha256=self.site_sha, site_json_sha256=self.prov["site_json_sha256"],
                           site_xml_sha256=self.prov["site_xml_sha256"], plate_root=self.plate_root)
            self.cam = cmod.SearchCamera(self.site, xml, MESH_DIR, derive_seed(self.seed, "camera"),
                                         backend=self.render_backend, backend_kw=bkw)
            self.clock.listeners.append(self.cam.on_exposure)
        cam_pos = np.asarray(self.site["camera"]["true"]["T_world_camera"], float)[:3, 3]
        self.pose = pmod.PoseSource(S, cam_pos, derive_seed(self.seed, "pose"), self._cap_radius())
        if self.pose_source == "cell":
            self.clock.listeners.append(self.pose.on_exposure)
        # the 10 Hz truth log the scoring reads (tests/voidscore): t, planar centroid, heading, busy, nodes
        self.ticks_per_log = int(round((1.0 / LOG_HZ) / self.X.ctrl_dt))
        self.log10 = {"t": [], "xy": [], "heading": [], "busy": [], "nodes": []}
        self.commands = []
        self.reports = []
        self.events = []
        self.frame_hashes = []
        self.physics_wall = 0.0
        self.X.observers.append(self._on_tick)
        self._on_tick(self.X, first=True)
        if self.cam is not None:
            self.cam.frame()          # exposure 0 rendered now: the renderer's one-time warm-up is part of the reset
        self.done = False
        self.done_reason = None
        self.calls = {}
        self.illegal_calls = 0
        self.internal_errors = 0
        self.internal_errors_live = 0
        self.render_failures = 0
        self.wall_start = time.time()
        self.wall_used = 0.0
        self.work_wall = 0.0
        self.program_used = 0.0
        self.steps_after_done = 0
        self.reference_result = None
        self.build_wall_s = time.time() - t0
        return {"ok": True}

    def _cap_radius(self):
        import mujoco
        S = self.S
        g = mujoco.mj_name2id(S.m, mujoco.mjtObj.mjOBJ_GEOM, "sock0")
        return float(S.m.geom_size[g][0]) if g >= 0 else 0.0175

    def _on_tick(self, X, first=False):
        t = X.t()
        if not first:
            self.tel.sample(t)
        self.clock.on_tick(t, self.S.d.qpos)
        if first or X.ticks % self.ticks_per_log == 0:
            n = self.S.nodes()
            x, y, th = self.robot_truth.pose()
            L = self.log10
            L["t"].append(round(t, 6))
            L["xy"].append([x, y])
            L["heading"].append(th)
            L["busy"].append(bool(X.busy))
            L["nodes"].append(n.tolist())

    def close(self):
        if self.cam is not None:
            self.cam.close()
            self.cam = None

    def _end(self, reason):
        if not self.done:
            self.done = True
            self.done_reason = str(reason)
            self.events.append([round(self.X.t(), 3) if self.X is not None else None, "end", str(reason)])
        elif self.done_reason is None:
            self.done_reason = str(reason)

    # ------------------------------------------------------------------ dispatch
    def handle(self, method, params):
        method = str(method)
        known = method in proto.PUBLIC_METHODS + proto.PRIVILEGED_METHODS + proto.PRACTICE_METHODS
        key = method if known else "_unknown"
        self.calls[key] = self.calls.get(key, 0) + 1
        t0 = time.time()
        live = not self.done
        try:
            res = self._dispatch(method, params)
            why = proto.check_reply(method, res)
            if why is not None:
                raise RuntimeError("reply outside the contract: " + why)
            return res
        except Exception:  # noqa: BLE001 - nothing a request does may crash the wire
            self.internal_errors += 1
            if live:
                self.internal_errors_live += 1
            self.log("[cell] internal error in %s\n%s" % (method, traceback.format_exc()))
            if method in proto.STATE_METHODS:
                self._end("internal_error")
            return {"error": "internal_error"}
        finally:
            if method in proto.WORK_METHODS:
                self.work_wall += time.time() - t0

    def _dispatch(self, method, params):
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return {"error": "bad_argument", "detail": "params must be an object"}
        if method in proto.PRACTICE_METHODS:
            if not (self.mode == "practice" and self.allow_reset):
                self.illegal_calls += 1
                return {"error": "forbidden"}
            p = dict(params)
            return self.reset(p.pop("site"), **p)
        if method in proto.PRIVILEGED_METHODS:
            if not self.privileged:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            return getattr(self, method)()
        if method not in proto.PUBLIC_METHODS:
            self.illegal_calls += 1
            return {"error": "forbidden"}
        if self.S is None:
            return {"error": "not_available", "detail": "the episode has not started"}
        try:
            params = check_args(method, params)
        except _Bad as e:
            return {"error": "bad_argument", "detail": str(e)[:200]}
        if self.done and method in proto.STATE_METHODS:
            return {"error": "episode_over", "detail": "the episode has ended (%s)" % self.done_reason}
        return getattr(self, "_m_" + method)(**params)

    # ------------------------------------------------------------------ the calls
    def _m_mission(self):
        return mmod.build(self.site, self.budgets.limits, self.pose_source)

    def _m_get_frame(self):
        if self.cam is None:
            return {"error": "not_available", "detail": "this cell runs without the camera"}
        if not self.budgets.take_frame():
            return {"error": "budget_exhausted", "detail": "the episode's %d frames are used" % self.limits["frames"]}
        try:
            f = self.cam.frame()
        except Exception:  # noqa: BLE001
            self.render_failures += 1
            self.log("[cell] frame not rendered\n%s" % traceback.format_exc())
            raise
        self.frame_hashes.append([int(f.index), f.sha256()])
        out = f.wire()
        out["frames_left"] = self.budgets.frames_left()
        return out

    def _m_telemetry(self):
        return self.tel.read()

    def _m_status(self):
        return _jsonable(self.X.status())

    def _m_run_primitive(self, name, bottom_nodes):
        r = self.X.run_primitive(name, bottom_nodes)
        self.commands.append({"t": round(self.X.t(), 6), "kind": "primitive", "name": str(name)[:64],
                              "bottom_nodes": list(bottom_nodes), "accepted": bool(r["accepted"]),
                              "error": r["error"]})
        return {"accepted": bool(r["accepted"]), "error": r["error"]}

    def _m_set_targets(self, lengths_mm):
        r = self.X.set_targets(lengths_mm)
        self.commands.append({"t": round(self.X.t(), 6), "kind": "set_targets",
                              "lengths_mm": [float(x) for x in lengths_mm], "accepted": bool(r["accepted"]),
                              "error": r["error"]})
        return {"accepted": bool(r["accepted"]), "error": r["error"]}

    def _m_stop(self):
        was = self.X.busy
        self.X.stop()
        self.commands.append({"t": round(self.X.t(), 6), "kind": "stop", "aborted": bool(was)})
        return {}

    def _m_report_point(self, i):
        ok, left = self.budgets.take_report(int(i))
        x, y, _th = self.robot_truth.pose()
        self.reports.append({"t": round(self.X.t(), 6), "i": int(i), "accepted": bool(ok), "xy": [x, y],
                             "busy": bool(self.X.busy)})
        if not ok:
            return {"accepted": False, "reports_left": 0, "error": "budget_exhausted"}
        return {"accepted": True, "reports_left": int(left), "error": None}

    def _m_step(self, dt):
        X = self.X
        dtf = float(dt)
        n = int(round(dtf / exmod.DT_QUANTUM))
        if not (exmod.DT_MIN - 1e-9 <= dtf <= exmod.DT_MAX + 1e-9) or abs(n * exmod.DT_QUANTUM - dtf) > 1e-9:
            return {"error": "bad_argument", "detail": "dt must be a multiple of 0.01 s between 0.01 and 10.0"}
        k = int(round(dtf / X.ctrl_dt))
        left = int(round((self.time_limit - X.t()) / X.ctrl_dt))
        run = max(0, min(k, left))
        t0 = time.perf_counter()
        X.run_ticks(run)
        self.physics_wall += time.perf_counter() - t0
        if X.t() >= self.time_limit - 1e-9:
            self._end("time_limit")
        return {"t": round(X.t(), 6), "done": bool(self.done)}

    def _m_finish(self):
        self._end("finish")
        return {"done": True}

    def _m_pose(self):
        if self.pose_source != "cell":
            return {"error": "not_available", "detail": "pose_source is 'estimated': the program estimates the pose"}
        return self.pose.reply()

    # ------------------------------------------------------------------ privileged
    def reference_run(self):
        """The privileged reference program inside the cell (reference_dir/reference.py, run(server))."""
        if not self.reference_dir:
            return {"error": "not_available", "detail": "this cell has no reference program"}
        path = os.path.join(self.reference_dir, "reference.py")
        spec = importlib.util.spec_from_file_location("voidcell_reference", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t0 = time.time()
        result = mod.run(self)
        self.reference_result = result
        return _jsonable({"ok": True, "wall_s": round(time.time() - t0, 2), "t": round(self.X.t(), 3),
                          "summary": (result or {}).get("summary") if isinstance(result, dict) else None})

    # ------------------------------------------------------------------ the record
    def end_state(self):
        x, y, th = self.robot_truth.pose()
        b = self.robot_truth.bottom()
        return {"t": round(self.X.t(), 6), "xy": [x, y], "heading": th, "busy": bool(self.X.busy),
                "bottom_nodes": b["nodes"], "rollable": b["rollable"], "end_triangle": b["end_triangle"]}

    def finalize(self):
        """The episode's record (root-only verify output; the practice debrief's source)."""
        if self.S is None:
            return {"cell_version": CELL_VERSION, "episode_score": 0.0, "done_reason": "no_episode",
                    "calls": dict(self.calls), "illegal_call_count": int(self.illegal_calls),
                    "internal_errors": int(self.internal_errors), "internal_errors_live": int(self.internal_errors_live)}
        if not self.done:
            self._end("solve_returned")
        rec = {
            "cell_version": CELL_VERSION, "ysim_version": self.ysim_version, "site": self.site_name,
            "site_sha256": self.site_sha, "site_provenance": self.prov, "kind": self.kind,
            "noise_seed": str(self.seed), "setdown_seed": str(self.setdown_seed), "setdown": self.setdown,
            "pose_source": self.pose_source, "render_backend": None if self.cam is None else self.cam.backend.version(),
            "done_reason": self.done_reason, "sim_time_s": round(self.X.t(), 6), "sim_time_limit_s": self.time_limit,
            "end": self.end_state(), "budgets": self.budgets.record(), "reports": self.reports,
            "commands": self.commands, "events": self.events,
            "log10_columns": ["t", "x", "y", "heading", "busy"],
            "log10": [[t, xy[0], xy[1], h, b] for t, xy, h, b in zip(self.log10["t"], self.log10["xy"],
                                                                     self.log10["heading"], self.log10["busy"])],
            "executor_history": self.X.history, "frame_hashes": self.frame_hashes,
            "telemetry": self.tel.record(),
            "exposures": self.clock.count,
            "camera": None if self.cam is None else {"exposures": self.cam.exposures, "rendered": self.cam.n_rendered,
                                                     "render_s": round(self.cam.render_s, 4),
                                                     "sensor_s": round(self.cam.sensor_s, 4)},
            "pose_updates": self.pose.n_updates,
            "illegal_call_count": int(self.illegal_calls), "internal_errors": int(self.internal_errors),
            "internal_errors_live": int(self.internal_errors_live), "render_failures": int(self.render_failures),
            "calls": dict(self.calls), "steps_after_done": int(self.steps_after_done),
            "program_time_s": float(self.program_used), "cell_work_s": float(self.work_wall),
            "wall_used_s": float(self.wall_used), "physics_wall_s": round(self.physics_wall, 4),
            "build_wall_s": round(self.build_wall_s, 4), "setdown_wall_s": round(self.setdown_wall_s, 4),
            "final_state_sha256": self.S.state_hash(),
            "trajectory_columns": self.trajectory_columns(),
        }
        if isinstance(self.reference_result, dict) and self.reference_result.get("summary") is not None:
            rec["reference"] = self.reference_result.get("summary")
        rec["episode_score"] = None
        if self.scorer is not None:
            sc = self.scorer(rec)
            rec["score"] = sc
            S = float(sc["S"])
            rec["episode_score"] = S if (math.isfinite(S) and 0.0 <= S <= 1.0 + 1e-9) else 0.0
        return _jsonable(rec)

    def trajectory_columns(self):
        return ["t", "x", "y", "heading", "busy"] + ["n%d_%s" % (i, a) for i in range(6) for a in "xyz"]

    def trajectory(self):
        L = self.log10
        if self.S is None or not L["t"]:
            return np.zeros((0, 0), np.float32)
        rows = [[t, xy[0], xy[1], h, float(b)] + list(np.asarray(n).reshape(-1))
                for t, xy, h, b, n in zip(L["t"], L["xy"], L["heading"], L["busy"], L["nodes"])]
        return np.asarray(rows, np.float32)

    def state_digest(self):
        return self.S.state_hash() if self.S is not None else None
