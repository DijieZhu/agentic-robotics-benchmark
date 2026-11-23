"""ParcelCellServer: the grader-owned cell of the label-facing station. It owns
the per-shift world (physics), the cameras, the tool-pose level, the safety
watch and the shift's record; the world advances only inside ``step``. The
program talks to it over the JSON-lines wire (``handle``); nothing the
program sends is a scoring input except through the physics.

Public calls (``cell_api.md`` is the agent-facing contract):
  describe()                              static facts
  get_state()                             the robot's own state (joints, tools, grippers), budgets
  step(q_target=None, grip=None, n=1, tool_pose=None)
                                          n ticks (20 ms each, at most 50 per call)
                                          holding the targets (per arm: joint targets or a
                                          tool pose); replies get_state()
  get_camera(name, depth=False)           the newest delivered frame of a camera
  finish()                                ends the shift
Privileged (the oracle token's cells only): get_state_privileged(),
reference_run(). Practice (a maintainers' development mode): reset(...),
truth().

Boundary: every argument of a public call is checked before anything runs
(``ARG_KINDS``); a call that fails answers {"error": "bad_params", "detail"}
and changes nothing. Unknown methods answer {"error": "forbidden"} (counted).
``handle()`` never raises: any other exception is logged and answered
{"error": "internal_error"}; inside a call that changes the world it also
ends the shift (scored as it stands).

Attributes the wire relies on (cellkit.wire.serve_episode): done,
done_reason, _end(reason), wall_start, wall_used, work_wall (the cell's own
time: simulation inside step and rendering inside get_camera, never charged
to the program), program_used, steps_after_done.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
import traceback
from collections import deque

import numpy as np

from . import CELL_VERSION
from .cameras import CameraSystem, NoFrameYet, NoFramesLeft
from .config import CAMERA_NAMES, cell_config, load_config, score_config
from .look import RenderError
from .episode import ShiftRecord
from .robot import GRIPPER, JOINT_NAMES, SIDES
from .safety import SafetyWatch
from .sensors import TorqueSensor
from .shiftgen import draw_line_shift, draw_shift, from_package, package_sha256, to_package
from .stream import Shift
from .toolpose import ToolPoseLevel
from .world import ShiftOver, ShiftWorld

__all__ = ["ParcelCellServer", "PUBLIC_METHODS", "PRIVILEGED_METHODS", "PRACTICE_METHODS",
           "ARG_KINDS", "check_args", "BadParams", "shift_digest"]

PUBLIC_METHODS = ("describe", "get_state", "step", "get_camera", "finish")
PRIVILEGED_METHODS = ("get_state_privileged", "reference_run")
PRACTICE_METHODS = ("reset", "truth")
STATE_METHODS = ("step", "finish", "reference_run", "reset")
KNOWN_METHODS = frozenset(PUBLIC_METHODS + PRIVILEGED_METHODS + PRACTICE_METHODS)

# The boundary. Kinds: q? (null, a list of 14 numbers, or an object {"left": 7 numbers or
# null, "right": ...}), grip2? (null or a list of 2 numbers), tool? (null or an object
# {"left": pose or null, "right": ...}, pose = {"pos": 3 numbers, "quat": 4 numbers of norm
# 1 within QUAT_TOL}), ticks (an integer-valued number 1 .. TICKS_MAX; the cell runs at most
# budgets.max_ticks_per_step of them), word (a string of at most WORD_MAX characters), bool
# (true / false). A number is a JSON number (not a boolean), finite, at most NUM_MAX in
# magnitude. One arm takes one command level per step: joint targets for an arm and a tool
# pose for the same arm in one step fail the check.
NUM_MAX = 1.0e6
TICKS_MAX = 1.0e6
WORD_MAX = 64
QUAT_TOL = 1.0e-3
ARM_KEYS = ("left", "right")
ARG_KINDS = {
    "describe": {}, "get_state": {}, "finish": {},
    "step": {"q_target": "q?", "grip": "grip2?", "n": "ticks", "tool_pose": "tool?"},
    "get_camera": {"name": "word", "depth": "bool"},
}


class BadParams(Exception):
    pass


def _num(name, v, lim=NUM_MAX):
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise BadParams("%s must be a number" % name)
    if isinstance(v, int):
        if not -lim <= v <= lim:
            raise BadParams("%s must be at most %g in magnitude" % (name, lim))
        return float(v)
    if not math.isfinite(v):
        raise BadParams("%s must be finite" % name)
    if abs(v) > lim:
        raise BadParams("%s must be at most %g in magnitude" % (name, lim))
    return float(v)


def _vec(name, v, n):
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise BadParams("%s must be null or a list of %d numbers" % (name, n))
    return [_num("%s[%d]" % (name, i), x) for i, x in enumerate(v)]


def _arms(name, v):
    if not isinstance(v, dict):
        raise BadParams("%s must be null, a list or an object with the keys left and right" % name)
    for k in v:
        if k not in ARM_KEYS:
            raise BadParams("%s has no key %r (the arms are left and right)" % (name, str(k)[:WORD_MAX]))
    return v


def _q_arg(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return _vec("q_target", v, 14)
    out = {}
    for k, x in _arms("q_target", v).items():
        out[k] = _vec("q_target.%s" % k, x, 7)
    return out


def _tool_arg(v):
    if v is None:
        return None
    out = {}
    for k, p in _arms("tool_pose", v).items():
        if p is None:
            out[k] = None
            continue
        if not isinstance(p, dict) or set(p) != {"pos", "quat"}:
            raise BadParams("tool_pose.%s must be null or an object {pos: [x, y, z], quat: [w, x, y, z]}" % k)
        pos = _vec("tool_pose.%s.pos" % k, p["pos"], 3)
        quat = _vec("tool_pose.%s.quat" % k, p["quat"], 4)
        if pos is None or quat is None:
            raise BadParams("tool_pose.%s needs pos and quat" % k)
        nq = math.sqrt(math.fsum(x * x for x in quat))
        if abs(nq - 1.0) > QUAT_TOL:
            raise BadParams("tool_pose.%s.quat must be a unit quaternion (w, x, y, z; norm 1 within %g)"
                            % (k, QUAT_TOL))
        out[k] = {"pos": pos, "quat": quat}
    return out


def check_args(method: str, params: dict) -> dict:
    """The arguments of a public call, checked (ARG_KINDS); raises BadParams."""
    kinds = ARG_KINDS[method]
    for name in params:
        if name not in kinds:
            raise BadParams("%s has no argument %r" % (method, str(name)[:WORD_MAX]))
    out = {}
    for name, v in params.items():
        k = kinds[name]
        if k == "q?":
            out[name] = _q_arg(v)
        elif k == "tool?":
            out[name] = _tool_arg(v)
        elif k == "grip2?":
            out[name] = _vec(name, v, 2)
        elif k == "ticks":
            x = _num(name, v, TICKS_MAX)
            if x != int(x) or x < 1:
                raise BadParams("%s must be a whole number >= 1" % name)
            out[name] = int(x)
        elif k == "word":
            if not isinstance(v, str):
                raise BadParams("%s must be a string" % name)
            if len(v) > WORD_MAX:
                raise BadParams("%s has at most %d characters" % (name, WORD_MAX))
            out[name] = v
        elif k == "bool":
            if not isinstance(v, bool):
                raise BadParams("%s must be true or false" % name)
            out[name] = v
    tp = out.get("tool_pose")
    if tp:
        qt = out.get("q_target")
        for k, p in tp.items():
            if p is None:
                continue
            if isinstance(qt, list) or (isinstance(qt, dict) and qt.get(k) is not None):
                raise BadParams("the %s arm got joint targets and a tool pose in one step: one level per arm" % k)
    return out


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


def _r(v, n=6):
    return [round(float(x), n) for x in v]


def shift_digest(package: dict) -> tuple[str, int]:
    """(sha256 hex of the shift package's canonical bytes, its first 8 bytes)."""
    h = package_sha256(package)
    return h, int(h[:16], 16)


class GripperStatus:
    """The grippers' "object detected" status, as the real gripper reports it:
    true when the fingers stopped short of (or beyond) the opening the command
    gives on nothing by more than ``short_m`` and moved less than ``still_m``
    over ``hold_s``. The free opening per command is the configuration's
    measured table (commands 0, 1/(n-1), .., 1)."""

    def __init__(self, cfg: dict, tick_s: float = 0.02):
        self.n = max(1, int(round(float(cfg.get("hold_s", 0.2)) / tick_s)))
        self.short = float(cfg.get("short_m", 0.003))
        self.still = float(cfg.get("still_m", 0.0005))
        tab = cfg.get("free_opening_m")
        self.table = (np.linspace(0.0, 1.0, len(tab)), np.asarray(tab, float)) if tab else None
        self.hist = deque(maxlen=self.n)
        self.detected = [False, False]

    def free_opening(self, command) -> float:
        if self.table is None:
            return float(GRIPPER.max_width) * (1.0 - float(command))
        return float(np.interp(float(command), *self.table))

    def update(self, openings, commands) -> list:
        self.hist.append((list(openings), list(commands)))
        h = self.hist
        det = []
        for i in range(2):
            if len(h) < self.n:
                det.append(False)
                continue
            ops = [e[0][i] for e in h]
            short = all(abs(e[0][i] - self.free_opening(e[1][i])) > self.short for e in h)
            det.append(bool(short and (max(ops) - min(ops)) < self.still))
        self.detected = det
        return det


class ParcelCellServer(object):
    def __init__(self, config=None, mode="hidden", privileged=False, allow_reset=False,
                 render=True, log=None, reference_dir=None, look=None, visuals=None):
        """config: the cell configuration (a dict or a JSON path, deep-merged over
        config.DEFAULTS; tests/grader_config.json). mode: hidden | practice.
        privileged: the oracle token's calls. reference_dir: the folder of the
        privileged reference program (controller.py). look: the camera backend
        (default config look.backend). visuals: the parcel visuals ("plain",
        "kit:<dir>" or an object)."""
        self.cfg = load_config(config)
        self.mode = mode
        self.privileged = bool(privileged)
        self.allow_reset = bool(allow_reset)
        self.render_enabled = bool(render)
        self.log = log or (lambda s: None)
        self.reference_dir = reference_dir
        self.look_name = look or self.cfg["look"]["backend"]
        self.visuals = visuals
        self.world = None
        self.wall_start = None
        self.wall_used = 0.0
        self.program_used = 0.0
        self.steps_after_done = 0
        self._t_reset = None
        self._extra_work = 0.0
        self.internal_errors = 0
        self.internal_errors_live = 0
        self.calls = {}
        self.illegal_calls = 0
        self.bad_params = 0
        self.step_calls = 0
        self.refused_calls = 0

    # ================================================================ shift setup
    def reset(self, shift, meta=None):
        """Build the cell for one shift: a Shift, a shift package (dict) or a
        path to one. Returns describe()."""
        if isinstance(shift, (str, bytes)) or hasattr(shift, "__fspath__"):
            with open(shift, "rb") as fh:
                shift = json.loads(fh.read())
        if isinstance(shift, dict):
            package = shift
            shift, m2 = from_package(package)
            meta = dict(m2, **(meta or {}))
        else:
            package = to_package(shift, meta)
        self.shift = shift
        self.meta = dict(meta or {})
        self.shift_sha256, digest = shift_digest(package)
        if self.world is not None:
            self.close()
        t0 = time.time()
        cc = cell_config(self.cfg)
        vis = self.visuals if self.visuals is not None else \
            (self.cfg.get("look") or {}).get("parcel_visuals", "plain")
        if isinstance(vis, str):
            from .kitcache import cache_dir
            from .visuals import make_visuals
            vis = make_visuals(vis, cache_dir=cache_dir(self.cfg.get("look")))
        self.world = w = ShiftWorld(shift, cc, visuals=vis, depth_meshes=self._depth_meshes())
        self.cams = CameraSystem(w, self.cfg, digest, look=self.look_name,
                                 render=self.render_enabled, log=self.log)
        wcfg = self.cfg["world"]
        self.tool = ToolPoseLevel(w, wcfg.get("tool_pose"), float(wcfg["target_speed_limit_rad_s"]))
        self.watch = SafetyWatch(w, self.cfg["safety"], log=self.log)
        self.cams.damage_fn = lambda: self.watch.damaged          # a damaged parcel is drawn crushed (cameras.py)
        self.line = bool(shift.setting)
        self.score_cfg = score_config(self.cfg, shift)
        self.record = ShiftRecord(w, self.watch, self.score_cfg)
        if self.line:
            w.damage_source = lambda: self.watch.damaged      # a damaged parcel is settled: the next one is fed
        self.watch.on_halt.append(self._on_halt)
        w.command_gate = self.watch.gate
        w.settle_listeners.append(self.cams.on_settle)
        w.tick_listeners += [self.cams.on_tick, self.watch.on_tick,
                             self.record.on_tick, self._on_tick]
        b = self.cfg["budgets"]
        self.duration = float(shift.duration)
        self.program_budget_s = float(b["program_time_factor"]) * self.duration + \
            float(b["program_time_extra_s"])
        self.frame_budget = int(math.floor(float(b["frames_per_shift_s"]) * self.duration))
        sn = self.cfg["sensors"]
        self.tau_sensor = TorqueSensor([float(w.model.actuator_forcerange[a, 1]) for a in w._arm_aid], digest,
                                       float(sn.get("tau_noise_rel", 0.0)), float(sn.get("tau_offset_rel", 0.0)),
                                       sn.get("tau_offset_period_s", (40.0, 240.0)))
        self.cams.reset_counters(self.frame_budget)
        self.tool.reset()
        self.watch.reset()
        self.record.reset()
        w.reset(force=True)   # again, now with the listeners (deterministic): the settle's exposures
        self.build_wall_s = time.time() - t0
        self.gripper_status = GripperStatus(wcfg.get("gripper_object") or {})
        self._q_hold = w.home.copy()
        self._grip_hold = np.zeros(2)
        self.calls = {}
        self.illegal_calls = 0
        self.bad_params = 0
        self.internal_errors = 0
        self.internal_errors_live = 0
        self.step_calls = 0
        self.steps_after_done = 0
        self.refused_calls = 0
        self.wall_start = None
        self.wall_used = 0.0
        self.program_used = 0.0
        self._extra_work = 0.0
        self._t_reset = time.time()
        self._ref = None                   # (t0, work0) while reference_run runs
        self.reference_result = None
        self._describe = None
        return self.describe()

    def _depth_meshes(self):
        """The depth twin's meshes [(V, F)] from the render bundle (``depth_near.npz``), when the look asks for it
        and the bundle has them; else None. config look.depth_twin: "auto" (the default: on with the filament
        backend, whose colour images draw the bundle's surroundings; off with the classic backend, whose colour
        images draw the cell's own geometry only, so that depth never shows what colour does not), true, or false."""
        lk = self.cfg.get("look") or {}
        twin = lk.get("depth_twin", "auto")
        if twin == "auto":
            twin = self.look_name == "filament"
        if not twin or not self.render_enabled:
            return None
        from .look_depth import load_depth_twin
        return load_depth_twin(lk, required=(self.look_name == "filament"), log=self.log)

    def close(self):
        """Free the shift's renderers, model and data (and return the freed heap
        to the system: a grader runs its shifts one after another in one
        process)."""
        if self.world is not None:
            try:
                self.cams.close()
            finally:
                self.world.close()
                self.world = None
                self.cams = self.tool = self.watch = self.record = None
                import gc
                gc.collect()
                try:
                    import ctypes
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except (OSError, AttributeError):
                    pass

    # ================================================================ wire attributes
    @property
    def done(self) -> bool:
        return self.world is not None and self.world.done

    @property
    def done_reason(self):
        return None if self.world is None else self.world.over_reason

    def _end(self, reason):
        if self.world is not None:
            self.world.end(reason)

    @property
    def work_wall(self) -> float:
        if self.world is None or self.cams is None:
            return float(self._extra_work)
        return float(self.world.tick_wall + self.cams.render_wall + self._extra_work)

    def program_time(self) -> float:
        t0 = self.wall_start if self.wall_start is not None else self._t_reset
        return max(0.0, (time.time() - t0) - self.work_wall) if t0 else 0.0

    # ================================================================ listeners
    def _on_halt(self, q):
        self._q_hold = q.copy()
        self._grip_hold = self.watch.frozen_grip.copy()
        if getattr(self, "tool", None) is not None:
            self.tool.clear()                  # a stop ends every tool-pose motion: the arms hold

    def _on_tick(self, world):
        self._gripper_tick()
        if self._ref is not None:
            t0, w0 = self._ref
            used = (time.time() - t0) - (self.work_wall - w0)
            if used > self.privileged_budget_s:
                world.end("program_time")

    @property
    def privileged_budget_s(self) -> float:
        """The privileged reference's own time cap: a multiple of the shift's
        duration."""
        b = self.cfg["budgets"]
        if "privileged_program_time_factor" in b:
            return float(b["privileged_program_time_factor"]) * float(self.duration)
        return float(b.get("privileged_program_time_s", 7200.0))

    def _gripper_tick(self):
        self.gripper_status.update(self._openings(rounded=False), [float(v) for v in self.world._grip_cmd])

    # ================================================================ public API
    def describe(self):
        if self._describe is not None:
            return json.loads(json.dumps(self._describe))
        w = self.world
        m = w.model
        b, S, sc, wc = (self.cfg["budgets"], self.cfg["safety"], self.cfg["score"],
                        self.cfg["world"])
        st = w.station
        ranges = [[float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])] for j in w._arm_jid]
        torque = [float(m.actuator_forcerange[a, 1]) for a in w._arm_aid]

        def box(name):
            g = w._gid(name)
            return {"centre": _r(m.geom_pos[g], 4), "half_size": _r(m.geom_size[g], 4)} \
                if g >= 0 else None
        from .station import FRAME_SOLIDS
        hw = ["belt_rail_far", "belt_rail_near_up", "belt_rail_near_down", "belt_lip_near_up",
              "belt_lip_near_down"] + list(FRAME_SOLIDS) + [str(h[0]) for h in st.hardware] + ["head_camera_housing"]
        T_ft = {}
        import mujoco
        d = w.data
        for s in SIDES:
            fs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_flange")
            ts = w._tcp_sid[s]
            Tf, Tt = np.eye(4), np.eye(4)
            Tf[:3, :3], Tf[:3, 3] = d.site_xmat[fs].reshape(3, 3), d.site_xpos[fs]
            Tt[:3, :3], Tt[:3, 3] = d.site_xmat[ts].reshape(3, 3), d.site_xpos[ts]
            T_ft[s.lower()] = np.round(np.linalg.inv(Tf) @ Tt, 9).tolist()
        out = {
            "cell_version": CELL_VERSION,
            "units": {"length": "m", "angle": "rad", "time": "s", "force": "N",
                      "frame": "world: the robot's column at the origin, z up, the robot faces +x, "
                               "the belt runs along +y"},
            "tick_s": 0.02, "physics_step_s": 0.002,
            "robot": {
                "joints": list(JOINT_NAMES), "joint_range": ranges,
                "torque_limit_Nm": torque,
                "command_speed_limit_rad_s": float(wc["target_speed_limit_rad_s"]),
                "command": "position targets; each tick the applied target moves toward the commanded "
                           "one by at most command_speed_limit_rad_s x tick_s; targets are clipped to "
                           "joint_range",
                "start_posture": _r(w.home, 6),
                "flange_sites": {"left": "Left_flange", "right": "Right_flange"},
                "T_flange_tcp": T_ft,
                "tcp": "the grasp centre between the finger pads",
                "tool_pose": self._tool_describe(),
            },
            "gripper": {"command_range": [0.0, 1.0], "command": "0 open .. 1 closed",
                        "max_opening_m": float(GRIPPER.max_width),
                        "opening_m": "measured distance between the finger pads",
                        "object_detected": "the fingers stopped short of the commanded opening "
                                           "(by more than %g m) and stood still for %g s"
                                           % (self.gripper_status.short, self.gripper_status.n * 0.02),
                        "free_opening_m": [round(self.gripper_status.free_opening(c), 4) for c in
                                           (0.0, 0.25, 0.5, 0.75, 1.0)],
                        "free_opening_at_command": [0.0, 0.25, 0.5, 0.75, 1.0]},
            "station": {
                "belt": {"top_z": float(st.belt_height), "centre_x": float(st.belt_center_x),
                         "width": float(st.belt_width), "direction": "+y",
                         "frame_half_length": float(st.visible_half_len),
                         "work_window_half_length": float(st.work_window_half)},
                "lines": {"entry_y": float(wc["entry_y"]),
                          "scan_y": float(wc["scan_y"]), "exit_y": float(wc["exit_y"])},
                "hardware": {n: box(n) for n in hw if box(n) is not None},
                "floor_z": 0.0,
            },
            "cameras": self.cams.describe(w.data),
            "sensor_models": {"rgb_noise_sigma": float(self.cfg["sensors"]["rgb_noise_sigma"]),
                              "depth_noise": "%g x distance (1 sigma)" % float(self.cfg["sensors"]["depth_noise_rel"]),
                              "depth_dropout": float(self.cfg["sensors"]["depth_dropout"]),
                              "depth_png": "16-bit, millimetres, 0 = no reading",
                              "tau": "the drives' torque readings: zero-mean noise of %g x the joint's torque limit "
                                     "(1 sigma, independent per joint and tick) plus a slowly wandering offset of at "
                                     "most %g x the limit; q and dq are exact"
                                     % (float(self.cfg["sensors"]["tau_noise_rel"]),
                                        float(self.cfg["sensors"]["tau_offset_rel"]))},
            "shift": self._shift_describe(),
            "budgets": {"program_time_s": self.program_budget_s,
                        "program_time": "the shift's wall clock minus the cell's own time "
                                        "(simulation in step, rendering in get_camera)",
                        "silence_s": float(b["silence_s"]), "frames": self.frame_budget,
                        "frames_pooled": list(CAMERA_NAMES),
                        "max_ticks_per_step": int(b["max_ticks_per_step"]),
                        "after_over": {"steps": int(b["after_over_steps"]),
                                       "requests": int(b["after_over_requests"]),
                                       "seconds": float(b["after_over_s"])}},
            "safety": dict({k: S[k] for k in ("structure_N", "self_N", "belt_N", "press_ticks",
                                              "stop_lock_s", "damage_N", "max_stops", "squeeze_N",
                                              "squeeze_ticks")},
                           damage_look="from the tick a parcel counts as damaged, every camera draws it crushed: "
                                       "darker, greyer faces crossed by dark irregular creases (board) or a scuffed, "
                                       "creased film, the waybill creased too; the mean grey level of its faces drops "
                                       "(by a fifth or more in the infeed and head cameras) and their share of edge "
                                       "pixels grows; its shape, pose and depth image do not change; the wire reports "
                                       "no damage"),
            "score": self._score_describe() if self.line else {"label_tilt_deg": float(sc["label_tilt_deg"]),
                      "max_angular_speed_rad_s": float(sc["max_angular_speed_rad_s"]),
                      "max_rel_speed_m_s": float(sc["max_rel_speed_m_s"]),
                      "rest_window_s": float(sc["rest_window_s"]),
                      "rest_clearance_max_m": float(sc["rest_clearance_max_m"]),
                      "readable": ("at the tick the parcel's centre first crosses the scan line going downstream: "
                                   "label within label_tilt_deg of vertical up; it rests on the belt (it touched "
                                   "the belt surface in at least one tick of the last rest_window_s, that tick "
                                   "included, and its lowest point is at most rest_clearance_max_m above the belt "
                                   "surface; one tick's contact never decides), turns slower than "
                                   "max_angular_speed_rad_s and moves slower than max_rel_speed_m_s relative to "
                                   "the belt (horizontal velocity); no part of the robot and no other parcel "
                                   "touches it; it has not been damaged"),
                      "duty": {"kinds": [str(k) for k in (sc.get("duty") or {}).get("kinds", [])],
                               "label_positions": [str(k) for k in (sc.get("duty") or {}).get("label_positions", [])]},
                      "classes": {"W": "in the station's duty (kind and label position at arrival), label not up",
                                  "U": "arrived label up (any kind)",
                                  "X": "outside the duty, label not up: passes untouched to the exception lane"},
                      "credit": "W readable - U not readable - lost (any class)"},
        }
        self._describe = _jsonable(out)
        return json.loads(json.dumps(self._describe))

    READABLE_TEXT = ("at the tick the parcel's centre first crosses the scan line going downstream: label within "
                     "label_tilt_deg of vertical up; it rests on the belt (it touched the belt surface in at least one "
                     "tick of the last rest_window_s, that tick included, and its lowest point is at most "
                     "rest_clearance_max_m above the belt surface; one tick's contact never decides), turns slower than "
                     "max_angular_speed_rad_s and moves slower than max_rel_speed_m_s relative to the belt (horizontal "
                     "velocity); no part of the robot and no other parcel touches it; it has not been damaged")

    def _shift_describe(self):
        if not self.line:
            return {"duration_s": self.duration}
        st = self.shift.setting
        lc = self.cfg["line"]["shifts"]
        z = [float(v) for v in st["zone_y"]]
        return {"duration_s": self.duration,
                "end": "the shift ends at duration_s, or at the end of the tick in which the last parcel of its list "
                       "is settled, whichever comes first",
                "arrivals": ("one parcel of the shift's list in the station at a time: the first is set down at the "
                             "shift's first tick, each next one at the first tick after the one before it is settled "
                             "(its centre crossed the scan line going downstream, it fell off the belt, or it was "
                             "damaged); set down at rest %g m above the belt on the belt's centre line, between y "
                             "%g and %g, at any yaw" % (0.002, float(lc["set_down_y"][0]), float(lc["set_down_y"][1]))),
                "list": ("the list holds as many parcels as fit into duration_s at the station's rated cycle: the "
                         "belt's ride over %g m (from the middle of the set-down range to the scan line) plus %g s per "
                         "parcel" % (float(lc["rated_ride_m"]), float(lc["rated_handling_s"]))),
                "due": ("each parcel is due at the scan line by its set-down time plus its free ride (from its "
                        "set-down y to the scan line at the shift's belt speed) plus %g s: the station's rated cycle "
                        "allows it that much delay; a parcel that crosses later is not readable"
                        % float(self.score_cfg["due_margin_s"])),
                "due_margin_s": float(self.score_cfg["due_margin_s"]),
                "belt": ("one speed for the whole shift, along +y; on an indexing line the belt stands while the "
                         "centre of the parcel in the station is between y %g and %g and runs again when it has left "
                         "that zone" % (z[0], z[1])),
                "indexing_zone_y": z}

    def _score_describe(self):
        sc = self.score_cfg
        return {"label_tilt_deg": float(sc["label_tilt_deg"]),
                "max_angular_speed_rad_s": float(sc["max_angular_speed_rad_s"]),
                "max_rel_speed_m_s": float(sc["max_rel_speed_m_s"]),
                "rest_window_s": float(sc["rest_window_s"]),
                "rest_clearance_max_m": float(sc["rest_clearance_max_m"]),
                "readable": self.READABLE_TEXT + "; and it crossed in time: at the latest at its due time (shift.due)",
                "duty": {"kinds": [str(k) for k in (sc.get("duty") or {}).get("kinds", [])],
                         "label_positions": [str(k) for k in (sc.get("duty") or {}).get("label_positions", [])]},
                "classes": {"W": "arrived with its label not up (any kind, any label position)",
                            "U": "arrived label up (any kind)"},
                "done": "a parcel of the list is done when it was judged at the scan line, fell off the belt or was "
                        "damaged; a parcel not done when the shift ends (never set down, or in the station and not "
                        "across) costs nothing and earns nothing",
                "lost": "a parcel that fell off the belt or was damaged",
                "credit": "W readable - U done and not readable - lost",
                "shift_score": "max(0, credit) / (number of W parcels in the list)"}

    def _tool_describe(self):
        c = self.tool.cfg
        return {"frame": "the tool frame: origin at the tool centre point; z along the tool axis (from the "
                         "flange towards the fingertips), x along the fingers' closing direction, y = z x x; "
                         "pose = position (m) and unit quaternion (w, x, y, z) in the world frame",
                "command": "step(tool_pose={arm: {pos, quat}}): per arm, instead of its joint targets; "
                           "one level per arm per step",
                "solver": "damped least squares from the present solution, at most %d iterations per tick, "
                          "the joint step scaled to the command speed limit, joint ranges less %g rad; "
                          "no obstacle avoidance" % (int(c["iterations"]), float(c["joint_margin_rad"])),
                "gravity_correction": bool(c.get("gravity_compensation", True)),
                "reached_m": float(c["reached_m"]), "reached_deg": float(c["reached_deg"])}

    def _tool_measured(self, q):
        t0 = time.time()
        try:
            return self.tool.measured(q)
        finally:
            self._extra_work += time.time() - t0        # the controller's own work: never program time

    def _state(self, consume=True, true_tau=False):
        """The robot's state as the wire gives it: ``tau`` is the torque sensor's reading (noise and offset,
        sensors.TorqueSensor); ``true_tau``: the true motor torques (the privileged state)."""
        w = self.world
        js = w.joint_state()
        gs = w.gripper_state()
        stop = self.watch.state()
        tau = js.tau if true_tau else self.tau_sensor.read(js.tau, int(w.n_ticks), float(w.time))
        return {
            "t": round(float(w.time), 6),
            "q": _r(js.q), "dq": _r(js.qd), "tau": _r(tau, 4), "q_target": _r(js.q_target),
            "tool": self._tool_measured(js.q),
            "tool_target": self.tool.report(),
            "gripper": {"command": _r(gs.command, 4), "opening_m": self._openings(),
                        "object_detected": [bool(v) for v in self.gripper_status.detected]},
            "stop": stop,
            "frames_left": int(self.cams.frames_left),
            "program_time_left": round(max(0.0, self.program_budget_s - self.program_time()), 3),
            "shift_over": bool(w.done),
            "over_reason": w.over_reason,
        }

    def _openings(self, rounded=True):
        """Each gripper's opening: the distance between its finger pads' inner
        faces along the closing axis (the gripper base's y axis)."""
        import mujoco
        w = self.world
        m, d = w.model, w.data
        ids = getattr(self, "_pad_ids", None)
        if ids is None or ids[0] is not w:
            per = []
            for s in SIDES:
                p = s[0]
                per.append((mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{p}_base"),
                            [(w._gid(f"{p}_left_pad{k}"), w._gid(f"{p}_right_pad{k}")) for k in ("1", "2")]))
            self._pad_ids = ids = (w, per)
        out = []
        for bid, pairs in ids[1]:
            R = d.xmat[bid].reshape(3, 3)
            gaps = [abs(float((R.T @ (d.geom_xpos[a] - d.geom_xpos[b]))[1]))
                    - m.geom_size[a][1] - m.geom_size[b][1] for a, b in pairs]
            g = max(0.0, min(gaps))
            out.append(round(g, 4) if rounded else g)
        return out

    def get_state(self):
        return self._state(consume=True)

    def step(self, q_target=None, grip=None, n=1, tool_pose=None):
        w = self.world
        self.step_calls += 1
        n = max(1, min(int(n), int(self.cfg["budgets"]["max_ticks_per_step"])))
        refused = None
        if q_target is not None or grip is not None or tool_pose is not None:
            if self.watch.stopped:
                refused = "locked_out" if self.watch.locked_out else "protective_stop"
                self.refused_calls += 1
            else:
                if isinstance(q_target, dict):                      # per arm
                    for k, v in q_target.items():
                        if v is None:
                            continue
                        side = SIDES[ARM_KEYS.index(k)]
                        ix = self.tool.idx[side]
                        self._q_hold = self._q_hold.copy()
                        self._q_hold[ix] = np.clip(np.asarray(v, float).reshape(7), w._ctrl_lo[ix], w._ctrl_hi[ix])
                        self.tool.clear(side)
                elif q_target is not None:
                    q = np.asarray(q_target, float).reshape(14)
                    self._q_hold = np.clip(q, w._ctrl_lo, w._ctrl_hi)
                    self.tool.clear()
                if tool_pose:
                    for k, p in tool_pose.items():
                        if p is not None:
                            self.tool.set_target(SIDES[ARM_KEYS.index(k)], p["pos"], p["quat"], w._q_applied)
                if grip is not None:
                    self._grip_hold = np.clip(np.asarray(grip, float).reshape(2), 0.0, 1.0)
        ran = 0
        if not w.done:
            for _ in range(n):
                if w.done:
                    break
                if self.tool.active:
                    t0 = time.time()
                    self._q_hold = self.tool.tick(self._q_hold, w._q_applied)
                    self._extra_work += time.time() - t0    # the controller's own work: never program time
                w.step(self._q_hold, self._grip_hold)
                ran += 1
        out = self._state(consume=True)
        out["ticks"] = ran
        out["refused"] = refused
        return out

    def get_camera(self, name, depth=False):
        if name not in self.cams.specs:
            return {"error": "bad_params", "detail": "unknown camera %r (cameras: %s)"
                    % (str(name)[:WORD_MAX], ", ".join(CAMERA_NAMES))}
        try:
            return self.cams.get(name, bool(depth))
        except NoFramesLeft:
            return {"error": "no_frames_left"}
        except NoFrameYet:
            return {"error": "no_frame_yet"}
        except RenderError as e:
            self.log("[cell] camera %s: %s" % (name, e))
            return {"error": "camera_unavailable"}

    def finish(self):
        self._end("finished")
        return {"ok": True, "t": round(float(self.world.time), 6)}

    # ================================================================ privileged
    def get_state_privileged(self):
        w = self.world
        parcels = []
        for ps in w.truth.parcel_states():
            p = w.slots[ps.slot].parcel
            parcels.append({"i": ps.slot, "arrival": ps.arrival, "kind": ps.kind,
                            "size": list(ps.size), "mass": p.mass, "friction": p.friction,
                            "label_face": p.label_face, "label_uv": list(p.label_uv),
                            "com_offset": list(p.com_offset),
                            "pos": ps.pos, "quat": ps.quat, "linvel": ps.linvel, "angvel": ps.angvel,
                            "label_normal": ps.label_normal,
                            "label_angle_deg": math.degrees(ps.label_angle),
                            "on_belt": ps.on_belt, "touching_robot": ps.touching_robot,
                            "touching_parcel": ps.touching_parcel,
                            "damaged": ps.slot in self.watch.damaged})
        pending = [{"i": k, "t": a.t, "parcel": json.loads(json.dumps(_parcel_dict(a.parcel)))}
                   for k, a in enumerate(self.shift.arrivals) if k >= w._next_arrival]
        tcp = {s.lower(): {"pos": w.tcp_pose(s)[0], "R": w.tcp_pose(s)[1]} for s in SIDES}
        return _jsonable({"t": float(w.time), "state": self._state(consume=False, true_tau=True),
                          "parcels": parcels, "pending": pending,
                          "belt": w.truth.belt_speed(), "tcp": tcp,
                          "record_so_far": self.record.so_far(),
                          "safety": self.watch.summary(),
                          "events": w.truth.events[-200:]})

    def truth(self):
        return self.get_state_privileged()

    def reference_run(self):
        """Runs the reference program inside the cell: reference_dir/controller.py,
        run(server). The reference acts through the cell's own command path (step,
        or world.step), so the speed limit, the safety watch and the damage rule
        apply to it."""
        if not self.reference_dir:
            return {"error": "no_reference", "detail": "this cell has no reference program"}
        path = os.path.join(self.reference_dir, "controller.py")
        spec = importlib.util.spec_from_file_location("parcelcell_reference_controller", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t0 = time.time()
        self._ref = (t0, self.work_wall)
        try:
            result = mod.run(self)
        except ShiftOver:
            result = {"note": "stepped past the shift's end"}
        finally:
            self._ref = None
        self.reference_result = _jsonable(result)
        return _jsonable({"ok": True, "wall_s": round(time.time() - t0, 3),
                          "t": round(float(self.world.time), 4),
                          "summary": (result or {}).get("summary") if isinstance(result, dict) else None})

    # ================================================================ dispatch
    def handle(self, method, params):
        method = str(method)
        key = method if method in KNOWN_METHODS else "_unknown"
        self.calls[key] = self.calls.get(key, 0) + 1
        live = not self.done
        try:
            return self._dispatch(method, params)
        except Exception:  # noqa: BLE001 - nothing a request does may crash the wire
            self.internal_errors += 1
            if live:
                self.internal_errors_live += 1
            self.log("[cell] internal error in %s\n%s" % (method, traceback.format_exc()))
            if method in STATE_METHODS:
                self._end("internal_error")
            return {"error": "internal_error"}

    def _dispatch(self, method, params):
        if params is None:
            params = {}
        if not isinstance(params, dict):
            self.bad_params += 1
            return {"error": "bad_params", "detail": "params must be an object"}
        if method in PRACTICE_METHODS:
            if not (self.mode == "practice" and (self.allow_reset or method == "truth")):
                self.illegal_calls += 1
                return {"error": "forbidden"}
            if method == "truth":
                return self.truth()
            return self._practice_reset(params)
        if method in PRIVILEGED_METHODS:
            if not self.privileged:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            return getattr(self, method)()
        if method not in PUBLIC_METHODS:
            self.illegal_calls += 1
            return {"error": "forbidden"}
        if self.world is None:
            return {"error": "no_shift", "detail": "the shift has not started"}
        try:
            params = check_args(method, params)       # the boundary: nothing runs for a call that fails it
        except BadParams as e:
            self.bad_params += 1
            return {"error": "bad_params", "detail": str(e)[:200]}
        out = getattr(self, method)(**params)
        if isinstance(out, dict) and out.get("error") == "bad_params":
            self.bad_params += 1
        return _jsonable(out)

    def _practice_reset(self, params):
        if "package" in params:
            self.reset(params["package"])
        else:
            seed = int(params.get("seed", 0))
            prof = str(params.get("profile", "full"))
            if prof.startswith("line"):
                sh, meta = draw_line_shift(seed, self.cfg, profile="bench" if prof == "line_bench" else "full")
            else:
                sh, meta = draw_shift(seed, self.cfg, profile=prof)
            self.reset(sh, meta)
        return {"ok": True, "shift_sha256": self.shift_sha256, "duration_s": self.duration}

    # ================================================================ end of shift
    def finalize(self):
        """The shift's record (idempotent; never raises on a built cell)."""
        w = self.world
        if not w.done:
            self._end("finished")
        score = self.record.finalize()
        cams = self.cams
        rt = {n: (sorted(v)[len(v) // 2] if v else None) for n, v in cams.render_times.items()}
        summary = {
            "cell_version": CELL_VERSION,
            "shift": {"sha256": self.shift_sha256, "seed": self.shift.seed, "meta": self.meta,
                      "duration_s": self.duration, "belt_speed": self.shift.belt_speed[0][1],
                      "n_parcels": len(self.shift.arrivals)},
            "done_reason": w.over_reason, "t_end": round(float(w.time), 4), "ticks": int(w.n_ticks),
            "shift_score": score["shift_score"], "credit": score["credit"],
            "n_W": score["n_W"], "n_U": score["n_U"], "n_X": score["n_X"], "W_readable": score["W_readable"],
            "U_not_readable": score["U_not_readable"], "lost": score["lost"],
            "lost_by": score["lost_by"], "lost_by_class": score["lost_by_class"], "duty": score["duty"], "max_entry_delay_s": score["max_entry_delay_s"],
            "parcels": score["parcels"],
            "safety": self.watch.summary(),
            "parcel_damage": {str(k): v for k, v in self.watch.damaged.items()},
            "frames": {"used": cams.used, "budget": cams.budget, "by_camera": dict(cams.by_camera),
                       "repeats": cams.repeats, "render_wall_s": round(cams.render_wall, 3),
                       "render_failures": cams.render_failures, "frames_lost": cams.frames_lost,
                       "render_median_s": rt, "look": self.look_name, "renderer": self._renderer_state()},
            "tool_pose": self.tool.summary(),
            "program_time_s": round(float(self.program_used or self.program_time()), 3),
            "program_budget_s": self.program_budget_s,
            "cell_work_s": round(self.work_wall, 3), "tick_wall_s": round(w.tick_wall, 3),
            "wall_time_s": round(float(self.wall_used), 3), "build_wall_s": round(self.build_wall_s, 3),
            "calls": dict(self.calls), "step_calls": int(self.step_calls),
            "steps_after_done": int(self.steps_after_done),
            "illegal_calls": int(self.illegal_calls), "bad_params": int(self.bad_params),
            "refused_calls": int(self.refused_calls),
            "internal_errors": int(self.internal_errors),
            "internal_errors_live": int(self.internal_errors_live),
            "privileged": self.privileged, "reference": self.reference_result,
            "safety_events": [[round(t, 3), e] for t, e in self.watch.events][:2000],
        }
        if cams.crushed:                       # the crushed looks drawn (a damaged parcel in a rendered frame)
            summary["frames"]["crushed"] = {str(k): v for k, v in cams.crushed.items()}
        if self.line:
            for k in ("setting", "not_done", "done", "list_done_t", "belt_halts", "belt_halted_s", "spawned", "late",
                      "due_margin_s"):
                summary[k] = score[k]
        return _jsonable(summary)

    def _renderer_state(self):
        """The filament renderer's view state for the record (dithering off, the view found, the probe tries of the
        search, the renderer's starts in this process: ``attempts``, at most 3); None otherwise."""
        lk = getattr(self.cams, "_look", None)
        if lk is None or getattr(lk, "name", "") != "filament":
            return None
        try:
            if getattr(lk, "engine", None) is None:      # no renderer: none started yet, or every start failed
                from .filament_engine import Engine
                st = Engine.start_state()
                if st is None:
                    return None
                return {"view_found": False, "dithering": None, "tries": None, "attempts": st.get("attempts")}
            v = dict(lk.engine.view_state)
        except Exception:  # noqa: BLE001 - a record of the renderer is never a scoring input
            return None
        return {"view_found": v.get("view_found"), "dithering": v.get("dithering"), "tries": len(v.get("tries") or []),
                "attempts": v.get("attempts")}

    def log_lines(self, summary=None):
        """A few lines for the maintainer's log."""
        s = summary or self.finalize()
        lines = ["shift %s (%s, %d parcels, belt %.3f m/s, %.0f s): score %.4f credit %d "
                 "(W %d readable %d, U %d not readable %d, X %d, lost %d) end %s"
                 % (s["shift"]["sha256"][:12], s["shift"]["meta"].get("belt_class", "?"),
                    s["shift"]["n_parcels"], s["shift"]["belt_speed"], s["shift"]["duration_s"],
                    s["shift_score"], s["credit"], s["n_W"], s["W_readable"], s["n_U"],
                    s["U_not_readable"], s["n_X"], s["lost"], s["done_reason"]),
                 "  stops %d locked_out %s peaks %s frames %d/%d program %.1f s cell %.1f s"
                 % (s["safety"]["protective_stop_count"], s["safety"]["locked_out"],
                    s["safety"]["peak_force_N"], s["frames"]["used"], s["frames"]["budget"],
                    s["program_time_s"], s["cell_work_s"])]
        for r in s["parcels"]:
            lines.append("  #%02d %s %-5s %-10s arr %.2f scan %s readable %s %s lost %s"
                         % (r["i"], r["cls"], r["kind"], r["label_position"], r["t_arrival"],
                            None if r["scan"] is None else r["scan"]["t"], r["readable"],
                            ",".join(r["failed"]), r["lost_reason"]))
        return lines


def _parcel_dict(p):
    from dataclasses import asdict
    return asdict(p)
