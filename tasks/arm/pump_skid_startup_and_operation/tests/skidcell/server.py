"""SkidCellServer: the grader-owned physics, the process and the skid cell's API (cell internals: the hidden grader runs this
class in `hidden` mode on the pinned hidden seeds; `practice` mode is a maintainers' development mode that nothing in the
image runs).

The server owns the MuJoCo model and data and the process model, steps both only inside `step`, evaluates every check
from its own state, applies the grip rule, reads the panel's buttons and selector, watches safety and writes the
episode summary. The program talks to it over the JSON-lines wire; nothing it sends is a scoring input except its
operator log (`log_reading`), which is checked against the instruments' true values.

API (metres / rad / wxyz; `cell_api.md` is the agent-facing contract):
  describe()                                  static facts (robot, rail, control, grip rule, cameras, budgets, safety)
  get_state()                                 sim_time, done, the arm's own state (the wrist's 6-D external wrench
                                              estimate, the protective stop), the rail, the horn
  set_control(mode, stiffness_scale)          joint | cartesian; stiffness_scale in [0.2, 1.0]
  command(q | tcp_pos+tcp_quat, width)        targets held until the next command
  command_rail(x, lift)                       the carriage along the rail and the lift's height
  step(n_ticks)                               advance n x 10 ms; {sim_time, done, events}
  get_camera(name, depth)                     RGB (png b64) + depth (16-bit png, mm)
  log_reading(tag, value, unit)               an entry in the operator's log sheet
  finish()                                    ends the episode
Practice servers (a maintainers' development mode) additionally accept reset(seed) and truth(); privileged servers
get_state_privileged() and reference_run() (the reference program inside the cell). The commissioning service of the
agent's sandbox (environment/commission_service/commissiond.py, v0.6) runs this class in `hidden` mode on its
development sites and answers `reset` itself; nothing of truth() or the score reaches its wire.

Safety (v0.6): a press of the robot on the skid past the safety limits for 3 ticks is a PROTECTIVE STOP, as on a
collaborative robot: the arm and the carriage halt where they are, motion commands are refused for
safety.stop_lock_s, the stop is counted and reported (`protective_stop:<robot body>:<N>N` in step's events,
get_state()["arm"]["protective_stop"]). Only damage-level contact (safety.damage_force_N for 3 ticks) or more than
safety.max_stops stops in a shift void the episode; either locks the robot out for the rest of the shift.

`handle()` never raises: malformed parameters are answered {"error": "bad_params: ..."}, any other exception is logged
and answered {"error": "internal_error"} (inside a call that changes the cell it also ends the episode, whose state is
then scored as it stands). The attributes the wire relies on: done, done_reason, _end(reason), wall_start, wall_used,
work_wall (the cell's own time inside step and get_camera, not charged to the program), program_used,
steps_after_done.
"""

import base64
import importlib.util
import io
import json
import math
import os
import time
import traceback

import numpy as np

from . import CELL_VERSION
from .episode import Monitor
from .layout import CONC_BAND, LEVEL_BAND, RAIL, SHIFT_END_S, STARTUP_DEADLINE_S, draw_layout
from .physics import BUTTONS, GRIP, WELD_ROT_ROWS, Scene
from .process import Process

__all__ = ["SkidCellServer", "DEFAULT_CONFIG", "PUBLIC_METHODS", "PRIVILEGED_METHODS", "WRENCH_NOISE"]

PUBLIC_METHODS = ("describe", "get_state", "set_control", "command", "command_rail", "step", "get_camera", "log_reading", "finish")
PRIVILEGED_METHODS = ("get_state_privileged", "reference_run")
PRACTICE_METHODS = ("reset", "truth")
STATE_METHODS = ("set_control", "command", "command_rail", "step", "reference_run", "reset")
CAMERA_NAMES = ("overview", "wrist")
BUTTON_MAP = {"START-101": "START-101", "STOP-101": "STOP-101", "START-301": "START-301", "STOP-301": "STOP-301",
              "RESET": "RESET", "ESTOP": "ESTOP"}

# the wrist's external wrench estimate (v0.6), a Franka-like joint-torque estimate: white noise per reading and a
# posture-dependent offset (the dynamic model's error: constant while the arm stands still, it changes as the arm moves)
WRENCH_NOISE = {"sigma_N": 0.3, "sigma_Nm": 0.03, "offset_N": (0.5, 2.0), "offset_Nm": (0.05, 0.2), "offset_rad_per_rad": (0.3, 1.0)}

DEFAULT_CONFIG = {
    "sim_cap_s": 1800.0,
    "program_time_cap_s": 2400.0,
    "silence_s": 180.0,
    "max_ticks_per_step": 1000,
    "camera_budget_frames": 900,
    "safety": {"robot_force_N": 40.0, "device_force_N": 150.0, "press_ticks": 3, "stop_lock_s": 5.0, "damage_force_N": 400.0,
               "max_stops": 5},
    "depth_noise": {"sigma_mm": 1.5, "dropout": 0.02, "min_m": 0.10, "max_m": 6.0},
    "rgb_noise_sigma": 2.0,
}


def _deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class SkidCellServer(object):
    def __init__(self, given_dir, config=None, mode="hidden", privileged=False, allow_reset=False, render=True, log=None,
                 reference_dir=None):
        import mujoco
        self.mujoco = mujoco
        self.given_dir = given_dir
        self.cfg = _deep_update(json.loads(json.dumps(DEFAULT_CONFIG)), config)
        self.mode = mode
        self.privileged = bool(privileged)
        self.allow_reset = bool(allow_reset)
        self.render_enabled = bool(render)
        self.log = log or (lambda s: None)
        self.reference_dir = reference_dir
        self.scene = None
        self.done = False
        self.done_reason = None
        self.wall_start = None
        self.wall_used = 0.0
        self.program_used = 0.0
        self.work_wall = 0.0
        self.internal_errors = 0

    # ------------------------------------------------------------------ episode setup
    def reset(self, seed):
        self.seed = int(seed)
        self.layout = draw_layout(self.seed, self.cfg)
        t0 = time.time()
        self.scene = Scene(self.given_dir, self.layout, self.cfg)
        self.build_wall_s = time.time() - t0
        self.proc = Process(self.layout)
        self.monitor = Monitor(self.layout)
        self.tick_s = self.scene.tick_s
        self.done = False
        self.done_reason = None
        self.tick_count = 0
        self.events_log = []
        self.violations = []
        self._press = {}
        self.illegal_calls = 0
        self.internal_errors = 0
        self.calls = {}
        self.step_calls = 0
        self.frames_used = 0
        self.frames_by_camera = dict((c, 0) for c in CAMERA_NAMES)
        self.render_wall_s = 0.0
        self.render_failures = 0
        self.frames_lost = 0                 # frames not rendered after the retry (camera_unavailable)
        self._closing = False
        self._slip = 0
        self._btn_down = dict((b, False) for b in BUTTONS)
        self._selector = self.layout["initial"].get("HS-101", "OFF")
        self.proc.set_selector(self._selector)
        self._horn = False
        self._marks = {"window_open": False, "shift_end": False}
        self._noise_rng = np.random.default_rng([self.seed & 0xFFFFFFFF, 0x5C1DCE11])
        self._f6 = np.zeros(6)
        # v0.6: protective stops; the wrist's wrench estimate's own generator (the cameras' and the cell's draws are
        # untouched: nothing here feeds back into the simulation)
        self.stops = []
        self._stop_until = None
        self._lockout = False
        self._latched = set()
        wr = np.random.default_rng([self.seed & 0xFFFFFFFF, 0x5C1DF7E5])
        W = WRENCH_NOISE
        self._ft_rng = wr
        self._ft_amp = np.concatenate([wr.uniform(*W["offset_N"], size=3), wr.uniform(*W["offset_Nm"], size=3)])
        self._ft_w = wr.uniform(*W["offset_rad_per_rad"], size=(6, 7)) * wr.choice([-1.0, 1.0], size=(6, 7))
        self._ft_phase = wr.uniform(0.0, 2.0 * math.pi, size=6)
        self._ft_sigma = np.array([W["sigma_N"]] * 3 + [W["sigma_Nm"]] * 3)
        return self.describe()

    # ------------------------------------------------------------------ API
    def describe(self):
        sc = self.scene
        arm = sc.arm
        return {
            "cell_version": CELL_VERSION,
            "units": {"length": "m", "angle": "rad", "quaternion": "wxyz", "time": "s"},
            "robot": {"model": "franka_emika_panda with its hand (MuJoCo Menagerie, Apache-2.0)",
                      "mount": "on the lift of a carriage that runs along the rail; the base faces +y (the skid)",
                      "joints": ["panda/joint%d" % (i + 1) for i in range(7)], "joint_range": arm.ranges(),
                      "home_q": [float(v) for v in arm.home_q], "tcp_site": "panda/tcp",
                      "tcp": "between the finger pads, 0.1034 m from the hand along its +z (the approach axis); +y = the fingers' closing axis",
                      "gripper": {"max_width_m": 0.08, "rate_m_s": 0.05}},
            "rail": {"axis": "x", "y": RAIL["y"], "x_range": [RAIL["x_lo"], RAIL["x_hi"]], "speed_m_s": RAIL["speed_m_s"],
                     "lift_range": [RAIL["lift_lo"], RAIL["lift_hi"]], "lift_speed_m_s": RAIL["lift_speed_m_s"],
                     "base": "the Panda's base (link0 origin) stands at (x, %.2f, lift)" % RAIL["y"]},
            "control": {"tick_s": self.tick_s, "joint_rate_limit_rad_s": 1.5, "cartesian_speed_limit_m_s": 0.5,
                        "cartesian_angular_limit_rad_s": 1.5, "stiffness_scale_range": [0.2, 1.0], "modes": ["joint", "cartesian"],
                        "cartesian_frame": "world (the skid's frame): the arm alone tracks the target from wherever the carriage stands"},
            "grip_rule": {"pinch_box_m": {"x": GRIP["box_x"], "z": [GRIP["box_z_lo"], GRIP["box_z_hi"]], "y": "half the opening"},
                          "axis_tolerance_deg": GRIP["angle_deg"], "slip_force_N": GRIP["slip_force_N"],
                          "slip_torque_Nm": GRIP["slip_torque_Nm"],
                          "slip_torque_about": "the held part's pivot (where its stem's axis meets the handle's plane)",
                          "slip_after_s": GRIP["slip_ticks"] * self.tick_s},
            "wrist_load": {"what": "the external wrench on the hand, estimated from the joint torques (as a Franka's O_F_ext_hat_K)",
                           "frame": "world axes; force_N at the tcp, torque_Nm about the tcp", "sign": "what the surroundings exert on the hand",
                           "noise": {"sigma_N": WRENCH_NOISE["sigma_N"], "sigma_Nm": WRENCH_NOISE["sigma_Nm"],
                                     "offset_up_to_N": WRENCH_NOISE["offset_N"][1], "offset_up_to_Nm": WRENCH_NOISE["offset_Nm"][1],
                                     "offset": "changes with the arm's posture, constant while it stands still"}},
            "cameras": sc.cameras.describe(),
            "skid": {"front_plane_y": 0.0, "x_extent": [0.0, self.layout["x_total"]], "floor_z": 0.0},
            "site": {"name": self.layout["site"]["name"], "equipment": [dict(e) for e in self.layout["site"]["equipment"]],
                     "log_sheet": list(self.layout["site"]["log_sheet"])},
            "schedule": {"startup_deadline_s": STARTUP_DEADLINE_S, "shift_end_s": SHIFT_END_S, "sim_cap_s": self.cfg["sim_cap_s"],
                         "level_band_pct": list(LEVEL_BAND), "conc_band_mgL": list(CONC_BAND)},
            "budgets": {"sim_time_cap_s": self.cfg["sim_cap_s"], "program_time_cap_s": self.cfg["program_time_cap_s"],
                        "program_time": "the episode's wall clock minus the cell's own time inside step (simulation) and get_camera (rendering)",
                        "silence_abort_s": self.cfg["silence_s"], "max_ticks_per_step": self.cfg["max_ticks_per_step"],
                        "camera_frames": int(self.cfg["camera_budget_frames"])},
            "safety": dict(self.cfg["safety"]),
        }

    def get_state(self):
        sc = self.scene
        arm = sc.arm
        p, R = arm.tcp_actual()
        return {"sim_time": float(sc.data.time), "done": bool(self.done),
                "arm": {"q": [float(v) for v in arm.q()], "dq": [float(v) for v in arm.dq()], "cmd_q": [float(v) for v in arm.q_cmd],
                        "tcp_pos": [float(v) for v in p], "tcp_quat": [float(v) for v in arm.mat_to_quat(R)], "mode": arm.mode,
                        "stiffness_scale": arm.stiffness_scale, "gripper_width": arm.width(), "holding": sc.held is not None,
                        "wrist_load": self._wrist_load(), "protective_stop": self._stop_state()},
                "rail": {"x": sc.rail_x(), "lift": sc.lift_z(), "x_goal": sc.rail_goal, "lift_goal": sc.lift_goal},
                "horn": bool(self.proc.horn)}

    def _wrist_load(self):
        """The wrist's external wrench estimate (world axes, at the tcp) with the sensor's noise and posture offset."""
        F, M = self.scene.external_wrench()
        q = self.scene.arm.q()
        w = np.concatenate([F, M]) + self._ft_amp * np.sin(self._ft_w @ q + self._ft_phase) + \
            self._ft_sigma * self._ft_rng.standard_normal(6)
        return {"force_N": [round(float(v), 4) for v in w[:3]], "torque_Nm": [round(float(v), 5) for v in w[3:]]}

    def _stop_state(self):
        t = float(self.scene.data.time)
        left = 0.0 if self._stop_until is None else max(0.0, self._stop_until - t)
        return {"active": bool(self._lockout or self._stop_until is not None), "remaining_s": round(left, 2),
                "count": len(self.stops), "locked_out": bool(self._lockout)}

    def _motion_refused(self):
        """The reply to a motion command while the robot is stopped (None when it may move)."""
        if self._lockout:
            return {"ok": False, "error": "locked_out", "protective_stops": len(self.stops)}
        if self._stop_until is not None:
            return {"ok": False, "error": "protective_stop",
                    "remaining_s": round(max(0.0, self._stop_until - float(self.scene.data.time)), 2)}
        return None

    def set_control(self, mode="joint", stiffness_scale=1.0):
        a = self.scene.arm
        a.set_mode(str(mode))
        a.set_stiffness(float(stiffness_scale))
        return {"ok": True, "mode": a.mode, "stiffness_scale": a.stiffness_scale}

    def command(self, q=None, tcp_pos=None, tcp_quat=None, width=None):
        if self.done:
            return {"ok": False, "error": "episode_over"}
        a = self.scene.arm
        if q is not None or tcp_pos is not None or tcp_quat is not None:
            refused = self._motion_refused()          # v0.6: no motion while the arm is in a protective stop
            if refused is not None:
                return refused
        if q is not None:
            if a.mode != "joint":
                return {"ok": False, "error": "q target needs joint mode"}
            a.command_joint(q)
        if tcp_pos is not None or tcp_quat is not None:
            if a.mode != "cartesian":
                return {"ok": False, "error": "tcp target needs cartesian mode"}
            if tcp_pos is None or tcp_quat is None:
                return {"ok": False, "error": "tcp_pos and tcp_quat go together"}
            a.command_cartesian(tcp_pos, tcp_quat)
        if width is not None:
            w = float(width)
            if not math.isfinite(w):
                raise ValueError("non-finite width")
            self._closing = (w < a.width() - 0.002) or w <= 0.001
            if self.scene.held is not None and w > a.width() + 0.002:
                self._release("opened")
            a.command_width(w)
        return {"ok": True}

    def command_rail(self, x=None, lift=None):
        if self.done:
            return {"ok": False, "error": "episode_over"}
        if x is not None or lift is not None:
            refused = self._motion_refused()
            if refused is not None:
                return refused
        self.scene.command_rail(x=x, lift=lift)
        return {"ok": True, "x_goal": self.scene.rail_goal, "lift_goal": self.scene.lift_goal}

    def step(self, n_ticks=1):
        n = int(n_ticks)
        if n < 1:
            return {"ok": False, "error": "n_ticks must be >= 1"}
        n = min(n, int(self.cfg["max_ticks_per_step"]))
        events = []
        self.step_calls += 1
        t = time.time()
        try:
            for _ in range(n):
                if self.done:
                    break
                events += self.physics_tick()
                if self.scene.data.time >= self.cfg["sim_cap_s"] - 1e-9:
                    self._end("sim_cap")
        finally:
            self.work_wall += time.time() - t
        ev = []
        for e in events:
            if e.startswith(("grasped:", "released:", "slipped:")):
                ev.append(e.split(":", 1)[0])          # the hand feels a grip; it does not know the device's tag
            elif e.startswith(("safety:", "protective_stop", "horn_", "window_open", "shift_end", "physics_fault")):
                ev.append(e)
        if self.done:
            ev.append("episode_over")
        return {"sim_time": float(self.scene.data.time), "done": bool(self.done), "events": ev}

    def get_camera(self, name="overview", depth=True):
        name = str(name)
        if name not in CAMERA_NAMES:
            return {"error": "unknown camera (cameras: %s)" % ", ".join(CAMERA_NAMES)}
        if not self.render_enabled:
            return {"error": "camera_unavailable"}
        if self.frames_used >= int(self.cfg["camera_budget_frames"]):
            return {"error": "camera_budget"}
        k = self.frames_used
        self.frames_used += 1
        self.frames_by_camera[name] += 1
        t_frame = time.time()
        try:
            out = self._frame(name, k, depth)
            if isinstance(out, dict) and out.get("error") == "camera_unavailable":
                # a frame the cell failed to render is not charged to the budget (53-scoring-11); the grader retries
                # an episode with render failures (verify.py)
                self.frames_used -= 1
                self.frames_by_camera[name] -= 1
            return out
        finally:
            self.work_wall += time.time() - t_frame

    def _frame(self, name, k, depth):
        t = time.time()
        sc = self.scene
        rgb = dep = None
        for attempt in range(2):                     # one retry: a transient renderer failure is the cell's
            try:
                sc.update_visuals(self.proc, float(sc.data.time))
                rgb, dep = sc.cameras.render(sc.data, name)
                break
            except Exception as e:  # noqa: BLE001 - a renderer failure never takes the cell down
                self.log("[cell] %s frame %d not rendered (attempt %d): %s" % (name, k, attempt + 1, type(e).__name__))
                self.render_failures += 1
        if rgb is None:
            self.frames_lost += 1
            self.render_wall_s += time.time() - t
            return {"error": "camera_unavailable"}
        from PIL import Image
        rng = self._noise_rng
        sig = float(self.cfg.get("rgb_noise_sigma", 0.0))
        if sig > 0:
            rgb = np.clip(rgb.astype(np.float32) + rng.normal(0.0, sig, rgb.shape), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG", compress_level=1)
        out = {"ok": True, "camera": name, "frame_index": int(k), "sim_time": float(sc.data.time), "width": int(rgb.shape[1]),
               "height": int(rgb.shape[0]), "format": "png", "rgb_png_b64": base64.b64encode(buf.getvalue()).decode("ascii")}
        if bool(depth):
            dn = self.cfg["depth_noise"]
            d = dep.astype(np.float32)
            d = d + rng.normal(0.0, float(dn["sigma_mm"]) * 1e-3, d.shape).astype(np.float32)
            invalid = ~np.isfinite(d) | (d < float(dn["min_m"])) | (d > float(dn["max_m"])) | (rng.random(d.shape) < float(dn["dropout"]))
            mm = np.clip(np.round(d * 1000.0), 0, 65535).astype(np.uint16)
            mm[invalid] = 0
            b2 = io.BytesIO()
            Image.fromarray(mm).save(b2, format="PNG", compress_level=1)
            out["depth_u16_mm_b64"] = base64.b64encode(b2.getvalue()).decode("ascii")
            out["depth_unit_m"] = 0.001
            out["depth_invalid_value"] = 0
        if name == "wrist":
            out["T_world_cam"] = sc.cameras.extrinsics(sc.data, name).tolist()
        self.render_wall_s += time.time() - t
        return out

    def log_reading(self, tag, value, unit):
        if self.done:
            return {"ok": False, "error": "episode_over"}
        rec = self.monitor.on_log(float(self.scene.data.time), tag, value, unit, self.proc)
        self._event("logged:%s" % rec["tag"])
        # the clock's round only: nothing in the reply depends on the skid's state (53-isolation-2)
        return {"ok": True, "tag": rec["tag"], "sim_time": rec["t"], "round": rec["round"], "in_window": rec["in_window"]}

    def finish(self):
        self._end("finish")
        return {"ok": True, "sim_time": float(self.scene.data.time)}

    # ------------------------------------------------------------------ privileged / practice
    def get_state_privileged(self):
        return self.truth()

    def truth(self):
        """Every device's pose and handles, the process state and the score so far: the practice cell's development call
        (and the reference program's eyes)."""
        sc = self.scene
        out = {"sim_time": float(sc.data.time), "state": self.get_state(), "devices": sc.device_truth(),
               "process": self.proc.state(), "values": self.proc.values(), "gauge_kinds": dict(self.layout["gauge_kinds"]),
               "selector": self._selector, "held": sc.held, "layout": {k: self.layout[k] for k in ("mirror", "z_suc", "z_dis", "z_mk", "initial",
                                                                                                    "wrong_at_start", "events", "faults", "styles",
                                                                                                    "meter_first", "panel_p101_left", "site")},
               "cameras": sc.cameras.truth(sc.data),
               "score_so_far": self.monitor.summary(self._voids(), float(sc.data.time))}
        return _jsonable(out)

    def reference_run(self):
        """Runs the reference program inside the cell (privileged; the grader passes its folder)."""
        if not self.reference_dir:
            return {"error": "no reference program"}
        path = os.path.join(self.reference_dir, "controller.py")
        spec = importlib.util.spec_from_file_location("skidcell_reference_controller", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t0 = time.time()
        result = mod.run(self)
        return {"ok": True, "wall_s": time.time() - t0, "sim_time": float(self.scene.data.time), "result": _jsonable(result)}

    # ------------------------------------------------------------------ dispatch
    def handle(self, method, params):
        self.calls[method] = self.calls.get(method, 0) + 1
        try:
            return self._dispatch(method, params)
        except Exception:  # noqa: BLE001 - nothing a request does may crash the wire
            self.internal_errors += 1
            self.log("[cell] internal error in %s\n%s" % (method, traceback.format_exc()))
            if method in STATE_METHODS:
                self._end("internal_error")
            return {"error": "internal_error"}

    def _dispatch(self, method, params):
        if method in PRACTICE_METHODS:
            if not (self.mode == "practice" and (self.allow_reset or method == "truth")):
                self.illegal_calls += 1
                return {"error": "forbidden"}
            if method == "reset":
                self.reset(int((params or {}).get("seed", 0)))
                return {"ok": True, "seed": self.seed}
            return self.truth()
        if method in PRIVILEGED_METHODS:
            if not self.privileged:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            return getattr(self, method)()
        if method not in PUBLIC_METHODS:
            self.illegal_calls += 1
            return {"error": "forbidden"}
        try:
            return getattr(self, method)(**(params or {}))
        except (TypeError, ValueError, KeyError, IndexError) as e:
            return {"error": "bad_params: %s" % str(e)[:200]}

    # ------------------------------------------------------------------ the tick
    def _event(self, e):
        self.events_log.append((float(self.scene.data.time), e))

    def _end(self, reason):
        if not self.done:
            self.done = True
            self.done_reason = reason

    def _release(self, why):
        sc = self.scene
        if sc.held is not None:
            tag = sc.held
            sc.release()
            sc.arm.hold_width(None)
            self._event("%s:%s" % ("slipped" if why == "slip" else "released", tag))
            return "%s:%s" % ("slipped" if why == "slip" else "released", tag)
        return None

    def physics_tick(self):
        sc = self.scene
        arm = sc.arm
        arm.tick()
        sc.tick_rail()
        sc.apply_detents()
        sc.step_physics()
        self.tick_count += 1
        t = float(sc.data.time)
        events = []
        if sc.physics_fault and not self.done:
            events.append("physics_fault")
            self._end("physics_fault")
        # the grip rule: closing fingers that reach a handle's width with the handle in the pinch box attach it
        if sc.held is None and self._closing:
            hit = sc.find_handle(max(arm.width(), 0.004) + 0.006)
            if hit is not None and arm.width() <= hit["width"] + 0.006:
                sc.attach(hit["tag"])
                arm.hold_width(hit["width"])
                self._closing = False
                events.append("grasped:%s" % hit["tag"])
        elif sc.held is not None:
            ld = sc.grip_load()
            if ld is not None and (ld[0] > GRIP["slip_force_N"] or ld[1] > WELD_ROT_ROWS * GRIP["slip_torque_Nm"]):
                self._slip += 1
                if self._slip >= GRIP["slip_ticks"]:
                    self._slip = 0
                    e = self._release("slip")
                    if e:
                        events.append(e)
            else:
                self._slip = 0
        # the panel
        for b in BUTTONS:
            depth = sc.button_depth(b)
            if not self._btn_down[b] and depth > 0.004:
                self._btn_down[b] = True
                self.proc.press(BUTTON_MAP[b])
                self.monitor.on_press(t, b)
                events.append("pressed:%s" % b)
            elif self._btn_down[b] and depth < 0.002:
                self._btn_down[b] = False
        sel = sc.selector(self._selector)
        if sel != self._selector:
            self._selector = sel
            self.proc.set_selector(sel)
            events.append("selector:%s" % sel)
        # the process and the procedure's checks
        u = sc.openings()
        self.proc.step(self.tick_s, u)
        self.monitor.tick(t, self.tick_s, self.proc, u, self._selector)
        if self.proc.horn != self._horn:
            self._horn = self.proc.horn
            events.append("horn_on" if self._horn else "horn_off")
        if not self._marks["window_open"] and t >= STARTUP_DEADLINE_S:
            self._marks["window_open"] = True
            events.append("window_open")
        if not self._marks["shift_end"] and t >= SHIFT_END_S:
            self._marks["shift_end"] = True
            events.append("shift_end")
        events += self._watch()
        for e in events:
            self.events_log.append((t, e))
        return events

    def _watch(self):
        """The safety watch (v0.6): a press of a robot link or finger past the limits for press_ticks ticks is a
        protective stop (the arm and the carriage halt, motion is refused for stop_lock_s; counted, reported); a press
        past damage_force_N for press_ticks ticks, or more than max_stops stops in the shift, is a safety void and
        locks the robot out for the rest of the shift. A press that was on when a stop came, or during its lock, counts
        again only once its force has fallen below the limit (the program gets the chance to back off)."""
        sc, m, d = self.scene, self.scene.model, self.scene.data
        events = []
        S = self.cfg["safety"]
        thr_r = float(S["robot_force_N"])
        thr_d = float(S["device_force_N"])
        thr_x = float(S.get("damage_force_N", 400.0))
        n_ticks = int(S.get("press_ticks", 3))
        t = float(d.time)
        if self._stop_until is not None and t >= self._stop_until - 1e-9:
            self._stop_until = None
            events.append("protective_stop_cleared")
        now = {}
        for i in range(d.ncon):
            con = d.contact[i]
            c1, c2 = sc.geom_class[con.geom1], sc.geom_class[con.geom2]
            robot1, robot2 = c1 in (1, 2), c2 in (1, 2)
            if robot1 == robot2:
                continue
            rg, og = (con.geom1, con.geom2) if robot1 else (con.geom2, con.geom1)
            oc = sc.geom_class[og]
            if oc not in (5, 7):
                continue
            self.mujoco.mj_contactForce(m, d, i, self._f6)
            key = (int(m.geom_bodyid[rg]), int(m.geom_bodyid[og]), int(oc))
            now[key] = max(now.get(key, 0.0), abs(float(self._f6[0])))
        press = {}
        for key, fn in now.items():
            thr = thr_d if key[2] == 7 else thr_r
            if fn > thr:
                cnt, peak, cx = self._press.get(key, (0, 0.0, 0))
                press[key] = (cnt + 1, max(peak, fn), cx + 1 if fn > thr_x else 0)
        self._press = press
        self._latched = set(k for k in self._latched if k in press)
        stopped = self._lockout or self._stop_until is not None
        if stopped:
            self._latched |= set(press)
        for key in sorted(press):
            cnt, peak, cx = press[key]
            link = m.body(key[0]).name
            other = m.body(key[1]).name
            tag = sc.body_tag.get(key[1])
            if cx == n_ticks:
                # damage-level contact: a void, and the robot is taken out of service for the rest of the shift
                what = "robot_damage:%s:%s:%.0fN" % (link, tag or other, peak)
                self.violations.append({"t": t, "what": what, "force_N": peak})
                events.append("safety:damage:%s:%.0fN" % (link, peak))
                if not self._lockout:
                    self._lockout = True
                    self._stop_until = None
                    self._halt()
                    events.append("safety:locked_out")
                stopped = True
                self._latched |= set(press)
            if cnt >= n_ticks and key not in self._latched and not stopped:
                # the protective stop: the event names the robot's link, never the device it pressed on
                self.stops.append({"t": t, "what": "%s:%s:%.0fN" % (link, tag or other, peak), "link": link, "force_N": peak,
                                   "class": "device" if key[2] == 7 else "structure"})
                events.append("protective_stop:%s:%.0fN" % (link, peak))
                self._halt()
                stopped = True
                self._latched |= set(press)
                if len(self.stops) > int(S.get("max_stops", 5)):
                    self._lockout = True
                    self.violations.append({"t": t, "what": "protective_stops:%d" % len(self.stops), "force_N": peak})
                    events.append("safety:locked_out")
                else:
                    self._stop_until = t + float(S.get("stop_lock_s", 5.0))
        return events

    def _halt(self):
        """Stop the arm and the carriage where they are (a protective stop): their targets become the measured pose."""
        sc = self.scene
        a = sc.arm
        q = np.clip(a.q(), a.lower, a.upper)
        a.q_cmd = q.copy()
        a.q_goal = q.copy()
        if a.mode == "cartesian":
            a.T_goal = a.fk_cmd()
        a.write_ctrl()
        sc.rail_goal = sc.rail_cmd = sc.rail_x()
        sc.lift_goal = sc.lift_cmd = sc.lift_z()

    def _voids(self):
        v = [{"t": float(t), "what": r} for t, r in self.proc.voids]
        v += [{"t": x["t"], "what": "safety:" + x["what"]} for x in self.violations]
        return v

    # ------------------------------------------------------------------ end of episode
    def finalize(self):
        if not self.done:
            self._end("solve_returned")
        sc = self.scene
        s = self.monitor.summary(self._voids(), float(sc.data.time))
        summary = {
            "cell_version": CELL_VERSION, "seed": self.seed, "done_reason": self.done_reason,
            "sim_time_s": float(sc.data.time), "wall_time_s": float(self.wall_used), "ticks": int(self.tick_count),
            "episode_score": float(s["score"]), "raw_score": float(s["raw_score"]), "voided": bool(s["voided"]), "voids": s["voids"],
            "parts": s["parts"], "startup": s["startup"], "lineup_at_start": s["lineup_at_start"], "first_start_t": s["first_start_t"],
            "window_fraction": s["window_fraction"], "events": s["events"], "shutdown": s["shutdown"], "log": s["log"], "logs": s["logs"],
            "violations": list(self.violations), "safety_ok": len(self.violations) == 0,
            "protective_stops": list(self.stops), "protective_stop_count": len(self.stops), "locked_out": bool(self._lockout),
            "t101_overflow_t": s.get("t101_overflow_t"),
            "illegal_call_count": int(self.illegal_calls), "internal_errors": int(self.internal_errors), "calls": dict(self.calls),
            "step_calls": int(self.step_calls), "program_time_s": float(self.program_used), "cell_work_s": float(self.work_wall),
            "camera_frames_total": int(self.frames_used), "camera_frames": dict(self.frames_by_camera),
            "render_wall_s": float(self.render_wall_s), "render_failures": int(self.render_failures), "frames_lost": int(self.frames_lost), "build_wall_s": float(self.build_wall_s),
            "process_log": [[round(t, 2), e] for t, e in self.proc.log], "events_log": [[round(t, 2), e] for t, e in self.events_log
                                                                                          if not e.startswith("logged:")][:4000],
            "layout": {"mirror": self.layout["mirror"], "wrong_at_start": self.layout["wrong_at_start"], "gauge_kinds": self.layout["gauge_kinds"],
                       "events": self.layout["events"], "faults": self.layout["faults"], "styles": self.layout["styles"],
                       "meter_first": self.layout["meter_first"], "panel_p101_left": self.layout["panel_p101_left"],
                       "site_area": self.layout["site"]["area"]},
        }
        return _jsonable(summary)
