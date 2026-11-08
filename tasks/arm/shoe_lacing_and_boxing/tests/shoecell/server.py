"""ShoeCellServer: the grader-owned physics and the finishing-cell API (PUBLIC CODE: the practice cell is this class in
`practice` mode; the hidden grader is this class in `hidden` mode on hidden seeds).

The server owns the MuJoCo model and data, steps physics only inside `step`, evaluates every predicate from its own
state, toggles the attach welds, the cradle clamp and the carton's snap-locks, watches safety and arrivals, and writes
the episode summary. The program talks to it over the JSON-lines wire; nothing it sends is a scoring input.

API (metres / rad / wxyz; `cell_api.md` is the agent-facing contract):
  describe()                                 static facts
  get_state()                                sim_time, done, the two arms' own state, clamp_engaged, pairs_shipped ...
  set_control(arm, mode, stiffness_scale)    joint | cartesian; stiffness_scale in [0.2, 1.0]
  command(arm, q | tcp_pos+tcp_quat, width)  targets held until the next command
  hook(on)                                   the hook tip's clip
  release_clamp()                            the cradle clamp
  step(n_ticks)                              advance n x 10 ms; {sim_time, done, events}
  get_camera(name, depth)                    RGB (png b64) + depth (16-bit png, mm)
  finish()                                   ends the episode
Practice servers additionally accept reset(seed) and truth(); privileged servers get_state_privileged() and
reference_run() (the reference controller inside the cell).

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
from .episode import CREDITS, LACED_SCALED, WorkOrder, draw_work_order, pair_milestones
from .physics import AGLET_SLIP_N, AGLET_SLIP_TICKS, Scene

__all__ = ["ShoeCellServer", "DEFAULT_CONFIG", "PUBLIC_METHODS", "PRIVILEGED_METHODS"]

PUBLIC_METHODS = ("describe", "get_state", "set_control", "command", "hook", "release_clamp", "step", "get_camera", "finish")
PRIVILEGED_METHODS = ("get_state_privileged", "reference_run")
PRACTICE_METHODS = ("reset", "truth")
# an internal error inside one of these calls ends the episode (the cell may be left half-changed)
STATE_METHODS = ("set_control", "command", "hook", "release_clamp", "step", "reference_run", "reset")
CAMERA_NAMES = ("overhead", "front", "side", "wrist_left", "wrist_right")

DEFAULT_CONFIG = {
    "pairs_per_episode": 1,
    "sim_cap_s": 900.0,
    "program_time_cap_s": 1200.0,
    "silence_s": 120.0,
    "max_ticks_per_step": 500,
    "settle_s": 0.5,
    "camera_budget_frames": 450,
    "fault_on_side_p": 0.25,
    "fault_uneven_p": 0.0,
    "clamp_rest_s": 0.3,
    "safety": {"panda_force_N": 30.0, "press_ticks": 3, "object_speed_m_s": 3.0},
    "depth_noise": {"sigma_mm": 1.5, "dropout": 0.03, "min_m": 0.15, "max_m": 3.0},
    "rgb_noise_sigma": 2.0,
    "zones": {
        "infeed": {"centre": [0.40, 0.26], "half": [0.24, 0.24]},
        "blank": {"centre": [0.30, -0.22]},
        "cradle": {"centre": [-0.36, 0.05], "yaw": -1.5707963267948966},
        "outfeed_x": 0.55,
    },
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
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class ShoeCellServer(object):
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
        self.wall_used = 0.0               # the episode's wall clock (set by the wire)
        self.program_used = 0.0            # its program time: the wall clock minus work_wall (set by the wire)
        # the cell's own work, not charged to the program's time budget: the simulation inside step() and the frame
        # production inside get_camera() (rendering, noise, encoding); the sim cap and the frame budget bound both
        self.work_wall = 0.0
        self.internal_errors = 0

    # ------------------------------------------------------------------ episode setup
    def reset(self, seed):
        self.seed = int(seed)
        self.order = draw_work_order(self.seed, self.cfg)
        t0 = time.time()
        self.scene = Scene(self.given_dir, self.order, self.cfg)
        self.build_wall_s = time.time() - t0
        self.tick_s = self.scene.tick_s
        self.done = False
        self.done_reason = None
        self.tick_count = 0
        self.events_log = []
        self.violations = []
        self._press = {}                         # (link body, other body) -> [ticks above the force limit, peak N]
        self.illegal_calls = 0
        self.internal_errors = 0
        self.calls = {}
        self.step_calls = 0
        self.frames_used = 0
        self.frames_by_camera = dict((c, 0) for c in CAMERA_NAMES)
        self.render_wall_s = 0.0
        self.render_failures = 0
        self.hook_loaded = None                  # the lace body held by the hook
        self.left_holding = None                 # (body name, handle) held by the gripper
        self._left_closing = False
        self.clamp_engaged = None                # the shoe id clamped in the cradle
        self._clamp_rest_ticks = 0
        self._clamp_candidate = None
        self._clamp_released_until = 0.0
        self._f6 = np.zeros(6)
        self._slip_ticks = {"left": 0, "right": 0}
        self._noise_rng = np.random.default_rng([self.seed & 0xFFFFFFFF, 0xC0FFEE])
        self.work = WorkOrder(self.scene, self.order, self.cfg, log=self.log)
        self.scene.settle(float(self.cfg["settle_s"]))
        self.work.start(0.0)
        self.scene.settle(0.3)
        self.scene.data.time = 0.0
        for arm in self.scene.arms.values():
            arm.rebase()
        return self.describe()

    # ------------------------------------------------------------------ API
    def describe(self):
        sc = self.scene
        z = self.cfg["zones"]
        robots = {}
        for name, arm in sc.arms.items():
            robots[name] = {"model": "franka_emika_panda (MuJoCo Menagerie, Apache-2.0)", "base_pos": [float(v) for v in sc.arm_base[name]["pos"]],
                            "base_quat": [float(v) for v in sc.arm_base[name]["quat"]],
                            "joints": ["%s/joint%d" % (name, i + 1) for i in range(7)], "joint_range": arm.ranges(),
                            "home_q": [float(v) for v in arm.home_q], "tcp_site": "%s/tcp" % name,
                            "tool": sc.tool_description(name)}
        return {
            "cell_version": CELL_VERSION,
            "units": {"length": "m", "angle": "rad", "quaternion": "wxyz", "time": "s"},
            "robots": robots,
            "control": {"tick_s": self.tick_s, "joint_rate_limit_rad_s": 1.5, "cartesian_speed_limit_m_s": 0.5,
                        "cartesian_angular_limit_rad_s": 1.5, "stiffness_scale_range": [0.2, 1.0], "modes": ["joint", "cartesian"],
                        "gripper": {"max_width_m": 0.08, "rate_m_s": 0.05}},
            "tools": sc.attach_rules(),
            "cameras": sc.cameras.describe(),
            "zones": {"infeed_tray": {"centre": z["infeed"]["centre"], "half_extents": z["infeed"]["half"], "top_z": sc.infeed_top_z},
                      "blank_drop": {"centre": z["blank"]["centre"], "jitter": "+-0.015 m, +-0.15 rad, either way round"},
                      "cradle": {"nominal_centre": z["cradle"]["centre"], "nominal_yaw": z["cradle"]["yaw"], "jitter": "+-0.02 m, +-0.12 rad per episode",
                                 "seat": sc.cradle_description()},
                      "outfeed_line_x": z["outfeed_x"]},
            "work_order": {"pairs": int(self.cfg["pairs_per_episode"]), "credits": dict(CREDITS),
                           "laced_scaled": "%s are credited times the fraction of the pair's shoes laced through every row (0, 1/2 or 1): a pair boxed unlaced earns the erected credit only" % ", ".join(LACED_SCALED),
                           "what_arrives": "two shoes with their laces threaded through row 1 and the lid tray on the infeed tray, the flat box blank at the blank spot; with more than one pair in the work order, the next pair the moment the previous box is shipped"},
            "objects": sc.object_description(),
            "budgets": {"sim_time_cap_s": self.cfg["sim_cap_s"], "program_time_cap_s": self.cfg["program_time_cap_s"],
                        "program_time": "the episode's wall clock minus the cell's own time inside step (simulation) and get_camera (rendering)",
                        "silence_abort_s": self.cfg["silence_s"],
                        "max_ticks_per_step": self.cfg["max_ticks_per_step"], "camera_frames": int(self.cfg["camera_budget_frames"])},
            "safety": dict(self.cfg["safety"]),
        }

    def get_state(self):
        sc = self.scene
        arms = {}
        for name, arm in sc.arms.items():
            p, R = arm.tcp_actual()
            st = {"q": [float(v) for v in arm.q()], "dq": [float(v) for v in arm.dq()], "cmd_q": [float(v) for v in arm.q_cmd],
                  "tcp_pos": [float(v) for v in p], "tcp_quat": [float(v) for v in arm.mat_to_quat(R)],
                  "mode": arm.mode, "stiffness_scale": arm.stiffness_scale}
            if name == "left":
                st["gripper_width"] = arm.width()
                st["holding"] = bool(self.left_holding is not None)
            else:
                st["hook_loaded"] = bool(self.hook_loaded is not None)
            arms[name] = st
        cur = self.work.current
        return {"sim_time": float(sc.data.time), "done": bool(self.done), "arms": arms,
                "clamp_engaged": self.clamp_engaged is not None, "pairs_shipped": int(self.work.shipped_count),
                "infeed_pair": (None if cur is None or cur["shipped"] else int(cur["index"]) + 1)}

    def set_control(self, arm="left", mode="joint", stiffness_scale=1.0):
        a = self._arm(arm)
        a.set_mode(str(mode))
        a.set_stiffness(float(stiffness_scale))
        return {"ok": True, "arm": arm, "mode": a.mode, "stiffness_scale": a.stiffness_scale}

    def command(self, arm="left", q=None, tcp_pos=None, tcp_quat=None, width=None):
        if self.done:
            return {"ok": False, "error": "episode_over"}
        a = self._arm(arm)
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
            if arm != "left":
                return {"ok": False, "error": "only the left arm has a gripper"}
            w = float(width)
            if not math.isfinite(w):
                raise ValueError("non-finite width")
            # closing = the target is below the measured opening (a fully closed target always counts, so a hand
            # whose fingers were pushed past zero can still grasp after the next close command)
            self._left_closing = (w < a.width() - 0.002) or w <= 0.001
            if self.left_holding is not None and w > a.width() + 0.002:
                self._release_left()
            a.command_width(w)
        return {"ok": True}

    def hook(self, on=True):
        if self.done:
            return {"ok": False, "error": "episode_over"}
        if bool(on):
            if self.hook_loaded is None:
                lace = self.scene.hook_catch()
                if lace is not None:
                    self.hook_loaded = lace
                    self._event("hook_caught")
        else:
            if self.hook_loaded is not None:
                self.scene.release_hook()
                self.hook_loaded = None
        return {"ok": True, "loaded": self.hook_loaded is not None}

    def release_clamp(self):
        if self.done:
            return {"ok": False, "error": "episode_over"}
        if self.clamp_engaged is not None:
            self.scene.set_clamp(self.clamp_engaged, False)
            self.clamp_engaged = None
        self._clamp_released_until = float(self.scene.data.time) + 2.0      # no immediate re-clamp while the shoe is lifted
        return {"ok": True}

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
        ev = [e for e in events if e.startswith(("safety:", "pair_arrived:", "box_shipped:", "aglet_slipped:", "physics_fault"))]
        if self.done:
            ev.append("episode_over")
        return {"sim_time": float(self.scene.data.time), "done": bool(self.done), "events": ev}

    def get_camera(self, name="overhead", depth=True):
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
            return self._frame(name, k, depth)
        finally:
            self.work_wall += time.time() - t_frame

    def _frame(self, name, k, depth):
        t = time.time()
        try:
            rgb, dep = self.scene.cameras.render(self.scene.data, name)
        except Exception as e:  # noqa: BLE001 - a renderer failure never takes the cell down
            self.log("[cell] %s frame %d not rendered: %s" % (name, k, type(e).__name__))
            self.render_failures += 1
            self.render_wall_s += time.time() - t
            return {"error": "camera_unavailable"}
        from PIL import Image
        rng = self._noise_rng
        sig = float(self.cfg.get("rgb_noise_sigma", 0.0))
        if sig > 0:
            rgb = np.clip(rgb.astype(np.float32) + rng.normal(0.0, sig, rgb.shape), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG", compress_level=1)
        out = {"ok": True, "camera": name, "frame_index": int(k), "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
               "format": "png", "rgb_png_b64": base64.b64encode(buf.getvalue()).decode("ascii")}
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
        self.render_wall_s += time.time() - t
        return out

    def finish(self):
        self._end("finish")
        return {"ok": True, "sim_time": float(self.scene.data.time)}

    # ------------------------------------------------------------------ privileged / practice
    def get_state_privileged(self):
        return self.truth()

    def truth(self):
        """Every pose, the eyelets, the lace passes, the carton states, the milestones and the cameras' exact calibration:
        the practice cell's development call (and the reference controller's eyes)."""
        sc = self.scene
        out = {"sim_time": float(sc.data.time), "state": self.get_state(), "pairs": []}
        for rec in self.work.pairs:
            ms, credit, detail = (rec["milestones"], rec["credit"], rec["detail"]) if rec["shipped"] else pair_milestones(sc, rec, self.cfg)
            shoes = []
            for s in range(2):
                sid = rec["shoe_ids"][s]
                lid = rec["lace_ids"][s]
                passes, snug = sc.lace_passes(lid, sid)
                shoes.append({"id": sid, "pose": sc.shoe_pose(sid), "upright": sc.upright(sid), "seated": sc.seated(sid),
                              "eyelets": sc.eyelets_world(sid), "handles": sc.shoe_handles(sid), "arrival": rec["spec"]["shoes"][s],
                              "lace": {"id": lid, "polyline": sc.lace_polyline(lid), "aglets": sc.aglet_centres(lid), "passes": passes, "snug": snug}})
            cst = sc.carton_state(rec["carton_id"])
            cst = {k2: v for k2, v in cst.items() if not k2.startswith("_")}
            lp, lq = sc.lid_pose()
            out["pairs"].append({"index": rec["index"], "shipped": rec["shipped"], "colours": {k2: rec["spec"][k2] for k2 in ("upper_colour", "sole_colour", "lace_colour")},
                                 "shoes": shoes, "carton": cst, "lid": {"pos": lp, "quat": lq, "closed": sc.lid_closed()},
                                 "handles": sc.carton_handles(rec["carton_id"]), "milestones": ms, "credit": credit, "detail": detail})
        out["cradle"] = sc.cradle_truth()
        out["clamp_engaged"] = self.clamp_engaged
        out["hook_loaded"] = self.hook_loaded
        out["left_holding"] = None if self.left_holding is None else self.left_holding[0]
        out["cameras"] = sc.cameras.truth(sc.data)
        out["batch"] = {"shoe": sc.batch, "carton": self.order["carton"], "lid": sc.lid["dims"]}
        out["work"] = {"pairs_shipped": self.work.shipped_count, "current_pair": None if self.work.current is None else self.work.current["index"]}
        return _jsonable(out)

    def reference_run(self):
        """Runs the reference controller inside the cell (privileged; the grader passes its folder)."""
        if not self.reference_dir:
            return {"error": "no reference controller"}
        path = os.path.join(self.reference_dir, "controller.py")
        spec = importlib.util.spec_from_file_location("shoecell_reference_controller", path)
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
        except Exception:  # noqa: BLE001 - nothing a request does may crash the wire; the episode is scored as it stands
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
        except (TypeError, ValueError, KeyError, IndexError) as e:     # malformed parameters are the caller's error, never a wire crash
            return {"error": "bad_params: %s" % str(e)[:200]}

    # ------------------------------------------------------------------ physics
    def _arm(self, name):
        if name not in self.scene.arms:
            raise KeyError("arm must be left or right")
        return self.scene.arms[name]

    def _event(self, e):
        self.events_log.append((float(self.scene.data.time), e))

    def _end(self, reason):
        if not self.done:
            self.done = True
            self.done_reason = reason

    def physics_tick(self):
        sc = self.scene
        for arm in sc.arms.values():
            arm.tick()
        sc.step_physics()
        self.tick_count += 1
        events = []
        if sc.physics_fault and not self.done:
            # MuJoCo detected a diverging state and reset the world: nothing after this point is meaningful
            events.append("physics_fault")
            self._end("physics_fault")
        if not sc.physics_fault:
            events += self._aglet_slip()
        events += self._grasp_logic()
        events += self._clamp_logic()
        sc.update_locks(self.tick_s)
        events += self.work.tick(float(sc.data.time))
        events += self._watch()
        for e in events:
            self.events_log.append((float(sc.data.time), e))
        return events

    def _aglet_slip(self):
        """The pinch and the clip are not stronger than AGLET_SLIP_N: a held aglet whose weld carries more than that for
        AGLET_SLIP_TICKS ticks in a row slips out (the gripper keeps its fingers where they are, the clip opens)."""
        events = []
        for arm, eq_name in (("left", "left/grasp"), ("right", "right/hook")):
            f = self.scene.aglet_pull(eq_name)
            if f is None or f <= AGLET_SLIP_N:
                self._slip_ticks[arm] = 0
                continue
            self._slip_ticks[arm] += 1
            if self._slip_ticks[arm] >= AGLET_SLIP_TICKS:
                self._slip_ticks[arm] = 0
                if arm == "left":
                    self._release_left()
                else:
                    self.scene.release_hook()
                    self.hook_loaded = None
                events.append("aglet_slipped:" + arm)
        return events

    def _grasp_logic(self):
        sc = self.scene
        left = sc.arms["left"]
        events = []
        if self.left_holding is None and self._left_closing:
            hit = sc.find_left_handle()
            if hit is not None and left.width() <= hit["pinch_width"] + 0.006:
                if hit.get("on_hook"):
                    sc.release_hook()                     # the gripper takes the aglet off the hook
                    events.append("hook_released:handover")
                sc.attach_left(hit)
                left.hold_width(hit["pinch_width"])
                self.left_holding = (hit["body"], hit["name"])
                self._left_closing = False
                events.append("grasped:" + hit["name"])
        return events

    def _release_left(self):
        if self.left_holding is not None:
            self.scene.release_left()
            self.scene.arms["left"].hold_width(None)
            self._event("released:" + self.left_holding[1])
            self.left_holding = None

    def _clamp_logic(self):
        sc = self.scene
        cur = self.work.current
        events = []
        if cur is None or cur["shipped"] or self.clamp_engaged is not None or sc.data.time < self._clamp_released_until:
            self._clamp_rest_ticks = 0
            return events
        cand = None
        for sid in cur["shoe_ids"]:
            if sc.seated(sid) and sc.at_rest(sid):
                cand = sid
                break
        if cand is not None and cand == self._clamp_candidate:
            self._clamp_rest_ticks += 1
        else:
            self._clamp_candidate = cand
            self._clamp_rest_ticks = 1 if cand is not None else 0
        if cand is not None and self._clamp_rest_ticks * self.tick_s >= float(self.cfg["clamp_rest_s"]):
            if self.left_holding is not None and self.left_holding[0] == cand:
                return events                                  # still in the gripper: clamp once released
            sc.set_clamp(cand, True)
            self.clamp_engaged = cand
            events.append("clamped:%d" % cand)
        return events

    def _watch(self):
        sc, m, d = self.scene, self.scene.model, self.scene.data
        events = []
        thr_f = float(self.cfg["safety"]["panda_force_N"])
        n_ticks = int(self.cfg["safety"].get("press_ticks", 3))
        # the strongest contact force between two bodies this tick: a press counts once it has stayed above the limit
        # for `press_ticks` consecutive control ticks (a single-step spike of MuJoCo's soft contact is not a press; a
        # crash or a held push is)
        now = {}
        for i in range(d.ncon):
            con = d.contact[i]
            c1, c2 = sc.geom_class[con.geom1], sc.geom_class[con.geom2]
            a1, a2 = sc.geom_arm[con.geom1], sc.geom_arm[con.geom2]
            # 1 = an arm link (not a pad, not the hook), 4 = a loose object, 5 = static environment; any contact
            # between the two arms (links, pads or the hook tool) counts as well
            link_vs_world = (c1 == 1 and c2 in (4, 5)) or (c2 == 1 and c1 in (4, 5))
            arm_vs_arm = a1 != 0 and a2 != 0 and a1 != a2
            if link_vs_world or arm_vs_arm:
                self.mujoco.mj_contactForce(m, d, i, self._f6)
                link = con.geom1 if (c1 == 1 or a1 == 1) else con.geom2
                other = con.geom2 if link == con.geom1 else con.geom1
                key = (int(m.geom_bodyid[link]), int(m.geom_bodyid[other]))
                now[key] = max(now.get(key, 0.0), abs(float(self._f6[0])))
        press = {}
        for key, fn in now.items():
            if fn > thr_f:
                cnt, peak = self._press.get(key, (0, 0.0))
                press[key] = (cnt + 1, max(peak, fn))
        self._press = press
        for key, (cnt, peak) in press.items():
            if cnt == n_ticks:
                what = "panda_contact:%s:%s:%.0fN" % (m.body(key[0]).name, m.body(key[1]).name, peak)
                if not self.violations or self.violations[-1]["t"] < d.time - 0.5 or self.violations[-1]["what"].split(":")[1:3] != what.split(":")[1:3]:
                    self.violations.append({"t": float(d.time), "what": what, "force_N": peak})
                    events.append("safety:" + what)
        thr_v = float(self.cfg["safety"]["object_speed_m_s"])
        for name, v in sc.object_speeds():
            if v > thr_v:
                what = "object_speed:%s:%.1fm/s" % (name, v)
                if not any(vi["what"].startswith("object_speed:" + name) for vi in self.violations[-3:]):
                    self.violations.append({"t": float(d.time), "what": what, "speed": v})
                    events.append("safety:" + what)
        return events

    # ------------------------------------------------------------------ end of episode
    def finalize(self):
        if not self.done:
            self._end("solve_returned")
        ws = self.work.summary(self.violations)
        summary = {
            "cell_version": CELL_VERSION, "seed": self.seed, "done_reason": self.done_reason,
            "sim_time_s": float(self.scene.data.time), "wall_time_s": float(self.wall_used), "ticks": int(self.tick_count),
            "work_score": float(ws["work_score"]), "pairs": ws["pairs"], "pairs_shipped": int(ws["pairs_shipped"]),
            "violations": list(self.violations), "safety_ok": len(self.violations) == 0,
            "illegal_call_count": int(self.illegal_calls), "internal_errors": int(self.internal_errors), "calls": dict(self.calls),
            "step_calls": int(self.step_calls), "program_time_s": float(self.program_used), "cell_work_s": float(self.work_wall),
            "camera_frames_total": int(self.frames_used), "camera_frames": dict(self.frames_by_camera),
            "render_wall_s": float(self.render_wall_s), "render_failures": int(self.render_failures), "build_wall_s": float(self.build_wall_s),
            "events": [[t, e] for t, e in self.events_log],
        }
        return _jsonable(summary)
