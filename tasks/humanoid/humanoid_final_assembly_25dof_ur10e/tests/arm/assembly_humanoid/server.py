"""The humanoid assembly cell behind the camera-only API: the server.

New file of this task. It takes the place of the shared servers (arm/assembly_car/server.py `CarInfoServer`,
arm/assembly_rov/server.py `RovInfoServer`, both vendored and unchanged) for this cell: the same seven calls and the
same reply formats, the same dispatch rule (anything else answers {"error": "forbidden"} and is counted), the same
routed control tick (every tick, whoever drives it, goes through the server and the safety monitor), the shared wire
(arm/assembly_cell/wire.py `serve_episode`, through `wire_guard.GuardedAgentProcess`) and client
(arm/assembly_cell/client.py `connect_stdio`: os.dup2(2, 1) before anything writes; this task's `client.py` keeps the
first wire). On top: the coupling's state and the fastening texts in describe(), the layout, the privileged calls with
the oracle token, scoring in finalize(), the episode's budgets.

  describe()                        the cell's public description: robot, control, gripper, cameras (names, sizes,
                                    depth), budgets, safety numbers (all from the configuration)
  get_state()                       sim_time, done, robot {q, dq, cmd_q, gripper_width, force_cmd, tcp_pos, tcp_quat,
                                    wrench (white noise + a per-episode bias), mode, stiffness_scale}
  set_control(mode, stiffness_scale)  mode "joint" | "cartesian"; stiffness_scale in [0.2, 1.0]
  command(q | tcp_pos + tcp_quat, width, force)
                                    q: 6 joint targets (joint mode) inside the joint ranges; tcp_pos (3, m) + tcp_quat
                                    (4, wxyz, not zero) (Cartesian mode); width in [0, max opening] m; force in the pad
                                    force range (N per pad)
  step(n_ticks)                     1 <= n; more than budgets.max_ticks_per_step (500) is cut to it and counted
                                    (steps_clamped); events only "safety:<what>" and "episode_over"
  get_camera(name, depth)           one frame of "top" or "wrist" through the ONE render interface (render_iface.py);
                                    budgets.camera_frames per episode, then {"error": "camera_budget"}; a frame the
                                    renderer could not make for this request answers {"error": "camera_unavailable"};
                                    a failure of the renderer itself (render_iface.RenderInfrastructureError, and any
                                    other exception while rendering) is never answered: it ends the verify as a task
                                    error
  finish()                          ends the episode

Every argument is checked at the boundary before anything is applied (a submission is hostile input): types (a
bool is not a number, a string is not a list), counts (exactly 6 / 3 / 4 numbers), finite values (no NaN or infinity),
ranges (refused, not clipped), integer sizes (no huge integers), unknown parameter names; a malformed call answers
{"error": "bad_params: ..."} and changes nothing. Budgets, all from the configuration's "budgets" block: simulated
seconds (sim_cap_s: the episode ends at the cap), camera frames (camera_frames), silence (silence_s, enforced by the
wire between two requests; the server also keeps the largest gap it saw), wall clock (wall_cap_s, enforced by the
wire; null until the budget rule computes it from the measured pace).
"""
import copy
import json
import math
import time

import numpy as np

from . import config as CF
from . import render_iface as RI

__all__ = ["HumanoidCellServer", "PUBLIC_METHODS", "PRIVILEGED_METHODS", "CELL_VERSION", "BadParams"]

CELL_VERSION = "humanoid-assembly-cell/0.2.0"
PUBLIC_METHODS = ("describe", "get_state", "set_control", "command", "step", "get_camera", "finish")
# the shared servers' privileged calls (arm/assembly_car/server.py, arm/assembly_rov/server.py): only with the
# oracle token (the server's `privileged`, set by the episode builder from episode.token_state); otherwise each answers
# {"error": "forbidden"} and is counted as an illegal call. There is no in-cell call that assembles the reference (the
# reference program runs over the wire like any program and reads the truth through these two calls); a call of such a
# name is an ordinary forbidden call
PRIVILEGED_METHODS = ("get_state_privileged", "describe_privileged")
MAX_METHOD_KEYS = 64                # distinct forbidden method names kept by name in the counters
BIG_INT = 2 ** 53


class BadParams(ValueError):
    pass


def _number(v, what, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BadParams("%s must be a number" % what)
    if isinstance(v, int) and abs(v) > BIG_INT:
        raise BadParams("%s is out of range" % what)
    x = float(v)
    if not math.isfinite(x):
        raise BadParams("%s must be finite" % what)
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise BadParams("%s must be within [%s, %s]" % (what, lo, hi))
    return x


def _vector(v, n, what, lo=None, hi=None):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise BadParams("%s must be a list of %d numbers" % (what, n))
    return [_number(x, "%s[%d]" % (what, i), None if lo is None else lo[i], None if hi is None else hi[i])
            for i, x in enumerate(v)]


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class HumanoidCellServer(object):
    """`cell` = a built scene.Cell of this episode (with its robot and monitor); `cfg` = the configuration (default the
    cell's); `seed` = the episode's seed (the wrench noise and bias); `renderer` = a render_iface backend name (default
    cfg["cameras"]["renderer"])."""

    def __init__(self, cell, cfg=None, seed=0, privileged=False, renderer=None, log=None, episode=None):
        if cell.ctrl is None or cell.monitor is None:
            raise ValueError("the server needs a cell with its robot and its safety monitor")
        self.cell = cell
        self.cfg = copy.deepcopy(cfg if cfg is not None else cell.cfg)
        CF.check_config(self.cfg)
        self.model, self.data, self.ctrl, self.monitor = cell.m, cell.d, cell.ctrl, cell.monitor
        self.privileged = bool(privileged)
        self.log = log or (lambda s: None)
        self.seed = int(seed)
        B = self.cfg["budgets"]
        self.sim_cap_s = float(B["sim_cap_s"])
        self.camera_frames = int(B["camera_frames"])
        self.silence_s = float(B["silence_s"])
        self.max_ticks = int(B["max_ticks_per_step"])
        self.wall_cap_s = None if B.get("wall_cap_s") is None else float(B["wall_cap_s"])
        self.camera_names = tuple(self.cfg["cameras"]["agent_cameras"])
        self.renderer_name = str(renderer or self.cfg["cameras"]["renderer"])
        self.render = RI.open_backend(self.renderer_name, cell, self.cfg)
        rb = cell.robot
        m = self.model
        self.jlo = [float(m.jnt_range[m.joint(j).id][0]) for j in rb["joints"]]
        self.jhi = [float(m.jnt_range[m.joint(j).id][1]) for j in rb["joints"]]
        self.force_range = [float(v) for v in self.cfg["gripper"]["pad_force_N"]]
        from ..assembly_rov import ur_controller as URC
        self.stiffness_range = [float(v) for v in URC.STIFFNESS_RANGE]
        self.sensor_force = m.sensor(rb["force_sensor"]).id
        self.sensor_torque = m.sensor(rb["torque_sensor"]).id
        wn = self.cfg["observation"]["wrench_noise"]
        r = np.random.default_rng([self.seed & 0xFFFFFFFF, (self.seed >> 32) & 0xFFFFFFFF, 0x3E17C4])
        self.wrench_bias = np.concatenate([r.uniform(-wn["bias_f_N"], wn["bias_f_N"], 3), r.uniform(-wn["bias_t_Nm"], wn["bias_t_Nm"], 3)])
        self.wrench_sigma = np.array([wn["sigma_f_N"]] * 3 + [wn["sigma_t_Nm"]] * 3, float)
        self._wrench_rng = np.random.default_rng([self.seed & 0xFFFFFFFF, (self.seed >> 32) & 0xFFFFFFFF, 0x0B5E])
        # counters
        self.done, self.done_reason = False, None
        self.calls, self.illegal_calls, self.bad_params, self.step_calls, self.steps_clamped = {}, 0, 0, 0, 0
        self.tick_count, self.frames_used, self.render_failures = 0, 0, 0
        self.frames_by_camera = {n: 0 for n in self.camera_names}
        self.events_log = []
        self.wall_start, self.wall_used = None, 0.0
        self._last_req, self.max_gap_s = None, 0.0
        self.render_wall_s = 0.0
        # the episode (episode.py: its draws, the coupling, the inputs of record) and the public description
        self.episode = episode
        self.coupling = getattr(cell, "coupling", None)
        self.token_mismatch = False
        self.reference_report = None
        self.reference_calls = 0
        self._public = None
        # every tick through the server (and so through the monitor and the sim cap)
        self._raw_tick = cell.tick_one
        cell.tick_one = self._routed_tick

    # ------------------------------------------------------------------ ticks
    def _routed_tick(self):
        if self.done:
            return []
        events = self._raw_tick()
        self.tick_count += 1
        for e in events:
            self.events_log.append((float(self.data.time), e))
        if float(self.data.time) >= self.sim_cap_s - 1e-9:
            self._end("sim_cap")
        return events

    def _end(self, reason):
        if not self.done:
            self.done, self.done_reason = True, reason

    # ------------------------------------------------------------------ the API
    def describe(self):
        m, rb, ct, S = self.model, self.cell.robot, self.ctrl, self.cfg["safety"]
        from ..assembly_rov import ur_controller as URC
        cams = {n: {"resolution": [int(x) for x in self.cfg["cameras"]["size"][n]], "rgb": True,
                    "depth": bool(self.cfg["cameras"]["depth"][n])} for n in self.camera_names}
        base_out = {
            "cell_version": CELL_VERSION, "units": {"length": "m", "angle": "rad", "quaternion": "wxyz", "force": "N"},
            "robot": {"model": "6-DoF UR10e-class arm with a wide two-finger parallel gripper",
                      "base_pos": [float(x) for x in self.cfg["station"]["ur_base_pos_m"]],
                      "joints": [j[len(rb["prefix"]):] for j in rb["joints"]],
                      "joint_range": [[lo, hi] for lo, hi in zip(self.jlo, self.jhi)],
                      "home_q": [float(x) for x in self.cfg["robot"]["home_q"]],
                      "torque_limits_Nm": [float(x) for x in rb["torque_limits_Nm"]],
                      "tcp_offset_from_flange_m": [0.0, 0.0, round(float(rb["tcp_from_flange_m"]), 4)],
                      "tcp_axes": "z = the approach (out of the flange), y = the closing axis, x across the pads",
                      "gripper": {"max_width_m": float(ct.max_width), "rate_m_s": float(URC.GRIP_RATE_M_S),
                                  "force_N": self.force_range, "default_force_N": float(self.cfg["gripper"]["force_N_default"]),
                                  "width": "the gap between the pad faces at their middle"}},
            "control": {"tick_s": float(self.cell.tick_s), "joint_rate_limit_rad_s": float(np.min(ct.rate_limit)),
                        "cartesian_speed_limit_m_s": float(URC.CART_V_MAX), "cartesian_angular_limit_rad_s": float(URC.CART_W_MAX),
                        "stiffness_scale_range": self.stiffness_range, "modes": ["joint", "cartesian"]},
            "cameras": cams,
            "budgets": {"sim_time_cap_s": self.sim_cap_s, "wall_clock_cap_s": self.wall_cap_s, "silence_abort_s": self.silence_s,
                        "max_ticks_per_step": self.max_ticks, "camera_frames": self.camera_frames},
            "safety": {"robot_force_N": float(S["robot_force_N"]), "part_force_N": float(S["part_force_N"]),
                       "part_force_window_s": round(int(S["part_ticks"]) * float(self.cell.tick_s), 6),
                       "part_speed_m_s": float(S["part_speed_m_s"]), "floor_contact": "the module is scrapped and the contact is a violation",
                       "violation_score_factor": float(self.cfg["scoring"]["safety_violation_factor"])},
        }
        if self.episode is not None:
            # the draw-independent public description (describe_h.py; its FIELD_TABLE lists every key)
            pub = self.public()
            out = base_out
            out["cameras"] = pub["cameras"]
            out["control"]["carry_limits"] = pub["carry_limits"]
            out["fastening"] = pub["fastening"]
            out["items"] = pub["items"]
            out["datums"] = pub["datums"]
            out["scoring"] = pub["scoring"]
            out["observation"] = pub["observation"]
            return out
        return base_out

    def public(self):
        """the episode's public description (describe_h.public_description): computed once, from the configuration and
        the fixture design of record only (never from the draw)"""
        if self._public is None:
            from . import describe_h as DH
            from ..assembly_rov import gripper_wide as GW
            self._public = DH.public_description(self.cfg, self.episode.inputs["des"], GW.DEFAULT_CFG["wrist_camera"])
        return copy.deepcopy(self._public)

    def get_state(self):
        d, ct = self.data, self.ctrl
        p, R = ct.tcp_actual()
        from ..assembly_rov.ur_controller import mat_to_quat
        adr = self.model.sensor_adr
        w = np.concatenate([d.sensordata[adr[self.sensor_force]:adr[self.sensor_force] + 3],
                            d.sensordata[adr[self.sensor_torque]:adr[self.sensor_torque] + 3]])
        w = w + self.wrench_bias + self._wrench_rng.normal(0.0, 1.0, 6) * self.wrench_sigma
        return {"sim_time": float(d.time), "done": bool(self.done),
                "robot": {"q": [float(v) for v in ct.q()], "dq": [float(v) for v in ct.dq()], "cmd_q": [float(v) for v in ct.q_cmd],
                          "gripper_width": float(ct.width()), "force_cmd": float(ct.force_cmd),
                          "tcp_pos": [float(v) for v in p], "tcp_quat": [float(v) for v in mat_to_quat(R)],
                          "wrench": [float(v) for v in w], "mode": ct.mode, "stiffness_scale": float(ct.stiffness_scale)}}

    def set_control(self, mode="joint", stiffness_scale=1.0):
        if not isinstance(mode, str) or mode not in ("joint", "cartesian"):
            raise BadParams("mode must be 'joint' or 'cartesian'")
        s = _number(stiffness_scale, "stiffness_scale", self.stiffness_range[0], self.stiffness_range[1])
        self.ctrl.set_mode(mode)
        self.ctrl.set_stiffness(s)
        return {"ok": True, "mode": self.ctrl.mode, "stiffness_scale": float(self.ctrl.stiffness_scale)}

    def command(self, q=None, tcp_pos=None, tcp_quat=None, width=None, force=None):
        qv = _vector(q, 6, "q", self.jlo, self.jhi) if q is not None else None
        pv = _vector(tcp_pos, 3, "tcp_pos", [-10.0] * 3, [10.0] * 3) if tcp_pos is not None else None
        rv = _vector(tcp_quat, 4, "tcp_quat", [-10.0] * 4, [10.0] * 4) if tcp_quat is not None else None
        wv = _number(width, "width", 0.0, float(self.ctrl.max_width)) if width is not None else None
        fv = _number(force, "force", self.force_range[0], self.force_range[1]) if force is not None else None
        if rv is not None and math.sqrt(sum(x * x for x in rv)) < 1e-6:
            raise BadParams("tcp_quat must not be zero")
        if (pv is None) != (rv is None):
            raise BadParams("tcp_pos and tcp_quat go together")
        if self.done:
            return {"ok": False, "error": "episode_over"}
        if qv is not None and self.ctrl.mode != "joint":
            return {"ok": False, "error": "q target needs joint mode"}
        if pv is not None and self.ctrl.mode != "cartesian":
            return {"ok": False, "error": "tcp target needs cartesian mode"}
        if qv is not None:
            self.ctrl.command_joint(qv)
        if pv is not None:
            self.ctrl.command_cartesian(pv, rv)
        if wv is not None:
            self.ctrl.command_width(wv)
        if fv is not None:
            self.ctrl.command_force(fv)
        return {"ok": True}

    def step(self, n_ticks=1):
        n = _number(n_ticks, "n_ticks", 1.0, None)
        if n != math.floor(n):
            raise BadParams("n_ticks must be a whole number")
        n = int(n)
        if n > self.max_ticks:
            self.steps_clamped += 1
            n = self.max_ticks
        self.step_calls += 1
        events = []
        for _ in range(n):
            if self.done:
                break
            events += self.cell.tick_one()
        ev = [e for e in events if e.startswith("safety:")]
        if self.done:
            ev.append("episode_over")
        return {"sim_time": float(self.data.time), "done": bool(self.done), "events": ev}

    def get_camera(self, name="top", depth=True):
        if not isinstance(name, str) or name not in self.camera_names:
            raise BadParams("unknown camera (cameras: %s)" % ", ".join(self.camera_names))
        if not isinstance(depth, bool):
            raise BadParams("depth must be true or false")
        if self.frames_used >= self.camera_frames:
            return {"error": "camera_budget"}
        k = self.frames_used
        self.frames_used += 1
        self.frames_by_camera[name] += 1
        want_depth = bool(depth) and bool(self.cfg["cameras"]["depth"][name])
        t = time.time()
        try:
            fr = self.render.frame(name, want_depth, k)
            out = RI.encode_reply(name, k, fr)
        except RI.RenderInfrastructureError as e:
            self.render_failures += 1
            self.render_wall_s += time.time() - t
            self.log("[cell] %s frame %d: the renderer failed: %s" % (name, k, e))
            raise
        except RI.RenderUnavailable as e:
            self.render_failures += 1
            self.render_wall_s += time.time() - t
            self.log("[cell] %s frame %d not rendered: %s" % (name, k, e))
            return {"error": "camera_unavailable"}
        except Exception as e:  # noqa: BLE001 - any other failure while rendering is the renderer's, never a wire error
            self.render_failures += 1
            self.render_wall_s += time.time() - t
            self.log("[cell] %s frame %d: the renderer failed: %s: %s" % (name, k, type(e).__name__, str(e)[:200]))
            raise RI.RenderInfrastructureError("%s frame %d: %s: %s" % (name, k, type(e).__name__, str(e)[:200])) from e
        self.render_wall_s += time.time() - t
        return out

    def finish(self):
        self._end("finish")
        return {"ok": True, "sim_time": float(self.data.time)}

    # ------------------------------------------------------------------ dispatch (never raises for a caller's error)
    def _count(self, method):
        if method in PUBLIC_METHODS or method in PRIVILEGED_METHODS:
            key = method
        else:
            key = "forbidden:" + method[:40]
            if key not in self.calls and sum(1 for k in self.calls if k.startswith("forbidden:")) >= MAX_METHOD_KEYS:
                key = "forbidden:<other>"
        self.calls[key] = self.calls.get(key, 0) + 1

    # ------------------------------------------------------------------ privileged (oracle token only)
    def describe_privileged(self):
        """the episode's draws (layout, clamp repeat, camera mounts and lenses) and digest, the hidden interfaces, the
        capture table, the coupling's states and the score"""
        ep, cp = self.episode, self.coupling
        out = {"episode": ep.privileged_record() if ep is not None else None,
               "interfaces": [{k: x[k] for k in ("id", "class", "module", "part_body", "carrier_body", "design_k", "seat_in_carrier",
                                                 "frame_in_part")} for x in (ep.inputs["interfaces"] if ep is not None else [])],
               "capture_table": cp.capture_table() if cp is not None else None,
               "states": dict(cp.state) if cp is not None else None, "score": float(cp.score()) if cp is not None else None}
        return _jsonable(out)

    def get_state_privileged(self):
        """every module's root pose and every kit joint, the coupling's records, the scrapped modules, the score; and
        every interface's carrier and part body pose ("interface_bodies")"""
        m, d = self.model, self.data
        mods = {}
        for mid, bl in self.cell.index.module_bodies.items():
            root = int(m.body(self.cell.bodies_of[mid][0]).id)
            mods[mid] = {"root_body": m.body(root).name, "pos": [float(x) for x in d.xpos[root]],
                         "quat": [float(x) for x in d.xquat[root]]}
        joints = {jn: float(self.cell.joint_q(jn)) for jn in self.cell.joint_params}
        # the true pose of every mate's carrier and part body (the reference places a module on its seat, which
        # describe_privileged gives in the carrier body's frame; the base's bodies have no other pose in this call)
        ibodies = {}
        for x in (self.episode.inputs["interfaces"] if self.episode is not None else []):
            ibodies[x["id"]] = {}
            for role, bn in (("carrier", x["carrier_body"]), ("part", x["part_body"])):
                b = int(m.body(bn).id)
                ibodies[x["id"]][role] = {"body": bn, "pos": [float(v) for v in d.xpos[b]], "quat": [float(v) for v in d.xquat[b]]}
        cp = self.coupling
        return _jsonable({"sim_time": float(d.time), "modules": mods, "joints_rad": joints, "interface_bodies": ibodies,
                          "states": dict(cp.state) if cp is not None else None,
                          "records": cp.record() if cp is not None else None,
                          "scrapped": dict(self.monitor.scrapped), "score": float(cp.score()) if cp is not None else None})

    def handle(self, method, params):
        now = time.monotonic()
        if self._last_req is not None:
            self.max_gap_s = max(self.max_gap_s, now - self._last_req)
        self._last_req = now
        if not isinstance(method, str):
            method = "<not a string>"
        self._count(method)
        if method in PRIVILEGED_METHODS:
            if not self.privileged:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            if params not in (None, {}):
                self.bad_params += 1
                return {"error": "bad_params: %s takes no parameters" % method}
            return getattr(self, method)()
        if method not in PUBLIC_METHODS:
            self.illegal_calls += 1
            return {"error": "forbidden"}
        if params is None:
            params = {}
        if not isinstance(params, dict) or not all(isinstance(k, str) for k in params):
            self.bad_params += 1
            return {"error": "bad_params: params must be an object"}
        try:
            out = getattr(self, method)(**params)
        except (BadParams, TypeError, ValueError, KeyError, IndexError, OverflowError, RecursionError, MemoryError) as e:
            self.bad_params += 1
            return {"error": "bad_params: %s" % str(e)[:200]}
        return out

    # ------------------------------------------------------------------ the wire's budgets and the episode's record
    def wire_budgets(self, wall_cap_s=None):
        """(wall_cap_s, silence_s) for wire.serve_episode: the configuration's, a wall cap given here overriding it"""
        wc = self.wall_cap_s if wall_cap_s is None else float(wall_cap_s)
        if wc is None:
            raise ValueError("budgets.wall_cap_s is not set (the budget rule computes it from the measured pace)")
        return wc, self.silence_s

    def finalize(self):
        if not self.done:
            self._end("solve_returned")
        try:
            self.render.close()
        except Exception:  # noqa: BLE001
            pass
        ms = self.monitor.summary()
        # the episode's mates (coupling_h.record) and the score's numbers of this episode: stage = the counted
        # weights / the total weight; value = stage x (1 without a counted violation, else scoring.safety_violation_factor);
        # the grader takes the mean over its episodes and the contract gate
        cp = self.coupling
        rec = cp.record() if cp is not None else []
        W = float(cp.total_w) if cp is not None else 0.0
        counted_w = float(sum(r["weight"] for r in rec if r["counted"]))
        stage = counted_w / W if W > 0 else 0.0
        f = float(self.cfg["scoring"]["safety_violation_factor"])
        value = stage * (1.0 if ms["safety_ok"] else f)
        ep = self.episode
        mates = {"mates": rec, "states": dict(cp.state) if cp is not None else {}, "weight_total": W, "weight_counted": counted_w,
                 "stage_score": stage, "value": value, "safety_violation_factor": f,
                 "n_locked_design": int(sum(1 for r in rec if r["counted"])),
                 "locked_off": [r["id"] for r in rec if r.get("locked") and r.get("design_ok") is False],
                 "scrapped": dict(cp.scrapped) if cp is not None else dict(ms["scrapped"]),
                 "coupling_metrics": cp.metrics() if cp is not None else None,
                 "coupling_events": [list(e) for e in cp.events][-400:] if cp is not None else [],
                 "layout_digest": ep.report["layout_digest"] if ep is not None else None,
                 "nominal_layout": bool(ep.draw["nominal"]) if ep is not None else None,
                 "token_mismatch": bool(self.token_mismatch), "reference_calls": int(self.reference_calls),
                 "reference": None if self.reference_report is None else {k: self.reference_report.get(k) for k in ("ok", "steps", "unplanned_contacts") if k in self.reference_report}}
        return _jsonable(dict(mates, **{
            "cell_version": CELL_VERSION, "seed": self.seed, "done_reason": self.done_reason,
            "sim_time_s": float(self.data.time), "wall_time_s": float(self.wall_used), "ticks": int(self.tick_count),
            "calls": dict(self.calls), "illegal_call_count": int(self.illegal_calls), "bad_params_count": int(self.bad_params),
            "step_calls": int(self.step_calls), "steps_clamped": int(self.steps_clamped),
            "camera_frames": dict(self.frames_by_camera), "camera_frames_total": int(self.frames_used),
            "render_failures": int(self.render_failures), "renderer": self.renderer_name, "render_wall_s": float(self.render_wall_s),
            "render_retries": int(getattr(self.render, "retries", 0)), "render_controls": int(getattr(self.render, "controls", 0)),
            "max_request_gap_s": float(self.max_gap_s), "privileged_mode": self.privileged,
            "safety": ms, "safety_ok": ms["safety_ok"], "violations_counted": ms["counted"],
            "events": [[t, e] for t, e in self.events_log][-400:],
        }))


def describe_json(server):
    return json.dumps(server.describe(), sort_keys=True)
