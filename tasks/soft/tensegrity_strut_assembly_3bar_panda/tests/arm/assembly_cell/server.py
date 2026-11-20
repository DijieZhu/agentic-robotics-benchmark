"""CellServer: the grader-owned physics and the assembly-cell API (CHAIN_PLAN 2.5, 4.3; 02 section 3).

The server owns MjModel / MjData, steps physics only inside `step`, evaluates every predicate from
MjData, toggles the welds, watches safety and grasp state, and writes the assembly record. The agent
talks to it over the JSON-lines wire (`wire.py`); nothing the agent sends is a scoring input.

API (metres / rad / wxyz):
  describe()                        static facts (parts, mates in scope, robot, cell, budgets, control)
  get_state()                       sim_time, robot {q, dq, cmd_q, gripper_width, tcp_pos, tcp_quat, wrench},
                                    parts {id: {pos, quat}}, mates {id: achieved}, held, done
  set_control(mode, stiffness_scale) mode joint | cartesian; stiffness_scale in [0.2, 1.0]
  command(q | tcp_pos+tcp_quat, width)   targets held until the next command
  step(n_ticks)                     advance n x 10 ms; returns {sim_time, done, events}
  get_camera(name)                  RGB (png b64) + depth (float16 b64) + K + T_world_cam
  finish()                          ends the episode
Practice servers additionally accept reset(seed); the hidden server answers {"error": "forbidden"}.
"""

import base64
import io
import json
import math
import os
import time

import numpy as np

from . import CELL_VERSION
from .cell_builder import DEFAULT_CELL_CONFIG, build_cell, load_givens, scope_parts
from .controller import PandaController, mat_to_quat
from .functional import DEFAULT_FUNCTIONAL_CONFIG, run_functional
from .layout import DEFAULT_LAYOUT_CONFIG, hidden_layout
from .predicates import MateTracker, body_pose

__all__ = ["CellServer", "DEFAULT_SERVER_CONFIG", "PUBLIC_METHODS"]

PUBLIC_METHODS = ("describe", "get_state", "set_control", "command", "step", "get_camera", "finish")

DEFAULT_SERVER_CONFIG = {
    "sim_cap_s": 600.0,
    "wall_cap_s": 480.0,
    "silence_s": 60.0,
    "max_ticks_per_step": 500,
    "settle_s": 0.6,
    "latch": {"hold_ticks": 50, "rel_speed_max_m_s": 0.005, "tolerance_scale": 1.0},
    "observation_noise": {"pos_mm": 0.0, "ang_deg": 0.0},      # rung >= 2: noise on the reported part poses (C axis)
    "safety": {"panda_force_N": 30.0, "part_speed_m_s": 3.0},
    "held": {"force_N": 1.0, "ticks": 5},
    "drop": {"speed_m_s": 0.25, "watch_ticks": 50},
    "lost_z_m": -0.05,
    "camera_size": [480, 640],
    "functional": dict(DEFAULT_FUNCTIONAL_CONFIG),
    "cell": {},
    "layout": {},
}


def _deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


class EpisodeOver(Exception):
    pass


class CellServer(object):
    def __init__(self, a2_dir, panda_dir, assembly_json=None, config=None, scope_ids=None, allow_reset=False, log=None):
        import mujoco
        self.mujoco = mujoco
        self.a2_dir = a2_dir
        self.panda_dir = panda_dir
        self.cfg = _deep_update(json.loads(json.dumps(DEFAULT_SERVER_CONFIG)), config)
        self.manifest, self.graph = load_givens(a2_dir)
        self.parts = {p["name"]: p for p in self.manifest["parts"]}
        self.scope_ids = list(scope_ids or self.graph["scopes"]["rung1_prefix"])
        self.allow_reset = bool(allow_reset)
        self.log = log or (lambda s: None)
        self.assembly_json = assembly_json
        self.chain = None
        self.joint_axis_in_child = {}
        if assembly_json:
            from ..chain_model import Chain
            with open(assembly_json) as f:
                self.design = json.load(f)
            # the requirements' tool frame (handoff.TOOL_POINT_SERVO_MM / TOOL_APPROACH_SERVO) for the envelope test
            self.chain = Chain(self.design, tool={"servo_id": "S6", "point": [0.0, 32.0, 46.0], "approach": [0.0, 1.0, 0.0]})
            for j in self.chain.joints:
                self.joint_axis_in_child[j.name] = [float(v) for v in j.axis]
        self.renderer = None
        self.model = None
        self.data = None
        self.seed = None
        self.done = False
        self.collision_cache = {}     # carved collision pieces per part: design-dependent, seed-independent

    # ------------------------------------------------------------------ episode setup
    def reset(self, seed):
        root, loose, mates = scope_parts(self.graph, self.scope_ids, self.manifest)
        layout = hidden_layout(seed, self.cfg["layout"], loose, self.parts)
        build = build_cell(self.a2_dir, self.panda_dir, layout, self.cfg["cell"], self.scope_ids, self.manifest, self.graph,
                           collision_cache=self.collision_cache)
        self.build = build
        self.model, self.meta = build.model, build.meta
        self.meta["joint_axis_in_child"] = self.joint_axis_in_child
        self.data = self.mujoco.MjData(self.model)
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.tick_s = self.model.opt.timestep * self.meta["config"]["tick_steps"]
        self.tick_steps = int(self.meta["config"]["tick_steps"])
        self.seed = int(seed)
        self.done = False
        self.done_reason = None
        self.tick_count = 0
        self.events_log = []
        self.violations = []
        self.illegal_calls = 0
        self.calls = {}
        self.step_calls = 0
        self.regrasps = 0
        self.held = None
        self._held_cand = None
        self._held_ticks = 0
        self._drop_watch = {}
        self._lost = set()
        self._dropped = set()
        self.wall_start = None
        self.wall_used = 0.0
        self.renderer = None
        m = self.model
        self.ctrl = PandaController(m, self.data, tick_s=self.tick_s)
        # settle the loose parts (welds inactive, servo actuators off, panda at home)
        for _ in range(int(self.cfg["settle_s"] / m.opt.timestep)):
            self.mujoco.mj_step(m, self.data)
        self.data.time = 0.0
        self.ctrl.q_cmd = self.ctrl.q().copy()
        self.ctrl.q_goal = self.ctrl.q_cmd.copy()
        self.ctrl.width_cmd = self.ctrl.width()
        self.ctrl.width_goal = self.ctrl.width_cmd
        self.ctrl.write_ctrl()
        lt = self.cfg["latch"]
        self.tracker = MateTracker(m, self.meta, hold_ticks=lt["hold_ticks"], rel_speed_max=lt["rel_speed_max_m_s"], tick_s=self.tick_s,
                                   tolerance_scale=lt.get("tolerance_scale", 1.0))
        on = self.cfg.get("observation_noise") or {}
        self._obs_noise = (float(on.get("pos_mm", 0.0)) * 1e-3, math.radians(float(on.get("ang_deg", 0.0))))
        self._obs_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x0B5E])   # per-episode, seed-derived
        # geom classification for safety / held
        self.geom_class = np.zeros(m.ngeom, dtype=int)   # 0 other, 1 panda link, 2 pad L, 3 pad R, 4 loose part, 5 static env
        self.geom_part = {}
        panda_bodies = set(b for b in range(m.nbody) if m.body(b).name.startswith("panda/"))
        loose_bodies = {m.body(n).id: n for n in self.meta["loose"]}
        for s in self.meta["servos"]:
            loose_bodies[m.body("horn_" + s).id] = s
        self.loose_body_ids = loose_bodies
        self.static_env_bodies = set([0, m.body("fixture").id, m.body(self.meta["root"]).id])
        pads = set(self.meta["pad_geoms"])
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            name = m.geom(g).name
            if name in pads:
                self.geom_class[g] = 2 if "left" in name else 3
            elif b in panda_bodies:
                self.geom_class[g] = 1
            elif b in loose_bodies:
                self.geom_class[g] = 4
                self.geom_part[g] = loose_bodies[b]
            elif b in self.static_env_bodies:
                self.geom_class[g] = 5
        self.part_body = {n: m.body(n).id for n in self.meta["loose"]}
        self.part_qadr = {n: m.jnt_qposadr[m.joint("free_" + n).id] for n in self.meta["loose"]}
        self.part_vadr = {n: m.jnt_dofadr[m.joint("free_" + n).id] for n in self.meta["loose"]}
        self.sensor_force = m.sensor("hand_force").id
        self.sensor_torque = m.sensor("hand_torque").id
        self._f6 = np.zeros(6)
        return self.describe()

    # ------------------------------------------------------------------ API
    def describe(self):
        m = self.model
        parts = []
        for n in [self.meta["root"]] + list(self.meta["loose"]):
            p = self.parts[n]
            e = {"id": n, "kind": p["kind"], "fixed": n == self.meta["root"], "mesh_visual": p["mesh_visual"],
                 "mesh_collision": list(p.get("mesh_collision", [])), "mass_kg": p["mass_kg"], "bbox_mm": p["bbox_mm"],
                 "com_mm": p["com_mm"]}
            if p["kind"] == "servo":
                e["catalogue_id"] = p["catalogue_id"]
                e["carrier_link"] = p["carrier_link"]
                e["designed_pose_in_carrier_mm"] = p["designed_pose_in_carrier"]
                e["collision_primitives"] = p["collision_primitives"]
                e["horn_axis_point_mm"] = [0.0, 7.0, 46.0]
                e["interchangeable_with"] = [s for s in self.meta["servos"] if s != n and self.parts[s]["catalogue_id"] == p["catalogue_id"]]
            parts.append(e)
        mates = []
        for g in self.graph["mates"]:
            if g["id"] in set(self.scope_ids):
                mates.append({k: v for k, v in g.items()})
        cfg = self.meta["config"]
        return {
            "cell_version": CELL_VERSION, "units": {"length": "m", "angle": "rad", "quaternion": "wxyz", "files": "mm"},
            "scope": list(self.scope_ids), "parts": parts, "mates": mates,
            "latch": self.graph.get("latch"),
            "robot": {"model": "franka_emika_panda (MuJoCo model collection, Apache-2.0)", "base_pos": cfg["panda_pos"],
                      "joints": ["panda/joint%d" % (i + 1) for i in range(7)],
                      "joint_range": [[float(m.jnt_range[j][0]), float(m.jnt_range[j][1])] for j in self.ctrl.joints],
                      "home_q": list(self.meta["panda_home"]), "tcp_site": "tcp", "tcp_offset_in_hand_m": [0.0, 0.0, cfg["tcp_offset_m"]],
                      "gripper": {"max_width_m": self.ctrl.max_width, "rate_m_s": 0.05, "pad_size_m": [0.017, 0.0165],
                                  "force_model": "2500 N/m x (object width - target width) / 2 per finger, capped at 60 N"},
                      "gravity_compensated": bool(cfg["panda_gravcomp"] > 0)},
            "control": {"tick_s": self.tick_s, "joint_rate_limit_rad_s": 1.5, "cartesian_speed_limit_m_s": 0.5,
                        "cartesian_angular_limit_rad_s": 1.5, "stiffness_scale_range": [0.2, 1.0], "modes": ["joint", "cartesian"]},
            "cell": {"table_top_z": 0.0, "fixture": {"pos": self.meta["layout"]["fixture"]["pos"], "top_z": self.meta["layout"]["fixture_top_z_m"],
                                                     "size_m": cfg["fixture"]["size_m"], "base_frame": "base part frame = fixture top frame, world axes"},
                     "trays": self.meta["layout"]["trays"], "friction": self.meta["layout"]["friction"],
                     "servo_collision_shrink_mm": cfg["shrink_mm"], "welds": "grader-owned, snapped to the designed pose when a mate latches"},
            "cameras": ["overview", "top", "oblique", "wrist"],
            "budgets": {"sim_time_cap_s": self.cfg["sim_cap_s"], "wall_clock_cap_s": self.cfg["wall_cap_s"], "silence_abort_s": self.cfg["silence_s"],
                        "max_ticks_per_step": self.cfg["max_ticks_per_step"]},
            "safety": self.cfg["safety"],
        }

    def get_state(self):
        d = self.data
        p, R = self.ctrl.tcp_actual()
        adr = self.model.sensor_adr
        wrench = list(d.sensordata[adr[self.sensor_force]:adr[self.sensor_force] + 3]) + list(d.sensordata[adr[self.sensor_torque]:adr[self.sensor_torque] + 3])
        parts = {}
        sp, sa = self._obs_noise
        for n in [self.meta["root"]] + list(self.meta["loose"]):
            b = self.model.body(n).id
            pos, quat = d.xpos[b], d.xquat[b]
            if sp > 0.0 or sa > 0.0:
                pos = pos + self._obs_rng.normal(0.0, sp, 3) if sp > 0.0 else pos
                if sa > 0.0:
                    ax = self._obs_rng.normal(0.0, 1.0, 3)
                    ax /= max(1e-12, np.linalg.norm(ax))
                    ang = self._obs_rng.normal(0.0, sa)
                    dq = np.array([math.cos(ang / 2)] + list(math.sin(ang / 2) * ax))
                    q = np.zeros(4)
                    self.mujoco.mju_mulQuat(q, dq, quat)
                    quat = q
            parts[n] = {"pos": [float(v) for v in pos], "quat": [float(v) for v in quat]}
        return {
            "sim_time": float(d.time), "done": bool(self.done),
            "robot": {"q": [float(v) for v in self.ctrl.q()], "dq": [float(v) for v in self.ctrl.dq()], "cmd_q": [float(v) for v in self.ctrl.q_cmd],
                      "gripper_width": self.ctrl.width(), "tcp_pos": [float(v) for v in p], "tcp_quat": [float(v) for v in mat_to_quat(R)],
                      "wrench": [float(v) for v in wrench], "mode": self.ctrl.mode, "stiffness_scale": self.ctrl.stiffness_scale},
            "parts": parts, "mates": self.tracker.status(), "held": self.held,
        }

    def set_control(self, mode="joint", stiffness_scale=1.0):
        self.ctrl.set_mode(str(mode))
        self.ctrl.set_stiffness(float(stiffness_scale))
        return {"ok": True, "mode": self.ctrl.mode, "stiffness_scale": self.ctrl.stiffness_scale}

    def command(self, q=None, tcp_pos=None, tcp_quat=None, width=None):
        if self.done:
            return {"ok": False, "error": "episode_over"}
        if q is not None:
            if self.ctrl.mode != "joint":
                return {"ok": False, "error": "q target needs joint mode"}
            self.ctrl.command_joint(q)
        if tcp_pos is not None or tcp_quat is not None:
            if self.ctrl.mode != "cartesian":
                return {"ok": False, "error": "tcp target needs cartesian mode"}
            if tcp_pos is None or tcp_quat is None:
                return {"ok": False, "error": "tcp_pos and tcp_quat go together"}
            self.ctrl.command_cartesian(tcp_pos, tcp_quat)
        if width is not None:
            self.ctrl.command_width(width)
        return {"ok": True}

    def step(self, n_ticks=1):
        n = int(n_ticks)
        if n < 1:
            return {"ok": False, "error": "n_ticks must be >= 1"}
        n = min(n, int(self.cfg["max_ticks_per_step"]))
        events = []
        self.step_calls += 1
        for _ in range(n):
            if self.done:
                break
            events += self.physics_tick(track=True)
            if self.data.time >= self.cfg["sim_cap_s"] - 1e-9:
                self._end("sim_cap")
        return {"sim_time": float(self.data.time), "done": bool(self.done), "events": events}

    def get_camera(self, name="overview"):
        name = str(name)
        if name not in ("overview", "top", "oblique", "wrist"):
            return {"ok": False, "error": "unknown camera"}
        try:
            if self.renderer is None:
                h, w = self.cfg["camera_size"]
                self.renderer = self.mujoco.Renderer(self.model, h, w)
            r = self.renderer
            r.update_scene(self.data, camera=name)
            rgb = r.render()
            r.enable_depth_rendering()
            r.update_scene(self.data, camera=name)
            depth = r.render().astype(np.float16)
            r.disable_depth_rendering()
        except Exception as e:  # noqa: BLE001 - a headless host without GL must not crash the episode
            return {"ok": False, "error": "camera_unavailable: %s" % type(e).__name__}
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        cam = self.model.camera(name).id
        h, w = rgb.shape[:2]
        fovy = float(self.model.cam_fovy[cam])
        fy = 0.5 * h / math.tan(math.radians(fovy) / 2)
        K = [[fy, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]]
        T = np.eye(4)
        T[:3, :3] = self.data.cam_xmat[cam].reshape(3, 3)
        T[:3, 3] = self.data.cam_xpos[cam]
        return {"ok": True, "width": int(w), "height": int(h), "rgb_png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "depth_f16_b64": base64.b64encode(depth.tobytes()).decode("ascii"), "K": K, "T_world_cam": T.tolist()}

    def finish(self):
        self._end("finish")
        return {"ok": True, "sim_time": float(self.data.time)}

    def _end(self, reason):
        if not self.done:
            self.done = True
            self.done_reason = reason

    # ------------------------------------------------------------------ dispatch
    def handle(self, method, params):
        self.calls[method] = self.calls.get(method, 0) + 1
        if method == "reset":
            if not self.allow_reset:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            seed = int((params or {}).get("seed", 0))
            self.reset(seed)
            return {"ok": True, "seed": seed}
        if method not in PUBLIC_METHODS:
            self.illegal_calls += 1
            return {"error": "forbidden"}
        try:
            return getattr(self, method)(**(params or {}))
        except (TypeError, ValueError, KeyError, IndexError) as e:     # malformed parameters are the caller's error, never a wire crash
            return {"error": "bad_params: %s" % str(e)[:200]}
        except ValueError as e:
            return {"error": "bad_value: %s" % e}

    # ------------------------------------------------------------------ physics
    def physics_tick(self, track=True):
        m, d = self.model, self.data
        self.ctrl.tick()
        for _ in range(self.tick_steps):
            self.mujoco.mj_step(m, d)
        self.tick_count += 1
        events = []
        if track:
            events += self.tracker.tick(d, d.time)
            events += self._watch(d)
            for e in events:
                self.events_log.append((float(d.time), e))
        return events

    def max_panda_contact_force(self):
        m, d = self.model, self.data
        mx = 0.0
        for i in range(d.ncon):
            con = d.contact[i]
            c1, c2 = self.geom_class[con.geom1], self.geom_class[con.geom2]
            if 1 in (c1, c2):
                self.mujoco.mj_contactForce(m, d, i, self._f6)
                mx = max(mx, abs(float(self._f6[0])))
        return mx

    def _welded(self, part):
        m, d = self.model, self.data
        for (b1, b2), eq in self.meta["welds"].items():
            if part in (b1, b2) and d.eq_active[m.equality(eq).id]:
                return True
        return False

    def _watch(self, d):
        m = self.model
        events = []
        thr_f = self.cfg["safety"]["panda_force_N"]
        left = {}
        right = {}
        for i in range(d.ncon):
            con = d.contact[i]
            c1, c2 = self.geom_class[con.geom1], self.geom_class[con.geom2]
            if c1 == 0 and c2 == 0:
                continue
            self.mujoco.mj_contactForce(m, d, i, self._f6)
            fn = abs(float(self._f6[0]))
            # safety: a Panda link (not a pad) pressing on the environment or a part
            if (c1 == 1 and c2 in (4, 5)) or (c2 == 1 and c1 in (4, 5)):
                if fn > thr_f:
                    other = con.geom2 if c1 == 1 else con.geom1
                    link = con.geom1 if c1 == 1 else con.geom2
                    what = "panda_contact:%s:%s:%.0fN" % (m.body(m.geom_bodyid[link]).name, m.body(m.geom_bodyid[other]).name, fn)
                    if not self.violations or self.violations[-1]["t"] < d.time - 0.5 or self.violations[-1]["what"].split(":")[1:3] != what.split(":")[1:3]:
                        self.violations.append({"t": float(d.time), "what": what, "force_N": fn})
                        events.append("safety:" + what)
            # held: both pads on the same part
            for cls, g in ((c1, con.geom2), (c2, con.geom1)):
                if cls in (2, 3) and self.geom_class[g] == 4:
                    part = self.geom_part[g]
                    (left if cls == 2 else right)[part] = (left if cls == 2 else right).get(part, 0.0) + fn
        thr_h = self.cfg["held"]["force_N"]
        cand = None
        best = 0.0
        for part in left:
            if part in right and left[part] >= thr_h and right[part] >= thr_h:
                s = min(left[part], right[part])
                if s > best:
                    cand, best = part, s
        if cand is not None and cand == self._held_cand:
            self._held_ticks += 1
        else:
            self._held_cand = cand
            self._held_ticks = 1 if cand is not None else 0
        new_held = self._held_cand if self._held_ticks >= self.cfg["held"]["ticks"] else None
        if new_held != self.held:
            if new_held is not None:
                self.regrasps += 1
                events.append("grasped:" + new_held)
            else:
                events.append("released:" + self.held)
                self._drop_watch[self.held] = {"t0": self.tick_count, "vmax": 0.0}
            self.held = new_held
        # part speeds, lost, dropped
        thr_v = self.cfg["safety"]["part_speed_m_s"]
        for n in self.meta["loose"]:
            v = float(np.linalg.norm(d.qvel[self.part_vadr[n]:self.part_vadr[n] + 3]))
            z = float(d.qpos[self.part_qadr[n] + 2])
            if v > thr_v:
                what = "part_speed:%s:%.1fm/s" % (n, v)
                if not any(vi["what"].startswith("part_speed:" + n) for vi in self.violations[-3:]):
                    self.violations.append({"t": float(d.time), "what": what, "speed": v})
                    events.append("safety:" + what)
            if z < self.cfg["lost_z_m"] and n not in self._lost:
                self._lost.add(n)
                events.append("lost:" + n)
            w = self._drop_watch.get(n)
            if w is not None:
                w["vmax"] = max(w["vmax"], v)
                if self.tick_count - w["t0"] >= self.cfg["drop"]["watch_ticks"]:
                    del self._drop_watch[n]
                    if w["vmax"] > self.cfg["drop"]["speed_m_s"] and not self._welded(n) and n not in self._dropped:
                        self._dropped.add(n)
                        events.append("dropped:" + n)
        return events

    # ------------------------------------------------------------------ end of episode
    def holds_without_weld(self, seconds=2.0):
        """Shadow rollout: deactivate every weld on a copy of the state, settle, re-evaluate (stat)."""
        m = self.model
        d2 = self.mujoco.MjData(m)
        d2.qpos[:] = self.data.qpos
        d2.qvel[:] = 0.0
        d2.ctrl[:] = self.data.ctrl
        d2.time = self.data.time
        saved = self.data.eq_active.copy()
        d2.eq_active[:] = 0
        for _ in range(int(seconds / m.opt.timestep)):
            self.mujoco.mj_step(m, d2)
        res = self.tracker.evaluate(d2)
        self.data.eq_active[:] = saved
        n = 0
        ok = 0
        for mid, st in self.tracker.state.items():
            if not st["achieved"]:
                continue
            n += 1
            r = res.get(mid)
            if r is not None and r.get("holds"):
                ok += 1
        return (ok / float(n)) if n else 0.0

    def finalize(self, functional_rng, run_functional_check=True):
        """Episode summary + functional check + as-built data. Call once after the episode ended."""
        if not self.done:
            self._end("solve_returned")
        rec = self.tracker.record()
        weights = {m["id"]: m["weight"] for m in self.meta["mates"]}
        W = sum(weights.values())
        achieved_w = sum(weights[r["id"]] for r in rec if r["achieved"])
        all_done = all(r["achieved"] for r in rec)
        held_at_end = self.held
        hww = self.holds_without_weld()
        summary = {
            "cell_version": CELL_VERSION, "seed": self.seed, "scope": list(self.scope_ids), "done_reason": self.done_reason,
            "sim_time_s": float(self.data.time), "wall_time_s": float(self.wall_used), "ticks": int(self.tick_count),
            "mates": rec, "weights": weights, "sticky_fraction": achieved_w / W if W else 0.0, "all_mates": bool(all_done),
            "held_at_end": held_at_end, "success": bool(all_done and held_at_end is None),
            "violations": list(self.violations), "safety_ok": len(self.violations) == 0,
            "illegal_call_count": int(self.illegal_calls), "calls": dict(self.calls), "step_calls": int(self.step_calls),
            "regrasps": int(self.regrasps), "lost_parts": sorted(self._lost), "dropped_parts": sorted(self._dropped),
            "holds_without_weld": float(hww), "events": [[t, e] for t, e in self.events_log],
            "layout": {"fixture": self.meta["layout"]["fixture"], "servo_tray": self.meta["layout"]["servo_tray"], "friction": self.meta["layout"]["friction"]},
        }
        if run_functional_check and self.chain is not None:
            summary["functional"] = run_functional(self, functional_rng, self.cfg["functional"], log=self.log)
        else:
            summary["functional"] = {"score": 0.0, "passed": 0, "n_configs": int(self.cfg["functional"]["n_configs"]), "completed_joints": [], "in_scope_joints": []}
        summary["joint_zero_offsets_rad"] = {r.get("joint") or r["id"].replace("horn_", ""): r["joint_zero_offset_rad"]
                                             for r in rec if r["kind"] == "horn_mate" and r["achieved"]}
        return summary

    def as_built_assembly(self, summary):
        """The design assembly.json with per-joint as_built_origin / joint_zero_offset_rad (mm, CHAIN_PLAN 2.4)."""
        if self.design is None:
            return None
        asm = json.loads(json.dumps(self.design))
        offsets = summary.get("joint_zero_offsets_rad", {})
        seats = {r["id"]: r for r in summary["mates"] if r["kind"] in ("sleeve_seat", "part_seat")}
        for j in asm["joints"]:
            j["joint_zero_offset_rad"] = float(offsets.get(j["name"], 0.0))
            j["as_built_origin"] = json.loads(json.dumps(j["origin"]))
            j["as_built"] = j["name"] in offsets
        for l in asm["links"]:
            for occ in l.get("servo_occurrences", []):
                r = seats.get("seat_" + occ["servo_id"])
                if r and r["achieved"]:
                    occ["as_built_servo"] = r["servo"]
        asm["as_built_record"] = {"seed": summary["seed"], "scope": summary["scope"], "cell_version": CELL_VERSION,
                                  "note": "welds snap to the designed relative pose, so the as-built frames equal the design; the yaw quadrant of each horn mate is the only as-built degree of freedom (joint_zero_offset_rad)"}
        return asm
