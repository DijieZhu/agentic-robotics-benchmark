"""CellServer of the six-servo arm: the shared `arm.assembly_cell.server.CellServer` with this arm's builder, layout,
pre-assembled groups and functional check. The wire API, the predicates, the latch, the safety watch and the grading
are the shared cell's (nothing is re-implemented here beyond `reset`, `describe`, `get_state`, `finalize`).
"""

import json
import math

import numpy as np

from ..assembly_cell.controller import PandaController, mat_to_quat
from ..assembly_cell.predicates import MateTracker
from ..assembly_cell.server import DEFAULT_SERVER_CONFIG as _SHARED_DEFAULTS
from ..assembly_cell.server import CellServer as _SharedCellServer
from ..assembly_cell.server import _deep_update
from ..chain_model import Chain
from . import CELL_VERSION
from .cell_builder import DEFAULT_PREASSEMBLY, build_cell, group_poses, load_givens, rest_poses, scope_parts
from .functional import run_functional
from .layout import hidden_layout

__all__ = ["CellServer", "DEFAULT_SERVER_CONFIG"]

DEFAULT_SERVER_CONFIG = json.loads(json.dumps(_SHARED_DEFAULTS))
DEFAULT_SERVER_CONFIG.update({
    "sim_cap_s": 600.0,
    "wall_cap_s": 480.0,
    "preassembly": json.loads(json.dumps(DEFAULT_PREASSEMBLY)),
    "functional": dict(_SHARED_DEFAULTS["functional"], envelope=None),
})


class CellServer(_SharedCellServer):
    def __init__(self, a2_dir, panda_dir, assembly_json=None, config=None, scope_ids=None, allow_reset=False, log=None):
        import mujoco
        self.mujoco = mujoco
        self.a2_dir = a2_dir
        self.panda_dir = panda_dir
        self.cfg = _deep_update(json.loads(json.dumps(DEFAULT_SERVER_CONFIG)), config)
        self.manifest, self.graph = load_givens(a2_dir)
        self.parts = {p["name"]: p for p in self.manifest["parts"]}
        self.scope_ids = list(scope_ids or self.graph["scopes"]["elbow"])
        self.allow_reset = bool(allow_reset)
        self.log = log or (lambda s: None)
        self.assembly_json = assembly_json
        if assembly_json is None:
            raise ValueError("the six-servo cell needs the design assembly.json (pre-assembled groups are placed by its FK)")
        with open(assembly_json) as f:
            self.design = json.load(f)
        self.chain = Chain(self.design)
        self.joint_axis_in_child = {j.name: [float(v) for v in j.axis] for j in self.chain.joints}
        self.preassembly = self.cfg["preassembly"]
        self.renderer = None
        self.model = None
        self.data = None
        self.seed = None
        self.done = False
        self.collision_cache = {}

    # ------------------------------------------------------------------ episode setup
    def reset(self, seed):
        root, loose, mates, member_of = scope_parts(self.graph, self.scope_ids, self.manifest, self.preassembly)
        gp = group_poses(self.design, self.preassembly)
        lay_cfg = dict(self.cfg["layout"])
        rests = rest_poses(self.design, self.manifest, self.a2_dir, self.preassembly, loose, gp, self.graph,
                           float(lay_cfg.get("base_yaw_on_fixture_rad", -1.5707963)))
        servo_loose = [n for n in loose if self.parts[n]["kind"] == "servo"]
        layout = hidden_layout(seed, self.cfg["layout"], loose, rests, servo_names=servo_loose)
        build = build_cell(self.a2_dir, self.panda_dir, layout, self.cfg["cell"], self.scope_ids, self.manifest, self.graph, self.design,
                           self.preassembly, collision_cache=self.collision_cache)
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
        # the shared tracker knows the root, the loose bodies and the horns; the pre-assembled members are bodies too
        for name in self.meta["preassembled"]:
            self.tracker.body[name] = m.body(name).id
        on = self.cfg.get("observation_noise") or {}
        self._obs_noise = (float(on.get("pos_mm", 0.0)) * 1e-3, math.radians(float(on.get("ang_deg", 0.0))))
        self._obs_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x0B5E])
        # geom classification: 0 other, 1 panda link, 2 pad L, 3 pad R, 4 loose part (compound members count as their root), 5 static env
        self.geom_class = np.zeros(m.ngeom, dtype=int)
        self.geom_part = {}
        panda_bodies = set(b for b in range(m.nbody) if m.body(b).name.startswith("panda/"))
        loose_bodies = {m.body(n).id: n for n in self.meta["loose"]}
        for s in self.meta["servos"]:
            loose_bodies[m.body("horn_" + s).id] = s
        for gname, g in self.meta["group_poses"].items():
            if not g["fixed"]:
                for mem in g["parts"]:
                    loose_bodies[m.body(mem).id] = g["root"]
        self.loose_body_ids = loose_bodies
        self.static_env_bodies = set([0, m.body("fixture").id, m.body(self.meta["root"]).id])
        for gname, g in self.meta["group_poses"].items():
            if g["fixed"]:
                for mem in g["parts"]:
                    self.static_env_bodies.add(m.body(mem).id)
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
        self._dmg_peak = {"pad": 0.0, "part": 0.0}       # damage watch (review 2026-09-27): peak pad-on-part / part-on-environment normal force
        self._dmg_over = {"pad": 0, "part": 0}
        return self.describe()

    # ------------------------------------------------------------------ damage watch
    def _watch(self, d):
        """The shared watch (30 N non-pad link rule, part speed, held / dropped / lost) plus the damage watch: the fingertip
        pads pressing on a part and a part pressing on the environment or another part are exempt from the 30 N rule, so
        their peak normal forces (summed per pad-part / part-body pair per tick) are recorded (summary["damage"], stats) and, when `damage.pad_force_N` /
        `damage.part_force_N` are set, exceeding them for `damage.ticks` consecutive ticks is a safety violation
        (`pad_force:<part>:<N>N`, `part_force:<part>:<other>:<N>N`). Rung 1 sets no cap (stats only)."""
        events = super()._watch(d)
        m = self.model
        dmg = self.cfg.get("damage") or {}
        pad_sum = {}       # (pad class, part) -> summed normal force of that pad on that part
        part_sum = {}      # (part, other body) -> summed normal force between them (a box in a socket has many contacts)
        for i in range(d.ncon):
            con = d.contact[i]
            c1, c2 = self.geom_class[con.geom1], self.geom_class[con.geom2]
            if (c1 in (2, 3) and c2 == 4) or (c2 in (2, 3) and c1 == 4):
                self.mujoco.mj_contactForce(m, d, i, self._f6)
                key = (c1 if c1 in (2, 3) else c2, self.geom_part[con.geom2 if c2 == 4 else con.geom1])
                pad_sum[key] = pad_sum.get(key, 0.0) + abs(float(self._f6[0]))
            elif (c1 == 4 and c2 in (4, 5)) or (c2 == 4 and c1 in (4, 5)):
                self.mujoco.mj_contactForce(m, d, i, self._f6)
                g_part = con.geom1 if c1 == 4 else con.geom2
                g_other = con.geom2 if c1 == 4 else con.geom1
                key = (self.geom_part[g_part], m.body(m.geom_bodyid[g_other]).name)
                part_sum[key] = part_sum.get(key, 0.0) + abs(float(self._f6[0]))
        pad_max = max(pad_sum.values()) if pad_sum else 0.0
        part_max = max(part_sum.values()) if part_sum else 0.0
        pad_arg = max(pad_sum, key=pad_sum.get)[1] if pad_sum else None
        part_arg = max(part_sum, key=part_sum.get) if part_sum else None
        self._dmg_peak["pad"] = max(self._dmg_peak["pad"], pad_max)
        self._dmg_peak["part"] = max(self._dmg_peak["part"], part_max)
        ticks = int(dmg.get("ticks", 5))
        for key, fn, cap, what in (("pad", pad_max, dmg.get("pad_force_N"), lambda: "pad_force:%s:%.0fN" % (pad_arg, pad_max)),
                                   ("part", part_max, dmg.get("part_force_N"), lambda: "part_force:%s:%s:%.0fN" % (part_arg[0], part_arg[1], part_max) if part_arg else "part_force:?")):
            if cap is None or fn <= float(cap):
                self._dmg_over[key] = 0
                continue
            self._dmg_over[key] += 1
            if self._dmg_over[key] >= ticks:
                w = what()
                if not any(v["what"].split(":")[:2] == w.split(":")[:2] and v["t"] > d.time - 0.5 for v in self.violations[-3:]):
                    self.violations.append({"t": float(d.time), "what": w, "force_N": fn})
                    events.append("safety:" + w)
        return events

    def damage_summary(self):
        dmg = self.cfg.get("damage") or {}
        rep = dmg.get("report_N") or {}
        pad_cap, part_cap = dmg.get("pad_force_N"), dmg.get("part_force_N")
        rep_pad = float(rep.get("pad_force_N", pad_cap if pad_cap is not None else 120.0))
        rep_part = float(rep.get("part_force_N", part_cap if part_cap is not None else 80.0))
        return {"max_pad_force_N": float(self._dmg_peak["pad"]), "max_part_force_N": float(self._dmg_peak["part"]),
                "caps_N": {"pad_force_N": pad_cap, "part_force_N": part_cap},
                "report_N": {"pad_force_N": rep_pad, "part_force_N": rep_part},
                "no_damage": bool(self._dmg_peak["pad"] <= rep_pad and self._dmg_peak["part"] <= rep_part)}

    # ------------------------------------------------------------------ API
    def _part_names_in_model(self):
        names = [self.meta["root"]]
        for gname, g in self.meta["group_poses"].items():
            if g["fixed"]:
                names += [n for n in g["parts"] if n not in names]
        for n in self.meta["loose"]:
            names.append(n)
            for gname, g in self.meta["group_poses"].items():
                if not g["fixed"] and g["root"] == n:
                    names += [x for x in g["parts"] if x not in names]
        return names

    def describe(self):
        m = self.model
        cfg = self.meta["config"]
        pre = self.meta["preassembled"]
        gp = self.meta["group_poses"]
        parts = []
        for n in self._part_names_in_model():
            p = self.parts[n]
            grp = pre.get(n)
            e = {"id": n, "kind": p["kind"], "fixed": bool(n == self.meta["root"] or (grp and gp[grp]["fixed"])),
                 "group": grp or ("proximal" if n == self.meta["root"] and any(g["fixed"] and g["root"] == n for g in gp.values()) else None),
                 "loose_body": n in self.meta["loose"], "mesh_visual": p["mesh_visual"], "mesh_collision": list(p.get("mesh_collision", [])),
                 "mass_kg": p["mass_kg"], "bbox_mm": p["bbox_mm"], "com_mm": p["com_mm"]}
            if p["kind"] == "servo":
                e["catalogue_id"] = p["catalogue_id"]
                e["carrier_link"] = p["carrier_link"]
                e["designed_pose_in_carrier_mm"] = p["designed_pose_in_carrier"]
                e["collision_primitives"] = p["collision_primitives"]
                e["horn_axis_point_mm"] = p["horn_axis_point_mm"]
                e["horn_axis_dir"] = p.get("horn_axis_dir", [0.0, 0.0, 1.0])
                e["interchangeable_with"] = [s for s in self.meta["servos"] if s != n and self.parts[s]["catalogue_id"] == p["catalogue_id"]]
            if p.get("role") == "bracket":
                e["attached_to"] = p["attached_to"]
                e["designed_pose_in_link_mm"] = p["designed_pose_in_link"]
            parts.append(e)
        groups = []
        for gname, g in gp.items():
            groups.append({"name": gname, "root": g["root"], "fixed": g["fixed"], "parts": list(g["parts"]),
                           "joint_angles_rad": g["q_full"],
                           "note": "pre-assembled by the cell at the design pose for these joint angles; moves as one rigid body"})
        mates = [dict(g) for g in self.graph["mates"] if g["id"] in set(self.scope_ids)]
        return {
            "cell_version": CELL_VERSION, "units": {"length": "m", "angle": "rad", "quaternion": "wxyz", "files": "mm"},
            "scope": list(self.scope_ids), "parts": parts, "preassembled_groups": groups, "mates": mates,
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
            "cell": {"table_top_z": 0.0,
                     "fixture": {"pos": self.meta["layout"]["fixture"]["pos"], "top_z": self.meta["layout"]["fixture_top_z_m"], "size_m": cfg["fixture"]["size_m"],
                                 "base_frame": "base part frame = fixture top frame yawed by base_yaw_rad", "base_yaw_rad": self.meta["layout"].get("base_yaw_rad", 0.0)},
                     "trays": self.meta["layout"]["trays"], "friction": self.meta["layout"]["friction"],
                     "servo_collision_shrink_mm": cfg["shrink_mm"], "carve_margin_mm": cfg["carve_margin_mm"],
                     "welds": "grader-owned, snapped to the designed pose when a mate latches"},
            "cameras": ["overview", "top", "oblique", "wrist"],
            "budgets": {"sim_time_cap_s": self.cfg["sim_cap_s"], "wall_clock_cap_s": self.cfg["wall_cap_s"], "silence_abort_s": self.cfg["silence_s"],
                        "max_ticks_per_step": self.cfg["max_ticks_per_step"]},
            "safety": self.cfg["safety"],
            "damage": self.cfg.get("damage"),
        }

    def get_state(self):
        d = self.data
        p, R = self.ctrl.tcp_actual()
        adr = self.model.sensor_adr
        wrench = list(d.sensordata[adr[self.sensor_force]:adr[self.sensor_force] + 3]) + list(d.sensordata[adr[self.sensor_torque]:adr[self.sensor_torque] + 3])
        parts = {}
        sp, sa = self._obs_noise
        for n in self._part_names_in_model():
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

    # ------------------------------------------------------------------ end of episode
    def finalize(self, functional_rng, run_functional_check=True):
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
            "layout": {"fixture": self.meta["layout"]["fixture"], "base_yaw_rad": self.meta["layout"].get("base_yaw_rad"),
                       "servo_tray": self.meta["layout"]["servo_tray"], "friction": self.meta["layout"]["friction"]},
            "preassembly": {g: {"root": v["root"], "fixed": v["fixed"], "joint_angles_rad": v["q_full"], "parts": list(v["parts"])} for g, v in self.meta["group_poses"].items()},
            "damage": self.damage_summary(),
        }
        if run_functional_check:
            summary["functional"] = run_functional(self, functional_rng, self.cfg["functional"], log=self.log)
        else:
            summary["functional"] = {"score": 0.0, "passed": 0, "n_configs": int(self.cfg["functional"]["n_configs"]), "completed_joints": [], "in_scope_joints": []}
        summary["joint_zero_offsets_rad"] = {r.get("joint") or r["id"].replace("horn_", ""): r["joint_zero_offset_rad"]
                                             for r in rec if r["kind"] == "horn_mate" and r["achieved"]}
        return summary
