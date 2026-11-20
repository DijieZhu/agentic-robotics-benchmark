"""The strut assembly cell behind the camera-only API (`StrutInfoServer`).

The physics is `cell.StrutCell`: the strut core (electronics bay base with its board, the aluminium rod) clamped in the
cradle, the loose modules (motor modules, the bay lid, the end caps) resting in the tray's nests, the coupling
(`coupling.py`: a module on the rod is held on its axis by a rail, the lid on its seat by a soft weld; a module at rest
in its window is welded). The Panda and its controller are the precedent's (`assembly_cell.controller`). The observation
side is the information tier's (`arm.assembly_info.cell_info.InfoCellServer`, inherited): uncalibrated RGB-D cameras
(top, wrist) through the render layer, a noisy flange F/T sensor, the joint state; no part poses, no mate events.

`StrutInfoServer.__init__` calls no parent constructor (the information tier's loads the six-servo arm's kit, as in
the quadruped cell); the inherited methods read the attributes set here and in `reset` / `_init_state`. The safety
watch (`_watch`, every control tick, whoever drives it: an agent's `step()` or the reference) is this cell's own, the
four rules:
  R1  the Panda's collision geoms other than the pads pressing on the cradle, the tray, the bench, a module or the rod:
      the normal forces of every non-pad geom of the arm summed per body it presses on, above `safety.panda_force_N`
      in any tick;
  R2  a module pressed on another body (the environment or another module): the normal forces summed per (module,
      other body) pair above `damage.part_force_N` for `damage.ticks` consecutive ticks; not counted: the pads' grip
      (reported only), the contacts inside a fastened cluster (bodies joined by a chain of LOCKED / LOCKED_OFF mates:
      a fastening's preload), the strut core on the cradle (both static: no contact);
  R3  per rod overhang (either side of the bay): the force on that part of the rod (the contacts on the rod's
      collision geom beyond the bay + the constraint force of every rail / lock weld of a module on it, gravity
      included) across the rod, above `rod.lateral_force_N` for `rod.ticks` consecutive ticks;
  R4  a module's centre of mass faster than `safety.part_speed_m_s`.
Every tick's readings (`self.readings`) and their peaks (`damage_summary()`) are kept; heights and speeds are taken at
each module's centre of mass (its body origin lies in the strut frame, up to 0.16 m off the part).

Per episode (`layout.episode_layout`): the cradle +-10 mm, each tray +-20 mm / +-10 deg, the units' nests permuted,
the motor modules' rolls 6-16 deg off a straight transfer, the end caps' rolls, each lid turned end for end half the
time. The built cell must pass the reach check (`reach_check`, after the quadruped cell: every loose unit keeps a grasp
that passes the reference's pick checks where it rests, every other unit in its nest) or the WHOLE layout is drawn
again (a sub-seed of the episode seed, at most `reach_check.max_draws` draws), the nominal layout last (with this
episode's cradle). With the oracle token (privileged mode) and `oracle_nominal_layout` the layout is the nominal one
and `reference_assemble()` runs the reference program (reference.py) inside the cell, through the same routed ticks
and safety watch.
"""

import json
import math
import time

import numpy as np

from ..assembly_cell.server import CellServer as _BaseCellServer
from ..assembly_cell.server import _deep_update
from ..assembly_info import layout_info as LI
from ..assembly_info.cell_info import DEFAULT_INFO_CONFIG, InfoCellServer, cell_datums
from . import fixture as FX
from . import layout as LY
from . import trays as TR
from .cell import CELL_CONFIG, StrutCell
from .coupling import body_pose as cpose
from .kit import StrutKit

CELL_VERSION = "strut-assembly-cell/1.0.0"
PUBLIC_METHODS = ("describe", "get_state", "set_control", "command", "step", "get_camera", "finish")
PRIVILEGED_METHODS = ("get_state_privileged", "describe_privileged", "reference_assemble")

_MOUNT_TEXT = {
    "top": "overhead, roughly 1.25 m above the bench near its middle, looking straight down; the image covers the whole working area "
           "(the parts trays, the cradle with the struts, the robot base, the bench board); image width roughly along the cell's x",
    "wrist": "on the Panda hand's +x flank, a few centimetres off the flange axis, tilted towards the fingertips",
}

# the documented datums: the bench board and the hand plate are the precedent's (cell_info); the cradle board lies flat
# on the cradle plate's +x edge (fixture.DEFAULT_FIXTURE["board"])
DEFAULT_STRUT_CONFIG = {
    "sim_cap_s": 1200.0,
    "wall_cap_s": 2400.0,
    "silence_s": 120.0,
    "max_ticks_per_step": 500,
    "camera_budget_frames": 600,
    "settle_s": 0.6,
    "scope": "single",
    "units": None,
    "safety": {"panda_force_N": 15.0, "part_speed_m_s": 3.0},
    "damage": {"pad_force_N": None, "part_force_N": 30.0, "ticks": 20, "exclude_locked_pairs": True},
    "rod": {"lateral_force_N": 4.0, "ticks": 20, "per_overhang": True, "include_coupling_forces": True},
    "held": {"force_N": 1.0, "ticks": 5},
    "drop": {"speed_m_s": 0.25, "watch_ticks": 50},
    "lost_z_m": -0.05,
    "photo": json.loads(json.dumps(DEFAULT_INFO_CONFIG["photo"])),
    "info_layout": {},
    "agent_cameras": ["top", "wrist"],
    # the agent's cameras: "filament" (agent_render.py: physically based, the baked hall) or "mujoco" (render_twin.py)
    "agent_renderer": "filament",
    "cell": {},
    "layout": {},
    # the reach check of every non-nominal layout (module docstring); max_grasps = the candidates tried per unit (the
    # bench plan's grasp of its nest first, then the unit's own in the planner's tilt order)
    "reach_check": {"enabled": False, "max_draws": 8, "max_grasps": 60},
    "oracle_nominal_layout": True,
    "privileged_wall_cap_s": 7200.0,
    "qc_jigs": False,
}


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


class _StrutBaseWatch(_BaseCellServer):
    """The bottom layer of the safety watch (R1, R4, held / released / dropped / lost at the centre of mass)."""

    def _watch_base(self, d):
        m = self.model
        mj = self.mujoco
        events = []
        thr_f = float(self.cfg["safety"]["panda_force_N"])
        left, right = {}, {}
        r1 = {}
        for i in range(d.ncon):
            con = d.contact[i]
            c1, c2 = self.geom_class[con.geom1], self.geom_class[con.geom2]
            if c1 == 0 and c2 == 0:
                continue
            mj.mj_contactForce(m, d, i, self._f6)
            fn = abs(float(self._f6[0]))
            if (c1 == 1 and c2 in (4, 5)) or (c2 == 1 and c1 in (4, 5)):
                # R1 sums over the whole arm (every non-pad collision geom of every link) per body it presses on
                other = con.geom2 if c1 == 1 else con.geom1
                key = int(m.geom_bodyid[other])
                r1[key] = r1.get(key, 0.0) + fn
            for cls, g in ((c1, con.geom2), (c2, con.geom1)):
                if cls in (2, 3) and self.geom_class[g] == 4:
                    part = self.geom_part[g]
                    (left if cls == 2 else right)[part] = (left if cls == 2 else right).get(part, 0.0) + fn
        r1max, r1arg = 0.0, None
        for key, f in r1.items():
            if f > r1max:
                r1max, r1arg = f, key
        self.readings["R1_N"] = r1max
        self.readings["R1_pair"] = None if r1arg is None else ("panda", m.body(r1arg).name)
        if r1max > thr_f:
            what = "robot_contact:panda:%s:%.1fN" % (m.body(r1arg).name, r1max)
            if not self.violations or self.violations[-1]["t"] < d.time - 0.5 or self.violations[-1]["what"].split(":")[1:3] != what.split(":")[1:3]:
                self.violations.append({"t": float(d.time), "what": what, "force_N": r1max, "rule": "R1"})
                events.append("safety:" + what)
        pad = {p: left.get(p, 0.0) + right.get(p, 0.0) for p in set(left) | set(right)}
        self.readings["pad_N"] = max(pad.values()) if pad else 0.0
        thr_h = self.cfg["held"]["force_N"]
        cand, best = None, 0.0
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
        thr_v = float(self.cfg["safety"]["part_speed_m_s"])
        v6 = self._v6
        vmax = 0.0
        for n in self.meta["loose"]:
            b = self.part_body[n]
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, b, v6, 0)        # at the body's centre of mass (xipos)
            v = float(np.linalg.norm(v6[3:]))
            z = float(d.xipos[b][2])
            vmax = max(vmax, v)
            if v > self._v_peak.get(n, 0.0):
                self._v_peak[n] = v
            if v > thr_v:
                what = "part_speed:%s:%.2fm/s" % (n, v)
                if not any(vi["what"].startswith("part_speed:" + n) for vi in self.violations[-3:]):
                    self.violations.append({"t": float(d.time), "what": what, "speed": v, "rule": "R4"})
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
        self.readings["R4_m_s"] = vmax
        return events


class StrutInfoServer(InfoCellServer, _StrutBaseWatch):
    """`kit_dir` = the hidden kit (gt: kit.json, mating_graph.json, meshes/), `panda_dir` = the given Panda model folder,
    `plan` = the hidden bench plan ({"layout": layout config overrides, ...}; the minimum version uses the nominal plan
    of layout.DEFAULT_LAYOUT)."""

    def __init__(self, kit_dir, panda_dir, plan=None, config=None, mode="hidden", privileged=False, render=True, log=None, kit_info=None):
        import mujoco
        if mode != "hidden":
            raise ValueError("this cell has no practice mode")
        self.mujoco = mujoco
        self.cfg = _deep_update(json.loads(json.dumps(DEFAULT_STRUT_CONFIG)), config or {})
        self.kit_dir, self.panda_dir = kit_dir, panda_dir
        self.kit = StrutKit(kit_dir)
        self.plan = dict(plan or {})
        self.scope = str(self.cfg.get("scope", "single"))
        lay = _deep_update(LY.defaults(self.scope), self.cfg.get("layout") or {})
        lay = _deep_update(lay, self.plan.get("layout") or {})
        self.layout_cfg = lay
        self.units = list(self.cfg.get("units") or lay["plan_nests"].keys())
        self.hullsets = {b: TR.HullSet(self.kit, b) for b in sorted(set(self.kit.units[u]["build"] for u in self.units))}
        self.mode, self.allow_reset = mode, False
        self.privileged, self.render_enabled = bool(privileged), bool(render)
        self.log = log or (lambda s: None)
        self.kit_info = kit_info or {}
        self.camera_names = tuple(self.cfg.get("agent_cameras") or ("top", "wrist"))
        self.model = self.data = self.seed = self.cell = None
        self.done = False
        self.session = None
        self.renderer = None
        self.token_mismatch = False
        self.calls = {}
        self.illegal_calls = 0
        self.reading_hooks = []

    # ------------------------------------------------------------------ episode setup
    def reset(self, seed):
        self.episode = LI.draw_episode(seed, self.cfg.get("info_layout"), strict=False)
        missing = [n for n in self.camera_names if self.episode["cameras"].get(n) is None]
        if missing and self.render_enabled:
            raise LI.MissingMount("camera(s) %s have no mount in the cell's config" % ", ".join(missing))
        nominal = bool(self.privileged and self.cfg.get("oracle_nominal_layout", True))
        t0 = time.time()
        rc = self.cfg.get("reach_check") or {}
        draws, reach = [], None
        for k in range(1 if (nominal or not rc.get("enabled")) else max(1, int(rc.get("max_draws", 8)))):
            sub = int(seed) if k == 0 else (int(seed) ^ (0x9E3779B97F4A7C15 * k)) & ((1 << 63) - 1)
            lay = self._draw_layout(sub, nominal)
            cell = self._build_cell(lay)
            if nominal or not rc.get("enabled"):
                break
            reach = self._reach_check(cell, lay)
            draws.append({"draw": k, "ok": reach["ok"], "failed": reach["failed"], "wall_s": reach["wall_s"]})
            if reach["ok"]:
                break
            self.log("[cell] layout draw %d failed the reach check (%s): drawn again" % (k, ", ".join(reach["failed"])))
        else:
            # every draw failed: the nominal layout (the bench plan's, reachable by construction) with this episode's cradle
            lay = self._draw_layout(int(seed), True)
            lay["fixture"] = {"xy": [float(v) for v in FX.cradle_xy(int(seed), (self.cfg.get("cell") or {}).get("fixture"))]}
            lay["jitter"]["fallback_nominal"] = True
            cell = self._build_cell(lay)
            draws.append({"draw": "nominal", "ok": True, "failed": []})
        lay["draws"] = draws
        self.log("[cell] strut cell for seed %d built in %.1f s (%d draw%s)" % (int(seed) & 0xFFFFFFFF, time.time() - t0, max(1, len(draws)),
                                                                             "" if len(draws) <= 1 else "s"))
        cell.recording = False
        self.cell = cell
        self.model, self.data = cell.m, cell.d
        self.ctrl, self.sm = cell.ctrl, cell.sm
        self.tracker = None
        self.meta = {"loose": list(cell.loose), "pad_geoms": list(cell.robot["pad_geoms"]), "root": list(cell.carriers.values())[0],
                     "welds": {}, "config": {"tick_steps": cell.tick_steps, "panda_pos": list(cell.cfg["panda_pos"]),
                                             "tcp_offset_m": float(cell.cfg["tcp_offset_m"])}}
        moved = {u: round(float(np.linalg.norm(cell.body_T(u)[:3, 3] - np.asarray(lay["parts"][u][0], float))) * 1e3, 4) for u in lay["parts"]}
        self.layout = dict(_jsonable(lay), settle_moves_mm=moved)
        self._init_state(seed)
        self._close_session()
        self.frames_used = 0
        self.frames_by_camera = {n: 0 for n in self.camera_names}
        self.render_wall_s = 0.0
        self.ep_seed = int(seed)
        hi = (int(seed) >> 32) & 0xFFFFFFFF
        self._wrench_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x3E17C4] + ([hi] if hi else []))
        # every control tick goes through the cell's watch, whoever drives it (an agent's step() or the reference)
        self._raw_tick = cell.tick_one

        def routed():
            self.physics_tick(track=True)
        cell.tick_one = routed
        return {"ok": True}

    def _draw_layout(self, seed, nominal):
        cc = self.cfg.get("cell") or {}
        return LY.episode_layout(seed, self.kit, self.layout_cfg, units=self.units, nominal=nominal, scope=self.scope,
                                 tray_cfg=cc.get("tray"), fixture_cfg=cc.get("fixture"), hullsets=self.hullsets)

    def _build_cell(self, lay):
        return StrutCell(self.kit_dir, self.panda_dir, lay, config=self.cfg.get("cell"), scope=self.scope, units=self.units, log=self.log,
                         settle_s=float(self.cfg["settle_s"]), jigs=bool(self.cfg.get("qc_jigs")), kit=self.kit)

    def _reach_check(self, cell, lay):
        """every loose unit keeps a grasp that passes the reference's pick checks where it rests in the built cell
        (reference.Assembler.pick_checks: approach from above, every IK solution, no contact on the approach and while
        closing, every other unit in its nest): the bench plan's grasp of its nest first (same build), then the unit's
        own candidates in the planner's tilt order (bench_plan.tilt_order for the slot its nest feeds in the plan; an
        end cap's 10 deg grid), at most `reach_check.max_grasps`"""
        from .bench_plan import NEST_SLOT_DEFAULT, nest_slot_of, tilt_order
        from .reference import Assembler
        t0 = time.time()
        rc = self.cfg.get("reach_check") or {}
        cap = int(rc.get("max_grasps", 60))
        asm = Assembler(cell, plan=self.plan, log=lambda *a: None, monitor=False)
        nest_slot = dict(NEST_SLOT_DEFAULT if self.scope == "default" else {}, **nest_slot_of(self.plan))
        by_nest = {st["nest"]: st for st in self.plan.get("steps", [])}
        failed, first = [], {}
        for u in cell.loose:
            nest = lay["nests"][u]
            sid = nest_slot[nest]
            T_u = asm.unit_T(u)
            cands = []
            st = by_nest.get(nest)
            if st and st.get("grasp") and st.get("build") == self.kit.units[u]["build"]:
                cands.append((np.array(st["grasp"]["G"], float), float(st["grasp"]["w"]), "plan"))
            cands += tilt_order(asm, sid, u, asm._own_candidates(u, sid, extra_from_pose=False), T_u)
            hit = None
            for i, (G, w, _desc) in enumerate(cands[:cap]):
                if asm.pick_checks(u, T_u, G, w)[0] is not None:
                    hit = i
                    break
            if hit is None:
                failed.append("pick %s (%s)" % (u, nest))
            else:
                first[u] = hit
        return {"ok": not failed, "failed": failed, "first_passing": first, "wall_s": round(time.time() - t0, 2)}

    def _init_state(self, seed):
        m = self.model
        cell = self.cell
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
        self._v_peak = {}
        self._v6 = np.zeros(6)
        self.wall_start = None
        self.wall_used = 0.0
        self.renderer = None
        self.tick_s = cell.tick_s
        self.tick_steps = cell.tick_steps
        self._obs_noise = (0.0, 0.0)
        self._obs_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x0B5E])
        self.geom_class = np.zeros(m.ngeom, dtype=int)
        self.geom_part = {}
        robot_bodies = set(b for b in range(m.nbody) if m.body(b).name.startswith("panda/"))
        loose = set(cell.loose)
        self.loose_body_ids = {m.body(u).id: u for u in loose}
        static = [0, m.body("cradle").id] + [m.body(c).id for c in cell.carriers.values()] + [m.body(t).id for t in cell.layout["trays"]]
        self.static_env_bodies = set(static)
        self.carrier_body_ids = {m.body(c).id: c for c in cell.carriers.values()}
        pads = set(self.meta["pad_geoms"])
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            name = m.geom(g).name
            if name in pads:
                self.geom_class[g] = 2 if "left" in name else 3
            elif b in robot_bodies:
                self.geom_class[g] = 1
            elif b in self.loose_body_ids:
                self.geom_class[g] = 4
                self.geom_part[g] = self.loose_body_ids[b]
            elif b in self.static_env_bodies:
                self.geom_class[g] = 5
        self.part_body = {n: m.body(n).id for n in cell.loose}
        self.part_qadr = {n: m.jnt_qposadr[m.joint("free_" + n).id] for n in cell.loose}
        self.part_vadr = {n: m.jnt_dofadr[m.joint("free_" + n).id] for n in cell.loose}
        self.sensor_force = m.sensor("hand_force").id
        self.sensor_torque = m.sensor("hand_torque").id
        self.wrist_parent = "panda/hand"
        self._f6 = np.zeros(6)
        self.readings = {"R1_N": 0.0, "R2_N": 0.0, "R2_raw_N": 0.0, "R3_N": 0.0, "R4_m_s": 0.0, "pad_N": 0.0}
        self._peak = {k: 0.0 for k in ("R1_N", "R2_N", "R2_raw_N", "R3_N", "R4_m_s", "pad_N")}
        self._r2_hist, self._r3_hist = [], []
        self._sustained = {"R2_N": 0.0, "R3_N": 0.0}
        self._r2_over, self._r3_over = 0, {}

    def _welded(self, part):
        st = self.sm.rec
        return any(r["fastened"] and r["unit"] == part for r in st.values())

    # ------------------------------------------------------------------ physics
    def physics_tick(self, track=True):
        if self.done:
            return []
        n0 = len(self.sm.events)
        self._raw_tick()
        self.tick_count += 1
        events = []
        if track:
            events += ["%s" % e for _t, e in self.sm.events[n0:]]
            events += self._watch(self.data)
            for e in events:
                self.events_log.append((float(self.data.time), e))
        if self.data.time >= self.cfg["sim_cap_s"] - 1e-9:
            self._end("sim_cap")
        return events

    def clusters(self):
        """{body id: cluster root}: bodies joined by a chain of fastened mates (the strut core and its locked units)"""
        parent = {}

        def root(b):
            parent.setdefault(b, b)
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            return b
        for sid, r in self.sm.rec.items():
            if r["fastened"] and r["unit"] is not None:
                x = self.sm.by_id[sid]
                parent[root(self.part_body[r["unit"]])] = root(x["bc"])
        return {b: root(b) for b in list(parent)}

    def _watch(self, d):
        """this cell's four rules (module docstring): the base layer (R1, R4, held / dropped / lost), then R2 and R3"""
        events = self._watch_base(d)
        m, mj = self.model, self.mujoco
        dmg = self.cfg.get("damage") or {}
        excl = bool(dmg.get("exclude_locked_pairs", True))
        cl = self.clusters() if excl else {}
        part_sum, raw_sum = {}, {}
        for i in range(d.ncon):
            con = d.contact[i]
            c1, c2 = self.geom_class[con.geom1], self.geom_class[con.geom2]
            if not ((c1 == 4 and c2 in (4, 5)) or (c2 == 4 and c1 in (4, 5))):
                continue
            mj.mj_contactForce(m, d, i, self._f6)
            fn = abs(float(self._f6[0]))
            g_part = con.geom1 if c1 == 4 else con.geom2
            g_other = con.geom2 if c1 == 4 else con.geom1
            bp, bo = int(m.geom_bodyid[g_part]), int(m.geom_bodyid[g_other])
            key = (self.geom_part[g_part], m.body(bo).name)
            raw_sum[key] = raw_sum.get(key, 0.0) + fn
            if excl and bp in cl and bo in cl and cl[bp] == cl[bo]:
                continue
            part_sum[key] = part_sum.get(key, 0.0) + fn
        pm = max(part_sum.values()) if part_sum else 0.0
        parg = max(part_sum, key=part_sum.get) if part_sum else None
        self.readings["R2_N"] = pm
        self.readings["R2_pair"] = parg
        self.readings["R2_raw_N"] = max(raw_sum.values()) if raw_sum else 0.0
        ticks = int(dmg.get("ticks", 20))
        cap = dmg.get("part_force_N")
        if cap is not None and pm > float(cap):
            self._r2_over += 1
            if self._r2_over >= ticks:
                w = "part_force:%s:%s:%.1fN" % (parg[0], parg[1], pm)
                if not any(v["what"].split(":")[:2] == w.split(":")[:2] and v["t"] > d.time - 0.5 for v in self.violations[-3:]):
                    self.violations.append({"t": float(d.time), "what": w, "force_N": pm, "rule": "R2"})
                    events.append("safety:" + w)
        else:
            self._r2_over = 0
        # R3: the rod's load per overhang
        rc = self.cfg.get("rod") or {}
        loads = self.rod_loads()
        r3max, r3arg = 0.0, None
        for strut, o in loads.items():
            for end in ("p", "n"):
                f = float(o[end]["lateral"])
                if f > r3max:
                    r3max, r3arg = f, (strut, end)
                key = (strut, end)
                thr = rc.get("lateral_force_N")
                if thr is not None and f > float(thr):
                    self._r3_over[key] = self._r3_over.get(key, 0) + 1
                    if self._r3_over[key] >= int(rc.get("ticks", 20)):
                        w = "rod_load:%s:%s:%.2fN" % (strut, end, f)
                        if not any(v["what"].split(":")[:3] == w.split(":")[:3] and v["t"] > d.time - 0.5 for v in self.violations[-3:]):
                            self.violations.append({"t": float(d.time), "what": w, "force_N": f, "rule": "R3"})
                            events.append("safety:" + w)
                else:
                    self._r3_over[key] = 0
        self.readings["R3_N"] = r3max
        self.readings["R3_where"] = r3arg
        # peaks and 0.2 s-sustained maxima (the definition of the rules' windows: the smallest reading over the window)
        for k in self._peak:
            self._peak[k] = max(self._peak[k], float(self.readings.get(k, 0.0)))
        for hist, k, n in ((self._r2_hist, "R2_N", ticks), (self._r3_hist, "R3_N", int(rc.get("ticks", 20)))):
            hist.append(float(self.readings[k]))
            if len(hist) > n:
                del hist[0]
            if len(hist) == n:
                self._sustained[k] = max(self._sustained[k], min(hist))
        for h in self.reading_hooks:
            h(self)
        return events

    def rod_loads(self):
        """R3's reading per overhang: the coupling's (coupling.StrutCoupling.rod_loads: the rail / lock welds' force on
        the strut side + the contacts on the rod's collision geom beyond the bay) plus, for every module bound or fastened
        on that end, the force it puts on the strut core through contacts with the core's other geoms (the bay): with it
        the reading is what the module and the rod's own contacts put on the strut from OUTSIDE the module-core pair (its
        weight, a push, a neighbour), and a preload between a module and the core (a module pressed on the bay end while
        welded) cancels instead of reading as a rod load"""
        m, d, mj = self.model, self.data, self.mujoco
        loads = self.sm.rod_loads(include_contacts=True)
        held = {}
        for x in self.sm.slots:
            if x["kind"] != "rod_sleeve":
                continue
            st = self.sm.rec[x["id"]]
            ci = self.sm.bound.get(x["id"])
            u = st["unit"] if st["fastened"] else (x["cands"][ci]["unit"] if ci is not None else None)
            if u is not None:
                held[self.part_body[u]] = (x["strut"], x["end"], x["bc"])
        if not held:
            return loads
        rods = set(self.sm.rod_geoms.values())
        f6 = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            for bu, bo, gu, go in ((b1, b2, con.geom1, con.geom2), (b2, b1, con.geom2, con.geom1)):
                h = held.get(bu)
                if h is None or bo != h[2] or go in rods:
                    continue
                mj.mj_contactForce(m, d, i, f6)
                fw = con.frame.reshape(3, 3).T @ f6[:3]          # on geom2 from geom1 (contact frame normal from 1 to 2)
                f_on_core = fw if go == con.geom2 else -fw
                loads[h[0]][h[1]]["contacts"] = loads[h[0]][h[1]]["contacts"] + f_on_core
                loads[h[0]][h[1]].setdefault("core_contacts", np.zeros(3))
                loads[h[0]][h[1]]["core_contacts"] = loads[h[0]][h[1]]["core_contacts"] + f_on_core
        for strut, o in loads.items():
            u_ax = cpose(d, m.body(self.cell.carriers[strut]).id)[:3, 2]
            for k in ("p", "n"):
                F = o[k]["welds"] + o[k]["contacts"]
                o[k]["F"] = F
                o[k]["lateral"] = float(np.linalg.norm(F - (F @ u_ax) * u_ax))
        return loads

    def damage_summary(self):
        return {"peak": {k: round(v, 4) for k, v in self._peak.items()},
                "sustained_0.2s": {k: round(v, 4) for k, v in self._sustained.items()},
                "max_part_speed_m_s": {k: round(v, 3) for k, v in self._v_peak.items()}}

    # ------------------------------------------------------------------ the restricted API
    def describe(self):
        m = self.model
        cams = {}
        for n in self.camera_names:
            c = self.episode["config"]["cameras"][n]
            cams[n] = {"resolution": list(c["size"]), "rgb": True, "depth": bool(c.get("depth", True)),
                       "nominal_hfov_deg": round(float(c["hfov_deg"])), "mount": _MOUNT_TEXT.get(n, "")}
        ct = self.ctrl
        fx = FX.merged((self.cfg.get("cell") or {}).get("fixture"))
        trays = TR.trays_of((self.cfg.get("cell") or {}).get("tray"), self.scope)
        pads = (self.cell.cfg.get("pads") or {})
        n_struts = len(self.cell.struts)
        counts = {}
        for u in self.cell.loose:
            counts[self.kit.units[u]["type"]] = counts.get(self.kit.units[u]["type"], 0) + 1
        if n_struts == 1:
            task = ("assemble the strut on the cradle: slide the two motor modules onto the rod, one on each side of the electronics bay, "
                    "at their design position and turn; put the bay lid on the bay; push an end cap onto each rod end (both caps of a strut "
                    "of one colour), as the picture manual shows")
            core_text = "the strut core (the electronics bay base with its board, the aluminium rod) clamped in the cradle, its rod"
        else:
            task = ("assemble the %d struts on the cradle: on each, slide two motor modules onto the rod, one on each side of the electronics "
                    "bay, at their design position and turn; put a bay lid on the bay; push an end cap onto each rod end (both caps of a strut "
                    "of one colour; each strut may take any colour), as the picture manual shows" % n_struts)
            core_text = ("%d strut cores (each the electronics bay base with its board and the aluminium rod) clamped in the cradle's seats "
                         "%.0f mm apart, their rods" % (n_struts, 1e3 * float(fx["seat_spacing_m"])))
        loose_text = ("the loose modules (%d motor modules, %d bay lid%s, %d end caps with their covers) resting in the nests of the parts tray%s; "
                      "the place%s of the tray%s and %s turn about the vertical vary between episodes, and so do which nest holds which module, a "
                      "motor module's turn about its own axis (the nests do not hold the mounting turn), an end cap's turn and a lid's "
                      "direction (either end first)") % (counts.get("M1", 0), counts.get("M2", 0), "" if counts.get("M2", 0) == 1 else "s", counts.get("M3", 0),
                                                         "" if len(trays) == 1 else "s", "" if len(trays) == 1 else "s", "" if len(trays) == 1 else "s",
                                                         "its" if len(trays) == 1 else "their")
        tray_frame = ("origin on the plate's top face at its centre; x across the nests, y along the modules' axes (a module's bore "
                      "axis lies along y, the battery leads' end of a motor module and the bore mouth of an end cap towards -y)")
        tray_docs = {}
        for tn, t in trays.items():
            nom = (self.layout_cfg.get("trays") or {}).get(tn) or {}
            tray_docs[tn] = {"plate_m": list(t["plate"]),
                             "nests": [{"nest": n, "kind": k, "x_in_tray_m": float(x), "y_in_tray_m": float(y)} for n, k, x, y in t["nests"]],
                             "nominal_place": "around (%.2f, %.2f) m, turned %.0f deg about the vertical (its -y towards the cradle)" % (
                                 float(nom.get("nominal_xy", [0, 0])[0]), float(nom.get("nominal_xy", [0, 0])[1]), float(nom.get("yaw_deg", 0.0)))}
        if len(trays) == 1:
            (tn, td), = tray_docs.items()
            trays_out = {"plate_m": td["plate_m"], "nests": [{k: v for k, v in n.items() if k != "y_in_tray_m"} for n in td["nests"]],
                         "tray_frame": tray_frame, "nominal_place": "around (%.2f, %.2f) m, its y along the cell's y" % tuple(
                             (self.layout_cfg.get("trays") or {}).get(tn, {}).get("nominal_xy", [0, 0]))}
        else:
            trays_out = {"trays": tray_docs, "tray_frame": tray_frame}
        return {
            "cell_version": CELL_VERSION, "units": {"length": "m", "angle": "rad", "quaternion": "wxyz"},
            "task": task,
            "robot": {"model": "7-DoF Panda-class arm (given model: panda/)", "base_pos": list(self.cell.cfg["panda_pos"]), "base_axes": "aligned with the cell frame",
                      "joints": ["panda/joint%d" % (i + 1) for i in range(7)],
                      "joint_range": [[float(m.jnt_range[j][0]), float(m.jnt_range[j][1])] for j in ct.joints],
                      "home_q": [float(v) for v in ct.q_cmd], "tcp_site": "tcp", "tcp_offset_in_hand_m": [0.0, 0.0, float(self.cell.cfg["tcp_offset_m"])],
                      "gripper": {"max_width_m": ct.max_width, "rate_m_s": 0.05, "pad_size_m": [0.017, 0.0165],
                                  "pads": "stock flat pads" if pads.get("profile", "flat") == "flat" else "pads with a shallow V groove along the hand's x",
                                  "force_model": "2500 N/m x (object width - target width) / 2 per finger, capped at 60 N"},
                      "gravity_compensated": True,
                      "ft_sensor": "6-axis at the hand flange (hand frame); noise ~0.3 N / 0.02 N m rms, an unknown per-episode bias up to 1 N / 0.05 N m; "
                                   "it includes the hand's own weight"},
            "control": {"tick_s": self.tick_s, "joint_rate_limit_rad_s": 1.5, "cartesian_speed_limit_m_s": 0.5, "cartesian_angular_limit_rad_s": 1.5,
                        "stiffness_scale_range": [0.2, 1.0], "modes": ["joint", "cartesian"]},
            "fastening": self._fastening_text(),
            "cameras": cams, "camera_budget_frames": int(self.cfg["camera_budget_frames"]),
            "datums": strut_datums(self.cfg),
            "items": ["%s along the cell's y about 75 mm above the bench, the lid's seat facing up; the cradle at about (%.2f, %.2f) m (+-10 mm)" % (
                          core_text, fx["nominal_xy"][0], fx["nominal_xy"][1]), loose_text],
            "trays": trays_out,
            "budgets": {"sim_time_cap_s": self.cfg["sim_cap_s"], "wall_clock_cap_s": self.cfg["wall_cap_s"], "silence_abort_s": self.cfg["silence_s"],
                        "max_ticks_per_step": self.cfg["max_ticks_per_step"], "camera_frames": int(self.cfg["camera_budget_frames"])},
            "safety": {"panda_force_N": self.cfg["safety"]["panda_force_N"], "part_speed_m_s": self.cfg["safety"]["part_speed_m_s"],
                       "part_force_N": self.cfg["damage"]["part_force_N"], "part_force_window_s": 0.01 * int(self.cfg["damage"]["ticks"]),
                       "rod_lateral_force_N": self.cfg["rod"]["lateral_force_N"], "rod_window_s": 0.01 * int(self.cfg["rod"]["ticks"]),
                       "rules": ["R1: the robot (any link or finger surface other than the two fingertip pads) pressing on the cradle, a tray, the "
                                 "bench, a module or a strut core above panda_force_N (normal forces summed over the robot per body it presses on) in "
                                 "any tick",
                                 "R2: a module pressed on the environment or another module above part_force_N for part_force_window_s (normal "
                                 "forces summed per module and body; the pads' grip does not count, nor do two modules fastened to each other)",
                                 "R3: the rod loaded across its axis above rod_lateral_force_N for rod_window_s on either side of the bay (everything "
                                 "the rod carries there: contacts and the modules on it, their weight included)",
                                 "R4: a module's centre of mass faster than part_speed_m_s"]},
        }

    def _fastening_text(self):
        sm = self.sm
        return {
            "rod": ("a module with a bore is taken onto the rod once its leading bore mouth is within %.1f mm of the rod's axis, its axis "
                    "within %.0f deg of the rod's and the rod end at least %.0f mm into the bore; it then slides along the rod and turns about it "
                    "with a little damping while it is held clear of the rod; let go before it is fastened, it tilts until its bore rests on "
                    "the rod, whose friction holds it (its turn moves by up to about 30 deg as it settles), and it is fastened where it rests "
                    "(moving slower than %.0f mm/s and turning slower than %.0f deg/s "
                    "relative to the strut for %d consecutive 10 ms ticks, it may still be held) once it is within the fastening window of its "
                    "seat (cell_manual.md gives the windows); an end cap is taken on only when the motor module of its rod end is fastened "
                    "and counts; nothing is reported; a fastened module cannot be moved again" % (
                        sm.cap_lat * 1e3, math.degrees(sm.cap_ang), sm.cap_depth * 1e3, sm.rel_speed_max * 1e3, math.degrees(sm.rest_ang), sm.hold_ticks)),
            "lid": ("the lid is taken onto its seat within %.0f mm across, at most %.0f mm short of home and %.0f deg; the cell pulls it home at "
                    "<= %.0f mm/s and fastens it when it rests there; it needs both motor modules of its strut fastened and counting "
                    "first" % (
                        sm.seat_lat * 1e3, sm.seat_depth * 1e3, math.degrees(sm.seat_ang), sm.pull_v * 1e3)),
        }

    def _session(self):
        if self.session is None:
            from . import render_twin
            t = time.time()
            want = str(self.cfg.get("agent_renderer", "filament"))
            if want == "filament":
                from . import agent_render
                ok, why = agent_render.available()
                if not ok:
                    raise RuntimeError("agent renderer filament unavailable: %s" % why)
                self.session = agent_render.open_session(self)
                self.renderer = "filament"
            else:
                self.session = render_twin.open_session(self)
                self.renderer = "mujoco"
            self.session_open_s = time.time() - t
            self.render_wall_s += time.time() - t
            self.log("[cell] %s render session opened in %.1f s" % (self.renderer, time.time() - t))
        return self.session

    def camera_spec(self, name):
        from render.spec import CameraSpec
        c = self.episode["cameras"][name]
        intr = c["intrinsics"]
        if name == "wrist":
            b = self.model.body(self.wrist_parent).id
            T = np.eye(4)
            T[:3, :3] = self.data.xmat[b].reshape(3, 3)
            T[:3, 3] = self.data.xpos[b]
            W = T @ np.asarray(c["hand_T_cam"], float)
        else:
            W = np.asarray(c["world_T_cam"], float)
        return CameraSpec(name=name, width=intr["width"], height=intr["height"], fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"],
                          world_T_cam=W.tolist(), k1=intr["k1"], k2=intr["k2"], p1=intr["p1"], p2=intr["p2"])

    def photo_spec(self, name):
        import dataclasses
        from render.spec import derive_seed
        p = super().photo_spec(name)
        return dataclasses.replace(p, colour_jitter=dataclasses.replace(p.colour_jitter, seed=derive_seed(self.ep_seed, "colour_unit", name)))

    def get_camera(self, name="top", depth=True):
        if str(name) not in self.camera_names:
            return {"error": "unknown camera (cameras: %s)" % ", ".join(self.camera_names)}
        return super().get_camera(name, depth=depth)

    def describe_privileged(self):
        out = {"layout": self.layout, "mates": [dict(g) for g in self.kit.mates(self.scope)], "states": dict(self.sm.state), "score": self.sm.score()}
        return _jsonable(out)

    def get_state_privileged(self):
        d = self.data
        parts = {n: {"pos": [float(v) for v in d.xpos[b]], "quat": [float(v) for v in d.xquat[b]], "com": [float(v) for v in d.xipos[b]]}
                 for n, b in self.part_body.items()}
        return {"sim_time": float(d.time), "parts": parts, "states": dict(self.sm.state), "records": _jsonable(self.sm.record()),
                "score": self.sm.score(), "readings": _jsonable(self.readings)}

    def reference_assemble(self):
        """The privileged reference program on this episode (oracle token only): reference.program inside the cell, on
        the bench plan's nominal layout (`plan`: the hidden bench_plan.json), every control tick through this server's
        routed tick and safety watch (reset replaced the cell's tick_one), with its own wall-clock cap
        (`privileged_wall_cap_s`: the silence rule does not run while the call is busy)."""
        try:
            from .reference import program
        except ImportError:
            return {"ok": False, "error": "no reference program in this build"}
        t0 = time.time()
        rep = program(self.cell, plan=self.plan, log=self.log, server=self, wall_cap_s=float(self.cfg.get("privileged_wall_cap_s", 7200.0)))
        rep.pop("events", None)
        return {"ok": True, "score": float(self.sm.score()), "sim_s": float(self.data.time), "wall_s": float(time.time() - t0),
                "report": _jsonable(rep)}

    def handle(self, method, params):
        if method in PRIVILEGED_METHODS:
            self.calls[method] = self.calls.get(method, 0) + 1
            if not self.privileged:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            return getattr(self, method)()
        if method == "truth":
            self.calls[method] = self.calls.get(method, 0) + 1
            self.illegal_calls += 1
            return {"error": "forbidden"}
        return super().handle(method, params)

    # ------------------------------------------------------------------ end of episode
    def finalize(self, functional_rng=None, run_functional_check=False):
        self._close_session()
        if not self.done:
            self._end("solve_returned")
        rec = self.sm.record()
        weights = {r["id"]: float(r["weight"]) for r in rec}
        W = sum(weights.values())
        achieved_w = sum(weights[r["id"]] for r in rec if r["achieved"])
        d = self.data
        return {
            "cell_version": CELL_VERSION, "seed": self.seed, "scope": self.scope, "done_reason": self.done_reason,
            "sim_time_s": float(d.time), "wall_time_s": float(self.wall_used), "ticks": int(self.tick_count),
            "mates": _jsonable(rec), "states": dict(self.sm.state), "weights": weights, "stage_score": achieved_w / W if W else 0.0,
            "n_achieved": int(sum(1 for r in rec if r["achieved"])), "n_fastened": int(sum(1 for r in rec if r.get("fastened"))),
            "all_mates": bool(all(r["achieved"] for r in rec)),
            "off_design": [r["id"] for r in rec if r.get("fastened") and not r["achieved"]],
            "violations": list(self.violations), "safety_ok": len(self.violations) == 0,
            "illegal_call_count": int(self.illegal_calls), "calls": dict(self.calls), "step_calls": int(self.step_calls),
            "lost_parts": sorted(self._lost), "dropped_parts": sorted(self._dropped),
            "events": [[t, e] for t, e in self.events_log][-400:], "coupling_events": [list(e) for e in self.sm.events],
            "captures": _jsonable(self.sm.captures), "layout": self.layout, "damage": self.damage_summary(),
            "camera_frames": dict(self.frames_by_camera), "camera_frames_total": int(self.frames_used),
            "render_wall_s": float(self.render_wall_s), "renderer": getattr(self, "renderer", None), "privileged_mode": bool(self.privileged),
            "lowest_part_z_m": float(self.cell.lowest_point()),
        }


def strut_datums(cfg=None):
    """the documented datums: the precedent's bench board and hand plate (cell_info.cell_datums) and the cradle board"""
    out = dict(cell_datums())
    fx = FX.merged(((cfg or {}).get("cell") or {}).get("fixture"))
    b = fx["board"]
    out.pop("fixture_board", None)
    out.pop("fixture", None)
    yaw = float(b.get("yaw_deg", 0.0)) % 180.0
    cols = "the cell's x" if abs(yaw) < 45.0 or abs(yaw - 180.0) < 45.0 else "the cell's y"
    out["cradle_board"] = {"squares": [b["cols"], b["rows"]], "square_mm": b["square_mm"], "margin_mm": b["margin_mm"], "thickness_mm": b["thickness_mm"],
                           "mounting": "flat on the cradle plate's top, at its +x edge (away from the robot), centred %.0f mm from the cradle's middle "
                                       "seat along x, columns along %s" % (float(b["x_mm"]), cols),
                           "note": "moves with the cradle (+-10 mm)"}
    out["cradle"] = {"plate_m": list(fx["plate"]), "seat_spacing_m": float(fx["seat_spacing_m"]), "rod_height_m": float(fx["rod_z"]),
                     "nominal_middle_seat_m": list(fx["nominal_xy"]), "tolerance_mm": 1e3 * float(fx["jitter_m"])}
    return out


__all__ = ["StrutInfoServer", "CELL_VERSION", "DEFAULT_STRUT_CONFIG", "PRIVILEGED_METHODS", "strut_datums", "CELL_CONFIG"]
