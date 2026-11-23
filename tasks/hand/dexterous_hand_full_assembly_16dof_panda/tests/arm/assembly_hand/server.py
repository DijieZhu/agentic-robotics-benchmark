"""The hand assembly cell behind the camera-only API (`HandInfoServer`).

The physics is `cell.HandCell`: the palm's bottom shell on the raised fixture with its rest deck and the thumb's pad,
the nine loose units (three finger modules, the thumb module, the electronics stack, the upper shell, the dorsal
cover, the power hub, the press-fit cap) resting in the nests of the digit tray and the parts tray, the coupling (`coupling.py`:
`module_slide`, `part_seat`, `pressed_seat`, the tool access and the requires). The Panda and its controller are the
precedent's (`assembly_cell.controller`). The observation side is the information tier's
(`arm.assembly_info.cell_info.InfoCellServer`, inherited): uncalibrated RGB-D cameras (top, wrist) through the render
layer, a noisy flange F/T sensor, the joint state; no part poses, no mate events.

`HandInfoServer.__init__` calls no parent constructor (the tensegrity cell's pattern); the inherited methods read the
attributes set here and in `reset` / `_init_state`. The safety watch (`_watch`, every control tick, whoever drives
it: an agent's `step()` or the reference) is this cell's own, four rules (every number one key of
`safety` / `damage` / `stops` in the config):
  R1  a Panda link other than the two pads (all faces of the pad bodies are pads) pressing on anything: the normal
      forces of every non-pad geom of the arm summed per body it presses on, above `safety.panda_force_N` in any tick;
  R2  a unit loaded, for `damage.ticks` consecutive ticks above `damage.part_force_N`: (a) the summed contact force
      between a unit and any other body but the pads (per unit and body; contacts inside a fastened cluster, bodies
      joined by LOCKED / LOCKED_OFF mates, are a fastening's preload and do not count); (b) for a fastened or engaged
      unit, the net force its fastening carries (`coupling.fastening_loads`: the lock weld's or the guide's
      constraint force plus the contacts of the same pair, as vectors, so that the preload cancels);
  R3  a product joint against its mechanical stop: the torque its limit carries (`coupling.stop_loads`), above
      `stops.torque_Nm` for `stops.ticks` consecutive ticks;
  R4  a unit's centre of mass (the mass-weighted centre of its links) faster than `safety.part_speed_m_s`; a unit
      that falls off the bench counts here.
Every tick's readings (`self.readings`) and their peaks and 0.2 s-sustained maxima (`damage_summary()`) are kept.

Per episode (`layout.episode_layout`): the fixture +-10 mm / +-2 deg, each tray +-10 mm / +-5 deg, each finger
chain's ab/ad angle in its nest +-10 deg, the thumb's side angle +-8 deg, the upper shell, cover, hub and cap
nominal or turned 180 deg. A draw that fails the layout's own validity is drawn again inside the layout; the built
cell must then pass the reach check (`reach_check`: every loose unit keeps a grasp of the cell's grasp table that
reaches with the clearances, where it rests) or the WHOLE layout is drawn again (at most `reach_check.max_draws`
draws), the nominal layout last. With the oracle token (privileged mode) and `oracle_nominal_layout` the layout is
the nominal one and `reference_assemble()` runs the reference program inside the cell.
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
from .cell import HandCell
from .kit import HandKit

CELL_VERSION = "hand-assembly-cell/0.1.0"
# the word an R2 (b) event carries in place of the mate's key (the load read through a unit's guide or fastening)
R2B_EVENT_LABEL = "fastening"
PUBLIC_METHODS = ("describe", "get_state", "set_control", "command", "step", "get_camera", "finish")
PRIVILEGED_METHODS = ("get_state_privileged", "describe_privileged", "reference_assemble")

DEFAULT_HAND_CONFIG = {
    "sim_cap_s": 1200.0,
    "wall_cap_s": 2400.0,
    "silence_s": 120.0,
    "max_ticks_per_step": 500,
    "camera_budget_frames": 600,
    "settle_s": 1.0,
    "units": None,
    # the safety numbers (the task's grader_config.json gives the frozen values)
    "safety": {"panda_force_N": 20.0, "part_speed_m_s": 3.0},
    "damage": {"part_force_N": 25.0, "ticks": 20, "exclude_locked_pairs": True, "fastening_loads": True},
    "stops": {"torque_Nm": 0.20, "ticks": 20},
    "held": {"force_N": 1.0, "ticks": 5},
    "drop": {"speed_m_s": 0.25, "watch_ticks": 50},
    "lost_z_m": -0.05,
    "photo": json.loads(json.dumps(DEFAULT_INFO_CONFIG["photo"])),
    "info_layout": {},
    "agent_cameras": ["top", "wrist"],
    "agent_renderer": "filament",
    "cell": {},
    "layout": {},
    "reach_check": {"enabled": False, "max_draws": 8, "max_grasps": 24},
    "oracle_nominal_layout": True,
    "privileged_wall_cap_s": 7200.0,
    "qc_jigs": False,
}


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


class _HandBaseWatch(_BaseCellServer):
    """The bottom layer of the safety watch (R1, R4, held / released / dropped / lost at each unit's centre of mass)."""

    def unit_com_vel(self, u):
        """(centre of mass (3,), its linear velocity (3,)) of a unit's links"""
        m, d, mj = self.model, self.data, self.mujoco
        M, c, v = 0.0, np.zeros(3), np.zeros(3)
        for b in self.unit_link_ids[u]:
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, b, self._v6, 0)
            mb = float(m.body_mass[b])
            M += mb
            c += mb * d.xipos[b]
            v += mb * self._v6[3:]
        return c / M, v / M

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
        vmax = 0.0
        for n in self.meta["loose"]:
            c, v3 = self.unit_com_vel(n)
            v = float(np.linalg.norm(v3))
            z = float(c[2])
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


class HandInfoServer(InfoCellServer, _HandBaseWatch):
    """`kit_dir` = the hidden kit (gt: kit.json, mating_graph.json, joints.json, meshes/), `panda_dir` = the given Panda
    model folder, `plan` = the hidden bench plan ({"layout": layout config overrides, ...})."""

    def __init__(self, kit_dir, panda_dir, plan=None, config=None, mode="hidden", privileged=False, render=True, log=None, kit_info=None):
        import mujoco
        if mode != "hidden":
            raise ValueError("this cell has no practice mode")
        self.mujoco = mujoco
        self.cfg = _deep_update(json.loads(json.dumps(DEFAULT_HAND_CONFIG)), config or {})
        self.kit_dir, self.panda_dir = kit_dir, panda_dir
        self.kit = HandKit(kit_dir)
        self.plan = dict(plan or {})
        lay = _deep_update(LY.defaults(), self.cfg.get("layout") or {})
        lay = _deep_update(lay, self.plan.get("layout") or {})
        self.layout_cfg = lay
        self.units = list(self.cfg.get("units") or [u for u in self.kit.units if self.kit.kind_of(u) != "PB"])
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
        draws = []
        cell = lay = None
        for k in range(1 if (nominal or not rc.get("enabled")) else max(1, int(rc.get("max_draws", 8)))):
            sub = int(seed) if k == 0 else (int(seed) ^ (0x9E3779B97F4A7C15 * k)) & ((1 << 63) - 1)
            lay = self._draw_layout(sub, nominal)
            cell = self._build_cell(lay)
            if nominal or not rc.get("enabled"):
                break
            reach = self._reach_check(cell)
            draws.append({"draw": k, "ok": reach["ok"], "failed": reach["failed"], "wall_s": reach["wall_s"]})
            if reach["ok"]:
                break
            self.log("[cell] layout draw %d failed the reach check (%s): drawn again" % (k, ", ".join(reach["failed"])))
        else:
            lay = self._draw_layout(int(seed), True)
            lay["jitter"]["fallback_nominal"] = True
            cell = self._build_cell(lay)
            draws.append({"draw": "nominal", "ok": True, "failed": []})
        lay["draws"] = draws
        self.log("[cell] hand cell for seed %d built in %.1f s (%d draw%s)" % (int(seed) & 0xFFFFFFFF, time.time() - t0, max(1, len(draws)),
                                                                           "" if len(draws) <= 1 else "s"))
        self.cell = cell
        self.model, self.data = cell.m, cell.d
        self.ctrl, self.sm = cell.ctrl, cell.sm
        self.tracker = None
        self.meta = {"loose": list(cell.loose), "pad_geoms": list(cell.robot["pad_geoms"]), "root": cell.base_body,
                     "welds": {}, "config": {"tick_steps": cell.tick_steps, "panda_pos": list(cell.cfg["panda_pos"]),
                                             "tcp_offset_m": float(cell.cfg["tcp_offset_m"])}}
        moved = {u: round(float(np.linalg.norm(cell.unit_T(u)[:3, 3] - cell.planned[u][:3, 3])) * 1e3, 4) for u in cell.loose}
        self.layout = dict(_jsonable(lay), settle_moves_mm=moved)
        self._init_state(seed)
        self._close_session()
        self.frames_used = 0
        self.frames_by_camera = {n: 0 for n in self.camera_names}
        self.render_wall_s = 0.0
        self.ep_seed = int(seed)
        hi = (int(seed) >> 32) & 0xFFFFFFFF
        self._wrench_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x3E17C4] + ([hi] if hi else []))
        self._raw_tick = cell.tick_one

        def routed():
            self.physics_tick(track=True)
        cell.tick_one = routed
        return {"ok": True}

    def _draw_layout(self, seed, nominal):
        return LY.episode_layout(seed, self.kit, self.layout_cfg, nominal=nominal, cfg_cell=self.cfg.get("cell"))

    def _build_cell(self, lay):
        return HandCell(self.kit_dir, self.panda_dir, lay, config=self.cfg.get("cell"), units=self.units, log=self.log,
                        settle_s=float(self.cfg["settle_s"]), jigs=bool(self.cfg.get("qc_jigs")), kit=self.kit)

    def _reach_check(self, cell):
        """every loose unit keeps a grasp of the cell's grasp table (reach.py) that reaches with the planner's
        clearances where it rests in the built cell"""
        from . import reach as RC
        t0 = time.time()
        rc = self.cfg.get("reach_check") or {}
        r = RC.nest_reach(cell, max_grasps=int(rc.get("max_grasps", 24)), need=1)
        failed = ["pick %s" % u for u, v in r["units"].items() if v["feasible"] < 1]
        return {"ok": not failed, "failed": failed, "wall_s": round(time.time() - t0, 2)}

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
        self.unit_link_ids = {u: list(cell.bodies_of[u]) for u in cell.loose}
        self.loose_body_ids = {b: u for u, bs in self.unit_link_ids.items() for b in bs}
        static = [0, m.body("fixture").id, m.body(cell.base_body).id] + [m.body(t).id for t in cell.layout["trays"]]
        self.static_env_bodies = set(static)
        # R1 counts every Panda link but the two pad bodies: every face of the finger bodies is a pad
        pad_bodies = {m.body("panda/left_finger").id: 2, m.body("panda/right_finger").id: 3}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b in pad_bodies:
                self.geom_class[g] = pad_bodies[b]
            elif b in robot_bodies:
                self.geom_class[g] = 1
            elif b in self.loose_body_ids:
                self.geom_class[g] = 4
                self.geom_part[g] = self.loose_body_ids[b]
            elif b in self.static_env_bodies:
                self.geom_class[g] = 5
        self.part_body = {u: cell.m.body(cell.unit_bodies[u]).id for u in cell.loose}
        # R3 reads the product's own joints only (the coupling's reading also sees the Panda's and the rails' limits)
        self.product_joints = set(n for u in cell.loose for n in cell.info[u]["joints"].values())
        self.sensor_force = m.sensor("hand_force").id
        self.sensor_torque = m.sensor("hand_torque").id
        self.wrist_parent = "panda/hand"
        self._f6 = np.zeros(6)
        keys = ("R1_N", "R2_N", "R2a_N", "R2b_N", "R3_Nm", "R4_m_s", "pad_N")
        self.readings = {k: 0.0 for k in keys}
        self._peak = {k: 0.0 for k in keys}
        self._r2_hist, self._r3_hist = [], []
        self._sustained = {"R2_N": 0.0, "R3_Nm": 0.0}
        self._r2_over, self._r3_over = 0, 0

    def _welded(self, part):
        return any(r["fastened"] and r["unit"] == part for r in self.sm.rec.values())

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
        """{body id: cluster root}: bodies joined by a chain of fastened mates (the palm base and its locked units)"""
        parent = {}

        def root(b):
            parent.setdefault(b, b)
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            return b
        base = self.model.body(self.cell.base_body).id
        for r in self.sm.rec.values():
            if r["fastened"] and r["unit"] is not None:
                for b in self.unit_link_ids[r["unit"]]:
                    parent[root(b)] = root(base)
        static = list(self.static_env_bodies)
        for b in static:
            parent[root(b)] = root(base)
        return {b: root(b) for b in list(parent)}

    def _watch(self, d):
        events = self._watch_base(d)
        m, mj = self.model, self.mujoco
        dmg = self.cfg.get("damage") or {}
        excl = bool(dmg.get("exclude_locked_pairs", True))
        cl = self.clusters() if excl else {}
        part_sum = {}
        for i in range(d.ncon):
            con = d.contact[i]
            c1, c2 = self.geom_class[con.geom1], self.geom_class[con.geom2]
            if not ((c1 == 4 and c2 in (4, 5)) or (c2 == 4 and c1 in (4, 5))):
                continue
            g_part = con.geom1 if c1 == 4 else con.geom2
            g_other = con.geom2 if c1 == 4 else con.geom1
            bp, bo = int(m.geom_bodyid[g_part]), int(m.geom_bodyid[g_other])
            if excl and bp in cl and bo in cl and cl[bp] == cl[bo]:
                continue
            mj.mj_contactForce(m, d, i, self._f6)
            fn = abs(float(self._f6[0]))
            other = self.geom_part.get(g_other) or m.body(bo).name
            key = (self.geom_part[g_part], other)
            part_sum[key] = part_sum.get(key, 0.0) + fn
        r2a = max(part_sum.values()) if part_sum else 0.0
        r2a_arg = max(part_sum, key=part_sum.get) if part_sum else None
        r2b, r2b_arg = 0.0, None
        if dmg.get("fastening_loads", True):
            for u, fl in self.sm.fastening_loads().items():
                if fl["N"] > r2b:
                    r2b, r2b_arg = fl["N"], (u, fl["mate"])
        self.readings["R2a_N"], self.readings["R2b_N"] = r2a, r2b
        pm = max(r2a, r2b)
        parg = r2a_arg if r2a >= r2b else r2b_arg
        self.readings["R2_N"] = pm
        self.readings["R2_pair"] = parg
        ticks = int(dmg.get("ticks", 20))
        cap = dmg.get("part_force_N")
        if cap is not None and pm > float(cap):
            self._r2_over += 1
            if self._r2_over >= ticks:
                w = "part_force:%s:%s:%.1fN" % (parg[0], parg[1], pm)
                # R2 (b): the event a program receives names the hold by a neutral word; the mate's key stays in the
                # grader's own violation record (`what`) only
                shown = w if r2a >= r2b else "part_force:%s:%s:%.1fN" % (parg[0], R2B_EVENT_LABEL, pm)
                if not any(v["what"].split(":")[:2] == w.split(":")[:2] and v["t"] > d.time - 0.5 for v in self.violations[-3:]):
                    self.violations.append({"t": float(d.time), "what": w, "force_N": pm, "rule": "R2"})
                    events.append("safety:" + shown)
        else:
            self._r2_over = 0
        st = self.cfg.get("stops") or {}
        loads = {jn: t for jn, t in self.sm.stop_loads().items() if jn in self.product_joints}
        r3, r3arg = 0.0, None
        for jn, t in loads.items():
            if t > r3:
                r3, r3arg = t, jn
        self.readings["R3_Nm"] = r3
        self.readings["R3_joint"] = r3arg
        thr = st.get("torque_Nm")
        if thr is not None and r3 > float(thr):
            self._r3_over += 1
            if self._r3_over >= int(st.get("ticks", 20)):
                w = "joint_stop:%s:%.3fNm" % (r3arg, r3)
                if not any(v["what"].split(":")[:2] == w.split(":")[:2] and v["t"] > d.time - 0.5 for v in self.violations[-3:]):
                    self.violations.append({"t": float(d.time), "what": w, "torque_Nm": r3, "rule": "R3"})
                    events.append("safety:" + w)
        else:
            self._r3_over = 0
        for k in self._peak:
            self._peak[k] = max(self._peak[k], float(self.readings.get(k, 0.0)))
        for hist, k, n in ((self._r2_hist, "R2_N", ticks), (self._r3_hist, "R3_Nm", int(st.get("ticks", 20)))):
            hist.append(float(self.readings[k]))
            if len(hist) > n:
                del hist[0]
            if len(hist) == n:
                self._sustained[k] = max(self._sustained[k], min(hist))
        for h in self.reading_hooks:
            h(self)
        return events

    def damage_summary(self):
        return {"peak": {k: round(v, 4) for k, v in self._peak.items()},
                "sustained_0.2s": {k: round(v, 4) for k, v in self._sustained.items()},
                "max_part_speed_m_s": {k: round(v, 3) for k, v in self._v_peak.items()}}

    # ------------------------------------------------------------------ the restricted API
    def describe(self):
        """the cell's own reply; its structure and field names are documented in the API text (the numbers here come
        from the config and the built cell)"""
        m = self.model
        cams = {}
        for n in self.camera_names:
            c = self.episode["config"]["cameras"][n]
            cams[n] = {"resolution": list(c["size"]), "rgb": True, "depth": bool(c.get("depth", True)),
                       "nominal_hfov_deg": round(float(c["hfov_deg"]))}
        ct = self.ctrl
        cc = self.cfg.get("cell") or {}
        fx = FX.merged(cc.get("fixture"))
        trays = TR.trays_of(cc.get("tray"))
        tray_docs = {}
        for tn, t in trays.items():
            nom = self.layout_cfg["trays"][tn]
            tray_docs[tn] = {"plate_m": list(t["plate"]),
                             "nests": [{"nest": n, "holds": self.kit.kinds[k]["name"], "x_in_tray_m": float(x), "y_in_tray_m": float(y)}
                                       for n, k, x, y in t["nests"]],
                             "nominal_xy_m": list(nom["nominal_xy"]), "nominal_yaw_deg": float(nom["yaw_deg"])}
        return {
            "cell_version": CELL_VERSION, "units": {"length": "m", "angle": "rad", "quaternion": "wxyz"},
            "robot": {"base_pos": list(self.cell.cfg["panda_pos"]), "joints": ["panda/joint%d" % (i + 1) for i in range(7)],
                      "joint_range": [[float(m.jnt_range[j][0]), float(m.jnt_range[j][1])] for j in ct.joints],
                      "home_q": [float(v) for v in ct.q_cmd], "tcp_site": "tcp", "tcp_offset_in_hand_m": [0.0, 0.0, float(self.cell.cfg["tcp_offset_m"])],
                      "gripper": dict({"max_width_m": ct.max_width, "rate_m_s": 0.05}, **self._gripper_doc()),
                      "gravity_compensated": True},
            "control": {"tick_s": self.tick_s, "joint_rate_limit_rad_s": 1.5, "cartesian_speed_limit_m_s": 0.5, "cartesian_angular_limit_rad_s": 1.5,
                        "stiffness_scale_range": [0.2, 1.0], "modes": ["joint", "cartesian"]},
            "cameras": cams, "camera_budget_frames": int(self.cfg["camera_budget_frames"]),
            "datums": hand_datums(self.cfg),
            "fixture": {"palmar_face_above_bench_m": 1e-3 * float(fx["palmar_above_bench_mm"]), "nominal_xy_m": list(fx["nominal_xy"]),
                        "nominal_yaw_deg": float(fx["nominal_yaw_deg"]), "jitter_m": float(self.layout_cfg["fixture"]["jitter_m"]),
                        "yaw_jitter_deg": float(self.layout_cfg["fixture"]["yaw_jitter_deg"])},
            "trays": tray_docs,
            "budgets": {"sim_time_cap_s": self.cfg["sim_cap_s"], "wall_clock_cap_s": self.cfg["wall_cap_s"], "silence_abort_s": self.cfg["silence_s"],
                        "max_ticks_per_step": self.cfg["max_ticks_per_step"], "camera_frames": int(self.cfg["camera_budget_frames"])},
            "fastening": {"hold_ticks": int(self.sm.hold_ticks), "rest_mm_s": float(self.sm.c["rest_mm_s"]),
                          "rest_deg_s": float(self.sm.c["rest_deg_s"]),
                          "seat_support_share_min": float(self.sm.c["part_seat"]["support_share_min"])},
            "safety": {"panda_force_N": self.cfg["safety"]["panda_force_N"], "part_speed_m_s": self.cfg["safety"]["part_speed_m_s"],
                       "part_force_N": self.cfg["damage"]["part_force_N"], "part_force_window_s": 0.01 * int(self.cfg["damage"]["ticks"]),
                       "stop_torque_Nm": self.cfg["stops"]["torque_Nm"], "stop_window_s": 0.01 * int(self.cfg["stops"]["ticks"])},
        }

    def _gripper_doc(self):
        """the grip force per finger as the cell applies it and the pads' main face as the hand model has it, read
        from the built model: the drive (`cell.gripper` kp and max_force_N) acts on the finger tendon, which gives
        each finger joint its coefficient of the drive's force and the drive half the squeeze per unit of the
        tendon's length; the main face is the largest face of a pad's collision boxes (across the closing direction,
        the finger's y)"""
        m = self.model
        gr = self.cell.cfg["gripper"]
        act = m.actuator("panda/gripper").id
        tid = int(m.actuator_trnid[act][0])
        adr, num = int(m.tendon_adr[tid]), int(m.tendon_num[tid])
        coef = sorted(float(m.wrap_prm[k]) for k in range(adr, adr + num))
        if len(coef) != 2 or abs(coef[0] - coef[1]) > 1e-12:
            raise ValueError("the finger tendon does not split the drive's force equally between the two fingers")
        kp = float(gr["kp"]) * coef[0] / 2.0
        cap = float(gr["max_force_N"]) * coef[0]
        faces = []
        for name in self.meta["pad_geoms"]:
            g = m.geom(name).id
            s = m.geom_size[g]
            faces.append((4.0 * float(s[0]) * float(s[2]), round(2.0 * float(s[0]), 6), round(2.0 * float(s[2]), 6)))
        face = max(faces)
        return {"pad_size_m": [face[1], face[2]],
                "force_model": "%g N/m x (object width - target width) per finger, capped at %g N" % (kp, cap)}

    def _session(self):
        if self.session is None:
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
                from . import render_twin
                self.session = render_twin.open_session(self)
                self.renderer = "mujoco"
            self.session_open_s = time.time() - t
            self.render_wall_s += time.time() - t
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
        out = {"layout": self.layout, "mates": [dict(g) for g in self.kit.mates()], "states": dict(self.sm.state), "score": self.sm.score()}
        return _jsonable(out)

    def get_state_privileged(self):
        d = self.data
        parts = {}
        for u in self.cell.loose:
            T = self.cell.unit_T(u)
            parts[u] = {"pos": [float(v) for v in T[:3, 3]], "quat": [float(v) for v in d.xquat[self.part_body[u]]],
                        "joints": {jn: float(v) for jn, v in self.cell.joint_q_all()[u].items()}}
        return {"sim_time": float(d.time), "parts": parts, "states": dict(self.sm.state), "records": _jsonable(self.sm.record()),
                "score": self.sm.score(), "readings": _jsonable(self.readings)}

    def reference_assemble(self):
        """The privileged reference program inside the cell, through the routed ticks and the watch"""
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
            "cell_version": CELL_VERSION, "seed": self.seed, "done_reason": self.done_reason,
            "sim_time_s": float(d.time), "wall_time_s": float(self.wall_used), "ticks": int(self.tick_count),
            "mates": _jsonable(rec), "states": dict(self.sm.state), "weights": weights, "stage_score": achieved_w / W if W else 0.0,
            "n_achieved": int(sum(1 for r in rec if r["achieved"])), "n_fastened": int(sum(1 for r in rec if r.get("fastened"))),
            "all_mates": bool(all(r["achieved"] for r in rec)),
            "off_design": [r["id"] for r in rec if r.get("fastened") and not r["achieved"]],
            "tool_blocked": {k: sorted(v) for k, v in self.sm.tool_blocks.items()},
            "violations": list(self.violations), "safety_ok": len(self.violations) == 0,
            "illegal_call_count": int(self.illegal_calls), "calls": dict(self.calls), "step_calls": int(self.step_calls),
            "lost_parts": sorted(self._lost), "dropped_parts": sorted(self._dropped),
            "events": [[t, e] for t, e in self.events_log][-400:], "coupling_events": [list(e) for e in self.sm.events],
            "layout": self.layout, "damage": self.damage_summary(),
            "camera_frames": dict(self.frames_by_camera), "camera_frames_total": int(self.frames_used),
            "render_wall_s": float(self.render_wall_s), "renderer": getattr(self, "renderer", None), "privileged_mode": bool(self.privileged),
            "lowest_part_z_m": float(self.cell.lowest_point()),
        }


def hand_datums(cfg=None):
    """the documented datums: the precedent's bench board and hand plate (cell_info.cell_datums) and the fixture board"""
    out = dict(cell_datums())
    fx = FX.merged(((cfg or {}).get("cell") or {}).get("fixture"))
    b = fx["board"]
    out.pop("fixture_board", None)
    out.pop("fixture", None)
    out["fixture_board"] = {"squares": [b["cols"], b["rows"]], "square_mm": b["square_mm"], "margin_mm": b["margin_mm"], "thickness_mm": b["thickness_mm"],
                            "centre_in_palm_frame_mm": list(b["xy_mm"]), "on": "the fixture's base plate", "note": "moves with the fixture"}
    return out


__all__ = ["HandInfoServer", "CELL_VERSION", "DEFAULT_HAND_CONFIG", "PRIVILEGED_METHODS", "hand_datums"]
