"""The safety monitor of the humanoid assembly cell, evaluated every control tick.

New file of this task. It takes the place of the precedent's cell watch (arm/assembly_car/cell.py `watch_tick`,
vendored, unchanged): the same per-tick reading of MuJoCo's contacts after the tick's mj_forward and the same event
thinning (at most one event per rule and key per `event_repeat_s`), with this task's four rules and
their definitions. Every number comes from the configuration's "safety" block (cell_config.json; the final values
follow max(proposal, min(cap, ceil(1.5 x reference maximum))), applied in that file, not here).

  R1 robot_contact  a robot collision geom presses on any body that is not the robot: the normal forces of that
                    (robot body, other body) pair summed over the pair's contacts, above robot_force_N on any tick. One
                    exemption: the gripper's pads on a body of a module (the grip). A pad on the base, a fixture, the
                    floor or anything else counts like any other geom of its link.
  R2 part_force     a module presses on one other body: the normal forces summed per (module, body pressed on) pair,
                    above part_force_N on part_ticks consecutive ticks (0.2 s), each pair with its own count. Not
                    counted: contacts with the gripper's pads; a robot link's contacts (R1 has them); contacts inside
                    one module; contacts between bodies joined by a lock (the lock clusters from `locks()`: the base and
                    every module locked to it, transitively). Counted: a module resting in its rack, a module on the
                    floor, a module on the base or on another module not joined to it by a lock.
  R4 part_speed     a module's centre-of-mass speed (all its bodies, mass weighted) above part_speed_m_s.
  R5 floor_contact  any body of a module touching the floor: the module is scrapped (`on_scrap(module, t, why)`,
                    the coupling's scrap()) and it is a violation; once per module.

A violation is an event "safety:<rule>:<what>" (the step() events), recorded with its time and value; each rule is
COUNTED once per episode (`counted[rule]` is 0 or 1), however often it fires (an episode with a
violation keeps scoring.safety_violation_factor of its score, one factor per episode). The maxima of every reading are
kept for the metrics (the watch's maximum is the number to quote).

The data the monitor reads: the cell's index (scene.CellIndex: each geom's class and owner, each module's bodies, the
floor's geoms) and two plug-ins: `locks()` -> iterable of (owner, owner) pairs currently joined by a lock (owners are
module ids and the base's id; default: none), `on_scrap(module, t, why)` (default: the monitor's own record only).
"""
import math

import numpy as np

from .scene import CLS

__all__ = ["SafetyMonitor", "RULES"]

RULES = {"R1": "robot_contact", "R2": "part_force", "R4": "part_speed", "R5": "floor_contact"}


class _Clusters(object):
    def __init__(self, edges):
        self.p = {}
        for a, b in edges:
            self.union(a, b)

    def find(self, a):
        p = self.p
        p.setdefault(a, a)
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def joined(self, a, b):
        if a is None or b is None:
            return False
        if a == b:
            return True
        if a not in self.p or b not in self.p:
            return False
        return self.find(a) == self.find(b)


class SafetyMonitor(object):
    def __init__(self, model, data, index, safety, tick_s, locks=None, on_scrap=None):
        import mujoco
        self.mj = mujoco
        self.m, self.d, self.ix = model, data, index
        self.S = dict(safety)
        self.tick_s = float(tick_s)
        self.locks = locks or (lambda: ())
        self.on_scrap = on_scrap
        self.robot_force_N = float(self.S["robot_force_N"])
        self.part_force_N = float(self.S["part_force_N"])
        self.part_ticks = int(self.S["part_ticks"])
        self.part_speed = float(self.S["part_speed_m_s"])
        self.repeat_s = float(self.S["event_repeat_s"])
        self.gclass = np.asarray(index.geom_class, int)
        self.gowner = list(index.geom_owner)
        self.bowner = list(index.body_owner)
        self.floor = set(index.floor_geoms)
        self.mod_bodies = {k: np.asarray(v, int) for k, v in index.module_bodies.items()}
        self.mod_mass = {k: np.asarray([model.body_mass[b] for b in v], float) for k, v in index.module_bodies.items()}
        self.root_of = np.asarray(model.body_rootid, int)
        self._f6 = np.zeros(6)
        self.reset()

    def reset(self):
        self.violations = []                 # every occurrence: {"t", "rule", "what", "value", "key"}
        self.events = []                     # (t, "safety:...") as emitted
        self.counted = {r: 0 for r in RULES}
        self.scrapped = {}                   # module -> {"t", "why"}
        self._last_event = {}                # (rule, key) -> t of its last event
        self._run = {}                       # R2 pair -> consecutive ticks above
        self._r1_over = set()
        self._r4_over = set()
        self.ticks = 0
        self.max = {"robot_force_N": 0.0, "robot_force_pair": None, "part_force_instant_N": 0.0, "part_force_instant_pair": None,
                    "part_force_sustained_N": 0.0, "part_force_sustained_pair": None, "part_speed_m_s": {},
                    "robot_contact_single_N": 0.0}
        self._hist = {}                      # R2 pair -> recent sums (for the sustained maximum)

    # ------------------------------------------------------------------ readings
    def module_speeds(self):
        """{module: centre-of-mass speed m/s} from the bodies' COM velocities (mass weighted)"""
        d = self.d
        out = {}
        for mid, bl in self.mod_bodies.items():
            if len(bl) == 0:
                continue
            cv = d.cvel[bl]                                   # (n, 6): rot, lin at the subtree COM of each root
            r = d.xipos[bl] - d.subtree_com[self.root_of[bl]]
            v = cv[:, 3:] + np.cross(cv[:, :3], r)
            w = self.mod_mass[mid]
            vc = (w[:, None] * v).sum(0) / max(float(w.sum()), 1e-12)
            out[mid] = float(np.linalg.norm(vc))
        return out

    def pair_forces(self):
        """(r1 {(robot body, other body): N}, r2 {(module, other body): N}, floor {module: body}, single max N) of the
        current contacts (normal forces, after the exclusions of the module text)"""
        m, d, mj = self.m, self.d, self.mj
        cl = _Clusters(self.locks())
        r1, r2, floor = {}, {}, {}
        single = 0.0
        R, P, M, NONE = CLS["robot"], CLS["pad"], CLS["module"], CLS["none"]
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            c1, c2 = int(self.gclass[g1]), int(self.gclass[g2])
            if c1 == NONE or c2 == NONE:
                continue
            if c1 in (R, P) and c2 in (R, P):
                continue                                      # the robot on itself
            if (c1 == P and c2 == M) or (c2 == P and c1 == M):
                continue                                      # a pad on a module: the grip (never R1, never an R2 pair)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if c1 in (R, P) or c2 in (R, P):
                # a link of the arm or of its gripper, a pad included, on any body that is not part of them
                mj.mj_contactForce(m, d, i, self._f6)
                fn = abs(float(self._f6[0]))
                rb, ob = (b1, b2) if c1 in (R, P) else (b2, b1)
                r1[(rb, ob)] = r1.get((rb, ob), 0.0) + fn
                single = max(single, fn)
                continue
            for gm, gm_c, go, bo in ((g1, c1, g2, b2), (g2, c2, g1, b1)):
                if gm_c != M:
                    continue
                mod = self.gowner[gm]
                if go in self.floor:
                    floor.setdefault(mod, int(m.geom_bodyid[gm]))
                other = self.bowner[bo]
                if other == mod:
                    continue                                  # inside one module
                if cl.joined(mod, other):
                    continue                                  # joined by a lock
                mj.mj_contactForce(m, d, i, self._f6)
                fn = abs(float(self._f6[0]))
                r2[(mod, bo)] = r2.get((mod, bo), 0.0) + fn
        return r1, r2, floor, single

    # ------------------------------------------------------------------ the tick
    def _emit(self, rule, key, what, value, t, events):
        self.violations.append({"t": t, "rule": rule, "what": what, "value": value, "key": key})
        self.counted[rule] = 1
        last = self._last_event.get((rule, key))
        if last is None or t - last >= self.repeat_s - 1e-9:
            self._last_event[(rule, key)] = t
            ev = "safety:%s:%s" % (RULES[rule], what)
            self.events.append((t, ev))
            events.append(ev)

    def tick(self):
        """evaluate the four rules on the present state (after the tick's mj_forward); returns the tick's events"""
        m, d = self.m, self.d
        t = float(d.time)
        self.ticks += 1
        events = []
        r1, r2, floor, single = self.pair_forces()
        bn = lambda b: m.body(int(b)).name           # noqa: E731
        # R1
        self.max["robot_contact_single_N"] = max(self.max["robot_contact_single_N"], single)
        over = set()
        for (rb, ob), fn in r1.items():
            if fn > self.max["robot_force_N"]:
                self.max["robot_force_N"], self.max["robot_force_pair"] = fn, "%s:%s" % (bn(rb), bn(ob))
            if fn > self.robot_force_N:
                over.add((rb, ob))
                self._emit("R1", (rb, ob), "%s:%s:%.0fN" % (bn(rb), bn(ob), fn), fn, t, events)
        self._r1_over = over
        # R2
        seen = set()
        for key, fn in r2.items():
            seen.add(key)
            if fn > self.max["part_force_instant_N"]:
                self.max["part_force_instant_N"], self.max["part_force_instant_pair"] = fn, "%s:%s" % (key[0], bn(key[1]))
            h = self._hist.setdefault(key, [])
            h.append(fn)
            if len(h) > self.part_ticks:
                del h[0]
            if len(h) == self.part_ticks and min(h) > self.max["part_force_sustained_N"]:
                self.max["part_force_sustained_N"], self.max["part_force_sustained_pair"] = min(h), "%s:%s" % (key[0], bn(key[1]))
            if fn > self.part_force_N:
                self._run[key] = self._run.get(key, 0) + 1
                if self._run[key] >= self.part_ticks:
                    self._emit("R2", key, "%s:%s:%.0fN" % (key[0], bn(key[1]), fn), fn, t, events)
            else:
                self._run[key] = 0
        for key in list(self._run):
            if key not in seen:
                self._run[key] = 0
        for key in list(self._hist):
            if key not in seen:
                self._hist[key] = []
        # R4
        sp = self.module_speeds()
        over = set()
        for mid, v in sp.items():
            if v > self.max["part_speed_m_s"].get(mid, 0.0):
                self.max["part_speed_m_s"][mid] = v
            if v > self.part_speed:
                over.add(mid)
                self._emit("R4", mid, "%s:%.2fm/s" % (mid, v), v, t, events)
        self._r4_over = over
        # R5
        for mid, b in sorted(floor.items()):
            if mid in self.scrapped:
                continue
            why = "touched the floor (%s)" % bn(b)
            self.scrapped[mid] = {"t": t, "why": why}
            if self.on_scrap is not None:
                self.on_scrap(mid, t, why)
            self._emit("R5", mid, "%s:%s" % (mid, bn(b)), 1.0, t, events)
        return events

    # ------------------------------------------------------------------ summary
    def n_counted(self):
        return int(sum(self.counted.values()))

    def summary(self):
        return {"ticks": int(self.ticks), "counted": dict(self.counted), "n_counted": self.n_counted(),
                "safety_ok": self.n_counted() == 0, "n_occurrences": len(self.violations),
                "events": [[t, e] for t, e in self.events][-400:],
                "violations": [{k: v for k, v in x.items() if k != "key"} for x in self.violations][-400:],
                "scrapped": dict(self.scrapped),
                "max": {k: (round(v, 4) if isinstance(v, float) else ({kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else v))
                        for k, v in self.max.items()},
                "numbers": {"robot_force_N": self.robot_force_N, "part_force_N": self.part_force_N, "part_ticks": self.part_ticks,
                            "part_force_window_s": round(self.part_ticks * self.tick_s, 6), "part_speed_m_s": self.part_speed,
                            "event_repeat_s": self.repeat_s}}
