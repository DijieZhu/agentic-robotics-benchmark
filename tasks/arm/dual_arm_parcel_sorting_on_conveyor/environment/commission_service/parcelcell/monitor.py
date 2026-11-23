"""Contact instrumentation at physics-step resolution (privileged side).

``ContactMonitor`` looks at the contact set after every 2 ms physics step
and keeps, per 20 ms control tick, the peak normal force of these classes:

* ``structure``  a robot collision geom on station structure (rails, lip,
                 conveyor shell and legs: ``station.STRUCTURE``);
* ``belt``       a robot collision geom on the moving belt surface;
* ``arm_arm``    a left-arm geom on a right-arm geom;
* ``arm_base``   an arm geom on the column / pedestal (base link);
* ``palm_L`` / ``palm_R``  the summed normal force of one gripper side on
                 parcels (the squeeze a palm applies).

Per tick the peaks are appended to lists (only ticks with a non-zero
value), so percentiles over a run can be taken. The largest single events
of the structure / belt / arm classes are kept with their geom names and
time, for diagnosis.

Never hand any of this to a controller under test.
"""

from __future__ import annotations

import mujoco
import numpy as np

CLASSES = ("structure", "belt", "arm_arm", "arm_base", "palm_L", "palm_R")


class ContactMonitor:
    def __init__(self, cell):
        self.cell = cell
        m = cell.model
        ng = m.ngeom
        self._is_robot = np.zeros(ng, bool)
        self._is_robot[cell.robot_geoms] = True
        self._is_struct = np.zeros(ng, bool)
        self._is_struct[cell.structure_geoms] = True
        self._belt = int(cell._belt_surface)
        # robot geom side: 0 column / base, 1 left arm, 2 right arm
        side = np.full(ng, -1, int)
        lroot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                  "Left_Shoulder_Base_Link")
        rroot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                  "Right_Shoulder_Base_Link")

        def root_of(b):
            while b > 0:
                if b == lroot:
                    return 1
                if b == rroot:
                    return 2
                b = m.body_parentid[b]
            return 0
        for g in cell.robot_geoms:
            side[g] = root_of(int(m.geom_bodyid[g]))
        self._side = side
        self._slot = cell.parcel_geom_slot
        self._gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or
                       f"geom{g}" for g in range(ng)]
        self._f6 = np.zeros(6)
        self.reset()

    def reset(self):
        self.ticks = {k: [] for k in CLASSES}       # per-tick peaks (> 0)
        self.tick_t = {k: [] for k in CLASSES}      # tick times of the peaks
        self.top = []         # (force, class, robot geom, other geom, t)
        self.peak = {k: 0.0 for k in CLASSES}
        self._cur = {k: 0.0 for k in CLASSES}
        self._ev = []

    def begin_tick(self):
        for k in CLASSES:
            self._cur[k] = 0.0
        self._ev = []

    def substep(self):
        c = self.cell
        m, d = c.model, c.data
        n = d.ncon
        if n == 0:
            return
        g = np.asarray(d.contact.geom[:n])
        r = self._is_robot[g]
        anyr = r[:, 0] | r[:, 1]
        if not anyr.any():
            return
        idx = np.nonzero(anyr)[0]
        sq = {1: 0.0, 2: 0.0}
        f6 = self._f6
        for k in idx:
            g1, g2 = int(g[k, 0]), int(g[k, 1])
            r1, r2 = r[k, 0], r[k, 1]
            if r1 and r2:
                s1, s2 = self._side[g1], self._side[g2]
                if s1 == s2:
                    continue
                cls = "arm_arm" if (s1 > 0 and s2 > 0) else "arm_base"
                rg, og = g1, g2
            else:
                rg, og = (g1, g2) if r1 else (g2, g1)
                if self._slot[og] >= 0:
                    cls = "palm"
                elif og == self._belt:
                    cls = "belt"
                elif self._is_struct[og]:
                    cls = "structure"
                else:
                    continue
            con = d.contact[k]
            if con.exclude != 0 or con.efc_address < 0:
                continue
            mujoco.mj_contactForce(m, d, int(k), f6)
            fn = float(f6[0])
            if cls == "palm":
                s = self._side[rg]
                if s in sq:
                    sq[s] += fn
                continue
            if fn > self._cur[cls]:
                self._cur[cls] = fn
            if fn > 2.0:
                self._ev.append((fn, cls, self._gname[rg], self._gname[og]))
        if sq[1] > self._cur["palm_L"]:
            self._cur["palm_L"] = sq[1]
        if sq[2] > self._cur["palm_R"]:
            self._cur["palm_R"] = sq[2]

    def end_tick(self, t):
        for k in CLASSES:
            v = self._cur[k]
            if v > 0.0:
                self.ticks[k].append(v)
                self.tick_t[k].append(t)
                if v > self.peak[k]:
                    self.peak[k] = v
        if self._ev:
            fn, cls, a, b = max(self._ev)
            self.top.append((round(fn, 2), cls, a, b, round(t, 3)))
            if len(self.top) > 400:
                self.top.sort(reverse=True)
                del self.top[200:]

    def summary(self) -> dict:
        out = {}
        for k in CLASSES:
            v = np.asarray(self.ticks[k], float)
            out[k] = dict(
                ticks=int(len(v)), peak=float(v.max()) if len(v) else 0.0,
                p99=float(np.percentile(v, 99)) if len(v) else 0.0,
                p50=float(np.percentile(v, 50)) if len(v) else 0.0)
        out["top"] = sorted(self.top, reverse=True)[:12]
        return out
