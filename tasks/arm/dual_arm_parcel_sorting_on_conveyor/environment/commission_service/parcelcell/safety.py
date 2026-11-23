"""The safety watch: every tick passes through it, whoever commands the robot
(the program over the wire or the reference inside the cell), because it is
a tick listener and the command gate of the world itself.

Contact classes (the normal force of each robot contact, read at the tick's
last physics step, per pair of bodies):
* ``structure``: a robot geom against any station part that is not a parcel
  and not the moving belt surface (conveyor shell and legs, rails, lip,
  floor; later station hardware the same way);
* ``self``: one arm against the other arm or against the robot's own column;
* ``belt``: a robot geom against the moving belt surface.
Thresholds: ``structure_N``, ``self_N``, ``belt_N``.

Protective stop: a pair above its threshold for ``press_ticks`` consecutive
ticks. The joint targets freeze at the measured posture (clipped to the
ranges), the grippers keep their command, motion commands are refused for
``stop_lock_s`` (the world keeps running, the belt too), the stop is counted.
A pair that was pressing when a stop came, or during its lock, counts again
only once its force has fallen below the threshold (the program gets the
chance to back off).

Lock-out (the rest of the shift): any pair above ``damage_N`` for
``press_ticks`` consecutive ticks, or the stop after ``max_stops`` stops (the
sixth). The robot holds its posture, every motion command is refused, the
world keeps running (parcels that cross later are still judged).

Parcel damage: the summed normal force of every robot contact on one parcel
above ``squeeze_N`` for ``squeeze_ticks`` consecutive ticks; that parcel is
damaged (judged lost).

What the program learns: ``stop`` in its state (active, until, count,
locked_out, the robot link of the latest stop). A stop names the robot's link
only, never what it pressed on; nothing about parcel damage.
"""

from __future__ import annotations

import mujoco
import numpy as np

__all__ = ["SafetyWatch", "CLASSES"]

CLASSES = ("structure", "self", "belt")


def _side_of(name: str) -> int:
    if name.startswith(("Left_", "L_")):
        return 1
    if name.startswith(("Right_", "R_")):
        return 2
    return 3                                   # the base, column and head


class SafetyWatch:
    def __init__(self, world, safety: dict, log=None):
        self.world = world
        self.cfg = dict(safety)
        self.log = log or (lambda s: None)
        m = world.model
        self._f6 = np.zeros(6)
        self.side = np.zeros(m.ngeom, np.int8)       # 0 not robot, 1 left, 2 right, 3 base
        for g in world.robot_geoms:
            self.side[g] = _side_of(m.body(int(m.geom_bodyid[g])).name)
        coll = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        self.structure = coll & (self.side == 0) & (world.parcel_geom_slot < 0)
        self.structure[world._belt_surface] = False
        self.thr = {"structure": float(self.cfg["structure_N"]),
                    "self": float(self.cfg["self_N"]), "belt": float(self.cfg["belt_N"])}
        self.on_halt = []                            # callables f(q_frozen)
        self.reset()

    def reset(self) -> None:
        self._press = {}
        self._latched = set()
        self._squeeze = {}
        self.stop_until = None
        self.locked_out = False
        self.lockout_reason = None
        self.stops = []
        self.violations = []
        self.damaged = {}                 # slot -> {"t", "peak_N", "reason"}
        self.peak = {c: 0.0 for c in CLASSES}
        self.peak_link = {c: None for c in CLASSES}
        self.peak_squeeze = {}            # slot -> N
        self.frozen_q = None
        self.frozen_grip = None
        self.last_link = None
        self.refused_ticks = 0
        self.events = []                  # (t, text) for the records

    # ---------------------------------------------------------------- state
    @property
    def stopped(self) -> bool:
        return self.locked_out or self.stop_until is not None

    def state(self) -> dict:
        return {"active": bool(self.stopped),
                "until": None if self.stop_until is None else round(self.stop_until, 4),
                "count": len(self.stops), "locked_out": bool(self.locked_out),
                "link": self.last_link}

    # ---------------------------------------------------------------- gate
    def gate(self, world, q, grip):
        if self.stopped and self.frozen_q is not None:
            self.refused_ticks += 1
            return self.frozen_q.copy(), self.frozen_grip.copy()
        return q, grip

    def _halt(self) -> None:
        w = self.world
        js = w.joint_state()
        q = np.clip(js.q, w._ctrl_lo, w._ctrl_hi)
        self.frozen_q = q.copy()
        self.frozen_grip = np.asarray(w._grip_cmd, float).copy()
        w._q_applied = q.copy()
        w.data.ctrl[w._arm_aid] = q
        for fn in self.on_halt:
            fn(q.copy())

    # ---------------------------------------------------------------- tick
    def _contacts(self):
        """{(class, key): (force, robot link name)} and {slot: summed force}."""
        w = self.world
        m, d = w.model, w.data
        now, squeeze = {}, {}
        n = d.ncon
        if n == 0:
            return now, squeeze
        geoms = np.asarray(d.contact.geom[:n])
        side = self.side[geoms]
        rob = side > 0
        cand = np.nonzero(rob[:, 0] | rob[:, 1])[0]
        for k in cand:
            g1, g2 = int(geoms[k, 0]), int(geoms[k, 1])
            s1, s2 = int(side[k, 0]), int(side[k, 1])
            con = d.contact[k]
            if con.exclude != 0 or con.efc_address < 0:
                continue
            if s1 and s2:
                if s1 == s2 or (s1 == 3 and s2 == 3):
                    continue                          # within one arm: not watched
                cls = "self"
                b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
                key = (min(b1, b2), max(b1, b2))
                arm = [b for b, s in ((b1, s1), (b2, s2)) if s != 3]
                link = "/".join(sorted(m.body(b).name for b in arm))
            else:
                rg, og = (g1, g2) if s1 else (g2, g1)
                slot = int(w.parcel_geom_slot[og])
                mujoco.mj_contactForce(m, d, k, self._f6)
                fn = abs(float(self._f6[0]))
                if slot >= 0:
                    squeeze[slot] = squeeze.get(slot, 0.0) + fn
                    continue
                if og == w._belt_surface:
                    cls = "belt"
                elif self.structure[og]:
                    cls = "structure"
                else:
                    continue
                rb = int(m.geom_bodyid[rg])
                key = (rb, int(m.geom_bodyid[og]))
                link = m.body(rb).name
                kk = (cls, key)
                f0 = now.get(kk, (0.0, link))[0]
                now[kk] = (max(f0, fn), link)
                continue
            mujoco.mj_contactForce(m, d, k, self._f6)
            fn = abs(float(self._f6[0]))
            kk = (cls, key)
            f0 = now.get(kk, (0.0, link))[0]
            now[kk] = (max(f0, fn), link)
        return now, squeeze

    def on_tick(self, world) -> None:
        S = self.cfg
        t = float(world.time)
        n_ticks = int(S["press_ticks"])
        thr_x = float(S["damage_N"])
        if self.stop_until is not None and t >= self.stop_until - 1e-9:
            self.stop_until = None
            self.events.append((t, "protective_stop_cleared"))
        now, squeeze = self._contacts()
        # parcel damage (squeeze)
        sq_n = int(S["squeeze_ticks"])
        sq_thr = float(S["squeeze_N"])
        sq = {}
        for slot, f in squeeze.items():
            if f > self.peak_squeeze.get(slot, 0.0):
                self.peak_squeeze[slot] = f
            if f > sq_thr:
                cnt, pk = self._squeeze.get(slot, (0, 0.0))
                sq[slot] = (cnt + 1, max(pk, f))
                if cnt + 1 >= sq_n and slot not in self.damaged:
                    self.damaged[slot] = {"t": round(t, 4), "peak_N": round(max(pk, f), 2),
                                          "reason": "squeeze"}
                    self.events.append((t, "parcel_damaged:%d" % slot))
        self._squeeze = sq
        # robot contacts
        press = {}
        for kk, (fn, link) in now.items():
            cls = kk[0]
            if fn > self.peak[cls]:
                self.peak[cls], self.peak_link[cls] = fn, link
            if fn > self.thr[cls]:
                cnt, pk, cx = self._press.get(kk, (0, 0.0, 0))
                press[kk] = (cnt + 1, max(pk, fn), cx + 1 if fn > thr_x else 0)
        self._press = press
        self._latched = set(k for k in self._latched if k in press)
        stopped = self.stopped
        if stopped:
            self._latched |= set(press)
        for kk in sorted(press, key=lambda x: (x[0], x[1])):
            cnt, pk, cx = press[kk]
            link = now[kk][1]
            if cx >= n_ticks and not self.locked_out:
                self.violations.append({"t": round(t, 4), "what": "damage_level", "class": kk[0],
                                        "link": link, "force_N": round(pk, 2)})
                self.events.append((t, "locked_out:damage:%s" % link))
                self.locked_out = True
                self.lockout_reason = "damage_level"
                self.stop_until = None
                self.last_link = link
                self._halt()
                stopped = True
                self._latched |= set(press)
            if cnt >= n_ticks and kk not in self._latched and not stopped:
                self.stops.append({"t": round(t, 4), "class": kk[0], "link": link,
                                   "force_N": round(pk, 2)})
                self.events.append((t, "protective_stop:%s" % link))
                self.last_link = link
                self._halt()
                stopped = True
                self._latched |= set(press)
                if len(self.stops) > int(S["max_stops"]):
                    self.locked_out = True
                    self.lockout_reason = "stops"
                    self.violations.append({"t": round(t, 4), "what": "stops", "class": kk[0],
                                            "link": link, "count": len(self.stops)})
                    self.events.append((t, "locked_out:stops"))
                else:
                    self.stop_until = t + float(S["stop_lock_s"])

    def summary(self) -> dict:
        return {"protective_stops": list(self.stops), "protective_stop_count": len(self.stops),
                "locked_out": bool(self.locked_out), "lockout_reason": self.lockout_reason,
                "violations": list(self.violations),
                "peak_force_N": {c: round(v, 2) for c, v in self.peak.items()},
                "peak_force_link": dict(self.peak_link),
                "refused_command_ticks": int(self.refused_ticks)}
