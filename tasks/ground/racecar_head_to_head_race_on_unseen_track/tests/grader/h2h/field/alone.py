"""The field alone: the three field cars of one heat run from the grid without the car.

    rec = run_alone(world, 1)            # heat 1, `alone.run_s` (300 s) of simulated time
    rec["pass"]                          # no wall contact, no contact between field cars or with anything else,
                                         # no staff reset (and no hold) of any field car

The model is the venue as the cell builds it (pushable ducts, their tracker) with the heat's three field cars, driven
by the opponents' driver unchanged, each placed at rest at its grid slot on its own line; the order of calls of a
physics step is the cell's (driver, physics step, kinematics refresh, duct tracker).

Recorded per heat: the first contact of any field car that is not with the floor (`first_contact`: its time, its
kind, and the two cars, or the car and the duct piece, or the car and the object). Recorded per car: contact steps
with a wall (a duct), with another field car and with anything else that is not the floor, and the first time of
each; staff resets; whether it was ever held; distance driven along its line; the
largest |lateral offset| of its reference point from its line; `trail_change_m`, the largest change of that offset
within any `alone.trail_window_s` (sampled every physics step), and when it happened. Per heat: for every pair of a
faster car behind a slower one, the first time the faster car's front came within `gap_keep_m` of the slower car's
rear along the faster car's line, the slower car's reference point no further from that line than the driver's
detection band plus half a car width (the start of a train).
"""

import math
import time

import mujoco
import numpy as np

from sketch.examples import constants as K
from sketch.others.params import load_params as load_others_params
from sketch.others.path import REF_X, car_extent
from sketch.others.sim import AvoidSim

from . import world as W

KIND_FLOOR, KIND_DUCT, KIND_OTHER = 1, 2, 3


def _quat_yaw(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def window_change(series, window):
    """(largest max - min of `series` over any `window` + 1 consecutive samples, index of that window's start)."""
    a = np.asarray(series, dtype=float)
    n = len(a)
    if n == 0:
        return 0.0, 0
    w = int(min(max(window, 0) + 1, n))
    k = int(math.floor(math.log2(w)))
    mx, mn = [a], [a]
    for j in range(1, k + 1):
        step = 1 << (j - 1)
        mx.append(np.maximum(mx[-1][:-step], mx[-1][step:]))
        mn.append(np.minimum(mn[-1][:-step], mn[-1][step:]))
    span = 1 << k
    m = n - w + 1
    hi = np.maximum(mx[k][:m], mx[k][w - span:w - span + m])
    lo = np.minimum(mn[k][:m], mn[k][w - span:w - span + m])
    r = hi - lo
    i = int(np.argmax(r))
    return float(r[i]), i


class FieldAlone(object):
    """One heat's field without the car (see the module text)."""

    def __init__(self, world, h, params=None, others=None):
        self.P = W.load_params(params)
        self.op = load_others_params(others)
        self.world, self.h = world, W.heat_index(h)
        self.view = W.heat_view(world, self.h)
        self.sim = AvoidSim(self.view, mode="dynamic", ego=False, pushable=True, monitor=False, params=self.op)
        m, d = self.sim.model, self.sim.data
        self.slow = self.sim.slow
        if [c.mode for c in self.slow.cars] != ["dynamic"] * len(self.slow.cars):
            raise RuntimeError("the field cars are not dynamic")
        for k, c in enumerate(self.slow.cars, start=1):
            self.slow._place_on_path(c, W.slot_path_station(world, self.h, k))
        self.slow.force_plan = True
        mujoco.mj_forward(m, d)
        self.dt = float(m.opt.timestep)
        self.N = int(round(K.TICK_S / self.dt))
        owner = np.full(m.ngeom, -1, dtype=np.int64)
        kind = np.zeros(m.ngeom, dtype=np.int64)
        roots = {c.body: i for i, c in enumerate(self.slow.cars)}
        for g in range(m.ngeom):
            r = int(m.body_rootid[int(m.geom_bodyid[g])])
            if r in roots:
                owner[g] = roots[r]
            rule = K.geom_rule(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            if rule is not None and rule[1] == "floor":
                kind[g] = KIND_FLOOR
            elif rule is not None and rule[1] == "duct":
                kind[g] = KIND_DUCT
            else:
                kind[g] = KIND_OTHER
        self.owner, self.kind = owner, kind
        self.gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self.first_contact = None
        n = len(self.slow.cars)
        self.stats = [{"wall_steps": 0, "car_steps": 0, "other_steps": 0, "first_wall_t_s": None, "first_car_t_s": None,
                       "first_other_t_s": None, "held": False, "deepest_m": 0.0} for _ in range(n)]
        self.hint = [None] * n
        self.lat = [[] for _ in range(n)]
        x0, x1, hw = car_extent(self.op["tail_box_centre_m"])
        self.front = x1 - REF_X
        self.rear = REF_X - x0
        self.hw = hw
        self.band = hw + float(self.op["follower"]["band_margin_m"])
        self.trains = {}
        self.pair_hint = {}
        self.steps = 0

    # ------------------------------------------------------------------ per physics step
    def _ref(self, c):
        q = self.sim.data.qpos[c.qa:c.qa + 7]
        yaw = _quat_yaw(q[3:7])
        return float(q[0]) + REF_X * math.cos(yaw), float(q[1]) + REF_X * math.sin(yaw)

    def _offsets(self):
        for i, c in enumerate(self.slow.cars):
            rx, ry = self._ref(c)
            if self.hint[i] is None:
                s, lat, d2 = c.path.project(rx, ry)
            else:
                s, lat, d2 = c.path.project(rx, ry, s_hint=self.hint[i], back_m=1.0, ahead_m=2.0)
                if d2 > 0.36:
                    s, lat, d2 = c.path.project(rx, ry)
            self.hint[i] = s
            self.lat[i].append(lat)

    def _contacts(self, t):
        d = self.sim.data
        n = d.ncon
        if not n:
            return
        g = d.contact.geom[:n]
        dist = d.contact.dist[:n]
        oa, ob = self.owner[g[:, 0]], self.owner[g[:, 1]]
        sel = np.nonzero((dist <= 0.0) & ((oa >= 0) | (ob >= 0)))[0]
        if not len(sel):
            return
        seen = set()
        for k in sel:
            a, b = int(g[k, 0]), int(g[k, 1])
            for me, other in ((int(oa[k]), b), (int(ob[k]), a)):
                if me < 0:
                    continue
                oo = int(self.owner[other])
                if oo == me:
                    continue
                if oo >= 0:
                    what = "car"
                elif self.kind[other] == KIND_FLOOR:
                    continue
                elif self.kind[other] == KIND_DUCT:
                    what = "wall"
                else:
                    what = "other"
                st = self.stats[me]
                st["deepest_m"] = max(st["deepest_m"], float(-dist[k]))
                if self.first_contact is None:
                    self.first_contact = self._what_met(t, me, other, oo, what)
                if (me, what) in seen:
                    continue
                seen.add((me, what))
                st[what + "_steps"] += 1
                if st["first_%s_t_s" % what] is None:
                    st["first_%s_t_s" % what] = round(t, 4)

    def _what_met(self, t, me, other, oo, what):
        cars = self.slow.cars
        out = {"t_s": round(t, 4), "kind": what}
        if what == "car":
            out["cars"] = sorted([cars[me].id, cars[oo].id])
        elif what == "wall":
            parts = self.gname[other].split("/")
            out["car"], out["piece"] = cars[me].id, parts[1] if len(parts) > 1 else self.gname[other]
        else:
            out["car"], out["geom"] = cars[me].id, self.gname[other]
        return out

    def _trains(self, t):
        cars = self.slow.cars
        for i, ci in enumerate(cars):
            for j, cj in enumerate(cars):
                if i == j or ci.factor <= cj.factor or (i, j) in self.trains:
                    continue
                rx, ry = self._ref(cj)
                hint = self.pair_hint.get((i, j))
                if hint is None:
                    s, lat, d2 = ci.path.project(rx, ry)
                else:
                    s, lat, d2 = ci.path.project(rx, ry, s_hint=hint, back_m=1.0, ahead_m=3.0)
                    if d2 > 0.36:
                        s, lat, d2 = ci.path.project(rx, ry)
                self.pair_hint[(i, j)] = s
                rel = (s - self.hint[i]) % ci.path.length
                if rel > 0.5 * ci.path.length or abs(lat) > self.band + self.hw:
                    continue
                if rel - self.front - self.rear <= ci.gap_keep:
                    self.trains[(i, j)] = round(t, 4)

    def run(self, run_s=None):
        run_s = float(self.P["alone"]["run_s"] if run_s is None else run_s)
        ticks = int(round(run_s / K.TICK_S))
        t0 = time.perf_counter()
        self._offsets()
        for tick in range(ticks):
            for _ in range(self.N):
                self.sim.physics_step()
                self.steps += 1
                t = self.steps * self.dt
                self._offsets()
                self._contacts(t)
            for i, c in enumerate(self.slow.cars):
                if c.held_until is not None:
                    self.stats[i]["held"] = True
            self._trains((tick + 1) * K.TICK_S)
            if not np.isfinite(self.sim.data.qpos).all():
                raise RuntimeError("the physics state is not finite at tick %d" % tick)
        return self.record(run_s, time.perf_counter() - t0)

    def record(self, run_s, wall_s):
        win = int(round(float(self.P["alone"]["trail_window_s"]) / self.dt))
        cars = []
        why = []
        for i, (c, st, sd) in enumerate(zip(self.slow.cars, self.stats, self.slow.stats())):
            lat = np.asarray(self.lat[i])
            ch, at = window_change(lat, win)
            row = {"id": c.id, "speed_factor": c.factor, "resets": int(sd["resets"]), "distance_m": sd["distance_m"],
                   "max_abs_lat_m": round(float(np.abs(lat).max()), 6), "trail_change_m": round(ch, 6),
                   "trail_change_at_s": round(at * self.dt, 4)}
            row.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in st.items()})
            cars.append(row)
            for what in ("wall", "car", "other"):
                if st[what + "_steps"]:
                    why.append("%s: %s contact from %.2f s" % (c.id, what, st["first_%s_t_s" % what]))
            if row["resets"] or st["held"]:
                why.append("%s: %d staff reset(s)" % (c.id, row["resets"]))
        trains = [{"follower": self.slow.cars[i].id, "leader": self.slow.cars[j].id, "t_s": t}
                  for (i, j), t in sorted(self.trains.items(), key=lambda kv: kv[1])]
        return {"world_id": self.world["world_id"], "layout_digest": self.world["layout_digest"], "heat": self.h,
                "tier": self.world["truth"]["head_to_head"]["tier"], "run_s": run_s, "physics_steps": self.steps,
                "cars": cars, "trains": trains, "first_train_t_s": trains[0]["t_s"] if trains else None,
                "first_contact": self.first_contact,
                "trail_change_m": max(r["trail_change_m"] for r in cars), "pass": not why, "why_not": why,
                "wall_s": round(wall_s, 3)}


def run_alone(world, h, run_s=None, params=None, others=None):
    return FieldAlone(world, h, params=params, others=others).run(run_s)
