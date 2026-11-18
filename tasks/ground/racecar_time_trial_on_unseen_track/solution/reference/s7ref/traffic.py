"""Slow cars in the dynamic phase of the avoidance test: passing where the track is wide enough, following
elsewhere (numpy only).

    traffic = Traffic(world, plan, line, profile, bounds, params)
    cap, path = traffic.update(truth, tick)     # per tick: a speed cap (or None) and a pass / return path (or None)

What the program knows (privileged): every slow car's path (the trajectory of its reference point, with its
reference speed) and speed factor from the world file, and each slow car's true pose and speed every tick.

- A pass. When a slow car is within `look_min_m` (or `look_s` seconds) ahead, the ego's arrival is predicted along
  its line at its profile speed and the slow car's along its path at speed_factor x reference speed (0.3 m ahead,
  as its follower does). The requirement covers the slow car's path from where it is now (less `behind_m`) to
  where the ego clears it (`ahead_m` ahead of it, plus `clear_extra_m`): a slower slow car is cleared earlier,
  inside that stretch, so the prediction's timing does not matter (packs, one slow car slowed by another). Over
  the stretch the ego needs a lateral offset at least `sep_m` + `dev_margin_m` from the slow car's path (its
  present tracking error added, fading over 1 m), on the left or the right, inside the ego's limits (walls,
  boxes). The side that needs the smaller deviation from the racing line is taken; the pass path starts at the
  car's present offset, ramps to the target (lateral acceleration `ramp_lat_frac` of the scaled grip), follows the
  slow car's path at that separation, and ramps back; its speed profile is recomputed from the present speed, and
  a plan that would need a step brake at once (`entry_slack_mps`) is refused. Slow cars less than `merge_gap_m`
  apart are passed in one manoeuvre.
- Keeping a plan. It is re-planned every `plan_every_ticks` until the cars are beside each other (then it is
  committed and runs to its end); a re-plan that fails keeps the old plan while it still clears the slow cars under
  the latest prediction, else the car returns to its line along the old path's shape (return path: offset and
  heading continuous, gentle, `return_lat_frac`, up to `return_ramp_max_m`). Closer than `abort_sep_m` while beside
  is logged as a near miss only: braking or swerving next to a car that never yields did more harm in development.
- Following. A slow car that the present path would pass closer than `sep_m` (over the stretch where the ego
  catches it; hysteresis `hyst_m`) caps the speed so that the ego can stop `gap_min_m` behind it with
  `follow_decel_frac` of the scaled grip after a response lag; the tracker lets a cap fall only gently. The cars of
  a pass in progress are left to the pass, except for an emergency cap when the bodies would overlap ahead.
"""

import math

import numpy as np

from .planner import Line

REF_X_M = 0.12892
EGO_FRONT_FROM_REF_M = 0.33446 - 0.12892
SLOW_REAR_FROM_REF_M = 0.12892 + 0.094          # the tail box overhangs the rear by 41 mm
SLOW_FRONT_FROM_REF_M = 0.33446 - 0.12892
EGO_REAR_FROM_REF_M = 0.12892 + 0.0533
CAR_WIDTH_M = 0.18492
G = 9.81


def _fill_nan(a):
    """Replace NaN by linear interpolation between the defined values (ends held)."""
    a = np.asarray(a, dtype=float)
    ok = ~np.isnan(a)
    if ok.all() or not ok.any():
        return np.where(ok, a, 0.0) if not ok.any() else a
    x = np.arange(len(a))
    return np.interp(x, x[ok], a[ok])


def _smooth_extreme(a, k, kind):
    """A running max (or min) over +-k samples, then a moving average over +-k: smooth and still on the safe side."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    pad = np.concatenate([np.full(k, a[0]), a, np.full(k, a[-1])])
    out = np.empty(n)
    for i in range(n):
        w = pad[i:i + 2 * k + 1]
        out[i] = w.max() if kind == "max" else w.min()
    pad = np.concatenate([np.full(k, out[0]), out, np.full(k, out[-1])])
    av = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    return np.maximum(av, a) if kind == "max" else np.minimum(av, a)


class PassPath(object):
    def __init__(self, line, prof, ident, cars, s_end_plan, side):
        self.committed = False
        self.near_miss = set()
        self.line = line
        self.loop = line.loop
        self.prof = prof
        self.ident = ident
        self.cars = cars
        self.s_end_plan = s_end_plan
        self.side = side

    def speed(self, s):
        return self.prof.speed(s)


class SlowCar(object):
    def __init__(self, plan, cfg):
        self.id = cfg["id"]
        self.sf = float(cfg["speed_factor"])
        a = np.asarray(cfg["path"]["samples"], dtype=float)
        self.p = a[:, :2]
        self.vref = a[:, 2]
        seg = np.roll(self.p, -1, axis=0) - self.p
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(self.seg_len.sum())
        n = len(a)
        sp, dp = np.empty(n), np.empty(n)
        hint = None
        for k in range(n):
            s_, d_, hint = plan.loop.project(self.p[k, 0], self.p[k, 1], hint=hint, window_m=2.0)
            sp[k], dp[k] = s_, d_
        self.s_plan, self.d_plan = sp, dp
        # cruise speed per sample: speed factor x reference speed 0.3 m ahead along the path
        ahead = np.searchsorted(self.s, (self.s + 0.3) % self.length, side="right") - 1
        self.v_cruise = self.sf * self.vref[np.clip(ahead, 0, n - 1)]
        self.n = n
        # its path's offset at every plan station (the path runs once round the track: interpolate over its
        # samples' plan stations, periodically)
        order = np.argsort(sp, kind="stable")
        xs = np.concatenate([sp[order] - plan.loop.length, sp[order], sp[order] + plan.loop.length])
        ys = np.concatenate([dp[order]] * 3)
        self.d_at_plan = np.interp(plan.loop.s, xs, ys)

    def nearest(self, x, y):
        d2 = (self.p[:, 0] - x) ** 2 + (self.p[:, 1] - y) ** 2
        return int(np.argmin(d2))

    def predict(self, k0, v0, horizon_s, accel):
        """(t, s_plan, d_plan) along its path from sample k0 at speed v0: arrays per sample until horizon_s."""
        ts, ss, dd = [0.0], [float(self.s_plan[k0])], [float(self.d_plan[k0])]
        t, v, k = 0.0, max(0.05, float(v0)), k0
        for _ in range(self.n):
            ds = float(self.seg_len[k])
            vc = max(0.05, float(self.v_cruise[k]))
            v1 = min(vc, math.sqrt(v * v + 2.0 * accel * ds)) if v < vc else vc
            t += ds / max(0.05, 0.5 * (v + v1))
            v = v1
            k = (k + 1) % self.n
            ts.append(t)
            ss.append(float(self.s_plan[k]))
            dd.append(float(self.d_plan[k]))
            if t > horizon_s:
                break
        return np.asarray(ts), np.asarray(ss), np.asarray(dd)


class Traffic(object):
    def __init__(self, world, plan, line, prof, bounds, params):
        self.plan = plan
        self.line = line
        self.prof = prof
        self.lo, self.hi = bounds
        self.P = params
        self.tp = params["traffic"]
        self.factor = prof.factor
        self.cars = {c["id"]: SlowCar(plan, c) for c in world["truth"]["avoidance"]["slow_cars"]}
        fl = world["truth"]["floor"]
        self.base_mu = min([float(fl["base_grip"])] + [float(p["grip"]) for p in fl["patches"]])
        self.reset()
        self.n_pass = 0
        self.n_abort = 0
        self.n_follow_ticks = 0
        self.n_plans = 0
        self.passes = []
        self.why = {}
        self.n_emergency = 0

    def reset(self):
        self.active = None
        self.in_way = set()
        self.hint_e = None
        self.hint_c = {}
        self.next_plan_tick = 0
        self.cur_path = None

    # ------------------------------------------------------------------ helpers
    def _ego(self, truth):
        x, y, psi = (float(v) for v in truth["car"]["pose"])
        rx, ry = x + REF_X_M * math.cos(psi), y + REF_X_M * math.sin(psi)
        s, d, self.hint_e = self.plan.loop.project(rx, ry, hint=self.hint_e, window_m=3.0)
        return s, d, max(0.0, float(truth["car"]["vel_mps"][0]))

    def _slow_states(self, truth):
        out = []
        for st in truth.get("slow_cars") or []:
            if not st.get("on_track"):
                continue
            car = self.cars.get(st["id"])
            if car is None:
                continue
            x, y, yaw = (float(v) for v in st["pose"])
            rx, ry = x + REF_X_M * math.cos(yaw), y + REF_X_M * math.sin(yaw)
            s, d, h = self.plan.loop.project(rx, ry, hint=self.hint_c.get(st["id"]), window_m=3.0)
            self.hint_c[st["id"]] = h
            k = car.nearest(rx, ry)
            dev = d - float(car.d_plan[k])
            out.append({"id": st["id"], "car": car, "s": s, "d": d, "v": float(st.get("speed_mps", 0.0)), "k": k,
                        "dev": dev, "pose": (x, y, yaw)})
        return out

    def _ego_predict(self, path_line, path_prof, s_ego_line, horizon_s):
        """(t, s_plan) of the ego along a line at its profile speed from line station s_ego_line."""
        lp = path_line.loop
        j = lp.index(s_ego_line)
        ts, ss = [0.0], [float(path_line.s_plan[j])]
        t = 0.0
        for _ in range(lp.n):
            v = max(0.3, float(path_prof.v[j]))
            t += lp.ds / v
            j = (j + 1) % lp.n
            ts.append(t)
            ss.append(float(path_line.s_plan[j]))
            if t > horizon_s:
                break
        return np.asarray(ts), np.asarray(ss)

    def _offsets_line(self, line):
        return line.offsets

    # ------------------------------------------------------------------ per tick
    def update(self, truth, tick):
        tp = self.tp
        s_e, d_e, v_e = self._ego(truth)
        slow = self._slow_states(truth)
        cap = None
        # ---- the active pass: committed (beside), finished, aborted?
        if self.active is not None:
            act = self.active
            done = True
            for sc in slow:
                if sc["id"] not in act.cars:
                    continue
                rel = self.plan.loop.gap(s_e, sc["s"])
                if rel < float(tp["done_m"]):
                    done = False
                beside = -float(tp["behind_m"]) <= rel <= float(tp["ahead_m"])
                if beside:
                    act.committed = True
                    sep_now = abs(d_e - sc["d"])
                    if sep_now < float(tp["abort_sep_m"]) and sc["id"] not in act.near_miss:
                        # closer than planned while beside: logged only. Braking or swerving next to a car that
                        # never yields did more harm than good in development; the faster car carries on.
                        act.near_miss.add(sc["id"])
                        self.n_abort += 1
                        self.passes.append({"tick": tick, "event": "near_miss", "cars": sorted(act.cars),
                                            "sep_m": round(sep_now, 3), "rel_m": round(rel, 3)})
            if self.active is not None and done and self.plan.loop.gap(s_e, act.s_end_plan) >= 0.0:
                if act.side != "return":
                    self.passes.append({"tick": tick, "event": "done", "cars": sorted(act.cars), "side": act.side})
                    self.n_pass += 1
                    self.active = self._return_path(truth, s_e, d_e, v_e, tick, from_path=act.line)  # None: on it
                else:
                    self.active = None
        # ---- plan (or re-plan, until the cars are beside each other) a pass of the nearest slow cars ahead
        replan = self.active is None or not self.active.committed
        if replan and slow and tick >= self.next_plan_tick:
            look = max(float(tp["look_min_m"]), float(tp["look_s"]) * v_e)
            ahead = [(self.plan.loop.gap(sc["s"], s_e), sc) for sc in slow]
            ahead = [(g, sc) for g, sc in ahead if -float(tp["behind_m"]) < g < look]
            if ahead:
                self.next_plan_tick = tick + int(tp["plan_every_ticks"])
                ahead.sort(key=lambda x: x[0])
                new = self._plan_pass(truth, s_e, d_e, v_e, ahead, tick)
                if new is not None:
                    self.active = new
                    self.n_plans += 1
                elif self.active is not None and not self.active.committed and self.active.side != "return":
                    # no new plan: keep the old one while it still clears every slow car under the latest
                    # prediction; otherwise back to the line along the old path's shape (behind the slow car,
                    # the following cap keeps the gap)
                    still = all(self._separation(self.active.line, sc, s_e, v_e, self.plan.loop.gap(sc["s"], s_e))
                                >= float(tp["sep_m"]) - float(tp["keep_slack_m"]) for sc in slow
                                if sc["id"] in self.active.cars)
                    if not still:
                        self.active = self._return_path(truth, s_e, d_e, v_e, tick, from_path=self.active.line)
        # ---- following: a slow car the ego's present path would pass closer than sep_m (over the whole stretch
        # where the ego catches it), with hysteresis; the cap lets the ego stop gap_min_m behind it
        path_line = self.active.line if self.active is not None else self.line
        passing = set(self.active.cars) if (self.active is not None and self.active.side in ("left", "right")) else set()
        for sc in slow:
            g = self.plan.loop.gap(sc["s"], s_e)
            if g <= -float(tp["behind_m"]) or g > max(float(tp["follow_look_m"]), float(tp["look_s"]) * v_e):
                self.in_way.discard(sc["id"])
                continue
            if sc["id"] in passing:
                self.in_way.discard(sc["id"])
                # the pass handles it; only an actual overlap of the bodies ahead brakes the car
                gap_b = g - EGO_FRONT_FROM_REF_M - SLOW_REAR_FROM_REF_M
                if 0.0 < gap_b < float(tp["emergency_gap_m"]) and abs(d_e - sc["d"]) < CAR_WIDTH_M + 0.05:
                    vc = max(0.0, sc["v"] - 0.3)
                    cap = vc if cap is None else min(cap, vc)
                    self.n_emergency += 1
                continue
            sep = self._separation(path_line, sc, s_e, v_e, g)
            need = float(tp["sep_m"]) + (float(tp["hyst_m"]) if sc["id"] in self.in_way else 0.0)
            if sep >= need:
                self.in_way.discard(sc["id"])
                continue
            self.in_way.add(sc["id"])
            if g <= 0.0:
                continue                                       # beside or just behind it: the pass logic decides
            gap = g - EGO_FRONT_FROM_REF_M - SLOW_REAR_FROM_REF_M - float(tp["gap_min_m"]) - v_e * float(tp["lag_s"])
            a = float(tp["follow_decel_frac"]) * self.factor ** 2 * self.base_mu * G
            vc = math.sqrt(max(0.0, sc["v"] ** 2 + 2.0 * a * max(0.0, gap))) if gap > 0.0 else max(0.0, sc["v"] - 0.3)
            cap = vc if cap is None else min(cap, vc)
        if cap is not None:
            self.n_follow_ticks += 1
        return cap, self.active

    def _separation(self, path_line, sc, s_e, v_e, g):
        """Smallest lateral distance between the ego's path and the slow car's path over the stations where the
        ego will be beside it: from behind_m behind its reference point to where the ego catches it (its present
        speed against the ego's) plus ahead_m; its present tracking error fades out over that stretch."""
        tp = self.tp
        plan = self.plan
        n, ds = plan.n, plan.loop.ds
        vs = max(sc["v"], 0.1)
        closing = max(v_e - vs, float(tp["closing_min_mps"]))
        reach = min(float(tp["catch_max_m"]), max(0.0, g) * vs / closing) + float(tp["ahead_m"]) + 0.5
        k0 = int(math.floor((sc["s"] - float(tp["behind_m"])) / ds))
        k1 = int(math.ceil((sc["s"] + reach) / ds))
        ks = np.arange(k0, k1 + 1) % n
        fade = np.clip(1.0 - (np.arange(len(ks)) * ds) / max(reach, 1e-6), 0.0, 1.0)
        d_slow = sc["car"].d_at_plan[ks] + sc["dev"] * fade
        return float(np.min(np.abs(path_line.offsets[ks] - d_slow))) - float(tp["dev_margin_m"])

    def _return_path(self, truth, s_e, d_e, v_e, tick, from_path=None):
        """A ramp back to the line, from the car's present offset or, with `from_path`, from that path's shape
        (offset and heading stay continuous); None when the car is on the line already."""
        tp = self.tp
        plan = self.plan
        n, ds, L = plan.n, plan.loop.ds, plan.loop.length
        n_line = self.line.offsets
        k0 = int(math.floor(s_e / ds))
        v_r = max(v_e, 1.0)
        look = (k0 + np.arange(0, int(round(3.0 / ds)))) % n
        if from_path is not None:
            dev_arr = from_path.offsets - n_line
            dev = float(np.max(np.abs(dev_arr[look])))
            dev_car = d_e - plan.loop.interp(n_line, s_e)
        else:
            dev_arr = None
            dev = abs(d_e - plan.loop.interp(n_line, s_e))
            dev_car = d_e - plan.loop.interp(n_line, s_e)
        if dev < float(tp["late_dev_m"]):
            return None
        a_ret = float(tp["return_lat_frac"]) * self.factor ** 2 * self.base_mu * G     # gentler than a pass ramp:
        Lr = min(float(tp["return_ramp_max_m"]), max(float(tp["ramp_min_m"]),          # the car often brakes
                                                     math.sqrt(6.0 * dev * v_r * v_r / a_ret)))  # meanwhile
        offs = n_line.copy()
        kb = int(round(1.0 / ds))
        for j in range(-kb, 0):
            u = (j + kb) / kb
            w = u * u * (3.0 - 2.0 * u)
            k = (k0 + j) % n
            offs[k] = n_line[k] + dev_car * w
        kr = int(math.ceil(Lr / ds))
        kblend = max(1, int(round(1.0 / ds)))
        for j in range(0, kr + 1):
            u = j / kr
            w = 1.0 - u * u * (3.0 - 2.0 * u)
            k = (k0 + j) % n
            if dev_arr is not None:
                # from where the car is into the old path's shape over 1 m (no kink if the car is off that path)
                b = min(1.0, j / kblend)
                b = b * b * (3.0 - 2.0 * b)
                base = dev_car + (dev_arr[k] - dev_car) * b
            else:
                base = dev_car
            offs[k] = n_line[k] + base * w
        near = np.zeros(n, dtype=bool)
        near[(k0 + np.arange(-kb, int(round(0.5 / ds)))) % n] = True
        offs = np.where(near, offs, np.clip(offs, self.lo, self.hi))     # the line may move towards a wall ahead
        line = Line(plan, offs)
        x, y, psi = (float(v) for v in truth["car"]["pose"])
        rx, ry = x + REF_X_M * math.cos(psi), y + REF_X_M * math.sin(psi)
        _sj, _dj, ij = line.loop.project(rx, ry)
        prof = plan.profile(line, self.factor, start_index=ij, start_speed=v_e)
        self.n_plans += 1
        if len(self.passes) < 400:
            self.passes.append({"tick": tick, "event": "return", "dev_m": round(dev, 3), "ramp_m": round(Lr, 2)})
        return PassPath(line, prof, "ret%d" % self.n_plans, set(), ((k0 + kr) * ds) % L, "return")

    # ------------------------------------------------------------------ planning a pass
    def _plan_pass(self, truth, s_e, d_e, v_e, ahead, tick):
        tp = self.tp
        plan = self.plan
        L = plan.loop.length
        n = plan.n
        ds = plan.loop.ds
        horizon = float(tp["horizon_s"])
        acc = float(tp["slow_accel_frac"]) * self.base_mu * G
        sep = float(tp["sep_m"])
        x, y, psi = (float(v) for v in truth["car"]["pose"])
        rx, ry = x + REF_X_M * math.cos(psi), y + REF_X_M * math.sin(psi)
        s_line, _d, _i = self.line.loop.project(rx, ry)
        te, se = self._ego_predict(self.line, self.prof, s_line, horizon)
        se_u = s_e + np.concatenate([[0.0], np.cumsum(np.diff(se) % L)])            # unwrapped ego stations
        req_l = np.full(n, -np.inf)
        req_r = np.full(n, np.inf)
        involved = []
        win_a = win_b = None
        for g, sc in ahead:
            car = sc["car"]
            tc, sc_s, sc_d = car.predict(sc["k"], sc["v"], horizon, acc)
            sc_u = s_e + g + np.concatenate([[0.0], np.cumsum(np.diff(sc_s) % L)])
            sj = np.interp(te, tc, sc_u, right=np.nan)
            rel = se_u - sj
            clear = np.nonzero((~np.isnan(rel)) & (rel >= float(tp["ahead_m"])))[0]
            if not len(clear):
                continue                                       # the ego does not clear it within the horizon
            # the stretch the requirement covers: from where the slow car is now (less behind_m) to where the ego
            # clears it at its cruise speed. A slower slow car is cleared earlier, inside the stretch; the timing
            # of the prediction does not matter (packs, a slow car slowed by another one)
            a_s = s_e + g - float(tp["behind_m"])
            b_s = float(se_u[clear[0]]) + float(tp["clear_extra_m"])
            if win_a is not None and a_s > win_b + float(tp["merge_gap_m"]):
                break                                          # the next car is far enough: pass it later
            win_a = a_s if win_a is None else min(win_a, a_s)
            win_b = b_s if win_b is None else max(win_b, b_s)
            ks = np.arange(int(math.floor(a_s / ds)), int(math.ceil(b_s / ds)) + 1)
            kk = ks % n
            dist = (ks - ks[0]) * ds
            dev0 = sc["dev"] * np.clip(1.0 - dist, 0.0, 1.0)          # its present tracking error, over 1 m
            d_slow = car.d_at_plan[kk] + dev0
            margin = float(tp["dev_margin_m"])
            np.maximum.at(req_l, kk, d_slow + sep + margin)
            np.minimum.at(req_r, kk, d_slow - sep - margin)
            involved.append(sc["id"])
        if not involved:
            self.why["no_window"] = self.why.get("no_window", 0) + 1
            return None
        # the requirement over the whole window (filled between samples, dilated by +-dilate_m)
        ka = int(math.floor(win_a / ds))
        kb = int(math.ceil(win_b / ds))
        kw = np.arange(ka, kb + 1) % n
        dil = max(1, int(round(float(tp["dilate_m"]) / ds)))
        rl, rr = req_l.copy(), req_r.copy()
        for j in range(1, dil + 1):
            rl = np.maximum(rl, np.maximum(np.roll(req_l, j), np.roll(req_l, -j)))
            rr = np.minimum(rr, np.minimum(np.roll(req_r, j), np.roll(req_r, -j)))
        wl = np.maximum.accumulate(np.where(np.isfinite(rl[kw]), rl[kw], -np.inf))
        # fill: the running extreme from both ends (no station of the window without a requirement)
        fl = np.where(np.isfinite(rl[kw]), rl[kw], np.nan)
        fr = np.where(np.isfinite(rr[kw]), rr[kw], np.nan)
        fl = _fill_nan(fl)
        fr = _fill_nan(fr)
        del wl
        n_line = self.line.offsets
        lo_w, hi_w = self.lo[kw], self.hi[kw]
        opts = []
        if np.all(fl <= hi_w):
            dev = float(np.max(np.maximum(0.0, fl - n_line[kw])))
            opts.append((dev, "left", np.maximum(n_line[kw], fl)))
        if np.all(fr >= lo_w):
            dev = float(np.max(np.maximum(0.0, n_line[kw] - fr)))
            opts.append((dev, "right", np.minimum(n_line[kw], fr)))
        if not opts:
            self.why["no_room"] = self.why.get("no_room", 0) + 1
            return None
        opts.sort(key=lambda o: o[0])
        dev, side, Tw = opts[0]
        Tw = _smooth_extreme(Tw, max(1, int(round(0.3 / ds))), "max" if side == "left" else "min")
        Tw = np.clip(Tw, lo_w, hi_w)                       # smoothing must not push the target past a limit
        req = fl if side == "left" else fr
        if (side == "left" and np.any(Tw < req - 1e-9)) or (side == "right" and np.any(Tw > req + 1e-9)):
            self.why["no_room"] = self.why.get("no_room", 0) + 1
            return None
        # ramp in. Every path starts where the car is: behind it the line blends into its present offset; ahead
        # of it, when the window is near, one ramp goes from that offset to the window's target; when the window
        # is far, the car first returns to the line, drives it, and ramps to the target before the window.
        v_r = max(v_e, 1.0)
        a_lat = float(tp["ramp_lat_frac"]) * self.factor ** 2 * self.base_mu * G
        offs = n_line.copy()
        T0 = float(Tw[0])
        k0 = int(math.floor(s_e / ds))
        dev_car = d_e - plan.loop.interp(n_line, s_e)

        def ramp_len(delta):
            return min(float(tp["ramp_max_m"]), max(float(tp["ramp_min_m"]), math.sqrt(6.0 * abs(delta) * v_r * v_r / a_lat)))

        kbh = int(round(1.0 / ds))
        for j in range(-kbh, 0):                                     # behind the car: line -> its offset
            u = (j + kbh) / kbh
            w = u * u * (3.0 - 2.0 * u)
            k = (k0 + j) % n
            offs[k] = n_line[k] + dev_car * w
        L_back = ramp_len(dev_car) if abs(dev_car) > 0.01 else 0.0
        Lr = ramp_len(T0 - n_line[kw[0]])
        dist = win_a - s_e
        if dist <= L_back + Lr:
            # near: one ramp from the car's offset to the target
            need = ramp_len(T0 - d_e)
            if dist < need * float(tp["ramp_short_frac"]) and abs(T0 - d_e) > float(tp["late_dev_m"]):
                self.why["late"] = self.why.get("late", 0) + 1
                return None
            Lb = max(dist, 1e-6)
            for k in range(k0, ka):
                u = min(1.0, max(0.0, (k * ds - s_e) / Lb))
                w = u * u * (3.0 - 2.0 * u)
                offs[k % n] = d_e + (T0 - d_e) * w
            from_car = True
        else:
            # far: back to the line first (if the car is off it), then the line, then the ramp to the target
            kr = int(math.ceil(L_back / ds)) if L_back > 0.0 else 0
            for j in range(0, kr + 1):
                if kr == 0:
                    break
                u = j / kr
                w = 1.0 - u * u * (3.0 - 2.0 * u)
                k = (k0 + j) % n
                offs[k] = n_line[k] + dev_car * w
            start = win_a - Lr
            for k in range(int(math.floor(start / ds)), ka):
                u = min(1.0, max(0.0, (k * ds - start) / Lr))
                w = u * u * (3.0 - 2.0 * u)
                kk = k % n
                if side == "left":
                    offs[kk] = n_line[kk] + max(0.0, T0 - n_line[kk]) * w
                else:
                    offs[kk] = n_line[kk] - max(0.0, n_line[kk] - T0) * w
            from_car = False
        offs[kw] = Tw
        # ramp out: back to the line
        T1 = float(Tw[-1])
        need_out = math.sqrt(6.0 * abs(T1 - n_line[(kb + 1) % n]) * v_r * v_r / a_lat)
        Lo = min(float(tp["ramp_max_m"]), max(float(tp["ramp_min_m"]), need_out))
        ko = int(math.ceil(Lo / ds))
        for j in range(1, ko + 1):
            k = (kb + j) % n
            u = j / ko
            w = 1.0 - u * u * (3.0 - 2.0 * u)
            if side == "left":
                offs[k] = n_line[k] + max(0.0, T1 - n_line[k]) * w
            else:
                offs[k] = n_line[k] - max(0.0, n_line[k] - T1) * w
        near = np.zeros(n, dtype=bool)
        near[(k0 + np.arange(-kbh, int(round(0.5 / ds)))) % n] = True
        bad = (~near) & ((offs > self.hi + 1e-6) | (offs < self.lo - 1e-6))
        if np.any(bad):                                    # a ramp would leave the limits: no pass from here
            self.why["ramp_out_of_limits"] = self.why.get("ramp_out_of_limits", 0) + 1
            return None
        line = Line(plan, offs)
        sj, _dj, ij = line.loop.project(rx, ry)
        prof = plan.profile(line, self.factor, start_index=ij, start_speed=v_e)
        if float(prof.v[ij]) < v_e - float(tp["entry_slack_mps"]):
            self.why["too_fast"] = self.why.get("too_fast", 0) + 1
            return None                                      # the path's own profile would need a step brake
        ident = "pass%d" % (self.n_plans + 1)
        end_s = (kb + ko) * ds
        if len(self.passes) < 400:
            self.passes.append({"tick": tick, "event": "plan", "cars": involved, "side": side, "dev_m": round(dev, 3),
                                "window_m": [round(win_a % L, 2), round(win_b % L, 2)], "ramp_m": [round(Lr, 2), round(Lo, 2)],
                                "from_car": from_car})
        return PassPath(line, prof, ident, set(involved), end_s % L, side)

    def summary(self):
        ev = [e for e in self.passes if e["event"] != "plan"]
        return {"passes_done": self.n_pass, "near_misses": self.n_abort, "plans": self.n_plans,
                "follow_ticks": self.n_follow_ticks, "emergency_ticks": self.n_emergency, "why_not": dict(self.why),
                "events": ev[:60]}
