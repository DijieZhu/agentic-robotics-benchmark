"""Racecraft of the reference program in a four-car heat: following, passing and merging back among opponents whose
lines, paces and states it knows (numpy only).

    rc = Racecraft(plan, line, prof, bounds, field, params, race_params)    # field: the heat's opponent records
    rc.reset()                                                                 # at the start and after a release
    cap, path = rc.update(truth, tick, t_s, base_line, base_prof)              # per tick

`cap` is a speed cap or None; `path` the path the car is to follow (a follow, pass or return path) or None (then the
car follows its base path: the racing line, or the join path that leads onto it).

What it knows (privileged): every opponent's line (the trajectory of its reference point with its reference speed),
its pace (a fixed fraction of that speed) and, every tick, its true pose and speed. It knows the opponents' driver
too: each follows its own line at its pace, accelerates gently, slows for anything close ahead in its band and follows
a slower car without overtaking it.

Prediction. Every `plan_every_ticks` the opponents are predicted for `horizon_s` on a time grid of `dt_s`, each along
its own line at its pace within the driver's acceleration limits, and behind a slower opponent in its band at the
driver's following distance (a train). The car itself is simulated on a candidate path with the rules its tracker
applies: the path's speed profile, the traction limit when it accelerates, the gentle fall of a speed cap, and the
following law behind every opponent ahead of it that is laterally closer than `sep_m`. An opponent of a pass whose
band the car occupies close ahead of it is held back in the simulation to the driver's following distance behind the
car (the driver slows for it), so the car may slot in between two opponents.

Rules. Against every opponent the car must keep, on the simulated motions: `sep_m` of lateral distance between the
reference points whenever the two bodies overlap along the track (`overlap_margin_m` added at both ends, so that it
also keeps that much room behind an opponent in its band); when it re-enters an opponent's band ahead of it, at least
`merge_gap_m` between its own tail and the opponent's nose; and after that, while it is in the band of an opponent it
has passed, at least `caught_gap_m` between them. A conflict is a breach of one of these rules.

Following. An opponent is in the way when the car, driving its present path freely, would conflict with it. Behind
an opponent in the way the speed is capped by
    v_cap = min(v_opp + k (g - g_f), sqrt(v_opp^2 + 2 a (g - g_f)))      g = gap of the reference points
with g_f = `follow_gap_m` (`follow_gap_train_m` behind an opponent that itself runs close behind another), k =
`follow_gain_per_s` and a = `follow_decel_frac` x grip x g, and bounded by the opponent's predicted motion: braking at
a from now, the car stays `follow_min_m` (`follow_min_train_m`) behind every predicted position of the opponent over
the next `anticipate_s`. The tracker lets a cap fall only gently. Closer than `follow_engage_m` the car moves onto the
opponent's own line (a follow path) and follows it there, directly behind it.

Passing. A pass of the opponent in the way, together with those running less than `train_gap_m` ahead of it, is
planned on either side: the opponent's line offset by `sep_m` to that side (for several opponents, each line where the
car is beside that opponent), reached from the car's offset while the car is still held behind the opponent at the
following distance (the release point is the end of that ramp), kept until the simulated merge point plus
`merge_extra_m`, then a ramp back to the racing line. Its speed profile is recomputed on the pass path from the present
speed with the grip under the path (at `pass_factor`); a path that needs a step brake is refused. A pass is accepted
only when the simulation of it finds no conflict with the opponents of the pass and none with another opponent until
`after_pass_s` after its end; when the merge point moves beyond the stretch the path is built again (at most
`max_builds` times). Of the two sides the one that completes the pass first is taken. Until it is released the plan is
checked every planning tick: a conflict within `keep_plan_s` drops it (a pass of another shape, a follow path or a
return path takes over, else the car stays on the plan, held); a later conflict rebuilds it or keeps it with the
release blocked until a check is clean. Once released (or beside an opponent) it is rebuilt from the car's present
offset when a conflict appears, and never abandoned for braking beside a car that does not yield.

Paths. Every new path starts at the car's offset and keeps, behind the car, the shape of the path it is on. A follow
path or a return path is taken only when the simulated car on it keeps the rules for `check_s`; otherwise the car
stays on its present path. When nothing is active the car is on its racing line.

Start. While the opponent in the way has not reached `start_cruise_frac` of its pace, or within `start_hold_s` of the
start, no pass is planned: the car keeps its following distance in the bunch of the grid.
"""

import bisect
import math

import numpy as np

from .planner import WHEEL_X_FROM_REF_M, WHEEL_Y_M, Line, Profile

REF_X_M = 0.12892
EGO_FRONT_M = 0.33446 - 0.12892            # the car's nose ahead of its reference point
EGO_REAR_M = 0.12892 + 0.0533              # its tail behind it
OPP_FRONT_M = 0.33446 - 0.12892
OPP_REAR_M = 0.12892 + 0.094               # an opponent's marker box overhangs its rear by 41 mm
CAR_WIDTH_M = 0.18492
G = 9.81
INF = float("inf")


def _smoothstep(u):
    return u * u * (3.0 - 2.0 * u)


def _fill_nan(a):
    """Replace NaN by linear interpolation between the defined values (ends held)."""
    a = np.asarray(a, dtype=float)
    ok = ~np.isnan(a)
    if ok.all() or not ok.any():
        return a
    x = np.arange(len(a))
    return np.interp(x, x[ok], a[ok])


def _running(a, back, ahead, kind):
    """Running max (or min) of a over the samples i - back .. i + ahead (ends held)."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    pad = np.concatenate([np.full(back, a[0]), a, np.full(ahead, a[-1])])
    out = a.copy()
    for j in range(back + ahead + 1):
        w = pad[j:j + n]
        out = np.maximum(out, w) if kind == "max" else np.minimum(out, w)
    return out


def _smooth_extreme(a, k, kind):
    """A running max (or min) over +-k samples, then a moving average over +-k: smooth and still on the safe side."""
    a = np.asarray(a, dtype=float)
    ext = _running(a, k, k, kind)
    pad = np.concatenate([np.full(k, ext[0]), ext, np.full(k, ext[-1])])
    av = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    return np.maximum(av, a) if kind == "max" else np.minimum(av, a)


def local_profile(plan, line, factor, i0, v0, reach_m, base):
    """The quasi-steady-state profile of `line` at `factor` from sample i0 at speed v0, computed over `reach_m` ahead
    of i0 (the stretch where the line may differ from `base`, plus braking room): the same capacities as the
    planner's profile (lateral c_lat f^2 mu g; acceleration min(c_trac f^2 mu g, motor envelope) inside the friction
    ellipse of exponent acc_exp; braking c_brk f^2 mu g inside the ellipse; the grip of the four wheels with the
    raised-edge derating, the smallest within the grip window), with the profile of `base` (a profile of a line that
    coincides with this one beyond the stretch) at the end of the stretch and outside it. Returns a planner Profile."""
    lp = line.loop
    n, ds = lp.n, lp.ds
    lim = plan.limits
    pp = plan.P["plan"]
    kw = int(round(float(pp["grip_window_m"]) / ds))
    m = min(n - 2 * kw - 2, int(math.ceil(reach_m / ds)) + 1)
    # base values on this line's samples (by plan station)
    bl = base.line
    j = np.searchsorted(bl.s_plan_sorted, line.s_plan) % bl.loop.n
    jj = bl.s_plan_order[j]
    v = np.asarray(base.v, dtype=float)[jj].copy()
    mu = np.asarray(base.mu, dtype=float)[jj].copy()
    # grip under the four wheels over the stretch (with the grip window's margins)
    idx = (i0 - kw + np.arange(m + 2 * kw)) % n
    pts = []
    for sx in (WHEEL_X_FROM_REF_M, -WHEEL_X_FROM_REF_M):
        for sy in (WHEEL_Y_M, -WHEEL_Y_M):
            pts.append(lp.p[idx] + sx * lp.tan[idx] + sy * lp.nl[idx])
    g = np.min(np.stack([plan.grip.grip_points(q) for q in pts]), axis=0)
    zone = float(pp.get("edge_zone_m", 0.0))
    if zone > 0.0:
        near = np.zeros(len(idx), dtype=bool)
        for q in pts:
            near |= plan.grip.near_raised_edge(q, zone)
        g = np.where(near, g * float(pp["edge_grip_factor"]), g)
    gw = g.copy()
    for k in range(1, kw + 1):
        gw[kw:-kw] = np.minimum(gw[kw:-kw], np.minimum(g[kw - k:len(g) - kw - k], g[kw + k:len(g) - kw + k]))
    seg = idx[kw:kw + m]
    mu_w = gw[kw:kw + m]
    mu[seg] = mu_w
    f2 = float(factor) ** 2
    kap = np.maximum(np.abs(lp.kappa[seg]), 1e-6)
    ay = lim.c_lat * f2 * mu_w * 9.81
    ab = lim.c_brk * f2 * mu_w * 9.81
    at = lim.c_trac * f2 * mu_w * 9.81
    vmax = np.minimum(lim.v_cap, np.sqrt(ay / kap)).tolist()
    kl, ayl, abl, atl = kap.tolist(), ay.tolist(), ab.tolist(), at.tolist()
    mv, ma = lim.mv.tolist(), lim.ma.tolist()
    pe = float(lim.acc_exp)
    two_ds = 2.0 * ds
    vv = list(vmax)
    vv[0] = min(vv[0], max(0.0, float(v0)))
    vv[-1] = min(vv[-1], float(v[seg[-1]]))                       # the base profile at the end of the stretch
    for q in range(m - 1):                                         # forward: acceleration
        vi = vv[q]
        lat = min(1.0, vi * vi * kl[q] / ayl[q])
        rem = max(0.0, 1.0 - lat ** pe) ** (1.0 / pe)
        if vi >= mv[-1]:
            amot = 0.0
        else:
            u = bisect.bisect_right(mv, vi) - 1
            amot = ma[u] + (ma[u + 1] - ma[u]) * (vi - mv[u]) / (mv[u + 1] - mv[u])
        a = min(atl[q], amot) * rem
        vj = math.sqrt(vi * vi + two_ds * a)
        if vj < vv[q + 1]:
            vv[q + 1] = vj
    for q in range(m - 1, 0, -1):                                  # backward: braking
        vj = vv[q]
        lat = vj * vj * kl[q] / ayl[q]
        rem = math.sqrt(max(0.0, 1.0 - lat * lat))
        vi = math.sqrt(vj * vj + two_ds * abl[q] * rem)
        if vi < vv[q - 1]:
            vv[q - 1] = vi
    v[seg] = vv
    t_lap = float(np.sum(ds * 2.0 / (v + np.roll(v, -1))))
    return Profile(line, v, factor, mu, t_lap)


def _predictive_cap(u, k0, g, g_min, a, dt, n_ahead):
    """The largest speed from which braking at `a` keeps the car `g_min` behind an opponent whose predicted
    plan-station increments are `u` (from step k0 on; its reference point is `g` ahead of the car's now)."""
    best = INF
    if k0 >= len(u):
        return best
    u0 = u[k0]
    for k in range(1, n_ahead + 1):
        if k0 + k >= len(u):
            break
        tau = k * dt
        room = g + (u[k0 + k] - u0) - g_min
        v_ok = (room + 0.5 * a * tau * tau) / tau
        if v_ok < best:
            best = v_ok
    return max(0.0, best)


class Opponent(object):
    """One opponent: its line in the plan frame, its pace, and a cursor that follows it along its line."""

    def __init__(self, plan, cfg, rp):
        self.id = str(cfg["id"])
        self.sf = float(cfg["speed_factor"])
        a = np.asarray(cfg["path"]["samples"], dtype=float)
        self.p = np.ascontiguousarray(a[:, :2])
        vref = a[:, 2].copy()
        n = len(a)
        self.n = n
        self.seg = np.roll(self.p, -1, axis=0) - self.p
        self.seg_len = np.hypot(self.seg[:, 0], self.seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(self.seg_len.sum())
        L = plan.loop.length
        sp, dp = np.empty(n), np.empty(n)
        hint = None
        for k in range(n):
            s_, d_, hint = plan.loop.project(self.p[k, 0], self.p[k, 1], hint=hint, window_m=2.0)
            sp[k], dp[k] = s_, d_
        self.s_plan, self.d_plan = sp, dp
        du = np.diff(np.concatenate([sp, sp[:1]]))
        du = (du + 0.5 * L) % L - 0.5 * L
        self.u_plan = np.concatenate([[0.0], np.cumsum(du)[:-1]])          # unwrapped plan station per sample
        self.u_lap = float(np.sum(du))
        ahead = (self.s + float(rp["opp_anticipation_m"])) % self.length
        idx = np.clip(np.searchsorted(self.s, ahead, side="right") - 1, 0, n - 1)
        frac = (ahead - self.s[idx]) / self.seg_len[idx]
        vr = vref[idx] * (1.0 - frac) + vref[(idx + 1) % n] * frac
        self.v_cruise = self.sf * vr
        order = np.argsort(sp, kind="stable")
        xs = np.concatenate([sp[order] - L, sp[order], sp[order] + L])
        ys = np.concatenate([dp[order]] * 3)
        self.d_at_plan = np.interp(plan.loop.s, xs, ys)                     # its offset at every plan station
        # extended tables for the vectorised lookups along the line (one sample past the end)
        self.s_ext = np.concatenate([self.s, [self.length]])
        self.u_ext = np.concatenate([self.u_plan, [self.u_lap]])
        self.d_ext = np.concatenate([dp, dp[:1]])
        self._s = self.s.tolist()
        self._len = self.seg_len.tolist()
        self._vc = self.v_cruise.tolist() + [float(self.v_cruise[0])]

    def locate(self, x, y, hint=None):
        """(station on its line, sample index) of the point: near the hint sample when one is given."""
        if hint is None:
            idx = np.arange(self.n)
        else:
            idx = (int(hint) + np.arange(-8, 16)) % self.n
        a, d, ln = self.p[idx], self.seg[idx], self.seg_len[idx]
        qx, qy = x - a[:, 0], y - a[:, 1]
        u = np.clip((qx * d[:, 0] + qy * d[:, 1]) / (ln * ln), 0.0, 1.0)
        fx, fy = qx - u * d[:, 0], qy - u * d[:, 1]
        j = int(np.argmin(fx * fx + fy * fy))
        if hint is not None and float(fx[j] * fx[j] + fy[j] * fy[j]) > 0.36:
            return self.locate(x, y)
        i = int(idx[j])
        return float((self.s[i] + u[j] * ln[j]) % self.length), i

    def cruise(self, sp):
        sp = sp % self.length
        i = bisect.bisect_right(self._s, sp) - 1
        i = min(max(i, 0), self.n - 1)
        f = (sp - self._s[i]) / self._len[i]
        return self._vc[i] + f * (self._vc[i + 1] - self._vc[i])

    def along(self, sp_unwrapped):
        """(plan-station increment from station 0 of its line, offset) at unwrapped stations along its line."""
        x = np.asarray(sp_unwrapped, dtype=float)
        laps = np.floor(x / self.length)
        r = x - laps * self.length
        u = np.interp(r, self.s_ext, self.u_ext) + laps * self.u_lap
        d = np.interp(r, self.s_ext, self.d_ext)
        return u, d


class Prediction(object):
    """Predicted motion of one opponent on the time grid: plan-station increment u (from its present station),
    gap0 (its present station minus the car's, wrapped), offset d, speed v."""

    def __init__(self, opp, state, u, d, v):
        self.opp = opp
        self.id = opp.id
        self.state = state
        self.u = u
        self.d = d
        self.v = v
        self.gap0 = state["gap0"]
        self.in_train = False


class PassPath(object):
    """A path of the racecraft: a pass (side "left" / "right"), a follow path ("follow") or a return path
    ("return"). Distances `x_release`, `x_end` are measured along the track from the car's station when it was
    planned (`s_start`); `x` is the distance the car has driven since."""

    def __init__(self, line, prof, ident, cars, side, s_end_plan, stretch, t_done, hold_id=None, s_release=None,
                 s_start=None, x_release=None, x_end=None):
        self.committed = False
        self.hold_id = hold_id                   # the car follows this opponent until it has driven x_release
        self.s_release = s_release
        self.released = x_release is None
        self.s_start = s_start
        self.x_release = x_release
        self.x_end = x_end
        self.x = 0.0
        self.s_last = s_start
        self.blocked = False
        self.line = line
        self.loop = line.loop
        self.prof = prof
        self.ident = ident
        self.cars = list(cars)
        self.side = side
        self.s_end_plan = s_end_plan
        self.stretch = stretch
        self.t_done = t_done

    def speed(self, s):
        return self.prof.speed(s)


class Racecraft(object):
    def __init__(self, plan, line, prof, bounds, field, params, race_params):
        self.plan = plan
        self.line = line
        self.prof = prof
        self.lo, self.hi = bounds
        self.P = params
        self.rp = race_params["racecraft"]
        self.factor = prof.factor
        self.pass_factor = float(self.rp.get("pass_factor") or prof.factor)       # the pace on a pass path
        self.opps = [Opponent(plan, c, self.rp) for c in field]
        self.by_id = {o.id: o for o in self.opps}
        fl = plan.world["truth"]["floor"]
        self.base_mu = min([float(fl["base_grip"])] + [float(p["grip"]) for p in fl["patches"]])
        self.lim = plan.limits
        self.n_plans = 0
        self.n_passes = 0
        self.n_follow_ticks = 0
        self.n_emergency = 0
        self.n_drop = 0
        self.n_rebuild = 0
        self.why = {}
        self.events = []
        self.up_to_pace = set()
        self.line_paths = {}
        self.reset()

    def reset(self):
        self.active = None
        self.preds = None
        self.pred_tick = 0
        self.cur_opps = None
        self.cur_tick = 0
        self.blockers = []
        self.next_pred_tick = 0
        self.next_pass_tick = 0
        self.next_follow_tick = 0
        self.hint_e = None
        self.hint_c = {}
        self.hint_k = {}

    # ------------------------------------------------------------------ states
    def _ego(self, truth):
        x, y, psi = (float(v) for v in truth["car"]["pose"])
        rx, ry = x + REF_X_M * math.cos(psi), y + REF_X_M * math.sin(psi)
        s, d, self.hint_e = self.plan.loop.project(rx, ry, hint=self.hint_e, window_m=3.0)
        return {"s": s, "d": d, "v": max(0.0, float(truth["car"]["vel_mps"][0])), "rx": rx, "ry": ry}

    def _opponents(self, truth, ego):
        out = []
        rows = truth.get("field")
        if rows is None:
            rows = truth.get("slow_cars") or []
        for st in rows:
            if not st.get("on_track", True):
                continue
            opp = self.by_id.get(str(st["id"]))
            if opp is None:
                continue
            x, y, yaw = (float(v) for v in st["pose"])
            rx, ry = x + REF_X_M * math.cos(yaw), y + REF_X_M * math.sin(yaw)
            s, d, h = self.plan.loop.project(rx, ry, hint=self.hint_c.get(opp.id), window_m=3.0)
            self.hint_c[opp.id] = h
            sp, k = opp.locate(rx, ry, hint=self.hint_k.get(opp.id))
            self.hint_k[opp.id] = k
            dev = d - self.plan.loop.interp(opp.d_at_plan, s)
            out.append({"id": opp.id, "opp": opp, "s": s, "d": d, "v": max(0.0, float(st.get("speed_mps", 0.0))),
                        "sp": sp, "dev": dev, "held": bool(st.get("held", False)), "rx": rx, "ry": ry,
                        "gap0": self.plan.loop.gap(s, ego["s"])})
        return out

    # ------------------------------------------------------------------ predictions
    def _predict(self, opps):
        rp = self.rp
        dt = float(rp["dt_s"])
        m = int(round(float(rp["horizon_s"]) / dt))
        a_up = float(rp["opp_accel_frac"]) * G
        a_dn = float(rp["opp_decel_frac"]) * G
        fade = float(rp["dev_fade_m"])
        preds = {}
        for st in opps:
            opp = st["opp"]
            mu = float(self.plan.grip.grip_at(st["rx"], st["ry"]))
            up, dn = a_up * mu, a_dn * mu
            x, v = 0.0, st["v"]
            xs, vs = [0.0], [v]
            sp0 = st["sp"]
            for _k in range(m):
                vc = opp.cruise(sp0 + x)
                if st["held"]:
                    vc = 0.0
                if v < vc:
                    v = min(vc, v + up * dt)
                else:
                    v = max(vc, v - dn * dt)
                x += v * dt
                xs.append(x)
                vs.append(v)
            xs = np.asarray(xs)
            u, d = opp.along(sp0 + xs)
            u0, _d0 = opp.along(np.array([sp0]))
            u = u - u0[0]
            dist = xs
            d = d + st["dev"] * np.clip(1.0 - dist / fade, 0.0, 1.0)
            preds[opp.id] = Prediction(opp, st, u, d, np.asarray(vs))
        self._trains(preds)
        return preds

    def _trains(self, preds):
        """A faster opponent that catches a slower one in its band follows it: its predicted stations are held
        behind the slower one's (two sweeps, enough for three cars)."""
        rp = self.rp
        L = self.plan.loop.length
        keep = OPP_FRONT_M + OPP_REAR_M + float(rp["opp_follow_gap_m"])
        band = float(rp["opp_lane_m"])
        ps = list(preds.values())
        close = float(rp["train_close_m"])
        for _sweep in range(2):
            for pj in ps:
                for pk in ps:
                    if pk is pj:
                        continue
                    ahead = (pk.gap0 - pj.gap0) % L                  # k ahead of j along the track at t = 0
                    if ahead <= 0.0 or ahead > 15.0:
                        continue
                    sk = (pj.state["s"] + ahead) % L
                    if abs(self.plan.loop.interp(pj.opp.d_at_plan, sk) - pk.state["d"]) > band:
                        continue
                    if ahead < close:
                        pj.in_train = True                             # j runs close behind k in its band
                    lim = ahead + pk.u - keep                          # j's increment may not pass this
                    if np.any(pj.u > lim):
                        u = np.minimum(pj.u, lim)
                        u = np.maximum.accumulate(u)
                        v = np.concatenate([[pj.v[0]], np.diff(u) / float(rp["dt_s"])])
                        pj.u = u
                        pj.v = np.minimum(pj.v, np.maximum(v, 0.0))

    # ------------------------------------------------------------------ the car on a path
    def _samples_ahead(self, path_line, path_v, path_mu, ego, reach_m):
        """Arrays along the path from the car's present position: distance x, plan-station increment u (from the
        car's station), offset n, profile speed v, grip mu; and the loop index of the car."""
        lp = path_line.loop
        s_path, _e, i0 = lp.project(ego["rx"], ego["ry"])
        k = int(math.ceil(reach_m / lp.ds)) + 2
        idx = (i0 + np.arange(k)) % lp.n
        x = np.arange(k) * lp.ds - (s_path - lp.s[i0]) % lp.length
        L = self.plan.loop.length
        sp = path_line.s_plan[idx]
        du = np.diff(sp)
        du = (du + 0.5 * L) % L - 0.5 * L
        u = np.concatenate([[0.0], np.cumsum(du)]) + self.plan.loop.gap(float(sp[0]), ego["s"])
        nn = path_line.offsets
        ext_s = np.concatenate([self.plan.loop.s - L, self.plan.loop.s, self.plan.loop.s + L])
        ext_n = np.concatenate([nn, nn, nn])
        n_off = np.interp(sp, ext_s, ext_n)
        return {"x": x, "u": u, "n": n_off, "v": np.asarray(path_v)[idx], "mu": np.asarray(path_mu)[idx], "i0": i0,
                "s_path": s_path}

    def _simulate(self, path_line, path_v, path_mu, ego, preds, follow=True, stop_at=None, hold=None, free=()):
        """Simulate the car on a path against the predicted opponents. Returns a dict with the time series (t, u,
        n, v), the conflicts [(t, id, kind)] and per opponent the first time it is beside (t_a) and the first time
        the car is merge_gap_m clear ahead of it (t_b). `hold` = (opponent id, distance): the car follows that
        opponent, whatever the lateral distance, until it has driven that far along the path (a pass: the car is held
        while it moves to the side); the opponents in `free` (those of the pass) are not followed otherwise."""
        rp = self.rp
        dt = float(rp["dt_s"])
        m = int(round(float(rp["horizon_s"]) / dt))
        L = self.plan.loop.length
        reach = 6.5 * float(rp["horizon_s"]) + 2.0
        A = self._samples_ahead(path_line, path_v, path_mu, ego, reach)
        xs, us, ns, vps, mus = A["x"].tolist(), A["u"].tolist(), A["n"].tolist(), A["v"].tolist(), A["mu"].tolist()
        nk = len(xs)
        ds = (xs[-1] - xs[0]) / max(1, nk - 1)
        x0 = 0.0
        f2 = self.factor ** 2
        c_trac = self.lim.c_trac
        sep = float(rp["sep_m"])
        band = float(rp["lane_m"])
        m_ov = float(rp["overlap_margin_m"])
        back_zone = -(EGO_FRONT_M + OPP_REAR_M + m_ov)
        ahead_zone = EGO_REAR_M + OPP_FRONT_M + m_ov
        merge_need = EGO_REAR_M + OPP_FRONT_M + float(rp["merge_gap_m"])
        caught_need = EGO_REAR_M + OPP_FRONT_M + float(rp["caught_gap_m"])
        nose_tail = EGO_FRONT_M + OPP_REAR_M
        g_f = float(rp["follow_gap_m"])
        g_min = float(rp["follow_min_m"])
        n_ant = int(round(float(rp["anticipate_s"]) / dt))
        kf = float(rp["follow_gain_per_s"])
        a_f = float(rp["follow_decel_frac"]) * self.base_mu * G
        a_fall = float(rp["cap_fall_frac"]) * self.base_mu * G
        look = max(float(rp["look_min_m"]), float(rp["look_s"]) * ego["v"])
        pl = [(p.id, p.gap0, p.u.tolist(), p.d.tolist(), p.v.tolist(), p.in_train) for p in preds.values()]
        g_f_tr = float(rp["follow_gap_train_m"])
        g_min_tr = float(rp["follow_min_train_m"])
        shift = {p.id: 0.0 for p in preds.values()}
        keep_ref = EGO_REAR_M + OPP_FRONT_M + float(rp["opp_follow_gap_m"])
        opp_lane = float(rp["opp_lane_m"])
        lat_prev = {}
        t_a, t_b = {}, {}
        rel = {p.id: [] for p in preds.values()}
        conflicts = []
        x = x0
        v = ego["v"]
        cap_prev = None
        T, U, N, V = [0.0], [], [], [v]
        j0 = 0
        for step in range(m + 1):
            # the car's state at this step
            while j0 + 1 < nk and xs[j0 + 1] <= x:
                j0 += 1
            if j0 + 1 >= nk:
                break
            f = (x - xs[j0]) / ds if ds > 0 else 0.0
            ue = us[j0] + f * (us[j0 + 1] - us[j0])
            ne = ns[j0] + f * (ns[j0 + 1] - ns[j0])
            vp = vps[j0] + f * (vps[j0 + 1] - vps[j0])
            U.append(ue)
            N.append(ne)
            cap = INF
            for oid, gap0, ou, od, ov, in_tr in pl:
                dl = ue - (gap0 + ou[step] - shift[oid])
                dl = (dl + 0.5 * L) % L - 0.5 * L                    # car minus opponent, wrapped
                lat = abs(ne - od[step])
                if oid in free and 0.0 < dl < keep_ref and lat < opp_lane:
                    # the car, passing, is close ahead in this opponent's band: its driver slows behind the car
                    shift[oid] += keep_ref - dl
                    dl = keep_ref
                rel[oid].append(dl)
                if back_zone < dl < ahead_zone:
                    if oid not in t_a:
                        t_a[oid] = step
                    if lat < sep:
                        conflicts.append((step * dt, oid, "beside" if dl > -nose_tail else "behind"))
                if dl >= merge_need and oid in t_a and oid not in t_b:
                    t_b[oid] = step
                lp = lat_prev.get(oid)
                if lp is not None and lp >= band > lat and 0.0 < dl < merge_need:
                    conflicts.append((step * dt, oid, "merge"))
                if oid in free and oid in t_b and lat < band and 0.0 < dl < caught_need:
                    conflicts.append((step * dt, oid, "caught"))       # a passed opponent closes up behind
                lat_prev[oid] = lat
                held = hold is not None and oid == hold[0] and x < hold[1]
                if (held or (follow and lat < sep and oid not in free)) and -look < dl < -nose_tail:
                    e = -dl - (g_f_tr if in_tr else g_f)
                    vo = ov[step]
                    vc = min(vo + kf * e, math.sqrt(max(0.0, vo * vo + 2.0 * a_f * e)))
                    vc = min(vc, _predictive_cap(ou, step, -dl, g_min_tr if in_tr else g_min, a_f, dt, n_ant))
                    if vc < cap:
                        cap = max(0.0, vc)
            if stop_at is not None and step * dt >= stop_at:
                break
            # the tracker: the cap falls gently, the target rises at most at the traction limit
            if cap < INF:
                prev = cap_prev if cap_prev is not None else v
                cap_eff = max(cap, min(v, prev) - a_fall * dt)
                cap_prev = cap_eff
            else:
                cap_eff = INF
                cap_prev = None
            target = min(vp, cap_eff)
            if target > v:
                a_up = min(c_trac * f2 * mus[j0] * G, self.lim.a_motor(v))
                v = min(target, v + a_up * dt)
            else:
                v = target
            x += v * dt
            T.append((step + 1) * dt)
            V.append(v)
        return {"u": U, "n": N, "v": V, "conflicts": conflicts, "t_a": t_a, "t_b": t_b, "dt": dt, "rel": rel}

    # ------------------------------------------------------------------ per tick
    def update(self, truth, tick, t_s, base_line, base_prof):
        rp = self.rp
        ego = self._ego(truth)
        opps = self._opponents(truth, ego)
        by = {o["id"]: o for o in opps}
        L = self.plan.loop.length
        self.cur_opps, self.cur_tick = opps, tick
        # the active path: a pass (released past its release station, committed when released or beside an
        # opponent, done at its end), a follow path (on the line of the opponent it follows) or a return path
        act = self.active
        if act is not None:
            if act.s_last is not None:
                act.x += self.plan.loop.gap(ego["s"], act.s_last)
                act.s_last = ego["s"]
            if act.side in ("left", "right"):
                m_ov = float(rp["overlap_margin_m"])
                if act.x_release is not None and not act.released and act.x >= act.x_release and not act.blocked:
                    act.released = True                      # only after a check without a conflict
                if act.released:
                    act.committed = True
                for oid in act.cars:
                    o = by.get(oid)
                    if o is not None:
                        dl = self.plan.loop.gap(ego["s"], o["s"])
                        if -(EGO_FRONT_M + OPP_REAR_M + m_ov) < dl < EGO_REAR_M + OPP_FRONT_M + m_ov:
                            act.committed = True
            if (act.x_end is not None and act.x >= act.x_end) or (act.x_end is None and act.s_end_plan is not None
                                                                     and 0.0 <= self.plan.loop.gap(ego["s"], act.s_end_plan) < 5.0):
                if act.side in ("left", "right"):
                    self.n_passes += 1
                    self._event(tick, "done", act)
                self.active = self._successor(truth, ego, tick, act)
                act = self.active
            elif act.side == "follow" and (act.cars[0] not in by or self.plan.loop.gap(by[act.cars[0]]["s"], ego["s"]) < 0.0
                                           or self.plan.loop.gap(by[act.cars[0]]["s"], ego["s"]) > float(rp["follow_leave_m"])):
                self._event(tick, "leave", act)
                self.active = self._successor(truth, ego, tick, act)
                act = self.active
            elif act.side == "follow" and act.x_end is not None and act.x >= act.x_end:
                self.active = act = self._opponent_line_path(act.cars[0])        # on the opponent's own line
        # planning tick
        if tick >= self.next_pred_tick:
            self.next_pred_tick = tick + int(rp["plan_every_ticks"])
            self.preds = self._predict(opps)
            self.pred_tick = tick
            preds = self.preds
            if act is None or act.side in ("return", "follow"):
                line = act.line if act is not None else base_line
                prof = act.prof if act is not None else base_prof
                sim = self._simulate(line, prof.v, prof.mu, ego, preds, follow=False)
                seen = []
                for _t, oid, _kind in sim["conflicts"]:
                    if oid not in seen and oid in by and by[oid]["gap0"] > 0.0:
                        seen.append(oid)
                self.blockers = seen
                if seen:
                    first = seen[0]
                    g = by[first]["gap0"]
                    if ((act is None or act.side != "follow" or act.cars[0] != first) and g < float(rp["follow_engage_m"])
                            and tick >= self.next_follow_tick):
                        fp = self._follow_path(truth, ego, by[first], tick, cur=line.offsets)
                        if fp is not None:
                            self.active = act = fp
                        else:
                            self.next_follow_tick = tick + int(rp["follow_retry_ticks"])
                    if tick >= self.next_pass_tick and self._may_pass(by.get(first), t_s):
                        self.next_pass_tick = tick + int(rp["pass_every_ticks"])
                        cur = act.line.offsets if act is not None else line.offsets
                        new = self._best_pass(truth, ego, by, preds, first, tick, cur=cur)
                        if new is not None:
                            self.active = act = new
            else:
                hold = None
                if act.x_release is not None and not act.released:
                    hold = (act.hold_id, max(0.0 if not act.blocked else 1.0, act.x_release - act.x))
                sim = self._simulate(act.line, act.prof.v, act.prof.mu, ego, preds, follow=True, hold=hold,
                                     free=act.cars)
                tbs = [sim["t_b"].get(oid) for oid in act.cars if oid in preds]
                t_cut = (max(tbs) * sim["dt"] if tbs and all(x is not None for x in tbs) else INF) + float(rp["after_pass_s"])
                bad = [c for c in sim["conflicts"] if c[1] in act.cars or c[0] <= t_cut]
                if bad and not act.committed and bad[0][0] > float(rp["keep_plan_s"]):
                    # the first conflict is far ahead: rebuild the pass on its side, else keep it for now (the car is
                    # still held behind the opponent; the next checks decide)
                    new = self._plan_pass(truth, ego, by, preds, act.cars, act.side, tick, cur=act.line.offsets)
                    if new is not None:
                        self.n_rebuild += 1
                        self._event(tick, "rebuild", new, bad[0])
                        self.active = act = new
                    else:
                        self._why("kept")
                        act.blocked = True                   # kept, but the car stays held until a clean check
                    bad = []
                elif not bad:
                    act.blocked = False
                if bad:
                    self.blockers = []
                    if not act.committed:
                        self.n_drop += 1
                        self._event(tick, "drop", act, bad[0])
                        new = self._best_pass(truth, ego, by, preds, act.cars[0], tick, cur=act.line.offsets)
                        if new is None and act.cars[0] in by:
                            new = self._follow_path(truth, ego, by[act.cars[0]], tick, cur=act.line.offsets)
                        if new is None:
                            new = self._return_path(truth, ego, tick, from_path=act.line)
                        if new is None:
                            act.blocked = True                   # no valid successor: stay on it, held behind
                            new = act
                        self.active = new
                    else:
                        new = self._plan_pass(truth, ego, by, preds, act.cars, act.side, tick, released=True,
                                              cur=act.line.offsets)
                        if new is not None:
                            new.committed = new.released = True
                            self.n_rebuild += 1
                            self._event(tick, "rebuild", new, bad[0])
                            self.active = new
                        else:
                            self._why("rebuild_failed")
                    act = self.active
        # caps: following (with the opponent's predicted braking), emergency
        cap = None
        look = max(float(rp["look_min_m"]), float(rp["look_s"]) * ego["v"])
        sep = float(rp["sep_m"])
        nose_tail = EGO_FRONT_M + OPP_REAR_M
        on_pass = act is not None and act.side in ("left", "right")
        for o in opps:
            dl = self.plan.loop.gap(ego["s"], o["s"])
            if not (-look < dl < -nose_tail + 0.05):
                continue
            lat = abs(ego["d"] - o["d"])
            if on_pass and o["id"] in act.cars:
                hold = o["id"] == act.hold_id and not act.released       # held until the release station
            elif on_pass:
                hold = lat < sep
            else:
                hold = o["id"] in self.blockers or lat < sep
            if hold:
                vc = self._follow_cap(o, -dl, tick)
                cap = vc if cap is None else min(cap, vc)
            gap_b = -dl - nose_tail
            if 0.0 < gap_b < float(rp["emergency_gap_m"]) and lat < CAR_WIDTH_M + 0.05:
                vc = max(0.0, o["v"] - 0.3)
                cap = vc if cap is None else min(cap, vc)
                self.n_emergency += 1
        if cap is not None:
            self.n_follow_ticks += 1
        return cap, self.active

    def _follow_cap(self, o, g, tick):
        """The following law behind opponent o at reference-point gap g, bounded by the opponent's predicted motion:
        braking gently from now, the car stays `follow_min_m` behind every predicted position of the opponent over
        the next `anticipate_s` (the opponent brakes behind a slower car sooner than the gap shows)."""
        rp = self.rp
        vo = o["v"]
        p = self.preds.get(o["id"]) if self.preds is not None else None
        train = p is not None and p.in_train
        g_f = float(rp["follow_gap_train_m" if train else "follow_gap_m"])
        g_min = float(rp["follow_min_train_m" if train else "follow_min_m"])
        e = g - g_f
        a_f = float(rp["follow_decel_frac"]) * self.base_mu * G
        cap = max(0.0, min(vo + float(rp["follow_gain_per_s"]) * e, math.sqrt(max(0.0, vo * vo + 2.0 * a_f * e))))
        if p is not None:
            dt = float(rp["dt_s"])
            k0 = int(round((tick - self.pred_tick) * 0.025 / dt))
            cap = min(cap, _predictive_cap(p.u, k0, g, g_min, a_f, dt, int(round(float(rp["anticipate_s"]) / dt))))
        return cap

    def _valid(self, line, prof, ego, horizon_s=None):
        """True when the car, on this path with the following law behind every opponent ahead in its band, keeps
        every rule of the module text against every opponent over the next `check_s` seconds."""
        if self.cur_opps is None:
            return True
        if self.preds is None or self.pred_tick != self.cur_tick:
            self.preds = self._predict(self.cur_opps)             # predictions of this tick
            self.pred_tick = self.cur_tick
        sim = self._simulate(line, prof.v, prof.mu, ego, self.preds, follow=True)
        lim = float(self.rp["check_s"]) if horizon_s is None else horizon_s
        return not any(c[0] <= lim for c in sim["conflicts"])

    def _may_pass(self, o, t_s):
        rp = self.rp
        if o is None:
            return False
        if t_s < float(rp["start_hold_s"]):
            self._why("start_hold")
            return False
        if o["id"] in self.up_to_pace:
            return True
        vc = o["opp"].cruise(o["sp"])
        if o["v"] < float(rp["start_cruise_frac"]) * vc:
            self._why("start_bunch")
            return False
        self.up_to_pace.add(o["id"])                  # the start is over for this opponent
        return True

    def _why(self, key):
        self.why[key] = self.why.get(key, 0) + 1

    def _event(self, tick, kind, path, extra=None):
        if len(self.events) < 2000:
            e = {"tick": tick, "event": kind, "cars": list(path.cars) if path is not None else [],
                 "side": path.side if path is not None else None}
            if extra is not None:
                e["conflict"] = [round(extra[0], 2), extra[1], extra[2]]
            if path is not None and path.stretch is not None:
                e["stretch_m"] = [round(path.stretch[0], 2), round(path.stretch[1], 2)]
            self.events.append(e)

    # ------------------------------------------------------------------ following on an opponent's line
    def _ramp_len(self, delta, v):
        rp = self.rp
        a_lat = float(rp["ramp_lat_frac"]) * self.factor ** 2 * self.base_mu * G
        v_r = max(v, 1.0)
        return min(float(rp["ramp_max_m"]), max(float(rp["ramp_min_m"]), math.sqrt(6.0 * abs(delta) * v_r * v_r / a_lat)))

    def _offsets_from_car(self, ego, target, L_in, behind=None):
        """Offsets per plan station: `target` (an array over the plan stations), reached from the car's present
        offset by a smoothstep over L_in; behind the car the shape of the path it is on (`behind`, offsets per plan
        station) is kept over 1 m, shifted onto the car (no kink, so the new path's profile is not cut at the car)."""
        plan = self.plan
        n, ds = plan.n, plan.loop.ds
        s_e, d_e = ego["s"], ego["d"]
        k_e = int(math.floor(s_e / ds))
        offs = np.asarray(target, dtype=float).copy()
        kbh = int(round(1.0 / ds))
        if behind is None:
            behind = np.full(n, d_e)
        shift = d_e - plan.loop.interp(behind, s_e)
        for j in range(-kbh, 0):
            k = (k_e + j) % n
            offs[k] = behind[k] + shift
        k_in = int(math.ceil(L_in / ds))
        for j in range(0, k_in + 1):
            k = (k_e + j) % n
            u = min(1.0, max(0.0, (k_e + j) * ds - s_e) / max(L_in, 1e-6))
            d_from = d_e
            offs[k] = d_from + (target[k] - d_from) * _smoothstep(u)
        return offs, k_e, k_in

    def _follow_path(self, truth, ego, o, tick, cur=None):
        """A path onto the line of opponent o (the car follows it there, directly behind it)."""
        plan = self.plan
        target = o["opp"].d_at_plan
        delta = ego["d"] - plan.loop.interp(target, ego["s"])
        L_in = self._ramp_len(delta, ego["v"])
        offs, _k_e, _k_in = self._offsets_from_car(ego, target, L_in, behind=cur)
        offs = np.clip(offs, self.lo, self.hi)
        line = Line(plan, offs)
        _sj, _dj, ij = line.loop.project(ego["rx"], ego["ry"])
        base = self._opponent_line(o["id"])[1]
        prof = local_profile(plan, line, self.factor, ij, ego["v"], L_in + float(self.rp["profile_extra_m"]), base)
        if float(prof.v[ij]) < ego["v"] - float(self.rp["entry_slack_mps"]):
            self._why("follow_too_fast")
            return None
        if not self._valid(line, prof, ego):
            self._why("follow_conflict")
            return None
        self.n_plans += 1
        p = PassPath(line, prof, "follow%d" % self.n_plans, [o["id"]], "follow", None, None, INF, s_start=ego["s"],
                     x_end=L_in + 2.0)
        self._event(tick, "follow", p)
        return p

    def _opponent_line(self, oid):
        """(line, profile) of an opponent's own line within the car's limits (built once per opponent)."""
        lp = self.line_paths.get(oid)
        if lp is None:
            line = Line(self.plan, np.clip(self.by_id[oid].d_at_plan, self.lo, self.hi))
            lp = (line, self.plan.profile(line, self.factor))
            self.line_paths[oid] = lp
        return lp

    def _opponent_line_path(self, oid):
        """The follow path on an opponent's own line, without a ramp."""
        line, prof = self._opponent_line(oid)
        return PassPath(line, prof, "on_%s" % oid, [oid], "follow", None, None, INF)

    # ------------------------------------------------------------------ planning a pass
    def _group(self, by, first):
        """The opponent in the way and those running less than train_gap_m ahead of it, nearest first."""
        L = self.plan.loop.length
        tg = float(self.rp["train_gap_m"])
        group = [first]
        cur = by[first]
        while len(group) < len(by):
            nxt, best = None, INF
            for oid, o in by.items():
                if oid in group:
                    continue
                g = (o["s"] - cur["s"]) % L
                if 0.0 < g < best:
                    nxt, best = oid, g
            if nxt is None or best > tg:
                break
            group.append(nxt)
            cur = by[nxt]
        return group

    def _best_pass(self, truth, ego, by, preds, first, tick, cur=None):
        if first not in by:
            return None
        group = self._group(by, first)
        best = None
        for side in ("left", "right"):
            cand = self._plan_pass(truth, ego, by, preds, group, side, tick, cur=cur)
            if cand is not None and (best is None or cand.t_done < best.t_done):
                best = cand
        if best is not None:
            self.n_plans += 1
            self._event(tick, "plan", best)
        return best

    def _plan_pass(self, truth, ego, by, preds, group, side, tick, released=False, cur=None):
        """A pass of the opponents of `group` (nearest first) on `side`, or None (the reason counted in `why`):
        the lines of the group offset by sep_m to that side (their envelope), reached from the car's offset while
        it is held behind the first of them, kept until the simulated merge point, then a ramp to the racing line."""
        rp = self.rp
        plan = self.plan
        n = plan.n
        ds = plan.loop.ds
        L = plan.loop.length
        group = [g for g in group if g in preds]
        if not group:
            return None
        n_line = self.line.offsets
        sep = float(rp["sep_m"]) + float(rp["dev_margin_m"])
        sgn = 1.0 if side == "left" else -1.0
        s_e, d_e, v_e = ego["s"], ego["d"], ego["v"]
        # the offset envelope of the group's lines (and their present tracking error on the car's side)
        env = None
        for oid in group:
            st = by[oid]
            dl = st["opp"].d_at_plan + (max(0.0, st["dev"]) if side == "left" else min(0.0, st["dev"]))
            env = dl if env is None else (np.maximum(env, dl) if side == "left" else np.minimum(env, dl))
        target_w = env + sgn * sep
        slack = float(rp["room_slack_m"])
        lead_line = by[group[0]]["opp"].d_at_plan
        rel0 = d_e - plan.loop.interp(lead_line, s_e)
        L_in = 0.0 if released else self._ramp_len(sgn * sep - rel0, v_e)
        k_e = int(math.floor(s_e / ds))
        ka = k_e + int(math.ceil(L_in / ds))
        B_rel = L_in + float(rp["pass_window0_m"])
        wins = None                                      # per opponent: where the car is beside it (simulated)
        for build in range(int(rp["max_builds"])):
            if B_rel > float(rp["stretch_max_frac"]) * L:
                self._why("too_long")
                return None
            kb = k_e + int(math.ceil(B_rel / ds))
            kw = np.arange(ka, kb + 1) % n
            if wins is None:
                tw = target_w[kw]
            else:
                tw = self._target_by_windows(by, group, wins, side, sep, k_e, ka, kb)
            if side == "left":
                over = tw > self.hi[kw] + slack
            else:
                over = tw < self.lo[kw] - slack
            if np.any(over):
                if wins is None and len(group) > 1:
                    # a train: each line only where the car is beside that car (from the estimate)
                    est = self._simulate(self.line, self.prof.v, self.prof.mu, ego, preds, follow=True,
                                         hold=(group[0], ka * ds - s_e), free=group)
                    if all(est["t_a"].get(g) is not None and est["t_b"].get(g) is not None for g in group):
                        wins = {g: (est["u"][est["t_a"][g]], est["u"][est["t_b"][g]]) for g in group}
                        B_rel = max(B_rel, max(w[1] for w in wins.values()) + float(rp["merge_extra_m"]))
                        continue
                self._why("no_room")
                return None
            # the target over the stretch, the racing line elsewhere; the ramp in from the car's offset
            tgt = n_line.copy()
            tgt[kw] = np.minimum(np.maximum(tw, self.lo[kw]), self.hi[kw])
            if not released:
                # while held: from the car's offset relative to the leading line to the side's offset
                kk = np.arange(k_e, ka + 1) % n
                tgt[kk] = np.minimum(np.maximum(target_w[kk], self.lo[kk]), self.hi[kk])
            offs, _k0, _kin = self._offsets_from_car(ego, tgt, max(L_in, float(rp["ramp_min_m"])), behind=cur)
            T1 = float(offs[kb % n])
            Lo = self._ramp_len(T1 - n_line[(kb + 1) % n], v_e)
            ko = int(math.ceil(Lo / ds))
            for j in range(1, ko + 1):                                 # ramp out: back to the racing line
                k = (kb + j) % n
                w = 1.0 - _smoothstep(j / ko)
                offs[k] = n_line[k] + (T1 - n_line[k]) * w
            near = np.zeros(n, dtype=bool)
            near[(k_e + np.arange(-int(round(1.0 / ds)), int(round(0.5 / ds)))) % n] = True
            offs = np.where(near, offs, np.clip(offs, self.lo, self.hi))
            line = Line(plan, offs)
            _sj, _dj, ij = line.loop.project(ego["rx"], ego["ry"])
            reach = (kb + ko) * ds - s_e + float(rp["profile_extra_m"])
            prof = local_profile(plan, line, self.pass_factor, ij, v_e, reach, self.prof)
            if float(prof.v[ij]) < v_e - float(rp["entry_slack_mps"]):
                self._why("too_fast")
                return None
            x_rel = 0.0 if released else max(0.0, ka * ds - s_e)
            sim = self._simulate(line, prof.v, prof.mu, ego, preds, follow=True, hold=(group[0], x_rel), free=group)
            tbs = [sim["t_b"].get(oid) for oid in group]
            t_done = max(tbs) * sim["dt"] if all(t is not None for t in tbs) else None
            t_cut = (t_done if t_done is not None else INF) + float(rp["after_pass_s"])
            mine = [c for c in sim["conflicts"] if c[1] in group]
            others = [c for c in sim["conflicts"] if c[1] not in group and c[0] <= t_cut]
            if not mine and not others and t_done is not None:
                u_b = max(sim["u"][tb] for tb in tbs)
                if u_b <= B_rel + 0.05:
                    return PassPath(line, prof, "pass%d" % (self.n_plans + 1), group, side, ((kb + ko) * ds) % L,
                                    (ka * ds - s_e, kb * ds - s_e), t_done, hold_id=group[0],
                                    s_release=None if released else (ka * ds) % L, s_start=s_e,
                                    x_release=None if released else ka * ds - s_e, x_end=(kb + ko) * ds - s_e)
            if others:
                oid = others[0][1]
                if oid in by and oid in preds and by[oid]["gap0"] > 0.0 and len(group) < len(self.opps):
                    group = group + [oid]
                    wins = None
                    st = by[oid]
                    dl = st["opp"].d_at_plan + (max(0.0, st["dev"]) if side == "left" else min(0.0, st["dev"]))
                    env = np.maximum(env, dl) if side == "left" else np.minimum(env, dl)
                    target_w = env + sgn * sep
                    continue
                self._why("conflict_other")
                return None
            if t_done is None:
                tbs_now = [sim["t_b"].get(oid) for oid in group]
                if any(t is None for t in tbs_now) and sim["t_a"]:
                    B_rel = B_rel + float(rp["extend_m"]) * 2.0
                    continue
                self._why("not_cleared")
                return None
            u_b = max(sim["u"][tb] for tb in tbs)
            B_rel = max(B_rel + float(rp["extend_m"]), u_b + float(rp["merge_extra_m"]))
            if len(group) > 1 and all(sim["t_a"].get(g) is not None for g in group):
                wins = {g: (sim["u"][sim["t_a"][g]], sim["u"][sim["t_b"][g]]) for g in group}
        self._why("builds")
        return None

    def _target_by_windows(self, by, group, wins, side, sep, k_e, ka, kb):
        """The pass target over plan indices ka..kb (unwrapped): each opponent's line offset by sep where the car
        is beside it (its window, widened by window_margin_m), filled linearly between the windows."""
        rp = self.rp
        ds = self.plan.loop.ds
        n = self.plan.n
        m = float(rp["window_margin_m"])
        tw = np.full(kb - ka + 1, np.nan)
        for g in group:
            st = by[g]
            dl = st["opp"].d_at_plan + (max(0.0, st["dev"]) if side == "left" else min(0.0, st["dev"]))
            ua, ub = wins[g]
            ja = max(ka, k_e + int(math.floor((ua - m) / ds)))
            jb = min(kb, k_e + int(math.ceil((ub + m) / ds)))
            if jb < ja:
                continue
            ks = np.arange(ja, jb + 1)
            val = dl[ks % n] + (sep if side == "left" else -sep)
            pos = ks - ka
            cur = tw[pos]
            if side == "left":
                tw[pos] = np.where(np.isnan(cur), val, np.maximum(cur, val))
            else:
                tw[pos] = np.where(np.isnan(cur), val, np.minimum(cur, val))
        if np.all(np.isnan(tw)):
            return np.full(kb - ka + 1, np.inf if side == "left" else -np.inf)
        return _fill_nan(tw)

    def _successor(self, truth, ego, tick, act):
        """After a pass, a follow path or a return path ends: a return path to the racing line when the car is off
        it, else nothing (the racing line); when the return path is not valid the present path is kept for 3 m more."""
        ret = self._return_path(truth, ego, tick, from_path=act.line)
        if ret is not None:
            return ret
        dev = abs(ego["d"] - self.plan.loop.interp(self.line.offsets, ego["s"]))
        if dev <= float(self.rp["on_line_m"]):
            return None
        act.side = "return"
        act.cars = []
        act.x_end = act.x + 3.0
        self._why("kept_path")
        return act

    def _return_path(self, truth, ego, tick, from_path=None):
        """A ramp back to the racing line, from the car's present offset or, with `from_path`, from that path's
        shape (offset and heading stay continuous); None when the car is on the line already."""
        rp = self.rp
        plan = self.plan
        n, ds, L = plan.n, plan.loop.ds, plan.loop.length
        n_line = self.line.offsets
        s_e, d_e, v_e = ego["s"], ego["d"], ego["v"]
        k0 = int(math.floor(s_e / ds))
        v_r = max(v_e, 1.0)
        look = (k0 + np.arange(0, int(round(3.0 / ds)))) % n
        dev_car = d_e - plan.loop.interp(n_line, s_e)
        if from_path is not None:
            dev_arr = from_path.offsets - n_line
            dev = float(np.max(np.abs(dev_arr[look])))
        else:
            dev_arr = None
            dev = abs(dev_car)
        if dev < float(rp["late_dev_m"]):
            return None
        a_ret = float(rp["return_lat_frac"]) * self.factor ** 2 * self.base_mu * G
        Lr = min(float(rp["return_ramp_max_m"]), max(float(rp["ramp_min_m"]), math.sqrt(6.0 * dev * v_r * v_r / a_ret)))
        offs = n_line.copy()
        kb = int(round(1.0 / ds))
        behind = from_path.offsets if from_path is not None else np.full(n, d_e)
        shift = d_e - plan.loop.interp(behind, s_e)
        for j in range(-kb, 0):                         # behind the car: the shape of the path it is on
            k = (k0 + j) % n
            offs[k] = behind[k] + shift
        kr = int(math.ceil(Lr / ds))
        kblend = max(1, int(round(1.0 / ds)))
        for j in range(0, kr + 1):
            w = 1.0 - _smoothstep(j / kr)
            k = (k0 + j) % n
            if dev_arr is not None:
                b = _smoothstep(min(1.0, j / kblend))
                base = dev_car + (dev_arr[k] - dev_car) * b
            else:
                base = dev_car
            offs[k] = n_line[k] + base * w
        near = np.zeros(n, dtype=bool)
        near[(k0 + np.arange(-kb, int(round(0.5 / ds)))) % n] = True
        offs = np.where(near, offs, np.clip(offs, self.lo, self.hi))
        line = Line(plan, offs)
        _sj, _dj, ij = line.loop.project(ego["rx"], ego["ry"])
        prof = local_profile(plan, line, self.factor, ij, v_e, Lr + float(rp["profile_extra_m"]), self.prof)
        if not self._valid(line, prof, ego):
            self._why("return_conflict")
            return None
        self.n_plans += 1
        p = PassPath(line, prof, "ret%d" % self.n_plans, [], "return", ((k0 + kr) * ds) % L, None, INF, s_start=s_e,
                     x_end=(k0 + kr) * ds - s_e)
        self._event(tick, "return", p)
        return p

    def summary(self):
        return {"passes_done": self.n_passes, "plans": self.n_plans, "drops": self.n_drop, "rebuilds": self.n_rebuild,
                "follow_ticks": self.n_follow_ticks, "emergency_ticks": self.n_emergency, "why_not": dict(self.why),
                "events": self.events[:150] + (self.events[-100:] if len(self.events) > 150 else [])}
