"""The two sensor models of the hidden world, the inertial unit and the fixes of the localiser, driven from the truth.


























"""
import collections
import hashlib
import math

import numpy as np

from . import rot


def keyed_seed(key, *parts):
    h = hashlib.blake2b(digest_size=8, key=key)
    h.update("|".join(str(p) for p in parts).encode())
    return int.from_bytes(h.digest(), "little")


def stream(key, *parts):
    return np.random.Generator(np.random.PCG64(keyed_seed(key, *parts)))


class AccelWander:
    """The accelerometer offset's slow change:
    per sensor axis a first-order process about the session's constant offset, generated exactly on the samples' own
    (irregular) times, x_k = a_k x_(k-1) + sd sqrt(1 - a_k^2) n_k with a_k = exp(-(t_k - t_(k-1)) / tau), started
    from its stationary distribution (x_0 = sd n_0) and drawn from its own keyed-hash stream. Its change over any one
    minute is clipped at the entry's value: every new value is clipped into [max - c, min + c] of the values of the
    last minute (a running maximum and minimum), so that any two values at most one minute apart differ by at most c.
    The offset has no linear ramp."""

    def __init__(self, rng, tau_s, sd, clip, t0, window_s=60.0):
        self.rng = rng
        self.tau = float(tau_s)
        self.sd = float(sd)
        self.c = float(clip)
        self.Wn = float(window_s)
        self.t = float(t0)
        self.x = self.sd * rng.standard_normal(3)
        self.hi = [collections.deque() for _ in range(3)]    # (t, x) with decreasing x: the window's maximum first
        self.lo = [collections.deque() for _ in range(3)]    # (t, x) with increasing x: the window's minimum first
        self.n = 0
        self.n_clipped = 0
        self.trace = None                                   # set to a list to record (t, x) (tests)
        self._push(self.t, self.x)

    def _push(self, t, x):
        for i in range(3):
            xi = float(x[i])
            h, lq = self.hi[i], self.lo[i]
            while h and h[-1][1] <= xi:
                h.pop()
            h.append((t, xi))
            while lq and lq[-1][1] >= xi:
                lq.pop()
            lq.append((t, xi))
        if self.trace is not None:
            self.trace.append((t, x.copy()))

    def at(self, t):
        """the wander at the sample time t (t later than the previous call)."""
        dt = t - self.t
        a = math.exp(-dt / self.tau)
        x = a * self.x + self.sd * math.sqrt(max(0.0, 1.0 - a * a)) * self.rng.standard_normal(3)
        t_old = t - self.Wn
        clipped = False
        for i in range(3):
            h, lq = self.hi[i], self.lo[i]
            while h and h[0][0] < t_old - 1e-12:
                h.popleft()
            while lq and lq[0][0] < t_old - 1e-12:
                lq.popleft()
            if h:
                b_lo, b_hi = h[0][1] - self.c, lq[0][1] + self.c
                if x[i] < b_lo:
                    x[i] = b_lo
                    clipped = True
                elif x[i] > b_hi:
                    x[i] = b_hi
                    clipped = True
        self.n += 1
        self.n_clipped += int(clipped)
        self.t = t
        self.x = x
        self._push(t, x)
        return x


class InertialUnit:
    def __init__(self, rng, intervals, start_index, mount_pos_body, R_mount, com, window_s, accel_offset_sensor,
                 wander, gyro_offset_body, noise_by_level, t0):
        """wander: an AccelWander (the offset's slow change on the samples' times) or None (a constant offset)."""
        self.rng = rng
        self.iv = intervals
        self.k = int(start_index) % len(intervals)
        self.r = np.asarray(mount_pos_body, float) - np.asarray(com, float)   # lever arm about the centre of mass
        self.Rm = np.asarray(R_mount, float)          # v_body = Rm v_sensor
        self.RmT = self.Rm.T
        self.W = float(window_s)
        self.ba0 = np.asarray(accel_offset_sensor, float)
        self.wander = wander
        self.bg = np.asarray(gyro_offset_body, float)
        self.noise = noise_by_level                   # level -> (gyro sd (3,), accel sd (3,)) sensor axes
        self.next_t = t0 + self.iv[self.k]
        self.k = (self.k + 1) % len(self.iv)
        self.t0 = t0
        self.hist = []                                # (t_start, t_end, f_unit_body (3,), w_start (3,), w_end (3,))

    def push(self, t_start, t_end, fs_cm, alpha, w_start, w_end):
        """record one step of truth: specific force at the centre of mass (body), angular acceleration, inertial rate
        at the step's ends. The specific force at the unit adds alpha x r + w x (w x r) (w at the step's middle)."""
        wm = 0.5 * (w_start + w_end)
        f_unit = fs_cm + np.cross(alpha, self.r) + np.cross(wm, np.cross(wm, self.r))
        self.hist.append((t_start, t_end, f_unit, np.asarray(w_start, float), np.asarray(w_end, float)))
        while len(self.hist) > 2 and self.hist[1][1] < t_end - 2 * self.W:
            self.hist.pop(0)

    def _avg(self, t):
        a0 = t - self.W
        F = np.zeros(3); Wv = np.zeros(3); tot = 0.0
        for (ts, te, f, w0, w1) in self.hist:
            lo, hi = max(ts, a0), min(te, t)
            if hi <= lo:
                continue
            L = hi - lo
            F += f * L
            # mean of the linear rate over [lo, hi]
            fl = (lo - ts) / (te - ts); fh = (hi - ts) / (te - ts)
            wl = w0 + (w1 - w0) * fl; wh = w0 + (w1 - w0) * fh
            Wv += 0.5 * (wl + wh) * L
            tot += L
        if tot <= 0:
            return np.zeros(3), np.zeros(3)
        return F / tot, Wv / tot

    def samples_until(self, t_now, level):
        out = []
        gsd, asd = self.noise[level]
        while self.next_t <= t_now + 1e-12:
            t = self.next_t
            f, w = self._avg(t)
            ba = self.ba0 + self.wander.at(t) if self.wander is not None else self.ba0
            rate = self.RmT @ (w + self.bg) + gsd * self.rng.standard_normal(3)
            acc = self.RmT @ f + ba + asd * self.rng.standard_normal(3)
            out.append({"stamp": float(t), "rate": rate, "accel": acc})
            self.next_t = t + self.iv[self.k]
            self.k = (self.k + 1) % len(self.iv)
        return out


def gross_error(rng, pos_range, att_range_rad):
    """the whole error of a grossly wrong fix (world.fix.gross): a position offset of uniform size in
    pos_range along a uniform direction and an attitude error of uniform angle in att_range_rad about a uniform axis.
    The draws are taken in this order: direction, size, axis, angle."""
    d = rng.standard_normal(3); d /= np.linalg.norm(d)
    dp = d * rng.uniform(*pos_range)
    e = rng.standard_normal(3); e /= np.linalg.norm(e)
    dq = rot.qexp(e * rng.uniform(*att_range_rad))
    return dp, dq


class SuccessProcess:
    """the two-state success process (tracking, outage; exponential stretches; in tracking each attempt
    succeeds with probability p_track). Its parameters are stand-ins for world.fix.success_process while that
    entry has no value; the draws are taken in a fixed order."""

    def __init__(self, rng, p, t0):
        self.rng, self.p = rng, p
        self.state_on = rng.random() < p["mean_on_s"] / (p["mean_on_s"] + p["mean_off_s"])
        self.state_until = t0 + rng.exponential(p["mean_on_s"] if self.state_on else p["mean_off_s"])

    def advance(self, t):
        while t >= self.state_until:
            self.state_on = not self.state_on
            self.state_until += self.rng.exponential(self.p["mean_on_s"] if self.state_on else self.p["mean_off_s"])

    def attempt(self, t_img):
        self.advance(t_img)
        return self.state_on and (self.rng.random() < self.p["p_track"])

    def forced(self, t_img):
        """a forced success puts the process into tracking."""
        self.state_on = True
        self.state_until = t_img + self.rng.exponential(self.p["mean_on_s"])


class FourStatProcess:
    """the replacement of the stand-in (world.fix.success_process): two states per session. Tracking
    lasts a geometric number of attempts (mean n_on) in which each attempt succeeds with probability p_track; an
    outage lasts a time drawn uniformly on [off_lo_s, off_hi_s], with off_hi_s at most the day's longest stretch less
    one attempt interval; and an attempt that would leave the next one more than the day's longest stretch after the
    last success succeeds, so that no stretch outside the scored blocks exceeds the day's longest. Its four parameters
    are fitted by fit_four_stat to the day's rate, median and 95th percentile of the gap and longest stretch; the
    targets are 's (they need the ratio of corrections to fixes from the records). p also holds longest_s and
    attempt_max_s (the largest attempt interval of the delay tables)."""

    def __init__(self, rng, p, t0):
        self.rng, self.p = rng, p
        self.state_on = True
        self.left_on = int(rng.geometric(1.0 / max(1.0, p["n_on"])))
        self.off_until = -1e18
        self.last_ok = float(t0)

    def attempt(self, t_img):
        p, rng = self.p, self.rng
        if not self.state_on:
            if t_img < self.off_until:
                return self._bound(t_img, False)
            self.state_on = True
            self.left_on = int(rng.geometric(1.0 / max(1.0, p["n_on"])))
        ok = bool(rng.random() < p["p_track"])
        self.left_on -= 1
        if self.left_on <= 0:
            self.state_on = False
            if p.get("q_long") is not None:
                # the two-part law (short and long outages), the same two-state structure
                if rng.random() < p["q_long"]:
                    self.off_until = t_img + float(rng.uniform(p["long_lo_s"], p["long_hi_s"]))
                else:
                    self.off_until = t_img + float(rng.uniform(p["off_lo_s"], p["off_hi_s"]))
            else:
                self.off_until = t_img + float(rng.uniform(p["off_lo_s"], p["off_hi_s"]))
        return self._bound(t_img, ok)

    def _bound(self, t_img, ok):
        if not ok and t_img + self.p["attempt_max_s"] - self.last_ok > self.p["longest_s"]:
            ok = True                       # the day's longest stretch is never exceeded
            self.state_on = True
            self.off_until = -1e18
            self.left_on = max(self.left_on, 1)
        if ok:
            self.last_ok = t_img
        return ok

    def forced(self, t_img):
        self.state_on = True
        self.off_until = -1e18
        self.left_on = max(self.left_on, 1)
        self.last_ok = t_img


class FixProcess:
    def __init__(self, rng, tables, p, t0, gross_whole_error=True, process="stand_in", after_outage=None):
        """p: dict(p_track, mean_on_s, mean_off_s, gross_lt20, gross_ge20, gross_pos (lo, hi), gross_att_rad (lo, hi),
        noise_ge20 (pos, att), noise_lt20 (pos, att), cam_t (3,), cam_q (4,)); for process="four_statistics" also
        n_on, off_lo_s, off_hi_s (and q_long, long_lo_s, long_hi_s for the two-part law). gross_whole_error=False gives
        additive gross offsets (comparison). after_outage: None (no draw) or dict(rng, share_lt20,
        min_stretch_s): the first successful fix after a stretch of at least min_stretch_s since the previous
        successful image is drawn with fewer than 20 landmarks with probability share_lt20, from its own stream rng."""
        self.rng = rng
        self.ao = after_outage
        if after_outage is not None:
            lm = np.asarray(tables["fix_landmarks"])
            self.ao_lt, self.ao_ge = lm[lm < 20], lm[lm >= 20]
            self.ao_any_success = False
            self.ao_log = []       # (t_img, stretch_s, landmarks) of every first fix after an outage
        self.T = tables
        self.p = p
        self.gross_whole_error = bool(gross_whole_error)
        self.sp = (FourStatProcess if process == "four_statistics" else SuccessProcess)(rng, p, t0)
        self.camera_id = 1000 + int(rng.integers(0, 1000))
        self.pending = []          # results not yet delivered: (arrival, fix dict)
        self.regs = []             # notices not yet delivered: (arrival, reg dict)
        self.next_image = t0 + float(rng.uniform(0.0, 1.0))
        self.last_success_arrival = t0
        self.last_success_img = t0
        self.log = []              # every attempt (for the tests and the logs)

    @property
    def state_on(self):
        return self.sp.state_on

    def _class_draws(self, r, n, pc_true, qc_true):
        """the ordinary noise of the fix's class and the gross draw (the draws in their fixed order), from the stream r."""
        p = self.p
        lt = n < 20
        sp, sa = p["noise_lt20"] if lt else p["noise_ge20"]
        pc = pc_true + sp * r.standard_normal(3)
        qc = rot.qmul(rot.qexp(sa * r.standard_normal(3)), qc_true)
        gross = False
        pg = p["gross_lt20"] if lt else p["gross_ge20"]
        if r.random() < pg:
            gross = True
            dp, dq = gross_error(r, p["gross_pos"], p["gross_att_rad"])
            if self.gross_whole_error:
                # the gross offsets are the fix's whole error (the ordinary noise drawn above is replaced, not
                # added to)
                pc = pc_true + dp
                qc = rot.qmul(dq, qc_true)
            else:
                pc = pc + dp
                qc = rot.qmul(dq, qc)
        return pc, qc, gross

    def step(self, t_now, truth_at, cap):
        """take images due by t_now; return (registrations, fixes) arriving in (t_prev, t_now]. cap: a function of
        the image time giving the cap in force (None where none is), or a number / None for every image of this call
        (the per-frame form). Where a cap c is in force, an attempt that would leave the next attempt more
        than c after the last successful image succeeds (forced); under c = 0 every attempt succeeds."""
        rng, p = self.rng, self.p
        cap_fn = cap if callable(cap) else (lambda _t, _c=cap: _c)
        while self.next_image <= t_now + 1e-12:
            t_img = self.next_image
            success = self.sp.attempt(t_img)
            if success:
                tau = float(rng.choice(self.T["fix_delay_success_s"]))
            else:
                tau = float(rng.choice(self.T["fix_delay_fail_s"]))
            forced = False
            cap_s = cap_fn(t_img)
            if cap_s is not None and not success:
                # where the cap is in force the stretch between the images of two successful fixes never exceeds it:
                # an attempt that would leave the next attempt beyond the cap succeeds
                if t_img + tau - self.last_success_img > cap_s:
                    success = True
                    forced = True
                    tau = float(rng.choice(self.T["fix_delay_success_s"]))
                    self.sp.forced(t_img)
            self.camera_id += 1
            cid = self.camera_id
            r_off = float(rng.choice(self.T["registration_stamp_offset_s"]))
            r_lag = max(0.0, float(rng.choice(self.T["registration_arrival_lag_s"])))
            self.regs.append((t_img + r_off + r_lag, {"camera_id": cid, "stamp": t_img + r_off}))
            fix = {"camera_id": cid, "stamp": t_img, "landmarks": 0, "position": None, "attitude": None}
            gross = False
            if success:
                n = int(rng.choice(self.T["fix_landmarks"]))
                pb, qb = truth_at(t_img)
                R = rot.qmat(qb)
                pc_true = pb + R @ p["cam_t"]
                qc_true = rot.qmul(qb, p["cam_q"])
                pc, qc, gross = self._class_draws(rng, n, pc_true, qc_true)
                if self.ao is not None:
                    # the first fix after an outage (the stretch since the previous successful image). The main
                    # stream's draws above are taken as without the hook (and discarded for this fix), so every later draw keeps
                    # its order; the fix's class, count, noise and gross draw come from the hook's own stream
                    if self.ao_any_success and t_img - self.last_success_img >= self.ao["min_stretch_s"]:
                        ra = self.ao["rng"]
                        sub = self.ao_lt if ra.random() < self.ao["share_lt20"] else self.ao_ge
                        n = int(sub[int(ra.integers(0, len(sub)))])
                        pc, qc, gross = self._class_draws(ra, n, pc_true, qc_true)
                        self.ao_log.append((t_img, t_img - self.last_success_img, n))
                    self.ao_any_success = True
                qc = rot.qnormalize(qc)
                fix.update(landmarks=n, position=pc, attitude=qc if qc[3] >= 0 else -qc)
                self.last_success_arrival = t_img + tau
                self.last_success_img = t_img
            self.pending.append((t_img + tau, fix))
            self.log.append((t_img, tau, success, forced, gross, fix["landmarks"], cid))   # cid added
            self.next_image = t_img + tau
        regs = [r for (ta, r) in self.regs if ta <= t_now + 1e-12]
        self.regs = [(ta, r) for (ta, r) in self.regs if ta > t_now + 1e-12]
        fixes = [f for (ta, f) in self.pending if ta <= t_now + 1e-12]
        self.pending = [(ta, f) for (ta, f) in self.pending if ta > t_now + 1e-12]
        return regs, fixes


# ------------------------------------------------------------------ the four statistics of the success process
def four_stat_of(t_img, success, t_lo=None, t_hi=None):
    """the four statistics of a run of attempts: successes per minute, the median and the 95th percentile of the gap
    between successive successful images, and the longest such gap (s); over [t_lo, t_hi] when given."""
    t_img = np.asarray(t_img, float); success = np.asarray(success, bool)
    lo = t_img[0] if t_lo is None else t_lo
    hi = t_img[-1] if t_hi is None else t_hi
    s = t_img[success & (t_img >= lo) & (t_img <= hi)]
    g = np.diff(s)
    return {"rate_per_min": float(len(s) / ((hi - lo) / 60.0)), "gap_p50_s": float(np.percentile(g, 50)),
            "gap_p95_s": float(np.percentile(g, 95)), "longest_s": float(g.max())}


def four_stat_moves(t_success, starts, ends):
    """the four statistics of successful image times on one population, control during
    moves [starts[k], ends[k]): rate = successes in moves / summed durations (per minute); gaps between successive successes inside
    the same move (a move's start and end cut the stream; the edge stretches are no gaps): median and 95th percentile; longest =
    the longest stretch without a success inside any move, the edge stretches included. Returns (statistics, pooled gaps)."""
    t = np.asarray(t_success, float); st = np.asarray(starts, float); en = np.asarray(ends, float)
    a = np.searchsorted(t, st, side="left"); b = np.searchsorted(t, en, side="left")
    inm = np.zeros(len(t), bool)
    mid = np.full(len(t), -1)
    for k in range(len(st)):
        inm[a[k]:b[k]] = True; mid[a[k]:b[k]] = k
    tt, mm = t[inm], mid[inm]
    same = mm[1:] == mm[:-1]
    g = np.diff(tt)[same]
    longest = 0.0
    for k in range(len(st)):
        ev = t[a[k]:b[k]]
        if len(ev):
            m = max(ev[0] - st[k], en[k] - ev[-1], float(np.diff(ev).max()) if len(ev) > 1 else 0.0)
        else:
            m = en[k] - st[k]
        longest = max(longest, float(m))
    dur = float(np.sum(en - st))
    return ({"rate_per_min": float(len(tt) / (dur / 60.0)), "gap_p50_s": float(np.percentile(g, 50)) if len(g) else None,
             "gap_p95_s": float(np.percentile(g, 95)) if len(g) else None, "longest_s": longest, "n": int(len(tt))}, g)


def simulate_four_stat(p, tables, rng, dur_s):
    """the success process alone with the world's attempt timing (one image at a time; success and failure delays
    drawn from the 2022 tables), for dur_s of world time; returns (image times, successes)."""
    sp = FourStatProcess(rng, p, 0.0)
    ds, df = tables["fix_delay_success_s"], tables["fix_delay_fail_s"]
    t, T, S = float(rng.uniform(0.0, 1.0)), [], []
    while t < dur_s:
        ok = sp.attempt(t)
        T.append(t); S.append(ok)
        t += float(rng.choice(ds if ok else df))
    return np.array(T), np.array(S, bool)


def fit_four_stat(targets, tables, rng_factory, dur_s=20000.0, max_eval=400):
    """the parameters (p_track, n_on, off_lo_s, off_hi_s) of FourStatProcess whose generated stream meets the targets:
    rate within 10 %, gap median and 95th percentile within 25 %, no gap longer than targets['longest_s'] (the outage
    is bounded by it less the largest attempt interval). rng_factory() gives the same random stream at every
    evaluation (common random numbers). Returns (params, statistics, ok)."""
    from scipy.optimize import minimize
    longest = float(targets["longest_s"])
    a_max = float(max(np.max(tables["fix_delay_success_s"]), np.max(tables["fix_delay_fail_s"])))
    hi_cap = longest - a_max

    def unpack(x):
        p_track = 1.0 / (1.0 + math.exp(-x[0]))
        n_on = 1.0 + math.exp(x[1])
        off_hi = hi_cap / (1.0 + math.exp(-x[2]))
        off_lo = off_hi / (1.0 + math.exp(-x[3]))
        return {"p_track": p_track, "n_on": n_on, "off_lo_s": off_lo, "off_hi_s": off_hi, "longest_s": longest, "attempt_max_s": a_max}

    def cost(x):
        p = unpack(x)
        T, S = simulate_four_stat(p, tables, rng_factory(), dur_s)
        st = four_stat_of(T, S)
        c = (math.log(st["rate_per_min"] / targets["rate_per_min"]) / 0.10) ** 2
        c += (math.log(st["gap_p50_s"] / targets["gap_p50_s"]) / 0.25) ** 2
        c += (math.log(st["gap_p95_s"] / targets["gap_p95_s"]) / 0.25) ** 2
        return c
    best = None
    for x0 in ([2.0, math.log(20.0), 0.0, 0.0], [1.0, math.log(5.0), 1.0, -1.0], [3.0, math.log(60.0), -1.0, 1.0]):
        r = minimize(cost, np.array(x0, float), method="Nelder-Mead", options={"maxfev": max_eval // 3, "xatol": 1e-3, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    p = unpack(best.x)
    T, S = simulate_four_stat(p, tables, rng_factory(), dur_s)
    st = four_stat_of(T, S)
    ok = (abs(st["rate_per_min"] / targets["rate_per_min"] - 1) <= 0.10 and abs(st["gap_p50_s"] / targets["gap_p50_s"] - 1) <= 0.25
          and abs(st["gap_p95_s"] / targets["gap_p95_s"] - 1) <= 0.25 and st["longest_s"] <= longest)
    p["process"] = "four_statistics"
    return p, st, bool(ok)
