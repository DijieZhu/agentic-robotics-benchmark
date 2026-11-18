"""Planning on the true world: free space as built, the raceline, the speed profile (numpy only).

    plan = Planner(world, params)                       # the centreline frame of the plan, the grip map
    bounds = plan.bounds(moved=None, boxes=None)        # lateral limits of the reference point, per plan station
    line = plan.raceline(bounds)                        # minimum-curvature line inside the limits -> Line
    prof = plan.profile(line, factor=0.93)              # quasi-steady-state speed on that line (the drive model's limits)
    prof.v, prof.lap_time_s

The reference point of the car (the wheelbase midpoint) is the point the lines are planned for. The
lateral limits keep the car's envelope (half width 0.0925 m, 0.205 m ahead of and 0.182 m behind the reference
point) `wall_clear_m` from every duct capsule, tape and box, plus the sweep of the envelope's corners in a bend.

Curvature cap: the line's curvature stays at or below `kappa_cap_frac_of_lock` (0.95) x the
curvature at full steering lock, tan(0.4189) / 0.25784 = 1.7269 /m (radius 0.5791 m): the cap is 1.6405 /m. Without
it the minimum-curvature line wrapped the free end of a shared wall at 1.90 /m, which the car cannot steer. The QP has box constraints only, so the cap is reached in two stages. (1) Re-weighting: where the line's
curvature exceeds the cap, the curvature weights of the plan stations within `kappa_cap_spread_m` are multiplied by
(kappa / cap)^gain and the QP is solved again, at most `kappa_cap_iterations` times. (2) If the cap still does not
hold (hairpins where the peak only moves along the bend): from the unweighted line again, the unweighted objective
plus a quadratic penalty rho (kbar_i(x) - target)^2, kbar_i the curvature linearised about the previous line (second
difference of the line points projected on its normal) averaged over +- `kappa_cap_penalty_mean_half_m` (first 0.2 m,
about Loop.kappa's own smoothing; then 0 m, the station's own curvature, if the first does not reach the cap), target
`kappa_cap_penalty_target_frac` x cap, on a one-sided active set of stations within `kappa_cap_penalty_spread_m` of
every sample above the cap (those whose kbar is at least `kappa_cap_penalty_release_frac` x target, and the most curved
one), at most `kappa_cap_penalty_iterations` solves per averaging.
The line returned is the first that keeps the cap, else the one with the smallest peak (`Line.kappa_cap_met` False).
A line that already keeps the cap is the unweighted solution, bit for bit.
"""

import math

import numpy as np

from .control import STEER_MAX_RAD, WHEELBASE_M
from .geometry import GripMap, Loop, Walls, periodic_min
from .qp import CyclicBand, box_qp, stencil_qp

KAPPA_LOCK_PER_M = math.tan(STEER_MAX_RAD) / WHEELBASE_M

G = 9.81
HALF_WIDTH_M = 0.09246
FRONT_FROM_REF_M = 0.33446 - 0.12892
REAR_FROM_REF_M = 0.12892 + 0.0533
REF_X_M = 0.12892
WHEEL_X_FROM_REF_M = 0.12892
WHEEL_Y_M = 0.07797


def resample_cols(arr, n):
    """Resample a closed table (m, k) whose first two columns are x, y at n points of uniform arc length; the
    other columns interpolated linearly along the polyline."""
    arr = np.asarray(arr, dtype=float)
    pts = arr[:, :2]
    seg = np.roll(pts, -1, axis=0) - pts
    ln = np.hypot(seg[:, 0], seg[:, 1])
    keep = ln > 1e-9
    arr, seg, ln = arr[keep], seg[keep], ln[keep]
    cum = np.concatenate([[0.0], np.cumsum(ln)])
    total = float(cum[-1])
    t = np.arange(n) * (total / n)
    i = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(ln) - 1)
    u = ((t - cum[i]) / ln[i])[:, None]
    nxt = np.roll(arr, -1, axis=0)
    return arr[i] * (1.0 - u) + nxt[i] * u, total


class Limits(object):
    """The car's limits, from the drive model's calibration and the reference program's probe of the drive model's car (program params `limits`)."""

    def __init__(self, cfg):
        self.c_lat = float(cfg["lateral_frac_of_grip_g"])
        self.c_brk = float(cfg["brake_frac_of_grip_g"])
        self.c_trac = float(cfg["traction_frac_of_grip_g"])
        self.v_cap = float(cfg["speed_cap_mps"])
        self.acc_exp = float(cfg.get("accel_combine_exp", 2.0))
        tab = np.asarray(cfg["motor_accel_table_grip1"], dtype=float)
        self.mv, self.ma = tab[:, 0], tab[:, 1] * float(cfg["motor_accel_use_frac"])

    def a_motor(self, v):
        return float(np.interp(v, self.mv, self.ma, right=0.0))


class Profile(object):
    def __init__(self, line, v, factor, mu, lap_time_s):
        self.line = line
        self.v = v
        self.factor = factor
        self.mu = mu
        self.lap_time_s = lap_time_s

    def speed(self, s):
        return self.line.loop.interp(self.v, s)


class Line(object):
    """A closed driving line: the loop (uniform 0.05 m), its offsets from the plan centreline and the plan stations
    of its samples. A raceline also carries its curvature cap: `kappa_cap` (None without a cap), `kappa_max`,
    `kappa_cap_met`, `kappa_cap_resolves`."""

    def __init__(self, plan, offsets, ds=0.05, kappa_half_m=0.15, kappa_smooth_m=0.2):
        self.plan = plan
        self.kappa_cap = None
        self.kappa_cap_met = None
        self.kappa_cap_resolves = 0
        self.offsets = np.asarray(offsets, dtype=float)
        pts = plan.c + self.offsets[:, None] * plan.nl
        self.loop = Loop(pts, ds=ds, kappa_half_m=kappa_half_m, kappa_smooth_m=kappa_smooth_m)
        # plan station of every line sample: the line's points were built one per plan station, so a resampled
        # sample between plan points i and i + 1 has the plan station s_i + u ds_plan
        sp = (plan.loop.s[self.loop.src_i] + self.loop.src_u * plan.loop.ds) % plan.loop.length
        self.s_plan = sp
        self.s_plan_order = np.argsort(sp, kind="stable")
        self.s_plan_sorted = sp[self.s_plan_order]
        self.kappa_max = float(np.max(np.abs(self.loop.kappa)))


class Planner(object):
    def __init__(self, world, params):
        self.world = world
        self.P = params
        pp = params["plan"]
        cl = np.asarray(world["truth"]["centreline"]["samples"], dtype=float)
        seg = np.roll(cl[:, :2], -1, axis=0) - cl[:, :2]
        total = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        n = max(64, int(round(total / float(pp["ds_m"]))))
        arr, _ = resample_cols(cl, n)
        self.loop = Loop(arr[:, :2], ds=total / n, kappa_half_m=0.3, kappa_smooth_m=0.3)
        self.c = self.loop.p
        self.nl = self.loop.nl
        self.left = arr[:, 2]
        self.right = arr[:, 3]
        self.n = n
        self.grip = GripMap(world)
        self.limits = Limits(params["limits"])

    # ------------------------------------------------------------------ free space
    def bounds(self, moved=None, boxes=None, margin_m=None):
        """(lo, hi, info): limits of the reference point's lateral offset at every plan station, and the free
        distances they came from. `boxes`: [(pose, size)] obstacles inside the track (polygons)."""
        pp = self.P["plan"]
        m0 = float(pp["wall_clear_m"] if margin_m is None else margin_m)
        reach = np.maximum(self.left, self.right) + 0.8
        walls = Walls.from_world(self.world, moved=moved, tapes=True)
        free_l, free_r = walls.free_along_normals(self.c, self.nl, reach)
        bl = np.minimum(self.left, free_l)
        br = np.minimum(self.right, free_r)
        box_side = []
        if boxes:
            bl, br, box_side = self._box_limits(bl, br, boxes, m0)
        k = np.abs(self.loop.kappa)
        sweep = 0.5 * k * max(FRONT_FROM_REF_M, REAR_FROM_REF_M) ** 2
        out_left = self.loop.kappa < 0.0                      # right turn: the outside of the bend is the left
        ml = HALF_WIDTH_M + m0 + np.where(out_left, sweep, 0.0)
        mr = HALF_WIDTH_M + m0 + np.where(~out_left, sweep, 0.0)
        hi = bl - ml
        lo = -(br - mr)
        tight = lo > hi
        mid = 0.5 * (lo + hi)
        lo = np.where(tight, mid, lo)
        hi = np.where(tight, mid, hi)
        return lo, hi, {"free_left": bl, "free_right": br, "tight": int(tight.sum()), "box_side": box_side}

    def _box_limits(self, bl, br, boxes, m0):
        """Cut the free distances where a box stands: the car passes each box on one side, with `box_clear_m`."""
        pp = self.P["plan"]
        clear = float(pp["box_clear_m"])
        side = []
        bl, br = bl.copy(), br.copy()
        for pose, size in boxes:
            cs, sn = math.cos(pose[2]), math.sin(pose[2])
            corners = [(pose[0] + cs * dx - sn * dy, pose[1] + sn * dx + cs * dy)
                       for dx in (-0.5 * size[0], 0.5 * size[0]) for dy in (-0.5 * size[1], 0.5 * size[1])]
            sd = [self.loop.project(x, y)[:2] for x, y in corners]
            s0 = sd[0][0]
            rel = [self.loop.gap(s, s0) for s, _ in sd]
            d_lo, d_hi = min(d for _, d in sd), max(d for _, d in sd)
            a, b = min(rel) - FRONT_FROM_REF_M - clear, max(rel) + REAR_FROM_REF_M + clear
            k_a, k_b = int(math.floor(a / self.loop.ds)), int(math.ceil(b / self.loop.ds))
            idx = (self.loop.index(s0) + np.arange(k_a, k_b + 1)) % self.n
            room_left = float(np.min(bl[idx])) - d_hi
            room_right = d_lo + float(np.min(br[idx]))
            go_left = room_left >= room_right
            if go_left:
                br[idx] = np.minimum(br[idx], -(d_hi + clear))      # the right limit moves to the box's left face
            else:
                bl[idx] = np.minimum(bl[idx], d_lo - clear)
            side.append({"s_m": round(s0, 3), "side": "left" if go_left else "right",
                         "room_m": round(max(room_left, room_right), 3)})
        return bl, br, side

    # ------------------------------------------------------------------ the raceline
    def kappa_cap(self):
        """The line's curvature cap: kappa_cap_frac_of_lock x the full-lock curvature; None when off."""
        frac = float(self.P["plan"].get("kappa_cap_frac_of_lock", 0.0))
        return frac * KAPPA_LOCK_PER_M if frac > 0.0 else None

    def raceline(self, lo, hi, weights=None, info=None):
        """Offsets n (per plan station) inside [lo, hi] minimising sum(kappa^2) + rate_weight sum(kappa'^2) over
        the samples (second and third differences of c + n nl over ds^2 and ds^3), plus a small first-difference
        regulariser: a box-constrained QP with a cyclic band Hessian (qp.box_qp); then the curvature cap
        by re-weighting (module text). Returns a Line."""
        pp = self.P["plan"]
        cap = self.kappa_cap()
        n_max = int(pp.get("kappa_cap_iterations", 0)) if cap is not None else 0
        spread = max(0, int(round(float(pp.get("kappa_cap_spread_m", 0.5)) / self.loop.ds)))
        gain = float(pp.get("kappa_cap_gain", 4.0))
        select = float(pp.get("kappa_cap_select_frac", 1.0))
        w = None if weights is None else np.asarray(weights, dtype=float)
        hist = []
        sub = {}
        line = None
        best = None
        qp = dict(reg=float(pp["smooth_reg"]), rate_weight=float(pp["curvature_rate_weight_m2"]), tol=float(pp["qp_tol"]),
                  passes=int(pp["qp_passes"]))
        for it in range(n_max + 1):
            sub = {}
            x = solve_raceline(self.c, self.nl, self.loop.ds, lo, hi, weights=w, info=sub, **qp)
            line = Line(self, x)
            if it == 0:
                first = line
            hist.append(round(line.kappa_max, 4))
            if best is None or line.kappa_max < best.kappa_max:
                best = line
            if cap is None or line.kappa_max <= cap or it == n_max:
                break
            k = np.abs(line.loop.kappa)
            over = k > cap * select
            ex = np.ones(self.n)
            np.maximum.at(ex, line.loop.src_i[over], k[over] / cap)
            ex = periodic_max(ex, spread)
            w = (np.ones(self.n) if w is None else w) * ex ** gain
        stage2 = 0
        if cap is not None and line.kappa_max > cap:
            n_pen = int(pp.get("kappa_cap_penalty_iterations", 0))
            sp2 = max(0, int(round(float(pp.get("kappa_cap_penalty_spread_m", 0.3)) / self.loop.ds)))
            target = float(pp.get("kappa_cap_penalty_target_frac", 0.98)) * cap
            rho = float(pp.get("kappa_cap_penalty_rho", 1000.0))
            release = float(pp.get("kappa_cap_penalty_release_frac", 0.0))
            halves = [int(round(float(h) / self.loop.ds)) for h in pp.get("kappa_cap_penalty_mean_half_m", [0.0])]
            for half in halves:                         # the 0.5 m mean first; the station's own curvature next
                active = {}
                line = first                            # stage 2 starts again from the unweighted line
                x = first.offsets
                for _it in range(n_pen):
                    # one-sided active set: near every sample above the cap, the stations whose curvature is at least
                    # release x target (and the most curved one in any case); stations that fell below it are let go
                    k = line.loop.kappa
                    kr = mean_curvature(station_curvature(self.c, self.nl, x), half)
                    for i in list(active):
                        if active[i] * kr[i] < release * target:
                            del active[i]
                    for q in np.nonzero(np.abs(k) > cap)[0]:
                        j = int(line.loop.src_i[q])
                        win = [jj % self.n for jj in range(j - sp2, j + sp2 + 1)]
                        sg = 1.0 if k[q] > 0.0 else -1.0
                        top = max(win, key=lambda jj: sg * kr[jj])
                        for jj in win:
                            if sg * kr[jj] >= release * target or jj == top:
                                active[jj] = sg
                    idx = sorted(active)
                    sub = {}
                    x = solve_raceline(self.c, self.nl, self.loop.ds, lo, hi, weights=weights, info=sub, x0=x,
                                       penalty=(idx, [active[i] for i in idx], target, rho, half), **qp)
                    line = Line(self, x)
                    stage2 += 1
                    hist.append(round(line.kappa_max, 4))
                    if line.kappa_max < best.kappa_max:
                        best = line
                    if line.kappa_max <= cap:
                        break
                if line.kappa_max <= cap:
                    break
            if line.kappa_max > cap:
                line = best
        line.kappa_cap = cap
        line.kappa_cap_met = cap is None or line.kappa_max <= cap
        line.kappa_cap_resolves = len(hist) - 1
        if info is not None:
            info.update(sub)
            info.update({"kappa_cap": None if cap is None else round(cap, 6), "kappa_max_history": hist,
                         "kappa_cap_met": line.kappa_cap_met, "kappa_cap_resolves": len(hist) - 1,
                         "kappa_cap_penalty_solves": stage2})
        return line

    # ------------------------------------------------------------------ the speed profile
    def wheel_grip(self, line):
        """Grip under the car's four wheels with the reference point on each line sample, heading along the line:
        the smallest of the four."""
        lp = line.loop
        t = lp.tan
        nl = lp.nl
        pts = []
        for sx in (WHEEL_X_FROM_REF_M, -WHEEL_X_FROM_REF_M):
            for sy in (WHEEL_Y_M, -WHEEL_Y_M):
                pts.append(lp.p + sx * t + sy * nl)
        mu = np.min(np.stack([self.grip.grip_points(q) for q in pts]), axis=0)
        zone = float(self.P["plan"].get("edge_zone_m", 0.0))
        if zone > 0.0:
            near = np.zeros(lp.n, dtype=bool)
            for q in pts:
                near |= self.grip.near_raised_edge(q, zone)
            mu = np.where(near, mu * float(self.P["plan"]["edge_grip_factor"]), mu)
        return mu

    def profile(self, line, factor, mu=None, window_m=None, cap_mps=None, start_index=None, start_speed=None):
        """Quasi-steady-state speed on `line` at `factor` x the grip-limited speed: lateral, braking and traction
        capacities are factor^2 x the drive model's fractions of grip x g (corner speeds scale by `factor`); the motor's
        acceleration envelope is not scaled. `window_m` takes the smallest grip within +- that distance."""
        pp = self.P["plan"]
        lp = line.loop
        if mu is None:
            mu = self.wheel_grip(line)
        w = float(pp["grip_window_m"] if window_m is None else window_m)
        mu_w = periodic_min(mu, int(round(w / lp.ds)))
        v = qss_speed(lp.kappa, mu_w, lp.ds, factor, self.limits, cap_mps=cap_mps, start_index=start_index,
                      start_speed=start_speed)
        t = float(np.sum(lp.ds * 2.0 / (v + np.roll(v, -1))))
        return Profile(line, v, factor, mu_w, t)


def periodic_max(a, half_window):
    """Largest value within +- half_window samples of a periodic array."""
    k = int(half_window)
    a = np.asarray(a, dtype=float)
    out = a.copy()
    for j in range(1, k + 1):
        out = np.maximum(out, np.maximum(np.roll(a, j), np.roll(a, -j)))
    return out


def qss_speed(kappa, mu, ds, factor, lim, cap_mps=None, start_index=None, start_speed=None):
    """Forward-backward quasi-steady-state speed on a closed path: lateral capacity c_lat f^2 mu g; acceleration
    min(c_trac f^2 mu g, motor envelope) inside (a_x/A)^p + (a_y/B)^p <= 1 with p = lim.acc_exp (2 = the friction
    ellipse); braking c_brk f^2 mu g inside the ellipse."""
    n = len(kappa)
    f2 = float(factor) ** 2
    ay = lim.c_lat * f2 * mu * G
    ab = lim.c_brk * f2 * mu * G
    at = lim.c_trac * f2 * mu * G
    k = np.maximum(np.abs(kappa), 1e-6)
    cap = lim.v_cap if cap_mps is None else min(lim.v_cap, float(cap_mps))
    vmax = np.minimum(cap, np.sqrt(ay / k))
    v = vmax.copy()
    i0 = int(np.argmin(vmax))
    if start_index is not None:                              # a start from a given speed at a given sample
        i0 = int(start_index) % n
        v[i0] = min(float(v[i0]), max(0.0, float(start_speed or 0.0)))
    kl, ayl, abl, atl, vm = k.tolist(), ay.tolist(), ab.tolist(), at.tolist(), v.tolist()
    two_ds = 2.0 * ds
    mv, ma = lim.mv, lim.ma
    pe = float(getattr(lim, "acc_exp", 2.0))
    laps = 1 if start_index is not None else 2
    for _lap in range(laps):                                 # forward: acceleration
        for q in range(n):
            i = (i0 + q) % n
            j = (i + 1) % n
            vi = vm[i]
            lat = min(1.0, vi * vi * kl[i] / ayl[i])
            rem = max(0.0, 1.0 - lat ** pe) ** (1.0 / pe)
            a = min(atl[i], float(np.interp(vi, mv, ma, right=0.0))) * rem
            vj = math.sqrt(vi * vi + two_ds * a)
            if vj < vm[j]:
                vm[j] = vj
            if start_index is not None and j == i0:
                break
    for _lap in range(2):                                    # backward: braking
        for q in range(n):
            j = (i0 - q) % n
            i = (j - 1) % n
            vj = vm[j]
            lat = vj * vj * kl[j] / ayl[j]
            rem = math.sqrt(max(0.0, 1.0 - lat * lat))
            vi = math.sqrt(vj * vj + two_ds * abl[j] * rem)
            if vi < vm[i]:
                vm[i] = vi
    return np.asarray(vm)


def raceline_hessian(c, nl, ds, weights=None, reg=0.01, rate_weight=0.0, rate_weights=None):
    """(H, f) of the raceline QP: sum_i w_i |D2 P|_i^2 / ds^4 + rate_weight r_i |D3 P|_i^2 / ds^6
    + reg (x_{i+1} - x_i)^2 / ds^2 with P_i = c_i + x_i nl_i (objective = 0.5 x'Hx + f'x + const)."""
    n = len(c)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    terms = [((-1, 0, 1), (1.0, -2.0, 1.0), w / ds ** 4)]
    if rate_weight > 0.0:
        r = np.ones(n) if rate_weights is None else np.asarray(rate_weights, dtype=float)
        terms.append(((-1, 0, 1, 2), (-1.0, 3.0, -3.0, 1.0), float(rate_weight) * r / ds ** 6))
    H, f = stencil_qp(c, nl, terms)
    if reg > 0.0:
        q = 2.0 * float(reg) / ds ** 2
        H.d[0] = H.d[0] + 2.0 * q
        H.d[1] = H.d[1] - q
    return H, f


def curvature_penalty(c, nl, x, idx, signs, target, rho, half=0):
    """Diagonals and linear term of sum over stations i in `idx` of rho (kbar_i(x) - signs_i target)^2. kbar_i is the
    mean over stations i - half .. i + half of k_j(x) = (P_{j-1} - 2 P_j + P_{j+1}) . nh_j / l_j^2, linearised about
    the line x (P = c + x nl; nh_j the unit left normal of that line at j, l_j the mean of its two segment lengths
    there): with half = 2 the mean spans 0.5 m, about what Loop.kappa smooths over. Plain arithmetic (no BLAS)."""
    n = len(c)
    P = c + np.asarray(x, dtype=float)[:, None] * nl
    half = int(half)
    width = 2 * half + 3
    diags = [np.zeros(n) for _ in range(width)]
    f = np.zeros(n)
    co = (1.0, -2.0, 1.0)
    rows = {}
    for j in set((i + o) % n for i in idx for o in range(-half, half + 1)):
        js = ((j - 1) % n, j, (j + 1) % n)
        tx, ty = P[js[2], 0] - P[js[0], 0], P[js[2], 1] - P[js[0], 1]
        tn = math.hypot(tx, ty)
        nhx, nhy = -ty / tn, tx / tn
        l1 = math.hypot(P[js[1], 0] - P[js[0], 0], P[js[1], 1] - P[js[0], 1])
        l2 = math.hypot(P[js[2], 0] - P[js[1], 0], P[js[2], 1] - P[js[1], 1])
        ell2 = (0.5 * (l1 + l2)) ** 2
        a = [co[m] * (float(nl[js[m], 0]) * nhx + float(nl[js[m], 1]) * nhy) / ell2 for m in range(3)]
        b = sum(co[m] * (float(c[js[m], 0]) * nhx + float(c[js[m], 1]) * nhy) for m in range(3)) / ell2
        rows[j] = (a, b)
    w = 1.0 / (2 * half + 1)
    for i, sg in zip(idx, signs):
        coef = [0.0] * width                       # coefficients of x_{i-half-1} .. x_{i+half+1}
        b = 0.0
        for o in range(-half, half + 1):
            a_j, b_j = rows[(i + o) % n]
            for m in range(3):
                coef[o + half + m] += w * a_j[m]
            b += w * b_j
        r = b - float(sg) * float(target)
        base = i - half - 1
        for m1 in range(width):
            j1 = (base + m1) % n
            f[j1] += 2.0 * rho * r * coef[m1]
            for m2 in range(m1, width):
                diags[m2 - m1][j1] += 2.0 * rho * coef[m1] * coef[m2]
    return diags, f


def station_curvature(c, nl, x):
    """Signed curvature of the line c + x nl at every plan station: second difference projected on the line's left
    normal over the squared mean segment length (the quantity `curvature_penalty` linearises)."""
    P = c + np.asarray(x, dtype=float)[:, None] * nl
    pm, pp_ = np.roll(P, 1, axis=0), np.roll(P, -1, axis=0)
    t = pp_ - pm
    tn = np.hypot(t[:, 0], t[:, 1])
    nhx, nhy = -t[:, 1] / tn, t[:, 0] / tn
    d2 = pm - 2.0 * P + pp_
    l1 = np.hypot(P[:, 0] - pm[:, 0], P[:, 1] - pm[:, 1])
    l2 = np.hypot(pp_[:, 0] - P[:, 0], pp_[:, 1] - P[:, 1])
    return (d2[:, 0] * nhx + d2[:, 1] * nhy) / (0.5 * (l1 + l2)) ** 2


def mean_curvature(k, half):
    """Mean of a periodic per-station curvature over stations i - half .. i + half (curvature_penalty's kbar)."""
    from .geometry import periodic_smooth
    return periodic_smooth(k, int(half))


def solve_raceline(c, nl, ds, lo, hi, weights=None, reg=0.01, rate_weight=0.0, tol=1e-9, info=None, passes=4,
                   penalty=None, x0=None):
    """The QP solved `passes` times: after each pass the sample spacing along the line, ds_l = |P_{i+1} - P_{i-1}| / 2,
    rescales the terms so that they approximate the integrals of kappa^2 and kappa'^2 along the line itself
    (weights (ds / ds_l)^3 and (ds / ds_l)^5): without it the inside of a bend, where the samples crowd, would look
    cheap to the optimiser. `penalty` = (stations, signs, target, rho): the curvature penalty of the cap's second
    stage, linearised about `x0` in the first pass and about the previous pass after that."""
    base = np.ones(len(c)) if weights is None else np.asarray(weights, dtype=float)
    w2, w3 = base, None
    x = None if x0 is None else np.asarray(x0, dtype=float)
    hist = []
    for k in range(int(passes)):
        H, f = raceline_hessian(c, nl, ds, weights=w2, reg=reg, rate_weight=rate_weight, rate_weights=w3)
        if penalty is not None and x is not None and len(penalty[0]):
            dp, fp = curvature_penalty(c, nl, x, *penalty)
            kb = max(H.k, len(dp) - 1)
            H = CyclicBand([(H.d[j] if j <= H.k else 0.0) + (dp[j] if j < len(dp) else 0.0) for j in range(kb + 1)])
            f = f + fp
        sub = {}
        x = box_qp(H, f, lo, hi, tol=tol, info=sub)
        P = c + x[:, None] * nl
        dl = 0.5 * np.hypot(*(np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)).T)
        ratio = np.clip(ds / np.maximum(dl, 1e-9), 0.3, 3.0)
        w2 = base * ratio ** 3
        w3 = ratio ** 5
        hist.append(sub.get("iterations"))
    if info is not None:
        info.update({"passes": int(passes), "ipm_iterations": hist})
    return x
