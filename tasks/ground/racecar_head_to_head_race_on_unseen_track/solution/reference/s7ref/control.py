"""Path tracking with the true pose (numpy only).

The command sent with a request acts one tick later (the delay line) and the servo follows its target at 3.2 rad/s
with a lag, so the controller works on the state predicted `predict_s` ahead (constant twist from the true pose,
velocity and yaw rate). Lateral law at the predicted reference point (a linearised kinematic bicycle, distance
constant Ld):

    u     = kappa(s + v tau_ff) - e / Ld^2 - 2 zeta e_chi / Ld - k_i * integral(e ds) / Ld^2
    steer = atan(wheelbase * u)

e = lateral offset of the reference point from the path (left +), e_chi = course angle of the reference point minus
the path heading. Speed: the profile's speed `tau_v` ahead of the predicted station where the profile rises,
`tau_v_down` ahead where it falls; never more than `dv_up` above the present speed (no wheelspin from a step);
capped by the caller (traffic, the end of practice). The understeer observer (gain 0 by default) and the
integral term (k_i 0) were tried in development and are off.
"""

import math

from .geometry import wrap

WHEELBASE_M = 0.25784
REF_X_M = 0.12892
STEER_MAX_RAD = 0.4189


class Tracker(object):
    def __init__(self, params):
        cp = params["control"]
        self.predict_s = float(cp["predict_s"])
        self.tau_ff = float(cp["tau_ff_s"])
        self.tau_v = float(cp["tau_v_s"])
        self.tau_v_down = float(cp.get("tau_v_down_s", cp["tau_v_s"]))
        self.ld = [float(v) for v in cp["ld_m"]]                  # [a, b, min, max]: Ld = clip(a + b v, min, max)
        self.zeta = float(cp["zeta"])
        self.k_i = float(cp["k_i"])
        self.i_leak = float(cp["i_leak_per_m"])
        self.i_max = float(cp["i_max_m2"])
        self.dv_up = float(cp["dv_up_mps"])
        self.servo_tau = float(cp["servo_lag_s"])
        self.us_gain = float(cp["understeer_gain"])
        self.us_alpha = float(cp["understeer_filter"])
        self.us_max = float(cp["understeer_max_per_m"])
        self.us_vmin = float(cp["understeer_min_speed_mps"])
        self.brake_cmd_frac = float(cp.get("brake_cmd_frac", 0.0))
        self.brake_floor_frac = float(cp.get("brake_floor_frac", 0.2))
        self.c_lat = float(params["limits"]["lateral_frac_of_grip_g"])
        self.n_brake_limited = 0
        self.tick_s = 0.025
        self.reset()

    def reset(self):
        self.hint = None
        self.path_id = None
        self.integ = 0.0
        self.delta_act = 0.0
        self.dk_us = 0.0
        self.vt_prev = None
        self.cap_prev = None
        self.last = {}

    def observe(self, steer_echo, truth):
        """Update the servo model (first-order lag of the echoed target) and the understeer observer: the
        kinematic curvature of the modelled wheel angle minus the measured path curvature of C (r / vx)."""
        a = 1.0 - math.exp(-self.tick_s / self.servo_tau)
        self.delta_act += (float(steer_echo) - self.delta_act) * a
        vx = float(truth["car"]["vel_mps"][0])
        r = float(truth["car"]["yaw_rate_radps"])
        if vx > self.us_vmin:
            raw = math.tan(self.delta_act) / WHEELBASE_M - r / vx
            self.dk_us += (raw - self.dk_us) * self.us_alpha
        else:
            self.dk_us *= (1.0 - self.us_alpha)
        self.dk_us = max(-self.us_max, min(self.us_max, self.dk_us))

    def predicted(self, truth):
        """Pose of C and the reference point predict_s ahead, and the car-frame velocity, yaw rate."""
        car = truth["car"]
        x, y, psi = (float(v) for v in car["pose"])
        vx, vy = (float(v) for v in car["vel_mps"])
        r = float(car["yaw_rate_radps"])
        T = self.predict_s
        # constant twist: rotate the body velocity with the heading
        if abs(r) > 1e-6:
            dpsi = r * T
            s1, c1 = math.sin(psi + dpsi) - math.sin(psi), math.cos(psi + dpsi) - math.cos(psi)
            xp = x + (vx * s1 + vy * c1) / r
            yp = y + (-vx * c1 + vy * s1) / r
            psip = psi + dpsi
        else:
            c, s = math.cos(psi), math.sin(psi)
            xp = x + (c * vx - s * vy) * T
            yp = y + (s * vx + c * vy) * T
            psip = psi
        rx, ry = xp + REF_X_M * math.cos(psip), yp + REF_X_M * math.sin(psip)
        return xp, yp, psip, rx, ry, vx, vy, r

    def command(self, truth, loop, kappa, speed_fn, cap=None, path_id=None, mu_here=None, a_up=None):
        """(steer, speed, info) for a path given as `loop` (geometry.Loop) with its curvature array, and
        `speed_fn(s)` -> target speed on it; `cap` limits the speed (traffic, the end of practice), falling at
        most at `brake_cmd_frac` of the friction left (with `mu_here`, the grip under the car). With `a_up` the
        target rises at most at that acceleration from the last target: below its profile (after a pass, when a
        cap lets go) the car comes back at the profile's own traction limit, not at full traction (accelerating
        hard and then braking for the next bend makes this car spin)."""
        if path_id != self.path_id:
            self.hint = None
            self.path_id = path_id
        xp, yp, psip, rx, ry, vx, vy, r = self.predicted(truth)
        s, e, i = loop.project(rx, ry, hint=self.hint, window_m=2.5)
        self.hint = i
        v = max(vx, 0.0)
        hpath = loop.heading_at(s)
        vlat = vy + r * REF_X_M
        chi = psip + (math.atan2(vlat, vx) if vx > 0.3 else 0.0)
        echi = wrap(chi - hpath)
        a, b, lo, hi = self.ld
        Ld = min(max(a + b * v, lo), hi)
        kff = loop.interp(kappa, s + v * self.tau_ff)
        ds = v * self.tick_s
        self.integ = (self.integ + e * ds) * max(0.0, 1.0 - self.i_leak * ds)
        self.integ = max(-self.i_max, min(self.i_max, self.integ))
        u = kff + self.us_gain * self.dk_us - e / (Ld * Ld) - 2.0 * self.zeta * echi / Ld - self.k_i * self.integ / (Ld * Ld)
        steer = math.atan(WHEELBASE_M * u)
        steer = max(-STEER_MAX_RAD, min(STEER_MAX_RAD, steer))
        v_here = float(speed_fn(s))
        v_ahead = float(speed_fn(s + max(v, 0.5) * self.tau_v))
        if v_ahead >= v_here:                                # accelerating: the target leads by tau_v
            vt = v_ahead
        else:                                                # braking: a shorter lead (the loop responds faster)
            vt = float(speed_fn(s + max(v, 0.5) * self.tau_v_down))
        vt = min(vt, v + self.dv_up)
        if a_up is not None:
            base = max(v, self.vt_prev) if self.vt_prev is not None else v
            vt = min(vt, base + float(a_up) * self.tick_s)
        if cap is not None and mu_here is not None and self.brake_cmd_frac > 0.0:
            # a cap (traffic, the end of practice) never falls faster than brake_cmd_frac of the friction the
            # present lateral acceleration leaves: a step of the target makes this car brake hard and spin. The
            # profile's own braking is not limited (it is consistent with the car's limits).
            g_mu = 9.81 * float(mu_here)
            a_lat = v * v * abs(kff)
            rem = math.sqrt(max(0.0, 1.0 - (a_lat / (self.c_lat * g_mu)) ** 2))
            a_dec = max(self.brake_floor_frac, self.brake_cmd_frac * rem) * g_mu
            prev = self.cap_prev if self.cap_prev is not None else v
            cap_eff = max(float(cap), min(v, prev) - a_dec * self.tick_s)
            if cap_eff > float(cap) + 1e-9:
                self.n_brake_limited += 1
            self.cap_prev = cap_eff
            vt = min(vt, cap_eff)
        else:
            self.cap_prev = None
        vt = max(vt, 0.0)
        self.vt_prev = vt
        self.last = {"s": s, "e": e, "echi": echi, "v": v, "vt": vt, "steer": steer}
        return steer, vt, self.last
