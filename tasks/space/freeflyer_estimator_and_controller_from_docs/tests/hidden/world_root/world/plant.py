"""Plant of the hidden world: the maker's allocation, servo, impeller and nozzle physics, with the constants parsed
from the maker's source (data/plant_constants.json). The delivered wrench takes the signed-axis gains of the fan
level in force; an external body wrench (the cabin air) is added to it. The maker's allocation, servo, impeller and
nozzle physics and the nozzle table are unchanged.



























"""
import json
import math
import os

import numpy as np

from . import rot

HERE = os.path.dirname(os.path.abspath(__file__))
CONSTANTS = os.path.join(HERE, "data", "plant_constants.json")


class Constants:
    def __init__(self, path=CONSTANTS, ramp_rpm_s=None):
        c = json.load(open(path))
        self.dt = float(c["DT"])
        self.pos = np.array(c["nozzle_offsets1"] + c["nozzle_offsets2"], float).reshape(12, 3)
        self.orient = np.array(c["nozzle_orientations1"] + c["nozzle_orientations2"], float).reshape(12, 3)
        self.width = np.tile(np.array(c["nozzle_widths"], float), 2)
        self.cd = np.concatenate([c["discharge_coeff1"], c["discharge_coeff2"]]).astype(float)
        self.leak = np.array(c["zero_thrust_area"], float)
        self.area_x = np.array(c["AREA_LOOKUP_INPUT"], float); self.area_y = np.array(c["CDP_LOOKUP_OUTPUT"], float)
        self.thr_x = np.array(c["THRUST_LOOKUP_BREAKPOINTS"], float); self.thr_y = np.array(c["THRUST_LOOKUP_CDP"], float)
        self.levels = np.array(c["IMPELLER_SPEEDS"], float)
        self.groups = [[k - 1 for k in g] for g in c["fam_nozzle_dirs"]]
        self.max_open, self.min_open = float(c["abp_nozzle_max_open_angle"]), float(c["abp_nozzle_min_open_angle"])
        self.flap_len, self.intake = float(c["abp_nozzle_flap_length"]), float(c["abp_nozzle_intake_height"])
        self.flap_gear = float(c["abp_nozzle_gear_ratio"])
        self.s_k, self.s_r = float(c["servo_motor_k"]), float(c["servo_motor_r"])
        self.s_fric, self.s_inertia = float(c["servo_motor_friction"]), float(c["servo_motor_gearbox_inertia"])
        self.s_gear = float(c["servo_motor_gear_ratio"])
        self.backlash = float(c["backlash_half_width"])
        self.i_speed_k, self.i_r, self.i_torque_k = float(c["imp_motor_speed_k"]), float(c["imp_motor_r"]), float(c["imp_motor_torque_k"])
        self.i_fric, self.i_inertia = float(c["imp_motor_friction_coeff"]), float(c["impeller_inertia"])
        self.speed2pwm, self.diam, self.rho = float(c["impeller_speed2pwm"]), float(c["impeller_diameter"]), float(c["air_density"])
        self.servo_pid = [float(v) for v in c["servo_pid"]]
        self.speed_pid = [float(v) for v in c["speed_pid"]]
        # ramp from the table when given (the file's value is the same constant, 200 rpm/s)
        self.ramp = float(c["impeller_ramp_rad_s2"]) if ramp_rpm_s is None else float(ramp_rpm_s) * 2 * math.pi / 60.0
        self.imp_axis = np.array([c["impeller_orientation1"], c["impeller_orientation2"]], float)
        self.voltage = 14.0   # world.battery_voltage_v (asserted equal in World)
        self.max_theta = self.flap_gear * (self.max_open - self.min_open)
        self.lo_open, self.hi_open = math.cos(self.max_open), math.cos(self.min_open)


def lookup(bp, y, x):
    x = np.asarray(x, float)
    k = np.clip(np.searchsorted(bp, x, side="right") - 1, 0, len(bp) - 2)
    r = (x - bp[k]) / (bp[k + 1] - bp[k])
    out = (1.0 - r) * y[k] + r * y[k + 1]
    out = np.where(x < bp[0], y[0], out)
    return np.where(x > bp[-1], y[-1], out)


def nozzle_matrix(C, com):
    arm = C.pos - np.asarray(com, float)[None, :]
    d = -C.orient
    return np.vstack([d.T, np.cross(arm, d).T])


def pinv_fw(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    sinv = np.where(s > np.finfo(float).eps, 1.0 / s, 0.0)
    return (Vt.T * sinv) @ U.T


class Allocator:
    """fam.cc for one wrench: allocate(F, T, level) -> servo_cmd (12,) int, impeller_cmd (2,) int, u (12,)."""

    def __init__(self, C, com):
        self.C = C
        self.com = np.asarray(com, float)
        self.Binv = pinv_fw(nozzle_matrix(C, self.com))
        self.Bt = self.Binv.T

    def allocate(self, F, T, level):
        C = self.C
        W = np.concatenate([np.asarray(F, float), np.asarray(T, float)])
        u = W @ self.Bt
        for g in C.groups:
            mn = min(0.0, float(u[g].min()))
            if mn < 0.0:
                u[g] -= mn
        pwm = np.zeros(12)
        imp = np.zeros(2, int)
        lv = int(level)
        if lv == 0:
            pwm[:] = 255.0
            return np.clip(np.trunc(pwm), 0, 255).astype(int), imp, u
        om = C.levels[lv - 1]
        imp[:] = int(C.speed2pwm * om + 0.5) & 0xFF
        for i in (0, 1):
            sl = slice(6 * i, 6 * i + 6)
            cd = C.cd[sl]
            uu = u[sl]
            tot = float((uu / cd).sum())
            cdp = float(lookup(C.thr_x, C.thr_y, tot / (om * om)))
            dp = om * om * C.diam ** 2 * C.rho * cdp
            npd = 2.0 * cd ** 2 * dp
            with np.errstate(divide="ignore", invalid="ignore"):
                area = uu / npd
                op = (C.intake - 0.5 * area / C.width[sl]) / C.flap_len
            op = np.where(np.isnan(op) | (op < C.lo_open), C.lo_open, op)
            op = np.where(op > C.hi_open, C.hi_open, op)
            th = np.arccos(op) - C.min_open
            pwm[sl] = 255.0 / (C.max_open - C.min_open) * th
        return np.clip(np.trunc(pwm), 0, 255).astype(int), imp, u


def speed_loop_equilibrium(C, om):
    """the speed loop's integrator value that holds the impeller at speed om with no load: the motor's
    voltage v = om / speed_k + r * friction * om / torque_k gives zero net torque; the PID's output at zero error is
    ki * integrator (the derivative term is zero at rest)."""
    kp, ki, kd, omax, omin = C.speed_pid
    v = om / C.i_speed_k + C.i_r * C.i_fric * om / C.i_torque_k
    assert omin < v < omax and abs(v) < C.voltage, "the equilibrium voltage is outside the loop's limits"
    return v / ki


def actuator_state(C, level, servo_cmd=None, speed_loop="equilibrium"):
    """the actuators at rest at a level: the servos at the commanded angle, the impellers at their quantised setpoint
    with the speed loop at its equilibrium (the other start: speed_loop="integrator_at_zero")."""
    if speed_loop not in ("equilibrium", "integrator_at_zero"):
        raise ValueError(speed_loop)
    a = dict(st=np.zeros(12), bl=np.zeros(12), pom=np.zeros(12), s_last=np.zeros(12), s_int=np.zeros(12),
             isp=np.zeros(2), prate=np.zeros(2), i_last=np.zeros(2), i_int=np.zeros(2), mtq=np.zeros(2),
             thrust=np.zeros(12), area=np.zeros(12))
    if level > 0:
        om = int(C.speed2pwm * C.levels[level - 1] + 0.5) / C.speed2pwm
        a["isp"][:] = om; a["prate"][:] = om
        if speed_loop == "equilibrium":
            a["i_int"][:] = speed_loop_equilibrium(C, om)
    if servo_cmd is not None:
        th = (C.max_theta / 255.0) * np.asarray(servo_cmd, float) / C.s_gear
        a["st"][:] = th; a["bl"][:] = th
    return a


def actuator_tick(C, a, servo_cmd, imp_cmd):
    """One propulsion_sim tick (in place), unchanged physics of the line's calibrated plant's batched tick for one robot."""
    dt = C.dt
    kp, ki, kd, omax, omin = C.servo_pid
    st, bl, pom = a["st"], a["bl"], a["pom"]
    hw = C.backlash
    lo = st < bl - hw
    hi = (st > bl + hw) & ~lo
    bl[lo] = st[lo] + hw
    bl[hi] = st[hi] - hw
    err = (C.max_theta / 255.0) * servo_cmd - bl * C.s_gear
    d = kd * err - a["s_last"]
    v = kp * err + ki * a["s_int"] + d
    a["s_last"] += d * dt
    a["s_int"] += err * dt
    v = np.clip(v, omin, omax)
    cur = (v - C.s_k * pom) / C.s_r
    st += dt * pom
    np.clip(st, 0.0, C.max_theta / C.s_gear, out=st)
    tq = C.s_k * cur - C.s_fric * pom
    pom += (1.0 / C.s_inertia) * dt * tq
    theta = bl * C.s_gear / C.flap_gear
    area = (C.intake - np.cos(C.min_open + theta) * C.flap_len) * 2.0 * C.width
    kp, ki, kd, omax, omin = C.speed_pid
    mc = C.ramp * dt
    sp = imp_cmd / C.speed2pwm
    sp = np.minimum(np.maximum(sp, a["prate"] - mc), a["prate"] + mc)
    a["prate"][:] = sp
    x = sp - a["isp"]
    d = kd * x - a["i_last"]
    v = kp * x + ki * a["i_int"] + d
    a["i_last"] += d * dt
    a["i_int"] += x * dt
    v = np.clip(v, omin, omax)
    v = np.clip(v, -C.voltage, C.voltage)
    cur = (v - a["isp"] / C.i_speed_k) / C.i_r
    mtq = cur * C.i_torque_k - C.i_fric * a["isp"]
    a["mtq"][:] = mtq
    a["isp"] += mtq / C.i_inertia * dt
    thrust = np.empty(12)
    for i in (0, 1):
        sl = slice(6 * i, 6 * i + 6)
        tot = C.leak[i] + float((C.cd[sl] * area[sl]).sum())
        cdp = float(lookup(C.area_x, C.area_y, tot))
        dp = a["isp"][i] ** 2 * cdp * C.diam ** 2 * C.rho
        thrust[sl] = 2.0 * dp * C.cd[sl] ** 2 * area[sl]
    a["thrust"][:] = thrust
    a["area"][:] = area
    return thrust


class Plant:
    """One robot in the turning cabin frame. Use: s = P.initial_state(...); P.step(s, wrench_cmd, level, ext)."""

    def __init__(self, gains_by_level, mass, com, inertia, delay_s, ramp_rpm_s, station_rate_cabin, dt, com_alloc=None):
        self.C = Constants(ramp_rpm_s=ramp_rpm_s)
        assert abs(self.C.dt - dt) < 1e-12
        self.mass = float(mass)
        self.com = np.asarray(com, float)                     # the true centre of mass (physics)
        self.J = np.asarray(inertia, float)
        self.Jinv = np.linalg.inv(self.J)
        self.delay = float(delay_s)
        assert self.delay >= 0.0 and self.delay == self.delay, "a pure delay is 0 or positive"
        # the allocator's centre of mass is the configured profile's (the maker's allocation as it is)
        self.com_alloc = self.com if com_alloc is None else np.asarray(com_alloc, float)
        self.alloc = Allocator(self.C, self.com_alloc)
        self.gains = {lv: (np.asarray(g["torque"], float), np.asarray(g["force"], float)) for lv, g in gains_by_level.items()}
        self.Om = np.asarray(station_rate_cabin, float)
        self.speed_loop_start = "equilibrium"
        arm = self.C.pos - self.com[None, :]
        self.t2f = -self.C.orient
        self.t2t = -np.cross(arm, self.C.orient)

    def delivered(self, a, omega, level):
        F = a["thrust"] @ self.t2f
        T = a["thrust"] @ self.t2t
        for i in (0, 1):
            h = self.C.imp_axis[i] * self.C.i_inertia * a["isp"][i]
            T = T + self.C.imp_axis[i] * (-a["mtq"][i]) + np.cross(omega, h)
        gt, gf = self.gains[max(1, int(level))]
        T = np.where(T >= 0, gt[0::2], gt[1::2]) * T
        F = np.where(F >= 0, gf[0::2], gf[1::2]) * F
        return F, T

    def initial_state(self, level, p_origin, q, t):
        cmd, imp, _ = self.alloc.allocate(np.zeros(3), np.zeros(3), level)
        a = actuator_state(self.C, level, cmd, speed_loop=self.speed_loop_start)
        for _ in range(250):
            actuator_tick(self.C, a, cmd.astype(float), imp.astype(float))
        q = rot.qnormalize(q)
        R = rot.qmat(q)
        s = dict(t=float(t), act=a, buf=[(-1e18, cmd.astype(float), imp.astype(float))], q=q,
                 p=np.asarray(p_origin, float) + R @ self.com, v=np.zeros(3), w=R.T @ self.Om, wrench=np.zeros(6),
                 level=int(level))
        F, T = self.delivered(a, s["w"], level)
        s["wrench"] = np.concatenate([F, T])
        return s

    def origin(self, s):
        return s["p"] - rot.qmat(s["q"]) @ self.com

    def step(self, s, wrench_cmd, level, ext_body):
        """Issue wrench_cmd at s['t'], tick the actuators (the command in force is the latest issued at or before
        t - delay), integrate one step of dt with the delivered wrench plus ext_body (F, T body). Returns the
        specific force at the centre of mass in body axes during the step, the angular acceleration, and the
        inertial rate at the step's start (for the inertial unit)."""
        C = self.C
        W = np.asarray(wrench_cmd, float)
        cmd, imp, _ = self.alloc.allocate(W[:3], W[3:], int(level))
        if self.delay == 0.0:
            # no pure delay: the command issued now is the command in force for this tick
            sc, ic = cmd.astype(float), imp.astype(float)
            s["buf"] = [(s["t"], sc, ic)]
        else:
            s["buf"].append((s["t"], cmd.astype(float), imp.astype(float)))
            tq = s["t"] - self.delay
            k = 0
            for j, (ti, _, _) in enumerate(s["buf"]):
                if ti <= tq + 1e-12:
                    k = j
            _, sc, ic = s["buf"][k]
            s["buf"] = s["buf"][k:]
        a = s["act"]
        actuator_tick(C, a, sc, ic)
        F, T = self.delivered(a, s["w"], level)
        s["wrench"] = np.concatenate([F, T])
        s["level"] = int(level)
        Ft = F + ext_body[:3]
        Tt = T + ext_body[3:]
        h = C.dt
        J, Ji = self.J, self.Jinv
        w = s["w"]
        f = lambda w_: Ji @ (Tt - np.cross(w_, J @ w_))  # noqa: E731
        k1 = f(w)
        k2 = f(w + 0.5 * h * k1)
        w_new = w + h * k2
        alpha = k2
        R0 = rot.qmat(s["q"])
        wm = 0.5 * (w + w_new)
        wrel = wm - R0.T @ self.Om
        s["q"] = rot.qnormalize(rot.qmul(s["q"], rot.qexp(wrel * h)))
        R1 = rot.qmat(s["q"])
        Rm = rot.qmat(rot.qnormalize(s["q"] + 0.0))
        fs = Ft / self.mass                                               # specific force, body axes
        acc = 0.5 * (R0 + R1) @ fs
        Om = self.Om
        cor = -2.0 * np.cross(Om, s["v"]) - np.cross(Om, np.cross(Om, s["p"]))
        acc = acc + cor
        s["p"] = s["p"] + h * s["v"] + 0.5 * h * h * acc
        s["v"] = s["v"] + h * acc
        w_start = w
        s["w"] = w_new
        s["t"] = s["t"] + h
        del Rm
        return fs, alpha, w_start, w_new


STEADY_TOL = 1e-12          # steady state: no state of the actuators changes by more than this (relative)
STEADY_MAX_TICKS = 5000     # 80 s: a start that is not at steady state by then is an error


def settle(C, a, cmd, imp, tol=STEADY_TOL, n_max=STEADY_MAX_TICKS):
    """tick the actuators on a constant command until they are at steady state: the largest change over one
    tick of the nozzle thrusts, the impeller speeds and the servo angles is at most tol times the largest of them (and at
    least two ticks). Returns the number of ticks."""
    prev = None
    for n in range(1, int(n_max) + 1):
        actuator_tick(C, a, cmd, imp)
        x = np.concatenate([a["thrust"], a["isp"], a["st"], a["bl"]])
        if prev is not None and float(np.max(np.abs(x - prev))) <= tol * max(1.0, float(np.max(np.abs(x)))):
            return n
        prev = x.copy()
    raise RuntimeError("the actuators are not at steady state after %d ticks" % n_max)


def static_response(plant, level, axis, mag):
    """delivered over commanded on one signed axis at steady state (the maker's nominal plant with the level's
    gains): the 'response per signed axis and fan level' of the documents' model. Both states (the command
    and zero) start with the speed loop at its equilibrium and are ticked until they are at steady state (settle()); with
    plant.speed_loop_start == "integrator_at_zero" both start with the speed loop's integrator at zero, inside the
    impeller's sag, and are ticked 300 steps (4.8 s), which reads 3.1 % low (the comparison
    flights only)."""
    C = plant.C
    W = np.zeros(6)
    W[axis] = mag
    old = getattr(plant, "speed_loop_start", "equilibrium") == "integrator_at_zero"
    cmd, imp, _ = plant.alloc.allocate(W[:3], W[3:], level)
    cmd0, imp0, _ = plant.alloc.allocate(np.zeros(3), np.zeros(3), level)
    if old:
        a = actuator_state(C, level, cmd, speed_loop="integrator_at_zero")
        for _ in range(300):
            actuator_tick(C, a, cmd.astype(float), imp.astype(float))
        a0 = actuator_state(C, level, cmd0, speed_loop="integrator_at_zero")
        for _ in range(300):
            actuator_tick(C, a0, cmd0.astype(float), imp0.astype(float))
    else:
        a = actuator_state(C, level, cmd)
        settle(C, a, cmd.astype(float), imp.astype(float))
        a0 = actuator_state(C, level, cmd0)
        settle(C, a0, cmd0.astype(float), imp0.astype(float))
    F, T = plant.delivered(a, np.zeros(3), level)
    F0, T0 = plant.delivered(a0, np.zeros(3), level)
    d = np.concatenate([F - F0, T - T0])
    return float(d[axis] / mag)
