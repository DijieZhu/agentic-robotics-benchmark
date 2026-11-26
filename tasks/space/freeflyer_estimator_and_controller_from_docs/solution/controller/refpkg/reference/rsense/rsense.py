# Generated file (the task's solution): do not edit; regenerate it from the maintained source.
import json
import math
import os
import re
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.environ.get('RS_PKG_ROOT') or os.path.dirname(os.path.dirname(HERE))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from world import rot, traj
from world.config import load_spec
from world.plant import Constants, Allocator
import fastplan
import rzone
GROUPS = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'B', 'C', 'D', 'E2', 'F', 'I', 'J', 'L']
ALLOWED = ['CO', 'EV', 'CS', 'RT', 'ZH', 'SB', 'VW', 'VL']
DEFAULT_OFF = ('ZH', 'VL')
SIGNED = ['+x', '-x', '+y', '-y', '+z', '-z']
PUBLISHED_IN_DOCUMENTS = {'world.imu.mount', 'world.fix.frame', 'world.imu.device_filter', 'world.mass_file'}
OWN_DESIGN = {'cert.checkout_sequence'}
KEYS_READ = set()
MODES = ('nominal', 'quiet', 'precision')
CHI2_3 = 16.27
GATE_FLOOR_POS, GATE_FLOOR_ATT = (0.01, 0.005)
INIT_S = 1.0
HOLD_KP, HOLD_KD = (0.35, 0.6)
EV_U_POS, EV_U_VIEW, EV_FIX_AGE = (0.01, 0.015, 8.0)
EV_RESERVE = 3.0
EV_VIEW_WAIT_OK, EV_VIEW_WAIT_ANY, EV_VIEW_CENTRE = (1.0, 2.0, 0.015)
EV_SLACK_OK, EV_SLACK_ANY = (1.0, 0.3)
EV_VIEW_PLAN_MARGIN = 0.35
SB_MARGIN = 0.75
SB_STEPS = 2
VW_CENTRE = 0.015
VW_SPEED = 0.003
VW_LIMIT = 0.033
EV_F = 1.0
CS_EXTRA_P, CS_EXTRA_A, CS_MAX_AGE, CS_VFLOOR = (0.005, 0.005, 6.0, 0.003)
RT_REPLAY_PER_STEP = 16
CO_D_SD = 0.003
LAG_GRID = tuple((0.004 * k for k in range(41)))

def _cross(a, b):
    a0, a1, a2 = (float(a[0]), float(a[1]), float(a[2]))
    b0, b1, b2 = (float(b[0]), float(b[1]), float(b[2]))
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])
_I15 = np.eye(15)

def pub(key):
    e = load_spec()[key]
    assert e.get('public') or key in PUBLISHED_IN_DOCUMENTS or key in OWN_DESIGN, 'the reference may not read ' + key
    KEYS_READ.add(key)
    return e['value']

def limits(mode, accel_factor=1.0):
    sp = pub('plan.speed_limit.' + mode)
    ac = pub('plan.acceleration_limit')[mode]
    return (sp['linear_m_s'], ac['linear_m_s2'] * accel_factor, sp['angular_rad_s'], ac['angular_rad_s2'] * accel_factor)

def _lv(d, lv):
    return d[lv] if lv in d else d[str(lv)]

class NominalModel:

    def __init__(self, C, al, gains):
        self.C, self.al, self.g = (C, al, gains)
        arm = C.pos - al.com[None, :]
        self.t2f = -C.orient
        self.t2t = -np.cross(arm, C.orient)

    def __call__(self, F, T, level, use_gains=True):
        sc, _, _ = self.al.allocate(F, T, level)
        om = self.C.levels[level - 1]
        return self.from_pwm(sc, (om, om), level, use_gains)

    def from_pwm(self, sc, speeds, level, use_gains=True):
        C = self.C
        theta = C.max_theta / 255.0 * np.asarray(sc, float) / C.flap_gear
        area = (C.intake - np.cos(C.min_open + theta) * C.flap_len) * 2.0 * C.width
        thrust = np.empty(12)
        for i in (0, 1):
            sl = slice(6 * i, 6 * i + 6)
            tot = C.leak[i] + float((C.cd[sl] * area[sl]).sum())
            k = min(max(int(np.searchsorted(C.area_x, tot, side='right') - 1), 0), len(C.area_x) - 2)
            r = (tot - C.area_x[k]) / (C.area_x[k + 1] - C.area_x[k])
            cdp = C.area_y[k] if tot < C.area_x[0] else C.area_y[-1] if tot > C.area_x[-1] else (1 - r) * C.area_y[k] + r * C.area_y[k + 1]
            om = float(speeds[i])
            dp = om * om * cdp * C.diam ** 2 * C.rho
            thrust[sl] = 2.0 * dp * C.cd[sl] ** 2 * area[sl]
        Fd = thrust @ self.t2f
        Td = thrust @ self.t2t
        if use_gains:
            gf, gt = self.g[max(1, level)]
            Fd = np.where(Fd >= 0, gf[0::2], gf[1::2]) * Fd
            Td = np.where(Td >= 0, gt[0::2], gt[1::2]) * Td
        return (Fd, Td)

class AttFilter:

    def __init__(self, q0, sd_gyro, s7, s9, Om, Rm, delay):
        self.q = rot.qnormalize(q0)
        self.bg = np.zeros(3)
        self.P = np.diag([0.05 ** 2] * 3 + [0.003 ** 2] * 3)
        self.t = None
        self.sd = sd_gyro
        self.s7, self.s9 = (s7, s9)
        self.Om = Om
        self.Rm = Rm
        self.delay = delay if s7 else 0.0
        self.w = np.zeros(3)
        self.bg_frozen = False
        self.samples = []
        self.snaps = []
        self.S_cum = np.zeros((3, 3))
        self.g = self.q.copy()

    def _prop(self, t_eff, rate):
        if self.t is None:
            self.t = t_eff
            return
        if t_eff <= self.t:
            return
        dt = t_eff - self.t if self.s7 else 0.01
        wb = self.Rm @ rate - self.bg
        R = rot.qmat(self.q)
        wrel = wb - (R.T @ self.Om if self.s9 else 0.0)
        self.q = rot.qnormalize(rot.qmul(self.q, rot.qexp(wrel * dt)))
        self.g = rot.qnormalize(rot.qmul(self.g, rot.qexp(wrel * dt)))
        self.w = wrel
        F = np.eye(6)
        F[:3, 3:] = -R * dt
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * self.sd ** 2 * 0.009 * dt
        Q[3:, 3:] = np.eye(3) * 2e-06 ** 2 * dt
        self.P = F @ self.P @ F.T + Q
        self.S_cum = self.S_cum + R * dt
        self.t = t_eff

    def add_samples(self, ins):
        for s in ins:
            te = s['stamp'] - self.delay
            r = np.asarray(s['rate'], float)
            self.samples.append((te, r))
            self._prop(te, r)
        if len(self.samples) > 800:
            del self.samples[:200]

    def snapshot(self, t):
        self.snaps.append((t, self.q.copy(), self.bg.copy(), self.P.copy(), len(self.samples), self.t, self.w.copy(), self.S_cum.copy(), self.g.copy()))
        if len(self.snaps) > 300:
            del self.snaps[:50]

    def at_time(self, t):
        k = None
        for i in range(len(self.snaps) - 1, -1, -1):
            if self.snaps[i][5] is not None and self.snaps[i][5] <= t:
                k = i
                break
        if k is None:
            return self.q.copy()
        _, q, bg, _, n, ts, w = self.snaps[k][:7]
        q = q.copy()
        tl = ts
        for te, r in self.samples:
            if te <= tl:
                continue
            if te > t:
                break
            wb = self.Rm @ r - bg
            R = rot.qmat(q)
            wrel = wb - (R.T @ self.Om if self.s9 else 0.0)
            q = rot.qnormalize(rot.qmul(q, rot.qexp(wrel * (te - tl if self.s7 else 0.01))))
            tl = te
        return q

    def update(self, q_meas, sd):
        e = rot.qlog(rot.qmul(q_meas, rot.qconj(self.q)))
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        S = self.P[:3, :3] + np.eye(3) * sd * sd
        K = self.P[:, :3] @ np.linalg.inv(S)
        dx = K @ e
        self.q = rot.qnormalize(rot.qmul(rot.qexp(dx[:3]), self.q))
        if not self.bg_frozen:
            self.bg = self.bg + dx[3:]
        self.P = (np.eye(6) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def innovation(self, q_meas, q_est, sd):
        e = rot.qlog(rot.qmul(q_meas, rot.qconj(q_est)))
        S = self.P[:3, :3] + np.eye(3) * (sd * sd + GATE_FLOOR_ATT ** 2)
        return (e, float(e @ np.linalg.solve(S, e)))

    def rollback_update(self, t_img, q_meas, sd):
        k = None
        for i in range(len(self.snaps) - 1, -1, -1):
            if self.snaps[i][5] is not None and self.snaps[i][5] <= t_img:
                k = i
                break
        if k is None:
            self.update(q_meas, sd)
            return
        _, q, bg, P, n, ts, w = self.snaps[k][:7]
        later = [s for s in self.samples if s[0] > ts]
        self.q, self.bg, self.P, self.t, self.w = (q.copy(), bg.copy(), P.copy(), ts, w.copy())
        done = False
        for te, r in later:
            if not done and te > t_img:
                self.update(q_meas, sd)
                done = True
            self._prop(te, r)
        if not done:
            self.update(q_meas, sd)
        del self.snaps[k + 1:]

    def delayed_pred(self, t_img):
        k = None
        for i in range(len(self.snaps) - 1, -1, -1):
            if self.snaps[i][5] is not None and self.snaps[i][5] <= t_img:
                k = i
                break
        if k is None:
            H = np.zeros((3, 6))
            H[:, :3] = np.eye(3)
            return (self.q.copy(), H)
        sn = self.snaps[k]
        ts, Sk, gk = (sn[5], sn[7], sn[8])
        g, S_i, tl = (gk.copy(), Sk.copy(), ts)
        bg = sn[2]
        for te, r in self.samples:
            if te <= ts:
                continue
            if te > t_img:
                break
            dt = te - tl if self.s7 else 0.01
            R = rot.qmat(g)
            wrel = self.Rm @ r - bg - (R.T @ self.Om if self.s9 else 0.0)
            S_i = S_i + R * dt
            g = rot.qnormalize(rot.qmul(g, rot.qexp(wrel * dt)))
            tl = te
        delta = rot.qmul(rot.qconj(g), self.g)
        q_img = rot.qnormalize(rot.qmul(self.q, rot.qconj(delta)))
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        H[:, 3:] = self.S_cum - S_i
        return (q_img, H)

    def innovation_delayed(self, q_meas, t_img, sd):
        q_img, H = self.delayed_pred(t_img)
        e = rot.qlog(rot.qmul(q_meas, rot.qconj(q_img)))
        S = H @ self.P @ H.T + np.eye(3) * (sd * sd + GATE_FLOOR_ATT ** 2)
        return (e, float(e @ np.linalg.solve(S, e)), q_img)

    def update_delayed(self, t_img, q_meas, sd):
        q_img, H = self.delayed_pred(t_img)
        if self.bg_frozen:
            H[:, 3:] = 0.0
        e = rot.qlog(rot.qmul(q_meas, rot.qconj(q_img)))
        S = H @ self.P @ H.T + np.eye(3) * sd * sd
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ e
        self.q = rot.qnormalize(rot.qmul(rot.qexp(dx[:3]), self.q))
        if not self.bg_frozen:
            self.bg = self.bg + dx[3:]
            self.w = self.w - dx[3:]
        self.P = (np.eye(6) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

class TransFilter:

    def __init__(self, p0, form, mass, Om, Rm, lever, sd_acc, q_model_rel, s3, s5, s6, s9, sig_g=None):
        self.form = form
        self.x = np.zeros(15)
        self.x[:3] = p0
        self.use_g = sig_g is not None
        sg = np.asarray(sig_g if sig_g is not None else [0.0, 0.0, 0.0], float)
        self.P = np.diag([0.05 ** 2] * 3 + [0.01 ** 2] * 3 + [0.5 ** 2] * 3 + [0.03 ** 2] * 3 + list(sg ** 2 + 1e-12))
        self.m = mass
        self.Om = Om
        self.Rm = Rm
        self.RmT = Rm.T
        self.r = lever
        self.sd_acc = np.asarray(sd_acc, float)
        self.qrel = q_model_rel
        self.s3, self.s5, self.s6, self.s9 = (s3, s5, s6, s9)
        self.hist = []
        self.ba_frozen = None
        ao = pub('told.accel_offset')['change_per_minute_max_mm_s2'] * 0.001
        fc = pub('told.steady_force_change_mN_per_min') * 0.001
        self.q_ba = (ao / 3.0) ** 2 / 60.0
        self.q_d = (fc / 3.0) ** 2 / 60.0
        self.shadows = []

    def predict_xp(self, x, P, dt, R, Feff, z_acc, n_acc, alpha, w, in_control=True):
        form = self.form if in_control else 'accel'
        p, v, ba, d = (x[:3], x[3:6], x[6:9], x[9:12])
        Om = self.Om if self.s9 else np.zeros(3)
        cor = -2.0 * _cross(Om, v) - _cross(Om, _cross(Om, p))
        lev = _cross(alpha, self.r) + _cross(w, _cross(w, self.r)) if self.s6 else np.zeros(3)
        g = x[12:15] if self.use_g else np.zeros(3)
        Fg = Feff * (1.0 + g)
        if form == 'accel' and n_acc > 0:
            fb = self.Rm @ (z_acc - ba) - lev
            a = R @ fb + cor
        else:
            a = R @ ((Fg + (d if self.s5 else 0.0)) / self.m) + cor
        xn = x.copy()
        xn[:3] = p + v * dt + 0.5 * a * dt * dt
        xn[3:6] = v + a * dt
        F = _I15.copy()
        F[:3, 3:6] = np.eye(3) * dt
        if form == 'accel' and n_acc > 0:
            F[3:6, 6:9] = -R @ self.Rm * dt
            F[:3, 6:9] = -0.5 * R @ self.Rm * dt * dt
        else:
            if self.s5:
                F[3:6, 9:12] = R * dt / self.m
                F[:3, 9:12] = 0.5 * R * dt * dt / self.m
            if self.use_g:
                Dg = R @ np.diag(Feff) / self.m
                F[3:6, 12:15] = Dg * dt
                F[:3, 12:15] = 0.5 * Dg * dt * dt
        Q = np.zeros((15, 15))
        if form == 'accel' and n_acc > 0:
            qa = self.sd_acc.max() ** 2 * 0.009
        else:
            amag = float(np.linalg.norm(Feff)) / self.m
            qa = 0.0005 ** 2 + (self.qrel * amag) ** 2
        Q[3:6, 3:6] = np.eye(3) * qa * dt
        Q[:3, :3] = np.eye(3) * 1e-10 * dt
        Q[6:9, 6:9] = np.eye(3) * self.q_ba * dt * (1.0 if self.s3 else 0.0)
        Q[9:12, 9:12] = np.eye(3) * self.q_d * dt * (1.0 if self.s5 and form == 'command' else 0.0)
        Q[12:15, 12:15] = np.eye(3) * 0.0001 ** 2 * dt * (1.0 if self.use_g else 0.0)
        x = xn
        P = F @ P @ F.T + Q
        if not self.s3 and self.ba_frozen is not None:
            x[6:9] = self.ba_frozen
        if form == 'command' and n_acc > 0:
            gg = x[12:15] if self.use_g else np.zeros(3)
            h = self.RmT @ ((Feff * (1.0 + gg) + (x[9:12] if self.s5 else 0.0)) / self.m + lev) + x[6:9]
            H = np.zeros((3, 15))
            H[:, 6:9] = np.eye(3)
            if self.s5:
                H[:, 9:12] = self.RmT / self.m
            if self.use_g:
                H[:, 12:15] = self.RmT @ np.diag(Feff) / self.m
            amag = float(np.linalg.norm(Feff)) / self.m
            lev_u = 0.3 * float(np.linalg.norm(alpha)) * float(np.linalg.norm(self.r)) if self.s6 else 0.0
            Rn = np.diag(self.sd_acc ** 2 / n_acc + (0.05 * amag) ** 2 + lev_u ** 2 + 0.0001 ** 2)
            if not self.s3 and self.ba_frozen is not None:
                H[:, 6:9] = 0.0
                h = h - x[6:9] + self.ba_frozen
            S = H @ P @ H.T + Rn
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z_acc - h)
            P = (_I15 - K @ H) @ P
            P = 0.5 * (P + P.T)
            if not self.s3 and self.ba_frozen is not None:
                x[6:9] = self.ba_frozen
        return (x, P)

    def predict(self, dt, R, Feff, z_acc, n_acc, alpha, w, in_control=True):
        sd_acc = self.sd_acc
        self.x, self.P = self.predict_xp(self.x, self.P, dt, R, Feff, z_acc, n_acc, alpha, w, in_control)
        self.sd_acc = sd_acc
        if in_control:
            for sh in self.shadows:
                sh.step(dt, R, Feff, n_acc, self)

    def lin_update(self, idx, z, sd):
        H = np.zeros((3, 15))
        H[:, idx] = np.eye(3)
        S = H @ self.P @ H.T + np.eye(3) * sd * sd
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(15) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def pos_update_xp(self, x, P, z, sd):
        H = np.zeros((3, 15))
        H[:, :3] = np.eye(3)
        S = P[:3, :3] + np.eye(3) * sd * sd
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z - x[:3])
        P = (np.eye(15) - K @ H) @ P
        P = 0.5 * (P + P.T)
        if not self.s3 and self.ba_frozen is not None:
            x[6:9] = self.ba_frozen
        return (x, P)

    def pos_update(self, z, sd):
        self.x, self.P = self.pos_update_xp(self.x, self.P, z, sd)

    def maha(self, z, sd, x=None, P=None):
        x = self.x if x is None else x
        P = self.P if P is None else P
        e = z - x[:3]
        S = P[:3, :3] + np.eye(3) * (sd * sd + GATE_FLOOR_POS ** 2)
        return float(e @ np.linalg.solve(S, e))

class Shadow:

    def __init__(self, tf, t0, mode, limit, sig_G, infl):
        P = np.zeros((15, 15))
        P[:12, :12] = tf.P[:12, :12] * infl
        P[12:, 12:] = np.diag(np.asarray(sig_G, float) ** 2)
        self.P = P
        self.t0, self.t, self.mode, self.limit = (t0, t0, mode, limit)
        self.cross = None
        self.done = False

    def step(self, dt, R, Feff, n_acc, tf):
        if self.done:
            return
        F = np.eye(15)
        F[:3, 3:6] = np.eye(3) * dt
        F[3:6, 9:12] = R * dt / tf.m
        F[:3, 9:12] = 0.5 * R * dt * dt / tf.m
        D = np.diag(Feff) / tf.m
        F[3:6, 12:15] = R @ D * dt
        F[:3, 12:15] = 0.5 * R @ D * dt * dt
        Q = np.zeros((15, 15))
        Q[3:6, 3:6] = np.eye(3) * 0.0002 ** 2 * dt
        Q[9:12, 9:12] = np.eye(3) * tf.q_d * dt
        Q[6:9, 6:9] = np.eye(3) * tf.q_ba * dt
        self.P = F @ self.P @ F.T + Q
        if n_acc > 0:
            H = np.zeros((3, 15))
            H[:, 6:9] = np.eye(3)
            H[:, 9:12] = tf.RmT / tf.m
            H[:, 12:15] = tf.RmT @ D
            Rn = np.diag(tf.sd_acc ** 2 / n_acc + 0.0001 ** 2)
            S = H @ self.P @ H.T + Rn
            K = self.P @ H.T @ np.linalg.inv(S)
            self.P = (np.eye(15) - K @ H) @ self.P
            self.P = 0.5 * (self.P + self.P.T)
        self.t += dt
        s3 = 3.0 * math.sqrt(max(0.0, float(max(np.linalg.eigvalsh(self.P[:3, :3])))))
        if s3 >= self.limit:
            self.cross = self.t - self.t0
            self.done = True
        elif self.t - self.t0 > 90.0:
            self.cross = None
            self.done = True

def checkout_numbers():
    txt = pub('cert.checkout_sequence')
    m_rest = re.search('([0-9.]+) s at rest', txt)
    m_pulse = re.search('one torque pulse of ([0-9.]+) s on each of the six signed axes, each followed by its return', txt)
    m_burn = re.search('one out-and-back burn of about ([0-9.]+) s on each of the six signed force axes', txt)
    m_tot = re.search('([0-9.]+) s in all', txt)
    assert m_rest and m_pulse and m_burn and m_tot
    return tuple((float(m.group(1)) for m in (m_rest, m_pulse, m_burn, m_tot)))

class RSense:

    def __init__(self, switches=None, form='command', uncertain=None, hold_zero=False, shadows=False):
        self.on = {g: g not in DEFAULT_OFF for g in GROUPS + ALLOWED}
        if switches:
            for k in switches:
                if k not in self.on:
                    raise ValueError('unknown switch %r' % k)
            self.on.update(switches)
        self.form = form
        self.uncertain = uncertain
        self.hold_zero = hold_zero
        self.want_shadows = shadows
        self.shadow_results = []
        self.stats = {'fixes': 0, 'accepted': 0, 'rejected': 0, 'candidates': 0, 'rollbacks': 0, 'pairs': 0, 'pairs_after_refusal': 0, 'resends': 0, 'replay_steps': 0}
        self.event_records = []
        self.fix_records = []
        self.lag_record = None

    def setup(self, paper, reads):
        on = self.on
        self.paper = paper
        sgn = self.uncertain
        rep = np.array([0.07, 0.07, 0.27])
        er = reads['effective_response']
        self.resp = {}
        for lv in (1, 2, 3):
            src = _lv(er, 2) if lv == 1 else _lv(er, lv)
            f = np.array([float(src['force'][s]) for s in SIGNED])
            t = np.array([float(src['torque'][s]) for s in SIGNED])
            if sgn:
                f = f * (1 + sgn * np.repeat(rep, 2))
                t = t * (1 + sgn * np.repeat(rep, 2))
            self.resp[lv] = (f, t)
        mf = pub('world.mass_file')
        if on['S4']:
            self.m = float(reads['mass']['mass_kg'])
        else:
            self.m = float(mf['mass_kg'])
        self.mass_ctl = float(reads['mass']['mass_kg'])
        self.J = np.asarray(reads['mass']['inertia'], float)
        self.com = np.asarray(reads['mass']['com_m'], float)
        nz = 1.0 + (sgn * 0.3 if sgn else 0.0)
        self.imu_sd = {}
        for lv in (1, 2, 3):
            v = _lv(reads['noise']['imu'], lv)
            self.imu_sd[lv] = (np.asarray(v[0], float) * nz, np.asarray(v[1], float) * nz)
        fx = reads['noise']['fix']
        self.fix_sd = {'ge20': (fx['ge20'][0] * nz, fx['ge20'][1] * nz), 'lt20': (fx['lt20'][0] * nz, fx['lt20'][1] * nz)}
        self.p_gross = {k: v * nz for k, v in reads['p_gross'].items()}
        lr = pub('told.lag_command_to_motion_ms')
        self.lag_prior = 0.5 * (float(lr[0]) + float(lr[1])) / 1000.0
        self.lag = self.lag_prior
        cr = pub('told.cabin_rotation')
        self.Om = 0.5 * (cr['rate_mrad_s'][0] + cr['rate_mrad_s'][1]) * 0.001 * np.array([0.0, -1.0, 0.0])
        im = pub('world.imu.mount')
        self.Rm = rot.rotz(90.0)
        self.lever = np.asarray(im['position_m'], float) - self.com
        fil = pub('world.imu.device_filter')
        self.dev_delay = float(fil['delay_ms']) / 1000.0
        fr = pub('world.fix.frame')['camera_mount']
        self.cam_t = np.asarray(fr['position_m'], float)
        self.cam_q = rot.qnormalize(fr['quaternion_xyzw'])
        self.mount = np.asarray(pub('jobs.camera_mount_m'), float)
        self.dt = float(pub('interface.step_period_ms')) / 1000.0
        self.fan = pub('mode.fan_level')
        self.tol = {m: pub('monitor.tolerance.' + m) for m in MODES}
        self.capture = float(pub('jobs.capture_s'))
        self.sb_caps = {m: float(c) for m, c in pub('sessions.outage_cap_s').items()} if on['SB'] else None
        self.sb_fix_interval_max = float(pub('told.fix_attempt_interval_s')[1]) if on['SB'] else None
        self.sb_fix_delay_max = float(pub('told.fix_delay_s')[1]) if on['SB'] else None
        self.dur_cache = {}
        self.later_events = []
        C = Constants()
        al = Allocator(C, self.com)
        raw = NominalModel(C, al, None)
        self.gains = {}
        self.model_ratio = {}
        for lv in (1, 2, 3):
            ratios = np.zeros(12)
            for ax in range(6):
                for si, sg in enumerate((1.0, -1.0)):
                    W = np.zeros(6)
                    W[ax] = sg * (0.1 if ax < 3 else 0.01)
                    Fm, Tm = raw(W[:3], W[3:], lv, use_gains=False)
                    ratios[2 * ax + si] = float(np.concatenate([Fm, Tm])[ax] / W[ax])
            self.model_ratio[lv] = ratios
            f, tq = self.resp[lv]
            self.gains[lv] = (f / ratios[:6], tq / ratios[6:])
        self.nominal = NominalModel(C, al, self.gains)
        self.cap = {}
        for lv in (1, 2, 3):
            caps = []
            for ax in range(6):
                for sg in (1, -1):
                    lo, hi = (0.0, 0.6 if ax < 3 else 0.06)
                    for _ in range(30):
                        mid = 0.5 * (lo + hi)
                        W = np.zeros(6)
                        W[ax] = sg * mid
                        sc, _, _ = al.allocate(W[:3], W[3:], lv)
                        if sc.max() >= 255:
                            hi = mid
                        else:
                            lo = mid
                    caps.append(lo)
            self.cap[lv] = np.array(caps).reshape(6, 2)
        sp = paper['start_pose']
        q0 = rot.qnormalize(sp['attitude'])
        self.att = AttFilter(q0, float(np.max(self.imu_sd[2][0])), on['S7'], on['S9'], self.Om, self.Rm, self.dev_delay)
        self.tf = TransFilter(np.asarray(sp['position'], float) + rot.qmat(q0) @ self.com, self.form, self.m, self.Om, self.Rm, self.lever, self.imu_sd[2][1], 0.03, on['S3'], on['S5'], on['S6'], on['S9'], sig_g=(rep if sgn else np.full(3, 0.03)) if on['S4'] else None)
        self.cmd_hist = []
        self.level = self.fan['nominal']
        self.job = None
        self.plan_sps = None
        self.prev_goal = (np.asarray(sp['position'], float), q0)
        self.start_pose = (np.asarray(sp['position'], float), q0)
        self.int_p = np.zeros(3)
        self.int_a = np.zeros(3)
        self.last_fix_img = -1000000000.0
        self.candidate = None
        self.prev_rejected = False
        self.t_ready_wait = None
        self.block_mode = 'nominal'
        self.mode_change_t = None
        self.last_cmd = np.zeros(6)
        self.heavy = False
        self.shadow_mode = None
        self.in_block = False
        self.was_in_control = False
        self.last_shadow_t = -1000000000.0
        self.accept_log = []
        self.init_done = False
        self.init_samples = []
        self.hold_steps = 0
        self.start_hold_s = float(pub('sessions.start_hold_s'))
        self.was_in_control_prev = False
        self.handover_pose = (np.asarray(sp['position'], float), q0, 0.1 / 3.0)
        self.t_last_est = -1000000000.0
        self.saw_not_in_control = False
        self.t_now = 0.0
        self.last_goal = None
        self.mode_from = 'nominal'
        self.job_index = {}
        for bi, b in enumerate(paper['blocks']):
            for ji, j in enumerate(b['jobs']):
                self.job_index[j['id']] = (bi, ji)
        self.co_sps = None
        self.co_t0 = None
        self.co_pulse_span = None
        self.co_cmd = []
        self.co_imu = []
        self.lag_done = False
        self.ev = None
        self.pending_event = None
        self.replay = None
        self.fix_queue = []
        self.last_innov = []
        self.time_left = None
        self.zv_due = False
        self.handover_due = False
        if on['EV'] and on['SB']:
            self._precompute_durations()
        self.zones, self.zone_record = rzone.load_for_paper(paper)
        self.zone_radius = float(pub('safety.body_radius_m')) if self.zones is not None else None
        self.zone_record['refused'] = []
        return {'mass_profile': {'mass_kg': self.mass_ctl, 'com_m': self.com.tolist(), 'inertia': self.J.tolist()}}

    def update_paper(self, paper):
        self.paper = paper
        for bi, b in enumerate(paper['blocks']):
            for ji, j in enumerate(b['jobs']):
                self.job_index[j['id']] = (bi, ji)
        if self.on['EV'] and self.on['SB']:
            self._precompute_durations()

    @staticmethod
    def _flown_kind(j):
        if j['kind'] == 'request':
            return 'leg' if j.get('request_shape') == 'leg' else 'view'
        return j['kind']

    def _plan_duration(self, j, p0, q0):
        key = (j['id'], tuple(np.round(np.asarray(p0, float), 9)), tuple(np.round(np.asarray(q0, float), 9)))
        D = self.dur_cache.get(key)
        if D is not None:
            return D
        on = self.on
        af = 1.0 if on['I'] else 0.5
        lim = limits(j['mode'], af)
        jj = dict(j, kind=self._flown_kind(j))
        p1, q1 = self._body_goal(jj)
        pm = fastplan.pose_move if on['RT'] else traj.pose_move
        if jj['kind'] == 'view' and on['F']:
            sps = pm(p0, q0, p1, q1, lim, 0.0, pivot_body=self.mount, dt_sp=0.1)
        elif jj['kind'] == 'view':
            sps = pm(p0, q0, p0, q1, lim, 0.0)
        else:
            sps = pm(p0, q0, p1, q1, lim, 0.0)
        D = float(sps[-1][0])
        self.dur_cache[key] = D
        return D

    def _precompute_durations(self):
        for b in self.paper['blocks']:
            sp = b['start_pose']
            p0, q0 = (np.asarray(sp['position'], float), rot.qnormalize(sp['attitude']))
            for j in b['jobs']:
                if j['kind'] == 'hold' or j['mode'] not in self.paper['release_modes']:
                    continue
                self._plan_duration(j, p0, q0)
                p0, q0 = self._body_goal(dict(j, kind=self._flown_kind(j)))

    def _need_after_exact(self, bi, ji, goal):
        ew = float(pub('score.end_window_s')[1])
        need = 0.0
        modes = []
        p0, q0 = goal
        for j in self.paper['blocks'][bi]['jobs'][ji + 1:]:
            if j['mode'] not in self.paper['release_modes']:
                continue
            if j['kind'] == 'hold':
                need += float(j['goal']['duration_s']) + SB_STEPS * self.dt
                continue
            k = self._flown_kind(j)
            need += self._plan_duration(j, p0, q0) + (self.capture if k == 'view' else ew) + SB_STEPS * self.dt
            modes.append(j['mode'])
            p0, q0 = self._body_goal(dict(j, kind=k))
        return (need, modes)

    def _fix_wait_at_cap(self, mode):
        cap = self.sb_caps.get(mode, self.sb_caps.get('nominal'))
        return (cap if cap > 0 else self.sb_fix_interval_max) - 0.3 + self.sb_fix_delay_max

    def _checkout_plan(self, t0):
        rest, pulse, burn, total = checkout_numbers()
        dt = self.dt
        n_rest, n_pulse, n_q = (int(round(rest / dt)), int(round(pulse / dt)), int(round(0.25 * burn / dt)))
        v_l, a_l, w_l, al_l = limits('nominal')
        alpha = min(al_l, w_l / (n_pulse * dt))
        acc = min(a_l, v_l / (n_q * dt))
        p0, q0 = self.start_pose
        R0 = rot.qmat(q0)
        segs = [(n_rest * dt, np.zeros(3), np.zeros(3))]
        for i in range(3):
            for sg in (1.0, -1.0):
                e = np.zeros(3)
                e[i] = sg * alpha
                segs += [(n_pulse * dt, np.zeros(3), e), (n_pulse * dt, np.zeros(3), -e)]
        for i in range(3):
            for sg in (1.0, -1.0):
                e = np.zeros(3)
                e[i] = sg * acc
                segs += [(n_q * dt, e, np.zeros(3)), (2 * n_q * dt, -e, np.zeros(3)), (n_q * dt, e, np.zeros(3))]
        sps = []
        t, p, v, q, w = (t0, p0.copy(), np.zeros(3), q0.copy(), np.zeros(3))
        pulse_t = [None, None]
        for k, (T, a_b, al_b) in enumerate(segs):
            a_c = R0 @ a_b
            sps.append((t, p.copy(), v.copy(), a_c.copy(), q.copy(), w.copy(), al_b.copy()))
            if k == 1:
                pulse_t[0] = t
            if k == 12:
                pulse_t[1] = t + T
            p = p + v * T + 0.5 * a_c * T * T
            v = v + a_c * T
            q = rot.qnormalize(rot.qmul(q, rot.qexp(w * T + 0.5 * al_b * T * T)))
            w = w + al_b * T
            t = t + T
        sps.append((t, p.copy(), np.zeros(3), np.zeros(3), q.copy(), np.zeros(3), np.zeros(3)))
        self.co_pulse_span = (pulse_t[0], pulse_t[1])
        return sps

    def _lag_from_checkout(self):
        rec = {'prior_s': self.lag_prior, 'used_s': self.lag_prior, 'why': None}
        try:
            if not self.co_cmd or self.co_pulse_span is None or self.co_pulse_span[0] is None:
                rec['why'] = 'no check-out commands'
                return rec
            tc = np.array([c[0] for c in self.co_cmd])
            Tc = np.array([c[1] for c in self.co_cmd])
            t_a, t_b = (self.co_pulse_span[0], min(self.co_pulse_span[1] + 2.0, float(tc[-1])))
            if t_b < t_a + 3.0:
                rec['why'] = 'the check-out ended before one whole pulse and its return'
                return rec
            S = [(s, r) for s, r in self.co_imu if t_a - 0.5 <= s <= t_b]
            if len(S) < 200:
                rec['why'] = 'fewer than 200 samples (%d)' % len(S)
                return rec
            st = np.array([s for s, _ in S])
            y = np.array([self.Rm @ r for _, r in S])
            edges = np.concatenate([tc, [tc[-1] + self.dt]])
            Jd = np.diag(self.J)
            cp = np.vstack([np.zeros(3), np.cumsum(np.maximum(Tc, 0.0) * self.dt, 0)]) / Jd
            cn = np.vstack([np.zeros(3), np.cumsum(np.minimum(Tc, 0.0) * self.dt, 0)]) / Jd
            rss = []
            for L in LAG_GRID:
                tot = 0.0
                for i in range(3):
                    xp = np.interp(st - L, edges, cp[:, i])
                    xn = np.interp(st - L, edges, cn[:, i])
                    X = np.stack([np.ones(len(st)), xp, xn], 1)
                    coef, *_ = np.linalg.lstsq(X, y[:, i], rcond=None)
                    tot += float(((X @ coef - y[:, i]) ** 2).sum())
                rss.append(tot)
            rss = np.array(rss)
            k = int(np.argmin(rss))
            rec['grid_min_s'] = LAG_GRID[k]
            rec['rss'] = rss.tolist()
            if not np.all(np.isfinite(rss)) or k == 0 or k == len(LAG_GRID) - 1:
                rec['why'] = 'the fit cannot be made (minimum at an end of the grid or a non-finite residual)'
                return rec
            a, b, c = (rss[k - 1], rss[k], rss[k + 1])
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if den > 0 else 0.0
            L = LAG_GRID[k] + max(-1.0, min(1.0, off)) * 0.004
            rec['used_s'] = float(L)
            rec['why'] = 'check-out'
            self.lag = float(L)
            return rec
        finally:
            self.lag_record = rec

    def _body_goal(self, job):
        if job['kind'] == 'view':
            g = job['goal']
            a = np.asarray(g['axis'], float)
            u = np.asarray(g['up'], float)
            ez = -u
            ey = np.cross(ez, a)
            R = np.stack([a, ey, ez], 1)
            q = rot.qfrom_mat(R)
            return (np.asarray(g['centre'], float) - R @ self.mount, q)
        return (np.asarray(job['goal']['position'], float), rot.qnormalize(job['goal']['attitude']))

    def _zone_breach(self, job):
        if self.zones is None or job['kind'] not in ('leg', 'slew', 'view'):
            return None
        pg = np.asarray(self._body_goal(job)[0], float)
        why = self.zones.breach(pg, self.zone_radius)
        if why is not None:
            self.zone_record['refused'].append((job['id'], why, float(min(self.zones.point(pg))) - self.zone_radius))
        return why

    def plan(self, job, frame):
        on = self.on
        if not self.lag_done:
            self.lag_done = True
            if on['CO']:
                self._lag_from_checkout()
        bi0, ji0 = self.job_index[job['id']]
        if ji0 == 0:
            sp0 = self.paper['blocks'][bi0]['start_pose']
            self.handover_pose = (np.asarray(sp0['position'], float), rot.qnormalize(sp0['attitude']), 0.1 / 3.0)
        elif self.saw_not_in_control and self.last_goal is not None:
            self.handover_pose = (self.last_goal[0], self.last_goal[1], 0.02)
        self.saw_not_in_control = False
        self._ingest(frame)
        self.pending_event = None
        if self._zone_breach(job) is not None:
            return {'refuse': 'outside_zones'}
        if job['mode'] not in self.paper['release_modes']:
            return {'refuse': 'mode_not_available'}
        mode = job['mode']
        self.block_mode = frame['mode']
        t0 = frame['t']
        lvc = int(frame['fan_level_commanded'])
        if lvc != self.level:
            self.mode_change_t = t0
            self.mode_from = [m for m in MODES if self.fan[m] == self.level][0]
            self.level = lvc
        bi, ji = self.job_index[job['id']]
        if ji == 0:
            sp = self.paper['blocks'][bi]['start_pose']
            self.prev_goal = (np.asarray(sp['position'], float), rot.qnormalize(sp['attitude']))
            self.last_goal = None
        elif self.last_goal is not None:
            self.prev_goal = self.last_goal
        p0, q0 = self.prev_goal
        self.job = job
        self.t_ready_wait = None
        self.ev = None
        if job['kind'] == 'hold':
            self.plan_sps = [(t0, p0, np.zeros(3), np.zeros(3), q0, np.zeros(3), np.zeros(3))]
            self.job_end_plan = t0
            return {'setpoints': []}
        delay = 0.0
        if self.mode_change_t is not None and on['E2']:
            S = self._spin_up(self.mode_from, mode)
            delay = max(0.0, self.mode_change_t + S - t0)
        self.mode_change_t = None
        af = 1.0 if on['I'] else 0.5
        lim = limits(mode if on['B'] else mode, af)
        p1, q1 = self._body_goal(job)
        pm = fastplan.pose_move if on['RT'] else traj.pose_move
        if job['kind'] == 'view':
            if on['F']:
                sps = pm(p0, q0, p1, q1, lim, t0, pivot_body=self.mount, dt_sp=0.1)
            else:
                sps = pm(p0, q0, p0, q1, lim, t0)
                p1 = p0
        else:
            sps = pm(p0, q0, p1, q1, lim, t0)
        if delay > 0:
            sps = [(t0, p0, np.zeros(3), np.zeros(3), q0, np.zeros(3), np.zeros(3))] + [(s[0] + delay,) + tuple(s[1:]) for s in sps]
            fill = []
            k = 1
            while k * 0.9 < delay:
                fill.append((t0 + k * 0.9, p0, np.zeros(3), np.zeros(3), q0, np.zeros(3), np.zeros(3)))
                k += 1
            sps = [sps[0]] + fill + sps[1:]
            sps = [s for i, s in enumerate(sps) if i == 0 or s[0] > sps[i - 1][0] + 1e-09]
        self.plan_sps = sps
        self.job_end_plan = sps[-1][0]
        self.goal_now = (p1, q1)
        self.last_goal = self._body_goal(job)
        if on['EV'] and on['SB']:
            self.need_after, self.later_events = self._need_after_exact(bi, ji, self.last_goal)
        elif on['EV']:
            self.need_after = self._need_after(bi, ji, self.last_goal)
        return {'setpoints': sps}

    def _need_after(self, bi, ji, goal):
        ew = float(pub('score.end_window_s')[1])
        need = 0.0
        p0, q0 = goal
        for j in self.paper['blocks'][bi]['jobs'][ji + 1:]:
            if j['mode'] not in self.paper['release_modes']:
                continue
            if j['kind'] == 'hold':
                need += float(j['goal']['duration_s'])
                continue
            p1, q1 = self._body_goal(j)
            v_l, a_l, w_l, al_l = limits(j['mode'])
            if j['kind'] == 'view' and self.on['F']:
                T = fastplan.duration(p0, q0, p1, q1, (v_l, a_l, w_l, al_l), pivot_body=self.mount) + EV_VIEW_PLAN_MARGIN
            else:
                th = rot.qangle(q0, q1)
                T = max(traj.trap_profile(float(np.linalg.norm(p1 - p0)), a_l, v_l)[0], traj.trap_profile(th, al_l, w_l)[0])
            need += T + (self.capture if j['kind'] == 'view' else ew)
            p0, q0 = (p1, q1)
        return need

    def _spin_up(self, mfrom, mto):
        txt = pub('jobs.spin_up_term')
        assert 'n = 2000, 2500, 2800 rpm' in txt
        rpm = {1: 2000.0, 2: 2500.0, 3: 2800.0}
        return abs(rpm[self.fan[mto]] - rpm[self.fan[mfrom]]) / 200.0 + 2.0

    def _feff(self, t):
        tl = t - self.lag
        F = np.zeros(3)
        T = np.zeros(3)
        for tc, Fc, Tc, lv, pred in reversed(self.cmd_hist):
            if tc <= tl + 1e-09:
                return pred
        return (F, T)

    def _estimate_step(self, fr):
        on = self.on
        t = fr['t']
        ins = fr['inertial']
        self.att.add_samples(ins)
        self.att.snapshot(t)
        if fr.get('job_id') == 'checkout' and on['CO']:
            for s in ins:
                self.co_imu.append((float(s['stamp']), np.asarray(s['rate'], float)))
        lv = max(1, int(fr['fan_level_commanded']))
        self.tf.sd_acc = self.imu_sd[lv][1]
        Feff, Teff = self._feff(t)
        alpha = np.linalg.solve(self.J, Teff - _cross(self.att.w, self.J @ self.att.w))
        w = self.att.w + np.zeros(3)
        if ins:
            z = np.mean([np.asarray(s['accel'], float) for s in ins], 0)
        else:
            z = np.zeros(3)
        R = rot.qmat(self.att.q)
        ic = bool(fr['in_control'])
        if ic and (not self.was_in_control) and self.on['S5']:
            dsd = CO_D_SD if self.on['CO'] and fr.get('job_id') == 'checkout' else 0.003
            self.tf.P[9:12, 9:12] += np.eye(3) * dsd ** 2
            if self.replay is not None:
                self.replay['P'][9:12, 9:12] += np.eye(3) * dsd ** 2
        self.was_in_control = ic
        dt_eff = t - self.t_last_est
        if dt_eff > 1e-09:
            self.tf.predict(min(dt_eff, 0.1), R, Feff, z, len(ins), alpha, w, True)
            self.tf.hist.append((t, self.tf.x.copy(), self.tf.P.copy(), (R, Feff, z, len(ins), alpha, w, ic)))
            if len(self.tf.hist) > 260 and self.replay is None:
                del self.tf.hist[:40]
            self.t_last_est = t
        if not self.init_done:
            self.init_samples += ins
            if self.init_samples and self.init_samples[-1]['stamp'] - self.init_samples[0]['stamp'] >= INIT_S:
                acc = np.mean([s['accel'] for s in self.init_samples], 0)
                rate = np.mean([s['rate'] for s in self.init_samples], 0)
                q0 = self.att.q
                lev0 = np.zeros(3)
                self.tf.x[6:9] = acc - self.Rm.T @ lev0
                self.tf.x[3:6] = 0.0
                for blk, sd_ in ((slice(6, 9), 0.003), (slice(3, 6), 0.002)):
                    self.tf.P[blk, :] = 0.0
                    self.tf.P[:, blk] = 0.0
                    self.tf.P[blk, blk] = np.eye(3) * sd_ ** 2
                self.att.bg = self.Rm @ rate - (rot.qmat(q0).T @ self.Om if on['S9'] else 0.0)
                self.att.P[3:, :] = 0.0
                self.att.P[:, 3:] = 0.0
                self.att.P[3:, 3:] = np.eye(3) * 0.0005 ** 2
                self.init_done = True
                if not on['S3']:
                    self.tf.ba_frozen = self.tf.x[6:9].copy()
                    self.att.bg_frozen = True
        Fh, Th = self.cmd_hist[-1][4] if self.cmd_hist else (np.zeros(3), np.zeros(3))
        holding = not ic and (float(np.linalg.norm(Fh)) < 0.02 and float(np.linalg.norm(Th)) < 0.002 or (on['ZH'] and t < self.start_hold_s))
        self.hold_steps = self.hold_steps + 1 if holding else 0
        if t < self.start_hold_s and self.hold_steps >= int(2.0 / self.dt) and (self.hold_steps % int(0.5 / self.dt) == 0):
            self.zv_due = True
        if self.zv_due and self.replay is None:
            self.zv_due = False
            self.tf.lin_update(slice(3, 6), np.zeros(3), 0.003)
        if ic and (not self.was_in_control_prev) and (self.handover_pose is not None):
            self.handover_due = True
        if self.handover_due and self.replay is None:
            self.handover_due = False
            hp, hq, sd_h = self.handover_pose
            z = hp + rot.qmat(hq) @ self.com
            e = z - self.tf.x[:3]
            if float(np.linalg.norm(e)) > 3.0 * math.sqrt(float(np.trace(self.tf.P[:3, :3])) / 3.0 + sd_h ** 2):
                self.tf.P[:3, :] *= 0.0
                self.tf.P[:, :3] *= 0.0
                self.tf.P[:3, :3] = np.eye(3) * 0.1 ** 2
                self.tf.P[3:6, 3:6] += np.eye(3) * 0.01 ** 2
            self.tf.lin_update(slice(0, 3), z, sd_h)
            self.tf.lin_update(slice(3, 6), np.zeros(3), 0.003)
            self.handover_pose = None
        self.was_in_control_prev = ic
        if on['RT']:
            self._advance_replay()
            self.fix_queue.extend(fr['fixes'])
            while self.fix_queue and self.replay is None:
                f = self.fix_queue.pop(0)
                self._fix(f, t)
        else:
            for f in fr['fixes']:
                self._fix(f, t)
        self.tf.shadows = [s for s in self.tf.shadows if not s.done]

    def _fix(self, f, t):
        on = self.on
        self.stats['fixes'] += 1
        cid = f.get('camera_id')
        if not f['landmarks']:
            self.fix_records.append((t, cid, float(f['stamp']), 0, None, None, 'no_pose'))
            return
        n = int(f['landmarks'])
        cls = 'lt20' if n < 20 and on['S2'] else 'ge20'
        sp, sa = self.fix_sd[cls]
        qc = rot.qnormalize(f['attitude'])
        q_meas = rot.qmul(qc, rot.qconj(self.cam_q))
        t_img = float(f['stamp']) if on['S1'] else t
        if on['S1'] and on['RT']:
            q_at = None
        else:
            q_at = self.att.at_time(t_img) if on['S1'] else self.att.q
        Rq = rot.qmat(q_meas)
        p_meas = np.asarray(f['position'], float) - Rq @ self.cam_t + Rq @ self.com
        k = None
        if on['S1']:
            for i in range(len(self.tf.hist) - 1, -1, -1):
                if self.tf.hist[i][0] <= t_img:
                    k = i
                    break
        if k is None:
            xk, Pk, tk = (self.tf.x, self.tf.P, t)
        else:
            tk, xk, Pk, _ = self.tf.hist[k]
        z = p_meas - xk[3:6] * (t_img - tk)
        outage = t_img - self.last_fix_img > 5.0
        d2p = self.tf.maha(z, sp, xk, Pk)
        if q_at is None:
            _, d2a, q_at = self.att.innovation_delayed(q_meas, t_img, sa)
        else:
            _, d2a = self.att.innovation(q_meas, q_at, sa)
        self.last_innov = (self.last_innov + [{'t': t, 't_img': t_img, 'camera_id': cid, 'landmarks': n, 'd2p': d2p, 'd2a': d2a, 'e_pos': (z - xk[:3]).tolist()}])[-4:]
        gate_ok = d2p < CHI2_3 and d2a < CHI2_3 if on['S2'] else True
        need_confirm = on['S8'] and outage or (on['S2'] and self.prev_rejected)
        trusted = n >= 20 if on['S2'] else True
        rec = lambda dec: self.fix_records.append((t, cid, t_img, n, d2p, d2a, dec))
        if need_confirm or not gate_ok:
            if gate_ok and (trusted or not (on['S8'] and outage)):
                self.candidate = None
                self.prev_rejected = False
                rec('gate')
                self._apply_at(t_img, p_meas, q_meas, sp, sa, on, inflate=False)
                return
            if self.candidate is not None:
                c = self.candidate
                dtc = t_img - c['t_img']
                if on['CS']:
                    if abs(dtc) > CS_MAX_AGE:
                        agree = False
                        self.fix_records.append((t, c['cid'], c['t_img'], c['n'], None, None, 'candidate_dropped_old'))
                    else:
                        vsd = max(math.sqrt(max(0.0, float(max(np.linalg.eigvalsh(Pk[3:6, 3:6]))))), CS_VFLOOR)
                        lim_p = 3.0 * math.sqrt(sp * sp + c['sp'] ** 2) + 3.0 * vsd * abs(dtc) + CS_EXTRA_P
                        pred = c['p'] + (xk[:3] - c['x_f'])
                        dq_f = rot.qmul(q_at, rot.qconj(c['q_f']))
                        pred_q = rot.qmul(dq_f, c['q'])
                        ang = rot.qangle(pred_q, q_meas)
                        agree = np.linalg.norm(p_meas - pred) < lim_p and ang < 3.0 * math.sqrt(sa * sa + c['sa'] ** 2) + CS_EXTRA_A
                else:
                    vsd = max(math.sqrt(max(0.0, float(max(np.linalg.eigvalsh(Pk[3:6, 3:6]))))), 0.003) + 0.002
                    lim_p = 3.0 * math.sqrt(sp * sp + c['sp'] ** 2) + 3.0 * vsd * abs(dtc) + 0.01
                    pred = c['p'] + xk[3:6] * dtc
                    ang = rot.qangle(c['q'], q_meas)
                    agree = np.linalg.norm(p_meas - pred) < lim_p and ang < 3.0 * math.sqrt(sa * sa + c['sa'] ** 2) + 0.02
                if agree:
                    self.stats['pairs'] += 1
                    if c.get('refused'):
                        self.stats['pairs_after_refusal'] += 1
                    self.fix_records.append((t, c['cid'], c['t_img'], c['n'], c['d2p'], c['d2a'], 'pair_older'))
                    rec('pair_newer')
                    self._apply_pair(c, t_img, p_meas, q_meas, sp, sa, on, inflate_older=not gate_ok)
                    self.candidate = None
                    self.prev_rejected = False
                    return
                if self.candidate is not None and (not (on['CS'] and abs(dtc) > CS_MAX_AGE)):
                    self.fix_records.append((t, c['cid'], c['t_img'], c['n'], c['d2p'], c['d2a'], 'candidate_replaced'))
            self.candidate = {'t_img': t_img, 'p': p_meas, 'q': q_meas, 'sp': sp, 'sa': sa, 'k_t': t_img, 'z_t': p_meas, 'q_meas': q_meas, 'cid': cid, 'n': n, 'd2p': d2p, 'd2a': d2a, 'x_f': xk[:3].copy(), 'q_f': np.asarray(q_at, float).copy(), 'refused': not gate_ok}
            self.stats['candidates'] += 1
            rec('candidate' if gate_ok else 'refused')
            if not gate_ok:
                self.stats['rejected'] += 1
                self.prev_rejected = True
            return
        self.candidate = None
        self.prev_rejected = False
        rec('gate')
        self._apply_at(t_img, p_meas, q_meas, sp, sa, on, inflate=False)

    def _apply_pair(self, c, t_img, p_meas, q_meas, sp, sa, on, inflate_older):
        if on['RT']:
            self._apply_at(c['k_t'], c['z_t'], c['q_meas'], c['sp'], c['sa'], on, inflate_older, then=(t_img, p_meas, q_meas, sp, sa))
        else:
            self._apply_at(c['k_t'], c['z_t'], c['q_meas'], c['sp'], c['sa'], on, inflate_older)
            self._apply_at(t_img, p_meas, q_meas, sp, sa, on, inflate=False)

    def _apply_at(self, t_img, p_meas, q_meas, sp, sa, on, inflate, then=None):
        self.stats['accepted'] += 1 + (1 if then is not None else 0)
        tn = self.t_now
        recent = [x for x in self.accept_log if tn - 10.0 <= x[0] < tn]
        steady = len(recent) >= 3 and (not any((x[1] for x in recent))) and (not inflate)
        self.accept_log.append((tn, bool(inflate)))
        if len(self.accept_log) > 50:
            del self.accept_log[:10]
        meas = [(t_img, p_meas, q_meas, sp, sa, inflate)]
        if then is not None:
            meas.append(then + (False,))
        for ti, pm_, qm_, sp_, sa_, _ in meas:
            if on['S1']:
                if on['RT']:
                    self.att.update_delayed(ti, qm_, sa_)
                else:
                    self.att.rollback_update(ti, qm_, sa_)
            else:
                self.att.update(qm_, sa_)
        tf = self.tf
        k = None
        if on['S1']:
            for i in range(len(tf.hist) - 1, -1, -1):
                if tf.hist[i][0] <= t_img:
                    k = i
                    break
        if k is None:
            for ti, pm_, qm_, sp_, sa_, infl in meas:
                if infl:
                    tf.P[:3, :3] += np.eye(3) * 0.1 ** 2
                    tf.P[3:6, 3:6] += np.eye(3) * 0.005 ** 2
                tf.pos_update(pm_, sp_)
        elif not on['RT']:
            tk, xk, Pk, _ = tf.hist[k]
            tf.x, tf.P = (xk.copy(), Pk.copy())
            if inflate:
                tf.P[:3, :3] += np.eye(3) * 0.1 ** 2
                tf.P[3:6, 3:6] += np.eye(3) * 0.005 ** 2
            tf.pos_update(p_meas - xk[3:6] * (t_img - tk), sp)
            later = tf.hist[k + 1:]
            new_hist = tf.hist[:k] + [(tk, tf.x.copy(), tf.P.copy(), tf.hist[k][3])]
            sh = tf.shadows
            tf.shadows = []
            for ti, _, _, inp in later:
                R, Feff, z, n, alpha, w, ic = inp
                tf.predict(self.dt, R, Feff, z, n, alpha, w, ic)
                new_hist.append((ti, tf.x.copy(), tf.P.copy(), inp))
            tf.shadows = sh
            tf.hist = new_hist
            self.stats['rollbacks'] += 1
            self.heavy = len(later) > 120
        else:
            tk, xk, Pk, _ = tf.hist[k]
            x, P = (xk.copy(), Pk.copy())
            if inflate:
                P[:3, :3] += np.eye(3) * 0.1 ** 2
                P[3:6, 3:6] += np.eye(3) * 0.005 ** 2
            x, P = tf.pos_update_xp(x, P, p_meas - xk[3:6] * (t_img - tk), sp)
            pend = []
            if then is not None:
                pend.append((then[0], then[1], then[3]))
            self.replay = {'x': x, 'P': P, 't_cur': tk, 'new_hist': tf.hist[:k] + [(tk, x.copy(), P.copy(), tf.hist[k][3])], 'pending': pend, 'n_steps': 0}
            self.stats['rollbacks'] += 1
        self.last_fix_img = max(self.last_fix_img, t_img, then[0] if then is not None else -1000000000.0)
        if self.want_shadows and self.in_block and (self.shadow_mode is not None) and (self.t_now - self.last_shadow_t >= 2.0):
            self._finish_replay()
            self.last_shadow_t = self.t_now
            tol = self.tol[self.shadow_mode]['position_m']
            sd = Shadow(tf, self.t_now, self.shadow_mode, 0.6 * tol, [0.07, 0.07, 0.27], 1.3 ** 2)
            sd.steady = steady
            tf.shadows.append(sd)
            self.shadow_results.append(sd)

    def _advance_replay(self, first=False, all_of_it=False):
        rp = self.replay
        if rp is None:
            return
        tf = self.tf
        hist = tf.hist
        i0 = len(hist)
        while i0 > 0 and hist[i0 - 1][0] > rp['t_cur'] + 1e-12:
            i0 -= 1
        n_done = 0
        x, P = (rp['x'], rp['P'])
        i = i0
        if rp['pending'] and rp['pending'][0][0] >= rp['t_cur'] - 1e-12 and (i0 >= len(hist) or hist[i0][0] > rp['pending'][0][0]):
            tp, pm_, sp_ = rp['pending'].pop(0)
            x, P = tf.pos_update_xp(x, P, pm_ - x[3:6] * (tp - rp['t_cur']), sp_)
            rp['new_hist'][-1] = (rp['new_hist'][-1][0], x.copy(), P.copy(), rp['new_hist'][-1][3])
        while i < len(hist) and (all_of_it or n_done < RT_REPLAY_PER_STEP):
            ti, _, _, inp = hist[i]
            R, Feff, z, n, alpha, w, ic = inp
            x, P = tf.predict_xp(x, P, self.dt, R, Feff, z, n, alpha, w, ic)
            rp['new_hist'].append((ti, x.copy(), P.copy(), inp))
            rp['t_cur'] = ti
            if rp['pending']:
                nxt = hist[i + 1][0] if i + 1 < len(hist) else None
                tp, pm_, sp_ = rp['pending'][0]
                if ti <= tp and (nxt is None or nxt > tp):
                    x, P = tf.pos_update_xp(x, P, pm_ - x[3:6] * (tp - ti), sp_)
                    rp['new_hist'][-1] = (ti, x.copy(), P.copy(), inp)
                    rp['pending'].pop(0)
            i += 1
            n_done += 1
        rp['x'], rp['P'] = (x, P)
        rp['n_steps'] += 1
        self.stats['replay_steps'] += 1
        if i >= len(hist):
            if rp['pending']:
                for tp, pm_, sp_ in rp['pending']:
                    x, P = tf.pos_update_xp(x, P, pm_, sp_)
            tf.x, tf.P = (x, P)
            nh = rp['new_hist']
            if len(nh) > 260:
                nh = nh[-220:]
            tf.hist = nh
            self.replay = None

    def _finish_replay(self):
        if self.replay is not None:
            self._advance_replay(all_of_it=True)

    def _ingest(self, fr):
        on = self.on
        t = fr['t']
        if not fr['in_control']:
            self.saw_not_in_control = True
        lvc = int(fr['fan_level_commanded'])
        if lvc != self.level:
            self.mode_change_t = t
            self.mode_from = [m for m in MODES if self.fan[m] == self.level][0]
            self.level = lvc
        pr = fr['propulsion']
        sc = np.asarray(pr['nozzle_command'], float)
        spd = pr['impeller_speed']
        pred = self.nominal.from_pwm(sc, spd, lvc, use_gains=on['S4']) if lvc > 0 else (np.zeros(3), np.zeros(3))
        if t > self.t_last_est + 1e-09:
            self.cmd_hist.append((t - self.dt, None, None, lvc, pred))
        self._trim()
        self._estimate_step(fr)

    def _notices(self, fr):
        for nt in fr.get('notices') or []:
            if not isinstance(nt, dict) or nt.get('kind') != 'event_ignored':
                continue
            pe = self.pending_event
            if pe is None or nt.get('job_id') != pe['job_id'] or nt.get('reason') not in ('step_not_applied', 'before_last_setpoint'):
                continue
            self.job, self.plan_sps, self.job_end_plan, self.goal_now, self.ev = (pe['job'], pe['plan_sps'], pe['job_end_plan'], pe['goal_now'], pe['ev'])
            self.prev_goal = pe['prev_goal']
            self.pending_event = None
            self.stats['resends'] += 1
            self.event_records.append({'t': fr['t'], 'job_id': nt.get('job_id'), 'kind': 'reopened', 'reason': nt.get('reason')})

    def step(self, fr):
        on = self.on
        t = fr['t']
        self.t_now = t
        self.heavy = False
        self.in_block = fr['in_control'] and fr['job_id'] not in (None, 'checkout')
        self.shadow_mode = fr['mode'] if self.in_block else None
        blk = fr.get('block') or {}
        self.time_left = blk.get('time_left_s')
        if not self.in_block and self.tf.shadows:
            for sh in self.tf.shadows:
                sh.done = True
            self.tf.shadows = []
        if on['EV']:
            self._notices(fr)
        self._ingest(fr)
        lvc = int(fr['fan_level_commanded'])
        if not fr['in_control']:
            self.int_p[:] = 0
            self.int_a[:] = 0
            return {'force': [0.0, 0.0, 0.0], 'torque': [0.0, 0.0, 0.0], 'event': 'none'}
        if not on['J'] and self.heavy:
            return {'force': None, 'torque': None, 'event': 'none'}
        co = fr['job_id'] == 'checkout'
        if co and on['CO']:
            if self.co_sps is None:
                self.co_t0 = t
                self.co_sps = self._checkout_plan(t)
            p_r, v_r, q_r, w_r = traj.ref_at(self.co_sps, t)
            s = self._sp_at_list(self.co_sps, t)
            a_r, al_r = (s[3], s[6])
            moving = self.co_sps[1][0] <= t < self.co_sps[-1][0]
            kind = 'checkout' if moving else 'hold'
        elif co or self.plan_sps is None:
            p_r, q_r = self.prev_goal
            v_r = np.zeros(3)
            w_r = np.zeros(3)
            a_r = np.zeros(3)
            al_r = np.zeros(3)
            kind = 'hold'
        else:
            p_r, v_r, q_r, w_r = traj.ref_at(self.plan_sps, t)
            s = self._sp_at(t)
            a_r, al_r = (s[3], s[6])
            kind = self.job['kind'] if self.job is not None else 'post_event'
        q_now = self.att.q
        tau_a = max(0.0, t - (self.att.t if self.att.t is not None else t))
        w = self.att.w
        Feff, Teff = self._feff(t)
        alpha = np.linalg.solve(self.J, Teff)
        q_now = rot.qnormalize(rot.qmul(q_now, rot.qexp(w * tau_a + 0.5 * alpha * tau_a * tau_a)))
        w_now = w + alpha * tau_a
        R = rot.qmat(q_now)
        rc = R @ self.com
        p = self.tf.x[:3] - rc
        v = self.tf.x[3:6] - _cross(R @ w_now, rc)
        a_now = R @ ((Feff + (self.tf.x[9:12] if on['S5'] else 0)) / self.m)
        if on['C']:
            tc = self.lag
            p = p + v * tc + 0.5 * a_now * tc * tc
            v = v + a_now * tc
            q_now = rot.qnormalize(rot.qmul(q_now, rot.qexp(w_now * tc + 0.5 * alpha * tc * tc)))
            w_now = w_now + alpha * tc
            kp, kd, kpa, kda = (0.6, 1.2, 4.0, 3.2)
        else:
            kp, kd, kpa, kda = (0.6 * 2.25, 1.2 * 1.5, 4.0 * 2.25, 3.2 * 1.5)
        if kind in ('hold', 'post_event'):
            kp, kd = (kp * HOLD_KP, kd * HOLD_KD)
        outage = t - self.last_fix_img > 6.0
        if on['S8'] and outage:
            kp *= 0.6
        R = rot.qmat(q_now)
        Om = self.Om if on['S9'] else np.zeros(3)
        e_p = p_r - p
        ne = float(np.linalg.norm(e_p))
        if ne > 0.05:
            e_p = e_p * (0.05 / ne)
        sat = self._sat_prev
        if on['S5'] and (not (on['S8'] and outage)) and (not (on['D'] and sat)):
            self.int_p += e_p * self.dt
            self.int_p = np.clip(self.int_p, -0.05, 0.05)
        a_des = a_r + kp * e_p + kd * (v_r - v) + (0.02 * self.int_p if on['S5'] else 0.0)
        a_des = a_des + 2.0 * _cross(Om, v) + _cross(Om, _cross(Om, p))
        Fb = self.mass_ctl * (R.T @ a_des)
        if on['S5']:
            Fb = Fb - self.tf.x[9:12]
        e_q = rot.qlog(rot.qmul(q_r, rot.qconj(q_now)))
        e_b = R.T @ e_q
        if on['S5'] and (not (on['D'] and sat)):
            self.int_a += e_b * self.dt
            self.int_a = np.clip(self.int_a, -0.2, 0.2)
        al_des = al_r + kpa * e_b + kda * (w_r - w_now) + (0.3 * self.int_a if on['S5'] else 0.0)
        Tb = self.J @ al_des + _cross(w_now, self.J @ w_now)
        if kind == 'hold' and self.hold_zero and (fr['job_id'] not in (None, 'checkout')):
            Fb = np.zeros(3)
            Tb = np.zeros(3)
        lvl = max(1, lvc)
        gf, gt = self.resp[lvl]
        if on['L']:
            g_learn = 1.0 + self.tf.x[12:15] if self.tf.use_g and on['S4'] else np.ones(3)
            Fc = Fb / (np.where(Fb >= 0, gf[0::2], gf[1::2]) * np.clip(g_learn, 0.5, 2.0))
            Tc = Tb / np.where(Tb >= 0, gt[0::2], gt[1::2])
        else:
            Fc, Tc = (Fb.copy(), Tb.copy())
        W = np.concatenate([Fc, Tc])
        self._sat_prev = False
        if on['D']:
            cap = self.cap[lvl] * 0.85
            s_ = 1.0
            for i in range(6):
                c = cap[i, 0] if W[i] >= 0 else cap[i, 1]
                if abs(W[i]) > c:
                    s_ = min(s_, c / abs(W[i]))
            if s_ < 1.0:
                W = W * s_
                self._sat_prev = True
        self.last_cmd = W.copy()
        if co and on['CO']:
            self.co_cmd.append((t, W[3:].copy()))
        ev = self._event(t, kind, p, v, q_now, w_now) if not on['EV'] else self._event_ev(t, kind, p, v, q_now, w_now)
        return {'force': W[:3].tolist(), 'torque': W[3:].tolist(), 'event': ev}
    _sat_prev = False

    def _sp_at(self, t):
        return self._sp_at_list(self.plan_sps, t)

    @staticmethod
    def _sp_at_list(sps, t):
        k = 0
        for i in range(len(sps)):
            if sps[i][0] <= t:
                k = i
            else:
                break
        if t >= sps[-1][0]:
            return sps[-1]
        return sps[k]

    def _trim(self):
        if len(self.cmd_hist) > 40:
            del self.cmd_hist[:10]
        return None

    def _event(self, t, kind, p, v, q, w):
        on = self.on
        if kind in ('hold', 'post_event', 'checkout') or self.job is None or t < self.job_end_plan - 1e-09:
            return 'none'
        gp, gq = self.goal_now
        mode = self.job['mode']
        tol = self.tol[mode if on['B'] else 'nominal']
        f = tol['position_m'] / 0.25
        branch = None
        if kind == 'view':
            if not on['F']:
                if self.t_ready_wait is None:
                    self.t_ready_wait = t
                ev = 'ready' if t - self.t_ready_wait >= 2.0 else 'none'
                branch = 'F_off_2s'
            else:
                ang = rot.qangle(q, gq)
                ok = ang < math.radians(0.5) and np.linalg.norm(w) < 0.003
                ev = 'ready' if ok or t >= self.job_end_plan + 0.5 else 'none'
                branch = 'ok' if ok else 'late_0.5s'
        else:
            ok = np.linalg.norm(p - gp) < 0.008 * f and np.linalg.norm(v) < 0.003 * f and (rot.qangle(q, gq) < math.radians(0.5 * f + 0.2)) and (np.linalg.norm(w) < 0.003)
            fresh = self.last_fix_img >= self.job_end_plan - 0.3
            ev = 'done' if ok and fresh or (fresh and t >= self.job_end_plan + 0.6) or t >= self.job_end_plan + 4.0 else 'none'
            branch = 'ok_fresh' if ok and fresh else 'fresh_0.6s' if fresh and t >= self.job_end_plan + 0.6 else 'timeout_4s'
        if ev != 'none':
            self._record_event(t, ev, branch, p, q, gp, gq, None)
            self.prev_goal = (gp, gq)
            self.job = None
            self.plan_sps = [(t, gp, np.zeros(3), np.zeros(3), gq, np.zeros(3), np.zeros(3))]
        return ev

    def _sigma3_pos(self, view=False):
        P = self.tf.P[:3, :3].copy()
        if view:
            Rm = rot.qmat(self.att.q) @ self.mount
            K = rot.skew(Rm)
            P = P + K @ self.att.P[:3, :3] @ K.T
        return 3.0 * math.sqrt(max(0.0, float(max(np.linalg.eigvalsh(P)))))

    def _event_ev(self, t, kind, p, v, q, w):
        on = self.on
        if kind in ('hold', 'post_event', 'checkout') or self.job is None:
            return 'none'
        gp, gq = self.goal_now
        mode = self.job['mode']
        tol = self.tol[mode if on['B'] else 'nominal']
        f = tol['position_m'] / 0.25
        f = EV_F
        if self.ev is None:
            self.ev = {'t_last_far': -1000000000.0}
        st = self.ev
        if kind == 'view':
            R = rot.qmat(q)
            c_est = p + R @ self.mount
            c_goal = gp + rot.qmat(gq) @ self.mount
            dc = float(np.linalg.norm(c_est - c_goal))
            ang = rot.qangle(q, gq)
            ok = ang < math.radians(0.5) and np.linalg.norm(w) < 0.003 and (dc < EV_VIEW_CENTRE)
            if dc >= 2.0 * EV_VIEW_CENTRE or ang >= math.radians(1.0):
                st['t_last_far'] = t
        else:
            dp = float(np.linalg.norm(p - gp))
            sv = float(np.linalg.norm(v))
            ok = dp < 0.008 * f and sv < 0.003 * f and (rot.qangle(q, gq) < math.radians(0.5 * f + 0.2)) and (np.linalg.norm(w) < 0.003)
            if dp >= 2 * 0.008 * f or sv >= 2 * 0.003 * f:
                st['t_last_far'] = t
        if t < self.job_end_plan - 1e-09 or self.replay is not None:
            return 'none'
        waited = t - self.job_end_plan
        trust_a = self.last_fix_img >= max(self.job_end_plan - 0.3, st['t_last_far'])
        s3 = self._sigma3_pos(view=kind == 'view')
        trust_b = s3 <= (EV_U_VIEW if kind == 'view' else EV_U_POS) and self.candidate is None and (not self.prev_rejected) and (t - self.last_fix_img <= EV_FIX_AGE)
        tl = self.time_left
        branch = None
        own = self.capture if kind == 'view' else float(pub('score.end_window_s')[1])
        if on['SB']:
            return self._event_sb(t, kind, p, v, q, w, st, ok, trust_a, trust_b, s3, waited, tl, own)
        slack = tl - getattr(self, 'need_after', 0.0) - own if tl is not None else None
        if kind == 'view':
            reserve = self.capture + 0.5
            if ok and trust_a:
                branch = 'A'
            elif ok and trust_b:
                branch = 'B'
            elif ok and waited >= EV_VIEW_WAIT_OK and (self.candidate is None) and (not self.prev_rejected):
                branch = 'fallback_ok'
            elif ok and slack is not None and (slack <= EV_SLACK_OK):
                branch = 'slack_ok'
            elif waited >= EV_VIEW_WAIT_ANY:
                branch = 'fallback_time'
            elif tl is not None and tl <= reserve or (slack is not None and slack <= EV_SLACK_ANY):
                branch = 'budget_end'
            ev = 'ready' if branch else 'none'
        else:
            if ok and trust_a:
                branch = 'A'
            elif ok and trust_b:
                branch = 'B'
            elif ok and slack is not None and (slack <= EV_SLACK_OK):
                branch = 'slack_ok'
            elif tl is not None and tl <= EV_RESERVE or (slack is not None and slack <= EV_SLACK_ANY):
                branch = 'budget_end'
            ev = 'done' if branch else 'none'
        if ev != 'none':
            self._record_event(t, ev, branch, p, q, gp, gq, s3, ok=ok, trust_a=trust_a, trust_b=trust_b, waited=waited, slack_s=slack)
            self.pending_event = {'job_id': self.job['id'], 'job': self.job, 'plan_sps': self.plan_sps, 'job_end_plan': self.job_end_plan, 'goal_now': self.goal_now, 'ev': self.ev, 'prev_goal': self.prev_goal}
            self.prev_goal = (gp, gq)
            self.job = None
            self.plan_sps = [(t, gp, np.zeros(3), np.zeros(3), gq, np.zeros(3), np.zeros(3))]
        return ev

    def _event_sb(self, t, kind, p, v, q, w, st, ok, trust_a, trust_b, s3, waited, tl, own):
        on = self.on
        gp, gq = self.goal_now
        avail = tl - own - getattr(self, 'need_after', 0.0) - SB_MARGIN if tl is not None else None
        if 'reserve' not in st:
            n = len(self.later_events)
            share = max(0.0, avail) / (n + 1) if avail is not None else 0.0
            st['avail0'] = avail
            st['share'] = share
            st['reserve'] = float(sum((min(self._fix_wait_at_cap(m), share) for m in self.later_events)))
        slack = avail - st['reserve'] if avail is not None else None
        out_of_time = tl is not None and tl <= (self.capture + 0.5 if kind == 'view' else EV_RESERVE) or (avail is not None and avail <= 0)
        branch = None
        if kind == 'view':
            R = rot.qmat(q)
            rm = R @ self.mount
            vc = float(np.linalg.norm(v + _cross(R @ w, rm)))
            dc = float(np.linalg.norm(p + rm - (gp + rot.qmat(gq) @ self.mount)))
            rest_fix = self.last_fix_img >= max(self.job_end_plan, st['t_last_far'])
            ok_rest = ok and dc < VW_CENTRE and (vc < VW_SPEED)
            may_wait = slack is not None and slack > 0 and (not out_of_time)
            risky = s3 > VW_LIMIT
            if on['VW'] and ok_rest and rest_fix:
                branch = 'A_rest'
            elif ok and trust_b and (not (on['VW'] and on['VL'] and may_wait)):
                branch = 'B'
            elif on['VW'] and on['VL'] and may_wait:
                branch = None
            elif on['VW'] and (not on['VL']) and (not risky) and ok:
                branch = 'within'
            elif on['VW'] and (not on['VL']) and risky and may_wait:
                branch = None
            elif ok and trust_a:
                branch = 'A'
            elif ok and waited >= EV_VIEW_WAIT_OK and (self.candidate is None) and (not self.prev_rejected):
                branch = 'fallback_ok'
            elif ok and slack is not None and (slack <= 0):
                branch = 'slack_ok'
            elif waited >= EV_VIEW_WAIT_ANY:
                branch = 'fallback_time'
            elif out_of_time:
                branch = 'budget_end'
            ev = 'ready' if branch else 'none'
        else:
            if ok and trust_a:
                branch = 'A'
            elif ok and trust_b:
                branch = 'B'
            elif ok and slack is not None and (slack <= 0):
                branch = 'slack_ok'
            elif out_of_time:
                branch = 'budget_end'
            ev = 'done' if branch else 'none'
        if ev != 'none':
            self._record_event(t, ev, branch, p, q, gp, gq, s3, ok=ok, trust_a=trust_a, trust_b=trust_b, waited=waited, slack_s=slack, avail_s=avail, reserve_s=st['reserve'], share_s=st['share'], avail0_s=st['avail0'], need_after_s=getattr(self, 'need_after', None), later_events=len(self.later_events))
            self.pending_event = {'job_id': self.job['id'], 'job': self.job, 'plan_sps': self.plan_sps, 'job_end_plan': self.job_end_plan, 'goal_now': self.goal_now, 'ev': self.ev, 'prev_goal': self.prev_goal}
            self.prev_goal = (gp, gq)
            self.job = None
            self.plan_sps = [(t, gp, np.zeros(3), np.zeros(3), gq, np.zeros(3), np.zeros(3))]
        return ev

    def _record_event(self, t, ev, branch, p, q, gp, gq, s3, **kw):
        t0 = self.job_end_plan
        fixes = [r for r in self.fix_records if r[2] >= t0 - 3.5 or r[0] >= t0]
        rec = {'t': t, 'job_id': self.job['id'], 'job_kind': self.job['kind'], 'event': ev, 'branch': branch, 't_plan_end': t0, 'waited_s': t - t0, 'time_left_s': self.time_left, 'P_pos': self.tf.P[:3, :3].tolist(), 'P_att': self.att.P[:3, :3].tolist(), 'sigma3_pos': s3 if s3 is not None else self._sigma3_pos(view=self.job['kind'] == 'view'), 'est_to_goal_m': float(np.linalg.norm(p - gp)), 'est_angle_to_goal_rad': float(rot.qangle(q, gq)), 'last_fix_age_s': t - self.last_fix_img, 'candidate_pending': self.candidate is not None, 'prev_rejected': bool(self.prev_rejected), 'last_innovations': list(self.last_innov), 'fixes_near_the_end': [{'t': r[0], 'camera_id': r[1], 't_img': r[2], 'landmarks': r[3], 'd2p': r[4], 'd2a': r[5], 'decision': r[6]} for r in fixes[-12:]], 'replay_pending': self.replay is not None, 'lag_s': self.lag}
        rec.update(kw)
        self.event_records.append(rec)
