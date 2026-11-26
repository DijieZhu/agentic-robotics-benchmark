"""Built-in control law of the hidden world: a port of the maker's firmware control law. The constants are read from
data/stock_law_constants.json; mass and inertia are given by the caller. Quaternions (w, x, y, z), body to cabin.
Flown at the start hold, between blocks and in a hand-back's recovery.



"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_K = json.load(open(os.path.join(HERE, "data", "stock_law_constants.json")))
CTL = _K["control"]
GAINS = {k: v for k, v in _K["gains"].items() if k != "vel_gain"}
MASS = None
INERTIA = None


def qmul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def qconj(q):
    return np.concatenate([q[..., :1], -q[..., 1:]], -1)


def qnorm_fw(q):
    q = np.where(q[..., :1] < 0, -q, q)
    m = np.sqrt(np.sum(q * q, -1, keepdims=True))
    return np.where(m > 1e-7, q / np.maximum(m, 1e-300), q)


def qerr(cmd, act):
    out = qnorm_fw(qmul(qconj(act), cmd))
    return np.abs(np.arccos(np.clip(out[..., 0], -1.0, 1.0))) * 2.0


def qmat(q):
    m = np.sqrt(np.sum(q * q, -1, keepdims=True))
    q = q / m
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - w * z); R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y); R[..., 2, 1] = 2 * (y * z + w * x); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def omega_matrix(v):
    o = np.zeros(v.shape[:-1] + (4, 4))
    o[..., 1, 0], o[..., 2, 0], o[..., 3, 0] = -v[..., 2], v[..., 1], -v[..., 0]
    o[..., 0, 1], o[..., 2, 1], o[..., 3, 1] = v[..., 2], -v[..., 0], -v[..., 1]
    o[..., 0, 2], o[..., 1, 2], o[..., 3, 2] = -v[..., 1], v[..., 0], -v[..., 2]
    o[..., 0, 3], o[..., 1, 3], o[..., 2, 3] = v[..., 0], v[..., 1], v[..., 2]
    return o


class BatchController:
    def __init__(self, B, mass, inertia, vel_gain=None, gains=GAINS):
        self.B = B
        self.vel_gain = np.ones(B) if vel_gain is None else np.asarray(vel_gain, float)
        self.mass = float(mass)
        self.J = np.asarray(inertia, float)
        self.Jinv = np.linalg.inv(self.J)
        g = {k: np.full(3, v) for k, v in gains.items()}
        self.vel_kd = g["vel_kd"]
        self.ki_lin = g["pos_ki"] / g["vel_kd"]; self.kp_lin = g["pos_kp"] / g["vel_kd"]
        self.ki_rot = g["att_ki"] / g["omega_kd"]; self.kp_rot = g["att_kp"] / g["omega_kd"]
        self.kd_rot = g["omega_kd"] * np.diag(self.J)
        self.rate = float(CTL["rate_hz"])
        self.g1, self.g2 = CTL["butterworth_gain_1"], CTL["butterworth_gain_2"]
        self.accel_gain = np.asarray(CTL["accel_gain"], float)
        self.alpha_gain = np.asarray(CTL["alpha_gain"], float)
        self.reset()

    def reset(self):
        B = self.B
        self.prev_mode = np.zeros((B, 5), int)
        self.fv = np.zeros((B, 3)); self.fw = np.zeros((B, 3))
        self.prev_pos = np.zeros((B, 3)); self.prev_att = np.zeros((B, 4))
        self.lin_int = np.zeros((B, 3)); self.rot_int = np.zeros((B, 3))

    def _filt(self, x, prev):
        tmp = x * self.g1 - prev * self.g2
        out = tmp + prev
        prev[:] = tmp
        return np.sum(out * out, 1)

    def step(self, dt, est_p, est_q, est_v, est_w, sp, mode):
        """dt (B,), est_* (B,..), sp = dict(pos, vel, acc, quat, omega, alpha) each (B,..) or (..), mode (B,) int.
        Returns force (B,3), torque (B,3), status (B,), traj_error_pos (B,), traj_error_att (B,)."""
        B = self.B
        dt = np.broadcast_to(np.asarray(dt, float), (B,))[:, None]
        P = {k: np.broadcast_to(np.asarray(v, float), (B,) + np.shape(v)[-1:]) for k, v in sp.items()}
        traj_pos = P["pos"] + P["vel"] * dt + 0.5 * P["acc"] * dt * dt
        traj_vel = P["vel"] + P["acc"] * dt
        traj_acc = P["acc"]
        traj_alpha = P["alpha"]
        traj_omega = P["omega"] + P["alpha"] * dt
        if np.all(dt == 0):
            traj_quat = qnorm_fw(P["quat"])
        else:
            traj_quat = np.empty((B, 4))
            for b in range(B):
                om, al, d, q = P["omega"][b], P["alpha"][b], float(dt[b, 0]), P["quat"][b]
                if not (np.any(om) or np.any(al)):
                    traj_quat[b] = qnorm_fw(q)
                    continue
                Ow, Oa = omega_matrix(om), omega_matrix(al)
                v = 0.5 * d * (0.5 * d * al + om)
                nv = float(np.linalg.norm(v))
                E = math.cos(nv) * np.eye(4) + (math.sin(nv) / nv if nv > 1e-12 else 1.0) * omega_matrix(v)
                E += (1.0 / 48.0) * d ** 3 * (Oa @ Ow - Ow @ Oa)
                c = E @ np.array([q[1], q[2], q[3], q[0]])
                traj_quat[b] = qnorm_fw(np.array([c[3], c[0], c[1], c[2]]))
        err_pos = np.linalg.norm(traj_pos - est_p, axis=1)
        err_att = qerr(traj_quat, est_q)
        mode = np.asarray(mode, int)
        self.prev_mode[:, 1:] = self.prev_mode[:, :-1].copy()
        self.prev_mode[:, 0] = mode
        vb = self._filt(est_v, self.fv) < CTL["stopping_vel_thresh_sq"]
        wb = self._filt(est_w, self.fw) < CTL["stopping_omega_thresh_sq"]
        pm = self.prev_mode
        stopped = vb & wb & (pm[:, 4] == 1) & (pm[:, 4] == pm[:, 3]) & (pm[:, 3] == pm[:, 2]) & (pm[:, 2] == pm[:, 1])
        has_att = np.any(self.prev_att != 0, 1)
        q_err = np.where(has_att, qerr(est_q, np.where(has_att[:, None], self.prev_att, np.array([1.0, 0, 0, 0]))), math.pi)
        pos_err2 = np.sum((self.prev_pos - est_p) ** 2, 1)
        cond = ((pos_err2 > CTL["stopped_pos_thresh"]) | (np.abs(q_err) > CTL["stopped_quat_thresh_rad"])) & (mode == 3)
        status = np.where(cond, 1, np.where(stopped, 3, mode))
        R = qmat(est_q)
        st = stopped[:, None]
        target = np.where(st, self.prev_pos, traj_pos)
        pos_err = target - est_p
        lo, hi = CTL["pos_int_limits"]
        on = (status > 1)[:, None]
        self.lin_int = np.where(on, self.lin_int + self.ki_lin * pos_err / self.rate, 0.0)
        pos_int = np.where(on, np.clip(self.lin_int, lo, hi), 0.0)
        tv = np.where(st, 0.0, traj_vel)
        ta = np.where(st, 0.0, traj_acc)
        vv = np.where(on, self.vel_gain[:, None] * (tv - est_v) + self.kp_lin * pos_err + pos_int, -est_v)
        a = np.einsum("bji,bj->bi", R, vv) * self.vel_kd
        bb = np.einsum("bji,bj->bi", R, ta * self.accel_gain)
        f = self.mass * (a + bb)
        mag = np.linalg.norm(f, axis=1, keepdims=True)
        lim = CTL["linear_force_limit_n"]
        f = np.where(mag < lim, f, lim / np.maximum(mag, 1e-300) * f)
        idle = (status == 0)[:, None]
        force = np.where(idle, 0.0, f)
        q_cmd = np.where(st, self.prev_att, traj_quat)
        q_out = qnorm_fw(qmul(qconj(est_q), q_cmd))
        att_err = q_out[:, 1:]
        lo, hi = CTL["att_int_limits"]
        self.rot_int = np.where(on, self.rot_int + att_err * self.ki_rot / self.rate, 0.0)
        att_int = np.where(on, np.clip(self.rot_int, lo, hi), 0.0)
        cmd_om = np.where(st, 0.0, traj_omega)
        cmd_al = np.where(st, 0.0, traj_alpha)
        rate_err = np.where(on, cmd_om + att_int + self.kp_rot * att_err - est_w, -est_w) * self.kd_rot
        Jw = est_w @ self.J.T
        torque = (self.alpha_gain * cmd_al) @ self.J.T + rate_err - np.cross(Jw, est_w)
        torque = np.where(idle, 0.0, torque)
        self.prev_pos = np.where(st, self.prev_pos, est_p)
        self.prev_att = np.where(st, self.prev_att, est_q)
        return force, torque, status, err_pos, err_att
