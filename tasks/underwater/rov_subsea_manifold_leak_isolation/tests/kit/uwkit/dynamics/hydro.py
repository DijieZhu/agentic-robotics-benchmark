"""Fossen 6-DoF hydrodynamic layer on top of MuJoCo rigid-body dynamics (multi-body).

Responsibility split (MuJoCo: rigid bodies, joints, contacts, gravity; this layer: the rest):
  restoring   buoyancy  f_sub * rho(depth) * V * g  at each body's centre of buoyancy (CoB)
  damping     -(D_l + D_q |nu_r|) nu_r  about each body's CoM, body axes, nu_r relative to the water
  added mass  "merge + residual predictor":
              * isotropic translational part m_iso and the diagonal rotational part I_A are merged
                into the MuJoCo body inertia at build time (exact for a constant current, see
                UWKIT_DYNAMICS.md section 3.3), the extra weight m_iso*g is cancelled by gravcomp;
              * the anisotropic translational residual dM = A_lin - m_iso of a "full" body is applied
                exactly: after all other forces are written, one extra acceleration pass gives qacc0
                and we solve (I + dM Lambda^-1) f = -dM (s0 + c) for the residual force f
                (Lambda^-1 = J M^-1 J^T at the CoM, so child bodies such as gripper jaws are included);
              * Coriolis / Munk terms of the residual: (dM v_r) x w and (dM v_r) x v_r (exact);
  unsteady current: (rho V + m_iso) * Dv_c/Dt  (Froude-Krylov + merged added mass), and the residual
              through c = w x v_c,b - R^T Dv_c/Dt; Dv_c/Dt is the finite difference of the current seen
              by the CoM between steps (zero for a constant uniform current).
Body kinds:
  "full"   : the vehicle hull (anisotropic added mass, full diagonal damping matrices)
  "simple" : appendages and free objects (buoyancy + isotropic added mass merged + drag)
Everything enters MuJoCo through data.xfrc_applied (world-frame force / torque at each body CoM),
between mj_step1 and mj_step2 (after mj_step the cached kinematics are those of the previous step).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import mujoco

from ..params import G
from ..frames import cross3, crossk

UWKIT_TEXT = "uwkit"          # name of the <custom><text> element that carries the uwkit metadata


@dataclass
class HydroBodyParams:
    kind: str                       # "full" | "simple"
    volume: float                   # displaced volume at full submergence [m^3]
    cob: np.ndarray                 # CoB in the body frame [m]
    added_mass: np.ndarray          # 6 diagonal magnitudes (simple: m_a x3, I_a x3)
    merged: np.ndarray              # [m_iso, Ixx_A, Iyy_A, Izz_A] that is inside the MuJoCo inertia
    damping_linear: np.ndarray      # 6
    damping_quadratic: np.ndarray   # 6
    rho_ref: float = 1000.0
    half_height: float = 0.05       # vertical half-extent for the partial-submergence ramp [m]
    rigid_mass: float = 0.0
    rigid_com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rigid_inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    def __post_init__(self):
        self.cob = np.asarray(self.cob, float).reshape(3)
        self.added_mass = np.asarray(self.added_mass, float).reshape(6)
        self.merged = np.asarray(self.merged, float).reshape(4)
        self.damping_linear = np.asarray(self.damping_linear, float).reshape(6)
        self.damping_quadratic = np.asarray(self.damping_quadratic, float).reshape(6)
        self.rigid_com = np.asarray(self.rigid_com, float).reshape(3)
        self.rigid_inertia = np.asarray(self.rigid_inertia, float).reshape(3, 3)

    def to_dict(self):
        return dict(kind=self.kind, volume=self.volume, cob=self.cob.tolist(),
                    added_mass=self.added_mass.tolist(), merged=self.merged.tolist(),
                    damping_linear=self.damping_linear.tolist(),
                    damping_quadratic=self.damping_quadratic.tolist(), rho_ref=self.rho_ref,
                    half_height=self.half_height, rigid_mass=self.rigid_mass,
                    rigid_com=self.rigid_com.tolist(), rigid_inertia=self.rigid_inertia.tolist())

    @staticmethod
    def from_dict(d):
        return HydroBodyParams(**d)


def _solve3(A, b):
    """3x3 linear solve by Cramer's rule (np.linalg.solve has ~15 us of overhead)."""
    (a, b_, c), (d, e, f), (g, h, i) = A.tolist()
    x, y, z = b.tolist()
    co0, co1, co2 = e * i - f * h, f * g - d * i, d * h - e * g
    det = a * co0 + b_ * co1 + c * co2
    return np.array([(x * co0 + b_ * (f * z - y * i) + c * (y * h - e * z)),
                     (a * (y * i - f * z) + x * co1 + c * (d * z - y * g)),
                     (a * (e * z - y * h) + b_ * (y * g - d * z) + x * co2)]) / det


def iso_split(added_mass):
    """Isotropic translational part (mid-range, minimises max |residual|) and rotational diagonal."""
    a = np.asarray(added_mass, float)
    return 0.5 * (a[:3].min() + a[:3].max()), a[3:].copy()


def read_uwkit_meta(model: mujoco.MjModel) -> dict:
    """The uwkit metadata JSON stored in the model's <custom><text name="uwkit">."""
    for i in range(model.ntext):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXT, i) == UWKIT_TEXT:
            a, n = int(model.text_adr[i]), int(model.text_size[i])
            raw = bytes(model.text_data[a:a + n]).split(b"\x00", 1)[0]
            return json.loads(raw.decode())
    raise KeyError("model carries no uwkit metadata (build it with uwkit.models.build_vehicle)")


class HydroLayer:
    def __init__(self, model, data, water, current, scheme="pc", unsteady_current=True,
                 bodies: dict | None = None, alpha_ema=0.3):
        if model.opt.density != 0 or model.opt.viscosity != 0:
            raise ValueError("MuJoCo's built-in fluid model must be off (opt.density = opt.viscosity = 0)")
        if scheme not in ("pc", "ema", "none"):
            raise ValueError("scheme must be 'pc' (merge + predictor), 'ema' (merge + EMA residual) or 'none'")
        self.m, self.d, self.water, self.current = model, data, water, current
        self.scheme, self.unsteady = scheme, bool(unsteady_current)
        self.alpha = float(alpha_ema)
        if bodies is None:
            bodies = {k: HydroBodyParams.from_dict(v) for k, v in read_uwkit_meta(model)["hydro_bodies"].items()}
        self.names = list(bodies)
        self.params = [bodies[k] for k in self.names]
        self.ids = np.array([model.body(k).id for k in self.names], int)
        P = self.params
        self.k = len(P)
        self.vol = np.array([p.volume for p in P])
        self.cob = np.array([p.cob for p in P])
        self.Dl = np.array([p.damping_linear for p in P])
        self.Dq = np.array([p.damping_quadratic for p in P])
        self.rho_ref = np.array([p.rho_ref for p in P])
        self.hh = np.array([max(p.half_height, 1e-4) for p in P])
        self.m_iso = np.array([p.merged[0] for p in P])
        self.root_ids = model.body_rootid[self.ids]
        self.full = [i for i, p in enumerate(P) if p.kind == "full"]
        self.dM = {i: P[i].added_mass[:3] - P[i].merged[0] for i in self.full}
        for i in self.full:
            b = self.ids[i]
            jnt = model.body_jntadr[b]
            if jnt < 0 or model.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE or model.body_parentid[b] != 0:
                raise ValueError(f"full hydro body {self.names[i]} must be a world child with a free joint")
        self._dof = {i: int(model.jnt_dofadr[model.body_jntadr[self.ids[i]]]) for i in self.full}
        self._J = np.zeros((3, model.nv))
        self._Y = np.zeros((3, model.nv))
        self._rg = {i: np.array(P[i].rigid_com) for i in self.full}
        self._mSrg = {i: -np.array([[0.0, -g[2], g[1]], [g[2], 0.0, -g[0]], [-g[1], g[0], 0.0]])
                      for i, g in self._rg.items()}
        self._vc_prev = None
        self._t_prev = None
        self._acc_est = {i: np.zeros(3) for i in self.full}
        self._vr_prev = {i: None for i in self.full}
        self.last = {}                      # diagnostics of the last step (per-body arrays)
        self.ez = np.array([0.0, 0.0, 1.0])

    # ------------------------------------------------------------------ kinematics
    def kinematics(self):
        """World / body velocities at each hydro body's CoM (fresh after mj_step1 / mj_forward)."""
        d, ids = self.d, self.ids
        R = d.xmat[ids].reshape(-1, 3, 3)
        xipos = d.xipos[ids]
        cvel = d.cvel[ids]
        w_w = cvel[:, :3]
        v_w = cvel[:, 3:] + crossk(w_w, xipos - d.subtree_com[self.root_ids])
        return R, d.xpos[ids], xipos, v_w, w_w

    # ------------------------------------------------------------------ main entry
    def apply(self, dt, t):
        """Add hydrodynamic forces to xfrc_applied (call after mj_step1, before thrusters / predictor).
        Returns the per-body water-relative body-frame velocities nu_r (k, 6) for thruster inflow."""
        d = self.d
        R, xpos, xipos, v_w, w_w = self.kinematics()
        vc_w = self.current(xipos, t)
        if self.unsteady and self._vc_prev is not None and dt > 0:
            ac_w = (vc_w - self._vc_prev) / dt
        else:
            ac_w = np.zeros_like(vc_w)
        self._vc_prev = vc_w
        # one batched R^T product for (v, w, v_c) of every body: (k, 3, 3)^T @ (k, 3, 3)
        S = np.stack([v_w, w_w, vc_w], axis=2)                    # (k, 3, 3), columns = vectors
        Sb = np.matmul(np.transpose(R, (0, 2, 1)), S)
        v_b, w_b, vc_b = Sb[:, :, 0], Sb[:, :, 1], Sb[:, :, 2]
        vr = v_b - vc_b
        nur = np.concatenate([vr, w_b], axis=1)
        cob_w = xpos + np.matmul(R, self.cob[:, :, None])[:, :, 0]
        z_cb = cob_w[:, 2]
        f_sub = np.minimum(np.maximum((self.hh - z_cb) / (2.0 * self.hh), 0.0), 1.0)
        rho = self.water.rho_at_depth(-z_cb) if self.water.rho_profile is not None else self.water.rho
        scale = f_sub * rho / self.rho_ref
        tau = -(scale[:, None] * (self.Dl + self.Dq * np.abs(nur))) * nur          # damping, body frame
        for i in self.full:                                                        # residual C_A
            a1 = self.dM[i] * vr[i]
            tau[i, :3] += cross3(a1, w_b[i])
            tau[i, 3:] += cross3(a1, vr[i])
        FM = np.matmul(R, tau.reshape(-1, 2, 3).transpose(0, 2, 1))              # (k, 3, 2)
        F, M = FM[:, :, 0], FM[:, :, 1]
        fb = f_sub * rho * self.vol * self.water.g                                  # buoyancy (world z)
        arm = cob_w - xipos
        M[:, 0] += arm[:, 1] * fb
        M[:, 1] -= arm[:, 0] * fb
        F[:, 2] += fb
        if self.unsteady:
            F += ((f_sub * rho * self.vol) + self.m_iso)[:, None] * ac_w
        d.xfrc_applied[self.ids, :3] += F
        d.xfrc_applied[self.ids, 3:] += M
        Fb = fb
        self.last = dict(R=R, xipos=xipos, v_b=v_b, w_b=w_b, vc_b=vc_b, vc_w=vc_w, ac_w=ac_w, nur=nur,
                         f_sub=f_sub, rho=rho, F_hydro=F, M_hydro=M, buoyancy=Fb)
        if self.scheme == "ema":
            for i in self.full:
                if self._vr_prev[i] is not None and dt > 0:
                    a_fd = (vr[i] - self._vr_prev[i]) / dt
                    self._acc_est[i] = (1 - self.alpha) * self._acc_est[i] + self.alpha * a_fd
                self._vr_prev[i] = vr[i].copy()
                d.xfrc_applied[self.ids[i], :3] += R[i] @ (-self.dM[i] * self._acc_est[i])
        return nur

    def correct(self):
        """Exact residual added-mass force for every full body (call after ALL other forces are written)."""
        if self.scheme != "pc" or not self.full:
            return
        m, d = self.m, self.d
        mujoco.mj_fwdActuation(m, d)
        mujoco.mj_fwdAcceleration(m, d)
        if d.nefc > 0:
            mujoco.mj_fwdConstraint(m, d)      # contacts / limits / welds active: constrained qacc0
            qacc = d.qacc
        else:
            qacc = d.qacc_smooth               # no constraint rows: qacc == qacc_smooth (and the
                                               # standalone mj_fwdConstraint call costs ~30 us)
        L = self.last
        for i in self.full:
            dM = self.dM[i]
            if not np.any(dM):
                continue
            j0 = self._dof[i]
            R = L["R"][i]
            Rt = R.T
            rg = self._rg[i]
            w = d.qvel[j0 + 3:j0 + 6]
            alpha = qacc[j0 + 3:j0 + 6]
            # body-frame CoM acceleration without the residual force:
            # s0 = R^T a_o + alpha x r_g + w x (w x r_g) - w x v_cg,b
            s0 = Rt @ qacc[j0:j0 + 3] + cross3(alpha, rg) + cross3(w, cross3(w, rg)) \
                - cross3(w, L["v_b"][i])
            c = cross3(w, L["vc_b"][i]) - Rt @ L["ac_w"][i]
            J = self._J
            J[:, j0:j0 + 3] = Rt
            J[:, j0 + 3:j0 + 6] = self._mSrg[i]
            mujoco.mj_solveM(m, d, self._Y, J)
            A = J @ self._Y.T
            J[:, j0:j0 + 6] = 0.0                 # leave J clean for the next full body
            A *= dM[:, None]
            A[0, 0] += 1.0
            A[1, 1] += 1.0
            A[2, 2] += 1.0
            f = -_solve3(A, dM * (s0 + c))
            d.xfrc_applied[self.ids[i], :3] += R @ f
            L.setdefault("f_res", {})[i] = f

    # ------------------------------------------------------------------ diagnostics
    def energy(self):
        """Kinetic energy (rigid + merged + residual added mass, relative to still water) plus the
        restoring potential sum_b (m_b g z_cg,b - rho V_b g z_cb,b).  Call after mj_forward."""
        m, d = self.m, self.d
        Mq = np.zeros(m.nv)
        mujoco.mj_mulM(m, d, Mq, d.qvel)
        ke = 0.5 * float(d.qvel @ Mq)
        R, xpos, xipos, v_w, w_w = self.kinematics()
        for i in self.full:
            vb = R[i].T @ v_w[i]
            ke += 0.5 * float(vb @ (self.dM[i] * vb))
        pe = 0.0
        g = self.water.g
        for i, p in enumerate(self.params):
            z_cg = xipos[i, 2]
            z_cb = (xpos[i] + R[i] @ p.cob)[2]
            rho = float(self.water.rho_at_depth(-z_cb))
            pe += p.rigid_mass * g * z_cg - rho * p.volume * g * z_cb
        # bodies without hydro records (dry internals) still carry gravity
        hydro = set(self.ids.tolist())
        for b in range(1, m.nbody):
            if b not in hydro:
                pe += m.body_mass[b] * (1.0 - m.body_gravcomp[b]) * g * d.xipos[b, 2]
        return ke + pe
