"""Analytic leg kinematics of the quadruped, vectorised over the four legs (order FL, FR, HL, HR).

The chain per leg (read from the compiled model, so it follows any change of the MJCF):
  base --(hip ab/ad, about x, at h0)--> hip --(t0)--> thigh: hip flexion about y --(s0)--> shank: knee about y
  --(f0)--> foot-sphere centre.
All link frames are parallel to the base at q = 0. Positive hip flexion swings the thigh backward.
"""

from __future__ import annotations

import numpy as np

from .fastmath import cross_rows

LEGS = ("FL", "FR", "HL", "HR")


class LegKinematics:
    def __init__(self, model, prefix: str = ""):
        P = lambda n: prefix + n
        self.h0 = np.array([model.body_pos[model.body(P(l + "_hip")).id] for l in LEGS])
        self.t0 = np.array([model.body_pos[model.body(P(l + "_thigh")).id] for l in LEGS])
        self.s0 = np.array([model.body_pos[model.body(P(l + "_shank")).id] for l in LEGS])
        self.f0 = np.array([model.site_pos[model.site(P(l + "_foot")).id] for l in LEGS])
        self.L1 = -self.s0[:, 2]                      # thigh length (0.16)
        self.L2 = -self.f0[:, 2]                      # shank length (0.16)
        self.Ly = self.t0[:, 1] + self.s0[:, 1] + self.f0[:, 1]   # lateral offset of the leg plane (+-0.06045)
        self.dx = self.t0[:, 0]
        # link inertials
        self.m_hip = np.array([model.body_mass[model.body(P(l + "_hip")).id] for l in LEGS])
        self.m_thigh = np.array([model.body_mass[model.body(P(l + "_thigh")).id] for l in LEGS])
        self.m_shank = np.array([model.body_mass[model.body(P(l + "_shank")).id] for l in LEGS])
        self.c_hip = np.array([model.body_ipos[model.body(P(l + "_hip")).id] for l in LEGS])
        self.c_thigh = np.array([model.body_ipos[model.body(P(l + "_thigh")).id] for l in LEGS])
        self.c_shank = np.array([model.body_ipos[model.body(P(l + "_shank")).id] for l in LEGS])
        # neutral foot position under the hip (leg plane), base frame, z = 0
        self.neutral = np.stack([self.h0[:, 0] + self.dx, self.h0[:, 1] + self.Ly, np.zeros(4)], axis=1)

    # ------------------------------------------------------------------ forward kinematics
    def fk(self, q: np.ndarray):
        """q (4, 3) -> foot positions (4, 3) in the base frame, Jacobians (4, 3, 3) d p / d q."""
        q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2]
        c1, s1 = np.cos(q1), np.sin(q1)
        s2, c2 = np.sin(q2), np.cos(q2)
        s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)
        L1, L2 = self.L1, self.L2
        vx = self.dx - L1 * s2 - L2 * s23
        vy = self.Ly
        vz = -L1 * c2 - L2 * c23
        p = np.stack([self.h0[:, 0] + vx, self.h0[:, 1] + c1 * vy - s1 * vz, self.h0[:, 2] + s1 * vy + c1 * vz], 1)
        J = np.zeros((4, 3, 3))
        # d/dq1
        J[:, 1, 0] = -s1 * vy - c1 * vz
        J[:, 2, 0] = c1 * vy - s1 * vz
        # d v / d q2 and d q3 in the hip frame, then rotate about x
        dvx2, dvz2 = -L1 * c2 - L2 * c23, L1 * s2 + L2 * s23
        dvx3, dvz3 = -L2 * c23, L2 * s23
        J[:, 0, 1] = dvx2
        J[:, 1, 1] = -s1 * dvz2
        J[:, 2, 1] = c1 * dvz2
        J[:, 0, 2] = dvx3
        J[:, 1, 2] = -s1 * dvz3
        J[:, 2, 2] = c1 * dvz3
        return p, J

    # ------------------------------------------------------------------ inverse kinematics
    def ik(self, p: np.ndarray, knee_sign: np.ndarray, q_prev: np.ndarray | None = None):
        """Foot targets (4, 3) in the base frame -> joint angles (4, 3). knee_sign -1: knee angle negative (front
        legs of the default stance), +1: positive (hind legs). Unreachable targets are projected onto the
        workspace boundary."""
        u = p - self.h0
        uy, uz = u[:, 1], u[:, 2]
        D2 = uy * uy + uz * uz
        Ly = self.Ly
        zpl = -np.sqrt(np.maximum(D2 - Ly * Ly, 1e-8))
        q1 = np.arctan2(uz, uy) - np.arctan2(zpl, Ly)
        q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
        xpl = u[:, 0] - self.dx
        L1, L2 = self.L1, self.L2
        r2 = xpl * xpl + zpl * zpl
        rmax = (L1 + L2) * 0.999
        rmin = np.abs(L1 - L2) + 0.03
        r = np.sqrt(r2)
        scale = np.clip(r, rmin, rmax) / np.maximum(r, 1e-9)
        xpl, zpl = xpl * scale, zpl * scale
        r2 = xpl * xpl + zpl * zpl
        c3 = np.clip((r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1.0, 1.0)
        q3 = knee_sign * np.arccos(c3)
        alpha = np.arctan2(-xpl, -zpl)
        beta = np.arctan2(L2 * np.sin(q3), L1 + L2 * np.cos(q3))
        q2 = alpha - beta
        return np.stack([q1, q2, q3], 1)

    def ik_leg(self, i: int, p: np.ndarray, knee_sign: float):
        """Single-leg version of ik(): foot target p (3,) in the base frame -> (3,) joint angles."""
        u = p - self.h0[i]
        uy, uz = u[1], u[2]
        Ly = self.Ly[i]
        zpl = -np.sqrt(max(uy * uy + uz * uz - Ly * Ly, 1e-8))
        q1 = np.arctan2(uz, uy) - np.arctan2(zpl, Ly)
        q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
        xpl = u[0] - self.dx[i]
        L1, L2 = self.L1[i], self.L2[i]
        r = np.hypot(xpl, zpl)
        rc = min(max(r, abs(L1 - L2) + 0.03), (L1 + L2) * 0.999)
        xpl, zpl = xpl * rc / max(r, 1e-9), zpl * rc / max(r, 1e-9)
        c3 = np.clip((rc * rc - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1.0, 1.0)
        q3 = knee_sign * np.arccos(c3)
        q2 = np.arctan2(-xpl, -zpl) - np.arctan2(L2 * np.sin(q3), L1 + L2 * np.cos(q3))
        return np.array([q1, q2, q3])

    # ------------------------------------------------------------------ mass properties
    def link_coms(self, q):
        """CoMs (4, 3) of hip, thigh, shank links and the hip-flexion / knee joint points, base frame."""
        q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2]
        c1, s1 = np.cos(q1), np.sin(q1)

        def ry(a, v):
            c, s = np.cos(a), np.sin(a)
            return np.stack([c * v[:, 0] + s * v[:, 2], v[:, 1], -s * v[:, 0] + c * v[:, 2]], 1)

        def rx(v):
            return np.stack([v[:, 0], c1 * v[:, 1] - s1 * v[:, 2], s1 * v[:, 1] + c1 * v[:, 2]], 1)
        hip = self.h0 + rx(self.c_hip)
        thigh = self.h0 + rx(self.t0 + ry(q2, self.c_thigh))
        shank = self.h0 + rx(self.t0 + ry(q2, self.s0 + ry(q3, self.c_shank)))
        p_hfe = self.h0 + rx(self.t0)
        p_kfe = self.h0 + rx(self.t0 + ry(q2, self.s0))
        a2 = np.stack([np.zeros(4), c1, s1], 1)          # hip-flexion / knee axis (Rx(q1) e_y)
        return hip, thigh, shank, p_hfe, p_kfe, a2

    def com_and_gravity(self, q, g_b, trunk_mass, trunk_com):
        """Whole-robot CoM (base frame) and the joint torques (4, 3) that hold each leg against gravity."""
        lc = self.link_coms(q)
        hip, thigh, shank = lc[0], lc[1], lc[2]
        msum = (self.m_hip @ hip + self.m_thigh @ thigh + self.m_shank @ shank)
        mt = trunk_mass + self.m_hip.sum() + self.m_thigh.sum() + self.m_shank.sum()
        com = (trunk_mass * trunk_com + msum) / mt
        return com, self.gravity_torque(q, g_b, lc)

    def gravity_torque(self, q, g_b, lc=None):
        """Joint torques (4, 3) that hold each leg against gravity (g_b = gravity vector in the base frame)."""
        hip, thigh, shank, p_hfe, p_kfe, a2 = self.link_coms(q) if lc is None else lc
        fh = self.m_hip[:, None] * g_b
        ft = self.m_thigh[:, None] * g_b
        fs = self.m_shank[:, None] * g_b
        ex = np.array([1.0, 0.0, 0.0])
        m1 = cross_rows(hip - self.h0, fh) + cross_rows(thigh - self.h0, ft) + cross_rows(shank - self.h0, fs)
        m2 = cross_rows(thigh - p_hfe, ft) + cross_rows(shank - p_hfe, fs)
        m3 = cross_rows(shank - p_kfe, fs)
        tau = np.stack([-(m1 @ ex), -np.sum(m2 * a2, 1), -np.sum(m3 * a2, 1)], 1)
        return tau
