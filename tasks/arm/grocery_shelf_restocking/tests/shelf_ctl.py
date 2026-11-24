"""Shelf restocking: the controller (the operational-space controller of box transport at 10 Hz, on the lift).

`LiftCtl(model, lift_z)`: the Panda's base on the vertical slide joint `lift` (shelf_sim.base_world(lift=True)),
driven by a PD force plus the joint's gravity load, its set point moving at most shelf_sim.LIFT_STEP per control
step; while it moves the operational-space target rides with the base. `step(dp, dr, grip, dlift)` is one control
step; `servo`, `hold` and `grasped` are the scripted helpers the reference uses.
"""

import math

import mujoco
import numpy as np

import box_transport_env as B
import shelf_sim as S

GRIP_DEPTH = 0.022            # m, the grip site below the unit's top (top grasp, palm clearing it)
LIFT = 0.028                  # m, carrying clearance of the unit's bottom over the board / lip


def rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th > math.pi - 1e-6:
        w, V = np.linalg.eigh((R + R.T) / 2)
        return V[:, np.argmax(w)] * th
    return v / (2 * math.sin(th)) * th


def down_R(fingers_along):
    """Grip site pointing down (approach -z), the fingers' closing axis along world x or y."""
    z = np.array([0.0, 0.0, -1.0])
    x = np.array([1.0, 0.0, 0.0]) if fingers_along == "x" else np.array([0.0, 1.0, 0.0])
    return np.column_stack([x, np.cross(z, x), z])


class LiftCtl:
    """Box transport's operational-space controller at 10 Hz on the lift: see the module."""

    _osc_torque = B.BoxTransportEnv._osc_torque

    def __init__(self, m, lift_z):
        self.lift_adr = m.joint("lift").qposadr[0]
        self.lift_dof = m.joint("lift").dofadr[0]
        self.lift_act = m.actuator("lift_motor").id
        self.lift_set = float(lift_z)
        self.model = m
        self.data = d = mujoco.MjData(m)
        self.jq = np.array([m.joint(f"r_joint{i}").qposadr[0] for i in range(1, 8)])
        self.jv = np.array([m.joint(f"r_joint{i}").dofadr[0] for i in range(1, 8)])
        self.act = np.array([m.actuator(f"r_torq_j{i}").id for i in range(1, 8)])
        self.gact = [m.actuator("r_g_gripper_finger_joint1").id, m.actuator("r_g_gripper_finger_joint2").id]
        self.gq = [m.joint("r_g_finger_joint1").qposadr[0], m.joint("r_g_finger_joint2").qposadr[0]]
        self.site = m.site("r_g_grip_site").id
        self.pads = (m.geom("r_g_finger1_pad_collision").id, m.geom("r_g_finger2_pad_collision").id)
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._M = np.zeros((m.nv, m.nv))
        self.q_null = B.HOME_Q.copy()
        self.t = 0
        mujoco.mj_resetData(m, d)
        d.qpos[self.lift_adr] = lift_z
        d.qpos[self.jq] = B.HOME_Q
        d.qpos[self.gq[0]], d.qpos[self.gq[1]] = 0.04, -0.04
        mujoco.mj_forward(m, d)
        for _ in range(250):
            d.ctrl[self.act] = 200.0 * (B.HOME_Q - d.qpos[self.jq]) - 20.0 * d.qvel[self.jv] + d.qfrc_bias[self.jv]
            d.ctrl[self.gact] = [0.04, -0.04]
            d.ctrl[self.lift_act] = self._lift_force(self.lift_set)
            mujoco.mj_step(m, d)
        self.target_pos = d.site_xpos[self.site].copy()
        self.target_R = d.site_xmat[self.site].reshape(3, 3).copy()

    def _lift_force(self, z_set):
        d = self.data
        return (S.LIFT_KP * (z_set - d.qpos[self.lift_adr]) - S.LIFT_KD * d.qvel[self.lift_dof]
                + d.qfrc_bias[self.lift_dof])

    def lift(self):
        return float(self.data.qpos[self.lift_adr])

    def step(self, dp, dr, grip, dlift=0.0):
        d = self.data
        dp = np.asarray(dp, float)
        n = np.linalg.norm(dp)
        if n > B.MAX_DPOS:
            dp = dp * (B.MAX_DPOS / n)
        dr = np.asarray(dr, float)
        n = np.linalg.norm(dr)
        if n > B.MAX_DROT:
            dr = dr * (B.MAX_DROT / n)
        self.target_pos = d.site_xpos[self.site] + dp
        self.target_R = B.axis_angle_mat(dr) @ d.site_xmat[self.site].reshape(3, 3)
        gt = 0.04 * (1.0 - float(np.clip(grip, -1, 1))) / 2.0
        z0 = self.lift_set
        z1 = float(np.clip(z0 + np.clip(dlift, -S.LIFT_STEP, S.LIFT_STEP), *S.LIFT_RANGE))
        self.lift_set = z1
        for i in range(B.SUBSTEPS):
            prev = d.qpos[self.lift_adr]
            d.ctrl[self.lift_act] = self._lift_force(z0 + (z1 - z0) * (i + 1) / B.SUBSTEPS)
            d.ctrl[self.act] = self._osc_torque()
            d.ctrl[self.gact] = [gt, -gt]
            mujoco.mj_step(self.model, d)
            self.target_pos[2] += d.qpos[self.lift_adr] - prev       # the target rides with the base
        self.t += 1
        if not np.all(np.isfinite(d.qpos)) or np.max(np.abs(d.qvel[self.jv])) > B.MAX_ARM_SPEED:
            raise FloatingPointError("unstable")

    def lift_to(self, z, grip, max_steps=200):
        for _ in range(max_steps):
            dz = z - self.lift_set
            if abs(dz) < 1e-6 and abs(self.lift() - z) < 0.001:
                return True
            self.step(np.zeros(3), np.zeros(3), grip, dlift=dz)
        return False

    def ee(self):
        return self.data.site_xpos[self.site].copy()

    def ee_R(self):
        return self.data.site_xmat[self.site].reshape(3, 3).copy()

    def servo(self, target, R_goal, grip, speed=0.03, tol=0.003, max_steps=200):
        integral = np.zeros(3)
        for _ in range(max_steps):
            err = np.asarray(target, float) - self.ee()
            if grip > 0:
                integral = np.clip(integral + 0.15 * err, -0.02, 0.02)
            dp = err + (integral if grip > 0 else 0.0)
            n = np.linalg.norm(dp)
            if n > speed:
                dp *= speed / n
            dr = rotvec(R_goal @ self.ee_R().T)
            n = np.linalg.norm(dr)
            if n > 0.15:
                dr *= 0.15 / n
            self.step(dp, dr, grip)
            if np.linalg.norm(err) < tol and np.linalg.norm(dr) < 0.02:
                return True
        return False

    def hold(self, grip, steps):
        for _ in range(steps):
            self.step(np.zeros(3), np.zeros(3), grip)

    def grasped(self, body):
        m, d = self.model, self.data
        touching = set()
        for c in d.contact[:d.ncon]:
            for g1, g2 in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if m.geom_bodyid[g2] == body:
                    touching.add(int(g1))
        return all(p in touching for p in self.pads)
