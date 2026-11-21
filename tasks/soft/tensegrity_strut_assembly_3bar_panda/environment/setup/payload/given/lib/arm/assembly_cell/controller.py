"""Grader-owned Panda controller (CHAIN_PLAN 2.5, 02 section 3.3).

* `joint` mode: the seven position actuators track a commanded configuration that moves towards the
  agent's target at <= 1.5 rad/s per joint (clamped to the joint ranges).
* `cartesian` mode: one damped-least-squares differential IK step per 10 ms tick on the COMMANDED
  configuration (a kinematic model, so contact compliance comes from the actuator stiffness, never
  from an integrator winding up against an obstacle): the tcp site moves towards the target pose at
  <= 0.5 m/s / 1.5 rad/s, joint speed <= 1.5 rad/s, ranges respected.
* gripper: the half-width position actuator on the finger tendon; the width command is rate-limited
  at 0.05 m/s; force = 2500 N/m x (object width - target width) / 2 per finger, capped at 60 N.
* `stiffness_scale` in [0.2, 1.0] scales the seven position gains (and their damping) of the Panda.
"""

import math

import numpy as np

__all__ = ["PandaController", "quat_to_mat", "mat_to_quat", "orientation_error"]

RATE_LIMIT_RAD_S = 1.5
GRIPPER_RATE_M_S = 0.05
CART_V_MAX = 0.5
CART_W_MAX = 1.5
CART_TAU = 0.08
DLS_LAMBDA = 0.05
RANGE_MARGIN = 0.01


def quat_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    from .layout import quat_from_rotmat
    return np.asarray(quat_from_rotmat(R), float)


def orientation_error(R_goal, R):
    """Rotation vector (world) that takes R to R_goal."""
    Re = R_goal @ R.T
    c = (np.trace(Re) - 1.0) / 2.0
    ang = math.acos(max(-1.0, min(1.0, c)))
    if ang < 1e-9:
        return np.zeros(3)
    axis = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-12:
        # angle ~ pi: take the axis from the symmetric part
        w, v = np.linalg.eigh(Re)
        axis = v[:, int(np.argmin(np.abs(w - 1.0)))]
        return axis / np.linalg.norm(axis) * ang
    return axis / n * ang


class PandaController(object):
    def __init__(self, model, data, tick_s=0.01, prefix="panda/"):
        import mujoco
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.tick_s = float(tick_s)
        self.joints = [model.joint(prefix + "joint%d" % (i + 1)).id for i in range(7)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joints])
        self.vadr = np.array([model.jnt_dofadr[j] for j in self.joints])
        self.acts = [model.actuator(prefix + "actuator%d" % (i + 1)).id for i in range(7)]
        self.grip_act = model.actuator(prefix + "gripper").id
        self.finger_q = [model.jnt_qposadr[model.joint(prefix + "finger_joint1").id], model.jnt_qposadr[model.joint(prefix + "finger_joint2").id]]
        self.tcp = model.site("tcp").id
        self.lower = np.array([model.jnt_range[j][0] for j in self.joints]) + RANGE_MARGIN
        self.upper = np.array([model.jnt_range[j][1] for j in self.joints]) - RANGE_MARGIN
        self.kp0 = np.array([model.actuator_gainprm[a][0] for a in self.acts])
        self.kv0 = np.array([-model.actuator_biasprm[a][2] for a in self.acts])
        self.max_width = float(model.actuator_ctrlrange[self.grip_act][1] * 2)
        self.scratch = mujoco.MjData(model)
        self.mode = "joint"
        self.stiffness_scale = 1.0
        self.q_cmd = data.qpos[self.qadr].copy()
        self.q_goal = self.q_cmd.copy()
        self.T_goal = None
        self.width_cmd = float(data.qpos[self.finger_q[0]] + data.qpos[self.finger_q[1]])
        self.width_goal = self.width_cmd
        self.set_stiffness(1.0)
        self.write_ctrl()

    # ---------------------------------------------------------------- kinematics on the commanded q
    def fk_cmd(self, q=None):
        q = self.q_cmd if q is None else q
        s = self.scratch
        s.qpos[:] = self.data.qpos
        s.qpos[self.qadr] = q
        self.mujoco.mj_kinematics(self.model, s)
        self.mujoco.mj_comPos(self.model, s)
        return s.site_xpos[self.tcp].copy(), s.site_xmat[self.tcp].reshape(3, 3).copy()

    def tcp_actual(self):
        return self.data.site_xpos[self.tcp].copy(), self.data.site_xmat[self.tcp].reshape(3, 3).copy()

    def _jacobian_cmd(self):
        s = self.scratch
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        self.mujoco.mj_jacSite(self.model, s, jp, jr, self.tcp)
        return jp[:, self.vadr], jr[:, self.vadr]

    # ---------------------------------------------------------------- API
    def set_stiffness(self, scale):
        scale = float(min(1.0, max(0.2, scale)))
        self.stiffness_scale = scale
        for i, a in enumerate(self.acts):
            self.model.actuator_gainprm[a][0] = scale * self.kp0[i]
            self.model.actuator_biasprm[a][1] = -scale * self.kp0[i]
            self.model.actuator_biasprm[a][2] = -scale * self.kv0[i]

    def set_mode(self, mode):
        if mode not in ("joint", "cartesian"):
            raise ValueError("mode must be joint or cartesian")
        if mode == self.mode:
            return
        self.mode = mode
        if mode == "joint":
            self.q_goal = self.q_cmd.copy()
        else:
            p, R = self.fk_cmd()
            self.T_goal = (p, R)

    def command_joint(self, q):
        q = np.asarray(q, float).reshape(7)
        if not np.all(np.isfinite(q)):
            raise ValueError("non-finite joint target")
        self.q_goal = np.clip(q, self.lower, self.upper)

    def command_cartesian(self, pos, quat):
        p = np.asarray(pos, float).reshape(3)
        q = np.asarray(quat, float).reshape(4)
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q)) and np.linalg.norm(q) > 1e-6):
            raise ValueError("non-finite Cartesian target")
        self.T_goal = (p, quat_to_mat(q))

    def command_width(self, width):
        w = float(width)
        if not math.isfinite(w):
            raise ValueError("non-finite width")
        self.width_goal = min(self.max_width, max(0.0, w))
        # the ramp starts from where the fingers ARE (a grasp holds them open on the object), so an
        # open command releases immediately instead of unwinding a fully-closed target first
        self.width_cmd = float(min(self.max_width, max(0.0, self.width())))

    # ---------------------------------------------------------------- per tick
    def tick(self):
        dq_max = RATE_LIMIT_RAD_S * self.tick_s
        if self.mode == "joint":
            step = np.clip(self.q_goal - self.q_cmd, -dq_max, dq_max)
            self.q_cmd = np.clip(self.q_cmd + step, self.lower, self.upper)
        else:
            p, R = self.fk_cmd()
            pg, Rg = self.T_goal
            e_p = pg - p
            e_r = orientation_error(Rg, R)
            v = e_p / CART_TAU
            nv = np.linalg.norm(v)
            if nv > CART_V_MAX:
                v *= CART_V_MAX / nv
            w = e_r / CART_TAU
            nw = np.linalg.norm(w)
            if nw > CART_W_MAX:
                w *= CART_W_MAX / nw
            twist = np.concatenate([v, w]) * self.tick_s
            if np.linalg.norm(twist) > 1e-9:
                jp, jr = self._jacobian_cmd()
                J = np.vstack([jp, jr])
                # joints at a range limit cannot move further into it
                at_lo = self.q_cmd <= self.lower + 1e-9
                at_hi = self.q_cmd >= self.upper - 1e-9
                W = np.eye(6) * DLS_LAMBDA ** 2
                dq = J.T @ np.linalg.solve(J @ J.T + W, twist)
                dq[(at_lo & (dq < 0)) | (at_hi & (dq > 0))] = 0.0
                m = np.max(np.abs(dq))
                if m > dq_max:
                    dq *= dq_max / m
                self.q_cmd = np.clip(self.q_cmd + dq, self.lower, self.upper)
        dw = GRIPPER_RATE_M_S * self.tick_s
        self.width_cmd += float(np.clip(self.width_goal - self.width_cmd, -dw, dw))
        self.write_ctrl()

    def write_ctrl(self):
        for i, a in enumerate(self.acts):
            self.data.ctrl[a] = self.q_cmd[i]
        self.data.ctrl[self.grip_act] = self.width_cmd / 2.0

    def width(self):
        return float(self.data.qpos[self.finger_q[0]] + self.data.qpos[self.finger_q[1]])

    def q(self):
        return self.data.qpos[self.qadr].copy()

    def dq(self):
        return self.data.qvel[self.vadr].copy()

    def cartesian_residual(self):
        if self.T_goal is None:
            return None
        p, R = self.tcp_actual()
        return float(np.linalg.norm(self.T_goal[0] - p)), float(np.linalg.norm(orientation_error(self.T_goal[1], R)))
