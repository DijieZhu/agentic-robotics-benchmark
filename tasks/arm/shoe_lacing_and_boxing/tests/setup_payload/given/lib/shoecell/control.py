"""Per-arm joint-position controller for the dual-Panda work cell.

Ported from the Agentic Robotics Benchmark assembly-cell controller
(tasks/arm/robotic_arm_full_assembly_so101_panda/setup/payload/given/lib/arm/assembly_cell/controller.py,
Apache-2.0, same benchmark) and generalised to one controller per arm with a name prefix,
an arbitrary tool site and an optional gripper.

* ``joint`` mode: the seven position actuators track a commanded configuration that moves
  towards the target at <= RATE_LIMIT_RAD_S per joint (clamped to the joint ranges minus a margin).
* ``cartesian`` mode: one damped-least-squares differential IK step per control tick on the
  COMMANDED configuration (a kinematic model: contact compliance comes from the actuator
  stiffness, never from an integrator winding up against an obstacle).  The tool site moves
  towards the target pose at <= CART_V_MAX m/s and <= CART_W_MAX rad/s, joint speed is capped
  at RATE_LIMIT_RAD_S and the ranges are respected.
* gripper (left hand only): the Menagerie finger tendon actuator (``<prefix>actuator8``,
  ctrl range 0..255) is driven by a width command in [0, MAX_WIDTH] that is ramp-limited at
  GRIPPER_RATE_M_S.
* ``stiffness_scale`` in [0.2, 1.0] scales the seven position gains and their damping.

Control tick = TICK_S (10 ms) = SUBSTEPS physics steps of 1 ms: ``tick()`` then
``SUBSTEPS x mj_step`` (see :func:`run_ticks`).  Nothing here reads a wall clock.

Runtime dependencies: mujoco + numpy only.
"""

import math

import mujoco
import numpy as np

__all__ = [
    "ArmController", "run_ticks", "ik_solve", "ik_solve_seeds", "arm_collisions", "SEED_OFFSETS",
    "quat_to_mat", "mat_to_quat", "quat_mul", "quat_inv", "orientation_error", "rotz",
    "TICK_S", "SUBSTEPS", "MAX_WIDTH",
]

TICK_S = 0.01
SUBSTEPS = 10
RATE_LIMIT_RAD_S = 1.5
GRIPPER_RATE_M_S = 0.05
CART_V_MAX = 0.5
CART_W_MAX = 1.5
CART_TAU = 0.08
DLS_LAMBDA = 0.05
RANGE_MARGIN = 0.01
MAX_WIDTH = 0.08          # two fingers x 0.04 m slide range
# joint offsets added to a reference configuration to seed ik_solve_seeds (wrist flips: joints 5 and 7)
SEED_OFFSETS = [
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, -math.pi),
    (0, 0, 0, 0, 0, 0, math.pi),
    (0, 0, 0, 0, math.pi / 2, 0, 0),
    (0, 0, 0, 0, -math.pi / 2, 0, 0),
    (0, 0, 0, 0, 0, 0, -math.pi / 2),
    (0, 0, 0, 0, 0, 0, math.pi / 2),
    (0, 0, 0, 0, math.pi / 2, 0, -math.pi / 2),
    (0, 0, 0, 0, -math.pi / 2, 0, math.pi / 2),
]


# ------------------------------------------------------------------------------------ rotations
def quat_to_mat(q):
    """3x3 rotation matrix of a wxyz quaternion (normalised first)."""
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    """wxyz quaternion of a 3x3 rotation matrix (MuJoCo's own conversion)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def quat_mul(a, b):
    """Hamilton product a*b of two wxyz quaternions."""
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def quat_inv(q):
    """Conjugate (inverse for a unit quaternion) of a wxyz quaternion."""
    out = np.zeros(4)
    mujoco.mju_negQuat(out, np.asarray(q, float))
    return out


def rotz(psi):
    """Rotation matrix about z by psi radians."""
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def orientation_error(R_goal, R):
    """Rotation vector (world frame) that takes R to R_goal."""
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


# ------------------------------------------------------------------------------------ controller
class ArmController(object):
    """Joint-position / differential-IK controller of one Panda arm in the cell.

    Parameters
    ----------
    model, data : the compiled cell.
    prefix : the arm's name prefix (``'left/'`` or ``'right/'``).
    tcp_site : name of the tool site the Cartesian mode moves (``'left/tcp'`` / ``'right/tcp'``).
    gripper : True when the arm carries the Menagerie hand (finger tendon actuator ``actuator8``).
    tick_s : control period in seconds (10 ms).
    """

    def __init__(self, model, data, prefix, tcp_site, gripper=False, tick_s=TICK_S):
        self.model = model
        self.data = data
        self.prefix = prefix
        self.tick_s = float(tick_s)
        self.joints = [model.joint(prefix + "joint%d" % (i + 1)).id for i in range(7)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joints])
        self.vadr = np.array([model.jnt_dofadr[j] for j in self.joints])
        self.acts = [model.actuator(prefix + "actuator%d" % (i + 1)).id for i in range(7)]
        self.tcp = model.site(tcp_site).id
        self.lower = np.array([model.jnt_range[j][0] for j in self.joints]) + RANGE_MARGIN
        self.upper = np.array([model.jnt_range[j][1] for j in self.joints]) - RANGE_MARGIN
        self.kp0 = np.array([model.actuator_gainprm[a][0] for a in self.acts])
        self.kv0 = np.array([-model.actuator_biasprm[a][2] for a in self.acts])
        self.has_gripper = bool(gripper)
        if gripper:
            self.grip_act = model.actuator(prefix + "actuator8").id
            self.finger_q = [model.jnt_qposadr[model.joint(prefix + "finger_joint1").id],
                             model.jnt_qposadr[model.joint(prefix + "finger_joint2").id]]
            # ctrl range 0..255 of the Menagerie tendon actuator <-> width 0..MAX_WIDTH
            self.ctrl_per_m = float(model.actuator_ctrlrange[self.grip_act][1]) / MAX_WIDTH
        self.scratch = mujoco.MjData(model)
        self.mode = "joint"
        self.stiffness_scale = 1.0
        self.q_cmd = data.qpos[self.qadr].copy()
        self.q_goal = self.q_cmd.copy()
        self.T_goal = None
        self.width_cmd = self.width() if gripper else 0.0
        self.width_goal = self.width_cmd
        self.set_stiffness(1.0)
        self.write_ctrl()

    # ---------------------------------------------------------------- kinematics on the commanded q
    def fk_cmd(self, q=None):
        """Tool site (pos, 3x3 mat) at the commanded (or given) configuration; the other joints as in data."""
        q = self.q_cmd if q is None else q
        s = self.scratch
        s.qpos[:] = self.data.qpos
        s.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, s)
        mujoco.mj_comPos(self.model, s)
        return s.site_xpos[self.tcp].copy(), s.site_xmat[self.tcp].reshape(3, 3).copy()

    def tcp_actual(self):
        """Measured tool site (pos, 3x3 mat) from data."""
        return self.data.site_xpos[self.tcp].copy(), self.data.site_xmat[self.tcp].reshape(3, 3).copy()

    def _jacobian_cmd(self):
        """(3x7, 3x7) translational / rotational Jacobians of the tool site at the scratch state."""
        s = self.scratch
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, s, jp, jr, self.tcp)
        return jp[:, self.vadr], jr[:, self.vadr]

    def ik_step(self, q, pos, R_goal, dq_max=None, free_axis=False):
        """One DLS step of q towards the tool pose (pos, R_goal); returns (q_new, e_pos_norm, e_rot_norm).

        free_axis=True leaves the rotation about the tool's own z axis unconstrained (5-DoF target:
        only the approach direction matters), used by the reach analysis.
        """
        p, R = self.fk_cmd(q)
        e_p = np.asarray(pos, float) - p
        e_r = orientation_error(R_goal, R)
        if free_axis:
            z = R[:, 2]
            e_r = e_r - np.dot(e_r, z) * z
        v = e_p / CART_TAU
        nv = np.linalg.norm(v)
        if nv > CART_V_MAX:
            v *= CART_V_MAX / nv
        w = e_r / CART_TAU
        nw = np.linalg.norm(w)
        if nw > CART_W_MAX:
            w *= CART_W_MAX / nw
        twist = np.concatenate([v, w]) * self.tick_s
        if np.linalg.norm(twist) < 1e-9:
            return q.copy(), float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
        jp, jr = self._jacobian_cmd()
        J = np.vstack([jp, jr])
        if free_axis:
            # drop the rotational component about the tool z axis from the problem
            z = R[:, 2]
            P = np.eye(3) - np.outer(z, z)
            J = np.vstack([jp, P @ jr])
        at_lo = q <= self.lower + 1e-9
        at_hi = q >= self.upper - 1e-9
        W = np.eye(6) * DLS_LAMBDA ** 2
        dq = J.T @ np.linalg.solve(J @ J.T + W, twist)
        dq[(at_lo & (dq < 0)) | (at_hi & (dq > 0))] = 0.0
        if dq_max is not None:
            m = np.max(np.abs(dq))
            if m > dq_max:
                dq *= dq_max / m
        q_new = np.clip(q + dq, self.lower, self.upper)
        return q_new, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))

    # ---------------------------------------------------------------- API
    def set_stiffness(self, scale):
        """Scale the seven arm position gains (and their damping) by scale in [0.2, 1]."""
        scale = float(min(1.0, max(0.2, scale)))
        self.stiffness_scale = scale
        for i, a in enumerate(self.acts):
            self.model.actuator_gainprm[a][0] = scale * self.kp0[i]
            self.model.actuator_biasprm[a][1] = -scale * self.kp0[i]
            self.model.actuator_biasprm[a][2] = -scale * self.kv0[i]

    def set_mode(self, mode):
        """Switch between 'joint' and 'cartesian'; the new goal starts at the current command."""
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
        """Set the joint-mode goal (7 radians, clipped to the ranges)."""
        q = np.asarray(q, float).reshape(7)
        if not np.all(np.isfinite(q)):
            raise ValueError("non-finite joint target")
        self.q_goal = np.clip(q, self.lower, self.upper)

    def command_cartesian(self, pos, quat):
        """Set the Cartesian-mode goal: tool site position (m) and wxyz orientation."""
        p = np.asarray(pos, float).reshape(3)
        q = np.asarray(quat, float).reshape(4)
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q)) and np.linalg.norm(q) > 1e-6):
            raise ValueError("non-finite Cartesian target")
        self.T_goal = (p, quat_to_mat(q))

    def command_width(self, width):
        """Set the gripper width goal in metres (0..MAX_WIDTH); the ramp starts from the measured width."""
        if not self.has_gripper:
            raise RuntimeError("arm %s has no gripper" % self.prefix)
        w = float(width)
        if not math.isfinite(w):
            raise ValueError("non-finite width")
        self.width_goal = min(MAX_WIDTH, max(0.0, w))
        # the ramp starts from where the fingers ARE (a grasp holds them open on the object), so an
        # open command releases immediately instead of unwinding a fully-closed target first
        self.width_cmd = float(min(MAX_WIDTH, max(0.0, self.width())))

    # ---------------------------------------------------------------- per tick
    def tick(self):
        """Advance the commanded configuration / width by one control period and write ctrl."""
        dq_max = RATE_LIMIT_RAD_S * self.tick_s
        if self.mode == "joint":
            step = np.clip(self.q_goal - self.q_cmd, -dq_max, dq_max)
            self.q_cmd = np.clip(self.q_cmd + step, self.lower, self.upper)
        else:
            pg, Rg = self.T_goal
            self.q_cmd, _, _ = self.ik_step(self.q_cmd, pg, Rg, dq_max=dq_max)
        if self.has_gripper:
            dw = GRIPPER_RATE_M_S * self.tick_s
            self.width_cmd += float(np.clip(self.width_goal - self.width_cmd, -dw, dw))
        self.write_ctrl()

    def write_ctrl(self):
        """Write the commanded joints (and width) into data.ctrl."""
        for i, a in enumerate(self.acts):
            self.data.ctrl[a] = self.q_cmd[i]
        if self.has_gripper:
            self.data.ctrl[self.grip_act] = self.width_cmd * self.ctrl_per_m

    def width(self):
        """Measured finger opening (sum of the two finger slides), metres."""
        if not self.has_gripper:
            return 0.0
        return float(self.data.qpos[self.finger_q[0]] + self.data.qpos[self.finger_q[1]])

    def q(self):
        """Measured arm configuration (7)."""
        return self.data.qpos[self.qadr].copy()

    def dq(self):
        """Measured arm joint velocities (7)."""
        return self.data.qvel[self.vadr].copy()

    def cartesian_residual(self):
        """(position error m, orientation error rad) of the measured tool pose vs the Cartesian goal."""
        if self.T_goal is None:
            return None
        p, R = self.tcp_actual()
        return float(np.linalg.norm(self.T_goal[0] - p)), float(np.linalg.norm(orientation_error(self.T_goal[1], R)))

    def joint_residual(self):
        """Max |q_goal - q_measured| over the seven joints (rad)."""
        return float(np.max(np.abs(self.q_goal - self.q())))


def run_ticks(model, data, controllers, n_ticks, substeps=SUBSTEPS):
    """Run n_ticks control ticks: every tick calls tick() on each controller then substeps x mj_step."""
    for _ in range(int(n_ticks)):
        for c in controllers:
            c.tick()
        for _ in range(substeps):
            mujoco.mj_step(model, data)


def ik_solve(ctrl, pos, R_goal, q0=None, iters=300, pos_tol=0.002, rot_tol=math.radians(1.0), free_axis=False):
    """Iterate ctrl.ik_step without rate limits from q0 (default: the commanded q).

    Returns (q, pos_err, rot_err, converged).  The controller's state is not modified.
    """
    q = (ctrl.q_cmd if q0 is None else np.asarray(q0, float)).copy()
    e_p = e_r = float("inf")
    for _ in range(int(iters)):
        q, e_p, e_r = ctrl.ik_step(q, pos, R_goal, dq_max=None, free_axis=free_axis)
        if e_p < pos_tol and e_r < rot_tol:
            break
    # final residual at q
    p, R = ctrl.fk_cmd(q)
    e_p = float(np.linalg.norm(np.asarray(pos, float) - p))
    e_rv = orientation_error(R_goal, R)
    if free_axis:
        z = R[:, 2]
        e_rv = e_rv - np.dot(e_rv, z) * z
    e_r = float(np.linalg.norm(e_rv))
    return q, e_p, e_r, bool(e_p < pos_tol and e_r < rot_tol)


def arm_collisions(ctrl, q, min_pen=0.001):
    """Contacts (body1, body2, penetration) of the arm's bodies at configuration q (kinematic check on the scratch data)."""
    m, s = ctrl.model, ctrl.scratch
    s.qpos[:] = ctrl.data.qpos
    s.qpos[ctrl.qadr] = q
    mujoco.mj_forward(m, s)
    out = []
    for i in range(s.ncon):
        c = s.contact[i]
        b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
        if (b1.startswith(ctrl.prefix) or b2.startswith(ctrl.prefix)) and -c.dist > min_pen:
            out.append((b1, b2, -float(c.dist)))
    return out


def ik_solve_seeds(ctrl, pos, R_goal, q_ref=None, seeds=None, iters=300, pos_tol=0.002, rot_tol=math.radians(1.0),
                   free_axis=False, collision_free=True, min_pen=0.001):
    """Global-ish IK: run ik_solve from several seeds, return the best (q, pos_err, rot_err, ok, collisions).

    The damped-least-squares step is local and stalls at joint limits for wrist-flipped targets (a tool
    pitched forward needs joint 5 or 7 on the other side of its range), so seeds = q_ref + SEED_OFFSETS
    (q_ref defaults to the commanded q) are tried in order; the first converged solution that is
    collision-free (when collision_free) wins, otherwise the one with the smallest residual.
    """
    q_ref = (ctrl.q_cmd if q_ref is None else np.asarray(q_ref, float)).copy()
    seeds = [q_ref + np.asarray(o, float) for o in (SEED_OFFSETS if seeds is None else seeds)]
    best = None
    for q0 in seeds:
        q0 = np.clip(q0, ctrl.lower, ctrl.upper)
        q, ep, er, ok = ik_solve(ctrl, pos, R_goal, q0=q0, iters=iters, pos_tol=pos_tol, rot_tol=rot_tol, free_axis=free_axis)
        cols = arm_collisions(ctrl, q, min_pen) if ok else []
        cand = (q, ep, er, ok, cols)
        if ok and (not collision_free or not cols):
            return cand
        score = (0 if ok else 1, len(cols), ep + 0.1 * er)
        if best is None or score < best[0]:
            best = (score, cand)
    return best[1]
