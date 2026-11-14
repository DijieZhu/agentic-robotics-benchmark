"""Grader-owned controller of the UR arm and its Robotiq 2F-85 (the PandaController API and semantics).

Numbers below were measured with the cell's robot smoke test (UR10e, the empty cell, 1 ms implicitfast, noslip 5, 10
physics steps per 10 ms tick; the test's keys in brackets).

* `joint` mode: the commanded configuration q_cmd moves towards the agent's target at <= 1.5 rad/s per joint (each
  joint clamped separately, as the Panda), clamped to the joint ranges minus 0.01 rad. 1.5 rad/s is under every UR
  joint speed limit (UR10e base / shoulder 120 deg/s = 2.09 rad/s, elbow and wrists 180 deg/s; UR5e 180 deg/s, read
  from the model's numeric `ur/joint_speed_limit_rad_s`): the rate is min(1.5, limit). A start from rest at 1.5
  rad/s saturates the shoulder torque, the joints lag up to 21 mrad and catch up at up to 1.8 rad/s, 4.7 mrad past
  the goal, within 1 mrad 30 ms after q_cmd arrives [b.step].
* `cartesian` mode: one damped-least-squares differential IK step per control tick on the COMMANDED configuration (a
  kinematic model, so contact compliance comes from the joint stiffness, never from an integrator winding up against
  an obstacle): the tcp moves towards the target pose at <= 0.5 m/s / 1.5 rad/s (P law, time constant 0.08 s),
  joint rate <= 1.5 rad/s, joints at a range limit frozen. The UR has no null space: near the wrist (sin q5 = 0),
  elbow (sin q3 = 0) and shoulder singularities the DLS (lambda 0.05) slows the tcp instead of spinning the joints.
  0.3 m lines: path deviation <= 0.46 mm at 0.25 m/s and <= 1.0 mm at 0.5 m/s (streamed targets), <= 1.8 mm with one
  far target (the 0.5 m/s cap); orientation error <= 1.5 / 2.8 / 6.0 mrad [c].
* the arm is driven by COMPUTED TORQUE through the six motors (ctrl = joint torque):
      tau = qfrc_bias - qfrc_passive + M(q) (Kp (q_ref - q) + Kd (dq_ref - dq))
  M(q) (with the joints' 0.1 kg m^2 armature), qfrc_bias (gravity, Coriolis) and qfrc_passive (the joints' damping
  10 / 10 / 5 / 2 / 2 / 2 N m s/rad) are MuJoCo's own for that step, so the law is exact for the modelled arm +
  gripper (no load: static tcp error < 1e-11 mm [b.static_no_load]) and the tracking error obeys
  e'' + Kd e' + Kp e = M^-1 tau_ext per joint; a held part's weight and every contact are the disturbances tau_ext.
  tau is kept inside the motor ratings by scaling the PD part uniformly (when a joint saturates the arm stays on its
  joint-space path instead of bending off it), then clipped.
* the torque is recomputed at EVERY physics step (1 kHz; `step_physics(n)`, or `torque()` between mj_step1 and
  mj_step2), not once per 10 ms tick [zoh]. At these gains a 10 ms zero-order hold is stable (its step even overshoots
  less, 0.7 against 4.7 mrad), but the wrists' loop then sits at Kd h = 1.8, 90 % of the hold's stability limit (2,
  critically damped double integrator): a 3 N m / 20 ms torque pulse on wrist 2 doubles its velocity transient (0.21
  against 0.094 rad/s), and with 100 rad/s wrists the hold rings (27 sign changes after the pulse, none per step).
  Per step the loop is a continuous PD (w h <= 0.09) at any gain the cell may want, as the Panda cell's MuJoCo
  position servos were and as the real e-Series servo (500 Hz) is. Cost: 11-24 us per step on this (loaded) host
  (mj_mulM plus six-element arithmetic on views of data; M and the bias are MuJoCo's own from mj_step1, step1 +
  step2 = one mj_step), against 23-52 us for the physics of the arm alone and 115-139 us holding a part [f].
* the cell's loop per control tick: `ct.tick()` (the commanded configuration, the gripper ctrl), then the tick's
  physics steps as `ct.step_physics()` or 10 x (mj_step1, `ct.torque()`, mj_step2), then mj_forward. A loop of plain
  mj_step would hold the torque of the tick's start (the zero-order hold above), and before any tick the arm holds
  the q_cmd set by the caller (`write_ctrl()` after setting q_cmd / width_cmd, as the Panda cells do).
* the reference inside a tick: q_ref runs linearly from the previous tick's q_cmd to the new one over the tick's
  physics steps with dq_ref = the tick's mean velocity (velocity feed-forward: a 0.5 rad/s joint ramp is tracked to
  0.0003 mrad [b.ramp]; a_ff = 0: a rate-limited command carries no acceleration information inside a tick).
* gains (OMEGA, per joint, rad/s; Kp = w^2, Kd = 2 w: critically damped): 45 / 45 / 60 for the base, shoulder and
  elbow, 90 for the wrists. With M(q) in the loop the joint stiffness is M Kp and the tcp's Cartesian stiffness is
  K = (J (M Kp)^-1 J^T)^-1 (`cartesian_stiffness()`). Over eight work poses of the cell (bench picks 0.35-0.9 m out,
  the stand's top and sides) the translational stiffness for a point force is 8.8-52 N/mm at scale 1.0 (79 N/mm along
  the arm near the shoulder singularity); the soft directions are the wrist's (its armature at the 0.16-0.3 m tool
  lever is ~1.7 kg of effective mass). A 1 kg part held sags 0.39-0.61 mm (1.07 mm at the stand's near side with the
  tool horizontal), as predicted by K to 0.1 % [b.static_1kg, gains.chosen.stiffness]; the kit's 150 g units sag
  < 0.1 mm. Pushing on a flat block (the tool also restrained in tilt and slide: K_zz, 20 % above the point-force
  value) a stalled push builds up 24 N per mm of residual at scale 1.0 and 7.2 N per mm at 0.3, steady (no build-up
  over 2 s) [g]. Gain sets compared [gains]: uniform 60 rad/s sags 1.5 mm and is 104 N/mm stiff (a 10 mm stall: 348
  against 241 N); uniform 45 sags 2.7 mm; 50 / 50 / 60 / 100 sags 0.97 mm at 268 N. The cost of unequal gains: the
  joints lag by different amounts in fast moves, so a 0.5 m/s line turns the tool by 2.8 mrad (0.4 with uniform
  gains); the path deviation is the same.
* `set_stiffness(scale)` in [0.2, 1.0]: Kp x scale, Kd x sqrt(scale) (the damping ratio stays 1 at every scale; the
  Panda scaled kv linearly, which left it at zeta 0.45 at 0.2).
* contact transients: the first 1-2 ms of a contact stop the arm's effective mass (7-9 kg vertically) on the stiff pad
  contact: touching a fixed block at 2 cm/s gives a 111-181 N one-step peak (0.13-0.19 N s), back under 20 N within
  3 ms; a per-tick (10 ms) safety watch sees the steady force [g].
* gripper: `command_width(w)` sets a target pad gap 0..0.085 m; the commanded gap ramps at 0.10 m/s (a full close
  takes 0.85 s) and is mapped to the 2F-85 servo's ctrl through the free-air opening table
  (`gripper_2f85.ctrl_of_width`). Closing from fully open onto 20-45 mm parts the contact peaks at 80-90 N per pad
  against the 78-81 N steady grip (0.12 m/s: 99-108 N; a ctrl step: 111-222 N) [table.closing_impact]. The ramp
  restarts from the measured gap only when that removes a jump: opening past a held part starts at the part (an open
  releases at once, as the Panda), while re-commanding a closed gripper keeps the squeeze (no momentary release, the
  Panda's rule would drop the grip for a ramp). `width()` is the measured gap between the pad faces at their middle.
"""
import math

import numpy as np

from . import gripper_2f85 as g2f
from .ur_robot import ACTUATOR_NAMES, JOINT_NAMES

__all__ = ["URController", "quat_to_mat", "mat_to_quat", "orientation_error", "OMEGA_RAD_S"]

RATE_LIMIT_RAD_S = 1.5
CART_V_MAX = 0.5
CART_W_MAX = 1.5
CART_TAU = 0.08
DLS_LAMBDA = 0.05
RANGE_MARGIN = 0.01
OMEGA_RAD_S = [45.0, 45.0, 60.0, 90.0, 90.0, 90.0]
ZETA = 1.0
GRIP_RATE_M_S = 0.10
STIFFNESS_RANGE = (0.2, 1.0)


def quat_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


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
        w, v = np.linalg.eigh(Re)
        axis = v[:, int(np.argmin(np.abs(w - 1.0)))]
        return axis / np.linalg.norm(axis) * ang
    return axis / n * ang


class URController(object):
    def __init__(self, model, data, tick_s=0.01, prefix="ur/", tcp_site="ur/tcp", omega=None, torque_every_step=True):
        import mujoco
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.tick_s = float(tick_s)
        self.prefix = prefix
        self.joints = [model.joint(prefix + j).id for j in JOINT_NAMES]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joints])
        self.vadr = np.array([model.jnt_dofadr[j] for j in self.joints])
        self.acts = np.array([model.actuator(prefix + a).id for a in ACTUATOR_NAMES])
        self.grip_act = model.actuator(prefix + "g/fingers_actuator").id
        self.pads = g2f.PadGeometry(model, prefix)
        self.gripper_qadr = list(self.pads.linkage_qadr)
        self.tcp = model.site(tcp_site).id
        self.lower = np.array([model.jnt_range[j][0] for j in self.joints]) + RANGE_MARGIN
        self.upper = np.array([model.jnt_range[j][1] for j in self.joints]) - RANGE_MARGIN
        self.tau_limit = np.array([model.actuator_ctrlrange[a][1] for a in self.acts])
        try:
            spd = np.array(model.numeric(prefix + "joint_speed_limit_rad_s").data, float)
        except KeyError:
            spd = np.full(6, math.pi)
        self.speed_limit = spd
        self.rate_limit = np.minimum(RATE_LIMIT_RAD_S, spd)
        self.omega = np.array(OMEGA_RAD_S if omega is None else omega, float)
        self.kp0 = self.omega ** 2
        self.kd0 = 2.0 * ZETA * self.omega
        self.max_width = g2f.MAX_WIDTH_M
        self.torque_every_step = bool(torque_every_step)
        self.n_sub = max(1, int(round(self.tick_s / model.opt.timestep)))
        self.scratch = mujoco.MjData(model)
        self._afull = np.zeros(model.nv)
        self._Ma = np.zeros(model.nv)
        self._views(data)
        self.mode = "joint"
        self.stiffness_scale = 1.0
        self.q_cmd = data.qpos[self.qadr].copy()
        self.q_goal = self.q_cmd.copy()
        self.q_prev = self.q_cmd.copy()
        self._seg = np.zeros(6)
        self.v_ref = np.zeros(6)
        self._sub = self.n_sub
        self.T_goal = None
        self.width_cmd = float(min(self.max_width, max(0.0, self.width())))
        self.width_goal = self.width_cmd
        self.last_tau = [0.0] * 6        # the torques written last (N m) and the PD scale used (1 = no saturation)
        self.last_scale = 1.0
        self.set_stiffness(1.0)
        self.write_ctrl()

    def _views(self, d):
        """numpy views of the arm's entries in data (the per-step torque runs 1000 times per simulated second:
        contiguous slices instead of fancy indexing; the six joints and motors are contiguous after add_ur)"""
        def sl(idx):
            idx = np.asarray(idx, int)
            return slice(int(idx[0]), int(idx[0]) + 6) if np.all(np.diff(idx) == 1) else idx
        self._qs, self._vs, self._as = sl(self.qadr), sl(self.vadr), sl(self.acts)
        self._contig = all(isinstance(x, slice) for x in (self._qs, self._vs, self._as))
        if self._contig:
            self._qpos, self._qvel = d.qpos[self._qs], d.qvel[self._vs]
            self._bias, self._passive = d.qfrc_bias[self._vs], d.qfrc_passive[self._vs]
            self._ctrl = d.ctrl[self._as]
        self._lim = [float(v) for v in self.tau_limit]

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

    def cartesian_stiffness(self, q=None, scale=None):
        """6x6 stiffness of the tcp (N/m, N m/rad; force / moment at the tcp, world axes) of the joint loop at q
        (default: the actual configuration) and the stiffness scale (default: the current one)"""
        m = self.model
        s = self.scratch
        s.qpos[:] = self.data.qpos
        if q is not None:
            s.qpos[self.qadr] = q
        self.mujoco.mj_forward(m, s)
        M = np.zeros((m.nv, m.nv))
        self.mujoco.mj_fullM(m, s, M)
        Ma = M[np.ix_(self.vadr, self.vadr)]
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        self.mujoco.mj_jacSite(m, s, jp, jr, self.tcp)
        J = np.vstack([jp[:, self.vadr], jr[:, self.vadr]])
        sc = self.stiffness_scale if scale is None else float(scale)
        C = J @ np.linalg.solve(Ma @ np.diag(sc * self.kp0), J.T)
        return np.linalg.inv(C)

    # ---------------------------------------------------------------- API
    def set_stiffness(self, scale):
        scale = float(min(STIFFNESS_RANGE[1], max(STIFFNESS_RANGE[0], scale)))
        self.stiffness_scale = scale
        self.kp = scale * self.kp0
        self.kd = math.sqrt(scale) * self.kd0

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
        q = np.asarray(q, float).reshape(6)
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
        meas = float(min(self.max_width, max(0.0, self.width())))
        # opening past where the pads are (a part holds them open): start the ramp at the pads, so the release is
        # immediate; closing / re-squeezing: keep the command where it is (no momentary drop of the grip)
        if self.width_goal > meas:
            self.width_cmd = max(self.width_cmd, meas)
        else:
            self.width_cmd = min(self.width_cmd, meas)

    # ---------------------------------------------------------------- per control tick
    def tick(self):
        q_before = self.q_cmd.copy()
        dq_max = self.rate_limit * self.tick_s
        if self.mode == "joint":
            step = np.clip(self.q_goal - self.q_cmd, -dq_max, dq_max)
            self.q_cmd = np.clip(self.q_cmd + step, self.lower, self.upper)
        else:
            p, R = self.fk_cmd()
            pg, Rg = self.T_goal
            v = (pg - p) / CART_TAU
            nv = np.linalg.norm(v)
            if nv > CART_V_MAX:
                v *= CART_V_MAX / nv
            w = orientation_error(Rg, R) / CART_TAU
            nw = np.linalg.norm(w)
            if nw > CART_W_MAX:
                w *= CART_W_MAX / nw
            twist = np.concatenate([v, w]) * self.tick_s
            if np.linalg.norm(twist) > 1e-9:
                jp, jr = self._jacobian_cmd()
                J = np.vstack([jp, jr])
                at_lo = self.q_cmd <= self.lower + 1e-9
                at_hi = self.q_cmd >= self.upper - 1e-9
                dq = J.T @ np.linalg.solve(J @ J.T + np.eye(6) * DLS_LAMBDA ** 2, twist)
                dq[(at_lo & (dq < 0)) | (at_hi & (dq > 0))] = 0.0
                r = np.max(np.abs(dq) / dq_max)
                if r > 1.0:
                    dq /= r
                self.q_cmd = np.clip(self.q_cmd + dq, self.lower, self.upper)
        self.q_prev = q_before
        self._seg = self.q_cmd - q_before
        self.v_ref = self._seg / self.tick_s
        self._sub = 0
        dw = GRIP_RATE_M_S * self.tick_s
        self.width_cmd += float(np.clip(self.width_goal - self.width_cmd, -dw, dw))
        self.data.ctrl[self.grip_act] = g2f.ctrl_of_width(self.width_cmd)
        if not self.torque_every_step:
            self.torque(hold=True)

    def write_ctrl(self):
        """(re)start holding the current q_cmd / width_cmd (after a caller set them): a zero-velocity reference at
        q_cmd, the gripper ctrl from width_cmd, and a torque for the present state (valid after mj_forward)"""
        self.q_prev = self.q_cmd.copy()
        self._seg = np.zeros(6)
        self.v_ref = np.zeros(6)
        self._sub = self.n_sub
        self.data.ctrl[self.grip_act] = g2f.ctrl_of_width(self.width_cmd)
        self.torque(hold=True)

    # ---------------------------------------------------------------- per physics step
    def torque(self, hold=False):
        """write the arm's joint torques for the present state; call after mj_step1 (or mj_forward), before
        mj_step2. Each call counts one physics step of the current tick (hold=True: evaluate without counting)."""
        m, d = self.model, self.data
        k, n = self._sub, self.n_sub
        if not hold:
            self._sub = k + 1
        if self._contig:
            qpos, qvel, bias, passive = self._qpos, self._qvel, self._bias, self._passive
        else:
            qpos, qvel = d.qpos[self.qadr], d.qvel[self.vadr]
            bias, passive = d.qfrc_bias[self.vadr], d.qfrc_passive[self.vadr]
        # the state is k physics steps into the tick: the reference is k/n along the tick's segment, moving at
        # v_ref; past the segment's end (more steps than a tick without a new tick) it holds q_cmd at rest
        if k < n:
            a = self.kp * (self.q_prev + (k / float(n)) * self._seg - qpos) + self.kd * (self.v_ref - qvel)
        else:
            a = self.kp * (self.q_cmd - qpos) - self.kd * qvel
        self._afull[self._vs] = a
        self.mujoco.mj_mulM(m, d, self._Ma, self._afull)
        u = self._Ma[self._vs].tolist()
        b = (bias - passive).tolist()
        # the PD part scaled uniformly so that every joint stays inside its rating (the path is kept), then clipped
        sc = 1.0
        for ui, bi, li in zip(u, b, self._lim):
            if ui > li - bi:
                sc = min(sc, (li - bi) / ui)
            elif ui < -li - bi:
                sc = min(sc, (-li - bi) / ui)
        sc = max(0.0, sc)
        tau = [max(-li, min(li, bi + sc * ui)) for ui, bi, li in zip(u, b, self._lim)]
        if self._contig:
            self._ctrl[:] = tau
        else:
            d.ctrl[self.acts] = tau
        self.last_tau = tau
        self.last_scale = sc
        return tau

    def step_physics(self, n=None):
        """n physics steps (default: one tick's) with the torque recomputed before each; the caller runs
        mj_forward afterwards (as the cells do) for positions, sensors and contacts of the new state"""
        m, d = self.model, self.data
        for _ in range(self.n_sub if n is None else int(n)):
            self.mujoco.mj_step1(m, d)
            if self.torque_every_step:
                self.torque()
            self.mujoco.mj_step2(m, d)

    # ---------------------------------------------------------------- state
    def width(self):
        return self.pads.gap(self.data)

    def q(self):
        return self.data.qpos[self.qadr].copy()

    def dq(self):
        return self.data.qvel[self.vadr].copy()

    def cartesian_residual(self):
        if self.T_goal is None:
            return None
        p, R = self.tcp_actual()
        return float(np.linalg.norm(self.T_goal[0] - p)), float(np.linalg.norm(orientation_error(self.T_goal[1], R)))
