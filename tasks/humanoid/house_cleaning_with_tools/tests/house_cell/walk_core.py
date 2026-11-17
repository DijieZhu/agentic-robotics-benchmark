"""Walking legs for the furniture cell's G1 robots, drawn only: an online, causal leg animator.

The cell's locomotion abstraction carries each robot's pelvis on four base joints (base_x, base_y, base_yaw, base_z)
and poses its legs from the stand height, so a robot that travels glides over the floor. The cameras draw the legs
walking instead: one `LegWalker` per robot, stepped at the cell's control rate with the simulated time and the robot's
base pose (the four base joints), keeps its own footsteps and returns the 12 leg joint angles to draw. Nothing else
changes: the pelvis, the torso, the arms and the head (with the head camera) stay where the physics has them, and the
physics keeps its posed legs.

The gait (root motion: the planted feet stay put, the pelvis moves as simulated):
* the base's velocity, yaw rate and their rates of change are estimated from its poses (causal, smoothed over 0.08 /
  0.12 s), and the base is predicted ahead from them: a gaining speed for a short while, a falling one as a decay to a
  stop (a slowing turn to about its end); the base counts as moving once it travels 5 cm/s or turns 0.15 rad/s;
* the planted legs take up what the base does within their reach: a foot steps when its spot under the hip, at the
  base predicted half a stance after the landing, is farther than a hold distance (a share of the stance leg's
  flat-foot reach at the stand height: 3.5 cm for the nearly straight legs at 0.79 m, up to 10 cm on bent knees;
  0.35 rad of yaw). So a robot working in place keeps its feet planted through small base corrections and the
  controller's last-millimetre goal snap. Walking on, the feet step alternately while they are 5 cm / 0.25 rad from
  those spots; a foot the base has left 15 cm / 0.6 rad from its spot, now and as predicted, steps at once (a foot
  that has just landed where the base is going is not left behind, and in a walk the other foot goes first when it
  needs a step too); still for 1.5 s, a foot beyond 6 cm / 0.3 rad takes one slow settle step (a 0.7 s swing);
* cadence 1.1 steps/s from rest, 1.55 at 0.3 m/s, 1.85 at 0.5 m/s, up to 1.9 turning on the spot and 2.6 side-stepping;
  the swing 78 % of a step (0.38-0.55 s, side-steps from 0.30 s). A landing lands where its spot under the hip is
  predicted half a stance later (the hip passes over the foot at mid-stance), within reach, its yaw within 0.5 rad of
  the pelvis, 0.9 of the other foot and 0.85 of its own lift-off, 0.17 m across and 2.5 cm between the soles from the
  other foot, fully on or off a drawn support such as the rug (the sole as a rectangle; moved inwards rather than
  outwards); a swing is planned at most 0.42 m long; its landing follows the base until 40 % of the swing, fading out
  by 75 % (which lengthens a swing that starts while the base speeds up, to about 0.6 m at most);
* a swing foot follows a lifted arc: minimum jerk along the ground, a lift of 2-5 cm by the step's length (less for a
  side-step) rising smoothly to its top at 42 % of the swing; a step onto a support reaches its height by the time the
  sole would reach the support, a step off it keeps that height until the sole has passed its edge;
* the feet roll heel to toe: a foot about to lift raises its heel from the other foot's mid-swing on (its toe-off, up
  to 0.25 rad for a foot well behind its hip; a foot not behind its hip keeps its edge); a lifted foot unrolls over
  the first 45 % of the swing and builds up its landing roll from 55 % on: a heel strike up to 0.2 rad for a landing
  ahead of the hip (along the foot), more where the leg would land with its knee bent less than 0.2 rad, at most 0.35;
  a landing behind the hip (a toe strike) or beside it (on the ball of the foot) up to 0.08 rad, more only where the
  knee would be bent less than 0.12 rad. A landed foot rolls flat over 0.25 s (after a toe strike or a side landing
  0.4 s); once its hip has passed over it, it unrolls over 0.12 s and changes its edge. Legs only: the pelvis keeps
  its height, and the cell's robots stand with nearly straight legs, so a planted foot that its leg cannot reach flat
  also rolls over its heel (the foot ahead of the hip) or its toe (behind), changing that edge only once flat (the
  roll over the new edge eases in over 0.15 s); a pelvis above the straight-leg height leaves the feet hovering as the
  posed legs do;
* each leg's 6 joints and its foot's roll by damped least squares on the robot's own leg chain (copied from the cell
  model at construction; Levenberg-Marquardt damping for a target out of reach), within the joint limits, with a soft
  floor that keeps a planted foot's knee bent at least 0.12 rad once the foot is off its spot under the hip (a
  straight knee is singular: lifting its foot would snap it; the foot's roll takes up the leg's length instead): one
  tracking step per tick from the previous tick's solution, each leg on its own (a rolled foot gets at least one, so
  that its roll follows a bound easing off; an idle robot's legs are solved again only while a roll still changes). A
  swing leg's drawn joints follow the solution within 9 rad/s and 300 rad/s^2, handed back to it over the last 20 % of
  the swing (the foot lands exactly).

`step` is meant to be called every control tick whatever the cameras do (the legs then depend on the motion only);
`pose_at` gives the legs to draw at a time between two steps without changing anything. Deterministic (the same base
poses at the same times give the same angles), numpy + mujoco only, no file I/O.
"""

from __future__ import annotations

import bisect
import collections
import math

import mujoco
import numpy as np

SIDES = ("left", "right")
LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
BASE_JOINTS = ("base_x", "base_y", "base_yaw", "base_z")
HEEL, TOE = 1, -1                  # a foot's pivot (the edge it rolls over when it is not flat)


class WalkParams:
    """The gait's numbers (SI units: m, s, rad)."""

    tick = 0.01                    # the planner's longest internal step (base poses between calls are interpolated)
    gap_reset = 1.0                # a longer gap between calls, time running backwards or a jump of the base ...
    jump_xy, jump_yaw, jump_z = 0.25, 0.6, 0.08     # ... beyond these in one call re-plants the feet (a reset)
    # the base's motion: velocity / yaw rate and their rates of change, smoothed; predicted ahead, a gaining speed
    # extended over at most acc_h (up to v_cap, w_cap), a falling one as a decay to a stop (time constant speed /
    # deceleration within tau_stop_min .. tau_stop_max; a turn slowing with a time constant up to stop_tau is looked
    # yaw_stop time constants ahead, about to its end); the prediction clipped to pred_max, pred_yaw_max
    tau_v, tau_a = 0.08, 0.12
    acc_h, v_cap, w_cap = 0.30, 0.50, 0.80
    tau_stop_min, tau_stop_max = 0.25, 3.0
    acc_eps, accw_eps = 0.05, 0.10                  # a rate of change below these counts as none (m/s^2, rad/s^2)
    yaw_stop, stop_tau = 2.5, 1.5
    pred_max, pred_yaw_max = 0.35, 0.90
    v_on, w_on = 0.05, 0.15        # the base counts as moving (it stops counting below half of these) ...
    motion_window = 0.25           # ... by its travel over this long
    # when a planted foot steps: its deviation from its spot under the hip at the base predicted half a stance after
    # the landing (position, yaw) beyond ...
    d_walk, yaw_walk = 0.05, 0.25  # ... in a bout of steps (a walking step within restart_s)
    d_hold_k, d_hold_min, d_hold_max = 0.6, 0.035, 0.10    # ... to start one or standing: this share of the stance
    yaw_hold = 0.35                # leg's flat-foot reach at the stand height (the planted legs absorb the rest)
    d_settle, yaw_settle, settle_wait = 0.06, 0.30, 1.5    # still this long: one settle step per foot beyond these,
    tsw_settle = 0.70                                      # a slow one (its swing at least this long)
    d_strain, yaw_strain = 0.15, 0.60   # a planted foot this far from its spot, now and as predicted, steps at once
    restart_s = 1.2
    step_least = 0.8               # a step shorter than this share of its threshold is not taken
    # cadence (steps/s): from the speed (f_min at rest, f_cruise at v_ref, on to f_fast at v_fast; side-stepping
    # f_lat at v_ref) and from the yaw rate (f_turn at w_ref), the larger of the two
    f_min, f_cruise, v_ref, f_fast, v_fast = 1.1, 1.55, 0.30, 1.85, 0.50
    f_lat = 2.6
    f_turn, w_ref = 1.9, 0.80
    sw_frac = 0.78                 # swing share of a step period ...
    tsw_min, tsw_max, tsw_lat = 0.38, 0.55, 0.30    # ... within these (side-stepping: from tsw_lat)
    tds_min = 0.10                 # least double support
    # a landing: within reach of its hip's spot, its yaw within reach of the pelvis, the other foot and its own
    # lift-off, w_min across (pelvis frame) and c_min between the soles from the other foot; a swing planned at most
    # step_max long (the landing's following of the base can lengthen it)
    reach_fwd, reach_back = 0.17, 0.17
    reach_in, reach_out = 0.10, 0.10
    yaw_rel_max, yaw_gap_max, yaw_step_max = 0.5, 0.9, 0.85
    w_min, c_min = 0.17, 0.025
    step_max = 0.42
    widen_max = 0.03               # side-stepping widens the stance by up to this per foot
    rug_along, rug_across = 0.05, 0.08              # a landing straddling a support's edge moves this far at most
    #                                (along counting double, outwards triple)
    u_rt0, u_rt1 = 0.40, 0.75      # a swing's landing follows the base, the following fading out between these ...
    retarget_v, retarget_w, retarget_z = 0.6, 1.5, 0.2      # ... moving at most this fast (m/s, rad/s, m/s)
    # the swing foot's arc: minimum jerk along the ground; a lift of h_base + h_gain * length + h_yaw * yaw (times
    # h_lat for a side-step) within h_min .. h_max, rising as u^lift_m to its top at lift_peak of the swing
    h_min, h_max, h_base, h_gain, h_yaw = 0.020, 0.050, 0.010, 0.12, 0.025
    h_lat = 0.7
    lift_peak, lift_m = 0.42, 2.5
    # the feet roll heel to toe (theta: a roll over the heel or the toe edge):
    to_theta, to_x0, to_x1 = 0.25, 0.04, 0.14       # the toe-off: a foot about to lift raises its heel up to to_theta
    #                                (by how far behind its hip it will lift, to_x0 .. to_x1), from to_from of the
    #                                other foot's swing to its expected lift-off; held to_hold longer, then let down
    #                                over to_fall if no step follows
    to_from, to_hold, to_fall = 0.5, 0.10, 0.25
    unroll = 0.45                  # a lifted foot unrolls its lift-off roll over this share of the swing ...
    hs_from = 0.55                 # ... and builds up its landing roll from this share on:
    hs_theta, hs_x0, hs_x1 = 0.20, 0.03, 0.12       # a heel strike up to hs_theta for a landing ahead of its hip
    #                                along the foot (by how far, hs_x0 .. hs_x1), more where the leg would land with
    #                                its knee bent less than land_knee, at most land_roll_max (rate-limited while the
    #                                landing follows the base)
    land_knee = 0.20
    land_roll_max, land_roll_rate = 0.35, 1.5
    ts_theta, ts_knee = 0.08, 0.12     # a toe strike (a landing behind its hip) and a side landing: their roll up to
    #                                ts_theta, more only where the leg would land with its knee bent less than ts_knee
    t_load, t_load_ts = 0.25, 0.40     # a landed foot eases its landing roll off over this long (after a toe strike
    #                                or a side landing: t_load_ts), and at the latest once its hip has passed over it
    t_flip, flip_theta = 0.12, 0.004    # a planted foot changes the edge it rolls over only once unrolled (over t_flip)
    t_flip_in = 0.15               # ... and its roll over the new edge eases in over this long
    pivot_band = 0.005             # hysteresis of the heel / toe choice (foot position against the hip)
    theta_max = 0.6                # largest roll of a foot over its heel or toe
    # the legs' IK (damped least squares on the robot's own leg chain)
    knee_min = 0.05                # least knee bend of the drawn legs
    knee_soft, knee_band, w_knee = 0.12, 0.04, 0.05   # a soft floor (easing in over knee_band, IK weight m per rad)
    #                                keeping a planted foot's knee bent: the foot's roll takes up the leg's length
    #                                instead (a straight knee is singular: lifting its foot would snap it) ...
    soft_d0, soft_d1 = 0.015, 0.055     # ... easing in as the foot's distance from its spot under the hip grows
    knee_reach = 0.10              # knee bend of the reach table (the flat-foot reach per stand height)
    w_rot, w_theta = 0.2, 0.003    # IK weights: rotation rows (m per rad), the roll drawn to 0
    ik_iters, ik_iters_eval = 4, 8                  # residual checks at most per tick / per pose_at and reset
    ik_tol, ik_tol_eval = 2e-5, 1e-7                # m: residual counted as converged per tick / per pose_at
    ik_track = 2e-2                # m: below this residual one step per tick tracks the moving targets
    ik_step = 0.4                  # largest joint change per IK iteration
    ik_lm = 0.05                   # IK damping per squared residual (m^2)
    roll_slack = 1e-3              # a foot rolled beyond its roll's lower bound by more than this steps the IK once a
    #                                tick even with no residual (so its roll follows a bound easing off)
    lim_v, lim_a, lim_until = 9.0, 300.0, 0.80      # a swing leg's drawn joints follow the IK's within lim_v rad/s
    #                                and lim_a rad/s^2, handed back to it from lim_until of the swing on
    still_tol = 1e-5               # base change below which an idle robot's legs are not solved again


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _minjerk(u: float) -> float:
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


def _ease(u: float) -> float:
    """Minimum-jerk progress of u clipped to [0, 1]."""
    return _minjerk(min(1.0, max(0.0, u)))


def _hull(points) -> np.ndarray:
    """Convex hull (counter-clockwise) of 2D points."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) < 3:
        return np.array(pts, float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1] + hi[:-1], float)


def support_from_geom(m: mujoco.MjModel, geom: int):
    """A flat support (its top's outline as a convex polygon, counter-clockwise, and its height) from a box or mesh geom
    of a static body: e.g. a drawn rug, so that feet stand on it rather than in it. The geom's pose is the model's
    default (qpos0) kinematics."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    R = d.geom_xmat[geom].reshape(3, 3)
    pos = d.geom_xpos[geom]
    if m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[geom])
        a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        V = m.mesh_vert[a:a + n].astype(np.float64) @ R.T + pos
    elif m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = m.geom_size[geom]
        V = np.array([(i * sx, j * sy, k * sz) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) @ R.T + pos
    else:
        raise ValueError("a support must be a box or a mesh geom")
    top = float(V[:, 2].max())
    upper = V[V[:, 2] > top - 0.5 * (top - float(V[:, 2].min())) - 1e-9]
    return _hull(upper[:, :2]), top


def _leg_model(m: mujoco.MjModel, prefix: str) -> mujoco.MjModel:
    """The robot's two leg chains in its pelvis frame (the pelvis is this model's world): the cell model's own body
    frames, joint axes and ranges."""
    B = mujoco.mjtObj.mjOBJ_BODY
    s = mujoco.MjSpec()
    s.modelname = "legs"
    s.compiler.degree = False                           # the ranges below are the compiled model's radians
    s.option.gravity = [0.0, 0.0, 0.0]
    for side in SIDES:
        parent = s.worldbody
        for w in LEG_JOINTS:
            bid = mujoco.mj_name2id(m, B, f"{prefix}{side}_{w}_link")
            if bid < 0:
                raise KeyError(f"{prefix}{side}_{w}_link")
            b = parent.add_body(name=f"{side}_{w}_link", pos=m.body_pos[bid].tolist(), quat=m.body_quat[bid].tolist())
            b.explicitinertial = True
            b.mass = 1.0
            b.inertia = [1e-3, 1e-3, 1e-3]
            jid = int(m.body_jntadr[bid])
            b.add_joint(name=f"{side}_{w}_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=m.jnt_axis[jid].tolist(),
                        pos=m.jnt_pos[jid].tolist(), range=m.jnt_range[jid].tolist(),
                        limited=mujoco.mjtLimited.mjLIMITED_TRUE)
            parent = b
    return s.compile()


class _Supports:
    """Support height under a sole: the floor (z = 0) or the top of a flat support (convex polygon, height)."""

    def __init__(self, supports):
        self.polys = []
        for poly, top in supports or ():
            P = np.asarray(poly, np.float64).reshape(-1, 2)
            if len(P) >= 3:
                # edge normals (outward) and offsets of the counter-clockwise polygon
                E = np.roll(P, -1, axis=0) - P
                N = np.stack([E[:, 1], -E[:, 0]], axis=1)
                N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
                self.polys.append((N, np.sum(N * P, axis=1), float(top), P))
        self.boxes = [(float(V[:, 0].min()), float(V[:, 0].max()), float(V[:, 1].min()), float(V[:, 1].max()))
                      for _, _, _, V in self.polys]

    def top_of(self, pts: np.ndarray) -> np.ndarray:
        """Support height at each 2D point."""
        z = np.zeros(len(pts))
        for N, c, top, _ in self.polys:
            inside = np.all(pts @ N.T <= c + 1e-9, axis=1)
            z = np.where(inside, np.maximum(z, top), z)
        return z

    def rest_many(self, C: np.ndarray):
        """For sole rectangles C (n, 4, 2: corners in order): the height a rigid flat sole rests at (the highest
        support it overlaps, else the floor), and whether it lies on one support only (each support holds either the
        whole sole or none of it). Exact: separating axes of the rectangle and the convex polygon."""
        n = C.shape[0]
        z = np.zeros(n)
        uni = np.ones(n, bool)
        if not self.polys:
            return z, uni
        ax = np.stack([C[:, 1] - C[:, 0], C[:, 3] - C[:, 0]], axis=1)
        ax /= np.maximum(np.linalg.norm(ax, axis=2, keepdims=True), 1e-12)
        pr = np.einsum("nak,nck->nac", ax, C)
        for N, c, top, V in self.polys:
            proj = C @ N.T                                          # (n, 4 corners, edges)
            inside = np.all(proj <= c + 1e-9, axis=(1, 2))
            sep = np.any(np.all(proj > c + 1e-9, axis=1), axis=1)  # an edge of the polygon separates
            pv = np.einsum("nak,vk->nav", ax, V)
            sep |= np.any((pv.min(axis=2) > pr.max(axis=2) + 1e-9) | (pv.max(axis=2) < pr.min(axis=2) - 1e-9), axis=1)
            z = np.where(~sep, np.maximum(z, top), z)
            uni &= inside | sep
        return z, uni

    def rest(self, C):
        """rest_many for one sole rectangle (corners as 4 (x, y) pairs)."""
        xs = [c[0] for c in C]
        ys = [c[1] for c in C]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if all(x1 < b[0] or x0 > b[1] or y1 < b[2] or y0 > b[3] for b in self.boxes):
            return 0.0, True                            # clear of every support (the usual case)
        z, u = self.rest_many(np.asarray(C, np.float64)[None])
        return float(z[0]), bool(u[0])


class LegWalker:
    """One robot's walking legs (drawn only). `step(t, base)` every control tick (base = the robot's base_x, base_y,
    base_yaw, base_z joint values at sim time t) returns the 12 leg joint angles to draw: left then right, each
    (hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll). `qpos_adr` are those joints' qpos addresses in the
    cell model, `base_adr` the base joints'."""

    def __init__(self, m: mujoco.MjModel, prefix: str, *, supports=(), params: WalkParams | None = None):
        P = self.P = params or WalkParams()
        J, B = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY
        self.prefix = prefix
        self.qpos_adr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, J, f"{prefix}{s}_{w}_joint")])
                                  for s in SIDES for w in LEG_JOINTS])
        self.base_adr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, J, prefix + n)]) for n in BASE_JOINTS])
        pelvis = mujoco.mj_name2id(m, B, prefix + "pelvis")
        if pelvis < 0:
            raise KeyError(prefix + "pelvis")
        self.z0 = float(m.body_pos[pelvis][2])          # pelvis height at base_z = 0 (world-aligned base joints)
        self.lm = _leg_model(m, prefix)
        self.ld = mujoco.MjData(self.lm)
        lm = self.lm
        self.foot_body = [mujoco.mj_name2id(lm, B, f"{s}_ankle_roll_link") for s in SIDES]
        hips = [mujoco.mj_name2id(lm, B, f"{s}_hip_pitch_link") for s in SIDES]
        self.hip_xy = [(float(lm.body_pos[b][0]), float(lm.body_pos[b][1])) for b in hips]   # base frame
        lo = lm.jnt_range[:, 0] + 1e-4
        hi = lm.jnt_range[:, 1] - 1e-4
        lo[[3, 9]] = np.maximum(lo[[3, 9]], P.knee_min)
        self.lo, self.hi = lo, hi
        self.supports = _Supports(supports)
        da, dc = np.meshgrid(np.arange(-P.rug_along, P.rug_along + 1e-9, 0.01),
                             np.arange(-P.rug_across, P.rug_across + 1e-9, 0.01), indexing="ij")
        cost = 2.0 * np.abs(da) + np.abs(dc)
        order = np.argsort(cost.ravel(), kind="stable")
        order = order[cost.ravel()[order] > 0.0]
        self._shift_grid = (da.ravel()[order], dc.ravel()[order], cost.ravel()[order])
        self._lift_n = P.lift_m * (1.0 - P.lift_peak) / P.lift_peak
        self._lift_k = 1.0 / (P.lift_peak ** P.lift_m * (1.0 - P.lift_peak) ** self._lift_n)
        self._sole(m, prefix)
        self._stance_table()
        self.jacp = np.zeros((2, 3, lm.nv))
        self.jacr = np.zeros((2, 3, lm.nv))
        self._J7 = np.zeros((2, 8, 7))
        self._lo7 = np.concatenate([self.lo.reshape(2, 6), np.zeros((2, 1))], axis=1)
        self._hi7 = np.concatenate([self.hi.reshape(2, 6), np.full((2, 1), P.theta_max)], axis=1)
        self._eye7 = np.broadcast_to(np.eye(7), (2, 7, 7)).copy()
        self._eye3 = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
        self._e7 = np.zeros((2, 8))
        self._J7[:, 6, 6] = P.w_theta
        self._fixed = np.zeros((2, 7))
        self._X = np.zeros((2, 7))
        self.theta = np.zeros(2)
        self.t = None
        self.reset_count = 0

    # ------------------------------------------------------------------------------------------- the model's numbers
    def _sole(self, m, prefix):
        """The sole in the ankle_roll_link frame, from the foot's drawn mesh (its lowest 1 mm): heel and toe x,
        half width (the sole as a rectangle)."""
        B = mujoco.mjtObj.mjOBJ_BODY
        b = mujoco.mj_name2id(m, B, f"{prefix}left_ankle_roll_link")
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        if geoms:
            g = geoms[0]
            mid = int(m.geom_dataid[g])
            V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            V = V @ R.reshape(3, 3).T + m.geom_pos[g]
            band = V[V[:, 2] < V[:, 2].min() + 0.001]
            self.x_heel, self.x_toe = float(band[:, 0].min()), float(band[:, 0].max())
            self.half_w = float(0.5 * (band[:, 1].max() - band[:, 1].min()))
        else:                                           # Menagerie's G1 foot
            self.x_heel, self.x_toe, self.half_w = -0.0658, 0.1424, 0.0378
        self._rect = ((self.x_heel, -self.half_w), (self.x_toe, -self.half_w), (self.x_toe, self.half_w),
                      (self.x_heel, self.half_w))

    def _fk_ankles(self, q6: np.ndarray):
        """Ankle-roll frames of both legs (pelvis frame) for the same 6 angles on both legs (a symmetric pose)."""
        ld = self.ld
        ld.qpos[:6] = q6
        ld.qpos[6:] = q6
        mujoco.mj_kinematics(self.lm, ld)
        return [ld.xpos[b].copy() for b in self.foot_body], [ld.xmat[b].reshape(3, 3).copy() for b in self.foot_body]

    def _stance_table(self):
        """The stance under the hips per stand height (the cell's posed squat: hip pitch -a, knee 2a, ankle pitch -a):
        the ankle-roll origin in the base frame, and the angles. Beyond the ankle's dorsiflexion limit a flat foot sits
        further forward (ankle at its limit, hip pitch th, knee lim - th). Above the straight-leg height the posed feet
        hover. Also per height: the flat-foot reach of a stance leg (how far its ankle can be from under the hip,
        the knee bent knee_reach), which sets how much base motion the planted feet absorb."""
        P = self.P
        p0, _ = self._fk_ankles(np.zeros(6))
        self.ankle_h0 = self.z0 + float(p0[0][2])      # ankle-roll origin above the support in the posed stance
        lim = -float(self.lo[4]) - 0.01                 # ankle dorsiflexion used at most in the stance
        a_min = 0.5 * P.knee_min
        # the sagittal hip-pitch to ankle-pitch length with the knee bent knee_reach, and the ankle pitch axis' height
        # above the sole of a flat foot
        B = mujoco.mjtObj.mjOBJ_BODY
        hp = mujoco.mj_name2id(self.lm, B, "left_hip_pitch_link")
        ap = mujoco.mj_name2id(self.lm, B, "left_ankle_pitch_link")
        self._fk_ankles(np.zeros(6))
        z_ap = float(self.ld.xpos[ap][2] - self.ld.xpos[self.foot_body[0]][2]) + self.ankle_h0
        z_hp = float(self.ld.xpos[hp][2])
        self._fk_ankles(np.array([0.0, 0.0, 0.0, P.knee_reach, 0.0, 0.0]))
        dv = self.ld.xpos[ap] - self.ld.xpos[hp]
        L = math.hypot(float(dv[0]), float(dv[2]))
        self._hip_off = [self.ld.xpos[hp].copy() * np.array([1.0, sg, 1.0]) for sg in (1.0, -1.0)]
        self._ap_up = float(self.ld.xpos[ap][2] - self.ld.xpos[self.foot_body[0]][2])
        self._leg_k = np.linspace(0.0, 1.6, 33)
        self._leg_L = []
        for kk in self._leg_k:
            self._fk_ankles(np.array([0.0, 0.0, 0.0, kk, 0.0, 0.0]))
            self._leg_L.append(float(np.linalg.norm(self.ld.xpos[ap] - self.ld.xpos[hp])))
        self._leg_land = float(np.interp(P.land_knee, self._leg_k, self._leg_L))
        self._leg_ts = float(np.interp(P.ts_knee, self._leg_k, self._leg_L))
        rows = []
        for a in np.concatenate([[a_min], np.linspace(0.04, 1.25, 122)]):
            if a <= lim:
                q = np.array([-a, 0.0, 0.0, 2 * a, -a, 0.0])
            else:
                lo_, hi_ = -2.0, -lim                  # hip pitch th: deeper squat <-> more hip flexion
                pa, _ = self._fk_ankles(np.array([-a, 0.0, 0.0, 2 * a, -a, 0.0]))
                z_t = float(pa[0][2])
                for _ in range(50):
                    th = 0.5 * (lo_ + hi_)
                    pp, _ = self._fk_ankles(np.array([th, 0.0, 0.0, lim - th, -lim, 0.0]))
                    if pp[0][2] > z_t:
                        lo_ = th
                    else:
                        hi_ = th
                q = np.array([th, 0.0, 0.0, lim - th, -lim, 0.0])
            pa, _ = self._fk_ankles(q)
            h = self.ankle_h0 - float(pa[0][2])
            V = h + z_hp - z_ap                          # hip pitch axis above the ankle pitch axis
            reach = math.sqrt(max(0.0, L * L - V * V))
            rows.append((h, float(pa[0][0]), float(pa[0][1]), float(pa[1][1]), q, reach))
        rows.sort(key=lambda r: r[0])
        self.tab_h = [r[0] for r in rows]
        self.tab_ax = [r[1] for r in rows]
        self.tab_ay = {"left": [r[2] for r in rows], "right": [r[3] for r in rows]}
        self.tab_q = np.array([r[4] for r in rows])
        self.tab_reach = [r[5] for r in rows]
        self.h_top = self.tab_h[-1]                     # highest pelvis with the feet flat on the support
        # below this pelvis height (above a lifted foot) the posed squat's ankle passes its dorsiflexion limit
        self.h_squat = min(r[0] for r in rows if r[4][4] > -lim + 1e-9)

    def _interp(self, arr, h):
        hs = self.tab_h
        if h <= hs[0]:
            return arr[0]
        if h >= hs[-1]:
            return arr[-1]
        i = bisect.bisect_right(hs, h) - 1
        u = (h - hs[i]) / (hs[i + 1] - hs[i])
        return arr[i] + (arr[i + 1] - arr[i]) * u

    def home(self, side: str, h: float):
        """The stance spot of a foot under its hip (ankle-roll origin, base frame x, y) for pelvis height h above the
        support."""
        return self._interp(self.tab_ax, h), self._interp(self.tab_ay[side], h)

    def posed(self, h: float) -> np.ndarray:
        """The stance's 12 angles for pelvis height h (both legs alike)."""
        q6 = self._interp(self.tab_q, h)
        return np.concatenate([q6, q6])

    def hold_reach(self, h: float) -> float:
        """How far (m) a planted foot may be from its spot under the hip before it steps, standing or to start a bout:
        a share of the stance leg's flat-foot reach at pelvis height h (above the straight-leg height: as at it)."""
        P = self.P
        return min(P.d_hold_max, max(P.d_hold_min, P.d_hold_k * self._interp(self.tab_reach, min(h, self.h_top))))

    # ------------------------------------------------------------------------------------------- public
    def reset(self, t: float, base) -> np.ndarray:
        """Both feet planted in the stance under the hips (a start or a teleport)."""
        x, y, yaw, bz = (float(v) for v in base)
        h = self.z0 + bz
        self.t = float(t)
        self.base = (x, y, yaw, bz)
        self.v = [0.0, 0.0]
        self.a = [0.0, 0.0]
        self.w = 0.0
        self.aw = 0.0
        self.hist = collections.deque([(self.t, x, y, yaw)])
        self.moving = False
        self.still_since = self.t
        self.settle_due = {s: False for s in SIDES}
        self.swing = None
        self.last_land = self.t - 10.0
        self.last_side = None
        self.last_walk = False
        self.plant = {}
        for s in SIDES:
            self.plant[s] = self._settled(self._home_world(s, x, y, yaw, h), s)
        self.pivot = {s: HEEL for s in SIDES}
        self.load = {s: None for s in SIDES}
        self.pre = {s: None for s in SIDES}
        self.flip = {s: None for s in SIDES}
        self.flip_t = {s: None for s in SIDES}
        self.q = self.posed(h - self.plant["left"][2])
        self.q_lim, self.v_lim, self.lim_off = self.q.copy(), np.zeros(12), np.zeros(6)
        self.q_draw, self.t_solve = self.q.copy(), self.t
        self.solved_for = None
        self.converged = False
        self._fixed[:] = 0.0
        self.theta = np.zeros(2)
        self.reset_count += 1
        self._solve(self.t, self.base, self.P.ik_iters_eval, commit=True)
        self.q_lim, self.v_lim, self.q_draw = self.q.copy(), np.zeros(12), self.q.copy()
        return self.q_draw.copy()

    def step(self, t: float, base) -> np.ndarray:
        """Advance to sim time t with the base pose `base` (base_x, base_y, base_yaw, base_z); returns the 12 angles
        (one tracking step of the IK). A call with the time running backwards, after a gap of more than gap_reset or
        with the base jumped away re-plants the feet (a reset); base poses between calls are interpolated."""
        P = self.P
        t = float(t)
        x, y, yaw, bz = (float(v) for v in base)
        if not all(math.isfinite(v) for v in (t, x, y, yaw, bz)):
            raise ValueError("non-finite time or base pose")
        if self.t is None:
            return self.reset(t, (x, y, yaw, bz))
        dt = t - self.t
        x0, y0, yaw0, bz0 = self.base
        dyaw = _wrap(yaw - yaw0)
        if dt < -1e-9 or dt > P.gap_reset or math.hypot(x - x0, y - y0) > P.jump_xy + 0.6 * max(dt, 0.0) \
                or abs(dyaw) > P.jump_yaw + 1.0 * max(dt, 0.0) or abs(bz - bz0) > P.jump_z + 0.2 * max(dt, 0.0):
            return self.reset(t, (x, y, yaw, bz))
        if dt > 1e-12:
            t_prev = self.t
            n = max(1, math.ceil(dt / P.tick - 1e-9))
            for i in range(1, n + 1):
                u = i / n
                self._advance(t_prev + dt * u if i < n else t,
                              (x0 + (x - x0) * u, y0 + (y - y0) * u, yaw0 + dyaw * u, bz0 + (bz - bz0) * u), dt / n)
            self.t = t
        self.base = (x, y, yaw, bz)
        return self._solve(t, self.base, P.ik_iters, commit=True).copy()

    def pose_at(self, t: float, base) -> np.ndarray:
        """The 12 angles at a time between two steps (t not before the last step, base the pose then), with the
        footsteps as of the last step; nothing changes."""
        return self._solve(float(t), tuple(float(v) for v in base), self.P.ik_iters_eval, commit=False)

    def feet(self, t: float | None = None) -> dict:
        """Per side: whether the foot stands, and its target pose on the ground (x, y, support z, yaw)."""
        t = self.t if t is None else float(t)
        out = {}
        for s in SIDES:
            sw = self.swing
            if sw is not None and sw["side"] == s and t < sw["t1"]:
                pose, u = self._swing_pose(t)
                out[s] = {"stance": False, "pose": pose, "u": u}
            else:
                pose = sw["P1"] if sw is not None and sw["side"] == s else self.plant[s]   # landed by then
                out[s] = {"stance": True, "pose": tuple(pose), "u": -1.0}
        return out

    # ------------------------------------------------------------------------------------------- planner
    def _home_world(self, side, x, y, yaw, h, widen=0.0):
        ax, ay = self.home(side, h)
        ay += widen if side == "left" else -widen
        c, s = math.cos(yaw), math.sin(yaw)
        return [x + c * ax - s * ay, y + s * ax + c * ay, 0.0, yaw]

    def _corners(self, pose, side):
        """The sole rectangle's corners in the world (x, y) for a foot pose."""
        c, s = math.cos(pose[3]), math.sin(pose[3])
        sg = 1.0 if side == "left" else -1.0
        return [(pose[0] + c * x - s * y * sg, pose[1] + s * x + c * y * sg) for x, y in self._rect]

    def _settled(self, pose, side):
        """The pose with its support height."""
        z, _ = self.supports.rest(self._corners(pose, side))
        return [pose[0], pose[1], z, pose[3]]

    def _timing(self):
        """(swing, double support and stance times, side-stepping share) from the base's speed and yaw rate."""
        P = self.P
        yaw = self.base[2]
        vx, vy = self.v
        v = math.hypot(vx, vy)
        vlat = abs(-math.sin(yaw) * vx + math.cos(yaw) * vy)
        lat = min(1.0, vlat / (v + 0.02))
        fc = P.f_cruise + (P.f_lat - P.f_cruise) * lat
        if v <= P.v_ref:
            fv = P.f_min + (fc - P.f_min) * v / P.v_ref
        else:
            fv = fc + max(0.0, P.f_fast - fc) * min(1.0, (v - P.v_ref) / (P.v_fast - P.v_ref))
        fw = P.f_min + (P.f_turn - P.f_min) * min(1.0, abs(self.w) / P.w_ref)
        tstep = 1.0 / max(fv, fw)
        tsw = min(P.tsw_max, max(P.tsw_min + (P.tsw_lat - P.tsw_min) * lat, P.sw_frac * tstep))
        tds = max(P.tds_min, tstep - tsw)
        return tsw, tds, tsw + 2.0 * tds, lat

    def _ahead(self, s, acc, horizon, cap, eps, stop=0.0):
        """Travel over `horizon` s from speed s >= 0 changing at acc along it: a gaining speed is extended over at most
        acc_h (up to cap), a falling one decays to a stop with the time constant s / -acc (a clear stop, the time
        constant at most stop_tau, looked at least `stop` time constants ahead)."""
        P = self.P
        if acc < -eps and s > 1e-6:
            tau = min(P.tau_stop_max, max(P.tau_stop_min, s / -acc))
            h = max(horizon, stop * tau) if tau <= P.stop_tau else horizon
            return s * tau * (1.0 - math.exp(-h / tau))
        if acc > eps and s < cap:
            t1 = min(horizon, P.acc_h, (cap - s) / acc)
            return s * horizon + acc * t1 * (horizon - 0.5 * t1)
        return s * horizon

    def _predict(self, horizon):
        """The base pose `horizon` seconds ahead (clipped): its yaw from the yaw rate and its rate of change, its
        position along its velocity (from rest: along its acceleration), on an arc."""
        P = self.P
        x, y, yaw, bz = self.base
        H = max(0.0, horizon)
        w = self.w
        if abs(w) > 1e-6:                               # a turn slowing down: its end (the feet turn to it)
            sg = 1.0 if w > 0.0 else -1.0
            dyaw = sg * self._ahead(abs(w), sg * self.aw, H, P.w_cap, P.accw_eps, P.yaw_stop)
        else:
            sg = 1.0 if self.aw > 0.0 else -1.0
            dyaw = sg * self._ahead(0.0, abs(self.aw), H, P.w_cap, P.accw_eps)
        dyaw = max(-P.pred_yaw_max, min(P.pred_yaw_max, dyaw))
        vx, vy = self.v
        ax, ay = self.a
        s = math.hypot(vx, vy)
        if s > 0.005:
            ux, uy = vx / s, vy / s
        else:
            na = math.hypot(ax, ay)
            if na < 1e-9:
                return x, y, yaw + dyaw, bz
            ux, uy = ax / na, ay / na
            s = max(0.0, vx * ux + vy * uy)
        d = min(P.pred_max, self._ahead(s, ax * ux + ay * uy, H, P.v_cap, P.acc_eps))
        c, sn = math.cos(0.5 * dyaw), math.sin(0.5 * dyaw)
        return x + d * (c * ux - sn * uy), y + d * (sn * ux + c * uy), yaw + dyaw, bz

    @staticmethod
    def _dev(a, b):
        """(position, yaw) deviation of foot pose a from b."""
        return math.hypot(a[0] - b[0], a[1] - b[1]), abs(_wrap(a[3] - b[3]))

    def _target(self, side, t_land, horizon, lat, tstep):
        """The landing pose of `side`: its stance spot under the hip at the predicted base pose, within reach of the hip
        at the landing, its yaw within reach of the pelvis, the other foot and its lift-off, apart from the other foot,
        fully on or off a support. Also how far ahead of its hip's spot at the landing it lands (for its roll)."""
        P = self.P
        other = self.plant["right" if side == "left" else "left"]
        xp, yp, yawp, bz = self._predict(horizon)
        h = self.z0 + bz - other[2]
        # side-stepping: both feet land wider, so that the closing step still keeps w_min to the other foot
        vlat = -math.sin(yawp) * self.v[0] + math.cos(yawp) * self.v[1]
        w0 = abs(self.home("left", h)[1] - self.home("right", h)[1])
        widen = min(P.widen_max, max(0.0, (P.w_min + 0.01 - w0 + abs(vlat) * tstep) / 2.0)) if lat > 0.5 else 0.0
        tgt = self._home_world(side, xp, yp, yawp, h, widen)
        # within reach of the hip at the landing
        xl, yl, yawl, _ = self._predict(t_land - self.t)
        hx, hy = self._home_world(side, xl, yl, yawl, h)[:2]
        c, s = math.cos(yawl), math.sin(yawl)
        dx, dy = tgt[0] - hx, tgt[1] - hy
        fx, fy = c * dx + s * dy, -s * dx + c * dy
        sg = 1.0 if side == "left" else -1.0
        fx = max(-P.reach_back, min(P.reach_fwd, fx))
        fy = sg * max(-P.reach_in, min(P.reach_out, sg * fy))
        tgt[0], tgt[1] = hx + c * fx - s * fy, hy + s * fx + c * fy
        dyl = _wrap(tgt[3] - yawl)
        if abs(dyl) > P.yaw_rel_max:
            tgt[3] = yawl + math.copysign(P.yaw_rel_max, dyl)
        if abs(_wrap(tgt[3] - other[3])) > P.yaw_gap_max:
            tgt[3] = other[3] + math.copysign(P.yaw_gap_max, _wrap(tgt[3] - other[3]))
        own = self.plant[side][3] if self.swing is None or self.swing["side"] != side else self.swing["P0"][3]
        if abs(_wrap(tgt[3] - own)) > P.yaw_step_max:
            tgt[3] = own + math.copysign(P.yaw_step_max, _wrap(tgt[3] - own))
        tgt = self._apart(tgt, other, yawp, side)
        tgt = self._on_support(tgt, other, yawp, side)
        ahead = c * (tgt[0] - hx) + s * (tgt[1] - hy)
        return tgt, ahead

    def _apart(self, p, other, yaw, side):
        P = self.P
        nx, ny = -math.sin(yaw), math.cos(yaw)
        sg = 1.0 if side == "left" else -1.0
        lat = ((p[0] - other[0]) * nx + (p[1] - other[1]) * ny) * sg
        if lat < P.w_min:
            p[0] += nx * sg * (P.w_min - lat)
            p[1] += ny * sg * (P.w_min - lat)
        for _ in range(12):
            c = self._clearance(p, other, side)
            if c >= P.c_min:
                break
            p[0] += nx * sg * (P.c_min - c + 0.002)
            p[1] += ny * sg * (P.c_min - c + 0.002)
        return p

    def _clearance(self, p, other, side):
        """Gap between the two soles' rectangles along the best separating axis (a lower bound of their distance;
        negative when they overlap)."""
        A = self._corners(p, side)
        Bc = self._corners(other, "right" if side == "left" else "left")
        best = -1e9
        for yaw in (p[3], other[3]):
            for ax, ay in ((math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))):
                pa = [ax * u + ay * v for u, v in A]
                pb = [ax * u + ay * v for u, v in Bc]
                best = max(best, min(pb) - max(pa), min(pa) - max(pb))
        return best

    def _on_support(self, p, other, yaw, side):
        """Move a landing that straddles a support's edge so its sole is fully on or off it (the cheapest shift within
        rug_along along the step and rug_across across it, along counting double), keeping it apart from the other
        foot; unchanged when none works (the sole then rests on the support's top, its edge under the sole)."""
        P = self.P
        if not self.supports.polys:
            return [p[0], p[1], 0.0, p[3]]
        z, full = self.supports.rest(self._corners(p, side))
        if full:
            return [p[0], p[1], z, p[3]]
        dx, dy = p[0] - self.plant[side][0], p[1] - self.plant[side][1]
        n = math.hypot(dx, dy)
        ux, uy = (dx / n, dy / n) if n > 0.02 else (math.cos(p[3]), math.sin(p[3]))
        da, dc, cost = self._shift_grid
        cx = p[0] + da * ux - dc * uy
        cy = p[1] + da * uy + dc * ux
        nx, ny = -math.sin(yaw), math.cos(yaw)
        sg = 1.0 if side == "left" else -1.0
        # a shift outwards (away from the other foot, widening the stance) costs three times as much as one inwards
        out = ((cx - p[0]) * nx + (cy - p[1]) * ny) * sg
        order = np.argsort(cost + 2.0 * np.maximum(out, 0.0), kind="stable")
        cx, cy = cx[order], cy[order]
        L = np.array(self._corners([0.0, 0.0, 0.0, p[3]], side))
        C = np.stack([cx[:, None] + L[None, :, 0], cy[:, None] + L[None, :, 1]], axis=2)
        zz, ok = self.supports.rest_many(C)
        ok &= ((cx - other[0]) * nx + (cy - other[1]) * ny) * sg >= P.w_min - 1e-6
        for i in np.flatnonzero(ok):                   # the cheapest first
            q = [float(cx[i]), float(cy[i]), float(zz[i]), p[3]]
            if self._clearance(q, other, side) >= P.c_min - 1e-6:
                return q
        return [p[0], p[1], z, p[3]]

    def _advance(self, t, base, dt):
        """One planner tick at time t (base pose there, dt since the previous tick)."""
        P = self.P
        x, y, yaw, bz = base
        x0, y0, yaw0, _ = self.base
        if dt > 1e-12:
            b = 1.0 - math.exp(-dt / P.tau_v)
            ba = 1.0 - math.exp(-dt / P.tau_a)
            vx0, vy0, w0 = self.v[0], self.v[1], self.w
            self.v[0] += b * ((x - x0) / dt - self.v[0])
            self.v[1] += b * ((y - y0) / dt - self.v[1])
            self.w += b * (_wrap(yaw - yaw0) / dt - self.w)
            self.a[0] += ba * ((self.v[0] - vx0) / dt - self.a[0])
            self.a[1] += ba * ((self.v[1] - vy0) / dt - self.a[1])
            self.aw += ba * ((self.w - w0) / dt - self.aw)
        self.base = (x, y, yaw, bz)
        self.t = t
        # moving: the base's travel over the last motion_window s (a goal pose's last-millimetre snap does not count)
        hist = self.hist
        hist.append((t, x, y, yaw))
        while len(hist) > 2 and hist[1][0] <= t - P.motion_window + 1e-9:
            hist.popleft()
        th, xh, yh, yawh = hist[0]
        span = t - th
        if span > 1e-9:
            v = math.hypot(x - xh, y - yh) / span
            wv = abs(_wrap(yaw - yawh)) / span
            if v > P.v_on or wv > P.w_on:
                if not self.moving:
                    self.settle_due = {s: False for s in SIDES}
                self.moving = True
            elif self.moving and v < 0.5 * P.v_on and wv < 0.5 * P.w_on:
                self.moving = False
                self.still_since = t
                self.settle_due = {s: True for s in SIDES}
        tsw, tds, tst, lat = self._timing()
        sw = self.swing
        if sw is not None:
            if t >= sw["t1"] - 1e-9:
                s = sw["side"]
                self.plant[s] = list(sw["P1"])
                self.pivot[s] = sw["c1"]
                self.load[s] = self._load_of(sw)
                self.flip_t[s] = None
                self.last_land, self.last_side, self.last_walk = sw["t1"], s, sw["walk"]
                self.swing = None

            else:
                u = (t - sw["t0"]) / (sw["t1"] - sw["t0"])
                # walking on: the stance foot's heel rises from the swing's middle to its own expected lift-off (its
                # toe-off), so that its knee bends before the foot leaves the floor
                o = "right" if sw["side"] == "left" else "left"
                if self.moving and u >= P.to_from and self.pre[o] is None and \
                        (self.pivot[o] == TOE or self.theta[SIDES.index(o)] <= P.flip_theta):
                    t_e = sw["t1"] + tds
                    hx_, _ = self.home(o, self.z0 + bz - self.plant[o][2])
                    c_, s_ = math.cos(yaw), math.sin(yaw)
                    behind = hx_ - (c_ * (self.plant[o][0] - x) + s_ * (self.plant[o][1] - y)) + \
                        (c_ * self.v[0] + s_ * self.v[1]) * (t_e - t)
                    r_ = min(1.0, max(0.0, (behind - P.to_x0) / (P.to_x1 - P.to_x0)))
                    if r_ > 0.0:                # a foot not behind its hip keeps its edge (no toe-off)
                        self.pre[o] = (t, t_e, P.to_theta * r_)
                g = 1.0 - _ease((u - P.u_rt0) / (P.u_rt1 - P.u_rt0))
                if g <= 0.0 and self.supports.polys:
                    p1 = sw["P1"]
                    lz = P.retarget_z * dt
                    p1[2] += max(-lz, min(lz, self._settled(p1, sw["side"])[2] - p1[2]))
                if g > 0.0:
                    tgt, _ = self._target(sw["side"], sw["t1"], (sw["t1"] - t) + 0.5 * tst if sw["walk"] else 0.0,
                                          lat, sw["t1"] - sw["t0"] + tds)
                    p1 = sw["P1"]
                    dx, dy = tgt[0] - p1[0], tgt[1] - p1[1]
                    n = math.hypot(dx, dy)
                    lim = g * P.retarget_v * dt
                    if n > lim:
                        dx, dy = dx * lim / n, dy * lim / n
                    lw = g * P.retarget_w * dt
                    dyaw = max(-lw, min(lw, _wrap(tgt[3] - p1[3])))
                    p1 = [p1[0] + dx, p1[1] + dy, p1[2], p1[3] + dyaw]
                    # the landing's height follows the support under it at the full rate, to the landing
                    if self.supports.polys:
                        lz = P.retarget_z * dt
                        p1[2] += max(-lz, min(lz, self._settled(p1, sw["side"])[2] - p1[2]))
                    sw["P1"] = p1
                    if u < 0.5 and p1[2] < sw["P0"][2] - 1e-3:   # (a descent starts at half the swing at the earliest)
                        sw["ug_dn"] = self._ground_span(sw, False)
                    if not sw["squat"]:
                        want, sw["ts"] = self._landing_roll(sw["side"], sw["P1"], sw["t1"], True)
                        lr = g * P.land_roll_rate * dt
                        sw["phi1"] += max(-lr, min(lr, want - sw["phi1"]))
                        if u < P.hs_from:
                            sw["c1"] = HEEL if sw["phi1"] < 0.0 else TOE if sw["phi1"] > 0.0 else sw["c1"]
                        elif sw["c1"] == HEEL:
                            sw["phi1"] = min(0.0, sw["phi1"])
                        else:
                            sw["phi1"] = max(0.0, sw["phi1"])
                return
        h = self.z0 + bz
        H = tsw + 0.5 * tst if self.moving else 0.0     # standing: the base as it is (no twitch extrapolated)
        xp, yp, yawp, _ = self._predict(H)
        dev_now, dev = {}, {}
        for s in SIDES:
            hs = h - self.plant[s][2]
            dev_now[s] = self._dev(self.plant[s], self._home_world(s, x, y, yaw, hs))
            dev[s] = self._dev(self.plant[s], self._home_world(s, xp, yp, yawp, hs))
        # a stance foot the base has left far from its spot under the hip, now and as predicted, steps next, without
        # the double support (a foot that has just landed where the base is going is not left behind)
        strain = {s: min(max(dev_now[s][0] / P.d_strain, dev_now[s][1] / P.yaw_strain),
                         max(dev[s][0] / P.d_strain, dev[s][1] / P.yaw_strain)) for s in SIDES}
        urgent = max(SIDES, key=lambda s: (strain[s], s == "left")) if max(strain.values()) > 1.0 else None
        if t - self.last_land < (tds if urgent is None else P.tds_min * 0.5):
            return
        hold = self.hold_reach(h - max(self.plant[s][2] for s in SIDES))
        bout = self.last_side is not None and (t - self.last_land) <= P.restart_s and self.last_walk
        settle = False
        if self.moving:
            dl, yl = (P.d_walk, P.yaw_walk) if bout else (max(P.d_walk, hold), P.yaw_hold)
        elif max(max(d[0] / hold, d[1] / P.yaw_hold) for d in dev_now.values()) > 1.0:
            dl, yl = hold, P.yaw_hold                   # standing: a foot the base has drifted away from
        elif t - self.still_since >= P.settle_wait and any(self.settle_due.values()):
            dl, yl, settle = P.d_settle, P.yaw_settle, True
        else:
            return
        need = {s: max(dev[s][0] / dl, dev[s][1] / yl) if (not settle or self.settle_due[s]) else 0.0 for s in SIDES}
        pick = None
        if not bout:
            cand = [s for s in SIDES if need[s] > 1.0]
            if cand:
                def score(s):
                    tg = self._home_world(s, xp, yp, yawp, h - self.plant[s][2])
                    sg = 1.0 if s == "left" else -1.0
                    nx, ny = -math.sin(yawp) * sg, math.cos(yawp) * sg
                    out = (tg[0] - self.plant[s][0]) * nx + (tg[1] - self.plant[s][1]) * ny
                    return need[s] * dl + 0.5 * max(0.0, out) + 0.05 * max(0.0, sg * _wrap(tg[3] - self.plant[s][3]))
                pick = max(cand, key=lambda s: (score(s), s == "left"))
        else:
            o = "right" if self.last_side == "left" else "left"
            if need[o] > 1.0:
                pick = o
            elif need[self.last_side] > 2.0:
                pick = self.last_side
        if urgent is not None and not (bout and urgent == self.last_side and need["right" if urgent == "left" else
                                                                                   "left"] > 1.0):
            pick = urgent                               # (the other foot of a bout, left behind too, goes first)
        if settle and pick is not None:
            self.settle_due[pick] = False
            tsw = max(tsw, P.tsw_settle)
        if pick is None:
            return
        tgt, ahead = self._target(pick, t + tsw, H, lat, tsw + tds)
        D, dyw = self._dev(tgt, self.plant[pick])
        if D > P.step_max:                              # one swing travels at most step_max (the next one catches up)
            p0 = self.plant[pick]
            r_ = P.step_max / D
            tgt = self._on_support([p0[0] + (tgt[0] - p0[0]) * r_, p0[1] + (tgt[1] - p0[1]) * r_, 0.0, tgt[3]],
                                   self.plant["right" if pick == "left" else "left"], tgt[3], pick)
            ahead -= (1.0 - r_) * D
            D = P.step_max
        if urgent is None and max(D / dl, dyw / yl) <= P.step_least:
            return                                      # the apart / support rules leave nothing worth a step
        lift = min(P.h_max, max(P.h_min, (P.h_base + P.h_gain * D + P.h_yaw * dyw) * (1.0 - (1.0 - P.h_lat) * lat)))
        k = SIDES.index(pick)
        c0 = self.pivot[pick]
        th0 = float(self.theta[k])
        phi0 = th0 if c0 == TOE else -th0
        # the landing roll: over the heel for a landing ahead of the hip (a heel strike), over the toe behind it
        phi1, ts = self._landing_roll(pick, tgt, t + tsw, True)
        c1 = HEEL if phi1 < 0.0 else TOE if phi1 > 0.0 else c0
        self.pre[pick] = None
        self.flip_t[pick] = None
        self.swing = {"side": pick, "t0": t, "t1": t + tsw, "P0": list(self.plant[pick]), "P1": tgt, "h": lift,
                      "walk": bool(self.moving),
                      "c0": c0, "c1": c1, "th0": th0, "phi0": phi0, "phi1": phi1, "ts": ts}
        sw = self.swing
        # a deep squat: the swing foot rolls over its toe throughout (decided once, at the lift-off)
        sw["squat"] = bool(h - max(sw["P0"][2], tgt[2]) - lift < self.h_squat and (c0 == TOE or th0 < 0.005))
        if sw["squat"]:
            sw["phi1"], sw["c1"] = 0.0, TOE
        if tgt[2] > sw["P0"][2] + 1e-3:
            sw["ug_up"] = self._ground_span(sw, True)
        elif tgt[2] < sw["P0"][2] - 1e-3:
            sw["ug_dn"] = self._ground_span(sw, False)
        self.load[pick] = None

    # ------------------------------------------------------------------------------------------- foot targets, IK
    def _lift(self, u):
        """Swing lift profile (0 .. 1 .. 0, highest at lift_peak): u^m (1 - u)^n, smooth at both ends."""
        return self._lift_k * u ** self.P.lift_m * (1.0 - u) ** self._lift_n

    @staticmethod
    def _ground(sw, u):
        """The support height under a swing foot: a step up (onto a support) reaches it by the time its sole would
        reach the support (at most half the swing, at least 15 %), a step down leaves it once the sole has passed its
        edge (from half the swing on, by 90 % at the latest): the foot does not cut through the support's edge."""
        z0, z1 = sw["P0"][2], sw["P1"][2]
        if z1 > z0:
            return z0 + (z1 - z0) * _ease(u / sw.get("ug_up", 0.5))
        d = sw.get("ug_dn", 0.5)
        return z0 + (z1 - z0) * _ease((u - d) / (1.0 - d))

    def _ground_span(self, sw, up):
        """Where along a swing between support heights its flat sole first lies over the higher support (a step up,
        up=True) or last lies over the one it leaves (a step down): the swing's share, sampled every 10 %."""
        P0, P1, side = sw["P0"], sw["P1"], sw["side"]
        zt = (P1[2] if up else P0[2]) - 1e-4
        last = 0.5
        for i in range(11):
            u = 0.1 * i if up else 1.0 - 0.05 * i
            s_ = _minjerk(u)
            pose = (P0[0] + (P1[0] - P0[0]) * s_, P0[1] + (P1[1] - P0[1]) * s_, 0.0,
                    P0[3] + _wrap(P1[3] - P0[3]) * s_)
            if self.supports.rest(self._corners(pose, side))[0] >= zt:
                return min(0.5, max(0.15, u)) if up else min(0.9, max(0.5, u + 0.05))
            last = u
        return 0.5 if up else max(0.5, last)

    def _swing_pose(self, t):
        sw = self.swing
        u = min(1.0, max(0.0, (t - sw["t0"]) / (sw["t1"] - sw["t0"])))
        s = _minjerk(u)
        P0, P1 = sw["P0"], sw["P1"]
        pose = (P0[0] + (P1[0] - P0[0]) * s, P0[1] + (P1[1] - P0[1]) * s,
                self._ground(sw, u) + sw["h"] * self._lift(u),
                P0[3] + _wrap(P1[3] - P0[3]) * s)
        return pose, u

    def _targets(self, t, base):
        """Per leg, in the pelvis frame: the flat foot (ankle-roll origin, yaw), the edge it may roll over (a point in
        the foot frame and the roll axis' sign: +1 the toe, the heel lifting; -1 the heel, the toe lifting), the pivot
        chosen, the roll's bounds and the knee's lower bound. The roll follows the gait: a swing foot unrolls its
        lift-off roll over the first `unroll` of the swing and builds up its landing roll (a heel strike for a landing
        ahead of the hip) from `hs_from` on; a landed foot eases that roll off over t_load; a stance foot rolls over the
        edge towards its hip as far as its leg cannot reach flat; in a deep squat (where the ankle's dorsiflexion limit
        would tip a lifted flat foot's toe into the floor) a swing foot rolls over its toe."""
        P = self.P
        x, y, yaw, bz = base
        hz = self.z0 + bz
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        sw = self.swing
        for k, side in enumerate(SIDES):
            hx, hy = self.hip_xy[k]
            hip = (x + c * hx - s * hy, y + s * hx + c * hy)
            th_lo, th_hi, ks, flip = 0.0, P.theta_max, P.knee_soft, False
            if sw is not None and sw["side"] == side and t < sw["t1"]:
                pose, u = self._swing_pose(t)
                ground = self._ground(sw, u)
                if sw["squat"]:
                    land = piv = TOE
                    ks = 0.0
                else:
                    phi = sw["phi0"] * (1.0 - _ease(u / P.unroll)) + \
                        sw["phi1"] * _ease((u - P.hs_from) / (1.0 - P.hs_from))
                    if u < P.hs_from:
                        piv = TOE if phi > 0.0 else HEEL if phi < 0.0 else (TOE if u < 0.5 else sw["c1"])
                        th_lo = th_hi = abs(phi)                # prescribed in the air
                        land = sw["c1"]
                        ks = 0.0
                    else:
                        land = piv = sw["c1"] if sw["phi1"] != 0.0 else \
                            self._pivot(side, sw["P1"], hip, hz - sw["P1"][2], sw["c1"], 0.0)[0]
                        th_lo = abs(phi)                        # the landing roll, more if the reach needs it
            else:
                landed = sw is not None and sw["side"] == side      # a swing past its landing time (pose_at)
                pose = sw["P1"] if landed else self.plant[side]
                ld = self._load_of(sw) if landed else self.load[side]
                pre = None if landed else self.pre[side]
                cur = sw["c1"] if landed else self.pivot[side]
                fl = None if landed else self.flip[side]
                ft = None if landed else self.flip_t[side]
                if ld is not None and t < ld[0] + ld[3] and cur == ld[2]:
                    # the landing roll eases off; once the hip has passed over the foot, it unrolls (over t_flip)
                    # and the edge changes
                    th_lo = ld[1] * (1.0 - _ease((t - ld[0]) / ld[3]))
                    land, flip = self._pivot(side, pose, hip, hz - pose[2], cur, float(self.theta[k]))
                    if flip and fl is not None:
                        th_hi = min(th_hi, fl[1] * (1.0 - _ease((t - fl[0]) / P.t_flip)))
                        th_lo = min(th_lo, th_hi)
                    elif land != cur:
                        th_lo = 0.0
                elif pre is not None and t < pre[1] + P.to_hold + P.to_fall:
                    land = TOE                                  # the toe-off: the heel rising, then let down
                    th_lo = pre[2] * (_ease((t - pre[0]) / max(1e-6, pre[1] - pre[0])) -
                                      _ease((t - pre[1] - P.to_hold) / P.to_fall))
                else:
                    land, flip = self._pivot(side, pose, hip, hz - pose[2], cur, float(self.theta[k]))
                    if flip and fl is not None:                 # unrolling before the edge changes
                        th_hi = min(th_hi, fl[1] * (1.0 - _ease((t - fl[0]) / P.t_flip)))
                piv = land
                ft = t if land != cur else ft
                if ft is not None and t < ft + P.t_flip_in:     # the roll over a new edge eases in
                    th_hi = min(th_hi, P.theta_max * _ease((t - ft) / P.t_flip_in))
                    th_lo = min(th_lo, th_hi)
                ground = pose[2]
            hov = max(0.0, hz - ground - self.h_top)          # above the straight-leg height: hover as posed
            dx, dy = pose[0] - x, pose[1] - y
            if ks > 0.0:                                      # the knee's soft floor eases in away from the hip
                hx_, hy_ = self.home(side, hz - ground)
                d = math.hypot(c * dx + s * dy - hx_, -s * dx + c * dy - hy_)
                ks *= _ease((d - P.soft_d0) / (P.soft_d1 - P.soft_d0))
            out.append(((c * dx + s * dy, -s * dx + c * dy, pose[2] + hov + self.ankle_h0 - hz), pose[3] - yaw,
                        self._edge(piv), -1.0 if piv == HEEL else 1.0, land, th_lo, th_hi, P.knee_min, ks, flip))
        return out

    def _landing_roll(self, side, P1, t_land, ret_kind=False):
        """The roll (signed: + over the toe, - over the heel) a landing at P1 takes: a heel strike by how far ahead of
        its hip's spot it lands along the foot, and at least as much as lets its knee bend land_knee at the base
        predicted for the landing (the leg as hip pitch to ankle pitch distance; the roll over the edge towards the
        hip shortens it); a toe strike behind the spot or a side landing (over the toe) up to ts_theta, more only to
        let the knee bend ts_knee. With ret_kind also whether it is a toe strike or a side landing."""
        P = self.P
        xl, yl, yawl, bz = self._predict(t_land - self.t)
        h = self.z0 + bz - P1[2]
        k = SIDES.index(side)
        c, s = math.cos(yawl), math.sin(yawl)
        hx, hy = self.home(side, h)
        # the landing against its hip's spot, along the foot (as the planted foot's edge is chosen) and across it
        dx, dy = P1[0] - (xl + c * hx - s * hy), P1[1] - (yl + s * hx + c * hy)
        cf, sf = math.cos(P1[3]), math.sin(P1[3])
        ahead, across = cf * dx + sf * dy, -sf * dx + cf * dy
        lon = abs(ahead) / max(1e-9, math.hypot(ahead, across))   # 1: along the foot, 0: a side landing
        heel = ahead > 0.0 and lon > 0.5                # a heel strike (a landing ahead of its hip)
        style = (P.hs_theta if heel else P.ts_theta * lon) * \
            min(1.0, max(0.0, (abs(ahead) - P.hs_x0) / (P.hs_x1 - P.hs_x0)))
        ho = self._hip_off[k]
        Hx, Hy, Hz = xl + c * ho[0] - s * ho[1], yl + s * ho[0] + c * ho[1], self.z0 + bz + ho[2]
        hov = max(0.0, h - self.h_top)
        ex = self.x_heel if heel else self.x_toe        # a side landing too lands on the ball of its foot
        Ex, Ey, Ez = P1[0] + cf * ex, P1[1] + sf * ex, P1[2] + hov
        px, pz = -ex, self.ankle_h0 + self._ap_up               # the ankle pitch axis from the edge (foot frame)
        Lk = self._leg_land if heel else self._leg_ts
        sgn = 1.0 if heel else -1.0

        def length(th):
            a = sgn * th
            qx, qz = px * math.cos(a) - pz * math.sin(a), px * math.sin(a) + pz * math.cos(a)
            return math.sqrt((Ex + cf * qx - Hx) ** 2 + (Ey + sf * qx - Hy) ** 2 + (Ez + qz - Hz) ** 2)
        need = 0.0
        if length(0.0) > Lk:
            lo_, hi_ = 0.0, P.land_roll_max
            if length(hi_) > Lk:
                need = hi_
            else:
                for _ in range(8):
                    mid = 0.5 * (lo_ + hi_)
                    if length(mid) > Lk:
                        lo_ = mid
                    else:
                        hi_ = mid
                need = hi_
        th = min(P.land_roll_max, max(style, need))
        if ret_kind:
            return (-th if heel else th), not heel
        return -th if heel else th

    def _pivot(self, side, pose, hip, h_rel, current, theta, flip=None):
        """Heel or toe: the foot's edge towards the hip, switching where the foot passes its stance spot under the hip
        (with hysteresis), and only while the foot is flat. Also whether a change waits for the foot to unroll."""
        c, s = math.cos(pose[3]), math.sin(pose[3])
        fx = (hip[0] - pose[0]) * c + (hip[1] - pose[1]) * s    # the hip ahead of the ankle, along the foot
        ref = -self.home(side, h_rel)[0]
        band = self.P.pivot_band
        want = current
        if current == HEEL and fx > ref + band:
            want = TOE
        elif current == TOE and fx < ref - band:
            want = HEEL
        if want != current and theta > self.P.flip_theta:
            return current, True
        return want, False

    def _edge(self, piv):
        return self.x_heel if piv == HEEL else self.x_toe

    def _solve(self, t, base, iters, commit):
        """The legs to draw at time t (the IK warm-started from the last solution; a swing leg's joints within the
        limiter's bounds); commit keeps the solution, the pivots and the limiter's state."""
        P = self.P
        if commit and self.swing is None and self.solved_for is not None and self.converged and \
                not self._rolling(t):
            b0 = self.solved_for
            if max(abs(base[0] - b0[0]), abs(base[1] - b0[1]), abs(_wrap(base[2] - b0[2])),
                   abs(base[3] - b0[3])) < P.still_tol and self.solved_plant == self._plant_key():
                return self.q_draw
        tg = self._targets(t, base)
        # (pose_at works on a copy of the active set: nothing it does reaches the next step)
        q, th, conv = self._ik(self.q.copy(), self.theta.copy(), tg, iters,
                               P.ik_tol if commit else P.ik_tol_eval,
                               self._fixed if commit else self._fixed.copy())
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(th))):
            q, th, conv = self.posed(self.z0 + base[3]), np.zeros(2), False
        # a swing leg in the air: its joints follow the IK's within lim_v / lim_a (the limited path handed back to
        # the IK's over the end of the swing, so that the foot lands exactly)
        sw = self.swing
        draw = q.copy()
        k = SIDES.index(sw["side"]) if sw is not None and t < sw["t1"] else None
        if k is not None:
            u = min(1.0, max(0.0, (t - sw["t0"]) / (sw["t1"] - sw["t0"])))
            blend = 1.0 - _ease((u - P.lim_until) / (1.0 - P.lim_until))
            j = slice(6 * k, 6 * k + 6)
            dt = t - self.t_solve
            if commit:
                if 1e-6 < dt < 0.05:
                    e = q[j] - self.q_lim[j]
                    ae = np.abs(e)
                    vd = np.sign(e) * np.minimum(np.minimum(ae / dt, np.sqrt(2.0 * P.lim_a * ae)), P.lim_v)
                    v = self.v_lim[j] + np.clip(vd - self.v_lim[j], -P.lim_a * dt, P.lim_a * dt)
                    self.q_lim[j] = self.q_lim[j] + v * dt
                    self.v_lim[j] = v
                elif dt >= 0.05:
                    self.q_lim[j], self.v_lim[j] = q[j], 0.0
                self.lim_off = (self.q_lim[j] - q[j]).copy()
            draw[j] = np.clip(q[j] + self.lim_off * blend, self.lo[j], self.hi[j])
        if commit:
            dt = t - self.t_solve
            for kk in range(2):
                if kk != k:                                  # stance: drawn as solved; the limiter follows
                    j = slice(6 * kk, 6 * kk + 6)
                    self.v_lim[j] = (q[j] - self.q_lim[j]) / dt if 1e-6 < dt < 0.05 else 0.0
                    self.q_lim[j] = q[j]
            self.q, self.theta, self.q_draw = q, th, draw
            self.t_solve = t
            self.converged = conv
            for kk, side in enumerate(SIDES):
                if sw is not None and sw["side"] == side and t < sw["t1"]:
                    sw["c1"] = tg[kk][4]
                    self.flip[side] = None
                else:
                    if tg[kk][4] != self.pivot[side]:
                        self.flip_t[side] = t
                    self.pivot[side] = tg[kk][4]
                    if not tg[kk][9]:
                        self.flip[side] = None
                    elif self.flip[side] is None:
                        self.flip[side] = (t, float(th[kk]))
            self.solved_for = tuple(base)
            self.solved_plant = self._plant_key()
        return draw

    def _load_of(self, sw):
        """The landed foot's roll to ease off: (landing time, roll, edge, how long)."""
        if abs(sw["phi1"]) <= 1e-9:
            return None
        return sw["t1"], abs(sw["phi1"]), sw["c1"], self.P.t_load_ts if sw.get("ts") else self.P.t_load

    def _rolling(self, t):
        """Whether a planted foot's roll is still changing with the time (a toe-off, a landing roll easing off, an
        edge change): the legs are solved again even when the base stands still."""
        P = self.P
        for s in SIDES:
            ld, pre, ft = self.load[s], self.pre[s], self.flip_t[s]
            if (ld is not None and t < ld[0] + ld[3]) or (pre is not None and t < pre[1] + P.to_hold + P.to_fall) \
                    or self.flip[s] is not None or (ft is not None and t < ft + P.t_flip_in):
                return True
        return False

    def _plant_key(self):
        return tuple(tuple(self.plant[s]) for s in SIDES) + (self.swing is not None,)

    def _ik(self, q, th, tg, iters, tol, fixed):
        """Damped least squares on both legs at once, 7 unknowns per leg: its 6 joints and the foot's roll theta >= 0
        over its pivot edge. The ankle-roll frame must match the flat foot rolled by theta (6 rows), theta is drawn to
        0 (a small weight), joints and theta held at their limits (a primal active set, kept between calls). Up to
        `iters` checks of the residual, each followed by a step unless it is below `tol` (m); after a step from a
        residual under `track` m no further check is made (the next call starts from it). Each leg iterates on its
        own (the other leg's checks and steps leave it alone). A knee below its soft floor is drawn up (a Huber
        penalty). Returns (q, theta, converged: both legs' last check passed)."""
        P, lm, ld = self.P, self.lm, self.ld
        # per leg, in one array: the flat foot's ankle-roll origin pF and yaw frame RF (pelvis frame), the roll axis
        # (+-the foot's y), the pivot point on it, the axis' cross-product matrix K and K @ K
        h0 = self.ankle_h0
        vals = []
        for (px, py, pz), yw, xc, sg, _, _, _, _, _, _ in tg:
            c, s = math.cos(yw), math.sin(yw)
            ax, ay = -s * sg, c * sg
            vals += (px, py, pz, c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0, ax, ay, 0.0, px + c * xc, py + s * xc, pz - h0,
                     0.0, 0.0, ay, 0.0, 0.0, -ax, -ay, ax, 0.0,
                     ax * ax - 1.0, ax * ay, 0.0, ax * ay, ay * ay - 1.0, 0.0, 0.0, 0.0, -1.0)
        V = np.array(vals).reshape(2, 36)
        pF, RF, axis, pivot = V[:, 0:3], V[:, 3:12].reshape(2, 3, 3), V[:, 12:15], V[:, 15:18]
        K, KK = V[:, 18:27].reshape(2, 3, 3), V[:, 27:36].reshape(2, 3, 3)
        r0 = (pF - pivot)[:, :, None]
        lo, hi, eye7 = self._lo7.copy(), self._hi7.copy(), self._eye7
        lo[:, 6] = (tg[0][5], tg[1][5])
        hi[:, 6] = (tg[0][6], tg[1][6])
        lo[:, 3] = np.maximum(lo[:, 3], (tg[0][7], tg[1][7]))
        X = self._X
        X[:, :6] = q.reshape(2, 6)
        X[:, 6] = np.clip(th, lo[:, 6], hi[:, 6])
        X[:, 3] = np.maximum(X[:, 3], lo[:, 3])                    # a swing knee's rising bound
        pinned = hi - lo < 1e-12                                   # a prescribed roll: held, never released
        fixed[pinned] = 1.0
        jacp, jacr, fb, J, e = self.jacp, self.jacr, self.foot_body, self._J7, self._e7
        wr, wt = P.w_rot, P.w_theta
        ks = np.array((tg[0][8], tg[1][8]))
        on = ks > 0.0
        J[:, 3:6, 6] = -wr * axis
        conv = np.zeros(2, bool)
        act = np.ones(2, bool)                      # legs still iterating (one leg's events do not move the other)
        slack = X[:, 6] > lo[:, 6] + P.roll_slack   # a roll above its lower bound: drawn down by at least one step
        for it in range(iters):
            ld.qpos[:] = X[:, :6].ravel()
            mujoco.mj_kinematics(lm, ld)
            th_ = X[:, 6]
            Ra = self._eye3 + np.sin(th_)[:, None, None] * K + (1.0 - np.cos(th_))[:, None, None] * KK
            arm = (Ra @ r0)[:, :, 0]
            Mf = ((Ra @ RF) @ ld.xmat[fb].reshape(2, 3, 3).transpose(0, 2, 1)).reshape(2, 9)
            e[:, :3] = (pivot + arm) - ld.xpos[fb]
            e[:, 3:6] = (0.5 * wr) * (Mf[:, (7, 2, 3)] - Mf[:, (5, 6, 1)])
            res = np.abs(e[:, :6]).max(axis=1)
            done = act & (res < tol)
            if it == 0:
                done &= ~slack
            conv |= done
            act &= ~done
            if not act.any():
                break
            e[:, 6] = -wt * th_
            # the knee's soft floor (Huber: quadratic over knee_band, then linear), the violation drawn to 0
            xk = np.where(on, ks - X[:, 3], 0.0)
            band = P.knee_band
            e[:, 7] = P.w_knee * np.where(xk <= 0.0, 0.0, np.where(xk < band, 0.5 * xk * xk / band, xk - 0.5 * band))
            J[:, 7, 3] = P.w_knee * np.where(xk <= 0.0, 0.0, np.minimum(1.0, xk / band))
            mujoco.mj_comPos(lm, ld)
            mujoco.mj_jacBody(lm, ld, jacp[0], jacr[0], fb[0])
            mujoco.mj_jacBody(lm, ld, jacp[1], jacr[1], fb[1])
            J[0, :3, :6] = jacp[0, :, :6]
            J[1, :3, :6] = jacp[1, :, 6:]
            J[0, 3:6, :6] = jacr[0, :, :6]
            J[1, 3:6, :6] = jacr[1, :, 6:]
            J[:, 3:6, :6] *= wr
            J[:, :3, 6] = -(K @ arm[:, :, None])[:, :, 0]           # the target moves about the pivot's axis
            JT = J.transpose(0, 2, 1)
            # Levenberg-Marquardt damping: negligible while tracking, large for a target out of reach (no flailing)
            A = JT @ J + self._eye7 * (1e-10 + P.ik_lm * (e[:, :6] ** 2).sum(axis=1))[:, None, None]
            g = (JT @ e[:, :, None])[:, :, 0]
            # primal active set (kept between calls): unknowns held at a limit. A held one no longer at its limit (the
            # limit moved) or whose multiplier says the objective gains inwards is released; a free one at its limit
            # the step pushes outwards is held; then the remaining free ones are solved again
            at_lo, at_hi = X <= lo + 1e-12, X >= hi - 1e-12
            fixed_was = fixed.copy()
            fixed[(fixed > 0.0) & ~(at_lo | at_hi)] = 0.0
            fixed[pinned] = 1.0
            released = np.zeros_like(pinned)
            for _ in range(4):
                keep = 1.0 - fixed
                A2 = A * (keep[:, :, None] * keep[:, None, :]) + eye7 * fixed[:, :, None]
                dx = np.linalg.solve(A2, (g * keep)[:, :, None])[:, :, 0] * keep
                push = (keep > 0.0) & ~released & ((at_lo & (dx < 0.0)) | (at_hi & (dx > 0.0)))
                if push.any():
                    fixed[push] = 1.0
                    continue
                if not fixed.any():
                    break
                r = (A @ dx[:, :, None])[:, :, 0] - g
                release = (fixed > 0.0) & ~pinned & ((at_lo & (r < -1e-12)) | (at_hi & (r > 1e-12)))
                if not release.any():
                    break
                fixed[release] = 0.0
                released |= release
            dx[~act] = 0.0
            Xn = X + dx
            cut = (np.abs(dx).max(axis=1) > P.ik_step) | ((Xn < lo) | (Xn > hi)).any(axis=1)
            if cut.any():
                with np.errstate(divide="ignore", invalid="ignore"):
                    room = np.where(dx < 0.0, (lo - X) / dx, np.where(dx > 0.0, (hi - X) / dx, np.inf))
                room = np.where(fixed > 0.0, np.inf, np.maximum(room, 0.0))
                alpha = np.minimum(np.minimum(1.0, room.min(axis=1)),
                                   P.ik_step / np.maximum(np.abs(dx).max(axis=1), P.ik_step))
                fixed[(room <= alpha[:, None] + 1e-12) & (room < 1.0 + 1e-12)] = 1.0
                Xn = np.clip(X + alpha[:, None] * dx, lo, hi)  # a limit cut the step short: check and step again
            fixed[~act] = fixed_was[~act]
            X[:] = Xn
            act &= cut | (res >= P.ik_track)
            if not act.any():
                break
        return X[:, :6].ravel(), X[:, 6].copy(), bool(conv.all())

