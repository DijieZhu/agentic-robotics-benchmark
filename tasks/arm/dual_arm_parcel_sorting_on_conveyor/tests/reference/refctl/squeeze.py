"""Squeeze-flip core: planned two-palm squeeze flips with checks,
retries and clean give-ups (privileged: reads simulator truth).

Outputs only the 14 joint position targets and the 2 gripper commands per
20 ms tick (``ParcelCell.step``). Nothing else is written to the simulator.

A job is planned completely before the parcel is touched:

* per arm one continuous IK chain through every Cartesian segment (wait or
  hover, catch, descend, close, lift, rotate, lower, gate, escape), solved at
  execution resolution from several start configurations (the arm's
  self-motion), with the wrist pitch kept away from its limits;
* time scaling of every free segment to a joint-speed and acceleration
  budget; the feedforward position command is computed for the whole plan
  and its rate checked against the cell's 6 rad/s limit (own budget 5.5);
* a collision check of every tick on a private model copy (robot geoms with
  a margin): robot vs station structure and belt surface, arm vs arm, robot
  vs the parcel outside the intended pad contacts (parcel motion predicted:
  belt speed before the catch, rigid with the palms while held, belt speed
  after the release);
* alternatives: squeeze height, squeeze-line x offset, initial pad roll,
  lift height, catch position, rotation overshoot; the first plan that
  passes everything is used. No plan: the parcel is let through untouched.

After the catch the manipulation is re-planned from the measured pose
(same parameters first, then the others). After the rotation the label is
read and the residual corrected inside the planned roll margin; after the
set-down the outcome is read again: done -> release; label off -> give up
cleanly (release, park) or retry when a retry plan exists.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline

from parcelcell import CellConfig, ParcelCell, StationConfig

from .geom import (DOWN, UP, XN, XP, YN, YP, aligned_dims, label_dir,
                   quat_mat, world_dims)
from .kin2 import ArmKin, rot_axis_angle, rot_err
from .motion import Dyn, joint_move, min_jerk, min_jerk_d, min_jerk_dd

DT = 0.02
G = 9.81
SIDES = ("Left", "Right")
SL = {"Left": slice(0, 7), "Right": slice(7, 14)}
SG = {"Left": 1.0, "Right": -1.0}

# park posture (every link behind the belt's near edge; the ready
# posture, touch-free at the station)
Q_PARK = np.array([-0.7842, 2.1067, 0.7271, 1.946, -1.9143, -1.54, -1.1825,
                   0.7842, -2.1067, -0.7271, 1.946, 1.9143, 1.54, 1.1825])
# palm-work rest postures (wrist pitch about -0.85 instead of
# -1.46, which sits at the limit under squeeze load)
Q_REST = {
    "Left": np.array([-0.9, 0.99, 1.58, 1.3, -1.94, -0.84, -1.09]),
    "Right": np.array([0.9, -0.99, -1.58, 1.3, 1.94, 0.84, 1.09]),
}

PAD_TIP = 0.0025           # pad tip faces this far behind the TCP (tool z)
OVERPRESS = 0.0            # m per palm: the squeeze comes from the feedforward force
PAD_SOLREF = float(__import__("os").environ.get("PARCEL_PAD_SOLREF", "0.005"))  # fingertip pad contact time constant (0.005)
MAX_DROP = 0.06           # m, largest set-down drop of a flipped parcel (released at belt speed)
SOFT_DROP = 0.08          # m, the same for a soft mailer (an option)
PAD_X = 0.0466             # pad centre offset along tool x (open)
PAD_HALF_W = 0.011         # pad half width along tool y
PAD_HALF_T = 0.004         # pad half thickness along tool x
V_CMD_BUDGET = 5.5         # rad/s, own budget for the commanded rate
V_PLAN = 2.6               # rad/s, joint speed of free segments
A_PLAN = 12.0              # rad/s^2
CLEAR_STRUCT = float(__import__("os").environ.get("CLEAR_MM", "4")) * 1e-3       # m, min distance robot - structure in plans
CLEAR_BELT = float(__import__("os").environ.get("CLEAR_MM", "4")) * 1e-3         # m, min distance robot - belt surface
CLEAR_ARMS = float(__import__("os").environ.get("CLEAR_MM", "5")) * 1e-3         # m, arm - arm and arm - column
CLEAR_PARCEL = float(__import__("os").environ.get("CLEAR_MM", "4")) * 1e-3       # m, robot - parcel outside intended contact


def station_lip_window(window_half=None) -> StationConfig:
    """The station with an open near side. ``window_half``
    (env PARCEL_WINDOW_HALF): length of the open near side (the 3 cm lip)
    either side of the station centre; default 0.42 (the source)."""
    import os
    wh = window_half if window_half is not None else float(os.environ.get("PARCEL_WINDOW_HALF", "0.42"))
    return StationConfig(belt_center_x=0.38, belt_height=0.90,
                         lip_full_window=True, lip_height=0.03,
                         solid_frame=True, work_window_half=wh)


def cell_config(station=None, **kw) -> CellConfig:
    kw.setdefault("lines", (("scan", 0.95),))
    kw.setdefault("target_slew_rad_s", 6.0)
    kw.setdefault("monitor_substeps", True)
    kw.setdefault("home", tuple(Q_PARK))
    # stiff fingertip pad contact
    # (the gripper model's own pad value is 0.004); with 0.02 a held
    # parcel creeps 2-3.5 cm in the pads during a 180-deg flip, with 0.005
    # about 7 mm
    kw.setdefault("pad_solref", PAD_SOLREF)
    return CellConfig(station=station or station_lip_window(), **kw)


def make_cell(station=None, **kw) -> ParcelCell:
    return ParcelCell(cell_config(station, **kw))


class PlanFail(Exception):
    pass


class RotShort(PlanFail):
    """The roll stopped short of the target: ``frac`` of it was traced."""

    def __init__(self, msg, frac):
        super().__init__(msg)
        self.frac = frac


# ======================================================================
# geometry helpers
# ======================================================================
def palm_frame(side, theta0=0.0, yaw=0.0, tilt=0.0):
    """Palm frame: tool z along -SG*y (toward the parcel face), tool x along
    world +x (horizontal pads), tilted down by ``tilt`` (about world x),
    yawed by ``yaw`` (about z), then rolled by ``theta0`` as a world
    rotation about +y (the flip axis): R = R_y(theta0) R_z(yaw) R_tilt R_h."""
    z = np.array([0.0, -SG[side], 0.0])
    x = np.array([1.0, 0.0, 0.0])
    R = np.column_stack([x, np.cross(z, x), z])
    if tilt:
        R = rot_axis_angle((1, 0, 0), SG[side] * tilt) @ R
    if yaw:
        R = rot_axis_angle((0, 0, 1), yaw) @ R
    if theta0:
        R = rot_axis_angle((0, 1, 0), theta0) @ R
    return R


def Ry(th):
    return rot_axis_angle((0, 1, 0), th)


def corners_xz(Lx, H):
    return np.array([[sx * Lx / 2, sz * H / 2] for sx in (-1, 1)
                     for sz in (-1, 1)])


def settled_span(rot, xs, L_face):
    """x span of a parcel after it settles about its lowest corner onto the
    face that is nearest to horizontal. rot: (4, 2) x-z corners relative to
    the squeeze line (already rotated), xs: squeeze line x, L_face: length
    of the target face along x."""
    i = int(np.argmin(rot[:, 1]))
    xp = xs + float(rot[i, 0])
    # the two corners adjacent to the lowest one: the one that falls is on
    # the side of the centre
    cx = xs + float(np.mean(rot[:, 0]))
    if cx < xp:                       # falls toward -x
        return xp - L_face, xp
    return xp, xp + L_face


# ======================================================================
# executor
# ======================================================================
class Exec:
    """Plays per-tick samples (q, qd, qdd, grip, palm forces) through the
    inverse-dynamics feedforward; guards the commanded rate (counted)."""

    def __init__(self, cell):
        self.cell = cell
        self.dyn = Dyn(cell.model)
        js = cell.joint_state()
        self.q = js.q.copy()
        self.qd = np.zeros(14)
        self.u_prev = js.q_target.copy()
        self.grip = np.zeros(2)
        self.hooks = []
        self.ticks = 0
        self.peak_rate = 0.0
        self.guard_clips = 0
        self.guard_log = []
        self.peak_qd = np.zeros(14)
        self.peak_tau = np.zeros(14)
        self.tau_sat_ticks = 0
        self._tlim = self.dyn.tlim
        self._sid = {s: cell.model.site(f"{s}_tcp").id for s in SIDES}

    def command(self, q, qd, qdd, F):
        tau_x = None
        if F is not None:
            for s in SIDES:
                f = F.get(s) if isinstance(F, dict) else None
                if f is None:
                    continue
                J = self.dyn.jac_point(q, None, site=f"{s}_tcp")
                t = J.T @ np.asarray(f, float)
                tau_x = t if tau_x is None else tau_x + t
        return self.dyn.command(q, qd, qdd, tau_x)

    def tick(self, q, qd=None, qdd=None, grip=None, F=None):
        q = np.asarray(q, float)
        qd = np.zeros(14) if qd is None else np.asarray(qd, float)
        u = self.command(q, qd, qdd, F)
        du = u - self.u_prev
        lim = V_CMD_BUDGET * DT * 1.06            # just under 6 rad/s
        if np.any(np.abs(du) > lim):
            self.guard_clips += 1
            j = int(np.argmax(np.abs(du)))
            self.guard_log.append((getattr(self, "cur_seg", "?"), j,
                                   round(float(np.abs(du[j])) / DT, 2)))
            u = self.u_prev + np.clip(du, -lim, lim)
        self.peak_rate = max(self.peak_rate, float(np.max(np.abs(u - self.u_prev))) / DT)
        self.u_prev = u
        self.q, self.qd = q, qd
        if grip is not None:
            self.grip = np.asarray(grip, float)
        self.cell.step(u, self.grip)
        js = self.cell.joint_state()
        self.peak_qd = np.maximum(self.peak_qd, np.abs(js.qd))
        at = np.abs(js.tau)
        self.peak_tau = np.maximum(self.peak_tau, at)
        if np.any(at > self._tlim - 0.2):
            self.tau_sat_ticks += 1
        self.ticks += 1
        for h in self.hooks:
            h(self)

    def hold(self, n, F=None, grip=None):
        for _ in range(int(n)):
            self.tick(self.q, None, None, grip, F)


# ======================================================================
# collision checking on a private model copy
# ======================================================================
class Collide:
    def __init__(self, cell, margin=0.014):
        self.cell = cell
        m = copy.deepcopy(cell.model)
        self.m = m
        self.d = mujoco.MjData(m)
        self.margin = margin
        m.geom_margin[cell.robot_geoms] = margin
        ng = m.ngeom
        self.is_robot = np.zeros(ng, bool)
        self.is_robot[cell.robot_geoms] = True
        self.is_struct = np.zeros(ng, bool)
        self.is_struct[cell.structure_geoms] = True
        self.belt = int(cell._belt_surface)
        side = np.full(ng, -1, int)
        lroot = m.body("Left_Shoulder_Base_Link").id
        rroot = m.body("Right_Shoulder_Base_Link").id

        def root_of(b):
            while b > 0:
                if b == lroot:
                    return 1
                if b == rroot:
                    return 2
                b = m.body_parentid[b]
            return 0
        for g in cell.robot_geoms:
            side[g] = root_of(int(m.geom_bodyid[g]))
        self.side = side
        # collision bit masks of the private copy (speed): no pairs inside
        # one arm (the wrist self-contact is checked separately), arms vs
        # each other, the column, the station and the parcels
        for g in range(ng):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            if self.is_robot[g]:
                if side[g] == 1:
                    m.geom_contype[g], m.geom_conaffinity[g] = 2, 5
                elif side[g] == 2:
                    m.geom_contype[g], m.geom_conaffinity[g] = 4, 3
                else:
                    m.geom_contype[g], m.geom_conaffinity[g] = 8, 6
            else:
                m.geom_contype[g], m.geom_conaffinity[g] = 1, 7
        # wrist self-contact pairs (yaw link vs roll link) per arm
        self.wrist_pairs = []
        for sd in ("Left", "Right"):
            a = [g for g in cell.robot_geoms if m.body(int(m.geom_bodyid[g])).name == f"{sd}_Wrist_Yaw_Link"]
            b = [g for g in cell.robot_geoms if m.body(int(m.geom_bodyid[g])).name == f"{sd}_Wrist_Roll_Link"]
            for ga in a:
                for gb in b:
                    self.wrist_pairs.append((int(ga), int(gb)))
        self._ft = np.zeros(6)
        self.slot_of = cell.parcel_geom_slot
        pads = set()
        for s in ("L", "R"):
            for n in ("right_pad1", "right_pad2", "left_pad1", "left_pad2"):
                pads.add(m.geom(f"{s}_{n}").id)
        self.is_pad = np.zeros(ng, bool)
        self.is_pad[list(pads)] = True
        self.is_grip = np.zeros(ng, bool)
        for g in cell.robot_geoms:
            bn = m.body(int(m.geom_bodyid[g])).name
            if bn.startswith(("L_", "R_")):
                self.is_grip[g] = True
        self.arm_qadr = cell._arm_qadr
        self.gname = [mujoco.mj_id2name(m, 5, g) or str(g) for g in range(ng)]
        jid = cell._arm_jid
        self.q_lo = m.jnt_range[jid, 0] - 0.006
        self.q_hi = m.jnt_range[jid, 1] + 0.006

    def refresh(self):
        """Copy the slot fields that change at spawn time and the world
        state (other parcels, gripper joints) from the cell."""
        c = self.cell
        for f in ("geom_size", "geom_pos", "geom_rbound", "geom_aabb",
                  "body_mass", "body_ipos"):
            getattr(self.m, f)[:] = getattr(c.model, f)
        self.m.geom_margin[c.robot_geoms] = self.margin
        self.d.qpos[:] = c.data.qpos

    def set_parcel(self, slot, pos, R):
        a = self.cell._slot_qadr[slot]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
        self.d.qpos[a:a + 3] = pos
        self.d.qpos[a + 3:a + 7] = q

    def check(self, q14, parcels=(), contact_ok=(), detail=False):
        """parcels: [(slot, pos, R)] poses to place; contact_ok: sides
        ("Left"/"Right") whose pads may touch the parcel. Returns a dict of
        minimum distances per class."""
        m, d = self.m, self.d
        d.qpos[self.arm_qadr] = q14
        for slot, pos, R in parcels:
            self.set_parcel(slot, pos, R)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        out = {"struct": 1.0, "belt": 1.0, "arms": 1.0, "parcel": 1.0,
               "self": 1.0}
        for ga, gb in self.wrist_pairs:
            dist = mujoco.mj_geomDistance(m, d, ga, gb, 0.01, self._ft)
            if dist < out["self"]:
                out["self"] = float(dist)
        worst = {}
        qq = np.asarray(q14)
        lim = float(min(np.min(qq - self.q_lo), np.min(self.q_hi - qq)))
        if lim < 0:
            out["self"] = lim - 1.0           # outside the joint range
            if detail:
                worst["self"] = ("joint_limit", int(np.argmin(np.minimum(qq - self.q_lo, self.q_hi - qq))), lim)
                out["worst"] = worst
            return out
        n = d.ncon
        if n == 0:
            return out
        g = np.asarray(d.contact.geom[:n])
        dist = np.asarray(d.contact.dist[:n])
        r = self.is_robot[g]
        for k in np.nonzero(r[:, 0] | r[:, 1])[0]:
            g1, g2 = int(g[k, 0]), int(g[k, 1])
            if r[k, 0] and r[k, 1]:
                s1, s2 = self.side[g1], self.side[g2]
                if s1 == s2 and self.is_grip[g1] and self.is_grip[g2]:
                    continue
                cls = "arms" if (s1 != s2 and s1 > 0 and s2 > 0) else (
                    "self" if s1 == s2 else "arms")
                rg = g1
            else:
                rg, og = (g1, g2) if r[k, 0] else (g2, g1)
                if self.slot_of[og] >= 0:
                    sd = "Left" if self.side[rg] == 1 else "Right"
                    # a palm in contact: any part of its gripper may touch
                    # (pads, and with a tilted palm the finger links)
                    if self.is_grip[rg] and sd in contact_ok:
                        continue
                    cls = "parcel"
                elif og == self.belt:
                    cls = "belt"
                elif self.is_struct[og]:
                    cls = "struct"
                else:
                    continue
            dv = float(dist[k])
            if dv < out[cls]:
                out[cls] = dv
                if detail:
                    worst[cls] = (self.gname[g1], self.gname[g2], dv)
        if detail:
            out["worst"] = worst
        return out

    def wrist_ok(self, q14, clear=0.004):
        m, d = self.m, self.d
        d.qpos[self.arm_qadr] = q14
        qq = np.asarray(q14)
        if np.any(qq < self.q_lo) or np.any(qq > self.q_hi):
            return False
        mujoco.mj_kinematics(m, d)
        for ga, gb in self.wrist_pairs:
            if mujoco.mj_geomDistance(m, d, ga, gb, 0.01, self._ft) < clear:
                return False
        return True

    @staticmethod
    def ok(res, belt_clear=None, contact=False):
        belt_clear = CLEAR_BELT if belt_clear is None else belt_clear
        pc = 0.001 if contact else CLEAR_PARCEL
        return (res["struct"] >= CLEAR_STRUCT and res["belt"] >= belt_clear
                and res["arms"] >= CLEAR_ARMS and res["parcel"] >= pc
                and res["self"] >= 0.0015)


# ======================================================================
# IK helpers
# ======================================================================
class IK:
    def __init__(self, model):
        self.kin = {s: ArmKin(model, s) for s in SIDES}
        self.rng = np.random.default_rng(7)

    def solve(self, side, pos, R, seeds=(), n_rand=0, margin=0.02,
              q_rest=None, iters=200):
        """Several IK solutions (list) of one pose."""
        k = self.kin[side]
        qr = Q_REST[side] if q_rest is None else q_rest
        cand = list(seeds) + [qr]
        cand += [self.rng.uniform(k.lo, k.hi) for _ in range(n_rand)]
        sols = []
        for s in cand:
            q, ok, ep, er = k.ik(pos, R=R, q0=s, q_rest=qr, iters=iters,
                                 margin=margin, null_gain=0.05)
            if ok and not any(np.max(np.abs(q - o)) < 0.05 for o in sols):
                sols.append(q)
        return sols

    def trace_seeded(self, side, poses, seeds, q_start, max_jump=0.25,
                     margin=0.006):
        """IK per pose seeded from a reference path (a re-plan of a known
        chain for slightly moved targets); continuity is checked."""
        k = self.kin[side]
        q = np.asarray(q_start, float).copy()
        out = np.empty((len(poses), 7))
        for i, ((p, R), sd) in enumerate(zip(poses, seeds)):
            qn, ok, ep, er = k.ik(p, R=R, q0=sd, iters=80, pos_tol=4e-4,
                                  rot_tol=4e-3, null_gain=0.0, margin=margin)
            if not ok or np.max(np.abs(qn - q)) > max_jump:
                qn2, ok2, ep2, er2 = k.ik(p, R=R, q0=q, iters=120, pos_tol=6e-4,
                                          rot_tol=6e-3, null_gain=0.0,
                                          margin=margin)
                if ok2 and np.max(np.abs(qn2 - q)) <= max_jump:
                    qn, ok = qn2, True
                else:
                    self._last_ok = i
                    raise PlanFail(f"{side} seeded IK fail {i}/{len(poses)}")
            q = qn
            out[i] = q
        self._last_ok = len(poses)
        return out

    def trace_partial(self, side, poses, q_start, max_jump=0.25, limit_gain=0.02):
        """Like trace, but returns the samples reached before the first
        failure (possibly none)."""
        out = []
        q = np.asarray(q_start, float)
        for pose in poses:
            try:
                W = self.trace(side, [pose], q, max_jump=max_jump, limit_gain=limit_gain)
            except PlanFail:
                break
            q = W[0]
            out.append(q)
        return np.array(out).reshape(-1, 7)

    def trace(self, side, poses, q_start, max_jump=0.12, margin=0.006,
              limit_gain=0.0):
        """Continuation along poses [(pos, R)] (or (pos, R, "axis"): only
        the tool axis R[:, 2] is held, the roll about it is free); returns
        (n, 7) or raises."""
        k = self.kin[side]
        q = np.asarray(q_start, float).copy()
        out = np.empty((len(poses), 7))
        for i, pose in enumerate(poses):
            p, R = pose[0], pose[1]
            if len(pose) > 2 and pose[2] == "axis":
                kw = dict(R=None, axis=R[:, 2])
            else:
                kw = dict(R=R)
            qn, ok, ep, er = k.ik(p, q0=q, iters=60, pos_tol=4e-4,
                                  rot_tol=4e-3, null_gain=0.0, margin=margin,
                                  limit_gain=limit_gain, **kw)
            if not ok:
                qn, ok, ep, er = k.ik(p, q0=q, iters=200, pos_tol=8e-4,
                                      rot_tol=8e-3, null_gain=0.0,
                                      margin=margin, limit_gain=limit_gain, **kw)
            if not ok and ep < 2.5e-3 and er < 0.02:
                ok = True                     # accept (servo force feedforward covers it)
            if not ok:
                raise PlanFail(f"{side} IK fail {i}/{len(poses)} "
                               f"({ep:.4f} m, {er:.3f} rad)")
            if np.max(np.abs(qn - q)) > max_jump:
                raise PlanFail(f"{side} jump {np.max(np.abs(qn - q)):.2f} at "
                               f"{i}/{len(poses)}")
            q = qn
            out[i] = q
        return out


# ======================================================================
# time scaling
# ======================================================================
def smooth_waypoints(W, win=7):
    """Savitzky-Golay smoothing of joint waypoints along the path index
    (IK residual jitter), endpoints kept exactly."""
    W = np.asarray(W, float)
    n = len(W)
    if n < 7:
        return W
    from scipy.signal import savgol_filter
    w = min(win, n - (1 - n % 2))
    if w < 5:
        return W
    Ws = savgol_filter(W, w, 3, axis=0, mode="interp")
    # keep the endpoints: blend the end corrections out linearly
    e0, e1 = W[0] - Ws[0], W[-1] - Ws[-1]
    u = np.linspace(0, 1, n)[:, None]
    return Ws + (1 - u) * e0 + u * e1


def retime(W, vmax=V_PLAN, amax=A_PLAN, tmin=0.2, smooth=True):
    """W: (n, d) joint waypoints along a path (row 0 = start), uniformly
    spaced in the path parameter. Returns per-tick (q, qd, qdd) with a
    minimum-jerk time law, peak joint speed <= vmax, acceleration about
    <= amax."""
    W = np.asarray(W, float)
    if len(W) < 2 or np.max(np.abs(W - W[0])) < 1e-9:
        n = max(1, int(round(tmin / DT)))
        Q = np.repeat(W[-1:], n, axis=0)
        return Q, np.zeros_like(Q), np.zeros_like(Q)
    if smooth:
        W = smooth_waypoints(W)
    s = np.linspace(0.0, 1.0, len(W))
    if len(W) > 2:
        sp = CubicSpline(s, W, axis=0, bc_type="natural")
    else:
        sp = None
    sd_ = np.linspace(0, 1, 400)
    if sp is not None:
        g1 = float(np.max(np.abs(sp(sd_, 1))))
        g2 = float(np.max(np.abs(sp(sd_, 2))))
    else:
        g1 = float(np.max(np.abs(W[1] - W[0])))
        g2 = 0.0
    T = max(tmin, 1.875 * g1 / vmax, math.sqrt(5.77 * g1 / amax),
            math.sqrt(3.5 * g2 / amax) if g2 > 0 else 0.0)
    for _ in range(6):
        n = max(2, int(math.ceil(T / DT)))
        T = n * DT
        tt = np.arange(1, n + 1) / n
        ss = min_jerk(tt)
        sd = min_jerk_d(tt) / T
        sdd = min_jerk_dd(tt) / (T * T)
        if sp is None:
            dW = W[1] - W[0]
            Q = W[0] + np.outer(ss, dW)
            QD = np.outer(sd, dW)
            QDD = np.outer(sdd, dW)
        else:
            Q = sp(ss)
            d1 = sp(ss, 1)
            d2 = sp(ss, 2)
            QD = d1 * sd[:, None]
            QDD = d2 * (sd[:, None] ** 2) + d1 * sdd[:, None]
        vm = float(np.max(np.abs(QD)))
        am = float(np.max(np.abs(QDD)))
        if vm <= vmax * 1.02 and am <= amax * 1.25:
            break
        T *= min(1.5, max(1.08, vm / vmax, math.sqrt(am / amax) * 0.9))
    Q[-1] = W[-1]
    return Q, QD, QDD


def retime_var(W, vmax=V_PLAN, amax=A_PLAN, tmin=0.2, smooth=True):
    """Variable-speed time scaling along the path: the path speed follows
    vmax / max_j |dq_j/ds| (slow where a joint is fast), with acceleration
    ramps at both ends and a smoothed speed profile. Falls back to
    ``retime`` for short paths."""
    W = np.asarray(W, float)
    if len(W) < 8 or np.max(np.abs(W - W[0])) < 1e-9:
        return retime(W, vmax, amax, tmin, smooth)
    if smooth:
        W = smooth_waypoints(W)
    s = np.linspace(0.0, 1.0, len(W))
    sp = CubicSpline(s, W, axis=0, bc_type="natural")
    N = 600
    sg = np.linspace(0, 1, N)
    g = np.max(np.abs(sp(sg, 1)), axis=1) + 1e-6
    v_lim = vmax / g
    # conservative local minimum, then smoothing
    k = 15
    pad = np.pad(v_lim, k, mode="edge")
    vmin = np.array([pad[i:i + 2 * k + 1].min() for i in range(N)])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    vs = np.convolve(np.pad(vmin, k, mode="edge"), ker, mode="valid")
    vs = np.minimum(vs, vmin * 1.05)
    a_s = amax / float(np.max(g))
    ds = 1.0 / (N - 1)
    # forward / backward acceleration passes (sdot^2 changes <= 2 a ds)
    vf = vs.copy()
    vf[0] = 0.0
    for i in range(1, N):
        vf[i] = min(vf[i], math.sqrt(vf[i - 1] ** 2 + 2 * a_s * ds))
    vf[-1] = 0.0
    for i in range(N - 2, -1, -1):
        vf[i] = min(vf[i], math.sqrt(vf[i + 1] ** 2 + 2 * a_s * ds))
    vf = np.maximum(vf, 1e-4)
    dt_ = 2 * ds / (vf[1:] + vf[:-1])
    tg = np.concatenate([[0.0], np.cumsum(dt_)])
    T = max(tmin, float(tg[-1]))
    n = max(2, int(math.ceil(T / DT)))
    tq = np.arange(1, n + 1) * (tg[-1] / n)
    ss = np.interp(tq, tg, sg)
    ss[-1] = 1.0
    Q = sp(ss)
    Qf = np.vstack([W[0][None], Q])
    QD = np.gradient(Qf, tg[-1] / n, axis=0)
    QDD = np.gradient(QD, tg[-1] / n, axis=0)
    QD[-1] = 0.0
    Q[-1] = W[-1]
    return Q, QD[1:], QDD[1:]


def quintic_move(q0, v0, q1, vmax=V_PLAN, amax=A_PLAN, tmin=0.5):
    """Joint move from (q0, v0, acc 0) to (q1, 0, 0), quintic per joint,
    duration found so that the peak speed <= vmax and acc about <= amax."""
    q0, v0, q1 = (np.asarray(x, float) for x in (q0, v0, q1))
    h = q1 - q0
    T = max(tmin, 1.875 * float(np.max(np.abs(h))) / vmax)
    for _ in range(30):
        n = max(2, int(math.ceil(T / DT)))
        T = n * DT
        a3 = (20 * h - 12 * v0 * T) / (2 * T ** 3)
        a4 = (-30 * h + 16 * v0 * T) / (2 * T ** 4)
        a5 = (12 * h - 6 * v0 * T) / (2 * T ** 5)
        t = (np.arange(1, n + 1) * DT)[:, None]
        Q = q0 + v0 * t + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
        QD = v0 + 3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4
        QDD = 6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3
        if float(np.max(np.abs(QD))) <= vmax * 1.02 and float(np.max(np.abs(QDD))) <= amax * 1.3:
            break
        T *= 1.1
    Q[-1] = q1
    return Q, QD, QDD


def diff_samples(Q, q_prev):
    """Velocities / accelerations of per-tick samples (time-locked paths)."""
    full = np.vstack([q_prev[None], Q])
    qd = np.gradient(full, DT, axis=0)
    qdd = np.gradient(qd, DT, axis=0)
    return qd[1:], qdd[1:]


# ======================================================================
# plan containers
# ======================================================================
@dataclass
class Seg:
    name: str
    Q: np.ndarray                 # (n, 14)
    QD: np.ndarray
    QDD: np.ndarray
    grip: tuple = (0.0, 0.0)
    F: list = None                # per tick {side: F_world} or None
    parcel: list = None           # per tick (pos, R) predicted, or None
    contact: tuple = ()           # sides whose pads touch the parcel
    kind: str = "free"            # free | timed | wait


@dataclass
class JobParams:
    op: str                       # flip
    theta: float                  # total rotation about +y (rad)
    theta0: float = 0.0           # initial pad roll (world rotation about y)
    tilt: float = 0.0
    dxs: float = 0.0              # squeeze line x offset from the centre
    zf: float = 0.5               # squeeze line height (fraction of H)
    lift_extra: float = 0.0
    y_c: float = 0.0              # catch position (parcel centre y)
    over: float = 0.0             # extra roll margin to check (rad)
    settle: bool = False          # rotate short of the target, gravity settle
    theta0R: float = None         # right palm's initial roll (None: = theta0)

    def key(self):
        return (round(self.theta, 3), round(self.theta0, 3), round(self.tilt, 3),
                round(self.dxs, 3), round(self.zf, 2), round(self.lift_extra, 3),
                round(self.y_c, 3), self.settle)


@dataclass
class ParcelInfo:
    slot: int
    uid: int
    kind: str
    size: tuple
    mass: float
    pos: np.ndarray
    vel: np.ndarray
    R: np.ndarray
    dims: np.ndarray              # world x, y, z extents (aligned)
    yaw_err: float
    label: str
    label_err_deg: float
    label_angle_deg: float
    label_n: np.ndarray


# ======================================================================
# the planner
# ======================================================================
# gripper collision envelope in the tool frame (open): x, y, z bounds
def _box(hx, hy, z0, z1):
    return [[sx * hx, sy * hy, sz] for sx in (-1, 1) for sy in (-1, 1)
            for sz in (z0, z1)]


# gripper collision envelope in the tool frame (open), from the measured
# collision geoms: base / mount, finger linkages, pads
GRIP_BOX = np.array(_box(0.0424, 0.0375, -0.163, -0.062)
                    + _box(0.0741, 0.0195, -0.107, -0.009)
                    + _box(0.0506, 0.011, -0.040, -0.0025))
PAD_BOX = np.array([[sx * 0.0506, sy * 0.011, sz]
                    for sx in (-1, 1) for sy in (-1, 1)
                    for sz in (-0.040, -0.0025)])


def lowest_point(R, p):
    """World z of the lowest point of the gripper envelope at pose (p, R)."""
    return float(np.min((GRIP_BOX @ R.T)[:, 2]) + p[2])


class Planner:
    """Builds and checks the per-tick plan of one job."""

    def __init__(self, ctl):
        self.ctl = ctl
        self.cell = ctl.cell
        self.st = ctl.cell.station
        self.zb = self.st.belt_height
        self.near_edge = self.st.belt_center_x - self.st.half_width
        self.ik = IK(ctl.cell.model)
        self.dyn = ctl.ex.dyn
        self.col = None

    # -------------------------------------------------------------- utils
    def fresh_collider(self):
        if self.col is None:
            self.col = Collide(self.cell)
        self.col.refresh()
        return self.col

    def pad_fit(self, Lx, H, theta0, zs_rel, xs_rel, kind):
        """Do both pads lie on the squeeze face (with margins)? zs_rel /
        xs_rel: squeeze line relative to the face centre."""
        c, s = math.cos(theta0), math.sin(theta0)
        # pad centres at +-PAD_X along tool x rotated by theta0 in x-z:
        # tool x at theta0 is (cos, 0, -sin) for the left palm convention
        ok = True
        zlo = -(H / 2) + (0.075 * H if kind == "soft" else 0.0) + 0.004
        zhi = H / 2 - (0.075 * H if kind == "soft" else 0.0) - 0.004
        for sgn in (-1, 1):
            px = xs_rel + sgn * PAD_X * c
            pz = zs_rel - sgn * PAD_X * s
            # pad patch extent: half thickness along tool x, half width
            # along tool y (= world z for horizontal pads)
            ex = abs(PAD_HALF_T * c) + abs(PAD_HALF_W * s)
            ez = abs(PAD_HALF_T * s) + abs(PAD_HALF_W * c)
            if px - ex < -Lx / 2 + 0.006 or px + ex > Lx / 2 - 0.006:
                ok = False
            if pz - ez < zlo or pz + ez > zhi:
                ok = False
        return ok

    # -------------------------------------------------------------- frames
    def job_geometry(self, pi, prm, stop_pose=None):
        """Static geometry of a job: stop pose of the parcel, squeeze line,
        palm frames, contact TCPs, lift / set-down heights."""
        Lx, Ly, H = (float(v) for v in pi.dims)
        zb = self.zb
        if stop_pose is None:
            c0 = np.array([pi.pos[0], prm.y_c, zb + H / 2])
            Rp0 = pi.R.copy()
            yaw = pi.yaw_err
        else:
            c0, Rp0, yaw = stop_pose
        ay = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        ax = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        zs = zb + prm.zf * H
        xs = c0[0] + prm.dxs
        s0 = np.array([xs, c0[1], zs]) + 0.0 * ay
        # squeeze line on the parcel's centre line in y
        s0 = s0 + ay * float((c0 - s0) @ ay)
        th0R = prm.theta0 if prm.theta0R is None else prm.theta0R
        R0 = {"Left": palm_frame("Left", prm.theta0, yaw, prm.tilt),
              "Right": palm_frame("Right", th0R, yaw, prm.tilt)}
        dface = Ly / 2 - PAD_TIP
        pc = {"Left": s0 + ay * dface, "Right": s0 - ay * dface}
        # corners (x-z) relative to the squeeze line
        rel = np.array([c0[0] - xs, c0[2] - zs])
        cor = corners_xz(Lx, H) + rel
        r = float(np.max(np.linalg.norm(cor, axis=1)))
        lift_z = max(zs + 0.03, zb + r + 0.018) + prm.lift_extra
        th = prm.theta
        c, s = math.cos(th), math.sin(th)
        rot = np.array([[c * x + s * z, -s * x + c * z] for x, z in cor])
        low = float(np.min(rot[:, 1]))
        z_set = zb - low + 0.0005
        x_near_new = xs + float(np.min(rot[:, 0]))
        x_far_new = xs + float(np.max(rot[:, 0]))
        # a rotation short of 180 / 90 deg: the parcel settles about its
        # lowest corner onto the target face; footprint after the settle
        x_near_new, x_far_new = settled_span(rot, xs, Lx if abs(abs(th) - math.pi) < 0.9 else H)
        lo_x = self.near_edge + 0.006
        hi_x = self.st.belt_center_x + self.st.half_width - 0.012
        dx_set = 0.0
        if x_near_new < lo_x:
            dx_set = lo_x - x_near_new
        elif x_far_new > hi_x:
            dx_set = hi_x - x_far_new
        # dims after the rotation (world-aligned approx)
        ext = np.array([float(np.ptp(rot[:, 0])), Ly, float(np.ptp(rot[:, 1]))])
        return dict(Lx=Lx, Ly=Ly, H=H, c0=c0, Rp0=Rp0, yaw=yaw, ay=ay, ax=ax,
                    s0=s0, zs=zs, xs=xs, R0=R0, pc=pc, lift_z=lift_z,
                    z_set=z_set, dx_set=dx_set, dims_after=ext, r=r)

    def held_pose(self, geo, s_pt, th):
        """Palm TCP poses and the parcel pose for squeeze point s_pt and
        rotation th (rigid hold)."""
        Rr = Ry(th)
        out = {}
        for side in SIDES:
            out[side] = (s_pt + Rr @ (geo["pc"][side] - geo["s0"]),
                         Rr @ geo["R0"][side])
        par = (s_pt + Rr @ (geo["c0"] - geo["s0"]), Rr @ geo["Rp0"])
        return out, par

    # -------------------------------------------------------------- chain
    def manip_paths(self, geo, prm, ov=None, gap_r=0.025):
        ov = OVERPRESS if ov is None else ov
        """Cartesian paths (lists of per-sample targets) of the manipulation
        from the stop (left at the face, right hovering) to the end of the
        set-down. Returns ordered segment specs."""
        s0, ay = geo["s0"], geo["ay"]
        R0 = geo["R0"]
        pc = geo["pc"]
        zs, lift_z, z_set = geo["zs"], geo["lift_z"], geo["z_set"]
        th = prm.theta
        H = geo["H"]
        specs = []
        # right hover / descend / close; left close
        pR_side = pc["Right"] - ay * gap_r
        pR_hover = pR_side.copy()
        pR_hover[2] = max(pR_side[2], self.zb + H + 0.022 - (
            lowest_point(R0["Right"], pR_side) - pR_side[2]))
        pL_stop = pc["Left"] + ay * 0.0005
        pL_sq = pc["Left"] - ay * ov
        pR_sq = pc["Right"] + ay * ov

        def lin(a, b, n):
            return [a + (b - a) * u for u in np.linspace(0, 1, n + 1)[1:]]
        nd = max(6, int(np.linalg.norm(pR_side - pR_hover) / 0.01))
        specs.append(("descend", {"Right": [(p, R0["Right"]) for p in lin(pR_hover, pR_side, nd)]},
                      dict(contact=("Left",), vmax=2.2)))
        nc = 8
        specs.append(("close", {"Left": [(p, R0["Left"]) for p in lin(pL_stop, pL_sq, nc)],
                                "Right": [(p, R0["Right"]) for p in lin(pR_side, pR_sq, nc)]},
                      dict(contact=("Left", "Right"), vmax=0.8, tmin=0.4)))
        # lift (both): squeeze point s0 -> lift
        s1 = s0.copy(); s1[2] = lift_z
        nl = max(6, int((lift_z - zs) / 0.008))
        lift = []
        for u in np.linspace(0, 1, nl + 1)[1:]:
            sp = s0 + (s1 - s0) * u
            hp, par = self.held_pose(geo, sp, 0.0)
            lift.append((hp, par))
        specs.append(("lift", {s: [(h[s][0] - ay * SG[s] * ov, h[s][1]) for h, _ in lift] for s in SIDES},
                      dict(contact=SIDES, vmax=1.6, tmin=0.35, parcel=[p for _, p in lift])))
        # rotate about the squeeze line at lift height
        nr = max(12, int(abs(th) / math.radians(3)))
        rot = []
        for u in np.linspace(0, 1, nr + 1)[1:]:
            hp, par = self.held_pose(geo, s1, th * u)
            rot.append((hp, par))
        specs.append(("rotate", {s: [(h[s][0] - Ry(th * u) @ (ay * SG[s] * ov), h[s][1])
                                     for (h, _), u in zip(rot, np.linspace(0, 1, nr + 1)[1:])] for s in SIDES},
                      dict(contact=SIDES, vmax=V_PLAN, tmin=0.45 * abs(th), parcel=[p for _, p in rot])))
        # set down: lower with the x shift; stop where a gripper would
        # come closer than 6 mm to the belt (the parcel drops the rest)
        Rth = Ry(th)
        z_min = max(self.zb + 0.006 - lowest_point(Rth @ R0[sd], np.zeros(3)) for sd in SIDES)
        # TCP height limit -> squeeze-line height limit
        z_min_s = z_min - max(pc[sd][2] - s0[2] for sd in SIDES)
        z_set_eff = max(z_set, z_min_s)
        geo["drop"] = z_set_eff - z_set
        s2 = s1 + np.array([geo["dx_set"], 0.0, z_set_eff - lift_z])
        nw = max(6, int(np.linalg.norm(s2 - s1) / 0.008))
        low = []
        for u in np.linspace(0, 1, nw + 1)[1:]:
            sp = s1 + (s2 - s1) * u
            hp, par = self.held_pose(geo, sp, th)
            low.append((hp, par))
        specs.append(("lower", {s: [(h[s][0] - Rth @ (ay * SG[s] * ov), h[s][1]) for h, _ in low] for s in SIDES},
                      dict(contact=SIDES, vmax=1.4, tmin=0.4, parcel=[p for _, p in low])))
        return specs, dict(pR_hover=pR_hover, pL_stop=pL_stop, s1=s1, s2=s2)

    def roll_window(self, side, pos, Rh, step_deg=5.0, lim_deg=230.0,
                    n_seeds=6):
        """Roll windows of a palm at a fixed TCP position: frames
        R_y(theta) @ Rh reachable continuously from theta = 0 (no self
        contact, 0.05 rad from the joint limits). Returns the branches as
        [(theta_min, theta_max, thetas, Qs)] (one per start solution) or
        None."""
        key = (side, tuple(np.round(pos, 3)), tuple(np.round(Rh, 3).ravel()))
        cache = getattr(self, "_wcache", None)
        if cache is None:
            cache = self._wcache = {}
        if key in cache:
            return cache[key]
        k = self.ik.kin[side]
        sols = self.ik.solve(side, pos, Rh, seeds=[Q_REST[side]],
                             n_rand=n_seeds, margin=0.03)
        best = None
        st = math.radians(step_deg)
        col = self.fresh_collider() if self.col is None else self.col
        q14 = Q_PARK.copy()

        def self_ok(qa):
            q14[SL[side]] = qa
            return col.wrist_ok(q14)
        for q0 in sols[:n_seeds]:
            if not self_ok(q0):
                continue
            th_list, q_list = [0.0], [q0]
            for direction in (1, -1):
                q = q0.copy()
                th = 0.0
                while abs(th) < math.radians(lim_deg):
                    th2 = th + direction * st
                    qn, ok, ep, er = k.ik(pos, R=Ry(th2) @ Rh, q0=q, iters=60,
                                          pos_tol=1.5e-3, rot_tol=0.012,
                                          null_gain=0.0, margin=0.05,
                                          limit_gain=0.03)
                    if not ok or np.max(np.abs(qn - q)) > 0.30 or not self_ok(qn):
                        break
                    q, th = qn, th2
                    th_list.append(th)
                    q_list.append(q)
            order = np.argsort(th_list)
            ths = np.array(th_list)[order]
            Qs = np.array(q_list)[order]
            w = (ths[0], ths[-1], ths, Qs)
            if best is None:
                best = []
            # keep distinct branches (different window ends)
            if not any(abs(w[0] - b[0]) < 1e-6 and abs(w[1] - b[1]) < 1e-6 for b in best):
                best.append(w)
        cache[key] = best
        return best

    def rotation_options(self, geo, prm, s1, settle_min=math.radians(115)):
        """Common roll windows of the two arms at the lift point, over the
        branch pairs: [(a, b, bl, br)] widest first (a, b = common window)."""
        th = prm.theta
        m = math.radians(3)
        wins = {}
        info = {}
        for side in SIDES:
            pos = s1 + (geo["pc"][side] - geo["s0"])
            Rh = palm_frame(side, 0.0, geo["yaw"], prm.tilt)
            w = self.roll_window(side, pos, Rh)
            if not w:
                return [], info
            wins[side] = w
            info[side] = [(round(math.degrees(b[0])), round(math.degrees(b[1]))) for b in w]
        opts = []
        need = settle_min if abs(abs(th) - math.pi) < 1e-6 else abs(th)
        # each palm rolls inside its own window from its own start roll
        for bl in wins["Left"]:
            for br in wins["Right"]:
                wl, wr = bl[1] - bl[0], br[1] - br[0]
                if min(wl, wr) - 2 * m - prm.over >= need:
                    opts.append((min(wl, wr), bl, br))
        opts.sort(key=lambda o: -o[0])
        # windows at the catch height (start roll) and at the set-down
        # height (end roll): unions of the branch intervals per arm
        self._geo_w = geo
        self._prm_w = prm
        return opts, info

    def set_height(self, geo, th):
        """Squeeze-line height when the parcel rotated by th touches the
        belt with its lowest corner."""
        rel = np.array([geo["c0"][0] - geo["xs"], geo["c0"][2] - geo["zs"]])
        cor = corners_xz(geo["Lx"], geo["H"]) + rel
        c, s_ = math.cos(th), math.sin(th)
        low = min(-s_ * x + c * z for x, z in cor)
        return self.zb - low + 0.0005

    def low_ok(self, t0, th, m=math.radians(2)):
        """Start roll inside a catch-height window and end roll inside a
        set-down-height window, for both arms."""
        geo, prm = self._geo_w, self._prm_w
        z_set = self.set_height(geo, th)
        for side in SIDES:
            if not any(a + m <= t0 <= b - m for a, b in self._catch_w[side]):
                return False
            Rh = palm_frame(side, 0.0, geo["yaw"], prm.tilt)
            p_set = geo["pc"][side].copy()
            p_set[2] = z_set + (geo["pc"][side][2] - geo["s0"][2])
            ws = self.roll_window(side, np.round(p_set, 3), Rh, n_seeds=4) or []
            if not any(w[0] + m <= t0 + th <= w[1] - m for w in ws):
                return False
        return True

    def theta0_range(self, geo, prm, s1):
        opts, info = self.rotation_options(geo, prm, s1)
        if not opts:
            return None, info
        return (opts[0][1][0], opts[0][1][1]), info

    def overshoot_ok(self, side, q_rot_end, geo, prm, s1, ov=None):
        ov = OVERPRESS if ov is None else ov
        """Can the palm roll a further prm.over (both directions where
        asked) beyond the planned rotation? (correction margin)"""
        if prm.over <= 0:
            return True
        ay = geo["ay"]
        th = prm.theta
        sgn = math.copysign(1.0, th)
        poses = []
        n = max(3, int(prm.over / math.radians(3)))
        for u in np.linspace(0, 1, n + 1)[1:]:
            t2 = th + sgn * prm.over * u
            hp, _ = self.held_pose(geo, s1, t2)
            poses.append((hp[side][0] - Ry(t2) @ (ay * SG[side] * ov), hp[side][1]))
        try:
            W = self.ik.trace(side, poses, q_rot_end, max_jump=0.25)
        except PlanFail:
            return False
        col = self.fresh_collider() if self.col is None else self.col
        q14 = Q_PARK.copy()
        for qa in W:
            q14[SL[side]] = qa
            if not col.wrist_ok(q14):
                return False
        return True

    def solve_chain(self, side, specs, q_start):
        """IK continuation of one arm through the segment specs; returns
        {segment name: (n, 7) waypoints incl. the start row}."""
        out = {}
        q = np.asarray(q_start, float)
        for name, paths, opt in specs:
            if side not in paths:
                out[name] = q[None].copy()
                continue
            W = self.ik.trace(side, paths[side], q, max_jump=0.25)
            out[name] = np.vstack([q[None], W])
            q = W[-1]
        return out

    def assemble(self, specs, chains, q14_start, grip=(0.0, 0.0)):
        """Time-scale every free segment for both arms together."""
        segs = []
        q = np.asarray(q14_start, float).copy()
        for name, paths, opt in specs:
            nL = len(chains["Left"][name])
            nR = len(chains["Right"][name])
            n = max(nL, nR)
            # resample each arm's waypoints to n rows (by index fraction)
            W = np.empty((n, 14))
            for side, nn in (("Left", nL), ("Right", nR)):
                C = chains[side][name]
                if nn == n:
                    W[:, SL[side]] = C
                else:
                    src = np.linspace(0, 1, nn)
                    dst = np.linspace(0, 1, n)
                    for j in range(7):
                        W[:, SL[side].start + j] = np.interp(dst, src, C[:, j])
            rt = retime_var if name == "rotate" else retime
            Q, QD, QDD = rt(W, vmax=opt.get("vmax", V_PLAN), amax=A_PLAN,
                            tmin=opt.get("tmin", 0.25))
            par = opt.get("parcel")
            if par is not None and len(par) > 0:
                # map ticks to path samples by the path parameter
                idx = np.clip(np.round(np.linspace(0, len(par) - 1, len(Q))).astype(int), 0, len(par) - 1)
                par_t = [par[i] for i in idx]
            else:
                par_t = None
            segs.append(Seg(name, Q, QD, QDD, grip, None, par_t,
                            opt.get("contact", ()), "free"))
            q = Q[-1]
        return segs

    # ------------------------------------------------------------- checks
    def check_segments(self, segs, slot, parcel_static=None, every=1,
                       detail=False):
        col = self.fresh_collider()
        worst = {"struct": 1.0, "belt": 1.0, "arms": 1.0, "parcel": 1.0,
                 "self": 1.0}
        where = {}
        for sg in segs:
            for i in range(0, len(sg.Q), every):
                par = sg.parcel[i] if sg.parcel is not None else parcel_static
                pl = [] if par is None else [(slot, par[0], par[1])]
                res = col.check(sg.Q[i], pl, contact_ok=sg.contact, detail=detail)
                for k in worst:
                    if res[k] < worst[k]:
                        worst[k] = res[k]
                        where[k] = (sg.name, i, res.get("worst", {}).get(k))
                if not Collide.ok(res, contact=bool(sg.contact)):
                    return False, worst, where
        return True, worst, where

    def rate_ok(self, segs, u_prev, F_fn=None):
        """Commanded position rate of the plan (with the feedforward)."""
        peak = 0.0
        u0 = np.asarray(u_prev, float)
        tpk = 0.0
        for sg in segs:
            for i in range(len(sg.Q)):
                F = None if F_fn is None else F_fn(sg, i)
                u = self.ctl.ex.command(sg.Q[i], sg.QD[i], sg.QDD[i], F)
                r = float(np.max(np.abs(u - u0))) / DT
                peak = max(peak, r)
                tau = self.dyn.tau_ff(sg.Q[i], sg.QD[i], sg.QDD[i])
                tpk = max(tpk, float(np.max(np.abs(tau) / self.dyn.tlim)))
                u0 = u
        return peak, tpk


# ======================================================================
# the controller
# ======================================================================
T_ACC, T_DEC = 0.15, 0.30          # catch: match the belt speed, then brake


SQUEEZE_A, SQUEEZE_B = 16.0, 8.0      # N, N/kg (the cell sums both palms against its
                                      # parcel-damage limit)


def squeeze_force(m):
    return SQUEEZE_A + SQUEEZE_B * m


class Core:
    def __init__(self, cell, log=None):
        self.cell = cell
        self.ex = Exec(cell)
        self.pl = Planner(self)
        self.st = cell.station
        self.zb = self.st.belt_height
        self.near_edge = self.st.belt_center_x - self.st.half_width
        self.log = log or (lambda *a: None)
        self.results = []
        self.grip = (0.0, 0.0)

    # ------------------------------------------------------- observation
    def parcels(self):
        out = []
        tr = self.cell.truth
        for ps in tr.parcel_states():
            s = self.cell.slots[ps.slot]
            dims = aligned_dims(ps.size, ps.quat)
            _, yaw_err = world_dims(ps.size, ps.quat)
            lab, lerr = label_dir(ps.label_normal)
            out.append(ParcelInfo(ps.slot, ps.uid, ps.kind, ps.size,
                                  float(s.parcel.mass), ps.pos.copy(),
                                  ps.linvel.copy(), quat_mat(ps.quat), dims,
                                  yaw_err, lab, lerr,
                                  math.degrees(ps.label_angle),
                                  ps.label_normal.copy()))
        return out

    def parcel(self, slot):
        if not self.cell.slots[slot].active:
            return None
        for pi in self.parcels():
            if pi.slot == slot:
                return pi
        return None

    def belt_speed(self):
        return float(self.cell.truth.belt_speed()["nominal"])

    def others(self, slot):
        """Other active parcels near the station: (slot, pos, R) now."""
        out = []
        for pi in self.parcels():
            if pi.slot != slot and -1.3 < pi.pos[1] < 1.3:
                out.append((pi.slot, pi.pos.copy(), pi.R.copy()))
        return out

    @staticmethod
    def others_at(others, v, t):
        return [(s_, p_ + np.array([0.0, v * t, 0.0]), R_) for s_, p_, R_ in others]

    # ------------------------------------------------------ job choice
    @staticmethod
    def job_class(pi):
        lab = pi.label
        if lab == UP:
            return "pass"
        if lab in (YP, YN):
            return "pass_y"               # another module's primitive
        if lab == DOWN:
            return "down"
        if lab == XP:
            return "away"
        if lab == XN:
            return "toward"
        return "pass"

    def candidates(self, pi, cls):
        """Ordered job parameter combos for a class (the initial pad roll
        theta0 is chosen later from the arms' roll windows)."""
        Lx, Ly, H = (float(v) for v in pi.dims)
        ths = {"away": (-math.pi / 2,), "down": (-math.pi,),
               "toward": (math.pi / 2,)}[cls]
        over = {"away": math.radians(10), "down": math.radians(6),
                "toward": math.radians(10)}[cls]
        les = (0.04, 0.08) if cls == "down" else (0.0, 0.03)
        dxss = (0.0, -0.025, 0.02)
        ycs = (0.04, 0.0, -0.025)
        out = []
        tilts = (0.0,) if H >= 0.12 else ((0.0, math.radians(12)) if H >= 0.075
                                          else (math.radians(12), math.radians(20), 0.0))
        for tilt in tilts:
            # squeeze heights: the gripper housing at least 8 mm above the
            # belt at the catch (lowest point of the envelope), pads on the
            # face (checked later)
            low_rel = -lowest_point(palm_frame("Left", 0.0, 0.0, tilt), np.zeros(3))
            zf_min = (low_rel + 0.008) / H
            if cls == "down":
                zfs = [zf_min + 0.01, max(zf_min + 0.01, 0.5), zf_min + 0.07, zf_min + 0.14]
            else:
                zfs = [max(zf_min + 0.01, 0.5), zf_min + 0.01, max(zf_min + 0.01, 0.62)]
            zz = []
            for z in zfs:
                z = round(z, 3)
                if z <= 0.9 and z not in zz:
                    zz.append(z)
            zfs = zz[:3]
            for le in les:
                for zf in zfs:
                    for dxs in dxss:
                        for yc in ycs:
                            for th in ths:
                                out.append(JobParams("flip", th, 0.0, tilt, dxs, zf, le, yc, over=over))
        return out

    def settle_ok(self, pi, prm, th):
        """A rotation short of 180 deg must let gravity finish it: resting
        on the edge of the target face, the centre of mass lies over that
        face by at least 1.5 cm (plus the centre-of-mass offset)."""
        if abs(abs(prm.theta) - math.pi) > 1e-6:
            return True                       # not a down job: no settle
        alpha = math.pi - abs(th)
        if alpha < math.radians(1):
            return True
        Lx, Ly, H = (float(v) for v in pi.dims)
        lever = (Lx / 2) * math.cos(alpha) - (H / 2) * math.sin(alpha)
        # privileged: the true centre-of-mass offset along world x
        par = self.cell.slots[pi.slot].parcel
        com_w = pi.R @ np.asarray(par.com_offset if par is not None else (0, 0, 0), float)
        return lever > 0.02 + abs(float(com_w[0])) + 0.5 * abs(float(com_w[2]))

    def pick_theta0(self, pi, prm):
        """(theta0, theta, anchor seeds) candidates for a combo: inside a
        common roll window of both arms, with both pads on the face; a down
        job takes the largest rotation in [145, 180] deg that fits."""
        geo = self.pl.job_geometry(pi, prm)
        s1 = geo["s0"].copy()
        s1[2] = geo["lift_z"]
        opts, info = self.pl.rotation_options(geo, prm, s1)
        if not opts:
            return [], info
        Lx, Ly, H = (float(v) for v in pi.dims)
        zs_rel = prm.zf * H - H / 2
        m = math.radians(3)
        down = abs(abs(prm.theta) - math.pi) < 1e-6
        out = []
        def pick_side(side, a, b, th, mag):
            if th < 0:
                lo, hi = a + m + mag + prm.over, b
            else:
                lo, hi = a, b - m - mag - prm.over
            if lo > hi + 1e-9:
                return None
            dlo, dhi = math.degrees(lo), math.degrees(hi)
            grid = list(np.arange(math.ceil(dlo), math.floor(dhi) + 1, 3.0)) or [0.5 * (dlo + dhi)]
            grid.sort(key=lambda d: -min(d - dlo, dhi - d))
            for d in grid:
                t0 = math.radians(d)
                R0_ = palm_frame(side, t0, pi.yaw_err, prm.tilt)
                if lowest_point(R0_, np.array([0, 0, self.zb + prm.zf * H])) < self.zb + 0.005:
                    continue
                if self.pl.pad_fit(Lx, H, t0, zs_rel, prm.dxs, pi.kind):
                    return t0
            return None
        for width, bl, br in opts[:4]:
            if down:
                # leave room at the end of the roll (the release needs the
                # arms off their limits): prefer 15 deg spare, then less
                widthm = width - 2 * m - prm.over
                mags = []
                for spare in (math.radians(15), math.radians(8), 0.0):
                    top = min(math.pi, widthm - spare)
                    mags += [math.radians(d) for d in range(int(math.degrees(top)), 129, -5)]
            else:
                mags = [abs(prm.theta)]
            got = None
            for mag in mags:
                for sgn in ((-1.0, 1.0) if down else (math.copysign(1.0, prm.theta),)):
                    th = sgn * mag
                    if not self.settle_ok(pi, prm, th):
                        continue
                    tL = pick_side("Left", bl[0], bl[1], th, mag)
                    tR = pick_side("Right", br[0], br[1], th, mag) if tL is not None else None
                    if tL is not None and tR is not None:
                        got = (tL, tR, th)
                        break
                if got:
                    break
            if got is None:
                continue
            tL, tR, th = got
            seeds = {"Left": (bl[2], bl[3]), "Right": (br[2], br[3])}
            out.append(((tL, tR), th, seeds))
            if len(out) >= 2:
                break
        return out, info

    # ------------------------------------------------------ planning
    def plan_job(self, pi, prm, v, q_now, t_lead, seeds=None):
        """Full plan from the current posture: approach, catch, manipulation.
        t_lead: time until the parcel's face would reach the catch start.
        Returns a dict or raises PlanFail."""
        pl = self.pl
        geo = pl.job_geometry(pi, prm)
        specs, key = pl.manip_paths(geo, prm)
        R0 = geo["R0"]
        ay = geo["ay"]
        # catch path of the left palm (time-locked samples)
        pL_stop = key["pL_stop"]
        d_catch = v * (T_ACC + T_DEC) / 2
        pL_wait = pL_stop - ay * d_catch
        n1, n2 = int(round(T_ACC / DT)), int(round(T_DEC / DT))
        ys, yy = [], 0.0
        for i in range(n1 + n2):
            t = (i + 1) * DT
            vp = v * t / T_ACC if i < n1 else v * max(0.0, 1 - (t - T_ACC) / T_DEC)
            yy += vp * DT
            ys.append(min(yy, d_catch))
        catch_poses = [(pL_wait + ay * y, R0["Left"]) for y in ys]
        pR_hover = key["pR_hover"]
        # one pose list per arm through the whole chain; anchored at the
        # end of the rotation (the most constrained pose): IK solutions
        # there, traced backward to the chain start and forward to the end
        chains = {}
        errs = []
        lower_cut = {}
        for side in SIDES:
            seq = []                       # (segment name, pose)
            if side == "Left":
                seq.append(("start", (pL_wait, R0["Left"])))
                seq += [("catch", ps) for ps in catch_poses]
            else:
                seq.append(("start", (pR_hover, R0["Right"])))
            for name, paths, opt in specs:
                if side in paths:
                    seq += [(name, ps) for ps in paths[side]]
            names = [nm for nm, _ in seq]
            poses = [ps for _, ps in seq]
            irot = [i for i, nm in enumerate(names) if nm == "rotate"]
            i0r, ia = irot[0], irot[-1]
            br = seeds.get(side) if seeds else None      # (thetas, Qs) branch
            got = None
            if br is not None:
                # anchor at the rotation start, the rotation follows the
                # window branch (IK seeded per sample from the sweep)
                ths_b, Qs_b = br
                def qb(t):
                    return np.array([np.interp(t, ths_b, Qs_b[:, j]) for j in range(7)])
                nr = len(irot)
                th0s_ = prm.theta0 if (side == "Left" or prm.theta0R is None) else prm.theta0R
                t_rot = [th0s_ + prm.theta * (k + 1) / nr for k in range(nr)]
                anchors = pl.ik.solve(side, poses[i0r - 1][0], poses[i0r - 1][1],
                                      seeds=[qb(th0s_)], n_rand=0, margin=0.01)
                anchors.sort(key=lambda q: float(np.max(np.abs(q - qb(th0s_)))))
                # (B) forward from the chain start (wait / hover pose): IK
                # there from several seeds, plain continuation to the
                # rotation start, the rotation seeded from the branch
                fw_seeds = pl.ik.solve(side, poses[0][0], poses[0][1],
                                       seeds=[qb(th0s_), Q_REST[side]], n_rand=4, margin=0.02)
                best_short = 0.0
                for q0f in fw_seeds[:4]:
                    if got is not None:
                        break
                    try:
                        pre = pl.ik.trace(side, poses[1:i0r], q0f, max_jump=0.25, limit_gain=0.02)
                        try:
                            rot = pl.ik.trace_seeded(side, [poses[i] for i in irot],
                                                     [qb(t) for t in t_rot], pre[-1], max_jump=0.25)
                        except PlanFail as e1:
                            k_ok = getattr(pl.ik, "_last_ok", 0)
                            raise RotShort(f"{side} roll short {k_ok}/{len(irot)}: {e1}",
                                           k_ok / max(1, len(irot)))
                        fwd = pl.ik.trace_partial(side, poses[ia + 1:], rot[-1])
                        if len(poses) - ia - 1 > 0 and len(fwd) < 0.4 * (len(poses) - ia - 1):
                            raise PlanFail(f"{side} lower reach {len(fwd)}/{len(poses) - ia - 1}")
                        Wall = np.vstack([q0f[None], pre, rot, fwd])
                        if side == "Left":
                            ic = [i for i, nm in enumerate(names) if nm == "catch"]
                            if np.max(np.abs(np.diff(Wall[[0] + ic], axis=0))) > 0.10:
                                raise PlanFail("Left catch jump")
                        if not pl.overshoot_ok(side, Wall[ia], geo, prm, key["s1"]):
                            raise PlanFail(f"{side} no roll margin")
                        got = (names, Wall)
                    except PlanFail as e:
                        errs.append("B " + str(e))
                        if isinstance(e, RotShort):
                            best_short = max(best_short, e.frac)
                for qa in (anchors[:2] if got is None else []):
                    try:
                        back = pl.ik.trace(side, poses[:i0r - 1][::-1], qa, max_jump=0.25,
                                           limit_gain=0.02)[::-1] if i0r - 1 > 0 else np.zeros((0, 7))
                        rot = pl.ik.trace_seeded(side, [poses[i] for i in irot],
                                                 [qb(t) for t in t_rot], qa, max_jump=0.25)
                        fwd = pl.ik.trace(side, poses[ia + 1:], rot[-1], max_jump=0.25,
                                          limit_gain=0.02) if ia + 1 < len(poses) else np.zeros((0, 7))
                        Wall = np.vstack([back, qa[None], rot, fwd])
                        if side == "Left":
                            ic = [i for i, nm in enumerate(names) if nm == "catch"]
                            if np.max(np.abs(np.diff(Wall[[0] + ic], axis=0))) > 0.10:
                                raise PlanFail("Left catch jump")
                        if not pl.overshoot_ok(side, Wall[ia], geo, prm, key["s1"]):
                            raise PlanFail(f"{side} no roll margin")
                        got = (names, Wall)
                        break
                    except PlanFail as e:
                        errs.append(str(e))
            else:
                anchors = pl.ik.solve(side, poses[ia][0], poses[ia][1],
                                      seeds=[Q_REST[side]], n_rand=10, margin=0.03)

                def cost(q):
                    k = pl.ik.kin[side]
                    xn = (q - (k.lo + k.hi) / 2) / ((k.hi - k.lo) / 2)
                    return float(np.sum(xn ** 8)) + 0.5 * abs(q[5])
                anchors = sorted(anchors, key=cost)[:6]
                for qa in anchors:
                    try:
                        back = pl.ik.trace(side, poses[:ia][::-1], qa, max_jump=0.25,
                                           limit_gain=0.02)[::-1]
                        fwd = pl.ik.trace(side, poses[ia + 1:], qa, max_jump=0.25,
                                          limit_gain=0.02) if ia + 1 < len(poses) else np.zeros((0, 7))
                        Wall = np.vstack([back, qa[None], fwd])
                        if side == "Left":
                            ic = [i for i, nm in enumerate(names) if nm == "catch"]
                            if np.max(np.abs(np.diff(Wall[[0] + ic], axis=0))) > 0.10:
                                raise PlanFail("Left catch jump")
                        if not pl.overshoot_ok(side, Wall[ia], geo, prm, key["s1"]):
                            raise PlanFail(f"{side} no roll margin")
                        got = (names, Wall)
                        break
                    except PlanFail as e:
                        errs.append(str(e))
            if got is None:
                if br is not None and best_short > 0:
                    raise RotShort(f"{side} chain: roll short", best_short)
                raise PlanFail(f"{side} chain: {errs[-1] if errs else 'no anchor IK'}")
            names, Wall = got
            n_low_all = sum(1 for nm in names if nm == "lower")
            n_low = len(Wall) - (len(names) - n_low_all)
            names = names[:len(Wall)]
            lower_cut[side] = (n_low, n_low_all)
            ch = {}
            q0 = Wall[0]
            prev = q0
            order = ["catch"] + [sp[0] for sp in specs]
            for nm in order:
                idx = [i for i, n_ in enumerate(names) if n_ == nm]
                if not idx:
                    ch[nm] = prev[None].copy()
                    continue
                ch[nm] = np.vstack([prev[None], Wall[idx]])
                prev = Wall[idx[-1]]
            chains[side] = (q0, ch)
            raw = getattr(self, "_raw_chain", {})
            raw[side] = (names, Wall)
            self._raw_chain = raw
        n_ok = min(lower_cut[s_][0] for s_ in SIDES)
        n_all = lower_cut["Left"][1]
        # stop the set-down where any robot part would come too close to the
        # belt / structure / the other arm (the parcel drops the rest)
        colc = pl.fresh_collider()
        lowL = chains["Left"][1]["lower"]
        lowR = chains["Right"][1]["lower"]
        par_low = None
        for name, paths, opt in specs:
            if name == "lower":
                par_low = opt.get("parcel")
        for k_ in range(1, min(n_ok, len(lowL) - 1, len(lowR) - 1) + 1):
            q14 = np.concatenate([lowL[k_], lowR[k_]])
            pl_ = [(pi.slot, par_low[k_ - 1][0], par_low[k_ - 1][1])] if par_low else []
            rr = colc.check(q14, pl_, contact_ok=SIDES)
            if rr["belt"] < 0.006 or rr["struct"] < CLEAR_STRUCT or rr["arms"] < CLEAR_ARMS:
                n_ok = k_ - 1
                break
        if n_ok < n_all:
            for s_ in SIDES:
                ch = chains[s_][1]
                ch["lower"] = ch["lower"][:n_ok + 1]
            for i_, (name, paths, opt) in enumerate(specs):
                if name == "lower":
                    for s_ in list(paths):
                        paths[s_] = paths[s_][:n_ok]
                    if opt.get("parcel"):
                        opt["parcel"] = opt["parcel"][:n_ok]
            seg_len = np.linalg.norm(key["s2"] - key["s1"])
            geo["drop"] = geo.get("drop", 0.0) + seg_len * (1 - n_ok / max(1, n_all))
            flat_end = abs(abs(prm.theta) - math.pi) < 1e-6 or abs(abs(prm.theta) - math.pi / 2) < 1e-6
            if flat_end:
                lim_drop = SOFT_DROP if pi.kind == "soft" else MAX_DROP
            else:
                # a tilted (settling) parcel: a soft mailer may drop further
                lim_drop = SOFT_DROP if pi.kind == "soft" else 0.045
            if geo["drop"] > lim_drop:
                raise PlanFail(f"set-down drop {geo['drop']:.3f}")
        q_start = q_now.copy()
        q_start[SL["Left"]] = chains["Left"][0]
        q_start[SL["Right"]] = chains["Right"][0]
        # approach (joint move)
        Qa, QDa, QDDa = joint_move(q_now, q_start, vmax=V_PLAN, tmin=0.5, amax=A_PLAN)
        if len(Qa) * DT + 0.1 > t_lead:
            raise PlanFail(f"too late ({t_lead:.2f} s)")
        # alternative approach: joint move to a posture 8 cm above the start
        # pose (same branch), then a straight descent
        app2 = None
        try:
            q_high = q_start.copy()
            Wd = {}
            for side in SIDES:
                p0, R0_ = self.fk(side, q_start)
                ups = [(p0 + np.array([0.0, 0.0, 0.08 * u]), R0_) for u in np.linspace(0, 1, 9)[1:]]
                Wu = pl.ik.trace(side, ups, q_start[SL[side]], max_jump=0.25)
                q_high[SL[side]] = Wu[-1]
                Wd[side] = np.vstack([Wu[::-1][1:], q_start[SL[side]][None]])
            Q1, QD1, QDD1 = joint_move(q_now, q_high, vmax=V_PLAN, tmin=0.5, amax=A_PLAN)
            Wdesc = np.repeat(q_high[None], 9, axis=0)
            for side in SIDES:
                Wdesc[1:, SL[side]] = Wd[side]
            Q2, QD2, QDD2 = retime(Wdesc, vmax=1.6, amax=A_PLAN, tmin=0.35)
            if (len(Q1) + len(Q2)) * DT + 0.1 <= t_lead:
                app2 = Seg("approach", np.vstack([Q1, Q2]), np.vstack([QD1, QD2]),
                           np.vstack([QDD1, QDD2]), (0, 0), None, None, (), "free")
        except PlanFail:
            app2 = None
        # catch as a timed segment (left moves, right holds)
        C = chains["Left"][1]["catch"][1:]
        Qc = np.repeat(q_start[None], len(C), axis=0)
        Qc[:, SL["Left"]] = C
        QDc, QDDc = diff_samples(Qc, q_start)
        # manipulation (free segments)
        chs = {s: chains[s][1] for s in SIDES}
        q_stop = Qc[-1]
        msegs = pl.assemble(specs, chs, q_stop)
        app = Seg("approach", Qa, QDa, QDDa, (0, 0), None, None, (), "free")
        cat = Seg("catch", Qc, QDc, QDDc, (0, 0), None, None, ("Left",), "timed")
        raw = dict(self._raw_chain)
        # parcel centre y during the catch: free ride, but never past the
        # left pads (the palm brakes it)
        y_pad = [float((pp @ ay)) for pp, _ in catch_poses]
        return dict(raw=raw, catch_pad_y=y_pad, prm=prm, geo=geo, key=key, approach=app,
                    approach2=app2, catch=cat,
                    manip=msegs, q_start=q_start, specs=specs)

    def check_plan(self, plan, pi, v):
        """Collision checks (parcel predicted) and the command rate."""
        pl = self.pl
        slot = pi.slot
        geo = plan["geo"]
        # approach + catch: parcel moving on the belt from its current pose
        col = pl.fresh_collider()
        t = 0.0
        y_stop_c = geo["c0"][1]
        oth = self.others(slot)
        t_wait = max(0.0, plan.get("t_lead", 0.0) - len(plan["approach"].Q) * DT)
        ypad = plan.get("catch_pad_y")
        Lyh = float(geo["Ly"]) / 2
        if plan.get("approach2") is not None:
            # direct approach first; the high-via approach if it collides
            bad = False
            tt0 = 0.0
            for i in range(len(plan["approach"].Q)):
                tt0 += DT
                pos = pi.pos + np.array([0.0, v * tt0, 0.0])
                res = col.check(plan["approach"].Q[i], [(slot, pos, pi.R)] + self.others_at(oth, v, tt0))
                if not Collide.ok(res):
                    bad = True
                    break
            if bad:
                plan["approach"] = plan["approach2"]
        for sg in (plan["approach"], plan["catch"]):
            for i in range(len(sg.Q)):
                t += DT
                if sg.name == "catch" and i == 0:
                    t += t_wait
                pos = pi.pos + np.array([0.0, v * t, 0.0])
                if pos[1] > y_stop_c:
                    pos[1] = y_stop_c
                if sg.name == "catch" and ypad is not None and i < len(ypad):
                    pos[1] = min(pos[1], ypad[i] - Lyh + PAD_TIP - 0.0005)
                res = col.check(sg.Q[i], [(slot, pos, pi.R)] + self.others_at(oth, v, t),
                                contact_ok=sg.contact)
                if not Collide.ok(res, contact=bool(sg.contact)):
                    res = col.check(sg.Q[i], [(slot, pos, pi.R)] + self.others_at(oth, v, t),
                                    contact_ok=sg.contact, detail=True)
                    return False, f"{sg.name} {i}/{len(sg.Q)} collision {res.get('worst')}"
        par0 = (geo["c0"], geo["Rp0"])
        t0m = t + 0.3
        tt = t0m
        for sg in plan["manip"]:
            for i in range(0, len(sg.Q)):
                tt += DT
                par = sg.parcel[i] if sg.parcel is not None else par0
                res = col.check(sg.Q[i], [(slot, par[0], par[1])] + self.others_at(oth, v, tt),
                                contact_ok=sg.contact)
                if not Collide.ok(res, contact=bool(sg.contact)):
                    res = col.check(sg.Q[i], [(slot, par[0], par[1])] + self.others_at(oth, v, tt),
                                    contact_ok=sg.contact, detail=True)
                    return False, f"manip collision {sg.name} {i}/{len(sg.Q)} {res.get('worst')}"
        plan["t_manip_end"] = tt
        return True, "ok"

    # ------------------------------------------------------ execution
    def F_for(self, name, i, n, geo, prm, m):
        """Palm force feedforward per segment and tick."""
        ay = geo["ay"]
        mg = m * G
        fsq = squeeze_force(m)
        u = (i + 1) / max(1, n)
        drag = 0.85 * mg
        if name in ("catch_wait", "descend"):
            return {"Left": -ay * drag}
        if name == "close":
            a = min(1.0, u / 0.8)
            return {"Left": -ay * (drag + fsq * a), "Right": ay * fsq * a}
        if name == "hold":
            return {"Left": -ay * (drag + fsq), "Right": ay * fsq}
        if name == "lift":
            w = min(1.0, u / 0.25)
            return {"Left": -ay * (fsq + drag * (1 - w)) + np.array([0, 0, mg / 2 * w]),
                    "Right": ay * fsq + np.array([0, 0, mg / 2 * w])}
        if name in ("rotate", "fix"):
            return {"Left": -ay * fsq + np.array([0, 0, mg / 2]),
                    "Right": ay * fsq + np.array([0, 0, mg / 2])}
        if name == "lower":
            w = 1.0 - max(0.0, (u - 0.7) / 0.3)
            return {"Left": -ay * (fsq + drag * (1 - w)) + np.array([0, 0, mg / 2 * w]),
                    "Right": ay * fsq + np.array([0, 0, mg / 2 * w])}
        if name == "hold2":
            return {"Left": -ay * (drag + fsq), "Right": ay * fsq}
        return None

    def play(self, sg, F_fn=None, grip=None, check=None):
        self.ex.cur_seg = sg.name
        n = len(sg.Q)
        for i in range(n):
            F = None if F_fn is None else F_fn(i, n)
            self.ex.tick(sg.Q[i], sg.QD[i], sg.QDD[i], grip if grip is not None else self.grip, F)
            if self.cell.done:
                return i + 1
            if check is not None and check(i):
                return i + 1
        return n

    def hold(self, seconds, F=None, check=None):
        self.ex.cur_seg = "hold"
        n = max(1, int(round(seconds / DT)))
        for i in range(n):
            self.ex.tick(self.ex.q, None, None, self.grip, F)
            if check is not None and check(i):
                return i + 1
        return n

    # ------------------------------------------------------ release
    def fk(self, side, q14):
        return self.pl.ik.kin[side].fk(np.asarray(q14)[SL[side]])

    def plan_release(self, slot, v, t_extra_wait=0.0, squeezing=True,
                     state=None):
        """Gate (both palms, still squeezing, accelerate the parcel to belt
        speed); gap (the left palm runs ahead and up, the right palm falls
        back and up); retreat (both palms, Cartesian, tool axis only, the
        roll about it free, joint-limit avoidance in the null space: up
        over the parcel and back toward the robot, the left tool axis
        turning down); then a joint move into the park posture. Checked
        against the parcel riding at belt speed. ``state`` = (q0, parcel
        pos, R, dims, mass) for a prediction (default: the cell now)."""
        if state is None:
            pi = self.parcel(slot)
            if pi is None:
                raise PlanFail("parcel gone")
            q0 = self.ex.q.copy()
            ppos, pR, dims, mass = pi.pos.copy(), pi.R.copy(), pi.dims, pi.mass
        else:
            q0, ppos, pR, dims, mass = state
            q0 = np.asarray(q0, float).copy()
        Lx, Ly, H = (float(x) for x in dims)
        pL, RL = self.fk("Left", q0)
        pR_, RR = self.fk("Right", q0)
        mg = mass * G
        a = float(np.clip(0.25 * G * Ly / max(H, 1e-3), 1.0, 3.5))
        T_a = v / a
        n_a = max(3, int(math.ceil(T_a / DT)))
        n_run = 3
        yg = []
        for i in range(n_a + n_run):
            t = (i + 1) * DT
            yg.append(0.5 * a * min(t, T_a) ** 2 + v * max(0.0, t - T_a))
        nG = len(yg)
        ex = self.ex
        fsq = squeeze_force(mass)
        ay = np.array([0.0, 1.0, 0.0])
        top = self.zb + H
        near = ppos[0] - Lx / 2

        def F_rel(i, n, n_g=nG):
            if not squeezing:
                if i < n_a:
                    return {"Left": -ay * 0.85 * mg * (1 - (i + 1) / n_a)}
                return None
            if i < n_a:
                frac = (i + 1) / n_a
                return {"Left": -ay * (fsq + 0.85 * mg * (1 - frac)),
                        "Right": ay * (fsq + mass * a)}
            if i < n_g:
                return {"Left": -ay * fsq, "Right": ay * fsq}
            k = i - n_g
            w = max(0.0, 1.0 - k / 3.0)
            if w <= 0:
                return None
            return {"Left": -ay * fsq * w, "Right": ay * fsq * w}

        col = self.pl.fresh_collider()
        errs = []
        oth = self.others(slot) if state is None else list(getattr(self, "_pred_others", []))
        # ---- variant P (low results resting on the belt): pin release
        on_belt_now = (ppos[2] - H / 2) < self.zb + 0.012
        if H <= min(0.17, 0.45 * Ly) and on_belt_now:
            try:
                return self.plan_release_pin(slot, v, q0, ppos, pR, (Lx, Ly, H), mass,
                                             squeezing, col, oth, state is None)
            except PlanFail as e:
                errs.append(f"pin: {e}")
        # ---- variant S (low results): stop release by sliding out. The
        # right palm backs off and rises (the belt keeps the parcel against
        # the left palm); the left palm slides along the face toward the
        # robot, over the lip, until it is off the face (the parcel is
        # dragged against the guide lip, then rides away with the belt).
        low_result = (H <= min(0.17, 1.0 * Ly))
        if low_result:
            for (vx, rise_extra) in ((0.28, 0.012), (0.22, 0.02), (0.34, 0.015)):
                try:
                    lip_top = self.zb + self.st.lip_height
                    low_rel = lowest_point(RL, np.zeros(3))
                    far_rel = float(np.max((GRIP_BOX @ RL.T)[:, 0]))
                    dx = max(0.0, pL[0] + far_rel - (near - 0.012))
                    dz = max(0.0, lip_top + rise_extra - (pL[2] + low_rel))
                    n_r = int(math.ceil(0.40 / DT))          # right palm leaves first
                    T_s = max(0.4, 1.875 * dx / vx)
                    n_s = int(math.ceil(T_s / DT))
                    Lpos, Rpos = [], []
                    for i in range(n_r + n_s):
                        t = (i + 1) * DT
                        ur = min_jerk(min(1.0, t / (n_r * DT)))
                        Rpos.append((pR_ + np.array([-0.02 * ur, -0.03 * ur, 0.09 * ur]), RR))
                        us = min_jerk(min(1.0, max(0.0, (t - n_r * DT * 0.8) / T_s)))
                        uz = min_jerk(min(1.0, max(0.0, (t - n_r * DT * 0.8) / (0.6 * T_s))))
                        p_ = pL + np.array([-dx * us, 0.0, dz * uz])
                        # off the face: the roll about the tool axis is free
                        pad_far = p_[0] + float(np.max((PAD_BOX @ RL.T)[:, 0]))
                        Lpos.append((p_, RL, "axis") if pad_far < near - 0.004 else (p_, RL))
                    QL = self.pl.ik.trace("Left", Lpos, q0[SL["Left"]], max_jump=0.09, limit_gain=0.02)
                    QR = self.pl.ik.trace("Right", Rpos, q0[SL["Right"]], max_jump=0.09, limit_gain=0.02)
                    Q1 = np.repeat(q0[None], len(QL), axis=0)
                    Q1[:, SL["Left"]] = QL
                    Q1[:, SL["Right"]] = QR
                    QD1, QDD1 = diff_samples(Q1, q0)
                    Qp, QDp, QDDp = joint_move(Q1[-1], Q_PARK, vmax=V_PLAN, tmin=0.6, amax=A_PLAN)
                    Q = np.vstack([Q1, Qp])
                    QD = np.vstack([QD1, QDp])
                    QDD = np.vstack([QDD1, QDDp])
                    # parcel: still until the pads leave the face, then rides
                    n_free = None
                    for i in range(len(QL)):
                        pk, Rk = self.fk("Left", Q1[i])
                        if pk[0] + float(np.max((PAD_BOX @ Rk.T)[:, 0])) < near - 0.002:
                            n_free = i
                            break
                    if n_free is None:
                        raise PlanFail("slide: pads never leave the face")
                    par = []
                    for i in range(len(Q)):
                        y_p = 0.0 if i <= n_free else v * (i - n_free) * DT
                        par.append((ppos + np.array([0, y_p, 0]), pR))
                    bad = None
                    for i in range(len(Q)):
                        ok_c = SIDES if i < 8 else (("Left",) if i <= n_free else ())
                        res = col.check(Q[i], [(slot, par[i][0], par[i][1])] + self.others_at(oth, v, (i + 1) * DT),
                                        contact_ok=ok_c, detail=True)
                        if not Collide.ok(res, contact=bool(ok_c) or i < n_free + 4):
                            bad = (i, res)
                            break
                    if bad is not None:
                        w = bad[1].get("worst", {})
                        raise PlanFail(f"slide collision at {bad[0]}/{len(Q)} (free {n_free}) "
                                       f"{ {k: (round(v_[2], 3), v_[0], v_[1]) for k, v_ in w.items()} }")
                    def F_slide(i, n, n_r=n_r):
                        if i < n_r:
                            w = 1.0 - (i + 1) / n_r
                            return {"Left": -ay * (0.85 * mg + (fsq * w if squeezing else 0.0)), "Right": (ay * fsq * w if squeezing else None)}
                        if i <= n_free:
                            return {"Left": -ay * 0.6 * mg}
                        return None
                    peak = 0.0
                    u0 = ex.u_prev.copy() if state is None else None
                    for i in range(len(Q)):
                        u = ex.command(Q[i], QD[i], QDD[i], F_slide(i, len(Q)))
                        if u0 is not None:
                            peak = max(peak, float(np.max(np.abs(u - u0))) / DT)
                        u0 = u
                    if peak > V_CMD_BUDGET:
                        raise PlanFail(f"slide rate {peak:.2f}")
                    seg = Seg("release", Q, QD, QDD, (0, 0), None, par, ("Left",), "timed")
                    info = dict(mode="slide", rate=round(peak, 2), n_gate=n_free,
                                variant=(vx, rise_extra), t_release=round(len(Q) * DT, 2))
                    return seg, None, F_slide, info
                except PlanFail as e:
                    errs.append(f"slide{(vx, rise_extra)}: {e}")
        # (gap, gap time, retreat x, rise above the top, tool-down angle,
        #  retreat time)
        variants = [(0.035, 0.12, 0.13, 0.07, 1.0, 0.75), (0.035, 0.12, 0.16, 0.05, 0.7, 0.8),
                    (0.04, 0.12, 0.10, 0.09, 1.3, 0.7), (0.03, 0.14, 0.18, 0.04, 0.4, 0.9),
                    (0.04, 0.10, 0.12, 0.08, 1.0, 0.6), (0.035, 0.12, 0.20, 0.06, 0.9, 1.0)]
        # side escape: at belt speed the squeeze is off, the left palm slides
        # along the face toward the robot (just over the lip) while keeping
        # pace with the parcel, its tool axis turning down (y reach)
        variants = [(0.012, 0.10, None, None, 0.7, 0.6), (0.012, 0.10, None, None, 1.1, 0.55),
                    (0.02, 0.10, None, None, 0.3, 0.7), (0.012, 0.08, None, None, 1.0, 0.45)] + variants
        for (gap, T_g, x_back, rise, beta, T_r) in variants:
            try:
                Lpos, Rpos = [], []
                for y in yg:
                    Lpos.append((pL + np.array([0, y, 0]), RL))
                    Rpos.append((pR_ + np.array([0, y, 0]), RR))
                ycur = yg[-1]
                low_rel = lowest_point(RL, np.zeros(3))
                if rise is None:
                    lip_top = self.zb + self.st.lip_height
                    dz = max(0.02, lip_top + 0.02 - (pL[2] + low_rel))
                    far_rel = float(np.max((GRIP_BOX @ RL.T)[:, 0]))
                    dx = max(0.0, pL[0] + far_rel - (near - 0.02))
                else:
                    dz = max(0.02, top + rise - (pL[2] + low_rel))
                    dx = max(0.0, pL[0] - max(0.10, near - x_back))
                n_gap = int(math.ceil(T_g / DT))
                n_ret = int(math.ceil(T_r / DT))
                side_mode = rise is None
                T_pace = 0.5 * T_g + 0.8 * T_r       # side escape: keep pace this long
                T_dec = 0.25
                n_extra = int(math.ceil(T_dec / DT)) if side_mode else 0
                for i in range(n_gap + n_ret + n_extra):
                    t = (i + 1) * DT
                    ug = min_jerk(min(1.0, t / T_g))
                    ur = min_jerk(min(1.0, max(0.0, t - 0.5 * T_g) / T_r))
                    if side_mode:
                        # y: pace with the parcel until mostly clear, then
                        # decelerate to rest
                        if t <= T_pace:
                            yv = v * t
                        else:
                            td_ = min(t - T_pace, T_dec)
                            yv = v * T_pace + v * (td_ - td_ * td_ / (2 * T_dec))
                        yL = ycur + yv + gap * ug
                    else:
                        # y: ahead of the parcel, then decelerate to rest
                        tr = max(0.0, t - 0.5 * T_g)
                        if tr < T_r:
                            yv = v * (tr - tr * tr / (2 * T_r))
                        else:
                            yv = v * T_r / 2
                        yL = ycur + v * min(t, 0.5 * T_g) + yv + gap * ug
                    p_ = pL + np.array([-dx * ur, yL, dz * ur + 0.01 * ug])
                    Rb = rot_axis_angle((1, 0, 0), beta * ur) @ RL
                    Lpos.append((p_, Rb, "axis"))
                    td = 0.16
                    yR = ycur + (v * (t - t * t / (2 * td)) if t < td else v * td / 2)
                    Rpos.append((pR_ + np.array([-0.04 * ur, yR - 0.03 * ur, 0.10 * ur]), RR, "axis"))
                QL = self.pl.ik.trace("Left", Lpos, q0[SL["Left"]], max_jump=0.105, limit_gain=0.02)
                QR = self.pl.ik.trace("Right", Rpos, q0[SL["Right"]], max_jump=0.105, limit_gain=0.02)
                Q1 = np.repeat(q0[None], len(QL), axis=0)
                Q1[:, SL["Left"]] = QL
                Q1[:, SL["Right"]] = QR
                QD1, QDD1 = diff_samples(Q1, q0)
                # come to rest (the retreat ends at rest), then park
                Qp, QDp, QDDp = joint_move(Q1[-1], Q_PARK, vmax=V_PLAN, tmin=0.6, amax=A_PLAN)
                Q = np.vstack([Q1, Qp])
                QD = np.vstack([QD1, QDp])
                QDD = np.vstack([QDD1, QDDp])
                par = []
                for i in range(len(Q)):
                    y_p = yg[i] if i < nG else yg[-1] + v * (i - nG + 1) * DT
                    par.append((ppos + np.array([0, y_p, 0]), pR))
                bad = None
                for i in range(len(Q)):
                    ok_c = SIDES if i < nG + 3 else ()
                    res = col.check(Q[i], [(slot, par[i][0], par[i][1])] + self.others_at(oth, v, (i + 1) * DT),
                                    contact_ok=ok_c, detail=True)
                    if not Collide.ok(res, contact=bool(ok_c) or i < nG + n_gap):
                        bad = (i, res)
                        break
                if bad is not None:
                    w = bad[1].get("worst", {})
                    raise PlanFail(f"collision at {bad[0]}/{len(Q)} (gate {nG}, gap {n_gap}) "
                                   f"{ {k: (round(v_[2], 3), v_[0], v_[1]) for k, v_ in w.items()} }")
                peak = 0.0
                u0 = ex.u_prev.copy() if state is None else None
                for i in range(len(Q)):
                    u = ex.command(Q[i], QD[i], QDD[i], F_rel(i, len(Q)))
                    if u0 is not None:
                        peak = max(peak, float(np.max(np.abs(u - u0))) / DT)
                    u0 = u
                if peak > V_CMD_BUDGET:
                    raise PlanFail(f"rate {peak:.2f}")
                seg = Seg("release", Q, QD, QDD, (0, 0), None, par, SIDES, "timed")
                info = dict(gate_acc=round(a, 2), rate=round(peak, 2), n_gate=nG,
                            variant=(gap, T_g, x_back, rise, beta, T_r),
                            t_release=round(len(Q) * DT, 2))
                return seg, None, F_rel, info
            except PlanFail as e:
                errs.append(f"{(gap, x_back, rise, beta)}: {e}")
        self._rel_errs = errs
        raise PlanFail("; ".join(errs[:3] + errs[-1:]))

    def plan_release_pin(self, slot, v, q0, ppos, pRot, dims, mass, squeezing, col,
                         oth, live):
        """Pin release (parcel resting on the belt, stopped against the left
        palm): the right palm leaves the upstream face, turns its tool down
        and presses the top face (friction holds the parcel against the belt
        drag); the left palm backs off, rises and parks; the right palm lifts
        off (the parcel rides away with the belt) and parks."""
        Lx, Ly, H = dims
        mg = mass * G
        fsq = squeeze_force(mass)
        F_pin = 2.2 * mg + 8.0
        top = self.zb + H
        ik = self.pl.ik
        pL, RL = self.fk("Left", q0)
        pR, RR = self.fk("Right", q0)
        ay = np.array([0.0, 1.0, 0.0])
        segsQ = []
        contact = []
        forces = []
        # A: right palm off the face (back -y, up)
        n_a = 20
        rise_a = max(0.03, top + 0.03 - (pR[2] + lowest_point(RR, np.zeros(3))))
        posesA = [(pR + np.array([0.0, -0.03 * min_jerk(min(1.0, 2 * u)), rise_a * min_jerk(u)]), RR)
                  for u in np.linspace(0, 1, n_a + 1)[1:]]
        WA = ik.trace("Right", posesA, q0[SL["Right"]], max_jump=0.1)
        QA = np.repeat(q0[None], n_a, axis=0)
        QA[:, SL["Right"]] = WA
        for i in range(n_a):
            w = 1.0 - min(1.0, (i + 1) / 12.0)
            forces.append({"Left": -ay * (0.85 * mg + fsq * w), "Right": ay * fsq * w if w > 0 else None})
            contact.append(SIDES)
        segsQ.append(QA)
        # pin pose: tool down, pads spread along y on the top face (downstream half)
        y_hi = ppos[1] + Ly / 2 - 0.058
        y_lo = ppos[1] - Ly / 2 + 0.058
        if y_hi < y_lo:
            raise PlanFail("pin: top face too short")
        x_p = ppos[0] + 0.0
        Rpin = np.column_stack([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        cands = []
        for fy in (0.25, 0.0, -0.2):
            y_p = float(np.clip(ppos[1] + fy * Ly, y_lo, y_hi))
            p_pin = np.array([x_p, y_p, top - 0.0005 + PAD_TIP])
            p_pre = p_pin + np.array([0.0, 0.0, 0.05])
            sols = ik.solve("Right", p_pre, Rpin, seeds=[QA[-1][SL["Right"]], Q_REST["Right"], Q_PARK[SL["Right"]]],
                            n_rand=3, margin=0.03)
            sols.sort(key=lambda q: float(np.max(np.abs(q - QA[-1][SL["Right"]]))))
            cands += [(p_pin, p_pre, q) for q in sols[:2]]
        if not cands:
            raise PlanFail("pin: no pre-pose IK")
        got = None
        for p_pin, p_pre, qpre in cands:
            qg = QA[-1].copy()
            qg[SL["Right"]] = qpre
            QB, _, _ = joint_move(QA[-1], qg, vmax=V_PLAN, tmin=0.5, amax=A_PLAN)
            try:
                down = [(p_pre + (p_pin - p_pre) * u, Rpin) for u in np.linspace(0, 1, 11)[1:]]
                WC = ik.trace("Right", down, qpre, max_jump=0.1)
            except PlanFail:
                continue
            # the swing must clear the parcel and the structure
            okc = True
            for qq in QB[::2]:
                rr = col.check(qq, [(slot, ppos, pRot)], contact_ok=("Left",))
                if not Collide.ok(rr, contact=True):
                    okc = False
                    break
            if not okc:
                continue
            got = (QB, qg, WC, p_pin)
            break
        if got is None:
            raise PlanFail("pin: no clear swing / descent to the top face")
        QB, qg, WC, p_pin = got
        segsQ.append(QB)
        for i in range(len(QB)):
            forces.append({"Left": -ay * 0.85 * mg})
            contact.append(("Left",))
        WCq = np.vstack([qg[SL["Right"]][None], WC])
        QC = np.repeat(qg[None], len(WCq), axis=0)
        QC[:, SL["Right"]] = WCq
        QC, _, _ = retime(QC, vmax=1.0, amax=A_PLAN, tmin=0.4)
        segsQ.append(QC)
        for i in range(len(QC)):
            forces.append({"Left": -ay * 0.85 * mg})
            contact.append(SIDES)
        # hold: ramp the pin force up (0.5 s)
        n_h = 25
        QH = np.repeat(QC[-1][None], n_h, axis=0)
        segsQ.append(QH)
        for i in range(n_h):
            w = (i + 1) / n_h
            forces.append({"Left": -ay * 0.85 * mg * (1 - 0.7 * w),
                           "Right": np.array([0.0, 0.0, -F_pin * w])})
            contact.append(SIDES)
        # D: left palm backs off (+y), rises, parks
        qd0 = QH[-1]
        posesD = [(pL + np.array([0.0, 0.030 * min_jerk(u), 0.012 * min_jerk(u)]), RL)
                  for u in np.linspace(0, 1, 13)[1:]]
        WD = ik.trace("Left", posesD, qd0[SL["Left"]], max_jump=0.1, limit_gain=0.02)
        p_up = posesD[-1][0]
        posesD2 = [(p_up + np.array([0.0, 0.01 * u, 0.10 * min_jerk(u)]), RL, "axis")
                   for u in np.linspace(0, 1, 13)[1:]]
        WD2 = ik.trace("Left", posesD2, WD[-1], max_jump=0.1, limit_gain=0.02)
        WDall = np.vstack([qd0[SL["Left"]][None], WD, WD2])
        QD_ = np.repeat(qd0[None], len(WDall), axis=0)
        QD_[:, SL["Left"]] = WDall
        QD_, _, _ = retime(QD_, vmax=1.6, amax=A_PLAN, tmin=0.6)
        segsQ.append(QD_)
        for i in range(len(QD_)):
            forces.append({"Right": np.array([0.0, 0.0, -F_pin])})
            contact.append(SIDES if i < max(16, len(QD_) // 2) else ("Right",))
        qL_park = QD_[-1].copy()
        qL_park[SL["Left"]] = Q_PARK[SL["Left"]]
        QDp, _, _ = joint_move(QD_[-1], qL_park, vmax=V_PLAN, tmin=0.6, amax=A_PLAN)
        segsQ.append(QDp)
        for i in range(len(QDp)):
            forces.append({"Right": np.array([0.0, 0.0, -F_pin])})
            contact.append(("Right",))
        # E: right palm lifts off (force ramp down, rise), parks
        n_e = 12
        qe0 = QDp[-1]
        posesE = [(p_pin + np.array([0.0, -0.01 * u, 0.06 * min_jerk(u)]), Rpin) for u in np.linspace(0, 1, n_e + 1)[1:]]
        WE = ik.trace("Right", posesE, qe0[SL["Right"]], max_jump=0.1)
        QE = np.repeat(qe0[None], n_e, axis=0)
        QE[:, SL["Right"]] = WE
        segsQ.append(QE)
        n_rel = sum(len(x) for x in segsQ[:-1])     # release starts with E
        for i in range(n_e):
            w = max(0.0, 1.0 - (i + 1) / 4.0)
            forces.append({"Right": np.array([0.0, 0.0, -F_pin * w])} if w > 0 else None)
            contact.append(("Right",) if i < 4 else ())
        QEp, _, _ = joint_move(QE[-1], Q_PARK, vmax=V_PLAN, tmin=0.6, amax=A_PLAN)
        segsQ.append(QEp)
        for i in range(len(QEp)):
            forces.append(None)
            contact.append(())
        Q = np.vstack(segsQ)
        QD, QDD = diff_samples(Q, q0)
        # parcel: still until the pin lifts (start of E + 2 ticks), then rides
        par = []
        for i in range(len(Q)):
            y_p_ = 0.0 if i < n_rel + 2 else v * (i - n_rel - 2) * DT
            par.append((ppos + np.array([0.0, y_p_, 0.0]), pRot))
        for i in range(len(Q)):
            res = col.check(Q[i], [(slot, par[i][0], par[i][1])] + self.others_at(oth, v, (i + 1) * DT),
                            contact_ok=contact[i], detail=True)
            if not Collide.ok(res, contact=bool(contact[i]) or (n_rel <= i < n_rel + 6)):
                w = res.get("worst", {})
                raise PlanFail(f"pin collision at {i}/{len(Q)} (release {n_rel}) "
                               f"{ {k: (round(v_[2], 3), v_[0], v_[1]) for k, v_ in w.items()} }")
        peak = 0.0
        u0 = self.ex.u_prev.copy() if live else None
        for i in range(len(Q)):
            u = self.ex.command(Q[i], QD[i], QDD[i], forces[i])
            if u0 is not None:
                peak = max(peak, float(np.max(np.abs(u - u0))) / DT)
            u0 = u
        if peak > V_CMD_BUDGET:
            raise PlanFail(f"pin rate {peak:.2f}")
        seg = Seg("release", Q, QD, QDD, (0, 0), None, par, SIDES, "timed")
        info = dict(mode="pin", rate=round(peak, 2), n_gate=n_rel, F_pin=round(F_pin, 1),
                    t_release=round(len(Q) * DT, 2))

        def F_pin_fn(i, n):
            return forces[i] if i < len(forces) else None
        return seg, None, F_pin_fn, info

    # ------------------------------------------------------ one parcel
    def lead_time(self, pi, prm, v):
        Ly = float(pi.dims[1])
        y_stop_face = prm.y_c + Ly / 2
        d_catch = v * (T_ACC + T_DEC) / 2
        y_w = y_stop_face - d_catch
        y_face = pi.pos[1] + Ly / 2
        return (y_w + v * T_ACC / 2 - y_face) / max(v, 1e-3) - T_ACC

    def find_plan(self, pi, cls, v, max_tries=80, deadline_s=60.0):
        import time as _t
        t0 = _t.time()
        q_now = self.ex.q.copy()
        combos = self.candidates(pi, cls)
        why = {}
        n = 0
        for prm in combos:
            if n >= max_tries or (deadline_s and _t.time() - t0 > deadline_s):
                why["budget"] = why.get("budget", 0) + 1
                break
            t_lead = self.lead_time(pi, prm, v)
            if t_lead < 0.6:
                why["late"] = why.get("late", 0) + 1
                continue
            th0s, winfo = self.pick_theta0(pi, prm)
            if not th0s:
                why["window"] = why.get("window", 0) + 1
                continue
            for (th0, th0R), th_rot, seeds in th0s:
                n += 1
                settle = abs(abs(th_rot) - math.pi) > 1e-6 and cls == "down"
                p2 = JobParams(**{**prm.__dict__, "theta0": th0, "theta0R": th0R,
                                  "theta": th_rot, "settle": settle})
                try:
                    try:
                        plan = self.plan_job(pi, p2, v, q_now, t_lead, seeds=seeds)
                    except RotShort as e:
                        mag = abs(th_rot) * e.frac - math.radians(6)
                        if cls != "down" or mag < math.radians(115) or \
                                not self.settle_ok(pi, p2, mag):
                            raise
                        p2 = JobParams(**{**p2.__dict__, "theta": math.copysign(mag, th_rot),
                                          "settle": True})
                        plan = self.plan_job(pi, p2, v, q_now, t_lead, seeds=seeds)
                except PlanFail as e:
                    k = " ".join(str(e).split(" ")[:3])
                    why[k] = why.get(k, 0) + 1
                    continue
                plan["t_lead"] = t_lead
                ok, msg = self.check_plan(plan, pi, v)
                if not ok:
                    why["collision"] = why.get("collision", 0) + 1
                    self._last_col = msg
                    continue
                rate = self.plan_rate(plan, pi)
                plan["rate"] = rate
                if rate > V_CMD_BUDGET:
                    why["rate"] = why.get("rate", 0) + 1
                    continue
                # the release must be plannable from the predicted end state
                try:
                    lw = plan["manip"][-1]
                    par_end = lw.parcel[-1] if lw.parcel else (plan["geo"]["c0"], plan["geo"]["Rp0"])
                    self._th_done = p2.theta
                    tme = plan.get("t_manip_end", 0.0)
                    self._pred_others = self.others_at(self.others(pi.slot), v, tme)
                    rel_pred = self.plan_release(pi.slot, v, state=(
                        lw.Q[-1], par_end[0], par_end[1], plan["geo"]["dims_after"], pi.mass))
                    rinfo = rel_pred[3]
                    plan["release_pred"] = rinfo
                    plan["release_seg"] = rel_pred
                    plan["q_end_pred"] = lw.Q[-1].copy()
                except PlanFail as e:
                    why["release"] = why.get("release", 0) + 1
                    self._last_rel = str(e)[:400]
                    continue
                finally:
                    self._th_done = 0.0
                    self._pred_others = []
                plan["tries"] = n
                plan["plan_s"] = _t.time() - t0
                plan["window"] = winfo
                return plan, why
        return None, why

    def plan_rate(self, plan, pi):
        """Peak commanded joint rate of the planned approach, catch and
        manipulation, with the force feedforward of the execution."""
        ex = self.ex
        geo, prm = plan["geo"], plan["prm"]
        ay = geo["ay"]
        mg = pi.mass * G
        n1 = int(round(T_ACC / DT))
        u0 = ex.u_prev.copy()
        peak = 0.0

        def run(sg, F_fn):
            nonlocal u0, peak
            n = len(sg.Q)
            for i in range(n):
                u = ex.command(sg.Q[i], sg.QD[i], sg.QDD[i], None if F_fn is None else F_fn(i, n))
                peak = max(peak, float(np.max(np.abs(u - u0))) / DT)
                u0 = u
        run(plan["approach"], None)

        def f_catch(i, n):
            lv = min(1.0, max(0.0, (i - (n1 - 3)) / 10.0))
            return {"Left": -ay * 0.8 * mg * lv}
        run(plan["catch"], f_catch)
        for sg in plan["manip"]:
            run(sg, lambda i, n, name=sg.name: self.F_for(name, i, n, geo, prm, pi.mass))
        return peak

    def run_parcel(self, slot):
        pi = self.parcel(slot)
        if pi is None:
            return dict(slot=slot, result="gone")
        cls = self.job_class(pi)
        self._th_done = 0.0
        res = dict(slot=slot, uid=pi.uid, label0=pi.label, cls=cls,
                   kind=pi.kind, dims=[round(float(x), 3) for x in pi.dims],
                   mass=round(pi.mass, 3), t0=self.cell.time)
        self.results.append(res)
        if cls in ("pass", "pass_y"):
            res["result"] = "pass"
            return res
        v = self.belt_speed()
        plan, why = self.find_plan(pi, cls, v)
        res["plan_why"] = why
        if plan is None:
            res["result"] = "no_plan"
            return res
        adm = getattr(self, "admit", None)
        if adm is not None and not adm(plan, pi, v):
            res["result"] = "skip"
            res["admit"] = plan.get("admit")
            return res
        prm = plan["prm"]
        res["prm"] = dict(theta=round(math.degrees(prm.theta)), theta0=round(math.degrees(prm.theta0)),
                          theta0R=None if prm.theta0R is None else round(math.degrees(prm.theta0R)),
                          zf=prm.zf, dxs=prm.dxs, lift_extra=prm.lift_extra, y_c=prm.y_c)
        res["plan_tries"] = plan["tries"]
        res["plan_s"] = round(plan["plan_s"], 2)
        geo = plan["geo"]
        ay = geo["ay"]
        m = pi.mass
        mg = m * G
        # 1. approach
        self.play(plan["approach"])
        # 2. wait for the catch start (re-timed on the measured parcel)
        pi2 = self.parcel(slot)
        if pi2 is None:
            res["result"] = "gone"
            return res
        t_left = self.lead_time(pi2, prm, v)
        if t_left > DT:
            self.hold(t_left)
        # 3. catch (left palm matches the belt speed, then brakes)
        cat = plan["catch"]
        n1 = int(round(T_ACC / DT))

        def f_catch(i, n):
            lv = min(1.0, max(0.0, (i - (n1 - 3)) / 10.0))
            return {"Left": -ay * 0.8 * mg * lv}
        self.play(cat, F_fn=f_catch)
        # 4. wait for a steady stop
        hist = []

        def stopped(i):
            ps = self.parcel(slot)
            if ps is None:
                return True
            hist.append(float(ps.pos[1]))
            return len(hist) >= 10 and (max(hist[-10:]) - min(hist[-10:])) < 0.003
        self.hold(6.0, F={"Left": -ay * 0.85 * mg}, check=stopped)
        pi3 = self.parcel(slot)
        if pi3 is None or len(hist) < 10 or (max(hist[-10:]) - min(hist[-10:])) > 0.004:
            res["result"] = "catch_fail"
            self.give_up(slot, res, stage="catch")
            return res
        res["t_catch"] = self.cell.time
        # 5. re-plan the manipulation from the measured stop pose
        try:
            msegs, geo = self.replan_manip(pi3, prm, plan)
        except PlanFail as e:
            g0 = plan["geo"]
            dpos = pi3.pos - g0["c0"]
            dyaw = abs(pi3.yaw_err - g0["yaw"])
            if abs(dpos[0]) < 0.025 and abs(dpos[1]) < 0.012 and abs(dpos[2]) < 0.008 \
                    and dyaw < math.radians(15):
                # the squeeze re-squares the parcel to the palms (yaw) and the
                # pads absorb the shift across the face
                # the stop matches the prediction: the checked plan stands
                msegs, geo = plan["manip"], g0
                res["replan_fallback"] = [round(float(x), 4) for x in dpos]
            else:
                res["result"] = "replan_fail"
                res["replan_why"] = str(e)[:200] + f" dpos {np.round(dpos, 4).tolist()} dyaw {math.degrees(dyaw):.1f}"
                self.give_up(slot, res, stage="stop")
                return res
        if msegs is not plan["manip"]:
            pr_ = dict(plan)
            pr_["manip"] = msegs
            pr_["approach"] = Seg("none", self.ex.q[None].copy(), np.zeros((1, 14)), np.zeros((1, 14)))
            pr_["catch"] = Seg("none", self.ex.q[None].copy(), np.zeros((1, 14)), np.zeros((1, 14)))
            r_ = self.plan_rate(pr_, pi3)
            if r_ > V_CMD_BUDGET:
                g0 = plan["geo"]
                dpos = pi3.pos - g0["c0"]
                if abs(dpos[0]) < 0.025 and abs(dpos[1]) < 0.012 and abs(dpos[2]) < 0.008:
                    msegs, geo = plan["manip"], g0
                    res["replan_rate_fallback"] = round(r_, 2)
                else:
                    res["result"] = "replan_fail"
                    res["replan_why"] = f"re-planned rate {r_:.2f}"
                    self.give_up(slot, res, stage="stop")
                    return res
        ay = geo["ay"]
        segs = {s.name: s for s in msegs}
        Fn = lambda name: (lambda i, n: self.F_for(name, i, n, geo, prm, m))
        for name in ("descend", "close"):
            self.play(segs[name], F_fn=Fn(name))
        self.hold(0.15, F=self.F_for("hold", 0, 1, geo, prm, m))
        self.play(segs["lift"], F_fn=Fn("lift"))
        self.play(segs["rotate"], F_fn=Fn("rotate"))
        self.hold(0.10, F=self.F_for("rotate", 0, 1, geo, prm, m))
        # 6. read the label; correct the residual inside the roll margin
        th_done = prm.theta
        fix = 0.0
        for it in range(2):
            ps = self.cell.truth.parcel_state(slot) if self.cell.slots[slot].active else None
            if ps is None:
                break
            n_ = ps.label_normal
            err = -math.atan2(n_[0], n_[2])
            # expected residual: the settle part of a short roll
            exp_res = (math.pi - abs(prm.theta)) if prm.settle else 0.0
            if abs(abs(err) - exp_res) < math.radians(12):
                break
            step = float(np.clip(err, -prm.over, prm.over))
            if abs(fix + step) > prm.over + 1e-6:
                step = math.copysign(max(0.0, prm.over - abs(fix)), step)
            if abs(step) < math.radians(1.5):
                break
            try:
                sg = self.rotate_more(geo, prm, th_done, step)
            except PlanFail:
                break
            self.play(sg, F_fn=Fn("rotate"))
            self.hold(0.08, F=self.F_for("rotate", 0, 1, geo, prm, m))
            th_done += step
            fix += step
        res["fix_deg"] = round(math.degrees(fix), 1)
        self._th_done = th_done
        # 7. set down: the checked plan's set-down when nothing was corrected
        # (the predicted release then starts exactly where it was planned),
        # else re-planned for the final angle from the measured pose
        low = None
        if abs(fix) < 1e-9 and "lower" in segs:
            low = segs["lower"]
            res["drop_mm"] = round(1000 * float(geo.get("drop", 0.0)), 1)
            res["lower_planned"] = True
        else:
            try:
                low = self.plan_lower(geo, prm, th_done, slot)
                res["drop_mm"] = round(1000 * getattr(self, "_lower_cut", 0.0), 1)
            except PlanFail as e:
                res["lower_fail"] = str(e)
                low = None
        if low is not None:
            self.play(low, F_fn=Fn("lower"))
        self.hold(0.12, F=self.F_for("hold2", 0, 1, geo, prm, m))
        ps = self.cell.truth.parcel_state(slot) if self.cell.slots[slot].active else None
        res["label_after_deg"] = None if ps is None else round(math.degrees(ps.label_angle), 1)
        res["on_belt_after"] = None if ps is None else bool(ps.on_belt)
        # 7b. retry: the label is off by more than the settle allows ->
        # lift again (still squeezed), roll by the residual inside the
        # planned margin, set down again (bounded: two attempts)
        lim = 14.0 if not prm.settle else (180.0 - math.degrees(abs(prm.theta))) + 10.0
        res["retries"] = 0
        for attempt in range(2):
            if ps is None or res["label_after_deg"] is None or res["label_after_deg"] <= lim:
                break
            if res["label_after_deg"] > lim + 25.0:
                break                      # far off: a retry would not recover it
            n_ = ps.label_normal
            err = -math.atan2(n_[0], n_[2])
            room = prm.over + math.radians(4) - abs(fix)
            step = float(np.clip(err, -room, room))
            if abs(step) < math.radians(2):
                break
            try:
                up = self.plan_lift_again(geo, slot)
                self.play(up, F_fn=Fn("lift"))
                sg = self.rotate_more_here(step)
                self.play(sg, F_fn=Fn("rotate"))
                th_done += step
                fix += step
                low = self.plan_lower(geo, prm, th_done, slot)
                self.play(low, F_fn=Fn("lower"))
                self.hold(0.12, F=self.F_for("hold2", 0, 1, geo, prm, m))
            except PlanFail as e:
                res["retry_fail"] = str(e)[:200]
                break
            res["retries"] += 1
            ps = self.cell.truth.parcel_state(slot) if self.cell.slots[slot].active else None
            res["label_after_deg"] = None if ps is None else round(math.degrees(ps.label_angle), 1)
        self._th_done = th_done
        # 8. release
        self.release(slot, v, res, plan=plan)
        res["t1"] = self.cell.time
        ok_lab = res["label_after_deg"] is not None and res["label_after_deg"] <= lim
        res["result"] = "done" if ok_lab else "fail_label"
        return res

    def plan_lift_again(self, geo, slot, dz=0.035):
        """Both palms (still squeezing) lift the parcel straight up."""
        q = self.ex.q.copy()
        n = 8
        W = np.repeat(q[None], n + 1, axis=0)
        for side in SIDES:
            p0, R = self.fk(side, q)
            poses = [(p0 + np.array([0.0, 0.0, dz * u]), R) for u in np.linspace(0, 1, n + 1)[1:]]
            W[1:, SL[side]] = self.pl.ik.trace(side, poses, q[SL[side]], max_jump=0.25)
        Q, QD, QDD = retime(W, vmax=1.2, amax=A_PLAN, tmin=0.35)
        return Seg("lift", Q, QD, QDD, (0, 0), None, None, SIDES, "free")

    def rotate_more_here(self, step):
        """Both palms roll by ``step`` about the current squeeze line."""
        q = self.ex.q.copy()
        pL, RL = self.fk("Left", q)
        pR, RR = self.fk("Right", q)
        n = max(3, int(abs(step) / math.radians(2)))
        W = np.repeat(q[None], n + 1, axis=0)
        for side, p0, R0 in (("Left", pL, RL), ("Right", pR, RR)):
            poses = [(p0, Ry(step * u) @ R0) for u in np.linspace(0, 1, n + 1)[1:]]
            W[1:, SL[side]] = self.pl.ik.trace(side, poses, q[SL[side]], max_jump=0.25)
        Q, QD, QDD = retime(W, vmax=1.2, amax=A_PLAN, tmin=0.25)
        return Seg("fix", Q, QD, QDD, (0, 0), None, None, SIDES, "free")

    # -------------------------------------------------------------------
    def replan_manip(self, pi, prm, plan):
        """Manipulation segments from the measured stop pose: the same
        chain, IK seeded waypoint by waypoint from the original plan."""
        pl = self.pl
        c0 = pi.pos.copy()
        # palms keep the planned yaw: the squeeze squares the parcel to them
        stop_pose = (c0, pi.R.copy(), plan["geo"]["yaw"])
        geo = pl.job_geometry(pi, prm, stop_pose=stop_pose)
        specs, key = pl.manip_paths(geo, prm)
        q = self.ex.q.copy()
        chs = {}
        for side in SIDES:
            names, Wall = plan["raw"][side]
            ref = {}
            for nm in [sp[0] for sp in specs]:
                ref[nm] = Wall[[i for i, n_ in enumerate(names) if n_ == nm]]
            ch = {}
            prev = q[SL[side]].copy()
            ch["catch"] = prev[None].copy()
            for name, paths, opt in specs:
                if side not in paths:
                    ch[name] = prev[None].copy()
                    continue
                W = pl.ik.trace_seeded(side, paths[side], ref[name], prev)
                ch[name] = np.vstack([prev[None], W])
                prev = W[-1]
            if not pl.overshoot_ok(side, ch["rotate"][-1], geo, prm, key["s1"]):
                raise PlanFail(f"{side} margin after the stop")
            chs[side] = ch
        segs = pl.assemble(specs, chs, q)
        ok, worst, where = pl.check_segments(segs, pi.slot, parcel_static=(c0, pi.R))
        if not ok:
            raise PlanFail(f"collision after the stop {where}")
        return segs, geo

    def rotate_more(self, geo, prm, th_from, step):
        pl = self.pl
        s1 = geo["s0"].copy()
        s1[2] = geo["lift_z"]
        n = max(3, int(abs(step) / math.radians(2)))
        q = self.ex.q.copy()
        paths = {}
        ov = OVERPRESS
        ay = geo["ay"]
        for s in SIDES:
            poses = []
            for u in np.linspace(0, 1, n + 1)[1:]:
                t2 = th_from + step * u
                hp, _ = pl.held_pose(geo, s1, t2)
                poses.append((hp[s][0] - Ry(t2) @ (ay * SG[s] * ov), hp[s][1]))
            paths[s] = poses
        W = np.repeat(q[None], n + 1, axis=0)
        for s in SIDES:
            W[1:, SL[s]] = pl.ik.trace(s, paths[s], q[SL[s]], max_jump=0.25)
        Q, QD, QDD = retime(W, vmax=1.5, amax=A_PLAN, tmin=0.25)
        return Seg("fix", Q, QD, QDD, (0, 0), None, None, SIDES, "free")

    def plan_lower(self, geo, prm, th, slot=None):
        """Set-down: the held parcel is lowered until its lowest corner
        (measured pose) touches the belt, shifted in x so that its near
        face lands at the near-side target."""
        pl = self.pl
        q = self.ex.q.copy()
        if not self.cell.slots[slot].active:
            raise PlanFail("parcel gone")
        ps = self.cell.truth.parcel_state(slot)
        Rp = quat_mat(ps.quat)
        half = np.asarray(ps.size, float) / 2
        if ps.kind == "soft":
            half = np.array([half[0], half[1], half[2]])
        cor = np.array([ps.pos + Rp @ (half * np.array(sg)) for sg in
                        [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]])
        dz = float(np.min(cor[:, 2])) - (self.zb + 0.0005)
        x_near = float(np.min(cor[:, 0]))
        x_far = float(np.max(cor[:, 0]))
        lab = ps.label_normal
        tilt_left = math.degrees(math.acos(max(-1.0, min(1.0, lab[2]))))
        if tilt_left > 8.0:
            # settles about the lowest corner: footprint after the settle
            i = int(np.argmin(cor[:, 2]))
            xp = float(cor[i, 0])
            Lf = float(geo["Lx"]) if abs(th) > math.radians(135) else float(geo["H"])
            if float(np.mean(cor[:, 0])) < xp:
                x_near, x_far = xp - Lf, xp
            else:
                x_near, x_far = xp, xp + Lf
        lo_x = self.near_edge + 0.006
        hi_x = self.st.belt_center_x + self.st.half_width - 0.012
        dxs = 0.0
        if x_near < lo_x:
            dxs = lo_x - x_near
        elif x_far > hi_x:
            dxs = hi_x - x_far
        ay = geo["ay"]
        nw = max(6, int(math.hypot(dz, dxs) / 0.008))
        W = np.repeat(q[None], nw + 1, axis=0)
        par = []
        n_reach = {}
        for side in SIDES:
            p0, R = self.fk(side, q)
            poses = []
            for u in np.linspace(0, 1, nw + 1)[1:]:
                poses.append((p0 + np.array([dxs * u, 0.0, -dz * u]), R))
                if side == "Left":
                    par.append((ps.pos + np.array([dxs * u, 0.0, -dz * u]), Rp))
            Wp = pl.ik.trace_partial(side, poses, q[SL[side]])
            n_reach[side] = len(Wp)
            W[1:1 + len(Wp), SL[side]] = Wp
        nk = min(n_reach.values())
        if nk == 0:
            raise PlanFail("set-down unreachable")
        nw_full = nw
        W = W[:nk + 1]
        par = par[:nk]
        nw = nk
        # stop the descent where a gripper would come closer than 5 mm to
        # the belt or 8 mm to the structure (the parcel then drops the
        # last millimetres)
        col = pl.fresh_collider()
        last_ok = 0
        for i in range(1, nw + 1):
            r = col.check(W[i], [(ps.slot, par[i - 1][0], par[i - 1][1])], contact_ok=SIDES)
            if r["belt"] < 0.005 or r["struct"] < CLEAR_STRUCT or r["arms"] < CLEAR_ARMS:
                break
            last_ok = i
        self._lower_cut = (1.0 - last_ok / max(1, nw_full)) * dz
        if last_ok == 0:
            raise PlanFail("no clearance for the set-down")
        W = W[:last_ok + 1]
        par = par[:last_ok]
        Q, QD, QDD = retime(W, vmax=1.4, amax=A_PLAN, tmin=0.4)
        idx = np.clip(np.round(np.linspace(0, len(par) - 1, len(Q))).astype(int), 0, len(par) - 1)
        return Seg("lower", Q, QD, QDD, (0, 0), None, [par[i] for i in idx], SIDES, "free")

    def release(self, slot, v, res, squeezing=True, plan=None):
        if squeezing and self._try_predicted(slot, res, plan, dq_max=0.01):
            return
        try:
            seg, park, F_rel, info = self.plan_release(slot, v, squeezing=squeezing)
        except PlanFail as e:
            res["release_fail"] = str(e)[:300]
            if squeezing and self._try_predicted(slot, res, plan, dq_max=0.03, clear=0.003):
                return
            # retry with tight clearances (1.5 mm) before the last resort
            global CLEAR_STRUCT, CLEAR_BELT, CLEAR_ARMS, CLEAR_PARCEL
            saved = (CLEAR_STRUCT, CLEAR_BELT, CLEAR_ARMS, CLEAR_PARCEL)
            try:
                CLEAR_STRUCT = CLEAR_BELT = CLEAR_ARMS = CLEAR_PARCEL = 0.0015
                seg, park, F_rel, info = self.plan_release(slot, v, squeezing=squeezing)
                info = dict(info, fallback="tight")
                res["release"] = info
                t0r = self.cell.time
                self.play(seg, F_fn=F_rel)
                res["t_release"] = round(t0r + (info.get("n_gate", 0) + 3) * DT, 3)
                res["t_parked"] = self.cell.time
                return
            except PlanFail as e2:
                res["release_fail2"] = str(e2)[:200]
            finally:
                CLEAR_STRUCT, CLEAR_BELT, CLEAR_ARMS, CLEAR_PARCEL = saved
            # the predicted release with a larger posture difference (the
            # executor's rate guard bounds the first ticks), re-checked
            # against the measured parcel with tight clearances
            if self._try_predicted(slot, res, plan, dq_max=0.08, clear=0.0015):
                return
            # gentle open: the squeeze opens without lifting, the parcel rests
            # on the belt against the left palm; then a stop-state release
            if squeezing and self.cell.slots[slot].active:
                try:
                    self.gentle_open(slot)
                    seg, park, F_rel, info = self.plan_release(slot, v, squeezing=False)
                    res["release"] = dict(info, fallback="gentle_open")
                    t0r = self.cell.time
                    self.play(seg, F_fn=F_rel)
                    res["t_release"] = round(t0r + (info.get("n_gate", 0) + 3) * DT, 3)
                    res["t_parked"] = self.cell.time
                    return
                except PlanFail as e3:
                    res["release_fail3"] = str(e3)[:160]
            # last resort: lift both palms straight up and park slowly
            res["emergency"] = True
            self.emergency_clear(slot)
            return
        res["release"] = info
        n_free = info.get("n_gate", 0) + 3
        t0r = self.cell.time
        self.play(seg, F_fn=F_rel)
        if park is not None:
            self.play(park)
        res["t_release"] = round(t0r + n_free * DT, 3)
        res["t_parked"] = self.cell.time

    def _try_predicted(self, slot, res, plan, dq_max=0.01, clear=None):
        pr = None if plan is None else plan.get("release_seg")
        if pr is None or not self.cell.slots[slot].active:
            return False
        dq = float(np.max(np.abs(self.ex.q - plan["q_end_pred"])))
        if dq >= dq_max:
            return False
        seg, park, F_rel, info = pr
        ps = self.cell.truth.parcel_state(slot)
        d0 = ps.pos - seg.parcel[0][0]
        Rm = quat_mat(ps.quat)
        col = self.pl.fresh_collider()
        global CLEAR_STRUCT, CLEAR_BELT, CLEAR_ARMS, CLEAR_PARCEL
        saved = (CLEAR_STRUCT, CLEAR_BELT, CLEAR_ARMS, CLEAR_PARCEL)
        try:
            if clear is not None:
                CLEAR_STRUCT = CLEAR_BELT = CLEAR_ARMS = CLEAR_PARCEL = clear
            for i in range(0, len(seg.Q), 2):
                res_c = col.check(seg.Q[i], [(slot, seg.parcel[i][0] + d0, Rm)],
                                  contact_ok=SIDES if i < 40 else ())
                if not Collide.ok(res_c, contact=i < 60):
                    return False
        finally:
            CLEAR_STRUCT, CLEAR_BELT, CLEAR_ARMS, CLEAR_PARCEL = saved
        res["release"] = dict(info, used="predicted", dq=round(dq, 4))
        t0r = self.cell.time
        self.play(seg, F_fn=F_rel)
        res["t_release"] = round(t0r + (info.get("n_gate", 0) + 3) * DT, 3)
        res["t_parked"] = self.cell.time
        return True

    def gentle_open(self, slot):
        """Open the squeeze without lifting: the right palm backs off 2 cm,
        the left palm holds (the belt keeps the parcel against it)."""
        q0 = self.ex.q.copy()
        pR, RR = self.fk("Right", q0)
        tgt = [(pR + np.array([0.0, -0.02 * min_jerk(u), 0.0]), RR) for u in np.linspace(0, 1, 13)[1:]]
        W = self.pl.ik.trace("Right", tgt, q0[SL["Right"]], max_jump=0.1)
        Wq = np.repeat(q0[None], 13, axis=0)
        Wq[1:, SL["Right"]] = W
        Q, QD, QDD = retime(Wq, vmax=0.8, amax=5.0, tmin=0.5)
        pi = self.parcel(slot)
        mg = (pi.mass if pi is not None else 1.0) * G
        fsq = squeeze_force(pi.mass if pi is not None else 1.0)
        ay = np.array([0.0, 1.0, 0.0])

        def F(i, n):
            w = max(0.0, 1.0 - (i + 1) / max(1, n // 2))
            return {"Left": -ay * (0.85 * mg + fsq * w), "Right": ay * fsq * w if w > 0 else None}
        self.play(Seg("open", Q, QD, QDD), F_fn=F)
        self.hold(0.3, F={"Left": -ay * 0.85 * mg})

    def emergency_clear(self, slot):
        """Last resort: drop the squeeze, back off and rise, then park."""
        q0 = self.ex.q.copy()
        W = [q0]
        for side in SIDES:
            pass
        poses = {}
        try:
            for side in SIDES:
                p, R = self.fk(side, q0)
                tgt = [(p + np.array([0, SG[side] * 0.03 * u, 0.12 * u]), R)
                       for u in np.linspace(0, 1, 13)[1:]]
                poses[side] = self.pl.ik.trace(side, tgt, q0[SL[side]], max_jump=0.25)
            Wq = np.repeat(q0[None], 13, axis=0)
            for side in SIDES:
                Wq[1:, SL[side]] = poses[side]
            Q, QD, QDD = retime(Wq, vmax=1.2, amax=5.0, tmin=0.6)
            self.play(Seg("clear", Q, QD, QDD))
        except PlanFail:
            pass
        Qp, QDp, QDDp = joint_move(self.ex.q, Q_PARK, vmax=1.6, tmin=1.0, amax=5.0)
        self.play(Seg("park", Qp, QDp, QDDp))

    def give_up(self, slot, res, stage):
        """End a job cleanly: the parcel rests on the belt, released, the
        arms go back to the park posture."""
        res["give_up"] = stage
        v = self.belt_speed()
        if stage in ("catch", "stop"):
            # the parcel stands on the belt against the left palm: the left
            # palm accelerates with it to belt speed and leaves (gate), the
            # right palm is still hovering
            self.release(slot, v, res, squeezing=False)
            return
        self.release(slot, v, res)
