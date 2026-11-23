"""Privileged reference controller for the dual-arm label-facing station.

Inputs: simulator truth (parcel pose, size, mass, label normal, belt speed).
Outputs: 14 joint position targets + 2 gripper commands per 20 ms tick
(``cell.step``). Nothing else is written to the simulator.

Primitives (both grippers open, used as two-point "palms": the fingertip
pads pressed on a parcel face, tool axis toward the parcel):
  catch      the left palm waits as a stop in the parcel's path; the belt
             pushes the parcel into it and it stops (quasi-static from then on).
  flip       two-palm squeeze on the upstream / downstream faces, lift,
             rotate about the squeeze line (world y), set down (held).
  tumble     the stopping palm lifts the downstream end by friction; the
             belt drag on the pivot edge and the palm form a couple that tips
             the parcel backward onto its upstream face (downstream face up).
  gate       release: the right palm leaves, the left palm accelerates
             with the parcel to belt speed and escapes forward / up.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .execu import Executor, SL
from .geom import (DOWN, UP, XN, XP, YN, YP, aligned_dims, label_dir,
                   quat_mat, world_dims)
from .kin import ArmKin, rot_axis_angle
from .motion import DT, joint_move, min_jerk
from .paths import (Q_PALM_NOM, SG, SIDES, PlanFail, lerp_pose, palm_R,
                    solve_path, time_scale)

G = 9.81


@dataclass
class Params:
    tilt_deg: float = 25.0           # palm tilt (tool axis down) for horizontal pads
    overpress: float = 0.006         # m, squeeze overpress per palm
    f_squeeze: float = 18.0          # N, squeeze force feedforward per palm
    clear: float = 0.015             # m, corner clearance during rotation
    vmax: float = 2.6                # rad/s, joint speed limit of planned paths
    v_fast: float = 3.2              # rad/s, free moves
    gate_acc: float = 1.5            # m/s^2, gate acceleration of the parcel
    stop_ymax: float = 0.17          # left palm TCP |y| reach for the stop
    ready_y: float = 0.22
    ready_z: float = 0.26            # above the belt top
    ready_x: float = 0.08            # park: all arm links behind the belt's near edge
    hover_clear: float = 0.045       # right palm above the parcel top when hovering
    catch_timeout: float = 8.0
    tumble_extra_deg: float = 14.0


@dataclass
class ParcelInfo:
    slot: int
    uid: int
    kind: str
    size: tuple
    mass: float
    pos: np.ndarray
    vel: np.ndarray
    dims: np.ndarray                 # world x, y, z extents (aligned)
    yaw_err: float
    label: str
    label_err_deg: float
    label_angle_deg: float


class RefController:
    def __init__(self, cell, params: Params | None = None, log=None):
        self.cell = cell
        self.p = params or Params()
        self.ex = Executor(cell)
        self.kin = self.ex.kin
        self.st = cell.station
        self.belt = self.st.belt_height
        self.log = log or (lambda *a: None)
        self.q_ready = self._solve_ready()
        self.events = []
        self.stats = dict(belt_tip_contacts=0, max_belt_force=0.0,
                          robot_belt_ticks=0, arm_arm_ticks=0)
        self._mass = {}
        self._speed_log = []
        self.grip_cmd = (0.0, 0.0)

    # ------------------------------------------------------------------
    # postures
    # ------------------------------------------------------------------
    def _solve_ready(self):
        p = self.p
        q = np.zeros(14)
        for s in SIDES:
            pos = (p.ready_x, SG[s] * p.ready_y, self.belt + p.ready_z)
            R = palm_R(s, 0.0, 0.0, math.radians(p.tilt_deg))
            qs, ok, ep, er = self.kin[s].ik(pos, R=R, q0=Q_PALM_NOM[s],
                                            q_rest=Q_PALM_NOM[s], iters=400,
                                            margin=0.03)
            if not ok:
                qs, ok, ep, er = self.kin[s].ik(pos, axis=R[:, 2],
                                                q0=Q_PALM_NOM[s],
                                                q_rest=Q_PALM_NOM[s],
                                                iters=400, margin=0.03)
            if not ok:
                raise RuntimeError(f"ready posture IK failed for {s}")
            q[SL[s]] = qs
        return q

    # ------------------------------------------------------------------
    # observation (privileged)
    # ------------------------------------------------------------------
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
                                  ps.linvel.copy(), dims, yaw_err, lab, lerr,
                                  math.degrees(ps.label_angle)))
        return out

    def parcel(self, slot):
        for pi in self.parcels():
            if pi.slot == slot:
                return pi
        return None

    def belt_speed(self):
        return float(self.cell.truth.belt_speed()["nominal"])

    # ------------------------------------------------------------------
    # execution helpers
    # ------------------------------------------------------------------
    def _tick(self, q=None, grip=(0.0, 0.0)):
        self.ex.tick(q, None, None, grip)
        self._monitor()

    def _monitor(self):
        tr = self.cell.truth
        for c in tr.contacts(kinds=("robot-belt",)):
            self.stats["robot_belt_ticks"] += 1
            self.stats["max_belt_force"] = max(self.stats["max_belt_force"],
                                               c.force_normal)
            break

    def play(self, Q, grip=None, force=None, check=None):
        """Play joint samples; force(i) -> {side: F or None}; check(i) -> bool
        (True = stop early). Returns the number of ticks played."""
        from .motion import finite_diff
        if grip is None:
            grip = self.grip_cmd
        qd, qdd = finite_diff(Q, self.ex.q)
        import inspect
        caller = inspect.stack()[1].function
        u0 = None if self.ex._u_prev is None else self.ex._u_prev.copy()
        pk = 0.0
        jmax = 0.0
        for i in range(len(Q)):
            if force is not None:
                fs = force(i)
                for s in SIDES:
                    F = fs.get(s)
                    self.ex.force[s] = None if F is None else (
                        "site", f"{s}_tcp", np.asarray(F, float))
            self.ex.tick(Q[i], qd[i], qdd[i], grip)
            if u0 is not None:
                pk = max(pk, float(np.abs(self.ex._u_prev - u0).max() / DT))
            u0 = self.ex._u_prev.copy()
            jmax = max(jmax, float(np.abs(qd[i]).max()))
            self._monitor()
            if self.cell.done or (check is not None and check(i)):
                self._speed_log.append((caller, len(Q), pk, jmax))
                return i + 1
        self._speed_log.append((caller, len(Q), pk, jmax))
        return len(Q)

    def hold(self, seconds, grip=None, check=None, force=None):
        n = max(1, int(round(seconds / DT)))
        Q = np.repeat(self.ex.q[None], n, axis=0)
        return self.play(Q, grip, force, check)

    def joint_to(self, q_goal, vmax=None, tmin=0.3, grip=None,
                 force=None, check=None):
        Q, _, _ = joint_move(self.ex.q, q_goal, vmax=vmax or self.p.v_fast,
                             tmin=tmin)
        return self.play(Q, grip, force, check)

    def cart_plan(self, fns, n=30, vmax=None, tmin=0.15, q0=None):
        """fns: {side: f(u) -> (pos, R)}; returns joint samples (m, 14)."""
        q0 = self.ex.q if q0 is None else q0
        Q = np.repeat(np.asarray(q0, float)[None], n + 1, axis=0)
        for s, f in fns.items():
            poses = [f(u) for u in np.linspace(0, 1, n + 1)[1:]]
            Q[1:, SL[s]] = solve_path(self.kin[s], poses, q0[SL[s]])
        return time_scale(Q, vmax=vmax or self.p.vmax, tmin=tmin)

    def ik_pose(self, side, pos, R, q0=None):
        k = self.kin[side]
        seeds = [] if q0 is None else [q0]
        seeds += [Q_PALM_NOM[side]]
        for sd in seeds:
            q, ok, ep, er = k.ik(pos, R=R, q0=sd, q_rest=Q_PALM_NOM[side],
                                 iters=300, margin=0.01)
            if ok:
                return q
        raise PlanFail(f"{side}: pose IK failed at {np.round(pos, 3)}")

    def tcp(self, side):
        return self.cell.tcp_pose(side)

    def contact_force(self, side, slot=None):
        f = 0.0
        pre = side[0] + "_"
        for c in self.cell.truth.contacts(kinds=("robot-parcel",)):
            if (c.robot_geom.startswith(pre) or c.robot_geom.startswith(side)) \
                    and (slot is None or c.slot == slot):
                f += c.force_normal
        return f

    # ------------------------------------------------------------------
    # collision prediction of a planned joint path against parcels that
    # ride the belt (privileged: parcel poses from truth, constant belt speed)
    # ------------------------------------------------------------------
    def path_collides(self, Q, margin=0.012, every=2, exclude=()):
        import mujoco
        cell = self.cell
        m = cell.model
        if not hasattr(self, "_cd"):
            self._cd = mujoco.MjData(m)
        d = self._cd
        d.qpos[:] = cell.data.qpos
        d.qvel[:] = 0
        v = self.belt_speed()
        act = [(i, cell._slot_qadr[i]) for i, sl in enumerate(cell.slots)
               if sl.active and i not in exclude]
        if not act:
            return False
        y0 = {i: float(cell.data.qpos[qa + 1]) for i, qa in act}
        for k in range(0, len(Q), every):
            d.qpos[self.ex.dyn.qadr] = Q[k]
            t = (k + 1) * DT
            for i, qa in act:
                d.qpos[qa + 1] = y0[i] + v * t
            mujoco.mj_fwdPosition(m, d)
            for c in range(d.ncon):
                con = d.contact[c]
                g1, g2 = int(con.geom1), int(con.geom2)
                r1, r2 = cell._geom_is_robot[g1], cell._geom_is_robot[g2]
                if r1 == r2:
                    continue
                og = g2 if r1 else g1
                if cell.parcel_geom_slot[og] >= 0 and con.dist < margin:
                    return True
        return False

    def go_ready(self, wait_max=20.0):
        """Joint move to the park posture, delayed while it would sweep
        through a parcel riding the belt."""
        self.ex.force = {s: None for s in SIDES}
        t_end = self.cell.time + wait_max
        while True:
            Q, _, _ = joint_move(self.ex.q, self.q_ready, vmax=self.p.v_fast, tmin=0.5)
            if not self.path_collides(Q) or self.cell.time > t_end or self.cell.done:
                break
            self.hold(0.2)
            if self.cell.time > t_end:
                break
        self.play(Q)
        self.grip_cmd = (0.0, 0.0)

    # ------------------------------------------------------------------
    # planning of one parcel's job
    # ------------------------------------------------------------------
    def choose_job(self, pi: ParcelInfo):
        """Sequence of operations for the label direction."""
        lab = pi.label
        if lab == UP:
            return []
        if lab == XP:
            return [("flip", -math.pi / 2, 0.0)]
        if lab == DOWN:
            return [("flip", -math.pi, 0.0)]
        if lab == XN:
            return [("flip", math.pi / 2, math.pi / 2)]
        if lab == YP:
            return [("tumble",)]
        if lab == YN:
            return [("tumble",), ("flip", -math.pi, 0.0)]
        return []

    def catch_geometry(self, pi: ParcelInfo, v_belt, op):
        """Catch position (parcel centre y) and palm placement."""
        Lx, Ly, H = pi.dims
        p = self.p
        # gate travel of the left palm after the job
        d_gate = v_belt * v_belt / (2 * p.gate_acc) + 0.02
        y_hi = p.stop_ymax - d_gate - Ly / 2            # left TCP limit
        y_lo = -p.stop_ymax + Ly / 2                     # right TCP limit
        if op[0] == "tumble":
            y_lo = max(y_lo, -0.06 + Ly / 2)
        y_c = min(y_hi, max(y_lo, 0.0))
        return y_c

    # ------------------------------------------------------------------
    def run_parcel(self, slot, deadline=None):
        """Handle one parcel from its approach to its release. Returns a
        dict with the outcome."""
        pi = self.parcel(slot)
        if pi is None:
            return dict(slot=slot, result="gone")
        job = self.choose_job(pi)
        res = dict(slot=slot, uid=pi.uid, label0=pi.label, job=[j[0] for j in job],
                   t0=self.cell.time)
        if not job:
            res["result"] = "pass"
            return res
        v = self.belt_speed()
        ok = False
        for attempt in range(3):
            pi = self.parcel(slot)
            if pi is None:
                break
            job = self.choose_job(pi)
            if not job:
                ok = True
                break
            try:
                ok = self._do_job(slot, job, v, res)
            except PlanFail as e:
                self.log(f"  plan fail: {e}")
                res["plan_fail"] = str(e)
                ok = False
                self._safe_release(slot)
                break
            pi = self.parcel(slot)
            if pi is None or pi.label == UP:
                break
        res["t1"] = self.cell.time
        res["result"] = "done" if ok else "fail"
        return res

    # ------------------------------------------------------------------
    def _palm_frames(self, psi0, tilt, yaw=0.0):
        return {"Left": palm_R("Left", psi0, yaw, tilt),
                "Right": palm_R("Right", -psi0, yaw, tilt)}

    def _key_poses_ok(self, pi, op, x_s, y_c, z_s, R0, theta, hover_gap):
        """IK check of the key poses of a flip / tumble at a catch geometry."""
        Lx, Ly, H = pi.dims
        p = self.p
        k = self.kin
        def ok(side, pos, R, q0):
            q, good, ep, er = k[side].ik(pos, R=R, q0=q0, q_rest=Q_PALM_NOM[side],
                                         iters=200, margin=0.01)
            if not good:
                q, good, ep, er = k[side].ik(pos, R=R, q0=Q_PALM_NOM[side],
                                             q_rest=Q_PALM_NOM[side], iters=300, margin=0.01)
            return q if good else None
        yd, yu = y_c + Ly / 2, y_c - Ly / 2
        qL = ok("Left", np.array([x_s, yd, z_s]), R0["Left"], Q_PALM_NOM["Left"])
        if qL is None:
            return False
        if op[0] == "tumble":
            return True
        qR = ok("Right", np.array([x_s, yu - hover_gap, self.belt + H + p.hover_clear + 0.02]),
                R0["Right"], Q_PALM_NOM["Right"])
        if qR is None:
            return False
        qR1 = ok("Right", np.array([x_s, yu - hover_gap, z_s]), R0["Right"], qR)
        if qR1 is None:
            return False
        qR2 = ok("Right", np.array([x_s, yu, z_s]), R0["Right"], qR1)
        if qR2 is None:
            return False
        from .flip import corner_radius
        r = corner_radius(Lx, H, 0.0, z_s - (self.belt + H / 2))
        lift_z = max(self.belt + r + p.clear, z_s + 0.03)
        if abs(theta) > 2.5:
            lift_z = max(lift_z, 1.045)
        qs = {"Left": qL, "Right": qR2}
        reach = {}
        for side, yy in (("Left", yd), ("Right", yu)):
            q = qs[side]
            reach[side] = 0.0
            n = max(4, int(abs(theta) / math.radians(10)))
            for th in np.linspace(0, theta, n + 1)[1:]:
                R = rot_axis_angle((0, 1, 0), th) @ R0[side]
                q2, good, ep, er = k[side].ik(np.array([x_s, yy, lift_z]), R=R, q0=q,
                                              iters=120, margin=0.01, null_gain=0.0)
                if not good or np.abs(q2 - q).max() > 0.8:
                    break
                q = q2
                reach[side] = th
        got = min(abs(reach["Left"]), abs(reach["Right"]))
        need = abs(theta) if abs(theta) < 2.5 else math.radians(150)
        if got + 1e-6 < need:
            return False
        if abs(theta) > 2.5:
            got = max(need, got - math.radians(8))
        self._theta_ok = math.copysign(min(got, abs(theta)), theta)
        self._theta_room = math.copysign(max(0.0, min(got, math.radians(170)) - abs(self._theta_ok)), theta)
        # coarse check of the lift path (z_s -> lift_z) for both palms
        for side, yy, q in (("Left", yd, qL), ("Right", yu, qR2)):
            for z in np.linspace(z_s, lift_z, 5)[1:]:
                q2, good, ep, er = k[side].ik(np.array([x_s, yy, z]), R=R0[side], q0=q, iters=100,
                                              margin=0.01, null_gain=0.0)
                if not good or np.abs(q2 - q).max() > 0.5:
                    return False
                q = q2
        return True

    def _do_job(self, slot, job, v, res):
        p = self.p
        pi = self.parcel(slot)
        op = job[0]
        Lx, Ly, H = pi.dims
        psi0 = op[2] if op[0] == "flip" else 0.0
        theta = op[1] if op[0] == "flip" else 0.0
        if op[0] == "flip" and abs(psi0) > 0.1:
            tilts = [0.0]                                  # vertical pads: no tilt
        elif op[0] == "flip" and abs(op[1]) > 2.5:
            tilts = [0.0, math.radians(10), math.radians(20)]
        elif op[0] == "flip":
            tilts = [0.0, math.radians(10), math.radians(18), math.radians(p.tilt_deg)]
        else:
            tilts = [math.radians(15), math.radians(p.tilt_deg)]
        y_c0 = self.catch_geometry(pi, v, op)
        z_s = self.belt + max(0.025, min(H - 0.025, 0.5 * H))
        if op[0] == "tumble":
            z_s = self.belt + max(0.025, min(H - 0.02, 0.45 * H))
        near = pi.pos[0] - Lx / 2
        chosen = None
        z_s0 = z_s
        for tilt in tilts:
            R0 = self._palm_frames(psi0, tilt, pi.yaw_err)
            # the gripper housing (behind the pads, +-38 mm across) must clear
            # the belt surface: squeeze line high enough for this tilt
            if op[0] == "flip" and abs(psi0) < 0.1:
                z_min = self.belt + 0.038 * math.cos(tilt) - 0.062 * math.sin(tilt) + 0.010
            elif op[0] == "flip":
                z_min = self.belt + 0.020            # vertical pads: housing above
            else:
                z_min = self.belt + 0.020
            z_s = max(z_s0, z_min)
            if z_s > self.belt + H - 0.012:
                continue
            for dx in (0.0, -0.03, -0.05, 0.02):
                x_s = pi.pos[0] + dx
                if x_s - 0.055 < near or x_s + 0.055 > near + Lx:
                    continue
                for dy in (0.0, -0.02, 0.02, -0.04):
                    y_c = y_c0 + dy
                    for hg in (0.035, 0.022):
                        if self._key_poses_ok(pi, op, x_s, y_c, z_s, R0, theta, hg):
                            chosen = (x_s, y_c, hg, tilt, z_s)
                            break
                    if chosen: break
                if chosen: break
            if chosen: break
        if chosen is None:
            raise PlanFail("no feasible catch geometry")
        x_s, y_c, hover_gap, tilt, z_s = chosen
        self._tilt = tilt
        self._zs = z_s
        if op[0] == "flip" and abs(psi0) > 0.1:
            # vertical pads: close the fingers so both pads fit on the face
            g = float(np.clip((0.093 - (H - 0.024)) / 0.083, 0.0, 0.6))
            self.grip_cmd = (g, g)
        else:
            self.grip_cmd = (0.0, 0.0)
        self._hg = hover_gap
        R0 = self._palm_frames(psi0, tilt, pi.yaw_err)
        res["catch"] = dict(x_s=round(x_s, 3), y_c=round(y_c, 3), hover_gap=hover_gap)
        self._xs = x_s
        # --- stop pose of the left palm, hover pose of the right palm
        stop = np.array([x_s, y_c + Ly / 2 + 0.002, z_s])
        hover = np.array([x_s, y_c - Ly / 2 - hover_gap,
                          self.belt + H + p.hover_clear + 0.02])
        q_goal = self.ex.q.copy()
        q_goal[SL["Left"]] = self.ik_pose("Left", stop, R0["Left"],
                                          self.ex.q[SL["Left"]])
        if op[0] == "flip":
            q_goal[SL["Right"]] = self.ik_pose("Right", hover, R0["Right"],
                                               self.ex.q[SL["Right"]])
        # soft catch: the left palm waits upstream of its stop, matches the
        # parcel's speed just before contact and brakes it to rest at the stop
        T_acc, T_d = 0.15, 0.30
        y_w = stop[1] - v * (T_acc + T_d) / 2
        wait = stop.copy(); wait[1] = y_w
        q_goal[SL["Left"]] = self.ik_pose("Left", wait, R0["Left"], self.ex.q[SL["Left"]])
        y_face = pi.pos[1] + Ly / 2
        Qm, _, _ = joint_move(self.ex.q, q_goal, vmax=p.v_fast, tmin=0.5)
        # start of the speed ramp (parcel face at y_w + v*T_acc/2 at ramp end)
        t_start = (y_w + v * T_acc / 2 - y_face) / max(v, 1e-3) - T_acc
        if t_start < len(Qm) * DT + 0.05:
            raise PlanFail(f"too late for the catch ({t_start:.2f} s)")
        self.play(Qm)
        t_left = t_start - len(Qm) * DT
        # re-time against the measured parcel position
        pi2 = self.parcel(slot)
        if pi2 is not None:
            t_left = (y_w + v * T_acc / 2 - (pi2.pos[1] + Ly / 2)) / max(v, 1e-3) - T_acc
        if t_left > DT:
            self.hold(t_left)
        # the catch trajectory of the left palm
        kL = self.kin["Left"]
        qq = self.ex.q[SL["Left"]].copy()
        rows = []
        n1, n2 = int(round(T_acc / DT)), int(round(T_d / DT))
        yy = y_w
        for i in range(n1 + n2):
            t = (i + 1) * DT
            if i < n1:
                vp = v * t / T_acc
            else:
                vp = v * max(0.0, 1 - (t - T_acc) / T_d)
            yy += vp * DT
            tgt = stop.copy(); tgt[1] = min(yy, stop[1])
            qn, okk, ep, er = kL.ik(tgt, R=R0["Left"], q0=qq, iters=60, margin=0.005, null_gain=0.0)
            qq = qn
            row = self.ex.q.copy(); row[SL["Left"]] = qn
            rows.append(row)
        mg0 = pi.mass * G
        lvl = {"v": 0.0}
        def ramp_force(i):
            if i >= n1 - 2:
                lvl["v"] = min(1.0, lvl["v"] + DT / 0.2)
            return {"Left": np.array([0, -0.8 * mg0 * lvl["v"], 0])}
        self.play(np.array(rows), force=ramp_force)
        t_arrive = 0.0
        # --- wait for the stop: ramp a stop force feedforward as the parcel
        # comes close (no on/off chatter), require a steady stop
        mg = pi.mass * G
        st_ = {"n": 0, "lvl": 0.0}
        def gap_to_palm():
            ps = self.parcel(slot)
            if ps is None:
                return 0.0
            return stop[1] - (ps.pos[1] + ps.dims[1] / 2)
        st_["lvl"] = 1.0
        def stop_force(i):
            return {"Left": np.array([0, -0.8 * mg, 0])}
        hist = []
        def stopped(i):
            ps = self.parcel(slot)
            if ps is None:
                return True
            hist.append(float(ps.pos[1]))
            # the parcel rocks a little against the palm: judge the stop by its
            # position over 0.2 s, not by its velocity
            return len(hist) >= 10 and (max(hist[-10:]) - min(hist[-10:])) < 0.003
        n = self.hold(t_arrive + p.catch_timeout, check=stopped, force=stop_force)
        pi = self.parcel(slot)
        if pi is None or len(hist) < 10 or (max(hist[-10:]) - min(hist[-10:])) > 0.004:
            raise PlanFail("catch failed")
        self._stop_force = lambda i: {"Left": np.array([0, -0.8 * mg, 0])}
        res["t_catch"] = self.cell.time
        self.hold(0.1, force=self._stop_force)
        self._released = False
        op = job[0]
        pi = self.parcel(slot)
        if op[0] == "flip":
            th = getattr(self, "_theta_ok", op[1])
            self._flip(slot, pi, th, op[2], True, res)
            self._gate2(slot, v, res)
        elif op[0] == "tumble":
            self._tumble(slot, pi, res)
            res["t_release"] = self.cell.time
        self.go_ready()
        return True

    def _gate2(self, slot, v, res):
        """Two-palm release after a flip: still squeezing, both palms
        accelerate the parcel (on the belt) to belt speed, then the right palm
        falls back and rises, the left palm opens a gap and lifts off. The
        parcel never sits still against one palm (a tall parcel would be
        tumbled backward by the belt drag / palm couple)."""
        p = self.p
        ps = self.parcel(slot)
        mg = self._held["mg"]
        Lx2, Ly2, H2 = ps.dims if ps is not None else (0.2, 0.2, 0.3)
        top = self.belt + H2
        near = (ps.pos[0] - Lx2 / 2) if ps is not None else 0.16
        a = float(np.clip(0.25 * G * Ly2 / max(H2, 1e-3), 1.2, 4.0))
        T_a = max(0.1, v / a)
        n_a = max(3, int(math.ceil(T_a / DT)))
        pl, Rl = self.tcp("Left")
        pr, Rr = self.tcp("Right")
        Rl, Rr = self._held["R"]["Left"], self._held["R"]["Right"]
        kL, kR = self.kin["Left"], self.kin["Right"]
        traj = []                     # (dyL, dzL, dxL, dyR, dzR, force_on)
        y = 0.0
        for i in range(n_a):
            t = min((i + 1) * DT, T_a)
            y = 0.5 * a * t * t + v * max(0.0, (i + 1) * DT - T_a)
            traj.append((np.array([0, y, 0.0]), np.array([0, y, 0.0]), True))
        for i in range(2):
            y += v * DT
            traj.append((np.array([0, y, 0.0]), np.array([0, y, 0.0]), True))
        # release: right decelerates + rises; left opens a gap, then swings
        # toward the robot (out of the parcel's cross-section) keeping pace
        yl, yr = y, y
        vr = v
        dx_need = max(0.0, pl[0] - (near - 0.055))
        Tx = max(0.18, 1.5 * dx_need / 0.9)
        nx = int(math.ceil(Tx / DT))
        ne = 4 + nx + 6
        zr = 0.0
        for i in range(1, ne + 1):
            if i <= 4:
                yl += (v + 0.30) * DT
                dl = np.array([0.0, yl, 0.0])
            else:
                u = min(1.0, (i - 4) / nx)
                s_ = u * u * (3 - 2 * u)
                yl += (v + 0.04) * DT
                dl = np.array([-dx_need * s_, yl, 0.03 * s_])
            vr = max(0.0, vr - (v / 0.15) * DT)
            yr += vr * DT - (0.02 / ne)
            zr = min(0.08, zr + 0.08 / max(1, ne - 3)) if i > 3 else 0.0
            dr = np.array([0.0, yr, zr])
            traj.append((dl, dr, False))
        nd = int(math.ceil(0.25 / DT))
        for i in range(1, nd + 1):
            u = i / nd
            yl += (v + 0.04) * DT * (1 - u)
            dl = traj[-1][0].copy(); dl[1] = yl
            traj.append((dl, traj[-1][1].copy(), False))
        qL = self.ex.q[SL["Left"]].copy(); qR = self.ex.q[SL["Right"]].copy()
        Q = []
        F = []
        for j, (dl, dr, fon) in enumerate(traj):
            for side, k, P0, R, d, qq in (("Left", kL, pl, Rl, dl, qL), ("Right", kR, pr, Rr, dr, qR)):
                qn, ok, ep, er = k.ik(P0 + d, R=R, q0=qq, iters=80, margin=0.005, null_gain=0.0)
                if not ok and not fon:
                    qn2, ok2, ep2, er2 = k.ik(P0 + d, axis=R[:, 2], q0=qq, iters=60, margin=0.005, null_gain=0.0)
                    if ok2:
                        qn = qn2
                step = np.abs(qn - qq).max()
                if step > 0.10:
                    qn = qq + (qn - qq) * (0.10 / step)
                if side == "Left":
                    qL = qn
                else:
                    qR = qn
            row = self.ex.q.copy(); row[SL["Left"]] = qL; row[SL["Right"]] = qR
            Q.append(row)
            F.append(fon)
        Q = np.array(Q)
        fsq = getattr(self, "_fsq", p.f_squeeze)
        def force(i):
            if F[i]:
                return {"Left": np.array([0, -fsq, 0.0]), "Right": np.array([0, fsq, 0.0])}
            return {"Left": None, "Right": None}
        self.play(Q, force=force)
        self.ex.force = {s: None for s in SIDES}
        res["t_release"] = self.cell.time
        res["gate_acc"] = a

    # ------------------------------------------------------------------
    def _flip(self, slot, pi, theta, psi0, first, res):
        p = self.p
        Lx, Ly, H = pi.dims
        tilt = getattr(self, "_tilt", math.radians(p.tilt_deg))
        R0 = self._palm_frames(psi0, tilt, pi.yaw_err)
        c = pi.pos
        x_s = getattr(self, "_xs", c[0])
        z_s = getattr(self, "_zs", self.belt + max(0.025, min(H - 0.025, 0.5 * H)))
        ay = np.array([0.0, 1.0, 0.0])
        mg = pi.mass * G
        # current left palm (stop) position defines the downstream face
        pl, _ = self.tcp("Left")
        y_face_d = c[1] + Ly / 2
        y_face_u = c[1] - Ly / 2
        def PL(gap, z):
            return np.array([x_s, y_face_d + gap, z])
        def PR(gap, z):
            return np.array([x_s, y_face_u - gap, z])
        ov = p.overpress
        # right palm: hover -> beside the upstream face -> close in
        pr, _ = self.tcp("Right")
        if not first:
            # the right palm comes from wherever it is: up first
            up = np.array([pr[0], pr[1], max(pr[2], self.belt + H + 0.06)])
            Q = self.cart_plan({"Right": lambda u: lerp_pose(pr, R0["Right"], up, R0["Right"], u)}, n=12)
            self.play(Q, force=lambda i: {"Left": np.array([0, -0.9 * mg, 0])})
            pr = up
        hg = getattr(self, "_hg", 0.03)
        hov = PR(hg, max(pr[2], self.belt + H + 0.04))
        q_h = self.ik_pose("Right", hov, R0["Right"], self.ex.q[SL["Right"]])
        qg = self.ex.q.copy(); qg[SL["Right"]] = q_h
        self.joint_to(qg, force=lambda i: {"Left": np.array([0, -0.9 * mg, 0])})
        Q = self.cart_plan({"Right": lambda u: lerp_pose(hov, R0["Right"], PR(hg, z_s), R0["Right"], u)}, n=16)
        self.play(Q, force=lambda i: {"Left": np.array([0, -0.9 * mg, 0])})
        # close: right moves in to overpress, left adds overpress
        pl_now, Rl = self.tcp("Left")
        Q = self.cart_plan({
            "Right": lambda u: lerp_pose(PR(hg, z_s), R0["Right"], PR(-ov, z_s), R0["Right"], u),
            "Left": lambda u: lerp_pose(pl_now, R0["Left"], PL(-ov, z_s), R0["Left"], u)},
            n=20, vmax=1.0, tmin=0.35)
        nQ = len(Q)
        fsq0 = max(p.f_squeeze, 9.0 * pi.mass + 6.0)
        def f_close(i):
            a = min(1.0, (i + 1) / max(1, 0.7 * nQ))
            return {"Left": np.array([0, -max(0.8 * mg * (1 - a), fsq0 * a), 0]),
                    "Right": np.array([0, fsq0 * a, 0])}
        self.play(Q, force=f_close)
        fsq = max(p.f_squeeze, 9.0 * pi.mass + 6.0)
        self._fsq = fsq
        def f_hold(lift):
            def f(i):
                return {"Left": np.array([0, -fsq, mg / 2 if lift else 0.0]),
                        "Right": np.array([0, fsq, mg / 2 if lift else 0.0])}
            return f
        self.hold(0.15, force=f_hold(False))
        # lift + rotate + lower
        from .flip import corner_radius, lowest_after
        xs_rel, zs_rel = 0.0, z_s - (self.belt + H / 2)
        r = corner_radius(Lx, H, xs_rel, zs_rel)
        lift_z = max(self.belt + r + p.clear, z_s + 0.03)
        if abs(theta) > 2.5:
            lift_z = max(lift_z, 1.045)
        low = lowest_after(Lx, H, xs_rel, zs_rel, theta)
        down_z = self.belt - low + (0.001 if abs(abs(theta) - math.pi) < 0.05 or abs(abs(theta) - math.pi / 2) < 0.05 else 0.004)
        L0, R0p = PL(-ov, z_s), PR(-ov, z_s)
        L1, R1 = PL(-ov, lift_z), PR(-ov, lift_z)
        Q = self.cart_plan({"Left": lambda u: lerp_pose(L0, R0["Left"], L1, R0["Left"], u),
                            "Right": lambda u: lerp_pose(R0p, R0["Right"], R1, R0["Right"], u)},
                           n=20, vmax=1.6, tmin=0.35)
        self.play(Q, force=f_hold(True))
        def rot(s, P0):
            def f(u):
                return P0, rot_axis_angle((0, 1, 0), theta * u) @ R0[s]
            return f
        nrot = max(24, int(abs(theta) / math.radians(3)))
        Q = self.cart_plan({"Left": rot("Left", L1), "Right": rot("Right", R1)},
                           n=nrot, tmin=0.5 * abs(theta))
        self.play(Q, force=f_hold(True))
        self.hold(0.12, force=f_hold(True))
        # residual in-hand slip: keep rotating about y (closed loop on the
        # measured label normal) until the label is vertical, within the
        # arms' roll window (planned room + 25 deg)
        extra = 0.0
        if abs(abs(theta) - math.pi / 2) < 0.05 or abs(theta) > 2.5:
            room = math.radians(25) + abs(getattr(self, "_theta_room", 0.0))
            for it in range(3):
                if not self.cell.slots[slot].active:
                    break
                n = self.cell.truth.parcel_state(slot).label_normal
                err = -math.atan2(n[0], n[2])
                if abs(err) < math.radians(2.5):
                    break
                step = float(np.clip(err, -(room - abs(extra)), room - abs(extra)))
                if abs(step) < math.radians(1):
                    break
                try:
                    th0 = theta + extra
                    Q = self.cart_plan({s_: (lambda s_, P0: (lambda u: (P0, rot_axis_angle((0, 1, 0), th0 + step * u) @ R0[s_])))(s_, P0)
                                        for s_, P0 in (("Left", L1), ("Right", R1))}, n=max(6, int(abs(step) / math.radians(3))), tmin=0.25)
                    self.play(Q, force=f_hold(True))
                    self.hold(0.08, force=f_hold(True))
                    extra += step
                except PlanFail:
                    break
            theta = theta + extra
        res["tilt_fix_deg"] = round(math.degrees(extra), 1)
        Rf = {s: rot_axis_angle((0, 1, 0), theta) @ R0[s] for s in SIDES}
        L2, R2 = PL(-ov, down_z), PR(-ov, down_z)
        Q = self.cart_plan({"Left": lambda u: lerp_pose(L1, Rf["Left"], L2, Rf["Left"], u),
                            "Right": lambda u: lerp_pose(R1, Rf["Right"], R2, Rf["Right"], u)},
                           n=20, vmax=1.2, tmin=0.4)
        self.play(Q, force=f_hold(True))
        self.hold(0.1, force=f_hold(False))
        self._held = dict(R=Rf, mg=mg)
        ps = self.parcel(slot)
        res.setdefault("ops", []).append(dict(op="flip", theta=math.degrees(theta),
                                              label_after=None if ps is None else ps.label,
                                              t=self.cell.time))

    # ------------------------------------------------------------------
    def _tumble(self, slot, pi, res, chain=False):
        """Belt-assisted tumble (downstream face up): closed loop on the
        parcel's pose, the stopping left palm slides up its downstream face
        keeping a set overpress into the face; pad friction lifts the
        downstream end; once it is up by the angle where the couple of belt
        drag (on the upstream bottom edge) and palm tips the parcel, the palm
        lets go and gets out of the way."""
        p = self.p
        Lx, Ly, H = pi.dims
        mg = pi.mass * G
        pl, Rl = self.tcp("Left")
        ps0 = self.cell.truth.parcel_state(slot)
        R0p = quat_mat(ps0.quat)
        # parcel body axes along world y and z at the start
        iy = int(np.argmax(np.abs(R0p[1, :])))
        iz = int(np.argmax(np.abs(R0p[2, :])))
        sy = np.sign(R0p[1, iy]); sz = np.sign(R0p[2, iz])
        h0 = pl[2] - self.belt
        phi_t = None
        for deg in range(0, 70):
            ph = math.radians(deg)
            lhs = 0.9 * (Ly * math.sin(ph) + (H - 0.012) * math.cos(ph))
            rhs = (Ly / 2) * math.cos(ph) - (H / 2) * math.sin(ph)
            if lhs >= rhs:
                phi_t = ph
                break
        phi_go = (phi_t if phi_t is not None else math.radians(50)) + math.radians(12)
        ov = 0.008
        kL = self.kin["Left"]
        qq = self.ex.q[SL["Left"]].copy()
        s_cur = h0
        rise = 0.12                       # m/s along the face
        ok = False
        t0 = self.cell.time
        while self.cell.time - t0 < 3.0 and not self.cell.done:
            st = self.cell.truth.parcel_state(slot) if self.cell.slots[slot].active else None
            if st is None:
                break
            Rp = quat_mat(st.quat)
            ay = Rp[:, iy] * sy
            az = Rp[:, iz] * sz
            pitch = math.atan2(ay[2], ay[1])          # downstream end up
            if pitch > phi_go:
                ok = True
                break
            s_cur = min(H - 0.012, s_cur + rise * DT)
            face = st.pos + ay * (Ly / 2) + az * (s_cur - H / 2)
            tgt = face - ay * ov
            qn, good, ep, er = kL.ik(tgt, R=Rl, q0=qq, iters=50, margin=0.005, null_gain=0.0)
            if not good:
                qn, good, ep, er = kL.ik(tgt, axis=Rl[:, 2], q0=qq, iters=50, margin=0.005, null_gain=0.0)
            step = np.abs(qn - qq).max()
            if step > 0.08:
                qn = qq + (qn - qq) * (0.08 / step)
            qq = qn
            row = self.ex.q.copy(); row[SL["Left"]] = qn
            self.ex.force["Left"] = ("site", "Left_tcp", np.array([0, -0.5 * mg, 0.5 * mg]))
            self.ex.tick(row, None, None, self.grip_cmd)
            self._monitor()
        self.ex.force = {s_: None for s_ in SIDES}
        # out of the way: toward the robot and up
        pe, Re = self.tcp("Left")
        near = pi.pos[0] - Lx / 2
        tgt = np.array([min(pe[0], near - 0.06), pe[1] + 0.03, pe[2] + 0.06])
        try:
            Q = self.cart_plan({"Left": lambda u: lerp_pose(pe, Re, tgt, Re, u)}, n=12, vmax=2.5, tmin=0.25)
            self.play(Q)
        except PlanFail:
            self.hold(0.1)
        def landed(i):
            ps = self.parcel(slot)
            return ps is None or (ps.label in (UP, DOWN) and ps.label_err_deg < 12
                                  and float(np.linalg.norm(self.cell.truth.parcel_state(slot).angvel)) < 0.5)
        self.hold(1.5, check=landed)
        ps = self.parcel(slot)
        res.setdefault("ops", []).append(dict(op="tumble", label_after=None if ps is None else ps.label,
                                              t=self.cell.time, pitch_reached=ok))

    # ------------------------------------------------------------------
    def _gate(self, slot, v, res):
        """Release: the left palm (the stop) accelerates with the parcel to
        belt speed at a rate the parcel cannot tip from, then lifts off /
        backs away toward the robot while keeping pace with the parcel, and
        ends outside the parcel's path (the contact force is ~0 once the
        parcel runs at belt speed, so the lift-off drags nothing)."""
        p = self.p
        ps = self.parcel(slot)
        pl, Rl = self.tcp("Left")
        if ps is not None:
            Lx2, Ly2, H2 = ps.dims
            top = self.belt + H2
            near = ps.pos[0] - Lx2 / 2
            a = float(np.clip(0.25 * G * Ly2 / max(H2, 1e-3), 1.2, 4.0))
        else:
            top, near, a = self.belt + 0.3, 0.16, 1.5
        T_a = max(0.08, v / a)
        n_a = max(3, int(math.ceil(T_a / DT)))
        pts = []
        for i in range(n_a):
            t = min((i + 1) * DT, T_a)
            y = 0.5 * a * t * t + v * max(0.0, (i + 1) * DT - T_a)
            pts.append(np.array([0.0, y, 0.0]))
        y = pts[-1][1]
        # open a gap (palm a little faster than the belt) before lifting off
        for i in range(4):
            y += (v + 0.30) * DT
            pts.append(np.array([0.0, y, 0.0]))
        # lift-off: up (or toward the robot) at ~0.9 m/s, y keeps pace
        dz_need = max(0.0, top + 0.035 - pl[2])
        dx_need = max(0.0, pl[0] - (near - 0.06))
        if dz_need <= dx_need:
            disp = np.array([-0.25 * dz_need, 0.0, dz_need])
        else:
            disp = np.array([-dx_need, 0.0, 0.25 * dx_need])
        L = float(np.linalg.norm(disp))
        Te = max(0.18, 1.6 * L / 0.9)
        ne = int(math.ceil(Te / DT))
        p0 = np.array([0.0, y, 0.0])
        for i in range(1, ne + 1):
            u = i / ne
            s_ = u * u * (3 - 2 * u)                     # smooth start/stop
            yy = y + (v + 0.03) * (i * DT)
            pts.append(np.array([disp[0] * s_, yy, disp[2] * s_]))
        # decelerate in y to rest over 0.25 s
        vy = v + 0.03
        nd = int(math.ceil(0.25 / DT))
        last = pts[-1].copy()
        for i in range(1, nd + 1):
            u = i / nd
            last = last + np.array([0.0, vy * DT * (1 - u), 0.0])
            pts.append(last.copy())
        k = self.kin["Left"]
        qq = self.ex.q[SL["Left"]].copy()
        Q = []
        for j, dp in enumerate(pts):
            tgt = pl + dp
            if j < n_a + 4:
                qn, ok, ep, er = k.ik(tgt, R=Rl, q0=qq, iters=80, margin=0.005, null_gain=0.0)
            else:
                qn, ok, ep, er = k.ik(tgt, axis=Rl[:, 2], q0=qq, iters=60, margin=0.005,
                                      null_gain=0.0)
                if not ok:
                    qn, ok, ep, er = k.ik(tgt, q0=qq, iters=60, margin=0.005, null_gain=0.0)
            step = np.abs(qn - qq).max()
            if step > 0.11:                       # cap at 5.5 rad/s
                qn = qq + (qn - qq) * (0.11 / step)
            qq = qn
            qrow = self.ex.q.copy(); qrow[SL["Left"]] = qn
            Q.append(qrow)
        self.ex.force = {s: None for s in SIDES}
        self.play(np.array(Q))
        res["t_release"] = self.cell.time
        res["gate_acc"] = a

    # ------------------------------------------------------------------
    def _safe_release(self, slot):
        """Abort: drop any squeeze, lift both palms straight up, go ready."""
        self.ex.force = {s: None for s in SIDES}
        try:
            fns = {}
            for s in SIDES:
                ps_, R_ = self.tcp(s)
                back = ps_ + np.array([0, SG[s] * 0.04, 0.0])
                upp = back + np.array([0, 0, 0.10])
                fns[s] = (lambda a, b, R: (lambda u: lerp_pose(a, R, b, R, u)))(ps_, upp, R_)
            Q = self.cart_plan(fns, n=12, vmax=2.0, tmin=0.4)
            self.play(Q)
        except PlanFail:
            pass
        self.go_ready()
