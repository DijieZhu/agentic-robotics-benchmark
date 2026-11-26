import dataclasses
import math
import os
import sys
from dataclasses import dataclass, field, replace

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import shift as SH
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline

from openarm_unit import base
from openarm_unit import hostcodec as hc
from openarm_unit import world as W
from openarm_unit.envelope import kt_from_registers

K = SH.K
SIDES = SH.SIDES
CYCLE = SH.CYCLE
R_PAD = K.CLAMP_LEVER_M
GRIP_OPEN = 0.45
GRIP_OPEN_SPEED = 2.0
DECLINE_AT_S = 0.5
DECLINE_PLANNED_S = K.SLOT["closing_end"]
RETREAT_AFTER_OPEN_S = 1.55
RETREAT_S = 1.5
BACK_S = 0.5
DECLINE_MODES = ("after_capture", "at_presentation")
DISPLACEMENT_SCOPES = ("tray", "shift")
LOWER_SPEED = 0.008
SHARE_OUT_S = 0.3
CLAMP_SHARE = 0.7
CLAMP_FLOOR_N = 10.0
SQUEEZE_ALLOWANCE_N = 4.5
FRICTION_WRENCH_IN_SQUEEZE = False
OPEN_TORQUE_NM = 0.5
SEAT_READ_AFTER_HELD_S = 0.1
GRIP_LEAD_S = 0.1
APPROACH_UP_M = 0.10
INT_TIME_CONSTANT_S = 0.3
INT_CLAMP_NM = 5.0
INT_JOINTS = 4


def maker_joint_table():
    m = base.maker_model()
    fr, dp = [], []
    for s in SIDES:
        for j in range(1, 8):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{s}_joint{j}")
            if jid < 0:
                raise RuntimeError(f"the maker's model has no joint openarm_{s}_joint{j}")
            da = int(m.jnt_dofadr[jid])
            fr.append(float(m.dof_frictionloss[da]))
            dp.append(float(m.dof_damping[da]))
    return tuple(fr), tuple(dp)


MAKER_MODEL_FRICTION14, MAKER_MODEL_DAMPING14 = maker_joint_table()
MAKER_MODEL_FRICTION = MAKER_MODEL_FRICTION14[:7]
MAKER_MODEL_DAMPING = MAKER_MODEL_DAMPING14[:7]


class PlanError(RuntimeError):
    ""


class MissingReply(SH.ReferenceAbort):
    ""


class SeatReadLate(SH.ReferenceAbort):
    ""


@dataclass(frozen=True)
class Beliefs:
    friction: dict
    torque_scale: tuple
    offset: tuple
    geometry: dict
    trays: tuple
    seats: dict
    shelf_pose: tuple
    thermal: dict
    squeeze: dict
    gripper: dict

    def fields(self):
        return tuple(f.name for f in dataclasses.fields(self))

    def differs_from(self, other):
        return tuple(n for n in self.fields() if _canon(getattr(self, n)) != _canon(getattr(other, n)))


def _canon(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _canon(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_canon(x) for x in v)
    if isinstance(v, np.ndarray):
        return tuple(_canon(x) for x in v.tolist())
    if isinstance(v, (float, np.floating)):
        return float(v)
    return v


ARM_MOD = tuple(i for i in range(16) if i not in (7, 15))


def beliefs_from_truth(unit, state, session, site, slots, spare_coil=True):
    t = unit.truth(("k", "b", "delta", "friction_profile"))
    prof = t.friction_profile
    profile = None if prof is None else (tuple(float(p[0]) for p in prof), tuple(tuple(p[1]) for p in prof))
    fb = [float(x) for x in unit.friction_base]
    base14 = tuple(fb[i] for i in ARM_MOD)
    damping = tuple(float(state.plant.damping[i]) for i in ARM_MOD)
    trays = []
    for sl in slots:
        trays.append((float(sl["mass_kg"]), (float(sl["fore_aft_m"]), float(sl["along_m"]), -float(sl["below_m"]))))
    ff = {s: float(fb[SH.GRIPPER[s]]) for s in SIDES}
    sized_for = {"trays": tuple(trays), "fixture_pose": SH.pose_of(site["F"]["true"]),
                 "shelf_pose": SH.pose_of(site["S"]["true"])}
    return Beliefs(
        friction={"terms": ((1.0, base14, profile),), "damping": damping},
        torque_scale=tuple(float(x) for x in t.k),
        offset=tuple(float(x) for x in t.b),
        geometry={"zero_residual": tuple(float(x) for x in t.delta),
                  "mounting": tuple((tuple(float(v) for v in p), tuple(float(v) for v in q))
                                    for p, q in state.plant.base_poses),
                  "fixture_pose": SH.pose_of(site["F"]["true"])},
        trays=tuple(trays),
        seats={s: tuple(SH.pose_of(x) for x in site["seats"][s]) for s in SIDES},
        shelf_pose=SH.pose_of(site["S"]["true"]),
        thermal={"h": tuple(float(x) for x in state.thermal.h), "ambient_C": float(session.ambient_C),
                 "start_offset_K": tuple(float(a - session.ambient_C) for a in session.start_winding_C())},
        squeeze={"kind": "reference", "factor": 1.0, "spare_coil": bool(spare_coil), "sized_for": sized_for},
        gripper={"finger_friction": ff, "pad_friction": dict(site["pad_friction"])},
    )


def friction_levels(fr, q14):
    lev = np.zeros(14)
    for w, b, prof in fr["terms"]:
        f = np.ones(14) if prof is None else _profile_factor(prof, q14)
        lev += float(w) * np.asarray(b, float) * f
    return lev


def _skew(r):
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def tray_rigid_error(e14, jac, tools, P):
    e14 = np.asarray(e14, float)
    P = np.asarray(P, float)
    dp, dth = np.zeros(3), np.zeros(3)
    for a, s in enumerate(SIDES):
        Jp, Jr = jac[s]
        ea = e14[7 * a:7 * a + 7]
        ps, ths = Jp @ ea, Jr @ ea
        dp += 0.5 * (ps + np.cross(ths, P - np.asarray(tools[s], float)))
        dth += 0.5 * ths
    return np.concatenate([dp, dth])


def cartesian_servo_stiffness(jac, tools, P, kp14):
    P = np.asarray(P, float)
    Ks = np.zeros((6, 6))
    for a, s in enumerate(SIDES):
        Jp, Jr = jac[s]
        J = np.vstack([Jp, Jr])
        kp = np.asarray(kp14[7 * a:7 * a + 7], float)
        KT = np.linalg.inv(J @ np.diag(1.0 / kp) @ J.T)
        X = np.eye(6)
        X[:3, 3:] = -_skew(np.asarray(tools[s], float) - P)
        Ks += X.T @ KT @ X
    return Ks


def wrench_to_joints(W, jac, tools, P):
    W = np.asarray(W, float)
    P = np.asarray(P, float)
    f, m = W[:3], W[3:]
    out = np.zeros(14)
    for a, s in enumerate(SIDES):
        Jp, Jr = jac[s]
        mt = m + np.cross(P - np.asarray(tools[s], float), f)
        out[7 * a:7 * a + 7] = Jp.T @ f + Jr.T @ mt
    return out


def plan_margins_by_kind(ref, kin):
    out = {}
    for c0, seg in ref.timeline:
        kind = getattr(seg, "kind", type(seg).__name__)
        if isinstance(seg, CartSeg):
            n = int(round(seg.T / CYCLE))
            ss = np.array([seg.f(i * CYCLE)[0] for i in range(n + 1)])
            mg = 1e9
            for s in SIDES:
                Q = seg.sp[s](ss)
                mg = min(mg, float(np.degrees(np.min(np.minimum(Q - kin.lo[s], kin.hi[s] - Q)))))
        elif isinstance(seg, JointSeg):
            n = int(round(seg.T / CYCLE))
            mg = 1e9
            for i in range(n + 1):
                st = seg.eval(i * CYCLE)
                for s in SIDES:
                    mg = min(mg, kin.margin_deg(s, st["q"][s]))
        else:
            mg = min(kin.margin_deg(s, seg.q_end[s]) for s in SIDES)
        out[kind] = min(out.get(kind, 1e9), mg)
    return out


_PROF_CACHE = {}


def _profile_factor(prof, q14):
    key = id(prof)
    fp = _PROF_CACHE.get(key)
    if fp is None or fp[0] is not prof:
        from openarm_unit.unit import FrictionProfile
        fp = (prof, FrictionProfile(prof[0], prof[1]))
        _PROF_CACHE[key] = fp
    return fp[1].factor(np.asarray(q14, float))


class CartSeg:

    def __init__(self, kin, p0, p1, T, profile, tray_quat, seats, d0=(0, 0, 0), d1=(0, 0, 0), sw0=45.0, sw1=45.0,
                 q_start=None, knots=None, kind="move", tray_quat1=None):
        self.kin, self.T, self.kind = kin, float(T), kind
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.d0, self.d1 = np.asarray(d0, float), np.asarray(d1, float)
        self.quat = tuple(tray_quat)
        self.quat1 = self.quat if tray_quat1 is None else tuple(tray_quat1)
        self.seats = seats
        self.f = profile(self.T)
        span = float(np.linalg.norm(self.p1 - self.p0) + np.linalg.norm(self.d1 - self.d0))
        n = knots or max(3, min(15, int(math.ceil(span / 0.02)) + 1))
        ss = np.linspace(0.0, 1.0, n)
        Q = {s: [] for s in SIDES}
        prev = q_start
        self.margin_deg = 1e9
        for sk in ss:
            ht = SH.hand_targets(self.p0 + sk * (self.p1 - self.p0), self._q(sk), seats,
                                 self.d0 + sk * (self.d1 - self.d0))
            sw = sw0 + sk * (sw1 - sw0)
            sol = {}
            for s in SIDES:
                q, mg = kin.solve(s, ht[s][0], ht[s][1], sw, None if prev is None else prev[s])
                if q is None and prev is not None:
                    q, mg = kin.solve(s, ht[s][0], ht[s][1], sw, None)
                if q is None:
                    raise PlanError(f"no IK for {s} at {ht[s][0]} swivel {sw}")
                sol[s] = q
                self.margin_deg = min(self.margin_deg, mg)
                Q[s].append(q)
            prev = sol
        self.sp = {s: CubicSpline(ss, np.array(Q[s]), axis=0) for s in SIDES}
        self.dsp = {s: self.sp[s].derivative(1) for s in SIDES}
        self.ddsp = {s: self.sp[s].derivative(2) for s in SIDES}
        self.q_end = {s: np.array(Q[s][-1]) for s in SIDES}
        self.q_begin = {s: np.array(Q[s][0]) for s in SIDES}

    def _q(self, s):
        if self.quat1 == self.quat:
            return self.quat
        return W.qnormalize(tuple(self.quat[i] + s * (self.quat1[i] - self.quat[i]) for i in range(4)))

    def sample_margin_deg(self):
        n = int(round(self.T / CYCLE))
        ss = np.array([self.f(i * CYCLE)[0] for i in range(n + 1)])
        mg = 1e9
        for k in SIDES:
            Q = self.sp[k](ss)
            mg = min(mg, float(np.degrees(np.min(np.minimum(Q - self.kin.lo[k], self.kin.hi[k] - Q)))))
        return mg

    def eval(self, t):
        s, sd, sdd = self.f(t)
        q, qd, qdd = {}, {}, {}
        for k in SIDES:
            q[k] = self.sp[k](s)
            d1 = self.dsp[k](s)
            qd[k] = d1 * sd
            qdd[k] = self.ddsp[k](s) * sd * sd + d1 * sdd
        dp = self.p1 - self.p0
        tray_pos = self.p0 + s * dp
        return {"q": q, "qd": qd, "qdd": qdd, "acc": dp * sdd, "tray_pos": tray_pos,
                "hands": SH.hand_targets(tray_pos, self._q(s), self.seats, self.d0 + s * (self.d1 - self.d0))}


class HoldSeg:
    def __init__(self, q, T, tray_pos, kind="hold"):
        self.q_end = {s: np.array(q[s]) for s in SIDES}
        self.q_begin = self.q_end
        self.T, self.kind = float(T), kind
        self.tray_pos = np.asarray(tray_pos, float)

    def eval(self, t):
        z = np.zeros(7)
        return {"q": self.q_end, "qd": {s: z for s in SIDES}, "qdd": {s: z for s in SIDES}, "acc": np.zeros(3),
                "tray_pos": self.tray_pos, "hands": None}


class JointSeg:
    def __init__(self, q0, q1, T, tray_pos=(0, 0, 0), kind="joint"):
        self.q0 = {s: np.array(q0[s], float) for s in SIDES}
        self.q_end = {s: np.array(q1[s], float) for s in SIDES}
        self.q_begin = self.q0
        self.T, self.kind = float(T), kind
        self.f = SH.quintic(self.T)
        self.tray_pos = np.asarray(tray_pos, float)

    def eval(self, t):
        s, sd, sdd = self.f(t)
        q, qd, qdd = {}, {}, {}
        for k in SIDES:
            dq = self.q_end[k] - self.q0[k]
            q[k], qd[k], qdd[k] = self.q0[k] + s * dq, sd * dq, sdd * dq
        return {"q": q, "qd": qd, "qdd": qdd, "acc": np.zeros(3), "tray_pos": self.tray_pos, "hands": None}


def target_swivels(kin_nominal, targets):
    out = []
    for t in targets:
        r = SH.solve_target(kin_nominal, tuple(t))
        out.append(45.0 if r is None else float(r["swivel"]))
    return out


def plan_moves(kin, targets, start, swivels=None, tray_quat=(1.0, 0.0, 0.0, 0.0), seats=None, q_start=None,
               swivel_choices=None):
    if swivels is None:
        swivels = target_swivels(kin, targets)
    sw_prev = swivels[0]
    prev = q_start
    p = np.asarray(start, float)
    moves, holds, segs = [], [], []
    chosen = []
    try:
        for tg, sw in zip(targets, swivels):
            best = None
            cands = [sw] + [x for x in SH.SWIVELS if x != sw] if swivel_choices is None else list(swivel_choices)
            for cand in cands:
                try:
                    mv = CartSeg(kin, p, tg, K.MOVE_S, SH.trapezoid, tray_quat, seats, sw0=sw_prev, sw1=cand,
                                 q_start=prev)
                except PlanError:
                    continue
                mg = mv.sample_margin_deg()
                if best is None or mg > best[1]:
                    best = (mv, mg, cand)
                if mg >= K.GENERATOR_STOP_MARGIN_DEG:
                    break
            if best is None:
                raise PlanError(f"no swivel solves the move to {tg}")
            mv, _, sw = best
            chosen.append(sw)
            hd = HoldSeg(mv.q_end, K.HOLD_S, tg)
            moves.append(mv)
            holds.append(hd)
            segs += [mv, hd]
            prev, p, sw_prev = mv.q_end, np.asarray(tg, float), sw
    except PlanError:
        return None
    return {"moves": moves, "holds": holds, "swivels": chosen, "segments": segs}


def _lerp_pose(a, b, k):
    p = tuple(float(a[0][i] + k * (b[0][i] - a[0][i])) for i in range(3))
    q = W.qnormalize(tuple(float(a[1][i] + k * (b[1][i] - a[1][i])) for i in range(4)))
    return p, q


class MinimalReference:

    def __init__(self, beliefs, slots, jobs, options=None):
        self.B = beliefs
        self.slots = slots
        self.jobs = jobs
        self.opt = dict(options or {})
        self.kp = np.array(K.REFERENCE_KP)
        self.kd = np.array(K.REFERENCE_KD)
        self.kin = SH.Kin(bases=beliefs.geometry["mounting"])
        self.kin_nom = SH.Kin()
        self.timeline = []
        self.grip = []
        self.share = []
        self.lengths = []
        self.parks = []
        self.decline_at = {}
        self.fig = {"captures": [], "plan_margin_min_deg": 1e9, "frame_use": {}, "squeeze": [], "late": 0}
        self._arrival_sign = np.ones(14)
        self._int = np.zeros(14)
        self._W = np.zeros(6)
        self._both_held = False
        self._replies = None
        self._disp = None
        self._p_sent = np.full(16, np.nan)
        self._sr = {"cap": {}, "armed": {}, "read_at": None, "done": False, "held": {s: False for s in SIDES}}
        self._abort = None
        self.fig["seat_reads"] = []
        self._clamp_true = {"left": 0.0, "right": 0.0}
        if self.opt.get("decline_mode", "after_capture") not in DECLINE_MODES:
            raise ValueError(f"decline_mode {self.opt.get('decline_mode')!r} is not one of {DECLINE_MODES}")
        if self.opt.get("displacement_scope", "tray") not in DISPLACEMENT_SCOPES:
            raise ValueError(f"displacement_scope {self.opt.get('displacement_scope')!r} is not one of "
                             f"{DISPLACEMENT_SCOPES}")
        self.slot_info = []
        self._decl = {}
        self._disp_slot = None
        self.fig["declines"] = []
        self.fig["missed_captures"] = []

    def setup(self, unit, site, slots):
        self.unit = unit
        Bu = unit.bus
        self.bus_of = [Bu.modules[i].bus for i in range(16)]
        self.esc = [Bu.esc_id[i] for i in range(16)]
        self.lim = [Bu.limits(i) for i in range(16)]
        self.reply_key = {(Bu.modules[i].bus, Bu.reply_id[i]): i for i in range(16)}
        self.i_full = {}
        for s in SIDES:
            g = SH.GRIPPER[s]
            st = unit.state
            self.i_full[s] = float(st.cal.imax[g]) * kt_from_registers(st.cal.registers.modules[g])
        mode = "mit" if self.B.squeeze.get("kind") == "mit" else "posforce"
        return {"gripper_mode": {"left": mode, "right": mode}}

    def start(self, unit):
        self.step0 = int(unit.bus.step_count)
        self._build(unit)

    def slot_plan_lengths(self):
        return list(self.lengths)

    def park_cycle(self, i):
        return self.parks[i]

    def _seat(self, k):
        return {s: self.B.seats[s][min(k, len(self.B.seats[s]) - 1)] for s in SIDES}

    def _tray_seated(self, rest_pose):
        return SH.Cell.tray_in(rest_pose)

    def _build(self, unit):
        cy = lambda t: int(round(t / CYCLE))
        self.timeline, self.grip, self.share, self.lengths, self.parks, self.decline_at = [], [], [], [], [], {}
        self.windows, self._wacc, self._wi = [], {}, 0
        self._slot_starts = []
        self.fig["squeeze"] = []
        t0 = unit.truth(("q",))
        q0 = {s: np.array([float(t0.q[base.HINGES.index(f"openarm_{s}_joint{j}")]) for j in range(1, 8)])
              for s in SIDES}
        Fp = self.B.geometry["fixture_pose"]
        tray_F = self._tray_seated(Fp)
        quat = tray_F[1]
        up = np.array([0.0, 0.0, APPROACH_UP_M])
        k_cap = 0
        sw_first = (self.jobs[0]["swivels"][0] if self.jobs and self.jobs[0] and self.jobs[0].get("swivels")
                    else 45.0)
        ready = CartSeg(self.kin, tray_F[0], tray_F[0], 0.1, SH.trapezoid, quat, self._seat(0), up, up, sw_first,
                        sw_first, knots=2)
        c = 0
        free = cy(K.FREE_INTERVAL_S)
        self._add(c, JointSeg(q0, ready.q_end, 5.0))
        c += cy(5.0)
        self._add(c, HoldSeg(ready.q_end, (free - c) * CYCLE, tray_F[0]))
        self.grip.append((0, ("open", None)))
        c = free
        q_cur = ready.q_end
        legacy_decline = self.opt.get("decline_mode", "after_capture") == "at_presentation"
        skip = self.opt.get("skip_job") or ()
        for i, sl in enumerate(self.slots):
            start = c
            info = {"slot": sl["slot"], "start": start, "seats": self._seat(k_cap), "k_cap": k_cap}
            self.slot_info.append(info)
            if self._declines(i) and legacy_decline:
                td = DECLINE_AT_S
                L = cy(td + K.SLOT["decline_next_s"])
                self._add(c, HoldSeg(q_cur, L * CYCLE, tray_F[0]))
                self.decline_at[i] = (start, start + cy(K.SLOT["decision_end"]), start + cy(td))
                self.lengths.append(L)
                self.parks.append(cy(td + K.SLOT["decline_clear_s"]))
                self.share.append((start, start + L, i, "none"))
                info.update(kind="decline_at_presentation", end=start + L, retreat=None)
                c += L
                continue
            if self._declines(i):
                seats = self._seat(k_cap)
                sw0 = self._approach_swivel(i, sw_first)
                ap = CartSeg(self.kin, tray_F[0], tray_F[0], 1.6, SH.trapezoid, quat, seats, up, (0, 0, 0), sw0, sw0,
                             q_start=q_cur, kind="approach")
                self._add(start, HoldSeg(q_cur, 0.4, tray_F[0]))
                self._add(start + cy(0.4), ap)
                c_dp = start + cy(DECLINE_PLANNED_S)
                self._add(start + cy(2.0), HoldSeg(ap.q_end, (c_dp - start - cy(2.0)) * CYCLE, tray_F[0],
                                                    kind="seat_hold"))
                self._add(c_dp, HoldSeg(ap.q_end, RETREAT_AFTER_OPEN_S, tray_F[0], kind="released"))
                c_r = c_dp + cy(RETREAT_AFTER_OPEN_S)
                ret = CartSeg(self.kin, tray_F[0], tray_F[0], RETREAT_S, SH.trapezoid, quat, seats, (0, 0, 0), up,
                              sw0, sw0, q_start=ap.q_end, kind="retreat")
                self._add(c_r, ret)
                k_cap += 1
                seats_n = self._seat(k_cap)
                sw_next = self._approach_swivel(i + 1, sw_first) if i + 1 < len(self.slots) else sw_first
                back = CartSeg(self.kin, tray_F[0], tray_F[0], BACK_S, SH.trapezoid, quat, seats_n, up, up, sw0,
                               sw_next, q_start=ret.q_end, knots=3, kind="transfer")
                c_b = c_r + cy(RETREAT_S)
                self._add(c_b, back)
                L = (c_dp - start) + cy(K.SLOT["decline_next_s"])
                self._add(c_b + cy(BACK_S), HoldSeg(back.q_end, (start + L - c_b - cy(BACK_S)) * CYCLE, tray_F[0]))
                q_cur = back.q_end
                lv = self._squeeze_levels(i, None, seated_only=True)
                self.grip.append((start + cy(2.4), ("close", None if lv is None else lv["hold"])))
                self.grip.append((c_dp, ("open", None)))
                self._decl[i] = {"planned": c_dp, "window_end": start + cy(K.SLOT["decision_end"]),
                                 "captured": {s: False for s in SIDES}, "done": False}
                self.lengths.append(L)
                self.parks.append((c_dp - start) + cy(K.SLOT["decline_clear_s"]))
                self.share.append((start, start + L, i, "none"))
                self.fig["squeeze"].append({"slot": sl["slot"], "decline": True, **(lv or {})})
                info.update(kind="decline", end=start + L, retreat=(c_r, c_r + cy(RETREAT_S)), sw_app=sw0,
                            sw_next=sw_next, seats_next=seats_n, k_next=k_cap)
                c += L
                continue
            job = self.jobs[i]
            seats = self._seat(k_cap)
            sw = job["swivels"]
            ap = CartSeg(self.kin, tray_F[0], tray_F[0], 1.6, SH.trapezoid, quat, seats, up, (0, 0, 0), sw[0], sw[0],
                         q_start=q_cur, kind="approach")
            self._add(start, HoldSeg(q_cur, 0.4, tray_F[0]))
            self._add(start + cy(0.4), ap)
            self._add(start + cy(2.0), HoldSeg(ap.q_end, 3.5, tray_F[0], kind="seat_hold"))
            Lp = np.array(tray_F[0]) + np.array([0, 0, K.LIFTED_POINT_M])
            lift = CartSeg(self.kin, tray_F[0], Lp, 1.5, SH.trapezoid, quat, seats, sw0=sw[0], sw1=sw[0],
                           q_start=ap.q_end, kind="lift")
            self._add(start + cy(5.5), lift)
            self._add(start + cy(7.0), HoldSeg(lift.q_end, 2.0, Lp, kind="hold"))
            fixed = job.get("swivel_fixed")
            skipping = skip is True or sl["slot"] in tuple(skip or ())
            n = len(job["targets"])
            Te = K.carry_end_s(n)
            if skipping:
                pre_t = np.asarray(job["targets"][-1], float)
                up_t = np.array([Lp[0], Lp[1], pre_t[2]])
                try:
                    mv1 = CartSeg(self.kin, Lp, up_t, K.MOVE_S, SH.trapezoid, quat, seats, sw0=sw[0], sw1=sw[0],
                                  q_start=lift.q_end)
                    mv2 = CartSeg(self.kin, up_t, pre_t, K.MOVE_S, SH.trapezoid, quat, seats, sw0=sw[0], sw1=sw[-1],
                                  q_start=mv1.q_end)
                except PlanError:
                    raise PlanError(f"slot {sl['slot']}: the way from L to the pre-place point does not solve")
                plan = {"moves": [mv1, mv2], "holds": [HoldSeg(mv1.q_end, K.HOLD_S, up_t),
                                                       HoldSeg(mv2.q_end, K.HOLD_S, pre_t)],
                        "swivels": [sw[0], sw[-1]]}
                move_starts = [start + cy(K.CARRY_START_S) + (n - 2) * cy(K.TARGET_S),
                               start + cy(K.CARRY_START_S) + (n - 1) * cy(K.TARGET_S)]
            else:
                plan = plan_moves(self.kin, job["targets"], Lp, swivels=sw, tray_quat=quat, seats=seats,
                                  q_start=lift.q_end, swivel_choices=None if fixed is None else (float(fixed),))
                if plan is None:
                    raise PlanError(f"slot {sl['slot']}: the carry does not solve on the believed kinematics")
                move_starts = [start + cy(K.CARRY_START_S) + k_t * cy(K.TARGET_S) for k_t in range(n)]
            for c_mv, mv, hd in zip(move_starts, plan["moves"], plan["holds"]):
                self._add(c_mv, mv)
                self._add(c_mv + cy(K.MOVE_S), hd)
            cc = start + cy(K.CARRY_START_S)
            for k_t in range(n):
                c_end = cc + cy(K.TARGET_S)
                tgt = tuple(float(x) for x in job["targets"][k_t])
                prev_t = tuple(float(x) for x in (job["targets"][k_t - 1] if k_t else Lp))
                self.windows.append((c_end - cy(K.HOLD_WINDOW_S), c_end, sl["slot"], k_t, tgt, prev_t))
                cc += cy(K.TARGET_S)
            assert cc == start + cy(Te)
            tray_S = self._tray_seated(self.B.shelf_pose)
            pre = np.asarray(job["targets"][-1], float)
            dz = float(np.linalg.norm(np.asarray(tray_S[0]) - pre))
            T_low = dz / LOWER_SPEED + 0.05
            low = CartSeg(self.kin, pre, tray_S[0], T_low, SH.ramped_constant, quat, seats, sw0=sw[-1], sw1=sw[-1],
                          q_start=plan["moves"][-1].q_end, kind="lower", tray_quat1=tray_S[1])
            self._add(cc, low)
            c_meet = cc + cy(T_low)
            c_open = c_meet + cy(SHARE_OUT_S) + cy(0.1)
            self._add(c_meet, HoldSeg(low.q_end, (cy(Te + 4.5) + start - c_meet) * CYCLE, tray_S[0], kind="placed"))
            ret = CartSeg(self.kin, tray_S[0], tray_S[0], 1.5, SH.trapezoid, tray_S[1], seats, (0, 0, 0), up, sw[-1],
                          sw[-1], q_start=low.q_end, kind="retreat")
            self._add(start + cy(Te + 4.5), ret)
            k_cap += 1
            seats_n = self._seat(k_cap)
            sw_next = sw_first
            if legacy_decline:
                for j in range(i + 1, len(self.slots)):
                    if not self._declines(j):
                        sw_next = self.jobs[j]["swivels"][0]
                        break
            elif i + 1 < len(self.slots):
                sw_next = self._approach_swivel(i + 1, sw_first)
            back_pre = CartSeg(self.kin, tray_S[0], tray_F[0], 2.0, SH.trapezoid, tray_S[1], seats, up, up, sw[-1],
                               sw[-1], q_start=ret.q_end, kind="transfer", tray_quat1=quat)
            back = CartSeg(self.kin, tray_F[0], tray_F[0], 0.5, SH.trapezoid, quat, seats_n, up, up, sw[-1], sw_next,
                           q_start=back_pre.q_end, knots=3, kind="transfer")
            self._add(start + cy(Te + 6.5), back_pre)
            self._add(start + cy(Te + 8.5), back)
            L = cy(Te + K.SLOT["next_tray_after_Te"])
            self._add(start + cy(Te + 9.0), HoldSeg(back.q_end, (L - cy(Te + 9.0)) * CYCLE, tray_F[0]))
            q_cur = back.q_end
            lv = self._squeeze_levels(i, [tuple(up_t), job["targets"][-1]] if skipping else job["targets"])
            lvh = None if lv is None else lv["hold"]
            lvm = None if lv is None else lv["move"]
            self.grip.append((start + cy(2.4), ("close", lvh)))
            self._slot_starts.append((start, start + L))
            lead = cy(GRIP_LEAD_S)
            for c_mv in move_starts:
                self.grip.append((c_mv - lead, ("level", lvm)))
                self.grip.append((c_mv + cy(K.MOVE_S), ("level", lvh)))
            self.grip.append((cc, ("level", lvh)))
            self.grip.append((c_open, ("open", None)))
            self.share.append((start, start + L, i, {"up": (start + cy(5.0), start + cy(5.5)),
                                                    "down": (c_meet, c_meet + cy(SHARE_OUT_S)), "open": c_open}))
            self.fig["squeeze"].append({"slot": sl["slot"], **(lv or {})})
            self.lengths.append(L)
            self.parks.append(cy(Te + K.SLOT["retreat_end_after_Te"]))
            r0 = start + cy(Te + 4.5)
            info.update(kind="skip" if skipping else "carry", end=start + L, retreat=(r0, r0 + cy(1.5)),
                        sw_app=sw[0], sw_next=sw_next, seats_next=seats_n, k_next=k_cap)
            c += L
        self.timeline.sort(key=lambda r: r[0])
        self.grip.sort(key=lambda r: r[0])
        self.fig["gripper_commands"] = [(int(c0), kind) for c0, (kind, _) in self.grip]
        self._ti = 0
        self._gi = 0
        self._si = 0
        self.end_cycle = c
        self._tray_F, self._quat, self._up = tray_F, quat, up

    def _approach_swivel(self, j, default):
        jb = self.jobs[j] if j < len(self.jobs) else None
        return float(jb["swivels"][0]) if jb and jb.get("swivels") else default

    def _declines(self, i):
        return self.slots[i]["label"] == "decline" and not self.opt.get("carry_declines")

    def _add(self, c, seg):
        self.timeline.append((int(c), seg))
        mg = getattr(seg, "margin_deg", None)
        if mg is not None:
            self.fig["plan_margin_min_deg"] = min(self.fig["plan_margin_min_deg"], mg)

    def _squeeze_levels(self, slot_i, targets, seated_only=False):
        sq = self.B.squeeze
        sz = sq.get("sized_for")
        if sq.get("kind") != "reference" or sz is None:
            return None
        assert not FRICTION_WRENCH_IN_SQUEEZE
        mass, com = (self.B.trays if sq.get("load") == "belief" else sz["trays"])[slot_i]
        tray_F = self._tray_seated(sz["fixture_pose"])
        tray_S = self._tray_seated(sz["shelf_pose"])
        Rt = SH.matrix_of_quat(tray_F[1])
        pw = sq.get("printed_worst")
        mu = self.B.gripper["pad_friction"] if pw is None else {s: float(pw["pad_friction"]) for s in SIDES}
        need = {"move": {s: 0.0 for s in SIDES}, "hold": {s: 0.0 for s in SIDES}}
        Rg = Rt @ SH.matrix_of_quat(SH.Cell.grip_quat())
        Lp = np.array(tray_F[0]) + np.array([0, 0, K.LIFTED_POINT_M])
        zero = (np.zeros(3), np.zeros(3), None, 0.0)
        path = [("hold", (zero[0], zero[1], None, 1.6)), ("hold", (zero[0], zero[1], None, 3.5))]
        if not seated_only:
            path += [("hold", (np.asarray(tray_F[0], float), Lp, SH.trapezoid, 1.5)),
                     ("hold", (zero[0], zero[1], None, 2.0))]
            p = np.asarray(Lp, float)
            for tg in targets:
                path += [("move", (p, np.asarray(tg, float), SH.trapezoid, K.MOVE_S)),
                         ("hold", (zero[0], zero[1], None, K.HOLD_S))]
                p = np.asarray(tg, float)
            pre = np.asarray(targets[-1], float)
            T_low = float(np.linalg.norm(np.asarray(tray_S[0]) - pre)) / LOWER_SPEED + 0.05
            path.append(("hold", (pre, np.asarray(tray_S[0], float), SH.ramped_constant, T_low)))
        for ph, (p0, p1, prof, T) in path:
            n = int(round(T / CYCLE))
            f_t = None if prof is None else prof(float(T))
            dp = p1 - p0
            for k in range(0, n + 1, 5):
                acc = np.zeros(3) if f_t is None else dp * f_t(k * CYCLE)[2]
                wr = SH.tray_wrench_shares(mass, com, acc, Rt) if not seated_only else \
                    {s: (np.zeros(3), np.zeros(3)) for s in SIDES}
                for s in SIDES:
                    f = Rg.T @ wr[s][0]
                    a = SQUEEZE_ALLOWANCE_N
                    Fs, Fn = math.hypot(f[0], f[2]) + a, abs(f[1]) + a
                    Fc = max(Fs / (CLAMP_SHARE * mu[s]), Fn / CLAMP_SHARE, CLAMP_FLOOR_N)
                    need[ph][s] = max(need[ph][s], Fc)
        if seated_only:
            need["move"] = dict(need["hold"])
        fac = float(sq.get("factor", 1.0))
        return {ph: {s: need[ph][s] * fac for s in SIDES} for ph in ("move", "hold")}

    def _seg_at(self, c):
        tl = self.timeline
        while self._ti + 1 < len(tl) and tl[self._ti + 1][0] <= c:
            self._ti += 1
        c0, seg = tl[self._ti]
        return seg, (c - c0) * CYCLE

    def _grip_at(self, c):
        g = self.grip
        while self._gi + 1 < len(g) and g[self._gi + 1][0] <= c:
            self._gi += 1
        return g[self._gi]

    def _share_at(self, c):
        for s0, s1, i, sh in self.share:
            if s0 <= c < s1:
                if sh == "none":
                    return i, 0.0
                u0, u1 = sh["up"]
                d0, d1 = sh["down"]
                if c < u0:
                    return i, 0.0
                if c < u1:
                    return i, (c - u0) / (u1 - u0)
                if c < d0:
                    return i, 1.0
                if c < d1:
                    return i, 1.0 - (c - d0) / (d1 - d0)
                return i, 0.0
        return None, 0.0

    def step(self, c, site_signal, replies):
        self._replies = replies
        if site_signal and site_signal.get("capture_missed"):
            self._on_missed(c, site_signal)
        seg, t = self._seg_at(c)
        st = seg.eval(t)
        slot_i, share = self._share_at(c)
        q, qd, qdd = st["q"], st["qd"], st["qdd"]
        tau, jac = self.kin.rne(q, qd, qdd)
        tools = {s: self.kin.tool_pose(s)[0].copy() for s in SIDES} if self.opt.get("integral") == "common" else None
        if share > 0.0:
            mass, com = self.B.trays[slot_i]
            Rt = SH.matrix_of_quat(SH.Cell.tray_in(self.B.geometry["fixture_pose"])[1])
            wr = SH.tray_wrench_shares(mass, com, st["acc"], Rt)
            for s in SIDES:
                Jp, Jr = jac[s]
                tau[s] = tau[s] + share * (Jp.T @ wr[s][0] + Jr.T @ wr[s][1])
        q14 = np.concatenate([q["left"], q["right"]])
        qd14 = np.concatenate([qd["left"], qd["right"]])
        lev = friction_levels(self.B.friction, q14)
        mv = np.abs(qd14) > 1e-6
        self._arrival_sign = np.where(mv, np.sign(qd14), self._arrival_sign)
        tau14 = np.concatenate([tau["left"], tau["right"]]) + np.asarray(self.B.friction["damping"]) * qd14 + lev * \
            self._arrival_sign
        frames = [None] * 16
        k = self.B.torque_scale
        b = self.B.offset
        dz = self.B.geometry["zero_residual"]
        kp, kd = self.kp, self.kd
        if self._abort is not None:
            raise self._abort
        add = self._additions(c, q14, qd14, jac, tools, st["tray_pos"])
        fade = self._displacement_fade(c)
        for a, i in enumerate(ARM_MOD):
            j = a % 7
            p = float(q14[a] - dz[i])
            if self._disp is not None:
                p = float(self._disp[a] + q14[a])
            v = float(qd14[a])
            if fade is not None and self._disp is not None:
                p = float(q14[a] + fade[0] * self._disp[a] - (1.0 - fade[0]) * dz[i])
                v = float(qd14[a] + fade[1] * (self._disp[a] + dz[i]))
            tff = (float(tau14[a]) + add[a] - b[i]) / k[i]
            kpi, kdi = float(kp[j]), float(kd[j])
            if self.opt.get("loop_wrist") and j >= 4:
                tff, kpi, kdi = self._host_wrist(i, a, p, v, tff)
            frames[i] = self._mit(i, p, v, kpi, kdi, tff)
            self._p_sent[i] = p
        for s in SIDES:
            frames[SH.GRIPPER[s]] = self._gripper_frame(c, s)
        decline = False
        for i, (d0, d1, dc) in self.decline_at.items():
            if dc <= c < d1:
                decline = True
        if self.opt.get("geom_mode") == "displacement":
            self._displacement_hook(c, q14)
        return frames, decline

    def _displacement_fade(self, c):
        if self._disp is None or self._disp_slot is None or \
                self.opt.get("displacement_scope", "tray") != "tray":
            return None
        r = self.slot_info[self._disp_slot].get("retreat")
        if r is None or c < r[0]:
            return None
        if c >= r[1]:
            self._disp = None
            self._disp_slot = None
            return None
        s, sd, _ = SH.quintic((r[1] - r[0]) * CYCLE)((c - r[0]) * CYCLE)
        return (1.0 - s, -sd)

    def _slot_at(self, c):
        for i, info in enumerate(self.slot_info):
            if info["start"] <= c < info["end"]:
                return i
        return None

    def _decline_watch(self, r):
        c = (int(r["step"]) - self.step0) // SH.SPC
        for i, d in self._decl.items():
            if d["done"]:
                continue
            info = self.slot_info[i]
            if c < info["start"] or c >= d["window_end"]:
                continue
            for s in SIDES:
                if int(r[f"g_{s}.weld_state"]) > 0:
                    d["captured"][s] = True
            if all(d["captured"].values()):
                d["done"] = True
                c_d = c + 1
                rec = {"slot": info["slot"], "both_captured_step": int(r["step"]), "decline_cycle": c_d,
                       "t_after_presentation_s": (c_d - info["start"]) * CYCLE, "planned_cycle": d["planned"]}
                self.fig["declines"].append(rec)
                if c_d >= d["window_end"]:
                    rec["too_late"] = True
                    return
                self._retime(d["planned"], c_d - d["planned"], c)
                self.decline_at[i] = (info["start"], d["window_end"], c_d)
            return

    def _retime(self, c_from, delta, c_now):
        if delta == 0:
            return
        f = lambda x: x + delta if x >= c_from else x
        for k, info in enumerate(self.slot_info):
            if info["start"] < c_from <= info["end"]:
                self.lengths[k] += delta
                self.parks[k] += delta
        self.timeline = sorted(((f(c0), sg) for c0, sg in self.timeline), key=lambda r: r[0])
        self.grip = sorted(((f(c0), g) for c0, g in self.grip), key=lambda r: r[0])
        sh = []
        for s0, s1, i, x in self.share:
            if x != "none":
                x = {"up": (f(x["up"][0]), f(x["up"][1])), "down": (f(x["down"][0]), f(x["down"][1])),
                     "open": f(x["open"])}
            sh.append((f(s0), f(s1), i, x))
        self.share = sh
        self.windows = [(f(w[0]), f(w[1])) + tuple(w[2:]) for w in self.windows]
        self.decline_at = {i: (f(a), f(b), f(dc)) for i, (a, b, dc) in self.decline_at.items()}
        for i, d in self._decl.items():
            if self.slot_info[i]["start"] >= c_from:
                d["planned"], d["window_end"] = f(d["planned"]), f(d["window_end"])
        for info in self.slot_info:
            later = info["start"] >= c_from
            info["start"], info["end"] = f(info["start"]), f(info["end"])
            if info.get("retreat") is not None:
                info["retreat"] = (f(info["retreat"][0]), f(info["retreat"][1]))
            if later and info.get("missed") is not None:
                info["missed"] = f(info["missed"])
        self._slot_starts = [(f(a), f(b)) for a, b in self._slot_starts]
        self.end_cycle = f(self.end_cycle)
        self._repoint(c_now)
        self.fig["gripper_commands"] = [(int(c0), kind) for c0, (kind, _) in self.grip]

    def _repoint(self, c_now):
        cyc = [c0 for c0, _ in self.timeline]
        self._ti = max(0, int(np.searchsorted(cyc, c_now, side="right")) - 1)
        gc = [c0 for c0, _ in self.grip]
        self._gi = max(0, int(np.searchsorted(gc, c_now, side="right")) - 1)

    def _on_missed(self, c, sig):
        i = sig.get("tray")
        if i is None:
            i = self._slot_at(c)
        if i is None or i >= len(self.slot_info):
            return
        info = self.slot_info[i]
        if info.get("missed") is not None or info["kind"] == "decline_at_presentation" or i in self.decline_at:
            return
        cy = lambda t: int(round(t / CYCLE))
        start, end_old = info["start"], info["end"]
        new_end = start + cy(K.SLOT["missed_next"])
        if i in self._decl:
            self._decl[i]["done"] = True
        c0s = [c0 for c0, _ in self.timeline]
        k = max(0, int(np.searchsorted(c0s, c, side="right")) - 1)
        c0, seg = self.timeline[k]
        st = seg.eval((c - c0) * CYCLE)
        q_now = {s: np.array(st["q"][s], float) for s in SIDES}
        tray_pos = np.asarray(st["tray_pos"], float)
        keep = lambda x: not (c <= x < end_old)
        self.timeline = [(x, sg) for x, sg in self.timeline if keep(x)]
        self.grip = [(x, g) for x, g in self.grip if keep(x)]
        self.windows = [w for w in self.windows if not (w[2] == info["slot"] and c <= w[0] < end_old)]
        self.share = [(s0, s1, j, x) if j != i else (start, end_old, i, "none") for s0, s1, j, x in self.share]
        self._retime(end_old, new_end - end_old, c - 1)
        tray_F, quat, up = self._tray_F, self._quat, self._up
        sw = info.get("sw_app", 45.0)
        self._add(c, HoldSeg(q_now, RETREAT_AFTER_OPEN_S, tray_pos, kind="released"))
        c_r = c + cy(RETREAT_AFTER_OPEN_S)
        ret = CartSeg(self.kin, tray_pos, tray_pos, RETREAT_S, SH.trapezoid, quat, info["seats"], (0, 0, 0), up, sw,
                      sw, q_start=q_now, kind="retreat")
        self._add(c_r, ret)
        c_b = c_r + cy(RETREAT_S)
        back = CartSeg(self.kin, tray_pos, tray_F[0], BACK_S, SH.trapezoid, quat, info.get("seats_next", info["seats"]),
                       up, up, sw, info.get("sw_next", sw), q_start=ret.q_end, knots=3, kind="transfer")
        self._add(c_b, back)
        self._add(c_b + cy(BACK_S), HoldSeg(back.q_end, max(0.0, (new_end - c_b - cy(BACK_S)) * CYCLE), tray_F[0]))
        self.grip.append((c, ("open", None)))
        self.timeline.sort(key=lambda r: r[0])
        self.grip.sort(key=lambda r: r[0])
        self.share = [(s0, s1, j, x) if j != i else (start, new_end, i, "none") for s0, s1, j, x in self.share]
        self.decline_at.pop(i, None)
        self.lengths[i] = cy(K.SLOT["missed_next"])
        self.parks[i] = cy(K.SLOT["missed_clear_end"])
        info.update(end=new_end, missed=c, retreat=(c_r, c_r + cy(RETREAT_S)))
        self._repoint(c - 1)
        self.fig["gripper_commands"] = [(int(x), kind) for x, (kind, _) in self.grip]
        self.fig["missed_captures"].append({"slot": info["slot"], "cycle": int(c),
                                            "t_after_presentation_s": (c - start) * CYCLE,
                                            "next_start_cycle": int(new_end)})

    def _use(self, name, x, lim):
        u = self.fig["frame_use"]
        u[name] = max(u.get(name, 0.0), abs(x) / lim)

    def _mit(self, i, p, v, kp, kd, t):
        pm, vm, tm = self.lim[i]
        self._use("p", p, pm)
        self._use("v", v, vm)
        self._use("kp", kp, hc.KP_RANGE[1])
        self._use("kd", kd, hc.KD_RANGE[1])
        self._use("t", t, tm)
        return hc.pack_mit(self.bus_of[i], self.esc[i], self.lim[i], p, v, kp, kd, t)

    def _gripper_frame(self, c, s):
        g = SH.GRIPPER[s]
        c0, (kind, lv) = self._grip_at(c)
        sg = 1.0 if s == "left" else -1.0
        kk = float(self.B.torque_scale[g])
        F = float(self.B.gripper["finger_friction"][s])
        sq = self.B.squeeze
        dz = float(self.B.geometry["zero_residual"][g])
        if sq.get("kind") == "mit":
            p = (0.0 if kind in ("close", "level") else sg * GRIP_OPEN) - dz
            return self._mit(g, p, 0.0, float(sq["kp"]), float(sq["kd"]), 0.0 if kind in ("close", "level") else 0.0)
        if kind == "open":
            ilim = (F + OPEN_TORQUE_NM) / (kk * self.i_full[s])
            return self._fp(g, sg * GRIP_OPEN - dz, GRIP_OPEN_SPEED, ilim)
        if sq.get("kind") == "fixed_ilim":
            return self._fp(g, 0.0 - dz, 1.0, float(sq["ilim"]))
        pw = sq.get("printed_worst")
        if pw is not None:
            kk = float(pw["torque_scale"])
            F = float(pw["finger_friction_Nm"])
        Fc = lv[s]
        C = Fc * R_PAD
        tau = C + F + 0.02
        if sq.get("spare_coil") and self._clamp_true[s] >= C - 1e-3:
            tau = max(C - 0.5 * F, 0.05)
        ilim = tau / (kk * self.i_full[s])
        return self._fp(g, 0.0 - dz, 1.0, ilim)

    def _fp(self, i, p, vlim, ilim):
        self._use("i_lim", ilim, 1.0)
        self._use("v_lim", vlim, 100.0)
        return hc.pack_posforce(self.bus_of[i], self.esc[i], p, vlim, min(1.0, max(0.0, ilim)))

    def _sensed(self):
        th = np.full(16, np.nan)
        om = np.full(16, np.nan)
        for f in self._replies or ():
            i = self.reply_key.get((f.bus, f.can_id))
            if i is None:
                continue
            r = hc.unpack_state(f.data, self.lim[i])
            if r is not None:
                th[i], om[i] = r["q"], r["dq"]
        return th, om

    def _additions(self, c, q14, qd14, jac=None, tools=None, tray_pos=None):
        mode = self.opt.get("integral")
        out = np.zeros(14)
        if not mode:
            return out
        e = self._joint_error()
        if mode == "common":
            if not self._both_held:
                self._W = np.zeros(6)
                return out
            P = np.asarray(tray_pos, float)
            ks = np.array([float(self.B.torque_scale[i]) for i in ARM_MOD])
            kp14 = ks * np.concatenate([self.kp, self.kp])
            Ki = cartesian_servo_stiffness(jac, tools, P, kp14) / INT_TIME_CONSTANT_S
            self._W = self._W + (Ki @ tray_rigid_error(e, jac, tools, P)) * CYCLE
            tau = wrench_to_joints(0.5 * self._W, jac, tools, P)
            self._int = np.clip(tau, -INT_CLAMP_NM, INT_CLAMP_NM)
            return self._int.copy()
        if mode == "left":
            e[7:] = 0.0
        elif mode != "per_arm":
            raise ValueError(f"integral mode {mode!r}")
        ki = np.zeros(14)
        for a in range(14):
            j = a % 7
            if j < INT_JOINTS:
                ki[a] = self.kp[j] / INT_TIME_CONSTANT_S
        self._int = np.clip(self._int + ki * e * CYCLE, -INT_CLAMP_NM, INT_CLAMP_NM)
        return self._int.copy()

    def _joint_error(self):
        th, _ = self._sensed()
        e = np.zeros(14)
        for a, i in enumerate(ARM_MOD):
            if not (np.isnan(th[i]) or np.isnan(self._p_sent[i])):
                e[a] = self._p_sent[i] - th[i]
        return e

    def _host_wrist(self, i, a, p, v, tff):
        th, om = self._sensed()
        if np.isnan(th[i]):
            return tff, 0.0, 0.0
        j = a % 7
        bel = float(self.opt.get("loop_damping", MAKER_MODEL_DAMPING)[j])
        tb = float(self.B.friction["damping"][a]) * v
        host = self.kp[j] * (p - th[i]) + self.kd[j] * (v - om[i])
        return tff + (host + (bel * v - tb)) / float(self.B.torque_scale[i]), 0.0, 0.0

    def _displacement_hook(self, c, q14):
        sr = self._sr
        if sr["read_at"] is None or c != sr["read_at"]:
            return
        th, _ = self._sensed()
        miss = [ARM_MOD[a] for a in range(14) if np.isnan(th[ARM_MOD[a]])]
        rec = self.fig["seat_reads"][-1]
        rec["read_cycle_done"] = int(c)
        if miss:
            rec["missing_modules"] = miss
            sr["read_at"] = None
            raise MissingReply(f"M-geom-d datum read at cycle {c}: no reply from module(s) {miss}")
        self._disp = np.array([th[ARM_MOD[a]] - q14[a] for a in range(14)])
        self._disp_slot = self._slot_at(c)
        rec["slot_index"] = self._disp_slot
        sr["read_at"] = None

    def on_trace(self, recs):
        if recs:
            for s in SIDES:
                self._clamp_true[s] = float(recs[-1][f"g_{s}.clamp_force"]) * R_PAD
            self._both_held = all(int(recs[-1][f"g_{s}.weld_state"]) == 2 for s in SIDES)
            self._hold_windows(recs)
        for r in recs:
            for s in SIDES:
                if int(r[f"g_{s}.weld_state"]) > 0 and not self.fig.get(f"_held_{s}"):
                    self.fig["captures"].append({"side": s, "step": int(r["step"]),
                                                 "clamp_N": float(r[f"g_{s}.clamp_force"])})
                self.fig[f"_held_{s}"] = int(r[f"g_{s}.weld_state"]) > 0
            self._seat_timing(r)
            if self._decl:
                self._decline_watch(r)

    def _seat_timing(self, r):
        sr = self._sr
        step = int(r["step"])
        ws = {s: int(r[f"g_{s}.weld_state"]) for s in SIDES}
        for s in SIDES:
            if ws[s] > 0 and not sr["held"][s]:
                sr["cap"][s] = step
                sr["armed"][s] = step + int(round(K.CAPACITY_RULE["armed_after_capture_s"] / SH.DT))
            sr["held"][s] = ws[s] > 0
        if all(ws[s] == 0 for s in SIDES):
            sr["cap"], sr["armed"], sr["done"] = {}, {}, False
            return
        if sr["done"] or not all(ws[s] == 2 for s in SIDES) or len(sr["armed"]) < 2:
            return
        sr["done"] = True
        c_held = (step - self.step0) // SH.SPC
        c_read = c_held + int(round(SEAT_READ_AFTER_HELD_S / CYCLE))
        read_step = self.step0 + c_read * SH.SPC
        armed = min(sr["armed"].values())
        rec = {"capture_steps": dict(sr["cap"]), "both_held_step": step, "both_held_cycle": int(c_held),
               "read_cycle": int(c_read), "read_step": int(read_step), "armed_step": int(armed)}
        self.fig["seat_reads"].append(rec)
        if read_step >= armed:
            rec["late"] = True
            if self.opt.get("geom_mode") == "displacement":
                self._abort = SeatReadLate(f"M-geom-d datum read at step {read_step} is not before the arming at "
                                           f"step {armed}")
            return
        sr["read_at"] = int(c_read)

    def _hold_windows(self, recs):
        w = self.windows
        for r in recs:
            c = (int(r["step"]) - self.step0) // SH.SPC
            while self._wi < len(w) and c >= w[self._wi][1]:
                self._wi += 1
            if self._wi >= len(w) or c < w[self._wi][0]:
                continue
            pose = np.asarray(r["tray.pose"], float)
            R = SH.matrix_of_quat(pose[3:7])
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
            a = self._wacc.setdefault(self._wi, [np.zeros(3), 0.0, 0, 0.0])
            a[0] += pose[:3]
            a[1] += tilt
            a[2] += 1
            a[3] = max(a[3], tilt)

    def hold_figures(self):
        rows = []
        for i, (c0, c1, slot, k_t, tgt, prev) in enumerate(self.windows):
            a = self._wacc.get(i)
            if not a or a[2] == 0:
                continue
            p = a[0] / a[2]
            rows.append({"slot": slot, "k": k_t, "target": tgt, "pos": p.tolist(), "from": prev,
                         "e_z_mm": 1e3 * abs(p[2] - tgt[2]), "e_xy_mm": 1e3 * math.hypot(p[0] - tgt[0], p[1] - tgt[1]),
                         "tilt_deg": a[1] / a[2]})
        rep = []
        by = {}
        for r in rows:
            by.setdefault((r["slot"], r["target"]), []).append(r)
        for (slot, tgt), rs in by.items():
            if len(rs) < 2:
                continue
            dirs = [np.array(tgt) - np.array(r["from"]) for r in rs]
            ok = any(SH._angle_deg(d1, d2) >= K.REPEAT_DIRECTION_MIN_DEG for d1 in dirs for d2 in dirs)
            if ok:
                d = max(float(np.linalg.norm(np.array(r1["pos"]) - np.array(r2["pos"]))) for r1 in rs for r2 in rs)
                rep.append({"slot": slot, "target": tgt, "rho_mm": 1e3 * d})
        return {"holds": rows, "repeatability": rep}

    def figures(self):
        out = {k: v for k, v in self.fig.items() if not k.startswith("_")}
        hf = self.hold_figures()
        out["hold_rows"] = hf["holds"]
        out["repeatability"] = hf["repeatability"]

        def stat(key, rows):
            v = sorted(r[key] for r in rows)
            if not v:
                return None
            return {"n": len(v), "median": v[len(v) // 2], "p90": v[min(len(v) - 1, int(math.ceil(0.9 * len(v))) - 1)],
                    "max": v[-1]}
        out["hold_stats"] = {k: stat(k, hf["holds"]) for k in ("e_z_mm", "e_xy_mm", "tilt_deg")}
        out["repeatability_stats"] = stat("rho_mm", hf["repeatability"])
        return out
