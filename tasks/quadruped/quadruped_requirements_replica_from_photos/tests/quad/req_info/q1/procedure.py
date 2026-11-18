"""The acceptance procedure B-14 on a replica model (public code; every number comes from the replica model, the layer,
the catalogue and the lot description passed in).

    proc = Procedure(model, layer)          the duty, its joint states, the contact rule and the mass regressor
    ev = proc.evaluate(bom, cat, lots)      the 16 items, utilisations, currents and traces of an internal BOM
    der = proc.derived(bom, cat, lots)      the 16 derived quantities of B-16

The duty (B-05..B-08): a stand-up from the crouch; one steady cycle of the stop-and-go crawl (for each leg in the order
rr, fr, rl, fl a four-foot shift of the trunk, then a three-foot swing of the leg); the four swing-apex poses held static
with the pod at the proof factor. The trunk is level throughout. Joint angles come from the analytic leg inverse
kinematics, rates from the analytic leg Jacobian (qd = J^-1 (pd_foot - pd_trunk), qdd = J^-1 (pdd_foot - pdd_trunk -
Jd qd)).

The loads (B-14 f): h = mj_inverse(q, qd, qdd) with contacts off; f = pinv(J_Sb^T, rcond = 1e-10) h_b are the
minimum-norm forces at the stance feet's sphere centres (J_S: the stance feet's translational Jacobians, b: the six
free-joint columns or rows); the joint load torques are tau = h_j - J_Sj^T f. Inverse dynamics and the contact rule are
linear in the BOM masses, so six inverse passes give the exact regressor
    tau = tau0 + m_hip_roll B_hip_roll + m_hip_pitch B_hip_pitch + m_knee B_knee + m_pack b_pack + c m_pod b_pod
(c = the proof factor in the proof poses, else 1); the normal forces follow the same formula.
"""
import copy
import hashlib
import json
import math

import mujoco
import numpy as np

from . import layer as L
from . import lotmodel as LM
from . import quadmodel as Q
from . import schemas as S
from .quadmodel import ProcedureError

KINDS = ("standup", "crawl", "proof")
TERMS = ("hip_roll", "hip_pitch", "knee", "pack", "pod")
MASS_TERMS = TERMS[:-1]                                  # the BOM masses (the pod is the payload)
CRAWL_ORDER = ("rr", "fr", "rl", "fl")                   # B-14 d: the swing order
CRAWL_START = {"rr": -1 / 2, "fr": -1 / 4, "rl": 0, "fl": 1 / 4}   # foot x offsets before the first swing, in steps
BINDING_KINDS = ("stance", "swing", "proof", "rms")      # util_kind codes 0..3
BOM_LINES = ("modules", "battery", "driver", "hub")     # the part fields of an internal BOM
_QUINTIC = ((3, 10), (4, -15), (5, 6))                   # s(u) = 10 u^3 - 15 u^4 + 6 u^5 (B-14 c)


def _quintic(u):
    """s(u), s'(u), s''(u) of the quintic for an array of phase values (any real u)."""
    u = np.asarray(u, float)
    s = sum(c * u ** k for k, c in _QUINTIC)
    s1 = sum(k * c * u ** (k - 1) for k, c in _QUINTIC)
    s2 = sum(k * (k - 1) * c * u ** (k - 2) for k, c in _QUINTIC)
    return s, s1, s2


def _div(num, den):
    """num / den elementwise; where den <= 0: +inf if num > 0, else 0."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(den > 0, out, np.where(num > 0, np.inf, 0))


def _finite(x):
    """x converts to a finite float (False for NaN, +-inf, an integer too large for a float, or a non-number)."""
    try:
        return math.isfinite(float(x))
    except (OverflowError, TypeError, ValueError):
        return False


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------- leg kinematics
def leg_ik(g, p):
    """Joint angles (m, 3) = (hip_roll, hip_pitch, knee) in qpos of one leg for foot points p (m, 3) in the trunk frame
    (B-05). ProcedureError for an unreachable point; nothing is clipped."""
    p = np.atleast_2d(np.asarray(p, float))
    dx = p[:, 0] - g["hip_x"]
    dy = p[:, 1] - g["hip_y"]
    dz = p[:, 2]
    ell = g["lateral_m"]
    r2 = dy * dy + dz * dz
    if np.any(r2 <= ell * ell):
        raise ProcedureError("a foot point is unreachable: it lies inside the leg's lateral offset")
    hp = np.sqrt(r2 - ell * ell)
    qr = np.arctan2(dz, dy) - np.arctan2(-hp, g["sy"] * ell)
    qr = math.pi - np.mod(math.pi - qr, 2 * math.pi)            # wrapped to (-pi, pi]
    xp = dx
    zp = -(hp - g["drop_m"])
    l1, l2 = g["thigh_m"], g["shank_m"]
    cos_k = (xp * xp + zp * zp - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if np.any(np.abs(cos_k) > 1):
        raise ProcedureError("a foot point is out of reach of the thigh and the shank")
    qk = np.arccos(cos_k) * (-1 if g["knee_char"] == ">" else 1)
    phi = np.arctan2(-xp, -zp)
    beta = np.arctan2(l2 * np.sin(qk), l1 + l2 * np.cos(qk))
    return np.stack([qr, phi - beta, qk], axis=1)


def leg_foot(g, q):
    """Closed-form foot point (m, 3) in the trunk frame of one leg at joint angles q (m, 3)."""
    q = np.atleast_2d(np.asarray(q, float))
    qr, qp, qk = q[:, 0], q[:, 1], q[:, 2]
    l1, l2 = g["thigh_m"], g["shank_m"]
    wx = -l1 * np.sin(qp) - l2 * np.sin(qp + qk)
    wy = g["sy"] * g["lateral_m"]
    wz = -g["drop_m"] - l1 * np.cos(qp) - l2 * np.cos(qp + qk)
    c, s = np.cos(qr), np.sin(qr)
    return np.stack([g["hip_x"] + wx, g["hip_y"] + c * wy - s * wz, s * wy + c * wz], axis=1)


def leg_jacobian(g, q, qd=None):
    """Analytic leg Jacobian J (m, 3, 3) of the closed-form foot point with respect to (hip_roll, hip_pitch, knee); with
    rates qd (m, 3) also Jd qd (m, 3) = sum_ij d2p / (dq_i dq_j) qd_i qd_j."""
    q = np.atleast_2d(np.asarray(q, float))
    qr, qp, qk = q[:, 0], q[:, 1], q[:, 2]
    l1, l2 = g["thigh_m"], g["shank_m"]
    c, s = np.cos(qr), np.sin(qr)
    cp, sp, cpk, spk = np.cos(qp), np.sin(qp), np.cos(qp + qk), np.sin(qp + qk)
    zero = np.zeros_like(qr)
    wy = g["sy"] * g["lateral_m"] + zero
    wz = -g["drop_m"] - l1 * cp - l2 * cpk

    def rot(vx, vz):                     # R v for v = (vx, 0, vz)
        return np.stack([vx, -s * vz, c * vz], axis=1)

    def rot1(vx, vz):                    # R' v for v = (vx, 0, vz)
        return np.stack([zero, -c * vz, -s * vz], axis=1)

    p_r = np.stack([zero, -s * wy - c * wz, c * wy - s * wz], axis=1)
    p_p = rot(-l1 * cp - l2 * cpk, l1 * sp + l2 * spk)
    p_k = rot(-l2 * cpk, l2 * spk)
    J = np.stack([p_r, p_p, p_k], axis=2)
    if qd is None:
        return J
    qd = np.atleast_2d(np.asarray(qd, float))
    vr, vp, vk = qd[:, 0:1], qd[:, 1:2], qd[:, 2:3]
    p_rr = np.stack([zero, -c * wy + s * wz, -s * wy - c * wz], axis=1)
    p_rp = rot1(-l1 * cp - l2 * cpk, l1 * sp + l2 * spk)
    p_rk = rot1(-l2 * cpk, l2 * spk)
    p_pp = rot(l1 * sp + l2 * spk, l1 * cp + l2 * cpk)
    p_pk = rot(l2 * spk, l2 * cpk)
    p_kk = p_pk
    jdqd = (p_rr * vr * vr + p_pp * vp * vp + p_kk * vk * vk
            + 2 * (p_rp * vr * vp + p_rk * vr * vk + p_pk * vp * vk))
    return J, jdqd


def _solve(J, rhs):
    try:
        return np.linalg.solve(J, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        raise ProcedureError("a leg Jacobian is singular on the duty") from None


# ---------------------------------------------------------------------------------------------- BOM conversion
def bom_from_json(bom_json):
    """bom-quad/1 -> the internal BOM {"battery_family", "modules": {type: pid}, "battery", "driver", "driver_count",
    "hub"} (ProcedureError on any structural problem)."""
    probs = S.validate(bom_json, "bom")
    if probs:
        raise ProcedureError("invalid BOM: " + "; ".join(probs[:8]))
    mods, singles = {}, {"battery": [], "driver": [], "hub": []}
    for line in bom_json["parts"]:
        role = line["role"]
        if role == "actuator":
            if "joint_type" not in line:
                raise ProcedureError("actuator line %r has no joint_type" % line["part_id"])
            t = line["joint_type"]
            if t in mods:
                raise ProcedureError("two actuator lines for %s" % t)
            if int(line["count"]) != len(S.LEGS):
                raise ProcedureError("the actuator line of %s has count %s, not one per leg" % (t, line["count"]))
            mods[t] = line["part_id"]
        else:
            singles[role].append(line)
    missing = [t for t in S.JOINT_TYPES if t not in mods]
    if missing:
        raise ProcedureError("no actuator line for %s" % ", ".join(missing))
    for role, lines in singles.items():
        if len(lines) != 1:
            raise ProcedureError("the BOM needs exactly one %s line, it has %d" % (role, len(lines)))
    for role in ("battery", "hub"):
        if int(singles[role][0]["count"]) != 1:
            raise ProcedureError("the %s line must have count 1" % role)
    return {"battery_family": bom_json["battery_family"], "modules": {t: mods[t] for t in S.JOINT_TYPES},
            "battery": singles["battery"][0]["part_id"], "driver": singles["driver"][0]["part_id"],
            "driver_count": int(singles["driver"][0]["count"]), "hub": singles["hub"][0]["part_id"]}


def check_bom_parts(bom, cat, lines):
    """ProcedureError unless every part of the given lines of an internal BOM is in the catalogue with the role of its
    line (lines: "modules" = the three actuator lines, "battery", "driver", "hub")."""
    roles = {"battery": "battery", "driver": "driver", "hub": "hub"}
    pairs = []
    for line in lines:
        if line == "modules":
            pairs += [(bom["modules"][t], "actuator") for t in S.JOINT_TYPES]
        else:
            pairs.append((bom[line], roles[line]))
    for pid, role in pairs:
        if not isinstance(pid, str) or not cat.has(pid):
            raise ProcedureError("BOM part %r is not in the catalogue" % (pid,))
        if cat.role(pid) != role:
            raise ProcedureError("BOM part %r is a %s, not a %s" % (pid, cat.role(pid), role))


def bom_to_json(bom, cat):
    """Internal BOM -> bom-quad/1 (lines in LINES order) with its total_cost_usd."""
    parts = [{"part_id": bom["modules"][t], "role": "actuator", "joint_type": t, "count": len(S.LEGS)}
             for t in S.JOINT_TYPES]
    parts.append({"part_id": bom["battery"], "role": "battery", "count": 1})
    parts.append({"part_id": bom["driver"], "role": "driver", "count": int(bom["driver_count"])})
    parts.append({"part_id": bom["hub"], "role": "hub", "count": 1})
    return {"format": "bom-quad/1", "battery_family": bom["battery_family"], "parts": parts,
            "total_cost_usd": cat.bom_cost(bom)}


# ---------------------------------------------------------------------------------------------- evaluation context
class EvalContext:
    """Capabilities and speed envelopes of one (catalogue, lot description, payload, mission, ceiling), cached for
    repeated BOM evaluations (the optimum, the repair, the utilisation tables)."""

    def __init__(self, proc, cat, lots, payload_kg=None, mission_time_s=None, ceiling_usd=None):
        b = proc.layer["brief"]
        self.proc = proc
        self.cat = cat
        self.lots = lots
        self.payload = float(b["payload_top_kg"] if payload_kg is None else payload_kg)
        self.mission = float(b["mission_time_s"] if mission_time_s is None else mission_time_s)
        self.ceiling = float(b["ceiling_usd"] if ceiling_usd is None else ceiling_usd)
        self._caps = {}
        self._env = {}
        self._gpack = {}

    def cap(self, pid, family):
        key = (pid, family)
        if key not in self._caps:
            self._caps[key] = LM.capability(pid, family, self.cat, self.lots, self.proc.layer)
        return self._caps[key]

    def pack_factor(self, pid):
        if pid not in self._gpack:
            self._gpack[pid] = LM.pack_factor(pid, self.cat, self.lots, self.proc.layer)
        return self._gpack[pid]

    def env(self, pid, family, joint_type):
        """tau_s u*(|qd| / w0, d) over the stand-up and crawl samples of the type's 4 joints (n_move, 4)."""
        key = (pid, family, joint_type)
        if key not in self._env:
            cap = self.cap(pid, family)
            qd = np.abs(self.proc._qd_mv[:, self.proc.cols[joint_type]])
            x = np.where(qd > 0, _div(qd, cap["w0"]), 0)
            self._env[key] = cap["tau_s"] * LM.droop_fraction_array(x, cap["droop"])
        return self._env[key]


# ---------------------------------------------------------------------------------------------- the procedure
class Procedure:
    """Everything of B-14 that does not depend on the BOM, built once."""

    def __init__(self, model, layer):
        probs = S.validate(layer, "layer")
        if probs:
            raise ProcedureError("invalid layer: " + "; ".join(probs[:8]))
        read = dict(L.duty(layer), **L.electrical(layer), **L.payload(layer))
        read.update({k: v for k, v in L.policy(layer).items() if k != "battery_families"})
        bad = [k for k in sorted(read) if not _finite(read[k])]
        if bad:
            raise ProcedureError("invalid layer: brief values not finite: %s" % ", ".join(bad))
        self.model = copy.deepcopy(model)
        self.layer = copy.deepcopy(layer)
        self._duty = L.duty(self.layer)
        self.dyn = Q.build(self.model, self._duty["gravity_m_s2"])
        self.geo = Q.leg_geometry(self.model)
        self.cols = {t: np.array([S.JOINTS.index(leg + "_" + t) for leg in S.LEGS]) for t in S.JOINT_TYPES}
        self._build_duty()
        self._build_contact_maps_and_regressor()

    # ------------------------------------------------------------------ the duty (B-05..B-08)
    def _steps(self, key):
        dt = float(self._duty["dt_s"])
        T = float(self._duty[key])
        if not dt > 0:
            raise ProcedureError("dt_s = %r is not a positive time step" % dt)
        n = int(round(T / dt))
        if n < 1 or abs(n * dt - T) > 1e-9:
            raise ProcedureError("%s = %r is not a whole number of time steps dt_s = %r" % (key, T, dt))
        return T, n

    def _stance_feet(self):
        """{leg: (3,)} the B-05 foot points in the trunk frame."""
        r_f = float(self.model["geometry"]["foot_radius"])
        h_s = float(self._duty["stance_height_m"])
        return {leg: np.array([g["hip_x"], g["hip_y"] + g["sy"] * g["lateral_m"], r_f - h_s])
                for leg, g in self.geo.items()}

    def _segment_state(self, seg, u):
        """Trunk p, v, a (m, 3) and feet p, v, a (m, 4, 3) on a segment at phase values u (m,); the closed forms hold
        for any real u (the derivative self-test evaluates them beyond [0, 1])."""
        u = np.atleast_1d(np.asarray(u, float))
        m = len(u)
        t0 = np.asarray(seg["trunk_from"], float)
        t1 = np.asarray(seg["trunk_to"], float)
        feet = np.array([seg["feet"][leg] for leg in S.LEGS], float)
        F = np.repeat(feet[None], m, axis=0)
        Fd = np.zeros_like(F)
        Fdd = np.zeros_like(F)
        if seg["kind"] in ("standup", "shift"):
            T_s = float(seg["T_s"])
            s, s1, s2 = _quintic(u)
            d = t1 - t0
            T = t0 + s[:, None] * d
            Td = s1[:, None] * d / T_s
            Tdd = s2[:, None] * d / (T_s * T_s)
            return T, Td, Tdd, F, Fd, Fdd
        T = np.repeat(t1[None], m, axis=0)
        Td = np.zeros_like(T)
        Tdd = np.zeros_like(T)
        k = S.LEGS.index(seg["leg"])
        step = float(self._duty["step_length_m"])
        lift = float(self._duty["swing_height_m"])
        s, s1, s2 = _quintic(u)
        F[:, k, 0] += step * s
        F[:, k, 2] += lift * np.sin(math.pi * u) ** 2
        if seg["kind"] == "swing":
            T_s = float(seg["T_s"])
            Fd[:, k, 0] = step * s1 / T_s
            Fd[:, k, 2] = math.pi * lift * np.sin(2 * math.pi * u) / T_s
            Fdd[:, k, 0] = step * s2 / (T_s * T_s)
            Fdd[:, k, 2] = 2 * math.pi * math.pi * lift * np.cos(2 * math.pi * u) / (T_s * T_s)
        return T, Td, Tdd, F, Fd, Fdd

    def _joint_states(self, T, Td, Tdd, F, Fd, Fdd, rates=True):
        """q, qd, qdd (m, 12) in qpos convention from the trunk and foot states (the trunk never rotates)."""
        m = len(T)
        q = np.zeros((m, len(S.JOINTS)))
        qd = np.zeros_like(q)
        qdd = np.zeros_like(q)
        for i, leg in enumerate(S.LEGS):
            g = self.geo[leg]
            ql = leg_ik(g, F[:, i] - T)
            q[:, 3 * i:3 * i + 3] = ql
            if rates:
                J = leg_jacobian(g, ql)
                v = _solve(J, Fd[:, i] - Td)
                _, jdqd = leg_jacobian(g, ql, v)
                qd[:, 3 * i:3 * i + 3] = v
                qdd[:, 3 * i:3 * i + 3] = _solve(J, Fdd[:, i] - Tdd - jdqd)
        return q, qd, qdd

    def _build_duty(self):
        d = self._duty
        h_s = float(d["stance_height_m"])
        h_0 = float(d["crouch_height_m"])
        step = float(d["step_length_m"])
        k_sw = float(d["sway_fraction"])
        T_up, n_up = self._steps("standup_time_s")
        T_sh, n_sh = self._steps("shift_time_s")
        T_sw, n_sw = self._steps("swing_time_s")
        P = self._stance_feet()
        r_f = float(self.model["geometry"]["foot_radius"])
        world_stance = {leg: np.array([P[leg][0], P[leg][1], r_f]) for leg in S.LEGS}
        self.segments = []
        rows = {k: [] for k in KINDS}           # (segment index, u array, stance (m, 4), swing (m,))

        def add(seg, kind, u, stance, swing):
            self.segments.append(seg)
            rows[kind].append((len(self.segments) - 1, np.asarray(u, float), stance, swing))

        def feet_rec(F):
            return {leg: [float(v) for v in F[leg]] for leg in S.LEGS}

        # stand-up (B-06): crouch -> stance, all four feet down
        seg = {"kind": "standup", "leg": None, "T_s": T_up, "n": n_up, "trunk_from": [0, 0, h_0],
               "trunk_to": [0, 0, h_s], "feet": feet_rec(world_stance)}
        add(seg, "standup", np.arange(n_up + 1) / n_up, np.ones((n_up + 1, len(S.LEGS)), bool),
            np.full(n_up + 1, -1))

        # crawl (B-07): the steady cycle, starting at the previous cycle's last shift target
        F = {leg: world_stance[leg] + np.array([CRAWL_START[leg] * step, 0, 0]) for leg in S.LEGS}

        def target(feet, swing_leg):
            c4 = np.mean([feet[leg][:2] for leg in S.LEGS], axis=0)
            c3 = np.mean([feet[leg][:2] for leg in S.LEGS if leg != swing_leg], axis=0)
            xy = c4 + k_sw * (c3 - c4)
            return np.array([xy[0], xy[1], h_s])

        last = CRAWL_ORDER[-1]
        prev = {leg: F[leg].copy() for leg in S.LEGS}
        prev[last] = prev[last] - np.array([step, 0, 0])
        trunk = target(prev, last)
        self._crawl_start = trunk.copy()
        proofs = []
        for leg in CRAWL_ORDER:
            k = S.LEGS.index(leg)
            tgt = target(F, leg)
            seg = {"kind": "shift", "leg": None, "T_s": T_sh, "n": n_sh, "trunk_from": [float(v) for v in trunk],
                   "trunk_to": [float(v) for v in tgt], "feet": feet_rec(F)}
            add(seg, "crawl", np.arange(1, n_sh + 1) / n_sh, np.ones((n_sh, len(S.LEGS)), bool), np.full(n_sh, -1))
            trunk = tgt
            stance = np.ones((n_sw, len(S.LEGS)), bool)
            stance[:-1, k] = False
            swing = np.full(n_sw, k)
            swing[-1] = -1
            seg = {"kind": "swing", "leg": leg, "T_s": T_sw, "n": n_sw, "trunk_from": [float(v) for v in trunk],
                   "trunk_to": [float(v) for v in trunk], "feet": feet_rec(F)}
            add(seg, "crawl", np.arange(1, n_sw + 1) / n_sw, stance, swing)
            proofs.append({"kind": "proof", "leg": leg, "T_s": 0, "n": 1, "trunk_from": [float(v) for v in trunk],
                           "trunk_to": [float(v) for v in trunk], "feet": feet_rec(F)})
            F[leg] = F[leg] + np.array([step, 0, 0])
        self.cycle_closure_m = float(np.abs(trunk - (self._crawl_start + np.array([step, 0, 0]))).max())
        # proof poses (B-08): the swing apex (u = 1/2) of each step, three stance feet, static
        for seg in proofs:
            k = S.LEGS.index(seg["leg"])
            stance = np.ones((1, len(S.LEGS)), bool)
            stance[0, k] = False
            add(seg, "proof", np.array([1 / 2]), stance, np.array([k]))

        self.n, self.q, self.qd, self.qdd = {}, {}, {}, {}
        self.trunk_p, self.trunk_v, self.trunk_a, self.feet_p = {}, {}, {}, {}
        self.stance, self.swing_leg, self.segment, self._u = {}, {}, {}, {}
        for kind in KINDS:
            parts = []
            for si, u, stance, swing in rows[kind]:
                T, Td, Tdd, Fp, Fd, Fdd = self._segment_state(self.segments[si], u)
                static = kind == "proof"
                if static:
                    Td, Tdd, Fd, Fdd = (np.zeros_like(x) for x in (Td, Tdd, Fd, Fdd))
                q, qd, qdd = self._joint_states(T, Td, Tdd, Fp, Fd, Fdd, rates=not static)
                parts.append((T, Td, Tdd, Fp, q, qd, qdd, stance, swing, np.full(len(u), si), u))
            merged = [np.concatenate([p[i] for p in parts]) for i in range(len(parts[0]))]
            (self.trunk_p[kind], self.trunk_v[kind], self.trunk_a[kind], self.feet_p[kind], self.q[kind],
             self.qd[kind], self.qdd[kind], self.stance[kind], self.swing_leg[kind], self.segment[kind],
             self._u[kind]) = merged
            self.n[kind] = len(merged[0])
        self.stance_q = self._joint_states(np.array([[0, 0, h_s]], float), np.zeros((1, 3)), np.zeros((1, 3)),
                                           np.array([[world_stance[leg] for leg in S.LEGS]]), np.zeros((1, 4, 3)),
                                           np.zeros((1, 4, 3)), rates=False)[0][0]
        lohi = Q.stops_qpos(self.model)
        lo = np.array([lohi[j][0] for j in S.JOINTS])
        hi = np.array([lohi[j][1] for j in S.JOINTS])
        allq = np.concatenate([self.q[k] for k in KINDS])
        self.stop_margin_rad = float(np.min(np.minimum(allq - lo, hi - allq)))
        # concatenated stand-up + crawl views ("move" samples)
        self._qd_mv = np.concatenate([self.qd["standup"], self.qd["crawl"]])
        self._qdd_mv = np.concatenate([self.qdd["standup"], self.qdd["crawl"]])
        self._stance_mv = np.concatenate([self.stance["standup"], self.stance["crawl"]])
        self._crawl = slice(self.n["standup"], self.n["standup"] + self.n["crawl"])

    # ------------------------------------------------------------------ dynamics (B-14 f, m)
    def _states(self, kind):
        dyn = self.dyn
        n = self.n[kind]
        qpos = np.zeros((n, dyn.model.nq))
        qvel = np.zeros((n, dyn.model.nv))
        qacc = np.zeros((n, dyn.model.nv))
        qpos[:, 0:3] = self.trunk_p[kind]
        qpos[:, 3] = 1
        qpos[:, dyn.qidx] = self.q[kind]
        qvel[:, 0:3] = self.trunk_v[kind]
        qvel[:, dyn.didx] = self.qd[kind]
        qacc[:, 0:3] = self.trunk_a[kind]
        qacc[:, dyn.didx] = self.qdd[kind]
        return qpos, qvel, qacc

    def _set_slot_masses(self, kg_by_slot):
        m = self.dyn.model
        for name, bid in self.dyn.slot.items():
            kg = float(kg_by_slot[name])
            m.body_mass[bid] = kg
            m.body_inertia[bid] = kg * self.dyn.box_per_kg[name]

    def _inverse(self, kind, jacobians=False):
        """h (n, nv) of every sample of a kind at the current slot masses; with jacobians also the 4 feet's
        translational Jacobians (n, 4, 3, nv)."""
        m, d = self.dyn.model, self.dyn.data
        qpos, qvel, qacc = self._states(kind)
        n = self.n[kind]
        H = np.empty((n, m.nv))
        J4 = np.zeros((n, len(S.LEGS), 3, m.nv)) if jacobians else None
        jac = np.zeros((3, m.nv))
        for i in range(n):
            d.qpos[:] = qpos[i]
            d.qvel[:] = qvel[i]
            d.qacc[:] = qacc[i]
            mujoco.mj_inverse(m, d)
            H[i] = d.qfrc_inverse
            if jacobians:
                for k, leg in enumerate(S.LEGS):
                    mujoco.mj_jacSite(m, d, jac, None, self.dyn.site[leg])
                    J4[i, k] = jac
        return H, J4

    def _contact(self, kind, H):
        """tau (n, 12) in JOINTS order and fz (n, 4) (NaN off stance) of generalized forces H by the contact rule."""
        hb = H[:, :6]
        tau_full = H - np.einsum("nij,nj->ni", self._M[kind], hb)
        fz = np.einsum("nlj,nj->nl", self._P[kind][:, :, 2, :], hb)
        return tau_full[:, self.dyn.didx], fz

    def _forces(self, kind, H):
        """f (n, 4, 3) at the stance feet (NaN off stance)."""
        return np.einsum("nlcj,nj->nlc", self._P[kind], H[:, :6])

    def _unit_masses(self, scale_slots=()):
        out = {name: 1 for name in self.dyn.slot}
        for name in scale_slots:
            out[name] = 2
        return out

    def _build_contact_maps_and_regressor(self):
        nv = self.dyn.model.nv
        self._M, self._P, self._J4 = {}, {}, {}
        base = {}
        try:
            self._set_slot_masses(self._unit_masses())
            for kind in KINDS:
                H, J4 = self._inverse(kind, jacobians=True)
                n = self.n[kind]
                M = np.zeros((n, nv, 6))
                P = np.full((n, len(S.LEGS), 3, 6), np.nan)
                st = self.stance[kind]
                for pattern in sorted({tuple(row) for row in st.tolist()}):
                    idx = np.flatnonzero((st == np.array(pattern)).all(axis=1))
                    legs = [k for k, on in enumerate(pattern) if on]
                    JS = J4[idx][:, legs].reshape(len(idx), 3 * len(legs), nv)
                    Pg = np.linalg.pinv(np.swapaxes(JS[:, :, :6], 1, 2), rcond=1e-10)
                    M[idx] = np.matmul(np.swapaxes(JS, 1, 2), Pg)
                    P[idx[:, None], np.array(legs)[None, :]] = Pg.reshape(len(idx), len(legs), 3, 6)
                self._M[kind], self._P[kind], self._J4[kind] = M, P, J4
                base[kind] = self._contact(kind, H)
            cols = {kind: {} for kind in KINDS}
            for t in TERMS:
                self._set_slot_masses(self._unit_masses(Q.TERM_SLOTS[t]))
                for kind in KINDS:
                    H, _ = self._inverse(kind)
                    tau, fz = self._contact(kind, H)
                    cols[kind][t] = (tau - base[kind][0], fz - base[kind][1])
        finally:
            self._set_slot_masses(self._unit_masses())
        self.tau0, self.fz0, self.tau_b, self.fz_b = {}, {}, {}, {}
        for kind in KINDS:
            self.tau_b[kind] = {t: cols[kind][t][0] for t in TERMS}
            self.fz_b[kind] = {t: cols[kind][t][1] for t in TERMS}
            self.tau0[kind] = base[kind][0] - sum(self.tau_b[kind][t] for t in TERMS)
            self.fz0[kind] = base[kind][1] - sum(self.fz_b[kind][t] for t in TERMS)
        self._finish_regressor()

    def _finish_regressor(self):
        """Stand-up + crawl ("move") concatenations of the regressor."""
        mv = ("standup", "crawl")
        self._tau0_mv = np.concatenate([self.tau0[k] for k in mv])
        self._B_mv = {t: np.concatenate([self.tau_b[k][t] for k in mv]) for t in TERMS}

    def replace_pod(self, pod):
        """The same duty with another pod (quad-model/1 "pod"): the pod column is re-run in one inverse pass."""
        new = copy.copy(self)
        new.model = copy.deepcopy(self.model)
        new.model["pod"] = copy.deepcopy(pod)
        new.dyn = Q.build(new.model, self._duty["gravity_m_s2"])
        new.tau_b = {k: dict(v) for k, v in self.tau_b.items()}
        new.fz_b = {k: dict(v) for k, v in self.fz_b.items()}
        try:
            new._set_slot_masses(new._unit_masses())
            for kind in KINDS:
                H, _ = new._inverse(kind)
                tau, fz = new._contact(kind, H)
                others = [t for t in TERMS if t != "pod"]
                new.tau_b[kind]["pod"] = tau - self.tau0[kind] - sum(self.tau_b[kind][t] for t in others)
                new.fz_b[kind]["pod"] = fz - self.fz0[kind] - sum(self.fz_b[kind][t] for t in others)
        finally:
            new._set_slot_masses(new._unit_masses())
        new._finish_regressor()
        return new

    # ------------------------------------------------------------------ loads for BOM masses
    def _factor(self, kind):
        return float(self._duty["proof_factor"]) if kind == "proof" else 1

    def loads(self, masses, payload_kg=None):
        """{"tau": {kind: (n, 12)}, "fz": {kind: (n, 4)}} for BOM masses {"hip_roll", "hip_pitch", "knee", "pack": kg}
        and the payload (None = the layer's payload_top_kg; the proof poses carry proof_factor x the payload)."""
        pay = float(self.layer["brief"]["payload_top_kg"] if payload_kg is None else payload_kg)
        out = {"tau": {}, "fz": {}}
        for kind in KINDS:
            c = self._factor(kind) * pay
            tau = self.tau0[kind] + c * self.tau_b[kind]["pod"]
            fz = self.fz0[kind] + c * self.fz_b[kind]["pod"]
            for t in MASS_TERMS:
                tau = tau + float(masses[t]) * self.tau_b[kind][t]
                fz = fz + float(masses[t]) * self.fz_b[kind][t]
            out["tau"][kind] = tau
            out["fz"][kind] = fz
        return out

    def _tau_move(self, masses, payload):
        tau = self._tau0_mv + payload * self._B_mv["pod"]
        for t in MASS_TERMS:
            tau = tau + float(masses[t]) * self._B_mv[t]
        return tau

    def _tau_proof(self, masses, payload):
        tau = self.tau0["proof"] + self._factor("proof") * payload * self.tau_b["proof"]["pod"]
        for t in MASS_TERMS:
            tau = tau + float(masses[t]) * self.tau_b["proof"][t]
        return tau

    def _fz_all(self, masses, payload):
        out = []
        for kind in KINDS:
            fz = self.fz0[kind] + self._factor(kind) * payload * self.fz_b[kind]["pod"]
            for t in MASS_TERMS:
                fz = fz + float(masses[t]) * self.fz_b[kind][t]
            out.append(fz)
        return out

    def masses_of(self, bom, cat):
        """The four BOM masses {"hip_roll", "hip_pitch", "knee", "pack": kg} from the catalogue (ProcedureError for a
        BOM part that is not in the catalogue or not of its line's role)."""
        check_bom_parts(bom, cat, ("modules", "battery"))
        out = {t: float(cat.get(bom["modules"][t])["mass_kg"]) for t in S.JOINT_TYPES}
        out["pack"] = float(cat.get(bom["battery"])["mass_kg"])
        return out

    # ------------------------------------------------------------------ pass rules (B-14 h-l)
    def _type_util(self, ctx, joint_type, pid, family, tau_mv, tau_pf):
        """Per joint of the type (4 entries): move, proof, rms, worst, binding code (BINDING_KINDS index)."""
        cap = ctx.cap(pid, family)
        cols = self.cols[joint_type]
        J = cap["J_out"]
        tau = tau_mv[:, cols]
        qdd = self._qdd_mv[:, cols]
        ratio = _div(np.abs(tau) + J * np.abs(qdd), ctx.env(pid, family, joint_type))
        kbin = np.argmax(ratio, axis=0)
        ar = np.arange(len(cols))
        u_move = ratio[kbin, ar]
        u_proof = _div(np.max(np.abs(tau_pf[:, cols]), axis=0), cap["tau_s"])
        motor = tau[self._crawl] + J * qdd[self._crawl]
        u_rms = _div(np.sqrt(np.mean(motor * motor, axis=0)), cap["tau_c"])
        worst = np.maximum(np.maximum(u_move, u_proof), u_rms)
        in_stance = self._stance_mv[kbin, np.arange(len(S.LEGS))]
        code = np.where(u_move >= worst, np.where(in_stance, 0, 1), np.where(u_proof >= worst, 2, 3))
        return u_move, u_proof, u_rms, worst, code

    def _phase_currents(self, ctx, modules, family, tau_mv):
        """Phase currents (n_move, 12) of the weakest expected units (B-14 j)."""
        out = np.empty_like(tau_mv)
        for t in S.JOINT_TYPES:
            cap = ctx.cap(modules[t], family)
            cols = self.cols[t]
            out[:, cols] = np.abs(tau_mv[:, cols] + cap["J_out"] * self._qdd_mv[:, cols]) * _div(cap["i_pk"],
                                                                                                 cap["tau_s"])
        return out

    def _bus_current(self, ctx, modules, family, tau_mv, i_ph):
        e = self.layer["brief"]
        v_s = L.module_voltage(self.layer, family)
        r = np.empty(len(S.JOINTS))
        for t in S.JOINT_TYPES:
            r[self.cols[t]] = ctx.cap(modules[t], family)["R_ph"]
        mech = np.maximum(0, tau_mv * self._qd_mv)
        power = np.sum(mech + float(e["copper_loss_factor"]) * r * i_ph * i_ph, axis=1)
        return float(e["tray_current_a"]) + power / (float(e["driver_efficiency"]) * v_s)

    def _charge_ah(self, ctx, i_bus):
        return float(np.mean(i_bus[self._crawl])) * ctx.mission / S.UNIT_TO_SI["h"][1]

    def _evaluate(self, ctx, bom):
        cat = ctx.cat
        check_bom_parts(bom, cat, BOM_LINES)
        fam = bom["battery_family"]
        mods = bom["modules"]
        caps = {t: ctx.cap(mods[t], fam) for t in S.JOINT_TYPES}
        masses = self.masses_of(bom, cat)
        tau_mv = self._tau_move(masses, ctx.payload)
        tau_pf = self._tau_proof(masses, ctx.payload)
        pack_id = bom["battery"]
        pack = cat.get(pack_id)
        window = cat.pack_window(pack_id)
        drv = cat.get(bom["driver"])
        hub = cat.get(bom["hub"])
        items, util = {}, {}
        for t in S.JOINT_TYPES:
            m = cat.get(mods[t])
            u_move, u_proof, u_rms, worst, code = self._type_util(ctx, t, mods[t], fam, tau_mv, tau_pf)
            fit = m["housing_class"] == self.model["housing_class"][t]
            vok = cat.window_inside(window, m["voltage_range_v"])
            for k, leg in enumerate(S.LEGS):
                j = leg + "_" + t
                util[j] = {"move": float(u_move[k]), "proof": float(u_proof[k]), "rms": float(u_rms[k]),
                           "worst": float(worst[k]), "kind": BINDING_KINDS[int(code[k])], "fit": bool(fit),
                           "voltage_ok": bool(vok)}
                items[j] = bool(fit and vok and worst[k] <= 1)
        i_ph = self._phase_currents(ctx, mods, fam, tau_mv)
        i_bus = self._bus_current(ctx, mods, fam, tau_mv, i_ph)
        charge = self._charge_ah(ctx, i_bus)
        phase_peak = np.max(i_ph, axis=0)
        phase_rms = np.sqrt(np.mean(i_ph[self._crawl] ** 2, axis=0))
        # battery (B-14 i)
        family_ok = int(pack["cells"]) == L.family_cells(fam)
        b_window = all(cat.window_inside(window, cat.get(mods[t])["voltage_range_v"]) for t in S.JOINT_TYPES)
        b_window = bool(b_window and cat.window_inside(window, drv["input_range_v"]))
        fit_ok = cat.fits(pack_id, self.model["pack_bay"]["inner_m"])
        u_cur = float(_div(np.max(i_bus), pack["continuous_current_a"]))
        usable = float(self.layer["brief"]["usable_fraction"]) * float(pack["capacity_ah"]) * ctx.pack_factor(pack_id)
        u_chg = float(_div(charge, usable))
        util["battery"] = {"charge": u_chg, "current": u_cur, "worst": max(u_chg, u_cur), "family_ok": family_ok,
                           "window_ok": b_window, "fit_ok": fit_ok}
        items["battery"] = bool(family_ok and b_window and fit_ok and u_chg <= 1 and u_cur <= 1)
        # drivers (B-14 k)
        count_ok = int(bom["driver_count"]) == cat.driver_count(bom["driver"])
        d_window = cat.window_inside(window, drv["input_range_v"])
        u_pk = float(_div(np.max(phase_peak), drv["peak_current_a"]))
        u_rms = float(_div(np.max(phase_rms), drv["continuous_current_a"]))
        util["drivers"] = {"peak": u_pk, "rms": u_rms, "worst": max(u_pk, u_rms), "count_ok": bool(count_ok),
                           "window_ok": bool(d_window)}
        items["drivers"] = bool(count_ok and d_window and u_pk <= 1 and u_rms <= 1)
        # hub and budget (B-14 l)
        ports_ok = int(hub["ports"]) >= int(bom["driver_count"])
        util["hub"] = {"worst": float(_div(int(bom["driver_count"]), int(hub["ports"]))), "ports_ok": bool(ports_ok)}
        items["hub"] = bool(ports_ok)
        cost = cat.bom_cost(bom)
        util["budget"] = {"worst": float(_div(cost, ctx.ceiling))}
        items["budget"] = bool(cost <= ctx.ceiling + 1e-9)
        fz = self._fz_all(masses, ctx.payload)
        trace = {"min_fz_n": float(min(np.nanmin(f) for f in fz)), "max_fz_n": float(max(np.nanmax(f) for f in fz)),
                 "stop_margin_rad": self.stop_margin_rad}
        currents = {"bus_peak_a": float(np.max(i_bus)), "bus_mean_crawl_a": float(np.mean(i_bus[self._crawl])),
                    "charge_ah": charge,
                    "phase_peak_a": {j: float(phase_peak[i]) for i, j in enumerate(S.JOINTS)},
                    "phase_rms_a": {j: float(phase_rms[i]) for i, j in enumerate(S.JOINTS)}}
        items = {i: items[i] for i in S.ITEMS}
        return {"items": items, "util": util, "caps": {t: dict(caps[t]) for t in S.JOINT_TYPES},
                "currents": currents, "trace": trace, "cost": cost, "v_s": L.module_voltage(self.layer, fam),
                "pass_fraction": sum(1 for v in items.values() if v) / len(S.ITEMS)}

    def evaluate(self, bom, cat, lots, *, payload_kg=None, mission_time_s=None, ceiling_usd=None):
        """The 16 items of B-14 h-l for an internal BOM with the delivered lots' weakest expected units."""
        ctx = EvalContext(self, cat, lots, payload_kg, mission_time_s, ceiling_usd)
        return self._evaluate(ctx, bom)

    # ------------------------------------------------------------------ derived quantities (B-16)
    def loaded_com(self, masses, payload_kg):
        """The whole system's CoM (3,) in the trunk frame at the B-05 stance with the BOM masses and the pod."""
        h_s = float(self._duty["stance_height_m"])
        trunk = np.array([0, 0, h_s], float)
        pts = Q.intended_points(self.model, trunk, self.stance_q)
        tot, mom = 0, np.zeros(3)
        for r in self.model["rigid_inertials"]:
            tot += float(r["mass_kg"])
            mom = mom + float(r["mass_kg"]) * pts["rig_" + r["name"]]
        for t in S.JOINT_TYPES:
            for name in Q.TERM_SLOTS[t]:
                tot += float(masses[t])
                mom = mom + float(masses[t]) * pts[name]
        tot += float(masses["pack"]) + float(payload_kg)
        mom = mom + float(masses["pack"]) * pts["slot_pack"] + float(payload_kg) * pts["slot_pod"]
        return mom / tot - trunk

    def derived(self, bom, cat, lots, *, payload_kg=None, mission_time_s=None):
        """{quantity: SI value} of the 16 DERIVED_QUANTITIES for an internal BOM (worst leg of the type; the delivered
        lot's weakest expected units)."""
        ctx = EvalContext(self, cat, lots, payload_kg, mission_time_s, None)
        ev = self._evaluate(ctx, bom)
        masses = self.masses_of(bom, cat)
        tau_mv = self._tau_move(masses, ctx.payload)
        tau_pf = self._tau_proof(masses, ctx.payload)
        out = {}
        for t in S.JOINT_TYPES:
            cols = self.cols[t]
            out["proof_peak_torque:" + t] = float(np.max(np.abs(tau_pf[:, cols])))
            cr = tau_mv[self._crawl][:, cols]
            out["duty_rms_torque:" + t] = float(np.max(np.sqrt(np.mean(cr * cr, axis=0))))
        out["standup_peak_torque:knee"] = float(np.max(np.abs(self._rows(tau_mv, "standup")[:, self.cols["knee"]])))
        for t in ("hip_pitch", "knee"):
            out["joint_peak_speed:" + t] = float(np.max(np.abs(self._qd_mv[:, self.cols[t]])))
        cur = ev["currents"]
        out["supply_peak_current"] = cur["bus_peak_a"]
        out["mission_charge"] = cur["charge_ah"]
        knees = [leg + "_knee" for leg in S.LEGS]
        out["drive_peak_current:knee"] = max(cur["phase_peak_a"][j] for j in knees)
        out["drive_rms_current:knee"] = max(cur["phase_rms_a"][j] for j in knees)
        out["robot_mass"] = (sum(float(r["mass_kg"]) for r in self.model["rigid_inertials"])
                             + len(S.LEGS) * sum(masses[t] for t in S.JOINT_TYPES) + masses["pack"])
        out["loaded_com_x"] = float(self.loaded_com(masses, ctx.payload)[0])
        out["peak_normal_force"] = float(max(np.nanmax(f) for f in self._fz_all(masses, ctx.payload)))
        return {q: out[q] for q in S.DERIVED_QUANTITIES}

    def _rows(self, tau_mv, kind):
        """The rows of one kind ("standup" or "crawl") of a stand-up + crawl load array."""
        return tau_mv[:self.n["standup"]] if kind == "standup" else tau_mv[self._crawl]

    # ------------------------------------------------------------------ interval bounds
    def _centre_radius(self, kind, box, payload, fixed=None, field="tau"):
        """Centre and radius over a mass box of a kind's regressor (tau or fz); `fixed` = {term: kg} replaces the box."""
        base = self.tau0 if field == "tau" else self.fz0
        cols = self.tau_b if field == "tau" else self.fz_b
        c = base[kind] + self._factor(kind) * payload * cols[kind]["pod"]
        r = np.zeros_like(c)
        for t in MASS_TERMS:
            if fixed is not None and t in fixed:
                c = c + float(fixed[t]) * cols[kind][t]
                continue
            lo, hi = (float(v) for v in box[t])
            c = c + (lo + hi) / 2 * cols[kind][t]
            r = r + (hi - lo) / 2 * np.abs(cols[kind][t])
        return c, r

    def normal_force_lower_bound(self, box, payload_kg=None):
        """Min over all samples and stance feet of fz_c - fz_r: every normal force over the mass box is >= this."""
        pay = float(self.layer["brief"]["payload_top_kg"] if payload_kg is None else payload_kg)
        out = []
        for kind in KINDS:
            c, r = self._centre_radius(kind, box, pay, field="fz")
            out.append(float(np.nanmin(c - r)))
        return min(out)

    def _lb_parts(self, joint_type, box, payload):
        """Interval parts of a type's 4 joints over the box: the centre with the type's own mass at the box middle, the
        radius of the other masses, the type's own regressor columns and the box middle."""
        lo, hi = (float(v) for v in box[joint_type])
        mid = (lo + hi) / 2
        fixed = {joint_type: mid}
        cols = self.cols[joint_type]
        cs, rs = self._centre_radius("standup", box, payload, fixed)
        cc, rc = self._centre_radius("crawl", box, payload, fixed)
        cp, rp = self._centre_radius("proof", box, payload, fixed)
        return (np.concatenate([cs, cc])[:, cols], np.concatenate([rs, rc])[:, cols], cp[:, cols], rp[:, cols],
                self._B_mv[joint_type][:, cols], self.tau_b["proof"][joint_type][:, cols], mid)

    def _util_lb_arrays(self, joint_type, box, payload, mass_kg, parts=None):
        """Centre and radius of the move and proof loads of a type's 4 joints with its own mass fixed at mass_kg."""
        c_mv, r_mv, c_pf, r_pf, b_mv, b_pf, mid = parts if parts is not None else self._lb_parts(joint_type, box,
                                                                                                    payload)
        d = float(mass_kg) - mid
        return c_mv + d * b_mv, r_mv, c_pf + d * b_pf, r_pf

    def _util_lb(self, ctx, joint_type, pid, family, arrays):
        cap = ctx.cap(pid, family)
        c_mv, r_mv, c_pf, r_pf = arrays
        J = cap["J_out"]
        qdd = self._qdd_mv[:, self.cols[joint_type]]
        need = np.maximum(0, np.abs(c_mv) - r_mv) + J * np.abs(qdd)
        u_move = np.max(_div(need, ctx.env(pid, family, joint_type)), axis=0)
        u_proof = _div(np.max(np.maximum(0, np.abs(c_pf) - r_pf), axis=0), cap["tau_s"])
        lb = np.maximum(0, np.abs(c_mv[self._crawl] + J * qdd[self._crawl]) - r_mv[self._crawl])
        u_rms = _div(np.sqrt(np.mean(lb * lb, axis=0)), cap["tau_c"])
        return float(np.max(np.maximum(np.maximum(u_move, u_proof), u_rms)))

    def util_lower_bound(self, joint_type, pid, family, box, cat, lots, payload_kg=None):
        """A lower bound of the worst utilisation of SKU `pid` at the type's 4 joints for every other BOM mass in
        `box` (the SKU's own mass in place of the box for its type)."""
        ctx = EvalContext(self, cat, lots, payload_kg)
        arrays = self._util_lb_arrays(joint_type, box, ctx.payload, float(cat.get(pid)["mass_kg"]))
        return self._util_lb(ctx, joint_type, pid, family, arrays)

    # ------------------------------------------------------------------ self-tests (they return measured errors)
    def selftest_xipos(self):
        """Max distance (m) of the compiled CoM of every added body from its intended point at the B-05 stance and at
        the four proof poses."""
        h_s = float(self._duty["stance_height_m"])
        errs = [self.dyn.xipos_error(np.array([0, 0, h_s], float), self.stance_q)]
        for i in range(self.n["proof"]):
            errs.append(self.dyn.xipos_error(self.trunk_p["proof"][i], self.q["proof"][i]))
        return max(errs)

    def selftest_inertia(self):
        return self.dyn.inertia_error()

    def _direct(self, masses, payload):
        """tau and fz of every kind by direct inverse passes with the slot masses set (the pod at the payload, x the
        proof factor in the proof poses)."""
        out = {}
        try:
            for kind in KINDS:
                kg = {}
                for t in MASS_TERMS:
                    for name in Q.TERM_SLOTS[t]:
                        kg[name] = float(masses[t])
                kg["slot_pod"] = self._factor(kind) * payload
                self._set_slot_masses(kg)
                H, _ = self._inverse(kind)
                out[kind] = (H, ) + self._contact(kind, H)
        finally:
            self._set_slot_masses(self._unit_masses())
        return out

    def selftest_regressor(self, rng, n_draws, box, payload_kg=None):
        """Max relative error of the regressor against direct inverse passes for mass vectors drawn uniformly in `box`
        (max abs difference / max(1, max abs direct), over tau and fz of every kind)."""
        pay = float(self.layer["brief"]["payload_top_kg"] if payload_kg is None else payload_kg)
        worst = 0
        for _ in range(int(n_draws)):
            masses = {t: float(rng.uniform(float(box[t][0]), float(box[t][1]))) for t in MASS_TERMS}
            reg = self.loads(masses, pay)
            direct = self._direct(masses, pay)
            for kind in KINDS:
                for key, idx in (("tau", 1), ("fz", 2)):
                    a = reg[key][kind]
                    b = direct[kind][idx]
                    err = float(np.nanmax(np.abs(a - b))) / max(1, float(np.nanmax(np.abs(b))))
                    worst = max(worst, err)
        return worst

    def selftest_contact(self, masses, payload_kg=None):
        """{"base_residual": max abs(h_b - J_Sb^T f), "centroid_form": max abs(f_B - f)} over every sample at `masses`
        (f_B: the minimum-norm solution of the wrench balance about the stance feet's centroid)."""
        pay = float(self.layer["brief"]["payload_top_kg"] if payload_kg is None else payload_kg)
        direct = self._direct(masses, pay)
        resid, cform = 0, 0
        for kind in KINDS:
            H = direct[kind][0]
            f = self._forces(kind, H)
            fz0 = np.nan_to_num(f)
            J4 = self._J4[kind]
            jbf = np.einsum("nlcj,nlc->nj", J4[:, :, :, :6], fz0)
            resid = max(resid, float(np.max(np.abs(H[:, :6] - jbf))))
            st = self.stance[kind]
            for pattern in sorted({tuple(row) for row in st.tolist()}):
                idx = np.flatnonzero((st == np.array(pattern)).all(axis=1))
                legs = [k for k, on in enumerate(pattern) if on]
                p = self.feet_p[kind][idx][:, legs]                         # (g, s, 3)
                c = p.mean(axis=1)
                A = np.zeros((len(idx), 6, 3 * len(legs)))
                for a, k in enumerate(legs):
                    A[:, 0:3, 3 * a:3 * a + 3] = np.eye(3)
                    r = p[:, a] - c
                    A[:, 3:6, 3 * a:3 * a + 3] = np.stack([np.stack([np.zeros(len(idx)), -r[:, 2], r[:, 1]], 1),
                                                           np.stack([r[:, 2], np.zeros(len(idx)), -r[:, 0]], 1),
                                                           np.stack([-r[:, 1], r[:, 0], np.zeros(len(idx))], 1)], 1)
                h = H[idx]
                w = np.concatenate([h[:, 0:3], h[:, 3:6] - np.cross(c - self.trunk_p[kind][idx], h[:, 0:3])], axis=1)
                fb = np.einsum("gij,gj->gi", np.linalg.pinv(A, rcond=1e-10), w)
                fm = f[idx][:, legs].reshape(len(idx), -1)
                cform = max(cform, float(np.max(np.abs(fb - fm))))
        return {"base_residual": resid, "centroid_form": cform}

    def selftest_derivatives(self, h_s):
        """Max scaled error abs(a - b) / max(1, abs(a), abs(b)) of the analytic qd against central differences of q,
        and of the analytic qdd against central differences of the analytic qd, at every stand-up and crawl sample
        with the time step h_s (s)."""
        worst = 0
        h = float(h_s)
        for kind in ("standup", "crawl"):
            seg_idx = self.segment[kind]
            for si in sorted(set(seg_idx.tolist())):
                seg = self.segments[si]
                rows = np.flatnonzero(seg_idx == si)
                u = self._u[kind][rows]
                du = h / float(seg["T_s"])
                qp, qdp, _ = self._joint_states(*self._segment_state(seg, u + du))
                qm, qdm, _ = self._joint_states(*self._segment_state(seg, u - du))
                pairs = ((self.qd[kind][rows], (qp - qm) / (2 * h)), (self.qdd[kind][rows], (qdp - qdm) / (2 * h)))
                for a, b in pairs:
                    err = np.abs(a - b) / np.maximum(1, np.maximum(np.abs(a), np.abs(b)))
                    worst = max(worst, float(np.max(err)))
        return worst

    def duty_fingerprint(self, every):
        """The quad-duty-set/1 content: sample counts, stance and swing digests, probes every `every` samples."""
        every = int(every)
        probes = []
        for kind in KINDS:
            n = self.n[kind]
            idx = list(range(0, n, every))
            if idx[-1] != n - 1:
                idx.append(n - 1)
            for i in idx:
                probes.append({"kind": kind, "index": i, "trunk_p": [float(v) for v in self.trunk_p[kind][i]],
                               "q": [float(v) for v in self.q[kind][i]], "qd": [float(v) for v in self.qd[kind][i]],
                               "qdd": [float(v) for v in self.qdd[kind][i]]})
        return {"format": "quad-duty-set/1", "counts": {k: int(self.n[k]) for k in KINDS}, "every": every,
                "stance_sha256": _digest({k: self.stance[k].astype(int).tolist() for k in KINDS}),
                "swing_sha256": _digest({k: self.swing_leg[k].astype(int).tolist() for k in KINDS}),
                "probes": probes}
