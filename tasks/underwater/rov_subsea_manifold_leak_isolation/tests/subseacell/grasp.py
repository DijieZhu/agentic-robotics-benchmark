"""Grasp rules of task 07 on the reference vehicle's hinged V jaws (cell side; DESIGN v1.2 section 1.4 "grasp rule";
decisions K1, L14, M11, M13; package P05).  No friction grip: a grasp is a constraint switched on by a rule.
Product version of `proto/g1_valve_turn/grasp.py` (weld, hinge lock, seating glide, slip) with the stall detector of
`proto/g2_grasp_approach/scene.py::G2PaddleGrasp` and the lifting-eye rule of `proto/p2_debris/site_world.py`.

Paddle handles (valve handles, the marker's handle, the hot-stab handle: 160 x 60 x 12 mm blades; graspable when the
blade is horizontal).  Paddle frame P: origin = blade centre, x = along the stem away from the vehicle (into the
panel), y = blade length, z = blade normal.  gripper_close() completes when the jaws stall (jaw angle changed by less
than stall_span_rad over the last stall_window_s, armed stall_arm_s after the command; M11) or close fully.  On a stall
with both jaws on one paddle, the rule is evaluated on the TCP pose relative to P:
  (a) jaw mid-plane on the blade mid-plane within centre_tol (the coupled jaws, 124 N each at the TCP, push the
      floating vehicle onto the blade, so this holds automatically at completion; the real vertical window is the
      entry of the open V: |dz| <= 24.3 - 0.44 x bite mm, a property of the jaw geometry - see jaw_gap_profile());
  (b) TCP within lateral_m of the blade centre along the blade length;
  (c) bite (jaw tip past the blade's outer edge) >= bite_min_m; the upper limit is the jaws' throat (no number in the
      rule: the shell collision stops the blade);
  (d) each component of the relative rotation vector <= angle_deg.
All four: weld at the current relative pose, hinge locked (valves), jaw servo frozen, the handle's geoms stop
colliding with the vehicle, and the weld target glides (quintic smootherstep) to the canonical pose (bite canon_bite_m,
centred, aligned) in seat_s = 0.5 s; the VEHICLE moves, the handle does not; the lock is released at the end of the
glide.  Otherwise (window miss) the jaws close through the blade -> closed_empty.
Slip: armed when the glide ends; the weld's load on the handle less the axial force and the torque about the tool axis
- lateral force > slip_lateral_N or bending moment (about the jaw-tip point) > slip_bending_Nm continuously for
slip_window_s -> the grasp is released, an event is logged, no penalty.  The torque about the tool axis is not
limited (lead-screw drive, not back-drivable).  gripper_open(): the weld lets go once the jaws opened release_open_m; a free handle (marker, stab) ignores the vehicle
until its blade is pass_clear_m from the TCP, so it falls out of the open V instead of resting on the lower jaw.

Lifting eyes (swivel eyes, inner diameter 80 mm; not collision geometry): when gripper_close() completes without a
paddle grasp and the TCP is within eye_capture_m (35 mm) of an eye centre, a ball joint (MuJoCo connect) is switched
on at the eye centre (anchored where it is: no snap) and the gripper reads stalled_on_object; every activation logs one
grasp event (scoring item D1 needs it).  gripper_open() releases it.

    from subseacell import grasp as GR
    GR.add_constraints(vb.spec, handles, eyes)        # before compile; handles from devices (valve / marker infos)
    G = GR.Gripper(sim, handles, eyes)                # after the simulator is built
    every physics step:  G.tick_pre(); sim.step(); G.tick_post()
    G.close(); G.open(); G.state; G.held; G.events; G.weld_load(); G.rel_pose(name)

Loads come from the weld rows only (J^T f on the hull's free joint -> world force at the hull origin + body-frame
torque -> moved to the jaw-tip point on the tool axis), independent of MuJoCo's weld row conventions.
"""
from __future__ import annotations

import collections
import copy
import math

import numpy as np
import mujoco

from uwkit.frames import quat_mul, quat_conj

# task values grasp.* (DEV/spec/task_values.json; tests/test_devices.py asserts the equality); configure() overrides
DEFAULTS = dict(
    seat_s=0.5, canon_bite_m=0.040, lateral_m=0.060, bite_min_m=0.020, angle_deg=15.0, centre_tol_m=0.004,
    slip_lateral_N=90.0, slip_bending_Nm=12.0, slip_window_s=0.2, release_open_m=0.002,
    stall_window_s=0.1, stall_span_rad=0.002, stall_arm_s=0.2, closed_rad=0.02, open_tol_rad=0.02,
    stall_rule="span",       # "span" (M11: jaw angle change over the window) | "rate" (gate G1's rule, for comparison only:
    stall_rate_rad_s=0.02,   #  |jaw rate| < stall_rate_rad_s continuously for stall_window_s)
    eye_capture_m=0.035, pass_clear_m=0.14,
    weld_solref=(0.004, 1.0), weld_solimp=(0.95, 0.99, 0.001, 0.5, 2.0), lock_solref=(0.004, 1.0),
    connect_solref=(0.004, 1.0), connect_solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
)
VALUES = copy.deepcopy(DEFAULTS)
GRASPED_BITS = (4, 1)        # = uwkit GRASPED_CONTYPE / CONAFFINITY: no contact with the vehicle, contact with the world
PUBLIC_STATES = ("open", "closing", "opening", "closed_empty", "stalled_on_object")


def configure(values: dict):
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown grasp values {sorted(unknown)}")
    VALUES.update(copy.deepcopy(values))


# ---------------------------------------------------------------------------------------------------- math helpers
def smootherstep(u):
    """Quintic: zero velocity and acceleration at both ends."""
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def _slerp(q0, q1, u):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if float(q0 @ q1) < 0.0:
        q1 = -q1
    dd = float(np.clip(q0 @ q1, -1.0, 1.0))
    if dd > 0.9995:
        q = q0 + u * (q1 - q0)
    else:
        th = math.acos(dd)
        q = (math.sin((1.0 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return v * th


def _mat_quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(np.asarray(R, float)).reshape(-1))
    return q


# ---------------------------------------------------------------------------------------------------- jaw geometry
def jaw_angle_for(model, data, theta):
    """Set both jaws (coupled) to the upper jaw angle theta [rad] in data.qpos."""
    ju, jl = model.joint("jaw_upper").id, model.joint("jaw_lower").id
    eqc = model.equality("gripper_coupling").id
    data.qpos[model.jnt_qposadr[ju]] = theta
    data.qpos[model.jnt_qposadr[jl]] = model.eq_data[eqc, 0] + model.eq_data[eqc, 1] * theta


def jaw_gap_profile(model, s_mm=np.arange(0.0, 71.0, 1.0), theta=None, lateral_mm=(-6.0, 0.0, 6.0)):
    """Vertical half-gaps (up, down) [mm] of the jaws (default fully open) along the tool axis, s = distance behind the
    jaw tip, by ray casts from the jaw mid-plane against the jaw collision geoms (pads + shells), on the jaw centre
    line and +-6 mm across.  Returns dict(s_mm, up_mm, down_mm, half_gap_mm = min over the lines, tip_opening_mm)."""
    from uwkit.dynamics.hydro import read_uwkit_meta
    d = mujoco.MjData(model)
    g = read_uwkit_meta(model)["gripper"]
    jaw_angle_for(model, d, float(g["theta_open_rad"]) if theta is None else float(theta))
    mujoco.mj_forward(model, d)
    sid = model.site("gripper_tcp").id
    p0 = d.site_xpos[sid].copy()
    R = d.site_xmat[sid].reshape(3, 3).copy()
    tip = float(g["pad"]["x_to"])
    grp = np.array([0, 0, 0, 1, 0, 0], np.uint8)
    jaws = {model.body("jaw_upper").id, model.body("jaw_lower").id}
    gid = np.zeros(1, np.int32)
    up, dn = [], []
    for s in s_mm:
        us, ds = [], []
        for yo in lateral_mm:
            p = p0 + R[:, 0] * (tip - s * 1e-3) + R[:, 1] * (yo * 1e-3)
            a = mujoco.mj_ray(model, d, p, R[:, 2], grp, 1, -1, gid)
            us.append(a * 1e3 if (a >= 0 and model.geom_bodyid[gid[0]] in jaws) else np.inf)
            b = mujoco.mj_ray(model, d, p, -R[:, 2], grp, 1, -1, gid)
            ds.append(b * 1e3 if (b >= 0 and model.geom_bodyid[gid[0]] in jaws) else np.inf)
        up.append(min(us))
        dn.append(min(ds))
    up, dn = np.array(up), np.array(dn)
    return dict(s_mm=np.asarray(s_mm, float), up_mm=up, down_mm=dn, half_gap_mm=np.minimum(up, dn),
                tip_opening_mm=float(np.linalg.norm(d.site_xpos[model.site("jaw_upper_tip").id] -
                                                    d.site_xpos[model.site("jaw_lower_tip").id]) * 1e3))


def vertical_window_mm(profile, bite_mm, thickness_mm=12.0):
    """Largest |dz| at which a blade of the given thickness fits between the open jaws at this bite (the blade
    occupies s in [0, bite] inside the jaws; the V narrows with s).  Negative = the blade cannot enter that deep."""
    s, h = profile["s_mm"], profile["half_gap_mm"]
    sel = s <= bite_mm + 1e-9
    return float(np.min(h[sel]) - 0.5 * thickness_mm)


def entry_line_fit(profile, s_range=(0.0, 45.0), thickness_mm=12.0):
    """Linear fit of the half gap over the pad region: half_gap = a + b s  ->  window = (a - t/2) - |b| bite."""
    s, h = profile["s_mm"], profile["half_gap_mm"]
    sel = (s >= s_range[0]) & (s <= s_range[1]) & np.isfinite(h)
    b, a = np.polyfit(s[sel], h[sel], 1)
    return dict(half_gap_at_tip_mm=float(a), slope_per_mm=float(-b), window_a0_mm=float(a - 0.5 * thickness_mm))


def jaw_touch_angle(model, thickness, lo=0.0, hi=None, iters=50):
    """Upper-jaw angle [rad] at which the inner pad faces are `thickness` apart at the TCP (blade centred)."""
    from uwkit.dynamics.hydro import read_uwkit_meta
    g = read_uwkit_meta(model)["gripper"]
    hi = float(g["theta_open_rad"]) if hi is None else hi
    d = mujoco.MjData(model)
    sid = model.site("gripper_tcp").id
    pads = [model.geom("jaw_upper_pad").id, model.geom("jaw_lower_pad").id]

    def gap(th):
        jaw_angle_for(model, d, th)
        mujoco.mj_kinematics(model, d)
        p0, R = d.site_xpos[sid], d.site_xmat[sid].reshape(3, 3)
        out = []
        for gi, sg in zip(pads, (1.0, -1.0)):
            Rg = d.geom_xmat[gi].reshape(3, 3)
            n = Rg[:, 2] * (1.0 if float(Rg[:, 2] @ R[:, 2]) > 0 else -1.0)     # pad normal pointing away from the gap
            face = d.geom_xpos[gi] - sg * n * model.geom_size[gi][2]            # inner face centre
            # intersection of the TCP z line with the inner face plane
            t = float((face - p0) @ n) / float(R[:, 2] @ n)
            out.append(t)
        return out[0] - out[1]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if gap(mid) < thickness:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------------------------------- constraints
def add_constraints(spec: mujoco.MjSpec, handles: dict, eyes: dict = None, hull="hull"):
    """Inactive constraints, before compile: weld grip_<handle> (hull, handle body) and, for hinged handles, a joint
    lock lock_<handle>; one connect eye_<body> per body carrying eyes (anchor set at activation)."""
    V = VALUES
    for name, h in handles.items():
        e = spec.add_equality(name=f"grip_{name}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1=hull, name2=h["body"], active=False)
        e.solref, e.solimp = list(V["weld_solref"]), list(V["weld_solimp"])
        if h.get("joint"):
            k = spec.add_equality(name=f"lock_{name}", type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                                  name1=h["joint"], active=False)
            k.solref = list(V["lock_solref"])
    for body in sorted({e["body"] for e in (eyes or {}).values()}):
        e = spec.add_equality(name=f"eye_{body}", type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1=body, name2=hull, active=False)
        e.solref, e.solimp = list(V["connect_solref"]), list(V["connect_solimp"])


def eye(name, body, point_b, inner_diameter_m=0.080):
    """Eye description: centre `point_b` in the frame of `body`."""
    return dict(name=name, body=body, point_b=[float(x) for x in point_b], inner_diameter_m=float(inner_diameter_m))


# ---------------------------------------------------------------------------------------------------- the rule
class Gripper:
    """Run-time grasp rules for one vehicle.  Call tick_pre() before and tick_post() after every physics step."""

    def __init__(self, sim, handles: dict, eyes: dict = None, hull="hull", on_event=None, trace_close=False):
        self.sim, self.m, self.d = sim, sim.model, sim.data
        m = self.m
        self.V = VALUES
        g = sim.meta["gripper"]
        self.g = g
        self.hull = m.body(hull).id
        self.vadr = int(m.jnt_dofadr[m.body_jntadr[self.hull]])
        self.tcp_sid = m.site("gripper_tcp").id
        self.tcp_B = np.array(m.site_pos[self.tcp_sid], float)
        self.q_HT = np.array(m.site_quat[self.tcp_sid], float)
        self.axis_B = np.asarray(g["axis_B"], float)
        self.tip_ahead = float(g["pad"]["x_to"])
        self.ju = m.joint("jaw_upper").id
        self.ju_qa, self.ju_da = int(m.jnt_qposadr[self.ju]), int(m.jnt_dofadr[self.ju])
        self.jl_qa = int(m.jnt_qposadr[m.joint("jaw_lower").id])
        self.jaw_bodies = {m.body("jaw_upper").id, m.body("jaw_lower").id}
        self.theta_open = float(g["theta_open_rad"])
        self.lever_pad = float(g["lever_tcp_m"])
        self.handles = {}
        for n, h in handles.items():
            hh = dict(h)
            hh["bid"] = m.body(h["body"]).id
            hh["geom_ids"] = [m.geom(x).id for x in h["geoms"]]
            hh["paddle_gid"] = m.geom(h["paddle_geom"]).id
            hh["eq"] = m.equality(f"grip_{n}").id
            hh["lock"] = m.equality(f"lock_{n}").id if h.get("joint") else None
            hh["jqa"] = int(m.jnt_qposadr[m.joint(h["joint"]).id]) if h.get("joint") else None
            hh["c_b"] = np.asarray(h["centre_b"], float)
            hh["R_bp"] = np.asarray(h["R_bp"], float)
            Dp = float(h["paddle_m"][0])
            hh["depth"] = Dp
            tcp_P = np.array([-0.5 * Dp + self.V["canon_bite_m"] - self.tip_ahead, 0.0, 0.0])
            hh["A_b"] = hh["c_b"] + hh["R_bp"] @ tcp_P                       # canonical TCP point in the body frame
            hh["q_target"] = quat_mul(self.q_HT, quat_conj(_mat_quat(hh["R_bp"])))   # body w.r.t. hull at canon
            self.handles[n] = hh
        self.paddle_of_geom = {hh["paddle_gid"]: n for n, hh in self.handles.items()}
        self.eyes = {}
        for n, e in (eyes or {}).items():
            ee = dict(e)
            ee["bid"] = m.body(e["body"]).id
            ee["eq"] = m.equality(f"eye_{e['body']}").id
            ee["p_b"] = np.asarray(e["point_b"], float)
            self.eyes[n] = ee
        self.on_event = on_event
        self.trace_close = bool(trace_close)
        self._vec = None
        self.reset()

    # ------------------------------------------------------------------------------------------------ state
    def reset(self):
        self.state = "open"
        self.held = None                 # ("paddle", name) | ("eye", name)
        self.seat = None                 # (t0, A_h0, q0) while gliding
        self.locked = None
        self.t_close = None
        self.pass_through = {}           # handle -> saved (geom, contype, conaffinity)
        self.over_t = 0.0
        self.events = []
        self.last_rule = None
        self.load = dict(lateral=0.0, bending=0.0, axial=0.0, torsion=0.0, residual_mm=0.0)
        n = max(1, int(round(self.V["stall_window_s"] / float(self.m.opt.timestep))))
        self._qh = collections.deque(maxlen=n)
        self.seat_stats = None           # per grasp: angle change in the glide, residual and loads in / after it
        self.close_log = []              # per gripper_close(): t_cmd, t_contact, t_done, result
        self._cl = None
        self._lit_t = 0.0
        self._open_from = None

    def public_state(self):
        return self.state

    def _event(self, **e):
        e.setdefault("t", round(float(self.d.time), 4))
        self.events.append(e)
        if self.on_event is not None:
            self.on_event(e)
        return e

    # ------------------------------------------------------------------------------------------------ commands
    def close(self):
        if self.held:
            return False
        self.state = "closing"
        self.t_close = float(self.d.time)
        self._qh.clear()
        self._lit_t = 0.0
        self.sim.set_gripper(0.0)
        self._cl = dict(t_cmd=self.t_close, t_contact=None, t_done=None, result=None)
        return True

    def open(self):
        self.state = "opening"
        if self.held:
            self._open_from = float(self.d.qpos[self.ju_qa])
        self.sim.set_gripper(1.0)
        return True

    # ------------------------------------------------------------------------------------------------ geometry
    def paddle_frame_w(self, name):
        h = self.handles[name]
        b = h["bid"]
        Rb = self.d.xmat[b].reshape(3, 3)
        return self.d.xpos[b] + Rb @ h["c_b"], Rb @ h["R_bp"]

    def rel_pose(self, name):
        """TCP frame relative to the paddle frame: lateral (along the blade), vertical (blade centre along the TCP z),
        bite (jaw tip past the blade's outer edge), relative rotation vector components [deg]."""
        h = self.handles[name]
        pP, RP = self.paddle_frame_w(name)
        pT = self.d.site_xpos[self.tcp_sid]
        RT = self.d.site_xmat[self.tcp_sid].reshape(3, 3)
        R_PT = RP.T @ RT
        p = RP.T @ (pT - pP)
        tip_P = p + R_PT @ np.array([self.tip_ahead, 0.0, 0.0])
        return dict(lateral=float(p[1]), vertical=float(-(R_PT.T @ p)[2]), bite=float(tip_P[0] + 0.5 * h["depth"]),
                    angles_deg=np.degrees(_rotvec(R_PT)).tolist())

    def _touch(self):
        """{handle: set of jaw bodies touching its paddle}, and whether any jaw touches anything else."""
        d, m = self.d, self.m
        out, other = {}, False
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            for ga, bb in ((g1, b2), (g2, b1)):
                if bb in self.jaw_bodies:
                    if ga in self.paddle_of_geom:
                        out.setdefault(self.paddle_of_geom[ga], set()).add(bb)
                    elif m.geom_bodyid[ga] not in self.jaw_bodies:
                        other = True
        return out, other

    def _set_bits(self, name, on):
        m = self.m
        if on:
            if name in self.pass_through:
                return
            ids = self.handles[name]["geom_ids"]
            self.pass_through[name] = [(gi, int(m.geom_contype[gi]), int(m.geom_conaffinity[gi])) for gi in ids]
            for gi in ids:
                m.geom_contype[gi], m.geom_conaffinity[gi] = GRASPED_BITS
        else:
            for gi, ct, ca in self.pass_through.pop(name, []):
                m.geom_contype[gi], m.geom_conaffinity[gi] = ct, ca

    def _freeze_jaws(self, theta=None):
        sl = self.sim._slew["gripper"]
        th = float(self.d.qpos[self.ju_qa]) if theta is None else float(theta)
        sl[1] = th
        self.d.ctrl[sl[0]] = th

    # ------------------------------------------------------------------------------------------------ attach
    def _attach(self, name, rule, seat=True):
        m, d = self.m, self.d
        h = self.handles[name]
        mujoco.mj_kinematics(m, d)
        xH, RH, qH = d.xpos[self.hull].copy(), d.xmat[self.hull].reshape(3, 3).copy(), d.xquat[self.hull].copy()
        b = h["bid"]
        xV, RV, qV = d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy(), d.xquat[b].copy()
        A_h0 = RH.T @ (xV + RV @ h["A_b"] - xH)
        q0 = quat_mul(quat_conj(qH), qV)
        q0 = q0 / np.linalg.norm(q0)
        eq = h["eq"]
        m.eq_data[eq, 0:3] = h["A_b"]
        if seat:
            m.eq_data[eq, 3:6] = A_h0
            m.eq_data[eq, 6:10] = q0
        else:
            m.eq_data[eq, 3:6] = self.tcp_B
            m.eq_data[eq, 6:10] = h["q_target"]
        m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1
        if seat and h["lock"] is not None:
            lk = h["lock"]
            m.eq_data[lk, :] = 0.0
            m.eq_data[lk, 0] = float(d.qpos[h["jqa"]])
            d.eq_active[lk] = 1
            self.locked = name
        self._set_bits(name, True)
        self._freeze_jaws()
        self.held = ("paddle", name)
        self.state = "stalled_on_object"
        self.over_t = 0.0
        t = float(d.time)
        self.seat = (t, A_h0, q0) if seat else None
        self.t_seat_end = t + self.V["seat_s"] if seat else t
        self.seat_stats = dict(handle=name, t_grasp=t, angle0_deg=self._angle(name), dtheta_max_deg=0.0,
                               residual_in_seat_mm=0.0, residual_after_seat_mm=0.0, lateral_in_seat_N=0.0,
                               bending_in_seat_Nm=0.0, lateral_after_N=0.0, bending_after_Nm=0.0,
                               over_in_seat_s=0.0, _run=0.0)
        self._event(ev="grasp", kind="paddle", handle=name, handle_kind=h["kind"], rule=rule)

    def attach_initial(self, name):
        """Start holding a free handle (the marker at the start of an episode): place its body at the canonical pose
        relative to the hull, jaws on the blade, weld on (no glide)."""
        m, d = self.m, self.d
        h = self.handles[name]
        jt = m.body_jntadr[h["bid"]]
        if jt < 0 or m.jnt_type[jt] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("attach_initial needs a free body")
        mujoco.mj_kinematics(m, d)
        xH, RH, qH = d.xpos[self.hull], d.xmat[self.hull].reshape(3, 3), d.xquat[self.hull]
        q_b = quat_mul(qH, h["q_target"])
        q_b = q_b / np.linalg.norm(q_b)
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, q_b)
        x_b = xH + RH @ self.tcp_B - Rb.reshape(3, 3) @ h["A_b"]
        qa = int(m.jnt_qposadr[jt])
        d.qpos[qa:qa + 3] = x_b
        d.qpos[qa + 3:qa + 7] = q_b
        da = int(m.jnt_dofadr[jt])
        d.qvel[da:da + 6] = 0.0
        th = jaw_touch_angle(m, float(h["paddle_m"][2]))
        jaw_angle_for(m, d, th)
        mujoco.mj_forward(m, d)
        self._attach(name, rule=dict(initial=True), seat=False)
        self._freeze_jaws(th)

    def _detach(self, why, **extra):
        m, d = self.m, self.d
        if self.held is None:
            return None
        kind, name = self.held
        if kind == "paddle":
            h = self.handles[name]
            d.eq_active[h["eq"]] = 0
            if self.locked:
                d.eq_active[self.handles[self.locked]["lock"]] = 0
                self.locked = None
            self.seat = None
            self._event(ev=why, kind="paddle", handle=name, angle_deg=self._angle(name), **extra)
        else:
            d.eq_active[self.eyes[name]["eq"]] = 0
            self._event(ev=why, kind="eye", eye=name, body=self.eyes[name]["body"], **extra)
        self.held = None
        return name

    def _angle(self, name):
        h = self.handles[name]
        return math.degrees(float(self.d.qpos[h["jqa"]])) if h["jqa"] is not None else None

    # ------------------------------------------------------------------------------------------------ loads
    def weld_load(self):
        """Load on the held handle through the weld: lateral force [N], bending [N m] about the jaw-tip point on the
        tool axis, axial force, torsion about the tool axis, position residual [mm]."""
        m, d = self.m, self.d
        h = self.handles[self.held[1]]
        eq = h["eq"]
        if self._vec is None or self._vec.shape[0] != d.nefc:
            self._vec = np.zeros(d.nefc)
        vec = self._vec
        vec[:] = 0.0
        rows = np.nonzero((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[:d.nefc] == eq))[0]
        vec[rows] = d.efc_force[rows]
        res = np.zeros(m.nv)
        mujoco.mj_mulJacTVec(m, d, res, vec)
        F = res[self.vadr:self.vadr + 3]
        RH = d.xmat[self.hull].reshape(3, 3)
        Mo = RH @ res[self.vadr + 3:self.vadr + 6]
        xH = d.xpos[self.hull]
        a = RH @ self.axis_B
        tip = d.site_xpos[self.tcp_sid] + a * self.tip_ahead
        Mt = Mo - np.cross(tip - xH, F)
        Fh, Mh = -F, -Mt
        ax = float(Fh @ a)
        tor = float(Mh @ a)
        lat = float(np.linalg.norm(Fh - ax * a))
        ben = float(np.linalg.norm(Mh - tor * a))
        b = h["bid"]
        p1 = xH + RH @ m.eq_data[eq, 3:6]
        p2 = d.xpos[b] + d.xmat[b].reshape(3, 3) @ m.eq_data[eq, 0:3]
        return dict(lateral=lat, bending=ben, axial=ax, torsion=tor, residual_mm=float(np.linalg.norm(p1 - p2) * 1e3))

    def eye_load(self):
        """Magnitude of the ball-joint force [N] of the held eye."""
        d = self.d
        eq = self.eyes[self.held[1]]["eq"]
        rows = np.nonzero((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[:d.nefc] == eq))[0]
        return float(np.linalg.norm(d.efc_force[rows])) if rows.size else 0.0

    # ------------------------------------------------------------------------------------------------ per step
    def tick_pre(self):
        """Before mj_step: advance the seating glide."""
        if self.seat is not None:
            t0, A0, q0 = self.seat
            u = smootherstep((float(self.d.time) - t0) / self.V["seat_s"])
            h = self.handles[self.held[1]]
            self.m.eq_data[h["eq"], 3:6] = (1.0 - u) * A0 + u * self.tcp_B
            self.m.eq_data[h["eq"], 6:10] = _slerp(q0, h["q_target"], u)

    def tick_post(self):
        """After mj_step: close completion and the rules, glide end, loads and slip, release on open."""
        d, dt, V = self.d, float(self.m.opt.timestep), self.V
        t = float(d.time)
        qj = float(d.qpos[self.ju_qa])
        if self.state == "closing":
            self._qh.append(qj)
            cl = self._cl
            if self.trace_close and cl is not None and cl["t_contact"] is None:
                touch, other = self._touch()
                if any(len(s) == 2 for s in touch.values()) or other:
                    cl["t_contact"] = t
            armed = t - self.t_close > V["stall_arm_s"]
            if V["stall_rule"] == "rate":
                self._lit_t = self._lit_t + dt if (abs(float(d.qvel[self.ju_da])) < V["stall_rate_rad_s"] and armed) else 0.0
                stalled = self._lit_t >= V["stall_window_s"] - 1e-9
            else:
                stalled = (armed and len(self._qh) == self._qh.maxlen and
                           max(self._qh) - min(self._qh) < V["stall_span_rad"])
            if qj <= V["closed_rad"]:
                self._finish_close(stalled=False)
            elif stalled:
                self._finish_close(stalled=True)
        elif self.state == "opening":
            full = qj >= self.theta_open - V["open_tol_rad"]
            if self.held and (self._open_from is None or qj - self._open_from > V["release_open_m"] / self.lever_pad or full):
                kind, name = self.held
                self._detach("release")
                if kind == "paddle" and self.handles[name]["jqa"] is not None:
                    self._set_bits(name, False)        # a valve handle stays on its hinge: contacts back at once
                # a free handle (marker, stab) keeps ignoring the vehicle until it is clear of the jaws: otherwise the
                # blade rests on the open lower jaw (inclined 24 deg, friction 1.2) and a marker released in mid
                # water hangs there instead of falling (measured, record P2_P05)
            if full and not self.held:
                self.state = "open"
        if self.held and self.held[0] == "paddle":
            self._paddle_hold_step(t, dt)
        # restore contacts of a passed-through blade once it is clear of the jaws
        if self.pass_through:
            held = self.held[1] if self.held and self.held[0] == "paddle" else None
            pT = d.site_xpos[self.tcp_sid]
            for n in list(self.pass_through):
                if n != held and np.linalg.norm(self.paddle_frame_w(n)[0] - pT) > V["pass_clear_m"]:
                    self._set_bits(n, False)

    def _paddle_hold_step(self, t, dt):
        V, d = self.V, self.d
        L = self.weld_load()
        self.load = L
        S = self.seat_stats
        name = self.held[1]
        if self.seat is not None:
            if S is not None:
                a = self._angle(name)
                if a is not None:
                    S["dtheta_max_deg"] = max(S["dtheta_max_deg"], abs(a - S["angle0_deg"]))
                S["residual_in_seat_mm"] = max(S["residual_in_seat_mm"], L["residual_mm"])
                S["lateral_in_seat_N"] = max(S["lateral_in_seat_N"], L["lateral"])
                S["bending_in_seat_Nm"] = max(S["bending_in_seat_Nm"], L["bending"])
                over = L["lateral"] > V["slip_lateral_N"] or L["bending"] > V["slip_bending_Nm"]
                S["_run"] = S["_run"] + dt if over else 0.0
                S["over_in_seat_s"] = max(S["over_in_seat_s"], S["_run"])
            if t - self.seat[0] >= V["seat_s"] - 1e-12:
                self.seat = None
                self.t_seat_end = t
                if self.locked:
                    d.eq_active[self.handles[self.locked]["lock"]] = 0
                    self.locked = None
                self._event(ev="seated", kind="paddle", handle=name, angle_deg=self._angle(name),
                            dtheta_seat_deg=None if S is None else S["dtheta_max_deg"])
            return
        if S is not None:
            S["residual_after_seat_mm"] = max(S["residual_after_seat_mm"], L["residual_mm"])
            S["lateral_after_N"] = max(S["lateral_after_N"], L["lateral"])
            S["bending_after_Nm"] = max(S["bending_after_Nm"], L["bending"])
        if L["lateral"] > V["slip_lateral_N"] or L["bending"] > V["slip_bending_Nm"]:
            self.over_t += dt
            if self.over_t >= V["slip_window_s"] - 1e-9:
                self._detach("slip", lateral=L["lateral"], bending=L["bending"])
                self.state = "closing"                     # the jaws close through the blade
                self.t_close = t
                self._qh.clear()
                self._cl = dict(t_cmd=t, t_contact=None, t_done=None, result=None, after="slip")
                self.sim.set_gripper(0.0)
        else:
            self.over_t = 0.0

    def _finish_close(self, stalled):
        t = float(self.d.time)
        touch, other = self._touch()
        both = [n for n, s in touch.items() if len(s) == 2]
        if stalled and both:
            name = both[0]
            rp = self.rel_pose(name)
            V = self.V
            ok = [abs(rp["vertical"]) <= V["centre_tol_m"], abs(rp["lateral"]) <= V["lateral_m"],
                  rp["bite"] >= V["bite_min_m"], max(abs(x) for x in rp["angles_deg"]) <= V["angle_deg"]]
            rule = dict(rp, ok=ok, t=t)
            self.last_rule = rule
            if all(ok):
                self._attach(name, rule)
                self._close_done("stalled_on_object", t)
                return
            self._set_bits(name, True)                     # window miss: the jaws close through the blade
            self._event(ev="window_miss", kind="paddle", handle=name, rule=rule)
            self._continue_closing(t)
            return
        if stalled and touch:
            for n in touch:                                # a blade touched by one jaw only: close through it
                self._set_bits(n, True)
            self._event(ev="stall_no_grasp", touch=sorted(touch))
            self._continue_closing(t)
            return
        # stroke end, or blocked by something that is not a graspable blade: eye rule, else closed_empty
        if self._eye_rule(t):
            self._close_done("stalled_on_object", t)
            return
        if stalled and other:
            self._event(ev="stall_no_grasp", touch=[])
        self.state = "closed_empty"
        self._close_done("closed_empty", t)

    def _continue_closing(self, t):
        self.state = "closing"
        self.t_close = t
        self._qh.clear()
        self._lit_t = 0.0

    def _close_done(self, result, t):
        if self._cl is not None:
            self._cl.update(t_done=t, result=result)
            self.close_log.append(self._cl)
            self._cl = None

    def _eye_rule(self, t):
        if not self.eyes:
            return False
        d, m = self.d, self.m
        pT = d.site_xpos[self.tcp_sid]
        best, bd = None, float("inf")
        for n, e in self.eyes.items():
            b = e["bid"]
            p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ e["p_b"]
            dist = float(np.linalg.norm(p - pT))
            if dist < bd:
                best, bd = n, dist
        e = self.eyes[best]
        if bd <= self.V["eye_capture_m"]:
            b = e["bid"]
            p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ e["p_b"]
            RH = d.xmat[self.hull].reshape(3, 3)
            m.eq_data[e["eq"], 0:3] = e["p_b"]
            m.eq_data[e["eq"], 3:6] = RH.T @ (p - d.xpos[self.hull])
            d.eq_active[e["eq"]] = 1
            self._freeze_jaws()
            self.held = ("eye", best)
            self.state = "stalled_on_object"
            self._event(ev="grasp", kind="eye", eye=best, body=e["body"], dist_m=round(bd, 5))
            return True
        if bd <= 0.1:
            self._event(ev="eye_miss", kind="eye", eye=best, body=e["body"], dist_m=round(bd, 5))
        return False
