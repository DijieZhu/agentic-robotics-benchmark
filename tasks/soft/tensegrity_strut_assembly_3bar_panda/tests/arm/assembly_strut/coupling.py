"""Coupling state machine of the strut assembly cell: one state per SLOT (a mate of the mating graph on one strut),
each slot taking whichever of its candidate units reaches it (the six motor modules, the three lids and the six end
caps are interchangeable; the motor modules come in two builds of one design). Written after the quadruped cell's
`coupling.py` (slot x candidate, one weld per pair declared in advance, a fastened unit leaves every other slot's
candidate list, the pull home at <= 10 mm/s and 10 deg/s), with a new kind for the rod.

Slot kinds:
  rod_sleeve  a module with a bore slides on the strut's rod (motor module: through bore, both ends; end cap: blind
              bore). The real clearance (0.11 mm) is below what contact simulation resolves, so the collision pieces
              carry a slot of radius R_s around the bore (fine_collision.py) and the capture stands in for the rod:
              CAPTURE when the leading mouth is within `capture_lateral` of the rod axis, the axes within
              `capture_ang` (unsigned: a module turned end for end is guided too), the rod end at least
              `capture_depth` into the bore. Then the module is welded to the slot's RAIL: two light bodies under the
              strut core on the rod axis, a slider along the rod and a hinge about it (joint damping = the damping
              along and about the rod). The weld holds the module on the axis (lateral, tilt); along and about the
              rod it moves freely, the rail following it; the weld's target is pulled from the pose at the capture to
              the axis at <= 10 mm/s and 10 deg/s. RELEASE when the module leaves twice the capture region or the
              mouth leaves the rod end. LOCK (HOLDING for `hold_ticks`: within the lock window of the design axial
              position, at rest relative to the strut) -> a rigid weld to the strut core where it stopped (on the
              axis, its axial position and roll). Score: the motor module when |axial| <= tol_axial and its rotation
              from the design <= tol_roll (a module turned end for end is 180 deg off); the end cap (roll free) when
              it is the strut's first cap or of that cap's colour; else LOCKED_OFF (welded, counts 0, no re-fit).
  part_seat   the bay lid on the strut core (the precedent's seat): capture within `seat_lateral` across, at most
              `seat_depth` short of home, `seat_ang`; a soft weld pulls it home at <= 10 mm/s; lock within
              tol_pos / tol_ang at rest -> rigid weld at the design.
States: WAITING (a required slot not achieved) -> FREE -> ENGAGED -> HOLDING -> LOCKED / LOCKED_OFF.

`declare(spec, kit, scope, carriers)` adds the rails and the welds to an MjSpec before compiling (call it once);
`StrutCoupling(model, data, kit, scope, carriers)` runs the state machine (`tick(sim_time)` once per control tick,
after the physics); `rod_loads()` gives each rod overhang's load (the welds' translational force on the rod side plus
the contacts on the rod's collision geom beyond the bay, summed per overhang), the reading of the rod rule R3.
"""
import math

import numpy as np

STATES = ("WAITING", "FREE", "ENGAGED", "HOLDING", "LOCKED", "LOCKED_OFF")

DEFAULTS = {
    "hold_ticks": 50, "rel_speed_max_m_s": 0.005, "rest_ang_speed_max_deg_s": 2.0,
    "rod_sleeve": {"capture_lateral_m": None, "capture_ang_deg": 5.0, "capture_depth_m": 0.003, "release_factor": 2.0,
                   "overhang_extra_m": 0.006,
                   "axial_damping_N_s_m": 0.5, "roll_damping_N_m_s_rad": 1.0e-4},
    "seat": {"capture_lateral_m": 0.004, "capture_seat_depth_m": 0.003, "capture_ang_deg": 8.0},
    "pull_speed_m_s": 0.010, "pull_rate_deg_s": 10.0,
    "soft_solref": [0.03, 1.0], "weld_solref": [0.002, 1.0], "weld_solimp": [0.99, 0.999, 0.001, 0.5, 2.0],
    "rail": {"mass_kg": 1.0e-3, "inertia_kg_m2": 1.0e-9, "armature_slide": 1.0e-4, "armature_hinge": 1.0e-8},
    "bay_half_length_m": 0.0715,
}


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _rotx_pi():
    T = np.eye(4)
    T[1, 1] = T[2, 2] = -1.0
    return T


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def _tz(z):
    T = np.eye(4)
    T[2, 3] = z
    return T


def rotation_angle(R):
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))


def quat_wxyz(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def body_pose(d, b):
    T = np.eye(4)
    T[:3, :3] = d.xmat[b].reshape(3, 3)
    T[:3, 3] = d.xpos[b]
    return T


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _mm(T):
    T = np.array(T, float).copy()
    T[:3, 3] *= 1e-3
    return T


def strut_of(mate):
    return mate["part_a"].split(".")[0]


def rail_names(sid):
    return "rail_%s" % sid, "rail_%s_roll" % sid


def weld_name(kind, sid, unit):
    return "%s_%s__%s" % (kind, sid, unit)


def _cfg(cfg):
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k].update(v)
        else:
            c[k] = v
    return c


def declare(spec, kit, scope, carriers, cfg=None):
    """the rails (rod_sleeve slots) and one inactive weld per (slot, candidate): `rw_` to the rail, `lw_` to the
    carrier (the lock; the seat's capture and lock). `carriers` = {strut: body name of its strut core}."""
    import mujoco
    c = _cfg(cfg)
    rl = c["rail"]
    for mt in kit.mates(scope):
        sid, carrier = mt["id"], carriers[strut_of(mt)]
        cb = spec.body(carrier)
        cands = kit.units_of_type(mt["part_b_type"])
        if mt["kind"] == "rod_sleeve":
            Fa = np.array(mt["frame_a_in_a"]["T"], float)
            end = np.array(mt["rod_end_in_a"], float) * 1e-3
            n1, n2 = rail_names(sid)
            rb = cb.add_body(name=n1, pos=end.tolist(), quat=quat_wxyz(Fa[:3, :3]).tolist(), gravcomp=1.0)
            rb.mass, rb.inertia, rb.explicitinertial = rl["mass_kg"], [rl["inertia_kg_m2"]] * 3, True
            j = rb.add_joint(name=n1 + "_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1.0],
                             damping=float(c["rod_sleeve"]["axial_damping_N_s_m"]), armature=float(rl["armature_slide"]))
            j.limited = mujoco.mjtLimited.mjLIMITED_FALSE
            rr = rb.add_body(name=n2, gravcomp=1.0)
            rr.mass, rr.inertia, rr.explicitinertial = rl["mass_kg"], [rl["inertia_kg_m2"]] * 3, True
            h = rr.add_joint(name=n2 + "_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1.0],
                             damping=float(c["rod_sleeve"]["roll_damping_N_m_s_rad"]), armature=float(rl["armature_hinge"]))
            h.limited = mujoco.mjtLimited.mjLIMITED_FALSE
            for u in cands:
                eq = spec.add_equality(name=weld_name("rw", sid, u), type=mujoco.mjtEq.mjEQ_WELD,
                                       objtype=mujoco.mjtObj.mjOBJ_BODY, name1=n2, name2=u)
                eq.active, eq.solref, eq.solimp = False, list(c["soft_solref"]), list(c["weld_solimp"])
        for u in cands:
            eq = spec.add_equality(name=weld_name("lw", sid, u), type=mujoco.mjtEq.mjEQ_WELD,
                                   objtype=mujoco.mjtObj.mjOBJ_BODY, name1=carrier, name2=u)
            eq.active, eq.solref, eq.solimp = False, list(c["weld_solref"]), list(c["weld_solimp"])


class StrutCoupling(object):
    def __init__(self, model, data, kit, scope, carriers, cfg=None, rod_geoms=None):
        import mujoco
        self.mj, self.m, self.d, self.kit = mujoco, model, data, kit
        c = self.c = _cfg(cfg)
        cap = kit.capture() or {}
        rs = c["rod_sleeve"]
        self.cap_lat = float(rs["capture_lateral_m"] if rs["capture_lateral_m"] is not None else cap.get("rod_sleeve_axis_distance_mm", 2.0) * 1e-3)
        self.cap_ang = math.radians(float(rs["capture_ang_deg"]))
        self.cap_depth = float(rs["capture_depth_m"])
        self.release_factor = float(rs["release_factor"])
        self.overhang_extra = float(rs.get("overhang_extra_m", 0.006))
        st = c["seat"]
        self.seat_lat, self.seat_depth, self.seat_ang = float(st["capture_lateral_m"]), float(st["capture_seat_depth_m"]), math.radians(float(st["capture_ang_deg"]))
        self.hold_ticks, self.rel_speed_max = int(c["hold_ticks"]), float(c["rel_speed_max_m_s"])
        self.rest_ang = math.radians(float(c["rest_ang_speed_max_deg_s"]))
        self.pull_v, self.pull_w = float(c["pull_speed_m_s"]), math.radians(float(c["pull_rate_deg_s"]))
        self.soft_solref, self.hard_solref = list(c["soft_solref"]), list(c["weld_solref"])
        bores = kit.bores() or {"M1": {"through": True, "length_mm": 55.0}, "M3": {"through": False, "depth_mm": 21.0}}
        self.carriers = dict(carriers)
        self.slots = []
        for mt in kit.mates(scope):
            x = {"id": mt["id"], "kind": mt["kind"], "weight": float(mt["weight"]), "requires": list(mt.get("requires", [])),
                 "strut": strut_of(mt), "bc": model.body(carriers[strut_of(mt)]).id, "type": mt["part_b_type"]}
            x["Fa"] = _mm(mt["frame_a_in_a"]["T"])
            x["Fb"] = _mm(mt["frame_b_in_b"]["T"])
            x["Tdes"] = _mm(mt["design_pose_b_in_a"])
            if mt["kind"] == "rod_sleeve":
                n1, n2 = rail_names(mt["id"])
                x["rail_base"] = model.body(n1).id
                x["rail_roll"] = model.body(n2).id
                x["qs"] = model.jnt_qposadr[model.joint(n1 + "_slide").id]
                x["qr"] = model.jnt_qposadr[model.joint(n2 + "_hinge").id]
                x["vs"] = model.jnt_dofadr[model.joint(n1 + "_slide").id]
                x["vr"] = model.jnt_dofadr[model.joint(n2 + "_hinge").id]
                x["home"] = float(mt["home_depth_mm"]) * 1e-3
                A0 = np.eye(4)
                A0[:3, :3] = x["Fa"][:3, :3]
                A0[:3, 3] = np.array(mt["rod_end_in_a"], float) * 1e-3
                x["A0"] = A0                     # the rod end frame in the carrier: z = the insertion along the rod
                b = bores[mt["part_b_type"]]
                x["through"] = bool(b["through"])
                x["L"] = float(b.get("length_mm", b.get("depth_mm", 0.0))) * 1e-3
                x["lock_win"] = float(mt["lock_axial_window_mm"]) * 1e-3
                x["tol_axial"] = float(mt["tol_axial_mm"]) * 1e-3
                x["tol_roll"] = math.radians(float(mt["tol_roll_deg"])) if mt.get("tol_roll_deg") is not None else None
                x["colour_rule"] = bool(mt.get("colour_rule"))
                x["end"] = "p" if mt["id"].endswith("_p") else "n"
            else:
                ins = np.array(mt["insertion_axis_in_a"], float)
                x["ins"] = ins / np.linalg.norm(ins)
                x["tol_pos"] = float(mt["tol_pos_mm"]) * 1e-3
                x["tol_ang"] = math.radians(float(mt["tol_ang_deg"]))
            x["cands"] = []
            for u in kit.units_of_type(mt["part_b_type"]):
                cd = {"unit": u, "bp": model.body(u).id, "lw": model.equality(weld_name("lw", mt["id"], u)).id,
                      "colour": kit.units[u].get("colour")}
                if mt["kind"] == "rod_sleeve":
                    cd["rw"] = model.equality(weld_name("rw", mt["id"], u)).id
                x["cands"].append(cd)
            self.slots.append(x)
        self.by_id = {x["id"]: x for x in self.slots}
        self.state = {x["id"]: "WAITING" for x in self.slots}
        self.rec = {x["id"]: {"id": x["id"], "kind": x["kind"], "weight": x["weight"], "unit": None, "fastened": False,
                              "achieved": False, "t_s": None, "axial_err_mm": None, "roll_err_deg": None, "pos_err_mm": None,
                              "ang_err_deg": None, "reversed": None, "why_off": None, "capture": None} for x in self.slots}
        self.hold = {x["id"]: 0 for x in self.slots}
        self.bound = {}             # slot -> candidate index
        self.flip = {}              # slot -> captured end for end (rod_sleeve)
        self.pull = {}
        self.used = {}              # unit -> slot that holds or fastened it
        self.strut_colour = {}
        self.events = []
        self.captures = []          # (slot, unit, lateral mm at the leading mouth, far mouth mm, angle deg, depth mm)
        self.total_w = sum(x["weight"] for x in self.slots)
        self.rod_geoms = dict(rod_geoms or {})          # strut -> geom id of its rod
        self.bay_half = float(c["bay_half_length_m"])

    # ------------------------------------------------------------------ helpers
    def score(self):
        return sum(x["weight"] for x in self.slots if self.rec[x["id"]]["achieved"]) / self.total_w

    def _vel(self, b):
        v = np.zeros(6)
        self.mj.mj_objectVelocity(self.m, self.d, self.mj.mjtObj.mjOBJ_BODY, b, v, 0)
        return v

    def _rel_speed(self, x, c):
        vc, vp = self._vel(x["bc"]), self._vel(c["bp"])
        return float(np.linalg.norm(vp[3:] - vc[3:])), float(np.linalg.norm(vp[:3] - vc[:3]))

    def rail_base_world(self, x):
        """the rod end frame of the slot in the world (the rail's zero: the slider moves its body along z from here)"""
        return body_pose(self.d, x["bc"]) @ x["A0"]

    def G_world(self, x, s, r):
        """the rail's roll frame in the world at slide s (m) and roll r (rad)"""
        return self.rail_base_world(x) @ _tz(s) @ _rz(r)

    def canonical(self, x, flip):
        """the unit's pose in the rail's roll frame on the axis: the design orientation, or turned end for end"""
        C = _inv(x["Fb"])
        return (_rotx_pi() @ C) if flip else C

    # ------------------------------------------------------------------ rod_sleeve geometry
    def sleeve_eval(self, x, c):
        """the module against the rod as it is now: leading mouth, depth, lateral, angle, axial error, roll"""
        A = self.rail_base_world(x)
        u, pe, x0 = A[:3, 2], A[:3, 3], A[:3, 0]
        B = body_pose(self.d, c["bp"]) @ x["Fb"]
        b, oB = B[:3, 2], B[:3, 3]
        cosang = float(b @ u)
        flip = cosang < 0
        ang = math.acos(min(1.0, abs(cosang)))
        if flip and not x["through"]:
            lead = None
        else:
            lead = (oB - x["L"] * b) if flip else oB
        s = float((oB - pe) @ u)
        mid = oB - 0.5 * x["L"] * b
        mid_home = pe + (x["home"] - 0.5 * x["L"]) * u
        r = {"flip": flip, "ang": ang, "s": s, "axial_err": float((mid - mid_home) @ u)}
        if lead is None:
            r.update({"depth": -1.0, "lateral": 1.0})
        else:
            dv = lead - pe
            r["depth"] = float(dv @ u)
            r["lateral"] = float(np.linalg.norm(dv - (dv @ u) * u))
        far = (oB if flip else oB - x["L"] * b) - pe
        r["lateral_far"] = float(np.linalg.norm(far - (far @ u) * u))
        xb = B[:3, 0] - (B[:3, 0] @ u) * u
        n = np.linalg.norm(xb)
        r["roll"] = math.atan2(float(np.cross(x0, xb) @ u), float(x0 @ xb)) if n > 1e-9 else 0.0
        return r

    def _sleeve_inside(self, r, x, f=1.0):
        """inside the capture region (f = 1) or inside the release region (f = release_factor); the leading mouth must
        be on this end's overhang: past the rod end, at most `overhang_extra` beyond home (the bay stops a module
        4.9 mm past home; a module on the other end of the rod is not this slot's)"""
        lo = self.cap_depth if f <= 1.0 else 0.0
        hi = x["home"] + self.overhang_extra * (1.0 if f <= 1.0 else 2.0)
        return (lo <= r["depth"] <= hi and r["lateral"] <= self.cap_lat * f and r["ang"] <= self.cap_ang * f)

    # ------------------------------------------------------------------ seat geometry
    def seat_eval(self, x, c):
        Tc, Tp = body_pose(self.d, x["bc"]), body_pose(self.d, c["bp"])
        A, B = Tc @ x["Tdes"] @ x["Fb"], Tp @ x["Fb"]
        E = _inv(A) @ B
        dw = B[:3, 3] - A[:3, 3]
        ins_w = Tc[:3, :3] @ x["ins"]
        axial = float(dw @ ins_w)
        return {"pos_err": float(np.linalg.norm(E[:3, 3])), "ang_err": rotation_angle(E[:3, :3]), "axial": axial,
                "lateral": float(np.linalg.norm(dw - axial * ins_w))}

    def _seat_inside(self, r, f=1.0):
        return (r["ang_err"] <= self.seat_ang * f and r["lateral"] <= self.seat_lat * f
                and -self.seat_depth * f <= r["axial"] <= self.seat_lat * f)

    # ------------------------------------------------------------------ welds
    def _weld(self, e, rel, anchor, soft):
        """weld e (body1 = rail roll / carrier, body2 = unit) holding the unit at `rel` (its pose in body1's frame);
        the weld point at `anchor` (unit frame)"""
        self.m.eq_data[e, 0:3] = anchor
        self.m.eq_data[e, 3:6] = rel[:3, 3] + rel[:3, :3] @ anchor
        self.m.eq_data[e, 6:10] = quat_wxyz(rel[:3, :3])
        self.m.eq_data[e, 10] = 1.0
        self.m.eq_solref[e] = self.soft_solref if soft else self.hard_solref
        self.d.eq_active[e] = 1

    def _pull(self, sid, T0, home, anchor, sim_time):
        """the soft target from the relative pose at the capture towards home, <= pull_speed at the anchor and
        <= pull_rate in angle"""
        from scipy.spatial.transform import Rotation, Slerp
        p = self.pull.get(sid)
        if p is None:
            dist = float(np.linalg.norm((T0[:3, :3] @ anchor + T0[:3, 3]) - (home[:3, :3] @ anchor + home[:3, 3])))
            ang = rotation_angle(T0[:3, :3].T @ home[:3, :3])
            p = self.pull[sid] = {"T0": T0, "t0": float(sim_time), "dur": max(dist / self.pull_v, ang / self.pull_w, 1e-3),
                                  "dist": dist, "ang": ang}
        s_ = min(1.0, (float(sim_time) - p["t0"]) / p["dur"])
        if s_ >= 1.0:
            return home, True
        R = Slerp([0, 1], Rotation.from_matrix([p["T0"][:3, :3], home[:3, :3]]))([s_]).as_matrix()[0]
        pa = (1 - s_) * (p["T0"][:3, :3] @ anchor + p["T0"][:3, 3]) + s_ * (home[:3, :3] @ anchor + home[:3, 3])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pa - R @ anchor
        return T, False

    def pull_target_rel(self, sid):
        """the capture weld's current target (rod_sleeve: the unit in the rail's roll frame; seat: in the carrier's
        frame), or None"""
        return self._cur_target.get(sid) if hasattr(self, "_cur_target") else None

    def _release(self, x, ci):
        c = x["cands"][ci]
        for k in ("rw", "lw"):
            if k in c:
                self.d.eq_active[c[k]] = 0
        self.pull.pop(x["id"], None)
        self.bound.pop(x["id"], None)
        self.flip.pop(x["id"], None)
        if self.used.get(c["unit"]) == x["id"]:
            self.used.pop(c["unit"], None)

    # ------------------------------------------------------------------ per control tick
    def tick(self, sim_time):
        if not hasattr(self, "_cur_target"):
            self._cur_target = {}
        for x in self.slots:
            sid = x["id"]
            st = self.rec[sid]
            if st["fastened"]:
                self.state[sid] = "LOCKED" if st["achieved"] else "LOCKED_OFF"
                continue
            if not all(self.rec[q]["achieved"] for q in x["requires"]):
                if sid in self.bound:
                    self._release(x, self.bound[sid])
                self.state[sid] = "WAITING"
                self.hold[sid] = 0
                continue
            if x["kind"] == "rod_sleeve":
                self._tick_sleeve(x, sim_time)
            else:
                self._tick_seat(x, sim_time)

    def _tick_sleeve(self, x, sim_time):
        mj, m, d = self.mj, self.m, self.d
        sid = x["id"]
        ci = self.bound.get(sid)
        if ci is not None:
            c = x["cands"][ci]
            r = self.sleeve_eval(x, c)
            if not self._sleeve_inside(r, x, self.release_factor):
                self._release(x, ci)
                self.events.append((sim_time, "released from capture %s (%s)" % (sid, c["unit"])))
                ci = None
        if ci is None:
            best = None
            for i, c in enumerate(x["cands"]):
                if c["unit"] in self.used:
                    continue
                r = self.sleeve_eval(x, c)
                if self._sleeve_inside(r, x) and (best is None or r["lateral"] < best[1]["lateral"]):
                    best = (i, r)
            if best is None:
                self.state[sid] = "FREE"
                self.hold[sid] = 0
                return
            ci, r = best
            c = x["cands"][ci]
            self.bound[sid], self.used[c["unit"]], self.flip[sid] = ci, sid, r["flip"]
            # the rail to the module's axial position and roll; zero rail velocity
            d.qpos[x["qs"]], d.qpos[x["qr"]] = r["s"], r["roll"]
            d.qvel[x["vs"]] = d.qvel[x["vr"]] = 0.0
            mj.mj_kinematics(m, d)
            self.captures.append({"slot": sid, "unit": c["unit"], "t_s": round(float(sim_time), 3), "flip": bool(r["flip"]),
                                  "lateral_mm": round(r["lateral"] * 1e3, 4), "lateral_far_mouth_mm": round(r["lateral_far"] * 1e3, 4),
                                  "angle_deg": round(math.degrees(r["ang"]), 4), "depth_mm": round(r["depth"] * 1e3, 3)})
            self.rec[sid]["capture"] = self.captures[-1]
            self.events.append((sim_time, "ENGAGED %s (%s%s)" % (sid, c["unit"], ", end for end" if r["flip"] else "")))
        c = x["cands"][ci]
        G = body_pose(d, x["rail_roll"])
        T0 = _inv(G) @ body_pose(d, c["bp"])
        home = self.canonical(x, self.flip[sid])
        anchor = x["Fb"][:3, 3]
        tgt, done = self._pull(sid, T0, home, anchor, sim_time)
        self._cur_target[sid] = tgt
        self._weld(c["rw"], tgt, anchor, soft=True)
        rs, ra = self._rel_speed(x, c)
        inwin = abs(r["axial_err"]) <= x["lock_win"]
        if not (done and inwin and rs < self.rel_speed_max and ra < self.rest_ang):
            self.state[sid] = "ENGAGED"
            self.hold[sid] = 0
            return
        self.hold[sid] += 1
        self.state[sid] = "HOLDING"
        if self.hold[sid] >= self.hold_ticks:
            self._lock_sleeve(x, ci, r, sim_time)

    def _lock_sleeve(self, x, ci, r, sim_time):
        d = self.d
        sid = x["id"]
        c = x["cands"][ci]
        flip = self.flip[sid]
        s, roll = float(d.qpos[x["qs"]]), float(d.qpos[x["qr"]])
        # where it stopped, on the axis: the rail frame now with the canonical pose
        Tw = self.G_world(x, s, roll) @ self.canonical(x, flip)
        Tc = body_pose(d, x["bc"])
        rel = _inv(Tc) @ Tw
        Tdes = x["Tdes"]
        rot_err = rotation_angle(Tdes[:3, :3].T @ rel[:3, :3])
        st = self.rec[sid]
        ok, why = True, None
        if abs(r["axial_err"]) > x["tol_axial"]:
            ok, why = False, "axial %.2f mm" % (r["axial_err"] * 1e3)
        if x["tol_roll"] is not None and rot_err > x["tol_roll"]:
            ok, why = False, "rotation %.1f deg from the design%s" % (math.degrees(rot_err), " (end for end)" if flip else "")
        if x["colour_rule"] and ok:
            col = self.strut_colour.get(x["strut"])
            if col is None:
                self.strut_colour[x["strut"]] = c["colour"]
            elif c["colour"] != col:
                ok, why = False, "colour %s on a %s strut" % (c["colour"], col)
        self.d.eq_active[c["rw"]] = 0
        self.pull.pop(sid, None)
        self._weld(c["lw"], rel, x["Fb"][:3, 3], soft=False)
        st.update({"fastened": True, "achieved": ok, "t_s": float(sim_time), "unit": c["unit"], "axial_err_mm": r["axial_err"] * 1e3,
                   "roll_err_deg": math.degrees(rot_err) if not flip else 180.0, "reversed": bool(flip), "why_off": why,
                   "rail_s_mm": s * 1e3, "rail_roll_deg": math.degrees(roll)})
        self.bound.pop(sid, None)
        self.used[c["unit"]] = sid
        self.state[sid] = "LOCKED" if ok else "LOCKED_OFF"
        self.events.append((sim_time, ("LOCKED %s (%s)" % (sid, c["unit"])) if ok else "LOCKED_OFF %s (%s: %s)" % (sid, c["unit"], why)))

    def _tick_seat(self, x, sim_time):
        sid = x["id"]
        ci = self.bound.get(sid)
        if ci is not None:
            c = x["cands"][ci]
            r = self.seat_eval(x, c)
            if not self._seat_inside(r, self.release_factor):
                self._release(x, ci)
                self.events.append((sim_time, "released from capture %s (%s)" % (sid, c["unit"])))
                ci = None
        if ci is None:
            best = None
            for i, c in enumerate(x["cands"]):
                if c["unit"] in self.used:
                    continue
                r = self.seat_eval(x, c)
                if self._seat_inside(r) and (best is None or r["pos_err"] < best[1]["pos_err"]):
                    best = (i, r)
            if best is None:
                self.state[sid] = "FREE"
                self.hold[sid] = 0
                return
            ci, r = best
            c = x["cands"][ci]
            self.bound[sid], self.used[c["unit"]] = ci, sid
            self.captures.append({"slot": sid, "unit": c["unit"], "t_s": round(float(sim_time), 3), "lateral_mm": round(r["lateral"] * 1e3, 4),
                                  "axial_mm": round(r["axial"] * 1e3, 4), "angle_deg": round(math.degrees(r["ang_err"]), 4)})
            self.rec[sid]["capture"] = self.captures[-1]
            self.events.append((sim_time, "ENGAGED %s (%s)" % (sid, c["unit"])))
        c = x["cands"][ci]
        Tc = body_pose(self.d, x["bc"])
        T0 = _inv(Tc) @ body_pose(self.d, c["bp"])
        anchor = x["Fb"][:3, 3]
        tgt, done = self._pull(sid, T0, x["Tdes"], anchor, sim_time)
        self._cur_target[sid] = tgt
        self._weld(c["lw"], tgt, anchor, soft=True)
        holds = r["pos_err"] <= x["tol_pos"] and r["ang_err"] <= x["tol_ang"]
        rs, ra = self._rel_speed(x, c)
        if not (holds and rs < self.rel_speed_max and ra < self.rest_ang):
            self.state[sid] = "ENGAGED"
            self.hold[sid] = 0
            return
        self.hold[sid] += 1
        self.state[sid] = "HOLDING"
        if self.hold[sid] >= self.hold_ticks:
            st = self.rec[sid]
            self.pull.pop(sid, None)
            self._weld(c["lw"], x["Tdes"], anchor, soft=False)
            st.update({"fastened": True, "achieved": True, "t_s": float(sim_time), "unit": c["unit"],
                       "pos_err_mm": r["pos_err"] * 1e3, "ang_err_deg": math.degrees(r["ang_err"])})
            self.bound.pop(sid, None)
            self.used[c["unit"]] = sid
            self.state[sid] = "LOCKED"
            self.events.append((sim_time, "LOCKED %s (%s)" % (sid, c["unit"])))

    def record(self):
        return [dict(self.rec[x["id"]]) for x in self.slots]

    # ------------------------------------------------------------------ the rod rule's reading (R3)
    def weld_force_world(self, e):
        """translational force of an active weld on its body1 (the rail or the strut core: the rod side), world"""
        d = self.d
        rows = np.where((d.efc_type[:d.nefc] == self.mj.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[:d.nefc] == e))[0]
        if len(rows) < 3:
            return np.zeros(3)
        return np.array(d.efc_force[rows[:3]], float)

    def rod_loads(self, include_contacts=True):
        """{strut: {"p": {...}, "n": {...}}}: per overhang the total force on the rod (world, N), its component across
        the rod (the R3 reading) and the parts it is made of"""
        m, d, mj = self.m, self.d, self.mj
        out = {}
        for x in self.slots:
            if x["kind"] != "rod_sleeve":
                continue
            o = out.setdefault(x["strut"], {"p": {"F": np.zeros(3), "welds": np.zeros(3), "contacts": np.zeros(3)},
                                            "n": {"F": np.zeros(3), "welds": np.zeros(3), "contacts": np.zeros(3)}})
            ci = self.bound.get(x["id"])
            st = self.rec[x["id"]]
            es = []
            if ci is not None:
                es.append(x["cands"][ci]["rw"])
            if st["fastened"]:
                es.extend(c["lw"] for c in x["cands"] if c["unit"] == st["unit"])
            for e in es:
                if d.eq_active[e]:
                    f = self.weld_force_world(e)
                    o[x["end"]]["welds"] += f
        if include_contacts and self.rod_geoms:
            f6 = np.zeros(6)
            for i in range(d.ncon):
                ct = d.contact[i]
                for strut, g in self.rod_geoms.items():
                    if ct.geom1 != g and ct.geom2 != g:
                        continue
                    mj.mj_contactForce(m, d, i, f6)
                    fw = ct.frame.reshape(3, 3).T @ f6[:3]
                    fr = -fw if ct.geom1 == g else fw         # the force on the rod
                    Tc = body_pose(d, m.geom_bodyid[g])
                    zl = float((Tc[:3, :3].T @ (ct.pos - Tc[:3, 3]))[2])
                    if abs(zl) <= self.bay_half:
                        continue
                    o = out.setdefault(strut, {"p": {"F": np.zeros(3), "welds": np.zeros(3), "contacts": np.zeros(3)},
                                               "n": {"F": np.zeros(3), "welds": np.zeros(3), "contacts": np.zeros(3)}})
                    o["p" if zl > 0 else "n"]["contacts"] += fr
        for strut, o in out.items():
            Tc = body_pose(d, m.body(self.carriers[strut]).id)
            u = Tc[:3, 2]
            for k in ("p", "n"):
                F = o[k]["welds"] + o[k]["contacts"]
                o[k]["F"] = F
                o[k]["lateral"] = float(np.linalg.norm(F - (F @ u) * u))
        return out
