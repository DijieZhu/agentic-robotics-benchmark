"""Coupling state machine of the quadruped assembly cell: one state per SLOT of the kit (`kit.json` "slots"), each slot
taking whichever of its candidate units reaches it (the four upper legs, the four lower legs, the four supports and
each pair of shoulders are identical parts).

Slot kinds:
  keyed  a unit's 18 x 10 key in an output pulley's socket (the pulley is the carrier: a rotor body on an unpowered
         hinge). The key frame (the key's root on the child's face, z into the socket, x along the long side) against
         the socket frame (the pulley face's centre, same axes): the axes within tol_ang, the axis lines within
         tol_pos, the key home within axial_tol, the rotation about the axis within tol_ang of a key orientation
         (k x 180 deg: the key fits two ways). `any_key` slots (HFE, knee: multi-turn joints, zeroed by calibration)
         count in either orientation; the shoulders' slots only in the design one (the flipped shoulder turns its HFE
         output towards the body).
  seat   a hip AA support on the body over the shoulder's journal: its seat frame (the bore, `frame_in_part`) within
         tol_pos / tol_ang of its design pose on the body (it has no key; the two screws locate it). Its capture reaches 8 mm short of home
         along the HAA axis (3 mm across, 6 deg): the M3 x 14 screws reach their threads and pull the bracket onto the
         plate's flange; the model has no bearing on the journal (the d25 x 4 seat clears the d20 journal), and the
         2F-85's fingers, on the bracket's 40.8 mm faces, meet the 46 mm shoulder and the stand's fork 4-6 mm short of
         home (place study, DESIGN.md 11.3).

States: WAITING (a required slot not achieved) -> FREE -> ENGAGED (a candidate inside the capture region: for a keyed
slot the key has entered the socket (its collision lets it in only when it is aligned within the socket's clearance),
for a seat the support at most `capture_seat_depth_m` short of home; a soft weld then holds the unit on the feature
and pulls it home over the last millimetres (the key's height, at most 6 mm; a seat 8 mm) at <= 10 mm/s and 10 deg/s:
the screws being driven, not a jump) -> HOLDING (the predicate holds and the
unit is at rest relative to the carrier, `hold_ticks` ticks) -> LOCKED (the weld turned rigid = the screws) or
LOCKED_OFF (a shoulder keyed the flipped way: welded, counts 0). A fastened unit leaves every other slot's candidate
list. Fastened joints stay free (unpowered actuators back-drive against their friction); a shoulder's fastening sets
its HAA hinge's range (the product's 215 deg, mapped through the as-built angle). Score = the achieved slots' weight /
the total weight.
"""

import math

import numpy as np

STATES = ("WAITING", "FREE", "ENGAGED", "HOLDING", "LOCKED", "LOCKED_OFF")


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
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


def weld_name(slot_id, unit):
    return "weld_%s__%s" % (slot_id, unit)


class CouplingStateMachine(object):
    def __init__(self, model, data, kit, hold_ticks=50, rel_speed_max=0.005, rest_ang_speed_max=math.radians(2.0),
                 capture_lateral_m=0.002, capture_ang_rad=math.radians(6.0), capture_key_frac=0.25, capture_seat_depth_m=0.008,
                 capture_seat_lateral_m=0.003, release_factor=2.0, soft_solref=(0.03, 1.0), pull_speed_m_s=0.010,
                 pull_rate_rad_s=math.radians(10.0)):
        self.m, self.d, self.kit = model, data, kit
        self.hold_ticks, self.rel_speed_max, self.rest_ang = int(hold_ticks), float(rel_speed_max), float(rest_ang_speed_max)
        self.cap_lat, self.cap_ang, self.key_frac = float(capture_lateral_m), float(capture_ang_rad), float(capture_key_frac)
        self.seat_depth, self.seat_lat = float(capture_seat_depth_m), float(capture_seat_lateral_m)
        self.release_factor = float(release_factor)
        self.soft_solref = [float(v) for v in soft_solref]
        # the capture pulls the part home at a finite rate (the screws being driven), not in one jump: a hand that still
        # holds the part (or its pads opening) would otherwise pivot it (a support turned 6.6 deg between the pads as the
        # soft weld yanked its bore 6.5 mm home in 0.1 s, and jammed)
        self.pull_v, self.pull_w = float(pull_speed_m_s), float(pull_rate_rad_s)
        self.pull = {}
        key_len = {s: v["len_mm"] * 1e-3 for s, v in (kit.get("carve") or {}).get("keys", {}).items()}
        self.slots = []
        for s in kit["slots"]:
            x = dict(s)
            x["tol_pos"] = s["tol_pos_mm"] * 1e-3
            x["tol_ang"] = math.radians(s["tol_ang_deg"])
            x["axial_tol"] = s.get("axial_tol_mm", 0.0) * 1e-3
            x["pitch"] = math.radians(s.get("pitch_deg", 360.0))
            x["bc"] = model.body(s["carrier"]).id
            x["Fc"] = np.array(s.get("frame_in_carrier", np.eye(4)), float)
            x["key_len"] = key_len.get(s["id"], 0.004)
            x["cands"] = []
            for c in s["candidates"]:
                x["cands"].append({"unit": c["unit"], "bp": model.body(c["part"]).id, "Tpc": np.array(c["T_part_in_carrier"], float),
                                   "Fp": np.array(c.get("frame_in_part", np.eye(4)), float), "eq": model.equality(weld_name(s["id"], c["unit"])).id})
            if s["kind"] == "seat":
                ins = np.array(s.get("insertion_robot_design") or s.get("insertion_robot"), float)
                x["ins_c"] = ins / np.linalg.norm(ins)        # the carrier is the body (robot frame = its frame)
            self.slots.append(x)
        self.by_id = {x["id"]: x for x in self.slots}
        self.hard_solref = {c["eq"]: model.eq_solref[c["eq"]].copy() for x in self.slots for c in x["cands"]}
        self.state = {x["id"]: "WAITING" for x in self.slots}
        self.rec = {x["id"]: {"id": x["id"], "kind": x["kind"], "weight": x["weight"], "unit": None, "fastened": False, "achieved": False,
                              "t_s": None, "pos_err_mm": None, "ang_err_deg": None, "axial_err_mm": None, "key_k": None} for x in self.slots}
        self.hold = {x["id"]: 0 for x in self.slots}
        self.bound = {}                 # slot -> candidate index (captured)
        self.used = {}                  # unit -> slot that fastened or holds it
        self.events = []
        self.total_w = sum(x["weight"] for x in self.slots)
        self.limits_set = {}

    # ------------------------------------------------------------------ predicates
    def score(self):
        return sum(x["weight"] for x in self.slots if self.rec[x["id"]]["achieved"]) / self.total_w

    def _vel(self, b):
        import mujoco
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
        return v

    def evaluate(self, x, c):
        d = self.d
        Tc, Tp = body_pose(d, x["bc"]), body_pose(d, c["bp"])
        if x["kind"] == "seat":
            # at the seat's frame on the part (the support's bore, `frame_in_part`), not at the body's origin: a tilt of
            # the bracket must not count as a shift of its far-away frame origin
            A, B = Tc @ c["Tpc"] @ c["Fp"], Tp @ c["Fp"]
            E = _inv(A) @ B
            dw = B[:3, 3] - A[:3, 3]
            ins_w = Tc[:3, :3] @ x["ins_c"]
            axial = float(dw @ ins_w)                        # < 0: short of home along the insertion
            r = {"pos_err": float(np.linalg.norm(E[:3, 3])), "ang_err": rotation_angle(E[:3, :3]), "axial": axial,
                 "lateral": float(np.linalg.norm(dw - axial * ins_w))}
            r["holds"] = r["pos_err"] <= x["tol_pos"] and r["ang_err"] <= x["tol_ang"]
            r["design_ok"] = True
        else:
            A, B = Tc @ x["Fc"], Tp @ c["Fp"]
            za = A[:3, 2]
            ang = math.acos(max(-1.0, min(1.0, float(za @ B[:3, 2]))))
            dv = B[:3, 3] - A[:3, 3]
            axial = float(dv @ za)                           # the key's root past the pulley face: 0 at home, < 0 outside
            line = float(np.linalg.norm(dv - axial * za))
            qe = quat_wxyz((_inv(A) @ B)[:3, :3])
            clock = wrap(2.0 * math.atan2(float(qe[3]), float(qe[0])))
            k = int(round(clock / x["pitch"]))
            res = wrap(clock - k * x["pitch"])
            nk = int(round(2 * math.pi / x["pitch"]))
            r = {"pos_err": line, "ang_err": ang, "axial": axial, "clock": clock, "k": k % nk, "clock_res": res,
                 "holds": ang <= x["tol_ang"] and line <= x["tol_pos"] and abs(axial) <= x["axial_tol"] and abs(res) <= x["tol_ang"]}
            r["design_ok"] = bool(x.get("any_key")) or (k - int(x.get("design_k", 0))) % nk == 0
        if r["holds"]:
            vc, vp = self._vel(x["bc"]), self._vel(c["bp"])
            r["rel_speed"] = float(np.linalg.norm(vp[3:] - vc[3:]))
            r["rel_ang_speed"] = float(np.linalg.norm(vp[:3] - vc[:3]))
        return r

    def _inside(self, x, r, f=1.0):
        if x["kind"] == "seat":
            return r["ang_err"] <= self.cap_ang * f and r["lateral"] <= self.seat_lat * f and -self.seat_depth * f <= r["axial"] <= self.seat_lat * f
        # the key has entered the socket (at least `key_frac` of its height) on the axis and on a key orientation; once
        # captured (f > 1) it is released only when the key is out of the socket by 2 mm
        lo = -x["key_len"] * (1.0 - self.key_frac) if f <= 1.0 else -(x["key_len"] + 0.002)
        return (r["pos_err"] <= self.cap_lat * f and r["ang_err"] <= self.cap_ang * f and abs(r["clock_res"]) <= self.cap_ang * f
                and lo <= r["axial"] <= 0.001 * f)

    # ------------------------------------------------------------------ welds
    def _target(self, x, c, r):
        """the unit's pose in the carrier's frame: a seat at its design, a key home on the nearest key orientation"""
        if x["kind"] == "seat":
            return c["Tpc"]
        return x["Fc"] @ _rotz(r["k"] * x["pitch"]) @ _inv(c["Fp"])

    def _pull_target(self, x, c, r, sim_time):
        """the soft weld's target: from the pose the part had at its capture towards home at <= pull_speed at the weld
        point and <= pull_rate in angle"""
        from scipy.spatial.transform import Rotation, Slerp
        sid = x["id"]
        home = self._target(x, c, r)
        if sid not in self.pull or self.pull[sid]["unit"] != c["unit"]:
            T0 = _inv(body_pose(self.d, x["bc"])) @ body_pose(self.d, c["bp"])        # the part in the carrier now
            a = c["Fp"][:3, 3]
            dist = float(np.linalg.norm((T0[:3, :3] @ a + T0[:3, 3]) - (home[:3, :3] @ a + home[:3, 3])))
            ang = rotation_angle(T0[:3, :3].T @ home[:3, :3])
            dur = max(dist / self.pull_v, ang / self.pull_w, 1e-3)
            self.pull[sid] = {"unit": c["unit"], "T0": T0, "t0": float(sim_time), "dur": dur}
        p = self.pull[sid]
        s_ = min(1.0, (float(sim_time) - p["t0"]) / p["dur"])
        if s_ >= 1.0:
            return home
        T0 = p["T0"]
        a = c["Fp"][:3, 3]
        R = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], home[:3, :3]]))([s_]).as_matrix()[0]
        # the weld point moves on the straight line, the orientation slerps
        pa = (1 - s_) * (T0[:3, :3] @ a + T0[:3, 3]) + s_ * (home[:3, :3] @ a + home[:3, 3])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pa - R @ a
        return T

    def _weld(self, c, rel, soft):
        e = c["eq"]
        # the weld point at the slot's frame on the part (the key's root, the support's bore), not at the part's body
        # origin (0.18 m from a support's bracket): a soft weld then pulls the feature home about the feature. MuJoCo
        # at run time: body2's point = the anchor (data[0:3], body2 frame), body1's point = data[3:6] as given (body1
        # frame), so data[3:6] is the anchor's home in the carrier's frame (checked: zero residual at the target,
        # none from a turn about the anchor)
        a2 = c["Fp"][:3, 3]
        self.m.eq_data[e, 0:3] = a2
        self.m.eq_data[e, 3:6] = rel[:3, 3] + rel[:3, :3] @ a2
        self.m.eq_data[e, 6:10] = quat_wxyz(rel[:3, :3])
        self.m.eq_data[e, 10] = 1.0
        self.m.eq_solref[e] = self.soft_solref if soft else self.hard_solref[e]
        self.d.eq_active[e] = 1

    def _release(self, x, ci):
        self.pull.pop(x["id"], None)
        c = x["cands"][ci]
        self.d.eq_active[c["eq"]] = 0
        self.m.eq_solref[c["eq"]] = self.hard_solref[c["eq"]]
        self.bound.pop(x["id"], None)
        if self.used.get(c["unit"]) == x["id"]:
            self.used.pop(c["unit"], None)

    # ------------------------------------------------------------------ joints
    def _haa_limits(self, x, c):
        """the HAA hinge's range when its shoulder fastens: the product's range mapped through the as-built angle"""
        rng = x.get("haa_range_deg")
        if not rng:
            return
        m, d = self.m, self.d
        hinge = [j for j in range(m.njnt) if m.jnt_bodyid[j] == x["bc"]]
        if not hinge:
            return
        j = hinge[0]
        a = m.jnt_qposadr[j]
        th0 = float(d.qpos[a])
        # the as-built HAA angle: the shoulder's design orientation is the rotor's zero with the key home (k = 0)
        q0 = th0
        lo, hi = math.radians(rng[0]), math.radians(rng[1])
        m.jnt_range[j] = [min(th0 + lo - q0, th0), max(th0 + hi - q0, th0)]
        m.jnt_limited[j] = 1
        self.limits_set[x["id"]] = tuple(m.jnt_range[j])

    # ------------------------------------------------------------------ per control tick
    def tick(self, sim_time):
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
            ci = self.bound.get(sid)
            if ci is not None:
                c = x["cands"][ci]
                r = self.evaluate(x, c)
                if not self._inside(x, r, self.release_factor):
                    self._release(x, ci)
                    self.events.append((sim_time, "released from capture %s (%s)" % (sid, c["unit"])))
                    ci = None
            if ci is None:
                best = None
                for i, c in enumerate(x["cands"]):
                    if c["unit"] in self.used:
                        continue
                    r = self.evaluate(x, c)
                    if self._inside(x, r):
                        if best is None or r["pos_err"] < best[1]["pos_err"]:
                            best = (i, r)
                if best is None:
                    self.state[sid] = "FREE"
                    self.hold[sid] = 0
                    continue
                ci, r = best
                c = x["cands"][ci]
                self.bound[sid] = ci
                self.used[c["unit"]] = sid
                self.events.append((sim_time, "ENGAGED %s (%s)" % (sid, c["unit"])))
            self._weld(c, self._pull_target(x, c, r, sim_time), soft=True)
            still = r["holds"] and r.get("rel_speed", 1.0) < self.rel_speed_max and r.get("rel_ang_speed", 1.0) < self.rest_ang
            if not still:
                self.state[sid] = "ENGAGED"
                self.hold[sid] = 0
                continue
            self.hold[sid] += 1
            self.state[sid] = "HOLDING"
            if self.hold[sid] >= self.hold_ticks:
                self._lock(x, ci, r, sim_time)

    def _lock(self, x, ci, r, sim_time):
        import mujoco
        sid = x["id"]
        self.pull.pop(sid, None)
        c = x["cands"][ci]
        st = self.rec[sid]
        st.update({"fastened": True, "t_s": float(sim_time), "unit": c["unit"], "pos_err_mm": r["pos_err"] * 1e3, "ang_err_deg": math.degrees(r["ang_err"]),
                   "axial_err_mm": r.get("axial", 0.0) * 1e3, "achieved": bool(r["design_ok"])})
        if x["kind"] == "keyed":
            st["key_k"] = int(r["k"])
        self._weld(c, self._target(x, c, r), soft=False)
        self.bound.pop(sid, None)
        self.used[c["unit"]] = sid
        if x["kind"] == "keyed" and x.get("haa_range_deg"):
            mujoco.mj_kinematics(self.m, self.d)
            self._haa_limits(x, c)
        self.state[sid] = "LOCKED" if st["achieved"] else "LOCKED_OFF"
        self.events.append((sim_time, ("LOCKED %s (%s)" % (sid, c["unit"])) if st["achieved"] else
                            "LOCKED_OFF %s (%s, key orientation %d, design %d)" % (sid, c["unit"], r["k"], x.get("design_k", 0))))

    def record(self):
        return [dict(self.rec[x["id"]]) for x in self.slots]
