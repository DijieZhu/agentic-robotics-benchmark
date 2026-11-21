"""Mate predicates, the latch rule and the snapped welds (CHAIN_PLAN 2.5 and 4.3).

* `sleeve_seat` / `part_seat`: full 6-DoF pose of part_b's frame_b in part_a's frame_a within
  tol_pos / tol_ang. Servos of the same SKU are interchangeable: the pocket accepts any servo with
  the same catalogue id that is not yet welded elsewhere (the physical servo is recorded).
* `horn_mate`: evaluated only when every `requires` seat is achieved, against the MEAN axis of the
  seated horns (one servo: that horn): axis angle <= tol_ang (same direction), axis line distance
  <= tol_pos, axial offset <= axial_tol, yaw about the axis within tol_ang of the design modulo
  90 deg (quadrant recorded).
* latch: predicate true for `hold_ticks` consecutive ticks with relative speed < rel_speed_max
  (the part may be held) -> the grader activates the weld(s) SNAPPED to the designed relative pose
  (rotated by the recorded quadrant) and records the pre-snap error. Welds are grader-owned; the
  agent has no call that touches them.
* horn fastening "at rest" (`horn_at_rest=True`, the information tier since its task version 3; default off, so every
  earlier consumer is unchanged): a horn mate is FASTENED wherever the link rests on the horn -- axis angle / axis line /
  axial offset within tolerance and the link moving slower than rel_speed_max AND turning slower than rest_ang_speed_max
  relative to the servo for `hold_ticks` consecutive ticks -- at whatever rotation about the horn axis it has (the weld
  keeps that rotation; only the axis and the axial offset are snapped). The mate is ACHIEVED only when that rotation is
  within tol_ang of the design (graded completion): a fork fastened off the design is welded (it cannot be mounted
  again) but counts nowhere (sticky fraction, success, as-built pose, functional check). The latch therefore carries no
  information about the design rotation: nothing distinguishes a fork resting at the design from one resting 20 deg off.
Everything reads MjData only (F1).
"""

import math

import numpy as np

__all__ = ["MateTracker", "rotation_angle", "quat_wxyz_from_mat", "mat_from_quat_wxyz", "body_pose"]


def rotation_angle(R):
    c = (np.trace(R) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def mat_from_quat_wxyz(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_wxyz_from_mat(R):
    from .layout import quat_from_rotmat
    return quat_from_rotmat(R)


def body_pose(data, body_id):
    T = np.eye(4)
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    T[:3, 3] = data.xpos[body_id]
    return T


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


class MateTracker(object):
    """Per-episode predicate evaluation + latch + snapped welds for the mates in scope."""

    def __init__(self, model, meta, hold_ticks=50, rel_speed_max=0.005, tick_s=0.01, tolerance_scale=1.0, horn_at_rest=False,
                 rest_ang_speed_max=math.radians(2.0)):
        import mujoco
        self.mujoco = mujoco
        self.model = model
        self.meta = meta
        # the mating graph's tolerances (tol_pos, tol_ang, axial_tol) scaled by the rung's `latch.tolerance_scale`
        self.tolerance_scale = float(tolerance_scale)
        self.mates = [dict(m, **{k: m[k] * self.tolerance_scale for k in ("tol_pos", "tol_ang", "axial_tol") if k in m}) for m in meta["mates"]]
        self.by_id = {m["id"]: m for m in self.mates}
        self.hold_ticks = int(hold_ticks)
        self.rel_speed_max = float(rel_speed_max)
        self.horn_at_rest = bool(horn_at_rest)
        self.rest_ang_speed_max = float(rest_ang_speed_max)
        self.tick_s = tick_s
        self.body = {}
        for name in [meta["root"]] + list(meta["loose"]):
            self.body[name] = model.body(name).id
        for s in meta["servos"]:
            self.body["horn_" + s] = model.body("horn_" + s).id
        self.horn_joint = {s: model.joint("hj_" + s).id for s in meta["servos"]}
        self.weld_id = {k: model.equality(v).id for k, v in meta["welds"].items()}
        self.catalogue = {s: meta["parts"][s].get("catalogue_id") for s in meta["servos"]}
        self.joint_axis_in_child = meta.get("joint_axis_in_child", {})
        self.state = {}
        for m in self.mates:
            self.state[m["id"]] = {"id": m["id"], "kind": m["kind"], "achieved": False, "fastened": False, "t_s": None, "hold": 0,
                                   "pos_err_mm": None, "ang_err_deg": None, "axial_err_mm": None, "yaw_err_deg": None,
                                   "yaw_quadrant_deg": None, "servo": None, "servos": None, "horn_angles_at_snap_rad": None,
                                   "joint_zero_offset_rad": None, "servo_drive": None, "weight": m["weight"], "evaluated": False}
        self.servo_used = {}      # physical servo -> seat mate id
        self.seat_servo = {}      # seat mate id -> physical servo
        self.events = []

    # ------------------------------------------------------------------ helpers
    def _vel(self, data, body_id):
        v = np.zeros(6)
        self.mujoco.mj_objectVelocity(self.model, data, self.mujoco.mjtObj.mjOBJ_BODY, body_id, v, 0)
        return v[3:6]   # linear (world) — mj_objectVelocity returns [rot, lin]

    def _angvel(self, data, body_id):
        v = np.zeros(6)
        self.mujoco.mj_objectVelocity(self.model, data, self.mujoco.mjtObj.mjOBJ_BODY, body_id, v, 0)
        return v[0:3]   # angular (world)

    def _seat_candidates(self, m):
        target = m["part_b"]
        if target in self.meta["servos"]:
            return [s for s in self.meta["servos"] if self.catalogue[s] == self.catalogue[target]
                    and (s not in self.servo_used or self.servo_used[s] == m["id"])]
        return [target]

    def _eval_seat(self, data, m):
        T_a = body_pose(data, self.body[m["part_a"]]) @ m["F_a"]
        best = None
        for c in self._seat_candidates(m):
            T_b = body_pose(data, self.body[c]) @ m["F_b"]
            E = _inv(T_a) @ T_b
            pe = float(np.linalg.norm(E[:3, 3]))
            ae = rotation_angle(E[:3, :3])
            if best is None or pe < best["pos_err"]:
                best = {"servo": c, "pos_err": pe, "ang_err": ae}
        if best is None:
            return None
        best["holds"] = best["pos_err"] <= m["tol_pos"] and best["ang_err"] <= m["tol_ang"]
        if best["holds"]:
            va = self._vel(data, self.body[m["part_a"]]) if m["part_a"] != self.meta["root"] else np.zeros(3)
            vb = self._vel(data, self.body[best["servo"]])
            best["rel_speed"] = float(np.linalg.norm(vb - va))
        return best

    def _horn_servos(self, m):
        """Physical servos seated in the pockets this horn mate requires (None if any is missing)."""
        out = []
        for s in m["servos"]:
            seat = "seat_" + s
            if seat not in self.seat_servo:
                return None
            out.append((s, self.seat_servo[seat]))
        for r in m["requires"]:
            if not self.state[r]["achieved"]:
                return None
        return out

    def _eval_horn(self, data, m):
        servos = self._horn_servos(m)
        if not servos:
            return None
        T_link = body_pose(data, self.body[m["part_b"]])
        hs, zs, os_, zl = [], [], [], []
        first = None
        for design_id, phys in servos:
            T_h = body_pose(data, self.body[phys]) @ m["F_a"]            # horn frame (servo-fixed)
            F_b = m["F_b_per_servo"].get(design_id, m["F_b"])
            T_lb = T_link @ F_b                                           # where the link says the horn should be
            if first is None:
                first = (design_id, phys, T_h, T_lb, F_b)
                zref, zlref = T_h[:3, 2], T_lb[:3, 2]
            zh = T_h[:3, 2] * (1.0 if T_h[:3, 2] @ zref > 0 else -1.0)
            zlb = T_lb[:3, 2] * (1.0 if T_lb[:3, 2] @ zlref > 0 else -1.0)
            hs.append(T_h[:3, 3]); zs.append(zh); os_.append(T_lb[:3, 3]); zl.append(zlb)
        h = np.mean(hs, axis=0); z = np.mean(zs, axis=0); z /= np.linalg.norm(z)
        o = np.mean(os_, axis=0); zlm = np.mean(zl, axis=0); zlm /= np.linalg.norm(zlm)
        ang = math.acos(max(-1.0, min(1.0, float(z @ zlm))))
        d = o - h
        axial = float(d @ z)
        line = float(np.linalg.norm(d - axial * z))
        # yaw about the first horn's axis: link frame_b relative to the horn frame
        E = _inv(first[2]) @ first[3]
        yaw = math.atan2(E[1, 0], E[0, 0])
        mod = m["yaw_modulo"]
        on_horn = ang <= m["tol_ang"] and line <= m["tol_pos"] and abs(axial) <= m["axial_tol"]
        if self.horn_at_rest:
            # fastened wherever it rests: the rotation about the axis is not part of the fastening predicate; it decides
            # only whether the fastened mate counts (graded completion, `design_yaw_ok`)
            k = 0
            yaw_res = yaw
            holds = on_horn
        else:
            k = int(round(yaw / mod))
            yaw_res = yaw - k * mod
            holds = on_horn and abs(yaw_res) <= m["tol_ang"]
        res = {"servos": servos, "pos_err": line, "ang_err": ang, "axial": axial, "yaw_res": yaw_res, "quadrant": k, "holds": holds, "first": first,
               "design_yaw_ok": abs(yaw_res) <= m["tol_ang"]}
        if holds:
            vb = self._vel(data, self.body[m["part_b"]])
            va = self._vel(data, self.body[first[1]])
            res["rel_speed"] = float(np.linalg.norm(vb - va))
            if self.horn_at_rest:
                wb = self._angvel(data, self.body[m["part_b"]])
                wa = self._angvel(data, self.body[first[1]])
                res["rel_ang_speed"] = float(np.linalg.norm(wb - wa))
        return res

    # ------------------------------------------------------------------ public
    def status(self):
        return {mid: bool(s["achieved"]) for mid, s in self.state.items()}

    def evaluate(self, data):
        """Raw predicate values per mate (for the record / practice output), no side effects."""
        out = {}
        for m in self.mates:
            if m["kind"] in ("sleeve_seat", "part_seat"):
                r = self._eval_seat(data, m)
            else:
                r = self._eval_horn(data, m)
            out[m["id"]] = r
        return out

    def tick(self, data, sim_time):
        """Advance the latch by one control tick; activate snapped welds; return events."""
        events = []
        for m in self.mates:
            st = self.state[m["id"]]
            if st["achieved"] or st["fastened"]:
                continue
            if m["kind"] in ("sleeve_seat", "part_seat"):
                r = self._eval_seat(data, m)
            else:
                r = self._eval_horn(data, m)
            st["evaluated"] = r is not None
            if r is None or not r["holds"] or r.get("rel_speed", 1.0) >= self.rel_speed_max or r.get("rel_ang_speed", 0.0) >= self.rest_ang_speed_max:
                st["hold"] = 0
                continue
            st["hold"] += 1
            if st["hold"] < self.hold_ticks:
                continue
            # latch (fastened); achieved = fastened at the design (always, unless a horn is fastened at rest off the design)
            st["fastened"] = True
            st["achieved"] = bool(r.get("design_yaw_ok", True)) if m["kind"] == "horn_mate" else True
            st["t_s"] = float(sim_time)
            st["pos_err_mm"] = r["pos_err"] * 1000.0
            st["ang_err_deg"] = math.degrees(r["ang_err"])
            if m["kind"] in ("sleeve_seat", "part_seat"):
                st["servo"] = r["servo"]
                if r["servo"] in self.meta["servos"]:
                    self.servo_used[r["servo"]] = m["id"]
                    self.seat_servo[m["id"]] = r["servo"]
                self._snap_seat(data, m, r["servo"])
            else:
                st["axial_err_mm"] = r["axial"] * 1000.0
                st["yaw_err_deg"] = math.degrees(r["yaw_res"])
                st["yaw_quadrant_deg"] = int(r["quadrant"] * round(math.degrees(m["yaw_modulo"]))) % 360
                st["servos"] = [phys for _d, phys in r["servos"]]
                st["servo"] = st["servos"][0]
                angles = {phys: float(data.qpos[self.model.jnt_qposadr[self.horn_joint[phys]]]) for _d, phys in r["servos"]}
                st["horn_angles_at_snap_rad"] = angles
                st["joint_zero_offset_rad"] = self._joint_zero_offset(m, r, angles)
                st["servo_drive"] = self._servo_drive(m, r, angles)
                self._snap_horn(data, m, r)
            events.append(("seated:" if st["achieved"] else "fastened_off_design:") + m["id"])
        self.events.extend(events)
        return events

    # ------------------------------------------------------------------ snapping
    def _activate(self, data, key, relpose):
        eq = self.weld_id[key]
        self.model.eq_data[eq, 0:3] = 0.0
        self.model.eq_data[eq, 3:6] = relpose[:3, 3]
        self.model.eq_data[eq, 6:10] = quat_wxyz_from_mat(relpose[:3, :3])
        self.model.eq_data[eq, 10] = 1.0
        data.eq_active[eq] = 1

    def _snap_seat(self, data, m, servo):
        rel = m["F_a"] @ _inv(m["F_b"])                     # part_b in part_a at the design
        self._activate(data, (m["part_a"], servo), rel)

    def _snap_horn(self, data, m, r):
        design_id, phys, T_h, T_lb, F_b = r["first"]
        # at rest: the weld keeps the rotation the link rests at (only the axis and the axial offset are snapped)
        T_link_target = T_h @ _rotz(self._psi(m, r)) @ _inv(F_b)
        for _d, p in r["servos"]:
            T_hb = body_pose(data, self.body["horn_" + p])
            self._activate(data, ("horn_" + p, m["part_b"]), _inv(T_hb) @ T_link_target)

    def _joint_zero_offset(self, m, r, angles):
        """Angle about the JOINT axis between the as-built link and the design when the (first) horn
        is at its own zero: sign x (quadrant - horn angle at snap); sign = horn axis . joint axis in the link."""
        design_id, phys, T_h, T_lb, F_b = r["first"]
        axis = self.joint_axis_in_child.get(m.get("joint"))
        if axis is None:
            sign = 1.0
        else:
            sign = 1.0 if float(np.asarray(axis, float) @ F_b[:3, 2]) >= 0 else -1.0
        psi_h = self._psi(m, r) - angles[phys]
        return float(sign * psi_h)

    def _psi(self, m, r):
        """The rotation about the (first) horn axis the weld keeps: the recorded quadrant x modulo (the snapped latch), or
        the resting rotation itself (horn fastening at rest)."""
        return float(r["yaw_res"]) if self.horn_at_rest else r["quadrant"] * m["yaw_modulo"]

    def _servo_drive(self, m, r, angles):
        """Per physical servo: sign = horn axis . joint axis (link frame), the hinge angle at the snap
        and the quadrant about ITS horn axis, so the functional check can drive
        theta_s(q) = theta_snap + sign_s * q - psi_h,s (q = design joint angle)."""
        design_first, phys_first, T_h, T_lb, F_b_first = r["first"]
        axis = self.joint_axis_in_child.get(m.get("joint"))
        out = {}
        sign_first = 1.0 if axis is None or float(np.asarray(axis, float) @ F_b_first[:3, 2]) >= 0 else -1.0
        psi_first = self._psi(m, r)
        for design_id, phys in r["servos"]:
            F_b = m["F_b_per_servo"].get(design_id, m["F_b"])
            sign = 1.0 if axis is None or float(np.asarray(axis, float) @ F_b[:3, 2]) >= 0 else -1.0
            out[phys] = {"sign": sign, "theta_snap": float(angles[phys]), "psi_h": float(psi_first * sign * sign_first), "design_id": design_id}
        return out

    def record(self):
        out = []
        for m in self.mates:
            st = dict(self.state[m["id"]])
            st.pop("hold", None)
            st.pop("evaluated", None)
            out.append(st)
        return out
