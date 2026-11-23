"""Coupling state machine of the hand assembly cell: one state per mate of the mating graph (a SLOT), each slot taking
whichever of its candidate units reaches it (the three finger modules are interchangeable). Written after the
tensegrity cell's coupling.py (slot x candidate, one weld per pair declared in advance, a fastened unit leaves every
other slot's candidate list, a light rail under the carrier that leaves the free directions free), with three kinds:

  module_slide  a digit's base in its pocket of the bottom shell, open above (finger channels; the thumb's wall).
                Finger: ENGAGED when the base is 8 mm or more inside the pocket but not in its last 3.0 mm (a base that
                first appears there, lowered from above, would land on the wall screws' ends), its centre line within
                the slot (0.5 mm a side), yaw and pitch within 3 deg, on the floor (gap <= 1.0 mm). Then the cell
                GUIDES, it does not carry: the base is welded to a rail whose six joints leave it free along the
                slide, up, in pitch and in roll, and hold its sideways offset and its yaw within the real play
                (+-0.195 mm, +-0.72 deg; joint limits), re-anchored to where the module is (the weld's target pulled
                from the pose at the capture to the rail at <= 10 mm/s and 10 deg/s). Nothing holds it up or down.
                RELEASED when the base leaves the first 7 mm of the pocket, lifts more than 2.0 mm off the floor, or
                pitches or yaws beyond 6 deg. Thumb: no guide at all (contact resolves its 1 mm of play); ENGAGED
                when within 10 mm of its wall but not in the last 3 mm, within its play along the wall, on the floor,
                within 3 deg. LOCK: engaged, the base's end face within 0.5 mm of the wall, at rest, the requires
                fastened, every joint with a mounted range inside it, tool access free -> a rigid weld at the design
                pose; the chain's joints stay free and get their mounted ranges (the shell's stops).
  part_seat     stack, top, cover, hub: ENGAGED when within 2.0 mm across the insertion axis, 5 deg, at most 3 mm above
                home. Once it moves slower than 50 mm/s and 30 deg/s relative to the carrier, the cell CENTRES it: a
                rail free along the insertion axis (it falls or is pushed; never moved along it by the cell), the
                weld's target pulled across and in angle at <= 10 mm/s and 10 deg/s. RELEASED beyond twice the window.
                LOCK: within 1.5 mm across and 3 deg, at most 0.5 mm above home, at rest and CARRIED BY ITS SEAT
                (the normal forces of its contacts with the parts it sits on, the fastened cluster and the fixture,
                projected on the insertion axis, carry at least part_seat.support_share_min of its weight along that
                axis; the gripper's contacts and friction on side faces do not count) for the hold ticks, the
                requires fastened, tool access free -> rigid weld at the design pose. The stack also seats turned
                180 deg about its screw pattern's centre: the same machine on that pose -> LOCKED_OFF.
  pressed_seat  the cap: ENGAGED when its rim is inside the opening (0.3 mm across, 3 deg, at most 7 mm above home);
                the rail holds it across and in tilt (welded to the rail with the lock weld's time constant and a
                harder impedance), its slide resists motion along the axis with the fit's friction force (frictionloss
                of the slide joint, solved as a hard dry friction): the cap stays where it is put (it gives back about
                5 um per N of the load it carried when let go) and the robot pushes it home. LOCK: at rest within
                0.5 mm of home, the requires fastened -> rigid weld at the design pose.

States: WAITING (a required mate not fastened) -> FREE -> ENGAGED -> HOLDING -> LOCKED / LOCKED_OFF.
TOOL ACCESS: a mate goes to HOLDING only while no other unit's collision geometry lies in any of its tool volumes
(capsules of the mating graph's segments, 2.5 mm diameter, on the base unit's body, no contact); its own unit, its
carriers and the robot are not counted.
`declare(spec, kit, base_body, unit_bodies)` adds the rails, the welds and the tool capsules to an MjSpec before
compiling; `HandCoupling(model, data, kit, base_body, unit_bodies)` runs the machine (`tick(sim_time)` once per
control tick, after the physics). `fastening_loads()` is the reading of rule R2 (b), `stop_loads()` of rule R3.
"""
import math

import numpy as np

STATES = ("WAITING", "FREE", "ENGAGED", "HOLDING", "LOCKED", "LOCKED_OFF")

DEFAULTS = {
    "hold_ticks": 50, "rest_mm_s": 5.0, "rest_deg_s": 2.0,
    # a finger's base is ENGAGED only down to floor_push_depth_mm + engage_below_margin_mm below its floor:
    # floor_push_depth_mm is how far below its seated height a base lying level on the channel's floor ledges gets
    # when it is pushed down with 25 N (the most a unit may carry for 0.2 s) anywhere along the last 4 mm of its
    # slide, measured on the floor as built (0.19-0.22 mm on the three lanes; up to 0.19 mm of it the contact's give),
    # rounded up
    "module_slide": {"release_inside_mm": 7.0, "thumb_engage_out_mm": 10.0, "thumb_release_out_mm": 12.0,
                     "lift_release_mm": 2.0, "tilt_release_deg": 6.0, "last_mm": 3.0,
                     "floor_push_depth_mm": 0.23, "engage_below_margin_mm": 0.05},
    # a seated part is fastened only while its seat carries it: over the hold ticks the normal forces of its contacts
    # with the parts it sits on (the fastened cluster's bodies and the fixture), projected on the insertion axis,
    # carry at least this share of its weight along that axis (the gripper's contacts and friction do not count)
    "part_seat": {"centre_below_mm_s": 50.0, "centre_below_deg_s": 30.0, "release_factor": 2.0,
                  "support_share_min": 0.5},
    # the press fit: its friction (the rail's slide) and the rail weld solved hard. With MuJoCo's default friction
    # impedance the 1 g rail's friction row is soft in the main solve (noslip then holds the rail), and the cap hung
    # on its rail like a spring of about 2.7 N/mm: pressed by the fit's 3 N it sat 1.1 mm below the rail and rose
    # back to it when let go; with these about 190 N/mm (0.015 mm back after a 2.8 N press)
    "pressed_seat": {"fit_friction_N": 3.0, "release_factor": 2.0, "friction_solref": [0.002, 1.0],
                     "friction_solimp": [0.999, 0.9999, 0.001, 0.5, 2.0],
                     "weld_solimp": [0.999, 0.9999, 0.001, 0.5, 2.0]},
    "pull_mm_s": 10.0, "pull_deg_s": 10.0,
    "soft_solref": [0.01, 1.0], "weld_solref": [0.002, 1.0], "weld_solimp": [0.99, 0.999, 0.001, 0.5, 2.0],
    "rail": {"mass_kg": 1.0e-3, "inertia_kg_m2": 1.0e-9, "armature_slide": 1.0e-4, "armature_hinge": 1.0e-8,
             "damping_slide": 0.5, "damping_hinge": 1.0e-4, "limit_solref": [0.005, 1.0]},
    "tool_radius_mm": 1.25,
}


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


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


def _mm(T):
    T = np.array(T, float).copy()
    T[:3, 3] *= 1e-3
    return T


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def zxy_angles(R):
    """R = Rz(yaw) Rx(pitch) Ry(roll) -> (yaw, pitch, roll)"""
    pitch = math.asin(max(-1.0, min(1.0, R[2, 1])))
    yaw = math.atan2(-R[0, 1], R[1, 1])
    roll = math.atan2(-R[2, 0], R[2, 2])
    return yaw, pitch, roll


def rail_name(key):
    return "rail_%s" % key


def weld_name(kind, key, unit):
    return "%s_%s__%s" % (kind, key, unit)


def _cfg(cfg):
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k].update(v)
        else:
            c[k] = v
    return c


def is_thumb(mt):
    return mt["b_kind"] == "thumb_module"


def declare(spec, kit, base_body, unit_bodies, cfg=None):
    """the rails, one inactive weld per (slot, candidate) to the rail (`rw_`) and to the base unit (`lw_`, the lock),
    and the tool capsules. base_body: the base unit's body name (its frame = the palm frame); unit_bodies: {unit:
    root link body name}."""
    import mujoco
    c = _cfg(cfg)
    rl = c["rail"]
    bb = spec.body(base_body)
    for mt in kit.mates():
        key = mt["key"]
        Tdes = _mm(mt["design_pose_b_in_a"])
        cands = kit.units_of_kind(mt["b_kind"])
        rail = None
        if mt["kind"] == "module_slide" and not is_thumb(mt):
            g = mt["windows_proposal"]["guide"]
            rail = bb.add_body(name=rail_name(key), pos=Tdes[:3, 3].tolist(), quat=quat_wxyz(Tdes[:3, :3]).tolist(),
                               gravcomp=1.0)
            specs = [("x", mujoco.mjtJoint.mjJNT_SLIDE, [1, 0, 0], g["lateral_play_mm"] * 1e-3),
                     ("y", mujoco.mjtJoint.mjJNT_SLIDE, [0, 1, 0], None),
                     ("z", mujoco.mjtJoint.mjJNT_SLIDE, [0, 0, 1], None),
                     ("yaw", mujoco.mjtJoint.mjJNT_HINGE, [0, 0, 1], math.radians(g["yaw_play_deg"])),
                     ("pitch", mujoco.mjtJoint.mjJNT_HINGE, [1, 0, 0], None),
                     ("roll", mujoco.mjtJoint.mjJNT_HINGE, [0, 1, 0], None)]
            for nm, jt, ax, lim in specs:
                slide = jt == mujoco.mjtJoint.mjJNT_SLIDE
                j = rail.add_joint(name="%s_%s" % (rail_name(key), nm), type=jt, axis=ax,
                                   damping=rl["damping_slide"] if slide else rl["damping_hinge"],
                                   armature=rl["armature_slide"] if slide else rl["armature_hinge"])
                if lim is not None:
                    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
                    j.range = [-lim, lim]
                    j.solref_limit = list(rl["limit_solref"])
                else:
                    j.limited = mujoco.mjtLimited.mjLIMITED_FALSE
        elif mt["kind"] in ("part_seat", "pressed_seat"):
            rail = bb.add_body(name=rail_name(key), pos=Tdes[:3, 3].tolist(), quat=quat_wxyz(Tdes[:3, :3]).tolist(),
                               gravcomp=1.0)
            ins = np.array(mt["insertion_axis_in_a"], float)
            ax = -(Tdes[:3, :3].T @ (ins / np.linalg.norm(ins)))
            fr = float(c["pressed_seat"]["fit_friction_N"]) if mt["kind"] == "pressed_seat" else 0.0
            j = rail.add_joint(name="%s_s" % rail_name(key), type=mujoco.mjtJoint.mjJNT_SLIDE, axis=ax.tolist(),
                               damping=rl["damping_slide"], armature=rl["armature_slide"], frictionloss=fr)
            j.limited = mujoco.mjtLimited.mjLIMITED_FALSE
            if mt["kind"] == "pressed_seat":
                j.solref_friction = list(c["pressed_seat"]["friction_solref"])
                j.solimp_friction = list(c["pressed_seat"]["friction_solimp"])
        if rail is not None:
            rail.mass, rail.inertia, rail.explicitinertial = rl["mass_kg"], [rl["inertia_kg_m2"]] * 3, True
            rw_imp = c["pressed_seat"]["weld_solimp"] if mt["kind"] == "pressed_seat" else c["weld_solimp"]
            for u in cands:
                eq = spec.add_equality(name=weld_name("rw", key, u), type=mujoco.mjtEq.mjEQ_WELD,
                                       objtype=mujoco.mjtObj.mjOBJ_BODY, name1=rail.name, name2=unit_bodies[u])
                eq.active, eq.solref, eq.solimp = False, list(c["soft_solref"]), list(rw_imp)
        for u in cands:
            eq = spec.add_equality(name=weld_name("lw", key, u), type=mujoco.mjtEq.mjEQ_WELD,
                                   objtype=mujoco.mjtObj.mjOBJ_BODY, name1=base_body, name2=unit_bodies[u])
            eq.active, eq.solref, eq.solimp = False, list(c["weld_solref"]), list(c["weld_solimp"])
        r = float(c["tool_radius_mm"]) * 1e-3
        for k, tv in enumerate(mt.get("tool_volumes", [])):
            for s, sg in enumerate(tv["segments"]):
                p0, p1 = np.array(sg["p0_mm"]) * 1e-3, np.array(sg["p1_mm"]) * 1e-3
                bb.add_geom(name="tool_%s_%02d_%d" % (key, k, s), type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                            fromto=np.concatenate([p0, p1]).tolist(), size=[r, 0, 0], contype=0, conaffinity=0,
                            group=5, rgba=[0, 0, 0, 0], mass=0.0)


class HandCoupling(object):
    def __init__(self, model, data, kit, base_body, unit_bodies, unit_geoms, unit_joints, cfg=None):
        """unit_bodies: {unit: root link body name}; unit_geoms: {unit: [collision geom names]}; unit_joints: {unit:
        {joint: joint name}} (the product joints, for the mounted ranges and rule R3)"""
        import mujoco
        self.mj, self.m, self.d, self.kit = mujoco, model, data, kit
        c = self.c = _cfg(cfg)
        self.hold_ticks = int(c["hold_ticks"])
        self.rest_v, self.rest_w = c["rest_mm_s"] * 1e-3, math.radians(c["rest_deg_s"])
        self.pull_v, self.pull_w = c["pull_mm_s"] * 1e-3, math.radians(c["pull_deg_s"])
        self.soft, self.hard = list(c["soft_solref"]), list(c["weld_solref"])
        self.bc = model.body(base_body).id
        self.unit_body = {u: model.body(b).id for u, b in unit_bodies.items()}
        self.unit_geoms = {u: [model.geom(g).id for g in gs] for u, gs in unit_geoms.items()}
        self.unit_joints = {u: {j: model.joint(n).id for j, n in js.items()} for u, js in unit_joints.items()}
        self.body_unit = {}
        for u, gs in self.unit_geoms.items():
            for g in gs:
                self.body_unit[int(model.geom_bodyid[g])] = u
        # every link body of each unit (its tree under the world), and the fixture that carries the palm base
        self.unit_link_bodies = {u: set(int(b) for b in range(model.nbody) if int(model.body_rootid[b]) == int(rb))
                                 for u, rb in self.unit_body.items()}
        self.unit_mass = {u: float(sum(model.body_mass[b] for b in bs)) for u, bs in self.unit_link_bodies.items()}
        self.support_static = {int(self.bc), int(model.body_parentid[self.bc])}
        self.support = {}           # slot -> the support share at its last tick (part_seat)
        self.slots = []
        for mt in kit.mates():
            x = {"key": mt["key"], "id": mt["id"], "kind": mt["kind"], "weight": float(mt["weight"]),
                 "requires": list(mt.get("requires", [])), "thumb": is_thumb(mt), "b_kind": mt["b_kind"],
                 "carriers": list(mt.get("carriers", [])), "w": mt["windows_proposal"]}
            x["Tdes"] = _mm(mt["design_pose_b_in_a"])
            ins = np.array(mt["insertion_axis_in_a"], float)
            x["ins"] = ins / np.linalg.norm(ins)
            x["depth"] = float(mt["insertion_depth_mm"]) * 1e-3
            x["targets"] = [("design", np.eye(4))]
            for off in mt.get("locked_off", []) or []:
                x["targets"].append((off["what"], _inv(x["Tdes"]) @ _mm(off["pose_b_in_a"])))
            if mt["kind"] in ("part_seat", "pressed_seat") or (mt["kind"] == "module_slide" and not x["thumb"]):
                x["rail"] = model.body(rail_name(mt["key"])).id
                jn = [model.joint(j).id for j in range(model.njnt)
                      if (model.joint(j).name or "").startswith(rail_name(mt["key"]) + "_")]
                x["rail_q"] = [int(model.jnt_qposadr[j]) for j in jn]
                x["rail_v"] = [int(model.jnt_dofadr[j]) for j in jn]
            x["tools"] = [model.geom(g).id for g in range(model.ngeom)
                          if (model.geom(g).name or "").startswith("tool_%s_" % mt["key"])]
            x["cands"] = []
            for u in kit.units_of_kind(mt["b_kind"]):
                cd = {"unit": u, "bp": self.unit_body[u], "lw": model.equality(weld_name("lw", mt["key"], u)).id}
                if "rail" in x:
                    cd["rw"] = model.equality(weld_name("rw", mt["key"], u)).id
                x["cands"].append(cd)
            self.slots.append(x)
        self.by_id = {x["id"]: x for x in self.slots}
        self.by_key = {x["key"]: x for x in self.slots}
        self.state = {x["key"]: "WAITING" for x in self.slots}
        self.rec = {x["key"]: {"key": x["key"], "id": x["id"], "kind": x["kind"], "weight": x["weight"], "unit": None,
                               "fastened": False, "achieved": False, "t_s": None, "target": None, "pos_err_mm": None,
                               "ang_err_deg": None, "why_off": None, "capture": None, "blocked_by": None,
                               "t_engaged_s": None} for x in self.slots}
        self.hold = {x["key"]: 0 for x in self.slots}
        self.bound = {}             # slot -> (candidate index, target index)
        self.pull = {}
        self.used = {}              # unit -> slot that holds or fastened it
        self.events = []
        self.total_w = sum(x["weight"] for x in self.slots)
        self.tool_blocks = {}       # slot -> set of units seen in its tool volumes while it was otherwise ready
        self.jmodel = kit.joint_model
        self._init_clusters()

    # ------------------------------------------------------------------ contact clusters
    CLUSTER_BIT = 2

    def _init_clusters(self):
        """one collision bit for the rigidly fastened group: the palm base from the start, every unit's welded root
        link from its lock on (a digit's base servo; a palm part as a whole). Its geoms get contype CLUSTER_BIT and
        conaffinity 1; every other colliding geom of the cell keeps contype 1 and gets conaffinity 1 | CLUSTER_BIT. So
        the group's members no longer collide with one another (the lock welds hold them; their contacts were the
        welds' preload, which rule R2 never counted), and everything else still collides with them and with itself."""
        m = self.m
        self.cluster_geoms = set()
        for g in range(m.ngeom):
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                m.geom_conaffinity[g] = int(m.geom_conaffinity[g]) | self.CLUSTER_BIT
        self._join_cluster(self.bc)

    def _join_cluster(self, body):
        m = self.m
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) == int(body) and (m.geom_contype[g] or m.geom_conaffinity[g]):
                m.geom_contype[g] = self.CLUSTER_BIT
                m.geom_conaffinity[g] = 1
                self.cluster_geoms.add(g)

    # ------------------------------------------------------------------ helpers
    def score(self):
        return sum(x["weight"] for x in self.slots if self.rec[x["key"]]["achieved"]) / self.total_w

    def _vel(self, b):
        v = np.zeros(6)
        self.mj.mj_objectVelocity(self.m, self.d, self.mj.mjtObj.mjOBJ_BODY, b, v, 0)
        return v

    def _rel_speed(self, b):
        vc, vp = self._vel(self.bc), self._vel(b)
        return float(np.linalg.norm(vp[3:] - vc[3:])), float(np.linalg.norm(vp[:3] - vc[:3]))

    def rel_pose(self, x, c, ti=0):
        """the unit's root link in the target frame (design pose, or a LOCKED_OFF pose) of its slot"""
        T0 = body_pose(self.d, self.bc) @ x["Tdes"] @ x["targets"][ti][1]
        return _inv(T0) @ body_pose(self.d, c["bp"])

    def _requires_ok(self, x):
        return all(self.rec[self.by_id[q]["key"]]["fastened"] for q in x["requires"])

    def _weld(self, e, rel, soft):
        """weld e (body1 = rail or base unit, body2 = unit) holding the unit at `rel` (its pose in body1's frame)"""
        anchor = np.zeros(3)
        self.m.eq_data[e, 0:3] = anchor
        self.m.eq_data[e, 3:6] = rel[:3, 3] + rel[:3, :3] @ anchor
        self.m.eq_data[e, 6:10] = quat_wxyz(rel[:3, :3])
        self.m.eq_data[e, 10] = 1.0
        self.m.eq_solref[e] = self.soft if soft else self.hard
        self.d.eq_active[e] = 1

    def _pull(self, key, T0, home, sim_time):
        from scipy.spatial.transform import Rotation, Slerp
        p = self.pull.get(key)
        if p is None:
            dist = float(np.linalg.norm(T0[:3, 3] - home[:3, 3]))
            ang = rotation_angle(T0[:3, :3].T @ home[:3, :3])
            p = self.pull[key] = {"T0": T0, "t0": float(sim_time), "dur": max(dist / self.pull_v, ang / self.pull_w, 1e-3),
                                  "dist": dist, "ang": ang}
        s_ = min(1.0, (float(sim_time) - p["t0"]) / p["dur"])
        if s_ >= 1.0:
            return home, True
        R = Slerp([0, 1], Rotation.from_matrix([p["T0"][:3, :3], home[:3, :3]]))([s_]).as_matrix()[0]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (1 - s_) * p["T0"][:3, 3] + s_ * home[:3, 3]
        return T, False

    def _release(self, x, why):
        ci, _ti = self.bound.pop(x["key"])
        c = x["cands"][ci]
        for k in ("rw", "lw"):
            if k in c:
                self.d.eq_active[c[k]] = 0
        self.pull.pop(x["key"], None)
        if self.used.get(c["unit"]) == x["key"]:
            self.used.pop(c["unit"], None)
        self.events.append((self.t, "released %s (%s): %s" % (x["key"], c["unit"], why)))

    def _set_rail(self, x, E):
        """the rail's joints to the unit's pose E in the rail's zero frame (module_slide: x y z yaw pitch roll, x and
        yaw clamped to the play; seat: the slide along the axis) -> the rail frame's pose in the target frame"""
        m, d = self.m, self.d
        qa = x["rail_q"]
        if x["kind"] == "module_slide":
            g = x["w"]["guide"]
            lat, yawp = g["lateral_play_mm"] * 1e-3, math.radians(g["yaw_play_deg"])
            yaw, pitch, roll = zxy_angles(E[:3, :3])
            q = [max(-lat, min(lat, E[0, 3])), E[1, 3], E[2, 3], max(-yawp, min(yawp, yaw)), pitch, roll]
            for a_, v_ in zip(qa, q):
                d.qpos[a_] = v_
            R = _rz(q[3]) @ _rx(q[4]) @ _ry(q[5])
            G = np.eye(4)
            G[:3, :3] = R
            G[:3, 3] = q[:3]
        else:
            ax = np.array(m.jnt_axis[[j for j in range(m.njnt) if int(m.jnt_qposadr[j]) == qa[0]][0]], float)
            s = float(E[:3, 3] @ ax)
            d.qpos[qa[0]] = s
            G = np.eye(4)
            G[:3, 3] = s * ax
        for v_ in x["rail_v"]:
            d.qvel[v_] = 0.0
        self.mj.mj_kinematics(m, d)
        return G

    # ------------------------------------------------------------------ tool access
    def tool_blockers(self, x, own_unit):
        """units whose collision geometry lies in one of the slot's tool volumes (own unit and carriers excepted)"""
        if not x["tools"]:
            return []
        excl = {own_unit} | set(self._carrier_units(x))
        out = set()
        fromto = np.zeros(6)
        for u, gs in self.unit_geoms.items():
            if u in excl or u in out:
                continue
            for g in gs:
                for tg in x["tools"]:
                    dc = float(np.linalg.norm(self.d.geom_xpos[g] - self.d.geom_xpos[tg]))
                    if dc > self.m.geom_rbound[g] + self.m.geom_rbound[tg] + 1e-4:
                        continue
                    if self.mj.mj_geomDistance(self.m, self.d, int(tg), int(g), 1e-3, fromto) <= 0.0:
                        out.add(u)
                        break
                if u in out:
                    break
        return sorted(out)

    def _carrier_units(self, x):
        """carriers of the graph as units of this cell (a 'finger_module_at_<lane>' is the unit locked in that lane)"""
        out = []
        for cr in x["carriers"]:
            if cr.startswith("finger_module_at_"):
                lane = cr[len("finger_module_at_"):]
                key = {"index": "seat_FI", "middle": "seat_FM", "ring": "seat_FR"}[lane]
                u = self.rec[key]["unit"]
                if u:
                    out.append(u)
            else:
                out.append(cr)
        return out

    def _mounted_ok(self, x, unit):
        kind = self.kit.kind_of(unit)
        js = self.jmodel["kinds"].get(kind, {}).get("joints", {})
        for jn, jr in js.items():
            if "range_mounted" not in jr:
                continue
            r = jr["range_mounted"][x["key"]]
            q = float(self.d.qpos[self.m.jnt_qposadr[self.unit_joints[unit][jn]]])
            if not (r[0] - 1e-4 <= q <= r[1] + 1e-4):
                return False
        return True

    def _set_mounted(self, x, unit):
        kind = self.kit.kind_of(unit)
        js = self.jmodel["kinds"].get(kind, {}).get("joints", {})
        for jn, jr in js.items():
            if "range_mounted" in jr:
                self.m.jnt_range[self.unit_joints[unit][jn]] = jr["range_mounted"][x["key"]]

    def support_share(self, x, unit):
        """the share of the unit's weight along its insertion axis that its seat carries now: the normal forces of
        the unit's contacts with the parts it may sit on (the palm base, the fixture and every link of every fastened
        unit), projected on the axis (towards the open side), over the unit's weight along it. The gripper's contacts,
        other loose units and every friction force do not count."""
        m, d, mj = self.m, self.d, self.mj
        ub = self.unit_link_bodies[unit]
        car = set(self.support_static)
        for r in self.rec.values():
            if r["fastened"] and r["unit"] is not None and r["unit"] != unit:
                car |= self.unit_link_bodies[r["unit"]]
        up = -(body_pose(d, self.bc)[:3, :3] @ x["ins"])
        w = -self.unit_mass[unit] * float(np.asarray(m.opt.gravity, float) @ up)
        if w <= 0.0:
            return 0.0
        f6 = np.zeros(6)
        fn = 0.0
        for i in range(d.ncon):
            ct = d.contact[i]
            b1, b2 = int(m.geom_bodyid[ct.geom1]), int(m.geom_bodyid[ct.geom2])
            if b1 in ub and b2 in car:
                sgn = -1.0
            elif b2 in ub and b1 in car:
                sgn = 1.0
            else:
                continue
            mj.mj_contactForce(m, d, i, f6)
            fn += sgn * float(f6[0]) * float(np.asarray(ct.frame[:3], float) @ up)
        return fn / w

    # ------------------------------------------------------------------ per kind: the engaging window
    def _eval(self, x, c, ti):
        E = self.rel_pose(x, c, ti)
        ang = rotation_angle(E[:3, :3])
        r = {"E": E, "ang": ang}
        if x["kind"] == "module_slide":
            out = -E[:3, 3] @ x["ins"]                  # how far out of home along the path (m)
            r["out"] = float(out)
            if x["thumb"]:
                wall = np.cross([0.0, 0.0, 1.0], x["ins"])
                wall /= np.linalg.norm(wall)
                r["lat"] = float(E[:3, 3] @ wall)
            else:
                r["lat"] = float(E[0, 3])
            r["lift"] = float(E[2, 3])
            yaw, pitch, _roll = zxy_angles(E[:3, :3])
            r["yaw"], r["pitch"] = yaw, pitch
        else:
            above = -E[:3, 3] @ x["ins"]
            r["above"] = float(above)
            r["lat"] = float(np.linalg.norm(E[:3, 3] + above * x["ins"]))
        return r

    def _engage_ok(self, x, r):
        w = x["w"]
        if x["kind"] == "module_slide":
            ms = self.c["module_slide"]
            last = ms["last_mm"] * 1e-3
            if x["thumb"]:
                return (last <= r["out"] <= ms["thumb_engage_out_mm"] * 1e-3 and -1.05e-3 <= r["lat"] <= 0.95e-3
                        and r["lift"] <= w["engaged"]["floor_gap_max_mm"] * 1e-3 and r["ang"] <= math.radians(w["engaged"]["yaw_pitch_deg"]))
            e = w["engaged"]
            inside = x["depth"] - r["out"]
            return (inside >= e["inside_min_mm"] * 1e-3 and r["out"] >= last and abs(r["lat"]) <= e["slot_half_mm"] * 1e-3
                    and abs(r["yaw"]) <= math.radians(e["yaw_pitch_deg"]) and abs(r["pitch"]) <= math.radians(e["yaw_pitch_deg"])
                    and -(ms["floor_push_depth_mm"] + ms["engage_below_margin_mm"]) * 1e-3 <= r["lift"] <= e["floor_gap_max_mm"] * 1e-3)
        e = w["engaged"]
        lat = e.get("across_mm", e.get("slot_half_mm"))
        return (r["lat"] <= lat * 1e-3 and r["ang"] <= math.radians(e.get("angle_deg", 5.0))
                and -1.0e-3 <= r["above"] <= e["above_home_max_mm"] * 1e-3)

    def _released(self, x, r):
        if x["kind"] == "module_slide":
            ms = self.c["module_slide"]
            if x["thumb"]:
                if r["out"] > ms["thumb_release_out_mm"] * 1e-3:
                    return "left the wall"
            elif x["depth"] - r["out"] < ms["release_inside_mm"] * 1e-3:
                return "left the pocket"
            if abs(r["lift"]) > ms["lift_release_mm"] * 1e-3:
                return "off the floor (lifted, or dropped past the pocket's mouth)"
            if r["ang"] > math.radians(ms["tilt_release_deg"]):
                return "tilted"
            return None
        e = x["w"]["engaged"]
        f = self.c[x["kind"]]["release_factor"]
        lat = e.get("across_mm", e.get("slot_half_mm"))
        if r["lat"] > f * lat * 1e-3 or r["ang"] > f * math.radians(e.get("angle_deg", 5.0)) or r["above"] > f * e["above_home_max_mm"] * 1e-3:
            return "left the seat's window"
        return None

    def _lock_ok(self, x, r):
        lk = x["w"]["lock"]
        if x["kind"] == "module_slide":
            return r["out"] <= lk["end_face_to_wall_mm"] * 1e-3 and abs(r["lift"]) <= 0.5e-3
        if x["kind"] == "part_seat":
            return r["lat"] <= lk["across_mm"] * 1e-3 and r["ang"] <= math.radians(lk["angle_deg"]) and r["above"] <= lk["above_home_max_mm"] * 1e-3
        return abs(r["above"]) <= lk["within_mm"] * 1e-3

    # ------------------------------------------------------------------ per control tick
    def tick(self, sim_time):
        self.t = float(sim_time)
        for x in self.slots:
            key = x["key"]
            st = self.rec[key]
            if st["fastened"]:
                self.state[key] = "LOCKED" if st["achieved"] else "LOCKED_OFF"
                continue
            if not self._requires_ok(x):
                if key in self.bound:
                    self._release(x, "a required mate is not fastened")
                self.state[key] = "WAITING"
                self.hold[key] = 0
                continue
            self._tick_slot(x, sim_time)

    def _tick_slot(self, x, sim_time):
        key = x["key"]
        b = self.bound.get(key)
        if b is not None:
            ci, ti = b
            c = x["cands"][ci]
            r = self._eval(x, c, ti)
            why = self._released(x, r)
            if why:
                self._release(x, why)
                b = None
        if b is None:
            best = None
            for i, c in enumerate(x["cands"]):
                if c["unit"] in self.used:
                    continue
                for ti in range(len(x["targets"])):
                    r = self._eval(x, c, ti)
                    if self._engage_ok(x, r) and (best is None or r["lat"] < best[2]["lat"]):
                        best = (i, ti, r)
            if best is None:
                self.state[key] = "FREE"
                self.hold[key] = 0
                return
            ci, ti, r = best
            c = x["cands"][ci]
            self.bound[key] = (ci, ti)
            self.used[c["unit"]] = key
            self.rec[key]["capture"] = {"unit": c["unit"], "t_s": round(float(sim_time), 4), "target": x["targets"][ti][0],
                                        "lat_mm": round(r["lat"] * 1e3, 4), "ang_deg": round(math.degrees(r["ang"]), 4)}
            self.rec[key]["t_engaged_s"] = float(sim_time)
            self.events.append((sim_time, "ENGAGED %s (%s, %s)" % (key, c["unit"], x["targets"][ti][0])))
            b = (ci, ti)
        ci, ti = b
        c = x["cands"][ci]
        ready = True
        if "rail" in x:
            home = x["targets"][ti][1]
            if key not in self.pull:
                if x["kind"] == "part_seat":
                    rw, rv = self._rel_speed(c["bp"])
                    pc = self.c["part_seat"]
                    if rv > pc["centre_below_mm_s"] * 1e-3 or rw > math.radians(pc["centre_below_deg_s"]):
                        ready = False
                if ready:
                    E = self.rel_pose(x, c, 0)
                    G = self._set_rail(x, E)          # the rail to where the unit is (its free directions)
                    tgt, done = self._pull(key, _inv(G) @ E, home, sim_time)
            else:
                tgt, done = self._pull(key, None, home, sim_time)
            if ready:
                # the press fit's rail carries the cap with the lock weld's time constant, so that the rail's slide
                # (the fit's friction) is what resists the cap along the axis: with the soft weld (whose stiffness
                # scales with the 1 g rail) the cap hung on the weld like a spring of about 0.1 N/mm, never moved the
                # rail, and sprang back 2.3 mm when the hand let it go; with this one, the slide's friction and this
                # weld solved hard (DEFAULTS["pressed_seat"]), the rail slides once the push passes the fit's 3 N and
                # a cap let go after a press stays where it was put (about 5 um back per N it carried)
                self._weld(c["rw"], tgt, soft=(x["kind"] != "pressed_seat"))
                ready = done
        r = self._eval(x, c, ti)
        rw, rv = self._rel_speed(c["bp"])
        at_rest = rv < self.rest_v and rw < self.rest_w
        cond = ready and self._lock_ok(x, r) and at_rest
        if cond and x["kind"] == "module_slide" and not self._mounted_ok(x, c["unit"]):
            cond = False
        if cond:
            blk = self.tool_blockers(x, c["unit"])
            if blk:
                self.tool_blocks.setdefault(key, set()).update(blk)
                self.rec[key]["blocked_by"] = sorted(self.tool_blocks[key])
                cond = False
        if cond and x["kind"] == "part_seat":
            # carried by its seat, not held over it (by the gripper, or by friction on side faces)
            sh = self.support_share(x, c["unit"])
            self.support[key] = round(sh, 4)
            if sh < float(self.c["part_seat"]["support_share_min"]):
                cond = False
        if not cond:
            self.state[key] = "ENGAGED"
            self.hold[key] = 0
            return
        self.hold[key] += 1
        self.state[key] = "HOLDING"
        if self.hold[key] >= self.hold_ticks:
            self._lock(x, ci, ti, r, sim_time)

    def _lock(self, x, ci, ti, r, sim_time):
        key = x["key"]
        c = x["cands"][ci]
        if "rw" in c:
            self.d.eq_active[c["rw"]] = 0
        self.pull.pop(key, None)
        rel = x["Tdes"] @ x["targets"][ti][1]
        self._weld(c["lw"], rel, soft=False)
        ok = ti == 0
        st = self.rec[key]
        st.update({"fastened": True, "achieved": ok, "t_s": float(sim_time), "unit": c["unit"], "target": x["targets"][ti][0],
                   "pos_err_mm": float(np.linalg.norm(r["E"][:3, 3]) * 1e3), "ang_err_deg": math.degrees(r["ang"]),
                   "why_off": None if ok else x["targets"][ti][0],
                   "time_to_lock_s": float(sim_time) - (st["t_engaged_s"] or float(sim_time))})
        if x["kind"] == "part_seat":
            st["support_share_at_lock"] = self.support.get(key)
        self.bound.pop(key, None)
        self.used[c["unit"]] = key
        self._join_cluster(c["bp"])
        if x["kind"] == "module_slide":
            self._set_mounted(x, c["unit"])
        self.state[key] = "LOCKED" if ok else "LOCKED_OFF"
        self.events.append((sim_time, ("LOCKED %s (%s)" % (key, c["unit"])) if ok else "LOCKED_OFF %s (%s)" % (key, c["unit"])))

    def record(self):
        return [dict(self.rec[x["key"]]) for x in self.slots]

    # ------------------------------------------------------------------ rule readings
    def weld_force_on_body2(self, e):
        d = self.d
        rows = np.where((d.efc_type[:d.nefc] == self.mj.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[:d.nefc] == e))[0]
        if len(rows) < 3:
            return np.zeros(3)
        return -np.array(d.efc_force[rows[:3]], float)

    def fastening_loads(self):
        """R2 (b): {unit: net force (world, N) its fastening carries}: the active weld's (lock or guide) translational
        force on the unit plus the contacts between the unit and its carriers, as vectors (the preload cancels)"""
        m, d, mj = self.m, self.d, self.mj
        out = {}
        for x in self.slots:
            st = self.rec[x["key"]]
            ci = None
            if st["fastened"]:
                ci = [i for i, c in enumerate(x["cands"]) if c["unit"] == st["unit"]][0]
            elif x["key"] in self.bound:
                ci = self.bound[x["key"]][0]
            if ci is None:
                continue
            c = x["cands"][ci]
            F = np.zeros(3)
            for k in ("lw", "rw"):
                if k in c and d.eq_active[c[k]]:
                    F += self.weld_force_on_body2(c[k])
            car = set()
            for u in self._carrier_units(x):
                car.update(self.unit_geoms.get(u, []))
            own = set(self.unit_geoms.get(c["unit"], []))
            f6 = np.zeros(6)
            for i in range(d.ncon):
                ct = d.contact[i]
                g1, g2 = int(ct.geom1), int(ct.geom2)
                if (g1 in own and g2 in car) or (g2 in own and g1 in car):
                    mj.mj_contactForce(m, d, i, f6)
                    fw = ct.frame.reshape(3, 3).T @ f6[:3]
                    F += -fw if g1 in own else fw
            out[c["unit"]] = {"mate": x["key"], "F": F, "N": float(np.linalg.norm(F))}
        return out

    def stop_loads(self):
        """R3: {joint name: torque (N m) the joint carries at its mechanical stop}: while a product joint's limit
        constraint is active, its force plus the joint's own friction-loss force (at rest the solver splits a load
        between the two arbitrarily, up to the 0.02 N m of friction; their sum is the load); 0 elsewhere"""
        d, mj = self.d, self.mj
        out = {}
        n = d.nefc
        lim = np.where(d.efc_type[:n] == mj.mjtConstraint.mjCNSTR_LIMIT_JOINT)[0]
        fr = np.where(d.efc_type[:n] == mj.mjtConstraint.mjCNSTR_FRICTION_DOF)[0]
        fr_by_dof = {int(d.efc_id[r]): float(d.efc_force[r]) for r in fr}
        for r in lim:
            j = int(d.efc_id[r])
            nm = self.m.joint(j).name
            if nm.startswith("rail_"):
                continue
            dof = int(self.m.jnt_dofadr[j])
            # the limit row's force pushes the joint away from its stop (sign by the side); friction with the sign
            # of the joint's generalized direction: add it along the limit's direction
            fl = float(d.efc_force[r])
            q = float(d.qpos[self.m.jnt_qposadr[j]])
            lo, hi = self.m.jnt_range[j]
            sgn = 1.0 if abs(q - lo) <= abs(q - hi) else -1.0       # the lower stop pushes +, the upper -
            tot = fl * sgn + fr_by_dof.get(dof, 0.0)
            out[nm] = abs(tot)
        return out
