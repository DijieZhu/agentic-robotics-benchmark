from dataclasses import dataclass

import mujoco
import numpy as np

from .hooks_api import Hook
from .state import N_MODULES, public_params
from .truth import frame_pose, relative_pose, site_pose

STATIC_FORMAT = "openarm_unit/Static/1"
STEEL_DENSITY_KG_M3 = 7850.0
PI = 3.141592653589793
PARK_POS = (0.0, 0.0, -10.0)
SIDES = ("left", "right")


class StaticError(ValueError):
    ""


def _cbrt(x):
    x = float(x)
    y = x if x >= 1.0 else 1.0
    for _ in range(200):
        y_new = (2.0 * y + x / (y * y)) / 3.0
        if y_new == y:
            break
        y = y_new
    return y


def steel_cylinder_inertia(mass):
    m = float(mass)
    r = _cbrt(m / (2.0 * PI * STEEL_DENSITY_KG_M3))
    h = 2.0 * r
    ixx = m * (3.0 * r * r + h * h) / 12.0
    return (ixx, ixx, m * r * r / 2.0)


@dataclass(frozen=True)
class PointMass:
    mass: float
    offset: tuple = (0.0, 0.0, 0.0)
    inertia: tuple = None

    def __post_init__(self):
        m = float(self.mass)
        if not (m > 0.0 and m < 1e3):
            raise StaticError("a point mass has a positive, finite mass")
        off = tuple(float(x) for x in self.offset)
        if len(off) != 3 or any(not np.isfinite(x) for x in off):
            raise StaticError("offset: three finite numbers (m, in the tool frame)")
        ine = steel_cylinder_inertia(m) if self.inertia is None else tuple(float(x) for x in self.inertia)
        if len(ine) != 3 or any(not (x > 0.0 and np.isfinite(x)) for x in ine):
            raise StaticError("inertia: three positive principal inertias")
        s = sorted(ine)
        if s[0] + s[1] < s[2] * (1.0 - 1e-12):
            raise StaticError("inertia: any two principal inertias sum to at least the third")
        object.__setattr__(self, "mass", m)
        object.__setattr__(self, "offset", off)
        object.__setattr__(self, "inertia", ine)

    def to_dict(self):
        return {"mass": self.mass, "offset": list(self.offset), "inertia": list(self.inertia)}


def _tool_frame_local(unit, s):
    m = unit.model
    if unit.tool_site_ids is None:
        return (np.array(unit.tool_offset, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]), int(unit.ee_body[s]))
    sid = unit.tool_site_ids[s]
    return (np.array(m.site_pos[sid], dtype=np.float64), np.array(m.site_quat[sid], dtype=np.float64),
            int(m.site_bodyid[sid]))


class PointMassHook(Hook):

    def __init__(self, side, point_mass, name=None, *, body_name=None, joint_name=None, weld_name=None,
                 park_name=None):
        if side not in SIDES:
            raise StaticError("side is 'left' or 'right'")
        if not isinstance(point_mass, PointMass):
            raise StaticError("point_mass is a PointMass")
        self.side = side
        self.pm = point_mass
        self.name = name or f"point_mass_{side}"
        self.body_name = body_name or f"{self.name}_body"
        self.joint_name = joint_name or f"{self.name}_free"
        self.weld_name = weld_name or f"{self.name}_weld"
        self.park_name = park_name or f"{self.name}_park"
        self.schedule = []
        self.attached = False
        self._ids = None
        self._ev = []

    def attach(self, step):
        self._add(int(step), "attach")

    def detach(self, step):
        self._add(int(step), "detach")

    def _add(self, step, what):
        if step < 0:
            raise StaticError("a step is a non-negative integer")
        self.schedule = sorted(self.schedule + [(step, what)])

    def declaration(self):
        return {"kind": "PointMassHook", "side": self.side, "point_mass": self.pm.to_dict(),
                "park_pos": list(PARK_POS)}

    def declare(self, spec):
        w = public_params()["numerics"]["weld"]["value"]
        b = spec.worldbody.add_body()
        b.name = self.body_name
        b.pos = list(PARK_POS)
        b.explicitinertial = True
        b.mass = self.pm.mass
        b.ipos = [0.0, 0.0, 0.0]
        b.iquat = [1.0, 0.0, 0.0, 0.0]
        b.inertia = list(self.pm.inertia)
        j = b.add_freejoint()
        j.name = self.joint_name
        for nm, active, b1 in ((self.weld_name, False, f"openarm_{self.side}_ee_base_link"),
                               (self.park_name, True, "world")):
            e = spec.add_equality()
            e.name = nm
            e.type = mujoco.mjtEq.mjEQ_WELD
            e.objtype = mujoco.mjtObj.mjOBJ_BODY
            e.name1 = b1
            e.name2 = self.body_name
            e.active = active
            e.solref = [float(x) for x in w["solref"]]
            imp = [float(x) for x in w["solimp"]]
            e.solimp = imp + [0.5, 2.0][: 5 - len(imp)]

    def bind(self, unit):
        m = unit.model
        O = mujoco.mjtObj
        s = SIDES.index(self.side)
        bid = mujoco.mj_name2id(m, O.mjOBJ_BODY, self.body_name)
        jid = mujoco.mj_name2id(m, O.mjOBJ_JOINT, self.joint_name)
        weld = mujoco.mj_name2id(m, O.mjOBJ_EQUALITY, self.weld_name)
        park = mujoco.mj_name2id(m, O.mjOBJ_EQUALITY, self.park_name)
        if min(bid, jid, weld, park) < 0:
            raise StaticError("the point mass was not declared into this model")
        tpos, tquat, ee = _tool_frame_local(unit, s)
        if ee != int(unit.ee_body[s]):
            raise StaticError("the tool frame is not on openarm_<side>_ee_base_link")
        off = np.zeros(3)
        mujoco.mju_rotVecQuat(off, np.array(self.pm.offset, dtype=np.float64), tquat)
        rel_pos = tpos + off
        m.eq_data[weld, :] = 0.0
        m.eq_data[weld, 3:6] = rel_pos
        m.eq_data[weld, 6:10] = tquat
        m.eq_data[weld, 10] = 1.0
        m.eq_data[park, :] = 0.0
        m.eq_data[park, 3:6] = np.array(PARK_POS)
        m.eq_data[park, 6] = 1.0
        m.eq_data[park, 10] = 1.0
        self._ids = {"body": bid, "qadr": int(m.jnt_qposadr[jid]), "dadr": int(m.jnt_dofadr[jid]), "weld": weld,
                     "park": park, "ee": ee, "rel_pos": rel_pos, "rel_quat": tquat}

    def park(self, unit):
        d, i = unit.data, self._ids
        d.qpos[i["qadr"]:i["qadr"] + 3] = np.array(PARK_POS)
        d.qpos[i["qadr"] + 3:i["qadr"] + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        d.qvel[i["dadr"]:i["dadr"] + 6] = 0.0
        d.eq_active[i["weld"]] = 0
        d.eq_active[i["park"]] = 1

    def place_attached(self, unit):
        m, d, i = unit.model, unit.data, self._ids
        mujoco.mj_kinematics(m, d)
        pos, quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(pos, quat, d.xpos[i["ee"]], d.xquat[i["ee"]], i["rel_pos"], i["rel_quat"])
        d.qpos[i["qadr"]:i["qadr"] + 3] = pos
        d.qpos[i["qadr"] + 3:i["qadr"] + 7] = quat
        d.qvel[i["dadr"]:i["dadr"] + 6] = 0.0
        d.eq_active[i["weld"]] = 1
        d.eq_active[i["park"]] = 0

    def reset(self, unit):
        self.attached = False
        self.park(unit)

    def pre_step(self, unit, n):
        while self.schedule and self.schedule[0][0] <= n:
            _step, what = self.schedule.pop(0)
            if what == "attach":
                self.place_attached(unit)
                self.attached = True
                self._ev.append({"kind": "weld_attached", "step": n, "weld": self.weld_name})
            else:
                self.park(unit)
                self.attached = False
                self._ev.append({"kind": "weld_detached", "step": n, "weld": self.weld_name})

    def events(self):
        e, self._ev = self._ev, []
        return e

    def truth(self, unit):
        d, i = unit.data, self._ids
        return {"attached": bool(self.attached), "pos": [float(x) for x in d.xpos[i["body"]]],
                "mass": self.pm.mass}

    def state(self):
        return {"attached": bool(self.attached), "schedule": [[int(s), w] for s, w in self.schedule]}

    def load_state(self, s):
        self.attached = bool(s["attached"])
        self.schedule = [(int(a), str(b)) for a, b in s["schedule"]]


class StaticResult(dict):

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


def _scratch_data(unit):
    held = getattr(unit, "_static_data", None)
    if held is None or held[0] is not unit.model:
        held = (unit.model, mujoco.MjData(unit.model))
        unit._static_data = held
    return held[1]


def _collision_geoms(m):
    return np.nonzero((np.asarray(m.geom_contype) != 0) | (np.asarray(m.geom_conaffinity) != 0))[0]


def _filtered(m, g1, g2, finger_tag):
    b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
    w1, w2 = int(m.body_weldid[b1]), int(m.body_weldid[b2])
    if w1 == w2:
        return True
    p1, p2 = int(m.body_weldid[m.body_parentid[w1]]), int(m.body_weldid[m.body_parentid[w2]])
    filterparent = not (int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    if filterparent and w1 != 0 and w2 != 0 and (w1 == p2 or w2 == p1):
        return True
    t1, t2 = int(finger_tag[g1]), int(finger_tag[g2])
    if t1 != 0 and t1 == t2:
        return True
    c1, a1 = int(m.geom_contype[g1]), int(m.geom_conaffinity[g1])
    c2, a2 = int(m.geom_contype[g2]), int(m.geom_conaffinity[g2])
    return not ((c1 & a2) or (c2 & a1))


def static(unit, bus_angles, payload=None, contacts=False, clearance=None, distmax=0.5):
    m = unit.model
    B = unit.bus
    th = np.array([float(x) for x in bus_angles], dtype=np.float64)
    if th.shape != (N_MODULES,) or not np.all(np.isfinite(th)):
        raise StaticError("sixteen finite module angles")
    pay = dict(payload or {})
    for s, p in pay.items():
        if s not in SIDES or not isinstance(p, PointMass):
            raise StaticError("payload: {'left' | 'right': PointMass}")
    sd = _scratch_data(unit)
    sd.qpos[:] = unit.data.qpos
    sd.qvel[:] = 0.0
    if sd.act.size:
        sd.act[:] = 0.0
    q16 = B.delta + B.sign * th
    sd.qpos[B.qadr] = q16
    for i, (qa, _da) in unit.finger2.items():
        sd.qpos[qa] = q16[i]
    mujoco.mj_kinematics(m, sd)
    mujoco.mj_comPos(m, sd)
    mujoco.mj_comVel(m, sd)
    poses = []
    for s in range(2):
        if unit.tool_site_ids is None:
            poses.append(frame_pose(sd, unit.ee_body[s], unit.tool_offset))
        else:
            poses.append(site_pose(m, sd, unit.tool_site_ids[s]))
    rel_p, rel_q = relative_pose(poses[0][0], poses[0][2], poses[0][1], poses[1][0], poses[1][1])
    bias = np.zeros(m.nv, dtype=np.float64)
    mujoco.mj_rne(m, sd, 0, bias)
    for s, p in pay.items():
        k = SIDES.index(s)
        tpos, tquat, ee = _tool_frame_local(unit, k)
        off = np.zeros(3)
        mujoco.mju_rotVecQuat(off, np.array(p.offset, dtype=np.float64), tquat)
        local = tpos + off
        point = np.zeros(3)
        mujoco.mju_mulMatVec3(point, sd.xmat[ee], local)
        point = point + sd.xpos[ee]
        force = -p.mass * np.array(m.opt.gravity, dtype=np.float64)
        mujoco.mj_applyFT(m, sd, force, np.zeros(3), point, ee, bias)
    g_model = bias[B.dadr].copy()
    lo = np.array(m.jnt_range[B.jid, 0], dtype=np.float64)
    hi = np.array(m.jnt_range[B.jid, 1], dtype=np.float64)
    a, b = B.sign * (lo - B.delta), B.sign * (hi - B.delta)
    stops = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)
    out = StaticResult(format=STATIC_FORMAT, q=q16.copy(), tool_pos=np.stack([p[0] for p in poses]),
                       tool_quat=np.stack([p[1] for p in poses]), tool_mat=np.stack([p[2] for p in poses]),
                       tool_rel_pos=rel_p, tool_rel_quat=rel_q, gravity=B.sign * g_model, gravity_model=g_model,
                       stops=stops, payload={s: p.to_dict() for s, p in pay.items()})
    if contacts or clearance:
        cc = unit.contact_classes
        flag = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        saved = int(m.opt.disableflags)
        try:
            m.opt.disableflags = saved & ~flag
            mujoco.mj_fwdPosition(m, sd)
            if contacts:
                codes = cc.active_codes(sd)
                pts = {}
                n = int(sd.ncon)
                g = np.array(sd.contact.geom[:n]) if n else np.zeros((0, 2), dtype=np.int64)
                fingers = [False, False]
                for j in range(n):
                    g1, g2 = int(g[j, 0]), int(g[j, 1])
                    if int(sd.contact.exclude[j]) != 0:
                        continue
                    t1, t2 = int(cc.finger_tag_of_geom[g1]), int(cc.finger_tag_of_geom[g2])
                    if t1 != 0 and t1 == t2:
                        fingers[t1 - 1] = True
                        continue
                    key = "|".join(sorted((cc.names[int(cc.geom_class[g1])], cc.names[int(cc.geom_class[g2])])))
                    pts[key] = pts.get(key, 0) + 1
                out["contacts"] = {k: pts[k] for k in sorted(pts)}
                out["contact_codes"] = [list(cc.pair_name(c)) for c in codes.tolist()]
                out["finger_pairs"] = fingers
            if clearance:
                out["clearance"] = _clearance(m, sd, cc, clearance, float(distmax))
        finally:
            m.opt.disableflags = saved
    return out


def _clearance(m, sd, cc, pairs, distmax):
    coll = _collision_geoms(m)
    by_class = {}
    for g in coll.tolist():
        by_class.setdefault(cc.names[int(cc.geom_class[g])], []).append(g)
    fromto = np.zeros(6, dtype=np.float64)
    res = {}
    for pair in pairs:
        a, b = (str(x) for x in pair)
        if a not in cc.names or b not in cc.names:
            raise StaticError(f"unknown contact class in {pair!r}; classes: {cc.names}")
        best, which = distmax, None
        ga, gb = by_class.get(a, []), by_class.get(b, [])
        for g1 in ga:
            for g2 in gb:
                if a == b and g2 <= g1:
                    continue
                if _filtered(m, g1, g2, cc.finger_tag_of_geom):
                    continue
                dist = float(mujoco.mj_geomDistance(m, sd, g1, g2, distmax, fromto))
                if dist < best:
                    best, which = dist, (g1, g2)
        names = None if which is None else [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
                                            for g in which]
        res["|".join((a, b))] = {"distance_m": best, "geoms": names}
    return res


__all__ = ["PointMass", "PointMassHook", "StaticResult", "StaticError", "static", "steel_cylinder_inertia"]
