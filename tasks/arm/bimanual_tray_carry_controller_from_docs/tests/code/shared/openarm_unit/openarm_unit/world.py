import math
import os

import mujoco
import numpy as np

from . import base
from . import static as _static
from .hooks_api import Hook, HookError, HookSet
from .state import public_params

WORLD_FORMAT = "openarm_unit/World/1"
SIDES = ("left", "right")
GRIPPER_OF = {"left": 7, "right": 15}
PARK_BASE_Z = 20.0
WELD_STATES = ("off", "ramping", "held")
_EQ = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
_SIX = np.arange(6, dtype=np.int64)
_LIMIT = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
PIN_SOLREF = (0.002, 0.5)
PIN_SOLIMP = (0.9999, 0.9999, 0.001, 0.5, 2.0)
PI = 3.141592653589793
_COLLISION = None


def maker_collision_class():
    global _COLLISION
    if _COLLISION is None:
        g = base.maker_spec().find_default("collision").geom
        _COLLISION = {"condim": int(g.condim), "solref": tuple(float(x) for x in g.solref),
                      "friction": tuple(float(x) for x in g.friction)}
    return _COLLISION
GRASP_CAPTURE_N = 5.0
GRASP_RELEASE_N = 2.0
CAPTURE_BOX = (0.008, 4.0)
SEAT_AXES = ("all", "closing")
SEAT_AXES_DEFAULT = {"scheduled": "all", "clamp": "closing"}


class WorldError(ValueError):
    ""


def clamp_lever():
    v = public_params()["gripper"]["r_pad_m"]["value"]
    if v is None:
        raise WorldError("params_public gripper.r_pad_m is null: the clamp lever has no value")
    r = float(v)
    if not (math.isfinite(r) and r > 0.0):
        raise WorldError(f"params_public gripper.r_pad_m: a finite positive length, not {v!r}")
    return r


_CODE_SHA = None


def world_code_sha256():
    global _CODE_SHA
    if _CODE_SHA is None:
        import hashlib
        with open(os.path.abspath(__file__), "rb") as f:
            _CODE_SHA = hashlib.sha256(f.read()).hexdigest()
    return _CODE_SHA


def _f3(v, what):
    t = tuple(float(x) for x in v)
    if len(t) != 3 or not all(math.isfinite(x) for x in t):
        raise WorldError(f"{what}: three finite numbers")
    return t


def _quat(q, what):
    t = tuple(float(x) for x in q)
    if len(t) != 4 or not all(math.isfinite(x) for x in t):
        raise WorldError(f"{what}: four finite numbers (w, x, y, z)")
    n = math.sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2] + t[3] * t[3])
    if not n > 0.0:
        raise WorldError(f"{what}: a non-zero quaternion")
    return (t[0] / n, t[1] / n, t[2] / n, t[3] / n)


def _pose(p, what):
    try:
        pos, quat = p
    except (TypeError, ValueError):
        raise WorldError(f"{what}: a pose ((x, y, z), (w, x, y, z))") from None
    return (_f3(pos, what), _quat(quat, what))


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def qconj(a):
    return (a[0], -a[1], -a[2], -a[3])


def qrot(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx))


def compose(a, b):
    (pa, qa), (pb, qb) = a, b
    r = qrot(qa, pb)
    return ((pa[0] + r[0], pa[1] + r[1], pa[2] + r[2]), qmul(qa, qb))


def inverse(a):
    p, q = a
    c = qconj(q)
    r = qrot(c, p)
    return ((-r[0], -r[1], -r[2]), c)


def qnormalize(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def interp_pose(a, b, t):
    (pa, qa), (pb, qb) = a, b
    if qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3] < 0.0:
        qb = (-qb[0], -qb[1], -qb[2], -qb[3])
    s = 1.0 - t
    p = (s * pa[0] + t * pb[0], s * pa[1] + t * pb[1], s * pa[2] + t * pb[2])
    q = qnormalize((s * qa[0] + t * qb[0], s * qa[1] + t * qb[1], s * qa[2] + t * qb[2], s * qa[3] + t * qb[3]))
    return (p, q)


def swing_twist(q, k):
    if q[0] < 0.0:
        q = (-q[0], -q[1], -q[2], -q[3])
    w, v = q[0], q[1 + k]
    n = math.sqrt(w * w + v * v)
    if not n > 0.0:
        tw = (1.0, 0.0, 0.0, 0.0)
    else:
        t = [w / n, 0.0, 0.0, 0.0]
        t[1 + k] = v / n
        tw = tuple(t)
    return qmul(q, qconj(tw)), tw


def closing_seat_target(capture, seat, k):
    (pc, qc), (ps, qs) = capture, seat
    p = [float(pc[0]), float(pc[1]), float(pc[2])]
    p[k] = float(ps[k])
    s_seat = swing_twist(qs, k)[0]
    t_cap = swing_twist(qc, k)[1]
    return (tuple(p), qnormalize(qmul(s_seat, t_cap)))


MISMATCH_KINDS = ("pad_plane", "closing", "swing", "twist")


def mismatch_six(capture, seat, k):
    (pc, qc), (ps, qs) = capture, seat
    dp = [float(pc[i]) - float(ps[i]) for i in range(3)]
    sc, tc = swing_twist(qc, k)
    ss, ts = swing_twist(qs, k)
    ds = qmul(sc, qconj(ss))
    dt = qmul(tc, qconj(ts))
    if ds[0] < 0.0:
        ds = (-ds[0], -ds[1], -ds[2], -ds[3])
    if dt[0] < 0.0:
        dt = (-dt[0], -dt[1], -dt[2], -dt[3])
    rot = [2.0 * ds[1], 2.0 * ds[2], 2.0 * ds[3]]
    rot[k] = 2.0 * dt[1 + k]
    kinds = ["pad_plane"] * 3 + ["swing"] * 3
    kinds[k], kinds[3 + k] = "closing", "twist"
    return dp + rot, kinds


def sin_series(x):
    x = float(x)
    term, s, x2 = x, x, x * x
    for k in range(1, 15):
        term = -term * x2 / ((2 * k) * (2 * k + 1))
        s = s + term
    return s


def deg(x):
    return float(x) * PI / 180.0


def body_pose(data, bid):
    return (tuple(data.xpos[bid].tolist()), tuple(data.xquat[bid].tolist()))


def steel_cylinder(mass):
    m = float(mass)
    if not m > 0.0:
        raise WorldError("a check weight has a positive mass")
    return tuple(float(x) for x in _static.steel_cylinder_inertia(m))


class World(HookSet):

    def __init__(self):
        super().__init__(())
        self._bound = None
        self._park_slots = 0
        self._gen = 0
        self._in_post = False
        self._key = None
        self._drop()
        self._geom_rest = None

    def _drop(self):
        self._gen += 1
        self._key = None
        self._lim = None
        self._scan = None
        self._ron = None
        self._poses = None

    def _add(self, hook):
        if self._bound is not None:
            raise WorldError("the world is declared before the unit is built")
        if any(h.name == hook.name for h in self.hooks):
            raise WorldError(f"two hooks share the name {hook.name!r}")
        if "." in hook.name or "|" in hook.name or not hook.name:
            raise WorldError("a hook name is non-empty and holds no '.' or '|'")
        hook.world = self
        self.hooks = self.hooks + (hook,)
        return hook

    def _park_pose(self):
        k = self._park_slots
        self._park_slots += 1
        return ((0.0, 0.0, PARK_BASE_Z + float(k)), (1.0, 0.0, 0.0, 0.0))

    def __getitem__(self, name):
        for h in self.hooks:
            if h.name == name:
                return h
        raise KeyError(name)

    def rest(self, name, pose=None, size=None, friction=None, parts=None, cls=None):
        return self._add(Rest(name, pose, size, friction, parts, cls))

    def weight(self, name, arm, mass, inertia=None, offset=(0.0, 0.0, 0.0)):
        return self._add(Weight(name, arm, mass, inertia, offset, self._park_pose()))

    def object(self, name, parts, grip_frames, load=None, pose=None):
        return self._add(Object(name, parts, grip_frames, load, pose, self._park_pose()))

    def grasp(self, name, arm, object, neck, seats, seat_angle, capture_box=CAPTURE_BOX, ramp_steps=None,
              trigger="scheduled", capture_N=GRASP_CAPTURE_N, release_N=GRASP_RELEASE_N, seat_axes=None):
        obj = None
        for h in self.hooks:
            if h.name == object and isinstance(h, Object):
                obj = h
        if obj is None:
            raise WorldError(f"grasp {name!r}: no object {object!r} declared before it")
        if neck not in obj.grip_frames:
            raise WorldError(f"grasp {name!r}: the object {object!r} has no neck {neck!r}")
        for h in self.hooks:
            if isinstance(h, Grasp) and h.arm == arm:
                if h.trigger == "clamp" or trigger == "clamp":
                    raise WorldError("one clamp-triggered grasp per gripper (its finger stop is the block's)")
        return self._add(Grasp(name, arm, obj, neck, seats, seat_angle, capture_box, ramp_steps, trigger,
                               capture_N, release_N, seat_axes))

    def push(self, name, target, schedule):
        return self._add(Push(name, target, schedule))

    def supply(self, name="supply"):
        return self._add(Supply(name))

    def place(self, object, pose, step):
        self[object].place(pose, step)

    def park(self, object, step):
        self[object].park(step)

    def set_load(self, object, load, step):
        self[object].set_load(load, step)

    def fit_object_to_seats(self, object, grasps, step):
        obj = self[object]
        gs = [self[g] for g in grasps]
        for g in gs:
            if not isinstance(g, Grasp) or g.obj is not obj or g.trigger != "scheduled":
                raise WorldError("fit_object_to_seats takes scheduled grasps of that object")
        obj._schedule(step, ("fit", [g.name for g in gs]))
        for g in gs:
            g.close(step)

    def bind(self, unit):
        if self._bound is not None and self._bound is not unit:
            raise WorldError("a world is bound to one unit: build another World from the same declaration for the "
                             "second unit")
        self._bound = unit
        self._drop()
        self._geom_rest = None
        super().bind(unit)

    def declaration(self):
        d = super().declaration()
        d["world_format"] = WORLD_FORMAT
        d["world_code_sha256"] = world_code_sha256()
        return d

    def load_state(self, s):
        self._drop()
        super().load_state(s)

    def set_world_state(self, unit, state):
        self._drop()
        super().set_world_state(unit, state)

    def reset(self, unit):
        self._drop()
        super().reset(unit)

    def pre_step(self, unit, n):
        self._drop()
        super().pre_step(unit, n)

    def events(self):
        out = []
        for h in self.hooks:
            if not isinstance(h, _WorldHook) or h._ev:
                for e in h.events():
                    e = dict(e)
                    e.setdefault("hook", h.name)
                    out.append(e)
        return out

    def post_step(self, unit, n):
        d = unit.data
        self._drop()
        self._poses = (d.xpos.copy(), d.xquat.copy())
        self._in_post = True
        try:
            super().post_step(unit, n)
        finally:
            self._in_post = False
        self._key = (d.qpos.tobytes(), d.qvel.tobytes())

    def observe(self, unit):
        if self._in_post:
            return
        d = unit.data
        k = self._key
        if k is None or k[0] != d.qpos.tobytes() or k[1] != d.qvel.tobytes():
            self._drop()
            self._key = (d.qpos.tobytes(), d.qvel.tobytes())

    def body_pose(self, unit, bid):
        if self._poses is None:
            return body_pose(unit.data, bid)
        xp, xq = self._poses
        return (tuple(xp[bid].tolist()), tuple(xq[bid].tolist()))

    def object_parts_of(self):
        out = {}
        for h in self.hooks:
            if isinstance(h, Object):
                for gid in h.geom_ids:
                    out[gid] = h
        return out

    def limit_scan(self, unit):
        if self._lim is not None:
            return self._lim
        d, m = unit.data, unit.model
        ne = int(d.nefc)
        rows = np.flatnonzero(d.efc_type[:ne] == _LIMIT) if ne else np.zeros(0, np.int64)
        if rows.shape[0] == 0:
            self._lim = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0, bool))
        else:
            ids = np.asarray(d.efc_id[rows], dtype=np.int64)
            if mujoco.mj_isSparse(m):
                jac = d.efc_J[d.efc_J_rowadr[rows]]
            else:
                jac = d.efc_J[rows * int(m.nv) + m.jnt_dofadr[ids]]
            self._lim = (ids, np.asarray(d.efc_force[rows], dtype=np.float64), jac > 0.0)
        return self._lim

    def _rest_tables(self, unit):
        if self._geom_rest is None:
            m = unit.model
            self._geom_rest = np.full(m.ngeom, -1, dtype=np.int64)
            self._geom_obj = np.zeros(m.ngeom, dtype=bool)
            self._rests = [h for h in self.hooks if isinstance(h, Rest)]
            for k, h in enumerate(self._rests):
                self._geom_rest[h.geom_ids] = k
            for gid in self.object_parts_of():
                self._geom_obj[gid] = True
            self._pair_keys = {}
            cc = unit.contact_classes
            self._nc = len(cc.names)
            rc = np.where(self._geom_rest >= 0, self._geom_rest * self._nc + cc.geom_class, -1)
            oc = np.where(self._geom_obj, cc.geom_class, -1)
            first = (rc[:, None] >= 0) & (oc[None, :] >= 0)
            second = (rc[None, :] >= 0) & (oc[:, None] >= 0)
            self._pair_rest_code = np.where(first, rc[:, None] * self._nc + oc[None, :],
                                            np.where(second, rc[None, :] * self._nc + oc[:, None], -1))

    def rest_object_normals(self, unit):
        if self._ron is not None:
            return self._ron
        self._rest_tables(unit)
        g, fn, _ = self.contact_scan(unit)
        out = {}
        if g.shape[0]:
            code = self._pair_rest_code[g[:, 0], g[:, 1]]
            sel = code >= 0
            if sel.any():
                cc = unit.contact_classes
                nc = self._nc
                code = code[sel]
                acc = {}
                get = acc.get
                for c, f in zip(code.tolist(), fn[sel].tolist()):
                    acc[c] = get(c, 0.0) + f
                keys = self._pair_keys
                for c, v in sorted(acc.items()):
                    kk = keys.get(c)
                    if kk is None:
                        ri, rest_cls, obj_cls = c // (nc * nc), (c // nc) % nc, c % nc
                        kk = keys[c] = (self._rests[ri].name, cc.names[rest_cls] + "|" + cc.names[obj_cls])
                    out.setdefault(kk[0], {})[kk[1]] = v
        self._ron = out
        return out

    def contact_scan(self, unit):
        if self._scan is not None:
            return self._scan
        d = unit.data
        nc = int(d.ncon)
        if nc == 0:
            self._scan = (np.zeros((0, 2), np.int64), np.zeros(0), np.zeros(0, np.int64))
        else:
            con = d.contact
            adr = np.array(con.efc_address[:nc], dtype=np.int64)
            keep = (adr >= 0) & (np.asarray(con.exclude[:nc]) == 0)
            g = np.array(con.geom[:nc], dtype=np.int64)
            if keep.all():
                a = adr
            else:
                g, a = g[keep], adr[keep]
            self._scan = (g, np.asarray(d.efc_force)[a], a)
        return self._scan


class _WorldHook(Hook):

    def __init__(self, name):
        self.name = str(name)
        self.world = None
        self.unit = None
        self._ev = []
        self._sched = []
        self._sparse = None

    def bind(self, unit):
        self.unit = unit

    def events(self):
        e, self._ev = self._ev, []
        return e

    def _event(self, kind, n, **fields):
        d = {"kind": kind, "step": int(n), "hook": self.name}
        d.update(fields)
        self._ev.append(d)

    def _schedule(self, step, action):
        step = int(step)
        if self.unit is not None and step < int(self.unit.bus.step_count):
            raise WorldError(f"{self.name}: step {step} has already run (the next step is {self.unit.bus.step_count})")
        self._sched.append([step, action])
        self._sched.sort(key=lambda e: e[0])

    def _due(self, n):
        out = []
        while self._sched and self._sched[0][0] <= n:
            out.append(self._sched.pop(0)[1])
        return out


class Rest(_WorldHook):

    def __init__(self, name, pose, size, friction, parts, cls):
        super().__init__(name)
        self.pose = _pose(pose, f"rest {name}") if pose is not None else ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        if parts is None:
            if size is None:
                raise WorldError(f"rest {name}: a size (half-extents) or parts")
            parts = [(None, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), size)]
        self.parts = []
        for p in parts:
            part, ppose, psize = p
            sz = _f3(psize, f"rest {name} part {part} size")
            if not all(x > 0.0 for x in sz):
                raise WorldError(f"rest {name}: positive half-extents")
            self.parts.append((None if part is None else str(part), _pose(ppose, f"rest {name} part {part}"), sz))
        names = [p[0] for p in self.parts]
        if len(set(names)) != len(names):
            raise WorldError(f"rest {name}: two parts share a name")
        self.friction = (maker_collision_class()["friction"] if friction is None
                         else tuple(float(x) for x in friction))
        if len(self.friction) != 3:
            raise WorldError("friction: (sliding, torsional, rolling)")
        self.cls = dict(cls or {})

    def body_name(self):
        return f"rest_{self.name}"

    def geom_name(self, part):
        return f"rest_{self.name}" + ("" if part is None else f"__{part}")

    def class_of(self, part):
        if part in self.cls:
            return str(self.cls[part])
        return self.name if part is None else f"{self.name}.{part}"

    def declare(self, spec):
        b = spec.worldbody.add_body()
        b.name = self.body_name()
        b.pos = list(self.pose[0])
        b.quat = list(self.pose[1])
        for part, (pp, pq), sz in self.parts:
            g = b.add_geom()
            g.name = self.geom_name(part)
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = list(sz)
            g.pos = list(pp)
            g.quat = list(pq)
            g.friction = list(self.friction)
            g.condim = maker_collision_class()["condim"]
            g.solref = list(maker_collision_class()["solref"])
            g.contype = 1
            g.conaffinity = 1

    def bind(self, unit):
        super().bind(unit)
        m = unit.model
        self.geom_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, self.geom_name(p[0])) for p in self.parts]
        self.bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.body_name())
        self._part_of = {g: p[0] for g, p in zip(self.geom_ids, self.parts)}

    def contact_classes(self):
        return {self.geom_name(p[0]): self.class_of(p[0]) for p in self.parts}

    def declaration(self):
        return {"kind": "Rest", "pose": [list(self.pose[0]), list(self.pose[1])], "friction": list(self.friction),
                "parts": [[p[0], [list(p[1][0]), list(p[1][1])], list(p[2])] for p in self.parts],
                "cls": {str(k): str(v) for k, v in sorted(self.cls.items(), key=lambda kv: str(kv[0]))}}

    def per_step_fields(self):
        return ("normal_N",)

    def per_step(self, unit, n, names):
        self.world.observe(unit)
        return {"normal_N": dict(self.world.rest_object_normals(unit).get(self.name, {}))}

    def truth(self, unit):
        self.world.observe(unit)
        m, d = unit.model, unit.data
        cc = unit.contact_classes
        mine = set(self.geom_ids)
        pairs = {}
        f6 = np.zeros(6, dtype=np.float64)
        objs = self.world.object_parts_of()
        for j in range(int(d.ncon)):
            c = d.contact[j]
            if int(c.exclude) != 0 or int(c.efc_address) < 0:
                continue
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if g1 not in mine and g2 not in mine:
                continue
            r, o = (g1, g2) if g1 in mine else (g2, g1)
            key = f"{cc.names[cc.geom_class[r]]}|{cc.names[cc.geom_class[o]]}"
            mujoco.mj_contactForce(m, d, j, f6)
            fr = c.frame
            fw = [float(f6[0]) * float(fr[i]) + float(f6[1]) * float(fr[3 + i]) + float(f6[2]) * float(fr[6 + i])
                  for i in range(3)]
            if o == g1:
                fw = [-x for x in fw]
            e = pairs.setdefault(key, {"normal_N": 0.0, "tangential_N": 0.0, "points": 0,
                                       "force_on_other_world_N": [0.0, 0.0, 0.0], "with_object": o in objs})
            e["normal_N"] += float(f6[0])
            e["tangential_N"] += math.sqrt(float(f6[1]) * float(f6[1]) + float(f6[2]) * float(f6[2]))
            e["points"] += 1
            e["force_on_other_world_N"] = [e["force_on_other_world_N"][i] + fw[i] for i in range(3)]
        return {"pose": [list(self.pose[0]), list(self.pose[1])], "contacts": pairs}


class _Welded(_WorldHook):

    def _weld_rows(self, unit, eid):
        d = unit.data
        n = int(d.nefc)
        if n == 0:
            return None
        rows = np.nonzero((d.efc_type[:n] == _EQ) & (d.efc_id[:n] == eid))[0]
        return rows if rows.shape[0] else None

    def _weld_force_on_body2(self, unit, eid, dadr):
        m, d = unit.model, unit.data
        rows = self._weld_rows(unit, eid)
        if rows is None:
            return (0.0,) * 6, (0.0, 0.0)
        nv = int(m.nv)
        f = d.efc_force
        J = d.efc_J
        sparse = self._sparse
        if sparse is None or sparse[0] is not m:
            sparse = self._sparse = (m, bool(mujoco.mj_isSparse(m)))
        if sparse[1]:
            gen = [0.0] * 6
            for r in rows.tolist():
                fr = float(f[r])
                adr, cnt = int(d.efc_J_rowadr[r]), int(d.efc_J_rownnz[r])
                cols = d.efc_J_colind[adr:adr + cnt]
                for k in range(cnt):
                    c = int(cols[k]) - dadr
                    if 0 <= c < 6:
                        gen[c] = gen[c] + float(J[adr + k]) * fr
        else:
            r0, nr = int(rows[0]), int(rows.shape[0])
            if int(rows[-1]) - r0 == nr - 1:
                blk = J[r0 * nv:(r0 + nr) * nv].reshape(nr, nv)[:, dadr:dadr + 6]
                fr = f[r0:r0 + nr]
            else:
                blk = J[(rows * nv + dadr)[:, None] + _SIX]
                fr = f[rows]
            gen = None
            for prod in (blk * fr[:, None]).tolist():
                if gen is None:
                    gen = prod
                else:
                    gen = [gen[0] + prod[0], gen[1] + prod[1], gen[2] + prod[2], gen[3] + prod[3], gen[4] + prod[4],
                           gen[5] + prod[5]]
        p = d.efc_pos[rows].tolist()
        vp = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) if len(p) >= 3 else 0.0
        vr = 2.0 * math.sqrt(p[3] * p[3] + p[4] * p[4] + p[5] * p[5]) if len(p) >= 6 else 0.0
        return tuple(gen), (vp, vr)

    @staticmethod
    def _express(F_world, tau_body2_local, body2_pose, frame_world):
        (po, qo), (pf, qf) = body2_pose, frame_world
        Mo = qrot(qo, tau_body2_local)
        r = (po[0] - pf[0], po[1] - pf[1], po[2] - pf[2])
        Fx, Fy, Fz = F_world
        Mp = (Mo[0] + r[1] * Fz - r[2] * Fy, Mo[1] + r[2] * Fx - r[0] * Fz, Mo[2] + r[0] * Fy - r[1] * Fx)
        c = qconj(qf)
        return qrot(c, F_world) + qrot(c, Mp)

    def _set_weld(self, unit, eid, relpose):
        m = unit.model
        m.eq_data[eid, 0:3] = 0.0
        m.eq_data[eid, 3:6] = relpose[0]
        m.eq_data[eid, 6:10] = relpose[1]
        m.eq_data[eid, 10] = 1.0


def _weld_numerics():
    w = public_params()["numerics"]["weld"]["value"]
    sr = [float(x) for x in w["solref"]]
    si = [float(x) for x in w["solimp"]] + [0.5, 2.0][len(w["solimp"]) - 3:]
    return sr, si


def _add_weld(spec, name, body1, body2, relpose, active):
    sr, si = _weld_numerics()
    e = spec.add_equality()
    e.type = mujoco.mjtEq.mjEQ_WELD
    e.objtype = mujoco.mjtObj.mjOBJ_BODY
    e.name = name
    e.name1 = body1
    e.name2 = body2
    e.active = bool(active)
    e.data[0:3] = [0.0, 0.0, 0.0]
    e.data[3:6] = list(relpose[0])
    e.data[6:10] = list(relpose[1])
    e.data[10] = 1.0
    e.solref = sr
    e.solimp = si
    return e


def _tool_frame(unit, side):
    return (tuple(float(x) for x in unit.tool_offset), (1.0, 0.0, 0.0, 0.0))


def closing_axis(model, side, G):
    O = mujoco.mjtObj
    ee = mujoco.mj_name2id(model, O.mjOBJ_BODY, base.EE_BODIES[side])
    pts, axes = [], []
    for fb in (f"openarm_{side}_ee_inner_finger", f"openarm_{side}_ee_outer_finger"):
        b = mujoco.mj_name2id(model, O.mjOBJ_BODY, fb)
        if b < 0 or int(model.body_parentid[b]) != ee:
            raise WorldError(f"the maker's finger body {fb} is not a child of the hand")
        pts.append(tuple(float(x) for x in model.body_pos[b]))
        bq = tuple(float(x) for x in model.body_quat[b])
        for j in range(model.njnt):
            if int(model.jnt_bodyid[j]) == b:
                axes.append(qrot(bq, tuple(float(x) for x in model.jnt_axis[j])))
    d = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2])
    n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    if not n > 0.0:
        raise WorldError(f"the finger hinges of the {side} gripper coincide")
    a = (d[0] / n, d[1] / n, d[2] / n)
    if any(abs(a[0] * h[0] + a[1] * h[1] + a[2] * h[2]) > 1e-12 for h in axes):
        return a, None
    at = qrot(qconj(G[1]), a)
    for k in range(3):
        if abs(abs(at[k]) - 1.0) <= 1e-12 and all(abs(at[i]) <= 1e-12 for i in range(3) if i != k):
            return a, k
    return a, None


def _fresh_kinematics(unit):
    mujoco.mj_kinematics(unit.model, unit.data)


class Weight(_Welded):

    def __init__(self, name, arm, mass, inertia, offset, park):
        super().__init__(name)
        if arm not in SIDES:
            raise WorldError("arm is 'left' or 'right'")
        self.arm = arm
        self.mass = float(mass)
        if not self.mass > 0.0:
            raise WorldError("a positive mass")
        if inertia is None:
            inertia = steel_cylinder(self.mass)
        self.inertia = _f3(inertia, "inertia")
        self.offset = _f3(offset, "offset")
        try:
            pmass = _static.PointMass(self.mass, self.offset, self.inertia)
        except _static.StaticError as e:
            raise WorldError(f"check weight {name}: {e}") from None
        b = self.body_name()
        self.park_slot = park
        self.hook = _static.PointMassHook(arm, pmass, name=b, body_name=b, joint_name=f"{b}_free",
                                          weld_name=f"{b}_hold", park_name=f"{b}_park")
        self.park_pose = (tuple(float(x) for x in _static.PARK_POS), (1.0, 0.0, 0.0, 0.0))
        self.attached = False

    def body_name(self):
        return f"weight_{self.name}"

    def declare(self, spec):
        self.hook.declare(spec)

    def bind(self, unit):
        super().bind(unit)
        self.hook.bind(unit)
        i = self.hook._ids
        self.bid, self.ee = int(i["body"]), int(i["ee"])
        self.qadr, self.dadr = int(i["qadr"]), int(i["dadr"])
        self.eq_hold, self.eq_park = int(i["weld"]), int(i["park"])
        self.rel = (tuple(float(x) for x in i["rel_pos"]), tuple(float(x) for x in i["rel_quat"]))
        tpos, tquat, _ee = _static._tool_frame_local(unit, SIDES.index(self.arm))
        self.tool_local = (tuple(float(x) for x in tpos), tuple(float(x) for x in tquat))

    def attach(self, step):
        self._schedule(step, ("attach",))

    def detach(self, step):
        self._schedule(step, ("detach",))

    def _do_attach(self, unit, n):
        self.hook.place_attached(unit)
        self.attached = True
        self._event("weld_attached", n, arm=self.arm, mass=self.mass)

    def _do_detach(self, unit, n):
        self.hook.park(unit)
        self.attached = False
        self._event("weld_detached", n, arm=self.arm)

    def reset(self, unit):
        self._sched = []
        self.hook.park(unit)
        self.attached = False

    def set_world_state(self, unit, state):
        s = dict(state)
        if s.get("attached"):
            self._do_attach(unit, unit.bus.step_count)
        elif "attached" in s:
            self._do_detach(unit, unit.bus.step_count)
        self._ev = []

    def pre_step(self, unit, n):
        if self._sched and self._sched[0][0] <= n:
            for a in self._due(n):
                if a[0] == "attach" and not self.attached:
                    self._do_attach(unit, n)
                elif a[0] == "detach" and self.attached:
                    self._do_detach(unit, n)

    def wrench(self, unit):
        if not self.attached:
            return (0.0,) * 6, (0.0, 0.0)
        F, v = self._weld_force_on_body2(unit, self.eq_hold, self.dadr)
        W = self.world
        tool = compose(W.body_pose(unit, self.ee), self.tool_local)
        return tuple(self._express(F[0:3], F[3:6], W.body_pose(unit, self.bid), tool)), v

    def per_step_fields(self):
        return ("attached", "wrench", "violation")

    def per_step(self, unit, n, names):
        self.world.observe(unit)
        out = {"attached": 1 if self.attached else 0}
        if "wrench" in names or "violation" in names:
            w, v = self.wrench(unit)
            out["wrench"] = np.array(w)
            out["violation"] = np.array(v)
        return out

    def truth(self, unit):
        self.world.observe(unit)
        w, v = self.wrench(unit)
        return {"attached": self.attached, "mass": self.mass, "wrench": list(w), "violation": list(v),
                "pose": [list(x) for x in body_pose(unit.data, self.bid)]}

    def state(self):
        return {"attached": self.attached, "sched": [[s, list(a)] for s, a in self._sched]}

    def load_state(self, s):
        self.attached = bool(s["attached"])
        self._sched = [[int(st), tuple(a)] for st, a in s["sched"]]

    def declaration(self):
        return {"kind": "Weight", "arm": self.arm, "mass": self.mass, "inertia": list(self.inertia),
                "offset": list(self.offset), "park": [list(self.park_pose[0]), list(self.park_pose[1])]}


_GEOM_TYPES = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
               "sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
               "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}
_SIZE_N = {"box": 3, "cylinder": 2, "sphere": 1, "capsule": 2, "ellipsoid": 3}


class Object(_WorldHook):

    def __init__(self, name, parts, grip_frames, load, pose, park):
        super().__init__(name)
        self.parts = []
        for p in parts:
            if not isinstance(p, dict):
                p = dict(zip(("part", "type", "pose", "size", "mass"), p))
            t = str(p["type"])
            if t not in _GEOM_TYPES:
                raise WorldError(f"object {name}: part type {t!r}")
            sz = tuple(float(x) for x in p["size"])
            if len(sz) != _SIZE_N[t] or not all(x > 0.0 for x in sz):
                raise WorldError(f"object {name}: part {p['part']}: {_SIZE_N[t]} positive sizes for a {t}")
            ms = float(p["mass"])
            if not ms > 0.0:
                raise WorldError(f"object {name}: part {p['part']}: a positive mass")
            fr = (maker_collision_class()["friction"] if p.get("friction") is None
                  else tuple(float(x) for x in p["friction"]))
            self.parts.append({"part": str(p["part"]), "type": t, "pose": _pose(p["pose"], f"part {p['part']}"),
                               "size": sz, "mass": ms, "collide": bool(p.get("collide", True)),
                               "cls": None if p.get("cls") is None else str(p["cls"]), "friction": fr})
        if len({p["part"] for p in self.parts}) != len(self.parts):
            raise WorldError(f"object {name}: two parts share a name")
        self.grip_frames = {str(k): _pose(v, f"grip frame {k}") for k, v in dict(grip_frames).items()}
        self.load0 = self._load_tuple(load)
        self.load = self.load0
        self.pose0 = None if pose is None else _pose(pose, f"object {name} pose")
        self.park_pose = park
        self.parked = self.pose0 is None
        self.grasps = []

    @staticmethod
    def _load_tuple(load):
        if load is None:
            return (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        mass, com, inertia = load
        mass = float(mass)
        if not mass >= 0.0:
            raise WorldError("a load mass >= 0")
        return (mass, _f3(com, "load com"), _f3(inertia, "load inertia"))

    def body_name(self):
        return f"obj_{self.name}"

    def geom_name(self, part):
        return f"obj_{self.name}__{part}"

    def class_of(self, p):
        return p["cls"] if p["cls"] is not None else f"{self.name}.{p['part']}"

    def declare(self, spec):
        b = spec.worldbody.add_body()
        b.name = self.body_name()
        start = self.pose0 if self.pose0 is not None else self.park_pose
        b.pos = list(start[0])
        b.quat = list(start[1])
        j = b.add_freejoint()
        j.name = f"{self.body_name()}_free"
        for p in self.parts:
            g = b.add_geom()
            g.name = self.geom_name(p["part"])
            g.type = _GEOM_TYPES[p["type"]]
            g.size = list(p["size"]) + [0.0] * (3 - len(p["size"]))
            g.pos = list(p["pose"][0])
            g.quat = list(p["pose"][1])
            g.mass = p["mass"]
            g.friction = list(p["friction"])
            g.condim = maker_collision_class()["condim"]
            g.solref = list(maker_collision_class()["solref"])
            g.contype = 1 if p["collide"] else 0
            g.conaffinity = 1 if p["collide"] else 0
        c = b.add_body()
        c.name = f"{self.body_name()}_load"
        c.mass = self.load0[0]
        c.ipos = list(self.load0[1])
        c.iquat = [1.0, 0.0, 0.0, 0.0]
        c.inertia = list(self.load0[2])
        c.explicitinertial = True
        _add_weld(spec, f"{self.body_name()}_park", "world", self.body_name(), self.park_pose, self.pose0 is None)

    def bind(self, unit):
        super().bind(unit)
        m = unit.model
        self.bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.body_name())
        self.load_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{self.body_name()}_load")
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{self.body_name()}_free")
        self.qadr, self.dadr = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        self.eq_park = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"{self.body_name()}_park")
        self.geom_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, self.geom_name(p["part"]))
                         for p in self.parts if p["collide"]]

    def contact_classes(self):
        return {self.geom_name(p["part"]): self.class_of(p) for p in self.parts}

    def place(self, pose, step):
        self._schedule(step, ("place", _pose(pose, "place pose")))

    def park(self, step):
        self._schedule(step, ("park",))

    def set_load(self, load, step):
        self._schedule(step, ("load", self._load_tuple(load)))

    def welded(self):
        return any(g.weld != "off" for g in self.grasps)

    def pose(self, unit):
        q = unit.data.qpos
        a = self.qadr
        return ((float(q[a]), float(q[a + 1]), float(q[a + 2])), (float(q[a + 3]), float(q[a + 4]),
                                                                  float(q[a + 5]), float(q[a + 6])))

    def _put(self, unit, pose):
        d = unit.data
        d.qpos[self.qadr:self.qadr + 3] = pose[0]
        d.qpos[self.qadr + 3:self.qadr + 7] = pose[1]
        d.qvel[self.dadr:self.dadr + 6] = 0.0

    def _apply_load(self, unit, load):
        m = unit.model
        m.body_mass[self.load_bid] = load[0]
        m.body_ipos[self.load_bid] = load[1]
        m.body_inertia[self.load_bid] = load[2]
        self.load = load

    def pre_step(self, unit, n):
        if not (self._sched and self._sched[0][0] <= n):
            return
        for a in self._due(n):
            kind = a[0]
            if kind == "fit":
                self._fit(unit, n, a[1])
                continue
            if self.welded():
                self._event(f"{kind}_refused", n, object=self.name, reason="welded")
                continue
            if kind == "place":
                self._put(unit, a[1])
                unit.data.eq_active[self.eq_park] = 0
                self.parked = False
                self._event("object_placed", n, object=self.name, pose=[list(a[1][0]), list(a[1][1])])
            elif kind == "park":
                self._put(unit, self.park_pose)
                unit.data.eq_active[self.eq_park] = 1
                self.parked = True
                self._event("object_parked", n, object=self.name)
            elif kind == "load":
                self._apply_load(unit, a[1])
                self._event("load_set", n, object=self.name, mass=a[1][0], com=list(a[1][1]))

    def _fit(self, unit, n, gnames):
        _fresh_kinematics(unit)
        poses = []
        for g in [self.world[x] for x in gnames]:
            hand = body_pose(unit.data, g.ee)
            poses.append(compose(hand, g.seat_rel(0)))
        k = 1.0 / len(poses)
        p = tuple(k * sum_fixed([q[0][i] for q in poses]) for i in range(3))
        q0 = poses[0][1]
        acc = [0.0, 0.0, 0.0, 0.0]
        for _, q in poses:
            s = 1.0 if q[0] * q0[0] + q[1] * q0[1] + q[2] * q0[2] + q[3] * q0[3] >= 0.0 else -1.0
            acc = [acc[i] + s * q[i] for i in range(4)]
        pose = (p, qnormalize(tuple(acc)))
        self._put(unit, pose)
        unit.data.eq_active[self.eq_park] = 0
        self.parked = False
        self._event("object_fitted", n, object=self.name, pose=[list(pose[0]), list(pose[1])], grasps=list(gnames))

    def reset(self, unit):
        self._sched = []
        start = self.pose0 if self.pose0 is not None else self.park_pose
        self._put(unit, start)
        unit.data.eq_active[self.eq_park] = 1 if self.pose0 is None else 0
        self.parked = self.pose0 is None
        self._apply_load(unit, self.load0)

    def set_world_state(self, unit, state):
        s = dict(state)
        if "pose" in s:
            self._put(unit, _pose(s["pose"], "pose"))
            unit.data.eq_active[self.eq_park] = 0
            self.parked = False
        if "twist" in s:
            unit.data.qvel[self.dadr:self.dadr + 6] = [float(x) for x in s["twist"]]
        if s.get("parked"):
            self._put(unit, self.park_pose)
            unit.data.eq_active[self.eq_park] = 1
            self.parked = True
        if "load" in s:
            self._apply_load(unit, self._load_tuple(s["load"]))

    def per_step_fields(self):
        return ("pose", "twist")

    def per_step(self, unit, n, names):
        d = unit.data
        return {"pose": d.qpos[self.qadr:self.qadr + 7].copy(), "twist": d.qvel[self.dadr:self.dadr + 6].copy()}

    def truth(self, unit):
        d = unit.data
        return {"pose": [list(x) for x in self.pose(unit)], "twist": [float(x) for x in d.qvel[self.dadr:self.dadr + 6]],
                "parked": self.parked, "load": [self.load[0], list(self.load[1]), list(self.load[2])],
                "mass_total": float(unit.model.body_mass[self.bid]) + self.load[0]}

    def state(self):
        return {"parked": self.parked, "load": [self.load[0], list(self.load[1]), list(self.load[2])],
                "sched": [[s, _jsonable_action(a)] for s, a in self._sched]}

    def load_state(self, s):
        self.parked = bool(s["parked"])
        self.load = (float(s["load"][0]), tuple(s["load"][1]), tuple(s["load"][2]))
        self._sched = [[int(st), _action_from_json(a)] for st, a in s["sched"]]

    def model_fields(self, unit):
        m = unit.model
        return {"load_mass": float(m.body_mass[self.load_bid]), "load_ipos": [float(x) for x in m.body_ipos[self.load_bid]],
                "load_inertia": [float(x) for x in m.body_inertia[self.load_bid]]}

    def load_model_fields(self, unit, f):
        m = unit.model
        m.body_mass[self.load_bid] = f["load_mass"]
        m.body_ipos[self.load_bid] = f["load_ipos"]
        m.body_inertia[self.load_bid] = f["load_inertia"]

    def declaration(self):
        parts = [{"part": p["part"], "type": p["type"], "pose": [list(p["pose"][0]), list(p["pose"][1])],
                  "size": list(p["size"]), "mass": p["mass"], "collide": p["collide"], "cls": p["cls"],
                  "friction": list(p["friction"])} for p in self.parts]
        return {"kind": "Object", "parts": parts,
                "grip_frames": {k: [list(v[0]), list(v[1])] for k, v in sorted(self.grip_frames.items())},
                "load": [self.load0[0], list(self.load0[1]), list(self.load0[2])],
                "pose": None if self.pose0 is None else [list(self.pose0[0]), list(self.pose0[1])],
                "park": [list(self.park_pose[0]), list(self.park_pose[1])]}


def sum_fixed(xs):
    s = 0.0
    for x in xs:
        s = s + float(x)
    return s


def _jsonable_action(a):
    out = []
    for x in a:
        if isinstance(x, tuple):
            out.append(_jsonable_action(x))
        elif isinstance(x, list):
            out.append([_jsonable_action(y) if isinstance(y, tuple) else y for y in x])
        else:
            out.append(x)
    return out


def _action_from_json(a):
    kind = a[0]
    if kind == "place":
        return ("place", (tuple(a[1][0]), tuple(a[1][1])))
    if kind == "load":
        return ("load", (float(a[1][0]), tuple(a[1][1]), tuple(a[1][2])))
    if kind == "fit":
        return ("fit", list(a[1]))
    return tuple(a)


class Grasp(_Welded):

    def __init__(self, name, arm, obj, neck, seats, seat_angle, capture_box, ramp_steps, trigger, capture_N,
                 release_N, seat_axes=None):
        super().__init__(name)
        if arm not in SIDES:
            raise WorldError("arm is 'left' or 'right'")
        if trigger not in ("scheduled", "clamp"):
            raise WorldError("trigger is 'scheduled' or 'clamp'")
        self.arm, self.obj, self.neck, self.trigger = arm, obj, str(neck), trigger
        self.seat_axes = SEAT_AXES_DEFAULT[trigger] if seat_axes is None else seat_axes
        if self.seat_axes not in SEAT_AXES:
            raise WorldError("seat_axes is 'all' or 'closing'")
        self.closing_k = None
        if isinstance(seats, (list, tuple)) and len(seats) == 2 and not isinstance(seats[0][0], (list, tuple)):
            seats = [seats]
        self.seats = [_pose(s, f"grasp {name} seat") for s in seats]
        if not self.seats:
            raise WorldError("at least one seat")
        self.seat_angle = float(seat_angle)
        if not (math.isfinite(self.seat_angle) and self.seat_angle > 0.0):
            raise WorldError("the seat angle is a finite angle measured from closed (> 0)")
        self.box_pos = float(capture_box[0])
        self.box_rot_deg = float(capture_box[1])
        self.box_sin = sin_series(deg(self.box_rot_deg) / 2.0)
        pp = public_params()["numerics"]["grasp_ramp_steps"]["value"]
        self.ramp_steps = int(pp if ramp_steps is None else ramp_steps)
        if self.ramp_steps < 1:
            raise WorldError("ramp_steps >= 1")
        self.capture_N, self.release_N = float(capture_N), float(release_N)
        obj.grasps.append(self)
        self.weld = "off"
        self.ramp_j = 0
        self.ramp_from = None
        self.ramp_to = None
        self.engaged = False
        self.stop_c = None
        self.stop_from = None
        self.stop_j = 0
        self.captures = 0
        self.missed_latch = False
        self.prev_c = None
        self.preload = None
        self.pinned = False
        self.C = 0.0
        self.capture_due = None
        self._obs = None

    def lever(self):
        return clamp_lever()

    def pin_name(self, k):
        return f"grasp_{self.name}_pin{k}"

    def declare(self, spec):
        _add_weld(spec, f"grasp_{self.name}", base.EE_BODIES[self.arm], self.obj.body_name(),
                  ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), False)
        for k in (1, 2):
            e = spec.add_equality()
            e.type = mujoco.mjtEq.mjEQ_JOINT
            e.objtype = mujoco.mjtObj.mjOBJ_JOINT
            e.name = self.pin_name(k)
            e.name1 = f"openarm_{self.arm}_finger_joint{k}"
            e.active = False
            e.data[0:5] = [0.0, 0.0, 0.0, 0.0, 0.0]
            e.solref = list(PIN_SOLREF)
            e.solimp = list(PIN_SOLIMP)

    def bind(self, unit):
        super().bind(unit)
        m = unit.model
        O = mujoco.mjtObj
        self.eid = mujoco.mj_name2id(m, O.mjOBJ_EQUALITY, f"grasp_{self.name}")
        self.pins = [mujoco.mj_name2id(m, O.mjOBJ_EQUALITY, self.pin_name(k)) for k in (1, 2)]
        self.ee = mujoco.mj_name2id(m, O.mjOBJ_BODY, base.EE_BODIES[self.arm])
        self.module = GRIPPER_OF[self.arm]
        self.fj = [mujoco.mj_name2id(m, O.mjOBJ_JOINT, f"openarm_{self.arm}_finger_joint{k}") for k in (1, 2)]
        self.fq = [int(m.jnt_qposadr[j]) for j in self.fj]
        self.fd = [int(m.jnt_dofadr[j]) for j in self.fj]
        self.nominal_range = [tuple(float(x) for x in m.jnt_range[j]) for j in self.fj]
        lo, hi = self.nominal_range[0]
        self.csign = 1.0 if hi > 0.0 else -1.0
        self.col = 0 if self.csign > 0.0 else 1
        self.c_open = hi if self.csign > 0.0 else -lo
        if not self.seat_angle < self.c_open:
            raise WorldError(f"grasp {self.name}: the seat angle lies between closed (0) and fully open "
                             f"({self.c_open!r} rad)")
        self.a0, self.a1 = None, None
        for e in range(m.neq):
            if (int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT) and int(m.eq_obj1id[e]) == self.fj[0]
                    and int(m.eq_obj2id[e]) == self.fj[1]):
                self.a0, self.a1 = float(m.eq_data[e, 0]), float(m.eq_data[e, 1])
        if self.a1 is None or self.a1 == 0.0:
            raise WorldError(f"grasp {self.name}: the maker's finger coupling is not in the model")
        self.r = self.lever()
        self.G = _tool_frame(unit, self.arm)
        self.N = self.obj.grip_frames[self.neck]
        self._jac_sign = 1.0 if self.col == 0 else -1.0
        try:
            self.closing_hand, self.closing_k = closing_axis(m, self.arm, self.G)
        except WorldError:
            if self.seat_axes == "closing":
                raise
            self.closing_hand, self.closing_k = None, None
        if self.seat_axes == "closing" and self.closing_k is None:
            raise WorldError(f"grasp {self.name}: seat_axes 'closing' needs the pads' closing line on an axis of "
                             f"the tool frame (it is {self.closing_hand!r} in the hand's frame)")

    def seat_rel(self, k):
        S = self.seats[min(k, len(self.seats) - 1)]
        return self.rel_of(S)

    def rel_of(self, X):
        return compose(self.G, inverse(compose(self.N, X)))

    def close(self, step):
        if self.trigger != "scheduled":
            raise WorldError(f"grasp {self.name}: close() belongs to the scheduled trigger")
        self._schedule(step, ("close",))

    def release(self, step):
        if self.trigger != "scheduled":
            raise WorldError(f"grasp {self.name}: release() belongs to the scheduled trigger")
        self._schedule(step, ("release",))

    def _c(self, unit):
        return self.csign * float(unit.data.qpos[self.fq[0]])

    def _q_of(self, c):
        q1 = self.csign * c
        return q1, (q1 - self.a0) / self.a1

    def _write_stop(self, unit, c):
        m = unit.model
        for j, (lo, hi) in zip(self.fj, self.nominal_range):
            if c is None:
                m.jnt_range[j, 0], m.jnt_range[j, 1] = lo, hi
            elif self.col == 0:
                m.jnt_range[j, 0], m.jnt_range[j, 1] = c, hi
            else:
                m.jnt_range[j, 0], m.jnt_range[j, 1] = lo, -c

    def _write_pins(self, unit):
        q1, q2 = self._q_of(self.stop_c)
        m = unit.model
        m.eq_data[self.pins[0], 0] = q1
        m.eq_data[self.pins[1], 0] = q2

    def _engage(self, unit, ramp):
        c = self._c(unit)
        if ramp and c < self.seat_angle:
            self.stop_from, self.stop_j = c, 0
            self.stop_c = c
        else:
            self.stop_from, self.stop_j = None, 0
            self.stop_c = self.seat_angle
        self.engaged = True
        self._write_stop(unit, self.stop_c)

    def _pin(self, unit, n, c_met):
        d = unit.data
        w0 = float(d.qvel[self.fd[0]])
        q1, q2 = self._q_of(self.stop_c)
        d.qpos[self.fq[0]], d.qpos[self.fq[1]] = q1, q2
        d.qvel[self.fd[0]] = 0.0
        d.qvel[self.fd[1]] = 0.0
        self._write_pins(unit)
        d.eq_active[self.pins[0]] = 1
        d.eq_active[self.pins[1]] = 1
        self._write_stop(unit, None)
        self.pinned = True
        self.C = 0.0
        self.prev_c = self.stop_c
        self._event("block_met", n, grasp=self.name, arm=self.arm, finger_angle=c_met,
                    finger_speed_rad_s=self.csign * w0, block_angle=self.stop_c)

    def _unpin(self, unit, n=None, why=None):
        d = unit.data
        d.eq_active[self.pins[0]] = 0
        d.eq_active[self.pins[1]] = 0
        was = self.pinned
        self.pinned = False
        self.C = 0.0
        if self.engaged:
            self._write_stop(unit, self.stop_c)
        if was and why is not None:
            self._event("pads_left", n, grasp=self.name, arm=self.arm, reason=why)

    def _disengage(self, unit):
        self.engaged = False
        self._unpin(unit)
        self.stop_c = self.stop_from = None
        self.stop_j = 0
        self._write_stop(unit, None)

    def tool_in_neck(self, unit, fresh=False):
        if fresh:
            _fresh_kinematics(unit)
        d = unit.data
        hand = body_pose(d, self.ee)
        objp = body_pose(d, self.obj.bid)
        return compose(inverse(compose(objp, self.N)), compose(hand, self.G))

    def _solved_tool_in_neck(self, unit):
        W = self.world
        return compose(inverse(compose(W.body_pose(unit, self.obj.bid), self.N)),
                       compose(W.body_pose(unit, self.ee), self.G))

    def inside(self, rel):
        p, q = rel
        if q[0] < 0.0:
            q = (-q[0], -q[1], -q[2], -q[3])
        b, s = self.box_pos, self.box_sin
        return (abs(p[0]) <= b and abs(p[1]) <= b and abs(p[2]) <= b and abs(q[1]) <= s and abs(q[2]) <= s
                and abs(q[3]) <= s)

    def _present_rel(self, unit):
        d = unit.data
        return compose(inverse(body_pose(d, self.ee)), body_pose(d, self.obj.bid))

    def _weld_on(self, unit, n, clamp_force_N):
        _fresh_kinematics(unit)
        rel0 = self._present_rel(unit)
        k = self.captures
        S = self.seats[min(k, len(self.seats) - 1)]
        tin = self.tool_in_neck(unit)
        if self.seat_axes == "all":
            self.ramp_from, self.ramp_to = rel0, self.seat_rel(k)
        else:
            self.ramp_from, self.ramp_to = tin, closing_seat_target(tin, S, self.closing_k)
        self.ramp_j = -1
        self._set_weld(unit, self.eid, rel0)
        unit.data.eq_active[self.eid] = 1
        self.weld = "ramping"
        mis = compose(inverse(S), tin)
        six, kinds = (mismatch_six(tin, S, self.closing_k) if self.closing_k is not None
                      else ([None] * 6, [None] * 6))
        removes = [True] * 6 if self.seat_axes == "all" else [kd in ("closing", "swing") for kd in kinds]
        self._event("grasp_closed", n, grasp=self.name, arm=self.arm, object=self.obj.name, neck=self.neck,
                    seat_index=k, seat_reused=k >= len(self.seats), clamp_force_N=clamp_force_N,
                    mismatch_pos_m=list(mis[0]), mismatch_quat=list(mis[1]), seat_axes=self.seat_axes,
                    mismatch_six=six, mismatch_kinds=kinds, seat_removes=removes)
        self.captures += 1

    def _weld_off(self, unit, n, why):
        unit.data.eq_active[self.eid] = 0
        self.weld = "off"
        self.ramp_from = self.ramp_to = None
        self.ramp_j = 0
        self._event("grasp_released", n, grasp=self.name, arm=self.arm, object=self.obj.name, reason=why,
                    clamp_force_N=self.C / self.r)

    def _t_F(self, unit):
        s = unit.data.qfrc_smooth
        tau = float(s[self.fd[0]]) + float(s[self.fd[1]]) / self.a1
        return -self.csign * tau, float(unit.model.dof_frictionloss[self.fd[0]])

    def observed_t_F(self, unit):
        W = self.world
        W.observe(unit)
        o = self._obs
        if o is not None and o[0] == W._gen:
            return o[1], o[2]
        t, F = self._t_F(unit)
        self._obs = (W._gen, t, F)
        return t, F

    def _closing_stop_torque(self, unit):
        ids, frc, lower = self.world.limit_scan(unit)
        tq = 0.0
        if ids.shape[0]:
            want_lower = self.col == 0
            for k in np.flatnonzero((ids == self.fj[0]) | (ids == self.fj[1])).tolist():
                if bool(lower[k]) == want_lower:
                    tq = tq + float(frc[k])
        return tq

    def _meets(self, unit, c, prev):
        sc = self.stop_c
        return c <= sc and (prev > sc or self._closing_stop_torque(unit) > 0.0)

    def clamp(self):
        C = self.C
        return C, C / (2.0 * self.r), C / self.r

    def pre_step(self, unit, n):
        if self.capture_due is not None:
            due, self.capture_due = self.capture_due, None
            if self.weld == "off" and self.pinned:
                self._weld_on(unit, n, due["clamp_force_N"])
        if self._sched and self._sched[0][0] <= n:
            for a in self._due(n):
                if a[0] == "close" and self.weld == "off":
                    _fresh_kinematics(unit)
                    if self.inside(self.tool_in_neck(unit)):
                        self._engage(unit, ramp=True)
                        c = self._c(unit)
                        if c <= self.stop_c and not self.pinned:
                            self._pin(unit, n, c)
                        elif self.pinned:
                            self._write_pins(unit)
                        self._weld_on(unit, n, self.C / self.r)
                        if self.stop_from is not None:
                            self.stop_j = -1
                    else:
                        rel = self.tool_in_neck(unit)
                        self._event("capture_missed", n, grasp=self.name, arm=self.arm, object=self.obj.name,
                                    tool_in_neck_pos_m=list(rel[0]), tool_in_neck_quat=list(rel[1]))
                elif a[0] == "release" and self.weld != "off":
                    self._weld_off(unit, n, "scheduled")
                    self._disengage(unit)
        if self.weld == "ramping":
            self.ramp_j += 1
            t = self.ramp_j / self.ramp_steps
            rel = self.ramp_to if self.ramp_j >= self.ramp_steps else interp_pose(self.ramp_from, self.ramp_to, t)
            if self.seat_axes == "closing":
                rel = self.rel_of(rel)
            self._set_weld(unit, self.eid, rel)
            if self.ramp_j >= self.ramp_steps:
                self.weld = "held"
                self._event("grasp_seated", n, grasp=self.name, arm=self.arm)
        if self.engaged and self.stop_from is not None:
            self.stop_j += 1
            if self.stop_j >= self.ramp_steps:
                self.stop_c, self.stop_from = self.seat_angle, None
            else:
                t = self.stop_j / self.ramp_steps
                self.stop_c = (1.0 - t) * self.stop_from + t * self.seat_angle
            if self.pinned:
                self._write_pins(unit)
            else:
                self._write_stop(unit, self.stop_c)

    def post_step(self, unit, n):
        t, F = self._t_F(unit)
        self._obs = (self.world._gen, t, F)
        if self.pinned:
            C = min(max(self.C, t - F), t + F)
            self.C = C if C > 0.0 else 0.0
        else:
            self.C = 0.0
        leaves = self.pinned and self.C == 0.0 and t + F < 0.0
        c = self._c(unit)
        prev = self.prev_c if self.prev_c is not None else c
        self.prev_c = c
        if self.trigger != "clamp":
            if self.engaged:
                if self.pinned:
                    if leaves:
                        self._unpin(unit, n, "net_torque_opens")
                elif self._meets(unit, c, prev):
                    self._pin(unit, n, c)
            if self.weld == "held" and self.preload is None:
                self.preload = self.wrench(unit)[0]
            return
        sa = self.seat_angle
        if self.engaged:
            fc = self.C / self.r
            if self.pinned:
                if self.weld == "off":
                    if self.capture_due is not None:
                        pass
                    elif fc > self.capture_N and self.inside(self._solved_tool_in_neck(unit)):
                        self.capture_due = {"step": int(n), "clamp_force_N": fc}
                    elif not self.inside(self._solved_tool_in_neck(unit)):
                        self._disengage(unit)
                        self._event("block_left", n, grasp=self.name, arm=self.arm)
                    elif leaves:
                        self._unpin(unit, n, "net_torque_opens")
                else:
                    if fc < self.release_N:
                        self._weld_off(unit, n, "clamp")
                        self.preload = None
                    if leaves:
                        self._unpin(unit, n, "net_torque_opens")
            else:
                if not self.inside(self._solved_tool_in_neck(unit)):
                    self._disengage(unit)
                    self._event("block_left", n, grasp=self.name, arm=self.arm)
                elif self._meets(unit, c, prev):
                    self._pin(unit, n, c)
                elif c > self.stop_c and self._closing_stop_torque(unit) == 0.0:
                    self._disengage(unit)
            if self.weld == "held" and self.preload is None:
                self.preload = self.wrench(unit)[0]
            return
        if c > sa:
            self.missed_latch = False
            return
        if prev > sa >= c:
            if self.weld == "off" and self.inside(self._solved_tool_in_neck(unit)):
                self._engage(unit, ramp=False)
                self._pin(unit, n, c)
            elif not self.missed_latch:
                rel = self._solved_tool_in_neck(unit)
                self.missed_latch = True
                self._event("capture_missed", n, grasp=self.name, arm=self.arm, object=self.obj.name,
                            tool_in_neck_pos_m=list(rel[0]), tool_in_neck_quat=list(rel[1]))

    def wrench(self, unit):
        if self.weld == "off":
            return (0.0,) * 6, (0.0, 0.0)
        F, v = self._weld_force_on_body2(unit, self.eid, self.obj.dadr)
        po = self.world.body_pose(unit, self.obj.bid)
        frame = compose(po, self.N)
        return tuple(self._express(F[0:3], F[3:6], po, frame)), v

    def per_step_fields(self):
        return ("weld_state", "engaged", "wrench", "violation", "clamp_torque", "pad_force", "clamp_force",
                "clamp_net_torque", "finger_friction")

    def per_step(self, unit, n, names):
        self.world.observe(unit)
        out = {"weld_state": WELD_STATES.index(self.weld), "engaged": 1 if self.engaged else 0}
        if "wrench" in names or "violation" in names:
            w, v = self.wrench(unit)
            out["wrench"] = np.array(w)
            out["violation"] = np.array(v)
        out["clamp_torque"], out["pad_force"], out["clamp_force"] = self.clamp()
        if "clamp_net_torque" in names or "finger_friction" in names:
            out["clamp_net_torque"], out["finger_friction"] = self.observed_t_F(unit)
        return out

    def truth(self, unit):
        self.world.observe(unit)
        w, v = self.wrench(unit)
        t, F = self.observed_t_F(unit)
        C, pad, cf = self.clamp()
        m, d, B = unit.model, unit.data, unit.bus
        i = self.module
        st = unit.stops.torques(d)
        pos = [int(unit.stops.pos_of_jid[j]) for j in self.fj]
        return {"arm": self.arm, "object": self.obj.name, "neck": self.neck, "trigger": self.trigger,
                "weld_state": self.weld, "ramp_step": self.ramp_j, "engaged": self.engaged, "pinned": self.pinned,
                "stop_angle": self.stop_c, "seat_angle": self.seat_angle, "captures": self.captures,
                "capture_due": None if self.capture_due is None else dict(self.capture_due),
                "wrench": list(w), "violation": list(v), "preload": None if self.preload is None else list(self.preload),
                "clamp_torque_Nm": C, "pad_force_N": pad, "clamp_force_N": cf, "clamp_net_torque_Nm": t,
                "finger_friction_Nm": F, "lever_m": self.r,
                "finger_stop_torques": [[float(x) for x in st[p]] for p in pos],
                "drive_torque_Nm": float(B.T_del[i]), "delivered_Nm": float(B.delivered()[i]),
                "tool_in_neck": [list(x) for x in self.tool_in_neck(unit)]}

    def state(self):
        def pose_j(p):
            return None if p is None else [list(p[0]), list(p[1])]
        return {"weld": self.weld, "ramp_j": self.ramp_j, "ramp_from": pose_j(self.ramp_from),
                "ramp_to": pose_j(self.ramp_to), "engaged": self.engaged, "stop_c": self.stop_c,
                "stop_from": self.stop_from, "stop_j": self.stop_j, "captures": self.captures,
                "missed_latch": self.missed_latch, "prev_c": self.prev_c, "pinned": self.pinned, "C": self.C,
                "capture_due": None if self.capture_due is None else dict(self.capture_due),
                "preload": None if self.preload is None else list(self.preload),
                "sched": [[s, list(a)] for s, a in self._sched]}

    def load_state(self, s):
        def pose_t(p):
            return None if p is None else (tuple(p[0]), tuple(p[1]))
        self.weld = s["weld"]
        self.ramp_j = int(s["ramp_j"])
        self.ramp_from, self.ramp_to = pose_t(s["ramp_from"]), pose_t(s["ramp_to"])
        self.engaged = bool(s["engaged"])
        self.stop_c, self.stop_from, self.stop_j = s["stop_c"], s["stop_from"], int(s["stop_j"])
        self.captures = int(s["captures"])
        self.missed_latch = bool(s["missed_latch"])
        self.prev_c = s["prev_c"]
        self.pinned = bool(s["pinned"])
        self.C = float(s["C"])
        cd = s["capture_due"]
        self.capture_due = None if cd is None else {"step": int(cd["step"]), "clamp_force_N": float(cd["clamp_force_N"])}
        self.preload = None if s["preload"] is None else tuple(s["preload"])
        self._sched = [[int(st), tuple(a)] for st, a in s["sched"]]
        self._obs = None

    def model_fields(self, unit):
        m = unit.model
        return {"eq_data": [float(x) for x in m.eq_data[self.eid]],
                "pin_data": [float(m.eq_data[e, 0]) for e in self.pins],
                "finger_ranges": [[float(x) for x in m.jnt_range[j]] for j in self.fj]}

    def load_model_fields(self, unit, f):
        m = unit.model
        m.eq_data[self.eid] = f["eq_data"]
        for e, v in zip(self.pins, f["pin_data"]):
            m.eq_data[e, 0] = v
        for j, r in zip(self.fj, f["finger_ranges"]):
            m.jnt_range[j] = r

    def reset(self, unit):
        self._sched = []
        unit.data.eq_active[self.eid] = 0
        self.weld = "off"
        self.ramp_j, self.ramp_from, self.ramp_to = 0, None, None
        self._disengage(unit)
        self.missed_latch = False
        self.prev_c = None
        self.preload = None
        self.capture_due = None
        self.C = 0.0
        self._obs = None

    def set_world_state(self, unit, state):
        s = dict(state)
        if s.get("held"):
            C = float(s.get("clamp_torque", 0.0))
            if not (math.isfinite(C) and C >= 0.0):
                raise WorldError("clamp_torque: a finite torque >= 0")
            _fresh_kinematics(unit)
            rel = self.seat_rel(self.captures)
            if s.get("place_object"):
                pose = compose(body_pose(unit.data, self.ee), rel)
                self.obj._put(unit, pose)
                unit.data.eq_active[self.obj.eq_park] = 0
                self.obj.parked = False
            self._set_weld(unit, self.eid, rel)
            unit.data.eq_active[self.eid] = 1
            self.weld = "held"
            self.ramp_j, self.ramp_from, self.ramp_to = self.ramp_steps, None, None
            self.captures += 1
            self.engaged = True
            self.stop_c, self.stop_from, self.stop_j = self.seat_angle, None, 0
            self.capture_due = None
            self._pin(unit, unit.bus.step_count, self._c(unit))
            self._ev = [e for e in self._ev if e["kind"] != "block_met"]
            self.C = C
        elif "held" in s:
            unit.data.eq_active[self.eid] = 0
            self.weld = "off"
            self.capture_due = None
            self._disengage(unit)
        self._obs = None

    def declaration(self):
        return {"kind": "Grasp", "arm": self.arm, "object": self.obj.name, "neck": self.neck,
                "seats": [[list(p), list(q)] for p, q in self.seats], "seat_angle": self.seat_angle,
                "capture_box": [self.box_pos, self.box_rot_deg], "ramp_steps": self.ramp_steps,
                "trigger": self.trigger, "seat_axes": self.seat_axes, "capture_N": self.capture_N,
                "release_N": self.release_N,
                "lever_m": self.lever(), "pin_solref": list(PIN_SOLREF), "pin_solimp": list(PIN_SOLIMP),
                "clamp_law": "dry-friction play on pinned fingers"}


class Push(_WorldHook):

    def __init__(self, name, target, schedule):
        super().__init__(name)
        kind, what = target
        self.target = (str(kind), what if kind == "tool" else _f3(what, "push point"))
        if kind == "tool" and what not in SIDES:
            raise WorldError("('tool', 'left' | 'right') or (object name, point)")
        pts = [(int(s), _f3(f, "force"), _f3(t, "torque")) for s, f, t in schedule]
        if any(pts[k + 1][0] <= pts[k][0] for k in range(len(pts) - 1)) or not pts:
            raise WorldError("a push schedule: entries at increasing integer steps")
        self.points = pts

    def bind(self, unit):
        super().bind(unit)
        m = unit.model
        if self.target[0] == "tool":
            self.bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base.EE_BODIES[self.target[1]])
            self.local = tuple(float(x) for x in unit.tool_offset)
        else:
            obj = self.world[self.target[0]]
            self.bid = obj.bid
            self.local = self.target[1]

    def value(self, n):
        pts = self.points
        if n < pts[0][0] or n > pts[-1][0]:
            return None
        for k in range(len(pts) - 1):
            s0, f0, t0 = pts[k]
            s1, f1, t1 = pts[k + 1]
            if s0 <= n <= s1:
                a = (n - s0) / (s1 - s0)
                return (tuple((1.0 - a) * f0[i] + a * f1[i] for i in range(3)),
                        tuple((1.0 - a) * t0[i] + a * t1[i] for i in range(3)))
        return pts[-1][1], pts[-1][2]

    def pre_step(self, unit, n):
        v = self.value(n)
        if v is None:
            return
        m, d = unit.model, unit.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = compose(body_pose(d, self.bid), (self.local, (1.0, 0.0, 0.0, 0.0)))[0]
        mujoco.mj_applyFT(m, d, np.array(v[0]), np.array(v[1]), np.array(p), self.bid, d.qfrc_applied)

    def truth(self, unit):
        v = self.value(unit.bus.step_count)
        return {"active": v is not None, "force": None if v is None else list(v[0]),
                "torque": None if v is None else list(v[1])}

    def declaration(self):
        return {"kind": "Push", "target": [self.target[0], self.target[1] if self.target[0] == "tool"
                                           else list(self.target[1])],
                "schedule": [[s, list(f), list(t)] for s, f, t in self.points]}


class Supply(_WorldHook):

    def set(self, volts, step):
        self._schedule(step, ("supply", float(volts)))

    def pre_step(self, unit, n):
        if self._sched and self._sched[0][0] <= n:
            for a in self._due(n):
                unit.bus.set_supply(a[1])
                self._event("supply_set", n, volts=a[1])

    def state(self):
        return {"sched": [[s, list(a)] for s, a in self._sched], "volts": float(self.unit.bus.supply_V)
                if self.unit is not None else None}

    def load_state(self, s):
        self._sched = [[int(st), tuple(a)] for st, a in s["sched"]]
        if s.get("volts") is not None:
            self.unit.bus.set_supply(s["volts"])

    def declaration(self):
        return {"kind": "Supply"}


def measure_lever(model=None):
    m = base.unit_model() if model is None else model
    tool = public_params()["conventions"]["tool_offset_m"]["value"]
    out = {}
    for side in SIDES:
        for fb in (f"openarm_{side}_ee_inner_finger", f"openarm_{side}_ee_outer_finger"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, fb)
            hinge = [float(x) for x in m.body_pos[b]]
            out[fb] = {"hinge_in_hand_m": hinge, "hinge_to_tool_point_along_z_m": hinge[2] - float(tool[2])}
    return out


def seat_angle_for_width(width_m, side="left", model=None, iters=48):
    m = base.unit_model() if model is None else model
    d = mujoco.MjData(m)
    j1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_finger_joint1")
    j2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_finger_joint2")
    s = 1.0 if m.jnt_range[j1, 1] > 0.0 else -1.0
    gs = []
    for fb in (f"openarm_{side}_ee_inner_finger", f"openarm_{side}_ee_outer_finger"):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, fb)
        gs.append([g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]])
    fromto = np.zeros(6)

    def put(a):
        d.qpos[m.jnt_qposadr[j1]] = s * a
        d.qpos[m.jnt_qposadr[j2]] = s * a
        mujoco.mj_kinematics(m, d)
    put(0.0)
    pads = min(((mujoco.mj_geomDistance(m, d, g1, g2, 1.0, fromto), g1, g2) for g1 in gs[0] for g2 in gs[1]))[1:]

    def gap(a):
        put(a)
        return mujoco.mj_geomDistance(m, d, pads[0], pads[1], 1.0, fromto)
    lo, hi = 0.0, 0.7854
    if not gap(hi) >= width_m:
        raise WorldError("wider than the gripper opens")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if gap(mid) >= width_m:
            hi = mid
        else:
            lo = mid
    return hi
