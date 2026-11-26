"""Truth and events of a unit: the types, the contact classes, and the robot's part of the truth read from a MuJoCo
state. PUBLIC.

PUBLIC: this module ships with the public runner of a submitted model; the truth of a model run is the model's own
state, read at the model's tool sites. It holds no data of a drawn unit and no constant of the thermal model or of the
reply noise, and imports only numpy and MuJoCo. A unit's truth is a grader's channel: nothing in `Truth` is reachable
through frames; whether a task hands any of it to an agent is the task's choice.

`Event(kind, step, fields)`: edge-triggered, with the physics step at which it happened (integer steps only). Kinds:
fault_set, fault_cleared, enabled, disabled, mode_changed, timeout, zero_saved, register_write_accepted,
register_write_refused, frame_dropped (from the bus layer); contact_begin / contact_end (pair = the two CLASS names,
sorted), stop_begin / stop_end (joint, which = 'lower' | 'upper': the joint-limit constraint active with more than
0.01 N m; for every one of the eighteen hinges, a task says which of them count), torque_clamped (module, which =
'limit' | 'peak', on = True at the first step where the drive's reference exceeds its current limit / envelope, or
where the clamped command exceeds the peak torque, False at the first step where it no longer does); the world's kinds
(weld_attached ...) come from its hooks.

`ContactClasses`: maps every geom of the compiled model to a class name. Default: the bodies of both arms are
'robot'; the geoms of the world body are 'pedestal' (a plane geom 'floor'); any other body's geoms are named after the
body. A task's profile maps body names or geom names to classes of its own (a geom's entry wins over its body's), so
that, e.g., one arm's links against the other arm's or against non-adjacent links of the same arm (self-collision is
on in the maker's model) form a class pair of their own. The touching finger pair of each gripper (the two fingers of
one gripper) is never reported: at the closed pose the maker's two fingers touch each other.

`StopReader`: which stop of a hinge a joint-limit row presses is read from the row itself (the sign of its Jacobian
entry: +1 on the lower stop, -1 on the upper), so that a stop moved at run time (a finger's closing stop set to a seat
angle) is read on its right side.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

TRUTH_FORMAT = "openarm_unit/Truth/1"
STOP_EVENT_THRESHOLD_NM = 0.01          # a stop event: the joint-limit constraint active with more than 0.01 N m
SIDES = ("left", "right")
_LIMIT = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)


def _plain(v):
    """A JSON-able copy (lists of Python floats / ints / bools) of a truth value."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


@dataclass(frozen=True)
class Event:
    """One event: its kind, the physics step at which it happened, and its fields."""
    kind: str
    step: int
    fields: dict = field(default_factory=dict)

    def __getitem__(self, k):
        if k == "kind":
            return self.kind
        if k == "step":
            return self.step
        return self.fields[k]

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default

    def to_dict(self):
        d = {"kind": self.kind, "step": int(self.step)}
        d.update(_plain(self.fields))
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        kind = d.pop("kind")
        step = int(d.pop("step"))
        return cls(kind, step, d)


class Truth:
    """The truth of a unit at one instant: copies of every field, by name (attribute or item access).
    `to_dict()` gives JSON-able data. The fields a unit fills are listed in `unit.OpenArmUnit.truth`."""

    __slots__ = ("_f",)

    def __init__(self, fields):
        object.__setattr__(self, "_f", dict(fields))

    def __getattr__(self, k):
        try:
            return self._f[k]
        except KeyError:
            raise AttributeError(k) from None

    def __getitem__(self, k):
        return self._f[k]

    def __contains__(self, k):
        return k in self._f

    def keys(self):
        return self._f.keys()

    def __setattr__(self, k, v):
        raise AttributeError("a Truth is read-only")

    def to_dict(self):
        d = {"format": TRUTH_FORMAT}
        d.update({k: _plain(v) for k, v in self._f.items()})
        return d


# ------------------------------------------------------------------------------------------------- contact classes
class ContactClasses:
    """Contact classes of every geom of a compiled model (module text). `mapping`: {body or geom name: class}."""

    def __init__(self, model, mapping=None, robot_bodies=()):
        mapping = dict(mapping or {})
        names = []
        robot = set(robot_bodies)
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if gname and gname in mapping:
                c = mapping[gname]
            elif bname in mapping:
                c = mapping[bname]
            elif bname in robot:
                c = "robot"
            elif b == 0:
                c = "floor" if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE) else "pedestal"
            else:
                c = bname or f"body{b}"
            names.append(str(c))
        self.names = tuple(sorted(set(names)))
        idx = {c: i for i, c in enumerate(self.names)}
        self.geom_class = np.array([idx[c] for c in names], dtype=np.int64)
        self.k = len(self.names)
        # the finger pair of each gripper: excluded by body (the two finger bodies of one side share a tag)
        tag = np.zeros(model.nbody, dtype=np.int64)
        for s, side in enumerate(SIDES):
            for fb in (f"openarm_{side}_ee_inner_finger", f"openarm_{side}_ee_outer_finger"):
                b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fb)
                if b >= 0:
                    tag[b] = s + 1
        self.finger_tag_of_geom = tag[np.asarray(model.geom_bodyid, dtype=np.int64)]
        # a contact can be listed outside the constraint (exclude != 0) only where a geom has a margin or a gap
        self.check_exclude = bool(np.any(model.geom_margin != 0.0) or np.any(model.geom_gap != 0.0))
        # the class-pair code (a k + b, a <= b) of every geom pair, prepared once; a gripper's own finger pair gets the
        # code k k, which no class pair has (the per-step scan reads one entry per contact and marks the codes present)
        gc = self.geom_class
        t = self.finger_tag_of_geom
        code = np.minimum(gc[:, None], gc[None, :]) * self.k + np.maximum(gc[:, None], gc[None, :])
        code[(t[:, None] != 0) & (t[:, None] == t[None, :])] = self.k * self.k
        self._pair_code = code.astype(np.int32)
        self._n_codes = self.k * self.k
        self._last = None                   # (the contacts' geom-pair bytes, their codes): the last scan's result

    def pair_name(self, code):
        a, b = divmod(int(code), self.k)
        return (self.names[a], self.names[b])

    def active_codes(self, data):
        """The class-pair codes (a k + b, a <= b) of the contacts of the last solved step, finger pairs excluded, and
        only contacts that enter the constraint (exclude == 0). A sorted int64 array of unique codes."""
        n = int(data.ncon)
        if n == 0:
            return _EMPTY
        con = data.contact
        g = con.geom[:n]
        # the codes are a function of the contacts' geom pairs (and exclusion flags) alone: the last result is
        # reused while those bytes repeat
        key = g.tobytes() if not self.check_exclude else g.tobytes() + con.exclude[:n].tobytes()
        last = self._last
        if last is None or last[0] != key:
            kk = self._n_codes
            codes = self._pair_code[g[:, 0], g[:, 1]]      # one prepared code per contact (k k: a finger pair)
            if self.check_exclude:
                codes = np.where(con.exclude[:n] == 0, codes, kk)
            present = np.zeros(kk + 1, dtype=bool)
            present[codes] = True
            last = self._last = (key, np.flatnonzero(present[:kk]))   # the codes present, ascending, each once
        out = last[1]
        return out.copy() if out.shape[0] else _EMPTY

    def pair_forces(self, model, data):
        """Per active class pair (finger pairs excluded): normal force, tangential force (magnitude) and the number of
        points, summed in contact order over the contacts of the last solved step (mj_contactForce, contact frame)."""
        out = {}
        n = int(data.ncon)
        f6 = np.zeros(6, dtype=np.float64)
        t = self.finger_tag_of_geom
        for j in range(n):
            c = data.contact[j]
            if int(c.exclude) != 0:
                continue
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if t[g1] != 0 and t[g1] == t[g2]:
                continue
            a, b = int(self.geom_class[g1]), int(self.geom_class[g2])
            key = "|".join(sorted((self.names[a], self.names[b])))
            mujoco.mj_contactForce(model, data, j, f6)
            tang = float(np.sqrt(f6[1] * f6[1] + f6[2] * f6[2]))
            e = out.setdefault(key, {"normal_N": 0.0, "tangential_N": 0.0, "points": 0})
            e["normal_N"] = e["normal_N"] + float(f6[0])
            e["tangential_N"] = e["tangential_N"] + tang
            e["points"] += 1
        return out


_EMPTY = np.zeros(0, dtype=np.int64)


# ------------------------------------------------------------------------------------------------------- stop rows
class StopReader:
    """The joint-limit constraint of a set of hinges at the last solved step: the torque on each of the two stops of
    every joint (lower >= 0 pushes up, upper <= 0 pushes down, in the joint's coordinate) and the active set. The side
    of a row is the sign of its Jacobian entry at the joint's dof (MuJoCo writes +1 for the lower stop, -1 for the
    upper), read at every call: a range changed at run time is read on its right side."""

    def __init__(self, model, jids):
        self.jids = np.asarray(jids, dtype=np.int64)
        self.n = len(self.jids)
        self.pos_of_jid = np.full(model.njnt, -1, dtype=np.int64)
        self.pos_of_jid[self.jids] = np.arange(self.n)
        self.qadr = np.asarray([model.jnt_qposadr[j] for j in self.jids], dtype=np.int64)
        self.dadr = np.asarray([model.jnt_dofadr[j] for j in self.jids], dtype=np.int64)
        self.nv = int(model.nv)
        self.sparse = bool(mujoco.mj_isSparse(model))

    def rows(self, data):
        """(joint positions in this reader, upper flag, force magnitude) of the active limit rows (efc_force >= 0)."""
        n = int(data.nefc)
        if n == 0:
            return None
        sel = data.efc_type[:n] == _LIMIT
        if not sel.any():
            return None
        rws = np.nonzero(sel)[0]
        jid = data.efc_id[rws]
        pos = self.pos_of_jid[jid]
        ok = pos >= 0
        if not ok.all():
            rws, pos = rws[ok], pos[ok]
        frc = data.efc_force[rws]
        if self.sparse:
            jac = data.efc_J[data.efc_J_rowadr[rws]]        # a hinge's limit row has one entry: its own dof
        else:
            jac = data.efc_J[rws * self.nv + self.dadr[pos]]
        upper = jac < 0.0
        return pos, upper, frc

    def torques(self, data):
        """(n x 2) torque on the lower and the upper stop of every joint."""
        out = np.zeros((self.n, 2), dtype=np.float64)
        r = self.rows(data)
        if r is None:
            return out
        pos, upper, frc = r
        for p, u, f in zip(pos.tolist(), upper.tolist(), frc.tolist()):
            if u:
                out[p, 1] = out[p, 1] - f
            else:
                out[p, 0] = out[p, 0] + f
        return out

    def active(self, data, threshold=STOP_EVENT_THRESHOLD_NM):
        """Sorted int64 codes 2 x position + (1 if upper) of the stops pressed with more than `threshold`."""
        r = self.rows(data)
        if r is None:
            return _EMPTY
        pos, upper, frc = r
        hit = frc > threshold
        if not hit.any():
            return _EMPTY
        present = np.zeros(2 * self.n, dtype=bool)
        present[pos[hit] * 2 + upper[hit]] = True
        return np.flatnonzero(present)                     # the codes present, ascending, each once


# ------------------------------------------------------------------------------------------------------ tool poses
def frame_pose(data, body_id, offset):
    """World position and orientation (quaternion w x y z) of a point `offset` (body frame) of body `body_id`
    (xpos + R offset, element-wise arithmetic: no matmul)."""
    x = data.xmat[body_id]
    p = data.xpos[body_id]
    o0, o1, o2 = float(offset[0]), float(offset[1]), float(offset[2])
    pos = np.array([p[0] + x[0] * o0 + x[1] * o1 + x[2] * o2,
                    p[1] + x[3] * o0 + x[4] * o1 + x[5] * o2,
                    p[2] + x[6] * o0 + x[7] * o1 + x[8] * o2], dtype=np.float64)
    return pos, np.array(data.xquat[body_id], dtype=np.float64), np.array(x, dtype=np.float64)


def site_pose(model, data, site_id):
    """World position, orientation (quaternion w x y z) and rotation matrix of a site after the kinematics: site_xpos,
    the body's xquat times the site's local quat (the product MuJoCo itself turns into site_xmat), and site_xmat."""
    s = int(site_id)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(quat, data.xquat[int(model.site_bodyid[s])], model.site_quat[s])
    return (np.array(data.site_xpos[s], dtype=np.float64), quat, np.array(data.site_xmat[s], dtype=np.float64))


def relative_pose(p_a, R_a, q_a, p_b, q_b):
    """Pose of frame b in frame a: R_a^T (p_b - p_a) and conj(q_a) q_b (element-wise arithmetic)."""
    d = [float(p_b[i]) - float(p_a[i]) for i in range(3)]
    r = R_a
    pos = np.array([r[0] * d[0] + r[3] * d[1] + r[6] * d[2],
                    r[1] * d[0] + r[4] * d[1] + r[7] * d[2],
                    r[2] * d[0] + r[5] * d[1] + r[8] * d[2]], dtype=np.float64)
    aw, ax, ay, az = float(q_a[0]), -float(q_a[1]), -float(q_a[2]), -float(q_a[3])
    bw, bx, by, bz = (float(v) for v in q_b)
    quat = np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], dtype=np.float64)
    return pos, quat
