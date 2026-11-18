"""The replica model quad-model/1 -> the dynamics model of acceptance procedure B-14 a (public code; no numbers).

The chain (trunk frame of Appendix D; every frame's axes are parallel to the trunk's at qpos = 0):
    trunk (free joint `root`)
      <leg>_shoulder  at (sx a, sy b, 0)           hinge <leg>_hip_roll,  axis +x
        <leg>_thigh   at (0, sy ell_leg, -drop)     hinge <leg>_hip_pitch, axis +y
          <leg>_shank at (0, 0, -thigh_length)      hinge <leg>_knee,      axis +y
            site <leg>_foot at (0, 0, -shank_length): the foot-sphere centre
with a = pitch_axis_spacing / 2, b = roll_axis_spacing / 2, drop = pitch_axis_drop and ell_leg = leg_offset
(+ hip_spacer on a spacer leg). Every rigid inertial of the model becomes a welded body `rig_<name>` and every BOM mass
a welded body `slot_<joint>`, `slot_pack` or `slot_pod` at 1 kg; each of these bodies has its CoM at its own origin
(explicit inertial at pos 0, slots also with quat 1 0 0 0), and the compiled CoM positions are self-tested against the
analytic frames.

knee_branch gives the knee configuration of B-05 in two characters, the first for the front legs (fl, fr) and the
second for the rear legs (rl, rr): '>' puts the knee axis behind the line from the leg's hip_pitch axis to its foot
point (toward -x of the trunk frame of Appendix D), '<' puts it ahead of that line (toward +x), so the two characters
picture the legs as seen from the left (+y) side with the nose on the left.

    probs = validate(model)               problems of a replica model ([] = valid)
    dyn = build(model, gravity_m_s2)      QuadDyn: compiled model, address maps, self-test results
    xml = build_xml(model, gravity_m_s2)  the byte-deterministic dynamics MJCF
"""
import math

import mujoco
import numpy as np

from . import schemas as S


class ProcedureError(ValueError):
    """An invalid model, layer, catalogue, lot description or BOM, or an unreachable pose."""


TERM_SLOTS = {t: tuple("slot_" + leg + "_" + t for leg in S.LEGS) for t in S.JOINT_TYPES}
TERM_SLOTS["pack"] = ("slot_pack",)
TERM_SLOTS["pod"] = ("slot_pod",)
SLOT_BODIES = tuple("slot_" + j for j in S.JOINTS) + ("slot_pack", "slot_pod")


def _type_of(joint):
    return joint.split("_", 1)[1]


def _leg_of(joint):
    return joint.split("_", 1)[0]


def _f(x):
    """repr of a float, with -0.0 written as 0.0."""
    v = float(x)
    if v == 0:
        v = float(0)
    return repr(v)


def _fs(values):
    return " ".join(_f(v) for v in values)


def box_per_kg(dims):
    """Diagonal inertia per kg of a solid box [A, B, C] about its centre, in the box's axes."""
    a, b, c = (float(x) for x in dims)
    return np.array([(b * b + c * c) / 12, (a * a + c * c) / 12, (a * a + b * b) / 12])


def quat_mat(q):
    """Rotation matrix of a MuJoCo quaternion [w, x, y, z] (normalised first)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _rx(q):
    c, s = math.cos(q), math.sin(q)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _ry(q):
    c, s = math.cos(q), math.sin(q)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


# ---------------------------------------------------------------------------------------------- geometry and angles
def leg_geometry(model):
    """{leg: {"sx", "sy", "hip_x", "hip_y", "lateral_m", "drop_m", "thigh_m", "shank_m", "foot_radius_m",
    "knee_char"}} of the canonical chain (hip_x, hip_y: the shoulder origin in the trunk frame; lateral_m: ell_leg)."""
    g = model["geometry"]
    a = float(g["pitch_axis_spacing"]) / 2
    b = float(g["roll_axis_spacing"]) / 2
    out = {}
    for leg in S.LEGS:
        sx, sy = S.LEG_SIGNS[leg]
        ell = float(g["leg_offset"]) + (float(g["hip_spacer"]) if leg in g["spacer_legs"] else 0)
        knee_char = g["knee_branch"][0 if sx > 0 else 1]
        out[leg] = {"sx": sx, "sy": sy, "hip_x": sx * a, "hip_y": sy * b, "lateral_m": ell,
                    "drop_m": float(g["pitch_axis_drop"]), "thigh_m": float(g["thigh_length"]),
                    "shank_m": float(g["shank_length"]), "foot_radius_m": float(g["foot_radius"]),
                    "knee_char": knee_char}
    return out


def angle_signs(model):
    """{joint: +1 | -1}: the Appendix-D angle of a joint is sign * qpos (hip_roll sy, hip_pitch sx, knee kappa)."""
    geo = leg_geometry(model)
    out = {}
    for leg in S.LEGS:
        g = geo[leg]
        out[leg + "_hip_roll"] = g["sy"]
        out[leg + "_hip_pitch"] = g["sx"]
        out[leg + "_knee"] = -1 if g["knee_char"] == ">" else 1
    return out


def stops_qpos(model):
    """{joint: (lo, hi)} in qpos: an Appendix-D pair [lo, hi] maps to [lo, hi] (sign +1) or [-hi, -lo] (sign -1)."""
    signs = angle_signs(model)
    out = {}
    for j in S.JOINTS:
        lo, hi = (float(x) for x in model["stops"][_type_of(j)])
        out[j] = (lo, hi) if signs[j] > 0 else (-hi, -lo)
    return out


def fk_frames(model, trunk_pos, q12):
    """Analytic frames (no MuJoCo) of the 13 bodies: {body: (R (3, 3), p (3,))} in the world, for a level trunk at
    `trunk_pos` and joint angles q12 (qpos, JOINTS order)."""
    geo = leg_geometry(model)
    q = np.asarray(q12, float)
    T = np.asarray(trunk_pos, float)
    eye = np.eye(3)
    out = {"trunk": (eye, T.copy())}
    for i, leg in enumerate(S.LEGS):
        g = geo[leg]
        qr, qp, qk = q[3 * i], q[3 * i + 1], q[3 * i + 2]
        r_sh = _rx(qr)
        p_sh = T + np.array([g["hip_x"], g["hip_y"], 0])
        r_th = r_sh @ _ry(qp)
        p_th = p_sh + r_sh @ np.array([0, g["sy"] * g["lateral_m"], -g["drop_m"]])
        r_sk = r_th @ _ry(qk)
        p_sk = p_th + r_th @ np.array([0, 0, -g["thigh_m"]])
        out[leg + "_shoulder"] = (r_sh, p_sh)
        out[leg + "_thigh"] = (r_th, p_th)
        out[leg + "_shank"] = (r_sk, p_sk)
    return out


def pod_com(model):
    """The pod's CoM in the trunk frame: mount_centre_m + (dx, 0, dz)."""
    pod = model["pod"]
    dx, dz = (float(v) for v in pod["mark_offset_m"])
    return np.asarray(pod["mount_centre_m"], float) + np.array([dx, 0, dz])


def local_points(model):
    """{added body name: (attach body, (3,) point in that body's frame)} for every rig_ and slot_ body."""
    out = {}
    for r in model["rigid_inertials"]:
        out["rig_" + r["name"]] = (r["attach"], np.asarray(r["com_m"], float))
    for j in S.JOINTS:
        s = model["seats"][j]
        out["slot_" + j] = (s["carrier"], np.asarray(s["pos_m"], float))
    out["slot_pack"] = ("trunk", np.asarray(model["pack_bay"]["centre_m"], float))
    out["slot_pod"] = ("trunk", pod_com(model))
    return out


def intended_points(model, trunk_pos, q12):
    """{added body name: (3,) world point}: the analytic frames applied to each rig_ / slot_ body's local point."""
    frames = fk_frames(model, trunk_pos, q12)
    out = {}
    for name, (body, p) in local_points(model).items():
        R, o = frames[body]
        out[name] = o + R @ p
    return out


def slot_inertias(model):
    """{slot body: (3,) diagonal inertia per kg in the slot frame} (B-14 a: the joint type's housing class box; the
    pack bay's box; the pod as a point mass)."""
    out = {}
    for j in S.JOINTS:
        cls = model["housing_class"][_type_of(j)]
        out["slot_" + j] = box_per_kg(model["class_box_m"][cls])
    out["slot_pack"] = box_per_kg(model["pack_bay"]["inner_m"])
    out["slot_pod"] = np.full(3, 1e-9)
    return out


# ---------------------------------------------------------------------------------------------- validation
def _finite_problems(obj, path):
    out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if not math.isfinite(float(obj)):
            out.append("%s: not a finite number" % path)
    elif isinstance(obj, dict):
        for k in sorted(obj):
            out += _finite_problems(obj[k], "%s/%s" % (path, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += _finite_problems(v, "%s/%d" % (path, i))
    return out


def _parallel_axis(mass, d):
    d = np.asarray(d, float)
    return mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


def _contents(model):
    """{body: [(mass, com (3,), inertia about the com (3, 3), principal moments (3,))]} of the 13 bodies: their rigid
    inertials and the slots they carry at 1 kg (in the body's frame)."""
    out = {b: [] for b in S.BODIES}
    for r in model["rigid_inertials"]:
        inertia = np.asarray(r["inertia_kg_m2"], float)
        out[r["attach"]].append((float(r["mass_kg"]), np.asarray(r["com_m"], float), inertia,
                                 np.linalg.eigvalsh((inertia + inertia.T) / 2)))
    per_kg = slot_inertias(model)
    for j in S.JOINTS:
        s = model["seats"][j]
        R = quat_mat(s["quat"])
        diag = per_kg["slot_" + j]
        out[s["carrier"]].append((1, np.asarray(s["pos_m"], float), R @ np.diag(diag) @ R.T, diag))
    out["trunk"].append((1, np.asarray(model["pack_bay"]["centre_m"], float), np.diag(per_kg["slot_pack"]),
                         per_kg["slot_pack"]))
    out["trunk"].append((1, pod_com(model), np.diag(per_kg["slot_pod"]), per_kg["slot_pod"]))
    return out


def validate(model):
    """Problems of a quad-model/1 ([] = valid). The schema comes first: a model that fails it is not
    checked further."""
    probs = S.validate(model, "quad_model")
    if probs:
        return probs
    probs = _finite_problems(model, "")
    if probs:
        return probs
    names = [r["name"] for r in model["rigid_inertials"]]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        probs.append("rigid inertial names are not unique: %s" % ", ".join(dup))
    bodies = list(S.BODIES) + ["rig_" + n for n in names] + list(SLOT_BODIES)
    if len(set(bodies)) != len(bodies):
        probs.append("body names collide")
    for j in S.JOINTS:
        if model["seats"][j]["carrier"] not in S.BODIES:
            probs.append("seat %s: carrier %r is not a body of the chain" % (j, model["seats"][j]["carrier"]))
    for r in model["rigid_inertials"]:
        if r["attach"] not in S.BODIES:
            probs.append("rigid inertial %s: attach %r is not a body of the chain" % (r["name"], r["attach"]))
    for t in S.JOINT_TYPES:
        cls = model["housing_class"][t]
        if cls not in model["class_box_m"]:
            probs.append("class_box_m lacks the housing class %s of %s" % (cls, t))
    for j in S.JOINTS:
        n = float(np.linalg.norm(np.asarray(model["seats"][j]["quat"], float)))
        if abs(n - 1) > 1e-9:
            probs.append("seat %s: quaternion norm %r is not 1" % (j, n))
    for r in model["rigid_inertials"]:
        inertia = np.asarray(r["inertia_kg_m2"], float)
        scale = float(np.abs(inertia).max())
        if float(np.abs(inertia - inertia.T).max()) > 1e-9 * scale:
            probs.append("rigid inertial %s: inertia tensor is not symmetric" % r["name"])
            continue
        ev = np.linalg.eigvalsh((inertia + inertia.T) / 2)
        if float(ev.min()) < -1e-9 * scale:
            probs.append("rigid inertial %s: a principal moment is negative" % r["name"])
        big = float(ev.max())
        a, b, c = sorted(float(x) for x in ev)
        if a + b < c - 1e-9 * big:
            probs.append("rigid inertial %s: the principal moments violate A + B >= C" % r["name"])
    pod = model["pod"]
    L, W, H = (float(v) for v in pod["box_m"])
    dx, dz = (float(v) for v in pod["mark_offset_m"])
    if abs(dx) > L / 2:
        probs.append("pod: mark_offset_m dx %r lies outside the box length" % dx)
    if dz < 0 or dz > H:
        probs.append("pod: mark_offset_m dz %r lies outside [0, box height]" % dz)
    for t in S.JOINT_TYPES:
        lo, hi = (float(v) for v in model["stops"][t])
        if not lo < hi:
            probs.append("stops %s: lower %r is not below upper %r" % (t, lo, hi))
    if probs:
        return probs
    for body, items in _contents(model).items():
        mass = sum(m for m, _, _, _ in items)
        if not mass > 0:
            probs.append("body %s carries no mass (no rigid inertial and no slot)" % body)
            continue
        com = sum(m * c for m, c, _, _ in items) / mass
        comp = sum(inertia + _parallel_axis(m, c - com) for m, c, inertia, _ in items)
        if not float(np.linalg.eigvalsh(comp).min()) > 0:
            probs.append("body %s: the composite inertia of its contents is not positive definite" % body)
        elif not any(m > 0 and float(np.min(pm)) > 0 for m, _, _, pm in items):
            probs.append("body %s: none of its contents has a positive mass and three positive principal moments "
                         "(the dynamics compiler refuses such a moving body)" % body)
    return probs


# ---------------------------------------------------------------------------------------------- the MJCF
_JOINT_ATTRS = 'pos="0 0 0" limited="false" armature="0" damping="0" frictionloss="0" stiffness="0"'


def _rig_lines(r, ind):
    inertia = np.asarray(r["inertia_kg_m2"], float)
    if not np.any(inertia):
        tail = 'quat="1 0 0 0" mass="%s" diaginertia="0 0 0"' % _f(r["mass_kg"])
    else:
        full = (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
        tail = 'mass="%s" fullinertia="%s"' % (_f(r["mass_kg"]), _fs(full))
    return [ind + '<body name="rig_%s" pos="%s">' % (r["name"], _fs(r["com_m"])),
            ind + '  <inertial pos="0 0 0" %s/>' % tail,
            ind + "</body>"]


def _slot_lines(name, pos, quat, diag, ind):
    head = '<body name="%s" pos="%s"' % (name, _fs(pos))
    if quat is not None:
        head += ' quat="%s"' % _fs(quat)
    return [ind + head + ">",
            ind + '  <inertial pos="0 0 0" quat="1 0 0 0" mass="1" diaginertia="%s"/>' % _fs(diag),
            ind + "</body>"]


def _contents_lines(model, body, ind):
    lines = []
    for r in model["rigid_inertials"]:
        if r["attach"] == body:
            lines += _rig_lines(r, ind)
    per_kg = slot_inertias(model)
    for j in S.JOINTS:
        s = model["seats"][j]
        if s["carrier"] == body:
            lines += _slot_lines("slot_" + j, s["pos_m"], s["quat"], per_kg["slot_" + j], ind)
    return lines


def build_xml(model, gravity_m_s2):
    """The dynamics MJCF of a valid replica model (byte-deterministic)."""
    geo = leg_geometry(model)
    per_kg = slot_inertias(model)
    out = ['<mujoco model="quad_replica">',
           '  <compiler angle="radian" inertiafromgeom="false" autolimits="false" balanceinertia="false"/>',
           '  <option gravity="0 0 %s">' % _f(-float(gravity_m_s2)),
           '    <flag constraint="disable"/>',
           "  </option>",
           "  <worldbody>",
           '    <body name="trunk" pos="0 0 0">',
           '      <freejoint name="root"/>']
    out += _contents_lines(model, "trunk", "      ")
    out += _slot_lines("slot_pack", model["pack_bay"]["centre_m"], None, per_kg["slot_pack"], "      ")
    out += _slot_lines("slot_pod", pod_com(model), None, per_kg["slot_pod"], "      ")
    i1, i2, i3, i4 = " " * 6, " " * 8, " " * 10, " " * 12
    for leg in S.LEGS:
        g = geo[leg]
        out.append(i1 + '<body name="%s_shoulder" pos="%s">' % (leg, _fs((g["hip_x"], g["hip_y"], 0))))
        out.append(i2 + '<joint name="%s_hip_roll" type="hinge" axis="1 0 0" %s/>' % (leg, _JOINT_ATTRS))
        out += _contents_lines(model, leg + "_shoulder", i2)
        out.append(i2 + '<body name="%s_thigh" pos="%s">' % (leg, _fs((0, g["sy"] * g["lateral_m"], -g["drop_m"]))))
        out.append(i3 + '<joint name="%s_hip_pitch" type="hinge" axis="0 1 0" %s/>' % (leg, _JOINT_ATTRS))
        out += _contents_lines(model, leg + "_thigh", i3)
        out.append(i3 + '<body name="%s_shank" pos="%s">' % (leg, _fs((0, 0, -g["thigh_m"]))))
        out.append(i4 + '<joint name="%s_knee" type="hinge" axis="0 1 0" %s/>' % (leg, _JOINT_ATTRS))
        out += _contents_lines(model, leg + "_shank", i4)
        out.append(i4 + '<site name="%s_foot" pos="%s"/>' % (leg, _fs((0, 0, -g["shank_m"]))))
        out.append(i3 + "</body>")
        out.append(i2 + "</body>")
        out.append(i1 + "</body>")
    out += ["    </body>", "  </worldbody>", "</mujoco>"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------- the compiled model
class QuadDyn:
    """The compiled dynamics model of a replica model, with its address maps and self-test results.

    xml, model (mujoco.MjModel), data (mujoco.MjData)
    qadr, dadr          {joint: qpos address}, {joint: dof address}
    site                {leg: site id of "<leg>_foot"}
    slot                {slot body name: body id}
    rig                 {rigid inertial name: body id}
    box_per_kg          {slot body name: (3,) diagonal inertia per kg in the slot frame}
    added               {added body name: body id} (every rig_ and slot_ body)
    xipos_error_m       max over the added bodies at qpos = 0 (trunk origin at the world origin)
    inertia_error_rel   max over the added bodies (compiled tensor vs the input tensor)
    """

    def __init__(self, src, xml, mjm):
        self.src = src
        self.xml = xml
        self.model = mjm
        self.data = mujoco.MjData(mjm)
        jid = {j: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) for j in S.JOINTS}
        self.qadr = {j: int(mjm.jnt_qposadr[i]) for j, i in jid.items()}
        self.dadr = {j: int(mjm.jnt_dofadr[i]) for j, i in jid.items()}
        self.site = {leg: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, leg + "_foot") for leg in S.LEGS}
        self.slot = {n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n) for n in SLOT_BODIES}
        self.rig = {r["name"]: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "rig_" + r["name"])
                    for r in src["rigid_inertials"]}
        self.added = {"rig_" + n: i for n, i in self.rig.items()}
        self.added.update(self.slot)
        self.box_per_kg = slot_inertias(src)
        self.qidx = np.array([self.qadr[j] for j in S.JOINTS])
        self.didx = np.array([self.dadr[j] for j in S.JOINTS])
        self.xipos_error_m = None
        self.inertia_error_rel = None

    def set_pose(self, trunk_pos, q12):
        """Set a level trunk at `trunk_pos` and the 12 joint angles (JOINTS order), zero rates; run the kinematics."""
        d = self.data
        d.qpos[:] = 0
        d.qvel[:] = 0
        d.qacc[:] = 0
        d.qpos[0:3] = trunk_pos
        d.qpos[3] = 1
        d.qpos[self.qidx] = q12
        mujoco.mj_kinematics(self.model, d)

    def xipos_error(self, trunk_pos, q12):
        """Max distance (m) of the compiled CoM of every added body from its intended world point."""
        self.set_pose(trunk_pos, q12)
        want = intended_points(self.src, trunk_pos, q12)
        return max(float(np.linalg.norm(self.data.xipos[i] - want[n])) for n, i in self.added.items())

    def inertia_error(self):
        """Max relative error of the compiled inertia tensors R(iquat) diag(body_inertia) R(iquat)^T against the
        input tensors (rig bodies) and the slot diagonals at 1 kg (relative to each tensor's largest absolute entry)."""
        m = self.model
        want = {}
        for r in self.src["rigid_inertials"]:
            want[self.rig[r["name"]]] = np.asarray(r["inertia_kg_m2"], float)
        for n, i in self.slot.items():
            want[i] = np.diag(self.box_per_kg[n])
        worst = 0
        for i, tensor in want.items():
            R = quat_mat(m.body_iquat[i])
            got = R @ np.diag(m.body_inertia[i]) @ R.T
            scale = float(np.abs(tensor).max())
            err = float(np.abs(got - tensor).max())
            worst = max(worst, err / scale if scale > 0 else err)
        return worst


def build(model, gravity_m_s2):
    """Validate, write the MJCF, compile it and self-test the compiled inertial frames (ProcedureError on failure)."""
    probs = validate(model)
    if probs:
        raise ProcedureError("invalid replica model: " + "; ".join(probs))
    xml = build_xml(model, gravity_m_s2)
    try:
        mjm = mujoco.MjModel.from_xml_string(xml)
    except Exception as e:  # noqa: BLE001 (a compile error of any kind is a model error)
        raise ProcedureError("the dynamics model does not compile: %s" % str(e).strip()) from None
    dyn = QuadDyn(model, xml, mjm)
    dyn.xipos_error_m = dyn.xipos_error(np.zeros(3), np.zeros(len(S.JOINTS)))
    if not dyn.xipos_error_m <= 1e-9:
        raise ProcedureError("compiled CoM positions differ from the intended points by %r m" % dyn.xipos_error_m)
    dyn.inertia_error_rel = dyn.inertia_error()
    return dyn
