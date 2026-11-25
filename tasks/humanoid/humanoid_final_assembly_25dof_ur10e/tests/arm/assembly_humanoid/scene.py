"""The humanoid assembly cell's scene builder: the cell assembled from DATA with MjSpec.

New file of this task (not a vendored one). It takes the place of the precedent cells' kit-driven builders
(arm/assembly_car/cell.py `build_spec`, the ROV cell's `RovCell`), which read their own kit formats; the vendored
packages are used as they are: the UR10e with the house wide gripper (`arm.assembly_rov.gripper_wide.add_ur_wide`, its
controller `WideController` = `arm.assembly_rov.ur_controller.URController` + the force register). The data contract
(schema `humanoid-cell-data/1`) is, in short:

  data = {"schema": "humanoid-cell-data/1", "name": str,
          "base": ASSEMBLY (mount "fixed"),  "modules": [ASSEMBLY (mount "free")],  "props": [ASSEMBLY (mount "free")],
          "fixtures": [FIXTURE],  "cameras": [CAMERA],  "provenance": {...}}
  ASSEMBLY = {"id", "kind": "base" | "module" | "prop", "mount": "fixed" | "free",
              "placement": {"frame": "world" | "cad", "pos": [m], "quat": [w, x, y, z]},
              "datum": {"waist_axis_point_m": [x, y, z], "sole_plane_z_m": z}       (the base only: the station's datum)
              "bodies": [BODY] (parents before children; the first is the root),
              "delivered_q": {joint: rad}, "self_collide": bool, "exclude_pairs": [[body, body]], "materials": {name: rgba}}
  BODY = {"name", "parent": name | null, "pose": {"pos", "quat"} (in the assembly frame, every joint at its ref),
          "joint": null | {"name", "type": "hinge" | "slide", "axis", "pos" (body frame), "range": [lo, hi] (rad / m),
                           "ref": the joint's value at the given poses (default 0; the kit's ref_rad_cad_pose),
                           "actuator": EC-A model | "ankle_pitch" | "ankle_roll" (the joint model's key), or explicit
                           "frictionloss" / "damping" / "armature"},
          "inertial": {"mass", "pos", "quat" + "diaginertia" | "fullinertia"} (body frame; required unless static),
          "geoms": [GEOM]}
  GEOM = {"name", "role": "collision" | "visual", "type": "mesh" | "box" | "cylinder" | "sphere" | "capsule" | "plane",
          "size": [...], "pos", "quat" (body frame), "mesh": {"vertices": "npz:<key>" | [...], "faces": "npz:<key>" | [...],
          "file": path relative to the data file}, "rgba" | "material", "rest": bool, "exact": bool,
          "contact": {"friction", "solref", "solimp", "condim", "priority", "margin", "gap"}}
  FIXTURE = {"id", "role": "stand" | "rack" | "bench" | "floor" | "pedestal" | "other", "placement",
             "collide_with_base": bool (default true), "rest_surface": bool (default false), "geoms": [GEOM]}
  CAMERA = {"name", "pos", "quat" | "lookat", "fovy_deg"} (world; the wrist camera comes with the robot)

The physics of record come from the ONE configuration (config.load_config: cell_config.json): angles in radians
(spec.compiler.degree = False, set before anything is added, and every compiled joint range asserted against its
intended range: compare int() joint types, count the joints checked, cover every hinge and slide of the model); the
0.5 ms step and the 10 ms control tick (20 steps, the arm's computed torque recomputed before every step); the
unpowered joint model (frictionloss, damping, armature by actuator type) with the pinned friction
impedance on every such joint; solref 0.002 (with the configuration's resting impedance) on both geoms of every resting pair (a fixture
flagged rest_surface, a module geom flagged rest); the house gripper's pad contact (condim 4, friction 0.7 / 0.005 /
0.0001, solref 1 ms), passed to the gripper's own patch function and asserted after the compile, together with the
gripper tables' physics (`gripper_wide.physics_mismatch` must be empty).

The station of record: the UR10e's base at the cell origin on its pedestal (0.85 m, the
floor at z = -0.85 m), the base placed so that its waist axis stands 0.75 m from the UR's base axis at psi = 90 deg
(the waist axis at -R (cos psi, sin psi)) and its sole plane `soles_above_floor_m` above the floor, facing +x
(`station_transform`). A placement with frame "cad" is relative to that station frame (the base's CAD frame, metres).

Collision bits: the robot keeps its own (1 / 1); base 2, fixtures 4, modules 8, props 16. The base touches the robot,
the modules, the props and every fixture that does not say collide_with_base false; fixtures touch the robot, the
modules and the props; modules and props touch everything. Inside an assembly: MuJoCo's parent-child filter, plus
every pair when self_collide is false, plus exclude_pairs. The contact clusters (clusters_h.py, attached by the
episode builder) rewrite geom_contype / geom_conaffinity (and the body masks as their OR) at run time; their guard runs
from Cell.pre_step_hooks before every physics step's collision and before the tick's mj_forward.
"""
import copy
import json
import math
import os

import numpy as np

from . import config as CF

__all__ = ["SCHEMA", "BIT_ROBOT", "BIT_BASE", "BIT_FIXTURE", "BIT_MODULE", "BIT_PROP", "CLS", "load_data",
           "station_transform", "apply_physics", "build_cell", "Cell", "CellIndex", "check_ranges", "SceneError"]

SCHEMA = "humanoid-cell-data/1"
BIT_ROBOT, BIT_BASE, BIT_FIXTURE, BIT_MODULE, BIT_PROP = 1, 2, 4, 8, 16
ALL_BITS = BIT_ROBOT | BIT_BASE | BIT_FIXTURE | BIT_MODULE | BIT_PROP
# geom classes of the index (the safety monitor reads them)
CLS = {"none": 0, "robot": 1, "pad": 2, "base": 3, "module": 4, "fixture": 5, "floor": 6, "prop": 7}
GEOM_TYPES = ("mesh", "box", "cylinder", "sphere", "capsule", "plane", "ellipsoid")
ROBOT_PREFIX = "ur/"


class SceneError(ValueError):
    pass


# ====================================================================================================== small math
def quat_wxyz(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def mat_of_quat(q):
    import mujoco
    q = np.asarray(q, float)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-9:
        raise SceneError("quaternion %r is not a rotation" % (list(q),))
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q / n)
    return R.reshape(3, 3)


def T_of(pose):
    pose = pose or {}
    T = np.eye(4)
    T[:3, :3] = mat_of_quat(pose.get("quat", [1.0, 0.0, 0.0, 0.0]))
    T[:3, 3] = np.asarray(pose.get("pos", [0.0, 0.0, 0.0]), float)
    return T


def inv_T(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def rotz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


# ====================================================================================================== data
def load_data(path):
    """(data dict, arrays {key: ndarray}, data_dir) of a scene data file (JSON; its 'npz' names the arrays file)"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != SCHEMA:
        raise SceneError("%s: schema %r, this builder reads %s" % (path, data.get("schema"), SCHEMA))
    ddir = os.path.dirname(os.path.abspath(path))
    arrays = {}
    if data.get("npz"):
        with np.load(os.path.join(ddir, data["npz"]), allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files}
    return data, arrays, ddir


def _arr(v, arrays, what):
    if isinstance(v, str):
        if not v.startswith("npz:"):
            raise SceneError("%s: %r is neither a list nor 'npz:<key>'" % (what, v))
        k = v[4:]
        if k not in arrays:
            raise SceneError("%s: no array %r in the data's npz" % (what, k))
        return np.asarray(arrays[k])
    return np.asarray(v)


# ====================================================================================================== station
def station_transform(station, datum):
    """4x4 T_world_cad (metres): the base's CAD frame in the cell (module docstring). `datum` = the base's
    {"waist_axis_point_m": [x, y, z], "sole_plane_z_m": z} in its CAD frame (metres)."""
    R = float(station["waist_axis_from_ur_m"])
    psi = math.radians(float(station["psi_deg"]))
    ur = np.asarray(station.get("ur_base_pos_m", [0.0, 0.0, 0.0]), float)
    T = rotz(float(station.get("robot_yaw_deg", 0.0)))
    w = np.asarray(datum["waist_axis_point_m"], float)
    axis_xy = ur[:2] - R * np.array([math.cos(psi), math.sin(psi)])
    rw = T[:3, :3] @ w
    T[0, 3] = axis_xy[0] - rw[0]
    T[1, 3] = axis_xy[1] - rw[1]
    z_floor = ur[2] - float(station["pedestal_m"])
    T[2, 3] = z_floor + float(station["soles_above_floor_m"]) - float(datum["sole_plane_z_m"])
    return T


def floor_z(station):
    return float(np.asarray(station.get("ur_base_pos_m", [0.0, 0.0, 0.0]), float)[2]) - float(station["pedestal_m"])


# ====================================================================================================== physics
def apply_physics(spec, P):
    """the configuration's physics on an MjSpec (called before anything is added and again after the robot's
    attachment, which brings options of its own)"""
    import mujoco
    spec.compiler.degree = bool(P["compiler_degree"])          # False (check_config refuses anything else)
    spec.compiler.autolimits = True
    spec.memory = int(P["arena_memory_bytes"])
    o = spec.option
    o.timestep = float(P["timestep"])
    o.integrator = {"implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                    "euler": mujoco.mjtIntegrator.mjINT_EULER, "rk4": mujoco.mjtIntegrator.mjINT_RK4}[P["integrator"]]
    o.cone = {"elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC, "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL}[P["cone"]]
    o.impratio = float(P["impratio"])
    o.noslip_iterations = int(P["noslip_iterations"])
    o.gravity = [0.0, 0.0, -float(P["gravity"])]
    if P["multiccd"]:
        o.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    else:
        o.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)


def physics_report(m, P, tick):
    import mujoco
    return {"timestep": float(m.opt.timestep), "tick_s": float(tick), "tick_steps": int(round(tick / m.opt.timestep)),
            "integrator_int": int(m.opt.integrator), "cone_int": int(m.opt.cone), "impratio": float(m.opt.impratio),
            "noslip_iterations": int(m.opt.noslip_iterations), "gravity": float(-m.opt.gravity[2]),
            "multiccd": not bool(int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD))}


def check_physics(m, P, tick, robot):
    """assert that the compiled model runs the configuration's physics and the house gripper's pad contact"""
    import mujoco
    rep = physics_report(m, P, tick)
    want = {"timestep": float(P["timestep"]), "tick_steps": int(P["tick_steps"]),
            "integrator_int": int({"implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                                   "euler": mujoco.mjtIntegrator.mjINT_EULER, "rk4": mujoco.mjtIntegrator.mjINT_RK4}[P["integrator"]]),
            "cone_int": int({"elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC, "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL}[P["cone"]]),
            "impratio": float(P["impratio"]), "noslip_iterations": int(P["noslip_iterations"]),
            "gravity": float(P["gravity"]), "multiccd": bool(P["multiccd"])}
    bad = {k: (rep[k], v) for k, v in want.items() if (abs(rep[k] - v) > 1e-12 if isinstance(v, float) else rep[k] != v)}
    if robot:
        from ..assembly_rov import gripper_wide as GW
        for gn in robot["pad_geoms"]:
            g = m.geom(gn).id
            have = {"solref": [float(x) for x in m.geom_solref[g]], "condim": int(m.geom_condim[g]),
                    "friction": [float(x) for x in m.geom_friction[g]], "priority": int(m.geom_priority[g])}
            exp = {"solref": [float(x) for x in P["pad_solref"]], "condim": int(P["pad_condim"]),
                   "friction": [float(x) for x in P["pad_friction"]], "priority": int(GW.DESIGN["priority"])}
            for k in exp:
                a, b = have[k], exp[k]
                if (isinstance(a, list) and max(abs(x - y) for x, y in zip(a, b)) > 1e-12) or (not isinstance(a, list) and a != b):
                    bad["pad %s %s" % (gn, k)] = (a, b)
        mm = GW.physics_mismatch(m, prefix=robot["prefix"], tick_s=tick)
        for k, v in mm.items():
            bad["gripper tables " + k] = v
        rep["gripper_tables_mismatch"] = {k: list(v) for k, v in mm.items()}
    if bad:
        raise SceneError("the compiled cell does not run the configuration's physics: %s" % (bad,))
    return rep


# ====================================================================================================== joint ranges
_ATTACHED = {}


def attached_intended(models_dir):
    """intended ranges of the attached robot's hinge / slide joints: the UR10e's XML and the patched wide gripper,
    each compiled ALONE with its own compiler settings (names as attached: 'ur/' and 'ur/g/')"""
    import mujoco
    key = os.path.abspath(models_dir)
    if key in _ATTACHED:
        return _ATTACHED[key]
    from ..assembly_rov import gripper_wide as GW
    from ..assembly_rov import ur_robot as UR
    rob = UR.UR_MODELS["ur10e"]
    out = {}
    ma = mujoco.MjModel.from_xml_path(os.path.join(models_dir, rob["dir"], rob["xml"]))
    for j in range(ma.njnt):
        if int(ma.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            out[ROBOT_PREFIX + ma.joint(j).name] = (float(ma.jnt_range[j][0]), float(ma.jnt_range[j][1]), bool(ma.jnt_limited[j]))
    mg = GW.wide_gripper_spec(GW.default_xml(models_dir)).compile()
    for j in range(mg.njnt):
        if int(mg.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            out[ROBOT_PREFIX + "g/" + mg.joint(j).name] = (float(mg.jnt_range[j][0]), float(mg.jnt_range[j][1]), bool(mg.jnt_limited[j]))
    _ATTACHED[key] = out
    return out


def check_ranges(m, intended, attached=None, tol=1e-9):
    """every hinge / slide joint of the compiled model against its intended range (rad / m): `intended` {name: (lo, hi)}
    for the joints built from data (limited), `attached` {name: (lo, hi, limited)} for the robot's. Joint types are
    compared as int() values (`m.jnt_type[j] in (mujoco.mjtJoint...)` is always false). Raises SceneError
    unless every hinge / slide joint is covered and matches and every intended joint is in the model; returns
    {"n_checked", "n_free", "n_hinge_slide", "rows"}."""
    import mujoco
    attached = attached or {}
    HINGE, SLIDE, FREE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_FREE)
    rows, bad, n_free, n_hs = {}, [], 0, 0
    for j in range(m.njnt):
        t = int(m.jnt_type[j])
        if t == FREE:
            n_free += 1
            continue
        if t not in (HINGE, SLIDE):
            bad.append((m.joint(j).name, "type %d" % t))
            continue
        n_hs += 1
        n = m.joint(j).name
        got = (float(m.jnt_range[j][0]), float(m.jnt_range[j][1]))
        lim = bool(m.jnt_limited[j])
        if n in intended:
            want, wlim, src = intended[n], True, "data"
        elif n in attached:
            want, wlim, src = attached[n][:2], attached[n][2], "robot model compiled alone"
        else:
            bad.append((n, "not covered"))
            continue
        err = max(abs(got[0] - want[0]), abs(got[1] - want[1]))
        ok = (lim == wlim) and (err <= tol if wlim else True)
        rows[n] = {"compiled": list(got), "intended": list(want), "limited": lim, "source": src, "max_abs_err": err,
                   "type": "hinge" if t == HINGE else "slide", "ok": ok}
        if not ok:
            bad.append((n, got, want, lim, wlim))
    missing = sorted(set(intended) - set(rows))
    n_checked = sum(1 for r in rows.values() if r["ok"])
    if bad or missing or n_checked != n_hs or n_checked != len(intended) + sum(1 for k in attached if k in rows):
        raise SceneError("joint ranges: bad %s; intended but not in the model %s; checked %d of %d"
                         % (bad[:8], missing[:8], n_checked, n_hs))
    return {"n_checked": n_checked, "n_free": n_free, "n_hinge_slide": n_hs, "rows": rows}


def check_joint_params(m, params, tol=1e-12):
    """the compiled frictionloss / damping / armature / friction impedance of every joint built from data"""
    bad = []
    for n, (fl, dmp, arm, sref, simp) in params.items():
        j = m.joint(n).id
        v = int(m.jnt_dofadr[j])
        have = (float(m.dof_frictionloss[v]), float(m.dof_damping[v]), float(m.dof_armature[v]),
                [float(x) for x in m.dof_solref[v]], [float(x) for x in m.dof_solimp[v]])
        if (abs(have[0] - fl) > tol or abs(have[1] - dmp) > tol or abs(have[2] - arm) > tol
                or max(abs(a - b) for a, b in zip(have[3], sref)) > tol or max(abs(a - b) for a, b in zip(have[4], simp)) > tol):
            bad.append((n, have, (fl, dmp, arm, sref, simp)))
    if bad:
        raise SceneError("joint model not as configured: %s" % (bad[:6],))
    return len(params)


# ====================================================================================================== the index
class CellIndex(object):
    """What the safety monitor and the server need to know about the compiled cell: per geom its class (CLS) and owner
    (module id, "base", fixture id, prop id, "robot"), per body its owner, each module's bodies and mass, the floor's
    geoms, the pads' geoms, the robot's bodies."""

    def __init__(self, m, robot, owners, floor_fixtures, module_ids, prop_ids, fixture_ids, base_id):
        self.m = m
        pads = set(m.geom(n).id for n in (robot or {}).get("pad_geoms", []))
        self.geom_class = np.zeros(m.ngeom, int)
        self.geom_owner = [None] * m.ngeom
        self.body_owner = [None] * m.nbody
        for b in range(m.nbody):
            nm = m.body(b).name
            if nm.startswith(ROBOT_PREFIX):
                self.body_owner[b] = "robot"
            elif nm in owners:
                self.body_owner[b] = owners[nm]
        floor_bodies = set(m.body(fx).id for fx in floor_fixtures)
        self.floor_geoms = set()
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            o = self.body_owner[b]
            self.geom_owner[g] = o
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.geom_class[g] = CLS["none"]
            elif g in pads:
                self.geom_class[g] = CLS["pad"]
            elif o == "robot":
                self.geom_class[g] = CLS["robot"]
            elif o == base_id and base_id is not None:
                self.geom_class[g] = CLS["base"]
            elif o in module_ids:
                self.geom_class[g] = CLS["module"]
            elif o in prop_ids:
                self.geom_class[g] = CLS["prop"]
            elif b in floor_bodies:
                self.geom_class[g] = CLS["floor"]
                self.floor_geoms.add(g)
            elif o in fixture_ids:
                self.geom_class[g] = CLS["fixture"]
            else:
                self.geom_class[g] = CLS["fixture"]          # world geoms (none are built here)
        self.pad_geoms = pads
        self.module_ids = list(module_ids)
        self.prop_ids = list(prop_ids)
        self.fixture_ids = list(fixture_ids)
        self.base_id = base_id
        self.module_bodies = {mid: [b for b in range(m.nbody) if self.body_owner[b] == mid] for mid in module_ids}
        self.module_mass = {mid: float(sum(m.body_mass[b] for b in bl)) for mid, bl in self.module_bodies.items()}
        self.geoms_of = {}
        for g in range(m.ngeom):
            if self.geom_owner[g] is not None and self.geom_class[g] != CLS["none"]:
                self.geoms_of.setdefault(self.geom_owner[g], []).append(g)
        self.robot_bodies = [b for b in range(m.nbody) if self.body_owner[b] == "robot"]

    def summary(self):
        inv = {v: k for k, v in CLS.items()}
        cnt = {}
        for c in self.geom_class:
            cnt[inv[int(c)]] = cnt.get(inv[int(c)], 0) + 1
        return {"geoms_by_class": cnt, "modules": {k: {"bodies": len(v), "mass_kg": round(self.module_mass[k], 6)}
                                                   for k, v in self.module_bodies.items()},
                "props": list(self.prop_ids), "fixtures": list(self.fixture_ids), "floor_geoms": len(self.floor_geoms)}


# ====================================================================================================== the builder
class _Builder(object):
    def __init__(self, spec, cfg, arrays, data_dir):
        self.spec, self.cfg, self.arrays, self.data_dir = spec, cfg, arrays, data_dir
        self.P = cfg["physics"]
        self.intended = {}
        self.joint_params = {}
        self.owners = {}            # body name -> owner id
        self.bodies_of = {}         # owner id -> [body names]
        self.names = {}             # element type -> names used
        self.n_mesh = 0
        self.delivered = {}         # joint name -> q
        self.free_joint_of = {}     # assembly id -> free joint name
        self.excludes = []

    def unique(self, name, what):
        """names are unique per element type (MuJoCo's rule: a body and a geom may share a name)"""
        seen = self.names.setdefault(what, set())
        if not name or name in seen:
            raise SceneError("%s name %r is empty or used twice" % (what, name))
        seen.add(name)
        return name

    def contact_of(self, geom, rest):
        P = self.P
        G = P["geom_defaults"]
        c = {"condim": int(G["condim"]), "friction": list(G["friction"]), "solref": list(G["solref"]), "solimp": list(G["solimp"])}
        if rest:
            c["solref"] = list(P["rest_contact"]["solref"])
            c["solimp"] = list(P["rest_contact"]["solimp"])
        c.update(copy.deepcopy(geom.get("contact") or {}))
        return c

    def add_geom(self, body, g, bits, rest_default=False, materials=None):
        import mujoco
        gtype = g.get("type")
        if gtype not in GEOM_TYPES:
            raise SceneError("geom %r: type %r" % (g.get("name"), gtype))
        role = g.get("role", "collision")
        if role not in ("collision", "visual"):
            raise SceneError("geom %r: role %r" % (g.get("name"), role))
        name = self.unique(g["name"], "geom")
        kw = {"name": name, "type": getattr(mujoco.mjtGeom, "mjGEOM_" + gtype.upper()),
              "pos": [float(v) for v in g.get("pos", [0.0, 0.0, 0.0])],
              "quat": [float(v) for v in g.get("quat", [1.0, 0.0, 0.0, 0.0])], "mass": 0.0}
        if gtype == "mesh":
            ms = g.get("mesh") or {}
            mname = self.unique(name + "__mesh", "mesh")
            if ms.get("file"):
                p = ms["file"] if os.path.isabs(ms["file"]) else os.path.join(self.data_dir or "", ms["file"])
                self.spec.add_mesh(name=mname, file=p)
            else:
                V = _arr(ms.get("vertices"), self.arrays, name + ".mesh.vertices").astype(float).reshape(-1, 3)
                if len(V) < 4 or not np.all(np.isfinite(V)):
                    raise SceneError("geom %r: a mesh needs at least 4 finite vertices" % name)
                kwm = {"name": mname, "uservert": V.reshape(-1).tolist()}
                if ms.get("faces") is not None:
                    F = _arr(ms["faces"], self.arrays, name + ".mesh.faces").astype(int).reshape(-1, 3)
                    kwm["userface"] = F.reshape(-1).tolist()
                self.spec.add_mesh(**kwm)
            kw["meshname"] = mname
            self.n_mesh += 1
        else:
            kw["size"] = [float(v) for v in g["size"]] + [0.0] * (3 - len(g["size"]))
        if g.get("material") and materials and g["material"] in materials:
            kw["rgba"] = [float(v) for v in materials[g["material"]]]
        elif g.get("rgba"):
            kw["rgba"] = [float(v) for v in g["rgba"]]
        if role == "visual":
            kw.update(contype=0, conaffinity=0, group=int(g.get("group", 2)))
            return body.add_geom(**kw)
        kw["group"] = int(g.get("group", 3))
        geom = body.add_geom(**kw)
        geom.contype, geom.conaffinity = bits
        c = self.contact_of(g, bool(g.get("rest", rest_default)))
        geom.condim = int(c["condim"])
        geom.friction = [float(v) for v in c["friction"]]
        geom.solref = [float(v) for v in c["solref"]]
        geom.solimp = [float(v) for v in c["solimp"]]
        if "priority" in c:
            geom.priority = int(c["priority"])
        if "margin" in c:
            geom.margin = float(c["margin"])
        if "gap" in c:
            geom.gap = float(c["gap"])
        return geom

    def add_joint(self, body, jd):
        import mujoco
        name = self.unique(jd["name"], "joint")
        jt = jd.get("type", "hinge")
        if jt not in ("hinge", "slide"):
            raise SceneError("joint %r: type %r (hinge or slide; a free root comes from the assembly's mount)" % (name, jt))
        rng = [float(v) for v in jd["range"]]
        if len(rng) != 2 or not all(math.isfinite(v) for v in rng) or rng[0] >= rng[1]:
            raise SceneError("joint %r: range %r" % (name, jd["range"]))
        if "actuator" in jd:
            fl, dmp, arm = CF.joint_params(self.cfg, jd["actuator"])
        else:
            fl, dmp, arm = float(jd["frictionloss"]), float(jd.get("damping", 0.0)), float(jd.get("armature", 0.0))
        jf = self.P["joint_friction_impedance"]
        j = body.add_joint(name=name, type=getattr(mujoco.mjtJoint, "mjJNT_" + jt.upper()),
                           axis=[float(v) for v in jd["axis"]], pos=[float(v) for v in jd.get("pos", [0.0, 0.0, 0.0])])
        j.range = rng
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        ref = float(jd.get("ref", 0.0))       # the joint's value at the given body poses (MuJoCo's ref: qpos0 = ref)
        if not (math.isfinite(ref) and rng[0] - 1e-12 <= ref <= rng[1] + 1e-12):
            raise SceneError("joint %r: ref %r outside its range" % (name, jd.get("ref")))
        j.ref = ref
        j.frictionloss = fl
        j.armature = arm
        try:
            j.damping = dmp
        except TypeError:                     # MuJoCo 3.14's MjsJoint.damping is a 3-vector (the first entry is the damping)
            j.damping = np.array([dmp, 0.0, 0.0])
        j.solref_friction = [float(v) for v in jf["solref_friction"]]
        j.solimp_friction = [float(v) for v in jf["solimp_friction"]]
        self.intended[name] = (rng[0], rng[1])
        self.joint_params[name] = (fl, dmp, arm, [float(v) for v in jf["solref_friction"]], [float(v) for v in jf["solimp_friction"]])
        return name

    def add_assembly(self, a, T_world_cad, bits):
        import mujoco
        aid = a["id"]
        kind = a.get("kind")
        mount = a.get("mount")
        if (kind == "base") != (mount == "fixed"):
            raise SceneError("assembly %r: the base is mounted 'fixed', modules and props 'free'" % aid)
        pl = a.get("placement") or {"frame": "world"}
        T_asm = (T_world_cad if pl.get("frame") == "cad" else np.eye(4)) @ T_of(pl)
        bodies = a.get("bodies") or []
        if not bodies or bodies[0].get("parent") is not None:
            raise SceneError("assembly %r: the first body must be the root (parent null)" % aid)
        made = {}
        Tw = {}
        self.bodies_of[aid] = []
        for k, bd in enumerate(bodies):
            name = self.unique(bd["name"], "body")
            Tb = T_asm @ T_of(bd.get("pose"))
            par = bd.get("parent")
            if k > 0 and par not in made:
                raise SceneError("body %r: parent %r not defined before it" % (name, par))
            if par is None:
                if k > 0:
                    raise SceneError("assembly %r has two roots" % aid)
                rel = Tb
                parent = self.spec.worldbody
            else:
                rel = inv_T(Tw[par]) @ Tb
                parent = made[par]
            body = parent.add_body(name=name, pos=rel[:3, 3].tolist(), quat=quat_wxyz(rel[:3, :3]).tolist())
            if par is None and mount == "free":
                fj = self.unique(aid + "/free", "joint")
                body.add_freejoint(name=fj)
                self.free_joint_of[aid] = fj
            if bd.get("joint"):
                if par is None:
                    raise SceneError("body %r: the root carries no hinge" % name)
                self.add_joint(body, bd["joint"])
            ine = bd.get("inertial")
            static = (par is None and mount == "fixed")
            if ine:
                mass = float(ine["mass"])
                if not (mass > 0 and math.isfinite(mass)):
                    raise SceneError("body %r: mass %r" % (name, mass))
                body.explicitinertial = True
                body.mass = mass
                body.ipos = [float(v) for v in ine.get("pos", [0.0, 0.0, 0.0])]
                if "fullinertia" in ine:
                    body.fullinertia = [float(v) for v in ine["fullinertia"]]
                else:
                    body.iquat = [float(v) for v in ine.get("quat", [1.0, 0.0, 0.0, 0.0])]
                    body.inertia = [max(float(v), 1e-12) for v in ine["diaginertia"]]
            elif not static:
                raise SceneError("body %r: an inertial is required (only the fixed base root may omit it)" % name)
            for g in bd.get("geoms") or []:
                self.add_geom(body, g, bits, materials=a.get("materials"))
            made[name] = body
            Tw[name] = Tb
            self.owners[name] = aid
            self.bodies_of[aid].append(name)
        for jn, q in (a.get("delivered_q") or {}).items():
            if jn not in self.intended:
                raise SceneError("assembly %r: delivered_q names %r, not one of its joints" % (aid, jn))
            lo, hi = self.intended[jn]
            if not (lo - 1e-12 <= float(q) <= hi + 1e-12):
                raise SceneError("assembly %r: delivered %s = %r outside its range [%r, %r]" % (aid, jn, q, lo, hi))
            self.delivered[jn] = float(q)
        names = self.bodies_of[aid]
        if a.get("self_collide", True) is False:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    self.excludes.append((names[i], names[j]))
        for b1, b2 in a.get("exclude_pairs") or []:
            self.excludes.append((b1, b2))
        return T_asm

    def add_fixture(self, fx, T_world_cad):
        fid = fx["id"]
        pl = fx.get("placement") or {"frame": "world"}
        T = (T_world_cad if pl.get("frame") == "cad" else np.eye(4)) @ T_of(pl)
        name = self.unique(fid, "body")
        body = self.spec.worldbody.add_body(name=name, pos=T[:3, 3].tolist(), quat=quat_wxyz(T[:3, :3]).tolist())
        aff = BIT_ROBOT | BIT_MODULE | BIT_PROP | (BIT_BASE if fx.get("collide_with_base", True) else 0)
        for g in fx.get("geoms") or []:
            self.add_geom(body, g, (BIT_FIXTURE, aff), rest_default=bool(fx.get("rest_surface", False)))
        self.owners[name] = fid
        self.bodies_of[fid] = [name]
        return T


def _station_fixtures(station):
    """the floor (a plane at the pedestal's foot) and the UR's pedestal (a cylinder, r 0.11 m, from 1 mm above the
    floor to 1 mm below the UR's mounting face)"""
    h = float(station["pedestal_m"])
    ur = [float(v) for v in station.get("ur_base_pos_m", [0.0, 0.0, 0.0])]
    fz = floor_z(station)
    s = float(station.get("floor_half_size_m", 4.0))
    return [
        {"id": "floor", "role": "floor", "placement": {"frame": "world", "pos": [0.0, 0.0, fz]},
         "geoms": [{"name": "floor_plane", "type": "plane", "size": [s, s, 0.1], "rgba": [0.12, 0.12, 0.12, 1.0]}]},
        {"id": "pedestal", "role": "pedestal", "placement": {"frame": "world", "pos": [ur[0], ur[1], ur[2] - h / 2.0]},
         "geoms": [{"name": "pedestal_column", "type": "cylinder", "size": [float(station["pedestal_radius_m"]), h / 2.0 - 0.001],
                    "rgba": [0.35, 0.36, 0.38, 1.0]}]},
    ]


def build_cell(data, cfg=None, arrays=None, data_dir=None, robot=True, robot_models_dir=None, monitor=True, log=None,
               settle=True, spec_hook=None):
    """Compile the cell from scene data (module docstring) and return a ready `Cell` (the arm at its home posture, the
    gripper open, every assembly at its placement and delivered joint state, settled for physics.settle_s and the
    clock reset to 0 when `settle`). `data["welds"]` = [{"name", "body1", "body2", "active": false}]: weld equalities
    with the configuration's weld impedance (the coupling's locks: one inactive weld per mate candidate, as the
    precedent); `spec_hook(spec, names)` is called just before the compile (for anything else to add)."""
    import mujoco
    log = log or (lambda s: None)
    cfg = CF.load_config() if cfg is None else CF.check_config(cfg)
    if data.get("schema") != SCHEMA:
        raise SceneError("data schema %r, this builder reads %s" % (data.get("schema"), SCHEMA))
    P = cfg["physics"]
    tick = CF.tick_s(cfg)
    spec = mujoco.MjSpec()
    spec.modelname = str(data.get("name") or "humanoid_assembly_cell")
    apply_physics(spec, P)
    B = _Builder(spec, cfg, arrays or {}, data_dir)
    rnames = None
    models_dir = None
    if robot:
        from ..assembly_rov import gripper_wide as GW
        from ..assembly_rov import ur_robot as UR
        models_dir = robot_models_dir or UR.ROBOT_MODELS
        st = cfg["station"]
        rnames = GW.add_ur_wide(spec, {"model": cfg["robot"]["model"], "pos": list(st["ur_base_pos_m"]),
                                       "yaw_deg": float(st["ur_yaw_deg"]), "models_dir": models_dir,
                                       "impratio": float(P["impratio"]),
                                       "design": {"solref": list(P["pad_solref"]), "condim": int(P["pad_condim"]),
                                                  "friction": list(P["pad_friction"])}})
        apply_physics(spec, P)
        for what, els in (("body", spec.bodies), ("geom", spec.geoms), ("joint", spec.joints), ("mesh", spec.meshes),
                          ("camera", spec.cameras)):
            for el in els:
                if el.name:
                    B.names.setdefault(what, set()).add(el.name)
        # the wrist camera's vertical field of the cameras block (hfov 80 deg at
        # 640 x 480) set on the robot's model camera BEFORE the compile (the vendored gripper_wide.py gives 70 deg and is
        # not edited); asserted after the compile (wrist_fovy below)
        wrist_fovy = ((cfg.get("cameras") or {}).get("mounts") or {}).get("wrist", {}).get("fovy_deg")
        if wrist_fovy is not None and rnames.get("wrist_camera"):
            hit = [c for c in spec.cameras if c.name == rnames["wrist_camera"]]
            if len(hit) != 1:
                raise SceneError("the robot's wrist camera %r is not in the model" % rnames["wrist_camera"])
            hit[0].fovy = float(wrist_fovy)
    base = data.get("base")
    if base is not None:
        if "datum" not in base:
            raise SceneError("the base needs its datum (waist_axis_point_m, sole_plane_z_m) for the station")
        T_world_cad = station_transform(cfg["station"], base["datum"])
    else:
        T_world_cad = station_transform(cfg["station"], {"waist_axis_point_m": [0.0, 0.0, 0.0], "sole_plane_z_m": 0.0})
    fixtures = _station_fixtures(cfg["station"]) + list(data.get("fixtures") or [])
    floor_fixtures, fixture_ids = [], []
    for fx in fixtures:
        B.add_fixture(fx, T_world_cad)
        fixture_ids.append(fx["id"])
        if fx.get("role") == "floor":
            floor_fixtures.append(fx["id"])
    base_id = None
    if base is not None:
        base_id = base["id"]
        B.add_assembly(base, T_world_cad, (BIT_BASE, BIT_ROBOT | BIT_MODULE | BIT_PROP))
    module_ids, prop_ids = [], []
    for a in data.get("modules") or []:
        if a.get("kind") != "module":
            raise SceneError("modules[]: %r is not kind 'module'" % a.get("id"))
        B.add_assembly(a, T_world_cad, (BIT_MODULE, ALL_BITS))
        module_ids.append(a["id"])
    for a in data.get("props") or []:
        if a.get("kind") != "prop":
            raise SceneError("props[]: %r is not kind 'prop'" % a.get("id"))
        B.add_assembly(a, T_world_cad, (BIT_PROP, ALL_BITS))
        prop_ids.append(a["id"])
    ids = [base_id] + module_ids + prop_ids + fixture_ids
    if len(set(i for i in ids if i is not None)) != len([i for i in ids if i is not None]) or "robot" in ids:
        raise SceneError("assembly / fixture ids must be unique and not 'robot': %r" % ids)
    for c in data.get("cameras") or []:
        name = B.unique(c["name"], "camera")
        pos = np.asarray(c["pos"], float)
        if "lookat" in c:
            z = pos - np.asarray(c["lookat"], float)
            z /= np.linalg.norm(z)
            up = np.asarray(c.get("up", [0.0, 1.0, 0.0]), float)
            x = np.cross(up, z)
            if np.linalg.norm(x) < 1e-9:
                x = np.array([1.0, 0.0, 0.0])
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            q = quat_wxyz(np.column_stack([x, y, z]))
        else:
            q = np.asarray(c.get("quat", [1.0, 0.0, 0.0, 0.0]), float)
        spec.worldbody.add_camera(name=name, pos=pos.tolist(), quat=[float(v) for v in q], fovy=float(c.get("fovy_deg", 45.0)))
    for b1, b2 in B.excludes:
        ex = spec.add_exclude()
        ex.bodyname1, ex.bodyname2 = b1, b2
    welds = []
    for w in data.get("welds") or []:
        name = B.unique(w["name"], "equality")
        for bn in (w["body1"], w["body2"]):
            if bn not in B.names.get("body", set()):
                raise SceneError("weld %r: no body %r" % (name, bn))
        e = spec.add_equality(name=name, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1=w["body1"], name2=w["body2"])
        e.active = bool(w.get("active", False))
        # anchor at body2's origin, the relative pose computed by the compiler from the build pose (the precedent's
        # data [0] * 10 + [1]; the coupling writes eq_data at lock time)
        e.data = [0.0] * 10 + [1.0]
        e.solref = [float(v) for v in P["weld_solref"]]
        e.solimp = [float(v) for v in P["weld_solimp"]]
        welds.append(name)
    if spec_hook is not None:
        spec_hook(spec, B.names)
    m = spec.compile()
    ranges = check_ranges(m, B.intended, attached_intended(models_dir) if robot else None,
                          tol=float(cfg["joint_model"]["range_tolerance_rad"]))
    n_params = check_joint_params(m, B.joint_params)
    phys = check_physics(m, P, tick, rnames)
    wrist_fovy_compiled = None
    if robot and rnames.get("wrist_camera"):
        wrist_fovy_compiled = float(m.cam_fovy[m.camera(rnames["wrist_camera"]).id])
        if wrist_fovy is not None and abs(wrist_fovy_compiled - float(wrist_fovy)) > 1e-9:
            raise SceneError("the wrist camera's compiled vertical field %r deg is not the configuration's %r deg"
                             % (wrist_fovy_compiled, wrist_fovy))
    index = CellIndex(m, rnames, B.owners, floor_fixtures, module_ids, prop_ids, fixture_ids, base_id)
    cell = Cell(spec, m, cfg, index, rnames, T_world_cad, B, data, log)
    cell.build_report = {"ranges": {k: v for k, v in ranges.items() if k != "rows"}, "range_rows": ranges["rows"],
                         "joint_params_checked": n_params, "physics": phys, "meshes": B.n_mesh,
                         "nbody": int(m.nbody), "ngeom": int(m.ngeom), "njnt": int(m.njnt), "nv": int(m.nv),
                         "index": index.summary(), "excludes": len(B.excludes), "welds": welds,
                         "wrist_camera_fovy_deg": wrist_fovy_compiled}
    if monitor:
        from .monitor import SafetyMonitor
        cell.monitor = SafetyMonitor(m, cell.d, index, cfg["safety"], tick)
    if settle:
        cell.settle(float(P["settle_s"]))
    log("[cell] built: %d bodies, %d geoms, %d joints (%d ranges checked)" % (m.nbody, m.ngeom, m.njnt, ranges["n_checked"]))
    return cell


# ====================================================================================================== the cell
class Cell(object):
    """The compiled cell and its control loop. `tick_one()` = one 10 ms control tick: the controller's tick (commanded
    configuration, gripper ctrl and force register), the tick's physics steps with the computed torque before each,
    mj_forward, the safety monitor (every tick), the hooks. A server routes every tick through itself
    by replacing `tick_one` (the precedent's pattern)."""

    def __init__(self, spec, m, cfg, index, robot, T_world_cad, builder, data, log):
        import mujoco
        self.mj = mujoco
        self.spec, self.m, self.cfg, self.index, self.robot = spec, m, cfg, index, robot
        self.d = mujoco.MjData(m)
        self.T_world_cad = T_world_cad
        self.tick_s = CF.tick_s(cfg)
        self.tick_steps = int(cfg["physics"]["tick_steps"])
        self.delivered = dict(builder.delivered)
        self.free_joint_of = dict(builder.free_joint_of)
        self.bodies_of = {k: list(v) for k, v in builder.bodies_of.items()}
        self.joint_params = dict(builder.joint_params)
        self.data = data
        self.log = log
        self.monitor = None
        self.hooks = []
        # callables run before every physics step's collision and once more before the tick's mj_forward (the
        # contact clusters' guard, clusters_h.ContactClusters.pre_step); empty = the code path of record
        self.pre_step_hooks = []
        self.clusters = None
        # a pace lever: skip the first physics step's mj_step1 of a tick when every input of it is as the tick's mj_forward
        # left it (off by default; the episode sets it from the configuration's "performance" block)
        self.skip_first_step1 = False
        self.n_step1_skipped = 0
        self._fp = None
        self.n_tick = 0
        self.ctrl = None
        self._apply_delivered()
        if robot:
            self._init_robot()
        mujoco.mj_forward(m, self.d)

    def _apply_delivered(self):
        m, d = self.m, self.d
        for jn, q in self.delivered.items():
            d.qpos[m.jnt_qposadr[m.joint(jn).id]] = q

    def _init_robot(self):
        from ..assembly_rov import gripper_wide as GW
        m, d, rb = self.m, self.d, self.robot
        q0 = np.array(self.cfg["robot"]["home_q"], float)
        for jn, q in zip(rb["joints"], q0):
            d.qpos[m.jnt_qposadr[m.joint(jn).id]] = q
        GW.set_opening(m, d, GW.MAX_WIDTH_M, prefix=rb["prefix"])
        self.mj.mj_forward(m, d)
        ct = GW.WideController(m, d, force_N=float(self.cfg["gripper"]["force_N_default"]), tick_s=self.tick_s,
                               prefix=rb["prefix"], tcp_site=rb["tcp_site"])
        ct.q_cmd = q0.copy()
        ct.q_goal = q0.copy()
        ct.command_width(ct.max_width)
        ct.width_cmd = ct.width_goal = ct.max_width
        self.mj.mj_forward(m, d)
        ct.write_ctrl()
        ct.apply_force()
        self.ctrl = ct

    # ------------------------------------------------------------------ ticks
    def tick_physics(self):
        """one control tick of the controller and the physics only (no monitor, no hooks: the settle before t = 0).
        With pre-step hooks (cell.pre_step_hooks: the contact clusters' guard) the controller's step_physics loop
        is written out here call for call (mj_step1, the controller's torque when it recomputes it every step, mj_step2)
        with the hooks before every step and once more before the tick's mj_forward; with none, the code path of
        record (the vendored controller's step_physics)."""
        pre = self.pre_step_hooks
        skip = self.skip_first_step1
        if self.ctrl is not None:
            self.ctrl.tick()
            if not pre and not skip:
                self.ctrl.step_physics(self.tick_steps)
            else:
                ct, m, d, mj = self.ctrl, self.m, self.d, self.mj
                for k in range(self.tick_steps):
                    for h in pre:
                        h()
                    if k == 0 and skip and self._same_as_last_forward():
                        # the position and velocity stages of this state are the ones the tick's mj_forward computed
                        # (every input of mj_step1 unchanged since: _same_as_last_forward); mj_step1 would recompute
                        # them bit for bit
                        self.n_step1_skipped += 1
                    else:
                        mj.mj_step1(m, d)
                    if ct.torque_every_step:
                        ct.torque()
                    mj.mj_step2(m, d)
        else:
            for _ in range(self.tick_steps):
                for h in pre:
                    h()
                self.mj.mj_step(self.m, self.d)
        for h in pre:
            h()
        self.mj.mj_forward(self.m, self.d)
        if skip:
            self._fingerprint()

    # the pace lever "skip_first_step1": the inputs of the position and velocity stages, taken right after the tick's
    # mj_forward and compared before the next tick's first physics step (any hook, server call, test or package that
    # changed one of them between the two makes the step run mj_step1 as of record)
    _FP_DATA = ("qpos", "qvel", "act", "mocap_pos", "mocap_quat", "eq_active")
    _FP_MODEL = ("geom_contype", "geom_conaffinity", "body_contype", "body_conaffinity", "geom_margin", "geom_gap",
                 "body_margin", "eq_data", "eq_solref", "eq_solimp", "geom_size", "geom_pos", "geom_quat",
                 "jnt_range", "opt.timestep")

    def _fingerprint(self):
        m, d = self.m, self.d
        fp = [np.array(getattr(d, k)).copy() for k in self._FP_DATA]
        fp += [np.array(getattr(m, k)).copy() for k in self._FP_MODEL if "." not in k]
        fp.append(float(m.opt.timestep))
        fp.append(float(d.time))
        self._fp = fp

    def _same_as_last_forward(self):
        fp = self._fp
        if fp is None:
            return False
        m, d = self.m, self.d
        i = 0
        for k in self._FP_DATA:
            if not np.array_equal(getattr(d, k), fp[i]):
                return False
            i += 1
        for k in self._FP_MODEL:
            if "." in k:
                continue
            if not np.array_equal(getattr(m, k), fp[i]):
                return False
            i += 1
        return fp[i] == float(m.opt.timestep) and fp[i + 1] == float(d.time)

    def tick_one(self):
        self.tick_physics()
        events = self.monitor.tick() if self.monitor is not None else []
        self.n_tick += 1
        for h in self.hooks:
            h()
        return events

    def settle(self, seconds):
        for _ in range(int(round(float(seconds) / self.tick_s))):
            self.tick_physics()
        self.mj.mj_forward(self.m, self.d)
        self.d.time = 0.0
        self.n_tick = 0
        if self.monitor is not None:
            self.monitor.reset()

    # ------------------------------------------------------------------ helpers
    def body_T(self, name):
        b = self.m.body(name).id
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[b].reshape(3, 3)
        T[:3, 3] = self.d.xpos[b]
        return T

    def joint_q(self, name):
        return float(self.d.qpos[self.m.jnt_qposadr[self.m.joint(name).id]])

    def save_model(self, path):
        """the cell as MJCF (mj_saveLastXML is empty for an MjSpec model)"""
        with open(path, "w") as f:
            f.write(self.spec.to_xml())
