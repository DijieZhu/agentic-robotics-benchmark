"""rd_info.grade: the grader of the robot-description stage (payload-safe: it reads a GT folder passed by the caller and
holds no instance number; tests/verify.py runs it on the photographed arm's GT).

    ref = load_reference(gt_dir)                 # a GT folder: robot.xml (+ assets/), calibration.json, calibration_gt.json
    cases = make_cases(ref, rng, cfg)            # per-run draws (OS entropy in verify): bus readings, targets, jaw readings
    rewards, stats, details = grade(sub_dir, ref, cases, cfg, log=print)

Everything is compared in the WORLD frame through the composite map bus -> pose: at a hidden bus reading b the
submission is evaluated at q = sign x b + zero_offset_rad of ITS calibration.json and the GT at its own, so body frames,
mesh origins and the zero convention are free.

Gate `model_loads` (rd_info.gate, public rules) multiplies every criterion. Criteria (weights DEFAULT_WEIGHTS):
  fk_tcp            200 readings: 0.60 L(tcp position; 0.5, 30 mm) + 0.25 L(approach axis; 0.25, 10 deg) + 0.15 L(closing
                    axis; 0.25, 10 deg)
  fk_axes           the same readings, joints shoulder_pitch .. jaw: 0.5 L(angle between the axis lines; 0.25, 10 deg) +
                    0.5 L(distance of the GT hub point from the submitted axis line; 0.5, 20 mm). Per link the mean of its
                    driving joint is the kinematic factor kf (base 1; the yaw axis item is a stat: it is fixed by the frame)
  downstream_reach  48 targets (GT tool poses at collision-free GT readings in the envelope): the grader's IK on the
                    submitted model -> bus through its calibration -> ticks -> clamped to the GT firmware limits -> executed
                    on the GT arm: 0.7 L(position; 1.0, 30 mm) + 0.3 L(approach; 0.3, 10 deg); IK failure or a GT contact
                    deeper than 1 mm scores 0
  gripper           5 jaw readings x 8 readings: 0.5 L(tcp-to-jaw_tip distance; 0.3, 10 mm) + 0.5 L(jaw_tip position;
                    0.5, 30 mm)
  limits_actuators  per joint: 0.6 lin(IoU of the submitted range in bus angles with the GT firmware range; 0.80 -> 0,
                    0.99 -> 1) + 0.4 x the pass fraction of the actuator rules (contract.ACTUATOR_BANDS)
  inertial          per link: the geometric mean of L(mass; 3 %, 30 %), L(world CoM at 16 readings; 1, 20 mm) and L(world
                    inertia about the CoM, Frobenius relative; 15 %, 100 %), x kf; mean over links x L(total mass; 1 %, 10 %)
                    (1.1.0; 1.0.x multiplied the three bands: `inertial.combine` "product")
  collision         per link at 32 readings: 0.4 lin(coverage of the GT surface within 2 mm; 0.60 -> 0, 0.92 -> 1) + 0.3
                    lin(p95 bloat outside the GT solid; 15 -> 0, 4 -> 1 mm) + 0.3 clean (share of 64 GT-operating readings
                    without a non-adjacent self-penetration > 2 mm; credited when the link's coverage > 0.5), x kf; mean
L(e; full, zero) = clamp(ln(zero / e) / ln(zero / full), 0, 1) (1 for e <= full).
"""

import json
import math
import os
import signal
import time

import numpy as np

try:
    from . import contract as C
    from . import gate as G
except ImportError:      # vendored flat (the grader's copy and the payload): rd_info_contract.py, rd_info_gate.py
    import rd_info_contract as C  # type: ignore
    import rd_info_gate as G  # type: ignore

GRADER_VERSION = "rd-info-grade/1.1.0"
GATES = ["model_loads"]
CRITERIA = ["fk_tcp", "fk_axes", "downstream_reach", "gripper", "limits_actuators", "inertial", "collision"]
DEFAULT_WEIGHTS = {"fk_tcp": 0.30, "fk_axes": 0.15, "downstream_reach": 0.20, "gripper": 0.10, "limits_actuators": 0.05,
                   "inertial": 0.10, "collision": 0.10}
AXIS_JOINTS = ["shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll", "jaw"]

DEFAULT_CONFIG = {
    "n_fk": 200, "n_targets": 48, "n_jaw": 5, "n_jaw_configs": 8, "n_inertial": 16, "n_collision": 32, "n_bloat": 8,
    "n_clean": 64,
    "fk_tcp": {"pos_mm": [0.5, 30.0], "approach_deg": [0.25, 10.0], "closing_deg": [0.25, 10.0], "w": [0.60, 0.25, 0.15]},
    "fk_axes": {"angle_deg": [0.25, 10.0], "dist_mm": [0.5, 20.0], "w": [0.5, 0.5]},
    "downstream": {"pos_mm": [1.0, 30.0], "approach_deg": [0.3, 10.0], "w": [0.7, 0.3], "contact_tol_mm": 1.0,
                   "envelope": {"r_m": [0.12, 0.30], "z_m": [0.02, 0.25], "azimuth_deg": 100.0},
                   "ik": {"restarts": 24, "max_iters": 100, "tol_pos_m": 1e-4, "tol_rot_deg": 0.1,
                          "fallback_tol_pos_m": 0.003, "fallback_tol_rot_deg": 2.0}},
    "gripper": {"dist_mm": [0.3, 10.0], "jaw_tip_mm": [0.5, 30.0], "w": [0.5, 0.5]},
    "limits": {"iou": [0.80, 0.99], "w": [0.6, 0.4]},
    "inertial": {"mass_rel": [0.03, 0.30], "com_mm": [1.0, 20.0], "inertia_rel": [0.15, 1.0], "total_rel": [0.01, 0.10],
                 "combine": "geomean"},
    "collision": {"samples": 600, "inset_mm": 0.5, "cover_mm": 2.0, "coverage": [0.60, 0.92], "bloat_mm": [15.0, 4.0],
                  "bloat_samples": 600, "clean_tol_mm": 2.0, "clean_min_coverage": 0.5, "w": [0.4, 0.3, 0.3],
                  "gt_dense_samples": 20000},
    "gt_sampling": {"max_candidates": 20000, "table_tol_mm": 0.0},
    "criterion_timeout_s": 300.0,
}


def _deep_update(base, over):
    out = json.loads(json.dumps(base))
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def config(over=None):
    return _deep_update(DEFAULT_CONFIG, over)


# ----------------------------------------------------------------------------- bands
def L(e, full, zero):
    """Log band: 1 for e <= full, 0 for e >= zero, ln(zero / e) / ln(zero / full) between."""
    e = float(e)
    if not math.isfinite(e):
        return 0.0
    if e <= full:
        return 1.0
    if e >= zero:
        return 0.0
    return float(math.log(zero / e) / math.log(zero / full))


def Lv(e, full, zero):
    e = np.asarray(e, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(zero / np.maximum(e, 1e-300)) / math.log(zero / full)
    out = np.where(e <= full, 1.0, out)
    out = np.where(e >= zero, 0.0, out)
    out = np.where(np.isfinite(e), out, 0.0)
    return np.clip(out, 0.0, 1.0)


def lin(x, zero, full):
    """Linear band from `zero` (0) to `full` (1), either direction."""
    x = float(x)
    if not math.isfinite(x):
        return 0.0
    t = (x - zero) / (full - zero)
    return float(min(1.0, max(0.0, t)))


def halton(n, dims, shift=None, skip=20):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37][:dims]
    out = np.zeros((n, dims))
    for d, p in enumerate(primes):
        for i in range(n):
            k, f, r = i + 1 + skip, 1.0, 0.0
            while k > 0:
                f /= p
                r += f * (k % p)
                k //= p
            out[i, d] = r
    if shift is not None:
        out = (out + np.asarray(shift, float)[None, :dims]) % 1.0
    return out


# ----------------------------------------------------------------------------- a compiled arm + its calibration
def _angle_deg(a, b):
    c = float(np.clip(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300), -1.0, 1.0))
    return math.degrees(math.acos(c))


def _line_angle_deg(a, b):
    c = abs(float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300)))
    return math.degrees(math.acos(min(1.0, c)))


class Arm(object):
    """A compiled robot model with its calibration: world quantities at bus readings (the composite map)."""

    def __init__(self, model, calibration):
        import mujoco
        self.mj = mujoco
        self.m = model
        self.d = mujoco.MjData(model)
        self.cal = calibration["joints"]
        self.jid = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in C.JOINTS}
        self.qadr = {j: int(model.jnt_qposadr[self.jid[j]]) for j in C.JOINTS}
        self.sid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in C.SITES}
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        self.bid = {b: names.index(b) for b in C.BODIES}
        self.groups = G.body_groups(model)
        self.members = {b: [i for i in range(model.nbody) if self.groups[i] == b] for b in C.BODIES}

    def q_of(self, bus):
        return np.array([C.q_from_bus(bus[k], self.cal[j]) for k, j in enumerate(C.JOINTS)])

    def set_q(self, q):
        self.d.qpos[:] = 0.0
        for k, j in enumerate(C.JOINTS):
            self.d.qpos[self.qadr[j]] = float(q[k])
        self.mj.mj_kinematics(self.m, self.d)

    def set_bus(self, bus):
        self.set_q(self.q_of(bus))

    def site(self, s):
        i = self.sid[s]
        return np.array(self.d.site_xpos[i]), np.array(self.d.site_xmat[i]).reshape(3, 3)

    def axis(self, j):
        i = self.jid[j]
        return np.array(self.d.xanchor[i]), np.array(self.d.xaxis[i])

    def body_T(self, b):
        i = self.bid[b]
        T = np.eye(4)
        T[:3, :3] = np.array(self.d.xmat[i]).reshape(3, 3)
        T[:3, 3] = self.d.xpos[i]
        return T

    def link_inertia(self, b):
        """(mass, CoM world, inertia about the CoM in world) of the link group b at the current state."""
        m = self.m
        mass, mc = 0.0, np.zeros(3)
        parts = []
        for i in self.members[b]:
            mi = float(m.body_mass[i])
            if mi <= 0:
                continue
            ci = np.array(self.d.xipos[i])
            Ri = np.array(self.d.ximat[i]).reshape(3, 3)
            Ii = Ri @ np.diag(np.asarray(m.body_inertia[i], float)) @ Ri.T
            parts.append((mi, ci, Ii))
            mass += mi
            mc += mi * ci
        if mass <= 0:
            return 0.0, np.full(3, np.nan), np.full((3, 3), np.nan)
        com = mc / mass
        I = np.zeros((3, 3))
        for mi, ci, Ii in parts:
            dd = ci - com
            I += Ii + mi * (np.dot(dd, dd) * np.eye(3) - np.outer(dd, dd))
        return mass, com, I


# ----------------------------------------------------------------------------- geometry helpers
def geom_trimesh(model, g):
    """Collision shape of geom g as a trimesh in the geom's own frame (mesh geoms collide as convex hulls)."""
    import mujoco
    import trimesh
    t = int(model.geom_type[g])
    s = np.asarray(model.geom_size[g], float)
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        V = np.asarray(model.mesh_vert[va:va + vn], float)
        try:
            return trimesh.convex.convex_hull(V)
        except Exception:  # noqa: BLE001
            return None
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * s)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=3, radius=s[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(height=2 * s[1], radius=s[0], count=(24, 24))
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=s[0], height=2 * s[1], sections=32)
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        m.apply_scale(s)
        return m
    return None


def visual_trimesh(model, g):
    """A visual mesh geom's triangle mesh (geom frame), not convexified."""
    import trimesh
    mid = int(model.geom_dataid[g])
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    V = np.asarray(model.mesh_vert[va:va + vn], float)
    F = np.asarray(model.mesh_face[fa:fa + fn], int)
    return trimesh.Trimesh(V, F, process=False)


def _geom_to_root(arm, g, root_body):
    """4x4 transform of geom g's frame in the frame of `root_body` (arm state at q = 0: bodies of one group are welded)."""
    Rg = np.array(arm.d.geom_xmat[g]).reshape(3, 3)
    pg = np.array(arm.d.geom_xpos[g])
    Tb = arm.body_T(root_body)
    T = np.eye(4)
    T[:3, :3] = Tb[:3, :3].T @ Rg
    T[:3, 3] = Tb[:3, :3].T @ (pg - Tb[:3, 3])
    return T


class Pieces(object):
    """Convex pieces (in one link frame) with a plane test and exact point distances."""

    def __init__(self, meshes):
        self.meshes = [mm for mm in meshes if mm is not None and len(mm.faces) >= 4]
        self.planes = []
        for mm in self.meshes:
            n = np.asarray(mm.face_normals, float)
            o = np.asarray(mm.triangles[:, 0, :], float)
            self.planes.append((n, np.einsum("ij,ij->i", o, n)))

    def __len__(self):
        return len(self.meshes)

    def lower_bound(self, pts):
        """Per point and piece: max plane distance (<= 0 inside, a lower bound of the outside distance)."""
        out = np.empty((len(pts), len(self.meshes)))
        for k, (n, d) in enumerate(self.planes):
            out[:, k] = (pts @ n.T - d[None, :]).max(axis=1)
        return out

    def inside_any(self, pts, tol=1e-9):
        if not self.meshes:
            return np.zeros(len(pts), bool)
        return (self.lower_bound(pts) <= tol).any(axis=1)

    def distance(self, pts, cap):
        """Distance of each point to the union of the pieces (0 inside), exact below `cap`, `cap` or more above it."""
        import trimesh
        pts = np.asarray(pts, float)
        if not self.meshes:
            return np.full(len(pts), np.inf)
        lb = self.lower_bound(pts)
        best = np.where(lb.min(axis=1) <= 0.0, 0.0, np.inf)
        for k, mm in enumerate(self.meshes):
            need = (best > 0) & (lb[:, k] < cap) & (lb[:, k] < best)
            if not need.any():
                continue
            idx = np.nonzero(need)[0]
            _cp, dist, _tri = trimesh.proximity.closest_point(mm, pts[idx])
            best[idx] = np.minimum(best[idx], np.asarray(dist, float))
        return np.where(np.isfinite(best), best, np.maximum(lb.min(axis=1), cap))


def link_pieces(arm, geoms_by_group):
    """{link: Pieces in the link's own frame} from the collidable geoms of each link group (state q = 0)."""
    arm.set_q(np.zeros(6))
    out = {}
    for b in C.BODIES:
        meshes = []
        for g in geoms_by_group.get(b, []):
            mm = geom_trimesh(arm.m, g)
            if mm is None:
                continue
            mm = mm.copy()
            mm.apply_transform(_geom_to_root(arm, g, b))
            meshes.append(mm)
        out[b] = Pieces(meshes)
    return out


def union_surface_samples(pieces, n, seed):
    """Samples of the outer surface of a union of convex pieces (points inside another piece dropped), link frame."""
    import trimesh
    if not len(pieces):
        return np.zeros((0, 3))
    areas = np.array([mm.area for mm in pieces.meshes])
    counts = np.maximum(1, np.round(n * 1.6 * areas / max(areas.sum(), 1e-12)).astype(int))
    pts = []
    for k, (mm, c) in enumerate(zip(pieces.meshes, counts)):
        p, _f = trimesh.sample.sample_surface(mm, int(c), seed=int(seed) + k)
        others = Pieces([q for i, q in enumerate(pieces.meshes) if i != k])
        if len(others):
            keep = ~(others.lower_bound(p) < -1e-6).any(axis=1)
            p = p[keep]
        pts.append(p)
    P = np.vstack(pts) if pts else np.zeros((0, 3))
    if len(P) > n:
        r = np.random.default_rng(int(seed))
        P = P[r.choice(len(P), n, replace=False)]
    return P


# ----------------------------------------------------------------------------- self-collision (GT and submission)
def adjacent_pairs():
    s = set()
    for b, p in C.PARENT.items():
        if p != "world":
            s.add((p, b))
            s.add((b, p))
    return s


class Collider(object):
    """Non-adjacent self-penetration and table penetration of a model's collidable geoms (geometric: mj_geomDistance;
    contact excludes, contype masks and the file layout cannot hide a pair)."""

    def __init__(self, arm):
        self.arm = arm
        m = arm.m
        self.geoms = G.collision_geoms_by_group(m, arm.groups)
        self.glist = sorted(g for b, gs in self.geoms.items() if b != "world" for g in gs)
        self.gbody = {g: arm.groups[int(m.geom_bodyid[g])] for g in self.glist}
        adj = adjacent_pairs()
        n = len(self.glist)
        self.cand = np.zeros((n, n), bool)
        for i in range(n):
            for k in range(i + 1, n):
                bi, bk = self.gbody[self.glist[i]], self.gbody[self.glist[k]]
                if bi != bk and (bi, bk) not in adj:
                    self.cand[i, k] = True
        self.rb = np.asarray(m.geom_rbound, float)[self.glist] if n else np.zeros(0)
        self.moving = [g for g in self.glist if self.gbody[g] != "base"]
        self.local_pts = {}
        for g in self.moving:
            mm = geom_trimesh(m, g)
            self.local_pts[g] = np.asarray(mm.vertices, float) if mm is not None else np.zeros((1, 3))
        self._fromto = np.zeros(6)

    def self_penetration(self):
        """Deepest non-adjacent penetration (m, >= 0) at the current state."""
        mj = self.arm.mj
        if not self.glist:
            return 0.0
        P = np.asarray(self.arm.d.geom_xpos, float)[self.glist]
        dist = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        close = self.cand & (dist <= self.rb[:, None] + self.rb[None, :])
        pen = 0.0
        for i, k in zip(*np.nonzero(close)):
            sd = float(mj.mj_geomDistance(self.arm.m, self.arm.d, self.glist[i], self.glist[k], 0.0, self._fromto))
            if sd < 0:
                pen = max(pen, -sd)
        return pen

    def table_penetration(self):
        """Deepest point of a moving link's collision geometry below the table plane z = 0 (m, >= 0)."""
        low = 0.0
        d = self.arm.d
        for g in self.moving:
            R = np.array(d.geom_xmat[g]).reshape(3, 3)
            z = (self.local_pts[g] @ R.T)[:, 2] + d.geom_xpos[g][2]
            low = min(low, float(z.min()))
        return -low


# ----------------------------------------------------------------------------- the reference (GT arm)
class Reference(object):
    """The GT arm of one instance: compiled model, exact calibration, firmware limits, collision, surface samples."""

    def __init__(self, gt_dir, cfg=None):
        import mujoco
        cfg = config(cfg)
        self.gt_dir = gt_dir
        with open(os.path.join(gt_dir, "calibration.json")) as f:
            cal = json.load(f)
        with open(os.path.join(gt_dir, "calibration_gt.json")) as f:
            cgt = json.load(f)
        self.model = mujoco.MjModel.from_xml_path(os.path.join(gt_dir, "robot.xml"))
        self.arm = Arm(self.model, cal)
        self.cal = cal["joints"]
        self.ticks = {j: (int(cgt["joints"][j]["firmware_min_ticks"]), int(cgt["joints"][j]["firmware_max_ticks"])) for j in C.JOINTS}
        self.bus_lo = np.array([C.bus_from_ticks(self.ticks[j][0]) for j in C.JOINTS])
        self.bus_hi = np.array([C.bus_from_ticks(self.ticks[j][1]) for j in C.JOINTS])
        self.collider = Collider(self.arm)
        self._samples = None
        self._pieces = None
        self._dense = None
        self.cfg = cfg

    # firmware: what the real servos do with a commanded tick
    def clamp_ticks(self, ticks):
        return np.array([min(max(int(t), self.ticks[j][0]), self.ticks[j][1]) for t, j in zip(ticks, C.JOINTS)])

    def contacts_ok(self, tol_m=0.0):
        """(self penetration, table penetration) at the current state of the GT arm."""
        return self.collider.self_penetration(), self.collider.table_penetration()

    def surface(self):
        """{link: (points (n, 3), normals)} of the GT link surface (visual meshes: printed parts + servos), link frame,
        inset along the inward normal (collision coverage), and dense clouds for the bloat distance."""
        if self._samples is not None:
            return self._samples, self._dense
        import mujoco
        import trimesh
        m = self.model
        a = self.arm
        a.set_q(np.zeros(6))
        cfg = self.cfg["collision"]
        out, dense = {}, {}
        for bi, b in enumerate(C.BODIES):
            meshes = []
            for g in range(m.ngeom):
                if a.groups[int(m.geom_bodyid[g])] != b:
                    continue
                if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                    continue
                if m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0:
                    continue                       # visual meshes only
                mm = visual_trimesh(m, g)
                mm.apply_transform(_geom_to_root(a, g, b))
                meshes.append(mm)
            mesh = trimesh.util.concatenate(meshes)
            p, fid = trimesh.sample.sample_surface(mesh, int(cfg["samples"]), seed=1000 + bi)
            nrm = np.asarray(mesh.face_normals, float)[fid]
            out[b] = (p - nrm * float(cfg["inset_mm"]) * 1e-3, nrm)
            pd, _ = trimesh.sample.sample_surface(mesh, int(cfg["gt_dense_samples"]), seed=2000 + bi)
            dense[b] = pd
        self._samples, self._dense = out, dense
        return out, dense

    def pieces(self):
        if self._pieces is None:
            self._pieces = link_pieces(self.arm, self.collider.geoms)
        return self._pieces


def load_reference(gt_dir, cfg=None):
    return Reference(gt_dir, cfg)


# ----------------------------------------------------------------------------- the per-run draws
def _sample_bus(ref, u):
    return ref.bus_lo[None, :] + u * (ref.bus_hi - ref.bus_lo)[None, :]


def _ik_reaches(ref, tgt, cfg, seed):
    """The grader's IK (the downstream settings) solves the target on the GT model to its tight tolerance."""
    try:
        from arm import ik_reference as IK
    except ImportError:  # vendored flat
        import ik_reference as IK  # type: ignore
    ik = cfg["downstream"]["ik"]
    m = ref.model
    import mujoco
    d = mujoco.MjData(m)
    q = IK.solve(m, d, {"pos": tgt["pos"], "approach": tgt["approach"]}, seed=seed, restarts=int(ik["restarts"]),
                 max_iters=int(ik["max_iters"]), tol_pos=float(ik["tol_pos_m"]), tol_rot_deg=float(ik["tol_rot_deg"]),
                 joints=C.ARM_JOINTS, site="tcp")
    return q is not None


def make_cases(ref, rng, cfg=None):
    """Hidden draws of one grading run (a numpy Generator seeded from OS entropy in verify). Every configuration is a
    bus reading inside the GT firmware box at which the GT arm has no self / table contact."""
    cfg = config(cfg)
    a = ref.arm
    shift = rng.random(6)
    shift2 = rng.random(6)
    ik_seed = int(rng.integers(1, 2 ** 31 - 1))
    n_need = int(cfg["n_fk"]) + int(cfg["n_clean"])
    U = halton(int(cfg["gt_sampling"]["max_candidates"]), 6, shift=shift)
    good = []
    tried = 0
    for u in U:
        tried += 1
        b = _sample_bus(ref, u[None, :])[0]
        a.set_bus(b)
        sp, tp = ref.contacts_ok()
        if sp > 0.0 or tp > float(cfg["gt_sampling"]["table_tol_mm"]) * 1e-3:
            continue
        good.append(b)
        if len(good) >= n_need:
            break
    good = np.array(good)
    if len(good) < n_need:
        raise RuntimeError("only {} collision-free GT readings in {} candidates".format(len(good), tried))
    order = rng.permutation(len(good))
    good = good[order]
    fk = good[:int(cfg["n_fk"])]
    clean = good[int(cfg["n_fk"]):n_need]
    # downstream targets: GT tool poses in the envelope at collision-free readings
    env = cfg["downstream"]["envelope"]
    targets = []
    U2 = halton(int(cfg["gt_sampling"]["max_candidates"]), 6, shift=shift2)
    for u in U2:
        b = _sample_bus(ref, u[None, :])[0]
        a.set_bus(b)
        p, R = a.site("tcp")
        r = math.hypot(p[0], p[1])
        az = math.degrees(math.atan2(-p[0], p[1]))
        if not (env["r_m"][0] <= r <= env["r_m"][1] and env["z_m"][0] <= p[2] <= env["z_m"][1] and abs(az) <= env["azimuth_deg"]):
            continue
        sp, tp = ref.contacts_ok()
        if sp > 0.0 or tp > 0.0:
            continue
        tgt = {"bus": b.tolist(), "pos": p.tolist(), "approach": R[:, 2].tolist()}
        if not _ik_reaches(ref, tgt, cfg, ik_seed + len(targets)):
            continue                              # a target the grader's controller cannot solve on the arm itself: not drawn
        targets.append(tgt)
        if len(targets) >= int(cfg["n_targets"]):
            break
    if len(targets) < int(cfg["n_targets"]):
        raise RuntimeError("only {} downstream targets".format(len(targets)))
    # jaw readings: stratified over the GT jaw firmware range
    nj = int(cfg["n_jaw"])
    jlo, jhi = ref.bus_lo[5], ref.bus_hi[5]
    jaw = np.array([jlo + (jhi - jlo) * (k + rng.random()) / nj for k in range(nj)])
    jaw_cfg = rng.choice(len(fk), int(cfg["n_jaw_configs"]), replace=False)
    return {"fk": fk, "clean": clean, "targets": targets, "jaw": jaw, "jaw_configs": jaw_cfg,
            "inertial": rng.choice(len(fk), int(cfg["n_inertial"]), replace=False),
            "collision": rng.choice(len(fk), int(cfg["n_collision"]), replace=False),
            "ik_seed": ik_seed, "tried": tried}


# ----------------------------------------------------------------------------- guard
class _Timeout(Exception):
    pass


def _timed(fn, seconds, name, log):
    """Run fn() under a wall-clock guard (SIGALRM in the main thread; plain call elsewhere)."""
    import threading
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        return fn()

    def handler(signum, frame):
        raise _Timeout(name)
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ----------------------------------------------------------------------------- criteria
def crit_fk(sub, ref, cases, cfg):
    """fk_tcp, fk_axes (+ the per-link kinematic factor) and their raw errors."""
    c1, c2 = cfg["fk_tcp"], cfg["fk_axes"]
    ep, ea, ex = [], [], []
    ang = {j: [] for j in C.JOINTS}
    dst = {j: [] for j in C.JOINTS}
    for b in cases["fk"]:
        ref.arm.set_bus(b)
        sub.set_bus(b)
        pg, Rg = ref.arm.site("tcp")
        pa, Ra = sub.site("tcp")
        ep.append(1e3 * float(np.linalg.norm(pa - pg)))
        ea.append(_angle_deg(Ra[:, 2], Rg[:, 2]))
        ex.append(_angle_deg(Ra[:, 0], Rg[:, 0]))
        for j in C.JOINTS:
            ag, ug = ref.arm.axis(j)
            aa, ua = sub.axis(j)
            ua = ua / max(np.linalg.norm(ua), 1e-300)
            ang[j].append(_line_angle_deg(ua, ug))
            dst[j].append(1e3 * float(np.linalg.norm(np.cross(ag - aa, ua))))
    ep, ea, ex = np.array(ep), np.array(ea), np.array(ex)
    s_tcp = c1["w"][0] * Lv(ep, *c1["pos_mm"]) + c1["w"][1] * Lv(ea, *c1["approach_deg"]) + c1["w"][2] * Lv(ex, *c1["closing_deg"])
    item = {j: c2["w"][0] * Lv(np.array(ang[j]), *c2["angle_deg"]) + c2["w"][1] * Lv(np.array(dst[j]), *c2["dist_mm"]) for j in C.JOINTS}
    fk_axes = float(np.mean([item[j].mean() for j in AXIS_JOINTS]))
    kf = {"base": 1.0}
    for b, j in C.JOINT_OF_BODY.items():
        kf[b] = float(item[j].mean())
    stats = {"fk_tcp_err_mean_mm": float(ep.mean()), "fk_tcp_err_p95_mm": float(np.percentile(ep, 95)),
             "fk_tcp_err_max_mm": float(ep.max()), "fk_approach_err_mean_deg": float(ea.mean()),
             "fk_closing_err_mean_deg": float(ex.mean())}
    for j in C.JOINTS:
        stats["axis_angle_mean_deg_" + j] = float(np.mean(ang[j]))
        stats["axis_dist_mean_mm_" + j] = float(np.mean(dst[j]))
        stats["axis_item_" + j] = float(item[j].mean())
    for b in C.BODIES:
        stats["kf_" + b] = kf[b]
    stats["kf_min"] = float(min(kf.values()))
    return float(s_tcp.mean()), fk_axes, kf, stats


def crit_gripper(sub, ref, cases, cfg):
    c = cfg["gripper"]
    e1, e2 = [], []
    for k in cases["jaw_configs"]:
        b0 = np.array(cases["fk"][int(k)], float)
        for jb in cases["jaw"]:
            b = b0.copy()
            b[5] = float(jb)
            ref.arm.set_bus(b)
            sub.set_bus(b)
            tg, _ = ref.arm.site("tcp")
            jg, _ = ref.arm.site("jaw_tip")
            ta, _ = sub.site("tcp")
            ja, _ = sub.site("jaw_tip")
            e1.append(1e3 * abs(float(np.linalg.norm(ja - ta)) - float(np.linalg.norm(jg - tg))))
            e2.append(1e3 * float(np.linalg.norm(ja - jg)))
    e1, e2 = np.array(e1), np.array(e2)
    s = c["w"][0] * Lv(e1, *c["dist_mm"]) + c["w"][1] * Lv(e2, *c["jaw_tip_mm"])
    return float(s.mean()), {"gripper_dist_err_mean_mm": float(e1.mean()), "gripper_jaw_tip_err_mean_mm": float(e2.mean())}


def crit_downstream(sub, ref, cases, cfg):
    """The grader's controller: IK on the submitted model, the command through its calibration to ticks, the GT servos
    clamp to their firmware limits, the GT arm executes."""
    try:
        from arm import ik_reference as IK
    except ImportError:  # vendored flat
        import ik_reference as IK  # type: ignore
    c = cfg["downstream"]
    ik = c["ik"]
    scores, eps, eas = [], [], []
    n_fail, n_contact, n_fallback = 0, 0, 0
    home_bus = np.zeros(6)
    for i, t in enumerate(cases["targets"]):
        tgt = {"pos": t["pos"], "approach": t["approach"]}
        seed = int(cases["ik_seed"]) + i
        q = IK.solve(sub.m, sub.d, tgt, seed=seed, restarts=int(ik["restarts"]), max_iters=int(ik["max_iters"]),
                     tol_pos=float(ik["tol_pos_m"]), tol_rot_deg=float(ik["tol_rot_deg"]), joints=C.ARM_JOINTS, site="tcp")
        if q is None:
            q = IK.solve(sub.m, sub.d, tgt, seed=seed, restarts=int(ik["restarts"]), max_iters=int(ik["max_iters"]),
                         tol_pos=float(ik["fallback_tol_pos_m"]), tol_rot_deg=float(ik["fallback_tol_rot_deg"]),
                         joints=C.ARM_JOINTS, site="tcp", polish=False)
            if q is not None:
                n_fallback += 1
        if q is None:
            n_fail += 1
            scores.append(0.0)
            continue
        ticks = []
        for k, j in enumerate(C.ARM_JOINTS):
            ticks.append(C.ticks_from_bus(C.bus_from_q(q[k], sub.cal[j])))
        ticks.append(C.ticks_from_bus(home_bus[5]))
        ticks = ref.clamp_ticks(ticks)
        bus = np.array([C.bus_from_ticks(tk) for tk in ticks])
        ref.arm.set_bus(bus)
        p, R = ref.arm.site("tcp")
        e_p = 1e3 * float(np.linalg.norm(p - np.asarray(t["pos"])))
        e_a = _angle_deg(R[:, 2], np.asarray(t["approach"]))
        sp, tp = ref.contacts_ok()
        eps.append(e_p)
        eas.append(e_a)
        if max(sp, tp) > float(c["contact_tol_mm"]) * 1e-3:
            n_contact += 1
            scores.append(0.0)
            continue
        scores.append(c["w"][0] * L(e_p, *c["pos_mm"]) + c["w"][1] * L(e_a, *c["approach_deg"]))
    st = {"downstream_ik_fail": float(n_fail), "downstream_gt_contact": float(n_contact), "downstream_ik_fallback": float(n_fallback),
          "downstream_err_mean_mm": float(np.mean(eps)) if eps else float("nan"),
          "downstream_err_p95_mm": float(np.percentile(eps, 95)) if eps else float("nan"),
          "downstream_approach_err_mean_deg": float(np.mean(eas)) if eas else float("nan")}
    return float(np.mean(scores)), st


def crit_limits(sub, ref, cfg):
    import mujoco
    c = cfg["limits"]
    B = C.ACTUATOR_BANDS
    m = sub.m
    per, st = [], {}
    for k, j in enumerate(C.JOINTS):
        ji = sub.jid[j]
        if int(m.jnt_limited[ji]):
            lo, hi = float(m.jnt_range[ji, 0]), float(m.jnt_range[ji, 1])
        else:
            lo, hi = -math.pi * 8, math.pi * 8
        b1, b2 = sorted([C.bus_from_q(lo, sub.cal[j]), C.bus_from_q(hi, sub.cal[j])])
        g1, g2 = ref.bus_lo[k], ref.bus_hi[k]
        inter = max(0.0, min(b2, g2) - max(b1, g1))
        union = max(b2, g2) - min(b1, g1)
        iou = inter / union if union > 0 else 0.0
        s_range = lin(iou, c["iou"][0], c["iou"][1])
        rules = []
        ai = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_" + j)
        if ai >= 0:
            kp = float(m.actuator_gainprm[ai, 0])
            kv = -float(m.actuator_biasprm[ai, 2])
            cr = [float(v) for v in m.actuator_ctrlrange[ai]]
            rules.append(bool(m.actuator_ctrllimited[ai]) and abs(cr[0] - lo) <= B["ctrlrange_tol_rad"] and abs(cr[1] - hi) <= B["ctrlrange_tol_rad"])
            fr = [float(v) for v in m.actuator_forcerange[ai]]
            f0, f1 = B["forcerange_fraction_of_stall"]
            rules.append(bool(m.actuator_forcelimited[ai]) and f0 * C.STALL_NM - 1e-9 <= -fr[0] <= f1 * C.STALL_NM + 1e-9
                         and f0 * C.STALL_NM - 1e-9 <= fr[1] <= f1 * C.STALL_NM + 1e-9)
            rules.append(B["kp"][0] <= kp <= B["kp"][1])
            rules.append(B["kv"][0] <= kv <= B["kv"][1])
        else:
            rules += [False] * 4
        dof = int(m.jnt_dofadr[ji])
        rules.append(B["armature"][0] <= float(m.dof_armature[dof]) <= B["armature"][1])
        rules.append(B["damping"][0] <= float(m.dof_damping[dof]) <= B["damping"][1])
        rules.append(B["frictionloss"][0] <= float(m.dof_frictionloss[dof]) <= B["frictionloss"][1])
        pf = float(np.mean(rules))
        per.append(c["w"][0] * s_range + c["w"][1] * pf)
        st["limits_iou_" + j] = float(iou)
        st["actuator_rules_pass_" + j] = pf
    return float(np.mean(per)), st


def crit_inertial(sub, ref, cases, cfg, kf):
    c = cfg["inertial"]
    per, st = [], {}
    idx = [int(i) for i in cases["inertial"]]
    tot_a, tot_g = 0.0, 0.0
    for b in C.BODIES:
        ref.arm.set_bus(cases["fk"][idx[0]])
        sub.set_bus(cases["fk"][idx[0]])
        mg, _, _ = ref.arm.link_inertia(b)
        ma, _, _ = sub.link_inertia(b)
        tot_a += ma
        tot_g += mg
        rel_m = abs(ma - mg) / max(mg, 1e-12)
        ec, ei = [], []
        for i in idx:
            ref.arm.set_bus(cases["fk"][i])
            sub.set_bus(cases["fk"][i])
            _, cg, Ig = ref.arm.link_inertia(b)
            _, ca, Ia = sub.link_inertia(b)
            ec.append(1e3 * float(np.linalg.norm(ca - cg)) if ma > 0 else float("inf"))
            ei.append(float(np.linalg.norm(Ia - Ig) / max(np.linalg.norm(Ig), 1e-15)) if ma > 0 else float("inf"))
        s_m = L(rel_m, *c["mass_rel"])
        s_c = float(np.mean(Lv(np.array(ec), *c["com_mm"])))
        s_i = float(np.mean(Lv(np.array(ei), *c["inertia_rel"])))
        prod = s_m * s_c * s_i
        comb = c.get("combine", "product")
        if comb not in ("product", "geomean"):
            raise ValueError("inertial.combine must be product or geomean")
        s = (prod if comb == "product" else prod ** (1.0 / 3.0)) * kf.get(b, 1.0)
        per.append(s)
        st["mass_rel_err_" + b] = float(rel_m)
        st["com_err_mean_mm_" + b] = float(np.mean(ec))
        st["inertia_rel_err_mean_" + b] = float(np.mean(ei))
        st["inertial_link_" + b] = float(s)
    rel_t = abs(tot_a - tot_g) / max(tot_g, 1e-12)
    st["total_mass_rel_err"] = float(rel_t)
    st["total_mass_kg_submitted"] = float(tot_a)
    return float(np.mean(per)) * L(rel_t, *c["total_rel"]), st


def crit_collision(sub, ref, cases, cfg, kf):
    c = cfg["collision"]
    samples, dense = ref.surface()
    gt_pieces = ref.pieces()
    sub_geoms = G.collision_geoms_by_group(sub.m, sub.groups)
    sub_pieces = link_pieces(sub, sub_geoms)
    from scipy.spatial import cKDTree
    trees = {b: cKDTree(dense[b]) for b in C.BODIES}
    idx = [int(i) for i in cases["collision"]]
    cov = {b: [] for b in C.BODIES}
    bloat_pts = {b: [] for b in C.BODIES}
    sub_surf = {b: union_surface_samples(sub_pieces[b], int(c["bloat_samples"]), 3000 + k) for k, b in enumerate(C.BODIES)}
    cap = float(c["cover_mm"]) * 1e-3
    for n, i in enumerate(idx):
        bus = cases["fk"][i]
        ref.arm.set_bus(bus)
        sub.set_bus(bus)
        for b in C.BODIES:
            Tg = ref.arm.body_T(b)
            Ta = sub.body_T(b)
            T = np.linalg.inv(Ta) @ Tg                       # GT link frame -> submitted link frame
            P = samples[b][0] @ T[:3, :3].T + T[:3, 3]
            if len(sub_pieces[b]):
                dd = sub_pieces[b].distance(P, cap * 1.5)
                cov[b].append(float(np.mean(dd <= cap)))
            else:
                cov[b].append(0.0)
            if n < int(cfg["n_bloat"]):
                S = sub_surf[b]
                if len(S):
                    Ti = np.linalg.inv(T)
                    Q = S @ Ti[:3, :3].T + Ti[:3, 3]            # submitted surface in the GT link frame
                    ins = gt_pieces[b].inside_any(Q)
                    dq, _ = trees[b].query(Q)
                    dq = np.where(ins, 0.0, dq)
                    bloat_pts[b].append(dq)
    # clean: non-adjacent self-penetration of the submitted collision model at GT-operating readings
    col = Collider(sub)
    n_clean, worst = 0, 0.0
    for bus in cases["clean"]:
        sub.set_bus(bus)
        pen = col.self_penetration()
        worst = max(worst, pen)
        n_clean += int(pen <= float(c["clean_tol_mm"]) * 1e-3)
    clean = n_clean / float(len(cases["clean"]))
    per, st = [], {"collision_clean_fraction": float(clean), "collision_worst_self_penetration_mm": 1e3 * worst}
    for b in C.BODIES:
        cv = float(np.mean(cov[b])) if cov[b] else 0.0
        if bloat_pts[b]:
            bl = 1e3 * float(np.percentile(np.concatenate(bloat_pts[b]), 95))
        else:
            bl = float("inf")
        s_cov = lin(cv, c["coverage"][0], c["coverage"][1])
        s_bl = lin(bl, c["bloat_mm"][0], c["bloat_mm"][1]) if math.isfinite(bl) else 0.0
        s_cl = clean if cv > float(c["clean_min_coverage"]) else 0.0
        s = (c["w"][0] * s_cov + c["w"][1] * s_bl + c["w"][2] * s_cl) * kf.get(b, 1.0)
        per.append(s)
        st["coverage_" + b] = cv
        st["bloat_p95_mm_" + b] = bl
        st["collision_link_" + b] = float(s)
    return float(np.mean(per)), st


# ----------------------------------------------------------------------------- grade
def zero_convention_dev_deg(sub, ref):
    """Per joint |q_submitted| at the bus reading where the GT stands at q = 0 (0: the same zero convention)."""
    out = []
    for j in C.JOINTS:
        b0 = C.bus_from_q(0.0, ref.cal[j])
        out.append(abs(math.degrees(C.q_from_bus(b0, sub.cal[j]))))
    return out


def grade(sub_dir, ref, cases, cfg=None, log=None, gate=None):
    """(rewards {gate + criteria}, stats, details). `gate` = a precomputed gate_model_loads result (ok, det, models)."""
    cfg = config(cfg)
    log = log or (lambda s: None)
    rewards = {k: 0.0 for k in GATES + CRITERIA}
    stats, details = {"grader_version_" + GRADER_VERSION: 1.0}, {}
    t0 = time.time()
    ok, det, models = gate if gate is not None else G.gate_model_loads(sub_dir)
    stats["gate_s"] = time.time() - t0
    details["model_loads"] = det.get("violations", [])
    for k in ("mesh_faces_total", "collision_geoms_total", "submission_bytes"):
        if k in det:
            stats[k] = float(det[k])
    stats["urdf_present"] = 1.0 if os.path.isfile(os.path.join(sub_dir, "robot.urdf")) else 0.0
    stats["estimate_present"] = 1.0 if os.path.isfile(os.path.join(sub_dir, "estimate.json")) else 0.0
    if not ok:
        log("gate model_loads failed: " + "; ".join(details["model_loads"][:4]))
        return rewards, stats, details
    rewards["model_loads"] = 1.0
    sub = Arm(models["robot"], models["calibration"])
    try:                                         # a stat never stops the grading
        stats["zero_convention_dev_deg_mean"] = float(np.mean(zero_convention_dev_deg(sub, ref)))
    except Exception:  # noqa: BLE001
        stats["zero_convention_dev_deg_mean"] = float("nan")
    tout = float(cfg["criterion_timeout_s"])
    kf = {b: 0.0 for b in C.BODIES}

    def run(name, fn):
        t = time.time()
        try:
            return _timed(fn, tout, name, log)
        except _Timeout:
            details[name] = ["criterion exceeded its {} s guard: 0".format(tout)]
            stats["timeout_" + name] = 1.0
        except Exception as exc:  # noqa: BLE001 - a submission that crashes a criterion earns 0 there
            import traceback
            details[name] = ["criterion raised on this submission: " + traceback.format_exc()[-600:]]
            stats["exception_" + name] = 1.0
            log("criterion {} raised: {}".format(name, exc))
        finally:
            stats["time_s_" + name] = time.time() - t
        return None

    r = run("fk", lambda: crit_fk(sub, ref, cases, cfg))
    if r is not None:
        rewards["fk_tcp"], rewards["fk_axes"], kf, st = r
        stats.update(st)
    r = run("downstream_reach", lambda: crit_downstream(sub, ref, cases, cfg))
    if r is not None:
        rewards["downstream_reach"], st = r
        stats.update(st)
    r = run("gripper", lambda: crit_gripper(sub, ref, cases, cfg))
    if r is not None:
        rewards["gripper"], st = r
        stats.update(st)
    r = run("limits_actuators", lambda: crit_limits(sub, ref, cfg))
    if r is not None:
        rewards["limits_actuators"], st = r
        stats.update(st)
    r = run("inertial", lambda: crit_inertial(sub, ref, cases, cfg, kf))
    if r is not None:
        rewards["inertial"], st = r
        stats.update(st)
    r = run("collision", lambda: crit_collision(sub, ref, cases, cfg, kf))
    if r is not None:
        rewards["collision"], st = r
        stats.update(st)
    for c in CRITERIA:
        rewards[c] = float(min(1.0, max(0.0, rewards[c])))
    stats["grade_s"] = time.time() - t0
    return rewards, stats, details


def overall(rewards, weights=None):
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    gate = 1.0
    for g in GATES:
        gate *= float(rewards.get(g, 0.0))
    return gate * sum(w[c] * float(rewards[c]) for c in CRITERIA) / sum(w[c] for c in CRITERIA)
