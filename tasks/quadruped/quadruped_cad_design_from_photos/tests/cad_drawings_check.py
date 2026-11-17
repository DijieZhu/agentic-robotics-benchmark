"""checks: the deterministic CAD / kinematics / drawing checks of the "arm links from orthographic views" task.

Shipped twice, byte-identical: in the agent's home (practice_checker.py runs it on the given drawings and the public
cases) and inside tests/ (the grader runs it with the hidden reference). It never reads hidden material itself; the
caller passes the cases and the reference in. Lengths mm, angles rad unless a name says otherwise. CPU only.

The arm: six identical bus servos S1..S6, seven links (base, shoulder, upper_arm, lower_arm, wrist, gripper, jaw),
six revolute joints base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll, jaw; every joint's servo sits in
the PARENT link and the child rides on its horn. A link is one or more printed parts (part ids P01..P10 of the given
drawings); a part file is one STEP solid in the part's frame, oriented exactly as its drawings (= the arm frame at the
zero pose: z up, +y forward, +x = pitch axes), translated freely.

Conventions: rpy = fixed-axis XYZ (R = Rz Ry Rx); a joint origin is the child frame in the parent frame at q = 0; axis
in the child frame through the child origin (URDF); a servo occurrence pose is the servo frame (servo_standin.json)
in the link frame; the tool centre point is declared in the gripper frame (`tcp`).
"""
import hashlib
import json
import math
import os
import tempfile

import numpy as np

TESS_TOL, TESS_ANG = 0.05, 0.3
ARM_JOINTS = ["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"]
JOINT_ORDER = ARM_JOINTS + ["jaw"]
LINK_ORDER = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]
DEFAULT_STRUCTURE = {
    "joints": [
        {"name": "base_yaw", "parent": "base", "child": "shoulder", "servo": "S1", "axis_at_zero": [0, 0, 1]},
        {"name": "shoulder_pitch", "parent": "shoulder", "child": "upper_arm", "servo": "S2", "axis_at_zero": [1, 0, 0]},
        {"name": "elbow_pitch", "parent": "upper_arm", "child": "lower_arm", "servo": "S3", "axis_at_zero": [1, 0, 0]},
        {"name": "wrist_pitch", "parent": "lower_arm", "child": "wrist", "servo": "S4", "axis_at_zero": [1, 0, 0]},
        {"name": "wrist_roll", "parent": "wrist", "child": "gripper", "servo": "S5", "axis_at_zero": [0, 0, 1]},
        {"name": "jaw", "parent": "gripper", "child": "jaw", "servo": "S6", "axis_at_zero": [1, 0, 0]},
    ],
    "link_parts": {"base": ["P01", "P02"], "shoulder": ["P03", "P04"], "upper_arm": ["P05"], "lower_arm": ["P06", "P07"],
                   "wrist": ["P08"], "gripper": ["P09"], "jaw": ["P10"]},
    "disclosed_parts": ["P10"],
}
G = 9.81
MOUNTED_MIN_RETENTION = 0.5
KEEPOUT_THR = 1e-3


# ----------------------------------------------------------------------------- transforms
def rpy_to_matrix(rpy):
    r, p, y = [float(v) for v in rpy]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_to_rpy(R):
    R = np.asarray(R, dtype=float)
    sy = max(-1.0, min(1.0, -R[2, 0]))
    p = math.asin(sy)
    if abs(math.cos(p)) > 1e-9:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:
        r = 0.0
        y = math.atan2(-R[0, 1], R[1, 1])
    return [r, p, y]


def pose(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def rot_axis(axis, q):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K
    T = np.eye(4)
    T[:3, :3] = R
    return T


def inv(T):
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def apply(T, p):
    return T[:3, :3] @ np.asarray(p, dtype=float) + T[:3, 3]


def angle_between(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))


def halton(n, dims, shift=None, skip=20):
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    out = np.zeros((n, dims))
    for d in range(dims):
        b = primes[d]
        for i in range(n):
            k, f, x = i + skip, 1.0, 0.0
            while k > 0:
                f /= b
                x += f * (k % b)
                k //= b
            out[i, d] = x
    if shift is not None:
        out = (out + np.asarray(shift)[None, :dims]) % 1.0
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def line_dist(pa, da, pb):
    d = np.asarray(pb, float) - np.asarray(pa, float)
    da = np.asarray(da, float) / np.linalg.norm(da)
    return float(np.linalg.norm(d - da * float(d @ da)))


# ----------------------------------------------------------------------------- meshes
def fill_small_holes(mesh, max_loop=64):
    """Close boundary loops of at most `max_loop` edges with triangle fans (the OCC mesher leaves a few micro-gaps
    on some imported B-reps). Returns (mesh, n_loops_closed, n_loops_left)."""
    import trimesh
    edges = mesh.edges_sorted
    bidx = trimesh.grouping.group_rows(edges, require_count=1)
    if len(bidx) == 0:
        return mesh, 0, 0
    directed = mesh.edges[bidx]                    # face-directed boundary edges (a -> b)
    nxt = {}
    for a, b in directed:
        nxt.setdefault(int(a), []).append(int(b))
    used = set()
    loops = []
    for a, b in directed:
        key = (int(a), int(b))
        if key in used:
            continue
        loop = [int(a)]
        cur = int(b)
        used.add(key)
        ok = True
        for _ in range(len(directed) + 1):
            if cur == loop[0]:
                break
            loop.append(cur)
            cands = [c for c in nxt.get(cur, []) if (cur, c) not in used]
            if not cands:
                ok = False
                break
            c = cands[0]
            used.add((cur, c))
            cur = c
        if ok and cur == loop[0] and len(loop) >= 3:
            loops.append(loop)
    new_faces = []
    left = 0
    for loop in loops:
        if len(loop) > max_loop:
            left += 1
            continue
        v0 = loop[0]
        for i in range(1, len(loop) - 1):
            new_faces.append([v0, loop[i + 1], loop[i]])   # reversed relative to the boundary direction
    if not new_faces:
        return mesh, 0, len(loops)
    faces = np.vstack([mesh.faces, np.asarray(new_faces, dtype=np.int64)])
    out = trimesh.Trimesh(mesh.vertices.copy(), faces, process=True)
    return out, len(loops) - left, left


def solid_to_trimesh(solid, tol=TESS_TOL, ang=TESS_ANG):
    """Watertight trimesh of a cadquery Solid (degenerate faces dropped, micro-gaps closed). Returns (mesh, notes)."""
    import trimesh
    vs, tris = solid.tessellate(tol, ang)
    verts = np.array([v.toTuple() for v in vs], dtype=float)
    faces = np.array(tris, dtype=np.int64)
    if len(faces) == 0:
        return None, ["tessellation produced no faces"]
    mesh = trimesh.Trimesh(verts, faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    notes = []
    if not mesh.is_watertight:
        mesh, closed, left = fill_small_holes(mesh)
        if closed:
            notes.append("closed {} micro-gap loop(s) of the tessellation".format(closed))
        if left:
            notes.append("{} boundary loop(s) too large to close".format(left))
    return mesh, notes


def trimesh_to_manifold(mesh):
    import manifold3d as m3
    return m3.Manifold(m3.Mesh(vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
                               tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32)))


def manifold_transform(man, T):
    return man.transform(np.ascontiguousarray(np.asarray(T)[:3, :], dtype=np.float32))


def manifold_box(lo, hi):
    import manifold3d as m3
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    size = hi - lo
    if np.any(size <= 0):
        return None
    return m3.Manifold.cube(size.tolist(), False).translate(lo.tolist())


class Body(object):
    """One closed solid: the OCP solid, its tessellation, its manifold, mass properties."""

    def __init__(self, name, solid):
        self.name = name
        self.solid = solid
        self.errors = []
        self.notes = []
        self.mesh = None
        self.manifold = None
        from OCP.BRepCheck import BRepCheck_Analyzer
        if not BRepCheck_Analyzer(solid.wrapped).IsValid():
            self.errors.append("BRepCheck_Analyzer reports an invalid solid")
        mesh, notes = solid_to_trimesh(solid)
        self.notes += notes
        if mesh is None or len(mesh.faces) < 4:
            self.errors.append("tessellation produced no faces")
            return
        if not mesh.is_watertight:
            self.errors.append("tessellated mesh is not watertight")
        if not mesh.is_winding_consistent:
            self.errors.append("tessellated mesh has inconsistent winding")
        if mesh.volume <= 0:
            self.errors.append("tessellated mesh has non-positive volume")
        self.mesh = mesh
        if not self.errors:
            self.manifold = trimesh_to_manifold(mesh)
            if str(self.manifold.status()) != "Error.NoError":
                self.errors.append("manifold construction failed: {}".format(self.manifold.status()))
                self.manifold = None
        self.volume = float(mesh.volume) if not self.errors else 0.0
        self.centroid = np.asarray(mesh.center_mass, float) if not self.errors else np.zeros(3)
        self.bounds = np.asarray(mesh.bounds, float) if not self.errors else np.zeros((2, 3))
        # enclosed voids (inner shells of the one solid) are invisible to the drawings and to the surface probes, so they
        # count as material for the mass model: `solid_volume` is the outer shell's volume, `void_ratio` the hollowed share
        self.solid_volume, self.void_ratio = self.volume, 0.0
        if not self.errors:
            try:
                comps = mesh.split(only_watertight=False)
            except Exception:  # noqa: BLE001
                comps = []
            if len(comps) > 1:
                outer = max(abs(float(c.volume)) for c in comps)
                if outer > self.volume + 1e-9:
                    self.solid_volume = outer
                    self.void_ratio = 1.0 - self.volume / outer

    @property
    def ok(self):
        return not self.errors and self.manifold is not None

    def contains(self, points):
        """Point-in-solid on the watertight tessellation (ray parity, batched; agrees with the B-rep classifier on every
        probe of this task, which sits >= 0.5 mm from any face, and is an order of magnitude faster)."""
        pts = np.asarray(points, float).reshape(-1, 3)
        if len(pts) == 0:
            return np.zeros(0, dtype=bool)
        return np.asarray(self.mesh.contains(pts), dtype=bool)

    def contains_brep(self, points):
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.TopAbs import TopAbs_IN, TopAbs_ON
        from OCP.gp import gp_Pnt
        cl = BRepClass3d_SolidClassifier(self.solid.wrapped)
        out = []
        for p in np.asarray(points, float).reshape(-1, 3):
            cl.Perform(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), 1e-6)
            out.append(cl.State() in (TopAbs_IN, TopAbs_ON))
        return np.array(out, dtype=bool)


def load_step_solids(path):
    import cadquery as cq
    try:
        wp = cq.importers.importStep(path)
    except Exception as exc:  # noqa: BLE001
        return [], "STEP reader failed: {}".format(str(exc)[:200])
    try:
        solids = wp.solids().vals()
    except Exception as exc:  # noqa: BLE001
        return [], "no solids in STEP: {}".format(str(exc)[:200])
    return solids, None


# ----------------------------------------------------------------------------- servo
class Servo(object):
    """The stand-in servo: geometry + the interface probes derived from servo_standin.json."""

    def __init__(self, json_path, step_path):
        with open(json_path) as f:
            self.meta = json.load(f)
        solids, err = load_step_solids(step_path)
        if err or len(solids) != 1:
            raise RuntimeError("servo stand-in STEP unusable: {}".format(err or len(solids)))
        self.body = Body("servo", solids[0])
        if not self.body.ok:
            raise RuntimeError("servo stand-in invalid: {}".format(self.body.errors))
        self.volume = self.body.volume
        m = self.meta
        # keep-out envelope: the stand-in with its flange outer faces pulled in by the seat tolerance (a separate STEP
        # next to the stand-in when the JSON names one; else the stand-in itself)
        self.keepout = self.body
        ko = m.get("keep_out_solid")
        if ko:
            kpath = os.path.join(os.path.dirname(step_path), ko)
            ksol, kerr = load_step_solids(kpath)
            if kerr or len(ksol) != 1:
                raise RuntimeError("servo keep-out STEP unusable: {}".format(kerr or len(ksol)))
            self.keepout = Body("servo_keepout", ksol[0])
            if not self.keepout.ok:
                raise RuntimeError("servo keep-out solid invalid: {}".format(self.keepout.errors))
        self.size = np.asarray(m["body"]["size"], float)
        self.com = np.asarray(m["centre_of_mass"], float)
        self.horn_point = np.asarray(m["horn"]["axis_point"], float)
        self.horn_dir = np.asarray(m["horn"]["axis_dir"], float)
        self.horn_face_z = float(m["horn"]["mating_face_z"])
        self.bolt_positions = np.asarray(m["horn"]["bolt_holes"]["positions_on_mating_face"], float)
        self.rear_point = np.asarray(m["rear_disc"]["axis_point"], float)
        self.rear_bottom_z = float(m["rear_disc"]["z"][0])
        self.flanges = m["flanges"]
        self.facts = m.get("actuator_facts", {})
        env = m.get("overall_envelope", m["body"])
        self.env_min = np.asarray(env["min"], float)
        self.env_max = np.asarray(env["max"], float)

    def bolt_radius(self):
        b = self.bolt_positions[0]
        return float(math.hypot(b[0] - self.horn_point[0], b[1] - self.horn_point[1]))

    def horn_probes(self, ring_r=8.0, n=12):
        """Legacy coverage ring at the horn face + 1 mm, the four bolt positions at the servo-frame angle, the centre."""
        z = self.horn_face_z + 1.0
        ring = [[self.horn_point[0] + ring_r * math.cos(a), self.horn_point[1] + ring_r * math.sin(a), z]
                for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]
        holes = [[p[0], p[1], z] for p in self.bolt_positions]
        centre = [[self.horn_point[0], self.horn_point[1], z]]
        return np.array(ring), np.array(holes), np.array(centre)

    def bolt_pattern_probes(self, side="horn", theta=0.0, ring_r=2.5, n=8):
        """The four-bolt square of the horn (or the rear disc) rotated by `theta` about the axis, probed 1 mm outboard
        of the mating face: per bolt a ring of radius `ring_r` around the bolt axis (material) and the axis point (empty);
        plus the centre point. A horn mates at any spline angle, so the caller sweeps theta over the 90 deg period."""
        if side == "horn":
            z = self.horn_face_z + 1.0
            c = self.horn_point
        else:
            z = self.rear_bottom_z - 1.0
            c = self.rear_point
        r = self.bolt_radius()
        base = math.atan2(self.bolt_positions[0][1] - self.horn_point[1], self.bolt_positions[0][0] - self.horn_point[0])
        bolts = []
        for k in range(4):
            a = base + theta + k * math.pi / 2
            bx, by = c[0] + r * math.cos(a), c[1] + r * math.sin(a)
            ring = np.array([[bx + ring_r * math.cos(t), by + ring_r * math.sin(t), z] for t in np.linspace(0, 2 * math.pi, n, endpoint=False)])
            bolts.append((ring, np.array([[bx, by, z]])))
        return bolts, np.array([[c[0], c[1], z]])

    def rear_probes(self, ring_r=8.0, n=12):
        z = self.rear_bottom_z - 1.0
        return np.array([[self.rear_point[0] + ring_r * math.cos(a), self.rear_point[1] + ring_r * math.sin(a), z]
                         for a in np.linspace(0, 2 * math.pi, n, endpoint=False)])

    def flange_probes(self, ring_r=3.0, n=12):
        """Per flange hole (top and bottom flange): (label, ring 1 mm outboard of the outer face, axis point)."""
        out = []
        for side in ("top", "bottom"):
            fl = self.flanges[side]
            z_out = float(fl["outer_face_z"])
            z = z_out + (1.0 if side == "top" else -1.0)
            for k, (hx, hy) in enumerate(fl["holes_xy"]):
                ring = [[hx + ring_r * math.cos(a), hy + ring_r * math.sin(a), z] for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]
                out.append(("{}{}".format(side[0], k), np.array(ring), np.array([[hx, hy, z]])))
        return out

    def hole_pairs(self):
        """Indices of the two flange holes of each pair (top k and bottom k share the axis)."""
        return list(range(len(self.flanges["top"]["holes_xy"])))


# ----------------------------------------------------------------------------- assembly
class Link(object):
    def __init__(self, spec):
        self.name = spec["name"]
        self.parts = [(p["part_id"], p["part_file"]) for p in spec["parts"]]
        self.mass_override = spec.get("mass_override_kg")
        self.occurrences = [(o["servo_id"], pose(o["xyz"], o["rpy"])) for o in spec.get("servo_occurrences", [])]
        self.bodies = []        # Body per part, in link frame (a part frame IS the link frame: every part of a link shares it)

    # aggregate geometry -------------------------------------------------------
    @property
    def manifold(self):
        m = self.bodies[0].manifold
        for b in self.bodies[1:]:
            m = m + b.manifold
        return m

    @property
    def volume(self):
        return float(sum(b.volume for b in self.bodies))

    @property
    def solid_volume(self):
        """Volume with enclosed voids filled (the printed material the mass model charges)."""
        return float(sum(b.solid_volume for b in self.bodies))

    @property
    def centroid(self):
        v = self.volume
        return sum(b.centroid * b.volume for b in self.bodies) / max(v, 1e-9)

    @property
    def bounds(self):
        lo = np.min([b.bounds[0] for b in self.bodies], axis=0)
        hi = np.max([b.bounds[1] for b in self.bodies], axis=0)
        return np.array([lo, hi])

    def contains(self, points):
        out = np.zeros(len(np.asarray(points).reshape(-1, 3)), dtype=bool)
        for b in self.bodies:
            out |= b.contains(points)
        return out


class Joint(object):
    def __init__(self, spec):
        self.name = spec["name"]
        self.parent = spec["parent_link"]
        self.child = spec["child_link"]
        self.origin = pose(spec["origin"]["xyz"], spec["origin"]["rpy"])
        a = np.asarray(spec["axis"], float)
        self.axis = a / (np.linalg.norm(a) if np.linalg.norm(a) > 0 else 1.0)
        self.lower = float(spec["limits"]["lower"])
        self.upper = float(spec["limits"]["upper"])
        self.driven_by = spec["driven_by"] if isinstance(spec["driven_by"], str) else str(spec["driven_by"][0])


class Chain(object):
    """The kinematic content of assembly.json (no geometry needed). q has one entry per joint (six)."""

    def __init__(self, spec):
        self.spec = spec
        self.links = dict((l["name"], Link(l)) for l in spec["links"])
        self.joints = [Joint(j) for j in spec["joints"]]
        self.root = spec["root_link"]
        self.order = [self.root] + [j.child for j in self.joints]
        self.lower = np.array([j.lower for j in self.joints])
        self.upper = np.array([j.upper for j in self.joints])
        tcp = spec["tcp"]
        self.tcp_link = tcp["link"]
        self.tcp_point = np.asarray(tcp["xyz"], float)
        a = np.asarray(tcp["approach"], float)
        self.tcp_approach = a / np.linalg.norm(a)
        self.n_arm = len(ARM_JOINTS)

    def fk(self, q):
        q = np.asarray(q, float)
        if len(q) < len(self.joints):
            q = np.concatenate([q, np.zeros(len(self.joints) - len(q))])
        T = {self.root: np.eye(4)}
        for j, qi in zip(self.joints, q):
            T[j.child] = T[j.parent] @ j.origin @ rot_axis(j.axis, qi)
        return T

    def occurrence(self, servo_id):
        for name, link in self.links.items():
            for sid, T in link.occurrences:
                if sid == servo_id:
                    return name, T
        return None, None

    def tcp(self, q, T=None):
        T = T or self.fk(q)
        Tw = T[self.tcp_link]
        return apply(Tw, self.tcp_point), Tw[:3, :3] @ self.tcp_approach

    def servo_axis_world(self, servo_id, T):
        name, Tocc = self.occurrence(servo_id)
        if name is None:
            return None, None
        return T[name] @ Tocc, name

    def joint_of_servo(self, servo_id):
        for j in self.joints:
            if servo_id == j.driven_by:
                return j
        return None

    def joint(self, name):
        for j in self.joints:
            if j.name == name:
                return j
        return None


def validate_assembly_dict(spec, schema, structure):
    """Schema + topology. Returns a list of error strings (empty = ok)."""
    import jsonschema
    errors = []
    try:
        jsonschema.validate(spec, schema)
    except jsonschema.ValidationError as exc:
        return ["assembly.json violates the schema: {}".format(exc.message[:300])]
    link_names = [l["name"] for l in spec["links"]]
    if sorted(link_names) != sorted(LINK_ORDER):
        errors.append("links must be exactly {} (got {})".format(LINK_ORDER, sorted(link_names)))
        return errors
    if spec["root_link"] != "base":
        errors.append("root link must be base")
    names = [j["name"] for j in spec["joints"]]
    if names != JOINT_ORDER:
        errors.append("joints must be exactly {} in this order (got {})".format(JOINT_ORDER, names))
        return errors
    topo = dict((j["name"], j) for j in structure["joints"])
    for j in spec["joints"]:
        t = topo[j["name"]]
        if j["parent_link"] != t["parent"] or j["child_link"] != t["child"]:
            errors.append("joint {} must connect {} -> {} (got {} -> {})".format(j["name"], t["parent"], t["child"], j["parent_link"], j["child_link"]))
        drv = j["driven_by"] if isinstance(j["driven_by"], str) else (j["driven_by"][0] if j["driven_by"] else None)
        if drv != t["servo"]:
            errors.append("joint {} must be driven by {} (got {})".format(j["name"], t["servo"], j["driven_by"]))
        if not (j["limits"]["lower"] < j["limits"]["upper"]):
            errors.append("joint {} limits must satisfy lower < upper".format(j["name"]))
        if abs(np.linalg.norm(np.asarray(j["axis"], float)) - 1.0) > 1e-3:
            errors.append("joint {} axis must be a unit vector".format(j["name"]))
    link_parts = structure["link_parts"]
    all_parts = sorted(p for ps in link_parts.values() for p in ps)
    used = []
    files = []
    for l in spec["links"]:
        pids = sorted(p["part_id"] for p in l["parts"])
        if pids != sorted(link_parts[l["name"]]):
            errors.append("link {} must consist of parts {} (got {})".format(l["name"], sorted(link_parts[l["name"]]), pids))
        used += pids
        files += [p["part_file"] for p in l["parts"]]
    if sorted(used) != all_parts:
        errors.append("every part id of {} must appear exactly once (got {})".format(all_parts, sorted(used)))
    if len(set(files)) != len(files):
        errors.append("two parts share one part file")
    carrier = {}
    for l in spec["links"]:
        for o in l.get("servo_occurrences", []):
            if o["servo_id"] in carrier:
                errors.append("servo {} placed twice".format(o["servo_id"]))
            carrier[o["servo_id"]] = l["name"]
            for v in list(o["xyz"]) + list(o["rpy"]):
                if not math.isfinite(float(v)):
                    errors.append("non-finite pose value in link {}".format(l["name"]))
    for t in structure["joints"]:
        if carrier.get(t["servo"]) != t["parent"]:
            errors.append("servo {} (joint {}) must be carried by link {} (got {})".format(t["servo"], t["name"], t["parent"], carrier.get(t["servo"])))
    if sorted(carrier) != sorted(t["servo"] for t in structure["joints"]):
        errors.append("servo occurrences must be exactly {} (got {})".format(sorted(t["servo"] for t in structure["joints"]), sorted(carrier)))
    if spec["tcp"]["link"] != "gripper":
        errors.append("tcp must be declared in the gripper link frame")
    if np.linalg.norm(np.asarray(spec["tcp"]["approach"], float)) < 1e-6:
        errors.append("tcp approach must be a non-zero vector")
    return errors


class Submission(object):
    """Loaded, validated submission. `gate_errors` non-empty means cad_valid = 0."""

    def __init__(self, sub_dir, schema, structure):
        self.dir = sub_dir
        self.gate_errors = []
        self.notes = []
        self.spec = None
        self.chain = None
        self.parts = {}          # part_id -> Body
        self.part_files = {}     # part_id -> absolute path
        path = os.path.join(sub_dir, "assembly.json")
        if not os.path.isfile(path):
            self.gate_errors.append("no assembly.json")
            return
        try:
            with open(path) as f:
                self.spec = json.load(f)
        except (OSError, ValueError) as exc:
            self.gate_errors.append("assembly.json unreadable: {}".format(exc))
            return
        if not isinstance(self.spec, dict):
            self.gate_errors.append("assembly.json is not an object")
            return
        self.gate_errors += validate_assembly_dict(self.spec, schema, structure)
        if self.gate_errors:
            return
        self.chain = Chain(self.spec)
        for name, link in self.chain.links.items():
            for pid, rel in link.parts:
                ppath = os.path.normpath(os.path.join(sub_dir, rel))
                if not ppath.startswith(os.path.normpath(sub_dir) + os.sep) or not os.path.isfile(ppath) or os.path.islink(ppath):
                    self.gate_errors.append("part {}: file {} missing (or a symlink)".format(pid, rel))
                    continue
                solids, err = load_step_solids(ppath)
                if err:
                    self.gate_errors.append("part {}: {}".format(pid, err))
                    continue
                if len(solids) != 1:
                    self.gate_errors.append("part {}: STEP must contain exactly one solid (has {})".format(pid, len(solids)))
                    continue
                body = Body(pid, solids[0])
                if not body.ok:
                    self.gate_errors.append("part {}: {}".format(pid, "; ".join(body.errors)))
                    continue
                for n in body.notes:
                    self.notes.append("{}: {}".format(pid, n))
                self.parts[pid] = body
                self.part_files[pid] = ppath
                link.bodies.append(body)

    @property
    def ok(self):
        return not self.gate_errors


# ----------------------------------------------------------------------------- scoring helpers
def score_ratio(ratio, thr):
    """1 at ratio <= thr, 0 at ratio >= 10*thr, log-linear between."""
    if ratio <= thr:
        return 1.0
    if ratio >= 10 * thr:
        return 0.0
    return max(0.0, 1.0 - math.log10(ratio / thr))


def ring_score(frac, lo=0.25, hi=0.75):
    return max(0.0, min(1.0, (float(frac) - lo) / (hi - lo)))


def band(value, full, zero):
    """1 when |value| <= full, 0 when |value| >= zero, linear between."""
    v = abs(float(value))
    if not math.isfinite(v):
        return 0.0
    if v <= full:
        return 1.0
    if v >= zero:
        return 0.0
    return (zero - v) / (zero - full)


# ----------------------------------------------------------------------------- gates and criteria
def check_cad_json_consistent(sub, servo, tol_deg=1.0, tol_mm=0.5, contact_mm=0.2):
    """Every joint axis coincides with its servo's horn axis; the child touches the parent or the servo at q = 0."""
    chain = sub.chain
    T = chain.fk(np.zeros(len(chain.joints)))
    details = []
    ok_all = True
    for j in chain.joints:
        Tc = T[j.child]
        axis_p = Tc[:3, 3]
        axis_d = Tc[:3, :3] @ j.axis
        child = chain.links[j.child]
        parent = chain.links[j.parent]
        cm = manifold_transform(child.manifold, Tc)
        pm = manifold_transform(parent.manifold, T[j.parent])
        gap = cm.min_gap(pm, 5.0)
        Tw_servo, carrier = chain.servo_axis_world(j.driven_by, T)
        good = True
        ang = off = float("nan")
        if Tw_servo is None or carrier != j.parent:
            details.append("{}: servo {} is not on the parent link".format(j.name, j.driven_by))
            good = False
        else:
            horn_p = apply(Tw_servo, servo.horn_point)
            horn_d = Tw_servo[:3, :3] @ servo.horn_dir
            ang = angle_between(horn_d, axis_d)
            ang = min(ang, 180.0 - ang)
            off = line_dist(horn_p, horn_d, axis_p)
            sm = manifold_transform(servo.body.manifold, Tw_servo)
            gap = min(gap, cm.min_gap(sm, 5.0))
            good = ang <= tol_deg and off <= tol_mm and gap <= contact_mm
        ok_all = ok_all and good
        details.append("{} ({}): axis angle {:.2f} deg, offset {:.3f} mm, gap {:.3f} mm -> {}".format(
            j.name, j.driven_by, ang, off, gap, "ok" if good else "FAIL"))
    return (1.0 if ok_all else 0.0), details


def bolt_pattern_score(link, T_servo_in_link, servo, side, steps=45):
    """Best retention of a four-bolt plate over the rotation of the bolt square about the axis (steps over the 90 deg
    period): per bolt ring_score(material fraction around the bolt axis, 1 mm outboard of the face) x empty axis; on
    the horn side the centre must be empty (centre screw / boss clearance); the rear passive disc needs no centre hole.
    Returns (score, theta_deg, detail)."""
    best = (0.0, 0.0, "no material")
    for k in range(steps):
        theta = (math.pi / 2) * k / steps
        bolts, centre = servo.bolt_pattern_probes(side, theta)
        centre_in = side == "horn" and bool(link.contains([apply(T_servo_in_link, centre[0])])[0])
        if centre_in:
            sc = 0.0
            det = "centre BLOCKED"
            if best[0] == 0.0:
                best = (0.0, math.degrees(theta), det)
        else:
            per = []
            for ring, axis_pt in bolts:
                frac = float(link.contains([apply(T_servo_in_link, p) for p in ring]).mean())
                empty = not link.contains([apply(T_servo_in_link, axis_pt[0])])[0]
                per.append(ring_score(frac) * (1.0 if empty else 0.0))
            sc = float(np.mean(per))
            det = "bolts {} at {:.0f} deg".format([round(v, 2) for v in per], math.degrees(theta))
        if sc > best[0] + 1e-9:
            best = (sc, math.degrees(theta), det)
        if sc >= 1.0:
            break
    return best


def max_empty_radius(link, T_servo_in_link, c, r_max, step=0.1, n=12):
    """Largest radius (<= r_max) of an empty ring around the axis point c (servo frame) inside the link (batched)."""
    radii = np.arange(step, r_max + 1e-9, step)
    angs = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = np.array([[c[0] + r * math.cos(a), c[1] + r * math.sin(a), c[2]] for r in radii for a in angs])
    inside = link.contains([apply(T_servo_in_link, q) for q in pts]).reshape(len(radii), n).any(axis=1)
    r_ok = 0.0
    for r, hit in zip(radii, inside):
        if hit:
            break
        r_ok = float(r)
    return r_ok


def bolt_pattern_eval(link, T_servo_in_link, servo, side, steps=45, clearance=False, r_max=1.0):
    """Retention of a four-bolt plate at the best rotation of the bolt square about the axis (steps over the 90 deg
    period), evaluated with pruning: the four bolt axes (and, on the horn side, the centre) must be empty before the
    material rings (radius 2.5 mm, 1 mm outboard of the face) are probed. Returns dict(score, theta_deg, detail,
    clearance_mm, centre_mm); the clearances (largest empty diameter at the bolt axes over a sub-degree refinement
    around the best angle, and at the centre) are computed only when `clearance` is requested."""
    thetas = [(math.pi / 2) * k / steps for k in range(steps)]
    axes, centres = [], []
    for th in thetas:
        bolts, centre = servo.bolt_pattern_probes(side, th)
        axes.append([b[1][0] for b in bolts])
        centres.append(centre[0])
    axes = np.array(axes)                  # (steps, 4, 3)
    centres = np.array(centres)            # (steps, 3)
    pts = np.concatenate([axes.reshape(-1, 3), centres])
    inside = link.contains([apply(T_servo_in_link, p) for p in pts])
    ax_in = inside[:steps * 4].reshape(steps, 4)
    ce_in = inside[steps * 4:]
    ok = ~ax_in.any(axis=1)
    if side == "horn":
        ok &= ~ce_in
    if not ok.any():
        det = "centre BLOCKED" if (side == "horn" and (~ax_in.any(axis=1)).any()) else "bolt axes blocked at every angle"
        return {"score": 0.0, "theta_deg": 0.0, "detail": det, "clearance_mm": 0.0, "centre_mm": 0.0}
    cand = [k for k in range(steps) if ok[k]]
    ring_pts = []
    for k in cand:
        bolts, _c = servo.bolt_pattern_probes(side, thetas[k])
        for ring, _a in bolts:
            ring_pts.append(ring)
    ring_pts = np.array(ring_pts)          # (len(cand)*4, n, 3)
    n_ring = ring_pts.shape[1]
    ring_in = link.contains([apply(T_servo_in_link, p) for p in ring_pts.reshape(-1, 3)]).reshape(len(cand), 4, n_ring)
    per = np.array([[ring_score(f) for f in row] for row in ring_in.mean(axis=2)])   # (len(cand), 4)
    scores = per.mean(axis=1)
    best = float(scores.max())
    kbest = [cand[i] for i in range(len(cand)) if scores[i] >= best - 1e-9]
    out = {"score": best, "theta_deg": math.degrees(thetas[kbest[0]]),
           "detail": "bolts {} at {:.0f} deg".format([round(float(v), 2) for v in per[cand.index(kbest[0])]], math.degrees(thetas[kbest[0]])),
           "clearance_mm": 0.0, "centre_mm": 0.0}
    if not clearance or best <= 0.0:
        return out
    coarse = []
    for k in kbest:
        bolts, _c = servo.bolt_pattern_probes(side, thetas[k])
        d = 2.0 * min(max_empty_radius(link, T_servo_in_link, ap[0], r_max, step=0.2, n=8) for _r, ap in bolts)
        coarse.append((d, thetas[k]))
    coarse.sort(reverse=True)
    best_d, centre_d = 0.0, 0.0
    theta0 = coarse[0][1]
    for dth in np.radians(np.arange(-1.5, 1.51, 0.5)):
        bolts, centre = servo.bolt_pattern_probes(side, theta0 + dth)
        d = 2.0 * min(max_empty_radius(link, T_servo_in_link, ap[0], r_max, step=0.1, n=8) for _r, ap in bolts)
        if d > best_d:
            best_d = d
            centre_d = 2.0 * max_empty_radius(link, T_servo_in_link, centre[0], 6.0, step=0.25, n=8)
    out["clearance_mm"], out["centre_mm"] = best_d, centre_d
    return out


def bolt_pattern_score(link, T_servo_in_link, servo, side, steps=45):
    r = bolt_pattern_eval(link, T_servo_in_link, servo, side, steps)
    return r["score"], r["theta_deg"], r["detail"]


def bolt_pattern_clearance(link, T_servo_in_link, servo, side, r_max=2.0, steps=45):
    r = bolt_pattern_eval(link, T_servo_in_link, servo, side, steps, clearance=True, r_max=r_max)
    return r["clearance_mm"], r["centre_mm"]


def check_component_fit(sub, servo, keepout_thr=KEEPOUT_THR):
    """Per servo occurrence: keep-out (servo solid vs carrier and vs driven link) x flange retention x horn plate x rear
    support. Returns (mean fit, details, {servo: retention}) where retention = keep-out x flange x horn (the gate)."""
    chain = sub.chain
    T0 = chain.fk(np.zeros(len(chain.joints)))
    per, details, retention = [], [], {}
    for lname in chain.order:
        link = chain.links[lname]
        for sid, Tocc in link.occurrences:
            sm = manifold_transform(servo.keepout.manifold, Tocc)
            vol = (sm ^ link.manifold).volume()
            ko = score_ratio(vol / servo.volume, keepout_thr)
            msg = "{} in {}: keep-out carrier {:.1f} mm3".format(sid, lname, vol)
            j = chain.joint_of_servo(sid)
            other = j.child if j is not None else None
            To = None
            if other is not None:
                To = inv(T0[other]) @ T0[lname] @ Tocc
                vol2 = (manifold_transform(servo.keepout.manifold, To) ^ chain.links[other].manifold).volume()
                ko = min(ko, score_ratio(vol2 / servo.volume, keepout_thr))
                msg += ", driven link {:.1f} mm3".format(vol2)
            # flanges: a hole counts as used when the carrier has a ring of material 1 mm outboard of the flange face
            # around the axis and the axis itself is empty (a screw path); two used holes give full retention
            hole_scores = {}
            for label, ring, axis_pt in servo.flange_probes():
                frac = float(link.contains([apply(Tocc, p) for p in ring]).mean())
                empty = not link.contains([apply(Tocc, axis_pt[0])])[0]
                s = ring_score(frac) * (1.0 if empty else 0.0)
                k = label[1:]
                hole_scores[k] = max(hole_scores.get(k, 0.0), s)
            top2 = sorted(hole_scores.values(), reverse=True)[:2]
            flange = max(0.0, min(1.0, sum(top2) / 2.0))
            msg += ", flange holes used {} -> {:.2f}".format({k: round(v, 2) for k, v in sorted(hole_scores.items())}, flange)
            horn = rear = 1.0
            if other is not None:
                obody = chain.links[other]
                horn, h_theta, h_det = bolt_pattern_score(obody, To, servo, "horn")
                rear_plate, r_theta, r_det = bolt_pattern_score(obody, To, servo, "rear")
                rear_pts = servo.rear_probes()
                f_c = float(link.contains([apply(Tocc, p) for p in rear_pts]).mean())
                seat = ring_score(f_c)
                rear = max(rear_plate, seat)
                msg += ", horn plate {} -> {:.2f}, rear plate {} -> {:.2f} / carrier seat ring {:.2f} -> {:.2f}".format(
                    h_det, horn, r_det, rear_plate, f_c, rear)
            ret = ko * flange * horn
            retention[sid] = float(ret)
            fit = ret * rear
            per.append(fit)
            details.append(msg + " -> keep-out {:.2f}, retention {:.2f}, fit {:.2f}".format(ko, ret, fit))
    return (float(np.mean(per)) if per else 0.0), details, retention


def link_collision_sets(sub, servo):
    sets = {}
    for name, link in sub.chain.links.items():
        items = [(b.manifold, b.volume) for b in link.bodies]
        for sid, Tocc in link.occurrences:
            items.append((manifold_transform(servo.body.manifold, Tocc), servo.volume))
        sets[name] = items
    return sets


def _aabb_overlap(a, b, margin=0.0):
    return all(a[i] - margin <= b[i + 3] and b[i] - margin <= a[i + 3] for i in range(3))


def operating_configs(sub, servo, unit_stream, n_wanted, ranges, table_z=-0.5):
    """Map unit points into the REFERENCE joint ranges (hidden), keep those inside the declared limits and above the
    mounting plane; the first `n_wanted` are the operating configurations. Returns (configs, examined, skipped_outside)."""
    chain = sub.chain
    sets = link_collision_sets(sub, servo)
    lo = np.array([ranges[j.name][0] for j in chain.joints])
    hi = np.array([ranges[j.name][1] for j in chain.joints])
    out = []
    examined = skipped = 0
    for u in np.asarray(unit_stream):
        examined += 1
        q = lo + u * (hi - lo)
        if np.any(q < chain.lower - 1e-9) or np.any(q > chain.upper + 1e-9):
            skipped += 1
            continue
        T = chain.fk(q)
        zmin = min(manifold_transform(m, T[n]).bounding_box()[2] for n in chain.order for m, _ in sets[n])
        if zmin >= table_z:
            out.append(q)
        if len(out) >= n_wanted:
            break
    return np.array(out).reshape(-1, len(chain.joints)), examined, skipped


def declared_range_configs(sub, servo, unit_stream, n_wanted, ranges, table_z=-0.5, margin=1e-6):
    """Configurations drawn from the DECLARED limits that lie outside the hidden operating box (at least one joint) with
    every part above the mounting plane: where the declared range exceeds the drawn arm's, it is tested for self-collision
    too. Returns (configs, examined, inside_share) where inside_share is the volume share of the declared box that lies
    inside the hidden box (product over joints of |declared ∩ hidden| / |declared|); 1.0 when nothing is declared outside."""
    chain = sub.chain
    lo_d, hi_d = np.asarray(chain.lower, float), np.asarray(chain.upper, float)
    lo_h = np.array([ranges[j.name][0] for j in chain.joints])
    hi_h = np.array([ranges[j.name][1] for j in chain.joints])
    n = len(chain.joints)
    share = 1.0
    for i in range(n):
        width = max(hi_d[i] - lo_d[i], 1e-9)
        inter = max(0.0, min(hi_d[i], hi_h[i]) - max(lo_d[i], lo_h[i]))
        share *= min(1.0, inter / width)
    if np.all(lo_d >= lo_h - margin) and np.all(hi_d <= hi_h + margin):
        return np.zeros((0, n)), 0, 1.0
    sets = link_collision_sets(sub, servo)
    out = []
    examined = 0
    for u in np.asarray(unit_stream):
        q = lo_d + u * (hi_d - lo_d)
        if np.all(q >= lo_h - margin) and np.all(q <= hi_h + margin):
            continue                      # inside the drawn arm's box: covered by the hidden draw
        examined += 1
        T = chain.fk(q)
        zmin = min(manifold_transform(m, T[k]).bounding_box()[2] for k in chain.order for m, _ in sets[k])
        if zmin >= table_z:
            out.append(q)
        if len(out) >= n_wanted:
            break
    return np.array(out).reshape(-1, n), examined, float(share)


def check_interference(sub, servo, configs, adjacent_thr=1e-2, nonadjacent_thr=1e-3):
    chain = sub.chain
    sets = link_collision_sets(sub, servo)
    names = chain.order
    totals = dict((n, sum(v for _, v in sets[n])) for n in names)
    adjacent = set()
    for j in chain.joints:
        adjacent.add((j.parent, j.child))
        adjacent.add((j.child, j.parent))
    scores = []
    worst = (1.0, "")
    for q in np.asarray(configs).reshape(-1, len(chain.joints)):
        T = chain.fk(q)
        placed = {}
        for n in names:
            placed[n] = []
            for man, vol in sets[n]:
                mw = manifold_transform(man, T[n])
                placed[n].append((mw, mw.bounding_box()))
        cfg = 1.0
        for i in range(len(names)):
            for k in range(i + 1, len(names)):
                a, b = names[i], names[k]
                thr = adjacent_thr if (a, b) in adjacent else nonadjacent_thr
                vol = 0.0
                for ma, ba in placed[a]:
                    for mb, bb in placed[b]:
                        if _aabb_overlap(ba, bb):
                            vol += (ma ^ mb).volume()
                ratio = vol / max(1e-9, min(totals[a], totals[b]))
                s = score_ratio(ratio, thr)
                if s < worst[0]:
                    worst = (s, "{}-{} overlap {:.1f} mm3 (ratio {:.2e}) at q={}".format(a, b, vol, ratio, np.round(q, 3).tolist()))
                cfg = min(cfg, s)
        scores.append(cfg)
    score = float(np.mean(scores)) if scores else 0.0
    details = ["{} configurations, {} clean, worst: {}".format(len(scores), int(sum(1 for s in scores if s >= 1.0)), worst[1] or "none")]
    return score, details, scores


def ik_reachable(chain, target_p, target_a, rng, restarts=8, iters=150, tol_p=2.0, tol_deg=3.0):
    """Damped least squares over the five arm joints (the jaw stays at 0) with joint-limit clamping and restarts."""
    n = chain.n_arm
    lo, hi = chain.lower[:n], chain.upper[:n]
    K = 40.0
    target_p = np.asarray(target_p, float)
    target_a = np.asarray(target_a, float) / np.linalg.norm(target_a)

    def error(q):
        p, a = chain.tcp(np.concatenate([q, [0.0]]))
        e = np.concatenate([target_p - p, K * np.cross(a, target_a)])
        if float(a @ target_a) < -0.9:
            e[3:] += K * np.array([0.3, 0.4, 0.5])
        return e, p, a

    for r in range(restarts):
        q = np.clip(np.zeros(n), lo, hi) if r == 0 else rng.uniform(lo, hi)
        lam = 1.0
        e, p, a = error(q)
        for _ in range(iters):
            if np.linalg.norm(target_p - p) <= tol_p and angle_between(a, target_a) <= tol_deg:
                return True
            J = np.zeros((6, n))
            h = 1e-5
            for i in range(n):
                qh = q.copy()
                qh[i] += h
                eh, _, _ = error(qh)
                J[:, i] = (e - eh) / h
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            q_new = np.clip(q + dq, lo, hi)
            e_new, p_new, a_new = error(q_new)
            if np.linalg.norm(e_new) < np.linalg.norm(e):
                q, e, p, a = q_new, e_new, p_new, a_new
                lam = max(lam * 0.5, 0.05)
            else:
                lam = min(lam * 3.0, 1e3)
                if lam >= 1e3:
                    break
    return False


def reachable_fraction(chain, targets, seed):
    rng = np.random.default_rng(seed)
    hits = [ik_reachable(chain, np.asarray(t["position"], float), np.asarray(t["approach"], float), rng) for t in targets]
    return (float(np.mean(hits)) if hits else 0.0), hits


def sample_targets(reference_chain, ranges, n, rng, z_min=20.0):
    """Targets = tool pose of the reference arm at random configurations of its ranges (tcp above z_min)."""
    out = []
    lo = np.array([ranges[j.name][0] for j in reference_chain.joints])
    hi = np.array([ranges[j.name][1] for j in reference_chain.joints])
    guard = 0
    while len(out) < n and guard < 50 * n:
        guard += 1
        q = rng.uniform(lo, hi)
        q[-1] = 0.0
        p, a = reference_chain.tcp(q)
        if p[2] < z_min:
            continue
        out.append({"position": [round(float(v), 3) for v in p], "approach": [round(float(v), 6) for v in a]})
    return out


def axis_line_world(chain, joint, T):
    Tc = T[joint.child]
    return Tc[:3, 3], Tc[:3, :3] @ joint.axis


def range_coverage(chain, ranges):
    """Per joint: |declared ∩ reference| / |reference|."""
    out = {}
    for j in chain.joints:
        lo, hi = ranges[j.name]
        inter = max(0.0, min(hi, j.upper) - max(lo, j.lower))
        out[j.name] = inter / max(1e-9, hi - lo)
    return out


def tcp_plausible(sub, max_dist_mm=15.0):
    """The declared TCP must lie within `max_dist_mm` of the gripper link or the jaw (at q = 0)."""
    chain = sub.chain
    T = chain.fk(np.zeros(len(chain.joints)))
    p, _ = chain.tcp(np.zeros(len(chain.joints)), T)
    best = float("inf")
    for name in ("gripper", "jaw"):
        link = chain.links[name]
        for b in link.bodies:
            import trimesh
            pl = apply(inv(T[name]), p)
            d = trimesh.proximity.closest_point(b.mesh, pl.reshape(1, 3))[1][0]
            best = min(best, float(d))
    return best <= max_dist_mm, best


def spec_items(sub, structure, ranges, rules):
    """Geometric convention items (fractions in [0, 1]) with details."""
    chain = sub.chain
    T = chain.fk(np.zeros(len(chain.joints)))
    tol_axis = float(rules.get("grading_tolerances", {}).get("axis_at_zero_deg", 2.0))
    items = []
    for t in structure["joints"]:
        j = chain.joint(t["name"])
        _, d = axis_line_world(chain, j, T)
        ang = angle_between(d, t["axis_at_zero"])
        items.append(("{} axis at q=0 {} vs {} ({:.2f} deg)".format(j.name, np.round(d, 3).tolist(), t["axis_at_zero"], ang), 1.0 if ang <= tol_axis else 0.0))
    _, approach = chain.tcp(np.zeros(len(chain.joints)), T)
    ang = angle_between(approach, [0, 0, 1])
    items.append(("tool approach at q=0 vs +z ({:.2f} deg)".format(ang), 1.0 if ang <= tol_axis else 0.0))
    # straight-up zero pose: shoulder -> elbow -> wrist offsets vertical, roll axis vertical (already an axis item)
    ps, ds = axis_line_world(chain, chain.joint("shoulder_pitch"), T)
    pe, de = axis_line_world(chain, chain.joint("elbow_pitch"), T)
    pw, dw = axis_line_world(chain, chain.joint("wrist_pitch"), T)
    for label, (pa, da, pb) in (("shoulder->elbow", (ps, ds, pe)), ("elbow->wrist", (pe, de, pw))):
        d = pb - pa
        d = d - da * float(d @ da)
        ang = angle_between(d, [0, 0, 1]) if np.linalg.norm(d) > 1e-6 else 90.0
        items.append(("{} offset vertical at q=0 ({:.2f} deg)".format(label, ang), 1.0 if ang <= tol_axis else 0.0))
    base = chain.links[chain.root]
    b = base.bounds
    items.append(("base on the mounting plane (z_min {:.2f})".format(b[0][2]), 1.0 if -0.2 <= b[0][2] <= 1.0 else 0.0))
    cov = range_coverage(chain, ranges)
    for name, c in cov.items():
        items.append(("{} declared limits cover the operating range: {:.2f}".format(name, c), float(c)))
    ok, dist = tcp_plausible(sub)
    items.append(("tcp within 15 mm of the gripper / jaw surface ({:.1f} mm)".format(dist), 1.0 if ok else 0.0))
    frac = float(np.mean([v for _, v in items]))
    return frac, ["{} -> {}".format(m, "ok" if v >= 1.0 else ("{:.2f}".format(v) if v > 0 else "FAIL")) for m, v in items]


def check_spec_geometry(sub, structure, ranges, rules, targets, seed, reference_chain=None):
    frac, hits = reachable_fraction(sub.chain, targets, seed)
    details = ["reachable {} / {} hidden targets".format(int(sum(hits)), len(hits))]
    items_frac, det = spec_items(sub, structure, ranges, rules)
    details += det
    if reference_chain is None:
        return None, frac, items_frac, details
    ref_frac, _ = reachable_fraction(reference_chain, targets, seed)
    ratio = 1.0 if ref_frac <= 0 else min(1.0, frac / ref_frac)
    details.insert(1, "reference reaches {:.3f}; ratio {:.3f}; convention items {:.3f}".format(ref_frac, ratio, items_frac))
    return ratio * items_frac, frac, items_frac, details


def mass_model(sub, servo, facts, density_kg_m3):
    out = {}
    for name, link in sub.chain.links.items():
        m_link = link.solid_volume * 1e-9 * density_kg_m3      # enclosed voids count as material
        if link.mass_override is not None:
            m_link = max(m_link, float(link.mass_override))
        m = m_link
        c = link.centroid * m_link
        for sid, Tocc in link.occurrences:
            ms = float(facts["mass_kg"])
            cs = apply(Tocc, servo.com)
            m += ms
            c = c + ms * cs
        out[name] = {"mass": m, "com": c / m, "link_mass": m_link}
    return out


def gravity_torques(chain, masses, q, payload_kg, T=None):
    T = T or chain.fk(q)
    tcp, _ = chain.tcp(q, T)
    g = np.array([0.0, 0.0, -G])
    taus = []
    for idx, j in enumerate(chain.joints):
        p_axis, d_axis = axis_line_world(chain, j, T)
        tau = np.zeros(3)
        for name in chain.order[idx + 1:]:
            mm = masses[name]
            cw = apply(T[name], mm["com"])
            tau += np.cross((cw - p_axis) * 1e-3, mm["mass"] * g)
        tau += np.cross((tcp - p_axis) * 1e-3, payload_kg * g)
        taus.append(abs(float(tau @ d_axis)))
    return np.array(taus)


def check_mass_budget(sub, servo, facts, requirements, density, configs, payloads_kg, reference_mass=None):
    chain = sub.chain
    masses = mass_model(sub, servo, facts, density)
    moving = sum(masses[n]["mass"] for n in chain.order[1:])
    budget = float(requirements["moving_mass_budget_kg"])
    util_max = float(requirements["payload"]["torque_utilisation_max"])
    stall = float(facts["stall_torque_nm"])
    payloads = [float(p) for p in np.atleast_1d(payloads_kg)]
    worst = np.zeros(len(chain.joints))
    for q in configs:
        T = chain.fk(q)
        for pl in payloads:
            worst = np.maximum(worst, gravity_torques(chain, masses, q, pl, T))
    ratios = worst / (util_max * stall)
    feasible = moving <= budget and float(ratios.max()) <= 1.0
    details = ["moving mass {:.3f} kg (budget {:.3f}); worst gravity torque per joint {} N*m over payloads {} kg; "
               "utilisation of {} x {} N*m: {}".format(moving, budget, np.round(worst, 3).tolist(), [round(p, 3) for p in payloads],
                                                      util_max, stall, np.round(ratios, 3).tolist())]
    if reference_mass is None:
        return None, moving, float(ratios.max()), feasible, details
    mass_ratio = reference_mass / max(moving, 1e-9)
    if mass_ratio >= 1.0 - 1e-6:
        mass_ratio = 1.0
    score = (1.0 if feasible else 0.0) * max(0.0, min(1.0, mass_ratio))
    details.append("reference moving mass {:.3f} kg -> mass ratio {:.3f}; feasible {}".format(reference_mass, min(1.0, reference_mass / max(moving, 1e-9)), feasible))
    return score, moving, float(ratios.max()), feasible, details


def wall_thickness_samples(body, samples=600, seed=0):
    import trimesh
    pts, fid = trimesh.sample.sample_surface(body.mesh, samples, seed=int(seed))
    normals = body.mesh.face_normals[fid]
    th = trimesh.proximity.thickness(body.mesh, pts, exterior=False, normals=normals, method="ray")
    th = np.asarray(th, float)
    return th[np.isfinite(th)]


def check_manufacturable(sub, servo, rules, seed, samples=600):
    chain = sub.chain
    env = sorted(rules["printer_envelope_mm"])
    min_wall = float(rules["min_wall_mm"])
    thin_allow = float(rules.get("thin_wall_fraction_full", 0.02))
    thin_zero = float(rules.get("thin_wall_fraction_zero", 0.15))
    min_hole = float(rules["min_through_hole_mm"])
    rng = np.random.default_rng(seed)
    details = []
    part_scores = []
    for pid in sorted(sub.parts):
        body = sub.parts[pid]
        ext = sorted((body.bounds[1] - body.bounds[0]).tolist())
        bbox_ok = all(e <= v + 1e-6 for e, v in zip(ext, env))
        th = wall_thickness_samples(body, samples, int(rng.integers(2 ** 31)))
        thin = float(np.mean(th < min_wall)) if len(th) else 1.0
        wall_score = 1.0 if thin <= thin_allow else max(0.0, 1.0 - (thin - thin_allow) / (thin_zero - thin_allow))
        s = (1.0 if bbox_ok else 0.0) * wall_score
        part_scores.append(s)
        details.append("{}: bbox {} mm {}; thin-wall samples {:.1%} (<{} mm) -> {:.2f}".format(
            pid, [round(e, 1) for e in ext], "ok" if bbox_ok else "EXCEEDS PRINTER", thin, min_wall, s))
    # interface holes: the used flange screw paths in the carrier and the horn bolt clearance holes in the driven link
    T0 = chain.fk(np.zeros(len(chain.joints)))
    hole_checks = hole_ok = 0
    for lname, link in chain.links.items():
        for sid, Tocc in link.occurrences:
            used = []
            for label, ring, axis_pt in servo.flange_probes():
                frac = float(link.contains([apply(Tocc, p) for p in ring]).mean())
                if frac < 0.75 or link.contains([apply(Tocc, axis_pt[0])])[0]:
                    continue           # not a used hole
                d = 2.0 * max_empty_radius(link, Tocc, axis_pt[0], min_hole / 2 + 0.1)
                used.append("{}:{:.1f}".format(label, d))
                hole_checks += 1
                hole_ok += 1 if d >= min_hole - 1e-9 else 0
            if used:
                details.append("{} flange screw paths (mm, need >= {}): {}".format(sid, min_hole, ", ".join(used)))
            j = chain.joint_of_servo(sid)
            if j is not None:
                other = chain.links[j.child]
                To = inv(T0[j.child]) @ T0[lname] @ Tocc
                d_min = float(rules.get("horn_bolt_hole_min_mm", 1.8))
                c_min = float(rules.get("horn_centre_hole_min_mm", 3.0))
                d_bolt, d_centre = bolt_pattern_clearance(other, To, servo, "horn", r_max=d_min / 2 + 0.1)
                hole_checks += 2
                hole_ok += (1 if d_bolt >= d_min - 1e-9 else 0) + (1 if d_centre >= c_min - 1e-9 else 0)
                details.append("{} horn plate: bolt holes >= {:.1f} mm (need {}), centre >= {:.1f} mm (need {})".format(sid, d_bolt, d_min, d_centre, c_min))
    hole_score = hole_ok / hole_checks if hole_checks else 1.0
    score = float(np.mean(part_scores)) * hole_score
    details.append("interface holes: {} / {} at or above the minimum diameters".format(hole_ok, hole_checks))
    return score, details


# ----------------------------------------------------------------------------- key dimensions (dimension_fidelity)
def _polygon_area_centroid(xy):
    """Signed area (positive = counter-clockwise) and centroid of a closed polygon (n, 2)."""
    x, y = xy[:, 0], xy[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = 0.5 * float(cross.sum())
    if abs(a) < 1e-12:
        return 0.0, np.array([float(x.mean()), float(y.mean())])
    cx = float(((x + xn) * cross).sum()) / (6.0 * a)
    cy = float(((y + yn) * cross).sum()) / (6.0 * a)
    return a, np.array([cx, cy])


def base_bottom_holes(link, d_min=3.0, d_max=9.0):
    """(x, y, equivalent diameter) of the through-openings of the base part that stands on the mounting plane (the root
    link's body with the lowest z), seen from below: the hole rings of the projection of its solid onto the plane
    (manifold3d `project`, exact polygon clipping), which are the enclosed background regions of its bottom-view drawing,
    kept when the equivalent diameter 2 sqrt(A / pi) lies in [d_min, d_max] mm. Modelling-agnostic: a cylinder, a
    polygonal prism or an extruded silhouette contour give the same opening."""
    if not link.bodies:
        return []
    body = min(link.bodies, key=lambda b: float(b.bounds[0][2]))
    try:
        rings = body.manifold.project().to_polygons()
    except Exception:  # noqa: BLE001
        return []
    holes = []
    for ring in rings:
        xy = np.asarray(ring, float).reshape(-1, 2)
        if len(xy) < 3:
            continue
        a, c = _polygon_area_centroid(xy)
        if a >= 0.0:
            continue                          # outer contour (positive orientation); holes are negative
        d = 2.0 * math.sqrt(-a / math.pi)
        if d_min <= d <= d_max:
            holes.append((round(float(c[0]), 3), round(float(c[1]), 3), round(d, 3)))
    merged = []
    for h in sorted(holes):
        if merged and abs(merged[-1][0] - h[0]) < 0.05 and abs(merged[-1][1] - h[1]) < 0.05:
            continue
        merged.append(list(h))
    return merged


def key_dimensions(sub, servo, wall_samples=600):
    """The measurable quantities the hidden reference is compared on (mm / deg, world frame at q = 0)."""
    chain = sub.chain
    T = chain.fk(np.zeros(len(chain.joints)))
    out = {"axes": {}, "servos": {}, "parts": {}}
    for j in chain.joints:
        p, d = axis_line_world(chain, j, T)
        out["axes"][j.name] = {"point": [float(v) for v in p], "dir": [float(v) for v in d]}
    ps, ds = axis_line_world(chain, chain.joint("shoulder_pitch"), T)
    pe, de = axis_line_world(chain, chain.joint("elbow_pitch"), T)
    pw, dw = axis_line_world(chain, chain.joint("wrist_pitch"), T)
    out["link_lengths"] = {"upper_arm": line_dist(ps, ds, pe), "forearm": line_dist(pe, de, pw)}
    tcp, appr = chain.tcp(np.zeros(len(chain.joints)), T)
    out["tcp"] = {"point": [float(v) for v in tcp], "approach": [float(v) for v in appr]}
    for lname, link in chain.links.items():
        for sid, Tocc in link.occurrences:
            Tw = T[lname] @ Tocc
            out["servos"][sid] = {"horn_point": [float(v) for v in apply(Tw, servo.horn_point)],
                                  "horn_dir": [float(v) for v in Tw[:3, :3] @ servo.horn_dir]}
    for pid in sorted(sub.parts):
        b = sub.parts[pid]
        ext = (b.bounds[1] - b.bounds[0]).tolist()
        th = wall_thickness_samples(b, wall_samples, seed=12345)
        out["parts"][pid] = {"extents_xyz": [float(v) for v in ext], "extents_sorted": sorted(float(v) for v in ext),
                             "volume_cm3": float(b.volume) / 1000.0, "solid_volume_cm3": float(b.solid_volume) / 1000.0,
                             "enclosed_void_ratio": float(b.void_ratio),
                             "wall_median_mm": float(np.median(th)) if len(th) else 0.0,
                             "wall_p10_mm": float(np.percentile(th, 10)) if len(th) else 0.0}
    out["base_bottom_holes"] = base_bottom_holes(chain.links[chain.root])
    return out


def _hungarian(cost):
    """Minimum-cost assignment on a rectangular cost matrix (scipy when present, else a small brute force / greedy)."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:  # noqa: BLE001
        pairs = []
        used = set()
        for i in np.argsort(cost.min(axis=1)):
            j = int(np.argmin([cost[i, k] if k not in used else np.inf for k in range(cost.shape[1])]))
            used.add(j)
            pairs.append((int(i), j))
        return pairs


def compare_key_dimensions(agent, ref, full_mm=0.5, zero_mm=3.0, full_deg=0.5, zero_deg=3.0, weights=None):
    """Score in [0, 1] plus details. Groups: axes (5 items), link lengths (2), tcp (1), servos (6), extents (n parts),
    walls (n parts), base bottom hole pattern (1)."""
    weights = weights or {"axes": 0.35, "servos": 0.25, "extents": 0.20, "walls": 0.10, "holes": 0.10}
    groups = {k: [] for k in weights}
    det = []

    def item(group, label, err, full, zero):
        s = band(err, full, zero)
        groups[group].append(s)
        det.append("{}: {} error {:.2f} -> {:.2f}".format(group, label, err, s))

    A, R = agent["axes"], ref["axes"]
    for name, comps in (("shoulder_pitch", (1, 2)), ("elbow_pitch", (1, 2)), ("wrist_pitch", (1, 2)), ("wrist_roll", (0, 1)), ("jaw", (1, 2))):
        pa, pr = np.asarray(A[name]["point"]), np.asarray(R[name]["point"])
        err = float(np.linalg.norm((pa - pr)[list(comps)]))
        item("axes", "{} axis position (mm)".format(name), err, full_mm, zero_mm)
    for name in ("upper_arm", "forearm"):
        item("axes", "{} axis distance (mm)".format(name), agent["link_lengths"][name] - ref["link_lengths"][name], full_mm, zero_mm)
    item("axes", "tcp position (mm)", float(np.linalg.norm(np.asarray(agent["tcp"]["point"]) - np.asarray(ref["tcp"]["point"]))), full_mm, zero_mm)
    for sid in sorted(ref["servos"]):
        if sid not in agent["servos"]:
            groups["servos"].append(0.0)
            continue
        dp = float(np.linalg.norm(np.asarray(agent["servos"][sid]["horn_point"]) - np.asarray(ref["servos"][sid]["horn_point"])))
        dd = angle_between(agent["servos"][sid]["horn_dir"], ref["servos"][sid]["horn_dir"])
        s = min(band(dp, full_mm, zero_mm), band(dd, full_deg, zero_deg))
        groups["servos"].append(s)
        det.append("servos: {} horn point {:.2f} mm, direction {:.2f} deg -> {:.2f}".format(sid, dp, dd, s))
    for pid in sorted(ref["parts"]):
        if pid not in agent["parts"]:
            groups["extents"].append(0.0)
            groups["walls"].append(0.0)
            continue
        ea = np.asarray(agent["parts"][pid]["extents_sorted"])
        er = np.asarray(ref["parts"][pid]["extents_sorted"])
        item("extents", "{} bounding-box extents (mm)".format(pid), float(np.abs(ea - er).max()), full_mm, zero_mm)
        item("walls", "{} thin-wall thickness p10 (mm)".format(pid), agent["parts"][pid]["wall_p10_mm"] - ref["parts"][pid]["wall_p10_mm"], full_mm, zero_mm)
    ha = np.asarray(agent.get("base_bottom_holes", []), float).reshape(-1, 3)
    hr = np.asarray(ref.get("base_bottom_holes", []), float).reshape(-1, 3)
    if len(hr) == 0:
        groups["holes"].append(1.0)
    elif len(ha) == 0:
        groups["holes"].append(0.0)
        det.append("holes: no holes on the base bottom face (reference has {})".format(len(hr)))
    else:
        cost = np.linalg.norm(hr[:, None, :2] - ha[None, :, :2], axis=2)
        pairs = _hungarian(cost)
        errs = [cost[i, j] for i, j in pairs]
        missing = len(hr) - len(pairs)
        extra = max(0, len(ha) - len(pairs))
        mean_err = float(np.mean(errs + [zero_mm] * (missing + extra)))
        item("holes", "base bottom hole pattern ({} ref, {} agent; mean matched distance mm)".format(len(hr), len(ha)), mean_err, full_mm, zero_mm)
    total = 0.0
    for k, w in weights.items():
        g = groups[k]
        total += w * (float(np.mean(g)) if g else 1.0)
    return total, det, {k: (float(np.mean(v)) if v else 1.0) for k, v in groups.items()}


def layout_max_error_mm(agent, ref):
    """Largest position error over the axis points, the tool point and the servo horn points (mm): a memorisation
    indicator (measurement from drawings is never this exact) recorded as a stat."""
    errs = []
    for name, comps in (("shoulder_pitch", (1, 2)), ("elbow_pitch", (1, 2)), ("wrist_pitch", (1, 2)), ("wrist_roll", (0, 1)), ("jaw", (1, 2))):
        if name in agent["axes"] and name in ref["axes"]:
            d = np.asarray(agent["axes"][name]["point"]) - np.asarray(ref["axes"][name]["point"])
            errs.append(float(np.linalg.norm(d[list(comps)])))
    errs.append(float(np.linalg.norm(np.asarray(agent["tcp"]["point"]) - np.asarray(ref["tcp"]["point"]))))
    for sid in ref["servos"]:
        if sid in agent["servos"]:
            errs.append(float(np.linalg.norm(np.asarray(agent["servos"][sid]["horn_point"]) - np.asarray(ref["servos"][sid]["horn_point"]))))
    return max(errs) if errs else float("inf")


# ----------------------------------------------------------------------------- drawing fidelity
def part_mesh_for_render(body):
    return body.mesh.vertices, body.mesh.faces


FLOOR_IOU_LO, FLOOR_IOU_HI = 0.70, 0.90


def silhouette_floor(iou_by_part, lo=FLOOR_IOU_LO, hi=FLOOR_IOU_HI):
    """Per part clamp((mean silhouette IoU over its views - lo) / (hi - lo), 0, 1); mean over the parts. Multiplies every
    criterion except drawing_fidelity: credit for a working arm is credit for the DRAWN arm, so parts that do not resemble
    their drawings (bounding boxes, other arms) earn nothing on fit, kinematics, dimensions or printability."""
    if not iou_by_part:
        return 0.0
    vals = [max(0.0, min(1.0, (float(v) - lo) / (hi - lo))) for v in iou_by_part.values()]
    return float(np.mean(vals))


def check_drawing_fidelity(sub, views_index, views_dir, render_mod, fidelity_mod, log=None, iou_full=0.97, iou_zero=0.80,
                           chamfer_full_mm=0.5, chamfer_zero_mm=4.0):
    """Re-render every submitted part in the given projections and compare with the given drawings. Returns (score,
    details, per_part, raw, summary) with summary = {floor, iou_mean, edge_chamfer_mean_mm, iou_by_part}."""
    from PIL import Image
    per_part = {}
    details = []
    raw = {}
    iou_by_part = {}
    chamfers = []
    for pid in sorted(views_index["parts"]):
        entry = views_index["parts"][pid]
        body = sub.parts.get(pid)
        if body is None:
            per_part[pid] = 0.0
            iou_by_part[pid] = 0.0
            details.append("{}: part missing -> 0".format(pid))
            continue
        verts, faces = part_mesh_for_render(body)
        scores = []
        ious = []
        raw[pid] = {}
        for view, vinfo in sorted(entry["views"].items()):
            given = np.asarray(Image.open(os.path.join(views_dir, vinfo["file"])).convert("RGB"))
            ch = int(vinfo["content_h"])
            mask_ref = render_mod.mask_from_png(given, ch)
            edges_ref = render_mod.edges_from_png(given, ch)
            img, info = render_mod.render_scene([(verts, faces, render_mod.MATERIAL_RGB)], view, float(vinfo["mm_per_px"]),
                                                width=int(vinfo["width"]), content_h=ch, band_px=int(vinfo["height"]) - ch,
                                                scale_bar_mm=float(vinfo.get("scale_bar_mm", 50.0)), label=None)
            mask_c = render_mod.mask_from_png(img, ch)
            edges_c = render_mod.edges_from_png(img, ch)
            r = fidelity_mod.view_fidelity(mask_ref, edges_ref, mask_c, edges_c, float(vinfo["mm_per_px"]), iou_full, iou_zero,
                                           chamfer_full_mm, chamfer_zero_mm)
            scores.append(r["score"])
            ious.append(float(r["iou"]))
            chamfers.append(min(float(r["chamfer_mm"]), 10.0) if math.isfinite(r["chamfer_mm"]) else 10.0)
            raw[pid][view] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            details.append("{} {}: IoU {:.4f} ({:.2f}), edge chamfer {:.3f} mm ({:.2f}) -> {:.3f}".format(
                pid, view, r["iou"], r["iou_score"], r["chamfer_mm"], r["edge_score"], r["score"]))
        per_part[pid] = float(np.mean(scores)) if scores else 0.0
        iou_by_part[pid] = float(np.mean(ious)) if ious else 0.0
        if log:
            log("[check]   drawing {} -> {:.3f} (mean IoU {:.4f})".format(pid, per_part[pid], iou_by_part[pid]))
    score = float(np.mean(list(per_part.values()))) if per_part else 0.0
    summary = {"floor": silhouette_floor(iou_by_part), "iou_by_part": iou_by_part,
               "iou_mean": float(np.mean(list(iou_by_part.values()))) if iou_by_part else 0.0,
               "edge_chamfer_mean_mm": float(np.mean(chamfers)) if chamfers else 10.0}
    return score, details, per_part, raw, summary


# ----------------------------------------------------------------------------- public-model reuse
def vertex_fingerprint(mesh, quant=0.01):
    """sha256 of the sorted, quantised vertex set (frame-dependent; catches re-exports of the same B-rep in the same frame)."""
    v = np.round(np.asarray(mesh.vertices, float) / quant).astype(np.int64)
    v = np.unique(v, axis=0)
    return hashlib.sha256(v.tobytes()).hexdigest()


def geometry_invariants(mesh):
    """Pose-invariant coarse signature: volume, area, sorted principal inertia (per unit mass), sorted AABB extents (frame
    dependent, informative only), and the pose- AND scale-free `shape` vector (area / volume^(2/3), principal inertia shares)."""
    vol = max(float(mesh.volume), 1e-9)
    I = np.asarray(mesh.moment_inertia, float) / vol
    eig = np.sort(np.linalg.eigvalsh(I))
    ext = sorted(float(v) for v in (mesh.bounds[1] - mesh.bounds[0]))
    tr = max(float(eig.sum()), 1e-12)
    shape = [float(mesh.area) / vol ** (2.0 / 3.0)] + [float(e) / tr for e in eig]
    return {"volume": float(mesh.volume), "area": float(mesh.area), "inertia": [float(e) for e in eig], "extents": ext, "shape": shape}


def invariants_match(a, b, rel=2e-3):
    def close(x, y):
        return abs(x - y) <= rel * max(abs(x), abs(y), 1e-9)
    return (close(a["volume"], b["volume"]) and close(a["area"], b["area"])
            and all(close(x, y) for x, y in zip(a["inertia"], b["inertia"])))


def invariants_match_similar(a, b, rel=2e-3):
    """Same geometry up to pose AND uniform scale (the scale-free shape vector agrees)."""
    sa, sb = a.get("shape"), b.get("shape")
    if not sa or not sb or len(sa) != len(sb):
        return False
    return all(abs(x - y) <= rel * max(abs(x), abs(y), 1e-9) for x, y in zip(sa, sb))


def check_public_model_reuse(sub, blocklist, exempt_parts=()):
    """(gate 1/0, details, n_geometry_matches, n_similar). Gate 0 when a non-exempt submitted file's sha256 or vertex
    fingerprint is in the blocklist (a verbatim copy or a same-frame re-export of a public model). The pose-invariant
    match count (same volume / area / inertia) and the pose-and-scale-invariant similarity count are stats: the reference
    geometry itself matches by construction."""
    sha = set(blocklist.get("sha256", {}).keys())
    vfp = set(blocklist.get("vertex_fingerprints", {}).keys())
    inv = blocklist.get("invariants", [])
    hits, matches, similar = [], 0, 0
    details = []
    for pid, path in sorted(sub.part_files.items()):
        body = sub.parts[pid]
        h = sha256_file(path)
        f = vertex_fingerprint(body.mesh)
        g = geometry_invariants(body.mesh)
        gm = any(invariants_match(g, ref) for ref in inv)
        gs = gm or any(invariants_match_similar(g, ref) for ref in inv)
        matches += 1 if gm else 0
        similar += 1 if gs else 0
        if pid in exempt_parts:
            details.append("{}: disclosed practice part, exempt (geometry match {})".format(pid, gm))
            continue
        if h in sha or f in vfp:
            hits.append(pid)
            details.append("{}: file digest or vertex fingerprint matches a public model -> gate 0".format(pid))
        elif gs:
            details.append("{}: pose-invariant geometry matches a public part{} (stat)".format(pid, "" if gm else " up to a uniform scale"))
    # any loose STEP/STL under the submission counts too
    for root, _dirs, files in os.walk(sub.dir):
        for fn in files:
            if fn.lower().endswith((".step", ".stp", ".stl")):
                p = os.path.join(root, fn)
                if p in sub.part_files.values():
                    continue
                if sha256_file(p) in sha:
                    hits.append(fn)
                    details.append("{}: extra file is a public model copy -> gate 0".format(fn))
    return (0.0 if hits else 1.0), details, matches, similar


# ----------------------------------------------------------------------------- MJCF (downstream consumer)
def _quat_from_matrix(R):
    m = R
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [w, x, y, z]


def _inertia_of_link(link, servo, masses, density, facts):
    m_link = masses["link_mass"]
    parts = []
    scale = 1.0
    vol = link.volume
    if link.mass_override is not None and vol > 0:
        scale = m_link / max(vol * 1e-9 * density, 1e-12)
    for b in link.bodies:
        I_b = np.asarray(b.mesh.moment_inertia, float) * 1e-15 * density * scale
        parts.append((b.volume * 1e-9 * density * scale, b.centroid * 1e-3, I_b))
    s = servo.size * 1e-3
    for sid, Tocc in link.occurrences:
        ms = float(facts["mass_kg"])
        I_box = ms / 12.0 * np.diag([s[1] ** 2 + s[2] ** 2, s[0] ** 2 + s[2] ** 2, s[0] ** 2 + s[1] ** 2])
        R = Tocc[:3, :3]
        parts.append((ms, apply(Tocc, servo.com) * 1e-3, R @ I_box @ R.T))
    m = sum(p[0] for p in parts)
    c = sum(p[0] * p[1] for p in parts) / m
    I = np.zeros((3, 3))
    for mi, ci, Ii in parts:
        d = ci - c
        I += Ii + mi * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return m, c, I


def write_mjcf(sub, servo, density, out_dir, facts):
    chain = sub.chain
    masses = mass_model(sub, servo, facts, density)
    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    lines = ['<mujoco model="arm_links_from_orthographic_views">',
             '  <compiler angle="radian" meshdir="meshes" autolimits="true"/>',
             '  <option gravity="0 0 -9.81"/>',
             '  <asset>']
    for name, link in chain.links.items():
        for b in link.bodies:
            path = os.path.join(mesh_dir, "{}.stl".format(b.name))
            b.mesh.export(path)
            lines.append('    <mesh name="{0}" file="{0}.stl" scale="0.001 0.001 0.001"/>'.format(b.name))
    lines.append('  </asset>')
    lines.append('  <worldbody>')
    by_child = dict((j.child, j) for j in chain.joints)

    def emit(name, depth):
        ind = "  " * (depth + 2)
        link = chain.links[name]
        if name == chain.root:
            lines.append('{}<body name="{}">'.format(ind, name))
        else:
            j = by_child[name]
            xyz = j.origin[:3, 3] * 1e-3
            quat = _quat_from_matrix(j.origin[:3, :3])
            lines.append('{}<body name="{}" pos="{:.9f} {:.9f} {:.9f}" quat="{:.9f} {:.9f} {:.9f} {:.9f}">'.format(ind, name, *(list(xyz) + quat)))
            lines.append('{}  <joint name="{}" type="hinge" axis="{:.9f} {:.9f} {:.9f}" range="{:.6f} {:.6f}"/>'.format(
                ind, j.name, j.axis[0], j.axis[1], j.axis[2], j.lower, j.upper))
        m, c, I = _inertia_of_link(link, servo, masses[name], density, facts)
        lines.append('{}  <inertial pos="{:.9f} {:.9f} {:.9f}" mass="{:.9f}" fullinertia="{:.12e} {:.12e} {:.12e} {:.12e} {:.12e} {:.12e}"/>'.format(
            ind, c[0], c[1], c[2], m, I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))
        for b in link.bodies:
            lines.append('{}  <geom type="mesh" mesh="{}" contype="0" conaffinity="0" mass="0"/>'.format(ind, b.name))
        for sid, Tocc in link.occurrences:
            s = servo.size * 1e-3
            cpos = apply(Tocc, servo.com) * 1e-3
            quat = _quat_from_matrix(Tocc[:3, :3])
            lines.append('{}  <geom name="{}_{}" type="box" size="{:.6f} {:.6f} {:.6f}" pos="{:.9f} {:.9f} {:.9f}" quat="{:.9f} {:.9f} {:.9f} {:.9f}" contype="0" conaffinity="0" mass="0" rgba="0.2 0.2 0.2 1"/>'.format(
                ind, name, sid, s[0] / 2, s[1] / 2, s[2] / 2, *(list(cpos) + quat)))
        for j in chain.joints:
            if j.parent == name:
                emit(j.child, depth + 1)
        lines.append('{}</body>'.format(ind))

    emit(chain.root, 0)
    lines.append('  </worldbody>')
    lines.append('</mujoco>')
    path = os.path.join(out_dir, "arm.xml")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def check_downstream(sub, servo, density, unit_configs, work_dir, facts):
    import mujoco
    chain = sub.chain
    details = []
    try:
        xml = write_mjcf(sub, servo, density, work_dir, facts)
    except Exception as exc:  # noqa: BLE001
        return 0.0, ["converter failed: {}".format(str(exc)[:200])]
    warnings_seen = []

    def _warn(msg):
        warnings_seen.append(str(msg))

    saved = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 2)
    try:
        try:
            mujoco.set_mju_user_warning(_warn)
        except Exception:  # noqa: BLE001
            pass
        try:
            model = mujoco.MjModel.from_xml_path(xml)
        except Exception as exc:  # noqa: BLE001
            model = None
            details.append("MJCF failed to compile: {}".format(str(exc)[:300]))
        finally:
            try:
                mujoco.set_mju_user_warning(None)
            except Exception:  # noqa: BLE001
                pass
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        tmp.seek(0)
        captured = tmp.read().decode("utf-8", "replace")
        tmp.close()
    if "WARNING" in captured.upper():
        warnings_seen.append(captured.strip()[:300])
    if model is None:
        return 0.0, details
    if warnings_seen:
        details.append("compile warnings: {}".format(warnings_seen[:3]))
        return 0.0, details
    data = mujoco.MjData(model)
    configs = chain.lower + np.asarray(unit_configs) * (chain.upper - chain.lower)
    worst_p = worst_a = 0.0
    for q in configs:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        T = chain.fk(q)
        for name in chain.order:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            p_mj = data.xpos[bid] * 1e3
            R_mj = data.xmat[bid].reshape(3, 3)
            worst_p = max(worst_p, float(np.linalg.norm(p_mj - T[name][:3, 3])))
            cosang = (np.trace(R_mj.T @ T[name][:3, :3]) - 1) / 2
            worst_a = max(worst_a, math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    ok = worst_p <= 1.0 and worst_a <= 0.5
    details.append("MJCF compiled ({} bodies, {} joints), FK deviation {:.3f} mm / {:.3f} deg over {} configs".format(
        model.nbody - 1, model.njnt, worst_p, worst_a, len(configs)))
    return (1.0 if ok else 0.0), details


# ----------------------------------------------------------------------------- driver
def load_given(given_dir):
    """Every given document the checks need (the drawings are read by check_drawing_fidelity from given_dir/views)."""
    import yaml
    with open(os.path.join(given_dir, "requirements.yaml")) as f:
        requirements = yaml.safe_load(f)
    with open(os.path.join(given_dir, "design_rules.yaml")) as f:
        rules = yaml.safe_load(f)
    with open(os.path.join(given_dir, "assembly_schema.json")) as f:
        schema = json.load(f)
    servo = Servo(os.path.join(given_dir, "servo_standin.json"), os.path.join(given_dir, "servo_standin.step"))
    structure = requirements.get("structure", DEFAULT_STRUCTURE)
    with open(os.path.join(given_dir, "views", "index.json")) as f:
        views_index = json.load(f)
    facts = dict(servo.facts)
    blocklist = None
    public_bl = os.path.join(given_dir, "practice", "public_blocklist.json")
    if os.path.isfile(public_bl):
        with open(public_bl) as f:
            doc = json.load(f)
        blocklist = {"sha256": dict((k, True) for k in doc.get("sha256", [])),
                     "vertex_fingerprints": dict((k, True) for k in doc.get("vertex_fingerprints", [])), "invariants": []}
    return {"requirements": requirements, "rules": rules, "schema": schema, "servo": servo, "structure": structure,
            "views_index": views_index, "views_dir": os.path.join(given_dir, "views"), "facts": facts, "blocklist": blocklist}


CRITERIA = ["drawing_fidelity", "dimension_fidelity", "component_fit", "interference_free", "spec_geometry",
            "mass_budget", "manufacturable", "downstream_consumer"]
GATES = ["cad_valid", "cad_json_consistent", "servos_mounted", "public_model_reuse"]


def run_checks(sub_dir, given, cases, reference=None, log=print, render_mod=None, fidelity_mod=None, skip_drawings=False):
    """Run every check. `cases`: configs_unit_stream (K,6), n_configs, configs_fk_unit (M,6), targets, payload_kg, seed,
    ranges (per joint [lo, hi], the operating range the configurations are drawn in). `reference` (hidden run):
    assembly (dict), moving_mass_kg, key_dimensions, blocklist. Without a reference the anchored criteria are None
    and the raw values are in `stats`. Returns (criteria, stats, details)."""
    servo, req, rules, schema, structure = given["servo"], given["requirements"], given["rules"], given["schema"], given["structure"]
    facts = given["facts"]
    density = float(rules["material"]["density_kg_m3"])
    criteria = dict((k, 0.0) for k in GATES + CRITERIA)
    stats = {"part_count": 0.0, "total_mass_kg": 0.0, "reachable_fraction": 0.0, "torque_utilisation": 0.0, "min_retention": 0.0,
             "source_geometry_matches": 0.0, "design_py_present": 1.0 if os.path.isfile(os.path.join(sub_dir, "design.py")) else 0.0}
    details = {}
    sub = Submission(sub_dir, schema, structure)
    stats["part_count"] = float(len(sub.parts))
    if not sub.ok:
        details["cad_valid"] = sub.gate_errors
        log("[check] cad_valid FAILED: " + " | ".join(sub.gate_errors[:8]))
        return criteria, stats, details
    criteria["cad_valid"] = 1.0
    details["cad_valid"] = ["{} parts load as single valid watertight solids; assembly.json valid".format(len(sub.parts))] + sub.notes
    log("[check] cad_valid 1.0" + (" ({})".format("; ".join(sub.notes)) if sub.notes else ""))

    blocklist = (reference or {}).get("blocklist") or given.get("blocklist") or {}
    reuse, det, matches, similar = check_public_model_reuse(sub, blocklist, exempt_parts=structure.get("disclosed_parts", []))
    criteria["public_model_reuse"] = reuse
    details["public_model_reuse"] = det
    stats["source_geometry_matches"] = float(matches)
    stats["source_geometry_similar"] = float(similar)
    stats["blocklist_entries"] = float(len(blocklist.get("sha256", {})) + len(blocklist.get("vertex_fingerprints", {})))
    log("[check] public_model_reuse {:.0f} (pose-invariant geometry matches: {}, up to scale: {}; {} blocklist entries)".format(
        reuse, matches, similar, int(stats["blocklist_entries"])))
    if reuse < 1.0:
        return criteria, stats, details

    tol = rules["grading_tolerances"]
    gate, det = check_cad_json_consistent(sub, servo, tol["axis_angle_deg"], tol["axis_offset_mm"], tol["contact_mm"])
    criteria["cad_json_consistent"] = gate
    details["cad_json_consistent"] = det
    log("[check] cad_json_consistent {:.1f}: {}".format(gate, " | ".join(det)))
    if gate < 1.0:
        return criteria, stats, details

    fit, det, retention = check_component_fit(sub, servo, float(rules.get("servo_mounting", {}).get("keep_out_volume_ratio", KEEPOUT_THR)))
    stats["min_retention"] = float(min(retention.values())) if retention else 0.0
    unmounted = sorted(s for s, r in retention.items() if r < MOUNTED_MIN_RETENTION)
    criteria["servos_mounted"] = 0.0 if unmounted or not retention else 1.0
    details["servos_mounted"] = ["retention per servo (keep-out x flanges x horn plate; gate at {:.2f}): {}{}".format(
        MOUNTED_MIN_RETENTION, ", ".join("{} {:.2f}".format(s, retention[s]) for s in sorted(retention)),
        "; NOT MOUNTED: {}".format(unmounted) if unmounted else "")]
    details["component_fit"] = det
    log("[check] servos_mounted {:.1f}: {}".format(criteria["servos_mounted"], details["servos_mounted"][0]))
    for d in det:
        log("        " + d)
    if criteria["servos_mounted"] < 1.0:
        return criteria, stats, details
    criteria["component_fit"] = fit
    log("[check] component_fit {:.3f}".format(fit))

    if not skip_drawings:
        if render_mod is None or fidelity_mod is None:
            from arm.cad_drawings import render as render_mod, fidelity as fidelity_mod  # noqa: F811
        fid, det, per_part, raw, summary = check_drawing_fidelity(sub, given["views_index"], given["views_dir"], render_mod, fidelity_mod, log=log,
                                                         iou_full=float(rules.get("drawing_fidelity", {}).get("iou_full", 0.97)),
                                                         iou_zero=float(rules.get("drawing_fidelity", {}).get("iou_zero", 0.80)),
                                                         chamfer_full_mm=float(rules.get("drawing_fidelity", {}).get("edge_chamfer_full_mm", 0.5)),
                                                         chamfer_zero_mm=float(rules.get("drawing_fidelity", {}).get("edge_chamfer_zero_mm", 4.0)))
        criteria["drawing_fidelity"] = fid
        floor = float(summary["floor"])
        details["drawing_fidelity"] = ["silhouette floor {:.3f} (per part clamp((mean IoU - {}) / {}, 0, 1), mean over the parts; multiplies every "
                                       "other criterion); mean IoU {:.4f}, mean edge chamfer {:.2f} mm".format(
                                           floor, FLOOR_IOU_LO, FLOOR_IOU_HI - FLOOR_IOU_LO, summary["iou_mean"], summary["edge_chamfer_mean_mm"])] + det
        for pid, s in per_part.items():
            stats["fidelity_{}".format(pid)] = float(s)
        stats["drawing_raw"] = raw
        stats["drawing_floor"] = floor
        stats["iou_mean"] = float(summary["iou_mean"])
        stats["edge_chamfer_mean_mm"] = float(summary["edge_chamfer_mean_mm"])
        log("[check] drawing_fidelity {:.3f} (silhouette floor {:.3f}, mean IoU {:.4f}, mean edge chamfer {:.2f} mm)".format(
            fid, floor, summary["iou_mean"], summary["edge_chamfer_mean_mm"]))

    ranges = cases["ranges"]
    configs, examined, skipped = operating_configs(sub, servo, cases["configs_unit_stream"], int(cases["n_configs"]), ranges)
    stats["operating_configs"] = float(len(configs))
    stats["configs_outside_declared_limits"] = float(skipped)
    # the declared range beyond the drawn arm's box is tested for self-collision too, weighted by its volume share
    extra, examined_x, inside_share = declared_range_configs(sub, servo, cases["configs_unit_stream"], int(cases.get("n_declared_configs", 16)), ranges)
    stats["declared_range_configs"] = float(len(extra))
    stats["declared_inside_share"] = float(inside_share)
    adj_thr = float(rules["clearances"].get("adjacent_ratio", 1e-2))
    non_thr = float(rules["clearances"].get("nonadjacent_ratio", 1e-3))
    if len(configs) == 0:
        details["interference_free"] = ["no operating configuration lies inside the declared limits above the mounting plane"]
        log("[check] interference_free 0.0: " + details["interference_free"][0])
    else:
        inter, det, _ = check_interference(sub, servo, configs, adj_thr, non_thr)
        note = ""
        if len(extra):
            inter_x, det_x, _ = check_interference(sub, servo, extra, adj_thr, non_thr)
            stats["declared_range_clean"] = float(inter_x)
            w_out = 1.0 - inside_share
            inter = (1.0 - w_out) * inter + w_out * inter_x
            note = "; declared limits beyond the drawn arm's range ({:.0%} of the declared box): {} -> {:.3f}, combined {:.3f}".format(
                w_out, det_x[0], inter_x, inter)
        criteria["interference_free"] = inter
        details["interference_free"] = [det[0] + note]
        log("[check] interference_free {:.3f}: {} ({} of {} sampled configurations usable, {} outside the declared limits{})".format(
            inter, det[0], len(configs), examined, skipped, note))

    ref_chain = ref_mass = ref_dims = None
    if reference is not None:
        ref_chain = Chain(reference["assembly"])
        ref_mass = float(reference["moving_mass_kg"])
        ref_dims = reference.get("key_dimensions")
    spec, frac, items, det = check_spec_geometry(sub, structure, ranges, rules, cases["targets"], cases["seed"], ref_chain)
    stats["reachable_fraction"] = frac
    stats["convention_items"] = items
    details["spec_geometry"] = det
    if spec is not None:
        criteria["spec_geometry"] = spec
    log("[check] spec_geometry {}: reachable fraction {:.3f}, convention items {:.3f}".format(
        "{:.3f}".format(spec) if spec is not None else "(raw)", frac, items))
    for d in det[1:]:
        log("        " + d)

    band_top = float(req["payload"]["static_check_band_kg"][1])
    payloads = [band_top, float(cases["payload_kg"])]
    if len(configs) == 0:
        configs = np.zeros((1, len(sub.chain.joints)))
    mass, moving, util, feasible, det = check_mass_budget(sub, servo, facts, req, density, configs, payloads, ref_mass)
    stats["total_mass_kg"] = moving
    stats["torque_utilisation"] = util
    stats["mass_feasible"] = 1.0 if feasible else 0.0
    details["mass_budget"] = det
    if mass is not None:
        criteria["mass_budget"] = mass
    log("[check] mass_budget {}: {}".format("{:.3f}".format(mass) if mass is not None else "(raw)", " | ".join(det)))

    dims = key_dimensions(sub, servo)
    stats["key_dimensions"] = dims
    if ref_dims is not None:
        dscore, det, groups = compare_key_dimensions(dims, ref_dims,
                                                     full_mm=float(rules.get("dimension_fidelity", {}).get("full_mm", 0.5)),
                                                     zero_mm=float(rules.get("dimension_fidelity", {}).get("zero_mm", 3.0)),
                                                     full_deg=float(rules.get("dimension_fidelity", {}).get("full_deg", 0.5)),
                                                     zero_deg=float(rules.get("dimension_fidelity", {}).get("zero_deg", 3.0)))
        criteria["dimension_fidelity"] = dscore
        details["dimension_fidelity"] = det
        for k, v in groups.items():
            stats["dims_{}".format(k)] = float(v)
        stats["layout_max_error_mm"] = float(layout_max_error_mm(dims, ref_dims))
        log("[check] dimension_fidelity {:.3f}: groups {} (largest layout error {:.3f} mm)".format(
            dscore, {k: round(v, 3) for k, v in groups.items()}, stats["layout_max_error_mm"]))
    stats["enclosed_void_ratio_max"] = float(max([p.get("enclosed_void_ratio", 0.0) for p in dims["parts"].values()] or [0.0]))

    man, det = check_manufacturable(sub, servo, rules, cases["seed"])
    criteria["manufacturable"] = man
    details["manufacturable"] = det
    log("[check] manufacturable {:.3f}".format(man))
    for d in det:
        log("        " + d)

    with tempfile.TemporaryDirectory(prefix="arm_mjcf_") as work:
        down, det = check_downstream(sub, servo, density, cases["configs_fk_unit"], work, facts)
    criteria["downstream_consumer"] = down
    details["downstream_consumer"] = det
    log("[check] downstream_consumer {:.1f}: {}".format(down, " | ".join(det)))
    # the silhouette floor: credit for a working arm is credit for the DRAWN arm
    floor = stats.get("drawing_floor")
    if floor is not None and floor < 1.0 - 1e-12:
        for k in CRITERIA:
            if k != "drawing_fidelity":
                criteria[k] = float(criteria[k]) * floor
        log("[check] silhouette floor {:.3f} applied to every criterion except drawing_fidelity".format(floor))
    return criteria, stats, details
