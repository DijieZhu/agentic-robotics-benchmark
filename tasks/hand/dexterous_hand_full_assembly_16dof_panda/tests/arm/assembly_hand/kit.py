"""Reader of the hidden kit of the hand assembly cell (the `gt/` tree: kit.json, mating_graph.json, joints.json,
meshes/visual/, meshes/collision/ with collision.json) and the MuJoCo bodies of one unit.

    kit = HandKit(gt_dir)
    kit.units                       {unit: {"kind": kind name}}               (kit.json units_of_the_cell)
    kit.kind_of(unit)               the kind key ("FM", "TM", "PB", ...)
    kit.mates()                     the mating graph's nine mates (palm frame, mm)
    kit.links(kind)                 the kind's links, root first, with their parent and joint
    kit.collision_hulls(kind)       [(link, vertices mm (n, 3), source solid)] in the kind's unit frame
    add_unit(spec, parent, kit, unit, ...)   the unit's link bodies (joints as simulated, inertia, collision geoms,
                                    the loop) under `parent`; returns {link: body name}

Every link body of a unit has the unit frame at the start pose (all joints 0): child bodies sit at the parent's origin
with no rotation, joints at their anchors, geoms in the unit frame. Lengths in the kit are mm; MuJoCo gets metres.
The joints are those of joints.json: the stops of the true solids, the free range while the unit is not fastened (the
coupling sets the mounted range at the lock), the servo's backdrive friction, damping and armature, the passive pins.
"""
import json
import os

import numpy as np

KIND_KEY = {"palm_base": "PB", "finger_module": "FM", "thumb_module": "TM", "electronics_stack": "BS",
            "upper_shell": "PT", "dorsal_cover": "PC", "power_hub": "HB", "press_cap": "CC"}


def read_ply(path):
    """binary little-endian PLY (float x y z, uchar-int faces) -> (V float64 (n, 3), F int64 (m, 3))"""
    data = open(path, "rb").read()
    end = data.find(b"end_header\n")
    nv = nf = 0
    for line in data[:end].decode("ascii").splitlines():
        t = line.split()
        if t[:2] == ["element", "vertex"]:
            nv = int(t[2])
        elif t[:2] == ["element", "face"]:
            nf = int(t[2])
    body = memoryview(data)[end + len(b"end_header\n"):]
    V = np.frombuffer(body[:12 * nv], dtype="<f4").reshape(nv, 3).astype(np.float64)
    rec = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    F = np.frombuffer(body[12 * nv:], dtype=rec, count=nf)["i"].astype(np.int64)
    return V, F


class HandKit(object):
    def __init__(self, gt_dir):
        self.dir = os.path.abspath(gt_dir)
        self.kit = json.load(open(os.path.join(self.dir, "kit.json")))
        self.graph = json.load(open(os.path.join(self.dir, "mating_graph.json")))
        self.joint_model = json.load(open(os.path.join(self.dir, "joints.json")))
        cj = os.path.join(self.dir, "meshes", "collision", "collision.json")
        self.collision = json.load(open(cj)) if os.path.isfile(cj) else None
        self.units = self.kit["units_of_the_cell"]
        self.kinds = self.kit["kinds"]
        self._ply = {}

    def kind_of(self, unit):
        return KIND_KEY[self.units[unit]["kind"]]

    def units_of_kind(self, kind_name):
        return [u for u, v in self.units.items() if v["kind"] == kind_name]

    def mates(self):
        return list(self.graph["mates"])

    def links(self, kind):
        """[(link, parent link or None, joint record or None)], root first"""
        k = self.kinds[kind]
        if not k.get("articulated"):
            return [("body", None, None)]
        js = k["joints"]
        child = {j["child"]: j for j in js}
        names = list(k["mass"]["links"].keys())
        roots = [n for n in names if n not in child]
        order, todo = list(roots), [n for n in names if n not in roots]
        while todo:
            for n in list(todo):
                if child[n]["parent"] in order:
                    order.append(n)
                    todo.remove(n)
        return [(n, child[n]["parent"] if n in child else None, child.get(n)) for n in order]

    def _ply_cached(self, rel):
        if rel not in self._ply:
            self._ply[rel] = read_ply(os.path.join(self.dir, rel))
        return self._ply[rel]

    def collision_hulls(self, kind):
        b = self.collision["kinds"][kind]
        V, _F = self._ply_cached(b["file"])
        return [(h["link"], V[h["v0"]:h["v0"] + h["nv"]], h["solid"]) for h in b["hulls"]]

    def joint(self, kind, name):
        return self.joint_model["kinds"][kind]["joints"][name]


def _quat_of(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def add_unit(spec, parent, kit, unit, pose=None, free=True, collision=True, prefix=None, mesh_cache=None,
             col_rgba=(0.85, 0.35, 0.3, 0.35), contype=1, conaffinity=1, joint_range="free"):
    """the unit's bodies under `parent` (an MjsBody). pose: the root link's 4x4 pose in the parent's frame (m);
    free: a free joint on the root (else fixed to the parent). joint_range: "free" or a mate key (its mounted ranges).
    Returns {"links": {link: body name}, "joints": {joint: joint name}, "geoms": [geom names], "loop": eq name or None}"""
    import mujoco
    kind = kit.kind_of(unit)
    kd = kit.kinds[kind]
    jm = kit.joint_model
    prefix = prefix or unit
    mesh_cache = {} if mesh_cache is None else mesh_cache
    T = np.eye(4) if pose is None else np.asarray(pose, float)
    out = {"links": {}, "joints": {}, "geoms": [], "loop": None}
    hulls = kit.collision_hulls(kind) if collision and kit.collision else []
    num = jm["dynamics"]["numerics"]
    bodies = {}
    for link, par, j in kit.links(kind):
        pb = parent if par is None else bodies[par]
        if par is None:
            b = pb.add_body(name="%s__%s" % (prefix, link), pos=T[:3, 3].tolist(), quat=_quat_of(T[:3, :3]).tolist())
            if free:
                b.add_freejoint(name="%s__free" % prefix)
        else:
            b = pb.add_body(name="%s__%s" % (prefix, link), pos=[0, 0, 0], quat=[1, 0, 0, 0])
            jr = kit.joint(kind, j["name"])
            dyn = jm["dynamics"][jr["type"]]
            if "range" in jr:
                rng = jr["range"]
            elif joint_range == "free":
                rng = jr["range_free"]
            else:
                rng = jr["range_mounted"][joint_range]
            jt = b.add_joint(name="%s__%s" % (prefix, j["name"]), type=mujoco.mjtJoint.mjJNT_HINGE,
                             pos=(np.array(j["anchor_mm"], float) * 1e-3).tolist(), axis=list(j["axis"]),
                             range=[float(rng[0]), float(rng[1])], frictionloss=float(dyn["frictionloss_Nm"]),
                             damping=float(dyn["damping_Nms_per_rad"]), armature=float(dyn["armature_kg_m2"]))
            jt.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            jt.solref_friction = list(num["solreffriction"])
            jt.solimp_friction = list(num["solimpfriction"])
            jt.solref_limit = list(num["limit_solref"])
            jt.solimp_limit = list(num["limit_solimp"])
            out["joints"][j["name"]] = jt.name
        ms = kd["mass"]["links"][link] if kd.get("articulated") else kd["mass"]
        b.mass = float(ms["mass_g"]) * 1e-3
        b.ipos = (np.array(ms["com_mm"], float) * 1e-3).tolist()
        I = np.array(ms["inertia_kg_m2"], float)
        b.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
        b.explicitinertial = True
        bodies[link] = b
        out["links"][link] = b.name
    for i, (link, V, _src) in enumerate(hulls):
        mn = "col_%s_%04d" % (kind, i)
        if mn not in mesh_cache:
            m = spec.add_mesh(name=mn)
            m.uservert = (V * 1e-3).flatten().tolist()
            mesh_cache[mn] = m
        g = bodies[link].add_geom(name="%s__c%04d" % (prefix, i), type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn,
                                  group=3, rgba=list(col_rgba), mass=0.0, contype=contype, conaffinity=conaffinity)
        out["geoms"].append(g.name)
    names = [n for n, _p, _j in kit.links(kind)]
    for a_ in range(len(names)):
        for b_ in range(a_ + 1, len(names)):
            spec.add_exclude(bodyname1=bodies[names[a_]].name, bodyname2=bodies[names[b_]].name)
    lp = jm["kinds"].get(kind, {}).get("loop") if kind in jm["kinds"] else None
    if lp:
        eq = spec.add_equality(name="%s__loop" % prefix, type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_BODY,
                               name1=bodies[lp["link_a"]].name, name2=bodies[lp["link_b"]].name)
        eq.data[0:3] = (np.array(lp["point_mm"], float) * 1e-3).tolist()
        eq.solref = list(num["loop_solref"])
        eq.solimp = list(num["loop_solimp"])
        out["loop"] = eq.name
    return out
