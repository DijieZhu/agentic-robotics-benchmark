"""Reader of the hidden kit of the strut assembly cell (the `gt/` tree: kit.json, mating_graph.json, meshes/visual/,
meshes/collision/ with collision.json) and the MuJoCo geoms of one module build.

    kit = StrutKit(gt_dir)
    kit.units                       {unit: {"type": "M1", "build": "M1a", "colour": ...}}  (kit.json units_of_the_cell)
    kit.mates(scope)                the mating graph's mates of a scope ("single" | "default")
    kit.collision_hulls(build)      [(vertices mm (n, 3), piece, kind)] in the build's module frame
    kit.visual_meshes(build)        [(material, V mm, F)] (the authors' triangles, by material)
    kit.inertial(build)             (mass kg, com m, iquat wxyz, principal inertia kg m^2)
    add_module(spec, body, kit, build, prefix, ...)   collision (and optionally visual) geoms + the explicit inertial

Lengths in the kit are mm; MuJoCo gets metres. Every visual triangle is a triangle of the authors' whole-strut mesh;
the collision pieces are derived from them (fine_collision.py).
"""
import json
import os

import numpy as np


def read_ply(path):
    """binary little-endian PLY (float x y z, uchar-int faces) -> (V float64 (n, 3), F int64 (m, 3))"""
    data = open(path, "rb").read()
    end = data.find(b"end_header\n")
    head = data[:end].decode("ascii").splitlines()
    nv = nf = 0
    for line in head:
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


def quat_wxyz(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


class StrutKit(object):
    def __init__(self, gt_dir):
        self.dir = os.path.abspath(gt_dir)
        self.kit = json.load(open(os.path.join(self.dir, "kit.json")))
        self.graph = json.load(open(os.path.join(self.dir, "mating_graph.json")))
        cj = os.path.join(self.dir, "meshes", "collision", "collision.json")
        self.collision = json.load(open(cj)) if os.path.isfile(cj) else None
        self.units = self.kit["units_of_the_cell"]
        self.modules = self.kit["modules"]
        self._ply = {}

    def mates(self, scope="default"):
        ids = set(self.graph["scopes"][scope])
        return [m for m in self.graph["mates"] if m["id"] in ids]

    def units_of_type(self, t):
        return [u for u, v in self.units.items() if v["type"] == t]

    def _ply_cached(self, rel):
        if rel not in self._ply:
            self._ply[rel] = read_ply(os.path.join(self.dir, rel))
        return self._ply[rel]

    def collision_hulls(self, build):
        b = self.collision["builds"][build]
        V, _F = self._ply_cached(b["file"])
        return [(V[h["v0"]:h["v0"] + h["nv"]], h["piece"], h["kind"]) for h in b["hulls"]]

    def visual_meshes(self, build):
        out = []
        for v in self.modules[build]["visual"]:
            V, F = self._ply_cached(v["file"])
            out.append((v["material"], V, F))
        return out

    def inertial(self, build):
        ms = self.modules[build]["mass"]
        mass = float(ms["mass_g"]) * 1e-3
        com = np.array(ms["com_mm_module"], float) * 1e-3
        I = np.array(ms["inertia_kg_m2_module"], float)
        w, R = np.linalg.eigh(I)
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]
        return mass, com, quat_wxyz(R), np.maximum(w, 1e-12)

    def capture(self):
        return dict(self.collision["capture"]) if self.collision else None

    def bores(self):
        return dict(self.collision["bores"]) if self.collision else None


def add_module(spec, body, kit, build, prefix, visual=False, collision=True, materials=None, mesh_cache=None,
               col_rgba=(0.9, 0.3, 0.3, 0.3), inertial=True):
    """the build's geoms on `body` (its frame = the module frame); meshes are added once per build (`mesh_cache`)"""
    import mujoco
    mesh_cache = {} if mesh_cache is None else mesh_cache
    geoms = []
    if collision:
        for i, (V, _pid, _k) in enumerate(kit.collision_hulls(build)):
            mn = "col_%s_%03d" % (build, i)
            if mn not in mesh_cache:
                ms = spec.add_mesh(name=mn)
                ms.uservert = (V * 1e-3).flatten().tolist()
                mesh_cache[mn] = ms
            g = body.add_geom(name="%s_col%03d" % (prefix, i), type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn, group=3,
                              rgba=list(col_rgba), mass=0.0)
            geoms.append(g)
    if visual:
        for mat, V, F in kit.visual_meshes(build):
            mn = "vis_%s_%s" % (build, mat)
            if mn not in mesh_cache:
                ms = spec.add_mesh(name=mn)
                ms.uservert = (V * 1e-3).flatten().tolist()
                ms.userface = F.flatten().tolist()
                ms.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                mesh_cache[mn] = ms
            kw = {"material": (materials or {}).get(mat)} if materials and mat in materials else {}
            body.add_geom(name="%s_vis_%s" % (prefix, mat), type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn, contype=0,
                          conaffinity=0, group=2, mass=0.0, **kw)
    if inertial:
        mass, com, iq, diag = kit.inertial(build)
        body.mass = mass
        body.ipos = com.tolist()
        body.iquat = iq.tolist()
        body.inertia = diag.tolist()
        body.explicitinertial = True
    return geoms
