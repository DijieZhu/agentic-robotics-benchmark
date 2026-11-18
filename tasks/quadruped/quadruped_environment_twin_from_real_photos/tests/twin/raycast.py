"""Ray casting against a twin's collision geometry (the shapes MuJoCo collides with), at qpos0.

    sc = static_scene(twin)            # the environment only (world-welded bodies outside units and gates)
    sc = full_scene(twin)              # everything that collides: environment, units, gates
    dist, owner, src, nrm = sc.cast(origin, dirs)          # one origin, many unit directions (mj_multiRay)
    dist, owner, src, nrm = cast_grouped(sc, origins, dirs)  # many rays, grouped by identical origins
    MappedScene(sc, simmap)            # the same scene seen from another frame: rays given in frame F, simmap F -> scene

A ray scene is a separate MjModel with every chosen geom as a static world geom at its qpos0 pose: a mesh becomes a
vertex-only mesh (MuJoCo ray-casts those as their convex hull, which is what it collides with; a mesh with faces
would be cast as its triangles), planes are infinite (as in collisions), primitives are themselves (a twin has no
height field: the gate refuses them); rgba alpha 1, no material, group 0, so no geom is skipped by the ray filter. Distances are
along unit directions; misses are -1 (owner and src -1). The cutoff is 1000 m: MuJoCo culls a whole geom by its
bounding sphere against the cutoff, and a plane's sphere has radius 0.
"""
import numpy as np

CUTOFF_M = 1000.0


class RayScene:
    """owners: one key per geom (e.g. "env", a unit body name, "gate:<body>"); owner_index refers to `self.keys`."""

    def __init__(self, model, data, geoms, owners=None):
        import mujoco
        m, d = model, data
        self.src = np.asarray(list(geoms), int)
        owners = list(owners) if owners is not None else ["env"] * len(self.src)
        self.keys = sorted(set(owners))
        kidx = {k: i for i, k in enumerate(self.keys)}
        self.owner_of = np.asarray([kidx[o] for o in owners], int)
        spec = mujoco.MjSpec()
        spec.compiler.degree = False
        spec.compiler.usethread = False      # as twin_gate.parse_copy: no thread per mesh under the address limit
        for k, g in enumerate(self.src):
            g = int(g)
            t = int(m.geom_type[g])
            geom = spec.worldbody.add_geom()
            geom.name = "g%d" % k
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.asarray(d.geom_xmat[g], float))
            geom.pos = np.asarray(d.geom_xpos[g], float)
            geom.quat = q
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = 0
            geom.rgba = [1.0, 1.0, 1.0, 1.0]
            if t == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid = int(m.geom_dataid[g])
                va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                mesh = spec.add_mesh()
                mesh.name = "m%d" % k
                mesh.uservert = np.asarray(m.mesh_vert[va:va + vn], float).ravel().tolist()
                mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                geom.type = mujoco.mjtGeom.mjGEOM_MESH
                geom.meshname = mesh.name
            elif t == int(mujoco.mjtGeom.mjGEOM_PLANE):
                geom.type = mujoco.mjtGeom.mjGEOM_PLANE
                geom.size = [0.0, 0.0, 1.0]                      # infinite, as in collisions
            else:
                geom.type = mujoco.mjtGeom(t)
                geom.size = np.asarray(m.geom_size[g], float)
        self.n = len(self.src)
        if self.n:
            self.model = spec.compile()
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)
        else:
            self.model = self.data = None

    def cast(self, origin, dirs, cutoff=CUTOFF_M):
        """(dist (n,), owner index (n,), source geom (n,), normal (n, 3)) of rays from one origin."""
        import mujoco
        D = np.ascontiguousarray(np.asarray(dirs, float).reshape(-1, 3))
        n = len(D)
        gid = np.full(n, -1, np.int32)
        dist = np.full(n, -1.0)
        nrm = np.zeros(3 * n)
        if n and self.model is not None:
            mujoco.mj_multiRay(self.model, self.data, np.asarray(origin, float).reshape(3).copy(), D.ravel(), None, 1,
                               -1, gid, dist, nrm, n, float(cutoff))
        hit = gid >= 0
        owner = np.where(hit, self.owner_of[np.where(hit, gid, 0)] if self.n else -1, -1)
        src = np.where(hit, self.src[np.where(hit, gid, 0)] if self.n else -1, -1)
        dist = np.where(hit, dist, -1.0)
        return dist, owner.astype(int), src.astype(int), nrm.reshape(n, 3)


class MappedScene:
    """A RayScene seen from another frame F: rays are given in F; simmap maps F -> the scene's frame
    (x_scene = s R x_F + t). Distances and normals come back in F."""

    def __init__(self, scene, simmap):
        self.scene = scene
        self.map = simmap
        self.keys = scene.keys

    def cast(self, origin, dirs, cutoff=CUTOFF_M):
        o = self.map.to_site(np.asarray(origin, float).reshape(1, 3))[0]
        u = self.map.dir_to_site(np.asarray(dirs, float).reshape(-1, 3))
        u = u / np.linalg.norm(u, axis=1)[:, None]
        dist, owner, src, nrm = self.scene.cast(o, u, cutoff)
        dist = np.where(dist >= 0, dist / self.map.s, -1.0)
        nrm = self.map.dir_to_agent(nrm)
        return dist, owner, src, nrm


def cast_grouped(scene, origins, dirs, cutoff=CUTOFF_M):
    """Cast rays (origins (n, 3), unit dirs (n, 3)): one mj_multiRay per distinct origin; results in input order."""
    O = np.asarray(origins, float).reshape(-1, 3)
    D = np.asarray(dirs, float).reshape(-1, 3)
    n = len(O)
    dist = np.full(n, -1.0)
    owner = np.full(n, -1, int)
    src = np.full(n, -1, int)
    nrm = np.zeros((n, 3))
    if not n:
        return dist, owner, src, nrm
    uo, inv = np.unique(O, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    for k in range(len(uo)):
        idx = np.nonzero(inv == k)[0]
        d_, o_, s_, n_ = scene.cast(uo[k], D[idx], cutoff)
        dist[idx], owner[idx], src[idx], nrm[idx] = d_, o_, s_, n_
    return dist, owner, src, nrm


def static_scene(twin):
    """The twin's environment: every collision geom of a world-welded body outside the units and the gates."""
    return RayScene(twin.model, twin.data, twin.static_geoms(), ["env"] * len(twin.static_geoms()))


def full_scene(twin, units="all"):
    """Everything that collides at qpos0: the environment ("env"), the units (owner = body name; units="ok" keeps
    only units the gate did not exclude) and the gates (owner "gate:<body>")."""
    geoms, owners = list(twin.static_geoms()), ["env"] * len(twin.static_geoms())
    for u in twin.units:
        if u["body_id"] < 0 or (units == "ok" and u["status"] != "ok"):
            continue
        geoms += u["geoms"]
        owners += [u["body"]] * len(u["geoms"])
    for gt in twin.gates:
        geoms += gt["geoms"]
        owners += ["gate:" + gt["body"]] * len(gt["geoms"])
    order = np.argsort(np.asarray(geoms, int), kind="stable")
    geoms = [geoms[i] for i in order]
    owners = [owners[i] for i in order]
    return RayScene(twin.model, twin.data, geoms, owners)
