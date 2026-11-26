"""Rays against a twin's collision geometry: the shapes the robot would collide with.

    sc = RayScene(model)                      # every geom of the compiled twin
    dist, src, nrm = sc.cast(origin, dirs, cutoff)          # one origin, many unit directions (one mj_multiRay)
    rng = sc.cast_frame(T_twin_depth, rays, cutoff)         # a depth frame: the ray table in the depth frame
    sc.report_lines()                         # what the rebuild did, printed by the gate and the check tool

A ray scene is a separate model in which every geom of the twin is a static world geom at its pose. Every geom
counts as collision geometry, whatever its contype, conaffinity, group or colour:
  - a mesh becomes a vertex-only copy of its vertices, which the simulator casts as their convex hull, the shape it
    collides with (a mesh with faces would be cast as its triangles, so a concave mesh would show its inside);
  - primitives are themselves; a plane is infinite, as in collisions;
  - every geom is rebuilt opaque (alpha 1), in group 0 and without material: the simulator's rays skip a geom whose
    alpha is exactly 0 and filter by group, and none of that may hide geometry from the rays.
Distances are along unit directions; a miss is -1 (and source -1). The cutoff given to the cast culls whole geoms by
their bounding sphere (a plane's sphere has radius 0, so it is culled by its origin) and does not clip distances:
the configuration's gate.ray_cutoff is far beyond any cabin, and the caller clips ranges itself. A ray that starts
inside a solid returns the distance to the face where it leaves the solid: test camera centres with insidesolid
before casting.
"""
import numpy as np

from . import need
from .posechain import depth_pose


def _eff_alpha(model, g):
    mid = int(model.geom_matid[g])
    if mid >= 0:
        return float(model.mat_rgba[mid][3])
    return float(model.geom_rgba[g][3])


def _name(model, g):
    import mujoco
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
    return nm if nm else "geom %d" % int(g)


class RayScene:
    """The ray scene of a compiled twin. src = the twin's geom index of each hit."""

    def __init__(self, model, data=None, geoms=None):
        import mujoco
        m = model
        if data is None:
            data = mujoco.MjData(m)
            mujoco.mj_kinematics(m, data)           # poses only: no collision pass on the twin
        d = data
        self.src = np.arange(m.ngeom, dtype=int) if geoms is None else np.asarray(list(geoms), int)
        self.forced = []            # (geom index, name, what was changed)
        self.n_casts = 0
        spec = mujoco.MjSpec()
        spec.compiler.degree = False
        spec.compiler.usethread = False
        for k, g in enumerate(self.src):
            g = int(g)
            t = int(m.geom_type[g])
            changes = []
            a = _eff_alpha(m, g)
            if a == 0.0:
                changes.append("alpha 0 (the rays would have skipped it)")
            elif a < 1.0:
                changes.append("alpha %.3g" % a)
            if int(m.geom_matid[g]) >= 0:
                changes.append("material removed")
            if int(m.geom_group[g]) != 0:
                changes.append("group %d set to 0" % int(m.geom_group[g]))
            if changes:
                self.forced.append((g, _name(m, g), changes))
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
            elif t in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_SPHERE),
                       int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                       int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):
                geom.type = mujoco.mjtGeom(t)
                geom.size = np.asarray(m.geom_size[g], float)
            else:
                raise ValueError("geom %d has a type the ray scene does not cast (%d)" % (g, t))
        self.n = len(self.src)
        if self.n:
            self.model = spec.compile()
            self.data = mujoco.MjData(self.model)
            mujoco.mj_kinematics(self.model, self.data)
        else:
            self.model = self.data = None

    def report_lines(self):
        lines = ["every geom (%d) is rebuilt opaque, in group 0 and without material before the rays are cast; "
                 "meshes are cast as their convex hulls" % self.n]
        if self.forced:
            lines.append("forced opaque or changed: %d geom(s)" % len(self.forced))
            for g, nm, ch in self.forced[:50]:
                lines.append("  %s: %s" % (nm, "; ".join(ch)))
            if len(self.forced) > 50:
                lines.append("  ... and %d more" % (len(self.forced) - 50))
        return lines

    def cast(self, origin, dirs, cutoff):
        """(dist (n,), source geom (n,), normal (n, 3)) of rays from one origin along unit directions dirs (n, 3).
        A direction with a non-finite component is a miss."""
        import mujoco
        D = np.ascontiguousarray(np.asarray(dirs, float).reshape(-1, 3))
        n = len(D)
        bad = ~np.all(np.isfinite(D), axis=1)
        Dc = D.copy()
        Dc[bad] = [0.0, 0.0, 1.0]
        gid = np.full(n, -1, np.int32)
        dist = np.full(n, -1.0)
        nrm = np.zeros(3 * n)
        self.n_casts += 1
        if n and self.model is not None:
            mujoco.mj_multiRay(self.model, self.data, np.asarray(origin, float).reshape(3).copy(), Dc.ravel(), None,
                               1, -1, gid, dist, nrm, n, float(cutoff))
        hit = (gid >= 0) & ~bad
        src = np.where(hit, self.src[np.where(hit, gid, 0)] if self.n else -1, -1)
        dist = np.where(hit, dist, -1.0)
        return dist, src.astype(int), nrm.reshape(n, 3)

    def cast_frame(self, T_twin_depth, rays, cutoff):
        """Range along each ray of a depth frame: rays (..., 3) unit vectors in the depth frame (NaN rays are
        misses), the depth camera placed by T_twin_depth. Returns ranges of rays' shape without the last axis,
        -1 where nothing is hit."""
        T = np.asarray(T_twin_depth, float)
        R = np.asarray(rays, float)
        shp = R.shape[:-1]
        dirs = R.reshape(-1, 3) @ T[:3, :3].T
        dist, _, _ = self.cast(T[:3, 3], dirs, cutoff)
        return dist.reshape(shp)


def cutoff_from_config(cfg):
    return float(need(cfg, "gate.ray_cutoff"))


def probe_frame(scene, solids, T_twin_wide, T_depth_wide, rays, cutoff):
    """One key frame: rule Z5 first (the wide or the depth camera's centre inside a geom's solid: no ray is cast and
    the frame is zero), then the depth frame's ranges. Returns {"z5": bool, "inside_geom": geom index or -1,
    "camera": "wide" / "depth" / None, "ranges": array or None, "T_twin_depth": 4 x 4}."""
    T_tw = np.asarray(T_twin_wide, float)
    T_td = depth_pose(T_tw, T_depth_wide)
    centres = np.array([T_tw[:3, 3], T_td[:3, 3]])
    inside, which = solids.contains(centres)
    if inside.any():
        k = int(np.argmax(inside))
        return {"z5": True, "inside_geom": int(which[k]), "camera": ("wide", "depth")[k], "ranges": None,
                "T_twin_depth": T_td}
    return {"z5": False, "inside_geom": -1, "camera": None, "ranges": scene.cast_frame(T_td, rays, cutoff),
            "T_twin_depth": T_td}
