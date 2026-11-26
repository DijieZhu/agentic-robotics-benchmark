"""Is a point inside the solid of a geom? (rule Z5: a camera centre inside any geom zeroes that frame).

    sol = Solids(model, scene=ray_scene)      # every geom of the compiled twin (ray_scene: raycast.RayScene)
    inside, which = sol.contains(points)      # (n,) bool and the first geom index that contains each point (-1)

The solid of a geom is the closed set the simulator collides with: a primitive is itself (box, sphere, capsule,
cylinder, ellipsoid, each by its closed form); a mesh is the convex hull of its vertices, tested with the hull's
face planes (n . x <= b for every face); a plane is the half-space behind it. A point on the surface counts as
inside. With a ray scene given, a mesh's hull is the one the simulator computed for the scene's vertex-only copy
(the very shape the rays meet); without one it is computed here (Qhull, through scipy). This test runs before any ray is cast: a ray that starts inside a solid returns the distance to the
face where it leaves it, so a camera inside a solid would see an empty room.
"""
import numpy as np


def _planes_of(V, F):
    """Outward face planes (normals (k, 3), offsets (k,)) of a convex hull given by its triangles."""
    T = V[F]
    cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ar = np.linalg.norm(cr, axis=1)
    keep = ar > 0
    n = cr[keep] / ar[keep][:, None]
    cen = V[np.unique(F)].mean(axis=0)
    flip = np.einsum("ij,ij->i", n, T[keep, 0] - cen) < 0
    n[flip] *= -1.0
    return n, np.einsum("ij,ij->i", n, T[keep, 0])


def _scene_hull(scene, k):
    """Planes, in the scene geom's own frame, of the hull the simulator built for the ray scene's mesh geom k, and
    that geom's world pose."""
    sm, sd = scene.model, scene.data
    mid = int(sm.geom_dataid[k])
    va, vn = int(sm.mesh_vertadr[mid]), int(sm.mesh_vertnum[mid])
    fa, fn = int(sm.mesh_faceadr[mid]), int(sm.mesh_facenum[mid])
    V = np.asarray(sm.mesh_vert[va:va + vn], float)
    F = np.asarray(sm.mesh_face[fa:fa + fn], np.int64)
    n, b = _planes_of(V, F)
    return (np.asarray(sd.geom_xpos[k], float).copy(), np.asarray(sd.geom_xmat[k], float).reshape(3, 3).copy(),
            np.concatenate([n, -b[:, None]], axis=1))


class Solids:
    def __init__(self, model, data=None, geoms=None, scene=None):
        import mujoco
        m = model
        if data is None:
            data = mujoco.MjData(m)
            mujoco.mj_kinematics(m, data)           # poses only: no collision pass on the twin
        d = data
        self.src = np.arange(m.ngeom, dtype=int) if geoms is None else np.asarray(list(geoms), int)
        self.items = []
        T = mujoco.mjtGeom
        row = {}
        if scene is not None:
            row = {int(g): k for k, g in enumerate(scene.src)}
        for g in self.src:
            g = int(g)
            t = int(m.geom_type[g])
            pos = np.asarray(d.geom_xpos[g], float).copy()
            R = np.asarray(d.geom_xmat[g], float).reshape(3, 3).copy()
            size = np.asarray(m.geom_size[g], float).copy()
            if t == int(T.mjGEOM_MESH):
                if g in row:
                    spos, sR, planes = _scene_hull(scene, row[g])
                    self.items.append(("mesh", g, spos, sR, planes))
                else:
                    from scipy.spatial import ConvexHull
                    mid = int(m.geom_dataid[g])
                    va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                    V = np.asarray(m.mesh_vert[va:va + vn], float)
                    self.items.append(("mesh", g, pos, R, ConvexHull(V).equations.copy()))
            elif t == int(T.mjGEOM_BOX):
                self.items.append(("box", g, pos, R, size))
            elif t == int(T.mjGEOM_SPHERE):
                self.items.append(("sphere", g, pos, R, size))
            elif t == int(T.mjGEOM_CAPSULE):
                self.items.append(("capsule", g, pos, R, size))
            elif t == int(T.mjGEOM_CYLINDER):
                self.items.append(("cylinder", g, pos, R, size))
            elif t == int(T.mjGEOM_ELLIPSOID):
                self.items.append(("ellipsoid", g, pos, R, size))
            elif t == int(T.mjGEOM_PLANE):
                self.items.append(("plane", g, pos, R, size))
            else:
                raise ValueError("geom %d has a type the inside test does not know (%d)" % (g, t))

    def contains(self, points):
        """(inside (n,), first containing geom index (n,), -1 if none) of world points (n, 3)."""
        P = np.asarray(points, float).reshape(-1, 3)
        n = len(P)
        which = np.full(n, -1, int)
        for kind, g, pos, R, prm in self.items:
            todo = which < 0
            if not todo.any():
                break
            L = (P[todo] - pos) @ R                     # local coordinates (R turns local vectors into world ones)
            if kind == "box":
                ins = np.all(np.abs(L) <= prm[:3], axis=1)
            elif kind == "sphere":
                ins = np.einsum("ij,ij->i", L, L) <= prm[0] * prm[0]
            elif kind == "capsule":
                z = np.clip(L[:, 2], -prm[1], prm[1])
                q = L - np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
                ins = np.einsum("ij,ij->i", q, q) <= prm[0] * prm[0]
            elif kind == "cylinder":
                ins = (np.abs(L[:, 2]) <= prm[1]) & (L[:, 0] ** 2 + L[:, 1] ** 2 <= prm[0] * prm[0])
            elif kind == "ellipsoid":
                ins = np.sum((L / prm[:3]) ** 2, axis=1) <= 1.0
            elif kind == "mesh":
                ins = np.max(L @ prm[:, :3].T + prm[:, 3], axis=1) <= 0.0
            else:                                       # plane: the half-space behind it (local z <= 0)
                ins = L[:, 2] <= 0.0
            idx = np.nonzero(todo)[0]
            which[idx[ins]] = g
        return which >= 0, which
