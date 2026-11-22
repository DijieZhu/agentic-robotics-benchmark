"""Dressing the render twin's spec (cell internals; look.py builds the twin): textured box meshes, faceted rubble chunks
with texture coordinates, the terrain's surface meshes by ground class, and the sky light each surface receives.

Everything here edits a COPY of the scene's spec (the render twin's). The physics model is never touched, and no random
generator of the cell is drawn from (the few random choices here use their own seeded generators).

* Box geoms become meshes with 4 vertices per face and texture coordinates in metres along the face, so a material
  repeats at its real size on every face (Filament's built-in box maps a uniform texture onto the top and bottom faces
  only).
* The rubble chunks (convex meshes without texture coordinates) get faceted copies with planar coordinates per face.
* The heightfield is drawn by meshes on its own grid, 4 mm above it, one per ground class and light level (asphalt,
  paving, concrete, tiles, rubble, the dark of the pits), with tiled materials; the heightfield itself is drawn
  transparent.
* Sky light: Filament's image light is not shadowed, so a surface indoors would be lit as if it stood in the street.
  Every static surface gets the share of the sky it sees as the ambient occlusion of its material (`LEVELS`): the
  street 1, inside a bay 0.24, the upper storeys 0.4, a crawl void under a dropped slab 0.06. The direct light (the sun,
  the lamps) is shadowed by Filament as usual.
"""

import math

import mujoco
import numpy as np

from . import pbr

__all__ = ["LEVELS", "classify_point", "level_of_geoms", "box_mesh", "prim_mesh", "faceted", "robot_bodies", "dress",
           "add_terrain", "lowpoly"]

LEVELS = {"ext": 1.0, "open": 0.50, "upper": 0.30, "int": 0.15, "void": 0.02}
# the robot is drawn in three versions (its surfaces at these light levels); each frame shows the one for where it is
ROBOT_LEVELS = ("ext", "int", "void")
ROBOT_LEVEL_OF = {"ext": "ext", "open": "ext", "upper": "int", "int": "int", "void": "void"}
STOREY, CEIL, T_EXT = 3.2, 3.0, 0.30
TERRAIN_DZ = 0.004
# ground classes of the site's painter -> (texture set, tint)
GROUND_CLASS = {0: ("asphalt", (1.0, 1.0, 1.0)), 1: ("paving", (1.0, 1.0, 1.0)), 2: ("ground_concrete", (1.0, 1.0, 1.0)),
                3: ("tiles", (1.0, 1.0, 1.0)), 4: ("rubble", (1.0, 1.0, 1.0)), 5: ("rubble_dark", (0.55, 0.54, 0.52))}


def _bays(plan):
    return [b for b in plan["bays"].values()]


def classify_point(plan, W, D, p, ext_band=T_EXT * 0.9, z_low=None):
    """The light level key of a point (x, y, z) of the site (`z_low`: the bottom of the surface it stands for, which
    decides whether it lies in a crawl void)."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    zl = z if z_low is None else float(z_low)
    if not (ext_band < x < W - ext_band and ext_band < y < D - ext_band):
        return "ext"
    if z >= STOREY:
        return "upper"
    for b in _bays(plan):
        if b["x0"] <= x <= b["x1"] and b["y0"] <= y <= b["y1"]:
            if b["state"] == "void":
                return "void" if zl < b.get("slab_h", 0.9) + 0.02 else "open"
            return "int"
    return "int"


def level_of_geoms(scene, robot_bodies):
    """Light level key of every geom of the physics model (None for the robot's, which move)."""
    m = scene.model
    d = mujoco.MjData(m)
    d.qpos[:] = scene.data.qpos
    mujoco.mj_kinematics(m, d)
    plan = scene.site.plan
    W, D = scene.site.W, scene.site.D
    out = []
    for i in range(m.ngeom):
        if int(m.geom_bodyid[i]) in robot_bodies:
            out.append(None)
            continue
        R = d.geom_xmat[i].reshape(3, 3)
        half = np.asarray(m.geom_aabb[i][3:], float)
        cz = float((d.geom_xpos[i] + R @ np.asarray(m.geom_aabb[i][:3], float))[2])
        ez = float(np.abs(R[2]) @ half)
        out.append(classify_point(plan, W, D, d.geom_xpos[i], z_low=cz - ez))
    return out


def box_mesh(sx, sy, sz):
    """A box of half-sizes (sx, sy, sz): 24 vertices, 12 outward triangles, texture coordinates in metres."""
    s = np.array([sx, sy, sz], float)
    V, T, F = [], [], []
    for ax, a1, a2 in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        for sg in (1.0, -1.0):
            base = len(V)
            for c1, c2 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[ax], p[a1], p[a2] = sg * s[ax], c1 * s[a1], c2 * s[a2]
                V.append(p)
                T.append((p[a1] + s[a1], p[a2] + s[a2]))
            nrm = np.zeros(3)
            nrm[ax] = sg
            if np.dot(np.cross(V[base + 1] - V[base], V[base + 2] - V[base]), nrm) > 0:
                F += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
            else:
                F += [(base, base + 2, base + 1), (base, base + 3, base + 2)]
    return np.array(V), np.array(T), np.array(F, int)


def prim_mesh(kind, size, seg=12):
    """A low-poly mesh of a MuJoCo primitive (its local frame, z the axis): (vertices, texture coordinates in metres,
    triangles). Filament tessellates primitives finely, which a CPU rasteriser pays for on every frame."""
    V, T, F = [], [], []
    if kind in ("cylinder", "capsule"):
        r, h = float(size[0]), float(size[1])
        rings = []
        if kind == "capsule":
            nh = max(2, seg // 4)
            for k in range(nh, 0, -1):                       # the lower cap, from its pole up
                a = (k / nh) * (math.pi / 2)
                rings.append((-h - r * math.sin(a), r * math.cos(a)))
            rings += [(-h, r), (h, r)]
            for k in range(1, nh + 1):
                a = (k / nh) * (math.pi / 2)
                rings.append((h + r * math.sin(a), r * math.cos(a)))
            rings = [(-h - r, 0.0)] + rings + [(h + r, 0.0)]
        else:
            rings = [(-h, 0.0), (-h, r), (h, r), (h, 0.0)]
        circ = 2 * math.pi * r
        acc = 0.0
        for i, (z, rr) in enumerate(rings):
            if i:
                acc += math.hypot(z - rings[i - 1][0], rr - rings[i - 1][1])
            for j in range(seg + 1):
                t = 2 * math.pi * j / seg
                V.append((rr * math.cos(t), rr * math.sin(t), z))
                T.append((circ * j / seg, acc))
        n = seg + 1
        for i in range(len(rings) - 1):
            for j in range(seg):
                a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j
                F += [(a, b, c), (a, c, d)]
        if kind == "cylinder":                               # flat caps with their own vertices
            for zc, sg in ((-h, -1), (h, 1)):
                base = len(V)
                V.append((0.0, 0.0, zc))
                T.append((0.0, 0.0))
                for j in range(seg):
                    t = 2 * math.pi * j / seg
                    V.append((r * math.cos(t), r * math.sin(t), zc))
                    T.append((r * math.cos(t), r * math.sin(t)))
                for j in range(seg):
                    a, b = base + 1 + j, base + 1 + (j + 1) % seg
                    F.append((base, b, a) if sg < 0 else (base, a, b))
    else:                                                    # ellipsoid / sphere: a UV sphere scaled
        a = np.array(size[:3] if kind == "ellipsoid" else [size[0]] * 3, float)
        nr = max(4, seg // 2)
        for i in range(nr + 1):
            ph = math.pi * i / nr
            for j in range(seg + 1):
                t = 2 * math.pi * j / seg
                V.append((a[0] * math.sin(ph) * math.cos(t), a[1] * math.sin(ph) * math.sin(t), -a[2] * math.cos(ph)))
                T.append((2 * math.pi * a.mean() * j / seg, math.pi * a.mean() * i / nr))
        n = seg + 1
        for i in range(nr):
            for j in range(seg):
                p0, p1, p2, p3 = i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j
                F += [(p0, p1, p2), (p0, p2, p3)]
    V = np.array(V, float)
    F = np.array(F, int)
    # outward winding: flip triangles whose normal points at the centre
    c = V[F].mean(1)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    bad = (nrm * c).sum(1) < 0
    F[bad] = F[bad][:, [0, 2, 1]]
    keep = np.linalg.norm(nrm, axis=1) > 1e-12
    return V, np.array(T, float), F[keep]


def _fromto_pose(ft):
    a, b = np.asarray(ft[:3], float), np.asarray(ft[3:6], float)
    d = b - a
    L = float(np.linalg.norm(d))
    z = d / L if L > 1e-9 else np.array([0.0, 0.0, 1.0])
    v = np.cross([0.0, 0.0, 1.0], z)
    s_, c_ = float(np.linalg.norm(v)), float(z[2])
    if s_ < 1e-9:
        q = [1.0, 0.0, 0.0, 0.0] if c_ > 0 else [0.0, 1.0, 0.0, 0.0]
    else:
        ang = math.atan2(s_, c_)
        v = v / s_
        q = [math.cos(ang / 2), *(v * math.sin(ang / 2))]
    return (a + b) / 2, q, L / 2


def faceted(verts):
    """A convex point set as a faceted mesh: (vertices, texture coordinates in metres, triangles)."""
    from scipy.spatial import ConvexHull
    P = np.asarray(verts, float).reshape(-1, 3)
    hull = ConvexHull(P)
    c = P.mean(0)
    V, T, F = [], [], []
    for tri in hull.simplices:
        a, b, e = P[tri[0]], P[tri[1]], P[tri[2]]
        n = np.cross(b - a, e - a)
        if np.dot(n, a - c) < 0:
            b, e = e, b
            n = -n
        ax = int(np.argmax(np.abs(n)))
        k1, k2 = [i for i in range(3) if i != ax]
        base = len(V)
        for q in (a, b, e):
            V.append(q)
            T.append((q[k1], q[k2]))
        F.append((base, base + 1, base + 2))
    return np.array(V), np.array(T), np.array(F, int)


def _shell(me):
    try:
        me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    except Exception:  # noqa: BLE001
        pass


def terrain_meshes(site, plan):
    """{(level, ground class): (vertices, texture coordinates, triangles)} covering the heightfield on its own grid."""
    H, G = site.H, site.ground
    ny, nx = H.shape
    gx = site.hx0 + np.arange(nx) * (site.hx1 - site.hx0) / (nx - 1)
    gy = site.hy0 + np.arange(ny) * (site.hy1 - site.hy0) / (ny - 1)
    # each cell's class and light level (at the cell's centre; the floor's level inside a bay)
    cx = 0.5 * (gx[:-1] + gx[1:])
    cy = 0.5 * (gy[:-1] + gy[1:])
    cls = G[:-1, :-1]
    lvl = np.full(cls.shape, "ext", object)
    for b in _bays(plan):
        ix = (cx >= b["x0"]) & (cx <= b["x1"])
        iy = (cy >= b["y0"]) & (cy <= b["y1"])
        key = "void" if b["state"] == "void" else "int"
        lvl[np.ix_(iy, ix)] = key
    hc = H[:-1, :-1]
    flat = ((np.abs(H[1:, :-1] - hc) < 0.002) & (np.abs(H[:-1, 1:] - hc) < 0.002) &
            (np.abs(H[1:, 1:] - hc) < 0.002))
    out = {}
    for key in np.unique(lvl):
        for c in np.unique(cls):
            mask = (lvl == key) & (cls == c)
            if not mask.any():
                continue
            Vs, Ts, Fs = [], [], []
            nv = 0
            # flat cells: merged into rectangles (runs along x, extended over equal runs of the next rows)
            fm = mask & flat
            done = np.zeros_like(fm)
            for j in range(fm.shape[0]):
                i = 0
                row = fm[j] & ~done[j]
                while i < fm.shape[1]:
                    if not row[i]:
                        i += 1
                        continue
                    i1 = i
                    z0 = hc[j, i]
                    while i1 + 1 < fm.shape[1] and row[i1 + 1] and abs(hc[j, i1 + 1] - z0) < 0.002:
                        i1 += 1
                    j1 = j
                    while j1 + 1 < fm.shape[0] and np.all(fm[j1 + 1, i:i1 + 1] & ~done[j1 + 1, i:i1 + 1]) and \
                            np.all(np.abs(hc[j1 + 1, i:i1 + 1] - z0) < 0.002):
                        j1 += 1
                    done[j:j1 + 1, i:i1 + 1] = True
                    xa, xb, ya, yb = gx[i], gx[i1 + 1], gy[j], gy[j1 + 1]
                    z = z0 + TERRAIN_DZ
                    Vs += [(xa, ya, z), (xb, ya, z), (xb, yb, z), (xa, yb, z)]
                    Ts += [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
                    Fs += [(nv, nv + 1, nv + 2), (nv, nv + 2, nv + 3)]
                    nv += 4
                    i = i1 + 1
            mask = mask & ~flat
            if not mask.any():
                if Vs:
                    out[(str(key), int(c))] = (np.array(Vs), np.array(Ts), np.array(Fs, int))
                continue
            jj, ii = np.nonzero(mask)               # cell rows (y), columns (x)
            corners = np.stack([jj * nx + ii, jj * nx + ii + 1, (jj + 1) * nx + ii + 1, (jj + 1) * nx + ii], 1)
            used, inv = np.unique(corners.ravel(), return_inverse=True)
            inv = inv.reshape(-1, 4)
            vy, vx = np.divmod(used, nx)
            V = np.stack([gx[vx], gy[vy], H[vy, vx] + TERRAIN_DZ], 1)
            T = np.stack([gx[vx], gy[vy]], 1)
            F = np.concatenate([inv[:, [0, 1, 2]], inv[:, [0, 2, 3]]], 0)
            if Vs:
                F = np.concatenate([np.array(Fs, int), F + nv], 0)
                V = np.concatenate([np.array(Vs), V], 0)
                T = np.concatenate([np.array(Ts), T], 0)
            out[(str(key), int(c))] = (V, T, F)
    return out


_PRIMS = {mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
          mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere"}


def lowpoly(spec, g, cache, seg=12, prefix="lkprim"):
    """Replace a primitive geom of a render twin's spec by a low-poly mesh in the same place."""
    kind = _PRIMS[g.type]
    ft = np.array(g.fromto, float)
    size = np.array(g.size, float)
    if kind in ("capsule", "cylinder") and np.all(np.isfinite(ft)) and np.any(ft != 0):
        pos, quat, half = _fromto_pose(ft)
        g.pos, g.quat = list(pos), list(quat)
        size = np.array([size[0], half, 0.0])
        g.fromto = [float("nan")] * 6
    key = (kind,) + tuple(round(float(v), 4) for v in size[:3])
    if key not in cache:
        V, T, F = prim_mesh(kind, size, seg)
        me = spec.add_mesh()
        me.name = "%s_%d" % (prefix, len(cache))
        me.uservert = V.flatten().tolist()
        me.usertexcoord = T.flatten().tolist()
        me.userface = F.flatten().tolist()
        me.smoothnormal = True
        _shell(me)
        cache[key] = me.name
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = cache[key]


def robot_bodies(model):
    """Body ids of the robot (the chassis' subtree) and of the kit it carries: the bodies that move with the robot."""
    ch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    root = int(model.body_rootid[ch])
    out = {b for b in range(model.nbody) if int(model.body_rootid[b]) == root}
    kit = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "kit")
    if kit >= 0:
        out.add(kit)
    return out


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _perimeter(c, W, D, band=0.45):
    """The outward normal of the building's outline a point lies on, or None."""
    cand = [(abs(c[0]), (-1.0, 0.0, 0.0)), (abs(c[0] - W), (1.0, 0.0, 0.0)), (abs(c[1]), (0.0, -1.0, 0.0)),
            (abs(c[1] - D), (0.0, 1.0, 0.0))]
    dist, n = min(cand, key=lambda t: t[0])
    inside = -0.6 <= c[0] <= W + 0.6 and -0.6 <= c[1] <= D + 0.6
    return np.array(n) if (dist < band and inside) else None


def dress(spec, scene, bank, body=None):
    """Re-dress the render twin's spec in place: every geom's material from `bank` at its light level, boxes with a
    textured material as textured meshes, the rubble chunks faceted. Returns the light level key per geom (None for
    the robot's)."""
    m = scene.model
    rb = robot_bodies(m)
    levels = level_of_geoms(scene, rb)
    what = {}
    for parts in getattr(scene.site, "parts", {}).values():
        for p in parts:
            what[p["name"]] = p.get("what")
    box_cache = {}
    prim_cache = {}
    W, D, plan = scene.site.W, scene.site.D, scene.site.plan
    halves = []

    def mesh_for(sz):
        sz = tuple(round(float(v), 4) for v in sz)
        if sz not in box_cache:
            V, T, F = box_mesh(*sz)
            me = spec.add_mesh()
            me.name = "lkbox_%d" % len(box_cache)
            me.uservert = V.flatten().tolist()
            me.usertexcoord = T.flatten().tolist()
            me.userface = F.flatten().tolist()
            _shell(me)
            box_cache[sz] = me.name
        return box_cache[sz]

    phys_mat = {}
    for i, g in enumerate(list(spec.geoms)):
        phys_mat[g.name] = g.material
        lvl = levels[i] if levels[i] is not None else "ext"
        robot = levels[i] is None
        if g.type == mujoco.mjtGeom.mjGEOM_HFIELD:
            g.material = bank.hidden()
            continue
        mat_before = g.material
        textured = bank.assign(g, lvl, robot, what.get(g.name))
        if textured and g.type == mujoco.mjtGeom.mjGEOM_BOX:
            size = np.array(g.size[:3], float)
            # an exterior wall: its outer half lit as the street, its inner half as the room behind it
            if body is not None and lvl == "ext" and not robot and int(scene.model.geom_bodyid[i]) == 0:
                c = np.array(g.pos, float)
                n = _perimeter(c, W, D)
                if n is not None:
                    R = _quat_mat(np.array(g.quat, float))
                    k = int(np.argmax(np.abs(R.T @ n)))
                    if abs((R.T @ n)[k]) > 0.99 and 2 * size[k] < 0.5 and size[(k + 1) % 3] > 0.05:
                        inner = classify_point(plan, W, D, c - n * 0.6)
                        if inner != "ext":
                            half = size.copy()
                            half[k] *= 0.5
                            sgn = float(np.sign((R.T @ n)[k]))
                            off = R[:, k] * sgn * half[k]
                            g.pos = (c + off).tolist()
                            g.size = half.tolist()
                            size = half
                            halves.append((c - off, np.array(g.quat, float), half.copy(), mat_before, inner))
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = mesh_for(size)
        elif g.type in _PRIMS:
            lowpoly(spec, g, prim_cache)
    # the robot's surfaces at the darker light levels: copies of its geoms on the same bodies
    robot_copies = {lv: [] for lv in ROBOT_LEVELS}
    for i, g in enumerate(list(spec.geoms)[:len(levels)]):
        if levels[i] is not None:
            continue
        robot_copies["ext"].append(g.name)
        src = phys_mat.get(g.name)
        for lv in ROBOT_LEVELS[1:]:
            c = g.parent.add_geom()
            c.name = "%s@%s" % (g.name, lv)
            c.type = g.type
            c.size = list(g.size)
            c.pos = list(g.pos)
            c.quat = list(g.quat)
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                c.meshname = g.meshname
            c.rgba = list(g.rgba)
            c.material = src or ""
            c.contype, c.conaffinity, c.group, c.mass = 0, 0, 1, 0.0
            bank.assign(c, lv, True, what.get(g.name))
            if c.type == mujoco.mjtGeom.mjGEOM_BOX and g.type == mujoco.mjtGeom.mjGEOM_MESH:
                c.type = mujoco.mjtGeom.mjGEOM_MESH
                c.meshname = g.meshname
            robot_copies[lv].append(c.name)
    bank.robot_copies = robot_copies
    for hk, (pos, quat, half, mat_before, inner) in enumerate(halves):
        g = body.add_geom()
        g.name = "lkhalf_%d" % hk
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = half.tolist()
        g.pos = pos.tolist()
        g.quat = quat.tolist()
        g.material = mat_before
        g.contype, g.conaffinity, g.group, g.mass = 0, 0, 2, 0.0
        bank.assign(g, inner, False, None)
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = mesh_for(half)
    for me in list(spec.meshes):
        if me.name.startswith("lkbox_") or not len(me.uservert) or len(me.usertexcoord):
            continue
        V, T, F = faceted(np.asarray(me.uservert))
        me.uservert = V.flatten().tolist()
        me.usertexcoord = T.flatten().tolist()
        me.userface = F.flatten().tolist()
        me.smoothnormal = False
        _shell(me)
    return levels


def add_terrain(spec, body, scene, bank):
    """The terrain's surface meshes as drawn-only geoms of `body` (the heightfield geom is drawn transparent)."""
    for (lvl, c), (V, T, F) in sorted(terrain_meshes(scene.site, scene.site.plan).items()):
        setname, tint = GROUND_CLASS[c]
        me = spec.add_mesh()
        me.name = "lkterrain_%s_%d" % (lvl, c)
        me.uservert = V.flatten().tolist()
        me.usertexcoord = T.flatten().tolist()
        me.userface = F.flatten().tolist()
        me.smoothnormal = True
        _shell(me)
        g = body.add_geom()
        g.name = me.name
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = me.name
        g.contype, g.conaffinity = 0, 0
        g.group = 2
        g.mass = 0.0
        g.material = bank.textured(setname, tint, lvl)


def _geom_mesh(spec_meshes, g):
    """Local (vertices, texture coordinates, triangles) of a mesh geom of the render twin, or None."""
    if g.type != mujoco.mjtGeom.mjGEOM_MESH or g.meshname not in spec_meshes:
        return None
    me = spec_meshes[g.meshname]
    V = np.asarray(me.uservert, float).reshape(-1, 3)
    T = np.asarray(me.usertexcoord, float).reshape(-1, 2)
    F = np.asarray(me.userface, int).reshape(-1, 3)
    if len(T) != len(V) or not len(F):
        return None
    sc = np.asarray(me.scale, float)
    return V * sc, T, F


# the drawn robot's looks that are a texture set of the look (worn paint, coated panels, brushed aluminium, the
# kit's cloth) with a tint; the others are plain
ROBOT_SETS = {"robot_paint": ("robot_paint", (0.93, 0.40, 0.05)), "robot_dark": ("robot_dark", (0.17, 0.18, 0.19)),
              "robot_shell": ("robot_dark", (0.21, 0.22, 0.23)), "robot_pad": ("robot_dark", (0.13, 0.13, 0.14)),
              "robot_alu": ("alu", (0.82, 0.83, 0.85)), "kit_bag": ("bag", (1.0, 1.0, 1.0))}


def draw_robot(spec, bank, looks):
    """The robot's and the kit's drawn parts from the look pack (robot.npz: per body and look one mesh in the body's
    frame, every part inside the task geom it stands for: boxes with broken edges, round tubes, one belt per track
    with its cleats, lenses, the kit as a pouch). On every body the pack knows, the task's own drawn geoms (all light
    levels) are deleted and one mesh geom per look and light level is added; the bodies get an explicit inertia (the
    render twin does kinematics only). Returns (geoms removed, geoms added), or None without the pack."""
    import os
    from . import pbr
    path = os.path.join(pbr.cache_dir(), "robot.npz")
    if not os.path.isfile(path):
        return None
    z = np.load(path)
    per_body = {}
    for k in z.files:
        if k.startswith("V:"):
            _, body, look = k.split(":")
            per_body.setdefault(body, []).append(look)
    level_of = {}
    for lv, names in bank.robot_copies.items():
        for n in names:
            level_of[n] = lv
    removed = added = 0
    for body in spec.bodies:
        if body.name not in per_body:
            continue
        for g in list(body.geoms):
            if g.name in level_of:
                bank.robot_copies[level_of[g.name]].remove(g.name)
                spec.delete(g)
                removed += 1
        body.explicitinertial = True
        body.mass = 1.0
        body.inertia = [0.01, 0.01, 0.01]
        for look in sorted(per_body[body.name]):
            key = "%s:%s" % (body.name, look)
            me = spec.add_mesh()
            me.name = "lkdrawn_%s_%s" % (body.name, look)
            me.uservert = z["V:" + key].astype(np.float64).flatten().tolist()
            me.usernormal = z["N:" + key].astype(np.float64).flatten().tolist()
            # texture coordinates in metres, box-mapped by each vertex's normal (the paint's and the coating's maps)
            Vr, Nr = z["V:" + key].astype(np.float64), z["N:" + key].astype(np.float64)
            ax = np.abs(Nr).argmax(1)
            Tr = np.where((ax == 0)[:, None], Vr[:, [1, 2]], np.where((ax == 1)[:, None], Vr[:, [0, 2]], Vr[:, [0, 1]]))
            me.usertexcoord = Tr.flatten().tolist()
            me.userface = z["F:" + key].astype(np.int32).flatten().tolist()
            try:
                me.maxhullvert = 8
            except Exception:  # noqa: BLE001
                pass
            _shell(me)
            rgb, rough, metal, emis = looks[look]
            for lv in ROBOT_LEVELS:
                ng = body.add_geom()
                ng.name = "%s@%s" % (me.name, lv)
                ng.type = mujoco.mjtGeom.mjGEOM_MESH
                ng.meshname = me.name
                if look in ROBOT_SETS:
                    ng.material = bank.textured(ROBOT_SETS[look][0], ROBOT_SETS[look][1], lv, 1.0, True)
                else:
                    ng.material = bank.plain((rgb[0], rgb[1], rgb[2], 1.0), rough, metal, lv if emis <= 0 else "ext", 0.0, emis, True)
                ng.contype, ng.conaffinity, ng.group, ng.mass = 0, 0, 1, 0.0
                bank.robot_copies[lv].append(ng.name)
                added += 1
    return removed, added


def batch_robot(spec, bank):
    """Merge, on every moving body of the robot, the geoms of one light-level version that share a material and a
    shading (flat boxes, smooth primitives) into ONE mesh in the body's frame, and delete the merged geoms.

    Why: a track belt is a loop of box facets whose side faces overlap in one plane, and the chassis parts abut. Where
    two drawn objects have the same depth at a sample, which one shows is decided by the order Filament happens to
    draw them in, and that order is not a function of the scene: the same frame came out with a few tens of different
    pixels (the belts' edges) in two fresh processes. Inside one mesh the triangles are drawn in the mesh's own order,
    so one object per body, version and material makes the frame a function of the scene. The render twin does
    kinematics only: the merged bodies get an explicit inertia in place of their geoms' masses."""
    meshes = {me.name: me for me in spec.meshes}
    level_of = {}
    for lv, names in bank.robot_copies.items():
        for n in names:
            level_of[n] = lv
    groups = {}
    for body in spec.bodies:
        for g in list(body.geoms):
            lv = level_of.get(g.name)
            if lv is None:
                continue
            got = _geom_mesh(meshes, g)
            if got is None:
                continue
            smooth = bool(meshes[g.meshname].smoothnormal)
            groups.setdefault((body.name, lv, g.material, smooth, int(g.group)), (body, []))[1].append((g, got))
    n_merged = n_new = 0
    for k, ((bname, lv, mat, smooth, group), (body, items)) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
        if len(items) < 2:
            continue
        Vs, Ts, Fs, n = [], [], [], 0
        for g, (V, T, F) in items:
            R = _quat_mat(np.asarray(g.quat, float))
            Vs.append(V @ R.T + np.asarray(g.pos, float))
            Ts.append(T)
            Fs.append(F + n)
            n += len(V)
        me = spec.add_mesh()
        me.name = "lkrobot_%d" % k
        me.uservert = np.concatenate(Vs).flatten().tolist()
        me.usertexcoord = np.concatenate(Ts).flatten().tolist()
        me.userface = np.concatenate(Fs).flatten().tolist()
        me.smoothnormal = smooth
        try:
            me.maxhullvert = 8
        except Exception:  # noqa: BLE001
            pass
        _shell(me)
        names = bank.robot_copies[lv]
        for g, _ in items:
            names.remove(g.name)
            spec.delete(g)
        body.explicitinertial = True
        body.mass = 1.0
        body.inertia = [0.01, 0.01, 0.01]
        ng = body.add_geom()
        ng.name = me.name
        ng.type = mujoco.mjtGeom.mjGEOM_MESH
        ng.meshname = me.name
        ng.material = mat
        ng.contype, ng.conaffinity, ng.group, ng.mass = 0, 0, group, 0.0
        names.append(ng.name)
        n_merged += len(items)
        n_new += 1
    return n_merged, n_new


def batch_statics(spec, bodies):
    """Merge every geom of the static `bodies` (world, decor) that is drawn as a textured mesh into one mesh per
    material, in world coordinates, and delete the merged geoms (a CPU rasteriser pays per drawn object; the render
    twin's geom ids mean nothing, its qpos layout is unchanged)."""
    meshes = {me.name: me for me in spec.meshes}
    batches = {}
    gone = []
    for body in bodies:
        for g in list(body.geoms):
            if not g.material or (g.name or "").startswith("dcbatch"):
                pass
            if "_body_" in (g.name or "") or (g.name or "").startswith(("veh_", "prop_")):   # posed bodies and vehicles keep their own
                continue                                                           # smooth-shaded meshes
            got = _geom_mesh(meshes, g)
            if got is None:
                continue
            V, T, F = got
            R = _quat_mat(np.asarray(g.quat, float))
            Vw = V @ R.T + np.asarray(g.pos, float)
            b = batches.setdefault(g.material, [[], [], [], 0])
            b[0].append(Vw)
            b[1].append(T)
            b[2].append(F + b[3])
            b[3] += len(V)
            gone.append(g)
    for g in gone:
        spec.delete(g)
    world = bodies[-1]
    for k, (mat, (Vs, Ts, Fs, n)) in enumerate(sorted(batches.items())):
        me = spec.add_mesh()
        me.name = "lkbatch_%d" % k
        me.uservert = np.concatenate(Vs).flatten().tolist()
        me.usertexcoord = np.concatenate(Ts).flatten().tolist()
        me.userface = np.concatenate(Fs).flatten().tolist()
        me.smoothnormal = False
        try:
            me.maxhullvert = 8
        except Exception:  # noqa: BLE001
            pass
        _shell(me)
        g = world.add_geom()
        g.name = me.name
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = me.name
        g.material = mat
        g.contype, g.conaffinity, g.group, g.mass = 0, 0, 2, 0.0
    return len(gone), len(batches)
