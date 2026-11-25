"""Cabinet door hanging: what the cameras draw (plain data: meshes, materials, frames; no renderer here).

One scene, two renderers: these are the demonstration video's own meshes. Every part of the simulated model is drawn
at the model's own geometry, read from the episode's model (so a unit's hinge heights, knuckle count and bore, and an
episode's cabinet pose, racks and stands are the episode's): the cabinet's panels, the two doors and their pulls, each
knuckle and each stand as the turned sleeve its ring of boxes stands for (with its bore), the pins with their grip
posts, the racks, the vacuum tool, both Pandas' visual meshes. Added to them (door_parts.py): hinge leaves with
screws, the detent groove under a class pin's head, the vacuum tool's bellows, fitting, hose and clips, the gripper's
soft pads, the bench as a worktop on a cabinet body in the table block's place, the fitter's things on it, two felt
pieces, a tool bag on the floor, the rail the cabinet hangs on. The room round the bench is the render bundle's.

`parts(unit, episode, model)`: [(body name or "world", part name, mesh (V, N, UV, F) in that body's frame, look,
options)], in a fixed order. `items(unit, episode, model)`: the same merged per frame and material for the
renderer, each a dict: frame ("world" | ("body", id)), look, mesh (float32 V, N, UV already divided by the look's
tile, uint32 F), mat (rgb sRGB 0..1, roughness, metallic, emissive), tex (None | ("static", name)).
"""
import math

import numpy as np

import door_geom as G
import door_looks as LK
import door_parts as P

TOP_T = 0.040                     # the worktop's thickness
PROUD = 0.0004                    # m: a robot's dark parts stand this far off the shell they lie on


def _srgb(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def texture_name(look):
    """the render bundle's name of a look's colour map (the map tinted to the look's mean), or None"""
    L = LK.LOOKS.get(look) or LK.LOOKS["default"]
    return ("%s__%s" % (L["col"][0], look)) if "col" in L else None


def material(look):
    """A look as the rasteriser's terms: (rgb sRGB, roughness, metallic, emissive, static texture name or None,
    metres per uv repeat or None for a map with its own uv)."""
    L = LK.LOOKS.get(look) or LK.LOOKS["default"]
    rough = L.get("rough")
    if rough is None:
        rough = (L.get("rgh") or (None, 0.5, 1.0))[1]
        rough = 0.5 if rough is None else rough
    tex = texture_name(look)
    rgb = np.ones(3) if tex else _srgb(L.get("base", [0.5, 0.5, 0.5]))
    tile = None if L.get("uv") == "own" else float(L.get("tile", 0.1))
    return (rgb, float(rough), float(L.get("metal", 0.0)), 0.0, tex, tile)


def quat_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def model_soup(m, g):
    """a mesh geom as a triangle soup with its normals, in its body's frame"""
    mid = int(m.geom_dataid[g])
    va = int(m.mesh_vertadr[mid])
    fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    Vm = m.mesh_vert[va:va + int(m.mesh_vertnum[mid])].astype(float)
    Fm = m.mesh_face[fa:fa + fn].astype(int)
    na = int(m.mesh_normaladr[mid])
    Nm = m.mesh_normal[na:na + int(m.mesh_normalnum[mid])].astype(float)
    FN = m.mesh_facenormal[fa:fa + fn].astype(int)
    R = quat_mat(m.geom_quat[g])
    return Vm[Fm.reshape(-1)] @ R.T + m.geom_pos[g], Nm[FN.reshape(-1)] @ R.T


def model_mesh(m, g):
    V, N = model_soup(m, g)
    return G.uv_box((V, N, np.zeros((len(V), 2)), np.arange(len(V)).reshape(-1, 3)))


def lift(mesh, d):
    """the mesh moved out by d along its surface: every vertex along the mean normal of all the vertices at its place
    (the faces carry their own vertices; moved along their own normals they would part at the edges)"""
    V, N, UV, F = mesh
    _u, inv = np.unique(np.round(V / 1e-6).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    S_ = np.zeros((int(inv.max()) + 1, 3))
    np.add.at(S_, inv, N)
    S_ /= np.linalg.norm(S_, axis=1, keepdims=True) + 1e-12
    return V + d * S_[inv], N, UV, F


def panda_look(m, g):
    rgba = m.mat_rgba[int(m.geom_matid[g])] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    if min(rgba[:3]) > 0.85:
        return "panda_white"
    if max(rgba[:3]) < 0.15:
        return "panda_black"
    if max(rgba[:3]) < 0.35 and max(rgba[:3]) - min(rgba[:3]) < 0.05:
        return "panda_grey"
    if rgba[0] > 0.9 and rgba[1] < 0.1:
        return "panda_led_red"
    if rgba[1] > 0.9 and rgba[0] < 0.1:
        return "panda_led_green"
    if rgba[2] > 0.7 and rgba[0] < 0.1:
        return "panda_blue"
    return "panda_white"


def ring_key(body, name):
    """the ring a box segment belongs to (a knuckle of the cabinet or of a door, a stand), or None"""
    if "_s" not in name:
        return None
    stem, seg = name.rsplit("_s", 1)
    if not seg.isdigit():
        return None
    if body == "carcass" and stem.startswith("ck"):
        return stem
    if body in ("door", "old") and stem.startswith(body + "_k"):
        return stem
    if body == "world" and stem.startswith("stand"):
        return stem
    return None


def parts(unit, episode, m):
    """Everything drawn, per body, in one order (see the module's text)."""
    import mujoco

    import door_sim as S
    BOX, CYL, CAP, MESH = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           mujoco.mjtGeom.mjGEOM_MESH)
    out = []

    def put(body, name, mesh, look, opt=None):
        out.append((body, name, mesh, look, opt or {}))

    def prim(g, bevel):
        t, s = int(m.geom_type[g]), m.geom_size[g]
        R, p = quat_mat(m.geom_quat[g]), m.geom_pos[g]
        if t == BOX:
            return G.transform(G.bevel_box(s[:3], bevel), R, p)
        h = s[1] + (s[0] * 0.6 if t == CAP else 0.0)
        return P.smooth_cyl(p - R[:, 2] * h, p + R[:, 2] * h, s[0], nseg=48, bevel=min(bevel, 0.45 * s[0]))

    n_stand = len(S.STANDS)

    def rule(body, name):
        """(look, bevel) of a primitive geom of the model, or None when it is not drawn as it is"""
        if name in ("floor", "wall", "table") or name.startswith("x_"):
            return None                                 # the room's, the bench's, or a contact shape of a drawn part
        if name.startswith(("rack_rail", "park_rail")):
            return "rack_steel", 0.0012
        if name.startswith("cleaf"):
            return "knuckle_steel", 0.0004
        if name.startswith("c_"):
            return "carcass", 0.0012
        if "_kleaf" in name:
            return ("knuckle_brass" if body == "door" else "knuckle_dark"), 0.0004
        if name.endswith("_leaf"):
            return ("door_new" if body == "door" else "door_old"), 0.0022
        if name.endswith("_handle"):
            return "handle", 0.002
        if body.startswith("pin"):
            return ("pin_steel" if int(body[3:]) < n_stand else "pin_old"), 0.0006
        if name == "r_g_cup_pad":
            return "cup_rubber", 0.0015
        if name.startswith("r_g_cup"):
            return "cup_alu", 0.0015
        return "default", 0.001

    rings, ring_order = {}, []
    table = None
    for g in range(m.ngeom):
        body = m.body(int(m.geom_bodyid[g])).name
        name = m.geom(g).name or ""
        rgba = m.mat_rgba[int(m.geom_matid[g])] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        t = int(m.geom_type[g])
        if name == "table":
            table = g
        if body.startswith(("l_", "r_")) and t == MESH:
            if int(m.geom_group[g]) != 1:
                continue                                # the collision hulls
            if name.endswith(("_g_finger1_visual", "_g_finger2_visual")):
                sg = 1.0 if name.endswith("finger1_visual") else -1.0
                V, N = model_soup(m, g)
                put(body, "finger_metal", P.finger_metal(V, N, sg), "finger")
                put(body, "finger_pad", P.finger_pad(sg), "finger_pad")
            else:
                look = panda_look(m, g)
                mesh = model_mesh(m, g)
                if look != "panda_white":               # a band or a cap that lies ON the white shell: proud of it, so no
                    mesh = lift(mesh, PROUD)            # pixel has two surfaces at one depth
                put(body, "panda_%d" % g, mesh, look)
            continue
        if t not in (BOX, CYL, CAP) or rgba[3] < 0.5:
            continue
        if body.startswith(("l_", "r_")) and not name.startswith("r_g_cup"):
            continue                                    # the robots' primitive collision shapes
        key = ring_key(body, name)
        if key is not None:
            if (body, key) not in rings:
                rings[(body, key)] = []
                ring_order.append((body, key))
            rings[(body, key)].append(g)
            continue
        r = rule(body, name)
        if r is not None:
            put(body, name, prim(g, r[1]), r[0])
    for body, key in ring_order:                        # each ring of boxes as the turned sleeve it stands for
        gs = rings[(body, key)]
        c = np.array([m.geom_pos[g] for g in gs]).mean(0)
        rc = float(np.mean([np.hypot(*(m.geom_pos[g][:2] - c[:2])) for g in gs]))
        ht, hz = float(m.geom_size[gs[0]][0]), float(m.geom_size[gs[0]][2])
        look = "stand_steel" if body == "world" else {"carcass": "knuckle_steel", "door": "knuckle_brass", "old": "knuckle_dark"}[body]
        put(body, key, P.sleeve([c[0], c[1], c[2] - hz], [c[0], c[1], c[2] + hz], rc + ht, rc - ht, nseg=64,
                                bevel=0.0008 if body != "world" else 0.0006), look)

    # the bench: a worktop on a cabinet body in the table block's footprint
    zt = float(S.TABLE_Z)
    cx, hx, hy = float(m.geom_pos[table][0]), float(m.geom_size[table][0]), float(m.geom_size[table][1])
    V, N, UV, F = G.xform2(G.slab(G.rounded_rect(hx, hy, 0.004, 5), zt - TOP_T, zt, 0.003, 0.002), 0, (cx, 0, 0))
    ax = np.abs(N).argmax(1)
    u = (V[:, 1] + hy) / (2 * hy)
    v = 1.0 - (V[:, 0] - cx + hx) / (2 * hx)
    UV = np.stack([np.where(ax == 1, 0.31 + (V[:, 2] - zt) / (2 * hy), u), np.where(ax == 0, 0.5 + (V[:, 2] - zt) / (2 * hx), v)], 1)
    put("world", "bench_top", (V, N, UV, F), "bench_top", {"uv": "own"})
    ins, ph = 0.02, 0.10
    put("world", "bench_body", P.box([cx - hx + ins, -hy + ins, ph], [cx + hx - ins, hy - ins, zt - TOP_T - 0.003], 0.0015), "bench_body")
    put("world", "bench_plinth", P.box([cx - hx + 0.06, -hy + 0.06, 0.0], [cx + hx - 0.06, hy - 0.06, ph], 0.001), "bench_plinth")
    nd = 4
    wdo = (2 * (hy - ins) - 0.004 * (nd - 1)) / nd
    for i in range(nd):                                 # door fronts on the robots' side
        ya = -hy + ins + i * (wdo + 0.004)
        put("world", "bench_front%d" % i, P.box([cx - hx + ins - 0.019, ya, ph + 0.003], [cx - hx + ins, ya + wdo, zt - TOP_T - 0.006], 0.0018), "bench_body")
        put("world", "bench_pull%d" % i, P.smooth_cyl([cx - hx + ins - 0.045, ya + wdo / 2 - 0.07, zt - TOP_T - 0.06],
                                                     [cx - hx + ins - 0.045, ya + wdo / 2 + 0.07, zt - TOP_T - 0.06], 0.005, nseg=24, bevel=0.0008), "handle")

    # the hardware, by the unit's hinges
    for hi, hz in enumerate((unit["hb"], unit["ht"])):
        for n_, me, lk in P.hinge_leaves(hz, False):
            put("carcass", "h%d_%s" % (hi, n_), me, lk)
        for n_, me, lk in P.hinge_leaves(hz, True):
            put("door", "h%d_%s" % (hi, n_), me, lk)
            put("old", "h%d_%s" % (hi, n_), me, {"leaf_brass": "leaf_dark", "screw": "screw_dark"}.get(lk, lk))
    for k in range(n_stand):                            # the detent's groove under a class pin's head
        g = m.geom("pin%d_shaft" % k).id
        put("pin%d" % k, "groove", P.detent_groove(float(m.geom_size[g][0]), -0.0065), "groove")
    for n_, me, lk in P.cup_extras():
        put("r_g_right_gripper", n_, me, lk)
    # the fitter's things, the felt pieces, the tool bag, the wall rail
    for n_, me, lk in P.bench_things(zt):
        put("world", n_, me, lk)
    pr = [m.geom("park_rail%d" % k).id for k in (0, 1)]
    px = sorted(float(m.geom_pos[g][0]) for g in pr)
    py, phy = float(m.geom_pos[pr[0]][1]), float(m.geom_size[pr[0]][1])
    for n_, me, lk in P.park_felt(px[0] + 0.0045, px[1] - 0.0045, py - phy + 0.004, py + phy - 0.004, zt):
        put("world", n_, me, lk)
    for n_, me, lk in P.tool_bag():
        put("world", n_, G.xform2(me, S.BAG_AT[2], (S.BAG_AT[0], S.BAG_AT[1], 0.0)), lk)
    for n_, me, lk in P.wall_rail(S.DEPTH, S.DOOR_W + S.EDGE_GAP, S.DOOR_H + 0.02):
        put("carcass", n_, me, lk)
    return out


class Collector(object):
    """Parts merged per frame and material, in the order they first appear."""

    def __init__(self):
        self.groups = {}

    def add(self, frame, look, mesh, opt=None):
        opt = opt or {}
        V, N, UV, F = [np.asarray(x) for x in mesh]
        if len(F) == 0:
            return
        rgb, rough, metal, emis, stex, tile = material(look)
        V, N, UV = V.astype(np.float64), N.astype(np.float64), UV.astype(np.float64)
        tex = ("static", stex) if stex is not None else None
        if not (opt.get("uv") == "own" or tile is None):
            UV = UV / tile
        F = F.reshape(-1, 3).astype(np.int64)
        g = self.groups.setdefault((frame, look), {"frame": frame, "look": look, "tex": tex, "mat": (rgb, rough, metal, emis),
                                                   "V": [], "N": [], "U": [], "F": [], "nv": 0})
        g["V"].append(V)
        g["N"].append(N)
        g["U"].append(UV)
        g["F"].append(F + g["nv"])
        g["nv"] += len(V)

    def items(self):
        out = []
        for g in self.groups.values():
            N = np.concatenate(g["N"])
            N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
            out.append({"frame": g["frame"], "look": g["look"], "tex": g["tex"], "mat": g["mat"],
                        "mesh": (np.ascontiguousarray(np.concatenate(g["V"]), np.float32), np.ascontiguousarray(N, np.float32),
                                 np.ascontiguousarray(np.concatenate(g["U"]), np.float32),
                                 np.ascontiguousarray(np.concatenate(g["F"]).reshape(-1), np.uint32))})
        return out


def items(unit, episode, m):
    col = Collector()
    for body, _name, mesh, look, opt in parts(unit, episode, m):
        col.add("world" if body == "world" else ("body", int(m.body(body).id)), look, mesh, opt)
    return col.items()
