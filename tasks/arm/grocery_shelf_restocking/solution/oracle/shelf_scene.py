"""Shelf restocking: what the cameras draw (plain data: meshes, materials, frames; no renderer here).

One scene, two renderers: the meshes are the approved look's own builders (shelf_parts, shelf_geom: the gondola bay
with its ticket rail, the cart, the tray tote, the products with their SKU's print and the code label, the Panda's
visual meshes with soft finger pads, the telescopic lift; the store aisle round the bay: the gondola runs on both
sides, back to back and across the aisle, end caps, every bay stocked with dressing SKUs, floor, ceiling with lamp
housings, walls, the stockroom door, the aisle marker; the stock-handling props of the render bundle).

`items(unit, episode, model, meta, bundle)` returns the list of drawn items, each a dict:
  name, frame ("world" | "bay" | ("body", id) | ("lift", k)), mesh (V, N, UV, F as float32 / uint32, uv already scaled),
  mat (rgb sRGB 0..1, roughness, metallic, emissive), tex (None | ("static", name) | ("sku", sku name) | ("code", n)
  | ("props", group, index)).
The store is built in the BAY's frame and stands at the episode's bay pose, with the bay's own width, depth and
height: the run of shelving continues the task's bay exactly, whatever the unit.
"""
import math

import numpy as np

import shelf_geom as G
import shelf_looks as LK
import shelf_parts as P

AISLE = 2.30          # m, between the two runs' front planes
CEIL = 3.30
N_SIDE = 3            # bays on each side of the task's bay
STORE_SEED = 1581
STOCK_NSEG = 16
DRESS_MIN_DE = 20.0   # CIE76: a dressing SKU of a kind the unit carries is not stocked next to the unit's bay when its
                      # body AND its band are both nearer than this to one of the unit's SKUs (`dress_for`)
BOOTH_X, BOOTH_PITCH = 200.0, 2.0     # the catalog pictures' stands, far behind every camera's far plane
CATALOG_GROUND = (0.6, 0.6, 0.62)
PAD_Y_BACK, PAD_Y_FACE, PAD_Z, PAD_HX = 0.0080, -0.0005, (0.0362, 0.0539), 0.0088
T0 = PAD_Y_BACK - PAD_Y_FACE


def _srgb(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def material(lk, looks):
    """A look as the rasteriser's terms: (rgb sRGB, roughness, metallic, emissive, static texture name or None,
    metres per uv repeat or None for a map with its own uv)."""
    L = looks.get(lk) or looks["default"]
    if "glass" in L:                                   # one opaque pass: a pane is a dark mirror
        return (_srgb(0.06 * np.asarray(L["glass"]["color"], float)), 0.08, 0.0, 0.0, None, 0.1)
    rough = L.get("rough")
    if rough is None:
        rough = (L.get("rgh") or (None, 0.5, 1.0))[1]
        rough = 0.5 if rough is None else rough
    tex = L["col"][0] if "col" in L else None
    rgb = np.ones(3) if tex else _srgb(L.get("base", [0.5, 0.5, 0.5]))
    tile = None if L.get("uv") == "own" else float(L.get("tile", 0.1))
    return (rgb, float(rough), float(L.get("metal", 0.0)), 0.0, tex, tile)


def lab(c):
    """CIELAB (D65) of an sRGB colour"""
    c = np.asarray(c, float)[:3]
    l_ = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    x = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ l_ / np.array([0.9505, 1.0, 1.089])
    f = np.where(x > 0.008856, np.cbrt(x), 7.787 * x + 16.0 / 116.0)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


def dress_for(unit, dress):
    """The dressing SKUs stocked round this unit's bay: a dressing SKU of a kind the unit carries (box or cylinder)
    whose body and band are both within DRESS_MIN_DE (CIE76) of one of the unit's SKUs is left out, so no product
    of the neighbouring bays can be taken for one of the catalogue. The order is the bundle's."""
    out = {}
    for name, d in dress.items():
        close = False
        for t in unit["skus"].values():
            if t["kind"] != d["kind"]:
                continue
            if (np.linalg.norm(lab(t["rgba"]) - lab(d["rgba"])) < DRESS_MIN_DE
                    and np.linalg.norm(lab(t["look"]["band"]) - lab(d["look"]["band"])) < DRESS_MIN_DE):
                close = True
        if not close:
            out[name] = d
    return out


def booths(col, unit, lays):
    """The catalog pictures' subjects: each SKU's unit alone on a grey ground (no code label), on a stand of its
    own; `booth_camera` gives the picture's camera."""
    for i, (name, sku) in enumerate(unit["skus"].items()):
        hh = sku["size"][2] if sku["kind"] == "box" else sku["size"][1]
        for _n, mesh, lk, opt in P.unit_parts(sku, lays[name]):
            if lk is None:
                col.add(("booth", i), "sku_" + name, mesh, {"uv": "own"}, tex=("sku", name), t=[0, 0, hh], R=np.eye(3))
            else:
                col.add(("booth", i), {"cap": "cap_" + name}.get(lk, lk), mesh, opt, t=[0, 0, hh], R=np.eye(3))
        col.add(("booth", i), "catalog_ground", _quad([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [0, 0, 1.0]))


def booth_origin(i):
    return np.array([BOOTH_X + BOOTH_PITCH * i, 0.0, 0.0])


def booth_camera(i, sku):
    """the catalog picture's camera, as before: 0.45 m in front of the unit's printed front, 2 cm above its middle,
    a 35 degree field of view; (pos, forward, up, [centre, half width, bottom, top, near])"""
    hh = sku["size"][2] if sku["kind"] == "box" else sku["size"][1]
    pos = booth_origin(i) + np.array([-0.45, 0.0, hh + 0.02])
    fwd = booth_origin(i) + np.array([0.0, 0.0, hh]) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    near = 0.05
    top = near * math.tan(math.radians(35.0) / 2)
    return pos, fwd, up, [0.0, top, -top, top, near]


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _quad(p, n, uvs=None):
    m = G.Mesh()
    m.quad(p, n, uvs or [(q[0], q[1]) for q in p])
    return m.arrays()


def quat_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def model_soup(m, g):
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


def _clip(poly, axis, val, keep_ge):
    out = []
    n = len(poly)
    for i in range(n):
        (p, pn), (q, qn) = poly[i], poly[(i + 1) % n]
        ip = (p[axis] >= val) == keep_ge
        iq = (q[axis] >= val) == keep_ge
        if ip:
            out.append((p, pn))
        if ip != iq:
            t = (val - p[axis]) / (q[axis] - p[axis])
            x = p + t * (q - p)
            x[axis] = val
            out.append((x, pn + t * (qn - pn)))
    return out


def finger_metal(V, N, sign, z_cut=0.0345):
    """the finger's visual mesh without its tip block (everything with z > z_cut and sign * y < PAD_Y_BACK cut away,
    the cut closed): the soft pad stands in the block's place"""
    V, N = V.copy(), N.copy()
    V[:, 1] *= sign
    N[:, 1] *= sign
    tris, cut = [], []
    for i in range(0, len(V), 3):
        poly = [(V[i + k], N[i + k]) for k in range(3)]
        low = _clip(poly, 2, z_cut, False)
        up = _clip(poly, 2, z_cut, True)
        if len(low) >= 3:
            tris.append(low)
        if len(up) >= 3:
            kept = _clip(up, 1, PAD_Y_BACK, True)
            if len(kept) >= 3:
                tris.append(kept)
                cut += [p[0] for p in kept if abs(p[0][1] - PAD_Y_BACK) < 1e-9]
    m = G.Mesh()
    for poly in tris:
        for k in range(1, len(poly) - 1):
            m.tri([poly[0][0], poly[k][0], poly[k + 1][0]], np.asarray([poly[0][1], poly[k][1], poly[k + 1][1]]), [(0, 0)] * 3)
    if cut:
        cp = np.asarray(cut)[:, [0, 2]]
        lo, hi = cp.min(0), cp.max(0)
        q = [np.array([lo[0], PAD_Y_BACK, lo[1]]), np.array([hi[0], PAD_Y_BACK, lo[1]]), np.array([hi[0], PAD_Y_BACK, hi[1]]),
             np.array([lo[0], PAD_Y_BACK, hi[1]])]
        m.quad(q, [0, -1.0, 0], [(0, 0)] * 4)
    Vm, Nm, UVm, Fm = m.arrays()
    Vm[:, 1] *= sign
    Nm[:, 1] *= sign
    return G.uv_box((Vm, Nm, UVm, Fm))


def pad_mesh():
    hz = (PAD_Z[1] - PAD_Z[0]) / 2
    o = G.rounded_rect(PAD_HX, hz, 0.0026, 6)
    return G.slab(o, 0.0, T0, 0.0012, 0.0, top=True, bottom=False)


class Collector(object):
    """Parts merged per frame, material and texture, in the order they first appear."""

    def __init__(self, looks):
        self.looks = looks
        self.groups = {}

    def add(self, frame, look, mesh, opt=None, tex=None, R=None, t=None, part=None):
        """part: a tag that keeps these meshes a renderable of their own (the renderer culls a renderable whose box is
        outside the camera's view: one per dressing bay)"""
        opt = opt or {}
        V, N, UV, F = [np.asarray(x) for x in mesh]
        if len(F) == 0:
            return
        rgb, rough, metal, emis, stex, tile = material(look, self.looks)
        V, N, UV = V.astype(np.float64), N.astype(np.float64), UV.astype(np.float64)
        if tex is None and stex is not None:
            tex = ("static", stex)
        if not (opt.get("uv") == "own" or tile is None):
            UV = UV / tile
        if R is not None:
            V, N = V @ np.asarray(R).T + np.asarray(t, float), N @ np.asarray(R).T
        F = F.reshape(-1, 3).astype(np.int64)
        if opt.get("double"):                          # the rasteriser culls back faces: both sides as geometry
            V, N, UV, F = np.concatenate([V, V]), np.concatenate([N, -N]), np.concatenate([UV, UV]), np.concatenate([F, F[:, ::-1] + len(V)])
        key = (frame, part, look if tex is None or tex[0] == "static" else None, tex)
        g = self.groups.setdefault(key, {"frame": frame, "look": look, "tex": tex, "mat": (rgb, rough, metal, emis), "V": [], "N": [], "U": [],
                                         "F": [], "nv": 0})
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


def consts(S, W):
    return {"BOARD_T": S.BOARD_T, "LIP_T": S.LIP_T, "LIP_H": S.LIP_H, "UPRIGHT_T": S.UPRIGHT_T, "BACK_T": S.BACK_T,
            "CART": W.CART, "TOTE": W.TOTE, "CODE": W.code_sizes()}


def store(col, bay, C, dress, lays):
    """The aisle round the task's bay, in the bay's frame (the approved look's store158.py, the same draws in the
    same order)."""
    rng = np.random.default_rng(STORE_SEED)
    LT, UT, KT = C["LIP_T"], C["UPRIGHT_T"], C["BACK_T"]
    width, depth, top_z = bay["width"], bay["depth"], bay["top_z"]
    xb = LT + depth + KT
    pitch = width + 2 * UT
    y_end = (N_SIDE + 0.5) * pitch

    def boards_draw():
        n = int(rng.integers(4, 6))
        z = float(rng.uniform(0.15, 0.20))
        out = []
        for _ in range(n):
            out.append(round(z, 3))
            z += float(rng.uniform(0.27, 0.40))
        return [b for b in out if b < top_z - 0.24]

    nbay = [0]

    def put_bay(R, t, stock=1.0, tickets=True):
        nbay[0] += 1
        part = nbay[0]
        boards = boards_draw()
        for _name, mesh, lk, opt in P.gondola(width, depth, boards, top_z, C, tickets=tickets):
            col.add("bay", lk, mesh, opt, R=R, t=t, part=part)
        if stock > 0:
            st = P.gondola_stock(None, dress, lays, rng, width, depth, boards, top_z, C, fill=(0.45 * stock, 1.0 * stock), nseg=STOCK_NSEG)
            for (n, _part, lk), mesh in st.items():
                if lk is None:
                    col.add("bay", "sku_" + n, mesh, {"uv": "own"}, tex=("static", "sku_" + n), R=R, t=t, part=part)
                else:
                    col.add("bay", {"cap": "cap_" + n}.get(lk, lk), mesh, {}, R=R, t=t, part=part)

    for k in range(-N_SIDE, N_SIDE + 1):
        if k != 0:
            put_bay(np.eye(3), [0, k * pitch, 0])
        put_bay(_rz(math.pi), [2 * xb + 0.004, k * pitch, 0], stock=0.8)
    for k in range(-N_SIDE, N_SIDE + 1):
        put_bay(_rz(math.pi), [-AISLE, k * pitch, 0])
    for sy in (-1, 1):
        put_bay(_rz(-sy * math.pi / 2), [xb + 0.002, sy * (y_end + 0.004), 0])
        put_bay(_rz(-sy * math.pi / 2), [-AISLE - xb - 0.002, sy * (y_end + 0.004), 0])
    X0, X1 = -AISLE - 3.4, 3.6
    Y0, Y1 = -y_end - 2.6, y_end + 4.2
    col.add("bay", "floor", _quad([(X0, Y0, 0), (X1, Y0, 0), (X1, Y1, 0), (X0, Y1, 0)], [0, 0, 1.0]))
    col.add("bay", "ceiling", _quad([(X0, Y1, CEIL), (X1, Y1, CEIL), (X1, Y0, CEIL), (X0, Y0, CEIL)], [0, 0, -1.0]))

    def wall(p0, p1, n):
        (x0, y0), (x1, y1) = p0, p1
        le = math.hypot(x1 - x0, y1 - y0)
        col.add("bay", "wall", _quad([(x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, CEIL), (x0, y0, CEIL)], n, [(0, 0), (le, 0), (le, CEIL), (0, CEIL)]))
        e = np.asarray(n, float) * 0.004
        col.add("bay", "wall_band", _quad([(x0 + e[0], y0 + e[1], 2.35), (x1 + e[0], y1 + e[1], 2.35), (x1 + e[0], y1 + e[1], 2.75),
                                           (x0 + e[0], y0 + e[1], 2.75)], n, [(0, 0), (le, 0), (le, 0.4), (0, 0.4)]))
        col.add("bay", "gondola_kick", _quad([(x0 + e[0], y0 + e[1], 0.0), (x1 + e[0], y1 + e[1], 0.0), (x1 + e[0], y1 + e[1], 0.12),
                                              (x0 + e[0], y0 + e[1], 0.12)], n, [(0, 0), (le, 0), (le, 0.12), (0, 0.12)]))
    wall((X0, Y0), (X1, Y0), [0, 1.0, 0])
    wall((X1, Y1), (X0, Y1), [0, -1.0, 0])
    wall((X1, Y0), (X1, Y1), [-1.0, 0, 0])
    wall((X0, Y1), (X0, Y0), [1.0, 0, 0])
    xc = -AISLE / 2
    yd = Y0 + 0.004
    I3 = np.eye(3)
    for sg in (-1, 1):
        x0, x1 = (xc - 0.90, xc - 0.006) if sg < 0 else (xc + 0.006, xc + 0.90)
        col.add("bay", "door_steel", G.transform(G.bevel_box([(x1 - x0) / 2, 0.02, 1.05], 0.003), I3, [(x0 + x1) / 2, yd + 0.03, 1.06]))
        cxw = (x0 + x1) / 2 - sg * 0.12
        col.add("bay", "rubber_black", G.cylinder((cxw, yd + 0.049, 1.50), (cxw, yd + 0.053, 1.50), 0.185, 40))
        col.add("bay", "door_glass", G.cylinder((cxw, yd + 0.053, 1.50), (cxw, yd + 0.055, 1.50), 0.160, 40))
        col.add("bay", "rubber_black", G.transform(G.bevel_box([(x1 - x0) / 2, 0.006, 0.14], 0.002), I3, [(x0 + x1) / 2, yd + 0.055, 0.17]))
    for x in (xc - 0.95, xc + 0.95):
        col.add("bay", "gondola_kick", G.transform(G.bevel_box([0.05, 0.035, 1.08], 0.003), I3, [x, yd + 0.03, 1.08]))
    col.add("bay", "gondola_kick", G.transform(G.bevel_box([1.0, 0.035, 0.05], 0.003), I3, [xc, yd + 0.03, 2.16]))
    for xa in (-AISLE / 2, 2 * xb + AISLE / 2 + 0.1, -AISLE - 2 * xb - AISLE / 2 + 0.3):
        y = Y0 + 1.3
        while y < Y1 - 0.8:
            col.add("bay", "fixture", G.transform(G.bevel_box([0.11, 0.78, 0.022], 0.004), I3, [xa, y, CEIL - 0.022]))
            y += 2.4
    ys = 1.9 * pitch
    for sg in (-1, 1):
        n = [0, sg, 0]
        x0, x1 = (xc - 0.6, xc + 0.6) if sg < 0 else (xc + 0.6, xc - 0.6)
        col.add("bay", "sign", _quad([(x1, ys + sg * 0.012, 2.42), (x0, ys + sg * 0.012, 2.42), (x0, ys + sg * 0.012, 2.78), (x1, ys + sg * 0.012, 2.78)],
                                     n, [(0, 0), (1, 0), (1, 1), (0, 1)]), {"uv": "own"})
    col.add("bay", "sign_edge", G.transform(G.bevel_box([0.602, 0.0115, 0.182], 0.002), I3, [xc, ys, 2.60]))
    col.add("bay", "wire", G.merge(*[G.cylinder((xc + sx, ys, 2.78), (xc + sx, ys, CEIL), 0.0012, 6) for sx in (-0.5, 0.5)]))


def cell(col, unit, model, meta, bay, C, lays):
    """The task's own things (the approved look's cell158.py): the cart, the tote, the lift's plate (world frame),
    the bay as a gondola bay (bay frame), every product with its SKU's print and its code label, the Panda's visual
    meshes with the fingers' soft pads, the robot's plate on the carriage, the telescopic column's stages."""
    import mujoco
    m = model
    for _name, mesh, lk, opt in P.cart(C["CART"], C["TOTE"]) + P.tote(C["CART"], C["TOTE"]) + P.lift_static():
        col.add("world", lk, mesh, opt)
    for _name, mesh, lk, opt in P.gondola(bay["width"], bay["depth"], bay["boards"], bay["top_z"], C):
        col.add("bay", lk, mesh, opt)
    skus = unit["skus"]
    for n in meta:
        b = m.body(n).id
        sku_name = meta[n]["sku"]
        sku = skus[sku_name]
        for _name, mesh, lk, opt in P.unit_parts(sku, lays[sku_name]):
            if lk is None:
                col.add(("body", b), "sku_" + sku_name, mesh, {"uv": "own"}, tex=("sku", sku_name))
            else:
                col.add(("body", b), {"cap": "cap_" + sku_name}.get(lk, lk), mesh, opt)
        col.add(("body", b), "code", P.code_decal(sku, meta[n]["code_at"], C["CODE"]), {"uv": "own"}, tex=("code", int(meta[n]["code"])))
    pads = []
    for g in range(m.ngeom):
        bid = int(m.geom_bodyid[g])
        b = m.body(bid).name
        if not b.startswith("r_") or int(m.geom_group[g]) != 1 or int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        name = m.geom(g).name or ""
        if name.endswith("_g_finger1_visual") or name.endswith("_g_finger2_visual"):
            sg = 1.0 if name.endswith("finger1_visual") else -1.0
            V, N = model_soup(m, g)
            col.add(("body", bid), "finger", finger_metal(V, N, sg))
            pads.append((bid, sg))
        else:
            col.add(("body", bid), panda_look(m, g), model_mesh(m, g))
    pm = pad_mesh()
    zc = 0.5 * (PAD_Z[0] + PAD_Z[1])
    for bid, sg in pads:
        R = np.array([[sg, 0, 0], [0, 0, 1.0], [0, -sg, 0]]).T
        col.add(("body", bid), "finger_pad", pm, R=R, t=[0, sg * PAD_Y_BACK, zc])
    cb = m.body("carriage").id
    for _name, mesh, lk, opt in P.carriage_parts(False):
        col.add(("body", cb), lk, mesh, opt)
    for k in range(len(P.LIFT_STAGES)):
        for _name, mesh, lk, opt in P.lift_stage(k):
            col.add(("lift", k), lk, mesh, opt)


def looks_of(unit, dress):
    looks = dict(LK.LOOKS)
    for name, sku in list(unit["skus"].items()) + list(dress.items()):
        box = sku["kind"] == "box"
        looks["sku_" + name] = {"base": [1, 1, 1], "rough": 0.36 if box else 0.28, "uv": "own"}
        looks["cap_" + name] = {"base": [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in sku["look"]["cap"]], "rough": 0.38}
    looks["code"] = {"base": [1, 1, 1], "rough": 0.62, "uv": "own"}
    looks["catalog_ground"] = {"base": [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in CATALOG_GROUND], "rough": 0.9}
    return looks


def lift_tops(lift):
    """the telescopic column's stage tops for the carriage at height `lift`"""
    n = len(P.LIFT_STAGES)
    return [lift * (k + 1) / n - (0.0 if k == n - 1 else 0.004) - (0.018 if k == n - 1 else 0.0) for k in range(n)]
