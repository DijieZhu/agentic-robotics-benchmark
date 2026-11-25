"""Harness cell: what the cameras draw (plain data: meshes, materials, frames; no renderer here).

One scene, two renderers: these are the demonstration video's own meshes. Every part of the simulated model is drawn at
the model's own geometry, read from the episode and its model: the bench, the form board with the episode's printed
drawing (the NOMINAL places: `board_print`), the connector with its bracket and strain-relief boot, each clip (two pins
with a rounded top, two lead-in fins, two retaining tabs that turn with their joints), each post with its cap, the
cable (a tube piece and a ball per segment, turning with the segment), the bench's reels, bins and tape at their
contact shapes' places, the Panda's visual meshes and the cable jaws. The room round the bench is the render bundle's.

`parts(ep, model)`: [(body name or "world", part name, mesh (V, N, UV, F) in that body's frame, look, options)], in a
fixed order. `items(ep, model)`: the same merged per frame and material for the renderer, each a dict: frame
("world" | ("body", id)), look, mesh (float32 V, N, UV already divided by the look's tile, uint32 F), mat (rgb sRGB
0..1, roughness, metallic, emissive), tex (None | ("static", name) | ("image", key)), image (uint8 rows bottom-up for
an "image" texture).
"""
import math

import numpy as np

import harness_env as H
import harness_geom as G
import harness_looks as LK
import harness_parts as P

TOP_T = 0.040                     # the worktop's thickness
PROUD = 0.0004                    # m: a robot's dark parts stand this far off the shell they lie on
PRINT_PX_PER_M = 2000.0           # the board's print: 0.5 mm a pixel
MESH = 7                          # mujoco.mjtGeom.mjGEOM_MESH
BOX, CYL, CAPS = 6, 5, 3


def _srgb(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def texture_name(look):
    L = LK.LOOKS.get(look) or LK.LOOKS["default"]
    return L["col"][0] if "col" in L else None


def material(look):
    """A look as the rasteriser's terms: (rgb sRGB, roughness, metallic, emissive, static texture name or None,
    metres per uv repeat or None for a map with its own uv)."""
    L = LK.LOOKS.get(look) or LK.LOOKS["default"]
    rough = L.get("rough")
    if rough is None:
        rough = (L.get("rgh") or (None, 0.5, 1.0))[1]
        rough = 0.5 if rough is None else rough
    tex = texture_name(look)
    rgb = np.ones(3) if (tex or look == "board") else _srgb(L.get("base", [0.5, 0.5, 0.5]))
    tile = None if L.get("uv") == "own" else float(L.get("tile", 0.1))
    return (rgb, float(rough), float(L.get("metal", 0.0)), 0.0, tex, tile)


def quat_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotz(a):
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def rotx(a):
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


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
    """the mesh moved out by d along its surface (every vertex along the mean normal of all the vertices at its place)"""
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


# ------------------------------------------------------------------------------------------------------------
# the board's print: the harness drawing at the NOMINAL places (what the job sheet says), 0.5 mm a pixel
# ------------------------------------------------------------------------------------------------------------
def board_print(ep):
    """uint8 RGB image of the board's top face, rows bottom-up (row 0 = the face's v = 0: the edge at the lowest y),
    columns along +x. An off-white laminate with a faint 5 cm grid, the route as a blue line with arrow heads from the
    connector's printed place through the fixtures' printed places, a box at each clip turned to its printed axis, a
    ring at each post with a tick on the side the post has to be on, and the fixture's number beside it; then the
    marks of use (_worn)."""
    import cv2
    hx, hy = H.BOARD_HALF
    W, Hh = int(round(2 * hx * PRINT_PX_PER_M)), int(round(2 * hy * PRINT_PX_PER_M))
    img = np.empty((Hh, W, 3), np.uint8)
    img[:] = (232, 230, 222)
    step = int(round(0.05 * PRINT_PX_PER_M))
    for x in range(step // 2, W, step):
        cv2.line(img, (x, 0), (x, Hh - 1), (214, 213, 206), 1, cv2.LINE_AA)
    for y in range(step // 2, Hh, step):
        cv2.line(img, (0, y), (W - 1, y), (214, 213, 206), 1, cv2.LINE_AA)
    cv2.rectangle(img, (8, 8), (W - 9, Hh - 9), (120, 124, 132), 2, cv2.LINE_AA)

    def px(p):                                      # world (x, y) -> pixel (column, row), row 0 at the lowest y
        return (int(round((p[0] - (H.BOARD_C[0] - hx)) * PRINT_PX_PER_M)), int(round((p[1] - (H.BOARD_C[1] - hy)) * PRINT_PX_PER_M)))
    sheet = H.job_sheet(ep)
    ink, ink2 = (34, 52, 140), (30, 34, 44)
    route = [np.array(sheet["connector"]["pos"])]
    nom = ep.nominal
    for f in nom["fixtures"]:
        route.append(H.to_world(f["corner"] if f["type"] == "post" else f["pos"]))
    route.append(H.to_world(nom["route"][-1]))
    for a, b in zip(route[:-1], route[1:]):
        cv2.line(img, px(a), px(b), ink, 5, cv2.LINE_AA)
        mid, d = (a + b) / 2, (b - a) / (np.linalg.norm(b - a) + 1e-9)
        n = np.array([-d[1], d[0]])
        tri = np.array([px(mid + 0.012 * d), px(mid - 0.006 * d + 0.007 * n), px(mid - 0.006 * d - 0.007 * n)], np.int32)
        cv2.fillConvexPoly(img, tri, ink, cv2.LINE_AA)
    c = np.array(sheet["connector"]["pos"]); hd = sheet["connector"]["heading"]
    e, n = np.array([math.cos(hd), math.sin(hd)]), np.array([-math.sin(hd), math.cos(hd)])
    box = np.array([px(c - 0.030 * e + 0.014 * n), px(c + 0.004 * e + 0.014 * n), px(c + 0.004 * e - 0.014 * n), px(c - 0.030 * e - 0.014 * n)], np.int32)
    cv2.polylines(img, [box], True, ink2, 3, cv2.LINE_AA)
    for k, f in enumerate(sheet["fixtures"]):
        p = np.array(f["pos"])
        if f["type"] == "clip":
            ax = H.to_world_angle(nom["fixtures"][k]["axis"])        # the drawing's own; the job sheet does not carry it
            e, n = np.array([math.cos(ax), math.sin(ax)]), np.array([-math.sin(ax), math.cos(ax)])
            box = np.array([px(p - 0.012 * e + 0.018 * n), px(p + 0.012 * e + 0.018 * n), px(p + 0.012 * e - 0.018 * n), px(p - 0.012 * e - 0.018 * n)], np.int32)
            cv2.polylines(img, [box], True, ink2, 3, cv2.LINE_AA)
        else:
            cv2.circle(img, px(p), int(round(0.022 * PRINT_PX_PER_M)), ink2, 3, cv2.LINE_AA)
            corner = H.to_world(nom["fixtures"][k]["corner"])
            d = (corner - p) / (np.linalg.norm(corner - p) + 1e-9)
            cv2.line(img, px(p + 0.022 * d), px(p + 0.034 * d), ink2, 5, cv2.LINE_AA)       # the cable passes on this side
        # the number stands off the route, upright for a reader at the robot's place (the image is stored bottom-up)
        lab = np.zeros((40, 40, 3), np.uint8)
        cv2.putText(lab, str(k + 1), (6, 31), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        lab = np.rot90(lab[::-1], 1)                # glyph turned so that its up is the board's +x
        q = px(p + np.array([0.032, 0.030]))
        y0, x0 = q[1] - 20, q[0] - 20
        if 0 <= y0 and y0 + 40 <= Hh and 0 <= x0 and x0 + 40 <= W:
            m_ = lab[:, :, 0:1].astype(np.float32) / 255.0
            img[y0:y0 + 40, x0:x0 + 40] = (img[y0:y0 + 40, x0:x0 + 40] * (1 - m_) + np.array(ink2, np.float32) * m_).astype(np.uint8)
    return np.ascontiguousarray(_worn(img, ep))


def _worn(img, ep):
    """The print on a board that has been in use: soiling in broad patches, fine scratches, a few rubbed stretches
    where cables were dragged, and filled screw holes of earlier fixtures (pale plugs, none nearer than 4 cm to a
    printed fixture). Drawn from the board's seed; the ink keeps its contrast (the marks are faint and thin)."""
    import cv2
    rng = np.random.default_rng(np.random.SeedSequence([int(ep.seed), 7]))
    Hh, W = img.shape[:2]
    f = img.astype(np.float32)
    # soiling: low-frequency darkening and a warm cast, up to 6 %
    low = cv2.resize(rng.normal(0.0, 1.0, (Hh // 96 + 2, W // 96 + 2)).astype(np.float32), (W, Hh), interpolation=cv2.INTER_CUBIC)
    low = np.clip(0.5 + 0.35 * low, 0.0, 1.0)
    f *= (1.0 - 0.06 * low)[..., None] * np.array([1.0, 0.992, 0.97], np.float32)[None, None, :]
    marks = np.zeros((Hh, W), np.float32)
    for _ in range(int(rng.integers(45, 70))):                       # fine scratches, 2 to 14 cm, mostly along the board
        c = np.array([rng.uniform(0, W), rng.uniform(0, Hh)])
        ang = rng.normal(math.pi / 2, 0.5) if rng.uniform() < 0.7 else rng.uniform(0, math.pi)
        L = rng.uniform(0.02, 0.14) * PRINT_PX_PER_M
        d = 0.5 * L * np.array([math.cos(ang), math.sin(ang)])
        bend = rng.normal(0.0, 0.06) * L * np.array([-math.sin(ang), math.cos(ang)])
        pts = np.array([c - d, c + bend, c + d]).round().astype(np.int32)
        cv2.polylines(marks, [pts], False, float(rng.uniform(0.25, 0.8)), 1, cv2.LINE_AA)
    for _ in range(int(rng.integers(5, 9))):                         # rubbed stretches: wider, fainter
        c = np.array([rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * Hh])
        ang = rng.uniform(0, math.pi)
        d = rng.uniform(0.03, 0.08) * PRINT_PX_PER_M * np.array([math.cos(ang), math.sin(ang)])
        cv2.line(marks, tuple((c - d).round().astype(int).tolist()), tuple((c + d).round().astype(int).tolist()),
                 float(rng.uniform(0.10, 0.22)), int(rng.integers(8, 20)), cv2.LINE_AA)
    marks = cv2.GaussianBlur(marks, (0, 0), 0.6)
    f *= (1.0 - 0.16 * np.clip(marks, 0.0, 1.0))[..., None]
    hx, hy = H.BOARD_HALF
    fixed = [H.to_world(g["corner"] if g["type"] == "post" else g["pos"]) for g in ep.nominal["fixtures"]]
    fixed.append(H.to_world(ep.nominal["connector"]["pos"]))
    n_holes, tries = int(rng.integers(6, 11)), 0
    while n_holes > 0 and tries < 200:                               # filled holes of earlier fixtures: pairs 16 mm apart
        tries += 1
        c = np.array([H.BOARD_C[0] + rng.uniform(-hx + 0.04, hx - 0.04), H.BOARD_C[1] + rng.uniform(-hy + 0.04, hy - 0.04)])
        if min(float(np.linalg.norm(c - q)) for q in fixed) < 0.04:
            continue
        ang = rng.uniform(0, math.pi)
        for sgn in (-1.0, 1.0):
            q = c + sgn * 0.008 * np.array([math.cos(ang), math.sin(ang)])
            px_ = (int(round((q[0] - (H.BOARD_C[0] - hx)) * PRINT_PX_PER_M)), int(round((q[1] - (H.BOARD_C[1] - hy)) * PRINT_PX_PER_M)))
            cv2.circle(f, px_, 5, (203.0, 198.0, 186.0), -1, cv2.LINE_AA)
            cv2.circle(f, px_, 5, (176.0, 172.0, 162.0), 1, cv2.LINE_AA)
        n_holes -= 1
    return np.clip(f + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------------
# parts
# ------------------------------------------------------------------------------------------------------------
def parts(ep, m):
    out = []

    def put(body, name, mesh, look, opt=None):
        out.append((body, name, mesh, look, opt or {}))

    R = ep.radius
    # ---- the bench: worktop, frame, shelf
    x0, x1 = H.BENCH_X
    y0, y1 = H.BENCH_Y
    bx, by, cx, cy = (x1 - x0) / 2, (y1 - y0) / 2, (x0 + x1) / 2, (y0 + y1) / 2
    Vb, Nb, UVb, Fb = G.slab(G.rounded_rect(bx, by, 0.004, 5), -TOP_T, 0.0, 0.003, 0.002)
    axn = np.abs(Nb).argmax(1)
    UVb = np.stack([np.where(axn == 1, 0.31 + Vb[:, 2] / (2 * by), (Vb[:, 1] + by) / (2 * by)),
                    np.where(axn == 0, 0.5 + Vb[:, 2] / (2 * bx), 1.0 - (Vb[:, 0] + bx) / (2 * bx))], 1)
    put("world", "bench_top", (Vb + [cx, cy, 0.0], Nb, UVb, Fb), "bench_top", {"uv": "own"})
    zf, leg = -H.BENCH_H, 0.025
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = cx + sx * (bx - 0.06), cy + sy * (by - 0.06)
            put("world", "leg", P.box([lx - leg, ly - leg, zf + 0.02], [lx + leg, ly + leg, -TOP_T], 0.002), "bench_steel")
            put("world", "foot", P.smooth_cyl([lx, ly, zf], [lx, ly, zf + 0.02], 0.03, nseg=32, bevel=0.003), "foot")
    for sy in (-1, 1):
        ly = cy + sy * (by - 0.06)
        put("world", "rail_x", P.box([x0 + 0.085, ly - 0.02, -TOP_T - 0.06], [x1 - 0.085, ly + 0.02, -TOP_T - 0.005], 0.002), "bench_steel")
        put("world", "rail_x", P.box([x0 + 0.085, ly - 0.02, zf + 0.16], [x1 - 0.085, ly + 0.02, zf + 0.20], 0.002), "bench_steel")
    for sx in (-1, 1):
        lx = cx + sx * (bx - 0.06)
        put("world", "rail_y", P.box([lx - 0.02, y0 + 0.085, -TOP_T - 0.06], [lx + 0.02, y1 - 0.085, -TOP_T - 0.005], 0.002), "bench_steel")
        put("world", "rail_y", P.box([lx - 0.02, y0 + 0.085, zf + 0.16], [lx + 0.02, y1 - 0.085, zf + 0.20], 0.002), "bench_steel")
    put("world", "shelf", P.box([x0 + 0.07, y0 + 0.07, zf + 0.20], [x1 - 0.07, y1 - 0.07, zf + 0.215], 0.0015), "bench_shelf")

    # ---- the form board with the episode's print
    hx, hy = H.BOARD_HALF
    V, N, UV, F_ = G.slab(G.rounded_rect(hx, hy, 0.006, 6), 0.0, H.BOARD_T, 0.0012, 0.0008)
    top = N[:, 2] > 0.9
    UV = np.stack([(V[:, 0] + hx) / (2 * hx), (V[:, 1] + hy) / (2 * hy)], 1)
    V = V + [H.BOARD_C[0], H.BOARD_C[1], 0.0]

    def sub(Fsel):
        used = np.unique(Fsel)
        remap = -np.ones(len(V), int)
        remap[used] = np.arange(len(used))
        return V[used], N[used], UV[used], remap[Fsel]
    put("world", "board_top", sub(F_[top[F_].all(1)]), "board", {"uv": "own", "image": "board_print"})
    put("world", "board_edge", G.uv_box(sub(F_[~top[F_].all(1)])), "board_edge")
    for sx in (-1, 1):
        for sy in (-1, 1):
            put("world", "board_screw", P.screw([H.BOARD_C[0] + sx * (hx - 0.02), H.BOARD_C[1] + sy * (hy - 0.02), H.BOARD_T], [0, 0, 1.0], r=0.0045, h=0.0006), "screw")

    # ---- the connector: housing, bracket (fixed to the board), the strain-relief boot (it moves with the cable's end)
    hd = ep.conn_heading
    Rc = rotz(hd)
    cc = np.array([ep.conn_pos[0], ep.conn_pos[1], H.BOARD_T + R])
    put("world", "conn_housing", G.transform(G.bevel_box([0.012, 0.009, 0.008], 0.0012), Rc, cc + Rc @ np.array([-0.012, 0, 0.0035])), "conn_body")
    for nm, half, at in (("bracket_foot", [0.011, 0.014, 0.0011], [-0.036, 0, -R + 0.0011]), ("bracket_up", [0.0011, 0.014, 0.011], [-0.0251, 0, -R + 0.011])):
        put("world", nm, G.transform(G.bevel_box(half, 0.0004), Rc, cc + Rc @ np.array(at)), "bracket")
    for sy in (0.007, -0.007):
        put("world", "bracket_screw", P.screw(cc + Rc @ np.array([-0.039, sy, -R + 0.0022]), [0, 0, 1.0], r=0.003, h=0.0012), "screw")
    boot = G.lathe([(0.0, 0.0), (R + 0.0032, 0.0), (R + 0.0028, 0.006), (R + 0.0012, 0.016), (R + 0.0004, 0.020)], nseg=32)
    Rb = Rc @ np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])           # the lathe's z axis along the connector's axis
    put("strain_relief", "conn_boot", G.transform(boot, Rb, [0.0, 0.0, 0.0]), "conn_boot")

    # ---- fixtures, in the frames of their bodies
    for i, f in enumerate(ep.fixtures):
        if f["type"] == "clip":
            off = f["gap"] / 2 + H.PIN_R
            L = H.barb_length(f["gap"], R)
            for sgn, nm in ((1, "a"), (-1, "b")):
                pin = G.merge(P.smooth_cyl([0, sgn * off, 0.0], [0, sgn * off, H.PIN_H], H.PIN_R, nseg=24, caps=True),
                              P.ball([0, sgn * off, H.PIN_H], H.PIN_R, nlat=8, nlon=24))
                put("clip%d" % i, "pin", pin, "pin_steel")
                fin = G.transform(G.bevel_box([H.BARB_W, H.LEAD_LEN / 2, H.LEAD_T], 0.0003), rotx(sgn * H.LEAD_ANGLE),
                                  [0, sgn * (off + (H.LEAD_LEN / 2) * math.cos(H.LEAD_ANGLE)), H.PIN_H + (H.LEAD_LEN / 2) * math.sin(H.LEAD_ANGLE)])
                put("clip%d" % i, "fin", fin, "fin_nylon")
                tab = G.transform(G.bevel_box([H.BARB_W, L / 2, H.BARB_T], 0.0001), np.eye(3), [0, -sgn * L / 2, 0.0])
                put("barb%d%s" % (i, nm), "tab", tab, "tab_nylon")
        else:
            p = np.array([f["pos"][0], f["pos"][1], H.BOARD_T])
            put("world", "post", P.smooth_cyl(p, p + [0, 0, H.POST_H], H.POST_R, nseg=64, bevel=0.0006), "post_alu")
            CR, PH = H.POST_CAP_R, H.POST_H
            capm = G.lathe([(0.0, PH), (CR - 0.001, PH), (CR, PH + 0.001), (CR, PH + 0.005), (CR - 0.001, PH + 0.006), (0.0, PH + 0.006)], nseg=64)
            put("world", "post_cap", G.transform(capm, np.eye(3), p), "post_cap")
            put("world", "post_screw", P.screw(p + [0, 0, PH + 0.006], [0, 0, 1.0], r=0.004, h=0.0008), "screw")

    # ---- the cable: per segment a tube piece along its body's x and a ball at its joint; a ferrule on the free end
    seg = ep.length / H.N_SEG
    # the joint's ball is 0.05 mm under the tube's radius: at the same radius its surface lies in the tubes'
    names = [m.body(b).name for b in range(m.nbody) if m.body(b).name.startswith("cB")]
    piece = G.merge(P.smooth_cyl([0, 0, 0], [seg, 0, 0], R, nseg=20, caps=False), P.ball([0, 0, 0], R - 0.00005, nlat=6, nlon=20))
    for bn in names:
        put(bn, "cable", piece, "cable")
    put(names[-1], "cable_end", P.smooth_cyl([seg - 0.001, 0, 0], [seg + 0.004, 0, 0], R * 0.72, nseg=20, caps=True), "cable_end")

    # ---- the bench's things, at their contact shapes' places and inside their sizes
    for name, kind, c, size in H.FURNITURE:
        c = np.array(c, float)
        if name.startswith("reel"):
            r, h = size
            zb = c[2] - h
            put("world", name, P.smooth_cyl([c[0], c[1], zb], [c[0], c[1], zb + 0.003], r, nseg=48, bevel=0.0008), "reel_flange")
            put("world", name, P.smooth_cyl([c[0], c[1], zb + 2 * h - 0.003], [c[0], c[1], zb + 2 * h], r, nseg=48, bevel=0.0008), "reel_flange")
            put("world", name, P.smooth_cyl([c[0], c[1], zb + 0.003], [c[0], c[1], zb + 2 * h - 0.003], r - 0.010, nseg=48),
                "reel_cable_blue" if name.endswith("blue") else "reel_cable_black")
        elif name.startswith("bin"):
            sx, sy, sz = size
            lo, hi = c - [sx, sy, sz], c + [sx, sy, sz]
            look, t = name, 0.003
            put("world", name, P.box(lo, [hi[0], hi[1], lo[2] + t], 0.0005), look)
            put("world", name, P.box(lo, [lo[0] + t, hi[1], hi[2]], 0.0005), look)
            put("world", name, P.box([hi[0] - t, lo[1], lo[2]], hi, 0.0005), look)
            put("world", name, P.box(lo, [hi[0], lo[1] + t, hi[2]], 0.0005), look)
            put("world", name, P.box([lo[0], hi[1] - t, lo[2]], hi, 0.0005), look)
        else:
            r, h = size
            put("world", name, P.sleeve([c[0], c[1], c[2] - h], [c[0], c[1], c[2] + h], r, r * 0.55, nseg=48, bevel=0.0008), "tape")

    # ---- the robot: the Panda's visual meshes; the fingers' metal and pads; the cable jaws at their geoms
    for g in range(m.ngeom):
        body = m.body(int(m.geom_bodyid[g])).name
        if not body.startswith("r_"):
            continue
        name = m.geom(g).name
        t = int(m.geom_type[g])
        if t == MESH and int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0:
            if name.endswith(("_g_finger1_visual", "_g_finger2_visual")):
                sg = 1.0 if name.endswith("finger1_visual") else -1.0
                Vg, Ng = model_soup(m, g)
                put(body, "finger_metal", P.finger_metal(Vg, Ng, sg), "finger")
                put(body, "finger_pad", P.finger_pad(sg), "finger_pad")
            else:
                look = panda_look(m, g)
                mesh = model_mesh(m, g)
                if look != "panda_white":
                    mesh = lift(mesh, PROUD)
                put(body, "panda_%d" % g, mesh, look)
        elif t == BOX and (name.endswith("_jaw") or "groove" in name):
            Rg = quat_mat(m.geom_quat[g])
            half = np.array(m.geom_size[g][:3], float)
            if not name.endswith("_jaw"):
                half[0] -= 0.0003       # a groove face ends 0.3 mm inside the carrier's end: drawn flush, the two end faces lie in one plane and the picture differs between processes in a pixel
            put(body, "jaw", G.transform(G.bevel_box(half, 0.0002), Rg, m.geom_pos[g]), "jaw" if name.endswith("_jaw") else "jaw_face")
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
        tex = ("image", opt["image"]) if opt.get("image") else (("static", stex) if stex is not None else None)
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

    def items(self, images):
        out = []
        for g in self.groups.values():
            N = np.concatenate(g["N"])
            N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
            it = {"frame": g["frame"], "look": g["look"], "tex": g["tex"], "mat": g["mat"],
                  "mesh": (np.ascontiguousarray(np.concatenate(g["V"]), np.float32), np.ascontiguousarray(N, np.float32),
                           np.ascontiguousarray(np.concatenate(g["U"]), np.float32),
                           np.ascontiguousarray(np.concatenate(g["F"]).reshape(-1), np.uint32))}
            if g["tex"] is not None and g["tex"][0] == "image":
                it["image"] = images[g["tex"][1]]
            out.append(it)
        return out


SKINNED = ("cable",)                # looks drawn as ONE mesh whose vertices follow their bodies (see items)


def items(ep, m):
    """The parts as the renderer's items: merged per frame and material. The cable is one item of its own kind
    (frame "skin", with `bone`: the body of every vertex): its forty segments overlap at every joint, and as forty
    meshes they were drawn in an order that could change from process to process, so that a pixel on a line where
    two of them cross differed by a level. As one mesh, moved by the renderer vertex by vertex, its triangles are
    always drawn in the same order."""
    col = Collector()
    skin = {}
    for body, _name, mesh, look, opt in parts(ep, m):
        if look in SKINNED and body != "world":
            V, N, UV, F = [np.asarray(x) for x in mesh]
            g = skin.setdefault(look, {"V": [], "N": [], "U": [], "F": [], "bone": [], "nv": 0})
            g["V"].append(V.astype(np.float64)); g["N"].append(N.astype(np.float64)); g["U"].append(UV.astype(np.float64))
            g["F"].append(F.reshape(-1, 3).astype(np.int64) + g["nv"])
            g["bone"].append(np.full(len(V), int(m.body(body).id), np.int32))
            g["nv"] += len(V)
            continue
        col.add("world" if body == "world" else ("body", int(m.body(body).id)), look, mesh, opt)
    out = col.items({"board_print": board_print(ep)})
    for look, g in skin.items():
        rgb, rough, metal, emis, stex, tile = material(look)
        N = np.concatenate(g["N"])
        out.append({"frame": "skin", "look": look, "tex": None, "mat": (rgb, rough, metal, emis),
                    "bone": np.concatenate(g["bone"]),
                    "mesh": (np.ascontiguousarray(np.concatenate(g["V"]), np.float32),
                             np.ascontiguousarray(N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12), np.float32),
                             np.ascontiguousarray(np.concatenate(g["U"]) / (tile or 1.0), np.float32),
                             np.ascontiguousarray(np.concatenate(g["F"]).reshape(-1), np.uint32))})
    return out
