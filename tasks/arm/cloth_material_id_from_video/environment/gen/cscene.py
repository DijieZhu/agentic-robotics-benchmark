"""cscene: what the three cameras of cloth_material_id_from_video draw besides the fabric and the room: the bench
(the table top of cphys with a steel frame), the robot's pedestal, the Panda's own visual meshes, the fabric clamp
in the Panda's fingers, the three machine-vision cameras on their floor stands, a stack of folded swatches. Plain
meshes (V, N, UV, F) with a look name each; crender.py draws them with MuJoCo's Filament renderer, and the same
parts are the path-traced reference scene the look was approved on (one scene, two renderers).

Nothing here is simulated: the table top's box, its front edge and the fabric are cphys / creal's; the clamp never
touches anything; the arm is drawn at the joint angles carm solves. The drawn table is the collision box with its
top edge broken by EDGE_R, so it lies inside that box.

Maintainers' side (the renderer of the recordings): never in the task image's layers or the agent's home.
"""
import math

import numpy as np

import cgeom as G
import cprim as P

FLOOR_Z = -0.78                   # the room's floor under the table top (z = 0)
TOP_T = 0.040                     # the table top's thickness (cphys: half size 0.02)
EDGE_R = 0.0022                   # the drawn top edge's break
PAD_T, JAW_T = 0.0015, 0.0035     # the upper jaw: a rubber pad under a plate
JAW_OPEN = 0.007                  # how far the upper jaw lifts when the clamp opens
MESH = 7                          # mujoco.mjtGeom.mjGEOM_MESH
LENS_LEN, BODY_FRONT = 0.034, 0.0268


def S(r, g, b):
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]


STEEL = [0.62, 0.615, 0.60]

# looks: base = linear rgb; col = a colour map of the render bundle (uv 0..1 over the part unless "tile" metres per repeat)
LOOKS = {
    "bench_lam": {"col": "lam_col", "rough": 0.46},
    "bench_edge": {"base": S(0.66, 0.665, 0.655), "rough": 0.46},
    "bench_steel": {"base": S(0.80, 0.81, 0.80), "rough": 0.44},
    "ped_steel": {"base": S(0.20, 0.21, 0.23), "rough": 0.44},
    "plate_steel": {"base": STEEL, "metal": 1.0, "rough": 0.30},
    "foot": {"base": [0.02, 0.02, 0.022], "rough": 0.7},
    "screw": {"base": [0.55, 0.55, 0.56], "metal": 1.0, "rough": 0.3},
    "boss_black": {"base": S(0.06, 0.06, 0.065), "metal": 1.0, "rough": 0.42},
    "clamp_alu": {"base": [0.70, 0.71, 0.73], "metal": 1.0, "rough": 0.34},
    "clamp_black": {"base": S(0.055, 0.055, 0.06), "metal": 1.0, "rough": 0.40},
    "blade": {"base": [0.66, 0.66, 0.67], "metal": 1.0, "rough": 0.22},
    "jaw_pad": {"base": [0.022, 0.022, 0.024], "rough": 0.70},
    "cam_body": {"base": S(0.07, 0.07, 0.075), "metal": 1.0, "rough": 0.45},
    "lens_barrel": {"base": S(0.03, 0.03, 0.032), "metal": 1.0, "rough": 0.36},
    "lens_glass": {"base": [0.01, 0.012, 0.02], "rough": 0.06},
    "stand_black": {"base": S(0.05, 0.05, 0.055), "rough": 0.42},
    "stand_chrome": {"base": [0.72, 0.72, 0.73], "metal": 1.0, "rough": 0.12},
    "panda_white": {"base": S(0.93, 0.93, 0.92), "rough": 0.36},
    "panda_black": {"base": S(0.06, 0.06, 0.065), "rough": 0.52},
    "panda_grey": {"base": S(0.25, 0.25, 0.26), "rough": 0.45},
    "panda_blue": {"base": S(0.04, 0.54, 0.78), "rough": 0.35},
    "panda_led_green": {"base": S(0.10, 0.75, 0.20), "rough": 0.35},
    "panda_led_red": {"base": S(0.80, 0.08, 0.06), "rough": 0.35},
    "default": {"base": [0.5, 0.5, 0.5], "rough": 0.5},
}
STACK = (("denim", 0.075, 0.055, 0.06), ("felt", 0.072, 0.052, -0.05), ("coated", 0.070, 0.050, 0.11))    # the folded swatches: look class, half sizes, yaw
for _n, _hx, _hy, _yaw in STACK:
    LOOKS["stack_" + _n] = {"col": "stack_%s_col" % _n, "rough": 0.9, "tile": 0.30}


def quat_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def model_soup(m, g):
    """a mesh geom of a MuJoCo model as a triangle soup with its normals, in its body's frame"""
    mid = int(m.geom_dataid[g])
    va = int(m.mesh_vertadr[mid])
    fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    Vm = m.mesh_vert[va:va + int(m.mesh_vertnum[mid])].astype(float)
    Fm = m.mesh_face[fa:fa + fn].astype(int)
    na = int(m.mesh_normaladr[mid])
    Nm = m.mesh_normal[na:na + int(m.mesh_normalnum[mid])].astype(float)
    FN = m.mesh_facenormal[fa:fa + fn].astype(int)
    R = quat_mat(m.geom_quat[g])
    V, N = Vm[Fm.reshape(-1)] @ R.T + m.geom_pos[g], Nm[FN.reshape(-1)] @ R.T
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


def soup_mesh(V):
    """a triangle soup (3 n, 3) as a mesh with flat normals (the camera body of the render bundle)"""
    V = np.asarray(V, float)
    fn = np.cross(V[1::3] - V[0::3], V[2::3] - V[0::3])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-15
    return G.uv_box((V, np.repeat(fn, 3, 0), np.zeros((len(V), 2)), np.arange(len(V)).reshape(-1, 3)))


def camera_frame(pos, target):
    """a camera's drawn frame: x along its view, y to its left, z up (columns)"""
    p, f = np.asarray(pos, float), np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    left = np.cross([0, 0, 1.0], f)
    left /= np.linalg.norm(left)
    return np.stack([f, left, np.cross(f, left)], 1)


def world_parts(edge_x, table_half, robot_base, cameras, cam_soup, rigs=True, stack=True):
    """The static parts in the world frame: [(name, mesh, look)], and the boxes of the bench's frame and the pedestal
    (for distance checks). cameras: [(name, position, look-at target)]; cam_soup: the camera body's triangles."""
    out, boxes = [], []
    TX, TY = table_half

    def put(name, mesh, look):
        out.append((name, mesh, look))

    def wbox(name, lo, hi, b, look):
        boxes.append({"name": name, "lo": [float(v) for v in lo], "hi": [float(v) for v in hi]})
        put(name, P.box(lo, hi, b), look)
    cx = edge_x - TX
    outline = [(x + cx, y) for x, y in G.rounded_rect(TX, TY, 0.002, 4)]
    put("top_body", G.slab(outline, -TOP_T, 0.0, EDGE_R, 0.0008, top=False, bottom=True), "bench_edge")
    Vt, Nt, _u, Ft = G.cap(G.inset(outline, EDGE_R), 0.0, True)
    put("top_face", (Vt, Nt, np.stack([(Vt[:, 0] - (cx - TX)) / (2 * TX), (Vt[:, 1] + TY) / (2 * TY)], 1), Ft), "bench_lam")
    leg = 0.020
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = cx + sx * (TX - 0.045), sy * (TY - 0.045)
            wbox("leg", [lx - leg, ly - leg, FLOOR_Z + 0.02], [lx + leg, ly + leg, -TOP_T], 0.002, "bench_steel")
            put("foot", P.smooth_cyl([lx, ly, FLOOR_Z], [lx, ly, FLOOR_Z + 0.02], 0.026, nseg=32, bevel=0.003), "foot")
    for sy in (-1, 1):
        ly = sy * (TY - 0.045)
        for z0, z1 in ((-TOP_T - 0.06, -TOP_T - 0.004), (FLOOR_Z + 0.16, FLOOR_Z + 0.20)):
            wbox("rail_x", [cx - TX + 0.065, ly - 0.015, z0], [cx + TX - 0.065, ly + 0.015, z1], 0.002, "bench_steel")
    for sx in (-1, 1):
        lx = cx + sx * (TX - 0.045) if sx < 0 else cx + TX - 0.22        # the front rail is set back: a sheet hanging over the front edge swings in under the top
        for z0, z1 in ((-TOP_T - 0.06, -TOP_T - 0.004), (FLOOR_Z + 0.16, FLOOR_Z + 0.20)):
            wbox("rail_y", [lx - 0.015, -TY + 0.065, z0], [lx + 0.015, TY - 0.065, z1], 0.002, "bench_steel")
    wbox("shelf", [cx - TX + 0.03, -TY + 0.03, FLOOR_Z + 0.20], [cx + TX - 0.03, TY - 0.03, FLOOR_Z + 0.218], 0.002, "bench_steel")
    bx, by = float(robot_base[0]), float(robot_base[1])
    put("ped_floor_plate", P.box([bx - 0.17, by - 0.17, FLOOR_Z], [bx + 0.17, by + 0.17, FLOOR_Z + 0.012], 0.002), "plate_steel")
    wbox("ped_column", [bx - 0.075, by - 0.075, FLOOR_Z + 0.012], [bx + 0.075, by + 0.075, -0.014], 0.004, "ped_steel")
    wbox("ped_top_plate", [bx - 0.105, by - 0.105, -0.014], [bx + 0.105, by + 0.105, 0.0], 0.0015, "plate_steel")
    for sx in (-1, 1):
        for sy in (-1, 1):
            put("ped_anchor", P.smooth_cyl([bx + sx * 0.135, by + sy * 0.135, FLOOR_Z + 0.012], [bx + sx * 0.135, by + sy * 0.135, FLOOR_Z + 0.022],
                                           0.011, nseg=6, bevel=0.001), "screw")
    if stack:
        z = 0.0
        for nm, hx, hy, yaw in STACK:
            t = 0.0057
            m = G.xform2(G.transform(G.bevel_box(np.array([hx, hy, t / 2]), 0.0012), np.eye(3), [0, 0, z + t / 2]), yaw, (cx - TX + 0.115, TY - 0.095, 0.0))
            put("stack_" + nm, m, "stack_" + nm)
            z += t
    stands = []
    cam_mesh = soup_mesh(cam_soup) if (rigs and cam_soup is not None) else None
    for name, pos, tgt in (cameras if rigs else []):
        p = np.asarray(pos, float)
        Rc = camera_frame(pos, tgt)
        f = Rc[:, 0]
        o = p - Rc @ np.array([BODY_FRONT + LENS_LEN, 0.0, 0.0])  # the body's origin: the lens front sits at the camera's position
        put(name + "_body", G.transform(cam_mesh, Rc, o), "cam_body")
        put(name + "_lens", G.transform(P.smooth_cyl([BODY_FRONT, 0, 0], [BODY_FRONT + LENS_LEN, 0, 0], 0.0150, nseg=48, bevel=0.0012), Rc, o), "lens_barrel")
        put(name + "_ring", G.transform(P.sleeve([BODY_FRONT + 0.010, 0, 0], [BODY_FRONT + 0.019, 0, 0], 0.0162, 0.0148, nseg=48, bevel=0.0006), Rc, o), "lens_barrel")
        put(name + "_glass", G.transform(P.smooth_cyl([BODY_FRONT + LENS_LEN - 0.0005, 0, 0], [BODY_FRONT + LENS_LEN + 0.0004, 0, 0], 0.0115, nseg=48), Rc, o), "lens_glass")
        fh = np.array([f[0], f[1], 0.0])
        fh /= np.linalg.norm(fh)
        c = p - fh * 0.050                                        # the stand's axis, behind the lens
        mount = o + Rc @ np.array([0.003, 0.0, -0.0145])
        ball = np.array([c[0], c[1], mount[2] - 0.030])
        put(name + "_plate", G.transform(P.box([-0.020, -0.016, -0.0185], [0.026, 0.016, -0.0145], 0.0008), Rc, o), "boss_black")
        put(name + "_stem", P.smooth_cyl(ball, mount + Rc @ np.array([0, 0, -0.004]), 0.005, nseg=24), "stand_chrome")
        put(name + "_ball", P.ball(ball, 0.014, 12, 24), "stand_chrome")
        put(name + "_head", P.smooth_cyl([c[0], c[1], ball[2] - 0.040], [c[0], c[1], ball[2] - 0.008], 0.017, nseg=32, bevel=0.002), "stand_black")
        zc = ball[2] - 0.040
        zm = FLOOR_Z + 0.55 * (zc - FLOOR_Z)
        put(name + "_col_up", P.smooth_cyl([c[0], c[1], zm - 0.05], [c[0], c[1], zc], 0.0095, nseg=32), "stand_chrome")
        put(name + "_col_low", P.smooth_cyl([c[0], c[1], FLOOR_Z + 0.012], [c[0], c[1], zm], 0.0135, nseg=32, bevel=0.001), "stand_black")
        put(name + "_collar", P.smooth_cyl([c[0], c[1], zm - 0.012], [c[0], c[1], zm + 0.016], 0.0185, nseg=32, bevel=0.002), "stand_black")
        put(name + "_base", P.smooth_cyl([c[0], c[1], FLOOR_Z], [c[0], c[1], FLOOR_Z + 0.012], 0.150, nseg=64, bevel=0.003), "stand_black")
        stands.append({"name": name, "lens_front": p.tolist(), "stand_axis": [float(c[0]), float(c[1])]})
    return out, boxes, stands


def arm_parts(m, prefix="r_"):
    """The Panda's visual meshes of a MuJoCo model that holds carm's arm: [(body id, name, mesh, look)]. Only mesh
    geoms of bodies named prefix...: carm's own clamp (boxes and a stem) is not drawn; clamp_parts is."""
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if int(m.geom_type[g]) != MESH or not m.body(b).name.startswith(prefix):
            continue
        out.append((b, "panda_%d" % g, model_soup(m, g), panda_look(m, g)))
    return out


def clamp_parts(jaw_w, r_clamp_eef, tool_tip, sheet):
    """The fabric clamp in the clamp frame of cphys (x into the fabric, y along the jaws, z up): (lower parts, upper
    parts, boxes). The lower part (a 0.15 mm blade under the fabric, the spine, the tang the Panda's fingers hold)
    moves with the commanded clamp pose; the upper part (pad and jaw plate) has its origin at the pad's lower face:
    drawn at the clamp pose lifted along the clamp's z by csheet.shell's pad_z while the jaws hold the fabric, and by
    pad_rest(thickness) + JAW_OPEN while they are open. sheet: the csheet module (its jaw and blade numbers)."""
    hw = jaw_w / 2 + 0.002
    bt, bb = sheet.BLADE_TOP, sheet.BLADE_TOP - sheet.BLADE_T
    boxes = {"lower": [["blade", [sheet.JAW_X0, -hw, bb], [sheet.JAW_X1, hw, bt]], ["spine", [-0.027, -hw, bb], [sheet.JAW_X0, hw, 0.021]]],
             "upper": [["pad", [sheet.JAW_X0 + 0.002, -hw + 0.001, 0.0], [sheet.JAW_X1 - 0.0005, hw - 0.001, PAD_T]],
                       ["jaw", [sheet.JAW_X0 + 0.0002, -hw, PAD_T], [sheet.JAW_X1, hw, PAD_T + JAW_T]]]}
    lower = [("blade", P.box(boxes["lower"][0][1], boxes["lower"][0][2], 0.00003), "blade"),
             ("spine", P.box(boxes["lower"][1][1], boxes["lower"][1][2], 0.0012), "clamp_black")]
    Re = np.asarray(r_clamp_eef, float)
    ax_t, ax_c, ax_w = Re[:, 2], Re[:, 1], Re[:, 0]              # tool axis, the fingers' closing direction (the tang's thickness), its width
    E = -ax_t * tool_tip
    reach = (E[2] - 0.0200) / max(1e-6, -ax_t[2])                 # along the tool axis from the eef down to the spine's top
    lower.append(("tang", G.transform(G.bevel_box(np.array([0.0105, 0.0012, (0.024 + reach) / 2]), 0.0004), np.stack([ax_w, ax_c, ax_t], 1),
                                      E + ax_t * (reach - 0.024) / 2), "clamp_alu"))
    for sy in (-1, 1):
        lower.append(("screw", P.smooth_cyl([-0.020, sy * 0.034, 0.021], [-0.020, sy * 0.034, 0.0245], 0.0042, nseg=24, bevel=0.0008), "screw"))
        lower.append(("hinge", P.smooth_cyl([-0.0125, sy * (hw - 0.004), 0.012], [-0.0125, sy * (hw + 0.0015), 0.012], 0.0030, nseg=20, bevel=0.0005), "screw"))
    upper = [("pad", P.box(boxes["upper"][0][1], boxes["upper"][0][2], 0.0003), "jaw_pad"),
             ("jaw", P.box(boxes["upper"][1][1], boxes["upper"][1][2], 0.0006), "clamp_black")]
    return lower, upper, boxes


def pad_rest(sheet, thickness):
    """the pad's lower face over the clamp plane on a flat held strip of the drawn thickness"""
    return sheet.BLADE_TOP + 2 * sheet.PAD_GAP + thickness


def fabric_looks(ids, n_looks):
    """Which look of the render bundle's library each fabric wears: {id: look index}. A function of the ids alone:
    the ids are put in the order of sha256("cloth_material_id_from_video/look/v1/" + id) and the k-th wears look
    k mod n_looks. Nothing of a fabric's material enters, and the ids are dealt before the materials are drawn."""
    import hashlib
    order = sorted(ids, key=lambda i: hashlib.sha256(("cloth_material_id_from_video/look/v1/" + str(i)).encode()).digest())
    return {i: k % int(n_looks) for k, i in enumerate(order)}
