"""Pipe spool push-fit assembly: the drawn parts as meshes (V, N, UV, F) in the frame of the body they belong to, and the
contact shapes of the drawn things an arm can reach. spool_scene.py places the meshes; spool_sim.py builds the
contact shapes into the model (the same boxes, so a hand cannot pass through what a camera shows).

Small builders (boxes with broken edges, turned sleeves and cylinders, balls, the Panda finger's metal and soft pad),
then this cell's parts: the bench, the pipes and the push-fit elbows inside their collision envelope, the camera
rig (a rear mast with a boom for the overhead and the front camera, a floor stand for the side camera, a camera
block at each wrist camera) and the stock on the bench's lower shelf.
"""
import math

import numpy as np

import spool_geom as G

PAD_Y_BACK, PAD_Y_FACE, PAD_Z, PAD_HX = 0.0080, -0.0005, (0.0362, 0.0539), 0.0088   # the gripper's tip block, finger frame
PAD_T = PAD_Y_BACK - PAD_Y_FACE


def box(lo, hi, b=0.0006):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return G.transform(G.bevel_box((hi - lo) / 2, b), np.eye(3), (lo + hi) / 2)


def sleeve(p0, p1, r_out, r_in, nseg=64, bevel=0.0):
    """a turned sleeve from p0 to p1: the outer wall (rims broken by `bevel`), the bore's wall and the two end faces"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    L = float(np.linalg.norm(ax))
    z = ax / L
    x = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    b = min(bevel, 0.45 * (r_out - r_in), 0.45 * L)
    rings = [(0.0, r_out - b, -0.7), (b, r_out, 0.0), (L - b, r_out, 0.0), (L, r_out - b, 0.7)] if b > 0 else [(0.0, r_out, 0.0), (L, r_out, 0.0)]
    m = G.Mesh()
    for i in range(nseg):
        a0, a1 = 2 * math.pi * i / nseg, 2 * math.pi * (i + 1) / nseg
        d0, d1 = math.cos(a0) * x + math.sin(a0) * y, math.cos(a1) * x + math.sin(a1) * y
        for (h0, r0, t0), (h1, r1, t1) in zip(rings[:-1], rings[1:]):
            m.quad([p0 + z * h0 + r0 * d0, p0 + z * h0 + r0 * d1, p0 + z * h1 + r1 * d1, p0 + z * h1 + r1 * d0],
                   np.asarray([d0 + t0 * z, d1 + t0 * z, d1 + t1 * z, d0 + t1 * z]),
                   [(r_out * a0, h0), (r_out * a1, h0), (r_out * a1, h1), (r_out * a0, h1)])
        m.quad([p0 + r_in * d1, p0 + r_in * d0, p0 + z * L + r_in * d0, p0 + z * L + r_in * d1], np.asarray([-d1, -d0, -d0, -d1]),
               [(r_in * a1, 0.0), (r_in * a0, 0.0), (r_in * a0, L), (r_in * a1, L)])
        ro = r_out - b
        m.quad([p0 + z * L + r_in * d0, p0 + z * L + ro * d0, p0 + z * L + ro * d1, p0 + z * L + r_in * d1], z,
               [(r_in * math.cos(a0), r_in * math.sin(a0)), (ro * math.cos(a0), ro * math.sin(a0)),
                (ro * math.cos(a1), ro * math.sin(a1)), (r_in * math.cos(a1), r_in * math.sin(a1))])
        m.quad([p0 + r_in * d1, p0 + ro * d1, p0 + ro * d0, p0 + r_in * d0], -z,
               [(r_in * math.cos(a1), r_in * math.sin(a1)), (ro * math.cos(a1), ro * math.sin(a1)),
                (ro * math.cos(a0), ro * math.sin(a0)), (r_in * math.cos(a0), r_in * math.sin(a0))])
    return m.arrays()


def smooth_cyl(p0, p1, r, nseg=64, bevel=0.0, caps=True):
    """a cylinder with smooth side normals (G.cylinder's sides are faceted), its rims broken by `bevel`"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    L = float(np.linalg.norm(ax))
    z = ax / L
    x = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    b = min(bevel, 0.45 * r, 0.45 * L)
    m = G.Mesh()
    rings = [(0.0, r - b, -0.7), (b, r, 0.0), (L - b, r, 0.0), (L, r - b, 0.7)] if b > 0 else [(0.0, r, 0.0), (L, r, 0.0)]
    for i in range(nseg):
        a0, a1 = 2 * math.pi * i / nseg, 2 * math.pi * (i + 1) / nseg
        d0, d1 = math.cos(a0) * x + math.sin(a0) * y, math.cos(a1) * x + math.sin(a1) * y
        for (h0, r0, t0), (h1, r1, t1) in zip(rings[:-1], rings[1:]):
            m.quad([p0 + z * h0 + r0 * d0, p0 + z * h0 + r0 * d1, p0 + z * h1 + r1 * d1, p0 + z * h1 + r1 * d0],
                   np.asarray([d0 + t0 * z, d1 + t0 * z, d1 + t1 * z, d0 + t1 * z]),
                   [(r * a0, h0), (r * a1, h0), (r * a1, h1), (r * a0, h1)])
        if caps:
            rc = r - b
            m.tri([p0 + z * L, p0 + z * L + rc * d0, p0 + z * L + rc * d1], z, [(0, 0), (rc * math.cos(a0), rc * math.sin(a0)), (rc * math.cos(a1), rc * math.sin(a1))])
            m.tri([p0, p0 + rc * d1, p0 + rc * d0], -z, [(0, 0), (rc * math.cos(a1), rc * math.sin(a1)), (rc * math.cos(a0), rc * math.sin(a0))])
    return m.arrays()


def tube(points, r, nseg=20):
    """a round tube along a polyline (smooth sides, a ball at every joint)"""
    ms = []
    P = [np.asarray(p, float) for p in points]
    for a, b in zip(P[:-1], P[1:]):
        ms.append(smooth_cyl(a, b, r, nseg=nseg, caps=True))
    for p in P[1:-1]:
        ms.append(ball(p, r))
    return G.merge(*ms)


def ball(c, r, nlat=8, nlon=16):
    m = G.Mesh()
    c = np.asarray(c, float)

    def pt(i, j):
        th, ph = math.pi * i / nlat, 2 * math.pi * j / nlon
        n = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        return c + r * n, n
    for i in range(nlat):
        for j in range(nlon):
            (a, na), (b, nb), (cc, nc), (dd, nd) = pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)
            m.quad([a, b, cc, dd], np.asarray([na, nb, nc, nd]), [(0, 0)] * 4)
    return m.arrays()


def screw(c, n, r=0.0034, h=0.0009):
    """a countersunk screw head: a low disc at c, proud by h along n, with a slot"""
    c, n = np.asarray(c, float), np.asarray(n, float)
    head = smooth_cyl(c - 0.0002 * n, c + h * n, r, nseg=24, bevel=0.0004)
    return head




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
    """a gripper finger's visual mesh (a triangle soup with normals, finger frame) without its tip block (everything
    with z > z_cut and sign * y < PAD_Y_BACK cut away, the cut closed): the soft pad stands in the block's place"""
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


def finger_pad(sign):
    """the soft pad in the finger's frame: its back on the shank's face, its face where the model's pad meets a part"""
    hz = (PAD_Z[1] - PAD_Z[0]) / 2
    V, N, UV, F = G.slab(G.rounded_rect(PAD_HX, hz, 0.0026, 6), 0.0, PAD_T, 0.0012, 0.0, top=True, bottom=False)
    R = np.array([[sign, 0, 0], [0, 0, -sign], [0, 1.0, 0]])
    return V @ R.T + np.array([0, sign * PAD_Y_BACK, 0.5 * (PAD_Z[0] + PAD_Z[1])]), N @ R.T, UV, (F if sign > 0 else F)



# ---------------------------------------------------------------------------------------------------- this cell
FLOOR_Z = -0.10                    # the hall's floor in the task's frame (the bench top is at 0.80: a 0.90 m bench)
TOP_T = 0.040
BENCH_X0 = -0.28
WALL = 0.0009                      # the copper tube's wall
MAST_XY = (-0.47, 0.0)
MAST_H = 0.02                      # the mast's and the boom's half section
BOOM_UP = 0.150                    # the boom's axis above the overhead camera's lens
WRIST_BOX = ((-0.030, -0.013, -0.013), (-0.006, 0.013, 0.013))      # the wrist camera's block, camera frame (x forward)
LENS_LEN, BODY_FRONT = 0.034, 0.0268


def cam_frame(pos, tgt):
    """a fixed camera's drawn frame (x forward, y left, z up) and the body mesh's origin: the lens front at `pos`"""
    p, f = np.asarray(pos, float), np.asarray(tgt, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    left = np.cross([0, 0, 1.0], f)
    left /= np.linalg.norm(left)
    Rc = np.stack([f, left, np.cross(f, left)], 1)
    return Rc, p - Rc @ np.array([BODY_FRONT + LENS_LEN, 0.0, 0.0])


def rig_geometry(cameras, table_z):
    """The rig's numbers from the task's cameras ({name: (position, look-at, ...)}): frames, mounts, the boom."""
    out = {}
    for k in ("overhead", "front", "side"):
        Rc, o = cam_frame(cameras[k][0], cameras[k][1])
        out[k] = {"R": Rc, "o": o, "p": np.asarray(cameras[k][0], float), "mount": o + Rc @ np.array([0.003, 0.0, -0.0185])}
    out["boom_z"] = float(cameras["overhead"][0][2]) + BOOM_UP
    out["boom_x1"] = float(cameras["overhead"][0][0]) + 0.06
    f = out["side"]["R"][:, 0]
    fh = np.array([f[0], f[1], 0.0])
    fh /= np.linalg.norm(fh)
    c = out["side"]["p"] - fh * 0.050
    out["side_axis"] = (float(c[0]), float(c[1]))
    out["side_ball"] = np.array([c[0], c[1], out["side"]["mount"][2] - 0.030])
    out["floor"] = table_z - 0.90
    return out


def contact_shapes(cameras, table_z):
    """Boxes (name, lo, hi) in the world frame of the drawn rig parts an arm can reach: the mast, the boom, the three
    fixed cameras (body and lens as one box each, axis-aligned round the drawn body), the side camera's stand."""
    g = rig_geometry(cameras, table_z)
    mx, my = MAST_XY
    out = [("x_mast", [mx - MAST_H, my - MAST_H, g["floor"]], [mx + MAST_H, my + MAST_H, g["boom_z"] + MAST_H]),
           ("x_boom", [mx - MAST_H, my - MAST_H, g["boom_z"] - MAST_H], [g["boom_x1"], my + MAST_H, g["boom_z"] + MAST_H])]
    for k in ("overhead", "front", "side"):
        Rc, o = g[k]["R"], g[k]["o"]
        cs = np.array([[x, y, z] for x in (-0.004, BODY_FRONT + LENS_LEN) for y in (-0.017, 0.017) for z in (-0.019, 0.017)]) @ Rc.T + o
        out.append(("x_cam_" + k, cs.min(0).tolist(), cs.max(0).tolist()))
    cx, cy = g["side_axis"]
    out.append(("x_side_stand", [cx - 0.02, cy - 0.02, g["floor"]], [cx + 0.02, cy + 0.02, float(g["side_ball"][2]) + 0.014]))
    out.append(("x_stem_overhead", [float(g["overhead"]["mount"][0]) - 0.006, -0.006, float(g["overhead"]["mount"][2])],
                [float(g["overhead"]["mount"][0]) + 0.006, 0.006, g["boom_z"] - MAST_H]))
    fm = g["front"]["mount"]
    out.append(("x_arm_front", [mx + MAST_H, -0.028, float(fm[2]) - 0.045], [float(fm[0]) + 0.006, 0.028, float(fm[2])]))
    return out


def bench(table_z, edge_x, half_y, bases):
    """the worktop on a steel frame, an adapter plate under each robot; [(name, mesh, look, options)], world frame"""
    out = []
    x0, x1, hy, zt = BENCH_X0, edge_x, half_y, table_z
    cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
    outline = [(x + cx, y) for x, y in G.rounded_rect(hx, hy, 0.003, 4)]
    er = 0.0015
    out.append(("top_body", G.xform2(G.slab(outline, -TOP_T, 0.0, er, 0.0008, top=False, bottom=True), 0.0, (0, 0, zt)), "bench_edge", {}))
    # the top face as tiles in one plane: the wood, and flush in it an adapter plate under each robot (no face of the
    # scene crosses another one or lies on one facing the same way: such pixels can go either way between processes)
    ph = 0.13
    xs = sorted({x0 + er, x1 - er} | {b[0] + sg * ph for b in bases for sg in (-1, 1)})
    ys = sorted({-hy + er, hy - er} | {b[1] + sg * ph for b in bases for sg in (-1, 1)})
    wood, plate = G.Mesh(), G.Mesh()
    for xa, xb in zip(xs[:-1], xs[1:]):
        for ya, yb in zip(ys[:-1], ys[1:]):
            mx_, my_ = (xa + xb) / 2, (ya + yb) / 2
            on = any(abs(mx_ - b[0]) < ph and abs(my_ - b[1]) < ph for b in bases)
            q = [np.array([xa, ya, zt]), np.array([xb, ya, zt]), np.array([xb, yb, zt]), np.array([xa, yb, zt])]
            (plate if on else wood).quad(q, [0, 0, 1.0], [((v[1] + hy) / (2 * hy), (v[0] - x0) / (2 * hx)) if not on else (v[0], v[1]) for v in q])
    out.append(("top_face", wood.arrays(), "bench_top", {"uv": "own"}))
    out.append(("plate", plate.arrays(), "plate_steel", {}))
    fz, leg = zt - 0.90, 0.025
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = cx + sx * (hx - 0.05), sy * (hy - 0.05)
            out.append(("leg", box([lx - leg, ly - leg, fz + 0.02], [lx + leg, ly + leg, zt - TOP_T], 0.002), "bench_steel", {}))
            out.append(("foot", smooth_cyl([lx, ly, fz], [lx, ly, fz + 0.02], 0.03, nseg=32, bevel=0.003), "foot", {}))
    for sy in (-1, 1):
        ly = sy * (hy - 0.05)
        for z0, z1 in ((zt - TOP_T - 0.07, zt - TOP_T - 0.004), (fz + 0.16, fz + 0.21)):
            out.append(("rail_x", box([cx - hx + 0.075, ly - 0.02, z0], [cx + hx - 0.075, ly + 0.02, z1], 0.002), "bench_steel", {}))
    for sx in (-1, 1):
        lx = cx + sx * (hx - 0.05)
        for z0, z1 in ((zt - TOP_T - 0.07, zt - TOP_T - 0.004), (fz + 0.16, fz + 0.21)):
            out.append(("rail_y", box([lx - 0.02, -hy + 0.075, z0], [lx + 0.02, hy - 0.075, z1], 0.002), "bench_steel", {}))
    out.append(("shelf", box([cx - hx + 0.075, -hy + 0.075, fz + 0.21], [cx + hx - 0.075, hy - 0.075, fz + 0.228], 0.002), "bench_steel", {}))
    for bx, by in bases:
        for ax_, ay_ in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            out.append(("plate_bolt", smooth_cyl([bx + ax_ * 0.115, by + ay_ * 0.115, zt], [bx + ax_ * 0.115, by + ay_ * 0.115, zt + 0.0045],
                                                 0.006, nseg=6, bevel=0.0008), "screw", {}))
    return out


def camera(name, Rc, o, cam_mesh):
    out = [(name + "_body", G.transform(cam_mesh, Rc, o), "cam_body", {}),
           (name + "_lens", G.transform(smooth_cyl([BODY_FRONT, 0, 0], [BODY_FRONT + LENS_LEN, 0, 0], 0.0150, nseg=48, bevel=0.0012), Rc, o), "lens_barrel", {}),
           (name + "_ring", G.transform(sleeve([BODY_FRONT + 0.010, 0, 0], [BODY_FRONT + 0.019, 0, 0], 0.0162, 0.0148, nseg=48, bevel=0.0006), Rc, o), "lens_barrel", {}),
           (name + "_glass", G.transform(smooth_cyl([BODY_FRONT + LENS_LEN - 0.0005, 0, 0], [BODY_FRONT + LENS_LEN + 0.0004, 0, 0], 0.0115, nseg=48), Rc, o), "lens_glass", {}),
           (name + "_plate", G.transform(box([-0.020, -0.016, -0.0185], [0.026, 0.016, -0.0145], 0.0008), Rc, o), "boss_black", {})]
    return out


def rig(cameras, table_z, cam_mesh):
    """the three fixed cameras, the rear mast and boom, the side camera's floor stand (world frame)"""
    g = rig_geometry(cameras, table_z)
    fz, bz = g["floor"], g["boom_z"]
    mx, my = MAST_XY
    h = MAST_H
    out = [("mast_base", smooth_cyl([mx, my, fz], [mx, my, fz + 0.014], 0.20, nseg=64, bevel=0.003), "stand_black", {}),
           ("mast", box([mx - h, my - h, fz + 0.014], [mx + h, my + h, bz - h], 0.002), "post_alu", {}),
           ("boom", box([mx - h, my - h, bz - h], [g["boom_x1"], my + h, bz + h], 0.002), "post_alu", {}),
           ("boom_gusset", box([mx + h, my - 0.003, bz - 0.10], [mx + 0.10, my + 0.003, bz - h], 0.0006), "boss_black", {})]
    for k in ("overhead", "front", "side"):
        out += camera(k, g[k]["R"], g[k]["o"], cam_mesh)
    m_ = g["overhead"]["mount"]
    out.append(("overhead_stem", smooth_cyl([m_[0], m_[1], bz - h], m_, 0.006, nseg=24), "stand_chrome", {}))
    m_ = g["front"]["mount"]
    out.append(("front_clamp", box([mx + h, -0.028, m_[2] - 0.045], [mx + h + 0.020, 0.028, m_[2] - 0.005], 0.002), "boss_black", {}))
    out.append(("front_arm", smooth_cyl([mx + h + 0.020, 0.0, m_[2] - 0.025], m_, 0.006, nseg=24), "stand_chrome", {}))
    cx, cy = g["side_axis"]
    b_, m_ = g["side_ball"], g["side"]["mount"]
    zc = b_[2] - 0.040
    zm = fz + 0.55 * (zc - fz)
    out += [("side_stem", smooth_cyl(b_, m_, 0.005, nseg=24), "stand_chrome", {}), ("side_ball", ball(b_, 0.014, 12, 24), "stand_chrome", {}),
            ("side_head", smooth_cyl([cx, cy, b_[2] - 0.040], [cx, cy, b_[2] - 0.008], 0.017, nseg=32, bevel=0.002), "stand_black", {}),
            ("side_col_up", smooth_cyl([cx, cy, zm - 0.05], [cx, cy, zc], 0.0095, nseg=32), "stand_chrome", {}),
            ("side_col_low", smooth_cyl([cx, cy, fz + 0.012], [cx, cy, zm], 0.0135, nseg=32, bevel=0.001), "stand_black", {}),
            ("side_collar", smooth_cyl([cx, cy, zm - 0.012], [cx, cy, zm + 0.016], 0.0185, nseg=32, bevel=0.002), "stand_black", {}),
            ("side_base", smooth_cyl([cx, cy, fz], [cx, cy, fz + 0.012], 0.150, nseg=64, bevel=0.003), "stand_black", {})]
    return out


def wrist_camera(Rb, pc):
    """the block at a wrist camera (hand frame): Rb's columns are the camera's forward, right and up"""
    lo, hi = WRIST_BOX
    return [("wcam_body", G.transform(box(lo, hi, 0.0012), Rb, pc), "cam_body", {}),
            ("wcam_lens", G.transform(smooth_cyl([-0.006, 0, 0], [0.0, 0, 0], 0.0075, nseg=32, bevel=0.0008), Rb, pc), "lens_barrel", {}),
            ("wcam_glass", G.transform(smooth_cyl([-0.0004, 0, 0], [0.0003, 0, 0], 0.0055, nseg=32), Rb, pc), "lens_glass", {})]


def pipe(length, r, chamfer, band_z=None, band_half=0.0):
    """a pipe in its own frame (axis z): a copper tube, and its band as a ring of tape"""
    out = [("tube", sleeve([0, 0, -length / 2], [0, 0, length / 2], r, r - WALL, nseg=96, bevel=min(chamfer, 0.4 * WALL)), "copper", {})]
    if band_z is not None:
        out.append(("band", sleeve([0, 0, band_z - band_half], [0, 0, band_z + band_half], r + 0.00035, r, nseg=96, bevel=0.0001), "band_tape", {}))
    return out


def elbow(r_pipe, clear, sock_l, bore_d, r_out, corner_half, or_depth):
    """A push-fit elbow in its frame (origin at the corner's centre, socket 0 along +x, socket 1 along +y), inside its
    collision envelope: the forged block, two barrels with a release collar at the mouth, the bore at the unit's
    clearance, the O-ring. Nothing drawn shows the unit's stop depth: the bore ends where the block closes it."""
    chx, chy, chz = corner_half
    r_bore = r_pipe + clear
    out = [("corner", G.bevel_box(np.array([chx, chy, chz]), 0.0005), "fit_brass", {})]
    for j, u in ((0, np.array([1.0, 0, 0])), (1, np.array([0, 1.0, 0]))):
        mouth = sock_l * u
        base = chx * u
        out += [("barrel%d" % j, sleeve(base, mouth - 0.008 * u, r_out - 0.0006, r_bore, nseg=96, bevel=0.0006), "fit_brass", {}),
                ("collar%d" % j, sleeve(mouth - 0.008 * u, mouth, r_out, r_bore, nseg=96, bevel=0.0008), "fit_collar", {}),
                ("oring%d" % j, sleeve(mouth - (or_depth + 0.0026) * u, mouth - or_depth * u, r_bore, r_pipe + 0.0001, nseg=64), "oring", {}),
                ("bore_end%d" % j, smooth_cyl(mouth - (bore_d + 0.0005) * u, mouth - bore_d * u, r_bore, nseg=64), "fit_bore", {})]
    return out


def stock(table_z, r_pipe, elbow_parts, corner_hz):
    """a bundle of spare tube and a bin of spare elbows on the bench's lower shelf (out of every arm's reach)"""
    zs = table_z - 0.90 + 0.228
    out = []
    for k in range(7):
        xx, zz = 0.52 + 0.0225 * (k % 4) + 0.011 * (k // 4), zs + r_pipe + 0.0195 * (k // 4)
        out.append(("stock_tube", sleeve([xx, -0.70, zz], [xx, 0.30, zz], r_pipe, r_pipe - WALL, nseg=48, bevel=0.0003), "copper", {}))
    bx0, bx1, by0, by1, bh, bw = 0.02, 0.26, 0.30, 0.62, 0.11, 0.003
    out.append(("bin_floor", box([bx0, by0, zs], [bx1, by1, zs + bw], 0.0005), "bin_blue", {}))
    for lo, hi in (([bx0, by0, zs + bw], [bx0 + bw, by1, zs + bh]), ([bx1 - bw, by0, zs + bw], [bx1, by1, zs + bh]),
                   ([bx0 + bw, by0, zs + bw], [bx1 - bw, by0 + bw, zs + bh]), ([bx0 + bw, by1 - bw, zs + bw], [bx1 - bw, by1, zs + bh])):
        out.append(("bin_wall", box(lo, hi, 0.001), "bin_blue", {}))
    for cxy in ((0.06, 0.33), (0.15, 0.42), (0.06, 0.51)):                  # three spare elbows, clear of each other
        for name, mesh, look, opt in elbow_parts:
            out.append(("spare_" + name, G.xform2(mesh, 0.0, (cxy[0], cxy[1], zs + bw + corner_hz)), look, opt))
    return out
