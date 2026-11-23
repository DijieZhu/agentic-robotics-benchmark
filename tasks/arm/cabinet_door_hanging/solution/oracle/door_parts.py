"""Cabinet door hanging: the drawn parts the simulated model has no geometry of its own for, as meshes (V, N, UV, F)
in the frame of the body they belong to: hinge leaves and screws, the detent groove under a pin's head, the vacuum
tool's bellows, fitting, hose and clips, the gripper's soft pads, the fitter's things on the bench, the felt pieces,
the tool bag, the wall rail; and the turned shapes the model's rings of boxes stand for (a knuckle, a stand: a sleeve
with its bore). door_scene.py places them; door_sim.py gives the ones a hand or a carried door can reach a contact
shape of the same size.
"""
import math

import numpy as np

import door_geom as G

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


def hinge_leaves(hz, door):
    """The leaves of one hinge (centre hz above the door's bottom edge) in the carcass frame / the door frame (the same
    frame when the door hangs: origin on the hinge axis at the door's bottom edge, x into the cabinet, the door's leaf
    towards -y). Carcass leaf: a 2.5 mm plate on the outside of the carcass's side panel (its outer face at y =
    +0.003), from the axis back to x = 0.060, three screws. Door leaf: a plate let into the door's back face (0.2 mm
    proud: the door must pass between the rack's rails, 1 mm away each side) with its fold round the hinge edge, three
    screws. Returns [(name, mesh, look)]."""
    z0, z1 = hz - 0.0375 + 0.002, hz + 0.0375 - 0.002
    out = []
    if not door:
        out.append(("leaf", box([0.0, 0.0030, z0], [0.060, 0.0055, z1], 0.0006), "leaf_steel"))
        for k, (x, z) in enumerate(((0.022, hz - 0.024), (0.046, hz), (0.022, hz + 0.024))):
            out.append(("leaf_screw%d" % k, screw([x, 0.0055, z], [0, 1.0, 0]), "screw"))
    else:
        out.append(("leaf", box([0.0088, -0.066, z0], [0.0092, -0.0142, z1], 0.00005), "leaf_brass"))
        out.append(("leaf_fold", box([-0.0015, -0.0140, z0], [0.0092, -0.0085, z1], 0.0004), "leaf_brass"))
        for k, (y, z) in enumerate(((-0.026, hz - 0.024), (-0.050, hz), (-0.026, hz + 0.024))):
            out.append(("leaf_screw%d" % k, screw([0.0092, y, z], [1.0, 0, 0], h=0.0002), "screw"))
    return out


def cup_extras():
    """The vacuum tool's fittings in the right gripper body's frame (the post runs along +z from the flange at z = 0
    to the arm at z = 0.09; the cup faces +x, its pad's back at x = 0.035): three bellows convolutions round the arm's
    end behind the pad, a push-in fitting on the arm's rear end, and the hose up the post's back with two clips."""
    out = []
    for k, (x, r) in enumerate(((0.0215, 0.0235), (0.0265, 0.0265), (0.0315, 0.0285))):
        out.append(("bellows%d" % k, smooth_cyl([x - 0.0022, 0, 0.09], [x + 0.0022, 0, 0.09], r, nseg=48, bevel=0.0018), "cup_rubber"))
    out.append(("fitting", smooth_cyl([-0.0105, 0, 0.09], [-0.0185, 0, 0.09], 0.0065, nseg=24, bevel=0.0008), "fitting"))
    out.append(("fitting_nut", smooth_cyl([-0.0185, 0, 0.09], [-0.0225, 0, 0.09], 0.0050, nseg=6, bevel=0.0004), "fitting"))
    hx = -0.0235
    out.append(("hose", tube([[-0.0225, 0, 0.09], [hx - 0.004, 0, 0.088], [hx - 0.001, 0, 0.078], [hx, 0, 0.060], [hx, 0, 0.012], [hx - 0.006, 0, 0.002]], 0.0038), "hose"))
    for k, z in enumerate((0.066, 0.026)):
        out.append(("hose_clip%d" % k, box([hx - 0.0052, -0.0055, z - 0.004], [-0.0170, 0.0055, z + 0.004], 0.0008), "clip"))
    return out


def detent_groove(shaft_r, z):
    """the detent's groove on a pin's shaft: a dark band (the shaft's own radius; a surface mark, no geometry)"""
    return smooth_cyl([0, 0, z - 0.0009], [0, 0, z + 0.0009], shaft_r + 0.00006, nseg=32, caps=False)


def bench_things(zt):
    """The fitter's things on the bench (world frame; the bench top at zt): all on the robots' side behind the bases,
    or at the bench's ends, away from the racks, the stands and the ways the arms take. [(name, mesh, look)]"""
    out = []
    # a spirit level along the back edge, left
    out.append(("level_body", box([-0.255, 0.42, zt], [-0.225, 0.78, zt + 0.050], 0.0015), "level_alu"))
    out.append(("level_vial", smooth_cyl([-0.240, 0.585, zt + 0.0505], [-0.240, 0.615, zt + 0.0505], 0.0055, nseg=20, bevel=0.001), "vial"))
    out.append(("level_end0", box([-0.2565, 0.4185, zt], [-0.2235, 0.4245, zt + 0.0515], 0.001), "rubber_black"))
    out.append(("level_end1", box([-0.2565, 0.7755, zt], [-0.2235, 0.7815, zt + 0.0515], 0.001), "rubber_black"))
    # a soft-face mallet and a pin punch, rear right
    out.append(("mallet_head", smooth_cyl([-0.235, -0.700, zt + 0.022], [-0.235, -0.610, zt + 0.022], 0.022, nseg=36, bevel=0.003), "mallet_head"))
    out.append(("mallet_handle", smooth_cyl([-0.235, -0.655, zt + 0.022], [-0.050, -0.730, zt + 0.014], 0.011, nseg=24, bevel=0.002), "ash"))
    out.append(("punch", smooth_cyl([-0.200, -0.560, zt + 0.005], [-0.070, -0.585, zt + 0.005], 0.005, nseg=8, bevel=0.0008), "tool_steel"))
    # a tape measure and a pencil, rear right
    out.append(("tape_case", box([-0.250, -0.500, zt], [-0.185, -0.435, zt + 0.032], 0.006), "tape_yellow"))
    out.append(("tape_hook", box([-0.186, -0.474, zt + 0.004], [-0.176, -0.461, zt + 0.006], 0.0005), "tool_steel"))
    out.append(("pencil", smooth_cyl([-0.150, -0.470, zt + 0.0036], [-0.020, -0.455, zt + 0.0036], 0.0036, nseg=6, bevel=0.0005), "pencil"))
    # a felt mat where the pulled pins are laid down (0.5 mm), and a felt strip in the parking rack's channel
    out.append(("pin_mat", box([0.215, 0.405, zt], [0.345, 0.565, zt + 0.0005], 0.0002), "felt"))
    return out


def park_felt(x0, x1, y0, y1, zt):
    return [("park_felt", box([x0, y0, zt], [x1, y1, zt + 0.0004], 0.0001), "felt")]


def tool_bag():
    """a canvas tool bag on the floor at the bench's right end (world frame, its own placement by the caller)"""
    out = [("bag_body", box([-0.20, -0.11, 0.0], [0.20, 0.11, 0.22], 0.03), "canvas"),
           ("bag_base", box([-0.205, -0.115, 0.0], [0.205, 0.115, 0.035], 0.01), "rubber_black")]
    for s in (-1, 1):
        pts = [[-0.09, s * 0.07, 0.215], [-0.07, s * 0.045, 0.30], [0.07, s * 0.045, 0.30], [0.09, s * 0.07, 0.215]]
        out.append(("bag_handle%d" % (s + 1), tube(pts, 0.008, nseg=12), "strap"))
    return out


def wall_rail(depth, width, top):
    """The hanging rail and its two brackets behind the cabinet, in the carcass frame: the task's wall stands 5 cm
    behind the carcass's back (x = 0.015 + depth); the rail fills that stand-off."""
    xb = 0.015 + depth
    out = [("rail", box([xb + 0.012, -width - 0.03, top - 0.075], [xb + 0.060, 0.03, top - 0.040], 0.002), "rail_steel")]
    for k, y in enumerate((-width + 0.05, -0.05)):
        out.append(("rail_bracket%d" % k, box([xb, y - 0.02, top - 0.10], [xb + 0.014, y + 0.02, top - 0.015], 0.0015), "rail_steel"))
        out.append(("rail_screw%d" % k, screw([xb + 0.014, y, top - 0.03], [1.0, 0, 0]), "screw"))
    out.append(("spacer", box([xb, -width + 0.03, 0.02], [xb + 0.060, -0.03, 0.045], 0.002), "rail_steel"))
    return out


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


def contact_shapes(unit, zt, depth, width, top, bag_at):
    """The contact shapes of the drawn parts a hand or a carried door can reach, each the drawn part's own size:
    [(name, frame, kind, a, b, r)] with frame "world" | "carcass" | "cup" (the vacuum tool's body); kind "box": a, b
    the two corners (in the frame; for the bag a box turned by bag_at's yaw about z at its place); kind "cyl": a, b
    the axis' ends and r the radius. Not listed, so drawn only: the hinge leaves, folds and screws (the doors': 0.2 mm
    proud of the door's own contact boxes, with 1 mm to the rack's rails; the carcass's: a 2.5 mm plate beside the
    knuckle column, where a door's knuckles pass when it is slid onto the hinge axis), the screws' heads (under 1 mm), the two felt pieces
    (half a millimetre; pins and a door rest on them), the gripper's soft pads (inside the fingers' own pads), the
    detent groove (a mark on the pin's shaft)."""
    out = []
    # the fitter's things on the bench (bench_things)
    out.append(("x_level", "world", "box", [-0.2565, 0.4185, zt], [-0.2235, 0.7815, zt + 0.0515], 0.0))
    out.append(("x_mallet_head", "world", "cyl", [-0.235, -0.700, zt + 0.022], [-0.235, -0.610, zt + 0.022], 0.022))
    out.append(("x_mallet_handle", "world", "cyl", [-0.235, -0.655, zt + 0.022], [-0.050, -0.730, zt + 0.014], 0.011))
    out.append(("x_punch", "world", "cyl", [-0.200, -0.560, zt + 0.005], [-0.070, -0.585, zt + 0.005], 0.005))
    out.append(("x_tape", "world", "box", [-0.250, -0.500, zt], [-0.176, -0.435, zt + 0.032], 0.0))
    out.append(("x_pencil", "world", "cyl", [-0.150, -0.470, zt + 0.0036], [-0.020, -0.455, zt + 0.0036], 0.0036))
    # the tool bag on the floor (tool_bag, at bag_at)
    out.append(("x_bag", "world", "bag", [-0.205, -0.115, 0.0], [0.205, 0.115, 0.30], 0.0))
    # the rail the cabinet hangs on (wall_rail), in the carcass frame
    xb = 0.015 + depth
    out.append(("x_rail", "carcass", "box", [xb + 0.012, -width - 0.03, top - 0.075], [xb + 0.060, 0.03, top - 0.040], 0.0))
    for k, y in enumerate((-width + 0.05, -0.05)):
        out.append(("x_rail_bracket%d" % k, "carcass", "box", [xb, y - 0.02, top - 0.10], [xb + 0.014, y + 0.02, top - 0.015], 0.0))
    out.append(("x_rail_spacer", "carcass", "box", [xb, -width + 0.03, 0.02], [xb + 0.060, -0.03, 0.045], 0.0))
    # the vacuum tool's bellows, fitting and hose with its clips (cup_extras)
    out.append(("x_bellows", "cup", "cyl", [0.0193, 0, 0.09], [0.0337, 0, 0.09], 0.0285))
    out.append(("x_fitting", "cup", "cyl", [-0.0225, 0, 0.09], [-0.0105, 0, 0.09], 0.0065))
    out.append(("x_hose", "cup", "box", [-0.0292, -0.0055, 0.002], [-0.0180, 0.0055, 0.0918], 0.0))
    return out
