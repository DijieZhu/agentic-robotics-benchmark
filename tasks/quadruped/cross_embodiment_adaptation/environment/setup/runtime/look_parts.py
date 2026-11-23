"""The course parts as built objects: steel barriers with orange caps and hazard ends, the duct, the tread-plate deck
with its guides, bollards, the dock beacon, the charging station, the pallet's foot brackets, the painted lane. Each
generator makes a part from the site's own numbers; a part lies inside or on the collision geom it stands for.
`Soup` collects what they make as triangles by group and material (hall_render.course maps them into the atlas)."""
import math

import numpy as np

import look_geom as G


class Soup(object):
    """the generators' writer: look(name, ...) declares a material; mesh(parent, name, arrays, look) adds a part"""

    def __init__(self):
        self.mats, self.parts, self.n = {}, [], 0

    def look(self, name, col=(0.5, 0.5, 0.5), rgh=0.5, nrm=None, metal=0.0, tile=1.0, emissive=None, opacity=1.0, ior=1.5,
             clearcoat=0.0):
        self.mats[name] = dict(col=col, rgh=rgh, metal=metal, tile=tile, emissive=emissive, opacity=opacity)

    def mesh(self, parent, name, m, look):
        V, N, UV, F = m
        self.n += 1
        F = np.asarray(F, np.int64).reshape(-1, 3)
        V, N = np.asarray(V, np.float32).astype(np.float64), np.asarray(N, np.float32).astype(np.float64)
        uv = (np.asarray(UV, np.float32) / np.float32(self.mats[look]["tile"])).astype(np.float64)
        T = np.concatenate([V[F], N[F], uv[F]], 2)
        fn = np.cross(T[:, 1, :3] - T[:, 0, :3], T[:, 2, :3] - T[:, 0, :3])
        back = (T[:, :, 3:6] * fn[:, None, :]).sum(2).mean(1) < 0      # a triangle wound against its normals: turned
        T[back] = T[back][:, ::-1]
        self.parts.append((parent, name, look, T))


I3 = np.eye(3)


def box(c, h, b=0.002):
    return G.transform(G.bevel_box(h, b), I3, c)


def cyl(p0, p1, r, bevel=0.0005, nseg=32):
    return G.cylinder(p0, p1, r, nseg=nseg, bevel=bevel)


def rslab(c, hx, hy, r, z0, z1, bt=0.002, bb=0.001):
    return G.transform(G.slab(G.rounded_rect(hx, hy, r), z0, z1, bt, bb), I3, (c[0], c[1], 0.0))


def tube(pts, r, nseg=12):
    """a round cable along a polyline (smoothed by Catmull-Rom)"""
    P = np.asarray(pts, float)
    Q = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)]
        for t in np.linspace(0, 1, 8, endpoint=False):
            Q.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    Q.append(P[-1])
    Q = np.asarray(Q)
    m = G.Mesh()
    rings = []
    up = np.array([0, 1.0, 0])
    s = 0.0
    for i in range(len(Q)):
        t = Q[min(i + 1, len(Q) - 1)] - Q[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        x = np.cross(up, t)
        x /= np.linalg.norm(x) + 1e-12
        y = np.cross(t, x)
        if i:
            s += np.linalg.norm(Q[i] - Q[i - 1])
        rings.append([(Q[i] + r * (math.cos(a) * x + math.sin(a) * y), math.cos(a) * x + math.sin(a) * y, (r * a, s))
                      for a in np.linspace(0, 2 * math.pi, nseg + 1)])
    for i in range(len(rings) - 1):
        for j in range(nseg):
            a, b, c, d = rings[i][j], rings[i][j + 1], rings[i + 1][j + 1], rings[i + 1][j]
            m.quad([a[0], d[0], c[0], b[0]], np.asarray([a[1], d[1], c[1], b[1]]), [a[2], d[2], c[2], b[2]])
    return m.arrays()


def prism(section, y0, y1):
    """a convex section [(x, z), ...] (counter-clockwise seen from -y) extruded from y0 to y1, flat shaded"""
    S = [np.array([x, 0, z]) for x, z in section]
    c = np.mean(S, 0)
    m = G.Mesh()
    n = len(S)
    for i in range(n):
        a, b = S[i], S[(i + 1) % n]
        e = b - a
        nn = np.array([e[2], 0, -e[0]])
        if np.dot(nn, (a + b) / 2 - c) < 0:
            nn = -nn
        le = np.linalg.norm(e)
        q = [a + [0, y0, 0], b + [0, y0, 0], b + [0, y1, 0], a + [0, y1, 0]]
        if np.dot(np.cross(q[1] - q[0], q[2] - q[0]), nn) < 0:
            q = [q[1], q[0], q[3], q[2]]
        m.quad(q, nn, [(0, y0), (le, y0), (le, y1), (0, y1)])
    for y, s in ((y0, -1), (y1, 1)):
        for i in range(1, n - 1):
            p = [S[0] + [0, y, 0], S[i] + [0, y, 0], S[i + 1] + [0, y, 0]]
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), [0, s, 0]) < 0:
                p = [p[0], p[2], p[1]]
            m.tri(p, np.array([0, s, 0.0]), [(q[0], q[2]) for q in p])
    return m.arrays()


def looks(w):
    A = "acg/"
    w.look("barrier", col="barrier_col.png", rgh=A + "Metal038_Roughness.jpg", nrm=A + "Metal038_NormalGL.jpg", metal=0.0, tile=0.9)
    w.look("rackbase", col="rackbase_col.png", rgh=A + "PaintedMetal004_Roughness.jpg", nrm=A + "PaintedMetal004_NormalGL.jpg", tile=0.8)
    for c in ("orange", "yellow", "teal", "green", "white", "blue"):
        w.look("paint_" + c, col="paint_%s_col.png" % c, rgh=A + "PaintedMetal004_Roughness.jpg", nrm=A + "PaintedMetal004_NormalGL.jpg",
               metal=A + "PaintedMetal004_Metalness.jpg", tile=0.7)
    w.look("hazard", col="hazard_col.png", rgh=A + "PaintedMetal004_Roughness.jpg", nrm=A + "PaintedMetal004_NormalGL.jpg", tile=0.25)
    w.look("floorpaint_yellow", col="floorpaint_yellow_col.png", rgh=0.62, tile=0.9)
    w.look("floorpaint_white", col="floorpaint_white_col.png", rgh=0.62, tile=0.9)
    w.look("galv", col="galv_col.png", rgh=A + "Metal038_Roughness.jpg", nrm=A + "Metal038_NormalGL.jpg", metal=0.9, tile=0.6)
    w.look("tread", col="tread_col.png", rgh=A + "DiamondPlate008C_Roughness.jpg", nrm=A + "DiamondPlate008C_NormalGL.jpg",
           metal=1.0, tile=0.45)
    w.look("steel_dark", col=A + "Metal046B_Color.jpg", rgh=A + "Metal046B_Roughness.jpg", nrm=A + "Metal046B_NormalGL.jpg", metal=1.0, tile=0.5)
    w.look("rubber", col=(0.022, 0.022, 0.024), rgh="lib/rubber_rgh.png", nrm="lib/rubber_nrm.png", tile=0.08)
    w.look("mat_blue", col="mat_blue_col.png", rgh=0.78, nrm="lib/rubber_nrm.png", tile=0.12)
    w.look("alu", col="lib/alu_rough_col.png", rgh="lib/alu_rough_rgh.png", nrm="lib/alu_rough_nrm.png", metal=1.0, tile=0.12)
    w.look("alu_brushed", col="lib/brushed_col.png", rgh="lib/brushed_rgh.png", nrm="lib/brushed_nrm.png", metal=1.0, tile=0.08)
    w.look("ano_black", col=(0.018, 0.019, 0.021), rgh="lib/ano_rgh.png", nrm="lib/ano_nrm.png", metal=1.0, tile=0.06)
    w.look("ano_fin", col=(0.10, 0.104, 0.11), rgh=0.38, metal=1.0, tile=0.06)
    w.look("ano_dark", col=(0.075, 0.078, 0.085), rgh="lib/ano_rgh.png", nrm="lib/ano_nrm.png", metal=1.0, tile=0.06)
    w.look("shell", col=(0.030, 0.031, 0.034), rgh=0.42, nrm="lib/peel_nrm.png", tile=0.05)
    w.look("plastic_grey", col=(0.42, 0.43, 0.44), rgh=0.45, nrm="lib/peel_nrm.png", tile=0.05)
    w.look("plastic_white", col=(0.80, 0.80, 0.78), rgh=0.4, nrm="lib/peel_nrm.png", tile=0.05)
    w.look("glass_dark", col=(0.004, 0.005, 0.008), rgh=0.03, clearcoat=1.0)
    w.look("lens", col=(0.006, 0.008, 0.016), rgh=0.02, clearcoat=1.0)
    w.look("laser_red", col=(0.20, 0.004, 0.004), rgh=0.04, clearcoat=1.0)
    w.look("steel_bolt", col=(0.55, 0.55, 0.56), rgh=0.3, metal=1.0)
    w.look("brass", col=(0.78, 0.57, 0.22), rgh=0.32, metal=1.0)
    w.look("led_green", col=(0.02, 0.3, 0.05), rgh=0.3, emissive=(0.3, 6.0, 0.8))
    w.look("pcb_green", col=(0.02, 0.16, 0.06), rgh=0.35)
    w.look("powder_grey", col="lib/powder_col.png", rgh="lib/powder_rgh.png", nrm="lib/powder_nrm.png", tile=0.2)


def barrier(w, parent, name, x0, x1, y0, y1, rack, hazard_lo, hazard_hi):
    """a wall box as a barrier run along its long axis; hazard_*: mark that end (a passage)"""
    H = 0.32
    along_y = (y1 - y0) > (x1 - x0)
    L, T = ((y1 - y0), (x1 - x0)) if along_y else ((x1 - x0), (y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    def place(m):
        return G.xform2(m, math.pi / 2 if along_y else 0.0, (cx, cy, 0))

    body = "rackbase" if rack else "barrier"
    w.mesh(parent, name + "_cap", place(box((0, 0, H - 0.014), (L / 2, T / 2, 0.014), 0.004)), "paint_orange")
    w.mesh(parent, name + "_kick", place(box((0, 0, 0.016), (L / 2 - 0.001, T / 2 - 0.001, 0.016), 0.002)), "rubber")
    nseg = max(1, int(round(L / 1.2)))
    gap = 0.004
    seg = (L - gap * (nseg - 1)) / nseg
    hz = 0.10
    for i in range(nseg):
        a = -L / 2 + i * (seg + gap)
        b = a + seg
        cuts = [(a, b, body)]
        if hazard_lo and i == 0 and seg > 2 * hz:
            cuts = [(a, a + hz, "hazard"), (a + hz + 0.002, b, body)]
        if hazard_hi and i == nseg - 1 and (cuts[-1][1] - cuts[-1][0]) > 2 * hz:
            la, lb, lk = cuts[-1]
            cuts[-1] = (la, lb - hz - 0.002, lk)
            cuts.append((lb - hz, lb, "hazard"))
        for a2, b2, lk in cuts:
            w.mesh(parent, name + "_b", place(box(((a2 + b2) / 2, 0, 0.032 + (H - 0.028 - 0.032) / 2),
                                                  ((b2 - a2) / 2, T / 2 - 0.004, (H - 0.028 - 0.032) / 2), 0.003)), lk)


def screws(c, hx, hy, z, r=0.004):
    return G.merge(*[cyl((c[0] + sx * hx, c[1] + sy * hy, z), (c[0] + sx * hx, c[1] + sy * hy, z + 0.0012), r, 0.0004, 16)
                     for sx in (-1, 1) for sy in (-1, 1)])


def fittings(w, parent, L):
    CLEAR, DEPTH, DECK_Z, BEVEL, TOE, RISE, COVER = .426, .068, .065, .33, .012, .082, .155
    for item in L["fittings"]:
        kind, x, lo, hi, length = item[:5]
        if len(item) > 6:                                           # the site's own duct clearance and guide rise (round 2)
            CLEAR, RISE = float(item[5]), float(item[6])
        wd, y = (hi - lo) / 2, (hi + lo) / 2
        if kind == "duct":
            top = CLEAR + DEPTH
            hy = wd + .045
            # the duct: a sheet body a little inside the box, a flange frame at each open end (the y ends), seams
            w.mesh(parent, "duct_body", box((x, y, (top + CLEAR) / 2), (length / 2 - 0.004, hy - 0.012, DEPTH / 2 - 0.004), 0.003), "galv")
            for s in (-1, 1):
                w.mesh(parent, "duct_flange", box((x, y + s * (hy - 0.006), (top + CLEAR) / 2), (length / 2, 0.006, DEPTH / 2), 0.0015), "galv")
            for k in (-1, 0, 1):                                    # standing seams across the top and bottom
                yy = y + k * hy * 0.5
                w.mesh(parent, "duct_seam", box((x, yy, (top + CLEAR) / 2), (length / 2 - 0.0035, 0.005, DEPTH / 2 - 0.003), 0.001), "galv")
            for rim in (lo - .025, hi + .025):                      # bracket shoes
                w.mesh(parent, "duct_shoe", box((x, rim, (.32 + CLEAR) / 2), (.075, .024, (CLEAR - .32) / 2), 0.003), "galv")
                w.mesh(parent, "duct_shoe_bolts", G.merge(*[cyl((x + sx * .055, rim - 0.0245 * sy, .36), (x + sx * .055, rim - 0.0238 * sy, .36), .006, 0.0003, 6)
                                                            for sx in (-1, 1) for sy in (-1, 1)]), "steel_bolt")
            cz = top
            cx = x
        else:
            top = DECK_Z
            w.mesh(parent, "deck_table", box((x, y, top / 2), (length / 2, wd, top / 2), 0.0015), "tread")
            for s in (-1, 1):
                sec = [(0., 0.), (s * BEVEL, 0.), (s * BEVEL, TOE), (0., top)]
                m = prism(sec, -wd, wd)
                w.mesh(parent, "deck_ramp", G.transform(m, I3, (x + s * length / 2, y, 0)), "tread")
            sec = [(-.305, top), (.305, top), (.249, top + RISE), (.061, top + RISE)]
            for (a, b) in ((-.050, .050), (-wd, -.200), (.200, wd)):
                w.mesh(parent, "deck_guide", G.transform(prism(sec, a, b), I3, (x, y, 0)), "paint_yellow")
            cz = top + RISE
            cx = x + COVER
        w.mesh(parent, kind + "_cover", box((cx, y, cz + .0007), (.080, .039, .0007), 0.0004), "paint_teal")
        w.mesh(parent, kind + "_cover_screws", screws((cx, y), .070, .030, cz + .0012, 0.0035), "steel_bolt")


def bollard(w, parent, x, y):
    w.mesh(parent, "bollard_flange", G.transform(G.cylinder((0, 0, 0), (0, 0, 0.012), 0.06, 40, True, 0.002), I3, (x, y, 0)), "steel_dark")
    w.mesh(parent, "bollard_bolts", G.merge(*[cyl((x + 0.05 * math.cos(a), y + 0.05 * math.sin(a), 0.012), (x + 0.05 * math.cos(a), y + 0.05 * math.sin(a), 0.017), 0.005, 0.0005, 6)
                                              for a in np.arange(4) * math.pi / 2 + math.pi / 4]), "steel_bolt")
    w.mesh(parent, "bollard_tube", G.transform(G.cylinder((0, 0, 0.012), (0, 0, 0.66), 0.048, 48, False, 0.0), I3, (x, y, 0)), "paint_yellow")
    for z in (0.45, 0.55):
        w.mesh(parent, "bollard_band", G.transform(G.cylinder((0, 0, z), (0, 0, z + 0.05), 0.0485, 48, False, 0.0), I3, (x, y, 0)), "rubber")
    w.mesh(parent, "bollard_collar", G.transform(G.cylinder((0, 0, 0.64), (0, 0, 0.68), 0.058, 48, True, 0.004), I3, (x, y, 0)), "steel_dark")
    R, zc = 0.105, 0.72
    prof = [(R * math.cos(a), zc + R * math.sin(a)) for a in np.linspace(-0.55, math.pi / 2, 28)]
    prof = [(0.0, prof[0][1])] + prof
    w.mesh(parent, "bollard_cap", G.transform(G.lathe(prof, 64), I3, (x, y, 0)), "paint_green")


def looks2(w):
    looks(w)
    w.look("lidar_window", col=(0.006, 0.006, 0.008), rgh=0.06, clearcoat=1.0)
    w.look("lidar_alu", col=(0.62, 0.63, 0.65), rgh=0.42, metal=1.0, nrm="lib/blast_nrm.png", tile=0.03)
    w.look("tube_black", col=(0.035, 0.036, 0.04), rgh="lib/powder_rgh.png", nrm="lib/powder_nrm.png", tile=0.05)
    w.look("webbing", col=(0.03, 0.03, 0.032), rgh=0.85, nrm="lib/rubber_nrm.png", tile=0.012)
    w.look("strap_magenta", col=(0.72, 0.06, 0.50), rgh=0.8, nrm="lib/rubber_nrm.png", tile=0.012)
    w.look("label_yellow", col=(0.93, 0.76, 0.05), rgh=0.45)
    w.look("label_black", col=(0.02, 0.02, 0.02), rgh=0.45)
    w.look("accent_blue", col=(0.04, 0.20, 0.62), rgh=0.35, metal=1.0)
    w.look("beacon_lens", col=(0.02, 0.12, 0.80), rgh=0.10, clearcoat=1.0, emissive=(0.0, 0.035, 0.42))
    w.look("beacon_core", col=(0.6, 0.7, 1.0), rgh=0.3, emissive=(2.0, 8.0, 40.0))
    w.look("antenna", col=(0.02, 0.02, 0.022), rgh=0.55)
    w.look("connector", col=(0.50, 0.51, 0.52), rgh=0.3, metal=1.0)
    w.look("prop_blur", col=(0.70, 0.72, 0.76), rgh=0.6, opacity=0.16)
    w.look("charger_orange", col="paint_orange_col.png", rgh="acg/PaintedMetal004_Roughness.jpg", tile=0.9)
    w.look("conduit", col="galv_col.png", rgh="acg/Metal038_Roughness.jpg", metal=0.9, tile=0.3)


def beacon(w, parent, x, y):
    w.mesh(parent, "beacon_flange", G.transform(G.cylinder((0, 0, 0), (0, 0, 0.012), 0.06, 40, True, 0.002), I3, (x, y, 0)), "steel_dark")
    w.mesh(parent, "beacon_bolts", G.merge(*[cyl((x + 0.05 * math.cos(a), y + 0.05 * math.sin(a), 0.012), (x + 0.05 * math.cos(a), y + 0.05 * math.sin(a), 0.017), 0.005, 0.0005, 6)
                                             for a in np.arange(4) * math.pi / 2 + math.pi / 4]), "steel_bolt")
    w.mesh(parent, "beacon_pole", G.transform(G.cylinder((0, 0, 0.012), (0, 0, 0.60), 0.030, 40, False, 0.0), I3, (x, y, 0)), "galv")
    w.mesh(parent, "beacon_box", G.transform(G.cylinder((0, 0, 0.26), (0, 0, 0.36), 0.052, 40, True, 0.004), I3, (x, y, 0)), "powder_grey")
    w.mesh(parent, "beacon_base", G.transform(G.cylinder((0, 0, 0.60), (0, 0, 0.665), 0.060, 48, True, 0.005), I3, (x, y, 0)), "ano_black")
    R, zc = 0.105, 0.72
    prof = [(R * math.cos(a), zc + R * math.sin(a)) for a in np.linspace(-0.55, math.pi / 2, 28)]
    w.mesh(parent, "beacon_dome", G.transform(G.lathe([(0.0, prof[0][1])] + prof, 64), I3, (x, y, 0)), "beacon_lens")
    w.mesh(parent, "beacon_ring", G.transform(G.cylinder((0, 0, 0.655), (0, 0, 0.672), 0.092, 48, True, 0.003), I3, (x, y, 0)), "alu_brushed")


def charger(w, parent, goal_y, bound):
    px, gy = 7.58, goal_y
    w.mesh(parent, "chg_base", box((px, gy, .006), (.17, .25, .006), 0.002), "steel_dark")
    w.mesh(parent, "chg_base_bolts", screws((px, gy), .15, .23, .012, 0.006), "steel_bolt")
    w.mesh(parent, "chg_post", box((px + .03, gy, .40), (.045, .09, .39), 0.006), "barrier")
    w.mesh(parent, "chg_body", rslab((px - .02, gy), .085, .20, .03, .62, 1.12, 0.012, 0.012), "plastic_white")
    w.mesh(parent, "chg_band", rslab((px - .02, gy), .0856, .2006, .03, .70, .74, 0.0, 0.0), "charger_orange")
    w.mesh(parent, "chg_fascia", box((px - .1058, gy, .93), (.0012, .15, .13), 0.0006), "glass_dark")
    w.mesh(parent, "chg_led", box((px - .1072, gy, .815), (.0008, .11, .006), 0.0004), "led_green")
    w.mesh(parent, "chg_label", box((px - .1058, gy - .12, .66), (.0008, .04, .03), 0.0003), "label_yellow")
    w.mesh(parent, "chg_vents", G.merge(*[box((px + .0655, gy, z), (.001, .14, .004), 0.0004) for z in np.arange(.70, 1.06, .03)]), "ano_black")
    w.mesh(parent, "chg_holster", cyl((px - .02, gy + .205, .86), (px - .02, gy + .235, .86), .035, 0.004, 28), "ano_black")
    w.mesh(parent, "chg_cable", tube([(px - .02, gy + .235, .84), (px - .03, gy + .30, .55), (px - .10, gy + .28, .22), (px - .22, gy + .14, .03),
                                      (px - .30, gy + .02, .012), (px - .36, gy - .05, .012)], 0.009), "rubber")
    # the conduit from the station along the floor to the hall's pole side
    w.mesh(parent, "chg_conduit", G.merge(cyl((px + .08, gy, .02), (px + .08, gy - 1.0, .02), .012, 0.0, 16),
                                          cyl((px + .08, gy - 1.0, .02), (px + 2.6, gy - 1.0, .02), .012, 0.0, 16)), "conduit")
    w.mesh(parent, "chg_clips", G.merge(*[box((px + .08 + k * .6, gy - 1.0, .014), (.02, .022, .014), 0.002) for k in range(1, 5)]), "steel_dark")


def lanes(w, parent, bound):
    """the yellow lane painted round the course"""
    b = bound
    x0, x1 = -1.14, 7.2
    o, t, z = 0.30, 0.04, 0.0012
    for (cx, cy, hx, hy) in (((x0 + x1) / 2, -b - o, (x1 - x0) / 2 + o + t, t), ((x0 + x1) / 2, b + o, (x1 - x0) / 2 + o + t, t),
                             (x0 - o, 0, t, b + o), (x1 + o, 0, t, b + o)):
        w.mesh(parent, "lane", box((cx, cy, z / 2), (hx, hy, z / 2), 0.0003), "floorpaint_yellow")


def pallet_feet(w, parent, rect):
    """the two foot brackets that keep the barrier pallet on its edge: a floor plate under it and an upright on each
    face (the task's collision geoms pallet_foot* and pallet_upright* are these boxes)"""
    x0, x1, y0, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for s in (-1, 1):
        w.mesh(parent, "pallet_foot", box((cx, cy + s * PALLET_FOOT_Y, 0.004), (PALLET_FOOT_X, 0.05, 0.004), 0.001), "steel_dark")
        for sx in (-1, 1):
            w.mesh(parent, "pallet_foot_up", box((cx + sx * PALLET_UPRIGHT_X, cy + s * PALLET_FOOT_Y, 0.04), (0.003, 0.05, 0.04), 0.001), "steel_dark")


PALLET_FOOT_X, PALLET_FOOT_Y, PALLET_UPRIGHT_X = 0.11, 0.30, 0.075


def pallet_matrix(rect):
    """the pack's pallet (x +/-0.6, y +/-0.4, z 0..0.144): its length up, its width along y, its thickness along x,
    centred in the rectangle"""
    x0, x1, y0, y1 = rect
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = (0, 0, 1), (0, 1, 0), (-1, 0, 0)
    M[:3, 3] = ((x0 + x1) / 2 + 0.072, (y0 + y1) / 2, 0.60)
    return M
