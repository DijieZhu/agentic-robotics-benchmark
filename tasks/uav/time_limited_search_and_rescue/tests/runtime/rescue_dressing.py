"""What stands in a site besides its structure, from the site alone: stock and furniture along walls, wall items,
ceiling lamps, paper on the floors, the receiving room's furniture, the yard round the building; and the drawn models
of the litter with its casualty and receiver and of the airframe. rescue_scene.build calls dress().

Where nothing may stand (the rule, stated in the task's instruction): lanes are the lines that join the centres of
two rooms through their doorway, the staging row's centre line, the ways to the inlet, the outlet, the turning
pocket, both receiving sides, the tool stands and the launch point; a hoist well is a point on both storeys. Every
footprint keeps LANE_HALF from every lane of its storey; anything taller than LOW_TOP also keeps TURN_R from every
room centre (a litter carried under the aircraft turns there).

Every standing thing is one box for the contact check (kind "prop") and its model's meshes for the cameras. The
models are the render bundle's library (RESCUE_RENDER_DIR/look, library.json)."""
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

DECK, SLAB = 2.60, 0.22
LANE_HALF, LOW_TOP, TURN_R = 0.72, 0.80, 1.25
TNARROW = [[("drawer_1", -0.55, 0.24, 0), ("drawer_2", 0.10, 0.24, 0), ("cardboard_1", 0.70, 0.20, 10)],
           [("cardboard_2", -0.45, 0.25, 15), ("cardboard_3", 0.05, 0.22, -10), ("box_small", 0.50, 0.20, 5), ("cardboard_1", 0.85, 0.20, -20)],
           [("drawer_2", -0.30, 0.24, 0), ("box_small", 0.35, 0.20, 12), ("cardboard_3", 0.75, 0.22, 30)],
           [("drawer_1", 0.0, 0.24, 0), ("cardboard_2", 0.62, 0.25, -8), ("cardboard_1", -0.62, 0.20, 25)]]
WNARROW = [("Medical_Bedside_Table_01", "Chair_01"), ("Wooden_Box_01", "Medical_Container_01"), ("Chair_02", "Medical_Bedside_Table_01"), ("Barrel_01", "Bucket_01")]


def library_dir():
    return os.path.join(os.environ.get("RESCUE_RENDER_DIR", "/opt/task-assets/payload/render"), "look")


def _bnd(ax, i):
    at = lambda k: 2 * ax[0] - ax[1] if k < 0 else (2 * ax[-1] - ax[-2] if k >= len(ax) else ax[k])  # noqa: E731
    return (at(i - 1) + ax[i]) / 2, (at(i + 1) + ax[i]) / 2


class Rule(object):
    """The lanes of a site and the test of a footprint against them."""

    def __init__(self, layout):
        cols, rows = [float(v) for v in layout.columns], [float(v) for v in layout.rows]
        self.cols, self.rows = cols, rows
        self.X0, self.X1 = _bnd(cols, 0)[0], _bnd(cols, 4)[1]
        self.YS, self.Y1 = _bnd(rows, 0)[0], _bnd(rows, 4)[1]
        self.YI = float(layout.inlet_y)
        self.pcx, self.pcy = (float(v) for v in layout.pocket)
        pad = [float(v) for v in layout.receiving_pad]
        self.pad = pad
        seg, cen = {0: [], 1: []}, {0: [], 1: []}
        self.halls = {tuple(sorted((tuple(int(v) for v in a), tuple(int(v) for v in b)))) for a, b in layout.halls}
        for (c0, r0, l0), (c1, r1, l1) in sorted(self.halls):
            p0, p1 = (cols[c0], rows[r0]), (cols[c1], rows[r1])
            if l0 == l1:
                seg[l0].append((p0, p1))
                cen[l0] += [p0, p1]
            else:
                for l_ in (l0, l1):
                    seg[l_].append((p0, p0))
                    cen[l_].append(p0)
        seg[0].append(((self.X0 + 0.6, rows[0]), (self.X1 - 0.6, rows[0])))
        seg[0].append(((cols[layout.inlet], rows[0]), (cols[layout.inlet], rows[1])))
        seg[0].append(((cols[layout.outlet], rows[0]), (self.pcx, self.pcy)))
        seg[0].append(((self.pcx - 3.9, self.pcy), (self.pcx + 3.9, self.pcy)))
        seg[0].append(((self.pcx, self.pcy), tuple(pad)))
        seg[0].append(((self.pcx, self.pcy), (2 * self.pcx - pad[0], pad[1])))
        for slot in (0, 1):
            dk = layout.dock(slot)
            seg[0].append(((float(dk[0]), rows[0]), (float(dk[0]), float(dk[1]))))
        seg[0].append(((float(layout.launch[0]), float(layout.launch[1])), (float(layout.launch[0]), rows[0])))
        # the litter where it lies (its length is a lane of its own) and the way from it to its room's centre; the
        # litter is turned over a receiving pad and in the turning pocket: those are turning points like a room's centre
        px, py, pyaw = (float(v) for v in layout.pod)
        tl = int(layout.target[2])
        ux, uy = -math.sin(pyaw) * 0.9, math.cos(pyaw) * 0.9
        seg[tl].append(((px - ux, py - uy), (px + ux, py + uy)))
        seg[tl].append(((px, py), (cols[int(layout.target[0])], rows[int(layout.target[1])])))
        cen[0] += [tuple(pad), (2 * self.pcx - pad[0], pad[1]), (self.pcx, self.pcy)]
        self.wells = {(int(w[0]), 4) for w in layout.wells}
        for l_ in (0, 1):
            for c_, r_ in sorted(self.wells):
                seg[l_].append(((cols[c_], rows[r_]), (cols[c_], rows[r_])))
        self.A = {l_: np.array([s[0] for s in seg[l_]], float) for l_ in (0, 1)}
        self.B = {l_: np.array([s[1] for s in seg[l_]], float) for l_ in (0, 1)}
        self.C = {l_: np.array(sorted(set(cen[l_])), float).reshape(-1, 2) for l_ in (0, 1)}

    def door(self, c0, c1):
        return tuple(sorted((tuple(c0), tuple(c1)))) in self.halls

    def lane_dist(self, pts, lvl):
        A_, B_ = self.A[lvl], self.B[lvl]
        ab = B_ - A_
        den = np.maximum((ab * ab).sum(1), 1e-12)
        t = np.clip(((pts[:, None, :] - A_[None]) * ab[None]).sum(2) / den[None], 0.0, 1.0)
        q = A_[None] + t[..., None] * ab[None]
        return float(np.sqrt(((pts[:, None, :] - q) ** 2).sum(2)).min())

    def free(self, x, y, yaw_deg, lx, ly, lvl, top=0.0):
        """how far the footprint (lx by ly, turned by yaw) is outside what must stay free; negative: it may not stand"""
        c_, s_ = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        loc = np.array([(i * lx / 4, j * ly / 4) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)])
        pts = np.stack([x + c_ * loc[:, 0] - s_ * loc[:, 1], y + s_ * loc[:, 0] + c_ * loc[:, 1]], 1)
        g = self.lane_dist(pts, lvl) - LANE_HALF
        if top > LOW_TOP and len(self.C[lvl]):
            g = min(g, float(np.sqrt(((pts[:, None, :] - self.C[lvl][None]) ** 2).sum(2)).min()) - TURN_R)
        return g

    def clear(self, x, y, lvl, rad):
        return self.lane_dist(np.array([[x, y]], float), lvl) - rad - LANE_HALF

    def clear_disc(self, x, y, rad):
        return self.lane_dist(np.array([[x, y]], float), 0) - rad - 0.32


def instances(layout, lib, signs=()):
    """[(asset, x, y, z, yaw in degrees)] of the library's models that stand in the site, [(lamp box centre, half)],
    [(paper corners 4 x 3, dirty)]"""
    R = Rule(layout)
    cols, rows, wells = R.cols, R.rows, R.wells
    out = []

    def size(nm):
        a = lib["assets"][nm]
        if "size" in a:
            return a["size"]
        return [a["hi"][k] - a["lo"][k] for k in range(3)]
    cells = [(cx_, ry, lvl) for lvl in (0, 1) for cx_ in range(5) for ry in range(1, 5) if not (lvl == 1 and (cx_, ry) in wells)]
    wards = []
    for cx_, ry, lvl in cells + [(cx_, 0, 0) for cx_ in range(5)]:
        if (cx_, ry) in wells:
            continue
        ccx, ccy = cols[cx_], rows[ry]
        hx, hy = (_bnd(cols, cx_)[1] - _bnd(cols, cx_)[0]) / 2 - 0.09, (_bnd(rows, ry)[1] - _bnd(rows, ry)[0]) / 2 - 0.09
        sides = []
        for wname, nb, ax_, sg in (("W", (cx_ - 1, ry), 0, -1), ("E", (cx_ + 1, ry), 0, 1), ("S", (cx_, ry - 1), 1, -1), ("N", (cx_, ry + 1), 1, 1)):
            if 0 <= nb[0] < 5 and 0 <= nb[1] < 5 and (R.door((cx_, ry, lvl), (*nb, lvl)) or (ry == 0 or nb[1] == 0)):
                continue
            if ry == 0 and wname == "N":
                continue
            if ry == 0 and wname == "S" and cx_ == layout.outlet:
                continue
            sides.append((wname, ax_, sg))
        hsh = int(hashlib.md5(("%d_%d_%d" % (cx_, ry, lvl)).encode()).hexdigest()[:6], 16)
        rs = np.random.default_rng(hsh)
        rs.shuffle(sides)
        if lvl == 1:                                    # the upper storey's rooms are wards
            for wname, ax_, sg in sides[:2]:
                wards.append((ccx, ccy, hx, hy, ax_, sg, hsh))
            continue
        for wname, ax_, sg in sides[:3]:
            tp = TNARROW[int(rs.integers(len(TNARROW)))]
            for nm, along, dist, yaw in tp:
                lx, ly, lz = size(nm)
                along = along + rs.uniform(-0.12, 0.12)
                if ax_ == 0:
                    x, y, yw = ccx + sg * (hx - dist), ccy + along, 90 + yaw
                else:
                    x, y, yw = ccx + along, ccy + sg * (hy - dist), yaw
                if R.free(x, y, yw, lx, ly, lvl, lz) < 0.0:
                    continue
                out.append((nm, round(x, 3), round(y, 3), round(lvl * DECK, 3), round(float(yw), 1)))
    for nm, x, y, yw, z in (("pallet_truck", R.X0 + 1.2, R.YS + 0.9, 20, 0.0), ("pallet_b", R.X0 + 2.6, R.YS + 0.75, 0, 0.0), ("stack_box_2", R.X0 + 2.6, R.YS + 0.75, 10, 0.144)):
        if R.clear(x, y, 0, 0.6) > 0:
            out.append((nm, round(x, 3), round(y, 3), z, float(yw)))
    # the receiving room (on the open station's side): a gurney, a bedside table with a container, a chair, a
    # poster and a medicine chest
    side = -1.0 if R.pad[0] < R.pcx else 1.0

    def rx(dx):
        """x at dx from the receiving room's outer wall towards the pocket"""
        return R.pcx + side * (5.45 - dx)
    ps = R.pcy - 2.30
    ry0 = ps + 0.08
    table_top = size("Medical_Bedside_Table_01")[2]
    stands = set()
    for nm, dx, y, zz, yw, on in (("Hospital_Gurney_01", 2.60, ps + 1.20, 0.0, 90.0, None), ("Medical_Bedside_Table_01", 0.42, ry0 + 1.60, 0.0, 90.0, None),
                                 ("Medical_Container_01", 0.42, ry0 + 2.20, 0.0, 78.0, None), ("Medical_Container_01", 0.44, ry0 + 1.60, table_top, 82.0, "Medical_Bedside_Table_01"),
                                 ("Chair_01", 0.48, ry0 + 0.75, 0.0, -90.0, None), ("Poster_01", 2.33, ry0 + 0.012, 1.25, 90.0, None),
                                 ("Medicine_Chest_Metal", 0.19, ry0 + 3.0, 1.30, 0.0, None)):
        x = rx(dx)
        yw = yw if side < 0 else 180.0 - yw
        lx, ly, lz = size(nm)
        flat = zz > 1.0                                 # on the wall
        if on is not None and on not in stands:
            continue
        if R.free(x, y, yw, 0.25 if flat else lx, 0.8 if flat else ly, 0, 0.0 if flat else zz + lz) < 0.0:
            continue
        stands.add(nm)
        out.append((nm, round(x, 3), round(y, 3), round(zz, 3), yw))
    # the wards
    for k, (ccx, ccy, hx, hy, ax_, sg, hsh) in enumerate(wards):
        along = -0.45
        for nm in WNARROW[(hsh + k) % len(WNARROW)]:
            lx, ly, lz = size(nm)
            yaw = 0.0
            if ly > lx:                                 # the long side along the wall
                lx, ly, yaw = ly, lx, 90.0
            if lx > 2 * (hy if ax_ == 0 else hx) - 0.3:
                continue
            al = along + lx / 2
            if ax_ == 0:
                x, y, yw = ccx + sg * (hx - ly / 2 - 0.06), ccy + al - 0.4, yaw + 90.0
            else:
                x, y, yw = ccx + al - 0.4, ccy + sg * (hy - ly / 2 - 0.06), yaw
            along += lx + 0.18
            if R.free(x, y, yw, lx, ly, 1, lz) < 0.0:
                continue
            out.append((nm, round(x, 3), round(y, 3), DECK, yw))
    # posters and medicine chests on inner partitions without a doorway
    for cx_, ry, lvl in cells:
        hsh = int(hashlib.md5(("w%d_%d_%d" % (cx_, ry, lvl)).encode()).hexdigest()[:6], 16)
        if hsh % 3 == 0:
            continue
        ccx, ccy = cols[cx_], rows[ry]
        hx, hy = (_bnd(cols, cx_)[1] - _bnd(cols, cx_)[0]) / 2 - 0.078, (_bnd(rows, ry)[1] - _bnd(rows, ry)[0]) / 2 - 0.078
        cand = []
        for nb, ax_, sg in (((cx_ - 1, ry), 0, -1), ((cx_ + 1, ry), 0, 1), ((cx_, ry - 1), 1, -1), ((cx_, ry + 1), 1, 1)):
            inside = 0 <= nb[0] < 5 and 1 <= nb[1] < 5
            if inside and not R.door((cx_, ry, lvl), (*nb, lvl)) and (cx_, ry) not in wells and nb not in wells:
                cand.append((ax_, sg))
        if not cand:
            continue
        ax_, sg = cand[hsh % len(cand)]
        nm = ("Poster_01", "Poster_03", "Medicine_Chest_Metal")[hsh % 3]
        off = ((hsh >> 4) % 100) / 100.0 * 1.2 - 0.6
        d_ = 0.012 if nm.startswith("Poster") else 0.11
        if ax_ == 0:
            x, y, yw = ccx + sg * (hx - d_), ccy + off, 0.0 if sg < 0 else 180.0
        else:
            x, y, yw = ccx + off, ccy + sg * (hy - d_), 90.0 if sg < 0 else -90.0
        if R.free(x, y, yw, 0.25, 0.8, lvl) < 0.0:
            continue
        out.append((nm, round(x, 3), round(y, 3), round(lvl * DECK + (1.05 if nm.startswith("Poster") else 1.25), 3), yw))
    # ceiling lamps
    lamps = []
    for lvl in (0, 1):
        for cx_ in range(5):
            for ry in range(1, 5):
                if lvl == 0 and (cx_, ry) in wells:
                    continue
                lamps.append(((cols[cx_], rows[ry], (lvl + 1) * DECK - SLAB - 0.022), (0.07, 0.62, 0.022) if (cx_ + ry) % 2 else (0.62, 0.07, 0.022)))
    for cx_ in range(5):
        lamps.append(((cols[cx_], (R.YS + R.YI) / 2, DECK - SLAB - 0.022), (0.62, 0.07, 0.022)))
    for dx, dy in ((0, 0.9), (0, -1.2), (-3.7, 0), (3.7, 0)):
        lamps.append(((R.pcx + dx, R.pcy + dy, DECK - SLAB - 0.022), (0.07, 0.62, 0.022)))
    # paper on the floors
    rng = np.random.default_rng(39)
    paper = []
    for cx_, ry, lvl in cells + [(cx_, 0, 0) for cx_ in range(5)]:
        for _ in range(int(rng.integers(2, 7))):
            x, y = cols[cx_] + rng.uniform(-1.15, 1.15), rows[ry] + rng.uniform(-1.15, 1.15)
            zf = lvl * DECK + 0.003 + rng.uniform(0, 0.002)
            if any(abs(zf - sz) < 0.5 and math.hypot(x - sx, y - sy) < 0.55 for sx, sy, sz in signs):
                continue
            ang = rng.uniform(0, math.pi)
            w_, h_ = (0.105, 0.1485) if rng.random() < 0.8 else (0.15, 0.21)
            paper.append((x, y, zf, math.degrees(ang), w_, h_, bool(rng.random() >= 0.6)))
    return out, lamps, paper


def _mesh(asset, name, V, F):
    ET.SubElement(asset, "mesh", name=name, inertia="shell", vertex=" ".join("%.5f" % v for v in np.asarray(V).reshape(-1)),
                  face=" ".join(str(int(i)) for i in np.asarray(F).reshape(-1)))


def _annulus(r0, r1, z0, z1, n=48):
    V, F = [], []
    for j in range(n):
        a = 2 * math.pi * j / n
        c, s = math.cos(a), math.sin(a)
        V += [(r0 * c, r0 * s, z0), (r1 * c, r1 * s, z0), (r1 * c, r1 * s, z1), (r0 * c, r0 * s, z1)]
    for j in range(n):
        a, b = 4 * j, 4 * ((j + 1) % n)
        F += [(a + 1, b + 1, b + 2), (a + 1, b + 2, a + 2),          # outer
              (b, a, a + 3), (b, a + 3, b + 3),                      # inner
              (a + 3, a + 2, b + 2), (a + 3, b + 2, b + 3),          # top
              (a, b, b + 1), (a, b + 1, a + 1)]                      # underside
    return V, F


def _disc_post(r, z0, z1, n=24):
    """a cylinder from z0 to z1"""
    V = [(0.0, 0.0, z0), (0.0, 0.0, z1)]
    F = []
    for j in range(n):
        a = 2 * math.pi * j / n
        V += [(r * math.cos(a), r * math.sin(a), z0), (r * math.cos(a), r * math.sin(a), z1)]
    for j in range(n):
        a, b = 2 + 2 * j, 2 + 2 * ((j + 1) % n)
        F += [(a, b, b + 1), (a, b + 1, a + 1), (1, a + 1, b + 1), (0, b, a)]
    return V, F


def dress(root, layout, obstacles):
    """Adds the site's dressing to the model: drawn meshes, one hidden box per standing thing (appended to
    `obstacles` as kind "prop"), the drawn litter, casualty, receiver and airframe (the task's own geoms of those
    bodies stay as the contact pieces and are not drawn)."""
    folder = library_dir()
    lib = json.load(open(os.path.join(folder, "library.json")))
    asset = root.find("asset")
    w = root.find("worldbody")
    ET.SubElement(root, "statistic", extent="32", center="0 2.75 3.1")   # the yard must not move the cameras' near plane
    layout_signs = []
    for g in w.iter("geom"):
        if g.get("type") == "plane" and g.get("name") != "floor":
            p = [float(v) for v in g.get("pos", "0 0 0").split()]
            layout_signs.append((p[0], p[1], p[2]))
    mats = {}

    def material(row):
        key = row.get("tex") or "c_%02x%02x%02x" % tuple(int(255 * c + 0.5) for c in row["rgb"])
        if key not in mats:
            name = "look_m%d" % len(mats)
            if "tex" in row:
                ET.SubElement(asset, "texture", name="look_t%d" % len(mats), type="2d", file=os.path.join(folder, row["tex"]))
                ET.SubElement(asset, "material", name=name, texture="look_t%d" % len(mats), rgba="1 1 1 1")
            else:
                ET.SubElement(asset, "material", name=name, rgba="%g %g %g 1" % tuple(row["rgb"]))
            mats[key] = name
        return mats[key]
    meshes = set()

    def draw(parent, prefix, rows, pos=None, yaw=None):
        for k, row in enumerate(rows):
            mn = "look_" + os.path.splitext(row["obj"])[0]
            if mn not in meshes:
                ET.SubElement(asset, "mesh", name=mn, file=os.path.join(folder, row["obj"]), inertia="shell")
                meshes.add(mn)
            at = dict(name="%s_%d" % (prefix, k), type="mesh", mesh=mn, material=material(row), contype="0", conaffinity="0", mass="0", group="2")
            if pos is not None:
                at["pos"] = "%g %g %g" % tuple(pos)
                at["euler"] = "0 0 %g" % yaw
            ET.SubElement(parent, "geom", at)
    bodies = {b.get("name"): b for b in w.iter("body")}
    for name in ("pod", "uav"):                         # the task's geoms of these bodies: contact pieces, not drawn
        for g in bodies[name].findall("geom"):
            g.set("rgba", " ".join(g.get("rgba", ".5 .5 .5 1").split()[:3] + ["0"]))
    draw(bodies["pod"], "look_pod", lib["fixed"]["pod"])
    draw(bodies["uav"], "look_uav", lib["fixed"]["uav"])
    # the receiver's ring is the site's flange, closed at the bottom, on a post that stands on the bed
    fr = float(layout.flange)
    fz0, fz1, bed = 0.164 - 0.023, 0.164 + 0.023, -0.066
    bore = min(fr - 0.010, 0.081)
    ring, steel = "%g %g %g 1" % tuple(lib["ring_rgb"]), "%g %g %g 1" % tuple(lib["steel_rgb"])
    for nm, (V, F), rgba in (("ring", _annulus(bore, fr, fz0, fz1), ring), ("cup", _disc_post(bore + 0.001, fz0, fz0 + 0.004), ring),
                             ("post", _disc_post(0.016 if fr < 0.06 else 0.022, bed, fz0), steel)):
        _mesh(asset, "look_receiver_" + nm, V, F)
        ET.SubElement(bodies["pod"], "geom", name="look_receiver_" + nm, type="mesh", mesh="look_receiver_" + nm, rgba=rgba,
                      contype="0", conaffinity="0", mass="0", group="2")
    ET.SubElement(bodies["pod"], "geom", name="look_receiver_keel", type="box", pos="0 0 %g" % (bed + 0.004), size="0.035 0.060 0.004", rgba=steel,
                  contype="0", conaffinity="0", mass="0", group="2")
    # the yard (mirrored with the building)
    for k, row in enumerate(lib["fixed"]["yard_mirror" if float(layout.pocket[0]) < 0 else "yard"]):
        draw(w, "look_yard_%d" % k, [row], pos=(0.0, 0.0, -0.03), yaw=0.0)
    inst, lamps, paper = instances(layout, lib, layout_signs)
    for i, (nm, x, y, z, yaw) in enumerate(inst):
        a = lib["assets"][nm]
        draw(w, "look_static_%d" % i, a["parts"], pos=(x, y, z), yaw=yaw)
        lo, hi = np.array(a["lo"]), np.array(a["hi"])
        c, h = (lo + hi) / 2, (hi - lo) / 2
        cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        name = "prop_%d" % i
        ET.SubElement(w, "geom", name=name, type="box", pos="%g %g %g" % (x + cy * c[0] - sy * c[1], y + sy * c[0] + cy * c[1], z + c[2]),
                      size="%g %g %g" % tuple(np.maximum(h, 0.004)), euler="0 0 %g" % yaw, rgba="1 0 0 0", contype="0", conaffinity="0", mass="0", group="4")
        obstacles.append(("prop", name))
    for i, (c, h) in enumerate(lamps):
        ET.SubElement(w, "geom", name="look_lamp_%d" % i, type="box", pos="%g %g %g" % tuple(c), size="%g %g %g" % tuple(h), rgba="0.8 0.8 0.78 1",
                      contype="0", conaffinity="0", mass="0", group="2")
    for i, (x, y, z, yaw, w_, h_, dirty) in enumerate(paper):
        ET.SubElement(w, "geom", name="look_paper_%d" % i, type="box", pos="%g %g %g" % (x, y, z), size="%g %g 0.0004" % (w_, h_), euler="0 0 %g" % yaw,
                      rgba="0.56 0.52 0.45 1" if dirty else "0.80 0.78 0.72 1", contype="0", conaffinity="0", mass="0", group="2")
    return len(inst)
