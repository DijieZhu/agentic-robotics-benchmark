"""The two trays of the hand assembly cell (workcell equipment, light machined aluminium) and where each unit lies in
its nest.

  digit tray, 0.36 x 0.24 m: three finger nests at 70 mm pitch and the thumb nest on a riser 110 mm high;
  parts tray, 0.24 x 0.20 m: a pocket for the electronics stack and 12 mm stands for the upper shell, the dorsal
  cover, the power hub and the press-fit cap.

Tray frame: origin on the plate's top face at its centre, x along the long side, z up. Every unit lies dorsal side up
with the palm frame's axes along the tray's (a finger's distal direction = the tray's +y), its footprint box centred
on the nest's point; a flat part may lie turned 180 deg about the vertical (layout.py draws it).

  finger nest  a block under the base servo (its lowest face, 0.5 mm side walls 1 mm off its sides, an end stop behind
               it), a small block under the ab/ad yoke's lower half next to the axis, and a fan deck under the chain's
               tip pad from 84 to 135 mm off the ab/ad axis over +-13 deg (the chain is drawn within +-10 deg), its top
               1.0 mm under the tip's lowest point: the chain lies on it as a mounted finger lies on the fixture's
               deck. Nothing stands beside the pitch yoke, where the gripper's fingers go down.
  thumb nest   a riser under the base servo and the base yoke (its top 0.1 mm under the yoke's lowest point), and a
               fan support under the thumb's distal links (the MP and IP links) over the drawn side angle
               +-8 deg (+3 deg margin), clear of the corner block's -y half and of everything below it, where the
               gripper's lower finger goes in under the corner block from the wrist side.
  stack pocket a 12 mm stand under the stack's centre with two 4 mm lips along the two faces the pads do not use.
  stands       a 12 mm block under the cap; four or five 6 x 6 mm posts under the top, the cover and the hub, each
               post's top 0.1 mm under the part's lower surface there and clear of the part's screws, so that the part
               lies level on them (the fingertips can go below the part's lowest face beside them).

    trays_of(cfg)                       {tray: {"plate": [x, y, t] m, "nests": [(nest, kind, x, y)]}}
    tray_geoms(cfg, tray)               [(name, "box", centre, half, quat) | (name, "prism", vertices)] in the tray frame
    nest_unit_T(kit, cfg, tray, nest, kind, turned=False)   the unit's root 4x4 (m) in the tray frame at rest
"""
import copy
import math

import numpy as np

DEFAULT_TRAY = {
    "trays": {
        "digit_tray": {"plate": [0.36, 0.24, 0.015],
                       "nests": [["nest_f1", "FM", -0.1355, 0.0], ["nest_f2", "FM", -0.0655, 0.0], ["nest_f3", "FM", 0.0045, 0.0],
                                 ["nest_thumb", "TM", 0.110, 0.064]]},
        "parts_tray": {"plate": [0.24, 0.20, 0.015],
                       # the cover and the top on the row away from the robot (the tray's -y), the stack, the hub and
                       # the cap on the near row: the cover's only grasp is a pinch on its wrist-end wall, which the
                       # cover's turn brings towards the robot (64 of 137 turned covers had no grasp on the near row)
                       "nests": [["nest_cover", "PC", -0.055, -0.045], ["nest_top", "PT", 0.065, -0.050], ["nest_stack", "BS", -0.080, 0.060],
                                 ["nest_hub", "HB", -0.008, 0.060], ["nest_cap", "CC", 0.078, 0.055]]},
    },
    # the finger nest, in the finger module's unit frame (mm: origin on the ab/ad axis, P's axes)
    "finger": {"base_block": {"x": [-9.5, 9.5], "y": [-25.5, -11.0]}, "wall_gap_mm": 1.0, "wall_t_mm": 4.0, "wall_h_mm": 8.0,
               "stop_gap_mm": 0.5, "yoke_pad": {"x": [-6.0, 6.0], "y": [11.0, 15.0], "below_mm": 0.3},
               "deck": {"r_mm": [84.0, 135.0], "half_deg": 13.0, "below_tip_mm": 1.0, "seg_deg": 6.5}, "t_mm": 3.0},
    # the thumb nest, in the thumb module's unit frame (mm: origin on the roll axis); the riser 110 mm high (the
    # thumb's lowest point 110 mm above the tray's top)
    "thumb": {"riser_mm": 110.0, "yoke_block": {"x": [-27.0, 9.0], "y": [-6.0, 26.0]}, "below_mm": 0.1,
              "distal": {"side_axis_xy_mm": [-21.866646, 6.256908], "margin_deg": 11.0, "y_min_mm": 24.0, "low_band_mm": 2.0,
                         "below_mm": 0.1},
              "post_half_mm": 6.0},
    # the stands (12 mm) and the stack's pocket, in the part's unit frame (mm)
    "stand_h_mm": 12.0,
    # (the cover and the cap rest on their rims: their stands span the whole footprint; the fingertips of their
    # grasps stay 1-13 mm above the lowest face)
    "stands": {"PT": {"x": [-16.0, 16.0], "y": [-20.0, 20.0]}, "PC": {"x": [-46.0, 46.0], "y": [-46.0, 46.0]},
               "HB": {"x": [-20.0, 20.0], "y": [-30.0, 30.0]}, "CC": {"x": [-31.0, 31.0], "y": [-33.0, 33.0]},
               "BS": {"u_half": 13.0, "w_half": 19.0}},
    # parts whose lowest faces are not one plane rest on four posts instead (6 x 6 mm, inside the footprint, clear of
    # the gripper's fingers), each post's top 0.1 mm under the part's own lower surface there: the part lies level,
    # as on its seat (unit frame, mm)
    # the cover is a shell open below: its skirt reaches z -13.6 on its +x side and at one wrist-end corner, the
    # rest of its lower edge stands 15 mm higher; two posts under the skirt, two taller ones on the thumb side
    # (every post stands clear of the part's TRUE solids, its drawn-only screws included: the true-solid
    # clipping audit found the top's 12 mm stand 2.5-9.3 mm3 into four of its screws' shanks and two of the cover's
    # posts 7.3 and 17.5 mm3 into a screw and into the cover's own skirt, which lies up to 0.9 mm below its collision
    # pieces there; the posts below are at spots where the post meets no true solid and stands at least 1.5 mm in plan
    # from every screw; the top lies on five posts instead of its stand)
    "posts": {"PC": [[33.0, -30.0], [39.0, 1.0], [-36.0, 16.0], [-32.0, -14.0], [-22.0, -40.0]],
              "PT": [[-37.0, -1.0], [-16.0, 21.0], [17.0, 21.0], [31.0, 5.0], [-24.0, -27.0]],
              "HB": [[-12.0, -20.0], [12.0, -20.0], [-12.0, 22.0], [12.0, 22.0]],
              # the stack's four at (u, w) = (+-8, +-13) mm of its turned frame (u = the pads' closing direction)
              "BS": [[7.3, 13.4], [15.3, -0.5], [-15.3, 0.5], [-7.3, -13.4]]},
    "post_half_mm": 3.0, "post_below_mm": 0.1,
    # the stack's pocket: two lips 4 mm above the stand's top, 1.5 mm off the stack's faces along w (the direction
    # across the faces the pads do not use: w = z x u, u = the grasp's closing direction (-0.4973, 0.8676) in P)
    "stack_pocket": {"u": [-0.4973, 0.8676], "lip_h_mm": 4.0, "lip_t_mm": 3.0, "gap_mm": 1.5, "lip_len_mm": 24.0},
    "nest_contact": {"solref": [0.002, 1.0], "priority": 1},
}


def merged(cfg=None):
    out = copy.deepcopy(DEFAULT_TRAY)
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def trays_of(cfg=None):
    c = merged(cfg)
    return {tn: {"plate": [float(v) for v in t["plate"]], "nests": [(str(n[0]), str(n[1]), float(n[2]), float(n[3])) for n in t["nests"]]}
            for tn, t in sorted(c["trays"].items())}


def _box(name, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return (name, "box", (0.5 * (lo + hi)).tolist(), (0.5 * (hi - lo)).tolist(), [1.0, 0.0, 0.0, 0.0])


def _lower_surface(hulls_eq, x, y):
    """the lowest z of a union of convex hulls (scipy ConvexHull equations) on the vertical line through (x, y), or
    None"""
    best = None
    for eq in hulls_eq:
        n, dd = eq[:, :3], eq[:, 3]
        rhs = -dd - n[:, 0] * x - n[:, 1] * y
        lo, hi = -1e9, 1e9
        dn = n[:, 2] < -1e-12
        up = n[:, 2] > 1e-12
        flat = ~(dn | up)
        if np.any(rhs[flat] < -1e-9):
            continue
        if dn.any():
            lo = float(np.max(rhs[dn] / n[dn, 2]))
        if up.any():
            hi = float(np.min(rhs[up] / n[up, 2]))
        if lo <= hi and (best is None or lo < best):
            best = lo
    return best


class UnitGeom(object):
    """the collision pieces of a kind in its unit frame (mm) at the start pose, per link"""

    def __init__(self, kit, kind):
        self.kind = kind
        self.by_link = {}
        for link, V, _src in kit.collision_hulls(kind):
            self.by_link.setdefault(link, []).append(np.asarray(V, float))
        self.all = np.vstack([v for vs in self.by_link.values() for v in vs])

    def hull_equations(self):
        if getattr(self, "_eq", None) is None:
            from scipy.spatial import ConvexHull
            self._eq = [ConvexHull(V).equations for vs in self.by_link.values() for V in vs if len(V) >= 4]
        return self._eq

    def pts(self, links=None):
        if links is None:
            return self.all
        return np.vstack([v for l in links for v in self.by_link[l]])

    def box(self):
        return self.all.min(0), self.all.max(0)


def _sector_prism(ax_xy, r0, r1, b0, b1, z0, z1, deg_dir):
    """a convex prism under an annular sector (mm), angles b in deg measured by deg_dir(b) -> unit 2-vector"""
    pts = []
    for r in (r0, r1):
        for b in (b0, b1):
            xy = np.asarray(ax_xy, float) + r * deg_dir(b)
            pts += [[xy[0], xy[1], z0], [xy[0], xy[1], z1]]
    xy = np.asarray(ax_xy, float) + r1 / math.cos(math.radians(0.5 * abs(b1 - b0))) * deg_dir(0.5 * (b0 + b1))
    pts += [[xy[0], xy[1], z0], [xy[0], xy[1], z1]]
    return np.array(pts)


def _finger_dir(deg):
    a = math.radians(deg)
    return np.array([math.sin(a), math.cos(a)])          # + towards +x (the kit's ab/ad sign)


def _ccw_dir(deg):
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)])


def nest_offset_mm(kit, cfg, kind, cache=None):
    """(xy offset of the unit origin from the nest point, z of the unit origin above the tray top), mm, for the
    unit lying as it lies in the nest (not turned)"""
    c = merged(cfg)
    ug = (cache or {}).get(kind) or UnitGeom(kit, kind)
    lo, hi = ug.box()
    cxy = 0.5 * (lo[:2] + hi[:2])
    if kind == "FM":
        f = c["finger"]
        tip_low = float(ug.pts(["dip"])[:, 2].min())
        z0 = float(f["t_mm"]) - (tip_low - float(f["deck"]["below_tip_mm"]))     # the deck's top = t_mm above the tray
        return -cxy, z0
    if kind == "TM":
        t = c["thumb"]
        return -cxy, float(t["riser_mm"]) - float(lo[2]) + float(t["below_mm"])
    return -cxy, float(c["stand_h_mm"]) - float(lo[2])


def nest_geoms(kit, cfg, kind, nest, nx, ny, cache=None, turned=False):
    """the nest's pieces in the tray frame (m); the unit lies with its origin at (nx, ny) + offset"""
    c = merged(cfg)
    mm = 1e-3
    ug = (cache or {}).get(kind) or UnitGeom(kit, kind)
    off, z0 = nest_offset_mm(kit, cfg, kind, cache)
    o = np.array([nx / mm + off[0], ny / mm + off[1]])      # the unit origin in the tray frame (mm)
    out = []
    pre = nest + "_"
    if kind == "FM":
        f = c["finger"]
        bb = f["base_block"]
        base_low = float(ug.pts(["base"])[:, 2].min())
        ztop = z0 + base_low
        x0, x1 = o[0] + bb["x"][0], o[0] + bb["x"][1]
        y0, y1 = o[1] + bb["y"][0], o[1] + bb["y"][1]
        out.append(_box(pre + "base_block", [x0 * mm, y0 * mm, 0.0], [x1 * mm, y1 * mm, ztop * mm]))
        g, wt, wh = float(f["wall_gap_mm"]), float(f["wall_t_mm"]), float(f["wall_h_mm"])
        out.append(_box(pre + "wall_a", [(x0 - g - wt) * mm, y0 * mm, 0.0], [(x0 - g) * mm, y1 * mm, (ztop + wh) * mm]))
        out.append(_box(pre + "wall_b", [(x1 + g) * mm, y0 * mm, 0.0], [(x1 + g + wt) * mm, y1 * mm, (ztop + wh) * mm]))
        sg = float(f["stop_gap_mm"])
        ymin = o[1] + float(ug.pts(["base"])[:, 1].min())
        out.append(_box(pre + "stop", [(x0 - g - wt) * mm, (ymin - sg - wt) * mm, 0.0], [(x1 + g + wt) * mm, (ymin - sg) * mm, (ztop + wh) * mm]))
        yp = f["yoke_pad"]
        Pa = ug.pts(["abad", "pitch"])
        m_ = (Pa[:, 1] >= yp["y"][0] - 3) & (Pa[:, 1] <= yp["y"][1] + 3) & (Pa[:, 0] >= yp["x"][0] - 3) & (Pa[:, 0] <= yp["x"][1] + 3)
        ylow = float(Pa[m_, 2].min()) if m_.any() else float(Pa[:, 2].min())
        out.append(_box(pre + "yoke_pad", [(o[0] + yp["x"][0]) * mm, (o[1] + yp["y"][0]) * mm, 0.0],
                        [(o[0] + yp["x"][1]) * mm, (o[1] + yp["y"][1]) * mm, (z0 + ylow - float(yp["below_mm"])) * mm]))
        dk = f["deck"]
        tip_low = float(ug.pts(["dip"])[:, 2].min())
        dtop = z0 + tip_low - float(dk["below_tip_mm"])
        h = float(dk["half_deg"])
        n = max(1, int(math.ceil(2 * h / float(dk["seg_deg"]))))
        for k in range(n):
            b0, b1 = -h + 2 * h * k / n, -h + 2 * h * (k + 1) / n
            P = _sector_prism(o, dk["r_mm"][0], dk["r_mm"][1], b0, b1, 0.0, dtop, _finger_dir)
            out.append((pre + "deck_%02d" % k, "prism", P * mm))
        return out
    if kind == "TM":
        t = c["thumb"]
        yb = t["yoke_block"]
        P1 = ug.pts(["roll", "base"])
        m_ = (P1[:, 0] >= yb["x"][0]) & (P1[:, 0] <= yb["x"][1]) & (P1[:, 1] >= yb["y"][0]) & (P1[:, 1] <= yb["y"][1])
        ztop = z0 + float(P1[m_, 2].min()) - float(t["below_mm"])
        out.append(_box(pre + "riser", [(o[0] + yb["x"][0]) * mm, (o[1] + yb["y"][0]) * mm, 0.0],
                        [(o[0] + yb["x"][1]) * mm, (o[1] + yb["y"][1]) * mm, ztop * mm]))
        ds = t["distal"]
        ax = np.asarray(ds["side_axis_xy_mm"], float)
        ymin = float(ds["y_min_mm"])
        # one support per distal link (the MP link and the IP link): the convex hull in plan of the link's lowest
        # points (within `low_band_mm` of its lowest z) on the far side of y_min, swept over the drawn side angle
        # +-8 deg plus the margin, its top 0.1 mm under that link's lowest point
        k = 0
        for ln in ("mp", "ip"):
            Pd = ug.pts([ln])
            zl = float(Pd[:, 2].min())
            low = Pd[(Pd[:, 2] <= zl + float(ds["low_band_mm"])) & (Pd[:, 1] >= ymin)]
            if len(low) < 3:
                continue
            sw = []
            for b in np.linspace(-float(ds["margin_deg"]), float(ds["margin_deg"]), 12):
                a = math.radians(b)
                R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
                sw.append((low[:, :2] - ax) @ R.T + ax)
            sw = np.vstack(sw)
            from scipy.spatial import ConvexHull
            hv = sw[ConvexHull(sw).vertices]
            hv[:, 1] = np.maximum(hv[:, 1], ymin)
            top = z0 + zl - float(ds["below_mm"])
            P = np.vstack([np.c_[hv, np.full(len(hv), top - 3.0)], np.c_[hv, np.full(len(hv), top)]])
            P[:, :2] += o
            out.append((pre + "distal_%02d" % k, "prism", P * mm))
            k += 1
        dlow = min(float(r[2][:, 2].min()) for r in out if r[0].startswith(pre + "distal_")) / mm + 3.0
        # the distal support's post: under its middle, from the plate
        dpz = [p for p in out if p[0].startswith(pre + "distal_")]
        if dpz:
            allp = np.vstack([p[2] for p in dpz]) / mm
            cx, cy = float(np.median(allp[:, 0])), float(np.median(allp[:, 1]))
            hp = float(t["post_half_mm"])
            out.append(_box(pre + "distal_post", [(cx - hp) * mm, (cy - hp) * mm, 0.0], [(cx + hp) * mm, (cy + hp) * mm, (dlow - 3.0) * mm]))
        return out
    st = c["stands"][kind]
    hgt = float(c["stand_h_mm"])
    if kind in c.get("posts", {}):
        eqs = ug.hull_equations()
        lo, _hi = ug.box()
        hp = float(c["post_half_mm"])
        R = np.diag([-1.0, -1.0]) if turned else np.eye(2)
        for i, (px, py) in enumerate(c["posts"][kind]):
            zs = [_lower_surface(eqs, px + dx, py + dy) for dx in (-hp, 0.0, hp) for dy in (-hp, 0.0, hp)]
            zs = [z for z in zs if z is not None]
            if not zs:
                raise ValueError("post %d of %s finds no material above it" % (i, kind))
            top = z0 + min(zs) - float(c["post_below_mm"])
            q = o + R @ np.array([px, py])
            out.append(_box(pre + "post_%d" % i, [(q[0] - hp) * mm, (q[1] - hp) * mm, 0.0], [(q[0] + hp) * mm, (q[1] + hp) * mm, top * mm]))
        if kind != "BS":
            return out
    elif kind == "BS":
        sp = c["stack_pocket"]
        u = np.array(sp["u"], float)
        u /= np.linalg.norm(u)
        a = math.atan2(u[1], u[0])
        q = [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]
        out.append((pre + "stand", "box", [o[0] * mm, o[1] * mm, 0.5 * hgt * mm], [float(st["u_half"]) * mm, float(st["w_half"]) * mm, 0.5 * hgt * mm], q))
    else:
        sx, sy = list(st["x"]), list(st["y"])
        if turned:
            sx, sy = [-sx[1], -sx[0]], [-sy[1], -sy[0]]
        out.append(_box(pre + "stand", [(o[0] + sx[0]) * mm, (o[1] + sy[0]) * mm, 0.0], [(o[0] + sx[1]) * mm, (o[1] + sy[1]) * mm, hgt * mm]))
    if kind == "BS":
        sp = c["stack_pocket"]
        u = np.array(sp["u"], float)
        u /= np.linalg.norm(u)
        w = np.array([-u[1], u[0]])
        hgt = z0 + float(ug.box()[0][2])
        P = ug.all
        ext = float(np.max(np.abs((P[:, :2] - 0.0) @ w)))
        for side in (-1.0, 1.0):
            cen = o + side * (ext + float(sp["gap_mm"]) + 0.5 * float(sp["lip_t_mm"])) * w
            a = math.atan2(w[1], w[0])
            q = [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]
            out.append((pre + "lip_%s" % ("a" if side < 0 else "b"), "box", [cen[0] * mm, cen[1] * mm, 0.5 * (hgt + float(sp["lip_h_mm"])) * mm],
                        [0.5 * float(sp["lip_t_mm"]) * mm, 0.5 * float(sp["lip_len_mm"]) * mm, 0.5 * (hgt + float(sp["lip_h_mm"])) * mm], q))
    return out


def tray_geoms(kit, cfg, tray, cache=None, turned=None):
    """the tray's plate and nests in its frame (m); turned: {nest: bool} (a flat part's stand follows its turn)"""
    t = trays_of(cfg)[tray]
    px, py, pt = t["plate"]
    out = [_box("plate", [-px / 2, -py / 2, -pt], [px / 2, py / 2, 0.0])]
    for nest, kind, nx, ny in t["nests"]:
        out += nest_geoms(kit, cfg, kind, nest, nx, ny, cache=cache, turned=bool((turned or {}).get(nest)))
    return out


def nest_unit_T(kit, cfg, tray, nest, cache=None, turned=False):
    """the unit's root pose (4x4, m) in the tray frame: dorsal side up, the palm frame's axes along the tray's (turned
    by 180 deg about the vertical through the footprint's centre when `turned`)"""
    t = trays_of(cfg)[tray]
    row = [r for r in t["nests"] if r[0] == nest][0]
    _n, kind, nx, ny = row
    off, z0 = nest_offset_mm(kit, cfg, kind, cache)
    T = np.eye(4)
    if turned:
        T[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        T[:3, 3] = [nx - off[0] * 1e-3, ny - off[1] * 1e-3, z0 * 1e-3]
    else:
        T[:3, 3] = [nx + off[0] * 1e-3, ny + off[1] * 1e-3, z0 * 1e-3]
    return T, kind
