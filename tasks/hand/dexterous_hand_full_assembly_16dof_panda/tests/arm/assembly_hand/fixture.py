"""The raised fixture of the hand assembly cell (workcell equipment, light machined aluminium): a base plate on the
bench, a pedestal that carries the palm's bottom shell on its palmar face 150 mm above the bench with two jaws on the
14 mm strip at the wrist end, one rest deck under the three fingers' tips over each finger's whole ab/ad range, a
10 x 10 mm pad on a slender post under the thumb's base yoke (relieved under the yoke's screw head), and the fixture's
datum board.

Fixture frame F: origin on the bench top under the palm frame's origin, axes = the palm frame's (P: +y towards the
fingertips, +z dorsal, -x towards the thumb). The palm frame sits at F + (0, 0, palm_z). In the cell the palm base is a
fixed child of the fixture (welded: both static, MuJoCo makes no contact between them).

    merged(cfg)                 the fixture's numbers (DEFAULT_FIXTURE updated by cfg)
    fixture_geoms(cfg)          [(name, "box", centre, half sizes, quat)] and [(name, "prism", vertices (n, 3))] in F (m)
    palm_T_in_fixture(cfg)      the palm frame's 4x4 pose (m) in F
    deck_sectors(cfg)           the rest deck's sectors per finger lane (P, mm and deg) as built

Every number below is a measured or derived value; the source of each is beside it.
"""
import math

import numpy as np

DEFAULT_FIXTURE = {
    # the palmar face (the bottom shell's lowest point, z -29.51 in P on the true solid; -29.7 on the collision pieces) 150 mm
    # above the bench: the palm frame's origin 179.51 mm above the bench
    "palmar_z_mm": -29.51,
    "palmar_above_bench_mm": 150.0,
    # the base plate (0.30 x 0.28 m), centred under the whole hand's footprint (P x -163.5..15.6 with the
    # thumb, y -41..178.8: centre (-74, 69) mm)
    "plate_mm": {"x": [-213.0, 67.0], "y": [-81.0, 219.0], "t": 12.0},
    # the pedestal: a block inside the palm's outline (the shell spans x -73.3..15.8, y -41.2..61.1 on its pieces),
    # top on the palmar face; on the thumb side it stops 2.9 mm short of the mounted thumb's base, which reaches below
    # the palmar face to x -43.9 (and moves away along -x on its path); distally 6 mm short of the mounted fingers'
    # ab/ad yokes, which reach below the palmar face from y 56 (a first pedestal to x -60 cut 8 mm into the
    # thumb's pieces)
    "pedestal_mm": {"x": [-41.0, 8.0], "y": [-27.0, 50.0]},
    # the two jaws on the wrist-end strip (y -41..-27, below z -7: the cover's skirt reaches z -5.6 there): 6 mm thick,
    # 0.5 mm off the shell's x ends
    "jaw_mm": {"y": [-41.0, -27.0], "top_z": -7.0, "t": 6.0, "gap": 0.5, "x_ends": [-73.3, 15.8]},
    # the rest deck: top 1.0 mm below the fingertips' lowest point (z -37.71 in P), radial band 84..135 mm from each
    # finger's ab/ad axis over that finger's mounted ab/ad range (deg, + towards +x; the mounted ranges of
    # gt/joints.json: index -45.69..+18.12, middle -45.69..+42.17, ring -26.01..+45.69) plus a 3 deg margin each end;
    # 6 mm thick, on two posts
    "deck": {"top_z_mm": -38.71, "r_mm": [84.0, 135.0], "t_mm": 6.0, "margin_deg": 3.0, "max_seg_deg": 10.0,
             "lanes": {"index": {"axis_xy_mm": [-59.3247, 46.1098], "range_deg": [-45.69, 18.12]},
                       "middle": {"axis_xy_mm": [-28.7247, 51.1098], "range_deg": [-45.69, 42.17]},
                       "ring": {"axis_xy_mm": [1.8753, 46.1098], "range_deg": [-26.01, 45.69]}},
             "posts": [{"lane": "ring", "r_mm": 118.0, "deg": 40.0}, {"lane": "index", "r_mm": 118.0, "deg": -30.0}],
             "post_half_mm": 5.0},
    # the thumb's pad: 10 x 10 mm, top at z -40.2 under the thumb's base yoke's lowest point (-82.0, -6.8), on a
    # 6 x 6 mm post; relieved 0.3 mm deep under the head of the yoke's screw (the head's footprint x -82.87..-79.78,
    # y -8.69..-5.63 with the thumb mounted and settled, its lowest point 0.18 mm below the pad's top: the screw is drawn
    # only, so without the relief the pad stood 0.18 mm inside it), the pocket 0.3 mm wider than the head all round
    "thumb_pad": {"xy_mm": [-82.0, -6.8], "half_mm": 5.0, "top_z_mm": -40.2, "t_mm": 4.0, "post_half_mm": 3.0,
                  "relief": {"x_mm": [-83.2, -79.45], "y_mm": [-9.0, -5.3], "depth_mm": 0.3}},
    # the fixture's datum board: 5 x 4 squares of 10 mm (4 x 3 inner corners: a 5 x 3 board's 4 x 2 inner corners are
    # refused by OpenCV's chessboard detectors, which need more than two corners each way), 4 mm margin, 2 mm thick,
    # flat on the plate beside the palm on the ring side (P x 17..65, y -26..32: plate on every side, clear of the
    # jaws and the palm's outline in plan, under nothing), its columns along P's y
    "board": {"cols": 5, "rows": 4, "square_mm": 10.0, "margin_mm": 4.0, "thickness_mm": 2.0, "xy_mm": [41.0, 3.0], "yaw_deg": 90.0},
    # the fixture's place in the cell (layout draws it): the palm frame's origin 0.44 m from the Panda's axis on the
    # bench's centre line, P's +y along the cell's +x (yaw 0: the fingers point away from the robot)
    "nominal_xy": [-0.01, 0.0],
    "nominal_yaw_deg": 0.0,
}


def merged(cfg=None):
    import copy
    out = copy.deepcopy(DEFAULT_FIXTURE)
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def palm_z_m(cfg=None):
    c = merged(cfg)
    return (float(c["palmar_above_bench_mm"]) - float(c["palmar_z_mm"])) * 1e-3


def palm_T_in_fixture(cfg=None):
    T = np.eye(4)
    T[2, 3] = palm_z_m(cfg)
    return T


def _box(name, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return (name, "box", (0.5 * (lo + hi)).tolist(), (0.5 * (hi - lo)).tolist(), [1.0, 0.0, 0.0, 0.0])


def _abad_dir(deg):
    """the in-plane direction of a finger turned by `deg` about its vertical ab/ad axis (+ towards +x of P)"""
    a = math.radians(float(deg))
    return np.array([math.sin(a), math.cos(a)])


def deck_sectors(cfg=None):
    c = merged(cfg)
    dk = c["deck"]
    out = {}
    for lane, L in dk["lanes"].items():
        a0 = float(L["range_deg"][0]) - float(dk["margin_deg"])
        a1 = float(L["range_deg"][1]) + float(dk["margin_deg"])
        out[lane] = {"axis_xy_mm": list(L["axis_xy_mm"]), "deg": [a0, a1], "r_mm": list(dk["r_mm"])}
    return out


def fixture_geoms(cfg=None):
    """the fixture in F (m): boxes (name, "box", centre, half sizes, quat wxyz) and convex prisms (name, "prism",
    vertices (n, 3))"""
    c = merged(cfg)
    mm = 1e-3
    zp = palm_z_m(c)
    out = []
    p = c["plate_mm"]
    t = float(p["t"]) * mm
    out.append(_box("plate", [p["x"][0] * mm, p["y"][0] * mm, 0.0], [p["x"][1] * mm, p["y"][1] * mm, t]))
    pz = zp + float(c["palmar_z_mm"]) * mm                    # the palmar face's height in F
    pd = c["pedestal_mm"]
    out.append(_box("pedestal", [pd["x"][0] * mm, pd["y"][0] * mm, t], [pd["x"][1] * mm, pd["y"][1] * mm, pz]))
    jw = c["jaw_mm"]
    jt, jg = float(jw["t"]) * mm, float(jw["gap"]) * mm
    ztop = zp + float(jw["top_z"]) * mm
    y0, y1 = float(jw["y"][0]) * mm, float(jw["y"][1]) * mm
    x0, x1 = float(jw["x_ends"][0]) * mm, float(jw["x_ends"][1]) * mm
    out.append(_box("jaw_thumb_side", [x0 - jg - jt, y0, t], [x0 - jg, y1, ztop]))
    out.append(_box("jaw_ring_side", [x1 + jg, y0, t], [x1 + jg + jt, y1, ztop]))
    # the jaws' feet: a bar from each jaw to the pedestal below the shell (z below the palmar face)
    out.append(_box("jaw_bar", [x0 - jg - jt, y0, t], [x1 + jg + jt, y1, pz]))
    # the rest deck
    dk = c["deck"]
    top = zp + float(dk["top_z_mm"]) * mm
    bot = top - float(dk["t_mm"]) * mm
    r0, r1 = float(dk["r_mm"][0]) * mm, float(dk["r_mm"][1]) * mm
    for lane, s in deck_sectors(c).items():
        ax = np.array(s["axis_xy_mm"], float) * mm
        a0, a1 = s["deg"]
        n = max(1, int(math.ceil((a1 - a0) / float(dk["max_seg_deg"]))))
        for k in range(n):
            b0 = a0 + (a1 - a0) * k / n
            b1 = a0 + (a1 - a0) * (k + 1) / n
            pts = []
            for r in (r0, r1):
                for b in (b0, b1):
                    xy = ax + r * _abad_dir(b)
                    pts.append([xy[0], xy[1], bot])
                    pts.append([xy[0], xy[1], top])
            # the outer arc's chord would cut inside the arc: one more radial point at the middle angle
            xy = ax + r1 / math.cos(math.radians(0.5 * (b1 - b0))) * _abad_dir(0.5 * (b0 + b1))
            pts.append([xy[0], xy[1], bot])
            pts.append([xy[0], xy[1], top])
            out.append(("deck_%s_%02d" % (lane, k), "prism", np.array(pts)))
    ph = float(dk["post_half_mm"]) * mm
    for i, po in enumerate(dk["posts"]):
        L = dk["lanes"][po["lane"]]
        xy = np.array(L["axis_xy_mm"], float) * mm + float(po["r_mm"]) * mm * _abad_dir(po["deg"])
        out.append(_box("deck_post_%d" % i, [xy[0] - ph, xy[1] - ph, t], [xy[0] + ph, xy[1] + ph, bot]))
    # the thumb's pad
    tp = c["thumb_pad"]
    cx, cy = float(tp["xy_mm"][0]) * mm, float(tp["xy_mm"][1]) * mm
    h = float(tp["half_mm"]) * mm
    ttop = zp + float(tp["top_z_mm"]) * mm
    rl = tp.get("relief")
    if not rl:
        out.append(_box("thumb_pad", [cx - h, cy - h, ttop - float(tp["t_mm"]) * mm], [cx + h, cy + h, ttop]))
    else:
        # the pad below the relief's floor, then its top layer round the pocket (four boxes)
        z1 = ttop - float(rl["depth_mm"]) * mm
        rx0, rx1 = float(rl["x_mm"][0]) * mm, float(rl["x_mm"][1]) * mm
        ry0, ry1 = float(rl["y_mm"][0]) * mm, float(rl["y_mm"][1]) * mm
        out.append(_box("thumb_pad", [cx - h, cy - h, ttop - float(tp["t_mm"]) * mm], [cx + h, cy + h, z1]))
        out.append(_box("thumb_pad_xlo", [cx - h, cy - h, z1], [rx0, cy + h, ttop]))
        out.append(_box("thumb_pad_xhi", [rx1, cy - h, z1], [cx + h, cy + h, ttop]))
        out.append(_box("thumb_pad_ylo", [rx0, cy - h, z1], [rx1, ry0, ttop]))
        out.append(_box("thumb_pad_yhi", [rx0, ry1, z1], [rx1, cy + h, ttop]))
    hp = float(tp["post_half_mm"]) * mm
    out.append(_box("thumb_pad_post", [cx - hp, cy - hp, t], [cx + hp, cy + hp, ttop - float(tp["t_mm"]) * mm]))
    return out


def board_pose_in_fixture(cfg=None):
    c = merged(cfg)
    b = c["board"]
    a = math.radians(float(b.get("yaw_deg", 0.0)))
    T = np.eye(4)
    T[:3, :3] = [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = [float(b["xy_mm"][0]) * 1e-3, float(b["xy_mm"][1]) * 1e-3, float(c["plate_mm"]["t"]) * 1e-3]
    return T


def footprint_xy(cfg=None):
    """the fixture's footprint on the bench (the plate's corners, F, m) and the deck's overhang beyond it"""
    c = merged(cfg)
    p = c["plate_mm"]
    mm = 1e-3
    P = np.array([[p["x"][0], p["y"][0]], [p["x"][1], p["y"][0]], [p["x"][1], p["y"][1]], [p["x"][0], p["y"][1]]], float) * mm
    return P
