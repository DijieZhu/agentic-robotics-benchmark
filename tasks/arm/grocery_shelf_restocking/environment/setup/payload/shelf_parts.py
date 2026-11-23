#!/usr/bin/env python3
"""What the cameras draw: the things of a store aisle as meshes (shelf_geom.py's (V, N, UV, F)), at the task's dimensions.

- products (the task's kinds: a box, a can; and a bottle): the task's primitive is the outer envelope; the print is
  the SKU's atlas (shelf_tex.py), uv 0..1;
- code_decal: the task's best-before code as a label on the side and at the place the task drew for the unit;
- gondola: a shelving bay in its own frame (the task's bay frame: x into the bay from the lips' outer face, y to
  the left, z up): uprights, perforated back, decks with a ticket rail in place of the lip, brackets, plinth, top;
- cart, tote: the task's cart block as a stock trolley, its tote as a shallow tray at the task's wall height;
- lift column, robot plate, camera heads.
Each builder returns a list of (name, mesh, look, opt).
"""
import math

import numpy as np

import shelf_geom as G

E = 0.00015                # a label's lift off the surface it sticks to


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# ------------------------------------------------------------------------------------------- products
def box_unit(sku, lay, bevel=0.0009):
    hx, hy, hz = sku["size"]
    V, N, UV, F = G.bevel_box([hx, hy, hz], bevel)
    Wm, Hm = lay["Wm"], lay["Hm"]
    x0 = lay["x0"]

    def face_of(n):
        a = int(np.abs(n).argmax())
        return (a, 1 if n[a] > 0 else -1)

    def uv_at(face, p):
        a, sg = face
        x, y, z = np.clip(p[0], -hx, hx), np.clip(p[1], -hy, hy), np.clip(p[2], -hz, hz)
        if a == 2:
            xm = (lay["top_x"] if sg > 0 else lay["bot_x"]) + (hy - y)
            ym = lay["top_y"] + (x + hx)
        else:
            if a == 0 and sg < 0:
                k, s = 0, hy - y
            elif a == 1 and sg < 0:
                k, s = 1, x + hx
            elif a == 0:
                k, s = 2, y + hy
            else:
                k, s = 3, hx - x
            xm, ym = x0[k] + s, hz - z
        return (xm / Wm, 1.0 - ym / Hm)

    uv = np.zeros((len(V), 2))
    for i in range(24):
        uv[i] = uv_at(face_of(N[i]), V[i])
    for b in range(24, 24 + 48, 4):                                # the twelve edge strips: the print of the face they start on
        f = face_of(N[b])
        for i in range(b, b + 4):
            uv[i] = uv_at(f, V[i])
    for b in range(72, len(V), 3):
        f = face_of(N[b])
        for i in range(b, b + 3):
            uv[i] = uv_at(f, V[i])
    return [("carton", (V, N, uv, F), None, {"uv": "own"})]


def _own_uv(m, hh):
    V, N, UV, F = m
    uv = np.stack([UV[:, 0] / (2 * math.pi), (V[:, 2] + hh) / (2 * hh)], 1)
    return V, N, uv, F


def can_unit(sku, nseg=48):
    r, hh = sku["size"]
    rb = r - 0.0007
    wrap = _own_uv(G.lathe([(rb, -hh + 0.0036), (rb, hh - 0.0036)], nseg, r_uv=1.0), hh)
    bot = G.lathe([(0, -hh + 0.0018), (r - 0.0040, -hh + 0.0018), (r - 0.0028, -hh), (r - 0.0006, -hh), (r, -hh + 0.0010),
                   (r, -hh + 0.0030), (rb, -hh + 0.0036)], nseg)
    top = G.lathe([(rb, hh - 0.0036), (r, hh - 0.0030), (r, hh - 0.0010), (r - 0.0006, hh), (r - 0.0028, hh),
                   (r - 0.0040, hh - 0.0028), (r - 0.0100, hh - 0.0030), (r - 0.0110, hh - 0.0024),
                   (0.62 * r + 0.0012, hh - 0.0024), (0.62 * r + 0.0006, hh - 0.0030), (0.62 * r - 0.0006, hh - 0.0030), (0.62 * r - 0.0012, hh - 0.0024),
                   (0.34 * r + 0.0012, hh - 0.0024), (0.34 * r + 0.0006, hh - 0.0029), (0.34 * r - 0.0006, hh - 0.0029), (0.34 * r - 0.0012, hh - 0.0024),
                   (0, hh - 0.0024)], nseg)       # round 2: two stamped expansion rings in the lid
    return [("wrap", wrap, None, {"uv": "own"}), ("ends", G.merge(bot, top), "can_metal", {})]


def bottle_unit(sku, nseg=48):
    r, hh = sku["size"]
    rn = 0.45 * r
    zs = hh * 0.30                                                 # the shoulder starts
    prof = [(0, -hh + 0.003), (r - 0.006, -hh + 0.003), (r - 0.002, -hh), (r, -hh + 0.004), (r, zs)]
    for k in range(1, 9):
        t = k / 8.0
        prof.append((r - (r - rn) * (0.5 - 0.5 * math.cos(math.pi * t)), zs + t * (hh * 0.92 - zs)))
    prof += [(rn, hh)]
    body = _own_uv(G.lathe(prof, nseg, r_uv=1.0), hh)
    cap = G.lathe([(rn, hh), (rn + 0.0012, hh + 0.0005), (rn + 0.0012, hh + 0.022), (rn - 0.0008, hh + 0.024), (0, hh + 0.024)], nseg)
    return [("body", body, None, {"uv": "own"}), ("cap", cap, "cap", {})]


def unit_parts(sku, lay, nseg=48):
    if sku["kind"] == "box":
        return box_unit(sku, lay)
    if sku["size"][1] > 0.08:
        return bottle_unit(sku, nseg)
    return can_unit(sku, nseg)


def code_decal(sku, at, sizes):
    """the label with the unit's code, in the unit's frame (shelf_world.add_code's layout: as on the -x side, turned
    by at["face_yaw"] about z; `offset` to the reader's right, `z` the centre's height). uv 0..1, u to the reader's
    right, v up."""
    pu, pv = sizes["patch"]
    R = _rz(at["face_yaw"])
    zc = at["z"]
    m = G.Mesh()
    if sku["kind"] == "box":
        side = int(round(at["face_yaw"] / (math.pi / 2))) % 4
        hf = sku["size"][0] if side % 2 == 0 else sku["size"][1]
        n = R @ np.array([-1.0, 0, 0])
        rd = R @ np.array([0, -1.0, 0])
        c = n * (hf + E) + rd * at["offset"] + np.array([0, 0, zc])
        zv = np.array([0, 0, 1.0])
        m.quad([c - rd * pu - zv * pv, c + rd * pu - zv * pv, c + rd * pu + zv * pv, c - rd * pu + zv * pv], n,
               [(0, 0), (1, 0), (1, 1), (0, 1)])
    else:
        r = sku["size"][0] - 0.0007 + E
        phi = math.pi + at["face_yaw"]
        ha = pu / r
        ns = 8
        for k in range(ns):
            a0, a1 = phi - ha + 2 * ha * k / ns, phi - ha + 2 * ha * (k + 1) / ns
            d0, d1 = np.array([math.cos(a0), math.sin(a0), 0]), np.array([math.cos(a1), math.sin(a1), 0])
            m.quad([r * d0 + [0, 0, zc - pv], r * d1 + [0, 0, zc - pv], r * d1 + [0, 0, zc + pv], r * d0 + [0, 0, zc + pv]],
                   np.asarray([d0, d1, d1, d0]), [(k / ns, 0), ((k + 1) / ns, 0), ((k + 1) / ns, 1), (k / ns, 1)])
    return m.arrays()


# ------------------------------------------------------------------------------------------- shelving
def _box(lo, hi, b=0.001):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return G.transform(G.bevel_box((hi - lo) / 2, b), np.eye(3), (lo + hi) / 2)


def gondola(width, depth, boards, top_z, C, brackets=True, tickets=True, back=True, top=True):
    """one bay in the bay frame. width: between the uprights' inner faces; depth: the lip's inner face to the back
    panel; boards: the decks' top heights; top_z: the underside of the top cover. C: BOARD_T, LIP_T, LIP_H, UPRIGHT_T,
    BACK_T (the task's shelf_sim constants). Everything is inside the task's boxes except the brackets under the decks
    at the uprights (15 mm thick, 45 mm tall at the back, clear of the lanes) and the ticket rail's 0.6 mm cover."""
    BT, LT, LH, UT, KT = C["BOARD_T"], C["LIP_T"], C["LIP_H"], C["UPRIGHT_T"], C["BACK_T"]
    xb = LT + depth + KT
    W2 = width / 2
    out = []
    z0 = boards[0] - BT
    for sg in (-1, 1):
        y0, y1 = (W2, W2 + UT) if sg > 0 else (-W2 - UT, -W2)
        out.append(("upright", _box((0, y0, 0), (xb, y1, top_z), 0.002), "gondola", {}))
    if back:
        m = _box((LT + depth, -W2, z0), (xb, W2, top_z), 0.0005)
        out.append(("back", G.uv_box(m), "gondola_back", {}))
    if top:
        out.append(("top", _box((0, -W2 - UT, top_z), (xb, W2 + UT, top_z + BT), 0.002), "gondola", {}))
    out.append(("plinth", _box((0.012, -W2, 0), (xb, W2, z0), 0.001), "gondola_kick", {}))
    for i, z in enumerate(boards):
        out.append(("deck", _box((LT, -W2, z - BT), (xb - KT, W2, z), 0.0012), "gondola_deck", {}))
        # the lip as a ticket rail: its front face carries the strip (u along the shelf, 0.5 m a repeat)
        V, N, UV, F = _box((0, -W2, z - BT), (LT, W2, z + LH), 0.0008)
        front = N[:, 0] < -0.5
        uv = np.stack([-V[:, 1] / 0.5, (V[:, 2] - (z - BT)) / (LH + BT)], 1)     # u to the reader's right (the bay's -y): round 1 had it mirrored
        fi = np.array([f for f in F if front[f].all()])
        ri = np.array([f for f in F if not front[f].all()])
        if tickets and len(fi):
            out.append(("rail_front", (V, N, uv, fi), "ticket", {"uv": "own"}))
            out.append(("rail", (V, N, UV, ri), "rail", {}))
        else:
            out.append(("rail", (V, N, UV, F), "rail", {}))
        if brackets:
            for sg in (-1, 1):
                ya, yb = (W2 - 0.015, W2) if sg > 0 else (-W2, -W2 + 0.015)
                prof = [(xb - KT, z - BT), (xb - KT, z - BT - 0.045), (LT + 0.03, z - BT - 0.012), (LT + 0.03, z - BT)]
                m = G.Mesh()
                P0 = [np.array([x, ya, zz]) for x, zz in prof]
                P1 = [np.array([x, yb, zz]) for x, zz in prof]
                m.quad([P0[0], P0[1], P0[2], P0[3]] if sg > 0 else [P0[3], P0[2], P0[1], P0[0]], [0, -1.0, 0], [(p[0], p[2]) for p in P0])
                m.quad([P1[3], P1[2], P1[1], P1[0]] if sg > 0 else [P1[0], P1[1], P1[2], P1[3]], [0, 1.0, 0], [(p[0], p[2]) for p in P1])
                m.quad([P0[1], P1[1], P1[2], P0[2]], [0.25, 0, -0.97], [(0, 0), (0.015, 0), (0.015, 0.2), (0, 0.2)])
                mm = m.arrays()
                Vb, Nb, UVb, Fb = mm
                # outward normals: flip a face that looks into the bracket
                out.append(("bracket", (Vb, Nb, UVb, Fb), "gondola", {"double": True}))
    return out


_STOCK_PARTS = {}          # nseg -> SKU name -> its unit meshes (dressing SKUs: one design per name)


def gondola_stock(unit_like, skus, lays, rng, width, depth, boards, top_z, C, fill=(0.75, 1.0), nseg=20):
    """products on a dressing bay: per deck one kind's SKUs in lanes across the width, faced to the front with a few
    millimetres and degrees of disorder, 1 to n deep. Returns {sku: [mesh parts...]} merged per SKU and look."""
    LT = C["LIP_T"]
    out = {}
    parts_of = _STOCK_PARTS.setdefault(nseg, {})
    names = list(skus)
    for i, z in enumerate(boards):
        head = (boards[i + 1] - C["BOARD_T"] if i + 1 < len(boards) else top_z) - z
        fit = [n for n in names if 2 * (skus[n]["size"][2] if skus[n]["kind"] == "box" else skus[n]["size"][1]) < head - 0.05]
        if not fit:
            continue
        pick = [fit[int(k)] for k in rng.choice(len(fit), size=min(len(fit), int(rng.integers(2, 5))), replace=False)]
        y = -width / 2 + 0.012
        k = 0
        while True:
            n = pick[k % len(pick)]
            sku = skus[n]
            wy = 2 * (sku["size"][1] if sku["kind"] == "box" else sku["size"][0])
            dx = 2 * sku["size"][0]
            hh = sku["size"][2] if sku["kind"] == "box" else sku["size"][1]
            lanes = int(rng.integers(2, 5))
            if y + lanes * (wy + 0.008) > width / 2 - 0.012:
                lanes = int((width / 2 - 0.012 - y) / (wy + 0.008))
                if lanes < 1:
                    break
            for _l in range(lanes):
                cy = y + wy / 2 + 0.004
                deep = int((depth - 0.02) / (dx + 0.006))
                nd = max(1, int(round(deep * rng.uniform(*fill))))
                # round 2: a lane shopped from: its row pushed or left back from the edge by 2 to 6 cm
                back = float(rng.uniform(0.02, 0.06)) if (rng.random() < 0.18 and nd < deep) else 0.0
                back = min(back, max(0.0, (depth - 0.02) - nd * (dx + 0.006) - 0.005))
                for s in range(nd):
                    cx = LT + 0.010 + dx / 2 + s * (dx + 0.006) + float(rng.uniform(0, 0.004)) + back
                    yaw = float(rng.normal(0, 0.035)) if sku["kind"] == "box" else float(rng.uniform(-math.pi, math.pi))
                    if sku["kind"] == "box" and s == 0 and rng.random() < 0.10:      # round 2: a front unit left turned
                        yaw = float(rng.choice([-1, 1]) * rng.uniform(0.18, 0.5))
                    if sku["kind"] != "box" and s == 0 and rng.random() < 0.7:
                        yaw = math.pi + float(rng.normal(0, 0.25))            # a faced can: its logo to the aisle
                    t = np.array([cx, cy + float(rng.normal(0, 0.0015)), z + hh + 0.0002])
                    if n not in parts_of:                  # a SKU's meshes are the same for every unit of it
                        parts_of[n] = unit_parts(sku, lays[n], nseg)
                    for name, m, lk, opt in parts_of[n]:
                        out.setdefault((n, name, lk), []).append(G.transform(m, _rz(yaw), t))
                y += wy + 0.008
            k += 1
    return {k: G.merge(*v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------- cart, tote
def _tube(p0, p1, r, nseg=12):
    return G.cylinder(p0, p1, r, nseg, caps=True)


def cart(CART, TOTE):
    """the task's cart block (0.38 x 0.32 m, top at 0.55 m) as a stock trolley: a laminate deck, a tubular frame, a
    lower shelf, four castors; a low push bar on the far side, under the deck's level. World frame."""
    cx, cy = CART["centre"]
    hx, hy = CART["half"]
    top = CART["top"]
    out = []
    out.append(("cart_deck", _box((cx - hx, cy - hy, top - 0.022), (cx + hx, cy + hy, top), 0.003), "cart_deck", {}))
    out.append(("cart_edge", _box((cx - hx, cy - hy, top - 0.030), (cx + hx, cy + hy, top - 0.022), 0.001), "cart_frame", {}))
    r = 0.011
    legs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * (hx - r - 0.004), cy + sy * (hy - r - 0.004)
            legs.append(_tube((x, y, 0.105), (x, y, top - 0.03), r))
            legs.append(_tube((x, y, 0.045), (x, y, 0.105), 0.007))                                        # the castor's stem
            wheel = G.cylinder((x - 0.011, y, 0.045), (x + 0.011, y, 0.045), 0.045, 28, caps=True, bevel=0.004)
            out.append(("castor", wheel, "castor", {}))
            out.append(("castor_fork", _box((x - 0.016, y - 0.012, 0.045), (x + 0.016, y + 0.012, 0.100), 0.002), "cart_frame", {}))
    for sy in (-1, 1):
        y = cy + sy * (hy - r - 0.004)
        for zz in (0.20, top - 0.045):
            legs.append(_tube((cx - hx + r, y, zz), (cx + hx - r, y, zz), 0.008))
    for sx in (-1, 1):
        x = cx + sx * (hx - r - 0.004)
        for zz in (0.20, top - 0.045):
            legs.append(_tube((x, cy - hy + r, zz), (x, cy + hy - r, zz), 0.008))
    out.append(("cart_frame", G.merge(*legs), "cart_frame", {}))
    out.append(("cart_shelf", _box((cx - hx + 0.012, cy - hy + 0.012, 0.196), (cx + hx - 0.012, cy + hy - 0.012, 0.208), 0.002), "cart_shelf", {}))
    # the push bar: on the far (-y) side, its top 0.03 m under the deck
    yb = cy - hy - 0.035
    bar = [_tube((cx - hx + 0.03, cy - hy + 0.01, top - 0.045), (cx - hx + 0.03, yb, top - 0.045), 0.009),
           _tube((cx + hx - 0.03, cy - hy + 0.01, top - 0.045), (cx + hx - 0.03, yb, top - 0.045), 0.009),
           _tube((cx - hx + 0.03, yb, top - 0.045), (cx + hx - 0.03, yb, top - 0.045), 0.009)]
    out.append(("push_bar", G.merge(*bar), "cart_frame", {}))
    out.append(("push_grip", _tube((cx - 0.10, yb, top - 0.045), (cx + 0.10, yb, top - 0.045), 0.0115, 16), "grip", {}))
    return out


def tote(CART, TOTE):
    """the task's tote (inner 0.34 x 0.27 m, walls 6 mm thick and 40 mm tall) as a shallow plastic tray: a floor of
    1.2 mm, ribbed walls with a rolled rim, rounded corners. World frame."""
    tx, ty = TOTE["centre"]
    ix, iy = TOTE["inner_half"]
    w, h = TOTE["wall"], TOTE["height"]
    z0 = CART["top"]
    o_out = G.rounded_rect(ix + w, iy + w, 0.016, 8)
    o_in = G.rounded_rect(ix, iy, 0.010, 8)
    parts = [G.wall(o_out, z0, z0 + h - 0.0015), G.wall(o_in, z0 + 0.0012, z0 + h - 0.0015, outward=False),
             G.band(o_out, o_in, z0 + h - 0.0015, z0 + h - 0.0015, up=True), G.cap(o_in, z0 + 0.0012, True)]
    ribs = []
    for k in range(-3, 4):
        x = tx + k * 0.085
        for sy in (-1, 1):
            ribs.append(_box((x - 0.004, ty + sy * (iy + w) - 0.0008, z0 + 0.004), (x + 0.004, ty + sy * (iy + w) + 0.0008, z0 + h - 0.006), 0.0005))
    body = G.transform(G.merge(*parts), np.eye(3), (tx, ty, 0.0))
    return [("tote", body, "tote", {"double": True}), ("tote_ribs", G.merge(*ribs), "tote", {})]


# ------------------------------------------------------------------------------------------- the lift
def lift_static():
    """on the floor under the robot: a base plate with floor bolts (the task's column stands on the floor at
    (-0.19..0.09, +-0.14))"""
    out = [("lift_plate", _box((-0.27, -0.22, 0.0), (0.17, 0.22, 0.016), 0.003), "steel_plate", {})]
    bolts = []
    for sx in (-0.24, 0.14):
        for sy in (-0.19, 0.19):
            bolts.append(G.cylinder((sx, sy, 0.016), (sx, sy, 0.026), 0.011, 6, caps=True))
    out.append(("lift_bolts", G.merge(*bolts), "bolt", {}))
    return out


LIFT_STAGES = ((0.132, 0.62), (0.116, 0.62), (0.100, 0.62))        # half side of each nested tube, its length


def lift_stage(k):
    """stage k of the telescopic column in its own frame: z = 0 at its top; a square tube with a collar"""
    hs, le = LIFT_STAGES[k]
    o = G.rounded_rect(hs, hs, 0.012, 6)
    tube = G.merge(G.wall(o, -le, 0.0), G.cap(o, -0.0005, True))      # under the collar's top face, not in its plane
    col = G.slab(G.rounded_rect(hs + 0.006, hs + 0.006, 0.014, 6), -0.030, 0.0, 0.002, 0.002)
    return [("lift_tube%d" % k, G.transform(tube, np.eye(3), (-0.05, 0, 0)), "lift_alu", {}),
            ("lift_collar%d" % k, G.transform(col, np.eye(3), (-0.05, 0, 0)), "lift_collar", {})]


def carriage_parts(cams):
    """on the lift's carriage (the robot's base frame: z = 0 at the base's foot): the robot's mounting plate and, if
    cams, the two camera heads the task mounts at (0.15, +-0.40, 0.42) looking at (0.66, 0, 0.28): a slim arm from
    the plate, a small machine-vision camera with a lens"""
    out = [("robot_plate", _box((-0.20, -0.15, -0.018), (0.10, 0.15, 0.0), 0.003), "steel_plate", {})]
    if not cams:
        return out
    for sg in (-1, 1):
        p = np.array([0.15, sg * 0.40, 0.42])
        t = np.array([0.66, 0.0, 0.28])
        f = (t - p) / np.linalg.norm(t - p)
        x = np.cross(f, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        up = np.cross(x, f)
        R = np.stack([f, -x, up], 1)                                # the camera's frame: x forward
        body = G.transform(G.bevel_box([0.021, 0.0145, 0.0145], 0.0015), R, p - f * 0.026)
        lens = G.cylinder(p - f * 0.006, p + f * 0.020, 0.0115, 24, caps=True, bevel=0.001)
        glass = G.cylinder(p + f * 0.0201, p + f * 0.0206, 0.0085, 24, caps=True)
        foot = np.array([-0.16, sg * 0.14, -0.009])
        elbow = np.array([-0.16, sg * 0.40, -0.009])
        knee = np.array([-0.16, sg * 0.40, 0.40])
        arm = G.merge(_tube(foot, elbow, 0.006), _tube(elbow, knee, 0.006), _tube(knee, p - f * 0.047 - up * 0.004, 0.005),
                      G.cylinder(elbow - [0, 0, 0.007], elbow + [0, 0, 0.007], 0.009, 12), G.cylinder(knee - [0.007, 0, 0], knee + [0.007, 0, 0], 0.008, 12))
        out += [("cam_body", body, "cam_body", {}), ("cam_lens", lens, "cam_lens", {}), ("cam_glass", glass, "cam_glass", {}),
                ("cam_arm", arm, "cam_arm", {})]
    return out
