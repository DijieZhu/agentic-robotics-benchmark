"""Shelf restocking: one episode's world (the bay, the cart and tote, the products, the Panda on its lift).

`draw_unit(seed)` draws a unit (the bay's design and its SKUs), `draw_episode(seed, unit)` an episode on it (the
bay's pose, the planogram's served lanes, the old stock, the delivery in the tote), `World(unit, episode)` builds
and holds the simulation: the lift controller (`shelf_ctl.LiftCtl`) plus a joint step, the cameras, and the
state the credit and the privileged reference read.

Frames: world z up; the robot's base on the lift at (0, 0, lift); the bay frame at the bay's front plane (the
lips' outer face) on the floor, x into the bay, y to the left, turned by the bay's yaw. A lane is a y band of one
shelf; its slots are numbered from the front. A unit's own frame: x toward its back, so a box faces out when its
yaw equals the bay's (its printed front, the face with the logo, is its -x face). Metres, radians.

A SKU's look (`add_look`, visual geoms only): a body colour, a label band round its middle, a logo (a square, a
disc or two bars, in its own colour and place) on a box's front or on both sides of a can or bottle, a bottle's
cap. The two SKUs of a shelf share a body colour half the time; their bands and logos tell them apart.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco
import numpy as np

import box_transport_env as B
import shelf_sim as S
from shelf_ctl import LiftCtl

N_LANES = 5
SHELVES = 3
LIP_GAP = 0.012              # m, the front unit's face from the lip's inner face when a lane is faced
SLOT_GAP = 0.012             # m, between units in a lane
BACK_CLEAR = 0.06            # m, the deepest slot's centre from the back panel: link 6 reaches 0.044 m past
                             # the grip site toward the back (it met the panel at 0.045)
# The cart clear of the bay's side; the tote's slots two rows apart along the fingers' axis (y): the palm
# (+-0.104 m along it) passes over the other row, and a neighbour in the same row is only the palm's 32 mm
# half-thickness away (a 3 x 4 grid at 0.09 m put the fingers and the palm on the neighbours).
# Its columns at x 0.12-0.36 (0.04-0.08 were out of reach for a hand pointing down; at 0.40 a turning palm met the
# bay), its far row 0.635 m out (at 0.685 the far corner, 0.77 m from the base, was at the edge of reach).
CART = {"centre": (0.22, -0.55), "half": (0.19, 0.16), "top": 0.55}
TOTE = {"centre": (0.22, -0.55), "inner_half": (0.17, 0.135), "wall": 0.006, "height": 0.04}
TOTE_GRID = (4, 2)           # 8 slots: the delivery, the extras and the units set aside
TOTE_PITCH = (0.085, 0.17)   # m, tote slots (x, y): a 0.07 m unit in an end slot clears the wall by 7 mm
SIDE_CLEAR = 0.17            # m, an outer lane's centre to the bay's side: the palm (0.104) and, on the +y side, the
                             # forearm (link 5 met the side 0.11 m beyond a lane 0.24 m out at 0.115)
TOTE_BASE = 0.60             # the lift's height for the tote
LIFT_BELOW_BOARD = 0.20      # the lift's height under a worked board (the reference's choice)
TODAY_RANGE = (20, 71)       # today's day number
OLD_SPAN = 14                # an unexpired old unit's code: today to today + OLD_SPAN
EXPIRED_SPAN = 9             # an expired unit's code: today - EXPIRED_SPAN to today - 1
NEW_LO, NEW_HI = 2, 24       # a delivered unit's code: today + NEW_LO to today + NEW_HI (unless inverted)
P_EXPIRED = 0.3              # each old unit of a served lane
MIN_FREE = 0                 # the tote's free slots at the start: MIN_FREE to MAX_FREE (the rest hold extras)
MAX_FREE = 1
OLD_DROP = 3                 # a served lane's old units: capacity - OLD_DROP (at least 1) to capacity
P_INVERSION = 0.6            # a served lane with unexpired old units: one delivered code earlier than one of theirs
MAX_DQ = 0.1
MAX_LIFT = S.LIFT_STEP

# Product kinds (primitive shapes; half sizes): one kind, so one height, per shelf.
KINDS = {
    "box":     {"kind": "box", "size": (0.025, 0.035, 0.07), "mass": (0.25, 0.40)},
    "can":     {"kind": "cyl", "size": (0.033, 0.061), "mass": (0.33, 0.42)},
    "bottle":  {"kind": "cyl", "size": (0.032, 0.095), "mass": (0.45, 0.65)},
    "tallbox": {"kind": "box", "size": (0.030, 0.030, 0.10), "mass": (0.35, 0.55)},
}
PALETTE = [(0.85, 0.65, 0.20), (0.75, 0.10, 0.12), (0.20, 0.45, 0.80), (0.30, 0.70, 0.30), (0.60, 0.30, 0.70),
           (0.95, 0.45, 0.10), (0.15, 0.60, 0.60), (0.55, 0.40, 0.25), (0.90, 0.85, 0.30), (0.35, 0.35, 0.80),
           (0.80, 0.30, 0.50), (0.20, 0.25, 0.30), (0.92, 0.92, 0.88), (0.45, 0.65, 0.15)]
LOGOS = ("square", "disc", "bars")


def half_h(sku):
    return sku["size"][2] if sku["kind"] == "box" else sku["size"][1]


def half_x(sku):
    return sku["size"][0]


def rng_of(seed, tag):
    seed = int(seed) & (2 ** 64 - 1)
    return np.random.default_rng([seed & 0xFFFFFFFF, seed >> 32, tag])


# ----------------------------------------------------------------------------------------------
# Units and episodes
# ----------------------------------------------------------------------------------------------
def draw_unit(seed):
    """A bay design and its SKUs: three shelves, one product kind each, two SKUs per shelf (one height per shelf,
    colours and footprints their own). Half the units give two shelves the same kind and height, so a unit's
    shape does not tell which of the two shelves it belongs to. Headroom = the height + 0.32-0.35 m; depth
    0.24-0.28 m; lane pitch 0.115-0.13 m. No two SKUs of a unit share a look (body, band, logo colour and shape)."""
    r = rng_of(seed, 1)
    names_k = list(r.permutation(list(KINDS)))
    if r.random() < 0.5:
        kinds = [names_k[0], names_k[0], names_k[1]]          # two shelves of one kind and one height
    else:
        kinds = names_k[:SHELVES]
    # tallest at the bottom keeps the top board in the lift's range
    kinds.sort(key=lambda k: -half_h(KINDS[k]))
    shelves, skus, z = [], {}, float(r.uniform(0.26, 0.34))
    z_of_kind = {}
    seen = set()

    def other(*avoid):
        return int(r.choice([c for c in range(len(PALETTE)) if c not in avoid]))
    for si, kind in enumerate(kinds):
        base = KINDS[kind]
        names = []
        if kind not in z_of_kind:
            z_of_kind[kind] = r.uniform(0.92, 1.05)   # one height per kind: the palm passes 5 mm over the neighbours
        z_scale = z_of_kind[kind]
        body0 = int(r.integers(len(PALETTE)))
        for j in range(2):
            scale = np.r_[r.uniform(0.92, 1.05, 2), z_scale]
            if base["kind"] != "box":
                scale = np.r_[scale[0], z_scale]
            size = np.asarray(base["size"], float) * scale
            if base["kind"] == "box":
                size[1] = min(size[1], 0.035)            # at most 0.07 m across the fingers
            else:
                size[0] = min(size[0], 0.035)
            name = f"{kind}{si}{'ab'[j]}"
            while True:
                body = body0 if (j == 0 or r.random() < 0.5) else other(body0)
                band = other(body)
                logo = other(body, band)
                shape = int(r.integers(len(LOGOS)))
                if (body, band, logo, shape) not in seen:
                    seen.add((body, band, logo, shape))
                    break
            look = {"band": list(PALETTE[band]), "logo": list(PALETTE[logo]), "logo_shape": LOGOS[shape],
                    "logo_z": round(float(r.uniform(-0.30, 0.15)), 3), "band_h": round(float(r.uniform(0.18, 0.35)), 3),
                    "cap": list(PALETTE[other(body)])}
            skus[name] = {"kind": base["kind"], "size": [round(float(v), 4) for v in size],
                          "mass": round(float(r.uniform(*base["mass"])), 3),
                          "rgba": list(PALETTE[body]) + [1.0], "look": look}
            names.append(name)
        h = 2 * max(half_h(skus[n]) for n in names)
        headroom = h + float(r.uniform(0.32, 0.35))   # 0.29-0.32 let link 5 meet the shelf above at the deepest slots
        shelves.append({"board": round(z, 4), "headroom": round(headroom, 4), "skus": names})
        z += headroom + S.BOARD_T
    return {"shelves": shelves, "skus": skus, "depth": round(float(r.uniform(0.24, 0.28)), 4),
            "pitch": round(float(r.uniform(0.115, 0.13)), 4), "top_z": round(z - S.BOARD_T, 4),
            "front_x": 0.50}


def lane_y(unit, li):
    return (li - (N_LANES - 1) / 2) * unit["pitch"]


def slot_x(unit, sku, k):
    """A slot's centre, bay frame x: slot 0 faced (its front face LIP_GAP behind the lip's inner face)."""
    hx = half_x(sku)
    return S.LIP_T + LIP_GAP + hx + k * (2 * hx + SLOT_GAP)


def capacity(unit, sku):
    k = 0
    while slot_x(unit, sku, k) <= S.LIP_T + unit["depth"] - BACK_CLEAR:
        k += 1
    return k


def lane_goal(olds, codes_new, today):
    """A served lane's goal, front to back: the unexpired old codes and the delivered codes, sorted (the earliest
    in front). olds: [(slot, push, code)]."""
    return sorted([c for _s, _p, c in olds if c >= today] + list(codes_new))


def draw_episode(seed, unit):
    """The bay's pose, `today`, the planogram (each lane's SKU), three served lanes of three SKUs (two on one shelf,
    one on another) with their old units and the delivery, the unserved lanes' stock (faced, in code order), the
    tote's contents; every unit with its best-before code. A served lane holds capacity - OLD_DROP (at least 1) to
    capacity old units, some expired; the delivery fills it to capacity once the expired units are gone. At least two
    served lanes need the codes (an expired unit, or a delivered code earlier than an old one). The tote holds the
    delivery, then extras (units of SKUs no served lane carries; they stay in the tote) until MIN_FREE to MAX_FREE
    of its slots are free. Draws that do not fit are drawn again from the same stream."""
    r = rng_of(seed, 2)
    pose = {"dx": float(r.uniform(-0.02, 0.02)), "dy": float(r.uniform(-0.02, 0.02)),
            "yaw": float(r.uniform(-math.radians(5), math.radians(5)))}
    n_slots = TOTE_GRID[0] * TOTE_GRID[1]
    for _attempt in range(1000):
        today = int(r.integers(*TODAY_RANGE))
        plan = {}
        for si, sh in enumerate(unit["shelves"]):
            for li in range(N_LANES):
                plan[(si, li)] = sh["skus"][int(r.integers(2))]
        lanes = list(plan)
        # three lanes of three SKUs: two on one shelf (one lane of each of its SKUs), one on another
        si = int(r.integers(SHELVES))
        pair = [(si, li) for li in r.permutation(N_LANES)]
        a = pair[0]
        b = next((k for k in pair[1:] if plan[k] != plan[a]), None)
        if b is None:
            continue
        lone_shelf = int(r.choice([s for s in range(SHELVES) if s != si]))
        served = [a, b, (lone_shelf, int(r.integers(N_LANES)))]
        served = [served[int(i)] for i in r.permutation(3)]
        stock, stock_codes, old, deliver, new_codes = {}, {}, {}, {}, {}
        need_codes = 0
        for key in lanes:
            sku = unit["skus"][plan[key]]
            cap = capacity(unit, sku)
            if key in served:
                o = int(r.integers(max(1, cap - OLD_DROP), cap + 1))
                expired = [bool(r.random() < P_EXPIRED) for _ in range(o)]
                if o == cap and not any(expired):
                    expired[int(r.integers(o))] = True
                fresh = [int(r.integers(today, today + OLD_SPAN + 1)) for _ in range(o)]
                codes = [int(r.integers(today - EXPIRED_SPAN, today)) if ex else c for ex, c in zip(expired, fresh)]
                n = cap - (o - sum(expired))
                newc = [int(r.integers(today + NEW_LO, today + NEW_HI + 1)) for _ in range(n)]
                fresh_old = [c for c in codes if c >= today]
                if fresh_old and r.random() < P_INVERSION:
                    top = max(fresh_old)
                    newc[int(r.integers(n))] = int(r.integers(today, top)) if top > today else today
                inverted = bool(fresh_old) and min(newc) < max(fresh_old)
                need_codes += bool(any(expired) or inverted)
                slots = sorted(int(s) for s in r.choice(cap, size=o, replace=False))
                order = r.permutation(o)               # codes in the lane in no particular order
                old[key] = [(s, float(r.uniform(0, SLOT_GAP)), codes[int(order[k])]) for k, s in enumerate(slots)]
                deliver[key] = n
                new_codes[key] = newc
            else:
                k = int(r.integers(1, cap + 1))
                stock[key] = k
                stock_codes[key] = sorted(int(r.integers(today, today + OLD_SPAN + 1)) for _ in range(k))
        n_new = sum(deliver.values())
        n_extra = n_slots - n_new - int(r.integers(MIN_FREE, MAX_FREE + 1))
        if need_codes < 2 or n_extra < 0:
            continue
        break
    else:
        raise RuntimeError("no episode drawn")
    tote_slots = [(i, j) for i in range(TOTE_GRID[0]) for j in range(TOTE_GRID[1])]
    picks = r.permutation(len(tote_slots))
    tote, k = [], 0
    for key in served:
        for c in new_codes[key]:
            i, j = tote_slots[int(picks[k])]
            k += 1
            if unit["skus"][plan[key]]["kind"] == "box":
                yaw = float(r.choice([0.0, math.pi]) + r.uniform(-math.radians(8), math.radians(8)))
            else:
                yaw = float(r.uniform(-math.pi, math.pi))
            tote.append({"sku": plan[key], "slot": [i, j], "yaw": yaw, "code": c,
                         "jitter": [float(v) for v in r.uniform(-0.004, 0.004, 2)]})
    # extras: units of the SKUs no served lane carries, not part of the delivery; they stay in the tote
    others = sorted(set(unit["skus"]) - {plan[key] for key in served})
    for _e in range(n_extra):
        i, j = tote_slots[int(picks[k])]
        k += 1
        name = others[int(r.integers(len(others)))]
        if unit["skus"][name]["kind"] == "box":
            yaw = float(r.choice([0.0, math.pi]) + r.uniform(-math.radians(8), math.radians(8)))
        else:
            yaw = float(r.uniform(-math.pi, math.pi))
        tote.append({"sku": name, "slot": [i, j], "yaw": yaw, "code": int(r.integers(today, today + OLD_SPAN + 1)),
                     "jitter": [float(v) for v in r.uniform(-0.004, 0.004, 2)], "extra": True})
    ks = lambda d: {f"{s}:{li}": v for (s, li), v in d.items()}  # noqa: E731
    return {"seed": int(seed), "pose": pose, "today": today,
            "planogram": ks(plan), "served": [f"{s}:{li}" for s, li in served], "old": ks(old),
            "deliver": ks(deliver), "stock": ks(stock), "stock_codes": ks(stock_codes), "tote": tote}


def tote_slot_xy(i, j):
    cx, cy = TOTE["centre"]
    return (cx + (i - (TOTE_GRID[0] - 1) / 2) * TOTE_PITCH[0], cy + (j - (TOTE_GRID[1] - 1) / 2) * TOTE_PITCH[1])


# ----------------------------------------------------------------------------------------------
# The world
# ----------------------------------------------------------------------------------------------
def bay_of(unit, episode):
    p = episode["pose"]
    sh = unit["shelves"]
    width = (N_LANES - 1) * unit["pitch"] + 2 * SIDE_CLEAR
    return S.default_bay(front_x=unit["front_x"] + p["dx"], y=p["dy"], yaw=p["yaw"], width=width,
                         depth=unit["depth"], boards=[s["board"] for s in sh], top_z=unit["top_z"])


def bay_to_world(bay, x, y, z):
    R, p = S.bay_frame(bay)
    return R @ np.array([x, y, 0.0]) + p + np.array([0.0, 0.0, z])


def product_spec(name, sku, pos, yaw):
    return {"name": name, "kind": sku["kind"], "size": sku["size"], "mass": sku["mass"], "pos": tuple(pos),
            "yaw": float(yaw), "rgba": tuple(sku["rgba"]), "friction": 0.6}


def layout(unit, episode):
    """Every product's name, SKU and start pose; returns (specs, meta) with meta[name] = role and lane."""
    bay = bay_of(unit, episode)
    yaw = bay["yaw"]
    specs, meta = [], {}
    rng = rng_of(episode["seed"], 3)
    for key, sku_name in episode["planogram"].items():
        si, li = map(int, key.split(":"))
        sku = unit["skus"][sku_name]
        z = unit["shelves"][si]["board"] + half_h(sku) + 0.0005
        if key in episode["stock"]:
            for k, code in enumerate(episode["stock_codes"][key]):
                x = slot_x(unit, sku, k) + float(rng.uniform(0, 0.004))
                y = lane_y(unit, li) + float(rng.uniform(-0.004, 0.004))
                n = f"u_{si}_{li}_s{k}"
                specs.append(product_spec(n, sku, bay_to_world(bay, x, y, z), yaw + rng.uniform(-0.05, 0.05)))
                meta[n] = {"role": "stock", "lane": key, "sku": sku_name, "code": code}
        for k, (slot, push, code) in enumerate(episode["old"].get(key, [])):
            x = slot_x(unit, sku, slot) + push
            y = lane_y(unit, li) + float(rng.uniform(-0.004, 0.004))
            n = f"u_{si}_{li}_o{k}"
            # a box stands face out; a can or bottle at any turn (its code may face the back)
            a = yaw + rng.uniform(-0.05, 0.05) if sku["kind"] == "box" else rng.uniform(-math.pi, math.pi)
            specs.append(product_spec(n, sku, bay_to_world(bay, x, y, z), a))
            meta[n] = {"role": "old", "lane": key, "sku": sku_name, "code": code}
    for k, t in enumerate(episode["tote"]):
        sku = unit["skus"][t["sku"]]
        x, y = tote_slot_xy(*t["slot"])
        z = CART["top"] + half_h(sku) + 0.0005
        n = f"u_t{k}"
        specs.append(product_spec(n, sku, (x + t["jitter"][0], y + t["jitter"][1], z), t["yaw"]))
        if t.get("extra"):
            meta[n] = {"role": "extra", "lane": None, "sku": t["sku"], "code": t["code"]}
            continue
        lane = next(key for key in episode["served"] if episode["planogram"][key] == t["sku"])
        meta[n] = {"role": "new", "lane": lane, "sku": t["sku"], "code": t["code"]}
    rc = rng_of(episode["seed"], 4)                    # the codes' places: a stream of their own
    for spc in specs:
        m = meta[spc["name"]]
        m["expired"] = m["code"] < episode["today"]
        m["code_at"] = draw_code_at(rc, unit["skus"][m["sku"]])
    return specs, meta


def _visual(body, gtype, size, pos, rgba, quat=None):
    g = body.add_geom()
    g.type = gtype
    g.size = list(size)
    g.pos = list(pos)
    if quat is not None:
        g.quat = list(quat)
    g.rgba = list(rgba) + [1.0] if len(rgba) == 3 else list(rgba)
    g.contype = 0
    g.conaffinity = 0
    g.mass = B.VISUAL_MASS
    g.group = 1
    return g


def add_look(body, sku):
    """A SKU's printed look on a product body (visual geoms; see the module's doc)."""
    look = sku["look"]
    e = 0.0008
    BOX, CYL = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    roll_x = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)     # a cylinder's axis along x
    hh = half_h(sku)
    zl = look["logo_z"] * hh
    if sku["kind"] == "box":
        hx, hy, _ = sku["size"]
        _visual(body, BOX, (hx + e, hy + e, look["band_h"] * hh), (0, 0, zl), look["band"])
        faces = [(-hx - 2 * e, hy)]                                     # the printed front only
    else:
        r = sku["size"][0]
        _visual(body, CYL, (r + e, look["band_h"] * hh, 0), (0, 0, zl), look["band"])
        hx, hy = r, r
        faces = [(-r - 2 * e, r), (r + 2 * e, r)]
        if sku["kind"] == "cyl" and hh > 0.08:                           # a bottle: its cap
            _visual(body, CYL, (r * 0.45, 0.012, 0), (0, 0, hh + 0.012), look["cap"])
    s = 0.45 * min(hy, look["band_h"] * hh + 0.01)
    for x, _w in faces:
        if look["logo_shape"] == "square":
            _visual(body, BOX, (e, s, s), (x, 0, zl), look["logo"])
        elif look["logo_shape"] == "disc":
            _visual(body, CYL, (s, e, 0), (x, 0, zl), look["logo"], quat=roll_x)
        else:
            for dy in (-0.5 * s, 0.5 * s):
                _visual(body, BOX, (e, 0.22 * s, s), (x, dy, zl), look["logo"])


# Seven segments: a (top), b (upper right), c (lower right), d (bottom), e (lower left), f (upper left), g (middle)
SEGMENTS = {0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc", 5: "afgcd", 6: "afgedc", 7: "abc",
            8: "abcdefg", 9: "abcdfg"}
CODE_DIGIT_H = 0.007          # m, a digit's height; every other code size scales with it (see `code_sizes`)
CODE_INK = (0.08, 0.08, 0.10)
CODE_PAPER = (0.96, 0.96, 0.93)
CODE_BELOW_TOP = 0.040        # m, the patch's top at least this far under the unit's top (clear of the fingers of a
                              # grasp from above; at 22-33 mm under the top a finger covered a code turned to it)
CODE_SPAN = 0.7               # the patch's centre lies in the unit's upper CODE_SPAN of its height (body, not cap)
CODE_LOGO_CLEAR = 0.002       # m, between the patch and a logo


def code_sizes(digit_h=None):
    """The code's sizes (m) for a digit height: digit h, w, stroke, the gap between the digits, the patch's half
    sizes (across, up). The proportions are those of the v1-v3 code (14 mm digits, 8 mm wide, 1.6 mm strokes, 4 mm
    apart, on a 23 x 20 mm patch)."""
    h = CODE_DIGIT_H if digit_h is None else digit_h
    k = h / 0.014
    return {"h": h, "w": 0.008 * k, "stroke": 0.0016 * k, "gap": 0.004 * k, "patch": (0.0115 * k, 0.0100 * k)}


def logo_box(sku):
    """A SKU's logo as (half size s, centre height): a square of half side s (a disc or bars lie inside it) on a
    box's -x side, or on a can's or bottle's -x and +x sides (`add_look`)."""
    look = sku["look"]
    hh = half_h(sku)
    hy = sku["size"][1] if sku["kind"] == "box" else sku["size"][0]
    return 0.45 * min(hy, look["band_h"] * hh + 0.01), look["logo_z"] * hh


def draw_code_at(rng, sku, digit_h=None):
    """Where a unit's code sits (its own frame): `face_yaw` turns the -x side's layout about z (a box: one of its
    four sides; a can or bottle: any angle), `offset` across the face (to the reader's right, a box only), `z` the
    patch's centre height: in the unit's upper CODE_SPAN, the patch's top CODE_BELOW_TOP or more under the unit's
    top. The patch never covers the logo (drawn again until it does not)."""
    c = code_sizes(digit_h)
    pu, pv = c["patch"]
    hh = half_h(sku)
    s, zl = logo_box(sku)
    for _ in range(1000):
        z = float(rng.uniform(hh * (1 - 2 * CODE_SPAN) + pv, hh - CODE_BELOW_TOP - pv))
        level = abs(z - zl) < s + pv + CODE_LOGO_CLEAR           # level with the logo
        if sku["kind"] == "box":
            side = int(rng.integers(4))
            half_w = sku["size"][1] if side % 2 == 0 else sku["size"][0]
            room = max(0.0, half_w - pu - 0.002)
            at = {"face_yaw": side * math.pi / 2, "offset": float(rng.uniform(-room, room)), "z": z}
            if not (level and side == 0 and abs(at["offset"]) < s + pu + CODE_LOGO_CLEAR):
                return at
        else:
            th = float(rng.uniform(-math.pi, math.pi))
            r = sku["size"][0]
            apart = min(abs(th), math.pi - abs(th))                # from the logos at 0 and pi
            if not (level and apart < math.atan((s + pu) / r) + 0.15):
                return {"face_yaw": th, "offset": 0.0, "z": z}
    raise RuntimeError("no place for the code")


def add_code(body, sku, code, at, digit_h=None):
    """A unit's best-before code: two seven-segment digits, dark on a light patch, on one side at `at`
    (`draw_code_at`); read with the unit upright, facing that side. Laid out as on the -x side (the reader looks
    along +x, their right is the unit's -y), then turned by `at["face_yaw"]` about z. The patch stands 2.4-4 mm and
    the digits 4-4.8 mm proud of the side, clear of the band and the logo."""
    BOX = mujoco.mjtGeom.mjGEOM_BOX
    c = code_sizes(digit_h)
    e = 0.0008
    th = at["face_yaw"]
    ct, st = math.cos(th), math.sin(th)
    quat = (math.cos(th / 2), 0.0, 0.0, math.sin(th / 2))
    if sku["kind"] == "box":
        hx = sku["size"][0] if at["face_yaw"] % math.pi < 1e-9 else sku["size"][1]
        depth = e                                      # the patch on the flat side
    else:
        hx = sku["size"][0]
        depth = 0.0035                                 # a flat label across the curve, its back inside the body
    x_face = -hx - 4 * e
    zc = at["z"]

    def put(size, x, y, z, rgba):
        _visual(body, BOX, size, (ct * x - st * y, st * x + ct * y, z), rgba, quat=quat)
    pu, pv = c["patch"]
    y0 = -at["offset"]
    put((depth, pu, pv), x_face + depth - e, y0, zc, CODE_PAPER)
    h, w, t = c["h"], c["w"], c["stroke"]
    seg = {"a": (0, h / 2, w / 2, t / 2), "d": (0, -h / 2, w / 2, t / 2), "g": (0, 0, w / 2, t / 2),
           "b": (w / 2, h / 4, t / 2, h / 4), "c": (w / 2, -h / 4, t / 2, h / 4),
           "e": (-w / 2, -h / 4, t / 2, h / 4), "f": (-w / 2, h / 4, t / 2, h / 4)}
    xs = x_face - 1.5 * e
    for i, digit in enumerate((code // 10 % 10, code % 10)):
        u0 = (i - 0.5) * (w + c["gap"])                # left digit first as seen from the face
        for name in SEGMENTS[digit]:
            du, dv, su, sv = seg[name]
            put((e / 2, su, sv), xs, y0 - (u0 + du), zc + dv, CODE_INK)


def add_cart(wb):
    cx, cy = CART["centre"]
    hx, hy = CART["half"]
    S.add_box(wb, "cart", (cx, cy, CART["top"] / 2), (hx, hy, CART["top"] / 2), (0.35, 0.35, 0.40, 1))
    tx, ty = TOTE["centre"]
    ix, iy = TOTE["inner_half"]
    w, h = TOTE["wall"], TOTE["height"]
    z = CART["top"] + h / 2
    col = (0.20, 0.30, 0.55, 1)
    S.add_box(wb, "tote_n", (tx, ty + iy + w / 2, z), (ix + w, w / 2, h / 2), col)
    S.add_box(wb, "tote_s", (tx, ty - iy - w / 2, z), (ix + w, w / 2, h / 2), col)
    S.add_box(wb, "tote_e", (tx + ix + w / 2, ty, z), (w / 2, iy, h / 2), col)
    S.add_box(wb, "tote_w", (tx - ix - w / 2, ty, z), (w / 2, iy, h / 2), col)


CAMS = {
    # name: (bitmask, width, height, fovy, mount, position, look-at) ; mount "carriage" = relative to the lift's base
    "left": (1, 400, 300, 60.0, "carriage", (0.15, 0.40, 0.42), (0.66, 0.0, 0.28)),
    "right": (2, 400, 300, 60.0, "carriage", (0.15, -0.40, 0.42), (0.66, 0.0, 0.28)),
    "wrist": (4, 256, 192, 75.0, None, None, None),
    "overview": (8, 640, 480, 55.0, "world", (-0.55, 0.55, 1.75), (0.66, 0.0, 0.85)),
}
ALL_CAMS = 15


def build(unit, episode):
    """Compile the episode's world. Returns (model, specs, meta, bay)."""
    bay = bay_of(unit, episode)
    specs, meta = layout(unit, episode)
    w = S.base_world(bay, specs, lift=True)
    add_cart(w.worldbody)
    carriage = w.body("carriage")
    for name, (_bit, _w, _h, fovy, mount, pos, tgt) in CAMS.items():
        if mount is None:
            continue
        parent = carriage if mount == "carriage" else w.worldbody
        c = parent.add_camera()
        c.name = name
        c.pos = list(pos)
        c.quat = list(B.quat_from_mat(B.look_at(pos, tgt)))
        c.fovy = fovy
    for spc in specs:
        sku = unit["skus"][meta[spc["name"]]["sku"]]
        add_look(w.body(spc["name"]), sku)
        add_code(w.body(spc["name"]), sku, meta[spc["name"]]["code"], meta[spc["name"]]["code_at"])
    m = S.finish(w.compile(), specs)
    m.site_rgba[:, 3] = 0.0                     # the grip site's visual line and other markers hidden
    return m, specs, meta, bay


class World(LiftCtl):
    """The episode's simulation with the controller on the lift, and a joint step."""

    def __init__(self, unit, episode, lift_z=TOTE_BASE):
        self.unit, self.episode = unit, episode
        m, self.specs, self.meta, self.bay = build(unit, episode)
        super().__init__(m, lift_z)
        self.bodies = {s["name"]: m.body(s["name"]).id for s in self.specs}
        self.sku_of = {s["name"]: self.meta[s["name"]]["sku"] for s in self.specs}
        self.start_pos = {n: self.data.xpos[b].copy() for n, b in self.bodies.items()}
        self.home_pos = self.ee()
        self.home_R = self.ee_R()

    def settle(self, steps=20):
        for _ in range(steps):
            self.step(np.zeros(3), np.zeros(3), -1)
        self.start_pos = {n: self.data.xpos[b].copy() for n, b in self.bodies.items()}

    def joint_step(self, q_target, grip, dlift=0.0):
        m, d = self.model, self.data
        q = d.qpos[self.jq].copy()
        qd = q + np.clip(np.asarray(q_target, float) - q, -MAX_DQ, MAX_DQ)
        self.q_null = np.asarray(q_target, float).copy()
        gt = 0.04 * (1.0 - float(np.clip(grip, -1, 1))) / 2.0
        z0 = self.lift_set
        z1 = float(np.clip(z0 + np.clip(dlift, -MAX_LIFT, MAX_LIFT), *S.LIFT_RANGE))
        self.lift_set = z1
        for i in range(B.SUBSTEPS):
            d.ctrl[self.lift_act] = self._lift_force(z0 + (z1 - z0) * (i + 1) / B.SUBSTEPS)
            d.ctrl[self.act] = 200.0 * (qd - d.qpos[self.jq]) - 20.0 * d.qvel[self.jv] + d.qfrc_bias[self.jv]
            d.ctrl[self.gact] = [gt, -gt]
            mujoco.mj_step(m, d)
        self.t += 1
        self.target_pos = d.site_xpos[self.site].copy()
        self.target_R = d.site_xmat[self.site].reshape(3, 3).copy()
        if not np.all(np.isfinite(d.qpos)):
            raise FloatingPointError("unstable")

    # --- state ---------------------------------------------------------------------------------------
    def pos(self, n):
        return self.data.xpos[self.bodies[n]].copy()

    def rot(self, n):
        return self.data.xmat[self.bodies[n]].reshape(3, 3).copy()

    def yaw(self, n):
        R = self.rot(n)
        return math.atan2(R[1, 0], R[0, 0])

    def tilt(self, n):
        return math.degrees(math.acos(max(-1.0, min(1.0, float(self.rot(n)[2, 2])))))

    def to_bay(self, p):
        R, pb = S.bay_frame(self.bay)
        return R.T @ (np.asarray(p, float) - pb)

    def unstable(self):
        d = self.data
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return True
        if np.max(np.abs(d.qvel[self.jv])) > B.MAX_ARM_SPEED:
            return True
        return any(np.linalg.norm(d.cvel[b][3:]) > B.MAX_OBJECT_SPEED for b in self.bodies.values())
