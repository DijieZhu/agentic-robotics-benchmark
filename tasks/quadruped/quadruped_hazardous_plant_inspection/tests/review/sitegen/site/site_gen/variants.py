"""Version 2's factors by candidate number, the obstacles off the drawing and the practice rule.

The table by candidate number, the same for every label (hidden, practice, doctest and development labels):

    (k - 1) mod 8   0       1       2       3       4       5       6       7
    bund kerb       low     high    low     high    low     high    low     high
    B-step          sills   room    room    sills   sills   room    room    sills
    B-low           chase   rack    chase   rack    rack    chase   rack    chase

and the missing sheet's place along / across / diagonal for (k - 1) mod 3 = 0, 1, 2. Types: slot i (in SLOT_ORDER)
gets TYPE_ORDER[(i + k + (k - 1) // 8) mod 8], then the repair pass over the measured exclusion list
(config/exclusion_v2.json; assignment_v2), which replaced version 1's single (B-low, level gauge) swap.

The two in-room obstacles, built into a decor room with one door:
  strip   B-step: a hose (0.4), a round floor pipe (0.3) or a cable trench cover (0.3: a plate strip 0.20 m deep, the
          bund kerb's section) from wall to wall parallel to the door wall, its near edge 2.2-2.8 m from it, sizes in
          the firmware's rated needs_high_step bands with the 15 % margin (hose, cover 4.95-5.65 cm; pipe 5.01-5.99 cm);
  rack    B-low: a pipe bridge from wall to wall parallel to the door wall, 0.8-1.2 m long along the way, the rated
          crouch clearance (0.283-0.297 m) under cross beams at both ends (hazard striped), its pipes stacked in
          staggered layers up to 1.7 m under a cable tray, so that it screens the back of the room.
Each has 1.0 m of flat floor before and behind it; that band, wall to wall, is a keep-out before the room is
furnished, and no mount spot, puddle, narrow gap or other element lies in it (row approach_flat).
"""

from __future__ import annotations

import itertools
import json
import math
import os

from .furnish import frame_for
from .util import uniform, weighted_key
from .world import sample_size

SLOT_ORDER = ("A", "B-step", "B-kerb", "B-low", "C-sealed", "C-moved", "C-relocated", "C-annex")
TYPE_ORDER = ("gauge_63", "gauge_100", "gauge_160", "level", "valve_lever", "valve_handwheel", "lamps", "thermal")
BUND = ("low", "high", "low", "high", "low", "high", "low", "high")
B_STEP = ("door_sills", "in_room", "in_room", "door_sills", "door_sills", "in_room", "in_room", "door_sills")
B_LOW = ("chase", "rack", "chase", "rack", "rack", "chase", "rack", "chase")
PLACES = ("along", "across", "diagonal")
BUND_CLASS = {"low": "needs_high_step", "high": "barrier"}
DEVIATION_CLASSES = ("door_sealed", "instrument_moved", "instrument_relocated", "annex_added", "zoning_outdated",
                     "repeater_dead", "opening_as_built", "opening_added", "partition_added")


# ============================================================================================ the table
def table_row(k: int) -> dict:
    """The factors of candidate k (1-based)."""
    k = int(k)
    if k < 1:
        raise ValueError("candidate numbers start at 1")
    c = (k - 1) % 8
    return {"k": k, "column": c, "bund": BUND[c], "b_step": B_STEP[c], "b_low": B_LOW[c],
            "sheet_place": PLACES[(k - 1) % 3]}


# ============================================================================================ the type rule
# Since 2026-11-16: types by the first formula, then a repair pass. The exclusion list
# holds the (slot, the slot's variant, the sheet's place where it matters, type) pairs that pass their placement
# stage on fewer than one sub-seed in five, measured on at least 20 sub-seeds of development labels; it replaces
# version 1's single swap and is written into selection.json with the table. Its one copy is config/exclusion_v2.json
# (the list with the measured rates; a development test checks the list against the rates).
EXCLUSION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                              "exclusion_v2.json")


def exclusion_record() -> dict:
    """The exclusion list's file (config/exclusion_v2.json): {"rule", "measured", "exclusion": [{slot, variant,
    place, type, ok, n, rate}], ...}; {"exclusion": []} when it does not exist."""
    if not os.path.exists(EXCLUSION_FILE):
        return {"exclusion": []}
    with open(EXCLUSION_FILE) as f:
        return json.load(f)


def exclusion_list() -> tuple:
    """((slot, variant, place, type), ...): variant '-' for a slot without one (A, C-*), place '*' where the sheet's
    place does not matter."""
    return tuple((str(e["slot"]), str(e["variant"]), str(e["place"]), str(e["type"]))
                 for e in exclusion_record().get("exclusion", []))


def slot_variant(slot, row) -> str:
    """The slot's own variant in table row `row`: B-step's (door sills / in the room), B-kerb's (the bund class),
    B-low's (chase / rack); '-' for the others."""
    return {"B-step": row["b_step"], "B-kerb": row["bund"], "B-low": row["b_low"]}.get(slot, "-")


def excluded(slot, typ, row, exclusion=None) -> bool:
    """Is (slot, the slot's variant in `row`, the row's sheet place, typ) on the exclusion list?"""
    ex = exclusion_list() if exclusion is None else exclusion
    var = slot_variant(slot, row)
    return any(s == slot and t == typ and v in ("-", "*", var) and p in ("*", row["sheet_place"])
               for s, v, p, t in ex)


def assignment_v2(k: int, b_low=None, exclusion=None, record=None) -> dict:
    """{slot: type} for candidate k: slot i -> TYPE_ORDER[(i + k + (k - 1) // 8) mod 8]; then in the fixed slot
    order A, B-step, B-kerb, B-low, C-sealed, C-moved, C-relocated, C-annex, a slot whose (slot, variant, place, type)
    is on the exclusion list swaps its type with the type of the first later slot in that order (wrapping round) for
    which neither slot is then on the list. `record` (a list): the swaps made, [(slot, other slot)]. b_low: the B-low
    variant (default the table's)."""
    k = int(k)
    row = dict(table_row(k))
    if b_low:
        row["b_low"] = b_low
    ex = exclusion_list() if exclusion is None else tuple(exclusion)
    a = {s: TYPE_ORDER[(i + k + (k - 1) // 8) % len(TYPE_ORDER)] for i, s in enumerate(SLOT_ORDER)}
    n = len(SLOT_ORDER)
    for i, s in enumerate(SLOT_ORDER):
        if not excluded(s, a[s], row, ex):
            continue
        for d in range(1, n):
            o = SLOT_ORDER[(i + d) % n]
            if not excluded(s, a[o], row, ex) and not excluded(o, a[s], row, ex):
                a[s], a[o] = a[o], a[s]
                if record is not None:
                    record.append((s, o))
                break
        else:
            if record is not None:
                record.append((s, None))           # no partner: the pair stays (reported; never seen)
    return a


def table(ks) -> list:
    """The table for a list of candidate numbers (what selection.json records before any run): the factors, the
    assignment after the repair pass and the swaps it made."""
    out = []
    for k in ks:
        r = table_row(k)
        rec = []
        r["assignment"] = assignment_v2(k, r["b_low"], record=rec)
        r["repairs"] = [list(x) for x in rec]
        out.append(r)
    return out


def selection_tables(ks) -> dict:
    """The block for selection.json: the exclusion list with its measured rates, the rule's words and the table."""
    ex = exclusion_record()
    return {"type_rule": "slot i gets TYPE_ORDER[(i + k + floor((k - 1) / 8)) mod 8] (TYPE_ORDER %s, slots %s); then, "
                         "in that slot order, a slot whose (slot, its variant, the sheet's place where it matters, "
                         "type) is on the exclusion list swaps its type with the type of the first later slot "
                         "(wrapping round) for which neither slot is then on the list (ruling R1)" %
                         (list(TYPE_ORDER), list(SLOT_ORDER)),
            "exclusion": ex.get("exclusion", []), "exclusion_measured": ex.get("measured"),
            "exclusion_rule": ex.get("rule"), "table": table(ks)}


def v2_config(k: int, label: str) -> dict:
    """The `ladder.v2` block a candidate is generated with (sets.py passes it as a config override)."""
    r = table_row(k)
    return {"k": int(k), "label": str(label), "bund": r["bund"], "b_step": r["b_step"], "b_low": r["b_low"],
            "sheet_place": r["sheet_place"]}


# ============================================================================================ practice rule
def practice_features(rec: dict) -> dict:
    """The features the practice rule asks of a practice building, from a candidate record (sets._one's record, which carries
    the site's own bund class, variants, missing sheet place and deviation classes)."""
    return {"bund": rec.get("bund"), "b_step": rec.get("b_step_variant"), "b_low": rec.get("b_low_variant"),
            "sheet_place": rec.get("sheet_place"), "deviations": set(rec.get("deviations") or [])}


def qualifies(rec: dict, min_score=0.80) -> bool:
    """Generated with every required row, and the frozen reference (run as the grader runs the oracle)
    delivered all ten items with accepted evidence and scored at least 0.80. `rec["reference"]` is the selection
    stage's record of that run: {"all_delivered": bool, "score": float} (absent: not qualified)."""
    if not (rec.get("generated") and rec.get("proofs_ok")):
        return False
    ref = rec.get("reference") or {}
    return bool(ref.get("all_delivered")) and float(ref.get("score", 0.0)) >= float(min_score)


def triple_ok(feats) -> bool:
    """The three buildings together show both bund classes, both B-step and both B-low variants, three different
    places of the missing sheet and every deviation class."""
    return (len({f["bund"] for f in feats}) == 2 and len({f["b_step"] for f in feats}) == 2 and
            len({f["b_low"] for f in feats}) == 2 and len({f["sheet_place"] for f in feats}) == 3 and
            set(DEVIATION_CLASSES) <= set().union(*[f["deviations"] for f in feats]))


def select_practice(records, qualify=qualifies) -> dict:
    """The practice rule: t1, t2, t3 are the first triple i < j < l, in the order (i, j, l)
    (lexicographic over the candidate numbers), of qualifying candidates whose buildings together show both bund
    classes, both B-step variants, both B-low variants, three different places of the missing sheet and every
    deviation class. Returns {"t1", "t2", "t3": records, "k": (i, j, l), "passed_over": [k of qualifying candidates
    before l that are not in the triple], "not_qualified": [k], "classes": [...]} or {"t1": None, ...} when no triple
    of the given candidates qualifies (the selection then reports it; the rule is not changed)."""
    recs = sorted(records, key=lambda r: int(r["k"]))
    ok = [r for r in recs if qualify(r)]
    notq = [int(r["k"]) for r in recs if not qualify(r)]
    feats = {int(r["k"]): practice_features(r) for r in ok}
    for t in itertools.combinations(ok, 3):
        if triple_ok([feats[int(r["k"])] for r in t]):
            ks = tuple(int(r["k"]) for r in t)
            passed = [int(r["k"]) for r in ok if int(r["k"]) < ks[2] and int(r["k"]) not in ks]
            classes = sorted(set().union(*[feats[k]["deviations"] for k in ks]))
            return {"t1": t[0], "t2": t[1], "t3": t[2], "k": ks, "passed_over": passed, "not_qualified": notq,
                    "classes": classes}
    return {"t1": None, "t2": None, "t3": None, "k": None, "passed_over": [int(r["k"]) for r in ok],
            "not_qualified": notq, "classes": []}


# ============================================================================================ rooms for the variants
def door_leaf_into(lay, room):
    """Length of the leaf that swings into `room` from its one door (0 when it swings out)."""
    d = lay.door(room.doors[0])
    if d.swing_into != room.id:
        return 0.0
    return d.width / 2.0 if d.kind == "double" else d.width


def inroom_candidates(lay, min_depth, exclude=()):
    """Decor rooms (change room, store, workshop, plant room, decontamination room, laboratory, office) with exactly
    one door, a drawn door on their front wall, and at least `min_depth` m of floor behind the door wall:
    [(room id, depth)] by room id."""
    out = []
    for r in sorted(lay.rooms.values(), key=lambda r: r.id):
        if r.type != "decor" or r.id in exclude or not r.on_drawing:
            continue
        if len(r.doors) != 1:
            continue
        d = lay.door(r.doors[0])
        if not d.on_drawing or d.kind not in ("door", "double"):
            continue
        f = frame_for(r)
        if f.Lv + 1e-6 < float(min_depth):
            continue
        out.append((r.id, round(float(f.Lv), 3)))
    return out


# the far end of each type's size rule and its largest angle off the face (acceptance_v1.yaml): how far
# behind a 5 cm strip an instrument must hang so that no camera in front of the strip gets an accepted frame (the
# proofs decide; this only ranks the rooms, "deeper rooms first")
SIZE_RULE = {"gauge_63": (2.97, 45.0), "gauge_100": (4.72, 45.0), "gauge_160": (5.0, 45.0), "level": (4.04, 30.0),
             "valve_lever": (5.0, 45.0), "valve_handwheel": (3.51, 45.0), "lamps": (5.0, 55.0), "thermal": (5.0, 60.0)}


def strip_slack(Lu, Lv, typ, d=0.06, a0=2.2, approach=1.0):
    """How much farther from the door wall than a0 the strip's near edge may lie in a room Lu wide and Lv deep while
    an instrument of type `typ` still finds a hidden place behind it: on the back wall when it is beyond the size
    rule's far end from the strip, or on a side wall far enough behind the strip that every view from the front part
    is too oblique or too far. Negative: no such place (estimate)."""
    D, th = SIZE_RULE.get(typ, (5.0, 60.0))
    t = math.radians(th)
    back = Lv - 0.5 - D - a0
    s_need = max(approach + 0.3, min(math.tan(t) * max(Lu - 0.3, 0.1), math.sin(t) * D) - d)
    side = Lv - 0.5 - s_need - d - a0
    return max(back, side)


# ============================================================================================ builders
def _band_world(f, v0, v1, Lu):
    return f.lrect_to_world(0.0, v0, Lu, v1)


def plan_strip(lay, rid, cfg, g, typ=None):
    """The B-step strip of room `rid` (before it is built): its kind, size, near edge and band (local v). With the
    slot's type, the near edge is drawn from the part of 2.2-2.8 m that still leaves the instrument a hidden place
    (strip_slack)."""
    S = cfg["ladder"]["v2_params"]["strip"]
    f = frame_for(lay.rooms[rid])
    kind = weighted_key(g, S["kind_weights"])
    a0, a1 = [float(v) for v in S["near_m"]]
    if typ is not None:
        a1 = max(a0, min(a1, a0 + strip_slack(f.Lu, f.Lv, typ, a0=a0, approach=float(S["approach_m"]))))
    a = round(uniform(g, a0, a1) / 0.01) * 0.01
    if kind == "floor_pipe":
        h = sample_size(cfg, g, "floor_pipe", "needs_high_step")
        dd = h
    elif kind == "hose":
        h = sample_size(cfg, g, "hose", "needs_high_step")
        dd = h
    else:
        h = sample_size(cfg, g, "trench_cover", "needs_high_step")
        dd = float(S["cover_depth_m"])
    ap = float(S["approach_m"])
    return {"variant": "in_room", "slot": "B-step", "room": rid, "kind": kind, "size_m": float(h),
            "near_m": float(a), "depth_m": float(dd), "far_m": float(a + dd), "approach_m": ap,
            "band_local": [0.0, a - ap, f.Lu, a + dd + ap], "Lu": float(f.Lu), "Lv": float(f.Lv),
            "band_world": _band_world(f, a - ap, a + dd + ap, f.Lu)}


def plan_rack(lay, rid, cfg, g):
    """The B-low pipe bridge of room `rid` (before it is built): length, clearance, near edge, band."""
    R = cfg["ladder"]["v2_params"]["rack"]
    f = frame_for(lay.rooms[rid])
    L0, L1 = [float(v) for v in R["length_m"]]
    L = round(uniform(g, L0, L1) / 0.05) * 0.05
    clr = sample_size(cfg, g, "low_overhead", "needs_crouch")
    ap = float(R["approach_m"])
    a_lo = float(R["near_min_m"])
    a_hi = min(float(R["near_max_m"]), f.Lv - L - float(R["back_min_m"]))
    if a_hi < a_lo - 1e-9:
        return None
    a = round(uniform(g, a_lo, max(a_lo, a_hi)) / 0.05) * 0.05
    return {"variant": "rack", "slot": "B-low", "room": rid, "kind": "low_overhead", "clearance_m": float(clr),
            "length_m": float(L), "near_m": float(a), "depth_m": float(L), "far_m": float(a + L), "approach_m": ap,
            "band_local": [0.0, a - ap, f.Lu, a + L + ap], "Lu": float(f.Lu), "Lv": float(f.Lv),
            "band_world": _band_world(f, a - ap, a + L + ap, f.Lu)}


def keep_out(W, plan):
    """The band (wall to wall, with its two approaches) is a keep-out of the room's furnishing (call after the
    structure, before furnish_all)."""
    W.keepouts.setdefault(plan["room"], []).append(list(plan["band_world"]))


def build_strip(W, plan, g):
    """Build the in-room strip of `plan` (a hose, a round floor pipe or a trench cover across the room); a tagged
    needs_high_step element (geom group 3). Returns the element."""
    lay = W.lay
    room = lay.rooms[plan["room"]]
    f = frame_for(room)
    Lu = f.Lu
    a, h, dd = plan["near_m"], plan["size_m"], plan["depth_m"]
    body = "elem/%s/strip" % room.id
    P = W.P
    prims = []
    if plan["kind"] == "hose":
        vc = a + h / 2
        prims.append(P.cyl("%s/hose" % body, f.w(0.02, vc, h / 2), f.w(Lu - 0.02, vc, h / 2), h / 2, kind="capsule",
                           mat="hose_red" if g.random() < 0.5 else "hose_black", body=body, owner=room.id,
                           friction=0.9))
        lr = [0.0, a - 0.02, Lu, a + h + 0.02]
        note = "hose across the room (in-room strip, B-step)"
    elif plan["kind"] == "floor_pipe":
        vc = a + h / 2
        prims.append(P.cyl("%s/pipe" % body, f.w(-0.10, vc, h / 2), f.w(Lu + 0.10, vc, h / 2), h / 2,
                           mat="pipe_dr", body=body, owner=room.id, friction=0.9))
        # no flanges (2026-11-16): a floor strip carries no visual part larger than what collides (the
        # two visual-only flanges of 1.35 x the radius made depth show a 7 cm obstacle where physics has 5-6 cm)
        lr = [0.0, a, Lu, a + h]
        note = "drain line across the room on the floor (in-room strip, B-step)"
    else:
        lo = f.w(-0.02, a, 0.0)
        hi = f.w(Lu + 0.02, a + dd, h)
        p0 = [min(lo[0], hi[0]), min(lo[1], hi[1]), 0.0]
        p1 = [max(lo[0], hi[0]), max(lo[1], hi[1]), h]
        prims.append(P.box_aabb("%s/cover" % body, p0, p1, mat="steel_plate", body=body, owner=room.id,
                                friction=0.8))
        lr = [0.0, a, Lu, a + dd]
        note = "cable trench cover across the room (in-room strip, B-step)"
    r = f.lrect_to_world(*lr)
    poly = [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]
    param = {"height": float(h)} if plan["kind"] != "floor_pipe" else {"diameter": float(h)}
    if plan["kind"] == "trench_cover":
        param["depth"] = float(dd)
    param["in_room"] = True
    e = W.element(plan["kind"], "needs_high_step", param, room.id, poly, note=note)
    W.tag(e, [p for p in prims if p is not None])
    plan["element"] = e.id
    return e


SERVICES_RACK = ("cw", "rw", "st", "dw", "ia", "fw", "dr")


def build_rack(W, plan, g):
    """Build the pipe bridge of `plan`: cross beams at both ends (hazard striped) and transverse supports, their
    underside at the recorded clearance; pipes from wall to wall in staggered layers up to 1.7 m under a cable tray;
    hangers at the walls. The beams, supports and every pipe below 0.5 m are the needs_crouch element's geoms (group
    3), as in the pipe chase. Returns the element."""
    from .furnish import PIPE_MAT
    lay = W.lay
    room = lay.rooms[plan["room"]]
    f = frame_for(room)
    Lu = f.Lu
    a, L, clr = plan["near_m"], plan["length_m"], plan["clearance_m"]
    R = W.cfg["ladder"]["v2_params"]["rack"]
    top = float(R["top_m"])
    body = "elem/%s/rack" % room.id
    P = W.P
    bt = 0.05
    elem = []

    def box_l(name, u0, v0, z0, u1, v1, z1, mat, group=0):
        lo = f.w(u0, v0, z0)
        hi = f.w(u1, v1, z1)
        return P.box_aabb("%s/%s" % (body, name), [min(lo[0], hi[0]), min(lo[1], hi[1]), z0],
                          [max(lo[0], hi[0]), max(lo[1], hi[1]), z1], mat=mat, body=body, owner=room.id, group=group)
    # cross beams at both ends and the transverse supports between them: their underside is the clearance
    elem.append(box_l("beam_front", -0.05, a, clr, Lu + 0.05, a + 0.06, clr + bt, "hazard_stripe"))
    elem.append(box_l("beam_back", -0.05, a + L - 0.06, clr, Lu + 0.05, a + L, clr + bt, "hazard_stripe"))
    ns = max(2, int(math.ceil(Lu / 1.2)) + 1)
    for k in range(ns):
        u = 0.10 + k * (Lu - 0.20) / (ns - 1)
        elem.append(box_l("support%d" % k, u - 0.025, a + 0.06, clr, u + 0.025, a + L - 0.06, clr + bt, "galv"))
    for k, (u0, u1) in enumerate(((0.0, 0.04), (Lu - 0.04, Lu))):
        for j, vv in enumerate((a + 0.01, a + L - 0.05)):
            box_l("hanger%d%d" % (k, j), u0, vv, clr, u1, vv + 0.04, 2.4, "galv")
    # pipes in staggered layers, wall to wall
    z_bot = clr + bt
    layer = 0
    n_pipes = 0
    while True:
        r = round(uniform(g, float(R["pipe_r_m"][0]), float(R["pipe_r_m"][1])), 3)
        zc = z_bot + r
        if zc + r > top - 0.06 + 1e-9:
            break
        pitch = 2 * r + 0.04
        n = max(2, int((L - 0.02) // pitch))
        span = (n - 1) * pitch
        v_first = a + 0.5 * (L - span)
        # even layers at v_first + i * pitch, odd layers half a pitch off (between the pipes below and above)
        pos = [v_first + i * pitch for i in range(n)] if layer % 2 == 0 else \
            [v_first + (i + 0.5) * pitch for i in range(-1, n)]
        pos = [vv for vv in pos if vv - r >= a + 0.005 and vv + r <= a + L - 0.005]
        for i, vv in enumerate(pos):
            sv = SERVICES_RACK[(i + 2 * layer) % len(SERVICES_RACK)]
            ps = P.pipe("%s/l%d_%d" % (body, layer, i), [f.w(-0.10, vv, zc), f.w(Lu + 0.10, vv, zc)], r,
                        PIPE_MAT[sv], body, owner=room.id, elbows=False, flange_every=2.0)
            n_pipes += 1
            elem += [p for p in ps if p.group in (0, 3) and p.zrange()[0] < 0.5]
        z_bot = zc + 0.6 * r                  # the next layer overlaps this one's vertical extent (staggered)
        layer += 1
    tray = box_l("tray", 0.0, a, top - 0.06, Lu, a + L, top, "tray")
    lr = [0.0, a, Lu, a + L]
    rw = f.lrect_to_world(*lr)
    poly = [[rw[0], rw[1]], [rw[2], rw[1]], [rw[2], rw[3]], [rw[0], rw[3]]]
    e = W.element("low_overhead", "needs_crouch", {"clearance": float(clr), "length": float(L), "rack": True,
                                                   "layers": int(layer), "pipes": int(n_pipes), "top": top},
                  room.id, poly, note="pipe bridge across the room (rack, B-low)")
    W.tag(e, [p for p in elem if p is not None])
    del tray
    plan["element"] = e.id
    plan["layers"] = int(layer)
    return e


def approach_rects(lay, plan):
    """World rects of the two approaches (1.0 m of flat floor before and behind the strip or bridge, wall to wall)
    and of the back part (from the far edge to the back wall)."""
    f = frame_for(lay.rooms[plan["room"]])
    a, b, ap = plan["near_m"], plan["far_m"], plan["approach_m"]
    return {"front": f.lrect_to_world(0.0, a - ap, f.Lu, a), "behind": f.lrect_to_world(0.0, b, f.Lu, b + ap),
            "element": f.lrect_to_world(0.0, a, f.Lu, b), "back_part": f.lrect_to_world(0.0, b, f.Lu, f.Lv),
            "front_part": f.lrect_to_world(0.0, 0.0, f.Lu, a)}


RACK_READ_CLEAR_M = 0.30   # since 2026-11-16: a rack's margin regions (both postures) and its reading pose
                           # lie at least this far beyond the bridge's far edge, so that the whole body is behind it
                           # (the body is 0.55 m long: half of it is 0.275 m). An earlier version clipped at the far edge itself;
                           # a check found crouching reading poses with the body centre 0.05 m past it


def rack_reading_rect(zones, clear=RACK_READ_CLEAR_M):
    """The part of a rack room's back part where B-low's reading place may lie: the back part (zones: a variant's
    `zones` as stored in ladder.v2_obstacles[] or as approach_rects gives them, world rects) with its side on the
    bridge's far edge moved `clear` metres away from the bridge. Works on stored site.json zones alone."""
    bp = [float(v) for v in zones["back_part"]]
    el = [float(v) for v in zones["element"]]
    # the side of the back part that the element band borders: its far edge (the back part runs from it to the back
    # wall, at least the 1.0 m approach away)
    cand = [(abs(el[3] - bp[1]), 1, +1), (abs(el[1] - bp[3]), 3, -1), (abs(el[2] - bp[0]), 0, +1),
            (abs(el[0] - bp[2]), 2, -1)]
    d, i, s = min(cand)
    if d > 1e-3:
        raise ValueError("the back part %s does not border the element band %s" % (bp, el))
    bp[i] += s * float(clear)
    return bp


def spot_in_back(W, plan, sp, margin=0.05):
    """Is mount spot `sp` (a wall stretch of the variant's room) behind the band (the back part, beyond the far
    approach)? Returns the part [s0, s1] of the spot that is, or None."""
    if sp.get("kind") in ("rule", "tank") or sp["room"] != plan["room"]:
        return None
    v_min = plan["far_m"] + plan["approach_m"] + margin
    a, b = sp["s"]
    if sp["wall"] == "back":
        return [a, b]
    if sp["wall"] == "front":
        return None
    lo = max(a, v_min + 0.3)
    return [lo, b] if b - lo >= 0.0 else None
