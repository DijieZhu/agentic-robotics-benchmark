"""The ten-slot ladder (layout: ladder_v1).

Every site carries exactly ten items, each in one slot:
  A          an instrument in the airlock-side stretch of the south leg, read from inside the base station's link
  B-step     at its drawn place, behind its room's high-step door sills
  B-kerb     at its drawn place inside the tank-room bund
  B-low      at its drawn place behind a needs_crouch element (the pipe chase's valve station)
  C-sealed   at its drawn place, but the drawn door is sealed; the as-built way is an undrawn opening (from the chase's
             station or from a neighbouring room / corridor)
  C-moved    on another wall of its drawn room; a decoy of the same type hangs at the drawn place
  C-relocated in a neighbouring room or on the corridor wall outside its room; the drawn place is an empty bracket
  C-annex    in the undrawn annex; the round sheet says the location is not recorded
  M          the survey mark, drawn and correct, in a zone drawn as controlled
  X          the unlisted source in a zone drawn as supervised
The slot rules are proved on acceptance regions (accept.py, proofs.py); a sub-seed that breaks any rule is redrawn
and the failed rule is recorded (site.json ladder.attempt_log).
"""

from __future__ import annotations

import math
import os
import shutil
import time

import mujoco
import numpy as np

from . import radiation as RAD
from . import sheet as SH
from . import stations as ST
from . import variants as VR
from .accept import (Acceptance, RayCaster, evgeom_from_feature, feature_of, generous_region,  # noqa: F401
                     heights, margin_region, nominal_head)
from .config import acceptance_path, evidence_label, file_label
from .discrepancy import _wall_added
from .furnish import hinge_inner_door_away
from .generator import GuaranteeError, eye_cameras, furnish_all, grid_markers
from .ip_models import IPWorks
from .ips import LAMP_COLORS, Tagger
from .layout import _check_openings, add_door, build_layout, wall_between
from .link import coverage, opaque_raster
from .mjcf import build_scene, write_xml
from .structure import build_structure, rebuild_leaves
from .trav import MODES, Grid, TravMap
from .util import Rng, choice, dump_json, poly_rect, rect_center, uniform, weighted_key
from .world import World

SLOTS = ("A", "B-step", "B-kerb", "B-low", "C-sealed", "C-moved", "C-relocated", "C-annex")
BASE_TYPE = {"gauge_63": "gauge", "gauge_100": "gauge", "gauge_160": "gauge", "level": "level",
             "valve_lever": "valve_lever", "valve_handwheel": "valve_handwheel", "lamps": "lamps",
             "thermal": "thermal"}
TYPE_TEXT = {"gauge_63": "dial gauge, 63 mm case", "gauge_100": "dial gauge, 100 mm case",
             "gauge_160": "dial gauge, 160 mm case", "level": "tubular level gauge, 0.40 m window",
             "valve_lever": "quarter-turn valve (lever)", "valve_handwheel": "rising-stem gate valve",
             "lamps": "indicator lamp panel", "thermal": "temperature point (hot component)"}
# slot -> (mount zones, modes the positive proof may use)
SLOT_ZONES = {"A": ("a_stretch",), "B-step": ("room",), "B-kerb": ("bund",), "B-low": ("chase_station",),
              "C-sealed": ("room",), "C-moved": ("room",), "C-relocated": ("room", "corridor"), "C-annex": ("room",)}
SLOT_MODES = {"A": ("walk",), "B-step": ("walk", "high_step"), "B-kerb": ("walk", "high_step"), "B-low": MODES,
              "C-sealed": MODES, "C-moved": MODES, "C-relocated": MODES, "C-annex": MODES}
MIN_SHIFT_M = 1.5   # C-moved / C-relocated: the drawn place is at least this far from the true place (leak row: 1.0)
MARK_CLEAR_M = 0.35  # the survey mark's floor: at least this clearance from any obstacle (walking footprint map)
SERVICES = [("cw", "cooling water"), ("dw", "demineralised water"), ("rw", "radwaste liquor"), ("ia", "instrument air"),
            ("fw", "fire water"), ("st", "steam"), ("dr", "drain")]


class LadderReject(GuaranteeError):
    """A sub-seed that breaks a ladder rule; .rule names it (recorded per attempt)."""

    def __init__(self, rule, detail=""):
        super().__init__("%s: %s" % (rule, detail))
        self.rule = rule
        self.detail = detail


# ============================================================================================ type assignment
def infeasible(cfg, assign):
    """The (slot, type) pairs of an assignment that are ruled out. Version 1: ladder.infeasible_pairs. Version 2
    (2026-11-16): the measured exclusion list (site_gen.variants: (slot, the slot's variant, the sheet's
    place, type), config/exclusion_v2.json) replaces version 1's single (B-low, level) rule; the repair pass of
    variants.assignment_v2 leaves no listed pair in a candidate's assignment, so this finds none for a table
    candidate."""
    V2 = (cfg.get("ladder") or {}).get("v2") or {}
    if V2:
        row = {"bund": V2.get("bund"), "b_step": V2.get("b_step"), "b_low": V2.get("b_low"),
               "sheet_place": V2.get("sheet_place")}
        if cfg["ladder"].get("infeasible_pairs") == []:
            return []                   # the measurement of rates: every pair is tried
        return [(s, t) for s, t in assign.items() if VR.excluded(s, t, row)]
    return [(s, t) for s, t in (cfg["ladder"].get("infeasible_pairs") or []) if assign.get(s) == t]


def assign_types(cfg, g):
    L = cfg["ladder"]
    a = L.get("assignment", "random")
    types = list(L["types"])
    if isinstance(a, dict):
        if sorted(a.keys()) != sorted(SLOTS) or sorted(a.values()) != sorted(types):
            raise ValueError("ladder.assignment must map the eight slots to the eight types")
        if infeasible(cfg, a):
            raise ValueError("ladder.assignment uses an infeasible slot/type pair %s (ladder.infeasible_pairs)" %
                             infeasible(cfg, a))
        return {k: a[k] for k in SLOTS}
    allowed = L.get("slot_types") or {}
    for _ in range(500):
        perm = [types[i] for i in g.permutation(len(types))]
        out = dict(zip(SLOTS, perm))
        if all(out[s] in allowed.get(s, types) for s in SLOTS):
            return out
    raise ValueError("no assignment satisfies ladder.slot_types")


def cross_design(seed=0, n_sites=6, types=None, infeasible_pairs=(("B-low", "level"),)):
    """A fixed type x slot design for a set of six sites: every type in slot A or a B slot on at least
    three sites, no type twice in the same slot, no pair of `infeasible_pairs` (default as ladder.infeasible_pairs).
    Returns a list of {slot: type} for ladder.assignment."""
    types = list(types or ["gauge_63", "gauge_100", "gauge_160", "level", "valve_lever", "lamps", "thermal",
                           "valve_handwheel"])
    bad = {(str(s), str(t)) for s, t in (infeasible_pairs or [])}
    g = np.random.default_rng(seed)
    AB = ["A", "B-step", "B-kerb", "B-low"]
    C = ["C-sealed", "C-moved", "C-relocated", "C-annex"]
    for _ in range(20000):
        rows = []
        col = {s: set() for s in SLOTS}
        cnt = {t: 0 for t in types}
        ok = True
        for i in range(n_sites):
            for _try in range(200):
                perm = [types[j] for j in g.permutation(len(types))]
                row = dict(zip(AB + C, perm))
                if any(row[s] in col[s] or (s, row[s]) in bad for s in SLOTS):
                    continue
                rows.append(row)
                break
            else:
                ok = False
                break
            for s in SLOTS:
                col[s].add(rows[-1][s])
            for s in AB:
                cnt[rows[-1][s]] += 1
        if ok and all(v >= 3 for v in cnt.values()):
            return rows
    raise RuntimeError("no cross design found")


# ============================================================================================ roles (before structure)
def _drawn_doors(lay, rid):
    return [lay.door(d) for d in lay.rooms[rid].doors if lay.door(d).on_drawing and lay.door(d).kind in
            ("door", "double")]


def _roles(lay, cfg, g):
    rooms = lay.rooms
    chase = rooms[lay.params["chase"]]
    annex_host = lay.params["annex_host"]
    tank = [r for r in rooms.values() if r.type == "tank_room"]
    if not tank:
        raise LadderReject("layout", "no tank room")
    excl = {"corridor", "airlock", "annex", "pipe_chase", "tank_room", "waste_store"}
    # C-sealed: a room with exactly one drawn door; its as-built way is an undrawn opening from the chase station
    # (it must be beside the chase) or from a neighbouring room / the corridor
    cands = []
    for r in sorted(rooms.values(), key=lambda r: r.id):
        if r.type in excl or r.id == annex_host:
            continue
        dd = _drawn_doors(lay, r.id)
        if len(dd) != 1 or len(rooms[r.id].doors) != 1:
            continue
        ways = []
        for w in wall_between(lay, r.id, chase.id):
            if w.hi - w.lo > 3.0:
                ways.append(("chase", w))
        for w in lay.walls:
            if r.id not in w.rooms or w.kind not in ("ordinary",) or w.openings or w.hi - w.lo < 2.8:
                continue
            o = w.rooms[0] if w.rooms[1] == r.id else w.rooms[1]
            ro = rooms.get(o)
            if ro is None or ro.type in ("airlock", "annex", "pipe_chase", "waste_store", "tank_room"):
                continue
            if ro.type == "corridor" and o == "C1" and min(w.lo, w.hi) < lay.params["corridor_sill_x"] + 2.0:
                continue
            ways.append(("room", w))
        if ways:
            cands.append((r, ways))
    if not cands:
        raise LadderReject("roles", "no room for C-sealed")
    # prefer the chase variant half of the time when it exists
    chase_c = [(r, [w for w in ways if w[0] == "chase"]) for r, ways in cands if any(w[0] == "chase" for w in ways)]
    if chase_c and g.random() < 0.5:
        r_s, ways = chase_c[int(g.integers(len(chase_c)))]
    else:
        r_s, ways = cands[int(g.integers(len(cands)))]
    way = ways[int(g.integers(len(ways)))]
    # B-step: the gated room (all its doors get needs_high_step sills)
    bs = [r for r in sorted(rooms.values(), key=lambda r: r.id) if r.type not in excl and r.id != r_s.id and
          r.id != annex_host and len(_drawn_doors(lay, r.id)) >= 1 and
          all(lay.door(d).kind in ("door", "double") for d in r.doors)]
    if not bs:
        raise LadderReject("roles", "no room for B-step")
    wts = [{"pump_room": 2.0, "electrical_room": 2.0, "valve_gallery": 0.8}.get(r.type, 1.0) for r in bs]
    r_b = choice(g, bs, wts)
    return {"C-sealed": r_s.id, "C-sealed_way": way, "B-step": r_b.id, "B-kerb": tank[0].id, "B-low": chase.id,
            "annex": lay.params["annex"], "annex_host": annex_host}


# ============================================================================================ version 2: steering
STEERABLE = ("B-step", "B-low", "C-moved", "C-relocated")
_EXCL_V2 = {"corridor", "airlock", "annex", "pipe_chase", "tank_room", "waste_store"}


def _bstep_sill_rooms(lay, excl_ids=()):
    """Version 1's B-step rooms (every door gets a high-step sill)."""
    return [r for r in sorted(lay.rooms.values(), key=lambda r: r.id) if r.type not in _EXCL_V2 and
            r.id not in excl_ids and len(_drawn_doors(lay, r.id)) >= 1 and
            all(lay.door(d).kind in ("door", "double") for d in r.doors)]


def _leg_cut_sides(lay, place):
    """The corridor legs the missing sheet cuts and the side of each that goes (known from the sheet's geometry
    before the cut's exact place is fixed): {leg: (axis, mid-line, 'lo'|'hi')}."""
    br = lay.building_rect
    i, j = SH.place_ij(br, place)
    xm, ym = SH.mid_lines(br)
    out = {}
    for leg in SH.CORRIDOR_LEGS:
        R = lay.rooms[leg].rect
        if (R[2] - R[0]) > (R[3] - R[1]):
            if SH.sheet_ij(br, xm, 0.5 * (R[1] + R[3]))[1] == j and R[0] < xm < R[2]:
                out[leg] = ("x", xm, "hi" if i == 1 else "lo")
        elif SH.sheet_ij(br, 0.5 * (R[0] + R[2]), ym)[0] == i and R[1] < ym < R[3]:
            out[leg] = ("y", ym, "hi" if j == 1 else "lo")
    return out


def _v2_plan(lay, cfg, rng, V2):
    """The missing sheet for this candidate and the rooms it takes, the items its layout fixes there (the tank
    room: B-kerb; the annex host: C-annex; in the chase variant the chase: B-low), the target count of the other
    slots drawn from the seed among the values the layout allows, and for each steerable slot whether it goes on or
    off the sheet.

    Since 2026-11-16 C-relocated is not steered. Its recorded room and its true place are drawn as in version 1
    (uniformly over the neighbour places, drawn or missing) and the count row with a redraw decides. So that its own
    draw stays free (no correlation between "the recorded room borders the missing part" and "the item is there"),
    the other slots are planned to 2 or 3 (counts minus the largest value): C-relocated on or off the sheet then
    keeps the count inside 2-4 either way. This count stands as built: the other slots planned to
    two or three, C-relocated adding none or one (so a count of four needs C-relocated on the sheet)."""
    g = rng("v2_plan")
    P = cfg["ladder"]["v2_params"]
    n, rect, mrooms, mrects = SH.missing_rooms(lay, V2["sheet_place"])
    M = set(mrooms)
    host = lay.params["annex_host"]
    tank = [r.id for r in lay.rooms.values() if r.type == "tank_room"]
    fixed = []
    if tank and tank[0] in M:
        fixed.append("B-kerb")
    if host in M:
        fixed.append("C-annex")
    if V2["b_low"] == "chase" and lay.params["chase"] in M:
        fixed.append("B-low")
    cuts = _leg_cut_sides(lay, V2["sheet_place"])
    # where each steerable slot could go (rooms), on and off the sheet
    S = P["strip"]
    Rk = P["rack"]
    if V2["b_step"] == "door_sills":
        bstep = [r.id for r in _bstep_sill_rooms(lay, (host,))]
    else:
        bstep = [rid for rid, _ in VR.inroom_candidates(lay, S["min_depth_m"], (host,))]
    rack = [rid for rid, _ in VR.inroom_candidates(lay, Rk["min_depth_m"], (host,))] if V2["b_low"] == "rack" else []
    moved = [r.id for r in sorted(lay.rooms.values(), key=lambda r: r.id) if r.on_drawing and
             r.type not in ("corridor", "airlock", "annex", "pipe_chase")]
    reloc_on, reloc_off = [], []
    for r in sorted(lay.rooms.values(), key=lambda r: r.id):
        # the recorded room R is drawn; the true place is a neighbouring room of R or the corridor wall outside
        # R: on the sheet when that neighbour is a missing room or the corridor part outside R is cut off
        if not r.on_drawing or r.type in ("corridor", "airlock", "annex", "pipe_chase", "tank_room") or r.id in M:
            continue
        nb = [o for o in _neighbours(lay, r.id) if lay.rooms[o].on_drawing and
              lay.rooms[o].type not in ("airlock", "pipe_chase", "annex")]
        if any(o in M or o in cuts for o in nb):
            reloc_on.append(r.id)
        if any(o not in M for o in nb):
            reloc_off.append(r.id)
    cand = {"B-step": (bstep, M), "C-moved": (moved, M)}
    # C-relocated is not among the steered slots (its place is drawn as in version 1; the count row decides)
    steer = ["B-step", "C-moved"] + (["B-low"] if V2["b_low"] == "rack" else [])
    avail = {}
    for s in steer:
        if s == "B-low":
            avail[s] = (any(x in M for x in rack), any(x not in M for x in rack))
        else:
            lst, _ = cand[s]
            avail[s] = (any(x in M for x in lst), any(x not in M for x in lst))
    forced_on = [s for s in steer if avail[s][0] and not avail[s][1]]
    free = [s for s in steer if avail[s][0] and avail[s][1]]
    f = len(fixed) + len(forced_on)
    counts = [int(c) for c in P["sheet"]["counts"]]
    # the other slots' target leaves room for C-relocated's own draw (0 or 1 more): 2 or 3 of counts 2-4
    counts = [c for c in counts if c < max(counts)]
    allowed = [c for c in counts if f <= c <= f + len(free)]
    if not allowed:
        raise LadderReject("sheet_items", "the layout fixes %d and allows %d more on the %s sheet" % (
            f, len(free), V2["sheet_place"]))
    target = allowed[int(g.integers(len(allowed)))]
    order = [free[int(i)] for i in g.permutation(len(free))]
    on = set(order[:target - f]) | set(forced_on)
    want = {s: (s in on) for s in steer}
    return {"sheet": n, "rect": rect, "rooms": mrooms, "rects": mrects, "fixed": fixed, "forced_on": forced_on,
            "target": target, "allowed": allowed, "want": want, "planned": target, "cuts_by_side": cuts,
            "relocated": {"recorded_rooms_bordering": len(reloc_on), "recorded_rooms_off": len(reloc_off),
                          "steered": False}}


def _flip_ok(plan, slot):
    """May steerable `slot` go to the other side of the sheet than planned (its placement found nothing there)
    while the count can still land inside 2-4? Updates the plan when it may."""
    if slot not in plan["want"]:
        return False
    nxt = plan["planned"] + (-1 if plan["want"][slot] else 1)
    lo, hi = min(plan["allowed"]), max(plan["allowed"])
    if not (lo <= nxt <= hi):
        return False
    plan["planned"] = nxt
    plan["want"][slot] = not plan["want"][slot]
    plan.setdefault("flipped", []).append(slot)
    return True


def _pick_side(plan, slot, cands, key, g, weights=None):
    """Choose from `cands` (room ids or objects with key()) on the planned side of the sheet; fall back to the other
    side when the count allows (_flip_ok). Returns the choice or None."""
    M = set(plan["rooms"])
    for _ in range(2):
        want = plan["want"].get(slot)
        pool = [c for c in cands if want is None or ((key(c) in M) == want)]
        if pool:
            if weights is not None:
                w = [weights(c) for c in pool]
                return choice(g, pool, w)
            return pool[int(g.integers(len(pool)))]
        if not _flip_ok(plan, slot):
            return None
    return None


CHASE_WAY_TYPES = ("valve_lever", "level")     # B-low types the chase holds (almost) only with C-sealed's way


def _roles_v2(lay, cfg, g, rng, V2, plan):
    """Version 2's roles: the B-step room (door sills: every door high-step; in the room: a deep single-door decor
    room, deeper rooms first), the B-low room (the chase, or a single-door decor room for the rack), each on the
    planned side of the missing sheet; then the C-sealed room among the drawn rooms, never one of the B rooms and
    never with its as-built way through a B room's wall (an opening there would bypass the B obstacle)."""
    gR = rng("v2_roles")
    rooms = lay.rooms
    chase = rooms[lay.params["chase"]]
    annex_host = lay.params["annex_host"]
    tank = [r for r in rooms.values() if r.type == "tank_room"]
    if not tank:
        raise LadderReject("layout", "no tank room")
    M = set(plan["rooms"])
    P = cfg["ladder"]["v2_params"]
    out = {"B-kerb": tank[0].id, "annex": lay.params["annex"], "annex_host": annex_host, "v2_plan": plan}
    # ---- B-step
    if V2["b_step"] == "door_sills":
        bs = _bstep_sill_rooms(lay, (annex_host,))
        wts = lambda r: {"pump_room": 2.0, "electrical_room": 2.0, "valve_gallery": 0.8}.get(r.type, 1.0)
        r_b = _pick_side(plan, "B-step", bs, lambda r: r.id, gR, weights=wts)
        if r_b is None:
            raise LadderReject("roles", "no room for B-step (door sills) on the planned side")
        out["B-step"] = r_b.id
    else:
        S = P["strip"]
        cands = VR.inroom_candidates(lay, S["min_depth_m"], (annex_host,))
        asg = cfg["ladder"].get("assignment")
        typ = asg.get("B-step") if isinstance(asg, dict) else None
        from .furnish import frame_for
        # deeper rooms first: first the rooms where the slot's type has a hidden place behind the strip
        # (variants.strip_slack >= 0) in the depth classes of strip.depth_first, deepest class first; then any room
        feas = {rid for rid, dep in cands if typ is None or
                VR.strip_slack(frame_for(lay.rooms[rid]).Lu, dep, typ, approach=float(S["approach_m"])) >= 0.0}
        tiers = [[rid for rid, dep in cands if rid in feas and dep + 1e-6 >= float(dmin)]
                 for dmin in [float(x) for x in (S.get("depth_first") or [])] + [float(S["min_depth_m"])]]
        tiers.append([rid for rid, _ in cands])
        chosen = None
        want = plan["want"]["B-step"]
        for pool in tiers:
            if any((rid in M) == want for rid in pool):
                chosen = _pick_side(plan, "B-step", pool, lambda x: x, gR)
                break
        if chosen is None:
            chosen = _pick_side(plan, "B-step", tiers[-1], lambda x: x, gR)
        out["B-step_feasible"] = chosen in feas if chosen else False
        if chosen is None:
            raise LadderReject("roles", "no room for B-step in the room (one door, >= %.1f m deep)" %
                               float(S["min_depth_m"]))
        out["B-step"] = chosen
    # ---- B-low
    if V2["b_low"] == "chase":
        out["B-low"] = chase.id
    else:
        R = P["rack"]
        cands = [rid for rid, _ in VR.inroom_candidates(lay, R["min_depth_m"], (annex_host, out["B-step"]))]
        if V2["b_step"] == "door_sills":
            # the rack room's door sill stays free: never a room whose door opens into the B-step room, every
            # door of which has a high-step sill (doctest k 85: the rack room entered over B-step's sill)
            cands = [rid for rid in cands if not any(out["B-step"] in lay.door(d).rooms for d in lay.rooms[rid].doors)]
        chosen = _pick_side(plan, "B-low", cands, lambda x: x, gR)
        if chosen is None:
            raise LadderReject("roles", "no room for the B-low rack (one door, >= %.1f m deep)" %
                               float(R["min_depth_m"]))
        out["B-low"] = chosen
    # the rooms whose obstacle an opening would bypass: the B-step room (its sills or its strip) and the rack room
    # (the chase stays a way for C-sealed, as in version 1: the opening leads from the station into the sealed room)
    b_rooms = {out["B-step"]} | ({out["B-low"]} if V2["b_low"] == "rack" else set())
    # ---- C-sealed: as version 1, among the drawn rooms, off the B rooms and their walls
    cands = []
    for r in sorted(rooms.values(), key=lambda r: r.id):
        if r.type in _EXCL_V2 or r.id == annex_host or r.id in b_rooms or r.id in M:
            continue
        dd = _drawn_doors(lay, r.id)
        if len(dd) != 1 or len(rooms[r.id].doors) != 1:
            continue
        ways = []
        for w in wall_between(lay, r.id, chase.id):
            if w.hi - w.lo > 3.0:
                ways.append(("chase", w))
        for w in lay.walls:
            if r.id not in w.rooms or w.kind not in ("ordinary",) or w.openings or w.hi - w.lo < 2.8:
                continue
            o = w.rooms[0] if w.rooms[1] == r.id else w.rooms[1]
            ro = rooms.get(o)
            if ro is None or ro.type in ("airlock", "annex", "pipe_chase", "waste_store", "tank_room") or o in b_rooms:
                continue
            if ro.type == "corridor" and o == "C1" and min(w.lo, w.hi) < lay.params["corridor_sill_x"] + 2.0:
                continue
            ways.append(("room", w))
        if ways:
            cands.append((r, ways))
    if not cands:
        raise LadderReject("roles", "no drawn room for C-sealed")
    chase_c = [(r, [w for w in ways if w[0] == "chase"]) for r, ways in cands if any(w[0] == "chase" for w in ways)]
    asg = cfg["ladder"].get("assignment")
    if V2["b_low"] == "chase" and isinstance(asg, dict) and asg.get("B-low") in CHASE_WAY_TYPES and chase_c:
        # a quarter-turn valve or a level gauge at the chase's station is read almost only through C-sealed's as-built
        # opening (measured 2026-11-16 on 36 sub-seeds: in a closed chase the level gauge passed on 0 of 20, the
        # valve on 1 of 20 (every other place seen from the hatches or without a margin region); with the opening all
        # eight types on 7 of 7). Version 1 drew the chase way half the time and redrew the sub-seed otherwise; here
        # it is taken whenever a drawn room beside the chase allows it (with the chase on the missing sheet, place
        # 'across', none does and the pre-checks decide as before)
        r_s, ways = chase_c[int(gR.integers(len(chase_c)))]
    elif chase_c and gR.random() < 0.5:
        r_s, ways = chase_c[int(gR.integers(len(chase_c)))]
    else:
        r_s, ways = cands[int(gR.integers(len(cands)))]
    out["C-sealed"] = r_s.id
    out["C-sealed_way"] = ways[int(gR.integers(len(ways)))]
    return out


def slot_on_sheet(miss, room_id, xy, annex_id, annex_host):
    """Is a listed instrument's TRUE place on the missing sheet? Its room is a missing room, it hangs on a cut-off
    corridor part, or it is in the annex of a host on the missing sheet (the annex is on no sheet; it goes with its
    host for this count)."""
    if room_id == annex_id:
        return miss.room_missing(annex_host)
    return miss.room_missing(room_id) or miss.in_cut(float(xy[0]), float(xy[1]))


def _v2_count(W, roles, placed):
    """Count the listed instruments whose true place is on the missing sheet (the required row
    sheet_items checks it again from the files): 2 to 4, else the sub-seed is redrawn."""
    miss = W.missing
    on = []
    for slot in SLOTS:
        p = placed[slot]
        if slot_on_sheet(miss, p["spot"]["room"], p["face"][:2], roles["annex"], roles["annex_host"]):
            on.append(slot)
    roles["v2_on_sheet"] = on
    lo, hi = [int(v) for v in (min(W.cfg["ladder"]["v2_params"]["sheet"]["counts"]),
                                max(W.cfg["ladder"]["v2_params"]["sheet"]["counts"]))]
    if not (lo <= len(on) <= hi):
        raise LadderReject("sheet_items", "%d listed instruments on the %s sheet (%s; planned %d)" % (
            len(on), miss.place, on, roles["v2_plan"]["planned"]))


def _v2_obstacles(W, cfg, rng, roles, V2):
    """Version 2: the in-room strip of B-step and the rack of B-low: planned in their
    rooms' frames, their bands kept out of the furnishing, built as tagged elements. Records go to roles['v2_obstacles']
    (site.json ladder.v2_obstacles: band, approaches, element)."""
    out = []
    W.v2_back_rooms = {}
    if V2["b_step"] == "in_room":
        asg = cfg["ladder"].get("assignment")
        p = VR.plan_strip(W.lay, roles["B-step"], cfg, rng("v2_strip"),
                          typ=asg.get("B-step") if isinstance(asg, dict) else None)
        VR.keep_out(W, p)
        VR.build_strip(W, p, rng("v2_strip_build"))
        out.append(p)
    if V2["b_low"] == "rack":
        p = VR.plan_rack(W.lay, roles["B-low"], cfg, rng("v2_rack"))
        if p is None:
            raise LadderReject("roles", "the rack room %s is too shallow for its bridge" % roles["B-low"])
        VR.keep_out(W, p)
        VR.build_rack(W, p, rng("v2_rack_build"))
        out.append(p)
    for p in out:
        # the room's reserved instrument walls go behind the band (furnish_all -> _reserve_instrument_walls)
        W.v2_back_rooms[p["room"]] = float(p["far_m"] + p["approach_m"])
    roles["v2_obstacles"] = out


def _station_span_world(lay, cfg):
    """World extent (along the chase) of the chase's valve station, by the chase rule's own formula."""
    ch = lay.rooms[lay.params["chase"]]
    x0, y0, x1, y1 = ch.inner
    Lv = (y1 - y0)
    st_len = round(min(3.4, max(2.4, Lv - 2.5)) / 0.05) * 0.05
    vs0 = round(0.5 * (Lv - st_len) / 0.05) * 0.05
    return y0 + vs0, y0 + vs0 + st_len


def _add_openings(lay, cfg, g, roles):
    """Layout-time deviations: the C-sealed room's undrawn as-built opening; 0-2 optional deviations (an added
    opening is cut here, an added partition after furnishing)."""
    kind, w = roles["C-sealed_way"]
    rid = roles["C-sealed"]
    if kind == "chase":
        a, b = _station_span_world(lay, cfg)
        width = 1.0
        c0 = round(0.5 * (a + b) / 0.05) * 0.05
    else:
        width = round(uniform(g, 1.0, 1.2) / 0.05) * 0.05
        lo, hi = w.lo + 0.8 + width / 2, w.hi - 0.8 - width / 2
        c0 = round(uniform(g, lo, max(lo, hi)) / 0.05) * 0.05
    d = add_door(lay, w, c0, width, kind="opening", into=rid, g=g)
    d.on_drawing = False
    d.label = "AS-BUILT OPENING"
    roles["as_built_opening"] = d.id
    # optional deviations
    O = cfg["ladder"]["optional_deviations"]
    n = int(g.integers(int(O["count"][0]), int(O["count"][1]) + 1))
    kinds = []
    wts = dict(O["weights"])
    for _ in range(n):
        if not wts:
            break
        k = weighted_key(g, wts)
        kinds.append(k)
        if k == "opening_added":
            wts.pop(k)
    roles["optional"] = kinds
    roles["added_openings"] = []
    if "opening_added" in kinds:
        avoid = {rid, roles["B-step"], roles["B-kerb"], roles["B-low"], roles["annex"], roles["annex_host"], "A1"}
        cands = []
        for w2 in lay.walls:
            ra, rb = lay.rooms.get(w2.rooms[0]), lay.rooms.get(w2.rooms[1])
            if ra is None or rb is None or set(w2.rooms) & avoid or w2.kind != "ordinary" or w2.openings:
                continue
            if {ra.type, rb.type} & {"airlock", "annex", "waste_store", "pipe_chase", "tank_room"}:
                continue
            if w2.hi - w2.lo < 2.8:
                continue
            if "C1" in w2.rooms and min(w2.lo, w2.hi) < lay.params["corridor_sill_x"] + 2.0:
                continue
            cands.append(w2)
        if cands:
            w2 = cands[int(g.integers(len(cands)))]
            width = round(uniform(g, 1.0, 1.3) / 0.05) * 0.05
            lo, hi = w2.lo + 0.9 + width / 2, w2.hi - 0.9 - width / 2
            c2 = round(uniform(g, lo, max(lo, hi)) / 0.05) * 0.05
            d2 = add_door(lay, w2, c2, width, kind="opening", g=g)
            d2.on_drawing = False
            roles["added_openings"].append(d2.id)


def _seal_door(W, roles):
    """The C-sealed room's drawn door: closed for good (leaf shut, steel straps across, a sealed notice)."""
    lay = W.lay
    d = _drawn_doors(lay, roles["C-sealed"])[0]
    d.state = "closed"
    d.blocked_by = "sealed"
    rebuild_leaves(W, d)
    w = lay.wall(d.wall)
    out_room = d.rooms[0] if d.rooms[1] == roles["C-sealed"] else d.rooms[1]
    s = 1 if out_room == w.rooms[1] else -1                    # the straps go on the face toward out_room
    face = (w.t_pos if s > 0 else w.t_neg) + 0.012
    body = "door/%s/seal" % d.id
    for k, z in enumerate((0.45, 1.0, 1.55)):
        if w.axis == "y":
            W.P.box("%s/strap%d" % (body, k), (d.center, w.coord + s * face, z), (d.width / 2 + 0.08, 0.008, 0.035),
                    mat="steel_dark", body=body, owner=d.id, group=1)
        else:
            W.P.box("%s/strap%d" % (body, k), (w.coord + s * face, d.center, z), (0.008, d.width / 2 + 0.08, 0.035),
                    mat="steel_dark", body=body, owner=d.id, group=1)
    mat = W.sign("sealed_%s" % d.id, "DOOR SEALED - NO ACCESS", "room")
    if w.axis == "y":
        W.P.box("%s/sign" % body, (d.center, w.coord + s * (face + 0.01), 1.28), (0.25, 0.004, 0.06),
                yaw=math.pi / 2 if s > 0 else -math.pi / 2, mat=mat, body=body, owner=d.id, group=1)
    else:
        W.P.box("%s/sign" % body, (w.coord + s * (face + 0.01), d.center, 1.28), (0.004, 0.25, 0.06),
                yaw=0.0 if s > 0 else math.pi, mat=mat, body=body, owner=d.id, group=1)
    roles["sealed_door"] = d.id


# ============================================================================================ reach
def _elem_mask(grid, elements, ids):
    m = np.zeros(grid.shape, np.uint8)
    for i in ids:
        e = elements[i]
        grid.fill_poly(m, e.footprint if hasattr(e, "footprint") else e["footprint"], 1)
    return m.astype(bool)


def _door_mask(grid, lay, did, depth_extra=0.15):
    d = lay.door(did)
    w = lay.wall(d.wall)
    a, b = d.center - d.width / 2 - 0.05, d.center + d.width / 2 + 0.05
    if w.axis == "x":
        poly = poly_rect([w.coord - w.t_neg - depth_extra, a, w.coord + w.t_pos + depth_extra, b])
    else:
        poly = poly_rect([a, w.coord - w.t_neg - depth_extra, b, w.coord + w.t_pos + depth_extra])
    m = np.zeros(grid.shape, np.uint8)
    grid.fill_poly(m, poly, 1)
    return m.astype(bool)


def _dilate(mask, r_m, res):
    from scipy import ndimage
    if r_m <= 0:
        return mask
    d = ndimage.distance_transform_edt(~mask)
    return d * res <= r_m + 1e-9


class Reach:
    """Reach sets of one traversability map (cells, any mode, or per mode)."""

    def __init__(self, trav, cfg):
        self.t = trav
        self.cfg = cfg
        self.pad = trav.node(0.0, 0.0, "walk")
        self._cache = {}

    def nodes(self, modes=MODES, blocked=None, key=None):
        k = (tuple(modes), key)
        if k in self._cache:
            return self._cache[k]
        enter = self.t.enter_blocked(blocked) if blocked is not None else None
        r = self.t.reach_from(self.pad, modes, enter)
        self._cache[k] = r
        return r

    def cells(self, modes=MODES, blocked=None, key=None):
        return self.t.cells(self.nodes(modes, blocked, key))

    def by_mode(self, modes=MODES, blocked=None, key=None):
        r = self.nodes(modes, blocked, key)
        return {m: self.t.cells_mode(r, m) for m in MODES}


# ============================================================================================ items
# A lever valve's red grip on its own red fire-water line is not readable from any pose (a readers' study, 2026-11-14:
# HV-019 of a test site, the grip touches the red line in every image): lever valves never sit on fire water.
NO_SERVICES = {"valve_lever": ("fw",)}


def _service(g, room, typ=None):
    opts = [s for s in SERVICES if s[0] not in NO_SERVICES.get(typ, ())]
    code, text = opts[int(g.integers(len(opts)))]
    return {"code": code, "text": text}


def truth_params(cfg, g, typ):
    """Truth and catalogue parameters of a ladder instrument of type `typ` (gauge_63, ..., thermal)."""
    I = cfg["ips"]
    L = cfg["ladder"]
    lo_f, hi_f = I["value_span"]
    if typ.startswith("gauge"):
        lo, hi, unit = choice(g, I["gauge_scales"])
        v = lo + uniform(g, lo_f, hi_f) * (hi - lo)
        stp = (hi - lo) / 200.0
        return {"lo": lo, "hi": hi, "unit": unit, "case_mm": int(typ.split("_")[1])}, \
            {"value": round(round(v / stp) * stp, 4), "unit": unit}
    if typ == "level":
        return {"window_m": float(L["level_window_m"])}, {"level_pct": float(round(uniform(g, 8, 92)))}
    if typ == "valve_lever":
        # each end at most about 20 %, throttled positions away from the ends, so that
        # neither a constant report nor the end positions expect more than the blind-report rule's 0.22 (site_gen.blind)
        T = L["truth"]
        pe = float(T["lever_end_p"])
        r = g.random()
        lo_m, hi_m = [float(v) for v in T["lever_mid_deg"]]
        ang = 0.0 if r < pe else (90.0 if r < 2 * pe else float(round(uniform(g, lo_m, hi_m))))
        return {}, {"angle_deg": ang, "state": "open" if ang == 0 else ("closed" if ang == 90 else "throttled")}
    if typ == "valve_handwheel":
        return {}, {"opening_pct": float(round(uniform(g, 5, 95)))}
    if typ == "lamps":
        n = int(g.integers(I["lamps_count"][0], I["lamps_count"][1] + 1))
        cols = [choice(g, LAMP_COLORS) for _ in range(n)]
        st = [bool(g.random() < 0.5) for _ in range(n)]
        if all(st):
            st[int(g.integers(n))] = False
        if not any(st):
            st[int(g.integers(n))] = True
        return {"colors": cols}, {"states": st}
    if typ == "thermal":
        T = L["thermal"]
        surf = sorted(T["surfaces"].keys())
        sfc = surf[int(g.integers(len(surf)))]
        # hot_C 40-110 deg C: a constant report expects about 0.15 (the blind-report rule: at most 0.22)
        return {"surface": sfc, "range": list(T["range"]), "unit": "°C"}, \
            {"temp_C": round(uniform(g, *T["hot_C"]), 1), "surface": sfc, "emissivity": float(T["surfaces"][sfc])}
    raise KeyError(typ)


def last_2009(cfg, g, typ, prm):
    """The last logged value of 2009, drawn independently of the true value."""
    if typ.startswith("gauge"):
        span = float(prm["hi"]) - float(prm["lo"])
        v = prm["lo"] + uniform(g, 0.1, 0.9) * span
        res = 10.0 ** math.floor(math.log10(span / 40.0))      # a logged reading: 0.1 bar, 1 deg C
        return {"value": round(round(v / res) * res, 4), "unit": prm["unit"]}
    if typ == "level":
        return {"value": float(round(uniform(g, 10, 90))), "unit": "%"}
    if typ == "valve_lever":
        return {"value": "open" if g.random() < 0.5 else "closed"}
    if typ == "valve_handwheel":
        return {"value": float(round(uniform(g, 0, 100) / 5) * 5), "unit": "% open"}
    if typ == "lamps":
        return {"value": ["on" if g.random() < 0.5 else "off" for _ in prm["colors"]]}
    if typ == "thermal":
        return {"value": round(uniform(g, 25, 70), 0), "unit": "°C"}
    return {}


class _Dry:
    """Build a station, read its slot pose, and roll the world back (a candidate is tried before it is built)."""

    def __init__(self, W):
        self.W = W

    def __enter__(self):
        self.np = len(self.W.P.prims)
        self.ns = len(self.W.slots)
        return self

    def __exit__(self, *a):
        for p in self.W.P.prims[self.np:]:
            self.W.P.names.discard(p.name)
        del self.W.P.prims[self.np:]
        del self.W.slots[self.ns:]
        return False


# ============================================================================================ main
# The dressed check inside the sub-seed loop (site_gen.sets --dressed-check sets it; None otherwise): a callable
# (site folder) -> {"required_ok", "failed", "diff_vs_stored", ...} run on the written files of a sub-seed that held
# every row; a sub-seed that the dressing refuses or whose rows do not hold dressed is recorded ("dressed") and the
# next one is drawn (2026-11-16: a development candidate was lost as a whole when the dressing refused a
# station bracket 0.2 mm proud of a skinned wall after generation). A sub-seed that passes gives the same files.
DRESSED_CHECK = None


def generate(seed, out_dir, cfg, maps=True, verbose=False):
    t0 = time.time()
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log = []
    for attempt in range(int(cfg["max_attempts"])):
        ta = time.time()
        try:
            site, prior, stats = _attempt(seed, attempt, out_dir, cfg, verbose)
        except LadderReject as e:
            log.append({"attempt": attempt, "rule": e.rule, "detail": str(e.detail)[:200], "s": round(time.time() - ta, 1)})
            if verbose:
                print("seed %d attempt %d rejected: %s %s" % (seed, attempt, e.rule, e.detail), flush=True)
            continue
        except (ValueError, IndexError, KeyError, RuntimeError, ZeroDivisionError) as e:
            log.append({"attempt": attempt, "rule": "grammar", "detail": "%s: %s" % (type(e).__name__, str(e)[:160]),
                        "s": round(time.time() - ta, 1)})
            if verbose:
                import traceback
                traceback.print_exc()
            continue
        stats["attempts"] = attempt + 1
        for f in os.listdir(out_dir):
            if f.startswith("_reject_"):
                os.remove(os.path.join(out_dir, f))
        timing = dict(stats.pop("timing_s"))
        timing["generate_s"] = time.time() - t0
        site["stats"] = stats
        site["ladder"]["attempt_log"] = [{k: v for k, v in r.items() if k != "s"} for r in log]
        dump_json(site, os.path.join(out_dir, "site.json"))
        dump_json(prior, os.path.join(out_dir, "prior.json"))
        dump_json(dict(timing, attempt_log=log), os.path.join(out_dir, "timing.json"))
        dressed = None
        if DRESSED_CHECK is not None:
            ta = time.time()
            try:
                dressed = DRESSED_CHECK(out_dir)
                bad = not (dressed.get("required_ok") and not dressed.get("diff_vs_stored"))
                why = "failed %s, differing %s" % (dressed.get("failed"), dressed.get("diff_vs_stored"))
            except Exception as e:  # noqa: BLE001  (the dressing refused the scene: DressError and the like)
                bad, why = True, "%s: %s" % (type(e).__name__, str(e)[:160])
                dressed = {"error": why}
            if bad:
                log.append({"attempt": attempt, "rule": "dressed", "detail": why[:200], "s": round(time.time() - ta, 1)})
                os.remove(os.path.join(out_dir, "site.json"))      # not a generated site (the next attempt writes it)
                if verbose:
                    print("seed %d attempt %d rejected dressed: %s" % (seed, attempt, why), flush=True)
                continue
        if maps:
            from .maps import make_maps
            tm = time.time()
            make_maps(out_dir)
            timing["maps_s"] = time.time() - tm
            dump_json(dict(timing, attempt_log=log), os.path.join(out_dir, "timing.json"))
        site["_timing"] = timing
        site["_attempt_log"] = log
        site["_dressed"] = dressed
        return site, prior
    dump_json({"seed": seed, "attempt_log": log}, os.path.join(out_dir, "failed.json"))
    err = GuaranteeError("seed %d: no attempt satisfied the ladder rules (%s)" % (
        seed, ", ".join(sorted({r["rule"] for r in log}))))
    err.attempt_log = log
    raise err


def _attempt(seed, attempt, out_dir, cfg, verbose):
    T = {}
    t = time.time()
    rng = Rng(seed, attempt)
    gL = rng("ladder")
    lay = build_layout(cfg, rng)
    V2 = cfg["ladder"].get("v2")
    if V2:
        # version 2: the missing sheet's rooms and the steering plan first, then the roles
        plan = _v2_plan(lay, cfg, rng, V2)
        roles = _roles_v2(lay, cfg, gL, rng, V2, plan)
        roles["v2"] = dict(V2)
        lay.params["ladder_gated"] = roles["B-step"] if V2["b_step"] == "door_sills" else None
        lay.params["free_sill_rooms"] = sorted(({roles["B-step"]} if V2["b_step"] == "in_room" else set()) |
                                               ({roles["B-low"]} if V2["b_low"] == "rack" else set()))
    else:
        roles = _roles(lay, cfg, gL)
        lay.params["ladder_gated"] = roles["B-step"]
    _add_openings(lay, cfg, gL, roles)
    _check_openings(lay)
    # the airlock's inner leaf opens away from the base station's aerial
    roles["inner_door_hinge"] = hinge_inner_door_away(lay)
    W = World(cfg, lay, rng)
    build_structure(W)
    _seal_door(W, roles)
    if V2:
        _v2_obstacles(W, cfg, rng, roles, V2)
    furnish_all(W)
    if str(cfg["ladder"].get("lighting")) == "bright":
        for Lg in W.lights:
            Lg["level"] = "bright"
    # optional partition across a corridor leg (not in the slot-A stretch)
    if "wall_added" in roles["optional"]:
        _wall_added(W, cfg, rng("partition"), [], [], lambda *a, **k: None, no_chase=True,
                    x_min=float(lay.params["corridor_sill_x"]) + 4.0)
    if V2:
        # the corridor cuts need the fire doors (0.3 m clear of every door opening, hatch and fire door)
        Sp = cfg["ladder"]["v2_params"]["sheet"]
        miss = SH.compute(lay, V2["sheet_place"], rng("v2_sheet"), W, shift_m=float(Sp["cut_shift_m"]),
                          clear_m=float(Sp["cut_clear_m"]))
        if miss is None:
            raise LadderReject("sheet_edges", "no corridor cut 0.3 m clear of the openings within 1.0 m")
        if miss.rooms != sorted(roles["v2_plan"]["rooms"]):
            raise RuntimeError("missing rooms changed after the layout")
        W.missing = miss
    # mount spots
    for rid, c in sorted(W.ctx.items()):
        r = lay.rooms[rid]
        if r.type in ("corridor", "airlock", "pipe_chase", "tank_room"):
            continue
        ST.scan_wall_spots(W, c, zone="room")
    ST.corridor_spots(W)
    T["world_s"] = time.time() - t
    # puddles off every required access: before the pre-model is built
    roles["pruned_puddles"] = _prune_puddles(W, cfg)
    # no tight passage (a leaf left ajar) on a required access: a door whose
    # narrow gap is the only way to some floor stands fully open instead
    roles["opened_ajar"] = _open_required_ajar(W, cfg)
    acc = Acceptance(cfg)
    # ------------------------------------------------------------------ pre-model (no instruments) and its map
    t = time.time()
    pre_dir = os.path.join(out_dir, "_pre")
    spec0, info0 = build_scene(W, None, pre_dir, [])
    m0 = spec0.compile()
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    bb = lay.bbox()
    grid = Grid(bb, float(cfg["grid_res"]))
    trav0 = TravMap(grid, cfg)
    trav0.rasterize(W.P.prims, W.elements, [r.rect for r in lay.rooms.values()])
    R0 = Reach(trav0, cfg)
    T["pre_s"] = time.time() - t
    cuts = _cuts(W, roles, grid)
    # link coverage (for slot A)
    lg = Grid(bb, 0.1)
    opaque = opaque_raster(W, lg)
    inside_l = np.zeros(lg.shape, np.uint8)
    for r in lay.rooms.values():
        lg.fill_poly(inside_l, poly_rect(r.rect), 1)
    ants = [(W.base_station["antenna"][0], W.base_station["antenna"][1], float(cfg["link"]["base_range"]))]
    ants += [(r["pos"][0], r["pos"][1], float(cfg["link"]["repeater_range"])) for r in W.repeaters if r["working"]]
    cov = coverage(lg, opaque, ants, inside_l.astype(bool))
    cov_base = coverage(lg, opaque, ants[:1], inside_l.astype(bool))
    # ------------------------------------------------------------------ the site-wide rule on the pre-map
    A_side = R0.cells(MODES, cuts["sill"], "sill")
    _check_site_rule(W, grid, A_side)
    # slot A is read inside the base station's link as the cell decides it (3-D rays against group-0 geometry, the
    # open door leaves included), walking posture
    from . import linkray as LR
    lk = LR.cell_link_config(cfg)
    link_cells = LR.link_mask(m0, d0, grid, A_side, LR.nodes_of(W.base_station, [], lk), float(lk["end_margin_m"]),
                              LR.lidar_heights()["walk"])
    # ------------------------------------------------------------------ types and slots
    t = time.time()
    assign = assign_types(cfg, gL)
    if infeasible(cfg, assign):
        # rejected at once instead of after a failed placement (the admitted sub-seeds do not change); version 2: the
        # (B-low, level) pair is infeasible in the chase only (in a rack room the proofs decide: infeasible())
        raise LadderReject("assign_infeasible", "slot/type pair %s cannot satisfy its rules" % infeasible(cfg, assign))
    placed = _place_slots(W, cfg, gL, acc, m0, d0, grid, trav0, R0, cuts, roles, assign, link_cells, A_side)
    T["slots_s"] = time.time() - t
    if MEASURE is not None:
        raise MeasureDone(T["slots_s"])
    if V2:
        _v2_count(W, roles, placed)
    # ------------------------------------------------------------------ sources: drums, the documented process
    # source (M's zone), the unlisted source X; the survey mark
    t = time.time()
    sources = _drums(W, cfg, rng)
    proc = _process_source(W, cfg, gL, roles)
    sources.append(proc)
    boxes = RAD.attenuators(W, cfg)
    bg = round(uniform(rng("radiation_bg"), *cfg["radiation"]["background"]), 3)
    dose = _calibrate(W, cfg, gL, trav0, sources, proc, boxes, bg, roles, g_target=rng("mark_target"),
                      keep_away=[tuple(placed[sl]["face"][:2]) for sl in ("C-moved", "C-relocated")
                                 if sl in placed and "face" in placed[sl]])
    zoning0 = _zoning_accessible(W, cfg, trav0, sources, boxes, bg)
    xsrc = _x_source(W, cfg, gL, roles, placed, trav0, R0, grid, proc, sources, boxes, bg, zoning0,
                     s_gyh=uniform(rng("x_strength"), *cfg["ladder"]["dose"]["source_gyh_at_1m"]))
    sources.append(xsrc)
    dose["x_source"] = xsrc["id"]
    T["sources_s"] = time.time() - t
    # ------------------------------------------------------------------ items, decoys, tags
    items, decoys, plain = _items(W, cfg, rng, placed, assign, roles, xsrc, dose,
                                  *((m0, d0) if V2 else (None, None)))
    if V2:
        W.missing.items = sorted(i["tag"] for i in items if i.get("ladder_slot") in roles["v2_on_sheet"])
    # the stations can narrow a passage: once more, no floor only behind a leaf left ajar
    roles["opened_ajar"] = roles.get("opened_ajar", []) + _open_required_ajar(W, cfg)
    markers = grid_markers(W, cfg, rng)
    gm = rng("markers_present")
    n_m = len(markers)
    k_present = int(round(float(cfg["ladder"]["grid_markers_present"]) * n_m))
    present = set(gm.choice(n_m, size=k_present, replace=False).tolist()) if n_m else set()
    for k, m in enumerate(markers):
        m["present"] = k in present
    ipw = IPWorks(cfg)
    for it in items + decoys + plain:
        if it["type"] == "hotspot":
            continue
        ipw.add(it["slot"], it["role"], it["tag"], it["truth"], it["params"])
    for m in markers:
        if m["present"]:
            ipw.add_marker(m)
    cams = eye_cameras(W, items, decoys)
    t = time.time()
    if os.path.isdir(os.path.join(out_dir, "assets")):
        shutil.rmtree(os.path.join(out_dir, "assets"))
    spec, info = build_scene(W, ipw, out_dir, cams)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    T["compile_s"] = time.time() - t
    for it in items + decoys + plain:
        mi = info["ip_models"].get(it["body"])
        if mi:
            it["backend"] = mi["backend"]
            it["model_names"] = {k: v for k, v in mi.items() if k in ("body", "geoms", "face_geoms", "joints", "backend",
                                                                       "bricks", "marker_id")}
    for m in markers:
        mi = info["ip_models"].get("marker/%s" % m["id"])
        if mi:
            m["model_names"] = {k: v for k, v in mi.items() if k in ("body", "geoms", "face_geoms", "backend",
                                                                       "dictionary")}
    from .ip_models import apply_truth_states
    apply_truth_states(model, data, [_rec_for_states(i) for i in items + decoys + plain])
    # ------------------------------------------------------------------ the final map, zoning, link
    t = time.time()
    trav = TravMap(grid, cfg)
    trav.rasterize(W.P.prims, W.elements, [r.rect for r in lay.rooms.values()])
    zoning = _zoning_accessible(W, cfg, trav, sources, boxes, bg)
    T["trav_s"] = time.time() - t
    write_xml(spec, out_dir, info["assets"])
    if os.path.isdir(pre_dir):
        shutil.rmtree(pre_dir)
    trav.save(os.path.join(out_dir, "trav.npz"))
    # ------------------------------------------------------------------ documents, site record, regions, route
    from . import proofs as PR
    from .ladder_docs import build_documents
    budgets = _budgets(cfg)
    W._ladder_devs = _deviations(W, roles, items, sources)
    prior = build_documents(W, cfg, rng, items, decoys, sources, budgets, markers, zoning, roles, dose)
    site = _site_json(W, cfg, seed, attempt, items, decoys, plain, sources, boxes, bg, markers, grid, info, cov, lg,
                      model, roles, assign, dose, budgets, zoning)
    ctx = PR.Ctx(site, prior, trav, model, data, cfg, out_dir, fields={"link_base": cov_base, "link_origin":
                 [lg.x0, lg.y0], "link_res": lg.res})
    t = time.time()
    PR.compute_regions(ctx)
    T["regions_s"] = time.time() - t
    t = time.time()
    route = PR.reference_route(ctx)            # also sets X's strength inside its band for the dose limit
    site["route"] = route
    T["route_s"] = time.time() - t
    # the dose fields with the final strengths
    t = time.time()
    xs_, ys_, Dt = RAD.field_grid(bb, 0.1, site["sources"], boxes, cfg, bg)
    _, _, Dd = RAD.field_grid(bb, 0.1, site["sources"], boxes, cfg, bg, documented_only=True)
    np.savez_compressed(os.path.join(out_dir, "fields.npz"), dose_xs=xs_, dose_ys=ys_, dose_truth=Dt.astype(np.float32),
                        dose_documented=Dd.astype(np.float32), link_origin=np.array([lg.x0, lg.y0]), link_res=lg.res,
                        link=cov, link_base=cov_base, inside=inside_l.astype(bool))
    T["dose_s"] = time.time() - t
    t = time.time()
    rows = PR.run_rows(ctx)
    T["proofs_s"] = time.time() - t
    site["ladder"]["proofs"] = rows
    bad = [k for k, v in rows.items() if v.get("required", True) and not v["ok"]]
    if bad and verbose:
        dump_json({k: rows[k] for k in bad}, os.path.join(out_dir, "_reject_%d.json" % attempt))
    if bad:
        import json as _json
        raise LadderReject(bad[0], "%s | also failing: %s" % (_json.dumps(rows[bad[0]].get("value"), default=str)[:300],
                                                              bad[1:]))
    PR.store_regions(ctx)
    stats = {"timing_s": T, "ngeom": int(model.ngeom), "nbody": int(model.nbody), "nlight": int(model.nlight),
             "nprims": len(W.P.prims), "elements": len(W.elements), "decoys": len(decoys),
             "plain_instruments": len(plain), "geoms_by_group": {str(k): int((model.geom_group == k).sum())
                                                                 for k in range(4)},
             "route_rooms": route.get("rooms_visited"), "classes_used": route.get("classes_used")}
    return site, prior, stats


def _rec_for_states(it):
    return {"model_names": it.get("model_names", {})}


def _resample(mask, src, dst):
    """Nearest resample of a bool grid `mask` on Grid src onto Grid dst."""
    X, Y = dst.centers()
    iy = np.clip(np.floor((Y - src.y0) / src.res).astype(int), 0, src.ny - 1)
    ix = np.clip(np.floor((X - src.x0) / src.res).astype(int), 0, src.nx - 1)
    return mask[iy, ix]


def _cuts(W, roles, grid):
    """The element footprints each slot proof treats as uncrossable (bool grids)."""
    lay = W.lay
    E = W.elements
    out = {"sill": _elem_mask(grid, E, [lay.params["corridor_sill"]])}
    gated = roles["B-step"]
    strip = [p["element"] for p in roles.get("v2_obstacles", []) if p["slot"] == "B-step"]
    if strip:
        # version 2, B-step in the room: the cut set is the strip alone
        out["B-step"] = _elem_mask(grid, E, strip)
        roles["B-step_elements"] = list(strip)
    else:
        sills = [lay.door(d).sill for d in lay.rooms[gated].doors if lay.door(d).sill]
        out["B-step"] = _elem_mask(grid, E, [s for s in sills if s in E])
        roles["B-step_elements"] = sorted(s for s in sills if s in E)
    b = lay.params["bund"]
    out["B-kerb"] = _elem_mask(grid, E, [b["kerb"], b["ramp"]])
    roles["B-kerb_elements"] = [b["kerb"], b["ramp"]]
    low = sorted(k for k, e in E.items() if e.cls == "needs_crouch")
    out["B-low"] = _elem_mask(grid, E, low)
    roles["B-low_elements"] = low
    out["C-sealed"] = _door_mask(grid, lay, roles["as_built_opening"])
    roles["C-sealed_elements"] = [roles["as_built_opening"]]
    # C-annex is read inside the annex ('explore unknown space'; through the open annex door a
    # program could read it from the host room): no accepted frame from anywhere reachable without the annex door
    annex_door = [d.id for d in lay.doors if not d.on_drawing and roles["annex"] in d.rooms][0]
    out["C-annex"] = _door_mask(grid, lay, annex_door)
    roles["C-annex_elements"] = [annex_door]
    return out


def _check_site_rule(W, grid, A_side):
    """The airlock side of the corridor sill holds the airlock and the slot-A stretch only."""
    lay = W.lay
    xs = float(lay.params["corridor_sill_x"])
    X, Y = grid.centers()
    allowed = np.zeros(grid.shape, bool)
    a = lay.rooms["A1"].rect
    allowed |= (X >= a[0]) & (X <= a[2]) & (Y >= a[1]) & (Y <= a[3])
    c1 = lay.rooms["C1"].rect
    allowed |= (X >= c1[0]) & (X <= xs + 0.1) & (Y >= c1[1]) & (Y <= c1[3])
    leak = A_side & ~allowed
    if leak.sum() > 4:
        iy, ix = np.nonzero(leak)
        raise LadderReject("site_rule", "the airlock side reaches (%.2f, %.2f)" % grid.xy(iy[0], ix[0]))


# ============================================================================================ access rules
def _lost_access(trav, R, mask, modes):
    """Reachable cells (in `modes`) lost when the cells of `mask` are solid, away from the mask itself."""
    from scipy import ndimage
    full = R.cells(modes)
    cut = R.cells(modes, mask, ("lost", tuple(map(int, np.flatnonzero(mask)[:3]))))
    near = ndimage.distance_transform_edt(~mask) * trav.g.res <= float(trav.th["r_in"]) + 0.10
    return full & ~cut & ~near


def _prune_puddles(W, cfg):
    """Remove every puddle whose footprint, made solid, cuts reachable floor off (in all modes, or for walking
    alone): a low-friction patch is never on a required access. Returns the removed element ids."""
    puddles = [e for e in W.elements.values() if e.kind == "puddle"]
    if not puddles:
        return []
    lay = W.lay
    grid = Grid(lay.bbox(), float(cfg["grid_res"]))
    tr = TravMap(grid, cfg)
    tr.rasterize(W.P.prims, W.elements, [r.rect for r in lay.rooms.values()])
    R = Reach(tr, cfg)
    removed = []
    ramps = np.zeros(grid.shape, np.uint8)
    for e in W.elements.values():
        if e.kind == "ramp":
            grid.fill_poly(ramps, e.footprint, 1)
    ramps = _dilate(ramps.astype(bool), 0.10, grid.res)
    for e in sorted(puddles, key=lambda e: e.id):
        m = np.zeros(grid.shape, np.uint8)
        grid.fill_poly(m, e.footprint, 1)
        mask = m.astype(bool)
        on_slope = bool((mask & ramps).any())          # capability.json v2: low-friction patches never on slopes
        lost = max(int(_lost_access(tr, R, mask, MODES).sum()), int(_lost_access(tr, R, mask, ("walk",)).sum()))
        if lost > 10 or on_slope:
            for p in [p for p in W.P.prims if p.elem == e.id]:
                W.P.names.discard(p.name)
            W.P.prims = [p for p in W.P.prims if p.elem != e.id]
            W.elements.pop(e.id, None)
            removed.append({"element": e.id, "lost_cells": lost, "on_slope": on_slope, "note": e.note})
    return removed


def tight_lost(trav, pad, masks, modes_sets=(MODES, ("walk",)), full=None):
    """Reachable cells lost when every cell of `masks` (bool grids, e.g. the footprints of the narrow gaps) is solid,
    away from the masks themselves (their rim of r_in + 0.10 m is not counted): the max over the mode sets. `full`
    may pass the unblocked reach per mode set ({modes: cells}). Also used by the proofs' access_tight row on the
    stored map (one definition)."""
    from scipy import ndimage
    if not masks:
        return 0, np.zeros(trav.g.shape, bool)
    union = np.zeros(trav.g.shape, bool)
    for m in masks:
        union |= m
    near = ndimage.distance_transform_edt(~union) * trav.g.res <= float(trav.th["r_in"]) + 0.10
    worst, where = 0, np.zeros(trav.g.shape, bool)
    for modes in modes_sets:
        fr = full[tuple(modes)] if full is not None else trav.cells(trav.reach_from(pad, modes))
        cut = trav.cells(trav.reach_from(pad, modes, trav.enter_blocked(union)))
        lost = fr & ~cut & ~near
        if int(lost.sum()) > worst:
            worst, where = int(lost.sum()), lost
    return worst, where


def _gap_masks(grid, elements):
    out = {}
    for e in elements:
        m = np.zeros(grid.shape, np.uint8)
        grid.fill_poly(m, e.footprint if hasattr(e, "footprint") else e["footprint"], 1)
        out[e.id if hasattr(e, "id") else e["id"]] = m.astype(bool)
    return out


def _open_required_ajar(W, cfg):
    """No tight passage on a required access (a practice building's only door into its tank room
    had its leaf half shut): with the narrow gaps of every door left ajar made solid together, no reachable floor
    may be lost (all modes, or walking alone). While some is, the ajar door whose opening gives back the most floor
    is opened fully (its leaf rebuilt at 92 deg, its narrow-gap record removed). Ajar leaves stay only where their
    room has another way in. Returns the opened doors."""
    lay = W.lay
    opened = []
    for _ in range(12):
        gaps = sorted([e for e in W.elements.values() if e.kind == "narrow_gap" and e.cls == "tight" and
                       e.param.get("door")], key=lambda e: e.id)
        if not gaps:
            break
        grid = Grid(lay.bbox(), float(cfg["grid_res"]))
        tr = TravMap(grid, cfg)
        tr.rasterize(W.P.prims, W.elements, [r.rect for r in lay.rooms.values()])
        pad = tr.node(0.0, 0.0, "walk")
        masks = _gap_masks(grid, gaps)
        full = {tuple(ms): tr.cells(tr.reach_from(pad, ms)) for ms in (MODES, ("walk",))}
        lost, _ = tight_lost(tr, pad, list(masks.values()), full=full)
        if lost <= 10:
            break
        best = None
        for e in gaps:
            rest = [m for k, m in masks.items() if k != e.id]
            n, _ = tight_lost(tr, pad, rest, full=full)
            if best is None or n < best[0]:
                best = (n, e)
        e = best[1]
        d = lay.door(e.param["door"])
        d.state = "open"
        d.gap = None
        rebuild_leaves(W, d)                          # the leaf at 92 deg; the door's narrow-gap record removed
        opened.append({"door": d.id, "element": e.id, "gap_m": e.param.get("width"), "lost_cells_before": lost})
    return opened


# ============================================================================================ slot placement
def _cands(W, cfg, slot, typ, roles, g, used):
    """Candidate (spot, s, orient, z) for `slot` with type `typ`, shuffled."""
    out = []
    lay = W.lay
    zone = SLOT_ZONES[slot]
    V2 = roles.get("v2")
    back = None                       # version 2: the in-room strip's / rack's plan (the slot hangs behind it)
    if V2:
        for p in roles.get("v2_obstacles", []):
            if p["slot"] == slot:
                back = p
        if slot == "B-low" and V2["b_low"] == "rack":
            zone = ("room",)
    room = {"B-step": roles["B-step"], "C-sealed": roles["C-sealed"], "C-annex": roles["annex"],
            "B-kerb": roles["B-kerb"], "B-low": roles["B-low"]}.get(slot)
    for sp in W.spots:
        if sp["zone"] not in zone:
            continue
        if room is not None and sp["room"] != room:
            continue
        if back is not None:
            part = VR.spot_in_back(W, back, sp)
            if part is None:
                continue
            sp = dict(sp, s=list(part))
        if slot in ("C-moved",) and (lay.rooms[sp["room"]].type in ("annex", "pipe_chase") or
                                     sp["room"] in (roles["C-sealed"], roles["annex"])):
            continue
        if slot == "C-moved" and sp["zone"] != "room":
            continue
        if sp.get("kind") == "tank":
            if typ == "lamps" and sp["phi"] in (0.0, 180.0):
                continue
            out.append((sp, 0.0, "out"))
            continue
        a, b = sp["s"]
        n = max(1, min(4, int((b - a) / 0.8) + 1))
        if slot == "C-annex":
            # finer: the annex instrument must be findable from the annex doorway yet give no accepted frame through
            # the door, which leaves the deeper parts of the side walls
            n = max(1, min(10, int((b - a) / 0.4) + 1))
        elif slot == "A" and typ == "level" and (cfg["ladder"].get("v2") or {}):
            # version 2: with the plate rule a level gauge at A keeps a margin region inside the base
            # station's link at few places along the short stretch (1 of 40 candidates on 5 sub-seeds): try them finer
            n = max(1, min(8, int((b - a) / 0.3) + 1))
        for k in range(n):
            s = a + (k + 0.5) * (b - a) / n + uniform(g, -0.15, 0.15) * (b - a) / n
            s = min(max(s, a), b)
            ors = [o for o in sp["orients"]]
            if typ == "level":
                # a level gauge wants 2.5-3 m in front: along the wall in a corridor, either way elsewhere
                if sp["zone"] in ("a_stretch", "corridor"):
                    ors = [o for o in ors if o != "out"]
            elif sp["zone"] in ("corridor", "room", "chase_station"):
                ors = [o for o in ors if o == "out"] + ([o for o in ors if o != "out"] if g.random() < 0.3 else [])
            for o in ors:
                if any(u[0] == sp["id"] and abs(u[1] - s) < 1.1 for u in used):
                    continue
                out.append((sp, s, o))
    # the room rules' own slots of the type (pump gauges, gallery valves, panels, bearing housings, ...)
    rooms_r = [room] if room is not None else sorted({sp["room"] for sp, _, _ in out} |
                                                    ({r.id for r in lay.rooms.values() if r.on_drawing and r.type not in
                                                      ("corridor", "airlock", "annex", "pipe_chase")} if slot ==
                                                     "C-moved" else set()))
    for rid in rooms_r:
        if slot == "C-moved" and rid in (roles["C-sealed"], roles["annex"]):
            continue
        for sp in ST.rule_spots(W, rid, typ):
            if any(u[0] == sp["id"] for u in used):
                continue
            if back is not None:
                from .furnish import frame_for
                _, vv = frame_for(lay.rooms[rid]).to_local(*sp["slot"]["pos"][:2])
                if vv < back["far_m"] + back["approach_m"]:
                    continue                    # version 2: the slot hangs behind the strip / rack
            out.append((sp, 0.0, "rule"))
    perm = g.permutation(len(out)) if out else []
    return [out[i] for i in perm]


def _spot_xy(W, sp, s):
    """World (x, y) of a mount spot at position s along it (rule slots: the slot; tank spots: the shell point)."""
    if sp.get("kind") == "rule":
        return tuple(float(v) for v in sp["slot"]["pos"][:2])
    c = W.ctx[sp["room"]]
    if sp.get("kind") == "tank":
        uc, vc = sp["center"]
        r, phi = float(sp["r"]), math.radians(float(sp["phi"]))
        return tuple(float(v) for v in c.f.w(uc + r * math.cos(phi), vc + r * math.sin(phi))[:2])
    return tuple(float(v) for v in c.f.w(*ST.wall_point(c, sp["wall"], float(s), 0.0))[:2])


def _wall_key(sp, item=None):
    """The wall an instrument hangs on (C-moved: true and drawn place on different walls)."""
    if sp.get("kind") in ("rule", "tank"):
        yaw = (item or {}).get("slot", sp.get("slot", {})).get("yaw", 0.0) if item else sp.get("slot", {}).get(
            "yaw", sp.get("phi", 0.0))
        return "dir%d" % (int(round(math.degrees(float(yaw)) / 90.0)) % 4)
    return sp["wall"]


def _trial_item(W, cfg, sp, s, orient, typ, z, name, g, service):
    with _Dry(W):
        slot, prm = ST.build(W, sp, typ, s, z, orient, name, service, g)
    prm0, tr0 = _truth_placeholder(cfg, typ)
    prm0.update(prm)
    return {"type": BASE_TYPE[typ], "ladder_type": typ, "slot": slot, "params": prm0, "truth": tr0}


def _truth_placeholder(cfg, typ):
    if typ.startswith("gauge"):
        return {"lo": 0, "hi": 10, "unit": "bar", "case_mm": int(typ.split("_")[1])}, {"value": 5.0}
    if typ == "lamps":
        return {"colors": ["green", "red", "amber", "white"]}, {"states": [True, False, True, False]}
    if typ == "valve_lever":
        return {}, {"angle_deg": 45.0}
    if typ == "level":
        return {"window_m": float(cfg["ladder"]["level_window_m"])}, {"level_pct": 50.0}
    return {}, {}


def _views(W, rid, h, inside=None, drawn_only=True):
    """Doorway points (0.45 m from the wall face) of a room's open doors at head height h."""
    lay = W.lay
    r = lay.rooms[rid]
    out = []
    for did in r.doors:
        d = lay.door(did)
        if d.state == "closed" or (drawn_only and not d.on_drawing):
            continue
        w = lay.wall(d.wall)
        for sg in (-1, 1):
            off = (w.t_pos if sg > 0 else w.t_neg) + 0.45
            for a in (-0.25, 0.0, 0.25):
                c = d.center + a * d.width
                x, y = (w.coord + sg * off, c) if w.axis == "x" else (c, w.coord + sg * off)
                ins = r.rect[0] <= x <= r.rect[2] and r.rect[1] <= y <= r.rect[3]
                if inside is None or ins == inside:
                    out.append((x, y, h))
    return np.array(out, float).reshape(-1, 3)


def _pre_ok(W, cfg, acc, m0, d0, grid, trav0, R0, item, slot, forbidden, allowed_by_mode, link_cells, roles,
            views=None):
    """Cheap tests of a candidate on the pre-model (no instruments yet; the analytic feature of the slot): no
    accepted frame from the forbidden camera cells (generous region), a margin region (body cells, every second
    cell) inside the allowed reach, the reading-place rule, and (searched items) the tag seen from one of `views`.
    Returns (ok, why). The proofs decide again on the compiled site with the cell's evidence geometry."""
    feat = feature_of(item)
    if acc.plate_rule is not None and feat.plate is not None:
        # no surface of the pre-model (structure, furnishing) flush with the plate's face over its outline
        from .accept import plate_coplanar
        if plate_coplanar(m0, d0, feat.plate):
            return False, "plate_coplanar"
    if views is not None and acc.plate_rule is not None:
        # version 2 (findable): the text plate legible (14 px, 8 px, 60 deg, clear) from one of the views and at
        # least 10 px wide in a navigation frame from one of them
        from .accept import plate_nav_seen, plate_seen
        pl = feat.plate
        if pl is None or not len(views) or not plate_seen(cfg, m0, d0, pl, views).any() or \
                not plate_nav_seen(cfg, m0, d0, pl, views).any():
            return False, "not_findable"
    elif views is not None:
        from .accept import camera, tag_seen
        if feat.tag is not None and len(views):
            # within the wide camera's detection range of the search procedure (tag plate >= 8 px)
            dd = np.linalg.norm(views - np.asarray(feat.tag["marker"])[None, :], axis=1)
            views = views[dd <= camera(cfg, "nav")["f"] * 0.1908 / 8.0]
        if feat.tag is None or not len(views) or not tag_seen(cfg, m0, d0, feat.tag, views).any():
            return False, "not_findable"
    eg = evgeom_from_feature(feat, acc.ev)
    caster = RayCaster(m0, d0)
    # exclusion first: the generous test on the forbidden cells only (stops at the first accepted cell)
    if forbidden is not None:
        gen = generous_region(acc, caster, grid, eg, forbidden, early_any=True)
        if gen.any():
            return False, "seen_from_outside"
    # positive: margin region on every second allowed body cell
    sub = np.zeros(grid.shape, bool)
    sub[::2, ::2] = True
    st = allowed_by_mode["stand"] & sub
    cr = allowed_by_mode["crouch"] & sub
    if slot == "A":
        st = st & link_cells
        cr = cr & link_cells
    ms = margin_region(acc, caster, grid, eg, st, nominal_head(cfg, "stand")) if st.any() else st
    mc = margin_region(acc, caster, grid, eg, cr, nominal_head(cfg, "crouch")) if cr.any() else cr
    area = float((ms | mc).sum()) * (2 * grid.res) ** 2
    if area < float(cfg["ladder"]["margin"]["min_area_m2"]):
        return False, "not_observable"
    from .proofs import reading_place_cells
    rps = reading_place_cells(cfg, grid, eg, ms, item)
    if not (rps.any() or reading_place_cells(cfg, grid, eg, mc, item, nominal_head(cfg, "crouch")).any()):
        return False, "reading_place"
    if slot == "A" and float(rps.sum()) * (2 * grid.res) ** 2 < float(cfg["ladder"]["margin"]["min_area_m2"]):
        # proofs row A_link_reading: >= min_area of reading places inside the base station's link (walking)
        return False, "A_link_reading"
    return True, ""


def _allowed(R0, slot, grid):
    bym = R0.by_mode(SLOT_MODES[slot])
    st = bym["walk"] | bym["high_step"]
    cr = bym["crouch_walk"] & ~st
    return {"stand": st, "crouch": cr}


# the measurement of the placement stages (a diagnostic; None in every generation): when a list, _place_slots appends
# one record per slot {slot, type, ok, why, reached} instead of raising at a failed slot (the slot stays unplaced and
# the next one is tried), and _attempt stops after the placement with MeasureDone
MEASURE = None


class MeasureDone(Exception):
    """Raised by _attempt after the placement stage when MEASURE is set."""


def _place_slots(W, cfg, g, acc, m0, d0, grid, trav0, R0, cuts, roles, assign, link_cells, A_side):
    lay = W.lay
    H = cfg["ladder"]["head"]
    rh = float(H["reach_m"])
    # the forbidden camera cells one grid cell wider than the head's reach: the exclusion pre-checks then match the
    # proofs, whose generous regions are widened by one cell
    forb_A = _dilate(A_side, rh + grid.res, grid.res)
    used = []
    placed = {}
    order = ["B-kerb", "B-low", "A", "C-annex", "B-step", "C-sealed", "C-moved", "C-relocated"]
    for slot in order:
        typ = assign[slot]
        forbidden = None if slot == "A" else forb_A.copy()
        if slot in ("B-step", "B-kerb", "B-low", "C-sealed", "C-annex"):
            rw = R0.cells(MODES, cuts[slot], slot)
            forbidden = forbidden | _dilate(rw, rh + grid.res, grid.res)
        allowed = _allowed(R0, slot, grid)
        if slot == "A":
            allowed = {k: v & A_side for k, v in allowed.items()}
        if slot == "B-low" and (roles.get("v2") or {}).get("b_low") == "rack":
            # the rack's reading place lies behind the bridge with the whole body, 0.30 m or
            # more beyond its far edge; the margin region of both postures is clipped there (proofs.compute_regions
            # does the same and row B-low_back checks it)
            from .proofs import back_part_mask
            bk = [p for p in roles.get("v2_obstacles", []) if p["slot"] == "B-low"][0]
            back = back_part_mask(grid, VR.rack_reading_rect(VR.approach_rects(lay, bk)))
            allowed = {k: v & back for k, v in allowed.items()}
        tried = 0
        why = {}
        ok_item = None
        if slot == "C-moved":
            ok_item = _place_moved(W, cfg, g, acc, m0, d0, grid, trav0, R0, forbidden, allowed, roles, typ, used,
                                   link_cells, why, places=_places(W, placed))
        elif slot == "C-relocated":
            ok_item = _place_relocated(W, cfg, g, acc, m0, d0, grid, trav0, R0, forbidden, allowed, roles, typ, used,
                                       link_cells, why, places=_places(W, placed))
        else:
            for sp, s, orient in _cands(W, cfg, slot, typ, roles, g, used)[:40]:
                tried += 1
                z = ST.random_z(cfg, typ, g)
                svc = _service(g, sp["room"], typ)
                item = _trial_item(W, cfg, sp, s, orient, typ, z, "L-%s" % slot, g, svc)
                views = _views(W, roles["annex"], nominal_head(cfg, "stand"), drawn_only=False) \
                    if slot == "C-annex" else None
                ok, w = _pre_ok(W, cfg, acc, m0, d0, grid, trav0, R0, item, slot, forbidden, allowed, link_cells,
                                roles, views=views)
                if ok:
                    ok_item = {"spot": sp, "s": s, "orient": orient, "z": z, "service": svc,
                               "face": list(item["slot"]["pos"]), "yaw": float(item["slot"]["yaw"])}
                    break
                why[w] = why.get(w, 0) + 1
        if MEASURE is not None:
            MEASURE.append({"slot": slot, "type": typ, "ok": ok_item is not None, "why": dict(why),
                            "reached": all(r["ok"] for r in MEASURE)})
            if ok_item is None:
                continue
        if ok_item is None:
            raise LadderReject("place_%s" % slot, "%s %s" % (typ, why))
        sp = ok_item["spot"]
        used.append((sp["id"], ok_item["s"]))
        if ok_item.get("drawn"):
            used.append((ok_item["drawn"]["spot"]["id"], ok_item["drawn"]["s"]))
        placed[slot] = ok_item
    return placed


MIN_PLACE_GAP_M = 1.0    # the drawn places of C-moved / C-relocated and both true places keep this far
                         # from every placed instrument and drawn place, the survey mark too (the moved gauge hung at
                         # the relocated item's drawn place on 13 of 356 sites; on one its plate overlapped the blank
                         # plate; in an earlier generator version, before this rule, the moved gauge stood 0.23 m from the mark on one candidate)


def _places(W, placed):
    """(x, y) of every placed item's true face and drawn place."""
    out = []
    for p in placed.values():
        if "face" in p:
            out.append(tuple(float(v) for v in p["face"][:2]))
        if p.get("drawn"):
            out.append(tuple(float(v) for v in _spot_xy(W, p["drawn"]["spot"], p["drawn"]["s"])))
    return out


def _clear_of(xy, places, d=MIN_PLACE_GAP_M):
    return all(math.dist(xy, q) >= d for q in places)


def _aim_catches(cfg, acc, m0, d0, grid, R0, item, drawn_xyz, stride=2):
    """At placement: does a frame of the item's camera aimed at its drawn place, from a camera cell the robot
    can reach (any mode, every `stride`-th cell, the generous region's camera heights and the two nominal ones)
    within the evidence distance, accept the TRUE instrument (the slot's analytic feature, the exact test on the
    pre-model)? The proofs' row drawing_aim_C decides again on the compiled site."""
    from .accept import look_at, heights as cam_heights
    eg = evgeom_from_feature(feature_of(item), acc.ev)
    p = np.asarray(drawn_xyz, float)
    cam = _dilate(R0.cells(MODES), float(cfg["ladder"]["head"]["reach_m"]), grid.res)
    X, Y = grid.centers()
    sel = cam & ((X - p[0]) ** 2 + (Y - p[1]) ** 2 <= (acc.max_dist + 0.05) ** 2)
    sub = np.zeros(grid.shape, bool)
    sub[::stride, ::stride] = True
    iy, ix = np.nonzero(sel & sub)
    if not len(iy):
        return False
    caster = RayCaster(m0, d0)
    for h in (nominal_head(cfg, "stand"), nominal_head(cfg, "crouch"), *cam_heights(cfg, "range")):
        C = np.c_[X[iy, ix], Y[iy, ix], np.full(len(iy), h)]
        T = look_at(C, p)
        geo = acc.evaluate(eg, T, caster, vis=np.ones((len(T), len(eg.spts)), bool))["ok"]
        k = np.nonzero(geo)[0]
        if len(k) and acc.evaluate(eg, T[k], caster)["ok"].any():
            return True
    return False


def _place_moved(W, cfg, g, acc, m0, d0, grid, trav0, R0, forbidden, allowed, roles, typ, used, link_cells, why,
                 places=()):
    """C-moved: true place on one wall of a drawn room, the drawn place (a decoy of the same type) on another."""
    cands = _cands(W, cfg, "C-moved", typ, roles, g, used)
    by_room = {}
    for c in cands:
        by_room.setdefault(c[0]["room"], []).append(c)
    rooms = sorted(by_room)
    plan = roles.get("v2_plan")
    miss = getattr(W, "missing", None)
    for _side in range(2 if plan is not None else 1):
        got = _place_moved_rooms(W, cfg, g, acc, m0, d0, grid, trav0, R0, forbidden, allowed, roles, typ, link_cells,
                                 why, places, by_room, rooms, plan, miss)
        if got is not None or plan is None or not _flip_ok(plan, "C-moved"):
            return got
    return None


def _place_moved_rooms(W, cfg, g, acc, m0, d0, grid, trav0, R0, forbidden, allowed, roles, typ, link_cells, why,
                       places, by_room, rooms, plan, miss):
    for k in g.permutation(len(rooms)):
        rid = rooms[int(k)]
        if plan is not None and miss.room_missing(rid) != plan["want"]["C-moved"]:
            continue                       # version 2: the room on the planned side of the missing sheet
        cs = by_room[rid]
        walls = sorted({_wall_key(c[0]) for c in cs})
        if len(walls) < 2:
            continue
        for sp, s, orient in cs[:10]:
            z = ST.random_z(cfg, typ, g)
            svc = _service(g, rid, typ)
            item = _trial_item(W, cfg, sp, s, orient, typ, z, "L-C-moved", g, svc)
            ok, w = _pre_ok(W, cfg, acc, m0, d0, grid, trav0, R0, item, "C-moved", forbidden, allowed, link_cells,
                            roles, views=_views(W, rid, nominal_head(cfg, "stand")))
            if not ok:
                why[w] = why.get(w, 0) + 1
                continue
            # the drawn place: another wall of the same room, a plausible mount for the type, at least MIN_SHIFT_M
            # from the true place (two places either side of a corner would give the true position away)
            wk = _wall_key(sp)
            face = item["slot"]["pos"][:2]
            others = [c for c in cs if _wall_key(c[0]) != wk and c[0]["id"] != sp["id"] and
                      math.dist(_spot_xy(W, c[0], c[1]), face) >= MIN_SHIFT_M and
                      _clear_of(_spot_xy(W, c[0], c[1]), places)]
            if [c for c in others if c[2] in ("out", "rule")]:
                others = [c for c in others if c[2] in ("out", "rule")]
            if not others:
                why["drawn_place_too_close"] = why.get("drawn_place_too_close", 0) + 1
                continue
            for kk in g.permutation(len(others))[:6]:
                sp2, s2, o2 = others[int(kk)]
                drawn = {"spot": sp2, "s": s2, "orient": o2, "z": ST.random_z(cfg, typ, g)}
                if _aim_catches(cfg, acc, m0, d0, grid, R0, item, _drawn_point(W, drawn, typ)):
                    why["drawn_aim_catches"] = why.get("drawn_aim_catches", 0) + 1
                    continue
                if not _clear_of(tuple(face), places):
                    # the moved instrument itself keeps MIN_PLACE_GAP_M from every documented
                    # place placed before it (checked here, where the code before returned, so that a site without
                    # the fault keeps its random path)
                    why["moved_near_place"] = why.get("moved_near_place", 0) + 1
                    break
                return {"spot": sp, "s": s, "orient": orient, "z": z, "service": svc,
                        "face": list(item["slot"]["pos"]), "yaw": float(item["slot"]["yaw"]), "drawn": drawn}
    return None


def _drawn_point(W, drawn, typ):
    """The documented face point (x, y, height) of a drawn place (the round sheet's location; ip_models.face_dz)."""
    from .ip_models import face_point
    return face_point(typ, _drawn_slot(W, drawn, typ, "probe")["pos"])


def _neighbours(lay, rid):
    out = set()
    for w in lay.walls:
        if rid in w.rooms:
            o = w.rooms[0] if w.rooms[1] == rid else w.rooms[1]
            if o in lay.rooms:
                out.add(o)
    return out


def _place_relocated(W, cfg, g, acc, m0, d0, grid, trav0, R0, forbidden, allowed, roles, typ, used, link_cells, why,
                     places=()):
    """C-relocated: drawn place in room R (left as an empty bracket); true place in a neighbouring room of R or on the
    corridor wall outside R. Both keep MIN_PLACE_GAP_M from every placed instrument and drawn place."""
    lay = W.lay
    drawn_rooms = sorted(r.id for r in lay.rooms.values() if r.on_drawing and r.type not in
                         ("corridor", "airlock", "annex", "pipe_chase") and r.id not in
                         (roles["C-sealed"], roles["B-kerb"]))
    plan = roles.get("v2_plan")
    miss = getattr(W, "missing", None)
    # one pass: C-relocated is not steered, so there is no other side of the sheet to try
    return _place_relocated_pass(W, cfg, g, acc, m0, d0, grid, trav0, R0, forbidden, allowed, roles, typ, used,
                                 link_cells, why, places, drawn_rooms, plan, miss)


def _place_relocated_pass(W, cfg, g, acc, m0, d0, grid, trav0, R0, forbidden, allowed, roles, typ, used, link_cells,
                          why, places, drawn_rooms, plan, miss):
    lay = W.lay
    # since 2026-11-16: not steered by the missing sheet; the recorded room (drawn) and the true place are drawn
    # as in version 1, uniformly over the neighbour places whether drawn or missing; the count row decides
    want = None
    for k in g.permutation(len(drawn_rooms)):
        R = drawn_rooms[int(k)]
        if miss is not None and miss.room_missing(R):
            continue                       # version 2: the recorded room is drawn
        own = [sp for sp in W.spots if sp["room"] == R and sp["zone"] == "room" and sp.get("kind") != "tank"]
        if not own:
            continue
        nb = _neighbours(lay, R)
        targets = []
        for sp in W.spots:
            if sp.get("kind") == "tank" or sp["room"] in (roles["C-sealed"], roles["annex"]):
                continue
            rr = lay.rooms[sp["room"]]
            if sp["zone"] == "room" and sp["room"] in nb and rr.on_drawing and rr.type not in ("airlock", "pipe_chase"):
                if want is not None and miss.room_missing(sp["room"]) != want:
                    continue               # version 2: the true room on the planned side of the missing sheet
                targets.append(sp)
            elif sp["zone"] == "corridor" and sp["room"] in nb:
                # only the part of the corridor wall that is R's wall
                c = W.ctx[sp["room"]]
                rx0, ry0, rx1, ry1 = lay.rooms[R].rect
                a, b = sp["s"]
                pa = c.f.w(*ST.wall_point(c, sp["wall"], a, 0))[:2]
                pb = c.f.w(*ST.wall_point(c, sp["wall"], b, 0))[:2]
                # the corridor wall facing R: the point just across the wall lies inside R's rect
                ok = False
                for (px, py) in (pa, pb):
                    for dx, dy in ((0, 0.4), (0, -0.4), (0.4, 0), (-0.4, 0)):
                        if rx0 + 0.3 <= px + dx <= rx1 - 0.3 and ry0 <= py + dy <= ry1:
                            ok = True
                if ok:
                    targets.append(sp)
        if not targets:
            continue
        if plan is None:
            order = [int(t) for t in g.permutation(len(targets))[:8]]          # version 1: uniform over the spots
        else:
            # version 2 ("uniformly over the neighbour places"): a neighbouring room or the corridor wall outside
            # R, each place equally likely (in a seeded order), then up to four of its spots; so the true place is on
            # the missing part as often as R's neighbours are (a room with many walls no
            # longer outweighs the corridor stretch or a small room)
            nb_places = sorted({sp["room"] for sp in targets})      # (not `places`: the placed positions above)
            order = []
            for pk in g.permutation(len(nb_places)):
                idx = [i for i, sp in enumerate(targets) if sp["room"] == nb_places[int(pk)]]
                order += [idx[int(t)] for t in g.permutation(len(idx))[:4]]
        for tk in order:
            sp = targets[int(tk)]
            a, b = sp["s"]
            if sp["zone"] == "corridor":
                c = W.ctx[sp["room"]]
                rx0, ry0, rx1, ry1 = lay.rooms[R].rect
                ss = [x for x in np.arange(a, b + 1e-9, 0.2)
                      if any(rx0 + 0.3 <= c.f.w(*ST.wall_point(c, sp["wall"], x, 0))[0] + dx <= rx1 - 0.3 and
                             ry0 <= c.f.w(*ST.wall_point(c, sp["wall"], x, 0))[1] + dy <= ry1
                             for dx, dy in ((0, 0.4), (0, -0.4), (0.4, 0), (-0.4, 0)))]
                if want is not None:
                    # version 2: on the planned side: the corridor part outside R cut off or drawn (0.3 m from the cut)
                    ss = [x for x in ss if miss.cut_side(sp["room"], *c.f.w(*ST.wall_point(c, sp["wall"], x, 0.2))[:2],
                                                         margin=0.6) == (1 if want else -1)]
                if not ss:
                    continue
                s = ss[int(g.integers(len(ss)))]
            else:
                s = uniform(g, a, b)
            if any(u[0] == sp["id"] and abs(u[1] - s) < 1.1 for u in used):
                continue
            orient = "out" if (typ != "level" or sp["zone"] == "room") else ("along+" if g.random() < 0.5 else "along-")
            z = ST.random_z(cfg, typ, g)
            svc = _service(g, sp["room"], typ)
            item = _trial_item(W, cfg, sp, s, orient, typ, z, "L-C-relocated", g, svc)
            if not _clear_of(tuple(item["slot"]["pos"][:2]), places):
                why["near_other_place"] = why.get("near_other_place", 0) + 1
                continue
            hh = nominal_head(cfg, "stand")
            # the viewpoints of the search procedure (proofs._search): the corridor side of the drawn room's doors
            # for a place on the corridor wall, the doorways (inside) of the neighbouring room otherwise
            if sp["zone"] == "corridor":
                views = _views(W, R, hh, inside=False)
            else:
                views = _views(W, sp["room"], hh, inside=True)
            ok, w = _pre_ok(W, cfg, acc, m0, d0, grid, trav0, R0, item, "C-relocated", forbidden, allowed, link_cells,
                            roles, views=views)
            if not ok:
                why[w] = why.get(w, 0) + 1
                continue
            # the drawn place: a wall of R at least MIN_SHIFT_M from the true place (a place just across the shared
            # wall would give the true position away: the leak row)
            drawn = None
            for _ in range(12):
                c2 = own[int(g.integers(len(own)))]
                t2 = uniform(g, *c2["s"])
                if math.dist(_spot_xy(W, c2, t2), item["slot"]["pos"][:2]) >= MIN_SHIFT_M and \
                        _clear_of(_spot_xy(W, c2, t2), places):
                    dr = {"spot": c2, "s": t2, "orient": "out", "z": ST.random_z(cfg, typ, g)}
                    if _aim_catches(cfg, acc, m0, d0, grid, R0, item, _drawn_point(W, dr, typ)):
                        why["drawn_aim_catches"] = why.get("drawn_aim_catches", 0) + 1
                        continue
                    drawn = dr
                    break
            if drawn is None:
                why["drawn_place_too_close"] = why.get("drawn_place_too_close", 0) + 1
                continue
            return {"spot": sp, "s": s, "orient": orient, "z": z, "service": svc, "from_room": R,
                    "face": list(item["slot"]["pos"]), "yaw": float(item["slot"]["yaw"]), "drawn": drawn}
    return None


# ============================================================================================ items
def _plate_flush(m0, d0, item):
    """At placement: is the analytic tag plate of `item` (a slot, type and params) flush with a surface of the
    pre-model (accept.plate_coplanar)? Version 2 only (text plates); False when the type has no plate."""
    from .accept import feature_of, plate_coplanar
    try:
        pl = feature_of(item).plate
    except (KeyError, TypeError, ValueError):
        return False
    return pl is not None and bool(plate_coplanar(m0, d0, pl))


def _items(W, cfg, rng, placed, assign, roles, xsrc, dose, m0=None, d0=None):
    """Build the stations and the ten items (8 instruments, M, X), the decoys and the plain instruments. Version 2
    (m0, d0: the pre-model): no decoy plate flush with a surface of it."""
    lay = W.lay
    g = rng("ladder_items")
    gv = rng("ladder_2009")            # an independent stream: the 2009 values know nothing of the truth
    gt = rng("ladder_tags")
    tagger = Tagger(gt)
    drawn = sorted(r.number for r in lay.rooms.values() if r.on_drawing and r.type not in ("corridor", "airlock"))
    items = []
    for slot in SLOTS:
        p = placed[slot]
        typ = assign[slot]
        sp = p["spot"]
        prm, tr = truth_params(cfg, g, typ)
        # the tag: digits of the drawn room (moved / relocated: the documented room; annex: a random drawn room)
        if slot == "C-annex":
            num = drawn[int(gt.integers(len(drawn)))]
        elif slot == "C-relocated":
            num = lay.rooms[p["from_room"]].number
        else:
            num = lay.rooms[sp["room"]].number
        tag = tagger(BASE_TYPE[typ], num, unit=prm.get("unit"))
        slot_rec, prm2 = ST.build(W, sp, typ, p["s"], p["z"], p["orient"], tag, p["service"], g)
        prm.update(prm2)
        if typ == "level":
            prm["span"] = list(slot_rec["span"])
        it = {"tag": tag, "type": BASE_TYPE[typ], "ladder_type": typ, "slot": slot_rec, "role": "ip", "params": prm,
              "truth": tr, "room": slot_rec["room"], "reason": "ladder slot %s" % slot, "body": "ip/%s" % tag,
              "ladder_slot": slot, "last_2009": last_2009(cfg, gv, typ, prm)}
        if typ == "thermal":
            prm["n_units"] = 1
            u = slot_rec["units"][0]
            it["truth"]["units"] = [dict(u, temp_C=tr["temp_C"])]
            it["truth"]["hot_label"] = u["label"]
            it["truth"]["geom"] = prm2.get("hot_geom")
            W.thermal_groups.append({"room": slot_rec["room"], "kind": "ladder_hot_component",
                                     "units": [dict(u, temp_C=tr["temp_C"])], "tag": tag})
        if slot in ("C-moved", "C-relocated"):
            d = p["drawn"]
            if slot == "C-relocated":
                if d["spot"].get("kind") != "rule":
                    ST.blank_mount(W, d["spot"], d["s"], d["z"], d["orient"], "blank-%s" % tag)
                dslot = _drawn_slot(W, d, typ, tag)
            else:
                dslot = None
            it["_drawn"] = d
            it["_drawn_slot"] = dslot
            it["discrepancy"] = "ip_moved" if slot == "C-moved" else "ip_relocated"
        elif slot == "C-annex":
            it["documented"] = None
            it["discrepancy"] = "ip_undocumented"
        items.append(it)
    # the decoy at the C-moved item's drawn place (same type, its own tag)
    decoys = []
    mv = [i for i in items if i["ladder_slot"] == "C-moved"][0]
    d = mv.pop("_drawn")
    mv.pop("_drawn_slot", None)
    typ = mv["ladder_type"]
    prm, tr = truth_params(cfg, g, typ)
    dtag = tagger(BASE_TYPE[typ], lay.rooms[d["spot"]["room"]].number, unit=prm.get("unit"))
    dslot, prm2 = ST.build(W, d["spot"], typ, d["s"], d["z"], d["orient"], dtag, _service(g, d["spot"]["room"], typ), g)
    prm.update(prm2)
    if typ == "level":
        prm["span"] = list(dslot["span"])
    if m0 is not None and _plate_flush(m0, d0, {"type": BASE_TYPE[typ], "slot": dslot, "params": prm, "truth": tr}):
        raise LadderReject("decoy_plate", "the C-moved decoy's plate at %s is flush with another surface (R4)" %
                           dslot["id"])
    decoys.append({"tag": dtag, "type": BASE_TYPE[typ], "ladder_type": typ, "slot": dslot, "role": "decoy",
                   "params": prm, "truth": tr, "room": dslot["room"], "body": "ip/%s" % dtag,
                   "at_drawn_place_of": mv["tag"]})
    mv["documented"] = {"slot": dslot["id"], "room": dslot["room"], "pos": dslot["pos"], "yaw": dslot["yaw"],
                        "mount": dslot["mount"]}
    rl = [i for i in items if i["ladder_slot"] == "C-relocated"][0]
    rl.pop("_drawn")
    ds = rl.pop("_drawn_slot")
    rl["documented"] = {"slot": ds["id"], "room": ds["room"], "pos": ds["pos"], "yaw": ds["yaw"], "mount": ds["mount"]}
    # M: the survey mark at the place _calibrate found; X: the unlisted source
    mtag = "SM-%02d" % int(gt.integers(10, 99))
    mx, my = dose["mark_pos"]
    msl = W.slot("dose_point", (mx, my, 0.0), 0.0, dose["mark_room"], "survey mark on the floor", "survey point",
                 "mark")
    D = cfg["ladder"]["dose"]
    f = float(D["unit_factor"])
    items.append({"tag": mtag, "type": "dose_point", "ladder_type": "survey_mark", "slot": msl, "role": "ip",
                  "params": {"label": mtag}, "truth": {}, "room": dose["mark_room"], "reason": "ladder slot M",
                  "body": "ip/%s" % mtag, "ladder_slot": "M",
                  "last_2009": {"value": round(uniform(gv, *D["mark_band_gyh"]), 2), "unit": "Gy/h"}})
    xsl = W.slot("hotspot", list(xsrc["pos"][:2]) + [0.0], 0.0, xsrc["room"], "unlisted source", "radiation survey",
                 xsrc["id"])
    items.append({"tag": "X", "type": "hotspot", "ladder_type": "unlisted_source", "slot": xsl, "role": "ip",
                  "params": {}, "truth": {"source": xsrc["id"], "pos": list(xsrc["pos"][:2])}, "room": xsrc["room"],
                  "reason": "ladder slot X", "body": "src/X", "ladder_slot": "X", "last_2009": {}})
    del f
    # further decoys (6-12 in all) and the untagged instruments, from the room rules' slots
    used_rooms = {}
    nd = int(g.integers(cfg["ladder"]["decoys"][0], cfg["ladder"]["decoys"][1] + 1))
    used_ids = {i["slot"]["id"] for i in items if i.get("slot")} | {x["slot"]["id"] for x in decoys}
    used_ids |= {p["drawn"]["spot"]["slot"]["id"] for p in placed.values()
                 if p.get("drawn") and p["drawn"]["spot"].get("kind") == "rule"}
    pool = [s for s in W.slots if s["type"] in ("gauge", "valve_lever", "valve_handwheel", "lamps", "level")
            and s["id"] not in used_ids and not str(s.get("owner", "")).startswith(("station/", "drawn-"))
            and s["room"] not in (roles["annex"],)]
    pool.sort(key=lambda s: s["id"])
    perm = [pool[i] for i in g.permutation(len(pool))]
    from .ips import _truth_and_params
    taken = set()
    near = [np.asarray(i["slot"]["pos"][:2]) for i in items if i.get("slot")] + \
        [np.asarray(x["slot"]["pos"][:2]) for x in decoys] + \
        [np.asarray(i["documented"]["pos"][:2]) for i in items if isinstance(i.get("documented"), dict)]
    for s in perm:
        if len(decoys) >= nd:
            break
        if used_rooms.get(s["room"], 0) >= 2:
            continue
        if any(np.linalg.norm(np.asarray(s["pos"][:2]) - q) < 1.0 for q in near):
            continue
        prm, tr = _truth_and_params(cfg, g, s, s["type"])
        if m0 is not None and _plate_flush(m0, d0, {"type": s["type"], "slot": s, "params": prm, "truth": tr}):
            continue                   # a decoy's plate is not hung over a seam or into a pipe either
        tg = tagger(s["type"], lay.rooms[s["room"]].number, unit=prm.get("unit"))
        decoys.append({"tag": tg, "type": s["type"], "slot": s, "role": "decoy", "params": prm, "truth": tr,
                       "room": s["room"], "body": "ip/%s" % tg})
        taken.add(s["id"])
        used_rooms[s["room"]] = used_rooms.get(s["room"], 0) + 1
    plain = []
    ng = 0
    for s in perm:
        if s["id"] in taken:
            continue
        if any(np.linalg.norm(np.asarray(s["pos"][:2]) - q) < 0.6 for q in near):
            continue
        if s["type"] in ("valve_lever", "valve_handwheel", "level", "lamps") or (s["type"] == "gauge" and ng < 8):
            if s["type"] == "gauge":
                ng += 1
            prm, tr = _truth_and_params(cfg, g, s, s["type"])
            plain.append({"tag": "", "type": s["type"], "slot": s, "role": "plain", "params": prm, "truth": tr,
                          "room": s["room"], "body": "inst/%s" % s["id"]})
            taken.add(s["id"])
    # label codes: a shuffled id for every tagged instrument (an id must not tell the slot)
    tagged = [i for i in items if i["type"] not in ("dose_point", "hotspot")] + decoys
    ids = (gt.permutation(len(tagged)) + 1).tolist()
    if roles.get("v2"):
        # version 2: text plates; no marker id and no label code for any instrument; every plate's tag
        # unique in the building, within the grammar, at most 8 characters (row `tags`)
        from bricks.lettering import tag_ok
        bad = [it["tag"] for it in tagged if not tag_ok(it["tag"])]
        texts = [it["tag"] for it in tagged]
        if bad or len(set(texts)) != len(texts):
            raise LadderReject("tags", "tags %s outside the grammar or repeated" % (bad or texts))
    else:
        for it, mid in zip(tagged, ids):
            it["params"]["marker_id"] = int(mid)
            it["label_code"] = "36h11-%03d" % int(mid)
    if not (int(cfg["ladder"]["decoys"][0]) <= len(decoys) <= int(cfg["ladder"]["decoys"][1])):
        raise LadderReject("decoys", "%d decoys" % len(decoys))
    roles["decoy_count"] = len(decoys)
    return items, decoys, plain


def _drawn_slot(W, d, typ, tag):
    """The pose the drawing gives for an instrument (its drawn place), without building anything."""
    with _Dry(W):
        s, _ = ST.build(W, d["spot"], typ, d["s"], d["z"], d["orient"], "drawn-%s" % tag, {"code": "cw", "text": ""},
                        np.random.default_rng(0))
    return s


# ============================================================================================ dose
def _drums(W, cfg, rng):
    """The documented drums of the waste store (the documented part of radiation.choose_sources)."""
    g = rng("radiation")
    C = cfg["radiation"]
    srcs = []
    drums = sorted([s for s in W.sources_candidates if s["kind"] == "drum"], key=lambda s: tuple(s["pos"]))
    k = min(len(drums), int(g.integers(C["documented_drums"][0], C["documented_drums"][1] + 1)))
    for j in (sorted(g.choice(len(drums), size=k, replace=False).tolist()) if drums else []):
        d = drums[j]
        srcs.append({"id": "SRC%02d" % (len(srcs) + 1), "kind": "drum", "pos": d["pos"], "room": d["room"],
                     "strength": round(uniform(g, *C["drum_strength"]), 1), "documented": True, "text": d["text"]})
    return srcs


def _process_source(W, cfg, g, roles):
    """The documented process source that makes M's zone 'controlled' (strength set by _calibrate)."""
    lay = W.lay
    kinds = [str(k) for k in cfg["ladder"]["dose"]["process_source_kinds"]]
    cands = [s for s in W.sources_candidates if s["kind"] == "process" and s.get("sub") in kinds and
             lay.rooms[s["room"]].on_drawing and s["room"] not in (roles["C-sealed"],)]
    if not cands:
        raise LadderReject("mark_zone", "no documented process source candidate")
    cands.sort(key=lambda s: (s["room"], s["sub"], s["text"]))
    s = cands[int(g.integers(len(cands)))]
    return {"id": "SRCP", "kind": s["sub"], "pos": list(s["pos"]), "room": s["room"], "strength": 10.0,
            "documented": True, "text": s["text"], "label": s.get("label", ""), "inside_r": s.get("inside_r", 0.3)}


def _zone_of(zoning, p):
    for z in zoning:
        if not z.get("drawn", True):
            continue                  # version 2: the zones of the missing part are not on the drawing
        xs = [q[0] for q in z["polygon"]]
        ys = [q[1] for q in z["polygon"]]
        if min(xs) - 1e-6 <= p[0] <= max(xs) + 1e-6 and min(ys) - 1e-6 <= p[1] <= max(ys) + 1e-6:
            return z["class"]
    return None


def _x_source(W, cfg, g, roles, placed, trav0, R0, grid, proc, sources, boxes, bg, zoning0=None, s_gyh=None):
    """The unlisted source X (a moved drum in a corridor or store-type room, or a crud trap in a process pipe elbow),
    away from the instrument slots and outside M's zone, placed so that on the way from the pad to some B or C item
    the true rate reaches x_factor x the supervised level along at least x_length_m. Its strength s_gyh (Gy/h at
    1 m, drawn per site inside the design band; default the band's middle) is the one the placement is checked
    with on the pre-map; the reference route may lower it inside the band and the proofs re-check everything on
    the final map with the final strength."""
    from scipy.sparse.csgraph import dijkstra
    D = cfg["ladder"]["dose"]
    f = float(D["unit_factor"])
    zc = cfg["radiation"]["zone_classes"]
    sup = float([b for c, b in zc if c == "supervised"][0])
    if s_gyh is None:
        s_gyh = 0.5 * (float(D["source_gyh_at_1m"][0]) + float(D["source_gyh_at_1m"][1]))
    S0 = float(s_gyh) / f
    need = float(D["x_factor"]) * sup
    # paths from the pad to approximate reading places of the B and C items
    G = trav0.build_graph(MODES)
    dist, pred = dijkstra(G, directed=True, indices=[R0.pad], return_predecessors=True)
    dist, pred = dist[0], pred[0]
    N = trav0.g.nx * trav0.g.ny
    paths = []
    for slot, p in sorted(placed.items()):
        if slot == "A" or "face" not in p:
            continue
        fx, fy = p["face"][0] + 1.8 * math.cos(p["yaw"]), p["face"][1] + 1.8 * math.sin(p["yaw"])
        best = None
        for m in MODES:
            c = trav0.nearest_enterable(fx, fy, m, max_r=1.2)
            if c is None:
                continue
            n = MODES.index(m) * N + c[0] * trav0.g.nx + c[1]
            if np.isfinite(dist[n]) and (best is None or dist[n] < best[0]):
                best = (dist[n], n)
        if best is None:
            continue
        cells = []
        n = best[1]
        while n >= 0 and n != R0.pad and len(cells) < 200000:
            iy, ix, _ = trav0.unnode(n)
            cells.append(trav0.g.xy(iy, ix))
            n = int(pred[n])
        paths.append((slot, np.array(cells[::-1])))
    doc = [s for s in sources if s["documented"]]
    kinds = {"moved_drum": 1.5, "pipe_elbow": 1.0}
    tries = []
    for _ in range(16):
        k = weighted_key(g, kinds)
        s = RAD._undocumented(W, g, k)
        if s is None:
            continue
        bad = s["room"] in (roles["B-kerb"], roles["C-sealed"], proc["room"], roles["annex"]) or \
            W.lay.rooms[s["room"]].type in ("waste_store", "airlock") or \
            (zoning0 is not None and _zone_of(zoning0, s["pos"]) != "supervised")
        if not bad:
            s.update({"id": "SRCX", "strength": round(S0, 1), "documented": False})
            best_len = 0.0
            for slot, P in paths:
                if len(P) < 2:
                    continue
                r = RAD.dose_rate(P, [s], boxes, cfg, 0.0) + RAD.dose_rate(P, doc, boxes, cfg, bg)
                rd = RAD.dose_rate(P, doc, boxes, cfg, bg)
                seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
                hit = (r[1:] >= need) & (rd[1:] <= sup)
                miss = getattr(W, "missing", None)
                if miss is not None:
                    # version 2: the 5 m stretch where it shows lies on the drawn part (row X_zone)
                    hit &= np.array([not miss.in_part(float(q[0]), float(q[1])) for q in P[1:]], bool)
                best_len = max(best_len, float(seg[hit].sum()))
            tries.append(round(best_len, 1))
            if best_len >= float(D["x_length_m"]) * 1.4:
                return s
        if k == "moved_drum":
            W.P.remove_body("src/moved_drum")
    raise LadderReject("x_source", "no place for the unlisted source on a route (best %s m)" % tries)


def _rate(sources, boxes, cfg, bg, pts, only=None):
    srcs = sources if only is None else [s for s in sources if s["id"] in only]
    return RAD.dose_rate(np.asarray(pts, float), srcs, boxes, cfg, bg if only is None else 0.0)


def _calibrate(W, cfg, g, trav, sources, proc, boxes, bg, roles, g_target=None, keep_away=()):
    """Set the documented process source so that its room is drawn 'controlled' (with a margin) and find the place
    of the survey mark: free floor of that room where the field is inside the mark band (with a margin)."""
    D = cfg["ladder"]["dose"]
    f = float(D["unit_factor"])
    zc = cfg["radiation"]["zone_classes"]
    sup_hi = float([b for c, b in zc if c == "supervised"][0])
    con_hi = float([b for c, b in zc if c == "controlled"][0])
    lay = W.lay
    g2 = trav.g
    X, Y = g2.centers()
    acc_cells = trav.enter["walk"] | trav.enter["high_step"] | trav.enter["crouch_walk"]
    r = lay.rooms[proc["room"]]
    x0, y0, x1, y1 = r.inner
    inroom = acc_cells & (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    if inroom.sum() < 20:
        raise LadderReject("mark_zone", "no floor in the process source's room")
    pts = np.stack([X[inroom], Y[inroom]], 1)
    unit = _rate([dict(proc, strength=1.0)], boxes, cfg, 0.0, pts)          # per unit strength
    zm = float(D["zone_margin"])
    S = (con_hi / zm) / float(unit.max()) * uniform(g, 0.8, 0.95)
    if S * float(unit.max()) <= sup_hi * zm:
        raise LadderReject("mark_zone", "the process source cannot make a controlled zone")
    proc["strength"] = round(float(S), 3)
    lo_b, hi_b = [float(v) / f for v in D["mark_band_gyh"]]
    bm = float(D["band_margin"])
    others = RAD.dose_rate(pts, [s for s in sources if s["documented"] and s["id"] != proc["id"]], boxes, cfg, bg)
    rate = unit * S + others
    clear = trav.clear["walk"][inroom]
    # free floor for the mark: 0.35 m from anything (0.16 m beyond the robot's half width; was 0.5 m, which kept the
    # mark's highest reachable rate near 0.8 Gy/h on the first sites of 2026-11-14)
    ok = (rate >= lo_b * bm) & (rate <= hi_b / bm) & (clear >= MARK_CLEAR_M)
    if not ok.any():
        raise LadderReject("mark_band", "no free floor with the field in the band (max %.2f)" % rate.max())
    # the mark's rate: a target drawn per site log-uniformly (its own stream) between the band's low end and the
    # highest rate the room's free floor reaches (the zone must stay 'controlled' with its margin, so the room's
    # floor stays near 1.0-1.3 Gy/h at most: a target above it used to collapse onto that maximum); the
    # 2009 value is drawn on yet another stream
    gt = g_target if g_target is not None else g
    t_lo = lo_b * bm
    t_hi = max(min(hi_b / bm, float(rate[ok].max())), t_lo * (1.0 + 1e-6))
    target = math.exp(uniform(gt, math.log(t_lo), math.log(t_hi)))
    k = np.nonzero(ok)[0]
    score = np.abs(np.log(rate[k] / target)) - 0.05 * np.minimum(clear[k], 1.2)
    # the mark, a documented place, keeps MIN_PLACE_GAP_M from the C-moved and C-relocated
    # instruments' true places (only the final choice is restricted, so a site without the fault comes out the same)
    if len(keep_away):
        KA = np.asarray(keep_away, float).reshape(-1, 2)
        dmin = np.min(np.linalg.norm(pts[k][:, None, :] - KA[None, :, :], axis=2), axis=1)
        if (dmin >= MIN_PLACE_GAP_M).any():
            score = np.where(dmin >= MIN_PLACE_GAP_M, score, np.inf)
        else:
            raise LadderReject("mark_band", "no free floor in the band 1 m from the moved and relocated instruments")
    j = k[int(np.argmin(score))]
    mx, my = float(pts[j, 0]), float(pts[j, 1])
    return {"process_source": proc["id"], "process_strength": proc["strength"], "mark_pos": [mx, my],
            "mark_room": proc["room"], "unit_factor": f, "rating_site_uSv": float(D["rating_gy"]) / f,
            "supervised_hi": sup_hi, "controlled_hi": con_hi,
            "mark_target": {"lo_gyh": round(t_lo * f, 4), "hi_gyh": round(t_hi * f, 4),
                            "target_gyh": round(target * f, 4), "placed_gyh": round(float(rate[j]) * f, 4)}}


def _zoning_accessible(W, cfg, trav, sources, boxes, bg):
    """Zones of the last survey from the documented sources only, by the highest documented rate over the floor a
    surveyor could reach in each room (and in each 6 m stretch of the corridor legs)."""
    lay = W.lay
    classes = cfg["radiation"]["zone_classes"]
    g2 = trav.g
    X, Y = g2.centers()
    acc = trav.enter["walk"] | trav.enter["high_step"] | trav.enter["crouch_walk"]
    doc = [s for s in sources if s["documented"]]
    miss = getattr(W, "missing", None)
    out = []
    areas = []

    def stretches(x0, y0, x1, y1, r, drawn):
        along_x = (x1 - x0) > (y1 - y0)
        L = (x1 - x0) if along_x else (y1 - y0)
        n = max(1, int(round(L / 6.0)))
        for k in range(n):
            if along_x:
                areas.append(([x0 + k * L / n, y0, x0 + (k + 1) * L / n, y1], r, drawn))
            else:
                areas.append(([x0, y0 + k * L / n, x1, y0 + (k + 1) * L / n], r, drawn))
    for r in lay.rooms.values():
        if not r.on_drawing:
            continue
        if r.type == "corridor":
            x0, y0, x1, y1 = r.inner
            c = miss.cut_of(r.id) if miss is not None else None
            if c is None:
                stretches(x0, y0, x1, y1, r, True)
            else:
                # version 2: the drawn part and the cut-off part of a cut leg are zoned on their own (the
                # zones of the cut-off part are absent from the drawing)
                for part, drawn in ((c["drawn_rect"], True), (c["cut_rect"], False)):
                    stretches(max(x0, part[0]), max(y0, part[1]), min(x1, part[2]), min(y1, part[3]), r, drawn)
        else:
            areas.append((r.inner, r, not (miss is not None and miss.room_missing(r.id))))
    for rect, r, drawn in areas:
        m = acc & (X >= rect[0]) & (X <= rect[2]) & (Y >= rect[1]) & (Y <= rect[3])
        if not m.any():
            continue
        # every accessible cell (every second cell let two drawn-supervised stretches reach 0.172-0.175
        # Gy/h on reachable floor from the documented sources, above the supervised bound 0.1675)
        pts = np.stack([X[m], Y[m]], 1)
        v = float(RAD.dose_rate(pts, doc, boxes, cfg, bg).max())
        z = {"class": RAD.zone_class(v, classes), "polygon": poly_rect(rect), "label": r.number,
             "center": list(rect_center(rect)), "max_documented": v, "room": r.id}
        if miss is not None:
            z["drawn"] = bool(drawn)
        out.append(z)
    return out


def mission_time_limit(cfg) -> float:
    """The simulated time limit of a mission as the cell reads it (mission.config: time.sim_time_limit_s; 1350 s since
    version 2.1, 2400 s before, from 2026-11-14 12:25): one definition, like the evidence conditions."""
    from mission.config import load_cfg
    return float(load_cfg(acceptance=acceptance_path(cfg))["time"]["sim_time_limit_s"])


def _budgets(cfg):
    D = cfg["ladder"]["dose"]
    f = float(D["unit_factor"])
    return {"time_limit_s": mission_time_limit(cfg), "dose_rating_gy": float(D["rating_gy"]),
            "dose_rating_site_uSv": float(D["rating_gy"]) / f, "dose_budget_uSv": float(D["rating_gy"]) / f,
            "unit_factor_gy_per_site_uSv": f,
            "frames": {"nav": 300, "inspect": 240, "depth": 1500, "thermal": 150, "lidar": 4000},
            "reports": {"instrument": 40, "mark": 6, "source": 3}}


# ============================================================================================ site.json
def _site_json(W, cfg, seed, attempt, items, decoys, plain, sources, boxes, bg, markers, grid, info, cov, lg, model,
               roles, assign, dose, budgets, zoning):
    from .generator import _site_json as base
    route0 = {"legs": [], "order": []}
    rep = {"ok": True}
    site = base(W, cfg, seed, attempt, items, decoys, plain, sources, boxes, bg, [], markers, route0, rep, [],
                budgets, grid, info, file_label(acceptance_path(cfg)), cov, lg, [i["type"] for i in items], model)
    site["schema"] = "site_gen/site/0.3-ladder"            # 0.3: body-semantics margin regions and poses
    site["layout"] = "ladder_v1"
    site["evidence_source"] = evidence_label()             # the grader's evidence model the regions were proved with
    v2 = bool(roles.get("v2"))
    for rec, d in zip(site["decoys"], decoys):
        if d.get("at_drawn_place_of"):
            rec["at_drawn_place_of"] = d["at_drawn_place_of"]
        if not v2:
            rec["label_code"] = d.get("label_code")
        else:
            rec["plate"] = {"text": d["tag"], "style": "text"}          # version 2: a text plate
            (rec.get("model_names") or {}).pop("marker_id", None)
    site["capability_bands"] = cfg["capability"]
    lay = W.lay
    byslot = {i["ladder_slot"]: i for i in items}
    for rec, it in zip(site["ips"], items):
        rec["slot"] = it["ladder_slot"]
        rec["ladder_type"] = it.get("ladder_type")
        if not v2:
            rec["label_code"] = it.get("label_code")
        elif it.get("ladder_slot") in SLOTS:
            rec["plate"] = {"text": it["tag"], "style": "text"}         # version 2: a text plate
            (rec.get("model_names") or {}).pop("marker_id", None)
        rec["last_2009"] = it.get("last_2009")
        rec["crossing"] = roles.get("%s_elements" % it["ladder_slot"], [lay.params["corridor_sill"]]
                                    if it["ladder_slot"] != "A" else [])
        if it["ladder_slot"] == "C-relocated":
            rec["drawn_room"] = it["documented"]["room"]
    site["ladder"] = {
        "assignment": assign, "slots": {k: v["tag"] for k, v in byslot.items()},
        "corridor_sill": lay.params["corridor_sill"], "corridor_sill_x": lay.params["corridor_sill_x"],
        "sealed_door": roles["sealed_door"], "as_built_opening": roles["as_built_opening"],
        "c_sealed_way": roles["C-sealed_way"][0], "c_sealed_room": roles["C-sealed"],
        "b_step_room": roles["B-step"], "b_step_elements": roles.get("B-step_elements"),
        "b_kerb_room": roles["B-kerb"], "b_kerb_elements": roles.get("B-kerb_elements"),
        "b_low_elements": roles.get("B-low_elements"), "annex": roles["annex"], "annex_host": roles["annex_host"],
        "annex_door": [d.id for d in lay.doors if not d.on_drawing and lay.params["annex"] in d.rooms][0],
        "optional_deviations": roles["optional"], "added_openings": roles.get("added_openings", []),
        "pruned_puddles": roles.get("pruned_puddles", []), "opened_ajar": roles.get("opened_ajar", []),
        "inner_door_hinge": list(roles.get("inner_door_hinge") or []),
        "partitions": getattr(W, "partitions", []), "dose": dose, "decoys": roles.get("decoy_count"),
        "config": {k: cfg["ladder"][k] for k in ("high_step_kinds", "high_step_band", "margin", "head",
                                                 "reading_place", "findable", "search", "dose", "high_step_draw",
                                                 "crouch_draw", "generic_high_step_max", "puddle_friction", "access", "truth",
                                                 "thermal") + (("v2", "v2_params") if roles.get("v2") else ())
                   if k in cfg["ladder"]},
        "acceptance_file": cfg["ladder"]["acceptance_file"],
    }
    if roles.get("v2"):
        _site_json_v2(W, site, roles)
    site["deviations"] = W._ladder_devs
    site["ladder"]["mod_log_truth"] = W._mod_log_truth
    site["discrepancies"] = [{"id": k + 1, "kind": d["kind"], "refs": [d["ref"]], "where": d["where"],
                              "text": d["text"], "prior_effect": "not on the drawing / the drawing is wrong"}
                             for k, d in enumerate(W._ladder_devs)]
    site["zoning_truth"] = zoning
    return site


def _site_json_v2(W, site, roles):
    """Version 2's records in site.json (grader / reference side only): the missing sheet, the factors of the
    candidate and the variants as built, the in-room obstacles with their bands and approaches."""
    lay = W.lay
    miss = W.missing
    site["missing_sheet"] = miss.to_json()
    for rec in site["rooms"]:
        rec["on_missing_sheet"] = bool(miss.room_missing(rec["id"]))
    on = set(roles.get("v2_on_sheet", []))
    for rec in site["ips"]:
        if rec.get("slot") in SLOTS:
            rec["on_missing_sheet"] = rec["slot"] in on
    L = site["ladder"]
    V2 = roles["v2"]
    kerb = W.elements[lay.params["bund"]["kerb"]]
    plan = roles["v2_plan"]
    L["v2"] = dict(V2)
    L["bund_class"] = "low" if kerb.cls == "needs_high_step" else "high"
    L["b_step_variant"] = V2["b_step"]
    L["b_low_variant"] = V2["b_low"]
    L["b_low_room"] = roles["B-low"]
    L["sheet_place"] = miss.place
    L["sheet_slots"] = sorted(on)
    L["sheet_plan"] = {"fixed": plan["fixed"], "forced_on": plan["forced_on"], "allowed": plan["allowed"],
                       "target": plan["target"], "final_plan": plan["planned"], "want": plan["want"],
                       "flipped": plan.get("flipped", [])}
    obs = []
    for p in roles.get("v2_obstacles", []):
        q = {k: v for k, v in p.items() if k not in ("band_world",)}
        q["band"] = list(p["band_world"])
        q["zones"] = VR.approach_rects(lay, p)
        obs.append(q)
    L["v2_obstacles"] = obs
    L["approach_zones"] = [[z["front"], z["behind"]] for z in (o["zones"] for o in obs)]


def _deviations(W, roles, items, sources):
    out = _deviation_list(W, roles, items, sources)
    lay = W.lay
    byslot = {i["ladder_slot"]: i for i in items}
    for d in out:
        k = d["kind"]
        if k in ("door_sealed", "opening_as_built", "opening_added"):
            d["where"] = list(lay.door(d["ref"]).xy())
        elif k == "instrument_moved":
            d["where"] = list(byslot["C-moved"]["slot"]["pos"][:2])
        elif k == "instrument_relocated":
            d["where"] = list(byslot["C-relocated"]["slot"]["pos"][:2])
        elif k == "annex_added":
            d["where"] = list(rect_center(lay.rooms[d["ref"]].rect))
        elif k == "zoning_outdated":
            d["where"] = list([s for s in sources if not s["documented"]][0]["pos"][:2])
        elif k == "repeater_dead":
            d["where"] = list([r for r in W.repeaters if not r["working"]][0]["pos"][:2])
        elif k == "partition_added":
            p = [q for q in getattr(W, "partitions", []) if "partition/%s" % q["leg"] == d["ref"]][0]
            d["where"] = [p["x"], 0.5 * (lay.rooms[p["leg"]].rect[1] + lay.rooms[p["leg"]].rect[3])]
        else:
            d["where"] = [0.0, 0.0]
    return out


def _deviation_list(W, roles, items, sources):
    lay = W.lay
    byslot = {i["ladder_slot"]: i for i in items}
    xs = [s for s in sources if not s["documented"]][0]
    out = [
        {"kind": "door_sealed", "room": roles["C-sealed"], "ref": roles["sealed_door"],
         "text": "door %s to room %s sealed" % (roles["sealed_door"], lay.rooms[roles["C-sealed"]].number)},
        {"kind": "instrument_moved", "room": byslot["C-moved"]["room"], "ref": byslot["C-moved"]["tag"],
         "text": "%s moved to another wall of room %s" % (byslot["C-moved"]["tag"],
                                                          lay.rooms[byslot["C-moved"]["room"]].number)},
        {"kind": "instrument_relocated", "room": byslot["C-relocated"]["documented"]["room"],
         "ref": byslot["C-relocated"]["tag"],
         "text": "%s relocated out of room %s" % (byslot["C-relocated"]["tag"],
                                                  lay.rooms[byslot["C-relocated"]["documented"]["room"]].number)},
        {"kind": "annex_added", "room": roles["annex_host"], "ref": roles["annex"],
         "text": "annex added outside room %s" % lay.rooms[roles["annex_host"]].number},
        {"kind": "zoning_outdated", "room": xs["room"], "ref": xs["id"],
         "text": "radiological conditions changed in %s, not re-surveyed" % lay.rooms[xs["room"]].number},
        {"kind": "repeater_dead", "room": [r for r in W.repeaters if not r["working"]][0]["room"],
         "ref": [r for r in W.repeaters if not r["working"]][0]["id"], "text": "link repeater out of service"},
        {"kind": "opening_as_built", "room": roles["C-sealed"], "ref": roles["as_built_opening"],
         "text": "wall opening made into room %s" % lay.rooms[roles["C-sealed"]].number},
    ]
    for did in roles.get("added_openings", []):
        d = lay.door(did)
        out.append({"kind": "opening_added", "room": d.rooms[1] if d.rooms[1] in lay.rooms else d.rooms[0], "ref": did,
                    "text": "wall opening made between %s and %s" % tuple(lay.rooms[x].number for x in d.rooms)})
    for p in getattr(W, "partitions", []):
        out.append({"kind": "partition_added", "room": p["leg"], "ref": "partition/%s" % p["leg"],
                    "text": "temporary shielding erected in corridor %s" % lay.rooms[p["leg"]].number})
    return out
