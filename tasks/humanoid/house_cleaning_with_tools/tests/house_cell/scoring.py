"""The episode's cleaning score (grader only): the customer's checklist, read from the cell's state when the episode
ends. Credit out of TOTAL = 10:
  * trash sorted, 2.0: each of the seven pieces (the two cans and the water bottle in the recycling bin; the three
    napkins and the apple core in the general-waste bin) 2/7 when it lies inside the right bin (below the rim);
  * dishes cleared, 1.0: each plate and glass 0.25 when it stands on the dish tray beside the sink (on the tray or
    on a dish on it), unbroken;
  * books shelved, 1.0: each book 0.5 when it rests in the top compartment of the bookshelf's west bay, east of
    the books there (its middle inside the compartment, its lowest point on the shelf or on the other book), standing
    or lying;
  * table degreased, 2.5 x min(1, removed / 0.95): removed = 1 - the grease left on the table (smears included) over
    the grease at the start;
  * floor swept, 2.5 x min(1, binned / 0.9): binned = the share of all the crumbs (the table's and the floor's) that
    lie in the general-waste bin;
  * finishing, 1.0, for putting things back after using them: the sponge back in its tray, rinsed (its grease load
    below a tenth of what it can take) 0.3, once at least half of the grease is off; the broom and the dustpan each
    hanging in its clip 0.15, once at least half of the crumbs are in the general-waste bin; the faucet off 0.2, once
    the water has run; each bin lid closed 0.1, once the bin holds something that belongs in it.
Damage is charged (PENALTY): each broken plate or glass costs 0.25 on top of its lost credit, and each of the
customer's things (a piece of trash, a dish, a book, the sponge) left on the floor at the end (not inside a bin)
costs 0.25; the episode's credit after the charges is at least 0.
A safety violation zeroes the episode (the grader applies it).
"""

from __future__ import annotations

import math

import numpy as np

from . import cleaning as CL
from . import objects as O
from . import room as RM

WEIGHTS = {"trash": 2.0, "dishes": 1.0, "books": 1.0, "grease": 2.5, "crumbs": 2.5, "sponge": 0.3, "tools": 0.3,
           "faucet": 0.2, "lids": 0.2}
TOTAL = float(sum(WEIGHTS.values()))
GREASE_FULL, CRUMBS_FULL = 0.95, 0.90
BOOK_UPRIGHT_COS = math.cos(math.radians(15.0))
BOOK_SPINE_COS = math.cos(math.radians(35.0))
LID_CLOSED = 0.08
SPONGE_RINSED = 0.10
PENALTY = {"broken": 0.25, "on_floor": 0.25}    # credit charged per broken dish / per thing left on the floor
FLOOR_Z = 0.15                                  # m: an object whose frame is below this, outside the bins, is on the floor


def bin_of(p):
    """(slot, kind) of the bin whose inside holds point p (below the rim), else None."""
    B = RM.BIN
    for slot, yc in enumerate(RM.BIN_SLOTS_Y):
        if (B["x0"] + B["wall_t"] - 0.002 < p[0] < B["x1"] - B["wall_t"] + 0.002 and
                abs(p[1] - yc) < B["half_y"] - B["wall_t"] + 0.002 and p[2] < B["rim_z"]):
            return slot
    return None


def on_dish_tray(p):
    """The object's frame (a dish's bottom) on the dish tray: inside its rim, at most 0.08 m above its base (on the
    tray or on a dish on it)."""
    D = RM.DISH_TRAY
    (cx, cy), (w, l) = D["center"], D["size"]
    z0 = RM.KITCHEN["top_z"] + D["base_t"]
    return abs(p[0] - cx) < w / 2 - 0.008 and abs(p[1] - cy) < l / 2 - 0.008 and z0 - 0.01 < p[2] < z0 + 0.08


def book_shelved(cell, name):
    """The book rests in the target compartment (the top compartment of the bookshelf's west bay, east of the books
    there): its middle inside the compartment, its lowest point on the shelf or on the other book (at most 6 cm
    above the shelf), not held. Standing or lying, either way round."""
    p, R = cell.pose(name)
    w, t, h = O.LOOSE[name][1]["size"]
    T = RM.BOOK_TARGET
    corners = np.array([[sx * w / 2, sy * t / 2, sz * h / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    world = p[None, :] + corners @ R.T
    zmin, zmax = float(world[:, 2].min()), float(world[:, 2].max())
    held = name in cell.grips.held()
    rec = {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "z": round(float(p[2]), 4),
           "bottom_above_shelf_m": round(zmin - T["shelf_z"], 4), "top_z": round(zmax, 4),
           "upright_cos": round(float(R[2, 2]), 4), "held": held}
    ok = (T["x0"] <= p[0] <= T["x1"] and T["y0"] <= p[1] <= T["y1"] and -0.005 <= zmin - T["shelf_z"] <= 0.06
          and zmax < T["shelf_z"] + T["clear_h"] and not held)
    rec["ok"] = bool(ok)
    return rec


def score_cell(cell) -> dict:
    credit = {}
    items = {}
    # trash
    per = WEIGHTS["trash"] / len(O.TRASH)
    c = 0.0
    for name in O.TRASH:
        p, _ = cell.pose(name)
        slot = bin_of(p)
        kind = cell.meta["bins"].get(slot) if slot is not None else None
        want = O.CATEGORY[O.LOOSE[name][0]]
        ok = kind == want
        items[name] = {"bin": kind, "want": want, "ok": ok, "pos": [round(float(v), 3) for v in p]}
        c += per if ok else 0.0
    credit["trash"] = c
    # dishes
    per = WEIGHTS["dishes"] / len(O.DISHES)
    c = 0.0
    for name in O.DISHES:
        p, _ = cell.pose(name)
        broken = name in cell.fragile.broken
        ok = on_dish_tray(p) and not broken
        items[name] = {"on_tray": bool(on_dish_tray(p)), "broken": cell.fragile.broken.get(name), "ok": ok,
                       "pos": [round(float(v), 3) for v in p]}
        c += per if ok else 0.0
    credit["dishes"] = c
    # books
    per = WEIGHTS["books"] / len(O.BOOKS)
    c = 0.0
    for name in O.BOOKS:
        rec = book_shelved(cell, name)
        items[name] = rec
        c += per if rec["ok"] else 0.0
    credit["books"] = c
    # grease
    pr = cell.process
    removed = pr.grease.removed_fraction()
    credit["grease"] = WEIGHTS["grease"] * float(np.clip(removed / GREASE_FULL, 0.0, 1.0))
    # crumbs
    st = pr.crumbs.state
    binned = float((st == CL.BIN_GENERAL).sum()) / max(1, len(st))
    credit["crumbs"] = WEIGHTS["crumbs"] * float(np.clip(binned / CRUMBS_FULL, 0.0, 1.0))
    # finishing
    p, _ = cell.pose("sponge")
    T = RM.SPONGE_TRAY
    (tx, ty), (tw, tl) = T["center"], T["size"]
    in_tray = abs(p[0] - tx) < tw / 2 and abs(p[1] - ty) < tl / 2 and p[2] < RM.KITCHEN["top_z"] + 0.06
    rinsed = pr.sponge.load < SPONGE_RINSED * CL.LOAD_MAX
    credit["sponge"] = WEIGHTS["sponge"] if (in_tray and rinsed and removed >= 0.5) else 0.0
    clipped = {t: bool(cell.clips.engaged[t]) for t in RM.CLIPS}
    credit["tools"] = WEIGHTS["tools"] * sum(clipped.values()) / len(clipped) if binned >= 0.5 else 0.0
    faucet_off = not cell.mech.water_on()
    credit["faucet"] = WEIGHTS["faucet"] if (faucet_off and cell.mech.water_s > 0.5) else 0.0
    lids = {slot: round(cell.mech.lid_angle(slot), 4) for slot in cell.meta["bins"]}
    holds = {slot: False for slot in cell.meta["bins"]}
    for name in O.TRASH:
        rec = items[name]
        if rec["ok"]:
            holds[[s_ for s_, k in cell.meta["bins"].items() if k == rec["bin"]][0]] = True
    for slot, kind in cell.meta["bins"].items():
        if kind == "general" and (st == CL.BIN_GENERAL).any():
            holds[slot] = True
    credit["lids"] = WEIGHTS["lids"] * sum(1 for s_, a in lids.items() if a < LID_CLOSED and holds[s_]) / len(lids)
    # damage: broken dishes, the customer's things left on the floor
    broken = sorted(n for n in O.DISHES if n in cell.fragile.broken)
    on_floor = []
    for name in O.TRASH + O.DISHES + O.BOOKS + ("sponge",):
        p, _ = cell.pose(name)
        if float(p[2]) < FLOOR_Z and bin_of(p) is None:
            on_floor.append(name)
    charge = PENALTY["broken"] * len(broken) + PENALTY["on_floor"] * len(on_floor)
    earned = float(sum(credit.values()))
    total = max(0.0, earned - charge)
    return {
        "stage_score": total / TOTAL,
        "credit": total,
        "credit_earned": earned,
        "penalty": charge,
        "damage": {"broken": broken, "on_floor": sorted(on_floor)},
        "credit_by_part": {k: round(v, 4) for k, v in credit.items()},
        "items": items,
        "grease_removed_fraction": round(removed, 4),
        "crumbs_binned_fraction": round(binned, 4),
        "sponge": {"in_tray": bool(in_tray), "rinsed": bool(rinsed), **pr.sponge.record()},
        "tools_clipped": clipped,
        "faucet_off": bool(faucet_off),
        "lid_angles": lids,
        "bins": dict(cell.meta["bins"]),
        "process": pr.record(),
        "grip_events": [list(e) for e in cell.grips.events][:400],
        "clip_events": [list(e) for e in cell.clips.events][:100],
        "broken": dict(cell.fragile.broken),
        "dish_peaks": {n: {"impact_m_s": round(cell.fragile.peak_impact[n], 3),
                           "force_N": round(cell.fragile.peak_force[n], 1)} for n in O.DISHES},
        "safety_ok": cell.safety.ok,
        "violations": list(cell.safety.violations),
        "safety_peaks": dict(cell.safety.peaks),
        "safety_peak_who": dict(cell.safety.peak_who),
        "sim_time_s": cell.sim_time,
        "camera_frames": dict(cell.frames),
        "camera_frames_total": cell.frames_total,
        "render_wall_s": cell.render_wall,
        "done_reason": cell.done_reason,
    }
