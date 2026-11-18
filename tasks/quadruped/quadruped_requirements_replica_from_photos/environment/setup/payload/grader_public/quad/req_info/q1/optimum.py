"""The cheapest accepted BOM J* of B-14 over a given catalogue, and the utilisation tables (public code;
every number comes from the inputs).

For each battery family: the packs of the family that fit the model's pack bay; per joint type the modules of the type's
housing class whose voltage range contains the pack's window. An exact interval screen drops a module whose lower
utilisation bound (every other BOM mass anywhere in the family's admissible mass box) exceeds the threshold. Then
pack x knee x hip_pitch x hip_roll in price order (ties: quote order) with a cost bound; each leaf evaluates every item
exactly, takes the cheapest driver type that passes and the cheapest hub with enough ports. Only a strict improvement
replaces the best, so the first BOM found wins a tie.

    best = optimal_bom(proc, cat, lots, node_cap)                 J* (every utilisation <= 1)
    optimal_bom(proc, cat, lots, node_cap, threshold=t)            every utilisation <= t instead
"""
import math

from . import layer as L
from . import schemas as S
from .procedure import EvalContext
from .quadmodel import ProcedureError


def _by_price(cat, pids):
    return sorted(pids, key=lambda p: (cat.price(p), cat.index(p)))


def admissible_modules(proc, cat, family, joint_type, pack):
    """Modules of the joint type's housing class whose voltage range contains the pack's window, by (price, quote)."""
    window = cat.pack_window(pack)
    cls = proc.model["housing_class"][joint_type]
    return _by_price(cat, [m["part_id"] for m in cat.modules
                           if m["housing_class"] == cls and cat.window_inside(window, m["voltage_range_v"])])


def fitting_packs(proc, cat, family):
    """Packs of the family's cell count that fit the model's pack bay, by (price, quote)."""
    cells = L.family_cells(family)
    inner = proc.model["pack_bay"]["inner_m"]
    return _by_price(cat, [p["part_id"] for p in cat.packs
                           if int(p["cells"]) == cells and cat.fits(p["part_id"], inner)])


def admissible_drivers(cat, pack):
    """Drivers whose input range contains the pack's window, by (driver_count x price, quote)."""
    window = cat.pack_window(pack)
    pids = [d["part_id"] for d in cat.drivers if cat.window_inside(window, d["input_range_v"])]
    return sorted(pids, key=lambda p: (cat.driver_count(p) * cat.price(p), cat.index(p)))


def _family_sets(proc, cat, family):
    packs = fitting_packs(proc, cat, family)
    adm = {(t, p): admissible_modules(proc, cat, family, t, p) for t in S.JOINT_TYPES for p in packs}
    union = {t: _by_price(cat, set().union(*(set(adm[(t, p)]) for p in packs))) for t in S.JOINT_TYPES}
    return packs, adm, union


def _box(cat, packs, union):
    box = {}
    for t in S.JOINT_TYPES:
        ms = [float(cat.get(p)["mass_kg"]) for p in union[t]]
        box[t] = (min(ms), max(ms))
    pm = [float(cat.get(p)["mass_kg"]) for p in packs]
    box["pack"] = (min(pm), max(pm))
    return box


def mass_box(proc, cat, family):
    """{"hip_roll", "hip_pitch", "knee", "pack": (lo, hi)} kg: the family's admissible BOM mass box (ProcedureError if a
    slot has no admissible part)."""
    packs, _, union = _family_sets(proc, cat, family)
    if not packs:
        raise ProcedureError("no pack of family %s fits the pack bay" % family)
    for t in S.JOINT_TYPES:
        if not union[t]:
            raise ProcedureError("no admissible module for %s on family %s" % (t, family))
    return _box(cat, packs, union)


def _leaf(proc, ctx, family, modules, pack, drivers, hubs, threshold):
    """(ok, bom, cost, utilisations) of one leaf: every item exactly, the cheapest passing driver, the cheapest hub."""
    cat = ctx.cat
    bom = {"battery_family": family, "modules": dict(modules), "battery": pack}
    masses = proc.masses_of(bom, cat)
    tau_mv = proc._tau_move(masses, ctx.payload)
    tau_pf = proc._tau_proof(masses, ctx.payload)
    utils = []
    ok = True
    for t in S.JOINT_TYPES:
        worst = proc._type_util(ctx, t, modules[t], family, tau_mv, tau_pf)[3]
        utils += [float(u) for u in worst]
        ok = ok and float(worst.max()) <= threshold
    i_ph = proc._phase_currents(ctx, modules, family, tau_mv)
    i_bus = proc._bus_current(ctx, modules, family, tau_mv, i_ph)
    p = cat.get(pack)
    usable = float(proc.layer["brief"]["usable_fraction"]) * float(p["capacity_ah"]) * ctx.pack_factor(pack)
    u_chg = _sdiv(proc._charge_ah(ctx, i_bus), usable)
    u_cur = _sdiv(float(i_bus.max()), p["continuous_current_a"])
    utils += [u_chg, u_cur]
    ok = ok and u_chg <= threshold and u_cur <= threshold
    peak = float(i_ph.max())
    rms = float(math.sqrt(max(float(v) for v in (i_ph[proc._crawl] ** 2).mean(axis=0))))
    chosen = None
    for d in drivers:
        e = cat.get(d)
        u_pk = _sdiv(peak, e["peak_current_a"])
        u_rms = _sdiv(rms, e["continuous_current_a"])
        utils += [u_pk, u_rms]
        if u_pk <= threshold and u_rms <= threshold:
            chosen = d
            break
    if chosen is None:
        return False, None, None, utils
    count = cat.driver_count(chosen)
    hub = next((h for h in hubs if int(cat.get(h)["ports"]) >= count), None)
    if hub is None:
        return False, None, None, utils
    bom.update({"driver": chosen, "driver_count": count, "hub": hub})
    cost = cat.bom_cost(bom)
    ok = ok and cost <= ctx.ceiling + 1e-9
    return ok, bom, cost, utils


def _sdiv(num, den):
    """num / den for scalars; den <= 0 gives +inf for num > 0, else 0."""
    num, den = float(num), float(den)
    if den > 0:
        return num / den
    return math.inf if num > 0 else 0


def optimal_bom(proc, cat, lots, node_cap, *, threshold=1, families=None, payload_kg=None, mission_time_s=None,
                ceiling_usd=None):
    """The cheapest BOM whose every utilisation is <= threshold, or None.

    -> {"bom", "cost", "family", "n_leaves", "n_kept": {joint_type: int}, "evaluation", "knife_edge_min",
        "n_kept_by_family": {family: {joint_type: int}}}
    A leaf count above node_cap raises ProcedureError("node cap ...")."""
    ctx = EvalContext(proc, cat, lots, payload_kg, mission_time_s, ceiling_usd)
    thr = float(threshold)
    fams = list(L.policy(proc.layer)["battery_families"] if families is None else families)
    hubs = _by_price(cat, [h["part_id"] for h in cat.hubs])
    best, n_leaves, knife, kept = None, 0, math.inf, {}
    for fam in fams:
        packs, adm, union = _family_sets(proc, cat, fam)
        if not packs or any(not union[t] for t in S.JOINT_TYPES) or not hubs:
            continue
        box = _box(cat, packs, union)
        keep = {}
        for t in S.JOINT_TYPES:
            keep[t] = []
            parts = proc._lb_parts(t, box, ctx.payload)
            for pid in union[t]:
                arrays = proc._util_lb_arrays(t, box, ctx.payload, float(cat.get(pid)["mass_kg"]), parts)
                if proc._util_lb(ctx, t, pid, fam, arrays) <= thr:
                    keep[t].append(pid)
        kept[fam] = {t: len(keep[t]) for t in S.JOINT_TYPES}
        drivers_any = set()
        for p in packs:
            drivers_any |= set(admissible_drivers(cat, p))
        if not drivers_any:
            continue
        drv_min = min(cat.driver_count(d) * cat.price(d) for d in drivers_any)
        hub_min = cat.price(hubs[0])
        n_legs = len(S.LEGS)
        for pack in packs:
            kp = {t: [pid for pid in keep[t] if pid in adm[(t, pack)]] for t in S.JOINT_TYPES}
            drivers = admissible_drivers(cat, pack)
            p_pack = cat.price(pack)
            for knee in kp["knee"]:
                for pitch in kp["hip_pitch"]:
                    for roll in kp["hip_roll"]:
                        c0 = n_legs * (cat.price(knee) + cat.price(pitch) + cat.price(roll)) + p_pack + drv_min + hub_min
                        if best is not None and c0 >= best["cost"] - 1e-9:
                            break
                        n_leaves += 1
                        if n_leaves > int(node_cap):
                            raise ProcedureError("node cap %d exceeded by the optimum search" % int(node_cap))
                        mods = {"hip_roll": roll, "hip_pitch": pitch, "knee": knee}
                        ok, bom, cost, utils = _leaf(proc, ctx, fam, mods, pack, drivers, hubs, thr)
                        knife = min([knife] + [abs(u - thr) for u in utils])
                        if ok and (best is None or cost < best["cost"] - 1e-9):
                            best = {"bom": bom, "cost": cost, "family": fam}
    if best is None:
        return None
    best.update({"n_leaves": n_leaves, "n_kept": kept[best["family"]], "evaluation": proc._evaluate(ctx, best["bom"]),
                 "knife_edge_min": knife, "n_kept_by_family": kept})
    return best


def utilisation_table(proc, cat, lots, bom, *, payload_kg=None, mission_time_s=None):
    """Utilisations of every admissible alternative with the rest of `bom` installed:
    {"hip_roll" | "hip_pitch" | "knee": {pid: worst utilisation of the type}, "pack": {pid: {"charge", "current"}},
     "drivers": {pid: {"peak", "rms"}}}."""
    ctx = EvalContext(proc, cat, lots, payload_kg, mission_time_s)
    fam = bom["battery_family"]
    out = {}
    for t in S.JOINT_TYPES:
        row = {}
        for pid in admissible_modules(proc, cat, fam, t, bom["battery"]):
            mods = dict(bom["modules"])
            mods[t] = pid
            masses = proc.masses_of(dict(bom, modules=mods), cat)
            worst = proc._type_util(ctx, t, pid, fam, proc._tau_move(masses, ctx.payload),
                                    proc._tau_proof(masses, ctx.payload))[3]
            row[pid] = float(worst.max())
        out[t] = row
    out["pack"] = {}
    for pid in fitting_packs(proc, cat, fam):
        u = proc._evaluate(ctx, dict(bom, battery=pid))["util"]["battery"]
        out["pack"][pid] = {"charge": u["charge"], "current": u["current"]}
    ev = proc._evaluate(ctx, bom)
    peak = max(ev["currents"]["phase_peak_a"].values())
    rms = max(ev["currents"]["phase_rms_a"].values())
    out["drivers"] = {}
    for pid in admissible_drivers(cat, bom["battery"]):
        e = cat.get(pid)
        out["drivers"][pid] = {"peak": _sdiv(peak, e["peak_current_a"]), "rms": _sdiv(rms, e["continuous_current_a"])}
    return out
