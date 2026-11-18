"""The coupled engineering-change repair (B-13) and the economy criterion C5 (public code; every
number comes from the inputs).

While an item fails, the first line in `order` that owns a failing item is replaced by the first catalogue part (by
price, then quote order) that makes the line's own items pass with every other line unchanged; everything is evaluated
again after each change (a heavier module or pack changes every load). A visited-state set and `max_steps` bound the
loop. The family is never changed. The BOM is repairable only if the final evaluation passes every item except the
budget; then J_eff = its cost + eco_usd x the replaced lines, else J_eff = inf.

    r = repair(proc, bom, cat, lots, eco_usd, max_steps, order)
    c5 = economy(J_star, r["J_eff"], ceiling_usd, band_usd, taper_frac)
"""
import copy
import math

from . import layer as L
from . import schemas as S
from .procedure import EvalContext

OWNED = {t: tuple(leg + "_" + t for leg in S.LEGS) for t in S.JOINT_TYPES}
OWNED.update({"battery": ("battery",), "drivers": ("drivers",), "hub": ("hub",)})


def _state(b):
    return (b["modules"]["hip_roll"], b["modules"]["hip_pitch"], b["modules"]["knee"], b["battery"], b["driver"],
            b["driver_count"], b["hub"])


def _current(b, line):
    if line in S.JOINT_TYPES:
        return b["modules"][line]
    return b[{"battery": "battery", "drivers": "driver", "hub": "hub"}[line]]


def _candidates(cat, b, line):
    by = lambda p: (cat.price(p), cat.index(p))  # noqa: E731
    if line in S.JOINT_TYPES:
        pids = sorted((m["part_id"] for m in cat.modules), key=by)
    elif line == "battery":
        cells = L.family_cells(b["battery_family"])
        pids = sorted((p["part_id"] for p in cat.packs if int(p["cells"]) == cells), key=by)
    elif line == "drivers":
        pids = sorted((d["part_id"] for d in cat.drivers), key=lambda p: (cat.driver_count(p) * cat.price(p),
                                                                        cat.index(p)))
    else:
        pids = sorted((h["part_id"] for h in cat.hubs), key=by)
    return [p for p in pids if p != _current(b, line)]


def _replace(cat, b, line, pid):
    trial = copy.deepcopy(b)
    if line in S.JOINT_TYPES:
        trial["modules"][line] = pid
    elif line == "battery":
        trial["battery"] = pid
    elif line == "drivers":
        trial["driver"] = pid
        trial["driver_count"] = cat.driver_count(pid)
    else:
        trial["hub"] = pid
    return trial


def repair(proc, bom, cat, lots, eco_usd, max_steps, order):
    """-> {"bom", "replaced", "J", "J_eff" (inf if irreparable), "repairable", "log", "final_items"}."""
    ctx = EvalContext(proc, cat, lots)
    b = copy.deepcopy(bom)
    replaced, seen, log, reason = 0, set(), [], None
    for step in range(int(max_steps) + 1):
        ev = proc._evaluate(ctx, b)
        state = _state(b)
        if state in seen:
            reason = "cycle"
            break
        seen.add(state)
        line = next((ln for ln in order if not all(ev["items"][i] for i in OWNED[ln])), None)
        if line is None:
            break
        if step == int(max_steps):
            reason = "max steps"
            break
        fixed = False
        for pid in _candidates(cat, b, line):
            trial = _replace(cat, b, line, pid)
            if all(proc._evaluate(ctx, trial)["items"][i] for i in OWNED[line]):
                log.append("%s: %s -> %s" % (line, _current(b, line), pid))
                b = trial
                replaced += 1
                fixed = True
                break
        if not fixed:
            reason = "no %s passes" % line
            break
    final = proc._evaluate(ctx, b)["items"]
    repairable = reason is None and all(v for k, v in final.items() if k != "budget")
    if reason is not None:
        log.append("irreparable: " + reason)
    elif not repairable:
        log.append("irreparable: items still failing (%s)" % ", ".join(k for k, v in final.items()
                                                                       if not v and k != "budget"))
    J_eff = cat.bom_cost(b) + float(eco_usd) * replaced if repairable else math.inf
    return {"bom": b, "replaced": replaced, "J": cat.bom_cost(bom), "J_eff": J_eff, "repairable": repairable,
            "log": log, "final_items": final}


def economy(J_star, J_eff, ceiling_usd, band_usd, taper_frac):
    """clamp(1 - (J_eff - J*) / band, 0, 1) x taper; taper = 1 at or below the ceiling, else
    max(0, 1 - (J_eff - ceiling) / (taper_frac x ceiling)); 0 for a non-finite J_eff."""
    J_eff = float(J_eff)
    if not math.isfinite(J_eff):
        return float(0)
    over = round(J_eff - float(J_star), 6)
    ratio = min(1, max(0, 1 - over / float(band_usd)))
    ceiling = float(ceiling_usd)
    if J_eff <= ceiling + 1e-9:
        taper = 1
    else:
        taper = max(0, 1 - (J_eff - ceiling) / (float(taper_frac) * ceiling))
    return float(ratio * taper)
