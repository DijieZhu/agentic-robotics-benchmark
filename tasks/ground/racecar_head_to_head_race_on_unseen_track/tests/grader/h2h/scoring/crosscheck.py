"""A second computation of the score, written apart from the scoring module, from the lists of the venue results.

    rec = score_run(results, references, config, gates, anchor_value)    # the numbers of a score record
    d = max_difference(rec_a, rec_b)                                      # the largest difference of their numbers

It never reads a heat block's own `C`, `beaten`, `K` or `classified`. For every heat session of the configuration:

- a field car's balance = its overtakes marked clean minus the times it re-overtook the car (overtake list);
- it is beaten when its balance is at least 1 and its progress at the end is below the car's;
- C = the sum of the three balances; K = the number of stops that gave a warning (stop list);
- classified when the session's lap list holds at least N - heat.classified_laps_short laps and K is below
  staff.warnings_end_session, with N the venue reference's lap count;
- a heat session that is not in the result scores 0.

The arithmetic is exact (fractions of integers and of the binary values of the weights) and becomes floating point
once, at the end. Gates as the scoring module reads them: the given ones, `step_called` also needing a venue that
answered a step and `clean_run` a result for every reference and no venue ended by the cell's own failure.
"""

from fractions import Fraction

GATES = ("submission_ok", "step_called", "clean_run", "no_network", "no_late_write")
OWN_FAILURES = ("internal_error", "physics_fault")
HEAT_NAMES = ("heat1", "heat2")

__all__ = ["score_run", "venue_numbers", "heat_numbers", "max_difference"]


def _clip(q):
    return Fraction(0) if q < 0 else (Fraction(1) if q > 1 else q)


def heat_numbers(session, n_laps, C_ref, config):
    """Exact (classified, place, pass, conduct, heat) of one heat session from its lists; None: never opened."""
    if session is None:
        return (0, Fraction(0), Fraction(0), Fraction(0), Fraction(0))
    blk = session["heat"]
    car_progress = blk["car"]["progress"]
    total, beaten = 0, 0
    for fc in blk["field"]:
        ups = sum(1 for e in blk["overtakes"] if e["car"] == fc["id"] and e["kind"] == "overtake" and e["clean"] is True)
        downs = sum(1 for e in blk["overtakes"] if e["car"] == fc["id"] and e["kind"] == "loss")
        total += ups - downs
        if ups - downs >= 1 and fc["progress"] < car_progress:
            beaten += 1
    warned = sum(1 for st in blk["stops"] if st["warned"] is True)
    short = config["heat"]["classified_laps_short"]
    end_at = config["staff"]["warnings_end_session"]
    classified = 1 if (len(session["laps"]) >= n_laps - short and warned < end_at) else 0
    w = config["scoring"]["heat"]
    place = Fraction(beaten, config["heat"]["cars"] - 1)
    pas = _clip(Fraction(total, C_ref - w["pass_margin"]))
    conduct = max(Fraction(0), 1 - Fraction(warned, w["warnings_zero_at"]))
    heat = classified * conduct * (Fraction(w["place_weight"]) * place + Fraction(w["pass_weight"]) * pas)
    return (classified, place, pas, conduct, heat)


def venue_numbers(result, reference, config):
    """{heat name: exact terms} and the exact venue value of one venue (result None: nothing was run)."""
    names = [s["name"] for s in config["sessions"] if s["name"] in HEAT_NAMES]
    by = {s["name"]: s for s in (result or {}).get("sessions") or []}
    heats = {}
    for i, name in enumerate(names):
        heats[name] = heat_numbers(by.get(name), reference["heat_laps"], reference["C_ref"][i], config)
    return heats, sum(h[4] for h in heats.values()) / len(names)


def score_run(results, references, config, gates=None, anchor_value=1.0):
    """The numbers of the score record as plain floats: {"venues": [{"world_id", <heat>: {...}, "venue"}],
    "gates", "gate", "head_to_head_score", "reward"}."""
    by_id = {r["world_id"]: r for r in results or [] if r is not None}
    given = dict(gates) if gates is not None else {k: True for k in GATES}
    rows, total = [], Fraction(0)
    for ref in references:
        res = by_id.get(ref["world_id"])
        heats, venue = venue_numbers(res, ref, config)
        row = {"world_id": ref["world_id"], "venue": float(venue)}
        for name, (c, pl, pa, co, he) in heats.items():
            row[name] = {"classified": c, "place": float(pl), "pass": float(pa), "conduct": float(co), "heat": float(he)}
        rows.append(row)
        total += venue
    got = [by_id.get(ref["world_id"]) for ref in references]
    g = {k: bool(given[k]) for k in GATES}
    g["step_called"] = g["step_called"] and any(r is not None and r.get("step_called") is True for r in got)
    g["clean_run"] = g["clean_run"] and all(r is not None for r in got) and \
        all(r is None or r.get("end_reason") not in OWN_FAILURES for r in got)
    gate = 1 if all(g.values()) else 0
    score = gate * total / len(references)
    reward = _clip(score / Fraction(anchor_value) / Fraction(config["scoring"]["full_at"]))
    return {"venues": rows, "gates": g, "gate": gate, "head_to_head_score": float(score), "reward": float(reward)}


def _numbers(obj, path, out):
    if isinstance(obj, bool):
        out[path] = 1.0 if obj else 0.0
    elif isinstance(obj, (int, float)):
        out[path] = float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ("world_id", "schema", "opened", "anchor"):
                _numbers(v, path + "/" + k, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _numbers(v, "%s/%d" % (path, i), out)
    return out


def max_difference(a, b):
    """Largest absolute difference between the numbers two score records share; infinity when one record has a
    number the other lacks."""
    na, nb = _numbers(a, "", {}), _numbers(b, "", {})
    if set(na) != set(nb):
        return float("inf")
    return max([abs(na[k] - nb[k]) for k in na] or [0.0])
