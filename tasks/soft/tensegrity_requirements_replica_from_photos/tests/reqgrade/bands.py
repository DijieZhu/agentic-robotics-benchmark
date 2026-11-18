"""Band functions and unit conversion of the C1-C3 grader. No value of any truth is written here.

A band spec, as the ledger (`band_draft`), the contract (`tolerance_classes`) and the freeze write it, is one of
    "exact" or "exact (enum)"                         equal (a number within EXACT_REL of the truth), else nothing
    {"full_at_mm": a, "zero_at_mm": b}                absolute error in the row's unit (millimetres)
    {"abs_full_at": a, "abs_zero_at": b}              absolute error in the row's unit
    {"rel_full_at": a, "rel_zero_at": b}              error relative to the truth
    {"kind": "exact" | "abs" | "rel", "full": a, "zero": b}   the normalised form (parse_spec's output)
Linear between the full point a and the zero point b: credit 1 at or below a, nothing at or beyond b.
"""
import math

EXACT_REL = 1e-9


def parse_spec(spec):
    """The normalised form of a band spec (ValueError for anything else: a defect of the truth files)."""
    if isinstance(spec, str) and spec.split(" ")[0] == "exact":
        return {"kind": "exact"}
    if isinstance(spec, dict):
        if spec.get("kind") == "exact":
            return {"kind": "exact"}
        if spec.get("kind") in ("abs", "rel"):
            return {"kind": spec["kind"], "full": float(spec["full"]), "zero": float(spec["zero"])}
        for kind, full, zero in (("abs", "full_at_mm", "zero_at_mm"), ("abs", "abs_full_at", "abs_zero_at"),
                                 ("rel", "rel_full_at", "rel_zero_at")):
            if full in spec and zero in spec:
                out = {"kind": kind, "full": float(spec[full]), "zero": float(spec[zero])}
                if not (math.isfinite(out["full"]) and math.isfinite(out["zero"]) and 0 <= out["full"] < out["zero"]):
                    raise ValueError("band spec %r: need 0 <= full < zero" % (spec,))
                return out
    raise ValueError("not a band spec: %r" % (spec,))


def band(err, full, zero):
    """Credit of an error: 1 at or below `full`, 0 at or beyond `zero`, linear in between; 0 for an error that is
    not a finite number (a score is never NaN)."""
    try:
        err, full, zero = float(err), float(full), float(zero)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(err):
        return 0.0
    if err <= full:
        return 1.0
    if not err < zero:
        return 0.0
    return (zero - err) / (zero - full)


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and _finite(v)


def _finite(v):
    try:
        return math.isfinite(float(v))
    except (OverflowError, TypeError, ValueError):
        return False


def error(agent, truth, spec):
    """The error that `spec` scores (absolute or relative), or None when `agent` is not comparable with `truth`."""
    if isinstance(truth, str) or not is_number(agent):
        return None
    diff = abs(float(agent) - float(truth))
    if spec["kind"] == "rel" and truth != 0:
        return diff / abs(float(truth))
    return diff


def score(agent, truth, spec):
    """Credit of `agent` against `truth` (both in the truth's unit) under the normalised band `spec`."""
    if spec["kind"] == "exact":
        if isinstance(truth, str):
            return 1.0 if isinstance(agent, str) and agent == truth else 0.0
        if not is_number(agent):
            return 0.0
        return 1.0 if abs(float(agent) - float(truth)) <= EXACT_REL * max(1.0, abs(float(truth))) else 0.0
    err = error(agent, truth, spec)
    return 0.0 if err is None else band(err, spec["full"], spec["zero"])


def within_zero_point(agent, truth, spec):
    """True when `agent` is right in the wide sense used for a W0 quantity: equal for an exact band, an error short of
    the zero point otherwise."""
    if spec["kind"] == "exact":
        return score(agent, truth, spec) > 0
    err = error(agent, truth, spec)
    return err is not None and err < spec["zero"]


def unit_table(units_json):
    """{unit: (dimension, factor)} of schemas/units.json."""
    out = {}
    for dim, block in sorted(units_json["dimensions"].items()):
        for unit, factor in sorted(block["units"].items()):
            out[unit] = (dim, float(factor))
    return out


def convert(value, unit_from, unit_to, units):
    """`value` in `unit_from` expressed in `unit_to` (the same unit: the value itself, unchanged); None when either unit
    is unknown, the dimensions differ or the value is not a finite number."""
    if not is_number(value):
        return None
    if unit_from == unit_to:
        return value
    a, b = units.get(unit_from), units.get(unit_to)
    if a is None or b is None or a[0] != b[0]:
        return None
    out = float(value) * a[1] / b[1]
    return out if math.isfinite(out) else None


def weighted_mean(pairs):
    """Sum of weight x score over the sum of the weights, both summed in the given order (so all-ones gives exactly
    one); 0 when there is no weight."""
    pairs = list(pairs)
    den = sum(w for w, _s in pairs)
    if not den > 0:
        return 0.0
    return sum(w * s for w, s in pairs) / den
