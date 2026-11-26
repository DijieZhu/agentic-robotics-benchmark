"""cgrading: strict params.json parser and scoring for cloth_material_id_from_video (hidden grader).

Ten fabrics A-J, four tests per family (slide, overhang) each; a fabric's score is the product of its two family means.

Every test is planned in cphys with the submitted material and executed once on the real fabric (creal, with the
fabric's real material from ref/fabrics.json).

params.json (numbers only):
{"fabrics": {"A": {"young": E, "poisson": nu, "thickness": t, "damping": d, "mu": m}, "B": {...}, "C": {...}}}
Each value is clamped to cphys.RANGES before use.
"""
import json
import math
import time

import numpy as np

import cphys as W
import creal as R
import ctasks as T

MAX_BYTES = 64 * 1024
FABRICS = tuple("ABCDEFGHIJ")
FAMS = ("slide", "overhang")


class InvalidSubmission(ValueError):
    pass


def _no_dupes(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise InvalidSubmission("duplicate key %r" % (k,))
        out[k] = v
    return out


def _reject_constant(name):
    raise InvalidSubmission("non-finite number %s" % name)


def _num(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise InvalidSubmission("%s must be a number" % where)
    try:
        f = float(v)
    except (OverflowError, ValueError):
        raise InvalidSubmission("%s is not a usable number" % where)
    if not math.isfinite(f):
        raise InvalidSubmission("%s must be finite" % where)
    return f


def parse_submission_bytes(data):
    """Returns (entries, invalid): entries[k] = clamped material dict for valid fabrics; invalid[k] = reason."""
    if data is None or len(data) > MAX_BYTES:
        raise InvalidSubmission("missing or larger than %d bytes" % MAX_BYTES)
    try:
        doc = json.loads(data.decode("utf-8"), object_pairs_hook=_no_dupes, parse_constant=_reject_constant)
    except InvalidSubmission:
        raise
    except (ValueError, RecursionError, MemoryError, UnicodeDecodeError) as exc:
        raise InvalidSubmission("not valid JSON (%s)" % type(exc).__name__)
    if not isinstance(doc, dict) or not isinstance(doc.get("fabrics"), dict):
        raise InvalidSubmission("top level must be an object with a 'fabrics' object")
    entries, invalid = {}, {}
    for k in FABRICS:
        e = doc["fabrics"].get(k)
        try:
            if not isinstance(e, dict):
                raise InvalidSubmission("missing")
            mat = {}
            for key in W.KEYS:
                if key not in e:
                    raise InvalidSubmission("%s missing" % key)
                mat[key] = _num(e[key], key)
            entries[k] = W.clamp_material(mat)
        except InvalidSubmission as exc:
            invalid[k] = str(exc)
    return entries, invalid


def load_ref(path):
    with open(path) as f:
        return json.load(f)["fabrics"]


def score_test(material, ref_f, test):
    try:
        q, info = T.run_test(material, ref_f["material"], ref_f["mass_kg"], test, R)
        q = float(q) if math.isfinite(q) else 0.0
    except Exception as exc:  # a submitted material that breaks the simulation scores 0 on that test
        q, info = 0.0, {"error": type(exc).__name__}
    return q, info


def aggregate(qs_by_test, tests):
    """Fabric score = the product over the two families of the family's mean test quality: a material is worth
    planning with for a fabric only if its plans transfer in both kinds of handling."""
    fam_q = {}
    for q, t in zip(qs_by_test, tests):
        fam_q.setdefault(t["fam"], []).append(q)
    fm = {f: float(np.mean(fam_q[f])) if f in fam_q else 0.0 for f in FAMS}
    return float(np.prod([fm[f] for f in FAMS])), fm
