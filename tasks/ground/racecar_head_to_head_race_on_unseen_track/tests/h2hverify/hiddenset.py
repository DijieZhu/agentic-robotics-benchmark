"""The fixed hidden set of the head-to-head day (grader side).

    tests/hidden/selection.json             the selection record; `selected[]` (cell, world_id, layout_digest) are the
                                            venues, in this order; `frozen_after_selection_sha256` pins the world files
                                            and the reference records by sha256; `placeholder` true marks stand-ins
    tests/hidden/<world_id>.json            the venue's world file (racecar-venue-world-h2h/1: the base world of the
                                            venue generator plus the grid and the other cars of both heats; the cell
                                            builds the venue from it)
    tests/hidden/reference/<world_id>.json  the frozen reference numbers (racecar-venue-reference-h2h/1: the venue's
                                            lap count, the reference program's clean overtake balance in each heat and
                                            the session time of its N-th lap in each heat)
    tests/hidden/oracle_token.sha256        the sha256 of the reference program's token (the cell's priv_unlock)

`load` returns the venues or raises ValueError with a message that names no value of the set (a malformed set is a
task error). Every world's `layout_digest` must equal the digest of its content (`check.world_digest`) and the
selection's, every reference's layout digest and lap count the world's, every pinned file its sha256. A reference
balance below `MIN_C_REF` is malformed: the score divides by `C_ref - 1` and leaves one overtake of margin.
"""

import hashlib
import json
import math
import os
import re

__all__ = ["load", "WORLD_SCHEMA", "REFERENCE_SCHEMA", "MIN_C_REF", "HEATS"]

WORLD_SCHEMA = "racecar-venue-world-h2h/1"
REFERENCE_SCHEMA = "racecar-venue-reference-h2h/1"
MIN_C_REF = 5
HEATS = 2
HEAT_CAP_S = 300.0

_ID = re.compile(r"^[a-z0-9_]{1,64}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def reference_errors(ref, world, placeholder=False):
    """[problem] of a reference record against its world (no value of either is named)."""
    err = []
    if ref.get("schema") != REFERENCE_SCHEMA:
        err.append("schema")
    if ref.get("world_id") != world.get("world_id") or ref.get("layout_digest") != world.get("layout_digest"):
        err.append("another venue")
    laps = world.get("venue_sheet", {}).get("heat_laps")
    if not _int(laps) or laps < 1 or ref.get("heat_laps") != laps:
        err.append("lap count")
    c_ref = ref.get("C_ref")
    if not isinstance(c_ref, list) or len(c_ref) != HEATS or not all(_int(c) and c >= MIN_C_REF for c in c_ref):
        err.append("reference balance")
    t_n = ref.get("T_N_s")
    ok_t = (lambda t: (_num(t) and 0.0 < float(t) <= HEAT_CAP_S) or (placeholder and t is None))
    if not isinstance(t_n, list) or len(t_n) != HEATS or not all(ok_t(t) for t in t_n):
        err.append("lap times")
    return err


def load(hidden_dir, check):
    """{"venues": [{"world_id", "cell", "layout_digest", "world", "reference"}], "placeholder": bool,
    "priv_token_sha256": hex, "n_pinned": int}. `check` is the contract's checker module (sketch.examples.check)."""
    sel_p = os.path.join(hidden_dir, "selection.json")
    try:
        sel = _json(sel_p)
    except (OSError, ValueError):
        raise ValueError("the selection record cannot be read")
    selected = sel.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("the selection record names no venue")
    placeholder = sel.get("placeholder")
    if not isinstance(placeholder, bool):
        raise ValueError("the selection record does not say whether it is a stand-in")
    pinned = sel.get("frozen_after_selection_sha256") or {}
    if not isinstance(pinned, dict):
        raise ValueError("the selection record's sha256 table is malformed")
    tok_p = os.path.join(hidden_dir, "oracle_token.sha256")
    try:
        with open(tok_p, "r", encoding="ascii") as f:
            tok_sha = f.read().strip()
    except OSError:
        raise ValueError("the oracle token's sha256 cannot be read")
    if not _HEX64.match(tok_sha):
        raise ValueError("the oracle token's sha256 is malformed")
    if sel.get("oracle_token_sha256") not in (None, tok_sha):
        raise ValueError("the selection record and oracle_token.sha256 name different tokens")
    venues, seen = [], set()
    for k, ent in enumerate(selected):
        wid = ent.get("world_id") if isinstance(ent, dict) else None
        if not isinstance(wid, str) or not _ID.match(wid) or wid in seen:
            raise ValueError("venue %d of the selection record is malformed" % k)
        seen.add(wid)
        wp = os.path.join(hidden_dir, wid + ".json")
        rp = os.path.join(hidden_dir, "reference", wid + ".json")
        for rel, p in (("%s.json" % wid, wp), ("reference/%s.json" % wid, rp)):
            if not os.path.isfile(p) or os.path.islink(p):
                raise ValueError("venue %d: a file of the hidden set is missing" % k)
            want = pinned.get("tests/hidden/" + rel)
            if want is not None and _sha(p) != want:
                raise ValueError("venue %d: a file of the hidden set differs from its pinned sha256" % k)
        world, ref = _json(wp), _json(rp)
        if world.get("schema") != WORLD_SCHEMA:
            raise ValueError("venue %d: the world file is not a head-to-head world" % k)
        if world.get("world_id") != wid or ref.get("world_id") != wid:
            raise ValueError("venue %d: the files name another venue" % k)
        digest = check.world_digest(world)
        if not (digest == world.get("layout_digest") == ent.get("layout_digest") == ref.get("layout_digest")):
            raise ValueError("venue %d: the layout digests do not agree" % k)
        bad = reference_errors(ref, world, placeholder=placeholder)
        if bad:
            raise ValueError("venue %d: the reference record is malformed (%s)" % (k, ", ".join(bad)))
        venues.append({"world_id": wid, "cell": ent.get("cell"), "layout_digest": digest, "world": world, "reference": ref})
    return {"venues": venues, "placeholder": placeholder, "priv_token_sha256": tok_sha, "n_pinned": len(pinned)}
