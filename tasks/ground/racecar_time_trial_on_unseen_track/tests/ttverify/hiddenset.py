"""The fixed hidden set (grader side): selection rule item 8.

    tests/hidden/selection.json             the selection record; `selected[]` (cell, seed_hex, world_id, layout_digest)
                                            are the venues, in this order; `frozen_after_selection_sha256` pins the
                                            world files and the reference records by sha256
    tests/hidden/<world_id>.json            the venue's world file (the cell builds the venue from it)
    tests/hidden/reference/<world_id>.json  the frozen reference numbers (records.schema.json #/$defs/reference)
    tests/hidden/oracle_token.sha256        the sha256 of the reference program's token (the cell's priv_unlock)

`load` returns the venues or raises ValueError with a message that names no value of the set (a malformed set is a
task error). Every world's `layout_digest` must equal the digest of its content (`check.world_digest`) and the
selection's, every reference's layout digest the world's, every pinned file its sha256.
"""

import hashlib
import json
import os
import re

__all__ = ["load"]

_ID = re.compile(r"^[a-z0-9_]{1,64}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


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
        if world.get("world_id") != wid or ref.get("world_id") != wid:
            raise ValueError("venue %d: the files name another venue" % k)
        digest = check.world_digest(world)
        if not (digest == world.get("layout_digest") == ent.get("layout_digest") == ref.get("layout_digest")):
            raise ValueError("venue %d: the layout digests do not agree" % k)
        err = check.check_record("reference", ref)
        if err:
            raise ValueError("venue %d: the reference record does not pass the contract's checker" % k)
        venues.append({"world_id": wid, "cell": ent.get("cell"), "layout_digest": digest, "world": world, "reference": ref})
    return {"venues": venues, "placeholder": bool(sel.get("placeholder")), "priv_token_sha256": tok_sha,
            "n_pinned": len(pinned)}
