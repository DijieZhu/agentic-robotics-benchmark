"""Robot envelope numbers used by the site family, read from ONE table.

Order: the capability sheet of the simulation base if it exists, else the maintainers' local table (LOCAL_PATH),
else the provisional values below (the site family's first estimates). The choice is made per key (a sheet that lacks
a key falls back for that key only) and recorded in every site package (the label and sha256 of each table read).
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_DIR = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CAP_PATH = os.path.normpath(os.path.join(DEV_DIR, "..", "..", "yale", "notes", "sim_base_data", "capability_sheet.json"))
LOCAL_PATH = os.path.join(DEV_DIR, "build", "capability_local.json")

# provisional values: the site family's first estimates [measured on a development build of the simulation base]
PROVISIONAL = {
    "W_ROLL": 0.333,           # m, end-cap span along the principal axis during a forward roll
    "W_SWEEP": 0.44,           # m, lateral width swept by one forward roll (max over rolls and start faces)
    "D_ROLL": 0.306,           # m, span along the rolling direction
    "H_ROLL": 0.263,           # m, highest point while a primitive runs
    "H_REST": 0.230,           # m, highest point at rest
    "R_TURN": [0.199, 0.270],  # m, max horizontal distance of the end caps from the start during ccw / cw
    "D_STOP": [0.132, 0.168],  # m, centre-to-wall distance when a roll is stopped by a wall
    "ROLL_MEAN": 0.24,         # m, mean forward-roll displacement on flat concrete
}
PROVISIONAL_SOURCE = ("provisional values: the site family's first estimates, measured on a development build of the "
                      "simulation base")
KEYS = tuple(PROVISIONAL)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick(table, key):
    """A key's value from a capability table: a number, a [lo, hi] list, or {"value": ...} / {"max": ...}."""
    if not isinstance(table, dict):
        return None
    src = table.get("keys", table)
    if key not in src:
        return None
    v = src[key]
    if isinstance(v, dict):
        for k in ("value", "max", "range"):
            if k in v:
                v = v[k]
                break
        else:
            return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(q, (int, float)) for q in v):
        return [float(v[0]), float(v[1])]
    return None


def load(cap_path=CAP_PATH, local_path=LOCAL_PATH):
    """Envelope values + per-key provenance (label and sha256 of each table that was read)."""
    tables = []
    for label, p in (("capability_sheet", cap_path), ("capability_local", local_path)):
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                tables.append((label, p, json.load(fh), _sha256(p)))
    values, source = {}, {}
    for k in KEYS:
        for label, p, t, _h in tables:
            v = _pick(t, k)
            if v is not None:
                if isinstance(PROVISIONAL[k], list) and not isinstance(v, list):
                    v = [v, v]
                if not isinstance(PROVISIONAL[k], list) and isinstance(v, list):
                    v = max(v)
                values[k], source[k] = v, label
                break
        else:
            values[k], source[k] = PROVISIONAL[k], "design_v1.0_section_3"
    files = [dict(table=label, sha256=h) for label, p, _t, h in tables]
    return dict(values=values, source=source, files_read=files)
