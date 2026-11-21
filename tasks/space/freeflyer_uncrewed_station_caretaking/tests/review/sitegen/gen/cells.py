"""Labels, seeds and the fixed arrangement of the six cells (site_spec ``layout.*``).

Seeds (site_spec ``layout.seed_rule`` "sha256(label / cell / k)"): the candidate seed of (label, cell, k) is the first
8 bytes (little endian) of sha256("<label>/<M>x<T>/<k>"); attempt j of that candidate draws from
sha256("<label>/<M>x<T>/<k>/<j>").  ``cell`` is written as the site_spec key does it (``L2xS``).

Labels (pairwise different): the 12 trial sites, the 4 debug sites the
reference program is tuned on, the hidden candidates.  The hidden label is fixed here so that the
comparison is made on the real seeds; the hidden-set selection generates its candidates with it after the rules are frozen.
"""
from __future__ import annotations

import hashlib

from cabincell.site import spec as SPEC

LABELS = {"trial": "p09_trial", "debug": "p09_debug", "hidden": "caretaking_hidden_v1"}      # [C]
CELL_ORDER = ("L2xS", "L2xP", "H2xS", "H2xP", "L1xS", "L1xP")
# [C] the 4 debug cells (S and P two each): every module, both failed-light cells, the moved unit of H2, the wrong
# advisory and the low battery of L1 x P, two- and three-inlet element units
DEBUG_CELLS = ("H2xS", "L1xS", "L2xP", "L1xP")
TRIAL_PER_CELL = 2
MODULE_ORDER = ("l2", "h2", "l1")


def cell_key(module: str, tier: str) -> str:
    return f"{module.upper()}x{tier}"


def split(cell: str) -> tuple:
    """('l2', 'S') of 'L2xS'."""
    m, t = cell.split("x")
    if t not in ("S", "P") or m.lower() not in MODULE_ORDER:
        raise ValueError(cell)
    return m.lower(), t


def row(cell: str, values=None) -> dict:
    """The fixed secondary attributes of a cell (site_spec ``layout.cell.<cell>``)."""
    v = values or SPEC.values()
    return dict(v[f"layout.cell.{cell}"])


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:8], "little")


def seed_text(label: str, cell: str, k: int, j: int | None = None) -> str:
    split(cell)
    return f"{label}/{cell}/{int(k)}" + ("" if j is None else f"/{int(j)}")


def candidate_seed(label: str, cell: str, k: int) -> int:
    return _seed(seed_text(label, cell, k))


def attempt_seed(label: str, cell: str, k: int, j: int) -> int:
    return _seed(seed_text(label, cell, k, j))


def derived_seed(seed: int, what: str) -> int:
    """A sub-stream of an attempt (e.g. the telemetry draw), independent of how many numbers the draw used."""
    return _seed(f"{int(seed)}/{what}")


def site_id(kind: str, cell: str, k: int) -> str:
    m, t = split(cell)
    return f"{kind}_{m}x{t.lower()}_{int(k)}"


def plan(kind: str) -> list:
    """[(cell, k)] of a kind: trial = every cell x k 0..1; debug = DEBUG_CELLS x k 0."""
    if kind == "trial":
        return [(c, k) for c in CELL_ORDER for k in range(TRIAL_PER_CELL)]
    if kind == "debug":
        return [(c, 0) for c in DEBUG_CELLS]
    raise ValueError(kind)


def check_arrangement(values=None) -> list:
    """Problems of the cell table against the counts it must give (site_spec layout.counts)."""
    v = values or SPEC.values()
    rows = {c: row(c, v) for c in CELL_ORDER}
    cnt = dict(low_battery=sum(r["battery"] == "low" for r in rows.values()),
               wrong_advisory=sum(r["advisory"] == "wrong_unit" for r in rows.values()),
               failed_light=sum(r["light_circuit"] == "failed" for r in rows.values()),
               moved_stowage_unit=sum(r["stowage_unit"] == "moved" for r in rows.values()))
    want = v["layout.counts"]
    out = [f"{k}: {cnt[k]} != {want[k]}" for k in want if cnt.get(k) != want[k]]
    for c, r in rows.items():
        m, t = split(c)
        if r["module"] != m.upper() or r["tier"] != t:
            out.append(f"{c}: row names {r['module']}x{r['tier']}")
        if r["blocked_unit_limit"] not in ("below", "either"):
            out.append(f"{c}: blocked_unit_limit {r['blocked_unit_limit']}")
    return out
