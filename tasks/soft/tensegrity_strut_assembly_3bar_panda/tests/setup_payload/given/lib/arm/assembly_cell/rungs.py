"""Difficulty-ladder rungs as configuration (CHAIN_PLAN 4.0: raising the rung is a config change + sibling folder).

`grader_config.json` carries the rung-1 knobs at its top level and, under `rungs.<n>`, the overrides of every
higher rung. `active_config(cfg)` deep-merges the active rung's overrides (`cfg["ladder_rung"]`, default 1)
into a copy of the top level -- every key except `note`, `scope` and `scope_add`, which `resolve_scope`
turns into the list of mate ids the cell evaluates. The verify stage and the practice grader call these
two functions, so a rung's knobs are what runs, not documentation.

Knobs honoured by the cell code today (everything under `layout`, `cell`, `latch`, `functional`, `safety`,
`observation_noise`, `pose_bands`, `sim_cap_s`, `wall_cap_s`, `episodes`): see the module docstrings of
layout.py (jitter, fixture jitter, friction range, Panda start noise, sockets), cell_builder.py (shrink,
carve margin), predicates.py (`latch.hold_ticks`, `latch.rel_speed_max_m_s`, `latch.tolerance_scale`),
server.py (`observation_noise.pos_mm / ang_deg` on the reported part poses), grading.py (`pose_bands`).
"""

import copy

__all__ = ["active_config", "resolve_scope", "RUNG_META_KEYS"]

RUNG_META_KEYS = ("note", "scope", "scope_add")


def _deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def active_config(cfg, rung=None):
    """A deep copy of `cfg` with `cfg["rungs"][str(rung)]` merged in (rung defaults to cfg["ladder_rung"], 1).
    The merged copy keeps `ladder_rung` = the active rung and records `scope` / `scope_add` from the rung."""
    out = copy.deepcopy(cfg or {})
    rung = int(out.get("ladder_rung", 1) if rung is None else rung)
    out["ladder_rung"] = rung
    over = (out.get("rungs") or {}).get(str(rung)) or {}
    _deep_update(out, {k: v for k, v in over.items() if k not in RUNG_META_KEYS})
    if "scope" in over:
        out["scope"] = over["scope"]
    if "scope_add" in over:
        out["scope_add"] = list(over["scope_add"])
    return out


def resolve_scope(cfg, graph):
    """Mate ids in scope: `cfg["scope"]` (a name in graph["scopes"], default rung1_prefix) + `cfg["scope_add"]`
    (ids that must exist in graph["mates"]; unknown ids raise)."""
    scopes = graph.get("scopes", {})
    name = cfg.get("scope", "rung1_prefix")
    if name not in scopes:
        raise KeyError("scope %r is not one of %s" % (name, sorted(scopes)))
    ids = list(scopes[name])
    known = {m["id"] for m in graph.get("mates", [])}
    for extra in cfg.get("scope_add") or []:
        if extra not in known:
            raise KeyError("scope_add id %r is not a mate of the mating graph" % extra)
        if extra not in ids:
            ids.append(extra)
    return ids
