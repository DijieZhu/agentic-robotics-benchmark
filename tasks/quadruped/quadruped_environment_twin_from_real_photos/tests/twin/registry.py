"""The criterion registry and the context object handed to every criterion.

    entries = criteria(cfg)                  # [Entry] in criteria.order
    ok, why = available(entry, truth)        # the module exists (importlib.util.find_spec) and its truth files exist
    res = run_criterion(entry, ctx)          # {"value", "ungated", "detail", "diagnostic", "reweight"}

A criterion is one file twin/score_cX.py with score(ctx) -> {"value": float in [0, 1], "detail": {...}} (optional
"diagnostic", "reweight") and an optional selftest(ctx, result). The orchestrator (part2.grade_twin) runs it in its own
stage; this module only resolves, calls and normalises.
"""
import importlib
import importlib.util
import math
import os

from . import load


class Entry:
    def __init__(self, key, e):
        self.key = key
        self.name = str(e["name"])
        self.module = str(e["module"])
        self.weight = float(e["weight"])
        self.truth = list(e["truth"])
        self.gate = e["gate"]


def criteria(cfg):
    c = cfg["criteria"]
    return [Entry(k, c[k]) for k in c["order"]]


def _modname(entry):
    pkg = __name__.rsplit(".", 1)[0]
    return "%s.%s" % (pkg, entry.module)


def available(entry, truth):
    """(True, None) or (False, why)."""
    try:
        spec = importlib.util.find_spec(_modname(entry))
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return False, "not implemented: no module twin/%s.py" % entry.module
    missing = [k for k in entry.truth if not truth.exists(k)]
    if missing:
        return False, "not implemented: its truth file(s) %s are missing" % ", ".join(missing)
    return True, None


def module(entry):
    return importlib.import_module(_modname(entry))


class Ctx:
    """What a criterion sees (read-only except cache). The core fills align, solids, matching, zones, q, q_detail."""

    def __init__(self, cfg, truth, twin, selftest=False):
        self.cfg = cfg
        self.truth = truth
        self.twin = twin
        self.selftest = bool(selftest)
        self.align = None
        self.solids = {}
        self.matching = None
        self.zones = None
        self.q, self.q_detail = {}, {}
        self.results = {}
        self.name = None
        self.sec = None
        self.deadline = None
        self.payload = load.payload()
        self.payload_dir = load.payload_dir()
        self._cache = {}

    def cache(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def static_scene(self):
        from . import raycast
        return self.cache("_static_scene", lambda: raycast.static_scene(self.twin))

    def full_scene(self):
        from . import raycast
        return self.cache("_full_scene", lambda: raycast.full_scene(self.twin, units="ok"))

    def support(self):
        from . import support
        return self.cache("_support", lambda: support.SupportField(self.twin))


def run_criterion(entry, ctx):
    """Call the module's score(ctx) and normalise the result (the stage wrapper adds the guard and the gate)."""
    mod = module(entry)
    ctx.name = entry.key
    ctx.sec = ctx.cfg.get(entry.key, {})
    out = mod.score(ctx)
    if not isinstance(out, dict) or "value" not in out:
        raise ValueError("criterion %s returned no value" % entry.key)
    try:
        v = float(out["value"])
    except (TypeError, ValueError):
        v = float("nan")
    diag = str(out.get("diagnostic", "") or "")
    if not math.isfinite(v):
        diag = ("non-finite value scored 0; " + diag).strip("; ")
        v = 0.0
    v = min(1.0, max(0.0, v))
    rw = out.get("reweight") or {}
    if rw:
        if not isinstance(rw, dict) or abs(sum(float(x) for x in rw.values())) > 1e-12:
            raise ValueError("criterion %s: reweight deltas must sum to 0" % entry.key)
    # "_stats": a module's clock- and machine-dependent numbers (the walk pool's wall times): kept apart from the
    # detail, they go to the stage record and from there to the timing file only, never into a score or the verdict
    return {"value": v, "detail": out.get("detail", {}), "diagnostic": diag[:240], "reweight": dict(rw),
            "module_file": os.path.basename(mod.__file__ or ""), "timing": out.get("_stats")}
