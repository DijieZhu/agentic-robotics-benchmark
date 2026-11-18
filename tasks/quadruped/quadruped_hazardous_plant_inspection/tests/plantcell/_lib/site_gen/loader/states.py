"""Put every instrument of a compiled site into its true state with the bricks package's state setters (needles,
levers, stems, liquid columns, lamp lenses). site.json records, per instrument, the bricks module, its spec and the
names of its bodies, geoms and joints (model_names.bricks)."""

from __future__ import annotations

import importlib
import os
import sys

import mujoco


def _ensure_bricks():
    """bricks is the instruments' package. In the task it sits beside site_gen (_lib/); in the source tree two levels
    up. Add the first of those that holds it when it is not importable yet."""
    try:
        importlib.import_module("bricks")
        return
    except ImportError:
        pass
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../site_gen/..
    for cand in (here, os.path.dirname(here)):
        if os.path.isdir(os.path.join(cand, "bricks")) and cand not in sys.path:
            sys.path.insert(0, cand)
            return


def _jsonable_to_spec(mod, d):
    """Rebuild a bricks spec dataclass (and its nested TagSpec) from its JSON form in site.json."""
    d = dict(d)
    cls = getattr(mod, d.pop("__class__"))
    if isinstance(d.get("tag"), dict):
        tagmod = importlib.import_module("bricks.targets.tag")
        t = dict(d["tag"])
        t.pop("__class__", None)
        d["tag"] = tagmod.TagSpec(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in t.items()})
    for k, v in list(d.items()):
        if isinstance(v, list) and k not in ("tag",):
            d[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v)
    return cls(**d)


def apply_truth_states(model, data, records) -> int:
    """records: site.json entries (ips, decoys, plain_instruments) with model_names.bricks. Returns the count set."""
    _ensure_bricks()
    from bricks.common import TargetNames  # noqa: WPS433
    n = 0
    for r in records:
        b = (r.get("model_names") or {}).get("bricks")
        if not b:
            continue
        mod = importlib.import_module(b["module"])
        spec = _jsonable_to_spec(mod, b["spec"])
        nm = b["names"]
        names = TargetNames(name=nm["name"], kind=nm["kind"], body=nm["body"], geoms=list(nm["geoms"]),
                            face_geoms=list(nm["face_geoms"]), bodies=list(nm["bodies"]), joints=list(nm["joints"]),
                            materials=list(nm["materials"]), state=dict(nm["state"]))
        names.state["spec"] = spec
        if "lenses" in names.state:
            names.state["lenses"] = [list(x) if isinstance(x, (list, tuple)) else x for x in names.state["lenses"]]
        if "forward" in mod.set_state.__code__.co_varnames:
            mod.set_state(model, data, names, b["value"], forward=False)
        else:
            mod.set_state(model, data, names, b["value"])
        n += 1
    mujoco.mj_forward(model, data)
    return n


def instrument_records(site: dict) -> list:
    """The records apply_truth_states takes from a site.json."""
    return list(site.get("ips", [])) + list(site.get("decoys", [])) + list(site.get("plain_instruments", []))
