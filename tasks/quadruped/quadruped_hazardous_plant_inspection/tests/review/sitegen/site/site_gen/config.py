"""Configuration: config/defaults.yaml, an optional user override file, and capability.json.

The capability bands (which heights / clearances / widths / slopes are free, need a special mode, or are barriers)
are the robot's measurements; `derived_thresholds` turns them into the numbers the traversability rasterizer
uses (the mid-points between neighbouring bands, so a size drawn inside a band with a margin is classified the same
way by the map and by the physics).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any

import yaml

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
SITE_DIR = os.path.dirname(PKG_DIR)                       # site/
SKETCH_DIR = os.path.dirname(SITE_DIR)                    # the library root
DEFAULTS = os.path.join(SITE_DIR, "config", "defaults.yaml")
CAPABILITY_JSON = os.path.join(SKETCH_DIR, "capability.json")
BRICKS_OBSERVABILITY = os.path.join(SKETCH_DIR, "bricks", "OBSERVABILITY.md")


class Cfg(dict):
    """A dict with attribute access (read-only convenience)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _robot_bands(cap: dict) -> dict:
    """Translate the robot's `quadruped-capability/1` file (written by the gait's tests: per element kind a list of
    tested sizes with a class free / needs_<mode> / avoid / barrier) into the generator's bands. A band edge is a
    tested size (the measurement already rounds toward the safe side); sizes classed 'avoid' fall between the bands,
    where nothing is drawn. A family without data keeps its placeholder band."""
    el = cap.get("elements", {}) if isinstance(cap.get("elements"), dict) else {}

    def rows(key):
        r = el.get(key, {}).get("per_size") or []
        return [(float(x["size"]), str(x.get("cls", ""))) for x in r if "size" in x]

    out = {}

    def harder_up(keys, need):            # sizes where larger is harder (steps, pipes, slopes)
        rr = []
        for k in keys:
            rr += rows(k)
        if not rr:
            return None
        free = [s for s, c in rr if c == "free"]
        mid = [s for s, c in rr if c == need]
        bar = [s for s, c in rr if c == "barrier"]
        if not mid:
            return None
        d = {"needs_high_step" if need.startswith("needs_high") else need: [min(mid), max(mid)]}
        d["free_max"] = max([s for s in free if s < min(mid)], default=0.5 * min(mid))
        d["barrier_min"] = min([s for s in bar if s > max(mid)], default=1.6 * max(mid))
        return d

    st = harder_up(["door_sill_height_m", "cable_protector_height_m"], "needs_high_step")
    if st:
        out["step"] = st
    fp = harder_up(["floor_pipe_diameter_m_square"], "needs_high_step")
    if fp:
        out["floor_pipe"] = fp
    oh = rows("overhead_clearance_m")
    if oh:
        cr = [s for s, c in oh if c.startswith("needs_crouch")]
        if cr:
            free = [s for s, c in oh if c == "free" and s > max(cr)]
            bar = [s for s, c in oh if c == "barrier" and s < min(cr)]
            out["overhead"] = {"needs_crouch": [min(cr), max(cr)], "free_min": min(free, default=max(cr) + 0.1),
                               "barrier_max": max(bar, default=0.7 * min(cr))}
    wd = rows("corridor_width_m")
    if wd:
        fr = [s for s, c in wd if c == "free"]
        if fr:
            out["width"] = {"free_min": min(fr)}
    sl = rows("ramp_up_deg")
    if sl:
        fr = [s for s, c in sl if c == "free"]
        if fr:
            out["slope_deg"] = {"free_max": max(fr)}
    return out


def _capability_from_json(path: str) -> tuple[dict, dict]:
    """Read capability.json: {"capability": {...}, "modes": {...}}, the bare capability dict, or the robot's
    `quadruped-capability/1` format (translated by _robot_bands)."""
    with open(path) as f:
        cap = json.load(f)
    if not isinstance(cap, dict):
        return {}, {}
    if str(cap.get("format", "")).startswith("quadruped-capability"):
        return _robot_bands(cap), {}
    bands = cap.get("capability", None)
    if bands is None:
        bands = {k: v for k, v in cap.items() if k in ("step", "floor_pipe", "overhead", "width", "slope_deg",
                                                      "friction", "band_margin")}
    modes = cap.get("modes", {}) if isinstance(cap.get("modes", {}), dict) else {}
    return bands, modes


def is_ladder(cfg) -> bool:
    return str(cfg.get("layout", "draft")).startswith("ladder")


def file_label(path: str) -> str:
    """A source file as the generated files name it: base name and sha256, no machine path (the packages are
    vendored into the public task repository; the label tells which version of a shared file a site was made with)."""
    try:
        with open(path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        h = "unreadable"
    return "%s sha256:%s" % (os.path.basename(path), h)


def load_config(path: str | None = None, overrides: dict | None = None, use_capability: bool = True) -> Cfg:
    with open(DEFAULTS) as f:
        cfg = yaml.safe_load(f)
    user = {}
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
    # the layout switch first: the ladder's base overrides go under the user's own settings
    lay = (overrides or {}).get("layout", user.get("layout", cfg.get("layout", "draft")))
    if str(lay).startswith("ladder"):
        _merge(cfg, copy.deepcopy(cfg.get("ladder", {}).get("overrides", {})))
    _merge(cfg, user)
    cfg["_capability_source"] = "defaults.yaml (placeholder bands)"
    if use_capability and os.path.exists(CAPABILITY_JSON):
        try:
            bands, modes = _capability_from_json(CAPABILITY_JSON)
            _merge(cfg["capability"], bands)
            for m, v in modes.items():
                if m in cfg["modes"] and isinstance(v, dict):
                    _merge(cfg["modes"][m], {k: v[k] for k in v if k in cfg["modes"][m]})
            cfg["_capability_source"] = file_label(CAPABILITY_JSON)
        except Exception as e:  # noqa: BLE001  (a broken file must not stop the generator; it is reported)
            cfg["_capability_source"] = "defaults.yaml (capability.json unreadable: %s)" % type(e).__name__
    if overrides:
        _merge(cfg, overrides)
    _apply_v2(cfg)
    return Cfg(cfg)


def is_v2(cfg) -> bool:
    """Version 2's generator: ladder.v2 holds the candidate's factors."""
    return bool((cfg.get("ladder") or {}).get("v2"))


def _apply_v2(cfg: dict) -> None:
    """With ladder.v2 set: the corridor element's kinds at 0.5 / 0.5 and the trench cover among the
    high-step kinds. Idempotent (proofs.load re-applies it to a stored config)."""
    L = cfg.get("ladder") or {}
    if not L.get("v2"):
        return
    P = L.get("v2_params") or {}
    if P.get("corridor_sill_kind_weights"):
        L.setdefault("corridor", {})["sill_kind_weights"] = dict(P["corridor_sill_kind_weights"])
    kinds = [str(k) for k in L.get("high_step_kinds", [])]
    for k in P.get("extra_high_step_kinds", []):
        if str(k) not in kinds:
            kinds.append(str(k))
    L["high_step_kinds"] = kinds


def evidence_label() -> str:
    """The cell's evidence model (mission/evidence.py) as the generated files name it (base name + sha256): the
    regions and proofs of a site hold for this version (it changed at 15:46, ~17:45 and 18:05 EST on
    2026-11-14)."""
    return file_label(os.path.join(SKETCH_DIR, "mission", "evidence.py"))


def acceptance_path(cfg) -> str:
    return os.path.join(SKETCH_DIR, str(cfg.get("ladder", {}).get("acceptance_file", "acceptance_v1.yaml")))


def load_acceptance(cfg) -> dict:
    """The acceptance conditions of an evidence frame (acceptance_v1.yaml, shared with the grader side)."""
    with open(acceptance_path(cfg)) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha1(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:12]


def derived_thresholds(cfg: dict) -> dict:
    """Numbers the map uses, from the capability bands (mid-points between neighbouring bands)."""
    c = cfg["capability"]
    st, fp, oh, wd, sl = c["step"], c["floor_pipe"], c["overhead"], c["width"], c["slope_deg"]
    return {
        "step_free": float(st["free_max"]) + 0.002,
        "step_high": 0.5 * (float(st["needs_high_step"][1]) + float(st["barrier_min"])),
        "pipe_free": float(fp["free_max"]) + 0.002,
        "pipe_high": 0.5 * (float(fp["needs_high_step"][1]) + float(fp["barrier_min"])),
        "clear_walk": 0.5 * (float(oh["needs_crouch"][1]) + float(oh["free_min"])),
        "clear_crouch": 0.5 * (float(oh["barrier_max"]) + float(oh["needs_crouch"][0])),
        "r_in": 0.25 * (float(wd["barrier_max"]) + float(wd["tight"][0])),
        "r_circ": 0.25 * (float(wd["tight"][1]) + float(wd["free_min"])),
        "slope_pass": 0.5 * (float(sl["slow"][1]) + float(sl["barrier_min"])),
        "slope_free": float(sl["free_max"]),
        "friction_slow": float(c["friction"]["slow_max"]),
    }


def observability_windows(cfg: dict) -> tuple[dict, str]:
    """Readable-region windows per IP type: the observability study's numbers when its notes have a
    machine-readable block (```yaml ... observability: ...```), otherwise the placeholders of the config."""
    win = copy.deepcopy(cfg["observability"])
    src = str(win.pop("source", "config placeholders"))
    if os.path.exists(BRICKS_OBSERVABILITY):
        try:
            txt = open(BRICKS_OBSERVABILITY).read()
            import re
            for block in re.findall(r"```yaml\n(.*?)```", txt, re.S):
                d = yaml.safe_load(block)
                if isinstance(d, dict) and isinstance(d.get("observability"), dict):
                    _merge(win, d["observability"])
                    src = BRICKS_OBSERVABILITY + " (yaml block)"
        except Exception:  # noqa: BLE001
            src = src + " (OBSERVABILITY.md yaml block unreadable)"
    return win, src


def get(cfg: Any, dotted: str, default=None):
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
