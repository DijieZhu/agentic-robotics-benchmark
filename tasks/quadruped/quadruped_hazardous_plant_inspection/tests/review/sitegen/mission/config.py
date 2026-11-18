"""Configuration loading.

  cfg = load_cfg()           config/mission.yaml (time, budgets, drift, dose, link, sensors, log)
                             + cfg['acceptance']: acceptance_v1.yaml as read (the shared file, no copy)
                             + cfg['evidence']:   the evidence gate and tolerances derived from that file, in the form
                                                  evidence.py and scoring.py read (evidence_from_acceptance)
                             + cfg['weights'], cfg['return']: the item weights and the return factor of the file
                             + an optional override file and in-code overrides (deep merge)

The package does not keep its own copy of any threshold, tolerance or weight of the acceptance file. Its only own
additions are in config/evidence_lane.yaml (how the geometric record measures a feature; no numbers of the file).
"""

from __future__ import annotations

import copy
import os

import yaml

from . import paths

# acceptance-file tolerance kinds -> the scorer's
_TOL_KIND = {"fraction_of_span": "fraction_of_span", "absolute": "absolute", "all_correct": "all_correct",
             "relative": "relative"}


def _merge(dst, src):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def load_acceptance(path: str | None = None) -> dict:
    with open(path or paths.ACCEPTANCE_YAML) as f:
        acc = yaml.safe_load(f)
    acc["_file"] = os.path.abspath(path or paths.ACCEPTANCE_YAML)
    return acc


def evidence_from_acceptance(acc: dict, lane: dict | None = None) -> dict:
    """The evidence gate of evidence.py / scoring.py from the shared acceptance file (acceptance_v1.yaml).

    Every number comes from the file: per type the camera, feature size in pixels, off-axis limit, minimum distance,
    in-frame share (the lever's 'lever_and_tag_plate' tests the lever and its tag plate, each at that share),
    visible share and tolerance; the 5 m distance cap; the zero factor; the value-free credit; the survey mark's
    tolerance and dwell; the source's distance credit and the E_X averaging window. `lane` (config/evidence_lane.yaml)
    adds only measuring conventions, clearly marked there."""
    out = {"source_file": acc.get("_file"), "version": acc.get("version"),
           "common": {"max_distance_m": float(acc["max_dist_all"]),
                      "value_free_credit": float(acc["value_free_credit"]),
                      "fall_off": float(acc["zero_factor"]),
                      "max_cited_frames": int(acc.get("max_cited_frames", 3)),
                      "lighting": acc.get("lighting")},
           "types": {}}
    for typ, r in acc["types"].items():
        tol = r["tolerance"]
        t = {"sensor": r["camera"], "feature": r["feature"], "min_px": float(r["feature_px"]),
             "max_off_axis_deg": float(r["max_off_axis"]), "min_distance_m": float(r.get("min_dist", 0.0)),
             "in_frame": r.get("in_frame"), "min_in_frame": float(r.get("in_frame_share", 1.0)),
             "min_visible_share": float(r["visible_share"]),
             "tolerance": {"kind": _TOL_KIND[tol["kind"]], "value": float(tol.get("value", 0.0)),
                           "unit": tol.get("unit")}}
        if r.get("in_frame") == "lever_and_tag_plate":
            t["min_tag_in_frame"] = float(r.get("in_frame_share", 1.0))
        if "size_rule" in r:
            t["size_rule"] = r["size_rule"]
        if "off_axis_note" in r:
            t["off_axis_note"] = r["off_axis_note"]
        out["types"][typ] = t
    mk = acc["mark"]
    out["mark"] = {"radius_m": float(mk["evidence"]["within_m"]), "dwell_s": float(mk["evidence"]["for_s"]),
                   "tolerance_rel": float(mk["tolerance"]["value"]), "count_rule": mk.get("count_rule")}
    src = acc["source"]
    out["source"] = {"full_m": float(src["full_credit_within_m"]), "zero_m": float(src["zero_at_m"]),
                     "ex_window_s": float(src["evidence"]["own_dosimeter_mean_over_s"])}
    if lane:
        _merge(out, lane)
    return out


def load_cfg(path: str | None = None, overrides: dict | None = None, acceptance: str | None = None) -> dict:
    with open(os.path.join(paths.CONFIG_DIR, "mission.yaml")) as f:
        cfg = yaml.safe_load(f)
    acc = load_acceptance(acceptance)
    with open(os.path.join(paths.CONFIG_DIR, "evidence_lane.yaml")) as f:
        lane = yaml.safe_load(f) or {}
    cfg["acceptance"] = acc
    cfg["evidence"] = evidence_from_acceptance(acc, lane.get("evidence_additions"))
    W = acc["weights"]
    cfg["weights"] = {"instrument_each": float(W["instrument_each"]), "mark": float(W["mark"]),
                      "source": float(W["source"]),
                      # a draft site with fewer than eight instruments shares the instruments' total equally
                      "instruments_total": 1.0 - float(W["mark"]) - float(W["source"])}
    rf = acc["return_factor"]
    cfg["return"] = {"within_m": float(rf["on_pad_upright_within_m"]), "value": float(rf["value"]),
                     "otherwise": float(rf["otherwise"])}
    cfg["dose"]["e_x_window_s"] = cfg["evidence"]["source"]["ex_window_s"]
    if path:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        _merge(cfg, overrides)
    return cfg
