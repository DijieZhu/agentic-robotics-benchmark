"""The documents of a ladder site (prior.json): the drawing as before, the round sheet, the
modification log, the facts that the facility notes state, the budgets. Nothing in them is ground truth only:
locations are the drawn ones, the 2009 readings are drawn independently of the true values, no normal ranges are
printed, the unlisted source is one line without a location, the modification log holds only true entries."""

from __future__ import annotations

from .ips import REPORT
from .prior import build_prior

TYPE_TEXT = {"gauge_63": "dial gauge, 63 mm case", "gauge_100": "dial gauge, 100 mm case",
             "gauge_160": "dial gauge, 160 mm case", "level": "tubular level gauge", "valve_lever": "quarter-turn valve",
             "valve_handwheel": "rising-stem gate valve", "lamps": "indicator lamp panel",
             "thermal": "temperature point (surface)"}
LOG_KIND = {"door_sealed": "door sealed", "instrument_moved": "instrument moved", "instrument_relocated":
            "instrument relocated", "annex_added": "annex added", "zoning_outdated": "process change, not re-surveyed",
            "repeater_dead": "link repeater out of service", "opening_as_built": "wall opening made",
            "opening_added": "wall opening made", "partition_added": "temporary shielding erected"}
IRRELEVANT = [("lighting replaced", None), ("floor repainted", None), ("signage renewed", None),
              ("fire extinguishers replaced", "corridor"), ("cable tray added", None), ("pump motor rewound", "pump_room"),
              ("breaker replaced", "electrical_room"), ("filters replaced", "decor:hvac_room"),
              ("handrail repaired", "tank_room"), ("drain grating renewed", None)]
FACTS = [
    "A listed instrument is in the room the round sheet gives, in a neighbouring room, or on the corridor wall "
    "outside that room.",
    "An instrument without a recorded location is in the one room that is not on the drawing: an annex built "
    "against the outside of an outer-wall room and entered through a door that is not drawn.",
    "About a third of the listed instruments are not at their drawn place or have no recorded location.",
    # 2026-11-14: what holds on every building; was "Usually
    # one drawn door of the building cannot be passed."
    "One drawn interior door of the building is sealed and cannot be passed. A few other doors may stand ajar; none "
    "of them is the only way to any part of the building.",
    "One room is not on the drawing.",
    "One unlisted source exists somewhere in the building; the zoning was not updated after it appeared.",
    "Grid marker plates hang where the drawing shows them; about three in four of them still exist.",
    "Both link repeaters are shown in service; their state was not checked.",
]
# version 2: the facts block when one drawing sheet is missing (each backed by
# a statement of the documents' own tests)
FACTS_V2 = [
    FACTS[0],
    "An instrument without a recorded location is in the one room that is on no sheet of the drawing: an annex built "
    "against the outside of an outer-wall room and entered through a door that is not drawn.",
    FACTS[2],
    FACTS[3],
    "One sheet of the drawing is missing from the archive copy; nothing of its rooms is drawn. A corridor that runs "
    "onto it is drawn up to a cut within 1 m of the sheet edge, and its entry says that it continues. Besides the "
    "rooms of the missing sheet, one room is on no sheet: the annex.",
    "One unlisted source exists somewhere in the building, in an area the drawing shows as supervised; the zoning "
    "was not updated after it appeared.",
    "Grid marker plates hang where the drawing shows them; about three in four of them still exist. Where a sheet is "
    "missing, no plate is listed and none hangs.",
    "Both link repeaters are shown in service; their state was not checked. The link plan shows them wherever they "
    "hang.",
    "Some of the listed instruments are on the missing sheet; the round sheet gives their recorded location as for "
    "every other instrument.",
    "The building's outer walls stand on the outline on every sheet, their inner face 0.125 m inside it.",
    "Every room of the drawing's sheets but the corridors, those of the missing sheet included, has its number on "
    "the plate beside its main door.",
    "Every listed instrument carries a tag plate with its tag in the plant lettering (one to three letters, a hyphen, "
    "two to five digits, at most eight characters); no two plates of the building carry the same tag; an "
    "instrument is the listed one only if its plate says so.",
]


def _scale(it):
    t = it["ladder_type"]
    p = it["params"]
    if t.startswith("gauge"):
        return {"range": [p["lo"], p["hi"]], "unit": p["unit"]}
    if t == "level":
        return {"range": [0, 100], "unit": "%"}
    if t == "valve_lever":
        return {"range": [0, 90], "unit": "deg (0 open, 90 closed)"}
    if t == "valve_handwheel":
        return {"range": [0, 100], "unit": "% open"}
    if t == "lamps":
        return {"range": None, "unit": "on / off per lamp"}
    if t == "thermal":
        return {"range": list(p.get("range", [0, 150])), "unit": "°C"}
    return {}


def _reading(v):
    """Rename a 2009 record so that no key collides with the ground-truth vocabulary (value, states, ...)."""
    if not v:
        return {}
    out = {"reading": v.get("value")}
    if "unit" in v:
        out["unit"] = v["unit"]
    return out


def build_documents(W, cfg, rng, items, decoys, sources, budgets, markers, zoning, roles, dose):
    lay = W.lay
    A = cfg["ladder"]
    from .config import load_acceptance
    acc = load_acceptance(cfg)
    w_i = float(acc["weights"]["instrument_each"])
    w_m = float(acc["weights"]["mark"])
    w_x = float(acc["weights"]["source"])
    for it in items:
        it["weight"] = w_m if it["ladder_slot"] == "M" else (w_x if it["ladder_slot"] == "X" else w_i)
    old = [dict(i) for i in items]
    prior = build_prior(W, cfg, rng, old, sources, [], budgets, markers, zoning)
    g = rng("ladder_docs")
    # ---------------------------------------------------------------- the round sheet
    wo = {i["tag"]: i for i in prior["work_order"]["items"]}
    rows = []
    for it in sorted([i for i in items if i["ladder_slot"] != "X"], key=lambda i: i["tag"]):
        e = wo[it["tag"]]
        loc = dict(e["location"])
        rec = {"tag": it["tag"], "type": it["type"], "report": REPORT[it["type"]], "weight": it["weight"]}
        if it["ladder_slot"] == "M":
            rec.update({"kind": "survey_mark", "type_text": "survey mark (dose rate on the floor)",
                        "report": "dose rate where the robot stands (Gy/h)",
                        "evidence": {"within_m": acc["mark"]["evidence"]["within_m"],
                                     "for_s": acc["mark"]["evidence"]["for_s"]}})
        else:
            rec.update({"kind": "instrument", "type_text": TYPE_TEXT[it["ladder_type"]], "scale": _scale(it)})
            if not (cfg.get("ladder") or {}).get("v2"):
                rec["label_code"] = it["label_code"]        # version 2: the tag is the only identifier
            t = it["ladder_type"]
            if t.startswith("gauge"):
                rec["case_mm"] = int(it["params"]["case_mm"])
            if t == "level":
                rec["window_m"] = float(it["params"]["window_m"])
            if t == "thermal":
                rec["surface"] = it["params"]["surface"]
                rec["component"] = (e.get("units") or [""])[0]
            if t == "lamps":
                rec["n_lamps"] = len(it["params"]["colors"])
                rec["colors"] = list(it["params"]["colors"])
            rec["service"] = e.get("service", "")
        if loc.get("documented"):
            loc = {"recorded": True, "room": loc["room"], "room_name": loc.get("room_name", ""), "pos": loc.get("pos"),
                   "height": loc.get("height"), "facing_deg": loc.get("facing_deg"), "mount": loc.get("mount", "")}
        else:
            loc = {"recorded": False, "text": "not recorded"}
        rec["location"] = loc
        rec["last_2009"] = dict(_reading(it.get("last_2009")), date="2009-%02d-%02d" % (int(g.integers(3, 10)),
                                                                                        int(g.integers(1, 28))))
        rows.append(rec)
    rows.append({"tag": "UNLISTED", "type": "hotspot", "kind": "unlisted_source", "weight": w_x,
                 "text": "One unlisted source exists somewhere in the building. Report its position.",
                 "report": "x, y of the source in the site frame"})
    prior["round_sheet"] = {"title": "INSPECTION ROUND - AUXILIARY BUILDING GROUND FLOOR",
                            "number": prior["work_order"]["number"], "drawing": prior["drawing"]["number"],
                            "last_round": "2009", "note": "Locations from the general arrangement drawing. Normal "
                            "ranges are not printed. Report each reading with the frame that shows it.",
                            "items": rows}
    # ---------------------------------------------------------------- the modification log
    entries = []
    truth = []
    devs = W._ladder_devs if hasattr(W, "_ladder_devs") else []
    for d in devs:
        logged = bool(g.random() < float(A["mod_log"]["p_entry"]))
        rn = lay.rooms[d["room"]].number if d["room"] in lay.rooms else d["room"]
        tr = {"kind_log": LOG_KIND[d["kind"]], "room_number": rn, "deviation": d["kind"], "logged": logged}
        truth.append(tr)
        if logged:
            entries.append({"year": int(g.integers(1994, 2009)), "kind": tr["kind_log"], "room": rn})
    n_irr = int(g.integers(int(A["mod_log"]["irrelevant"][0]), int(A["mod_log"]["irrelevant"][1]) + 1))
    pool = []
    for txt, need in IRRELEVANT:
        rs = [r for r in lay.rooms.values() if r.on_drawing and r.type not in ("airlock",)]
        if need == "corridor":
            rs = [r for r in rs if r.type == "corridor"]
        elif need and need.startswith("decor:"):
            rs = [r for r in rs if r.type == "decor" and r.kind == need.split(":")[1]]
        elif need:
            rs = [r for r in rs if r.type == need]
        elif True:
            rs = [r for r in rs if r.type != "corridor"]
        if rs:
            pool.append((txt, sorted(rs, key=lambda r: r.number)))
    for k in g.permutation(len(pool))[:n_irr]:
        txt, rs = pool[int(k)]
        r = rs[int(g.integers(len(rs)))]
        truth.append({"kind_log": txt, "room_number": r.number, "deviation": None, "logged": True})
        entries.append({"year": int(g.integers(1994, 2009)), "kind": txt, "room": r.number})
    entries.sort(key=lambda e: (e["year"], e["room"], e["kind"]))
    prior["modification_log"] = {"title": "MODIFICATION RECORD (after the last drawing revision)",
                                 "note": "Entries give the kind of change and the room only.", "entries": entries}
    W._mod_log_truth = truth
    # ---------------------------------------------------------------- the facility notes' facts
    v2 = bool((cfg.get("ladder") or {}).get("v2"))
    prior["facility_facts"] = {"statements": FACTS_V2 if v2 else FACTS,
                               "location_bounds": {"listed": "drawn room, a neighbouring room, or the corridor wall "
                                                             "outside the drawn room",
                                                   "not_recorded": "the one undrawn annex"}}
    if v2:
        prior["facility_facts"]["location_bounds"] = {
            "listed": "the recorded room, a neighbouring room, or the corridor wall outside the recorded room",
            "not_recorded": "the annex, the one room on no sheet"}
    # documented sources: the inventory by room (counts and kinds, no positions)
    inv = {}
    for s in sources:
        if s["documented"]:
            k = lay.rooms[s["room"]].number
            inv.setdefault(k, {"room": k, "count": 0, "text": "waste drums" if s["kind"] == "drum" else s["text"]})
            inv[k]["count"] += 1
    prior["work_order"]["documented_sources"] = [inv[k] for k in sorted(inv)]
    # the legacy work order (the mission package still reads it): the mark's dwell and unit as the round sheet
    # gives them
    for e in prior["work_order"]["items"]:
        if e.get("type") == "dose_point":
            e["dwell_s"] = float(acc["mark"]["evidence"]["for_s"])
            e["report"] = "dose rate where the robot stands (Gy/h)"
    prior["budgets"] = budgets
    return public_format(prior, cfg)


# fields of a round-sheet row that the work-order item of the same tag carries too (the cell's mission() document)
ROUND_SHEET_FIELDS = ("label_code", "type_text", "kind", "scale", "case_mm", "surface", "component", "window_m",
                      "n_lamps", "colors", "last_2009")
# version 2: no label code; the tag is the only thing that identifies a plate
ROUND_SHEET_FIELDS_V2 = tuple(k for k in ROUND_SHEET_FIELDS if k != "label_code")
FORMAT = "plant-documents/1"
MARK_REPORT = "dose rate (Gy/h) at the survey mark"
SOURCE_REPORT = "position (x, y in the site frame) of the unlisted source"


def public_format(prior: dict, cfg=None) -> dict:
    """prior.json as the public document a program is given (2026-11-14): no generator name, no
    site units; each work-order item carries its round-sheet row's fields; the mark is reported in Gy/h without a
    dwell suggestion, the source as a position; budgets in design units; the zoning classes' upper bounds in Gy/h
    (they were in the site unit, without a unit, the last one the string "1.0e9")."""
    import copy
    rs = {it.get("tag"): it for it in prior["round_sheet"]["items"]}
    v2 = bool(cfg is not None and (cfg.get("ladder") or {}).get("v2"))
    for it in prior["work_order"]["items"]:
        r = rs.get(it["tag"])
        if r is not None:
            for k in (ROUND_SHEET_FIELDS_V2 if v2 else ROUND_SHEET_FIELDS):
                if k in r:
                    it[k] = copy.deepcopy(r[k])
        if v2:
            it.pop("label_code", None)
        if it.get("type") == "dose_point":
            it["report"] = MARK_REPORT
            it.pop("dwell_s", None)
        if it.get("type") == "hotspot":
            it["report"] = SOURCE_REPORT
            loc = it.get("location") or {}
            if "text" in loc:
                loc["text"] = "an unlisted source exists somewhere in the building; report its position"
    for it in prior["round_sheet"]["items"]:
        it.pop("evidence", None)
        if it.get("type") == "dose_point":
            it["report"] = MARK_REPORT
        if it.get("type") == "hotspot":
            it["report"] = SOURCE_REPORT
    rsv = prior["work_order"].get("radiation_survey")
    if isinstance(rsv, dict):
        rsv["report"] = "dose-rate readings along the route; the position of any unlisted source"
    prior["work_order"]["title"] = "Inspection round, auxiliary building ground floor (first robotic round)"
    b = prior.get("budgets") or {}
    prior["budgets"] = {k: copy.deepcopy(b[k]) for k in ("time_limit_s", "dose_rating_gy", "frames", "reports")
                        if k in b}
    prior.pop("schema", None)
    prior.pop("dose_units", None)
    z = prior.get("zoning") or {}
    if cfg is not None and z.get("classes"):
        f = float(cfg["ladder"]["dose"]["unit_factor"])
        z["classes"] = [[c, (round(float(b) * f, 4) if float(b) < 1e8 else None)] for c, b in z["classes"]]
        z["unit"] = "Gy/h"
    prior["format"] = FORMAT
    prior["datum"]["uncertainty"] = {"pos_m": 0.05, "yaw_deg": 2.0}           # the set-down error
    return prior
