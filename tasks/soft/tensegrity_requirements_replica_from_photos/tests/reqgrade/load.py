"""Inputs of the C1-C3 grader: the submission, the truth and the settings. No value of any truth is written here.

Submission. `load_submission(sub_dir)` reads the three deliverables through schema_check.read_submission, the same
bounded reader the public check uses (regular files only, never a symbolic link, at most MAX_FILE_BYTES each; YAML
through the bounded composer: nesting, node and alias budget, integer length; a node budget for JSON; finite numbers
and valid text). There is one reader, so the agent's check and the grader cannot read a file differently.

Truth. `truth_from(gt, cfg)` builds the truth of C1-C3 from verify.py's gt mapping ({stem: parsed JSON of every
gt/*.json}, "freeze" None until gt/freeze.json exists) and from the copies next to this module:
    gt["ledger"]                        rows, values, tiers, draft bands, the C1 records and the fields of each block
    gt["neutral_lists"]                 the W0 truths (requirements_w0)
    gt["freeze"]                        overrides, see below
    schemas/units.json                  unit factors (byte copy of the public schema)
    schemas/quantities.json             the vocabulary (byte copy of the public schema)
    schemas/contract/quantity_classes.json   W0 name -> ledger row (and vector index), tolerance classes (hidden)
    schemas/contract/ledger_map.json         ledger rows ruled out of the graded list before the freeze (hidden)
The graded list, the bands and the tiers follow gt/freeze.json; without it, the ledger's draft: rows flagged graded,
less the ruled-out rows (ledger_map.json, and grader_config.json c1.ruled_out_ledger_rows).
freeze.json keys read here (all optional): "graded" (ledger row ids: exactly these are graded), "ungraded" (row ids
taken off the draft list), "bands" ({row id: band spec}), "tiers" ({row id: tier}), "w0_extra" (row ids whose
quantity counts as W0 in the requirements sheet), "sha256" ({file name in gt/: hex digest}; load_gt checks it, a
mismatch is a task error). A C2 row that is not graded counts as W0 by itself.

Settings (`settings(cfg)`, cfg = the whole grader_config.json; a missing key is a task error that names it):
    c1.tier_weights, c2.tier_weights, c3.tier_weights   {"W1": w, "W2": w, "W3": w} (or one top-level tier_weights)
    c1.groups                           {"topology": w, "dimensions": w, "counts": w}
    c1.geometry_unit                    unit of cable_net.json geometry (optional; metres)
    c1.topology_blocks                  {block: w} for the blocks of the ledger's C1 records
    c1.block_requires                   {block: [blocks]}: a record earns `block` only when its fields of the listed
                                        blocks are right too; or c1.ends_counted_only_with_right_harness_field = true,
                                        read as {"ends": every block whose name ends in "_harness"}
    c2.source_class_miss_factor         credit factor of a matched entry none of whose sources is of an accepted class
"""
import hashlib
import json
import os
import re

from . import bands as B
from . import schema_check as SC

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(PKG_DIR)
SCHEMA_DIR = os.path.join(PKG_DIR, "schemas")
SUBMISSION_FILES = {key: name for key, name, _schema in SC.FILES}
MAX_FILE_BYTES = SC.MAX_FILE_BYTES

# ledger key of a C1 topology record -> (section of cable_net.json, field that identifies the record)
_RECORD_KEY = re.compile(r"^(?:(?P<kind>cable|sensor|controller)_(?P<index>\d+)|spare_input_(?P<strut>[a-z]+))$", re.ASCII)
_SECTION = {"cable": ("cables", "motor_index"), "sensor": ("sensors", "sensor_index"),
            "controller": ("controllers", "controller_index"), "spare_input": ("spare_inputs", "strut")}


# ------------------------------------------------------------------------------------------------------ submission
def load_submission(sub_dir):
    """({key: document}, [problem text]) of the submission folder, through the public check's reader; the problems
    are all G0."""
    docs, problems = SC.read_submission(sub_dir)
    return docs, [p["message"] for p in problems]


# ------------------------------------------------------------------------------------------------------ files
def _json(path):
    with open(path, encoding="utf8") as fh:
        return json.load(fh)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(SC.READ_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_config(tests_dir=TESTS_DIR):
    return _json(os.path.join(tests_dir, "grader_config.json"))


def load_gt(tests_dir=TESTS_DIR):
    """verify.py's gt mapping read from tests_dir/gt, the freeze's sha256 pins checked (RuntimeError on a mismatch)."""
    gt_dir = os.path.join(tests_dir, "gt")
    gt = {fn[:-len(".json")]: _json(os.path.join(gt_dir, fn)) for fn in sorted(os.listdir(gt_dir))
          if fn.endswith(".json")}
    gt.setdefault("freeze", None)
    for name, digest in sorted(((gt["freeze"] or {}).get("sha256") or {}).items()):
        got = _sha256(os.path.join(gt_dir, name))
        if got != digest:
            raise RuntimeError("gt/%s has sha256 %s, freeze.json pins %s" % (name, got, digest))
    return gt


# ------------------------------------------------------------------------------------------------------ settings
def settings(cfg):
    """The C1-C3 settings of a grader_config.json document (KeyError naming every missing key)."""
    missing = []

    def need(block, *keys):
        src = cfg.get(block) if isinstance(cfg.get(block), dict) else {}
        for k in keys:
            if k in src:
                return src[k]
            if k == "tier_weights" and "tier_weights" in cfg:
                return cfg["tier_weights"]
        missing.append("%s.%s" % (block, keys[0]))
        return None

    c1 = cfg.get("c1") if isinstance(cfg.get("c1"), dict) else {}
    blocks = need("c1", "topology_blocks")
    if "block_requires" in c1:
        requires = c1["block_requires"]
    elif "ends_counted_only_with_right_harness_field" in c1:
        harness = sorted(b for b in (blocks or {}) if b.endswith("_harness"))
        requires = {"ends": harness} if c1["ends_counted_only_with_right_harness_field"] else {}
    else:
        requires = None
        missing.append("c1.block_requires (or c1.ends_counted_only_with_right_harness_field)")
    out = {"c1": {"tier_weights": need("c1", "tier_weights"), "groups": need("c1", "groups"),
                  "topology_blocks": blocks, "block_requires": requires,
                  "ruled_out": sorted(c1.get("ruled_out_ledger_rows") or []),
                  "geometry_unit": c1.get("geometry_unit", "m")},
           "c2": {"tier_weights": need("c2", "tier_weights"),
                  "source_class_factor": need("c2", "source_class_miss_factor", "source_class_factor")},
           "c3": {"tier_weights": need("c3", "tier_weights")}}
    if missing:
        raise KeyError("grader_config.json lacks " + ", ".join(missing))
    return out


# ------------------------------------------------------------------------------------------------------ truth
def _record(row):
    m = _RECORD_KEY.match(row["key"])
    if not m:
        raise ValueError("ledger row %s: key %r is not a cable_net record" % (row["id"], row["key"]))
    kind = m.group("kind") or "spare_input"
    section, field = _SECTION[kind]
    value = dict(row["value"])
    index = int(m.group("index")) if m.group("index") is not None else m.group("strut")
    if kind == "spare_input" and value.get("strut") != index:
        raise ValueError("ledger row %s: spare input key and strut disagree" % row["id"])
    fields = {k: v for k, v in value.items() if not (kind == "spare_input" and k == "strut")}
    blocks = {b: list(fs) for b, fs in sorted(row["blocks"].items())}
    for b, fs in blocks.items():
        for f in fs:
            if f not in fields:
                raise ValueError("ledger row %s: block %s names field %s, which the value lacks" % (row["id"], b, f))
    return {"id": row["id"], "key": row["key"], "section": section, "index_field": field, "index": index,
            "fields": fields, "blocks": blocks}


def truth_from(gt, cfg=None, schema_dir=SCHEMA_DIR):
    """The truth of C1-C3 (see the module text), as plain data in the ledger's row order."""
    ledger, neutral = gt["ledger"], gt["neutral_lists"]
    fz = gt.get("freeze") or {}
    contract_dir = os.path.join(schema_dir, "contract")
    qclasses = _json(os.path.join(contract_dir, "quantity_classes.json"))
    lmap = _json(os.path.join(contract_dir, "ledger_map.json"))
    units = B.unit_table(_json(os.path.join(schema_dir, "units.json")))
    vocabulary = sorted(_json(os.path.join(schema_dir, "quantities.json"))["quantities"])
    ruled_out = set(lmap.get("ruled_out_ledger_rows") or {})
    c1cfg = (cfg or {}).get("c1") if isinstance((cfg or {}).get("c1"), dict) else {}
    ruled_out |= set(c1cfg.get("ruled_out_ledger_rows") or [])

    rows = {r["id"]: r for r in ledger["rows"]}
    if "graded" in fz:
        graded = set(fz["graded"])
        unknown = sorted(graded - set(rows))
        if unknown:
            raise ValueError("freeze.json grades unknown ledger rows %s" % unknown)
    else:
        graded = {i for i, r in rows.items() if r.get("graded")} - ruled_out - set(fz.get("ungraded") or [])
    band_over, tier_over = fz.get("bands") or {}, fz.get("tiers") or {}

    def spec(row):
        return B.parse_spec(band_over.get(row["id"], row.get("band_draft")))

    def tier(row):
        return tier_over.get(row["id"], row.get("tier"))

    by_comp = {}
    for r in ledger["rows"]:
        by_comp.setdefault(r["component"], []).append(r)
    records = [_record(r) for r in by_comp.get("C1T", []) if r["id"] in graded]
    dims = [{"id": r["id"], "key": r["key"], "value": r["value"], "unit": r["unit"], "tier": tier(r), "band": spec(r)}
            for r in by_comp.get("C1D", []) if r["id"] in graded]
    counts = [{"id": r["id"], "key": r["key"], "value": r["value"], "tier": tier(r), "band": spec(r)}
              for r in by_comp.get("C1N", []) if r["id"] in graded]
    refs = [{"id": r["id"], "quantity": r["key"], "comparator_class": SC.COMPARATOR_CLASS[r["comparator"]],
             "value": r["value"], "unit": r["unit"], "tier": tier(r), "band": spec(r),
             "accepted": sorted(r.get("accepted_source_classes") or [])}
            for r in by_comp.get("C2", []) if r["id"] in graded]
    derived = [{"id": r["id"], "quantity": r["key"], "value": r["value"], "unit": r["unit"], "tier": tier(r),
                "band": spec(r)} for r in by_comp.get("C3", []) if r["id"] in graded]

    # W0 of the requirements sheet: the contract's W0 names (truth from neutral_lists), every C2 row that is not
    # graded, and the freeze's w0_extra rows whose key is a name of the vocabulary
    w0_rows = {r["id"]: r for r in neutral["requirements_w0"]}
    tol = qclasses["tolerance_classes"]
    w0 = {}
    for name, q in sorted(qclasses["quantities"].items()):
        if q.get("class") != "w0":
            continue
        row = w0_rows[q["ledger"]]
        value = row["truth"] if q.get("ledger_index") is None else row["truth"][q["ledger_index"]]
        w0[name] = {"id": q["ledger"], "quantity": name, "value": value, "unit": row["unit"],
                    "band": B.parse_spec(tol[q["tolerance_class"]]["ledger_band"])}
    extra = set(fz.get("w0_extra") or [])
    for r in by_comp.get("C2", []):
        if r["id"] not in graded or r["id"] in extra:
            w0[r["key"]] = {"id": r["id"], "quantity": r["key"], "value": r["value"], "unit": r["unit"], "band": spec(r)}
    for i in sorted(extra):
        r = rows.get(i)
        if r is not None and r["key"] in vocabulary and r["key"] not in w0 and not isinstance(r["value"], dict):
            w0[r["key"]] = {"id": i, "quantity": r["key"], "value": r["value"], "unit": r["unit"], "band": spec(r)}
    refs = [x for x in refs if x["id"] not in extra]
    graded_names = {x["quantity"] for x in refs}
    classes = {n: ("graded" if n in graded_names else "w0" if n in w0 else "unsourced") for n in vocabulary}

    for x in dims + counts + refs + derived:
        if x["tier"] is None:
            raise ValueError("ledger row %s has no tier" % x["id"])
    return {"c1": {"records": records, "dimensions": dims, "counts": counts},
            "c2": {"reference": refs, "w0": w0, "classes": classes},
            "c3": {"derived": derived},
            "units": units,
            "provenance": {"freeze": bool(gt.get("freeze")), "graded_rows": len(graded)}}


def load_truth(tests_dir=TESTS_DIR, cfg=None):
    return truth_from(load_gt(tests_dir), cfg)
