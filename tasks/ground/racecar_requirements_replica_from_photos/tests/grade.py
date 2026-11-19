"""Static grader of the race-car replica task (rc-replica-grader/1; the scoring that tests/README.md describes).

Never imports or executes the submission. The three files are read as data through schema_check.py, the public
structure check (a byte copy of the agent's; its sha256 is pinned in grader_config.json), so the grader and the
agent's check take one loading path: a YAML node budget (aliases accepted, cycles and expansions refused), no
over-long integers, no duplicate keys, strict JSON, finite numbers, the three schemas. Everything else comes from
this folder: procedure.py (a byte copy of the agent's public procedure rc-procedure/2, sha256 pinned),
schemas.json, grader_config.json, anchor.json and the hidden world under gt/.

    value, metrics, diagnostic = grade_stage(stage_dir, submission_dir, g1=True)     # what verify.py calls

and, piece by piece (to grade many submissions on one shared enumeration):

    stg = load_stage(stage_dir)              # modules, pinned bytes, config, schemas (WorldError when broken)
    w = load_world(stg)                      # gt/ files, structure checked against the config's closed lists
    st = State(stg, w)                       # the 17 pass items over every catalogue car; J* recomputed
    st.check_stored(w.optimum)               # WorldError on any difference from optimum.json
    docs = read_submission(stg, folder)      # gate G0 on the files (SubmissionError)
    docs = check_documents(stg, st.cat, docs)
    value, metrics, terms = st.score(docs)

reward = G0 x G1 x (0.30 C1 + 0.10 C2 + 0.20 C3 + 0.10 C4 + 0.30 C5) (G1 is verify.py's lockdown gate):

  C1 kinematic replication: per sketch quantity, a length scores 1 for an error <= max(1.5 % of the truth, 1.0 mm)
     and 0 for an error >= max(6 %, 4.0 mm), linear between; an angle 0.5 deg / 3 deg; a count must be equal. The
     group mean, weighted heights 0.30, layout 0.25, steering 0.35, chassis 0.05, gears 0.05. tolerance and
     sources are not scored.
  C2 requirements sheet: a facet counts when its id is a reference facet, |value - reference| <= tolerance, its
     unit is the reference unit string and one of its sources is a reference source; C2 = 2 x counted /
     (reference facets + submitted facets).
  C3 derived requirements: the 7 predictions against the procedure's value on the real car (gt/car_truth.json)
     with the agent's BOM and the arrival lots (world.json lot_table); relative error <= 3 % scores 1, >= 15 %
     scores 0, linear between; a wrong unit string, or a procedure value that is not finite, scores 0.
  C4 BOM acceptance: 1 when all 17 pass items hold for the agent's BOM on the real car, else 0.
  C5 BOM economy: J_eff = min over the passing cars c of price(c) + ECO x (lines where c differs from the BOM);
     C5 = clip(1 - (J_eff - J*) / ECO, 0, 1). J* = the cheapest passing car, enumerated here over the whole
     catalogue (ties: the earlier car in product order).

A stored world value that differs from its recomputation (J*, its car, the passing set), a grader file that is
broken or a pinned module whose bytes differ ends the run as a TASK ERROR (WorldError -> robotics_grader.fail),
never as a score. The verdict's metrics hold finite numbers only, with a key set that does not depend on the
submission; the diagnostic (at most 2000 characters) names the submission's own structure problems and nothing of
the hidden world.

The procedure surface this grader reads (procedure.py, rc-procedure/2, offers it):
    PROCEDURE_VERSION == "rc-procedure/2"; SLOT_ORDER (the ten purchase lines)
    load_catalogue(obj) -> cat with cat.codes[slot] (list, index order), cat.sizes, cat.n_combinations
    load_car(obj) -> car (an rc-car/1 record; ValueError when malformed)
    iter_cars(cat, block) -> (n, 10) integer index rows, itertools.product order over SLOT_ORDER
    price_block(idx, cat) -> (n,) USD
    prepare(cat, car, context) -> prepared, context = {"lot_table", "cycles", "scenarios"} of world.json
    evaluate_block(idx, cat, prepared, [(quantity, conditions), ...]) -> list of (n,) float64
    sat(q, comparator, bound) -> (n,) bool, comparators "<=", ">=" and "==" (the fit item)

Formats of the hidden world (gt/; names in grader_config.json world_files):
    catalogue.json        rc-catalogue/2 (the procedure's catalogue, read by procedure.load_catalogue)
    car_truth.json        rc-car/1 (the real car, read by procedure.load_car)
    world.json            {"schema": "rc-world/2", "rows": [{"id", "quantity", "comparator", "bound",
                           "conditions", ...} x 17, in the order of the config's pass_items], "derived": [{"id",
                           "quantity", "conditions"} x 7, the config's order], "lot_table", "cycles", "scenarios",
                           ...}; other keys are ignored here
    key_quantities.json   {"schema": "rc-key-quantities/1", "sketch": {<33 sketch ids>: number}}
    reference_sheet.json  {"schema": "rc-reference-sheet/1", "facets": [{"id", "value", "unit", "tolerance",
                           "sources": [source ids]} x 20, the config's order]}
    optimum.json          {"schema": "rc-optimum/1", "J_usd", "car": {slot: SKU}, "passing_set": {"n", "sha256"}}
"""
import hashlib
import json
import math
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRADER_VERSION = "rc-replica-grader/1"
CONFIG_FORMAT = "rc-replica-grader-config/1"
SUBMISSION_FILES = ("requirements.yaml", "bom.json", "kinematic_sketch.json")
CRITERIA = (("C1", "kinematic_replication"), ("C2", "requirements_sheet"), ("C3", "derived_requirements"),
            ("C4", "bom_acceptance"), ("C5", "bom_economy"))
CRITERION_NAME = dict(CRITERIA)
WORLD_FILES = ("catalogue", "car", "world", "key_quantities", "reference_sheet", "optimum")
SKETCH_KINDS = ("length", "angle", "count")
COMPARATORS = ("<=", ">=", "==")


class SubmissionError(ValueError):
    """Gate G0: the submission breaks a structure rule. It scores 0; it is never a task error. The message names
    only the submission's own content and public constants."""


class WorldError(RuntimeError):
    """The grader's own files are broken, or a stored world value differs from its recomputation. A task error,
    never a score. The message names files and checks, never a hidden value."""


# ================================================================================================ small helpers
def ramp(x):
    """clip(x, 0, 1); 0 for a value that is not finite."""
    x = float(x)
    return min(max(x, 0.0), 1.0) if math.isfinite(x) else 0.0


def canonical_digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _one_line(text, limit=300):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def utf8_text(text):
    """text as valid UTF-8 (a lone surrogate from a YAML or JSON escape is written as its escape)."""
    return str(text).encode("utf-8", "backslashreplace").decode("utf-8")


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _strict_json(text):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError("duplicate key %r" % (k,))
            out[k] = v
        return out

    def const(name):
        raise ValueError("%s is not a number" % name)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=const)


def _read_json(path, what):
    try:
        return _strict_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorldError("%s cannot be read: %s" % (what, _one_line(exc, 200))) from None


def _load_module(name, path):
    """The module at path, executed from the very bytes whose sha256 is returned (so a pin checks the code that
    runs), as a private module: nothing is imported from anywhere else."""
    try:
        src = Path(path).read_bytes()
    except OSError as exc:
        raise WorldError("%s cannot be read: %s" % (Path(path).name, _one_line(exc, 200))) from None
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    try:
        exec(compile(src, str(path), "exec"), mod.__dict__)  # noqa: S102 (our own pinned file)
    except Exception as exc:  # noqa: BLE001
        raise WorldError("%s cannot be loaded: %s" % (Path(path).name, type(exc).__name__)) from None
    return mod, hashlib.sha256(src).hexdigest()


# ================================================================================================ the stage
class Stage(object):
    """The modules, the configuration and the schemas of one grader folder."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def check_config(cfg, P, SC, psha, ssha):
    """WorldError when grader_config.json is not a complete rc-replica-grader-config/1 object that agrees with the
    pinned modules."""
    need = ("format", "contract", "procedure_sha256", "schema_check_sha256", "weights", "c1", "c3", "eco_usd",
            "limits", "tolerances", "block_cars", "cars", "slot_order", "sketch", "facets", "derived",
            "pass_items", "world_dir", "world_files")
    if not isinstance(cfg, dict) or any(k not in cfg for k in need) or cfg["format"] != CONFIG_FORMAT:
        raise WorldError("grader_config.json is not an %s object with %s" % (CONFIG_FORMAT, ", ".join(need)))
    if cfg["procedure_sha256"] != psha:
        raise WorldError("procedure.py differs from the bytes grader_config.json pins")
    if cfg["schema_check_sha256"] != ssha:
        raise WorldError("schema_check.py differs from the bytes grader_config.json pins")
    if getattr(P, "PROCEDURE_VERSION", None) != "rc-procedure/2":
        raise WorldError("procedure.py is not rc-procedure/2")
    for fn in ("load_catalogue", "load_car", "iter_cars", "price_block", "prepare", "evaluate_block", "sat"):
        if not callable(getattr(P, fn, None)):
            raise WorldError("procedure.py has no function %s" % fn)
    w = cfg["weights"]
    if not isinstance(w, dict) or set(w) != set(CRITERION_NAME) or not all(_is_num(v) and v >= 0 for v in
                                                                           w.values()) \
            or abs(math.fsum(w.values()) - 1.0) > 1e-12:
        raise WorldError("grader_config.json: weights must name C1..C5 and sum to 1")
    if list(cfg["slot_order"]) != list(P.SLOT_ORDER):
        raise WorldError("grader_config.json: slot_order differs from the procedure's")
    c1 = cfg["c1"]
    groups = c1.get("group_weights") if isinstance(c1, dict) else None
    bands = c1.get("bands") if isinstance(c1, dict) else None
    if not isinstance(groups, dict) or not all(_is_num(v) and v >= 0 for v in groups.values()) or abs(
            math.fsum(groups.values()) - 1.0) > 1e-12:
        raise WorldError("grader_config.json: c1.group_weights must sum to 1")
    lb, ab = (bands or {}).get("length"), (bands or {}).get("angle")
    if not (isinstance(lb, dict) and all(_is_num(lb.get(k)) and lb[k] >= 0 for k in
                                         ("full_rel", "full_abs", "zero_rel", "zero_abs"))
            and lb["full_rel"] < lb["zero_rel"] and lb["full_abs"] < lb["zero_abs"]
            and isinstance(ab, dict) and all(_is_num(ab.get(k)) and ab[k] >= 0 for k in ("full_abs", "zero_abs"))
            and ab["full_abs"] < ab["zero_abs"]):
        raise WorldError("grader_config.json: c1.bands must give the length and angle bands")
    c3 = cfg["c3"]
    if not (isinstance(c3, dict) and _is_num(c3.get("full_rel")) and _is_num(c3.get("zero_rel"))
            and 0 <= c3["full_rel"] < c3["zero_rel"]):
        raise WorldError("grader_config.json: c3 must give full_rel < zero_rel")
    if not (_is_num(cfg["eco_usd"]) and cfg["eco_usd"] > 0):
        raise WorldError("grader_config.json: eco_usd must be a positive number")
    lim = cfg["limits"]
    want = {"max_file_bytes": SC.MAX_FILE_BYTES, "max_yaml_nodes": SC.MAX_YAML_NODES, "max_abs_int": SC.MAX_ABS_INT}
    if not isinstance(lim, dict) or any(lim.get(k) != v for k, v in want.items()) or not (
            _is_int(lim.get("diagnostic_chars")) and 0 < lim["diagnostic_chars"] <= 2000) or not (
            _is_num(lim.get("max_error_metric")) and lim["max_error_metric"] > 0):
        raise WorldError("grader_config.json: limits must equal schema_check.py's and give diagnostic_chars in "
                         "1..2000 and max_error_metric")
    tol = cfg["tolerances"]
    if not isinstance(tol, dict) or set(tol) != {"j_usd_rel", "j_usd_abs", "facet_rel"} or not all(
            _is_num(v) and v >= 0 for v in tol.values()):
        raise WorldError("grader_config.json: tolerances must give j_usd_rel, j_usd_abs, facet_rel")
    if not (_is_int(cfg["block_cars"]) and cfg["block_cars"] >= 1 and _is_int(cfg["cars"]) and cfg["cars"] >= 1):
        raise WorldError("grader_config.json: block_cars and cars must be positive integers")
    sk = cfg["sketch"]
    if not isinstance(sk, list) or not sk or not all(
            isinstance(x, dict) and isinstance(x.get("id"), str) and x.get("group") in groups
            and x.get("kind") in SKETCH_KINDS and isinstance(x.get("unit"), str) for x in sk) \
            or len({x["id"] for x in sk}) != len(sk) or {x["group"] for x in sk} != set(groups):
        raise WorldError("grader_config.json: sketch must list unique {id, group, kind, unit} over the c1 groups")
    for key in ("facets", "derived"):
        lst = cfg[key]
        if not isinstance(lst, list) or not lst or not all(
                isinstance(x, dict) and isinstance(x.get("id"), str) and isinstance(x.get("unit"), str)
                for x in lst) or len({x["id"] for x in lst}) != len(lst):
            raise WorldError("grader_config.json: %s must list unique {id, unit}" % key)
    pi = cfg["pass_items"]
    if not isinstance(pi, list) or not pi or not all(
            isinstance(x, dict) and isinstance(x.get("id"), str) and isinstance(x.get("quantity"), str)
            and x.get("comparator") in COMPARATORS for x in pi) or len({x["id"] for x in pi}) != len(pi):
        raise WorldError("grader_config.json: pass_items must list unique {id, quantity, comparator}")
    files = cfg["world_files"]
    if not isinstance(files, dict) or set(files) != set(WORLD_FILES) or not all(
            isinstance(v, str) and v and "/" not in v and v not in (".", "..") for v in files.values()) \
            or not isinstance(cfg["world_dir"], str) or "/" in cfg["world_dir"] or cfg["world_dir"] in ("", ".."):
        raise WorldError("grader_config.json: world_dir and world_files (%s) are invalid" % ", ".join(WORLD_FILES))


def load_stage(stage=None):
    """Stage of a grader folder: procedure.py and schema_check.py executed from their pinned bytes, the checked
    configuration and the three submission schemas. WorldError when anything is broken."""
    stage = Path(stage or HERE)
    P, psha = _load_module("rc_replica_procedure", stage / "procedure.py")
    SC, ssha = _load_module("rc_replica_schema_check", stage / "schema_check.py")
    cfg = _read_json(stage / "grader_config.json", "grader_config.json")
    check_config(cfg, P, SC, psha, ssha)
    schemas = _read_json(stage / "schemas.json", "schemas.json")
    if not isinstance(schemas, dict) or set(schemas) != set(SUBMISSION_FILES) or not all(
            isinstance(v, dict) for v in schemas.values()):
        raise WorldError("schemas.json must hold exactly the schemas of %s" % ", ".join(SUBMISSION_FILES))
    if tuple(SC.FILES) != SUBMISSION_FILES:
        raise WorldError("schema_check.py names other submission files")
    return Stage(dir=stage, P=P, SC=SC, cfg=cfg, schemas=schemas, procedure_sha256=psha, schema_check_sha256=ssha)


# ================================================================================================ the hidden world
class World(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


def check_world_objects(cfg, world, keyq, ref, opt):
    """Structure of world.json, key_quantities.json, reference_sheet.json and optimum.json against the config's
    closed lists (WorldError)."""
    if not isinstance(world, dict) or world.get("schema") != "rc-world/2":
        raise WorldError("world.json is not an rc-world/2 object")
    for k in ("rows", "derived", "lot_table", "cycles", "scenarios"):
        if k not in world:
            raise WorldError("world.json lacks %s" % k)
    if not all(isinstance(world[k], dict) for k in ("lot_table", "cycles", "scenarios")):
        raise WorldError("world.json: lot_table, cycles and scenarios must be objects")
    rows = world["rows"]
    want = cfg["pass_items"]
    if not isinstance(rows, list) or len(rows) != len(want):
        raise WorldError("world.json: rows must hold the %d pass items" % len(want))
    for r, x in zip(rows, want):
        if not isinstance(r, dict) or r.get("id") != x["id"] or r.get("quantity") != x["quantity"] or r.get(
                "comparator") != x["comparator"] or not _is_num(r.get("bound")) or not isinstance(
                r.get("conditions"), dict):
            raise WorldError("world.json: row %s must be {id, quantity, comparator, bound, conditions} as the "
                             "config's pass item" % x["id"])
    der = world["derived"]
    dwant = cfg["derived"]
    if not isinstance(der, list) or len(der) != len(dwant) or not all(
            isinstance(d, dict) and d.get("id") == x["id"] and d.get("quantity") == x["id"]
            and isinstance(d.get("conditions"), dict) for d, x in zip(der, dwant)):
        raise WorldError("world.json: derived must list the %d derived quantities with their conditions"
                         % len(dwant))
    if not isinstance(keyq, dict) or keyq.get("schema") != "rc-key-quantities/1" or not isinstance(
            keyq.get("sketch"), dict) or set(keyq["sketch"]) != {x["id"] for x in cfg["sketch"]}:
        raise WorldError("key_quantities.json must give every sketch quantity")
    for x in cfg["sketch"]:
        v = keyq["sketch"][x["id"]]
        if not _is_num(v) or (x["kind"] == "count" and not _is_int(v)):
            raise WorldError("key_quantities.json: %s must be a finite number (an integer for a count)" % x["id"])
    if not isinstance(ref, dict) or ref.get("schema") != "rc-reference-sheet/1" or not isinstance(
            ref.get("facets"), list) or [f.get("id") if isinstance(f, dict) else None for f in ref["facets"]] != [
            x["id"] for x in cfg["facets"]]:
        raise WorldError("reference_sheet.json must list the %d facets in the config's order" % len(cfg["facets"]))
    for f, x in zip(ref["facets"], cfg["facets"]):
        if not (_is_num(f.get("value")) and f.get("unit") == x["unit"] and _is_num(f.get("tolerance"))
                and f["tolerance"] >= 0 and isinstance(f.get("sources"), list) and f["sources"]
                and all(isinstance(s, str) for s in f["sources"])):
            raise WorldError("reference_sheet.json: facet %s must give value, the contract unit, tolerance >= 0 and "
                             "sources" % x["id"])
    if not isinstance(opt, dict) or opt.get("schema") != "rc-optimum/1" or not _is_num(opt.get("J_usd")) \
            or not isinstance(opt.get("car"), dict) or not isinstance(opt.get("passing_set"), dict):
        raise WorldError("optimum.json is not an rc-optimum/1 object")


def load_world(stg):
    """The gt/ files of the stage, parsed and checked for structure (WorldError)."""
    cfg = stg.cfg
    wd = Path(stg.dir) / cfg["world_dir"]
    f = cfg["world_files"]
    objs = {k: _read_json(wd / f[k], f[k]) for k in WORLD_FILES}
    check_world_objects(cfg, objs["world"], objs["key_quantities"], objs["reference_sheet"], objs["optimum"])
    try:
        cat = stg.P.load_catalogue(objs["catalogue"])
    except Exception as exc:  # noqa: BLE001
        raise WorldError("%s is not a valid catalogue: %s" % (f["catalogue"], _one_line(exc, 200))) from None
    try:
        car = stg.P.load_car(objs["car"])
    except Exception as exc:  # noqa: BLE001
        raise WorldError("%s is not a valid car record: %s" % (f["car"], _one_line(exc, 200))) from None
    if tuple(cat.sizes) != tuple(len(cat.codes[s]) for s in stg.P.SLOT_ORDER):
        raise WorldError("the catalogue's sizes differ from its code lists")
    return World(catalogue=objs["catalogue"], cat=cat, car_obj=objs["car"], car=car, world=objs["world"],
                 key_quantities=objs["key_quantities"]["sketch"], reference=objs["reference_sheet"]["facets"],
                 optimum=objs["optimum"])


# ================================================================================================ scoring pieces
def c1_item_score(kind, nominal, truth, bands):
    """The score of one sketch quantity (C1)."""
    if kind == "count":
        return 1.0 if nominal == truth else 0.0
    err = abs(float(nominal) - float(truth))
    if kind == "angle":
        full, zero = float(bands["angle"]["full_abs"]), float(bands["angle"]["zero_abs"])
    else:
        b = bands["length"]
        t = abs(float(truth))
        full = max(float(b["full_rel"]) * t, float(b["full_abs"]))
        zero = max(float(b["zero_rel"]) * t, float(b["zero_abs"]))
    return ramp((zero - err) / (zero - full))


def c1_score(sketch, truth, cfg):
    """(C1, {id: score}, {group: mean score}): the group means weighted by the config's group weights; a missing
    key scores 0."""
    bands = cfg["c1"]["bands"]
    items, groups = {}, {}
    for x in cfg["sketch"]:
        cell = sketch.get(x["id"]) if isinstance(sketch, dict) else None
        s = 0.0
        if isinstance(cell, dict) and _is_num(cell.get("nominal")):
            s = c1_item_score(x["kind"], cell["nominal"], truth[x["id"]], bands)
        items[x["id"]] = s
        groups.setdefault(x["group"], []).append(s)
    means = {g: math.fsum(v) / len(v) for g, v in groups.items()}
    c1 = math.fsum(float(w) * means[g] for g, w in cfg["c1"]["group_weights"].items())
    return c1, items, means


def facet_counts(sub, ref, facet_rel):
    """True when the submitted facet counts against its reference facet (C2)."""
    v = sub.get("value")
    if not _is_num(v) or sub.get("unit") != ref["unit"]:
        return False
    r, tol = float(ref["value"]), float(ref["tolerance"])
    if not abs(float(v) - r) <= tol + facet_rel * max(1.0, abs(r)):
        return False
    return bool(set(sub.get("sources") or []) & set(ref["sources"]))


def c2_score(facets, reference, facet_rel):
    """(C2, counted, submitted, {id: counted})."""
    by_id = {}
    for f in facets:
        by_id.setdefault(f["id"], f)
    per = {r["id"]: (r["id"] in by_id and facet_counts(by_id[r["id"]], r, facet_rel)) for r in reference}
    counted = sum(1 for v in per.values() if v)
    submitted = len(facets)
    return 2.0 * counted / (len(reference) + submitted), counted, submitted, per


def c3_item(value, unit, truth, truth_unit, band):
    """(score, relative error) of one derived prediction (C3)."""
    if not (_is_num(value) and math.isfinite(float(truth))) or unit != truth_unit:
        return 0.0, math.inf
    t = float(truth)
    e = abs(float(value) - t)
    rel = (e / abs(t)) if t != 0.0 else (0.0 if e == 0.0 else math.inf)
    return ramp((float(band["zero_rel"]) - rel) / (float(band["zero_rel"]) - float(band["full_rel"]))), rel


def c5_score(j_eff, j_star, eco):
    return ramp(1.0 - (float(j_eff) - float(j_star)) / float(eco))


# ================================================================================================ the enumeration
class State(object):
    """The 17 pass items of the hidden world over every catalogue car (procedure.iter_cars order), the prices,
    the passing set and J*; and the scoring of submissions against them."""

    def __init__(self, stg, w, block=None):
        P, cfg = stg.P, stg.cfg
        self.stg, self.P, self.cfg, self.w = stg, P, cfg, w
        self.block = int(block or cfg["block_cars"])
        self.cat, self.car = w.cat, w.car
        self.rows = w.world["rows"]
        self.ids = [r["id"] for r in self.rows]
        self.derived = w.world["derived"]
        self.ctx = {"lot_table": w.world["lot_table"], "cycles": w.world["cycles"], "scenarios": w.world["scenarios"]}
        self.sizes = np.array(self.cat.sizes, dtype=np.int64)
        self.n = int(np.prod(self.sizes, dtype=np.int64))
        if self.n != int(cfg["cars"]) or int(self.cat.n_combinations) != self.n:
            raise WorldError("the catalogue has another number of cars than grader_config.json states")
        self.strides = np.array([int(np.prod(self.sizes[i + 1:], dtype=np.int64)) for i in range(len(self.sizes))],
                                dtype=np.int64)
        try:
            self.prep = P.prepare(self.cat, self.car, self.ctx)
        except Exception as exc:  # noqa: BLE001
            raise WorldError("the real car and the world cannot be prepared: %s" % _one_line(exc, 200)) from None
        self.requests = [(r["quantity"], r["conditions"]) for r in self.rows]
        self.S = np.zeros((len(self.rows), self.n), dtype=bool)
        self.price = np.empty(self.n, dtype=np.float64)
        a = 0
        try:
            for idx in P.iter_cars(self.cat, self.block):
                idx = np.asarray(idx)
                b = a + idx.shape[0]
                if b > self.n or not np.array_equal(idx.astype(np.int64) @ self.strides, np.arange(a, b)):
                    raise WorldError("procedure.iter_cars is not in product order")
                vals = P.evaluate_block(idx, self.cat, self.prep, self.requests)
                for k, (r, v) in enumerate(zip(self.rows, vals)):
                    self.S[k, a:b] = P.sat(v, r["comparator"], r["bound"])
                self.price[a:b] = P.price_block(idx, self.cat)
                a = b
        except WorldError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WorldError("the world rows cannot be evaluated: %s" % _one_line(exc, 200)) from None
        if a != self.n:
            raise WorldError("procedure.iter_cars does not enumerate every car")
        if not np.all(np.isfinite(self.price)):
            raise WorldError("a car price is not finite")
        self.passing = self.S.all(0)
        self.pass_pos = np.flatnonzero(self.passing)
        if self.pass_pos.size == 0:
            raise WorldError("the world has no passing car")
        self.pass_rows = (self.pass_pos[:, None] // self.strides[None, :]) % self.sizes[None, :]
        self.pass_price = self.price[self.pass_pos]
        j = int(np.argmin(self.pass_price))
        self.j_pos = int(self.pass_pos[j])
        self.j_star = float(self.pass_price[j])

    # ---------------------------------------------------------------------------------------- addressing
    def index_of(self, bom):
        """(10,) int64 SKU indices of bom = {slot: SKU}; ValueError when a code is not of its slot."""
        try:
            return np.array([self.cat.codes[s].index(bom[s]) for s in self.P.SLOT_ORDER], dtype=np.int64)
        except (KeyError, ValueError, TypeError):
            raise ValueError("not a car of the catalogue") from None

    def car_at(self, pos):
        row = (int(pos) // self.strides) % self.sizes
        return {s: self.cat.codes[s][int(i)] for s, i in zip(self.P.SLOT_ORDER, row)}

    def passing_set(self):
        cars = [[self.cat.codes[s][int(i)] for s, i in zip(self.P.SLOT_ORDER, row)] for row in self.pass_rows]
        return {"n": len(cars), "sha256": canonical_digest(cars)}

    def optimum(self):
        return {"schema": "rc-optimum/1", "J_usd": self.j_star, "car": self.car_at(self.j_pos),
                "passing_set": self.passing_set()}

    def check_stored(self, opt):
        tol = self.cfg["tolerances"]
        if opt.get("car") != self.car_at(self.j_pos):
            raise WorldError("optimum.json: the optimum car differs from its recomputation")
        if not math.isclose(float(opt["J_usd"]), self.j_star, rel_tol=float(tol["j_usd_rel"]),
                            abs_tol=float(tol["j_usd_abs"])):
            raise WorldError("optimum.json: the optimum price differs from its recomputation")
        if opt.get("passing_set") != self.passing_set():
            raise WorldError("optimum.json: the passing set differs from its recomputation")
        return True

    # ---------------------------------------------------------------------------------------- one car
    def car_values(self, idx):
        """(row sat (17,), derived values (7,)) of one car evaluated on its own; WorldError when its row results
        differ from the enumeration's (the procedure's block result must equal its single-car result)."""
        P = self.P
        req = self.requests + [(d["quantity"], d["conditions"]) for d in self.derived]
        try:
            vals = P.evaluate_block(np.asarray(idx, dtype=np.int64)[None, :].astype(np.int16), self.cat, self.prep,
                                    req)
        except Exception as exc:  # noqa: BLE001
            raise WorldError("a car cannot be evaluated: %s" % _one_line(exc, 200)) from None
        k = len(self.rows)
        sat = np.array([bool(P.sat(v, r["comparator"], r["bound"])[0]) for r, v in zip(self.rows, vals[:k])])
        pos = int(np.asarray(idx, dtype=np.int64) @ self.strides)
        if not np.array_equal(sat, self.S[:, pos]):
            raise WorldError("the procedure's single-car result differs from its block result")
        return sat, [float(np.asarray(v, dtype=np.float64)[0]) for v in vals[k:]]

    def j_eff(self, idx):
        """(J_eff, lines changed by the repair, position of the repaired car)."""
        ham = (self.pass_rows != np.asarray(idx, dtype=np.int64)[None, :]).sum(1)
        cost = self.pass_price + float(self.cfg["eco_usd"]) * ham
        j = int(np.argmin(cost))
        return float(cost[j]), int(ham[j]), int(self.pass_pos[j])

    # ---------------------------------------------------------------------------------------- scoring
    def empty_metrics(self):
        cfg = self.cfg
        m = {"gate_g0_structure": 0.0}
        m.update({name: 0.0 for _, name in CRITERIA})
        m.update({"c1_group:%s" % g: 0.0 for g in cfg["c1"]["group_weights"]})
        m.update({"sketch_score:%s" % x["id"]: 0.0 for x in cfg["sketch"]})
        m.update({"facets_counted": 0.0, "facets_submitted": 0.0})
        m.update({"facet_counted:%s" % x["id"]: 0.0 for x in cfg["facets"]})
        m.update({"derived_score:%s" % x["id"]: 0.0 for x in cfg["derived"]})
        m.update({"derived_relative_error:%s" % x["id"]: 0.0 for x in cfg["derived"]})
        m.update({"acceptance:%s" % r: 0.0 for r in self.ids})
        m.update({"bom_cost_usd": 0.0, "effective_cost_usd": 0.0, "optimum_cost_usd": self.j_star,
                  "eco_lines": 0.0, "bom_lines_differing_from_optimum": 0.0,
                  "passing_cars": float(self.pass_pos.size), "cars_enumerated": float(self.n)})
        return m

    def score(self, docs):
        """(value, metrics, terms) of documents that passed gate G0 (check_documents)."""
        cfg = self.cfg
        req, bom, sketch = (docs[k] for k in SUBMISSION_FILES)
        try:
            idx = self.index_of(bom)
        except ValueError:
            raise SubmissionError("bom.json: a line names a code that is not an SKU of that line") from None
        c1, c1_items, c1_groups = c1_score(sketch, self.w.key_quantities, cfg)
        c2, counted, submitted, per = c2_score(req["facets"], self.w.reference, float(cfg["tolerances"]["facet_rel"]))
        sat, dvals = self.car_values(idx)
        dscore, derr = {}, {}
        for x, t in zip(cfg["derived"], dvals):
            cell = req["derived"].get(x["id"]) if isinstance(req["derived"], dict) else None
            if isinstance(cell, dict):
                s, e = c3_item(cell.get("value"), cell.get("unit"), t, x["unit"], cfg["c3"])
            else:
                s, e = 0.0, math.inf
            dscore[x["id"]], derr[x["id"]] = s, e
        c3 = math.fsum(dscore.values()) / len(dscore)
        c4 = float(bool(sat.all()))
        je, lines, _ = self.j_eff(idx)
        c5 = c5_score(je, self.j_star, cfg["eco_usd"])
        terms = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
        value = math.fsum(float(cfg["weights"][k]) * terms[k] for k in CRITERION_NAME)
        cap = float(cfg["limits"]["max_error_metric"])
        m = self.empty_metrics()
        m["gate_g0_structure"] = 1.0
        for k, name in CRITERIA:
            m[name] = float(terms[k])
        m.update({"c1_group:%s" % g: float(v) for g, v in c1_groups.items()})
        m.update({"sketch_score:%s" % k: float(v) for k, v in c1_items.items()})
        m.update({"facets_counted": float(counted), "facets_submitted": float(submitted)})
        m.update({"facet_counted:%s" % k: float(v) for k, v in per.items()})
        m.update({"derived_score:%s" % k: float(v) for k, v in dscore.items()})
        m.update({"derived_relative_error:%s" % k: float(min(v, cap)) for k, v in derr.items()})
        m.update({"acceptance:%s" % r: float(s) for r, s in zip(self.ids, sat)})
        pos = int(idx @ self.strides)
        m.update({"bom_cost_usd": float(self.price[pos]), "effective_cost_usd": je, "eco_lines": float(lines),
                  "bom_lines_differing_from_optimum": float(((pos // self.strides) % self.sizes != (
                      self.j_pos // self.strides) % self.sizes).sum())})
        terms.update(value=value, J_eff=je, eco_lines=lines, bom_pos=pos, sat=sat, derived_truth=dvals,
                     derived_score=dscore, derived_error=derr, sketch_scores=c1_items, facets_counted=per)
        return float(value), m, terms


# ================================================================================================ gate G0
def read_submission(stg, folder):
    """{file name: document} of the three submission files through schema_check.py's reading path; SubmissionError
    on the first file that is missing, not a regular file, too large, not UTF-8 or does not parse."""
    docs = {}
    for name in SUBMISSION_FILES:
        try:
            docs[name] = stg.SC.read_document(folder, name)
        except stg.SC.Problem as exc:
            raise SubmissionError(str(exc)) from None
        except Exception as exc:  # noqa: BLE001 (never a task error)
            raise SubmissionError("%s cannot be read (%s)" % (name, type(exc).__name__)) from None
    return docs


def check_documents(stg, cat, docs):
    """Gate G0 on parsed documents: schema_check.py's checks (finite numbers and integer sizes, the schemas, a facet
    id at most once), then every BOM code an SKU of its line in the catalogue. Returns docs; SubmissionError on the
    first file that fails."""
    if not isinstance(docs, dict) or set(docs) != set(SUBMISSION_FILES):
        raise SubmissionError("the submission must consist of %s" % ", ".join(SUBMISSION_FILES))
    for name in SUBMISSION_FILES:
        try:
            probs = stg.SC.check_document(name, docs[name], stg.schemas[name])
        except Exception as exc:  # noqa: BLE001
            probs = ["%s cannot be checked (%s)" % (name, type(exc).__name__)]
        if probs:
            raise SubmissionError("; ".join(probs[:3]))
    bom = docs["bom.json"]
    bad = [s for s in stg.P.SLOT_ORDER if not (isinstance(bom.get(s), str) and bom[s] in cat.codes[s])]
    if bad or set(bom) != set(stg.P.SLOT_ORDER):
        raise SubmissionError("bom.json: not an SKU of its line: %s" % ", ".join(bad or ["(the line set)"]))
    return docs


# ================================================================================================ the whole grade
def grade_stage(stage=None, submission=None, g1=True, block=None, stg=None, state=None):
    """Grade one submission folder against the stage folder's world: (value, metrics, diagnostic). WorldError when
    the grader's files are broken or a stored value differs from its recomputation; this is checked whatever the
    submission is. value = G0 x the weighted sum of C1..C5 (G1 is applied by verify.py)."""
    stg = stg or load_stage(stage)
    if state is None:
        w = load_world(stg)
        state = State(stg, w, block=block)
        state.check_stored(w.optimum)
    docs, diagnostic = None, ""
    if g1 and submission is not None:
        try:
            docs = check_documents(stg, state.cat, read_submission(stg, submission))
        except SubmissionError as exc:
            diagnostic = "G0: " + str(exc)
    value, metrics = 0.0, state.empty_metrics()
    if docs is not None:
        try:
            value, metrics, _ = state.score(docs)
        except SubmissionError as exc:
            diagnostic = "G0: " + str(exc)
            value, metrics = 0.0, state.empty_metrics()
    for k, v in metrics.items():
        if not (isinstance(v, float) and math.isfinite(v)):
            raise WorldError("metric %s is not a finite float" % k)
    limit = int(stg.cfg["limits"]["diagnostic_chars"])
    return float(value), metrics, utf8_text(diagnostic)[:limit]
