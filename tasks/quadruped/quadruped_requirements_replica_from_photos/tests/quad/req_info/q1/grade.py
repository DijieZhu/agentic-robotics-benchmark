"""The criteria of the quadruped requirements task (public code; no defaults: the grader config is an input).

    gt = load_gt(gt_dir)                                  the grader's GT bundle of one instance
    docs, problems = load_submission(sub_dir, roles)      parse + schema (+ the structure rules that need no layer)
    problems = g0_rules(docs, roles, layer, cat)          gate G0
    out = grade(docs, gt, cat, cfg, proc)                 {"criteria", "stats", "diagnostics", "overall", "optimum"}

C1 kinematic replication (lengths, stops, classes), C2 the requirements sheet against the reference sheet (F1 of greedy
matches), C3 the derived items for the agent's own BOM on the real unit, C4 acceptance of the agent's BOM (all 16 items
of B-14), C5 economy (the agent's BOM after the engineering-change repair against the live optimum J*).
"""
import json
import math
import os
import sys

import yaml

from . import layer as L
from . import optimum as OPT
from . import repair as RP
from . import schemas as S
from .procedure import BINDING_KINDS, Procedure, bom_from_json

CRITERIA = ("kinematic_replication", "requirements_sheet", "derived_requirements", "bom_acceptance", "bom_economy")
SUBMISSION_FILES = {"requirements": "requirements.yaml", "bom": "bom.json", "kinematic_sketch": "kinematic_sketch.json"}
GT_FILES = ("quad_model", "layer", "catalogue", "lots", "key_quantities", "reference_sheet", "optimum", "duty_set",
            "derived_for_optimal", "optimal_bom")
COMPARATOR_NORM = {"<=": "<=", "<": "<=", "=<": "<=", ">=": ">=", ">": ">=", "=>": ">=", "==": "==", "=": "=="}
MAX_FILE_BYTES = 2 * 10 ** 6
# A YAML alias repeats a node, so a short file can stand for a document many orders of magnitude larger than itself.
# requirements.yaml is refused when its aliases form a cycle or expand it past MAX_YAML_NODES nodes: twice the byte
# limit, which no file within that limit reaches without aliases (a YAML text holds fewer than two nodes per byte).
# It is refused too when it writes an integer in more characters than Python converts between integers and decimal
# strings (sys.get_int_max_str_digits()): a decimal integer that long does not load anyway, and the other forms YAML
# reads as integers (hexadecimal, octal, binary, base 60) would only build numbers too large to print, base 60 slowly.
MAX_YAML_NODES = 2 * MAX_FILE_BYTES
_YAML_INT_TAGS = frozenset(tag for tag, fn in yaml.SafeLoader.yaml_constructors.items()
                           if fn is yaml.constructor.SafeConstructor.construct_yaml_int)


def band(err, full, zero):
    """1 at or below `full`, 0 at or above `zero`, linear in between; 0 for an error that is not a number (a score is
    never NaN)."""
    err, full, zero = float(err), float(full), float(zero)
    if err <= full:
        return float(1)
    if not err < zero:
        return float(0)
    return (zero - err) / (zero - full)


def _fin(x):
    """A stat value: a float, non-finite values -> -1."""
    v = float(x)
    return v if math.isfinite(v) else float(-1)


def _finite(x):
    """x converts to a finite float (False for NaN, +-inf and an integer too large for a float)."""
    try:
        return math.isfinite(float(x))
    except (OverflowError, TypeError, ValueError):
        return False


def _config(cfg):
    """(grader block, weights) of the whole quad-grader-config/1 or of its grader block with the weights added."""
    if "grader" in cfg:
        return cfg["grader"], cfg["weights"]
    if "weights" not in cfg:
        raise KeyError("weights (pass the whole grader config, or its grader block with the weights)")
    return cfg, cfg["weights"]


# ---------------------------------------------------------------------------------------------- inputs
def load_gt(gt_dir):
    """The GT bundle: {"quad_model", "layer", "catalogue", "lots", "key_quantities", "reference_sheet", "optimum",
    "duty_set", "derived_for_optimal", "optimal_bom", "dynamics_xml", "dir"} (a missing file raises)."""
    gt = {"dir": gt_dir}
    for k in GT_FILES:
        with open(os.path.join(gt_dir, k + ".json")) as fh:
            gt[k] = json.load(fh)
    with open(os.path.join(gt_dir, "quad_dynamics.xml")) as fh:
        gt["dynamics_xml"] = fh.read()
    return gt


def _yaml_children(node):
    if isinstance(node, yaml.SequenceNode):
        return node.value
    if isinstance(node, yaml.MappingNode):
        return [n for pair in node.value for n in pair]
    return []


def _yaml_nodes(root, cap):
    """The node count of the YAML node graph `root` with every alias expanded (a scalar, a sequence and a mapping count
    one wherever they occur, a mapping's keys included; a merge key's mappings count where they are merged), found
    without expanding anything: each distinct node is visited once. ValueError when the aliases form a cycle, the
    count passes `cap` or an integer scalar is longer than sys.get_int_max_str_digits() characters."""
    size, path = {}, set()
    digits = sys.get_int_max_str_digits()
    todo = [(root, False)]
    while todo:
        node, done = todo.pop()
        key = id(node)
        if done:
            n = 1 + sum(size[id(c)] for c in _yaml_children(node))
            if n > cap:
                raise ValueError("its YAML aliases expand it past %d nodes" % cap)
            size[key] = n
            path.discard(key)
        elif key in size:
            continue
        elif key in path:
            raise ValueError("its YAML aliases form a cycle")
        else:
            if isinstance(node, yaml.ScalarNode) and node.tag in _YAML_INT_TAGS and 0 < digits < len(node.value):
                raise ValueError("it writes an integer in more than %d characters" % digits)
            path.add(key)
            todo.append((node, True))
            todo.extend((c, False) for c in _yaml_children(node))
    return size[id(root)]


def _load_yaml(text):
    """yaml.safe_load(text), with two bounds: the text is composed into its node graph, which is refused (ValueError)
    when its aliases form a cycle or expand it past MAX_YAML_NODES nodes, or when it writes an integer in more
    characters than Python converts, and only then turned into Python objects. Any other text loads exactly as
    yaml.safe_load loads it."""
    loader = yaml.SafeLoader(text)
    try:
        root = loader.get_single_node()
        if root is None:
            return None
        _yaml_nodes(root, MAX_YAML_NODES)
        return loader.construct_document(root)
    finally:
        loader.dispose()


def load_submission(sub_dir, quote_roles):
    """(docs, problems): each file regular (no symlink), at most 2 MB, parsed (YAML with _load_yaml, JSON), a mapping,
    valid against its schema (first 8 errors per file; a document the schema check cannot even report on is a problem
    too); then the G0 rules that need no layer (quote roles, structure, sketch)."""
    docs, problems = {}, []
    for key, name in SUBMISSION_FILES.items():
        path = os.path.join(sub_dir, name)
        if os.path.islink(path) or not os.path.isfile(path):
            problems.append("%s missing (or not a regular file)" % name)
            continue
        if os.path.getsize(path) > MAX_FILE_BYTES:
            problems.append("%s is larger than %d bytes" % (name, MAX_FILE_BYTES))
            continue
        try:
            with open(path) as fh:
                txt = fh.read()
            doc = _load_yaml(txt) if name.endswith(".yaml") else json.loads(txt)
        except Exception as e:  # noqa: BLE001 (any parse error is a G0 problem)
            problems.append("%s does not parse: %s" % (name, " ".join(str(e).split())))
            continue
        if not isinstance(doc, dict):
            problems.append("%s is not a mapping" % name)
            continue
        try:
            errs = S.validate(doc, key)
        except Exception as e:  # noqa: BLE001 (e.g. a YAML integer too long to print in a schema message)
            problems.append("%s: the schema check cannot report on it: %s" % (name, " ".join(str(e).split())))
            continue
        if errs:
            problems += ["%s: %s" % (name, e) for e in errs[:8]]
            continue
        docs[key] = doc
    if problems:
        return docs, problems
    return docs, _g0(docs, quote_roles, None, None)


def _g0(docs, quote_roles, families, cat):
    p = []
    req = docs["requirements"]["requirements"]
    ids = [r["id"] for r in req]
    if len(set(ids)) != len(ids):
        p.append("requirement ids are not unique")
    for r in req:
        unit = S.UNIT_TO_SI[r["unit"]] if r["unit"] in S.UNIT_TO_SI else None
        if unit is None or unit[0] != S.QUANTITY_SI_UNIT[r["quantity"]]:
            p.append("%s: unit %r is not a unit of %s" % (r["id"], r["unit"], r["quantity"]))
        if not _finite(r["bound"]):
            p.append("%s: bound not finite" % r["id"])
    bom = docs["bom"]
    acts, roles = {}, {"battery": [], "driver": [], "hub": []}
    for line in bom["parts"]:
        pid = line["part_id"]
        if pid not in quote_roles:
            p.append("BOM part %r is not on the quote" % pid)
        elif quote_roles[pid] != line["role"]:
            p.append("BOM part %r has role %r, the quote says %r" % (pid, line["role"], quote_roles[pid]))
        if line["role"] == "actuator":
            if "joint_type" not in line:
                p.append("actuator line %r has no joint_type" % pid)
                continue
            t = line["joint_type"]
            if t in acts:
                p.append("two actuator lines for %s" % t)
            acts[t] = line
            if int(line["count"]) != len(S.LEGS):
                p.append("the actuator line of %s has count %s (one module per leg)" % (t, line["count"]))
        else:
            roles[line["role"]].append(line)
    for t in S.JOINT_TYPES:
        if t not in acts:
            p.append("no actuator line for %s" % t)
    if len(bom["parts"]) != len(S.LINES):
        p.append("the BOM has %d lines, it needs exactly %d" % (len(bom["parts"]), len(S.LINES)))
    for role, lines in roles.items():
        if len(lines) != 1:
            p.append("the BOM needs exactly one %s line, it has %d" % (role, len(lines)))
    for role in ("battery", "hub"):
        if len(roles[role]) == 1 and int(roles[role][0]["count"]) != 1:
            p.append("the %s line must have count 1" % role)
    if families is not None and bom["battery_family"] not in families:
        p.append("battery_family %r is not one of %s" % (bom["battery_family"], list(families)))
    if cat is not None and len(roles["driver"]) == 1:
        d = roles["driver"][0]
        if cat.has(d["part_id"]) and cat.role(d["part_id"]) == "driver":
            need = cat.driver_count(d["part_id"])
            if int(d["count"]) != need:
                p.append("the driver line has count %s, %s needs %d" % (d["count"], d["part_id"], need))
    sk = docs["kinematic_sketch"]
    for t in S.JOINT_TYPES:
        if t in acts and sk["modules"][t] != acts[t]["part_id"]:
            p.append("sketch module of %s (%s) differs from the BOM line (%s)" % (t, sk["modules"][t],
                                                                                   acts[t]["part_id"]))
    for k in S.GEOMETRY_KEYS:
        g = sk["geometry"][k]
        if not (_finite(g["nominal"]) and _finite(g["tol"])):
            p.append("geometry %s not finite" % k)
    for t in S.JOINT_TYPES:
        if not all(_finite(v) for v in sk["stops"][t]):
            p.append("stops of %s not finite" % t)
    legs = sk["spacer_legs"]
    if len(set(legs)) != len(legs) or any(leg not in S.LEGS for leg in legs):
        p.append("spacer_legs must be distinct leg names")
    return p


def g0_rules(docs, quote_roles, layer, cat=None):
    """Gate G0: requirement ids and units, quote roles, the six BOM lines, the family, the driver count
    (with a catalogue) and the sketch."""
    return _g0(docs, quote_roles, L.policy(layer)["battery_families"], cat)


# ---------------------------------------------------------------------------------------------- C1
def c1_kinematic(sketch, kq, cfg):
    """(score, groups, items) of C1 against the C1 truth (the "c1" block of key_quantities)."""
    c = cfg["c1"]
    truth = kq["c1"] if "c1" in kq else kq
    ad = truth["appendix_d"]
    mm = S.UNIT_TO_SI["mm"][1]
    items, lengths, cover, tols = {}, [], [], []
    for k in S.GEOMETRY_KEYS:
        g = float(ad[k])
        nom = float(sketch["geometry"][k]["nominal"])
        tol = float(sketch["geometry"][k]["tol"])
        err = abs(nom - g)
        full = max(float(c["length_full"]["abs_m"]), float(c["length_full"]["rel"]) * abs(g))
        zero = max(float(c["length_zero"]["abs_m"]), float(c["length_zero"]["rel"]) * abs(g))
        lengths.append(band(err, full, zero))
        items["len_err_mm:" + k] = err / mm
        cover.append(float(1) if err <= tol else float(0))
        tols.append(tol / mm)
    stops = []
    for t in S.JOINT_TYPES:
        a = sorted(float(v) for v in sketch["stops"][t])
        g = sorted(float(v) for v in truth["stops"][t])
        for side, ai, gi in (("lower", a[0], g[0]), ("upper", a[1], g[1])):
            err = abs(math.degrees(ai - gi))
            items["stop_err_deg:%s:%s" % (t, side)] = err
            stops.append(band(err, *c["stop_deg"]))
    classes = []
    for t in S.JOINT_TYPES:
        ok = sketch["housing_class"][t] == truth["housing_class"][t]
        items["class_ok:housing_class:" + t] = float(ok)
        classes.append(float(ok))
    ok = set(sketch["spacer_legs"]) == set(truth["spacer_legs"])
    items["class_ok:spacer_legs"] = float(ok)
    classes.append(float(ok))
    ok = sketch["pack_bay"] == truth["pack_bay"]
    items["class_ok:pack_bay"] = float(ok)
    classes.append(float(ok))
    groups = {"lengths": sum(lengths) / len(lengths), "stops": sum(stops) / len(stops),
              "classes": sum(classes) / len(classes)}
    w = c["group_weights"]
    score = sum(float(w[k]) * groups[k] for k in groups) / sum(float(w[k]) for k in groups)
    items["tol_coverage"] = sum(cover) / len(cover)
    items["tol_mean_mm"] = sum(tols) / len(tols)
    return score, groups, items


# ---------------------------------------------------------------------------------------------- C2
def c2_sheet(items, rstar, cfg):
    """(F1, report): greedy matching in R* order of the agent's non-derived items on quantity, normalised comparator and
    a converting unit; each match scores the band of its R* item (x clause_weight off the R* item's sources)."""
    c = cfg["c2"]
    agent = [r for r in items if str(r["source_id"]) != "derived" and r["quantity"] not in S.DERIVED_QUANTITIES]
    ref = rstar["items"]
    used = set()
    total = 0
    rep = {"matched": [], "partial": [], "missing": [], "extra": [], "clause_mismatch": []}
    for rf in ref:
        full, zero = c[rf["band"] + "_rel"]
        best, best_i, best_clause = 0, None, True
        for i, it in enumerate(agent):
            if i in used or it["quantity"] != rf["quantity"]:
                continue
            if COMPARATOR_NORM[str(it["comparator"]).strip()] != COMPARATOR_NORM[str(rf["comparator"]).strip()]:
                continue
            conv = S.to_si(it["bound"], it["unit"])
            if conv is None or conv[1] != S.QUANTITY_SI_UNIT[rf["quantity"]]:
                continue
            err = abs(conv[0] - float(rf["bound"])) / max(abs(float(rf["bound"])), 1e-9)
            w = band(err, full, zero)
            clause_ok = str(it["source_id"]) in rf["source_ids"]
            if not clause_ok:
                w *= float(c["clause_weight"])
            if w > best:
                best, best_i, best_clause = w, i, clause_ok
        if best_i is not None:
            used.add(best_i)
        total += best
        if best_i is None or best <= 0:
            rep["missing"].append(rf["id"])
        elif best >= 1 - 1e-9:
            rep["matched"].append(rf["id"])
        else:
            rep["partial"].append(rf["id"])
        if best_i is not None and not best_clause:
            rep["clause_mismatch"].append(rf["id"])
    rep["extra"] = [str(agent[i]["id"]) for i in range(len(agent)) if i not in used]
    if not agent or not ref or total <= 0:
        return float(0), rep
    return 2 * total / (len(agent) + len(ref)), rep


# ---------------------------------------------------------------------------------------------- C3
def c3_derived(items, gder, cfg):
    """(score, errs, signed): the first item per derived quantity (file order) against the truth for the agent's BOM;
    e = |v - g| / max(|g|, floor of the SI unit), band(e, rel); a missing or non-converting item scores 0."""
    c = cfg["c3"]
    full, zero = c["rel"]
    by_q = {}
    for it in items:
        if it["quantity"] in S.DERIVED_QUANTITIES and it["quantity"] not in by_q:
            by_q[it["quantity"]] = it
    scores, errs, signed = [], {}, {}
    for q in S.DERIVED_QUANTITIES:
        g = float(gder[q])
        unit = S.QUANTITY_SI_UNIT[q]
        floor = float(c["floors"][unit])
        it = by_q[q] if q in by_q else None
        conv = None if it is None else S.to_si(it["bound"], it["unit"])
        if conv is None or conv[1] != unit:
            scores.append(float(0))
            errs[q] = None
            signed[q] = None
            continue
        d = (conv[0] - g) / max(abs(g), floor)
        errs[q] = abs(d)
        signed[q] = d
        scores.append(band(abs(d), full, zero))
    return sum(scores) / len(scores), errs, signed


# ---------------------------------------------------------------------------------------------- the whole grade
def _bom_parts(bom):
    return [bom["modules"][t] for t in S.JOINT_TYPES] + [bom["battery"], bom["driver"], bom["hub"]]


def _same_bom(a, b):
    keys = ("battery_family", "battery", "driver", "driver_count", "hub")
    return all(a[k] == b[k] for k in keys) and all(a["modules"][t] == b["modules"][t] for t in S.JOINT_TYPES)


def _cut(text, n):
    """text cut to at most n characters (an ellipsis marks a cut when there is room for it)."""
    text = str(text)
    if len(text) <= n:
        return text
    return text[:n - 3] + "..." if n >= 3 else text[:max(0, n)]


def grade(docs, gt, cat, cfg, proc=None):
    """{"criteria": {criterion: score}, "stats": {name: float}, "diagnostics": {criterion: str}, "overall": float,
    "optimum": the live optimum record | None}. `cfg` is the grader config (the whole quad-grader-config/1 or its grader
    block with the weights); there is no default config."""
    g, weights = _config(cfg)
    per = int(g["diagnostics"]["per_criterion_chars"])
    layer = gt["layer"]
    out = {"criteria": {c: float(0) for c in CRITERIA}, "stats": {}, "diagnostics": {c: "" for c in CRITERIA},
           "overall": float(0), "optimum": None}
    st = out["stats"]
    roles = cat.quote_roles()
    probs = ["%s missing" % SUBMISSION_FILES[k] for k in SUBMISSION_FILES if k not in docs]
    if not probs:
        for k, name in SUBMISSION_FILES.items():
            probs += ["%s: %s" % (name, e) for e in S.validate(docs[k], k)[:8]]
    if not probs:
        probs = g0_rules(docs, roles, layer, cat)
    st["gate_g0_structure"] = float(0) if probs else float(1)
    st["g0_problems"] = float(len(probs))
    if probs:
        out["diagnostics"][CRITERIA[0]] = _cut("G0 structure: " + "; ".join(probs[:6]), per)
        for c in CRITERIA[1:]:
            out["diagnostics"][c] = "gated by G0"
        out["stats"] = {k: _fin(v) for k, v in st.items()}
        return out
    crit = out["criteria"]
    diag = out["diagnostics"]
    # C1
    c1, groups, c1i = c1_kinematic(docs["kinematic_sketch"], gt["key_quantities"], g)
    crit["kinematic_replication"] = c1
    st.update({"c1_" + k: v for k, v in groups.items()})
    st.update(c1i)
    worst_len = max(S.GEOMETRY_KEYS, key=lambda k: c1i["len_err_mm:" + k])
    diag["kinematic_replication"] = "lengths %.3f, stops %.3f, classes %.3f; largest length error %s %.1f mm" % (
        groups["lengths"], groups["stops"], groups["classes"], worst_len, c1i["len_err_mm:" + worst_len])
    # C2
    items = docs["requirements"]["requirements"]
    c2, rep = c2_sheet(items, gt["reference_sheet"], g)
    crit["requirements_sheet"] = c2
    st.update({"c2_matched": len(rep["matched"]), "c2_partial": len(rep["partial"]), "c2_missing": len(rep["missing"]),
               "c2_extra": len(rep["extra"]), "c2_clause_mismatch": len(rep["clause_mismatch"])})
    diag["requirements_sheet"] = "missing %s; partial %s; extra %s" % (rep["missing"], rep["partial"], rep["extra"])
    # the BOM criteria
    bom = bom_from_json(docs["bom"])
    absent = sorted({p for p in _bom_parts(bom) if not cat.has(p)})
    if absent:
        st["catalogue_missing_parts"] = float(len(absent))
        diag["bom_acceptance"] = "the catalogue lacks %s: C3-C5 are 0" % absent
    else:
        if proc is None:
            proc = Procedure(gt["quad_model"], layer)
        lots = gt["lots"]
        # C3
        gder = proc.derived(bom, cat, lots)
        c3, errs, signed = c3_derived(items, gder, g)
        crit["derived_requirements"] = c3
        for q in S.DERIVED_QUANTITIES:
            st["c3_relerr:" + q] = float(-1) if errs[q] is None else errs[q]
            st["c3_signed:" + q] = float(0) if signed[q] is None else signed[q]
            st["c3_missing:" + q] = float(1) if errs[q] is None else float(0)
        worst = sorted((q for q in S.DERIVED_QUANTITIES if errs[q] is not None), key=lambda q: -errs[q])
        miss = [q for q in S.DERIVED_QUANTITIES if errs[q] is None]
        diag["derived_requirements"] = "missing %s; largest errors %s" % (
            miss, ", ".join("%s %+.3f" % (q, signed[q]) for q in worst[:4]))
        # C4
        ev = proc.evaluate(bom, cat, lots)
        all_pass = all(ev["items"].values())
        mode = g["c4"]["mode"]
        if mode == "all":
            crit["bom_acceptance"] = float(1) if all_pass else float(0)
        elif mode == "fraction":
            crit["bom_acceptance"] = float(ev["pass_fraction"])
        else:
            raise KeyError("c4 mode %r" % mode)
        st["c4_item_fraction"] = ev["pass_fraction"]
        for i in S.ITEMS:
            st["c4_pass:" + i] = float(ev["items"][i])
            st["util:" + i] = ev["util"][i]["worst"]
        for j in S.JOINTS:
            st["util_kind:" + j] = float(BINDING_KINDS.index(ev["util"][j]["kind"]))
        st["bus_peak_a"] = ev["currents"]["bus_peak_a"]
        st["charge_ah"] = ev["currents"]["charge_ah"]
        st["bom_cost_usd"] = ev["cost"]
        failing = [i for i in S.ITEMS if not ev["items"][i]]
        diag["bom_acceptance"] = ("all 16 items pass" if not failing else
                                  "failing: " + ", ".join("%s (u %.3f)" % (i, ev["util"][i]["worst"]) for i in failing))
        # C5
        best = OPT.optimal_bom(proc, cat, lots, g["optimum"]["node_cap"])
        stored = gt["optimum"]
        if best is None or not _same_bom(best["bom"], bom_from_json(stored["bom"])) \
                or abs(best["cost"] - float(stored["cost"])) > 1e-9:
            raise RuntimeError("the live optimum %s differs from the stored optimum %s (task error)"
                               % (None if best is None else best["cost"], stored["cost"]))
        out["optimum"] = best
        J_star = best["cost"]
        pol = L.policy(layer)
        rp = RP.repair(proc, bom, cat, lots, pol["eco_usd"], g["repair"]["max_steps"], g["repair"]["order"])
        ceiling = float(pol["ceiling_usd"])
        c5 = RP.economy(J_star, rp["J_eff"], ceiling, g["c5"]["band_usd"], g["c5"]["taper_frac"])
        crit["bom_economy"] = c5
        finite = math.isfinite(rp["J_eff"])
        st.update({"J": rp["J"], "J_eff": rp["J_eff"] if finite else float(-1), "J_star": J_star,
                   "repairs": rp["replaced"], "repairable": float(rp["repairable"]), "ceiling_usd": ceiling,
                   "c5_over_usd": (rp["J_eff"] - J_star) if finite else float(-1),
                   "optimum_beaten": float(finite and rp["J_eff"] < J_star - 1e-9 and all_pass),
                   "n_leaves": best["n_leaves"], "knife_edge_min": best["knife_edge_min"]})
        diag["bom_economy"] = "J %.2f, J_eff %s, J* %.2f; repairs: %s" % (
            rp["J"], ("%.2f" % rp["J_eff"]) if finite else "inf", J_star, "; ".join(rp["log"]) or "none")
    out["diagnostics"] = {c: _cut(diag[c], per) for c in CRITERIA}
    out["stats"] = {k: _fin(v) for k, v in st.items()}
    out["overall"] = sum(float(weights[c]) * crit[c] for c in CRITERIA) / sum(float(weights[c]) for c in CRITERIA)
    return out
