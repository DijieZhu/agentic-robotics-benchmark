"""The bill-of-materials matcher shared by C4 (bom_buildability) and C5 (bom_procurement). v1.0 of the scoring,
as amended by v1.1 (the `attributes_not` condition of a neutral rule, the fold of
step 4, quantities compared as the order they make, two more writing-form rules of `c4.normalise`).

Pipeline (every step deterministic):

  1. Truth lines. The graded C4 rows of gt/ledger.json (gt/freeze.json, once it exists: exactly its `graded` ids
     when it lists them, less its `ungraded` ids),
     written in the public contract's names: the ledger's category and attribute keys are renamed by the maps of
     grader_config.json (`c4.category_ledger_to_contract`, `c4.discriminating[*].ledger_key`,
     `c4.token_ledger_to_contract`). What identifies a truth line is fixed per category by `c4.recognised_by`:
     `vendor_part_number` (the bought boards: the normalised vendor and part number fall in the row's accepted set),
     `tags` (made parts: the line's paper tags are a non-empty subset of the row's tags) or `attributes` (everything
     else: each discriminating attribute of `c4.discriminating[category]` within its tolerance). Tiers come from the
     ledger (freeze `tiers` override); the matching tolerances from the ledger row's own tolerance field where the
     config names one, else from the config (freeze `c4_tolerances` override).
  2. Pre-processing of the agent's lines. A line whose category is out of scope (`c4.out_of_scope_categories`) or
     that falls on the neutral list (`c4.neutral_rules`) leaves: it is neither rewarded, nor counted in precision, nor
     priced. A neutral rule holds when the line's category is the rule's, every attribute of its `attributes` is
     present and equal after normalisation, and every attribute of its `attributes_not` is absent or, after
     normalisation, not equal to the value given. The rest are merged: lines of the same category whose
     discriminating attributes are identical after normalisation (for the bought boards: the same vendor and part
     number; for made parts: tags that name the same truth part, else the same tag set) become one line with the
     quantities added.
  3. One-to-one assignment (Hungarian, scipy.optimize.linear_sum_assignment) of merged lines to truth lines,
     maximising the tier-weighted line score (right quantity 1.0, wrong quantity 0.4); among equal maxima the earlier
     merged line (by its smallest line_id) and the earlier truth line (by ledger order) are preferred through a
     secondary integer term that can never outweigh one unit of the primary one. A quantity is right when it makes
     the same order as the truth's: `c4.robots` x (`c4.struts_per_robot` x qty_per_strut + qty_per_robot_extra)
     pieces, the count docs/conventions.md states (cost.order_units); the same order written per robot instead of per
     strut is the same quantity.
  4. Fold (docs/conventions.md: "splitting a part over several lines does no harm"). Every merged line that takes
     part but is assigned to no truth line, in merged-line order, is recognised against the assigned truth lines in
     ledger order; at the first one that recognises it, it is folded into the merged line assigned to that truth
     line: line ids and lines appended, quantities added, the fold recorded in `excluded` as (its first line_id,
     "folded"). Line scores are taken on the folded quantities, and C4 and C5 read the folded result. Two candidate
     lines that the same truth line cannot both recognise (another length outside the tolerance, another entry) are
     never folded together: the second stays a line recognised as no part.

Strings (vendor, part number, series, thread, package, tokens) are compared without regard to case, spaces and
hyphens, as docs/conventions.md promises the agent, plus the writing-form rules of `c4.normalise` (legal-form
suffix words of a vendor name; a maker prefix, `series` / `style` words and a trailing pitch, size or position count
around a connector series (`series_tail_regex`, applied repeatedly); `#` / `UNC` and a trailing thread class
(`thread_class_regex`) around a unified thread; package-name aliases). This module carries no truth value and no
price: every value comes from gt/ and grader_config.json.
"""
import math
import re
import unicodedata

from . import cost as K

__all__ = ["norm_text", "Normaliser", "truth_lines", "prepare_lines", "recognises", "match_bom"]

_HYPHENS = "-‐‑‒–—―−﹣－"
_SPACE_HYPHEN = re.compile("[\\s" + re.escape(_HYPHENS) + "]+")
_PARENS = re.compile(r"\([^()]*\)")


def norm_text(s):
    """NFKC, case-folded, without whitespace and hyphens (the comparison docs/conventions.md promises)."""
    return _SPACE_HYPHEN.sub("", unicodedata.normalize("NFKC", str(s)).casefold())


def _finite_number(x):
    return (not isinstance(x, bool)) and isinstance(x, (int, float)) and math.isfinite(float(x))


def _integral(x):
    return _finite_number(x) and float(x).is_integer()


def _count(x):
    """A whole non-negative quantity, or None."""
    if not _integral(x) or float(x) < 0:
        return None
    return int(x)


class Normaliser:
    """The writing-form rules of grader_config.json `c4.normalise`."""

    def __init__(self, ncfg):
        self.vendor_suffix = set(ncfg.get("vendor_suffix_words", []))
        self.vendor_alias = dict(ncfg.get("vendor_aliases", {}))
        self.series_prefix = list(ncfg.get("series_prefix_words", []))
        self.series_suffix = list(ncfg.get("series_suffix_words", []))
        self.thread_prefix = list(ncfg.get("thread_prefixes", []))
        self.thread_suffix = list(ncfg.get("thread_suffixes", []))
        self.package_alias = dict(ncfg.get("package_aliases", {}))
        self.pn_prefix = list(ncfg.get("part_number_strip_prefixes", []))
        tail, cls = ncfg.get("series_tail_regex"), ncfg.get("thread_class_regex")
        self.series_tail = re.compile(tail) if tail else None
        self.thread_class = re.compile(cls) if cls else None

    def vendor(self, s):
        words = [w for w in re.split(r"[^0-9a-z]+", unicodedata.normalize("NFKC", str(s)).casefold()) if w]
        while len(words) > 1 and words[-1] in self.vendor_suffix:
            words.pop()
        joined = "".join(words)
        return self.vendor_alias.get(joined, joined)

    def part_number(self, s):
        t = norm_text(s)
        for p in self.pn_prefix:
            if t.startswith(p) and len(t) > len(p):
                t = t[len(p):]
        return t

    @staticmethod
    def _strip(t, prefixes, suffixes):
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if t.startswith(p) and len(t) > len(p):
                    t, changed = t[len(p):], True
            for q in suffixes:
                if t.endswith(q) and len(t) > len(q):
                    t, changed = t[:-len(q)], True
        return t

    def series(self, s):
        t = self._strip(norm_text(_PARENS.sub("", str(s))), self.series_prefix, self.series_suffix)
        while self.series_tail is not None:          # a trailing pitch, size or position count, then the words again
            u = self.series_tail.sub("", t)
            if u == t or not u:
                break
            t = self._strip(u, self.series_prefix, self.series_suffix)
        return t

    def thread(self, s):
        t = norm_text(s)
        if self.thread_class is not None:            # the tolerance class after UNC / UNF (2B, 1A, ...)
            t = self.thread_class.sub("", t) or t
        return self._strip(t, self.thread_prefix, self.thread_suffix)

    def package(self, s):
        t = norm_text(_PARENS.sub("", str(s)))
        return self.package_alias.get(t, t)

    def value(self, kind, v):
        """The comparable form of an attribute value of `kind` (None when the value cannot be of that kind)."""
        if kind in ("integer", "number"):
            return float(v) if _finite_number(v) else None
        if isinstance(v, bool) or not isinstance(v, (str, int, float)):
            return None
        if kind == "series":
            return self.series(v)
        if kind == "thread":
            return self.thread(v)
        if kind == "package":
            return self.package(v)
        return norm_text(v)                                  # string, token


# ------------------------------------------------------------------------------------------------ truth
def _freeze(gt):
    return gt.get("freeze") or {}


def truth_lines(gt, cfg):
    """The graded C4 truth lines in contract form, in ledger order. Each: {id, category, mode, tier, weight, qty,
    money_row, identities | tags | attrs}."""
    c4 = cfg["c4"]
    nm = Normaliser(c4.get("normalise", {}))
    fz = _freeze(gt)
    graded = set(fz["graded"]) if isinstance(fz.get("graded"), list) else None
    ungraded = set(fz.get("ungraded", []))
    tiers = dict(fz.get("tiers", {}))
    tol_over = dict(fz.get("c4_tolerances", {}))
    dropped = {k: set(v) for k, v in fz.get("c4_dropped_attributes", {}).items()}
    alt_values = fz.get("c4_accepted_values", {})
    extra_pn = {}
    for src in (c4.get("accepted_part_numbers_extra", {}), fz.get("c4_accepted_part_numbers_extra", {})):
        for rid, v in src.items():
            extra_pn.setdefault(rid, []).extend(v["part_numbers"] if isinstance(v, dict) else v)
    cat_map = c4["category_ledger_to_contract"]
    modes = c4.get("recognised_by", {})
    tokmap = c4.get("token_ledger_to_contract", {})
    sellers = c4.get("seller_vendors", {})
    out = []
    for row in gt["ledger"]["rows"]:
        if row.get("component") != c4.get("ledger_component", "C4") or row["id"] in ungraded:
            continue
        if not (row["id"] in graded if graded is not None else row.get("graded")):
            continue
        cat = cat_map[row["category"]]
        mode = modes.get(cat, "attributes")
        tier = tiers.get(row["id"], row["tier"])
        val = row["value"]
        t = {"id": row["id"], "category": cat, "mode": mode, "tier": tier,
             "weight": float(c4["tier_weights"][tier]),
             "qty": (_count(val.get("qty_per_strut", 0)) or 0, _count(val.get("qty_per_robot_extra", 0)) or 0),
             "money_row": row.get("money_row")}
        ma = row.get("match_attributes") or {}
        if mode == "vendor_part_number":
            makers = {nm.vendor(ma["vendor"])}
            pns = {nm.part_number(p) for p in (row.get("accepted_part_numbers") or ma.get("part_numbers") or [])}
            pns |= {nm.part_number(p) for p in extra_pn.get(row["id"], [])}
            ids = [(makers, pns)]
            dist = ma.get("distributor_part_numbers") or []
            site = (row.get("vendor_link_part") or {}).get("site")
            if dist and site in sellers:
                ids.append(({nm.vendor(sellers[site])}, {nm.part_number(p) for p in dist}))
            t["identities"] = ids
        elif mode == "tags":
            t["tags"] = frozenset(str(x).strip().upper() for x in ma["tags_any_of"])
        else:
            attrs = []
            for spec in c4["discriminating"].get(cat, []):
                name = spec["attribute"]
                if name in dropped.get(row["id"], set()) or spec["ledger_key"] not in ma:
                    continue
                raw = ma[spec["ledger_key"]]
                raw = tokmap.get(cat, {}).get(name, {}).get(raw, raw) if isinstance(raw, str) else raw
                kind = spec["compare"]
                accepted = [raw] + list(alt_values.get(row["id"], {}).get(name, []))
                tol = 0.0
                if kind == "number":
                    key = "%s.%s" % (cat, name)
                    if key in tol_over:
                        tol = float(tol_over[key])
                    elif spec.get("tolerance_ledger_key") and spec["tolerance_ledger_key"] in ma:
                        tol = float(ma[spec["tolerance_ledger_key"]])
                    else:
                        tol = float(spec.get("tolerance", 0.0))
                attrs.append({"attribute": name, "compare": kind, "accepted": [nm.value(kind, a) for a in accepted],
                              "tolerance": tol})
            t["attrs"] = attrs
        out.append(t)
    return out


# ------------------------------------------------------------------------------------------------ agent lines
def _neutral(line, rules, nm):
    """The id of the first neutral rule the line falls on, else None. A rule's `attributes` must all be present and
    equal; its `attributes_not` must each be absent or, normalised, unequal to the value given."""
    cat = line.get("category")
    attrs = line.get("attributes") if isinstance(line.get("attributes"), dict) else {}
    for rule in rules:
        cats = rule.get("category_any_of") or [rule.get("category")]
        if cat not in cats:
            continue
        ok = True
        for name, (kind, want) in sorted((rule.get("attributes") or {}).items()):
            got = nm.value(kind, attrs.get(name)) if name in attrs else None
            if got is None or got != nm.value(kind, want):
                ok = False
                break
        if ok:
            for name, (kind, avoid) in sorted((rule.get("attributes_not") or {}).items()):
                got = nm.value(kind, attrs.get(name)) if name in attrs else None
                if got is not None and got == nm.value(kind, avoid):
                    ok = False
                    break
        if ok:
            return rule["id"]
    return None


def _merge_key(line, truth_by_cat, nm, modes):
    """What makes two lines 'the same part' (the same category and the same discriminating
    attributes): the vendor and part number for a category recognised by them; for made parts the truth part that
    owns every tag of the line (tags are disjoint between truth parts), else the tag set itself; for a category with
    truth lines the normalised values of the attributes those truth lines discriminate on; for any other category
    every attribute (with vendor and part number)."""
    cat = line.get("category")
    attrs = line.get("attributes") if isinstance(line.get("attributes"), dict) else {}
    mode = modes.get(cat, "attributes")
    if mode == "vendor_part_number":
        v = line.get("vendor")
        p = line.get("part_number")
        return (cat, "vp", nm.vendor(v) if isinstance(v, str) else None,
                nm.part_number(p) if isinstance(p, str) else None)
    if mode == "tags":
        tags = line.get("tags") if isinstance(line.get("tags"), list) else []
        ts = frozenset(str(x).strip().upper() for x in tags if isinstance(x, str))
        owners = [t["id"] for t in truth_by_cat.get(cat, {}).get("lines", []) if ts and ts <= t["tags"]]
        if len(owners) == 1:
            return (cat, "part", owners[0])
        return (cat, "tags", tuple(sorted(ts)))
    if cat in truth_by_cat:
        names = sorted({a["attribute"]: a["compare"] for t in truth_by_cat[cat]["lines"] for a in t["attrs"]}.items())
        return (cat, "attrs", tuple((n, nm.value(k, attrs[n]) if n in attrs else None) for n, k in names))
    # a category without truth lines: every attribute counts (strings and tokens normalised, numbers as floats)
    items = []
    for n in sorted(attrs):
        v = attrs[n]
        items.append((n, float(v) if _finite_number(v) else (norm_text(v) if isinstance(v, str) else repr(v))))
    v, p = line.get("vendor"), line.get("part_number")
    return (cat, "other", tuple(items), nm.vendor(v) if isinstance(v, str) else None,
            nm.part_number(p) if isinstance(p, str) else None)


def prepare_lines(bom, truth, cfg):
    """(merged participating lines, excluded) for one bom.json document. A merged line: {key, category, line_ids,
    lines (the member dicts, in line_id order), qty (per strut, per robot extra)}; excluded: [(line_id, reason)]."""
    c4 = cfg["c4"]
    nm = Normaliser(c4.get("normalise", {}))
    oos = set(c4.get("out_of_scope_categories", []))
    rules = c4.get("neutral_rules", [])
    truth_by_cat = {}
    for t in truth:
        truth_by_cat.setdefault(t["category"], {"mode": t["mode"], "lines": []})["lines"].append(t)
    raw = bom.get("lines") if isinstance(bom, dict) and isinstance(bom.get("lines"), list) else []
    # defence in depth (G0 has validated the document): a category that is not a string never matches anything
    lines = [ln if isinstance(ln.get("category"), str) else dict(ln, category=None) for ln in raw if isinstance(ln, dict)]
    order = sorted(range(len(lines)), key=lambda i: (str(lines[i].get("line_id", "")), i))
    excluded, groups = [], {}
    for i in order:
        ln = lines[i]
        lid = str(ln.get("line_id", ""))
        if ln.get("category") in oos:
            excluded.append((lid, "out_of_scope"))
            continue
        neu = _neutral(ln, rules, nm)
        if neu:
            excluded.append((lid, "neutral:" + neu))
            continue
        key = _merge_key(ln, truth_by_cat, nm, c4.get("recognised_by", {}))
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"key": key, "category": ln.get("category"), "line_ids": [], "lines": [], "qty": [0, 0]}
        g["line_ids"].append(lid)
        g["lines"].append(ln)
        g["qty"][0] += _count(ln.get("qty_per_strut", 0)) or 0
        g["qty"][1] += _count(ln.get("qty_per_robot_extra", 0)) or 0
    merged = sorted(groups.values(), key=lambda g: (g["line_ids"][0], repr(g["key"])))
    for g in merged:
        g["qty"] = tuple(g["qty"])
    return merged, excluded


# ------------------------------------------------------------------------------------------------ recognition
def recognises(t, g, nm):
    """True when truth line `t` recognises merged line `g` (the category first, then the category's identity)."""
    if g["category"] != t["category"]:
        return False
    if t["mode"] == "vendor_part_number":
        for ln in g["lines"]:
            v, p = ln.get("vendor"), ln.get("part_number")
            if not (isinstance(v, str) and isinstance(p, str)):
                return False
            nv, npn = nm.vendor(v), nm.part_number(p)
            if not any(nv in makers and npn in pns for makers, pns in t["identities"]):
                return False
        return True
    if t["mode"] == "tags":
        tags = set()
        for ln in g["lines"]:
            tl = ln.get("tags") if isinstance(ln.get("tags"), list) else []
            tags |= {str(x).strip().upper() for x in tl if isinstance(x, str)}
        return bool(tags) and tags <= t["tags"]
    for ln in g["lines"]:
        attrs = ln.get("attributes") if isinstance(ln.get("attributes"), dict) else {}
        for a in t["attrs"]:
            if a["attribute"] not in attrs:
                return False
            got = nm.value(a["compare"], attrs[a["attribute"]])
            if got is None:
                return False
            if a["compare"] == "integer":
                ok = float(got).is_integer() and any(w is not None and got == w for w in a["accepted"])
            elif a["compare"] == "number":
                ok = any(w is not None and abs(got - w) <= a["tolerance"] + 1e-9 for w in a["accepted"])
            else:
                ok = any(w is not None and got == w for w in a["accepted"])
            if not ok:
                return False
    return True


# ------------------------------------------------------------------------------------------------ assignment
def match_bom(bom, gt, cfg):
    """Match one bom.json document. Returns {truth, merged, excluded, pairs: {merged index: truth index},
    line_score: {truth index: 1.0 | 0.4}}; `merged` and `pairs` after the fold of step 4."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    c4 = cfg["c4"]
    nm = Normaliser(c4.get("normalise", {}))
    truth = truth_lines(gt, cfg)
    merged, excluded = prepare_lines(bom, truth, cfg)
    right = float(c4["line_scores"]["right_quantity"])
    wrong = float(c4["line_scores"]["wrong_quantity"])
    robots, spr = int(c4["robots"]), int(c4["struts_per_robot"])

    def same_order(qa, qb):
        """The same number of pieces in the order (docs/conventions.md's count), however it is written."""
        return K.order_units(qa[0], qa[1], robots, spr) == K.order_units(qb[0], qb[1], robots, spr)

    n, m = len(merged), len(truth)
    pairs = {}
    if n and m:
        scale = 1000
        # the secondary (tie-break) terms of one assignment sum to less than `big`, so they never outweigh one unit
        # of the primary term; every entry stays an integer below 2**53 (exact in float64)
        big = 10 ** len(str(min(n, m) * (n * (m + 1) + m) + 1))
        cost = np.zeros((n, m), dtype=np.float64)
        allowed = np.zeros((n, m), dtype=bool)
        for i, g in enumerate(merged):
            for j, t in enumerate(truth):
                if not recognises(t, g, nm):
                    continue
                s = right if same_order(g["qty"], t["qty"]) else wrong
                primary = int(round(t["weight"] * s * scale))
                secondary = (n - i) * (m + 1) + (m - j)
                cost[i, j] = -float(primary * big + secondary)
                allowed[i, j] = True
        rows, cols = linear_sum_assignment(cost)
        for i, j in sorted(zip(rows.tolist(), cols.tolist())):
            if allowed[i, j]:
                pairs[i] = j
    # step 4, the fold: an unassigned line recognised as a part that an assigned line holds joins that line
    if pairs:
        holder = {j: i for i, j in pairs.items()}
        into = {}
        for i, g in enumerate(merged):
            if i in pairs:
                continue
            for j in sorted(holder):
                if recognises(truth[j], g, nm):
                    into[i] = holder[j]
                    break
        if into:
            for i in sorted(into):
                h, g = merged[into[i]], merged[i]
                h["line_ids"] = h["line_ids"] + g["line_ids"]
                h["lines"] = h["lines"] + g["lines"]
                h["qty"] = (h["qty"][0] + g["qty"][0], h["qty"][1] + g["qty"][1])
                excluded.append((g["line_ids"][0], "folded"))
            keep = [i for i in range(len(merged)) if i not in into]
            new_index = {old: k for k, old in enumerate(keep)}
            merged = [merged[i] for i in keep]
            pairs = {new_index[i]: j for i, j in pairs.items()}
    line_score = {j: (right if same_order(merged[i]["qty"], truth[j]["qty"]) else wrong)
                  for i, j in sorted(pairs.items())}
    return {"truth": truth, "merged": merged, "excluded": excluded, "pairs": pairs, "line_score": line_score}
