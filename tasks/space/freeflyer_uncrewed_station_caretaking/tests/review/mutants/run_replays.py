"""run_replays.py -- replay the mutant / benign table through the grader (records only, no simulation).

    python -I -B run_replays.py [--json OUT] [--only ID [ID ...]] [--quiet]

For every case of cases.py (and of any cases_*.py next to it): build the base record and the case record
(recordkit.py), grade both with tests/cabincell/grade.py (the same grader_config values, the tier's reference energy
as the E basis) and check
  1. the items that differ from the base (and R) are exactly the named ones plus the listed rule consequences;
  2. every expected value holds (+-1e-9);
  3. the 'keep' items equal the base's;
  4. a benign case loses nothing: no item differs and the score equals the base's.
One line per case, then the totals; exit 0 only when no case fails and every row of the mutant / benign table has a case.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import recordkit as RK  # noqa: E402
import cases as CS  # noqa: E402

TOL = 1e-9
BASES = {"ref": RK.reference, "hold": RK.hold_to_end, "u1": RK.u1_isolation, "nodock": RK.nodock}


def all_cases() -> list:
    out = list(CS.CASES)
    for p in sorted(glob.glob(os.path.join(HERE, "cases_*.py"))):
        name = "replay_" + os.path.basename(p)[:-3]
        spec = importlib.util.spec_from_file_location(name, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out.extend(getattr(mod, "CASES", []))
    return out


def config_for(criterion):
    if not criterion:
        return RK.CONFIG
    vals = dict(RK.CONFIG["values"])
    vals["grader.L.criterion"] = criterion
    return dict(RK.CONFIG, values=vals, sha256=None)


def run_case(c: dict, cache: dict) -> dict:
    cfg = config_for(c.get("criterion"))
    ref_e = RK.REF_ENERGY_WH[c["tier"]]
    key = (c["tier"], c["base"], c.get("criterion"))
    if key not in cache:
        b = BASES[c["base"]](c["tier"], f"base_{c['base']}_{c['tier']}")
        cache[key] = RK.G.grade(b.record(), b.site, ref_e, cfg, RK.RUNTIME)
    base = cache[key]
    s = BASES[c["base"]](c["tier"], c["id"])
    c["build"](s)
    g = RK.G.grade(s.record(), s.site, ref_e, cfg, RK.RUNTIME)
    diff = sorted(k for k in RK.G.ITEMS if abs(g["items"][k] - base["items"][k]) > TOL)
    if abs(g["R"] - base["R"]) > TOL:
        diff.append("R")
    want = sorted(set(c["named"]) | set(c["implied"]))
    checks = dict(diff_is_named_plus_implied=(sorted(diff) == want))
    bad_vals = {}
    for k, v in c["expect"].items():
        got = g["R"] if k == "R" else g["items"][k]
        if abs(got - v) > TOL:
            bad_vals[k] = dict(expected=v, got=got)
    checks["expected_values"] = not bad_vals
    checks["kept"] = all(abs(g["items"][k] - base["items"][k]) <= TOL for k in c["keep"])
    if c["kind"] == "benign":
        checks["benign_no_loss"] = (not diff) and abs(g["score"] - base["score"]) <= TOL
    passed = all(checks.values())
    return dict(id=c["id"], row=c["row"], kind=c["kind"], tier=c["tier"], base=c["base"], criterion=c.get("criterion"), text=c["text"],
                design=CS.DESIGN_ROWS.get(c["row"]) if c["row"] else None, named=list(c["named"]), implied=c["implied"],
                expect=c["expect"], keep=list(c["keep"]), diff=diff, base_score=base["score"], score=g["score"],
                base_items=base["items"], items=g["items"], R=g["R"], checks=checks, bad_values=bad_vals, passed=passed, note=c.get("note", ""))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="replay the mutant / benign cases through the grader")
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    cases = [c for c in all_cases() if not a.only or c["id"] in a.only]
    ids = [c["id"] for c in cases]
    dup = sorted(set(i for i in ids if ids.count(i) > 1))
    cache, rows = {}, []
    for c in cases:
        r = run_case(c, cache)
        rows.append(r)
        if not a.quiet or not r["passed"]:
            lost = ",".join(r["diff"]) or "-"
            why = (" = named [" + ",".join(r["named"]) + "]" + (" + by rule [" + ",".join(sorted(r["implied"])) + "]" if r["implied"] else "")
                   if r["kind"] == "mutant" else "")
            print(f"{'ok  ' if r['passed'] else 'FAIL'} {r['id']:7s} {r['kind']:6s} {(r['row'] or 'extra'):5s} {r['tier']} "
                  f"base {r['base']:6s} {r['base_score']:.4f} -> {r['score']:.4f}  changed [{lost}]{why}  {r['text']}")
            if not r["passed"]:
                print("       checks", r["checks"], "bad", r["bad_values"], "want", sorted(set(r["named"]) | set(r["implied"])))
    covered = set(r["row"] for r in rows if r["row"])
    missing = [k for k in CS.DESIGN_ROWS if k not in covered] if not a.only else []
    n_fail = sum(1 for r in rows if not r["passed"])
    summary = dict(cases=len(rows), passed=len(rows) - n_fail, failed=n_fail,
                   mutants=sum(1 for r in rows if r["kind"] == "mutant"), benign=sum(1 for r in rows if r["kind"] == "benign"),
                   table_rows=len(CS.DESIGN_ROWS), table_rows_missing=missing, duplicate_ids=dup,
                   rows_mutant=sum(1 for k in CS.DESIGN_ROWS if k.startswith("M")), rows_benign=sum(1 for k in CS.DESIGN_ROWS if k.startswith("B")),
                   grader_config_sha256=RK.CONFIG["sha256"], runtime_sha256=RK.RUNTIME["sha256"])
    ok = n_fail == 0 and not missing and not dup
    summary["ok"] = ok
    print(f"replays: {summary['cases']} cases ({summary['mutants']} mutant, {summary['benign']} benign), {n_fail} failing; "
          f"table rows {summary['table_rows']} ({summary['rows_mutant']} mutant, {summary['rows_benign']} benign), missing {missing}; "
          f"{'OK' if ok else 'NOT OK'}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(summary=summary, cases=rows), fh, indent=1, sort_keys=True, default=str)
            fh.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
