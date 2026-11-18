"""The site sets of the task: seeds from labels, the slot-to-type assignment by
candidate number, generation, packages and records.

    python -m site_gen.sets --label hidden --k 1-24 --out <dir> --workers 8
    python -m site_gen.sets --label practice --k 1-8 --out <dir> --workers 8
    python -m site_gen.sets --select-practice <dir>            # t1, t2, t3 from the generated practice candidates

seed_k = int.from_bytes(sha256(b"quadruped_hazardous_plant_inspection/<label>/v1/<k>").digest()[:8], "big") >> 1
(the practice and hidden sets are disjoint by construction: different labels).

Variety (a "should"): candidate k gets the assignment slot i -> type (i + k) mod 8 in the order below, so
six consecutive candidates never repeat a type in a slot; the one infeasible pair (B-low, level) is resolved by
swapping B-low's type with C-sealed's (one candidate in eight).

Practice selection: t1 = the first generated practice candidate, t2 the next, t3 the next one after t2 with
which the three together show every deviation class (the always-present seven and both optional ones, an extra
partition and an extra opening); the candidates skipped for t3 are recorded.

Version 2 (`--version v2`):

    python -m site_gen.sets --version v2 --label hidden --k 1-48 --out <dir> --workers 8
    python -m site_gen.sets --table 1-48                      # the factor table by candidate number (JSON)
    python -m site_gen.sets --select-practice-v2 <dir>         # the practice rule (records.jsonl + reference.jsonl)

seed_k from the label .../<label>/v2/<k>; candidate k's factors (bund class, B-step and B-low variants, the missing
sheet's place) and its type assignment come from site_gen.variants.table_row / assignment_v2 (slot i -> type
(i + k + (k - 1) // 8) mod 8, then the repair pass over the measured exclusion list, config/exclusion_v2.json);
they reach the generator as `ladder.v2` and `ladder.assignment`. The records carry the admission and hollowing columns
of the admission rows (v2_record). `--table 1-48` prints the block for selection.json (the rule, the exclusion list with its rates,
the table).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

TASK = "quadruped_hazardous_plant_inspection"
SLOT_ORDER = ("A", "B-step", "B-kerb", "B-low", "C-sealed", "C-moved", "C-relocated", "C-annex")
TYPE_ORDER = ("gauge_63", "gauge_100", "gauge_160", "level", "valve_lever", "valve_handwheel", "lamps", "thermal")
DEVIATION_CLASSES = ("door_sealed", "instrument_moved", "instrument_relocated", "annex_added", "zoning_outdated",
                     "repeater_dead", "opening_as_built", "opening_added", "partition_added")


def seed_for(label: str, k: int, version: str = "v1") -> int:
    h = hashlib.sha256(("%s/%s/%s/%d" % (TASK, label, version, int(k))).encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def rotation_assignment(k: int, infeasible=(("B-low", "level"),)) -> dict:
    """{slot: type} for candidate k (see the module doc)."""
    a = {s: TYPE_ORDER[(i + int(k)) % len(TYPE_ORDER)] for i, s in enumerate(SLOT_ORDER)}
    for s, t in infeasible:
        if a.get(s) == t:
            other = "C-sealed" if s != "C-sealed" else "C-moved"
            a[s], a[other] = a[other], a[s]
    return a


def variety_table(assignments) -> dict:
    """Over every window of six consecutive assignments: the most times a type repeats in one slot."""
    worst = 0
    for i in range(max(1, len(assignments) - 5)):
        win = assignments[i:i + 6]
        for s in SLOT_ORDER:
            ts = [a[s] for a in win]
            worst = max(worst, max(ts.count(t) for t in set(ts)))
    return {"worst_repeat_in_a_slot_over_6": worst, "ok": worst <= 2}


def candidate_config(label, k, version="v1", cfg_path=None):
    """(seed, assignment, cfg) of candidate k of `label`. Version 2: the seed of label .../v2/<k>, the
    factors and the type assignment by candidate number (site_gen.variants), passed as `ladder.v2`."""
    from site_gen.config import load_config
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "ladder.yaml")
    seed = seed_for(label, k, version)
    if version == "v2":
        from site_gen.variants import assignment_v2, v2_config
        V2 = v2_config(k, label)
        assign = assignment_v2(k, V2["b_low"])
        cfg = load_config(cfg_path, overrides={"ladder": {"assignment": assign, "v2": V2}})
    else:
        assign = rotation_assignment(k)
        cfg = load_config(cfg_path, overrides={"ladder": {"assignment": assign}})
    return seed, assign, cfg


def v2_record(site) -> dict:
    """The columns version 2's admission and hollowing records gain: the table's factors as built, the items
    on the missing sheet, the as-built way of C-sealed, the in-room obstacles."""
    L = site["ladder"]
    ms = site.get("missing_sheet") or {}
    out = {"bund": L.get("bund_class"), "b_step_variant": L.get("b_step_variant"),
           "b_low_variant": L.get("b_low_variant"), "sheet_place": ms.get("place"), "missing_sheet": ms.get("sheet"),
           "sheet_count": ms.get("count"), "sheet_slots": L.get("sheet_slots"),
           "missing_rooms": len(ms.get("rooms") or []), "c_sealed_way": L.get("c_sealed_way"),
           "corridor_sill_kind": next((e["kind"] for e in site["elements"] if e["id"] == L.get("corridor_sill")), None),
           "obstacles": [{"slot": o["slot"], "kind": o["kind"], "room": o["room"], "near_m": o["near_m"],
                          "size_m": o.get("size_m", o.get("clearance_m"))} for o in L.get("v2_obstacles", [])],
           "sheet_plan": L.get("sheet_plan")}
    return out


def _one(args):
    label, k, out, cfg_path, pack_dir = args[:5]
    version = args[5] if len(args) > 5 else "v1"
    from site_gen.ladder import generate
    seed, assign, cfg = candidate_config(label, k, version, cfg_path)
    name = "%s_k%02d" % (label, k)
    d = os.path.join(out, name)
    t = time.time()
    rec = {"label": label, "version": version, "k": int(k), "seed": seed, "name": name, "dir": d,
           "assignment": assign}
    if version == "v2":
        rec["table"] = cfg["ladder"]["v2"]
    try:
        if version == "v2" and len(args) > 6 and args[6]:
            # the rows hold on the dressed scene too (the dressing; the bundle from
            # SITE_GEN_DRESS_BUNDLE), checked inside the generator's sub-seed loop (ladder.DRESSED_CHECK): a sub-seed
            # that the dressing refuses or whose rows do not hold dressed is redrawn, not the whole candidate lost
            import site_gen.ladder as _LD
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
            import check_ladder
            _LD.DRESSED_CHECK = lambda dd: check_ladder.check_site(dd, indep=False, dressed=True)
        site, prior = generate(seed, d, cfg, maps=False)
        rows = site["ladder"]["proofs"]
        rec.update({"generated": True, "attempts": site["stats"]["attempts"],
                    "attempt_log": [r["rule"] for r in site["ladder"]["attempt_log"]],
                    "proofs_ok": all(v["ok"] for v in rows.values() if v.get("required", True)),
                    "failed_rows": [k2 for k2, v in rows.items() if v.get("required", True) and not v["ok"]],
                    "deviations": sorted({x["kind"] for x in site["deviations"]}),
                    "route": {kk: site["route"].get(kk) for kk in ("length_m", "total_time_s", "dose_fraction_of_rating",
                                                                  "n_rooms_visited", "modes_used")},
                    "observable_areas": rows.get("observable_areas", {}).get("value"),
                    "x_gyh_at_1m": site["ladder"]["dose"].get("x_strength_gyh_at_1m"),
                    "s": round(time.time() - t, 1),
                    "timing": {kk: round(float(v), 1) for kk, v in (site.get("_timing") or {}).items()
                               if isinstance(v, (int, float))}})
        if version == "v2":
            rec.update(v2_record(site))
        if len(args) > 6 and args[6]:
            # the dressed check made inside the sub-seed loop (above): its result for the admitted sub-seed
            dr = site.get("_dressed")
            if version == "v2" and isinstance(dr, dict) and "required_ok" in dr:
                rec["dressed"] = {"required_ok": dr["required_ok"], "failed": dr["failed"],
                                  "diff_vs_stored": dr["diff_vs_stored"], "s": dr["s"], "in_loop": True}
                rec["proofs_ok"] = bool(rec["proofs_ok"] and dr["required_ok"] and not dr["diff_vs_stored"])
            else:
                # version 1, or no in-loop result: every row computed again on the dressed model
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
                try:
                    import check_ladder
                    dr = check_ladder.check_site(d, indep=False, dressed=True)
                    rec["dressed"] = {"required_ok": dr["required_ok"], "failed": dr["failed"],
                                      "diff_vs_stored": dr["diff_vs_stored"], "s": dr["s"]}
                    rec["proofs_ok"] = bool(rec["proofs_ok"] and dr["required_ok"] and not dr["diff_vs_stored"])
                except Exception as e:  # noqa: BLE001  (the site was generated; the dressed check could not run)
                    rec["dressed"] = {"error": "%s: %s" % (type(e).__name__, str(e)[:300])}
                    rec["proofs_ok"] = False
        if pack_dir:
            from site_gen.loader import write_package
            man = write_package(d, os.path.join(pack_dir, name + ".tar.xz"), name=name,
                                label="%s/%s/%s/%d" % (TASK, label, version, k), kind=label,
                                generator={"seed": seed, "k": int(k), "assignment": assign,
                                           "config_hash": site.get("config_hash")})
            rec["package"] = {"path": os.path.join(pack_dir, name + ".tar.xz"), **man["archive"]}
    except Exception as e:  # noqa: BLE001
        rec.update({"generated": False, "error": "%s: %s" % (type(e).__name__, str(e)[:300]),
                    "attempt_log": [r.get("rule") for r in getattr(e, "attempt_log", [])],
                    "tb": traceback.format_exc()[-2000:], "s": round(time.time() - t, 1)})
    return rec


def select_practice(records):
    """t1, t2, t3 (see the module doc). records: _one results in k order."""
    ok = [r for r in sorted(records, key=lambda r: r["k"]) if r.get("generated") and r.get("proofs_ok")]
    if len(ok) < 3:
        return None
    t1, t2 = ok[0], ok[1]
    have = set(t1["deviations"]) | set(t2["deviations"])
    skipped = []
    for r in ok[2:]:
        if set(DEVIATION_CLASSES) <= have | set(r["deviations"]):
            return {"t1": t1, "t2": t2, "t3": r, "skipped_for_t3": skipped,
                    "classes": sorted(have | set(r["deviations"]))}
        skipped.append(r["k"])
    return {"t1": t1, "t2": t2, "t3": None, "skipped_for_t3": skipped, "classes": sorted(have)}


def _ks(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="hidden, practice, doctest, or a development label (dev-<lane>, review-...)")
    ap.add_argument("--version", default="v1", choices=("v1", "v2"),
                    help="v2: seeds of label .../v2/<k> and the factors by candidate number (04_v2_spec.md S-1, S-2)")
    ap.add_argument("--k", default="1-6")
    ap.add_argument("--out")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     "config", "ladder.yaml"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-packages", action="store_true")
    ap.add_argument("--dressed-check", action="store_true",
                    help="v2: re-run every row on the dressed scene after generation (record 'dressed'; a required row "
                         "failing or differing there makes proofs_ok false)")
    ap.add_argument("--select-practice", default=None, help="a folder with records.jsonl of practice candidates")
    ap.add_argument("--select-practice-v2", default=None,
                    help="F-1 on a folder with records.jsonl of v2 practice candidates and reference.jsonl (k, "
                         "all_delivered, score) of the frozen reference's runs on them")
    ap.add_argument("--table", default=None, help="print the table by candidate number for these k (JSON) and exit")
    a = ap.parse_args(argv)
    if a.table:
        # the block selection.json records before any seed of the hidden or practice label is generated: the
        # type rule's words, the exclusion list with its measured rates, and the table (factors, repaired assignment)
        from site_gen.variants import selection_tables
        print(json.dumps(selection_tables(_ks(a.table)), indent=1))
        return 0
    if a.select_practice_v2:
        from site_gen.variants import select_practice as select_v2
        recs = [json.loads(x) for x in open(os.path.join(a.select_practice_v2, "records.jsonl"))]
        refp = os.path.join(a.select_practice_v2, "reference.jsonl")
        ref = {int(json.loads(x)["k"]): json.loads(x) for x in open(refp)} if os.path.exists(refp) else {}
        for r in recs:
            r["reference"] = ref.get(int(r["k"]))
        sel = select_v2(recs)
        json.dump(sel, open(os.path.join(a.select_practice_v2, "practice_selection_v2.json"), "w"), indent=1,
                  default=str)
        print(json.dumps({k: (v["name"] if isinstance(v, dict) else v) for k, v in sel.items()}, default=str))
        return 0
    if a.select_practice:
        recs = [json.loads(x) for x in open(os.path.join(a.select_practice, "records.jsonl"))]
        sel = select_practice(recs)
        json.dump(sel, open(os.path.join(a.select_practice, "practice_selection.json"), "w"), indent=1, default=str)
        print(json.dumps({k: (v["name"] if isinstance(v, dict) else v) for k, v in (sel or {}).items()}, default=str))
        return 0
    import re
    if not a.label or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", a.label):
        ap.error("--label must be a lower-case word (hidden, practice, doctest, dev-<lane>, ...)")
    os.makedirs(a.out, exist_ok=True)
    pack = None if a.no_packages else os.path.join(a.out, "packages")
    if pack:
        os.makedirs(pack, exist_ok=True)
    jobs = [(a.label, k, a.out, a.config, pack, a.version, a.dressed_check) for k in _ks(a.k)]
    path = os.path.join(a.out, "records.jsonl")
    with open(path, "a") as fo:
        if a.workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
                for r in pool.imap_unordered(_one, jobs):
                    fo.write(json.dumps(r, default=str) + "\n")
                    fo.flush()
                    print(json.dumps({k: r.get(k) for k in ("k", "generated", "attempts", "proofs_ok", "s", "error")}),
                          flush=True)
        else:
            for j in jobs:
                r = _one(j)
                fo.write(json.dumps(r, default=str) + "\n")
                fo.flush()
                print(json.dumps({k: r.get(k) for k in ("k", "generated", "attempts", "proofs_ok", "s", "error")}),
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
