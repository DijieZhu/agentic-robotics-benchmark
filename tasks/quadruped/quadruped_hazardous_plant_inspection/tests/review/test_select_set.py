"""Unit tests of the version 2 selection tool (tests/review/select_set.py): the parts that need no
simulator. stdlib only.

    python3 -B -I tests/review/test_select_set.py

Checks: the seed of a label (the seed formula, the hidden / practice / development labels); version 2's JSON-lines
records (also one line alone) and version 1's forms; the candidate's package by any `<label>_kNN.tar.xz` name; row 1
from a version 2 record (failed rows, a recheck where there is one, a record for other bytes); row 2's plate
condition from the reference's own log (read as the tag no later than the report; a read after the report, a wrong
text and a missing read all fail); rows 2 and 3 together; the hollowing features with the table's columns, the
missing sheet's count and C-sealed's way. Fails on version 1's select_set.py (no labels, no plate condition, no
JSON lines); passes after.

Version 2.0: the quota rule on synthetic admission records (`quota_select`): a candidate skipped by
each quota, a set short of six, no tie (the order is k: the input's order does not matter, of two alike the lower k is
taken), an undecided candidate stops the rule only before the set is complete, and on 400 random record sets a complete
set always holds 3 / 3 of each kerb class and of each B variant and 2 / 2 / 2 sheet places, every skip is a real quota
overrun at its turn and no admitted candidate before the sixth member is left out without one; `rule` writes the
exclusion list with its sha256, the quota text and the note on row 3 and refuses another list, a tables block whose list
differs, a hidden label without tables and a table that does not cover the k range; `decide` end to end on a synthetic
selection folder (frozen files, proofs, oracle and baseline records of 20 candidates): the set by the quota rule, every
candidate's status, the counts, and exit 1 with fewer than six. Fails on an earlier select_set.py (no quota_select;
`decide` takes the first six admitted); passes after.

Version 2.1: the set is twelve and every quota is doubled (at most six of each kerb
class, B-step and B-low variant, at most four of each sheet place), so the cases above read "seventh" and "fifth",
"twelve", 6 / 6 and 4 / 4 / 4; row 3's time part is the absolute 1200 s (a reference at 1146 s passes under a 1500 s
limit although that is 0.76 of it; 1200.5 s fails under any limit), and `rule` writes version 2.1's note and row 3's words.
Fails on version 2.0's select_set.py (six, 3 / 3 / 3 / 2, row 3 as 50 % of the limit); passes after.
"""

import hashlib
import json
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import select_set as S  # noqa: E402


def check(cond, label, out):
    out.append(("ok   " if cond else "FAIL ") + label)


def run():
    out = []
    want = int.from_bytes(hashlib.sha256(b"quadruped_hazardous_plant_inspection/hidden/v2/7").digest()[:8], "big") >> 1
    check(S.seed_k(7) == want and S.seed_k(7, "hidden") == want, "seed_k(7) is S-1's hidden/v2/7", out)
    check(S.seed_k(1, "dev-sitegen2") == 3949738296678582259, "a development label's seed (dev-sitegen2 k 1)", out)
    check(S.seed_k(3, "practice") != S.seed_k(3, "hidden"), "practice and hidden seeds differ", out)
    with tempfile.TemporaryDirectory() as d:
        recs = [{"k": 1, "seed": S.seed_k(1, "x"), "generated": True, "proofs_ok": True, "failed_rows": [],
                 "package": {"sha256": "a" * 64}, "bund": "low", "b_step_variant": "door_sills",
                 "b_low_variant": "chase", "sheet_place": "along", "sheet_count": 2, "sheet_slots": ["B-kerb"],
                 "c_sealed_way": "chase", "table": {"bund": "low", "b_step": "door_sills", "b_low": "chase",
                                                    "sheet_place": "along"}, "deviations": ["door_sealed"]},
                {"k": 2, "generated": True, "proofs_ok": True, "failed_rows": ["tags"], "package": {"sha256": "b" * 64}}]
        p = os.path.join(d, "r.jsonl")
        with open(p, "w") as f:
            f.write("\n".join(json.dumps(r) for r in recs) + "\n")
        pr = S.load_proofs(p)
        check(sorted(pr) == [1, 2] and pr[1]["bund"] == "low", "JSON-lines records read", out)
        p1 = os.path.join(d, "one.jsonl")
        with open(p1, "w") as f:
            f.write(json.dumps(recs[0]) + "\n")
        check(sorted(S.load_proofs(p1)) == [1], "one record alone read as JSON lines", out)
        p2 = os.path.join(d, "v1.json")
        with open(p2, "w") as f:
            json.dump({"rows": [dict(recs[0], k=4)]}, f)
        check(sorted(S.load_proofs(p2)) == [4], "version 1's candidates.json rows read", out)
        for name in ("dev-x_k03.tar.xz", "hidden_k05.tar.xz", "c06.tar.xz"):
            open(os.path.join(d, name), "w").close()
        check(S.candidate_path(d, 3).endswith("dev-x_k03.tar.xz") and S.candidate_path(d, 5).endswith(
            "hidden_k05.tar.xz") and S.candidate_path(d, 6).endswith("c06.tar.xz") and S.candidate_path(d, 9) is None,
            "candidate packages by name", out)
        r1 = S.row1_of(pr[1], {"sha256": "a" * 64})
        check(r1["ok"] and r1["recheck_from_files_ok"] is None, "row 1: a clean version 2 record", out)
        check(not S.row1_of(pr[2], {"sha256": "b" * 64})["ok"], "row 1: a failed row fails", out)
        check(not S.row1_of(pr[1], {"sha256": "c" * 64})["ok"], "row 1: a record for other bytes fails", out)
        check(not S.row1_of(dict(pr[1], recheck_from_files={"required_ok": False}), {"sha256": "a" * 64})["ok"],
              "row 1: a failed recheck fails", out)
        # row 2's plate condition from the reference's log
        items = [{"tag": "PI-%d" % i, "type": "gauge", "delivered": True, "evidence_ok": True} for i in range(8)] + \
                [{"tag": "SM-1", "type": "dose_point", "delivered": True, "E": 1}, {"tag": "X", "type": "hotspot",
                                                                                   "delivered": True, "E": 1}]
        log = []
        for i in range(8):
            log.append({"t": 10.0 * i, "kind": "plate_read", "tag": "PI-%d" % i, "ok": True, "text": "PI-%d" % i})
            log.append({"t": 10.0 * i + 1, "kind": "report", "tag": "PI-%d" % i})
        run_dir = os.path.join(d, "a1")
        os.makedirs(run_dir)

        def plates(lg):
            with open(os.path.join(run_dir, "reference_log.json"), "w") as f:
                json.dump({"log": lg}, f)
            return S.plates_before_reports(run_dir, {"items": items})

        pl = plates(log)
        check(pl["read_before_report"] == 8 and pl["instruments"] == 8, "row 2 plates: 8 / 8 read first", out)
        late = [dict(n, t=n["t"] + 5) if n["kind"] == "plate_read" and n["tag"] == "PI-3" else n for n in log]
        check(plates(late)["read_before_report"] == 7, "row 2 plates: a read after its report fails", out)
        wrong = [dict(n, ok=False, text="PI-33") if n["kind"] == "plate_read" and n["tag"] == "PI-5" else n
                 for n in log]
        check(plates(wrong)["read_before_report"] == 7, "row 2 plates: a wrong text fails", out)
        gone = [n for n in log if not (n["kind"] == "plate_read" and n["tag"] == "PI-6")]
        check(plates(gone)["read_before_report"] == 7, "row 2 plates: a missing read fails", out)
        att = {"items": items, "R": 1.0, "episode_score": 1.0, "sim_time_s": 1100.0, "sim_time_limit_s": 2400.0,
               "dose_fraction": 0.3, "frames": {"nav": 30, "inspect": 20, "lidar": 1000}, "plates": plates(log)}
        cfg = {"budgets": {"frames": {"nav": 300, "inspect": 240, "lidar": 4000}}}
        r2, r3 = S.reference_rows({"attempts": [att], "counted_attempt": 1}, cfg)
        check(r2["ok"] and r2["plates_ok"] and r3["ok"], "rows 2 and 3 pass on a clean mission", out)
        r2b, _ = S.reference_rows({"attempts": [dict(att, plates=plates(late))], "counted_attempt": 1}, cfg)
        check(not r2b["ok"] and not r2b["plates_ok"], "row 2 fails when one plate was read after its report", out)
        _, r3b = S.reference_rows({"attempts": [dict(att, sim_time_s=1257.35)], "counted_attempt": 1}, cfg)
        check(not r3b["ok"], "row 3 fails at 1257 s, 0.524 of 2400 s (k02 before R8)", out)
        # row 3's time part is the absolute bound of 1200 s, whatever the mission's limit
        _, r3c = S.reference_rows({"attempts": [dict(att, sim_time_s=1146.0, sim_time_limit_s=1500.0)],
                                   "counted_attempt": 1}, cfg)
        check(r3c["ok"] and r3c["sim_share"] == 0.764 and r3c["sim_time_max_s"] == 1200.0,
              "R26 row 3: 1146 s passes under a 1500 s limit (share 0.764 recorded, bound 1200 s)", out)
        _, r3d = S.reference_rows({"attempts": [dict(att, sim_time_s=1200.0, sim_time_limit_s=1350.0)],
                                   "counted_attempt": 1}, cfg)
        _, r3e = S.reference_rows({"attempts": [dict(att, sim_time_s=1200.5, sim_time_limit_s=2400.0)],
                                   "counted_attempt": 1}, cfg)
        check(r3d["ok"] and not r3e["ok"], "R26 row 3: 1200 s passes (limit 1350 s), 1200.5 s fails (limit 2400 s)",
              out)
        f = S.features_of(pr[1])
        check(f.get("v2.bund") == "low" and f.get("v2.sheet_count") == 2 and f.get("v2.c_sealed_way") == "chase"
              and f.get("table.b_low") == "chase" and f.get("v2.on_missing_sheet=B-kerb") is True,
              "hollowing features: the table's columns, sheet_count, C-sealed", out)
    quota_tests(out)
    rule_tests(out)
    decide_tests(out)
    return out


# ------------------------------------------------------------------------------------------- the quota rule
COLS = [("low", "door_sills", "chase"), ("high", "in_room", "rack"), ("low", "in_room", "chase"),
        ("high", "door_sills", "rack"), ("low", "door_sills", "rack"), ("high", "in_room", "chase"),
        ("low", "in_room", "rack"), ("high", "door_sills", "chase")]
PLACES = ("along", "across", "diagonal")


def table_q(k):
    """The table by candidate number for candidate k: (bund, b_step, b_low) by (k - 1) mod 8, the sheet place by (k - 1) mod 3."""
    b, st, lo = COLS[(k - 1) % 8]
    return {"bund": b, "b_step": st, "b_low": lo, "sheet_place": PLACES[(k - 1) % 3]}


def counts_of(qs):
    out = {q: {} for q in S.QUOTA}
    for v in qs:
        for q in S.QUOTA:
            out[q][str(v[q])] = out[q].get(str(v[q]), 0) + 1
    return out


def mk(k, b, st, lo, pl):
    return {"k": k, "admitted": True, "quota": {"bund": b, "b_step": st, "b_low": lo, "sheet_place": pl}}


def balanced12(c):
    """A complete set's counts, 6 / 6 of each kerb class and B variant, 4 / 4 / 4 sheet places."""
    return (c["bund"] == {"low": 6, "high": 6} and c["b_step"] == {"door_sills": 6, "in_room": 6}
            and c["b_low"] == {"chase": 6, "rack": 6} and c["sheet_place"] == {"along": 4, "across": 4, "diagonal": 4})


def seq(n, fixed):
    """n admitted candidates k = 1..n that share the values of `fixed` and vary the other quota columns (alternating
    the two-valued ones, cycling the sheet place), so that only the fixed column can reach its quota."""
    out = []
    for i in range(n):
        v = {"bund": ("low", "high")[i % 2], "b_step": ("door_sills", "in_room")[i % 2],
             "b_low": ("chase", "rack")[(i // 2) % 2], "sheet_place": PLACES[i % 3]}
        v.update(fixed)
        out.append(mk(i + 1, v["bund"], v["b_step"], v["b_low"], v["sheet_place"]))
    return out


def quota_tests(out):
    check(S._parse_k("1-16,20,33-35,3") == list(range(1, 17)) + [20, 33, 34, 35] and S._parse_k("7") == [7],
          "run --k: a list of ranges and values", out)
    check(S.N_SET == 12 and S.QUOTA == {"bund": 6, "b_step": 6, "b_low": 6, "sheet_place": 4},
          "R26: twelve, quotas 6 / 6 / 6 / 4", out)
    # 1. the table's own first 48, all admitted: the set is k 1..12 (the table is balanced by construction)
    all48 = [{"k": k, "admitted": True, "quota": table_q(k)} for k in range(1, 49)]
    r = S.quota_select(all48)
    check(r["set"] == list(range(1, 13)) and r["complete"] and not r["skipped"],
          "R26: all admitted -> k 1..12 (balanced table)", out)
    check(balanced12(r["counts"]), "R26: the set's counts 6/6, 6/6, 6/6, 4/4/4", out)
    # 2. like an earlier check's label: k3 and k8 do not build, k1 and k5 fail row 3; the first twelve admitted are
    #    unbalanced, the quota rule skips what would overrun and goes on
    drop = {3, 8, 1, 5}
    cands = [{"k": k, "admitted": k not in drop, "quota": table_q(k)} for k in range(1, 49)]
    r = S.quota_select(cands)
    first12 = [k for k in range(1, 49) if k not in drop][:12]
    c12 = counts_of([table_q(k) for k in first12])
    check(any(max(v.values()) > S.QUOTA[q] for q, v in c12.items()),
          "R26 (setup): the first twelve admitted of this case are unbalanced (%s: %s)" % (first12, c12), out)
    check(r["complete"] and balanced12(r["counts"]),
          "R26: with k 1, 3, 5, 8 out the set is still balanced: %s" % r["set"], out)
    check(len(r["skipped"]) >= 1 and all("skipped by the quota" in r["status"][k] for k in r["skipped"]),
          "R26: a candidate skipped by a quota (%s: %s)" % (r["skipped"][:1], r["status"].get(r["skipped"][0])
                                                           if r["skipped"] else None), out)
    # 3. each quota on its own: the seventh of a kerb class, B-step or B-low variant, the fifth of a sheet place
    r = S.quota_select(seq(7, {"bund": "low"}) + [mk(8, "high", "in_room", "chase", "across")])
    check(r["set"] == [1, 2, 3, 4, 5, 6, 8] and "seventh with its bund kerb class low" in r["status"][7],
          "R26: the seventh low kerb is skipped (%s)" % r["status"][7], out)
    r = S.quota_select(seq(5, {"sheet_place": "along"}))
    check(r["set"] == [1, 2, 3, 4] and "fifth with its sheet place along" in r["status"][5],
          "R26: the fifth sheet place is skipped (%s)" % r["status"][5], out)
    r = S.quota_select(seq(7, {"b_step": "door_sills"}))
    check(r["set"] == [1, 2, 3, 4, 5, 6] and "seventh with its B-step variant door_sills" in r["status"][7],
          "R26: the seventh B-step variant is skipped", out)
    r = S.quota_select(seq(7, {"b_low": "rack"}))
    check(r["set"] == [1, 2, 3, 4, 5, 6] and "seventh with its B-low variant rack" in r["status"][7],
          "R26: the seventh B-low variant is skipped", out)
    # 4. fewer than twelve: only low kerbs admitted
    r = S.quota_select([{"k": k, "admitted": table_q(k)["bund"] == "low", "quota": table_q(k)} for k in range(1, 49)])
    check(not r["complete"] and len(r["set"]) == 6, "R26: fewer than twelve (only low kerbs admitted): %s" % r["set"],
          out)
    # 5. no tie: the order is k (any input order gives the same set; of alike candidates the lower k are taken)
    rnd = random.Random(7)
    sh = list(cands)
    rnd.shuffle(sh)
    check(S.quota_select(sh)["set"] == S.quota_select(cands)["set"], "R26: the input's order does not matter", out)
    twins = [mk(k, "low", "door_sills", "chase", "along") for k in (5, 2, 9, 14, 11)]
    check(S.quota_select(twins)["set"] == [2, 5, 9, 11],
          "R26: of alike candidates the lower k are taken (2, 5, 9, 11; 14 skipped)", out)
    # 6. an undecided candidate stops the rule before the set is complete, not after
    r = S.quota_select([dict(c, admitted=None) if c["k"] == 3 else c for c in all48])
    check(r["set"] == [1, 2] and r["stopped_at"] == 3 and not r["complete"]
          and r["status"][4].startswith("not decided (an earlier"), "R26: an undecided k 3 stops the rule", out)
    r = S.quota_select([dict(c, admitted=None) if c["k"] == 13 else c for c in all48])
    check(r["set"] == list(range(1, 13)) and r["complete"] and r["stopped_at"] is None
          and r["status"][13] == "not decided (the set was complete)", "R26: an undecided k 13 after the set: no matter",
          out)
    # 7. random record sets: balanced when complete; every skip is a real overrun at its turn; no admitted candidate
    #    before the set's last member (or anywhere when short) left out without a quota reason; never more than twelve
    bad = 0
    for trial in range(400):
        rr = random.Random(1000 + trial)
        p_adm = rr.choice((0.3, 0.6, 0.9))
        cs = [{"k": k, "admitted": rr.random() < p_adm, "quota": table_q(k)} for k in range(1, 49)]
        res = S.quota_select(cs)
        if res["complete"] and not balanced12(res["counts"]):
            bad += 1
        last = res["set"][-1] if res["set"] else 0
        for c in cs:
            if not c["admitted"] or c["k"] in res["set"]:
                continue
            if c["k"] < last or not res["complete"]:
                cnt = counts_of([table_q(k) for k in res["set"] if k < c["k"]])
                over = any(cnt[q].get(str(c["quota"][q]), 0) + 1 > S.QUOTA[q] for q in S.QUOTA)
                if not over or c["k"] not in res["skipped"]:
                    bad += 1
        if len(res["set"]) > S.N_SET:
            bad += 1
    check(bad == 0, "R26: 400 random record sets: balanced when complete, every skip a real overrun (%d bad)" % bad,
          out)


class _Args:
    pass


def rule_tests(out):
    with tempfile.TemporaryDirectory() as d:
        bad_ex = os.path.join(d, "excl_other.json")
        with open(bad_ex, "w") as f:
            json.dump({"rule": "test", "exclusion": [{"slot": "B-step", "variant": "in_room", "place": "*",
                                                      "type": "thermal"}]}, f)
        real = S.EXCLUSION_FILE
        dev = os.path.join(os.path.dirname(S.TASK), os.path.basename(S.TASK) + "-dev", "sketch", "site", "config",
                           "exclusion_v2.json")
        src = real if os.path.isfile(real) and S.sha256_file(real) == S.EXCLUSION_SHA256 else dev
        if not (os.path.isfile(src) and S.sha256_file(src) == S.EXCLUSION_SHA256):
            check(False, "R12 rule: the measured exclusion list (3413ac3b) not found at %s or %s" % (real, dev), out)
            return
        exd = S.load_json(src)
        rows = [dict(table_q(k), k=k) for k in range(1, 49)]
        paths = {}
        for name, tb in (("good", {"type_rule": "t", "exclusion": exd["exclusion"], "table": rows}),
                         ("otherlist", {"type_rule": "t", "exclusion": exd["exclusion"][:1], "table": rows}),
                         ("short", {"type_rule": "t", "exclusion": exd["exclusion"], "table": rows[:40]})):
            paths[name] = os.path.join(d, "tables_%s.json" % name)
            with open(paths[name], "w") as f:
                json.dump(tb, f)
        n = [0]

        def args(**kw):
            a = _Args()
            n[0] += 1
            a.out, a.label, a.k_range, a.tables, a.exclusion = os.path.join(d, "o%d" % n[0]), "hidden", "1-48", \
                paths["good"], src
            for k, v in kw.items():
                setattr(a, k, v)
            return a

        def refused(a):
            try:
                S.cmd_rule(a)
            except SystemExit:
                return True
            return False

        check(refused(args(exclusion=bad_ex)), "R12 rule: an exclusion list other than 3413ac3b is refused", out)
        check(refused(args(tables=paths["otherlist"])), "R12 rule: a tables block whose list differs is refused", out)
        check(refused(args(tables=None)), "R12 rule: the hidden label without tables is refused", out)
        check(refused(args(tables=paths["short"])), "R12 rule: a table that does not cover k 1-48 is refused", out)
        a = args()
        rc = S.cmd_rule(a)
        sel = S.load_json(os.path.join(a.out, "selection.json"))
        check(rc == 0 and sel["exclusion"]["sha256"] == S.EXCLUSION_SHA256 and len(sel["exclusion"]["entries"]) == 2
              and "seventh with its bund kerb class" in sel["rule"]["set"] and "fifth with its sheet place" in
              sel["rule"]["set"] and "complete at twelve" in sel["rule"]["set"]
              and sel["rule"]["quota"]["sheet_place"] == 4 and sel["rule"]["quota"]["bund"] == 6
              and "at most 1200 s of simulated time" in sel["rule"]["rows"]["3"]
              and "sim_time_s <= 1200 s" in sel["rule"]["how_the_rows_are_read"]["3"]
              and "R26" in sel["rule"]["source"] and "R26" in sel["rule"]["notes"]
              and "time-optimal" in sel["rule"]["notes"]["R13"] and sel["tables"]["table"][0]["k"] == 1,
              "R26 rule: selection.json holds the list with its sha256, the doubled quota rule, row 3's 1200 s, "
              "R13's and R26's notes, the table", out)
        check(S.cmd_rule(a) == 0, "R12 rule: written again with the same inputs: nothing changes", out)
        fa = _Args()
        fa.out, fa.label, fa.k_range, fa.candidates, fa.proofs = os.path.join(d, "no_rule"), "hidden", "1-48", d, \
            paths["good"]
        try:
            S.cmd_freeze(fa)
            refused_freeze = False
        except SystemExit:
            refused_freeze = True
        check(refused_freeze, "R12 freeze: the hidden label without a rule written first is refused", out)


def decide_tests(out):
    """`decide` end to end on a synthetic selection folder of 48 candidates (twelve of them form the set)."""
    with tempfile.TemporaryDirectory() as d:
        o = os.path.join(d, "out")
        os.makedirs(o)
        frozen = S.frozen_hashes()
        fd = S.digest_of(frozen)
        cfg = S.load_json(os.path.join(S.TESTS, "grader_config.json"))
        lim = (cfg.get("budgets") or {}).get("frames") or {}
        drop = {1: "row3", 5: "row1", 8: "row2"}
        recs, cfz = [], {}
        for k in range(1, 49):
            sha = hashlib.sha256(b"pkg%d" % k).hexdigest()
            q = table_q(k)
            recs.append({"k": k, "seed": S.seed_k(k, "dev-test"), "generated": True,
                         "proofs_ok": drop.get(k) != "row1", "failed_rows": ["tags"] if drop.get(k) == "row1" else [],
                         "package": {"sha256": sha}, "bund": q["bund"], "b_step_variant": q["b_step"],
                         "b_low_variant": q["b_low"], "sheet_place": q["sheet_place"], "table": dict(q, k=k),
                         "sheet_count": 2})
            cfz[S.key_of(k)] = {"k": k, "seed": S.seed_k(k, "dev-test"), "package": "dev-test_k%02d.tar.xz" % k,
                                "sha256": sha}
        with open(os.path.join(o, "proofs.json"), "w") as f:
            f.write("\n".join(json.dumps(r) for r in recs) + "\n")
        sel = {"rule": S.RULE, "label": "dev-test", "k_range": [1, 48],
               "tables": {"table": [dict(table_q(k), k=k) for k in range(1, 49)]}, "frozen": frozen,
               "candidates_frozen": cfz, "proofs": {"file": "proofs.json",
                                                    "sha256": S.sha256_file(os.path.join(o, "proofs.json"))},
               "candidates": {}, "set": [], "amendments": []}
        S.dump_json(sel, os.path.join(o, "selection.json"))
        items = [{"tag": "PI-%d" % i, "type": "gauge", "delivered": True, "evidence_ok": True} for i in range(8)] + \
                [{"tag": "SM-1", "type": "dose_point", "delivered": True, "E": 1},
                 {"tag": "X", "type": "hotspot", "delivered": True, "E": 1}]

        def put(k, prog, score, att=None):
            jd = os.path.join(o, "runs", S.key_of(k), prog)
            os.makedirs(jd, exist_ok=True)
            res = {"frozen_digest": fd, "package_sha256": cfz[S.key_of(k)]["sha256"], "score": score,
                   "attempts": [dict(att or {"items": items, "episode_score": score}, cell_failed=[])],
                   "counted_attempt": 1}
            S.dump_json(res, os.path.join(jd, "result.json"))

        for k in range(1, 49):
            if drop.get(k) == "row1":
                continue
            # 1146 s under a 1500 s limit (0.764 of it) is within row 3's absolute 1200 s
            att = {"items": items, "R": 1.0, "episode_score": 1.0, "sim_time_s": 1146.0, "sim_time_limit_s": 1500.0,
                   "dose_fraction": 0.62 if drop.get(k) == "row3" else 0.3,
                   "frames": {nm: int(0.2 * v) for nm, v in lim.items()},
                   "plates": {"log": True, "instruments": 8, "read_before_report": 7 if drop.get(k) == "row2" else 8}}
            put(k, "oracle", 1.0, att)
            for b in S.ROW4:
                put(k, b, 0.0)
        a = _Args()
        a.out = o
        rc = S.cmd_decide(a)
        sel = S.load_json(os.path.join(o, "selection.json"))
        adm = [k for k in range(1, 49) if k not in drop]
        want = S.quota_select([{"k": k, "admitted": k not in drop, "quota": table_q(k)} for k in range(1, 49)])
        check(rc == 0 and sel["complete"] and [sel["candidates"][x]["k"] for x in sel["set"]] == want["set"]
              and len(want["set"]) == 12, "R26 decide: the set by the quota rule %s (admitted %s)" % (want["set"], adm),
              out)
        st = {sel["candidates"][x]["k"]: sel["candidates"][x]["status"] for x in sel["candidates"]}
        check(st[1].startswith("dropped (row 3") and st[5].startswith("dropped (row 1") and
              st[8].startswith("dropped (row 2") and all(st[k].startswith("admitted") for k in adm),
              "R26 decide: every candidate's status (%s)" % sorted(set(st.values())), out)
        qr = sel.get("quota_result") or {}
        check(balanced12(qr.get("counts_in_set", {})) and qr.get("limits") == S.QUOTA
              and qr.get("admitted") == adm and "table" in (sel.get("hollowing") or {}),
              "R26 decide: counts 6/6, 6/6, 6/6, 4/4/4, admitted list and the hollowing table in selection.json", out)
        # fewer than twelve: every high kerb fails row 4 (corridor-only 0.5)
        for k in range(1, 49):
            if table_q(k)["bund"] == "high" and k not in drop:
                put(k, "corridor_only", 0.5)
        rc2 = S.cmd_decide(a)
        sel2 = S.load_json(os.path.join(o, "selection.json"))
        check(rc2 == 1 and not sel2["complete"] and len(sel2["set"]) == 6,
              "R26 decide: fewer than twelve -> exit 1, stop and report (%s)" % sel2["set"], out)


def main():
    res = run()
    for r in res:
        print(r)
    bad = [r for r in res if not r.startswith("ok")]
    print("test_select_set: %s" % ("OK" if not bad else "FAILED (%d)" % len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
