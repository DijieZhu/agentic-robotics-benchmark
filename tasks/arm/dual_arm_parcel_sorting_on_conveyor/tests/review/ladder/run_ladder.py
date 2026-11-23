"""The degenerate ladder: the first certificate of the grader (a maintainer's tool; nothing runs it at verify).

    python3 -B tests/review/ladder/run_ladder.py --out DIR [--jobs 6] [--shifts 3] [--only p1_hold_start ...]
    python3 -B tests/review/ladder/run_ladder.py --out DIR --wire-kit <folder holding cellkit/> --only p1_hold_start

Runs every degenerate program of ``programs/`` (each a ``solve(cell)`` program written for the wire), the example
program of the agent's payload, and the state-writing grader controls of ``controls.py``, on the review shifts
(``common.review_shifts``: labelled seeds "review", one per belt-speed class), in process through the cell's own
``handle()`` (``common.LocalCell``). With ``--wire-kit`` a program runs instead as a child process over the
JSON-lines wire of a cellkit (the client's launcher, the program's own process), a step towards the full graded path;
the full path (verify, the run account, the seccomp filter) is the packaging's and is marked pending in each program
(``WIRE_RUN``).

Expected: every program and the negative control score 0 on every shift (programs 1 and 6: credit exactly 0); the
positive control scores 1. ``DIR/ladder.json`` holds every record's summary; the table is printed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.dirname(HERE)
sys.path.insert(0, REVIEW)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import common  # noqa: E402

PROGRAMS = os.path.join(HERE, "programs")
EXAMPLE = os.path.join(common.PAYLOAD, "example", "sort.py")

EXPECT = {
    "p0_return_at_once": (0.0, None, "returns at once: every parcel lost"),
    "p1_hold_start": (0.0, 0, "do nothing: holds the start posture"),
    "p2_arms_out": (0.0, None, "both hands held in the parcels' path"),
    "p3_shove_against_guards": (0.0, None, "shoves every parcel against the near lip (the guards keep it on the belt)"),
    "p4_random_targets": (0.0, None, "random joint targets around the start posture, approached at 1 rad/s"),
    "p5_block_all": (0.0, None, "a hand held across the belt as a dam"),
    "p6_privileged_hold": (0.0, 0, "privileged: reads the truth, touches nothing"),
    "e0_example": (0.0, 0, "the agent's example program"),
}


# on the line setting nothing is lost by not being across: a program that returns at once, holds still or reads
# the truth ends with credit exactly 0
LINE_EXPECT = {
    "p0_return_at_once": (0.0, 0, "returns at once: nothing done, nothing lost (line setting)"),
}


def entries(only=None):
    out = []
    for p in sorted(glob.glob(os.path.join(PROGRAMS, "p*.py"))):
        name = os.path.splitext(os.path.basename(p))[0]
        out.append((name, "program", p))
    out.append(("e0_example", "program", EXAMPLE))
    from controls import CONTROLS
    for name in CONTROLS:
        out.append((name, "control", None))
    if only:
        out = [e for e in out if e[0] in only]
    return out


def summarize(rec, info):
    s = rec["safety"]
    return {"shift_score": rec["shift_score"], "credit": rec["credit"], "n_W": rec["n_W"], "n_U": rec["n_U"], "n_X": rec["n_X"],
            "W_readable": rec["W_readable"], "U_not_readable": rec["U_not_readable"], "lost": rec["lost"],
            "lost_by": rec["lost_by"], "done_reason": rec["done_reason"], "t_end": rec["t_end"],
            "stops": s["protective_stop_count"], "locked_out": s["locked_out"], "lockout_reason": s["lockout_reason"],
            "peak_force_N": s["peak_force_N"], "damaged": len(rec["parcel_damage"]),
            "frames_used": rec["frames"]["used"], "illegal_calls": rec["illegal_calls"],
            "refused_calls": rec["refused_calls"], "internal_errors": rec["internal_errors"],
            "max_entry_delay_s": rec["max_entry_delay_s"],
            "failed": sorted({f for p in rec["parcels"] for f in p["failed"]}),
            "program_error": (info.get("program_error") or "")[-600:] or None, "wall_s": info.get("wall_s"),
            "state_digest": info.get("state_digest")}


def _job(args):
    name, kind, path, k, seed, package, out = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    t0 = time.time()
    if kind == "control":
        from controls import CONTROLS
        rec, info = common.run_program_in_process(None, package, control=CONTROLS[name][0])
    else:
        sys.path.insert(0, os.path.dirname(path))
        mod = common.load_program(path)
        priv = bool(getattr(mod, "PRIVILEGED", False))
        rec, info = common.run_program_in_process(path, package, privileged=priv)
    row = {"entry": name, "kind": kind, "k": k, "seed": seed, "belt_class": package["meta"]["belt_class"],
           "belt_speed": package["belt_speed"][0][1], "n_parcels": len(package["arrivals"]),
           "duration_s": package["duration"], **summarize(rec, info), "job_wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(out, "%s__k%d.json" % (name, k)), "w") as fh:
        json.dump({"row": row, "record": rec}, fh, indent=1, default=float)
    return row


def _wire_job(name, path, package, kit_parent, out):
    """One program over a cellkit's JSON-lines wire (child process, the client's launcher)."""
    import tempfile
    sys.path.insert(0, kit_parent)
    from cellkit import wire  # noqa: E402
    sys.path.remove(kit_parent)
    from parcelcell.client import launcher_argv
    srv = common.make_server()
    srv.reset(package)
    work = tempfile.mkdtemp(prefix="ladder_wire_", dir=out)
    prog = os.path.join(work, "prog")
    os.makedirs(prog)
    for f in glob.glob(os.path.join(os.path.dirname(path), "*.py")):
        if f == path or os.path.basename(f).startswith("_"):
            with open(f) as a, open(os.path.join(prog, "sort.py" if f == path else os.path.basename(f)), "w") as b:
                b.write(a.read())
    argv = launcher_argv(sys.executable, common.TESTS, prog, "sort")
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1", "HOME": work,
           "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"), "ARB_GIVEN_DIR": common.GIVEN}
    proc = wire.AgentProcess(argv, cwd=prog, env=env, stderr_path=os.path.join(work, "stderr.log"))
    try:
        reason = wire.serve_episode(srv, proc, srv.program_budget_s, srv.cfg["budgets"]["silence_s"])
    finally:
        proc.terminate()
    rec = srv.finalize()
    srv.close()
    return reason, rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--shifts", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--wire-kit", default=None, help="a folder that holds cellkit/ (wire mode)")
    ap.add_argument("--setting", default="rated", choices=("rated", "line"),
                    help="rated: the review shifts of the rated setting; line: line review shifts (slow running, "
                         "medium indexing, fast running)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    shifts = common.review_shifts(n=a.shifts) if a.setting == "rated" else common.line_review_shifts(n=a.shifts)
    ents = entries(a.only)
    if a.wire_kit:
        rows = []
        for name, kind, path in ents:
            if kind != "program":
                continue
            for k, seed, pkg in shifts:
                reason, rec = _wire_job(name, path, pkg, a.wire_kit, a.out)
                rows.append({"entry": name, "k": k, "wire_end": reason, "shift_score": rec["shift_score"],
                             "credit": rec["credit"], "done_reason": rec["done_reason"],
                             "program_time_s": rec["program_time_s"], "frames": rec["frames"]["used"]})
                print(json.dumps(rows[-1]))
        common.dump_json(os.path.join(a.out, "ladder_wire.json"), rows)
        return 0
    jobs = [(n, kind, p, k, seed, pkg, a.out) for n, kind, p in ents for k, seed, pkg in shifts]
    import multiprocessing as mp
    rows = []
    with mp.get_context("fork").Pool(min(a.jobs, 8), maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_job, jobs):
            rows.append(row)
            print("%-22s k=%-3d %-6s score %.4f credit %3d (W %d readable %d, U %d not %d, X %d, lost %d) stops %d %s %.0fs"
                  % (row["entry"], row["k"], row["belt_class"], row["shift_score"], row["credit"], row["n_W"],
                     row["W_readable"], row["n_U"], row["U_not_readable"], row["n_X"], row["lost"], row["stops"],
                     row["done_reason"], row["job_wall_s"]), flush=True)
    from controls import CONTROLS
    exp = dict(EXPECT)
    if a.setting == "line":
        exp.update(LINE_EXPECT)
    for n, (fn, score, why) in CONTROLS.items():
        exp[n] = (score, None, why)
    if a.setting == "rated":                         # no due time on the rated setting: a held parcel still counts
        exp["c7_turn_one_w_late"] = ("credit 1", None, "rated setting: no due time (credit 1)")
        exp["c8_two_w_one_u_late"] = ("credit 2", None, "rated setting: no due time (credit 2)")
    table = []
    ok_all = True
    for name, kind, path in ents:
        rs = sorted([r for r in rows if r["entry"] == name], key=lambda r: r["k"])
        e_score, e_credit, why = exp.get(name, (0.0, None, ""))
        if e_score == "credit 1":                    # exactly one credit: the score is 1 / #W of the shift
            ok = all(r["credit"] == 1 and r["n_W"] > 0 and abs(r["shift_score"] - 1.0 / r["n_W"]) < 1e-12 for r in rs)
        elif e_score == "credit 0":
            ok = all(r["credit"] == 0 and r["shift_score"] == 0.0 for r in rs)
        elif e_score == "credit 2":
            ok = all(r["credit"] == 2 and r["n_W"] > 0 and abs(r["shift_score"] - 2.0 / r["n_W"]) < 1e-12 for r in rs)
        else:
            ok = all(abs(r["shift_score"] - e_score) < 1e-9 and (e_credit is None or r["credit"] == e_credit)
                     for r in rs)
        ok_all &= ok
        table.append({"entry": name, "kind": kind, "what": why, "expected_score": e_score,
                      "expected_credit": e_credit,
                      "measured": [{"k": r["k"], "belt_class": r["belt_class"], "score": r["shift_score"],
                                    "credit": r["credit"], "lost": r["lost"], "stops": r["stops"],
                                    "locked_out": r["locked_out"], "done": r["done_reason"]} for r in rs],
                      "pass": ok,
                      "wire_run": "n/a (state write)" if kind == "control" else
                      getattr(common.load_program(path), "WIRE_RUN", "pending")})
    res = {"shifts": [{"k": k, "seed": seed, "belt_class": p["meta"]["belt_class"],
                       "belt_speed": p["belt_speed"][0][1], "n_parcels": len(p["arrivals"]),
                       "duration_s": p["duration"]} for k, seed, p in shifts],
           "table": table, "rows": rows, "all_pass": ok_all}
    common.dump_json(os.path.join(a.out, "ladder.json"), res)
    for t in table:
        if t["expected_score"] in ("credit 1", "credit 0", "credit 2"):
            t["expected_credit"] = int(t["expected_score"][-1])
            t["expected_score"] = -1.0                # printed as "credit n" below
    print("\n| entry | what | expected | " + " | ".join("k=%d (%s)" % (k, p["meta"]["belt_class"]) for k, _, p in shifts)
          + " | pass |")
    print("|---|---|---|" + "---|" * len(shifts) + "---|")
    for t in table:
        cells = ["%.3f (credit %d, lost %d, stops %d%s)" % (m["score"], m["credit"], m["lost"], m["stops"],
                                                           ", locked out" if m["locked_out"] else "")
                 for m in t["measured"]]
        print("| %s | %s | %s | %s | %s |" % (t["entry"], t["what"], "%g" % t["expected_score"] +
                                             ("" if t["expected_credit"] is None else " (credit %d)" % t["expected_credit"]),
                                             " | ".join(cells), "yes" if t["pass"] else "NO"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
