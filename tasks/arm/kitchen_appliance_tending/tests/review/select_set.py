"""Selection of the fixed hidden episode set (a maintainer's tool; the grader never runs it).

The rule, fixed before any candidate ran:

1. Freeze. The grader (tests/verify.py, tests/episode.py, tests/appliance_env.py, tests/units.json) and the reference
   (solution/oracle/*.py, solution/oracle/units.json, solution/solve.sh) are committed first: `draw` refuses to start
   while any of them differs from the checkout's HEAD and records their sha256 and the commit; `run` and `make` refuse
   to go on when any of them differs from the recorded hashes.
2. Draw. `draw` takes the set's key from os.urandom(32) (hex) and writes it to the record before any candidate runs.
   Unit u's candidate i (i = 0, 1, 2, ...) has the 64-bit seed
   int.from_bytes(blake2b(b"candidate|<u>|<i>", key=key, digest_size=8).digest(), "little").
3. Run. Per unit, candidates in the order i = 0, 1, ...: the privileged reference in process (reference.Driver on the
   true simulator, each request run as oracle_main.run_request runs it), then, if that scored 1.0, three times over
   the real wire: verify.run_episodes with that one episode and the oracle submission as solution/solve.sh lays it out
   (a subprocess speaking the grading wire, given the token truth), the probe lock at the grader's rate, each run with
   a fresh OS-entropy stream (robotics_grader.seed_rng).
4. Keep. A candidate is kept when it scores 1.0 in process and in each of the three wire runs; per unit the first two
   kept are the unit's episodes. Every candidate tried is recorded: kept, or rejected with its scores and the requests
   it lost.
5. Cover. The 54 requests of the 18 kept episodes (in process) must hold every request type the instruction lists
   (REQUEST_TYPES). If they do not, the draw is void: `draw --void REASON` records why, draws a new key, and `run`
   starts again; the record keeps every draw. A draw is void too when a frozen file changes after it was made (the
   grader's rule or the reference changed): `draw --void REASON --refreeze` records why, keeps that draw's frozen
   hashes, commit and candidates with it, freezes the checkout's HEAD again (step 1) and draws a new key.
6. Order. `make` shuffles the 18 (units in sorted order, each unit's two in candidate order; a Fisher-Yates in which
   position i swaps with int.from_bytes(blake2b(b"order|<i>", key=key, digest_size=8).digest(), "little") mod (i + 1),
   i from 17 down to 1) and writes tests/hidden/episodes.json in that order: the graded order.

    python3 -B tests/review/select_set.py draw --record tests/hidden/episode_selection.json
    python3 -B tests/review/select_set.py run --record tests/hidden/episode_selection.json --work DIR [--workers 3]
    python3 -B tests/review/select_set.py make --record tests/hidden/episode_selection.json --out tests/hidden/episodes.json
    python3 -B tests/review/select_set.py amend --record tests/hidden/episode_selection.json --reason TEXT \
        --evidence EVIDENCE.json [--workers 3]

7. Amend (a maintainer's decision to keep a made set when a frozen file changed in a way that cannot change what the
   reference does or how it is scored, the renderer-only amendment of shoe_lacing_and_boxing's selection): `amend`
   refuses while a frozen file differs from the checkout's HEAD, re-runs the privileged reference in process (step 3)
   on every episode of the made draw with the files as they are now (each must score 1.0 again, with the same steps
   and per-request outcomes as when it was kept; otherwise it refuses and the draw must be voided), and appends to the
   record's `amendments`: the reason, the frozen files that changed (their recorded and their new sha256), the commit
   that holds them (the last commit that changed any frozen file), the re-run per episode and the maintainer's
   evidence (a JSON file: e.g. the physics comparison and the validate run). Then it pins the frozen files' new
   hashes and that commit, so `run` / `make` check against them again; tests/hidden/episodes.json is not touched. A
   later change of a frozen file needs a new amendment (or a new draw).

`run` needs the task image's Python (3.12, MuJoCo 3.16, NumPy, SciPy, robosuite 1.5.2's model files) and the cameras'
renderer (kitchen_render.py: MuJoCo's Filament on Vulkan, lavapipe in the image, with the room bundle). Outside the image the oracle submission's run.sh names this interpreter, and the
controller is started directly rather than through nonet_exec.py (which needs root to change user); everything else is
verify.run_episodes as the grader runs it.
"""

import argparse
import datetime
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
ORACLE = os.path.join(TASK, "solution", "oracle")
sys.dont_write_bytecode = True

FORMAT = "kitchen_appliance_tending episode selection 1"
SET_FORMAT = "kitchen_appliance_tending fixed set 1"
FROZEN = ("tests/verify.py", "tests/episode.py", "tests/appliance_env.py", "tests/units.json",
          "tests/grader_config.json", "solution/oracle/appliance_env.py", "solution/oracle/episode.py",
          "solution/oracle/oracle_main.py", "solution/oracle/reference.py", "solution/oracle/units.json",
          "solution/solve.sh")
PER_UNIT = 2
WIRE_RUNS = 3
MAX_CANDIDATES = 16
REQUEST_TYPES = ("burner_on", "burner_off", "burners_on", "burners_off", "all_off", "burner_level",
                 "start", "stop", "preheat", "oven_off", "oven_level")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def frozen_hashes():
    return {p: sha256_file(os.path.join(TASK, p)) for p in FROZEN}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args):
    return subprocess.run(["git", "-C", TASK] + list(args), capture_output=True, text=True, timeout=60)


def frozen_committed():
    tracked = git("ls-files", "--error-unmatch", "--", *FROZEN)
    clean = git("diff", "--quiet", "HEAD", "--", *FROZEN)
    return tracked.returncode == 0 and clean.returncode == 0


def load(path):
    with open(path) as f:
        return json.load(f)


def save(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def keyed(key_hex, msg):
    return int.from_bytes(hashlib.blake2b(msg.encode(), key=bytes.fromhex(key_hex), digest_size=8).digest(), "little")


def candidate_seed(key_hex, unit, i):
    return keyed(key_hex, "candidate|%s|%d" % (unit, i))


def check_frozen(rec):
    now = frozen_hashes()
    bad = [p for p in FROZEN if rec["frozen"].get(p) != now[p]]
    if bad:
        raise SystemExit("frozen files changed since the draw: " + ", ".join(bad))


def open_draw(rec):
    d = rec["draws"][-1]
    if d["status"] != "open":
        raise SystemExit("the latest draw is %s: draw again" % d["status"])
    return d


# ------------------------------------------------------------------------------------------- one candidate
def _modules():
    for p in (ORACLE, TESTS):
        if p not in sys.path:
            sys.path.insert(0, p)              # TESTS first: verify, episode, appliance_env (the oracle's are identical)
    import appliance_env as E  # noqa: E402
    import episode as EP  # noqa: E402
    import oracle_main as O  # noqa: E402
    import reference as R  # noqa: E402
    import verify as V  # noqa: E402
    return E, EP, O, R, V


def _specs(E):
    units = load(os.path.join(TESTS, "units.json"))
    return {u: E.unit_spec(c["model"], files=E.read_unit_files(os.path.join(TESTS, c["dir"])), quirks=c.get("quirks"))
            for u, c in sorted(units.items())}


def _req(r):
    t = r["target"]
    return {"id": r["id"], "what": r["what"], "target": list(t) if isinstance(t, (list, tuple)) else t, "text": r["text"],
            "t_open": int(r["t_open"]), "t_close": None if r["t_close"] is None else int(r["t_close"]),
            "outcome": r["outcome"]}


def in_process(E, EP, O, R, spec, seed):
    t0 = time.time()
    ep = EP.Episode(E.ApplianceEnv(spec, None, render=False), seed)
    ep.reset()
    initial = ep.discrete()
    drv = R.Driver(ep)
    drv.MAX_STEPS = 10 ** 6
    status, unstable = [], False
    while ep.current is not None and not unstable:
        r = ep.current
        try:
            O.run_request(drv, r["text"])
            status.append("ok")
        except R.PrimitiveError as exc:
            status.append("fail: %s" % exc.stage)
        while ep.current is not None and ep.current["id"] == r["id"]:
            drv.tick(drv.site_pos(), drv.site_R(), -1.0)
            if ep.env.unstable():
                unstable = True
                break
    sc = ep.score()
    out = {"credit": 0.0 if unstable else float(sc["credit"]), "unstable": unstable, "steps": int(ep.env.t),
           "jitter_m_m_rad": [round(float(x), 6) for x in ep.env.jitter], "initial": initial,
           "requests": [dict(_req(q), primitive=status[i] if i < len(status) else "none")
                        for i, q in enumerate(sc["requests"])], "wall_s": round(time.time() - t0, 1)}
    ep.env.close()
    return out


def submission(work):
    """The oracle submission as solution/solve.sh lays it out, with run.sh naming this interpreter."""
    if os.path.isdir(work):
        shutil.rmtree(work)
    shutil.copytree(ORACLE, work)
    shutil.copy(os.path.join(TASK, "solution", "oracle_token"), os.path.join(work, ".oracle_token"))
    with open(os.path.join(work, "run.sh"), "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\n"
                "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
                "exec %s -I -B oracle_main.py --units units.json\n" % sys.executable)
    os.chmod(os.path.join(work, "run.sh"), 0o755)
    return work


def over_wire(V, specs, unit, seed, work):
    from robotics_grader import seed_rng

    def spawn(self):
        return subprocess.Popen(["bash", self.entry], cwd=self.submission, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                                bufsize=1, start_new_session=True)
    V.GuardedSolver._spawn = spawn
    # step 3: each wire run with a fresh OS-entropy stream, not the grader's per-episode stream (verify.probe_stream
    # keys on the episode's index in the graded order, which `make` decides later)
    V.probe_stream = lambda _index, _seed: seed_rng()
    config = load(os.path.join(TESTS, "grader_config.json"))
    t0 = time.monotonic()
    (scores, aborted, faults, _pu, _pr, _eg, _be, lock, records) = V.run_episodes(
        config, os.getuid(), "none", t0, specs, work, True, [(unit, seed)])
    rec = records[0]
    return {"credit": float(scores[0]), "reason": rec["reason"], "steps": int(rec["steps"]), "aborted": int(aborted),
            "sim_faults": int(faults), "probes": int(lock.total), "probes_passed": int(lock.passed),
            "probes_idle_skipped": int(lock.idle_skipped),
            "requests": [_req(q) for q in rec["requests"]], "wall_s": round(time.monotonic() - t0, 1)}


def run_unit(args):
    unit, key_hex, work_root, start = args
    E, EP, O, R, V = _modules()
    specs = _specs(E)
    out, kept = [], 0
    for i in range(start, MAX_CANDIDATES):
        seed = candidate_seed(key_hex, unit, i)
        c = {"unit": unit, "index": i, "seed": seed}
        c["in_process"] = in_process(E, EP, O, R, specs[unit], seed)
        c["wire"] = []
        ok = c["in_process"]["credit"] == 1.0
        for k in range(WIRE_RUNS if ok else 0):
            w = over_wire(V, specs, unit, seed, submission(os.path.join(work_root, "%s_%d_%d" % (unit, i, k))))
            c["wire"].append(w)
            if w["credit"] != 1.0:
                ok = False
                break
        c["status"] = "kept" if ok else "rejected"
        if not ok:
            lost = [q for q in c["in_process"]["requests"] if q["outcome"] != "completed"]
            for w in c["wire"]:
                lost += [dict(q, run="wire") for q in w["requests"] if q["outcome"] != "completed"]
            c["lost"] = lost
        out.append(c)
        print("[select] %s candidate %d: %s (in process %.3f; wire %s)" % (
            unit, i, c["status"], c["in_process"]["credit"], [round(w["credit"], 3) for w in c["wire"]]),
            file=sys.stderr, flush=True)
        kept += ok
        if kept >= PER_UNIT:
            break
    return unit, out


# ------------------------------------------------------------------------------------------- commands
def cmd_draw(a):
    rec = load(a.record) if os.path.exists(a.record) else None
    if not frozen_committed():
        raise SystemExit("commit the frozen files first: " + ", ".join(FROZEN))
    head = git("rev-parse", "HEAD").stdout.strip()
    if rec is not None:
        last = rec["draws"][-1]
        if a.refreeze:
            if not a.void or last["status"] not in ("open", "made"):
                raise SystemExit("--refreeze voids the latest open or made draw: pass --void REASON")
            last["frozen"], last["frozen_commit"] = rec["frozen"], rec["frozen_commit"]
            if "after_selection" in rec:
                last["after_selection"] = rec.pop("after_selection")
            rec["frozen"], rec["frozen_commit"] = frozen_hashes(), head
        elif last["status"] == "made":
            raise SystemExit("the latest draw is made: after a frozen file changed, void it with --void REASON "
                             "--refreeze")
        elif last["status"] == "open" and not a.void:
            raise SystemExit("the latest draw is open: pass --void REASON to void it")
        if a.void and last["status"] != "void":
            last["status"] = "void"
            last["void_reason"] = a.void
            last["voided_utc"] = utc_now()
    if rec is None:
        rec = {"format": FORMAT, "rule": "tests/review/select_set.py (module docstring)",
               "frozen": frozen_hashes(), "frozen_commit": head, "draws": [], "candidates": []}
    else:
        check_frozen(rec)
    rec["draws"].append({"draw": len(rec["draws"]) + 1, "drawn_utc": utc_now(), "key": os.urandom(32).hex(),
                         "source": "os.urandom(32)", "frozen_commit": rec["frozen_commit"], "status": "open"})
    save(a.record, rec)
    print("draw %d written to %s" % (len(rec["draws"]), a.record))


def cmd_run(a):
    rec = load(a.record)
    check_frozen(rec)
    d = open_draw(rec)
    units = sorted(load(os.path.join(TESTS, "units.json")))
    done = {}
    for c in rec["candidates"]:
        if c["draw"] == d["draw"]:
            done.setdefault(c["unit"], []).append(c)
    todo = []
    for u in units:
        cs = done.get(u, [])
        if sum(c["status"] == "kept" for c in cs) < PER_UNIT:
            rec["candidates"] = [c for c in rec["candidates"] if not (c["draw"] == d["draw"] and c["unit"] == u)]
            todo.append((u, d["key"], a.work, 0))
    os.makedirs(a.work, exist_ok=True)
    with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
        for unit, cs in pool.imap_unordered(run_unit, todo):
            rec = load(a.record)
            rec["candidates"] = [c for c in rec["candidates"] if not (c["draw"] == d["draw"] and c["unit"] == unit)]
            rec["candidates"] += [dict(c, draw=d["draw"]) for c in cs]
            rec["candidates"].sort(key=lambda c: (c["draw"], c["unit"], c["index"]))
            save(a.record, rec)
            print("%s: %d candidates, %d kept" % (unit, len(cs), sum(c["status"] == "kept" for c in cs)), flush=True)


def cmd_make(a):
    rec = load(a.record)
    check_frozen(rec)
    d = open_draw(rec)
    units = sorted(load(os.path.join(TESTS, "units.json")))
    chosen = []
    for u in units:
        kept = [c for c in rec["candidates"] if c["draw"] == d["draw"] and c["unit"] == u and c["status"] == "kept"]
        kept.sort(key=lambda c: c["index"])
        if len(kept) < PER_UNIT:
            raise SystemExit("unit %s has %d kept candidates: run again" % (u, len(kept)))
        chosen += kept[:PER_UNIT]
    types = {}
    for c in chosen:
        for q in c["in_process"]["requests"]:
            types[q["what"]] = types.get(q["what"], 0) + 1
    missing = [t for t in REQUEST_TYPES if t not in types]
    if missing:
        raise SystemExit("the draw misses request types %s: void it (draw --void ...) and run again" % missing)
    order = list(range(len(chosen)))
    for i in range(len(order) - 1, 0, -1):
        k = keyed(d["key"], "order|%d" % i) % (i + 1)
        order[i], order[k] = order[k], order[i]
    graded = [chosen[j] for j in order]
    tried = [c for c in rec["candidates"] if c["draw"] == d["draw"]]
    rejected = ["%s#%d (in process %.3f, wire %s)" % (c["unit"], c["index"], c["in_process"]["credit"],
                                                      [round(w["credit"], 3) for w in c["wire"]])
                for c in tried if c["status"] == "rejected"]
    d.update(status="made", made_utc=utc_now(), request_types=dict(sorted(types.items())),
             graded_order=["%s#%d" % (c["unit"], c["index"]) for c in graded])
    save(a.record, rec)
    voided = [x for x in rec["draws"] if x["status"] == "void"]
    source = ("Selected %s by tests/review/select_set.py (the rule in its docstring), draw %d of %d (%d voided); key "
              "os.urandom(32), drawn %s, written to tests/hidden/episode_selection.json before any candidate ran. "
              "Frozen at the draw (sha256 in the record; commit %s): %s. Per unit, candidate seeds from the key in a "
              "fixed order; kept: the reference scored 1.0 in process and in %d runs over the real wire "
              "(verify.run_episodes, the oracle submission, the probe lock); the first %d kept per unit. Tried %d, "
              "rejected %d: %s. Request types over the %d requests: %s. Graded order: a Fisher-Yates keyed by the "
              "same key.") % (
        d["made_utc"][:10], d["draw"], len(rec["draws"]), len(voided), d["drawn_utc"], rec["frozen_commit"][:12],
        ", ".join(FROZEN), WIRE_RUNS, PER_UNIT, len(tried), len(rejected), "; ".join(rejected) or "none",
        sum(types.values()), ", ".join("%s %d" % kv for kv in sorted(types.items())))
    out = {"format": SET_FORMAT, "episodes": [{"unit": c["unit"], "seed": c["seed"]} for c in graded],
           "source": source}
    save(a.out, out)
    print("wrote %d episodes to %s" % (len(graded), a.out))


def _amend_one(args):
    """Step 3's in-process run of one episode of the made draw (a worker of `amend`)."""
    name, unit, seed = args
    E, EP, O, R, _V = _modules()
    specs = _specs(E)
    return name, in_process(E, EP, O, R, specs[unit], seed)


def cmd_amend(a):
    rec = load(a.record)
    if not frozen_committed():
        raise SystemExit("commit the frozen files first: " + ", ".join(FROZEN))
    made = [d for d in rec["draws"] if d["status"] == "made"]
    if not made or rec["draws"][-1] is not made[-1]:
        raise SystemExit("amend keeps the latest draw, which must be made")
    d = made[-1]
    now = frozen_hashes()
    changed = {p: {"recorded": rec["frozen"].get(p), "now": now[p]} for p in FROZEN if rec["frozen"].get(p) != now[p]}
    if not changed:
        raise SystemExit("no frozen file changed since the record's hashes: nothing to amend")
    with open(a.evidence) as f:
        evidence = json.load(f)
    # the commit that holds the frozen files as they are now: the last one that changed any of them (later commits
    # touch other files only, so this one stays valid when they are reordered)
    commit = git("log", "-1", "--format=%H", "--", *FROZEN).stdout.strip()
    by_name = {"%s#%d" % (c["unit"], c["index"]): c for c in rec["candidates"] if c["draw"] == d["draw"]}
    jobs = [(name, by_name[name]["unit"], by_name[name]["seed"]) for name in d["graded_order"]]
    with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
        results = dict(pool.imap_unordered(_amend_one, jobs))
    rerun, bad = [], []
    for name, _unit, seed in jobs:
        r, was = results[name], by_name[name]["in_process"]
        same = (r["credit"] == 1.0 and r["steps"] == was["steps"]
                and [(q["what"], q["outcome"], q["t_close"]) for q in r["requests"]]
                == [(q["what"], q["outcome"], q["t_close"]) for q in was["requests"]])
        rerun.append({"episode": name, "seed": seed, "credit": r["credit"], "steps": r["steps"],
                      "outcomes": [q["outcome"] for q in r["requests"]], "t_close": [q["t_close"] for q in r["requests"]],
                      "same_as_kept": same, "wall_s": r["wall_s"]})
        print("[amend] %s: credit %.3f, %d steps, same as when kept: %s" % (name, r["credit"], r["steps"], same),
              file=sys.stderr, flush=True)
        if not same:
            bad.append(name)
    if bad:
        raise SystemExit("the reference no longer repeats its kept runs on %s: void the draw instead" % ", ".join(bad))
    amendments = rec.setdefault("amendments", [])
    amendments.append({
        "amendment": len(amendments) + 1, "amended_utc": utc_now(), "draw": d["draw"], "commit": commit,
        "reason": a.reason, "changed": changed, "frozen_before": rec["frozen"],
        "frozen_commit_before": rec["frozen_commit"],
        "rerun_in_process": {"rule": "step 3's in-process run of the privileged reference on every episode of the made "
                                     "draw, with the frozen files of the commit; each scored 1.0 with the steps and "
                                     "the per-request outcomes and closing steps recorded when it was kept",
                             "episodes": rerun},
        "evidence": evidence})
    rec["frozen"], rec["frozen_commit"] = now, commit
    save(a.record, rec)
    check_frozen(rec)
    print("amendment %d written to %s (%d frozen files re-pinned at %s)" % (len(amendments), a.record, len(changed),
                                                                          commit[:12]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("draw")
    p.add_argument("--record", required=True)
    p.add_argument("--void", default=None, help="void the open draw, with this reason")
    p.add_argument("--refreeze", action="store_true",
                   help="with --void: void the latest draw (open or made) because a frozen file changed, and freeze "
                        "the checkout's HEAD again")
    p = sub.add_parser("run")
    p.add_argument("--record", required=True)
    p.add_argument("--work", required=True, help="a scratch folder for the oracle submissions")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("make")
    p.add_argument("--record", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("amend")
    p.add_argument("--record", required=True)
    p.add_argument("--reason", required=True, help="why the made set is kept although frozen files changed")
    p.add_argument("--evidence", required=True, help="a JSON file of the maintainer's evidence (physics, validate)")
    p.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    {"draw": cmd_draw, "run": cmd_run, "make": cmd_make, "amend": cmd_amend}[a.cmd](a)


if __name__ == "__main__":
    main()
