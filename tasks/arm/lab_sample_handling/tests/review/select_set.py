"""Selection of the fixed hidden episode set (a maintainer's tool; the grader never runs it).

The rule, fixed before any candidate ran:

1. Freeze. The grader (tests/verify.py, tests/episode.py, tests/lab_env.py, tests/lab_sim.py,
   tests/box_transport_env.py, tests/units.json, tests/grader_config.json) and the reference (the solution/oracle/
   modules and units.json, solution/solve.sh) are committed first: `draw` refuses to start while any of them differs
   from the checkout's HEAD and records their sha256 and the commit; `run` and `make` refuse to go on when any of them
   differs from the recorded hashes.
2. Draw. `draw` takes the set's key from os.urandom(32) (hex) and writes it to the record before any candidate runs.
   Unit u's candidate i (i = 0, 1, 2, ...) has the 64-bit seed
   int.from_bytes(blake2b(b"candidate|<u>|<i>", key=key, digest_size=8).digest(), "little").
3. Run. Per unit, candidates in the order i = 0, 1, ...: the privileged reference three times over the real wire,
   stopping at the first run below credit 1.0: verify.run_episodes with that one episode and the oracle submission as
   solution/solve.sh lays it out (a subprocess speaking the grading wire, given the token truth), the probe lock at
   the grader's rate, each run with a fresh OS-entropy stream (robotics_grader.seed_rng). The reference thinks under a
   wall-clock limit per step (wire_world.THINK_S: it holds still while it has not answered), so its runs of one
   episode can differ; it is run only as the grader runs it, as a separate process (in the grader's process, its
   thinking would share the interpreter with the simulation and hold far more often).
4. Keep. A candidate is kept when it scores credit 1.0 in each of the three wire runs; per unit the first two kept
   are the unit's episodes, at most MAX_CANDIDATES tried. Every candidate tried is recorded: kept, or rejected with
   its scores and the parts it lost.
5. Cover. Among the 12 kept episodes (lab_sim.draw_unit(seed)["lid_closed"]) the lid starts shut in at least COVER_LID_SHUT and open in at least
   COVER_LID_OPEN (lab_sim draws it shut with probability 0.5, so a draw misses this about 4 % of the time). If it does
   not, the draw is void: `draw --void REASON` records why, draws a new key, and `run` starts again; the record keeps
   every draw. A draw is void too when a frozen file changes after it was made: `draw --void REASON --refreeze`
   records why, keeps that draw's frozen hashes, commit and candidates with it, freezes the checkout's HEAD again
   (step 1) and draws a new key.
6. Order. `make` shuffles the 12 (units in sorted order, each unit's two in candidate order; a Fisher-Yates in which
   position i swaps with int.from_bytes(blake2b(b"order|<i>", key=key, digest_size=8).digest(), "little") mod (i + 1),
   i from 11 down to 1) and writes tests/hidden/episodes.json in that order: the graded order.

    python3 -B tests/review/select_set.py draw --record tests/hidden/episode_selection.json
    python3 -B tests/review/select_set.py run --record tests/hidden/episode_selection.json --work DIR [--workers 6]
    python3 -B tests/review/select_set.py make --record tests/hidden/episode_selection.json --out tests/hidden/episodes.json
    python3 -B tests/review/select_set.py amend --record tests/hidden/episode_selection.json --reason TEXT \
        --evidence EVIDENCE.json

7. Amend (added 2026-11-18 by the maintainers; a maintainer's decision to keep a made set when frozen files changed
   in a way that changes no episode: not its bench, its worklist, its fills, its unit or its credit; the amendments
   of kitchen_appliance_tending's selection). `amend` refuses while a frozen file differs from the checkout's HEAD
   and appends to the record's `amendments`: the reason, the frozen files that changed (their recorded and their new
   sha256), the commit that holds them (the last commit that changed any frozen file) and the maintainer's evidence,
   a JSON file with one full verify of the privileged reference on the made set as the grader runs it (the real
   wire, the grader's probe schedule, the grading image) with the files of that commit: every episode, in the graded
   order, with its credit and steps. `amend` refuses unless each of them scored credit 1.0 again; then the draw must
   be voided instead. Then it pins the frozen files' new hashes and that commit, so `run` / `make` check against
   them again; tests/hidden/episodes.json is not touched. A later change of a frozen file needs a new amendment (or
   a new draw).

`run` needs the task image's Python (3.12, MuJoCo 3.16.0, NumPy 2.5.4, SciPy 1.18.1, OpenCV; environment/
requirements.txt) and an OSMesa renderer (MUJOCO_GL=osmesa). Outside the image the oracle submission's run.sh names
this interpreter, and the controller is started directly rather than through nonet_exec.py (which needs root to change
user); everything else is verify.run_episodes as the grader runs it.
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

FORMAT = "lab_sample_handling episode selection 1"
SET_FORMAT = "lab_sample_handling fixed set 1"
ORACLE_MODULES = ("bench_estimators.py", "box_transport_env.py", "centrifuge_probe.py", "episode.py", "fill_read.py",
                  "lab_env.py", "lab_render.py", "lab_sim.py", "localize_probe.py", "mass_probe.py", "oracle_main.py",
                  "rotor_angle.py", "tube_insert_probe.py", "turn_probe.py", "wire_world.py", "worklist_probe.py",
                  "worklist_scene.py")
FROZEN = (("tests/verify.py", "tests/episode.py", "tests/lab_env.py", "tests/lab_sim.py", "tests/box_transport_env.py",
           "tests/lab_render.py", "tests/lab_look.json", "tests/units.json", "tests/grader_config.json")
          + tuple("solution/oracle/" + f for f in ORACLE_MODULES)
          + ("solution/oracle/lab_look.json", "solution/oracle/units.json", "solution/solve.sh"))
PER_UNIT = 2
WIRE_RUNS = 3
MAX_CANDIDATES = 8
COVER_LID_SHUT = 3
COVER_LID_OPEN = 3
ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MUJOCO_GL": "osmesa",
       "PYOPENGL_PLATFORM": "osmesa"}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def frozen_hashes():
    return {p: sha256_file(os.path.join(TASK, p)) for p in FROZEN}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args):
    return subprocess.run(["git", "-C", TASK] + list(args), capture_output=True, text=True, timeout=60, check=False)


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
    os.environ.update(ENV)
    for p in (ORACLE, TESTS):
        if p not in sys.path:
            sys.path.insert(0, p)              # TESTS first: verify, episode, lab_env, lab_sim (the oracle's identical)
    import lab_sim as S  # noqa: E402
    import verify as V  # noqa: E402
    return S, V


def _units():
    return dict(sorted(load(os.path.join(TESTS, "units.json"))["units"].items()))


def drawn(S, seed):
    """What the seed draws, for the record: the lid's start and the worklist's sizes (no simulation)."""
    ep = S.draw_unit(seed)
    return {"lid_closed_at_start": bool(ep.get("lid_closed")), "n_cold": len(ep["cold"]), "n_spin": len(ep["spin"]),
            "n_preloaded": len(ep.get("preloaded", {}))}


def submission(work):
    """The oracle submission as solution/solve.sh lays it out, with run.sh naming this interpreter."""
    if os.path.isdir(work):
        shutil.rmtree(work)
    shutil.copytree(ORACLE, work, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(os.path.join(TASK, "solution", "oracle_token"), os.path.join(work, ".oracle_token"))
    with open(os.path.join(work, "run.sh"), "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\n"
                "export %s\n"
                "exec %s -I -B oracle_main.py\n" % (" ".join("%s=%s" % kv for kv in sorted(ENV.items())),
                                                   sys.executable))
    os.chmod(os.path.join(work, "run.sh"), 0o755)
    return work


def over_wire(V, units, unit, seed, work):
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
    (scores, aborted, faults, _pu, per_part, _eg, _be, lock, records) = V.run_episodes(
        config, os.getuid(), "none", t0, units, work, True, [(unit, seed)])
    rec = records[0]
    return {"credit": float(scores[0]), "reason": rec["reason"], "steps": int(rec["steps"]), "aborted": int(aborted),
            "sim_faults": int(faults), "probes": int(lock.total), "probes_passed": int(lock.passed),
            "probes_idle_skipped": int(lock.idle_skipped),
            "parts": {k: v[0] for k, v in per_part.items() if v}, "wall_s": round(time.monotonic() - t0, 1)}


def run_unit(args):
    unit, key_hex, work_root, start = args
    S, V = _modules()
    units = _units()
    out, kept = [], 0
    for i in range(start, MAX_CANDIDATES):
        seed = candidate_seed(key_hex, unit, i)
        c = {"unit": unit, "index": i, "seed": seed, "drawn": drawn(S, seed), "wire": []}
        ok = True
        for k in range(WIRE_RUNS):
            w = over_wire(V, units, unit, seed, submission(os.path.join(work_root, "%s_%d_%d" % (unit, i, k))))
            c["wire"].append(w)
            if w["credit"] != 1.0:
                ok = False
                break
        c["status"] = "kept" if ok else "rejected"
        if not ok:
            c["lost"] = [dict(run=k, reason=w["reason"], steps=w["steps"], parts=w["parts"])
                         for k, w in enumerate(c["wire"]) if w["credit"] != 1.0]
        out.append(c)
        print("[select] %s candidate %d: %s (wire %s, steps %s)" % (
            unit, i, c["status"], [round(w["credit"], 3) for w in c["wire"]], [w["steps"] for w in c["wire"]]),
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
    units = sorted(_units())
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
    units = sorted(_units())
    chosen = []
    for u in units:
        kept = [c for c in rec["candidates"] if c["draw"] == d["draw"] and c["unit"] == u and c["status"] == "kept"]
        kept.sort(key=lambda c: c["index"])
        if len(kept) < PER_UNIT:
            raise SystemExit("unit %s has %d kept candidates: run again" % (u, len(kept)))
        chosen += kept[:PER_UNIT]
    shut = sum(c["drawn"]["lid_closed_at_start"] for c in chosen)
    if shut < COVER_LID_SHUT or len(chosen) - shut < COVER_LID_OPEN:
        raise SystemExit("the draw has the lid shut at the start in %d of %d episodes: void it (draw --void ...) and "
                         "run again" % (shut, len(chosen)))
    order = list(range(len(chosen)))
    for i in range(len(order) - 1, 0, -1):
        k = keyed(d["key"], "order|%d" % i) % (i + 1)
        order[i], order[k] = order[k], order[i]
    graded = [chosen[j] for j in order]
    tried = [c for c in rec["candidates"] if c["draw"] == d["draw"]]
    rejected = ["%s#%d (wire %s)" % (c["unit"], c["index"], [round(w["credit"], 3) for w in c["wire"]])
                for c in tried if c["status"] == "rejected"]
    d.update(status="made", made_utc=utc_now(), lid_shut_at_start=shut,
             graded_order=["%s#%d" % (c["unit"], c["index"]) for c in graded])
    save(a.record, rec)
    voided = [x for x in rec["draws"] if x["status"] == "void"]
    source = ("Selected %s by tests/review/select_set.py (the rule in its docstring), draw %d of %d (%d voided); key "
              "os.urandom(32), drawn %s, written to tests/hidden/episode_selection.json before any candidate ran. "
              "Frozen at the draw (sha256 in the record; commit %s): %s. Per unit, candidate seeds from the key in a "
              "fixed order; kept: the reference scored credit 1.0 in %d runs over the real wire "
              "(verify.run_episodes, the oracle submission, the probe lock); the first %d kept per unit. Tried %d, "
              "rejected %d: %s. The lid starts shut in %d of the %d. Graded order: a Fisher-Yates keyed by the same "
              "key.") % (
        d["made_utc"][:10], d["draw"], len(rec["draws"]), len(voided), d["drawn_utc"], rec["frozen_commit"][:12],
        ", ".join(FROZEN), WIRE_RUNS, PER_UNIT, len(tried), len(rejected), "; ".join(rejected) or "none", shut,
        len(chosen))
    out = {"format": SET_FORMAT, "episodes": [{"unit": c["unit"], "seed": c["seed"]} for c in graded],
           "source": source}
    save(a.out, out)
    print("wrote %d episodes to %s" % (len(graded), a.out))


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
    evidence = load(a.evidence)
    episodes = evidence.get("episodes", [])
    order = [name.split("#")[0] for name in d["graded_order"]]
    if [e.get("unit") for e in episodes] != order:
        raise SystemExit("the evidence must list the made set's episodes in the graded order: " + " ".join(order))
    bad = [n for n, e in enumerate(episodes) if float(e.get("credit", 0.0)) != 1.0 or e.get("reason") != "ok"]
    if bad:
        raise SystemExit("the reference no longer completes episode(s) %s: void the draw instead" % bad)
    # the commit that holds the frozen files as they are now: the last one that changed any of them
    commit = git("log", "-1", "--format=%H", "--", *FROZEN).stdout.strip()
    amendments = rec.setdefault("amendments", [])
    amendments.append({
        "amendment": len(amendments) + 1, "amended_utc": utc_now(), "draw": d["draw"], "commit": commit,
        "reason": a.reason, "changed": changed, "frozen_before": rec["frozen"],
        "frozen_commit_before": rec["frozen_commit"],
        "rule": "one full verify of the privileged reference on the made set as the grader runs it, with the frozen "
                "files of the commit; every episode scored credit 1.0 again",
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
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("make")
    p.add_argument("--record", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("amend")
    p.add_argument("--record", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--evidence", required=True, help="a JSON file: the full reference verify on the made set")
    a = ap.parse_args()
    {"draw": cmd_draw, "run": cmd_run, "make": cmd_make, "amend": cmd_amend}[a.cmd](a)


if __name__ == "__main__":
    main()
