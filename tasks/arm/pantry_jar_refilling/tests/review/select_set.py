"""Selection of the hidden units and the fixed hidden episode set (a maintainer's tool; the grader never runs it).

The rule, fixed before any unit or candidate was drawn:

0. Units. `units` takes a key from os.urandom(32) (hex), writes it with the rule to tests/hidden/units_selection.json,
   and writes tests/units.json: hidden unit u<k> (k = 0..5) is jar_sim.draw_unit(seed) for the 64-bit seed
   int.from_bytes(blake2b(b"unit|<k>", key=key, digest_size=8).digest(), "little"). It refuses to run twice.
1. Freeze. The grader's episode loop and world (tests/jar_grade.py, tests/episode.py, tests/jar_env.py,
   tests/jar_sim.py, tests/units.json, tests/grader_config.json; verify.py only stages the submission around
   jar_grade.run_episodes, which is what selection runs) and the reference (the solution/oracle/ modules, solution/solve.sh) are
   committed first: `draw` refuses to start while any of them differs from the checkout's HEAD and records their
   sha256 and the commit; `run` and `make` refuse to go on when any of them differs from the recorded hashes.
2. Draw. `draw` takes the set's key from os.urandom(32) (hex) and writes it to the record before any candidate runs.
   Unit u's candidate i (i = 0, 1, 2, ...) has the 64-bit seed
   int.from_bytes(blake2b(b"candidate|<u>|<i>", key=key, digest_size=8).digest(), "little").
3. Run. Candidates i = 0..BATCH-1 of every unit, then the next BATCH of a unit that has fewer than two kept, up to
   MAX_CANDIDATES: the privileged reference three times over the real wire (jar_grade.run_episodes with that one
   episode and the oracle submission as solution/solve.sh lays it out, given the token truth; the probe lock at the
   grader's rate with a fresh OS-entropy stream per run), stopping at the first run below credit 1.0.
4. Keep. A candidate is kept when it scores credit 1.0 in each of the three wire runs; per unit the first two kept in
   candidate order are the unit's episodes. Every candidate tried is recorded: kept, or rejected with its scores.
5. Cover. Among the 12 kept episodes every place of the row (left, middle, right) holds a listed jar in at least
   COVER_PLACE episodes. If not, or if a unit has fewer than two kept within MAX_CANDIDATES, the draw is void:
   `draw --void REASON` records why and draws a new key; the record keeps every draw.
6. Order. `make` shuffles the 12 (units in sorted order, each unit's two in candidate order; a Fisher-Yates in which
   position i swaps with int.from_bytes(blake2b(b"order|<i>", key=key, digest_size=8).digest(), "little") mod (i + 1),
   i from 11 down to 1) and writes tests/hidden/episodes.json in that order: the graded order.

    python3 -B tests/review/select_set.py units
    python3 -B tests/review/select_set.py draw --record tests/hidden/episode_selection.json
    python3 -B tests/review/select_set.py run --record tests/hidden/episode_selection.json --work DIR [--workers 24]
    python3 -B tests/review/select_set.py make --record tests/hidden/episode_selection.json --out tests/hidden/episodes.json

`run` needs the task image's Python (3.12, MuJoCo 3.16.0, NumPy 2.5.4; environment/requirements.txt, and robosuite
1.5.2's model files) and an OSMesa renderer (MUJOCO_GL=osmesa). Outside the image the oracle submission's run.sh names
this interpreter, and the controller is started directly rather than through nonet_exec.py (which needs root to change
user); everything else is jar_grade.run_episodes as the grader runs it.
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
UNITS_RECORD = os.path.join(TESTS, "hidden", "units_selection.json")
sys.dont_write_bytecode = True

FORMAT = "pantry_jar_refilling episode selection 1"
SET_FORMAT = "pantry_jar_refilling fixed set 1"
N_UNITS = 6
ORACLE_MODULES = ("episode.py", "jar_env.py", "jar_sim.py", "oracle_main.py", "reference.py", "wire_world.py")
FROZEN = (("tests/jar_grade.py", "tests/episode.py", "tests/jar_env.py", "tests/jar_sim.py",
           "tests/units.json", "tests/grader_config.json")
          + tuple("solution/oracle/" + f for f in ORACLE_MODULES) + ("solution/solve.sh",))
PER_UNIT = 2
WIRE_RUNS = 3
BATCH = 4
MAX_CANDIDATES = 12
COVER_PLACE = 4
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


def _modules():
    os.environ.update(ENV)
    for p in (ORACLE, TESTS):
        if p not in sys.path:
            sys.path.insert(0, p)              # TESTS first: jar_grade, episode, jar_env, jar_sim (the oracle's identical)
    import jar_grade as G  # noqa: E402
    import jar_sim as S  # noqa: E402
    return S, G


def _units():
    return dict(sorted(load(os.path.join(TESTS, "units.json"))["units"].items()))


def drawn(S, seed, unit):
    """What the seed draws, for the record (no simulation): the listed jars' places (0 left, 1 middle, 2 right) and
    their kinds and counts."""
    ep = S.draw_episode(seed, unit)
    ys = {int(j): p[1] for j, p in ep["places"].items()}
    order = sorted(ys, key=lambda j: -ys[j])                 # left (+y) to right
    return {"listed_places": sorted(order.index(j) for j in ep["listed"]), "refill": ep["refill"]}


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


def over_wire(G, units, unit, seed, work):
    from robotics_grader import Solver, seed_rng

    class HostSolver(Solver):
        def _spawn(self):
            return subprocess.Popen(["bash", self.entry], cwd=self.submission, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                                    bufsize=1, start_new_session=True)

    # step 3: each wire run with a fresh OS-entropy stream, not the grader's per-episode stream (probe_stream keys on
    # the episode's index in the graded order, which `make` decides later)
    G.probe_stream = lambda _index, _seed: seed_rng()
    config = dict(G.CONFIG, **{k: v for k, v in load(os.path.join(TESTS, "grader_config.json")).items()
                               if k in G.CONFIG})
    config["grader_budget_s"] = None
    t0 = time.monotonic()
    solver = HostSolver(step_timeout=config["step_timeout_s"], submission=work)
    try:
        records, lock = G.run_episodes(solver, [(0, unit, seed)], units, True, config=config)
    finally:
        solver.close()
    r = records[0]
    sc = r.get("score") or {}
    return {"credit": float(r["credit"]), "reason": r["reason"], "steps": int(r["steps"]),
            "sim_fault": bool(r["sim_fault"]), "probes": int(lock.total), "probes_passed": int(lock.passed),
            "probes_idle_skipped": int(lock.idle_skipped),
            "jars": {k: {"pieces_in": v["pieces_in"], "wrong_in": v["wrong_in"], "sealed": v["sealed"]}
                     for k, v in sc.get("jars", {}).items()},
            "halved": bool(sc.get("penalized", False)), "wall_s": round(time.monotonic() - t0, 1)}


def run_candidate(args):
    unit, i, key_hex, work_root = args
    S, G = _modules()
    units = _units()
    seed = candidate_seed(key_hex, unit, i)
    c = {"unit": unit, "index": i, "seed": seed, "drawn": drawn(S, seed, units[unit]), "wire": []}
    ok = True
    for k in range(WIRE_RUNS):
        w = over_wire(G, units, unit, seed, submission(os.path.join(work_root, "%s_%d_%d" % (unit, i, k))))
        c["wire"].append(w)
        if w["credit"] != 1.0:
            ok = False
            break
    c["status"] = "kept" if ok else "rejected"
    print("[select] %s candidate %d: %s (wire %s, steps %s)" % (
        unit, i, c["status"], [round(w["credit"], 3) for w in c["wire"]], [w["steps"] for w in c["wire"]]),
        file=sys.stderr, flush=True)
    return c


# ------------------------------------------------------------------------------------------- commands
def cmd_units(_a):
    if os.path.exists(UNITS_RECORD):
        raise SystemExit("the hidden units are drawn already (%s)" % UNITS_RECORD)
    S, _G = _modules()
    key = os.urandom(32).hex()
    seeds = {"u%d" % k: keyed(key, "unit|%d" % k) for k in range(N_UNITS)}
    save(UNITS_RECORD, {"rule": "tests/review/select_set.py (module docstring, step 0)", "drawn_utc": utc_now(),
                        "key": key, "source": "os.urandom(32)", "seeds": seeds})
    units = {u: S.draw_unit(s) for u, s in seeds.items()}
    save(os.path.join(TESTS, "units.json"), {"units": units})
    print("wrote %d hidden units" % len(units))


def cmd_draw(a):
    rec = load(a.record) if os.path.exists(a.record) else None
    if not frozen_committed():
        raise SystemExit("commit the frozen files first: " + ", ".join(FROZEN))
    head = git("rev-parse", "HEAD").stdout.strip()
    if rec is not None:
        last = rec["draws"][-1]
        if last["status"] == "made":
            raise SystemExit("the latest draw is made")
        if last["status"] == "open" and not a.void:
            raise SystemExit("the latest draw is open: pass --void REASON to void it")
        if a.void and last["status"] != "void":
            last.update(status="void", void_reason=a.void, voided_utc=utc_now())
        check_frozen(rec)
    else:
        rec = {"format": FORMAT, "rule": "tests/review/select_set.py (module docstring)",
               "frozen": frozen_hashes(), "frozen_commit": head, "draws": [], "candidates": []}
    rec["draws"].append({"draw": len(rec["draws"]) + 1, "drawn_utc": utc_now(), "key": os.urandom(32).hex(),
                         "source": "os.urandom(32)", "frozen_commit": rec["frozen_commit"], "status": "open"})
    save(a.record, rec)
    print("draw %d written to %s" % (len(rec["draws"]), a.record))


def cmd_run(a):
    rec = load(a.record)
    check_frozen(rec)
    d = open_draw(rec)
    units = sorted(_units())
    os.makedirs(a.work, exist_ok=True)
    while True:
        mine = [c for c in rec["candidates"] if c["draw"] == d["draw"]]
        todo = []
        for u in units:
            cs = sorted((c for c in mine if c["unit"] == u), key=lambda c: c["index"])
            if sum(c["status"] == "kept" for c in cs) >= PER_UNIT:
                continue
            start = len(cs)
            todo += [(u, i, d["key"], a.work) for i in range(start, min(start + BATCH, MAX_CANDIDATES))]
        if not todo:
            break
        with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
            for c in pool.imap_unordered(run_candidate, todo):
                rec = load(a.record)
                rec["candidates"].append(dict(c, draw=d["draw"]))
                rec["candidates"].sort(key=lambda c: (c["draw"], c["unit"], c["index"]))
                save(a.record, rec)
    for u in units:
        cs = [c for c in rec["candidates"] if c["draw"] == d["draw"] and c["unit"] == u]
        print("%s: %d candidates, %d kept" % (u, len(cs), sum(c["status"] == "kept" for c in cs)), flush=True)


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
            raise SystemExit("unit %s has %d kept candidates: void the draw" % (u, len(kept)))
        chosen += kept[:PER_UNIT]
    places = [sum(p in c["drawn"]["listed_places"] for c in chosen) for p in range(3)]
    if min(places) < COVER_PLACE:
        raise SystemExit("listed jars by place %s: below %d, void the draw" % (places, COVER_PLACE))
    order = list(range(len(chosen)))
    for i in range(len(order) - 1, 0, -1):
        k = keyed(d["key"], "order|%d" % i) % (i + 1)
        order[i], order[k] = order[k], order[i]
    graded = [chosen[j] for j in order]
    tried = [c for c in rec["candidates"] if c["draw"] == d["draw"]]
    rejected = ["%s#%d (wire %s)" % (c["unit"], c["index"], [round(w["credit"], 3) for w in c["wire"]])
                for c in tried if c["status"] == "rejected"]
    d.update(status="made", made_utc=utc_now(), listed_by_place=places,
             graded_order=["%s#%d" % (c["unit"], c["index"]) for c in graded])
    save(a.record, rec)
    voided = [x for x in rec["draws"] if x["status"] == "void"]
    source = ("Selected %s by tests/review/select_set.py (the rule in its docstring), draw %d of %d (%d voided); key "
              "os.urandom(32), drawn %s, written to tests/hidden/episode_selection.json before any candidate ran. "
              "Frozen at the draw (sha256 in the record): %s. Per unit, candidate seeds from the key in a fixed "
              "order; kept: the reference scored credit 1.0 in %d runs over the real wire (jar_grade.run_episodes, "
              "the oracle submission, the probe lock); the first %d kept per unit. Tried %d, rejected %d: %s. Listed "
              "jars by place (left, middle, right): %s. Graded order: a Fisher-Yates keyed by the same key.") % (
        d["made_utc"][:10], d["draw"], len(rec["draws"]), len(voided), d["drawn_utc"], ", ".join(FROZEN), WIRE_RUNS,
        PER_UNIT, len(tried), len(rejected), "; ".join(rejected) or "none", places)
    out = {"format": SET_FORMAT, "episodes": [{"unit": c["unit"], "seed": c["seed"]} for c in graded],
           "source": source}
    save(a.out, out)
    print("wrote %d episodes to %s" % (len(graded), a.out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("units")
    p = sub.add_parser("draw")
    p.add_argument("--record", required=True)
    p.add_argument("--void", default=None, help="void the open draw, with this reason")
    p = sub.add_parser("run")
    p.add_argument("--record", required=True)
    p.add_argument("--work", required=True, help="a scratch folder for the oracle submissions")
    p.add_argument("--workers", type=int, default=24)
    p = sub.add_parser("make")
    p.add_argument("--record", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"units": cmd_units, "draw": cmd_draw, "run": cmd_run, "make": cmd_make}[a.cmd](a)


if __name__ == "__main__":
    main()
