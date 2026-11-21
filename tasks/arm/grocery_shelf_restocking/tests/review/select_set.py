"""Selection of the fixed hidden episode set (a maintainer's tool; the grader never runs it).

The rule, fixed before any candidate ran:

1. Freeze. The grader (tests/verify.py, tests/episode.py, tests/shelf_env.py, tests/shelf_world.py,
   tests/shelf_ctl.py, tests/shelf_sim.py, tests/box_transport_env.py, tests/units.json, tests/grader_config.json,
   and since draw 2 the launcher and the numerics fingerprint: tests/nonet_exec.py, tests/numerics_check.py,
   tests/numerics.json)
   and the reference (the solution/oracle/ modules, solution/solve.sh) are committed first: `draw` refuses to start while any of them differs
   from the checkout's HEAD and records their sha256; `run` and `make` refuse to go on when any of them
   differs from the recorded hashes.
2. Draw. `draw` takes the set's key from os.urandom(32) (hex) and writes it to the record before any candidate runs.
   Unit u's candidate i (i = 0, 1, 2, ...) has the i-th 64-bit seed, among
   int.from_bytes(blake2b(b"candidate|<u>|<j>", key=key, digest_size=8).digest(), "little") (j = 0, 1, 2, ...), whose
   episode solution/oracle/staging.keep keeps (the tote alone is not enough; the online strategy restocks every
   lane).
3. Run. Per unit, candidates in the order i = 0, 1, ...: the privileged reference three times over the real wire,
   stopping at the first run below credit 1.0: verify.run_episodes with that one episode and the oracle submission as
   solution/solve.sh lays it out (a subprocess speaking the grading wire, given the token truth), the probe lock at
   the grader's rate, each run with a fresh OS-entropy stream (robotics_grader.seed_rng). Until draw 1 the reference
   thought under a wall-clock limit per step (it held still while it had not answered), so its runs of one episode
   could differ; since draw 2 it has no clock of its own, and the three runs differ only in which steps are probed.
   It is run only as the grader runs it, as a separate process.
4. Keep. A candidate is kept when it scores credit 1.0 in each of the three wire runs; per unit the first two kept
   are the unit's episodes, at most MAX_CANDIDATES tried. Every candidate tried is recorded: kept, or rejected with
   its scores and the parts it lost.
5. Cover. Among the 12 kept episodes (shelf_world.draw_episode) every shelf is restocked in at least COVER_SHELF
   episodes, at least COVER_EXPIRED have an expired unit and at least COVER_INVERTED a delivered unit dated
   earlier than an old one of its lane. If not, the draw is void: `draw --void REASON` records why, draws a new key, and `run` starts again; the record keeps
   every draw. A draw is void too when a frozen file changes after it was made: `draw --void REASON --refreeze`
   records why, keeps that draw's frozen hashes and candidates with it, freezes the checkout's HEAD again
   (step 1) and draws a new key.
6. Order. `make` shuffles the 12 (units in sorted order, each unit's two in candidate order; a Fisher-Yates in which
   position i swaps with int.from_bytes(blake2b(b"order|<i>", key=key, digest_size=8).digest(), "little") mod (i + 1),
   i from 11 down to 1) and writes tests/hidden/episodes.json in that order: the graded order.

    python3 -B tests/review/select_set.py draw --record tests/hidden/episode_selection.json
    python3 -B tests/review/select_set.py run --record tests/hidden/episode_selection.json --work DIR [--workers 6]
    python3 -B tests/review/select_set.py make --record tests/hidden/episode_selection.json --out tests/hidden/episodes.json

`run` needs the task image (Python 3.12, MuJoCo 3.16.0, NumPy 2.5.4, SciPy 1.18.1, OpenCV; environment/
requirements.txt; the numerics pin of environment/Dockerfile, which the image's .pth gives every /opt/venv Python)
and an OSMesa renderer (MUJOCO_GL=osmesa). Outside the image the oracle submission's run.sh names
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

FORMAT = "grocery_shelf_restocking episode selection 1"
SET_FORMAT = "grocery_shelf_restocking fixed set 1"
ORACLE_MODULES = ("box_transport_env.py", "episode.py", "oracle_main.py", "reference.py", "shelf_ctl.py", "shelf_env.py",
                  "shelf_sim.py", "shelf_world.py", "staging.py", "wire_world.py")
FROZEN = (("tests/verify.py", "tests/episode.py", "tests/shelf_env.py", "tests/shelf_world.py", "tests/shelf_ctl.py",
           "tests/shelf_sim.py", "tests/box_transport_env.py", "tests/units.json", "tests/grader_config.json",
           "tests/nonet_exec.py", "tests/numerics_check.py", "tests/numerics.json")
          + tuple("solution/oracle/" + f for f in ORACLE_MODULES)
          + ("solution/solve.sh",))
PER_UNIT = 2
WIRE_RUNS = 3
MAX_CANDIDATES = 8
COVER_SHELF = 4
COVER_EXPIRED = 3
COVER_INVERTED = 3
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


def candidate_seed(key_hex, unit, i, unit_spec=None):
    """The i-th hash seed whose episode staging.keep keeps (needs the modules on sys.path)."""
    import shelf_world as S
    import staging as G
    if unit_spec is None:
        unit_spec = _units()[unit]
    n, j = -1, 0
    while True:
        seed = keyed(key_hex, "candidate|%s|%d" % (unit, j))
        if G.keep(unit_spec, S.draw_episode(seed, unit_spec))[0]:
            n += 1
            if n == i:
                return seed
        j += 1


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
            sys.path.insert(0, p)              # TESTS first: verify and the world files (the oracle's identical)
    import shelf_world as S  # noqa: E402
    import verify as V  # noqa: E402
    return S, V


def _units():
    return dict(sorted(load(os.path.join(TESTS, "units.json"))["units"].items()))


def drawn(S, unit, seed):
    """What the seed draws, for the record: the restocked lanes, their old units and deliveries (no simulation)."""
    ep = S.draw_episode(seed, unit)
    today = ep["today"]
    new = {k: [t["code"] for t in ep["tote"] if t["sku"] == ep["planogram"][k]] for k in ep["served"]}
    fresh = {k: [c for *_x, c in ep["old"][k] if c >= today] for k in ep["served"]}
    return {"served": ep["served"], "old": {k: len(v) for k, v in ep["old"].items()}, "deliver": ep["deliver"],
            "shelves": sorted({int(k.split(":")[0]) for k in ep["served"]}),
            "expired": sum(c < today for k in ep["served"] for *_x, c in ep["old"][k]),
            "inverted": sum(bool(fresh[k]) and min(new[k]) < max(fresh[k]) for k in ep["served"])}


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
    (scores, aborted, _skipped, faults, _pu, per_part, _eg, _be, lock, records) = V.run_episodes(
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
        seed = candidate_seed(key_hex, unit, i, units[unit])
        c = {"unit": unit, "index": i, "seed": seed, "drawn": drawn(S, units[unit], seed), "wire": []}
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
    if rec is not None:
        last = rec["draws"][-1]
        if a.refreeze:
            if not a.void or last["status"] not in ("open", "made"):
                raise SystemExit("--refreeze voids the latest open or made draw: pass --void REASON")
            last["frozen"] = rec["frozen"]
            rec["frozen"] = frozen_hashes()
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
               "frozen": frozen_hashes(), "draws": [], "candidates": []}
    else:
        check_frozen(rec)
    rec["draws"].append({"draw": len(rec["draws"]) + 1, "drawn_utc": utc_now(), "key": os.urandom(32).hex(),
                         "source": "os.urandom(32)", "status": "open"})
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
    per_shelf = [sum(s in c["drawn"]["shelves"] for c in chosen) for s in range(3)]
    expired = sum(c["drawn"]["expired"] > 0 for c in chosen)
    inverted = sum(c["drawn"]["inverted"] > 0 for c in chosen)
    if min(per_shelf) < COVER_SHELF or expired < COVER_EXPIRED or inverted < COVER_INVERTED:
        raise SystemExit("the draw restocks the shelves in %s episodes, has an expired unit in %d and an inverted lane "
                         "in %d: void it (draw --void ...) and run again" % (per_shelf, expired, inverted))
    order = list(range(len(chosen)))
    for i in range(len(order) - 1, 0, -1):
        k = keyed(d["key"], "order|%d" % i) % (i + 1)
        order[i], order[k] = order[k], order[i]
    graded = [chosen[j] for j in order]
    tried = [c for c in rec["candidates"] if c["draw"] == d["draw"]]
    rejected = ["%s#%d (wire %s)" % (c["unit"], c["index"], [round(w["credit"], 3) for w in c["wire"]])
                for c in tried if c["status"] == "rejected"]
    d.update(status="made", made_utc=utc_now(), shelves_restocked=per_shelf, with_expired=expired,
             with_inverted=inverted,
             graded_order=["%s#%d" % (c["unit"], c["index"]) for c in graded])
    save(a.record, rec)
    voided = [x for x in rec["draws"] if x["status"] == "void"]
    source = ("Selected %s by tests/review/select_set.py (the rule in its docstring), draw %d of %d (%d voided); key "
              "os.urandom(32), drawn %s, written to tests/hidden/episode_selection.json before any candidate ran. "
              "Frozen at the draw (sha256 in the record): %s. Per unit, candidate seeds from the key in a "
              "fixed order; kept: the reference scored credit 1.0 in %d runs over the real wire "
              "(verify.run_episodes, the oracle submission, the probe lock); the first %d kept per unit. Tried %d, "
              "rejected %d: %s. Episodes restocking shelf 0 / 1 / 2: %s; with an expired unit: %d, with a "
              "delivered unit dated before an old one: %d, of the %d. "
              "Graded order: a Fisher-Yates keyed by the same key.") % (
        d["made_utc"][:10], d["draw"], len(rec["draws"]), len(voided), d["drawn_utc"],
        ", ".join(FROZEN), WIRE_RUNS, PER_UNIT, len(tried), len(rejected), "; ".join(rejected) or "none",
        "/".join(map(str, per_shelf)), expired, inverted, len(chosen))
    out = {"format": SET_FORMAT, "episodes": [{"unit": c["unit"], "seed": c["seed"]} for c in graded],
           "source": source}
    save(a.out, out)
    print("wrote %d episodes to %s" % (len(graded), a.out))


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
    a = ap.parse_args()
    {"draw": cmd_draw, "run": cmd_run, "make": cmd_make}[a.cmd](a)


if __name__ == "__main__":
    main()
