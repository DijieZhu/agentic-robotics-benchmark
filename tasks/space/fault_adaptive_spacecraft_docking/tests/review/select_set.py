"""Pre-registered selection of the fixed hidden set (a maintainer's tool; the grader never runs it).

The rule, fixed before any candidate ran:

1. Freeze. The reference (solution/privileged/oracle.py, its simulator copy, solution/solve.sh) and the grader
   (tests/verify.py, tests/env.py) are committed first: `draw` refuses to start while any of them differs from the
   commit it records, and records their sha256; `run` and `make` refuse to go on when any of them, or the
   submission's copy of the reference, differs.
2. Draw. `draw` takes the candidate seeds from os.urandom (8 bytes each, little-endian; a draw below 2**32 or a
   duplicate is drawn again) and, with the first batch, the set's probe key (os.urandom(32), hex), and writes
   selection.json before anything runs. Batch 1 is 64 candidates. (The draw of 2026-11-05 also redrew the practice
   grader's seeds, as its record says; those were all below 2**13, and the practice tools are gone since.)
3. Run. Every candidate, in drawn order, goes through the grader's own episode loop (verify.run_episode): a token run
   of a submission that solution/solve.sh installed, a fresh program per candidate (run_episode.HostSolver), the
   candidate's own match-lock stream (verify.episode_stream: the set's key and its seed), the full 400-step horizon.
   Candidates run in parallel; their order is the draw index.
4. Keep. A candidate is kept when it docked (info["success"]) and its episode did not abort. The first 48 kept, in
   drawn order, are the set, in that order. Every candidate is recorded: kept; dropped, with its outcome; or ran after
   the 48th kept and not used. With fewer than 48 kept, `draw --batch 2` adds 32 more candidates once batch 1 is done.
5. Record. Each candidate's world (the hidden draws the hollowing check compares; `report` prints the check) and,
   from an in-process replay of the same episode (which must match the grader-exact run bit for bit), how many of
   the reference's policies (every one of its family) dock from its planning state.

    python3 -B select_set.py draw --selection SEL.json [--batch 1] [--count 64]
    python3 -B select_set.py run --selection SEL.json --submission DIR [--workers 8]
    python3 -B select_set.py make --selection SEL.json --out EPISODES.json [--size 48]
    python3 -B select_set.py report --selection SEL.json

`run` needs the interpreter run.sh names (/opt/venv/bin/python3): run it in a container of the task image. Paths
are found from this file, so it runs from any checkout.
"""

import argparse
import datetime
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
sys.path.insert(0, TESTS)

import oracle_batch  # noqa: E402
import run_episode  # noqa: E402
import verify  # noqa: E402

O = oracle_batch.O
SEL_FORMAT = "fault_adaptive_spacecraft_docking selection 1"
FROZEN = ("solution/privileged/oracle.py", "solution/privileged/spacecraft_docking_env.py", "solution/solve.sh",
          "tests/env.py", "tests/verify.py")
BATCH_SIZE = {1: 64}
LATER_BATCH = 32
SET_SIZE = 48


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def frozen_hashes():
    return {p: sha256_file(os.path.join(TASK, p)) for p in FROZEN}


def load(path):
    with open(path) as f:
        return json.load(f)


def save(path, sel):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sel, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head():
    try:
        return subprocess.run(["git", "-C", TASK, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def frozen_committed():
    """True when every frozen file is tracked and equals its committed version (git, run in the task folder)."""
    try:
        tracked = subprocess.run(["git", "-C", TASK, "ls-files", "--error-unmatch", "--"] + list(FROZEN),
                                 capture_output=True, timeout=30)
        clean = subprocess.run(["git", "-C", TASK, "diff", "--quiet", "HEAD", "--"] + list(FROZEN), timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return tracked.returncode == 0 and clean.returncode == 0


# ------------------------------------------------------------------------------------------------ the world
def _docking_columns(env, swirl):
    """The near-field columns at the docking pose (on the port, facing it: the body's +x points radially outward)."""
    cols = []
    for n in range(6):
        fx, fy, tq = env.columns[n]
        a = max(0.0, fx / max(math.hypot(fx, fy), 1e-9))
        loss = env.plume_strength * (0.35 + 0.65 * a ** 4)
        cols.append(((1.0 - loss) * fx - swirl[n] * loss * fy, (1.0 - loss) * fy + swirl[n] * loss * fx,
                     (1.0 - loss) * tq))
    return cols


def outward_authority(env, swirl):
    """The largest outward acceleration (m/s^2, bias included) the re-keyed plant can produce at the docking pose with
    no tangential and no yaw acceleration, realized thrust in [0, 1] per nozzle (a small LP, solved over its
    vertices); None when it cannot even cancel the bias."""
    scale = [1.0] * 6
    scale[env.rekey_failure_nozzle] = env.rekey_failure_scale
    scale[env.rekey_secondary_failure_nozzle] = env.rekey_secondary_failure_scale
    cols = [tuple(c * s for c in col) for col, s in zip(_docking_columns(env, swirl), scale)]
    bias = env.bias
    best = None
    for i, j in itertools.combinations(range(6), 2):
        a11, a12, a21, a22 = cols[i][1], cols[j][1], cols[i][2], cols[j][2]
        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-12:
            continue
        rest = [k for k in range(6) if k not in (i, j)]
        for bits in itertools.product((0.0, 1.0), repeat=4):
            r1 = -bias[1] - sum(cols[k][1] * b for k, b in zip(rest, bits))
            r2 = -bias[2] - sum(cols[k][2] * b for k, b in zip(rest, bits))
            wi = (r1 * a22 - a12 * r2) / det
            wj = (a11 * r2 - a21 * r1) / det
            if -1e-9 <= wi <= 1 + 1e-9 and -1e-9 <= wj <= 1 + 1e-9:
                val = bias[0] + cols[i][0] * wi + cols[j][0] * wj + sum(cols[k][0] * b for k, b in zip(rest, bits))
                best = val if best is None or val > best else best
    return best


def world_summary(seed):
    env = verify.FaultAdaptiveDocking()
    env.reset(seed)
    swirl0 = list(env.plume_swirl)
    patterns = 0
    for flips in itertools.product((1.0, -1.0), repeat=6):
        a = outward_authority(env, [s * f for s, f in zip(swirl0, flips)])
        patterns += int(a is not None and a >= 0.0)
    start_bearing = (math.atan2(env.y, env.x) - env.phase + math.pi) % (2 * math.pi) - math.pi
    return {
        "failure_step": env.failure_step, "rekey_earliest": env.rekey_earliest, "rekey_deadline": env.rekey_deadline,
        "rekey_proximity": env.rekey_proximity, "failure_scale": env.failure_scale,
        "secondary_failure_scale": env.secondary_failure_scale, "rekey_failure_scale": env.rekey_failure_scale,
        "rekey_secondary_failure_scale": env.rekey_secondary_failure_scale,
        "rekey_pair_shares_a_nozzle": len({env.failure_nozzle, env.secondary_failure_nozzle}
                                          & {env.rekey_failure_nozzle, env.rekey_secondary_failure_nozzle}) > 0,
        "port_amplitude": env.port_amplitude, "port_omega": env.port_omega,
        "start_radius": math.hypot(env.x, env.y), "start_bearing_from_port": start_bearing,
        "phase_rate": env.phase_rate, "rekey_phase_rate": env.rekey_phase_rate,
        "plume_parity": 1 if env.plume_swirl[0] > 0 else -1,
        "outward_authority_initial": outward_authority(env, swirl0),
        "swirl_patterns_with_outward_authority": patterns,
    }


# ------------------------------------------------------------------------------------------------ the runs
def _run_candidate(args):
    submission, seed, key_hex = args
    solver = run_episode.HostSolver(step_timeout=30.0, submission=submission)
    try:
        row = run_episode.episode(solver, seed, key_hex, True)
    finally:
        solver.close()
    oracle = O.Oracle()
    rep = oracle_batch.run_one(seed, key_hex, oracle=oracle, record=True)
    pl = oracle.episode.committed().planner
    docking = None
    if pl is not None and pl.base is not None and pl.base.steps == pl.t_p:
        docking = 0
        for v in O.FAMILY:
            r, gap = O.clone(pl.base), pl.base_gap
            start = pl.t_p + v.delay
            while True:
                u, gap = O.law(r, v, r.steps >= start, gap)
                _, _, done, _ = r.step(u)
                if done or r.steps + O.HOLD - r.hold_steps > O.STEPS:
                    break
            docking += int(bool(r.success))
    return {
        "quality": row["quality"], "success": row["success"], "reason": row["reason"], "steps": row["steps"],
        "aborted": str(row["reason"]).startswith("aborted"), "probes": row["probes"],
        "probes_passed": row["probes_passed"], "replies_sha256": row["replies_sha256"],
        "inprocess_equal": rep["quality"] == row["quality"] and rep["replies"] == row["_replies"],
        "t_ready": rep["t_ready"], "rekey_step": rep["rekey_step"], "chosen": rep["chosen"], "tried": rep["tried"],
        "variant": rep["variant"], "planned_equals_executed": rep["planned_equals_executed"],
        "docking_policies": docking, "world_sha256": verify.world_digest(seed), "world": world_summary(seed),
    }


def cmd_draw(a):
    if a.batch == 1:
        if os.path.exists(a.selection):
            raise SystemExit("{} exists: batch 1 starts a selection".format(a.selection))
        if not frozen_committed():
            raise SystemExit("the frozen files are not committed as they are: commit the freeze first")
        sel = {"format": SEL_FORMAT, "probe_key": os.urandom(32).hex(), "frozen": frozen_hashes(),
               "frozen_commit": git_head(), "rule": "tests/review/select_set.py (module docstring)",
               "batches": [], "candidates": []}
    else:
        sel = load(a.selection)
        if sel["frozen"] != frozen_hashes():
            raise SystemExit("the frozen files changed since batch 1: a new selection, not a new batch")
        if any(c["status"] == "drawn" for c in sel["candidates"]):
            raise SystemExit("an earlier batch has not run yet")
    count = a.count or BATCH_SIZE.get(a.batch, LATER_BATCH)
    seen = {c["seed"] for c in sel["candidates"]}
    redrawn, new = 0, []
    while len(new) < count:
        s = int.from_bytes(os.urandom(8), "little")
        if s < (1 << 32) or s in seen:
            redrawn += 1
            continue
        seen.add(s)
        new.append(s)
    for s in new:
        sel["candidates"].append({"draw": len(sel["candidates"]), "batch": a.batch, "seed": s, "status": "drawn"})
    sel["batches"].append({"batch": a.batch, "drawn_utc": utc_now(), "count": count, "redrawn": redrawn,
                           "source": "os.urandom(8) per seed, little-endian"})
    save(a.selection, sel)
    print("batch {}: {} candidates drawn ({} redrawn) -> {}".format(a.batch, count, redrawn, a.selection))


def cmd_run(a):
    sel = load(a.selection)
    now = frozen_hashes()
    if sel["frozen"] != now:
        raise SystemExit("the frozen files changed since the draw: {}".format(
            [p for p in FROZEN if sel["frozen"][p] != now[p]]))
    sub = os.path.abspath(a.submission)
    for name, frozen in (("oracle.py", "solution/privileged/oracle.py"),
                         ("spacecraft_docking_env.py", "solution/privileged/spacecraft_docking_env.py")):
        if sha256_file(os.path.join(sub, name)) != sel["frozen"][frozen]:
            raise SystemExit("the submission's {} is not the frozen one".format(name))
    if verify.token_state(sub) != (True, False):
        raise SystemExit("the submission carries no matching reference token")
    todo = [c for c in sel["candidates"] if c["status"] == "drawn"]
    jobs = [(sub, c["seed"], sel["probe_key"]) for c in todo]
    started = utc_now()
    with multiprocessing.Pool(a.workers) as pool:
        for c, res in zip(todo, pool.imap(_run_candidate, jobs, chunksize=1)):
            c.update(res)
            c["status"] = "ran"
            print("draw {:3d}: {} {:>26s} steps={} q={:.4f} chosen={} docking_policies={} same={}".format(
                c["draw"], "ok  " if c["success"] else "DROP", c["reason"], c["steps"], c["quality"], c["chosen"],
                c["docking_policies"], c["inprocess_equal"]), flush=True)
    sel.setdefault("runs", []).append({"started_utc": started, "finished_utc": utc_now(), "candidates": len(todo),
                                       "python": sys.version.split()[0], "workers": a.workers})
    save(a.selection, sel)


def classify(sel, size):
    kept, dropped, after = [], [], []
    for c in sorted(sel["candidates"], key=lambda c: c["draw"]):
        if c["status"] == "drawn":
            raise SystemExit("draw {} has not run".format(c["draw"]))
        if len(kept) == size:
            after.append(c)
        elif c["success"] and not c["aborted"]:
            kept.append(c)
        else:
            dropped.append(c)
    return kept, dropped, after


def cmd_make(a):
    sel = load(a.selection)
    if sel["frozen"] != frozen_hashes():
        raise SystemExit("the frozen files changed since the draw")
    kept, dropped, after = classify(sel, a.size)
    if len(kept) < a.size:
        raise SystemExit("only {} kept: draw another batch".format(len(kept)))
    if not all(c["inprocess_equal"] for c in kept):
        raise SystemExit("a kept candidate's in-process replay differs from its grader-exact run")
    for c in kept:
        c["status"] = "kept"
    for c in dropped:
        c["status"] = "dropped"
    for c in after:
        c["status"] = "after_cutoff"
    batches = "; ".join("batch {} drawn {} ({} seeds, {} redrawn)".format(b["batch"], b["drawn_utc"], b["count"],
                                                                         b["redrawn"]) for b in sel["batches"])
    frozen = ", ".join("{} {}".format(p, h) for p, h in sel["frozen"].items())
    drops = "; ".join("draw {} ({}, step {}, quality {:.4f})".format(c["draw"], c["reason"], c["steps"], c["quality"])
                      for c in dropped) or "none"
    source = (
        "Selected {}. Candidates: {}, from os.urandom (8 bytes per seed, little-endian; a draw below 2**32 or a "
        "duplicate drawn again), with the probe key (os.urandom(32)), written to selection.json before any "
        "candidate ran. Frozen before the draw (sha256; commit {}): {}. Rule (tests/review/select_set.py): "
        "every candidate in drawn order through verify.run_episode, a token run of the solve.sh submission, a fresh "
        "program per candidate, its own lock stream (the key and its seed), the full 400-step horizon; kept if it "
        "docked without an abort; the first {} kept, in drawn order, are the set, in that order. Kept draws: {}. "
        "Dropped: {}. Ran after the {}th kept, not used: draws {}. Not run: none. Every candidate with its outcome and "
        "world: selection.json."
    ).format(utc_now()[:10], batches, (sel.get("frozen_commit") or "?")[:12], frozen, a.size,
             ", ".join(str(c["draw"]) for c in kept), drops, a.size,
             ", ".join(str(c["draw"]) for c in after) or "none")
    episodes = {"format": verify.SET_FORMAT, "probe_key": sel["probe_key"],
                "episodes": [{"seed": c["seed"], "world_sha256": c["world_sha256"]} for c in kept], "source": source}
    with open(a.out, "w") as f:
        json.dump(episodes, f, indent=1)
        f.write("\n")
    sel["set"] = {"made_utc": utc_now(), "size": a.size, "kept": [c["draw"] for c in kept],
                  "dropped": [c["draw"] for c in dropped], "after_cutoff": [c["draw"] for c in after]}
    save(a.selection, sel)
    print("set: {} kept, {} dropped, {} after the cutoff -> {}".format(len(kept), len(dropped), len(after), a.out))


RANGES = {  # instruction.md's ranges (env.py's draws), for the hollowing check
    "failure_step": (58, 103), "rekey_earliest": (108, 128), "rekey_deadline": (148, 165),
    "rekey_proximity": (1.18, 1.30), "failure_scale": (0.35, 0.50), "secondary_failure_scale": (0.55, 0.70),
    "rekey_failure_scale": (0.45, 0.60), "rekey_secondary_failure_scale": (0.65, 0.80),
    "port_amplitude": (0.10, 0.16), "port_omega": (0.18, 0.28), "start_radius": (2.85, 3.65),
    "start_bearing_from_port": (-1.45, 1.45), "phase_rate": (-0.125, 0.125), "rekey_phase_rate": (-0.045, 0.045),
}


def cmd_report(a):
    sel = load(a.selection)
    kept, dropped, after = classify(sel, a.size)
    print("kept {}, dropped {}, after the cutoff {}".format(len(kept), len(dropped), len(after)))
    keys = list(RANGES) + ["outward_authority_initial", "swirl_patterns_with_outward_authority"]
    print("{:34s} {:>22s} {:>22s} {:>22s}".format("parameter", "range", "kept min..max", "dropped values"))
    for k in keys:
        kv = [c["world"][k] for c in kept if c["world"][k] is not None]
        dv = [c["world"][k] for c in dropped]
        rng = "{}..{}".format(*RANGES[k]) if k in RANGES else "-"
        fmt = (lambda x: "None" if x is None else ("{:.3f}".format(x) if isinstance(x, float) else str(x)))
        print("{:34s} {:>22s} {:>22s} {}".format(k, rng, "{}..{}".format(fmt(min(kv)), fmt(max(kv))) if kv else "-",
                                                 ", ".join(fmt(x) for x in dv)))
    for c in dropped:
        print("drop draw {}: {} steps={} q={:.4f} rekey={} t_ready={} tried={} docking_policies={} world={}".format(
            c["draw"], c["reason"], c["steps"], c["quality"], c["rekey_step"], c["t_ready"], c["tried"],
            c["docking_policies"], json.dumps(c["world"], sort_keys=True)))
    dk = sorted(c["docking_policies"] for c in kept if c["docking_policies"] is not None)
    if dk:
        print("kept: policies (of {}) that dock from the planning state: min {} median {} max {}".format(
            len(O.FAMILY), dk[0], dk[len(dk) // 2], dk[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("draw")
    p.add_argument("--selection", required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--count", type=int, default=0)
    p.set_defaults(fn=cmd_draw)
    p = sub.add_parser("run")
    p.add_argument("--selection", required=True)
    p.add_argument("--submission", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("make")
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=SET_SIZE)
    p.set_defaults(fn=cmd_make)
    p = sub.add_parser("report")
    p.add_argument("--selection", required=True)
    p.add_argument("--size", type=int, default=SET_SIZE)
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
