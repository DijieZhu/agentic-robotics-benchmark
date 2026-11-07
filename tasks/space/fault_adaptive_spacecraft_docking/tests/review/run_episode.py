"""Grader-exact episodes of a submission (a maintainer's tool; the grader never runs it).

`verify.run_episode` with a `HostSolver`: the kit's Solver whose program is `bash run.sh` of the given submission
folder, started as the current user (no runuser, no seccomp filter), so a run.sh that names the image's
/opt/venv/bin/python3 needs a container of the task image. Whether the episodes are token runs is decided as the
grader decides it (`verify.token_state` against tests/hidden/oracle_token.sha256). Each episode's match lock comes
from its own stream (`verify.episode_stream`: the key and the seed).

    python3 -B run_episode.py --submission DIR --out FILE.jsonl [--count 20] [--rng 20261105] [--seeds S,S,...]
                              [--key HEX] [--one-process] [--compare] [--workers 4]

Default: one fresh program per episode (`--workers` of them in parallel). `--one-process`: every episode through one
program, in order, as the grader runs them (with --compare, the reset-hygiene check). `--compare`: each episode is
also run in process (oracle_batch.run_one, the privileged reference served without a pipe) and the replies and the
quality must be bit-identical. One JSON line per episode: seed, quality, success, reason, steps, probes, the slowest
reply (wall seconds on this machine), the sha256 of the reply sequence, and the comparison.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
sys.path.insert(0, TESTS)

import oracle_batch  # noqa: E402  (also puts solution/privileged on the path)
import verify  # noqa: E402
from robotics_grader import ProbeLock, Solver  # noqa: E402


class HostSolver(Solver):
    """The kit's Solver, its program started as the current user (the grader's GuardedSolver drops to the agent
    account behind a seccomp filter instead; the wire is the same)."""

    def __init__(self, *args, **kwargs):
        self.replies = []
        self.max_s = 0.0
        super().__init__(*args, **kwargs)

    def _spawn(self):
        return subprocess.Popen(["bash", self.entry], cwd=self.submission, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1,
                                start_new_session=True)

    def act(self, t, obs):
        asked = time.monotonic()
        action = super().act(t, obs)
        self.max_s = max(self.max_s, time.monotonic() - asked)
        self.replies.append(json.dumps(action))
        return action


def _lock(key_hex, seed):
    cfg = oracle_batch.config()
    lock = ProbeLock(None, rate=cfg.get("probe_rate", 0.06), pass_fraction=cfg.get("probe_pass_fraction", 0.35),
                     tolerance=1e-4)
    lock.rng = verify.episode_stream(int(key_hex, 16), seed)
    return lock


def episode(solver, seed, key_hex, privileged):
    """One episode through `solver` (a HostSolver): the record of the module docstring, with the replies."""
    first = len(solver.replies)
    solver.max_s = 0.0
    lock = _lock(key_hex, seed)
    quality, info, steps, _ = verify.run_episode(seed, solver, lock, privileged, verify.BlankCamera(),
                                                 int(key_hex, 16))
    replies = solver.replies[first:]
    return {"seed": seed, "quality": quality, "success": bool(info.get("success")),
            "reason": info.get("failure_reason", ""), "steps": steps, "probes": lock.total,
            "probes_passed": lock.passed, "replies": len(replies), "max_request_s": solver.max_s,
            "replies_sha256": hashlib.sha256("\n".join(replies).encode()).hexdigest(), "_replies": replies}


def compare(row, key_hex):
    ref = oracle_batch.run_one(row["seed"], key_hex, record=True)
    row["inprocess_quality"] = ref["quality"]
    row["inprocess_equal"] = ref["quality"] == row["quality"] and ref["replies"] == row["_replies"]
    return row


def _fresh(args):
    submission, seed, key_hex, privileged, timeout, do_compare = args
    solver = HostSolver(step_timeout=timeout, submission=submission)
    try:
        row = episode(solver, seed, key_hex, privileged)
    finally:
        solver.close()
    if do_compare:
        compare(row, key_hex)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--rng", type=int, default=20261105)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--key", default=oracle_batch.DEV_KEY)
    ap.add_argument("--one-process", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--step-timeout", type=float, default=30.0)
    a = ap.parse_args()
    sub = os.path.abspath(a.submission)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()] or oracle_batch.dev_seeds(a.count, a.rng)
    privileged, mismatch = verify.token_state(sub)
    print("token run: {} (mismatch {})".format(privileged, mismatch), file=sys.stderr)
    if a.one_process:
        solver = HostSolver(step_timeout=a.step_timeout, submission=sub)
        try:
            rows = [episode(solver, s, a.key, privileged) for s in seeds]
        finally:
            solver.close()
        if a.compare:
            rows = [compare(r, a.key) for r in rows]
    else:
        jobs = [(sub, s, a.key, privileged, a.step_timeout, a.compare) for s in seeds]
        with multiprocessing.Pool(a.workers) as pool:
            rows = list(pool.imap(_fresh, jobs, chunksize=1))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_replies"}) + "\n")
    ok = sum(r["success"] for r in rows)
    line = "episodes {}: docked {}; mean quality {:.6f}".format(len(rows), ok, sum(r["quality"] for r in rows) / len(rows))
    if a.compare:
        line += "; bit-identical to the in-process run: {} / {}".format(sum(r["inprocess_equal"] for r in rows), len(rows))
    print(line)


if __name__ == "__main__":
    main()
