"""Dev batch of the privileged reference, in process (a maintainer's tool; the grader never runs it).

The grader's own episode loop (`verify.run_episode`, a token run) with the reference (solution/privileged/oracle.py)
served in this process: every message goes through a JSON round trip, as on the wire; the match lock is real, each
episode's stream keyed by a dev key and the episode's seed (`verify.episode_stream`). The seeds are dev seeds, never
the hidden set's: 64-bit draws of random.Random(--rng), each >= 2**32.

    python3 -B oracle_batch.py --out results.jsonl [--count 2000] [--rng 20261105] [--workers 8] [--key HEX]
    python3 -B oracle_batch.py --out results.jsonl --seeds 4294967296,18446744073709551557

Run it with Python 3.12 or newer, as the grader is (3.11's builtin sum differs in fuel's last bit). One JSON line per
episode, in seed order: quality, success, reason, steps; the first event's step, the re-key's step and its window;
t_ready, the policy chosen, every policy rolled out (docked, steps, quality); simulated planning steps; the slowest
request (wall seconds on this machine); probes total and passed; the planned and the executed outcome; steps after
t_ready with no verified request. The summary goes to stdout. Paths are found from this file, so it runs from any
checkout.

The planner's invariant is that a planned dock is an executed dock: when the policy the reference commits docks in its
rollout, the episode docks at the same step with the same quality, and when no policy docks, the episode does not dock
either (`planned_equals_executed`). It says nothing more about an episode without a planned dock: the reference then
keeps the policy with the best quality among rollouts that stop as soon as a dock can no longer be completed in time
(at about step 371), so that fallback's last steps are never simulated, and the episode can still end in a collision
or an escape that its rollout did not reach.
"""

import argparse
import collections
import hashlib
import json
import multiprocessing
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
sys.dont_write_bytecode = True
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TASK, "solution", "privileged"))

import oracle as O  # noqa: E402
import verify  # noqa: E402
from robotics_grader import ProbeLock, SolverAborted  # noqa: E402

DEV_KEY = hashlib.sha256(b"fault_adaptive_spacecraft_docking dev lock key").hexdigest()


class RecordingEnv(verify.FaultAdaptiveDocking):
    """The grader's simulator, remembered after its episode (the review reads the hidden draws and the re-key step)."""

    last = None

    def reset(self, seed):
        RecordingEnv.last = self
        return super().reset(seed)


verify.FaultAdaptiveDocking = RecordingEnv


class InProcess(object):
    """The kit Solver's protocol, served by an Oracle in this process (each message JSON round-tripped)."""

    def __init__(self, oracle, record=False):
        self.oracle = oracle
        self.max_s = 0.0
        self.requests = 0
        self.record = record
        self.replies = []

    def _send(self, msg):
        line = json.dumps(msg)
        t0 = time.perf_counter()
        out = self.oracle.handle(line)
        dt = time.perf_counter() - t0
        return json.loads(out), dt

    def reset(self):
        self._send({"type": "reset"})

    def act(self, t, obs):
        reply, dt = self._send({"type": "act", "t": int(t), "obs": obs})
        self.requests += 1
        self.max_s = max(self.max_s, dt)
        if "action" not in reply:
            raise SolverAborted("solver error: {}".format(reply.get("error", "no action")))
        if self.record:
            self.replies.append(json.dumps(reply["action"]))
        return reply["action"]


def config():
    with open(os.path.join(TESTS, "grader_config.json")) as f:
        return json.load(f)


def dev_seeds(count, rng_seed):
    rnd, seen, out = random.Random(rng_seed), set(), []
    while len(out) < count:
        s = rnd.getrandbits(64)
        if s >= 1 << 32 and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def run_one(seed, key_hex=DEV_KEY, oracle=None, record=False, cam=None):
    """One token-run episode of the reference in process: the record described in the module docstring. The frames
    are black (verify.BlankCamera) unless `cam` renders them: the reference reads none."""
    cfg = config()
    oracle = oracle or O.Oracle()
    solver = InProcess(oracle, record=record)
    lock = ProbeLock(None, rate=cfg.get("probe_rate", 0.06), pass_fraction=cfg.get("probe_pass_fraction", 0.35),
                     tolerance=1e-4)
    lock.rng = verify.episode_stream(int(key_hex, 16), seed)
    quality, info, steps, _ = verify.run_episode(seed, solver, lock, True, cam or verify.BlankCamera(),
                                                 int(key_hex, 16))
    env = RecordingEnv.last
    ep = oracle.episode
    ep.finish()
    c = ep.committed()
    pl = c.planner
    planned = list(pl.results[pl.chosen]) if pl is not None and pl.chosen is not None else None
    executed = [bool(info.get("success")), steps, quality]
    same = None
    if planned is not None:
        # A planned dock is an executed dock (the same step, the same quality), and a plan without a dock does not
        # dock. A plan without a dock is not compared further: its rollout was cut short (module docstring).
        same = planned[0] == executed[0] and (not planned[0] or planned[1:] == executed[1:])
    out = {
        "seed": seed, "quality": quality, "success": bool(info.get("success")),
        "reason": info.get("failure_reason", ""), "steps": steps,
        "failure_step": env.failure_step, "first_event_seen": ep.stats["first_event_seen"],
        "rekey_earliest": env.rekey_earliest, "rekey_deadline": env.rekey_deadline,
        "rekey_step": env.rekey_step if env.rekey_active else None, "t_ready": c.t_ready,
        "chosen": pl.chosen if pl is not None else None,
        "variant": O.FAMILY[pl.chosen].describe() if pl is not None and pl.chosen is not None else None,
        "tried": len(pl.results) if pl is not None else 0,
        "results": [list(r) for r in pl.results] if pl is not None else [],
        "simulated": pl.simulated if pl is not None else 0,
        "requests": solver.requests, "max_request_s": solver.max_s,
        "probes": lock.total, "probes_passed": lock.passed,
        "planned": planned, "executed": executed, "planned_equals_executed": same,
        "verified": ep.stats["verified"], "steps_unverified_after_ready": ep.stats["steps_unverified_after_ready"],
    }
    if record:
        out["replies"] = solver.replies
    return out


def _worker(args):
    seed, key_hex = args
    return run_one(seed, key_hex)


def summarize(rows):
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    lines = ["episodes {}: docked {} ({:.2f}%)".format(n, len(ok), 100.0 * len(ok) / max(n, 1))]
    reasons = collections.Counter(r["reason"] for r in rows if not r["success"])
    lines.append("failures by reason: {}".format(dict(reasons)))
    if ok:
        qs = sorted(r["quality"] for r in ok)
        st = sorted(r["steps"] for r in ok)
        lines.append("docked quality mean {:.5f} min {:.5f} max {:.5f}; dock step median {} max {}".format(
            sum(qs) / len(qs), qs[0], qs[-1], st[len(st) // 2], st[-1]))
    probes = sum(r["probes"] for r in rows)
    passed = sum(r["probes_passed"] for r in rows)
    per = sorted(r["probes_passed"] / r["probes"] for r in rows if r["probes"])
    lines.append("probes {} passed {} ({:.4f}); per-episode pass fraction min {:.3f}".format(
        probes, passed, passed / max(probes, 1), per[0] if per else float("nan")))
    pv = [r for r in rows if r["planned_equals_executed"] is not None]
    lines.append("a planned dock is an executed dock, and no unplanned dock: {} / {} (no plan: {})".format(
        sum(1 for r in pv if r["planned_equals_executed"]), len(pv), n - len(pv)))
    lines.append("steps after t_ready with no verified request: {}".format(
        sum(r["steps_unverified_after_ready"] for r in rows)))
    at_earliest = sum(1 for r in rows if r["rekey_step"] is not None and r["rekey_step"] == r["rekey_earliest"])
    lines.append("re-key at rekey_earliest: {} / {}; t_ready - rekey_step: {}".format(
        at_earliest, sum(1 for r in rows if r["rekey_step"] is not None),
        dict(collections.Counter(r["t_ready"] - r["rekey_step"] for r in rows
                                 if r["t_ready"] is not None and r["rekey_step"] is not None))))
    first = sum(1 for r in ok if r["chosen"] == 0)
    lines.append("docked with the first policy: {}; policies rolled out mean {:.2f}; simulated steps mean {:.0f} max {}"
                 .format(first, sum(r["tried"] for r in rows) / max(n, 1),
                         sum(r["simulated"] for r in rows) / max(n, 1), max(r["simulated"] for r in rows)))
    lines.append("slowest request {:.3f} s; requests per episode mean {:.1f}".format(
        max(r["max_request_s"] for r in rows), sum(r["requests"] for r in rows) / max(n, 1)))
    for r in rows:
        if not r["success"]:
            lines.append("  FAIL {} {} steps={} q={:.4f} rekey={} (earliest {}, deadline {}) t_ready={} tried={}".format(
                r["seed"], r["reason"], r["steps"], r["quality"], r["rekey_step"], r["rekey_earliest"],
                r["rekey_deadline"], r["t_ready"], r["tried"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--rng", type=int, default=20261105)
    ap.add_argument("--seeds", default="", help="comma-separated seeds instead of --count / --rng")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--key", default=DEV_KEY, help="the dev lock key (64 hex digits)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()] or dev_seeds(a.count, a.rng)
    t0 = time.time()
    with multiprocessing.Pool(a.workers) as pool:
        rows = list(pool.imap(_worker, [(s, a.key) for s in seeds], chunksize=1))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(summarize(rows))
    print("wall {:.0f} s, {} workers, python {}".format(time.time() - t0, a.workers, sys.version.split()[0]))


if __name__ == "__main__":
    main()
