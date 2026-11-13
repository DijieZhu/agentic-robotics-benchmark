"""Grade a submission over the stdio wire (P3 prototype of tests/verify.py and the practice grader).

Runs `bash run.sh` in --submission through the robotics_grader kit's Solver (READY, reset, act t obs,
close; per-request deadline; restarts) with its ProbeLock (repeated `t`, real and perturbed copies in a
hidden order), on episodes of the given units. Each observation carries `unit` (the unit's id) and the
open `request`; with --truth (the oracle's token run) also `truth`.

    python -I wire_grade.py --assets <models> --units units.json --submission <dir> --seeds 0 1 [--truth]
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "shared"))
import numpy as np  # noqa: E402
from robotics_grader import ProbeLock, Solver, SolverAborted, seed_rng  # noqa: E402

import appliance_env as E  # noqa: E402
import episode as EP  # noqa: E402


class LocalSolver(Solver):
    """The kit's Solver, started as the current user (a host-side run is not root; the task's grader
    uses the kit's Solver unchanged, which starts run.sh as the agent through runuser)."""

    def _spawn(self):
        import subprocess
        return subprocess.Popen(["bash", self.entry], cwd=self.submission, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                                bufsize=1, start_new_session=True)


def wire_obs(ep, obs, uid, truth):
    w = E.to_wire({k: v for k, v in obs.items() if k != "request"})
    w["request"] = obs.get("request")
    w["unit"] = uid
    if truth:
        w["truth"] = ep.env.truth()
    return w


def run(a):
    units = json.load(open(a.units))
    rng = seed_rng()
    lock = ProbeLock(rng, rate=a.probe_rate, pass_fraction=0.1, tolerance=1e-4)
    solver = LocalSolver(step_timeout=a.step_timeout, submission=a.submission)
    results = []
    try:
        for uid in sorted(units):
            spec = E.unit_spec(units[uid]["model"])
            for seed in a.seeds:
                ep = EP.Episode(E.ApplianceEnv(spec, os.path.expanduser(a.assets), render=True), seed)
                t0 = time.time()
                reason = "ok"
                try:
                    obs = ep.reset()
                    solver.reset()
                    t, done = 0, False
                    while not done:
                        w = wire_obs(ep, obs, uid, a.truth)
                        if lock.due():
                            other = ep.env.perturbed_observation(obs, np.random.default_rng(t))
                            other["request"] = obs.get("request")
                            action = lock.probe(solver, t, w, perturbed=wire_obs(ep, other, uid, a.truth))
                        else:
                            action = solver.act(t, w)
                        if not (isinstance(action, list) and len(action) in (7, 8)
                                and all(isinstance(x, (int, float)) and math.isfinite(x) for x in action)):
                            raise SolverAborted("malformed action")
                        if len(action) == 7:
                            action = action + [0]
                        obs, done, _ = ep.step(action)
                        t += 1
                    sc = ep.score()
                except SolverAborted as exc:
                    sc = {"score": 0.0, "requests": ep.score()["requests"] if getattr(ep, "requests", None) else []}
                    reason = "aborted: %s" % str(exc)[:120]
                ep.env.close()
                rec = {"unit": uid, "model": units[uid]["model"], "seed": seed, "score": sc["score"], "reason": reason,
                       "requests": sc["requests"], "steps": ep.env.t, "wall_s": round(time.time() - t0, 1)}
                results.append(rec)
                print("%-4s %-24s seed %-3d score %.2f %s  %s" % (uid, units[uid]["model"], seed, rec["score"], reason,
                      " | ".join("%s %s%s" % (q["text"], "R" if q["reached"] else "-", "C" if q["completed"] else "-")
                                 for q in sc["requests"])), flush=True)
    finally:
        try:
            solver.close()
        except Exception:  # noqa: BLE001
            pass
    mean = float(np.mean([r["score"] for r in results])) if results else 0.0
    print("mean_request_credit %.4f over %d episodes; probes %d, lock ok %s" % (mean, len(results), lock.total, lock.ok))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--truth", action="store_true", help="token run: add the truth to every observation")
    ap.add_argument("--probe-rate", type=float, default=0.01)
    ap.add_argument("--step-timeout", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
