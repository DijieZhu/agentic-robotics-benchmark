"""Practice grader - run your controller the way the real grading run will.

    /opt/venv/bin/python3 /home/user/practice_grader/grade.py [--missions 3] [--seed 1]

What is the same as the real run:

* the wire: `bash run.sh` in /home/user/submission, READY within 120 s, every reply within
  30 s, one JSON object per line, and a few steps asked twice with the same `t` (once with
  different target descriptions);
* the sandbox around your controller: no IP sockets, loopback included (started through
  /home/user/aerial/netguard.py exactly as in the real run);
* the simulator, the mission assembly, the actions, the observation, the termination rules
  and the mission result: the same simulator bridge and the same rules module.

What is different:

* the missions are assembled from the practice scenes in this image, not the held-out
  scenes the real run uses;
* it prints the raw metric and per-mission outcomes; it knows no reference value.

`--seed` picks a different set of practice missions; `--no-guard` starts your controller
without the socket guard, for debugging only.
"""

import argparse
import json
import os
import queue
import random
import subprocess
import sys
import threading

AERIAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aerial")
sys.path.insert(0, AERIAL)

from sar_protocol import (  # noqa: E402
    aggregate,
    assemble_missions,
    load_pool,
    perturb_targets,
    run_mission,
)
from sim_client import SimClient  # noqa: E402

STEP_TIMEOUT_S = 30.0
READY_TIMEOUT_S = 4 * STEP_TIMEOUT_S
REPEAT_STEPS = (2, 25, 70)
BUDGET = 1000


class WireError(Exception):
    pass


class Wire(object):
    def __init__(self, submission, guard):
        argv = ["bash", "run.sh"]
        if guard:
            argv = [sys.executable, "-I", "-S", os.path.join(AERIAL, "netguard.py"), "exec",
                    submission] + argv
        self.proc = subprocess.Popen(argv, cwd=submission, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, universal_newlines=True,
                                     bufsize=1, start_new_session=True)
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        for _ in range(200):
            if self._readline(READY_TIMEOUT_S).strip() == "READY":
                return
        raise WireError("your controller never printed READY")

    def _pump(self):
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _readline(self, timeout):
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise WireError("no reply within {:.0f}s - the real run aborts the mission "
                            "here".format(timeout))
        if line is None:
            raise WireError("your controller closed its stdout (crashed? see its stderr)")
        return line

    def request(self, message):
        try:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
        except (IOError, OSError):
            raise WireError("your controller's stdin is closed")
        while True:
            line = self._readline(STEP_TIMEOUT_S).strip()
            if not line:
                continue
            try:
                reply = json.loads(line)
            except ValueError:
                raise WireError("non-JSON on stdout: {!r} - stdout is the wire; send "
                                "diagnostics to stderr".format(line[:120]))
            if not isinstance(reply, dict):
                raise WireError("reply is not a JSON object")
            return reply

    def act(self, t, obs):
        reply = self.request({"type": "act", "t": t, "obs": obs})
        if "action" not in reply:
            raise WireError("no action at t={}: {}".format(t, reply.get("error", reply)))
        return reply["action"]

    def close(self):
        try:
            self.request({"type": "close"})
        except WireError:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="/home/user/submission")
    parser.add_argument("--pool", default="/opt/aerialsim/practice/pool.json")
    parser.add_argument("--missions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", default="airsim", choices=("airsim", "mock"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--no-guard", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.submission, "run.sh")):
        print("[practice] no run.sh in {}".format(args.submission))
        return 2
    pool = load_pool(args.pool)
    rng = random.Random(args.seed)
    missions = assemble_missions(pool, rng, per_map=args.missions)[:args.missions]
    descriptions = sorted({t["description"] for entry in pool["maps"].values()
                           for t in entry["targets"]})
    stats = {"repeated": 0, "differed": 0}
    log_path = os.path.join(os.path.expanduser("~"), "practice_sim.log")
    print("[practice] starting the simulator (first scene load can take minutes); "
          "simulator log: {}".format(log_path), file=sys.stderr)
    sim = SimClient(backend=args.backend, log_path=log_path,
                    extra_args=["--image-size", str(args.image_size)])
    outcomes = []
    try:
        wire = Wire(args.submission, guard=not args.no_guard)
        try:
            for index, mission in enumerate(missions):
                decoys = [rng.choice(descriptions) for _ in mission["targets"]]

                def probe(t, obs, _decoys=decoys):
                    if t not in REPEAT_STEPS:
                        return None
                    fake = perturb_targets(obs, _decoys)
                    if rng.random() < 0.5:
                        other, real = wire.act(t, fake), wire.act(t, obs)
                    else:
                        real, other = wire.act(t, obs), wire.act(t, fake)
                    stats["repeated"] += 1
                    stats["differed"] += int(json.dumps(other) != json.dumps(real))
                    return real

                wire.request({"type": "reset"})
                outcome = run_mission(sim, mission, wire.act, BUDGET, probe=probe)
                outcomes.append(outcome)
                print("[practice] mission {} in {}: targets {} -> {}".format(
                    index, mission["map_name"],
                    [t["description"][:40] for t in mission["targets"]],
                    json.dumps(outcome.record())), file=sys.stderr)
        finally:
            wire.close()
    except WireError as exc:
        print("[practice] {}".format(exc))
        return 1
    finally:
        sim.close()

    if stats["repeated"] and stats["differed"] == 0:
        print("[practice] WARNING: your reply never changed when the target descriptions "
              "changed at a repeated t. The real run treats a controller that ignores its "
              "targets as a zero regardless of its metric.", file=sys.stderr)
    summary = aggregate(outcomes)
    summary.update({
        "metric": "geolocation_score", "direction": "higher",
        "repeated_t_queries": stats["repeated"],
        "repeated_t_answers_differed": stats["differed"],
        "note": "practice scenes only; this is the raw metric, not a grade",
    })
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
