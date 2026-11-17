"""Practice grader - run your supervisor the way the real grading run will.

    /opt/venv/bin/python3 /home/user/practice_grader/grade.py [--per-kind 1] [--seed 1]
        [--kinds construction_zone occluded_pedestrian ...] [--towns Town01 Town02 Town03]

What is the same as the real run:

* the wire: `bash run.sh` in /home/user/submission, READY within 40 s, every reply within
  10 s, one JSON object per line, and a few steps asked twice with the same `t` (once with
  a perturbed front camera);
* the sandbox around your supervisor: no IP sockets, loopback included (started through
  /home/user/drive/netguard.py exactly as in the real run);
* the simulator, the nominal planner, the observation and the termination rules: the same
  bridge and protocol modules.

What is different:

* the practice towns Town01-Town03, visible seeds, and every scenario in its plain form; the
  real run uses a fixed hidden set of 24 episodes on other towns, staged in forms practice
  does not show;
* it reports what happened in each episode (how it ended, how far the car got,
  collisions, red lights, harsh braking, mean speed), not a score.

`--no-guard` starts your supervisor without the socket guard, for debugging only.
"""

import argparse
import json
import os
import queue
import random
import subprocess
import sys
import threading

DRIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drive")
sys.path.insert(0, DRIVE)

from drive_protocol import KINDS, make_specs, probe_observation, run_episode  # noqa: E402
from sim_client import SimClient  # noqa: E402

STEP_TIMEOUT_S = 10.0
READY_TIMEOUT_S = 4 * STEP_TIMEOUT_S
REPEAT_EVERY = 97
# The towns practice drives in; the real run uses others.
PRACTICE_TOWNS = ("Town01", "Town02", "Town03")
# Town01 and Town02 have no four-way junction, so red_light_runner cannot be staged there.
NO_CROSS_TRAFFIC = ("Town01", "Town02")


class WireError(Exception):
    pass


class Wire(object):
    def __init__(self, submission, guard):
        argv = ["bash", "run.sh"]
        if guard:
            argv = [sys.executable, "-I", "-S", os.path.join(DRIVE, "netguard.py"), "exec",
                    submission] + argv
        self.proc = subprocess.Popen(argv, cwd=submission, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, universal_newlines=True,
                                     bufsize=1, start_new_session=True)
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        for _ in range(200):
            if self._readline(READY_TIMEOUT_S).strip() == "READY":
                return
        raise WireError("your supervisor never printed READY")

    def _pump(self):
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _readline(self, timeout):
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise WireError("no reply within {:.0f}s - the real run aborts the episode "
                            "here".format(timeout))
        if line is None:
            raise WireError("your supervisor closed its stdout (crashed? see its stderr)")
        return line

    def request(self, message):
        try:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
        except (IOError, OSError):
            raise WireError("your supervisor's stdin is closed")
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


def summarize(outcomes, stats):
    n = float(max(1, len(outcomes)))
    records = [o.record() for o in outcomes]
    hit = lambda key: sum(1 for r in records if r["collisions"].get(key, 0)) / n  # noqa: E731
    return {
        "episodes": len(records),
        "completed": sum(1 for r in records if r["termination"] == "completed"),
        "route_completion": round(sum(r["route_completion"] for r in records) / n, 3),
        "episodes_with_pedestrian_collision": round(hit("walker"), 3),
        "episodes_with_vehicle_collision": round(hit("vehicle"), 3),
        "episodes_with_other_collision": round(hit("static"), 3),
        "red_lights": sum(r["red_lights"] for r in records),
        "harsh_brakes": sum(r["harsh_brakes"] for r in records),
        "mean_speed": round(sum(r["mean_speed"] for r in records) / n, 2),
        "repeated_t_queries": stats["repeated"],
        "repeated_t_answers_differed": stats["differed"],
        "note": "practice towns and seeds only, plain scenarios; not a score",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="/home/user/submission")
    parser.add_argument("--per-kind", type=int, default=1)
    parser.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    parser.add_argument("--towns", nargs="+", default=list(PRACTICE_TOWNS), choices=PRACTICE_TOWNS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", default="carla", choices=("carla", "mock"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--no-guard", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.submission, "run.sh")):
        print("[practice] no run.sh in {}".format(args.submission))
        return 2
    rng = random.Random(args.seed)
    specs = make_specs(rng, args.towns, args.per_kind, kinds=args.kinds)
    for spec in specs:
        if spec["kind"] == "red_light_runner" and spec["town"] in NO_CROSS_TRAFFIC:
            spec["town"] = "Town03"
    specs.sort(key=lambda spec: spec["town"])     # each town loaded once
    stats = {"repeated": 0, "differed": 0}
    log_path = os.path.join(os.path.expanduser("~"), "practice_sim.log")
    print("[practice] starting CARLA (a minute or two); simulator log: {}".format(log_path),
          file=sys.stderr)
    sim = SimClient(backend=args.backend, log_path=log_path,
                    extra_args=["--image-size", str(args.image_size)])
    outcomes = []
    try:
        wire = Wire(args.submission, guard=not args.no_guard)
        try:
            for index, spec in enumerate(specs):
                def probe(t, obs, state):
                    if t % REPEAT_EVERY != 50:
                        return None
                    fake = probe_observation(obs, state)
                    if rng.random() < 0.5:
                        other, real = wire.act(t, fake), wire.act(t, obs)
                    else:
                        real, other = wire.act(t, obs), wire.act(t, fake)
                    stats["repeated"] += 1
                    stats["differed"] += int(json.dumps(other) != json.dumps(real))
                    return real

                wire.request({"type": "reset"})
                outcome = run_episode(sim, spec, wire.act, probe=probe)
                outcomes.append(outcome)
                print("[practice] episode {} {} {} seed={}: {}".format(
                    index, spec["town"], spec["kind"], spec["seed"],
                    json.dumps(outcome.record())), file=sys.stderr)
        finally:
            wire.close()
    except WireError as exc:
        print("[practice] {}".format(exc))
        return 1
    finally:
        sim.close()

    if stats["repeated"] and stats["differed"] == 0:
        print("[practice] WARNING: your reply never changed when the front camera showed an "
              "obstacle at a repeated t. The real run treats a supervisor that ignores its "
              "cameras as a zero regardless of its metric.", file=sys.stderr)
    print(json.dumps(summarize(outcomes, stats), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
