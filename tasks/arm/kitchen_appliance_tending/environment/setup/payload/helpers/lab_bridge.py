"""Run a wire-protocol controller against a hidden unit for ONE lab trial (optional boilerplate).

    /opt/venv/bin/python3 /home/user/helpers/lab_bridge.py --unit u1 --submission /home/user/submission \
        --log /home/user/trials/u1_try1.jsonl

It costs one trial (the lab charges it when the trial starts). It starts `bash run.sh` in the
submission directory exactly like the grader does (READY, reset, act, close; same observation
dict), feeds it the lab's observations step by step until the episode ends (its last request has closed), and writes one JSON
line per step ({"t", "obs", "action"}) to --log (images included whenever the controller asked
for them). It reports nothing about scores; evaluate the logged observations yourself. Use
--steps to stop earlier (the trial is still charged).
Unlike the grader it sends every step once (no repeated-t queries) and does not apply the socket
restriction; use the practice grader to test those.
"""

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the agent home
from labclient import LabClient  # noqa: E402

STEP_TIMEOUT = 30.0
READY_TIMEOUT = 120.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True)
    ap.add_argument("--submission", default="/home/user/submission")
    ap.add_argument("--log", required=True)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    proc = subprocess.Popen(["bash", "run.sh"], cwd=a.submission, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            universal_newlines=True, bufsize=1, start_new_session=True)
    lines = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)
    threading.Thread(target=pump, daemon=True).start()

    def read(timeout):
        while True:
            line = lines.get(timeout=timeout)
            if line is None:
                raise RuntimeError("controller closed its stdout")
            line = line.strip()
            if line:
                return line

    def request(msg):
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        return json.loads(read(STEP_TIMEOUT))

    n = 0
    try:
        if read(READY_TIMEOUT) != "READY":
            raise RuntimeError("controller did not print READY first")
        request({"type": "reset"})
        with LabClient() as lab, open(a.log, "w") as log:
            obs = lab.reset(a.unit)
            done = False
            while not done and (a.steps is None or n < a.steps):
                reply = request({"type": "act", "t": n, "obs": obs})
                if "action" not in reply:
                    raise RuntimeError("controller error: {}".format(reply.get("error", "no action")))
                action = reply["action"]
                log.write(json.dumps({"t": n, "obs": obs, "action": action}) + "\n")
                obs, done = lab.step(action)
                n += 1
            log.write(json.dumps({"t": n, "obs": obs, "action": None}) + "\n")
        request({"type": "close"})
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    print(json.dumps({"unit": a.unit, "steps": n, "log": a.log}))


if __name__ == "__main__":
    main()
