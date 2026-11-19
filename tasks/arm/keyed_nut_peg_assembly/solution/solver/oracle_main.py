"""Reference solution's wire loop: answers the grading wire with `oracle_ctrl.Oracle`.

It needs the ground truth that the grader attaches (`_gt`) only when the submission carries the
reference solution's token; without it, it answers a no-op action. It acts on the real observation's
proprioception, which `_gt["proprio"]` carries, so a repeated step is answered identically whatever copy it
comes with (the reference's run does not depend on the grader's probe draws). Repeated step indices are
handled like helpers/wire_skeleton.py: every query of a step is answered from a copy of the
controller state from before that step, and when the next step arrives the copy whose observation is
closest to it is kept.
"""

import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import oracle_ctrl  # noqa: E402

NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0]


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a["ee_pos"], b["ee_pos"])))


def main():
    out = sys.stdout
    ctrl = oracle_ctrl.KitOracle()
    cur_t, snapshot, candidates = None, None, []
    print("READY", file=out, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        kind = req.get("type")
        if kind == "reset":
            ctrl = oracle_ctrl.KitOracle()
            cur_t, snapshot, candidates = None, None, []
            print(json.dumps({"ok": True}), file=out, flush=True)
        elif kind == "act":
            t, obs = int(req["t"]), req["obs"]
            if t != cur_t:
                if candidates:
                    ctrl = min(candidates, key=lambda c: _dist(c[0], obs))[1]
                snapshot = copy.deepcopy(ctrl)
                candidates = []
                cur_t = t
                work = ctrl
            else:
                work = copy.deepcopy(snapshot)
            gt = obs.get("_gt")
            action = work.act(dict(obs, **gt.get("proprio", {})), gt) if gt else list(NOOP)
            candidates.append((obs, work))
            print(json.dumps({"action": [float(a) for a in action]}), file=out, flush=True)
        elif kind == "close":
            print(json.dumps({"ok": True}), file=out, flush=True)
            return


if __name__ == "__main__":
    main()
