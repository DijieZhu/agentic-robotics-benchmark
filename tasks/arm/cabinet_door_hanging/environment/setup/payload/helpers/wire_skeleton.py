"""Boilerplate for the grading wire (optional to use; it contains no control logic).

Copy this file INTO your submission folder and import it from there: at grading time only
/home/user/submission exists, so nothing outside it can be imported.

    cp /home/user/helpers/wire_skeleton.py /home/user/submission/

    # in /home/user/submission/controller.py (run.sh: cd "$(dirname "$0")"; exec python3 controller.py)
    # (python3 -I leaves the script's own folder off sys.path: then add it yourself before this import)
    from wire_skeleton import serve

    class MyController(object):
        def reset(self):            # called at the start of every episode
            ...
        def act(self, obs):         # obs: the observation dict; return {"l": [...], "r": [...], "cams": mask}
            return {...}

    serve(MyController)            # prints READY, then answers reset / act / close

Repeated step indices: the grader may query the same `t` more than once with different
observations and applies only one reply (it does not say which). `serve` keeps a copy of the
controller state from before the first query of each `t`, answers every query of that `t` from a
fresh copy of it, and when step `t + 1` arrives keeps the candidate whose observation is closest
to the new one (both hands' positions, ee_pos). Diagnostics go to stderr; stdout carries protocol
messages only.
"""

import copy
import json
import math
import sys
import traceback


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for s in ("l", "r") for x, y in zip(a[s]["ee_pos"][:3], b[s]["ee_pos"][:3])))


def serve(controller_factory):
    ctrl = controller_factory()
    cur_t = None
    snapshot = None
    candidates = []          # (obs, controller copy after answering)
    out = sys.stdout
    print("READY", file=out, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            print(json.dumps({"error": "bad json"}), file=out, flush=True)
            continue
        kind = req.get("type")
        try:
            if kind == "reset":
                ctrl = controller_factory()
                ctrl.reset()
                cur_t, snapshot, candidates = None, None, []
                print(json.dumps({"ok": True}), file=out, flush=True)
            elif kind == "act":
                t, obs = int(req["t"]), req["obs"]
                if t != cur_t:
                    if len(candidates) > 1:
                        ctrl = min(candidates, key=lambda c: _dist(c[0], obs))[1]
                    elif candidates:
                        ctrl = candidates[0][1]
                    snapshot = copy.deepcopy(ctrl)
                    candidates = []
                    cur_t = t
                    work = ctrl
                else:
                    work = copy.deepcopy(snapshot)
                action = work.act(obs)
                candidates.append((obs, work))
                act = {"l": [float(a) for a in action["l"]], "r": [float(a) for a in action["r"]],
                       "cams": int(action.get("cams", 0))}
                print(json.dumps({"action": act}), file=out, flush=True)
            elif kind == "close":
                print(json.dumps({"ok": True}), file=out, flush=True)
                return
            else:
                print(json.dumps({"error": "unknown request"}), file=out, flush=True)
        except Exception as exc:  # noqa: BLE001 - keep serving; the grader aborts the episode
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"error": str(exc)[:200]}), file=out, flush=True)
