"""Reference controller (privileged): token-gated replay of a recorded rescue.

Only `_gt.state_id`, which the grader adds to the observation for a matching token, tells it which recorded rescue to
replay. The rescue's actions were recorded once on the author's machine by a frozen learned policy driving the arm from
the same restored failure state. This file shows that each hidden state can be completed; it knows nothing about how.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def hold():
    a = np.zeros(12, np.float32)
    a[6] = -1.0        # gripper open
    a[11] = -1.0       # arm mode (no base motion)
    return a


def load(state_id):
    with np.load(os.path.join(HERE, "rescues", state_id + ".npz")) as z:
        return np.asarray(z["action"], np.float32)


def main():
    print("READY", flush=True)
    actions, state = None, None
    for line in sys.stdin:
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "reset":
            actions, state = None, None
            print(json.dumps({"ok": True}), flush=True)
        elif kind == "close":
            print(json.dumps({"ok": True}), flush=True)
            return
        elif kind == "act":
            t, obs = msg["t"], msg["obs"]
            if actions is None:
                state = obs.get("_gt", {}).get("state_id")
                actions = load(state) if state else np.zeros((0, 12), np.float32)
            a = actions[t] if t < len(actions) else hold()
            print(json.dumps({"action": [float(x) for x in a]}), flush=True)


if __name__ == "__main__":
    main()
