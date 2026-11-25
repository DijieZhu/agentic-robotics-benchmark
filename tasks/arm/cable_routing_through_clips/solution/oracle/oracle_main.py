"""cable_routing_through_clips: the reference as a closed-loop submission (`run.sh`: `python3 -I oracle_main.py`).

Speaks the grader's wire (one JSON object per line on stdin, one reply per line on stdout): READY, then
{"type": "reset"} -> {"ok": true}, {"type": "act", "t", "obs"} -> {"action": [...]}, {"type": "close"} -> {"ok": true}
and exit. The program is fitter.Fitter (a generator of actions fed with the observation and the grader's token
truth). Without `truth` in the observation (not a token run) it hands the job over at once.

The program runs on the real observation, which the token truth carries (`truth.obs`), so over the wire it does
exactly what it does in process, where the graded set was chosen. Each reply is the program's action moved by the
difference between the hand's position in the request and in the real observation: nothing for the real request,
and for the other request of a probe pair what the program's last line (a step toward a target from where the
hand is) would give for it.

A file `variant` beside this one names a control program (fitter.variant: stop_after_1, stop_after_2,
no_tension_watch, wrong_post_side, wrong_clip_orientation, nominal_positions); absent = the reference.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fitter as F  # noqa: E402

HAND_OVER = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0, 1]


class Oracle(object):
    def __init__(self):
        name = "reference"
        try:
            with open(os.path.join(HERE, "variant")) as fh:
                name = fh.read().strip() or "reference"
        except OSError:
            pass
        self.cls = F.variant(name)
        self.reset()

    def reset(self):
        self.gen = None
        self.t = None
        self.done = False

    def act(self, t, obs):
        if "truth" not in obs or self.done:
            return list(HAND_OVER)
        truth = obs["truth"]
        real = truth["obs"]
        if t != self.t:
            try:
                if self.gen is None:
                    self.st = {"obs": real, "truth": truth}
                    self.gen = self.cls(obs["job"]).run(self.st)
                    a = next(self.gen)
                else:
                    a = self.gen.send((real, truth))
                a = [float(x) for x in a[:7]] + [0]
            except StopIteration:
                self.done = True
                a = list(HAND_OVER)
            self.t, self.action = t, a
        a = list(self.action)
        if len(a) == 8:
            for k in range(3):
                a[k] -= obs["tcp"]["pos"][k] - real["tcp"]["pos"][k]
        return a


def main():
    out = sys.stdout
    out.write("READY\n")
    out.flush()
    oracle = Oracle()
    for line in sys.stdin:
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "act":
            reply = {"action": oracle.act(int(msg["t"]), msg["obs"])}
        elif kind == "reset":
            oracle.reset()
            reply = {"ok": True}
        elif kind == "close":
            out.write(json.dumps({"ok": True}) + "\n")
            out.flush()
            return
        else:
            reply = {"error": "unknown message"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
