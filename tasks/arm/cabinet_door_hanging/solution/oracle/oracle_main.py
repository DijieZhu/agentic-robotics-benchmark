"""Cabinet door hanging: the reference as a closed_loop submission (`run.sh`: `python -I oracle_main.py`).

Speaks the grader's wire (one JSON object per line on stdin, one reply per line on stdout; logs on stderr): READY,
then {"type": "reset"} -> {"ok": true}, {"type": "act", "t", "obs"} -> {"action": {...}}, {"type": "close"} ->
{"ok": true} and exit. Without `truth` in the observation (not a token run) it holds still. See `wire_world.py`.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from wire_world import Oracle  # noqa: E402


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
            oracle.stop()
            out.write(json.dumps({"ok": True}) + "\n")
            out.flush()
            return
        else:
            reply = {"error": "unknown message"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
