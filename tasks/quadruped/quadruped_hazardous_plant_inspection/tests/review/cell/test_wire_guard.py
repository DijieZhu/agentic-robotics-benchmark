"""The request-line guard of the scored wire (plantcell/wire_guard.py).

    python -I -B tests/review/cell/test_wire_guard.py [--json OUT]

The shared `cellkit.wire.serve_episode` loop driven with crafted request lines, once with the shared kit's
`AgentProcess.next_request` ("before") and once with `GuardedProcess` (what tests/verify.py and run_wire.py use). No
program process and no cell: the lines go straight into the process object's queue, a stand-in server answers
(`get_state`, `finish`, everything else `forbidden`), and every reply is encoded as the real `reply` encodes it.
  A. a line nested 100,000 deep, then `finish`: before, RecursionError out of serve_episode (verify.py: "the wire
     failed", the grader stops); with the guard, the mission ends as `protocol`;
  B. method and id each a list nested 5,000 deep, then `finish`: with the guard `forbidden` with id null, then the
     finish;
  C. id a 4,000-digit integer, an object, a 300-character string; method a number: with the guard the replies carry
     id null and the number is refused as `forbidden`; ordinary ids (7, "a", null) are echoed as they are.
"""

import argparse
import json
import os
import queue
import sys

TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TESTS)
sys.dont_write_bytecode = True        # (no __pycache__ in the task tree from a review run)

from cellkit import wire  # noqa: E402
from plantcell.wire_guard import GuardedProcess  # noqa: E402


class StandIn(object):
    """The server's side of serve_episode: handle(), done, done_reason, _end(), work_wall."""

    def __init__(self):
        self.done = False
        self.done_reason = None
        self.work_wall = 0.0
        self.calls = []

    def handle(self, method, params):
        self.calls.append(method if len(method) <= 64 else method[:64] + "...")
        if method == "finish":
            self.done, self.done_reason = True, "finish"
            return {"ok": True, "sim_time": 0.0}
        if method == "get_state":
            return {"sim_time": 0.0, "done": self.done}
        return {"error": "forbidden", "detail": "not a public call"}

    def _end(self, reason):
        if not self.done:
            self.done, self.done_reason = True, reason


def fake_proc(cls, lines):
    p = object.__new__(cls)
    p._q = queue.Queue()
    p.stalled = False
    p.sent = []
    enc = getattr(wire, "_jsonable", str)

    def reply(obj, timeout=None):
        p.sent.append(json.loads(json.dumps(obj, default=enc)))     # the real reply's encoding
        return True
    p.reply = reply
    for ln in lines:
        p._q.put(("line", ln + "\n"))
    p._q.put(("eof", None))
    return p


def episode(cls, lines):
    srv = StandIn()
    p = fake_proc(cls, lines)
    try:
        reason = wire.serve_episode(srv, p, 60.0, 30.0)
        return {"reason": reason, "replies": p.sent, "calls": srv.calls, "raised": None}
    except BaseException as e:  # noqa: BLE001 - what the shared loop lets out is the finding
        return {"reason": None, "replies": p.sent, "calls": srv.calls, "raised": type(e).__name__}


def nested(n):
    return "[" * n + "]" * n


def show_id(x):
    """A reply's id, printable (the before side echoes whatever came)."""
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x if abs(x) < 10 ** 18 else "(a %d-digit integer)" % len(str(abs(x)))
    if isinstance(x, str):
        return x if len(x) <= 40 else "(a %d-character string)" % len(x)
    return "(a %s)" % type(x).__name__


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    fin = json.dumps({"id": 99, "method": "finish", "params": {}})
    cases = {
        "A_deep_line": [nested(100000), fin],
        "B_deep_method_and_id": ['{"id": %s, "method": %s, "params": {}}' % (nested(5000), nested(5000)), fin],
        "C_odd_ids_and_method": [json.dumps({"id": 10 ** 3999, "method": "get_state", "params": {}}),
                                 json.dumps({"id": {"a": 1}, "method": "get_state", "params": {}}),
                                 json.dumps({"id": "x" * 300, "method": "get_state", "params": {}}),
                                 json.dumps({"id": 5, "method": 12345, "params": {}}),
                                 json.dumps({"id": 7, "method": "get_state", "params": {}}),
                                 json.dumps({"id": "a", "method": "get_state", "params": {}}),
                                 json.dumps({"id": None, "method": "get_state", "params": None}), fin],
    }
    out = {"before": {}, "guard": {}}
    problems = []
    for name, lines in cases.items():
        for side, cls in (("before", wire.AgentProcess), ("guard", GuardedProcess)):
            r = episode(cls, lines)
            out[side][name] = {"reason": r["reason"], "raised": r["raised"], "calls": r["calls"],
                               "reply_ids": [show_id(x.get("id")) for x in r["replies"]],
                               "reply_kinds": [x.get("error") or "result" for x in r["replies"]]}
    g, b = out["guard"], out["before"]
    if b["A_deep_line"]["raised"] != "RecursionError":
        problems.append("before: the deep line did not raise out of serve_episode (%s): the test does not reproduce "
                        "the defect" % b["A_deep_line"])
    if g["A_deep_line"]["raised"] or g["A_deep_line"]["reason"] != "protocol":
        problems.append("guard: the deep line: %s" % g["A_deep_line"])
    if g["B_deep_method_and_id"]["raised"] or g["B_deep_method_and_id"]["reply_kinds"] != ["forbidden", "result"] \
            or g["B_deep_method_and_id"]["reply_ids"] != [None, 99] or g["B_deep_method_and_id"]["reason"] != "finish":
        problems.append("guard: deep method and id: %s" % g["B_deep_method_and_id"])
    c = g["C_odd_ids_and_method"]
    if c["raised"] or c["reply_ids"] != [None, None, None, 5, 7, "a", None, 99] or \
            c["reply_kinds"] != ["result", "result", "result", "forbidden", "result", "result", "result", "result"] \
            or c["calls"][3] != "(not a name)" or c["reason"] != "finish":
        problems.append("guard: odd ids and method: %s" % c)
    out["problems"] = problems
    print(json.dumps(out, indent=1, default=str))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
    print("PASS" if not problems else "FAIL (%d problems)" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
