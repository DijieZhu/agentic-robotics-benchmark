"""earthquake_search_and_rescue: a request line the JSON parser cannot read ends the episode as `protocol`, a line
nested deeper than the parser's recursion limit included.

The verify stage serves the program under test with tests/cellkit/wire.py, and run_episode turns anything that
escapes `serve_episode` into a task error ("the wire failed": the whole verify stage stops, no verdict). A line of
100,000 nested brackets is 200 KB, far below the wire's 4 MiB line limit, and json.loads raises RecursionError on it
(not ValueError); `next_request` must treat it as it treats any other malformed line. The kit's real AgentProcess runs
a small child program (this interpreter, -I) that writes the line, and a stand-in server answers get_state / finish.
Standard library only.
"""
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1] / "tasks/ground/earthquake_search_and_rescue/tests/cellkit"
DEEP = 100000

# the program under test: writes the line chosen by argv[1] (nesting depth argv[2]), reads its reply (if any), then
# sends finish
CHILD = r'''
import json, sys
n = int(sys.argv[2])
deep = "[" * n + "]" * n
line = {
    "deep_line": deep,
    "deep_object": '{"a":' * n + "1" + "}" * n,
    "deep_params": '{"id": 1, "method": "get_state", "params": {"x": ' + deep + "}}",
    "malformed": '{"id": 1, "method": "get_state", "params": {}',
    "plain": json.dumps({"id": 1, "method": "get_state", "params": {}}),
}[sys.argv[1]]
for ln in (line, json.dumps({"id": 2, "method": "finish", "params": {}})):
    try:
        sys.stdout.write(ln + "\n")
        sys.stdout.flush()
    except OSError:
        break
    if not sys.stdin.readline():
        break
'''


def _wire():
    """The task's cellkit.wire under a name of its own (other tests load other tasks' kits); no __pycache__."""
    name = "earthquake_search_and_rescue_cellkit"
    if name + ".wire" not in sys.modules:
        saved = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec = importlib.util.spec_from_file_location(name, KIT / "__init__.py",
                                                          submodule_search_locations=[str(KIT)])
            pkg = importlib.util.module_from_spec(spec)
            sys.modules[name] = pkg
            spec.loader.exec_module(pkg)
            importlib.import_module(name + ".wire")
        finally:
            sys.dont_write_bytecode = saved
    return sys.modules[name + ".wire"]


class StandIn:
    """The cell's side of serve_episode: handle(), done, done_reason, _end(), work_wall."""

    def __init__(self):
        self.done = False
        self.done_reason = None
        self.work_wall = 0.0
        self.calls = []

    def handle(self, method, params):
        self.calls.append(method)
        if method == "finish":
            self.done, self.done_reason = True, "finish"
            return {"ok": True, "sim_time": 0.0}
        if method == "get_state":
            return {"sim_time": 0.0, "done": self.done}
        return {"error": "forbidden"}

    def _end(self, reason):
        if not self.done:
            self.done, self.done_reason = True, reason


class WireUnreadableLine(unittest.TestCase):
    def episode(self, case):
        wire = _wire()
        server = StandIn()
        proc = wire.AgentProcess([sys.executable, "-I", "-c", CHILD, case, str(DEEP)], cwd=tempfile.gettempdir())
        try:
            reason = wire.serve_episode(server, proc, 60.0, 30.0)
        finally:
            proc.terminate(grace=2.0)
        return reason, server

    def test_the_parser_raises_recursionerror_on_the_deep_line(self):
        """The premise: not a ValueError, so a plain `except ValueError` lets it out."""
        with self.assertRaises(RecursionError):
            json.loads("[" * DEEP + "]" * DEEP)

    def test_a_line_nested_100000_deep_ends_the_episode_as_protocol(self):
        for case in ("deep_line", "deep_object", "deep_params"):
            with self.subTest(case=case):
                reason, server = self.episode(case)
                self.assertEqual(reason, "protocol")
                self.assertEqual(server.calls, [])
                self.assertEqual(server.done_reason, "protocol")

    def test_a_malformed_line_ends_it_the_same_way(self):
        reason, server = self.episode("malformed")
        self.assertEqual((reason, server.calls), ("protocol", []))

    def test_an_ordinary_request_is_answered(self):
        reason, server = self.episode("plain")
        self.assertEqual((reason, server.calls), ("finish", ["get_state", "finish"]))


if __name__ == "__main__":
    unittest.main()
