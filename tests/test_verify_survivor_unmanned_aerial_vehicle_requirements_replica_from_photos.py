"""unmanned_aerial_vehicle_requirements_replica_from_photos: a process of the agent account that outlives the
verify-start kill (the base image's supervisord respawns the desktop it runs as that account, as fast as it is killed)
is stopped, not held against the submission; one that is still running once stopped still fails the verify.

Runs the grader's own `freeze_and_kill` (and its helpers), taken from tests/verify.py by name, against a stand-in for
the account's process list and real child processes playing the respawned desktop (each SIGKILL starts the next one,
as supervisord does). The clock runs four times as fast, so the 8 s kill takes 2 s. Stdlib only.
"""
import ast
import os
import signal
import subprocess
import time
import types
import unittest
from pathlib import Path

VERIFY = (Path(__file__).resolve().parents[1]
          / "tasks/uav/unmanned_aerial_vehicle_requirements_replica_from_photos/tests/verify.py")


class _Desktop:
    """The account's processes: one child at a time; `respawn` True: a SIGKILL starts the next one."""

    def __init__(self, respawn, stoppable=True):
        self.respawn, self.stoppable = respawn, stoppable
        self.procs = [subprocess.Popen(["sleep", "60"])]

    def live(self, uid):
        return [p.pid for p in self.procs if p.poll() is None]

    def kill(self, pid, sig):
        proc = next((p for p in self.procs if p.pid == pid), None)
        if proc is None or proc.poll() is not None:
            raise ProcessLookupError(pid)
        if sig == signal.SIGSTOP and not self.stoppable:
            return
        os.kill(pid, sig)
        if sig == signal.SIGKILL:
            proc.wait()
            if self.respawn:
                self.procs.append(subprocess.Popen(["sleep", "60"]))

    def state(self, pid):
        return Path("/proc/%d/status" % pid).read_text().split("State:")[1].split()[0]

    def close(self):
        for p in self.procs:
            if p.poll() is None:
                os.kill(p.pid, signal.SIGCONT)
                p.kill()
                p.wait()


def _functions(names):
    tree = ast.parse(VERIFY.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names), "verify.py lacks %s" % (set(names) - {n.name for n in body})
    return compile(ast.Module(body=body, type_ignores=[]), str(VERIFY), "exec")


class SurvivorAtVerifyStart(unittest.TestCase):
    def _run(self, desktop):
        self.addCleanup(desktop.close)
        fast = types.SimpleNamespace(monotonic=lambda: time.monotonic() * 4, sleep=time.sleep)
        ns = {"os": types.SimpleNamespace(kill=desktop.kill), "signal": signal, "time": fast, "Path": Path,
              "sys": __import__("sys"), "live": desktop.live}
        exec(_functions(["state", "still_running", "freeze_and_kill"]), ns)
        ns["live"] = desktop.live
        return ns["freeze_and_kill"](4242)

    def test_respawned_desktop_is_stopped_not_failed(self):
        desktop = _Desktop(respawn=True)
        killed, frozen = self._run(desktop)
        self.assertGreater(killed, 1)
        self.assertEqual(frozen, 1)
        self.assertEqual([desktop.state(p) for p in desktop.live(0)], ["T"])

    def test_a_survivor_that_cannot_be_stopped_still_fails(self):
        with self.assertRaisesRegex(RuntimeError, "could not be stopped"):
            self._run(_Desktop(respawn=True, stoppable=False))

    def test_clean_kill_leaves_nothing(self):
        desktop = _Desktop(respawn=False)
        self.assertEqual(self._run(desktop), (1, 0))
        self.assertEqual(desktop.live(0), [])

    def test_the_metric_records_what_was_left_stopped(self):
        snapshot = next(n for n in ast.parse(VERIFY.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "snapshot")
        src = ast.unparse(snapshot).replace("(", "").replace(")", "").replace(" ", "")
        self.assertIn("killed,frozen=freeze_and_killuid", src)
        self.assertIn("agent_processes_left=floatfrozen", src)


if __name__ == "__main__":
    unittest.main()
