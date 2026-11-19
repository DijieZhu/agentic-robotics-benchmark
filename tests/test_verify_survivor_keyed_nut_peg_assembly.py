"""keyed_nut_peg_assembly: a process of the agent account that outlives the grader's bounded kill (the base image's
supervisord respawns the desktop it runs as that account, as fast as it is killed) is stopped, not held against the
submission; one that is still running once stopped still fails the run (cleanup_failed), before and after the solver.

Runs the grader's own `stop_agent_processes` (and its helpers), taken from tests/verify.py by name, against a
stand-in for the account's process list and real child processes playing the respawned desktop (each SIGKILL starts
the next one, as supervisord does). The clock runs five times as fast, so the 5 s kill takes 1 s. Stdlib only:
tests/verify.py imports the simulator (mujoco, numpy) that the host venv does not have.
"""
import ast
import os
import signal
import subprocess
import time
import types
import unittest
from pathlib import Path

VERIFY = Path(__file__).resolve().parents[1] / "tasks/arm/keyed_nut_peg_assembly/tests/verify.py"


class _Desktop:
    """The account's processes: one child at a time; `respawn` True: a SIGKILL starts the next one."""

    def __init__(self, respawn, stoppable=True):
        self.respawn, self.stoppable = respawn, stoppable
        self.procs = [subprocess.Popen(["sleep", "60"])]

    def pids(self, uid):
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


def _tree():
    return ast.parse(VERIFY.read_text())


def _functions(names):
    body = [n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names), "verify.py lacks %s" % (set(names) - {n.name for n in body})
    return compile(ast.Module(body=body, type_ignores=[]), str(VERIFY), "exec")


class SurvivorAroundTheSolver(unittest.TestCase):
    def _run(self, desktop):
        self.addCleanup(desktop.close)
        logs = []
        fast = types.SimpleNamespace(monotonic=lambda: time.monotonic() * 5, sleep=lambda s: time.sleep(s / 5))
        ns = {"os": types.SimpleNamespace(kill=desktop.kill), "signal": signal, "time": fast, "log": logs.append}
        exec(_functions(["kill_agent_processes", "_proc_state", "_still_running", "stop_agent_processes"]), ns)
        ns["_agent_pids"] = desktop.pids
        return ns["stop_agent_processes"](4242), logs

    def test_respawned_desktop_is_stopped_not_failed(self):
        desktop = _Desktop(respawn=True)
        ok, logs = self._run(desktop)
        self.assertTrue(ok)
        self.assertGreater(len(desktop.procs), 2)
        self.assertEqual([desktop.state(p) for p in desktop.pids(0)], ["T"])
        self.assertTrue(any("outlived the kill" in m for m in logs), logs)

    def test_a_survivor_that_cannot_be_stopped_still_fails(self):
        ok, _ = self._run(_Desktop(respawn=True, stoppable=False))
        self.assertFalse(ok)

    def test_clean_kill_records_nothing(self):
        desktop = _Desktop(respawn=False)
        ok, logs = self._run(desktop)
        self.assertTrue(ok)
        self.assertEqual(desktop.pids(0), [])
        self.assertEqual(logs, [])

    def test_main_gates_on_the_stop_before_and_after_the_solver(self):
        main = next(n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name == "main")
        src = ast.unparse(main)
        self.assertIn("if not stop_agent_processes(uid):", src)
        self.assertIn("cleanup_ok = stop_agent_processes(uid)", src)
        self.assertNotIn("kill_agent_processes(", src)

    def test_every_episode_starts_after_the_stop(self):
        loop = next(n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name == "run_episodes")
        src = ast.unparse(loop)
        self.assertIn("stop_agent_processes(uid)", src)
        self.assertNotIn("kill_agent_processes(", src)


if __name__ == "__main__":
    unittest.main()
