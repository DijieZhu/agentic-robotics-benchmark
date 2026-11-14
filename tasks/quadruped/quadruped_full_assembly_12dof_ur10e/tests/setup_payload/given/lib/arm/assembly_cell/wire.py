"""JSON-lines RPC between the agent's program (caller) and the grader's CellServer (server).

One JSON object per line. The agent process writes requests on ITS stdout and reads replies on ITS
stdin:  {"id": n, "method": "step", "params": {"n_ticks": 10}}  ->  {"id": n, "result": {...}} or
{"id": n, "error": "forbidden"}. The grader spawns the program with `runuser -u <agent user>` (F2),
keeps a per-request silence deadline, a wall-clock cap for the whole episode, and never trusts
anything on the wire as a scoring input (F1). The child's stderr goes to a log file.
"""

import json
import os
import queue
import signal
import subprocess
import threading
import time

__all__ = ["AgentProcess", "serve_episode", "SolverMissing"]

MAX_LINE = 4 * 1024 * 1024


class SolverMissing(Exception):
    """No program where the task said one would be: an honest zero, not a task error."""


class AgentProcess(object):
    def __init__(self, argv, cwd, user=None, env=None, stderr_path=None, cpu_s=None):
        """`argv` as given; `user` -> `runuser -u <user> --` (root only), `cpu_s` -> `prlimit --cpu` (see runner.wrap_limits).
        The child starts its own session so the whole process group can be killed at the end of the episode."""
        from .runner import wrap_limits
        cmd = wrap_limits(list(argv), cpu_s=cpu_s, user=user)
        self.stderr_file = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,   # noqa: S603
                                     stderr=self.stderr_file, env=env, text=True, bufsize=1, start_new_session=True)
        self._q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                if len(line) > MAX_LINE:
                    self._q.put(("toolong", None))
                    break
                self._q.put(("line", line))
        except Exception:  # noqa: BLE001
            pass
        self._q.put(("eof", None))

    def next_request(self, timeout):
        """(kind, obj): kind in {"req", "eof", "bad"}; raises queue.Empty on silence."""
        while True:
            kind, line = self._q.get(timeout=timeout)
            if kind == "eof":
                return "eof", None
            if kind == "toolong":
                return "bad", "line too long"
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                return "bad", line[:120]
            if not isinstance(obj, dict) or "method" not in obj:
                return "bad", line[:120]
            return "req", obj

    def reply(self, obj):
        try:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def alive(self):
        return self.proc.poll() is None

    def terminate(self, grace=5.0):
        """Close the wire, give the program `grace` seconds to exit, then SIGKILL its whole process group."""
        try:
            self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)      # children the program left behind (its own session)
        except (OSError, ProcessLookupError):
            pass
        if self.stderr_file not in (None, subprocess.DEVNULL):
            try:
                self.stderr_file.close()
            except Exception:  # noqa: BLE001
                pass


def serve_episode(server, proc, wall_cap_s, silence_s, log=None):
    """Answer the agent's requests until the episode ends. Returns the end reason."""
    log = log or (lambda s: None)
    t0 = time.time()
    server.wall_start = t0
    reason = None
    n_req = 0
    while True:
        remaining = wall_cap_s - (time.time() - t0)
        if remaining <= 0:
            reason = "wall_cap"
            break
        try:
            kind, obj = proc.next_request(min(silence_s, remaining))
        except queue.Empty:
            reason = "wall_cap" if time.time() - t0 >= wall_cap_s else "silence"
            break
        if kind == "eof":
            reason = "eof"
            break
        if kind == "bad":
            log("[wire] protocol error: %r" % (obj,))
            reason = "protocol"
            break
        n_req += 1
        method = str(obj.get("method"))
        params = obj.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        rid = obj.get("id")
        result = server.handle(method, params)
        if isinstance(result, dict) and "error" in result and len(result) == 1:
            ok = proc.reply({"id": rid, "error": result["error"]})
        else:
            ok = proc.reply({"id": rid, "result": result})
        if not ok:
            reason = "eof"
            break
        if server.done and method == "finish":
            reason = "finish"
            break
        if server.done and server.done_reason == "sim_cap":
            # the program may keep calling; give it the chance to finish gracefully, but stop when it
            # has had its state and still steps
            if method == "step" and n_req > 0:
                server.steps_after_done = getattr(server, "steps_after_done", 0) + 1
                if server.steps_after_done > 50:
                    reason = "sim_cap"
                    break
    server.wall_used = time.time() - t0
    if not server.done:
        server._end(reason)
    return reason
