"""JSON-lines RPC between the agent's program (caller) and the grader's CellServer (server).

One JSON object per line. The agent process writes requests on ITS stdout and reads replies on ITS
stdin:  {"id": n, "method": "step", "params": {"n_ticks": 10}}  ->  {"id": n, "result": {...}} or
{"id": n, "error": "forbidden"} (an error may carry a human-readable "detail": {"id": n, "error": "bad_params",
"detail": "n_ticks must be ..."}). A server answer that is a dict whose only keys are "error" (and "detail") is sent
as an error, anything else as the result. The grader spawns the program with `runuser -u <runner account>` (F2),
keeps a per-request silence deadline, a cap on the program's time for the whole episode, and never trusts
anything on the wire as a scoring input (F1). The child's stderr goes to a log file.

Program time = the episode's wall clock minus the server's `work_wall` (the cell's own simulation and rendering, which
the sim cap and the frame budget bound): the budget then measures the program, not how fast the grading machine
simulates and renders (a cap on the plain wall clock made the same program score differently on another machine).

Adapted from the arm tasks' copy: error replies may carry "detail"; replies are written with a deadline (a program that
stops reading its megabyte camera reply ends the episode as "stalled" instead of blocking the grader); request lines
are read with a bounded length; and once the episode is over (sim cap, cell error) the program gets at most
AFTER_DONE_MAX_STEPS steps / AFTER_DONE_MAX_REQUESTS requests / AFTER_DONE_MAX_S seconds more.
End reasons: finish, eof, silence, stalled, protocol, program_time_cap, or the cell's own (sim_cap, physics_fault, internal_error).
"""

import io
import json
import os
import queue
import select
import signal
import subprocess
import threading
import time

__all__ = ["AgentProcess", "serve_episode", "SolverMissing", "is_error"]

MAX_LINE = 4 * 1024 * 1024
ERROR_KEYS = frozenset(("error", "detail"))
# once the episode is over (sim cap, or an error inside the cell) the program gets this much more before the wire
# closes: steps, requests of any kind, seconds (a `finish` ends it at once)
AFTER_DONE_MAX_STEPS = 50
AFTER_DONE_MAX_REQUESTS = 200
AFTER_DONE_MAX_S = 60.0


def _jsonable(obj):
    """json.dumps fallback: numpy scalars / arrays as plain values, anything else as its str (a reply never fails to
    serialise)."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    return str(obj)


def is_error(result):
    """True when a server answer is an error reply ({"error": code} or {"error": code, "detail": text})."""
    return isinstance(result, dict) and "error" in result and set(result) <= ERROR_KEYS


class SolverMissing(Exception):
    """No program where the task said one would be: an honest zero, not a task error."""


class AgentProcess(object):
    def __init__(self, argv, cwd, user=None, env=None, stderr_path=None, cpu_s=None):
        """`argv` as given; `user` -> `runuser -u <user> --` (root only), `cpu_s` -> `prlimit --cpu` (see runner.wrap_limits).
        The child starts its own session so the whole process group can be killed at the end of the episode.

        Replies are written to the program's stdin through a NON-BLOCKING descriptor with a deadline (`reply`): a camera
        frame is megabytes, far above a pipe's buffer, and a program that asks for one and then stops reading must not
        hold the grader past its caps. Requests are read with a bounded line length (no unbounded line in memory)."""
        from .runner import wrap_limits
        cmd = wrap_limits(list(argv), cpu_s=cpu_s, user=user)
        self.stderr_file = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,   # noqa: S603
                                     stderr=self.stderr_file, env=env, start_new_session=True)
        self._in_fd = self.proc.stdin.fileno()
        os.set_blocking(self._in_fd, False)
        self.stalled = False           # a reply could not be delivered before its deadline
        self._q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            reader = io.TextIOWrapper(self.proc.stdout, encoding="utf-8", errors="replace", newline="\n")
            while True:
                line = reader.readline(MAX_LINE + 1)
                if not line:
                    break
                if len(line) > MAX_LINE and not line.endswith("\n"):
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

    def reply(self, obj, timeout=None):
        """Write one reply line; False when the program is gone or (`timeout` s) did not take it in time (`stalled`)."""
        view = memoryview((json.dumps(obj, default=_jsonable) + "\n").encode("utf-8"))
        deadline = None if timeout is None else time.time() + float(timeout)
        while len(view):
            try:
                n = os.write(self._in_fd, view)
                view = view[n:]
                continue
            except BlockingIOError:
                pass
            except (OSError, ValueError):            # BrokenPipeError: the program closed its end or exited
                return False
            wait = None if deadline is None else deadline - time.time()
            if wait is not None and wait <= 0:
                self.stalled = True
                return False
            try:
                select.select([], [self._in_fd], [], wait)
            except (OSError, ValueError):
                return False
        return True

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


def serve_episode(server, proc, program_cap_s, silence_s, log=None):
    """Answer the agent's requests until the episode ends. Returns the end reason.

    `program_cap_s` caps the program time (module docstring); `silence_s` caps the time from a reply to the next
    request."""
    log = log or (lambda s: None)
    t0 = time.time()
    server.wall_start = t0

    def program_time():
        return (time.time() - t0) - float(getattr(server, "work_wall", 0.0))

    reason = None
    n_req = 0
    after_done = 0
    done_since = None
    while True:
        remaining = program_cap_s - program_time()
        if remaining <= 0:
            reason = "program_time_cap"
            break
        try:
            kind, obj = proc.next_request(min(silence_s, remaining))
        except queue.Empty:
            reason = "program_time_cap" if program_time() >= program_cap_s else "silence"
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
        params = obj.get("params")
        if params is None:
            params = {}                     # anything else that is not an object is the server's bad_params
        rid = obj.get("id")
        result = server.handle(method, params)
        if is_error(result):
            msg = {"id": rid, "error": result["error"]}
            if "detail" in result:
                msg["detail"] = result["detail"]
        else:
            msg = {"id": rid, "result": result}
        # the program must take its reply within the silence deadline too (and within its time cap)
        ok = proc.reply(msg, timeout=max(1.0, min(silence_s, program_cap_s - program_time())))
        if not ok:
            reason = "stalled" if getattr(proc, "stalled", False) else "eof"
            if reason == "stalled":
                log("[wire] the program did not read its reply to %s within %.0f s" % (method, silence_s))
            break
        if server.done and method == "finish":
            reason = "finish"
            break
        if server.done:
            # the episode is over (sim cap, or an error inside the cell) and nothing changes any more: the program may
            # still read its state and finish gracefully, but it cannot keep the grader busy until its time cap
            if done_since is None:
                done_since = time.time()
            after_done += 1
            if method == "step":
                server.steps_after_done = getattr(server, "steps_after_done", 0) + 1
            if (getattr(server, "steps_after_done", 0) > AFTER_DONE_MAX_STEPS or after_done > AFTER_DONE_MAX_REQUESTS
                    or time.time() - done_since > AFTER_DONE_MAX_S):
                reason = server.done_reason or "done"
                break
    server.wall_used = time.time() - t0
    server.program_used = program_time()
    if not server.done:
        server._end(reason)
    return reason
