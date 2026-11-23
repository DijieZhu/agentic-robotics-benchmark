"""JSON-lines RPC between the agent's program (caller) and the grader's ParcelCellServer (server).

One JSON object per line. The program writes requests on ITS stdout and reads replies on ITS stdin:
{"id": n, "method": "step", "params": {"n": 10}}  ->  {"id": n, "result": {...}} or {"id": n, "error": "forbidden"}
(an error may carry a human-readable "detail": {"id": n, "error": "bad_params", "detail": "n must be ..."}). A server
answer that is a dict whose only keys are "error" (and "detail") is sent as an error, anything else as the result.
The grader starts the program as its runner account (`runuser`), keeps the budgets below, and never takes anything
on the wire as a scoring input: the shift is scored on the cell's own world state.

Budgets of one shift (`serve_episode`):
  * program time = the shift's wall clock minus the server's `work_wall` (the cell's own simulation inside `step` and
    rendering inside `get_camera`): the budget measures the program, not how fast the grading machine simulates and
    renders. The budget itself is per shift (`program_cap_s`, the cell's `program_budget_s`: a factor times the
    shift's duration plus a fixed allowance);
  * one silence (from a reply sent to the next request received) longer than `silence_s` ends the shift. Nothing of
    the cell runs during a silence, so its wall clock is program time. Before the first request the program has the
    start window `start_s` instead (its interpreter start and imports), inside the same budget;
  * a reply is written with a deadline (a program that stops reading a megabyte camera reply ends the shift instead
    of blocking the grader); request lines are read with a bounded length;
  * once the shift is over (its duration elapsed, a finish, a cell fault) the program gets at most `after_over`
    steps / requests / seconds more, to read its state and exit.

End reasons (the value `serve_episode` returns):
  finished      the program called finish()
  shift_end     the cell ended the shift: its duration elapsed or its list was settled (also when the program then
                exits or finishes)
  stalled       a silence above `silence_s`, the start window exceeded, or a reply the program did not take in time
  program_time  the program-time budget is used up
  protocol      a request line that is not one JSON request object (or longer than MAX_LINE)
  crashed       the program exited or closed its end of the wire while the shift was live
  or the cell's own end reason when the cell ended the shift otherwise (physics_fault, internal_error, or the
  privileged reference's own time cap, program_time).
Whatever the reason, the shift keeps what had crossed the scan line; the parcels not across count as the shift's setting
says (lost on the rated setting, not done on the line setting: parcelcell/episode.py).

Request-line guard: any failure of the JSON parser on a line (a line nested deeper than the parser's limit included)
is a protocol end, never an exception out of the wire; the request's `method` reaches the server as a string of at
most METHOD_MAX characters ("(not a name)" otherwise, which the server refuses as forbidden) and its `id` as a value
the reply can always echo (null, true / false, a 64-bit integer, a string of at most ID_STR_MAX characters; else
null). `parse_request` is shared with the commissioning service.
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

__all__ = ["AgentProcess", "serve_episode", "SolverMissing", "is_error", "parse_request", "plain_request",
           "plain_id", "END_REASONS", "CELL_REASONS", "MAX_LINE", "START_S", "AFTER_OVER"]

MAX_LINE = 4 * 1024 * 1024
ERROR_KEYS = frozenset(("error", "detail"))
METHOD_MAX = 64
ID_STR_MAX = 256
# program start to the first request: the interpreter's start and the program's imports (the import check's wall limit)
START_S = 120.0
# once the shift is over: this much more before the wire closes (steps, requests of any kind, seconds); a finish ends
# it at once
AFTER_OVER = {"steps": 50, "requests": 200, "seconds": 60.0}
END_REASONS = ("finished", "shift_end", "stalled", "program_time", "protocol", "crashed")
# the cell's own end reasons as end reasons of the shift: the cell names them as the wire does (finished, shift_end,
# program_time, physics_fault, internal_error); the map only renames a legacy name
CELL_REASONS = {"finish": "finished"}


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


def plain_id(rid):
    """A request id as the reply echoes it: null, true / false, a 64-bit integer or a short string; else null."""
    if rid is None or isinstance(rid, bool):
        return rid
    if isinstance(rid, int) and -(1 << 63) <= rid < (1 << 63):
        return rid
    if isinstance(rid, str) and len(rid) <= ID_STR_MAX:
        return rid
    return None


def plain_request(obj):
    """The request object with a plain `method` and `id` (in place; returned)."""
    m = obj.get("method")
    obj["method"] = m[:METHOD_MAX] if isinstance(m, str) else "(not a name)"
    obj["id"] = plain_id(obj.get("id"))
    return obj


def parse_request(line):
    """(kind, obj) of one request line: ("req", request), ("bad", text) or (None, None) for a blank line."""
    if isinstance(line, bytes):
        line = line.decode("utf-8", "replace")
    line = line.strip()
    if not line:
        return None, None
    try:
        obj = json.loads(line)
    except Exception:  # noqa: BLE001 - ValueError, RecursionError, MemoryError: the program's doing either way
        return "bad", "(a request line the JSON parser could not read) " + line[:80]
    if not isinstance(obj, dict) or "method" not in obj:
        return "bad", line[:120]
    return "req", plain_request(obj)


class SolverMissing(Exception):
    """No program where the task said one would be: an honest zero, not a task error."""


class AgentProcess(object):
    def __init__(self, argv, cwd, user=None, env=None, stderr_path=None, cpu_s=None, nonet=False, max_line=MAX_LINE):
        """`argv` as given; `user` -> `runuser -u <user> --` (root only), `cpu_s` -> `prlimit --cpu` (see runner.wrap_limits).
        The child starts its own session so the whole process group can be killed at the end of the shift.

        Replies are written to the program's stdin through a NON-BLOCKING descriptor with a deadline (`reply`): a camera
        frame is about a megabyte, far above a pipe's buffer, and a program that asks for one and then stops reading
        must not hold the grader past its caps. Requests are read with a bounded line length (no unbounded line in
        memory: `max_line` bytes)."""
        from .runner import wrap_limits
        self.max_line = int(max_line)
        cmd = wrap_limits(list(argv), cpu_s=cpu_s, user=user, nonet=nonet)
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
                line = reader.readline(self.max_line + 1)
                if not line:
                    break
                if len(line) > self.max_line and not line.endswith("\n"):
                    self._q.put(("toolong", None))
                    break
                self._q.put(("line", line))
        except Exception:  # noqa: BLE001
            pass
        self._q.put(("eof", None))

    def next_request(self, timeout):
        """(kind, obj): kind in {"req", "eof", "bad"}; raises queue.Empty on silence."""
        deadline = time.time() + float(timeout)
        while True:
            kind, line = self._q.get(timeout=max(0.0, deadline - time.time()))
            if kind == "eof":
                return "eof", None
            if kind == "toolong":
                return "bad", "line too long"
            k, obj = parse_request(line)
            if k is None:
                continue
            return k, obj

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


def serve_episode(server, proc, program_cap_s, silence_s, log=None, start_s=START_S, after_over=None):
    """Answer the program's requests until the shift ends. Returns the end reason (module docstring).

    `proc`: anything with next_request(timeout) -> (kind, obj) raising queue.Empty on silence, reply(obj, timeout) ->
    bool and a `stalled` flag (AgentProcess; the commissioning service's socket). `program_cap_s`: the shift's
    program-time budget; `silence_s`: the cap on one silence; `start_s`: the start window before the first request;
    `after_over`: {"steps", "requests", "seconds"} once the shift is over (AFTER_OVER by default)."""
    log = log or (lambda s: None)
    caps = dict(AFTER_OVER)
    caps.update(after_over or {})
    program_cap_s, silence_s = float(program_cap_s), float(silence_s)
    start_s = silence_s if start_s is None else float(start_s)
    t0 = time.time()
    server.wall_start = t0

    def program_time():
        return (time.time() - t0) - float(getattr(server, "work_wall", 0.0))

    reason = None
    n_req = 0
    after = 0
    done_since = None
    while True:
        remaining = program_cap_s - program_time()
        if remaining <= 0:
            reason = "program_time"
            break
        limit = silence_s if n_req else start_s
        try:
            kind, obj = proc.next_request(min(limit, remaining))
        except queue.Empty:
            if program_time() >= program_cap_s:
                reason = "program_time"
            else:
                reason = "stalled"
                log("[wire] no request within %.0f s %s" % (limit, "of the program's start" if not n_req else "of the last reply"))
            break
        if kind == "eof":
            reason = "crashed"
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
        was_done = bool(server.done)
        result = server.handle(method, params)
        if is_error(result):
            msg = {"id": rid, "error": result["error"]}
            if "detail" in result:
                msg["detail"] = result["detail"]
        else:
            msg = {"id": rid, "result": result}
        # the program must take its reply within the silence cap too (and within its budget)
        ok = proc.reply(msg, timeout=max(1.0, min(silence_s, program_cap_s - program_time())))
        if not ok:
            reason = "stalled" if getattr(proc, "stalled", False) else "crashed"
            if reason == "stalled":
                log("[wire] the program did not read its reply to %s within the deadline" % method)
            break
        if server.done:
            if method == "finish":
                break
            # the shift is over and nothing changes any more: the program may still read its state and finish, but
            # it cannot keep the grader busy until its budget is used up
            if done_since is None:
                done_since = time.time()
            after += 1
            if method == "step" and was_done:
                server.steps_after_done = int(getattr(server, "steps_after_done", 0)) + 1
            if (int(getattr(server, "steps_after_done", 0)) > int(caps["steps"]) or after > int(caps["requests"])
                    or time.time() - done_since > float(caps["seconds"])):
                break
    server.wall_used = time.time() - t0
    server.program_used = program_time()
    if server.done:
        cell = server.done_reason or "shift_end"
        return CELL_REASONS.get(cell, cell)
    server._end(reason)
    return reason
