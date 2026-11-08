"""Run a program over the real wire against a practice cell (PUBLIC CODE; used by ~/practice/grade.py).

The program is started exactly as the grader starts it (`bash run.sh` when the submission ships one, else the client
launcher with the library on sys.path), as the current user, and served one JSON-lines request at a time with the same
budgets as the grader: the silence limit, the program-time cap (the wall clock minus the cell's own time inside step and
get_camera) and the limits after the episode is over. Nothing the program prints on the wire is a scoring input: the
cell's own state is.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

from .client import launcher_argv

__all__ = ["ProgramProcess", "serve_program"]

MAX_LINE = 4 * 1024 * 1024
# once the episode is over (sim cap, physics fault) the program gets this much more, as on the grader's wire
AFTER_DONE_MAX_STEPS = 50
AFTER_DONE_MAX_REQUESTS = 200
AFTER_DONE_MAX_S = 60.0


class ProgramProcess(object):
    def __init__(self, argv, cwd, env=None, stderr=None):
        self.proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,   # noqa: S603
                                     stderr=stderr, env=env, text=True, bufsize=1, start_new_session=True)
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

    def terminate(self, grace=5.0):
        try:
            self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def serve_program(server, program_dir, lib_dir, program_cap_s, silence_s, log=None, module="packer"):
    """Start the program in `program_dir` and answer its requests until the episode ends. Returns the end reason.

    `program_cap_s` caps the program time (the wall clock minus the server's `work_wall`); `silence_s` caps the time
    from a reply to the next request."""
    log = log or (lambda s: None)
    argv = launcher_argv(sys.executable, lib_dir, program_dir, module)
    env = dict(os.environ)
    env["PYTHONPATH"] = lib_dir
    env.setdefault("ARB_GIVEN_DIR", os.path.dirname(os.path.dirname(lib_dir)))
    proc = ProgramProcess(argv, cwd=program_dir, env=env, stderr=sys.stderr)
    t0 = time.time()
    server.wall_start = t0

    def program_time():
        return (time.time() - t0) - float(getattr(server, "work_wall", 0.0))

    reason, after_done, done_since = None, 0, None
    try:
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
            method = str(obj.get("method"))
            params = obj.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            result = server.handle(method, params)
            if isinstance(result, dict) and "error" in result and len(result) == 1:
                ok = proc.reply({"id": obj.get("id"), "error": result["error"]})
            else:
                ok = proc.reply({"id": obj.get("id"), "result": result})
            if not ok:
                reason = "eof"
                break
            if server.done and method == "finish":
                reason = "finish"
                break
            if server.done:
                # the episode is over and nothing changes any more: the program may still read its state and finish
                if done_since is None:
                    done_since = time.time()
                after_done += 1
                if method == "step":
                    server.steps_after_done = getattr(server, "steps_after_done", 0) + 1
                if (getattr(server, "steps_after_done", 0) > AFTER_DONE_MAX_STEPS or after_done > AFTER_DONE_MAX_REQUESTS
                        or time.time() - done_since > AFTER_DONE_MAX_S):
                    reason = server.done_reason or "done"
                    break
    finally:
        proc.terminate()
    server.wall_used = time.time() - t0
    server.program_used = program_time()
    if not server.done:
        server._end(reason)
    return reason
