"""The shared wire with this task's guard.

New file of this task, beside the vendored arm/assembly_cell/wire.py (unchanged; its `serve_episode` is used as it
is). The shared wire catches only ValueError around json.loads, so one request line nested about 10,000 deep
(RecursionError, MemoryError) would escape `serve_episode` and end the verify as a task error that a submission could
cause at will (a parser's own limits are hostile input too). `GuardedAgentProcess` (a wire.AgentProcess) turns such a
line into the program's protocol error and hands `serve_episode` only a sanitized request: the method a string of at
most 200 characters (else "<invalid>"), params an object (else {}), the id an integer, a short string or null (else
null), so that nothing the program sent is repr()'d or re-serialised deeply. Replies that cannot be serialised become
{"id": null, "error": "unserialisable"}.

Two bounds of its own: the program's pipe is read in pieces of at most `max_line` characters (default the wire's
MAX_LINE), so a line that passes that limit, with or without its newline, ends the episode as a protocol error after
at most one line's worth has been read; the program's standard error goes to the log file through a pipe and a
writer thread that keeps its first `stderr_cap_bytes` (default STDERR_CAP_MB = 5 MB) and drops the rest, the log's
last line then saying how many bytes were dropped (`drain_capped`, the same writer the grader's import check uses).
`serve(server, argv, cwd, env, wall_cap_s=None, stderr_path=None)` runs one episode over the wire.
"""
import json
import os
import queue
import subprocess
import threading

from ..assembly_cell import wire as W

__all__ = ["GuardedAgentProcess", "serve", "drain_capped", "STDERR_CAP_MB", "STDERR_CAP_BYTES"]

STDERR_CAP_MB = 5
STDERR_CAP_BYTES = STDERR_CAP_MB * 1024 * 1024


def drain_capped(fd, log, cap, tail_bytes=0):
    """the capped writer of a program's standard error: reads the pipe `fd` in pieces of 64 KiB until its end, writes
    the first `cap` bytes into the open binary file `log` and drops the rest, then writes one last line saying how many
    bytes were dropped (when any were) and closes `log`. Returns {"kept", "dropped", "tail"}: "tail" holds the last
    `tail_bytes` bytes read, kept or dropped (b"" for 0)."""
    kept = dropped = 0
    tail = bytearray()
    try:
        while True:
            b = os.read(fd, 1 << 16)
            if not b:
                break
            room = cap - kept
            if room > 0:
                log.write(b[:room])
                kept += min(room, len(b))
            dropped += max(0, len(b) - max(room, 0))
            if tail_bytes:
                tail += b[-tail_bytes:]
                del tail[:max(0, len(tail) - tail_bytes)]
    except Exception:  # noqa: BLE001
        pass
    try:
        if dropped:
            log.write(b"\n[grader] the program's standard error was cut at %d bytes: %d bytes dropped\n" % (cap, dropped))
        log.close()
    except Exception:  # noqa: BLE001
        pass
    return {"kept": kept, "dropped": dropped, "tail": bytes(tail)}


def _clean_id(v):
    if v is None or (isinstance(v, int) and not isinstance(v, bool) and abs(v) < 2 ** 53):
        return v
    if isinstance(v, str) and len(v) <= 64:
        return v
    return None


class GuardedAgentProcess(W.AgentProcess):
    def __init__(self, argv, cwd, user=None, env=None, stderr_path=None, cpu_s=None, max_line=None, stderr_cap_bytes=None):
        """wire.AgentProcess's start, with the program's standard error through a capped pipe into `stderr_path`"""
        from ..assembly_cell.runner import wrap_limits
        self.max_line = int(W.MAX_LINE if max_line is None else max_line)
        self.stderr_cap = int(STDERR_CAP_BYTES if stderr_cap_bytes is None else stderr_cap_bytes)
        self.stderr_kept = self.stderr_dropped = 0
        cmd = wrap_limits(list(argv), cpu_s=cpu_s, user=user)
        self._err_log = open(stderr_path, "ab") if stderr_path else None
        self.stderr_file = subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,   # noqa: S603
                                     stderr=subprocess.PIPE if self._err_log else subprocess.DEVNULL, env=env, text=True,
                                     bufsize=1, start_new_session=True)
        self._q = queue.Queue()
        self._err_thread = None
        if self._err_log:
            self._err_thread = threading.Thread(target=self._drain_stderr, daemon=True)
            self._err_thread.start()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        """the wire's lines, read in pieces of at most max_line characters: a longer line is 'toolong'"""
        try:
            while True:
                line = self.proc.stdout.readline(self.max_line + 1)
                if not line:
                    break
                if len(line) > self.max_line:
                    self._q.put(("toolong", None))
                    break
                self._q.put(("line", line))
        except Exception:  # noqa: BLE001
            pass
        self._q.put(("eof", None))

    def _drain_stderr(self):
        c = drain_capped(self.proc.stderr.fileno(), self._err_log, self.stderr_cap)
        self.stderr_kept, self.stderr_dropped = c["kept"], c["dropped"]

    def terminate(self, grace=5.0):
        W.AgentProcess.terminate(self, grace)
        if self._err_thread is not None:
            self._err_thread.join(timeout=grace)

    def next_request(self, timeout):
        try:
            kind, obj = W.AgentProcess.next_request(self, timeout)
        except (RecursionError, MemoryError):
            return "bad", "unparsable request line"
        if kind != "req":
            return kind, obj
        method = obj.get("method")
        if not isinstance(method, str) or len(method) > 200:
            method = "<invalid>"
        params = obj.get("params")
        if params is not None and not isinstance(params, dict):
            params = {}
        return "req", {"id": _clean_id(obj.get("id")), "method": method, "params": params or {}}

    def reply(self, obj):
        try:
            text = json.dumps(obj)
        except (TypeError, ValueError, RecursionError, OverflowError):
            text = json.dumps({"id": None, "error": "unserialisable"})
        try:
            self.proc.stdin.write(text + "\n")
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False


def serve(server, argv, cwd, env=None, wall_cap_s=None, stderr_path=None, user=None, cpu_s=None, log=None):
    """one episode of `server` (HumanoidCellServer) with the program `argv` (its launcher) over the guarded wire;
    returns the end reason ("finish", "eof", "silence", "wall_cap", "sim_cap", "protocol")"""
    wc, silence = server.wire_budgets(wall_cap_s)
    proc = GuardedAgentProcess(argv, cwd=cwd, user=user, env=env, stderr_path=stderr_path, cpu_s=cpu_s)
    try:
        return W.serve_episode(server, proc, wc, silence, log=log)
    finally:
        proc.terminate()
