"""The grader's side of the wire: run a program as its own process and answer its JSON lines.

    reason = serve(venue, argv, cwd=run_dir, env=env, stderr_path=log)

Single-threaded on purpose: the request is read with select() + os.read on a non-blocking pipe, so the arrival time
of a request is taken when its line is complete (no reader thread between the program and the clock), and a reply is
written with a deadline (a program that stops reading cannot hold the grader). Program time = wall clock from the moment
a reply is ready to be written (a write into the pipe takes microseconds; a write that blocks because the program
does not read is the program's time) to the arrival of the next request; it is charged to the open session (on-track) or to the desk time before
the next session by `venue.account`. Budgets: on-track and desk time (the venue's caps), silence
(`budgets.silence_s` between a reply and the next request), the import check (`budgets.import_check_s` from the
start of the process to its first request), CPU (`/proc/<pid>/stat` of the program, every `cpu_poll_s`, against
`budgets.cpu_factor` x the wall caps). After the venue is over the program may make `after_over.max_requests` more
requests within `after_over.max_s` (all answered `venue_over`); then the wire closes.

End reasons (records.schema.json result.end_reason): finish, eof, silence, import_timeout (no first request within
`budgets.import_check_s`), stalled, protocol, program_time_cap, desk_time_cap, cpu_cap,
void, physics_fault, internal_error.
"""

import json
import os
import select
import signal
import subprocess
import time
import traceback

from carkit import protocol

__all__ = ["ProgramProcess", "serve", "encode_reply"]

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def encode_reply(rid, result):
    """One reply line as bytes. A step result is written by hand (it is the hot path: 1081 integers); everything else
    through json.dumps. Both give compact JSON."""
    if protocol_is_error(result):
        msg = {"id": rid, "error": result["error"]}
        if result.get("detail"):
            msg["detail"] = result["detail"]
        return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
    if isinstance(result, dict) and "scan" in result and "tick" in result and "truth" not in result:
        head = '{"id":%d,"result":{"tick":%d,"scan":[' % (rid, result["tick"])
        body = ",".join(map(str, result["scan"]))
        tail = '],"imu":%s,"wheel_speed":%s,"steer_echo":%s,"events":%s}}\n' % (
            json.dumps(result["imu"], separators=(",", ":")), json.dumps(result["wheel_speed"]),
            json.dumps(result["steer_echo"]), json.dumps(result["events"], separators=(",", ":")))
        return (head + body + tail).encode("ascii")
    return (json.dumps({"id": rid, "result": result}, separators=(",", ":")) + "\n").encode("utf-8")


def protocol_is_error(result):
    return isinstance(result, dict) and "error" in result and set(result) <= {"error", "detail"}


class ProgramProcess(object):
    """The program's process: stdin / stdout pipes (non-blocking on our side), its own session (process group)."""

    def __init__(self, argv, cwd=None, env=None, stderr_path=None):
        self.stderr_file = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(list(argv), cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,  # noqa: S603
                                     stderr=self.stderr_file, start_new_session=True, bufsize=0)
        self.rfd = self.proc.stdout.fileno()
        self.wfd = self.proc.stdin.fileno()
        os.set_blocking(self.rfd, False)
        os.set_blocking(self.wfd, False)
        self.buf = bytearray()
        self.eof = False
        self.stalled = False
        self.t_start = time.perf_counter()

    def read_line(self, timeout):
        """("line", bytes), ("eof", None), ("timeout", None) or ("toolong", None)."""
        deadline = time.perf_counter() + max(0.0, timeout)
        while True:
            i = self.buf.find(b"\n")
            if i >= 0:
                line = bytes(self.buf[:i])
                del self.buf[:i + 1]
                return "line", line
            if len(self.buf) > protocol.MAX_LINE_BYTES:
                return "toolong", None
            if self.eof:
                return "eof", None
            wait = deadline - time.perf_counter()
            if wait <= 0.0:
                return "timeout", None
            try:
                r, _, _ = select.select([self.rfd], [], [], wait)
            except (OSError, ValueError):
                self.eof = True
                continue
            if not r:
                continue
            try:
                chunk = os.read(self.rfd, 1 << 20)
            except BlockingIOError:
                continue
            except OSError:
                chunk = b""
            if not chunk:
                self.eof = True
            else:
                self.buf += chunk

    def write(self, data, timeout):
        """Write one reply; False when the program is gone or did not take it within `timeout` s (`stalled`)."""
        view = memoryview(data)
        deadline = time.perf_counter() + timeout
        while len(view):
            try:
                n = os.write(self.wfd, view)
                view = view[n:]
                continue
            except BlockingIOError:
                pass
            except (OSError, ValueError):
                return False
            wait = deadline - time.perf_counter()
            if wait <= 0.0:
                self.stalled = True
                return False
            try:
                select.select([], [self.wfd], [], wait)
            except (OSError, ValueError):
                return False
        return True

    def cpu_s(self):
        """CPU seconds of the program (user + system, its threads, and the children it waited for)."""
        try:
            with open("/proc/%d/stat" % self.proc.pid, "rb") as f:
                fields = f.read().rsplit(b")", 1)[1].split()
            return sum(int(fields[i]) for i in (11, 12, 13, 14)) / float(CLK_TCK)
        except (OSError, IndexError, ValueError):
            return 0.0

    def alive(self):
        return self.proc.poll() is None

    def terminate(self, grace=5.0):
        """Close the wire, give the program `grace` seconds, then SIGKILL its whole process group."""
        for f in (self.proc.stdin,):
            try:
                f.close()
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
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        try:
            self.proc.stdout.close()
        except Exception:  # noqa: BLE001
            pass
        if self.stderr_file not in (None, subprocess.DEVNULL):
            try:
                self.stderr_file.close()
            except Exception:  # noqa: BLE001
                pass


def serve(venue, argv, cwd=None, env=None, stderr_path=None, log=None, cpu_poll_s=1.0, grace_s=5.0, stats=None):
    """Run `argv` (the program) against `venue` until the venue or the program ends. Returns the venue's end reason
    (also in venue.result()). `stats` (a dict) receives counters and the per-request program times."""
    log = log or (lambda s: None)
    cfg = venue.cfg
    silence = float(cfg["budgets"]["silence_s"])
    import_cap = float(cfg["budgets"]["import_check_s"])
    cpu_cap = float(venue.caps["cpu"])
    ao = venue.params.get("after_over", {"max_requests": 50, "max_s": 10.0})
    st = stats if stats is not None else {}
    st.update({"requests": 0, "program_times": [], "cell_times": [], "handle_times": [], "after_over": 0, "cpu_s": 0.0})
    proc = ProgramProcess(argv, cwd=cwd, env=env, stderr_path=stderr_path)
    reason = None
    first = True
    t_last = proc.t_start
    t_over = None
    t_cpu = time.perf_counter()
    try:
        while reason is None:
            if venue.over:
                if t_over is None:
                    t_over = time.perf_counter()
                if st["after_over"] >= int(ao["max_requests"]) or time.perf_counter() - t_over > float(ao["max_s"]):
                    reason = "eof"
                    break
            left = venue.budget_left()
            if first:
                timeout = min(silence, import_cap)
            elif venue.over:
                timeout = max(0.0, float(ao["max_s"]) - (time.perf_counter() - t_over))
            else:
                timeout = min(silence, left)
            kind, line = proc.read_line(timeout)
            now = time.perf_counter()
            dt = now - t_last
            if kind == "timeout":
                if first or venue.over:
                    reason = ("import_timeout" if import_cap <= silence else "silence") if first else "eof"
                else:
                    r = venue.account(dt)
                    reason = r or "silence"
                break
            if kind == "eof":
                if not first:
                    venue.account(dt)
                reason = "eof"
                break
            if kind == "toolong":
                reason = "protocol"
                break
            try:
                obj = json.loads(line)
            except ValueError:
                obj = None
            env_err = protocol.check_envelope(obj)
            if env_err is not None and env_err[0] == "protocol":
                log("[wire] protocol error: %r" % (line[:120],))
                reason = "protocol"
                break
            if not first:
                r = venue.account(dt)
                if r is not None:
                    reason = r
                    break
                st["program_times"].append(dt)
            first = False
            st["requests"] += 1
            if venue.over:
                st["after_over"] += 1
            tc = time.perf_counter()
            if env_err is not None:
                venue.counts["bad_calls"] += 1
                rid = obj.get("id") if isinstance(obj, dict) else None
                data = (json.dumps(protocol.error_reply(rid, "bad_params", env_err[1]), separators=(",", ":")) + "\n").encode("utf-8")
            else:
                try:
                    result = venue.handle(obj["method"], obj.get("params"))
                    st["handle_times"].append(time.perf_counter() - tc)
                except Exception:  # noqa: BLE001
                    log("[wire] internal error:\n" + traceback.format_exc())
                    venue.end("internal_error")
                    reason = "internal_error"
                    break
                data = encode_reply(obj["id"], result)
            t_last = time.perf_counter()          # the reply exists now: from here on the time is the program's
            st["cell_times"].append(t_last - tc)
            ok = proc.write(data, timeout=max(1.0, min(silence, venue.budget_left())))
            if not ok:
                reason = "stalled" if proc.stalled else "eof"
                break
            if t_last - t_cpu >= cpu_poll_s:
                t_cpu = t_last
                st["cpu_s"] = proc.cpu_s()
                if st["cpu_s"] > cpu_cap:
                    reason = "cpu_cap"
                    break
    finally:
        st["cpu_s"] = max(st.get("cpu_s", 0.0), proc.cpu_s())
        proc.terminate(grace=grace_s)
        st["exit_code"] = proc.proc.returncode
    venue.end(reason)
    return venue.end_reason
