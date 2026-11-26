"""The controller as a child process speaking JSON lines (the grader's side of the wire).

    r = Runner(program_dir, run_root, cfg)       # cfg = config.Config
    r.start(wall_s, cpu_s)                       -> (ok, info): the interpreter's start, the import and Controller()
                                                    within wall_s of wall clock and cpu_s of processor time
    r.call(op, payload, cpu_s, wall_s)           -> (status, result, cpu_used_s, wall_s, why)
        status "ok" | "error" (the controller raised) | "cpu" | "timeout" | "dead" | "protocol"
    r.close()                                    -> stats

Processor time is measured by the grader: the interpreter's process CPU clock (every thread it ran, the exited ones
too, in nanoseconds) plus its waited-for children (cutime + cstime of /proc/<pid>/stat), and the same for every other
live process of the controller's process group, listed at every call. A /proc entry is classified once by its start
time. The kernel keeps the waited-for children's total in nanoseconds and /proc gives it in clock ticks: a child's work
is never lost, it is charged to the call in which the total crosses a tick (the sum over the calls is exact to one
tick). The processor time charged to a call is everything the group used since the previous reply (work done between
calls by threads or children counts against the next call), with a high-water mark (rule charge.since_previous_reply).

The process is ended only when its reply stream cannot go on (rules process.end.*): a reply line longer than the line
cap; the stream closed (the process exited); a line that the client did not write (not JSON, or not the client's reply
object: the stream is out of step and a later reply could be read as the answer to an earlier request); a call cut at
its wall cap or at a processor cap reached while waiting. A reply that was read whole is handed on as it is: an
exception the controller raised ("error") and a reply whose numbers are not finite or beyond the float range ("ok",
refused by the reply's form in rules.py) leave the process alive with its state. The controller is stopped only through
the process group recorded at its start (procs.kill_group).

When the runner's own process is root, the controller's process runs under the account named in the runner's
configuration (config.controller_account), with that account's group and no supplementary groups, from a run folder
prepared for it (procs.open_run_dir_for); an account that does not exist is a task error and nothing is started. The
accounting above (/proc and the process clock), the stop through the recorded group and the removal of the run folder
work the same for a process of another account. When the runner is not root, the controller runs under the runner's
own account and none of this is done.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

from . import procs

CLIENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client.py")
CLIENT_NAME = "ff_client.py"
BOOT = ("import sys; sys.path.insert(0, sys.argv[1]); from ff_client import main; sys.exit(main(sys.argv[2:]))")
CLK = os.sysconf("SC_CLK_TCK")
OVERFLOW = "\x00OVERFLOW"


def _int(s):
    """a JSON integer; one too long for the interpreter's integer reading is beyond the float range anyway."""
    try:
        return int(s)
    except ValueError:
        return float("inf")


def read_line(line):
    """the object of one whole reply line, or None when the line is not JSON. The JSON constants NaN and Infinity and
    numbers beyond the float range are read (as non-finite floats or long integers): the reply's form refuses them."""
    try:
        return json.loads(line, parse_int=_int)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None


def _stat_fields(pid):
    with open("/proc/{}/stat".format(pid)) as fh:
        st = fh.read()
    return st[st.rindex(")") + 2:].split()


def _proc_ns(pid):
    """CPU (ns) of process pid: every thread it ran, live or exited, plus its waited-for children. A zombie has no
    process clock: its own time is read from /proc in clock ticks."""
    try:
        f = _stat_fields(pid)
    except (OSError, ValueError, IndexError):
        return None
    children = int((int(f[13]) + int(f[14])) * 1e9 / CLK)
    try:
        own = time.clock_gettime_ns(((~int(pid)) << 3) | 2)       # CPUCLOCK_SCHED of the process
    except OSError:
        own = int((int(f[11]) + int(f[12])) * 1e9 / CLK)
    return own + children


class Runner(object):
    def __init__(self, program_dir, run_root, cfg, python=None, stderr_path=None, sandbox_required=True,
                 extra_read_dirs=(), mem_cap_bytes=None):
        self.program_dir = program_dir
        self.run_root = run_root
        self.cfg = cfg
        self.python = python or sys.executable
        self.stderr_path = stderr_path or (run_root.rstrip("/") + "_stderr.log")
        self.sandbox_required = bool(sandbox_required)
        self.extra_read_dirs = list(extra_read_dirs)
        self.mem_cap = mem_cap_bytes
        self.proc = None
        self.pgid = None
        self.q = None
        self.dead = True
        self.n_calls = 0
        self._cpu_last = None
        self._members = {}           # pid -> start time (ticks) of the live members of the group seen so far
        self._others = {}            # pid -> start time of /proc entries that are not members
        self.stats = {"starts": 0, "deaths": [], "sandbox": None, "stray_processes": 0, "import_s": None,
                      "group_listing_s": 0.0}

    # ------------------------------------------------------------------------------------------------ start
    def sandbox_paths(self, rd):
        """what the controller's process may read (the submission copy, the client, the interpreter's installation,
        the system libraries) and write (its own HOME and TMPDIR)."""
        pre = sorted(set(os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix,
                                                       os.path.dirname(os.path.realpath(self.python)))))
        read_dirs = [rd["program"], rd["lib"]] + pre + ["/usr", "/lib", "/lib64", "/etc/ld.so.cache",
                                                        "/etc/localtime", "/sys/devices/system/cpu"]
        read_dirs += self.extra_read_dirs
        return {"read_dirs": read_dirs, "rw_dirs": [rd["home"], rd["tmp"]],
                "rw_files": ["/dev/null", "/dev/urandom", "/dev/zero"]}

    def start(self, wall_s, cpu_s=None):
        """start the controller's process: the interpreter, the client's sandbox, the import of controller.py and
        Controller(), within wall_s of wall clock and cpu_s of processor time (what is left of the session's set-up
        allowance). info holds the wall and processor time the start used (also when it failed) and, on failure,
        "status": "timeout" or "cpu" for a cut, "failed" for a start that cannot go on, "task_error" for a missing
        sandbox or a missing account."""
        acct = None
        if os.geteuid() == 0:
            acct = procs.account(self.cfg.controller_account)
            if acct is None:
                return False, {"wall_s": 0.0, "cpu_s": 0.0, "status": "task_error", "task_error": True,
                               "error": "the controller's account %r does not exist or is a root account: the "
                                        "controller is not started" % (self.cfg.controller_account,)}
        if os.path.isdir(self.run_root):
            procs.unlock_tree(self.run_root)
            import shutil
            shutil.rmtree(self.run_root, ignore_errors=True)
        rd = procs.prepare_run_dir(self.run_root, self.program_dir, CLIENT, CLIENT_NAME)
        self.rd = rd
        if acct is not None:
            problems = procs.open_run_dir_for(rd, acct[1], acct[2])
            if problems:
                return False, {"wall_s": 0.0, "cpu_s": 0.0, "status": "task_error", "task_error": True,
                               "error": "the run folder cannot be prepared for the controller's account: %s"
                                        % (problems[:3],)}
        env = procs.child_env(rd["home"], rd["tmp"])
        box = self.sandbox_paths(rd)
        argv = [self.python, "-I", "-B", "-c", BOOT, rd["lib"], json.dumps(box)]
        cap = self.mem_cap

        def pre():
            if cap:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (int(cap), int(cap)))
        err = open(self.stderr_path, "ab")
        kw = {}
        if acct is not None:
            os.fchmod(err.fileno(), 0o600)
            kw = {"user": acct[1], "group": acct[2], "extra_groups": []}
        t0 = time.time()
        self.proc = subprocess.Popen(argv, cwd=rd["program"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                     env=env, bufsize=0, start_new_session=True, preexec_fn=pre if cap else None, **kw)
        err.close()
        self.pgid = int(self.proc.pid)          # recorded: the session and the group the controller leads
        self.leaf = int(self.proc.pid)
        self.dead = False
        self.stats["starts"] += 1
        self._members = {}
        self._others = {}
        self._cpu_last = None
        self.q = queue.Queue()
        q, pr, cap_line = self.q, self.proc, self.cfg.line_cap

        def pump():
            # bounded lines: a reply longer than the cap without a newline cannot be read (the stream cannot go on)
            fd = pr.stdout.fileno()
            buf = b""
            try:
                while True:
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if len(line) > cap_line:
                            q.put(OVERFLOW)
                            return
                        q.put(line)
                    if len(buf) > cap_line:
                        q.put(OVERFLOW)
                        return
            except (OSError, ValueError):
                pass
            q.put(None)
        threading.Thread(target=pump, daemon=True).start()

        def used():
            return {"wall_s": time.time() - t0, "cpu_s": self.cpu_ns() * 1e-9}
        deadline = t0 + float(wall_s)
        while True:
            try:
                line = self.q.get(timeout=min(0.05, max(0.0005, deadline - time.time())))
                break
            except queue.Empty:
                if cpu_s is not None and self.cpu_ns() * 1e-9 > float(cpu_s):
                    u = used()
                    self._kill("process.end.cpu_cut", "start")
                    return False, dict(u, status="cpu", error="the start used up the processor allowance")
                if time.time() > deadline:
                    u = used()
                    self._kill("process.end.wall_cut", "start")
                    return False, dict(u, status="timeout", error="the start used up the wall allowance")
        u = used()
        self.stats["import_s"] = round(u["wall_s"], 3)
        if line is None:
            self._kill("process.end.stream_closed", "start")
            return False, dict(u, status="failed", error="the process ended at its start")
        if line == OVERFLOW:
            self._kill("process.end.line_over_cap", "start")
            return False, dict(u, status="failed", error="the ready line is over the cap")
        rep = read_line(line)
        if not isinstance(rep, dict) or "ready" not in rep:
            self._kill("process.end.out_of_step", "start")
            return False, dict(u, status="failed", error="the ready line is not the client's")
        sb = rep.get("sandbox") or {}
        self.stats["sandbox"] = sb
        if self.sandbox_required and not (sb.get("seccomp") and sb.get("landlock") and sb.get("subreaper")):
            self._kill("task_error.sandbox_missing", "start")
            return False, dict(u, status="task_error", error="sandbox not installed", sandbox=sb, task_error=True)
        if not rep.get("ready"):
            self._kill("process.end.not_ready", "start")
            return False, dict(u, status="failed", error=str(rep.get("error")))
        if cpu_s is not None and u["cpu_s"] > float(cpu_s):
            self._kill("process.end.cpu_cut", "start")
            return False, dict(u, status="cpu", error="the start used up the processor allowance")
        return True, dict(u, ready=True, sandbox=sb)

    # ------------------------------------------------------------------------------------------------ cpu
    def _list_group(self):
        """the live members of the controller's process group (at every call; one /proc listing)."""
        t0 = time.perf_counter()
        pids = set()
        for name in os.listdir("/proc"):
            if name.isdigit():
                pids.add(int(name))
        for p in list(self._members):
            if p not in pids:
                del self._members[p]
        for p in list(self._others):
            if p not in pids:
                del self._others[p]
        for p in pids:
            if p in self._members or p in self._others or p == self.leaf:
                continue
            try:
                f = _stat_fields(p)
            except (OSError, ValueError, IndexError):
                continue
            if int(f[2]) == self.pgid:
                self._members[p] = f[19]
            else:
                self._others[p] = f[19]
        self.stats["group_listing_s"] += time.perf_counter() - t0
        return list(self._members)

    def cpu_ns(self):
        """CPU (ns) the controller has used so far: the interpreter, then the other members of its group (a child
        reaped in between is missed once rather than counted twice)."""
        tot = _proc_ns(self.leaf) or 0
        for p in self._list_group():
            v = _proc_ns(p)
            if v:
                tot += v
        return tot

    # ------------------------------------------------------------------------------------------------ call
    def call(self, op, payload, cpu_s, wall_s):
        """cpu_s None: no processor cap on the call (a step: the late-reply rule reads the processor time used)."""
        if self.dead or self.proc is None:
            return "dead", None, 0.0, 0.0, "not running"
        msg = {"op": op}
        if payload:
            msg.update(payload)
        c0 = self._cpu_last if self._cpu_last is not None else self.cpu_ns()
        try:
            data = (json.dumps(msg, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        except (TypeError, ValueError) as e:
            raise RuntimeError("the grader's own message is not JSON: {}".format(e))
        t0 = time.time()
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except (OSError, ValueError):
            self._kill("process.end.stream_closed", op)
            return "dead", None, 0.0, 0.0, "the request stream is closed"
        deadline = t0 + float(wall_s)
        while True:
            try:
                line = self.q.get(timeout=min(0.05, max(0.0005, deadline - time.time())))
                break
            except queue.Empty:
                if cpu_s is not None and (self.cpu_ns() - c0) * 1e-9 > float(cpu_s):
                    used = (self.cpu_ns() - c0) * 1e-9
                    self._kill("process.end.cpu_cut", op)
                    return "cpu", None, used, time.time() - t0, "processor cap reached while waiting"
                if time.time() > deadline:
                    used = (self.cpu_ns() - c0) * 1e-9
                    self._kill("process.end.wall_cut", op)
                    return "timeout", None, used, time.time() - t0, "wall cap"
        wall = time.time() - t0
        self.n_calls += 1
        c1 = max(self.cpu_ns(), c0)                # high-water mark: a process missed in one reading is charged later
        cpu_used = (c1 - c0) * 1e-9
        self._cpu_last = c1
        if line is None:
            self._kill("process.end.stream_closed", op)
            return "dead", None, cpu_used, wall, "end of stream"
        if line == OVERFLOW:
            self._kill("process.end.line_over_cap", op)
            return "protocol", None, cpu_used, wall, "reply line over the cap"
        rep = read_line(line)
        if not isinstance(rep, dict) or "ok" not in rep:
            # a line the client did not write: the stream is out of step
            self._kill("process.end.out_of_step", op)
            return "protocol", None, cpu_used, wall, "not JSON" if rep is None else "not a reply object"
        if not rep.get("ok"):
            return "error", rep.get("error"), cpu_used, wall, "the controller raised"
        if cpu_s is not None and cpu_used > float(cpu_s):
            return "cpu", None, cpu_used, wall, "processor cap (the reply came in over it)"
        return "ok", rep.get("result"), cpu_used, wall, ""

    # ------------------------------------------------------------------------------------------------ stop
    def _kill(self, rule, call):
        """end the controller's process group; rule = the key of the rule that ends it (process.end.*, task_error.*)."""
        self.stats["deaths"].append("{} ({})".format(rule, call))
        if self.proc is not None:
            procs.kill_group(self.pgid)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.dead = True

    def close(self):
        """end the controller; returns the stats (stray_processes: members of its group still alive after the end)."""
        if self.proc is not None and not self.dead:
            try:
                self.proc.stdin.write(b'{"op":"end"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        left = []
        if self.pgid is not None:
            for name in os.listdir("/proc"):
                if not name.isdigit():
                    continue
                try:
                    f = _stat_fields(name)
                except (OSError, ValueError, IndexError):
                    continue
                if int(f[2]) == self.pgid and f[0] != "Z":
                    left.append(int(name))
        self.stats["stray_processes"] = max(self.stats["stray_processes"], len(left))
        if left or not self.dead:
            self._kill("process.end.session_end", "close")
        return dict(self.stats)
