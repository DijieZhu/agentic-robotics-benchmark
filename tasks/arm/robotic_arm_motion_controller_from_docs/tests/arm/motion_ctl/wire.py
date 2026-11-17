"""The controller as a deprivileged child process (grader side), on the runner helpers of `arm.assembly_cell.runner`
(prepare_run_dir, child_env, wrap_limits, runner_account, kill_uid_processes):

    r = Runner(program_dir, run_root, client_path)   # client_path = motion_ctl/client.py (copied alone as mc_client.py:
                                                     # the child never sees the grader's code)
    r.start()                                        -> (ok, info): import + Controller(workdir) within IMPORT_S
    r.call(op, payload, cpu_s, wall_s)               -> (status, reply, cpu_used_s, wall_s)
                                                        status "ok" | "error" (the controller raised) | "cpu" | "timeout"
                                                        | "dead" | "protocol"
    r.close()                                        -> {"runner_processes_killed", "stray_processes"}

CPU is measured by the grader: each process's CPU clock (clock_getcpuclockid: every thread it ran, exited ones too, in
nanoseconds) plus its waited-for children (/proc cutime + cstime), for the controller's interpreter and every other
process of the runner account, listed at every call as root (when not root: the child's session, every REFRESH calls);
the client makes the interpreter the subreaper of what it starts, so no descendant leaves the account's view before it
is waited for. A call over its CPU cap or its wall cap kills the child (the next call fails "dead"). A reply line
longer than MAX_LINE is a protocol error. Agent code never runs as root: with
no unprivileged account or no runuser the run is refused (fail closed).
"""

import json
import os
import pwd
import queue
import shutil
import subprocess
import threading
import time

from arm.assembly_cell import runner as RUN

BOOT = ("import sys; sys.path.insert(0, sys.argv[1]); from mc_client import main; "
        "sys.exit(main(sys.argv[2:]))")
IMPORT_S = 60.0
NPROC_MAX = 256
REFRESH = 200
MAX_LINE = 1 << 20              # bytes of one reply line
CLK = os.sysconf("SC_CLK_TCK")


def _runuser():
    for p in (shutil.which("runuser"), "/usr/sbin/runuser", "/sbin/runuser", "/usr/bin/runuser", "/bin/runuser"):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _children_ns(pid):
    """CPU (ns) of pid's children that have exited and been waited for (cutime + cstime of /proc/pid/stat)."""
    try:
        with open("/proc/{}/stat".format(pid)) as fh:
            st = fh.read()
        f = st[st.rindex(")") + 2:].split()
        return int((int(f[13]) + int(f[14])) * 1e9 / CLK)
    except (OSError, ValueError, IndexError):
        return 0


def _proc_ns(pid):
    """CPU (ns) of process pid: every thread it ran, live or exited (the process CPU clock of clock_getcpuclockid), plus
    its waited-for children. (Summing the live threads' schedstat would lose the threads that have exited.)"""
    try:
        own = time.clock_gettime_ns(((~int(pid)) << 3) | 2)       # CPUCLOCK_SCHED of the process
    except OSError:
        return 0
    return own + _children_ns(pid)


def _session_pids(sid):
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/{}/stat".format(name)) as fh:
                st = fh.read()
            f = st[st.rindex(")") + 2:].split()
            if int(f[3]) == int(sid):
                out.append(int(name))
        except (OSError, ValueError, IndexError):
            continue
    return out


class Runner(object):
    def __init__(self, program_dir, run_root, client_path, user="nobody", python="python3", cpu_cap_s=7200,
                 stderr_path=None, extra_env=None):
        self.program_dir = program_dir
        self.run_root = run_root
        self.client_path = client_path
        self.python = python
        self.cpu_cap = int(cpu_cap_s)
        self.stderr_path = stderr_path or (run_root.rstrip("/") + "_stderr.log")
        self.extra_env = extra_env or {}
        acct = RUN.runner_account(user) if os.getuid() == 0 else None
        self.user, self.uid, self.gid = (acct if acct else (None, None, None))
        self.proc = None
        self.leaf = None
        self.q = None
        self.dead = False
        self.n_calls = 0
        self._others = []
        self._cpu_last = None
        self.stats = {"runner_account": self.user or pwd.getpwuid(os.getuid()).pw_name,
                      "runner_uid": self.uid if self.uid is not None else os.getuid(), "runner_processes_killed": 0,
                      "stray_processes": 0, "import_s": None, "deaths": []}

    # ------------------------------------------------------------------------------------------------------ start
    def start(self):
        if os.path.isdir(self.run_root):
            shutil.rmtree(self.run_root, ignore_errors=True)
        owner = (self.uid, self.gid) if self.uid is not None else None
        rd = RUN.prepare_run_dir(self.run_root, self.program_dir, {}, owner=owner)
        shutil.copyfile(self.client_path, os.path.join(rd["lib"], "mc_client.py"))
        os.chmod(os.path.join(rd["lib"], "mc_client.py"), 0o644)
        self.rd = rd
        env = RUN.child_env(rd["home"], rd["tmp"], rd["lib"], "", threads=1, extra=self.extra_env)
        env.pop("ARB_GIVEN_DIR", None)
        env.pop("PYTHONPATH", None)
        argv = [self.python, "-B", "-E", "-s", "-c", BOOT, rd["lib"], rd["home"]]
        cmd = RUN.wrap_limits(argv, cpu_s=self.cpu_cap, user=None)
        if cmd and os.path.basename(cmd[0]) == "prlimit" and "--" in cmd:
            cmd.insert(cmd.index("--"), "--nproc={}".format(NPROC_MAX))
        if os.getuid() == 0:
            if not self.user:
                raise RuntimeError("no unprivileged runner account: refusing to run the controller as root")
            ru = _runuser()
            if not ru:
                raise RuntimeError("runuser not found: refusing to run the controller as root")
            cmd = [ru, "-p", "-u", self.user, "--"] + cmd
            cmd = ["/bin/sh", "-c", 'echo 0 > /proc/self/oom_score_adj 2>/dev/null; exec "$@"', "sh"] + cmd
        err = open(self.stderr_path, "ab")
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, cwd=rd["program"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                     env=env, text=True, bufsize=1, start_new_session=True)
        err.close()
        self.q = queue.Queue()
        q, pr = self.q, self.proc

        def pump():
            # bounded lines: a reply longer than MAX_LINE without a newline is a protocol error (the child is killed)
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
                        q.put(line.decode("utf-8", "replace"))
                    if len(buf) > MAX_LINE:
                        q.put("\x00OVERFLOW")
                        buf = b""
                        break
            except (OSError, ValueError):
                pass
            q.put(None)
        threading.Thread(target=pump, daemon=True).start()
        try:
            line = self.q.get(timeout=IMPORT_S)
        except queue.Empty:
            self._kill("import timeout")
            return False, {"error": "import timeout"}
        self.stats["import_s"] = round(time.time() - t0, 3)
        if line is None:
            self._kill("died at import")
            return False, {"error": "died at import"}
        try:
            rep = json.loads(line)
        except ValueError:
            rep = None
        if not isinstance(rep, dict):
            self._kill("protocol at import")
            return False, {"error": "protocol"}
        self._find_leaf()
        self.stats["no_network"] = bool(rep.get("no_network"))     # the client's seccomp filter (no inet sockets)
        if os.getuid() == 0:
            self._assert_deprivileged()
        if not rep.get("ready"):
            self._kill("import failed")
            return False, rep
        return True, rep

    def _find_leaf(self):
        best = None
        for pid in self._pids_full():
            try:
                with open("/proc/{}/cmdline".format(pid), "rb") as fh:
                    cmd = fh.read()
            except OSError:
                continue
            if b"mc_client" in cmd and os.path.basename(cmd.split(b"\0")[0]).startswith(b"python"):
                best = pid
        self.leaf = best

    def _assert_deprivileged(self):
        if self.leaf is None:
            return
        with open("/proc/{}/status".format(self.leaf)) as fh:
            st = fh.read()
        uid = int([l for l in st.splitlines() if l.startswith("Uid:")][0].split()[1])
        if uid == 0 or (self.uid is not None and uid != self.uid):
            self._kill("uid")
            raise RuntimeError("the controller process runs as uid {}: refused".format(uid))

    # -------------------------------------------------------------------------------------------------------- cpu
    def _pids_full(self):
        if self.uid is not None:
            return RUN.uid_processes(self.uid)
        return _session_pids(self.proc.pid) if self.proc is not None else []

    def cpu_ns(self):
        """CPU (ns) the controller has used so far (the interpreter first, then the other processes: a child reaped in
        between is missed once rather than counted twice)."""
        if self.uid is not None or self.n_calls % REFRESH == 0:
            self._others = [p for p in self._pids_full() if p != self.leaf]
        tot = _proc_ns(self.leaf) if self.leaf else 0
        for p in self._others:
            tot += _proc_ns(p)
        return tot

    # ------------------------------------------------------------------------------------------------------- call
    def call(self, op, payload=None, cpu_s=1.0, wall_s=10.0):
        if self.dead or self.proc is None:
            return "dead", None, 0.0, 0.0
        msg = {"op": op}
        if payload:
            msg.update(payload)
        # the CPU charged to a call is everything the controller used since the previous reply (work done between calls,
        # by threads or children, counts against the next call)
        c0 = self._cpu_last if self._cpu_last else self.cpu_ns()
        t0 = time.time()
        try:
            self.proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            self._kill("stdin closed")
            return "dead", None, 0.0, 0.0
        status, rep = "ok", None
        deadline = t0 + float(wall_s)
        while True:
            try:
                line = self.q.get(timeout=min(0.25, max(0.001, deadline - time.time())))
                break
            except queue.Empty:
                if (self.cpu_ns() - c0) * 1e-9 > float(cpu_s):
                    self._kill("{} cpu".format(op))
                    return "cpu", None, (self.cpu_ns() - c0) * 1e-9, time.time() - t0
                if time.time() > deadline:
                    self._kill("{} timeout".format(op))
                    return "timeout", None, (self.cpu_ns() - c0) * 1e-9, time.time() - t0
        self.n_calls += 1
        c1 = max(self.cpu_ns(), c0)                # high-water mark: a process missed in one reading is charged later
        cpu_used = (c1 - c0) * 1e-9
        self._cpu_last = c1
        wall = time.time() - t0
        if line is None:
            self._kill("eof")
            return "dead", None, cpu_used, wall
        if line == "\x00OVERFLOW":
            self._kill("reply too long")
            return "protocol", None, cpu_used, wall
        try:
            rep = json.loads(line)
        except ValueError:
            rep = None
        if not isinstance(rep, dict):              # not JSON, or JSON that is not an object (a stray line on fd 1)
            self._kill("protocol")
            return "protocol", None, cpu_used, wall
        if not rep.get("ok"):
            status = "error"
        if cpu_used > float(cpu_s):
            status = "cpu"
        return status, rep, cpu_used, wall

    # ------------------------------------------------------------------------------------------------------- stop
    def _kill(self, why):
        self.stats["deaths"].append(why)
        if self.proc is not None:
            try:
                os.killpg(self.proc.pid, 9)
            except (OSError, ProcessLookupError):
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self.uid is not None:
            k, rem = RUN.kill_uid_processes(self.uid)
            self.stats["runner_processes_killed"] += k
        self.dead = True

    def close(self):
        if self.proc is not None and not self.dead:
            try:
                self.proc.stdin.write('{"op":"end"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        left = [p for p in self._pids_full()] if self.proc is not None else []
        self.stats["stray_processes"] = len(left)
        if left or not self.dead:
            self._kill("close")
        return dict(self.stats)
