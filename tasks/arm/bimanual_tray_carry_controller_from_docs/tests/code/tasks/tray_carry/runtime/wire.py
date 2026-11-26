
import json
import os
import pwd
import queue
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import procutil as PU

CLIENT_PATH = os.path.join(HERE, "client.py")
CLIENT_NAME = "tc_client"
BOOT = ("import sys; sys.path.insert(0, sys.argv[1]); from tc_client import main; "
        "sys.exit(main(sys.argv[2:]))")
IMPORT_S = 60.0
NPROC_MAX = 256
REFRESH = 250
MAX_LINE = 1 << 20
MUJOCO_GL = "disable"
CHILD_THREADS = 1
CHILD_NUMERICS = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256"}
PY_FLAGS = ("-B", "-E", "-s")
CLK = os.sysconf("SC_CLK_TCK")


def _runuser():
    for p in (shutil.which("runuser"), "/usr/sbin/runuser", "/sbin/runuser", "/usr/bin/runuser", "/bin/runuser"):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _stat(pid):
    try:
        with open("/proc/{}/stat".format(pid)) as fh:
            st = fh.read()
        f = st[st.rindex(")") + 2:].split()
        return int((int(f[13]) + int(f[14])) * 1e9 / CLK), int(f[3])
    except (OSError, ValueError, IndexError):
        return 0, None


def _clock_ns(pid):
    try:
        return time.clock_gettime_ns(((~int(pid)) << 3) | 2)
    except OSError:
        return 0


def _zombie_ns(pid):
    try:
        with open("/proc/{}/stat".format(pid)) as fh:
            st = fh.read()
        f = st[st.rindex(")") + 2:].split()
        return int((int(f[11]) + int(f[12])) * 1e9 / CLK)
    except (OSError, ValueError, IndexError):
        return 0


def _proc_ns(pid):
    own = _clock_ns(pid)
    if own == 0:
        own = _zombie_ns(pid)
    return own + _stat(pid)[0]


def _starttime(pid):
    try:
        with open("/proc/{}/stat".format(pid)) as fh:
            st = fh.read()
        return int(st[st.rindex(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _state(pid):
    try:
        with open("/proc/{}/stat".format(pid)) as fh:
            st = fh.read()
        return st[st.rindex(")") + 2:].split()[0]
    except (OSError, ValueError, IndexError):
        return None


def _ns_last_pid():
    try:
        with open("/proc/sys/kernel/ns_last_pid") as fh:
            return int(fh.read())
    except (OSError, ValueError):
        return None


class Runner(object):
    def __init__(self, program_dir, run_root, user="nobody", python=None, cpu_cap_s=7200, stderr_path=None,
                 extra_env=None, refresh=REFRESH, trigger="auto"):
        self.program_dir = program_dir
        self.run_root = run_root
        self.python = python or sys.executable
        self.cpu_cap = int(cpu_cap_s)
        self.stderr_path = stderr_path or (run_root.rstrip("/") + "_stderr.log")
        self.extra_env = extra_env or {}
        self.refresh = int(refresh)
        if trigger not in ("auto", "cutime", "ns_last_pid"):
            raise ValueError("trigger: 'auto', 'cutime' or 'ns_last_pid'")
        self.trigger = ("ns_last_pid" if os.getuid() == 0 else "cutime") if trigger == "auto" else trigger
        self._last_pid = None
        acct = PU.runner_account(user) if os.getuid() == 0 else None
        self.user, self.uid, self.gid = (acct if acct else (None, None, None))
        self.proc = None
        self.pgid = None
        self.leaf = None
        self.sid = None
        self.q = None
        self.dead = False
        self.n_calls = 0
        self._others = []
        self._seen = {}
        self._cpu_last = None
        self._leaf_children_last = 0
        self.stats = {"runner_account": self.user or pwd.getpwuid(os.getuid()).pw_name,
                      "runner_uid": self.uid if self.uid is not None else os.getuid(), "runner_processes_killed": 0,
                      "stray_processes": 0, "import_s": None, "deaths": [], "listings": 0, "listing_s": 0.0,
                      "descendants_seen": 0}

    def start(self):
        if os.path.isdir(self.run_root):
            shutil.rmtree(self.run_root, ignore_errors=True)
        owner = (self.uid, self.gid) if self.uid is not None else None
        rd = PU.prepare_run_dir(self.run_root, self.program_dir, {}, owner=owner)
        shutil.copyfile(CLIENT_PATH, os.path.join(rd["lib"], CLIENT_NAME + ".py"))
        os.chmod(os.path.join(rd["lib"], CLIENT_NAME + ".py"), 0o644)
        self.rd = rd
        env = PU.child_env(rd["home"], rd["tmp"], rd["lib"], "", threads=CHILD_THREADS, mujoco_gl=MUJOCO_GL,
                           extra=self.extra_env)
        env.pop("ARB_GIVEN_DIR", None)
        env.pop("PYTHONPATH", None)
        env["MUJOCO_GL"] = MUJOCO_GL
        env.pop("PYOPENGL_PLATFORM", None)
        env.update(CHILD_NUMERICS)
        argv = [self.python] + list(PY_FLAGS) + ["-c", BOOT, rd["lib"], rd["home"]]
        cmd = PU.wrap_limits(argv, cpu_s=self.cpu_cap, user=None)
        if self.uid is not None and cmd and os.path.basename(cmd[0]) == "prlimit" and "--" in cmd:
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
        self.pgid = int(self.proc.pid)
        self.sid = self.pgid
        self.q = queue.Queue()
        q, pr = self.q, self.proc

        def pump():
            fd = pr.stdout.fileno()
            buf = b""
            try:
                while True:
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    buf += chunk
                    over = False
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if len(line) > MAX_LINE:
                            over = True
                            break
                        q.put(line.decode("utf-8", "replace"))
                    if over or len(buf) > MAX_LINE:
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
        self.stats["no_network"] = bool(rep.get("no_network"))
        self._fail_closed_on_no_network()
        if os.getuid() == 0:
            self._assert_deprivileged()
        if not rep.get("ready"):
            self._kill("import failed")
            return False, rep
        self._leaf_children_last = _stat(self.leaf)[0] if self.leaf else 0
        return True, rep

    def _listing(self):
        t0 = time.perf_counter()
        if self.uid is not None:
            pids = PU.uid_processes(self.uid)
            if self.leaf is not None:
                pids = [p for p in pids if p != self.leaf]
        elif self.proc is not None:
            pids = PU.descendants(self.leaf or self.proc.pid, self.sid)
            if self.leaf is not None:
                pids = [p for p in pids if p != self.leaf]
        else:
            pids = []
        for p in pids:
            if p not in self._seen:
                st = _starttime(p)
                if st is not None:
                    self._seen[p] = st
        self.stats["listings"] += 1
        self.stats["listing_s"] += time.perf_counter() - t0
        return pids

    def _survivors(self):
        out = []
        for p, st in sorted(self._seen.items()):
            if p == self.leaf or (self.proc is not None and p == self.proc.pid):
                continue
            if _starttime(p) == st and _state(p) not in (None, "Z", "X"):
                out.append(p)
        return out

    def _find_leaf(self):
        best = None
        cands = [self.proc.pid] + (PU.uid_processes(self.uid) if self.uid is not None
                                   else PU.descendants(self.proc.pid, self.sid))
        for pid in cands:
            try:
                with open("/proc/{}/cmdline".format(pid), "rb") as fh:
                    cmd = fh.read()
            except OSError:
                continue
            if CLIENT_NAME.encode() in cmd and os.path.basename(cmd.split(b"\0")[0]).startswith(b"python"):
                best = pid
        self.leaf = best

    def _fail_closed_on_no_network(self):
        if os.getuid() == 0 and not self.stats.get("no_network"):
            self._kill("no network filter")
            raise RuntimeError("the controller's no-network filter is not installed: refusing to grade")

    def _assert_deprivileged(self):
        if self.leaf is None:
            return
        with open("/proc/{}/status".format(self.leaf)) as fh:
            st = fh.read()
        uid = int([ln for ln in st.splitlines() if ln.startswith("Uid:")][0].split()[1])
        if uid == 0 or (self.uid is not None and uid != self.uid):
            self._kill("uid")
            raise RuntimeError("the controller process runs as uid {}: refused".format(uid))

    def cpu_ns(self, force_listing=False):
        leaf_children = _stat(self.leaf)[0] if self.leaf else 0
        moved = False
        if self.trigger == "ns_last_pid":
            lp = _ns_last_pid()
            moved = lp is None or lp != self._last_pid
            self._last_pid = lp
        if (force_listing or moved or self.n_calls % self.refresh == 0
                or leaf_children > self._leaf_children_last):
            self._others = self._listing()
            self.stats["descendants_seen"] = max(self.stats["descendants_seen"], len(self._others))
        self._leaf_children_last = leaf_children
        tot = (_clock_ns(self.leaf) + leaf_children) if self.leaf else 0
        for p in self._others:
            tot += _proc_ns(p)
        return tot

    def call(self, op, payload=None, cpu_s=1.0, wall_s=10.0):
        if self.dead or self.proc is None:
            return "dead", None, 0.0, 0.0
        msg = {"op": op}
        if payload:
            msg.update(payload)
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
                if (self.cpu_ns(force_listing=True) - c0) * 1e-9 > float(cpu_s):
                    used = (self.cpu_ns() - c0) * 1e-9
                    self._kill("{} cpu".format(op))
                    return "cpu", None, used, time.time() - t0
                if time.time() > deadline:
                    used = (self.cpu_ns() - c0) * 1e-9
                    self._kill("{} timeout".format(op))
                    return "timeout", None, used, time.time() - t0
        self.n_calls += 1
        c1 = max(self.cpu_ns(), c0)
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
        except (ValueError, RecursionError):
            rep = None
        if not isinstance(rep, dict):
            self._kill("protocol")
            return "protocol", None, cpu_used, wall
        if not rep.get("ok"):
            status = "error"
        if cpu_used > float(cpu_s):
            status = "cpu"
        return status, rep, cpu_used, wall

    def _kill(self, why):
        self.stats["deaths"].append(why)
        P = self.pgid
        if self.proc is not None and isinstance(P, int) and P > 1000:
            try:
                os.killpg(P, 9)
            except (OSError, ProcessLookupError):
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self.uid is not None:
            k, rem = PU.kill_uid_processes(self.uid)
            self.stats["runner_processes_killed"] += k
        self.dead = True

    def close(self):
        if self.proc is not None and not self.dead:
            self._listing()
            try:
                self.proc.stdin.write('{"op":"end"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        if self.proc is not None:
            self._listing()
        left = self._survivors()
        if self.proc is not None and self.proc.poll() is None:
            left = left + [self.proc.pid]
        self.stats["stray_processes"] = len(left)
        self.stats["stray_pids"] = list(left)
        self.stats["processes_seen"] = len(self._seen)
        if left or not self.dead:
            self._kill("close")
        self.stats["calls"] = self.n_calls
        self.stats["trigger"] = self.trigger
        self.stats["refresh"] = self.refresh
        return dict(self.stats)
