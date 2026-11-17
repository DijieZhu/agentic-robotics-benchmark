#!/usr/bin/env python3
"""Interface self-test of a DA-6 motion controller: the protocol and the timing, nothing else.

    python3 check_controller.py PATH/TO/submission/controller/controller.py [options]

It runs the controller the way the cell computer does (controller interface specification, sections 2 to 8) and
calls it with payloads in the formats of that specification, but against a TOY plant: six toy servos (a first-order
lag towards the goal, a fixed command latency, a speed limit) of which servos 1 and 2 carry a toy two-link pendulum,
so that their load registers move. The toy plant is not the DA-6: it has none of the arm's kinematics, dynamics or
geometry, and the toy jobs are not chosen for the arm. The self-test tells whether the controller loads, answers every
call in the right format and keeps its budgets; it tells nothing about how the DA-6 would move.

Checks
  static    the .py files of the controller folder, with the rules of the acceptance's network audit (imports of
            networking, ctypes / cffi and process modules, calls that start a program, imports under a computed name,
            exec / eval / compile of computed code); packages this python does not have, the folder's size, the
            Controller class and its methods, file paths outside the folder (the documents are not on the controller
            computer)
  start     the folder copied to a scratch directory (regular files only, read-only, no __pycache__), started with
            `python3 -B -E -s` in that copy, an empty workdir as HOME, a fresh TMPDIR, BLAS threads 1, stdout to the
            log; import + Controller(workdir) within the budget
  calls     setup(unit, cell, tool), plan(job, state) (plan(job) when plan takes the job alone), step(state); each
            call is charged the CPU time the controller used since its previous reply, as at acceptance: its threads
            (also those that ended before the reply), the child processes it has waited for and (Linux) its running
            child processes, the time between the calls included; wall-clock time against the budgets; exceptions;
            replies that cannot be encoded
  cycles    the 10 ms cycle as at acceptance: a step over 10 ms of CPU is executed floor(cpu / 10 ms) cycles late
            (no state for the cycles in between), a step over the hard limits ends the process (restart, setup again);
            the fraction of bus cycles answered within 10 ms of CPU
  protocol  every step reply read as the cell computer reads it: an object or None, `goal`, `speed` and `acc` each
            null or a list of six finite numbers (no booleans; goal entries within +-8192, speed and acc
            entries within +-1e6), goals rounded to whole ticks; a malformed reply aborts the job, as at
            acceptance. Warnings for what is accepted but probably not meant: goals outside the EEPROM limits, speed
            or acc entries that are negative or below 1 (read as 0), path_start / done that are not booleans, unknown
            keys, path_start in a reach job. The answer of plan, path_start and done of path jobs, done within the
            job's cap.

Options
  --unit FILE       the unit payload for setup: {"serial", "calibration"} or a bare calibration record (default:
                    ../unit/example_calibration_record.json next to this file if it exists, else a toy record)
  --scenario FILE   cells, tools and jobs to run instead of the built-in toy scenario (the format of --dump-scenario)
  --dump-scenario   print the built-in toy scenario as JSON and exit
  --start TICKS     where the toy servos start in each cell: zero (the record's zero_ticks, the default) or six
                    comma-separated tick values
  --json FILE       write the full report as JSON
  --log FILE        copy the controller's log (its stderr and print output) to FILE
  --quiet           print the summary only

Exit status: 0 no errors, 1 errors, 2 the controller could not be started. Standard library and numpy only. Run it
with the python3 of the image (the acceptance uses that one; packages of other interpreters or of your user
site-packages are not there). Not covered: anything about the DA-6 (accuracy, clearance, contact, feasibility), the
unprivileged account and the network isolation of the acceptance runs (there creating a network socket fails; only
local Unix-domain sockets can be created), and (on systems without Linux's /proc) the CPU time of child processes that
are still running (the acceptance counts it).
"""

import argparse
import ast
import inspect
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

# ------------------------------------------------------------------- numbers of the controller interface specification
JOINTS = ['base_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch', 'wrist_roll', 'jaw']
TICKS_PER_TURN = 4096
CYCLE_S = 0.01
STEP_CPU_S = 0.01
STEP_KILL_CPU_S = 1.0
STEP_KILL_WALL_S = 5.0
INIT_WALL_S = 60.0
SETUP_CPU_S = 60.0
SETUP_WALL_S = 120.0
PLAN_CPU_S = 20.0
PLAN_WALL_S = 60.0
ON_TIME_MIN = 0.9
SETTLE_S = 0.5
SIZE_CAP_MB = 80
NETWORK_MODULES = ['socket', '_socket', 'ssl', '_ssl', 'urllib', 'urllib2', 'urllib3', 'http', 'httplib', 'requests',
                   'httpx', 'aiohttp', 'ftplib', 'smtplib', 'imaplib', 'poplib', 'nntplib', 'telnetlib', 'websocket',
                   'websockets', 'paramiko', 'pycurl', 'xmlrpc', 'asyncio', 'socketserver', 'grpc', 'zmq',
                   'huggingface_hub', 'boto3', 'botocore', 'wget', 'ctypes', '_ctypes', 'cffi', 'selectors',
                   'multiprocessing.connection', 'multiprocessing.managers', 'logging.handlers', 'subprocess', 'pty',
                   '_posixsubprocess']
PROCESS_CALLS = ['system', 'popen', 'execv', 'execve', 'execvp', 'execvpe', 'execl', 'execle', 'execlp', 'execlpe',
                 'spawnv', 'spawnve', 'spawnvp', 'spawnvpe', 'spawnl', 'spawnle', 'spawnlp', 'spawnlpe',
                 'posix_spawn', 'posix_spawnp', 'fork_exec']
DYNAMIC_IMPORT_CALLS = ['__import__', 'import_module', 'find_spec', 'spec_from_file_location', 'load_module']
CODE_CALLS = ['exec', 'eval', 'compile']
RECORD_FORMAT = 'calibration-record/1'
RECORD_CONVENTION = 'q = direction x (ticks - zero_ticks) x 2 pi / 4096 (q = design angle of the kinematics sheet)'
RECORD_UNCERTAINTY = {'arm joints': 0.5, 'jaw': 1.0}
TOY_TICK_RANGE = [1024, 3072]
SCENE_FORMAT = 'scene-description/1'
SCENE_FRAME = "cell (the table's reference corner): x, y >= 0 on the table, z up; metres, degrees"
SCENE_TABLE = {'z': 0.0, 'note': 'the robot base stands on the table; the table top is z = 0'}
SURVEY_TOLERANCE = {'objects_mm': 2.0, 'objects_deg': 1.0, 'robot_base_mm': 1.0, 'robot_base_deg': 0.3}
CLEARANCE_MM = 10.0
APPROACH_TOL_DEG = 5.0
JAW_TOL_DEG = 5.0
AXIS_TOL_DEG = 8.0
GAP_MM = 3.0
GAP_TOL_MM = 1.5
MAX_REPLY_LINE = 1048576
GOAL_BOUND = 8192
REG_BOUND = 1000000.0
TOY_PART = {'kind': 'part',
            'half_mm': [12.0, 12.0, 15.0],
            'seat_mm': {'x': 4.16, 'z': 6.0},
            'mass_range_kg': [0.03, 0.15],
            'com_note': 'the centre of mass is inside the part; not known'}
NEEDLE_TOOL = {'kind': 'needle',
               'length_mm': 45.0,
               'radius_mm': 1.2,
               'mass_kg': 0.02,
               'axis_offset_x_mm': -0.001,
               'holder_half_mm': [7.84, 6.0, 9.0],
               'holder_centre_z_mm': -3.0}

HERE = os.path.dirname(os.path.abspath(__file__))
OVERFLOW = "\x00OVERFLOW"
CMD_KEYS = ("goal", "speed", "acc", "path_start", "done")
PLAN_STATE_KEYS = ("position", "speed", "load", "moving")
TICK_RAD = 2.0 * math.pi / TICKS_PER_TURN

# the toy plant (toy values; nothing here describes the DA-6)
TOY = {"latency_s": 0.030, "tau_s": 0.060, "vmax_ticks_s": 3000.0, "moving_ticks_s": 5.0, "substeps": 5,
       "load_refresh_s": 0.020, "stall_nm": 3.0, "g": 9.81, "m1_kg": 0.40, "l1_m": 0.20, "m2_kg": 0.25, "l2_m": 0.15}


# ===================================================================================== child side (standard library)
try:
    import resource
except ImportError:                     # not on every platform: the CPU of finished child processes is then left out
    resource = None


def _cpu_now():
    """CPU time of this process (all its threads, also finished ones) plus that of the child processes it has
    waited for."""
    c = time.process_time()
    if resource is not None:
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        c += r.ru_utime + r.ru_stime
    return c


def _trace():
    import traceback
    return traceback.format_exc().strip().splitlines()[-8:]


def takes_state(fn):
    """plan(job, state) or the older plan(job): two positional parameters (besides self) or *args, as the cell
    computer decides it."""
    try:
        ps = [p for p in inspect.signature(fn).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)]
    except (TypeError, ValueError):
        return False
    return any(p.kind == p.VAR_POSITIONAL for p in ps) or len(ps) >= 2


def _subreaper():
    """Orphans of the controller's processes become this process's children (Linux: PR_SET_CHILD_SUBREAPER), as on the
    controller computer, so they stay in the CPU accounting until they have been waited for."""
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass


def _child(program_dir, workdir):
    """The controller's process: import controller.py from the working directory, one JSON line per request. Each
    reply carries the CPU time this process used since its previous reply (the first request: since its arrival)."""
    import traceback
    _subreaper()
    out = sys.stdout
    sys.stdout = sys.stderr
    here = os.path.normcase(HERE)
    sys.path[:] = [program_dir] + [p for p in sys.path if p and os.path.normcase(os.path.abspath(p)) != here]
    os.chdir(program_dir)
    mark = [_cpu_now()]

    def say(rep):
        try:
            body = json.dumps(rep, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            body = json.dumps({"ok": False, "error": "reply: {}".format(e), "encode": True}, separators=(",", ":"))
        now = _cpu_now()
        tail = ',"cpu_s":{:.9f}}}'.format(now - mark[0])
        mark[0] = now
        out.write(body[:-1] + tail + "\n")
        out.flush()

    ctl = None
    try:
        import controller  # noqa: the submission's module
        ctl = controller.Controller(workdir)
        rep = {"ok": True, "ready": True}
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        rep = {"ok": False, "ready": False, "error": "import: {}: {}".format(type(e).__name__, e), "trace": _trace()}
    say(rep)
    if ctl is None:
        return 1
    first = True
    for line in sys.stdin:
        if first:
            mark[0] = _cpu_now()
            first = False
        try:
            req = json.loads(line)
        except ValueError:
            say({"ok": False, "error": "bad request"})
            continue
        op = req.get("op")
        try:
            if op == "setup":
                ctl.setup(req["unit"], req["cell"], req["tool"])
                rep = {"ok": True}
            elif op == "plan":
                if takes_state(ctl.plan):
                    r = ctl.plan(req["job"], req.get("state"))
                else:
                    r = ctl.plan(req["job"])
                rep = {"ok": True, "result": "infeasible" if r == "infeasible" else "ok"}
                if not (isinstance(r, str) and r in ("ok", "infeasible")):
                    rep["raw"] = repr(r)[:100]
            elif op == "step":
                rep = {"ok": True, "cmd": ctl.step(req["state"])}
            elif op == "end":
                say({"ok": True})
                return 0
            else:
                rep = {"ok": False, "error": "unknown op {!r}".format(op)}
        except BaseException as e:  # noqa: BLE001
            traceback.print_exc()
            rep = {"ok": False, "error": "{}: {}".format(type(e).__name__, e), "trace": _trace()}
        say(rep)
    return 0


# =================================================================================================== static checks
def _literal(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _hit(name):
    """A networking, ctypes / cffi or process module (or a part of one): the acceptance's network audit."""
    parts = (name or "").split(".")
    return any(".".join(parts[:k]) in NETWORK_MODULES for k in range(1, len(parts) + 1))


def audit_tree(tree):
    """[(line, what)] of one parsed file, by the rules of the acceptance's network audit: imports of networking, ctypes
    / cffi and process modules; calls that start a program; imports under a computed name; exec / eval / compile of
    computed code."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(n.lineno, a.name) for a in n.names if _hit(a.name)]
        elif isinstance(n, ast.ImportFrom):
            if n.level == 0 and _hit(n.module):
                out.append((n.lineno, n.module))
            elif n.level == 0 and n.module and any(_hit(n.module + "." + a.name) for a in n.names):
                out.append((n.lineno, n.module + "." + ",".join(a.name for a in n.names)))
        elif isinstance(n, ast.Call):
            f = n.func
            fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            owner = f.value.id if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else ""
            arg0 = n.args[0] if n.args else None
            if fname in DYNAMIC_IMPORT_CALLS:
                if arg0 is None or not _literal(arg0):
                    if fname not in ("spec_from_file_location", "load_module"):
                        out.append((n.lineno, "%s(<computed name>)" % fname))
                elif _hit(arg0.value):
                    out.append((n.lineno, arg0.value))
            elif fname in CODE_CALLS and isinstance(f, ast.Name) and (arg0 is None or not _literal(arg0)):
                out.append((n.lineno, "%s(<computed code>)" % fname))
            elif fname in PROCESS_CALLS and (owner in ("os", "posix", "_posixsubprocess") or isinstance(f, ast.Name)):
                out.append((n.lineno, "%s.%s (starts a program)" % (owner or "", fname)))
    return out


def _audit_message(what):
    if what.endswith("(starts a program)"):
        return "{}: starting another program is not allowed".format(what[:-len(" (starts a program)")].lstrip("."))
    if what.endswith("(<computed name>)"):
        return "{}: an import under a computed name is not allowed".format(what)
    if what.endswith("(<computed code>)"):
        return "{}: exec / eval / compile of computed code is not allowed".format(what)
    return "import of {}: networking, ctypes / cffi and process modules are not allowed".format(what)


OUTSIDE_PATH = re.compile(r"^(/(home|root|Users)/|~/)|(^|/)given/(robot|servo|unit|cells|acceptance|controller)(/|$)")


def static_checks(folder, R):
    py, total, links = [], 0, 0
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not os.path.islink(os.path.join(root, d))]
        for n in files:
            p = os.path.join(root, n)
            if os.path.islink(p) or not os.path.isfile(p):
                links += 1
                continue
            total += os.path.getsize(p)
            if n.endswith(".py"):
                py.append(p)
    R.info["folder"] = {"files": len(py), "bytes": total, "skipped_links": links}
    if total > SIZE_CAP_MB * 1e6:
        R.add("error", "static", "the controller folder holds {:.1f} MB (at most {} MB)".format(total / 1e6, SIZE_CAP_MB))
    if links:
        R.add("warning", "static", "{} symbolic link(s) or special file(s) in the folder: they are not copied".format(links))
    local = set()
    for n in os.listdir(folder):
        if n.endswith(".py"):
            local.add(n[:-3])
        elif os.path.isdir(os.path.join(folder, n)) and n != "__pycache__":
            local.add(n)
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | set(sys.builtin_module_names)
    third = {}
    ctl_class = None
    for p in sorted(py):
        rel = os.path.relpath(p, folder)
        try:
            with open(p, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), p)
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            R.add("error", "static", "{}: cannot be parsed: {}".format(rel, e))
            continue
        for line, what in audit_tree(tree):
            R.add("error", "static", "{}:{}: {}".format(rel, line, _audit_message(what)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    third.setdefault(a.name.split(".")[0], set()).add(rel)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                third.setdefault(node.module.split(".")[0], set()).add(rel)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and OUTSIDE_PATH.search(node.value):
                R.add("warning", "static", "{}:{}: a path outside the controller folder ({!r}): the controller computer has "
                                           "only the files of submission/controller/; copy what the controller reads "
                                           "into it".format(rel, node.lineno, node.value[:80]))
            elif isinstance(node, ast.ClassDef) and node.name == "Controller" and rel == "controller.py":
                ctl_class = node
    if not os.path.isfile(os.path.join(folder, "controller.py")):
        R.add("error", "static", "the folder has no controller.py")
    elif ctl_class is None:
        R.add("error", "static", "controller.py defines no class Controller at module level")
    else:
        meth = {n.name: n for n in ctl_class.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name, nargs in (("__init__", 2), ("setup", 4), ("plan", 3), ("step", 2)):
            f = meth.get(name)
            if f is None:
                if name != "__init__" and ctl_class.bases:
                    R.add("warning", "static", "Controller defines no {}() itself (inherited ones are not checked)".format(name))
                elif name != "__init__":
                    R.add("error", "static", "Controller has no method {}()".format(name))
                continue
            pos = len(f.args.posonlyargs) + len(f.args.args)
            if name == "plan" and pos == 2 and f.args.vararg is None:
                R.add("note", "static", "Controller.plan(self, job) takes no state: the cell computer calls it with the "
                                        "job alone")
            elif pos < nargs and f.args.vararg is None:
                R.add("error", "static", "Controller.{}() takes {} positional arguments with self, the cell computer passes "
                                         "{}".format(name, pos, nargs))
    others = sorted(m for m in third if m not in stdlib and m not in local and not _hit(m))
    R.info["packages"] = others
    if others:
        code = ("import importlib.util, json, sys; n = json.loads(sys.argv[1]); "
                "print(json.dumps([m for m in n if importlib.util.find_spec(m) is None]))")
        try:
            tmp = tempfile.mkdtemp(prefix="cc_imports_")
            out = subprocess.run([sys.executable, "-E", "-s", "-c", code, json.dumps(others)], cwd=tmp,
                                 capture_output=True, text=True, timeout=60)
            shutil.rmtree(tmp, ignore_errors=True)
            missing = json.loads(out.stdout.strip() or "[]")
        except (OSError, ValueError, subprocess.SubprocessError):
            missing = []
            R.add("warning", "static", "could not check whether the packages {} are installed".format(", ".join(others)))
        for m in missing:
            R.add("error", "static", "package {} is not installed for {} (used in {})".format(
                m, sys.executable, ", ".join(sorted(third[m]))))


# ===================================================================================================== the child
CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _proc_stat(pid):
    """The fields of /proc/<pid>/stat after the command name, or None."""
    try:
        with open("/proc/{}/stat".format(pid)) as fh:
            st = fh.read()
        return st[st.rindex(")") + 2:].split()
    except (OSError, ValueError):
        return None


def _proc_cpu_s(pid):
    """utime + stime of a process (all its threads), seconds; None without /proc."""
    f = _proc_stat(pid)
    try:
        return (int(f[11]) + int(f[12])) / float(CLK)
    except (TypeError, ValueError, IndexError):
        return None


def _descendants(pid):
    """The live child processes of pid and theirs (Linux: /proc/<pid>/task/<tid>/children); None without them."""
    out, stack, seen = [], [pid], {pid}
    found = False
    while stack:
        p = stack.pop()
        try:
            tids = os.listdir("/proc/{}/task".format(p))
        except OSError:
            continue
        for tid in tids:
            try:
                with open("/proc/{}/task/{}/children".format(p, tid)) as fh:
                    kids = [int(x) for x in fh.read().split()]
                found = True
            except (OSError, ValueError):
                continue
            for k in kids:
                if k not in seen:
                    seen.add(k)
                    out.append(k)
                    stack.append(k)
    return out if found else None


def _threads_cpu_s(pid):
    """CPU time of every thread of pid, seconds (/proc schedstat, nanoseconds); 0 when it is gone."""
    tot = 0
    try:
        tids = os.listdir("/proc/{}/task".format(pid))
    except OSError:
        return 0.0
    for tid in tids:
        try:
            with open("/proc/{}/task/{}/schedstat".format(pid, tid)) as fh:
                tot += int(fh.read().split()[0])
        except (OSError, ValueError, IndexError):
            continue
    return tot * 1e-9


def _proc_clock_s(pid):
    """CPU time of a whole process (all its threads, also those that have ended), seconds: Linux's per-process CPU
    clock; where it cannot be read, the sum over its live threads."""
    try:
        return time.clock_gettime(((~int(pid)) << 3) | 2)
    except (OSError, ValueError, OverflowError, AttributeError):
        return _threads_cpu_s(pid)


def _session_pids(sid):
    out = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return out
    for name in names:
        if not name.isdigit():
            continue
        f = _proc_stat(name)
        try:
            if f is not None and int(f[3]) == int(sid) and f[0] != "Z":
                out.append(int(name))
        except (ValueError, IndexError):
            continue
    return out


def _copy_tree(src, dst):
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not os.path.islink(os.path.join(root, d))]
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for n in files:
            p = os.path.join(root, n)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            shutil.copyfile(p, os.path.join(dst, rel, n))


def _set_modes(top, dmode, fmode):
    for root, dirs, files in os.walk(top, topdown=False):
        for n in files:
            os.chmod(os.path.join(root, n), fmode)
        for n in dirs:
            os.chmod(os.path.join(root, n), dmode)
    os.chmod(top, dmode)


def _rmtree(top):
    try:
        _set_modes(top, 0o755, 0o644)
    except OSError:
        pass
    shutil.rmtree(top, ignore_errors=True)


class Child(object):
    """The controller in its own process, as on the controller computer (one JSON line per call).

    CPU accounting as at acceptance: a call is charged everything the controller used since its previous reply (the
    first call of a process: since that call was sent). The controller process reports its own CPU (all its threads
    and the child processes it has waited for); on Linux the CPU of its running child processes is added from /proc."""

    def __init__(self, folder, scratch, log_path):
        self.folder, self.scratch, self.log_path = folder, scratch, log_path
        self.proc, self.q, self.dead, self.root = None, None, True, None
        self.kids_last = None             # CPU (s) of the running child processes at the previous reply
        self.total_last = None            # the whole tree's CPU (s, /proc) at the previous reply: the kill check
        self.procfs = False

    def start(self):
        self.root = tempfile.mkdtemp(prefix="run", dir=self.scratch)
        prog, home, tmp = [os.path.join(self.root, n) for n in ("program", "home", "tmp")]
        _copy_tree(self.folder, prog)
        _set_modes(prog, 0o555, 0o444)
        os.makedirs(home)
        os.makedirs(tmp)
        env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": home, "TMPDIR": tmp, "TMP": tmp,
               "TEMP": tmp, "LANG": "C.UTF-8", "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1"}
        cmd = [sys.executable, "-B", "-E", "-s", os.path.abspath(__file__), "--child", prog, home]
        log = open(self.log_path, "ab")
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, cwd=prog, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env,
                                     text=True, bufsize=1, start_new_session=True)
        log.close()
        self.q = queue.Queue()
        self.dead = False
        self.kids_last = self.total_last = None
        self.procfs = _descendants(self.proc.pid) is not None

        def pump(pr, q):
            # bounded lines, as the cell computer reads them: a reply longer than MAX_REPLY_LINE is a protocol error
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
                    if len(buf) > MAX_REPLY_LINE + 64:          # + the CPU field this test appends
                        q.put(OVERFLOW)
                        buf = b""
                        break
            except (OSError, ValueError):
                pass
            q.put(None)
        threading.Thread(target=pump, args=(self.proc, self.q), daemon=True).start()
        try:
            line = self.q.get(timeout=INIT_WALL_S)
        except queue.Empty:
            self.kill()
            return False, {"error": "import + Controller(workdir) did not answer within {:.0f} s".format(INIT_WALL_S),
                           "wall_s": time.perf_counter() - t0}
        wall = time.perf_counter() - t0
        if line is None:
            self.kill()
            return False, {"error": "the controller process ended during the import (see the log)", "wall_s": wall}
        try:
            rep = json.loads(line)
        except ValueError:
            self.kill()
            return False, {"error": "protocol: {!r}".format(line[:200]), "wall_s": wall}
        rep["wall_s"] = wall
        if not rep.get("ready"):
            self.kill()
            return False, rep
        return True, rep

    def _kids_cpu_s(self):
        kids = _descendants(self.proc.pid) if self.procfs else None
        return sum(_proc_clock_s(k) for k in kids) if kids else 0.0

    def _total_cpu_s(self):
        c = _proc_cpu_s(self.proc.pid)
        if c is None:
            return None
        try:
            c = time.clock_gettime(((~int(self.proc.pid)) << 3) | 2)
        except (OSError, ValueError, OverflowError, AttributeError):
            pass
        f = _proc_stat(self.proc.pid)
        try:
            c += (int(f[13]) + int(f[14])) / float(CLK)          # the child processes it has waited for
        except (TypeError, ValueError, IndexError):
            pass
        return c + self._kids_cpu_s()

    def call(self, op, payload, cpu_cap, wall_cap):
        """(status, reply, cpu_s, wall_s); status "ok" | "error" | "cpu" | "timeout" | "dead" | "protocol"."""
        if self.dead or self.proc is None:
            return "dead", None, 0.0, 0.0
        msg = {"op": op}
        msg.update(payload or {})
        if self.total_last is None:                 # the first call of this process: charged from its own start
            self.total_last = self._total_cpu_s()
            self.kids_last = self._kids_cpu_s()
        base = self.total_last
        t0 = time.perf_counter()
        try:
            self.proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            self.kill()
            return "dead", None, 0.0, 0.0
        deadline = t0 + wall_cap
        while True:
            try:
                line = self.q.get(timeout=min(0.25, max(0.001, deadline - time.perf_counter())))
                break
            except queue.Empty:
                now = time.perf_counter()
                c = self._total_cpu_s()
                if base is not None and c is not None and c - base > cpu_cap:
                    self.kill()
                    return "cpu", None, c - base, now - t0
                if now > deadline:
                    self.kill()
                    return "timeout", None, (c - base) if (c is not None and base is not None) else 0.0, now - t0
        wall = time.perf_counter() - t0
        if line is None:
            self.kill()
            return "dead", None, 0.0, wall
        if line == OVERFLOW:
            self.kill()
            return "protocol", {"line": "a reply line longer than {} bytes".format(MAX_REPLY_LINE)}, 0.0, wall
        try:
            rep = json.loads(line)
        except ValueError:
            self.kill()
            return "protocol", {"line": line[:200]}, 0.0, wall
        kids = self._kids_cpu_s()
        # a child process that ended and was waited for leaves the running ones and enters the controller's own count
        cpu = max(0.0, float(rep.pop("cpu_s", 0.0)) + kids - self.kids_last)
        self.kids_last = kids
        self.total_last = self._total_cpu_s()
        status = "ok" if rep.get("ok") else "error"
        if cpu > cpu_cap:
            status = "cpu"
        return status, rep, cpu, wall

    def kill(self):
        if self.proc is not None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.dead = True

    def close(self):
        """End the process ({"op": "end"}); returns the number of its processes still running after 5 s."""
        stray = 0
        if self.proc is not None:
            if not self.dead:
                try:
                    self.proc.stdin.write('{"op":"end"}\n')
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=5)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
                time.sleep(0.2)
                stray = len(_session_pids(self.proc.pid))
            self.kill()
        if self.root:
            _rmtree(self.root)
        return stray


# ===================================================================================================== toy plant
class ToyPlant(object):
    """Six toy servos on a toy bus. NOT the DA-6: each servo follows its goal as a first-order lag (TOY["tau_s"]) with
    a speed limit (TOY["vmax_ticks_s"], lowered by the goal-speed register: a whole number, 0 = no limit), after a
    fixed latency (TOY["latency_s"]); goals are clamped to the record's limits; the acceleration register is ignored.
    Servos 1 and 2 carry a toy two-link pendulum (upright at their record zero) whose gravity torque shows in their
    load registers, sampled every TOY["load_refresh_s"] and held in between."""

    def __init__(self, record, start_ticks):
        import numpy as np
        self.np = np
        sv = [record["servos"][j] for j in JOINTS]
        self.lo = np.array([s["min_ticks"] for s in sv], float)
        self.hi = np.array([s["max_ticks"] for s in sv], float)
        self.zero = np.array([s["zero_ticks"] for s in sv], float)
        self.dirn = np.array([s["direction"] for s in sv], float)
        self.p = np.clip(np.array(start_ticks, float), self.lo, self.hi)
        self.v = np.zeros(6)
        self.goal = self.p.copy()
        self.vmax = np.full(6, TOY["vmax_ticks_s"])
        self.vlim = self.vmax.copy()
        self.fifo = []
        self.t = 0.0
        self._load = (None, [0] * 6)

    def cycle(self, goal=None, speed=None):
        np = self.np
        if goal is not None:
            g = np.clip(np.array(goal, float), self.lo, self.hi)
            s = None if speed is None else np.floor(np.abs(np.array(speed, float)))
            self.fifo.append((self.t + TOY["latency_s"], g, s))
        dt = CYCLE_S / TOY["substeps"]
        for _ in range(TOY["substeps"]):
            while self.fifo and self.fifo[0][0] <= self.t + 1e-12:
                _, g, s = self.fifo.pop(0)
                self.goal = g
                self.vlim = self.vmax.copy() if s is None else np.where(s > 0, np.minimum(s, self.vmax), self.vmax)
            self.v = np.clip((self.goal - self.p) / TOY["tau_s"], -self.vlim, self.vlim)
            self.p = self.p + self.v * dt
            self.t = round(self.t + dt, 9)

    def _loads(self):
        th = self.dirn * (self.p - self.zero) * TICK_RAD
        g, m1, l1, m2, l2 = TOY["g"], TOY["m1_kg"], TOY["l1_m"], TOY["m2_kg"], TOY["l2_m"]
        g2 = m2 * g * 0.5 * l2 * math.sin(th[0] + th[1])
        g1 = m1 * g * 0.5 * l1 * math.sin(th[0]) + m2 * g * (l1 * math.sin(th[0]) + 0.5 * l2 * math.sin(th[0] + th[1]))
        load = [0] * 6
        for i, tau in ((0, -g1), (1, -g2)):
            load[i] = int(max(-1000, min(1000, round(1000.0 * self.dirn[i] * tau / TOY["stall_nm"]))))
        return load

    def read(self):
        np = self.np
        k = int(math.floor(self.t / TOY["load_refresh_s"] + 1e-9))
        if self._load[0] != k:
            self._load = (k, self._loads())
        return {"position": [int(round(x)) for x in self.p], "speed": [int(round(x)) for x in self.v],
                "load": list(self._load[1]), "moving": bool(self.fifo) or bool(np.any(np.abs(self.v) > TOY["moving_ticks_s"]))}


# ================================================================================================== toy scenario
def _rotz(deg, v):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]


def _unitv(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _r(v, nd=5):
    return [round(float(x), nd) for x in v]


def toy_scenario():
    """Three toy cells in the payload formats of the specification (every coordinate of the cells positive). Positions,
    sizes and times are toy values in a toy cell: not chosen for the DA-6, and the self-test does not judge what the
    controller does with them."""
    base = {"pos": [0.4, 0.32, 0.0], "yaw_deg": 90.0}

    def cell_pt(p):
        q = _rotz(base["yaw_deg"], p)
        return _r([q[0] + base["pos"][0], q[1] + base["pos"][1], q[2] + base["pos"][2]])

    def cell_dir(v):
        return _r(_rotz(base["yaw_deg"], _unitv(v)), 6)

    def obj(name, kind, xy, prims):
        p = cell_pt([xy[0], xy[1], 0.0])
        return {"name": name, "kind": kind, "pose": {"pos": [round(p[0], 6), round(p[1], 6), 0.0],
                                                     "yaw_deg": round(base["yaw_deg"], 4)}, "primitives": prims}

    def box(name, half, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        return {"name": name, "type": "box", "size": _r(half), "pos": _r(pos), "quat": _r(quat, 6)}

    def cyl(name, rad, hh, pos):
        return {"name": name, "type": "cylinder", "size": _r([rad, hh]), "pos": _r(pos), "quat": [1.0, 0.0, 0.0, 0.0]}

    def scene(objects):
        return {"format": SCENE_FORMAT, "frame": SCENE_FRAME, "robot_base": dict(base), "objects": objects,
                "table": dict(SCENE_TABLE), "survey_tolerance": dict(SURVEY_TOLERANCE)}

    def reach(jid, tool, pos, app, jaw, exempt, tt):
        return {"id": jid, "type": "reach", "tool": tool, "clearance_mm": CLEARANCE_MM, "time_target_s": tt,
                "exempt": exempt, "position": cell_pt(pos), "approach": cell_dir(app), "approach_tol_deg": APPROACH_TOL_DEG,
                "jaw_axis": None if jaw is None else cell_dir(jaw), "jaw_tol_deg": JAW_TOL_DEG}

    def line_path(jid, a, b, speed, normal, att, exempt):
        length = math.sqrt(sum((y - x) ** 2 for x, y in zip(a, b)))
        dur = length / speed
        t = [round(k * CYCLE_S, 3) for k in range(int(round(dur / CYCLE_S)) + 1)]
        nrm = _unitv(normal)
        pts = [cell_pt([a[i] + (b[i] - a[i]) * min(1.0, tk / dur) + nrm[i] * GAP_MM * 1e-3 for i in range(3)]) for tk in t]
        return {"id": jid, "type": "path", "tool": "needle", "clearance_mm": CLEARANCE_MM,
                "time_target_s": round(att + t[-1] + 0.5, 2), "exempt": exempt, "t": t, "points": pts,
                "normal": cell_dir(nrm), "axis_tol_deg": AXIS_TOL_DEG, "gap_mm": GAP_MM, "gap_tol_mm": GAP_TOL_MM,
                "speed_mm_s": round(speed * 1e3, 2), "shape": {"shape": "line", "length_mm": round(length * 1e3, 1)},
                "approach_time_target_s": att}

    tilt = math.radians(25.0)
    n_board = [0.0, -math.sin(tilt), math.cos(tilt)]
    centre = [0.0, 0.26, 0.05]
    board_obj = [centre[0], centre[1]]
    q_board = (math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0)
    board_pos = [0.0, -n_board[1] * 0.003, centre[2] - n_board[2] * 0.003]
    walls = [box("tray_floor", [0.05, 0.04, 0.002], [0, 0, 0.002]), box("tray_wxn", [0.002, 0.04, 0.025], [-0.048, 0, 0.025]),
             box("tray_wxp", [0.002, 0.04, 0.025], [0.048, 0, 0.025]), box("tray_wyn", [0.05, 0.002, 0.025], [0, -0.038, 0.025]),
             box("tray_wyp", [0.05, 0.002, 0.025], [0, 0.038, 0.025])]
    return {"format": "toy-scenario/1", "note": "toy cells and jobs: they test the interface, not the DA-6",
            "cells": [
                {"name": "toy cell 1", "tool": {"kind": "gripper"},
                 "cell": scene([obj("block", "fixture", (0.12, 0.2), [box("block", [0.03, 0.03, 0.03], [0, 0, 0.03])]),
                                obj("post", "post", (-0.15, 0.18), [cyl("post", 0.012, 0.09, [0, 0, 0.09])])]),
                 "jobs": [reach("T1-01", "gripper", [0.12, 0.2, 0.09], [0, 0, -1], [1, 0, 0], ["block"], 2.0),
                          reach("T1-02", "gripper", [-0.05, 0.25, 0.15], [-0.05, 0.25, -0.05], None, [], 2.0),
                          reach("T1-03", "gripper", [0.05, 0.3, 0.12], [0, 1, 0], None, [], 2.0)]},
                {"name": "toy cell 2", "tool": dict(TOY_PART),
                 "cell": scene([obj("tray", "bin", (0.0, 0.22), walls)]),
                 "jobs": [reach("T2-01", "gripper+part", [0.0, 0.22, 0.035], [0, 0, -1], [1, 0, 0], ["tray"], 2.0),
                          reach("T2-02", "gripper+part", [0.15, 0.15, 0.12], [0, 0, -1], [1, -1, 0], [], 2.0),
                          reach("T2-03", "gripper+part", [0.05, 0.2, 0.1], [0, 0, -1], [0, 1, 0], [], 2.0)]},
                {"name": "toy cell 3", "tool": dict(NEEDLE_TOOL),
                 "cell": scene([obj("board", "board", board_obj, [box("board", [0.06, 0.045, 0.003], board_pos, q_board)]),
                                obj("cup", "cup", (-0.12, 0.2), [cyl("cup", 0.025, 0.025, [0, 0, 0.025])])]),
                 "jobs": [line_path("T3-01", [-0.02, 0.26, 0.05], [0.02, 0.26, 0.05], 0.02, n_board, 2.0, ["board"]),
                          reach("T3-02", "needle", [-0.12, 0.2, 0.058], [0, 0, -1], None, ["cup"], 2.0),
                          reach("T3-03", "needle", [-0.1, 0.22, 0.07], [0, 0, -1], None, ["cup"], 2.0)]}]}


def toy_unit():
    servos = {j: {"id": i + 1, "direction": 1, "zero_ticks": TICKS_PER_TURN // 2, "min_ticks": TOY_TICK_RANGE[0],
                  "max_ticks": TOY_TICK_RANGE[1]} for i, j in enumerate(JOINTS)}
    rec = {"format": RECORD_FORMAT, "serial": "TOY-0001", "convention": RECORD_CONVENTION,
           "uncertainty_deg": dict(RECORD_UNCERTAINTY), "servos": servos}
    return {"serial": "TOY-0001", "calibration": rec}


def start_ticks(record, pose):
    """The toy servos' start: the record's zero_ticks ("zero") or six given tick values, inside the limits."""
    if pose == "zero":
        ticks = [record["servos"][j]["zero_ticks"] for j in JOINTS]
    else:
        ticks = [int(v) for v in pose.split(",")]
        if len(ticks) != 6:
            raise SystemExit("--start: six comma-separated tick values or zero")
    return [min(max(t, record["servos"][j]["min_ticks"]), record["servos"][j]["max_ticks"]) for t, j in zip(ticks, JOINTS)]


# ======================================================================================================== report
class Report(object):
    def __init__(self, quiet=False):
        self.findings, self.jobs, self.cells, self.info = [], [], [], {}
        self.calls = {"start": [], "setup": [], "plan": [], "step": []}
        self.cycles = self.late_cycles = self.restarts = 0
        self.quiet = quiet
        self._seen = {}

    def add(self, severity, where, message, trace=None):
        key = (severity, where, message)
        if key in self._seen:
            self._seen[key]["count"] += 1
            return
        f = {"severity": severity, "where": where, "message": message, "count": 1}
        if trace:
            f["trace"] = list(trace)[-3:]
        self._seen[key] = f
        self.findings.append(f)

    def say(self, text):
        if not self.quiet:
            print(text, flush=True)

    def n(self, severity):
        return sum(1 for f in self.findings if f["severity"] == severity)


def _fmt_cpu(s):
    return "{:.2f} ms".format(s * 1e3) if s < 1.0 else "{:.2f} s".format(s)


def parse_cmd(cmd):
    """(goal, speed, acc) of a step reply as the cell computer reads it, or None when it is malformed (a protocol
    error: the job is aborted). `cmd` is the reply with None read as {} (no command)."""
    if not isinstance(cmd, dict):
        return None

    def vec(key, lo, hi):
        v = cmd.get(key)
        if v is None:
            return True, None
        if not isinstance(v, (list, tuple)) or len(v) != 6:
            return False, None
        out = []
        for x in v:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return False, None
            try:
                fx = float(x)
            except OverflowError:
                return False, None
            if not math.isfinite(fx) or not (lo <= fx <= hi):
                return False, None
            out.append(fx)
        return True, out
    ok, goal = vec("goal", -GOAL_BOUND, GOAL_BOUND)
    if not ok:
        return None
    ok, spd = vec("speed", -REG_BOUND, REG_BOUND)
    if not ok:
        return None
    ok, acc = vec("acc", -REG_BOUND, REG_BOUND)
    if not ok:
        return None
    if goal is not None:
        goal = [int(round(x)) for x in goal]
    if spd is not None:
        spd = [abs(x) for x in spd]
    if acc is not None:
        acc = [abs(x) for x in acc]
    return goal, spd, acc


def malformed_why(cmd):
    """Why parse_cmd rejects a reply, in words."""
    if not isinstance(cmd, dict):
        return "the reply is {}, not an object (dict) or None".format(type(cmd).__name__)
    for key, bound in (("goal", GOAL_BOUND), ("speed", REG_BOUND), ("acc", REG_BOUND)):
        v = cmd.get(key)
        if v is None:
            continue
        if not isinstance(v, (list, tuple)):
            return "{} is {}, not a list of six numbers or null".format(key, type(v).__name__)
        if len(v) != 6:
            return "{} has {} entries, not six".format(key, len(v))
        for i, x in enumerate(v):
            if isinstance(x, bool):
                return "{}[{}] is {}: booleans are not numbers".format(key, i, x)
            if not isinstance(x, (int, float)):
                return "{}[{}] is {}, not a number".format(key, i, type(x).__name__)
            try:
                fx = float(x)
            except OverflowError:
                return "{}[{}] is too large".format(key, i)
            if not math.isfinite(fx):
                return "{}[{}] = {} is not finite".format(key, i, x)
            if abs(fx) > bound:
                return "{}[{}] = {} is outside +-{:g}".format(key, i, x, bound)
    return "malformed"


def check_cmd(cmd, record, where, R, path_job):
    """Read a step reply as the cell computer does; returns (ok, goal, speed, acc, path_start, done) with ok False
    for a protocol error (the reply is not written and the job is aborted). Adds warnings and notes for what the cell
    computer accepts but the controller probably did not mean."""
    if cmd is None:
        R.add("note", where, "step returned None: no command (the servos keep their goals), not done")
        return True, None, None, None, False, False
    parsed = parse_cmd(cmd)
    if parsed is None:
        R.add("error", where, "protocol error: {}; the job is aborted (as at acceptance)".format(malformed_why(cmd)))
        return False, None, None, None, False, False
    goal, spd, acc = parsed
    missing = [k for k in CMD_KEYS if k not in cmd]
    if missing:
        R.add("note", where, "reply without {}: read as null / false".format(", ".join(missing)))
    extra = sorted(str(k) for k in cmd if k not in CMD_KEYS)
    if extra:
        R.add("warning", where, "unknown keys in the reply (ignored): {}".format(", ".join(extra)))
    if goal is not None:
        if any(not isinstance(v, int) for v in cmd["goal"]):
            R.add("note", where, "goal values that are not whole numbers: rounded to the nearest whole tick")
        for i, j in enumerate(JOINTS):
            s = record["servos"][j]
            if not s["min_ticks"] <= goal[i] <= s["max_ticks"]:
                R.add("warning", where, "goal of servo {} ({}) outside its EEPROM limits [{}, {}]: the servo clamps "
                                        "it".format(i + 1, j, s["min_ticks"], s["max_ticks"]))
    for key, vals in (("speed", spd), ("acc", acc)):
        if vals is None:
            continue
        if any(x < 0 for x in cmd[key]):
            R.add("warning", where, "{} values below 0: written without their sign".format(key))
        if any(0 < x < 1 for x in vals):
            R.add("warning", where, "{} values between 0 and 1: rounded down to 0, i.e. the servo's {}".format(
                key, "maximum speed" if key == "speed" else "default acceleration"))
    flags = []
    for key in ("path_start", "done"):
        v = cmd.get(key, False)
        if not isinstance(v, bool):
            R.add("warning", where, "{} is {!r}, not true or false: read as {}".format(key, v, bool(v)))
        flags.append(bool(v))
    if flags[0] and not path_job:
        R.add("warning", where, "path_start in a reach job: ignored")
    return True, goal, spd, acc, flags[0], flags[1]


# ========================================================================================================= runs
class Session(object):
    def __init__(self, folder, unit, R, log_path, scratch):
        self.folder, self.unit, self.R, self.log_path, self.scratch = folder, unit, R, log_path, scratch
        self.record = unit["calibration"]
        self.child = None

    def start(self, where):
        self.child = Child(self.folder, self.scratch, self.log_path)
        ok, info = self.child.start()
        self.R.calls["start"].append({"where": where, "ok": ok, "wall_s": info.get("wall_s"), "cpu_s": info.get("cpu_s")})
        if not ok:
            self.R.add("error", where, info.get("error", "the controller did not start"))
            return False
        self.R.info["cpu_accounting"] = ("since the previous reply: the controller process (threads, waited-for child "
                                         "processes)" + (" and its running child processes (/proc)" if self.child.procfs
                                                         else "; running child processes are not measured here"))
        if info.get("wall_s", 0.0) > INIT_WALL_S:
            self.R.add("error", where, "import + Controller(workdir) took {:.1f} s (at most {:.0f} s)".format(
                info["wall_s"], INIT_WALL_S))
        self.R.say("[start] import + Controller(workdir): {:.2f} s wall-clock (at most {:.0f} s)".format(
            info.get("wall_s", 0.0), INIT_WALL_S))
        return True

    def restart(self, where, cell):
        self.R.restarts += 1
        self.R.say("  restart: a new controller process, then setup again")
        if self.child is not None:
            self.child.close()
        return self.start(where) and self.setup(cell, where)

    def setup(self, cell, where):
        payload = {"unit": self.unit, "cell": cell["cell"], "tool": cell["tool"]}
        st, rep, cpu, wall = self.child.call("setup", payload, SETUP_CPU_S, SETUP_WALL_S)
        self.R.calls["setup"].append({"where": where, "status": st, "cpu_s": cpu, "wall_s": wall})
        self.R.say("  setup ({} tool): {}, cpu {}, wall {:.2f} s (at most {:.0f} s CPU, {:.0f} s wall-clock)".format(
            cell["tool"].get("kind"), st, _fmt_cpu(cpu), wall, SETUP_CPU_S, SETUP_WALL_S))
        return self._status(st, rep, cpu, wall, "setup", where, SETUP_CPU_S, SETUP_WALL_S)

    def _status(self, st, rep, cpu, wall, op, where, cpu_cap, wall_cap):
        if st == "ok":
            if wall > wall_cap:
                self.R.add("error", where, "{} took {:.1f} s wall-clock (at most {:.0f} s)".format(op, wall, wall_cap))
                return False
            return True
        if st == "error":
            err = (rep or {}).get("error", "?")
            if (rep or {}).get("encode"):
                self.R.add("error", where, "{}: the reply cannot be encoded as JSON ({}); return plain Python values "
                                           "(int(), float(), bool(), .tolist())".format(op, err))
            else:
                self.R.add("error", where, "{} raised {}".format(op, err), trace=(rep or {}).get("trace"))
        elif st == "cpu":
            self.R.add("error", where, "{} used more than {} of CPU (since the previous reply): the process is "
                                       "ended".format(op, _fmt_cpu(cpu_cap)))
        elif st == "timeout":
            self.R.add("error", where, "{} took more than {:.0f} s wall-clock: the process is ended".format(op, wall_cap))
        elif st == "dead":
            self.R.add("error", where, "the controller process ended during {} (see the log)".format(op))
        else:
            self.R.add("error", where, "{}: the reply is not one JSON line of at most {} bytes ({}): the process is "
                                       "ended".format(op, MAX_REPLY_LINE, (rep or {}).get("line", "")[:80]))
        return False

    def run_cell(self, ci, cell, start_pose):
        where = "cell {} ({})".format(ci + 1, cell.get("name", ""))
        self.R.say("[{}] tool {}".format(where, cell["tool"].get("kind")))
        toy = ToyPlant(self.record, start_ticks(self.record, start_pose))
        if not self.setup(cell, where) and not self.restart(where, cell):
            self.R.add("error", where, "setup failed twice: the cell's jobs are not run")
            return
        for job in cell["jobs"]:
            self.run_job(toy, cell, job)

    def run_job(self, toy, cell, job):
        R = self.R
        jid = job.get("id", "?")
        where = "job {}".format(jid)
        st0 = toy.read()
        state0 = {k: st0[k] for k in PLAN_STATE_KEYS}
        st, rep, cpu, wall = self.child.call("plan", {"job": job, "state": state0}, PLAN_CPU_S, PLAN_WALL_S)
        R.calls["plan"].append({"where": where, "status": st, "cpu_s": cpu, "wall_s": wall})
        rec = {"id": jid, "type": job.get("type"), "plan": st, "plan_cpu_s": cpu, "cycles": 0, "late": 0, "status": None}
        R.jobs.append(rec)
        if not self._status(st, rep, cpu, wall, "plan", where, PLAN_CPU_S, PLAN_WALL_S):
            rec["status"] = "plan " + st
            R.say("  {} {:5s} plan {}: the job is not run".format(jid, job.get("type", "?"), st))
            if st != "error" or self.child.dead:
                self.restart(where, cell)
            return
        if rep.get("raw") is not None:
            R.add("warning", where, "plan returned {} (read as \"ok\"); return \"ok\" or \"infeasible\"".format(rep["raw"]))
        rec["answer"] = rep.get("result")
        if rep.get("result") == "infeasible":
            rec["status"] = "infeasible"
            R.say("  {} {:5s} plan \"infeasible\" (cpu {}): the arm holds".format(jid, job.get("type"), _fmt_cpu(cpu)))
            return
        is_path = job.get("type") == "path"
        T = float(job.get("time_target_s", 2.0))
        cap = (2.0 * float(job.get("approach_time_target_s", T)) + float(job["t"][-1]) + 2.0) if is_path else max(2.0 * T, T + 2.0)
        t0 = toy.t
        cycle = late_total = 0
        path_t0 = done_t = None
        status = "ok"
        cpus = []
        while True:
            state = toy.read()
            state = {"cycle": cycle, "t": round(toy.t - t0, 4), "position": state["position"], "speed": state["speed"],
                     "load": state["load"], "moving": state["moving"]}
            st, rep, cpu, wall = self.child.call("step", {"state": state}, STEP_KILL_CPU_S, STEP_KILL_WALL_S)
            cpus.append(cpu)
            if st != "ok":
                self._status(st, rep, cpu, wall, "step", where + " cycle {}".format(cycle), STEP_KILL_CPU_S, STEP_KILL_WALL_S)
                status = "step " + st
                if st != "error" or self.child.dead:
                    rec["status"] = status
                    self._job_line(job, rec, cpus, cycle, None, path_t0)
                    self.restart(where, cell)
                    return
                break
            if wall > STEP_KILL_WALL_S:
                R.add("error", where, "a step took {:.1f} s wall-clock (more than {:.0f} s ends the process)".format(
                    wall, STEP_KILL_WALL_S))
            ok, goal, spd, acc, pstart, done = check_cmd(rep.get("cmd"), self.record, where, R, is_path)
            if not ok:
                status = "step protocol (malformed command)"
                break
            late = int(cpu // CYCLE_S) if cpu > CYCLE_S else 0
            R.cycles += 1 + late
            if late:
                R.late_cycles += 1 + late
                late_total += 1 + late
            for _ in range(late):
                toy.cycle(None)
            toy.cycle(goal, spd)
            cycle += 1 + late
            if is_path and path_t0 is None and pstart:
                path_t0 = toy.t
            elif is_path and path_t0 is not None and pstart:
                R.add("note", where, "path_start set again: only the first one counts")
            if done:
                done_t = toy.t
                break
            if toy.t - t0 > cap + 1e-9:
                status = "cap"
                break
        rec["cycles"], rec["late"] = cycle, late_total
        if status == "cap":
            R.add("warning", where, "no done within the job's cap ({:.2f} s): at acceptance the job would not be "
                                    "completed".format(cap))
        if done_t is not None:
            for _ in range(int(round(SETTLE_S / CYCLE_S))):
                toy.cycle(None)
            if is_path:
                dur = float(job["t"][-1])
                if path_t0 is None:
                    R.add("error", where, "done without path_start: a path job needs path_start (the job is not completed)")
                    status = "no path_start"
                elif done_t + 1e-9 < path_t0 + dur:
                    R.add("error", where, "done {:.2f} s before the end of the path (path_start at t = {:.2f} s, "
                                          "duration {:.2f} s): the job is not completed".format(
                                              path_t0 + dur - done_t, path_t0 - t0, dur))
                    status = "done before the path ended"
        if late_total:
            R.add("note", where, "{} cycle(s) late or skipped because of steps over {} of CPU".format(
                late_total, _fmt_cpu(STEP_CPU_S)))
        rec["status"] = status if done_t is not None or status != "ok" else "ended"
        rec["done_s"] = None if done_t is None else round(done_t - t0, 3)
        self._job_line(job, rec, cpus, cycle, done_t if done_t is None else done_t - t0, path_t0 if path_t0 is None else path_t0 - t0)

    def _job_line(self, job, rec, cpus, cycles, done_s, path_s):
        if cpus:
            srt = sorted(cpus)
            rec["step_cpu_ms"] = {"mean": 1e3 * sum(cpus) / len(cpus), "p95": 1e3 * srt[int(0.95 * (len(srt) - 1))],
                                  "max": 1e3 * srt[-1]}
            self.R.calls["step"].extend(cpus)
        txt = "  {} {:5s} plan \"{}\" (cpu {}): {} cycles".format(job.get("id"), job.get("type", "?"), rec.get("answer", "-"),
                                                               _fmt_cpu(rec.get("plan_cpu_s", 0.0)), cycles)
        if path_s is not None:
            txt += ", path_start at {:.2f} s".format(path_s)
        txt += ", done at {:.2f} s".format(done_s) if done_s is not None else ", {}".format(rec.get("status"))
        if cpus:
            txt += "; step cpu mean {:.2f} ms, max {:.2f} ms; late {}".format(rec["step_cpu_ms"]["mean"],
                                                                           rec["step_cpu_ms"]["max"], rec.get("late", 0))
        self.R.say(txt)


def load_unit(path):
    if path is None:
        cand = os.path.join(HERE, "..", "unit", "example_calibration_record.json")
        if os.path.isfile(cand):
            path = cand
        else:
            return toy_unit(), "toy record TOY-0001 (built in)"
    with open(path) as fh:
        d = json.load(fh)
    if "calibration" not in d:
        d = {"serial": d.get("serial", "?"), "calibration": d}
    rec = d["calibration"]
    if rec.get("format") != RECORD_FORMAT or sorted(rec.get("servos", {})) != sorted(JOINTS):
        raise SystemExit("{}: not a {} with the six servos {}".format(path, RECORD_FORMAT, ", ".join(JOINTS)))
    return d, "{} ({})".format(d["serial"], os.path.normpath(path))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Interface self-test of a DA-6 motion controller (protocol and timing; "
                                             "toy plant, not the DA-6).")
    ap.add_argument("controller", nargs="?", help="path to controller.py (or to its folder)")
    ap.add_argument("--unit")
    ap.add_argument("--scenario")
    ap.add_argument("--dump-scenario", action="store_true")
    ap.add_argument("--start", default="zero")
    ap.add_argument("--json")
    ap.add_argument("--log")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.dump_scenario:
        print(json.dumps(toy_scenario(), indent=1))
        return 0
    if not a.controller:
        ap.error("the path of controller.py is required")
    path = os.path.abspath(a.controller)
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if not os.path.isdir(path) and os.path.basename(path) != "controller.py":
        scratch_src = tempfile.mkdtemp(prefix="cc_src_")
        shutil.copyfile(path, os.path.join(scratch_src, "controller.py"))
        folder = scratch_src
        note = "{} (run as controller.py)".format(a.controller)
    else:
        scratch_src, note = None, a.controller
    R = Report(quiet=a.quiet)
    unit, unit_note = load_unit(a.unit)
    if a.scenario:
        with open(a.scenario) as fh:
            scenario = json.load(fh)
        bad = [i + 1 for i, c in enumerate(scenario.get("cells", [])) if not {"tool", "cell", "jobs"} <= set(c)]
        if not scenario.get("cells") or bad:
            raise SystemExit("{}: needs \"cells\": [{{\"tool\", \"cell\", \"jobs\"}}, ...] (see --dump-scenario)".format(
                a.scenario))
    else:
        scenario = toy_scenario()
    R.info.update({"controller": note, "python": sys.executable, "python_version": sys.version.split()[0],
                   "unit": unit_note})
    print("DA-6 controller interface self-test: protocol and timing only (a toy plant, not the DA-6)")
    print("  controller  {}".format(note))
    print("  python      {} ({})".format(sys.executable, sys.version.split()[0]))
    print("  unit        {}".format(unit_note))
    static_checks(folder, R)
    for f in R.findings:
        R.say("[static] {}: {}".format(f["severity"], f["message"]))
    R.say("[static] {} .py file(s), {:.2f} MB; packages outside the standard library: {}".format(
        R.info["folder"]["files"], R.info["folder"]["bytes"] / 1e6, ", ".join(R.info.get("packages", [])) or "none"))
    scratch = tempfile.mkdtemp(prefix="check_controller_")
    log_path = os.path.join(scratch, "controller.log")
    S = Session(folder, unit, R, log_path, scratch)
    code = 0
    try:
        if not S.start("start"):
            code = 2
        else:
            for ci, cell in enumerate(scenario["cells"]):
                S.run_cell(ci, cell, a.start)
            stray = S.child.close() if S.child else 0
            if stray:
                R.add("error", "end", "{} process(es) of the controller still running after the end".format(stray))
    finally:
        if a.log and os.path.isfile(log_path):
            shutil.copyfile(log_path, a.log)
        tail = []
        if os.path.isfile(log_path):
            with open(log_path, errors="replace") as fh:
                tail = fh.read().splitlines()[-12:]
        if S.child is not None and S.child.proc is not None and not S.child.dead:
            S.child.kill()
        _rmtree(scratch)
        if scratch_src:
            shutil.rmtree(scratch_src, ignore_errors=True)
    frac = 1.0 - R.late_cycles / R.cycles if R.cycles else None
    if code == 0 and frac is not None and frac < ON_TIME_MIN:
        R.add("error", "cycles", "{:.1f} % of the cycles answered within {} of CPU (at least {:.0f} % required)".format(
            100.0 * frac, _fmt_cpu(STEP_CPU_S), 100.0 * ON_TIME_MIN))
    steps = sorted(R.calls["step"])
    print("summary")
    if R.info.get("cpu_accounting"):
        print("  cpu         charged {}".format(R.info["cpu_accounting"]))
    if steps:
        print("  steps       {} calls; CPU mean {:.2f} ms, p95 {:.2f} ms, max {:.2f} ms (on time: at most {})".format(
            len(steps), 1e3 * sum(steps) / len(steps), 1e3 * steps[int(0.95 * (len(steps) - 1))], 1e3 * steps[-1],
            _fmt_cpu(STEP_CPU_S)))
    if frac is not None:
        print("  cycles      {} bus cycles, {} late or skipped: {:.1f} % answered in time (at least {:.0f} % required)".format(
            R.cycles, R.late_cycles, 100.0 * frac, 100.0 * ON_TIME_MIN))
    print("  restarts    {}".format(R.restarts))
    for sev in ("error", "warning", "note"):
        items = [f for f in R.findings if f["severity"] == sev]
        if items:
            print("  {}s ({})".format(sev, len(items)))
            for f in items:
                print("    [{}] {}{}".format(f["where"], f["message"], "  (x{})".format(f["count"]) if f["count"] > 1 else ""))
                for line in f.get("trace", []):
                    print("        | " + line)
    if code == 2 or (R.n("error") and not any(f.get("trace") for f in R.findings)):
        if tail:
            print("  last lines of the controller's log:")
            for line in tail:
                print("    | " + line)
    if code == 0 and R.n("error"):
        code = 1
    print("RESULT: {}".format({0: "interface checks passed (this says nothing about how the controller would do on "
                                  "the DA-6)", 1: "errors found", 2: "the controller could not be started"}[code]))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"exit": code, "info": R.info, "findings": R.findings, "jobs": R.jobs, "cycles": R.cycles,
                       "late_cycles": R.late_cycles, "on_time_fraction": frac, "restarts": R.restarts,
                       "calls": {k: (v if k != "step" else {"n": len(v), "max_s": max(v) if v else None})
                                 for k, v in R.calls.items()}}, fh, indent=1)
    return code


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        sys.exit(_child(sys.argv[2], sys.argv[3]))
    sys.exit(main())
