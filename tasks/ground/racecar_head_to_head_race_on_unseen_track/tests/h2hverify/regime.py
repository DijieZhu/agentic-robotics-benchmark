"""How the grader runs the agent's program (grader side, stdlib only): the run regime.

Every process of a program (the import check, then one process per venue) starts as

    setpriv --reuid=<runner> --regid=<runner> --clear-groups --inh-caps=-all --bounding-set=-all --no-new-privs --
      prlimit --cpu=<backstop> --as=<bytes> --fsize=<bytes> --core=0 --
        python3 -I -B -c <cellkit.nonet>  --      (the house filter: no socket but AF_UNIX, EPERM)
          python3 -I -B -c <netkill>  --          (the same calls KILL the process: the attempt is recorded;
                                                   a new process is refused: fork / vfork / clone without
                                                   CLONE_THREAD fail with EPERM)
            python3 -B -E -s -c <carkit bootstrap> <lib> drive

`setpriv`, `prlimit` and the two filter stages all exec, so the program keeps the pid that `sketch.wire.server`
started and reads its CPU time from (`/proc/<pid>/stat`; `runuser` would fork and hide it); that server already
starts it in a session of its own (process group) and kills the group at the end. The runner account is `nobody`
(`verify_config.json`), never root and never the agent account; when the grader is not root (a maintainer's host run)
the account switch is skipped and the rest is the same. The per-run directory comes from the house kit
(`cellkit.runner.prepare_run_dir`): a root-owned copy of the submission (regular files only), a root-owned copy of
the client library `carkit`, an empty HOME and TMPDIR owned by the runner; its name is an opaque random tag. The
environment is the cell's own `sketch.wire.runner.child_env` (the contract: PATH, HOME, TMPDIR, LANG, PYTHONPATH,
MPLBACKEND, the numeric libraries single-threaded, NPY_DISABLE_CPU_FEATURES; the OpenBLAS kernel never pinned) plus
`program_env_extra` of verify_config.json (MUJOCO_GL and PYOPENGL_PLATFORM = osmesa, the values of the
agent's own session; without them `import mujoco` loads its GLFW backend, whose loader starts a process to read the
library's version, which the regime now refuses). `program_env(base)` makes that environment.

After a program's process group is gone, a process of the runner account still alive (a fork that left the group,
e.g. a daemon that called setsid) is a SURVIVOR: it is counted (gate `clean_run`), then every process of the account
is frozen and killed (`cellkit.lockdown.stop_and_kill_uid`; one that cannot be killed is counted too), and what the
account wrote under the world-writable folders is swept (`cellkit.runner`), so nothing passes from one venue to the
next. No RLIMIT_NPROC (house rule: per uid across containers); instead the killing filter refuses every new process, so a process of the account other than the program itself can no longer exist: no survivor, no
worker the program reaps before its venue ends, no CPU time outside the program's own /proc/<pid>/stat.
"""

import json
import os
import secrets
import shutil
import subprocess
import sys
import time

from cellkit import lockdown, nonet, runner

from . import netkill

__all__ = ["Regime", "TaskError", "STICKY_DIRS"]

STICKY_DIRS = [d for d in ("/tmp", "/dev/shm", "/var/tmp", "/run/lock") if os.path.isdir(d)]
MB = 1024 * 1024


class TaskError(RuntimeError):
    """The grading broke (never the agent's zero)."""


_PROBE = r'''
import json, os, resource, sys
st = {}
with open("/proc/self/status") as f:
    for line in f:
        k, _, v = line.partition(":")
        st[k] = v.strip()
try:
    os.listdir(sys.argv[1]); stage = "readable"
except PermissionError:
    stage = "denied"
except OSError as e:
    stage = "error %s" % e.__class__.__name__
print(json.dumps({"uid": os.getuid(), "euid": os.geteuid(), "gid": os.getgid(), "groups": sorted(os.getgroups()),
                  "no_new_privs": st.get("NoNewPrivs"), "seccomp": st.get("Seccomp"),
                  "seccomp_filters": st.get("Seccomp_filters"), "cap_eff": st.get("CapEff"), "cap_bnd": st.get("CapBnd"),
                  "rlimit_as": resource.getrlimit(resource.RLIMIT_AS)[0], "rlimit_cpu": resource.getrlimit(resource.RLIMIT_CPU)[0],
                  "rlimit_core": resource.getrlimit(resource.RLIMIT_CORE)[0], "stage": stage}))
'''


# the tools the instruction names (numpy, scipy, OpenCV, Pillow, MuJoCo), imported as a program would at a venue
_TOOLS = "import numpy, scipy, cv2, PIL.Image, mujoco; print('tools-ok')"


class Regime(object):
    def __init__(self, vcfg, runs_root, agent_user=None, log=None):
        self.cfg = vcfg
        self.log = log or (lambda s: None)
        self.runs_root = runs_root
        name = vcfg.get("runner_account", "nobody")
        self.acct = runner.runner_account(name, None)              # None when not root: a host run as oneself
        if os.getuid() == 0:
            if self.acct is None:
                raise TaskError("the runner account %r does not exist in this image" % name)
            if self.acct[1] == 0 or (agent_user and self.acct[0] == agent_user):
                raise TaskError("the runner account must be neither root nor the agent account")
        self.name, self.uid, self.gid = self.acct if self.acct else (None, None, None)
        self.as_bytes = int(vcfg["limits"]["address_space_mb"]) * MB
        self.fsize_bytes = int(vcfg["limits"]["file_size_mb"]) * MB
        self.survivor_wait_s = float(vcfg["limits"].get("survivor_wait_s", 3.0))
        self.env_extra = {str(k): str(v) for k, v in sorted((vcfg.get("program_env_extra") or {}).items())}
        self.baseline = set(lockdown.live_uid_processes(self.uid)) if self.uid is not None else set()
        self.stats = {"runs": 0, "survivors": 0, "killed": 0, "left": 0, "stray_files": 0, "symlinks_skipped": 0,
                      "network_attempts": 0, "baseline_processes": len(self.baseline)}
        self.survivor_cmdlines = []
        os.makedirs(runs_root, exist_ok=True)
        os.chmod(runs_root, 0o755)
        parent = os.path.dirname(os.path.abspath(runs_root))
        if os.getuid() == 0 and os.path.isdir(parent):
            os.chmod(parent, 0o755)

    # ------------------------------------------------------------------ the wrapper
    def wrap(self, argv, cpu_s, filters=True, kill_filter=True):
        """argv as every program process starts (module text). `filters=False` / `kill_filter=False` leave out the two
        filter stages / only the killing one (the self-tests of the house filter)."""
        a = list(argv)
        if filters and kill_filter:
            a = netkill.netkill_argv(a)
        if filters:
            a = nonet.nonet_argv(a)
        a = ["prlimit", "--cpu=%d" % int(cpu_s), "--as=%d" % self.as_bytes, "--fsize=%d" % self.fsize_bytes,
             "--core=0", "--"] + a
        if self.acct is not None and os.getuid() == 0:
            a = ["setpriv", "--reuid=%d" % self.uid, "--regid=%d" % self.gid, "--clear-groups", "--inh-caps=-all",
                 "--bounding-set=-all", "--no-new-privs", "--"] + a
        return a

    def program_env(self, base):
        """The program's environment function: the cell's `base(home, tmp, lib)` plus `program_env_extra` (which may
        not replace a variable of the cell's)."""
        extra = dict(self.env_extra)

        def env(home, tmp, lib):
            e = base(home, tmp, lib)
            clash = sorted(set(e) & set(extra))
            if clash:
                raise TaskError("program_env_extra would replace the cell's %s" % ", ".join(clash))
            e.update(extra)
            return e
        return env

    # ------------------------------------------------------------------ the run directory
    def prepare(self, program_src, carkit_dir):
        """A fresh run directory under an opaque tag (cellkit.runner.prepare_run_dir)."""
        tag = "run-" + secrets.token_hex(8)
        root = os.path.join(self.runs_root, tag)
        owner = (self.uid, self.gid) if self.acct is not None and os.getuid() == 0 else None
        dirs = runner.prepare_run_dir(root, program_src, {"carkit": carkit_dir}, owner=owner)
        tests = os.path.join(dirs["lib"], "carkit", "tests")
        if os.path.isdir(tests):
            shutil.rmtree(tests)
        dirs["tag"] = tag
        self.stats["symlinks_skipped"] = max(self.stats["symlinks_skipped"], int(dirs["skipped"]))
        return dirs

    def snapshot(self, dirs):
        return {r: runner.snapshot_tree(r, (dirs["root"],)) for r in STICKY_DIRS}

    # ------------------------------------------------------------------ after a run
    def live(self):
        if self.uid is None:
            return []
        return [p for p in lockdown.live_uid_processes(self.uid) if p not in self.baseline]

    def survivors(self):
        """Processes of the runner account (not of the baseline) still alive `survivor_wait_s` after the program's
        group was killed: [(pid, cmdline)]."""
        deadline = time.time() + self.survivor_wait_s
        while True:
            pids = self.live()
            if not pids or time.time() >= deadline:
                break
            time.sleep(0.05)
        out = []
        for pid in pids:
            try:
                with open("/proc/%d/cmdline" % pid, "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace")[:200]
            except OSError:
                cmd = "?"
            out.append((pid, cmd))
        return out

    def cleanup(self, dirs, before):
        """Survivors, then freeze-and-kill the account, sweep, remove the run directory. Returns this run's counts."""
        surv = self.survivors()
        killed = left = 0
        if self.uid is not None:
            killed, left = lockdown.stop_and_kill_uid(self.uid, grace_s=8.0)
        stray = 0
        if before is not None and self.uid is not None:
            for r in STICKY_DIRS:
                after = runner.snapshot_tree(r, (dirs["root"],))
                new, modified = runner.tree_changes(before.get(r, {}), after, self.uid)
                stray += len(new) + len(modified)
                runner.sweep_new_entries(r, before.get(r, {}), after, self.uid, (dirs["root"],))
        shutil.rmtree(dirs["root"], ignore_errors=True)
        self.stats["runs"] += 1
        self.stats["survivors"] += len(surv)
        self.stats["killed"] += int(killed)
        self.stats["left"] += int(left)
        self.stats["stray_files"] += stray
        self.survivor_cmdlines += [c for _p, c in surv][:5]
        return {"survivors": len(surv), "survivor_cmdlines": [c for _p, c in surv][:5], "killed": int(killed),
                "left": int(left), "stray_files": stray}

    def note_exit(self, returncode):
        """Record a network attempt when the process ended by the killing filter. Returns True then."""
        if netkill.attempted(returncode):
            self.stats["network_attempts"] += 1
            return True
        return False

    # ------------------------------------------------------------------ the import check
    def import_check(self, program_src, carkit_dir, child_env, wall_s, cpu_s):
        """Import the submission's `drive` once under the regime (no wire, no cell): (ok, why, record)."""
        dirs = self.prepare(program_src, carkit_dir)
        before = self.snapshot(dirs)
        code = ("import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); import drive; "
                "assert callable(getattr(drive, 'solve', None)), 'drive.solve is not callable'")
        argv = self.wrap([sys.executable, "-B", "-E", "-s", "-c", code, dirs["lib"], dirs["program"]], cpu_s=cpu_s)
        env = child_env(dirs["home"], dirs["tmp"], dirs["lib"])
        rec = {"returncode": None, "network_attempt": False, "timed_out": False}
        t0 = time.time()
        try:
            r = subprocess.run(argv, cwd=dirs["program"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=wall_s, start_new_session=True)
            rec["returncode"] = r.returncode
            tail = (r.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            why = "ok" if r.returncode == 0 else "importing drive.py failed (exit %s): %s" % (
                r.returncode, (tail[-1] if tail else "?")[:300])
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            rec["timed_out"] = True
            ok, why = False, "importing drive.py took more than %.0f s" % wall_s
        finally:
            rec.update(self.cleanup(dirs, before))
        rec["wall_s"] = round(time.time() - t0, 3)
        if self.note_exit(rec["returncode"]):
            rec["network_attempt"] = True
            ok, why = False, "importing drive.py ended by the no-network filter (a socket other than AF_UNIX)"
        return ok, why, rec

    # ------------------------------------------------------------------ self-tests (a failure is a task error)
    def selftest(self, child_env, stage_dir):
        """The regime holds here: the account, no new privileges, two seccomp filters, no capabilities, the limits,
        the stage closed to the program; the house filter alone refuses an inet socket; the killing filter kills; no
        new process can be made and a thread can; the documented tools import in the program's
        environment (`child_env`, which must be the program's: `program_env`). Returns a dict of findings; raises
        TaskError when something does not hold."""
        work = os.path.join(self.runs_root, "selftest-" + secrets.token_hex(4))
        os.makedirs(work, exist_ok=True)
        os.chmod(work, 0o755)
        env = child_env(work, work, work)
        out = {}
        try:
            argv = self.wrap([sys.executable, "-I", "-B", "-c", _PROBE, stage_dir], cpu_s=60)
            r = subprocess.run(argv, capture_output=True, text=True, timeout=60, env=env, cwd="/")
            if r.returncode != 0:
                raise TaskError("the run regime's probe failed (exit %s): %s" % (r.returncode, (r.stderr or "").strip()[-300:]))
            p = json.loads(r.stdout.strip().splitlines()[-1])
            out["probe"] = p
            problems = []
            if os.getuid() == 0:
                if p["uid"] != self.uid or p["euid"] != self.uid or p["gid"] != self.gid:
                    problems.append("the program does not run as %s" % self.name)
                if p["groups"] not in ([], [self.gid]):
                    problems.append("the program keeps supplementary groups %s" % p["groups"])
                if p["stage"] != "denied":
                    problems.append("the program can list the stage folder (%s)" % p["stage"])
            if str(p["cap_eff"]).strip("0") != "":
                problems.append("the program has effective capabilities %s" % p["cap_eff"])
            if p["no_new_privs"] != "1":
                problems.append("no_new_privs is not set")
            if p["seccomp"] != "2" or int(p["seccomp_filters"] or 0) < 2:
                problems.append("the two seccomp filters are not installed (Seccomp %s, filters %s)" % (
                    p["seccomp"], p["seccomp_filters"]))
            if p["rlimit_as"] != self.as_bytes or p["rlimit_core"] != 0 or p["rlimit_cpu"] != 60:
                problems.append("the limits are not in place (%s, %s, %s)" % (p["rlimit_as"], p["rlimit_core"], p["rlimit_cpu"]))
            # the house filter on its own (its own self-test expects EPERM, so without the killing stage)
            ok, why = nonet.selftest(wrap=self._wrap_house, env=env)
            out["nonet"] = why
            if not ok:
                problems.append("the house no-network filter: " + why)
            # the whole chain as a program runs: account, limits, house filter, killing filter
            ok, why = netkill.selftest(wrap=lambda a: self.wrap(a, cpu_s=60, filters=True, kill_filter=False), env=env)
            out["netkill"] = why
            if not ok:
                problems.append("the killing filter: " + why)
            # no new process under the whole chain, a thread still works
            ok, why = netkill.proc_selftest(wrap=lambda a: self.wrap(a, cpu_s=60, filters=True, kill_filter=False), env=env)
            out["noproc"] = why
            if not ok:
                problems.append("the process refusal: " + why)
            # the documented tools import under the regime, in the program's environment
            argv = self.wrap([sys.executable, "-B", "-E", "-s", "-c", _TOOLS], cpu_s=120)
            r = subprocess.run(argv, capture_output=True, text=True, timeout=300, env=env, cwd="/")
            out["tools"] = "ok" if r.returncode == 0 and "tools-ok" in (r.stdout or "") else "FAILED (rc %s) %s" % (
                r.returncode, (r.stderr or "").strip()[-200:])
            if out["tools"] != "ok":
                problems.append("the documented tools do not import under the regime: " + out["tools"])
            if problems:
                raise TaskError("the run regime does not hold in this sandbox: " + "; ".join(problems))
        finally:
            if self.uid is not None:
                lockdown.stop_and_kill_uid(self.uid, grace_s=5.0)
            shutil.rmtree(work, ignore_errors=True)
        return out

    def _wrap_house(self, argv):
        """nonet.selftest hands over argv already behind the house filter: add the account and the limits only."""
        return self.wrap(argv, cpu_s=60, filters=False)

    def finish(self):
        """Remove the runs folder (every run directory is gone already)."""
        shutil.rmtree(self.runs_root, ignore_errors=True)
