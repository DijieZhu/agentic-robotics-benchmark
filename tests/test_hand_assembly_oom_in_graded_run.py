"""dexterous_hand_full_assembly_16dof_panda: in a graded run every process of the program is the memory killer's first
choice.

The verifier is one memory cgroup that holds the grader (the cell and its render session) and the submitted program.
verify.py starts every process of the program (the import check's and each episode's launcher) with oom_score_adj
1000 (`choom -n 1000 --` after runuser and prlimit; inherited by its children) and keeps the order by KILLING (the root
sampler SIGKILLs any process of the program found below 1000), so that a program that fills the memory is killed and
the grader writes its verdict. The check grades a small program through the real path (tests/test.sh as root in the
task image, --network none, one episode): at import, in its episode and in a child it forks the program reads its own
value; the child writes 0 to its own value and is then SIGKILLed by the sampler within one pass (it ends by signal 9,
counted in `runner_oom_adj_killed`), the parent lives, steps again and the episode is scored (contract_ok 1); the
program's environment holds exactly the documented variables (no `PWD`); a root observer in the container reads the
value of every interpreter process of the program (argv[0] the interpreter; the launch chain's runuser / prlimit /
choom are not asserted on: they run as the program's account for a few milliseconds before choom sets the value) and of
the grader. A second test launches the real chain (runuser -> prlimit -> choom -> python) twenty times under the
sampler and checks that none of the launches is killed (the chain's first milliseconds are spared) and that each runs.
Needs Docker and the task image named by ARB_TASK_IMAGE (built from environment/), with the kit's local copy
tests/assets/gt.tar.xz in the task folder; skipped otherwise. The container is named
`<ARB_TEST_CONTAINER_PREFIX>_<pid>` (default prefix `hand03_oomtest`).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

SLUG = "dexterous_hand_full_assembly_16dof_panda"


def _task_dir():
    env = os.environ.get("ARB_TASK_DIR")
    here = Path(__file__).resolve()
    for cand in ([Path(env)] if env else []) + [here.parents[1] / "tasks" / "hand" / SLUG, here.parents[1].parent / SLUG]:
        if (cand / "tests" / "verify.py").is_file():
            return cand
    raise unittest.SkipTest("task folder %s not found" % SLUG)


TASK = _task_dir()
TESTS = TASK / "tests"
IMAGE = os.environ.get("ARB_TASK_IMAGE", "")
PREFIX = os.environ.get("ARB_TEST_CONTAINER_PREFIX") or "hand03_oomtest"
ENV_KEYS = ["ARB_GIVEN_DIR", "HOME", "LANG", "LP_NUM_THREADS", "MKL_NUM_THREADS", "MPLBACKEND", "MUJOCO_GL", "NPY_DISABLE_CPU_FEATURES",
            "OMP_NUM_THREADS", "OPENBLAS_CORETYPE", "OPENBLAS_L2_SIZE", "OPENBLAS_NUM_THREADS", "PATH", "PYOPENGL_PLATFORM",
            "PYTHONPATH", "TEMP", "TMP", "TMPDIR"]

PROGRAM = '''import json
import os
import sys
import time


def _adj(pid="self"):
    try:
        with open("/proc/%s/oom_score_adj" % pid) as fh:
            return fh.read().strip()
    except OSError:
        return "?"


print("[probe] import pid %d oom_score_adj %s" % (os.getpid(), _adj()), file=sys.stderr, flush=True)
time.sleep(3)


def solve(cell):
    print("[probe] solve pid %d oom_score_adj %s" % (os.getpid(), _adj()), file=sys.stderr, flush=True)
    print("[probe] env " + json.dumps(sorted(os.environ)), file=sys.stderr, flush=True)
    pid = os.fork()
    if pid == 0:
        start = _adj()
        with open("/proc/self/oom_score_adj", "w") as fh:
            fh.write("0")                       # undo the order: a legitimate program never does this
        print("[probe] child pid %d started at %s wrote 0 read %s" % (os.getpid(), start, _adj()), file=sys.stderr, flush=True)
        while True:                             # wait to be SIGKILLed by the sampler (next pass, 0.5 s)
            time.sleep(0.05)
        os._exit(0)
    _, st = os.waitpid(pid, 0)
    print("[probe] child ended: %s" % (("signal %d" % os.WTERMSIG(st)) if os.WIFSIGNALED(st) else ("exit %d" % os.WEXITSTATUS(st))),
          file=sys.stderr, flush=True)
    for _ in range(3):
        cell.step(100)
'''

OBSERVER = r'''
while [ ! -f /rec/test_rc.txt ]; do
  for p in $(ps -e -o pid= 2>/dev/null); do
    a=$(cat /proc/$p/oom_score_adj 2>/dev/null) || continue
    u=$(stat -c %u /proc/$p 2>/dev/null) || continue
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-200)
    echo "uid $u pid $p oom_score_adj $a cmd $c"
  done
  sleep 0.3
done | awk '!seen[$0]++ { print $0; fflush() }' > /rec/observer.log
'''


def _docker_ok():
    if not IMAGE:
        return "set ARB_TASK_IMAGE to the task image (built from environment/) to run the graded run"
    if not shutil.which("docker"):
        return "no docker command: no container can be started here"
    r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return "image %s not found: no container can be started for it here" % IMAGE
    if not (TESTS / "assets" / "gt.tar.xz").is_file():
        return "the kit's local copy tests/assets/gt.tar.xz is not in the task folder"
    return None


class TestProgramIsTheMemoryKillersFirstChoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        why = _docker_ok()
        if why:
            raise unittest.SkipTest(why)

    def test_oom_score_adj_in_a_graded_run(self):
        tmp = tempfile.mkdtemp(prefix="hand_assembly_oom_", dir=os.environ.get("ARB_TEST_TMP") or None)
        name = "%s_%d" % (PREFIX, os.getpid())
        try:
            stage, sub, rec = (os.path.join(tmp, d) for d in ("stage", "sub", "rec"))
            # the submission first, then the stage: the grader's late-write gate dates the agent phase's end by the stage
            os.makedirs(sub)
            os.makedirs(rec)
            with open(os.path.join(sub, "assemble.py"), "w") as fh:
                fh.write(PROGRAM)
            time.sleep(2)
            shutil.copytree(str(TESTS), stage, ignore=shutil.ignore_patterns("__pycache__"))
            cfgp = os.path.join(stage, "grader_config.json")
            cfg = json.load(open(cfgp))
            cfg["episodes"] = 1
            json.dump(cfg, open(cfgp, "w"), indent=1)
            script = ("(%s) &\ncd /opt/ale/tests && bash test.sh 2> /rec/grader_stderr.log; echo $? > /rec/test_rc.txt; sleep 1; "
                      "chown -R %d:%d /opt/ale/tests /home/user/submission /rec; chmod -R u+rwX /opt/ale/tests /rec"
                      % (OBSERVER, os.getuid(), os.getgid()))
            r = subprocess.run(["docker", "run", "--rm", "--name", name, "--cpus", "4", "--memory", "8g", "--network", "none",
                                "-u", "root", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "ARB_RECORD_ROOT=/rec",
                                "-v", "%s:/opt/ale/tests" % stage, "-v", "%s:/home/user/submission" % sub, "-v", "%s:/rec" % rec,
                                IMAGE, "bash", "-c", script], capture_output=True, text=True, timeout=1800)
            err = open(os.path.join(rec, "grader_stderr.log")).read() if os.path.isfile(os.path.join(rec, "grader_stderr.log")) else ""
            self.assertEqual(open(os.path.join(rec, "test_rc.txt")).read().strip(), "0", (r.stdout + r.stderr + err)[-4000:])
            verdict = json.load(open(os.path.join(stage, "verdict.local.json")))
            self.assertEqual(verdict["metrics"]["contract_ok"], 1.0, err[-3000:])
            obs = open(os.path.join(rec, "observer.log")).read().splitlines()
            # the interpreter's processes only (argv[0]); the forked child's own write of 0 is excluded by its pid
            prog = [l for l in obs if l.startswith("uid 65534 ") and re.search(r" cmd \S*python[0-9.]* ", l)]
            imp = [l for l in prog if "import_module('assemble" in l]
            launch = [l for l in prog if "client import main" in l]
            self.assertTrue(imp, "the import check's process was not seen:\n" + "\n".join(obs[-40:]))
            self.assertTrue(launch, "the episode's launcher was not seen:\n" + "\n".join(obs[-40:]))
            grader = [l for l in obs if l.startswith("uid 0 ") and "verify.py" in l]
            self.assertTrue(grader)
            print("\n[oom] grader: %s" % grader[0])
            logs = os.path.join(rec, "logs")
            text = "".join(open(os.path.join(logs, f)).read() for f in os.listdir(logs) if f.startswith("agent_"))
            self.assertRegex(text, r"\[probe\] solve pid \d+ oom_score_adj 1000")
            m = re.search(r"\[probe\] child pid (\d+) started at (\S+) wrote 0 read (\S+)", text)
            self.assertTrue(m, text[-2000:])
            self.assertEqual(m.group(2), "1000", "the forked child did not inherit the floor: " + m.group(0))
            self.assertRegex(text, r"\[probe\] child ended: signal 9")   # SIGKILLed by the sampler for lowering its value
            self.assertGreaterEqual(verdict["metrics"]["runner_oom_adj_killed"], 1.0)
            m = re.search(r"\[probe\] env (\[.*\])", text)
            self.assertTrue(m, text[-2000:])
            self.assertEqual(json.loads(m.group(1)), ENV_KEYS)
            m = re.search(r"memory killer: the grader's oom_score_adj (-?\d+)", err)
            self.assertTrue(m, err[-2000:])
            print("[oom] the grader's own value: %s (-500 where the container allows lowering it)" % m.group(1))
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=120)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_launch_chain_is_not_killed_over_twenty_launches(self):
        # the sampler (period 0.01 s, uid nobody) runs while the real chain runuser -> prlimit -> choom -> python is
        # launched twenty times; the chain's helpers run at the inherited value for a few ms before choom writes 1000,
        # and must be spared (never killed) so that every launch runs. A sampler that raised instead of killing never
        # killed the chain either, so this test also passes with one -- it guards the kill against a regression of the
        # sparing.
        driver = r'''
import os, sys, subprocess, threading, time
sys.path.insert(0, "/opt/ale/tests")
from arm.assembly_info import netaudit
uid = __import__("pwd").getpwnam("nobody").pw_uid
smp = netaudit.SocketSampler(uid=uid, period_s=0.01, oom_floor=1000)
rcs = []
with smp:
    for i in range(20):
        r = subprocess.run(["runuser", "-p", "-u", "nobody", "--", "prlimit", "--cpu=30", "--core=0", "--",
                            "choom", "-n", "1000", "--", sys.executable, "-I", "-B", "-c",
                            "import time,os; open('/proc/self/oom_score_adj').read(); time.sleep(0.2)"])
        rcs.append(r.returncode)
    time.sleep(0.3)
import json as _j; print("LAUNCH_RESULT " + _j.dumps({"rcs": rcs, "killed": smp.oom_killed, "samples": smp.samples}))
'''
        name = "%s_launch_%d" % (PREFIX, os.getpid())
        tmp = tempfile.mkdtemp(prefix="hand_assembly_launch_", dir=os.environ.get("ARB_TEST_TMP") or None)
        try:
            script = os.path.join(tmp, "driver.py")
            with open(script, "w") as fh:
                fh.write(driver)
            r = subprocess.run(["docker", "run", "--rm", "--name", name, "--cpus", "4", "--memory", "4g", "--network", "none",
                                "-u", "root", "-v", "%s:/driver.py" % script, "-v", "%s:/opt/ale/tests" % TESTS,
                                IMAGE, "python3", "-I", "-B", "/driver.py"],
                               capture_output=True, text=True, timeout=600)
            line = [l for l in r.stdout.splitlines() if l.startswith("LAUNCH_RESULT ")]
            self.assertTrue(line, "driver gave no result (rc %s)\n%s" % (r.returncode, (r.stdout + r.stderr)[-3000:]))
            res = json.loads(line[-1][len("LAUNCH_RESULT "):])
            print("\n[launch] %s" % res)
            self.assertEqual(res["rcs"], [0] * 20, "a launch was killed or failed: %s" % res["rcs"])
            self.assertEqual(res["killed"], 0, "the sampler killed the launch chain (%d kills)" % res["killed"])
            self.assertGreater(res["samples"], 20, "the sampler barely sampled: %s" % res["samples"])
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=120)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
