"""Every end of a practice mission gives the team the reason and the debrief of what happened until then
(`environment/practice_ground/practiced.py`, `environment/setup/payload/training/run_mission.py`).

    MUJOCO_GL=osmesa python -I tests/review/cell/test_practice_endings.py --hall H.tar.xz --work DIR

A private practice ground with short limits (the hall at 900 s and a copy at 4 s of simulated time, 15 s of program
time, 8 s of silence, a snapshot of the running debrief every 2 s); for the endings the cell decides (the dose rating,
the robot lost) and for the service's own failures, a wrapper of the service injects them after the program's 12th
`step` (`INJECT`). One mission per ending, through run_mission.py; each must end with the tool's exit status 0, its
`end:` line naming the practice ground's reason, a debrief (a dict with `ended_by` and the items), no process of the
program left, within the time the ending needs:

  finish             the example program                                    reason finish
  time_limit         steps on the 4 s copy until the end, then 50 more      reason time_limit
  dose_limit         (injected: the world ends the mission)                 reason dose_limit
  robot_lost         (injected)                                             reason robot_lost
  program_time       a busy loop after one step                             reason program_time_cap
  silence            a sleep after one step                                 reason silence
  protocol           a line that is not a request                           reason protocol (...)
  program_exit       os._exit(3) after one step                             reason eof
  cpu                the program sends itself SIGXCPU (as its CPU limit     reason eof; the tool says the program was
                     would)                                                 killed by signal 24
  service_error      (injected: an exception in the service)                reason practice_ground_failure (...)
  service_killed     (injected: the mission's process is killed)            reason practice_ground_failure, the
                                                                            debrief of the last snapshot
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True        # (no __pycache__ in the task tree from a review run)
TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK = os.path.dirname(TESTS)
PG = os.path.join(TASK, "environment", "practice_ground")
PAYLOAD = os.path.join(TASK, "environment", "setup", "payload")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_practice import sha, stop  # noqa: E402

WRAPPER = r'''
import os, sys, signal
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import practiced
from plantcell import server as S
MODE = os.environ.get("INJECT", "")
AFTER = int(os.environ.get("INJECT_AFTER", "12"))
orig = S.PlantCellServer.handle
def handle(self, method, params):
    res = orig(self, method, params)
    if method == "step" and MODE:
        self._inj = getattr(self, "_inj", 0) + 1
        if self._inj == AFTER:
            if MODE in ("dose_limit", "robot_lost"):
                self.world.end(MODE)
            elif MODE == "service_killed":
                os.kill(os.getpid(), signal.SIGKILL)
            elif MODE == "service_error":
                raise RuntimeError("injected")
    return res
S.PlantCellServer.handle = handle
practiced.main()
'''

HEAD = "import os, sys, time, signal\n\ndef solve(robot):\n    open(%(pids)r, 'w').write(str(os.getpid()))\n"
STEPPER = HEAD + "    for _ in range(40):\n        robot.step(10)\n"
PROGRAMS = {
    "finish": HEAD + "    robot.step(10)\n    robot.finish()\n",
    "time_limit": HEAD + "    for _ in range(200):\n        robot.step(100)\n",
    "dose_limit": STEPPER,
    "robot_lost": STEPPER,
    # busy between cheap requests (never 8 s silent): the program-time budget of 15 s runs out
    "program_time": HEAD + "    robot.step(1)\n    while True:\n        t = time.time()\n        while time.time() - t < 3.0:\n"
                           "            pass\n        robot.get_state()\n",
    "silence": HEAD + "    robot.step(1)\n    time.sleep(10 ** 6)\n",
    "protocol": HEAD + "    robot.step(1)\n    sys.__stdout__.write('this is not a request\\n')\n    sys.__stdout__.flush()\n"
                       "    robot.step(1)\n",
    "program_exit": HEAD + "    robot.step(1)\n    os._exit(3)\n",
    "cpu": HEAD + "    robot.step(1)\n    os.kill(os.getpid(), signal.SIGXCPU)\n",
    "service_error": STEPPER,
    "service_killed": STEPPER,
}
EXPECT = {"finish": "finish", "time_limit": "time_limit", "dose_limit": "dose_limit", "robot_lost": "robot_lost",
          "program_time": "program_time_cap", "silence": "silence", "protocol": "protocol", "program_exit": "eof",
          "cpu": "eof", "service_error": "practice_ground_failure", "service_killed": "practice_ground_failure"}
SITE = {"time_limit": "hall_short"}
INJECT = {"dose_limit": "dose_limit", "robot_lost": "robot_lost", "service_error": "service_error",
          "service_killed": "service_killed"}


def start_ground(work, env, inject):
    """The private practice ground through the wrapper (the service itself, with an injection when asked)."""
    import signal  # noqa: F401
    e = dict(env, INJECT=inject or "", INJECT_AFTER="12")
    p = subprocess.Popen([sys.executable, "-s", os.path.join(work, "pg", "practiced_inject.py")], env=e,
                         stdout=subprocess.DEVNULL, stderr=open(os.path.join(work, "daemon.err"), "ab"),
                         start_new_session=True)
    for _ in range(240):
        if os.path.exists(env["PLANTCELL_PRACTICE_SOCKET"]):
            break
        time.sleep(0.25)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default=None, help="comma-separated endings")
    a = ap.parse_args(argv)
    problems = []
    work = os.path.abspath(a.work)
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "pg", "sites"))
    shutil.copy(os.path.join(PG, "practiced.py"), os.path.join(work, "pg"))
    shutil.copytree(os.path.join(PG, "plantcell"), os.path.join(work, "pg", "plantcell"))
    shutil.copy(a.hall, os.path.join(work, "pg", "sites", "hall.tar.xz"))
    with open(os.path.join(work, "pg", "practiced_inject.py"), "w") as f:
        f.write(WRAPPER)
    sites = {"rule": "test", "missions": 30, "open": 2, "program_time_cap_s": 15.0, "silence_s": 8.0,
             "config": {"time": {"sim_time_limit_s": 2400.0}},
             "sites": [{"name": "hall", "package": "sites/hall.tar.xz", "sha256": sha(a.hall), "kind": "hall",
                        "sim_time_limit_s": 900.0, "about": "test hall"},
                       {"name": "hall_short", "package": "sites/hall.tar.xz", "sha256": sha(a.hall), "kind": "hall",
                        "sim_time_limit_s": 4.0, "about": "test hall, 4 s"}]}
    json.dump(sites, open(os.path.join(work, "pg", "sites.json"), "w"), indent=1)
    run = os.path.join(work, "run")
    os.makedirs(run)
    env = dict(os.environ, PLANTCELL_PRACTICE_SOCKET=os.path.join(run, "practice.sock"),
               PLANTCELL_PRACTICE_STATUS=os.path.join(run, "practice_status.json"),
               PLANTCELL_PRACTICE_DEBRIEFS=os.path.join(run, "debriefs"),
               PLANTCELL_PRACTICE_STATE=os.path.join(work, "pg", "state"),
               PLANTCELL_PRACTICE_SNAPSHOT_S="2")
    sock, status = env["PLANTCELL_PRACTICE_SOCKET"], env["PLANTCELL_PRACTICE_STATUS"]
    names = [n for n in PROGRAMS if not a.only or n in a.only.split(",")]
    out = {}
    current_inject = None
    d = None
    try:
        for name in names:
            inject = INJECT.get(name)
            if d is None or inject != current_inject:
                if d is not None:
                    stop(d)
                    try:
                        os.unlink(sock)
                    except OSError:
                        pass
                d = start_ground(work, env, inject)
                current_inject = inject
            sub = os.path.join(work, "sub_" + name)
            os.makedirs(sub)
            pids_p = os.path.join(work, name + ".pid")
            with open(os.path.join(sub, "patrol.py"), "w") as f:
                f.write(PROGRAMS[name] % {"pids": pids_p})
            t0 = time.time()
            try:
                r = subprocess.run([sys.executable, "-B", os.path.join(PAYLOAD, "training", "run_mission.py"),
                                    "--site", SITE.get(name, "hall"), "--program", sub, "--socket", sock,
                                    "--status", status, "--debrief", os.path.join(work, name + "_debrief.json")],
                                   capture_output=True, text=True, timeout=300,
                                   env=dict(os.environ, HOME=work, PYTHONDONTWRITEBYTECODE="1"))
                rc, txt = r.returncode, r.stdout + r.stderr
            except subprocess.TimeoutExpired as e:
                rc = "timeout"
                txt = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            wall = time.time() - t0
            try:
                db = json.load(open(os.path.join(work, name + "_debrief.json")))
            except (OSError, ValueError):
                db = None
            end = [ln for ln in txt.splitlines() if ln.startswith("end:")]
            ended_by = db.get("ended_by") if isinstance(db, dict) else None
            if name in ("dose_limit", "robot_lost") and isinstance(db, dict):
                ended_by = db.get("ended")        # the cell ended the mission; the program then finished
            items = len(db.get("items", [])) if isinstance(db, dict) else 0
            time.sleep(0.5)
            try:
                pid = int(open(pids_p).read())
            except (OSError, ValueError):
                pid = None
            left = [pid] if pid is not None and os.path.exists("/proc/%d" % pid) and \
                open("/proc/%d/stat" % pid).read().split(") ", 1)[1].split()[0] != "Z" else []
            row = {"rc": rc, "wall_s": round(wall, 1), "ended_by": ended_by, "items": items,
                   "snapshot": (db or {}).get("snapshot") if isinstance(db, dict) else None,
                   "failure": (db or {}).get("failure") if isinstance(db, dict) else None,
                   "end": end[0][:300] if end else None, "processes_left": left}
            out[name] = row
            want = EXPECT[name]
            ok = (rc == 0 and isinstance(ended_by, str) and ended_by.startswith(want) and items > 0
                  and end and want.split(" ")[0] in end[0] and not left and pid is not None)
            if name == "cpu" and not (end and "signal 24" in end[0]):
                ok = False
            if name == "service_killed" and not (row["failure"] and row["snapshot"]):
                ok = False
            if not ok:
                problems.append("%s: %s (expected the reason %s and a debrief with the items)" % (name, row, want))
            for pid in left:
                try:
                    os.kill(int(pid), 9)
                except (OSError, ValueError):
                    pass
    finally:
        if d is not None:
            stop(d)
    print(json.dumps(out, indent=1))
    for p in problems:
        print("PROBLEM", p)
    print("practice endings: %s" % ("OK" if not problems else "FAIL (%d problems)" % len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
