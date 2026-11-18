"""`training/run_mission.py` ends with the mission: when the practice ground ends the mission (silence)
or the program exits, the tool prints the debrief, stops the program's processes and returns.

    MUJOCO_GL=osmesa python -I tests/review/cell/test_run_mission.py --hall H.tar.xz --work DIR [--silence 20]

A private practice ground (as test_practice.py) whose silence limit is SILENCE s; four programs, one practice mission
each, through the payload's run_mission.py:
  busy      one step, then a busy loop                        -> ended_by silence, the tool back within SILENCE + 30 s
  sleeper   one step, then sleeps forever                     -> ended_by silence, likewise
  forker    one step, a child process that sleeps forever,    -> ended_by silence, likewise; the child is gone too
            then sleeps
  hard_exit one step, then os._exit(3)                        -> ended_by eof, the tool back within 30 s
Each time: exit status 0, the debrief printed (a dict with `ended_by`), every process of the program gone.
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
from test_practice import sha, start, stop  # noqa: E402

PROGRAMS = {
    "busy": "import os\n\ndef solve(robot):\n    open(%(pids)r, 'w').write(str(os.getpid()))\n    robot.step(1)\n"
            "    while True:\n        pass\n",
    "sleeper": "import os, time\n\ndef solve(robot):\n    open(%(pids)r, 'w').write(str(os.getpid()))\n    robot.step(1)\n"
               "    time.sleep(10 ** 6)\n",
    "forker": "import os, time\n\ndef solve(robot):\n    robot.step(1)\n    child = os.fork()\n    if child == 0:\n"
              "        time.sleep(10 ** 6)\n        os._exit(0)\n"
              "    open(%(pids)r, 'w').write('%%d %%d' %% (os.getpid(), child))\n    time.sleep(10 ** 6)\n",
    "hard_exit": "import os\n\ndef solve(robot):\n    open(%(pids)r, 'w').write(str(os.getpid()))\n    robot.step(1)\n"
                 "    os._exit(3)\n",
}
EXPECT = {"busy": "silence", "sleeper": "silence", "forker": "silence", "hard_exit": "eof"}


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                            # a zombie is gone for every purpose here
        with open("/proc/%d/stat" % pid) as f:
            return f.read().split(") ", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--silence", type=float, default=20.0)
    a = ap.parse_args(argv)
    problems = []
    work = os.path.abspath(a.work)
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "pg", "sites"))
    shutil.copy(os.path.join(PG, "practiced.py"), os.path.join(work, "pg"))
    shutil.copytree(os.path.join(PG, "plantcell"), os.path.join(work, "pg", "plantcell"))
    shutil.copy(a.hall, os.path.join(work, "pg", "sites", "hall.tar.xz"))
    sites = {"rule": "test", "missions": 8, "open": 2, "program_time_cap_s": 1200.0, "silence_s": a.silence,
             "config": {"time": {"sim_time_limit_s": 2400.0}},
             "sites": [{"name": "hall", "package": "sites/hall.tar.xz", "sha256": sha(a.hall), "kind": "hall",
                        "sim_time_limit_s": 900.0, "about": "test hall"}]}
    json.dump(sites, open(os.path.join(work, "pg", "sites.json"), "w"), indent=1)
    run = os.path.join(work, "run")
    os.makedirs(run)
    env = dict(os.environ, PLANTCELL_PRACTICE_SOCKET=os.path.join(run, "practice.sock"),
               PLANTCELL_PRACTICE_STATUS=os.path.join(run, "practice_status.json"),
               PLANTCELL_PRACTICE_DEBRIEFS=os.path.join(run, "debriefs"),
               PLANTCELL_PRACTICE_STATE=os.path.join(work, "pg", "state"))
    sock, status = env["PLANTCELL_PRACTICE_SOCKET"], env["PLANTCELL_PRACTICE_STATUS"]
    d = start(work, env)
    out = {}
    try:
        for name, src in PROGRAMS.items():
            sub = os.path.join(work, "sub_" + name)
            os.makedirs(sub)
            pids_p = os.path.join(work, name + ".pids")
            with open(os.path.join(sub, "patrol.py"), "w") as f:
                f.write(src % {"pids": pids_p})
            t0 = time.time()
            try:
                r = subprocess.run([sys.executable, "-B", os.path.join(PAYLOAD, "training", "run_mission.py"),
                                    "--site", "hall", "--program", sub, "--socket", sock, "--status", status,
                                    "--debrief", os.path.join(work, name + "_debrief.json")],
                                   capture_output=True, text=True, timeout=a.silence + 240.0,
                                   env=dict(os.environ, HOME=work, PYTHONDONTWRITEBYTECODE="1"))
                rc, txt = r.returncode, r.stdout
            except subprocess.TimeoutExpired as e:
                rc, txt = "timeout", (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) \
                    else (e.stdout or "")
            wall = time.time() - t0
            time.sleep(1.0)
            try:
                pids = [int(x) for x in open(pids_p).read().split()]
            except (OSError, ValueError):
                pids = []
            left = [p for p in pids if alive(p)]
            try:
                db = json.load(open(os.path.join(work, name + "_debrief.json")))
            except (OSError, ValueError):
                db = None
            ended = (db or {}).get("ended_by") if isinstance(db, dict) else None
            limit = (a.silence + 30.0) if EXPECT[name] == "silence" else 30.0
            row = {"rc": rc, "wall_s": round(wall, 1), "ended_by": ended, "pids": pids, "left": left,
                   "debrief_printed": "debrief: {" in (txt or ""), "end_line": [ln for ln in (txt or "").splitlines()
                                                                               if ln.startswith("end:")][:1]}
            out[name] = row
            if rc != 0 or not row["debrief_printed"] or ended != EXPECT[name] or wall > limit or left or not pids:
                problems.append("%s: %s (expected ended_by %s within %.0f s, no process left)" % (
                    name, row, EXPECT[name], limit))
            for p in left:                          # (the test cleans up whatever a failing tool left)
                try:
                    os.kill(p, 9)
                except OSError:
                    pass
    finally:
        stop(d)
    print(json.dumps(out, indent=1))
    for p in problems:
        print("PROBLEM", p)
    print("run_mission ends with the mission: %s" % ("OK" if not problems else "FAIL (%d problems)" % len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
