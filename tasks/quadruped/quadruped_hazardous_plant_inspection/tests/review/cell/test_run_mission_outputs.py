"""`training/run_mission.py` opens its output files before it reserves a mission (an earlier
check: with `--transcript` in a folder that did not exist, version 1's tool reserved the mission and
started the program, then raised FileNotFoundError, which cost a mission and left the run folder and the program
behind).

    MUJOCO_GL=osmesa python -I tests/review/cell/test_run_mission_outputs.py --hall H.tar.xz --work DIR

A private practice ground (as test_practice.py), a program that takes one step and finishes, through the payload's
run_mission.py:
  unwritable   --transcript under a path whose parent is a regular file: the tool exits non-zero saying no mission
               was used, the practice ground's `missions_used` is unchanged, no practice_run_* folder is left in TMPDIR
  new_folder   --transcript and --debrief in folders that do not exist yet: the folders are made, the mission runs
               (one mission used), the transcript holds the mission's requests and the debrief file the debrief, no
               practice_run_* folder is left
Fails on version 1's run_mission.py (unwritable: a mission used, FileNotFoundError, the run folder left; new_folder:
FileNotFoundError after reset); passes after.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True        # (no __pycache__ in the task tree from a review run)
TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK = os.path.dirname(TESTS)
PG = os.path.join(TASK, "environment", "practice_ground")
PAYLOAD = os.path.join(TASK, "environment", "setup", "payload")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_practice import sha, start, stop  # noqa: E402

PROGRAM = "def solve(robot):\n    robot.step(1)\n    robot.finish()\n"


def used(status):
    with open(status) as f:
        return int(json.load(f)["missions_used"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--run-mission", default=os.path.join(PAYLOAD, "training", "run_mission.py"),
                    help=argparse.SUPPRESS)          # (another copy of the tool, e.g. version 1's, for the 'before')
    a = ap.parse_args(argv)
    problems = []
    work = os.path.abspath(a.work)
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "pg", "sites"))
    shutil.copy(os.path.join(PG, "practiced.py"), os.path.join(work, "pg"))
    shutil.copytree(os.path.join(PG, "plantcell"), os.path.join(work, "pg", "plantcell"))
    shutil.copy(a.hall, os.path.join(work, "pg", "sites", "hall.tar.xz"))
    sites = {"rule": "test", "missions": 8, "open": 2, "program_time_cap_s": 1200.0, "silence_s": 60.0,
             "config": {"time": {"sim_time_limit_s": 2400.0}},
             "sites": [{"name": "hall", "package": "sites/hall.tar.xz", "sha256": sha(a.hall), "kind": "hall",
                        "sim_time_limit_s": 900.0, "about": "test hall"}]}
    json.dump(sites, open(os.path.join(work, "pg", "sites.json"), "w"), indent=1)
    run = os.path.join(work, "run")
    os.makedirs(run)
    tmpdir = os.path.join(work, "tmp")
    os.makedirs(tmpdir)
    env = dict(os.environ, PLANTCELL_PRACTICE_SOCKET=os.path.join(run, "practice.sock"),
               PLANTCELL_PRACTICE_STATUS=os.path.join(run, "practice_status.json"),
               PLANTCELL_PRACTICE_DEBRIEFS=os.path.join(run, "debriefs"),
               PLANTCELL_PRACTICE_STATE=os.path.join(work, "pg", "state"))
    sock, status = env["PLANTCELL_PRACTICE_SOCKET"], env["PLANTCELL_PRACTICE_STATUS"]
    sub = os.path.join(work, "sub")
    os.makedirs(sub)
    with open(os.path.join(sub, "patrol.py"), "w") as f:
        f.write(PROGRAM)
    blocker = os.path.join(work, "a_file")
    with open(blocker, "w") as f:
        f.write("not a folder\n")
    d = start(work, env)
    out = {}

    def tool(extra):
        r = subprocess.run([sys.executable, "-B", a.run_mission, "--site", "hall", "--program", sub, "--socket", sock,
                            "--status", status] + extra, capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, HOME=work, TMPDIR=tmpdir, PYTHONDONTWRITEBYTECODE="1"))
        return r.returncode, r.stdout + r.stderr

    try:
        n0 = used(status)
        rc, txt = tool(["--transcript", os.path.join(blocker, "wire.jsonl")])
        n1 = used(status)
        left = glob.glob(os.path.join(tmpdir, "practice_run_*"))
        out["unwritable"] = {"rc": rc, "missions_used": [n0, n1], "run_folders_left": left, "said": txt.strip()[-200:]}
        if rc == 0 or n1 != n0 or left or "no practice mission was used" not in txt:
            problems.append("unwritable: %s" % out["unwritable"])
        tr_p = os.path.join(work, "new", "deeper", "wire.jsonl")
        db_p = os.path.join(work, "new2", "debrief.json")
        rc, txt = tool(["--transcript", tr_p, "--debrief", db_p])
        n2 = used(status)
        left = glob.glob(os.path.join(tmpdir, "practice_run_*"))
        lines = open(tr_p).read().splitlines() if os.path.isfile(tr_p) else []
        try:
            db = json.load(open(db_p))
        except (OSError, ValueError):
            db = None
        out["new_folder"] = {"rc": rc, "missions_used": [n1, n2], "transcript_lines": len(lines),
                             "debrief_keys": sorted(db)[:8] if isinstance(db, dict) else None,
                             "run_folders_left": left, "said": txt.strip()[-200:]}
        if rc != 0 or n2 != n1 + 1 or len(lines) < 2 or not isinstance(db, dict) or left:
            problems.append("new_folder: %s" % out["new_folder"])
    finally:
        stop(d)
    print(json.dumps(out, indent=1))
    for p in problems:
        print("PROBLEM", p)
    print("run_mission opens its outputs before the mission: %s" % ("OK" if not problems else
                                                                       "FAIL (%d problems)" % len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
