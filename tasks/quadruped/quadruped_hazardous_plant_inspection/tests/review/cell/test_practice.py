"""Test of the practice ground (environment/practice_ground/practiced.py) and the payload's training/run_mission.py.

    MUJOCO_GL=osmesa python -I tests/review/cell/test_practice.py --hall H.tar.xz --site S.tar.xz --work DIR

A private copy of the practice ground (practiced.py and its byte copy of the cell) runs in DIR with a test
sites.json (the given archives as 'hall' and 't1', an allowance of 3 missions, 2 at a time) and its own socket,
status, debrief and state paths. Checked:
  1. run_mission.py runs the example program in a mission on t1 and prints the debrief (per item, no total);
  2. the reset reply, one mission per connection, `forbidden` for the privileged calls and a second reset;
  3. two missions at a time: a third connection is told `busy`;
  4. the allowance: the fourth reset is refused (`practice_allowance_used`), and the count survives a restart;
  5. the hall: its simulated-time cap (900 s) and the true pose at 2 Hz in its debrief; the buildings' cap (1350 s since
     version 2.1; from the task's own practice sites.json) in the reset reply and describe();
  6. a package whose sha256 does not match sites.json is refused without using a mission.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True        # (no __pycache__ in the task tree from a review run)
TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK = os.path.dirname(TESTS)
PG = os.path.join(TASK, "environment", "practice_ground")
PAYLOAD = os.path.join(TASK, "environment", "setup", "payload")
sys.path.insert(0, os.path.join(PAYLOAD, "given", "lib"))

from plantcell.client import CellError, connect_practice  # noqa: E402


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def start(work, env):
    p = subprocess.Popen([sys.executable, "-s", os.path.join(work, "pg", "practiced.py")], env=env,
                         stdout=subprocess.DEVNULL, stderr=open(os.path.join(work, "daemon.err"), "ab"),
                         start_new_session=True)
    for _ in range(240):
        if os.path.exists(env["PLANTCELL_PRACTICE_SOCKET"]):
            break
        time.sleep(0.25)
    return p


def stop(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        pass
    p.wait()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    problems = []
    work = os.path.abspath(a.work)
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "pg", "sites"))
    shutil.copy(os.path.join(PG, "practiced.py"), os.path.join(work, "pg"))
    shutil.copytree(os.path.join(PG, "plantcell"), os.path.join(work, "pg", "plantcell"))
    shutil.copy(a.hall, os.path.join(work, "pg", "sites", "hall.tar.xz"))
    shutil.copy(a.site, os.path.join(work, "pg", "sites", "t1.tar.xz"))
    shutil.copy(a.site, os.path.join(work, "pg", "sites", "bad.tar.xz"))
    # the run settings of the task's own practice sites.json (the scored missions' time limit; the hall's own 900 s)
    with open(os.path.join(PG, "sites.json")) as f:
        real = json.load(f)
    building_limit = float(real["config"]["time"]["sim_time_limit_s"])
    hall_limit = float(next(s["sim_time_limit_s"] for s in real["sites"] if s["name"] == "hall"))
    if building_limit != 1350.0 or hall_limit != 900.0:          # version 2.1 (version 2.0: 2400)
        problems.append("practice sites.json: buildings %s s, hall %s s (expected 1350 / 900)" % (building_limit,
                                                                                               hall_limit))
    sites = {"rule": "test", "missions": 3, "open": 2, "program_time_cap_s": 1200.0, "silence_s": 180.0,
             "config": {"time": {"sim_time_limit_s": building_limit}},
             "sites": [{"name": "hall", "package": "sites/hall.tar.xz", "sha256": sha(a.hall), "kind": "hall",
                        "sim_time_limit_s": hall_limit, "about": "test hall"},
                       {"name": "t1", "package": "sites/t1.tar.xz", "sha256": sha(a.site), "kind": "practice",
                        "about": "test building"},
                       {"name": "bad", "package": "sites/bad.tar.xz", "sha256": "0" * 64, "kind": "practice",
                        "about": "a package that does not match"}]}
    json.dump(sites, open(os.path.join(work, "pg", "sites.json"), "w"), indent=1)
    run = os.path.join(work, "run")
    os.makedirs(run)
    env = dict(os.environ, PLANTCELL_PRACTICE_SOCKET=os.path.join(run, "practice.sock"),
               PLANTCELL_PRACTICE_STATUS=os.path.join(run, "practice_status.json"),
               PLANTCELL_PRACTICE_DEBRIEFS=os.path.join(run, "debriefs"),
               PLANTCELL_PRACTICE_STATE=os.path.join(work, "pg", "state"))
    sock, status = env["PLANTCELL_PRACTICE_SOCKET"], env["PLANTCELL_PRACTICE_STATUS"]
    d = start(work, env)
    try:
        # 6. a mismatching package: refused, no mission used
        r = connect_practice(path=sock)
        try:
            r.reset("bad")
            problems.append("a package with the wrong sha256 was opened")
        except CellError as e:
            if e.code != "unavailable":
                problems.append("bad package: %s" % e)
        r.close()
        st = json.load(open(status))
        if st["missions_used"] != 0:
            problems.append("the refused package used a mission")
        # 1. run_mission.py with the example program on t1
        sub = os.path.join(work, "submission")
        os.makedirs(sub)
        shutil.copy(os.path.join(PAYLOAD, "example", "patrol.py"), sub)
        t0 = time.time()
        out = subprocess.run([sys.executable, "-B", os.path.join(PAYLOAD, "training", "run_mission.py"), "--site", "t1",
                              "--program", sub, "--socket", sock, "--status", status,
                              "--debrief", os.path.join(work, "debrief_t1.json")],
                             capture_output=True, text=True, timeout=1800,
                             env=dict(os.environ, HOME=work, PYTHONDONTWRITEBYTECODE="1"))   # no bytecode in the payload
        wall1 = time.time() - t0
        if out.returncode != 0 or "debrief:" not in out.stdout:
            problems.append("run_mission.py failed: %s %s" % (out.returncode, (out.stderr or out.stdout)[-800:]))
        db = json.load(open(os.path.join(work, "debrief_t1.json")))
        if not db or len(db.get("items", [])) != 10:
            problems.append("debrief items: %s" % (None if not db else len(db.get("items", []))))
        txt = json.dumps(db).lower()
        for k in ('"score', '"total', '"rooms"', '"walls"'):
            if k in txt:
                problems.append("the debrief contains %s" % k)
        if "true_pose" in db:
            problems.append("a building debrief carries the true pose (hall only)")
        # 2. one connection: the reset reply, forbidden calls, a second reset
        r1 = connect_practice(path=sock)
        first = r1.reset("hall")
        if not (first.get("ok") and first.get("mission") == 2 and abs(first.get("sim_time_limit_s", 0) - 900.0) < 1e-9):
            problems.append("hall reset reply %s" % first)
        for fn, code in ((lambda: r1._call("reference_run"), "forbidden"),
                         (lambda: r1._call("get_state_privileged"), "forbidden"),
                         (lambda: r1.reset("t1"), "one_mission_per_connection")):
            try:
                fn()
                problems.append("expected %s" % code)
            except CellError as e:
                if e.code != code:
                    problems.append("expected %s, got %s" % (code, e.code))
        # 3. a second mission at the same time, then a third connection: busy
        r2 = connect_practice(path=sock)
        second = r2.reset("t1")
        if abs(float(second.get("sim_time_limit_s", 0)) - building_limit) > 1e-9:
            problems.append("t1 reset reply %s (expected the buildings' %s s)" % (second, building_limit))
        if abs(float(r2.describe()["time"]["sim_time_limit_s"]) - building_limit) > 1e-9:
            problems.append("t1 describe() quotes another time limit")
        r3 = connect_practice(path=sock)
        try:
            r3.reset("t1")
            problems.append("a third concurrent mission was opened")
        except CellError as e:
            if e.code != "busy":
                problems.append("third connection: %s" % e)
        r3.close()
        # 5. the hall: walk a little, finish, the debrief arrives as the last line with the true pose at 2 Hz
        r1.set_mode("walk")
        r1.walk(0.2, 0.0, 0.0)
        r1.step(300)
        r1.finish()
        try:
            r1.get_state()                    # the mission is over: the next line is the debrief
        except CellError as e:
            if e.code not in ("mission_over", "closed"):
                problems.append("after finish: %s" % e)
        hall_db = r1.debrief
        if not hall_db or "true_pose" not in hall_db or abs(hall_db["true_pose"]["hz"] - 2.0) > 1e-9:
            problems.append("hall debrief without the true pose at 2 Hz")
        elif len(hall_db["true_pose"]["samples"]) < 6:
            problems.append("hall true pose: %d samples" % len(hall_db["true_pose"]["samples"]))
        if hall_db and hall_db["budget_use"]["sim_time_limit_s"] != 900.0:
            problems.append("hall sim time limit %s" % hall_db["budget_use"]["sim_time_limit_s"])
        r1.close()
        r2.finish()
        r2.close()
        time.sleep(1.0)
        # 4. the allowance (3 used) and a restart
        st = json.load(open(status))
        if st["missions_used"] != 3 or st["missions_left"] != 0:
            problems.append("status after 3 missions: %s" % {k: st[k] for k in ("missions_used", "missions_left")})
        stop(d)
        os.unlink(sock)
        d = start(work, env)
        st = json.load(open(status))
        if st["missions_used"] != 3:
            problems.append("the allowance did not survive the restart: %s" % st["missions_used"])
        r4 = connect_practice(path=sock)
        try:
            r4.reset("t1")
            problems.append("a fourth mission was opened")
        except CellError as e:
            if e.code != "practice_allowance_used":
                problems.append("fourth mission: %s" % e)
        r4.close()
        n_db = len(os.listdir(env["PLANTCELL_PRACTICE_DEBRIEFS"]))
        if n_db != 3:
            problems.append("%d debrief files (expected 3)" % n_db)
    finally:
        stop(d)
    print(out.stdout[-1500:])
    print(json.dumps({"example_mission_wall_s": round(wall1, 1), "debrief_files": n_db}, indent=1))
    for p in problems:
        print("PROBLEM", p)
    print("practice ground: %s (%d problems)" % ("OK" if not problems else "FAIL", len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
