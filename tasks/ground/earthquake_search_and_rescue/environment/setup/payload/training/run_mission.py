#!/opt/venv/bin/python3
"""Training mission: runs YOUR program over the real wire in one mission on a site of the training ground (the test
lanes or a training rubble site, never a graded site) under the grader's checks and run regime, and prints the robot's
mission log.

    /opt/venv/bin/python3 ~/training/run_mission.py --site lanes                     # ~/submission on the test lanes
    /opt/venv/bin/python3 ~/training/run_mission.py --site 3 --keep ~/runs/m3 --transcript ~/runs/m3/wire.jsonl
    /opt/venv/bin/python3 ~/training/run_mission.py --list                           # the sites and missions left
    /opt/venv/bin/python3 ~/training/run_mission.py --check                          # the grader's checks only

What the grader does to a submission, this script does first, with the grader's own code (submission_checks.py, next
to this file), BEFORE it uses a training mission:
  * the submission checks 1-3 (rescue.py defines solve; no weight files, no training calls, the size cap; no forbidden
    import or call);
  * a READ-ONLY copy of the submission folder (regular files only) as the program's working directory, an empty
    writable HOME and TMPDIR, the grader's environment (nothing of yours except PATH), `python -B -E -s`, the
    no-network filter, the CPU cap (4 x the program-time budget);
  * check 4: `import rescue` within 120 s and 120 CPU-s in that regime.
A submission that fails a check is refused here as the grader refuses it (there it scores 0 on every site), and no
training mission is used. Two differences remain, neither of which a program can use: the program runs as you (the
grader runs it as an unprivileged account that cannot read your home), and the training ground's cell renders on this
sandbox's CPUs.

The program's stdin / stdout carry the wire to the training ground, its stderr comes out here. The training ground
applies the graded mission's budgets itself (simulated time, sensor frames, program time, silence) and ends the
mission when one runs out. `reset` is this script's (the program's own `reset` calls answer `forbidden`, as in the
grader). Each run uses one mission of the training ground's allowance (robot_manual.md section 9). Keep this script
and submission_checks.py out of your submission folder: they start processes, which the grader's audit refuses.
"""

import argparse
import json
import os
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GIVEN = os.path.join(os.path.dirname(HERE), "given")
LIB = os.path.join(GIVEN, "lib")
sys.path.insert(0, LIB)

sys.path.insert(0, HERE)

import submission_checks as checks  # noqa: E402
from rubblecell.client import TRAINING_STATUS, CellError, connect_training, launcher_argv  # noqa: E402

PROGRAM_TIME_S = 1200.0
CPU_FACTOR = 4.0
NOT_ON_THE_WIRE = ("reset", "truth", "get_state_privileged", "reference_run")


def copy_program(src, dst):
    """The grader's copy: regular files and directories only, then read-only for everyone."""
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d != "__pycache__" and not os.path.islink(os.path.join(dp, d))]
        rel = os.path.relpath(dp, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for fn in fns:
            p = os.path.join(dp, fn)
            if not os.path.islink(p) and stat.S_ISREG(os.lstat(p).st_mode):
                shutil.copyfile(p, os.path.join(dst, rel, fn))
    for dp, dns, fns in os.walk(dst, topdown=False):
        for fn in fns:
            os.chmod(os.path.join(dp, fn), 0o444)
        os.chmod(dp, 0o555)


def remove_run(run):
    for dp, dns, fns in os.walk(run):
        os.chmod(dp, 0o755)
    shutil.rmtree(run, ignore_errors=True)


def limits(cpu_s):
    def set_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return set_limits


def grader_checks(program):
    """(ok, why, run): the submission checks 1-3 on the folder, the grader's run folder, then check 4 (the import)."""
    ok, why, _net = checks.check_submission(program, os.path.join(LIB, "rubblecell"))
    if not ok:
        return False, why, None
    run = tempfile.mkdtemp(prefix="training_run_")
    prog = os.path.join(run, "program")
    copy_program(program, prog)
    home, tmp = os.path.join(run, "home"), os.path.join(run, "tmp")
    os.makedirs(home)
    os.makedirs(tmp)
    env = checks.run_environment(home, tmp, LIB, GIVEN)
    argv = checks.nonet_argv(checks.import_check_argv(sys.executable, LIB, prog))
    try:
        r = subprocess.run(argv, cwd=prog, env=env, capture_output=True, text=True, timeout=checks.IMPORT_WALL_S,
                           preexec_fn=limits(checks.IMPORT_CPU_S), start_new_session=True)
        if r.returncode != 0:
            return False, "importing rescue.py failed: " + ((r.stderr.strip().splitlines() or ["?"])[-1][:300]), run
    except subprocess.TimeoutExpired:
        return False, "importing rescue.py took more than %d s" % checks.IMPORT_WALL_S, run
    return True, "ok", run


def elide(obj):
    if isinstance(obj, dict):
        return {k: ("<%d bytes>" % len(v) if k.endswith("_b64") and isinstance(v, str) else elide(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [elide(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", help="a training site: its name or number (--list)")
    ap.add_argument("--program", default=os.path.expanduser("~/submission"))
    ap.add_argument("--keep", default=None, help="keep the run's HOME and TMPDIR in this folder")
    ap.add_argument("--transcript", default=None, help="write every request and reply (images elided) here, JSON lines")
    ap.add_argument("--list", action="store_true", help="print the training ground's sites and allowance")
    ap.add_argument("--check", action="store_true", help="run the grader's submission checks only (no mission is used)")
    args = ap.parse_args()
    if args.check:
        ok, why, run = grader_checks(os.path.abspath(os.path.expanduser(args.program)))
        if run:
            remove_run(run)
        print("submission checks: %s" % ("pass" if ok else "REFUSED (the grader would score every site 0): " + why))
        sys.exit(0 if ok else 1)
    if args.list or not args.site:
        try:
            print(json.dumps(json.load(open(TRAINING_STATUS)), indent=1))
        except (OSError, ValueError) as e:
            print("training ground status unreadable: %s" % e)
        return
    ok, why, run = grader_checks(os.path.abspath(os.path.expanduser(args.program)))
    if not ok:
        if run:
            remove_run(run)
        sys.exit("submission checks: REFUSED (the grader would score every site 0; no training mission was used): " + why)
    prog, home, tmp = os.path.join(run, "program"), os.path.join(run, "home"), os.path.join(run, "tmp")
    env = checks.run_environment(home, tmp, LIB, GIVEN)
    robot = connect_training()
    first = robot.reset(args.site)
    print("training mission %d on site %s; missions left after this one: %d" % (first["mission"], first["site"],
                                                                                first["missions_left"]), flush=True)
    cpu_s = int(CPU_FACTOR * PROGRAM_TIME_S)
    proc = subprocess.Popen(checks.nonet_argv(launcher_argv(sys.executable, LIB, prog)), cwd=prog, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1, preexec_fn=limits(cpu_s), start_new_session=True)
    tr = open(args.transcript, "w") if args.transcript else None
    t0 = time.time()
    seen = {"requests": 0, "frames": 0, "sweeps": 0, "thermal": 0, "steps": 0, "events": [], "sim_time": 0.0, "end": None}
    mission_log = None
    send, recv = robot._send, robot._recv
    while True:
        line = proc.stdout.readline()
        if not line:
            seen["end"] = seen["end"] or "the program exited"
            break
        try:
            req = json.loads(line)
            method = str(req.get("method"))
        except ValueError:
            seen["end"] = "protocol: a line that is not a request"
            break
        seen["requests"] += 1
        if method in NOT_ON_THE_WIRE:
            reply_obj = {"id": req.get("id"), "error": "forbidden"}
        else:
            try:
                send(line if line.endswith("\n") else line + "\n")
            except CellError:
                pass
            raw = recv()
            if not raw:
                seen["end"] = seen["end"] or "the training ground closed the mission"
                break
            reply_obj = json.loads(raw)
            if reply_obj.get("event") == "mission_over":
                seen["end"] = "the mission is over: %s" % reply_obj.get("reason")
                mission_log = reply_obj.get("mission_log")
                break
        res = reply_obj.get("result")
        if isinstance(res, dict):
            if res.get("ok") and method == "get_camera":
                seen["frames"] += 1
            if res.get("ok") and method == "get_lidar":
                seen["sweeps"] += 1
            if res.get("ok") and method == "get_thermal":
                seen["thermal"] += 1
            if method == "step" and "sim_time" in res:
                seen["steps"] += 1
            if "sim_time" in res:
                seen["sim_time"] = float(res["sim_time"])
            for e in res.get("events", []) if method == "step" else []:
                seen["events"].append((round(seen["sim_time"], 1), e))
        if tr is not None:
            tr.write(json.dumps({"t_wall": round(time.time() - t0, 3), "request": req, "reply": elide(reply_obj)}) + "\n")
        try:
            proc.stdin.write(json.dumps(reply_obj) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            seen["end"] = "the program stopped reading"
            break
        if method == "finish" and "result" in reply_obj:
            seen["end"] = "finish"
            break
    try:
        proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    # the grader kills what the program left running and counts it (a process left behind fails the submission)
    left = 0
    try:
        os.killpg(proc.pid, 0)
        left = 1
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if mission_log is None:
        # the mission's last line (the robot's mission log) follows the program's finish
        try:
            raw = recv()
            obj = json.loads(raw) if raw else {}
            mission_log = obj.get("mission_log")
        except (ValueError, CellError, OSError):
            mission_log = None
    robot.close()
    if tr is not None:
        tr.close()
    print("end: %s; program exit code %s; sim %.1f s; wall %.0f s; requests %d; camera frames %d; lidar sweeps %d; thermal %d" % (
        seen["end"], proc.returncode, seen["sim_time"], time.time() - t0, seen["requests"], seen["frames"], seen["sweeps"],
        seen["thermal"]))
    if seen["events"]:
        print("events: %s" % seen["events"][:40])
    if seen["steps"] == 0:
        print("note: the program made no step call (a submission that never steps on any site fails the grader's contract)")
    if left:
        print("note: the program left a process running after the mission (the grader fails a submission for that)")
    print("mission log: %s" % json.dumps(mission_log, indent=1))
    if args.keep:
        os.makedirs(args.keep, exist_ok=True)
        for name in ("home", "tmp"):
            dst = os.path.join(args.keep, name)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(os.path.join(run, name), dst)
        print("the run's HOME and TMPDIR: %s" % args.keep)
    remove_run(run)


if __name__ == "__main__":
    try:
        main()
    except CellError as e:
        sys.exit("training ground: %s" % e)
