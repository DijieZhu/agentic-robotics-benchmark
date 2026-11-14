#!/opt/venv/bin/python3
"""Commissioning run: runs YOUR program over the real wire in one commissioning shift on a development site of the
series (the commissioning service's sister sites, never a graded site), the way the grader runs it.

    /opt/venv/bin/python3 ~/commission/run_shift.py --site 3                      # ~/submission on development site 3
    /opt/venv/bin/python3 ~/commission/run_shift.py --site 3 --keep ~/runs/s3 --transcript ~/runs/s3/wire.jsonl

The program runs as you, from a fresh copy of its folder (regular files only), with the grader's environment (an empty
HOME and TMPDIR, PYTHONPATH = the given client library, ARB_GIVEN_DIR = ~/given, MUJOCO_GL=osmesa, BLAS threads 2),
the grader's launcher and its CPU cap (4 x the program-time budget); its stdin / stdout carry the wire to the
commissioning service, its stderr comes out here. The service applies the graded episode's budgets itself (1800 s
simulated, 600 frames, 1200 s of program time, 180 s of silence) and ends the shift when one runs out. `reset` is
this script's (the program's own `reset` calls answer `forbidden`, as in the grader). Each run uses one shift of the
commissioning window (`cell_manual.md` section 8; the shifts left: /run/skidcell/commission_status.json). Keep this
script out of your submission folder: it starts a process, which the grader's audit refuses in your program's files.

What it prints is what the program and the robot cell saw: how the shift ended, the simulated time, the frames, the
grips, slips and protective stops reported to the program. There is no score and no debrief: judge the shift by what
your program logged and saved (--keep keeps the run's HOME and TMPDIR, where it may write its own records and frames).
"""

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GIVEN = os.environ.get("ARB_GIVEN_DIR", os.path.join(os.path.dirname(HERE), "given"))
LIB = os.path.join(GIVEN, "lib")
sys.path.insert(0, LIB)

from skidcell.client import COMMISSION_STATUS, CellError, connect_commissioning, launcher_argv  # noqa: E402

PROGRAM_TIME_S = 1200.0
CPU_FACTOR = 4.0
NOT_ON_THE_WIRE = ("reset", "truth", "get_state_privileged", "reference_run")


def copy_program(src, dst):
    """The submission folder's regular files (no symbolic links, no __pycache__), as the grader copies it."""
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d != "__pycache__" and not os.path.islink(os.path.join(dp, d))]
        rel = os.path.relpath(dp, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for fn in fns:
            p = os.path.join(dp, fn)
            if os.path.isfile(p) and not os.path.islink(p):
                shutil.copy2(p, os.path.join(dst, rel, fn))


def elide(obj):
    """A reply for the transcript: the images' base64 replaced by their length."""
    if isinstance(obj, dict):
        return {k: ("<%d bytes>" % len(v) if k.endswith("_b64") and isinstance(v, str) else elide(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [elide(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=int, required=True, help="the development site (1 .. the number of sites)")
    ap.add_argument("--program", default=os.path.expanduser("~/submission"))
    ap.add_argument("--keep", default=None, help="keep the run's HOME and TMPDIR in this folder")
    ap.add_argument("--transcript", default=None, help="write every request and reply (images elided) here, JSON lines")
    args = ap.parse_args()

    cell = connect_commissioning()
    first = cell.reset(args.site)
    print("commissioning shift %d on development site %d (%s); shifts left after this one: %d" % (
        first["shift"], first["site"], first["site_name"], first["shifts_left"]), flush=True)
    run = tempfile.mkdtemp(prefix="commission_run_")
    prog = os.path.join(run, "program")
    copy_program(os.path.abspath(args.program), prog)
    home, tmp = os.path.join(run, "home"), os.path.join(run, "tmp")
    os.makedirs(home)
    os.makedirs(tmp)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "TMPDIR": tmp, "LANG": "C.UTF-8",
           "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "PYTHONPATH": LIB, "ARB_GIVEN_DIR": GIVEN,
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    cpu_s = int(CPU_FACTOR * PROGRAM_TIME_S)

    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))

    proc = subprocess.Popen(launcher_argv(sys.executable, LIB, prog), cwd=prog, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1, preexec_fn=limits, start_new_session=True)
    tr = open(args.transcript, "w") if args.transcript else None
    t0 = time.time()
    seen = {"requests": 0, "frames": 0, "grasped": 0, "slipped": 0, "released": 0, "protective_stop": [], "safety": [],
            "sim_time": 0.0, "end": None}
    send, recv = cell._send, cell._recv
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
                pass                                # the service closed the shift: its last word, if any, follows
            raw = recv()
            if not raw:
                seen["end"] = seen["end"] or "the commissioning service closed the shift"
                break
            reply_obj = json.loads(raw)
            if reply_obj.get("event") == "shift_over":
                seen["end"] = "the shift is over: %s" % reply_obj.get("reason")
                break
        res = reply_obj.get("result")
        if isinstance(res, dict):
            if method == "get_camera" and res.get("ok"):
                seen["frames"] += 1
            if "sim_time" in res:
                seen["sim_time"] = float(res["sim_time"])
            for e in res.get("events", []) if method == "step" else []:
                if e in ("grasped", "slipped", "released"):
                    seen[e] += 1
                elif e.startswith("protective_stop:"):
                    seen["protective_stop"].append((round(seen["sim_time"], 1), e))
                elif e.startswith("safety:"):
                    seen["safety"].append((round(seen["sim_time"], 1), e))
        if tr is not None:
            tr.write(json.dumps({"t_wall": round(time.time() - t0, 3), "request": req, "reply": elide(reply_obj)}) + "\n")
        try:
            proc.stdin.write(json.dumps(reply_obj) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            seen["end"] = "the program stopped reading"
            break
        if method == "finish" and "result" in reply_obj:
            seen["end"] = "finish"                  # the service ends the shift here, as the grader's wire does
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
    cell.close()
    if tr is not None:
        tr.close()
    print("end: %s; program exit code %s; sim %.1f s; wall %.0f s; requests %d; frames %d" % (
        seen["end"], proc.returncode, seen["sim_time"], time.time() - t0, seen["requests"], seen["frames"]))
    print("grip events: grasped %d, slipped %d, released %d" % (seen["grasped"], seen["slipped"], seen["released"]))
    print("protective stops: %d %s" % (len(seen["protective_stop"]), seen["protective_stop"][:10]))
    if seen["safety"]:
        print("safety: %s" % seen["safety"][:5])
    try:
        st = json.load(open(COMMISSION_STATUS))
        print("commissioning window: %d of %d shifts used" % (st["shifts_used"], st["commissioning_shifts"]))
    except (OSError, ValueError, KeyError):
        pass
    if args.keep:
        os.makedirs(args.keep, exist_ok=True)
        for name in ("home", "tmp"):
            dst = os.path.join(args.keep, name)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(os.path.join(run, name), dst)
        print("the run's HOME and TMPDIR: %s" % args.keep)
    shutil.rmtree(run, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except CellError as e:
        sys.exit("commissioning: %s" % e)
