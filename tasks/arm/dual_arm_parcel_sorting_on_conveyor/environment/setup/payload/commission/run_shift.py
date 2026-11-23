#!/opt/venv/bin/python3
"""Commissioning run: runs YOUR program over the real wire in one commissioning shift on a development shift of the
line (the commissioning service's shifts, never a graded one), the way the grader runs it.

    /opt/venv/bin/python3 ~/commission/run_shift.py --site 3                      # ~/submission on development shift 3
    /opt/venv/bin/python3 ~/commission/run_shift.py --site 3 --keep ~/runs/s3 --transcript ~/runs/s3/wire.jsonl

The program runs as you, from a fresh read-only copy of its folder (regular files only: no symbolic links, no
__pycache__), with the grader's environment (an empty HOME and TMPDIR, PYTHONPATH = the given client library,
ARB_GIVEN_DIR = ~/given, MUJOCO_GL=osmesa, BLAS / OpenMP / rasteriser threads 2, nothing else of your shell's
environment), the grader's launcher (`python -B -E -s` through the client's bootstrap, or `bash run.sh`), its CPU cap
(4 x the shift's program-time budget), its no-network filter (no inet socket: socket() fails for every address family
but AF_UNIX) and its out-of-memory preference (the program's processes go first); its stdin / stdout carry the wire to
the commissioning service, its stderr comes out here. The service applies the graded shift's budgets itself (the
shift's duration, the pooled frame budget, the program-time budget, the silence cap, the start window) and ends the
shift when one runs out. Each run uses one shift of the commissioning window (the shifts left:
/run/parcelcell/commission_status.json). Keep this script out of your submission folder: it starts a process, which
the grader's audit refuses in your program's files.

What it prints is what the program and the station saw: how the shift ended, the simulated time, the frames, the
protective stops and refused commands reported to the program. There is no score and no debrief: judge the shift by
what your program logged and saved (--keep keeps the run's HOME and TMPDIR, where it may write its own records and
frames).
"""

import argparse
import json
import os
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GIVEN = os.environ.get("ARB_GIVEN_DIR", os.path.join(os.path.dirname(HERE), "given"))
LIB = os.path.join(GIVEN, "lib")
sys.path.insert(0, LIB)

from parcelcell.client import COMMISSION_SOCKET, COMMISSION_STATUS, launcher_argv  # noqa: E402

COMMISSION_SOCKET = os.environ.get("PARCELCELL_COMMISSION_SOCKET", COMMISSION_SOCKET)
COMMISSION_STATUS = os.environ.get("PARCELCELL_COMMISSION_STATUS", COMMISSION_STATUS)

CPU_FACTOR = 4.0
OOM_FIRST = ["choom", "-n", "1000", "--"]
# the grader's no-network filter (installs a seccomp filter, then execs the program): the same code the grader runs
NONET_SRC = r'''
import ctypes, os, platform, struct, sys
ARCH = {"x86_64": (0xC000003E, 41, 53, 425, 0x40000000), "aarch64": (0xC00000B7, 198, 199, 425, 0)}
def ins(code, jt, jf, k):
    return struct.pack("HBBI", code, jt, jf, k)
def main():
    audit, nr_socket, nr_socketpair, nr_uring, x32 = ARCH[platform.machine()]
    LD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06
    allow, deny = 0x7FFF0000, 0x00050000 | 1
    prog = [ins(LD, 0, 0, 4), ins(JEQ, 1, 0, audit), ins(RET, 0, 0, deny), ins(LD, 0, 0, 0)]
    if x32:
        prog += [ins(JGE, 0, 1, x32), ins(RET, 0, 0, deny)]
    prog += [ins(JEQ, 0, 1, nr_uring), ins(RET, 0, 0, deny), ins(JEQ, 2, 0, nr_socket), ins(JEQ, 1, 0, nr_socketpair),
             ins(RET, 0, 0, allow), ins(LD, 0, 0, 16), ins(JEQ, 0, 1, 1), ins(RET, 0, 0, allow), ins(RET, 0, 0, deny)]
    raw = b"".join(prog)
    buf = ctypes.create_string_buffer(raw, len(raw))
    class Fprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
    fp = Fprog(len(raw) // 8, ctypes.cast(buf, ctypes.c_void_p))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(22, 2, ctypes.byref(fp), 0, 0) != 0:
        sys.stderr.write("[nonet] the seccomp filter could not be installed (errno %d)\n" % ctypes.get_errno())
        os._exit(97)
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    os.execvp(argv[0], argv)
main()
'''


def copy_program(src, dst):
    """The submission folder's regular files (no symbolic links, no __pycache__), read-only, as the grader copies it."""
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d != "__pycache__" and not os.path.islink(os.path.join(dp, d))]
        rel = os.path.relpath(dp, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for fn in fns:
            p = os.path.join(dp, fn)
            if os.path.isfile(p) and not os.path.islink(p):
                shutil.copyfile(p, os.path.join(dst, rel, fn))
    for dp, dns, fns in os.walk(dst):
        for fn in fns:
            os.chmod(os.path.join(dp, fn), 0o444)
    for dp, dns, fns in os.walk(dst, topdown=False):
        for d in dns:
            os.chmod(os.path.join(dp, d), 0o555)
    os.chmod(dst, 0o555)


def make_writable(root):
    for dp, dns, fns in os.walk(root):
        os.chmod(dp, 0o755)
        for d in dns:
            try:
                os.chmod(os.path.join(dp, d), 0o755)
            except OSError:
                pass


def elide(obj):
    """A reply for the transcript: the images' base64 replaced by their length."""
    if isinstance(obj, dict):
        return {k: ("<%d bytes>" % len(v) if k.endswith("_b64") and isinstance(v, str) else elide(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [elide(v) for v in obj]
    return obj


def filter_works(python):
    """True when the no-network filter installs here (as the grader proves it before its first shift)."""
    probe = "import socket, sys\ntry:\n    socket.socket(socket.AF_INET, socket.SOCK_STREAM)\nexcept PermissionError:\n    sys.exit(0)\nsys.exit(3)\n"
    try:
        r = subprocess.run([python, "-I", "-B", "-c", NONET_SRC, "--", python, "-I", "-B", "-c", probe],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def read_line(sock, buf, timeout):
    """(line, rest) from the socket within `timeout` s; line b"" on EOF or timeout."""
    deadline = time.time() + timeout
    while b"\n" not in buf:
        left = deadline - time.time()
        if left <= 0:
            return b"", buf
        r, _, _ = select.select([sock], [], [], left)
        if not r:
            return b"", buf
        chunk = sock.recv(1 << 20)
        if not chunk:
            return b"", buf
        buf += chunk
    line, rest = buf.split(b"\n", 1)
    return line, rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=int, required=True, help="the development shift (1 .. the number of development shifts)")
    ap.add_argument("--program", default=os.path.expanduser("~/submission"))
    ap.add_argument("--keep", default=None, help="keep the run's HOME and TMPDIR in this folder")
    ap.add_argument("--transcript", default=None, help="write every request and reply (images elided) here, JSON lines")
    args = ap.parse_args()
    tr = None
    if args.transcript:                    # opened first: a bad path must not use a shift of the window
        os.makedirs(os.path.dirname(os.path.abspath(args.transcript)), exist_ok=True)
        tr = open(args.transcript, "w")

    py = sys.executable
    use_filter = filter_works(py)
    if not use_filter:
        print("note: the no-network filter does not install here; the program runs without it", flush=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(COMMISSION_SOCKET)
    except OSError as e:
        sys.exit("commissioning: the service is not reachable at %s (%s)" % (COMMISSION_SOCKET, e))
    sock.sendall((json.dumps({"id": 0, "method": "reset", "params": {"site": int(args.site)}}) + "\n").encode())
    line, svc_buf = read_line(sock, b"", 300.0)          # the shift's build: up to about a minute
    if not line:
        sys.exit("commissioning: no answer to reset(site)")
    first = json.loads(line)
    if "error" in first:
        sys.exit("commissioning: %s: %s" % (first.get("error"), first.get("detail", "")))
    info = first["result"]
    print("commissioning shift %d on development shift %d (%s; %.0f s; program-time budget %.0f s; %d frames); "
          "shifts left after this one: %d" % (info["shift"], info["site"], info["kind"], info["duration_s"],
                                               info["program_time_s"], info["frames"], info["shifts_left"]), flush=True)
    run = tempfile.mkdtemp(prefix="commission_run_")
    prog = os.path.join(run, "program")
    copy_program(os.path.abspath(args.program), prog)
    home, tmp = os.path.join(run, "home"), os.path.join(run, "tmp")
    os.makedirs(home, mode=0o700)
    os.makedirs(tmp, mode=0o700)
    env = {"PATH": os.environ.get("PATH", "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"), "HOME": home, "TMPDIR": tmp,
           "TMP": tmp, "TEMP": tmp, "LANG": "C.UTF-8", "MPLBACKEND": "Agg", "PYTHONPATH": LIB, "ARB_GIVEN_DIR": GIVEN,
           "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "LP_NUM_THREADS": "2"}
    cpu_s = int(CPU_FACTOR * float(info["program_time_s"]))
    argv = OOM_FIRST + launcher_argv(py, LIB, prog)
    if use_filter:
        argv = [py, "-I", "-B", "-c", NONET_SRC, "--"] + argv
    if shutil.which("prlimit"):
        argv = ["prlimit", "--cpu=%d" % cpu_s, "--core=0", "--"] + argv
    proc = subprocess.Popen(argv, cwd=prog, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            start_new_session=True)
    out_fd, in_fd = proc.stdout.fileno(), proc.stdin.fileno()
    os.set_blocking(in_fd, False)
    t0 = time.time()
    pending = {}
    to_prog = b""
    prog_buf = b""
    seen = {"requests": 0, "frames": 0, "repeats": 0, "t": 0.0, "stops": 0, "locked_out": False, "refused": 0,
            "frames_left": None, "program_time_left": None, "over_reason": None, "end": None}
    prog_open = True                     # the program's stdout (its requests)
    svc_open = True                      # the service's side of the connection
    wind_down = None                     # when one side ended: at most 10 s more to deliver what is pending
    while True:
        if (not prog_open or not svc_open) and wind_down is None:
            wind_down = time.time() + 10.0
        if not svc_open and (not to_prog or not prog_open):
            break
        if wind_down is not None and time.time() > wind_down:
            break
        rl = ([sock] if svc_open else []) + ([out_fd] if prog_open else [])
        wl = [in_fd] if to_prog else []
        r, w, _ = select.select(rl, wl, [], 0.5)
        if in_fd in w:
            try:
                n = os.write(in_fd, to_prog)
                to_prog = to_prog[n:]
            except BlockingIOError:
                pass
            except OSError:
                to_prog = b""                    # the program is gone or closed its stdin
                seen["end"] = seen["end"] or "the program stopped reading"
        if out_fd in r:
            data = os.read(out_fd, 1 << 20)
            if not data:
                prog_open = False
                seen["end"] = seen["end"] or "the program exited"
                try:
                    sock.shutdown(socket.SHUT_WR)       # the service sees the end of the wire, as the grader does
                except OSError:
                    pass
            prog_buf += data
            while b"\n" in prog_buf:
                ln, prog_buf = prog_buf.split(b"\n", 1)
                seen["requests"] += 1
                try:
                    req = json.loads(ln)
                    if isinstance(req, dict):
                        pending[json.dumps(req.get("id"))] = req
                except ValueError:
                    pass
                try:
                    sock.sendall(ln + b"\n")
                except OSError:
                    seen["end"] = seen["end"] or "the commissioning service closed the shift"
        if svc_open and sock in r:
            try:
                chunk = sock.recv(1 << 20)
            except OSError:
                chunk = b""
            if not chunk:
                svc_open = False
                seen["end"] = seen["end"] or "the commissioning service closed the shift"
            svc_buf += chunk
            while b"\n" in svc_buf:
                ln, svc_buf = svc_buf.split(b"\n", 1)
                try:
                    rep = json.loads(ln)
                except ValueError:
                    continue
                if rep.get("event") == "shift_over":
                    seen["end"] = "%s; the shift is over: %s" % (seen["end"], rep.get("reason")) if seen["end"] else \
                        "the shift is over: %s" % rep.get("reason")
                    to_prog += ln + b"\n"
                    continue
                req = pending.pop(json.dumps(rep.get("id")), {})
                res = rep.get("result")
                if isinstance(res, dict):
                    if "camera" in res:
                        seen["frames" if not res.get("repeat") else "repeats"] += 1
                    if "t" in res and "q" in res:
                        seen["t"] = float(res["t"])
                        seen["frames_left"] = res.get("frames_left")
                        seen["program_time_left"] = res.get("program_time_left")
                        seen["over_reason"] = res.get("over_reason")
                        stop = res.get("stop") or {}
                        seen["stops"] = int(stop.get("count", seen["stops"]) or 0)
                        seen["locked_out"] = bool(stop.get("locked_out"))
                        if res.get("refused"):
                            seen["refused"] += 1
                    if req.get("method") == "finish":
                        seen["end"] = seen["end"] or "finish"
                if tr is not None:
                    tr.write(json.dumps({"t_wall": round(time.time() - t0, 3), "request": req, "reply": elide(rep)}) + "\n")
                to_prog += ln + b"\n"
    try:
        proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    try:
        os.killpg(proc.pid, 9)
    except OSError:
        pass
    sock.close()
    if tr is not None:
        tr.close()
    print("end: %s; program exit code %s; simulated %.2f s; wall %.0f s; requests %d; frames %d (+%d repeats); "
          "frames left %s; program time left %s s" % (seen["end"], proc.returncode, seen["t"], time.time() - t0,
                                                       seen["requests"], seen["frames"], seen["repeats"],
                                                       seen["frames_left"], seen["program_time_left"]))
    print("protective stops: %d; locked out: %s; refused commands: %d; shift over: %s" % (
        seen["stops"], seen["locked_out"], seen["refused"], seen["over_reason"]))
    try:
        st = json.load(open(COMMISSION_STATUS))
        print("commissioning window: %d of %d shifts used" % (st["shifts_used"], st["commissioning_shifts"]))
    except (OSError, ValueError, KeyError):
        pass
    make_writable(run)
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
    main()
