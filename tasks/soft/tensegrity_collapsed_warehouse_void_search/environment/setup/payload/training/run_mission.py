#!/opt/venv/bin/python3
"""Practice run: runs YOUR program over the real wire in one episode on a site of the practice ground (the
calibration site or a practice site, never a scored site), the way a scored episode runs it, and prints the
trainers' debrief.

    /opt/venv/bin/python3 ~/training/run_mission.py --list                         # the sites and episodes left
    /opt/venv/bin/python3 ~/training/run_mission.py --site cal                     # ~/submission on the calibration site
    /opt/venv/bin/python3 ~/training/run_mission.py --site p2 --keep ~/runs/r3 --transcript ~/runs/r3/wire.jsonl

The program runs as a scored episode runs it, except that it runs as you: from a read-only copy of its folder
(regular files only), with read-only copies of the client library (PYTHONPATH) and of the givens (ARB_GIVEN_DIR), an
empty HOME and TMPDIR, two BLAS and rasteriser threads, the scored episodes' numerical kernels (NUMERICS below),
the scored episodes' launcher (`python -B -E -s` through voidcell.client, which imports operate.py and calls
solve(station)) and their CPU cap; its stdin / stdout carry the wire to the practice ground, its stderr comes out
here. The practice ground applies the scored episodes' budgets
itself (simulated time, camera frames, reports, program time, silence) and ends the episode when one runs out.
`reset` is this script's (the program's own `reset` calls answer `forbidden`, as in a scored episode). Each run uses
one episode of the practice ground's allowance. The debrief is also saved in /run/voidcell/debriefs/. Keep this script
out of your submission folder: it starts a process, which the scored run's audit refuses.

The episode ends this script too: when the practice ground ends the episode (a budget, the silence limit) or the
program exits, the program's processes are stopped, the debrief is printed and the script returns. What is stopped is
the program's process group: a process that leaves the group (a daemon started with setsid, for instance) is not
stopped by a practice run and keeps running in your sandbox; in a scored episode every process of the program's
account is stopped. The files this script writes (--transcript, --debrief, --keep) are opened, their folders made,
before the episode is reserved: a path that cannot be written stops the script without using an episode.
"""

import argparse
import json
import os
import resource
import select
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GIVEN = os.path.join(os.path.dirname(HERE), "given")
LIB = os.path.join(GIVEN, "lib")
sys.path.insert(0, LIB)

from voidcell.client import PRACTICE_SOCKET, PRACTICE_STATUS, PROGRAM, StationError, launcher_argv  # noqa: E402

PROGRAM_TIME_S = 600.0            # the practice ground's program-time budget (its status file says it; this if not)
CPU_S = 2400.0                    # the program's CPU seconds (the status file says it; this if not)
LAST_WORD_S = 60.0                # how long the practice ground's last word (episode_over) is awaited at the end
DEBRIEFS_DIR = "/run/voidcell/debriefs"
EVENT = "episode_over"
NOT_ON_THE_WIRE = ("reset",)      # this script's own; every other call goes to the practice ground
# the numerical kernels of a scored episode's program, on every x86-64 CPU: OpenBLAS's Haswell kernel with its
# built-in blocking (L2 size 256) and numpy's SIMD paths up to AVX2; the program's numbers are those of a scored run
NUMERICS = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256",
            "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR"}


def copy_program(src, dst):
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d != "__pycache__" and not os.path.islink(os.path.join(dp, d))]
        rel = os.path.relpath(dp, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for fn in fns:
            p = os.path.join(dp, fn)
            if os.path.isfile(p) and not os.path.islink(p):
                shutil.copy2(p, os.path.join(dst, rel, fn))


def set_modes(top, dir_mode, file_mode):
    os.chmod(top, dir_mode)
    for dp, dns, fns in os.walk(top):
        for d in dns:
            os.chmod(os.path.join(dp, d), dir_mode)
        for fn in fns:
            os.chmod(os.path.join(dp, fn), file_mode)


class Ground(object):
    """This script's connection to the practice ground: whole lines, and whether a line is waiting."""

    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(path)
        except OSError as e:
            self.sock.close()
            raise StationError("unreachable", "the practice ground is not reachable at %s (%s)" % (path, e))
        self.buf = b""
        self.eof = False

    def send(self, text):
        try:
            self.sock.sendall(text.encode("utf-8"))
        except OSError as e:
            raise StationError("closed", "the practice ground closed the connection (%s)" % e)

    def line(self, timeout=None):
        """The next line ('' at the end of the connection); None if none came within `timeout` s."""
        end = None if timeout is None else time.time() + timeout
        while b"\n" not in self.buf and not self.eof:
            left = None if end is None else max(0.0, end - time.time())
            self.sock.settimeout(left)
            try:
                chunk = self.sock.recv(1 << 20)
            except (socket.timeout, BlockingIOError, InterruptedError):
                return None
            except OSError:
                chunk = b""
            if not chunk:
                self.eof = True
            self.buf += chunk
        if b"\n" in self.buf:
            ln, self.buf = self.buf.split(b"\n", 1)
            return ln.decode("utf-8", "replace") + "\n"
        ln, self.buf = self.buf, b""
        return ln.decode("utf-8", "replace")

    def call(self, method, params, rid=0):
        self.send(json.dumps({"id": rid, "method": method, "params": params}) + "\n")
        raw = self.line()
        if not raw:
            raise StationError("closed", "the practice ground closed the connection")
        rep = json.loads(raw)
        if rep.get("event") == EVENT:
            raise StationError("episode_over", "the practice ground ended the connection (%s)" % rep.get("reason"))
        if "error" in rep:
            raise StationError(str(rep["error"]), str(rep.get("detail") or ""))
        return rep.get("result")

    def half_close(self):
        try:
            self.sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def stop_program(proc):
    """The program's process group (it runs in its own session) is stopped; a process that left the group (a daemon)
    is not (the module's docstring says so; a scored episode stops every process of the program's account)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            proc.wait(timeout=3)
            if sig == signal.SIGTERM:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            break
        except subprocess.TimeoutExpired:
            continue
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def elide(obj):
    if isinstance(obj, dict):
        return {k: ("<%d bytes>" % len(v) if k.endswith("_b64") and isinstance(v, str) else elide(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [elide(v) for v in obj]
    return obj


def open_output(path):
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "w")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", help="a practice site: its name or number (--list)")
    ap.add_argument("--program", default=os.path.expanduser("~/submission"))
    ap.add_argument("--keep", default=None, help="keep the run's HOME and TMPDIR in this folder")
    ap.add_argument("--transcript", default=None, help="write every request and reply (frames elided) here, JSON lines")
    ap.add_argument("--debrief", default=None, help="also write the debrief (JSON) to this file")
    ap.add_argument("--list", action="store_true", help="print the practice ground's sites and allowance")
    ap.add_argument("--socket", default=PRACTICE_SOCKET, help=argparse.SUPPRESS)
    ap.add_argument("--status", default=PRACTICE_STATUS, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.list or not args.site:
        try:
            print(json.dumps(json.load(open(args.status)), indent=1))
        except (OSError, ValueError) as e:
            print("practice ground status unreadable: %s" % e)
        return
    if not os.path.isfile(os.path.join(args.program, PROGRAM + ".py")):
        sys.exit("no %s.py in %s" % (PROGRAM, args.program))
    tr = dbf = None
    try:
        tr = open_output(args.transcript)
        dbf = open_output(args.debrief)
        if args.keep:
            os.makedirs(os.path.abspath(os.path.expanduser(args.keep)), exist_ok=True)
            if not os.access(os.path.expanduser(args.keep), os.W_OK | os.X_OK):
                raise OSError("%s is not writable" % args.keep)
    except OSError as e:
        for f in (tr, dbf):
            if f is not None:
                f.close()
        sys.exit("cannot write the run's outputs (%s); no practice episode was used" % e)
    run = tempfile.mkdtemp(prefix="practice_run_")
    read_only = []
    procs = []
    try:
        prog = os.path.join(run, "program")
        copy_program(os.path.abspath(args.program), prog)
        lib = os.path.join(run, "lib")
        copy_program(LIB, lib)
        given = os.path.join(run, "given")
        copy_program(GIVEN, given)
        read_only.extend((prog, lib, given))
        for d in read_only:
            set_modes(d, 0o555, 0o444)
        home, tmp = os.path.join(run, "home"), os.path.join(run, "tmp")
        os.makedirs(home)
        os.makedirs(tmp)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "TMPDIR": tmp, "TMP": tmp, "TEMP": tmp,
               "LANG": "C.UTF-8", "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "PYTHONPATH": lib,
               "MPLBACKEND": "Agg", "ARB_GIVEN_DIR": given,
               "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "LP_NUM_THREADS": "2"}
        env.update(NUMERICS)
        try:
            with open(args.status) as f:
                st = json.load(f)
            cap = float(st.get("program_time_s") or PROGRAM_TIME_S)
            cpu_s = int(float(st.get("cpu_s") or CPU_S))
        except (OSError, ValueError, TypeError):
            cap, cpu_s = PROGRAM_TIME_S, int(CPU_S)

        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))

        ground = Ground(args.socket)
        first = ground.call("reset", {"site": str(args.site)})
        print("practice episode %d on site %s (simulated time limit %.0f s, program time %.0f s); episodes left after "
              "this one: %d" % (first["episode"], first["site"], first.get("time_limit_s", 0.0), cap,
                                first["episodes_left"]), flush=True)
        proc = subprocess.Popen(launcher_argv(sys.executable, lib, prog), cwd=prog, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, bufsize=0, preexec_fn=limits, start_new_session=True)
        procs.append(proc)
        os.set_blocking(proc.stdin.fileno(), False)
        t0 = time.time()
        seen = {"requests": 0, "frames": 0, "commands": 0, "reports": 0, "sim_time": 0.0}
        over = {}
        note = []

        def last_word(raw):
            try:
                obj = json.loads(raw) if raw and raw.strip() else None
            except ValueError:
                obj = None
            if isinstance(obj, dict) and obj.get("event") == EVENT:
                over.update(obj)
            else:
                n = ("the practice ground closed the connection without its last word (the debriefs are saved in %s)"
                     % DEBRIEFS_DIR)
                if n not in note:
                    note.append(n)

        def wait_last_word():
            if not over:
                ground.half_close()
                last_word(ground.line(timeout=LAST_WORD_S))

        def to_program(data):
            view = memoryview(data)
            while len(view):
                if b"\n" in ground.buf or ground.eof:
                    last_word(ground.line(timeout=0.0))
                    return "ground"
                r, w, _ = select.select([ground.sock], [proc.stdin], [], None)
                if ground.sock in r:
                    raw = ground.line(timeout=0.0)
                    if raw is not None:
                        last_word(raw)
                        return "ground"
                if proc.stdin in w:
                    try:
                        view = view[os.write(proc.stdin.fileno(), view):]
                    except BlockingIOError:
                        continue
                    except OSError:
                        return "program"
            return None

        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ, "program")
        sel.register(ground.sock, selectors.EVENT_READ, "ground")
        pbuf = b""
        program_open = True
        while True:
            if b"\n" in ground.buf or ground.eof:
                last_word(ground.line(timeout=0.0))
                break
            if b"\n" not in pbuf:
                if not program_open:
                    wait_last_word()
                    break
                ready = [key.data for key, _ in sel.select(timeout=None)]
                if "ground" in ready:
                    raw = ground.line(timeout=0.0)
                    if raw is not None:
                        last_word(raw)
                        break
                if "program" in ready:
                    try:
                        chunk = os.read(proc.stdout.fileno(), 1 << 20)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        program_open = False
                        sel.unregister(proc.stdout)
                        note.append("the program closed the wire")
                        continue
                    pbuf += chunk
                continue
            line_b, pbuf = pbuf.split(b"\n", 1)
            line = line_b.decode("utf-8", "replace") + "\n"
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise ValueError("not an object")
                method = str(req.get("method"))
            except ValueError:
                note.append("the program sent a line that is not a request")
                try:
                    ground.send(line)
                except StationError:
                    pass
                wait_last_word()
                break
            seen["requests"] += 1
            if method in NOT_ON_THE_WIRE:
                reply_obj = {"id": req.get("id"), "error": "forbidden"}
            else:
                try:
                    ground.send(line)
                except StationError:
                    pass
                raw = ground.line()
                try:
                    reply_obj = json.loads(raw) if raw and raw.strip() else None
                except ValueError:
                    reply_obj = None
                if not isinstance(reply_obj, dict) or reply_obj.get("event") == EVENT:
                    last_word(raw)
                    break
            res = reply_obj.get("result")
            if isinstance(res, dict):
                if method == "get_frame" and "index" in res:
                    seen["frames"] += 1
                if method in ("run_primitive", "set_targets") and res.get("accepted"):
                    seen["commands"] += 1
                if method == "report_point" and res.get("accepted"):
                    seen["reports"] += 1
                if method == "step" and "t" in res:
                    seen["sim_time"] = float(res["t"])
            if tr is not None:
                tr.write(json.dumps({"t_wall": round(time.time() - t0, 3), "request": req,
                                     "reply": elide(reply_obj)}) + "\n")
            w = to_program((json.dumps(reply_obj) + "\n").encode("utf-8"))
            if w == "ground":
                break
            if w == "program":
                note.append("the program stopped reading the wire")
                wait_last_word()
                break
            if method == "finish" and "result" in reply_obj:
                break
        sel.close()
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=10 if over.get("reason") == "finish" or not over else 0.5)
        except subprocess.TimeoutExpired:
            pass
        rc_own = proc.poll()
        stop_program(proc)
        wait_last_word()
        ground.close()
        if tr is not None:
            tr.close()
        cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_used = cpu.ru_utime + cpu.ru_stime
        if rc_own is None:
            how = "was stopped by this script at the episode's end"
        elif rc_own >= 0:
            how = "exited with code %d" % rc_own
        elif -rc_own in (signal.SIGXCPU, signal.SIGKILL) and cpu_used >= cpu_s - 1:
            how = "was stopped by its CPU-time limit (%d CPU-s used of %d)" % (cpu_used, cpu_s)
        else:
            how = "was killed by signal %d" % -rc_own
        debrief = over.get("debrief")
        reason = over.get("reason")
        ended = debrief.get("end_reason") if isinstance(debrief, dict) else None
        if not reason:
            said = "(no word from the practice ground)"
        elif ended and ended != reason:
            said = "%s (the practice ground then closed the connection: %s)" % (ended, reason)
        else:
            said = reason
        print("end: the episode ended: %s%s; the program %s%s; sim %.1f s; wall %.0f s; requests %d; frames %d; "
              "commands %d; reports %d" % (
                  said, (" (%s)" % over["detail"]) if over.get("detail") else "", how,
                  ("; " + "; ".join(note)) if note else "", seen["sim_time"], time.time() - t0, seen["requests"],
                  seen["frames"], seen["commands"], seen["reports"]))
        print("debrief: %s" % json.dumps(debrief, indent=1))
        if dbf is not None:
            json.dump(debrief, dbf, indent=1)
            dbf.close()
        if args.keep:
            os.makedirs(args.keep, exist_ok=True)
            for name in ("home", "tmp"):
                dst = os.path.join(args.keep, name)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(os.path.join(run, name), dst)
            print("the run's HOME and TMPDIR: %s" % args.keep)
    finally:
        for proc in procs:
            stop_program(proc)
        for f in (tr, dbf):
            if f is not None and not f.closed:
                f.close()
        for d in read_only:
            if os.path.isdir(d):
                set_modes(d, 0o755, 0o644)
        shutil.rmtree(run, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except StationError as e:
        sys.exit("practice ground: %s" % e)
