#!/opt/venv/bin/python3
"""Practice mission: runs YOUR program over the real wire in one mission on a site of the practice ground (the
training hall or a practice building, never a scored site), the way a scored mission runs it, and prints the
trainers' debrief.

    /opt/venv/bin/python3 ~/training/run_mission.py --list                         # the sites and missions left
    /opt/venv/bin/python3 ~/training/run_mission.py --site hall                    # ~/submission in the hall
    /opt/venv/bin/python3 ~/training/run_mission.py --site t2 --keep ~/runs/m3 --transcript ~/runs/m3/wire.jsonl

The program runs as a scored mission runs it, except that it runs as you: from a read-only copy of its folder
(regular files only), with read-only copies of the client library (PYTHONPATH) and of the givens (ARB_GIVEN_DIR),
an empty HOME and TMPDIR, the image's rendering settings and two BLAS and rasteriser threads, the scored missions' launcher
(`python -B -E -s` through plantcell.client) and their CPU cap (4 x the program-time budget); its stdin / stdout
carry the wire to the practice ground, its stderr comes out here. The practice ground applies the scored missions'
budgets itself (simulated time, sensor frames, reports, program time, silence) and ends the mission when one runs
out. `reset` is this script's (the program's own `reset` calls answer `forbidden`, as in a scored mission). Each run
uses one mission of the practice ground's allowance. The debrief is also saved in /run/plantcell/debriefs/. Keep this
script out of your submission folder: it starts a process, which the scored run's audit refuses.

The mission ends this script too: when the practice ground ends the mission (a budget, 180 s of silence) or the
program exits, the program's processes are stopped, the debrief is printed and the script returns. What is stopped
is the program's process group: a process that leaves the group (a daemon started with setsid, for instance) is not
stopped by a practice run and keeps running in your sandbox; in a scored mission every process of the program's
account is stopped. The files this script writes (--transcript, --debrief, --keep) are opened, their folders made,
before the mission is reserved: a path that cannot be written stops the script without using a mission.
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

from plantcell.client import PRACTICE_SOCKET, PRACTICE_STATUS, CellError, launcher_argv  # noqa: E402

PROGRAM_TIME_S = 1200.0          # the practice ground's program-time budget (its status file says it; this if not)
CPU_FACTOR = 4.0
LAST_WORD_S = 60.0               # how long the practice ground's last word (mission_over) is awaited at the end
DEBRIEFS_DIR = "/run/plantcell/debriefs"
NOT_ON_THE_WIRE = ("reset",)            # this script's own; every other call goes to the practice ground


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
            raise CellError("unreachable", "the practice ground is not reachable at %s (%s)" % (path, e))
        self.buf = b""
        self.eof = False

    def send(self, text):
        try:
            self.sock.sendall(text.encode("utf-8"))
        except OSError as e:
            raise CellError("closed", "the practice ground closed the connection (%s)" % e)

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
            raise CellError("closed", "the practice ground closed the connection")
        rep = json.loads(raw)
        if "error" in rep:
            raise CellError(str(rep["error"]), str(rep.get("detail") or ""))
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
    is not (the module's docstring says so; a scored mission stops every process of the program's account)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            proc.wait(timeout=3)
            if sig == signal.SIGTERM:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)       # its children, if any are left
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
    """One of this script's output files (--transcript, --debrief), opened for writing with its folder made: before
    the mission is reserved, so that a path that cannot be written costs no mission. None when not asked for."""
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
    ap.add_argument("--transcript", default=None, help="write every request and reply (images elided) here, JSON lines")
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
    if not os.path.isfile(os.path.join(args.program, "patrol.py")):
        sys.exit("no patrol.py in %s" % args.program)
    # everything on this side that can fail is done BEFORE a mission is reserved (reset uses one mission of the
    # practice ground's allowance): the output files are opened (their folders made), the --keep folder is made,
    # and the program, the client library and the givens are copied; and whatever happens after reset, the program's
    # processes are stopped and the run's temporary folder removed
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
        sys.exit("cannot write the run's outputs (%s); no practice mission was used" % e)
    run = tempfile.mkdtemp(prefix="practice_run_")
    read_only = []
    procs = []
    try:
        prog = os.path.join(run, "program")
        copy_program(os.path.abspath(args.program), prog)
        lib = os.path.join(run, "lib")                     # the client library, read-only, as in a scored mission
        copy_program(LIB, lib)
        given = os.path.join(run, "given")             # the givens, read-only (ARB_GIVEN_DIR), as in a scored mission
        copy_program(GIVEN, given)
        read_only.extend((prog, lib, given))
        for d in read_only:
            set_modes(d, 0o555, 0o444)
        home, tmp = os.path.join(run, "home"), os.path.join(run, "tmp")
        os.makedirs(home)
        os.makedirs(tmp)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "TMPDIR": tmp, "TMP": tmp,
               "TEMP": tmp,
               "LANG": "C.UTF-8", "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "PYTHONPATH": lib,
               "MPLBACKEND": "Agg", "ARB_GIVEN_DIR": given,
               "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "LP_NUM_THREADS": "2"}
        try:
            with open(args.status) as f:
                cap = float(json.load(f).get("program_time_cap_s") or PROGRAM_TIME_S)
        except (OSError, ValueError, TypeError):
            cap = PROGRAM_TIME_S
        cpu_s = int(CPU_FACTOR * cap)

        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))

        ground = Ground(args.socket)
        first = ground.call("reset", {"site": str(args.site)})
        print("practice mission %d on site %s (simulated time limit %.0f s); missions left after this one: %d" % (
            first["mission"], first["site"], first.get("sim_time_limit_s", 0.0), first["missions_left"]), flush=True)
        proc = subprocess.Popen(launcher_argv(sys.executable, lib, prog), cwd=prog, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, bufsize=0, preexec_fn=limits, start_new_session=True)
        procs.append(proc)
        os.set_blocking(proc.stdin.fileno(), False)
        t0 = time.time()
        seen = {"requests": 0, "frames": 0, "sweeps": 0, "thermal": 0, "events": [], "sim_time": 0.0}
        over = {}                                      # the practice ground's last word (mission_over), once it came
        note = []                                          # what this script saw happen

        def last_word(raw):
            """The practice ground's line that ends the mission (`mission_over`: the reason and the debrief)."""
            try:
                obj = json.loads(raw) if raw and raw.strip() else None
            except ValueError:
                obj = None
            if isinstance(obj, dict) and obj.get("event") == "mission_over":
                over.update(obj)
            else:
                n = ("the practice ground closed the connection without its last word (the debriefs are saved in %s)"
                     % DEBRIEFS_DIR)
                if n not in note:
                    note.append(n)

        def wait_last_word():
            if not over:
                ground.half_close()                        # the end of this connection's requests ends the mission
                last_word(ground.line(timeout=LAST_WORD_S))

        def to_program(data):
            """A reply to the program, watching the practice ground meanwhile: the mission can end while the program
            does not read (a full pipe). None when written; "ground" when the mission ended meanwhile; "program" when
            the program cannot take it."""
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

        # wait on the program and on the practice ground at once: the ground can end the mission while the program is
        # silent or busy (a budget, 180 s of silence), and the program can exit without a word
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
                        program_open = False                 # the program exited (or closed its stdout)
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
                # as in a scored mission, a line that is not a request ends the mission: the practice ground says so
                note.append("the program sent a line that is not a request")
                try:
                    ground.send(line)
                except CellError:
                    pass
                wait_last_word()
                break
            seen["requests"] += 1
            if method in NOT_ON_THE_WIRE:
                reply_obj = {"id": req.get("id"), "error": "forbidden"}
            else:
                try:
                    ground.send(line)
                except CellError:
                    pass
                raw = ground.line()
                try:
                    reply_obj = json.loads(raw) if raw and raw.strip() else None
                except ValueError:
                    reply_obj = None
                if not isinstance(reply_obj, dict) or reply_obj.get("event") == "mission_over":
                    last_word(raw)
                    break
            res = reply_obj.get("result")
            if isinstance(res, dict):
                if method in ("get_camera", "get_depth") and "frame_id" in res:
                    seen["frames"] += 1
                if method == "get_lidar" and "ranges_f32_b64" in res:
                    seen["sweeps"] += 1
                if method == "get_thermal" and "frame_id" in res:
                    seen["thermal"] += 1
                if "sim_time" in res:
                    seen["sim_time"] = float(res["sim_time"])
                for e in res.get("events", []) if method == "step" else []:
                    seen["events"].append((round(float(e.get("t", 0.0)), 1), e.get("type")))
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
        rc_own = proc.poll()                               # how the program ended by itself, if it did
        stop_program(proc)                                 # the mission is over: nothing of the program outlives it
        wait_last_word()
        ground.close()
        if tr is not None:
            tr.close()
        cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_used = cpu.ru_utime + cpu.ru_stime
        if rc_own is None:
            how = "was stopped by this script at the mission's end"
        elif rc_own >= 0:
            how = "exited with code %d" % rc_own
        elif -rc_own in (signal.SIGXCPU, signal.SIGKILL) and cpu_used >= cpu_s - 1:
            how = "was stopped by its CPU-time limit (%d CPU-s used of %d)" % (cpu_used, cpu_s)
        else:
            how = "was killed by signal %d" % -rc_own
        debrief = over.get("debrief")
        reason = over.get("reason")                        # why the practice ground closed this mission's connection
        ended = debrief.get("ended") if isinstance(debrief, dict) else None     # why the mission itself ended
        if not reason:
            said = "(no word from the practice ground)"
        elif ended and ended != reason:
            said = "%s (the practice ground then closed the connection: %s)" % (ended, reason)
        else:
            said = reason
        print("end: the mission ended: %s%s; the program %s%s; sim %.1f s; wall %.0f s; requests %d; camera "
              "frames %d; lidar sweeps %d; thermal %d" % (
                  said, (" (%s)" % over["detail"]) if over.get("detail") else "", how, ("; " + "; ".join(note)) if note
                  else "", seen["sim_time"], time.time() - t0, seen["requests"], seen["frames"], seen["sweeps"],
                  seen["thermal"]))
        if seen["events"]:
            print("events: %s" % seen["events"][:40])
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
            stop_program(proc)                             # a no-op when it is already stopped
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
    except CellError as e:
        sys.exit("practice ground: %s" % e)
