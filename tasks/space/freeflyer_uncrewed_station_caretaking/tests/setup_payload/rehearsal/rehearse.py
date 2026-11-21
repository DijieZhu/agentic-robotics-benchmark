#!/opt/venv/bin/python3
"""Rehearsal run: runs YOUR program in one run of a rehearsal card in laboratory module L3, the rehearsal facility
(never a scored module), the way a scored run runs it, and keeps what the run gave back.

    /opt/venv/bin/python3 ~/rehearsal/rehearse.py --list                    # the cards, the runs and minutes left
    /opt/venv/bin/python3 ~/rehearsal/rehearse.py --card EX-B1              # ~/submission on card EX-B1
    /opt/venv/bin/python3 ~/rehearsal/rehearse.py --card EX-D --out ~/rehearsals/d1 --transcript

The program (caretake.py with solve(robot), in --program, default ~/submission) runs as a scored run runs it, except
that it runs as you: from a read-only copy of its folder (regular files only), with read-only copies of the client
library (PYTHONPATH) and of the givens (ARB_GIVEN_DIR), an empty HOME and TMPDIR, the image's rendering settings,
the scored runs' launcher (``python -B -E -s`` through cabinkit.client) and their CPU cap (4 x the program-time
budget).  Its stdin / stdout carry the wire to the rehearsal service; its stderr goes to program.log in the run's
folder.  The service applies the scored runs' budgets itself (the run's simulated time, frames, captures, the report,
program time, silence) and ends the run when one runs out.  ``reset`` is this script's (the program's own reset
calls answer 'forbidden', as in a scored run).  Each run uses one of the rehearsal's runs and its simulated time
(reserved at the start, the unused part given back at a normal end).

What the run's folder (--out, default ~/rehearsals/run_<n>_<card>) holds when the run is over:
  program.log           the program's stderr (its prints)
  frames/               every camera frame the program received, as the service sent it: <id>_<camera>.png
                        (colour: 8-bit RGB; depth: 16-bit millimetres, 0 = no return; amplitude: <id>_<camera>_amp.png)
                        and frames.json (frame id, camera, time, size, intrinsics); --frames none keeps none.  A colour
                        frame takes about 3 MB; after 1 GB in one run the frames are listed but not written
  debrief.json          the observation sheet of the run (what a crew member watching it would tell you)
  station_record.json   the station's record of the run: the live telemetry at 1 Hz, the station requests with their
                        replies, the advisories raised during the run
  run.json              how the run ended (the service's reason, the program's exit), simulated time used and the
                        allowance left, this script's counts
  transcript.jsonl      with --transcript: every request and reply, images elided
The sheet and the station record are also saved in /run/rehearsal/debriefs/ (run_<n>.json, run_<n>_station.json).

The run's end ends this script too: when the service ends the run (a budget, silence) or the program exits, the
program's process group is stopped, the outputs are written and the script returns.  A process that leaves the group
(a daemon started with setsid, for instance) is not stopped by a rehearsal run and keeps running in your sandbox; in a
scored run every process of the program's account is stopped.  The run's folder is made, and the program copied,
before the run is reserved: a folder that cannot be written stops the script without using a run.  Keep this script
out of your submission folder: it starts a process, which the scored run's audit refuses.
"""

import argparse
import base64
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
SOCKET = os.environ.get("REHEARSAL_SOCKET", "/run/rehearsal/rehearsal.sock")
STATUS = os.environ.get("REHEARSAL_STATUS", "/run/rehearsal/status.json")
DEBRIEFS_DIR = "/run/rehearsal/debriefs"
PROGRAM_TIME_S = 600.0           # the program-time budget (the status file says it; this if not)
CPU_FACTOR = 4.0
LAST_WORD_S = 180.0              # the service's last line comes after the 60 s settle of the run's world
FRAME_BYTES_MAX = 1_000_000_000  # frames beyond this many bytes in one run are not written (run.json says so)
NOT_ON_THE_WIRE = ("reset",)


class RehearsalError(RuntimeError):
    def __init__(self, code, detail=""):
        RuntimeError.__init__(self, code + ((": " + detail) if detail else ""))
        self.code, self.detail = code, detail


def copy_tree(src, dst):
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d != "__pycache__" and not os.path.islink(os.path.join(dp, d))]
        rel = os.path.relpath(dp, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for fn in fns:
            p = os.path.join(dp, fn)
            if os.path.isfile(p) and not os.path.islink(p) and not fn.endswith(".pyc"):
                shutil.copy2(p, os.path.join(dst, rel, fn))


def set_modes(top, dir_mode, file_mode):
    os.chmod(top, dir_mode)
    for dp, dns, fns in os.walk(top):
        for d in dns:
            os.chmod(os.path.join(dp, d), dir_mode)
        for fn in fns:
            os.chmod(os.path.join(dp, fn), file_mode)


class Service(object):
    """This script's connection to the rehearsal service: whole lines, and whether a line is waiting."""

    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(path)
        except OSError as e:
            self.sock.close()
            raise RehearsalError("unreachable", "the rehearsal service is not reachable at %s (%s)" % (path, e))
        self.buf = b""
        self.eof = False

    def send(self, text):
        try:
            self.sock.sendall(text.encode("utf-8"))
        except OSError as e:
            raise RehearsalError("closed", "the rehearsal service closed the connection (%s)" % e)

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
            raise RehearsalError("closed", "the rehearsal service closed the connection")
        rep = json.loads(raw)
        if "error" in rep:
            raise RehearsalError(str(rep["error"]), str(rep.get("detail") or ""))
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
    """The program's process group (it runs in its own session) is stopped."""
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
        return {k: ("<%d bytes>" % len(v) if k.endswith("_b64") and isinstance(v, str) else elide(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [elide(v) for v in obj]
    return obj


class Frames(object):
    """The frames the program received, written as the service sent them."""

    def __init__(self, folder, keep):
        self.folder, self.keep = folder, keep
        self.index, self.bytes, self.dropped = [], 0, 0

    def add(self, res):
        if not isinstance(res, dict) or "frame_id" not in res:
            return
        row = {k: res.get(k) for k in ("frame_id", "camera", "t", "width", "height", "intrinsics", "depth_unit_m")
               if k in res}
        files = []
        if self.keep:
            for key, suffix in (("image_png_b64", ""), ("depth_png_b64", ""), ("amplitude_png_b64", "_amp")):
                b64 = res.get(key)
                if not isinstance(b64, str):
                    continue
                raw = base64.b64decode(b64)
                if self.bytes + len(raw) > FRAME_BYTES_MAX:
                    self.dropped += 1
                    continue
                name = "%04d_%s%s.png" % (int(res["frame_id"]), res.get("camera", "cam"), suffix)
                with open(os.path.join(self.folder, name), "wb") as f:
                    f.write(raw)
                self.bytes += len(raw)
                files.append(name)
        row["files"] = files
        self.index.append(row)

    def close(self):
        with open(os.path.join(self.folder, "frames.json"), "w") as f:
            json.dump(self.index, f, indent=1)


def _status(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", help="a rehearsal card: EX-A, EX-B1, EX-B2, EX-C or EX-D (--list)")
    ap.add_argument("--program", default=os.path.expanduser("~/submission"))
    ap.add_argument("--out", default=None, help="the run's folder (default ~/rehearsals/run_<n>_<card>)")
    ap.add_argument("--frames", choices=("all", "none"), default="all", help="write the frames the program received")
    ap.add_argument("--transcript", action="store_true", help="also write every request and reply (images elided)")
    ap.add_argument("--list", action="store_true", help="print the rehearsal's cards and allowance")
    ap.add_argument("--socket", default=SOCKET, help=argparse.SUPPRESS)
    ap.add_argument("--status", default=STATUS, help=argparse.SUPPRESS)
    ap.add_argument("--lib", default=LIB, help=argparse.SUPPRESS)
    ap.add_argument("--given", default=GIVEN, help=argparse.SUPPRESS)
    args = ap.parse_args()
    st = _status(args.status)
    if args.list or not args.card:
        print(json.dumps(st, indent=1) if st is not None else "rehearsal status unreadable (%s)" % args.status)
        return
    if not os.path.isfile(os.path.join(args.program, "caretake.py")):
        sys.exit("no caretake.py in %s" % args.program)
    # everything on this side that can fail is done BEFORE a run is reserved
    out = args.out
    if out is None:
        used = int((st or {}).get("runs_used", 0))
        out = os.path.expanduser("~/rehearsals/run_%02d_%s" % (used + 1, args.card))
        if os.path.exists(out):
            out += time.strftime("_%H%M%S")
    out = os.path.abspath(os.path.expanduser(out))
    try:
        os.makedirs(os.path.join(out, "frames"), exist_ok=True)
        if not os.access(out, os.W_OK | os.X_OK):
            raise OSError("%s is not writable" % out)
        plog = open(os.path.join(out, "program.log"), "wb")
        tr = open(os.path.join(out, "transcript.jsonl"), "w") if args.transcript else None
    except OSError as e:
        sys.exit("cannot write the run's folder (%s); no rehearsal run was used" % e)
    frames = Frames(os.path.join(out, "frames"), args.frames == "all")
    run = tempfile.mkdtemp(prefix="rehearsal_run_")
    read_only, procs = [], []
    try:
        prog = os.path.join(run, "program")
        copy_tree(os.path.abspath(args.program), prog)
        lib = os.path.join(run, "lib")
        copy_tree(args.lib, lib)
        given = os.path.join(run, "given")
        copy_tree(args.given, given)
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
        cap = float((st or {}).get("program_time_cap_s") or PROGRAM_TIME_S)
        cpu_s = int(CPU_FACTOR * cap)

        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))

        sys.path.insert(0, lib)
        from cabinkit.client import launcher_argv
        svc = Service(args.socket)
        first = svc.call("reset", {"card": str(args.card)})
        print("rehearsal run %d on card %s (simulated time %.0f s); runs left after this one: %d, simulated minutes "
              "left: %.1f" % (first["run"], first["card"], first["sim_time_limit_s"], first["runs_left"],
                              first["sim_minutes_left"]), flush=True)
        proc = subprocess.Popen(launcher_argv(sys.executable, lib, prog), cwd=prog, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=plog, bufsize=0, preexec_fn=limits, start_new_session=True)
        procs.append(proc)
        os.set_blocking(proc.stdin.fileno(), False)
        t0 = time.time()
        seen = {"requests": 0, "frames": 0, "sim_time": 0.0}
        over, note = {}, []

        def last_word(raw):
            try:
                obj = json.loads(raw) if raw and raw.strip() else None
            except ValueError:
                obj = None
            if isinstance(obj, dict) and obj.get("event") == "run_over":
                over.update(obj)
            elif "no last line from the service" not in note:
                note.append("no last line from the service (see %s)" % DEBRIEFS_DIR)

        def wait_last_word():
            if not over:
                svc.half_close()
                last_word(svc.line(timeout=LAST_WORD_S))

        def to_program(data):
            view = memoryview(data)
            while len(view):
                if b"\n" in svc.buf or svc.eof:
                    last_word(svc.line(timeout=0.0))
                    return "service"
                r, w, _ = select.select([svc.sock], [proc.stdin], [], None)
                if svc.sock in r:
                    raw = svc.line(timeout=0.0)
                    if raw is not None:
                        last_word(raw)
                        return "service"
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
        sel.register(svc.sock, selectors.EVENT_READ, "service")
        pbuf = b""
        program_open = True
        while True:
            if b"\n" in svc.buf or svc.eof:
                last_word(svc.line(timeout=0.0))
                break
            if b"\n" not in pbuf:
                if not program_open:
                    wait_last_word()
                    break
                ready = [key.data for key, _ in sel.select(timeout=None)]
                if "service" in ready:
                    raw = svc.line(timeout=0.0)
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
                    svc.send(line)
                except RehearsalError:
                    pass
                wait_last_word()
                break
            seen["requests"] += 1
            if method in NOT_ON_THE_WIRE:
                reply_obj = {"id": req.get("id"), "error": "forbidden"}
            else:
                try:
                    svc.send(line)
                except RehearsalError:
                    pass
                raw = svc.line()
                try:
                    reply_obj = json.loads(raw) if raw and raw.strip() else None
                except ValueError:
                    reply_obj = None
                if not isinstance(reply_obj, dict) or reply_obj.get("event") == "run_over":
                    last_word(raw)
                    break
            res = reply_obj.get("result")
            if isinstance(res, dict):
                if method == "camera" and "frame_id" in res:
                    seen["frames"] += 1
                    frames.add(res)
                if "t" in res and isinstance(res["t"], (int, float)):
                    seen["sim_time"] = float(res["t"])
            if tr is not None:
                tr.write(json.dumps({"t_wall": round(time.time() - t0, 3), "request": req, "reply": elide(reply_obj)}) + "\n")
            w = to_program((json.dumps(reply_obj) + "\n").encode("utf-8"))
            if w == "service":
                break
            if w == "program":
                note.append("the program stopped reading the wire")
                wait_last_word()
                break
        sel.close()
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=10 if not over else 0.5)
        except subprocess.TimeoutExpired:
            pass
        rc_own = proc.poll()
        stop_program(proc)
        wait_last_word()
        svc.close()
        frames.close()
        cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_used = cpu.ru_utime + cpu.ru_stime
        if rc_own is None:
            how = "was stopped by this script at the run's end"
        elif rc_own >= 0:
            how = "exited with code %d" % rc_own
        elif -rc_own in (signal.SIGXCPU, signal.SIGKILL) and cpu_used >= cpu_s - 1:
            how = "was stopped by its CPU-time limit (%d CPU-s used of %d)" % (cpu_used, cpu_s)
        else:
            how = "was killed by signal %d" % -rc_own
        summary = {"service": {"reason": over.get("reason"), "detail": over.get("detail"), "run": over.get("run")},
                   "program": how, "notes": note, "wall_s": round(time.time() - t0, 1), "requests": seen["requests"],
                   "frames_received": seen["frames"], "frames_written_bytes": frames.bytes,
                   "frames_not_written": frames.dropped}
        with open(os.path.join(out, "run.json"), "w") as f:
            json.dump(summary, f, indent=1)
        with open(os.path.join(out, "debrief.json"), "w") as f:
            json.dump(over.get("debrief"), f, indent=1)
        with open(os.path.join(out, "station_record.json"), "w") as f:
            json.dump(over.get("station_record"), f)
        rs = over.get("run") or {}
        print("end: %s; the program %s%s; simulated %s s of %s s; wall %.0f s; requests %d; frames %d" % (
            over.get("reason") or "(no word from the rehearsal service)", how, ("; " + "; ".join(note)) if note else "",
            rs.get("sim_time_s"), rs.get("sim_limit_s"), time.time() - t0, seen["requests"], seen["frames"]))
        print("debrief: %s" % json.dumps(over.get("debrief"), indent=1))
        print("the run's folder: %s" % out)
    finally:
        for proc in procs:
            stop_program(proc)
        for f in (plog, tr):
            if f is not None and not f.closed:
                f.close()
        for d in read_only:
            if os.path.isdir(d):
                set_modes(d, 0o755, 0o644)
        shutil.rmtree(run, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except RehearsalError as e:
        sys.exit("rehearsal service: %s" % e)
