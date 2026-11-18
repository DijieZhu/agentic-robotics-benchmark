"""Practice runs in the training tank and at the sister site.

    python3 run_session.py status                         the allowance left per facility
    python3 run_session.py tank   [options]               one training-tank run of your program
    python3 run_session.py sister [options]               one sister-site run of your program

The script connects to the practice facilities, starts one run (reset), imports intervene.py from --program
(default /home/user/submission) and calls its solve(vehicle) with the run's vehicle - the same calls as in an
evaluation run, plus vehicle.instruments() in the tank - then ends the run (finish) and writes what the run gave back
to --out (default /home/user/practice_runs): run_<facility>_<n>.json with the reset reply (the settings as run), the work
order, how the run ended, and in the tank the instrument record of the whole run (tracking-system pose of the vehicle
at 5 Hz in the tank frame, the six handle encoders, the receptacle lamps, the set points).  Anything else you want to
keep, your program records itself.  Every reset uses one run of that facility's allowance; at most two runs can be
open at a time (a third is refused with "busy" and costs nothing).

Tank settings (all optional; the operator's defaults in brackets):
    --current M_S             flow generator set speed, 0 .. 0.5 m/s along +x of the tank [0]
    --current-steps T:V,...   set-speed steps instead, e.g. 0:0.1,120:0.25,300:0.4 (seconds of the run : m/s)
    --bubbler L_MIN           diffuser gas flow, 0 (off) or 0.5 .. 2.9 L/min [0]
    --turbidity LEVEL         clear | medium | murky [clear]
    --port-bar BAR            test port TP-1 pressure, 0 .. 60 bar [0]
    --handle TAG=STATE        a training-panel handle as found, e.g. --handle PV2=closed (repeatable) [all open]
    --d1 WHERE                panel (in front of the port row, as drawn) | bay (next to D4) [panel]
    --stab-port PORT          where the hot stab is latched at the start: PARK-A | TP-1 [PARK-A]
    --setup JSON              the same settings as one JSON object (keys current_m_s, bubbler_L_min, turbidity,
                              test_port_bar, handles, d1_stand, stab_port)
Other options:
    --program DIR             folder with intervene.py (solve(vehicle))
    --out DIR                 where the run record goes
    --socket PATH             the facilities' socket (default $SUBSEACELL_PRACTICE_SOCKET or /run/subseacell/practice.sock)
    --lib DIR                 folder holding subseacell/client.py (default: ../lib next to this script, or
                              /home/user/given/lib)
"""

import argparse
import importlib
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def _lib_dir(arg):
    for c in (arg, os.environ.get("SUBSEACELL_LIB"), os.path.join(HERE, "..", "lib"), "/home/user/given/lib"):
        if c and os.path.isfile(os.path.join(c, "subseacell", "client.py")):
            return os.path.abspath(c)
    raise SystemExit("subseacell/client.py not found: give --lib")


def _status_path():
    sock = os.environ.get("SUBSEACELL_PRACTICE_SOCKET") or "/run/subseacell/practice.sock"
    return os.environ.get("SUBSEACELL_PRACTICE_STATUS") or os.path.join(os.path.dirname(sock), "practice_status.json")


def tank_setup(a):
    s = {}
    if a.setup:
        s.update(json.loads(a.setup))
    if a.current is not None:
        s["current_m_s"] = float(a.current)
    if a.current_steps:
        steps = []
        for part in a.current_steps.split(","):
            t, v = part.split(":")
            steps.append([float(t), float(v)])
        s["current_m_s"] = steps
    if a.bubbler is not None:
        s["bubbler_L_min"] = float(a.bubbler)
    if a.turbidity:
        s["turbidity"] = a.turbidity
    if a.port_bar is not None:
        s["test_port_bar"] = float(a.port_bar)
    for h in a.handle or ():
        tag, _, state = h.partition("=")
        s.setdefault("handles", {})[tag.strip()] = state.strip()
    if a.d1:
        s["d1_stand"] = a.d1
    if a.stab_port:
        s["stab_port"] = a.stab_port
    return s


def _connect(CL, path):
    """A practice Vehicle on its own socket (so the end-of-run line can be read after the program is done)."""
    import socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
    except OSError as e:
        sock.close()
        raise SystemExit("the practice facilities are not reachable at %s (%s)" % (path, e))
    rf = sock.makefile("r", encoding="utf-8", newline="\n")
    wf = sock.makefile("w", encoding="utf-8", newline="\n")

    def send(text):
        try:
            wf.write(text)
            wf.flush()
        except OSError as e:
            raise CL.CellError("closed", "the practice connection closed (%s)" % e)

    def recv():
        try:
            return rf.readline()
        except OSError:
            return ""
    v = CL.Vehicle(send, recv)

    def close():
        for f in (wf, rf):
            try:
                f.close()
            except OSError:
                pass
        sock.close()
    v._close = close
    return v


def run(facility, a):
    lib = _lib_dir(a.lib)
    if lib not in sys.path:
        sys.path.insert(0, lib)
    from subseacell import client as CL
    path = a.socket or os.environ.get("SUBSEACELL_PRACTICE_SOCKET") or CL.PRACTICE_SOCKET
    params = {"site": facility}
    if facility == "tank":
        params["setup"] = tank_setup(a)
    v = _connect(CL, path)
    rec = {"facility": facility, "requested": params, "wall_start": time.time()}
    try:
        reply = v.raw_call("reset", **params)
    except CL.CellError as e:
        v.close()
        raise SystemExit("reset refused: %s" % e)
    rec["reset"] = reply
    print("[practice] %s run %s started (%s runs of it left)" % (facility, reply.get("run"), reply.get("runs_left")),
          file=sys.stderr)
    try:
        rec["work_order"] = v.mission()
    except CL.CellError as e:
        rec["work_order"] = {"error": str(e)}
    prog = os.path.abspath(a.program)
    sys.path.insert(0, prog)
    t0 = time.time()
    try:
        mod = importlib.import_module("intervene")
        getattr(mod, "solve")(v)
        rec["program"] = "returned"
    except CL.CellError as e:
        rec["program"] = "cell error: %s" % e
    except Exception:  # noqa: BLE001
        rec["program"] = "exception"
        rec["traceback"] = traceback.format_exc()
        traceback.print_exc()
    rec["program_wall_s"] = round(time.time() - t0, 3)
    try:
        v.finish()
    except CL.CellError:
        pass
    if v.run_over is None:
        while True:
            line = v._recv()
            if not line:
                break
            try:
                msg = json.loads(line)
            except ValueError:
                break
            if msg.get("event") == "run_over":
                v.run_over, v.debrief = msg.get("reason"), msg.get("debrief")
                break
    v.close()
    rec["ended_by"] = v.run_over
    rec["instrument_record"] = v.debrief
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, "run_%s_%s.json" % (facility, reply.get("run")))
    with open(fn, "w") as f:
        json.dump(rec, f, indent=1)
    print("[practice] %s run %s ended (%s); record %s" % (facility, reply.get("run"), v.run_over, fn), file=sys.stderr)
    return fn


def main(argv=None):
    ap = argparse.ArgumentParser(description="practice runs in the training tank and at the sister site")
    ap.add_argument("facility", choices=["status", "tank", "sister"])
    ap.add_argument("--program", default="/home/user/submission")
    ap.add_argument("--out", default="/home/user/practice_runs")
    ap.add_argument("--socket", default=None)
    ap.add_argument("--lib", default=None)
    ap.add_argument("--setup", default=None)
    ap.add_argument("--current", type=float, default=None)
    ap.add_argument("--current-steps", default=None)
    ap.add_argument("--bubbler", type=float, default=None)
    ap.add_argument("--turbidity", default=None)
    ap.add_argument("--port-bar", type=float, default=None)
    ap.add_argument("--handle", action="append")
    ap.add_argument("--d1", default=None)
    ap.add_argument("--stab-port", default=None)
    a = ap.parse_args(argv)
    if a.facility == "status":
        p = _status_path()
        with open(p) as f:
            print(f.read())
        return 0
    if a.facility == "sister" and (a.setup or a.current is not None or a.current_steps or a.bubbler is not None
                                   or a.turbidity or a.port_bar is not None or a.handle or a.d1 or a.stab_port):
        raise SystemExit("the sister site has no settings")
    run(a.facility, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
