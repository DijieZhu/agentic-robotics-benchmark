"""practiced.py -- the rehearsal service of the training module L3.

A crew does not hand an autonomy program a station it has never flown in: the operator keeps a training module (L3,
a sister module of the laboratory modules, with one cabin air unit whose inlets can be fitted with either build, one
stowage unit, a training dock and lights) and runs rehearsal cards on it.  This service is that facility.  It runs the
scored runs' simulation cell (cabincell/, a byte copy of the task cell without the grader)
on a site drawn afresh for every run from one of five cards (rehearsal_cards.py; nothing of a scored site), and
answers exactly what a scored run answers (the robot's public calls), plus ``reset(card)``, which starts a run.  When
the run ends, the service sends the run's last line: why it ended, the observation sheet of the run
(rehearsal_debrief.py: categories only) and the station's record of the run (the live telemetry at 1 Hz, the station
requests, the advisories); the sheet is also saved, world-readable, in the debrief folder.  No score, no pose, no
sealed area, no item trajectory and nothing of the site's drawn values is ever sent.

The image installs this folder for the service's own account only (mode 700: the agent account can neither read nor
list it); the environment's setup starts the service before the agent's session, under a loop that restarts it
should it ever exit.  It listens on the unix socket SOCKET (mode 666).  The robot and station meshes are read through
the cell's ``assetroot`` from the same asset root that verify uses (CABINCELL_ASSET_ROOT or the image's
/opt/cabincell/assets): the image holds one copy.

A connection is one run.  Its first request is ``reset`` with params {"card": "EX-A" | "EX-B1" | "EX-B2" | "EX-C" |
"EX-D"}; the run then goes on under the scored runs' budgets (the cell's: simulated time -- here the run's reservation
--, frames, captures, reports; and the wire's: program time = the wall clock since the reset minus the cell's work,
silence between requests).  When a budget ends the run, the service sends {"event": "run_over"} and closes the
connection.  Every end of a run sends it, with the reason: the end of the connection's requests (the program exited),
the cell's time, program time, silence, a line that is not a request, and the service's own failure
(``rehearsal_failure``): should a run's process die (killed, out of memory), the service sends that reason with the
latest snapshot of the sheet (kept every SNAPSHOT_EVERY_S of wall clock, marked ``snapshot``).

The allowance (runtime site_spec ``rehearsal.budget``): ``sim_min`` simulated minutes in the whole session, at most
``per_run_max_min`` per run, at most ``max_runs`` runs, at most ``max_concurrent`` at a time.  A reset RESERVES
min(per-run maximum, what is left) under an exclusive lock (the ledger survives a restart of the service); the run's
simulated time is its reservation.  A run that ends normally is charged the simulated time its program used (the
settle after it is not charged) and the rest of the reservation goes back; a run that ends abnormally (the service's
failure, its process killed, a debrief that could not be made, a restart of the service) keeps the whole reservation
charged.  The run count is never given back.  The status is in STATUS (world-readable).

Adapted from the practice ground of the quadruped plant inspection task of this benchmark (its practiced.py): the
forked run per connection, the busy reply,
the ledger under flock, the snapshots and the fallback last line of a dead run.  Here: cards drawn per run instead
of stored sites, the simulated-minute allowance with reservation and charge, the observation sheet of categories
(no per-item truth), the station record, the cell of this task.
"""

import fcntl
import json
import math
import os
import signal
import socket
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SOCKET = os.environ.get("REHEARSAL_SOCKET", "/run/rehearsal/rehearsal.sock")
STATUS = os.environ.get("REHEARSAL_STATUS", "/run/rehearsal/status.json")
DEBRIEFS = os.environ.get("REHEARSAL_DEBRIEFS", "/run/rehearsal/debriefs")
STATE = os.environ.get("REHEARSAL_STATE", os.path.join(HERE, "state"))
RUNTIME = os.path.join(HERE, "cabincell", "data", "site_spec.json")
STATION_MAP = os.path.join(HERE, "station_map")
AFTER_DONE_MAX_REQUESTS = 200
AFTER_DONE_MAX_S = 60.0
MAX_LINE = 4 * 1024 * 1024
MIN_RUN_S = 10.0                    # a reservation shorter than this is not a run: the allowance is used up
SNAPSHOT_EVERY_S = float(os.environ.get("REHEARSAL_SNAPSHOT_S", "60"))
RUNNING = os.path.join(STATE, "running")
LAST_LINE_TIMEOUT_S = 60.0
FIRST_LINE_S = 30.0                 # the reset must come this soon after the connection


def log(msg):
    try:
        with open(os.path.join(STATE, "practiced.log"), "a") as f:
            f.write("%s [%d] %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), os.getpid(), msg))
    except OSError:
        pass


def _write_json(path, obj, mode):
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _runtime_values():
    import hashlib
    raw = open(RUNTIME, "rb").read()
    want = open(RUNTIME[:-len(".json")] + ".sha256", "rb").read().decode("ascii").split()[0]
    if hashlib.sha256(raw).hexdigest() != want:
        raise RuntimeError("the cell's runtime site_spec does not match its sha256 file")
    return json.loads(raw.decode("ascii"))["values"]


V = _runtime_values()
ALLOW = dict(V["rehearsal.budget"])
CARDS = tuple(V["rehearsal.cards"])
PROGRAM_TIME_CAP_S = float(V["cell.budget.program_compute_s"])
SILENCE_S = float(V["cell.budget.silence_s"])


# ================================================================================================ the ledger
class Ledger(object):
    """Runs and simulated minutes of the session, kept under an exclusive lock (the runs are forked children)."""

    def __init__(self, state=None):
        self.state = state or STATE
        os.makedirs(self.state, exist_ok=True)
        self.path = os.path.join(self.state, "ledger.json")
        self.lock = os.path.join(self.state, "ledger.lock")

    def _read(self):
        try:
            with open(self.path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"runs_used": 0, "charged_s": 0.0, "runs": []}

    def _write(self, d):
        _write_json(self.path, d, 0o600)

    def status(self, d=None):
        d = d or self._read()
        total = 60.0 * float(ALLOW["sim_min"])
        left = max(0.0, total - float(d["charged_s"]))
        open_ = sum(1 for e in d["runs"] if "ended" not in e)
        return {"cards": list(CARDS), "runs_total": int(ALLOW["max_runs"]), "runs_used": int(d["runs_used"]),
                "runs_left": max(0, int(ALLOW["max_runs"]) - int(d["runs_used"])),
                "sim_minutes_total": float(ALLOW["sim_min"]), "sim_minutes_left": round(left / 60.0, 2),
                "per_run_max_min": float(ALLOW["per_run_max_min"]), "at_a_time": int(ALLOW["max_concurrent"]),
                "open_runs": open_, "program_time_cap_s": PROGRAM_TIME_CAP_S, "silence_s": SILENCE_S,
                "debriefs": DEBRIEFS}

    def publish(self):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            _write_json(STATUS, self.status(), 0o644)

    def take(self, card, pid=None):
        """(run number, reserved simulated seconds, status) or (None, reason, status) -- atomic."""
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            if int(d["runs_used"]) >= int(ALLOW["max_runs"]):
                return None, "runs", self.status(d)
            left = 60.0 * float(ALLOW["sim_min"]) - float(d["charged_s"])
            if left < MIN_RUN_S:
                return None, "minutes", self.status(d)
            if sum(1 for e in d["runs"] if "ended" not in e) >= int(ALLOW["max_concurrent"]):
                return None, "busy", self.status(d)
            reserve = min(60.0 * float(ALLOW["per_run_max_min"]), left)
            d["runs_used"] = int(d["runs_used"]) + 1
            d["charged_s"] = float(d["charged_s"]) + reserve
            run = int(d["runs_used"])
            d["runs"].append({"run": run, "card": card, "pid": os.getpid() if pid is None else int(pid),
                              "started": time.time(), "reserved_s": reserve})
            self._write(d)
            st = self.status(d)
            _write_json(STATUS, st, 0o644)
            return run, reserve, st

    def close(self, run, reason, used_s=None, extra=None):
        """End of a run.  ``used_s`` (the program's simulated time) given = a normal end: the charge is
        min(reservation, used_s) and the rest goes back; None = an abnormal end: the reservation stays charged.
        ``extra``: the run's cost (CPU and wall seconds of its process), kept in the ledger entry."""
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            for e in d["runs"]:
                if e.get("run") == run and "ended" not in e:
                    res = float(e["reserved_s"])
                    if used_s is not None and math.isfinite(float(used_s)):
                        charge = min(res, max(0.0, float(used_s)))
                        d["charged_s"] = max(0.0, float(d["charged_s"]) - (res - charge))
                        e["charged_s"] = round(charge, 3)
                        e["refund_s"] = round(res - charge, 3)
                    else:
                        e["charged_s"] = res
                        e["refund_s"] = 0.0
                    e.update({"ended": time.time(), "end": str(reason)[:120]})
                    e.update(extra or {})
            self._write(d)
            _write_json(STATUS, self.status(d), 0o644)

    def cancel(self, run, why):
        """A run whose cell could not be built (nothing was simulated): its reservation and its count go back."""
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            for e in d["runs"]:
                if e.get("run") == run and "ended" not in e:
                    d["charged_s"] = max(0.0, float(d["charged_s"]) - float(e["reserved_s"]))
                    d["runs_used"] = max(0, int(d["runs_used"]) - 1)
                    e.update({"ended": time.time(), "end": "cancelled: " + str(why)[:100], "charged_s": 0.0,
                              "refund_s": float(e["reserved_s"]), "cancelled": True})
            self._write(d)
            _write_json(STATUS, self.status(d), 0o644)

    def run_of_pid(self, pid):
        d = self._read()
        hits = [e for e in d["runs"] if e.get("pid") == pid and "ended" not in e]
        return hits[-1]["run"] if hits else None

    def close_orphans(self, alive):
        """Runs left open by a previous service (their processes are gone): abnormal ends."""
        d = self._read()
        for e in d["runs"]:
            if "ended" not in e and e.get("pid") not in alive:
                self.close(e["run"], "rehearsal_failure (the service restarted)", None)


# ================================================================================================ the cell
def _install_cell():
    """Import the copy of the cell with the service's own measurement table in place of the grader."""
    import practice_measure
    import cabincell
    mod = types.ModuleType("cabincell.grade")
    mod.load_config = practice_measure.load_config
    mod.__doc__ = "the rehearsal service's stand-in (practice_measure.py): no grader in this copy of the cell"
    sys.modules["cabincell.grade"] = mod
    cabincell.grade = mod
    os.environ.setdefault("CABINCELL_STATION_MAP", STATION_MAP)
    import rehearsal_cards as RC
    from station import sites  # noqa: F401  (the builder whose colour table the training colours join)
    RC.install_colours()
    return practice_measure


def open_cell(card, reserve_s, run, nonce_hex=None):
    """(server, site info) of a new run: the card's site drawn afresh, the cell built with the renderer, the run's
    simulated time = its reservation.  The site folder is written in the state folder (the service's account only)."""
    PM = _install_cell()
    import rehearsal_cards as RC
    from cabincell import sensors_render as SR
    nonce_hex = nonce_hex or os.urandom(16).hex()
    site, info = RC.draw_site(card, nonce_hex)
    folder = os.path.join(STATE, "sites", "run_%03d" % run)
    RC.write_package(site, folder)
    srv = SR.make_server(config=PM.load_config(), limits={"sim_time_s": float(reserve_s)}, log=log)
    srv.reset(folder, seed=RC.derived(info["seed"], "cell"), label="rehearsal_run_%03d" % run)
    return srv, info


def run_section(server, card, run, reason, st, snapshot=False):
    c = server.cell
    rec = getattr(server, "_record", None)
    prog_end = (rec or {}).get("times", {}).get("program_end") if rec else None
    return {"card": card, "run": run, "ended_by": reason,
            "program_exit": (rec or {}).get("times", {}).get("exit") if rec else None,
            "sim_time_s": round(float(prog_end if prog_end is not None else c.t_mission()), 1),
            "sim_limit_s": round(float(c.budgets.sim_time_s), 1),
            "runs_left": st.get("runs_left"), "sim_minutes_left": st.get("sim_minutes_left"), "snapshot": bool(snapshot)}


def finish(server, card, run, reason, ledger):
    """The run's end: the cell's record (the settle runs here), the sheet, the station record; the ledger charges the
    program's simulated time.  Returns the last line's object."""
    import rehearsal_debrief as RD
    if not server.done:
        server._end(reason or "closed")
    rec = server.finalize(exit_code=getattr(server, "exit_code", None))
    used = float(rec["times"]["program_end"])
    sheet = RD.sheet(server.cell, rec)
    station = RD.station_record(server.cell)
    cost = getattr(server, "run_cost", None)
    if cost is not None:
        cost = dict(cpu_s=round(time.process_time() - cost[0], 2), wall_s=round(time.time() - cost[1], 2),
                    sim_s=round(float(server.cell.t_mission()), 2))
    ledger.close(run, reason, used, {"cost": cost} if cost else None)
    st = ledger.status()
    return {"id": None, "event": "run_over", "reason": reason, "run": run_section(server, card, run, reason, st),
            "debrief": sheet, "station_record": station}


# ================================================================================================ the wire
class Conn(object):
    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def read(self, timeout):
        deadline = time.time() + timeout
        while b"\n" not in self.buf:
            left = deadline - time.time()
            if left <= 0:
                raise socket.timeout()
            self.sock.settimeout(left)
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                return None
            self.buf += chunk
            if len(self.buf) > MAX_LINE and b"\n" not in self.buf:
                raise ValueError("line too long")
        line, self.buf = self.buf.split(b"\n", 1)
        try:
            obj = json.loads(line.decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - not JSON, not UTF-8, or nested too deep: a protocol end
            raise ValueError("not a JSON line (%s)" % type(e).__name__) from None
        if not isinstance(obj, dict) or "method" not in obj:
            raise ValueError("one request object per line")
        return obj

    def send(self, obj, timeout):
        self.sock.settimeout(timeout)
        self.sock.sendall((json.dumps(obj, default=_jsonable) + "\n").encode("utf-8"))


def _jsonable(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _plain_id(rid):
    """A request id as the reply echoes it: null, true / false, a 64-bit integer, a short string; else null."""
    if rid is None or isinstance(rid, bool):
        return rid
    if isinstance(rid, int) and -(1 << 63) <= rid < (1 << 63):
        return rid
    if isinstance(rid, str) and len(rid) <= 256:
        return rid
    return None


def _reply(rid, result):
    if isinstance(result, dict) and "error" in result and set(result) <= {"error", "detail"}:
        return dict({"id": rid}, **result)
    return {"id": rid, "result": result}


def _snapshot_path(pid=None):
    return os.path.join(RUNNING, "%d.json" % (os.getpid() if pid is None else pid))


def _keep_snapshot(server, card, run, ledger):
    """The running run's sheet so far, kept for the service should this process die."""
    try:
        import rehearsal_debrief as RD
        st = ledger.status()
        obj = {"run": run, "card": card, "sim_time_s": round(float(server.cell.t_mission()), 1),
               "run_section": run_section(server, card, run, "running", st, snapshot=True),
               "debrief": RD.snapshot_sheet(server.cell)}
        os.makedirs(RUNNING, exist_ok=True)
        _write_json(_snapshot_path(), obj, 0o600)
    except Exception as e:  # noqa: BLE001 - a snapshot is a fallback, never a reason to end the run
        log("run %s: no snapshot: %r" % (run, e))


def _save_debrief(obj, run):
    os.makedirs(DEBRIEFS, exist_ok=True)
    keep = {k: obj.get(k) for k in ("reason", "run", "debrief")}
    _write_json(os.path.join(DEBRIEFS, "run_%03d.json" % run), keep, 0o644)
    if obj.get("station_record") is not None:
        _write_json(os.path.join(DEBRIEFS, "run_%03d_station.json" % run), obj["station_record"], 0o644)


def serve(sock, ledger):
    """One rehearsal run on one connection.  Returns whether the run's last line went out (the parent sends a
    fallback when it did not)."""
    conn = Conn(sock)
    silence, cap = SILENCE_S, PROGRAM_TIME_CAP_S
    server = card = run = reason = t0 = None
    n_after, done_since = 0, None
    last_snap = time.time()
    sent_last = False
    try:
        while True:
            if server is not None:
                prog = (time.time() - t0) - float(server.work_wall)
                if prog >= cap:
                    reason = "program_time_cap"
                    break
                wait = min(silence, cap - prog)
            else:
                wait = FIRST_LINE_S
            try:
                req = conn.read(wait)
            except socket.timeout:
                reason = "silence" if server is None or (time.time() - t0) - float(server.work_wall) < cap \
                    else "program_time_cap"
                break
            except (ValueError, UnicodeDecodeError) as e:
                reason = "protocol (%s)" % str(e)[:60]
                break
            if req is None:
                reason = "eof"
                break
            method = req.get("method")
            method = method[:64] if isinstance(method, str) else "(not a name)"
            rid, params = _plain_id(req.get("id")), req.get("params")
            if params is None:
                params = {}
            if method == "reset":
                if server is not None:
                    res = {"error": "one_run_per_connection", "detail": "this connection's run has begun: open a new one"}
                elif not isinstance(params, dict) or params.get("card") not in CARDS:
                    res = {"error": "bad_params", "detail": "reset(card): card is one of %s" % ", ".join(CARDS)}
                else:
                    card = params["card"]
                    r, what, st = ledger.take(card)
                    if r is None:
                        res = {"error": {"runs": "rehearsal_runs_used", "minutes": "rehearsal_minutes_used",
                                         "busy": "busy"}[what],
                               "detail": {"runs": "all %d rehearsal runs are used" % int(ALLOW["max_runs"]),
                                          "minutes": "the rehearsal's simulated minutes are used",
                                          "busy": "at most %d rehearsal runs at a time" % int(ALLOW["max_concurrent"])}[what]}
                        card = None
                    else:
                        run, reserve = r, what
                        run_cost0 = (time.process_time(), time.time())
                        try:
                            cell, _info = open_cell(card, reserve, run)
                        except Exception as e:  # noqa: BLE001 - nothing was simulated: the run is not counted
                            log("run %d (%s): the cell could not be built: %r" % (run, card, e))
                            ledger.cancel(run, type(e).__name__)
                            res = {"error": "unavailable", "detail": "the training module cannot be opened now"}
                            run, card = None, None
                        else:
                            server = cell
                            server.run_cost = run_cost0
                            t0 = time.time()
                            server.wall_start = t0
                            st = ledger.status()
                            log("run %d: card %s, reserved %.0f s" % (run, card, reserve))
                            res = {"ok": True, "card": card, "run": run, "sim_time_limit_s": round(float(reserve), 1),
                                   "runs_left": st["runs_left"], "sim_minutes_left": st["sim_minutes_left"]}
                            _keep_snapshot(server, card, run, ledger)
                            last_snap = time.time()
            elif server is None:
                res = {"error": "no_run", "detail": "a rehearsal run starts with reset(card)"}
            else:
                res = server.handle(method, params)
            conn.send(_reply(rid, res), timeout=silence)
            if server is not None and not server.done and time.time() - last_snap >= SNAPSHOT_EVERY_S:
                _keep_snapshot(server, card, run, ledger)
                last_snap = time.time()
            if server is not None and server.done:
                if done_since is None:
                    done_since = time.time()
                n_after += 1
                if n_after > AFTER_DONE_MAX_REQUESTS or time.time() - done_since > AFTER_DONE_MAX_S:
                    reason = server.done_reason or "done"
                    break
    except (OSError, socket.timeout) as e:
        reason = reason or "connection (%s)" % type(e).__name__
    except Exception as e:  # noqa: BLE001 - the service's own failure: the run ends with this reason
        reason = "rehearsal_failure (%s)" % type(e).__name__
        log("run %s: the service failed: %r" % (run, e))
    finally:
        if server is not None:
            server.wall_used = time.time() - t0
            server.program_used = server.wall_used - float(server.work_wall)
            obj = None
            try:
                if str(reason).startswith("rehearsal_failure"):
                    raise RuntimeError("the service failed during the run")
                obj = finish(server, card, run, reason, ledger)
            except Exception as e:  # noqa: BLE001 - the latest snapshot stands in for the sheet; nothing goes back
                log("run %s: no debrief: %r" % (run, e))
                obj = _fallback(run, card, reason if str(reason).startswith("rehearsal_failure")
                                else "rehearsal_failure (%s)" % type(e).__name__, None, ledger)
            try:
                _save_debrief(obj, run)
            except Exception as e:  # noqa: BLE001
                log("run %s: the debrief could not be saved: %r" % (run, e))
            try:
                conn.send(obj, timeout=LAST_LINE_TIMEOUT_S)
                sent_last = True
            except (OSError, socket.timeout):
                pass
            log("run %s: ended (%s) at sim %.1f s, wall %.0f s" % (run, obj.get("reason"), float(server.cell.t_mission()),
                                                                  time.time() - t0))
            try:
                server.close()
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                conn.send({"id": None, "event": "run_over", "reason": reason or "closed", "debrief": None},
                          timeout=5.0)
                sent_last = True
            except (OSError, socket.timeout):
                pass
        try:
            os.unlink(_snapshot_path())
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
    return sent_last


def _fallback(run, card, reason, pid, ledger, detail=None):
    """The last line of a run that ended abnormally: the latest snapshot of its sheet; the ledger keeps the whole
    reservation charged."""
    snap = None
    try:
        with open(_snapshot_path(pid)) as f:
            snap = json.load(f)
    except (OSError, ValueError):
        snap = None
    run = run if run is not None else (snap or {}).get("run")
    card = card if card is not None else (snap or {}).get("card")
    if run is not None:
        ledger.close(run, reason, None)
    st = ledger.status()
    rs = dict((snap or {}).get("run_section") or {"card": card, "run": run, "program_exit": None, "sim_time_s": None,
                                                   "sim_limit_s": None, "snapshot": True})
    rs.update({"ended_by": reason, "runs_left": st["runs_left"], "sim_minutes_left": st["sim_minutes_left"], "snapshot": True})
    obj = {"id": None, "event": "run_over", "reason": reason, "run": rs, "debrief": (snap or {}).get("debrief"),
           "station_record": None}
    if detail:
        obj["detail"] = detail
    return obj


def _tell_failure(fd, pid, status, ledger):
    """A run's process died without its last line (killed, out of memory, a crash): the service sends the
    connection run_over with the reason rehearsal_failure and the latest snapshot, saves it, keeps the reservation."""
    why = ("killed by signal %d" % os.WTERMSIG(status)) if os.WIFSIGNALED(status) else \
        ("exit status %d" % os.WEXITSTATUS(status))
    run = ledger.run_of_pid(pid)
    obj = _fallback(run, None, "rehearsal_failure", pid, ledger,
                    detail="the run's process stopped (%s) before the run's end" % why)
    log("run %s: its process %d stopped (%s); the latest snapshot goes out" % (run, pid, why))
    try:
        s2 = socket.socket(fileno=fd)
    except OSError:
        s2 = None
    if s2 is not None:
        try:
            Conn(s2).send(obj, timeout=LAST_LINE_TIMEOUT_S)
        except (OSError, socket.timeout):
            pass
        try:
            s2.close()
        except OSError:
            pass
    else:
        try:
            os.close(fd)
        except OSError:
            pass
    if obj["run"].get("run") is not None:
        try:
            _save_debrief(obj, obj["run"]["run"])
        except Exception as e:  # noqa: BLE001
            log("run %s: the failure's record: %r" % (run, e))
    try:
        os.unlink(_snapshot_path(pid))
    except OSError:
        pass


def _reap(children, ledger):
    for pid in list(children):
        try:
            done, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            done, status = pid, 0
        if not done:
            continue
        fd = children.pop(pid)
        if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
            try:
                os.close(fd)
            except OSError:
                pass
        else:
            _tell_failure(fd, pid, status, ledger)


def _busy(conn, helpers):
    """Answer one connection's first request with 'busy' (in a short-lived child, so that a slow client never holds
    the service; the child is reaped later, ``helpers``)."""
    pid = os.fork()
    if pid == 0:
        try:
            c = Conn(conn)
            req = c.read(10.0)
            c.send({"id": _plain_id((req or {}).get("id")), "error": "busy",
                    "detail": "at most %d rehearsal runs at a time" % int(ALLOW["max_concurrent"])}, timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
        finally:
            os._exit(0)
    conn.close()
    helpers.add(pid)


def _reap_helpers(helpers):
    for pid in list(helpers):
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            done = pid
        if done:
            helpers.discard(pid)


def main(argv=None):
    ledger = Ledger()
    ledger.close_orphans(alive=set())
    for d in (os.path.dirname(SOCKET), DEBRIEFS, os.path.dirname(STATUS)):
        if d:
            os.makedirs(d, exist_ok=True)
    os.chmod(DEBRIEFS, 0o755)
    try:
        os.unlink(SOCKET)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET)
    os.chmod(SOCKET, 0o666)
    srv.listen(16)
    ledger.publish()
    log("listening on %s: cards %s, %d runs, %s simulated minutes, %d at a time" % (
        SOCKET, ",".join(CARDS), int(ALLOW["max_runs"]), ALLOW["sim_min"], int(ALLOW["max_concurrent"])))
    import numpy  # noqa: F401  (imported once here: the forked runs share it)
    children = {}                       # pid -> a copy of its connection's descriptor (for the fallback last line)
    helpers = set()                     # the short-lived children that answer 'busy'
    srv.settimeout(1.0)
    while True:
        _reap(children, ledger)
        _reap_helpers(helpers)
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        conn.settimeout(None)
        _reap(children, ledger)
        if len(children) >= int(ALLOW["max_concurrent"]):
            _busy(conn, helpers)
            continue
        pid = os.fork()
        if pid == 0:
            srv.close()
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            ok = False
            try:
                ok = serve(conn, ledger)
            except Exception as e:  # noqa: BLE001 - a run's failure never takes the service down
                log("run failed: %r" % e)
            finally:
                os._exit(0 if ok else 3)      # 3: the last line did not go out (the parent sends its fallback)
        children[pid] = os.dup(conn.fileno())
        conn.close()


if __name__ == "__main__":
    main()
