"""The practice facilities' service: the training tank and the sister site, for the team's development runs (task 07,
DESIGN v1.2 section 1.6; rulings JUDG 4.5, K4, U1; package P20).

An operator does not send a vehicle program to a damaged manifold that never ran in water: the vehicle's maker keeps a
training tank (a panel segment of the standard, a pressurised test port, a hot stab, a bubbler, a flow generator, three
water clarities, two stands of the dropped-object register, a code board, the cage, a tracking system) and the field
has a sister site (a pipeline end termination with a two-valve panel and a seep).  This service is both.  It runs the
scored runs' cell (subseacell: a byte copy without the reference program, the scoring module, the generator and any
scored site) on the two facility packages listed in sites.json (practice/facility.py: schema subseacell-practice/1,
guarded by path, sha256, schema and label), and answers exactly what a scored run answers - the vehicle's 17 calls -
plus reset(site, setup), which starts a run, and, in the tank only, instruments(): the tank's instrument readings
(tracking-system pose 5 Hz, the six handle encoders, the receptacles' status lamps, the set points).  The sister site
answers the 17 calls only: no truth, no debrief, no score.  Nothing anywhere is a score.

The image installs this folder for the practice account only (mode 700: the agent account can neither read nor list
it); the environment's setup starts it before the agent's session, under a loop that restarts it should it ever exit.
It listens on the unix socket $SUBSEACELL_PRACTICE_SOCKET (default /run/subseacell/practice.sock, mode 666).

A connection is one run.  Its first request is reset(site[, setup]) (site: "tank" or "sister", by name only; setup:
the tank's settings, facility.normalise_tank_setup).  The run then has the scored runs' budgets except the simulated
time (600 s tank, 1800 s sister): camera frames, pings, protective stops (the cell's), program time = wall clock since
the reset minus the cell's own work, and silence between requests (the service's).  When the run is over (a budget,
finish(), the vehicle disabled, ...) the service answers up to 50 more steps / 200 requests / 60 s, then sends
{"id": null, "event": "run_over", "reason": ..., "debrief": ...} and closes; the debrief is the tank's instrument
record of the whole run (also saved world-readable in $SUBSEACELL_PRACTICE_RECORDS/tank_run_<n>.json) and null at the
sister site.  Allowance: `runs` per facility in the whole session (counted at the start of a run, under an exclusive
lock; the count survives a restart of the service), at most `open` runs at a time over both facilities (one of `open`
slot locks held by the run's process; a run that cannot get one is answered busy and costs nothing); a run whose cell
cannot be built costs nothing either.  Status: $SUBSEACELL_PRACTICE_STATUS (world-readable JSON).
"""

import fcntl
import hashlib
import json
import os
import signal
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, ".."))


def _first_dir(*cands):
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return cands[-1]


CELL = os.environ.get("SUBSEACELL_PRACTICE_CELL") or _first_dir(os.path.join(HERE, "cell"), os.path.join(DEV, "cell"))
KIT = os.environ.get("SUBSEACELL_PRACTICE_KIT") or (HERE if os.path.isdir(os.path.join(HERE, "uwkit"))
                                                    else os.path.normpath(os.path.join(DEV, "..", "..", "shared")))
for _p in (KIT, CELL, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SOCKET = os.environ.get("SUBSEACELL_PRACTICE_SOCKET", "/run/subseacell/practice.sock")
STATUS = os.environ.get("SUBSEACELL_PRACTICE_STATUS", "/run/subseacell/practice_status.json")
RECORDS = os.environ.get("SUBSEACELL_PRACTICE_RECORDS", "/run/subseacell/tank_records")
STATE = os.environ.get("SUBSEACELL_PRACTICE_STATE", os.path.join(HERE, "state"))
SITES_DIR = os.environ.get("SUBSEACELL_PRACTICE_SITES", os.path.join(HERE, "sites"))
VALUES = os.environ.get("SUBSEACELL_TASK_VALUES") or (os.path.join(HERE, "task_values.json")
                                                      if os.path.isfile(os.path.join(HERE, "task_values.json"))
                                                      else os.path.join(DEV, "spec", "task_values.json"))
MAX_LINE = 4 * 1024 * 1024
MAX_CHILDREN = 16
# maintainer measurements only: pin run k (slot k) to the k-th CPU of this list (e.g. "5,17"); unset in the task image
PIN = [int(c) for c in os.environ.get("SUBSEACELL_PRACTICE_PIN", "").split(",") if c.strip()]


def _sites():
    with open(os.path.join(SITES_DIR, "sites.json"), encoding="utf-8") as f:
        return json.load(f)


SITES = _sites()
FACILITIES = {e["name"]: e for e in SITES["facilities"]}


def log(msg):
    try:
        os.makedirs(STATE, exist_ok=True)
        with open(os.path.join(STATE, "practiced.log"), "a") as f:
            f.write("%s [%d] %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), os.getpid(), msg))
    except OSError:
        pass


def _write_json(path, obj, mode):
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


# ============================================================================================ allowance and slots
class Ledger(object):
    """Runs per facility, kept under an exclusive lock (runs are forked processes)."""

    def __init__(self):
        os.makedirs(STATE, exist_ok=True)
        self.path = os.path.join(STATE, "ledger.json")
        self.lock = os.path.join(STATE, "ledger.lock")

    def _read(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = {}
        d.setdefault("used", {})
        d.setdefault("runs", [])
        return d

    def status(self, d=None):
        d = d or self._read()
        fac = {}
        for n, e in FACILITIES.items():
            used = int(d["used"].get(n, 0))
            fac[n] = {"runs": int(e["runs"]), "used": used, "left": max(0, int(e["runs"]) - used),
                      "sim_time_limit_s": float(e["max_sim_s"]), "about": e.get("about", "")}
        return {"facilities": fac, "open_at_a_time": int(SITES["open"]), "records": RECORDS}

    def publish(self):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            _write_json(STATUS, self.status(), 0o644)

    def left(self, name):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_SH)
            return self.status()["facilities"][name]["left"]

    def take(self, name, seed):
        """One run of facility `name` -> (run number over both facilities, status) or (None, status)."""
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            used = int(d["used"].get(name, 0))
            if used >= int(FACILITIES[name]["runs"]):
                return None, self.status(d)
            d["used"][name] = used + 1
            n = len(d["runs"]) + 1
            d["runs"].append({"run": n, "facility": name, "index": used + 1, "started": time.time(), "pid": os.getpid(),
                              "seed": str(seed)})
            _write_json(self.path, d, 0o600)
            st = self.status(d)
            _write_json(STATUS, st, 0o644)
            return n, st

    def close(self, run, **kw):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            for e in d["runs"]:
                if e.get("run") == run:
                    e.update(kw)
                    e["ended"] = time.time()
            _write_json(self.path, d, 0o600)


class Slots(object):
    """`open` slot locks: a run holds one (non-blocking flock) for as long as its process lives."""

    def __init__(self, n):
        os.makedirs(STATE, exist_ok=True)
        self.paths = [os.path.join(STATE, "slot%d.lock" % k) for k in range(int(n))]
        self.held = None

    def release(self):
        if self.held is not None:
            try:
                self.held[1].close()
            except OSError:
                pass
            self.held = None

    def acquire(self):
        if self.held is not None:
            return self.held[0]
        for k, p in enumerate(self.paths):
            f = open(p, "a")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            self.held = (k, f)
            return k
        return None


# ============================================================================================ the wire
class Conn(object):
    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def read(self, timeout):
        """The next request object; None on EOF; socket.timeout on silence; ValueError on a malformed line."""
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
        except RecursionError:
            raise ValueError("nesting too deep") from None
        if not isinstance(obj, dict) or "method" not in obj:
            raise ValueError("one request object per line")
        return obj

    def send(self, obj, timeout):
        self.sock.settimeout(timeout)
        self.sock.sendall((json.dumps(obj, allow_nan=False) + "\n").encode("utf-8"))


def _reply(rid, result):
    if isinstance(result, dict) and "error" in result and set(result) <= {"error", "detail"}:
        return dict({"id": rid}, **result)
    return {"id": rid, "result": result}


def _clean(PF, res, method):
    """No reply carries the words of facility.REPLY_WORDS (keys or text); one that would is not sent."""
    if PF.reply_words(res):
        log("reply of %s withheld: %s" % (method[:40], PF.reply_words(res)[:3]))
        return {"error": "internal_error"}
    return res


# ============================================================================================ one run
def _start(PF, SV, params, ledger, slots, conn_state):
    """reset(site, setup) -> (reply, server or None)."""
    if not isinstance(params, dict) or set(params) - {"site", "setup"}:
        return {"error": "bad_params", "detail": "reset(site, setup=None): site is one of %s" % ", ".join(FACILITIES)}, None
    name = params.get("site")
    if not isinstance(name, str) or name not in FACILITIES:
        return {"error": "bad_params", "detail": "site must be one of %s (by name)" % ", ".join(FACILITIES)}, None
    e = FACILITIES[name]
    try:
        fac = PF.FacilitySite.load(e["package"], SITES_DIR, e["sha256"])
    except (PF.GuardError, OSError) as ex:
        log("site %s refused by the guard: %s" % (name, ex))
        return {"error": "forbidden"}, None
    seed = SITES.get("seed")
    seed = int.from_bytes(os.urandom(8), "big") >> 1 if seed is None else int(seed) + int(e.get("seed_offset", 0))
    try:
        ses = fac.session(seed, params.get("setup"))
    except PF.SetupError as ex:
        return {"error": "bad_params", "detail": "setup: %s" % ex}, None
    if ledger.left(name) <= 0:
        return {"error": "practice_allowance_used", "detail": "all %d %s runs are used" % (int(e["runs"]), name)}, None
    slot = slots.acquire()
    if slot is None:
        return {"error": "busy", "detail": "at most %d practice runs at a time" % int(SITES["open"])}, None
    t_slot = time.time()
    if PIN and slot < len(PIN):
        try:
            os.sched_setaffinity(0, {PIN[slot]})
        except OSError:
            pass
    try:
        server = SV.SubseaCellServer(values=SV.BU.load_task_values(VALUES),
                                     config={"budgets": {"sim_time_s": float(e["max_sim_s"])}}, mode="practice",
                                     allow_reset=True, privileged=False, render=bool(SITES.get("render", True)),
                                     log=log, world_factory=PF.world_factory, practice_sites={},
                                     site_guard=PF.server_site_guard)
        server.reset(ses, seed=seed, kind="practice", name=name)
    except Exception as ex:  # noqa: BLE001 - a cell that cannot be built costs no run
        log("site %s: the cell could not be built: %r" % (name, ex))
        slots.release()
        return {"error": "unavailable", "detail": "this practice facility cannot be opened now"}, None
    run, st = ledger.take(name, seed)
    if run is None:
        server.close()
        slots.release()
        return {"error": "practice_allowance_used", "detail": "all %d %s runs are used" % (int(e["runs"]), name)}, None
    t0 = time.time()
    server.wall_start = t0
    conn_state.update(run=run, name=name, t0=t0, seed=seed, setup=ses.setup, slot=slot, t_slot=t_slot)
    log("run %d: %s (%s of %d), seed %d, setup %s" % (run, name, st["facilities"][name]["used"], int(e["runs"]), seed,
                                                     json.dumps(ses.setup, sort_keys=True)))
    left = st["facilities"][name]["left"]
    return {"ok": True, "site": name, "run": run, "runs_left": left, "sim_time_limit_s": float(e["max_sim_s"]),
            "setup": ses.setup}, server


def serve(sock, ledger):
    import facility as PF
    from subseacell import server as SV
    conn = Conn(sock)
    slots = Slots(SITES["open"])
    silence = float(SITES["silence_s"])
    cap = float(SITES["program_time_cap_s"])
    ad = SITES.get("after_done", {})
    server, reason, n_after, done_since = None, None, 0, None
    cs = {}
    try:
        while True:
            if server is not None:
                prog = (time.time() - cs["t0"]) - float(server.work_wall)
                if prog >= cap:
                    reason = "program_time_cap"
                    break
                wait = min(silence, cap - prog)
            else:
                wait = silence
            try:
                req = conn.read(wait)
            except socket.timeout:
                reason = "silence" if server is None or (time.time() - cs["t0"]) - float(server.work_wall) < cap \
                    else "program_time_cap"
                break
            except (ValueError, UnicodeDecodeError) as ex:
                reason = "protocol (%s)" % str(ex)[:60]
                break
            if req is None:
                reason = "eof"
                break
            rid, method, params = req.get("id"), req.get("method"), req.get("params")
            method = method if isinstance(method, str) else str(method)
            if method == "reset":
                if server is not None:
                    res = {"error": "one_run_per_connection", "detail": "this connection's run has begun: open a new one"}
                else:
                    res, server = _start(PF, SV, {} if params is None else params, ledger, slots, cs)
                    if res.get("error") == "busy":
                        conn.send(_reply(rid, res), timeout=silence)
                        reason = "busy"
                        break
            elif server is None:
                res = {"error": "no_mission", "detail": "a practice run starts with reset(site)"}
            else:
                res = server.handle(method, params)
            conn.send(_reply(rid, _clean(PF, res, method)), timeout=silence)
            if server is not None and server.done:
                if method == "finish":
                    reason = "finish"
                    break
                if done_since is None:
                    done_since = time.time()
                n_after += 1
                if (int(server.steps_after_done) > int(ad.get("max_steps", 50)) or n_after > int(ad.get("max_requests", 200))
                        or time.time() - done_since > float(ad.get("max_s", 60.0))):
                    reason = server.done_reason or "done"
                    break
                if method == "step":
                    server.steps_after_done += 1
    except (OSError, socket.timeout) as ex:
        reason = reason or "connection (%s)" % type(ex).__name__
    finally:
        if server is not None:
            _finish_run(PF, server, conn, ledger, cs, reason)
        try:
            sock.close()
        except OSError:
            pass


def _finish_run(PF, server, conn, ledger, cs, reason):
    t_end = time.time()
    server.wall_used = t_end - cs["t0"]
    server.program_used = server.wall_used - float(server.work_wall)
    if not server.done:
        server._end(reason or "closed")
    W = server.world
    debrief = None
    if getattr(W, "kind", None) == "tank":
        debrief = dict(W.instrument_log(), run=cs["run"], ended_by=reason, sim_time_s=round(float(W.sim.time), 3))
        try:
            os.makedirs(RECORDS, exist_ok=True)
            os.chmod(RECORDS, 0o755)
            _write_json(os.path.join(RECORDS, "tank_run_%03d.json" % cs["run"]), debrief, 0o644)
        except OSError as ex:
            log("run %d: tank record not saved: %r" % (cs["run"], ex))
    try:
        conn.send({"id": None, "event": "run_over", "reason": reason, "debrief": _clean(PF, debrief, "run_over")
                   if debrief is not None else None}, timeout=5.0)
    except (OSError, socket.timeout):
        pass
    tm = os.times()
    rec = server.finalize()
    cpu = float(tm.user + tm.system)
    log("run %d: %s ended (%s) at sim %.1f s, wall %.0f s, program %.0f s, cell work %.0f s, cpu %.1f s, frames %s, "
        "pings %d, illegal calls %d" % (cs["run"], cs["name"], reason, float(W.sim.time), server.wall_used,
                                        server.program_used, float(server.work_wall), cpu, rec.get("frames"),
                                        int(rec.get("pings", 0)), int(rec.get("illegal_call_count", 0))))
    ledger.close(cs["run"], end=reason, done_reason=server.done_reason, slot=cs["slot"], t_slot=cs["t_slot"],
                 sim_time_s=round(float(W.sim.time), 2), cpu_s=round(cpu, 2),
                 wall_s=round(server.wall_used, 2), program_s=round(server.program_used, 2),
                 cell_work_s=round(float(server.work_wall), 2), build_wall_s=round(float(server.build_wall_s), 2),
                 frames=rec.get("frames"), pings=int(rec.get("pings", 0)), modules=_cell_modules())
    server.close()


def _cell_modules():
    """The cell / generator modules this run's process has loaded (the service never imports the scoring module, the
    reference program or any generator; tests/test_practice.py checks this list)."""
    gen = ("facilities", "tag_textures", "panel_cad", "standin_manifold", "sitegen", "reference", "sample_frames")
    return sorted(m for m in sys.modules if m.startswith("subseacell") or m.split(".")[0] in gen)


# ============================================================================================ the service
def _busy(conn_sock):
    try:
        conn_sock.setblocking(False)
        conn_sock.send((json.dumps({"id": None, "error": "busy", "detail": "the service is busy"}) + "\n").encode())
    except OSError:
        pass
    try:
        conn_sock.close()
    except OSError:
        pass


def main():
    ledger = Ledger()
    os.makedirs(os.path.dirname(SOCKET) or ".", exist_ok=True)
    try:
        os.unlink(SOCKET)
    except FileNotFoundError:
        pass
    os.makedirs(RECORDS, exist_ok=True)
    os.chmod(RECORDS, 0o755)
    import facility as PF            # noqa: F401 - import the cell and the kit once, before forking
    from subseacell import server as SV  # noqa: F401
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET)
    os.chmod(SOCKET, 0o666)
    srv.listen(32)
    ledger.publish()
    log("listening on %s: %s, %d at a time, packages %s" % (
        SOCKET, ", ".join("%s %d runs" % (n, int(e["runs"])) for n, e in FACILITIES.items()), int(SITES["open"]),
        ", ".join("%s %s" % (e["package"], e["sha256"][:12]) for e in FACILITIES.values())))
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    children = set()
    while True:
        conn, _ = srv.accept()
        for pid in list(children):
            try:
                done, _st = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                done = pid
            if done:
                children.discard(pid)
        if len(children) >= MAX_CHILDREN:
            _busy(conn)
            continue
        pid = os.fork()
        if pid == 0:
            srv.close()
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            try:
                serve(conn, ledger)
            except Exception as ex:  # noqa: BLE001 - a run's failure never takes the service down
                log("connection failed: %r" % ex)
            finally:
                os._exit(0)
        children.add(pid)
        conn.close()


if __name__ == "__main__":
    main()
