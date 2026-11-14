"""The commissioning service: the series' cell on its development sites, for the agent's commissioning shifts (v0.6).

An automation engineer does not hand over a robot program that never ran on a real skid: it is commissioned on a
sister site of the series (factory and site acceptance tests, teach-in, tuning) and then deployed to the sites it will
run unattended. This service is that sister site. It runs the graded cell (skidcell/, a byte copy of the grader's
tests/skidcell/ without its client) on the development sites listed in sites.json, which are neither the graded sites
nor candidates of their selection, and answers exactly what the graded cell answers: the cell's public calls, plus
`reset(site)`, which starts a shift on development site `site`. It answers no ground truth, no score, no debrief.

environment/Dockerfile installs this folder under /opt/skidcell-commission for the account `skidcell` only (mode 700:
the agent account can neither read nor list it); environment/setup/setup.sh starts it before the agent's session, under
a loop that restarts it should it ever exit. It listens on the unix socket /run/skidcell/commission.sock (mode 666).

A connection is one commissioning shift. Its first request is reset(site) (1 .. the number of sites); the shift then
runs under the graded episode's budgets: the cell's own (1800 s of simulated time, 600 camera frames) and the wire's
(1200 s of program time = the wall clock since the reset minus the cell's time inside step and get_camera; 180 s of
silence; after the episode is over, 50 more steps, 200 requests or 60 s). When a budget ends the shift, the service
closes the connection, as the grader's wire does. The commissioning window: at most `shifts` shifts in the whole
session (a shift counts from its reset; the count survives a restart of the service, in state/ledger.json) and at
most `open` at a time. The budget is shown in /run/skidcell/commission_status.json (world-readable); the service's own
log (state/commissiond.log, readable by its account only) records the shifts' sites, times and end reasons, nothing of
the process or the score.

Wire: JSON lines {"id": n, "method": ..., "params": {...}} -> {"id": n, "result": ...} or {"id": n, "error": ...,
"detail": ...}, the graded cell's wire (skidcell.client.connect_commissioning speaks it).
"""

import fcntl
import json
import os
import signal
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SOCKET = os.environ.get("SKIDCELL_COMMISSION_SOCKET", "/run/skidcell/commission.sock")
STATUS = os.environ.get("SKIDCELL_COMMISSION_STATUS", "/run/skidcell/commission_status.json")
STATE = os.environ.get("SKIDCELL_COMMISSION_STATE", os.path.join(HERE, "state"))
GIVEN = os.environ.get("SKIDCELL_COMMISSION_GIVEN", os.path.join(HERE, "given"))   # holds panda/ (the robot model)
SITES = json.load(open(os.path.join(HERE, "sites.json")))
PUBLIC = ("describe", "get_state", "set_control", "command", "command_rail", "step", "get_camera", "log_reading", "finish")
AFTER_DONE_MAX_STEPS = 50
AFTER_DONE_MAX_REQUESTS = 200
AFTER_DONE_MAX_S = 60.0
MAX_LINE = 4 * 1024 * 1024


def log(msg):
    try:
        with open(os.path.join(STATE, "commissiond.log"), "a") as f:
            f.write("%s [%d] %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), os.getpid(), msg))
    except OSError:
        pass


def _write_json(path, obj, mode):
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


class Ledger(object):
    """The commissioning window's shift count, kept under an exclusive lock (the sessions run in forked children)."""

    def __init__(self):
        os.makedirs(STATE, exist_ok=True)
        self.path = os.path.join(STATE, "ledger.json")
        self.lock = os.path.join(STATE, "ledger.lock")

    def _read(self):
        try:
            with open(self.path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"shifts_used": 0, "shifts": []}

    def status(self, d=None):
        d = d or self._read()
        n = int(SITES["shifts"])
        return {"commissioning_shifts": n, "shifts_used": int(d["shifts_used"]), "shifts_left": max(0, n - int(d["shifts_used"])),
                "sites": len(SITES["seeds"]), "open_at_a_time": int(SITES["open"])}

    def publish(self):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            _write_json(STATUS, self.status(), 0o644)

    def take(self, site):
        """One shift of the window for `site` -> (shift number, status) or (None, status) when the window is used up."""
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            if int(d["shifts_used"]) >= int(SITES["shifts"]):
                return None, self.status(d)
            d["shifts_used"] = int(d["shifts_used"]) + 1
            d["shifts"].append({"shift": d["shifts_used"], "site": int(site), "started": time.time(), "pid": os.getpid()})
            _write_json(self.path, d, 0o600)
            st = self.status(d)
            _write_json(STATUS, st, 0o644)
            return d["shifts_used"], st

    def close(self, shift, reason, sim_time):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            for e in d["shifts"]:
                if e.get("shift") == shift:
                    e.update({"ended": time.time(), "end": reason, "sim_time_s": round(float(sim_time), 2)})
            _write_json(self.path, d, 0o600)


class Conn(object):
    """One client connection: JSON lines with a bounded length and a deadline per request."""

    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def read(self, timeout):
        """The next request object; None on EOF; raises socket.timeout on silence, ValueError on a malformed line."""
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
        obj = json.loads(line.decode("utf-8"))
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


def _reply(rid, result):
    if isinstance(result, dict) and "error" in result and set(result) <= {"error", "detail"}:
        return dict({"id": rid}, **result)
    return {"id": rid, "result": result}


def serve(sock, ledger):
    """One commissioning shift on one connection."""
    conn = Conn(sock)
    silence = float(SITES["silence_s"])
    cap = float(SITES["program_time_cap_s"])
    server = None
    shift = None
    reason = None
    t0 = None
    n_after = 0
    done_since = None
    try:
        while True:
            if server is not None:
                prog = (time.time() - t0) - float(server.work_wall)
                if prog >= cap:
                    reason = "program_time_cap"
                    break
                wait = min(silence, cap - prog)
            else:
                wait = silence
            try:
                req = conn.read(wait)
            except socket.timeout:
                reason = "silence" if server is None or (time.time() - t0) - float(server.work_wall) < cap else "program_time_cap"
                break
            except (ValueError, UnicodeDecodeError) as e:
                reason = "protocol (%s)" % str(e)[:60]
                break
            if req is None:
                reason = "eof"
                break
            rid, method, params = req.get("id"), str(req.get("method")), req.get("params")
            params = params if isinstance(params, dict) else {}
            if method == "reset":
                if server is not None:
                    res = {"error": "one_shift_per_connection", "detail": "this connection's shift has begun: open a new one"}
                else:
                    try:
                        site = int(params.get("site"))
                    except (TypeError, ValueError):
                        site = -1
                    if not 1 <= site <= len(SITES["seeds"]):
                        res = {"error": "bad_params", "detail": "reset(site): site is 1 .. %d (the development sites)" % len(SITES["seeds"])}
                    else:
                        shift, st = ledger.take(site)
                        if shift is None:
                            res = {"error": "commissioning_window_used", "detail": "all %d commissioning shifts are used" % st["commissioning_shifts"]}
                        else:
                            from skidcell.server import SkidCellServer
                            server = SkidCellServer(GIVEN, config=SITES["config"], mode="hidden", privileged=False, render=True, log=log)
                            server.reset(int(SITES["seeds"][site - 1]))
                            t0 = time.time()
                            server.wall_start = t0
                            log("shift %d: site %d" % (shift, site))
                            res = {"ok": True, "site": site, "site_name": server.layout["site"]["name"], "shift": shift,
                                   "shifts_left": st["shifts_left"]}
            elif server is None:
                res = {"error": "no_shift", "detail": "a commissioning shift starts with reset(site), site 1 .. %d" % len(SITES["seeds"])}
            elif method in PUBLIC:
                res = server.handle(method, params)
            else:
                server.illegal_calls += 1
                res = {"error": "forbidden"}
            conn.send(_reply(rid, res), timeout=silence)
            if server is not None and server.done:
                if method == "finish":
                    reason = "finish"
                    break
                if done_since is None:
                    done_since = time.time()
                n_after += 1
                if method == "step":
                    server.steps_after_done = getattr(server, "steps_after_done", 0) + 1
                if (getattr(server, "steps_after_done", 0) > AFTER_DONE_MAX_STEPS or n_after > AFTER_DONE_MAX_REQUESTS
                        or time.time() - done_since > AFTER_DONE_MAX_S):
                    reason = server.done_reason or "done"
                    break
    except (OSError, socket.timeout) as e:
        reason = reason or "connection (%s)" % type(e).__name__
    finally:
        if reason in ("program_time_cap", "silence") or (server is not None and server.done and reason not in ("finish", "eof")):
            try:                                # a budget ended the shift: say so before the connection closes
                conn.send({"id": None, "event": "shift_over", "reason": reason}, timeout=5.0)
            except (OSError, socket.timeout):
                pass
        if server is not None:
            if not server.done:
                server._end(reason or "closed")
            log("shift %d: ended (%s) at sim %.1f s, wall %.0f s, program %.0f s, frames %d" % (
                shift, reason, float(server.scene.data.time), time.time() - t0, (time.time() - t0) - float(server.work_wall),
                int(server.frames_used)))
            ledger.close(shift, reason, float(server.scene.data.time))
        try:
            sock.close()
        except OSError:
            pass


def main():
    ledger = Ledger()
    try:
        os.unlink(SOCKET)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET)
    os.chmod(SOCKET, 0o666)
    srv.listen(16)
    ledger.publish()
    log("listening on %s: %d sites, %d shifts, %d at a time" % (SOCKET, len(SITES["seeds"]), SITES["shifts"], SITES["open"]))
    children = set()
    busy = set()
    while True:
        conn, _ = srv.accept()
        for pid in list(children):
            try:
                done, _st = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                done = pid
            if done:
                children.discard(pid)
        for pid in list(busy):
            try:
                done, _st = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                done = pid
            if done:
                busy.discard(pid)
        if len(children) >= int(SITES["open"]):
            # as many shifts open as the service takes: the connection's first request is answered `busy` (by a child,
            # so that the service never waits on a client), and the connection is closed; no shift is used
            pid = os.fork()
            if pid == 0:
                srv.close()
                try:
                    c = Conn(conn)
                    req = c.read(10.0)
                    c.send({"id": (req or {}).get("id"), "error": "busy",
                            "detail": "at most %d commissioning shifts at a time" % SITES["open"]}, timeout=5.0)
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    os._exit(0)
            busy.add(pid)
            conn.close()
            continue
        pid = os.fork()
        if pid == 0:
            srv.close()
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            try:
                serve(conn, ledger)
            except Exception as e:  # noqa: BLE001 - a session's failure never takes the service down
                log("session failed: %r" % e)
            finally:
                os._exit(0)
        children.add(pid)
        conn.close()


if __name__ == "__main__":
    main()
