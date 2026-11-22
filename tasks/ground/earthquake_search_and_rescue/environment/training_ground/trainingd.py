"""The training ground: the team's rubble training site, for the agent's development missions.

A response-robot team does not deploy an autonomy mode that never ran in rubble: it commissions it on the standard
test lanes and on training rubble piles. This service is that training ground. It runs the graded robot cell
(rubblecell/, a byte copy of the grader's tests/rubblecell/ without the reference program) on the sites listed in
sites.json: the test lanes and training sites, none of them a graded site or a candidate of the hidden set's
selection, and answers exactly what the graded cell answers (the robot's public calls), plus `reset(site)`, which
starts a mission on a site. It is a commissioning ground, not a rehearsal with an answer sheet: it never says where
the person was, how far off a report was, where the kit ended up relative to them, or what the mission would have
scored. When a mission ends, the robot's own mission log (how the mission ended, the simulated time, the reports made
and received by the team, the events the program already saw on the wire, the budgets used) is sent as the last line
of the connection and saved, world-readable, in /run/rubblecell/logs/. Nothing of the site (no map, no layout, no
position of anything) is ever sent.

environment/Dockerfile installs this folder under /opt/rubblecell-training for the account `rubblecell` only (mode 700:
the agent account can neither read nor list it); environment/setup/setup.sh starts it before the agent's session,
under a loop that restarts it should it ever exit. It listens on the unix socket /run/rubblecell/training.sock.

A connection is one mission. Its first request is reset(site) (a site name or its number, sites.json); the mission
then runs under the graded mission's budgets: the cell's own (simulated time, camera frames, lidar sweeps, thermal
frames) and the wire's (program time = the wall clock since the reset minus the cell's time inside step and the
sensors; silence between requests). When a budget ends the mission, the service sends {"event": "mission_over"} with
the mission log and closes the connection. The allowance: at most `missions` missions in the whole session (counted
from their reset; the count survives a restart of the service) and at most `open` at a time; the status is in
/run/rubblecell/training_status.json (world-readable).
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

SOCKET = os.environ.get("RUBBLECELL_TRAINING_SOCKET", "/run/rubblecell/training.sock")
STATUS = os.environ.get("RUBBLECELL_TRAINING_STATUS", "/run/rubblecell/training_status.json")
LOGS = os.environ.get("RUBBLECELL_TRAINING_LOGS", "/run/rubblecell/logs")
STATE = os.environ.get("RUBBLECELL_TRAINING_STATE", os.path.join(HERE, "state"))
SITES = json.load(open(os.path.join(HERE, "sites.json")))
PUBLIC = ("describe", "get_state", "drive", "set_flippers", "set_gauge", "set_mast", "set_head", "set_arm", "arm_to",
          "gripper", "step", "get_camera", "get_lidar", "get_thermal", "report_survivor", "finish")
AFTER_DONE_MAX_STEPS = 50
AFTER_DONE_MAX_REQUESTS = 200
AFTER_DONE_MAX_S = 60.0
MAX_LINE = 4 * 1024 * 1024


def log(msg):
    try:
        with open(os.path.join(STATE, "trainingd.log"), "a") as f:
            f.write("%s [%d] %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), os.getpid(), msg))
    except OSError:
        pass


def _write_json(path, obj, mode):
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _site_list():
    return [{"number": i + 1, "name": s["name"], "about": s.get("about", "")} for i, s in enumerate(SITES["sites"])]


class Ledger(object):
    """The training ground's mission count, kept under an exclusive lock (the missions run in forked children)."""

    def __init__(self):
        os.makedirs(STATE, exist_ok=True)
        self.path = os.path.join(STATE, "ledger.json")
        self.lock = os.path.join(STATE, "ledger.lock")

    def _read(self):
        try:
            with open(self.path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"missions_used": 0, "missions": []}

    def status(self, d=None):
        d = d or self._read()
        n = int(SITES["missions"])
        return {"training_missions": n, "missions_used": int(d["missions_used"]),
                "missions_left": max(0, n - int(d["missions_used"])), "open_at_a_time": int(SITES["open"]),
                "sites": _site_list(), "mission_logs": LOGS}

    def publish(self):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            _write_json(STATUS, self.status(), 0o644)

    def take(self, site_idx):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            if int(d["missions_used"]) >= int(SITES["missions"]):
                return None, self.status(d)
            d["missions_used"] = int(d["missions_used"]) + 1
            d["missions"].append({"mission": d["missions_used"], "site": SITES["sites"][site_idx]["name"],
                                  "started": time.time(), "pid": os.getpid()})
            _write_json(self.path, d, 0o600)
            st = self.status(d)
            _write_json(STATUS, st, 0o644)
            return d["missions_used"], st

    def close(self, mission, reason, sim_time):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            for e in d["missions"]:
                if e.get("mission") == mission:
                    e.update({"ended": time.time(), "end": reason, "sim_time_s": round(float(sim_time), 2)})
            _write_json(self.path, d, 0o600)


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


def _site_index(v):
    names = [s["name"] for s in SITES["sites"]]
    if isinstance(v, str) and v in names:
        return names.index(v)
    try:
        k = int(v)
    except (TypeError, ValueError):
        return -1
    return k - 1 if 1 <= k <= len(names) else -1


WIRE_EVENTS = ("grasped", "released", "slipped", "grip_slip", "kit_unlatched", "link_up", "link_down", "report_sent",
               "collapse", "protective_stop", "robot_overturned", "robot_fell")


def mission_log(server, mission, site_name, reason):
    """The robot's own log of the exercise: what the program could already read on the wire, and nothing of the site's
    truth (no position of the person, no error of a report, no distance of the kit, no score)."""
    ev = [[round(float(t), 1), e] for t, e in server.events_log if e in WIRE_EVENTS]
    return {
        "mission": mission, "site": site_name, "ended": reason, "sim_time_s": round(float(server.d.time), 1),
        "reports_made": len(server.reports), "reports_received_by_the_team": sum(1 for r in server.reports if r["sent"]),
        "kit_left_the_cradle": bool(not server.kit_latched), "robot_lost": server.lost,
        "events": ev[:400],
        "budgets_used": {"camera_frames": int(server.frames_used), "lidar_sweeps": int(server.lidar_used),
                         "thermal_frames": int(server.thermal_used)},
    }


def serve(sock, ledger):
    """One training mission on one connection."""
    conn = Conn(sock)
    silence = float(SITES["silence_s"])
    cap = float(SITES["program_time_cap_s"])
    server = None
    mission = None
    site_name = None
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
                    res = {"error": "one_mission_per_connection", "detail": "this connection's mission has begun: open a new one"}
                else:
                    k = _site_index(params.get("site"))
                    if k < 0:
                        res = {"error": "bad_params", "detail": "reset(site): one of %s (or its number)" %
                               ", ".join(s["name"] for s in SITES["sites"])}
                    else:
                        mission, st = ledger.take(k)
                        if mission is None:
                            res = {"error": "training_allowance_used", "detail": "all %d training missions are used" % st["training_missions"]}
                        else:
                            from rubblecell.server import RubbleCellServer
                            site = SITES["sites"][k]
                            site_name = site["name"]
                            server = RubbleCellServer(config=SITES["config"], mode="hidden", privileged=False, render=True, log=log)
                            server.reset(int(site["seed"]), site["kind"])
                            t0 = time.time()
                            server.wall_start = t0
                            log("mission %d: site %s" % (mission, site_name))
                            res = {"ok": True, "site": site_name, "mission": mission, "missions_left": st["missions_left"]}
            elif server is None:
                res = {"error": "no_mission", "detail": "a training mission starts with reset(site)"}
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
        if server is not None:
            if not server.done:
                server._end(reason or "closed")
            try:
                db = mission_log(server, mission, site_name, reason)
                os.makedirs(LOGS, exist_ok=True)
                _write_json(os.path.join(LOGS, "mission_%03d.json" % mission), db, 0o644)
            except Exception as e:  # noqa: BLE001
                db = {"error": "no mission log (%s)" % type(e).__name__}
            try:
                conn.send({"id": None, "event": "mission_over", "reason": reason, "mission_log": db}, timeout=5.0)
            except (OSError, socket.timeout):
                pass
            log("mission %d: ended (%s) at sim %.1f s, wall %.0f s" % (mission, reason, float(server.d.time), time.time() - t0))
            ledger.close(mission, reason, float(server.d.time))
            try:
                server.sensors.close()
            except Exception:  # noqa: BLE001
                pass
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
    os.makedirs(LOGS, exist_ok=True)
    os.chmod(LOGS, 0o755)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET)
    os.chmod(SOCKET, 0o666)
    srv.listen(16)
    ledger.publish()
    log("listening on %s: %d sites, %d missions, %d at a time" % (SOCKET, len(SITES["sites"]), SITES["missions"], SITES["open"]))
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
        if len(children) >= int(SITES["open"]):
            pid = os.fork()
            if pid == 0:
                srv.close()
                try:
                    c = Conn(conn)
                    req = c.read(10.0)
                    c.send({"id": (req or {}).get("id"), "error": "busy",
                            "detail": "at most %d training missions at a time" % SITES["open"]}, timeout=5.0)
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    os._exit(0)
            conn.close()
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            continue
        pid = os.fork()
        if pid == 0:
            srv.close()
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            try:
                serve(conn, ledger)
            except Exception as e:  # noqa: BLE001 - a mission's failure never takes the service down
                log("mission failed: %r" % e)
            finally:
                os._exit(0)
        children.add(pid)
        conn.close()


if __name__ == "__main__":
    main()
