"""The practice ground: the operator's training hall and three practice buildings, for the team's development missions.

An inspection team does not deploy an autonomy mode that never ran on a real floor: the robot's operator keeps a
training hall (lanes of every obstacle class in several sizes, an instrument wall, a check source, surveyed grid
markers, a base station) and practice buildings like the real ones, and after each exercise the trainers debrief the
team. This service is that practice ground. It runs the scored missions' robot cell (plantcell/, a byte copy of the
cell without the reference program) on the stored site packages listed in sites.json, none of them a scored site
or a candidate of the hidden set's selection, and answers exactly what a scored mission answers (the robot's public
calls), plus `reset(site)`, which starts a mission on a site. When a mission ends, the trainers' debrief (per item of
the work order: delivered or not, the evidence accepted or which condition failed, the value's error, the item's
true position; the mark's and the source's errors; the events; the budget use; in the training hall also the
robot's true pose at 2 Hz) is sent as the last line of the connection and saved, world-readable, in
/run/plantcell/debriefs/. No total, and nothing of the buildings' as-built layout, is ever sent.

The image installs this folder under /opt/plantcell-practice for the account `plantcell` only (mode 700: the agent
account can neither read nor list it); environment/setup/setup.sh starts it before the agent's session, under a loop
that restarts it should it ever exit. It listens on the unix socket /run/plantcell/practice.sock.

A connection is one mission. Its first request is reset(site) (a site name or its number, sites.json); the mission
then runs under the scored missions' budgets: the cell's own (simulated time, sensor frames, reports; the training
hall is capped at its own simulated time) and the wire's (program time = the wall clock since the reset minus the
cell's time inside step and the sensors; silence between requests). When a budget ends the mission, the service
sends {"event": "mission_over"} with the debrief and closes the connection. Every end of a mission sends it, with
the reason: finish, a budget of the cell or of the wire, the end of the connection's requests (the program exited), a
line that is not a request, and the service's own failure: an error inside the service ends the mission with the
reason `practice_ground_failure`, and should a mission's process die (killed, out of memory) the service sends the
same reason with the latest snapshot of the debrief (taken every SNAPSHOT_EVERY_S of wall clock, marked `snapshot`).
The allowance: at most `missions`
missions in the whole session (counted from their reset; the count survives a restart of the service) and at most
`open` at a time; the status is in /run/plantcell/practice_status.json (world-readable).
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
sys.path.insert(0, HERE)

SOCKET = os.environ.get("PLANTCELL_PRACTICE_SOCKET", "/run/plantcell/practice.sock")
STATUS = os.environ.get("PLANTCELL_PRACTICE_STATUS", "/run/plantcell/practice_status.json")
DEBRIEFS = os.environ.get("PLANTCELL_PRACTICE_DEBRIEFS", "/run/plantcell/debriefs")
STATE = os.environ.get("PLANTCELL_PRACTICE_STATE", os.path.join(HERE, "state"))
SITES = json.load(open(os.path.join(HERE, "sites.json")))
PUBLIC = ("describe", "mission", "get_state", "get_camera", "get_depth", "get_thermal", "get_lidar", "walk",
          "set_mode", "set_body", "set_head", "set_lamp", "step", "report", "report_mark", "report_source", "finish")
WORK = ("step", "get_camera", "get_depth", "get_thermal", "get_lidar")
AFTER_DONE_MAX_STEPS = 50
AFTER_DONE_MAX_REQUESTS = 200
AFTER_DONE_MAX_S = 60.0
MAX_LINE = 4 * 1024 * 1024
# a running mission's debrief is kept this often (s of wall clock): it is what the team gets should the mission's
# process die (the service's own failure)
SNAPSHOT_EVERY_S = float(os.environ.get("PLANTCELL_PRACTICE_SNAPSHOT_S", "30"))
RUNNING = os.path.join(STATE, "running")
LAST_LINE_TIMEOUT_S = 60.0


def log(msg):
    try:
        with open(os.path.join(STATE, "practiced.log"), "a") as f:
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
    return [{"number": i + 1, "name": s["name"], "about": s.get("about", ""),
             "sim_time_limit_s": float(s.get("sim_time_limit_s", SITES["config"]["time"]["sim_time_limit_s"]))}
            for i, s in enumerate(SITES["sites"])]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Ledger(object):
    """The practice ground's mission count, kept under an exclusive lock (the missions run in forked children)."""

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
        return {"practice_missions": n, "missions_used": int(d["missions_used"]),
                "missions_left": max(0, n - int(d["missions_used"])), "open_at_a_time": int(SITES["open"]),
                "program_time_cap_s": float(SITES["program_time_cap_s"]), "silence_s": float(SITES["silence_s"]),
                "sites": _site_list(), "debriefs": DEBRIEFS}

    def mission_of(self, pid):
        """The mission a mission's process (its pid) took, from the ledger; None if it took none."""
        d = self._read()
        hits = [e for e in d["missions"] if e.get("pid") == pid and "ended" not in e]
        return hits[-1]["mission"] if hits else None

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
    """A request id as the reply echoes it (the rule of plantcell.wire_guard.plain_id): a short string, an
    integer or null."""
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


def _site_index(v):
    names = [s["name"] for s in SITES["sites"]]
    if isinstance(v, str) and v in names:
        return names.index(v)
    if isinstance(v, bool):
        return -1
    try:
        k = int(v)
    except (TypeError, ValueError, OverflowError):
        return -1
    return k - 1 if 1 <= k <= len(names) else -1


def _open_mission(k, mission):
    """A cell on practice site k (its stored package, checked against sites.json's sha256)."""
    from plantcell.server import PlantCellServer
    site = SITES["sites"][k]
    pkg = os.path.join(HERE, site["package"])
    if site.get("sha256") and _sha256(pkg) != site["sha256"]:
        raise RuntimeError("practice package %s does not match sites.json" % site["package"])
    server = PlantCellServer(config=SITES["config"], mode="practice", privileged=False, log=log)
    # each practice mission draws its own sensor noise and set-down error (unless sites.json fixes a seed): two
    # missions of one program on one site are two different runs, as on a real floor
    seed = site.get("noise_seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") >> 2
    server.reset(pkg, seed=seed, kind=site.get("kind", "practice"), sim_time_limit=site.get("sim_time_limit_s"),
                 name=site["name"])
    return server, site


def _snapshot_path(pid=None):
    return os.path.join(RUNNING, "%d.json" % (os.getpid() if pid is None else pid))


def _keep_snapshot(server, site, mission):
    """The running mission's debrief so far, kept for the service should this process die."""
    try:
        hall = str(site.get("kind")) == "hall"
        db = server.debrief(mission=mission, true_pose_hz=2.0 if hall else None, snapshot=True)
        os.makedirs(RUNNING, exist_ok=True)
        _write_json(_snapshot_path(), {"mission": mission, "sim_time_s": round(float(server.world.time), 1), "debrief": db},
                    0o600)
    except Exception as e:  # noqa: BLE001 - a snapshot is a fallback, never a reason to end the mission
        log("mission %s: no snapshot: %r" % (mission, e))


def serve(sock, ledger):
    """One practice mission on one connection. Returns whether the mission's last line (mission_over with the
    debrief) went out; the service sends a fallback when it did not."""
    conn = Conn(sock)
    silence = float(SITES["silence_s"])
    cap = float(SITES["program_time_cap_s"])
    server = None
    site = None
    mission = None
    reason = None
    t0 = None
    n_after = 0
    done_since = None
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
                wait = silence
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
                params = {}                 # anything else that is not an object is the cell's bad_params
            if method == "reset" and not isinstance(params, dict):
                res = {"error": "bad_params", "detail": "reset(site): params must be an object"}
            elif method == "reset":
                if server is not None:
                    res = {"error": "one_mission_per_connection",
                           "detail": "this connection's mission has begun: open a new one"}
                else:
                    k = _site_index(params.get("site"))
                    if k < 0:
                        res = {"error": "bad_params", "detail": "reset(site): one of %s (or its number)" %
                               ", ".join(s["name"] for s in SITES["sites"])}
                    elif ledger.status()["missions_left"] <= 0:
                        res = {"error": "practice_allowance_used",
                               "detail": "all %d practice missions are used" % int(SITES["missions"])}
                    else:
                        try:
                            cell, site_k = _open_mission(k, None)      # built first: a failure costs no mission
                        except Exception as e:  # noqa: BLE001
                            log("site %s: the cell could not be built: %r" % (SITES["sites"][k]["name"], e))
                            cell = None
                            res = {"error": "unavailable", "detail": "this practice site cannot be opened now"}
                        if cell is not None:
                            mission, st = ledger.take(k)
                            if mission is None:
                                cell.close()
                                res = {"error": "practice_allowance_used",
                                       "detail": "all %d practice missions are used" % st["practice_missions"]}
                            else:
                                server, site = cell, site_k
                                t0 = time.time()
                                server.wall_start = t0
                                log("mission %d: site %s, noise seed %s" % (mission, site["name"], server.seed))
                                res = {"ok": True, "site": site["name"], "mission": mission,
                                       "missions_left": st["missions_left"],
                                       "sim_time_limit_s": float(server.world.sim_time_limit)}
                                _keep_snapshot(server, site, mission)     # a snapshot from the start
                                last_snap = time.time()
            elif server is None:
                res = {"error": "no_mission", "detail": "a practice mission starts with reset(site)"}
            elif method in PUBLIC:
                res = server.handle(method, params)
            else:
                server.illegal_calls += 1
                res = {"error": "forbidden"}
            conn.send(_reply(rid, res), timeout=silence)
            if server is not None and not server.done and time.time() - last_snap >= SNAPSHOT_EVERY_S:
                _keep_snapshot(server, site, mission)
                last_snap = time.time()
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
    except Exception as e:  # noqa: BLE001 - the service's own failure: the mission ends with this reason
        reason = "practice_ground_failure (%s)" % type(e).__name__
        log("mission %s: the service failed: %r" % (mission, e))
    finally:
        if server is not None:
            if t0 is not None:
                server.wall_used = time.time() - t0
                server.program_used = server.wall_used - float(server.work_wall)
            if not server.done:
                server._end(reason or "closed")
            try:
                hall = str(site.get("kind")) == "hall"
                db = server.debrief(mission=mission, true_pose_hz=2.0 if hall else None)
                db["ended_by"] = reason
                os.makedirs(DEBRIEFS, exist_ok=True)
                _write_json(os.path.join(DEBRIEFS, "mission_%03d.json" % mission), db, 0o644)
            except Exception as e:  # noqa: BLE001 - the latest snapshot, if one was kept, stands in for the debrief
                log("mission %s: no debrief: %r" % (mission, e))
                try:
                    with open(_snapshot_path()) as f:
                        db = dict(json.load(f).get("debrief") or {})
                except (OSError, ValueError):
                    db = {}
                db["ended_by"] = reason
                db["failure"] = {"what": "the debrief could not be computed (%s); this is the latest snapshot"
                                         % type(e).__name__}
            try:
                conn.send({"id": None, "event": "mission_over", "reason": reason, "debrief": db},
                          timeout=LAST_LINE_TIMEOUT_S)
                sent_last = True
            except (OSError, socket.timeout):
                pass
            log("mission %d: ended (%s) at sim %.1f s, wall %.0f s" % (mission, reason, float(server.world.time),
                                                                       time.time() - t0))
            ledger.close(mission, reason, float(server.world.time))
            server.close()
        else:
            # no mission began on this connection: its last word is the reason (no debrief exists)
            try:
                conn.send({"id": None, "event": "mission_over", "reason": reason or "closed", "debrief": None},
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


def _tell_failure(fd, pid, status, ledger):
    """A mission's process died without its last line (killed, out of memory, a crash): the service sends the
    connection mission_over with the reason practice_ground_failure and the latest snapshot of the debrief, saves it
    and closes the ledger's entry."""
    why = ("killed by signal %d" % os.WTERMSIG(status)) if os.WIFSIGNALED(status) else \
        ("exit status %d" % os.WEXITSTATUS(status))
    snap = None
    try:
        with open(_snapshot_path(pid)) as f:
            snap = json.load(f)
    except (OSError, ValueError):
        snap = None
    mission = (snap or {}).get("mission") or ledger.mission_of(pid)
    db = dict((snap or {}).get("debrief") or {})
    db["ended_by"] = "practice_ground_failure"
    db["failure"] = {"what": "the mission's process stopped (%s) before the mission's end" % why,
                     "debrief_as_of_sim_s": (snap or {}).get("sim_time_s")}
    reason = "practice_ground_failure"
    log("mission %s: its process %d stopped (%s); the snapshot of sim %s s goes out" % (
        mission, pid, why, (snap or {}).get("sim_time_s")))
    try:
        s2 = socket.socket(fileno=fd)
    except OSError:
        s2 = None
    if s2 is not None:
        try:
            Conn(s2).send({"id": None, "event": "mission_over", "reason": reason, "detail": why, "debrief": db},
                          timeout=LAST_LINE_TIMEOUT_S)
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
    if mission is not None:
        try:
            db["mission"] = mission
            os.makedirs(DEBRIEFS, exist_ok=True)
            _write_json(os.path.join(DEBRIEFS, "mission_%03d.json" % mission), db, 0o644)
            ledger.close(mission, reason, float((snap or {}).get("sim_time_s") or 0.0))
        except Exception as e:  # noqa: BLE001
            log("mission %s: the failure's record: %r" % (mission, e))
    try:
        os.unlink(_snapshot_path(pid))
    except OSError:
        pass


def _reap(children, ledger):
    """Collect the missions' processes that ended; a failed one gets the service's fallback last line."""
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


def main():
    ledger = Ledger()
    try:
        os.unlink(SOCKET)
    except FileNotFoundError:
        pass
    os.makedirs(DEBRIEFS, exist_ok=True)
    os.chmod(DEBRIEFS, 0o755)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET)
    os.chmod(SOCKET, 0o666)
    srv.listen(16)
    ledger.publish()
    log("listening on %s: %d sites, %d missions, %d at a time" % (SOCKET, len(SITES["sites"]), SITES["missions"],
                                                                 SITES["open"]))
    children = {}                       # pid -> a copy of its connection's descriptor (for the fallback last line)
    srv.settimeout(1.0)
    while True:
        _reap(children, ledger)
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        conn.settimeout(None)
        _reap(children, ledger)
        if len(children) >= int(SITES["open"]):
            pid = os.fork()
            if pid == 0:
                srv.close()
                try:
                    c = Conn(conn)
                    req = c.read(10.0)
                    c.send({"id": (req or {}).get("id"), "error": "busy",
                            "detail": "at most %d practice missions at a time" % SITES["open"]}, timeout=5.0)
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
            ok = False
            try:
                ok = serve(conn, ledger)
            except Exception as e:  # noqa: BLE001 - a mission's failure never takes the service down
                log("mission failed: %r" % e)
            finally:
                os._exit(0 if ok else 3)      # 3: the last line did not go out (the service sends its fallback)
        children[pid] = os.dup(conn.fileno())
        conn.close()


if __name__ == "__main__":
    main()
