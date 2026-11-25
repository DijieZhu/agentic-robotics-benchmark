"""The practice ground: the rescue team's calibration site and three practice sites, for the team's development runs.

A rescue team does not deploy an operating program that never ran on a real floor: the team keeps a calibration
site (every dimension surveyed and published: open gaps, a covered passage into a roofed room, a low slot, a mouth
with a back wall, a free-standing wall, gravel and dust strips, grid lines and crosses on a flat floor) and practice
sites like the real ones, and after each run the trainers debrief the team. This service is that practice ground. It
runs the scored episodes' station (voidcell/, a byte copy of the cell without the reference program and without any
token, plus voidcell/voidscore/, a byte copy of the scoring and the debrief) on the stored site packages listed in
sites.json, none of them a scored site or a candidate of the hidden set's selection, and answers exactly what a
scored episode answers (the station's public calls), plus `reset(site)`, which starts an episode on a site. When an
episode ends, the trainers' debrief is sent as the connection's last line ({"event": "episode_over", "reason",
"debrief"}) and saved, world-readable, in /run/voidcell/debriefs/:
  calibration site   the robot's six true node positions twice a second (in the roofed room too) and every primitive's
                     true SE(2) displacement;
  practice site      for every report the true distance, whether the dwell counted (and which condition failed), the
                     credit; whether the robot was retrieved; the stuck periods; the budget use; the events.
No total, and nothing of a practice site's layout or of the robot's path there, is ever sent (voidscore/debrief.py
SCHEMA: the keys are fixed).

The image installs this folder under /opt/voidcell-practice for the service's own account only (mode 700: the agent
account can neither read nor list it); environment/setup/setup.sh starts it before the agent's session, under a loop
that restarts it should it ever exit. It listens on the unix socket /run/voidcell/practice.sock (mode 666).

A connection is one episode. Its first request is reset(site) (a site name or its number, sites.json); the episode
then runs under the scored episodes' budgets: the cell's own (simulated time: the calibration site has its own limit;
camera frames; reports) and the wire's (program time = the wall clock since the reset minus the cell's time inside
step, get_frame, telemetry and pose; silence between requests). When a budget ends the episode, the service sends the
last line with the debrief and closes the connection. Every end of an episode sends it, with the reason: finish, a
budget of the cell or of the wire, the end of the connection's requests (the program exited), a line that is not a
request, and the service's own failure (`practice_ground_failure`, with the latest snapshot of the debrief, taken
every SNAPSHOT_EVERY_S of wall clock and marked `snapshot`). Every error code the service answers is one of the
station's (voidcell/_lib/protocol.py ERROR_CODES), with a detail text.
The allowance: at most `episodes` episodes in the whole session (counted from their reset; the count survives a
restart of the service) and at most `open` at a time; the status is in /run/voidcell/practice_status.json
(world-readable).
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

SOCKET = os.environ.get("VOIDCELL_PRACTICE_SOCKET", "/run/voidcell/practice.sock")
STATUS = os.environ.get("VOIDCELL_PRACTICE_STATUS", "/run/voidcell/practice_status.json")
DEBRIEFS = os.environ.get("VOIDCELL_PRACTICE_DEBRIEFS", "/run/voidcell/debriefs")
STATE = os.environ.get("VOIDCELL_PRACTICE_STATE", os.path.join(HERE, "state"))
SITES_FILE = os.environ.get("VOIDCELL_PRACTICE_SITES", os.path.join(HERE, "sites.json"))
SITES = json.load(open(SITES_FILE))
SITES_DIR = os.path.dirname(os.path.abspath(SITES_FILE))
AFTER_DONE_MAX_STEPS = 50
AFTER_DONE_MAX_REQUESTS = 200
AFTER_DONE_MAX_S = 60.0
MAX_LINE = 4 * 1024 * 1024
SNAPSHOT_EVERY_S = float(os.environ.get("VOIDCELL_PRACTICE_SNAPSHOT_S", "30"))
RUNNING = os.path.join(STATE, "running")
LAST_LINE_TIMEOUT_S = 60.0
EVENT = "episode_over"


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


def _limits():
    import voidcell  # noqa: F401
    from voidcell._lib import budgets
    return budgets.merged((SITES.get("config") or {}).get("limits"))


def _site_time_limit(s):
    return float(s.get("time_limit_s") or _limits()["time_limit_s"])


def _site_list():
    return [{"number": i + 1, "name": s["name"], "about": s.get("about", ""), "time_limit_s": _site_time_limit(s)}
            for i, s in enumerate(SITES["sites"])]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Ledger(object):
    """The practice ground's episode count, kept under an exclusive lock (the episodes run in forked children)."""

    def __init__(self):
        os.makedirs(STATE, exist_ok=True)
        self.path = os.path.join(STATE, "ledger.json")
        self.lock = os.path.join(STATE, "ledger.lock")

    def _read(self):
        try:
            with open(self.path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"episodes_used": 0, "episodes": []}

    def status(self, d=None):
        d = d or self._read()
        n = int(SITES["episodes"])
        lim = _limits()
        return {"practice_episodes": n, "episodes_used": int(d["episodes_used"]),
                "episodes_left": max(0, n - int(d["episodes_used"])), "open_at_a_time": int(SITES["open"]),
                "program_time_s": float(lim["program_time_s"]), "silence_s": float(lim["silence_s"]),
                "cpu_s": float(lim["cpu_s"]), "sites": _site_list(), "debriefs": DEBRIEFS}

    def episode_of(self, pid):
        d = self._read()
        hits = [e for e in d["episodes"] if e.get("pid") == pid and "ended" not in e]
        return hits[-1]["episode"] if hits else None

    def publish(self):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            _write_json(STATUS, self.status(), 0o644)

    def take(self, site_idx):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            if int(d["episodes_used"]) >= int(SITES["episodes"]):
                return None, self.status(d)
            d["episodes_used"] = int(d["episodes_used"]) + 1
            d["episodes"].append({"episode": d["episodes_used"], "site": SITES["sites"][site_idx]["name"],
                                  "started": time.time(), "pid": os.getpid()})
            _write_json(self.path, d, 0o600)
            st = self.status(d)
            _write_json(STATUS, st, 0o644)
            return d["episodes_used"], st

    def close(self, episode, reason, sim_time):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            for e in d["episodes"]:
                if e.get("episode") == episode:
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


def _acceptance():
    from voidcell.voidscore import scoring
    return scoring.load_acceptance()


def _open_episode(k):
    """A cell on practice site k (its stored package, checked against sites.json's sha256) and its debrief recorder."""
    import voidcell  # noqa: F401
    from voidcell._lib import budgets, mission
    from voidcell.server import VoidCellServer
    from voidcell.voidscore import debrief, scoring
    acc = _acceptance()
    scoring.check_cell_constants(mission, budgets, acc)
    site = SITES["sites"][k]
    pkg = os.path.join(SITES_DIR, site["package"])
    if site.get("sha256") and _sha256(pkg) != site["sha256"]:
        raise RuntimeError("practice package %s does not match sites.json" % site["package"])
    cfg = dict(SITES.get("config") or {})
    if cfg.get("plate_root"):                    # the plate bundles of the practice sites (plate_filament)
        cfg["plate_root"] = os.path.normpath(os.path.join(SITES_DIR, str(cfg["plate_root"])))
    limits = dict(cfg.get("limits") or {})
    limits["reports_per_point"] = int(acc["reports_per_point"])
    cfg["limits"] = limits
    server = VoidCellServer(config=cfg, mode="practice", privileged=False, allow_reset=False, log=log)
    # each practice episode draws its own set-down error and sensor noise (unless sites.json fixes a seed): two
    # episodes of one program on one site are two different runs, as on a real floor
    seed = site.get("noise_seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") >> 2
    server.reset(pkg, seed=seed, kind=site.get("kind", "practice"), sim_time_limit=_site_time_limit(site),
                 name=site["name"])
    scoring.check_site(server.site, acc)
    return server, site, debrief.Recorder(server), acc


def _kind(site):
    return "cal" if str(site.get("kind")) == "cal" else "practice"


def _debrief(server, rec, site, acc, episode, reason, snapshot=False):
    from voidcell.voidscore import debrief
    return debrief.build(server, rec, _kind(site), acc, episode=episode, ended_by=reason, snapshot=snapshot)


def _snapshot_path(pid=None):
    return os.path.join(RUNNING, "%d.json" % (os.getpid() if pid is None else pid))


def _keep_snapshot(server, rec, site, acc, episode):
    try:
        db = _debrief(server, rec, site, acc, episode, None, snapshot=True)
        os.makedirs(RUNNING, exist_ok=True)
        _write_json(_snapshot_path(), {"episode": episode, "sim_time_s": round(float(server.X.t()), 1), "debrief": db},
                    0o600)
    except Exception as e:  # noqa: BLE001 - a snapshot is a fallback, never a reason to end the episode
        log("episode %s: no snapshot: %r" % (episode, e))


def serve(sock, ledger):
    """One practice episode on one connection. Returns whether the episode's last line went out."""
    from voidcell._lib import protocol
    conn = Conn(sock)
    lim = _limits()
    silence = float(lim["silence_s"])
    cap = float(lim["program_time_s"])
    server = site = rec = acc = episode = reason = t0 = done_since = None
    n_after = 0
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
                params = {}
            if method == "reset" and not isinstance(params, dict):
                res = {"error": "bad_argument", "detail": "reset(site): params must be an object"}
            elif method == "reset":
                if server is not None:
                    res = {"error": "forbidden", "detail": "this connection's episode has begun: open a new connection"}
                else:
                    k = _site_index(params.get("site"))
                    if k < 0:
                        res = {"error": "bad_argument", "detail": "reset(site): one of %s (or its number)" %
                               ", ".join(s["name"] for s in SITES["sites"])}
                    elif ledger.status()["episodes_left"] <= 0:
                        res = {"error": "budget_exhausted",
                               "detail": "all %d practice episodes are used" % int(SITES["episodes"])}
                    else:
                        try:
                            cell = _open_episode(k)          # built first: a failure costs no episode
                        except Exception as e:  # noqa: BLE001
                            log("site %s: the cell could not be built: %r" % (SITES["sites"][k]["name"], e))
                            cell = None
                            res = {"error": "not_available", "detail": "this practice site cannot be opened now"}
                        if cell is not None:
                            episode, st = ledger.take(k)
                            if episode is None:
                                cell[0].close()
                                res = {"error": "budget_exhausted",
                                       "detail": "all %d practice episodes are used" % st["practice_episodes"]}
                            else:
                                server, site, rec, acc = cell
                                t0 = time.time()
                                server.wall_start = t0
                                log("episode %d: site %s, noise seed %s" % (episode, site["name"], server.seed))
                                res = {"ok": True, "site": site["name"], "episode": episode,
                                       "episodes_left": st["episodes_left"], "time_limit_s": float(server.time_limit)}
                                _keep_snapshot(server, rec, site, acc, episode)
                                last_snap = time.time()
            elif server is None:
                res = {"error": "not_available", "detail": "a practice episode starts with reset(site)"}
            elif method in protocol.PUBLIC_METHODS:
                res = server.handle(method, params)
            else:
                server.illegal_calls += 1
                res = {"error": "forbidden"}
            conn.send(_reply(rid, res), timeout=silence)
            if server is not None and not server.done and time.time() - last_snap >= SNAPSHOT_EVERY_S:
                _keep_snapshot(server, rec, site, acc, episode)
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
    except Exception as e:  # noqa: BLE001 - the service's own failure: the episode ends with this reason
        reason = "practice_ground_failure (%s)" % type(e).__name__
        log("episode %s: the service failed: %r" % (episode, e))
    finally:
        if server is not None:
            if t0 is not None:
                server.wall_used = time.time() - t0
                server.program_used = server.wall_used - float(server.work_wall)
            if not server.done:
                server._end(reason or "closed")
            try:
                db = _debrief(server, rec, site, acc, episode, reason)
                os.makedirs(DEBRIEFS, exist_ok=True)
                _write_json(os.path.join(DEBRIEFS, "episode_%03d.json" % episode), db, 0o644)
            except Exception as e:  # noqa: BLE001 - the latest snapshot, if one was kept, stands in for the debrief
                log("episode %s: no debrief: %r" % (episode, e))
                try:
                    with open(_snapshot_path()) as f:
                        db = dict(json.load(f).get("debrief") or {})
                except (OSError, ValueError):
                    db = {}
                db["ended_by"] = reason
                db["failure"] = {"what": "the debrief could not be computed (%s); this is the latest snapshot"
                                         % type(e).__name__}
            try:
                conn.send({"id": None, "event": EVENT, "reason": reason, "debrief": db}, timeout=LAST_LINE_TIMEOUT_S)
                sent_last = True
            except (OSError, socket.timeout):
                pass
            log("episode %d: ended (%s) at sim %.1f s, wall %.0f s" % (episode, reason, float(server.X.t()),
                                                                       time.time() - t0))
            ledger.close(episode, reason, float(server.X.t()))
            server.close()
        else:
            try:
                conn.send({"id": None, "event": EVENT, "reason": reason or "closed", "debrief": None}, timeout=5.0)
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
    """An episode's process died without its last line (killed, out of memory, a crash): the connection gets the
    last line with the reason practice_ground_failure and the latest snapshot of the debrief."""
    why = ("killed by signal %d" % os.WTERMSIG(status)) if os.WIFSIGNALED(status) else \
        ("exit status %d" % os.WEXITSTATUS(status))
    snap = None
    try:
        with open(_snapshot_path(pid)) as f:
            snap = json.load(f)
    except (OSError, ValueError):
        snap = None
    episode = (snap or {}).get("episode") or ledger.episode_of(pid)
    db = dict((snap or {}).get("debrief") or {})
    db["ended_by"] = "practice_ground_failure"
    db["failure"] = {"what": "the episode's process stopped (%s) before the episode's end" % why,
                     "debrief_as_of_sim_s": (snap or {}).get("sim_time_s")}
    reason = "practice_ground_failure"
    log("episode %s: its process %d stopped (%s); the snapshot of sim %s s goes out" % (
        episode, pid, why, (snap or {}).get("sim_time_s")))
    try:
        s2 = socket.socket(fileno=fd)
    except OSError:
        s2 = None
    if s2 is not None:
        try:
            Conn(s2).send({"id": None, "event": EVENT, "reason": reason, "detail": why, "debrief": db},
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
    if episode is not None:
        try:
            os.makedirs(DEBRIEFS, exist_ok=True)
            _write_json(os.path.join(DEBRIEFS, "episode_%03d.json" % episode), db, 0o644)
            ledger.close(episode, reason, float((snap or {}).get("sim_time_s") or 0.0))
        except Exception as e:  # noqa: BLE001
            log("episode %s: the failure's record: %r" % (episode, e))
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
    log("listening on %s: %d sites, %d episodes, %d at a time" % (SOCKET, len(SITES["sites"]), SITES["episodes"],
                                                                 SITES["open"]))
    children = {}
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
                            "detail": "at most %d practice episodes at a time" % SITES["open"]}, timeout=5.0)
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
            except Exception as e:  # noqa: BLE001 - an episode's failure never takes the service down
                log("episode failed: %r" % e)
            finally:
                os._exit(0 if ok else 3)
        children[pid] = os.dup(conn.fileno())
        conn.close()


if __name__ == "__main__":
    main()
