"""The commissioning service: the task's cell on its development kits, for the agent's commissioning window.

A controller for an assembly cell is not handed over before it has run on the cell: it is commissioned there on
pre-series kits, in a window the plant grants. This service is that window. It runs the grader's own simulator
(keyed_env.py in this folder is a byte copy of tests/keyed_env.py) on the development kits of kits.json, which are
neither graded kits nor candidates of the graded set's selection, and answers what the cell itself gives a
controller: the observations of the public wire (camera images on request, joint readings, the wrist wrench) and how
an episode ended (its steps run out, a safety stop, a budget). It answers no ground truth (no nut or peg pose, no
groove), no score, no completion verdict and no debrief, and it holds no code that could compute one: the credit rule
lives only in the grader, and an episode here never ends because a kit is complete.

environment/Dockerfile installs this folder under /opt/keyedcell-commission for the account `keyedcell` only (mode
700: the agent account can neither read nor list it); environment/setup/setup.sh starts it before the agent's
session, under a loop that restarts it should it ever exit. It listens on the unix socket
/run/keyedcell/commission.sock (mode 666).

A connection is one commissioning episode. Its first request is reset(kit) (1 .. the number of development kits):
the kit is laid out the way it is every time (its own layout and noise seed), and the reply carries the first
observation with all three images. Then step(action) applies one 10 Hz action of the public format and answers the
next observation, until the episode is over: after the takt's steps (keyed_env.MAX_STEPS), on a safety stop (a
ruined nut, an unstable simulation), when the episode's controller budget is used up (the graded budget: the
service's wait for each request plus the fixed charge per image asked for), after `silence_s` without a request, or
when the client closes it. close() answers how the episode ended and, if asked, the three cameras' images of the
final state. About 1 % of the step replies also carry `probe_obs`, the slightly perturbed copy of the observation
that the grader uses for its repeated-step queries, so that a client can run a controller the way the grader does.

The commissioning window: at most `episodes` episodes in the whole session (an episode counts from its reset; the
count survives a restart of the service, in state/ledger.json) and at most `open` at a time. The window is shown in
/run/keyedcell/commission_status.json (world-readable); the service's own log (state/commissiond.log, readable by
its account only) records the episodes' kits, times and end reasons.

Wire: JSON lines {"id": n, "method": ..., "params": {...}} -> {"id": n, "result": ...} or {"id": n, "error": ...,
"detail": ...} (~/commission/client.py in the agent's home speaks it).
"""

import fcntl
import json
import os
import random
import resource
import signal
import socket
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SOCKET = os.environ.get("KEYEDCELL_COMMISSION_SOCKET", "/run/keyedcell/commission.sock")
STATUS = os.environ.get("KEYEDCELL_COMMISSION_STATUS", "/run/keyedcell/commission_status.json")
STATE = os.environ.get("KEYEDCELL_COMMISSION_STATE", os.path.join(HERE, "state"))
with open(os.path.join(HERE, "kits.json")) as _f:
    KITS = json.load(_f)
PUBLIC = ("reset", "step", "close")
RESET_WAIT_S = 30.0              # a connection's first request
AFTER_DONE_MAX_S = 60.0          # after the episode is over: time left to close it (and fetch the final images)
AFTER_DONE_MAX_REQUESTS = 20
MAX_LINE = 65536                 # a request line (an action is 8 numbers)
SEND_TIMEOUT_S = 60.0
BUSY_GRACE_S = 2.0               # a new connection waits this long for a slot before it is answered `busy`


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
    """The commissioning window's episode count, kept under an exclusive lock (the episodes run in forked children)."""

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
        n = int(KITS["episodes"])
        return {"commissioning_episodes": n, "episodes_used": int(d["episodes_used"]),
                "episodes_left": max(0, n - int(d["episodes_used"])), "development_kits": len(KITS["kits"]),
                "open_at_a_time": int(KITS["open"])}

    def publish(self):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            _write_json(STATUS, self.status(), 0o644)

    def take(self, kit):
        """One episode of the window for `kit` -> (episode number, status) or (None, status) when it is used up."""
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            if int(d["episodes_used"]) >= int(KITS["episodes"]):
                return None, self.status(d)
            d["episodes_used"] = int(d["episodes_used"]) + 1
            d["episodes"].append({"episode": d["episodes_used"], "kit": int(kit), "started": time.time(),
                                  "pid": os.getpid()})
            _write_json(self.path, d, 0o600)
            st = self.status(d)
            _write_json(STATUS, st, 0o644)
            return d["episodes_used"], st

    def refund(self, episode):
        """Give back an episode whose cell could not be started (a fault of the service, not a trial)."""
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            if d["episodes"] and d["episodes"][-1].get("episode") == episode and int(d["episodes_used"]) == episode:
                d["episodes"].pop()
                d["episodes_used"] = episode - 1
            else:                                   # a later episode began meanwhile: keep the count, mark the entry
                for e in d["episodes"]:
                    if e.get("episode") == episode:
                        e.update({"ended": time.time(), "end": "cell fault at start", "steps": 0})
            _write_json(self.path, d, 0o600)
            _write_json(STATUS, self.status(d), 0o644)

    def close(self, episode, reason, steps):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read()
            for e in d["episodes"]:
                if e.get("episode") == episode:
                    e.update({"ended": time.time(), "end": reason, "steps": int(steps)})
            _write_json(self.path, d, 0o600)


class Conn(object):
    """One client connection: JSON lines with a bounded length and a deadline per request."""

    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def read(self, timeout):
        """The next request object; None on EOF; raises socket.timeout on silence, ValueError on a malformed line."""
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise socket.timeout()
            self.sock.settimeout(left)
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                return None
            self.buf += chunk
            if len(self.buf) > MAX_LINE and b"\n" not in self.buf:
                raise ValueError("line too long")
        line, self.buf = self.buf.split(b"\n", 1)
        if len(line) > MAX_LINE:
            raise ValueError("line too long")
        try:
            obj = json.loads(line.decode("utf-8"))
        except RecursionError:
            raise ValueError("nested too deeply")
        if not isinstance(obj, dict) or "method" not in obj:
            raise ValueError("one request object per line")
        return obj

    def send(self, obj, timeout=SEND_TIMEOUT_S):
        self.sock.settimeout(timeout)
        self.sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def _wire(E, obs):
    """An observation in its wire form, or None if it holds a non-finite number (only after an unstable step)."""
    wire = E.to_wire(obs)
    try:
        json.dumps(wire, allow_nan=False)
    except ValueError:
        return None
    return wire


def _image_charge(E, action, charges):
    """(charge in s, count) of the images an action asks for; (0, 0) for a malformed action (the step refuses it)."""
    try:
        _mv, _g, render = E.parse_action(action)
    except (E.ActionError, TypeError, ValueError, OverflowError):
        return 0.0, 0
    return sum(float(charges[c]) for c in render), len(render)


def _reset(params, ledger):
    """reset(kit) -> (reply, kit number, episode number, E, env, obs); env is None when no episode began."""
    kit = params.get("kit")
    n = len(KITS["kits"])
    if isinstance(kit, bool) or not isinstance(kit, int) or not 1 <= kit <= n:
        return {"error": "bad_params", "detail": "reset(kit): kit is an integer 1 .. %d (the development kits)" % n}, None, None, None, None, None
    episode, st = ledger.take(kit)
    if episode is None:
        return ({"error": "commissioning_window_used",
                 "detail": "all %d commissioning episodes are used" % st["commissioning_episodes"]}, None, None, None, None, None)
    try:
        import keyed_env as E                      # the cell: loaded in the episode's own process
        entry = KITS["kits"][kit - 1]
        env = E.NutEnv(entry["kit"]["nuts"], early_end=True)
        obs = env.reset(int(entry["layout_seed"]), entry["kit"]["order"])
    except Exception:  # noqa: BLE001 - the cell could not be started: the episode is not counted
        log("episode %d: kit %d could not be started\n%s" % (episode, kit, traceback.format_exc()))
        ledger.refund(episode)
        return {"error": "cell_fault", "detail": "the cell could not be started; no episode was used"}, None, None, None, None, None
    log("episode %d: kit %d" % (episode, kit))
    reply = {"ok": True, "kit": kit, "episode": episode, "episodes_left": st["episodes_left"],
             "max_steps": int(env.max_steps), "controller_budget_s": float(KITS["controller_budget_s"]),
             "obs": _wire(E, obs)}
    return reply, kit, episode, E, env, obs


def serve(sock, ledger):
    """One commissioning episode on one connection."""
    conn = Conn(sock)
    silence = float(KITS["silence_s"])
    budget = float(KITS["controller_budget_s"])
    charges = dict(KITS["image_charge_s"])
    probe_rate = float(KITS["probe_rate"])
    rng = random.Random(int.from_bytes(os.urandom(8), "little"))
    E = env = obs = episode = kit_no = None
    ended = None                 # how the episode ended, once it is over
    close_reason = None          # why the connection ends
    spent = 0.0                  # the episode's controller time: the waits for the client's requests + image charges
    images = 0
    last_sent = None
    done_since = None
    n_after = 0
    try:
        while True:
            if env is None:
                wait = RESET_WAIT_S
            elif ended is None:
                wait = min(silence, max(0.0, budget - spent) + 0.05)
            else:
                wait = max(0.0, AFTER_DONE_MAX_S - (time.monotonic() - done_since))
            try:
                req = conn.read(wait)
            except socket.timeout:
                if env is not None and ended is None:
                    spent += time.monotonic() - last_sent
                    ended = ("controller budget ({:.0f} s) used up at step {}".format(budget, env.t) if spent >= budget
                             else "silence ({:.0f} s without a request) at step {}".format(silence, env.t))
                close_reason = "timeout"
                break
            except (ValueError, UnicodeDecodeError) as e:
                close_reason = "protocol (%s)" % str(e)[:60]
                try:
                    conn.send({"id": None, "error": "bad_request", "detail": "one JSON request object per line, at most 64 KiB"},
                              timeout=5.0)
                except (OSError, socket.timeout):
                    pass
                break
            if req is None:
                close_reason = "eof"
                break
            rid, method, params = req.get("id"), str(req.get("method")), req.get("params")
            params = params if isinstance(params, dict) else {}
            if env is not None and ended is None:
                spent += time.monotonic() - last_sent
            last = False
            try:
                if method == "reset":
                    if env is not None:
                        res = {"error": "one_episode_per_connection",
                               "detail": "this connection's episode has begun: open a new one"}
                    else:
                        res, kit_no, episode, E, env, obs = _reset(params, ledger)
                elif method not in PUBLIC:
                    res = {"error": "forbidden", "detail": "the wire has reset, step and close"}
                elif env is None:
                    res = {"error": "no_episode",
                           "detail": "an episode starts with reset(kit), kit 1 .. %d" % len(KITS["kits"])}
                elif method == "close":
                    if ended is None:
                        ended = "closed at step {}".format(env.t)
                    res = {"ok": True, "ended": ended, "steps": int(env.t), "controller_s": round(spent, 1),
                           "controller_budget_s": budget, "images_requested": images}
                    if params.get("images") is True:
                        try:
                            res["final_images"] = {cam: E.encode_image(env.render(cam)) for cam in E.CAMERA_NAMES}
                        except Exception:  # noqa: BLE001 - e.g. after an unstable step
                            res["final_images"] = None
                    last = True
                elif ended is not None:
                    res = {"error": "episode_over", "detail": ended}
                else:                                           # step
                    action = params.get("action")
                    charge, n_img = _image_charge(E, action, charges)
                    if spent + charge > budget:
                        spent += charge
                        ended = "controller budget ({:.0f} s) used up at step {}".format(budget, env.t)
                        res = {"obs": None, "done": True, "ended": ended}
                    else:
                        try:
                            obs, done, info = env.step(action)
                        except E.ActionError as e:
                            res = {"error": "bad_action", "detail": str(e)[:120]}
                        else:
                            spent += charge
                            images += n_img
                            res = {"obs": _wire(E, obs), "done": bool(done)}
                            if done:
                                if env.fault is not None:
                                    ended = "safety stop at step {}: unstable simulation ({})".format(env.t, env.fault)
                                elif info["ruined"]:
                                    ended = "safety stop at step {}: nut {} ruined (pin forced through the groove wall)".format(
                                        env.t, ", ".join(env.work_order()[k]["colour"] for k in info["ruined"]))
                                else:
                                    ended = "all steps run"
                                res["ended"] = ended
                            elif res["obs"] is not None and rng.random() < probe_rate:
                                probe = env.perturbed_observation(obs, rng)
                                res["probe_obs"] = E.to_wire({k: v for k, v in probe.items() if k not in E.CAMERA_NAMES})
            except Exception:  # noqa: BLE001 - the cell failed: say so, never why (the log keeps the trace)
                log("episode %s: cell fault\n%s" % (episode, traceback.format_exc()))
                if ended is None and env is not None:
                    ended = "cell fault at step {}".format(getattr(env, "t", 0))
                res = {"error": "cell_fault", "detail": "the cell failed; the episode is over"}
                last = True
            if isinstance(res, dict) and "error" in res:
                conn.send(dict({"id": rid}, **res))
            else:
                conn.send({"id": rid, "result": res})
            last_sent = time.monotonic()
            if last:
                close_reason = close_reason or method
                break
            if ended is not None:
                if done_since is None:
                    done_since = last_sent
                n_after += 1
                if n_after > AFTER_DONE_MAX_REQUESTS:
                    close_reason = "after the end"
                    break
    except (OSError, socket.timeout) as e:
        close_reason = close_reason or "connection (%s)" % type(e).__name__
    finally:
        if env is not None:
            if ended is None:
                ended = "connection closed at step {}".format(env.t)
            if close_reason in ("timeout", "after the end"):
                try:                                # a limit ended the episode: say so before the connection closes
                    conn.send({"id": None, "event": "episode_over", "reason": ended}, timeout=5.0)
                except (OSError, socket.timeout):
                    pass
            log("episode %d: kit %d ended (%s; connection: %s), steps %d, controller %.1f s, images %d" % (
                episode, kit_no, ended, close_reason, env.t, spent, images))
            ledger.close(episode, ended, env.t)
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            sock.close()
        except OSError:
            pass


def main():
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
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
    log("listening on %s: %d development kits, %d episodes, %d at a time" % (
        SOCKET, len(KITS["kits"]), KITS["episodes"], KITS["open"]))
    children = set()
    busy = set()

    def reap():
        for pids in (children, busy):
            for pid in list(pids):
                try:
                    done, _st = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    done = pid
                if done:
                    pids.discard(pid)

    while True:
        conn, _ = srv.accept()
        reap()
        grace = time.monotonic() + BUSY_GRACE_S     # an episode that has just been closed is still leaving
        while len(children) >= int(KITS["open"]) and time.monotonic() < grace:
            time.sleep(0.05)
            reap()
        if len(children) >= int(KITS["open"]):
            # as many episodes open as the service takes: the connection's first request is answered `busy` (by a
            # child, so that the service never waits on a client), and the connection is closed; no episode is used
            pid = os.fork()
            if pid == 0:
                srv.close()
                try:
                    c = Conn(conn)
                    req = c.read(10.0)
                    c.send({"id": (req or {}).get("id"), "error": "busy",
                            "detail": "at most %d commissioning episodes at a time" % KITS["open"]}, timeout=5.0)
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
            except Exception as e:  # noqa: BLE001 - an episode's failure never takes the service down
                log("episode failed: %r" % e)
            finally:
                os._exit(0)
        children.add(pid)
        conn.close()


if __name__ == "__main__":
    main()
