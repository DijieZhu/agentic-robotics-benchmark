#!/usr/bin/env python3
"""Practice grader: drives your controller (run.sh) over the grading wire on the practice failure states and prints the
raw outcome per state and per job. It uses the grader's own wire rules (`wire_rules.py`, the same file the grader
imports): deadlines, READY, the controller time cap, the action check, the 64 KiB line cap, the bounded reply queue,
discarding lines written without a request, restarts after an aborted episode (at most 3), the image cap, the success
rule and the step budget. Like the grader it pauses the controller (SIGSTOP) whenever no request is outstanding; here
that pauses run.sh's process group (the grader pauses every process of your account and also caps their number), and
the pause is not checked: the grader confirms that every thread of every process of your account has stopped and
ends the episode when one has not within 5 s.
It starts run.sh with the grader's environment: PATH, HOME and LANG only, nothing inherited from your shell. Each
request's 10 s deadline and its controller time start before the request is written, as in grading.
The kitchens and states are practice ones (RoboCasa pretrain split). It prints the raw outcome only: per state whether
it was recovered, per job the recovered fraction and their mean over the jobs on these practice states. It never prints
the grader's ratio or reward.

  /opt/venv/bin/python3 practice_grader/grade.py [--submission ~/submission] [--states p00,p03] [--max-steps N]
"""
import argparse
import fcntl
import json
import os
import queue
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# The controller's whole environment, as the grader's launcher gives it (nothing else is inherited from your shell).
CONTROLLER_ENV = {"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/home/user", "LANG": "C.UTF-8"}
HOME = os.path.dirname(HERE)
STATES = os.path.join(HOME, "practice_states")
sys.path.insert(0, HOME)
import wire_rules as W  # noqa: E402


class Aborted(Exception):
    pass


class Controller:
    """run.sh over JSON lines on stdin/stdout with the grader's rules; restarts after an aborted episode."""

    def __init__(self, folder):
        self.folder = folder
        self.restarts = 0
        self.broken = False
        self.dead = False
        self.last_reply_s = 0.0
        self._start()

    def _start(self):
        self.proc = subprocess.Popen(["bash", "run.sh"], cwd=self.folder, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None, universal_newlines=True, bufsize=1, start_new_session=True,
                                     env=dict(CONTROLLER_ENV))
        self.lines = queue.Queue(maxsize=W.MAX_QUEUED_LINES)
        threading.Thread(target=self._pump, args=(self.proc, self.lines), daemon=True).start()
        deadline = time.monotonic() + W.READY_TIMEOUT_S
        for _ in range(200):
            if self._get(deadline).strip() == "READY":
                self._signal(signal.SIGSTOP)
                return
        raise Aborted("200 lines before READY")

    def _signal(self, sig):
        try:
            os.killpg(self.proc.pid, sig)
        except OSError:
            pass

    @staticmethod
    def _pump(proc, lines):
        while True:
            try:
                line = proc.stdout.readline(W.MAX_LINE + 2)
            except (ValueError, OSError, UnicodeDecodeError):
                break
            if not line or W.line_too_long(line):
                break
            lines.put(line)
        lines.put(None)

    def _get(self, deadline):
        left = deadline - time.monotonic()
        try:
            line = self.lines.get(timeout=max(0.0, left))
        except queue.Empty:
            raise Aborted("deadline missed")
        if line is None:
            raise Aborted("controller closed stdout (or wrote a line over 64 KiB)")
        return line

    def _discard_stale(self):
        for _ in range(2000):
            drained = False
            while True:
                try:
                    line = self.lines.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    raise Aborted("controller closed stdout")
                drained = True
            try:
                pending = struct.unpack("i", fcntl.ioctl(self.proc.stdout.fileno(), termios.FIONREAD, b"\0\0\0\0"))[0]
            except (OSError, ValueError):
                pending = 0
            if not drained and not pending:
                return
            time.sleep(0.005)
        raise Aborted("controller keeps writing without a request")

    def _stop(self):
        self._signal(signal.SIGCONT)
        try:
            self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        self._signal(signal.SIGKILL)
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass

    def request(self, msg):
        if self.dead:
            raise Aborted("no restarts left")
        try:
            return self._exchange(msg)
        except Aborted:
            self.broken = True
            raise

    def _write_within(self, data, deadline):
        """Write the request against the reply's deadline, as the grader does: non-blocking, a chunk at a time as the
        pipe drains; a request not read by the deadline ends the episode like a late reply."""
        try:
            fd = self.proc.stdin.fileno()
        except (OSError, ValueError):
            raise Aborted("controller stdin closed")
        view = memoryview(data.encode("utf-8"))
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while view:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise Aborted("deadline missed (the request was not read in time)")
                if not select.select([], [fd], [], left)[1]:
                    continue
                try:
                    view = view[os.write(fd, view):]
                except BlockingIOError:
                    continue
                except OSError:
                    raise Aborted("controller stdin closed")
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    def _exchange(self, msg):
        self._discard_stale()
        data = json.dumps(msg) + "\n"
        t0 = time.monotonic()            # the clock and the deadline start before the write, as in grading
        deadline = t0 + W.STEP_TIMEOUT_S
        self._signal(signal.SIGCONT)
        try:
            self._write_within(data, deadline)
            for _ in range(1000):
                line = self._get(deadline).strip()
                if not line:
                    continue
                try:
                    reply = json.loads(line)
                except (ValueError, RecursionError):
                    raise Aborted("non-JSON on the wire: {!r}".format(line[:120]))
                if not isinstance(reply, dict):
                    raise Aborted("reply is not an object")
                return reply
            raise Aborted("too many blank lines")
        finally:
            self.last_reply_s = time.monotonic() - t0
            self._signal(signal.SIGSTOP)

    def reset(self):
        if self.dead:
            raise Aborted("no restarts left")
        if self.broken:
            if self.restarts >= W.MAX_RESTARTS:
                self.dead = True
                self._stop()             # as the grader: no controller is left behind once the restarts are used up
                raise Aborted("already restarted {} times".format(self.restarts))
            self.restarts += 1
            self._stop()
            try:
                self._start()
            except Aborted as exc:
                self.dead = True
                raise Aborted("could not restart: {}".format(exc))
            self.broken = False
        self.request({"type": "reset"})

    def act(self, t, obs):
        reply = self.request({"type": "act", "t": t, "obs": obs})
        if "action" not in reply:
            self.broken = True
            raise Aborted("reply without 'action'")
        return reply["action"]

    def close(self):
        """As the grader: the close message is written under a 1 s deadline, never blocking, and not at all once an
        exchange has failed (a request that timed out in the write has left the pipe full)."""
        self._signal(signal.SIGCONT)
        if self.proc.poll() is None and not self.broken and not self.dead:
            try:
                self._write_within(json.dumps({"type": "close"}) + "\n", time.monotonic() + 1.0)
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        self._stop()


def run_state(ctrl, st, max_steps):
    budget = min(max_steps or st["step_budget"], st["step_budget"])
    spec = {"id": st["id"], "task": st["task"], "seed": st["seed"], "split": st["split"],
            "prefix_file": os.path.join(STATES, st["prefix"]), "qpos_T_file": os.path.join(STATES, st["qpos"]),
            "cards_file": os.path.join(STATES, st["card"]), "roles": st.get("roles"), "step_budget": budget,
            "max_frames": W.MAX_FRAMES, "privileged": False}
    sim = subprocess.Popen(["/opt/venv/bin/python3", "-s", "-B", os.path.join(HOME, "sim_worker.py")], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1)

    def tell(msg):
        sim.stdin.write(json.dumps(msg) + "\n")
        sim.stdin.flush()
        line = sim.stdout.readline()
        if not line:
            raise SystemExit("the simulator exited")
        return json.loads(line)

    ctime, t = 0.0, 0
    try:
        ready = tell(spec)
        if not ready.get("ready"):
            raise SystemExit("cannot restore {}: {}".format(st["id"], ready.get("error")))
        prefix_err = ready.get("prefix_max_err")
        reply = tell({"first": True})
        run = 0
        try:
            ctrl.reset()
            while t < budget:
                action = ctrl.act(t, reply["obs"])
                ctime += ctrl.last_reply_s
                if ctime > W.CONTROLLER_CAP_S:
                    raise Aborted("controller time cap exceeded")
                try:
                    controls, cams = W.check_action(action)
                except W.Malformed as exc:
                    raise Aborted(str(exc))
                except Exception as exc:  # noqa: BLE001
                    raise Aborted("unusable reply ({})".format(type(exc).__name__))
                reply = tell({"action": controls, "cams": cams})
                t += 1
                run = run + 1 if reply["success"] else 0
                if run >= W.SUCCESS_RUN:
                    return True, "steps={} controller_s={:.1f} prefix_err={}".format(t, ctime, prefix_err)
            return False, "steps={} controller_s={:.1f} prefix_err={}".format(t, ctime, prefix_err)
        except Aborted as exc:
            ctrl.broken = True
            return False, "aborted: {} (t={}, controller_s={:.1f})".format(exc, t, ctime)
    finally:
        try:
            sim.stdin.write(json.dumps({"close": True}) + "\n")
            sim.stdin.close()
            sim.wait(timeout=30)
        except Exception:  # noqa: BLE001
            sim.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default=os.path.join(HOME, "submission"))
    ap.add_argument("--states", default="")
    ap.add_argument("--max-steps", type=int, default=0, help="shorter episodes for quick tests")
    args = ap.parse_args()
    index = json.load(open(os.path.join(STATES, "index.json")))
    wanted = set(args.states.split(",")) if args.states else None
    try:
        ctrl = Controller(args.submission)
    except Aborted as exc:
        print("controller did not start: {}".format(exc))
        return
    results = []
    try:
        for st in index["states"]:
            if wanted and st["id"] not in wanted:
                continue
            ok, note = run_state(ctrl, st, args.max_steps)
            results.append((st["task"], ok))
            print("{} {}: recovered={} {}".format(st["id"], st["task"], ok, note), flush=True)
    finally:
        ctrl.close()
    mean, per_job = W.job_balanced(results)
    for job, frac in sorted(per_job.items()):
        print("  {}: {:.3f}".format(job, frac))
    print("recovered {} of {} practice states; mean over jobs of the recovered fraction {:.3f}".format(
        sum(ok for _, ok in results), len(results), mean))


if __name__ == "__main__":
    main()
