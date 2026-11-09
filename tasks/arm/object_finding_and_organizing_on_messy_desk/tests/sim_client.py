"""sim_client — talk to the messy-desk simulator process from the task's py3.12 side.

The simulator (env.py in this folder) runs under the Isaac interpreter, /opt/isaac/bin/python3.
This module is stdlib-only and keeps 3.8-compatible syntax; the agent's lab_client.py is the
same start logic for the desk lab.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

ISAAC_PYTHON = os.environ.get("MESSY_DESK_PYTHON", "/opt/isaac/bin/python3")


class SimError(Exception):
    """The simulator process failed. A TASK error, never the agent's fault."""


class SimClient(object):
    """`ready_timeout` bounds one start attempt. Isaac's renderer occasionally deadlocks on the
    very first frame of a fresh process (the process sits idle forever before announcing
    readiness); such an attempt is killed and the simulator started again, `attempts` times.
    The first start in a fresh sandbox compiles the renderer's shaders: 547-621 s measured on an
    RTX 4090 (driver 580.178); a start with that cache in place takes 5-15 s. The default leaves
    room for the cold start on a slower GPU."""

    def __init__(self, env_script, python=None, ready_timeout=1500.0, step_timeout=120.0, log_path=None, attempts=3):
        self.python = python or ISAAC_PYTHON
        self.env_script = env_script
        self.step_timeout = float(step_timeout)
        self._log = open(log_path, "a") if log_path else subprocess.DEVNULL
        t0 = time.time()
        error = None
        for attempt in range(1, max(1, int(attempts)) + 1):
            self._launch()
            try:
                hello = self._read(ready_timeout)
                if hello.get("ready"):
                    self.startup_s, self.start_attempts = time.time() - t0, attempt
                    return
                error = SimError("simulator did not announce readiness: %r" % (hello,))
            except SimError as exc:
                error = exc
            self._kill()
            if self._log is not subprocess.DEVNULL:
                self._log.write("\n[sim_client] start attempt %d failed (%s); restarting\n" % (attempt, error))
                self._log.flush()
        raise SimError("simulator did not start in %d attempts: %s" % (attempt, error))

    def _launch(self):
        self.proc = subprocess.Popen([self.python, self.env_script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._log, universal_newlines=True, bufsize=1,
                                     start_new_session=True)
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self._lines), daemon=True).start()

    def _kill(self):
        try:
            os.killpg(self.proc.pid, 9)
        except OSError:
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _pump(proc, lines):
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _read(self, timeout):
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise SimError("simulator reply exceeded %.0fs" % timeout)
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise SimError("simulator reply exceeded %.0fs" % timeout)
            if line is None:
                raise SimError("simulator closed its stdout (exit code %s)" % self.proc.poll())
            line = line.strip()
            if not line.startswith("{"):
                continue                      # a stray log line, never the wire
            try:
                return json.loads(line)
            except ValueError:
                continue

    def _ask(self, msg, timeout=None):
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (IOError, OSError, ValueError):
            raise SimError("simulator stdin closed")
        reply = self._read(timeout or self.step_timeout)
        if not reply.get("ok"):
            raise SimError("simulator error on %s: %s" % (msg.get("cmd"), reply.get("error")))
        return reply

    def info(self):
        return self._ask({"cmd": "info"})["info"]

    def reset(self, seed, horizon=None):
        r = self._ask({"cmd": "reset", "seed": int(seed), "horizon": horizon}, timeout=max(self.step_timeout, 600.0))
        return r["obs"], r["state"]

    def step(self, action):
        r = self._ask({"cmd": "step", "action": [float(x) for x in action]})
        return r["obs"], r["state"]

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
            self.proc.stdin.flush()
        except (IOError, OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        if self._log is not subprocess.DEVNULL:
            try:
                self._log.close()
            except Exception:  # noqa: BLE001
                pass
