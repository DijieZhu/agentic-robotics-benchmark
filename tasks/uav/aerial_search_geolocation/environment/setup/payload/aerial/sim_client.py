"""sim_client - drive the simulator bridge from the task's py3.12 side.

The simulator side (OpenCV, msgpack and the AirSim RPC) runs in its own Python 3.10 venv,
while the verify stage must run on the engine's 3.12 `python3`. So the simulator runs as a
child process, `sim_bridge.py` under `/opt/aerialsim/venv/bin/python`, spoken to over a
JSON-lines pipe. Nothing listens on a port for this pipe (the scene's own AirSim server
does, on loopback; see netguard).

Shared byte-identically by the grading stage and the agent payload. stdlib only, 3.8-compatible syntax.
"""

import json
import os
import queue
import subprocess
import threading
import time

BRIDGE_PYTHON = os.environ.get("AERIALSIM_PYTHON", "/opt/aerialsim/venv/bin/python")
HERE = os.path.dirname(os.path.abspath(__file__))


class SimError(Exception):
    """The simulator failed. A TASK error (or an infrastructure one), never the agent's."""


class SimClient(object):
    """One bridge process. Opening a scene is slow (Unreal start-up), so reuse a client
    across the episodes of one map; the bridge reuses the running scene by itself."""

    def __init__(self, backend="airsim", python=None, bridge=None, ready_timeout=900.0,
                 step_timeout=120.0, log_path=None, extra_args=None, run_as=None):
        """`run_as` names an account to run the bridge (and so Unreal) as: Unreal refuses
        to run as root, and the grading stage is root."""
        self.step_timeout = float(step_timeout)
        argv = [python or BRIDGE_PYTHON, "-u", bridge or os.path.join(HERE, "sim_bridge.py"),
                "--backend", backend] + list(extra_args or [])
        if run_as:
            argv = ["runuser", "-u", run_as, "--"] + argv
        self._log = open(log_path, "a") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._log, universal_newlines=True, bufsize=1,
                                     start_new_session=True)
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        hello = self._read(ready_timeout)
        if not hello.get("ready"):
            self.close()
            raise SimError("simulator bridge did not start: {!r}".format(hello)[:400])
        self.info = hello

    def _pump(self):
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _read(self, timeout):
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise SimError("simulator bridge timed out after {:.0f}s".format(timeout))
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                raise SimError("simulator bridge exited (code {})".format(self.proc.poll()))
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except ValueError:
                continue   # library noise on stdout; the bridge's replies are JSON objects

    def call(self, message, timeout=None):
        try:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
        except (IOError, OSError, ValueError) as exc:
            raise SimError("simulator bridge stdin closed: {}".format(exc))
        reply = self._read(timeout or self.step_timeout)
        if "error" in reply:
            raise SimError("simulator: {}".format(reply["error"])[:2000])
        return reply

    def reset(self, episode, budget, success_radius):
        # A reset may open a new Unreal scene: allow a long first start.
        return self.call({"cmd": "reset", "episode": episode, "budget": int(budget),
                          "success_radius": float(success_radius)}, timeout=900.0)

    def step(self, action):
        return self.call({"cmd": "step", "action": str(action)})

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                pass
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 9)
            except OSError:
                pass
        if self._log is not subprocess.DEVNULL:
            self._log.close()
            self._log = subprocess.DEVNULL

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False
