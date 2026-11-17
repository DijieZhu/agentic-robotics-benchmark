"""sim_client - drive the CARLA simulator bridge from the task's py3.12 side.

The CARLA 0.9.15 Python client exists only for Python <= 3.10, while the grading stage must
run on the engine's 3.12 `python3`. So the simulator side runs as a child process,
`drive_bridge.py` under `/opt/carla/venv/bin/python`, spoken to over a JSON-lines pipe.
Nothing listens on a port for this pipe (CARLA itself does, on loopback; see netguard).

Shared byte-identically by the grading stage and the agent payload. stdlib only, 3.8-compatible syntax.
"""

import json
import os
import queue
import subprocess
import threading
import time

BRIDGE_PYTHON = os.environ.get("DRIVE_SIM_PYTHON", "/opt/carla/venv/bin/python")
HERE = os.path.dirname(os.path.abspath(__file__))


class SimError(Exception):
    """The simulator failed. A TASK error (or an infrastructure one), never the agent's.
    `restartable`: the bridge (or its CARLA server) is gone, so a new bridge may try again."""

    def __init__(self, message, restartable=False):
        Exception.__init__(self, message)
        self.restartable = restartable


class SimClient(object):
    """One bridge process. Starting CARLA and loading a town is slow, so reuse a client
    across episodes; the bridge keeps the server running and reloads towns as needed."""

    def __init__(self, backend="carla", python=None, bridge=None, ready_timeout=900.0,
                 step_timeout=120.0, log_path=None, extra_args=None, run_as=None):
        """`run_as` names an account to run the bridge (and so Unreal) as: Unreal refuses
        to run as root, and the grading stage is root."""
        self.step_timeout = float(step_timeout)
        argv = [python or BRIDGE_PYTHON, "-u", bridge or os.path.join(HERE, "drive_bridge.py"),
                "--backend", backend] + list(extra_args or [])
        if run_as:
            argv = ["runuser", "-u", run_as, "--"] + argv
        self._log = open(log_path, "a") if log_path else subprocess.DEVNULL
        self._argv, self._ready_timeout = argv, float(ready_timeout)
        self.restarts = 0
        self._start()

    def _start(self):
        # a fixed string hash, so nothing in the bridge can depend on the process's hash seed
        env = dict(os.environ, PYTHONHASHSEED="0")
        self.proc = subprocess.Popen(self._argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._log, universal_newlines=True, bufsize=1,
                                     start_new_session=True, env=env)
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self._lines), daemon=True).start()
        hello = self._read(self._ready_timeout)
        if not hello.get("ready"):
            self.close()
            raise SimError("simulator bridge did not start: {!r}".format(hello)[:400])
        self.info = hello

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
                raise SimError("simulator bridge timed out after {:.0f}s".format(timeout))
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                raise SimError("simulator bridge exited (code {})".format(self.proc.poll()),
                               restartable=True)
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
            raise SimError("simulator bridge stdin closed: {}".format(exc), restartable=True)
        reply = self._read(timeout or self.step_timeout)
        if "error" in reply:
            raise SimError("simulator: {}".format(reply["error"])[:2000],
                           restartable=bool(reply.get("server_exited")))
        return reply

    def reset(self, spec):
        """Build the scenario `spec` (town, kind, seed); may load a town: a long timeout.

        CARLA 0.9.15 can crash while it builds an episode after many episodes in one server.
        When the bridge or its server is gone, a new bridge (and server) builds the same
        episode again from its spec, once; nothing of the episode has run yet."""
        message = {"cmd": "reset", "spec": spec}
        try:
            return self.call(message, timeout=900.0)
        except SimError as exc:
            if not exc.restartable:
                raise
        self._replace()
        return self.call(message, timeout=900.0)

    def _replace(self):
        """A new bridge in place of one that has gone (or is going)."""
        old = self.proc
        try:
            old.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(old.pid, 9)
            except OSError:
                pass
        self.restarts += 1
        self._start()

    def step(self, mode, control):
        """One control period: mode 0 applies the nominal planner's control, mode 1 the
        given [throttle, steer, brake]."""
        return self.call({"cmd": "step", "mode": int(mode),
                          "control": [float(v) for v in control]})

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
