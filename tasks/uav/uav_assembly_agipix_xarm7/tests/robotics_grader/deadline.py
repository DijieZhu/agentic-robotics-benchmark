"""A verdict before the verifier's own time limit, never a run without one.

The engine stops a verify stage that outlives the task's `verifier.timeout_sec` and records no
reward: the run then reads as a broken task. Whatever an agent's submission can cause must be
a score instead. `VerifierDeadline` is the last line of that: the grader arms it at its start
with a time safely inside the engine's limit and a function that writes the verdict for "ran
out of time"; if the grader is still running then, the function runs and the process ends with
status 0. A grader that finishes disarms it.

It does not replace per-request deadlines or a budget on the solver's own time; it bounds
what those do not see (the grader's own simulation and rendering made slow by a loaded host).
"""

import os
import threading
import time

__all__ = ["VerifierDeadline"]


class VerifierDeadline(object):
    """`with VerifierDeadline(seconds, on_expire): grade()`.

    `on_expire(elapsed_seconds)` must write the verdict (reward and the cause in the metrics)
    and return; the process then exits 0 without running anything else (`os._exit`, so no
    grading thread can overwrite the verdict afterwards). `exit=False` is for tests.
    """

    def __init__(self, seconds, on_expire, exit=True):
        self.seconds = float(seconds)
        self.on_expire = on_expire
        self.exit = bool(exit)
        self.expired = False
        self._start = None
        self._timer = None
        self._lock = threading.Lock()

    def arm(self):
        self._start = time.monotonic()
        self._timer = threading.Timer(self.seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()
        return self

    def disarm(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _fire(self):
        with self._lock:
            if self._timer is None:      # disarmed while the timer thread was starting
                return
            self.expired = True
        try:
            self.on_expire(time.monotonic() - self._start)
        finally:
            if self.exit:
                os._exit(0)

    def __enter__(self):
        return self.arm()

    def __exit__(self, *_exc):
        self.disarm()
        return False
