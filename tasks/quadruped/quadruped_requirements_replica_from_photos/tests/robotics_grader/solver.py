"""Driving an agent-built solver across the privilege boundary (the closed_loop pattern).

The grader owns the simulator and asks the agent's policy for one action at a time. Three
properties make that safe to score:

1. **The solver runs as the agent, not as root.** `tests/test.sh` is root — the framework
   runs it on the task's behalf — so spawning the solver directly would give the agent's
   code the grader's own privileges. `stage.deprivileged` drops to the agent account, and
   from there `/opt/ale` is untraversable and the grader's memory unreadable.
2. **Every request has a single deadline.** A solver that hangs must cost the episode,
   not the run. Reads happen on a queue fed by a reader thread, so a stall is a timeout
   rather than a wedged pipe — and the deadline is fixed once, when the request is written
   (the `READY` handshake likewise), so blank lines, a partial line or a trickle of output
   are all charged against the same `step_timeout` and cannot extend it. Missing the
   deadline aborts the episode exactly as a slow reply does.
3. **The wire carries no scoring input.** The solver returns actions. The metric is
   computed by the grader from its own simulator state, so nothing the solver prints can
   move its score.

The protocol is one JSON object per line, stdout only, `READY` first (within
`READY_TIMEOUT_FACTOR` step deadlines of launch):

    {"type": "reset"}                        -> {"ok": true}
    {"type": "act", "t": 0, "obs": <json>}   -> {"action": <json>}
    {"type": "close"}                        -> {"ok": true}, then exit

`t` is the step index inside the episode — an integer the GRADER owns, restarting at 0
after every `reset`. `obs` is the observation exactly as the task's environment produces
it (a flat list for most tasks; a task with structured observations documents the shape
in its instruction). The solver answers every request from the observation it carries.

**Nothing on the wire says whether a request is a probe.** The match lock (`ProbeLock`)
asks a probed step twice with the SAME `t` — once with the real observation, once with a
perturbed copy, in an order the solver cannot predict — and applies only the real reply
to the simulator. A repeated `t` is therefore an ordinary event the instruction tells the
agent about ("a step may be queried more than once; repeated queries do not advance the
episode; count steps by `t`, not by requests"), and nothing on the wire tells the solver
which of the two requests is the probe. The old design flagged probes with `"probe": true`,
which let a canned policy pass the lock by reacting only to flagged requests. The lock
catches a policy that ignores its input; it is not a proof of honesty, because a solver
that notices a repeated `t` can vary its answer on purpose. Isolation, not the lock, keeps
a score honest.

**An aborted episode does not take the rest of the run with it.** After a timeout, a
crash, a protocol error or a reply without an action, the wire may still hold a line meant
for an earlier request, so the next `reset` stops the solver (its whole process group) and
starts `run.sh` again, which must print `READY` again. At most `max_restarts` restarts
happen in a run (3 by default); after that, or after a restart that fails, every remaining
episode aborts at once instead of costing another deadline. The instruction must say so,
because a solver that keeps a PID file, a fixed port or a lock file has to survive being
started twice.
"""

import json
import os
import signal
import subprocess
import time

try:  # py3
    import queue
except ImportError:  # pragma: no cover — py2 never runs here
    import Queue as queue  # type: ignore[no-redef]
import threading

from .stage import agent_home, deprivileged

__all__ = ["READY_TIMEOUT_FACTOR", "ProbeLock", "Solver", "SolverAborted", "SolverMissing"]

#: A solver may print library noise before its handshake; tolerate a bounded banner.
MAX_PRE_READY_LINES = 200

#: Blank lines the kit skips while waiting for one request's reply, before it gives up on
#: that request. A well-behaved solver writes none; the cap bounds a solver that floods
#: blank lines faster than the deadline so one request cannot spin the grader.
MAX_BLANK_LINES = 1000

#: `READY` must arrive within this many step deadlines of launch (model loading is slower
#: than a step). With the template's 30 s step deadline that is 120 s; the instruction
#: quotes the product, so keep the two in sync.
READY_TIMEOUT_FACTOR = 4


class SolverAborted(Exception):
    """The solver crashed, stalled, or broke the protocol.

    This aborts the EPISODE, not the run: an aborted episode counts against the agent's
    score (it failed to produce a usable policy), which is different from a task error.
    """


class SolverMissing(Exception):
    """No submission where the task said one would be.

    An honest ZERO, not a task error: `ale validate`'s untouched pass runs the verify
    stage against a sandbox nobody touched, and demands a real all-zero verdict from it —
    a crash there reads as a broken task. The grader catches this and writes
    `zero_verdict(...)`; the checked path is in the message so an author who declared the
    wrong path can still see what was looked for.
    """


class Solver(object):
    """The agent's `run.sh`, running as the agent, spoken to over stdio.

    Launched as `bash run.sh` with the working directory set to the submission folder
    (`<agent home>/submission` unless `submission=` says otherwise), in its own process
    group: stopping the solver stops everything `run.sh` started, unless a process left the
    group on purpose (`setsid`).
    """

    def __init__(self, step_timeout=30.0, submission=None, entry="run.sh", max_restarts=3):
        self.submission = submission or os.path.join(agent_home(), "submission")
        if not os.path.isfile(os.path.join(self.submission, entry)):
            raise SolverMissing("no submission at {}/{}".format(self.submission, entry))
        self.step_timeout = float(step_timeout)
        self.entry = entry
        self.max_restarts = int(max_restarts)
        self.restarts = 0
        self.proc = None
        self._broken = False   # the wire may be out of step: restart before the next episode
        self._dead = False     # no more restarts: every later episode aborts at once
        self._closed = False
        self._start()

    # -- process ------------------------------------------------------------- #
    def _spawn(self):
        return deprivileged(
            ["bash", self.entry],
            cwd=self.submission,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            bufsize=1,
            # Its own process group, so stopping it also stops what `run.sh` started:
            # `runuser` does not forward a kill to its child.
            start_new_session=True,
        )

    def _start(self):
        self.proc = self._spawn()
        # One queue per process: a late line from a killed process lands in the old queue.
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self._lines), daemon=True).start()
        try:
            self._await_ready()
        except SolverAborted:
            self._stop(grace=0)        # never leave a solver that did not come up running
            raise

    def _restart(self):
        if self.restarts >= self.max_restarts:
            self._dead = True
            raise SolverAborted(
                "solver already restarted {} times (max_restarts); the remaining episodes "
                "abort".format(self.restarts))
        self.restarts += 1
        self._stop(grace=0)
        try:
            self._start()
        except (SolverAborted, OSError) as exc:
            self._dead = True
            raise SolverAborted("solver could not be restarted: {}".format(exc))
        self._broken = False

    def _stop(self, grace):
        """Stop the solver and its whole process group; idempotent."""
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except (IOError, OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=grace)
        except Exception:  # noqa: BLE001 — TimeoutExpired on py3.3+, anything else is moot
            pass
        # Always, not only on a timeout: after a crash `run.sh` is gone but a helper it
        # started (a model server, say) may still hold a port, GPU memory or a lock file.
        _kill_group(self.proc.pid)
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        _await_group_exit(self.proc.pid)

    # -- wire ---------------------------------------------------------------- #
    @staticmethod
    def _pump(proc, lines):
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)  # EOF

    def _readline(self, deadline, budget, what="step"):
        """One line, or an abort, against a single wall-clock `deadline`.

        `deadline` is a `time.monotonic()` instant fixed when the request was written (or
        the launch began), NOT reset per line: blank lines, a partial line or a trickle of
        output are read against the time that remains, so they cannot extend the budget.
        `budget` is the whole allowance, named only for the abort message.
        """
        remaining = deadline - time.monotonic()
        try:
            line = self._lines.get(timeout=max(0.0, remaining))
        except queue.Empty:
            raise SolverAborted(
                "solver exceeded its {:g}s {} deadline".format(budget, what))
        if line is None:
            raise SolverAborted("solver closed its stdout")
        return line

    def _await_ready(self):
        # One deadline for the whole handshake: a banner that trickles out slowly, or
        # floods blank lines, cannot push READY past the budget.
        budget = self.step_timeout * READY_TIMEOUT_FACTOR
        deadline = time.monotonic() + budget
        for _ in range(MAX_PRE_READY_LINES):
            line = self._readline(deadline, budget, what="READY")
            if line.strip() == "READY":
                return
        raise SolverAborted(
            "solver wrote {} lines before READY".format(MAX_PRE_READY_LINES))

    def request(self, message):
        if self._dead:
            raise SolverAborted("solver could not be restarted after an earlier abort")
        try:
            return self._exchange(message)
        except SolverAborted:
            self._broken = True
            raise

    def _exchange(self, message):
        try:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
        except (IOError, OSError, ValueError):
            raise SolverAborted("solver stdin closed")
        # One deadline per request, fixed the moment it is written. Blank lines are still
        # skipped, but every read — blank, slow or partial — is charged against this one
        # deadline, so a solver cannot keep the request alive by dribbling output; and the
        # blank-line count is capped so a flood faster than the deadline cannot spin.
        deadline = time.monotonic() + self.step_timeout
        for _ in range(MAX_BLANK_LINES):
            line = self._readline(deadline, self.step_timeout).strip()
            if not line:
                continue
            try:
                reply = json.loads(line)
            except (ValueError, RecursionError):  # RecursionError: absurdly nested JSON
                raise SolverAborted(
                    "solver wrote non-JSON on the wire: {!r}".format(line[:120])
                )
            if isinstance(reply, dict):
                return reply
            raise SolverAborted("solver reply is not an object")
        raise SolverAborted(
            "solver wrote {} blank lines to one request without a reply".format(
                MAX_BLANK_LINES))

    # -- protocol ------------------------------------------------------------ #
    def reset(self):
        """Start an episode. After an aborted one, this first restarts the solver."""
        if self._dead:
            raise SolverAborted("solver could not be restarted after an earlier abort")
        if self._broken:
            self._restart()
        self.request({"type": "reset"})

    def act(self, t, obs):
        """One action for step `t` of the current episode, from observation `obs`.

        The request is exactly `{"type": "act", "t": t, "obs": obs}` — no other key, ever.
        The same `t` may be asked more than once (see `ProbeLock.probe`); nothing on the
        wire says which of those requests the simulator will apply, and that is the point.
        """
        reply = self.request({"type": "act", "t": int(t), "obs": obs})
        if "action" not in reply:
            # Possibly a stray line answering the wrong request: resynchronise next episode.
            self._broken = True
            raise SolverAborted("solver error: {}".format(reply.get("error", "no action")))
        return reply["action"]

    def close(self):
        """Ask the solver to exit, then stop its process group. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"type": "close"}) + "\n")
                self.proc.stdin.flush()
            except (IOError, OSError, ValueError):
                pass
        self._stop(grace=10)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _kill_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (OSError, ProcessLookupError):  # already gone (ESRCH), or never ours
        pass


def _await_group_exit(pgid, timeout=5.0):
    """Wait (bounded) until no process is left in the group, so a restart never races a
    dying helper for its port or lock file."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except (OSError, ProcessLookupError):
            return
        time.sleep(0.05)


class ProbeLock(object):
    """The closed_loop match lock: does the action actually depend on the observation?

    A replay or a constant is the cheapest way to fake a policy, and both are invisible in
    the metric alone when a task's reward tolerates mediocrity. So the grader occasionally
    asks a step twice — once with the true observation, once with a perturbed copy — and
    compares. A policy that reads its input answers differently; a canned stream answers
    identically.

    The two requests carry the SAME step index `t` and nothing else that differs, and
    their order is drawn from the grader's hidden RNG, so on the wire a probe is
    indistinguishable from any other repeated query (`Solver.act`). The perturbed reply is
    never applied to the simulator, so a probe cannot change the episode it audits. Verdict
    is a fraction, not a single sample, because a genuine policy may legitimately saturate
    its action range on some states.
    """

    def __init__(self, rng, rate=0.1, pass_fraction=0.5, magnitude=0.4, dims=3,
                 tolerance=1e-4):
        self.rng = rng
        self.rate = float(rate)
        self.pass_fraction = float(pass_fraction)
        self.magnitude = float(magnitude)
        self.dims = int(dims)
        self.tolerance = float(tolerance)
        self.total = 0
        self.passed = 0

    def due(self):
        """Whether this step should carry a probe."""
        return self.rng.random() < self.rate

    def perturb(self, values):
        """Shift the leading components of a flat observation by a hidden amount."""
        shifted = list(values)
        for i in range(min(self.dims, len(shifted))):
            shifted[i] = shifted[i] + self.rng.uniform(-self.magnitude, self.magnitude)
        return shifted

    def probe(self, solver, t, obs, perturbed=None):
        """Query step `t` twice — real and perturbed, in a hidden order — and return the
        REAL action, the only one the caller may apply to the simulator.

        `perturbed` defaults to `self.perturb(obs)`; a task whose observation is not a
        flat list passes its own perturbed copy. Both requests are plain `Solver.act`
        calls with the same `t`, so the wire never says which one is the probe.
        """
        if perturbed is None:
            perturbed = self.perturb(obs)
        if self.rng.random() < 0.5:
            action_probe = solver.act(t, perturbed)
            action_real = solver.act(t, obs)
        else:
            action_real = solver.act(t, obs)
            action_probe = solver.act(t, perturbed)
        self.record(action_probe, action_real)
        return action_real

    def record(self, action_probe, action_real):
        """Score one probe pair. Returns True if this pair showed sensitivity.

        A reply that is not a nest of numbers is a protocol error: the episode aborts.
        """
        self.total += 1
        try:
            a = [float(x) for x in _flatten(action_probe)]
            b = [float(x) for x in _flatten(action_real)]
        except (TypeError, ValueError, OverflowError):
            raise SolverAborted("solver action is not numeric")
        moved = len(a) != len(b) or max(
            [abs(x - y) for x, y in zip(a, b)] or [0.0]
        ) > self.tolerance
        if moved:
            self.passed += 1
        return moved

    @property
    def ok(self):
        """True when enough probes showed the policy reacting to its input.

        No probes fired (a very short episode, or rate 0) is not a failure — absence of
        evidence is not evidence of cheating, and the metric still has to be earned.
        """
        return self.total == 0 or (float(self.passed) / self.total) >= self.pass_fraction

    def rewards(self):
        """Lock outcomes, for the record. Never scored, but always visible."""
        return {
            "probes_total": float(self.total),
            "probes_passed": float(self.passed),
            "match_lock_ok": 1.0 if self.ok else 0.0,
        }


def _flatten(value):
    """Leaves of a nested list, depth-first, without recursion: `json.loads` accepts deeper
    nesting than the interpreter's recursion limit on some versions."""
    out, stack = [], [value]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        elif hasattr(item, "ravel"):  # numpy array, without importing numpy
            out.extend(float(x) for x in item.ravel().tolist())
        else:
            out.append(item)
    return out
