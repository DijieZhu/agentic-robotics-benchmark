"""The kit keeps a solver from slowing the grader, counts probe pairs by one rule everywhere, and builds probe
copies the way real observations are built (`shared/robotics_grader/solver.py`, `probe.py`, `deadline.py`).

- A solver and what it starts run at the lowest nice value from their first instruction; with `max_processes` set,
  too many processes in the account end the episode as an abort, not as a hung grader.
- `replies_differ` is the lock's one rule; every task's practice grader that has a `differ(a, b)` counts with
  it (their copies are executed here against the kit's on a table of pairs).
- `stale_copy`: a probe copy whose poses are the kinematics of the joints one physics step earlier has the real
  observation's residual; the exact-kinematics copy has none and is told apart by a threshold.
- `VerifierDeadline` writes its verdict when the time is up and stays silent when disarmed.

The priority, process-count, `replies_differ`, `stale_copy` and deadline tests fail on the kit as it was (the
names do not exist there); the practice-grader test fails for every grader that counted a length mismatch as a
difference.
"""

from __future__ import annotations

import ast
import math
import os
import random
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "shared"))

from robotics_grader import deadline as deadline_mod, probe as probe_mod, solver as solver_mod  # noqa: E402

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc, nice values")

# Reports its own nice value and its child's; with "fork" in the observation it starts sleepers.
REPORTER = """
    import json, os, subprocess, sys
    print("READY", flush=True)
    kids = []
    for line in sys.stdin:
        req = json.loads(line.strip() or "{}")
        if req.get("type") == "act":
            if req["obs"] and req["obs"][0] > 0:
                for _ in range(int(req["obs"][0])):
                    kids.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                                 start_new_session=True))
            child = subprocess.run([sys.executable, "-c", "import os; print(os.getpriority(os.PRIO_PROCESS, 0))"],
                                   capture_output=True, text=True)
            print(json.dumps({"action": [float(os.getpriority(os.PRIO_PROCESS, 0)),
                                         float(child.stdout.strip() or -99)]}), flush=True)
        else:
            print(json.dumps({"ok": True}), flush=True)
        if req.get("type") == "close":
            break
    for k in kids:
        k.kill()
"""


class LocalSolver(solver_mod.Solver):
    """The kit's Solver over the real wire, started in-process (no account switch, no clearing)."""

    def __init__(self, script, step_timeout=10.0, **kwargs):
        super().__init__(step_timeout, submission=os.path.dirname(script), entry=os.path.basename(script),
                         clear_account=False, **kwargs)

    def _spawn(self):
        return subprocess.Popen([sys.executable, self.entry], cwd=self.submission, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                                bufsize=1, start_new_session=True)

    def _uid(self):          # the test's own account stands in for the agent's
        return os.getuid()


def script(tmp_path, body, name="solver.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return str(path)


# ---------------------------------------------------------------- priority and process count
def test_solver_and_its_children_run_at_the_lowest_nice_value(tmp_path):
    with LocalSolver(script(tmp_path, REPORTER)) as solver:
        solver.reset()
        own, child = solver.act(0, [0.0])
        assert own == solver_mod.LOWEST_NICE
        assert child == solver_mod.LOWEST_NICE
        assert solver.priority["nice"] == solver_mod.LOWEST_NICE


def test_the_grader_keeps_its_own_priority_after_a_spawn(tmp_path):
    """Every thread of the grader, not only the caller: the spawn lowers a thread of its own, which ends. (Root in
    a sandbox cannot raise a priority back, so nothing of the grader's may be lowered at all.)"""
    def nices():
        return {int(t): os.getpriority(os.PRIO_PROCESS, int(t)) for t in os.listdir("/proc/self/task")}

    before = nices()
    with LocalSolver(script(tmp_path, REPORTER)) as solver:
        solver.reset()
        during = nices()
        assert all(during[t] == n for t, n in before.items() if t in during)
        assert os.getpriority(os.PRIO_PROCESS, 0) == before[os.getpid()]
        assert max(during.values()) < solver_mod.LOWEST_NICE or before[os.getpid()] == solver_mod.LOWEST_NICE
    assert os.getpriority(os.PRIO_PROCESS, 0) == before[os.getpid()]


def test_the_idle_class_is_off_unless_asked_for(tmp_path):
    with LocalSolver(script(tmp_path, REPORTER)) as solver:
        assert os.sched_getscheduler(solver.proc.pid) == os.SCHED_OTHER and solver.priority["idle"] == 0

    class Idle(LocalSolver):
        idle_class = True

    with Idle(script(tmp_path, REPORTER)) as solver:
        assert os.sched_getscheduler(solver.proc.pid) == os.SCHED_IDLE and solver.priority["idle"] == 1
    assert os.sched_getscheduler(0) == os.SCHED_OTHER


def test_a_spawn_that_fails_raises_in_the_callers_thread(tmp_path):
    class Broken(LocalSolver):
        def _spawn(self):
            raise OSError("no such launcher")

    with pytest.raises(OSError, match="no such launcher"):
        Broken(script(tmp_path, REPORTER))


def test_priority_can_be_left_alone(tmp_path):
    class Plain(LocalSolver):
        lowest_priority = False

    before = os.getpriority(os.PRIO_PROCESS, 0)
    with Plain(script(tmp_path, REPORTER)) as solver:
        solver.reset()
        own, _child = solver.act(0, [0.0])
        assert own == before


def test_too_many_processes_end_the_episode_and_the_next_one_restarts(tmp_path):
    class Capped(LocalSolver):
        max_processes = 0

    with Capped(script(tmp_path, REPORTER)) as solver:
        solver.max_processes = solver_mod.agent_process_count(os.getuid()) + 8
        solver.reset()
        solver.act(0, [30.0])                     # 30 sleepers in their own sessions
        with pytest.raises(solver_mod.SolverAborted, match="too many processes"):
            for t in range(1, solver_mod.PROCESS_CHECK_EVERY + 2):
                solver.act(t, [0.0])
        solver.max_processes = solver_mod.agent_process_count(os.getuid()) + 400
        solver.reset()                            # restarted: the wire is in step again
        assert solver.restarts == 1
        assert solver.act(0, [0.0])[0] == solver_mod.LOWEST_NICE


def test_no_process_is_counted_unless_a_task_sets_the_cap(tmp_path):
    with LocalSolver(script(tmp_path, REPORTER)) as solver:
        assert solver.max_processes == 0
        solver.reset()
        solver.act(0, [30.0])
        for t in range(1, solver_mod.PROCESS_CHECK_EVERY + 2):
            solver.act(t, [0.0])


# ---------------------------------------------------------------- the lock's rule, in the kit and in practice
PAIRS = [
    ([0.0, 1.0], [0.0, 1.0], False),
    ([0.0, 1.0], [0.0, 1.0 + 5e-5], False),
    ([0.0, 1.0], [0.0, 1.0 + 2e-4], True),
    ([0.0, 1.0], [0.0, 1.0, 2.0], False),             # a different count of numbers shows nothing
    ([0.0] * 9, [0.0] * 8 + [0.5], True),
    ([0.0] * 9, [0.0] * 7 + [0.5], False),
    ([[0.0, 1.0], [2.0]], [[0.0, 1.0], [2.5]], True),
    ([], [], False),
]


@pytest.mark.parametrize("first,second,expected", PAIRS)
def test_replies_differ(first, second, expected):
    assert solver_mod.replies_differ(first, second) is expected


def test_the_lock_counts_with_replies_differ_and_a_canonical_form():
    lock = solver_mod.ProbeLock(random.Random(0))
    for first, second, expected in PAIRS:
        assert lock.record(first, second) is expected
    assert solver_mod.replies_differ([1.0, 9.0], [2.0], canonical=lambda a: a[:1]) is True
    with pytest.raises(solver_mod.SolverAborted):
        lock.record(["x"], [1.0])


def practice_graders():
    out = []
    for path in sorted((REPO_ROOT / "tasks").glob("*/*/environment/setup/payload/practice_grader/grade.py")):
        text = path.read_text()
        if "\ndef differ(" in text:
            out.append(path)
    return out


#: Practice graders whose pair count is being changed in a pull request of their own; each must leave this set
#: when that pull request merges. Empty now.
PENDING: set = set()


@pytest.mark.parametrize("path", practice_graders(), ids=lambda p: p.parents[4].name)
def test_practice_grader_counts_pairs_as_the_lock_does(path):
    task = path.parents[4].name
    tree = ast.parse(path.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("differ", "_flat", "flat", "_flatten")]
    names = {f.name for f in fns}
    if task in PENDING:
        pytest.skip("changed in the task's own pull request")
    assert "differ" in names, "the practice grader has no differ(a, b)"
    scope: dict = {}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102 — repository code
    for first, second, expected in PAIRS:
        if any(isinstance(v, list) for v in first):      # these graders take the task's flat action only
            continue
        assert bool(scope["differ"](first, second)) is expected, (task, first, second)


# ---------------------------------------------------------------- probe copies
def planar(q, lengths=(0.4, 0.3, 0.2)):
    x = y = a = 0.0
    for qi, li in zip(q, lengths):
        a += qi
        x += li * math.cos(a)
        y += li * math.sin(a)
    return (x, y, a)


def residual(pose, q):
    fx, fy, fa = planar(q)
    return math.hypot(pose[0] - fx, pose[1] - fy)


def trajectory(steps, speed, rng):
    q = [0.3, -0.5, 0.8]
    rows = []
    for _ in range(steps):
        stale = list(q)
        q = [qi + speed * (0.5 + rng.random()) for qi in q]
        rows.append((stale, list(q)))
    return rows


@pytest.mark.parametrize("speed", [0.0, 2e-5, 2e-3])
def test_stale_copy_has_the_real_observations_residual(speed):
    rng = random.Random(7)
    sep_exact = sep_stale = 0
    for stale, observed in trajectory(400, speed, rng):
        real = residual(planar(stale), observed)
        copy_q = [qi + rng.uniform(-0.001, 0.001) for qi in observed]
        exact = residual(planar(copy_q), copy_q)
        built = residual(probe_mod.stale_copy(planar, stale, observed, copy_q), copy_q)
        assert exact == 0.0
        # choosing the candidate with the larger residual as "real", as a holder of the model would
        sep_exact += int(real > exact)
        sep_stale += int(real > built)
        assert abs(built - real) <= 2e-3 * real + 1e-15
    if speed > 0:
        assert sep_exact == 400                  # the old copy: told apart every time
        assert 120 < sep_stale < 280             # the new one: a coin toss
    else:
        assert sep_exact == 0 and sep_stale == 0   # at rest both residuals are zero


def test_stale_copy_checks_its_lengths():
    with pytest.raises(ValueError):
        probe_mod.stale_copy(planar, [0.0], [0.0, 0.0], [0.0, 0.0])


# ---------------------------------------------------------------- the verifier's last deadline
def test_deadline_writes_its_verdict_once_the_time_is_up(tmp_path):
    out = tmp_path / "verdict"
    guard = deadline_mod.VerifierDeadline(0.2, lambda elapsed: out.write_text("out of time %.1f" % elapsed), exit=False)
    with guard:
        time.sleep(0.6)
    assert guard.expired and out.read_text().startswith("out of time")


def test_deadline_is_silent_when_the_grader_finishes(tmp_path):
    out = tmp_path / "verdict"
    guard = deadline_mod.VerifierDeadline(0.3, lambda elapsed: out.write_text("x"), exit=False)
    with guard:
        pass
    time.sleep(0.5)
    assert not guard.expired and not out.exists()


def test_deadline_ends_the_process_with_status_zero(tmp_path):
    code = textwrap.dedent("""
        import sys, time
        sys.path.insert(0, %r)
        from robotics_grader.deadline import VerifierDeadline
        with VerifierDeadline(0.2, lambda s: open(%r, "w").write("zero")):
            time.sleep(30)
        open(%r, "w").write("overwritten")
    """) % (str(REPO_ROOT / "shared"), str(tmp_path / "v"), str(tmp_path / "v"))
    t0 = time.monotonic()
    done = subprocess.run([sys.executable, "-c", code], timeout=20)
    assert done.returncode == 0 and time.monotonic() - t0 < 10
    assert (tmp_path / "v").read_text() == "zero"
