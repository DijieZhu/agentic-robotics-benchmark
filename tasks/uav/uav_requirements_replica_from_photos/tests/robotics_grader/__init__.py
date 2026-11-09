"""robotics_grader — the robotics domain's shared verify-stage machinery.

A verify script's job is to compute one honest number under the task's own physics. The
parts that are the *same* for every robotics task — talking to an agent-built solver
across a privilege boundary, sampling per-run seeds that never touch disk, proving the
solver actually reacts to its input, and turning a raw metric into a bounded reward —
live here so a task author writes only the physics.

Three authoring patterns, one verify stage:

* **open_loop** — the task hands out problems and scores returned answers under its own
  dynamics. Use `seed_rng`, `require`, and `ratio_rewards`; no solver process at all if
  the agent's `run.sh` already produced its answers during the agent phase.
* **closed_loop** — the grader owns the simulator and drives the agent's policy step by
  step. Use `Solver` (spawns at agent UID, JSON-lines wire keyed by the grader-owned step
  index `t`, per-step deadline) plus `ProbeLock`, whose probes are indistinguishable on
  the wire: a probed step is simply asked twice with the same `t`, in a hidden order.
* **artifact_rollout** — the agent leaves a frozen artifact and the grader rolls it out.
  Use `deprivileged` to deserialize it: `torch.load` as root is grader compromise, and
  the structural absence of a wire is what makes this pattern lock-free.

Everything here is stdlib-only, and deliberately stays syntax-compatible with old
interpreters even though the engine now demands more: at verify time ALE stages its own
`ale_verify` package into the sandbox's `python3` and refuses anything below 3.12, so the
interpreter that imports this kit is always 3.12+. The 3.8-compatible syntax is kept
anyway — it costs nothing, and agent-side copies of the wire (a practice grader, a solver
template) may still run on whatever interpreter a legacy stack carries.
"""

from .scoring import ratio_rewards, read_anchor, stage_config, zero_verdict
from .seeds import seed_rng
from .solver import READY_TIMEOUT_FACTOR, ProbeLock, Solver, SolverAborted, SolverMissing
from .stage import agent_user, deprivileged, fail, require, stage_dir, verdict_path, write_verdict

__all__ = [
    "READY_TIMEOUT_FACTOR",
    "ProbeLock",
    "Solver",
    "SolverAborted",
    "SolverMissing",
    "agent_user",
    "deprivileged",
    "fail",
    "ratio_rewards",
    "read_anchor",
    "require",
    "seed_rng",
    "stage_config",
    "stage_dir",
    "verdict_path",
    "write_verdict",
    "zero_verdict",
]
