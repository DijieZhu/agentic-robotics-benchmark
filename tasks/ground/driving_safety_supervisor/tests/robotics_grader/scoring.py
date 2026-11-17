"""Turning a raw metric into a reward.

A robotics metric is whatever the task measures — a reward sum, a success rate, an ATE in
metres. None of those are comparable across tasks, so the benchmark scores the *ratio* to
a reference implementation's real measured value, then saturates it against the band a
genuinely good run lands in:

    ratio  = clamp(measured / anchor, 0, cap)          (higher-is-better metrics)
    ratio  = clamp(anchor / measured, 0, cap)          (lower-is-better metrics)
    reward = clamp(ratio / full_at, 0, 1)              (the framework's primary)

`full_at` is the task's own declaration of "as good as the reference", sized by the
measured seed-to-seed spread of the metric (`tests/anchor.json`). It exists because of
the validate double pass: an untouched sandbox must score 0.0 and the oracle 1.0 on `reward`,
after the engine's half-up rounding to two decimals (validation fails otherwise), and an oracle
re-run on fresh hidden seeds does not reproduce its anchor to the digit —
with the template's measured 1.16x spread over 12 runs, an honest oracle lands anywhere in
ratio ~0.86–1.16. Saturating at `full_at` makes "inside the reference band" an exact 1.0
while everything below it keeps a continuous gradient, which is what the leaderboard needs.

Two rules that are not negotiable:

* **The anchor is the reference implementation's own measured value, never the paper's
  printed number.** A paper's figure was produced on other hardware, other seeds, and
  often a kinder evaluation; scoring against it makes the ratio a measure of that gap.
* **The anchor never appears on an agent-readable path.** It lives in the verify stage
  folder, which is absent from the sandbox while the agent works. The agent is told what
  to maximize, not what number to hit.

`cap` exists because a task where beating the reference by 10x is possible would otherwise
let one task dominate an aggregate. 1.5 is the domain default: real headroom, bounded. The
capped ratio itself — the domain's leaderboard score, where "matched the reference" and
"beat the reference" stay distinguishable — rides along in the verdict's `metrics`, which the engine
persists but never gates on.
"""

import json
import math
import os
import sys

from .stage import fail, stage_dir

__all__ = ["ratio", "ratio_rewards", "read_anchor", "stage_config", "zero_verdict"]

DEFAULT_CAP = 1.5


def stage_config(name="grader_config.json", default=None):
    """Grader knobs, read from the stage folder (`tests/grader_config.json`).

    Deliberately not in task.toml: its `[metadata]` has no runtime behavior, and episode
    counts, probe rates and deadlines are the grader's business anyway.

    NOT `config.json` — the engine writes its own verification config (Judge settings)
    to `tests/config.json` at verify time (`ALE_VERIFY_CONFIG_PATH`), silently replacing
    any file the task shipped under that name. `instruction.md`, `trajectory.json`,
    `llm-judge.jsonl` and `agent-judge.jsonl` are reserved the same way.
    """
    path = os.path.join(stage_dir(), name)
    try:
        with open(path) as handle:
            return json.load(handle)
    except (IOError, OSError, ValueError):
        return {} if default is None else default


def read_anchor(name="anchor.json"):
    """The reference value this task is scored against, with its metric name.

    Returns `(None, payload)` when the anchor is still null. That is the deliberate
    BOOTSTRAP path of task authoring: a task cannot be scored before anyone has measured
    its reference implementation, but the author needs that measurement, and it comes from
    a real run. So the grader runs the episodes, reports the measured value in the verdict,
    and scores 0 — which the domain's onboarding gate reports as `needs_anchor`, and that
    is exactly right. The task is not admissible yet; it is one recorded number away.

    The flow, start to finish: leave `value` null → run `ale validate` → read the metric
    off the verdict → write it into anchor.json → re-run → the oracle lands near ratio 1.0.

    A missing or unreadable file is still a task error: that is a broken task, not an
    unfinished one.
    """
    path = os.path.join(stage_dir(), name)
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (IOError, OSError, ValueError) as exc:
        fail("cannot read the anchor at {}: {}".format(path, exc))
    value = payload.get("value")
    return (None if value is None else float(value)), payload


def ratio(measured, anchor, cap=DEFAULT_CAP, higher_is_better=True):
    """The bounded ratio of a measured metric to its anchor.

    Requires a metric on a POSITIVE scale, where 0 is the worst possible outcome. That is
    not a formality: with a negative anchor the division inverts the ordering, so a policy
    that genuinely improves on the reference scores *lower* — silently, with no error, for
    the life of the task. See `_check_scale`.
    """
    measured = float(measured)
    anchor = float(anchor)
    _check_scale(measured, anchor)
    if higher_is_better:
        # inf would silently score the cap; on a higher-is-better metric it can only be
        # a grader defect, so it is a task error rather than a jackpot.
        if not math.isfinite(measured):
            fail("measured is {} — a higher-is-better metric must be finite".format(measured))
        raw = measured / anchor if anchor else 0.0
    else:
        # Lower-is-better inverts: a measured 0 would be a perfect score, which for error
        # metrics means "the rollout never happened" far more often than "flawless".
        # inf is the legitimate worst case ("no usable episode") and scores 0.
        raw = (anchor / measured) if measured > 0 else 0.0
    return max(0.0, min(raw, float(cap)))


def _check_scale(measured, anchor):
    """Refuse to score a metric whose sign breaks the ratio.

    A task error, not a zero: the task is mis-specified, and every score it ever produced
    would be ordered backwards. The fix is on the metric's side — a negative reward sum
    becomes a positive cost to minimize (`direction: lower`), or gets shifted onto a
    positive scale where 0 really is the worst case.
    """
    if anchor <= 0.0:
        fail(
            "anchor is {} — the ratio needs a metric on a positive scale where 0 is the "
            "worst outcome. A negative or zero anchor inverts the ordering, so a better "
            "policy would score lower. Recast the metric: report the cost and set "
            "`direction: lower`, or shift it so 0 is the floor.".format(anchor)
        )
    if measured < 0.0:
        fail(
            "measured {} is negative against a positive anchor {} — the metric is not on "
            "the scale it declared. Recast it as a cost with `direction: lower`, or shift "
            "it so 0 is the floor.".format(measured, anchor)
        )


def ratio_rewards(measured, anchor, cap=DEFAULT_CAP, higher_is_better=True,
                  metric_name="measured", locks_ok=True, extra=None, full_at=1.0):
    """Build the verdict envelope — `{"rewards": ..., "metrics": ...}` — from one
    measured value.

    `rewards` carries only the gated score: `reward = clamp(ratio / full_at, 0, 1)`.
    `reward`, rounded half-up to two decimals, must be 0.0 from an untouched sandbox and
    1.0 from the oracle (validation fails otherwise), so anything that is not that score —
    the raw metric, the capped ratio, the lock outcomes — goes in `metrics`, which
    verification.json records and nothing gates.

    `full_at` is where the reward saturates, in ratio units: the worst ratio a genuinely
    on-reference run can land on under fresh hidden seeds. Size it from a real measured
    spread (run the oracle several times), not from hope — too high and an honest oracle
    fails its own gate by luck; too low and a mediocre solution reads as perfect. It is
    part of `tests/anchor.json` because it is calibration data with the same provenance
    obligations as the anchor.

    `locks_ok=False` zeroes the score outright. A failed match lock does not mean "scored
    a little lower"; it means the number was not produced by solving the task, so there is
    nothing to scale.

    `anchor=None` is the bootstrap run (see `read_anchor`): the measured value is reported
    and the score is 0, so the run reads as "not admissible yet" rather than as a pass.
    """
    full_at = float(full_at)
    if not math.isfinite(full_at) or not 0.0 < full_at <= float(cap):
        fail("full_at is {} — it must lie in (0, cap]; it is the ratio at which the "
             "reward saturates to 1.0, sized from the oracle's measured seed-to-seed "
             "spread".format(full_at))
    if anchor is None:
        value = 0.0
        print("[grader] anchor is null — this run MEASURED {} = {}. Record it in "
              "tests/anchor.json and re-run; scoring 0 until then.".format(
                  metric_name, measured), file=sys.stderr, flush=True)
    else:
        value = ratio(measured, anchor, cap=cap, higher_is_better=higher_is_better)
    if not locks_ok:
        value = 0.0
    metrics = {
        "ratio": value,                 # the domain's leaderboard score, capped at `cap`
        "anchor_recorded": 0.0 if anchor is None else 1.0,
    }
    # The raw metric, for the record — unless it is non-finite (the lower-is-better
    # "no usable episode" sentinel), which the engine's record refuses and which would
    # read as a real measurement anyway. episodes_aborted tells that story.
    if math.isfinite(float(measured)):
        metrics[metric_name] = float(measured)
    if extra:
        metrics.update({k: float(v) for k, v in extra.items()})
    return {
        "rewards": {"reward": max(0.0, min(value / full_at, 1.0))},
        "metrics": metrics,
    }


def zero_verdict(reason=None, anchored=False, extra=None):
    """The honest zero: the verdict an absent or unusable submission earns.

    Under `ale validate`, an untouched sandbox must produce a real verdict of reward 0.0 —
    a crash would read as a broken task, not as an agent that did nothing. The reward
    keys match `ratio_rewards`' exactly, and the reason rides along as the verdict's
    `diagnostic`, which `write_verdict` records in `verification.json`.

    Deliberately reports NO raw metric: nothing was measured, and a fabricated 0.0 would
    read as a perfect run on any lower-is-better metric. `anchored` states the truth about
    the anchor, which exists independently of whether a submission does.
    """
    if reason:
        print("[grader] scoring 0: {}".format(reason), file=sys.stderr, flush=True)
    metrics = {"ratio": 0.0, "anchor_recorded": 1.0 if anchored else 0.0}
    if extra:
        metrics.update({k: float(v) for k, v in extra.items()})
    verdict = {"rewards": {"reward": 0.0}, "metrics": metrics}
    if reason:
        verdict["diagnostic"] = "scoring 0: {}".format(reason)
    return verdict
