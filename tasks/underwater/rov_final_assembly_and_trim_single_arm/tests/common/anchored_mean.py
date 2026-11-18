"""Live-anchored episode-mean criteria (CHAIN_PLAN.md D6 and section 4.0).

Every episode-mean criterion of a rollout grader is
    clamp(mean_agent / mean_oracle_on_the_same_seeds, 0, 1)      (full_at 1.0)
where the oracle is the scripted reference run INSIDE verify on the same drawn seeds
(deterministic CPU MuJoCo, single thread, fixed timestep). Precondition M13': oracle success
>= 0.90 on three fresh seed families at authoring time. The zero-oracle guard: when the
oracle scores 0 on the drawn seeds the anchor is meaningless; by default that is a TASK ERROR
(the precondition is violated, not the agent), never a free 1.0 for the agent.

stdlib only; `robotics_grader.scoring.ratio_rewards` stays the entry point for stored anchors.
"""

import math

__all__ = ["anchored_ratio", "anchored_episode_mean", "AnchoredCriterion", "ZeroOracleError"]


class ZeroOracleError(RuntimeError):
    """The live oracle scored zero on the drawn seeds: precondition M13' violated (task error)."""


def anchored_ratio(agent_value, oracle_value, full_at=1.0, zero_oracle="error", eps=1e-12):
    """clamp((agent / oracle) / full_at, 0, 1) with the zero-oracle guard.

    `zero_oracle`: "error" raises ZeroOracleError (default: a task error, not a score);
    "zero" scores 0; "raw" scores clamp(agent, 0, 1) (only for criteria already in [0, 1]).
    """
    a = float(agent_value)
    o = float(oracle_value)
    if not math.isfinite(a) or not math.isfinite(o):
        raise ValueError("anchored_ratio needs finite values, got agent={} oracle={}".format(a, o))
    if a < 0 or o < 0:
        raise ValueError("anchored_ratio needs metrics on a positive scale (0 = worst), got agent={} oracle={}".format(a, o))
    full_at = float(full_at)
    if not 0 < full_at <= 1.5:
        raise ValueError("full_at must lie in (0, 1.5], got {}".format(full_at))
    if o <= eps:
        if zero_oracle == "error":
            raise ZeroOracleError("live oracle scored {} on the drawn seeds: the anchor is undefined (M13' violated)".format(o))
        if zero_oracle == "zero":
            return 0.0
        if zero_oracle == "raw":
            return max(0.0, min(a, 1.0))
        raise ValueError("unknown zero_oracle policy {!r}".format(zero_oracle))
    return max(0.0, min((a / o) / full_at, 1.0))


def anchored_episode_mean(agent_values, oracle_values, full_at=1.0, zero_oracle="error"):
    """Ratio of the two episode means; the lists must be per-episode values on the SAME seeds."""
    agent_values = [float(v) for v in agent_values]
    oracle_values = [float(v) for v in oracle_values]
    if len(agent_values) != len(oracle_values):
        raise ValueError("agent and oracle must be scored on the same seeds: {} vs {} episodes".format(
            len(agent_values), len(oracle_values)))
    if not agent_values:
        raise ValueError("no episodes")
    a = sum(agent_values) / len(agent_values)
    o = sum(oracle_values) / len(oracle_values)
    return anchored_ratio(a, o, full_at=full_at, zero_oracle=zero_oracle)


class AnchoredCriterion(object):
    """One live-anchored criterion with the stats a verify stage should log next to it."""

    def __init__(self, name, agent_values, oracle_values, seeds=None, full_at=1.0, zero_oracle="error"):
        self.name = name
        self.agent = [float(v) for v in agent_values]
        self.oracle = [float(v) for v in oracle_values]
        self.seeds = list(seeds) if seeds is not None else None
        if self.seeds is not None and len(self.seeds) != len(self.agent):
            raise ValueError("seeds and episode values differ in length")
        self.full_at = full_at
        self.zero_oracle = zero_oracle
        self.score = anchored_episode_mean(self.agent, self.oracle, full_at, zero_oracle)

    @property
    def agent_mean(self):
        return sum(self.agent) / len(self.agent)

    @property
    def oracle_mean(self):
        return sum(self.oracle) / len(self.oracle)

    def stats(self):
        """Flat float stats (`<name>_agent_mean`, `<name>_oracle_mean`, `<name>_n_episodes`)."""
        return {"{}_agent_mean".format(self.name): self.agent_mean,
                "{}_oracle_mean".format(self.name): self.oracle_mean,
                "{}_n_episodes".format(self.name): float(len(self.agent))}

    def reason(self):
        return "{}: agent mean {:.4f} / oracle mean {:.4f} on the same {} seeds -> {:.4f}".format(
            self.name, self.agent_mean, self.oracle_mean, len(self.agent), self.score)
