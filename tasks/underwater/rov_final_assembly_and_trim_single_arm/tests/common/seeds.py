"""Per-run hidden seeds with a logged entropy and an optional pinned stream (CHAIN_PLAN.md D6).

The rule of shared/robotics_grader/seeds.py stands: the seeds an episode is scored under are
sampled fresh, inside the grader process, after the agent is gone, and never reach a path the
agent's UID can stat. This module adds what the chain needs on top of it:

* the 64-bit OS entropy that seeds the stream is exposed as two 32-bit halves so the verify
  stage can log them as stats (`hidden_seed_hi`, `hidden_seed_lo`): the whole drawn stream is
  reproducible from the two numbers (splitmix64 is deterministic), which is what reverification
  and difficulty rounds need, without a seed file inside the sandbox;
* `pinned`: an optional `tests/hidden/seeds.json` (root-only stage folder) either pins the
  entropy (`{"entropy_hi", "entropy_lo"}`) or an explicit seed list (`{"seeds": [...]}`), for
  reverification and for difficulty rounds only: a published task ships without it;
* `derive(label)`: independent sub-streams per criterion (assembly episodes, kitchen episodes,
  FK configurations ...) from the same logged entropy, so the live anchor (the oracle run on the
  same drawn seeds, section 4.0) and the agent's rollout share exactly the same seed list.

stdlib only.
"""

import hashlib
import json
import os
import struct

__all__ = ["seed_rng", "SeedStream", "load_pinned", "write_pinned"]

_MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15


class SeedStream(object):
    """Deterministic-per-instance, unpredictable-across-runs integer source (splitmix64)."""

    def __init__(self, entropy=None, pinned_seeds=None, label="root"):
        if pinned_seeds is not None:
            self._pinned = [int(s) for s in pinned_seeds]
            self._pinned_pos = 0
            raw = int(hashlib.sha256(json.dumps(self._pinned).encode()).hexdigest()[:16], 16)
        else:
            self._pinned = None
            raw = entropy if entropy is not None else struct.unpack("<Q", os.urandom(8))[0]
        self.entropy = int(raw) & _MASK or _GOLDEN
        self._state = self.entropy
        self.label = label
        self.drawn = 0

    # -- stats ---------------------------------------------------------------------------
    @property
    def hi(self):
        return (self.entropy >> 32) & 0xFFFFFFFF

    @property
    def lo(self):
        return self.entropy & 0xFFFFFFFF

    def stats(self, prefix="hidden_seed"):
        """`{hidden_seed_hi, hidden_seed_lo, seeds_pinned}` as floats for `v.stat(...)`."""
        return {"{}_hi".format(prefix): float(self.hi), "{}_lo".format(prefix): float(self.lo),
                "seeds_pinned": 1.0 if self._pinned is not None else 0.0}

    # -- drawing -------------------------------------------------------------------------
    def _next(self):
        self._state = (self._state + _GOLDEN) & _MASK
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def seed(self, bits=31):
        """One seed, sized for the consumer (gymnasium wants < 2**31)."""
        self.drawn += 1
        if self._pinned is not None:
            if self._pinned_pos >= len(self._pinned):
                raise RuntimeError("pinned seed list exhausted after {} seeds (stream '{}'); a pinned run must pin "
                                   "at least as many seeds as the grader draws".format(len(self._pinned), self.label))
            v = self._pinned[self._pinned_pos]
            self._pinned_pos += 1
            return v & ((1 << bits) - 1)
        return self._next() & ((1 << bits) - 1)

    def seeds(self, count, bits=31):
        return [self.seed(bits) for _ in range(count)]

    def random(self):
        """A float in [0, 1): for probe scheduling, not for physics."""
        return (self._next() >> 11) * (1.0 / (1 << 53))

    def uniform(self, low, high):
        return low + (high - low) * self.random()

    def choice(self, items):
        return items[self._next() % len(items)]

    def derive(self, label):
        """An independent sub-stream keyed by `label` (same entropy -> same sub-stream)."""
        h = hashlib.sha256("{}:{}".format(self.entropy, label).encode()).digest()
        return SeedStream(entropy=struct.unpack("<Q", h[:8])[0], label=label)


def load_pinned(path):
    """Parse a pinned-seed file: {"entropy_hi", "entropy_lo"} or {"seeds": [...]}. None if absent."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    if "seeds" in doc:
        return {"seeds": [int(s) for s in doc["seeds"]]}
    if "entropy_hi" in doc and "entropy_lo" in doc:
        return {"entropy": (int(doc["entropy_hi"]) << 32) | int(doc["entropy_lo"])}
    raise ValueError("{}: expected 'seeds' or 'entropy_hi'/'entropy_lo'".format(path))


def write_pinned(stream, path, note=None):
    """Write the pin file that reproduces `stream` (entropy halves), for reverification."""
    doc = {"entropy_hi": stream.hi, "entropy_lo": stream.lo, "note": note or
           "pins the hidden seed stream of one verify run; reverification only, never shipped with a published task"}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    return path


def seed_rng(entropy=None, pinned=None):
    """The grader's hidden seed source. Call once per verify run.

    `entropy` is for self-tests. `pinned` is a path to `tests/hidden/seeds.json`, a parsed
    pin dict, or a list of seeds; absent (the published task) -> os.urandom.
    """
    if isinstance(pinned, str):
        pinned = load_pinned(pinned)
    if isinstance(pinned, (list, tuple)):
        pinned = {"seeds": list(pinned)}
    if pinned:
        if "seeds" in pinned:
            return SeedStream(pinned_seeds=pinned["seeds"])
        return SeedStream(entropy=pinned["entropy"])
    return SeedStream(entropy)
