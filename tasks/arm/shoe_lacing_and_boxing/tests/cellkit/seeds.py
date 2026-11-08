"""Hidden seeds: this task's fixed set, or a stream from a logged entropy (CHAIN_PLAN.md D6).

This task grades a FIXED hidden set (2026-11-06, the owner's scoring rule for contributed tasks):
`tests/hidden/seeds.json` (`{"seeds": [...]}`, in the root-only stage folder, never on a path the
agent's UID can stat) lists the episodes' seeds, which were drawn once from os.urandom and selected
by a pre-registered rule (`tests/review/select_set.py`, the record in `tests/hidden/selection.json`);
the task ships it, every submission meets the same episodes, and `verify.py` (`hidden_set`) refuses
to grade without it. The rule of shared/robotics_grader/seeds.py (seeds sampled fresh inside the
grader process on every run) is what this task used before. What this module offers:

* `load_pinned` / `seed_rng(pinned=...)`: a pin file either lists the seeds (`{"seeds": [...]}`:
  the fixed set; `seed()` / `seeds()` return them in order and refuse to run past the end) or pins
  the entropy of a stream (`{"entropy_hi", "entropy_lo"}`: the reverification of a run drawn from
  entropy). A pinned list is not a stream: `derive()` of a pinned stream hashes the list into OTHER
  seeds, so a consumer of the fixed set reads the list itself (verify.py does);
* without a pin, a stream from 64 bits of OS entropy, exposed as two 32-bit halves so a verify
  stage can log them as stats (`hidden_seed_hi`, `hidden_seed_lo`; for a pinned list they are the
  halves of the list's sha256): the whole drawn stream is reproducible from the two numbers
  (splitmix64 is deterministic), without a seed file inside the sandbox;
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
           "pins the seed stream of one run drawn from entropy, for its reverification; not a fixed hidden set"}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    return path


def seed_rng(entropy=None, pinned=None):
    """A hidden seed source. Call once per run.

    `entropy` is for self-tests. `pinned` is a path to a pin file (`tests/hidden/seeds.json`), a
    parsed pin dict, or a list of seeds; absent -> os.urandom (this task's verify never grades
    without its fixed set: verify.hidden_set).
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
