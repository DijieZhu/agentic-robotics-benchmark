"""Per-run hidden seeds.

The rule the whole benchmark rests on: the seeds an episode is scored under are sampled
fresh, inside the grader process, after the agent is gone — and the values never reach a
path the agent's UID can stat. Anything weaker turns "solve the task" into "memorize the
evaluation".

A stage script gets an integer stream from `seed_rng` and nothing else. There is no
seed file, no environment variable carrying the seed, and no derivation from anything the
agent could have observed.

The stream must also be unpredictable from its own outputs. One grader process draws the
episode seeds, the probe schedule and the probe perturbations from it, and the agent's
solver sees the perturbations: at a probed step it receives the real and the perturbed
observation, and their difference is a draw. A stream whose outputs reveal its state
(splitmix64, any LCG or Mersenne Twister) would let a solver reconstruct it from a few
probes and compute the next episode's seed. So every draw is a keyed BLAKE2b block: the
key is fresh OS entropy, and past outputs say nothing about the key or about later draws.
"""

import hashlib
import os
import struct

__all__ = ["seed_rng"]


class _SeedStream(object):
    """A deterministic-per-instance, unpredictable integer source.

    Keyed from OS entropy, so two runs of the same task score different worlds and no
    output predicts another; drawn by a fixed algorithm (BLAKE2b in counter mode), so one
    grader process can reproduce its own stream if it needs to replay an episode within
    the run. stdlib only — a kit cannot assume numpy.
    """

    def __init__(self, entropy=None):
        if entropy is None:
            self._key = os.urandom(32)
        else:  # the self-test's reproducible streams
            self._key = hashlib.blake2b(repr(int(entropy)).encode("ascii"), digest_size=32).digest()
        self._counter = 0

    def _next(self):
        self._counter += 1
        block = hashlib.blake2b(struct.pack("<Q", self._counter), key=self._key, digest_size=8)
        return struct.unpack("<Q", block.digest())[0]

    def seed(self, bits=31):
        """One seed, sized for the consumer (gymnasium wants < 2**31).

        Use `bits=64` when the agent holds a copy of the simulator: then a 31-bit seed can
        be found by brute force from an episode's first observations.
        """
        if not 1 <= int(bits) <= 64:
            raise ValueError("seed bits must be 1 to 64, got {!r}".format(bits))
        return self._next() & ((1 << int(bits)) - 1)

    def seeds(self, count, bits=31):
        return [self.seed(bits) for _ in range(count)]

    def random(self):
        """A float in [0, 1) — for probe scheduling, not for physics."""
        return (self._next() >> 11) * (1.0 / (1 << 53))

    def uniform(self, low, high):
        return low + (high - low) * self.random()

    def choice(self, items):
        if not items:
            raise IndexError("choice from an empty sequence")
        return items[self._next() % len(items)]


def seed_rng(entropy=None):
    """The grader's own hidden seed source. Call once per verify run.

    `entropy` exists for the kit's self-test; a real verify stage passes nothing, so the
    stream comes from `os.urandom` and lives only in this process's memory.
    """
    return _SeedStream(entropy)
