"""aerial_search_geolocation: the probe lock counts a pair of replies the way the instruction does.

The instruction allows two reply shapes, a bare move (`[move]`, or the move as a number) and the full report of
1 + 4 x 3 numbers, and the run scores 0 if fewer than 25 % of the probe pairs "get two different replies". The kit's
lock compares values only (two replies with a different count of numbers do not differ for it), so the task's
verify.py gives it a fixed-length form of each reply (`canonical=`). The lock is built here by verify.main() itself,
with its outside world stubbed up to that call, and fed with the repository's kit (shared/robotics_grader):

- a bare move against a full report differs, a bare move as a number or as a one-element list is the same reply;
- two replies with the same count of numbers are compared by value with the kit's tolerance;
- a count of numbers that is not 1 or 13 still differs from a valid one; it is the real reply's `parse_reply` that
  aborts the mission for it, and the lock's form never raises for it;
- a reply that is not a nest of numbers still aborts as the kit's "not numeric";
- an honest controller that sends a bare move while it knows nothing for the asked targets passes every probe.

No simulator, GPU or network: the solver is a stand-in object with the kit's `act(t, obs)`.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK = REPO_ROOT / "tasks/uav/aerial_search_geolocation"
TESTS = TASK / "tests"
CONFIG = json.loads((TESTS / "grader_config.json").read_text())

# the task's module names, which other tests may import from their own folders: imported here with those set
# aside, and taken out of sys.modules again afterwards
_NAMES = ("robotics_grader", "sar_protocol", "sim_client")


def _load():
    saved_path, saved_flag = list(sys.path), sys.dont_write_bytecode
    saved = {n: sys.modules.pop(n) for n in list(sys.modules) if n.split(".")[0] in _NAMES}
    sys.dont_write_bytecode = True  # (no __pycache__ in the repository or the task folder)
    try:
        sys.path.insert(0, str(REPO_ROOT / "shared"))
        from robotics_grader import seeds, solver  # noqa: E402  (the repository's kit, before verify.py's own path)
        sys.path.insert(0, str(TESTS))
        import sar_protocol  # noqa: E402
        spec = importlib.util.spec_from_file_location("aerial_search_geolocation_verify", TESTS / "verify.py")
        verify = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verify)
        assert verify.ProbeLock is solver.ProbeLock
        return seeds, solver, sar_protocol, verify
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_flag
        for n in [n for n in sys.modules if n.split(".")[0] in _NAMES]:
            del sys.modules[n]
        sys.modules.update(saved)


seeds, kit, sar_protocol, verify = _load()


class _Built(Exception):
    pass


def lock_kwargs():
    """The keyword arguments verify.main() passes to ProbeLock (main() is stopped there)."""
    seen = {}

    def built(*args, **kwargs):
        seen.update(kwargs)
        raise _Built()

    with tempfile.TemporaryDirectory() as home:
        os.makedirs(os.path.join(home, "submission"))
        open(os.path.join(home, "submission", "run.sh"), "w").close()
        pool = {"maps": {"m": {"slots": [[0.0, 0.0, 0.0]], "targets": [{"description": "a red fire truck"}]}}}
        stubs = {
            "stage_config": lambda: dict(CONFIG),
            "seed_rng": lambda: object(),
            "read_anchor": lambda: (None, {}),
            "agent_home": lambda: home,
            "check_pool_private": lambda path: None,
            "load_pool": lambda path: pool,
            "assemble_missions": lambda *a, **k: [{"map_name": "m", "targets": []}],
            "token_state": lambda: (False, False),
            "guarded_entry": lambda python: home,
            "log": lambda message: None,
            "ProbeLock": built,
        }
        with mock.patch.multiple(verify, **stubs):
            try:
                verify.main()
            except _Built:
                pass
    return seen


KWARGS = lock_kwargs()
N = int(CONFIG["targets_per_mission"])
FULL = [1, 1, 12.0, -3.0, 1.5] + [0] * (4 * N - 4)


def full(index, value):
    out = list(FULL)
    out[index] = value
    return out


def record(probe, real):
    lock = kit.ProbeLock(seeds.seed_rng(entropy=7), **KWARGS)
    return lock.record(probe, real)


DIFFER = [
    ("bare move [1] vs full report", [1], FULL),
    ("bare move 1 (a number) vs full report", 1, FULL),
    ("bare move vs a full report of zeros with the same move", [1], [1] + [0] * (4 * N)),
    ("two bare moves, different", [1], [6]),
    ("two bare moves as numbers, different", 1, 6),
    ("bare move as a number vs a different one-element list", 1, [6]),
    ("two full reports, 2e-4 apart in one coordinate", FULL, full(2, 12.0 + 2e-4)),
    ("two full reports, the move differs only", FULL, full(0, 6)),
    ("two numbers vs full report", [1, 1], FULL),
    ("two numbers vs bare move", [1, 1], [1]),
    ("one number too many vs full report", FULL + [0], FULL),
    ("no number vs bare move", [], [1]),
    ("the same wrong count, different values", [1, 1], [1, 2]),
]
SAME = [
    ("two bare moves, equal", [1], [1]),
    ("bare move as a number vs the same one-element list", 1, [1]),
    ("two full reports, equal", FULL, list(FULL)),
    ("two full reports, 5e-5 apart in one coordinate", FULL, full(2, 12.0 + 5e-5)),
    ("the same wrong count, equal values", [1, 1], [1, 1]),
    ("nested lists the kit flattens vs the full report they hold", [[1], [1, 12.0, -3.0, 1.5], [0] * (4 * N - 4)], FULL),
]
NOT_NUMERIC = [
    ("a string entry", [1, "a"], FULL),
    ("a string entry after the full report's length", FULL + ["a"], FULL),
    ("null", None, [1]),
    ("an object", {"move": 1}, [1]),
    ("an integer too large for a float", [10 ** 400], [1]),
]


class ProbeLockTests(unittest.TestCase):
    def test_verify_builds_the_lock_with_the_tasks_rate_threshold_and_canonical_form(self):
        self.assertEqual(KWARGS.get("rate"), CONFIG["probe_rate"])
        self.assertEqual(KWARGS.get("pass_fraction"), CONFIG["probe_pass_fraction"])
        self.assertTrue(callable(KWARGS.get("canonical")), "verify.py gives the lock no canonical form")

    def test_pairs_the_instruction_calls_different_differ(self):
        for name, a, b in DIFFER:
            with self.subTest(name):
                self.assertTrue(record(a, b))
                self.assertTrue(record(b, a))

    def test_pairs_of_the_same_reply_do_not_differ(self):
        for name, a, b in SAME:
            with self.subTest(name):
                self.assertFalse(record(a, b))
                self.assertFalse(record(b, a))

    def test_a_reply_that_is_not_numbers_aborts_as_before(self):
        for name, a, b in NOT_NUMERIC:
            with self.subTest(name):
                with self.assertRaisesRegex(kit.SolverAborted, "not numeric"):
                    record(a, b)
                with self.assertRaisesRegex(kit.SolverAborted, "not numeric"):
                    record(b, a)

    def test_a_wrong_count_is_left_to_parse_reply(self):
        canonical = KWARGS.get("canonical")
        self.assertTrue(callable(canonical))
        sizes = set()
        for count in range(0, 4 * (1 + 4 * N)):
            reply = [1] + [0] * (count - 1) if count else []
            sizes.add(len(canonical(reply)))
            if count not in (1, 1 + 4 * N):
                with self.assertRaises(sar_protocol.ProtocolError):
                    sar_protocol.parse_reply(reply, N)
            else:
                sar_protocol.parse_reply(reply, N)
        self.assertEqual(sizes, {2 + 4 * N})
        self.assertEqual(len(canonical(1)), 2 + 4 * N)

    def test_an_honest_two_shape_controller_passes_every_probe(self):
        """Sends a bare move while it has located none of the asked targets, the full report otherwise."""
        located = {"a red fire truck": [12.0, -3.0, 1.5]}

        class TwoShapes(object):
            def act(self, t, obs):
                asked = [x["description"] for x in obs["targets"]]
                if not any(d in located for d in asked):
                    return [1]
                action = [1]
                for d in asked:
                    action += [1] + located[d] if d in located else [0, 0, 0, 0]
                return action

        lock = kit.ProbeLock(seeds.seed_rng(entropy=20261120), **KWARGS)
        obs = {"targets": [{"description": d} for d in ("a red fire truck", "a round hay bale", "a blue tent")]}
        decoys = ["a beach table", "a brown and white horse", "a scooter"]
        for t in range(40):
            real = lock.probe(TwoShapes(), t, obs, perturbed=sar_protocol.perturb_targets(obs, decoys))
            self.assertEqual(sar_protocol.parse_reply(real, N)[0], 1)
        self.assertEqual((lock.passed, lock.total), (40, 40))
        self.assertTrue(lock.ok)


if __name__ == "__main__":
    unittest.main()
