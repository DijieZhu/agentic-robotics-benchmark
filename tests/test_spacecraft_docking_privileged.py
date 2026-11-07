"""fault_adaptive_spacecraft_docking: the privileged reference channel, the fixed hidden episode set, the network
guard, the sensors' observation and what the agent is given.

The grader hands the simulator's current truth only to a submission whose .oracle_token (a regular file, never a
link) hashes to tests/hidden/oracle_token.sha256; nothing agent-visible names the channel; building the block draws
nothing and changes nothing; the block carries every quantity env.step reads that is in effect now and nothing drawn
for the future; the hidden episodes are a fixed set of 48 distinct 64-bit seeds, each pinned by its world digest; an
episode's probe stream depends only on the set's key and its seed; the reference reproduces its recorded result on the
set; a verify that cannot install the network filter, or cannot render the camera, is a task error before any
controller starts. Round 2 (2026-11-07): the observation is what the chaser's sensors deliver (the docking camera's
frame, the star tracker, the IMU, the boom's telemetry, the fuel gauge, the clock), built on env.py's unchanged
observation without a draw of its own from env.py's generator: a frame is missing exactly when env.py's pose was not
fresh, the star tracker exactly during the re-key's restart, the probe copy shifts the frame's craft and the star
tracker by the lock's three draws; the controller's replies share one time budget and a request's write is bounded by
its deadline. The agent gets a datasheet, not an executable model: every range it states contains env.py's values
and its constants, and it withholds the grader's formulas and its noise and dropout laws; the sensors' documents
carry camera.py's numbers.

The renderer (Filament on lavapipe, the camera bundle) is not needed here: the observation is tested with stand-in
cameras; the frame determinism test runs only where the renderer and the bundle are installed.
"""
import ast
import copy
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

TASK = Path(__file__).resolve().parents[1] / "tasks/space/fault_adaptive_spacecraft_docking"
MB_NONET = Path(__file__).resolve().parents[1] / "tasks/arm/arm_hidden_mechanism_box/tests/nonet_exec.py"
GIVEN = TASK / "environment/setup/payload/given"


def _load():
    saved_path, saved_flag = list(sys.path), sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # (no __pycache__ in the task folder)
    try:
        for d in ("solution/privileged", "tests/review", "tests"):
            sys.path.insert(0, str(TASK / d))
        import oracle  # noqa: E402
        import oracle_batch  # noqa: E402
        import verify  # noqa: E402
        import camera  # noqa: E402
        import make_sample_frames  # noqa: E402
        return verify, oracle, oracle_batch, camera, make_sample_frames
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_flag


verify, oracle, oracle_batch, camera, make_sample_frames = _load()
SET = json.loads((TASK / "tests/hidden/episodes.json").read_text())
SELECTION = json.loads((TASK / "tests/hidden/selection.json").read_text())
HIDDEN_SEEDS = {e["seed"] for e in SET["episodes"]} | {c["seed"] for c in SELECTION["candidates"]}


def scripted(t, seed):
    """A deterministic command that is not a controller: mostly small duties, a few large ones."""
    return [((t * 7 + i * 13 + seed % 97) % 23) / 22.0 * (0.3 if (t + i) % 5 else 1.0) for i in range(6)]


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class RecordingCamera(object):
    """A stand-in for camera.Camera: remembers what it was asked to render and returns bytes that encode the call."""

    def __init__(self):
        self.calls = []

    def frame(self, key, seed, stamp, x, y, theta, phase, port_radius):
        self.calls.append((stamp, x, y, theta, phase, port_radius))
        tag = hashlib.sha256(repr((key, seed, stamp, x, y, theta, phase, port_radius)).encode()).digest()
        return (tag * (camera.FRAME_BYTES // len(tag) + 1))[:camera.FRAME_BYTES]


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sub = os.path.join(self.tmp.name, "submission")
        os.makedirs(self.sub)
        self.token = b"0123456789abcdef" * 4
        self.digest = os.path.join(self.tmp.name, "oracle_token.sha256")
        with open(self.digest, "w") as f:
            f.write(hashlib.sha256(self.token).hexdigest() + "\n")
        self.path = os.path.join(self.sub, ".oracle_token")

    def tearDown(self):
        self.tmp.cleanup()

    def state(self):
        return verify.token_state(self.sub, digest_path=self.digest)

    def test_no_token_is_neither(self):
        self.assertEqual(self.state(), (False, False))

    def test_matching_token_is_privileged(self):
        with open(self.path, "wb") as f:
            f.write(self.token + b"\n")
        self.assertEqual(self.state(), (True, False))

    def test_wrong_token_is_a_mismatch(self):
        with open(self.path, "wb") as f:
            f.write(b"f" * 64)
        self.assertEqual(self.state(), (False, True))

    def test_a_link_is_never_followed(self):
        real = os.path.join(self.tmp.name, "real_token")
        with open(real, "wb") as f:
            f.write(self.token)
        os.symlink(real, self.path)
        self.assertEqual(self.state(), (False, True))

    def test_a_directory_a_fifo_or_a_huge_file_is_a_mismatch(self):
        os.makedirs(self.path)
        self.assertEqual(self.state(), (False, True))
        os.rmdir(self.path)
        os.mkfifo(self.path)  # (opened without blocking: no writer needed, no hang)
        self.assertEqual(self.state(), (False, True))
        os.remove(self.path)
        with open(self.path, "wb") as f:
            f.write(self.token + b" " * (verify.TOKEN_MAX_BYTES + 10))
        self.assertEqual(self.state(), (False, True))

    def test_the_reference_token_matches_the_digest(self):
        token = (TASK / "solution/privileged/oracle_token").read_bytes().strip()
        self.assertEqual(len(token), 64)  # 256 bits, hex
        want = (TASK / "tests/hidden/oracle_token.sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(token).hexdigest(), want)
        self.assertEqual(verify.TOKEN_DIGEST, str(TASK / "tests/hidden/oracle_token.sha256"))

    def test_nothing_agent_visible_names_the_channel_or_the_set(self):
        token = (TASK / "solution/privileged/oracle_token").read_text().strip()
        digest = (TASK / "tests/hidden/oracle_token.sha256").read_text().split()[0]
        needles = [token, digest, ".oracle_token", "oracle_token", "privileged_mode", '"privileged"',
                   verify.PRIV_FORMAT, verify.SET_FORMAT, SET["probe_key"], "episodes.json", "selection.json"]
        needles += [str(s) for s in HIDDEN_SEEDS] + ["%016x" % s for s in HIDDEN_SEEDS]
        visible = [TASK / "instruction.md"] + [p for p in (TASK / "environment").rglob("*") if p.is_file()]
        for p in visible:
            text = p.read_text(errors="replace")
            for needle in needles:
                self.assertNotIn(needle, text, "{} mentions {}".format(p.relative_to(TASK), needle[:16]))
        for p in [TASK / "instruction.md"] + sorted(GIVEN.rglob("*.md")):   # the documents the agent reads
            for word in ("privileged", "token", "oracle", "hidden set", "fixed set"):
                self.assertIsNone(re.search(r"\b{}\b".format(word), p.read_text(), re.I),
                                  "{} says {!r}".format(p.relative_to(TASK), word))


class EpisodeSetTests(unittest.TestCase):
    def test_fixed_set_is_48_distinct_64_bit_seeds(self):
        seeds, key = verify.load_episodes(str(TASK / "tests/hidden/episodes.json"))
        self.assertEqual(len(seeds), 48)  # instruction.md: "48 hidden episodes"
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertTrue(all(2 ** 32 <= s < 2 ** 64 for s in seeds))
        self.assertTrue(0 < key < 2 ** 256)
        self.assertEqual(SET["format"], verify.SET_FORMAT)
        for e in SET["episodes"]:
            self.assertEqual(verify.world_digest(e["seed"]), e["world_sha256"])
        self.assertTrue("48 hidden episodes" in (TASK / "instruction.md").read_text(),
                        "instruction.md does not say '48 hidden episodes'")

    def test_the_set_decides_the_count(self):
        self.assertNotIn("episodes", json.loads((TASK / "tests/grader_config.json").read_text()))

    def test_a_changed_world_is_a_task_error(self):
        spec = copy.deepcopy(SET)
        spec["episodes"][3]["world_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "episodes.json")
            with open(path, "w") as f:
                json.dump(spec, f)
            with self.assertRaises(SystemExit):
                verify.load_episodes(path)

    def test_the_selection_record_matches_the_set(self):
        kept = [c for c in sorted(SELECTION["candidates"], key=lambda c: c["draw"]) if c["status"] == "kept"]
        self.assertEqual([c["seed"] for c in kept], [e["seed"] for e in SET["episodes"]])
        self.assertEqual(SELECTION["probe_key"], SET["probe_key"])
        self.assertTrue(all(c["success"] and not c["aborted"] and c["inprocess_equal"] for c in kept))

    def test_probe_stream_depends_on_key_and_seed_only(self):
        seeds, key = verify.load_episodes(str(TASK / "tests/hidden/episodes.json"))
        a = verify.episode_stream(key, seeds[0])
        b = verify.seed_rng(entropy=(key << 64) | seeds[0])
        self.assertEqual([a.random() for _ in range(50)], [b.random() for _ in range(50)])
        c = verify.episode_stream(key, seeds[1])
        d = verify.episode_stream(key, seeds[0])
        self.assertNotEqual([c.random() for _ in range(5)], [d.random() for _ in range(5)])

    def test_the_samples_and_the_example_are_dev_episodes_with_a_dev_noise_key(self):
        seeds = [s for _, s in make_sample_frames.EPISODES] + [make_sample_frames.ACT_SEED]
        self.assertFalse(set(seeds) & HIDDEN_SEEDS)
        text = (TASK / "tests/hidden/episodes.json").read_text() + (TASK / "tests/hidden/selection.json").read_text()
        for s in seeds:
            self.assertNotIn(str(s), text)
            self.assertNotIn("%x" % s, text)
        self.assertNotEqual(make_sample_frames.DEV_NOISE_KEY, int(SET["probe_key"], 16))
        self.assertNotEqual(oracle_batch.DEV_KEY, SET["probe_key"])


def _env_vars(env):
    d = {k: copy.deepcopy(v) for k, v in vars(env).items() if k != "rng"}
    d["_rng_counter"] = env.rng._counter
    return d


class BlockTests(unittest.TestCase):
    SEEDS = [(1 << 33) + 7 * i for i in range(20)]

    def test_building_the_block_draws_nothing_and_changes_nothing(self):  # (I1)
        for seed in self.SEEDS[:5]:
            env = verify.FaultAdaptiveDocking()
            env.reset(seed)
            for t in range(env.max_steps):
                before = _env_vars(env)
                block = verify.privileged_block(env)
                self.assertEqual(_env_vars(env), before)
                self.assertEqual(block["format"], verify.PRIV_FORMAT)
                self.assertEqual(set(block["state"]),
                                 set(verify.PRIV_STATE + verify.PRIV_ROUTING + verify.PRIV_CONSTANTS))
                self.assertEqual(json.loads(json.dumps(block)), block)  # (floats round-trip exactly)
                if env.step(scripted(t, seed))[2]:
                    break

    def test_the_block_covers_what_step_reads_and_nothing_else_is_unclassified(self):
        sent = set(verify.PRIV_STATE + verify.PRIV_ROUTING + verify.PRIV_CONSTANTS)
        never = set(verify.NEVER_SENT_FUTURE + verify.NEVER_SENT_OBSERVATION + verify.NEVER_SENT_OUTCOME)
        self.assertFalse(sent & never)
        self.assertEqual(oracle.STATE, verify.PRIV_STATE)
        self.assertEqual(oracle.ROUTING, verify.PRIV_ROUTING)
        self.assertEqual(oracle.PRIV_FORMAT, verify.PRIV_FORMAT)
        tree = ast.parse((TASK / "tests/env.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FaultAdaptiveDocking")
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        class_attrs = {t.id for n in cls.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        read = set()
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name in ("step", "_plume_column", "goal_state", "port_radius",
                                                               "episode_quality"):
                for node in ast.walk(fn):
                    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                            and node.value.id == "self" and isinstance(node.ctx, ast.Load)):
                        read.add(node.attr)
        read -= methods | class_attrs
        self.assertEqual(read - sent - never, set(), "read by the plant but neither sent nor withheld")
        self.assertEqual(sent - read, set(), "sent but never read by the plant")
        env = verify.FaultAdaptiveDocking()
        env.reset(self.SEEDS[0])
        for t in range(200):
            env.step(scripted(t, self.SEEDS[0]))
        self.assertEqual(set(vars(env)) - sent - never, set(), "an attribute of env.py that the block does not classify")

    def test_a_replica_of_the_block_predicts_the_next_block_exactly_after_the_rekey(self):
        counts = {}
        for seed in self.SEEDS:
            env = verify.FaultAdaptiveDocking()
            env.reset(seed)
            prev = None
            for t in range(env.max_steps):
                block = json.loads(json.dumps(verify.privileged_block(env)))["state"]
                if prev is not None:
                    pst, pu = prev
                    r = oracle.replica(pst)
                    r.step(pu)
                    ok = oracle.same_state(r, block)
                    if pst["rekey_active"]:
                        phase = "after_rekey"
                    elif pst["routing_active"]:
                        phase = "between" if not block["rekey_active"] else "rekey_transition"
                    else:
                        phase = "before_events" if not block["routing_active"] else "event_transition"
                    counts[(phase, ok)] = counts.get((phase, ok), 0) + 1
                u = scripted(t, seed)
                prev = (block, u)
                if env.step(u)[2]:
                    break
        for phase in ("before_events", "between", "after_rekey"):
            self.assertEqual(counts.get((phase, False), 0), 0, phase)
            self.assertGreater(counts.get((phase, True), 0), 0, phase)
        self.assertEqual(counts.get(("rekey_transition", True), 0), 0)
        self.assertEqual(counts.get(("event_transition", True), 0), 0)


class FakeSolver(object):
    """A scripted program on the wire: it records every request exactly as the kit serialises it and answers with an
    action computed from the sensors' fields (so the episode is non-trivial and every probe pair differs)."""

    def __init__(self):
        self.requests = []

    def reset(self):
        self.requests.append(json.dumps({"type": "reset"}))

    def act(self, t, obs):
        self.requests.append(json.dumps({"type": "act", "t": int(t), "obs": obs}))
        yaw = obs["star_tracker_yaw"] if obs["star_tracker_yaw"] is not None else 0.3
        data = obs["frame"]["data"] if obs["frame"] is not None else ""
        h = int(hashlib.sha256(data.encode()).hexdigest()[:8], 16) / 2.0 ** 32
        return [abs(math.sin(3.0 * yaw + 2.0 * obs["imu"][0] + h + 0.7 * i)) * 0.4 for i in range(6)]


def _wire_at(seed, steps, cam=None, privileged=False, key=12345):
    """(env, sensors, wire, rec) after `steps` scripted steps of a dev seed."""
    env = verify.FaultAdaptiveDocking()
    sensors = verify.Sensors(cam or RecordingCamera(), key, seed)
    obs16 = env.reset(seed)
    for t in range(steps):
        obs16 = env.step(scripted(t, seed))[0]
    wire, rec = sensors.observe(env, obs16)
    if privileged:
        wire["privileged"] = verify.privileged_block(env)
    return env, sensors, wire, rec


class ObservationTests(unittest.TestCase):
    SEEDS = [(1 << 34) + 11 * i for i in range(12)]

    def test_the_request_is_the_documented_object_and_the_token_adds_the_block_last(self):
        seed, key = SET["episodes"][1]["seed"], int(SET["probe_key"], 16)
        runs = []
        for privileged in (False, True):
            sol = FakeSolver()
            lock = verify.ProbeLock(None, rate=0.06, pass_fraction=0.35, tolerance=1e-4)
            lock.rng = verify.episode_stream(key, seed)
            verify.run_episode(seed, sol, lock, privileged, RecordingCamera(), key)
            runs.append([json.loads(r) for r in sol.requests])
        plain, token = runs
        self.assertEqual(len(plain), len(token))
        for a, b in zip(plain, token):
            if a["type"] != "act":
                self.assertEqual(a, b)
                continue
            self.assertEqual(tuple(a["obs"]), verify.OBS_KEYS)
            self.assertEqual(tuple(b["obs"]), verify.OBS_KEYS + ("privileged",))
            self.assertEqual(b["obs"]["privileged"]["format"], verify.PRIV_FORMAT)
            del b["obs"]["privileged"]
            self.assertEqual(a, b)                       # the token changes nothing else
        acts = [r for r in plain if r["type"] == "act"]
        fr = next(r["obs"]["frame"] for r in acts if r["obs"]["frame"] is not None)
        self.assertEqual(set(fr), {"stamp", "data"})
        self.assertEqual(len(fr["data"]), 409600)
        line = json.dumps(acts[5])
        self.assertLess(len(line), verify.PIPE_BYTES // 2)

    def test_no_pose_truth_in_the_observation(self):
        env, sensors, wire, rec = _wire_at(self.SEEDS[0], 30)
        flat = json.dumps({k: v for k, v in wire.items() if k != "frame"})
        for name in ("x", "y", "theta", "vx", "vy", "omega", "phase", "pose", "port_x", "valid", "age"):
            self.assertNotIn('"{}"'.format(name), flat)
        self.assertEqual(set(wire), set(verify.OBS_KEYS))
        self.assertEqual(len(wire["imu"]), 3)
        for k in ("fuel", "episode_fraction", "boom_encoder"):
            self.assertIsInstance(wire[k], float)

    def test_a_frame_is_missing_exactly_when_env_s_pose_was_not_fresh(self):
        counts = {"fresh": 0, "missing": 0}
        for seed in self.SEEDS:
            env = verify.FaultAdaptiveDocking()
            cam = RecordingCamera()
            sensors = verify.Sensors(cam, 777, seed)
            obs16 = env.reset(seed)
            for t in range(env.max_steps):
                wire, rec = sensors.observe(env, obs16)
                if obs16[13] == 1.0:
                    counts["fresh"] += 1
                    stamp = env.steps - int(obs16[12])
                    self.assertEqual(wire["frame"]["stamp"], stamp)
                    self.assertEqual(stamp, max(0, env.steps - env.pose_delay))
                    self.assertEqual(rec[0], stamp)
                    # env.py's delayed pose (round 1's fields 0-7) came from this very record: within its noise
                    self.assertLessEqual(abs(obs16[0] - rec[1]), 0.012)
                    self.assertLessEqual(abs(obs16[1] - rec[2]), 0.012)
                    self.assertLessEqual(abs(obs16[4] - rec[4]), 0.007)
                    self.assertLessEqual(abs(obs16[5] - rec[5]), 0.007)
                    self.assertEqual(cam.calls[-1][0], stamp)
                    self.assertEqual(cam.calls[-1][1:4], rec[1:4])
                    self.assertEqual(cam.calls[-1][4], rec[6])
                    self.assertEqual(cam.calls[-1][5], math.hypot(rec[4], rec[5]))
                else:
                    counts["missing"] += 1
                    self.assertIsNone(wire["frame"])
                self.assertEqual(wire["imu"], obs16[8:11])
                self.assertEqual((wire["fuel"], wire["episode_fraction"], wire["boom_encoder"]),
                                 (obs16[11], obs16[14], obs16[15]))
                obs16, _, done, _ = env.step(scripted(t, seed))
                if done:
                    break
        self.assertGreater(counts["fresh"], 0)
        self.assertGreater(counts["missing"], 0)

    def test_the_star_tracker_is_missing_exactly_in_the_restart(self):
        restarts = []
        for seed in self.SEEDS:
            env = verify.FaultAdaptiveDocking()
            sensors = verify.Sensors(RecordingCamera(), 777, seed)
            obs16 = env.reset(seed)
            out = 0
            for t in range(env.max_steps):
                wire, _ = sensors.observe(env, obs16)
                event_blackout = (env.rekey_active and env.rekey_step <= env.steps
                                  < env.rekey_step + env.rekey_blackout_duration)
                self.assertEqual(wire["star_tracker_yaw"] is None, event_blackout)
                if event_blackout:
                    out += 1
                    self.assertIsNone(wire["frame"])
                else:
                    self.assertLessEqual(abs(wrap(wire["star_tracker_yaw"] - env.theta)), 6 * camera.STAR_TRACKER_SIGMA)
                obs16, _, done, _ = env.step(scripted(t, seed))
                if done:
                    break
            if env.rekey_active and env.steps >= env.rekey_step + env.rekey_blackout_duration:
                restarts.append(out)
        self.assertTrue(restarts)
        self.assertTrue(all(29 <= n <= 41 for n in restarts), restarts)

    def test_the_sensors_draw_nothing_from_env_and_change_nothing(self):
        seed = self.SEEDS[3]
        a, b = verify.FaultAdaptiveDocking(), verify.FaultAdaptiveDocking()
        sensors = verify.Sensors(RecordingCamera(), 5, seed)
        oa, ob = a.reset(seed), b.reset(seed)
        for t in range(200):
            before = _env_vars(a)
            sensors.observe(a, oa)
            self.assertEqual(_env_vars(a), before)
            oa = a.step(scripted(t, seed))[0]
            ob = b.step(scripted(t, seed))[0]
            self.assertEqual(oa, ob)

    def test_the_star_tracker_noise_is_keyed_and_gaussian(self):
        z = [camera.star_tracker_noise(99, 4242, k) for k in range(4000)]
        self.assertEqual(z[:5], [camera.star_tracker_noise(99, 4242, k) for k in range(5)])
        self.assertNotEqual(z[:5], [camera.star_tracker_noise(98, 4242, k) for k in range(5)])
        mean = sum(z) / len(z)
        var = sum((v - mean) ** 2 for v in z) / len(z)
        self.assertLess(abs(mean), 0.06)
        self.assertLess(abs(var - 1.0), 0.08)

    def test_the_copy_draws_three_numbers_and_shifts_the_frame_and_the_star_tracker(self):  # (I2)
        for privileged in (False, True):
            cam = RecordingCamera()
            env, sensors, wire, rec = _wire_at(self.SEEDS[1], 25, cam=cam, privileged=privileged)
            self.assertIsNotNone(wire["frame"])
            real = json.loads(json.dumps(wire))
            rng = verify.seed_rng(entropy=5)
            copy_obs = verify.perturbed_observation(wire, rng, sensors.shifted(rec))
            self.assertEqual(rng._counter, 3)
            ref = verify.seed_rng(entropy=5)
            dx, dy, dyaw = ref.uniform(-0.28, 0.28), ref.uniform(-0.28, 0.28), ref.uniform(-0.18, 0.18)
            stamp, x, y, th, gx, gy, gyaw = rec
            self.assertEqual(cam.calls[-1], (stamp, x + dx, y + dy, wrap(th + dyaw), gyaw, math.hypot(gx, gy)))
            self.assertEqual(copy_obs["frame"]["stamp"], stamp)
            self.assertNotEqual(copy_obs["frame"]["data"], wire["frame"]["data"])
            self.assertEqual(copy_obs["star_tracker_yaw"], wrap(wire["star_tracker_yaw"] + dyaw))
            for k in ("imu", "fuel", "episode_fraction", "boom_encoder"):
                self.assertEqual(copy_obs[k], wire[k])
            self.assertEqual(tuple(copy_obs), tuple(wire))
            self.assertEqual(json.loads(json.dumps(wire)), real)   # (the real request is untouched)
            if privileged:
                st, st0 = copy_obs["privileged"]["state"], wire["privileged"]["state"]
                self.assertEqual(st["x"], st0["x"] + dx)
                self.assertEqual(st["y"], st0["y"] + dy)
                self.assertEqual(st["theta"], wrap(st0["theta"] + dyaw))
                self.assertEqual({k: v for k, v in st.items() if k not in ("x", "y", "theta")},
                                 {k: v for k, v in st0.items() if k not in ("x", "y", "theta")})

    def test_a_request_without_a_frame_or_a_star_tracker_has_a_copy_without_them(self):
        obs = {"imu": [0.1, 0.2, 0.3], "fuel": 0.9, "episode_fraction": 0.5, "boom_encoder": 1.1,
               "star_tracker_yaw": None, "frame": None}
        rng = verify.seed_rng(entropy=9)

        def never(*_):
            raise AssertionError("rendered a copy for a request without a frame")
        out = verify.perturbed_observation(obs, rng, never)
        self.assertEqual(out, obs)
        self.assertEqual(rng._counter, 3)

    def test_both_modes_consume_the_lock_stream_identically(self):  # (I2)
        seed, key = SET["episodes"][1]["seed"], int(SET["probe_key"], 16)
        counters = []
        for privileged in (False, True):
            lock = verify.ProbeLock(None, rate=0.06, pass_fraction=0.35, tolerance=1e-4)
            lock.rng = verify.episode_stream(key, seed)
            _, _, steps, _ = verify.run_episode(seed, FakeSolver(), lock, privileged, RecordingCamera(), key)
            counters.append((lock.rng._counter, lock.total, steps))
        self.assertEqual(counters[0], counters[1])
        self.assertEqual(counters[0][0], counters[0][2] + 4 * counters[0][1])  # 1 per step + 3 + 1 per probe

    def test_a_real_frame_is_rendered_once_per_stamp(self):
        cam = RecordingCamera()
        env = verify.FaultAdaptiveDocking()
        sensors = verify.Sensors(cam, 3, self.SEEDS[2])
        obs16 = env.reset(self.SEEDS[2])
        for t in range(60):
            sensors.observe(env, obs16)
            obs16 = env.step(scripted(t, self.SEEDS[2]))[0]
        stamps = [c[0] for c in cam.calls]
        self.assertEqual(len(stamps), len(set(stamps)))


def _renderer_here():
    import importlib.util
    bundle = (os.path.isfile(os.path.join(os.environ.get("FAULT_DOCKING_RENDER", camera.DEFAULT_BUNDLE), "bundle.json"))
              or (TASK / "environment/assets/payload/render/bundle.json").is_file())
    mods = all(importlib.util.find_spec(m) is not None for m in ("numpy", "mujoco", "PIL"))
    return bundle and mods and any(os.path.isfile(p) for p in camera.LAVAPIPE_ICDS)


@unittest.skipUnless(_renderer_here(), "the renderer or the camera bundle is not installed here (both are in the task "
                                       "image)")
class RendererTests(unittest.TestCase):
    """Only where the renderer and the bundle are installed (a maintainer's machine, the task image)."""

    def test_the_same_state_renders_the_same_frame(self):
        bundle = os.environ.get("FAULT_DOCKING_RENDER") or (
            camera.DEFAULT_BUNDLE if os.path.isfile(os.path.join(camera.DEFAULT_BUNDLE, "bundle.json"))
            else str(TASK / "environment/assets/payload/render"))
        code = ("import sys, hashlib, json; sys.dont_write_bytecode = True; sys.path.insert(0, {tests!r}); "
                "import camera as C; c = C.Camera(bundle={bundle!r}); st = json.load(open({canary!r}))['state']; "
                "a = c.frame(1, 2, 3, *st); b = c.frame(1, 2, 3, *st); "
                "print('RESULT ' + json.dumps([a.tobytes() == b.tobytes(), int(a.max()), "
                "hashlib.sha256(a.tobytes()).hexdigest()]), flush=True);"
                " c.close()").format(tests=str(TASK / "tests"), bundle=bundle,
                                      canary=str(TASK / "tests/camera_canary.json"))
        out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=300)
        lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        self.assertTrue(lines, out.stderr[-2000:])
        same, top, sha = json.loads(lines[-1][len("RESULT "):])
        self.assertTrue(same)
        self.assertEqual(top, 255)       # (the canary state shows the LEDs)


class CopyTests(unittest.TestCase):
    def test_the_reference_s_simulator_copy_is_the_grader_s(self):
        self.assertEqual((TASK / "tests/env.py").read_bytes(),
                         (TASK / "solution/privileged/spacecraft_docking_env.py").read_bytes())

    def test_the_launcher_is_the_mechanism_box_one(self):
        self.assertEqual((TASK / "tests/nonet_exec.py").read_bytes(), MB_NONET.read_bytes())

    def test_the_planning_quota_covers_the_family(self):
        worst = len(oracle.FAMILY) * (oracle.STEPS - 118) + oracle.PLAN_OFFSET  # (t_p >= 108 + 2 + PLAN_OFFSET)
        self.assertGreaterEqual(oracle.PLAN_OFFSET * oracle.QUOTA, worst)

    def _block(self, name):
        text = (GIVEN / name).read_text()
        blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
        return json.loads(blocks[-1])

    def test_the_sensor_documents_carry_camera_py_s_numbers(self):
        self.assertEqual(self._block("camera.md"), json.loads(json.dumps(camera.CAMERA_SPEC)))
        self.assertEqual(self._block("target.md"), json.loads(json.dumps(camera.TARGET_SPEC)))
        text = (GIVEN / "camera.md").read_text()
        for s in ("(-0.082, 0, -0.115)", "fx = fy = 320", "(319.5, 239.5)", "90.0 x 73.74", "sqrt(0.8**2 + DN / 39)",
                  "2.0e-4 rad", "0.006 m/s", "0.004 rad/s", "7 to 18 steps", "29 to 41 observations"):
            self.assertIn(s, text)
        self.assertAlmostEqual(2 * math.degrees(math.atan(camera.WIDTH / 2 / camera.FX)), 90.0, places=6)
        self.assertAlmostEqual(2 * math.degrees(math.atan(camera.HEIGHT / 2 / camera.FY)), 73.74, places=2)
        target = (GIVEN / "target.md").read_text()
        for name, x, y, z in camera.DT_LEDS + camera.BT_LEDS:
            self.assertIn("{:g}".format(abs(y)), target)
        self.assertIn("r_port - 0.1299", target)
        self.assertAlmostEqual(camera.BT_APEX_OFF, 0.1299, places=4)

    def test_the_canary_record(self):
        rec = json.loads((TASK / "tests/camera_canary.json").read_text())
        self.assertEqual(len(rec["state"]), 5)
        self.assertRegex(rec["rgb_sha256"], r"^[0-9a-f]{64}$")


class AgentVisibleTests(unittest.TestCase):
    """The agent is docked zero-shot (the owner, 2026-11-05 and 2026-11-07): it gets a datasheet and the sensors'
    documents, not the grader's simulator, not the grader's renderer and not a grader of its own on visible episodes."""

    PAYLOAD = TASK / "environment/setup/payload"

    def visible(self):
        return [TASK / "instruction.md"] + sorted(p for p in (TASK / "environment").rglob("*") if p.is_file())

    def test_the_payload_is_the_documents_and_the_instruction_points_to_them(self):
        files = sorted(str(p.relative_to(self.PAYLOAD)) for p in self.PAYLOAD.rglob("*") if p.is_file())
        self.assertEqual(files, ["given/camera.md", "given/frames/README.md", "given/model.md", "given/target.md"])
        text = (TASK / "instruction.md").read_text()
        for p in ("/home/user/given/model.md", "/home/user/given/camera.md", "/home/user/given/target.md",
                  "/home/user/given/frames/"):
            self.assertIn(p, text)

    def test_no_agent_visible_file_carries_the_simulator_the_renderer_or_a_practice_grader(self):
        code = sorted({line.strip() for line in (TASK / "tests/env.py").read_text().splitlines()
                       if "self." in line and len(line.strip()) >= 20})
        self.assertGreater(len(code), 150)
        names = ("FaultAdaptiveDocking", "spacecraft_docking_env", "practice", "grade.py", "env.py", "_RNG",
                 "blake2b", "frame_noise_rng", "PCG64", "SeedSequence")
        docs = [TASK / "instruction.md"] + sorted(GIVEN.rglob("*.md"))
        for p in self.visible():
            if p.suffix in (".png", ".ktx", ".npz"):
                continue
            text = p.read_text(errors="replace")
            where = p.relative_to(TASK)
            for name in names:
                self.assertNotIn(name.lower(), text.lower(), "{} names {}".format(where, name))
            for line in code:
                self.assertNotIn(line, text, "{} carries the simulator's line {!r}".format(where, line))
        for p in docs:
            text = p.read_text()
            for word in ("Filament", "lavapipe", "camera.py", "render bundle", "bundle.json", "/opt/fault-docking"):
                self.assertNotIn(word, text, "{} names the grader's renderer ({})".format(p.relative_to(TASK), word))


# ---------------------------------------------------------------------------------------- the datasheet
def _num(token):
    """A number matched whole (not inside a longer number)."""
    return r"(?<![0-9.])" + re.escape(token) + r"(?![0-9])"


def _word(token):
    return r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])"


# Withheld: the grader's formulas, its update order, its noise and dropout laws and the distribution families
# (round 1's page stated them all). Numbers are matched whole; the geometry of target.md is exempt from the numbers.
WITHHELD_FORMULAS = [re.escape(t) for t in ("mod 13", "U(", "U{", "** 4", "**4", "z_i <-", "max(K1", "(d1 * s1)",
                                            "30 + (", "0.24 + 0.028", "0.145 + 0.02", "0.235 + 0.025", "0.65*a",
                                            "0.65 * al", "0.35 + 0.65", "0.12 * (F", "0.12 * field")]
WITHHELD_FORMULAS += [_word(t) for t in ("hypot", "cell_n", "memory_i", "threshold_i", "sw_n", "al_n", "C_n",
                                         "loss_n", "realized_duty_i")]
WITHHELD_NUMBERS = [_num(t) for t in ("0.035", "0.998", "0.996", "0.012", "0.008", "0.145", "0.155", "0.235 +",
                                      "0.245", "0.028", "0.92", "0.98", "1.04", "1.10", "1.16", "0.35", "0.0069",
                                      "0.0046", "0.0040", "0.0035")]


class DatasheetTests(unittest.TestCase):
    """given/model.md (and the instruction, camera.md, target.md, frames/README.md) state the plant at datasheet
    level: every range or nominal value with its tolerance contains what env.py draws and uses (20,000 dev worlds),
    and nothing states the grader's formulas, update order, noise processes or distribution families."""

    DOCS = ("model.md", "camera.md", "target.md", "frames/README.md")
    WORLDS = 20000

    @classmethod
    def setUpClass(cls):
        cls.page = (GIVEN / "model.md").read_text()
        cls.instruction = (TASK / "instruction.md").read_text()
        rnd, seeds = random.Random(20261107), []
        while len(seeds) < cls.WORLDS:
            s = rnd.getrandbits(64)
            if s >= 1 << 32 and s not in HIDDEN_SEEDS:
                seeds.append(s)
        cls.worlds = []
        for s in seeds:
            env = verify.FaultAdaptiveDocking()
            obs = env.reset(s)
            cls.worlds.append((env, obs))

    def test_nothing_states_the_grader_s_formulas_or_noise_laws(self):
        texts = {"instruction.md": self.instruction}
        texts.update({d: (GIVEN / d).read_text() for d in self.DOCS})
        for name, text in texts.items():
            for pat in WITHHELD_FORMULAS:
                self.assertIsNone(re.search(pat, text), "{} states {}".format(name, pat))
            if name == "target.md":
                continue
            for pat in WITHHELD_NUMBERS:
                self.assertIsNone(re.search(pat, text), "{} states {}".format(name, pat))
        self.assertIn("It is not an executable model", " ".join(self.page.split()))
        self.assertIn("It is not an executable model", " ".join(self.instruction.split()))

    def check(self, text, ok, where=None):
        self.assertIn(" ".join(text.split()), " ".join((where or self.page).split()))
        bad = [i for i, (env, obs) in enumerate(self.worlds) if not ok(env, obs)]
        self.assertEqual(bad[:3], [], "a dev world outside the page's statement {!r}".format(text))

    def test_the_thrusters(self):
        N = sys.modules["env"]._NOMINAL                  # (the grader's env.py, imported through verify)
        table = re.findall(r"^\| (\d) \| (-?[\d.]+) \| (-?[\d.]+) \| (-?[\d.]+) \|$", self.page, re.M)
        self.assertEqual([tuple(float(v) for v in r[1:]) for r in table], [tuple(c) for c in N])

        def types(env):
            """The nominal geometry of each physical nozzle: the force direction (within 0.18 rad) and the torque's
            sign tell the six apart."""
            out = []
            for cx, cy, ct in env.columns:
                out.append(min(range(6), key=lambda g: abs(wrap(math.atan2(cy, cx) - math.atan2(N[g][1], N[g][0])))
                               + (0.0 if (ct > 0) == (N[g][2] > 0) else 10.0)))
            return out

        def errors(env):
            ok = True
            for (cx, cy, ct), g in zip(env.columns, types(env)):
                gx, gy, gt = N[g]
                eff = math.hypot(cx, cy) / math.hypot(gx, gy)
                ok &= 0.70 <= eff <= 1.28
                ok &= abs(wrap(math.atan2(cy, cx) - math.atan2(gy, gx))) <= 0.18 + 1e-12
                ok &= 0.82 <= ct / (eff * gt) <= 1.18
            return ok
        self.check("effectiveness 0.70-1.28 of nominal", lambda e, o: errors(e))
        self.check("pointing error up to 0.18 rad either way", lambda e, o: errors(e))
        self.check("yaw moment arm 0.82-1.18 of nominal", lambda e, o: errors(e))
        self.check("in any of the 720 assignments", lambda e, o: sorted(types(e)) == list(range(6)))
        self.check("up to 0.010 m/s² on each linear axis and up to\n  0.007 rad/s² in yaw",
                   lambda e, o: abs(e.bias[0]) <= 0.010 and abs(e.bias[1]) <= 0.010 and abs(e.bias[2]) <= 0.007)
        self.check("A nozzle at full duty for one step uses 0.00034 of a full tank",
                   lambda e, o: sys.modules["env"].FUEL_BURN == 0.00034 and e.fuel == 1.0)

    def test_the_damping(self):
        env = verify.FaultAdaptiveDocking()
        env.reset(5 << 40)
        env.bias = [0.0, 0.0, 0.0]
        env.x, env.y, env.vx, env.vy, env.omega = 4.0, 0.0, 1.0, 0.0, 1.0
        env.step([0.0] * 6)
        per_s_v = 1.0 - env.vx ** (1.0 / 0.12)
        per_s_w = 1.0 - env.omega ** (1.0 / 0.12)
        self.assertIn("takes 1.5-2.0 % of the translational velocity and 3.0-3.6 % of the yaw rate per second",
                      " ".join(self.page.split()))
        self.assertTrue(0.015 <= per_s_v <= 0.020, per_s_v)
        self.assertTrue(0.030 <= per_s_w <= 0.036, per_s_w)

    def test_the_bus(self):
        self.check("severe one keeps 35-50 % and the moderate one 55-70 %",
                   lambda e, o: 0.35 <= e.failure_scale <= 0.50 and 0.55 <= e.secondary_failure_scale <= 0.70
                   and e.failure_nozzle != e.secondary_failure_nozzle)
        self.check("keeps 45-60 % and 65-80 %",
                   lambda e, o: 0.45 <= e.rekey_failure_scale <= 0.60 and 0.65 <= e.rekey_secondary_failure_scale <= 0.80
                   and e.rekey_failure_nozzle != e.rekey_secondary_failure_nozzle)
        self.check("a starting offset of 1 to 5 places",
                   lambda e, o: 1 <= e.routing_initial_shift <= 5 and 1 <= e.rekey_initial_shift <= 5)
        self.check("Its ordering differs from the first\n  event's", lambda e, o: e.rekey_order != e.routing_order)
        self.check("one exponent\n  per physical nozzle, 0.65-1.60",
                   lambda e, o: all(0.65 <= v <= 1.60 for v in e.failure_exponents + e.rekey_exponents))

        def lags(e, lo, hi, mem):
            tau = [-0.12 / math.log(m) for m in mem]
            return all(lo <= t <= hi for t in tau) and all(a < b for a, b in zip(tau, tau[1:]))
        self.check("time constants of 60-90 ms after the first event and 80-125 ms after the re-key, rising with the slot's index",
                   lambda e, o: lags(e, 0.060, 0.090, e.failure_valve_memory)
                   and lags(e, 0.080, 0.125, e.rekey_valve_memory))
        self.check("**The first event** fires in a step from 58 to 103", lambda e, o: 58 <= e.failure_step <= 103)
        self.check("armed from a step between 108 and 128, and at least 25 steps after the first event",
                   lambda e, o: 108 <= e.rekey_earliest <= 128 and e.rekey_earliest >= e.failure_step + 25)
        self.check("within a hidden 1.18-1.30 m of the port", lambda e, o: 1.18 <= e.rekey_proximity <= 1.30)
        self.check("between steps 148 and 165", lambda e, o: 148 <= e.rekey_deadline <= 165)
        self.check("a new speed of 0.020-0.045 rad/s, of the opposite sign",
                   lambda e, o: 0.020 <= abs(e.rekey_phase_rate) <= 0.045 and e.rekey_phase_rate * e.phase_rate < 0)

        def remount(e):
            (a, b), (c, d) = e.rekey_imu_linear
            ang = math.atan2(c, a)
            det = a * d - b * c
            return any(abs(ang - s) < 1e-12 for s in (-0.79, -0.43, 0.43, 0.79)) and abs(abs(det) - 1) < 1e-12
        self.check("turned a further 0.43 or 0.79 rad either way, possibly mirrored", lambda e, o: remount(e))
        self.check("deliver nothing for the next 29 to 41 observations",
                   lambda e, o: 29 <= e.rekey_blackout_duration - 1 <= 41)
        flat = " ".join(self.page.split())
        self.assertIn("by two when the commanded duties of that command sum to more than 1.35", flat)
        self.assertIn("below 0.12 arms it", flat)
        self.assertIn("above 0.78", flat)

    def test_the_target_the_boom_and_the_start(self):
        self.check("0.060-0.125 rad/s in\n  either direction", lambda e, o: 0.060 <= abs(e.phase_rate) <= 0.125)
        self.check("amplitude `A` of 0.10-0.16 m", lambda e, o: 0.10 <= e.port_amplitude <= 0.16)
        self.check("advancing at 0.18-0.28 rad/s from a phase anywhere in [-pi, pi)",
                   lambda e, o: 0.18 <= e.port_omega <= 0.28 and -math.pi <= e.port_extension_phase < math.pi)
        self.check("The target's phase starts anywhere in [-pi, pi).", lambda e, o: -math.pi <= e.phase < math.pi)
        self.check("2.85-3.65 m from the target's centre, at a bearing within 1.45 rad either way of the port's",
                   lambda e, o: 2.85 <= math.hypot(e.x, e.y) <= 3.65
                   and abs(wrap(math.atan2(e.y, e.x) - e.phase)) <= 1.45 + 1e-12)
        self.check("Its yaw lies within 1.1 rad either way of the port's bearing",
                   lambda e, o: abs(wrap(e.theta - e.phase)) <= 1.1 + 1e-12)
        self.check("its velocity is up to 0.10 m/s on each inertial axis\n  and its yaw rate up to 0.11 rad/s",
                   lambda e, o: abs(e.vx) <= 0.10 and abs(e.vy) <= 0.10 and abs(e.omega) <= 0.11)

    def test_the_plume(self):
        self.check("starts at a surface gap of 0.65 m", lambda e, o: e.plume_onset_gap == 0.65)
        self.check("0.9-1.2 times\n  the lost part", lambda e, o: all(0.9 <= abs(s) <= 1.2 for s in e.plume_swirl))
        self.check("nozzles 0, 2 and 4 one way, 1, 3 and 5 the other",
                   lambda e, o: all(e.plume_swirl[n] * e.plume_swirl[0] * (1 if n % 2 == 0 else -1) > 0
                                    for n in range(6)))
        self.check("0.24-0.38 s of full-duty firing",
                   lambda e, o: all(0.24 <= v <= 0.38 + 1e-12 for v in e.plume_switch_energy)
                   and all(a < b for a, b in zip(e.plume_switch_energy, e.plume_switch_energy[1:])))
        env = verify.FaultAdaptiveDocking()
        env.reset(7 << 40)
        env.x, env.y, env.theta = env.port_radius(), 0.0, 0.0      # at the port, the target's centre along -x body
        away = env._plume_column((0.0, 0.0, 1.0), 0)               # (no thrust direction: alignment 0)
        aimed = env._plume_column((1.0, 0.0, 1.0), 0)               # thrust along +x: exhaust along -x, at the target
        flat = " ".join(self.page.split())
        self.assertIn("at the port, 29 % with its exhaust pointing away from the target", flat)
        self.assertIn("up to 82 % aimed straight at it", flat)
        self.assertAlmostEqual(1.0 - away[0][2], 0.287, places=3)
        self.assertAlmostEqual(1.0 - aimed[0][2], 0.82, places=12)

    def test_the_sensors(self):
        cam_doc = (GIVEN / "camera.md").read_text()
        self.check("a fixed latency of 1, 2 or 3 steps", lambda e, o: 1 <= e.pose_delay <= 3, cam_doc)
        self.check("two outages of 7 to 18 steps (0.8 to 2.2 s) each,\nat varying times in the first 21 s",
                   lambda e, o: all(7 <= b - a <= 18 and b * 0.12 <= 21.0 + 1e-9 for a, b in e.blackouts), cam_doc)
        self.check("at most 0.006 m/s² on each linear axis",
                   lambda e, o: abs(o[8]) <= 0.006 and abs(o[9]) <= 0.006, cam_doc)        # (t = 0: noise alone)
        self.check("at most 0.004 rad/s² on\nthe yaw channel", lambda e, o: abs(o[10]) <= 0.004, cam_doc)
        self.check("either `r_port` itself or `2 * 1.14 - r_port`",
                   lambda e, o: e.port_encoder_sign in (-1.0, 1.0) and e.rekey_port_encoder_sign in (-1.0, 1.0), cam_doc)

    def test_the_docking_test_and_the_quality_are_exact(self):
        flat = " ".join(self.page.split())
        for s in ("0.085 m, 0.060 m/s, 0.020 rad and 0.015 rad/s", "the hold reaches 30",
                  "`0.94 + 0.04 * fuel + 0.02 * (1 - k / 400)`", "`|p| < 0.82`", "`|p| > 6.2`",
                  "`0.03 * clamp((initial error - best error) / initial error, 0, 1)`"):
            self.assertIn(s, flat)

    def test_the_page_s_example_request(self):
        line = next(l for l in self.page.splitlines() if l.startswith('{"type": "act"'))
        page_req = json.loads(line)
        req = make_sample_frames.act_request(verify.BlankCamera())
        mine = json.loads(make_sample_frames.cut(req))
        self.assertEqual({k: v for k, v in page_req["obs"].items() if k != "frame"},
                         {k: v for k, v in mine["obs"].items() if k != "frame"})
        self.assertEqual(page_req["obs"]["frame"]["stamp"], mine["obs"]["frame"]["stamp"])
        self.assertEqual((page_req["type"], page_req["t"]), ("act", make_sample_frames.ACT_T))
        self.assertEqual(tuple(page_req["obs"]), verify.OBS_KEYS)


class _LocalGuarded(verify.GuardedSolver):
    """GuardedSolver's wire and budget with its program started as the current user (no seccomp, no account change)."""

    def _spawn(self):
        proc = subprocess.Popen(["bash", self.entry], cwd=self.submission, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1,
                                start_new_session=True)
        proc.stdin = verify.BoundedStdin(proc.stdin, self.step_timeout)
        return proc


class WireBudgetTests(unittest.TestCase):
    def program(self, tmp, body):
        sub = os.path.join(tmp, "submission")
        os.makedirs(sub)
        with open(os.path.join(sub, "ctl.py"), "w") as f:
            f.write(body)
        with open(os.path.join(sub, "run.sh"), "w") as f:
            f.write("exec {} -u ctl.py\n".format(sys.executable))
        return sub

    def test_the_reply_budget_aborts_the_rest_of_the_run(self):
        body = ("import sys, json, time\nprint('READY', flush=True)\nfor line in sys.stdin:\n"
                "    m = json.loads(line)\n    time.sleep(0.15 if m['type'] == 'act' else 0)\n"
                "    print(json.dumps({'action': [0]*6} if m['type'] == 'act' else {'ok': True}), flush=True)\n")
        with tempfile.TemporaryDirectory() as tmp:
            s = _LocalGuarded(step_timeout=10, submission=self.program(tmp, body), reply_budget=0.5)
            try:
                s.reset()
                obs = {"frame": {"stamp": 0, "data": "A" * 409600}}
                done = 0
                with self.assertRaises(verify.SolverAborted):
                    for t in range(20):
                        s.act(t, obs)
                        done += 1
                self.assertIn(done, (2, 3))
                self.assertTrue(s.budget_spent)
                self.assertGreater(s.reply_s, 0.5)
                with self.assertRaises(verify.SolverAborted):
                    s.reset()                               # every later episode aborts at once
            finally:
                s.close()

    def test_a_program_that_does_not_read_cannot_stall_the_grader(self):
        body = ("import sys, json, time\nprint('READY', flush=True)\n"
                "while True:\n    print(json.dumps({'action': [0]*6}), flush=True)\n    time.sleep(0.01)\n")
        with tempfile.TemporaryDirectory() as tmp:
            s = _LocalGuarded(step_timeout=1.5, submission=self.program(tmp, body))
            try:
                t0 = time.monotonic()
                with self.assertRaises(verify.SolverAborted):
                    for t in range(50):                    # 410 KB a line, never read: the 1 MB pipe fills
                        s.act(t, {"frame": {"stamp": 0, "data": "A" * 409600}})
                self.assertLess(time.monotonic() - t0, 30)
            finally:
                s.close()


class NetworkGuardTests(unittest.TestCase):
    """The set is public: where the seccomp filter cannot be installed, nothing is graded with the network open."""

    def run_main(self, mode, solver, camera_error=None):
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "submission")
            os.makedirs(sub)
            with open(os.path.join(sub, "run.sh"), "w") as f:
                f.write("echo READY\n")
            stub = mock.Mock()
            started_cam = (mock.patch.object(verify.camera, "Camera", side_effect=camera_error) if camera_error
                           else mock.patch.object(verify, "start_camera", return_value=(stub, {"camera_canary_match": 1.0})))
            with mock.patch.object(verify, "egress_mode", return_value=mode), \
                    mock.patch.object(verify, "GuardedSolver", side_effect=solver) as started, \
                    mock.patch.object(verify, "write_verdict") as written, \
                    mock.patch.object(verify, "own_submission"), \
                    mock.patch.object(verify, "SUBMISSION", sub), \
                    started_cam, \
                    mock.patch.dict(os.environ, {"ALE_STAGE_DIR": str(TASK / "tests")}), \
                    mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                code = None
                try:
                    verify.main()
                except SystemExit as exc:
                    code = exc.code
        return code, started, written, err.getvalue()

    def test_without_the_filter_the_verify_is_a_task_error_and_no_controller_starts(self):
        code, started, written, err = self.run_main("unavailable", AssertionError("a controller started"))
        self.assertEqual(code, 1)
        started.assert_not_called()
        written.assert_not_called()
        self.assertIn("TASK ERROR: the grading environment cannot block the controller's network", err)

    def test_with_the_filter_the_verdict_records_it(self):
        code, started, written, _ = self.run_main("seccomp", verify.SolverMissing("no submission"))
        self.assertIsNone(code)
        started.assert_called_once()
        metrics = written.call_args[0][0]["metrics"]
        self.assertEqual(metrics["egress_guard_seccomp"], 1.0)
        self.assertEqual(metrics["camera_canary_match"], 1.0)
        self.assertNotIn("egress_detected", metrics)

    def test_without_the_camera_the_verify_is_a_task_error_and_no_controller_starts(self):
        code, started, written, err = self.run_main("seccomp", AssertionError("a controller started"),
                                                    camera_error=verify.camera.CameraError("no lavapipe here"))
        self.assertEqual(code, 1)
        started.assert_not_called()
        written.assert_not_called()
        self.assertIn("TASK ERROR: no lavapipe here", err)


class DeterminismCanaryTests(unittest.TestCase):
    def test_set_episode_0_reproduces_its_recorded_quality(self):
        kept = [c for c in sorted(SELECTION["candidates"], key=lambda c: c["draw"]) if c["status"] == "kept"]
        first = kept[0]
        self.assertEqual(first["seed"], SET["episodes"][0]["seed"])
        got = oracle_batch.run_one(first["seed"], SET["probe_key"])
        self.assertTrue(got["success"])
        self.assertEqual((got["steps"], got["probes"], got["probes_passed"]),
                         (first["steps"], first["probes"], first["probes_passed"]))
        if sys.version_info >= (3, 12):
            self.assertEqual(got["quality"], first["quality"])
        else:  # (before 3.12 the builtin sum is not compensated: fuel can differ in its last bit)
            self.assertAlmostEqual(got["quality"], first["quality"], delta=1e-12)


if __name__ == "__main__":
    unittest.main()
