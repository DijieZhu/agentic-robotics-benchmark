"""Hidden grader for fault-adaptive spacecraft docking.

Runs as root in the separate verifier sandbox (task.toml), from this folder. It owns the simulator (env.py; the agent
gets no copy of it, only the plant's datasheet, environment/setup/payload/given/model.md, and its sensors' documents),
drives the agent's controller (`bash run.sh` in /home/user/submission) over the kit's wire, and scores every episode
from its own simulator state; nothing the controller prints is a scoring input.

The hidden episodes are a fixed set (hidden/episodes.json): the same 48 seeds of 64 bits, in the same order, for every
submission. A seed fixes every hidden parameter of its episode (env.py draws them all in `reset`), and
`load_episodes` recomputes each world's digest, so a change to env.py's draws is a task error instead of a
silently different set. Each episode's match lock draws from its own stream, keyed by the set's hidden key and the
episode's seed (the kit's `seed_rng` with explicit entropy), so a submission meets the same probes every time, whatever
ran before it; nothing in a verify draws from OS entropy. A probed step is asked twice with the same `t` (the real
observation and a copy whose camera frame shows the craft displaced, in a hidden order); only the real reply reaches
the simulator. An aborted episode (a missed deadline, a crash, a protocol error, a malformed action, the reply budget
spent) scores 0 and stays in the mean.

The observation (round 2, 2026-11-07): the controller sees what the chaser's sensors deliver, never the simulator's
pose. env.py is unchanged and still makes its 16-number observation (its draws and its pose history are what they were),
and `Sensors` turns it into the wire's object: the IMU, the fuel gauge, the mission clock and the boom's encoder
telemetry as before; the star tracker's yaw (camera.star_tracker; none during the re-key's restart); and the docking
camera's frame (camera.py: the record env.py's delayed pose came from, rendered, with the stamp it was captured at),
missing exactly when env.py's pose was not fresh. The eight pose numbers, the pose's age and its validity flag are
gone. The frame travels inline (base64 in the request line, about 410 KB); `GuardedSolver` writes it with the
request's deadline and charges the controller's reply time against one budget for the run (`reply_budget_s`).

The controller has no network while it is graded: `GuardedSolver` starts it through nonet_exec.py (no_new_privs, a
seccomp filter under which socket() and socketpair() work for AF_UNIX only, then the agent's account and a fixed
environment: PATH /opt/venv/bin:/usr/local/bin:/usr/bin:/bin, HOME /home/user, LANG C.UTF-8). A self-test before
episode 0 checks that the filter holds here. Where it cannot be installed, the verify ends as a task error before any
controller starts: the set is public, so nothing is graded with the network open. Where the camera cannot be rendered
(the render canary, before episode 0), the verify ends as a task error too.

The reference program's channel: a submission whose `.oracle_token` (a regular file, never a link) hashes to
hidden/oracle_token.sha256 gets, in every observation, the simulator's current truth (`privileged_block`: every quantity
`env.step` reads that is in effect now, named as env.py names it; never a value drawn for the future, never the noise
stream), as one more key, "privileged", after the sensors' keys; every other submission gets the sensors' object
alone. A probed step's copy shifts the block's true x, y and yaw by the same draws as the copy's frame and star
tracker, so the probe asks the same question of every program. The channel exists to show that the episodes can be
done; nothing else about the run changes, and the metrics record `privileged_mode` and `oracle_token_mismatch`.
"""

import base64
import fcntl
import hashlib
import hmac
import json
import math
import os
import pwd
import re
import select
import stat
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import camera  # noqa: E402  (stdlib at import; the renderer is built in main)
from env import FaultAdaptiveDocking  # noqa: E402
from robotics_grader import (  # noqa: E402
    ProbeLock,
    Solver,
    SolverAborted,
    SolverMissing,
    agent_user,
    fail,
    ratio_rewards,
    read_anchor,
    seed_rng,
    stage_config,
    write_verdict,
    zero_verdict,
)
from robotics_grader.stage import secure_exec_path  # noqa: E402

SUBMISSION = os.path.join(os.environ.get("ALE_HOME", "/home/user"), "submission")
HIDDEN = os.path.join(HERE, "hidden")  # (root's alone, like all of this folder at verify time)
SET_PATH = os.path.join(HIDDEN, "episodes.json")
SET_FORMAT = "fault_adaptive_spacecraft_docking fixed set 1"
TOKEN_FILE = ".oracle_token"
TOKEN_DIGEST = os.path.join(HIDDEN, "oracle_token.sha256")
TOKEN_MAX_BYTES = 4096
NONET = os.path.join(HERE, "nonet_exec.py")  # (run as root from this root-only folder)

# The privileged block: what env.step reads that is in effect now (an AST test checks the lists against env.py).
PRIV_FORMAT = "fault_adaptive_spacecraft_docking privileged 1"
PRIV_STATE = ("steps", "x", "y", "vx", "vy", "theta", "omega", "phase", "phase_rate", "port_extension_phase",
              "fuel", "routing_active", "rekey_active", "routing_relay_armed", "nozzle_exponents", "valve_memory",
              "slot_valves", "plume_swirl", "plume_energy", "first_relay_reversals", "rekey_relay_reversals",
              "hold_steps", "best_error")
PRIV_ROUTING = ("routing_order", "routing_phase", "routing_direction", "failure_nozzle",
                "secondary_failure_nozzle", "failure_scale", "secondary_failure_scale")  # null until routing_active
PRIV_CONSTANTS = ("columns", "bias", "port_amplitude", "port_omega", "plume_onset_gap", "plume_strength",
                  "plume_switch_energy", "initial_error")
# Never sent. The pre-drawn future reaches the block through PRIV_STATE / PRIV_ROUTING when it takes effect.
NEVER_SENT_FUTURE = ("failure_step", "failure_exponents", "failure_valve_memory", "routing_initial_shift",
                     "rekey_earliest", "rekey_deadline", "rekey_proximity", "rekey_order", "rekey_initial_shift",
                     "rekey_direction", "rekey_failure_nozzle", "rekey_secondary_failure_nozzle",
                     "rekey_failure_scale", "rekey_secondary_failure_scale", "rekey_exponents",
                     "rekey_valve_memory", "rekey_phase_rate")
NEVER_SENT_OBSERVATION = ("rng", "pose_delay", "blackouts", "rekey_blackout_duration", "rekey_step", "pose_history",
                          "last_visible_pose", "last_accel_body", "imu_angle", "imu_mirror", "imu_yaw_sign",
                          "rekey_imu_linear", "rekey_imu_yaw_sign", "port_encoder_sign", "rekey_port_encoder_sign")
NEVER_SENT_OUTCOME = ("success", "failure_reason")  # (always False and "" while a request is open)


def own_submission(path):
    """Hand the restored artifact back to the agent account.

    In the separate verifier the engine restores /home/user/submission as root (every
    artifact write is the framework's), so its files land root-owned with their recorded
    modes. The kit Solver runs `bash run.sh` AS THE AGENT: the agent must be able to read
    its own controller (a file it had left non-world-readable would otherwise be
    unreadable) and write scratch beside it, exactly as it could while it worked. Walk the
    tree without following links — a freshly restored tree has none, but lchown is the safe
    habit — and give every entry, then the folder itself, to the agent. A no-op when the
    path is missing or is a link; the Solver then scores a real zero through SolverMissing.
    Runs as root, which the verify stage is.
    """
    if not os.path.isdir(path) or os.path.islink(path):
        return
    try:
        account = pwd.getpwnam(agent_user())
    except KeyError:
        return
    uid, gid = account.pw_uid, account.pw_gid
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        for name in dirnames + filenames:
            try:
                os.lchown(os.path.join(dirpath, name), uid, gid)
            except OSError:
                pass
    os.lchown(path, uid, gid)


# ------------------------------------------------------------------------------------------------ the fixed set
def _canon(value):
    """A world value with every float at 13 significant digits: immune to a last-bit libm difference between hosts."""
    if isinstance(value, float):
        return format(value, ".12e")
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    return value


def world_digest(seed):
    """sha256 of the world `env.reset(seed)` builds (every attribute but the generator), floats canonicalised."""
    env = FaultAdaptiveDocking()
    env.reset(seed)
    world = {k: _canon(v) for k, v in vars(env).items() if k != "rng"}
    return hashlib.sha256(json.dumps(world, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def _is_seed(value):
    return isinstance(value, int) and not isinstance(value, bool) and (1 << 32) <= value < (1 << 64)


def load_episodes(path=None):
    """The fixed hidden set: (seeds in grading order, probe key as an int). A missing or malformed set, or a world
    that no longer matches its digest, is a task error; no message names a value of the set."""
    path = path or SET_PATH
    try:
        with open(path) as f:
            spec = json.load(f)
    except (OSError, ValueError) as exc:
        fail("cannot read the hidden episode set: {}".format(type(exc).__name__))
    if not isinstance(spec, dict) or spec.get("format") != SET_FORMAT:
        fail("the hidden episode set is not of the format '{}'".format(SET_FORMAT))
    key = spec.get("probe_key")
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
        fail("the hidden episode set's probe_key is not 64 lowercase hex digits")
    episodes = spec.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        fail("the hidden episode set holds no episodes")
    seeds = []
    for k, item in enumerate(episodes):
        if not isinstance(item, dict) or not _is_seed(item.get("seed")) or not isinstance(item.get("world_sha256"), str):
            fail("hidden episode {}: not a seed in [2**32, 2**64) with a world digest".format(k))
        seeds.append(item["seed"])
    if len(set(seeds)) != len(seeds):
        fail("the hidden episode set repeats a seed")
    for k, item in enumerate(episodes):
        if world_digest(item["seed"]) != item["world_sha256"]:
            fail("hidden episode {}: its world no longer matches the recorded digest (env.py's draws changed)".format(k))
    return seeds, int(key, 16)


def episode_stream(probe_key, seed):
    """The match lock's stream for one episode: keyed by the set's key and the episode's seed alone."""
    return seed_rng(entropy=(probe_key << 64) | seed)


# ------------------------------------------------------------------------------------------------ the channel
def token_state(submission, digest_path=None):
    """(privileged, mismatch) for the submission's .oracle_token against hidden/oracle_token.sha256.

    Only a regular file counts (opened without following a link and without blocking, at most TOKEN_MAX_BYTES); a token
    that is there but does not match (a wrong value, a link, a directory, a FIFO, an oversized file) is recorded as a
    mismatch and otherwise ignored."""
    digest_path = digest_path or TOKEN_DIGEST
    path = os.path.join(submission, TOKEN_FILE)
    try:
        with open(digest_path) as f:
            want = f.read().split()[0].strip().lower()
    except (IOError, OSError, IndexError):
        return False, False
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return False, os.path.lexists(path)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > TOKEN_MAX_BYTES:
            return False, True
        data = os.read(fd, TOKEN_MAX_BYTES + 1)
    except OSError:
        return False, True
    finally:
        os.close(fd)
    ok = hmac.compare_digest(hashlib.sha256(data.strip()).hexdigest(), want)
    return ok, not ok


def _plain(v):
    return [_plain(x) for x in v] if isinstance(v, (list, tuple)) else v


def privileged_block(env):
    """The simulator's current truth: everything env.step reads that is in effect now. Draws nothing, writes nothing."""
    state = {k: _plain(getattr(env, k)) for k in PRIV_STATE + PRIV_CONSTANTS}
    state.update({k: (_plain(getattr(env, k)) if env.routing_active else None) for k in PRIV_ROUTING})
    return {"format": PRIV_FORMAT, "state": state}


# ------------------------------------------------------------------------------------------------ the sensors
OBS_KEYS = ("imu", "fuel", "episode_fraction", "boom_encoder", "star_tracker_yaw", "frame")  # (the wire's order)
CANARY_PATH = os.path.join(HERE, "camera_canary.json")


def restart_active(env):
    """True during the re-key's restart: the L - 1 observations env.py's camera pose is blacked out after the re-key
    (its `_is_blackout` event term). The star tracker is out for exactly these. Reads attributes only."""
    return bool(env.rekey_active and env.rekey_step <= env.steps < env.rekey_step + env.rekey_blackout_duration)


class BlankCamera(object):
    """A stand-in for camera.Camera that renders nothing: every frame is black (the review tools and the repository
    test, where nothing reads the image). Never used by main()."""

    BLACK = bytes(camera.FRAME_BYTES)

    def frame(self, key, seed, stamp, x, y, theta, phase, port_radius):
        return self.BLACK


class Sensors(object):
    """One episode's sensors on top of env.py's observation: no draw from env.py's generator, nothing written to env.

    The frame of a fresh observation is the pose record with the stamp env.py's delayed pose came from (the step
    count minus the pose's age), rendered: the craft at the record's x, y and yaw, the target at its phase, the boom
    at the record's port radius. Its noise field is keyed by (noise key, seed, stamp) alone. The star tracker reads the
    craft's yaw now with its own keyed noise, and nothing during the re-key's restart."""

    def __init__(self, cam, noise_key, seed):
        self.cam = cam
        self.key = int(noise_key)
        self.seed = int(seed)
        self._last = None           # (stamp, encoded data) of the last real frame (t <= d repeats the stamp-0 frame)
        self.renders = 0
        self.render_s = 0.0

    def _render(self, stamp, x, y, theta, phase, port_radius):
        t0 = time.monotonic()
        try:
            raw = self.cam.frame(self.key, self.seed, stamp, x, y, theta, phase, port_radius)
            data = base64.b64encode(raw if isinstance(raw, bytes) else raw.tobytes()).decode("ascii")
        except Exception as exc:  # noqa: BLE001 - the renderer is the grader's: a failure is a task error, not a score
            raise camera.CameraError("the camera failed to render: {}: {}".format(type(exc).__name__, exc)[:300])
        self.renders += 1
        self.render_s += time.monotonic() - t0
        return data

    @staticmethod
    def record(env, stamp):
        for item in env.pose_history:
            if item[0] == stamp:
                return item
        raise camera.CameraError("no pose record for stamp {}".format(stamp))

    def observe(self, env, obs16):
        """(the wire's observation, the frame's pose record or None) for env.py's observation `obs16` at env.steps."""
        rec = None
        frame = None
        if obs16[13] == 1.0:
            stamp = env.steps - int(obs16[12])
            rec = self.record(env, stamp)
            if self._last is None or self._last[0] != stamp:
                _, x, y, th, gx, gy, gyaw = rec
                self._last = (stamp, self._render(stamp, x, y, th, gyaw, math.hypot(gx, gy)))
            frame = {"stamp": stamp, "data": self._last[1]}
        yaw = None if restart_active(env) else camera.star_tracker(env.theta, self.key, self.seed, env.steps)
        obs = {"imu": [obs16[8], obs16[9], obs16[10]], "fuel": obs16[11], "episode_fraction": obs16[14],
               "boom_encoder": obs16[15], "star_tracker_yaw": yaw, "frame": frame}
        return obs, rec

    def shifted(self, rec):
        """The probe copy's frame: the same record with the craft displaced, the same noise field."""
        def frame(dx, dy, dyaw):
            stamp, x, y, th, gx, gy, gyaw = rec
            th2 = (th + dyaw + math.pi) % (2.0 * math.pi) - math.pi
            return {"stamp": stamp, "data": self._render(stamp, x + dx, y + dy, th2, gyaw, math.hypot(gx, gy))}
        return frame


def perturbed_observation(obs, rng, shifted_frame=None):
    """The probed step's copy (three draws, in this order, as in round 1): its frame shows the craft displaced by
    (dx, dy) and turned by dyaw at the frame's capture, its star-tracker yaw is turned by dyaw; every other sensor reads
    as in the real request; a field the real request lacks (no frame, no star tracker) stays absent. In a token run the
    block's true x, y and yaw move by the same draws, so a program that reads the block answers the copy on its own."""
    dx = rng.uniform(-0.28, 0.28)
    dy = rng.uniform(-0.28, 0.28)
    dyaw = rng.uniform(-0.18, 0.18)
    copy = dict(obs)
    if obs.get("frame") is not None:
        copy["frame"] = shifted_frame(dx, dy, dyaw)
    if obs.get("star_tracker_yaw") is not None:
        copy["star_tracker_yaw"] = (obs["star_tracker_yaw"] + dyaw + math.pi) % (2.0 * math.pi) - math.pi
    if "privileged" in obs:
        st = dict(obs["privileged"]["state"])
        st["x"] += dx
        st["y"] += dy
        st["theta"] = (st["theta"] + dyaw + math.pi) % (2.0 * math.pi) - math.pi
        copy["privileged"] = dict(obs["privileged"], state=st)
    return copy


def camera_canary(cam):
    """(match, seconds): one render of a fixed dev state against the hash taken in the task image (camera_canary.json).
    It never scores: a mismatch says the renderer draws other bytes here than in the image the samples came from."""
    with open(CANARY_PATH) as f:
        spec = json.load(f)
    t0 = time.monotonic()
    rgb = cam.render_rgb(*spec["state"])
    seconds = time.monotonic() - t0
    return float(hashlib.sha256(rgb.tobytes()).hexdigest() == spec["rgb_sha256"]), seconds


# ------------------------------------------------------------------------------------------------ the network
def agent_ids():
    pw = pwd.getpwnam(agent_user())
    return pw.pw_uid, pw.pw_gid


F_SETPIPE_SZ = 1031          # (Linux fcntl)
PIPE_BYTES = 1 << 20         # a request line is about 410 KB: the controller's stdin pipe holds two


class BoundedStdin(object):
    """The controller's stdin as the kit's Solver writes it. A request line (about 410 KB with its frame) is larger
    than a default pipe, and a plain write would wait for as long as the controller does not read, outside the kit's
    per-request deadline. Here the pipe holds 1 MB, the descriptor is non-blocking, and a write that the controller does
    not take within `timeout` seconds fails as a closed pipe, which the kit charges to the episode (SolverAborted).
    `started` is when the last write began (after the grader's own serialisation of the line)."""

    def __init__(self, f, timeout):
        self._f = f
        self._fd = f.fileno()
        try:
            fcntl.fcntl(self._fd, F_SETPIPE_SZ, PIPE_BYTES)
        except OSError:
            pass
        os.set_blocking(self._fd, False)
        self.timeout = float(timeout)
        self.started = None

    def write(self, text):
        self.started = time.monotonic()
        view = memoryview(text.encode("utf-8"))
        deadline = self.started + self.timeout
        while view:
            try:
                view = view[os.write(self._fd, view):]
            except BlockingIOError:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise OSError("the controller did not read its request within {:g} s".format(self.timeout))
                select.select([], [self._fd], [], left)
        return len(text)

    def flush(self):
        pass

    def close(self):
        try:
            self._f.close()
        except (IOError, OSError, ValueError):
            pass


class GuardedSolver(Solver):
    """The kit's Solver (the kit stays as it is), its program started through nonet_exec.py: no_new_privs, a seccomp
    filter under which socket() and socketpair() work for AF_UNIX only (io_uring and foreign syscall ABIs refused), then
    the agent's account and a fixed environment (PATH /opt/venv/bin:/usr/local/bin:/usr/bin:/bin, HOME /home/user,
    LANG C.UTF-8). There is no unguarded way to start it: main() stops with a task error before this class is used
    where the egress self-test fails.

    Its stdin is a BoundedStdin (the request's deadline covers the write of a frame), and the controller's reply time
    is summed over the run: from the moment a request's write begins to the moment its reply is read (`reply_s`).
    Once the sum passes `reply_budget` seconds, that request's episode and every later one abort (scored 0); the
    controller's start-up (READY) is bounded by the kit's handshake deadline instead and is not summed."""

    def __init__(self, *args, **kwargs):
        budget = kwargs.pop("reply_budget", None)
        self.reply_budget = None if budget is None else float(budget)
        self.reply_s = 0.0
        self.budget_spent = False
        super(GuardedSolver, self).__init__(*args, **kwargs)

    def _spawn(self):
        secure_exec_path()  # (as the kit's deprivileged does: nothing agent-writable left on root's PATH)
        try:
            uid, gid = agent_ids()
        except KeyError:
            raise SolverAborted("no agent account to run the controller as")
        cmd = [sys.executable, "-I", NONET, "--uid", str(uid), "--gid", str(gid), "--cwd", self.submission,
               "--", "bash", self.entry]
        try:
            proc = subprocess.Popen(cmd, cwd=self.submission, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1,
                                    start_new_session=True, env={"PATH": "/usr/bin:/bin"})
        except OSError as exc:
            raise SolverAborted("the controller could not be started: {}".format(exc))
        proc.stdin = BoundedStdin(proc.stdin, self.step_timeout)
        return proc

    def _spent(self):
        return SolverAborted("the controller's replies took more than the reply budget of {:g} s".format(
            self.reply_budget))

    def reset(self):
        if self.budget_spent:
            raise self._spent()
        super(GuardedSolver, self).reset()

    def request(self, message):
        if self.budget_spent:
            raise self._spent()
        stdin = self.proc.stdin if self.proc is not None else None
        try:
            return super(GuardedSolver, self).request(message)
        finally:
            began = getattr(stdin, "started", None)
            if began is not None:
                self.reply_s += time.monotonic() - began
                stdin.started = None
            if self.reply_budget is not None and self.reply_s > self.reply_budget and not self.budget_spent:
                self.budget_spent = True
                raise self._spent()


def egress_self_test(uid, gid):
    """True if a program started through the launcher cannot open an internet socket (AF_INET, AF_INET6, AF_VSOCK)
    and can open a Unix one."""
    probe = ("import socket,sys\n"
             "bad=0\n"
             "for fam in (socket.AF_INET, socket.AF_INET6, 40):\n"
             "    try:\n"
             "        socket.socket(fam, socket.SOCK_STREAM).close(); bad=1\n"
             "    except OSError:\n"
             "        pass\n"
             "try:\n"
             "    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close()\n"
             "except OSError:\n"
             "    bad=1\n"
             "sys.exit(3 if bad else 0)\n")
    try:
        r = subprocess.run([sys.executable, "-I", NONET, "--uid", str(uid), "--gid", str(gid), "--cwd", "/", "--",
                            sys.executable, "-I", "-c", probe],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, env={"PATH": "/usr/bin:/bin"})
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def egress_mode():
    """Decided once, before episode 0: "seccomp" (the launcher's filter holds) or "unavailable" (it cannot be installed
    here, or there is no agent account to test it with). main() grades only under "seccomp": with "unavailable" the
    verify ends as a task error, because the set is public and a controller with the network could fetch it."""
    secure_exec_path()
    try:
        uid, gid = agent_ids()
    except KeyError:
        return "unavailable"
    return "seccomp" if egress_self_test(uid, gid) else "unavailable"


# ------------------------------------------------------------------------------------------------ the episodes
def run_episode(seed, solver, lock, privileged, cam, noise_key):
    """One hidden episode: (quality, info, steps, sensors). `lock.rng` is the episode's own stream; `cam` renders the
    frames (camera.Camera in the grader), `noise_key` keys the sensors' noise (the set's hidden key in the grader)."""
    env = FaultAdaptiveDocking()
    sensors = Sensors(cam, noise_key, seed)
    quality, info = 0.0, {"success": False, "collision": False, "failure_reason": "not_started"}
    try:
        obs16 = env.reset(seed)
        solver.reset()
        for t in range(env.max_steps):
            wire, rec = sensors.observe(env, obs16)
            if privileged:
                wire["privileged"] = privileged_block(env)
            if lock.due():
                action = lock.probe(solver, t, wire,
                                    perturbed=perturbed_observation(wire, lock.rng, sensors.shifted(rec)))
            else:
                action = solver.act(t, wire)
            obs16, quality, done, info = env.step(action)
            if done:
                break
    except (SolverAborted, ValueError, TypeError, OverflowError) as exc:
        quality, info = 0.0, {"success": False, "collision": False, "failure_reason": "aborted: {}".format(exc)}
    return float(quality), info, env.steps, sensors


def run_episodes(config, seeds, probe_key, solver, privileged, cam):
    lock = ProbeLock(
        None,
        rate=config.get("probe_rate", 0.06),
        pass_fraction=config.get("probe_pass_fraction", 0.35),
        tolerance=1e-4,
    )
    qualities = []
    successes = collisions = aborted = 0
    renders, render_s = 0, 0.0
    for episode, seed in enumerate(seeds):
        # one probe stream per episode, keyed by the set's hidden key and the episode's seed: the same probes for the
        # same submission every time, whatever ran before
        lock.rng = episode_stream(probe_key, seed)
        quality, info, steps, sensors = run_episode(seed, solver, lock, privileged, cam, probe_key)
        qualities.append(quality)
        successes += int(bool(info.get("success")))
        collisions += int(bool(info.get("collision")))
        aborted += int(str(info.get("failure_reason", "")).startswith("aborted"))
        renders += sensors.renders
        render_s += sensors.render_s
        print(
            "[grader] episode {}: quality={:.4f} success={} steps={} reason={} frames={} reply_s={:.1f}".format(
                episode, quality, bool(info.get("success")), steps, info.get("failure_reason", ""), sensors.renders,
                solver.reply_s
            ),
            file=sys.stderr,
            flush=True,
        )
    stats = {"camera_frames_rendered": float(renders),
             "camera_frame_ms_mean": 1e3 * render_s / renders if renders else 0.0,
             "reply_s_total": solver.reply_s, "reply_budget_spent": float(solver.budget_spent)}
    return qualities, successes, collisions, aborted, lock, stats


def start_camera():
    """The grader's renderer and its canary, before episode 0; a renderer that cannot start is a task error."""
    t0 = time.monotonic()
    try:
        cam = camera.Camera()
        match, canary_s = camera_canary(cam)
    except camera.CameraError as exc:
        fail(str(exc))
    except (OSError, ValueError, KeyError) as exc:
        fail("the camera cannot be rendered here: {}: {}".format(type(exc).__name__, exc))
    print("[grader] camera: ready in {:.1f} s, canary {} ({:.2f} s)".format(
        time.monotonic() - t0, "matches the task image" if match else "DIFFERS from the task image", canary_s),
        file=sys.stderr, flush=True)
    return cam, {"camera_canary_match": match, "camera_canary_s": canary_s, "camera_start_s": time.monotonic() - t0}


def main():
    config = stage_config()
    seeds, probe_key = load_episodes()
    anchor, anchor_meta = read_anchor()
    # read before any of the submission's code runs
    privileged, mismatch = token_state(SUBMISSION)
    if privileged or mismatch:
        print("[grader] {}".format("privileged mode (oracle token)" if privileged else "oracle token mismatch: ignored"),
              file=sys.stderr, flush=True)
    flags = {"privileged_mode": float(privileged), "oracle_token_mismatch": float(mismatch)}
    # Separate verifier: the restored submission is root-owned; hand it to the agent before its program runs.
    own_submission(SUBMISSION)
    egress = egress_mode()
    print("[grader] egress guard: {}".format(egress), file=sys.stderr, flush=True)
    if egress != "seccomp":
        # Fail closed: the set is public, so a controller with the network could fetch it and replay its episode.
        fail("the grading environment cannot block the controller's network (the seccomp filter does not hold "
             "here); nothing is graded with the network open")
    flags["egress_guard_seccomp"] = 1.0  # (a verdict exists only when the filter held)
    cam, cam_flags = start_camera()      # (in this thread: the renderer is bound to the thread that made it)
    flags.update(cam_flags)
    try:
        try:
            # Raises SolverMissing (no run.sh) or SolverAborted (it never comes up): a real zero, not a broken task.
            solver = GuardedSolver(step_timeout=config.get("step_timeout_s", 30), submission=SUBMISSION,
                                   reply_budget=config.get("reply_budget_s", 1800))
        except (SolverMissing, SolverAborted) as exc:
            write_verdict(zero_verdict(reason=str(exc), anchored=anchor is not None, extra=flags))
            return
        try:
            qualities, successes, collisions, aborted, lock, stats = run_episodes(
                config, seeds, probe_key, solver, privileged, cam)
        except camera.CameraError as exc:
            fail(str(exc))
        finally:
            solver.close()
    finally:
        cam.close()

    measured = sum(qualities) / len(qualities) if qualities else 0.0
    extra = dict(
        lock.rewards(),
        episodes=float(len(qualities)),
        episodes_completed=float(len(qualities) - aborted),
        episodes_aborted=float(aborted),
        docking_successes=float(successes),
        target_collisions=float(collisions),
        **dict(flags, **stats)
    )
    for k, q in enumerate(qualities):
        extra["episode_{}_quality".format(k)] = q
    verdict = ratio_rewards(
        measured,
        anchor,
        cap=config.get("scoring_cap", 1.5),
        higher_is_better=True,
        metric_name="mean_docking_quality",
        locks_ok=lock.ok,
        extra=extra,
        full_at=anchor_meta.get("full_at", 1.0),
    )
    write_verdict(verdict)
    print(
        "[grader] measured={:.5f} anchor={} ratio={:.4f} reward={:.4f} lock={} successes={}/{} aborted={} "
        "privileged={} egress={} frames={} frame_ms={:.1f} reply_s={:.1f}".format(
            measured, anchor, verdict["metrics"]["ratio"], verdict["rewards"]["reward"], lock.ok, successes,
            len(qualities), aborted, privileged, egress, int(stats["camera_frames_rendered"]),
            stats["camera_frame_ms_mean"], stats["reply_s_total"]
        ),
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
