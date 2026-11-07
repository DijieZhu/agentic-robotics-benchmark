"""The grader for driving_safety_supervisor. Runs as ROOT in a separate verification
sandbox, after the agent is gone.

Closed-loop pattern: this process owns the simulator (CARLA 0.9.15 through `drive_bridge`,
a py3.10 child process run as its own account) and asks the agent's supervisor, every
0.1 s of simulated time, whether to let the nominal planner's control through or to
replace it. The episode score comes from the simulator's own route progress and infraction
record, never from anything the supervisor reports.

Why the result can be trusted:

* Separate verification: a fresh sandbox holds only the submission; nothing the agent left
  running in its own sandbox can reach the grading run.
* The supervisor runs at the agent's UID (`robotics_grader.Solver`) and, through
  `netguard`, with no IP sockets: CARLA's RPC port listens on loopback and would otherwise
  hand out every actor's true state (and the controls of the world). The guard is tested
  before any episode runs. CARLA runs as a third account (`carlasim`), from a copy of the
  bridge and its stagings in /run that only root and that account can read, removed when the
  run ends.
* The hidden episodes are a fixed set (hidden/episodes.json): the same 24 specs (town, kind,
  variant, seed, distractors), in the same order, for every submission. Each also holds its
  own probe-lock key and the digest of the layout its seed builds; the bridge reports that
  digest at every reset (`drive_bridge.layout_digest`: route, spawned actors, every draw),
  and an episode whose layout differs from the recorded one is a task error, never a
  silently different episode. Specs reach the bridge over root's pipe only; nothing of the
  set is logged. Each episode's world is reloaded before it is built, so nothing an earlier
  episode did carries into a later one.
* The probe lock re-asks random steps with a road user pasted onto the road in the front
  camera, so a supervisor that does not look at its camera fails the lock and scores 0.
  Every episode draws its probes from its own stream (the set's `lock_key`), so a
  submission meets the same probe schedule in every run, whatever ran before.
* The episode score (`episode_score`) lives only here; the practice grader reports what
  happened in an episode, not how it scores.

Oracle token: a submission whose `.oracle_token` (a regular file, never a link, at most
TOKEN_MAX_BYTES) hashes to `oracle_token.sha256` (next to this file; the token itself lives
only in `solution/`, which is absent while the agent works) runs in privileged mode: the
bridge adds a ground-truth block to every observation. Any other token is ignored and
reported as `oracle_token_mismatch`.
"""

import grp
import hashlib
import hmac
import json
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from drive_protocol import (  # noqa: E402
    EpisodeOutcome,
    ProtocolError,
    probe_observation,
    run_episode,
)
from drive_variants import MAX_DISTRACTORS, VARIANTS, graded_cells  # noqa: E402
from episode_score import aggregate, score  # noqa: E402
from robotics_grader import (  # noqa: E402
    ProbeLock,
    Solver,
    SolverAborted,
    SolverMissing,
    fail,
    ratio_rewards,
    read_anchor,
    seed_rng,
    stage_config,
    write_verdict,
    zero_verdict,
)
from robotics_grader.stage import agent_home, agent_user, secure_exec_path  # noqa: E402
from sim_client import SimClient, SimError  # noqa: E402

SET_PATH = os.path.join(HERE, "hidden", "episodes.json")
SET_FORMAT = "driving_safety_supervisor fixed set 1"
SEED_RANGE = (1 << 32, 1 << 62)
TOKEN_FILE = ".oracle_token"
TOKEN_MAX_BYTES = 4096
_HEX64 = re.compile(r"[0-9a-f]{64}")


def log(message):
    print("[grader] {}".format(message), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------- the fixed set
def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def load_set(config, path=None):
    """The fixed hidden set, checked: its format, one episode per graded cell in grading order
    (graded_cells over grader_config's towns), each kind the one its variant stages, seeds
    distinct integers in [2**32, 2**62), distractors in 0..MAX_DISTRACTORS, lock keys and
    layout digests 64 lowercase hex digits, lock keys and ids distinct. Any defect is a task
    error whose message names no value of the set."""
    path = path or SET_PATH
    try:
        with open(path) as handle:
            spec = json.load(handle)
    except (OSError, ValueError) as exc:
        fail("cannot read the hidden episode set: {}".format(type(exc).__name__))
    if not isinstance(spec, dict) or spec.get("format") != SET_FORMAT:
        fail("the hidden episode set is not of the format '{}'".format(SET_FORMAT))
    towns = list(config.get("towns", ["Town04", "Town05", "Town10HD"]))
    cells = graded_cells(towns)
    episodes = spec.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(cells):
        fail("the hidden episode set does not hold {} episodes".format(len(cells)))
    kind_of = {v: kind for kind, variants in VARIANTS.items() for v in variants}
    seeds, keys, ids = set(), set(), set()
    for k, (ep, cell) in enumerate(zip(episodes, cells)):
        if not isinstance(ep, dict):
            fail("hidden episode {}: not an object".format(k))
        if (ep.get("town"), ep.get("variant")) != cell:
            fail("hidden episode {}: not the graded cell due at this place in grading order".format(k))
        if ep.get("kind") != kind_of.get(ep.get("variant")):
            fail("hidden episode {}: its kind is not the one its variant stages".format(k))
        seed = ep.get("seed")
        if not _is_int(seed) or not SEED_RANGE[0] <= seed < SEED_RANGE[1]:
            fail("hidden episode {}: the seed is not an integer in [2**32, 2**62)".format(k))
        if not _is_int(ep.get("distractors")) or not 0 <= ep["distractors"] <= MAX_DISTRACTORS:
            fail("hidden episode {}: distractors not in 0..{}".format(k, MAX_DISTRACTORS))
        for field in ("lock_key", "layout_sha256"):
            if not isinstance(ep.get(field), str) or not _HEX64.fullmatch(ep[field]):
                fail("hidden episode {}: {} is not 64 lowercase hex digits".format(k, field))
        if not isinstance(ep.get("id"), str) or not ep["id"]:
            fail("hidden episode {}: no id".format(k))
        if seed in seeds or ep["lock_key"] in keys or ep["id"] in ids:
            fail("hidden episode {}: a seed, lock key or id of an earlier episode".format(k))
        seeds.add(seed)
        keys.add(ep["lock_key"])
        ids.add(ep["id"])
    return episodes


class LayoutChecked(object):
    """The simulator, with each reset checked against the layout digest the set recorded for
    the episode (`want`; None records it unchecked, for a maintainer's run off the set)."""

    def __init__(self, sim, want, index):
        self.sim, self.want, self.index = sim, want, index
        self.layout = None

    def reset(self, spec):
        state = self.sim.reset(spec)
        got = state.get("truth", {}).get("layout")
        self.layout = got if isinstance(got, str) else ""
        if self.want is not None and not hmac.compare_digest(
                self.layout.encode("ascii", "replace"), self.want.encode("ascii")):
            fail("hidden episode {}: the simulator built another layout than the set records "
                 "(the bridge, a staging or the simulator changed)".format(self.index))
        return state

    def step(self, mode, control):
        return self.sim.step(mode, control)


def grade_episode(sim, solver, lock, ep, index, privileged):
    """One hidden episode exactly as graded: the lock on the episode's own stream, the spec
    over root's pipe with its layout checked at reset, the probe pairs, an abort as a zero.
    Returns (outcome, the layout digest the bridge reported)."""
    lock.rng = seed_rng(entropy=int(ep["lock_key"], 16))
    spec = {key: ep[key] for key in ("town", "kind", "variant", "seed", "distractors")}
    if privileged:
        spec["privileged"] = True
    checked = LayoutChecked(sim, ep.get("layout_sha256"), index)

    def probe(t, obs, state):
        if not lock.due():
            return None
        return lock.probe(solver, t, obs, perturbed=probe_observation(obs, state))

    try:
        solver.reset()
        outcome = run_episode(checked, spec, solver.act, probe=probe)
    except (SolverAborted, ProtocolError) as exc:
        outcome = EpisodeOutcome({"kind": spec["kind"], "variant": spec["variant"]},
                                 aborted=True, note="aborted: {}".format(exc)[:120])
    return outcome, checked.layout


# ---------------------------------------------------------------------------- the submission
def token_state(submission, digest_path=None):
    """(privileged, mismatch) for the submission's .oracle_token against oracle_token.sha256.

    Only a regular file counts (opened without following a link and without blocking, at
    most TOKEN_MAX_BYTES); a token that is there but does not match (a wrong value, a link,
    a directory, a FIFO, an oversized file) is recorded as a mismatch and otherwise ignored."""
    digest_path = digest_path or os.path.join(HERE, "oracle_token.sha256")
    path = os.path.join(submission, TOKEN_FILE)
    try:
        with open(digest_path) as handle:
            want = handle.read().split()[0].strip().lower()
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


def own_submission(path):
    """Hand the restored submission back to the agent account.

    In the separate verifier the engine restores /home/user/submission as root, so its files
    land root-owned with their recorded modes. The supervisor runs AS THE AGENT: it must be
    able to read its own files (one left non-world-readable would otherwise be unreadable)
    and write beside them, exactly as it could while it worked. The walk follows no link (a
    restored tree has none; lchown is the safe habit). A no-op when the path is missing or is
    a link; the Solver then scores a real zero through SolverMissing."""
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


# ---------------------------------------------------------------------------- the sandbox
def guarded_entry(python):
    guard_src = os.path.join(HERE, "netguard.py")
    check = subprocess.run([python, "-I", "-S", guard_src, "selftest"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
    if check.returncode != 0:
        fail("netguard does not hold in this sandbox: {}".format(check.stdout.strip()))
    folder = tempfile.mkdtemp(prefix="drive-guard-",
                              dir="/run" if os.path.isdir("/run") else None)
    os.chmod(folder, 0o755)
    shutil.copy(guard_src, os.path.join(folder, "netguard.py"))
    submission = os.path.join(agent_home(), "submission")
    with open(os.path.join(folder, "run.sh"), "w") as handle:
        handle.write("#!/bin/bash\nexec {py} -I -S {guard} exec {sub} bash run.sh\n".format(
            py=python, guard=os.path.join(folder, "netguard.py"), sub=submission))
    for name in ("netguard.py", "run.sh"):
        os.chmod(os.path.join(folder, name), 0o644)
    return folder


def readable_bridge(sim_user):
    """(bridge path, folder to remove): the bridge with its stagings and probe sprites in a
    folder of /run owned by root and the simulator account's group, 0750 (files 0640), so the
    simulator can read them and the supervisor cannot. A task error if the agent's account
    belongs to that group. With no simulator account, the bridge in place and no folder."""
    bridge = os.path.join(HERE, "drive_bridge.py")
    if not sim_user:
        return bridge, None
    try:
        gid = pwd.getpwnam(sim_user).pw_gid
    except KeyError:
        fail("there is no account {!r} to run the simulator as".format(sim_user))
    agent = agent_user()
    try:
        if pwd.getpwnam(agent).pw_gid == gid or agent in grp.getgrgid(gid).gr_mem:
            fail("the agent's account belongs to the simulator account's group")
    except KeyError:
        pass
    folder = tempfile.mkdtemp(prefix="drive-bridge-", dir="/run" if os.path.isdir("/run") else None)
    os.chown(folder, 0, gid)
    os.chmod(folder, 0o750)
    for name in ("drive_bridge.py", "drive_variants.py"):
        target = os.path.join(folder, name)
        shutil.copy(os.path.join(HERE, name), target)
        os.chown(target, 0, gid)
        os.chmod(target, 0o640)
    sprites = os.path.join(folder, "probe_sprites")
    shutil.copytree(os.path.join(HERE, "probe_sprites"), sprites)
    os.chown(sprites, 0, gid)
    os.chmod(sprites, 0o750)
    for name in os.listdir(sprites):
        os.chown(os.path.join(sprites, name), 0, gid)
        os.chmod(os.path.join(sprites, name), 0o640)
    return os.path.join(folder, "drive_bridge.py"), folder


def start_simulator(config):
    """(SimClient, its log path, the bridge folder to remove); a task error if it does not
    come up."""
    sim_user = config.get("sim_user") or None
    sim_log = os.path.join(tempfile.mkdtemp(prefix="drive-simlog-"), "sim.log")
    sim_args = ["--image-size", str(config.get("image_size", 512)),
                "--background", str(config.get("background_vehicles", 20)),
                "--walkers", str(config.get("background_walkers", 20))]
    bridge, folder = readable_bridge(sim_user)
    try:
        sim = SimClient(backend=config.get("backend", "carla"), extra_args=sim_args,
                        log_path=sim_log, bridge=bridge, run_as=sim_user,
                        step_timeout=float(config.get("sim_step_timeout_s", 120)))
    except SimError as exc:
        _tail(sim_log)
        if folder:
            shutil.rmtree(folder, ignore_errors=True)
        fail("simulator did not start: {}".format(exc))
    return sim, sim_log, folder


def main():
    # nothing an unprivileged account could plant may precede the system tools root runs
    secure_exec_path()
    config = stage_config()
    episodes = load_set(config)
    anchor, anchor_meta = read_anchor()

    submission = os.path.join(agent_home(), "submission")
    entry = os.path.join(submission, "run.sh")
    if not os.path.isfile(entry):
        write_verdict(zero_verdict(reason="no submission at {}".format(entry),
                                   anchored=anchor is not None))
        return

    log("{} hidden episodes over {} towns".format(len(episodes), len(set(e["town"] for e in episodes))))
    # read before any of the submission's code runs
    privileged, mismatch = token_state(submission)
    if privileged:
        log("oracle token matched: privileged mode")
    elif mismatch:
        log("oracle token mismatch: ignored")
    own_submission(submission)

    folder = guarded_entry(config.get("guard_python", "/opt/venv/bin/python3"))
    log("supervisor runs as {} under netguard".format(agent_user()))
    lock = ProbeLock(None, rate=config.get("probe_rate", 0.02),
                     pass_fraction=config.get("probe_pass_fraction", 0.25))

    sim, sim_log, bridge_folder = start_simulator(config)
    outcomes, aborted, checked = [], 0, 0
    try:
        try:
            solver = Solver(step_timeout=config.get("step_timeout_s", 10), submission=folder)
        except (SolverMissing, SolverAborted) as exc:
            write_verdict(zero_verdict(reason=str(exc), anchored=anchor is not None))
            return
        try:
            for index, ep in enumerate(episodes):
                outcome, layout = grade_episode(sim, solver, lock, ep, index, privileged)
                aborted += int(outcome.aborted)
                checked += int(bool(layout))
                outcomes.append(outcome)
                log("episode {} [{} {} {}+{}] {}".format(
                    index, ep["town"], ep["kind"], ep["variant"], ep["distractors"],
                    outcome.record()))
        finally:
            solver.close()
    except SimError as exc:
        _tail(sim_log)
        fail("simulator failed mid-run: {}".format(exc))
    finally:
        sim.close()
        shutil.rmtree(folder, ignore_errors=True)
        if bridge_folder:
            shutil.rmtree(bridge_folder, ignore_errors=True)

    metrics = aggregate(outcomes)
    measured = metrics.pop("safety_drive_score")
    extra = dict(metrics)
    extra.update(lock.rewards())
    extra["episodes_aborted"] = float(aborted)
    extra["episodes_completed"] = float(sum(1 for o in outcomes
                                            if o.truth.get("termination") == "completed"))
    extra["layout_checked"] = 1.0 if checked == len(episodes) else 0.0
    for index, outcome in enumerate(outcomes):
        extra["episode_{:02d}_score".format(index)] = score(outcome)
    extra["privileged_mode"] = 1.0 if privileged else 0.0
    extra["oracle_token_mismatch"] = 1.0 if mismatch else 0.0
    verdict = ratio_rewards(
        measured, anchor,
        cap=config.get("scoring_cap", 1.5),
        higher_is_better=anchor_meta.get("direction", "higher") == "higher",
        metric_name="safety_drive_score",
        locks_ok=lock.ok,
        extra=extra,
        full_at=anchor_meta.get("full_at", 1.0),
    )
    write_verdict(verdict)
    log("safety_drive_score={:.4f} RC={:.3f} ped_hits={:.3f} anchor={} ratio={:.3f} "
        "reward={:.3f} lock={} ({}/{}) aborted={}".format(
            measured, metrics["route_completion"], metrics["pedestrian_collision_rate"],
            anchor, verdict["metrics"]["ratio"], verdict["rewards"]["reward"], lock.ok,
            lock.passed, lock.total, aborted))


def _tail(path, lines=60):
    try:
        with open(path) as handle:
            for line in handle.readlines()[-lines:]:
                print("[sim] " + line.rstrip(), file=sys.stderr)
    except OSError:
        pass


if __name__ == "__main__":
    main()
