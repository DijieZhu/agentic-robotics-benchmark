"""The grader for aerial_search_geolocation. Runs as ROOT in a separate verification
sandbox, after the agent is gone.

Closed-loop pattern: this process owns the simulator (a TravelUAV Unreal scene with the
AirSim plugin, in a child process: `sim_client` / `sim_bridge`) and asks
the agent's controller for one move plus its current target reports per step over the
kit's stdio wire. The mission's result is computed from the simulator's own state and the
placed target points, never from anything the controller says about its own success.

Why the result can be trusted:

* Separate verification. The verify image (tests/Dockerfile) carries the held-out scenes
  and their hidden placement pool; the agent's image has neither, so no target position
  can have been memorised.
* The controller runs at the agent's UID (`robotics_grader.Solver`) and, through `netguard`,
  with no IP sockets: the simulator API listens on loopback and would otherwise hand out the
  drone's true pose. The guard is tested before any mission runs. The simulator itself runs
  as a third account (`aerialsim`); the held-out scenes are readable by it alone.
* The pool is root-only (0600) and missions are assembled from the kit's in-memory seed
  stream: which start, which targets, whether a decoy is present.
* The probe lock re-asks random steps with different target descriptions (same images), so
  a controller that ignores what it is asked to find fails the lock and scores 0.

Oracle token. A submission whose `.oracle_token` hashes to `hidden/oracle_token.sha256`
runs in privileged mode: every observation also carries the placed objects' true positions
and how far each move could go before touching geometry (`sar_protocol.privileged_view`). That is how the reference expert (solution/) is graded; the
token exists only in solution/, which is absent from the sandbox while the agent works. A
wrong token is ignored and recorded (`oracle_token_mismatch`).
"""

import hashlib

import os
import shutil
import stat
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

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
from robotics_grader.stage import agent_home, agent_user  # noqa: E402
from sar_protocol import (  # noqa: E402
    MissionOutcome,
    ProtocolError,
    aggregate,
    assemble_missions,
    load_pool,
    perturb_targets,
    run_mission,
)
from sim_client import SimClient, SimError  # noqa: E402


def log(message):
    print("[grader] {}".format(message), file=sys.stderr, flush=True)


def guarded_entry(python):
    """A root-owned, world-readable folder whose run.sh starts the controller under
    netguard; the kit's Solver launches `bash run.sh` in it as the agent."""
    guard_src = os.path.join(HERE, "netguard.py")
    check = subprocess.run([python, "-I", "-S", guard_src, "selftest"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
    if check.returncode != 0:
        fail("netguard does not hold in this sandbox: {}".format(check.stdout.strip()))
    folder = tempfile.mkdtemp(prefix="aerial-guard-",
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
    """The stage folder is root-only; the simulator account gets its own copy."""
    bridge = os.path.join(HERE, "sim_bridge.py")
    if not sim_user:
        return bridge
    folder = tempfile.mkdtemp(prefix="aerial-bridge-")
    os.chmod(folder, 0o755)
    copy = os.path.join(folder, "sim_bridge.py")
    shutil.copy(bridge, copy)
    os.chmod(copy, 0o644)
    return copy


def token_state():
    """(privileged, mismatch): the submission's .oracle_token against hidden/oracle_token.sha256."""
    path = os.path.join(agent_home(), "submission", ".oracle_token")
    want_path = os.path.join(HERE, "hidden", "oracle_token.sha256")
    if not os.path.isfile(path) or os.path.islink(path) or not os.path.isfile(want_path):
        return False, False
    try:
        with open(path, "rb") as handle:
            token = handle.read().strip()
        with open(want_path) as handle:
            want = handle.read().strip()
    except OSError:
        return False, False
    got = hashlib.sha256(token).hexdigest()
    return got == want, got != want


def check_pool_private(path):
    mode = os.stat(path).st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        fail("hidden pool {} is readable by non-root accounts (mode {:o})".format(
            path, stat.S_IMODE(mode)))


def main():
    config = stage_config()
    rng = seed_rng()
    anchor, anchor_meta = read_anchor()
    budget = int(config.get("action_budget", 1000))

    entry = os.path.join(agent_home(), "submission", "run.sh")
    if not os.path.isfile(entry):
        write_verdict(zero_verdict(reason="no submission at {}".format(entry),
                                   anchored=anchor is not None))
        return

    pool_path = config.get("pool", "/opt/aerialsim/hidden/pool.json")
    backend = config.get("backend", "airsim")
    if backend == "airsim":
        check_pool_private(pool_path)
    pool = load_pool(pool_path)
    missions = assemble_missions(
        pool, rng, per_map=int(config.get("missions_per_map", 8)),
        n_targets=int(config.get("targets_per_mission", 3)),
        min_dist=float(config.get("min_target_distance_m", 12.0)),
        max_dist=float(config.get("max_target_distance_m", 35.0)),
        absent_probability=float(config.get("absent_probability", 0.5)))
    if not missions:
        fail("no mission could be assembled from the hidden pool")
    descriptions = sorted({t["description"] for entry in pool["maps"].values()
                           for t in entry["targets"]})
    log("{} hidden missions over {} scenes; budget {} actions".format(
        len(missions), len({m["map_name"] for m in missions}), budget))

    privileged, mismatch = token_state()
    folder = guarded_entry(config.get("guard_python", "/opt/venv/bin/python3"))
    log("controller runs as {} under netguard{}".format(
        agent_user(), "; privileged mode (oracle token)" if privileged else ""))
    lock = ProbeLock(rng, rate=config.get("probe_rate", 0.04),
                     pass_fraction=config.get("probe_pass_fraction", 0.25))

    sim_user = config.get("sim_user") or None
    sim_log = os.path.join(tempfile.mkdtemp(prefix="aerial-simlog-"), "sim.log")
    sim_args = ["--image-size", str(config.get("image_size", 256)),
                "--gpu", str(config.get("gpu", 0))]
    try:
        sim = SimClient(backend=backend, extra_args=sim_args, log_path=sim_log,
                        bridge=readable_bridge(sim_user), run_as=sim_user,
                        step_timeout=float(config.get("sim_step_timeout_s", 120)))
    except SimError as exc:
        _tail(sim_log)
        fail("simulator did not start: {}".format(exc))

    outcomes, aborted = [], 0
    try:
        try:
            solver = Solver(step_timeout=config.get("step_timeout_s", 30), submission=folder)
        except (SolverMissing, SolverAborted) as exc:
            write_verdict(zero_verdict(reason=str(exc), anchored=anchor is not None))
            return
        try:
            for index, mission in enumerate(missions):
                decoys = [rng.choice(descriptions) for _ in mission["targets"]]

                def probe(t, obs, _decoys=decoys):
                    if not lock.due():
                        return None
                    return lock.probe(solver, t, obs, perturbed=perturb_targets(obs, _decoys))

                try:
                    solver.reset()
                    outcome = run_mission(sim, mission, solver.act, budget, probe=probe,
                                          privileged=privileged)
                except (SolverAborted, ProtocolError) as exc:
                    aborted += 1
                    outcome = MissionOutcome([0.0] * len(mission["targets"]), False, 0.0,
                                             budget, 0, "aborted: {}".format(exc)[:120],
                                             aborted=True)
                outcomes.append(outcome)
                log("mission {} [{}] decoy={} {}".format(
                    index, mission["map_name"],
                    any(t["position"] is None for t in mission["targets"]), outcome.record()))
        finally:
            solver.close()
    except SimError as exc:
        _tail(sim_log)
        fail("simulator failed mid-run: {}".format(exc))
    finally:
        sim.close()
        shutil.rmtree(folder, ignore_errors=True)

    metrics = aggregate(outcomes)
    measured = metrics.pop("geolocation_score")
    extra = dict(metrics)
    extra.update(lock.rewards())
    extra["missions_aborted"] = float(aborted)
    extra["privileged_mode"] = 1.0 if privileged else 0.0
    extra["oracle_token_mismatch"] = 1.0 if mismatch else 0.0
    verdict = ratio_rewards(
        measured, anchor,
        cap=config.get("scoring_cap", 1.5),
        higher_is_better=anchor_meta.get("direction", "higher") == "higher",
        metric_name="geolocation_score",
        locks_ok=lock.ok,
        extra=extra,
        full_at=anchor_meta.get("full_at", 1.0),
    )
    write_verdict(verdict)
    log("geolocation_score={:.4f} target_credit={:.3f} collisions={:.3f} anchor={} "
        "ratio={:.3f} reward={:.3f} lock={} ({}/{}) aborted={}".format(
            measured, metrics["target_credit"], metrics["collision_rate"], anchor,
            verdict["metrics"]["ratio"], verdict["rewards"]["reward"], lock.ok,
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
