"""Grader of cross_embodiment_adaptation: the agent's mission program on the hidden sites.

Runs as root in a SEPARATE verifier sandbox: the engine restores only the declared artifact
/home/user/submission there, and setup does not run. For every site of tests/hidden/sites.json
(fixed, chosen by the pre-registered rule in tests/review/selection.md) the grader

  * starts its own robot service (graded mode: one mission, no reset or truth endpoint),
  * runs the program once (`bash run.sh` from a read-only copy of the submission, as the agent
    account, a minimal environment, a fresh HOME, no network: tests/nonet.py),
  * stops it when the mission has ended (10 s grace), when its program time (wall time minus
    the time the service spends answering) passes the budget, or at the wall cap,
  * kills every process of the agent account (bounded, then frozen: the #64 survivor pattern
    of arm_hidden_mechanism_box) and reads the mission's outcome from the service's own ledger,
  * removes everything the account left behind (files on every writable mount, System V IPC) and
    restores the files it can write but does not own (tests/isolation.py): a mission starts from
    the state the first one started from, so no site is a trial for a later one.

Before the first mission the grader checks the machine's arithmetic (tests/numerics_check.py against
tests/numerics.json) and refuses to grade on a machine that computes differently.

The score is the fraction of hidden missions in which the Go1 safely docked, relative to
the anchor. A service failure voids the run without redeploying the target.
"""
import json
import math
import os
import pwd
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from isolation import purge_files, purge_ipc, restore, writable_snapshot  # noqa: E402
from nonet import nonet_argv, selftest  # noqa: E402
from robotics_grader import (  # noqa: E402
    agent_user, ratio_rewards, read_anchor, require, stage_config, stage_dir, write_verdict, zero_verdict,
)
from robotics_grader.stage import secure_exec_path  # noqa: E402

RUNTIME = HERE/"runtime"
PAYLOAD = HERE/"setup_payload"
HIDDEN = HERE/"hidden"
WORK = Path("/opt/arb-verify")              # root-only records of the graded missions
RUN = Path("/opt/arb-run")                  # the program's read-only copy and its homes
SOCKET = Path("/run/visual_robotics/world.sock")
PUBLIC_CLIENT = Path("/opt/visual_robotics/client")
PYTHON = "/opt/venv/bin/python3"
RUNUSER = "/usr/sbin/runuser"
ENV = "/usr/bin/env"
MAX_SUBMISSION_BYTES = 64*1024*1024
MAX_SUBMISSION_FILES = 2000

STAGE_COUNTS = (
    "corridor_entered", "first_aisle_correct", "first_shortcut_selected", "fell_before_aisle",
    "partition_crossed", "middle_crossed", "second_aisle_entered", "second_aisle_correct",
    "second_shortcut_selected", "second_partition_crossed", "dock_zone_entered",
    "contact_first", "contact_middle", "contact_second", "contact_boundary", "contact_bollard",
    "duct_entered", "duct_cleared", "deck_entered", "deck_cleared", "contact_duct", "contact_deck",
    "contact_load_floor", "target_budget_stop",
)
VERIFIER_STOPS = {
    "program_exit_before_done": "program exited",
    "program_timeout": "program time budget spent",
    "mission_wall_timeout": "mission wall cap reached",
}


def log(msg):
    print("[grader] " + msg, file=sys.stderr, flush=True)


# --- the agent account's processes (arm_hidden_mechanism_box tests/verify.py, #64) ------------- #
def _agent_pids(uid):
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/{}/status".format(name)) as f:
                zombie = False
                for line in f:
                    if line.startswith("State:"):
                        # a zombie runs no code; it only waits to be reaped
                        zombie = line.split()[1].startswith(("Z", "X"))
                    elif line.startswith("Uid:"):
                        if int(line.split()[1]) == uid and not zombie:
                            pids.append(int(name))
                        break
        except (OSError, ValueError, IndexError):
            continue
    return pids


def kill_agent_processes(uid, rounds=20, pause=0.25):
    """Kill every process of the agent account; True when none is left (bounded)."""
    for _ in range(rounds):
        pids = _agent_pids(uid)
        if not pids:
            return True
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(pause)
    return not _agent_pids(uid)


def _proc_state(pid):
    """The /proc state letter of `pid` ("X" once it is gone)."""
    try:
        with open("/proc/{}/status".format(pid)) as f:
            for line in f:
                if line.startswith("State:"):
                    return line.split()[1]
    except (OSError, IndexError):
        pass
    return "X"


def _still_running(uid, bound_s=3.0):
    """SIGSTOP every live process of `uid` until a pass finds none still running (a respawn that appears meanwhile is
    stopped by the next pass) or `bound_s` elapsed; returns the PIDs still running (empty: every one is stopped)."""
    deadline = time.monotonic() + float(bound_s)
    while True:
        for pid in _agent_pids(uid):
            try:
                os.kill(pid, signal.SIGSTOP)
            except OSError:
                pass
        time.sleep(0.05)
        running = [pid for pid in _agent_pids(uid) if _proc_state(pid) not in ("T", "t", "X")]
        if not running or time.monotonic() > deadline:
            return running


def stop_agent_processes(uid):
    """kill_agent_processes, then: True when none is left or every one left is stopped. The base image's supervisord
    respawns the agent account's desktop as fast as it is killed, so in the separate verifier a process that outlives
    the bounded kill is the image's own; it is stopped (a stopped process does not exit, so it is not respawned) and
    only one still running once stopped fails the run (cleanup_failed)."""
    if kill_agent_processes(uid):
        return True
    running = _still_running(uid)
    log("{} process(es) of the agent account outlived the kill, {} still running after the freeze".format(
        len(_agent_pids(uid)), len(running)))
    return not running


# --- the submission ---------------------------------------------------------------------------- #
def copy_submission(source, target):
    """Regular files and directories of `source` (no link is followed or copied) into a root-owned,
    world-readable `target`. Returns None, or the reason the submission is unusable."""
    source = Path(source)
    if not source.is_dir() or source.is_symlink():
        return "no submission folder"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    total = files = 0
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        rel = Path(dirpath).relative_to(source)
        dirnames[:] = [d for d in dirnames if d != "__pycache__" and not (Path(dirpath)/d).is_symlink()]
        for d in dirnames:
            (target/rel/d).mkdir()
        for name in filenames:
            path = Path(dirpath)/name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                continue
            files += 1
            total += info.st_size
            if files > MAX_SUBMISSION_FILES or total > MAX_SUBMISSION_BYTES:
                return "submission larger than {} files / {} MB".format(MAX_SUBMISSION_FILES,
                                                                         MAX_SUBMISSION_BYTES >> 20)
            shutil.copyfile(path, target/rel/name, follow_symlinks=False)
            os.chmod(target/rel/name, 0o755 if info.st_mode & 0o111 else 0o644)
    for dirpath, _, _ in os.walk(target):
        os.chmod(dirpath, 0o755)
    if not (target/"run.sh").is_file():
        return "no run.sh in the submission"
    return None


def stage_public_client():
    """The client the agent was given, at the path it had (setup does not run in this sandbox)."""
    if PUBLIC_CLIENT.exists():
        shutil.rmtree(PUBLIC_CLIENT)
    shutil.copytree(PAYLOAD, PUBLIC_CLIENT)
    for path in [PUBLIC_CLIENT, *PUBLIC_CLIENT.rglob("*")]:
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    os.chmod(PUBLIC_CLIENT.parent, 0o755)


# --- one mission ------------------------------------------------------------------------------- #
def _ledger(directory):
    try:
        return json.loads((directory/"state.json").read_text())
    except (OSError, ValueError):
        return None


def _stop(proc, grace=10.):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _tail_reader(stream, keep, limit=16384):
    """Keep the last `limit` bytes of the program's output, never more (it is never printed: a hidden site's
    layout must not reach the published verify log)."""
    def pump():
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                keep[0] = (keep[0]+chunk)[-limit:]
        except (OSError, ValueError):
            pass
    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    return thread


def program_argv(home):
    env = ["PATH=/opt/venv/bin:/usr/bin:/bin", "HOME={}".format(home), "TMPDIR={}/tmp".format(home),
           "LANG=C.UTF-8", "ROBOT_SOCKET={}".format(SOCKET), "MUJOCO_GL=osmesa", "LIBGL_ALWAYS_SOFTWARE=1",
           "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "PYTHONNOUSERSITE=1"]
    return [RUNUSER, "-u", agent_user(), "--", ENV, "-i"] + env + nonet_argv(["/bin/bash", "run.sh"], PYTHON)


def run_mission(index, seed, uid, gid, config, token_sha256, snapshot):
    """Run the frozen program once; infra records an error, not a retry."""
    directory = WORK/"mission-{:02d}-0".format(index)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, mode=0o700)
    (directory/"seed").write_text(str(seed))
    os.chmod(directory/"seed", 0o600)
    if SOCKET.exists() or SOCKET.is_symlink():
        SOCKET.unlink()
    SOCKET.parent.mkdir(parents=True, exist_ok=True)
    os.chown(SOCKET.parent, 0, 0)
    os.chmod(SOCKET.parent, 0o755)
    service_log = open(directory/"service.log", "wb")
    service_argv = [PYTHON, "-E", "-s", "-B", str(RUNTIME/"service.py"), "--mode", "graded",
                    "--directory", str(directory), "--socket", str(SOCKET), "--seed-file", str(directory/"seed"),
                    "--token-sha256", token_sha256]
    service = subprocess.Popen(
        service_argv,
        cwd=str(RUNTIME), stdout=service_log, stderr=service_log, start_new_session=True,
        env={"PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8", "MUJOCO_GL": "osmesa",
             "LIBGL_ALWAYS_SOFTWARE": "1", "LP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
             "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"})
    record = {"site": index, "infra": False}
    try:
        deadline = time.monotonic()+180
        while not (directory/"ready").exists():
            if service.poll() is not None or time.monotonic() > deadline:
                record["infra"] = True
                record["reason"] = "robot service did not start"
                return record
            time.sleep(.1)
        home = RUN/"home-{:02d}-0".format(index)
        if home.exists():
            shutil.rmtree(home)
        (home/"tmp").mkdir(parents=True)
        for path in (home, home/"tmp"):
            os.chown(path, uid, gid)
            os.chmod(path, 0o700)
        budget = float(config.get("program_seconds", 600))
        wall_cap = float(config.get("mission_wall_seconds", 1800))
        grace = float(config.get("exit_grace_seconds", 10))
        output = [b""]
        started = time.monotonic()
        program = subprocess.Popen(program_argv(home), cwd=str(RUN/"program"), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={"PATH": "/usr/bin:/bin"})
        reader = _tail_reader(program.stdout, output)
        ended_at, reason, program_time = None, "program exited", 0.
        while True:
            time.sleep(.2)
            now = time.monotonic()
            state = _ledger(directory) or {}
            busy = float(state.get("busy_seconds", 0.))
            in_flight = state.get("request_started")
            if in_flight:
                busy += max(0., time.time()-float(in_flight))
            program_time = max(0., now-started-busy)
            if state.get("done") and ended_at is None:
                ended_at = now
            if program.poll() is not None:
                reason = "program exited" if not state.get("done") else "mission ended"
                break
            if ended_at is not None and now-ended_at > grace:
                reason = "mission ended (program stopped after the grace period)"
                break
            if program_time > budget:
                reason = "program time budget spent"
                break
            if now-started > wall_cap:
                reason = "mission wall cap reached"
                break
            if service.poll() is not None:
                reason = "robot service exited"
                record["infra"] = True
                break
        wall = time.monotonic()-started
        record["cleanup_ok"] = stop_agent_processes(uid)
        # Nothing the program left may reach the next site (tests/isolation.py).
        left = purge_files(uid, keep=(str(RUN/"program"),), keep_dirs=(account_home(),)) + purge_ipc(uid)
        restored, unrestorable = restore(snapshot)
        record["left_behind"] = left
        record["files_restored"] = restored
        if unrestorable:
            record["cleanup_ok"] = False
            log("site {}: {} shared file(s) changed and could not be restored".format(index, len(unrestorable)))
        if program.poll() is None:
            try:
                os.killpg(program.pid, signal.SIGKILL)
            except OSError:
                pass
            program.wait()
        reader.join(5)
        _stop(service)
        state = _ledger(directory) or {}
        if state.get("internal_errors"):
            record["infra"] = True
            reason = "robot service internal error"
        metrics = state.get("metrics") or {}
        record.update({"reason": reason, "program_seconds": round(program_time, 2), "wall_seconds": round(wall, 2),
                       "requests": int(state.get("requests", 0)),
                       "success": float(metrics.get("success", 0.)), "switched": float(metrics.get("switched", 0.)),
                       "scout_actions": float(metrics.get("scout_actions", 0)),
                       "ground_looks": int(metrics.get("ground_looks", 0)),
                       "deck_support_feet": int(metrics.get("deck_support_feet", 0)),
                       "target_actions": float(metrics.get("target_actions", 0)),
                       "target_ticks": int(metrics.get("target_ticks", 0)),
                       "final_goal_distance": float(metrics.get("final_goal_distance", -1)),
                       "contact_stop": float(metrics.get("target_contact", 0)),
                       "target_fell": float(metrics.get("target_fell", 0)),
                       "requests_exhausted": int(state.get("requests_exhausted", False)),
                       "max_forward_m": float(metrics.get("max_forward_m", 0)),
                       "travel_m": float(metrics.get("travel_m", 0)),
                       **{key: metrics.get(key) for key in (
                           "target_outcome", "contact_wall", "contact_part", "min_duct_gap_m", "passage_choices",
                           "passage_lateral_margin_m")},
                       **{key: int(metrics.get(key, 0)) for key in STAGE_COUNTS},
                       **{key: int(reason == value) for key, value in VERIFIER_STOPS.items()}})
        (directory/"program_output.txt").write_bytes(output[0])
        (directory/"record.json").write_text(json.dumps(record, indent=1))
        return record
    finally:
        _stop(service)
        service_log.close()
        if SOCKET.exists():
            SOCKET.unlink()


def account_home():
    return os.path.normpath(os.path.expanduser("~"+agent_user()))


def check_numerics():
    """Refuse to grade where the simulation or a program's arithmetic differs from the recorded machine."""
    expected = json.loads((HERE/"numerics.json").read_text())
    proc = subprocess.run([PYTHON, "-I", "-B", str(HERE/"numerics_check.py")], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=600,
                          env={"PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8",
                               "MUJOCO_GL": "osmesa", "LIBGL_ALWAYS_SOFTWARE": "1", "LP_NUM_THREADS": "2"})
    require(proc.returncode == 0, "numerics check failed to run: {}".format(proc.stderr.decode("utf-8", "replace")[-400:]))
    report = json.loads(proc.stdout.decode().strip().splitlines()[-1])
    require(report["fingerprint"] == expected["fingerprint"],
            "this machine computes differently from the recorded one (differs: {}); not graded".format(
                ", ".join(report.get("differs") or ["?"])))
    log("numerics fingerprint matches ({})".format(report["fingerprint"][:12]))


def main():
    started = time.monotonic()
    config = stage_config()
    anchor, calibration = read_anchor()
    try:
        account = pwd.getpwnam(agent_user())
    except KeyError:
        write_verdict(zero_verdict("no agent account", anchored=anchor is not None))
        return
    uid, gid = account.pw_uid, account.pw_gid
    secure_exec_path()
    if not stop_agent_processes(uid):
        write_verdict(zero_verdict("agent processes could not be stopped (cleanup_failed)",
                                   anchored=anchor is not None))
        return
    WORK.mkdir(parents=True, exist_ok=True)
    os.chmod(WORK, 0o700)
    RUN.mkdir(parents=True, exist_ok=True)
    os.chmod(RUN, 0o755)
    problem = copy_submission(os.path.join(os.path.expanduser("~"+agent_user()), "submission"), RUN/"program")
    if problem:
        write_verdict(zero_verdict(problem, anchored=anchor is not None))
        return
    sites = json.loads((HIDDEN/"sites.json").read_text())["seeds"]
    token_sha256 = (HIDDEN/"oracle_token.sha256").read_text().split()[0]
    require(len(token_sha256) == 64, "malformed oracle token hash")
    check_numerics()
    stage = Path(stage_dir())
    stage_mode = stat.S_IMODE(stage.stat().st_mode)
    os.chmod(stage, 0o700)     # nothing of the tests stage is readable while agent code runs
    try:
        stage_public_client()
        ok, detail = selftest(lambda argv: [RUNUSER, "-u", agent_user(), "--", ENV, "-i",
                                            "PATH=/opt/venv/bin:/usr/bin:/bin"] + argv, PYTHON)
        require(ok, detail)
        # The submission was copied (root-owned, RUN/program); what the account still owns anywhere goes,
        # and the files it can write without owning them are recorded, before the first mission.
        purge_files(uid, keep=(str(RUN/"program"),), keep_dirs=(account_home(),))
        purge_ipc(uid)
        as_account = lambda argv: [RUNUSER, "-u", agent_user(), "--", *argv]  # noqa: E731
        as_account.uid = uid
        snapshot = writable_snapshot(as_account, skip=(str(RUN), str(WORK)))
        log("isolation: {} file(s) the account can write without owning them are restored after every site"
            .format(len(snapshot)))
        records, cleanup_ok = [], True
        for index, seed in enumerate(sites):
            record = run_mission(index, int(seed), uid, gid, config, token_sha256, snapshot)
            require(not record["infra"], "site {}: service failure, no redeployment ({})".format(
                index, record.get("reason")))
            cleanup_ok = cleanup_ok and record.get("cleanup_ok", True)
            records.append(record)
            log("site {}: success={:.0f} first={} second={} dock_zone={} scout={:.0f} target={:.0f} "
                "program={:.0f}s wall={:.0f}s ({}; {})".format(
                    index, record["success"], record["first_aisle_correct"], record["second_aisle_correct"],
                    record["dock_zone_entered"], record["scout_actions"], record["target_actions"],
                    record["program_seconds"], record["wall_seconds"], record["target_outcome"], record["reason"]))
    finally:
        os.chmod(stage, stage_mode)
    measured = sum(r["success"] for r in records)/len(records)
    if not cleanup_ok:
        log("a process of the agent account survived a mission: the run scores 0 (cleanup_failed)")
        measured = 0.
    extra = {"missions": len(records), "missions_succeeded": sum(r["success"] for r in records),
             "missions_switched": sum(r["switched"] for r in records),
             "missions_contact_stop": sum(r["contact_stop"] for r in records),
             "mean_scout_actions": sum(r["scout_actions"] for r in records)/len(records),
             "mean_ground_looks": sum(r["ground_looks"] for r in records)/len(records),
             "mean_target_actions": sum(r["target_actions"] for r in records)/len(records),
             "missions_fell": sum(r["target_fell"] for r in records),
             "mean_max_forward_m": sum(r["max_forward_m"] for r in records)/len(records),
             "mean_travel_m": sum(r["travel_m"] for r in records)/len(records),
             "mean_program_seconds": sum(r["program_seconds"] for r in records)/len(records),
             "cleanup_failed": 0. if cleanup_ok else 1., "verify_seconds": time.monotonic()-started}
    counts = (*STAGE_COUNTS, "requests_exhausted", *VERIFIER_STOPS)
    extra.update({"missions_"+key: sum(r[key] for r in records) for key in counts})
    for r in records:
        prefix = "site{}_".format(r["site"])
        for key in ("success", "switched", "contact_stop", "target_fell", "max_forward_m", "target_ticks",
                    "final_goal_distance", "ground_looks", "deck_support_feet", *counts):
            extra[prefix+key] = r[key]
        if r["min_duct_gap_m"] is not None:
            extra[prefix+"min_duct_gap_m"] = r["min_duct_gap_m"]
    extra = {k: float(v) for k, v in extra.items() if math.isfinite(float(v))}
    write_verdict(ratio_rewards(measured, anchor, metric_name=calibration["metric"], extra=extra,
                                full_at=calibration["full_at"]))


if __name__ == "__main__":
    main()
