"""Grader of time_limited_search_and_rescue: the agent's flight program on the hidden sites.

Runs as root in a SEPARATE verifier sandbox: the engine restores only the declared artifact
/home/user/submission there, and setup does not run. For every site of tests/hidden/sites.json
(fixed, chosen by the pre-registered rule in tests/review/selection.md) the grader

  * starts its own aircraft service (graded mode: one flight, no reset, no truth),
  * runs the program once (`bash run.sh` from a read-only copy of the submission, as the agent
    account, a minimal environment, a fresh HOME, no network: tests/nonet.py),
  * stops it when the flight has ended (10 s grace), when its program time (wall time minus
    the time the service spends answering) passes the budget, or at the wall cap,
  * kills every process of the agent account (bounded, then frozen) and reads the flight's
    outcome from the service's own ledger.

The score is the mean rescue progress over the hidden flights (a completed handoff is 1.0),
relative to the anchor. A service failure voids the run; no site is ever flown twice.

Oracle token: a submission whose .oracle_token hashes to tests/oracle_token.sha256 gets the
privileged `_gt` block in every answer (solution/oracle/pilot.py plans on it). Any other token is
ignored and reported.
"""
import hashlib
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

from nonet import nonet_argv, selftest  # noqa: E402
from robotics_grader import (  # noqa: E402
    agent_user, clear_agent_account, ratio_rewards, read_anchor, require, stage_config, stage_dir, write_verdict,
    zero_verdict,
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

PROGRESS_KEYS = ("entered_wing", "mezzanine", "located", "grasped", "lifted", "cargo_return", "delivered")
ABORTS = ("wall", "inspection", "cargo", "slab", "roof", "throat_entry", "throat_exit", "loading_wall", "prop",
          "unsafe_release")
NUMERICS_OFF = "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR"
TOKEN_FILE = ".oracle_token"
TOKEN_MAX_BYTES = 4096
VERIFIER_STOPS = {
    "program_exit_before_done": "program exited",
    "program_timeout": "program time budget spent",
    "flight_wall_timeout": "flight wall cap reached",
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


# Where the agent account can write in the verifier image (enumerated with `find / -writable` as that account,
# tests/review/writable_paths.md): the trees it owns (/usr/local/{include,lib,share,bin}, its home, /tmp/runtime-user),
# the world-writable directories (/tmp, /var/tmp, /run/lock, /dev/shm, /dev/mqueue), two world-writable lock files,
# and System V IPC. close_agent_writable() takes the first and third away once; purge_agent_files() empties the
# rest before every site, so that nothing a program wrote during one site's flight is there for the next.
SKIP_TREES = ("/proc", "/sys")
IPCRM = "/usr/bin/ipcrm"


def close_agent_writable(uid, gid):
    """Once, before any agent code: every file and directory the agent account owns becomes root's (so it can no
    longer write there), every other file loses its world- and agent-group-write bit, and the directories that stay
    world-writable are returned for purge_agent_files(). The program's own per-site home is created later."""
    scratch = []
    for top, dirs, files in os.walk("/", topdown=True, followlinks=False):
        dirs[:] = [d for d in dirs if os.path.join(top, d) not in SKIP_TREES and not os.path.ismount(os.path.join(top, d))
                   or os.path.join(top, d) in ("/dev", "/dev/shm", "/dev/mqueue", "/run", "/tmp")]
        for name in [".", *dirs, *files]:
            path = top if name == "." else os.path.join(top, name)
            try:
                info = os.lstat(path)
            except OSError:
                continue
            if stat.S_ISLNK(info.st_mode):
                if info.st_uid == uid:
                    try:
                        os.lchown(path, 0, 0)
                    except OSError:
                        pass
                continue
            mode = stat.S_IMODE(info.st_mode)
            try:
                if info.st_uid == uid or info.st_gid == gid:
                    os.lchown(path, 0 if info.st_uid == uid else info.st_uid, 0 if info.st_gid == gid else info.st_gid)
                    os.chmod(path, mode & ~0o6000)
                if stat.S_ISDIR(info.st_mode):
                    if mode & 0o002 and name == ".":
                        scratch.append(path)
                elif mode & 0o002 and not stat.S_ISCHR(info.st_mode) and not stat.S_ISSOCK(info.st_mode) \
                        and not stat.S_ISFIFO(info.st_mode) and not stat.S_ISBLK(info.st_mode):
                    os.chmod(path, mode & ~0o002)
            except OSError:
                pass
    return sorted(set(scratch) | {"/dev/mqueue"})


def purge_agent_files(uid, scratch):
    """Before every site: what the agent account left in the world-writable directories, the earlier sites' homes,
    and System V IPC objects are removed."""
    for top in [*scratch, str(RUN)]:
        try:
            names = os.listdir(top)
        except OSError:
            continue
        for name in names:
            path = os.path.join(top, name)
            try:
                if os.lstat(path).st_uid != uid:
                    continue
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.unlink(path)
            except OSError:
                pass
    if os.path.exists(IPCRM):
        # As the agent account: a container's root has no CAP_IPC_OWNER, and the owner may always remove its own.
        subprocess.run([RUNUSER, "-u", agent_user(), "--", ENV, "-i", IPCRM, "--all"], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=30)


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
           "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "PYTHONNOUSERSITE=1",
           # One numerics path on every grading host (environment/Dockerfile sets the same two in the agent's
           # sandbox): without them an image-driven program's arithmetic, and so its flight, differs between an
           # AVX2 and an AVX-512 machine (measured; tests/review/host_dependence.md).
           "OPENBLAS_CORETYPE=HASWELL", "NPY_DISABLE_CPU_FEATURES="+NUMERICS_OFF]
    return [RUNUSER, "-u", agent_user(), "--", ENV, "-i"] + env + nonet_argv(["/bin/bash", "run.sh"], PYTHON)


def token_state(program):
    """(privileged, mismatch): whether the submission carries the oracle token."""
    path = program/TOKEN_FILE
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > TOKEN_MAX_BYTES:
            return False, path.exists()
        digest = hashlib.sha256(path.read_bytes().strip()).hexdigest()
        return digest == (HERE/"oracle_token.sha256").read_text().split()[0], True
    except (OSError, IndexError):
        return False, False


def run_mission(index, seed, uid, gid, config, privileged=False):
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
                    "--directory", str(directory), "--socket", str(SOCKET), "--seed-file", str(directory/"seed")]
    if privileged:
        service_argv.append("--truth")
    service = subprocess.Popen(
        service_argv,
        cwd=str(RUNTIME), stdout=service_log, stderr=service_log, start_new_session=True,
        env={"PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8", "MUJOCO_GL": "osmesa",
             "LIBGL_ALWAYS_SOFTWARE": "1", "LP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
             "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1", "OPENBLAS_CORETYPE": "HASWELL",
             "NPY_DISABLE_CPU_FEATURES": NUMERICS_OFF})
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
                reason = "flight wall cap reached"
                break
            if service.poll() is not None:
                reason = "robot service exited"
                record["infra"] = True
                break
        wall = time.monotonic()-started
        record["cleanup_ok"] = stop_agent_processes(uid)
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
                       "requests_exhausted": int(state.get("requests_exhausted", False)),
                       "progress": float(metrics.get("rescue_progress", 0.)),
                       "steps": int(metrics.get("steps", 0)), "rooms_seen": int(metrics.get("rooms_seen", 0)),
                       "tool_changes": int(metrics.get("tool_changes", 0)),
                       "fit_misses": int(metrics.get("fit_misses", 0)),
                       "close_attempts": int(metrics.get("close_attempts", 0)),
                       **{key: int(metrics.get(key, 0)) for key in PROGRESS_KEYS},
                       **{"abort_"+key: int(metrics.get(key+"_abort", 0)) for key in ABORTS},
                       **{key: int(reason == value) for key, value in VERIFIER_STOPS.items()}})
        record["ended_by"] = next((key for key in ABORTS if record["abort_"+key]),
                                  "handoff" if record["delivered"] else "no contact")
        (directory/"program_output.txt").write_bytes(output[0])
        (directory/"record.json").write_text(json.dumps(record, indent=1))
        return record
    finally:
        _stop(service)
        service_log.close()
        if SOCKET.exists():
            SOCKET.unlink()


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
    # The kit's clearing of the agent's account (once per verifier sandbox: every supervisord program off, every
    # process of the account killed until none comes back), before any agent code is started.
    clear_agent_account()
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
    stage = Path(stage_dir())
    stage_mode = stat.S_IMODE(stage.stat().st_mode)
    os.chmod(stage, 0o700)     # nothing of the tests stage is readable while agent code runs
    try:
        stage_public_client()
        ok, detail = selftest(lambda argv: [RUNUSER, "-u", agent_user(), "--", ENV, "-i",
                                            "PATH=/opt/venv/bin:/usr/bin:/bin"] + argv, PYTHON)
        require(ok, detail)
        privileged, mismatch = token_state(RUN/"program")
        if privileged:
            log("oracle token matched: privileged mode")
        elif mismatch:
            log("the submission carries an .oracle_token that does not match: ignored")
        records, cleanup_ok = [], True
        scratch = close_agent_writable(uid, gid)
        log("agent account closed out of the image: {} world-writable directories are emptied before every site".format(
            len(scratch)))
        for index, seed in enumerate(sites):
            purge_agent_files(uid, scratch)
            record = run_mission(index, int(seed), uid, gid, config, privileged)
            require(not record["infra"], "site {}: service failure, the site is not flown again ({})".format(
                index, record.get("reason")))
            cleanup_ok = cleanup_ok and record.get("cleanup_ok", True)
            records.append(record)
            log("site {}: progress={:.2f} reached={} ended by {} after {} actions; program={:.0f}s wall={:.0f}s ({})".format(
                index, record["progress"], "+".join(k for k in PROGRESS_KEYS if record[k]) or "nothing",
                record["ended_by"], record["steps"], record["program_seconds"], record["wall_seconds"],
                record["reason"]))
    finally:
        os.chmod(stage, stage_mode)
    measured = sum(r["progress"] for r in records)/len(records)
    if not cleanup_ok:
        log("a process of the agent account survived a flight: the run scores 0 (cleanup_failed)")
        measured = 0.
    n = len(records)
    extra = {"flights": n, "flights_delivered": sum(r["delivered"] for r in records),
             "mean_steps": sum(r["steps"] for r in records)/n,
             "mean_program_seconds": sum(r["program_seconds"] for r in records)/n,
             "oracle_token_matched": 1. if privileged else 0., "oracle_token_mismatch": 1. if mismatch and not privileged else 0.,
             "cleanup_failed": 0. if cleanup_ok else 1., "verify_seconds": time.monotonic()-started}
    counts = (*PROGRESS_KEYS, *("abort_"+k for k in ABORTS), "requests_exhausted", *VERIFIER_STOPS)
    extra.update({"flights_"+key: sum(r[key] for r in records) for key in counts})
    for r in records:
        prefix = "site{}_".format(r["site"])
        for key in ("progress", "steps", "rooms_seen", *counts):
            extra[prefix+key] = r[key]
    extra = {k: float(v) for k, v in extra.items() if math.isfinite(float(v))}
    write_verdict(ratio_rewards(measured, anchor, metric_name=calibration["metric"], extra=extra,
                                full_at=calibration["full_at"]))


if __name__ == "__main__":
    main()
