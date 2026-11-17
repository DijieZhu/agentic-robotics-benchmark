"""Grader of uav_assembly_agipix_xarm7: scores physical workcell outcomes across a process/UID
boundary. Runs as root in the verify stage (tests/test.sh).

Verification is separate (task.toml): this runs in a fresh verifier sandbox started from the task image, into which the
engine restored only the declared artifact, /home/user/submission (root-owned). Nothing else of the agent's sandbox is
here, environment/setup/setup.sh never ran, and nothing of the agent's home but that submission is read.
  1. checks the grader's root-owned copy of the givens (ARB_ASSETS = /opt/arb; tests/test.sh builds it from the image's
     task assets and tests/setup_payload/) file by file against given_hashes.json, untouched run included: the cell's
     physics (given/kit, given/xarm7), the cameras' scenery (render/) and the program's runner (cell_runner.py,
     cell_client.py) come from it, and a mismatch is a task error;
  2. no regular submission/assemble.py -> zero verdict;
  3. snapshots the submission into a private temporary folder (regular files only, at most 60 MiB, read-only), keeps
     the agent account's processes stopped (verify_freeze) and checks the oracle token: a .oracle_token whose sha256
     equals oracle_token.sha256 runs the privileged reference (reference.py, motion.py) on the nominal layout;
  4. the static half of the network audit (netaudit.py): the snapshot's own .py files import no networking module,
     load no module under a computed name, execute no computed code and start no other program; a hit -> zero verdict
     (metric network_imports);
  5. per episode (grader_config.json `episodes`, a 63-bit seed from OS entropy each): builds the cell (cell.py), runs the
     snapshot's assemble.py as `nobody` through /opt/arb/cell_runner.py with a minimal environment (HOME = TMPDIR = an
     empty work folder), serves its requests over stdin/stdout (server.py; camera frames from rendering.py) within the
     simulated-time, wall, camera-frame and silence budgets, then kills its process group and every process of `nobody`.
     The runtime half of the network audit spans the program's run: the namespace's outbound counters before and after
     (the grader itself opens no socket meanwhile) and `nobody`'s remote sockets sampled every socket_sample_s; any
     activity voids the submission (the remaining episodes are not run; metrics net_outbound_events,
     remote_sockets_seen, net_<counter>);
  6. assembly_score = the episode mean of cell.result()'s score (the fastened mates' weight / the total weight; an unsafe
     episode scores 0), set against tests/anchor.json, the reward zeroed by a network hit (network_audit_ok 0). The
     episode records (episode.json, trajectory.npz, program.log) are written root-only to OUT, outside /opt/ale, so that a
     kept verifier sandbox still holds them; they are never scoring inputs.
The network audit is enforced in the sandbox (root, ALE_STAGE_DIR set); a diagnostic run on a host shares the host's
network, so there it is recorded (network_audit_enforced 0) but voids nothing.
"""
import hashlib
import json
import os
from pathlib import Path
import pwd
import queue
import secrets
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

import netaudit
from robotics_grader import read_anchor, ratio_rewards, stage_config, write_verdict, zero_verdict
from verify_freeze import AgentFreezer, uid_processes

STAGE = Path(os.environ.get("ALE_STAGE_DIR", Path(__file__).parent)).resolve()
TASK_HOME = Path(os.environ.get("ALE_HOME", "/home/user"))
ASSETS = Path(os.environ.get("ARB_ASSETS", "/opt/arb"))
# The episode records: root-only and outside /opt/ale, which the engine deletes before it keeps a verifier sandbox
# ([sandbox_retention] verifier = "keep").
OUT = Path(os.environ.get("ARB_VERIFY_OUT", "/opt/arb-verify/out" if os.getuid() == 0 else STAGE / "out"))


def snapshot(source, destination):
    """No symlink following, special files, shared ownership, or executable loading."""
    destination.mkdir(mode=0o755)
    total = 0
    for base, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(base)/d).is_symlink()]
        rel = Path(base).relative_to(source)
        target = destination / rel
        target.mkdir(exist_ok=True, mode=0o755)
        for name in files:
            src = Path(base) / name
            fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("Submission contains a special file")
                total += info.st_size
                if total > 60 * 1024**2:
                    raise ValueError("Submission exceeds 60 MiB")
                with os.fdopen(os.dup(fd), "rb") as stream, (target/name).open("wb") as out:
                    shutil.copyfileobj(stream, out)
                (target/name).chmod(0o444)
            finally:
                os.close(fd)
    for base, dirs, files in os.walk(destination):
        Path(base).chmod(0o555)


def root_assets_check():
    want = json.loads((STAGE / "given_hashes.json").read_text())
    for rel, digest in want.items():
        p = ASSETS / rel
        st = p.lstat()
        if not stat.S_ISREG(st.st_mode) or (os.getuid() == 0 and st.st_uid != 0) or st.st_mode & 0o022:
            raise RuntimeError("Trusted payload permissions: " + rel)
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Trusted payload hash mismatch: " + rel)


def network_audit_enforced():
    return os.getuid() == 0 and bool(os.environ.get("ALE_STAGE_DIR"))


def static_network_audit(program):
    """[(file, finding)]: the static half of the network audit over the snapshot's own .py files. Copies of the shipped
    client and runner (setup_payload/, byte-identical to /opt/arb's) are exempt."""
    shipped = {hashlib.sha256(p.read_bytes()).hexdigest() for p in (STAGE / "setup_payload").rglob("*.py")}
    return netaudit.scan_imports(str(program), allow_digests=shipped)


def static_network_reason(hits):
    listed = "; ".join("%s: %s" % hit for hit in hits[:5]) + (" and %d more" % (len(hits) - 5) if len(hits) > 5 else "")
    return ("network audit: the submission's own .py files import a networking module, load a module under a computed "
            "name, execute computed code or start a program (%s)" % listed)


def runtime_network_reason(index, network):
    """None, or why episode `index`'s network record voids the submission."""
    if not network["enforced"] or (network["outbound"] <= 0 and network["remote_sockets_seen"] <= 0):
        return None
    grown = ", ".join("%s +%d" % (k, v) for k, v in sorted(network["delta"].items()) if v > 0) or "none"
    return ("network audit: network activity while the program ran in episode %d (outbound counters: %s; remote "
            "sockets of the runner account seen: %d)" % (index, grown, network["remote_sockets_seen"]))


def kill_group(process, runner_uid):
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    # Children can setsid; the dedicated nobody UID is exclusive to submitted code
    # inside the task sandbox. Do not apply this sweep in host diagnostic mode.
    if os.getuid() == 0 and os.environ.get("ALE_STAGE_DIR"):
        for pid in uid_processes(runner_uid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def write_response(process, response, deadline):
    """A client that stops reading must not hang the privileged verifier."""
    pending = memoryview((json.dumps(response, allow_nan=False) + "\n").encode())
    fd = process.stdin.fileno()
    os.set_blocking(fd, False)
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("RPC reply was not consumed within the program budget")
        if not select.select([], [fd], [], remaining)[1]:
            raise TimeoutError("RPC reply was not consumed within the program budget")
        try:
            written = os.write(fd, pending[:65536])
        except BlockingIOError:
            continue
        pending = pending[written:]


def episode(config, seed, program, out, privileged):
    from cell import Cell
    from server import Server, RequestError
    import numpy as np
    cell = Cell(ASSETS, STAGE / "assembly_graph.json", seed=seed,
                jitter=not (privileged and config["reference_nominal_layout"]))
    cell.recording = True
    server = Server(cell, config, privileged)
    runner_uid = pwd.getpwnam("nobody").pw_uid if os.getuid() == 0 else os.getuid()
    process, failure, stderr_thread = None, None, None
    started = time.monotonic()
    cap = config["reference_wall_cap_s"] if privileged else config["wall_cap_s"]
    env = {"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "HOME": str(out / "work"), "TMPDIR": str(out / "work"),
           "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "MPLBACKEND": "Agg"}
    (out / "work").mkdir(mode=0o700)
    if os.getuid() == 0:
        os.chown(out / "work", runner_uid, pwd.getpwnam("nobody").pw_gid)
    argv = [sys.executable, "-s", str(ASSETS/"cell_runner.py"), str(program)]
    if os.getuid() == 0:
        argv = ["/usr/sbin/runuser", "-u", "nobody", "--"] + argv
    messages = queue.Queue(maxsize=2)
    def read_requests():
        try:
            while True:
                line = process.stdout.readline(65537)
                if not line:
                    messages.put(None); return
                if len(line) > 65536 or not line.endswith(b"\n"):
                    messages.put(ValueError("RPC request exceeds 64 KiB")); return
                messages.put(line)
        except Exception as error:
            messages.put(error)
    enforced = network_audit_enforced()
    counters = netaudit.NetCounters()
    sampler = netaudit.SocketSampler(uid=runner_uid if enforced else None, period_s=config.get("socket_sample_s", .5))
    log = (out/"program.log").open("w")
    try:
        with counters, sampler:
            try:
                process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           bufsize=0, env=env, cwd=out/"work", start_new_session=True)
                def drain_stderr():
                    remaining = 4 * 1024**2
                    try:
                        while True:
                            chunk = process.stderr.read(4096)
                            if not chunk:
                                return
                            if remaining > 0:
                                log.write(chunk[:remaining].decode("utf-8", errors="replace")); log.flush()
                                remaining -= len(chunk)
                    except (ValueError, UnicodeError):
                        return
                stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
                stderr_thread.start()
                threading.Thread(target=read_requests, daemon=True).start()
                while not server.done:
                    remaining = cap - (time.monotonic()-started)
                    if remaining <= 0:
                        failure = "wall_budget"; break
                    try:
                        request = messages.get(timeout=min(remaining, config["silence_s"]))
                    except queue.Empty:
                        failure = "program_silence"; break
                    if request is None:
                        break
                    if isinstance(request, Exception):
                        failure = str(request); break
                    try:
                        try:
                            obj = json.loads(request)
                        except (ValueError, RecursionError):
                            raise RequestError("Malformed or excessively nested JSON request") from None
                        value = server.handle(obj)
                        response = dict(ok=True, result=value)
                    except RequestError as error:
                        response = dict(ok=False, error=str(error))
                    write_response(process, response, min(started+cap, time.monotonic()+config["silence_s"]))
            finally:
                # Every process of the runner account is gone before the counters are read.
                kill_group(process, runner_uid)
    except (BrokenPipeError, UnicodeError, TimeoutError) as error:
        failure = type(error).__name__
    finally:
        if stderr_thread is not None:
            stderr_thread.join(timeout=2)
        log.close()
        server.close()
    result = cell.result()
    result.update(seed=seed, program_error=failure, wall_s=time.monotonic()-started,
                  camera_frames=server.frames, calls=server.calls, privileged=privileged,
                  network=dict(enforced=enforced, outbound=counters.total, delta=counters.delta,
                               remote_sockets_seen=sampler.max_seen, socket_samples=sampler.samples))
    # A crash keeps physically earned partial credit. Safety still zeroes it.
    (out/"episode.json").write_text(json.dumps(result, indent=2, default=float)+"\n")
    np.savez_compressed(out/"trajectory.npz", qpos=np.asarray(cell.frames), dt=.04, seed=seed)
    shutil.rmtree(out/"work")
    return result


def main():
    anchor, meta = read_anchor()
    # The givens first, whatever was submitted: a verifier whose copy differs from given_hashes.json is a task error.
    root_assets_check()
    submission = TASK_HOME / "submission"
    if not (submission / "assemble.py").is_file() or (submission / "assemble.py").is_symlink():
        write_verdict(zero_verdict(reason="No regular assemble.py submission", anchored=anchor is not None))
        return
    if os.getuid() == 0:
        STAGE.chmod(0o700)
    freezer = AgentFreezer.for_user(TASK_HOME.name).start()
    try:
        config = stage_config()
        # Snapshot and runner working directories are outside the private stage,
        # readable by nobody; the grader/config/poses remain untraversable.
        with tempfile.TemporaryDirectory(prefix="arb-assembly-", dir="/tmp") as temp:
            work = Path(temp)
            work.chmod(0o755)
            program = work/"program"
            try:
                snapshot(submission, program)
            except (OSError, ValueError) as error:
                write_verdict(zero_verdict(reason=str(error), anchored=anchor is not None)); return
            token = program/".oracle_token"
            privileged = token.is_file() and hashlib.sha256(token.read_bytes().strip()).hexdigest() == (STAGE/"oracle_token.sha256").read_text().strip()
            hits = static_network_audit(program)
            if hits:
                write_verdict(zero_verdict(reason=static_network_reason(hits), anchored=anchor is not None,
                                           extra=dict(network_audit_ok=0, network_imports=len(hits),
                                                      privileged_reference=float(privileged))))
                return
            out = OUT
            out.mkdir(mode=0o700, parents=True, exist_ok=True)
            results, network_failure = [], None
            for i in range(config["episodes"]):
                # Public runner data lives outside private evidence path.
                run = work/f"ep{i:02d}"
                run.mkdir(mode=0o755)
                result = episode(config, secrets.randbits(63), program, run, privileged)
                results.append(result)
                shutil.copytree(run, out/run.name, dirs_exist_ok=True)
                net = result["network"]
                print(json.dumps(dict(episode=i, score=result["score"], safety=result["safety_ok"],
                                      sim_s=result["sim_time_s"], net_outbound=net["outbound"],
                                      remote_sockets=net["remote_sockets_seen"])), flush=True)
                network_failure = runtime_network_reason(i, net)
                if network_failure:
                    break   # the submission is void: further episodes cannot change the verdict
            measured = sum(x["score"] for x in results)/len(results)
            verdict = ratio_rewards(measured, anchor, metric_name="assembly_score", higher_is_better=True,
                                    full_at=meta["full_at"], locks_ok=network_failure is None)
            net_totals = {}
            for x in results:
                for k, v in x["network"]["delta"].items():
                    net_totals["net_" + k] = net_totals.get("net_" + k, 0) + max(0, v)
            verdict["metrics"].update(safety_failure_rate=sum(not x["safety_ok"] for x in results)/len(results),
                                      completed_mates_mean=sum(len(x["achieved"]) for x in results)/len(results),
                                      privileged_reference=float(privileged), episodes_run=float(len(results)),
                                      network_audit_ok=float(network_failure is None),
                                      network_audit_enforced=float(network_audit_enforced()), network_imports=0.0,
                                      net_outbound_events=float(sum(x["network"]["outbound"] for x in results)),
                                      remote_sockets_seen=float(max(x["network"]["remote_sockets_seen"] for x in results)),
                                      **{k: float(v) for k, v in net_totals.items()})
            if network_failure:
                print("[grader] scoring 0: " + network_failure, file=sys.stderr, flush=True)
                verdict["diagnostic"] = "scoring 0: " + network_failure
            write_verdict(verdict)
    finally:
        freezer.stop()


if __name__ == "__main__":
    main()
