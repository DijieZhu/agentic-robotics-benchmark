"""Hidden grader for multisensor_rig_calibration_from_logs (artifact rollout on a fixed set of hidden rigs).

Runs as root in a SEPARATE verifier sandbox: no agent process exists there, the setup stage has not
run there, and the engine restores only the declared artifact /home/user/submission.

1. Checks the numerics fingerprint (numerics_check.py against numerics.json): a host that does not
   reproduce it is a task error, never a score.
2. Fetches the hidden rigs' logs: ONE archive, named and pinned (size, sha256) in grader_config.json
   `hidden`, downloaded into this root-only folder and unpacked into a fresh work folder; every
   unpacked file is checked against hidden/<rig>/sha256.json. The rigs are a FIXED set, generated once
   (tests/gen/, pre-registered seeds); their true parameters, faults and held-out sequences are in
   hidden/<rig>/ and never leave this folder (but for the reference, see 4).
3. Clears the agent's account (robotics_grader.account.clear_agent_account) and runs the submission's
   `bash run.sh --input <rig> --output <out>` once per rig, all rigs at once, each at the agent's UID
   through nonet_exec.py (a seccomp filter: AF_UNIX sockets only, so no network), at the lowest
   scheduling priority, with cwd /home/user/submission, an address-space limit of `run_memory_mb`
   and a limit of `run_cpu_s` seconds of the run's OWN processor time (user + system of every process
   of the run, sampled from /proc; the grader's work and other runs do not count), plus a wall-clock
   stop at `run_wall_s`. A run that reaches a limit is killed with its processes and scores on
   whatever files it has left (normally none).
4. The reference (solution/solve.sh) is token-privileged: when the submission's .oracle_token hashes
   to hidden/oracle_token.sha256, each run also gets `--truth <dir>` with that rig's true parameters
   and fault list. No other submission ever sees them.
5. Reads each rig's calibration.yaml and fault_report.json (regular files only, at most 1 MiB, as
   data: yaml.safe_load / json) and measures them with rigmetrics.py (byte-identical to the agent's
   practice copy). Each rig earns a credit in [0, 1]:

       credit = 0.10 * faults + 0.90 * (c_projection * c_motion * c_time) ** (1/3)

   c_projection = c(projection_px), c_motion = mean of c(rpe_rot_deg) and c(rpe_trans_m),
   c_time = c(time_rms_s), all over the two held-out drives (one before
   the given sequences, one after; on a rig with a knock the mount differs between them); c(e) = clamp(log(zero/e) / log(zero/full), 0, 1) with the `full` and
   `zero` points of grader_config.json `credit`. The three are measured on the held-out sequence and
   each needs several parameters right together; their geometric mean is zero when one is. A
   calibration.yaml that is missing or does not parse earns 0 on that part; a missing or malformed
   fault_report.json earns 0 on the fault term.
6. The metric `mean_rig_credit` (higher is better) is the mean over the rigs; the kit turns it into
   the reward against the anchor.

Nothing the submission prints is read. The log names rigs by index only. The grader's own failures
(the archive cannot be fetched or does not match, an account that cannot be cleared, a fingerprint
mismatch) are task errors (`fail`), never a zero.
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
import tarfile
import tempfile
import time
import urllib.request

# One BLAS thread everywhere: the measurement's arithmetic must not depend on the host's core count.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

STAGE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE)

from robotics_grader import (agent_user, fail, ratio_rewards, read_anchor,  # noqa: E402
                             stage_config, write_verdict, zero_verdict)
from robotics_grader.account import clear_agent_account  # noqa: E402
from robotics_grader.stage import secure_exec_path  # noqa: E402

HOME = os.environ.get("ALE_HOME", "/home/user")
SUBMISSION = os.path.join(HOME, "submission")
NONET = os.path.join(STAGE, "nonet_exec.py")
HIDDEN = os.path.join(STAGE, "hidden")
TOKEN_SHA = os.path.join(HIDDEN, "oracle_token.sha256")
CLK_TCK = os.sysconf("SC_CLK_TCK")
MAX_READ = 1 << 20


def log(msg):
    print("[grader] " + msg, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- agent processes

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
                        zombie = line.split()[1].startswith(("Z", "X"))
                    elif line.startswith("Uid:"):
                        if int(line.split()[1]) == uid and not zombie:
                            pids.append(int(name))
                        break
        except (OSError, ValueError, IndexError):
            continue
    return pids


def kill_agent_processes(uid, rounds=20, pause=0.25):
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


# ----------------------------------------------------------------------------- the hidden rigs

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_hidden(cfg, work):
    """Download the hidden rigs' logs (one pinned archive) into this root-only folder, unpack them under
    work/rigs and check every file against hidden/<rig>/sha256.json. Returns the rig names."""
    spec = cfg["hidden"]
    rigs = list(spec["rigs"])
    arc = os.path.join(STAGE, "hidden_logs.tar")
    local = os.environ.get("RIGCAL_HIDDEN_ARCHIVE")          # a maintainer's local copy (never set by the engine)
    last = None
    for attempt in range(4):
        try:
            if local:
                shutil.copyfile(local, arc)
            else:
                req = urllib.request.Request(spec["url"], headers={"User-Agent": "rigcal-verify"})
                with urllib.request.urlopen(req, timeout=120) as r, open(arc, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            if os.path.getsize(arc) == int(spec["bytes"]) and _sha256(arc) == spec["sha256"]:
                last = None
                break
            last = "size or sha256 mismatch"
        except Exception as exc:  # network errors of any kind: retry, then a task error
            last = "{}: {}".format(type(exc).__name__, str(exc)[:200])
        time.sleep(5 * (attempt + 1))
    if last is not None:
        fail("the hidden rigs' archive could not be fetched: " + last)
    os.chmod(arc, 0o600)
    root = os.path.join(work, "rigs")
    os.makedirs(root)
    with tarfile.open(arc) as tf:
        tf.extractall(root, filter="data")
    os.remove(arc)
    for name in rigs:
        with open(os.path.join(HIDDEN, name, "sha256.json")) as f:
            want = json.load(f)
        rd = os.path.join(root, name)
        have = {}
        for dirpath, _dirs, files in os.walk(rd):
            for fn in files:
                p = os.path.join(dirpath, fn)
                have[os.path.relpath(p, rd)] = _sha256(p)
        if have != want:
            fail("hidden rig {} does not match its file list ({} files against {})".format(name, len(have), len(want)))
    extra = sorted(set(os.listdir(root)) - set(rigs))
    if extra:
        fail("the hidden archive holds unexpected entries: {}".format(extra[:5]))
    return rigs


def oracle_token_ok():
    """True when the submission's .oracle_token (a regular file, not followed if a link, at most 256 bytes)
    hashes to hidden/oracle_token.sha256: the reference, and nothing else."""
    try:
        fd = os.open(os.path.join(SUBMISSION, ".oracle_token"), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return False
        data = os.read(fd, 257)
    finally:
        os.close(fd)
    if len(data) > 256:
        return False
    try:
        with open(TOKEN_SHA) as f:
            want = f.read().split()[0]
    except (OSError, IndexError):
        return False
    return hashlib.sha256(data.strip()).hexdigest() == want


def _open_tree(root):
    for dirpath, dirnames, filenames in os.walk(root):
        os.chmod(dirpath, 0o755)
        for f in filenames:
            os.chmod(os.path.join(dirpath, f), 0o644)


# ----------------------------------------------------------------------------- running

def _read_small(path):
    """Contents of a regular file of at most MAX_READ bytes, never following a symlink."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        return None, "cannot open ({})".format(type(exc).__name__)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None, "not a regular file"
        if st.st_size > MAX_READ:
            return None, "larger than 1 MiB"
        data = b""
        while len(data) <= MAX_READ:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            data += chunk
        if len(data) > MAX_READ:
            return None, "larger than 1 MiB"
        return data.decode("utf-8", "replace"), None
    finally:
        os.close(fd)


def _proc_cpu(uid):
    """{(pid, starttime): (session id, user + system ticks, run marker or None)} of the agent account's processes."""
    out = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            if os.stat("/proc/" + name).st_uid != uid:
                continue
            with open("/proc/{}/stat".format(name), "rb") as f:
                rest = f.read().rsplit(b")", 1)[1].split()
            sid, ticks, start = int(rest[3]), int(rest[11]) + int(rest[12]), int(rest[19])
            marker = None
            try:
                with open("/proc/{}/environ".format(name), "rb") as f:
                    for item in f.read().split(b"\0"):
                        if item.startswith(b"RIGCAL_RUN="):
                            marker = int(item[11:])
                            break
            except (OSError, ValueError):
                pass
            out[(int(name), start)] = (sid, ticks, marker)
        except (OSError, ValueError, IndexError):
            continue
    return out


def _lowest_priority():
    try:
        os.nice(19)
    except OSError:
        pass


def run_submission(rig_dirs, out_dirs, truth_dirs, uid, gid, config):
    """All rigs at once. Each run is charged its OWN processor time: user + system of every process of the run
    (its session, or its RIGCAL_RUN mark), read from /proc every 0.2 s, the largest value seen per process. A
    process of the agent account that carries neither is charged to every run. Returns (status, tails):
    status[i] = (how, exit code, wall seconds, cpu seconds)."""
    cpu_limit = float(config.get("run_cpu_s", 1800))
    wall_limit = float(config.get("run_wall_s", 7200))
    mem_mb = int(config.get("run_memory_mb", 6144))
    procs = []
    logs = []
    for i, (rd, od) in enumerate(zip(rig_dirs, out_dirs)):
        lp = os.path.join(os.path.dirname(od), "run_%d.log" % i)
        fh = open(lp, "wb")
        cmd = [sys.executable, "-I", NONET, "--uid", str(uid), "--gid", str(gid), "--cwd", SUBMISSION,
               "--rlimit-as-mb", str(mem_mb), "--env", "RIGCAL_RUN=%d" % i, "--",
               "bash", "run.sh", "--input", rd, "--output", od]
        if truth_dirs is not None:
            cmd += ["--truth", truth_dirs[i]]
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True, env={"PATH": "/usr/bin:/bin"}, preexec_fn=_lowest_priority)
        procs.append(p)
        logs.append((fh, lp))
    t0 = time.monotonic()
    sid_run = {p.pid: i for i, p in enumerate(procs)}
    seen = {}                                  # (pid, start) -> (run index or -1, ticks)
    status = [None] * len(procs)

    def cpu_of(i):
        return sum(t for (r, t) in seen.values() if r == i or r == -1) / float(CLK_TCK)

    while any(s is None for s in status):
        for key, (sid, ticks, marker) in _proc_cpu(uid).items():
            run = sid_run.get(sid, marker if marker is not None and 0 <= marker < len(procs) else -1)
            old = seen.get(key)
            seen[key] = (run if old is None or old[0] == -1 else old[0], max(ticks, old[1] if old else 0))
        now = time.monotonic() - t0
        for i, p in enumerate(procs):
            if status[i] is not None:
                continue
            rc = p.poll()
            used = cpu_of(i)
            if rc is not None:
                status[i] = ("exit", rc, now, used)
            elif used > cpu_limit or now > wall_limit:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
                p.wait()
                status[i] = ("cpu limit" if used > cpu_limit else "wall limit", None, now, used)
        time.sleep(0.2)
    kill_agent_processes(uid)
    tails = []
    for fh, lp in logs:
        fh.close()
        try:
            with open(lp, "rb") as f:
                f.seek(max(0, os.path.getsize(lp) - 600))
                tails.append(f.read().decode("utf-8", "replace"))
        except OSError:
            tails.append("")
    return status, tails


# ----------------------------------------------------------------------------- scoring

def credit(err, full, zero):
    if err is None or not math.isfinite(err):
        return 0.0
    if err <= full:
        return 1.0
    if err >= zero:
        return 0.0
    return max(0.0, min(1.0, math.log(zero / err) / math.log(zero / full)))


def rig_credit(m, cfg):
    c = cfg["credit"]
    w_f = float(cfg.get("fault_weight", 0.10))
    terms = {
        "projection": credit(m.get("projection_px"), *c["projection_px"]),
        "motion": 0.5 * (credit(m.get("rpe_rot_deg"), *c["rpe_rot_deg"])
                         + credit(m.get("rpe_trans_m"), *c["rpe_trans_m"])),
        "time": credit(m.get("time_rms_s"), *c["time_rms_s"]),
        "faults": float(m.get("fault_f1", 0.0)),
    }
    joint = (terms["projection"] * terms["motion"] * terms["time"]) ** (1.0 / 3.0)
    terms["joint"] = joint
    return w_f * terms["faults"] + (1.0 - w_f) * joint, terms


def score_rig(truth, held, out_dir, cfg):
    import rigmetrics as M
    m = {}
    notes = []
    text, err = _read_small(os.path.join(out_dir, "calibration.yaml"))
    if text is None:
        notes.append("calibration.yaml: " + err)
    else:
        try:
            cal = M.parse_calibration(text)
            m.update(M.calibration_metrics(truth, held, cal))
        except M.CalibrationError as exc:
            notes.append(str(exc))
    text, err = _read_small(os.path.join(out_dir, "fault_report.json"))
    if text is None:
        notes.append("fault_report.json: " + err)
    else:
        try:
            rep = M.parse_fault_report(text)
            m["fault_f1"] = M.fault_f1(truth["faults"], rep)
        except M.CalibrationError as exc:
            notes.append(str(exc))
    total, terms = rig_credit(m, cfg)
    return total, terms, m, notes


# ----------------------------------------------------------------------------- main

def check_numerics():
    """A host whose arithmetic differs from the pinned fingerprint cannot grade: a task error."""
    with open(os.path.join(STAGE, "numerics.json")) as f:
        want = json.load(f)
    try:
        r = subprocess.run([sys.executable, "-I", os.path.join(STAGE, "numerics_check.py")], capture_output=True,
                           text=True, timeout=300, env={"PATH": "/opt/venv/bin:/usr/bin:/bin"})
        got = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as exc:
        fail("the numerics fingerprint could not be computed: {}: {}".format(type(exc).__name__, str(exc)[:200]))
    if got.get("fingerprint") != want.get("fingerprint"):
        diff = sorted(k for k in want.get("math", {}) if got.get("math", {}).get(k) != want["math"][k])
        fail("this host does not reproduce the numerics fingerprint (parts that differ: {})".format(diff[:8]))
    log("numerics fingerprint ok")


def main():
    secure_exec_path()
    cfg = stage_config()
    anchor, meta = read_anchor()
    anchored = anchor is not None
    check_numerics()
    run_sh = os.path.join(SUBMISSION, "run.sh")
    if not os.path.isfile(run_sh):
        write_verdict(zero_verdict(reason="no /home/user/submission/run.sh", anchored=anchored),
                      diagnostic="scoring 0: no /home/user/submission/run.sh")
        return
    try:
        pw = pwd.getpwnam(agent_user())
    except KeyError:
        fail("the agent account does not exist in the verifier sandbox")
    uid, gid = pw.pw_uid, pw.pw_gid

    cleared = clear_agent_account()
    if not cleared.get("cleared") and not cleared.get("skipped"):
        fail("the agent's account could not be cleared before grading: {}".format(
            "; ".join(cleared.get("left", []))[:300]))
    if not kill_agent_processes(uid):
        fail("processes of the agent account could not be stopped before grading")

    work = tempfile.mkdtemp(prefix="rigcal_", dir="/tmp")
    os.chmod(work, 0o755)
    t0 = time.monotonic()
    names = fetch_hidden(cfg, work)
    n_rigs = len(names)
    log("fetched and checked {} hidden rigs in {:.0f} s".format(n_rigs, time.monotonic() - t0))
    import rigmetrics
    rig_dirs = [os.path.join(work, "rigs", n) for n in names]
    out_dirs = [os.path.join(work, "out", "rig_%02d" % i) for i in range(n_rigs)]
    try:
        results = [rigmetrics.load_truth_files(rig_dirs[i], os.path.join(HIDDEN, names[i])) for i in range(n_rigs)]
    except Exception as exc:
        fail("a hidden rig's truth could not be read: {}: {}".format(type(exc).__name__, str(exc)[:300]))
    _open_tree(os.path.join(work, "rigs"))
    os.makedirs(os.path.join(work, "out"), exist_ok=True)
    os.chmod(os.path.join(work, "out"), 0o755)
    for od in out_dirs:
        os.makedirs(od)
        os.chown(od, uid, gid)
        os.chmod(od, 0o755)
    truth_dirs = None
    privileged = oracle_token_ok()
    if privileged:                                           # the reference alone is handed the truth
        truth_dirs = []
        for i in range(n_rigs):
            td = os.path.join(work, "truth", "rig_%02d" % i)
            os.makedirs(td)
            for fn in ("ground_truth.yaml", "faults.json"):
                shutil.copyfile(os.path.join(HIDDEN, names[i], fn), os.path.join(td, fn))
            truth_dirs.append(td)
        _open_tree(os.path.join(work, "truth"))

    t1 = time.monotonic()
    status, tails = run_submission(rig_dirs, out_dirs, truth_dirs, uid, gid, cfg)
    log("submission runs finished in {:.0f} s: {}".format(time.monotonic() - t1,
                                                           [(s[0], s[1], round(s[2]), round(s[3])) for s in status]))

    credits, per_terms, per_metrics, notes_all = [], [], [], []
    for i in range(n_rigs):
        truth, held = results[i]
        total, terms, m, notes = score_rig(truth, held, out_dirs[i], cfg)
        if status[i][0] != "exit":
            notes.append("run reached its %s (%.0f s of its own processor time, %.0f s on the clock) and was killed"
                         % (status[i][0], status[i][3], status[i][2]))
        elif status[i][1] != 0:
            notes.append("run.sh exited with status %s" % status[i][1])
        credits.append(total)
        per_terms.append(terms)
        per_metrics.append(m)
        notes_all.append(notes)
        log("rig_%02d credit %.4f terms %s metrics %s notes %s" % (
            i, total, {k: round(v, 3) for k, v in terms.items()},
            {k: round(v, 6) for k, v in m.items()}, notes))
    measured = sum(credits) / len(credits)

    extra = {"rigs": float(n_rigs), "oracle_token": 1.0 if privileged else 0.0}
    for k in ("projection", "motion", "time", "joint", "faults"):
        extra["term_" + k] = sum(t[k] for t in per_terms) / n_rigs
    for i, c in enumerate(credits):
        extra["rig_%02d_credit" % i] = c
    for key in ("projection_px", "rpe_rot_deg", "rpe_trans_m", "time_rms_s", "imu_rot_deg",
                "gyro_bias_err_rps", "accel_bias_err_mps2", "fault_f1"):
        vals = [m[key] for m in per_metrics if key in m]
        if vals:
            vals.sort()
            extra["median_" + key] = vals[len(vals) // 2]
        extra["rigs_with_" + key] = float(len(vals))
    extra["runs_timed_out"] = float(sum(1 for s in status if s[0] != "exit"))
    extra["max_run_cpu_s"] = max(s[3] for s in status)
    extra["max_run_wall_s"] = max(s[2] for s in status)
    extra["runs_failed"] = float(sum(1 for s in status if s[0] == "exit" and s[1] != 0))

    verdict = ratio_rewards(measured, anchor, cap=cfg.get("scoring_cap", 1.5), higher_is_better=True,
                            metric_name=meta.get("metric", "mean_rig_credit"), extra=extra,
                            full_at=meta.get("full_at", 1.0))
    lines = ["mean_rig_credit %.4f over %d hidden rigs" % (measured, n_rigs)]
    for i in range(n_rigs):
        lines.append("rig_%02d: credit %.3f %s %s%s" % (
            i, credits[i], {k: round(v, 3) for k, v in per_terms[i].items()},
            {k: (round(v, 5) if isinstance(v, float) else v) for k, v in per_metrics[i].items()},
            (" | " + "; ".join(notes_all[i])) if notes_all[i] else ""))
        if not per_metrics[i] and tails[i].strip():
            lines.append("  run output tail: " + tails[i].strip().replace("\n", " | ")[-300:])
    shutil.rmtree(work, ignore_errors=True)
    write_verdict(verdict, diagnostic="\n".join(lines))


if __name__ == "__main__":
    main()
