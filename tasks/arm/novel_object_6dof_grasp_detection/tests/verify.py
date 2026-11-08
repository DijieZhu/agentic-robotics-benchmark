"""Verify stage: re-run the submitted grasp detector on the hidden frames and score it.

Pattern: artifact_rollout. The agent leaves a Python module, /home/user/submission/solver.py,
whose predict(rgb, depth, intrinsics, depth_scale) returns grasps. This stage:

  1. scores an honest 0 when there is no solver.py (the untouched pass of `ale validate`);
  2. stops every process the agent's account left running. The image's supervisord respawns the desktop it runs
     as that account as fast as it is killed, so one that outlives the kill is frozen and recorded (only one still
     running once frozen voids the run), and every process of the account this stage did not start stays stopped
     until the verdict is written;
  3. fetches the hidden frames (28 GraspNet-1Billion novel-object scenes x 16 views, meshes and
     poses for their 26 objects; pinned and hash-checked) into this root-only stage folder;
  4. runs the detector in 8 processes as the agent's account, offline (seccomp, see
     predict_child.py), each on a block of 56 frames of one fixed shuffle (grader_config
     frame_order_seed), handing a process one frame's inputs (colour, depth, intrinsics) at a
     time over a pipe, and only after it has answered the previous frame. Every call is timed on
     this process's clock from the hand-over; an array that is not back within the limit is never
     accepted, and the process is killed and restarted (a fresh import) for the next frame, within
     the block's time budget. The arrays received in time go to a root-only folder; no frame is
     ever on disk where the agent's account could read it;
  5. stops the agent's account again and scores the accepted arrays as root with graspnetAPI's
     own force-closure evaluation (score_frames.py), which never imports submitted code;
  6. reports the AP the submission adds over the provided starting detector, RE-Bench style:
     ap_gain = max(0, mean AP over all 448 frames - the starting detector's own mean AP on them)
     (a frame without a valid array counts 0), as a ratio to the reference implementation's own
     measured gain. Copying the starting detector therefore scores 0 and the reference 1.

Everything here except the two helper scripts runs on the 3.12 /opt/venv interpreter with the
standard library and the vendored kit; graspnetAPI lives on /opt/grasp (Python 3.10).
"""
import hashlib
import json
import os
import pwd
import random
import select
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time

from robotics_grader import (agent_user, deprivileged, fail, ratio_rewards, read_anchor,
                             require, stage_config, stage_dir, write_verdict, zero_verdict)

GRASP_PY = "/opt/grasp/bin/python"
HOME = os.environ.get("ALE_HOME", "/home/user")
SUBMISSION = os.path.join(HOME, "submission")
# Set by make_workdir().
WORK = RESULTS = None
CHILD_ENV = {
    # subprocess resolves `runuser` (in /usr/sbin) through this PATH, not the grader's own
    "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    "HOME": HOME,
    "LANG": "C.UTF-8",
    "MPLBACKEND": "Agg",
    "GRASPNET_DATA": "/data/graspnet",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def log(msg):
    print("[grader] " + msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- processes
def _live_uid_pids(uid):
    """PIDs owned by `uid` that can still run. A zombie is already dead: SIGKILL cannot remove it, only its
    parent's wait() can, and in a fresh verifier sandbox nothing may reap it."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/status" % entry) as handle:
                lines = handle.read().splitlines()
        except (OSError, ValueError):
            continue
        state = next((l.split()[1] for l in lines if l.startswith("State:")), "")
        owner = next((int(l.split()[1]) for l in lines if l.startswith("Uid:")), -1)
        if owner == uid and state not in ("Z", "X"):
            pids.append(int(entry))
    return pids


def _proc_state(pid):
    """The /proc state letter of `pid` ("X" once it is gone)."""
    try:
        with open("/proc/%d/status" % pid) as handle:
            for line in handle:
                if line.startswith("State:"):
                    return line.split()[1]
    except (OSError, IndexError):
        pass
    return "X"


def _describe(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as handle:
            return "%d %s" % (pid, handle.read().replace(b"\0", b" ")[:120].decode(errors="replace"))
    except OSError:
        return "%d (gone)" % pid


def _freeze(uid, bound_s=3.0):
    """SIGSTOP every live process of `uid` until a pass finds none still running (a respawn that appears meanwhile is
    stopped by the next pass) or `bound_s` elapsed. Returns (stopped, running): the PIDs seen stopped and those still
    running at the end (empty: every one is stopped)."""
    deadline = time.time() + float(bound_s)
    stopped = set()
    while True:
        for pid in _live_uid_pids(uid):
            try:
                os.kill(pid, signal.SIGSTOP)
            except OSError:
                pass
        time.sleep(0.05)
        running = []
        for pid in _live_uid_pids(uid):
            if _proc_state(pid) in ("T", "t"):
                stopped.add(pid)
            else:
                running.append(pid)
        if not running or time.time() > deadline:
            return sorted(stopped), running


def stop_account(uid):
    """SIGKILL every live process of `uid`, repeating until none is left (they may fork). Returns the PIDs that
    outlived the kill and were frozen instead.

    The image's supervisord (PID 1) runs the desktop as the agent's account and respawns it as fast as it is killed,
    so in this separate verifier, where nothing of the agent's ever ran, a process that outlives the 20 passes is the
    image's own. It is frozen (a stopped process cannot exit, so it is not respawned) and recorded; only one still
    running once frozen voids the run."""
    for _ in range(20):
        pids = _live_uid_pids(uid)
        if not pids:
            return []
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.2)
    stopped, running = _freeze(uid)
    log("%d process(es) of uid %d outlived the kill, frozen: %s" % (len(stopped), uid,
                                                                     "; ".join(map(_describe, stopped)) or "none"))
    if running:
        log("still running as uid %d after the freeze: %s" % (uid, "; ".join(map(_describe, running))))
        fail("%d process(es) of uid %d outlived SIGKILL and could not be stopped" % (len(running), uid))
    return stopped


def _descends_from(pid, ancestor):
    """True if `ancestor` is a (grand)parent of `pid` (by /proc PPid)."""
    for _ in range(64):
        try:
            with open("/proc/%d/status" % pid) as handle:
                ppid = next(int(l.split()[1]) for l in handle if l.startswith("PPid:"))
        except (OSError, ValueError, StopIteration):
            return False
        if ppid == ancestor:
            return True
        if ppid <= 1:
            return False
        pid = ppid
    return False


class AccountFreezer(threading.Thread):
    """Keeps every process of the agent's account that this grader did not start stopped, polling /proc every
    `period_s`, until the verdict is written. The solver processes run as that account too and could SIGCONT a frozen
    process, or kill the desktop so that supervisord starts it again (running the account's autostart entries outside
    the network guard); either is stopped again within one period. Processes descended from this grader (the
    solvers, under runuser) are left alone."""

    def __init__(self, uid, period_s=0.1):
        threading.Thread.__init__(self, name="freezer", daemon=True)
        self.uid, self.period_s, self.frozen = uid, float(period_s), set()
        self.halt = threading.Event()
        self.me = os.getpid()

    def _exempt(self, pid):
        return _descends_from(pid, self.me)

    def freeze_now(self):
        for pid in _live_uid_pids(self.uid):
            if pid == self.me or self._exempt(pid):
                continue
            try:
                os.kill(pid, signal.SIGSTOP)
                self.frozen.add(pid)
            except OSError:
                pass

    def run(self):
        while not self.halt.is_set():
            try:
                self.freeze_now()
            except Exception:                              # the freezer must never take the grader down
                pass
            self.halt.wait(self.period_s)


# --------------------------------------------------------------------------- hidden frames
def fetch_hidden(cfg):
    sealed = os.path.join(stage_dir(), "sealed")
    marker = os.path.join(sealed, ".complete")
    if os.path.isfile(marker):
        return sealed
    shutil.rmtree(sealed, ignore_errors=True)
    os.makedirs(sealed, mode=0o700)
    os.chmod(sealed, 0o700)
    tarball = os.path.join(stage_dir(), "hidden.tar.gz")
    t0 = time.time()
    for url, dest, digest in ((cfg["hidden_bundle_url"], tarball, cfg["hidden_bundle_sha256"]),
                              (cfg["hidden_manifest_url"], os.path.join(sealed, "manifest.json"),
                               cfg["hidden_manifest_sha256"])):
        rc = subprocess.call(["curl", "-fsSL", "--retry", "10", "--retry-delay", "5",
                              "--retry-all-errors", "-o", dest, url])
        require(rc == 0, "could not download %s (curl exit %d)" % (url, rc))
        got = subprocess.check_output(["sha256sum", dest]).split()[0].decode()
        require(got == digest, "hash mismatch for %s: %s" % (url, got))
    rc = subprocess.call(["tar", "-xzf", tarball, "-C", sealed, "--strip-components=1",
                          "--no-same-owner"])
    require(rc == 0, "could not unpack the hidden frames (tar exit %d)" % rc)
    os.remove(tarball)
    open(marker, "w").close()
    log("hidden frames ready in %.0fs" % (time.time() - t0))
    return sealed


def hidden_frames(sealed):
    manifest = json.load(open(os.path.join(sealed, "manifest.json")))
    return [(r["scene"], v) for r in manifest["roles"]["hidden"] for v in r["views"]]


def privileged(cfg):
    """True only for a submission that carries the oracle token (solution/oracle_token).

    The token's SHA-256 is in grader_config.json; the agent never sees solution/ or the token, so only the
    reference detector run by `ale validate` is handed the ground truth of the frames. Its only job is to
    show that the task can be done."""
    path = os.path.join(SUBMISSION, "oracle_token")
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_size > 256:
            return False
        with open(path, "rb") as handle:
            token = handle.read().strip()
    except OSError:
        return False
    return hashlib.sha256(token).hexdigest() == cfg.get("oracle_token_sha256")


def export_ground_truth(sealed, frames):
    """Copy what graspnetAPI needs to label grasps (poses and meshes of the scenes' objects) into WORK/gt."""
    gt = os.path.join(WORK, "gt")
    objects = set()
    for sid in sorted({sid for sid, _ in frames}):
        src = os.path.join(sealed, "scenes", "scene_%04d" % sid)
        dst = os.path.join(gt, "scenes", "scene_%04d" % sid)
        os.makedirs(os.path.join(dst, "realsense", "meta"))
        os.makedirs(os.path.join(dst, "realsense", "annotations"))
        for name in ("camK.npy", "camera_poses.npy", "cam0_wrt_table.npy"):
            shutil.copyfile(os.path.join(src, "realsense", name), os.path.join(dst, "realsense", name))
        shutil.copyfile(os.path.join(src, "object_id_list.txt"), os.path.join(dst, "object_id_list.txt"))
        for s2, view in frames:
            if s2 == sid:
                for sub, ext in (("meta", "mat"), ("annotations", "xml")):
                    f = "%04d.%s" % (view, ext)
                    shutil.copyfile(os.path.join(src, "realsense", sub, f),
                                    os.path.join(dst, "realsense", sub, f))
        with open(os.path.join(src, "object_id_list.txt")) as handle:
            objects.update(int(v) for v in handle.read().split())
    for obj in sorted(objects):
        shutil.copytree(os.path.join(sealed, "models", "%03d" % obj), os.path.join(gt, "models", "%03d" % obj))
    CHILD_ENV["ORACLE_GT_ROOT"] = gt
    log("oracle token accepted: ground truth of %d scenes and %d objects exported" %
        (len({sid for sid, _ in frames}), len(objects)))


def make_workdir():
    """WORK: a fresh root-owned, world-traversable folder holding the root-owned copy of predict_child.py
    (and, for the privileged reference only, the ground truth). RESULTS: root-only, where the arrays the
    grader received in time are written; the agent's account can neither read nor write it. No frame is
    ever written anywhere the agent's account can read: each one reaches its solver process over a pipe,
    when that process asks for the next frame."""
    global WORK, RESULTS
    WORK = tempfile.mkdtemp(prefix="grasp_grading_", dir="/tmp")
    RESULTS = os.path.join(stage_dir(), "results")
    shutil.rmtree(RESULTS, ignore_errors=True)
    os.makedirs(RESULTS, mode=0o700)
    os.chmod(RESULTS, 0o700)


def export_child(sealed, frames, with_ground_truth=False):
    shutil.copyfile(os.path.join(stage_dir(), "predict_child.py"),
                    os.path.join(WORK, "predict_child.py"))
    if with_ground_truth:
        export_ground_truth(sealed, frames)
    for root, dirs, files in os.walk(WORK):
        os.chmod(root, 0o755)
        for name in files:
            os.chmod(os.path.join(root, name), 0o644)


# --------------------------------------------------------------------------- solver runs
# Wire format between this process and predict_child.py: a 1-byte tag, a 4-byte big-endian length, the payload.
MAX_TEXT = 4096
MAX_RESULT = 1024 + 5000 * 17 * 8               # the largest valid array as .npy (score_frames.MAX_BYTES)


def frame_message(sealed, name, privileged_run):
    sid, view = (int(v) for v in name.split("_"))
    src = os.path.join(sealed, "scenes", "scene_%04d" % sid, "realsense")
    blobs = []
    for path in (os.path.join(src, "rgb", "%04d.png" % view), os.path.join(src, "depth", "%04d.png" % view),
                 os.path.join(src, "camK.npy")):
        with open(path, "rb") as handle:
            blobs.append(handle.read())
    header = json.dumps({"name": name if privileged_run else "", "sizes": [len(b) for b in blobs]})
    payload = header.encode() + b"\n" + b"".join(blobs)
    return b"F" + struct.pack(">I", len(payload)) + payload


class Slot(threading.Thread):
    """One solver process working through a contiguous block of frames, restarted after a kill.

    Every frame is handed over only when the process has answered the previous one, and timed here, on the
    grader's monotonic clock, from the moment the hand-over starts: an array that is not back within the
    frame limit is never accepted, the process is killed and a fresh one (a fresh import) takes the next
    frame. Only arrays received in time are written to RESULTS, so only they are scored."""

    def __init__(self, index, names, cfg, logs, sealed, uid, privileged_run):
        threading.Thread.__init__(self, name="slot%d" % index, daemon=True)
        self.index, self.remaining, self.cfg, self.logs = index, list(names), cfg, logs
        self.sealed, self.uid, self.privileged_run = sealed, uid, privileged_run
        self.frame_limit = cfg["oracle_frame_timeout_s"] if privileged_run else cfg["frame_timeout_s"]
        # The block's own time budget: every frame at the limit plus the first import and `spare_imports`
        # re-imports at theirs. A process that is restarted after a timeout or a death pays its import out of
        # it; once it is spent, the block's remaining frames score 0. It keeps the stage inside its deadline.
        self.budget = len(self.remaining) * self.frame_limit + \
            (1 + int(cfg["spare_imports"])) * cfg["import_timeout_s"]
        self.budget_end = None
        self.attempt, self.proc, self.buf = 0, None, bytearray()
        self.status, self.elapsed = {}, {}
        self.guard_failed, self.error = False, None

    # ---- the child process
    def _start(self):
        self.attempt += 1
        tag = "%d_%d" % (self.index, self.attempt)
        log_file = open(os.path.join(self.logs, "solver_%s.log" % tag), "wb")
        self.proc = deprivileged(
            [GRASP_PY, "-I", os.path.join(WORK, "predict_child.py"), tag],
            cwd=SUBMISSION, env=CHILD_ENV, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log_file, start_new_session=True)
        log_file.close()
        self.wfd, self.rfd = self.proc.stdin.fileno(), self.proc.stdout.fileno()
        os.set_blocking(self.wfd, False)
        os.set_blocking(self.rfd, False)
        self.buf = bytearray()
        self.started_at = time.monotonic()

    def _kill(self):
        """SIGKILL the process group and every process of the agent's account still in this attempt's session."""
        proc = self.proc
        if proc is None:
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        for pid in _live_uid_pids(self.uid):
            try:
                if os.getsid(pid) == proc.pid:
                    os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        proc.wait()
        for handle in (proc.stdin, proc.stdout):
            try:
                handle.close()
            except OSError:
                pass
        self.proc = None

    def _send(self, data, deadline):
        """Write `data` before `deadline`; False if the process stopped reading or the time ran out."""
        view = memoryview(data)
        while view:
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            _, ready, _ = select.select([], [self.wfd], [], min(left, 1.0))
            if not ready:
                continue
            try:
                view = view[os.write(self.wfd, view):]
            except BlockingIOError:
                continue
            except OSError:                                # BrokenPipeError: the process is gone
                return False
        return True

    def _recv(self, deadline, cap):
        """The next message as (tag, payload); ('timeout'|'eof'|'oversize', None) otherwise."""
        while True:
            if len(self.buf) >= 5:
                size = struct.unpack(">I", bytes(self.buf[1:5]))[0]
                if size > cap:
                    return "oversize", None
                if len(self.buf) >= 5 + size:
                    tag, payload = bytes(self.buf[:1]), bytes(self.buf[5:5 + size])
                    del self.buf[:5 + size]
                    return tag, payload
            left = deadline - time.monotonic()
            if left <= 0:
                return "timeout", None
            ready, _, _ = select.select([self.rfd], [], [], min(left, 1.0))
            if not ready:
                continue
            try:
                chunk = os.read(self.rfd, 1 << 20)
            except BlockingIOError:
                continue
            except OSError:
                return "eof", None
            if not chunk:
                return "eof", None
            self.buf += chunk

    # ---- the frames
    def _give_up(self, why):
        for name in self.remaining:
            self.status[name] = why
        self.remaining = []

    def _import(self):
        """Start a process and wait for its import; True when it is ready for frames."""
        if time.monotonic() >= self.budget_end:
            self._give_up("timeout: the block's time budget (%ds) was spent" % self.budget)
            return False
        self._start()
        # G comes before the submission runs, so its wait is never cut short by the block's budget
        tag, payload = self._recv(self.started_at + self.cfg["import_timeout_s"], MAX_TEXT)
        if tag != b"G":
            # Before G only predict_child.py's own code has run (the seccomp filter goes in before the
            # submission is imported), so this is the grader's failure, never the agent's.
            self.guard_failed = True
            self._kill()
            self._give_up("network guard could not be installed (%s)" %
                          (payload.decode(errors="replace")[:200] if payload else tag))
            return False
        deadline = min(self.started_at + self.cfg["import_timeout_s"], self.budget_end)
        tag, payload = self._recv(deadline, MAX_TEXT)
        if tag == b"R":
            return True
        self._kill()
        if tag == b"I":
            why = "import failed: " + payload.decode(errors="replace")[:200]
        elif tag == "timeout":
            why = "import did not finish in %ds" % self.cfg["import_timeout_s"] \
                if time.monotonic() < self.budget_end else \
                "timeout: the block's time budget (%ds) was spent" % self.budget
        else:
            why = "import failed: solver process ended before importing (%s)" % tag
        self._give_up(why)
        return False

    def _frames(self):
        """Hand out frames until one ends the process (a timeout or a death) or none is left."""
        while self.remaining:
            name = self.remaining.pop(0)
            message = frame_message(self.sealed, name, self.privileged_run)
            t0 = time.monotonic()
            if t0 >= self.budget_end:
                self.remaining.insert(0, name)
                self._give_up("timeout: the block's time budget (%ds) was spent" % self.budget)
                break
            deadline = min(t0 + self.frame_limit, self.budget_end)
            tag, payload = ("eof", None) if not self._send(message, deadline) else \
                self._recv(deadline, MAX_RESULT)
            if tag == "eof" and time.monotonic() >= deadline:
                tag = "timeout"
            self.elapsed[name] = round(time.monotonic() - t0, 2)
            if tag == b"A":
                with open(os.path.join(RESULTS, name + ".npy"), "wb") as handle:
                    handle.write(payload)
                self.status[name] = "returned"
                continue
            if tag == b"E":
                self.status[name] = "no array: " + payload.decode(errors="replace")[:200]
                continue
            if tag == "timeout":
                self.status[name] = "timeout: no array within %ds (process killed)" % self.frame_limit
            elif tag == "eof":
                try:
                    code = self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    code = "unknown: it closed the channel"
                self.status[name] = "solver process died (exit %s)" % code
            else:
                self.status[name] = "protocol violation (%s); process killed" % (tag,)
            self._kill()
            return
        self._kill()

    def run(self):
        self.budget_end = time.monotonic() + self.budget
        try:
            while self.remaining:
                if not self._import():
                    break
                self._frames()
        except Exception as exc:                           # a grader bug: fail the verify, never score it
            self.error = "%s: %s" % (type(exc).__name__, exc)
            self._give_up("grader error")
        finally:
            self._kill()


def frame_order(names, cfg, privileged_run):
    """The order the frames are handed out in. A submission gets them in one fixed shuffle that mixes the scenes
    (grader_config frame_order_seed), so a block holds a couple of views of many scenes and every call stands on
    its own frame, as at deployment. The privileged reference gets them in scene order, only so that it loads each
    scene's meshes once per block; that changes its run time, not what it is scored on."""
    if privileged_run:
        return list(names)
    order = list(names)
    random.Random(int(cfg["frame_order_seed"])).shuffle(order)
    return order


def run_solvers(names, cfg, logs, sealed, uid, privileged_run):
    n = int(cfg["solver_processes"])
    names = frame_order(names, cfg, privileged_run)
    per = -(-len(names) // n)             # contiguous blocks of the handed-out order
    slots = [Slot(i, names[i * per:(i + 1) * per], cfg, logs, sealed, uid, privileged_run) for i in range(n)]
    t0 = time.time()
    for slot in slots:
        slot.start()
    for slot in slots:
        slot.join()
    status, elapsed = {}, {}
    for slot in slots:
        status.update(slot.status)
        elapsed.update(slot.elapsed)
    errors = ["slot %d: %s" % (s.index, s.error) for s in slots if s.error]
    require(not errors, "grader error while running the solver: " + "; ".join(errors))
    restarts = sum(max(0, s.attempt - 1) for s in slots)
    return status, elapsed, restarts, time.time() - t0, any(s.guard_failed for s in slots)


# --------------------------------------------------------------------------- main
def main():
    cfg = stage_config()
    anchor, meta = read_anchor()
    cap = cfg.get("scoring_cap", 1.5)
    if not os.path.isfile(os.path.join(SUBMISSION, "solver.py")):
        write_verdict(zero_verdict(reason="no /home/user/submission/solver.py",
                                   anchored=anchor is not None))
        return

    account = pwd.getpwnam(agent_user())
    outlived = len(stop_account(account.pw_uid))
    freezer = AccountFreezer(account.pw_uid)
    freezer.start()
    sealed = fetch_hidden(cfg)
    frames = hidden_frames(sealed)
    require(len(frames) == cfg["expected_frames"],
            "hidden manifest lists %d frames, expected %d" % (len(frames), cfg["expected_frames"]))
    make_workdir()
    privileged_run = privileged(cfg)
    export_child(sealed, frames, with_ground_truth=privileged_run)
    logs = os.path.join(stage_dir(), "logs")
    os.makedirs(logs, exist_ok=True)

    names = ["%04d_%04d" % f for f in frames]
    status, elapsed, restarts, predict_wall, guard_failed = run_solvers(
        names, cfg, logs, sealed, account.pw_uid, privileged_run)
    outlived += len(stop_account(account.pw_uid))
    require(not guard_failed, "the network guard could not be installed; see %s" % logs)
    log("prediction finished in %.0fs (%d restarts)" % (predict_wall, restarts))

    out_json = os.path.join(stage_dir(), "scores.json")
    t0 = time.time()
    rc = subprocess.call([GRASP_PY, "-I", os.path.join(stage_dir(), "score_frames.py"),
                          sealed, RESULTS, out_json, str(cfg["score_workers"])],
                         cwd=stage_dir())
    require(rc == 0, "score_frames.py exited %d" % rc)
    scores = json.load(open(out_json))
    score_wall = time.time() - t0
    require(scores["frames"] == len(frames), "scored %d of %d frames" % (scores["frames"], len(frames)))

    rows = scores["per_frame"]
    for row in rows:
        row["seconds"] = elapsed.get(row["frame"])
        if status.get(row["frame"]) != "returned":
            # only arrays received in time are in RESULTS; anything else scores 0 whatever it is
            require(row["ap"] == 0.0 and row["status"] != "ok",
                    "frame %s has no accepted array but was scored" % row["frame"])
        if row["status"] != "ok" or status.get(row["frame"]) != "returned":
            row["detail"] = status.get(row["frame"], "not run")
    ok = sum(1 for r in rows if r["status"] == "ok")
    timeouts = sum(1 for r in rows if str(status.get(r["frame"], "")).startswith("timeout"))
    times = [t for t in elapsed.values() if t is not None]
    print(json.dumps({"mean_ap": scores["mean_ap"], "frames": len(rows), "frames_valid": ok,
                      "per_frame": rows}), flush=True)
    log("mean AP %.3f over %d frames (%d with a valid array, %d timeouts); starting detector %.3f"
        % (scores["mean_ap"], len(rows), ok, timeouts, float(meta["baseline_mean_ap"])))

    baseline = float(meta["baseline_mean_ap"])
    gain = max(0.0, scores["mean_ap"] - baseline)
    write_verdict(ratio_rewards(
        gain, anchor, cap=cap, higher_is_better=True, metric_name="ap_gain",
        full_at=meta.get("full_at", 1.0),
        extra={"mean_ap": scores["mean_ap"], "frames": len(rows), "frames_valid": ok,
               "frames_timeout": timeouts, "solver_restarts": restarts,
               "frame_s_max": round(max(times), 1) if times else 0.0,
               "predict_wall_s": round(predict_wall, 1),
               "score_wall_s": round(score_wall, 1),
               "agent_processes_outlived_kill": outlived,
               "agent_processes_frozen": len(freezer.frozen)}))
    freezer.halt.set()


if __name__ == "__main__":
    main()
