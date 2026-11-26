"""Verify-start lockdown of the six-servo assembly grader (CHAIN_PLAN 4.0 stage lock, hardened after the 2026-09-27
adversarial review). stdlib only: verify.py runs it BEFORE numpy / mujoco are imported.

The review demonstrated, inside the real image, a background process left by the agent (its own uid, started with
`nohup` during the agent phase) that polls for `/opt/ale/verify/anchor/*` and copies the reference program into the
still-writable `/home/user/submission` WHILE the engine uploads the verify stage -- i.e. before the first line of
verify.py runs, so no lock inside the grader can be early enough. The copy scored 1.0. Nothing in the engine kills the
agent account's processes between the agent phase and verify (Q13), and a byte comparison with the anchor cannot be a
gate because `ale validate`'s oracle phase legitimately stages the very same files.

What closes it (every function below is used by verify.py's `_early_lockdown`, run as root first thing):

  * `stage_upload_time(stage)`: the earliest ctime of a regular file under the stage. `docker cp` preserves mtimes
    but ctime is set by the kernel at extraction and the engine does not chown the verify stage (FRAMEWORK identity),
    so this is the instant the engine began uploading it -- after the agent phase ended, before anything the agent
    left running could see the anchor. ctime cannot be set from user space.
  * `late_entries(root, t_ref)`: every entry under `root` whose ctime or mtime is at or after `t_ref`, i.e. written,
    renamed, linked or re-permissioned after the agent phase ended. In the submission they zero the contract gate
    (the deliverable must be final when the session ends); elsewhere in the agent home and the world-writable sticky
    directories they are `quarantine`d into a root-only folder so the deprivileged program cannot read them (a poller
    that stashes the anchor in /tmp or ~/x for the program to exec at run time).
  * `stop_and_kill_uid(uid)`: SIGSTOP every live process of the uid until the set is stable (a stopped process cannot
    fork), then SIGKILL, until none is left (zombies excluded: a container without an init never reaps them). Used for
    the agent account at verify start (a survivor voids the run: the regime is compromised) and for the runner account
    after every episode (the review's 'denominator attack': a forking survivor of the 5 s SIGKILL loop hogging the
    CPUs while the anchor runs).
  * `snapshot_dir(src, dst)`: a root-owned 0700 copy of the submission (regular files only) taken before any agent
    code runs; every later read of the submission uses it.
  * `identical_files(a, b)`: how many regular files under `a` are byte-identical (or, for .py, AST-identical) to a file
    under `b` -- a STAT (`anchor_identical_files`), never a gate (see above).
"""

import ast
import hashlib
import os
import shutil
import signal
import stat
import time

__all__ = ["live_uid_processes", "stop_and_kill_uid", "stage_upload_time", "late_entries", "quarantine",
           "snapshot_dir", "identical_files", "process_age_ms"]


def live_uid_processes(uid):
    """PIDs of every live (non-zombie) process owned by `uid` (Linux /proc)."""
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/{}/status".format(name)) as fh:
                u, state = None, None
                for line in fh:
                    if line.startswith("Uid:"):
                        u = int(line.split()[1])
                    elif line.startswith("State:"):
                        state = line.split()[1]
                    if u is not None and state is not None:
                        break
            if u == int(uid) and not (state or "").startswith("Z"):
                pids.append(int(name))
        except (OSError, ValueError, IndexError):
            continue
    return pids


def _signal_all(pids, sig):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def stop_and_kill_uid(uid, grace_s=5.0, settle_s=0.05):
    """Freeze, then kill, every live process of `uid`. Returns (killed, remaining).

    Loop: list -> SIGSTOP all -> list again; when no new pid appeared (a stopped process cannot fork; a child forked
    between the two scans is caught by the next one) SIGKILL them all and re-list. Repeats until nothing live is left
    or `grace_s` elapsed."""
    seen = set()
    deadline = time.time() + float(grace_s)
    remaining = []
    while True:
        pids = live_uid_processes(uid)
        if not pids:
            return len(seen), 0
        seen.update(pids)
        _signal_all(pids, signal.SIGSTOP)
        again = live_uid_processes(uid)
        if set(again) - set(pids):
            _signal_all(again, signal.SIGSTOP)      # newcomers frozen too; scan once more before killing
            seen.update(again)
            if time.time() > deadline:
                _signal_all(again, signal.SIGKILL)
                time.sleep(settle_s)
                remaining = live_uid_processes(uid)
                return len(seen), len(remaining)
            continue
        _signal_all(again, signal.SIGKILL)
        time.sleep(settle_s)
        remaining = live_uid_processes(uid)
        if not remaining:
            return len(seen), 0
        if time.time() > deadline:
            return len(seen), len(remaining)


def stage_upload_time(stage_dir):
    """Earliest ctime (s) of a regular file under `stage_dir`, or None. Call BEFORE anything chmods the files."""
    t = None
    for dirpath, _, filenames in os.walk(stage_dir):
        for fn in filenames:
            try:
                st = os.lstat(os.path.join(dirpath, fn))
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode) and (t is None or st.st_ctime < t):
                t = st.st_ctime
    return t


def late_entries(root, t_ref, exclude=(), skip_uids=(0,)):
    """Entries under `root` (files, dirs, links; `root` itself excluded) whose ctime or mtime >= t_ref, as a sorted list
    of (relpath, kind, ctime, mtime). Paths under `exclude` are skipped, and so are entries owned by a uid in `skip_uids`
    (root by default: the agent account cannot create root-owned entries, and the engine's own sockets / pid files in
    /tmp must stay where they are); the walk still descends into skipped directories."""
    out = []
    if t_ref is None or not os.path.isdir(root):
        return out
    ex = tuple(os.path.abspath(e) for e in exclude)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if any(os.path.abspath(dirpath).startswith(e) for e in ex):
            dirnames[:] = []
            continue
        for name in filenames + dirnames:
            p = os.path.join(dirpath, name)
            if any(os.path.abspath(p).startswith(e) for e in ex):
                continue
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if st.st_uid in skip_uids:
                continue
            if max(st.st_ctime, st.st_mtime) >= t_ref:
                kind = "dir" if stat.S_ISDIR(st.st_mode) else ("link" if stat.S_ISLNK(st.st_mode) else "file")
                out.append((os.path.relpath(p, root), kind, st.st_ctime, st.st_mtime))
    return sorted(out)


def quarantine(root, entries, qdir):
    """Move the listed regular files and links (relpaths under `root`) into `qdir` (created 0700). Directories are left in
    place: a directory is 'late' because an entry was added or removed after the reference time, and the added entries
    are late themselves (moved here), so moving the directory would only take old content with it (the agent's whole
    ~/given for one stashed file). Returns the number moved."""
    moved = 0
    if not entries:
        return moved
    os.makedirs(qdir, exist_ok=True)
    os.chmod(qdir, 0o700)
    done = []
    for rel, kind, _, _ in sorted(entries, key=lambda e: e[0].count(os.sep)):
        if kind == "dir" or any(rel == d or rel.startswith(d + os.sep) for d in done):
            continue
        src = os.path.join(root, rel)
        dst = os.path.join(qdir, root.strip(os.sep).replace(os.sep, "_") + "__" + rel.replace(os.sep, "_"))
        try:
            if os.path.lexists(dst):
                dst += ".%d" % moved
            shutil.move(src, dst)
            moved += 1
            done.append(rel)
        except (OSError, shutil.Error):
            try:                                        # last resort: make it unreadable where it is
                os.chmod(src, 0)
                moved += 1
                done.append(rel)
            except OSError:
                pass
    return moved


def snapshot_dir(src, dst):
    """Root-owned 0700 / 0600 copy of `src` (regular files and directories only; symbolic links, special files and
    __pycache__ dropped). Returns the number of skipped entries."""
    from arm.assembly_cell.runner import chmod_tree, copy_regular_files
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    skipped = copy_regular_files(src, dst)
    chmod_tree(dst, 0o700, 0o600)
    return skipped


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _ast_digest(p):
    try:
        src = open(p, encoding="utf-8", errors="replace").read()
        return hashlib.sha256(ast.dump(ast.parse(src)).encode()).hexdigest()
    except (SyntaxError, ValueError, OSError):
        return None


def identical_files(a_dir, b_dir):
    """(byte_identical, ast_identical): how many regular files under `a_dir` equal (sha256) some regular file under
    `b_dir`, and how many .py files under `a_dir` have the same AST (comments / formatting stripped) as one there."""
    b_sha, b_ast = set(), set()
    for dirpath, _, fns in os.walk(b_dir):
        for fn in fns:
            p = os.path.join(dirpath, fn)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            b_sha.add(_sha(p))
            if fn.endswith(".py"):
                d = _ast_digest(p)
                if d:
                    b_ast.add(d)
    n_sha = n_ast = 0
    for dirpath, _, fns in os.walk(a_dir):
        for fn in fns:
            p = os.path.join(dirpath, fn)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            if _sha(p) in b_sha:
                n_sha += 1
            elif fn.endswith(".py") and _ast_digest(p) in b_ast:
                n_ast += 1
    return n_sha, n_ast


def process_age_ms():
    """Age of the calling process in ms (how long the pre-lock window inside this process lasted), or -1."""
    try:
        with open("/proc/self/stat") as fh:
            st = fh.read()
        start_ticks = float(st[st.rindex(")") + 2:].split()[19])
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
        return max(0.0, (uptime - start_ticks / os.sysconf("SC_CLK_TCK")) * 1000.0)
    except (OSError, ValueError, IndexError):
        return -1.0
