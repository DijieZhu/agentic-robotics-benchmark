"""Run-folder and process helpers of the runner (grader side, standard library only):

  * the run folder holds program/ (a read-only copy of the submission's controller folder: regular files and folders
    only; symbolic links, special files and __pycache__ dropped), lib/ (the client alone) and empty home/ and tmp/;
  * child_env passes PATH, HOME, TMPDIR, LANG, the thread pins and the numerics pins only (nothing else of the
    grader's environment);
  * there is no kill by account: the runner stops the controller only through the process group it recorded when it
    started it (kill_group), and the controller cannot leave that group (the client's system-call filter refuses
    setsid and setpgid);
  * when the runner runs as root, the controller's process runs under the account named in the runner's
    configuration (account), with that account's group and no supplementary groups, and the run folder is prepared
    for it (open_run_dir_for); when the runner is not root none of this is done: the controller runs under the
    runner's own account.
"""
import os
import shutil
import signal
import stat

PINS = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "OPENBLAS_L2_SIZE")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def copy_regular_files(src, dst):
    """copy the tree src to dst keeping regular files and folders only; returns the number of skipped entries."""
    skipped = 0
    os.makedirs(dst, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        rel = os.path.relpath(dirpath, src)
        target = dst if rel == "." else os.path.join(dst, rel)
        keep = []
        for d in dirnames:
            p = os.path.join(dirpath, d)
            if d == "__pycache__" or os.path.islink(p):
                skipped += 1 if os.path.islink(p) else 0
                continue
            os.makedirs(os.path.join(target, d), exist_ok=True)
            keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            p = os.path.join(dirpath, f)
            if os.path.islink(p) or not stat.S_ISREG(os.lstat(p).st_mode):
                skipped += 1
                continue
            shutil.copyfile(p, os.path.join(target, f))
    return skipped


def chmod_tree(root, dir_mode, file_mode):
    os.chmod(root, dir_mode)
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            os.chmod(os.path.join(dirpath, d), dir_mode)
        for f in filenames:
            os.chmod(os.path.join(dirpath, f), file_mode)


def prepare_run_dir(root, program_dir, client_path, client_name):
    """lay out one run under root: program/ (read-only copy), lib/<client_name> (read-only), home/ and tmp/ (empty,
    owner-only). Returns the paths."""
    os.makedirs(root, exist_ok=True)
    os.chmod(root, 0o755)
    program = os.path.join(root, "program")
    skipped = copy_regular_files(program_dir, program)
    chmod_tree(program, 0o555, 0o444)
    lib = os.path.join(root, "lib")
    os.makedirs(lib, exist_ok=True)
    shutil.copyfile(client_path, os.path.join(lib, client_name))
    chmod_tree(lib, 0o555, 0o444)
    out = {"root": root, "program": program, "lib": lib, "skipped": skipped}
    for name in ("home", "tmp"):
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o700)
        out[name] = d
    return out


def child_env(home, tmpdir):
    """a minimal environment: PATH, a fresh HOME and TMPDIR, the thread pins (one thread each) and the house's numerics
    pins when the grader has them set."""
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": home, "TMPDIR": tmpdir,
           "TMP": tmpdir, "TEMP": tmpdir, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    for k in THREAD_VARS:
        env[k] = "1"
    for k in PINS:
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def account(name):
    """(name, uid, gid) of the account the controller's process runs under when the runner is root, read from the
    system's account list; None when no such account exists or when its uid or its group is 0 (the controller is
    never started as root)."""
    import pwd
    try:
        pw = pwd.getpwnam(name)
    except (KeyError, TypeError, ValueError):
        return None
    if pw.pw_uid == 0 or pw.pw_gid == 0:
        return None
    return pw.pw_name, int(pw.pw_uid), int(pw.pw_gid)


def _searchable(st, uid, gid):
    """whether the account (uid, gid, no supplementary groups) may search the folder whose stat is st."""
    if st.st_uid == uid:
        return bool(st.st_mode & stat.S_IXUSR)
    if st.st_gid == gid:
        return bool(st.st_mode & stat.S_IXGRP)
    return bool(st.st_mode & stat.S_IXOTH)


def open_run_dir_for(rd, uid, gid):
    """(the runner runs as root) prepare the run folder rd (prepare_run_dir's paths) for the account (uid, gid):
    home/ and tmp/ become its own (owner-only); program/ and lib/ stay the runner's read-only copies, readable and not
    writable by it; the run folder may be searched by it but not listed; the folder that holds the run folder (the
    grader's work folder) gains the search bit for others when the account cannot search it, never the read bit, and
    every other entry of that folder is closed to the account; every folder above it must already be searchable by the
    account. Returns the problems found (a folder above that the account cannot search, an entry of the work folder
    that the account owns): a start with a problem cannot go on."""
    problems = []
    for name in ("home", "tmp"):
        os.chown(rd[name], uid, gid)
        os.chmod(rd[name], stat.S_IRWXU)
    root = os.path.abspath(rd["root"])
    os.chmod(root, stat.S_IRWXU | stat.S_IXGRP | stat.S_IXOTH)
    work = os.path.dirname(root)
    st = os.stat(work)
    if not _searchable(st, uid, gid):
        if st.st_uid == os.geteuid():
            os.chmod(work, stat.S_IMODE(st.st_mode) | stat.S_IXOTH)
        else:
            problems.append(work)
    d = os.path.dirname(work)
    while True:
        if not _searchable(os.stat(d), uid, gid):
            problems.append(d)
        up = os.path.dirname(d)
        if up == d:
            break
        d = up
    for name in os.listdir(work):
        p = os.path.join(work, name)
        if p == root:
            continue
        st = os.lstat(p)
        if stat.S_ISLNK(st.st_mode):
            continue
        if st.st_uid == uid:
            problems.append(p)
            continue
        mode = stat.S_IMODE(st.st_mode)
        closed = mode & ~stat.S_IRWXO
        if st.st_gid == gid:
            closed &= ~stat.S_IRWXG
        if closed != mode:
            os.chmod(p, closed)
    return problems


def unlock_tree(root):
    """make a run folder removable again (the copies were made read-only)."""
    for dirpath, dirnames, filenames in os.walk(root):
        try:
            os.chmod(dirpath, 0o755)
        except OSError:
            pass


def kill_group(pgid):
    """SIGKILL the process group pgid, which the runner recorded when it started the controller (the controller's
    interpreter leads it). Refuses anything that is not such a group: an id that is not an integer above 1 (0 would
    name the caller's own group, 1 the system's first process), or the grader's own group. Inside a container the
    controller's ids are small: any id above 1 that the runner recorded is ended."""
    if not isinstance(pgid, int) or isinstance(pgid, bool) or pgid <= 1:
        return False
    if pgid == os.getpgrp():
        return False
    try:
        os.killpg(pgid, signal.SIGKILL)
        return True
    except (OSError, ProcessLookupError):
        return False
