import os
import pwd
import shutil
import signal
import stat
import time

PYTHON_FLAGS = ["-B", "-E", "-s"]


def wrap_limits(argv, cpu_s=None, user=None):
    wrapped = list(argv)
    if cpu_s is not None and shutil.which("prlimit"):
        wrapped = ["prlimit", "--cpu={}".format(int(cpu_s)), "--core=0", "--"] + wrapped
    if user and os.getuid() == 0 and shutil.which("runuser"):
        wrapped = ["runuser", "-p", "-u", user, "--"] + wrapped
    return wrapped


def child_env(home, tmpdir, lib_dir, given_dir, threads=2, mujoco_gl=None, extra=None):
    env = {"PATH": os.environ.get("PATH", "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"),
           "HOME": home, "TMPDIR": tmpdir, "TMP": tmpdir, "TEMP": tmpdir, "LANG": "C.UTF-8", "MPLBACKEND": "Agg",
           "PYTHONPATH": lib_dir, "ARB_GIVEN_DIR": given_dir,
           "MUJOCO_GL": mujoco_gl or os.environ.get("MUJOCO_GL", "osmesa"), "PYOPENGL_PLATFORM": "osmesa",
           "OMP_NUM_THREADS": str(int(threads)), "OPENBLAS_NUM_THREADS": str(int(threads)),
           "MKL_NUM_THREADS": str(int(threads))}
    if extra:
        env.update(extra)
    return env


def runner_account(preferred="nobody", fallback=None):
    if os.getuid() != 0:
        return None
    for name in [n for n in (preferred, fallback) if n]:
        try:
            pw = pwd.getpwnam(name)
            return name, pw.pw_uid, pw.pw_gid
        except KeyError:
            continue
    return None


def copy_regular_files(src, dst):
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


def prepare_run_dir(root, program_dir, lib_dirs, owner=None, private=False):
    os.makedirs(root, exist_ok=True)
    os.chmod(root, 0o755)
    program = os.path.join(root, "program")
    skipped = copy_regular_files(program_dir, program)
    chmod_tree(program, 0o700 if private else 0o755, 0o600 if private else 0o644)
    lib = os.path.join(root, "lib")
    os.makedirs(lib, exist_ok=True)
    for name, path in lib_dirs.items():
        copy_regular_files(path, os.path.join(lib, name))
    chmod_tree(lib, 0o755, 0o644)
    out = {"root": root, "program": program, "lib": lib, "skipped": skipped}
    for name in ("home", "tmp"):
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o700)
        if owner is not None:
            os.chown(d, owner[0], owner[1])
        out[name] = d
    return out


def uid_processes(uid):
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/{}/status".format(name)) as fh:
                for line in fh:
                    if line.startswith("Uid:"):
                        if int(line.split()[1]) == int(uid):
                            pids.append(int(name))
                        break
        except (OSError, ValueError):
            continue
    return pids


def kill_uid_processes(uid, grace_s=5.0):
    if os.getuid() != 0 or int(uid) in (0, os.getuid()) or int(uid) < 1:
        raise RuntimeError("kill_uid_processes: only as root, for a dedicated runner account")
    killed = 0
    seen = set()
    deadline = time.time() + grace_s
    while True:
        pids = uid_processes(uid)
        for pid in pids:
            if pid not in seen:
                seen.add(pid)
                killed += 1
            if pid > 1:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        if not pids or time.time() > deadline:
            return killed, len(pids)
        time.sleep(0.1)


def _stat_fields(pid):
    with open("/proc/{}/stat".format(pid)) as fh:
        st = fh.read()
    return st[st.rindex(")") + 2:].split()


def descendants(root_pid, sid):
    ppid, sess = {}, {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            f = _stat_fields(name)
            ppid[int(name)] = int(f[1])
            sess[int(name)] = int(f[3])
        except (OSError, ValueError, IndexError):
            continue
    out = set(p for p, s in sess.items() if s == int(sid))
    tree = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for p, pp in ppid.items():
            if pp in tree and p not in tree:
                tree.add(p)
                changed = True
    out |= tree
    out.discard(int(root_pid))
    return sorted(out)
