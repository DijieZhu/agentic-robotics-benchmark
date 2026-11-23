"""Nothing of one graded mission reaches the next (grader side, stdlib only).

The eight missions run in one sandbox under one account. A fresh HOME and TMPDIR do not stop a program from
leaving a file in /tmp, /dev/shm, a lock directory or its real home, or a System V / POSIX IPC object, and
reading it on the next site: early sites would become trials for later ones. After every mission, once the
account's processes are gone, the grader therefore

  * removes every file, link and (emptied) directory the account owns on every writable mount,
  * removes the account's System V shared memory, semaphores and message queues,
  * restores every regular file the account can write but does not own to its content at the start of
    verification (listed once, as the account, before the first mission).
"""
import ctypes
import os
import stat
import subprocess

SKIP_FS = {"proc", "sysfs", "cgroup", "cgroup2", "devpts", "securityfs", "debugfs", "tracefs", "pstore", "bpf",
           "fusectl", "configfs", "hugetlbfs", "nsfs", "binfmt_misc", "autofs", "rpc_pipefs", "selinuxfs"}
RESTORE_LIMIT = 4 << 20


def mounts():
    """Mount points that can hold a file, longest first; one entry per device."""
    seen, out = set(), []
    with open("/proc/mounts") as f:
        rows = [line.split() for line in f]
    for _, point, kind, options, *_ in sorted(rows, key=lambda r: -len(r[1])):
        point = point.replace("\\040", " ")
        if kind in SKIP_FS or "ro" in options.split(",") and kind != "overlay":
            continue
        try:
            dev = os.lstat(point).st_dev
        except OSError:
            continue
        if stat.S_ISDIR(os.lstat(point).st_mode) and dev not in seen:
            seen.add(dev)
            out.append(point)
    return out


def _walk(top):
    """Every entry under `top` on its own device, children before their directory, no link followed."""
    try:
        dev = os.lstat(top).st_dev
    except OSError:
        return
    stack = [(top, False)]
    while stack:
        path, visited = stack.pop()
        if visited:
            yield path, True
            continue
        try:
            entries = list(os.scandir(path))
        except OSError:
            continue
        stack.append((path, True))
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(info.st_mode):
                if info.st_dev == dev:
                    stack.append((entry.path, False))
            else:
                yield entry.path, False


def purge_files(uid, keep=(), keep_dirs=()):
    """Remove what the account owns. `keep`: paths left alone with everything under them; `keep_dirs`:
    directories that stay (emptied). Returns the count."""
    keep = tuple(os.path.normpath(k) for k in keep)
    points = mounts()
    removed = 0
    for mount in points:
        for path, is_dir in _walk(mount):
            if any(path == k or path.startswith(k+os.sep) for k in keep) or path in points or path in keep_dirs:
                continue
            try:
                if os.lstat(path).st_uid != uid:
                    continue
                os.rmdir(path) if is_dir else os.unlink(path)
                removed += 1
            except OSError:
                continue          # a directory that still holds another account's files stays
    return removed


def purge_ipc(uid):
    """IPC_RMID on every System V object of the account. Returns the count."""
    libc = ctypes.CDLL(None, use_errno=True)
    removed = 0
    for name, column, call in (("shm", 7, lambda i: libc.shmctl(i, 0, None)),
                               ("sem", 4, lambda i: libc.semctl(i, 0, 0)),
                               ("msg", 7, lambda i: libc.msgctl(i, 0, None))):
        try:
            with open("/proc/sysvipc/"+name) as f:
                rows = [line.split() for line in f][1:]
        except OSError:
            continue
        for row in rows:
            try:
                if int(row[column]) == uid and call(int(row[1])) == 0:
                    removed += 1
            except (ValueError, IndexError):
                continue
    return removed


def writable_snapshot(as_account, skip=()):
    """Regular files the account can write and does not own, with their bytes: {path: (size, mtime_ns, data)}.
    `as_account(argv)` returns the argv that runs a command as the account."""
    argv = as_account(["/usr/bin/find", *mounts(), "-xdev", "-type", "f", "-writable", "!", "-user",
                       str(as_account.uid), "-print0"])
    out = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=600).stdout
    snapshot = {}
    for raw in out.split(b"\0"):
        path = os.fsdecode(raw)
        if not path or any(path == s or path.startswith(s+os.sep) for s in skip):
            continue
        try:
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode):
                continue
            data = None
            if info.st_size <= RESTORE_LIMIT:
                with open(path, "rb") as f:
                    data = f.read()
            snapshot[path] = (info.st_size, info.st_mtime_ns, data)
        except OSError:
            continue
    return snapshot


def restore(snapshot):
    """Put back every snapshot file that changed. Returns (restored, unrestorable paths)."""
    restored, failed = 0, []
    for path, (size, mtime_ns, data) in snapshot.items():
        try:
            info = os.lstat(path)
            if info.st_size == size and info.st_mtime_ns == mtime_ns:
                continue
        except OSError:
            info = None
        if data is None:
            failed.append(path)
            continue
        try:
            with open(path, "wb") as f:
                f.write(data)
            os.utime(path, ns=(mtime_ns, mtime_ns))
            restored += 1
        except OSError:
            failed.append(path)
    return restored, failed
