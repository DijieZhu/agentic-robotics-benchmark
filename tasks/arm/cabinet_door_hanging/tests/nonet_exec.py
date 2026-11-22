"""Launch the agent's solver without network access.

Run by the grader as root:

    python3 -I nonet_exec.py --uid U --gid G --cwd DIR -- bash run.sh

In this process, before exec, it:
  1. sets PR_SET_NO_NEW_PRIVS;
  2. raises its own oom_score_adj to 1000 (the solver is the first thing the kernel kills
     under memory pressure, never the grader) and sets RLIMIT_CORE to 0;
  3. installs a seccomp filter (native syscall ABI only) under which socket() and
     socketpair() succeed only for AF_UNIX and fail with EPERM for every other address
     family, io_uring_setup fails, and any non-native syscall ABI (i386, x32) fails;
  4. drops to the agent account (setgroups([]), setgid, setuid);
  5. execs the command with a fixed whitelist environment (PATH, HOME, LANG only).

The filter is inherited by every descendant and cannot be removed. No capability is needed.
"""

import argparse
import ctypes
import os
import platform
import resource
import struct
import sys

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
EPERM = 1
AF_UNIX = 1
# arch -> (AUDIT_ARCH, socket, socketpair, io_uring_setup, x32 bit)
ARCH = {
    "x86_64": (0xC000003E, 41, 53, 425, 0x40000000),
    "aarch64": (0xC00000B7, 198, 199, 425, 0),
}
LD_W_ABS, JEQ_K, JGE_K, RET_K = 0x20, 0x15, 0x35, 0x06
OFF_NR, OFF_ARCH, OFF_ARG0 = 0, 4, 16
WHITELIST_ENV = {"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/home/user", "LANG": "C.UTF-8"}


def _ins(code, jt, jf, k):
    return struct.pack("HBBI", code, jt, jf, k)


def build_filter(machine):
    audit, nr_socket, nr_socketpair, nr_uring, x32 = ARCH[machine]
    deny = SECCOMP_RET_ERRNO | EPERM
    prog = [_ins(LD_W_ABS, 0, 0, OFF_ARCH),
            _ins(JEQ_K, 1, 0, audit),
            _ins(RET_K, 0, 0, deny),
            _ins(LD_W_ABS, 0, 0, OFF_NR)]
    if x32:
        prog += [_ins(JGE_K, 0, 1, x32), _ins(RET_K, 0, 0, deny)]
    prog += [_ins(JEQ_K, 0, 1, nr_uring), _ins(RET_K, 0, 0, deny),
             _ins(JEQ_K, 2, 0, nr_socket),
             _ins(JEQ_K, 1, 0, nr_socketpair),
             _ins(RET_K, 0, 0, SECCOMP_RET_ALLOW),
             _ins(LD_W_ABS, 0, 0, OFF_ARG0),
             _ins(JEQ_K, 0, 1, AF_UNIX),
             _ins(RET_K, 0, 0, SECCOMP_RET_ALLOW),
             _ins(RET_K, 0, 0, deny)]
    return b"".join(prog)


def install_filter():
    machine = platform.machine()
    if machine not in ARCH:
        raise OSError("unsupported architecture " + machine)
    prog = build_filter(machine)
    buf = ctypes.create_string_buffer(prog, len(prog))

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = SockFprog(len(prog) // 8, ctypes.cast(buf, ctypes.c_void_p))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_SECCOMP failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", type=int, default=None)
    ap.add_argument("--gid", type=int, default=None)
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--rlimit-nproc", type=int, default=0, help="0 = leave unchanged (the limit is per host UID and "
                    "would be shared by concurrent sandboxes)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit(2)
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if a.rlimit_nproc > 0:
        hard = resource.getrlimit(resource.RLIMIT_NPROC)[1]
        lim = a.rlimit_nproc if hard == resource.RLIM_INFINITY else min(a.rlimit_nproc, hard)
        resource.setrlimit(resource.RLIMIT_NPROC, (lim, lim))
    install_filter()
    if os.getuid() == 0:
        if a.gid is not None:
            os.setgroups([])
            os.setgid(a.gid)
        if a.uid is not None:
            os.setuid(a.uid)
    elif a.uid is not None and a.uid != os.getuid():
        sys.exit("nonet_exec: cannot change user without root")
    if a.cwd:
        os.chdir(a.cwd)
    os.execvpe(cmd[0], cmd, dict(WHITELIST_ENV))


if __name__ == "__main__":
    main()
