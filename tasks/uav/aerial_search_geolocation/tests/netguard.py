"""netguard - start the agent's controller with no IP networking, loopback included.

Why this exists. The simulator (Unreal Engine + ProjectAirSim) is driven over TCP on the
sandbox's loopback interface, and its API can read the drone's true pose, list scene
objects and teleport the drone. The agent's controller runs at the agent's UID inside the
same sandbox, so a socket to 127.0.0.1 would be a direct line to the ground truth the task
is about. ProjectAirSim's optional client authorization does not close this: the server
keeps one global "authorized until" time, so once the grader authorizes, every client is
authorized. The verify sandbox also has no NET_ADMIN, so iptables owner rules are out.

What it does. Before exec'ing the controller it installs a seccomp filter (no privilege
needed: PR_SET_NO_NEW_PRIVS first) that makes `socket(AF_INET|AF_INET6, ...)` fail with
EACCES, refuses `io_uring_setup` (io_uring can open sockets without the socket syscall),
and refuses every syscall made through a foreign ABI (x32, or i386 `int 0x80`, where
`socketcall` multiplexes `socket`). A seccomp filter is inherited by every child and can
never be removed, so nothing the controller starts can reach the simulator either.
Unix-domain sockets, pipes, files and the GPU are untouched.

Usage (as the account that will run the controller):

    /opt/venv/bin/python3 -I -S netguard.py exec <dir> <program> [args...]
    /opt/venv/bin/python3 -I -S netguard.py selftest

`-I -S` matter: an isolated interpreter ignores PYTHON* variables and the user's
site-packages, so nothing the agent left in its home runs before the filter is in place.

stdlib only, 3.8-compatible syntax; byte-identical in the grading stage and the agent payload.
"""

import ctypes
import errno
import os
import platform
import socket
import struct
import sys

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2

SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000

# BPF opcodes
LD_W_ABS = 0x20     # BPF_LD | BPF_W | BPF_ABS
JEQ_K = 0x15        # BPF_JMP | BPF_JEQ | BPF_K
JGE_K = 0x35        # BPF_JMP | BPF_JGE | BPF_K
RET_K = 0x06        # BPF_RET | BPF_K

# struct seccomp_data offsets
OFF_NR = 0
OFF_ARCH = 4
OFF_ARG0_LO = 16    # little-endian: the low 32 bits of args[0]

AF_INET = 2
AF_INET6 = 10

ARCHES = {
    # machine: (AUDIT_ARCH, __NR_socket, __NR_io_uring_setup)
    "x86_64": (0xC000003E, 41, 425),
    "aarch64": (0xC00000B7, 198, 425),
}
X32_SYSCALL_BIT = 0x40000000


def _program():
    machine = platform.machine()
    if machine not in ARCHES:
        raise SystemExit("netguard: unsupported architecture {!r}".format(machine))
    arch, nr_socket, nr_io_uring_setup = ARCHES[machine]
    deny = SECCOMP_RET_ERRNO | errno.EACCES
    # (code, jt, jf, k); jt/jf are offsets from the NEXT instruction
    return [
        (LD_W_ABS, 0, 0, OFF_ARCH),            # 0  A = arch
        (JEQ_K, 1, 0, arch),                   # 1  native ABI? -> 3
        (RET_K, 0, 0, deny),                   # 2  foreign ABI (i386 socketcall): deny
        (LD_W_ABS, 0, 0, OFF_NR),              # 3  A = syscall number
        (JGE_K, 0, 1, X32_SYSCALL_BIT),        # 4  x32 ABI? -> 5 else 6
        (RET_K, 0, 0, deny),                   # 5
        (JEQ_K, 3, 0, nr_socket),              # 6  socket -> 10
        (JEQ_K, 0, 1, nr_io_uring_setup),      # 7  io_uring_setup -> 8 else 9
        (RET_K, 0, 0, deny),                   # 8
        (RET_K, 0, 0, SECCOMP_RET_ALLOW),      # 9
        (LD_W_ABS, 0, 0, OFF_ARG0_LO),         # 10 A = domain
        (JEQ_K, 2, 0, AF_INET),                # 11 -> 14
        (JEQ_K, 1, 0, AF_INET6),               # 12 -> 14
        (RET_K, 0, 0, SECCOMP_RET_ALLOW),      # 13 AF_UNIX and friends
        (RET_K, 0, 0, deny),                   # 14
    ]


def install():
    """Install the filter on the calling thread's process. Irreversible."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                           ctypes.c_ulong, ctypes.c_ulong]
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    program = _program()
    raw = b"".join(struct.pack("<HBBI", c, jt, jf, k) for c, jt, jf, k in program)
    buf = ctypes.create_string_buffer(raw, len(raw))

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = SockFprog(len(program), ctypes.addressof(buf))
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p,
                           ctypes.c_ulong, ctypes.c_ulong]
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP) failed")


def _blocked(family):
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
    except OSError as exc:
        return exc.errno == errno.EACCES
    sock.close()
    return False


def selftest():
    """Fork, guard the child, and check what it can and cannot open. Exit 0 iff sound."""
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(read_end)
        try:
            install()
            unix_ok = True
            try:
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close()
            except OSError:
                unix_ok = False
            report = "{}{}{}".format(int(_blocked(socket.AF_INET)),
                                     int(_blocked(socket.AF_INET6)), int(unix_ok))
        except Exception as exc:  # noqa: BLE001
            report = "ERR " + str(exc)
        os.write(write_end, report.encode())
        os._exit(0)
    os.close(write_end)
    report = os.read(read_end, 256).decode()
    os.waitpid(pid, 0)
    sound = report == "111"
    print("netguard selftest: {} (inet blocked, inet6 blocked, unix allowed = {})".format(
        "ok" if sound else "FAILED", report), file=sys.stderr)
    return 0 if sound else 1


def main(argv):
    if len(argv) >= 2 and argv[1] == "selftest":
        return selftest()
    if len(argv) >= 4 and argv[1] == "exec":
        workdir, program = argv[2], argv[3:]
        install()
        os.chdir(workdir)
        os.execvp(program[0], program)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
