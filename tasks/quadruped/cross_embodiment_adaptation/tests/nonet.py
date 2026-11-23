"""No network for the agent's mission program during verify (grader side, stdlib only).

The verify stage runs with open egress (the engine opens it before tests/test.sh). Every process of
the agent's program is started through `NONET_SRC`: a small interpreter, already running as the agent
account, that lowers its own OOM priority (the kernel kills the program before the robot service or
the grader), drops core dumps, caps file sizes, sets PR_SET_NO_NEW_PRIVS and installs a seccomp
filter under which socket() and socketpair() succeed only for AF_UNIX (every other address family
fails with EPERM), io_uring_setup fails and any non-native syscall ABI (i386, x32) fails, and then
execs the program. The filter is inherited by every descendant and cannot be removed. The robot
service is a Unix socket, so no honest program needs an inet socket. `selftest()` proves, once per
verify, that the filter installs on this kernel and denies an inet socket. Ported from the pump skid
task's cellkit/nonet.py.
"""
import subprocess
import sys

__all__ = ["NONET_SRC", "nonet_argv", "selftest"]

# Run as `python3 -I -B -c NONET_SRC -- <argv...>`: installs the limits and the filter, then execs argv.
NONET_SRC = r'''
import ctypes, os, platform, resource, struct, sys
ARCH = {"x86_64": (0xC000003E, 41, 53, 425, 0x40000000), "aarch64": (0xC00000B7, 198, 199, 425, 0)}
def ins(code, jt, jf, k):
    return struct.pack("HBBI", code, jt, jf, k)
def main():
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 31, 1 << 31))
    audit, nr_socket, nr_socketpair, nr_uring, x32 = ARCH[platform.machine()]
    LD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06
    allow, deny = 0x7FFF0000, 0x00050000 | 1
    prog = [ins(LD, 0, 0, 4), ins(JEQ, 1, 0, audit), ins(RET, 0, 0, deny), ins(LD, 0, 0, 0)]
    if x32:
        prog += [ins(JGE, 0, 1, x32), ins(RET, 0, 0, deny)]
    prog += [ins(JEQ, 0, 1, nr_uring), ins(RET, 0, 0, deny), ins(JEQ, 2, 0, nr_socket), ins(JEQ, 1, 0, nr_socketpair),
             ins(RET, 0, 0, allow), ins(LD, 0, 0, 16), ins(JEQ, 0, 1, 1), ins(RET, 0, 0, allow), ins(RET, 0, 0, deny)]
    raw = b"".join(prog)
    buf = ctypes.create_string_buffer(raw, len(raw))
    class Fprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
    fp = Fprog(len(raw) // 8, ctypes.cast(buf, ctypes.c_void_p))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(22, 2, ctypes.byref(fp), 0, 0) != 0:
        sys.stderr.write("[nonet] the seccomp filter could not be installed (errno %d)\n" % ctypes.get_errno())
        os._exit(97)
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    os.execv(argv[0], argv)
main()
'''


def nonet_argv(argv, python):
    """argv run under the no-network filter, by `python` in isolated mode (no bytecode, no user site)."""
    return [python, "-I", "-B", "-c", NONET_SRC, "--"] + list(argv)


_PROBE = ("import socket, sys\n"
          "try:\n"
          "    socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
          "except PermissionError:\n"
          "    pass\n"
          "else:\n"
          "    sys.exit(3)\n"
          "a, b = socket.socketpair()\n"
          "sys.exit(0)\n")


def selftest(wrap, python, timeout=60.0):
    """(ok, detail): an inet socket is denied and an AF_UNIX pair still works under the filter, launched
    the way the program is (`wrap`: the caller's account switch and environment)."""
    argv = wrap(nonet_argv([python, "-I", "-B", "-c", _PROBE], python))
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd="/")
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, "no-network self-test did not run: %s" % e
    if r.returncode == 0:
        return True, "ok"
    return False, "no-network self-test failed (exit %d): %s" % (r.returncode, (r.stderr or "").strip()[-300:])


if __name__ == "__main__":
    print(selftest(lambda argv: argv, sys.executable))
