"""A network attempt KILLS the program; a new process is REFUSED (grader side, stdlib only).

The house filter `cellkit.nonet` (installed first, byte-identical to the benchmark's behavior tasks) makes every
socket() / socketpair() but AF_UNIX fail with EPERM: nothing of the program reaches a network. It cannot tell the
grader that a program tried. This second filter, stacked on it, answers the same calls with SECCOMP_RET_KILL_PROCESS:
the kernel applies the strictest action of all installed filters, so the attempt ends the process at once with SIGSYS,
before the call does anything, and the grader reads the attempt from the exit status (-SIGSYS: `attempted`). The same
for io_uring_setup (io_uring can open sockets without the socket call) and for any syscall made through a foreign ABI
(i386 `int 0x80`, where socketcall multiplexes socket; x32). Like the house filter it is inherited by every descendant
and cannot be removed, needs no privilege (PR_SET_NO_NEW_PRIVS first) and leaves pipes, files and Unix-domain sockets
alone. A program cannot catch SIGSYS from a KILL_PROCESS action.

The same filter refuses every NEW PROCESS (the static audit cannot see every way to
reach os.fork, and a child the program reaps before its venue ends was invisible to the survivor check and to the CPU
accounting, which reads the program's own /proc/<pid>/stat): fork() and vfork() fail with EPERM, clone() fails with
EPERM unless it makes a thread of the same process (CLONE_THREAD set), and clone3() fails with ENOSYS, so that the C
library falls back to clone() for a new thread (glibc >= 2.34 does: its pthread_create and posix_spawn try clone3
first; seccomp cannot read clone3's flags, which sit in a structure in memory). So os.fork, os.forkpty, os.system,
os.posix_spawn, subprocess, multiprocessing and every other route a Python program has fail with PermissionError /
OSError, and no process exists but the program's own: threads of it stay possible (their CPU time is in the program's
/proc/<pid>/stat), and execve() stays possible (the stages before the program exec, so the filter cannot refuse it;
an exec replaces the program in place: the same pid, process group, limits, filters and CPU accounting).

    argv = netkill_argv(program_argv)          # python3 -I -B -c NETKILL_SRC -- program_argv...
    ok, detail = selftest(wrap)                # proves, as the program would run, that AF_UNIX works and AF_INET kills
    ok, detail = proc_selftest(wrap)           # proves that no new process can be made and a thread still can
"""

import json
import signal
import subprocess
import sys

__all__ = ["NETKILL_SRC", "netkill_argv", "attempted", "selftest", "proc_selftest", "SIGSYS"]

SIGSYS = int(signal.SIGSYS)

# Run as `python3 -I -B -c NETKILL_SRC -- <argv...>`: installs the filter, then execs argv (environment inherited).
# ARCH: audit arch, socket, socketpair, io_uring_setup, x32 bit, clone, clone3, fork, vfork (None: no such syscall).
NETKILL_SRC = r'''
import ctypes, os, platform, struct, sys
ARCH = {"x86_64": (0xC000003E, 41, 53, 425, 0x40000000, 56, 435, 57, 58),
        "aarch64": (0xC00000B7, 198, 199, 425, 0, 220, 435, None, None)}
def ins(code, jt, jf, k):
    return struct.pack("HBBI", code, jt, jf, k)
def main():
    audit, nr_socket, nr_socketpair, nr_uring, x32, nr_clone, nr_clone3, nr_fork, nr_vfork = ARCH[platform.machine()]
    LD, JEQ, JGE, JSET, RET = 0x20, 0x15, 0x35, 0x45, 0x06
    allow, kill = 0x7FFF0000, 0x80000000
    eperm, enosys, clone_thread = 0x00050000 | 1, 0x00050000 | 38, 0x00010000
    prog = [ins(LD, 0, 0, 4), ins(JEQ, 1, 0, audit), ins(RET, 0, 0, kill), ins(LD, 0, 0, 0)]
    if x32:
        prog += [ins(JGE, 0, 1, x32), ins(RET, 0, 0, kill)]
    prog += [ins(JEQ, 0, 1, nr_uring), ins(RET, 0, 0, kill)]
    for nr in (nr_fork, nr_vfork):
        if nr is not None:
            prog += [ins(JEQ, 0, 1, nr), ins(RET, 0, 0, eperm)]
    prog += [ins(JEQ, 0, 1, nr_clone3), ins(RET, 0, 0, enosys),
             ins(JEQ, 0, 4, nr_clone), ins(LD, 0, 0, 16), ins(JSET, 0, 1, clone_thread), ins(RET, 0, 0, allow),
             ins(RET, 0, 0, eperm)]
    prog += [ins(JEQ, 2, 0, nr_socket), ins(JEQ, 1, 0, nr_socketpair),
             ins(RET, 0, 0, allow), ins(LD, 0, 0, 16), ins(JEQ, 0, 1, 1), ins(RET, 0, 0, allow), ins(RET, 0, 0, kill)]
    raw = b"".join(prog)
    buf = ctypes.create_string_buffer(raw, len(raw))
    class Fprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
    fp = Fprog(len(raw) // 8, ctypes.cast(buf, ctypes.c_void_p))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(22, 2, ctypes.byref(fp), 0, 0) != 0:
        sys.stderr.write("[netkill] the seccomp filter could not be installed (errno %d)\n" % ctypes.get_errno())
        os._exit(96)
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    os.execvp(argv[0], argv)
main()
'''


def netkill_argv(argv, python=None):
    """argv run under the killing filter (the launcher interpreter is the grader's own Python, isolated mode)."""
    return [python or sys.executable, "-I", "-B", "-c", NETKILL_SRC, "--"] + list(argv)


def attempted(returncode):
    """True when a process ended by the filter's SIGSYS (its return code as subprocess reports it)."""
    return returncode is not None and int(returncode) == -SIGSYS


_PROBE_UNIX = ("import socket, sys\n"
               "s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.close()\n"
               "a, b = socket.socketpair(); a.close(); b.close()\n"
               "sys.stdout.write('unix-ok\\n'); sys.stdout.flush()\n")
_PROBES = {"inet": _PROBE_UNIX + "socket.socket(socket.AF_INET, socket.SOCK_STREAM)\nsys.exit(3)\n",
           "inet6": _PROBE_UNIX + "socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)\nsys.exit(3)\n",
           "netlink": _PROBE_UNIX + "socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, 0)\nsys.exit(3)\n",
           "unix_only": _PROBE_UNIX + "sys.exit(0)\n"}


def selftest(wrap=None, env=None, timeout=60.0):
    """(ok, detail): under the filter (as `wrap` runs a program: account, limits, the house filter) a Unix socket and
    a socket pair work, and an AF_INET, AF_INET6 or AF_NETLINK socket ends the process with SIGSYS."""
    details = []
    ok = True
    for name, src in sorted(_PROBES.items()):
        argv = netkill_argv([sys.executable, "-I", "-B", "-c", src])
        if wrap is not None:
            argv = wrap(argv)
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, cwd="/")
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, "netkill self-test %s did not run: %s" % (name, e)
        want_kill = name != "unix_only"
        good = "unix-ok" in (r.stdout or "") and (attempted(r.returncode) if want_kill else r.returncode == 0)
        ok &= good
        details.append("%s:%s(rc %s)" % (name, "ok" if good else "FAILED", r.returncode))
        if not good:
            details.append((r.stderr or "").strip()[-200:])
    return ok, " ".join(details)


# Every route to a new process a Python program has (fork, forkpty, vfork-based subprocess, posix_spawn, the C
# library's system()), each must fail; a thread must still start, run and finish. One JSON line on stdout.
_PROBE_PROC = r'''
import errno, json, os, subprocess, sys, threading
r = {}
def attempt(name, fn):
    try:
        fn()
        r[name] = "made"
    except OSError as e:
        r[name] = "refused" if e.errno in (errno.EPERM, errno.ENOSYS, errno.EAGAIN) else "error %s" % e.errno
def _fork():
    if os.fork() == 0:
        os._exit(0)
def _forkpty():
    if os.forkpty()[0] == 0:
        os._exit(0)
def _spawn():
    os.posix_spawn("/bin/true", ["/bin/true"], {})
def _sub():
    subprocess.run(["/bin/true"])
attempt("fork", _fork)
attempt("forkpty", _forkpty)
attempt("posix_spawn", _spawn)
attempt("subprocess", _sub)
r["system"] = "made" if os.system("true") == 0 else "refused"
box = []
t = threading.Thread(target=lambda: box.append(sum(range(1000))))
t.start()
t.join()
r["thread"] = "ran" if box == [499500] else "did not run"
sys.stdout.write(json.dumps(r, sort_keys=True) + "\n")
'''


def proc_selftest(wrap=None, env=None, timeout=60.0):
    """(ok, detail): under the filter (as `wrap` runs a program) fork, forkpty, posix_spawn, subprocess and system()
    make no process, and a thread of the program starts and finishes."""
    argv = netkill_argv([sys.executable, "-I", "-B", "-c", _PROBE_PROC])
    if wrap is not None:
        argv = wrap(argv)
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, cwd="/")
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, "process self-test did not run: %s" % e
    try:
        got = json.loads((r.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, "process self-test: no report (rc %s) %s" % (r.returncode, (r.stderr or "").strip()[-200:])
    want = {"fork": "refused", "forkpty": "refused", "posix_spawn": "refused", "subprocess": "refused",
            "system": "refused", "thread": "ran"}
    ok = r.returncode == 0 and got == want
    return ok, " ".join("%s:%s" % (k, got.get(k)) for k in sorted(want)) + ("" if ok else " (rc %s)" % r.returncode)


if __name__ == "__main__":
    print(selftest())
    print(proc_selftest())
