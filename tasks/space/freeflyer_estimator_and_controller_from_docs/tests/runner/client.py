"""Child side of the wire (standard library only; copied alone into the run's lib folder as ff_client.py, so that the
controller's process never sees the grader's code):

  python3 -I -B -c <BOOT> <lib dir> <sandbox json>     (cwd = the read-only copy of the submission's controller/)

Before the controller is imported this process (1) becomes the subreaper of what the controller starts, so that every
descendant stays visible to the grader's processor accounting until it has been waited for; (2) restricts its own file
system view with Landlock (read: the submission copy, the client, the interpreter's installation and the system's
libraries; read and write: its own HOME and TMPDIR; nothing else); (3) installs a seccomp filter on itself and on
everything it starts (no socket of any family, no signal to any process, no tracing or reading of another
process's memory, no new namespace, no leaving the process group or the session, no io_uring, no mount); (4) points
file descriptor 1 at the error stream so that the controller's prints never reach the wire. The ready line reports
which of them were installed; the grader refuses to run when a required one is missing. Then it answers JSON lines:

    {"op": "setup", "session": {...}}                    -> {"ok": true, "result": <return value>}
    {"op": "plan", "job": {...}, "frame": {...}}         -> {"ok": true, "result": {...}}
    {"op": "step", "frame": {...}}                       -> {"ok": true, "result": {...}}
    {"op": "end"}                                        -> {"ok": true} and exit

An exception inside the controller answers {"ok": false, "error": "<type>: <message>"}. Return values are written as
JSON with numpy values turned into lists and floats; a NaN or an infinity is written as the JSON constant, which the
grader reads whole and refuses by the reply's form (a protocol error; the process lives on)."""

import json
import os
import platform
import sys
import traceback


# ------------------------------------------------------------------------------------------------ the sandbox
def _subreaper():
    try:
        import ctypes
        return ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0      # PR_SET_CHILD_SUBREAPER
    except Exception:                              # noqa: BLE001
        return False


# x86_64 system-call numbers refused with EPERM (the filter refuses every other ABI and architecture)
_DENY_X86_64 = {
    "socket": 41, "connect": 42, "accept": 43, "bind": 49, "listen": 50, "socketpair": 53, "accept4": 288,
    "kill": 62, "tkill": 200, "tgkill": 234, "rt_sigqueueinfo": 129, "rt_tgsigqueueinfo": 297,
    "pidfd_send_signal": 424, "pidfd_open": 434, "pidfd_getfd": 438,
    "ptrace": 101, "process_vm_readv": 310, "process_vm_writev": 311, "process_madvise": 440, "kcmp": 312,
    "setsid": 112, "setpgid": 109, "unshare": 272, "setns": 308,
    "mount": 165, "umount2": 166, "pivot_root": 155, "chroot": 161, "fsopen": 430, "fsmount": 432, "fspick": 433,
    "move_mount": 429, "open_tree": 428, "mount_setattr": 442,
    "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
    "bpf": 321, "perf_event_open": 298, "keyctl": 250, "add_key": 248, "request_key": 249, "userfaultfd": 323,
    "move_pages": 279, "migrate_pages": 256, "name_to_handle_at": 303, "open_by_handle_at": 304,
}
_CLONE3_X86_64 = 435           # refused with ENOSYS, so that the C library falls back to clone (whose flags are seen)
_CLONE_X86_64 = 56
_CLONE_NEW_MASK = 0x7E020000   # CLONE_NEWNS | NEWCGROUP | NEWUTS | NEWIPC | NEWUSER | NEWPID | NEWNET


def _seccomp():
    """the system-call filter (an allow list would break numpy and scipy; this is a deny list on one ABI)."""
    try:
        import ctypes
        if platform.machine() != "x86_64":
            return False
        audit_arch = 0xC000003E
        LD, JEQ, JGE, RET, AND = 0x20, 0x15, 0x35, 0x06, 0x54
        ALLOW, EPERM, ENOSYS = 0x7FFF0000, 0x00050000 | 1, 0x00050000 | 38
        prog = []
        # [0] load arch; [1] arch == x86_64 else deny; [2] load nr; [3] other ABIs (x32) denied
        prog.append((LD, 0, 0, 4))
        prog.append(("ARCH",))
        prog.append((LD, 0, 0, 0))
        prog.append(("ABI_X32",))
        deny = sorted(set(_DENY_X86_64.values()))
        for nr in deny:
            prog.append(("DENY_IF", nr))
        prog.append(("NOSYS_IF", _CLONE3_X86_64))
        prog.append(("CLONE_IF", _CLONE_X86_64))
        prog.append((RET, 0, 0, ALLOW))
        # tail: [deny] [nosys] [clone check: load arg0, and mask, jeq 0 -> allow else deny]
        n_body = len(prog)
        i_deny = n_body
        i_nosys = n_body + 1
        i_clone = n_body + 2           # LD arg0
        out = []
        for i, ins in enumerate(prog):
            if ins[0] == "ARCH":
                out.append((JEQ, 0, i_deny - i - 1, audit_arch))
            elif ins[0] == "ABI_X32":
                out.append((JGE, i_deny - i - 1, 0, 0x40000000))
            elif ins[0] == "DENY_IF":
                out.append((JEQ, i_deny - i - 1, 0, ins[1]))
            elif ins[0] == "NOSYS_IF":
                out.append((JEQ, i_nosys - i - 1, 0, ins[1]))
            elif ins[0] == "CLONE_IF":
                out.append((JEQ, i_clone - i - 1, 0, ins[1]))
            else:
                out.append(ins)
        out.append((RET, 0, 0, EPERM))                 # i_deny
        out.append((RET, 0, 0, ENOSYS))                # i_nosys
        out.append((LD, 0, 0, 16))                     # i_clone: low word of arg0 (the flags)
        out.append((AND, 0, 0, _CLONE_NEW_MASK))
        out.append((JEQ, 0, 1, 0))
        out.append((RET, 0, 0, ALLOW))
        out.append((RET, 0, 0, EPERM))
        for ins in out:
            if ins[1] > 255 or ins[2] > 255:
                return False

        class Filt(ctypes.Structure):
            _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]

        class Prog(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filt))]
        arr = (Filt * len(out))(*[Filt(*f) for f in out])
        fp = Prog(len(out), arr)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:            # PR_SET_NO_NEW_PRIVS
            return False
        return libc.prctl(22, 2, ctypes.byref(fp), 0, 0) == 0     # PR_SET_SECCOMP, SECCOMP_MODE_FILTER
    except Exception:                              # noqa: BLE001
        return False


def _landlock(read_dirs, rw_dirs, read_files):
    """restrict this process (and what it starts) to the given paths; returns the Landlock ABI or 0."""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        NR_CREATE, NR_ADD, NR_RESTRICT = 444, 445, 446
        if platform.machine() != "x86_64":
            return 0
        abi = libc.syscall(NR_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(1))   # LANDLOCK_CREATE_RULESET_VERSION
        if abi < 1:
            return 0
        nbits = {1: 13, 2: 14, 3: 15, 4: 15}.get(int(abi), 16)
        handled = (1 << nbits) - 1
        EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1, 2, 4, 8
        TRUNCATE = (1 << 14) if nbits > 14 else 0
        IOCTL_DEV = (1 << 15) if nbits > 15 else 0
        file_rights = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE | IOCTL_DEV

        class Attr(ctypes.Structure):
            _fields_ = [("handled_access_fs", ctypes.c_uint64)]

        class Beneath(ctypes.Structure):
            _pack_ = 1
            _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]
        attr = Attr(handled)
        rfd = libc.syscall(NR_CREATE, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint32(0))
        if rfd < 0:
            return 0
        ro = EXECUTE | READ_FILE | READ_DIR
        rules = [(p, ro) for p in read_dirs] + [(p, handled) for p in rw_dirs] + \
                [(p, (READ_FILE | WRITE_FILE) & file_rights) for p in read_files]
        for p, acc in rules:
            if not os.path.exists(p):
                continue
            fd = os.open(p, os.O_PATH | os.O_CLOEXEC)
            if not os.path.isdir(p):
                acc &= file_rights
            b = Beneath(acc, fd)
            r = libc.syscall(NR_ADD, ctypes.c_int(rfd), ctypes.c_int(1), ctypes.byref(b), ctypes.c_uint32(0))
            os.close(fd)
            if r != 0:
                os.close(rfd)
                return 0
        if libc.prctl(38, 1, 0, 0, 0) != 0:            # PR_SET_NO_NEW_PRIVS
            os.close(rfd)
            return 0
        r = libc.syscall(NR_RESTRICT, ctypes.c_int(rfd), ctypes.c_uint32(0))
        os.close(rfd)
        return int(abi) if r == 0 else 0
    except Exception:                              # noqa: BLE001
        return 0


# ------------------------------------------------------------------------------------------------ the wire
def _plain(x):
    """numpy values (and tuples) as JSON's lists and numbers; anything else unchanged."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    tl = getattr(x, "tolist", None)
    if tl is not None and not isinstance(x, (str, bytes)):
        return _plain(tl())
    return x


def _say(fd, obj):
    os.write(fd, (json.dumps(_plain(obj), separators=(",", ":")) + "\n").encode("utf-8"))


def main(argv):
    box = json.loads(argv[0]) if argv else {}
    wire_out = os.dup(1)                           # the reply stream, kept on a private descriptor
    os.dup2(2, 1)                                  # the controller's prints (and fd 1 writes) go to the error stream
    sys.stdout = sys.stderr
    reaper = _subreaper()
    ll = _landlock(box.get("read_dirs", []), box.get("rw_dirs", []), box.get("rw_files", []))
    sc = _seccomp()
    sys.path.insert(0, os.getcwd())
    try:
        import controller as _c                    # noqa: the submission's module
        ctl = _c.Controller()
    except BaseException as e:                     # noqa: BLE001
        traceback.print_exc()
        _say(wire_out, {"ok": False, "ready": False, "error": "import: {}: {}".format(type(e).__name__, e),
                        "sandbox": {"subreaper": reaper, "landlock": ll, "seccomp": sc}})
        return 1
    _say(wire_out, {"ok": True, "ready": True, "sandbox": {"subreaper": reaper, "landlock": ll, "seccomp": sc}})
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            _say(wire_out, {"ok": False, "error": "bad request"})
            continue
        op = req.get("op")
        try:
            if op == "setup":
                rep = {"ok": True, "result": ctl.setup(req["session"])}
            elif op == "plan":
                rep = {"ok": True, "result": ctl.plan(req["job"], req["frame"])}
            elif op == "step":
                rep = {"ok": True, "result": ctl.step(req["frame"])}
            elif op == "end":
                _say(wire_out, {"ok": True})
                return 0
            else:
                rep = {"ok": False, "error": "unknown op"}
        except BaseException as e:                 # noqa: BLE001
            traceback.print_exc()
            rep = {"ok": False, "error": "{}: {}".format(type(e).__name__, e)}
        try:
            _say(wire_out, rep)
        except (TypeError, ValueError) as e:
            _say(wire_out, {"ok": False, "error": "reply: {}".format(e)})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
