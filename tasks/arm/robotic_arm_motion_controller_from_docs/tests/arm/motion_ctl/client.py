"""Child side of the controller runner (runs as the unprivileged runner account; stdlib only).

    python3 -B -E -s -c <BOOT> <lib dir> <workdir>      (cwd = the read-only copy of the submission's controller/)

Before the import it makes this process the subreaper of what the controller starts (the grader's CPU accounting sees
every descendant) and installs a seccomp filter that refuses every socket but a local Unix one (`_no_network`: the
controller PC has no network; the ready line reports it). Imports `controller.py` from the working directory, builds
`Controller(workdir)` and answers JSON lines on stdin, one
reply line per request on stdout (the controller's own prints go to stderr: stdout is redirected before the import):

    {"op": "setup", "unit": {...}, "cell": {...}, "tool": {...}}  -> {"ok": true}
    {"op": "plan", "job": {...}, "state": {...}}                     -> {"ok": true, "result": "ok" | "infeasible"}
    {"op": "step", "state": {...}}                                   -> {"ok": true, "cmd": {...}}
    {"op": "end"}                                                    -> {"ok": true} and exit

An exception inside the controller answers {"ok": false, "error": "<type>: <message>"} (the grader decides what it
means); a reply that is not JSON-serialisable answers {"ok": false, "error": "reply: ..."}.
"""

import json
import os
import sys
import traceback


def _takes_state(fn):
    """plan(job, state) or the older plan(job): two positional parameters (besides self) or *args."""
    import inspect
    try:
        ps = [p for p in inspect.signature(fn).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL) and p.name != "truth"]
    except (TypeError, ValueError):
        return False
    return any(p.kind == p.VAR_POSITIONAL for p in ps) or len(ps) >= 2


def _subreaper():
    """Orphans of the controller's processes become this process's children (PR_SET_CHILD_SUBREAPER), so every process
    the controller starts stays visible to the grader's CPU accounting until it has been waited for."""
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)
    except Exception:                              # noqa: BLE001
        pass


def _no_network():
    """A seccomp filter on this process and everything it starts (threads, children, programs they exec): socket(2)
    only for AF_UNIX, io_uring refused, the other syscall ABIs refused; the controller PC has no network. Returns True
    when installed."""
    try:
        import ctypes
        import platform
        tbl = {"x86_64": (0xC000003E, 41, 425), "aarch64": (0xC00000B7, 198, 425)}   # audit arch, socket, io_uring_setup
        if platform.machine() not in tbl:
            return False
        audit_arch, nr_socket, nr_uring = tbl[platform.machine()]
        LD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06
        ALLOW, DENY = 0x7FFF0000, 0x00050000 | 1      # SECCOMP_RET_ALLOW, SECCOMP_RET_ERRNO | EPERM
        prog = [(LD, 0, 0, 4), (JEQ, 0, 7, audit_arch), (LD, 0, 0, 0), (JGE, 5, 0, 0x40000000), (JEQ, 0, 2, nr_socket),
                (LD, 0, 0, 16), (JEQ, 1, 2, 1), (JEQ, 1, 0, nr_uring), (RET, 0, 0, ALLOW), (RET, 0, 0, DENY)]

        class Filt(ctypes.Structure):
            _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]

        class Prog(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filt))]
        arr = (Filt * len(prog))(*[Filt(*f) for f in prog])
        fp = Prog(len(prog), arr)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:            # PR_SET_NO_NEW_PRIVS
            return False
        return libc.prctl(22, 2, ctypes.byref(fp), 0, 0) == 0     # PR_SET_SECCOMP, SECCOMP_MODE_FILTER
    except Exception:                              # noqa: BLE001
        return False


def _say(out, obj):
    out.write(json.dumps(obj, separators=(",", ":")) + "\n")
    out.flush()


def main(argv):
    workdir = argv[0] if argv else os.getcwd()
    _subreaper()
    no_net = _no_network()
    out = sys.stdout
    sys.stdout = sys.stderr
    sys.path.insert(0, os.getcwd())
    try:
        import controller as _c                    # noqa: the submission's module
        ctl = _c.Controller(workdir)
    except BaseException as e:                     # noqa: BLE001
        traceback.print_exc()
        _say(out, {"ok": False, "ready": False, "error": "import: {}: {}".format(type(e).__name__, e)})
        return 1
    _say(out, {"ok": True, "ready": True, "no_network": bool(no_net)})
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            _say(out, {"ok": False, "error": "bad request"})
            continue
        op = req.get("op")
        try:
            if op == "setup":
                if "truth" in req:
                    ctl.setup(req["unit"], req["cell"], req["tool"], truth=req["truth"])
                else:
                    ctl.setup(req["unit"], req["cell"], req["tool"])
                rep = {"ok": True}
            elif op == "plan":
                kw = {"truth": req["truth"]} if "truth" in req else {}
                if _takes_state(ctl.plan):
                    r = ctl.plan(req["job"], req.get("state"), **kw)
                else:
                    r = ctl.plan(req["job"], **kw)
                rep = {"ok": True, "result": "infeasible" if r == "infeasible" else "ok"}
            elif op == "step":
                rep = {"ok": True, "cmd": ctl.step(req["state"])}
            elif op == "end":
                _say(out, {"ok": True})
                return 0
            else:
                rep = {"ok": False, "error": "unknown op"}
        except BaseException as e:                 # noqa: BLE001
            traceback.print_exc()
            rep = {"ok": False, "error": "{}: {}".format(type(e).__name__, e)}
        try:
            _say(out, rep)
        except (TypeError, ValueError) as e:
            _say(out, {"ok": False, "error": "reply: {}".format(e)})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
