
import json
import os
import sys
import traceback


def _subreaper():
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)
    except Exception:
        pass


def _no_network():
    try:
        import ctypes
        import platform
        import signal
        tbl = {"x86_64": (0xC000003E, 41, 425, 13, 157), "aarch64": (0xC00000B7, 198, 425, 134, 167)}
        if platform.machine() not in tbl:
            return False
        audit_arch, nr_socket, nr_uring, nr_sigaction, nr_prctl = tbl[platform.machine()]
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        LD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06
        ALLOW, DENY = 0x7FFF0000, 0x00050000 | 1
        SIGCHLD, PR_SET_CHILD_SUBREAPER = 17, 36
        A0, A1_LO, A1_HI = 16, 24, 28
        prog = [(LD, 0, 0, 4), (JEQ, 0, 17, audit_arch), (LD, 0, 0, 0), (JGE, 15, 0, 0x40000000),
                (JEQ, 0, 2, nr_socket), (LD, 0, 0, A0), (JEQ, 11, 12, 1),
                (JEQ, 11, 0, nr_uring),
                (JEQ, 0, 6, nr_sigaction), (LD, 0, 0, A0), (JEQ, 0, 7, SIGCHLD),
                (LD, 0, 0, A1_LO), (JEQ, 0, 6, 0), (LD, 0, 0, A1_HI), (JEQ, 3, 4, 0),
                (JEQ, 0, 2, nr_prctl), (LD, 0, 0, A0), (JEQ, 1, 0, PR_SET_CHILD_SUBREAPER),
                (RET, 0, 0, ALLOW), (RET, 0, 0, DENY)]

        class Filt(ctypes.Structure):
            _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]

        class Prog(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filt))]
        arr = (Filt * len(prog))(*[Filt(*f) for f in prog])
        fp = Prog(len(prog), arr)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            return False
        return libc.prctl(22, 2, ctypes.byref(fp), 0, 0) == 0
    except Exception:
        return False


def _say(out, obj):
    out.write(json.dumps(obj, separators=(",", ":")) + "\n")
    out.flush()


def main(argv):
    _subreaper()
    no_net = _no_network()
    out = sys.stdout
    sys.stdout = sys.stderr
    sys.path.insert(0, os.getcwd())
    try:
        import controller as _c
        ctl = _c.Controller()
    except BaseException as e:
        traceback.print_exc()
        _say(out, {"ok": False, "ready": False, "no_network": bool(no_net),
                   "error": "import: {}: {}".format(type(e).__name__, e)})
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
                    r = ctl.setup(req["unit_record"], req["site"], truth=req["truth"])
                else:
                    r = ctl.setup(req["unit_record"], req["site"])
                rep = {"ok": True, "result": r}
            elif op == "step":
                rep = {"ok": True, "result": ctl.step(req["feedback"])}
            elif op == "end":
                _say(out, {"ok": True})
                return 0
            else:
                rep = {"ok": False, "error": "unknown op"}
        except BaseException as e:
            traceback.print_exc()
            rep = {"ok": False, "error": "{}: {}".format(type(e).__name__, e)}
        try:
            _say(out, rep)
        except (TypeError, ValueError, OverflowError, RecursionError) as e:
            _say(out, {"ok": False, "error": "reply: {}".format(e), "reply_error": True})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
