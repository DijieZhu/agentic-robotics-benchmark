#!/usr/bin/env python3
"""Run whatever is in /home/user/submission/solver.py on the practice frames the way the graded run
calls it, and report whether every call meets the contract and the time limits. Free and
unlimited. It does not score grasps: there are no meshes here to score them against.

    /opt/grasp/bin/python ~/practice_check/check.py              # 2 views per practice scene
    /opt/grasp/bin/python ~/practice_check/check.py --views 8    # every practice view

Like the graded run, it imports solver.py once with OMP_NUM_THREADS=1, the working directory set
to the submission folder and the same network guard (no AF_INET/AF_INET6 sockets, so no localhost
TCP either, and no io_uring), then calls predict() frame by frame. The graded run uses 8 such
processes at once on 8 CPUs, so a call that is close to the limit here may run over there.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, ctypes, json, platform, signal, struct, sys, time

import numpy as np
from PIL import Image

DATA = os.environ.get("GRASPNET_DATA", "/data/graspnet")
SUBMISSION = os.path.join(os.path.expanduser("~"), "submission")
MAX_HEIGHT, MAX_DEPTH, MAX_SUBMIT = 0.1, 0.1, 5000
IMPORT_TIMEOUT, CALL_TIMEOUT = 120, 10


def block_network():
    """The graded run's network guard (a seccomp filter), installed before solver.py is imported."""
    machine = platform.machine()
    if machine in ("x86_64", "AMD64"):
        audit_arch, nr_socket = 0xC000003E, 41
    elif machine in ("aarch64", "arm64"):
        audit_arch, nr_socket = 0xC00000B7, 198
    else:
        raise RuntimeError("no network guard for architecture " + machine)
    nr_io_uring_setup = 425          # the same number on every architecture
    LD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06      # BPF_LD|W|ABS, JMP|JEQ|K, JMP|JGE|K, RET|K
    ALLOW, ERRNO_EACCES, KILL = 0x7FFF0000, 0x00050000 | 13, 0x80000000

    def ins(code, k, jt=0, jf=0):
        return struct.pack("=HBBI", code, jt, jf, k)

    # struct seccomp_data: nr at 0, arch at 4, args[0] at 16 (low 32 bits, little-endian)
    program = [
        ins(LD, 4),                              # 0  arch
        ins(JEQ, audit_arch, 1, 0),              # 1  native ABI -> 3
        ins(RET, KILL),                          # 2  any other ABI
        ins(LD, 0),                              # 3  syscall number
        ins(JGE, 0x40000000, 7, 0),              # 4  x32 ABI -> 12
        ins(JEQ, nr_io_uring_setup, 5, 0),       # 5  io_uring -> 11
        ins(JEQ, nr_socket, 0, 3),               # 6  not socket() -> 10
        ins(LD, 16),                             # 7  socket domain
        ins(JEQ, 2, 2, 0),                       # 8  AF_INET -> 11
        ins(JEQ, 10, 1, 0),                      # 9  AF_INET6 -> 11
        ins(RET, ALLOW),                         # 10
        ins(RET, ERRNO_EACCES),                  # 11
        ins(RET, KILL),                          # 12
    ]
    blob = b"".join(program)
    buf = ctypes.create_string_buffer(blob, len(blob))

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = SockFprog(len(program), ctypes.cast(buf, ctypes.c_void_p))
    libc = ctypes.CDLL(None, use_errno=True)
    PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 38, 22, 2
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP) failed")

    import socket
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            socket.socket(family, socket.SOCK_STREAM).close()
        except PermissionError:
            continue
        raise RuntimeError("network guard is not effective for family %d" % family)


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def validate(arr):
    """The contract the graded run applies; a frame that fails it scores 0 there."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 17 or arr.shape[0] == 0:
        raise ValueError("shape %s" % (arr.shape,))
    if len(arr) > MAX_SUBMIT:
        raise ValueError("%d grasps exceeds the %d cap" % (len(arr), MAX_SUBMIT))
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite values")
    if np.any(arr[:, 1] < 0):
        raise ValueError("negative width")
    if np.any(arr[:, 2] <= 0) or np.any(arr[:, 2] > MAX_HEIGHT):
        raise ValueError("height out of range")
    if np.any(arr[:, 3] <= 0) or np.any(arr[:, 3] > MAX_DEPTH):
        raise ValueError("depth out of range")
    R = arr[:, 4:13].reshape(-1, 3, 3)
    if np.abs(np.einsum("nij,nik->njk", R, R) - np.eye(3)).max() > 1e-4:
        raise ValueError("rotation not orthonormal")
    if np.any(np.abs(np.linalg.det(R) - 1.0) > 1e-4):
        raise ValueError("rotation determinant is not +1")
    return arr


def frames(n_views):
    m = json.load(open(f"{DATA}/manifest.json"))
    return [(r["scene"], v) for r in m["roles"]["selfcheck"] for v in r["views"][:n_views]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=2, help="views per practice scene")
    a = ap.parse_args()
    signal.signal(signal.SIGALRM, _alarm)
    block_network()
    os.chdir(SUBMISSION)
    sys.path.insert(0, SUBMISSION)
    t0 = time.time()
    try:
        signal.alarm(IMPORT_TIMEOUT)
        import solver
    except Timeout:
        sys.exit("import took more than %d s: every frame would score 0" % IMPORT_TIMEOUT)
    finally:
        signal.alarm(0)
    print("import %.1fs" % (time.time() - t0), flush=True)

    ok, times = 0, []
    todo = frames(a.views)
    for sid, view in todo:
        d = f"{DATA}/scenes/scene_{sid:04d}/realsense"
        rgb = np.array(Image.open(f"{d}/rgb/{view:04d}.png"))
        depth = np.array(Image.open(f"{d}/depth/{view:04d}.png"))
        K = np.load(f"{d}/camK.npy")
        status, n, t1 = "ok", 0, time.time()
        try:
            signal.alarm(CALL_TIMEOUT)
            n = len(validate(solver.predict(rgb, depth, K, 1000.0)))
            ok += 1
        except Timeout:
            status = "over %d s" % CALL_TIMEOUT
        except Exception as exc:
            status = "%s: %s" % (type(exc).__name__, str(exc)[:60])
        finally:
            signal.alarm(0)
        times.append(time.time() - t1)
        print("scene %04d view %03d  %5d grasps  %5.1fs  %s" % (sid, view, n, times[-1], status), flush=True)
    print("\n%d of %d calls valid and in time; slowest %.1fs, mean %.1fs"
          % (ok, len(todo), max(times), float(np.mean(times))))


if __name__ == "__main__":
    main()
