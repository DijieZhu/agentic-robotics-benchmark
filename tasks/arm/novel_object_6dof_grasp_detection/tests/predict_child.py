#!/usr/bin/env python3
"""Run the submitted detector on hidden frames handed over one at a time, as the agent's account, offline.

verify.py copies this file to a root-owned directory and starts it with
`runuser -u <agent> -- /opt/grasp/bin/python -I predict_child.py <tag>`, so the agent can neither
edit it nor plant code that runs before it (-I: no user site-packages, no PYTHON* variables,
no script directory on sys.path).

Before the submission is imported, the process installs a seccomp filter that refuses
AF_INET/AF_INET6 sockets and io_uring. The filter is inherited by every thread and child
process and cannot be removed, so the detector is graded offline even though the engine
opens egress for the verify stage (the hidden frames' labels are public on the internet).

Protocol: messages over the process's stdin (from the grader) and stdout (to the grader), each a
1-byte tag, a 4-byte big-endian length and the payload. Before the submission is imported, fd 0
and fd 1 are moved out of the way (stdin becomes /dev/null, stdout the log), so nothing the
submission prints reaches the grader.
  child -> grader   G  the seccomp filter is in place      X  it could not be installed (text)
                    R  the submission imported              I  it did not (text, first 2000 characters)
                    A  the returned array (.npy bytes)      E  no array for this frame (text)
  grader -> child   F  one frame: a JSON header line {"name", "sizes"}, then rgb.png, depth.png
                       and camK.npy; "name" is empty except for the privileged reference
The grader sends the next frame only after it has the answer to the current one, times every
call on its own clock and kills the process when a call runs over; nothing here keeps time.
The process ends when the grader closes its stdin.
"""
import ctypes
import os
import platform
import struct
import sys
import time

SUBMISSION = os.path.join(os.environ.get("HOME", "/home/user"), "submission")


def block_network():
    """Install the seccomp filter. Raises if it cannot, so grading never runs unguarded."""
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


MAX_ARRAY_ROWS = 5000


def send(out, tag, payload=b""):
    out.write(tag + struct.pack(">I", len(payload)) + payload)
    out.flush()


def read_exact(inp, n):
    chunks, left = [], n
    while left:
        chunk = inp.read(left)
        if not chunk:
            return None
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def recv(inp):
    head = read_exact(inp, 5)
    if head is None:
        return None, None
    payload = read_exact(inp, struct.unpack(">I", head[1:])[0])
    return (head[:1], payload) if payload is not None else (None, None)


def decode_frame(payload):
    import io
    import json
    import numpy as np
    from PIL import Image
    line, _, blob = payload.partition(b"\n")
    header = json.loads(line.decode())
    parts, at = [], 0
    for size in header["sizes"]:
        parts.append(blob[at:at + size])
        at += size
    rgb = np.array(Image.open(io.BytesIO(parts[0])))
    depth = np.array(Image.open(io.BytesIO(parts[1])))
    K = np.load(io.BytesIO(parts[2]), allow_pickle=False)
    return header.get("name", ""), rgb, depth, K


def main():
    tag = sys.argv[1]
    # The protocol channel: private duplicates of fd 0 and fd 1 (not inherited by subprocesses).
    inp = os.fdopen(os.dup(0), "rb", buffering=0)
    out = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)                                        # the submission's prints go to the log
    try:
        block_network()
    except Exception as exc:
        print("[child %s] network guard failed: %s" % (tag, exc), file=sys.stderr, flush=True)
        send(out, b"X", str(exc)[:300].encode(errors="replace"))
        return 3
    send(out, b"G")

    import io
    import numpy as np

    sys.path.insert(0, SUBMISSION)
    try:
        import solver                                    # the submitted code
        predict = solver.predict
    except BaseException as exc:                         # noqa: BLE001 — any failure is the agent's
        send(out, b"I", ("%s: %s" % (type(exc).__name__, exc))[:2000].encode(errors="replace"))
        return 4
    send(out, b"R")

    while True:
        kind, payload = recv(inp)
        if kind != b"F":
            return 0
        name, rgb, depth, K = decode_frame(payload)
        if name:
            os.environ["ORACLE_FRAME"] = name            # sent only to the privileged reference
        t0 = time.time()
        try:
            grasps = np.asarray(predict(rgb, depth, K, 1000.0), dtype=np.float64)
            if grasps.ndim != 2 or grasps.shape[1:] != (17,) or len(grasps) > MAX_ARRAY_ROWS:
                raise ValueError("returned shape %s" % (grasps.shape,))
            buf = io.BytesIO()
            np.save(buf, grasps, allow_pickle=False)
            reply = (b"A", buf.getvalue())
        except BaseException as exc:                     # noqa: BLE001 — one bad frame, not the rest
            reply = (b"E", ("%s: %s" % (type(exc).__name__, exc))[:300].encode(errors="replace"))
        print("[child %s] %s %.1fs" % (tag, name or "frame", time.time() - t0), file=sys.stderr, flush=True)
        send(out, *reply)


if __name__ == "__main__":
    sys.exit(main())
