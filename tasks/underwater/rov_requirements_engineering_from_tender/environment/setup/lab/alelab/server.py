#!/usr/bin/env python3
"""alelab: a budgeted lab service that holds a task's hidden "real" environment.

Started once by a task's environment/setup/setup.sh as root, cwd = the stage (/opt/ale/setup):

    env -i PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 HOME=/nonexistent \\
        python3 -I -B lab/alelab/server.py --config "$PWD/lab/lab.json"

The command returns once the lab is serving (exit 0) or failed to start (exit 1); the service
itself is a fully detached daemon (setsid, stdio on /dev/null) and survives into the agent phase.

Process layout
  supervisor (root, comm "alelab-sup"): binds the unix socket, writes the pid file, forks and
      restarts the worker. It never reads the hidden instance file and never accepts a connection.
  worker (comm "alelab-worker"): forked as root, loads the task plugin and the hidden instance
      file, then drops to the dedicated lab account (setgroups([]), setgid, setuid, not dumpable)
      and serves the socket. It never runs as root or as the agent account while serving.
  Both run under PR_SET_NO_NEW_PRIVS and a seccomp filter under which socket()/socketpair() work
  only for AF_UNIX (no TCP/UDP, no network connection at any time, including before the seal).

Protocol: one JSON object per line in each direction over the unix socket.
  {"op": "hello"}                               -> ops, instance ids, public info, limits
  {"op": "budget"}                              -> {"remaining": n}
  {"op": "reset", "instance": id}               -> charges ONE trial, starts a trial, -> obs
  {"op": "step", "action": ...}                 -> obs, done (inside the current trial)
  {"op": "execute", "instance": id, "plan": ..} -> charges ONE trial, runs it open loop -> outcome
  {"op": "close"}                               -> ends the trial and the connection
Errors: {"ok": false, "error": <code>, "message": <fixed text>}. No tracebacks, no server state.

Budget: one counter behind one lock, charged atomically (persisted with fsync) when a reset or
execute is accepted and before any simulation. Disconnects, bad later steps, plugin failures and
worker restarts never refund. A restarted worker reads the counter back from the lab-only state
file; a missing or corrupt state file after the first start means zero trials (fail closed).

Plugin contract (a Python file named in the config; see tools/lab/examples/toy_mass/plugin.py):
  make_plugin(instances: dict, api) -> plugin object with
    SCHEMA: {"info": {...}, "reset": {...}, "step": {...}, "execute": {...}}   (whitelist)
    instance_ids() -> list[str]
    info() -> dict                                  (public constants only)
    MESSAGES: {code: fixed public text}             (the only texts a rejection may carry)
    validate_action(action) -> action               raise api.BadRequest(code) on bad input
    validate_plan(plan) -> plan                     raise api.BadRequest(code) on bad input
    reset(instance) -> (episode, obs)               episode.step(action) -> (obs, done)
    execute(instance, plan) -> outcome
  Every reply payload is checked against SCHEMA before it is sent (exact keys, types, shapes,
  finiteness); a mismatch is answered with {"error": "internal"} and logged lab-side only.
  validate_* see only the request (no instance id, no episode) and must use public constants only;
  rejections are free, so they must never depend on hidden state (rev-leak lab_leak_r1 F1). Before
  serving, the worker runs a fixed probe set through validate_* once without and once after a reset
  of every instance and refuses to start if any accept/reject decision or message differs.
  Rejected steps count toward the per-trial step cap. Import everything at module load: after the
  privilege drop the worker cannot read files under /opt/ale.

Dependencies: Python 3.12 standard library (numpy only if the plugin uses it).
"""

import argparse
import base64
import ctypes
import importlib.util
import inspect
import json
import math
import os
import platform
import pwd
import re
import resource
import select
import signal
import socket
import stat
import struct
import sys
import threading
import time
import traceback

PROTOCOL = 1
OPS = ("hello", "budget", "reset", "step", "execute", "close")
REQUEST_KEYS = {
    "hello": {"op"},
    "budget": {"op"},
    "reset": {"op", "instance"},
    "step": {"op", "action"},
    "execute": {"op", "instance", "plan"},
    "close": {"op"},
}
ERRORS = {
    "bad_json": "request is not one valid JSON object on one line",
    "line_too_long": "request line exceeds the size limit",
    "too_complex": "request nesting depth or element count exceeds the limit",
    "bad_request": "request fields are missing, unknown or of the wrong type",
    "unknown_op": "unknown op",
    "unsupported": "this lab does not support that op",
    "bad_instance": "unknown instance id",
    "bad_action": "action rejected",
    "bad_plan": "plan rejected",
    "no_trial": "no active trial on this connection; send reset first",
    "trial_over": "this trial reached its step or time limit",
    "budget_exhausted": "no trials remaining",
    "busy": "too many open connections",
    "rate_limited": "too many requests per second",
    "internal": "internal lab error",
}
DEFAULTS = {
    "plugin": None,  # path to the plugin .py (relative to the config file)
    "plugin_factory": "make_plugin",
    "plugin_options": {},  # public options handed to the factory as api.options
    "instances": None,  # hidden instance file (relative to the config file)
    "budget": None,  # total trials for the whole agent phase
    "user": "alelab",  # dedicated lab account (created in the task image)
    "agent_user": "user",  # the evaluated account; the lab refuses to run as it
    "socket": "/run/ale-lab/lab.sock",
    "state_dir": "/var/lib/ale-lab",
    "max_line_bytes": 1 << 20,
    "max_depth": 16,
    "max_items": 1_000_000,
    "max_connections": 4,
    "idle_timeout_s": 600.0,
    "max_requests_per_s": 2000.0,
    "trial_step_cap": 5000,
    "trial_wall_s": 900.0,
    "call_timeout_s": 120.0,
    "min_reply_s": {},  # per-op minimum reply latency, e.g. {"execute": 2.0}
    "max_restarts": 20,
    "start_timeout_s": 120.0,
    "require_seccomp": True,
    "env": {},  # PUBLIC constants only (e.g. MUJOCO_GL); never hidden values
    "max_instances_bytes": 64 << 20,
    "max_log_bytes": 64 << 20,
    "validation_check": True,  # F1: validate_* decisions must not depend on instance/state
}
INSTANCE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAX_ARRAY_ELEMS = 16_000_000

PR_SET_NAME = 15
PR_CAPBSET_DROP = 24
# capabilities the root supervisor keeps (worker re-fork as root, then drop): CHOWN, DAC_OVERRIDE,
# KILL, SETGID, SETUID
SUPERVISOR_CAPS = (0, 1, 5, 6, 7)
MESSAGE_CODE_RE = re.compile(r"^[a-z0-9_]{1,40}$")
# generic validation probes (F1); plugins may add VALIDATION_PROBES = {"action": [...], "plan": [...]}
GENERIC_PROBES = [
    None,
    True,
    0,
    0.0,
    1,
    -1,
    0.5,
    1e6,
    -1e6,
    1e300,
    "",
    "x",
    "../x",
    [],
    [0],
    [0.0, 0.0, 0.0],
    [[0]],
    [None],
    ["x"],
    {},
    {"a": 1},
    [0] * 1000,
    [1e6] * 10,
]
PR_SET_DUMPABLE = 4
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
EPERM = 1
# machine -> (AUDIT_ARCH, nr socket, nr socketpair, nr io_uring_setup, x32 syscall bit)
SECCOMP_ARCH = {
    "x86_64": (0xC000003E, 41, 53, 425, 0x40000000),
    "aarch64": (0xC00000B7, 198, 199, 425, 0),
}


class BadRequest(Exception):
    """Raised by a plugin's validate_* for input it rejects: BadRequest(code), where code is a key of
    the plugin's fixed MESSAGES table. The agent receives MESSAGES[code]; any other code or free text
    is replaced by the generic message (and logged lab-side)."""


class Api:
    BadRequest = BadRequest

    def __init__(self, options):
        self.options = dict(options)


class ReplyError(Exception):
    def __init__(self, code, message=None):
        super().__init__(code)
        self.code = code
        self.message = message


# ----------------------------------------------------------------------------------------------
# small helpers

_libc = None


def _prctl(option, arg2=0, arg3=0, arg4=0, arg5=0):
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
    if _libc.prctl(option, arg2, arg3, arg4, arg5) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _set_comm(name):
    try:
        buf = ctypes.create_string_buffer(name.encode()[:15])
        _prctl(PR_SET_NAME, ctypes.addressof(buf))
    except OSError:
        pass


def _seccomp_program(machine):
    audit, nr_socket, nr_socketpair, nr_uring, x32 = SECCOMP_ARCH[machine]
    ld, jeq, jge, ret = 0x20, 0x15, 0x35, 0x06
    deny = SECCOMP_RET_ERRNO | EPERM

    def ins(code, jt, jf, k):
        return struct.pack("HBBI", code, jt, jf, k)

    prog = [
        ins(ld, 0, 0, 4),
        ins(jeq, 1, 0, audit),
        ins(ret, 0, 0, deny),
        ins(ld, 0, 0, 0),
    ]
    if x32:
        prog += [ins(jge, 0, 1, x32), ins(ret, 0, 0, deny)]
    prog += [
        ins(jeq, 0, 1, nr_uring),
        ins(ret, 0, 0, deny),
        ins(jeq, 2, 0, nr_socket),
        ins(jeq, 1, 0, nr_socketpair),
        ins(ret, 0, 0, SECCOMP_RET_ALLOW),
        ins(ld, 0, 0, 16),  # args[0] low word = address family
        ins(jeq, 0, 1, socket.AF_UNIX),
        ins(ret, 0, 0, SECCOMP_RET_ALLOW),
        ins(ret, 0, 0, deny),
    ]
    return b"".join(prog)


def install_unix_only_seccomp():
    """PR_SET_NO_NEW_PRIVS + a filter under which only AF_UNIX sockets can be created."""
    machine = platform.machine()
    if machine not in SECCOMP_ARCH:
        raise OSError(0, "unsupported architecture for the seccomp filter: " + machine)
    prog = _seccomp_program(machine)
    buf = ctypes.create_string_buffer(prog, len(prog))

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = SockFprog(len(prog) // 8, ctypes.addressof(buf))
    _prctl(PR_SET_NO_NEW_PRIVS, 1)
    _prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.addressof(fprog))


def trim_capabilities(keep=SUPERVISOR_CAPS):
    """Drop every capability not in `keep` from the bounding, effective, permitted and inheritable sets."""
    for cap in range(64):
        if cap in keep:
            continue
        try:
            _prctl(PR_CAPBSET_DROP, cap)
        except OSError:
            pass  # cap beyond the kernel's last cap (EINVAL)

    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    mask = 0
    for cap in keep:
        mask |= 1 << cap
    hdr = Header(0x20080522, 0)
    data = (Data * 2)(
        Data(mask & 0xFFFFFFFF, mask & 0xFFFFFFFF, 0), Data(mask >> 32, mask >> 32, 0)
    )
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.capset(ctypes.byref(hdr), ctypes.byref(data)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _decide(fn, value, messages):
    try:
        fn(value)
        return "accept"
    except BadRequest as exc:
        code = exc.args[0] if exc.args else None
        return "reject:" + (
            code if isinstance(code, str) and code in messages else "<generic>"
        )
    except Exception as exc:  # noqa: BLE001 - the decision table records any failure type
        return "error:" + type(exc).__name__


def validation_table(plugin, messages, probes):
    table = []
    for name in ("validate_action", "validate_plan"):
        fn = getattr(plugin, name, None)
        if callable(fn):
            table += [
                "%s:%d:%s" % (name, i, _decide(fn, v, messages))
                for i, v in enumerate(probes[name])
            ]
    return table


def check_validation_consistency(plugin, ids, messages):
    """F1: identical accept/reject decisions (and message codes) with no episode and after a reset of
    every instance. Returns (ok, n_probes, digest)."""
    import hashlib

    extra = getattr(plugin, "VALIDATION_PROBES", {}) or {}
    probes = {
        "validate_action": GENERIC_PROBES + list(extra.get("action", [])),
        "validate_plan": GENERIC_PROBES + list(extra.get("plan", [])),
    }
    base = validation_table(plugin, messages, probes)
    ok = True
    if callable(getattr(plugin, "reset", None)):
        for inst in ids:
            plugin.reset(inst)
            if validation_table(plugin, messages, probes) != base:
                ok = False
    return ok, len(base), hashlib.sha256("\n".join(base).encode()).hexdigest()[:16]


def seccomp_selftest():
    """True when an AF_INET/AF_INET6 socket cannot be created and AF_UNIX still can."""
    blocked = True
    for fam in (socket.AF_INET, socket.AF_INET6):
        try:
            s = socket.socket(fam, socket.SOCK_STREAM)
            s.close()
            blocked = False
        except OSError:
            pass
    try:
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close()
    except OSError:
        return False
    return blocked


class Log:
    """Append-only JSON-lines log in the lab-only state dir (0600). Size-capped."""

    def __init__(self, path, cap):
        self.path, self.cap = path, cap
        self.lock = threading.Lock()
        fd = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600
        )
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            os.close(fd)
            raise RuntimeError("log path is not a plain file")
        self.fh = os.fdopen(fd, "a", buffering=1)
        self.size = st.st_size

    def write(self, **rec):
        rec = {"t": round(time.time(), 3), **rec}
        line = json.dumps(rec, default=str, separators=(",", ":")) + "\n"
        with self.lock:
            if self.size + len(line) > self.cap:
                return
            try:
                self.fh.write(line)
                self.size += len(line)
            except OSError:
                pass


# ----------------------------------------------------------------------------------------------
# configuration


def load_config(path):
    path = os.path.abspath(path)
    with open(path, "rb") as fh:
        raw = json.loads(fh.read(1 << 20))
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError("unknown config keys: %s" % sorted(unknown))
    cfg = dict(DEFAULTS)
    cfg.update(raw)
    base = os.path.dirname(path)
    for key in ("plugin", "instances"):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError("config needs %r" % key)
        cfg[key] = os.path.normpath(os.path.join(base, cfg[key]))
    if (
        not isinstance(cfg["budget"], int)
        or isinstance(cfg["budget"], bool)
        or cfg["budget"] < 0
    ):
        raise ValueError("config needs a non-negative integer 'budget'")
    for key in ("socket", "state_dir"):
        if not os.path.isabs(cfg[key]):
            raise ValueError("%s must be an absolute path" % key)
    sockdir = os.path.dirname(cfg["socket"])
    if sockdir in ("/", "/tmp", "/var/tmp", "/dev/shm", "/run"):
        raise ValueError(
            "socket must live in its own root-owned directory (e.g. /run/ale-lab/)"
        )
    if not isinstance(cfg["env"], dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in cfg["env"].items()
    ):
        raise ValueError("env must map strings to strings")
    if not isinstance(cfg["min_reply_s"], dict):
        raise ValueError("min_reply_s must be an object")
    return cfg


def _secure_dir(path, mode, uid, gid):
    """Create (or check) a directory owned by uid:gid with exactly `mode`; refuse symlinks."""
    try:
        os.mkdir(path, mode)
    except FileExistsError:
        pass
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise RuntimeError("%s exists and is not a directory" % path)
    if st.st_uid not in (0, uid):
        raise RuntimeError("%s has an unexpected owner" % path)
    os.chown(path, uid, gid)
    os.chmod(path, mode)


def _accounts(cfg):
    lab = pwd.getpwnam(cfg["user"])
    if lab.pw_uid == 0:
        raise RuntimeError("the lab account must not be root")
    try:
        agent = pwd.getpwnam(cfg["agent_user"])
    except KeyError:
        agent = None
    if agent is not None and agent.pw_uid == lab.pw_uid:
        raise RuntimeError("the lab account must differ from the agent account")
    return lab


# ----------------------------------------------------------------------------------------------
# budget


class Budget:
    """Single locked trial counter, persisted atomically on every charge (lab-only 0600 file)."""

    def __init__(self, path, log):
        self.path, self.log = path, log
        self.lock = threading.Lock()
        self.broken = False
        try:
            with open(path, "rb") as fh:
                data = json.loads(fh.read(4096))
            self.total = int(data["total"])
            self.used = int(data["used"])
            if self.total < 0 or self.used < 0:
                raise ValueError("negative")
        except Exception as exc:  # fail closed: never reset to full
            self.total, self.used, self.broken = 0, 0, True
            log.write(kind="budget_state_unreadable", err=type(exc).__name__)

    @staticmethod
    def initialize(path, total, uid, gid):
        if os.path.lexists(path):
            return False
        Budget._write(path, {"total": total, "used": 0, "version": 1})
        os.chown(path, uid, gid)
        return True

    @staticmethod
    def _write(path, data):
        tmp = path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, json.dumps(data).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        dfd = os.open(os.path.dirname(path), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    def remaining(self):
        with self.lock:
            return 0 if self.broken else max(0, self.total - self.used)

    def charge(self):
        """Returns the remaining count after charging one trial, or None when exhausted."""
        with self.lock:
            if self.broken or self.used >= self.total:
                return None
            self.used += 1
            try:
                self._write(
                    self.path, {"total": self.total, "used": self.used, "version": 1}
                )
            except OSError as exc:
                self.broken = (
                    True  # cannot persist -> stop charging rather than risk a refund
                )
                self.log.write(kind="budget_persist_failed", err=str(exc))
                raise ReplyError("internal")
            return self.total - self.used


# ----------------------------------------------------------------------------------------------
# request parsing (data only) and reply whitelisting


def _reject_constant(name):
    raise ValueError("non-finite constant")


def _finite_float(text):
    v = float(text)
    if not math.isfinite(v):
        raise ValueError("non-finite number")
    return v


def _bounded_int(text):
    if len(text) > 19:
        raise ValueError("integer too long")
    return int(text)


def _no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate key")
        out[k] = v
    return out


def _check_shape(obj, max_depth, max_items):
    stack, items = [(obj, 1)], 0
    while stack:
        cur, depth = stack.pop()
        items += 1
        if depth > max_depth or items > max_items:
            raise ReplyError("too_complex")
        if isinstance(cur, dict):
            stack.extend((v, depth + 1) for v in cur.values())
        elif isinstance(cur, list):
            stack.extend((v, depth + 1) for v in cur)


def parse_request(line, cfg):
    try:
        text = line.decode("utf-8")
        obj = json.loads(
            text,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            parse_int=_bounded_int,
            object_pairs_hook=_no_duplicates,
        )
    except ReplyError:
        raise
    except (ValueError, RecursionError):  # UnicodeDecodeError is a ValueError
        raise ReplyError("bad_json")
    if not isinstance(obj, dict):
        raise ReplyError("bad_json")
    _check_shape(obj, cfg["max_depth"], cfg["max_items"])
    op = obj.get("op")
    if not isinstance(op, str):
        raise ReplyError("bad_request")
    if op not in REQUEST_KEYS:
        raise ReplyError("unknown_op")
    if set(obj) != REQUEST_KEYS[op]:
        raise ReplyError("bad_request")
    return obj


_ARRAY_RE = re.compile(r"^(f64|f32|i64|u8)\[([0-9*]+(?:,[0-9*]+)*)\]$")


def _as_list(value):
    return value.tolist() if hasattr(value, "tolist") else value


def _flatten(value, dims, path):
    """Check a nested list against dims (ints or None for '*'); return the flat list."""
    if not dims:
        return [value]
    if not isinstance(value, (list, tuple)):
        raise TypeError("%s: expected a list" % path)
    if dims[0] is not None and len(value) != dims[0]:
        raise TypeError("%s: wrong length" % path)
    out = []
    for v in value:
        out.extend(_flatten(v, dims[1:], path))
    return out


def check_value(spec, value, path):
    """Validate one reply value against its schema spec; return the JSON-safe value."""
    if isinstance(spec, dict):
        if not isinstance(value, dict) or set(value) != set(spec):
            raise TypeError("%s: keys differ from the schema" % path)
        return {k: check_value(spec[k], value[k], path + "." + k) for k in spec}
    if not isinstance(spec, str):
        raise TypeError("%s: bad schema" % path)
    if spec == "float":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            and not hasattr(value, "__float__")
        ):
            raise TypeError("%s: expected a number" % path)
        v = float(value)
        if not math.isfinite(v):
            raise TypeError("%s: non-finite" % path)
        return v
    if spec == "int":
        if isinstance(value, bool):
            raise TypeError("%s: expected an int" % path)
        if hasattr(value, "item") and not isinstance(value, int):
            value = value.item()
        if not isinstance(value, int) or abs(value) > 2**53:
            raise TypeError("%s: expected an int" % path)
        return int(value)
    if spec == "bool":
        if hasattr(value, "item") and not isinstance(value, bool):
            value = value.item()
        if not isinstance(value, bool):
            raise TypeError("%s: expected a bool" % path)
        return value
    if spec.startswith("enum:"):
        allowed = spec[5:].split("|")
        if not isinstance(value, str) or value not in allowed:
            raise TypeError("%s: not an allowed enum value" % path)
        return value
    m = _ARRAY_RE.match(spec)
    if not m:
        raise TypeError("%s: unknown schema type" % path)
    kind = m.group(1)
    dims = [None if d == "*" else int(d) for d in m.group(2).split(",")]
    if kind == "u8":
        if (
            not hasattr(value, "tobytes")
            or not hasattr(value, "shape")
            or str(getattr(value, "dtype", "")) != "uint8"
        ):
            raise TypeError("%s: expected a uint8 array" % path)
        shape = list(value.shape)
        if len(shape) != len(dims) or any(
            d is not None and d != s for d, s in zip(dims, shape)
        ):
            raise TypeError("%s: wrong shape" % path)
        if math.prod(shape) > MAX_ARRAY_ELEMS:
            raise TypeError("%s: too large" % path)
        return {
            "dtype": "uint8",
            "shape": shape,
            "b64": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        }
    nested = _as_list(value)
    flat = _flatten(nested, dims, path)
    if len(flat) > MAX_ARRAY_ELEMS:
        raise TypeError("%s: too large" % path)
    for x in flat:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError("%s: non-numeric element" % path)
        if kind in ("f64", "f32") and not math.isfinite(x):
            raise TypeError("%s: non-finite element" % path)
        if kind == "i64" and (not isinstance(x, int) or abs(x) > 2**53):
            raise TypeError("%s: non-integer element" % path)
    if kind in ("f64", "f32"):
        return _map_nested(nested, float)
    return nested


def _map_nested(value, fn):
    if isinstance(value, (list, tuple)):
        return [_map_nested(v, fn) for v in value]
    return fn(value)


def build_payload(schema, kind, value):
    spec = schema.get(kind)
    if not isinstance(spec, dict):
        raise TypeError("schema has no %r section" % kind)
    return check_value(spec, value, kind)


# ----------------------------------------------------------------------------------------------
# worker: serving


class Trial:
    __slots__ = ("episode", "instance", "t0", "steps")

    def __init__(self, episode, instance):
        self.episode, self.instance, self.t0, self.steps = (
            episode,
            instance,
            time.monotonic(),
            0,
        )


class Worker:
    def __init__(
        self, cfg, plugin, schema, log, trials_log, budget, listener, messages=None
    ):
        self.cfg, self.plugin, self.schema = cfg, plugin, schema
        self.messages = dict(messages or {})
        self.log, self.trials_log, self.budget = log, trials_log, budget
        self.listener = listener
        self.instances = tuple(plugin.instance_ids())
        self.plugin_lock = threading.Lock()
        self.calls = {}  # thread id -> start time of the running plugin call
        self.calls_lock = threading.Lock()
        self.active = 0
        self.active_lock = threading.Lock()
        self.conn_seq = 0
        self.supported = ["hello", "budget", "close"]
        for op, attr in (("reset", "reset"), ("step", "reset"), ("execute", "execute")):
            if callable(getattr(plugin, attr, None)) and isinstance(
                schema.get(op), dict
            ):
                self.supported.append(op)

    # -- plugin calls under one lock, watched for wall-clock overruns
    def call(self, fn, *args):
        with self.plugin_lock:
            tid = threading.get_ident()
            with self.calls_lock:
                self.calls[tid] = time.monotonic()
            try:
                return fn(*args)
            finally:
                with self.calls_lock:
                    self.calls.pop(tid, None)

    def watchdog(self):
        limit = float(self.cfg["call_timeout_s"])
        while True:
            time.sleep(1.0)
            now = time.monotonic()
            with self.calls_lock:
                over = any(now - t0 > limit for t0 in self.calls.values())
            if over:
                self.log.write(kind="plugin_call_timeout", limit_s=limit)
                os._exit(
                    75
                )  # the supervisor restarts the worker; the charged trial stays charged

    def serve(self):
        threading.Thread(target=self.watchdog, daemon=True).start()
        self.listener.settimeout(None)
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError as exc:
                self.log.write(kind="accept_error", err=str(exc))
                time.sleep(0.05)
                continue
            with self.active_lock:
                admit = self.active < int(self.cfg["max_connections"])
                if admit:
                    self.active += 1
                    self.conn_seq += 1
                    cid = self.conn_seq
            if not admit:
                try:
                    conn.settimeout(1.0)
                    conn.sendall(self.error_line("busy"))
                except OSError:
                    pass
                conn.close()
                continue
            threading.Thread(target=self.handle, args=(conn, cid), daemon=True).start()

    def error_line(self, code, message=None):
        msg = ERRORS[code] if message is None else message
        return (
            json.dumps({"ok": False, "error": code, "message": msg}) + "\n"
        ).encode()

    def send(self, conn, obj):
        conn.sendall(
            (json.dumps(obj, allow_nan=False, separators=(",", ":")) + "\n").encode()
        )

    def handle(self, conn, cid):
        state = {"trial": None}
        peer_uid = -1
        try:
            try:
                peer_uid = struct.unpack(
                    "3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                )[1]
            except OSError:
                pass
            self.trials_log.write(kind="connect", conn=cid, peer_uid=peer_uid)
            conn.settimeout(float(self.cfg["idle_timeout_s"]))
            cap = int(self.cfg["max_line_bytes"])
            rate = float(self.cfg["max_requests_per_s"])
            tokens, last = rate, time.monotonic()
            buf = b""
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    if len(buf) > cap:
                        conn.sendall(self.error_line("line_too_long"))
                        return
                    chunk = conn.recv(1 << 16)
                    if not chunk:
                        return
                    buf += chunk
                    continue
                line, buf = buf[:nl], buf[nl + 1 :]
                if len(line) > cap:
                    conn.sendall(self.error_line("line_too_long"))
                    return
                if not line.strip():
                    continue
                now = time.monotonic()
                tokens = min(rate, tokens + (now - last) * rate)
                last = now
                if tokens < 1.0:
                    conn.sendall(self.error_line("rate_limited"))
                    continue
                tokens -= 1.0
                t_start = time.monotonic()
                reply, op, close = self.dispatch(line, state, cid)
                pad = self.cfg["min_reply_s"].get(op) if op else None
                if pad:
                    delay = float(pad) - (time.monotonic() - t_start)
                    if delay > 0:
                        time.sleep(delay)
                conn.sendall(reply)
                if close:
                    return
        except (OSError, socket.timeout):
            pass
        finally:
            trial = state.get("trial")
            if trial is not None:
                self.trials_log.write(
                    kind="trial_end",
                    conn=cid,
                    instance=trial.instance,
                    steps=trial.steps,
                    reason="disconnect",
                )
            try:
                conn.close()
            except OSError:
                pass
            with self.active_lock:
                self.active -= 1

    def dispatch(self, line, state, cid):
        """Returns (reply bytes, op or None, close?)."""
        op = None
        try:
            req = parse_request(line, self.cfg)
            op = req["op"]
            if op not in self.supported:
                raise ReplyError("unsupported")
            body = getattr(self, "op_" + op)(req, state, cid)
            body = {"ok": True, "op": op, **body}
            try:
                data = (
                    json.dumps(body, allow_nan=False, separators=(",", ":")) + "\n"
                ).encode()
            except (TypeError, ValueError):
                raise ReplyError("internal")
            return data, op, op == "close"
        except ReplyError as exc:
            if exc.code == "internal":
                self.log.write(kind="internal_error", conn=cid, op=op)
            return self.error_line(exc.code, exc.message), op, False
        except Exception:
            self.log.write(
                kind="internal_error",
                conn=cid,
                op=op,
                tb=traceback.format_exc(limit=20),
            )
            return self.error_line("internal"), op, False

    def _instance(self, req):
        inst = req.get("instance")
        if (
            not isinstance(inst, str)
            or not INSTANCE_RE.match(inst)
            or inst not in self.instances
        ):
            raise ReplyError("bad_instance")
        return inst

    def _payload(self, kind, value):
        try:
            return build_payload(self.schema, kind, value)
        except TypeError as exc:
            self.log.write(kind="reply_schema_violation", section=kind, detail=str(exc))
            raise ReplyError("internal")

    def _plugin_validate(self, name, value, code):
        """validate_* see only the request value (F1); messages only from the plugin's fixed table."""
        fn = getattr(self.plugin, name, None)
        if not callable(fn):
            raise ReplyError("unsupported")
        try:
            return self.call(fn, value)
        except BadRequest as exc:
            key = exc.args[0] if exc.args else None
            if isinstance(key, str) and key in self.messages:
                raise ReplyError(code, self.messages[key])
            self.log.write(kind="plugin_message_not_in_table", op=name)
            raise ReplyError(code)

    def op_hello(self, req, state, cid):
        info = (
            self._payload("info", self.call(self.plugin.info))
            if "info" in self.schema
            else {}
        )
        return {
            "protocol": PROTOCOL,
            "ops": list(self.supported),
            "instances": list(self.instances),
            "info": info,
            "limits": {
                "max_line_bytes": int(self.cfg["max_line_bytes"]),
                "trial_step_cap": int(self.cfg["trial_step_cap"]),
                "trial_wall_s": float(self.cfg["trial_wall_s"]),
                "max_connections": int(self.cfg["max_connections"]),
            },
        }

    def op_budget(self, req, state, cid):
        return {"remaining": self.budget.remaining()}

    def op_close(self, req, state, cid):
        trial = state.get("trial")
        if trial is not None:
            self.trials_log.write(
                kind="trial_end",
                conn=cid,
                instance=trial.instance,
                steps=trial.steps,
                reason="close",
            )
            state["trial"] = None
        return {}

    def _end_trial(self, state, cid, reason):
        trial = state.get("trial")
        if trial is not None:
            self.trials_log.write(
                kind="trial_end",
                conn=cid,
                instance=trial.instance,
                steps=trial.steps,
                reason=reason,
            )
        state["trial"] = None

    def _charge(self, cid, op, inst):
        remaining = self.budget.charge()
        if remaining is None:
            self.trials_log.write(
                kind="refused",
                conn=cid,
                op=op,
                instance=inst,
                reason="budget_exhausted",
            )
            raise ReplyError("budget_exhausted")
        self.trials_log.write(
            kind="charge", conn=cid, op=op, instance=inst, remaining=remaining
        )
        return remaining

    def op_reset(self, req, state, cid):
        inst = self._instance(req)
        self._end_trial(state, cid, "replaced")
        remaining = self._charge(cid, "reset", inst)
        try:
            result = self.call(self.plugin.reset, inst)
            episode, obs = result
        except Exception:
            self.log.write(
                kind="plugin_error",
                op="reset",
                conn=cid,
                tb=traceback.format_exc(limit=20),
            )
            raise ReplyError("internal")
        payload = self._payload("reset", obs)
        state["trial"] = Trial(episode, inst)
        return {"instance": inst, "obs": payload, "remaining": remaining}

    def op_step(self, req, state, cid):
        trial = state.get("trial")
        if trial is None:
            raise ReplyError("no_trial")
        if trial.steps >= int(
            self.cfg["trial_step_cap"]
        ) or time.monotonic() - trial.t0 > float(self.cfg["trial_wall_s"]):
            self._end_trial(state, cid, "limit")
            raise ReplyError("trial_over")
        trial.steps += 1  # every step request in a trial counts toward the cap, rejected or not (F1)
        action = self._plugin_validate("validate_action", req["action"], "bad_action")
        try:
            obs, done = self.call(trial.episode.step, action)
        except Exception:
            self.log.write(
                kind="plugin_error",
                op="step",
                conn=cid,
                tb=traceback.format_exc(limit=20),
            )
            self._end_trial(state, cid, "plugin_error")
            raise ReplyError("internal")
        payload = self._payload("step", obs)
        if not isinstance(done, bool):
            done = bool(done)
        if done:
            self._end_trial(state, cid, "done")
        return {"obs": payload, "done": done, "step": trial.steps}

    def op_execute(self, req, state, cid):
        inst = self._instance(req)
        plan = self._plugin_validate("validate_plan", req["plan"], "bad_plan")
        remaining = self._charge(cid, "execute", inst)
        try:
            outcome = self.call(self.plugin.execute, inst, plan)
        except Exception:
            self.log.write(
                kind="plugin_error",
                op="execute",
                conn=cid,
                tb=traceback.format_exc(limit=20),
            )
            raise ReplyError("internal")
        payload = self._payload("execute", outcome)
        return {"instance": inst, "outcome": payload, "remaining": remaining}


def run_worker(cfg, listener, ready_fd, lab):
    """Forked from the supervisor as root: load the hidden instances, drop privileges, serve."""
    _set_comm("alelab-worker")
    os.umask(0o077)
    state_dir = cfg["state_dir"]
    log = Log(os.path.join(state_dir, "lab.log"), int(cfg["max_log_bytes"]))
    trials_log = Log(os.path.join(state_dir, "trials.jsonl"), int(cfg["max_log_bytes"]))
    for fh in (log, trials_log):
        os.fchown(fh.fh.fileno(), lab.pw_uid, lab.pw_gid)
    try:
        for k, v in cfg["env"].items():
            os.environ[k] = v
        plugin_dir = os.path.dirname(cfg["plugin"])
        sys.path.insert(0, plugin_dir)
        spec = importlib.util.spec_from_file_location(
            "alelab_task_plugin", cfg["plugin"]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with open(cfg["instances"], "rb") as fh:
            blob = fh.read(int(cfg["max_instances_bytes"]) + 1)
        if len(blob) > int(cfg["max_instances_bytes"]):
            raise ValueError("instance file too large")
        instances = json.loads(blob)
        del blob
        plugin = getattr(module, cfg["plugin_factory"])(
            instances, Api(cfg["plugin_options"])
        )
        del instances
        sys.path.remove(plugin_dir)
        schema = getattr(plugin, "SCHEMA", None) or getattr(module, "SCHEMA", None)
        if not isinstance(schema, dict):
            raise TypeError("plugin has no SCHEMA dict")
        ids = list(plugin.instance_ids())
        if not ids or not all(isinstance(i, str) and INSTANCE_RE.match(i) for i in ids):
            raise TypeError("plugin instance ids must match %s" % INSTANCE_RE.pattern)
        if "info" in schema:
            build_payload(schema, "info", plugin.info())
        messages = (
            getattr(plugin, "MESSAGES", None) or getattr(module, "MESSAGES", None) or {}
        )
        if not isinstance(messages, dict) or not all(
            isinstance(k, str)
            and MESSAGE_CODE_RE.match(k)
            and isinstance(v, str)
            and 0 < len(v) <= 160
            and all(32 <= ord(c) < 127 for c in v)
            for k, v in messages.items()
        ):
            raise TypeError(
                "plugin MESSAGES must map [a-z0-9_] codes to ASCII texts of at most 160 chars"
            )
        for vname in ("validate_action", "validate_plan"):
            vfn = getattr(plugin, vname, None)
            if callable(vfn):
                params = [
                    q
                    for q in inspect.signature(vfn).parameters.values()
                    if q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)
                    and q.default is q.empty
                ]
                if len(params) != 1:
                    log.write(kind="validator_signature_invalid", name=vname)
                    raise TypeError(
                        "%s must take exactly one argument, the request value (rev-leak F1)"
                        % vname
                    )
        if cfg["validation_check"]:
            v_ok, v_n, v_digest = check_validation_consistency(plugin, ids, messages)
        else:
            v_ok, v_n, v_digest = None, 0, ""
        if v_ok is False:
            log.write(kind="validation_depends_on_state", probes=v_n)
            raise RuntimeError(
                "validate_* decisions differ between instances/states (rev-leak F1)"
            )
        # -- privilege drop (irreversible)
        os.chdir("/")
        os.setgroups([])
        os.setgid(lab.pw_gid)
        os.setuid(lab.pw_uid)
        try:
            os.setuid(0)
            raise RuntimeError("privilege drop is reversible")
        except PermissionError:
            pass
        _prctl(PR_SET_DUMPABLE, 0)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        sec_ok = seccomp_selftest()
        log.write(
            kind="worker_ready",
            uid=os.getuid(),
            gid=os.getgid(),
            seccomp_inet_blocked=sec_ok,
            instances=len(ids),
            protocol=PROTOCOL,
            validation_consistent=v_ok,
            validation_probes=v_n,
            validation_digest=v_digest,
        )
        if cfg["require_seccomp"] and not sec_ok:
            raise RuntimeError(
                "seccomp self-test failed: an AF_INET socket could be created"
            )
        budget = Budget(os.path.join(state_dir, "budget.json"), log)
        worker = Worker(
            cfg, plugin, schema, log, trials_log, budget, listener, messages
        )
    except Exception:
        log.write(kind="worker_start_failed", tb=traceback.format_exc(limit=30))
        os.write(ready_fd, b"F")
        os._exit(1)
    os.write(ready_fd, b"R")
    os.close(ready_fd)
    try:
        worker.serve()
    except Exception:
        log.write(kind="worker_crashed", tb=traceback.format_exc(limit=30))
    os._exit(1)


# ----------------------------------------------------------------------------------------------
# supervisor (root; never reads the hidden instances, never accepts a connection)


def _close_fds(keep):
    for fd in range(3, 1024):
        if fd not in keep:
            try:
                os.close(fd)
            except OSError:
                pass


def run_supervisor(cfg, lab, notify_fd):
    _set_comm("alelab-sup")
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sup_log = Log(
        os.path.join(cfg["state_dir"], "supervisor.log"), int(cfg["max_log_bytes"])
    )
    try:
        install_unix_only_seccomp()
        seccomp = True
    except OSError as exc:
        seccomp = False
        sup_log.write(kind="seccomp_install_failed", err=str(exc))
        if cfg["require_seccomp"]:
            os.write(notify_fd, b"F")
            os._exit(1)
    try:
        trim_capabilities()
        caps_trimmed = True
    except OSError as exc:
        caps_trimmed = False
        sup_log.write(kind="capability_trim_failed", err=str(exc))
    sock_path = cfg["socket"]
    if os.path.lexists(sock_path):
        os.unlink(sock_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(sock_path)
    os.chmod(
        sock_path, 0o666
    )  # before chown: without CAP_FOWNER root cannot chmod a file it no longer owns
    os.chown(sock_path, lab.pw_uid, lab.pw_gid)
    listener.listen(16)
    pidfile = os.path.join(os.path.dirname(sock_path), "supervisor.pid")
    tmp = pidfile + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("%d\n" % os.getpid())
    os.chmod(tmp, 0o644)
    os.replace(tmp, pidfile)
    sup_log.write(
        kind="supervisor_start",
        pid=os.getpid(),
        seccomp=seccomp,
        caps_trimmed=caps_trimmed,
    )

    stopping = {"flag": False, "child": None}

    def on_term(signum, frame):
        stopping["flag"] = True
        child = stopping["child"]
        if child:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    restarts, first = 0, True
    while not stopping["flag"]:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            try:
                sup_log.fh.close()
            except OSError:
                pass
            if notify_fd is not None:
                try:
                    os.close(notify_fd)
                except OSError:
                    pass
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            run_worker(cfg, listener, w, lab)
            os._exit(1)
        os.close(w)
        stopping["child"] = pid
        status = b""
        try:
            status = os.read(r, 1)
        except OSError:
            pass
        os.close(r)
        if first:
            first = False
            if notify_fd is not None:
                try:
                    os.write(notify_fd, b"R" if status == b"R" else b"F")
                    os.close(notify_fd)
                except OSError:
                    pass
                notify_fd = None
        while True:
            try:
                _, code = os.waitpid(pid, 0)
                break
            except ChildProcessError:
                code = -1
                break
            except InterruptedError:
                continue
        stopping["child"] = None
        sup_log.write(kind="worker_exit", pid=pid, status=code, ready=status == b"R")
        if stopping["flag"]:
            break
        if status != b"R":
            sup_log.write(kind="worker_failed_to_start")
            break
        restarts += 1
        if restarts > int(cfg["max_restarts"]):
            sup_log.write(kind="restart_limit_reached", restarts=restarts - 1)
            break
        time.sleep(0.5)
    try:
        listener.close()
        if os.path.lexists(sock_path):
            os.unlink(sock_path)
        os.unlink(pidfile)
    except OSError:
        pass
    sup_log.write(kind="supervisor_exit", restarts=restarts)
    os._exit(0)


def start(config_path):
    if os.geteuid() != 0:
        print("alelab: must be started as root (from environment/setup/setup.sh)", file=sys.stderr)
        return 2
    try:
        cfg = load_config(config_path)
        lab = _accounts(cfg)
        os.umask(0o077)
        sockdir = os.path.dirname(cfg["socket"])
        os.makedirs(os.path.dirname(sockdir), exist_ok=True)
        _secure_dir(sockdir, 0o755, 0, 0)
        os.makedirs(os.path.dirname(cfg["state_dir"]), exist_ok=True)
        _secure_dir(cfg["state_dir"], 0o700, lab.pw_uid, lab.pw_gid)
        Budget.initialize(
            os.path.join(cfg["state_dir"], "budget.json"),
            cfg["budget"],
            lab.pw_uid,
            lab.pw_gid,
        )
    except Exception as exc:
        print("alelab: configuration error: %s" % type(exc).__name__, file=sys.stderr)
        return 1
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        os.setsid()
        if os.fork() > 0:
            os._exit(0)
        devnull = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(devnull, fd)
        if devnull > 2:
            os.close(devnull)
        _close_fds({w})
        os.chdir("/")
        for k in list(os.environ):
            if k not in ("PATH", "LANG", "HOME"):
                del os.environ[k]
        try:
            run_supervisor(cfg, lab, w)
        except Exception:
            try:
                Log(
                    os.path.join(cfg["state_dir"], "supervisor.log"),
                    int(cfg["max_log_bytes"]),
                ).write(kind="supervisor_crashed", tb=traceback.format_exc(limit=20))
            except Exception:  # noqa: BLE001
                pass
        finally:
            os._exit(1)
    os.close(w)
    os.waitpid(pid, 0)
    deadline = time.monotonic() + float(cfg["start_timeout_s"])
    status = b""
    while time.monotonic() < deadline:
        ready, _, _ = select.select([r], [], [], max(0.0, deadline - time.monotonic()))
        if ready:
            status = os.read(r, 1)
            break
    os.close(r)
    if status == b"R":
        print("alelab: ready (socket %s)" % cfg["socket"], file=sys.stderr)
        return 0
    print("alelab: failed to start (details in the lab-only log)", file=sys.stderr)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="alelab lab service (start as root from environment/setup/setup.sh)"
    )
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    return start(args.config)


if __name__ == "__main__":
    sys.exit(main())
