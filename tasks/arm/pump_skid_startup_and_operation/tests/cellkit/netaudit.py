"""The T10 network audit of a program executed at verify (PLAN.md T10, E-ASM-5; grader side, stdlib only).

Verify runs with open egress (engine), so a program the grader executes could download the public arm or fetch material a
background process pushed out before the lock. Three checks, all feeding the contract gate:
  * static (`scan_imports`): the submission's own .py files (files byte-identical to a shipped library file exempt,
    `allow_digests`) import no networking module (`NETWORK_MODULES`, incl. `multiprocessing.connection`, `logging.handlers`,
    `xmlrpc`, `ctypes`), load no module under a computed name (`__import__` / `importlib.import_module` /
    `importlib.util.find_spec` with a non-literal argument), execute no computed code (`exec` / `eval` / `compile` with a
    non-literal argument) and start no other program (`subprocess`, `os.system`, `os.popen`, `os.exec*`, `os.spawn*`,
    `os.posix_spawn*`, `pty`); every hit is listed
  * runtime, namespace-wide (`NetCounters`): the container network namespace's counters of outbound activity (TCP active
    opens, UDP / UDP-Lite datagrams sent, ICMP echo requests sent; v4 and v6: /proc/net/snmp, /proc/net/snmp6) read before
    and after every agent episode; the grader opens no socket while the program runs (the wire is a pipe pair), so any
    increase is the program's (or a process it started): stat `net_*`, gate 0. This catches a connect / close shorter than
    any sampling period, a `curl` launched from run.sh and DNS lookups.
  * runtime, per process (`SocketSampler`): the remote inet sockets (tcp / udp, v4 / v6, neither loopback nor unspecified)
    of every process of the runner account, sampled every `period_s`; any sample > 0 zeroes the gate (stat
    `remote_sockets_seen`)
"""

import ast
import hashlib
import os
import threading
import time

__all__ = ["NETWORK_MODULES", "PROCESS_CALLS", "scan_imports", "SocketSampler", "NetCounters", "net_counters", "count_remote_sockets", "uid_pids"]

NETWORK_MODULES = ("socket", "_socket", "ssl", "_ssl", "urllib", "urllib2", "urllib3", "http", "httplib", "requests", "httpx", "aiohttp", "ftplib",
                   "smtplib", "imaplib", "poplib", "nntplib", "telnetlib", "websocket", "websockets", "paramiko", "pycurl", "xmlrpc", "asyncio",
                   "socketserver", "grpc", "zmq", "huggingface_hub", "boto3", "botocore", "wget", "ctypes", "_ctypes", "cffi", "selectors",
                   "multiprocessing.connection", "multiprocessing.managers", "logging.handlers", "subprocess", "pty", "_posixsubprocess")
# calls that start another program or run computed code (attribute or plain names; `os.` calls by their attribute)
PROCESS_CALLS = ("system", "popen", "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe", "spawnv", "spawnve",
                 "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe", "posix_spawn", "posix_spawnp", "fork_exec")
DYNAMIC_IMPORT_CALLS = ("__import__", "import_module", "find_spec", "spec_from_file_location", "load_module")
CODE_CALLS = ("exec", "eval", "compile")


def _hit(name):
    name = name or ""
    parts = name.split(".")
    return any(".".join(parts[:k]) in NETWORK_MODULES for k in range(1, len(parts) + 1))


def _literal(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def scan_imports(root, allow_digests=()):
    """[(relpath, module)] of networking imports in the .py files under `root` (symbolic links skipped)."""
    allow = set(allow_digests or ())
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            if os.path.islink(p):
                continue
            try:
                data = open(p, "rb").read()
            except OSError:
                continue
            if hashlib.sha256(data).hexdigest() in allow:
                continue
            rel = os.path.relpath(p, root)
            try:
                tree = ast.parse(data.decode("utf-8", "replace"))
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    out += [(rel, a.name) for a in n.names if _hit(a.name)]
                elif isinstance(n, ast.ImportFrom):
                    if n.level == 0 and _hit(n.module):
                        out.append((rel, n.module))
                    elif n.level == 0 and n.module and any(_hit(n.module + "." + a.name) for a in n.names):
                        out.append((rel, n.module + "." + ",".join(a.name for a in n.names)))
                elif isinstance(n, ast.Call):
                    f = n.func
                    fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
                    owner = f.value.id if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else ""
                    arg0 = n.args[0] if n.args else None
                    if fname in DYNAMIC_IMPORT_CALLS:
                        if arg0 is None or not _literal(arg0):
                            if fname not in ("spec_from_file_location", "load_module"):
                                out.append((rel, "%s(<computed name>)" % fname))
                        elif _hit(arg0.value):
                            out.append((rel, arg0.value))
                    elif fname in CODE_CALLS and isinstance(f, ast.Name) and (arg0 is None or not _literal(arg0)):
                        out.append((rel, "%s(<computed code>)" % fname))
                    elif fname in PROCESS_CALLS and (owner in ("os", "posix", "_posixsubprocess") or isinstance(f, ast.Name)):
                        out.append((rel, "%s.%s (starts a program)" % (owner or "", fname)))
    return out


# ------------------------------------------------------------------------------------------------ namespace-wide counters
_SNMP_KEYS = {("Tcp", "ActiveOpens"): "tcp_active_opens", ("Udp", "OutDatagrams"): "udp_out", ("UdpLite", "OutDatagrams"): "udplite_out",
              ("Icmp", "OutEchos"): "icmp_echo_out"}
_SNMP6_KEYS = {"Udp6OutDatagrams": "udp6_out", "UdpLite6OutDatagrams": "udplite6_out", "Icmp6OutEchos": "icmp6_echo_out"}


def net_counters(snmp="/proc/net/snmp", snmp6="/proc/net/snmp6"):
    """{counter: int} of this network namespace's outbound activity (TCP active opens count v4 and v6)."""
    out = {v: 0 for v in list(_SNMP_KEYS.values()) + list(_SNMP6_KEYS.values())}
    try:
        with open(snmp) as fh:
            lines = [l.split() for l in fh if l.strip()]
        for a, b in zip(lines[0::2], lines[1::2]):
            proto = a[0].rstrip(":")
            for name, val in zip(a[1:], b[1:]):
                k = _SNMP_KEYS.get((proto, name))
                if k:
                    out[k] = int(val)
    except (OSError, ValueError, IndexError):
        out["unreadable"] = 1
    try:
        with open(snmp6) as fh:
            for l in fh:
                f = l.split()
                if len(f) == 2 and f[0] in _SNMP6_KEYS:
                    out[_SNMP6_KEYS[f[0]]] = int(f[1])
    except (OSError, ValueError):
        pass
    return out


class NetCounters(object):
    """`with NetCounters() as nc: <run the program>` -> nc.delta {counter: increase}, nc.total (sum of the increases)."""

    def __init__(self, reader=net_counters):
        self.reader = reader
        self.before = self.after = None
        self.delta = {}
        self.total = 0

    def __enter__(self):
        self.before = self.reader()
        return self

    def __exit__(self, *exc):
        self.after = self.reader()
        self.delta = {k: int(self.after.get(k, 0)) - int(self.before.get(k, 0)) for k in self.before if k != "unreadable"}
        self.total = sum(max(0, v) for v in self.delta.values())
        return False


def _parse_proc_net(path):
    out = {}
    try:
        with open(path) as fh:
            next(fh, None)
            for line in fh:
                f = line.split()
                if len(f) >= 10:
                    out[f[9]] = f[2]
    except OSError:
        pass
    return out


def _is_loopback_or_unspecified(rem):
    ip, _, port = rem.partition(":")
    if port == "0000" or set(ip) <= {"0"}:
        return True
    if len(ip) == 8:
        return ip[-2:].upper() == "7F"
    if len(ip) == 32:
        return ip.upper() == "00000000000000000000000001000000" or (ip[:24].upper() == "0000000000000000FFFF0000" and ip[-2:].upper() == "7F")
    return False


def count_remote_sockets(pids):
    """Remote (non-loopback) inet sockets held by `pids` (the method of `behavior/wire.count_remote_sockets`)."""
    inodes = set()
    for pid in pids:
        fd_dir = "/proc/{}/fd".format(pid)
        try:
            names = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in names:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                inodes.add(target[8:-1])
    if not inodes:
        return 0
    n = 0
    for name in ("tcp", "tcp6", "udp", "udp6"):
        table = _parse_proc_net("/proc/net/" + name)
        for ino in inodes:
            rem = table.get(ino)
            if rem is not None and not _is_loopback_or_unspecified(rem):
                n += 1
    return n


def uid_pids(uid):
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if os.stat("/proc/" + d).st_uid == uid:
                out.append(int(d))
        except OSError:
            continue
    return out


class SocketSampler(object):
    """Samples the remote inet sockets of every process of `uid` (or of the `pids_fn()` list) every `period_s`."""

    def __init__(self, uid=None, pids_fn=None, period_s=0.5):
        self.uid, self.pids_fn, self.period_s = uid, pids_fn, float(period_s)
        self.max_seen = 0
        self.samples = 0
        self._stop = threading.Event()
        self._t = None

    def _pids(self):
        if self.pids_fn is not None:
            return list(self.pids_fn())
        return uid_pids(self.uid) if self.uid is not None else []

    def _run(self):
        while not self._stop.is_set():
            try:
                n = count_remote_sockets(self._pids())
            except Exception:  # noqa: BLE001
                n = 0
            self.samples += 1
            self.max_seen = max(self.max_seen, n)
            self._stop.wait(self.period_s)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=5.0)
        return False
