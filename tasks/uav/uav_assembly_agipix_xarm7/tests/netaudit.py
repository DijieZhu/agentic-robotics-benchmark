"""Network audit of the submitted program, which runs during verify (grader side, stdlib only).

The engine opens the network before the verify stage, and the submitted program runs during it (as `nobody`, through
cell_runner.py), so it could download material (around the submission's size cap) or reach anything public while it is
being scored. Three checks, any hit voids the submission (verify.py):
  * static (`scan_imports`): the submission's own .py files (files byte-identical to a shipped one exempt,
    `allow_digests`) import no networking module (`NETWORK_MODULES`, incl. `multiprocessing.connection`,
    `logging.handlers`, `xmlrpc`, `ctypes`, `subprocess`), load no module under a computed name (`__import__` /
    `importlib.import_module` / `importlib.util.find_spec` with a non-literal argument), execute no computed code
    (`exec` / `eval` / `compile` with a non-literal argument) and start no other program (`os.system`, `os.popen`,
    `os.exec*`, `os.spawn*`, `os.posix_spawn*`); every hit is listed. Each file is parsed from its bytes, so its PEP 263
    coding declaration is honoured exactly as the interpreter honours it (a `# coding: utf-7` file decoded as UTF-8 hides
    an `import socket` in a comment). The scan is a first line only: an attribute path to a module (one that imports
    `socket` itself), an aliased loader, compiled bytecode or a native extension pass it, which is why the runtime checks
    exist;
  * runtime, namespace-wide (`NetCounters`): the container network namespace's counters of outbound activity (TCP
    active opens, UDP / UDP-Lite datagrams sent, ICMP echo requests sent; v4 and v6: /proc/net/snmp, /proc/net/snmp6) read
    before and after every episode; the grader opens no socket while the program runs (the connection is a pipe pair), so
    any increase is the program's or a process it started. This catches a connect / close shorter than any sampling
    period and DNS lookups;
  * runtime, per account (`SocketSampler`): the remote inet sockets (tcp / udp, v4 / v6, neither loopback nor
    unspecified) of the runner account, sampled every `period_s`. A socket counts when the owner-uid column of
    /proc/net/{tcp,tcp6,udp,udp6} names the account, or when one of the account's processes holds it (its
    /proc/<pid>/fd links). The owner uid is the part that works in the engine's sandbox: there root has no CAP_SYS_PTRACE,
    so the kernel refuses it another account's /proc/<pid>/fd links.

Ported from robotic_arm_full_assembly_openarm_panda (tests/arm/assembly_info/netaudit.py): the same module and call
lists and counters; this copy parses the bytes and adds the owner-uid socket count.
"""

import ast
import hashlib
import os
import threading

__all__ = ["NETWORK_MODULES", "PROCESS_CALLS", "scan_imports", "SocketSampler", "NetCounters", "net_counters",
           "count_remote_sockets", "remote_sockets", "uid_pids"]

NETWORK_MODULES = ("socket", "_socket", "ssl", "_ssl", "urllib", "urllib2", "urllib3", "http", "httplib", "requests", "httpx",
                   "aiohttp", "ftplib", "smtplib", "imaplib", "poplib", "nntplib", "telnetlib", "websocket", "websockets",
                   "paramiko", "pycurl", "xmlrpc", "asyncio", "socketserver", "grpc", "zmq", "huggingface_hub", "boto3",
                   "botocore", "wget", "ctypes", "_ctypes", "cffi", "selectors", "multiprocessing.connection",
                   "multiprocessing.managers", "logging.handlers", "subprocess", "pty", "_posixsubprocess")
# calls that start another program (attribute or plain names; `os.` calls by their attribute)
PROCESS_CALLS = ("system", "popen", "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe", "spawnv",
                 "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe", "posix_spawn", "posix_spawnp",
                 "fork_exec")
DYNAMIC_IMPORT_CALLS = ("__import__", "import_module", "find_spec", "spec_from_file_location", "load_module")
CODE_CALLS = ("exec", "eval", "compile")


def _hit(name):
    name = name or ""
    parts = name.split(".")
    return any(".".join(parts[:k]) in NETWORK_MODULES for k in range(1, len(parts) + 1))


def _literal(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def scan_imports(root, allow_digests=()):
    """[(relpath, finding)] of the forbidden forms in the .py files under `root`, every folder included (a program can
    put any of them on sys.path); symbolic links skipped."""
    allow = set(allow_digests or ())
    out = []
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            if os.path.islink(p):
                continue
            try:
                with open(p, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            if hashlib.sha256(data).hexdigest() in allow:
                continue
            rel = os.path.relpath(p, root)
            try:
                tree = ast.parse(data)          # bytes: the coding declaration applies, as on import
            except (SyntaxError, ValueError):
                continue                        # the interpreter cannot import it either
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
_SNMP_KEYS = {("Tcp", "ActiveOpens"): "tcp_active_opens", ("Udp", "OutDatagrams"): "udp_out",
              ("UdpLite", "OutDatagrams"): "udplite_out", ("Icmp", "OutEchos"): "icmp_echo_out"}
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


# ------------------------------------------------------------------------------------------------ per-account sockets
def _proc_net_rows(path):
    """[(remote address, owner uid, inode)] of a /proc/net/{tcp,tcp6,udp,udp6} table."""
    out = []
    try:
        with open(path) as fh:
            next(fh, None)
            for line in fh:
                f = line.split()
                if len(f) >= 10:
                    out.append((f[2], f[7], f[9]))
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
        return ip.upper() == "00000000000000000000000001000000" or (
            ip[:24].upper() == "0000000000000000FFFF0000" and ip[-2:].upper() == "7F")
    return False


def _held_inodes(pids):
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
                continue                        # another account's links need CAP_SYS_PTRACE
            if target.startswith("socket:[") and target.endswith("]"):
                inodes.add(target[8:-1])
    return inodes


def remote_sockets(uid=None, pids=(), proc_net="/proc/net"):
    """Remote (neither loopback nor unspecified) inet sockets owned by `uid` or held by one of `pids`."""
    inodes = _held_inodes(pids)
    owner = None if uid is None else str(int(uid))
    n = 0
    for name in ("tcp", "tcp6", "udp", "udp6"):
        for rem, row_uid, ino in _proc_net_rows(os.path.join(proc_net, name)):
            if (row_uid == owner or ino in inodes) and not _is_loopback_or_unspecified(rem):
                n += 1
    return n


def count_remote_sockets(pids):
    """Remote inet sockets held by `pids` (the siblings' method; blind to another account without CAP_SYS_PTRACE)."""
    return remote_sockets(None, pids)


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
    """Samples the remote inet sockets of `uid` (owned, or held by its processes) or of the `pids_fn()` list every
    `period_s`."""

    def __init__(self, uid=None, pids_fn=None, period_s=0.5):
        self.uid, self.pids_fn, self.period_s = uid, pids_fn, float(period_s)
        self.max_seen = 0
        self.samples = 0
        self._stop = threading.Event()
        self._t = None

    def count(self):
        if self.pids_fn is not None:
            return remote_sockets(None, list(self.pids_fn()))
        if self.uid is None:
            return 0
        return remote_sockets(self.uid, uid_pids(self.uid))

    def _run(self):
        while not self._stop.is_set():
            try:
                n = self.count()
            except Exception:  # noqa: BLE001 - the sampler must never take the grader down
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
