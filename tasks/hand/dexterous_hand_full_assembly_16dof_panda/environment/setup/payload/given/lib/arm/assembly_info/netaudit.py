"""The network audit of a program executed at verify (grader side, stdlib only).

Verify runs with open egress (engine), so a program the grader executes could download the public arm or fetch material a
background process pushed out before the lock. Three checks, all feeding the contract gate:
  * static (`scan_imports`; one file's bytes: `source_hits`): the submission's own .py files (files byte-identical to a
    shipped library file exempt, `allow_digests`) import no networking module (`NETWORK_MODULES`, incl.
    `multiprocessing.connection`, `logging.handlers`, `xmlrpc`, `ctypes`), load no module under a computed name
    (`__import__` / `importlib.import_module` / `importlib.util.find_spec` with a non-literal argument), execute no
    computed code (`exec` / `eval` / `compile` with a non-literal argument) and start no other program (`subprocess`,
    `os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `os.posix_spawn*`, `pty`); every hit is listed. It fails closed: a
    .py file that cannot be parsed (a syntax error, a null byte, bytes its declared or default encoding cannot decode,
    nesting too deep for the parser, too little memory) is a hit `cannot be parsed (<error>)`, and so is a file that
    cannot be read; sizes are checked (lstat) before anything is read and, once the files together pass `max_bytes`,
    the scan stops with the hit `beyond the audit's size limit`; only regular files are read (symbolic links, FIFOs,
    devices and sockets are skipped: the program's run folder never holds them).
  * source encodings (`encoding_problem`): a .py file may declare only UTF-8, ASCII or Latin-1 as its text encoding (a
    PEP 263 coding line in the first two lines; a UTF-8 BOM is fine; no declaration means UTF-8). A file that declares
    any other known codec is a hit `declares the text encoding <name>; sources must be UTF-8, ASCII or Latin-1`,
    decided from the first two lines ALONE, before the bytes are decoded with that codec anywhere: a codec such as
    punycode or utf-7 is a pure-Python, quadratic decoder, so a large file that declared it would make a decode (in the
    grader's route gate or in the parser) outlast the verifier. An unknown codec name is left to the parser, which
    reports it as a syntax error (`cannot be parsed (SyntaxError)`). `.sh` files are not parsed and declare nothing: the
    grader reads them as UTF-8 with replacement whatever they contain.
    The grader itself does not decode, split or parse a submission's bytes: it runs `source_hits` / `tree_hits`, the
    encoding gate and the route gate's string checks in a separate process with a memory and a time limit per file, and
    a whole-audit time limit, and refuses a file that does not fit for that reason, not as a networking import.
    `scan_imports` parses in the calling process, with whatever memory that process has: Python's parser needs a few
    hundred times a dense file's size in memory. Bulk data belongs in a data file (for example a .npy or a .json file),
    not in a .py file.
  * runtime, namespace-wide (`NetCounters`): the container network namespace's counters of outbound activity (TCP active
    opens, UDP / UDP-Lite datagrams sent, ICMP echo requests sent; v4 and v6: /proc/net/snmp, /proc/net/snmp6) read before
    and after every agent episode; the grader opens no socket while the program runs (the wire is a pipe pair), so any
    increase is the program's (or a process it started): stat `net_*`, gate 0. This catches a connect / close shorter than
    any sampling period, a `curl` launched from run.sh and DNS lookups.
  * runtime, per process (`SocketSampler`): the remote inet sockets (tcp / udp, v4 / v6, neither loopback nor unspecified)
    of every process of the runner account, sampled every `period_s`; any sample > 0 zeroes the gate (stat
    `remote_sockets_seen`). With `oom_floor` (run as root) each pass also KILLS (SIGKILL) any of these processes whose
    `oom_score_adj` reads below that floor and counts the kills (`oom_killed`; a record only). Every process of the
    program is started at the floor (`choom`, which writes the value and then execs the interpreter, same pid) and a
    legitimate program never lowers it; a process found below the floor has lowered its own value, so it is stopped (it
    then ends as a crash under the episode's rules) rather than raised. The launch chain's own helpers (`runuser`,
    `prlimit`, `choom`), which run as the account for a few milliseconds at the inherited value before `choom` writes
    the floor, are told apart by their executable name and are never killed, so an ordinary launch is never stopped.
"""

import ast
import codecs
import hashlib
import io
import os
import signal
import stat
import threading
import time
import tokenize

__all__ = ["NETWORK_MODULES", "PROCESS_CALLS", "scan_imports", "source_hits", "tree_hits", "encoding_problem",
           "scan_encoding", "ENCODING_ALLOW", "SocketSampler", "NetCounters", "net_counters", "count_remote_sockets",
           "uid_pids"]

NETWORK_MODULES = ("socket", "_socket", "ssl", "_ssl", "urllib", "urllib2", "urllib3", "http", "httplib", "requests", "httpx", "aiohttp", "ftplib",
                   "smtplib", "imaplib", "poplib", "nntplib", "telnetlib", "websocket", "websockets", "paramiko", "pycurl", "xmlrpc", "asyncio",
                   "socketserver", "grpc", "zmq", "huggingface_hub", "boto3", "botocore", "wget", "ctypes", "_ctypes", "cffi", "selectors",
                   "multiprocessing.connection", "multiprocessing.managers", "logging.handlers", "subprocess", "pty", "_posixsubprocess")
# calls that start another program or run computed code (attribute or plain names; `os.` calls by their attribute)
PROCESS_CALLS = ("system", "popen", "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe", "spawnv", "spawnve",
                 "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe", "posix_spawn", "posix_spawnp", "fork_exec")
DYNAMIC_IMPORT_CALLS = ("__import__", "import_module", "find_spec", "spec_from_file_location", "load_module")
CODE_CALLS = ("exec", "eval", "compile")
# the most bytes of .py files one scan reads (the submission size cap of the route gate; verify.py passes its own)
MAX_BYTES = 60 * 1024 * 1024


# ------------------------------------------------------------------------------------------------ source encodings
# The only text encodings a .py source may declare (PEP 263): UTF-8 (the default; a BOM is fine), ASCII, Latin-1. Any
# other known codec is refused, from the declaration alone, before the bytes are decoded with it: pure-Python codecs
# (punycode, utf-7, ...) decode in time quadratic in the file's length, so a large file that declared one would make a
# decode outlast the verifier (it did: a 2 MB punycode .sh took 1063 s in the grader's route gate before this gate).
def _normal(name):
    try:
        return codecs.lookup(name).name
    except (LookupError, TypeError, ValueError):
        return None


ENCODING_ALLOW = frozenset(n for n in (_normal(x) for x in ("utf-8", "utf-8-sig", "ascii", "latin-1")) if n)
ENCODING_ALLOW_LABEL = "UTF-8, ASCII or Latin-1"


def declared_encoding(data):
    """The text encoding a .py file declares (PEP 263, in its first two lines; a UTF-8 BOM gives 'utf-8-sig'), as the
    name `tokenize.detect_encoding` reports, or None when it declares nothing (the default UTF-8) or names a codec the
    standard library does not know (left to the parser, which raises a SyntaxError). Reads the first two lines only and
    never decodes the body."""
    head = data[:8192] if isinstance(data, (bytes, bytearray)) else data
    try:
        return tokenize.detect_encoding(io.BytesIO(bytes(head)).readline)[0]
    except SyntaxError:
        return None
    except Exception:  # noqa: BLE001 - any other failure of the probe: leave it to the parser
        return None


def encoding_problem(rel, data):
    """(rel, reason) when a .py file declares a known text encoding that is not UTF-8 / ASCII / Latin-1, else None.
    Decided from the declaration alone; the bytes are not decoded with the declared codec here."""
    enc = declared_encoding(data)
    if enc is None:
        return None
    canon = _normal(enc)
    if canon is None or canon in ENCODING_ALLOW:
        return None
    return (rel, "declares the text encoding %s; sources must be %s" % (enc, ENCODING_ALLOW_LABEL))


def scan_encoding(data):
    """The codec to decode a .py file's bytes with for the route gate's string checks: its declared encoding when that
    is one of UTF-8 / ASCII / Latin-1, else 'utf-8' (the default, and the fallback for a declaration the gate has
    already refused). Never raises."""
    enc = declared_encoding(data)
    canon = _normal(enc) if enc is not None else None
    return enc if (canon is not None and canon in ENCODING_ALLOW) else "utf-8"


def _hit(name):
    name = name or ""
    parts = name.split(".")
    return any(".".join(parts[:k]) in NETWORK_MODULES for k in range(1, len(parts) + 1))


def _literal(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def scan_imports(root, allow_digests=(), max_bytes=MAX_BYTES):
    """[(relpath, module)] of networking imports in the .py files under `root` (regular files only). A file that cannot
    be read or parsed is a hit (`cannot be read`, `cannot be parsed (<error>)`), and a file that declares a text
    encoding other than UTF-8 / ASCII / Latin-1 is a hit `declares the text encoding <name>; ...`; once the files'
    sizes (lstat, before the read) pass `max_bytes`, the file that passes it is a hit (`beyond the audit's size limit`)
    and the scan stops. Parses in this process (no memory or time limit of its own)."""
    allow = set(allow_digests or ())
    out = []
    left = None if max_bytes is None else int(max_bytes)
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if left is not None:
                left -= st.st_size
                if left < 0:
                    out.append((rel, "beyond the audit's size limit"))
                    return out
            try:
                with open(p, "rb") as fh:
                    data = fh.read(st.st_size)
            except OSError:
                out.append((rel, "cannot be read"))
                continue
            if hashlib.sha256(data).hexdigest() in allow:
                continue
            out += source_hits(rel, data)
    return out


def source_hits(rel, data):
    """[(rel, what)] of one .py file's bytes: its hits, the one hit `declares the text encoding <name>; ...` when it
    declares a text encoding that is not UTF-8 / ASCII / Latin-1 (decided before the body is decoded), or the one hit
    `cannot be parsed (<error>)` when the parser refuses the bytes (read as Python reads them, PEP 263: the declared
    encoding, else UTF-8)."""
    enc = encoding_problem(rel, data)
    if enc is not None:
        return [enc]
    try:
        tree = ast.parse(data)
    except Exception as e:  # noqa: BLE001 - SyntaxError, ValueError, RecursionError, MemoryError, ...: fail closed
        return [(rel, "cannot be parsed (%s)" % type(e).__name__)]
    return tree_hits(rel, tree)


def tree_hits(rel, tree):
    """[(rel, what)] of one parsed file (`ast.parse` of its bytes)."""
    out = []
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


# the launch chain's own helpers run as the runner account for a few milliseconds, at the inherited oom_score_adj,
# before `choom` writes the floor and execs the interpreter: told apart by their executable name so they are never
# killed (the interpreter the account finally runs has already had the floor written by `choom`).
LAUNCH_CHAIN_COMMS = frozenset(("runuser", "prlimit", "choom"))


def _proc_comm(pid):
    try:
        with open("/proc/%d/comm" % pid) as fh:
            return fh.read().strip()
    except OSError:
        return ""


class SocketSampler(object):
    """Samples the remote inet sockets of every process of `uid` (or of the `pids_fn()` list) every `period_s`. With
    `oom_floor`, each pass first KILLS (SIGKILL) any of these processes whose `oom_score_adj` reads below that floor and
    counts the kills in `oom_killed`: every process of the program is started at the floor and a legitimate program
    never lowers it, so one found below it has lowered its own value and is stopped (it then ends as a crash under the
    episode's rules). The launch chain's own helpers (runuser / prlimit / choom), which briefly run as the account at
    the inherited value before choom writes the floor, are spared by their executable name."""

    def __init__(self, uid=None, pids_fn=None, period_s=0.5, oom_floor=None):
        self.uid, self.pids_fn, self.period_s = uid, pids_fn, float(period_s)
        self.oom_floor = None if oom_floor is None else int(oom_floor)
        self.max_seen = 0
        self.samples = 0
        self.oom_killed = 0
        self._stop = threading.Event()
        self._t = None

    def _pids(self):
        if self.pids_fn is not None:
            return list(self.pids_fn())
        return uid_pids(self.uid) if self.uid is not None else []

    def _enforce_oom(self, pids):
        for pid in pids:
            try:
                with open("/proc/%d/oom_score_adj" % pid) as fh:
                    v = int(fh.read().strip())
            except (OSError, ValueError):
                continue
            if v >= self.oom_floor:
                continue
            if _proc_comm(pid) in LAUNCH_CHAIN_COMMS:
                continue                                # the launch chain before choom's write: spared
            try:
                os.kill(pid, signal.SIGKILL)
                self.oom_killed += 1
            except OSError:
                continue

    def _run(self):
        while not self._stop.is_set():
            try:
                pids = self._pids()
            except Exception:  # noqa: BLE001
                pids = []
            if self.oom_floor is not None:
                self._enforce_oom(pids)
            try:
                n = count_remote_sockets(pids)
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
