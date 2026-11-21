"""The submission's static audit and digest: the reference implementation of the gates `submission_ok` (the static
part) and `no_late_write`. The task body's verify.py calls these; what
they cannot see (a process that survives its venue, a network attempt, reading hidden files) is the isolation's
job.

    rep = audit_submission("/home/user/submission", config["route_gate"])   # {"ok", "reasons", "warnings", ...}
    d = submission_digest("/home/user/submission")                           # sealed at session end, compared at verify

`submission_ok` (static part) is true exactly when:

- `drive.py` exists as a regular file and defines `solve` at top level (a `def`, or a name bound by assignment or by
  `from ... import solve`);
- every file is a regular file (no symbolic links, devices, sockets), the total is at most `max_submission_mb`;
- no file has an extension of `forbidden_extensions` (weights and model files);
- no Python file contains a `forbidden_calls` text (training calls: `torch.optim`, `.backward(`, `jax.grad`);
- no Python file imports a module that starts processes or opens the network (`PROCESS_MODULES`, `NETWORK_MODULES`)
  or a foreign-function interface (`FFI_MODULES`), calls a process function of `os` (`OS_PROCESS_CALLS`), makes a
  computed import (`__import__` / `importlib.import_module` / `importlib.__import__` with a non-literal name, or
  `importlib.util.spec_from_file_location` / `module_from_spec` at all), or calls `exec`, `eval` or `compile` with a
  non-literal first argument; every Python file parses.

What a static audit cannot exclude (`getattr(os, name)` with a computed name, code hidden in data and run by a
literal-free path the audit does not know): those are warnings here (`COMPUTED_ATTR` on the modules above) and the
isolation's to stop (no fork / no network in the sandbox's seccomp profile, the program's own account).
"""

import ast
import hashlib
import os
import stat

PROCESS_MODULES = ("subprocess", "multiprocessing", "pty", "pexpect", "sh", "plumbum", "concurrent.futures.process")
NETWORK_MODULES = ("socket", "ssl", "http", "urllib", "urllib3", "requests", "httpx", "aiohttp", "ftplib", "smtplib",
                   "telnetlib", "poplib", "imaplib", "nntplib", "socketserver", "xmlrpc", "websocket", "websockets",
                   "paramiko", "asyncssh", "grpc", "zmq")
FFI_MODULES = ("ctypes", "cffi", "_ctypes")
OS_PROCESS_CALLS = ("system", "popen", "fork", "forkpty", "vfork", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv",
                    "spawnve", "spawnvp", "spawnvpe", "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp",
                    "execvpe", "posix_spawn", "posix_spawnp", "setsid", "daemon", "register_at_fork")
PROCESS_CLASSES = ("ProcessPoolExecutor",)
DEFAULT_ROUTE_GATE = {"forbidden_calls": ["torch.optim", ".backward(", "jax.grad"],
                      "forbidden_extensions": [".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".h5", ".pb", ".pkl", ".npz"],
                      "max_submission_mb": 60}


def _mod_hit(name, banned):
    return any(name == b or name.startswith(b + ".") for b in banned)


def _literal(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))


def _call_name(func):
    """Dotted name of a call target (`os.fork`, `importlib.import_module`, `exec`) or None."""
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
        return ".".join(reversed(parts))
    return None


def _defines_solve(tree):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "solve" for t in node.targets):
            return True
        if isinstance(node, ast.ImportFrom) and any((a.asname or a.name) == "solve" for a in node.names):
            return True
    return False


def audit_python(src, rel, route_gate=None):
    """(reasons, warnings) of one Python source text."""
    rg = dict(DEFAULT_ROUTE_GATE, **(route_gate or {}))
    reasons, warnings = [], []
    for pat in rg["forbidden_calls"]:
        if pat in src:
            reasons.append("%s: training call %r" % (rel, pat))
    try:
        tree = ast.parse(src, filename=rel)
    except (SyntaxError, ValueError) as e:
        return reasons + ["%s: does not parse (%s)" % (rel, e.__class__.__name__)], warnings
    os_alias = {"os"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "os" and a.asname:
                    os_alias.add(a.asname)
                for banned, what in ((PROCESS_MODULES, "process module"), (NETWORK_MODULES, "network module"),
                                     (FFI_MODULES, "foreign-function module")):
                    if _mod_hit(a.name, banned):
                        reasons.append("%s:%d imports %s (%s)" % (rel, node.lineno, a.name, what))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for banned, what in ((PROCESS_MODULES, "process module"), (NETWORK_MODULES, "network module"),
                                 (FFI_MODULES, "foreign-function module")):
                if mod and _mod_hit(mod, banned):
                    reasons.append("%s:%d imports from %s (%s)" % (rel, node.lineno, mod, what))
                for a in node.names:
                    full = (mod + "." + a.name) if mod else a.name
                    if mod and _mod_hit(full, banned) and not _mod_hit(mod, banned):
                        reasons.append("%s:%d imports %s (%s)" % (rel, node.lineno, full, what))
            if mod == "os":
                for a in node.names:
                    if a.name in OS_PROCESS_CALLS:
                        reasons.append("%s:%d imports os.%s (starts or detaches a process)" % (rel, node.lineno, a.name))
            if mod in ("concurrent.futures",) and any(a.name in PROCESS_CLASSES for a in node.names):
                reasons.append("%s:%d imports a process pool" % (rel, node.lineno))
            if mod == "importlib.util" and any(a.name in ("spec_from_file_location", "module_from_spec") for a in node.names):
                reasons.append("%s:%d imports a module loader (computed import)" % (rel, node.lineno))
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name is None:
                continue
            head, _, last = name.rpartition(".")
            if head in os_alias and last in OS_PROCESS_CALLS:
                reasons.append("%s:%d calls %s (starts or detaches a process)" % (rel, node.lineno, name))
            if last in PROCESS_CLASSES:
                reasons.append("%s:%d uses a process pool" % (rel, node.lineno))
            if name in ("__import__", "importlib.import_module", "importlib.__import__", "builtins.__import__"):
                if not node.args or not _literal(node.args[0]):
                    reasons.append("%s:%d computed import (%s with a non-literal name)" % (rel, node.lineno, name))
            if name in ("importlib.util.spec_from_file_location", "importlib.util.module_from_spec", "imp.load_source",
                        "importlib.machinery.SourceFileLoader", "runpy.run_path", "runpy.run_module"):
                reasons.append("%s:%d loads a module by path (%s)" % (rel, node.lineno, name))
            if name in ("exec", "eval", "compile", "builtins.exec", "builtins.eval", "builtins.compile"):
                if not node.args or not _literal(node.args[0]):
                    reasons.append("%s:%d %s with a non-literal argument" % (rel, node.lineno, name))
            if name == "getattr" and len(node.args) >= 2 and not _literal(node.args[1]):
                tgt = _call_name(node.args[0]) if isinstance(node.args[0], (ast.Name, ast.Attribute)) else None
                if tgt is not None and (tgt in os_alias or _mod_hit(tgt, PROCESS_MODULES + NETWORK_MODULES)):
                    warnings.append("%s:%d getattr(%s, <computed>) (COMPUTED_ATTR)" % (rel, node.lineno, tgt))
    return reasons, warnings


def tree_sha256(root):
    """sha256 of every regular file under `root` (the givens, e.g. /home/user/given/interface/carkit): a submission
    may carry byte-identical copies of them (carkit opens a Unix socket for development), which the Python audit
    skips; they still count for the size."""
    out = set()
    if not root or not os.path.isdir(root):
        return out
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for f in fns:
            p = os.path.join(dp, f)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            out.add(_file_sha256(p))
    return out


def _file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for b in iter(lambda: fp.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def audit_submission(sub_dir, route_gate=None, exempt_sha256=None):
    """The static audit of a submission directory: {"ok", "reasons", "warnings", "files", "bytes", "py_files",
    "exempt_files", "has_drive", "has_solve"}. `route_gate` is the grader configuration's `route_gate` block;
    `exempt_sha256` the sha256 set of the givens (`tree_sha256`), whose byte-identical copies the Python audit skips
    (drive.py itself is never exempt)."""
    rg = dict(DEFAULT_ROUTE_GATE, **(route_gate or {}))
    exempt = set(exempt_sha256 or ())
    out = {"ok": False, "reasons": [], "warnings": [], "files": 0, "bytes": 0, "py_files": 0, "exempt_files": 0,
           "has_drive": False, "has_solve": False}
    if not os.path.isdir(sub_dir) or os.path.islink(sub_dir):
        out["reasons"].append("no submission directory")
        return out
    ext = tuple(e.lower() for e in rg["forbidden_extensions"])
    for dp, dns, fns in os.walk(sub_dir, followlinks=False):
        for d in list(dns):
            if os.path.islink(os.path.join(dp, d)):
                out["reasons"].append("%s: symbolic link" % os.path.relpath(os.path.join(dp, d), sub_dir))
                dns.remove(d)
        dns[:] = [d for d in dns if d != "__pycache__"]
        for f in fns:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, sub_dir)
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                out["reasons"].append("%s: symbolic link" % rel)
                continue
            if not stat.S_ISREG(st.st_mode):
                out["reasons"].append("%s: not a regular file" % rel)
                continue
            out["files"] += 1
            out["bytes"] += st.st_size
            if f.lower().endswith(ext):
                out["reasons"].append("%s: weights or model file" % rel)
            if f.endswith(".py"):
                out["py_files"] += 1
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
                if rel != "drive.py" and exempt and _file_sha256(p) in exempt:
                    out["exempt_files"] += 1
                    continue
                r, w = audit_python(src, rel, rg)
                out["reasons"] += r
                out["warnings"] += w
                if rel == "drive.py":
                    out["has_drive"] = True
                    try:
                        out["has_solve"] = _defines_solve(ast.parse(src))
                    except (SyntaxError, ValueError):
                        out["has_solve"] = False
    if out["bytes"] > float(rg["max_submission_mb"]) * 1024 * 1024:
        out["reasons"].append("submission is %.1f MB, the cap is %s MB" % (out["bytes"] / 1048576.0, rg["max_submission_mb"]))
    if not out["has_drive"]:
        out["reasons"].append("no drive.py")
    elif not out["has_solve"]:
        out["reasons"].append("drive.py defines no solve")
    out["ok"] = not out["reasons"]
    return out


def submission_digest(sub_dir):
    """sha256 over the sorted (relative path, sha256 of content) of every regular file (symbolic links and other
    entries are listed by their kind): sealed when the agent's session ends, compared at verify (`no_late_write`)."""
    h = hashlib.sha256()
    if not os.path.isdir(sub_dir):
        return hashlib.sha256(b"no submission").hexdigest()
    rows = []
    for dp, dns, fns in os.walk(sub_dir, followlinks=False):
        dns.sort()
        for f in fns + [d for d in dns if os.path.islink(os.path.join(dp, d))]:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, sub_dir)
            st = os.lstat(p)
            if stat.S_ISREG(st.st_mode):
                rows.append("%s\0file\0%s" % (rel, _file_sha256(p)))
            elif stat.S_ISLNK(st.st_mode):
                rows.append("%s\0link\0%s" % (rel, os.readlink(p)))
            else:
                rows.append("%s\0other\0%o" % (rel, st.st_mode))
    for r in sorted(rows):
        h.update(r.encode("utf-8", "surrogateescape") + b"\n")
    return h.hexdigest()
