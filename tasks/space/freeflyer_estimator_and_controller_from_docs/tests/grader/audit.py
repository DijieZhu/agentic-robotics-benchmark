"""The static import audit of the submission (score.gates: "the static import audit passes"), standard library only.

Every .py file of the submission folder is parsed (never imported or run). The audit passes when every import
statement names a module whose top-level name is one of:
  - the standard library of the verifying interpreter (sys.stdlib_module_names), for the entry "python standard
    library" of interface.libraries;
  - a library named in interface.libraries ("numpy", "scipy");
  - a module or package of the submission itself (the name of a .py file or of a folder anywhere in the submission,
    so that a controller may put its own folders on its path);
a relative import (from . import x) is the submission's own. A .py file that does not parse cannot be audited and fails
the audit. Imports made at run time by name (importlib, __import__) are not seen by a static audit; the controller's
sandbox bounds what they could reach.
"""
import ast
import os
import sys

STDLIB_ENTRY = "python standard library"


def allowed_names(libraries, root):
    names = set()
    for lib in libraries:
        if lib == STDLIB_ENTRY:
            names |= set(sys.stdlib_module_names)
        else:
            names.add(lib)
    own = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d != "__pycache__" and not os.path.islink(os.path.join(dirpath, d))]
        own |= set(dirnames)
        own |= {f[:-3] for f in filenames if f.endswith(".py")}
    return names, own


def audit(root, libraries):
    """(passes, findings): findings name the file and the module of each import outside the allowed set, or a file
    that does not parse."""
    if root is None or not os.path.isdir(root):
        return True, []
    names, own = allowed_names(libraries, root)
    findings = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d != "__pycache__" and not os.path.islink(os.path.join(dirpath, d))]
        for f in sorted(filenames):
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, root)
            try:
                with open(p, "rb") as fh:
                    tree = ast.parse(fh.read(), filename=rel)
            except (SyntaxError, ValueError, OSError, RecursionError, MemoryError) as exc:
                findings.append("%s: cannot be parsed (%s)" % (rel, type(exc).__name__))
                continue
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        continue
                    mods = [node.module or ""]
                for m in mods:
                    top = m.split(".")[0]
                    if top not in names and top not in own:
                        findings.append("%s: imports %s" % (rel, m))
    return (not findings), findings
