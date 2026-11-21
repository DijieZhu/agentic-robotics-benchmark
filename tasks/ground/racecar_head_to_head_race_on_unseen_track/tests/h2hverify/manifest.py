"""The vendored grader's files and their sha256 (grader side, stdlib only).

`tests/grader_manifest.json` (written by the maintainers' assembly script) pins every file
of `tests/grader/` (the shared cell, generator and contract trees, this task's heat cell, field and scoring
modules, the client library), `tests/cellkit/` (the house's process kit, byte-identical to the benchmark's behavior
tasks) and `tests/h2hverify/` by sha256.
`check` compares the stage with it BEFORE any of those modules is imported: a changed, missing or extra file is a
task error, never a score.
"""

import hashlib
import json
import os

__all__ = ["check", "sha256_file", "MANIFEST_SCHEMA"]

MANIFEST_SCHEMA = "racecar-grader-manifest/1"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(stage_dir, name="grader_manifest.json"):
    """[problem text] (empty when the stage's vendored files are exactly the manifest's)."""
    path = os.path.join(stage_dir, name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError) as e:
        return ["the grader manifest cannot be read: %s" % e.__class__.__name__]
    if man.get("schema") != MANIFEST_SCHEMA:
        return ["the grader manifest has the schema %r" % man.get("schema")]
    files = man.get("files") or {}
    problems = []
    for rel, want in sorted(files.items()):
        p = os.path.join(stage_dir, rel)
        if not os.path.isfile(p) or os.path.islink(p):
            problems.append("missing: %s" % rel)
        elif sha256_file(p) != want:
            problems.append("changed: %s" % rel)
    for root in man.get("roots") or []:
        base = os.path.join(stage_dir, root)
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d != "__pycache__"]
            for fn in fns:
                if fn.endswith((".pyc", ".pyo")):
                    continue
                rel = os.path.relpath(os.path.join(dp, fn), stage_dir).replace(os.sep, "/")
                if rel not in files:
                    problems.append("not in the manifest: %s" % rel)
    return problems
