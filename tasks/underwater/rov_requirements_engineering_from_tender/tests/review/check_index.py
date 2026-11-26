#!/usr/bin/env python3
"""Check that the review evidence still describes this task folder (maintainer tool; standard library only).

    python3 -I -B tests/review/check_index.py

Reads evidence_index.json next to this file and recomputes the sha256 of every task file it lists (relative to the task
folder, the parent of tests/), and of the files its publication entry names (the published Dockerfile and the two
files the published form adds). Prints each file whose bytes changed since the evidence was taken and the evidence
entries whose inputs it is among ('tests/world/*' style entries match by prefix). Exit 0 when nothing changed, 1 when
something did (the listed checks must be run again before they are cited), 2 on a missing file.
"""
import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
TASK = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def affected(rel, reports):
    out = []
    for r in reports:
        for entry in r.get("computed_on", []):
            if entry == rel or (entry.endswith("*") and rel.startswith(entry[:-1])):
                out.append(r["id"])
                break
    return out


def pinned(idx):
    """{relative path: sha256} of every file the index pins: task_files, then the publication entry's files."""
    want = dict(idx["task_files"])
    pub = idx.get("publication") or {}
    for rel, entry in sorted((pub.get("changed_task_files") or {}).items()):
        want.setdefault(rel, entry["published_form"])
    for rel, digest in sorted((pub.get("added_task_files") or {}).items()):
        want.setdefault(rel, digest)
    return want


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    idx = json.loads((HERE / "evidence_index.json").read_text(encoding="utf-8"))
    changed, missing = [], []
    want = pinned(idx)
    for rel, digest in sorted(want.items()):
        p = TASK / rel
        if not p.is_file():
            missing.append(rel)
        elif sha(p) != digest:
            changed.append(rel)
    for rel in changed:
        print("changed  %-55s inputs of: %s" % (rel, ", ".join(affected(rel, idx["reports"])) or "-"))
    for rel in missing:
        print("missing  %s" % rel)
    print("%d files checked, %d changed, %d missing" % (len(want), len(changed), len(missing)))
    return 2 if missing else 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
