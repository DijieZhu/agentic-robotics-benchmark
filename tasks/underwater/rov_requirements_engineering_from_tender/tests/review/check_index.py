#!/usr/bin/env python3
"""Check that the review evidence still describes this task folder (maintainer tool; standard library only).

    python3 -I -B tests/review/check_index.py [--dev DEV_FOLDER]

Reads evidence_index.json next to this file, recomputes the sha256 of every task file it lists (relative to the task
folder, the parent of tests/) and, with --dev (the task's development folder in the development repository), of every
cited report and its companions. Prints each file whose bytes changed since the evidence was taken and the reports
whose inputs it is among ('tests/world/*' style entries match by prefix). Exit 0 when nothing changed, 1 when
something did (the listed reports must be re-run before they are cited again), 2 on a missing file.
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dev", type=pathlib.Path, help="the task's development folder")
    args = ap.parse_args(argv)
    idx = json.loads((HERE / "evidence_index.json").read_text(encoding="utf-8"))
    changed, missing = [], []
    for rel, want in sorted(idx["task_files"].items()):
        p = TASK / rel
        if not p.is_file():
            missing.append(rel)
        elif sha(p) != want:
            changed.append(rel)
    for rel in changed:
        print("changed  %-55s inputs of: %s" % (rel, ", ".join(affected(rel, idx["reports"])) or "-"))
    if args.dev:
        for r in idx["reports"]:
            for rel, want in [(r["path"], r["sha256"])] + sorted(r.get("also", {}).items()):
                p = args.dev / rel
                if not p.is_file():
                    missing.append(str(args.dev / rel))
                elif sha(p) != want:
                    changed.append(rel)
                    print("changed  %-55s (report %s)" % (rel, r["id"]))
    for rel in missing:
        print("missing  %s" % rel)
    n = len(idx["task_files"]) + (sum(1 + len(r.get("also", {})) for r in idx["reports"]) if args.dev else 0)
    print("%d files checked, %d changed, %d missing" % (n, len(changed), len(missing)))
    return 2 if missing else 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
