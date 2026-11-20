"""The generator's sha256 (``provenance.generator_sha256`` of every site it writes; the hidden-set rule "freeze the generator,
the cell and the reference before the candidates are run").

The digest covers every file whose content can change a generated site: this package, the fixture
(``review/sitegen/fixture``), the site parts (``cabincell/site``), the telemetry synthesis (``cabincell/
telemetry_synth.py``), the task firmware (``cabincell/firmware``: the pre-grasp palm transform and zone rules), the
vendored simulation package and station library (``cabincell/_lib``), the cell's data (``cabincell/data``: the
runtime site_spec, the station data, the robot model and its asset manifest, which pins the robot archive) and the
asset root resolver (``cabincell/assetroot.py``).  Python caches are skipped.  digest = sha256 over the sorted lines
"<path relative to tests/> <sha256 of the file>\\n".

    python -I -B freeze.py            prints the digest and the per-part digests as JSON
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
PARTS = ("review/sitegen/gen", "review/sitegen/fixture", "cabincell/site", "cabincell/telemetry_synth.py",
         "cabincell/firmware", "cabincell/_lib", "cabincell/data", "cabincell/assetroot.py")
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIX = (".pyc", ".pyo")


def files(tests: str = TESTS) -> list:
    out = []
    for part in PARTS:
        p = os.path.join(tests, part)
        if os.path.isfile(p):
            out.append(part)
            continue
        for root, dirs, fs in os.walk(p):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for f in sorted(fs):
                if f.endswith(SKIP_SUFFIX) or f == "MUJOCO_LOG.TXT":
                    continue
                out.append(os.path.relpath(os.path.join(root, f), tests).replace(os.sep, "/"))
    return sorted(set(out))


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table(tests: str = TESTS) -> list:
    return [(rel, _sha(os.path.join(tests, rel))) for rel in files(tests)]


def digest(tests: str = TESTS) -> str:
    return hashlib.sha256("".join(f"{rel} {h}\n" for rel, h in table(tests)).encode("utf-8")).hexdigest()


def report(tests: str = TESTS) -> dict:
    T = table(tests)
    parts = {}
    for part in PARTS:
        rows = [(r, h) for r, h in T if r == part or r.startswith(part + "/")]
        parts[part] = dict(files=len(rows), sha256=hashlib.sha256("".join(f"{r} {h}\n" for r, h in rows).encode("utf-8")).hexdigest())
    return dict(generator_sha256=hashlib.sha256("".join(f"{r} {h}\n" for r, h in T).encode("utf-8")).hexdigest(), files=len(T),
                parts=parts)


if __name__ == "__main__":
    print(json.dumps(report(), indent=1))
    sys.exit(0)
