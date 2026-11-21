"""The runtime copy of the task's site_spec (``tests/cabincell/data/site_spec.json``, generated, never
edited by hand; no grader thresholds, no QC values) with its sha256 self-check.

    from cabincell.site import spec
    v = spec.values()                       # {key: value}
    spec.value("world.blocker.hold_N")      # {'S': [5.5, 8.5], 'P': [1.15, 1.6]}

``CABINCELL_SITE_SPEC`` overrides the path (its sha256 file is ``<path minus .json>.sha256``).  Standard library only.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.normpath(os.path.join(HERE, "..", "data", "site_spec.json"))


def path() -> str:
    return os.environ.get("CABINCELL_SITE_SPEC") or RUNTIME


def sha_path(p: str) -> str:
    return p[:-len(".json")] + ".sha256" if p.endswith(".json") else p + ".sha256"


@functools.lru_cache(maxsize=4)
def _load(p: str) -> dict:
    raw = open(p, "rb").read()
    want = open(sha_path(p), "rb").read().decode("ascii").split()[0]
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise RuntimeError(f"site_spec {p}: sha256 {got} != {want} (its .sha256 file)")
    return json.loads(raw.decode("ascii"))


def document() -> dict:
    """The whole runtime document (schema, version, values, ...); raises if the sha256 file does not match."""
    return _load(path())


def values() -> dict:
    return document()["values"]


def value(key: str):
    """One value; KeyError names the key."""
    v = values()
    if key not in v:
        raise KeyError(f"site_spec has no key {key!r}")
    return v[key]


def sha256() -> str:
    """sha256 of the runtime copy (what a site's provenance records)."""
    return hashlib.sha256(open(path(), "rb").read()).hexdigest()
