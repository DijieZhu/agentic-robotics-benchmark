"""The cell's render bundle (version 2): found through ONE path, pinned, checked before a
mission.

The task image unpacks the bundle (environment/stage_assets.py) under /opt/task-assets, readable by root and the
practice ground's account only, and points SITE_GEN_DRESS_BUNDLE at it (/opt/task-assets/look, a link to the pinned
bundle's folder). The grader's cell (tests/test.sh exports the same two variables), the oracle (the same cell) and the
practice ground (setup.sh exports them for its service) all build their worlds through MissionWorld, which calls
check() first:

  * SITE_GEN_DRESS=1 (required; the image, test.sh and setup.sh set it): SITE_GEN_DRESS_BUNDLE must name a folder with
    a manifest.json that lists at least one file, and every listed file must exist with its sha256; anything else
    raises BundleError, so the mission never starts with undressed (or wrongly dressed) frames. No other folder is
    tried: the variable is the one path.
  * The pin: the manifest's own sha256 must equal the one in the cell's `config/render_bundle.json`
    (`manifest_sha256`), so that a bundle whose files were altered together with a rewritten manifest is refused as
    well. In the task's cell (the vendored layout: tests/plantcell/_lib, and its byte copy in the practice ground) the
    pin is always in force and a missing pin file is a BundleError; in the source tree only when MISSION_BUNDLE_PIN is set
    (`1`: the source tree's config/render_bundle.json, or a path to a pin file), so that development runs
    with a copy of the frozen bundle are not refused while the pin moves (the manifest was rebuilt once).
  * SITE_GEN_DRESS=0 or unset: nothing is checked here (development runs, version 1 packages); the loader's own switch decides.

The manifest's format is the look bundle's (`files`: {relative path: {"sha256": ...}}, also a plain sha256 string per
file); the result (bundle name, the manifest's sha256, file count, whether the pin was checked) goes into the mission's
record.
"""

from __future__ import annotations

import hashlib
import json
import os

ENV_SWITCH = "SITE_GEN_DRESS"
ENV_PATH = "SITE_GEN_DRESS_BUNDLE"
ENV_PIN = "MISSION_BUNDLE_PIN"
PIN_FILE = "render_bundle.json"


class BundleError(RuntimeError):
    """The render bundle is required and missing, incomplete, altered or not the pinned one."""


def required() -> bool:
    return os.environ.get(ENV_SWITCH, "").strip().lower() in ("1", "on", "true", "yes", "required")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pin_path():
    """The pin file in force, or None (source tree without MISSION_BUNDLE_PIN)."""
    from . import paths
    if paths.VENDORED:
        return os.path.join(paths.CONFIG_DIR, PIN_FILE)
    v = os.environ.get(ENV_PIN, "").strip()
    if not v or v.lower() in ("0", "off", "false", "no"):
        return None
    if v.lower() in ("1", "on", "true", "yes"):
        return os.path.join(paths.CONFIG_DIR, PIN_FILE)
    return v


def read_pin(path):
    """The pin's record ({"name", "manifest_sha256", ...}); BundleError when it is missing or unusable."""
    try:
        with open(path) as f:
            pin = json.load(f)
        want = pin["manifest_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise BundleError("the render bundle's pin %s is missing or unusable (%s)" % (path, e)) from None
    if not (isinstance(want, str) and len(want) == 64):
        raise BundleError("the render bundle's pin %s holds no sha256" % path)
    return pin


def verify(root: str, pin=None) -> dict:
    """Check the bundle at root: its manifest against the pin (when given), every file against its manifest; the
    bundle's record, or BundleError."""
    mp = os.path.join(root, "manifest.json")
    if not os.path.isfile(mp):
        raise BundleError("the render bundle is required (%s=1) but %s has no manifest.json" % (ENV_SWITCH, root))
    raw = open(mp, "rb").read()
    msha = hashlib.sha256(raw).hexdigest()
    if pin is not None and msha != pin["manifest_sha256"]:
        raise BundleError("render bundle %s: its manifest (sha256 %s) is not the pinned one (%s)"
                          % (root, msha[:12], pin["manifest_sha256"][:12]))
    try:
        man = json.loads(raw)
    except ValueError as e:
        raise BundleError("the render bundle's manifest is not JSON (%s)" % e) from None
    files = man.get("files") if isinstance(man, dict) else None
    if not isinstance(files, dict) or not files:
        raise BundleError("the render bundle's manifest lists no files")
    bad = []
    for rel, rec in sorted(files.items()):
        want = rec.get("sha256") if isinstance(rec, dict) else rec
        p = os.path.join(root, rel)
        if (not isinstance(want, str) or os.path.isabs(rel) or ".." in rel.split("/") or not os.path.isfile(p)
                or _sha256(p) != want):
            bad.append(rel)
    if bad:
        raise BundleError("render bundle %s: %d file(s) missing or altered (%s)" % (root, len(bad), ", ".join(bad[:5])))
    return {"path": os.path.abspath(root), "name": man.get("name") or man.get("bundle") or os.path.basename(
        os.path.realpath(root)), "format": man.get("format"), "manifest_sha256": msha, "files": len(files),
            "pinned": pin is not None}


def check():
    """None when the bundle is not required; else its record (BundleError when it is not right)."""
    if not required():
        return None
    root = os.environ.get(ENV_PATH, "").strip()
    if not root:
        raise BundleError("the render bundle is required (%s=1) but %s is not set" % (ENV_SWITCH, ENV_PATH))
    pp = pin_path()
    return verify(root, read_pin(pp) if pp is not None else None)
