"""Rebuild base/ from a checkout of the maker's repository and verify every file against base/MANIFEST.json.

The mesh files under base/**/assets/ are not committed; this script restores them (and the vendored XML files and the
licence) from a checkout of github.com/enactic/openarm_mujoco at the commit the manifest pins, and checks each file's
sha256 before and after the copy. The manifest is never rewritten.

Usage:
  python -I -B base/restore.py <path to an openarm_mujoco checkout>            # HEAD must be the pinned commit
  python -I -B base/restore.py <checkout> --no-commit-check                     # a plain folder laid out like a checkout
  python -I -B base/restore.py --verify                                         # only verify base/ against the manifest
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
UNIT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UNIT_DIR))

from openarm_unit import base as B  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", nargs="?", help="root of an openarm_mujoco checkout (the folder holding v2/)")
    ap.add_argument("--no-commit-check", action="store_true", help="accept a folder that is not a git checkout")
    ap.add_argument("--verify", action="store_true", help="verify base/ only")
    a = ap.parse_args(argv)
    if a.verify:
        r = B.verify_tree(B.BASE_DIR, B.load_manifest(B.BASE_DIR))
    elif a.checkout:
        r = B.restore_base(Path(a.checkout), B.BASE_DIR, require_commit=not a.no_commit_check)
    else:
        ap.error("give a checkout, or --verify")
    print(json.dumps({k: (v[:10] if isinstance(v, list) else v) for k, v in r.items()}, indent=1))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
