"""The station look (the modules' meshes and textures) is a task asset: one deterministic archive whose manifest is
``data/station/ASSETS.json`` (schema ``cabincell-assets/1``, component 'station', archive top 'station'); the cell's
``assetroot.find('station')`` (tests/cabincell/assetroot.py) returns the verified unpacked directory -- the look root
the look scenes reference (``meshes/``, ``tex/``, ``tex/plates/``).

``look_root()`` uses assetroot when it can be imported (the cell puts tests/cabincell on sys.path), else the
directory ``CABINCELL_STATION_LOOK`` names, verified against the same manifest.  Physics never needs the look.
"""
from __future__ import annotations

import os
import sys

from . import data

COMPONENT = "station"


def _assetroot():
    here = os.path.dirname(os.path.abspath(__file__))
    cell = os.path.normpath(os.path.join(here, "..", ".."))
    if cell not in sys.path:
        sys.path.append(cell)
    import assetroot  # noqa: E402
    return assetroot


def manifest() -> dict:
    return data.load_json("ASSETS.json")


def verify(look_dir: str) -> list:
    """Problems of an unpacked look directory against the manifest (empty = it matches)."""
    ar = _assetroot()
    return ar.verify_tree(look_dir, manifest(), data_dir=os.path.dirname(data.data_dir()))


def look_root() -> str:
    """The verified look directory (see the module doc)."""
    d = os.environ.get("CABINCELL_STATION_LOOK")
    if d:
        probs = verify(d)
        if probs:
            m = manifest()
            raise RuntimeError(f"CABINCELL_STATION_LOOK={d} does not match the station look manifest ({len(probs)} problems, "
                               f"first {probs[:3]}); expected {m['archive']['name']} sha256 {m['archive']['sha256']}")
        return d
    return str(_assetroot().find(COMPONENT))
