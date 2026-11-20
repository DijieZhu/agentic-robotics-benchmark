"""Paths and loaders of the station data (``tests/cabincell/data/station``).

Scenes: 'mission' (modules l2, h2, l1; scene tag 'l123') and 'practice' (module l3; tag 'l3').
Files: station_features.json / practice_features.json (neutral catalogues), scene_<tag>.xml (physics scene, no file
assets), look_<tag>.xml (the same scene with the look; meshdir / texturedir 'LOOK_ROOT' replaced at load time),
zones.json, map_features.json, station_v03.json (the v0.3 additions: stowage-unit sites with their siting record,
registered / moved positions, inlet ends over a wall box, r_face ranges), ASSETS.json (the look archive's manifest).  ``STATION_DATA_DIR`` overrides the directory.
"""
from __future__ import annotations

import functools
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "data", "station"))
SCENES = {"mission": dict(tag="l123", modules=("l2", "h2", "l1"), features="station_features.json"),
          "practice": dict(tag="l3", modules=("l3",), features="practice_features.json")}


def data_dir() -> str:
    return os.environ.get("STATION_DATA_DIR") or DEFAULT_DATA_DIR


def path(name: str) -> str:
    return os.path.join(data_dir(), name)


def scene_of_module(module: str) -> str:
    return "practice" if module == "l3" else "mission"


@functools.lru_cache(maxsize=None)
def _load(p: str) -> str:
    with open(p) as fh:
        return fh.read()


def load_json(name: str) -> dict:
    """Parsed JSON file of the data directory (a fresh object on every call)."""
    return json.loads(_load(path(name)))


def features(scene: str = "mission") -> dict:
    return load_json(SCENES[scene]["features"])


def scene_text(scene: str = "mission", look: bool = False) -> str:
    tag = SCENES[scene]["tag"]
    return _load(path(("look_" if look else "scene_") + tag + ".xml"))


def zones(scene: str = "mission") -> list:
    return load_json("zones.json")[SCENES[scene]["tag"]]


def v03() -> dict:
    """The v0.3 additions (``station_v03.json``, written by the siting pipeline)."""
    return load_json("station_v03.json")


def unit_sites() -> dict:
    """{'sites': [...], 'registered': {module: id}, 'moved': {module: [ids]}, 'practice': {module: id}}."""
    V = v03()
    return dict(sites=V["unit_sites"], registered=V["registered"], moved=V["moved"], practice=V["practice"])


def unit_site(site_id: str) -> dict:
    for r in v03()["unit_sites"]:
        if r["id"] == site_id:
            return r
    raise KeyError(site_id)


def inlet_by_id(feats: dict, iid: str) -> dict:
    for i in feats["v02"]["inlets"]:
        if i["id"] == iid:
            return i
    raise KeyError(iid)


def module_record(feats: dict, mid: str) -> dict:
    return feats["modules"][mid]
