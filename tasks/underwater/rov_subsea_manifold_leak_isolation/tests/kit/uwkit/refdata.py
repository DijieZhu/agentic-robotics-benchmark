"""Access to the reference-vehicle data files (JSON only; no MuJoCo, usable from pure-numpy graders).

Files (written by the reference-vehicle digest generator; maintainer / verifier side, they carry real part names):
  spec_values.json             cross-task constants, one entry per key: {"value", "unit", "kind", "source"}
  vehicle_parts_manifest.json  every part instance: mass, centroid, displacement, centre of buoyancy, inertia, pose, meshes
  mesh library                 <mesh_root>/{lo,hi,coll}/... referenced by the manifest (large; never inside this package)

Paths are resolved in this order: explicit argument > environment variable > default next to the package:
  UWKIT_SPEC_VALUES  (default <package>/../spec_values.json)
  UWKIT_MANIFEST     (default <package>/../vehicle_parts_manifest.json)
  UWKIT_MESHLIB      (default: the manifest's "mesh_root")
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.normpath(os.path.join(_HERE, ".."))
CONFIGS = ("std", "heavy", "mission", "mission_skid")      # mission_skid: digest v2, task 02 host only (manifest items / spec keys)
WATERS = ("fresh_20C", "sea_SP35_20C")


def spec_values_path(path: Optional[str] = None) -> str:
    return path or os.environ.get("UWKIT_SPEC_VALUES") or os.path.join(_SHARED, "spec_values.json")


def manifest_path(path: Optional[str] = None) -> str:
    return path or os.environ.get("UWKIT_MANIFEST") or os.path.join(_SHARED, "vehicle_parts_manifest.json")


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict:
    with open(path) as f:
        return json.load(f)


def load_json(path: str) -> dict:
    return _load(os.path.abspath(path), os.path.getmtime(path))


def load_spec_values(path: Optional[str] = None) -> dict:
    """The whole spec_values.json (cached per path and modification time)."""
    return load_json(spec_values_path(path))


def load_manifest(path: Optional[str] = None) -> dict:
    return load_json(manifest_path(path))


def spec(key: str, path: Optional[str] = None, default=KeyError):
    """The 'value' of one spec_values entry, e.g. spec('mission.mass_kg')."""
    vals = load_spec_values(path)["values"]
    if key not in vals:
        if default is KeyError:
            raise KeyError(f"spec_values has no entry {key!r}")
        return default
    return vals[key]["value"]


def mesh_root(manifest: Optional[dict] = None, path: Optional[str] = None) -> str:
    if path:
        return path
    env = os.environ.get("UWKIT_MESHLIB")
    if env:
        return env
    man = manifest if manifest is not None else load_manifest()
    return man["mesh_root"]


def water_density(water="fresh_20C", path: Optional[str] = None) -> float:
    """Water density [kg/m^3] by name ('fresh_20C' | 'sea_SP35_20C') or pass-through for a number."""
    if isinstance(water, (int, float)):
        return float(water)
    key = {"fresh_20C": "water_density_fresh_20C", "fresh": "water_density_fresh_20C",
           "sea_SP35_20C": "water_density_sea_SP35_20C", "sea": "water_density_sea_SP35_20C"}.get(water)
    if key is None:
        raise ValueError(f"unknown water {water!r}; use 'fresh_20C', 'sea_SP35_20C' or a density")
    # the uwkit.* entries carry the unrounded density used by the budget
    full = spec(f"uwkit.mission.{'fresh_20C' if 'fresh' in key else 'sea_SP35_20C'}", path, default=None)
    if full is not None:
        return float(full["rho_kg_m3"])
    return float(spec(key, path))


def gravity(path: Optional[str] = None) -> float:
    return float(spec("gravity", path))


def manifest_items(config: str, manifest: Optional[dict] = None, include_derived: bool = True) -> list:
    """All manifest entries (instances + derived pseudo-items) that belong to a configuration."""
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}")
    man = manifest if manifest is not None else load_manifest()
    out = [i for i in man["items"] if config in i["configs"]]
    if include_derived:
        out += [i for i in man.get("derived_items", []) if config in i["configs"]]
    return out
