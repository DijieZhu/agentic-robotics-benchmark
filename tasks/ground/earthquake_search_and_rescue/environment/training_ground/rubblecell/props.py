"""The look pack's models (cell internals): furniture, the rescue post's kit, ruin pieces, window modules, apartment
blocks, wall decals. `<look dir>/props.npz` holds, per model and part, vertices (metres), normals, texture coordinates
and triangles; `props.json` the models' sizes and each part's material (a colour map under `props/`, or plain
parameters). A model's frame: the footprint's centre at the origin, its lowest point at z = 0, its length along x, its
front (a facade's outside, a cupboard's doors) at -y. Without the pack (a development checkout) the site is built of
the primitives it used to be."""
import json
import os

import numpy as np

from . import pbr

_C = {}


def have():
    return os.path.isfile(os.path.join(pbr.cache_dir(), "props.npz"))


def _index():
    if "J" not in _C:
        _C["J"] = json.load(open(os.path.join(pbr.cache_dir(), "props.json")))
    return _C["J"]


def size(model):
    """(length, depth, height) of a model in metres."""
    return tuple(float(v) for v in _index()["models"][model]["size"])


def parts(model):
    """[(material key, V, N, T, F)] of a model."""
    if "Z" not in _C:
        _C["Z"] = np.load(os.path.join(pbr.cache_dir(), "props.npz"))
    z = _C["Z"]
    out = []
    for i, key in enumerate(_index()["models"][model]["parts"]):
        tag = "%s:%d" % (model, i)
        out.append((key, z["V:" + tag].astype(np.float64), z["N:" + tag].astype(np.float64),
                    z["T:" + tag].astype(np.float64), z["F:" + tag].astype(np.int32)))
    return out


def material(key):
    """{"tex": file under props/} or {"rgb", "rough", "metal", "glass", "emission"}."""
    return _index()["materials"][key]


def texture_path(key):
    return os.path.join(pbr.cache_dir(), "props", material(key)["tex"])
