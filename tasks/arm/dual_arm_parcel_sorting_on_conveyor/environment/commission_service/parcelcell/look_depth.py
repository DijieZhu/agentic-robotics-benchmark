"""The depth twin's meshes: the station's surroundings near the cameras (fence panels and posts, the gantry post, the
conveyor's frame members, rails and legs, the photo-eye, the light curtain, the camera posts) as the render bundle
carries them (``depth_near.npz`` / ``depth_near.json``, world frame, metres), so that a depth image shows what the
colour image shows. The cell adds them to the shift's model as drawn-only geoms that only the depth pass draws
(``world.ShiftWorld._add_depth_twin``).

    meshes = load_depth_twin(cfg["look"])      # [(V (n, 3), F (m, 3))] or None

The bundle's folder is the look's (``look.render_dir``, else $PARCELCELL_RENDER_DIR, else /opt/task-assets/render).
Without the files the classic backend's depth shows the cell's own geometry only (recorded once in the log); the
filament backend needs them (its bundle is checked as a whole).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

__all__ = ["load_depth_twin", "DEPTH_FILES"]

DEFAULT_RENDER_DIR = "/opt/task-assets/render"
DEPTH_FILES = ("depth_near.npz", "depth_near.json")
_CACHE = {}


def _render_dir(options):
    o = options or {}
    return os.path.abspath(o.get("render_dir") or os.environ.get("PARCELCELL_RENDER_DIR") or DEFAULT_RENDER_DIR)


def load_depth_twin(options=None, required=False, log=None):
    """[(V float64 (n, 3), F int32 (m, 3))] per near mesh (vertices renumbered per mesh), or None when the bundle
    has no depth twin and it is not ``required``."""
    from .look import RenderError
    rdir = _render_dir(options)
    npz = os.path.join(rdir, DEPTH_FILES[0])
    if not os.path.isfile(npz):
        if required:
            raise RenderError("renderer: the render bundle in %s lacks %s" % (rdir, DEPTH_FILES[0]))
        if rdir not in _CACHE and log is not None:
            log("[cell] no depth twin in %s: depth shows the cell's own geometry only" % rdir)
        _CACHE[rdir] = None
        return None
    if rdir in _CACHE and _CACHE[rdir] is not None:
        return _CACHE[rdir][1]
    with open(npz, "rb") as fh:
        raw = fh.read()
    digest = hashlib.sha256(raw).hexdigest()
    man = os.path.join(rdir, "manifest.json")
    if os.path.isfile(man):
        with open(man) as fh:
            want = json.load(fh).get("files", {}).get(DEPTH_FILES[0])
        if want is not None and want != digest:
            raise RenderError("renderer: render bundle file changed: %s" % DEPTH_FILES[0])
    import io
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        V, F, ofs = np.asarray(z["V"], np.float64), np.asarray(z["F"], np.int64), [int(v) for v in z["mesh_ofs"]]
    ofs = ofs + [len(F)]
    out = []
    for i in range(len(ofs) - 1):
        f = F[ofs[i]:ofs[i + 1]]
        if len(f) == 0:
            continue
        used = np.unique(f)
        remap = np.full(int(used.max()) + 1, -1, np.int64)
        remap[used] = np.arange(len(used))
        out.append((V[used].copy(), remap[f].astype(np.int32)))
    _CACHE[rdir] = (digest, out)
    return out
