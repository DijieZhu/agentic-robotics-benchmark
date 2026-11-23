"""Parcel visuals: what a parcel looks like, behind one interface.

The per-shift world (``world.ShiftWorld``) builds each parcel's physics itself
(collision geoms, explicit inertial, label site: ``_geom.parcel_fields`` and
``_geom.label_layout``) and then asks a visuals object to dress the body.
A visuals object may add only drawn geoms (contype 0, conaffinity 0, group 0)
and the assets they use (meshes, textures, materials); the world checks that
after compiling (``check_visual_only``). Swapping one visuals object for
another therefore changes the pictures and nothing of the physics (a unit
test compares the compiled physics fields and a ride bit for bit).

Interface (duck-typed):

    prepare(spec, shift)                       once per shift, before any add
    add(spec, body, i, parcel, layout) -> [geom names]
        body:   the parcel's MjSpec body (frame = the parcel's frame: x, y, z
                along its size axes, origin at the box centre)
        layout: the label placement the physics uses (``_geom.label_layout``:
                "label" (centre, half sizes), "site_pos", "site_quat", plus
                "face" and "uv"); a visual label must be drawn there
    describe() -> dict                         what the visuals are (records)

``PlainVisuals``: a kraft box (soft mailers: a flattened box with a rounded
cover) and a flat label patch (white sheet with two dark code blocks): the
simulator's own look, what the classic renderer draws today.

``KitVisuals``: the parcel kit (``parcelcell.parcelkit``): every parcel's own
visual mesh, maps (base colour, normal, occlusion-roughness-metallic) and
waybill, generated from the shift itself (the parcel's kind, size, label face
and offset, and a seed derived from the shift's seed and the parcel's index),
at agent quality (512 px maps). The same shift gives the same bytes, made
afresh or read from the shift-look cache (``parcelcell.kitcache``). The
parcel's label site is the kit's placement record (the real paper's centre on
the label face; z = the face normal, x = the waybill's reading direction), so
the score and the picture agree on where the label is; there are no
primitive label geoms. The printed weight is random (never the mass).
The classic renderer draws the base colour; the house renderer all maps.
"""

from __future__ import annotations

import hashlib
import json
import os

import mujoco
import numpy as np

from . import _geom

__all__ = ["PlainVisuals", "KitVisuals", "check_visual_only", "make_visuals",
           "KRAFT_RGBA", "MAILER_RGBA"]

KRAFT_RGBA = (0.72, 0.55, 0.38, 1.0)
MAILER_RGBA = (0.80, 0.78, 0.74, 1.0)
LABEL_RGBA = (0.97, 0.97, 0.95, 1.0)
INK_RGBA = (0.08, 0.08, 0.08, 1.0)


def _vis_geom(body, name, **kw):
    kw.setdefault("group", 0)
    return body.add_geom(name=name, contype=0, conaffinity=0, mass=0, **kw)


def _label_pieces(body, prefix, layout):
    """White sheet + two dark code blocks at the physics label layout."""
    names = []
    for nm, rgba in (("label", LABEL_RGBA), ("label_qr", INK_RGBA),
                     ("label_bar", INK_RGBA)):
        pos, half = layout[nm]
        _vis_geom(body, f"{prefix}_{nm}", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[float(v) for v in pos], size=[float(v) for v in half],
                  rgba=list(rgba))
        names.append(f"{prefix}_{nm}")
    return names


class PlainVisuals:
    """Plain box and a flat label patch (the simulator's own look)."""

    name = "plain"

    def prepare(self, spec, shift) -> None:
        return None

    def add(self, spec, body, i, parcel, layout) -> list:
        f = _geom.parcel_fields(parcel.kind, parcel.size, parcel.mass,
                                parcel.friction)
        prefix = f"parcel{i}"
        rgba = KRAFT_RGBA if parcel.kind == "rigid" else MAILER_RGBA
        names = []
        _vis_geom(body, f"{prefix}_geom_visual", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[float(v) for v in f["box_size"]], rgba=list(rgba))
        names.append(f"{prefix}_geom_visual")
        if parcel.kind == "soft":
            pad_rgba = [c * 0.95 for c in rgba[:3]] + [1.0]
            _vis_geom(body, f"{prefix}_pad_visual",
                      type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      size=[float(v) for v in f["pad_size"]], rgba=pad_rgba)
            names.append(f"{prefix}_pad_visual")
        names += _label_pieces(body, prefix, layout)
        return names

    def describe(self) -> dict:
        return {"visuals": self.name}


class KitVisuals:
    """The parcel kit's visuals (module docstring)."""

    name = "kit"

    def __init__(self, quality: str = "agent", cache_dir: str | None = None):
        self.quality = str(quality)
        self.cache_dir = cache_dir
        self._recs = []
        self.prepare_wall_s = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def shift_seed(shift) -> int:
        """The kit's seed of a shift: the shift's seed, else a hash of it."""
        if shift.seed is not None:
            return int(shift.seed) & 0x7FFFFFFFFFFFFFFF
        h = hashlib.sha256(shift.to_json().encode()).digest()
        return int.from_bytes(h[:8], "big") >> 1

    def prepare(self, spec, shift) -> None:
        import time
        from .parcelkit.identity import identities
        from .parcelkit.parcel import make_parcel
        t0 = time.time()
        seed = self.shift_seed(shift)
        n = len(shift.arrivals)
        ids = identities(n, seed)
        self._recs = []
        # OpenCV's optimised kernels (SIMD dispatch, IPP) differ between CPUs in the last bit of float filters:
        # the plain kernels give the same bytes on every host (numpy's dispatch is pinned by the environment)
        import cv2
        opt = cv2.useOptimized()
        cv2.setUseOptimized(False)
        try:
            self._make(shift, seed, ids, make_parcel)
        finally:
            cv2.setUseOptimized(opt)
        self.prepare_wall_s = time.time() - t0

    def _make(self, shift, seed, ids, make_parcel):
        from . import kitcache
        self.cache_hits = self.cache_misses = 0
        for i, (a, ident) in enumerate(zip(shift.arrivals, ids)):
            p = a.parcel
            spec_d = {"kind": p.kind, "size": tuple(p.size), "label_face": int(p.label_face),
                      "label_uv": tuple(p.label_uv)}
            pseed = (seed * 1000003 + i) & 0x7FFFFFFF
            key = kitcache.entry_key(spec_d, pseed, ident.to_dict(), self.quality) if self.cache_dir else None
            hit = kitcache.load(self.cache_dir, key) if key else None
            if hit is not None:
                rec, arr = hit
                rec["_arrays"] = {self.quality: arr}
                self.cache_hits += 1
                self._recs.append(rec)
                continue
            rec = make_parcel(spec_d, pseed, (self.quality,), None, name="parcel%d" % i,
                              identity=ident, check=False)
            rec.pop("_build", None)          # the full-size working buffers (about 0.2 GB a parcel)
            self.cache_misses += 1
            if key:
                from .parcelkit.parcel import strip
                kitcache.store(self.cache_dir, key, json.loads(json.dumps(strip(rec))), rec["_arrays"][self.quality])
            self._recs.append(rec)

    def label_site(self, i, parcel):
        """(pos, quat) of parcel i's label site in its body frame, or None."""
        wb = self._recs[i]["waybill"] if i < len(self._recs) else None
        if not wb:
            return None
        return [float(v) for v in wb["site"]["pos"]], [float(v) for v in wb["site"]["quat_wxyz"]]

    def add(self, spec, body, i, parcel, layout) -> list:
        from .parcelkit.mj import add_parcel_visual
        g = add_parcel_visual(spec, body, self._recs[i], self.quality, prefix="parcel%d_kit" % i)
        self._recs[i].pop("_arrays", None)   # the spec holds its own copy now
        return [g.name]

    def record(self, i) -> dict:
        """The kit's record of parcel i (waybill placement, identity, look)."""
        from .parcelkit.parcel import strip
        r = strip(self._recs[i])
        return {k: r[k] for k in ("name", "kind", "seed", "waybill", "identity", "look") if k in r}

    def describe(self) -> dict:
        return {"visuals": self.name, "quality": self.quality,
                "prepare_wall_s": round(self.prepare_wall_s, 3), "numerics": numerics_regime(),
                "cache": {"on": bool(self.cache_dir), "hits": self.cache_hits, "misses": self.cache_misses}}


def numerics_regime() -> dict:
    """The CPU features numpy dispatches to and the BLAS kernel pin: the kit's
    bytes are the same on every host only under the same regime (the grader
    and the commissioning service set ``NPY_DISABLE_CPU_FEATURES`` and the
    OpenBLAS kernel before numpy is imported)."""
    feats = {}
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as cf
        feats = {k: bool(v) for k, v in cf.items() if k.startswith("AVX512")}
    except ImportError:
        pass
    try:
        import cv2
        cv_opt = bool(cv2.useOptimized())
    except ImportError:
        cv_opt = None
    return {"avx512_dispatch": any(feats.values()), "opencv_optimized_outside_the_kit": cv_opt,
            "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
            "OPENBLAS_CORETYPE": os.environ.get("OPENBLAS_CORETYPE")}


def make_visuals(spec_or_name=None, cache_dir=None):
    """"plain" (default: the simulator's own look, for tooling), "kit" /
    "kit:<quality>" (the parcel kit; the graded and commissioning cells) or an
    object with the interface. ``cache_dir``: the kit's shift-look cache
    (``parcelcell.kitcache``; None: none)."""
    if spec_or_name is None or spec_or_name == "plain":
        return PlainVisuals()
    if isinstance(spec_or_name, str) and spec_or_name.split(":")[0] == "kit":
        q = spec_or_name.split(":", 1)[1] if ":" in spec_or_name else "agent"
        return KitVisuals(q, cache_dir=cache_dir)
    return spec_or_name


def check_visual_only(m: mujoco.MjModel, geom_ids) -> None:
    """Every geom a visuals object added is drawn only: no contact, group 0."""
    for g in geom_ids:
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            raise AssertionError("visual geom %s takes part in contacts"
                                 % mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
        if m.geom_group[g] != 0:
            raise AssertionError("visual geom %s is not in group 0"
                                 % mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
