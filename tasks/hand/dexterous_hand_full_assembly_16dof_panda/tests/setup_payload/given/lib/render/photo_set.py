"""A task's FIXED photo set, built deterministically from a seed: agent-visible images + a PUBLIC photos.json, and a
HIDDEN photos_gt.json with everything exact (grader-only).

    spec = PhotoSetSpec(set_id, scene, shots=[ShotSpec(label, rig=CameraRig(...), n_views=4), ...], photo, seed)
    summary = build_photo_set(spec, public_dir, hidden_dir)          # writes the files, raises on any leak
    report = verify_photo_set(public_dir, hidden_dir)                # re-render from photos_gt.json, compare bytes
    leaks = find_leaks(public_doc, gt_doc)                           # [] when photos.json holds no exact camera number
    lens = draw_lens(rig, seed)                                      # RA5: the one lens of a photo set (CameraRig.shared_lens)
    lens.scaled(width) / lens.breathe(seed, i, width=)               # the same lens at another output size (same aspect)
    render_record(gt_doc)                                            # RA6: per shot LOD + backend build, explicit
    assert_independent_sets(practice_gt, hidden_gt)                  # a practice set shares no seed / lens / camera

public_dir/                          (agent-visible: a task's setup/payload/...)
    photo_01.jpg ...                 the photographs (no EXIF / PNG text chunks)
    photos.json                      per photo: file, size, format, sha256, the shot's public label, a focal-length
                                     CLASS (35 mm-equivalent band), which scale references are in view; the nominal
                                     sizes of the scale references; the author's public notes
hidden_dir/                          (grader-only: the task's hidden folder)
    photos_gt.json                   the exact RenderRequests (scene, cameras incl. intrinsics / extrinsics /
                                     distortion / exposure, photo spec, seeds), provenance (backend + version + request
                                     digest + GL / library versions + image sha256), per-view visible pixels of every
                                     scale reference, the scale references' exact world poses and feature points
    gt/seg_01.png ...                uint16 PNG, geom id + 1 (0 = background); labels in photos_gt.json
    gt/depth_01.npy ...              float32 metres along the optical axis (if save_depth)
public_dir/depth_01.png ...          (if save_sensor_depth) the AGENT-VISIBLE sensor depth of an RGB-D camera: 16-bit PNG,
                                     value x unit_m metres, 0 = invalid (PhotoSpec.depth_sensor; listed per photo in
                                     photos.json under "sensor_depth")

Lens (RA5): CameraRig.shared_lens = True draws ONE lens per photo set and lens_id (focal length / hfov, aspect,
principal point, distortion) from derive_seed(set seed, "lens", lens_id) instead of per photo; each photo then only
breathes: fx, fy x (1 + focus_breathing x U(-1, 1)) and the principal point +- principal_breathing_px (autofocus focus
breathing of a phone camera).  Poses, exposure and white balance still vary per photo, and the per-photo draws of
the pose are the same with and without a shared lens.  Rigs sharing a lens agree on its bands incl. the breathing
(CameraRig.lens_signature, pixel bands per width); a rig of another size with the same aspect ratio gets the lens
scaled.  photos_gt.json["lenses"] records each lens (hidden), also the lenses of explicit cameras (PhotoSetSpec.lenses).

Rule: nothing exact about a camera ever enters photos.json (disclosure "focal_class", the default) - with
disclosure "calibrated_intrinsics" the exact fx fy cx cy + distortion are published (a pre-calibrated camera) but
never a pose; "none" publishes only file names and sizes.  build_photo_set runs find_leaks and refuses to write a
leaking public file.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import postprocess
from . import scale_refs as SR
from .qc import ref_pixels
from .spec import (CameraSpec, PhotoSpec, RenderRequest, SceneSpec, StateSpec, _Spec, canonical_json, derive_seed, rng,
                   sha256_hex)

PHOTO_SET_FORMAT = "arb-photo-set/1"
GT_FORMAT = "arb-photo-set-gt/1"
DISCLOSURES = ("none", "focal_class", "calibrated_intrinsics")

# 35 mm-equivalent focal length classes (diagonal convention: f35 = f_px * 43.27 mm / image diagonal px)
FOCAL_CLASSES = [("ultra-wide", 0.0, 20.0), ("wide", 20.0, 30.0), ("normal", 30.0, 60.0), ("short tele", 60.0, 105.0),
                 ("tele", 105.0, 1e9)]


@dataclass
class CameraRig(_Spec):
    """A distribution of hand-held camera poses around a target, sampled deterministically (sample_cameras).

    Camera centre = target + d (cos el cos az, cos el sin az, sin el); az from world +x counter-clockwise, el above the
    horizontal.  The n views spread evenly over `azimuth_deg` (+- azimuth_jitter_deg each); every other quantity is
    uniform in its band.  `target` may be [x, y, z] or "body:NAME" / "site:NAME" / "geom:NAME" of the scene."""
    target: Any = field(default_factory=lambda: [0.0, 0.0, 0.0])
    target_jitter_m: float = 0.0
    distance_m: List[float] = field(default_factory=lambda: [0.5, 0.8])
    azimuth_deg: List[float] = field(default_factory=lambda: [0.0, 360.0])
    azimuth_jitter_deg: float = 10.0
    elevation_deg: List[float] = field(default_factory=lambda: [15.0, 45.0])
    roll_deg: float = 0.0
    width: int = 1280
    height: int = 960
    hfov_deg: List[float] = field(default_factory=lambda: [50.0, 65.0])
    aspect_jitter: float = 0.0
    principal_jitter_px: float = 0.0
    k1: List[float] = field(default_factory=lambda: [0.0, 0.0])
    k2: List[float] = field(default_factory=lambda: [0.0, 0.0])
    p1: List[float] = field(default_factory=lambda: [0.0, 0.0])
    p2: List[float] = field(default_factory=lambda: [0.0, 0.0])
    k3: List[float] = field(default_factory=lambda: [0.0, 0.0])
    exposure_ev: List[float] = field(default_factory=lambda: [0.0, 0.0])
    wb_jitter: float = 0.0
    # ---- added after 1.0.0 (omitted from to_dict while default; RA5)
    shared_lens: bool = False            # one lens per photo set and lens_id (module docstring); False = a lens per photo
    lens_id: str = "main"                # rigs of one set with the same lens_id share the lens (their lens bands must agree)
    focus_breathing: float = 0.0         # per-photo relative focal change of the shared lens (0.03 = +-3 %); a hfov band
                                         # +-d deg at hfov h corresponds to focus_breathing ~ radians(d) / sin(h)
    principal_breathing_px: float = 0.0  # per-photo principal-point wander of the shared lens (pixels)
    focal_px: Optional[List[float]] = None   # fx band in pixels; replaces hfov_deg when given (same random draw)

    _OMIT_IF_DEFAULT = frozenset({"shared_lens", "lens_id", "focus_breathing", "principal_breathing_px", "focal_px"})

    def lens_signature(self):
        """The bands that define the lens, incl. its focus breathing, with pixel quantities divided by the image width:
        rigs sharing a lens must agree on them.  One lens may deliver images of different sizes with the same aspect
        ratio (a phone's 2016 x 1512 and 1600 x 1200 outputs of one sensor): LensSpec.scaled."""
        from fractions import Fraction
        W = float(self.width)
        g = lambda x: float("%.12g" % float(x))  # noqa: E731
        band = lambda b: [g(v) for v in b]  # noqa: E731
        return canonical_json([str(Fraction(int(self.width), int(self.height))),
                               band(self.hfov_deg) if self.focal_px is None else None,
                               None if self.focal_px is None else [g(v / W) for v in self.focal_px],
                               g(self.aspect_jitter), g(self.principal_jitter_px / W),
                               band(self.k1), band(self.k2), band(self.p1), band(self.p2), band(self.k3),
                               g(self.focus_breathing), g(self.principal_breathing_px / W)])

    def draw_fx(self, r) -> float:
        """One uniform draw of the focal length: from focal_px if given, else from the hfov_deg band."""
        if self.focal_px is not None:
            return _u(r, self.focal_px)
        return 0.5 * self.width / math.tan(math.radians(_u(r, self.hfov_deg)) / 2.0)


@dataclass
class LensSpec(_Spec):
    """One physical lens + sensor (hidden): the intrinsics every photo of the lens starts from."""
    lens_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    focus_breathing: float = 0.0
    principal_breathing_px: float = 0.0

    def scaled(self, width: int, height: Optional[int] = None) -> "LensSpec":
        """The same lens delivering a width x height image of the same aspect ratio (the sensor binned / downscaled):
        s = width / self.width; fx, fy and principal_breathing_px x s, cx, cy -> (c + 0.5) s - 0.5 (pixel centres at
        integer coordinates); distortion (normalised coordinates) and focus_breathing unchanged."""
        width = int(width)
        H = int(round(self.height * width / float(self.width))) if height is None else int(height)
        if width == self.width and H == self.height:
            return self
        if width * self.height != H * self.width:
            raise ValueError("lens %r is %d x %d; %d x %d has another aspect ratio (a lens only scales)" % (self.lens_id, self.width, self.height,
                                                                                                         width, H))
        s = width / float(self.width)
        return self.copy(width=width, height=H, fx=self.fx * s, fy=self.fy * s, cx=(self.cx + 0.5) * s - 0.5, cy=(self.cy + 0.5) * s - 0.5,
                         principal_breathing_px=self.principal_breathing_px * s)

    def camera_kwargs(self, focus_scale: float = 1.0, principal_offset=(0.0, 0.0), width: Optional[int] = None) -> Dict[str, Any]:
        """Keyword arguments for CameraSpec.look_at (explicit ShotSpec.cameras sharing this lens); `width` scales the
        lens to another output size of the same aspect ratio (LensSpec.scaled; principal_offset is in that size's px)."""
        ln = self if width is None else self.scaled(width)
        return {"width": ln.width, "height": ln.height, "fx": ln.fx * focus_scale, "fy": ln.fy * focus_scale,
                "cx": ln.cx + float(principal_offset[0]), "cy": ln.cy + float(principal_offset[1]),
                "k1": ln.k1, "k2": ln.k2, "p1": ln.p1, "p2": ln.p2, "k3": ln.k3}

    def breathe(self, seed: int, index: int, width: Optional[int] = None) -> Dict[str, Any]:
        """camera_kwargs of photo `index` with its focus-breathing draw (stream rng(seed, "rig", index, "breathing")),
        optionally at another output width of the same aspect ratio."""
        ln = self if width is None else self.scaled(width)
        u = rng(seed, "rig", index, "breathing").uniform(-1.0, 1.0, 3)
        return ln.camera_kwargs(1.0 + ln.focus_breathing * u[0], (ln.principal_breathing_px * u[1], ln.principal_breathing_px * u[2]))


def draw_lens(rig: "CameraRig", seed: int) -> LensSpec:
    """The lens of `rig` drawn from `seed` (build_photo_set uses derive_seed(set seed, "lens", rig.lens_id)): fx from
    rig.focal_px or from the rig.hfov_deg band, aspect 1 + aspect_jitter U(-1, 1), principal point (W-1)/2, (H-1)/2 +
    principal_jitter_px U(-1, 1), distortion in the k / p bands.  Every value is drawn even when its band is empty."""
    r = rng(seed, "lens", rig.lens_id)
    fx = rig.draw_fx(r)
    asp = 1.0 + rig.aspect_jitter * r.uniform(-1, 1)
    pj = r.uniform(-1, 1, 2) * rig.principal_jitter_px
    dist = {k: _u(r, getattr(rig, k)) for k in ("k1", "k2", "p1", "p2", "k3")}
    return LensSpec(lens_id=rig.lens_id, width=rig.width, height=rig.height, fx=fx, fy=fx * asp, cx=(rig.width - 1) / 2.0 + pj[0],
                    cy=(rig.height - 1) / 2.0 + pj[1], focus_breathing=rig.focus_breathing,
                    principal_breathing_px=rig.principal_breathing_px, **dist)


@dataclass
class ShotSpec(_Spec):
    """One configuration of the scene photographed from one or more cameras.  `label` is PUBLIC text."""
    label: str
    state: Optional[StateSpec] = None
    cameras: List[CameraSpec] = field(default_factory=list)
    rig: Optional[CameraRig] = None
    n_views: int = 0
    hide_bodies: List[str] = field(default_factory=list)


@dataclass
class PhotoSetSpec(_Spec):
    set_id: str
    scene: SceneSpec
    shots: List[ShotSpec]
    photo: PhotoSpec = field(default_factory=PhotoSpec)
    seed: int = 0
    backend: str = "mujoco"
    version: str = "1"
    file_pattern: str = "photo_{index:02d}.{ext}"
    disclosure: str = "focal_class"
    camera_note: str = "hand-held camera; camera poses are not known; the lens is not calibrated; exposure varies between photos"
    public_notes: List[str] = field(default_factory=list)
    save_segmentation: bool = True
    save_depth: bool = False
    min_ref_pixels: int = 150
    min_views_with_ref: int = 0          # QC gate: at least this many photos must show a scale reference
    # ---- added after 1.0.0 (omitted from to_dict while default)
    save_sensor_depth: bool = False      # RA1: write the agent-visible sensor depth (needs photo.depth_sensor) to public_dir
    sensor_depth_pattern: str = "depth_{index:02d}.png"
    lenses: List[LensSpec] = field(default_factory=list)   # RA5: lenses of explicit ShotSpec.cameras (draw_lens / LensSpec.breathe),
                                                          # recorded hidden in photos_gt.json["lenses"] and protected by find_leaks

    _OMIT_IF_DEFAULT = frozenset({"save_sensor_depth", "sensor_depth_pattern", "lenses"})


def _u(r, band):
    lo, hi = float(band[0]), float(band[1])
    x = r.uniform(0.0, 1.0)             # always draw: changing one band never re-draws another quantity
    return lo + (hi - lo) * x


def sample_cameras(rig: CameraRig, n: int, seed: int, prefix: str = "cam", scene: Optional[SceneSpec] = None,
                   target_xyz=None, lens_seed: Optional[int] = None, lens: Optional[LensSpec] = None) -> List[CameraSpec]:
    """n cameras of `rig` (class docstring).  With rig.shared_lens the intrinsics come from `lens` (scaled to the rig's
    size) or else draw_lens(rig, lens_seed) (default lens_seed = seed: one lens for these n views), plus each view's
    focus breathing; the pose draws are the same as without a shared lens."""
    if target_xyz is None:
        if isinstance(rig.target, str):
            from .backends.mujoco_scene import resolve_point
            target_xyz = resolve_point(scene, rig.target)
        else:
            target_xyz = np.asarray(rig.target, float)
    target_xyz = np.asarray(target_xyz, float)
    cams = []
    if rig.shared_lens:
        lens = (lens or draw_lens(rig, seed if lens_seed is None else lens_seed)).scaled(rig.width, rig.height)
    else:
        lens = None
    a0, a1 = [float(v) for v in rig.azimuth_deg]
    full = abs((a1 - a0) % 360.0) < 1e-9 and a1 != a0
    for i in range(n):
        r = rng(seed, "rig", i)
        frac = (i + 0.5) / n if not full else i / n
        az = a0 + (a1 - a0) * frac + rig.azimuth_jitter_deg * r.uniform(-1, 1)
        el = _u(r, rig.elevation_deg)
        d = _u(r, rig.distance_m)
        tj = r.normal(0, 1, 3) * rig.target_jitter_m
        roll = rig.roll_deg * r.uniform(-1, 1)
        fx = rig.draw_fx(r)
        asp = 1.0 + rig.aspect_jitter * r.uniform(-1, 1)
        pj = r.uniform(-1, 1, 2) * rig.principal_jitter_px
        dist = {k: _u(r, getattr(rig, k)) for k in ("k1", "k2", "p1", "p2", "k3")}
        ev = _u(r, rig.exposure_ev)
        wb = (1.0 + rig.wb_jitter * r.uniform(-1, 1, 3)).tolist()
        t = target_xyz + tj
        ca, sa, ce, se = math.cos(math.radians(az)), math.sin(math.radians(az)), math.cos(math.radians(el)), math.sin(math.radians(el))
        eye = t + d * np.array([ce * ca, ce * sa, se])
        if lens is not None:                                   # one lens: only focus breathing per photo
            intr = lens.breathe(seed, i)
            cams.append(CameraSpec.look_at("%s%02d" % (prefix, i + 1), eye, t, roll_deg=roll, exposure_ev=ev, white_balance=wb, **intr))
            continue
        cams.append(CameraSpec.look_at("%s%02d" % (prefix, i + 1), eye, t, width=rig.width, height=rig.height, fx=fx, fy=fx * asp,
                                       cx=(rig.width - 1) / 2.0 + pj[0], cy=(rig.height - 1) / 2.0 + pj[1], roll_deg=roll,
                                       exposure_ev=ev, white_balance=wb, **dist))
    return cams


def focal_class(camera: CameraSpec) -> Dict[str, Any]:
    f35 = camera.fx * 43.2666 / math.hypot(camera.width, camera.height)
    for name, lo, hi in FOCAL_CLASSES:
        if lo <= f35 < hi:
            return {"focal_length_class": name, "equiv_35mm_focal_band_mm": [lo, hi if hi < 1e8 else None]}
    raise AssertionError


# ----------------------------------------------------------------------------- leak detection
_FORBIDDEN_KEYS = {"fx", "fy", "cx", "cy", "k", "k1", "k2", "k3", "p1", "p2", "world_t_cam", "extrinsics", "intrinsics", "pose",
                   "pos", "quat", "position", "rotation", "seed", "seeds", "exposure_ev", "white_balance", "request", "requests",
                   "provenance", "request_digest", "resolved_scene_digest", "r_cam_world", "t_cam_world", "distortion"}
_CALIBRATED_ALLOWED = {"fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2", "intrinsics", "distortion"}
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


# values that are public by construction (image sizes, nominal reference sizes, class bands, hashes, versions)
_PUBLIC_BY_CONSTRUCTION = {"width", "height", "scale_references", "focal_length_classes", "equiv_35mm_focal_band_mm",
                           "sha256", "version", "format", "file", "set", "unit_m", "invalid_value"}


def _walk(obj, path="", numbers=True):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield ("key", path + "/" + str(k), k)
            yield from _walk(v, path + "/" + str(k), numbers and str(k) not in _PUBLIC_BY_CONSTRUCTION)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, "%s[%d]" % (path, i), numbers)
    elif isinstance(obj, bool) or obj is None or not numbers:
        return
    elif isinstance(obj, (int, float)):
        yield ("num", path, repr(obj))
    elif isinstance(obj, str):
        for m in _NUM.finditer(obj):
            yield ("num", path, m.group(0))


def _sig_digits(tok: str) -> int:
    t = tok.lstrip("-").split("e")[0].split("E")[0].replace(".", "").lstrip("0")
    return len(t.rstrip("0")) if "." not in tok else len(t)


def _depth_sensor_doc(gt_doc) -> Dict[str, Any]:
    return (gt_doc.get("photo_set_spec", {}).get("photo") or {}).get("depth_sensor") or {}


def _systematic_values(gt_doc) -> List[tuple]:
    """(value, public prior, label) of the depth unit's hidden systematic error in every written form a note might use:
    scale, scale - 1, (scale - 1) in % and in mm per m; offset in m and in mm.  Matched sign-blind under the relaxed rule
    of find_leaks: any decimal token (whatever its significant digits: "0.004", "3.7", "4.0") whose precision resolves the
    value's distance from the prior; integers ("4 mm", "12") only under the general rule (>= 3 significant digits)."""
    ds = _depth_sensor_doc(gt_doc)
    out = []
    if ds:
        sc, off = float(ds.get("scale", 1.0)), float(ds.get("offset_m", 0.0))
        if abs(sc - 1.0) > 1e-12:
            out += [(sc, 1.0, "depth scale"), (sc - 1.0, 0.0, "depth scale error"), ((sc - 1.0) * 100.0, 0.0, "depth scale error (%)"),
                    ((sc - 1.0) * 1000.0, 0.0, "depth scale error (mm per m)")]
        if abs(off) > 1e-12:
            out += [(off, 0.0, "depth offset (m)"), (off * 1000.0, 0.0, "depth offset (mm)")]
    return out


def _hidden_numbers(gt_doc) -> List[float]:
    vals: List[float] = []
    for v in gt_doc.get("views", []):
        c = v["camera"]
        for k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2", "exposure_ev"):
            vals.append(float(c[k]))
        vals += [float(x) for x in c["white_balance"]]
        vals += [float(x) for row in c["world_T_cam"][:3] for x in row]
    for r in gt_doc.get("scale_refs", []):
        vals += [float(x) for x in r["pos"]] + [float(x) for x in r["quat"]]
    for s in gt_doc.get("seeds", {}).values():
        vals.append(float(s))
    for ln in gt_doc.get("lenses", {}).values():                      # RA5: the shared lenses
        vals += [float(ln[k]) for k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2")]
    ds = _depth_sensor_doc(gt_doc)
    if ds:                                                            # RA1: the unit's systematic error, imager focal, unit seed
        vals += [float(ds.get("scale", 1.0)), float(ds.get("offset_m", 0.0)), float(ds.get("disparity_focal_px", 0.0)),
                 float(ds.get("unit_seed", 0))]
    return [x for x in vals if abs(x) > 1e-12 and abs(abs(x) - 1.0) > 1e-12]


def find_leaks(public_doc, gt_doc, disclosure: Optional[str] = None) -> List[str]:
    """Exact hidden camera / pose / seed numbers or forbidden keys in the public manifest."""
    disclosure = disclosure or public_doc.get("disclosure", "focal_class")
    allowed = _CALIBRATED_ALLOWED if disclosure == "calibrated_intrinsics" else set()
    hidden = np.array(_hidden_numbers(gt_doc), float)
    if disclosure == "calibrated_intrinsics":   # intrinsics are public on purpose: only poses / seeds / exposure stay hidden
        keep = []
        for v in gt_doc.get("views", []):
            c = v["camera"]
            keep += [float(x) for row in c["world_T_cam"][:3] for x in row] + [float(c["exposure_ev"])] + [float(x) for x in c["white_balance"]]
        for r in gt_doc.get("scale_refs", []):
            keep += [float(x) for x in r["pos"]] + [float(x) for x in r["quat"]]
        keep += [float(s) for s in gt_doc.get("seeds", {}).values()]
        ds = _depth_sensor_doc(gt_doc)                                # the depth unit stays hidden: a factory-calibrated
        if ds:                                                        # colour camera still has an unknown depth bias
            keep += [float(ds.get("scale", 1.0)), float(ds.get("offset_m", 0.0)), float(ds.get("unit_seed", 0))]
        hidden = np.array([x for x in keep if abs(x) > 1e-12 and abs(abs(x) - 1.0) > 1e-12], float)
    syst = _systematic_values(gt_doc)                                 # every disclosure
    leaks = []
    for kind, path, val in _walk(public_doc):
        if kind == "key":
            if str(val).lower() in _FORBIDDEN_KEYS and str(val).lower() not in allowed:
                leaks.append("forbidden key %r at %s" % (val, path))
            continue
        dec = len(val.split(".")[1].split("e")[0].split("E")[0]) if "." in val else 0
        sig = _sig_digits(val)
        x = float(val)
        tol = 0.5 * 10 ** (-dec) if dec else 0.5
        if len(hidden) and (sig >= 3 or (dec >= 3 and sig >= 2)):     # 30 / 0.5 / 60 are everyday numbers; 0.051 is not
            close = np.abs(hidden - x) <= tol * (1 + 1e-9)
            if np.any(close):
                leaks.append("number %s at %s equals a hidden value (%s)" % (val, path, hidden[close][0]))
                continue
        if syst and "." in val:                                       # relaxed rule for the depth unit's systematic error
            for h, prior, label in syst:
                if tol < abs(abs(h) - abs(prior)) and abs(abs(x) - abs(h)) <= tol * (1 + 1e-9):
                    leaks.append("number %s at %s gives the hidden %s (%s)" % (val, path, label, h))
                    break
    return leaks


def _load_gt(doc):
    if isinstance(doc, (str, os.PathLike)):
        p = os.path.join(doc, "photos_gt.json") if os.path.isdir(doc) else doc
        with open(p) as f:
            return json.load(f)
    return doc


def independence_problems(practice_gt, hidden_gt, check_intrinsics: bool = True) -> List[str]:
    """Why a practice photo set (whose photos_gt.json may be public, T2) would reveal the hidden set's camera draws:
    a shared seed (PhotoSetSpec.seed, any derived seed, the depth unit's seed), a shared lens (fx, fy, cx, cy), a camera
    pose or (check_intrinsics) a camera's full intrinsics + distortion equal between the sets, or the same depth-unit
    scale / offset.  [] = independent.  Camera, lens and noise seeds depend only on PhotoSetSpec.seed and the rigs, not
    on the scene, so a practice set built with the hidden set's seed reproduces its cameras whatever the scene.
    `check_intrinsics=False` for a fixed, known camera (degenerate bands give equal intrinsics by construction)."""
    a, b = _load_gt(practice_gt), _load_gt(hidden_gt)
    out: List[str] = []

    def seeds(gt):
        vals = {int(v) for v in gt.get("seeds", {}).values()}
        vals.add(int(gt.get("photo_set_spec", {}).get("seed", 0)))
        us = int(_depth_sensor_doc(gt).get("unit_seed", 0) or 0)
        if us:
            vals.add(us)
        return vals - {0}
    shared = seeds(a) & seeds(b)
    if shared:
        out.append("shared seeds %s" % sorted(shared)[:4])
    if int(a.get("photo_set_spec", {}).get("seed", 0)) == int(b.get("photo_set_spec", {}).get("seed", 0)):
        out.append("same PhotoSetSpec.seed %d (cameras, lenses and noise follow the seed, not the scene)"
                   % int(a.get("photo_set_spec", {}).get("seed", 0)))

    def close(x, y, rel=1e-9):
        return abs(float(x) - float(y)) <= rel * max(1.0, abs(float(x)), abs(float(y)))
    for la in a.get("lenses", {}).values():
        for lb in b.get("lenses", {}).values():
            if all(close(la[k], lb[k]) for k in ("fx", "fy", "cx", "cy")):
                out.append("lens %r of the practice set equals lens %r of the hidden set" % (la.get("lens_id"), lb.get("lens_id")))
    intr = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
    for va in a.get("views", []):
        ca = va["camera"]
        Ta = np.asarray(ca["world_T_cam"], float)[:3]
        for vb in b.get("views", []):
            cb = vb["camera"]
            if np.allclose(Ta, np.asarray(cb["world_T_cam"], float)[:3], rtol=0.0, atol=1e-9):
                out.append("camera pose of %s equals that of hidden %s" % (va["file"], vb["file"]))
            if check_intrinsics and all(close(ca[k], cb[k], 1e-12) for k in intr):
                out.append("camera intrinsics of %s equal those of hidden %s" % (va["file"], vb["file"]))
    da, db = _depth_sensor_doc(a), _depth_sensor_doc(b)
    for k, neutral in (("scale", 1.0), ("offset_m", 0.0)):
        if da and db and float(da.get(k, neutral)) != neutral and close(da.get(k, neutral), db.get(k, neutral), 1e-12):
            out.append("depth unit %s equal in both sets" % k)
    return out


def assert_independent_sets(practice_gt, hidden_gt, check_intrinsics: bool = True) -> None:
    """Raise ValueError if independence_problems(...) is not empty; every regen_givens --check that builds practice sets
    next to a hidden set calls it (render/README.md, public / hidden rule)."""
    probs = independence_problems(practice_gt, hidden_gt, check_intrinsics)
    if probs:
        raise ValueError("practice photo set is not independent of the hidden set: %s" % "; ".join(probs[:6]))


# ----------------------------------------------------------------------------- build
def _write_json(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")


def _check_dirs(public_dir, hidden_dir):
    p, h = os.path.realpath(public_dir), os.path.realpath(hidden_dir)
    if p == h or p.startswith(h + os.sep) or h.startswith(p + os.sep):
        raise ValueError("public_dir and hidden_dir must be disjoint (got %s / %s)" % (public_dir, hidden_dir))


def _r9(a):
    return np.round(np.asarray(a, float), 9).tolist()


def build_photo_set(spec: PhotoSetSpec, public_dir: str, hidden_dir: str, backend=None) -> Dict[str, Any]:
    from .backends import get_backend
    if spec.disclosure not in DISCLOSURES:
        raise ValueError("disclosure must be one of %s" % (DISCLOSURES,))
    _check_dirs(public_dir, hidden_dir)
    be = backend or get_backend(spec.backend)
    if be.name != spec.backend:
        raise ValueError("backend object %r != spec.backend %r" % (be.name, spec.backend))
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(os.path.join(hidden_dir, "gt"), exist_ok=True)
    outputs = ["rgb", "segmentation"] + (["depth"] if spec.save_depth else []) + (["sensor_depth"] if spec.save_sensor_depth else [])
    if spec.save_sensor_depth and spec.photo.depth_sensor is None:
        raise ValueError("save_sensor_depth needs PhotoSetSpec.photo.depth_sensor")
    photos_pub, views_gt, requests, provs, seeds = [], [], [], [], {}
    lenses: Dict[str, Any] = {}
    lens_sig: Dict[str, str] = {}
    lens_obj: Dict[str, LensSpec] = {}
    for ln in spec.lenses:                                       # explicit-camera lenses (RA5): recorded + protected
        if ln.lens_id in lenses:
            raise ValueError("PhotoSetSpec.lenses: lens_id %r twice" % ln.lens_id)
        lenses[ln.lens_id] = ln.to_dict()
    index = 0
    labels_by_shot = []
    for k, shot in enumerate(spec.shots):
        scene_k = spec.scene.copy(state=shot.state or spec.scene.state,
                                  hide_bodies=list(spec.scene.hide_bodies) + list(shot.hide_bodies))
        if shot.cameras:
            cams = list(shot.cameras)
        else:
            if shot.rig is None or shot.n_views <= 0:
                raise ValueError("shot %r needs cameras or a rig with n_views > 0" % shot.label)
            seeds["shot%d_cameras" % k] = derive_seed(spec.seed, "shot", k, "cameras")
            lens = None
            if shot.rig.shared_lens:
                lid = shot.rig.lens_id
                if lid in lenses and lid not in lens_obj:
                    raise ValueError("lens_id %r is both an explicit PhotoSetSpec.lenses entry and a CameraRig.shared_lens" % lid)
                if lid in lens_sig and lens_sig[lid] != shot.rig.lens_signature():
                    raise ValueError("shots sharing lens %r have different lens bands (aspect ratio / hfov or focal_px per width / "
                                     "aspect jitter / principal / distortion / focus breathing): one lens has one set of bands" % lid)
                if lid not in lens_obj:                          # drawn once, at the first rig's size; other sizes scale it
                    lens_sig[lid] = shot.rig.lens_signature()
                    lens_seed = seeds.setdefault("lens_%s" % lid, derive_seed(spec.seed, "lens", lid))
                    lens_obj[lid] = draw_lens(shot.rig, lens_seed)
                    lenses[lid] = lens_obj[lid].to_dict()
                lens = lens_obj[lid]
            cams = sample_cameras(shot.rig, shot.n_views, seeds["shot%d_cameras" % k], prefix="s%dv" % (k + 1), scene=scene_k,
                                  lens=lens)
        seeds["shot%d_photo" % k] = derive_seed(spec.seed, "shot", k, "photo")
        req = RenderRequest(scene=scene_k, cameras=cams, photo=spec.photo, seed=seeds["shot%d_photo" % k], backend=spec.backend,
                            outputs=outputs, label="%s/%s#%d" % (spec.set_id, spec.version, k))
        res = be.render(req)
        requests.append(req.to_dict())
        provs.append(res.provenance.to_dict())
        labels_by_shot.append({str(i): v for i, v in res.segmentation_labels.items() if v.get("geom") or v.get("scale_ref")})
        for j, cam in enumerate(cams):
            index += 1
            fname = spec.file_pattern.format(index=index, ext=spec.photo.extension)
            with open(os.path.join(public_dir, fname), "wb") as f:
                f.write(res.encoded[j])
            px = ref_pixels(res.segmentation[j], res.segmentation_labels)
            in_view = sorted(n for n, c in px.items() if c >= spec.min_ref_pixels)
            pub = {"file": fname, "width": cam.width, "height": cam.height, "format": spec.photo.image_format,
                   "sha256": sha256_hex(res.encoded[j]), "shot": shot.label, "scale_references_in_view": in_view}
            if spec.save_sensor_depth:
                dname = spec.sensor_depth_pattern.format(index=index)
                with open(os.path.join(public_dir, dname), "wb") as f:
                    f.write(res.sensor_depth_encoded[j])
                pub["sensor_depth"] = {"file": dname, "sha256": sha256_hex(res.sensor_depth_encoded[j]), "format": "png16",
                                       "unit_m": spec.photo.depth_sensor.quantisation_m, "invalid_value": 0,
                                       "registration": "aligned to this photo's pixels (same camera)"}
            if spec.disclosure == "focal_class":
                pub.update(focal_class(cam))
            elif spec.disclosure == "calibrated_intrinsics":
                pub["intrinsics"] = {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
                                     "distortion": {"k1": cam.k1, "k2": cam.k2, "p1": cam.p1, "p2": cam.p2, "k3": cam.k3},
                                     "model": "OpenCV pinhole + radial-tangential; pixel centres at integer coordinates"}
            photos_pub.append(pub)
            gt = {"file": fname, "shot": k, "shot_label": shot.label, "camera": cam.to_dict(), "K": _r9(cam.K),
                  "opencv_R_cam_world": _r9(cam.opencv_extrinsics()[0]), "opencv_t_cam_world": _r9(cam.opencv_extrinsics()[1]),
                  "hfov_deg": round(cam.hfov_deg(), 9), "vfov_deg": round(cam.vfov_deg(), 9), "ref_pixels": px,
                  "sha256": sha256_hex(res.encoded[j]), "postprocess_jitter": postprocess.jitter_values(spec.photo, req.seed, j)}
            if spec.save_segmentation:
                from PIL import Image
                seg16 = (np.asarray(res.segmentation[j]) + 1).astype(np.uint16)
                sp = os.path.join("gt", "seg_%02d.png" % index)
                Image.fromarray(seg16).save(os.path.join(hidden_dir, sp), format="PNG", compress_level=9)
                gt["segmentation_file"] = sp
            if spec.save_depth:
                dp = os.path.join("gt", "depth_%02d.npy" % index)
                np.save(os.path.join(hidden_dir, dp), np.asarray(res.depth[j], np.float32))
                gt["depth_file"] = dp
            if spec.save_sensor_depth:
                gt["sensor_depth_file"] = pub["sensor_depth"]["file"]
                gt["sensor_depth_sha256"] = pub["sensor_depth"]["sha256"]
            views_gt.append(gt)
    n_with_ref = sum(1 for p in photos_pub if p["scale_references_in_view"])
    if n_with_ref < spec.min_views_with_ref:
        raise ValueError("QC: only %d photos show a scale reference (min_views_with_ref = %d)" % (n_with_ref, spec.min_views_with_ref))
    public_doc = {"format": PHOTO_SET_FORMAT, "set": spec.set_id, "version": spec.version, "disclosure": spec.disclosure,
                  "photos": photos_pub, "camera": spec.camera_note,
                  "scale_references": [SR.nominal(r) for r in spec.scene.scale_refs],
                  "conventions": "image origin top-left, u right, v down; scale references are physical objects lying in the scene",
                  "notes": list(spec.public_notes)}
    if spec.disclosure == "focal_class":
        public_doc["focal_length_classes"] = {n: [lo, hi if hi < 1e8 else None] for n, lo, hi in FOCAL_CLASSES}
        public_doc["focal_length_class_rule"] = "35 mm-equivalent focal length band (diagonal convention); approximate, not a calibration"
    gt_doc = {"format": GT_FORMAT, "set": spec.set_id, "version": spec.version, "photo_set_spec": spec.to_dict(),
              "requests": requests, "provenance": provs, "views": views_gt, "seeds": seeds,
              "segmentation_labels": labels_by_shot,
              "scale_refs": [{"name": r.name, "kind": r.kind, "pos": r.pos, "quat": r.quat, "params": r.params,
                              "nominal": SR.nominal(r),
                              "features_world": {k: (_r9(v) if isinstance(v, np.ndarray) else v) for k, v in SR.features(r).items()}}
                             for r in spec.scene.scale_refs],
              "rule": "GRADER-ONLY: exact cameras, seeds, poses, segmentation; never copy into setup/ or image/"}
    if lenses:
        gt_doc["lenses"] = lenses
    leaks = find_leaks(public_doc, gt_doc, spec.disclosure)
    if leaks:
        raise ValueError("photos.json would leak hidden values: %s" % leaks[:5])
    _write_json(os.path.join(public_dir, "photos.json"), public_doc)
    _write_json(os.path.join(hidden_dir, "photos_gt.json"), gt_doc)
    return {"n_photos": index, "n_with_scale_ref": n_with_ref, "public": public_doc, "request_digests": [p["request_digest"] for p in provs],
            "backend_version": provs[0]["backend_version"] if provs else None}


def render_record(gt_doc) -> List[Dict[str, Any]]:
    """RA6: per shot of a photos_gt.json (a dict or a path), the level of detail and the backend build it was rendered
    with, explicit even for sets written before the LOD existed (no SceneSpec.lod = "full"):
    {shot, lod, lod_digest, lod_recipe, backend, backend_version, render_layer_version, postprocess_version, mujoco,
     mujoco_gl, gl_vendor, gl_renderer, gl_version, sensor_depth_model}."""
    if isinstance(gt_doc, (str, os.PathLike)):
        with open(gt_doc) as f:
            gt_doc = json.load(f)
    out = []
    for k, (rq, pv) in enumerate(zip(gt_doc.get("requests", []), gt_doc.get("provenance", []))):
        lod = pv.get("lod") or {}
        env = pv.get("environment", {})
        out.append({"shot": k, "lod": rq.get("scene", {}).get("lod", "full"), "lod_digest": lod.get("digest"),
                    "lod_recipe": lod.get("recipe"), "backend": pv.get("backend"), "backend_version": pv.get("backend_version"),
                    "render_layer_version": pv.get("render_layer_version"), "postprocess_version": pv.get("postprocess_version"),
                    "mujoco": env.get("mujoco"), "mujoco_gl": env.get("mujoco_gl"), "gl_vendor": env.get("gl_vendor"),
                    "gl_renderer": env.get("gl_renderer"), "gl_version": env.get("gl_version"),
                    "sensor_depth_model": pv.get("sensor_depth_model")})
    return out


def verify_photo_set(public_dir: str, hidden_dir: str, backend=None, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Re-render every request of photos_gt.json with its recorded backend and compare the bytes with the files.
    Differences mean the software stack drifted (e.g. another GL driver): the shipped files stay the given; publishing
    re-rendered files would be a NEW task version (render/README.md).  `base_dir` replaces the recorded scene base_dir
    (the task folder moved, e.g. into a verify sandbox); the request digest ignores it only if the recorded paths were
    relative."""
    from .backends import get_backend
    with open(os.path.join(hidden_dir, "photos_gt.json")) as f:
        gt = json.load(f)
    if base_dir is not None:
        for rd in gt["requests"]:
            rd["scene"]["base_dir"] = base_dir
    out = {"identical": True, "mismatches": [], "recorded": [p["backend_version"] for p in gt["provenance"]], "now": []}
    by_shot: Dict[int, List[dict]] = {}
    for v in gt["views"]:
        by_shot.setdefault(v["shot"], []).append(v)
    for k, rd in enumerate(gt["requests"]):
        req = RenderRequest.from_dict(rd)
        be = backend or get_backend(req.backend)
        res = be.render(req)
        out["now"].append(res.provenance.backend_version)
        if res.provenance.request_digest != gt["provenance"][k]["request_digest"]:
            out["identical"] = False
            out["mismatches"].append("shot %d: request digest changed (scene files edited?)" % k)
        for j, v in enumerate(by_shot.get(k, [])):
            with open(os.path.join(public_dir, v["file"]), "rb") as f:
                on_disk = sha256_hex(f.read())
            now = sha256_hex(res.encoded[j])
            if now != on_disk or on_disk != v["sha256"]:
                out["identical"] = False
                out["mismatches"].append("%s: file %s, recorded %s, re-rendered %s" % (v["file"], on_disk[:12], v["sha256"][:12], now[:12]))
            if v.get("sensor_depth_file"):
                with open(os.path.join(public_dir, v["sensor_depth_file"]), "rb") as f:
                    sd_disk = sha256_hex(f.read())
                sd_now = sha256_hex(res.sensor_depth_encoded[j])
                if sd_now != sd_disk or sd_disk != v["sensor_depth_sha256"]:
                    out["identical"] = False
                    out["mismatches"].append("%s: sensor depth differs" % v["sensor_depth_file"])
        out.setdefault("environment_now", res.provenance.environment)
    out["environment_recorded"] = gt["provenance"][0]["environment"] if gt["provenance"] else {}
    return out
