"""Plate bundles (format arb-plate/1, as this reader checks it): one per site package, the
search camera's offline environment (the plate and its auxiliary maps) that the plate backend composes the robot into.

    B = PlateBundle(root, site_sha256, spec, look, site_json_sha256=..., site_xml_sha256=...)

The bundle of a site lives in <root>/<site package sha256>/ (the sha256 the cell computes for the episode's package,
voidcell/_lib/sitepkg.py). Opening checks, in this order, and raises PlateBundleError on the first failure (the
episode cannot be built: a task error, never a fall-back to another image): a root is configured; the folder and its
meta.json exist; the format; the site package sha256 recorded in meta.json is the episode's (and the site.json /
site.xml sha256 when the cell knows them); the camera recorded in meta.json is the episode's camera; the display
transform recorded is the look's; every file listed exists as a regular file with the recorded size and sha256 and
no file the format needs is missing; the arrays have the camera's shapes; plate_ldr.png is the look's display
transform of plate_lin (sampled pixels).
"""
import hashlib
import io
import json
import os
import re

import numpy as np

from . import plate_look

FORMAT = "arb-plate/1"
REQUIRED = ("plate_ldr.png", "plate_lin.npz", "depth.npz", "depth4.npz", "sunfrac.npz", "air.npz", "sunvis.npz",
            "probes.json", "light.json")
_SHA = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*(/[A-Za-z0-9_][A-Za-z0-9_.\-]*)?")
MAX_FILE_BYTES = 64 << 20
LDR_CHECK_STRIDE = 7          # every 7th pixel of plate_ldr.png is re-derived from plate_lin at open
LDR_CHECK_MAX_SHARE = 0.002   # share of sampled pixels allowed to differ (by at most one level; rounding at a step)


class PlateBundleError(RuntimeError):
    """The episode's plate bundle is missing, belongs to another site or camera, or is damaged."""


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_dir(root, site_sha256):
    return os.path.join(os.fspath(root), str(site_sha256))


def _regular(path):
    return os.path.isfile(path) and not os.path.islink(path)


def _npz(raw, key):
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(key)
        return z[key]


class PlateBundle:
    def __init__(self, root, site_sha256, spec, look, site_json_sha256=None, site_xml_sha256=None):
        if not root:
            raise PlateBundleError("no plate root configured (the cell config's plate_root)")
        sha = str(site_sha256 or "").strip().lower()
        if not _SHA.fullmatch(sha):
            raise PlateBundleError("the episode's site package sha256 %r is not a sha256" % (site_sha256,))
        self.root = os.path.realpath(os.fspath(root))
        self.dir = bundle_dir(self.root, sha)
        meta_path = os.path.join(self.dir, "meta.json")
        if not os.path.isdir(self.root):
            raise PlateBundleError("plate root %s is not a folder" % self.root)
        if not _regular(meta_path):
            raise PlateBundleError("no plate bundle for site package %s under %s" % (sha[:16], self.root))
        with open(meta_path, "rb") as fh:
            raw_meta = fh.read(1 << 20)
        self.meta_sha256 = hashlib.sha256(raw_meta).hexdigest()
        try:
            meta = json.loads(raw_meta.decode("utf-8"))
        except ValueError as e:
            raise PlateBundleError("%s: unreadable (%s)" % (meta_path, e))
        self.meta = meta
        if meta.get("format") != FORMAT:
            raise PlateBundleError("%s: format %r, not %s" % (meta_path, meta.get("format"), FORMAT))
        if str(meta.get("site_package_sha256", "")).lower() != sha:
            raise PlateBundleError("the plate bundle in %s was baked for site package %s, not for the episode's %s"
                                   % (self.dir, str(meta.get("site_package_sha256"))[:16], sha[:16]))
        for key, want in (("site_json_sha256", site_json_sha256), ("site_xml_sha256", site_xml_sha256)):
            if want and str(meta.get(key, "")).lower() != str(want).lower():
                raise PlateBundleError("the plate bundle's %s %s differs from the episode's %s"
                                       % (key, str(meta.get(key))[:16], str(want)[:16]))
        self._check_camera(meta.get("camera") or {}, spec)
        disp = meta.get("display") or {}
        if disp.get("curve") != look["display"]["curve"] or abs(float(disp.get("gain", -1)) -
                                                                  float(look["display"]["gain"])) > 1e-12:
            raise PlateBundleError("the plate bundle was baked with the display transform %r, the look has %r"
                                   % (disp, {k: look["display"][k] for k in ("curve", "gain")}))
        files = meta.get("files")
        if not isinstance(files, dict):
            raise PlateBundleError("%s: no file list" % meta_path)
        for name in REQUIRED:
            if name not in files:
                raise PlateBundleError("%s: the format's file %s is not listed" % (meta_path, name))
        self.raw = {}
        for name, rec in sorted(files.items()):
            if not _NAME.fullmatch(name) or ".." in name.split("/"):
                raise PlateBundleError("%s: unsafe file name %r" % (meta_path, name))
            p = os.path.join(self.dir, name)
            if not _regular(p):
                raise PlateBundleError("plate bundle file %s is missing" % p)
            size = os.path.getsize(p)
            if size > MAX_FILE_BYTES or size != int(rec.get("bytes", -1)):
                raise PlateBundleError("plate bundle file %s has %d bytes, meta.json records %s" % (p, size, rec.get("bytes")))
            with open(p, "rb") as fh:
                b = fh.read()
            if hashlib.sha256(b).hexdigest() != str(rec.get("sha256", "")).lower():
                raise PlateBundleError("plate bundle file %s does not match its recorded sha256" % p)
            self.raw[name] = b
        try:
            self._load(spec, look)
        except PlateBundleError:
            raise
        except Exception as e:  # noqa: BLE001 - any unreadable or misshapen array is a damaged bundle
            raise PlateBundleError("plate bundle %s: unreadable content (%s: %s)" % (self.dir, type(e).__name__, e))
        self._check_ldr(look)
        self.raw = None

    # ------------------------------------------------------------------ checks
    @staticmethod
    def _check_camera(cam, spec):
        try:
            same = (int(cam["width"]) == int(spec.width) and int(cam["height"]) == int(spec.height)
                    and all(abs(float(cam[k]) - float(getattr(spec, k))) <= 1e-9 for k in ("fx", "fy", "cx", "cy"))
                    and np.allclose(np.asarray(cam["T_world_camera"], float), np.asarray(spec.T, float), rtol=0,
                                    atol=1e-9))
        except (KeyError, TypeError, ValueError):
            same = False
        if not same:
            raise PlateBundleError("the plate bundle was baked for another camera than the episode's search camera")

    def _load(self, spec, look):
        from PIL import Image
        H, W = int(spec.height), int(spec.width)
        ss = int(look["window"]["supersample"])
        r = self.raw
        with Image.open(io.BytesIO(r["plate_ldr.png"])) as im:
            if im.mode != "RGB":
                raise PlateBundleError("plate_ldr.png is %s, not RGB" % im.mode)
            self.plate_ldr = np.ascontiguousarray(np.asarray(im, np.uint8))
        self.plate_lin = _npz(r["plate_lin.npz"], "plate_lin").astype(np.float32)
        self.depth1 = _npz(r["depth.npz"], "depth").astype(np.float32)
        self.depth4 = _npz(r["depth4.npz"], "depth4").astype(np.float32)
        self.phi = _npz(r["sunfrac.npz"], "phi").astype(np.float32)
        self.A = _npz(r["air.npz"], "A").astype(np.float32)
        self.A_hi = _npz(r["air.npz"], "A_hi").astype(np.float32)
        self.T_hi = _npz(r["air.npz"], "T_hi").astype(np.float32)
        self.venv = {k: _npz(r["sunvis.npz"], k) for k in ("V", "x", "y", "z")}
        self.venv["V"] = self.venv["V"].astype(np.float32)
        for k in ("x", "y", "z"):
            self.venv[k] = self.venv[k].astype(np.float64)
        want = {"plate_ldr": (H, W, 3), "plate_lin": (H, W, 3), "depth1": (H, W), "depth4": (ss * H, ss * W),
                "phi": (H, W), "A": (H, W, 3), "A_hi": (H, W, 3), "T_hi": (H, W)}
        for k, shp in want.items():
            if getattr(self, k).shape != shp:
                raise PlateBundleError("plate bundle %s: %s has shape %s, the camera needs %s"
                                       % (self.dir, k, getattr(self, k).shape, shp))
        V, xs, ys, zs = self.venv["V"], self.venv["x"], self.venv["y"], self.venv["z"]
        if V.shape != (len(zs), len(ys), len(xs)) or min(len(xs), len(ys), len(zs)) < 2:
            raise PlateBundleError("plate bundle %s: the sun-visibility grid is misshapen" % self.dir)
        for a in (self.plate_lin, self.depth1, self.depth4, self.phi, self.A, self.A_hi, self.T_hi, V):
            if not np.all(np.isfinite(a)):
                raise PlateBundleError("plate bundle %s: an array holds non-finite values" % self.dir)
        self.light = json.loads(r["light.json"].decode("utf-8"))
        k = np.asarray(self.light["sun_dir"], float)
        if k.shape != (3,) or not (np.linalg.norm(k) > 0.5 and k[2] < 0):
            raise PlateBundleError("plate bundle %s: light.json sun_dir %r is not a downward direction" % (self.dir, k))
        self.sun_dir = k / np.linalg.norm(k)
        self.sun_lux = float(self.light["sun_illuminance_lux"])
        pj = json.loads(r["probes.json"].decode("utf-8"))
        self.probes = []
        for p in pj["probes"]:
            name = str(p["file"])
            if name not in r:
                raise PlateBundleError("plate bundle %s: probe %s is not in the file list" % (self.dir, name))
            self.probes.append({"xyz": [float(v) for v in p["xyz"]], "ktx": r[name], "file": name})
        if not self.probes:
            raise PlateBundleError("plate bundle %s: no probe" % self.dir)

    def _check_ldr(self, look):
        flat_lin = self.plate_lin.reshape(-1, 3)[::LDR_CHECK_STRIDE]
        flat_ldr = self.plate_ldr.reshape(-1, 3)[::LDR_CHECK_STRIDE]
        d = np.abs(plate_look.display(flat_lin, look).astype(np.int16) - flat_ldr.astype(np.int16)).max(axis=1)
        share = float(np.mean(d > 0))
        if int(d.max()) > 1 or share > LDR_CHECK_MAX_SHARE:
            raise PlateBundleError("plate bundle %s: plate_ldr.png is not the look's display transform of plate_lin "
                                   "(max %d levels, %.3f %% of sampled pixels differ)" % (self.dir, int(d.max()), 100 * share))
        self.ldr_check = {"sampled": int(len(d)), "differ": int((d > 0).sum()), "max_levels": int(d.max())}

    def facts(self):
        return {"bundle_meta_sha256": self.meta_sha256, "site_package_sha256": self.meta.get("site_package_sha256"),
                "probes": len(self.probes), "ldr_check": getattr(self, "ldr_check", None)}
