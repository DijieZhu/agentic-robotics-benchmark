"""The hidden truth (tests/gt/), loaded and checked once per verify.

    truth = Truth(gt, cfg, vocabulary)      # gt: {key: path} (grader_config.json "gt", resolved); raises TruthError
    truth.twin, truth.cameras, truth.region, truth.units, truth.unit, truth.surfaces, truth.c3
    truth.path(key), truth.exists(key), truth.load(key)        # module-owned files, parsed and cached
    truth.static_scene(), truth.full_scene()                   # raycast scenes of the true twin (site frame)

The true twin goes through the same public gate as a submission and must pass it with no exclusion and labels from
the vocabulary. Every file is checked against it when it is loaded (ids, counts, shapes; the true units' surface samples are
recomputed from the true twin and checked against items_surface.npz's digest); the deeper
reproduction checks (cameras, c3 rays, zones) are part of the self-test (part2.selftest).
"""
import json
import os

import numpy as np

from . import load


class TruthError(RuntimeError):
    """The truth fails its own checks: a task error, never the agent's fault."""


class TrueCamera:
    def __init__(self, rec):
        self.photo = int(rec["photo"])
        self.name = str(rec["name"])
        self.W, self.H = int(rec["width"]), int(rec["height"])
        self.K = np.array([[float(rec["fx"]), 0.0, float(rec["cx"])], [0.0, float(rec["fy"]), float(rec["cy"])],
                           [0.0, 0.0, 1.0]])
        self.C = np.asarray(rec["centre"], float).reshape(3)
        self.T = np.asarray(rec["R_site_from_cam"], float).reshape(3, 3)
        if self.name != "photo_%03d" % self.photo:
            raise TruthError("cameras_true.json: camera %r is not photo_%03d" % (self.name, self.photo))
        if abs(np.linalg.det(self.T) - 1.0) > 1e-6 or np.max(np.abs(self.T.T @ self.T - np.eye(3))) > 1e-6:
            raise TruthError("cameras_true.json: %s has no proper rotation" % self.name)


class TrueUnit:
    FIELDS = ("id", "label", "alt_labels", "zone", "symmetry", "rest", "ignore", "lift_probe")

    def __init__(self, rec, k):
        for f in self.FIELDS:
            if f not in rec:
                raise TruthError("items_truth.json: unit %s lacks %r" % (rec.get("id"), f))
        self.k = int(k)
        self.id = str(rec["id"])
        self.label = str(rec["label"])
        self.alt_labels = [str(v) for v in rec["alt_labels"]]
        self.zone = rec["zone"]
        self.symmetry = str(rec["symmetry"])
        self.rest = str(rec["rest"])
        self.ignore = bool(rec["ignore"])
        self.lift_probe = bool(rec["lift_probe"])
        self.notes = rec.get("notes", "")
        if self.symmetry not in ("none", "c2", "c4", "free"):
            raise TruthError("items_truth.json: unit %s has symmetry %r" % (self.id, self.symmetry))
        if self.rest not in ("static", "held"):
            raise TruthError("items_truth.json: unit %s has rest %r" % (self.id, self.rest))

    def labels(self):
        return {self.label, *self.alt_labels}


def _read_json(path, fmt):
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError) as exc:
        raise TruthError("%s cannot be read: %s" % (os.path.basename(path), str(exc)[:200]))
    if fmt is not None and d.get("format") != fmt:
        raise TruthError("%s has format %r, not %r" % (os.path.basename(path), d.get("format"), fmt))
    return d


def _read_npz(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError) as exc:
        raise TruthError("%s cannot be read: %s" % (os.path.basename(path), str(exc)[:200]))


C3_ARRAYS = {"classes": 1, "class": 1, "photo": 1, "pixel": 2, "origin": 2, "direction": 2, "point": 2,
             "normal": 2, "range": 1, "incidence_deg": 1}


class Truth:
    def __init__(self, gt, cfg, vocabulary=None):
        from . import match, zones
        self.cfg = cfg
        self.paths = dict(gt)
        self.vocabulary = load.vocabulary(vocabulary)
        self.twin = load.load_twin(self.paths["twin"], self.vocabulary)
        tw = self.twin
        if not tw.ok:
            raise TruthError("the true twin fails the gate: %s" % "; ".join(tw.reasons[:5]))
        bad = [u["body"] for u in tw.units if u["status"] != "ok"]
        if bad:
            why = {u["body"]: u for u in tw.report["units"] if u["body"] in bad}
            raise TruthError("true units excluded by the gate: %s" % "; ".join(
                "%s (%s)" % (b, "; ".join(why[b].get("reasons", [])[:2])) for b in bad[:5]))
        badlab = [u["body"] for u in tw.units if not u["label_ok"]]
        if badlab:
            raise TruthError("true labels outside the class vocabulary: %s" % ", ".join(badlab[:5]))
        # cameras
        cams = _read_json(self.paths["cameras_true"], "site-cameras-true/1")
        if cams.get("pixel_convention") != "opencv-centre" or cams.get("frame") != "site":
            raise TruthError("cameras_true.json must be in the site frame with the opencv-centre pixel convention")
        self.cameras = {}
        for rec in cams["cameras"]:
            c = TrueCamera(rec)
            if c.photo in self.cameras:
                raise TruthError("cameras_true.json lists photo %d twice" % c.photo)
            self.cameras[c.photo] = c
        # region and zones
        self.region = zones.Region(_read_json(self.paths["region"], "site-region/1"))
        # units
        it = _read_json(self.paths["items_truth"], "site-items-truth/1")
        self.units = [TrueUnit(r, k) for k, r in enumerate(it["units"])]
        self.unit = {u.id: u for u in self.units}
        if len(self.unit) != len(self.units):
            raise TruthError("items_truth.json repeats a unit id")
        twin_units = {u["body"]: u for u in tw.units}
        if set(twin_units) != set(self.unit):
            raise TruthError("items_truth.json and the true twin's units differ: only in the truth %s, only in the "
                             "twin %s" % (sorted(set(self.unit) - set(twin_units))[:5],
                                          sorted(set(twin_units) - set(self.unit))[:5]))
        for u in self.units:
            if twin_units[u.id]["label"] != u.label:
                raise TruthError("unit %s: label %r in the twin, %r in items_truth.json" % (
                    u.id, twin_units[u.id]["label"], u.label))
            if u.zone not in self.region.zone_ids and not u.ignore:
                raise TruthError("unit %s has zone %r, which region.json does not define" % (u.id, u.zone))
            for lab in u.labels():
                if lab not in self.vocabulary:
                    raise TruthError("unit %s: label %r is not in the vocabulary" % (u.id, lab))
        # surfaces: recomputed from the true twin and checked against the stored digest
        self.surfaces = match.TrueSurfaces(_read_npz(self.paths["items_surface"]), self.units, cfg, tw)
        # c3 samples
        self.c3 = _read_npz(self.paths["c3_samples"])
        n = None
        for k, nd in C3_ARRAYS.items():
            if k not in self.c3:
                raise TruthError("c3_samples.npz lacks %r" % k)
            if k != "classes":
                if n is None:
                    n = len(self.c3[k])
                if len(self.c3[k]) != n or self.c3[k].ndim != nd:
                    raise TruthError("c3_samples.npz: %r has shape %s" % (k, self.c3[k].shape))
        self.n_c3 = int(n or 0)
        self._cache = {}
        self._scenes = {}

    # files owned by the criterion modules
    def path(self, key):
        return self.paths.get(key)

    def exists(self, key):
        p = self.paths.get(key)
        return bool(p) and os.path.exists(p)

    def load(self, key):
        """A truth file by its gt key, parsed once: .json -> dict, .npz -> {name: array}; TruthError when absent."""
        if key not in self._cache:
            p = self.paths.get(key)
            if not p or not os.path.exists(p):
                raise TruthError("the truth file %r is missing" % key)
            self._cache[key] = _read_npz(p) if p.endswith(".npz") else _read_json(p, None)
        return self._cache[key]

    def static_scene(self):
        if "static" not in self._scenes:
            from . import raycast
            self._scenes["static"] = raycast.static_scene(self.twin)
        return self._scenes["static"]

    def full_scene(self):
        if "full" not in self._scenes:
            from . import raycast
            self._scenes["full"] = raycast.full_scene(self.twin)
        return self._scenes["full"]

    def counted_units(self):
        """The true units that count (not 'ignore'), in items_truth order."""
        return [u for u in self.units if not u.ignore]
