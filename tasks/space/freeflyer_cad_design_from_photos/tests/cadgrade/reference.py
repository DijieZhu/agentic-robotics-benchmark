"""The standard, read through ../makerstd: every tag's maker triangles at its place in configuration Z (body frame, mm),
the graded samples of its maker file (the seen tables), dense samples of its triangles, and the six hinge axis lines.

At construction every pin is checked (the ten meshes are read, xz-compressed ones decompressed), placement.json is
re-derived from the pinned description, plugin and launch texts and compared item by item (1e-12), and every seen table
is checked against seen.json (its sha256, its record count, its triangle indices, the file's flat-list digest when the
index records it). Any difference is a TaskError: the task is broken, the document is not scored.
"""
import json
import os

import numpy as np

from .geom import area_centroid, sample_count, sample_triangles
from .util import TaskError, apply, rotate, sha256_file

SEEN_SCHEMA = "ffcad.seen_index/1"


class Place(object):
    """One tag's place: its maker triangles, dense samples (with normals and a KD-tree), graded samples, centroid."""

    def __init__(self, tag, maker_file, T_mm, triangles):
        self.tag, self.maker_file, self.T = tag, maker_file, np.asarray(T_mm, float)
        self.triangles = triangles
        self.centroid, self.area = area_centroid(triangles)
        self.dense = self.normals = self.tree = self.tree_coarse = None
        self.graded = None
        self.capped = False


class Reference(object):
    def __init__(self, stage_dir, cfg, seen_index=None, log=None):
        from makerstd import pinned, seen_codec, standard
        from makerstd.errors import MakerStdError
        self.cfg = cfg
        self.log = log or (lambda m: None)
        self.stage_dir = os.path.realpath(stage_dir)
        R = cfg.get("reference", {})
        maker_root = os.path.join(stage_dir, R.get("maker_root", "reference/maker"))
        placement_path = os.path.join(stage_dir, R.get("placement", "reference/placement.json"))
        self.seen_index = os.path.abspath(seen_index or os.path.join(stage_dir, R.get("seen_index", "reference/seen.json")))
        self.tags = sorted(cfg["part_weights"])
        try:
            pins = pinned.load_pins(maker_root)
            std = standard.load(maker_root, placement_path, pins)
            if sorted(std.tags) != self.tags:
                raise TaskError("the standard's tags %s differ from the configuration's" % sorted(std.tags))
            self.files = sorted(set(std.maker_file(t) for t in self.tags))
            self.meshes = {f: std.mesh(f) for f in self.files}
            self.file_triangles = {f: self.meshes[f].triangles_mm() for f in self.files}
            self.places = {}
            for t in self.tags:
                f = std.maker_file(t)
                T = np.asarray(std.T_body_file_mm(t), float)
                self.places[t] = Place(t, f, T, std.tag_triangles_mm(t))
            self.placement = std.placement
        except MakerStdError as exc:
            raise TaskError("the standard cannot be used: %s" % exc)
        self.hinges = self._hinges(cfg)
        self._read_seen(seen_codec)
        self._targets = {}

    # ------------------------------------------------------------------ the hinges
    def _hinges(self, cfg):
        out = {}
        for j in self.placement.get("joints", []):
            if j.get("type", {}).get("text") != "revolute":
                continue
            a, b = j["tags"]
            line = j["axis_line_at_Z"]
            out[frozenset((a, b))] = {"joint": j["name"], "tags": [a, b], "point_mm": np.asarray(line["point_mm"], float),
                                      "dir": np.asarray(line["dir"], float)}
        want = set(frozenset(p) for p in cfg["revolute_pairs"])
        if set(out) != want:
            raise TaskError("the standard's hinges %s differ from the configuration's revolute pairs" % (
                sorted(sorted(k) for k in out)))
        return out

    def hinge(self, pair):
        return self.hinges[frozenset(pair)]

    # ------------------------------------------------------------------ the seen tables
    def _read_seen(self, seen_codec):
        from makerstd.errors import MakerStdError
        path = self.seen_index
        try:
            with open(path, "rb") as fh:
                idx = json.loads(fh.read().decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise TaskError("the seen index cannot be read: %s" % exc)
        if not isinstance(idx, dict) or idx.get("schema") != SEEN_SCHEMA or not isinstance(idx.get("files"), dict):
            raise TaskError("the seen index is not a %s file" % SEEN_SCHEMA)
        root = os.path.realpath(os.path.dirname(path))
        roots = [root, self.stage_dir]          # "bin" relative to seen.json's folder, or to the stage (tests/) folder
        self.seen = {}
        self.seen_info = {"rule": str(idx.get("rule", ""))[:300], "proxy": bool(idx.get("proxy", False))}
        for f in self.files:
            e = idx["files"].get(f)
            if not isinstance(e, dict):
                raise TaskError("the seen index has no table for %s" % f)
            p = None
            for r in roots:
                c = os.path.realpath(os.path.join(r, str(e.get("bin", ""))))
                if os.path.commonpath([r, c]) == r and os.path.isfile(c):
                    p = c
                    break
            if p is None:
                raise TaskError("the seen table of %s is missing" % f)
            if sha256_file(p) != e.get("sha256"):
                raise TaskError("the seen table of %s differs from its sha256 in the index" % f)
            mesh = self.meshes[f]
            if "flat_list_sha256" in e and e["flat_list_sha256"] != mesh.digest():
                raise TaskError("the seen table of %s indexes another triangle numbering" % f)
            with open(p, "rb") as fh:
                data = fh.read()
            try:
                tri, U, V = seen_codec.decode(data, n_triangles=mesh.triangle_count)
            except MakerStdError as exc:
                raise TaskError("the seen table of %s: %s" % (f, exc))
            if int(e.get("records", -1)) != len(tri):
                raise TaskError("the seen table of %s holds %d records, the index says %s" % (f, len(tri), e.get("records")))
            self.seen[f] = seen_codec.points(self.file_triangles[f], tri, U, V)

    # ------------------------------------------------------------------ samples
    def prepare(self, tags):
        """Dense samples (file frame, seeded per maker file, placed per tag), graded samples and KD-trees of the
        given tags' places."""
        from scipy.spatial import cKDTree
        S = self.cfg["sampling"]
        cache = {}
        for t in tags:
            pl = self.places[t]
            if pl.tree is not None:
                continue
            f = pl.maker_file
            if f not in cache:
                tri = self.file_triangles[f]
                _c, area = area_centroid(tri)
                n, capped = sample_count(area, S["dense_per_mm2"], S["dense_min"], S["dense_max"])
                P, N, _i = sample_triangles(tri, n, int(S["seed"]) + 1000 + self.files.index(f))
                cache[f] = (P, N, capped)
            P, N, capped = cache[f]
            pl.dense = apply(pl.T, P)
            pl.normals = rotate(pl.T, N)
            pl.tree = cKDTree(pl.dense)
            pl.tree_coarse = cKDTree(pl.dense[::int(self.cfg["search"]["coarse_stride"])])
            pl.capped = capped
            pl.graded = apply(pl.T, self.seen[f])

    def target(self, tags, stride=1):
        """(points, normals, KD-tree) of the dense samples of the given places together (every stride-th sample)."""
        from scipy.spatial import cKDTree
        key = (tuple(sorted(tags)), int(stride))
        if key not in self._targets:
            self.prepare(tags)
            P = np.vstack([self.places[t].dense[::int(stride)] for t in sorted(tags)])
            N = np.vstack([self.places[t].normals[::int(stride)] for t in sorted(tags)])
            self._targets[key] = (P, N, cKDTree(P))
        return self._targets[key]

    def release(self, tag):
        pl = self.places[tag]
        pl.dense = pl.normals = pl.tree = pl.tree_coarse = pl.graded = None
