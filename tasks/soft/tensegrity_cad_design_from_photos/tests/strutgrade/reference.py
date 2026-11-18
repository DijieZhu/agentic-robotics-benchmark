"""The hidden reference of the strut grader, read from tests/reference/:

  families.json        the 8 designs (mesh file, sha256, size class, symmetry, material) and the 13 instances (tag,
                       family, pose = design file -> strut frame); the designs are the authors' own mesh pieces.
  features.json        holes, flat faces, functional primitives, supports, weights (optional here: the base does
                       not use it, the scoring items do).
  purchased.json       the purchased types (part number, mass, symmetry, anchor constituent, local frame) and their
                       reference instances (local -> strut frame).
  purchased_pins.json  the vendor STEP of every purchased type (path under the purchased root, sha256, which STEP solid
                       is which constituent, its volume (adaptive integration, eps 1e-7) and centroid).
  seen.json            the "can be seen" marks (optional here).

The designs are taken as meshes: a Geo is built directly from each PLY (no BREP, no repair). The vendor
STEPs are read from the purchased root (the image's /opt/task-assets/payload/purchased) in spawned worker processes;
a file whose sha256 differs from its pin, or a solid whose volume or centroid differs from the pin, is a task error.
"""
import json
import os

import numpy as np

from . import detenv
from .geom import (TaskError, apply, geo_from_mesh, inv, mass_props, mesh_mass, mesh_solid, read_brep,
                   register_exact, sha256_file, solids_of, unique_vertices)


def _read_ply(path):
    import trimesh
    m = trimesh.load(path, file_type="ply", process=False, force="mesh")
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def parse_symmetry(sym, V=None):
    """{"kind": none | cyclic | continuous, "order", "flip", "axis_point", "axis_dir", "axis_points"} of a design
    (families.json symmetry block). axis_points: the two ends of the design's extent along its axis (design frame),
    what a symmetric design contributes to the whole-strut alignment."""
    group = str(sym.get("group", "C1"))
    if group.startswith("D_inf"):
        kind, order, flip = "continuous", 0, True
    elif group.startswith("C_inf"):
        kind, order, flip = "continuous", 0, False
    elif group.startswith("C") and group[1:].isdigit() and int(group[1:]) > 1:
        kind, order, flip = "cyclic", int(group[1:]), False
    else:
        kind, order, flip = "none", 1, False
    out = {"group": group, "kind": kind, "order": order, "flip": flip}
    if kind != "none":
        p = np.asarray(sym["axis_point_mm"], float)
        d = np.asarray(sym["axis_dir"], float)
        d = d / np.linalg.norm(d)
        out["axis_point"], out["axis_dir"] = p, d
        if V is not None and len(V):
            h = (np.asarray(V, float) - p) @ d
            out["axis_points"] = np.array([p + float(h.min()) * d, p + float(h.max()) * d])
            out["axis_length"] = float(h.max() - h.min())
    return out


def _step_worker(job):
    tname, path, out_dir = job
    from OCP.BRepTools import BRepTools
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    r = STEPControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        raise IOError("cannot read %s" % path)
    r.TransferRoots()
    shape = r.OneShape()
    out = []
    for k, s in enumerate(solids_of(shape)):
        p = os.path.join(out_dir, "%s_%03d.brep" % (tname, k))
        BRepTools.Write_s(s, p)
        out.append(p)
    return tname, out


class StepReader(object):
    """Read the vendor STEPs (the slow part of loading the reference) in spawned worker processes, started at once;
    each worker writes the STEP's solids, in the reader's order, as BREP files. result() = {type: [brep paths]}."""

    def __init__(self, jobs, tmpdir, workers):
        out_dir = os.path.join(tmpdir, "vendor")
        os.makedirs(out_dir, exist_ok=True)
        self.jobs = sorted([(t, p, out_dir) for t, p in jobs], key=lambda j: (-os.path.getsize(j[1]), j[0]))
        self.ex = self.futs = None
        if workers > 1 and len(self.jobs) > 1:
            import concurrent.futures as cf
            import multiprocessing as mp
            self.ex = cf.ProcessPoolExecutor(max_workers=min(workers, len(self.jobs)), mp_context=mp.get_context("spawn"))
            self.futs = [self.ex.submit(_step_worker, j) for j in self.jobs]

    def result(self):
        if self.ex is None:
            return dict(_step_worker(j) for j in self.jobs)
        try:
            return dict(f.result() for f in self.futs)
        finally:
            self.ex.shutdown()


class Reference(object):
    """The reference (module docstring). Attributes:

    designs   key -> {"key", "file", "V", "F", "geo" (Geo, design frame), "volume", "com", "I", "size_class",
              "symmetry" (parse_symmetry), "material", "density_g_mm3", "tags", "extent"}
    instances tag -> {"tag", "family", "end", "W" (design -> strut, 4x4)}
    tags, family_of (tag -> design key), family_tags (design key -> sorted tags)
    features, seen   the raw JSON of features.json / seen.json, or None when absent
    types     purchased type -> {"name", "pn", "mass_g", "symmetry", "axis_local", "cons" [constituents], "anchor",
              "V", "centroid", "spec"}; constituent = {"id", "type", "k", "V", "com", "I" (default integration),
              "V_pin" (eps 1e-7), "verts", "size", "geo", "cls"}
    classes   congruence classes of all constituents (as in the UAV CAD task): {"id", "rep", "members"
              [(constituent, M)], "types"}
    placements purchased reference instance id -> {"id", "type", "W" (local -> strut)}
    """

    def __init__(self, ref_dir, purchased_root, cfg, tmpdir, log=None):
        self.dir, self.cfg, self.tmp = ref_dir, cfg, tmpdir
        self.log = log or (lambda m: None)
        with open(os.path.join(ref_dir, "families.json")) as fh:
            self.fam_spec = json.load(fh)
        if not str(self.fam_spec.get("schema", "")).startswith("tensegrity_cad_design_from_photos/reference/families"):
            raise TaskError("families.json schema %r" % self.fam_spec.get("schema"))
        with open(os.path.join(ref_dir, "purchased.json")) as fh:
            self.pur_spec = json.load(fh)
        pending = self._start_purchased(purchased_root)
        mats = self.fam_spec["materials"]
        self.designs = {}
        for fam in sorted(self.fam_spec["families"], key=lambda f: f["key"]):
            path = os.path.join(ref_dir, fam["file"])
            if not os.path.isfile(path):
                raise TaskError("reference design %s is missing: %s" % (fam["key"], path))
            if sha256_file(path) != fam["sha256"]:
                raise TaskError("reference design %s does not match its sha256" % fam["key"])
            V, F = _read_ply(path)
            if len(F) != int(fam["triangles"]):
                raise TaskError("reference design %s has %d triangles, families.json %d" % (fam["key"], len(F), fam["triangles"]))
            geo = geo_from_mesh(V, F, detenv.seed_for(cfg, "reference", fam["key"]), cfg)
            if geo.man is None:
                raise TaskError("reference design %s is not a closed manifold" % fam["key"])
            vol, com, I = mesh_mass(V, F)
            self.designs[fam["key"]] = {
                "key": fam["key"], "file": fam["file"], "V": V, "F": F, "geo": geo, "volume": vol, "com": com, "I": I,
                "size_class": fam["size_class"], "symmetry": parse_symmetry(fam.get("symmetry") or {}, V),
                "material": fam["material"], "density_g_mm3": float(mats[fam["material"]]["density_g_cm3"]) / 1000.0,
                "tags": list(fam["tags"]), "extent": float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))}
        self.instances = {}
        for it in self.fam_spec["instances"]:
            if it["family"] not in self.designs:
                raise TaskError("instance %s names an unknown design %s" % (it["tag"], it["family"]))
            self.instances[it["tag"]] = {"tag": it["tag"], "family": it["family"], "end": it.get("end"),
                                         "W": np.asarray(it["pose"], float)}
        self.tags = sorted(self.instances)
        self.family_of = {t: v["family"] for t, v in self.instances.items()}
        self.family_tags = {}
        for t in self.tags:
            self.family_tags.setdefault(self.family_of[t], []).append(t)
        self.features = self._optional_json("features.json")
        self.seen = self._optional_json("seen.json")
        self.materials = mats
        self._load_purchased(pending)
        self.placements = {}
        for pl in self.pur_spec["instances"]:
            if pl["type"] not in self.types:
                raise TaskError("purchased instance %s names an unknown type %s" % (pl["id"], pl["type"]))
            self.placements[pl["id"]] = {"id": pl["id"], "type": pl["type"], "end": pl.get("end"),
                                         "W": np.asarray(pl["local_to_strut"], float), "spec": pl}

    def _optional_json(self, name):
        p = os.path.join(self.dir, name)
        if not os.path.isfile(p):
            return None
        with open(p) as fh:
            return json.load(fh)

    # ------------------------------------------------------------------ purchased types
    def _start_purchased(self, root):
        with open(os.path.join(self.dir, "purchased_pins.json")) as fh:
            pins = json.load(fh)
        jobs = []
        for tname in sorted(self.pur_spec["types"]):
            pin = pins["types"].get(tname)
            if pin is None:
                raise TaskError("purchased type %s has no pin" % tname)
            path = os.path.join(root, pin["step"])
            if not os.path.isfile(path):
                raise TaskError("the vendor file of %s is missing: %s" % (tname, path))
            digest = sha256_file(path)
            if digest != pin["sha256"]:
                raise TaskError("the vendor file %s does not match its pin (sha256 %s, pinned %s)" % (path, digest, pin["sha256"]))
            jobs.append((tname, path))
        workers = int(self.cfg.get("recognition", {}).get("step_workers", 2))
        return pins, StepReader(jobs, self.tmp, workers)

    def _load_purchased(self, pending):
        pins, reader = pending
        breps = reader.result()
        self.types = {}
        for tname in sorted(self.pur_spec["types"]):
            t = self.pur_spec["types"][tname]
            pin = pins["types"][tname]
            files = breps[tname]
            if len(files) != len(pin["solids"]):
                raise TaskError("%s: the vendor STEP has %d solids, the pin %d" % (tname, len(files), len(pin["solids"])))
            byid = {c["id"]: c for c in t["constituents"]}
            order = [c["id"] for c in t["constituents"]]
            cons = [None] * len(order)
            for k, (f, ps) in enumerate(zip(files, pin["solids"])):
                s = solids_of(read_brep(f))[0]
                Vp, cp, _Ip = mass_props(s, eps=float(self.cfg["recognition"].get("pin_volume_eps", 1e-7)))
                if abs(Vp - float(ps["volume_mm3"])) > 1e-5 * max(1.0, abs(float(ps["volume_mm3"]))) or \
                        np.linalg.norm(cp - np.asarray(ps["centroid_mm"], float)) > 1e-3:
                    raise TaskError("%s solid %d does not match its pin (volume %.6f vs %.6f)" % (tname, k, Vp, ps["volume_mm3"]))
                V, c, I = mass_props(s)
                cid = ps["constituent"]
                geo = mesh_solid(s, self.cfg, self.tmp, detenv.seed_for(self.cfg, "purchased", cid), volume=V)
                verts = unique_vertices(s)
                ext = verts.max(axis=0) - verts.min(axis=0) if len(verts) else np.zeros(3)
                cons[order.index(cid)] = {"id": cid, "type": tname, "k": order.index(cid), "V": V, "com": c, "I": I,
                                          "V_pin": Vp, "verts": verts, "size": float(np.linalg.norm(ext)), "geo": geo,
                                          "shape": s, "role": byid[cid].get("source_role")}
            if any(c is None for c in cons):
                raise TaskError("%s: a constituent has no vendor solid" % tname)
            vol = sum(c["V"] for c in cons)
            cen = sum(c["com"] * c["V"] for c in cons) / vol
            anchor = order.index(t["anchor"])
            sym = t.get("symmetry") or {}
            axis = t.get("axis_local") or t.get("output_axis_local") or {"point_mm": [0, 0, 0], "dir": sym.get("axis_local", [0, 0, 1])}
            self.types[tname] = {"name": tname, "pn": t.get("part_number"), "mass_g": float(t["mass_g"]), "cons": cons,
                                 "anchor": anchor, "V": vol, "centroid": cen, "spec": t,
                                 "symmetry": {"continuous": bool(sym.get("continuous", False)),
                                              "flip": bool(sym.get("flip_symmetric", False)),
                                              "order": sym.get("order", 1),
                                              "axis_point": np.asarray(axis.get("point_mm", [0, 0, 0]), float),
                                              "axis_dir": np.asarray(axis.get("dir", sym.get("axis_local", [0, 0, 1])), float)}}
            st = self.types[tname]["symmetry"]
            if st["continuous"]:
                allv = np.vstack([c["verts"] for c in cons])
                d = st["axis_dir"] / np.linalg.norm(st["axis_dir"])
                h = (allv - st["axis_point"]) @ d
                st["axis_points"] = np.array([st["axis_point"] + float(h.min()) * d, st["axis_point"] + float(h.max()) * d])
        self._classes()

    def _classes(self):
        """Congruence classes of all constituents (exact registration on their vertex sets): the representative is the
        first member; M = the member's pose relative to the representative (rep local -> member local)."""
        tol = float(self.cfg["recognition"]["exact_tol_mm"])
        self.classes = []
        for tname in sorted(self.types):
            for c in self.types[tname]["cons"]:
                placed = False
                for cl in self.classes:
                    rep = cl["rep"]
                    if len(rep["verts"]) != len(c["verts"]) or abs(rep["V"] - c["V"]) > 1e-6 * max(1.0, rep["V"]):
                        continue
                    got = register_exact(rep, c, tol)
                    if got is not None:
                        cl["members"].append((c, got[0]))
                        c["cls"] = cl["id"]
                        placed = True
                        break
                if not placed:
                    cl = {"id": len(self.classes), "rep": c, "members": [(c, np.eye(4))]}
                    c["cls"] = cl["id"]
                    self.classes.append(cl)
        for cl in self.classes:
            cl["types"] = sorted(set(m[0]["type"] for m in cl["members"]))

    # ------------------------------------------------------------------ helpers for the items
    def instance_points(self, tag, P_design):
        """Design-frame points of an instance's design, in the strut frame."""
        return apply(self.instances[tag]["W"], P_design)

    def design_to_instance(self, tag):
        return np.array(self.instances[tag]["W"], float)

    def instance_to_design(self, tag):
        return inv(self.instances[tag]["W"])
