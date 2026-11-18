"""Gates and local rules of the strut grader.

Only three conditions zero the whole document (GradeContext.gate = {"reason", "message"}):

  unreadable      the document cannot be read: no extraction, the extractor's own errors for the document (it does
                  not open; more shape leaves or solids than the reader reads; more solids than gates.max_solids)
  no_solid        the Assembly (or, without an Assembly, the document's visible objects) holds no usable solid
  too_few_fitted  fewer than gates.min_fittable (3) pieces can be fitted: tagged parts that are a valid solid plus
                  recognised purchased instances (the whole-strut alignment needs three)

Everything else is local and costs only the piece concerned, with a diagnostic and a statistic:
  - a tagged leaf (label P01-P13) that is not a PartDesign Body holding exactly one valid closed solid: the tag gets
    no credit (no part for that tag); its valid solids stay in the document as extra solids;
  - a tag on more than one leaf (also one Body reached twice through links): none of them is credited; their solids
    are extra solids;
  - an untagged leaf with invalid solids, open shells or loose faces, or a solid that cannot be read: the bad pieces
    are left out (counted), its valid solids stay;
  - the same object reached twice at the same place (a folder and a part listing it) is one solid;
  - no Assembly: the visible solids are graded, without joint credit (statistic no_assembly).
Extra solids (untagged and not recognised as purchased, or demoted by the rules above) take part in mass, fit and
clearance as obstacles at the printed-part density; they are not fitted and do not take part in the alignment.
"""
import json
import os

import numpy as np

from .geom import TaskError, brep_valid, heal_tolerances, mass_props, read_brep, solids_of, unique_vertices

FORMAT_PREFIX = "strut-fcstd-extract/"
REASONS = ("unreadable", "no_solid", "too_few_fitted")


class Submission(object):
    """The extracted document: every usable solid with its tag (or None), component, mass properties and vertices.

    solids: [{"sid", "tag" (credited tag or None), "label", "component", "leaf_type", "file", "shape", "V", "com",
              "I", "verts", "size", "demoted" (why a tagged solid lost its tag, else absent)}]
    gate:   None, or {"reason": "unreadable" | "no_solid", "message"}
    model:  the extractor's model.json; stats: the local-rule statistics; joints, grounded, no_assembly."""

    def __init__(self, ex_dir, tags, cfg, diags):
        self.solids = []
        self.cfg = cfg
        self.gate = None
        self.model = {}
        self.no_assembly = False
        self.stats = st = {"tags_invalid": 0, "tags_duplicated": 0, "ignored_non_solid_leaves": 0,
                           "ignored_bad_solids": 0, "ignored_unreadable_leaves": 0, "no_assembly": 0,
                           "duplicate_leaves_ignored": 0, "solids_healed": 0}
        self._notes = 0
        path = os.path.join(ex_dir, "model.json")
        if not os.path.isfile(path):
            self._zero("unreadable", "No readable FreeCAD document.", diags)
            return
        try:
            with open(path) as fh:
                model = json.load(fh)
        except ValueError as exc:
            raise TaskError("the extraction's model.json is not JSON: %s" % exc)
        if not isinstance(model, dict) or not str(model.get("format", "")).startswith(FORMAT_PREFIX):
            raise TaskError("the extraction's model.json has an unknown format")
        if model.get("extractor_failure"):
            raise TaskError("the FreeCAD reader failed: " + " | ".join(str(e) for e in model.get("errors", []))[-1500:])
        self.model = model
        if model.get("errors"):
            self._zero("unreadable", "The document cannot be read: %s" % "; ".join(str(e)[:300] for e in model["errors"][:3]), diags)
            return
        if model.get("no_assembly"):
            self.no_assembly = True
            st["no_assembly"] = 1
            diags.append("No Assembly in the document: its visible solids are graded, without joint credit.")
        G = cfg["gates"]
        ex_real = os.path.realpath(ex_dir)
        seen = set()
        tagged = {}          # tag -> [(leaf, comp name, good solid records, single valid Body?)]
        untagged = []        # (leaf, comp name, solid record)
        tagset = set(tags)

        def note(msg):
            if self._notes < 25:
                diags.append("Local: " + msg)
                self._notes += 1

        for comp in model.get("components", []):
            for leaf in comp.get("leaves", []):
                label = (leaf.get("label") or "").strip()
                tag = label if label in tagset else None
                where = "%s (%s)" % (label or leaf.get("target"), leaf.get("type"))
                if leaf.get("error"):
                    st["ignored_unreadable_leaves"] += 1
                    note("the shape of %s cannot be read (%s): left out" % (where, str(leaf["error"])[:120]))
                    if tag:
                        tagged.setdefault(tag, []).append((leaf, comp["name"], [], False))
                    continue
                key = (leaf.get("target"), tuple(np.round(np.asarray(leaf.get("placement", np.eye(4)), float).ravel(), 6)))
                if key in seen:
                    st["duplicate_leaves_ignored"] += 1
                    continue
                seen.add(key)
                sols = leaf.get("solids", [])
                good = [x for x in sols if x.get("valid") and x.get("closed")]
                free = int(leaf.get("n_free_faces", 0) or 0)
                if tag is not None:
                    single = (leaf.get("type") == "PartDesign::Body" and len(sols) == 1 and len(good) == 1 and
                              bool(leaf.get("valid")) and free == 0)
                    tagged.setdefault(tag, []).append((leaf, comp["name"], good, single))
                    continue
                if not sols:
                    st["ignored_non_solid_leaves"] += 1
                    if free:
                        note("%s has open faces and no solid: left out" % where)
                    continue
                if free or len(good) < len(sols) or not leaf.get("valid"):
                    st["ignored_bad_solids"] += (len(sols) - len(good)) + (1 if free else 0)
                    note("%s has %d invalid or open solid(s)%s: those are left out" % (
                        where, len(sols) - len(good), " and loose faces" if free else ""))
                for x in good:
                    untagged.append((leaf, comp["name"], x))

        def add(leaf, comp_name, x, tag, demoted=None):
            p = os.path.realpath(os.path.join(ex_dir, x["brep"]))
            if os.path.commonpath([ex_real, p]) != ex_real:
                raise TaskError("the extractor returned an escaping shape path")
            rec = {"sid": len(self.solids), "tag": tag, "label": (leaf.get("label") or "").strip(),
                   "component": comp_name, "leaf_type": leaf.get("type"), "file": p}
            if demoted:
                rec["demoted"] = demoted
            self.solids.append(rec)
            return rec

        for tag in sorted(tagged):
            entries = tagged[tag]
            if len(entries) > 1:
                st["tags_duplicated"] += 1
                note("tag %s is on %d leaves: none of them is credited; their solids count as extra solids" % (tag, len(entries)))
                for leaf, cname, good, _single in entries:
                    for x in good:
                        add(leaf, cname, x, None, demoted="duplicate tag " + tag)
                continue
            leaf, cname, good, single = entries[0]
            if single:
                add(leaf, cname, good[0], tag)
                continue
            st["tags_invalid"] += 1
            n = len(leaf.get("solids", []))
            note("%s is a %s with %d solid(s), %d valid and closed: not a PartDesign Body of exactly one valid closed "
                 "solid, so the tag gets no credit; its valid solids count as extra solids" % (tag, leaf.get("type"), n, len(good)))
            for x in good:
                add(leaf, cname, x, None, demoted="invalid tagged leaf " + tag)
        for leaf, cname, x in untagged:
            add(leaf, cname, x, None)
        if len(self.solids) > int(G["max_solids"]):
            self.solids = []
            self._zero("unreadable", "Too many solids in the document (more than %d): it is not read." % int(G["max_solids"]), diags)
            return
        keep = []
        for s in self.solids:
            try:
                shape = read_brep(s["file"])
                sols = solids_of(shape)
                if len(sols) != 1:
                    raise ValueError("%d solids in one solid file" % len(sols))
                s["shape"] = sols[0]
                if G.get("recheck_validity", True) and not brep_valid(sols[0]):
                    if heal_tolerances(sols[0]):
                        st["solids_healed"] += 1
                    else:
                        raise ValueError("BRepCheck reports an invalid solid")
                V, c, I = mass_props(sols[0])
                if not V > float(G["min_volume_mm3"]):
                    raise ValueError("no volume")
                s["V"], s["com"], s["I"] = V, c, I
                s["verts"] = unique_vertices(sols[0])
                ext = s["verts"].max(axis=0) - s["verts"].min(axis=0) if len(s["verts"]) else np.zeros(3)
                s["size"] = float(np.linalg.norm(ext))
            except Exception as exc:  # noqa: BLE001 - a solid that does not load is left out (local)
                st["ignored_bad_solids"] += 1
                note("a solid of %s cannot be used (%s): left out" % (s["label"] or s["component"], str(exc)[:100]))
                if s["tag"] is not None:
                    st["tags_invalid"] += 1
                continue
            keep.append(s)
        for k, s in enumerate(keep):
            s["sid"] = k
        self.solids = keep
        if not self.solids:
            self._zero("no_solid", "The %s holds no solid." % ("document" if self.no_assembly else "Assembly"), diags)
            return
        self.joints = list(model.get("joints", []))
        self.grounded = [g for g in model.get("grounded", []) if g]

    def _zero(self, reason, message, diags):
        self.gate = {"reason": reason, "message": message}
        diags.append(message)

    @property
    def ok(self):
        return self.gate is None


def fittable_count(ctx):
    """Tagged parts that are a usable solid (a credited tag) plus recognised purchased instances."""
    return len(ctx.parts) + len(ctx.instances)


def fit_gate(ctx):
    """The third whole-document zero: fewer than gates.min_fittable pieces to fit. Returns the gate or None."""
    n = fittable_count(ctx)
    need = int(ctx.cfg["gates"].get("min_fittable", 3))
    ctx.stats["fittable_pieces"] = float(n)
    if n < need:
        return {"reason": "too_few_fitted",
                "message": "Only %d piece(s) can be fitted (tagged parts that are a valid solid and recognised purchased "
                           "instances); the alignment of the strut needs %d: the document scores 0." % (n, need)}
    return None
