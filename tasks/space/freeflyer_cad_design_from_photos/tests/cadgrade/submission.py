"""The extraction (fcstd_extract.py's model.json, format uav-fcstd-extract/2) read into the grader's tables.

A member is a component of the Assembly as the reader lists it (members of document folders inside the Assembly
included); without an Assembly the reader lists the document's visible top-level objects instead. A member's tag is its
own Label (the component's Label, ends trimmed, exact): P01..P12. Its solids are every solid of its shape leaves (a
Body, a Part feature, the members of an App::Part or nested container, the target of a link). Kept: the solids the
reader marks valid and closed; left out and reported: invalid or open solids, open shells and loose faces, leaves whose
shape cannot be read. The same object reached twice at the same placement inside one member counts once. A tag on two
or more members is void; members without a tag are ignored; both are reported.
"""
import json
import os

import numpy as np

from .util import TaskError

FORMAT_PREFIX = "uav-fcstd-extract/"


class Member(object):
    def __init__(self, name, label, tag, mtype, folder):
        self.name, self.label, self.tag, self.type, self.folder = name, label, tag, mtype, folder
        self.solids = []          # [{"file", "leaf", "volume_reader"}] kept by the reader's flags
        self.left_out = 0         # invalid or open solids, unreadable leaves
        self.free_geometry = 0    # leaves with open shells or loose faces
        self.wires = 0            # leaves with wires only (sketches, datum lines): not surfaces, ignored
        self.duplicates = 0


class Submission(object):
    """readable False (with reasons) when the document cannot be read: the reader's errors, its solid limit, no
    model.json. Every other finding is local and goes to the notes."""

    def __init__(self, ex_dir, cfg, tags):
        self.dir = ex_dir
        self.readable = False
        self.reasons = []
        self.notes = []
        self.members = []
        self.joints = []
        self.grounded = []
        self.no_assembly = False
        self.stats = {}
        path = os.path.join(ex_dir, "model.json")
        if not os.path.isfile(path):
            self.reasons.append("the FreeCAD reader wrote no result")
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
            self.reasons.extend("the document cannot be read: %s" % str(e)[:300] for e in model["errors"][:5])
            return
        self.no_assembly = bool(model.get("no_assembly"))
        n_solids = sum(len(leaf.get("solids", [])) for c in model.get("components", []) for leaf in c.get("leaves", []))
        self.stats["n_solids_in_document"] = n_solids
        if n_solids > int(cfg["gates"]["max_solids"]):
            self.reasons.append("the document holds %d solids, more than %d" % (n_solids, int(cfg["gates"]["max_solids"])))
            return
        ex_real = os.path.realpath(ex_dir)
        tagset = set(tags)
        for comp in model.get("components", []):
            label = str(comp.get("label") or "")
            tag = label.strip() if label.strip() in tagset else None
            m = Member(str(comp.get("name")), label, tag, comp.get("type"), comp.get("folder"))
            self.members.append(m)
            if comp.get("error"):
                m.left_out += 1
                self.note("the member %s cannot be listed (%s)" % (self._show(m), str(comp["error"])[:120]))
            seen = set()
            for leaf in comp.get("leaves", []):
                if leaf.get("error"):
                    m.left_out += 1
                    self.note("a shape of %s cannot be read (%s): left out" % (self._show(m), str(leaf["error"])[:120]))
                    continue
                key = (leaf.get("target"), tuple(np.round(np.asarray(leaf.get("placement", np.eye(4)), float).ravel(), 6)))
                if key in seen:
                    m.duplicates += 1
                    continue
                seen.add(key)
                sols = leaf.get("solids", [])
                good = [x for x in sols if x.get("valid") and x.get("closed")]
                if len(good) < len(sols):
                    m.left_out += len(sols) - len(good)
                    self.note("%s has %d invalid or open solid(s): left out" % (self._show(m), len(sols) - len(good)))
                if int(leaf.get("n_free_faces", 0) or 0):
                    m.free_geometry += 1
                    if m.tag:
                        self.note("%s has open shells or loose faces: left out" % self._show(m))
                elif not sols and int(leaf.get("n_free_edges", 0) or 0):
                    m.wires += 1
                for x in good:
                    p = os.path.realpath(os.path.join(ex_dir, str(x.get("brep"))))
                    if os.path.commonpath([ex_real, p]) != ex_real:
                        raise TaskError("the extraction names a shape outside its folder")
                    m.solids.append({"file": p, "volume_reader": float(x.get("volume", 0.0) or 0.0)})
        self.joints = [j for j in model.get("joints", []) if isinstance(j, dict)]
        self.grounded = [g for g in model.get("grounded", []) if g]
        self.readable = True
        by_tag = {}
        for m in self.members:
            if m.tag:
                by_tag.setdefault(m.tag, []).append(m)
        self.void = sorted(t for t, ms in by_tag.items() if len(ms) > 1)
        for t in self.void:
            self.note("tag %s is on %d members (%s): void" % (t, len(by_tag[t]), ", ".join(m.name for m in by_tag[t])))
        untagged = [m for m in self.members if not m.tag]
        if untagged:
            self.note("%d member(s) without a tag ignored: %s" % (
                len(untagged), ", ".join(repr(m.label)[:40] for m in untagged[:6])))
        self.by_tag = {t: ms[0] for t, ms in by_tag.items() if len(ms) == 1}
        self.stats.update({
            "n_members": len(self.members), "n_members_tagged": sum(1 for m in self.members if m.tag),
            "n_members_untagged": len(untagged), "n_tags_void": len(self.void),
            "n_solids_left_out": sum(m.left_out for m in self.members),
            "n_members_with_free_geometry": sum(1 for m in self.members if m.free_geometry),
            "n_duplicate_leaves": sum(m.duplicates for m in self.members),
            "n_joints": len(self.joints), "n_grounded": len(self.grounded), "no_assembly": 1 if self.no_assembly else 0})

    def note(self, msg):
        if len(self.notes) < 25:
            self.notes.append("Local: " + msg)

    @staticmethod
    def _show(m):
        return "%s (%s)" % (m.tag or repr(m.label)[:40], m.type)

    def member_table(self, valid_counts):
        """The members table of the combining layer: valid_counts {member name: usable solids}."""
        return [{"name": m.name, "label": m.label, "valid_solids": int(valid_counts.get(m.name, 0))} for m in self.members]
