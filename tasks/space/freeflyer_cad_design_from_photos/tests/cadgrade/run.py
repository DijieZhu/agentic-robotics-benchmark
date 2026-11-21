"""The grade of one extraction of the submitted document against the standard (every number in grader_config.json).

The rules (the instruction states each of them):

 1. Members and tags. A tag P01..P12 is the Label of one member of the Assembly (members of document folders inside
    it included), ends trimmed, compared exactly; a member may be a PartDesign Body, a Part feature, an App::Part or
    nested container, or an App::Link, and its own Label is read. Every valid closed solid under a tagged member
    belongs to its tag (one or more). Invalid or open solids, open shells and loose faces are left out and reported. A
    tag on two or more members is void: no surface credit, its joints do not count, it is neither joined nor present.
    Members without a tag are ignored and reported. Without an Assembly the reader lists the document's visible
    top-level objects as the members; joint_axes and structure are then 0.
 2. Only these score the whole document 0, with the reason: no document (verify.py); a document that cannot be read
    (it does not open, it is cut off, it exceeds the reader's limits of 4000 shape leaves or 6000 solids); no valid
    closed solid on any tag; a published file of the maker in the submission (verify.py); a write after the session
    ended (verify.py). Everything else costs only the item it touches. A grading item that fails scores 0 and counts in
    grader_item_errors; a failure outside every item (the standard, its pins, the seen tables, the alignment) is the
    grader's own and fails the run as a task error.
 3. Alignment: the whole unit, rigidly, no scale, no mirror (align.py), from the tagged parts' query samples onto the
    standard's dense samples; the tie between near-equal fits goes to the one that puts the tagged parts nearest their
    places; the labels inside a pair never decide.
 4. Pairs: P02 / P03 and P04 / P05 go to their two places one to one, the assignment with the smaller sum of the
    distances between the tags' and the places' surface centroids; an exact tie takes the higher pair score.
 5. surface: per place, band(p90; surface.p90_mm) of two distance sets pooled: (a) query samples of the outer shells
    of the tag's solids (each solid's own boundary; the solids are not united) to the dense samples of all of the
    place's maker triangles; (b) the place's graded samples (seen tables) to the dense samples of the tag's solids;
    a distance is to the tangent disk (radius disk_mm) of the nearest dense sample. Weighted mean over the twelve tags
    (part_weights); a missing, void or solid-less tag scores 0. Sample counts are area x density per tag, clamped to
    [min, max]. Distances are exact up to the search caps; a p90 that depends on points beyond them is reported as a
    lower bound (p90_exact false) and is then at or beyond the zero line, so its score is exactly 0.
 6. joint_axes: for each of the six hinges, the unsuppressed Revolute joints whose two members carry the hinge's two
    tags (either order); per joint frame (the reader writes one per reference), the frame's z axis through its origin
    after the alignment: score = w_offset band(distance from the standard joint origin to that line) + w_angle
    band(angle between the lines, 0-90 deg); a joint scores the mean of its frames; a hinge takes its best joint (none:
    0). A Revolute joint between other tags is ignored and reported. Mean of the six.
 7. structure: (share of the fixed parts joined to P01, whose member must itself carry a Grounded joint) x (share of
    the twelve tags present on exactly one member with a usable solid); joined = union-find over the members and a
    ground node through Grounded joints and unsuppressed Fixed joints, never through a Revolute joint.
 8. floor = rising(surface; floor.ramp); design_score = w_surface surface + floor (w_joint_axes joint_axes +
    w_structure structure).
"""
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

from . import VERSION
from . import score
from .align import align
from .geom import (area_centroid, brep_valid, mass_props, mesh_solid, pooled_quantile, read_brep, sample_count,
                   sample_triangles, solids_of, valid_after_tolerance_update)
from .reference import Reference
from .submission import Submission
from .util import (TaskError, apply, finite, inv, item_error, limit_cpus, line_angle_deg, orthonormal,
                   point_line_distance, rotate)

CRITERIA = ("surface", "joint_axes", "structure")


def family_index(tag, cfg):
    """The tags in order with the two tags of a pair at one index: seeds and orders that do not depend on which tag
    of a pair is written where."""
    order = []
    pair_of = {}
    for a, b in cfg["pairs"]:
        pair_of[a] = pair_of[b] = a
    for t in sorted(cfg["part_weights"]):
        f = pair_of.get(t, t)
        if f not in order:
            order.append(f)
    return order.index(pair_of.get(tag, tag)) + 1


class TagGeo(object):
    """The submitted solids of one tag: their tessellations (document frame), surface centroid, area, samples."""

    def __init__(self, tag, triangles, cfg):
        S = cfg["sampling"]
        self.tag = tag
        self.triangles = triangles
        self.centroid, self.area = area_centroid(triangles)
        base = int(S["seed"]) + 100 * family_index(tag, cfg)
        nq, self.query_capped = sample_count(self.area, S["query_per_mm2"], S["query_min"], S["query_max"])
        nd, self.dense_capped = sample_count(self.area, S["dense_per_mm2"], S["dense_min"], S["dense_max"])
        self.query = sample_triangles(triangles, nq, base + 1)[0]
        P, N, _i = sample_triangles(triangles, nd, base + 2)
        from scipy.spatial import cKDTree
        self.dense, self.normals = P, N
        self.tree = cKDTree(P) if len(P) else None
        self.tree_coarse = cKDTree(P[::int(cfg["search"]["coarse_stride"])]) if len(P) else None


def _load_solids(member, cfg, notes, stats):
    """The usable solids of a member: read back from BREP, BRepCheck (once more after a tolerance update), volume."""
    G = cfg["gates"]
    out = []
    for s in member.solids:
        try:
            shape = read_brep(s["file"])
            sols = solids_of(shape)
            if len(sols) != 1:
                raise ValueError("%d solids in one solid file" % len(sols))
            sol = sols[0]
            if G.get("recheck_validity", True) and not brep_valid(sol):
                if not valid_after_tolerance_update(sol):
                    raise ValueError("BRepCheck reports an invalid solid")
                stats["solids_valid_after_tolerance_update"] = stats.get("solids_valid_after_tolerance_update", 0) + 1
            V, c, I = mass_props(sol)
            if not V > float(G["min_volume_mm3"]):
                raise ValueError("no volume")
            out.append({"shape": sol, "V": V, "c": c, "I": I})
        except Exception as exc:  # noqa: BLE001 - a solid that does not load is left out
            stats["n_solids_left_out_by_grader"] = stats.get("n_solids_left_out_by_grader", 0) + 1
            if len(notes) < 40:
                notes.append("Local: a solid of %s cannot be used (%s): left out" % (member.tag, str(exc)[:100]))
    return out


def _assign_pairs(geos, ref, A, cfg, pair_score):
    """{place tag: submitted tag} (rule 4)."""
    out = {t: t for t in geos if t not in set(x for p in cfg["pairs"] for x in p)}
    for a, b in cfg["pairs"]:
        mine = [t for t in (a, b) if t in geos]
        ca, cb = ref.places[a].centroid, ref.places[b].centroid

        def dist(t, c):
            return float(np.linalg.norm(apply(A, geos[t].centroid) - c))

        if len(mine) == 2:
            same = dist(a, ca) + dist(b, cb)
            swap = dist(a, cb) + dist(b, ca)
            if same < swap:
                out.update({a: a, b: b})
            elif swap < same:
                out.update({a: b, b: a})
            else:
                s1 = pair_score(a, a) + pair_score(b, b)
                s2 = pair_score(b, a) + pair_score(a, b)
                out.update({a: b, b: a} if s2 > s1 else {a: a, b: b})
        elif len(mine) == 1:
            t = mine[0]
            da, db = dist(t, ca), dist(t, cb)
            if da < db:
                out[a] = t
            elif db < da:
                out[b] = t
            else:
                out[b if pair_score(b, t) > pair_score(a, t) else a] = t
    return out


def _surface_of(place, geo, A, Ainv, cfg):
    """The pooled p90 of a submitted tag at a standard place (rule 5), its one-way p90s and sample counts. Distances are
    exact up to the search caps; where the pooled p90 depends on points beyond them it is reported as a lower bound
    (p90_exact false), which happens only when that bound already reaches the band's zero line (the score is 0)."""
    disk = float(cfg["sampling"]["disk_mm"])
    q = float(cfg["surface"]["quantile"])
    zero = float(cfg["surface"]["p90_mm"][1])
    parts = [(apply(A, geo.query), place.dense, place.normals, place.tree, place.tree_coarse)]
    if len(place.graded) and geo.tree is not None:
        parts.append((apply(Ainv, place.graded), geo.dense, geo.normals, geo.tree, geo.tree_coarse))
    p90, exact, D = pooled_quantile(parts, q, zero, disk, cfg["search"]["caps_mm"])
    one = [float(np.quantile(d, q)) if len(d) and np.all(np.isfinite(d)) else None for d in D]
    return {"p90_mm": p90, "p90_exact": bool(exact), "submitted_to_maker_p90_mm": one[0],
            "maker_to_submitted_p90_mm": one[1] if len(one) > 1 else None,
            "n_query": int(len(D[0])), "n_graded": int(len(D[1])) if len(D) > 1 else 0,
            "n_beyond_search_cap": int(sum(int(np.sum(~np.isfinite(d))) for d in D))}


def _frames(joint, hinge, A):
    """[[offset_mm, angle_deg] per joint frame] of a submitted Revolute joint against a standard hinge (rule 6)."""
    out = []
    for k in ("jcs1", "jcs2"):
        M = joint.get(k)
        if M is None:
            continue
        M = np.asarray(M, float)
        if M.shape != (4, 4) or not np.all(np.isfinite(M)):
            raise ValueError("joint frame %s is not a finite 4 x 4 matrix" % k)
        R = orthonormal(M[:3, :3])
        o = apply(A, M[:3, 3])
        d = rotate(A, R[:, 2])
        out.append([point_line_distance(hinge["point_mm"], o, d), line_angle_deg(d, hinge["dir"])])
    return out or None


def grade(ex_dir, stage_dir, cfg, log=None, ref=None, seen_index=None):
    """The grade of one extraction (module docstring). `ref`: an already loaded Reference (development batches grade
    many extractions against one; verify loads it once per document)."""
    t0 = time.time()
    log = log or (lambda m: print("[grade] %s" % m, file=sys.stderr, flush=True))
    res = {"ok": True, "format": VERSION, "design_score": 0.0, "criteria": {k: 0.0 for k in CRITERIA}, "floor": 0.0,
           "checks": {"cad_valid": 0.0}, "zero_case": None, "tags": {}, "hinges": {}, "structure": {},
           "statistics": {"grader_item_errors": 0}, "diagnostics": [], "timing": {}}
    stats, diags, timing = res["statistics"], res["diagnostics"], res["timing"]
    if ref is None:
        ref = Reference(stage_dir, cfg, seen_index=seen_index, log=log)
    timing["t_reference_s"] = time.time() - t0
    stats["seen_tables_proxy"] = 1 if ref.seen_info.get("proxy") else 0
    sub = Submission(ex_dir, cfg, ref.tags)
    stats.update(sub.stats)
    if not sub.readable:
        res["zero_case"] = "unreadable"
        diags.extend(sub.reasons)
        diags.append("The document cannot be read: it scores 0.")
        timing["t_total_s"] = time.time() - t0
        return res
    diags.extend(sub.notes)
    t1 = time.time()
    usable, counts, notes = {}, {}, []
    for tag in sorted(sub.by_tag):
        m = sub.by_tag[tag]
        usable[tag] = _load_solids(m, cfg, notes, stats)
        counts[m.name] = len(usable[tag])
    diags.extend(notes)
    members = sub.member_table(counts)
    present = score.present_tags(members, cfg)
    stats["n_tags_present"] = len(present)
    stats["n_solids_usable"] = sum(len(usable[t]) for t in present)
    timing["t_solids_s"] = time.time() - t1
    if not present:
        res["zero_case"] = "no_valid_solid"
        diags.append("No tag holds a valid closed solid: the document scores 0.")
        timing["t_total_s"] = time.time() - t0
        return res
    res["checks"]["cad_valid"] = 1.0
    t1 = time.time()
    tmp = tempfile.mkdtemp(prefix="ffcad_grade_")
    geos = {}
    mesh_notes = 0
    try:
        for tag in present:
            tris = []
            for rec in usable[tag]:
                try:
                    got = mesh_solid(rec["shape"], cfg, tmp, rec["V"], rec["c"], rec["I"])
                except Exception as exc:  # noqa: BLE001
                    got = None
                    item_error(stats, diags, "the tessellation of a solid of %s" % tag, exc)
                if got is None:
                    stats["n_solids_not_tessellated"] = stats.get("n_solids_not_tessellated", 0) + 1
                    continue
                V, F, n = got
                mesh_notes += 1 if n else 0
                tris.append(V[F])
                rec.pop("shape", None)
            if tris:
                geos[tag] = np.vstack(tris)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    stats["n_solids_with_mesh_notes"] = mesh_notes
    timing["t_mesh_s"] = time.time() - t1
    t1 = time.time()
    geos = {t: TagGeo(t, T, cfg) for t, T in geos.items()}
    stats["sampling_capped"] = sum(1 for g in geos.values() if g.query_capped or g.dense_capped)
    timing["t_samples_s"] = time.time() - t1
    for tag in present:
        if tag not in geos:
            item_error(stats, diags, "the surface of %s" % tag, ValueError("none of its solids tessellates"))
    if not geos:
        raise TaskError("no solid of any tag could be tessellated")
    # ---- alignment (rule 3)
    t1 = time.time()
    pair_of = {}
    for a, b in cfg["pairs"]:
        pair_of[a] = pair_of[b] = (a, b)
    places = sorted(set(x for t in geos for x in pair_of.get(t, (t,))))
    ref.prepare(places)
    blocks = sorted(((g.query, g.centroid, t) for t, g in geos.items()),
                    key=lambda b: (family_index(b[2], cfg), tuple(np.round(b[1], 6))))
    choices = {t: [ref.places[p].centroid for p in pair_of.get(t, (t,))] for t in geos}
    Al = cfg["alignment"]
    coarse = ref.target(places, int(Al["coarse_stride"]))
    fine = ref.target(places, int(Al["fine_stride"]))
    A = align(blocks, coarse, fine, choices, cfg, stats)
    Ainv = inv(A)
    res["alignment"] = {"T_document_to_standard": [[float(x) for x in row] for row in A]}
    timing["t_alignment_s"] = time.time() - t1
    # ---- surface (rules 4, 5)
    t1 = time.time()
    cache = {}

    def pair_score(place_tag, sub_tag):
        key = (place_tag, sub_tag)
        if key not in cache:
            cache[key] = _surface_of(ref.places[place_tag], geos[sub_tag], A, Ainv, cfg)
        return score.band_value(cache[key]["p90_mm"], cfg)

    assign = _assign_pairs(geos, ref, A, cfg, pair_score)
    tag_surface = {}
    for place_tag in ref.tags:
        sub_tag = assign.get(place_tag)
        rec = {"present": sub_tag is not None, "assigned_tag": sub_tag, "weight": float(cfg["part_weights"][place_tag]),
               "score": 0.0}
        if sub_tag is not None:
            try:
                if (place_tag, sub_tag) not in cache:
                    cache[(place_tag, sub_tag)] = _surface_of(ref.places[place_tag], geos[sub_tag], A, Ainv, cfg)
                m = cache[(place_tag, sub_tag)]
                rec.update(m)
                rec["score"] = score.band_value(m["p90_mm"], cfg)
                rec["n_solids"] = len(usable[sub_tag])
                rec["capped"] = bool(geos[sub_tag].query_capped or geos[sub_tag].dense_capped or
                                     ref.places[place_tag].capped)
                rec["centroid_offset_mm"] = float(np.linalg.norm(apply(A, geos[sub_tag].centroid) -
                                                                 ref.places[place_tag].centroid))
            except Exception as exc:  # noqa: BLE001
                item_error(stats, diags, "the surface of %s" % place_tag, exc)
                rec["score"] = 0.0
                rec["error"] = str(exc)[:200]
            tag_surface[sub_tag] = rec["score"]
        res["tags"][place_tag] = rec
    stats["tags_reassigned"] = sum(1 for p, t in assign.items() if p != t)
    timing["t_surface_s"] = time.time() - t1
    # ---- joints (rule 6)
    t1 = time.time()
    one, _void = score.tag_table(members, cfg)
    label_of = {n: t for t, n in one.items()}
    jt = []
    for j in sub.joints:
        rec = {"name": j.get("name"), "type": j.get("type"), "part1": j.get("part1"), "part2": j.get("part2"),
               "suppressed": bool(j.get("suppressed")), "frames": None}
        if j.get("error"):
            rec["error"] = str(j["error"])[:200]
            rec["type"] = "unreadable"
        a, b = label_of.get(j.get("part1")), label_of.get(j.get("part2"))
        if (str(rec["type"]).lower() == "revolute" and not rec["suppressed"] and a and b and
                frozenset((a, b)) in ref.hinges):
            try:
                rec["frames"] = _frames(j, ref.hinge((a, b)), A)
            except Exception as exc:  # noqa: BLE001
                item_error(stats, diags, "the joint %s" % j.get("name"), exc)
        jt.append(rec)
    stats["n_joints_unreadable"] = sum(1 for r in jt if r["type"] == "unreadable")
    stats["n_revolute_suppressed"] = sum(1 for r in jt if str(r["type"]).lower() == "revolute" and r["suppressed"])
    # ---- the combining layer (rules 6-8)
    out = score.combine(members, jt, sub.grounded, tag_surface, sub.no_assembly, cfg)
    for h in out["hinges"]:
        key = "%s-%s" % tuple(h["pair"])
        fr = h["frames"] or []
        res["hinges"][key] = {"score": h["score"], "candidates": h["candidates"], "joint": h["best"],
                              "offset_mm": [f[0] for f in fr], "angle_deg": [f[1] for f in fr]}
        if h["candidates"] > 1:
            diags.append("Local: %d Revolute joints join %s and %s: the best one (%s) counts" % (
                h["candidates"], h["pair"][0], h["pair"][1], h["best"]))
    if out["ignored_revolute"]:
        diags.append("Local: %d Revolute joint(s) not at a hinge of the original unit ignored: %s" % (
            len(out["ignored_revolute"]), ", ".join(str(x["name"]) for x in out["ignored_revolute"][:6])))
    stats["n_revolute_ignored"] = len(out["ignored_revolute"])
    res["structure"] = out["structure_detail"]
    for k in CRITERIA:
        res["criteria"][k] = finite(out[k])
    res["floor"] = finite(out["floor"])
    res["design_score"] = finite(out["design_score"])
    timing["t_joints_s"] = time.time() - t1
    timing["t_total_s"] = time.time() - t0
    log("design_score %.6f (surface %.4f, joint_axes %.4f, structure %.4f, floor %.4f); alignment residual %.4f mm" % (
        res["design_score"], res["criteria"]["surface"], res["criteria"]["joint_axes"], res["criteria"]["structure"],
        res["floor"], stats.get("alignment_residual_mm", float("nan"))))
    return res


def jsonable(x):
    """Plain JSON values: numpy scalars and arrays to Python, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def main(argv):
    import argparse
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="grade one extraction of the submitted document")
    ap.add_argument("extraction")
    ap.add_argument("output")
    ap.add_argument("--config", default=os.path.join(here, "grader_config.json"))
    ap.add_argument("--stage", default=here)
    a = ap.parse_args(argv[1:])
    with open(a.config) as fh:
        cfg = json.load(fh)
    limit_cpus(int(cfg.get("grader_cpus", 1)))
    try:
        res = grade(a.extraction, a.stage, cfg)
    except TaskError as exc:  # the standard, its pins, the seen tables, an extraction the reader did not write
        res = {"ok": False, "task_error": str(exc), "error": traceback.format_exc()[-3000:]}
    except Exception:  # noqa: BLE001 - outside every grading item: a grader fault, never a silent 0 (task error)
        res = {"ok": False, "task_error": "the grader failed outside a grading item", "error": traceback.format_exc()[-3000:]}
    with open(a.output, "w") as fh:
        json.dump(jsonable(res), fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")
    if res.get("ok"):
        print(json.dumps({k: res[k] for k in ("design_score", "criteria", "checks", "zero_case")}, sort_keys=True),
              file=sys.stderr)
    return 0
