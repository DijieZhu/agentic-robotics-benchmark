"""Criterion P, placement, mounting and joints (weight 0.15):

  P = placement.w_made 0.5 * made + placement.w_purchased 0.3 * purchased + placement.w_joints 0.2 * joints

made       each of the 13 reference instances: the submitted part assigned to it by the whole-strut alignment, its pose
           in the strut frame A inv(T_fit) against the instance pose W: band(position; made_position_mm 0.5 / 4.0) x
           (1 - w_orientation + w_orientation band(orientation; made_orientation_deg 1.5 / 12)), w_orientation 0.3.
           Position error = displacement of the design's centroid; orientation error = the smallest turn between the
           two poses modulo the design's symmetry group (families.json: end cap C4 about its axis; end cap cover and
           winch continuous about their axes, the winch also end for end, so only the angle between the axes counts;
           the rest none). A missing instance scores 0; the 13 are averaged.
purchased  each reference placement (purchased.json, weight_in_position_score: each gearmotor 2, the rod 2; the
           boards join with 1 each when they are in the payload), the recognised instance assigned to it by the
           alignment, at A W:
             gearmotor: completeness x band(centroid position; purchased_position_mm 0.4 / 3.0) x
                        (1 - w_orientation + w_orientation band(orientation; purchased_orientation_deg 1.5 / 12))
             rod:       band(lateral offset of its axis from the reference midpoint; rod_lateral_mm 0.2 / 1.5) x
                        band(axial position of its midpoint; rod_axial_mm 0.5 / 4.0) x
                        band(angle between the axes, end for end free; rod_angle_deg 0.3 / 3)
           weighted mean; a missing placement scores 0 with its weight.
joints     joints.w_fixed 0.6 * connected + joints.w_revolute 0.4 * revolute:
             connected: the share of pieces reached from a Grounded component through unsuppressed Fixed joints, made
                        parts and purchased placements half each (a winch reached through its unsuppressed Revolute
                        counts as reached: that joint is its mount); a document without an Assembly has none
             revolute:  per winch (P08, P09), the best unsuppressed Revolute joining the winch's component to the
                        component of its end's gearmotor or to one rigidly joined to it (Fixed): band(distance of the
                        joint's axis from the motor's output axis, at the joint's origin; revolute_offset_mm 0.2 / 1.0)
                        x band(angle between them, sense free; revolute_angle_deg 1 / 8), the worse of the joint's two
                        frames (jcs1, jcs2); the two winches averaged. A Fixed winch loses only this share.
The Grounded part is only the root of the joint graph; it takes no part in the alignment.
"""
import math

import numpy as np

from .geom import apply, band, inv, rot_angle_deg, axis_angle_R

NAME = "placement"


def _axis_angle(a, b, flip):
    c = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    ang = math.degrees(math.acos(c))
    return min(ang, 180.0 - ang) if flip else ang


def orientation_error(Rs, Rw, sym):
    """Smallest turn (deg) between two design -> strut rotations modulo the design's symmetry group (parse_symmetry)."""
    if sym["kind"] == "continuous":
        a = sym["axis_dir"]
        return _axis_angle(Rs @ a, Rw @ a, sym.get("flip", False))
    rel = Rw.T @ Rs
    if sym["kind"] == "cyclic":
        n = int(sym["order"])
        return min(rot_angle_deg(rel @ axis_angle_R(sym["axis_dir"], 2.0 * math.pi * k / n)) for k in range(n))
    return rot_angle_deg(rel)


def made_scores(ctx, P):
    ref = ctx.ref
    al = ctx.alignment
    wo = float(P["w_orientation"])
    out, scores = {}, []
    for tag in ref.tags:
        sid = al["assignment"].get(tag)
        if sid is None:
            out[tag] = {"present": False, "score": 0.0}
            scores.append(0.0)
            continue
        d = ref.designs[ref.family_of[tag]]
        X = ctx.part_pose(sid)
        W = ref.instances[tag]["W"]
        e_p = float(np.linalg.norm(apply(X, d["com"]) - apply(W, d["com"])))
        e_r = orientation_error(X[:3, :3], W[:3, :3], d["symmetry"])
        sc = band(e_p, P["made_position_mm"]) * (1.0 - wo + wo * band(e_r, P["made_orientation_deg"]))
        out[tag] = {"present": True, "position_error_mm": e_p, "orientation_error_deg": e_r, "score": sc}
        scores.append(sc)
    return float(np.mean(scores)) if scores else 0.0, out


def _weight(ctx, P, typ):
    t = ctx.ref.types[typ]["spec"]
    return float(P["type_weights"].get(typ, t.get("weight_in_position_score", P["type_default_weight"])))


def purchased_scores(ctx, P):
    ref = ctx.ref
    al = ctx.alignment
    wo = float(P["w_orientation"])
    out = {}
    wsum = wacc = 0.0
    for pid in sorted(ref.placements):
        pl = ref.placements[pid]
        typ = ref.types[pl["type"]]
        w = _weight(ctx, P, pl["type"])
        wsum += w
        k = al["purchased_assignment"].get(pid)
        if k is None:
            out[pid] = {"type": pl["type"], "present": False, "weight": w, "score": 0.0}
            continue
        it = ctx.instances[k]
        Xa, Wr = ctx.instance_pose(k), pl["W"]
        sym = typ["symmetry"]
        rec = {"type": pl["type"], "present": True, "weight": w, "completeness": float(it["completeness"])}
        if sym["continuous"]:
            a, p = sym["axis_dir"] / np.linalg.norm(sym["axis_dir"]), sym["axis_point"]
            ps, ds = apply(Xa, p), Xa[:3, :3] @ a
            pr, dr = apply(Wr, p), Wr[:3, :3] @ a
            v = ps - pr
            lat = float(np.linalg.norm((pr - ps) - ((pr - ps) @ ds) * ds))
            ax = abs(float(v @ dr))
            ang = _axis_angle(ds, dr, sym["flip"])
            sc = it["completeness"] * band(lat, P["rod_lateral_mm"]) * band(ax, P["rod_axial_mm"]) * \
                band(ang, P["rod_angle_deg"])
            rec.update(lateral_mm=lat, axial_mm=ax, angle_deg=ang)
        else:
            c = typ["centroid"]
            e_p = float(np.linalg.norm(apply(Xa, c) - apply(Wr, c)))
            e_r = rot_angle_deg(Wr[:3, :3].T @ Xa[:3, :3])
            sc = it["completeness"] * band(e_p, P["purchased_position_mm"]) * \
                (1.0 - wo + wo * band(e_r, P["purchased_orientation_deg"]))
            rec.update(position_error_mm=e_p, orientation_error_deg=e_r)
        rec["score"] = float(sc)
        out[pid] = rec
        wacc += w * sc
    return (wacc / wsum if wsum > 0 else 0.0), out


def _graph(ctx, edges_of):
    g = {}
    for j in ctx.sub.joints:
        e = edges_of(j)
        if e is None:
            continue
        a, b = e
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set()).add(a)
    return g


def _reach(g, roots):
    seen = set(r for r in roots if r)
    todo = list(seen)
    while todo:
        for nxt in g.get(todo.pop(), set()) - seen:
            seen.add(nxt)
            todo.append(nxt)
    return seen


def _usable(j, kind):
    return (not j.get("suppressed")) and (not j.get("error")) and str(j.get("type", "")).lower() == kind and \
        j.get("part1") and j.get("part2") and j.get("part1") != j.get("part2")


def _joints(ctx, P):
    J = P["joints"]
    ref = ctx.ref
    al = ctx.alignment
    rec = {"no_assembly": bool(ctx.sub.no_assembly)}
    if ctx.sub.no_assembly:
        return 0.0, dict(rec, connected=0.0, revolute=0.0)
    comp = {s["sid"]: s["component"] for s in ctx.solids}
    winch_tags = [t for t in ref.tags if ref.family_of[t] == J["winch_family"]]
    winch_comps = set(comp[al["assignment"][t]] for t in winch_tags if al["assignment"].get(t) is not None)
    fixed = _graph(ctx, lambda j: (j["part1"], j["part2"]) if _usable(j, "fixed") else None)
    reached = _reach(fixed, ctx.sub.grounded)
    for j in ctx.sub.joints:
        if _usable(j, "revolute"):
            a, b = j["part1"], j["part2"]
            if a in winch_comps and b in reached:
                reached.add(a)
            elif b in winch_comps and a in reached:
                reached.add(b)
    made = [1.0 if (al["assignment"].get(t) is not None and comp[al["assignment"][t]] in reached) else 0.0
            for t in ref.tags]
    wsum = wcon = 0.0
    for pid in sorted(ref.placements):
        w = _weight(ctx, P, ref.placements[pid]["type"])
        wsum += w
        k = al["purchased_assignment"].get(pid)
        if k is None:
            continue
        sids = list(ctx.instances[k]["slots"].values())
        vol = sum(ctx.solid(s)["V"] for s in sids)
        con = sum(ctx.solid(s)["V"] for s in sids if comp[s] in reached)
        wcon += w * (con / vol if vol > 0 else 0.0)
    connected = 0.5 * float(np.mean(made)) + 0.5 * (wcon / wsum if wsum > 0 else 0.0)
    # Revolute joints of the winches on their motors' output axes
    rev = []
    pa = al["purchased_assignment"]
    out_w = {}
    for tag in winch_tags:
        sid = al["assignment"].get(tag)
        motor_pid = J["winch_motor"][tag]
        k = pa.get(motor_pid)
        if sid is None or k is None:
            rev.append(0.0)
            out_w[tag] = {"score": 0.0, "why": "winch or motor not found"}
            continue
        cw = comp[sid]
        it = ctx.instances[k]
        cm = comp[it["slots"][ref.types[it["type"]]["anchor"]]]
        body = _reach(fixed, [cm])
        oa = ref.types[it["type"]]["spec"].get("output_axis_local") or {"point_mm": [0, 0, 0], "dir": [0, 0, 1]}
        p_m = apply(it["W"], np.asarray(oa["point_mm"], float))
        d_m = it["W"][:3, :3] @ np.asarray(oa["dir"], float)
        d_m = d_m / np.linalg.norm(d_m)
        best = 0.0
        best_rec = {"score": 0.0, "why": "no Revolute joins the winch to its motor"}
        for j in ctx.sub.joints:
            if not _usable(j, "revolute"):
                continue
            ends = {j["part1"], j["part2"]}
            if cw not in ends or not (ends - {cw}) & body:
                continue
            ss, wd, wa = [], 0.0, 0.0
            for key in ("jcs1", "jcs2"):
                if j.get(key) is None:
                    ss.append(0.0)
                    continue
                Jm = np.asarray(j[key], float)
                o, z = Jm[:3, 3], Jm[:3, 2]
                v = o - p_m
                dist = float(np.linalg.norm(v - (v @ d_m) * d_m))
                ang = _axis_angle(z, d_m, True)
                ss.append(band(dist, P["revolute_offset_mm"]) * band(ang, P["revolute_angle_deg"]))
                wd, wa = max(wd, dist), max(wa, ang)
            worst = min(ss)
            if worst > best or best_rec.get("joint") is None:
                best = max(best, worst)
                best_rec = {"score": best, "joint": j.get("name"), "offset_mm": wd, "angle_deg": wa}
        rev.append(best)
        out_w[tag] = best_rec
    revolute = float(np.mean(rev)) if rev else 0.0
    sc = float(J["w_fixed"]) * connected + float(J["w_revolute"]) * revolute
    rec.update(connected=connected, connected_made=float(np.mean(made)),
               connected_purchased=(wcon / wsum if wsum > 0 else 0.0), revolute=revolute, winches=out_w)
    return sc, rec


def item(ctx):
    """Criterion P (module docstring)."""
    P = ctx.cfg["placement"]
    made, made_rec = made_scores(ctx, P)
    pur, pur_rec = purchased_scores(ctx, P)
    jt, jrec = _joints(ctx, P)
    score = float(P["w_made"]) * made + float(P["w_purchased"]) * pur + float(P["w_joints"]) * jt
    ctx.stats.update(placement_made=made, placement_purchased=pur, placement_joints=jt,
                     joints_connected=float(jrec.get("connected", 0.0)), joints_revolute=float(jrec.get("revolute", 0.0)))
    per_part = dict(made_rec)
    per_part.update(pur_rec)
    per_part["_joints"] = jrec
    diags = []
    bad = sorted([(r["score"], t) for t, r in made_rec.items() if r["score"] < 1.0])[:3]
    for s, t in bad:
        r = made_rec[t]
        diags.append("Placement %s: %s" % (t, "missing" if not r["present"] else "%.2f mm, %.1f deg from its place (%.2f)" % (
            r["position_error_mm"], r["orientation_error_deg"], s)))
    for pid, r in sorted(pur_rec.items()):
        if r["score"] < 1.0:
            diags.append("Placement %s (%s): %s" % (pid, r["type"], "not found" if not r["present"] else "score %.2f" % r["score"]))
    if jrec.get("no_assembly"):
        diags.append("Joints: no Assembly, no joint credit")
    else:
        for t, r in sorted(jrec.get("winches", {}).items()):
            if r["score"] < 1.0:
                diags.append("Joint of %s: %s" % (t, r.get("why") or "Revolute %.2f mm, %.1f deg off the motor shaft" % (
                    r.get("offset_mm", 0.0), r.get("angle_deg", 0.0))))
    return {"score": score, "per_part": per_part, "diagnostics": diags[:6]}
