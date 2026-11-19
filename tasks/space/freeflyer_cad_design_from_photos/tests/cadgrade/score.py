"""The combining layer: from the member, joint and surface tables to surface, joint_axes, structure, the floor and the
design score. Pure functions on plain tables, so the worked examples of the rules can be fed to them directly.

Tables (the same shapes for a real grade and for a worked example):
    members   [{"name": member (component) name, "label": its own Label as written, "valid_solids": n}]
    joints    [{"name", "type" ("Fixed", "Revolute", ...), "part1", "part2" (member names or None), "suppressed",
                "frames": [[offset_mm, angle_deg], [offset_mm, angle_deg]] (a Revolute joint between the two tags of a
                standard hinge: one entry per joint frame; None when the frames could not be read)}]
    grounded  [member names of the Grounded joints]
    tag_surface {tag: surface score} of the tags that hold a valid closed solid (pairs already matched to places)
"""
from .util import band, rising

GROUND = "<ground>"


def tag_table(members, cfg):
    """({tag: member name} of the tags carried by exactly one member, sorted void tags (on more than one member)).
    A tag is the member's own Label with the ends trimmed, compared exactly."""
    tags = set(cfg["part_weights"])
    seen = {}
    for m in members:
        lab = str(m.get("label") or "").strip()
        if lab in tags:
            seen.setdefault(lab, []).append(m["name"])
    one = {t: v[0] for t, v in seen.items() if len(v) == 1}
    void = sorted(t for t, v in seen.items() if len(v) > 1)
    return one, void


def present_tags(members, cfg):
    """Tags on exactly one member that holds at least one valid closed solid."""
    one, _void = tag_table(members, cfg)
    by_name = {m["name"]: m for m in members}
    return sorted(t for t, n in one.items() if int(by_name[n].get("valid_solids", 0)) >= 1)


def band_value(p90_mm, cfg):
    """A tag's surface score: band(pooled p90; surface.p90_mm)."""
    return band(p90_mm, cfg["surface"]["p90_mm"])


def surface_value(tag_surface, present, cfg):
    """Weighted mean over the twelve tags; a tag that is not present scores 0."""
    W = cfg["part_weights"]
    tot = sum(float(w) for w in W.values())
    return sum(float(W[t]) * float(tag_surface.get(t, 0.0)) for t in W if t in present) / tot


def frame_score(offset_mm, angle_deg, cfg):
    J = cfg["joint_axes"]
    return float(J["w_offset"]) * band(offset_mm, J["offset_mm"]) + float(J["w_angle"]) * band(angle_deg, J["angle_deg"])


def joint_score(frames, cfg):
    """The mean of the scores of a joint's frames (0 when it has none)."""
    if not frames:
        return 0.0
    return sum(frame_score(o, a, cfg) for o, a in frames) / float(len(frames))


def joint_axes(members, joints, cfg):
    """(joint_axes, per hinge {"pair", "score", "candidates", "best", "frames"}, ignored Revolute joints).
    For each of the six standard hinges: the best-scoring unsuppressed Revolute joint whose two members carry the
    hinge's two tags (either order); none -> 0. A Revolute joint between other tags is ignored and listed."""
    one, _void = tag_table(members, cfg)
    label_of = {n: t for t, n in one.items()}
    pairs = [tuple(p) for p in cfg["revolute_pairs"]]
    keys = {frozenset(p): p for p in pairs}
    per = {p: {"pair": list(p), "score": 0.0, "candidates": 0, "best": None, "frames": None} for p in pairs}
    ignored = []
    for j in joints:
        if str(j.get("type", "")).lower() != "revolute" or j.get("suppressed"):
            continue
        a, b = label_of.get(j.get("part1")), label_of.get(j.get("part2"))
        key = frozenset((a, b))
        if a is None or b is None or a == b or key not in keys:
            ignored.append({"name": j.get("name"), "tags": [a, b]})
            continue
        rec = per[keys[key]]
        rec["candidates"] += 1
        s = joint_score(j.get("frames"), cfg)
        if rec["best"] is None or s > rec["score"]:
            rec.update(score=s, best=j.get("name"), frames=j.get("frames"))
    value = sum(per[p]["score"] for p in pairs) / float(len(pairs))
    return value, [per[p] for p in pairs], ignored


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def connected_fixed_parts(members, joints, grounded, cfg):
    """(the fixed parts joined to the grounded tag, whether its member carries a Grounded joint). Union-find over the
    members and one ground node: every Grounded joint joins its member to the ground, every unsuppressed Fixed joint
    joins its two members; Revolute joints are never edges."""
    one, _void = tag_table(members, cfg)
    parent = {m["name"]: m["name"] for m in members}
    parent[GROUND] = GROUND

    def union(a, b):
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb:
            parent[rb] = ra

    for g in grounded:
        if g in parent:
            union(GROUND, g)
    for j in joints:
        if str(j.get("type", "")).lower() != "fixed" or j.get("suppressed") or j.get("error"):
            continue
        a, b = j.get("part1"), j.get("part2")
        if a in parent and b in parent and a != b:
            union(a, b)
    base = one.get(cfg["grounded"])
    base_grounded = base is not None and base in grounded
    if not base_grounded:
        return [], False
    root = _find(parent, base)
    return sorted(t for t in cfg["fixed_parts"] if t in one and _find(parent, one[t]) == root), True


def structure(members, joints, grounded, cfg):
    connected, base_grounded = connected_fixed_parts(members, joints, grounded, cfg)
    present = present_tags(members, cfg)
    first = len(connected) / float(len(cfg["fixed_parts"]))
    second = len(present) / float(len(cfg["part_weights"]))
    return first * second, {"connected": connected, "base_grounded": base_grounded, "present": present,
                            "connected_share": first, "present_share": second}


def design_score(q1, q2, q3, cfg):
    w = cfg["weights"]
    f = rising(q1, cfg["floor"]["ramp"])
    return float(w["surface"]) * q1 + f * (float(w["joint_axes"]) * q2 + float(w["structure"]) * q3), f


def combine(members, joints, grounded, tag_surface, no_assembly, cfg):
    """Every quantity of the design score from the tables."""
    one, void = tag_table(members, cfg)
    present = present_tags(members, cfg)
    q1 = surface_value(tag_surface, present, cfg)
    if no_assembly:
        q2, per, ignored = 0.0, [{"pair": list(p), "score": 0.0, "candidates": 0, "best": None, "frames": None}
                                 for p in cfg["revolute_pairs"]], []
        q3, st = 0.0, {"connected": [], "base_grounded": False, "present": present, "connected_share": 0.0,
                       "present_share": len(present) / float(len(cfg["part_weights"]))}
    else:
        q2, per, ignored = joint_axes(members, joints, cfg)
        q3, st = structure(members, joints, grounded, cfg)
    total, floor = design_score(q1, q2, q3, cfg)
    return {"surface": q1, "joint_axes": q2, "structure": q3, "floor": floor, "design_score": total,
            "hinges": per, "ignored_revolute": ignored, "structure_detail": st, "void_tags": void,
            "present": present, "tag_members": one}
