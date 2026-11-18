"""C1 net_and_geometry_replication: cable_net.json against the ledger's C1 rows. No value of any truth is written here.

    value, det = score(docs, gt, cfg)      the interface verify.py calls (gt: its mapping of gt/*.json; cfg: the
                                           whole grader_config.json); det = {"diagnostic", "stats", "parts", "detail"}
    out = grade(docs, truth, settings)     the same on load.truth_from / load.settings output

Three groups, weighted by settings c1.groups (a group without any graded row drops out):
  topology    blocks of records (settings c1.topology_blocks). A ledger record names, per block, the fields that
              block scores. A record is found by its index (motor_index, sensor_index, controller_index; a spare
              input by its strut) and earns a block when every field of that block is right and, for a block listed
              in c1.block_requires, every field of the required blocks is right as well (the end nodes count only
              with the right lead). A block scores the mean over its records; a missing record earns nothing.
              Fields: winch and hook are nodes (strut and end); between is the unordered pair of two nodes;
              connector_pad is a designator and a pad, the pad compared as a number when it is one; motor_connector
              and strut are compared as written.
  dimensions  every graded length: the band of its row on the error in the row's unit (geometry is in metres);
              weighted by tier; a missing or non-numeric value earns nothing.
  counts      every graded count, exact, weighted by tier.
"""
from . import bands as B
from . import load as L

FIELD_KIND = {"winch": "node", "hook": "node", "between": "node_pair", "connector_pad": "pad",
              "motor_connector": "text", "strut": "text"}
GEOMETRY_UNIT = "m"


def _node(v):
    if isinstance(v, dict) and isinstance(v.get("strut"), str) and isinstance(v.get("end"), str):
        return v["strut"], v["end"]
    return None


def _node_pair(v):
    if not isinstance(v, (list, tuple)):
        return None
    nodes = [_node(x) for x in v]
    if not nodes or any(n is None for n in nodes):
        return None
    return tuple(sorted(nodes))


def _pad(v):
    if not isinstance(v, str) or "." not in v:
        return None
    designator, pad = v.rsplit(".", 1)
    return designator, (int(pad) if pad.isdigit() else pad)


def field_right(field, agent, truth):
    kind = FIELD_KIND.get(field)
    if kind == "node":
        return _node(agent) is not None and _node(agent) == _node(truth)
    if kind == "node_pair":
        return _node_pair(agent) is not None and _node_pair(agent) == _node_pair(truth)
    if kind == "pad":
        return _pad(agent) is not None and _pad(agent) == _pad(truth)
    if kind == "text":
        return isinstance(agent, str) and agent == truth
    raise ValueError("C1: no comparison for field %r (add it to FIELD_KIND)" % field)


def _agent_records(net):
    """{(section, index): record} of the submitted cable net (the first record of an index; G2 makes it the only)."""
    out = {}
    for section, field in (("cables", "motor_index"), ("sensors", "sensor_index"),
                           ("controllers", "controller_index"), ("spare_inputs", "strut")):
        recs = net.get(section)
        for r in recs if isinstance(recs, list) else []:
            if isinstance(r, dict) and field in r and not isinstance(r[field], bool):
                key = (section, r[field])
                if isinstance(r[field], (int, str)) and key not in out:
                    out[key] = r
    return out


def topology(net, records, cfg):
    weights, requires = cfg["topology_blocks"], cfg["block_requires"]
    agent = _agent_records(net)
    per_block = {}
    detail = []
    for rec in records:
        a = agent.get((rec["section"], rec["index"]))
        right = {f: (a is not None and f in a and field_right(f, a[f], t)) for f, t in sorted(rec["fields"].items())}
        earned = {}
        for block, fields in sorted(rec["blocks"].items()):
            if block not in weights:
                raise ValueError("C1: record %s scores block %r, which grader_config.json does not weight"
                                 % (rec["id"], block))
            ok = a is not None and all(right[f] for f in fields)
            for req in requires.get(block, []):
                ok = ok and all(right[f] for f in rec["blocks"].get(req, []))
            earned[block] = 1.0 if ok else 0.0
            per_block.setdefault(block, []).append(earned[block])
        detail.append({"id": rec["id"], "record": "%s %s=%s" % (rec["section"], rec["index_field"], rec["index"]),
                       "present": a is not None, "fields_right": right, "blocks": earned})
    means = {b: sum(v) / len(v) for b, v in sorted(per_block.items())}
    value = B.weighted_mean((float(weights[b]), means[b]) for b in sorted(means))
    return value, means, {b: (int(sum(v)), len(v)) for b, v in sorted(per_block.items())}, detail


def dimensions(net, dims, tier_weights, units, geometry_unit=GEOMETRY_UNIT):
    geom = net.get("geometry") if isinstance(net.get("geometry"), dict) else {}
    pairs, detail = [], []
    for d in dims:
        raw = geom.get(d["key"])
        val = B.convert(raw, geometry_unit, d["unit"], units)
        s = 0.0 if val is None else B.score(val, d["value"], d["band"])
        err = None if val is None else B.error(val, d["value"], d["band"])
        pairs.append((float(tier_weights[d["tier"]]), s))
        detail.append({"id": d["id"], "key": d["key"], "tier": d["tier"], "present": raw is not None,
                       "error": err, "score": s})
    return B.weighted_mean(pairs), detail


def counts(net, cnts, tier_weights):
    given = net.get("counts") if isinstance(net.get("counts"), dict) else {}
    pairs, detail = [], []
    for c in cnts:
        raw = given.get(c["key"])
        s = B.score(raw, c["value"], c["band"]) if B.is_number(raw) else 0.0
        pairs.append((float(tier_weights[c["tier"]]), s))
        detail.append({"id": c["id"], "key": c["key"], "present": raw is not None, "score": s})
    return B.weighted_mean(pairs), detail


def grade(docs, truth, settings):
    net = docs.get("cable_net") if isinstance(docs.get("cable_net"), dict) else {}
    t = truth["c1"]
    cfg = settings["c1"]
    tw = cfg["tier_weights"]
    groups = {}
    topo, means, tallies, rec_detail = topology(net, t["records"], cfg)
    if t["records"]:
        groups["topology"] = topo
    dim, dim_detail = dimensions(net, t["dimensions"], tw, truth["units"], cfg.get("geometry_unit", GEOMETRY_UNIT))
    if t["dimensions"]:
        groups["dimensions"] = dim
    cnt, cnt_detail = counts(net, t["counts"], tw)
    if t["counts"]:
        groups["counts"] = cnt
    value = B.weighted_mean((float(cfg["groups"][g]), groups[g]) for g in sorted(groups))
    parts = dict(groups)
    parts.update({"topology_" + b: m for b, m in means.items()})
    stats = {"c1_" + k: float(v) for k, v in sorted(parts.items())}
    stats["c1_dimensions_full"] = float(sum(1 for d in dim_detail if d["score"] >= 1.0))
    stats["c1_dimensions_graded"] = float(len(dim_detail))
    diag = "topology %.4f (%s); dimensions %.4f (%d of %d at full credit); counts %.4f (%d of %d)" % (
        topo, ", ".join("%s %d/%d" % (b, n, d) for b, (n, d) in tallies.items()), dim,
        sum(1 for d in dim_detail if d["score"] >= 1.0), len(dim_detail),
        cnt, sum(1 for c in cnt_detail if c["score"] >= 1.0), len(cnt_detail))
    return {"score": value, "parts": parts, "stats": stats,
            "detail": {"records": rec_detail, "dimensions": dim_detail, "counts": cnt_detail}, "diagnostic": diag}


def score(docs, gt, cfg):
    out = grade(docs, L.truth_from(gt, cfg), L.settings(cfg))
    return out["score"], {k: out[k] for k in ("diagnostic", "stats", "parts", "detail")}
