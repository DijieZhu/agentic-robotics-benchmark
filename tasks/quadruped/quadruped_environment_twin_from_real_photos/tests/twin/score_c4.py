"""C4 static physics, and the physics of the graded twin that C5 reuses.

    phys = physics(ctx)            # the standard settle of the graded twin (cached as ctx.cache("settle"))
    r = score(ctx)                 # {"value", "detail"}: C4 = mean over zones of
                                   #   w_static x static_share + w_support x support_F1

The settle. twin_sim's standard simulation on the gate's physics model (excluded units deleted), the gates held at
their photographed position, the agent units paired with HELD true units held kinematic like their true pair: first
the quasi-static relaxation (twin_sim settle.relax_s), then the dynamic settle (settle.seconds); every displacement is
measured against the SUBMITTED pose, so what the relaxation moves counts as drift.

static_share (per zone). Over the true units of the zone that count (not 'ignore'), rest (not held) and stayed
static in every physics variant of the truth (support_truth.json static_units): sum of q_i x [the paired agent unit
is static], with static = the twin_sim rule (the maxima over the relaxation and the settle of the displacement, in
site metres (x s*), within static.pos_mm, and of the rotation within static.rot_deg). Denominator max(number of those
true units, submitted units the zone counts minus those paired with a held or a variant-unstable true unit).

support_F1 (per zone). The required support edges of the truth against the twin's support edges mapped into true
names, each reproduced edge weighted min(q_a, q_b) (an environment part has q = 1):
F1 = 2 x sum of the weights / (n_required + n_twin_edges); a zone with neither is left out.
- A twin edge comes from twin_sim's support pairs (contacts whose normal into the supported unit points up, mean
  up-force over the last support.avg_last_s of the settle): the supporter and the supported unit are mapped into true
  names and the forces of the pairs mapped onto the same (supporter, supported) are summed; the edge exists when the
  sum reaches support.force_frac of the supported unit's weight (several floor tiles under one unit are one floor).
- Names: a unit through its pair ("agent:<body>" when unpaired); an environment supporter is classified by WHERE the
  unit rests on it: the force-weighted mean contact point, mapped into the site frame, against the true twin's
  environment parts (support_truth.json env_parts: every true static collision geom by class): the class of the
  highest true part top under the point within C4.env_snap_m of its height ("env:floor", "env:dock", "env:kerb",
  "env:wall", ...), "env:none" when no true part top is that close; a gate supporter is "gate:other".
- Left out on both sides: edges that involve a held, an 'ignore' or (for the supported side) a variant-unstable true
  unit, edges the truth lists as ambiguous (present in some variants only, or on an environment class that is not a
  scored support class), and edges whose supported agent unit is not counted (outside the region, or matched to an
  'ignore' entry).
- The zone of an edge is the zone of its supported unit (a true unit's zone, an unpaired agent unit's allocated zone).

C4_zone = w_static x static_share + w_support x support_F1 over the terms the zone has (renormalised); C4 = the mean
over the zones that have one. The truth's builder maps the true
twin's variants with the same functions (edges_from_pairs, EnvParts), so the true twin reproduces its own truth.
"""
import math
import weakref

import numpy as np

from .truth import TruthError
from .zones import mean_over_zones

FORMAT_SUPPORT = "site-support-truth/1"
GATE_NODE = "gate:other"
NONE_NODE = "env:none"

_CACHE = weakref.WeakKeyDictionary()


# ---------------------------------------------------------------------------------------------------------------
# environment parts of the true twin: what a unit rests on, by position and height

def geom_class(name):
    """The environment class of a true static geom from its name `<class>_<anything>` (INTERFACES 2.1)."""
    nm = str(name or "")
    head = nm.split("_", 1)[0] if "_" in nm else nm
    return head if head in ("floor", "wall", "pillar", "dock", "kerb", "cage", "rail", "door", "rack", "fixture",
                            "other") else "other"


def env_parts_of_twin(twin):
    """{"env:<class>": [geom names]} over every static collision geom of a (true) twin, names sorted."""
    import mujoco
    m = twin.model
    out = {}
    for g in twin.static_geoms():
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
        out.setdefault("env:" + geom_class(nm), []).append(nm)
    return {k: sorted(v) for k, v in sorted(out.items())}


class EnvParts:
    """The true twin's environment parts as column shapes (site frame = the true twin's world, qpos0): convex pieces
    and up-facing planes, each with its class. classify(points, tol) -> one name per point."""

    def __init__(self, twin, env_parts):
        import mujoco
        m, d = twin.model, twin.data
        self.classes = sorted(env_parts)
        by_name = {}
        for g in twin.static_geoms():
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
            by_name.setdefault(nm, []).append(int(g))
        self.pieces, self.planes = [], []
        seen = set()
        for cls in self.classes:
            for nm in env_parts[cls]:
                gs = by_name.get(nm)
                if not gs:
                    raise TruthError("support_truth.json env_parts names %r, which is no static collision geom of the "
                                     "true twin" % nm)
                for g in gs:
                    seen.add(g)
                    t = int(m.geom_type[g])
                    if t == int(mujoco.mjtGeom.mjGEOM_PLANE):
                        n = np.asarray(d.geom_xmat[g], float).reshape(3, 3)[:, 2]
                        if n[2] > 0.1:
                            self.planes.append((cls, np.asarray(d.geom_xpos[g], float).copy(), n.copy()))
                    else:
                        pc = twin.pieces([g])
                        if pc:
                            self.pieces.append((cls, pc[0]))
        missing = sorted(set(twin.static_geoms()) - seen)
        if missing:
            raise TruthError("support_truth.json env_parts leaves %d static collision geoms of the true twin out "
                             "(first: %s)" % (len(missing), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, missing[0])))

    def classify(self, X, tol):
        """Per point: "env:<class>" of the highest true part top within tol of the point's height whose column does
        not reach above the point + tol, else NONE_NODE. The column test is per shape: a wall that reaches past the
        point is no support there, while the floor under it still is."""
        X = np.atleast_2d(np.asarray(X, float))
        best = np.full(len(X), -np.inf)
        name = [NONE_NODE] * len(X)
        for cls, p in self.pieces:
            a0, a1 = p.z_interval(X[:, 0], X[:, 1])
            ok = np.isfinite(a0) & (a1 <= X[:, 2] + tol) & (a1 >= X[:, 2] - tol) & (a1 > best)
            for k in np.nonzero(ok)[0]:
                best[k], name[k] = a1[k], cls
        for cls, p0, n in self.planes:
            zp = p0[2] - (n[0] * (X[:, 0] - p0[0]) + n[1] * (X[:, 1] - p0[1])) / n[2]
            ok = (zp <= X[:, 2] + tol) & (zp >= X[:, 2] - tol) & (zp > best)
            for k in np.nonzero(ok)[0]:
                best[k], name[k] = zp[k], cls
        return name


# ---------------------------------------------------------------------------------------------------------------
# the support truth

class SupportTruth:
    def __init__(self, doc, truth, cfg):
        if doc.get("format") != FORMAT_SUPPORT:
            raise TruthError("support_truth.json has format %r, not %r" % (doc.get("format"), FORMAT_SUPPORT))
        ids = set(truth.unit)
        self.env_parts = {str(k): [str(x) for x in v] for k, v in doc["env_parts"].items()}
        self.scored = sorted(str(c) for c in doc["scored_env_classes"])
        want = sorted("env:" + c for c in cfg["C4"]["support_classes"])
        if self.scored != want:
            raise TruthError("support_truth.json scores the environment classes %s, the grader %s" % (self.scored, want))
        if abs(float(doc["env_snap_m"]) - float(cfg["C4"]["env_snap_m"])) > 1e-12:
            raise TruthError("support_truth.json was built with env_snap_m %s, the grader uses %s" % (
                doc["env_snap_m"], cfg["C4"]["env_snap_m"]))
        self.held = sorted(str(h) for h in doc["held"])
        held_truth = sorted(u.id for u in truth.units if u.rest == "held")
        if self.held != held_truth:
            raise TruthError("support_truth.json holds %s, items_truth.json %s" % (self.held, held_truth))
        self.ignore = sorted(u.id for u in truth.units if u.ignore)
        self.static_units = sorted(str(u) for u in doc["static_units"])
        bad = [u for u in self.static_units if u not in ids or truth.unit[u].ignore or u in self.held]
        if bad:
            raise TruthError("support_truth.json static_units holds units that are not resting counted units: %s" % bad)
        self.not_static = {str(r["unit"]): list(r.get("not_static_in", [])) for r in doc.get("not_static", [])}
        counted_rest = sorted(u.id for u in truth.counted_units() if u.rest == "static")
        if sorted(set(self.static_units) | set(self.not_static)) != counted_rest:
            raise TruthError("support_truth.json static_units + not_static do not cover the resting counted units")

        def node_ok(n, side):
            if n in ids:
                return not truth.unit[n].ignore and n not in self.held
            return side == "supporter" and n in self.scored

        self.required = sorted((str(e["supporter"]), str(e["supported"])) for e in doc["required"])
        badr = [e for e in self.required if not (node_ok(e[0], "supporter") and node_ok(e[1], "supported"))]
        if badr:
            raise TruthError("support_truth.json required edges outside the scored nodes: %s" % badr[:5])
        self.ambiguous = set((str(e["supporter"]), str(e["supported"])) for e in doc["ambiguous"])
        self.nominal_ambiguous = set((str(e["supporter"]), str(e["supported"])) for e in doc["ambiguous"]
                                     if "nominal" in e.get("present_in", []))
        if set(self.required) & self.ambiguous:
            raise TruthError("support_truth.json lists an edge as required and ambiguous")
        self.excluded_units = set(self.held) | set(self.ignore)

    def ignored(self, a, b):
        """Why the mapped edge a -> b is left out on both sides (None: it counts)."""
        if a in self.excluded_units or b in self.excluded_units:
            return "held or ignore unit"
        if b in self.not_static:
            return "supported unit not static in every variant"
        if (a, b) in self.ambiguous:
            return "ambiguous in the truth"
        return None


def _cached(truth, key, fn):
    d = _CACHE.setdefault(truth, {})
    if key not in d:
        d[key] = fn()
    return d[key]


def support_truth(truth, cfg):
    return _cached(truth, "support_truth", lambda: SupportTruth(truth.load("support_truth"), truth, cfg))


def env_parts(truth, cfg):
    return _cached(truth, "env_parts", lambda: EnvParts(truth.twin, support_truth(truth, cfg).env_parts))


# ---------------------------------------------------------------------------------------------------------------
# mapping support pairs into edges (shared with the truth builder)

def edges_from_pairs(pairs, sup_name, sub_name, frac):
    """pairs: twin_sim settle()["support_pairs"]; sup_name(pair) / sub_name(pair) -> mapped name or None (dropped).
    The forces of pairs mapped onto the same (supporter, supported) are summed; an edge needs force_frac of the
    supported unit's weight. Returns (edges {(a, b): weight_frac}, rows [per pair: agent and mapped names, force])."""
    agg, wt, rows = {}, {}, []
    for p in pairs:
        a, b = sup_name(p), sub_name(p)
        rows.append({"pair": [p["supporter"], p["supported"]], "role": p.get("supporter_role"), "mapped": [a, b],
                     "force_N": p["force_N"], "point": p.get("point")})
        if a is None or b is None or a == b:
            continue
        agg[(a, b)] = agg.get((a, b), 0.0) + float(p["force_N"])
        wt[(a, b)] = float(p["weight_N"])
    out = {}
    for k in sorted(agg):
        w = wt[k]
        if w > 0 and agg[k] >= frac * w:
            out[k] = round(agg[k] / w, 6)
    return out, rows


# ---------------------------------------------------------------------------------------------------------------
# the physics of the graded twin (cached for C5)

def _pairs(ctx):
    M = ctx.matching
    a2t = dict(M.a2t)
    t2a = dict(M.t2a)
    removed = {p["agent"]: p["true"] for p in M.pairs if p.get("ignore")}
    return a2t, t2a, removed


def physics(ctx):
    """The standard settle of the graded twin, computed once per grade (C4 and C5 share it)."""
    return ctx.cache("settle", lambda: _physics(ctx))


def _physics(ctx):
    import mujoco
    TS = ctx.payload["twin_sim"]
    tw = ctx.twin
    a2t, _t2a, removed = _pairs(ctx)
    m = tw.physics_model()
    held_true = {u.id for u in ctx.truth.units if u.rest == "held"}
    free_units = set()
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.body_parentid[m.jnt_bodyid[j]] == 0:
            free_units.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])))
    # agent units paired with a held true unit are held, the pairs with 'ignore' entries included [the truth
    # builders hold every held unit, ignore entries too (the real site's foam roll u052, whose support is
    # hidden); left free, it falls in the true twin's own settle and the self-test fails]
    pairs = sorted(set(a2t.items()) | set(removed.items()))
    hold = {a: "world" for a, t in pairs if t in held_true and a in free_units}
    roles = TS.standardize(m, gates=tw.gate_bodies(), hold=hold)
    st = TS.settle(m, roles)
    return {"model": m, "roles": roles, "settle": st, "hold": sorted(hold), "finite": bool(st["finite"]),
            "s": float(ctx.align.s)}


def unit_static(phys, name, cfg_sim, s):
    """The static rule in site metres: the maxima over the relaxation and the settle."""
    r = phys["settle"]["items"].get(name)
    if r is None or not phys["finite"]:
        return False, None
    pos_mm, rot_deg = float(cfg_sim["static"]["pos_mm"]), float(cfg_sim["static"]["rot_deg"])
    dmm = float(r["max_drift_mm"]) * s
    if s == 1.0:
        ok = bool(r["static"])
    else:
        ok = bool(math.isfinite(dmm) and dmm <= pos_mm and float(r["max_rot_deg"]) <= rot_deg)
    return ok, {"max_drift_site_mm": dmm if math.isfinite(dmm) else None, "max_rot_deg": r["max_rot_deg"],
                "relax_drift_mm": r.get("relax_drift_mm"), "settle_drift_mm": r.get("settle_drift_mm"), "static": ok}


def twin_edges(ctx, phys, ST, parts):
    """The graded twin's support edges in true names, after the ignore rules: (edges {(a, b): weight_frac},
    rows, ignored [(a, b, why)])."""
    if not phys["finite"]:
        return {}, [], []
    a2t, _t2a, removed = _pairs(ctx)
    counted = set(ctx.zones.of_agent)
    tol = float(ctx.cfg["C4"]["env_snap_m"])
    simmap = ctx.align.map
    pairs = phys["settle"]["support_pairs"]
    env_pts = sorted({i for i, p in enumerate(pairs) if p["supporter_role"] == "env" and p.get("point")})
    cls = {}
    if env_pts:
        X = simmap.to_site(np.asarray([pairs[i]["point"] for i in env_pts], float))
        for i, c in zip(env_pts, parts.classify(X, tol)):
            cls[i] = c
    index = {id(p): i for i, p in enumerate(pairs)}

    def unit_name(b):
        if b in removed:
            return removed[b]
        if b in a2t:
            return a2t[b]
        return "agent:" + b

    def sup_name(p):
        role = p["supporter_role"]                       # from the geom, never from the (agent's) name
        if role == "env":
            return cls.get(index[id(p)], NONE_NODE)
        if role == "gate":
            return GATE_NODE
        if role != "item":
            return None
        return unit_name(p["supporter"])

    def sub_name(p):
        b = p["supported"]
        if b in removed:
            return removed[b]
        if b not in a2t and b not in counted:
            return None
        return unit_name(b)

    frac = float(phys["roles"].config["support"]["force_frac"])
    edges, rows = edges_from_pairs(pairs, sup_name, sub_name, frac)
    out, ignored = {}, []
    for (a, b), wf in sorted(edges.items()):
        why = ST.ignored(a, b)
        if why:
            ignored.append((a, b, why))
            continue
        out[(a, b)] = wf
    return out, rows, ignored


# ---------------------------------------------------------------------------------------------------------------
# the criterion

def _zone_of_node(ctx, n):
    if n.startswith("agent:"):
        return ctx.zones.of_agent.get(n[len("agent:"):])
    u = ctx.truth.unit.get(n)
    return u.zone if u is not None else None


def score(ctx):
    sec = ctx.sec
    truth = ctx.truth
    ST = support_truth(truth, ctx.cfg)
    parts = env_parts(truth, ctx.cfg)
    phys = physics(ctx)
    cfg_sim = phys["roles"].config
    s = phys["s"]
    a2t, t2a, _removed = _pairs(ctx)
    Z = ctx.zones
    q = ctx.q
    # static share
    not_scored_true = set(ST.held) | set(ST.not_static)
    rows = {}
    for tid in ST.static_units:
        a = t2a.get(tid)
        ok, info = unit_static(phys, a, cfg_sim, s) if a is not None else (False, None)
        rows[tid] = {"agent": a, "q": float(q.get(tid, 0.0)) if a is not None else 0.0, "static": ok, "settle": info,
                     "zone": truth.unit[tid].zone}
    per = {}
    for z in Z.ids:
        trues = [t for t in ST.static_units if truth.unit[t].zone == z]
        num = sum(rows[t]["q"] * (1.0 if rows[t]["static"] else 0.0) for t in trues)
        n_out = sum(1 for a, zz in Z.of_agent.items() if zz == z and a2t.get(a) in not_scored_true)
        den = max(len(trues), int(Z.n_sub.get(z, 0)) - n_out)
        per[z] = {"static": {"numerator": num, "denominator": den, "n_true": len(trues), "n_submitted": Z.n_sub.get(z, 0),
                             "n_paired_not_scored": n_out, "share": (num / den) if den > 0 else None}}
    # support F1
    edges, erows, ignored = twin_edges(ctx, phys, ST, parts)
    req = set(ST.required)

    def qn(n):
        if n.startswith("env:") or n.startswith("gate:"):
            return 1.0
        if n.startswith("agent:"):
            return 0.0
        return float(q.get(n, 0.0))

    tw_set = set(edges)
    tp = sorted(req & tw_set)
    for z in Z.ids:
        rz = [e for e in req if truth.unit[e[1]].zone == z]
        wz = [e for e in tw_set if _zone_of_node(ctx, e[1]) == z]
        tz = [e for e in tp if truth.unit[e[1]].zone == z]
        w = sum(min(qn(a), qn(b)) for a, b in tz)
        n = len(rz) + len(wz)
        per[z]["support"] = {"n_required": len(rz), "n_twin_edges": len(wz), "tp_weight": w,
                             "f1": (2.0 * w / n) if n else None}
    w_st, w_su = float(sec["w_static"]), float(sec["w_support"])
    for z in Z.ids:
        terms = [(w_st, per[z]["static"]["share"]), (w_su, per[z]["support"]["f1"])]
        terms = [(w, v) for w, v in terms if v is not None]
        tot = sum(w for w, _v in terms)
        per[z]["score"] = (sum(w * v for w, v in terms) / tot) if tot > 0 else None
    value = mean_over_zones({z: per[z]["score"] for z in Z.ids})
    no_zone = sorted(e for e in tw_set if _zone_of_node(ctx, e[1]) is None)
    detail = {"per_zone": per, "static": rows, "s_star": s, "held": phys["hold"],
              "settle": {"finite": phys["finite"], "unstable": phys["settle"]["unstable"],
                         "warnings": phys["settle"]["warnings"], "relax_s": phys["settle"]["relax_s"],
                         "seconds": phys["settle"]["seconds"]},
              "support": {"required": [list(e) for e in sorted(req)], "twin": [list(e) for e in sorted(tw_set)],
                          "tp": [list(e) for e in tp], "missed": [list(e) for e in sorted(req - tw_set)],
                          "extra": [list(e) for e in sorted(tw_set - req)], "ignored": [list(e) for e in ignored],
                          "edges_without_zone": [list(e) for e in no_zone], "pairs": erows},
              "static_share": mean_over_zones({z: per[z]["static"]["share"] for z in Z.ids}),
              "support_f1": mean_over_zones({z: per[z]["support"]["f1"] for z in Z.ids})}
    diag = ""
    if not phys["finite"]:
        diag = "the standard settle went non-finite: no unit is static and no support edge exists"
    return {"value": value, "detail": detail, "diagnostic": diag}


def selftest(ctx, result):
    """On the true twin: the relaxation leaves every resting unit ('ignore' entries included) within
    C4.selftest.relax_mm, and the settle every counted one within settle_mm / settle_deg (ignore entries:
    ignore_settle_mm); every unit of static_units is static; the mapped
    edges are exactly the required ones (and the nominal ambiguous ones are the ones ignored); every environment pair
    of the settle is classified onto the class its true geom's name gives. Raises TruthError."""
    sec = ctx.cfg["C4"]["selftest"]
    truth = ctx.truth
    ST = support_truth(truth, ctx.cfg)
    phys = physics(ctx)
    st = phys["settle"]
    if not st["finite"] or st["warnings"] or st["unstable"]:
        raise TruthError("the true twin's settle is not clean: finite %s, warnings %s" % (st["finite"], st["warnings"]))
    worst = {"relax_mm": 0.0, "settle_mm": 0.0, "settle_deg": 0.0, "ignore_mm": 0.0}
    for u in truth.units:
        r = st["items"].get(u.id)
        if r is None or r["held"]:
            continue
        # the relaxation must leave every unit of the true twin, 'ignore' entries included, within relax_mm
        worst["relax_mm"] = max(worst["relax_mm"], float(r["relax_drift_mm"]))
        if u.ignore:
            worst["ignore_mm"] = max(worst["ignore_mm"], float(r["max_drift_mm"]))
            continue
        worst["settle_mm"] = max(worst["settle_mm"], float(r["max_drift_mm"]))
        worst["settle_deg"] = max(worst["settle_deg"], float(r["max_rot_deg"]))
    if (worst["relax_mm"] > float(sec["relax_mm"]) or worst["settle_mm"] > float(sec["settle_mm"])
            or worst["settle_deg"] > float(sec["settle_deg"]) or worst["ignore_mm"] > float(sec["ignore_settle_mm"])):
        raise TruthError("the true twin is not at rest: %s (limits %s)" % (worst, sec))
    det = (result or {}).get("detail", {})
    bad = [t for t, r in det.get("static", {}).items() if not r["static"]]
    if bad:
        raise TruthError("true units listed static are not static in the grader's settle: %s" % bad[:5])
    sup = det.get("support", {})
    got = set(tuple(e) for e in sup.get("twin", []))
    if got != set(ST.required):
        raise TruthError("the true twin's support edges differ from the truth: missing %s, extra %s" % (
            sorted(set(ST.required) - got)[:5], sorted(got - set(ST.required))[:5]))
    amb = set((a, b) for a, b, why in sup.get("ignored", []) if why == "ambiguous in the truth")
    if amb != ST.nominal_ambiguous:
        raise TruthError("the true twin's ambiguous edges differ: %s vs %s" % (sorted(amb)[:5],
                                                                             sorted(ST.nominal_ambiguous)[:5]))
    name_class = {}
    for cls, names in ST.env_parts.items():
        for nm in names:
            name_class[nm] = cls
    wrong = []
    for row in sup.get("pairs", []):
        a = row["pair"][0]
        if row.get("role") != "env":
            continue
        want = name_class.get(a[len("env:"):])
        if want is not None and row["mapped"][0] != want:
            wrong.append((a, row["pair"][1], row["mapped"][0], want))
    if wrong:
        raise TruthError("the environment classifier maps true support contacts onto the wrong part: %s" % wrong[:5])
    return {"worst": worst, "edges": len(got), "nominal_ambiguous": len(amb)}
