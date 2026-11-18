"""C5 interaction: w_lift x lift + w_gate x gate (ungated; the orchestrator multiplies by C1).

Lift (over the whole site). The lift probes of the truth (probe_truth.json "lift"): each probed true unit's paired
agent unit is lifted by C5.lift_m (in the twin's own units: lift_m / s*) with twin_sim's standard drive, from the
settled state of C4's settle (score_c4.physics: the relaxation and the dynamic settle, held true units' pairs held);
every other unit's displacement is read in site metres (x s*) and classified by twin_sim's outcome rule (carried:
rises with it; disturbed: moves more than 2 cm or turns more than 10 deg; at s* = 1 exactly, twin_sim's own label).
Outcome sets hold the other units that end "carried" or "disturbed", by true name (an agent unit through its pair,
an unpaired one as "agent:<body>", which never matches); left out on both sides: units paired with held or 'ignore'
true units, units matched to 'ignore' entries, the probe's ambiguous units (outcomes that differ between the
truth's variants), agent units the zones do not count (outside the region), and held agent units.
- A probe whose true set is non-empty scores q_i x J (J = |units in both sets with the same outcome| / |units in
  either set|); 0 when the true unit has no pair (or its pair was excluded) or the probe went non-finite.
- An empty-set probe only charges false movers (two units merged into one body, a unit resting on its neighbour):
  charge 1 when the twin's set is not empty (C5.empty_charge "any"; "share": the share of the taking-part units that
  moved), 1 for a non-finite probe; a probe whose true unit has no pair is not run and charges nothing (C1 and C2
  charge the missing unit).
- lift = max(0, mean over the non-empty probes of q_i x J - false_mover_penalty x mean over the empty probes that ran
  of their charge).
- Fewer than C5.min_nonempty_probes non-empty probes in the truth (after the variants): the lift term is not scored,
  C5 is the gate term alone and the weights move (C5.fallback_weights: C4 0.14, C5 0.06), returned as a reweight.

Gate. Per true gate (probe_truth.json "gates", site frame), the agent gate whose leaf centre (the centre of its
collision geometry at qpos0, twin_gate.gate_travel, mapped by the alignment) lies within C5.gate_match_m; one to one,
nearest first. A matched gate scores band(angle between the slide axes as lines, axis_band_deg) x travel_rule; the
travel rule (twin_sim.gate_rule) is checked in the TRUE opening direction: the true axis
times its opening sign, mapped into the twin, gives the sign of the agent's slide axis that opens. Its closing
condition allows C5.max_closing_m of travel beyond the surveyed position against that direction, plus the true
gate's own travel that way (0 for a gate surveyed shut), so a range symmetric about the surveyed position fails.
gate = sum of the scores / max(number of true gates, number of submitted gates that are matched or whose mapped leaf
centre lies in the region).
"""
import math
import weakref

import numpy as np

from . import score_c4
from .config import band_down
from .truth import TruthError

FORMAT_PROBE = "site-probe-truth/1"
MOVED = ("carried", "disturbed")
_CACHE = weakref.WeakKeyDictionary()


class ProbeTruth:
    def __init__(self, doc, truth, cfg, sim_cfg):
        if doc.get("format") != FORMAT_PROBE:
            raise TruthError("probe_truth.json has format %r, not %r" % (doc.get("format"), FORMAT_PROBE))
        ids = set(truth.unit)
        held = {u.id for u in truth.units if u.rest == "held"}
        ign = {u.id for u in truth.units if u.ignore}
        self.excluded = held | ign
        sec = cfg["C5"]
        # the public probe protocol must be the one the truth was built with and the one the grader states
        prot = doc.get("protocol", {})
        if abs(float(sec["lift_m"]) * 1000.0 - float(sim_cfg["lift"]["lift_mm"])) > 1e-9:
            raise TruthError("C5.lift_m %s differs from twin_sim_config lift.lift_mm %s" % (
                sec["lift_m"], sim_cfg["lift"]["lift_mm"]))
        if abs(float(sec["disturbed_m"]) * 1000.0 - float(sim_cfg["outcome"]["disturbed_mm"])) > 1e-9:
            raise TruthError("C5.disturbed_m %s differs from twin_sim_config outcome.disturbed_mm %s" % (
                sec["disturbed_m"], sim_cfg["outcome"]["disturbed_mm"]))
        for k in ("lift", "outcome"):
            mine = {kk: v for kk, v in sim_cfg[k].items() if kk != "note"}
            theirs = {kk: v for kk, v in prot.get(k, {}).items() if kk != "note"}
            if mine != theirs:
                raise TruthError("probe_truth.json was built with the probe protocol %s %s, twin_sim_config has %s" % (
                    k, theirs, mine))
        gr = sim_cfg["gate_rule"]
        if (float(gr["min_width_ratio"]), float(gr["max_width_ratio"]), float(gr["min_open_m"]),
                float(gr["max_closing_m"])) != (
                float(sec["travel_width_ratio"][0]), float(sec["travel_width_ratio"][1]), float(sec["min_open_m"]),
                float(sec["max_closing_m"])):
            raise TruthError("C5's travel rule %s / %s / %s differs from twin_sim_config gate_rule %s" % (
                sec["travel_width_ratio"], sec["min_open_m"], sec["max_closing_m"], gr))
        lift = doc["lift"]
        self.nonempty, self.empty = [], []
        seen = set()
        for rec in lift["nonempty"]:
            u = str(rec["unit"])
            S = {str(k): "carried" for k in rec["carried"]}
            S.update({str(k): "disturbed" for k in rec["disturbed"]})
            amb = set(str(k) for k in rec.get("ambiguous", []))
            self._check(u, S, amb, ids, held, ign, seen)
            if not S:
                raise TruthError("probe_truth.json: the non-empty probe of %s has an empty set" % u)
            self.nonempty.append({"unit": u, "set": S, "ambiguous": amb})
        for rec in lift["empty"]:
            u = str(rec["unit"])
            amb = set(str(k) for k in rec.get("ambiguous", []))
            self._check(u, {}, amb, ids, held, ign, seen)
            self.empty.append({"unit": u, "set": {}, "ambiguous": amb})
        self.entries = doc.get("entries", {})
        self.gates = []
        for g in doc.get("gates", []):
            rec = {"id": str(g["id"]), "leaf_centre": np.asarray(g["leaf_centre"], float).reshape(3),
                   "axis": np.asarray(g["axis"], float).reshape(3), "opening_sign": int(g["opening_sign"]),
                   "width_m": float(g["width_m"]), "travel_m": [float(v) for v in g["travel_m"]],
                   "qpos0_m": float(g["qpos0_m"])}
            n = float(np.linalg.norm(rec["axis"]))
            if rec["opening_sign"] not in (1, -1) or not n > 0:
                raise TruthError("probe_truth.json: gate %s needs a unit axis and an opening sign +-1" % rec["id"])
            rec["axis"] = rec["axis"] / n
            lo, hi = rec["travel_m"]
            q0 = rec["qpos0_m"]
            if not (lo - 1e-9 <= q0 <= hi + 1e-9):
                raise TruthError("probe_truth.json: gate %s is surveyed outside its own travel" % rec["id"])
            # how far the true gate itself can travel beyond its surveyed position against its opening direction
            # (site metres: the truth is the site frame); the closing condition of the travel rule adds it
            rec["closing_m"] = max(0.0, (q0 - lo) if rec["opening_sign"] > 0 else (hi - q0))
            self.gates.append(rec)
        tgates = sorted(g["body"] for g in truth.twin.gates)
        if sorted(g["id"] for g in self.gates) != tgates:
            raise TruthError("probe_truth.json gates %s are not the true twin's gates %s" % (
                sorted(g["id"] for g in self.gates), tgates))

    def _check(self, u, S, amb, ids, held, ign, seen):
        if u not in ids or u in held or u in ign:
            raise TruthError("probe_truth.json probes %r, which is not a resting counted true unit" % u)
        if u in seen:
            raise TruthError("probe_truth.json probes %s twice" % u)
        seen.add(u)
        bad = [k for k in list(S) + sorted(amb) if k not in ids or k in held or k in ign or k == u]
        if bad:
            raise TruthError("probe_truth.json: the probe of %s names units outside the counted resting units: %s" % (
                u, bad[:5]))
        if set(S) & amb:
            raise TruthError("probe_truth.json: the probe of %s lists a unit both in its set and as ambiguous" % u)


def probe_truth(truth, cfg, sim_cfg):
    d = _CACHE.setdefault(truth, {})
    if "probe_truth" not in d:
        d["probe_truth"] = ProbeTruth(truth.load("probe_truth"), truth, cfg, sim_cfg)
    return d["probe_truth"]


def _jaccard(true_set, twin_set):
    nodes = set(true_set) | set(twin_set)
    if not nodes:
        return 1.0, 0, 0
    inter = sum(1 for n in nodes if n in true_set and n in twin_set and true_set[n] == twin_set[n])
    return inter / len(nodes), inter, len(nodes)


def _classify(o, s, driven_site_mm, sim_cfg, TS):
    if s == 1.0:
        return o["outcome"]
    return TS.classify(s * np.asarray(o["d_mm"], float), float(o["rot_deg"]), driven_site_mm, sim_cfg, True)


def lift_term(ctx, PT, phys):
    """(lift or None when not scored, detail)."""
    TS = ctx.payload["twin_sim"]
    sec = ctx.sec
    a2t, t2a, removed = score_c4._pairs(ctx)
    counted = set(ctx.zones.of_agent)
    s = phys["s"]
    roles = phys["roles"]
    m = phys["model"]
    sim_cfg = roles.config
    held_true = {u.id for u in ctx.truth.units if u.rest == "held"}
    ign_true = {u.id for u in ctx.truth.units if u.ignore}
    unit_names = set(roles.item_names())
    held_agent = {roles.name(b) for b in roles.hold}
    L_site_mm = 1000.0 * float(sec["lift_m"])
    driven = [0.0, 0.0, L_site_mm]
    scored = len(PT.nonempty) >= int(sec["min_nonempty_probes"])
    rows, ne_scores, charges = {}, [], []
    for probe, kind in [(p, "nonempty") for p in PT.nonempty] + [(p, "empty") for p in PT.empty]:
        u = probe["unit"]
        j = t2a.get(u)
        row = {"kind": kind, "agent": j, "q": float(ctx.q.get(u, 0.0)) if j is not None else 0.0,
               "true_set": dict(probe["set"])}
        if j is None or j not in unit_names or j in held_agent:
            why = "the true unit is not paired" if j is None else (
                "its pair was excluded by the gate" if j not in unit_names else "its pair is held")
            if kind == "nonempty":
                row.update(score=0.0, jaccard=0.0, why=why)
                ne_scores.append(0.0)
            else:
                row.update(charge=None, why=why + ": not run")
            rows["lift:" + u] = row
            continue
        if not phys["finite"]:
            r = {"finite": False, "outcomes": {}}
        else:
            r = TS.lift_probe(m, roles, j, qpos=phys["settle"]["qpos_end"], lift_mm=L_site_mm / s, config=sim_cfg)
        if not r["finite"]:
            if kind == "nonempty":
                row.update(score=0.0, jaccard=0.0, why="the probe went non-finite")
                ne_scores.append(0.0)
            else:
                row.update(charge=1.0, why="the probe went non-finite")
                charges.append(1.0)
            rows["lift:" + u] = row
            continue
        S_a, n_part, outs = {}, 0, {}
        for k, o in sorted(r["outcomes"].items()):
            if k == j or o["held"]:
                continue
            if k in removed:
                continue
            t = a2t.get(k)
            if t is not None and (t in held_true or t in ign_true or t in probe["ambiguous"]):
                continue
            if t is None and k not in counted:
                continue
            node = t if t is not None else "agent:" + k
            cls = _classify(o, s, driven, sim_cfg, TS)
            outs[node] = cls
            n_part += 1
            if cls in MOVED:
                S_a[node] = cls
        row["twin_set"] = S_a
        row["outcomes"] = outs
        if kind == "nonempty":
            J, inter, uni = _jaccard(probe["set"], S_a)
            row.update(score=row["q"] * J, jaccard=J, intersection=inter, union=uni)
            ne_scores.append(row["score"])
        else:
            if sec["empty_charge"] == "share":
                c = (len(S_a) / n_part) if n_part else 0.0
            else:
                c = 1.0 if S_a else 0.0
            row.update(charge=c, false_movers=sorted(S_a), taking_part=n_part)
            charges.append(c)
        rows["lift:" + u] = row
    base = (sum(ne_scores) / len(ne_scores)) if ne_scores else 0.0
    pen = (sum(charges) / len(charges)) if charges else 0.0
    lift = max(0.0, base - float(sec["false_mover_penalty"]) * pen)
    detail = {"scored": scored, "n_nonempty": len(PT.nonempty), "n_empty": len(PT.empty),
              "n_empty_run": len(charges), "nonempty_mean": base, "false_mover_mean": pen, "lift": lift,
              "probes": rows}
    return (lift if scored else None), detail


def gate_term(ctx, PT):
    """(gate or None without true gates, detail)."""
    TS = ctx.payload["twin_sim"]
    sec = ctx.sec
    mp = ctx.align.map
    s = float(ctx.align.s)
    rule = {"min_width_ratio": float(sec["travel_width_ratio"][0]),
            "max_width_ratio": float(sec["travel_width_ratio"][1]), "min_open_m": float(sec["min_open_m"]),
            "max_closing_m": float(sec["max_closing_m"])}
    tw = ctx.twin
    agents = []
    for g in tw.gates:
        tr = g.get("travel") or {}
        if tr.get("leaf_centre") is not None:
            c = np.asarray(tr["leaf_centre"], float)
            ax = np.asarray(tr["axis"], float)
        else:
            c = np.asarray(tw.data.xpos[g["body_id"]], float)
            ax = np.asarray(tw.data.xaxis[g["joint_id"]], float)
        cs = mp.to_site(c[None])[0]
        axs = mp.dir_to_site(ax[None])[0]
        axs = axs / max(float(np.linalg.norm(axs)), 1e-12)
        agents.append({"body": g["body"], "centre_site": cs, "axis_site": axs, "travel": tr})
    lim = float(sec["gate_match_m"])
    cand = []
    for k, tg in enumerate(PT.gates):
        for i, ag in enumerate(agents):
            dist = float(np.linalg.norm(ag["centre_site"] - tg["leaf_centre"]))
            if dist <= lim:
                cand.append((dist, tg["id"], ag["body"], k, i))
    cand.sort()
    used_t, used_a, match = set(), set(), {}
    for dist, _tid, _ab, k, i in cand:
        if k in used_t or i in used_a:
            continue
        used_t.add(k)
        used_a.add(i)
        match[k] = (i, dist)
    rows, total = {}, 0.0
    for k, tg in enumerate(PT.gates):
        if k not in match:
            rows[tg["id"]] = {"agent": None, "score": 0.0, "why": "no agent gate within %.2f m of the leaf" % lim}
            continue
        i, dist = match[k]
        ag = agents[i]
        c = abs(float(ag["axis_site"] @ tg["axis"]))
        ang = math.degrees(math.acos(min(1.0, c)))
        b = band_down(ang, *sec["axis_band_deg"])
        opening = tg["opening_sign"] * tg["axis"]
        dot = float(ag["axis_site"] @ opening)
        sign = 1 if dot > 1e-9 else (-1 if dot < -1e-9 else 0)
        tr = ag["travel"] or {}
        rr = TS.gate_rule(tr, sign, rule, scale=s, true_closing_m=tg["closing_m"]) if tr.get("width_m") is not None \
            else {"ok": False, "reasons": ["the gate has no collision geometry"], "sign": sign}
        sc = b * (1.0 if rr["ok"] else 0.0)
        total += sc
        rows[tg["id"]] = {"agent": ag["body"], "distance_m": dist, "axis_deg": ang, "axis_band": b,
                          "opening_sign_agent": sign, "true_closing_m": tg["closing_m"], "rule": rr, "score": sc}
    n_sub = 0
    outside = []
    for i, ag in enumerate(agents):
        if i in used_a or bool(ctx.truth.region.inside(ag["centre_site"][None, :2])[0]):
            n_sub += 1
        else:
            outside.append(ag["body"])
    n_true = len(PT.gates)
    den = max(n_true, n_sub)
    detail = {"per_true_gate": rows, "n_true": n_true, "n_submitted_in_region": n_sub,
              "submitted_outside_region": outside, "denominator": den,
              "agents": {ag["body"]: {"centre_site": ag["centre_site"], "axis_site": ag["axis_site"]} for ag in agents}}
    if n_true == 0:
        return None, detail
    gate = total / den
    detail["gate"] = gate
    return gate, detail


def reweight(truth, cfg):
    """The weight change C5 asks for, from the truth alone: {C4: +d, C5: -d} when fewer than min_nonempty_probes
    non-empty lift probes survive the truth's variants, else {} (the orchestrator may apply it even when C5's stage
    fails, so a submission is weighted like the self-test)."""
    sec = cfg["C5"]
    n = len(truth.load("probe_truth")["lift"]["nonempty"])
    if n >= int(sec["min_nonempty_probes"]):
        return {}
    fb, crit = sec["fallback_weights"], cfg["criteria"]
    return {"C4": float(fb["C4"]) - float(crit["C4"]["weight"]), "C5": float(fb["C5"]) - float(crit["C5"]["weight"])}


def score(ctx):
    sec = ctx.sec
    phys = score_c4.physics(ctx)
    PT = probe_truth(ctx.truth, ctx.cfg, phys["roles"].config)
    lift, ld = lift_term(ctx, PT, phys)
    gate, gd = gate_term(ctx, PT)
    terms = []
    if lift is not None:
        terms.append((float(sec["w_lift"]), lift))
    if gate is not None:
        terms.append((float(sec["w_gate"]), gate))
    tot = sum(w for w, _v in terms)
    value = (sum(w * v for w, v in terms) / tot) if tot > 0 else 0.0
    out = {"value": value, "detail": {"lift": ld, "gate": gd, "lift_scored": lift is not None,
                                      "terms": [[w, v] for w, v in terms]}}
    if not ld["scored"]:
        out["reweight"] = reweight(ctx.truth, ctx.cfg)
        out["diagnostic"] = ("lift not scored: %d non-empty lift probes survive the truth's variants (< %d); C5 is the "
                             "gate term alone" % (ld["n_nonempty"], int(sec["min_nonempty_probes"])))
    if not terms:
        out["diagnostic"] = "neither a lift nor a gate term exists in the truth"
    return out


def selftest(ctx, result):
    """On the true twin: every probe entry the truth records as unanimous is reproduced (including 'stayed'); every
    true gate is matched to itself at distance 0 and angle 0 with its rule met; the stored gate geometry is the true
    twin's. Raises TruthError."""
    phys = score_c4.physics(ctx)
    PT = probe_truth(ctx.truth, ctx.cfg, phys["roles"].config)
    det = (result or {}).get("detail", {})
    probes = det.get("lift", {}).get("probes", {})
    wrong, n = [], 0
    for u, ent in sorted(PT.entries.items()):
        row = probes.get("lift:" + u)
        if row is None or "outcomes" not in row:
            raise TruthError("the true twin's lift probe of %s did not run" % u)
        for k, e in sorted(ent.items()):
            if not e.get("unanimous"):
                continue
            n += 1
            if row["outcomes"].get(k) != e["nominal"]:
                wrong.append((u, k, row["outcomes"].get(k), e["nominal"]))
    if wrong:
        raise TruthError("the true twin's probe outcomes differ from the truth on %d entries: %s" % (len(wrong),
                                                                                                    wrong[:5]))
    gd = det.get("gate", {}).get("per_true_gate", {})
    for tg in PT.gates:
        r = gd.get(tg["id"])
        if r is None or r.get("agent") != tg["id"] or r["distance_m"] > 1e-6 or r["axis_deg"] > 1e-4 or not r["rule"]["ok"]:
            raise TruthError("the true gate %s does not score itself: %s" % (tg["id"], r))
        g = next(x for x in ctx.truth.twin.gates if x["body"] == tg["id"])
        tr = g.get("travel") or {}
        if (float(np.max(np.abs(np.asarray(tr["leaf_centre"], float) - tg["leaf_centre"]))) > 1e-6
                or float(np.max(np.abs(np.asarray(tr["axis"], float) - tg["axis"]))) > 1e-6
                or abs(float(tr["width_m"]) - tg["width_m"]) > 1e-6
                or float(np.max(np.abs(np.asarray(tr["range_m"], float) - np.asarray(tg["travel_m"])))) > 1e-9
                or abs(float(tr["qpos0_m"]) - tg["qpos0_m"]) > 1e-9):
            raise TruthError("probe_truth.json's gate %s differs from the true twin's gate geometry" % tg["id"])
    return {"entries_checked": n, "gates_checked": len(PT.gates)}
