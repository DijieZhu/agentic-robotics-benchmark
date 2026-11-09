"""Grader part 2: the physics criteria (C4 static physics, C5 interaction), the truth's self-tests, and the pipeline
one twin goes through (gate, G0's settle check, part 1, C4, C5), run by the verify worker.

    truth = part1.Truth(gt_twin_dir, depth_ref_png, depth_class_png, vocabulary)
    truth2 = Truth2(truth, support_truth_json, probe_truth_json)        # TruthError when the files disagree
    res = grade_twin(folder, truth, truth2, cfg, guard)                   # JSON-ready: G0, criteria, details
    selftest(truth, truth2, cfg)                                          # the true twin through grade_twin

The physics of a twin (`settle_twin`, `probe_twin`) is twin_sim's standard simulation on the gate's physics model
(load.Twin.physics_model(): the excluded items deleted):
- the twin's items paired (part 1) with the true twin's held items are held kinematically, with the true twin's hold
  frames (an item resting in the drawer follows the drawer in the drawer probe), as in the truth's own runs;
- the standard settle (twin_sim.settle) from qpos0; the static rule with the displacement in true units (x s*);
- support edges (twin_sim's rule, averaged over the end of the settle) are mapped into true names: an item through its
  pair (an unpaired item keeps its own name, "agent:<body>"), a furniture or drawer supporter onto the true surface
  it touches: the mean of the contact points between the two in the settled state (the closest points when no contact
  is active), mapped into the true twin's world through the alignment (R_pp, s*), snapped to the nearest
  environment geom of the true twin within static_physics.snap_mm (else "furniture:other");
- the probes from the settled state: each true item that rests is lifted through its paired agent item by
  lift_mm / s* (the twin's units), the drawer pulled out by out_mm / s*; displacements are read in true units (x s*)
  and classified by twin_sim's outcome rule (at s* = 1 exactly, twin_sim's own classification).
The probes run only for paired true items that rest, so their number never exceeds the true twin's probes.

Guards. Each stage of grade_twin runs under a wall-clock alarm (Guard); a timeout, a MemoryError or an exception in a
stage scores that stage's criteria 0 with a diagnostic (the gate and the settle check: G0 = 0; part 1: C1, C2, C3,
C6 and, without pairs, C4 and C5; the settle: C4 and C5; the probes: C5). The caller (verify.py) runs all of this in
a separate process under an address-space limit and a hard timeout.
"""
import json
import math
import os
import signal
import sys
import time

import numpy as np

from . import align, load, part1, regions, score_c4, score_c5

FORMAT = "twin-score-part2/1"
CRITERIA = ("inventory", "pose_shape", "environment", "static_physics", "interaction", "scale")
PART1_KEYS = {"inventory": "C1", "pose_shape": "C2", "environment": "C3", "scale": "C6"}

DEFAULTS = {
    "weights": {"inventory": 0.15, "pose_shape": 0.25, "environment": 0.10, "static_physics": 0.20,
                "interaction": 0.25, "scale": 0.05},
    "static_physics": {"w_static": 0.5, "w_support": 0.5, "snap_mm": 25.0},
    "interaction": {"w_lift": 0.7, "w_drawer": 0.3, "axis_deg": [4.0, 15.0], "false_mover_penalty": 1.0},
    "selftest": {"gt_settle_mm": 1.0, "gt_settle_deg": 0.5},
}


class TruthError(part1.TruthError):
    """The truth fails its own checks: a task error, never the agent's fault."""


class StageTimeout(Exception):
    pass


def merged(over=None):
    import copy
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------------------------------------------
# the truth of part 2

def _box_sdf(p, c, R, half):
    loc = R.T @ (np.asarray(p, float) - c)
    q = np.abs(loc) - half
    return float(np.linalg.norm(np.maximum(q, 0.0)) + min(float(np.max(q)), 0.0))


class Truth2:
    """The support and probe truth, checked against the true twin; the true environment for the snap; the true
    drawer axis."""

    def __init__(self, truth, support_path, probe_path, cfg=None):
        import mujoco
        self.cfg = merged(cfg)
        TS = load.payload()["twin_sim"]
        with open(support_path) as f:
            sup = json.load(f)
        with open(probe_path) as f:
            pro = json.load(f)
        if sup.get("format") != "gt-support-truth/2" or pro.get("format") != "gt-probe-truth/2":
            raise TruthError("unexpected truth formats %r / %r" % (sup.get("format"), pro.get("format")))
        tw = truth.twin
        names = [it["body"] for it in tw.items]
        self.held = dict(pro["hold_frames"])
        if set(self.held) != set(sup["excluded_items"]) or set(self.held) != set(pro["excluded_items"]):
            raise TruthError("the held items of the support and probe truths differ")
        if not set(self.held) <= set(names) or any(v not in ("world", "drawer") for v in self.held.values()):
            raise TruthError("held items or hold frames not in the true twin")
        self.static_items = [n for n in names if n not in self.held]
        self.required = sorted((e["supporter"], e["supported"]) for e in sup["required"])
        self.ambiguous = set((e["supporter"], e["supported"]) for e in sup["ambiguous"])
        self.nominal_ambiguous = set((e["supporter"], e["supported"]) for e in sup["ambiguous"]
                                     if "nominal" in e.get("present_in", []))
        self.lift = {}
        for i in self.static_items:
            p = pro["probes"].get("lift:" + i)
            if p is None:
                raise TruthError("no probe truth for lift:%s" % i)
            self.lift[i] = self._probe(p, i, names)
        if any(("lift:" + h) in pro["probes"] for h in self.held):
            raise TruthError("the probe truth lifts a held item")
        dname = pro.get("scored_drawer_probe", "drawer:out")
        if dname not in pro["probes"]:
            raise TruthError("no probe truth for %s" % dname)
        self.drawer = self._probe(pro["probes"][dname], None, names)
        # the true environment (every collision geom outside the items), at qpos0, for the snap
        roles = TS.Roles(tw.model, drawer=tw.drawer)
        self.drawer_node = roles.name(roles.drawer) if roles.drawer is not None else "drawer"
        self.env = []
        for g in range(tw.model.ngeom):
            if roles.geom_role[g] not in ("furniture", "drawer"):
                continue
            if int(tw.model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise TruthError("true environment geom %d is not a box" % g)
            self.env.append((roles.geom_node[g], np.asarray(tw.data.geom_xpos[g], float).copy(),
                             np.asarray(tw.data.geom_xmat[g], float).reshape(3, 3).copy(),
                             np.asarray(tw.model.geom_size[g], float).copy()))
        for a, _b in self.required:
            if a.startswith("furniture:") and a not in {e[0] for e in self.env}:
                raise TruthError("required supporter %s is not a true environment geom" % a)
        # the work areas agree with the support truth: every required or ambiguous edge stays inside one area (an
        # item supporter's area, a furniture supporter's from regions.of_supporter); held drawer items are in the
        # drawer's area
        on, of_item, of_sup = regions.config_regions(truth.cfg)
        if on:
            for a, b in self.required + sorted(self.ambiguous):
                ra = of_item.get(a, of_sup.get(a))
                if ra is None or ra != of_item.get(b):
                    raise TruthError("support edge %s -> %s crosses work areas (%s, %s)" % (a, b, ra, of_item.get(b)))
            for h, fr in self.held.items():
                if fr == "drawer" and of_item.get(h) != of_sup.get("drawer"):
                    raise TruthError("held drawer item %s is not in the drawer's work area" % h)
        if roles.drawer_joint is None:
            raise TruthError("the true twin has no drawer joint")
        self.axis_cam = tw.R_cv.T @ np.asarray(tw.data.xaxis[roles.drawer_joint], float)
        self.R_cv, self.p = np.asarray(tw.R_cv, float), np.asarray(tw.p, float)

    def _probe(self, p, lifted, names):
        items = p["items"]
        bad = [k for k in items if k not in names or k in self.held or k == lifted]
        if bad:
            raise TruthError("probe truth entries outside the resting true items: %s" % bad[:5])
        S = {k: v for k, v in p["gt_outcome_set"].items()}
        if any(not items[k]["unambiguous"] or items[k]["nominal"] != c for k, c in S.items()):
            raise TruthError("a probe's outcome set disagrees with its entries")
        return {"set": S, "ambiguous": set(p.get("ambiguous_items", [])),
                "entries": {k: (v["nominal"], bool(v["unambiguous"])) for k, v in items.items()}}

    def snap(self, X_world):
        """(node, signed distance m) of the true environment geom nearest to a point of the true world."""
        best = None
        for node, c, R, h in self.env:
            d = _box_sdf(X_world, c, R, h)
            if best is None or d < best[1]:
                best = (node, d)
        lim = float(self.cfg["static_physics"]["snap_mm"]) / 1000.0
        if best is None or best[1] > lim:
            return "furniture:other", (best[1] if best else None)
        return best


# ---------------------------------------------------------------------------------------------------------------
# alignment helpers

class Mapping:
    """The twin's world into the true camera frame (x_true = s* (M x + t)) and on into the true twin's world."""

    def __init__(self, tw, res1, truth2):
        al = res1["alignment"]
        self.s = float(al["s_star"])
        R_pp = np.asarray(al["R_pp"], float)
        R_arg = None if float(al["R_pp_info"].get("angle_deg", 0.0)) == 0.0 else R_pp
        self.M, self.t = align.frame_map(tw, R_arg)
        self.truth2 = truth2

    def to_true_world(self, x):
        xc = self.s * (self.M @ np.asarray(x, float) + self.t)
        return self.truth2.R_cv @ xc + self.truth2.p

    def axis_angle_deg(self, axis_world):
        a = self.M @ np.asarray(axis_world, float)
        b = self.truth2.axis_cam
        c = abs(float(a @ b)) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))
        return math.degrees(math.acos(min(1.0, c)))


def pairs_maps(res1):
    a2t = {p["agent"]: p["true"] for p in res1["pairs"]}
    t2a = {p["true"]: p["agent"] for p in res1["pairs"]}
    q = {p["true"]: float(p["q"]) for p in res1["pairs"]}
    return a2t, t2a, q


# ---------------------------------------------------------------------------------------------------------------
# physics of a twin

def _contact_point(m, d, Ga, Gb):
    """Mean position of the active contacts between geom sets Ga and Gb, else the midpoint of their closest points
    (within 50 mm), else None."""
    import mujoco
    Ga, Gb = set(int(g) for g in Ga), set(int(g) for g in Gb)
    pts = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        if (g1 in Ga and g2 in Gb) or (g2 in Ga and g1 in Gb):
            pts.append(np.asarray(c.pos, float).copy())
    if pts:
        return np.mean(pts, axis=0), "contacts"
    best = None
    ft = np.zeros(6)
    for ga in sorted(Ga):
        for gb in sorted(Gb):
            dist = mujoco.mj_geomDistance(m, d, ga, gb, 0.05, ft)
            if dist < 0.05 and (best is None or dist < best[0]):
                best = (dist, 0.5 * (ft[:3] + ft[3:]))
    return (best[1], "closest") if best is not None else (None, "none")


def settle_twin(tw, res1, truth, truth2, cfg):
    """The twin's standard settle with its held items; the static flags in true units and the mapped support edges.
    Returns a dict with the model, roles and settled qpos (kept in memory for the probes) and JSON-ready info."""
    import mujoco
    TS = load.payload()["twin_sim"]
    a2t, t2a, q = pairs_maps(res1)
    mp = Mapping(tw, res1, truth2)
    m = tw.physics_model()
    names_m = set()
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            names_m.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])))
    hold = {a: truth2.held[t] for a, t in a2t.items() if t in truth2.held and a in names_m}
    roles = TS.standardize(m, drawer=tw.drawer, hold=hold)
    st = TS.settle(m, roles)
    pos_mm, rot_deg = TS.static_rule(roles.config)
    static, rows = {}, {}
    for nm, r in st["items"].items():
        if r["held"]:
            continue
        if not st["finite"]:
            ok = False
        elif mp.s == 1.0:
            ok = bool(r["static"])
        else:
            ok = bool(math.isfinite(r["max_drift_mm"]) and r["max_drift_mm"] * mp.s <= pos_mm
                      and r["max_rot_deg"] <= rot_deg)
        static[nm] = ok
        rows[nm] = {"max_drift_true_mm": (r["max_drift_mm"] * mp.s) if math.isfinite(r["max_drift_mm"]) else None,
                    "max_rot_deg": r["max_rot_deg"] if math.isfinite(r["max_rot_deg"]) else None, "static": ok,
                    "true": a2t.get(nm)}
    edges, edge_rows = set(), []
    ignored = {"held": [], "ambiguous": []}
    if st["finite"]:
        d = mujoco.MjData(m)
        d.qpos[:] = st["qpos_end"]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        item_names = set(roles.item_names())
        for e in st["support_edges"]:
            sup, sub = e["supporter"], e["supported"]
            t_sub = a2t.get(sub, "agent:" + sub)
            how = None
            if sup in item_names:
                t_sup = a2t.get(sup, "agent:" + sup)
            else:
                Ga = [g for g in range(m.ngeom) if roles.geom_node[g] == sup]
                bid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, sub))
                Gb = [g for g in range(m.ngeom) if int(roles.geom_item[g]) == bid]
                pt, how = _contact_point(m, d, Ga, Gb)
                if pt is None:
                    t_sup, dist = "furniture:other", None
                else:
                    t_sup, dist = truth2.snap(mp.to_true_world(pt))
                how = {"from": how, "snap_dist_mm": None if dist is None else 1000.0 * dist}
            row = {"agent": [sup, sub], "true": [t_sup, t_sub], "weight_frac": e["weight_frac"]}
            if how:
                row["snap"] = how
            edge_rows.append(row)
            if t_sup in truth2.held or t_sub in truth2.held:
                ignored["held"].append([t_sup, t_sub])
                continue
            if (t_sup, t_sub) in truth2.ambiguous:
                ignored["ambiguous"].append([t_sup, t_sub])
                continue
            edges.add((t_sup, t_sub))
    info = {"s_star": mp.s, "finite": bool(st["finite"]), "warnings": st["warnings"], "held": sorted(hold),
            "items": rows, "edges": edge_rows, "ignored_edges": ignored,
            "initial_penetration_mm": st.get("initial_penetration_mm", {})}
    return {"model": m, "roles": roles, "qpos": st["qpos_end"], "finite": bool(st["finite"]), "static": static,
            "edges": edges, "mapping": mp, "info": info, "pairs": (a2t, t2a, q)}


def _outcome(o, s, driven_true, lift, cfg):
    if s == 1.0:
        return o["outcome"]
    TS = load.payload()["twin_sim"]
    return TS.classify(s * np.asarray(o["d_mm"], float), float(o["rot_deg"]), driven_true, cfg, lift)


def probe_twin(phys, truth2, cfg):
    """The lift probes of the paired true items that rest, and the drawer pull-out, from the settled state."""
    import mujoco
    TS = load.payload()["twin_sim"]
    m, roles, q0, mp = phys["model"], phys["roles"], phys["qpos"], phys["mapping"]
    a2t, t2a, _q = phys["pairs"]
    scfg = roles.config
    s = mp.s
    L = float(scfg["lift"]["lift_mm"])
    lifts = {}
    for i in truth2.static_items:
        j = t2a.get(i)
        if j is None or mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, j) not in roles.items:
            continue
        r = TS.lift_probe(m, roles, j, qpos=q0, lift_mm=L / s)
        outs = {k: _outcome(o, s, [0.0, 0.0, L], True, scfg) for k, o in r["outcomes"].items() if not o["held"]}
        lifts[i] = {"finite": bool(r["finite"]), "outcomes": outs}
    drawer, axis = None, None
    if roles.drawer_joint is not None:
        r = TS.drawer_probe(m, roles, "out", qpos=q0, dist_mm=float(scfg["drawer"]["out_mm"]) / s)
        driven = s * np.asarray(r["drawer_d_mm"], float)
        outs = {k: _outcome(o, s, driven, False, scfg) for k, o in r["outcomes"].items() if not o["held"]}
        drawer = {"finite": bool(r["finite"]), "outcomes": outs, "drawer_d_true_mm": driven.tolist()}
        d0 = mujoco.MjData(m)
        d0.qpos[:] = m.qpos0
        mujoco.mj_forward(m, d0)
        axis = mp.axis_angle_deg(d0.xaxis[roles.drawer_joint])
    return lifts, drawer, axis


# ---------------------------------------------------------------------------------------------------------------
# the pipeline of one twin

class Guard:
    """Run stages under a wall-clock alarm: min(cap of the stage, time left before the deadline)."""

    def __init__(self, deadline=None, caps=None, enabled=True):
        self.deadline = deadline
        self.caps = dict(caps or {})
        self.enabled = enabled

    def left(self):
        return float("inf") if self.deadline is None else self.deadline - time.time()

    def run(self, name, fn, *args):
        if not self.enabled:
            return fn(*args)
        cap = min(float(self.caps.get(name, float("inf"))), self.left())
        if cap <= 0.5:
            raise StageTimeout("no time left for stage %s" % name)
        if not math.isfinite(cap):
            return fn(*args)

        def _alarm(_sig, _frm):
            raise StageTimeout("stage %s timed out after %.0f s" % (name, cap))

        old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, cap)
        try:
            return fn(*args)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old)


def _err(exc):
    if isinstance(exc, StageTimeout):
        return "timeout: %s" % exc
    if isinstance(exc, MemoryError):
        return "out of memory (the grader's address-space limit) on this twin"
    return "%s: %s" % (type(exc).__name__, str(exc)[:300])


def _peak_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def grade_twin(folder, truth, truth2, cfg, guard, record=None, keep=None):
    """G0 (gate, settle check) and the six criteria of one twin folder. record(res) is called after every stage with
    the partial result; keep (a dict) receives the in-memory objects (twin, part-1 result, physics) for self-tests."""
    cfg = merged(cfg)
    res = {"format": FORMAT, "G0": 0.0, "g0": {}, "criteria": {}, "diagnostics": {}, "details": {}, "stages": {},
           "stats": {}}

    def stage(name, fn, *args):
        t = time.time()
        try:
            out = guard.run(name, fn, *args)
            res["stages"][name] = {"ok": True, "s": time.time() - t, "peak_rss_mb": _peak_mb()}
            return True, out
        except load.payload()["twin_gate"].TaskError:
            raise
        except part1.TruthError:
            raise
        except BaseException as exc:  # noqa: BLE001 - the twin's fault or the grader's limits: this stage scores 0
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            res["stages"][name] = {"ok": False, "s": time.time() - t, "error": _err(exc), "peak_rss_mb": _peak_mb()}
            return False, None
        finally:
            if record:
                record(res)

    def zero_from(names, why):
        for c in names:
            res["criteria"].setdefault(c, 0.0)
            res["diagnostics"].setdefault(c, why)

    # G0, the gate
    ok, tw = stage("gate", load.load_twin, folder, truth.vocabulary)
    if not ok:
        res["g0"] = {"gate": False, "reasons": [res["stages"]["gate"]["error"]]}
        zero_from(CRITERIA, "G0: " + res["stages"]["gate"]["error"])
        return res
    res["g0"] = {"gate": bool(tw.ok), "reasons": tw.reasons[:20], "excluded": tw.excluded(), "n_items": tw.n_items}
    res["stats"]["n_items"] = float(tw.n_items)
    res["stats"]["n_excluded"] = float(len(tw.excluded()))
    if not tw.ok:
        zero_from(CRITERIA, "G0 gate: " + "; ".join(tw.reasons[:4]))
        return res
    if keep is not None:
        keep["twin"] = tw
    # G0, the settle check
    ok, chk = stage("g0_settle", load.g0_settle, tw)
    if not ok:
        res["g0"]["settle"] = {"fail": True, "reasons": [res["stages"]["g0_settle"]["error"]]}
        zero_from(CRITERIA, "G0 settle check: " + res["stages"]["g0_settle"]["error"])
        return res
    res["g0"]["settle"] = {k: chk.get(k) for k in ("fail", "reasons", "max_energy_gain_J", "energy_bound_J",
                                                    "max_speed_excess_m_s", "max_speed_excess_item", "fail_time_s")}
    if chk["fail"]:
        zero_from(CRITERIA, "G0 settle check: " + "; ".join(chk["reasons"][:3]))
        return res
    res["G0"] = 1.0
    record and record(res)
    # part 1
    ok, r1 = stage("part1", part1.score_part1, folder, truth)
    if not ok or not r1["g0"]["pass"]:
        why = res["stages"]["part1"].get("error") or "part 1 found the gate failing"
        zero_from(CRITERIA, "part 1: " + why)
        return res
    for c, k in PART1_KEYS.items():
        res["criteria"][c] = float(r1[k]["score"])
    al = r1["alignment"]
    res["stats"].update({"s_star": float(al["s_star"]), "n_pairs": float(al["n_pairs"]),
                         "r_pp_deg": float(al["R_pp_info"].get("angle_deg", 0.0))})
    res["details"]["part1"] = {"alignment": {k: al[k] for k in ("s_star", "n_pairs", "R_pp_info", "agent_K")},
                               "pairs": r1["pairs"],
                               "C1": {k: r1["C1"].get(k) for k in ("score", "tp_label", "n_pairs", "n_true",
                                                                    "n_submitted")},
                               "C2": {"score": r1["C2"]["score"],
                                      "per_true": {t: {k: v.get(k) for k in ("agent", "q", "b_pos", "b_rot", "b_iou",
                                                                              "pos_err_mm", "rot_deg", "capped_iou")}
                                                   for t, v in r1["C2"].get("per_true", {}).items()}},
                               "C3": {k: r1["C3"].get(k) for k in ("score", "depth_share", "vertical", "vertical_deg",
                                                                    "depth_scale_used", "hit_frac", "gt_agree_frac",
                                                                    "error", "ungated", "gate_inventory")},
                               "C6": dict(r1["C6"]),
                               "areas": r1.get("areas"),
                               "per_area": {"C1": r1["C1"].get("per_area"), "C2": r1["C2"].get("per_area")}}
    res["stats"]["c3_gt_agree_frac"] = float(r1["C3"].get("gt_agree_frac") or 0.0)
    if keep is not None:
        keep["part1"] = r1
    record and record(res)
    # C4: the settle
    ok, phys = stage("c4", settle_twin, tw, r1, truth, truth2, cfg)
    if not ok:
        zero_from(("static_physics", "interaction"), "the settle: " + res["stages"]["c4"]["error"])
        return res
    a2t, t2a, q = phys["pairs"]
    agent_static = phys["static"]
    c4, i4 = score_c4.score(truth2, q, t2a, agent_static, tw.n_items, phys["edges"], cfg, r1.get("areas"))
    res["criteria"]["static_physics"] = float(c4)
    res["details"]["c4"] = dict(i4, settle=phys["info"])
    res["stats"].update({"static_share": float(i4["static_share"]), "support_f1": float(i4["support_f1"])})
    if keep is not None:
        keep["phys"] = phys
    record and record(res)
    # C5: the probes
    ok, pr = stage("c5", probe_twin, phys, truth2, cfg)
    if not ok:
        zero_from(("interaction",), "the probes: " + res["stages"]["c5"]["error"])
        return res
    lifts, drawer, axis = pr
    travel = tw.report.get("drawer_travel") or {}
    c5, i5 = score_c5.score(truth2, q, t2a, a2t, lifts, drawer, axis, cfg, r1.get("areas"),
                            drawer_travel_ok=bool(travel.get("ok", False)))
    i5["drawer_travel"] = travel
    res["criteria"]["interaction"] = float(c5)
    res["details"]["c5"] = i5
    res["details"]["c5_raw"] = {"lifts": lifts, "drawer": drawer}
    res["stats"].update({"lift": float(i5["lift"]), "lift_penalty": float(i5["lift_penalty"]),
                         "drawer": float(i5["drawer"])})
    if axis is not None:
        res["stats"]["drawer_axis_deg"] = float(axis)
    if keep is not None:
        keep["probes"] = pr
    return res


def twin_score(res, cfg, g1=1.0):
    w = merged(cfg)["weights"]
    tot = sum(float(w[c]) for c in CRITERIA)
    s = sum(float(w[c]) * float(res["criteria"].get(c, 0.0)) for c in CRITERIA) / tot
    return float(g1) * float(res.get("G0", 0.0)) * s


# ---------------------------------------------------------------------------------------------------------------
# self-tests of the truth

def selftest(truth, truth2, cfg, gt_dir):
    """The true twin through grade_twin (the same code as every submission): the gate with no exclusions, G0's settle
    check, every criterion exactly 1; its settle within selftest.gt_settle_mm / gt_settle_deg; its mapped support
    edges equal to the required ones (and the nominal ambiguous ones ignored); every unambiguous probe entry equal to
    its nominal outcome; C3's ray cast agreeing with the reference (part1.selftest). Raises TruthError."""
    cfg = merged(cfg)
    keep = {}
    t0 = time.time()
    res = grade_twin(gt_dir, truth, truth2, cfg, Guard(enabled=False), keep=keep)
    out = {"criteria": dict(res["criteria"]), "G0": res["G0"]}
    if res["G0"] != 1.0:
        raise TruthError("the true twin fails G0: %s" % res["diagnostics"])
    if res["g0"].get("excluded"):
        raise TruthError("the gate excludes true items: %s" % res["g0"]["excluded"])
    bad = {c: res["criteria"].get(c) for c in CRITERIA if res["criteria"].get(c) != 1.0}
    if bad:
        raise TruthError("the true twin does not score 1 in every criterion: %s; %s" % (
            bad, {c: res["diagnostics"].get(c) for c in bad}))
    part1.selftest(truth, keep["part1"])
    # the settle
    st = cfg["selftest"]
    info = res["details"]["c4"]["settle"]
    if not info["finite"] or info["warnings"]:
        raise TruthError("the true twin's settle is not clean: finite %s, warnings %s" % (info["finite"], info["warnings"]))
    worst_mm = max(r["max_drift_true_mm"] for r in info["items"].values())
    worst_deg = max(r["max_rot_deg"] for r in info["items"].values())
    if worst_mm > float(st["gt_settle_mm"]) or worst_deg > float(st["gt_settle_deg"]):
        raise TruthError("the true twin's settle moves %.3f mm / %.3f deg (limit %s mm / %s deg)" % (
            worst_mm, worst_deg, st["gt_settle_mm"], st["gt_settle_deg"]))
    # the support edges
    got = set(keep["phys"]["edges"])
    if got != set(truth2.required):
        raise TruthError("the true twin's support edges differ from the truth: missing %s, extra %s" % (
            sorted(set(truth2.required) - got), sorted(got - set(truth2.required))))
    amb = set(tuple(e) for e in info["ignored_edges"]["ambiguous"])
    if amb != truth2.nominal_ambiguous:
        raise TruthError("the true twin's ambiguous edges differ: %s vs %s" % (sorted(amb), sorted(truth2.nominal_ambiguous)))
    # the probes: every unambiguous entry
    lifts, drawer, _axis = keep["probes"]
    n_checked, wrong = 0, []
    for i in truth2.static_items:
        r = lifts.get(i)
        if r is None or not r["finite"]:
            raise TruthError("the true twin's lift probe of %s did not run" % i)
        for k, (nominal, unamb) in truth2.lift[i]["entries"].items():
            if unamb:
                n_checked += 1
                if r["outcomes"].get(k) != nominal:
                    wrong.append(("lift:" + i, k, r["outcomes"].get(k), nominal))
    if drawer is None or not drawer["finite"]:
        raise TruthError("the true twin's drawer probe did not run")
    for k, (nominal, unamb) in truth2.drawer["entries"].items():
        if unamb:
            n_checked += 1
            if drawer["outcomes"].get(k) != nominal:
                wrong.append(("drawer", k, drawer["outcomes"].get(k), nominal))
    if wrong:
        raise TruthError("the true twin's probe outcomes differ from the truth on %d entries: %s" % (len(wrong), wrong[:5]))
    out.update({"settle_worst_mm": worst_mm, "settle_worst_deg": worst_deg, "edges": len(got),
                "probe_entries_checked": n_checked, "c3_gt_agree_frac": res["stats"].get("c3_gt_agree_frac"),
                "seconds": time.time() - t0, "stages": res["stages"]})
    return out


# ---------------------------------------------------------------------------------------------------------------
# the worker (a separate process started by verify.py)

def _clean(x):
    """JSON-safe copy: numpy scalars and arrays to Python, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in (sorted(x) if isinstance(x, set) else x)]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class Progress:
    def __init__(self, path):
        self.path = path
        self.doc = {"format": "twin-worker-progress/1", "selftest": None, "submission": None, "done": False}

    def write(self):
        self.doc["peak_rss_mb"] = _peak_mb()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_clean(self.doc), f, allow_nan=False)
        os.replace(tmp, self.path)


def worker_main(job_path):
    """Entry of the worker process: limits itself, builds the truth, self-tests it, grades the submission; every step
    is written to the progress file. Exit 0 when done, 3 on a truth failure."""
    import resource
    with open(job_path) as f:
        job = json.load(f)
    lim = int(job["address_space_mb"]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    prog = Progress(job["progress"])
    prog.write()
    deadline = float(job["deadline_wall"])
    caps = job.get("stage_caps_s", {})
    with open(job["grader_config"]) as f:
        gcfg = json.load(f)
    cfg = merged({k: v for k, v in gcfg.items() if k in DEFAULTS})
    t0 = time.time()
    try:
        def build():
            truth = part1.Truth(job["gt_twin"], job["depth_ref"], job["depth_class"], vocabulary=job["vocabulary"],
                                config=gcfg.get("part1"))
            truth2 = Truth2(truth, job["support_truth"], job["probe_truth"], cfg)
            st = selftest(truth, truth2, cfg, job["gt_twin"])
            return truth, truth2, st
        truth, truth2, st = Guard(deadline, {"selftest": caps.get("selftest", float("inf"))}).run("selftest", build)
        prog.doc["selftest"] = dict(ok=True, truth_build_and_selftest_s=time.time() - t0, **st)
    except BaseException as exc:  # noqa: BLE001 - a truth that cannot be built or checked is a task error
        if isinstance(exc, KeyboardInterrupt):
            raise
        import traceback
        prog.doc["selftest"] = {"ok": False, "reason": _err(exc), "trace": traceback.format_exc()[-1500:],
                                "s": time.time() - t0}
        prog.write()
        return 3
    prog.write()

    def record(res):
        prog.doc["submission"] = res
        prog.write()

    try:
        res = grade_twin(job["submission"], truth, truth2, cfg, Guard(deadline, caps), record=record)
    except BaseException as exc:  # noqa: BLE001 - only task errors reach here (vocabulary, truth)
        if isinstance(exc, KeyboardInterrupt):
            raise
        import traceback
        prog.doc["task_error"] = {"reason": _err(exc), "trace": traceback.format_exc()[-1500:]}
        prog.write()
        return 3
    res["twin_score_g1_1"] = twin_score(res, cfg)
    prog.doc["submission"] = res
    prog.doc["done"] = True
    prog.write()
    return 0


if __name__ == "__main__":  # pragma: no cover - verify.py starts the worker through a bootstrap
    sys.exit(worker_main(sys.argv[1]))
