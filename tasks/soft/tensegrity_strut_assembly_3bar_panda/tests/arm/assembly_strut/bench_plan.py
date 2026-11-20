"""The bench plan of the strut cell's reference program (hidden; `tests/hidden/bench_plan.json`): which unit lies in
which nest of the trays, which strut end each motor nest feeds (that end's straight-transfer turn is the nest's base
roll, layout.py) and its hidden roll offset, which slot each nest's unit goes to, and the grasp that works at the nest
and at the slot, stored with the plan so that the runtime candidate list starts with it.

The trays' nests are workcell equipment (trays.py: their places are fixed); the plan chooses the assignment. The steps
are planned in the REVERSE assembly order: when a unit's step is checked, every unit of a LATER step is
still in its planned nest and every unit of an EARLIER step is on its strut at its design pose (an end cap at the free
roll its grasp gives it), so the pick check at the nest and the place check at the slot see the scene the reference
will meet, the modules already on the neighbouring struts included. Candidate assignments in a fixed order, the
nominal one first; the first assignment where every step has a grasp passing both checks is the plan. The nests' roll
offsets (layout `nest_offset_sign` x `nest.roll_offset_deg`, +-11 deg, alternating over the nests) are the layout's;
the plan records them and the resulting tilt of each motor module's pick.

Scopes: `single` (the minimum version: one strut, five steps, eight candidate assignments) and `default` (three
struts, 15 steps; the guide's order strut by strut: both motor modules, the lid, both end caps; struts s1, s2, s3 =
nearest the robot first). The default scope's nominal assignment (layout.SCOPE_LAYOUT["default"]): the left tray feeds
the p ends and the lids, the right tray the n ends, the nest nearest a strut's x feeding that strut (NEST_SLOT); its
alternatives turn which motor / cap nest of a tray feeds which strut (the strut order reversed, then shifted).

    plan = plan_scope(gt_dir, panda_dir, scope="default")   # {"scope", "layout": {...}, "steps": [...], ...}
    plan = plan_single(gt_dir, panda_dir)                   # = plan_scope(..., scope="single")
"""
import itertools
import json
import math
import time

import numpy as np

ORDER = {
    "single": ("s2_motor_p", "s2_motor_n", "s2_lid", "s2_cap_p", "s2_cap_n"),
    "default": tuple("%s_%s" % (s, k) for s in ("s1", "s2", "s3") for k in ("motor_p", "motor_n", "lid", "cap_p", "cap_n")),
}
ORDER_SINGLE = ORDER["single"]

# the default scope's nominal nest -> slot (the left tray's nests from the back of the bench forwards feed s1, s2, s3;
# the right tray is turned half a turn: its nest 1 is the front one)
NEST_SLOT_DEFAULT = {"l_motor1": "s1_motor_p", "l_motor2": "s2_motor_p", "l_motor3": "s3_motor_p",
                     "r_motor3": "s1_motor_n", "r_motor2": "s2_motor_n", "r_motor1": "s3_motor_n",
                     "l_cap1": "s1_cap_p", "l_cap2": "s2_cap_p", "l_cap3": "s3_cap_p",
                     "r_cap3": "s1_cap_n", "r_cap2": "s2_cap_n", "r_cap1": "s3_cap_n",
                     "l_lid1": "s1_lid", "l_lid2": "s2_lid", "l_lid3": "s3_lid"}


def _assignments_single():
    """[(plan_nests, nest_end, nest_slot)] in the fixed preference order (the nominal first)"""
    out = []
    for motors, ends, caps in itertools.product((("M1_1", "M1_2"), ("M1_2", "M1_1")), (("p", "n"), ("n", "p")), (("M3_1", "M3_2"), ("M3_2", "M3_1"))):
        plan_nests = {motors[0]: "motor_a", motors[1]: "motor_b", "M2_1": "lid", caps[0]: "cap_a", caps[1]: "cap_b"}
        nest_end = {"motor_a": ends[0], "motor_b": ends[1]}
        nest_slot = {"motor_a": "s2_motor_%s" % ends[0], "motor_b": "s2_motor_%s" % ends[1], "lid": "s2_lid", "cap_a": "s2_cap_p", "cap_b": "s2_cap_n"}
        out.append((plan_nests, nest_end, nest_slot))
    return out


def _assignments_default():
    """the nominal assignment, then the struts fed by each tray's motor and cap nests reordered (reversed, shifted);
    the units stay with their slots (strut s_k's motor modules M1_{2k-1} / M1_{2k}, its caps of one colour, M3_{2k-1}
    at p, M3_{2k} at n, its lid M2_k)"""
    from .layout import SCOPE_LAYOUT
    base = SCOPE_LAYOUT["default"]
    slot_unit = {}
    for nest, sid in NEST_SLOT_DEFAULT.items():
        u = [k for k, v in base["plan_nests"].items() if v == nest][0]
        slot_unit[sid] = u
    out = []
    for perm in ((0, 1, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1)):
        nest_slot = {}
        for nest, sid in NEST_SLOT_DEFAULT.items():
            k = int(sid[1]) - 1
            nest_slot[nest] = "s%d%s" % (perm[k] + 1, sid[2:]) if nest[2:].startswith(("motor", "cap")) else sid
        plan_nests = {slot_unit[sid]: nest for nest, sid in nest_slot.items()}
        out.append((plan_nests, dict(base["nest_end"]), nest_slot))
    return out


def _slot_unit(plan_nests, nest_slot):
    """{slot: unit} of an assignment"""
    inv = {v: k for k, v in plan_nests.items()}
    return {sid: inv[nest] for nest, sid in nest_slot.items()}


def design_pose(asm, sid, G=None):
    """a slot's unit at home (an end cap's free roll from its grasp, or 0 without one)"""
    x = asm.cell.slot(sid)
    if x["kind"] == "rod_sleeve":
        roll = 0.0
        if x["tol_roll"] is None and G is not None:
            roll = asm.free_roll(sid, G)
        return asm.cell.sleeve_pose(sid, x["home"], roll)
    return asm.cell.design_world(sid)


def _tilt_deg(T_tcp):
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(T_tcp[2, 2])))))


def tilt_order(asm, sid, u, cands, T_nest):
    """the candidates ordered by how far the hand leans off straight down where it matters (stable within 1 deg bins):
    a motor module at its design pose (picked tilted by its nest's offset, level at home), an end cap at
    its nest (its turn on the rod is free: the hand is level at home by construction), the lid at its nest"""
    t = asm.kit.units[u]["type"]
    out = []
    for i, (G, w, desc) in enumerate(cands):
        if t == "M1":
            tilt = _tilt_deg(asm.cell.sleeve_pose(sid, asm.cell.slot(sid)["home"], 0.0) @ G)
        else:
            tilt = _tilt_deg(T_nest @ G)
        out.append((int(tilt), i, (G, w, desc)))
    out.sort(key=lambda o: (o[0], o[1]))
    return [o[2] for o in out]


def plan_scope(gt_dir, panda_dir, scope="default", log=print, config=None, max_grasps=None, assignments=None):
    from .layout import SCOPE_LAYOUT
    from .reference import Assembler
    from .server import StrutInfoServer
    t0 = time.time()
    tried = []
    order = ORDER[scope]
    cands_all = _assignments_single() if scope == "single" else _assignments_default()
    if assignments is not None:
        cands_all = [cands_all[k] for k in assignments]
    for k, (plan_nests, nest_end, nest_slot) in enumerate(cands_all):
        lay = dict(SCOPE_LAYOUT[scope], plan_nests=plan_nests, nest_end=nest_end)
        cfg = dict(config or {}, scope=scope)
        srv = StrutInfoServer(gt_dir, panda_dir, plan={"layout": lay}, config=cfg, privileged=True, render=False, log=lambda s: None)
        srv.reset(0)
        cell = srv.cell
        asm = Assembler(cell, log=log, monitor=False)
        su = _slot_unit(plan_nests, nest_slot)
        steps = [(sid, su[sid]) for sid in order]
        found, why = {}, {}
        for i in range(len(steps) - 1, -1, -1):
            sid, u = steps[i]
            poses = {}
            for j in range(i):
                sj, uj = steps[j]
                Gj = (found.get(uj) or {}).get("G")
                poses[uj] = design_pose(asm, sj, Gj)
            T_nest = cell.body_T(u)
            cands = tilt_order(asm, sid, u, asm._own_candidates(u, sid), T_nest)
            if max_grasps:
                cands = cands[:max_grasps]
            rej = {}
            t_step = time.time()
            for G, w, desc in cands:
                q_pick, q_pg, r1 = asm.pick_checks(u, T_nest, G, w, poses=poses)
                if q_pick is None:
                    key = r1.split(" on ")[0].split(" while ")[0]
                    rej[key] = rej.get(key, 0) + 1
                    continue
                # the place check in the scene of step i: the earlier units at their design, this unit moving
                q_pre, hold, r2 = asm.place_checks(sid, u, G, w, poses=poses, q_seed=q_pick)
                if q_pre is None:
                    key = "place: " + str(hold).split(" at ")[0]
                    rej[key] = rej.get(key, 0) + 1
                    continue
                T_pick = T_nest @ G
                found[u] = {"G": G, "w": w, "desc": desc, "q_pick": q_pick, "q_pg": q_pg, "q_pre": q_pre, "hold": hold,
                            "pick_tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, -float(T_pick[2, 2]))))),
                            "rejected_before": int(sum(rej.values())), "wall_s": round(time.time() - t_step, 1)}
                break
            if u not in found:
                why[sid] = rej
                log("  bench plan %d: no grasp for %s at %s (%s)" % (k, u, sid, sorted(rej.items(), key=lambda kv: -kv[1])[:5]))
                break
            log("  bench plan %d: %-11s %-5s nest %-9s grasp %s, %.1f mm wide, pick tilt %.1f deg (%d rejected first)" % (
                k, sid, u, plan_nests[u], found[u]["desc"], found[u]["w"] * 1e3, found[u]["pick_tilt_deg"], found[u]["rejected_before"]))
        tried.append({"assignment": k, "plan_nests": plan_nests, "nest_end": nest_end, "nest_slot": nest_slot, "ok": len(found) == len(steps), "why": why})
        if len(found) != len(steps):
            continue
        L = srv.layout
        out = {
            "scope": scope, "source": "bench_plan.plan_scope (reverse assembly order)",
            "layout": {"plan_nests": plan_nests, "nest_end": nest_end, "nest_slot": nest_slot, "nest_offset_sign": dict(srv.layout_cfg["nest_offset_sign"]),
                       "nest": {"roll_offset_deg": float(srv.layout_cfg["nest"]["roll_offset_deg"]), "roll_jitter_deg": float(srv.layout_cfg["nest"]["roll_jitter_deg"])}},
            "nominal": {"trays": L["trays"], "rolls_deg": L["rolls_deg"], "design_roll_deg_by_end": L["design_roll_deg_by_end"],
                        "design_roll_deg_by_tray_end": L.get("design_roll_deg_by_tray_end"), "lid_flip": L["lid_flip"],
                        "nest_roll_offset_deg": {n: float(srv.layout_cfg["nest_offset_sign"][n]) * float(srv.layout_cfg["nest"]["roll_offset_deg"])
                                                 for n in srv.layout_cfg["nest_offset_sign"]}},
            "steps": [], "assignments_tried": tried, "wall_s": round(time.time() - t0, 1),
        }
        if scope == "single":
            out["layout"].pop("nest_slot")
        for sid, u in steps:
            f = found[u]
            out["steps"].append({"slot": sid, "unit": u, "nest": plan_nests[u], "build": cell.kit.units[u]["build"],
                                 "grasp": {"G": np.round(f["G"], 7).tolist(), "w": round(float(f["w"]), 6), "desc": f["desc"]},
                                 "pick_tilt_deg": round(f["pick_tilt_deg"], 3), "hold_mm": round(f["hold"] * 1e3, 1),
                                 "rejected_before": f["rejected_before"],
                                 "q_pick": np.round(f["q_pick"], 5).tolist(), "q_pre": np.round(f["q_pre"], 5).tolist()})
        return out
    return {"scope": scope, "source": "bench_plan.plan_scope", "steps": [], "assignments_tried": tried, "error": "no feasible assignment",
            "wall_s": round(time.time() - t0, 1)}


def plan_single(gt_dir, panda_dir, log=print, config=None, max_grasps=None):
    return plan_scope(gt_dir, panda_dir, scope="single", log=log, config=config, max_grasps=max_grasps)


def nest_slot_of(plan):
    """{nest: slot} of a plan (its layout's nest_slot, else from its steps)"""
    ns = dict(((plan or {}).get("layout") or {}).get("nest_slot") or {})
    for st in (plan or {}).get("steps", []):
        ns.setdefault(st["nest"], st["slot"])
    return ns


def save(plan, path):
    json.dump(plan, open(path, "w"), indent=1)
