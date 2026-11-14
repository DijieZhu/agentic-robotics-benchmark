"""The bench plan of the reference program: where each unit starts on the bench, in which resting orientation, and the
grasp that works there and at its slot.

`rest_plan(asm, ...)` -> {"rest": {unit: {"pos", "quat", "yaw_deg", "grasp": {"G", "w", "desc"}, "step"}}, "steps",
"source"}. The spots are chosen in the REVERSE order of the steps (`reference.shop_steps`): when a unit's spot is
chosen, every unit still on the bench at its turn (those of later steps) is already where it will be, and the robot is
built up to that step at its design. Per unit: its stable rests (`rests.stable_rests` on its collision hull with its
centre of mass) x 4 yaws x the bench spots (a 50 mm grid in the UR's reach, outside the stand's zone, footprints 40 mm
apart), nearest the stand first. A rest standing on an end is never used (`_stands`: the unit's height at that rest is
its largest dimension, the footprint's sides taken along its principal axes, as `unit_rests` turns it): every unit
lies flat (owner, 2026-11-09: a shoulder module standing upright on the bench looked implausible in the demo video; the
spot-major search had taken S3's third rest, 192 mm tall on a 47 x 44 mm end, at a spot near the stand). Of the kit's
rests this removes only each shoulder's third one (on its end; the two lying rests stay); the lower legs' key-up rule
picks among the remaining rests as before. A shoulder / support is accepted at the first spot where one
of its grasps passes the pick checks there and the place checks at its slot (cached per step and grasp: the slot does
not see the bench). A leg step places the lower leg (key up: its most likely rest) and the upper leg together: the
lower leg's spot where one of the upper leg's grasps passes the HFE place check (with the lower leg riding on it) and
the knee's place check (the upper leg lowered onto the lower leg there), then the upper leg's spot where that grasp
passes the pick checks.
"""
import math
import time

import numpy as np

from .coupling import _inv, quat_wxyz

GAP_M = 0.04
GRID_M = 0.05
REACH_M = (0.38, 0.92)
# around the stand, in the ROBOT frame (the body's ends +-0.22 m plus the shoulders' and supports' 60 mm pre-insertion
# and a hand's width; the legs +-0.17 m plus the HFE pre-insertion and a hand's width)
STAND_ZONE = {"x": 0.38, "y": 0.32}
BENCH_X, BENCH_Y = (-0.50, 1.40), 0.66


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def unit_hull_points(cell, units, T_units):
    """collision vertices of the units' bodies (every body of each unit) placed at T_units {unit: world pose of its root}"""
    m, d = cell.m, cell.d
    pts = []
    for u in units:
        Tu_now = cell.body_T(u)
        X = T_units[u] @ _inv(Tu_now)
        for g in cell.geoms_of[u]:
            k = m.geom_dataid[g]
            if k < 0:
                continue
            v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            R = d.geom_xmat[g].reshape(3, 3)
            w = v @ R.T + d.geom_xpos[g]
            pts.append(w @ X[:3, :3].T + X[:3, 3])
    return np.concatenate(pts)


def _stands(P):
    """True when a unit resting with its points at P (bench frame) stands on an end: its height is its largest
    dimension (the footprint's sides taken along its principal axes)"""
    c2 = P[:, :2] - P[:, :2].mean(0)
    _w, v_ = np.linalg.eigh(c2.T @ c2)
    Q = c2 @ v_
    return float(np.ptp(P[:, 2])) >= float(np.ptp(Q, axis=0).max())


def unit_rests(cell, u, group_rel=None, prefer_axis_up=None, n=3):
    """[(R, zmin)] stable rests of the unit (with the units of group_rel {unit: pose rel. to u} riding on it) on a flat
    bench, the unit frame's rotation; the footprint's long side along x"""
    from .rests import stable_rests
    m, d = cell.m, cell.d
    Tu = cell.body_T(u)
    T_units = {u: Tu}
    for g, Trel in (group_rel or {}).items():
        T_units[g] = Tu @ Trel
    V = unit_hull_points(cell, list(T_units), T_units)
    V = V @ _inv(Tu)[:3, :3].T + _inv(Tu)[:3, 3]
    mass, mom = 0.0, np.zeros(3)
    for b in range(m.nbody):
        uu = cell.unit_of_body.get(b)
        if uu in T_units:
            Tb = T_units[uu] @ _inv(cell.body_T(uu)) @ cell.body_T(m.body(b).name)
            c = Tb[:3, :3] @ m.body_ipos[b] + Tb[:3, 3]
            mass += m.body_mass[b]
            mom += m.body_mass[b] * c
    com = _inv(Tu)[:3, :3] @ (mom / mass) + _inv(Tu)[:3, 3]
    rs = stable_rests(V, com, min_margin=0.002)
    rs = [r for r in rs if not _stands(V @ r["R"].T)]               # never on an end (module docstring)
    if prefer_axis_up is not None:
        rs = sorted(rs, key=lambda r: -float((r["R"] @ np.asarray(prefer_axis_up, float))[2]))[:1]
    out = []
    for r in rs[:n]:
        R = r["R"]
        P = V @ R.T
        c2 = P[:, :2] - P[:, :2].mean(0)
        _w, v_ = np.linalg.eigh(c2.T @ c2)
        ang = math.atan2(v_[1, 1], v_[0, 1])
        R = _rotz(-ang) @ R
        P = V @ R.T
        out.append((R, float(P[:, 2].min()), V))
    return out


def spots(T_robot):
    """bench positions (x, y) in reach outside the stand's zone (a box in the robot frame), nearest the stand first"""
    out = []
    Ti = _inv(T_robot)
    for x in np.arange(BENCH_X[0] + 0.1, BENCH_X[1] - 0.1, GRID_M):
        for y in np.arange(-BENCH_Y + 0.06, BENCH_Y - 0.06 + 1e-9, GRID_M):
            r = math.hypot(x, y)
            if not (REACH_M[0] <= r <= REACH_M[1]):
                continue
            pr = Ti[:3, :3] @ np.array([x, y, 0.0]) + Ti[:3, 3]
            if abs(pr[0]) < STAND_ZONE["x"] and abs(pr[1]) < STAND_ZONE["y"]:
                continue
            out.append((float(x), float(y)))
    c = T_robot[:2, 3]
    out.sort(key=lambda p: abs(math.hypot(p[0] - c[0], p[1] - c[1]) - 0.45) + 0.3 * max(0.0, p[0] - c[0]))
    return out


def edge_spots(VL, R0L, key_pt, edge_in_m=0.020, x_range=(-0.40, 0.55)):
    """[(xy, yaw index)] for the lower leg lying key up at a side edge of the bench (y = +-BENCH_Y): its long axis
    across the edge, its knee (key) end `edge_in_m` inside it, so that the upper leg lowered onto it and the hand that
    holds the upper leg's middle are out over the edge (the 2F-85's linkage, +-62-74 mm along a vertical closing
    axis, hangs below the bench top there)"""
    out = []
    for k in range(4):
        R = _rotz(k * math.pi / 2) @ R0L
        P = VL @ R.T
        pk = R @ key_pt
        for side in (1, -1):
            # the key end must be L's extreme towards the edge
            if side * pk[1] < side * (P[:, 1].max() if side > 0 else P[:, 1].min()) - 0.025:
                continue
            y = side * (BENCH_Y + 0.04) - pk[1] - side * edge_in_m       # BENCH_Y + 0.04 = the bench's real edge (0.70 m)
            for x in np.arange(x_range[0], x_range[1] + 1e-9, GRID_M):
                out.append(((float(x - pk[0]), float(y)), k))
    out.sort(key=lambda e: abs(e[0][0] - 0.25))
    return out


def rest_plan(asm, stand_xy=(0.62, 0.0), legs=("FL", "FR", "HL", "HR"), log=print, max_spots=400, max_leg_spots=120,
              max_leg_grasps=12):
    """the bench plan on the Assembler `asm` (its cell built with every unit far away, the UR at its ready posture)"""
    from .reference import shop_steps
    cell, kit = asm.cell, asm.kit
    t0 = time.time()
    steps = shop_steps(kit, legs)
    Tw = cell.design_world()
    far = {}
    for i, u in enumerate(kit["loose"]):
        T = np.eye(4)
        T[:3, 3] = [3.0 + 0.4 * i, 3.0, 0.5]
        far[u] = T
    step_of = {}
    for i, st in enumerate(steps):
        step_of[st["unit"]] = i
        if st["kind"] == "leg":
            step_of[st["part"]] = i
    plan, feet = {}, {}

    def scene(i):
        """{unit: world pose} at step i: the units of earlier steps at their design, those of step i and later at
        their planned spots"""
        P = dict(far)
        for st in steps[:i]:
            P[st["unit"]] = Tw[st["unit"]]
            if st["kind"] == "leg":
                P[st["part"]] = Tw[st["part"]]
        for u, pr in plan.items():
            if step_of[u] >= i:
                P[u] = pr["T"]
        return P

    def footprint(V, R, xy):
        P = V @ R.T
        lo, hi = P.min(0), P.max(0)
        return (xy[0] + lo[0], xy[0] + hi[0], xy[1] + lo[1], xy[1] + hi[1])

    def free_fp(fp, extra=()):
        if fp[0] < BENCH_X[0] or fp[1] > BENCH_X[1] or fp[2] < -BENCH_Y - 0.038 or fp[3] > BENCH_Y + 0.038:
            return False                                           # the bench's edges: x -0.55 / 1.45, y +-0.70
        for f in list(feet.values()) + list(extra):
            if fp[0] < f[1] + GAP_M and f[0] < fp[1] + GAP_M and fp[2] < f[3] + GAP_M and f[2] < fp[3] + GAP_M:
                return False
        return True

    def pose(R, zmin_V, xy):
        P = zmin_V @ R.T
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [xy[0], xy[1], -float(P[:, 2].min()) + 0.002]
        return T

    def ins_of(sid, T_moving_home=None, carrier_moves=False):
        s = cell.slots[sid]
        if s["kind"] == "seat":
            v = cell.T_robot[:3, :3] @ np.array(s.get("insertion_robot_design") or s["insertion_robot"], float)
        else:
            T_c = Tw[s["carrier"]] if T_moving_home is None else T_moving_home @ _inv(Tw[kit["bodies"][s["carrier"]]["unit"]]) @ Tw[s["carrier"]]
            v = (T_c @ np.array(s["frame_in_carrier"]))[:3, 2]
            if carrier_moves:
                v = -v
        return v / np.linalg.norm(v)

    place_cache = {}
    sp = spots(cell.T_robot)[:max_spots]
    for i in range(len(steps) - 1, -1, -1):
        st = steps[i]
        sid, u = st["slot"], st["unit"]
        why = {}

        def bump(k):
            why[k] = why.get(k, 0) + 1
        P_place = scene(i)
        if st["kind"] != "leg":
            P_place.pop(u, None)
            ins = ins_of(sid)
            # the unit's grasps that work at its slot (independent of the bench), then the spots where one picks it
            place_ok = []
            for gi, (G, w, desc) in enumerate(asm.grasps[u]):
                q_pre, hold, r2 = asm.place_checks(sid, u, G, w, Tw[u], ins, poses=P_place)
                if q_pre is None:
                    bump("place: " + str(hold).split(" at ")[0])
                else:
                    place_ok.append((G, w, desc, hold))
            rests_u = unit_rests(cell, u, n=3)
            found = None
            for xy in (sp if place_ok else []):
                for R0, _z, V in rests_u:
                    for k in range(4):
                        R = _rotz(k * math.pi / 2) @ R0
                        cand = [g for g in place_ok if (R @ g[0][:3, 2])[2] <= -0.5]
                        if not cand:
                            continue
                        fp = footprint(V, R, xy)
                        if not free_fp(fp):
                            bump("overlap")
                            continue
                        T_pick = pose(R, V, xy)
                        P_pick = scene(i)
                        P_pick[u] = T_pick
                        for G, w, desc, hold in cand:
                            q_pick, q_pg, r1 = asm.pick_checks(u, T_pick, G, w, poses=P_pick)
                            if q_pick is None:
                                bump(r1.split(" at ")[0])
                                continue
                            found = (T_pick, fp, G, w, desc, hold, k)
                            break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                log("  bench plan: no spot for %s (%s); %s" % (u, sid, sorted(why.items(), key=lambda kv: -kv[1])[:6]))
                continue
            T_pick, fp, G, w, desc, hold, k = found
            feet[u] = fp
            plan[u] = {"T": T_pick, "grasp": {"G": np.round(G, 6).tolist(), "w": round(float(w), 5), "desc": desc}, "step": sid,
                       "yaw_deg": 90 * k, "hold_mm": [round(hold * 1e3, 1)]}
            log("  bench plan: %-3s (%s) at (%.2f, %.2f) yaw %d, grasp %s, hold %.0f mm" % (u, sid, T_pick[0, 3], T_pick[1, 3], 90 * k, desc, hold * 1e3))
            continue
        # ---- a leg: the lower leg key up, the upper leg lowered onto it, then the HFE
        L, knee = st["part"], st["knee"]
        P_place.pop(u, None)
        P_place.pop(L, None)
        c_k = [c for c in cell.slots[knee]["candidates"] if c["unit"] == L][0]
        key_dir = np.array(c_k["frame_in_part"])[:3, 2]
        rests_L = unit_rests(cell, L, prefer_axis_up=key_dir, n=1)
        rests_U = unit_rests(cell, u, n=3)
        L_rel = _inv(Tw[u]) @ Tw[L]
        ins_hfe = ins_of(sid)
        # the upper leg's grasps that pass the HFE (with the lower leg riding on it): independent of the bench
        hfe_ok = []
        for gi, (G, w, desc) in enumerate(asm.grasps[u]):
            q_pre, hold, r2 = asm.place_checks(sid, u, G, w, Tw[u], ins_hfe, poses=P_place, group_rel={L: L_rel})
            if q_pre is not None:
                hfe_ok.append((gi, G, w, desc, hold, q_pre))
            else:
                bump("hfe: " + str(hold).split(" at ")[0])
        found = None
        R0L, _zL, VL = rests_L[0]
        key_pt = np.array(c_k["frame_in_part"])[:3, 3]                   # the key's root on the lower leg (its frame)
        L_spots = edge_spots(VL, R0L, key_pt)
        # the grasps that come from above at one of the upper leg's rests first (the others cannot pick it)
        hfe_ok.sort(key=lambda h: 0 if any((R @ h[1][:3, 2])[2] < -0.5 for R, _z, _V in rests_U) else 1)
        for gi, G, w, desc, hold_h, q_pre_h in hfe_ok[:max_leg_grasps]:
            # 1. the upper leg's spot where this grasp picks it (the bench as at this step, without its lower leg)
            u_spot = None
            for xy in sp:
                for R0, _z, V in rests_U:
                    for k in range(4):
                        R = _rotz(k * math.pi / 2) @ R0
                        if (R @ G[:3, 2])[2] > -0.5:                 # this grasp would not come from above
                            continue
                        fp = footprint(V, R, xy)
                        if not free_fp(fp):
                            continue
                        T_pick = pose(R, V, xy)
                        P_pick = scene(i)
                        P_pick[u] = T_pick
                        q_pick, q_pg, r1 = asm.pick_checks(u, T_pick, G, w, poses=P_pick)
                        if q_pick is None:
                            bump("pick: " + r1.split(" at ")[0])
                            continue
                        u_spot = (T_pick, fp, k)
                        break
                    if u_spot:
                        break
                if u_spot:
                    break
            if u_spot is None:
                continue
            T_pick, fp, k = u_spot
            # 2. the lower leg's spot (key up) where this grasp lowers the upper leg onto it; 3. the pick re-checked
            for xy_L, kL in L_spots[:max_leg_spots]:
                for _once in (0,):
                    RL = _rotz(kL * math.pi / 2) @ R0L
                    fpL = footprint(VL, RL, xy_L)
                    if not free_fp(fpL, extra=(fp,)):
                        bump("overlap")
                        continue
                    T_L = pose(RL, VL, xy_L)
                    T_Uk = T_L @ _inv(Tw[L]) @ Tw[u]                 # the upper leg home on it (its knee at index)
                    ins_k = ins_of(knee, T_moving_home=T_Uk, carrier_moves=True)
                    P_k = scene(i)
                    P_k.pop(u, None)
                    P_k[L] = T_L
                    q_k, hold_k, r_k = asm.place_checks(knee, u, G, w, T_Uk, ins_k, poses=P_k)
                    if q_k is None:
                        bump("knee: " + str(hold_k).split(" at ")[0])
                        continue
                    P_pick = scene(i)
                    P_pick[u] = T_pick
                    P_pick[L] = T_L
                    if asm.pick_checks(u, T_pick, G, w, poses=P_pick)[0] is None:
                        bump("pick with the lower leg near")
                        continue
                    found = (T_L, fpL, kL, T_pick, fp, k, G, w, desc, hold_k, hold_h)
                    break
                if found:
                    break
            if found:
                break
        if not found:
            log("  bench plan: no spots for the leg %s + %s (%s); HFE grasps %d; %s" % (u, L, sid, len(hfe_ok), sorted(why.items(), key=lambda kv: -kv[1])[:6]))
            continue
        T_L, fpL, kL, T_pick, fp, k, G, w, desc, hold_k, hold_h = found
        feet[L], feet[u] = fpL, fp
        plan[L] = {"T": T_L, "step": knee, "yaw_deg": 90 * kL, "grasp": None, "hold_mm": []}
        plan[u] = {"T": T_pick, "grasp": {"G": np.round(G, 6).tolist(), "w": round(float(w), 5), "desc": desc}, "step": sid,
                   "yaw_deg": 90 * k, "hold_mm": [round(hold_k * 1e3, 1), round(hold_h * 1e3, 1)]}
        log("  bench plan: %-3s (%s) key up at (%.2f, %.2f) yaw %d; %-3s at (%.2f, %.2f) yaw %d, grasp %s, hold knee %.0f / HFE %.0f mm (%d HFE grasps)" % (
            L, knee, T_L[0, 3], T_L[1, 3], 90 * kL, u, T_pick[0, 3], T_pick[1, 3], 90 * k, desc, hold_k * 1e3, hold_h * 1e3, len(hfe_ok)))
    rest = {}
    for u, pr in plan.items():
        T = pr["T"]
        rest[u] = {"pos": np.round(T[:3, 3], 5).tolist(), "quat": np.round(quat_wxyz(T[:3, :3]), 7).tolist(),
                   "grasp": pr["grasp"], "step": pr["step"], "yaw_deg": pr["yaw_deg"], "hold_mm": pr["hold_mm"]}
    return {"rest": rest, "steps": steps, "source": "bench_plan.rest_plan", "wall_s": round(time.time() - t0, 1)}
