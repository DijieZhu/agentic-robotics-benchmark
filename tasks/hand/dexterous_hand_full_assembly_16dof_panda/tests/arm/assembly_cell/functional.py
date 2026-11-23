"""Functional-articulation check (CHAIN_PLAN 4.3 `functional_articulation`).

After the episode the grader opens the gripper and retracts the Panda to its home configuration
(its own controller), powers the seated servos of every COMPLETED joint (kp 15 N m/rad, forcerange
+-stall; actuator group 1 enabled), and drives `n_configs` hidden operating-range configurations on
the completed joint prefix (3 s interpolation + 1 s hold each). A configuration passes iff the last
completed link's pose in the base frame is within pos_tol / ang_tol of the design FK
(`arm.chain_model.Chain.fk` on the design assembly.json), no contact > contact_N occurs between
non-adjacent assembly bodies (or between an assembly body other than the base and the fixture /
table), and no servo stalls (|force| at the limit for > stall_s). Score =
(completed joints / in-scope joints) x (passed / n_configs).
"""

import math

import numpy as np

from .layout import quat_from_rotmat
from .predicates import body_pose, rotation_angle

__all__ = ["run_functional", "DEFAULT_FUNCTIONAL_CONFIG"]

DEFAULT_FUNCTIONAL_CONFIG = {"n_configs": 8, "pos_tol_mm": 10.0, "ang_tol_deg": 5.0, "contact_N": 5.0, "stall_s": 0.5,
                             "retreat_s": 3.0, "move_s": 3.0, "hold_s": 1.0, "stall_fraction": 0.98,
                             # operating range of the design (design_rules.yaml): joint box x TCP inside the reach
                             # envelope x every frame above the mounting plane x design penetration-free in this
                             # collision model (the CAD grader tolerates <= 1e-2 volume overlap of adjacent links,
                             # which rigid-body physics turns into a hard stop)
                             "envelope": {"radius_mm": [120.0, 300.0], "z_mm": [20.0, 180.0], "azimuth_deg": [-90.0, 90.0]},
                             "design_penetration_mm": 0.2, "max_candidates": 4000}


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def tcp_in_envelope(p_mm, env):
    """The CAD grader's rule: cylindrical bands about the yaw axis, azimuth measured from +y towards -x."""
    r = math.hypot(p_mm[0], p_mm[1])
    if not (env["radius_mm"][0] <= r <= env["radius_mm"][1]):
        return False
    if not (env["z_mm"][0] <= p_mm[2] <= env["z_mm"][1]):
        return False
    az = math.degrees(math.atan2(-p_mm[0], p_mm[1]))
    return env["azimuth_deg"][0] <= az <= env["azimuth_deg"][1]


def chain_adjacency(chain, tracker, m):
    """Body pairs allowed to touch: parent / child links, a link and the servos it carries or sits on,
    a servo and its horn (CHAIN_PLAN 4.3 (b): only NON-adjacent contact fails)."""
    adj = set()
    names = set(n for n in [tracker.meta["root"]] + list(tracker.meta["loose"]))

    def add(a, b):
        if a in names or a.startswith("horn_"):
            if b in names or b.startswith("horn_"):
                try:
                    adj.add(frozenset([m.body(a).id, m.body(b).id]))
                except KeyError:
                    pass
    carriers = {}
    for name, link in chain.links.items():
        for sid, _T in link.occurrences:
            carriers[sid] = name
            add(name, sid)
            add(name, "horn_" + sid)
    for j in chain.joints:
        add(j.parent, j.child)
        for sid in j.driven_by:
            add(sid, j.child)
            add(sid, j.parent)
            add("horn_" + sid, j.child)
            add("horn_" + sid, j.parent)
    # servos seated in a pocket may be any servo of the SKU: map design ids to physical servos
    for seat_id, phys in tracker.seat_servo.items():
        design = seat_id.replace("seat_", "")
        for pair in list(adj):
            ids = list(pair)
            for a in ids:
                if m.body(a).name == design:
                    other = ids[0] if ids[1] == a else ids[1]
                    add(phys, m.body(other).name)
                    add("horn_" + phys, m.body(other).name)
        add(phys, "horn_" + phys)
    for s in tracker.meta["servos"]:
        add(s, "horn_" + s)
    return adj


def design_config_ok(server, chain, q_full, cfg, scratch):
    """Operating-range test of one FULL design configuration (mm chain): TCP in envelope, every link
    and servo frame above the mounting plane, and the design placed in the cell's collision model
    (welded assembly at the design FK) shows no penetration deeper than design_penetration_mm."""
    m = server.model
    T = chain.fk(q_full)
    if chain.tool is not None:
        p, _a = chain.tcp(q_full, T)
        if not tcp_in_envelope(p, cfg["envelope"]):
            return False
    for name in chain.order:
        if T[name][2, 3] < -0.5:
            return False
        for sid, Tocc in chain.links[name].occurrences:
            if (T[name] @ Tocc)[2, 3] < -0.5:
                return False
    # place the built parts at the design FK (base frame -> world) in the scratch data
    T_base = body_pose(server.data, m.body(server.meta["root"]).id)
    scratch.qpos[:] = server.data.qpos
    placed = set([m.body(server.meta["root"]).id])
    for name in chain.order:
        if name == server.meta["root"] or name not in server.part_qadr:
            continue
        Tw = T[name].copy()
        Tw[:3, 3] *= 1e-3
        Tw = T_base @ Tw
        adr = server.part_qadr[name]
        scratch.qpos[adr:adr + 3] = Tw[:3, 3]
        scratch.qpos[adr + 3:adr + 7] = quat_from_rotmat(Tw[:3, :3])
        placed.add(m.body(name).id)
        for sid, Tocc in chain.links[name].occurrences:
            phys = server.tracker.seat_servo.get("seat_" + sid)
            if phys is None or phys not in server.part_qadr:
                continue
            Ts = T[name] @ Tocc
            Ts = Ts.copy()
            Ts[:3, 3] *= 1e-3
            Ts = T_base @ Ts
            adr = server.part_qadr[phys]
            scratch.qpos[adr:adr + 3] = Ts[:3, 3]
            scratch.qpos[adr + 3:adr + 7] = quat_from_rotmat(Ts[:3, :3])
            placed.add(m.body(phys).id)
            placed.add(m.body("horn_" + phys).id)
    server.mujoco.mj_forward(m, scratch)
    thr = -cfg["design_penetration_mm"] * 1e-3
    for i in range(scratch.ncon):
        c = scratch.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 in placed and b2 in placed and c.dist < thr:
            n1, n2 = m.body(b1).name, m.body(b2).name
            if n1.startswith("horn_") or n2.startswith("horn_"):
                continue
            return False
    return True


def sample_operating_configs(server, chain, rng, n, cfg, log=None):
    """Full design configurations of the operating range, drawn uniformly in the joint box (rejection)."""
    import mujoco
    scratch = mujoco.MjData(server.model)
    lo, hi = chain.lower, chain.upper
    out = []
    tried = 0
    while len(out) < n and tried < int(cfg["max_candidates"]):
        tried += 1
        q = lo + rng.random(len(lo)) * (hi - lo)
        if design_config_ok(server, chain, q, cfg, scratch):
            out.append(q)
    if log:
        log("[functional] %d operating-range configurations from %d candidates" % (len(out), tried))
    return out


def completed_prefix(chain, tracker):
    """Joints (in chain order) whose horn mate is achieved, stopping at the first incomplete one."""
    done = []
    for j in chain.joints:
        mid = "horn_" + j.name
        st = tracker.state.get(mid)
        if st is None or not st["achieved"]:
            break
        done.append(j)
    return done


def run_functional(server, rng, config=None, log=None):
    """Run the check on `server` (episode over, welds active). Returns the result dict."""
    cfg = dict(DEFAULT_FUNCTIONAL_CONFIG)
    cfg.update(config or {})
    m, d = server.model, server.data
    chain = server.chain
    tracker = server.tracker
    mujoco = server.mujoco
    in_scope_joints = [j for j in chain.joints if ("horn_" + j.name) in tracker.by_id]
    done = completed_prefix(chain, tracker)
    res = {"in_scope_joints": [j.name for j in in_scope_joints], "completed_joints": [j.name for j in done],
           "n_configs": int(cfg["n_configs"]), "passed": 0, "configs": [], "score": 0.0, "retreat_contact_N": 0.0}
    if not in_scope_joints:
        return res
    # 1. retreat: open, home, joint mode, full stiffness
    ctrl = server.ctrl
    ctrl.set_mode("joint")
    ctrl.set_stiffness(1.0)
    ctrl.command_width(ctrl.max_width)
    ctrl.command_joint(server.meta["panda_home"])
    max_c = 0.0
    for _ in range(int(cfg["retreat_s"] / server.tick_s)):
        server.physics_tick(track=False)
        max_c = max(max_c, server.max_panda_contact_force())
    res["retreat_contact_N"] = float(max_c)
    if not done:
        return res
    # 2. power the servos of the completed joints
    m.opt.disableactuator = 0
    drives = []       # (act id, sign, theta_snap, psi_h, joint index)
    for k, j in enumerate(done):
        st = tracker.state["horn_" + j.name]
        for phys, dr in (st.get("servo_drive") or {}).items():
            a = m.actuator("act_horn_" + phys).id
            drives.append((a, dr["sign"], dr["theta_snap"], dr["psi_h"], k, phys))
    # the as-built joint angles now (from the hinge angles of the first servo of each joint)
    def q_now():
        q = np.zeros(len(done))
        seen = set()
        for a, sign, th_snap, psi_h, k, phys in drives:
            if k in seen:
                continue
            seen.add(k)
            th = float(d.qpos[m.jnt_qposadr[m.joint("hj_" + phys).id]])
            q[k] = sign * (psi_h + th - th_snap)
        return q

    def set_targets(q):
        for a, sign, th_snap, psi_h, k, phys in drives:
            d.ctrl[a] = th_snap + sign * q[k] - psi_h

    # the assembly = the base + every welded part (+ their horns); loose parts count as "in the way"
    body_of = {}
    assembly_bodies = set([m.body(server.meta["root"]).id])
    welded_parts = set()
    for (b1, b2), eqname in server.meta["welds"].items():
        if d.eq_active[m.equality(eqname).id]:
            for b in (b1, b2):
                welded_parts.add(b[5:] if b.startswith("horn_") else b)
    for name in welded_parts:
        assembly_bodies.add(m.body(name).id)
        if name in server.meta["servos"]:
            assembly_bodies.add(m.body("horn_" + name).id)
    loose_bodies = set(m.body(n).id for n in server.meta["loose"] if n not in welded_parts)
    for n in server.meta["loose"]:
        if n in server.meta["servos"] and n not in welded_parts:
            loose_bodies.add(m.body("horn_" + n).id)
    panda_bodies = set(b for b in range(m.nbody) if m.body(b).name.startswith("panda/"))
    adjacent = chain_adjacency(chain, tracker, m)
    for (b1, b2), eqname in server.meta["welds"].items():
        eq = m.equality(eqname).id
        if d.eq_active[eq]:
            adjacent.add(frozenset([m.body(b1).id, m.body(b2).id]))
    base_id = m.body(server.meta["root"]).id
    static_env = set(server.static_env_bodies)
    n_stall_ticks = int(cfg["stall_s"] / server.tick_s)
    forcerange = {a: float(m.actuator_forcerange[a][1]) for a, *_ in drives}
    T_base = body_pose(d, base_id)
    last_link = done[-1].child
    last_id = m.body(last_link).id
    q_lo = np.array([j.lower for j in done])
    q_hi = np.array([j.upper for j in done])
    f6 = np.zeros(6)
    n_move = int(cfg["move_s"] / server.tick_s)
    n_hold = int(cfg["hold_s"] / server.tick_s)
    q_cur = q_now()
    passed = 0
    samples = sample_operating_configs(server, chain, rng, int(cfg["n_configs"]), cfg, log=log)
    res["n_sampled"] = len(samples)
    for c in range(int(cfg["n_configs"])):
        if c >= len(samples):
            res["configs"].append({"q": None, "pass": False, "reason": "no operating-range configuration found"})
            continue
        q_t = np.asarray(samples[c][:len(done)], float)
        bad_contact = 0.0
        bad_pairs = {}
        stall_ticks = {a: 0 for a in forcerange}
        stalled = False
        q_start = q_cur.copy()
        for t in range(n_move + n_hold):
            alpha = min(1.0, (t + 1) / float(n_move))
            set_targets(q_start + alpha * (q_t - q_start))
            server.physics_tick(track=False)
            for i in range(d.ncon):
                con = d.contact[i]
                b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
                in1, in2 = b1 in assembly_bodies, b2 in assembly_bodies
                if not (in1 or in2):
                    continue
                other = b2 if in1 else b1
                mine = b1 if in1 else b2
                if other in panda_bodies:
                    continue
                mujoco.mj_contactForce(m, d, i, f6)
                fn = abs(float(f6[0]))
                if fn <= cfg["contact_N"]:
                    continue
                bad = False
                if in1 and in2:
                    bad = frozenset([b1, b2]) not in adjacent
                elif other in loose_bodies:
                    bad = True                                   # a loose part in the way of the moving arm
                elif other in static_env and mine != base_id:
                    bad = True
                if bad:
                    bad_contact = max(bad_contact, fn)
                    key = "%s|%s" % (m.body(mine).name, m.body(other).name)
                    bad_pairs[key] = max(bad_pairs.get(key, 0.0), fn)
            for a in forcerange:
                if abs(float(d.actuator_force[a])) >= cfg["stall_fraction"] * forcerange[a]:
                    stall_ticks[a] += 1
                    if stall_ticks[a] >= n_stall_ticks:
                        stalled = True
                else:
                    stall_ticks[a] = 0
        q_cur = q_t.copy()
        # compare the last completed link with the design FK in the base frame
        q_full = np.zeros(len(chain.joints))
        q_full[:len(done)] = q_t
        T_design = chain.fk(q_full)[last_link].copy()
        T_design[:3, 3] *= 1e-3
        T_built = _inv(T_base) @ body_pose(d, last_id)
        pos_err = float(np.linalg.norm(T_built[:3, 3] - T_design[:3, 3]))
        ang_err = rotation_angle(T_built[:3, :3] @ T_design[:3, :3].T)
        q_meas = q_now()
        ok = (pos_err <= cfg["pos_tol_mm"] * 1e-3 and ang_err <= math.radians(cfg["ang_tol_deg"]) and bad_contact == 0.0 and not stalled)
        passed += int(ok)
        res["configs"].append({"q": [float(v) for v in q_t], "q_measured": [float(v) for v in q_meas], "pos_err_mm": pos_err * 1000.0,
                               "ang_err_deg": math.degrees(ang_err), "bad_contact_N": bad_contact, "bad_contact_pairs": bad_pairs,
                               "stalled": stalled, "pass": bool(ok)})
        if log:
            log("[functional] config %d q=%s pos_err %.2f mm ang_err %.2f deg contact %.1f N stall %s -> %s" % (
                c, np.round(q_t, 3).tolist(), pos_err * 1000, math.degrees(ang_err), bad_contact, stalled, "pass" if ok else "FAIL"))
    res["passed"] = passed
    res["score"] = (len(done) / float(len(in_scope_joints))) * (passed / float(cfg["n_configs"]))
    m.opt.disableactuator = 1 << 1
    return res
