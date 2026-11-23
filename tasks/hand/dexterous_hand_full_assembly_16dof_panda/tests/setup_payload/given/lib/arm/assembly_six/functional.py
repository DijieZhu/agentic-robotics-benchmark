"""Functional-articulation check of the six-servo arm with a PRE-ASSEMBLED joint prefix (CHAIN_PLAN 4.3
`functional_articulation`; shared `assembly_cell/functional.py` generalised).

After the episode the grader retracts the Panda, powers the horn actuators of the completed joints (kp 15 N m/rad,
forcerange +-stall) and drives `n_configs` hidden configurations of the DRIVEN joints (3 s interpolation + 1 s hold
each). Joint roles, in chain order (`joint_roles`):
    fixed   the joints of the fixed pre-assembled group (the proximal half on the fixture): their design angle is the
            group's q; they never move
    done    an in-scope joint whose horn mate is achieved (driven by its seated servo's horn actuator)
    open    an in-scope joint whose horn mate is not achieved (stops the completed run)
    rigid   the joints inside a free pre-assembled compound (welded by design at the group's q; not driven)
A configuration passes iff the last DONE joint's child link, expressed in the base frame, is within pos_tol / ang_tol of
the design FK (`Chain.fk` at q_full = fixed / target / rigid angles), no contact > contact_N occurs between non-adjacent
assembly bodies (or between an assembly body other than the base and the fixture / table), and no servo stalls.
Score = (done joints / in-scope joints) x (passed / n_configs). Operating range: driven angles uniform in their ranges,
rejected when the design placed at that configuration has any frame below the base plane or a penetration deeper than
`design_penetration_mm` in the cell's collision model (incl. the fixture and the table).
"""

import math

import numpy as np

from ..assembly_cell.functional import DEFAULT_FUNCTIONAL_CONFIG, _inv
from ..assembly_cell.predicates import body_pose, rotation_angle
from .layout import quat_from_rotmat

__all__ = ["run_functional", "joint_roles", "DEFAULT_FUNCTIONAL_CONFIG"]


def joint_roles(chain, tracker, meta):
    """[(joint, role, q_fixed_or_None)] in chain order (roles: fixed / done / open / rigid / absent)."""
    gp = meta.get("group_poses", {})
    fixed_q, rigid_q = {}, {}
    for g in gp.values():
        members = set(g["parts"]) | {g["root"]}
        for j in chain.joints:
            if j.parent in members and j.child in members:
                (fixed_q if g["fixed"] else rigid_q)[j.name] = float(g["q_full"][j.name])
    out = []
    stop = False
    for j in chain.joints:
        mid = "horn_" + j.name
        if j.name in fixed_q:
            out.append((j, "fixed", fixed_q[j.name]))
        elif mid in tracker.by_id:
            if tracker.state[mid]["achieved"] and not stop:
                out.append((j, "done", None))
            else:
                stop = True
                out.append((j, "open", None))
        elif j.name in rigid_q:
            out.append((j, "rigid", rigid_q[j.name]))
        else:
            out.append((j, "absent", None))
    return out


def _assembly_and_adjacency(server, chain, tracker, m, d):
    """(assembly body ids, adjacent pairs (frozensets of body ids), loose body ids) for the contact rule."""
    names = set(server.meta["parts"])
    def bid(n):
        try:
            return m.body(n).id
        except KeyError:
            return None
    root_id = m.body(server.meta["root"]).id
    assembly = set([root_id])
    # pre-assembled members of the FIXED groups + every welded part (+ its horn) + members of a welded compound root
    fixed_members = set()
    compound_members = {}
    for g in server.meta["group_poses"].values():
        if g["fixed"]:
            fixed_members |= set(g["parts"])
        else:
            compound_members[g["root"]] = list(g["parts"])
    for n in fixed_members:
        b = bid(n)
        if b is not None:
            assembly.add(b)
    welded_parts = set()
    for (b1, b2), eqname in server.meta["welds"].items():
        if d.eq_active[m.equality(eqname).id]:
            for b in (b1, b2):
                welded_parts.add(b[5:] if b.startswith("horn_") else b)
    for name in list(welded_parts):
        for mem in compound_members.get(name, []):
            welded_parts.add(mem)
    for name in welded_parts:
        b = bid(name)
        if b is not None:
            assembly.add(b)
        if name in server.meta["servos"]:
            assembly.add(m.body("horn_" + name).id)
    loose_bodies = set()
    for n in server.meta["loose"]:
        if n in welded_parts:
            continue
        loose_bodies.add(m.body(n).id)
        for mem in compound_members.get(n, []):
            loose_bodies.add(m.body(mem).id)
        if n in server.meta["servos"]:
            loose_bodies.add(m.body("horn_" + n).id)
    # adjacency from the design: parent/child links, a link and the parts attached to it (servos, brackets), a servo and its horn,
    # the servo and the horn-side link, members of one group with each other
    adj = set()

    def add(a, b):
        ia, ib = bid(a), bid(b)
        if ia is not None and ib is not None and ia != ib:
            adj.add(frozenset([ia, ib]))
    design = server.meta["design"]
    for l in design["links"]:
        for ap in l.get("attached_parts", []):
            add(l["name"], ap["name"])
            for occ in l.get("servo_occurrences", []):
                add(ap["name"], occ["servo_id"])
        for occ in l.get("servo_occurrences", []):
            add(l["name"], occ["servo_id"])
            add(l["name"], "horn_" + occ["servo_id"])
    for j in chain.joints:
        add(j.parent, j.child)
        for sid in j.driven_by:
            add(sid, j.child)
            add("horn_" + sid, j.child)
            add(sid, j.parent)
            add("horn_" + sid, j.parent)
            phys = tracker.seat_servo.get("seat_" + sid)
            if phys:
                add(phys, j.child)
                add("horn_" + phys, j.child)
                add(phys, j.parent)
                add(phys, "horn_" + phys)
    for s in server.meta["servos"]:
        add(s, "horn_" + s)
    for g in server.meta["group_poses"].values():
        mem = list(g["parts"]) + [g["root"]]
        for a in mem:
            for b in mem:
                if a < b:
                    add(a, b)
    return assembly, adj, loose_bodies


def design_config_ok(server, chain, q_full, cfg, scratch, movable):
    """Operating-range test of one design configuration: every link / servo frame above the base plane and the design
    placed in the cell's collision model shows no penetration deeper than design_penetration_mm (movable parts only:
    the fixed proximal is where it is)."""
    m = server.model
    T = chain.fk(q_full)
    for name in chain.order:
        if T[name][2, 3] < -0.5:
            return False
        for sid, Tocc in chain.links[name].occurrences:
            if (T[name] @ Tocc)[2, 3] < -0.5:
                return False
    T_base = body_pose(server.data, m.body(server.meta["root"]).id)
    scratch.qpos[:] = server.data.qpos
    placed = set([m.body(server.meta["root"]).id])
    for name in movable:
        if name not in server.part_qadr:
            continue
        Tw = T[name].copy() if name in T else None
        if Tw is None:
            # a servo: carrier link FK x occurrence; a bracket: link x attachment
            carrier, Tocc = chain.occurrence(name)
            if carrier is None:
                continue
            Tw = T[carrier] @ Tocc
        Tw = Tw.copy()
        Tw[:3, 3] *= 1e-3
        Tw = T_base @ Tw
        adr = server.part_qadr[name]
        scratch.qpos[adr:adr + 3] = Tw[:3, 3]
        scratch.qpos[adr + 3:adr + 7] = quat_from_rotmat(Tw[:3, :3])
        placed.add(m.body(name).id)
        for mem in server.meta["group_poses"].get({g["root"]: k for k, g in server.meta["group_poses"].items()}.get(name, ""), {}).get("parts", []):
            placed.add(m.body(mem).id)
    server.mujoco.mj_forward(m, scratch)
    thr = -cfg["design_penetration_mm"] * 1e-3
    panda = set(b for b in range(m.nbody) if m.body(b).name.startswith("panda/"))
    for i in range(scratch.ncon):
        c = scratch.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in placed or b2 in placed) and c.dist < thr:
            if b1 in panda or b2 in panda:
                continue
            n1, n2 = m.body(b1).name, m.body(b2).name
            if n1.startswith("horn_") or n2.startswith("horn_"):
                continue
            return False
    return True


def run_functional(server, rng, config=None, log=None):
    cfg = dict(DEFAULT_FUNCTIONAL_CONFIG)
    cfg.update(config or {})
    m, d = server.model, server.data
    chain, tracker, mujoco = server.chain, server.tracker, server.mujoco
    roles = joint_roles(chain, tracker, server.meta)
    in_scope = [j for j, r, _ in roles if r in ("done", "open")]
    done = [j for j, r, _ in roles if r == "done"]
    res = {"in_scope_joints": [j.name for j in in_scope], "completed_joints": [j.name for j in done],
           "fixed_joints": {j.name: q for j, r, q in roles if r == "fixed"}, "rigid_joints": {j.name: q for j, r, q in roles if r == "rigid"},
           "n_configs": int(cfg["n_configs"]), "passed": 0, "configs": [], "score": 0.0, "retreat_contact_N": 0.0}
    if not in_scope:
        return res
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
    m.opt.disableactuator = 0
    drives = []
    for k, j in enumerate(done):
        st = tracker.state["horn_" + j.name]
        for phys, dr in (st.get("servo_drive") or {}).items():
            a = m.actuator("act_horn_" + phys).id
            drives.append((a, dr["sign"], dr["theta_snap"], dr["psi_h"], k, phys))

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

    def q_full_of(q_done):
        q = np.zeros(len(chain.joints))
        for i, (j, r, qf) in enumerate(roles):
            if r in ("fixed", "rigid"):
                q[i] = qf
        for k, j in enumerate(done):
            q[chain.joints.index(j)] = q_done[k]
        return q

    assembly_bodies, adjacent, loose_bodies = _assembly_and_adjacency(server, chain, tracker, m, d)
    for (b1, b2), eqname in server.meta["welds"].items():
        if d.eq_active[m.equality(eqname).id]:
            adjacent.add(frozenset([m.body(b1).id, m.body(b2).id]))
    panda_bodies = set(b for b in range(m.nbody) if m.body(b).name.startswith("panda/"))
    base_id = m.body(server.meta["root"]).id
    static_env = set(server.static_env_bodies)
    n_stall_ticks = int(cfg["stall_s"] / server.tick_s)
    forcerange = {a: float(m.actuator_forcerange[a][1]) for a, *_ in drives}
    T_base = body_pose(d, base_id)
    last_link = done[-1].child
    last_id = m.body(last_link).id
    f6 = np.zeros(6)
    n_move = int(cfg["move_s"] / server.tick_s)
    n_hold = int(cfg["hold_s"] / server.tick_s)
    # operating-range configurations of the DRIVEN joints
    scratch = mujoco.MjData(m)
    movable = [n for n in server.meta["loose"]]
    lo = np.array([j.lower for j in done])
    hi = np.array([j.upper for j in done])
    samples, tried = [], 0
    while len(samples) < int(cfg["n_configs"]) and tried < int(cfg["max_candidates"]):
        tried += 1
        q = lo + rng.random(len(lo)) * (hi - lo)
        if design_config_ok(server, chain, q_full_of(q), cfg, scratch, movable):
            samples.append(q)
    res["n_sampled"] = len(samples)
    if log:
        log("[functional] %d operating-range configurations from %d candidates (driven %s, fixed %s)" % (len(samples), tried, [j.name for j in done], res["fixed_joints"]))
    q_cur = q_now()
    passed = 0
    for c in range(int(cfg["n_configs"])):
        if c >= len(samples):
            res["configs"].append({"q": None, "pass": False, "reason": "no operating-range configuration found"})
            continue
        q_t = np.asarray(samples[c], float)
        bad_contact, bad_pairs = 0.0, {}
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
                    bad = True
                elif other in static_env and mine != base_id and m.body(mine).name not in server.meta.get("preassembled", {}):
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
        T_design = chain.fk(q_full_of(q_t))[last_link].copy()
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
    res["score"] = (len(done) / float(len(in_scope))) * (passed / float(cfg["n_configs"]))
    m.opt.disableactuator = 1 << 1
    return res
