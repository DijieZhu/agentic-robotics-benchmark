"""The humanoid assembly cell's CONTACT CLUSTERS: a collision pair is handed to MuJoCo's narrow phase only while it
can touch.

New file of this task. The cell's profile put about 80 % of a tick into the narrow
phase and about 90 % of the narrow-phase tests on pairs that are apart: a module resting in its rack has some 210
candidate pairs with the rack's pieces (their boxes overlap), of which a handful touch. This module switches such
pairs off while they PROVABLY cannot touch, so that the contact list at every collision call is the one MuJoCo would
have made with every pair on (the physics of record, the monitor's readings and the coupling's measurements unchanged;
the equivalence runs compare the contact list tick by tick and the state bit for bit).

THE RULE
  Geoms. Every collision geom keeps the cell's masks (scene.BIT_*) except a REDUCIBLE geom (a module's, or a fixture's
  other than the floor and the configuration's never_reduce list: the UR's pedestal) while it is REDUCED: a reduced
  geom collides with the robot (its links and the pads) and the floor only; every other pair it is in is off. Robot
  pairs and floor pairs are NEVER switched off (safety rules R1 and R5 always see them); the base's geoms are never
  reduced (they are partners only).
  Bits: a reduced module geom has contype 64 / conaffinity = its own conaffinity AND the robot's contype (1); a reduced
  fixture geom 128 / the same; the floor's conaffinity gains bit 64 (it touches reduced module geoms, as before).
  `mask_relation_check()` proves, class by class, that the new predicate (t1 & a2) | (t2 & a1) equals the cell's own
  predicate except on pairs that hold a reduced geom and a partner other than the robot and the floor. Body masks
  (MuJoCo's broadphase filter, mjModel.body_contype / body_conaffinity) are kept equal to the OR of their geoms'.
  Slots. Every body with collision geoms is in one SLOT: a static slot per static body (welded to the world: the
  fixtures, the base's pelvis), a mover slot per kinematic tree (a module, the base's moving bodies, a prop); the robot
  and the floor are in none (their pairs are never off).
  The query (at the attach and at a control tick's end when the policy below says so): MuJoCo's own collision on a
  scratch MjData at the present qpos with EVERY mask as the cell's and every geom margin = query_margin_m: the contact
  list is then every candidate pair within that margin with its distance (MuJoCo's filters: masks, the parent-child
  filter, the exclusions, the BVH). For each reducible geom g and slot s: M[g, s] = the least distance from g to a geom
  of s (the query's), or, when s has none within the margin, max(margin, the gap between g's box and s's box); for g's
  own slot, the margin. A reducible geom is REDUCED when every M[g, s] exceeds tolerance_m + hysteresis_m (+ a cost
  term: horizon_ticks x the last tick's displacement of both slots).
  The guard, BEFORE EVERY PHYSICS STEP and before the tick's mj_forward (Cell.tick_physics calls pre_step(); the coupling
  measurements' Runner as copied here too): D[s] = a bound of how far any point of slot s's collision geoms lies from where it was at
  the query, from the two configurations themselves (free joint: |dp| + angle x R; hinge: |dq| x R; R = the largest
  distance of the joint's subtree geoms from its anchor at the query; a second pass adds angle x 2 x the first-order
  bound for the lever arms of the telescoping chain); for every reduced geom g in slot o and every slot s the pair stays
  off only while M[g, s] - D[o] - D[s] > tolerance_m (D of a static slot = 0): the distance now is at least the query's
  minus both displacements; any geom that fails gets its own masks back before the step's collision. The full check
  is skipped when a cheap bound shows it would pass: every slot's D grows by at most Lmax x max |dq| over the movers'
  qpos entries since the last full check (unit quaternions: angle <= 2 pi |dq|_inf), so 2 x that below the slack of
  the last full check (min of M - D[o] - D[s] - tolerance) keeps every pair apart. So at every collision call every
  pair that is off is at a positive distance (the proof needs no speed limit and no switching period: the displacement
  is read from the state itself before every collision call).
  "Everything on": a module scrapped (the monitor's R5) -> the cell's masks for the rest of the episode; a module
  DROPPED (it moved more than drop_m_per_tick in one tick: 5 mm = 0.5 m/s, twice the carry limit) -> the cell's masks
  until every module has been quiet (<= quiet_m_per_tick) for quiet_ticks ticks, then a new query; a qpos written from
  outside the physics (a test, a privileged set) is a displacement like any other; a tick whose physics steps did not
  all pass the guard (a loop that bypassed it) -> everything on and an incident in the record.
  Re-query policy (cost only, never safety): at a tick's end, at least requery_min_ticks after the last query, when a
  geom was switched back on since then or a slot moved more than requery_disp_m.

Two cost settings (both default to the first rule when absent: reduce_base false,
requery_settled_ticks 0; the configuration of record sets reduce_base true and requery_settled_ticks 20):
  reduce_base: the BASE's geoms are reducible too (they were partners only). A reduced base geom has the bits of a reduced
  module geom (contype 64, conaffinity = its own AND the robot's contype): it collides with the robot and the floor only.
  Its margin M[g, s] to a slot s none of whose geoms the cell's own masks let it collide with (the base's own slots:
  base-base pairs are off in the cell, scene.BIT_BASE) is infinite: no displacement brings such a pair into the contact
  list. Every other margin and the guard are as above, so a reduced base geom is off only while proven apart.
  requery_settled_ticks: the re-query policy gains one case: at a tick's end, at least requery_min_ticks after the last
  query, when that query ran while something moved (its horizon term above requery_disp_m) and every mover has since been
  settled for requery_settled_ticks ticks (the horizon term a query would use now at most a tenth of requery_disp_m): one
  query more, which reduces the geoms the moving state kept on. A query never switches off a pair the guard has not
  proved apart (cost only).

States (what the rule gives per state of a module): at rest in its fixture only the
geoms touching or within the tolerance of a rack piece, of each other or of the base take part (and every robot and
floor pair); held and carried: the module's geoms come back on as it moves and are reduced again at the next query
wherever it is far from everything but the pads; inside the capture zone of its interface: the seat's pieces near
the interface take part, the rest of the module is reduced; locked: as at rest (the lock changes no rule).

API: ContactClusters(cell, cfg=None, log=None).attach() (the query, pre_step into cell.pre_step_hooks, tick_end into
cell.hooks after the coupling); .pre_step(), .tick_end(), .query(), .all_on(why, permanent=False), .stats(),
mask_relation_check(cell or model, ...). clusters_config(cfg) reads the configuration's block "contact_clusters".
"""
import math

import numpy as np

__all__ = ["ClustersError", "clusters_config", "ContactClusters", "RED_MOD_BIT", "RED_FIX_BIT", "mask_relation_check"]

RED_MOD_BIT = 64
RED_FIX_BIT = 128
DEFAULTS = {"enabled": True, "query_margin_m": 0.002, "tolerance_m": 1e-5, "hysteresis_m": 2e-5,
            "horizon_ticks": 20, "requery_min_ticks": 10, "requery_min_reenabled": 10, "requery_disp_m": 0.001,
            "drop_m_per_tick": 0.005, "quiet_m_per_tick": 0.0005,
            "quiet_ticks": 10, "never_reduce": ["pedestal"], "reduce_base": False, "requery_settled_ticks": 0}


class ClustersError(RuntimeError):
    pass


def clusters_config(cfg):
    """the configuration's "contact_clusters" block over the defaults (a block that is absent = the defaults with
    enabled False: the cell of record without the block runs without clusters)"""
    blk = (cfg or {}).get("contact_clusters")
    out = dict(DEFAULTS)
    if blk is None:
        out["enabled"] = False
        return out
    for k, v in blk.items():
        if k in ("sources", "about"):
            continue
        if k not in DEFAULTS:
            raise ClustersError("contact_clusters: unknown key %r" % k)
        out[k] = v
    if not (isinstance(out["requery_settled_ticks"], int) and not isinstance(out["requery_settled_ticks"], bool)
            and out["requery_settled_ticks"] >= 0):
        raise ClustersError("contact_clusters.requery_settled_ticks must be a whole number >= 0 (0: off)")
    for k in ("query_margin_m", "tolerance_m", "hysteresis_m", "requery_disp_m", "drop_m_per_tick", "quiet_m_per_tick",
              "horizon_ticks", "requery_min_ticks", "requery_min_reenabled", "quiet_ticks"):
        if not (isinstance(out[k], (int, float)) and not isinstance(out[k], bool) and math.isfinite(out[k]) and out[k] > 0):
            raise ClustersError("contact_clusters.%s must be a positive number, got %r" % (k, out[k]))
    if not out["tolerance_m"] < out["query_margin_m"]:
        raise ClustersError("contact_clusters: tolerance_m must be below query_margin_m")
    if not isinstance(out["reduce_base"], bool):
        raise ClustersError("contact_clusters.reduce_base must be true or false, got %r" % (out["reduce_base"],))
    return out


def _cls_names():
    from .scene import CLS
    return {int(v): k for k, v in CLS.items()}


class ContactClusters(object):
    def __init__(self, cell, cfg=None, log=None):
        import mujoco
        self.mj = mujoco
        self.cell = cell
        self.m, self.d = cell.m, cell.d
        self.P = clusters_config(cfg if cfg is not None else cell.cfg)
        self.log = log or (lambda s: None)
        m, ix = self.m, cell.index
        self.dt = float(m.opt.timestep)
        self.tick_steps = int(cell.tick_steps)
        self.orig_t = np.array(m.geom_contype, dtype=np.int32)
        self.orig_a = np.array(m.geom_conaffinity, dtype=np.int32)
        self.orig_bt = np.array(m.body_contype, dtype=np.int32)
        self.orig_ba = np.array(m.body_conaffinity, dtype=np.int32)
        used = int(np.bitwise_or.reduce(self.orig_t)) | int(np.bitwise_or.reduce(self.orig_a))
        if used & (RED_MOD_BIT | RED_FIX_BIT):
            raise ClustersError("the cell's masks already use bit 64 or 128 (%d)" % used)
        self.gadr = np.asarray(m.body_geomadr, int)
        self.gnum = np.asarray(m.body_geomnum, int)
        bt, ba = self._body_or(self.orig_t, self.orig_a)
        if not (np.array_equal(bt, self.orig_bt) and np.array_equal(ba, self.orig_ba)):
            raise ClustersError("the compiled body masks are not the OR of their geoms' masks")
        names = _cls_names()
        gcls = [names[int(c)] for c in ix.geom_class]
        self.gcls = gcls
        coll = np.array([c != "none" for c in gcls])
        robot = np.array([c in ("robot", "pad") for c in gcls])
        floor = np.array([c == "floor" for c in gcls])
        never = set(self.P["never_reduce"])
        own = list(ix.geom_owner)
        is_mod = np.array([c == "module" for c in gcls])
        is_fix = np.array([c == "fixture" and own[g] not in never for g, c in enumerate(gcls)])
        # (a cost setting): with reduce_base the base's geoms are reducible too (a reduced base
        # geom collides with the robot and the floor only, as a reduced module geom: bit 64); default false = the first rule
        is_base = np.array([c == "base" and own[g] not in never for g, c in enumerate(gcls)]) & bool(self.P["reduce_base"])
        self.is_base_red = is_base
        self.reducible = is_mod | is_fix | is_base
        self.robot_t = int(np.bitwise_or.reduce(self.orig_t[robot])) if robot.any() else 0
        # the cell's masks with the floor's conaffinity widened by bit 64 (no geom carries 64 unless reduced)
        self.full_t = self.orig_t.copy()
        self.full_a = self.orig_a.copy()
        self.full_a[floor] |= RED_MOD_BIT
        self.red_t = np.where(is_mod | is_base, RED_MOD_BIT, np.where(is_fix, RED_FIX_BIT, 0)).astype(np.int32)
        self.red_a = (self.orig_a & self.robot_t).astype(np.int32)
        self.is_mod, self.is_fix, self.robot_g, self.floor_g, self.coll = is_mod, is_fix, robot, floor, coll
        # ------------------------------------------------------------------ slots
        nb = m.nbody
        body_has = np.zeros(nb, bool)
        for g in np.nonzero(coll)[0]:
            body_has[int(m.geom_bodyid[g])] = True
        self.body_slot = -np.ones(nb, int)
        self.slot_name, self.slot_static = [], []
        bowner = list(ix.body_owner)
        root = np.asarray(m.body_rootid, int)
        weld = np.asarray(m.body_weldid, int)
        tree_slot = {}
        for b in range(1, nb):
            if not body_has[b]:
                continue
            if bowner[b] == "robot":
                continue
            gs = range(self.gadr[b], self.gadr[b] + self.gnum[b])
            if any(floor[g] for g in gs):
                continue
            if weld[b] == 0 and int(m.body_mocapid[b]) < 0:
                self.body_slot[b] = len(self.slot_name)
                self.slot_name.append("static:" + m.body(b).name)
                self.slot_static.append(True)
            else:
                r = int(root[b])
                if r not in tree_slot:
                    tree_slot[r] = len(self.slot_name)
                    self.slot_name.append("tree:" + m.body(r).name + ("" if bowner[b] is None else ":" + str(bowner[b])))
                    self.slot_static.append(False)
                self.body_slot[b] = tree_slot[r]
        self.ns = len(self.slot_name)
        self.slot_static = np.array(self.slot_static, bool)
        self.geom_slot = np.array([self.body_slot[int(m.geom_bodyid[g])] if coll[g] else -1 for g in range(m.ngeom)], int)
        if np.any(self.reducible & (self.geom_slot < 0)):
            raise ClustersError("a reducible geom has no slot")
        # the movers' joints (every joint of a body in a mover slot), by type, with each joint's subtree geoms
        J = {"free": [], "ball": [], "hinge": [], "slide": []}
        tnames = {int(mujoco.mjtJoint.mjJNT_FREE): "free", int(mujoco.mjtJoint.mjJNT_BALL): "ball",
                  int(mujoco.mjtJoint.mjJNT_HINGE): "hinge", int(mujoco.mjtJoint.mjJNT_SLIDE): "slide"}
        parent = np.asarray(m.body_parentid, int)
        sub = {}
        for b in range(nb):
            x = b
            while x > 0:
                sub.setdefault(x, []).append(b)
                x = int(parent[x])
        for j in range(m.njnt):
            b = int(m.jnt_bodyid[j])
            s = int(self.body_slot[b]) if self.body_slot[b] >= 0 else -1
            if s < 0:
                # a joint of a body without collision geoms (e.g. a link between): its subtree's slot
                ss = {int(self.body_slot[x]) for x in sub.get(b, []) if self.body_slot[x] >= 0}
                if not ss:
                    continue
                if len(ss) != 1:
                    raise ClustersError("joint %s moves bodies of %d slots" % (m.joint(j).name, len(ss)))
                s = ss.pop()
            if self.slot_static[s]:
                raise ClustersError("joint %s moves a static slot" % m.joint(j).name)
            geoms = [g for x in sub.get(b, []) for g in range(self.gadr[x], self.gadr[x] + self.gnum[x]) if coll[g]]
            J[tnames[int(m.jnt_type[j])]].append({"j": j, "slot": s, "adr": int(m.jnt_qposadr[j]), "body": b,
                                                   "geoms": np.array(geoms, int)})
        self.J = J
        self._jf_adr = np.array([x["adr"] for x in J["free"]], int)
        self._jf_slot = np.array([x["slot"] for x in J["free"]], int)
        self._jb_adr = np.array([x["adr"] for x in J["ball"]], int)
        self._jb_slot = np.array([x["slot"] for x in J["ball"]], int)
        self._jh_adr = np.array([x["adr"] for x in J["hinge"]], int)
        self._jh_slot = np.array([x["slot"] for x in J["hinge"]], int)
        self._js_adr = np.array([x["adr"] for x in J["slide"]], int)
        self._js_slot = np.array([x["slot"] for x in J["slide"]], int)
        self.R_free = np.zeros(len(J["free"]))
        self.R_ball = np.zeros(len(J["ball"]))
        self.R_hinge = np.zeros(len(J["hinge"]))
        self._jf_pidx = (self._jf_adr[:, None] + np.arange(3)).reshape(-1) if len(self._jf_adr) else np.zeros(0, int)
        self._jf_qidx = (self._jf_adr[:, None] + 3 + np.arange(4)).reshape(-1) if len(self._jf_adr) else np.zeros(0, int)
        self._jb_qidx = (self._jb_adr[:, None] + np.arange(4)).reshape(-1) if len(self._jb_adr) else np.zeros(0, int)
        self._mov_idx = np.unique(np.concatenate([self._jf_pidx, self._jf_qidx, self._jb_qidx, self._jh_adr, self._js_adr]).astype(int))
        self.module_slots = sorted({int(self.body_slot[b]) for mid in ix.module_ids for b in ix.module_bodies[mid]
                                    if self.body_slot[b] >= 0})
        # ------------------------------------------------------------------ state
        self.red_idx = np.nonzero(self.reducible)[0]
        self.nr = len(self.red_idx)
        self.pos_in_red = -np.ones(m.ngeom, int)
        self.pos_in_red[self.red_idx] = np.arange(self.nr)
        self.own = self.geom_slot[self.red_idx]
        self._order = np.argsort(self.own, kind="stable")
        so = self.own[self._order]
        self._own_slots, self._own_starts = np.unique(so, return_index=True)
        self.M = np.full((self.nr, self.ns), np.inf)
        # (reduce_base) a reduced BASE geom and a slot none of whose geoms the cell's own masks let it collide with
        # (the base's own pieces: base-base pairs are off in the cell, scene.BIT_BASE) can never touch: its margin there
        # is infinite (no displacement can bring a pair into the contact list that the masks of record exclude);
        # only the base's rows: every module and fixture geom keeps its margins as before
        self.never_pair = np.zeros((self.nr, self.ns), bool)
        if is_base.any():
            ot, oa = self.orig_t, self.orig_a
            st = np.zeros(self.ns, np.int64)
            sa = np.zeros(self.ns, np.int64)
            for s in range(self.ns):                           # any(x & a_h) over a slot's geoms = x & OR(a_h)
                hs = np.nonzero(self.geom_slot == s)[0]
                st[s] = int(np.bitwise_or.reduce(ot[hs])) if len(hs) else 0
                sa[s] = int(np.bitwise_or.reduce(oa[hs])) if len(hs) else 0
            bi = np.nonzero(is_base[self.red_idx])[0]
            gb = self.red_idx[bi]
            self.never_pair[bi] = ((ot[gb].astype(np.int64)[:, None] & sa[None, :]) | (st[None, :] & oa[gb].astype(np.int64)[:, None])) == 0
        self.reduced = np.zeros(self.nr, bool)
        self.mins = np.full((self.ns, self.ns), np.inf)
        self.D = np.zeros(self.ns)
        self.q_prev = np.array(self.d.qpos, float)        # the qpos of the last query (the displacement's origin)
        self.q_check = np.array(self.d.qpos, float)       # the qpos of the last full guard check
        self.q_tick = np.array(self.d.qpos, float)        # the qpos at the last tick's end (the drop test)
        self.slack = -1.0                                 # min over (o, s) of mins - D[o] - D[s] - tolerance at q_check
        self._Lmax = 0.0                                  # set by every query (the movers' qpos entries: self._mov_idx above)
        self.dq = mujoco.MjData(m)
        self.mode = "off"                      # off (not attached) | rule | all_on | all_on_permanent
        self.why_all_on = None
        self.steps_since_tick = 0
        self.ticks_since_query = 0
        self.reenabled_since_query = 0
        self.quiet_run = 0
        self.D_tick_start = np.zeros(self.ns)
        self.D_last_tick = np.zeros(self.ns)
        self.v_at_query = 0.0
        self.settled_run = 0
        self.n = {"queries": 0, "query_s": 0.0, "query_collision_s": 0.0, "guard_calls": 0, "guard_full_checks": 0, "guard_s": 0.0, "guard_trips": 0, "reenabled": 0,
                  "tick_s": 0.0, "all_on_events": [], "incidents": [], "reduced_after_query": []}
        self.attached = False
        self._dirty = True

    # ------------------------------------------------------------------ masks
    def _body_or(self, t, a):
        bt = np.zeros(self.m.nbody, np.int32)
        ba = np.zeros(self.m.nbody, np.int32)
        has = self.gnum > 0
        if has.any():
            starts = self.gadr[has]
            bt[has] = np.bitwise_or.reduceat(np.asarray(t, np.int32), starts)
            ba[has] = np.bitwise_or.reduceat(np.asarray(a, np.int32), starts)
        return bt, ba

    def _write(self, geoms=None):
        """write the masks of `geoms` (default all) from the reduced state, and every body mask as the OR"""
        m = self.m
        t = self.full_t.copy()
        a = self.full_a.copy()
        rg = self.red_idx[self.reduced]
        t[rg] = self.red_t[rg]
        a[rg] = self.red_a[rg]
        if geoms is None:
            m.geom_contype[:] = t
            m.geom_conaffinity[:] = a
        else:
            m.geom_contype[geoms] = t[geoms]
            m.geom_conaffinity[geoms] = a[geoms]
        bt, ba = self._body_or(np.asarray(m.geom_contype, np.int32), np.asarray(m.geom_conaffinity, np.int32))
        m.body_contype[:] = bt
        m.body_conaffinity[:] = ba

    def _write_full(self):
        m = self.m
        m.geom_contype[:] = self.full_t
        m.geom_conaffinity[:] = self.full_a
        bt, ba = self._body_or(self.full_t, self.full_a)
        m.body_contype[:] = bt
        m.body_conaffinity[:] = ba

    # ------------------------------------------------------------------ the query
    def query(self):
        import time
        t0 = time.perf_counter()
        mj, m, d, dq = self.mj, self.m, self.d, self.dq
        P = self.P
        mq = float(P["query_margin_m"])
        self._write_full()
        gm0 = np.array(m.geom_margin, float)
        bm0 = np.array(m.body_margin, float)
        try:
            m.geom_margin[:] = mq
            m.body_margin[:] = mq
            dq.qpos[:] = d.qpos
            if m.nmocap:
                dq.mocap_pos[:] = d.mocap_pos
                dq.mocap_quat[:] = d.mocap_quat
            tc = time.perf_counter()
            mj.mj_kinematics(m, dq)
            mj.mj_collision(m, dq)
            self.n["query_collision_s"] += time.perf_counter() - tc
        finally:
            m.geom_margin[:] = gm0
            m.body_margin[:] = bm0
        n = int(dq.ncon)
        G = np.array(dq.contact.geom[:n], int).reshape(-1, 2)
        dist = np.array(dq.contact.dist[:n], float)
        # world boxes of every collision geom (MuJoCo's geom_aabb: centre offset and half sizes in the geom frame)
        R = np.asarray(dq.geom_xmat, float).reshape(-1, 3, 3)
        c = np.asarray(dq.geom_xpos, float) + np.einsum("gij,gj->gi", R, np.asarray(m.geom_aabb[:, :3], float))
        h = np.einsum("gij,gj->gi", np.abs(R), np.asarray(m.geom_aabb[:, 3:], float))
        lo, hi = c - h, c + h
        slo = np.full((self.ns, 3), np.inf)
        shi = np.full((self.ns, 3), -np.inf)
        for s in range(self.ns):
            gs = np.nonzero(self.geom_slot == s)[0]
            if len(gs):
                slo[s] = lo[gs].min(0)
                shi[s] = hi[gs].max(0)
        rl, rh = lo[self.red_idx], hi[self.red_idx]
        gap = np.maximum(0.0, np.maximum(slo[None, :, :] - rh[:, None, :], rl[:, None, :] - shi[None, :, :]))
        gapn = np.sqrt((gap ** 2).sum(-1))                      # (nr, ns): box-to-box distance
        M = np.maximum(mq, gapn)
        M[np.arange(self.nr), self.own] = mq
        for a_, b_ in ((0, 1), (1, 0)):
            if not n:
                break
            ia = self.pos_in_red[G[:, a_]]
            sb = self.geom_slot[G[:, b_]]
            ok = (ia >= 0) & (sb >= 0)                         # a robot, pad or floor partner: never switched off
            np.minimum.at(M, (ia[ok], sb[ok]), dist[ok])
        M[self.never_pair] = np.inf                            # (reduce_base only): pairs the masks of record exclude
        self.M = M
        # the movers' lever arms at this configuration
        xpos = np.asarray(dq.xpos, float)
        xanc = np.asarray(dq.xanchor, float)
        rb = np.asarray(m.geom_rbound, float)
        gx = np.asarray(dq.geom_xpos, float)
        for arr, kind in ((self.R_free, "free"), (self.R_ball, "ball"), (self.R_hinge, "hinge")):
            for i, x in enumerate(self.J[kind]):
                a0 = xpos[x["body"]] if kind == "free" else xanc[x["j"]]
                gs = x["geoms"]
                arr[i] = float(np.max(np.linalg.norm(gx[gs] - a0, axis=1) + rb[gs])) if len(gs) else 0.0
        # reduce a geom only where it would stay off for a while: its margins must exceed the tolerance, the
        # hysteresis and horizon_ticks times the last tick's displacement of its own slot and of the partner slot
        # (a cost rule, not a safety rule: the guard is what keeps every off pair apart)
        tol = float(P["tolerance_m"]) + float(P["hysteresis_m"])
        v = np.maximum(0.0, self.D_last_tick) * float(P["horizon_ticks"])
        self.reduced = ((M - v[self.own][:, None] - v[None, :]).min(1) > tol)
        self.v_at_query = float(v.max(initial=0.0))             # (requery_settled_ticks): the horizon term it used
        self.settled_run = 0
        self.D[:] = 0.0
        self.q_prev[:] = d.qpos
        self.q_check[:] = d.qpos
        self._set_L()
        self._update_mins()
        self._write()
        self._set_slack()
        self._dirty = False
        self.ticks_since_query = 0
        self.reenabled_since_query = 0
        self.n["queries"] += 1
        self.n["query_s"] += time.perf_counter() - t0
        if len(self.n["reduced_after_query"]) < 2000:
            self.n["reduced_after_query"].append([round(float(d.time), 4), int(self.reduced.sum()), n])

    def _update_mins(self, slots=None):
        """mins[o, s] = the least M[g, s] over the reduced geoms of own slot o (inf when none): vectorised over every
        row (the slots argument is accepted for the callers' readability; every row is recomputed)"""
        Mm = np.where(self.reduced[:, None], self.M, np.inf)[self._order]
        self.mins[:] = np.inf
        if len(self._own_starts):
            self.mins[self._own_slots] = np.minimum.reduceat(Mm, self._own_starts, axis=0)

    # ------------------------------------------------------------------ the guard (before every physics step)
    @staticmethod
    def _qangle(q1, q0):
        """rotation angles between rows of two (n, 4) quaternion arrays (not necessarily unit)"""
        dot = np.abs((q1 * q0).sum(1)) / np.maximum(1e-300, np.sqrt((q1 * q1).sum(1) * (q0 * q0).sum(1)))
        return 2.0 * np.arccos(np.minimum(1.0, dot))

    def _displacement(self, q, q0=None):
        """a bound, per slot, of how far any point of the slot's collision geoms lies from where it was at configuration
        q0 (default: the last query's): free joint |dp| + angle x R, ball angle x R, hinge |dq| x R, slide |dq|, R = the
        joint's lever arm at the query; then a second pass adding angle x 2 x the first-order bound (the lever arm at the
        intermediate configurations of the telescoping chain is at most R + the first-order bound). Endpoint to endpoint:
        no sum over steps (the distance now is at least the query's minus this)."""
        qp = self.q_prev if q0 is None else q0
        ns = self.ns
        D1 = np.zeros(ns)
        rots = []
        if len(self._jf_adr):
            dp = np.sqrt(((q[self._jf_pidx] - qp[self._jf_pidx]).reshape(-1, 3) ** 2).sum(1))
            ang = self._qangle(q[self._jf_qidx].reshape(-1, 4), qp[self._jf_qidx].reshape(-1, 4))
            D1 += np.bincount(self._jf_slot, weights=dp + ang * self.R_free, minlength=ns)
            rots.append((self._jf_slot, ang))
        if len(self._jb_adr):
            ang = self._qangle(q[self._jb_qidx].reshape(-1, 4), qp[self._jb_qidx].reshape(-1, 4))
            D1 += np.bincount(self._jb_slot, weights=ang * self.R_ball, minlength=ns)
            rots.append((self._jb_slot, ang))
        if len(self._jh_adr):
            ang = np.abs(q[self._jh_adr] - qp[self._jh_adr])
            D1 += np.bincount(self._jh_slot, weights=ang * self.R_hinge, minlength=ns)
            rots.append((self._jh_slot, ang))
        if len(self._js_adr):
            D1 += np.bincount(self._js_slot, weights=np.abs(q[self._js_adr] - qp[self._js_adr]), minlength=ns)
        D2 = D1.copy()
        for sl, ang in rots:
            D2 += np.bincount(sl, weights=ang * 2.0 * D1[sl], minlength=ns)
        return D2

    def _set_L(self):
        """the cheap bound of the guard: from the last full check on, any slot's displacement bound grows by at most
        Lmax x max |dq| over the movers' qpos entries: hinge / slide |dq| x (R + 2 D + 0.1 m) / |dq|; free sqrt(3) |dp|_inf
        + 2 pi |dquat|_inf x (R + 2 D + 0.1 m) (the angle between two unit quaternions is at most pi |q - q'|_2 <=
        2 pi |q - q'|_inf); ball 2 pi |dquat|_inf x (R + 2 D + 0.1 m); R the lever arm at the query, D the slot's
        displacement bound at the last full check (the lever arm now is at most R + 2 D), the 0.1 m the growth while the
        cheap test passes (it passes only below the slack, millimetres); summed over a slot's joints"""
        ns = self.ns
        L = np.zeros(ns)
        g = 2.0 * self.D + 0.1
        if len(self._jf_adr):
            L += np.bincount(self._jf_slot, weights=math.sqrt(3.0) + 2.0 * math.pi * (self.R_free + g[self._jf_slot]), minlength=ns)
        if len(self._jb_adr):
            L += np.bincount(self._jb_slot, weights=2.0 * math.pi * (self.R_ball + g[self._jb_slot]), minlength=ns)
        if len(self._jh_adr):
            L += np.bincount(self._jh_slot, weights=self.R_hinge + g[self._jh_slot], minlength=ns)
        if len(self._js_adr):
            L += np.bincount(self._js_slot, weights=np.ones(len(self._js_adr)), minlength=ns)
        self._Lmax = float(L.max(initial=0.0))

    def _set_slack(self):
        fin = np.isfinite(self.mins)
        self.slack = float((self.mins - (self.D[:, None] + self.D[None, :]))[fin].min()) - float(self.P["tolerance_m"]) if fin.any() else np.inf

    def pre_step(self):
        """the guard, before every physics step's collision and before the tick's mj_forward (see the module doc):
        the cheap test first (every slot's bound grows by at most Lmax x max |dq| since the last full check: when twice
        that stays below the slack no pair can have reached its margin); else the full check of every reduced geom from
        the displacement bound between the last query's configuration and this one"""
        self.steps_since_tick += 1
        if self.mode != "rule":
            return
        q = self.d.qpos
        self.n["guard_calls"] += 1
        if not self._dirty:
            mi = self._mov_idx
            dq = float(np.max(np.abs(q[mi] - self.q_check[mi]))) if len(mi) else 0.0
            if 2.0 * dq * self._Lmax < self.slack:
                return
        import time
        t0 = time.perf_counter()
        self.n["guard_full_checks"] += 1
        qa = np.array(q, float)
        self.D = self._displacement(qa)
        self.D[self.slot_static] = 0.0
        self._dirty = False
        tol = float(self.P["tolerance_m"])
        bad = (self.mins - (self.D[:, None] + self.D[None, :])) <= tol
        if bad.any():
            self.n["guard_trips"] += 1
            rows = np.nonzero(bad.any(1))[0]
            sel = np.nonzero(self.reduced & np.isin(self.own, rows))[0]
            if len(sel):
                slack = (self.M[sel] - self.D[self.own[sel]][:, None] - self.D[None, :]).min(1)
                off = sel[slack <= tol]
                if len(off):
                    self.reduced[off] = False
                    on = self.red_idx[off]
                    self._write(on)
                    self.n["reenabled"] += int(len(on))
                    self.reenabled_since_query += int(len(on))
            self._update_mins()
        self.q_check[:] = qa
        self._set_L()
        self._set_slack()
        self.n["guard_s"] += time.perf_counter() - t0

    # ------------------------------------------------------------------ everything on
    def all_on(self, why, permanent=False):
        if self.mode == "all_on_permanent":
            return
        self.reduced[:] = False
        self.mins[:] = np.inf
        self._write_full()
        self.slack = np.inf
        self.mode = "all_on_permanent" if permanent else "all_on"
        self.why_all_on = why
        self.quiet_run = 0
        if len(self.n["all_on_events"]) < 200:
            self.n["all_on_events"].append([round(float(self.d.time), 4), why, bool(permanent)])

    # ------------------------------------------------------------------ the tick's end (a cell hook after the coupling)
    def tick_end(self):
        import time
        t0 = time.perf_counter()
        d = self.d
        if self.steps_since_tick != self.tick_steps + 1 and self.mode == "rule":
            self.n["incidents"].append([round(float(d.time), 4), "%d guard calls in this tick (%d expected: every physics step and the tick's mj_forward)"
                                        % (self.steps_since_tick, self.tick_steps + 1)])
            self.all_on("a tick whose physics steps bypassed the guard")
        self.steps_since_tick = 0
        mon = self.cell.monitor
        if mon is not None and mon.scrapped and self.mode != "all_on_permanent":
            self.all_on("module scrapped: %s" % ",".join(sorted(mon.scrapped)), permanent=True)
        if self.mode == "all_on_permanent":
            self.n["tick_s"] += time.perf_counter() - t0
            return
        # the movers' displacement in this tick (endpoint to endpoint, from the state itself, also in all-on mode)
        q = np.array(d.qpos, float)
        Dt = self._displacement(q, self.q_tick)
        self.q_tick[:] = q
        if self.mode == "rule":
            self.D = self._displacement(q)
            self.D[self.slot_static] = 0.0
        self.D_last_tick = np.array(Dt, float)
        mods = [s for s in self.module_slots]
        fast = [self.slot_name[s] for s in mods if Dt[s] > float(self.P["drop_m_per_tick"])]
        quiet = all(Dt[s] <= float(self.P["quiet_m_per_tick"]) for s in mods)
        if fast and self.mode == "rule":
            self.all_on("dropped (moved more than %.1f mm in a tick): %s" % (1e3 * float(self.P["drop_m_per_tick"]), ",".join(fast)))
        elif self.mode == "all_on":
            self.quiet_run = self.quiet_run + 1 if quiet else 0
            if self.quiet_run >= int(self.P["quiet_ticks"]):
                self.mode = "rule"
                self.query()
        elif self.mode == "rule":
            self.ticks_since_query += 1
            # (requery_settled_ticks, a re-query policy setting; 0 = the first policy): one query more when every
            # mover has been settled for that many ticks (the horizon term a query would use now, horizon_ticks x the
            # largest last-tick displacement of a mover, at most a tenth of requery_disp_m) after a query that ran while
            # something moved (its horizon term above requery_disp_m): the geoms the moving state kept on are reduced
            # again at rest. Cost only: a query never switches a pair off that the guard has not proved apart
            ns_ = int(self.P["requery_settled_ticks"])
            if ns_ > 0:
                hz = float(self.P["horizon_ticks"]) * float(np.max(Dt[~self.slot_static], initial=0.0)) if self.ns else 0.0
                self.settled_run = self.settled_run + 1 if hz <= 0.1 * float(self.P["requery_disp_m"]) else 0
            if self.ticks_since_query >= int(self.P["requery_min_ticks"]) and (
                    self.reenabled_since_query >= int(self.P["requery_min_reenabled"])
                    or float(self.D.max(initial=0.0)) > float(self.P["requery_disp_m"])):
                self.query()
            elif (ns_ > 0 and self.v_at_query > float(self.P["requery_disp_m"]) and self.settled_run >= ns_
                  and self.ticks_since_query >= int(self.P["requery_min_ticks"])):
                self.n["settled_queries"] = self.n.get("settled_queries", 0) + 1
                self.query()
        self.n["tick_s"] += time.perf_counter() - t0

    # ------------------------------------------------------------------ plug-in
    def attach(self):
        cell = self.cell
        if self.attached:
            raise ClustersError("contact clusters already attached")
        if not hasattr(cell, "pre_step_hooks"):
            raise ClustersError("the cell has no pre_step_hooks (scene.Cell)")
        need = set()
        for kind in ("free", "ball", "hinge", "slide"):
            for x in self.J[kind]:
                n = {"free": 7, "ball": 4, "hinge": 1, "slide": 1}[kind]
                need.update(range(x["adr"], x["adr"] + n))
        if set(int(i) for i in self._mov_idx) != need:
            raise ClustersError("the guard's cheap test does not cover every mover's qpos entry")
        cell.pre_step_hooks.append(self.pre_step)
        cell.hooks.append(self.tick_end)
        cell.clusters = self
        self.mode = "rule"
        self.q_tick[:] = self.d.qpos
        self.query()
        self.attached = True
        return self

    def stats(self):
        r = self.reduced
        by = {}
        for g, on in zip(self.red_idx, r):
            k = "%s:%s" % (self.gcls[g], self.cell.index.geom_owner[g])
            x = by.setdefault(k, [0, 0])
            x[0] += 1
            x[1] += int(on)
        return {"mode": self.mode, "why_all_on": self.why_all_on, "reducible": int(self.nr), "reduced_now": int(r.sum()),
                "by_owner_reducible_reduced": by, "slots": list(self.slot_name),
                "counters": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.n.items()
                             if k not in ("reduced_after_query",)},
                "reduced_after_query_tail": self.n["reduced_after_query"][-20:]}


def mask_relation_check(cl):
    """the exhaustive class test of the masks a ContactClusters object writes: for every pair of geom classes
    (original contype, original conaffinity, kind) x (full, reduced), the predicate (t1 & a2) | (t2 & a1) of the
    written masks equals the cell's own predicate, except that a pair holding a reduced geom and a geom that is neither
    the robot's nor the floor's is off. Returns {"class_pairs", "mismatches", "first"}."""
    reps = {}
    for g in np.nonzero(cl.coll)[0]:
        kind = "robot" if cl.robot_g[g] else ("floor" if cl.floor_g[g] else ("reducible" if cl.reducible[g] else "other"))
        key = (int(cl.orig_t[g]), int(cl.orig_a[g]), kind, bool(cl.is_mod[g]))
        reps.setdefault(key, g)
    bad, n = [], 0
    states = []
    for key, g in reps.items():
        for red in ((False, True) if key[2] == "reducible" else (False,)):
            t = int(cl.red_t[g]) if red else int(cl.full_t[g])
            a = int(cl.red_a[g]) if red else int(cl.full_a[g])
            states.append((key, red, t, a, g))
    for k1, r1, t1, a1, g1 in states:
        for k2, r2, t2, a2, g2 in states:
            n += 1
            orig = bool((cl.orig_t[g1] & cl.orig_a[g2]) | (cl.orig_t[g2] & cl.orig_a[g1]))
            always1, always2 = k1[2] in ("robot", "floor"), k2[2] in ("robot", "floor")
            off = (r1 and not always2) or (r2 and not always1)
            want = orig and not off
            got = bool((t1 & a2) | (t2 & a1))
            if want != got:
                bad.append((k1, r1, k2, r2, want, got))
    return {"class_pairs": n, "mismatches": len(bad), "first": bad[:5]}
