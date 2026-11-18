"""Collision-free UR motions in the quadruped cell (privileged: the planner sees the cell's exact state).

`Planner(cell)` keeps its own copy of the cell's model and checks a UR posture, with the held unit (and whatever is
fastened to it: the upper leg carries its lower leg) carried rigidly by the hand, against everything else in the
CURRENT scene: the bench, the stand, the robot being built at its hinges' current angles, the units on the bench.
Clearances (the full-arm task's): the arm and the gripper body `clearance_m` (8 mm), the fingers `finger_clear` (2 mm),
the held unit `held_clearance_m` (4 mm); the UR against itself only where it interpenetrates (the gripper's own
linkage never counts). `free(q0, q1)` checks the joint-space segment; `plan(q0, q1)` returns a straight segment when it
is free, else an RRT-Connect path (sampled in the box around both ends grown by `box_grow_rad`, inside the joint
ranges; shortcut smoothing); `line_free(qs)` checks a list of postures (a Cartesian path).

Collision bits on the copy: the UR and the held unit (contype 2, conaffinity 2), everything else (1, 2): the rest never
collides with itself (the half-built robot costs no contacts), the UR collides with itself and with the rest.
"""
import math

import numpy as np

from .coupling import quat_wxyz


class Planner(object):
    def __init__(self, cell, clearance_m=0.008, held_clearance_m=0.004, finger_clear_m=0.002, seed=0, box_grow_rad=0.8):
        import copy
        import mujoco
        from .gripper_2f85 import set_opening
        self.mj, self.cell, self._set_opening = mujoco, cell, set_opening
        self.m = copy.deepcopy(cell.m)
        self.d = mujoco.MjData(self.m)
        m = self.m
        rb = cell.robot
        self.prefix = rb["prefix"]
        self.clear, self.held_clear, self.finger_clear = float(clearance_m), float(held_clearance_m), float(finger_clear_m)
        self.box_grow = float(box_grow_rad)
        self.robot = set(b for b in range(m.nbody) if m.body(b).name.startswith(self.prefix))
        self.fingers = set(m.body(n).id for side in ("right", "left") for n in rb["finger_bodies"][side])
        self.gripper = set(m.body(n).id for n in rb["gripper_bodies"])
        self.col = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.base_bits = {}
        for g in self.col:
            if m.geom_bodyid[g] in self.robot:
                m.geom_margin[g] = self.clear
                m.geom_contype[g], m.geom_conaffinity[g] = 2, 2
            else:
                m.geom_contype[g], m.geom_conaffinity[g] = 1, 2
            self.base_bits[g] = (int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
        self.jids = [m.joint(j).id for j in rb["joints"]]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jids])
        self.lo = np.array([m.jnt_range[j][0] for j in self.jids]) + 0.02
        self.hi = np.array([m.jnt_range[j][1] for j in self.jids]) - 0.02
        self.tcp = m.site(rb["tcp_site"]).id
        self.rng = np.random.default_rng(seed)
        self.held = None                     # (root unit, G = root pose in the tcp frame)
        self.carried = []                    # [(free-joint qpos address, pose relative to the root)] incl. the root
        self.held_bodies = set()
        self.held_geoms = []
        self.allow = set()
        self.n_checks = 0
        self.unit_of_body = dict(cell.unit_of_body)
        self._h0 = {}

    # ------------------------------------------------------------------ the scene
    def _free_adr(self, unit):
        return self.m.jnt_qposadr[self.m.joint("free_" + unit).id]

    def _set_pose(self, unit, T):
        a = self._free_adr(unit)
        self.d.qpos[a:a + 3] = T[:3, 3]
        self.d.qpos[a + 3:a + 7] = quat_wxyz(T[:3, :3])

    def sync(self, held=None, group=(), allow=(), poses=None, width=None, q=None):
        """take the cell's current state; `held` = (unit, G: its pose in the tcp frame) carried by the hand together
        with the units in `group` (rigidly, as they are now relative to it); `poses` {unit: T} overrides where units
        are (a hypothetical scene; applied before the group is measured); `width` the finger opening; `allow` units or
        body names whose contacts are allowed (the unit being grasped)"""
        c, m, d = self.cell, self.m, self.d
        d.qpos[:] = c.d.qpos
        for name, T in (poses or {}).items():
            self._set_pose(name, T)
        if q is not None:
            d.qpos[self.qadr] = q
        self.mj.mj_kinematics(m, d)
        if width is not None:
            self._set_opening(m, d, float(width), prefix=self.prefix)
        for g in self.held_geoms:
            m.geom_contype[g], m.geom_conaffinity[g] = self.base_bits[g]
            m.geom_margin[g] = 0.0
        self.held_geoms, self.carried, self.held_bodies = [], [], set()
        self.held = held
        if held is not None:
            root = held[0]
            units = [root] + [u for u in group if u != root]
            Tr = self._body_T(root)
            for u in units:
                self.carried.append((self._free_adr(u), np.linalg.inv(Tr) @ self._body_T(u)))
            self.held_bodies = set(b for b, u in self.unit_of_body.items() if u in units)
            self.held_geoms = [g for g in self.col if m.geom_bodyid[g] in self.held_bodies]
            for g in self.held_geoms:
                m.geom_contype[g], m.geom_conaffinity[g] = 2, 2
                m.geom_margin[g] = self.held_clear
        ab = set()
        for n in allow:
            if n in c.geoms_of:
                ab |= set(b for b, u in self.unit_of_body.items() if u == n)
            else:
                ab.add(m.body(n).id)
        self.allow = ab

    def _body_T(self, name):
        b = self.m.body(name).id
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[b].reshape(3, 3)
        T[:3, 3] = self.d.xpos[b]
        return T

    def _place_held(self):
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[self.tcp].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[self.tcp]
        Tr = T @ self.held[1]
        for a, Trel in self.carried:
            P = Tr @ Trel
            self.d.qpos[a:a + 3] = P[:3, 3]
            self.d.qpos[a + 3:a + 7] = quat_wxyz(P[:3, :3])

    def hits(self, q, tol_m=0.0005, finger_clear=None, link_clear=None, held_clear=None):
        """[(UR or held body, other body, distance m)] of the contacts closer than the clearances at posture q (the
        held unit: closer than its clearance - tol_m; the UR against itself: interpenetrating)"""
        mj, m, d = self.mj, self.m, self.d
        self.n_checks += 1
        d.qpos[self.qadr] = q
        mj.mj_kinematics(m, d)
        if self.held is not None:
            self._place_held()
            mj.mj_kinematics(m, d)
        mj.mj_collision(m, d)
        fc = self.finger_clear if finger_clear is None else finger_clear
        lc = self.clear if link_clear is None else link_clear
        hc = self.held_clear if held_clear is None else held_clear
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            r1, r2 = b1 in self.robot, b2 in self.robot
            h1, h2 = b1 in self.held_bodies, b2 in self.held_bodies
            if r1 and r2:
                if (b1 in self.gripper and b2 in self.gripper) or c.dist >= 0.0:
                    continue
                out.append((m.body(b1).name, m.body(b2).name, float(c.dist)))
                continue
            if (r1 or h1) == (r2 or h2):
                continue                      # the hand on the held unit, or two scene bodies
            mine, other = (b1, b2) if (r1 or h1) else (b2, b1)
            if other in self.allow:
                continue
            if mine in self.held_bodies:
                if c.dist < hc - tol_m:
                    out.append((m.body(mine).name, m.body(other).name, float(c.dist)))
            elif mine in self.fingers:
                if c.dist < fc:
                    out.append((m.body(mine).name, m.body(other).name, float(c.dist)))
            elif c.dist < lc:
                out.append((m.body(mine).name, m.body(other).name, float(c.dist)))
        return out

    def ok(self, q):
        hs = self.hits(q)
        if self._h0:
            # near-contacts the start posture already had are tolerated while they do not get closer (a hand that has
            # just let a part go starts inside the clearance of the robot being built)
            hs = [h for h in hs if not ((h[0], h[1]) in self._h0 and h[2] >= self._h0[(h[0], h[1])] - 0.0005)]
        return not hs

    # ------------------------------------------------------------------ segments and paths
    def free(self, q0, q1, step=0.004):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = max(1, int(math.ceil(float(np.max(np.abs(q1 - q0))) / step)))
        return all(self.ok(q0 + (q1 - q0) * k / n) for k in range(1, n + 1))

    def line_free(self, qs, moving_away_ok=True):
        """every posture free; with moving_away_ok a contact already there at the first posture is tolerated while it
        does not get closer (backing out of a tight spot)"""
        if not qs:
            return True
        h0 = {(h[0], h[1]): h[2] for h in self.hits(qs[0])} if moving_away_ok else {}
        for q in qs:
            for a, b, dist in self.hits(q):
                if (a, b) in h0 and dist >= h0[(a, b)] - 0.0005:
                    continue
                return False
        return True

    def plan(self, q0, q1, max_iter=4000, step=0.25):
        """[q0, ..., q1] collision-free (straight if possible, else RRT-Connect + shortcuts); None if not found. The
        start's own near-contacts are tolerated along the path while they do not get closer."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        self._h0 = {(h[0], h[1]): h[2] for h in self.hits(q0)}
        try:
            return self._plan(q0, q1, max_iter, step)
        finally:
            self._h0 = {}

    def _plan(self, q0, q1, max_iter, step):
        if not self.ok(q1):
            self.why = "goal in collision: %s" % sorted(set(h[1] for h in self.hits(q1)))[:4]
            return None
        if self.free(q0, q1):
            return [q0, q1]
        lo = np.maximum(self.lo, np.minimum(q0, q1) - self.box_grow)
        hi = np.minimum(self.hi, np.maximum(q0, q1) + self.box_grow)
        ta, tb = [(q0, -1)], [(q1, -1)]

        def nearest(t, q):
            return int(np.argmin([np.linalg.norm(x[0] - q) for x in t]))

        def extend(t, q):
            i = nearest(t, q)
            v = q - t[i][0]
            dist = float(np.linalg.norm(v))
            qn = q if dist <= step else t[i][0] + v / dist * step
            if not self.free(t[i][0], qn):
                return None
            t.append((qn, i))
            return len(t) - 1

        def chain(t, i):
            out = []
            while i >= 0:
                out.append(t[i][0])
                i = t[i][1]
            return out

        swapped = False
        for it in range(max_iter):
            qr = lo + (hi - lo) * self.rng.random(len(lo)) if it % 5 else (tb[-1][0] if not swapped else q1)
            ia = extend(ta, qr)
            if ia is None:
                ta, tb, swapped = tb, ta, not swapped
                continue
            qa = ta[ia][0]
            while True:
                ib = extend(tb, qa)
                if ib is None:
                    break
                if np.linalg.norm(tb[ib][0] - qa) < 1e-9:
                    pa, pb = chain(ta, ia)[::-1], chain(tb, ib)
                    path = pa + pb[1:]
                    if swapped:
                        path = path[::-1]
                    return self.shortcut(path)
            ta, tb, swapped = tb, ta, not swapped
        self.why = "no path in %d RRT iterations (start hits %s)" % (max_iter, sorted(set(k[1] for k in self._h0))[:4])
        return None

    def shortcut(self, path, rounds=60):
        path = [np.asarray(p, float) for p in path]
        for _ in range(rounds):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self.free(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path
