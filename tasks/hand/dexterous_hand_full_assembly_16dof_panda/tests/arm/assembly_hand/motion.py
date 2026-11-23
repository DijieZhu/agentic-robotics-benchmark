"""Collision-free Panda motions in the hand assembly cell (privileged: the planner sees the cell's exact state).
After the tensegrity cell's `motion.py` (the SO-101 cell's Panda planner with the quadruped cell's tolerance of the
start posture's own near-contacts), for articulated units: a unit is all of its link bodies, the held unit is carried
rigidly by the hand at its current joint angles (its root link's free joint set from the tcp, the chain following
through its joint angles as they are now).

`Planner(cell)` keeps its own copy of the cell's model and checks a Panda posture, with the held unit carried by the
hand, against everything else in the CURRENT scene: the bench, the fixture with its deck and the thumb's pad, the
palm base with the units fastened on it, the trays, their nests and the units still in them. Clearances (the
precedents'): the arm and the hand `clearance_m` (8 mm), the fingers `finger_clear` (2 mm), the held
unit `held_clearance_m` (4 mm). `free(q0, q1)` checks the joint-space segment in 5 mm-equivalent steps;
`plan(q0, q1)` returns a straight segment when it is free, else an RRT-Connect path (sampled in the box around both
ends grown by `box_grow_rad`, inside the joint ranges; shortcut smoothing; a fixed random stream, so a plan is
reproducible); `line_free(qs)` checks a list of postures (a Cartesian path). The start posture's own near-contacts
are tolerated along a path while they do not get closer (a hand that has just let a unit go starts inside the
clearance of the build).

Collision bits on the copy: the Panda and the held unit (contype 2, conaffinity 1), everything else (1, 2): only the
Panda / held unit against the rest is checked (the build's own fastened contacts cost nothing).
"""
import math

import numpy as np

from .coupling import quat_wxyz


class Planner(object):
    def __init__(self, cell, clearance_m=0.008, held_clearance_m=0.004, finger_clear_m=0.002, seed=0, box_grow_rad=0.8):
        import copy
        import mujoco
        self.mj, self.cell = mujoco, cell
        self.m = copy.deepcopy(cell.m)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.clear, self.held_clear, self.finger_clear = float(clearance_m), float(held_clearance_m), float(finger_clear_m)
        self.box_grow = float(box_grow_rad)
        self.panda = set(b for b in range(m.nbody) if m.body(b).name.startswith("panda/"))
        self.fingers = set(b for b in self.panda if "finger" in m.body(b).name)
        self.col = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        for g in self.col:
            if m.geom_bodyid[g] in self.panda:
                m.geom_margin[g] = self.clear
                m.geom_contype[g], m.geom_conaffinity[g] = 2, 1
            else:
                m.geom_margin[g] = 0.0
                m.geom_contype[g], m.geom_conaffinity[g] = 1, 2
        self.pj = [m.joint("panda/joint%d" % (i + 1)).id for i in range(7)]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.pj])
        self.lo = np.array([m.jnt_range[j][0] for j in self.pj]) + 0.02
        self.hi = np.array([m.jnt_range[j][1] for j in self.pj]) - 0.02
        self.tcp = m.site("tcp").id
        self.finger_q = list(cell.ctrl.finger_q)
        self.rng = np.random.default_rng(seed)
        self.unit_bodies = {u: set(bs) for u, bs in cell.bodies_of.items()}
        self.held = None                    # (unit, G = its root link's pose in the tcp frame)
        self.held_geoms = []
        self.held_ids = set()
        self.allow = set()
        self.n_checks = 0
        self._h0 = {}
        self.why = ""

    # ------------------------------------------------------------------ the scene
    def _free_adr(self, unit):
        return self.m.jnt_qposadr[self.m.joint("%s__free" % unit).id]

    def _set_pose(self, unit, T, joints=None):
        a = self._free_adr(unit)
        self.d.qpos[a:a + 3] = T[:3, 3]
        self.d.qpos[a + 3:a + 7] = quat_wxyz(T[:3, :3])
        for jn, v in (joints or {}).items():
            self.d.qpos[self.m.jnt_qposadr[self.cell.joint_ids[unit][jn]]] = float(v)

    def _bodies(self, names):
        out = set()
        for n in names:
            if n in self.unit_bodies:
                out |= self.unit_bodies[n]
            else:
                out.add(self.m.body(n).id)
        return out

    def sync(self, held=None, allow=(), poses=None, width=None, joints=None):
        """take the cell's current state (every body, every joint, the finger opening) and what the hand carries;
        `poses` {unit: T of its root link} and `joints` {unit: {joint: rad}} override where units are and how their
        chains stand (a hypothetical scene), `width` the finger opening, `allow` units or body names whose contacts
        are allowed (the unit being grasped)"""
        c, m, d = self.cell, self.m, self.d
        d.qpos[:] = c.d.qpos
        for name, T in (poses or {}).items():
            self._set_pose(name, T, (joints or {}).get(name))
        for name, js in (joints or {}).items():
            if name not in (poses or {}):
                for jn, v in js.items():
                    d.qpos[m.jnt_qposadr[c.joint_ids[name][jn]]] = float(v)
        if width is not None:
            for fq in self.finger_q:
                d.qpos[fq] = float(width) / 2.0
        for g in self.held_geoms:
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 2
            m.geom_margin[g] = 0.0
        self.held_geoms = []
        self.held = held
        self.held_ids = set()
        if held is not None:
            self.held_ids = set(self.unit_bodies[held[0]])
            self.held_adr = self._free_adr(held[0])
            self.held_geoms = [g for g in self.col if m.geom_bodyid[g] in self.held_ids]
            for g in self.held_geoms:
                m.geom_contype[g], m.geom_conaffinity[g] = 2, 1
                m.geom_margin[g] = self.held_clear
        self.allow = self._bodies(allow)
        self.mj.mj_kinematics(m, d)

    def _place_held(self):
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[self.tcp].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[self.tcp]
        P = T @ self.held[1]
        a = self.held_adr
        self.d.qpos[a:a + 3] = P[:3, 3]
        self.d.qpos[a + 3:a + 7] = quat_wxyz(P[:3, :3])

    def hits(self, q, tol_m=0.0005, finger_clear=None, link_clear=None, held_clear=None):
        """[(Panda or held body, other body, distance m)] of the contacts closer than the clearances at posture q"""
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
        hid = self.held_ids
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            s1, s2 = (b1 in self.panda or b1 in hid), (b2 in self.panda or b2 in hid)
            if s1 == s2:
                continue
            mine, other = (b1, b2) if s1 else (b2, b1)
            if other in self.allow:
                continue
            if mine in hid:
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
            hs = [h for h in hs if not ((h[0], h[1]) in self._h0 and h[2] >= self._h0[(h[0], h[1])] - 0.0005)]
        return not hs

    def tcp_T(self, q):
        self.d.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.m, self.d)
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[self.tcp].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[self.tcp]
        return T

    # ------------------------------------------------------------------ segments and paths
    def free(self, q0, q1, step=0.005):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = max(1, int(math.ceil(float(np.max(np.abs(q1 - q0))) / step)))
        return all(self.ok(q0 + (q1 - q0) * k / n) for k in range(1, n + 1))

    def line_free(self, qs, moving_away_ok=True, **kw):
        """every posture free; with moving_away_ok a contact already there at the first posture is tolerated while it
        does not get closer (backing out of a tight spot); kw: the clearances of `hits`"""
        if not qs:
            return True
        h0 = {(h[0], h[1]): h[2] for h in self.hits(qs[0], **kw)} if moving_away_ok else {}
        for q in qs:
            for a, b, dist in self.hits(q, **kw):
                if (a, b) in h0 and dist >= h0[(a, b)] - 0.0005:
                    continue
                self.why = "line: %s - %s at %.1f mm" % (a, b, dist * 1e3)
                return False
        return True

    def plan(self, q0, q1, max_iter=4000, step=0.25):
        """[q0, ..., q1] collision-free (straight if possible, else RRT-Connect + shortcuts); None if not found"""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        self._h0 = {(h[0], h[1]): h[2] for h in self.hits(q0)}
        try:
            return self._plan(q0, q1, max_iter, step)
        finally:
            self._h0 = {}

    def _plan(self, q0, q1, max_iter, step):
        if not self.ok(q1):
            self.why = "goal in collision: %s" % sorted(set((h[0], h[1], round(h[2] * 1e3, 1)) for h in self.hits(q1)))[:4]
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
            qr = lo + (hi - lo) * self.rng.random(7) if it % 5 else (tb[-1][0] if not swapped else q1)
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
        self.why = "no path in %d RRT iterations (start near %s)" % (max_iter, sorted(set(k[1] for k in self._h0))[:4])
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
