"""The active window of mode `active_window`: only ducts near a car, slow car or box are dynamic.

Every piece exists twice in the model (sketch.ducts.build):
* the frozen layer: world-body geoms `duct/<D>/<k>/col` and `.../lidar` with the contract's names. Cheap: MuJoCo's
  broad phase and ray casting handle world-body geoms through one bounding volume hierarchy, where every piece body
  costs a broad-phase entry per step even asleep.
* `window.slots` dynamic chains `duct/<slot>/<k>` of planar pieces (the same piece model as modes sleep / awake),
  with collision capsules only.

A duct is moved into a free slot when a window centre comes within `margin_on_m` + (its speed x the check
interval) of the duct's skin: the slot's piece frames are set to the duct's present poses (joint values 0), its
bending springs get the offsets that keep their rest shape the laid shape of the world file, its capsules and the
duct's frozen collision geoms swap (contact bits), and from then on the frozen layer's LiDAR cylinders follow the
slot's pieces after every physics step. A duct goes back to the frozen layer when no centre is within
`margin_off_m` of it and its pieces have been at rest for `rest_steps` checks: its poses are written into the frozen
geoms ("written back"). Moving a frozen geom outside its leaf box of the world body's bounding volume hierarchy would
hide it from collisions and rays, so leaf boxes are built `bvh_inflate_m` larger and refitted (leaf and ancestors)
when a written pose leaves them.

Centres: the ego car `car/chassis`, slow cars `slow<i>/chassis`, boxes `box/<B>` (whatever the model holds), each
a circle around its footprint. Decisions depend only on the state, in a fixed order (ducts in world-file order,
slots by index), so the window is deterministic.
"""

import math

import mujoco
import numpy as np

from sketch.examples import constants as K

from . import geom
from .build import duct_bits, slot_ids
from .params import bend_solref, piece_constants

CAR_CENTRE_X = 0.5 * (K.X_FRONT_M + K.X_REAR_M)          # footprint centre along the car's x axis (frame C)
CAR_RADIUS = math.hypot(0.5 * (K.X_FRONT_M - K.X_REAR_M), 0.5 * K.ENVELOPE_M[1])


def _quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class _Slot(object):
    pass


class ActiveWindow(object):
    """Owned by a DuctTracker of mode `active_window`; the tracker calls `step()` after every physics step."""

    def __init__(self, model, data, world, params, tracker):
        self.m, self.d, self.world, self.p, self.tr = model, data, world, params, tracker
        w = params["window"]
        self.margin_on = float(w["margin_on_m"])
        self.margin_off = float(w["margin_off_m"])
        self.every = max(1, int(w["check_every_steps"]))
        self.rest_speed = float(w["rest_speed_mps"])
        self.rest_steps = int(w["rest_steps"])
        self.inflate = float(w["bvh_inflate_m"])
        self.sync_tol = float(w["sync_tol"])
        self.slot_sleep = bool(w["slot_sleep"])
        self.park = np.array([float(v) for v in w["park_origin_m"]])
        self.bits = duct_bits(params)
        self.dt = float(model.opt.timestep)
        ducts = world["truth"]["ducts"]
        self.ducts = ducts
        n = tracker.n
        self.n = n
        self.lengths = tracker.lengths
        self.slices = tracker.slices
        self.world_poses = tracker.world_poses
        self.pose = tracker.world_poses.copy()
        # ---- the frozen layer
        gid = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)  # noqa: E731
        self.col = np.array([gid(K.duct_geom(did, k, "col")) for did, k in tracker.ids], dtype=int)
        self.lid = np.array([gid(K.duct_geom(did, k, "lidar")) for did, k in tracker.ids], dtype=int)
        if (self.col < 0).any() or (self.lid < 0).any():
            raise ValueError("the frozen layer lacks duct geoms of the world file")
        if (model.geom_bodyid[self.col] != 0).any() or (model.geom_bodyid[self.lid] != 0).any():
            raise ValueError("the frozen layer's geoms must hang on the world body")
        # the axis-segment extents (lo, hi) of every collision capsule, for the window's distance test
        ext = []
        for j, d in enumerate(ducts):
            c, nn = float(d["piece_length_m"]), len(d["pieces"])
            ext += [geom.collision_extent(k, nn, c) for k in range(nn)]
        self.ext = np.array(ext, dtype=float).reshape(-1, 2)
        self._dirty = np.ones(len(self.slices), dtype=bool)
        self._circ = np.zeros((len(self.slices), 3))
        self._bvh_init()
        # ---- slots
        nslots = int(params["window"]["slots"])
        self.slot_names = slot_ids(world, nslots)
        self.slots = [self._bind_slot(i, sid) for i, sid in enumerate(self.slot_names)]
        self.active = {}                          # duct index -> slot index
        self.slot_of = [None] * len(self.slots)
        self.rest_count = {}
        self.overflows = 0
        self.max_active = 0
        self.activations = 0
        self.counter = 0
        self.reconfigs = 0
        self.bvh_refits = 0
        self._last_q = {}
        # ---- window centres
        self.centres = self._bind_centres()
        for s in self.slots:
            self._park(s)

    # ------------------------------------------------------------------------------------------ BVH of the world body
    def _bvh_init(self):
        m = self.m
        adr, num = int(m.body_bvhadr[0]), int(m.body_bvhnum[0])
        self.bvh_adr = adr
        child = m.bvh_child.reshape(-1, 2)
        self.bvh_parent = {}
        rel = all((c < 0 or c < num) for c in child[adr:adr + num].reshape(-1))
        for i in range(adr, adr + num):
            for c in child[i]:
                if c >= 0:
                    self.bvh_parent[(adr + c) if rel else c] = i
        self.bvh_child = child
        self.bvh_rel = rel
        nodeid = m.bvh_nodeid
        self.leaf_of = {}
        for i in range(adr, adr + num):
            g = int(nodeid[i])
            if g >= 0:
                self.leaf_of[g] = i
        aabb = m.bvh_aabb.reshape(-1, 6)
        # MuJoCo puts only geoms that can collide into a body's hierarchy: the LiDAR cylinders (no contact bits) are
        # tested by the ray functions directly and need nothing here (test_s4_window checks that rays find a moved
        # one). Inflate the leaves of the frozen collision capsules (once per model: a second window on the same
        # model finds them inflated already) and refit the ancestors.
        marker = float(self.inflate)
        todo = []
        for g in list(self.col):
            i = self.leaf_of.get(int(g))
            if i is None:
                raise ValueError("geom %d has no leaf in the world body's bounding volume hierarchy" % g)
            c, h = self._geom_box(int(g))
            if np.all(aabb[i][3:5] >= h[:2] + marker - 1e-9):
                continue
            aabb[i][:3] = c
            aabb[i][3:6] = h + np.array([marker, marker, 0.0])
            todo.append(i)
        if todo:
            self._refit_ancestors(todo)

    def _geom_box(self, g):
        """World-frame AABB (centre, half sizes) of a world-body geom from its model pose and local box."""
        m = self.m
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        R = R.reshape(3, 3)
        loc = m.geom_aabb[g]
        return m.geom_pos[g] + np.einsum("ij,j->i", R, loc[:3]), np.einsum("ij,j->i", np.abs(R), loc[3:6])   # BLAS-free

    def _children(self, i):
        out = []
        for c in self.bvh_child[i]:
            if c >= 0:
                out.append((self.bvh_adr + c) if self.bvh_rel else int(c))
        return out

    def _refit_ancestors(self, leaves):
        aabb = self.m.bvh_aabb.reshape(-1, 6)
        nodes = set()
        for i in leaves:
            j = self.bvh_parent.get(i)
            while j is not None:
                nodes.add(j)
                j = self.bvh_parent.get(j)
        # children before parents: deeper nodes first (depth = number of ancestors)
        depth = {}
        for j in nodes:
            k, p = 0, self.bvh_parent.get(j)
            while p is not None:
                k += 1
                p = self.bvh_parent.get(p)
            depth[j] = k
        for j in sorted(nodes, key=lambda x: -depth[x]):
            boxes = [aabb[c] for c in self._children(j)]
            lo = np.min([b[:3] - b[3:] for b in boxes], axis=0)
            hi = np.max([b[:3] + b[3:] for b in boxes], axis=0)
            aabb[j][:3] = 0.5 * (lo + hi)
            aabb[j][3:6] = 0.5 * (hi - lo)

    def _contain(self, geoms):
        """After frozen-layer geoms moved: refit the leaves (and ancestors) of those that left their leaf box."""
        aabb = self.m.bvh_aabb.reshape(-1, 6)
        bad = []
        for g in geoms:
            i = self.leaf_of.get(int(g))
            if i is None:                         # not in the hierarchy (a LiDAR cylinder)
                continue
            c, h = self._geom_box(int(g))
            if np.any(c - h < aabb[i][:3] - aabb[i][3:] - 1e-12) or np.any(c + h > aabb[i][:3] + aabb[i][3:] + 1e-12):
                aabb[i][:3] = c
                aabb[i][3:6] = h + np.array([self.inflate, self.inflate, 0.0])
                bad.append(i)
        if bad:
            self._refit_ancestors(bad)
            self.bvh_refits += len(bad)

    # ------------------------------------------------------------------------------------------ frozen layer
    def _write_frozen(self, idx, poses, which=("col", "lid")):
        """Put the frozen-layer geoms of pieces idx at poses (model geom_pos / geom_quat)."""
        m = self.m
        idx = np.asarray(idx, dtype=int)
        p = np.asarray(poses, dtype=float).reshape(-1, 3)
        cy, sy = np.cos(p[:, 2]), np.sin(p[:, 2])
        # a geom made with fromto along x has its z axis along the piece axis: quat = yaw * (rotation y by 90 deg)
        h = 0.5 * p[:, 2]
        qw, qz = np.cos(h), np.sin(h)
        r2 = math.sqrt(0.5)
        quat = np.stack([qw * r2, -qz * r2, qw * r2, qz * r2], axis=1)
        # the same rotation as a matrix (geom frame: z along the axis, x vertical-ish): columns x, y, z
        mat = np.zeros((len(idx), 9))
        mat[:, 2], mat[:, 5] = cy, sy                       # z axis = (cos yaw, sin yaw, 0)
        mat[:, 1], mat[:, 4] = -sy, cy                      # y axis = (-sin yaw, cos yaw, 0)
        mat[:, 6] = -1.0                                    # x axis = (0, 0, -1)
        moved = []
        d = self.d
        if "col" in which:
            mid = 0.5 * (self.ext[idx, 0] + self.ext[idx, 1])
            pos = np.stack([p[:, 0] + cy * mid, p[:, 1] + sy * mid, np.full(len(idx), K.DUCT_RADIUS_M)], axis=1)
            g = self.col[idx]
            m.geom_pos[g] = pos
            m.geom_quat[g] = quat
            # world-body geoms are "static": with the sleep flag on MuJoCo never recomputes their world poses, so
            # they are written here as well (with the flag off the kinematics recomputes the same values)
            d.geom_xpos[g] = pos
            d.geom_xmat[g] = mat
            moved += list(g)
        if "lid" in which:
            pos = np.stack([p[:, 0], p[:, 1], np.full(len(idx), K.DUCT_RADIUS_M)], axis=1)
            g = self.lid[idx]
            m.geom_pos[g] = pos
            m.geom_quat[g] = quat
            d.geom_xpos[g] = pos
            d.geom_xmat[g] = mat
        if moved:
            self._contain(moved)

    # ------------------------------------------------------------------------------------------ slots
    def _bind_slot(self, i, sid):
        m = self.m
        s = _Slot()
        s.index, s.id = i, sid
        bodies = []
        k = 0
        while True:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, K.duct_body(sid, k))
            if b < 0:
                break
            bodies.append(b)
            k += 1
        if not bodies:
            raise ValueError("slot chain %r is not in the model (built with fewer window slots?)" % sid)
        s.n_max = len(bodies)
        s.body = np.array(bodies, dtype=int)
        s.qadr = np.array([int(m.jnt_qposadr[m.body_jntadr[b]]) for b in bodies], dtype=int)
        s.vadr = np.array([int(m.jnt_dofadr[m.body_jntadr[b]]) for b in bodies], dtype=int)
        s.qidx = (s.qadr[:, None] + np.arange(3)[None, :])
        s.vidx = (s.vadr[:, None] + np.arange(3)[None, :])
        s.col = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, K.duct_geom(sid, k, "col")) for k in range(s.n_max)])
        s.pin = np.array([-1] + [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "%s/pin" % K.duct_body(sid, k))
                                 for k in range(1, s.n_max)])
        s.bend = np.array([-1] + [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "%s/bend" % K.duct_body(sid, k))
                                  for k in range(1, s.n_max)])
        s.trees = np.array(sorted({int(m.body_treeid[b]) for b in bodies}), dtype=int)
        s.cfg = (s.n_max, round(2.0 * float(m.geom_size[s.col[0]][1]) + 0.0, 9))      # (pieces, capsule length of piece 0)
        # the build's piece length: piece 1's frame is one length further along x than piece 0's
        s.c = (math.hypot(float(m.body_pos[bodies[1]][0] - m.body_pos[bodies[0]][0]),
                          float(m.body_pos[bodies[1]][1] - m.body_pos[bodies[0]][1])) if s.n_max > 1 else 0.0)   # BLAS-free
        s.n = s.n_max
        s.duct = None
        s.frame = np.zeros((s.n_max, 3))
        return s

    def _set_bits(self, geoms, on):
        ct, ca = self.bits if on else (0, 0)
        self.m.geom_contype[geoms] = ct
        self.m.geom_conaffinity[geoms] = ca

    def _park(self, s):
        m, d = self.m, self.d
        self._set_bits(s.col, False)
        m.dof_frictionloss[s.vidx.reshape(-1)] = 0.0
        for e in list(s.pin[1:]) + list(s.bend[1:]):
            d.eq_active[e] = 0
        for k, b in enumerate(s.body):
            m.body_pos[b] = [self.park[0] + (k + 0.5) * max(s.c, 0.5), self.park[1] + 2.0 * s.index, K.DUCT_RADIUS_M]
            m.body_quat[b] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[s.qidx.reshape(-1)] = 0.0
        d.qvel[s.vidx.reshape(-1)] = 0.0
        if self.slot_sleep:
            # wake it so that the kinematics moves its bodies to the parking place; it falls asleep there
            d.tree_asleep[s.trees] = -(mujoco.mjMINAWAKE + 1)
        s.duct = None

    def _configure(self, s, n, c):
        """Give slot s the geometry and inertia of an n-piece duct with pieces of length c (only when it differs)."""
        if s.n == n and abs(s.c - c) < 1e-9:
            return
        m = self.m
        pc = piece_constants(self.p, c)
        mass, inertia = pc["mass_kg"], pc["inertia_kgm2"]
        for k in range(s.n_max):
            b = s.body[k]
            if k >= n:
                continue
            m.body_mass[b] = mass
            m.body_inertia[b] = inertia
            m.body_subtreemass[b] = mass
            m.body_invweight0[b] = [2.0 / (3.0 * mass), 1.0 / (3.0 * inertia[2])]
            m.dof_invweight0[s.vidx[k]] = [1.0 / mass, 1.0 / mass, 1.0 / inertia[2]]
            m.dof_M0[s.vidx[k]] = [mass, mass, inertia[2]]
            lo, hi = geom.collision_extent(k, n, c)
            g = s.col[k]
            hl = 0.5 * (hi - lo)
            m.geom_size[g] = [K.DUCT_RADIUS_M, hl, 0.0]
            m.geom_pos[g] = [0.5 * (lo + hi), 0.0, 0.0]
            m.geom_rbound[g] = hl + K.DUCT_RADIUS_M
            m.geom_aabb[g] = [0.0, 0.0, 0.0, K.DUCT_RADIUS_M, K.DUCT_RADIUS_M, hl + K.DUCT_RADIUS_M]
            bv = int(m.body_bvhadr[b])
            m.bvh_aabb.reshape(-1, 6)[bv] = [0.0, 0.0, 0.0, K.DUCT_RADIUS_M, K.DUCT_RADIUS_M, hl + K.DUCT_RADIUS_M]
            if int(m.body_bvhnum[b]) != 1:
                raise ValueError("slot piece body %d must carry exactly one geom" % b)
            if k >= 1:
                e = s.bend[k]
                m.eq_solref[e] = bend_solref(self.p, inertia[2], inertia[2], pc["bend_stiffness_nm_per_rad"],
                                             pc["bend_damping_nms_per_rad"])
        s.n, s.c = n, c
        self.reconfigs += 1

    def _activate(self, j, s):
        m, d = self.m, self.d
        duct = self.ducts[j]
        sl = self.slices[j]
        n, c = sl.stop - sl.start, float(duct["piece_length_m"])
        self._configure(s, n, c)
        pc = piece_constants(self.p, c)
        cur = self.pose[sl]
        lay = self.world_poses[sl]
        for k in range(s.n_max):
            b = s.body[k]
            if k < n:
                x, y, yaw = cur[k]
                m.body_pos[b] = [x, y, K.DUCT_RADIUS_M]
                m.body_quat[b] = geom.yaw_quat(yaw)
                s.frame[k] = cur[k]
                m.dof_frictionloss[s.vidx[k]] = pc["frictionloss"]
                if k >= 1:
                    # pin: the head of piece k-1 and the point of piece k that is there now
                    pa, pb = cur[k - 1], cur[k]
                    head = np.array([pa[0] + 0.5 * c * math.cos(pa[2]), pa[1] + 0.5 * c * math.sin(pa[2])])
                    cb, sb = math.cos(pb[2]), math.sin(pb[2])
                    rel = head - pb[:2]
                    m.eq_data[s.pin[k]][:6] = [0.5 * c, 0.0, 0.0, cb * rel[0] + sb * rel[1], -sb * rel[0] + cb * rel[1], 0.0]
                    # bend: rest shape = the laid relative yaw, whatever the frames are now
                    m.eq_data[s.bend[k]][:5] = [geom.wrap((lay[k, 2] - lay[k - 1, 2]) - (pb[2] - pa[2])), 1.0, 0.0, 0.0, 0.0]
                    d.eq_active[s.pin[k]] = 1
                    d.eq_active[s.bend[k]] = 1
            else:
                m.dof_frictionloss[s.vidx[k]] = 0.0
                if k >= 1:
                    d.eq_active[s.pin[k]] = 0
                    d.eq_active[s.bend[k]] = 0
        d.qpos[s.qidx.reshape(-1)] = 0.0
        d.qvel[s.vidx.reshape(-1)] = 0.0
        self._set_bits(s.col[:n], True)
        self._set_bits(s.col[n:], False)
        self._set_bits(self.col[sl], False)
        if self.slot_sleep:
            d.tree_asleep[s.trees] = -(mujoco.mjMINAWAKE + 1)
        s.duct = j
        self.active[j] = s.index
        self.slot_of[s.index] = j
        self.rest_count[j] = 0
        self._last_q[j] = np.zeros(3 * n)
        self.activations += 1
        self.max_active = max(self.max_active, len(self.active))

    def _deactivate(self, j):
        s = self.slots[self.active.pop(j)]
        sl = self.slices[j]
        self.pose[sl] = self._slot_poses(s, sl.stop - sl.start)
        self._write_frozen(np.arange(sl.start, sl.stop), self.pose[sl])
        self._dirty[j] = True
        self._set_bits(self.col[sl], True)
        self.slot_of[s.index] = None
        self.rest_count.pop(j, None)
        self._last_q.pop(j, None)
        self._park(s)

    def _slot_poses(self, s, n):
        q = self.d.qpos[s.qidx[:n].reshape(-1)].reshape(-1, 3)
        return geom.pose_from_q(s.frame[:n], q)

    # ------------------------------------------------------------------------------------------ centres
    def _bind_centres(self):
        """Window centres: ("free", qpos adr, dof adr, footprint offset, radius) for bodies on a free joint, ("mocap",
        mocap id, -1, offset, radius) for mocap bodies (a slow car replayed as mocap, the obstacle model's choice)."""
        m = self.m
        out = []
        cars = ["car/chassis"] + ["slow%d/chassis" % i for i in range(len(self.world["truth"]["avoidance"]["slow_cars"]))]
        boxes = [(K.box_body(bx["id"]), 0.5 * math.hypot(bx["size_m"][0], bx["size_m"][1]))
                 for bx in self.world["truth"]["avoidance"]["boxes"]]
        for name, off, r in [(c, CAR_CENTRE_X, CAR_RADIUS) for c in cars] + [(b, 0.0, rb) for b, rb in boxes]:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if b < 0:
                continue
            if int(m.body_mocapid[b]) >= 0:
                out.append(("mocap", int(m.body_mocapid[b]), -1, off, r))
            elif int(m.body_jntnum[b]) >= 1 and int(m.jnt_type[m.body_jntadr[b]]) == int(mujoco.mjtJoint.mjJNT_FREE):
                j = int(m.body_jntadr[b])
                out.append(("free", int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), off, r))
        self._mocap_last = {}
        return out

    def _centre_state(self):
        q, v = self.d.qpos, self.d.qvel
        xy, rad, spd = [], [], []
        for kind, a, va, off, r in self.centres:
            if kind == "free":
                yaw = _quat_yaw(q[a + 3:a + 7]) if off else 0.0
                p = [q[a] + off * math.cos(yaw), q[a + 1] + off * math.sin(yaw)]
                sp = math.hypot(v[va], v[va + 1]) + (r * abs(v[va + 5]) if off else 0.0)
            else:
                mp, mq = self.d.mocap_pos[a], self.d.mocap_quat[a]
                yaw = _quat_yaw(mq) if off else 0.0
                p = [float(mp[0]) + off * math.cos(yaw), float(mp[1]) + off * math.sin(yaw)]
                last = self._mocap_last.get(a)
                sp = 0.0 if last is None else math.hypot(p[0] - last[0], p[1] - last[1]) / (self.every * self.dt)
                self._mocap_last[a] = p
                sp = max(sp, 8.0)              # a replayed body may jump: assume the fastest car
            xy.append(p)
            rad.append(r)
            spd.append(sp)
        return np.array(xy, dtype=float).reshape(-1, 2), np.array(rad), np.array(spd)

    def _circles(self):
        """Bounding circle (centre, radius) of every duct's collision skin, from the present poses (lazy)."""
        if self._dirty.any():
            for j in np.nonzero(self._dirty)[0]:
                sl = self.slices[int(j)]
                p = self.pose[sl]
                o = p[:, :2].mean(axis=0)
                rho = float((np.hypot(p[:, 0] - o[0], p[:, 1] - o[1]) + 0.5 * self.lengths[sl]).max()) + K.DUCT_RADIUS_M
                self._circ[j] = [o[0], o[1], rho]
            self._dirty[:] = False
        return self._circ

    def _clearance(self, cxy, crad, horizon):
        """(n_centres, n_ducts) smallest distance between a centre's circle and a duct's collision skin; ducts
        whose bounding circle is farther than `horizon` from a centre get +inf for it (not computed)."""
        circ = self._circles()
        out = np.full((len(cxy), len(self.slices)), np.inf)
        for i, (x, y) in enumerate(cxy):
            coarse = np.hypot(circ[:, 0] - x, circ[:, 1] - y) - circ[:, 2] - crad[i]
            cand = np.nonzero(coarse < horizon[i])[0]
            if not len(cand):
                continue
            idx = np.concatenate([np.arange(self.slices[j].start, self.slices[j].stop) for j in cand])
            p = self.pose[idx]
            cs, sn = np.cos(p[:, 2]), np.sin(p[:, 2])
            e = self.ext[idx]
            dist = geom.point_segment_distance(x, y, p[:, 0] + cs * e[:, 0], p[:, 1] + sn * e[:, 0],
                                               p[:, 0] + cs * e[:, 1], p[:, 1] + sn * e[:, 1]) - K.DUCT_RADIUS_M - crad[i]
            starts = np.cumsum([0] + [self.slices[j].stop - self.slices[j].start for j in cand[:-1]])
            out[i, cand] = np.minimum.reduceat(dist, starts)
        return out

    # ------------------------------------------------------------------------------------------ the per-step work
    def step(self):
        """Sync active ducts and run the window check every `check_every_steps`; returns the piece indices whose
        poses may have changed. A duct's poses are re-read when one of its joint values moved by more than
        `sync_tol` since the last read (creep of a resting duct does not cost a sync every step)."""
        changed = []
        for j, si in self.active.items():
            s = self.slots[si]
            sl = self.slices[j]
            n = sl.stop - sl.start
            q = self.d.qpos[s.qidx[:n].reshape(-1)]
            if float(np.abs(q - self._last_q[j]).max()) > self.sync_tol:
                self._last_q[j] = q.copy()
                self.pose[sl] = geom.pose_from_q(s.frame[:n], q.reshape(-1, 3))
                self._write_frozen(np.arange(sl.start, sl.stop), self.pose[sl], which=("lid",))
                self._dirty[j] = True
                changed.extend(range(sl.start, sl.stop))
        self.counter += 1
        if self.counter % self.every == 0:
            changed.extend(self._check())
        return np.array(sorted(set(changed)), dtype=int)

    def _check(self):
        if not self.centres:
            return []
        cxy, crad, spd = self._centre_state()
        reach = spd * self.every * self.dt
        clear = self._clearance(cxy, crad, self.margin_off + reach + 0.05)       # (nc, nd)
        need = (clear < (self.margin_on + reach)[:, None]).any(axis=0)
        keep = (clear < (self.margin_off + reach)[:, None]).any(axis=0)
        changed = []
        # rest bookkeeping and deactivation, in world-file order
        for j in sorted(self.active):
            s = self.slots[self.active[j]]
            n = self.slices[j].stop - self.slices[j].start
            vmax = float(np.abs(self.d.qvel[s.vidx[:n].reshape(-1)]).max())
            self.rest_count[j] = self.rest_count[j] + 1 if vmax < self.rest_speed else 0
            if not keep[j] and self.rest_count[j] >= self.rest_steps:
                self._deactivate(j)
                changed.extend(range(self.slices[j].start, self.slices[j].stop))
        for j in np.nonzero(need)[0]:
            j = int(j)
            if j in self.active:
                continue
            free = [s for s in self.slots if s.duct is None]
            if not free:
                self.overflows += 1
                continue
            self._activate(j, free[0])
        return changed

    # ------------------------------------------------------------------------------------------ the tracker's side
    def poses(self, idx=None):
        return self.pose.copy() if idx is None else self.pose[idx].copy()

    def write(self, idx, poses):
        """Put pieces at poses, at rest: an active duct among them is frozen first (the window re-activates it at
        the next check if a centre is near)."""
        idx = np.asarray(idx, dtype=int)
        touched = sorted({int(x) for x in self.tr.duct_of[idx]})
        for j in touched:
            if j in self.active:
                self._deactivate(j)
        self.pose[idx] = np.asarray(poses, dtype=float).reshape(-1, 3)
        self._write_frozen(idx, self.pose[idx])
        self._dirty[touched] = True

    def force_check(self):
        """Run the window check now (after placing a car by hand)."""
        return self._check()

    def n_active_pieces(self):
        return sum(self.slices[j].stop - self.slices[j].start for j in self.active)

    def stats(self):
        return {"active_ducts": len(self.active), "max_active_ducts": self.max_active, "activations": self.activations,
                "overflows": self.overflows, "reconfigurations": self.reconfigs, "bvh_refits": self.bvh_refits,
                "slots": len(self.slots)}
