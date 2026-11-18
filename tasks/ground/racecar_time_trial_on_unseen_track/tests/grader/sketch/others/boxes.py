"""Cardboard boxes of the avoidance test.

    add_boxes(spec, world)                          # into the MjSpec of the variant `avoid`
    boxes = BoxTracker(model, data, world, seed)    # the cell passes seed = default_rng([truth.seeds.putback, 5])
    every physics step: boxes.update() -> largest displacement now (m)
    boxes.displaced(0.05) -> [box ids]; boxes.put_back(ids); boxes.poses(); boxes.set_poses(p, reference)

A box is a free rigid body: one collision box `box/<B>/col` (group 1, category box) and the same shape as LiDAR
target `box/<B>/lidar` (group 2, opaque). Its mass is the world file's `mass_kg`, its inertia that of a thin-walled
box of uniform areal density (an empty cardboard box), centred. Its contacts with the floor, the cars and the ducts
use the box's friction `box_floor_friction` (the geom's priority is above the floor's, so MuJoCo takes the box's
parameters instead of the larger of the two).

Displacement: the largest horizontal distance of the four bottom corners of the box (its body frame)
from where they were when the box was last set down. Put-back: to the world-file pose plus an error drawn from the
stream: a position error uniform in a disc of radius `putback_pos_frac` x err and a yaw error uniform in
+-(1 - putback_pos_frac) x err / (half diagonal), so that no corner is more than `staff.putback_err_m` (err) from its
world-file place. Three uniform numbers per box put back, whatever happens.
"""

import math

import mujoco
import numpy as np

from sketch.examples import constants as K

from .params import load_params

FLOOR_PREFIXES = ("floor", "zone/")


def _yaw_quat(yaw):
    return [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]


def shell_inertia(size, mass):
    """Principal moments of a thin-walled closed box (outer size [a, b, c], uniform areal density) of mass `mass`."""
    a, b, c = (float(v) for v in size)
    area = 2.0 * (a * b + b * c + c * a)
    sig = mass / area

    def moment(p, q, r):            # about the axis along p; q, r the other two sizes
        # faces normal to p (area q r): plate moments (q^2 + r^2)/12; faces normal to q (area p r): r^2/12 + (q/2)^2;
        # faces normal to r (area p q): q^2/12 + (r/2)^2
        return 2.0 * sig * (q * r * (q * q + r * r) / 12.0 + p * r * (r * r / 12.0 + q * q / 4.0)
                            + p * q * (q * q / 12.0 + r * r / 4.0))

    return [moment(a, b, c), moment(b, c, a), moment(c, a, b)]


def _floor_priority(spec):
    pr = [int(g.priority) for g in spec.geoms if g.name and (g.name == "floor" or g.name.startswith("zone/"))]
    return max(pr) if pr else 0


def add_boxes(spec, world, params=None):
    """The boxes of `truth.avoidance.boxes` as free bodies at their world-file poses, resting on the floor."""
    p = load_params(params)
    bp = p["box"]
    prio = max(int(bp["priority"]), _floor_priority(spec) + 1)
    mu = float(p["box_floor_friction"])
    ct, ca = K.CONTACT["box"]
    for bx in world["truth"]["avoidance"]["boxes"]:
        lx, ly, lz = (float(v) for v in bx["size_m"])
        x, y, yaw = (float(v) for v in bx["pose"])
        body = spec.worldbody.add_body(name=K.box_body(bx["id"]), pos=[x, y, 0.5 * lz], quat=_yaw_quat(yaw))
        body.add_freejoint(name=K.box_body(bx["id"]) + "/root")
        body.mass = float(bx["mass_kg"])
        body.inertia = shell_inertia(bx["size_m"], float(bx["mass_kg"]))
        body.ipos = [0.0, 0.0, 0.0]
        body.explicitinertial = True
        half = [0.5 * lx, 0.5 * ly, 0.5 * lz]
        body.add_geom(name=K.box_geom(bx["id"], "col"), type=mujoco.mjtGeom.mjGEOM_BOX, size=half, mass=0.0,
                      group=K.GROUP_COLLISION, contype=ct, conaffinity=ca, priority=prio, friction=[mu, 0.005, 0.0001],
                      condim=int(bp["condim"]), solref=list(bp["solref"]), rgba=list(bp["rgba"]))
        body.add_geom(name=K.box_geom(bx["id"], "lidar"), type=mujoco.mjtGeom.mjGEOM_BOX, size=half, mass=0.0,
                      group=K.GROUP_LIDAR, contype=0, conaffinity=0, rgba=list(bp["rgba"]))


def _rng(seed, world):
    if isinstance(seed, np.random.Generator):
        return seed
    base = world["truth"]["seeds"]["putback"] if seed is None else int(seed)
    return np.random.default_rng([int(base), 5])          # k = 5: the boxes' stream of the putback seed (the cell does the same)


def _quat_mats(q):
    """Rotation matrices (n, 3, 3) of unit quaternions (n, 4) [w, x, y, z]."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1)], axis=1)


class BoxTracker(object):
    """Displacement, put-back and state transfer of the boxes (the DuctTracker interface)."""

    def __init__(self, model, data, world, seed=None, putback_err_m=None, params=None):
        p = load_params(params)
        self.model, self.data = model, data
        self.boxes = world["truth"]["avoidance"]["boxes"]
        self.ids = [b["id"] for b in self.boxes]
        self.n = len(self.boxes)
        self.err = float(p["putback_err_m"] if putback_err_m is None else putback_err_m)
        self.pos_frac = float(p["box"]["putback_pos_frac"])
        self.rng = _rng(seed, world)
        qa, va = [], []
        for b in self.boxes:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, K.box_body(b["id"]) + "/root")
            if jid < 0:
                raise KeyError("box %r is not in the model (add_boxes, variant 'avoid')" % b["id"])
            qa.append(int(model.jnt_qposadr[jid]))
            va.append(int(model.jnt_dofadr[jid]))
        self.qa = np.array(qa, dtype=int)
        self.va = np.array(va, dtype=int)
        self.qidx = (self.qa[:, None] + np.arange(7)[None, :]).reshape(-1)
        self.size = np.array([b["size_m"] for b in self.boxes], dtype=float).reshape(-1, 3)
        h = 0.5 * self.size
        sx = np.array([1.0, -1.0, -1.0, 1.0])
        sy = np.array([1.0, 1.0, -1.0, -1.0])
        self.local = np.stack([sx[None, :] * h[:, :1], sy[None, :] * h[:, 1:2], -np.repeat(h[:, 2:3], 4, axis=1)], axis=2)
        self.half_diag = np.hypot(h[:, 0], h[:, 1])
        self.world = np.array([b["pose"] for b in self.boxes], dtype=float).reshape(-1, 3)
        self.ref = self.world.copy()
        self.ref_corners = self._corners_of(self.ref)
        self.last = np.zeros(self.n)

    # ------------------------------------------------------------------ geometry
    def _corners_of(self, poses):
        """Floor projections (n, 4, 2) of the bottom corners of boxes standing upright at planar poses."""
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        lx, ly = self.local[:, :, 0], self.local[:, :, 1]
        return np.stack([poses[:, :1] + c[:, None] * lx - s[:, None] * ly,
                         poses[:, 1:2] + s[:, None] * lx + c[:, None] * ly], axis=2)

    def corners_now(self):
        """Floor projections (n, 4, 2) of the bottom corners of the boxes in their present state (from qpos)."""
        q = self.data.qpos[self.qidx].reshape(self.n, 7)
        R = _quat_mats(q[:, 3:7])
        w = q[:, None, :3] + np.einsum("nij,nkj->nki", R, self.local)
        return w[:, :, :2]

    def poses(self):
        """Present planar poses (n, 3) [x, y, yaw] in world-file order."""
        q = self.data.qpos[self.qidx].reshape(self.n, 7)
        w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.stack([q[:, 0], q[:, 1], yaw], axis=1)

    def reference_poses(self):
        return self.ref.copy()

    # ------------------------------------------------------------------ the tracker
    def update(self):
        """Largest displacement of any box now (m); called after every physics step, so it is plain Python
        (equal to the vectorised `corners_now` to rounding; a test compares them)."""
        if not self.n:
            return 0.0
        q = self.data.qpos[self.qidx].tolist()
        best = 0.0
        last = self.last
        for i in range(self.n):
            px, py, _, w, x, y, z = q[7 * i:7 * i + 7]
            r00, r01, r02 = 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)
            r10, r11, r12 = 2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)
            dmax = 0.0
            for (lx, ly, lz), (rx, ry) in zip(self._local_py[i], self._ref_py[i]):
                dx = px + r00 * lx + r01 * ly + r02 * lz - rx
                dy = py + r10 * lx + r11 * ly + r12 * lz - ry
                dd = dx * dx + dy * dy
                if dd > dmax:
                    dmax = dd
            dmax = math.sqrt(dmax)
            last[i] = dmax
            if dmax > best:
                best = dmax
        return best

    @property
    def ref_corners(self):
        return self._ref_corners

    @ref_corners.setter
    def ref_corners(self, value):
        self._ref_corners = value
        self._ref_py = [[(float(a), float(b)) for a, b in box] for box in value]
        self._local_py = [[(float(a), float(b), float(c)) for a, b, c in box] for box in self.local]

    def displaced(self, threshold_m):
        return [self.ids[i] for i in range(self.n) if self.last[i] >= threshold_m]

    def put_back(self, boxes=None):
        """Put the listed boxes (ids; default all) back at their world-file pose plus the put-back error; that pose
        is their new reference."""
        todo = set(self.ids if boxes is None else boxes)
        unknown = todo - set(self.ids)
        if unknown:
            raise KeyError("unknown box ids %s" % sorted(unknown))
        for i, bid in enumerate(self.ids):
            if bid not in todo:
                continue
            u = self.rng.random(3)
            r = self.pos_frac * self.err * math.sqrt(float(u[0]))
            a = 2.0 * math.pi * float(u[1])
            dyaw = (1.0 - self.pos_frac) * self.err / float(self.half_diag[i]) * (2.0 * float(u[2]) - 1.0)
            x, y, yaw = self.world[i]
            pose = [x + r * math.cos(a), y + r * math.sin(a), math.atan2(math.sin(yaw + dyaw), math.cos(yaw + dyaw))]
            self._set(i, pose)
            self.ref[i] = pose
            self.last[i] = 0.0
        self.ref_corners = self._corners_of(self.ref)
        mujoco.mj_forward(self.model, self.data)

    def _set(self, i, pose):
        qa, va = int(self.qa[i]), int(self.va[i])
        self.data.qpos[qa:qa + 7] = [pose[0], pose[1], 0.5 * self.size[i, 2]] + _yaw_quat(pose[2])
        self.data.qvel[va:va + 6] = 0.0

    def set_poses(self, poses, reference=None):
        """Set the boxes upright at planar poses (n, 3), at rest; `reference` (n, 3) replaces the poses they were last
        set down at (default: unchanged)."""
        poses = np.asarray(poses, dtype=float).reshape(-1, 3)
        for i in range(self.n):
            self._set(i, poses[i])
        if reference is not None:
            self.ref = np.asarray(reference, dtype=float).reshape(-1, 3).copy()
            self.ref_corners = self._corners_of(self.ref)
        mujoco.mj_forward(self.model, self.data)

    def total_displacement(self):
        """Sum over the boxes of the horizontal distance of their centre from the world-file pose (reported only)."""
        if not self.n:
            return 0.0
        return float(np.linalg.norm(self.poses()[:, :2] - self.world[:, :2], axis=1).sum())
