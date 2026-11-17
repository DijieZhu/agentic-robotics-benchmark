"""The privileged reference cleaner (the oracle's route; grader-side only).

It drives the robot through the cell's own controller (the same `command` / `step` a program uses) but reads the true
object poses, the lids' and the faucet's joints, the grease film, the sponge's state and the crumbs. The job, in the
order a cleaner works:
  1. open both bin lids;
  2. clear the coffee table: the can and the bottle into the recycling bin, the napkin and the apple core into the
     general-waste bin;
  3. clear the dining table: the plates and the glasses onto the dish tray by the sink, the can into the recycling
     bin, the napkins into the general-waste bin, the two books flat into the bookshelf's target compartment;
  4. degrease the table: the sponge from its tray (from the stance where the arm reaches it in its most natural
     posture), the water on, soap from the pump, wipe the stain in rows from both long sides of the table, rinsing
     and re-soaping as the sponge loads up, until it is clean; rinse the sponge; push the crumbs left on the table off
     its long edges; rinse it again, back into its tray, the water off;
  5. sweep, group of crumbs by group: the dustpan set down by them, the broom from its clip, strokes into the pan,
     the broom back in its clip; the pan taken again from its lip's side; at the end it is emptied into the
     general-waste bin (tipped towards the robot) and hung back in its clip;
  6. close both lids.

The arms move as a person's would (NaturalIK, posture_cost, Reference.refine_path): every planned arm configuration is
solved with a posture term in the arm's redundancy (the elbow kept below the shoulder and beside or in front of the body,
the joints off their limits, the twisting joints near their middle), and every executed joint path keeps the planned
hand path while its arm configurations are resolved as a person's, from the previous one (the swivel that puts the
elbow down and a little out, the wrist near neutral where the hand's pose allows, the joints kept off their stops); a
grasp is chosen for the carry it leaves (carry_order), a held item is carried in front of the chest in the most natural
of a few postures (canonical_carry), and a free arm hangs at home.
"""

from __future__ import annotations

import heapq
import math
import time

import mujoco
import numpy as np

from . import cleaning as CL
from . import layout as LAYOUT
from . import objects as O
from . import room as RM
from .robot import ARM_HOME, PELVIS_HEIGHT, _quat_err

TICK = 0.01
ROBOT_R = 0.33              # m: the robot's footprint radius for path planning (arms in the carry pose)
HARD_R = 0.20               # m: its hips' half width (the clearance kept near a walk's start and goal)
GRID = 0.05


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _axis_angle(axis, ang):
    """Rotation matrix about the unit `axis` by `ang`."""
    x, y, z = axis
    c, s, C = math.cos(ang), math.sin(ang), 1 - math.cos(ang)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def hand_frame(approach, pinch):
    """Grasp-site rotation (columns x, y, z) from its approach (x) and pinch (y) directions (room frame)."""
    x = np.asarray(approach, float)
    x = x / np.linalg.norm(x)
    y = np.asarray(pinch, float)
    y = y - (y @ x) * x
    y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def level(R):
    """The rotation nearest R with its z axis straight up (an object set flat; its heading kept)."""
    z = np.array([0.0, 0.0, 1.0])
    x = R[:, 0] - (R[:, 0] @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = R[:, 1] - (R[:, 1] @ z) * z
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def tilted(forward_xy, down_deg):
    """A horizontal direction tilted down by `down_deg`."""
    f = np.array([forward_xy[0], forward_xy[1], 0.0])
    f /= np.linalg.norm(f)
    a = math.radians(down_deg)
    return f * math.cos(a) + np.array([0.0, 0.0, -math.sin(a)])


def _rot2(v, a):
    """A horizontal direction (x, y) turned by `a` about the vertical."""
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def geom_lowest(m, d, g):
    """The lowest point (z) of a primitive collider as it is now."""
    t = int(m.geom_type[g])
    c = d.geom_xpos[g]
    R = d.geom_xmat[g].reshape(3, 3)
    sz = m.geom_size[g]
    G = mujoco.mjtGeom
    if t == G.mjGEOM_SPHERE:
        return float(c[2] - sz[0])
    if t == G.mjGEOM_BOX:
        return float(c[2] - np.abs(R[2, :]) @ sz)
    if t in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE):
        ax = R[:, 2]
        rad = sz[0] * math.sqrt(max(0.0, 1.0 - ax[2] ** 2))
        cap = sz[0] if t == G.mjGEOM_CAPSULE else 0.0
        return float(c[2] - abs(ax[2]) * sz[1] - rad - cap * (1.0 if t == G.mjGEOM_CAPSULE else 0.0) * 0.0 - cap)
    return float(c[2] - m.geom_rbound[g])


class Planner:
    """A* on a 5 cm occupancy grid of the room's floor: whatever stands on the floor, inflated by the robot's
    footprint radius (the rug and the drawn decor are no obstacles)."""

    def __init__(self, radius=ROBOT_R):
        self.nx, self.ny = int(RM.ROOM_X / GRID), int(RM.ROOM_Y / GRID)
        self.occ = np.zeros((self.nx, self.ny), bool)
        boxes = []
        K = RM.KITCHEN
        boxes.append((K["x0"], K["y0"], K["top_x1"], K["y1"]))
        for yc in RM.BIN_SLOTS_Y:
            boxes.append((RM.BIN["x0"], yc - RM.BIN["half_y"], RM.BIN["x1"] + 0.03, yc + RM.BIN["half_y"]))
        D = RM.DINING
        (dx, dy), (lx, ly) = D["center"], D["size"]
        boxes.append((dx - lx / 2, dy - ly / 2, dx + lx / 2, dy + ly / 2))
        for ch in RM.CHAIRS.values():
            # (8 cm round a chair: its back stands at elbow height, and a walking robot drifts about that much
            # off its planned line)
            x, y = ch["center"]
            w, l = RM.CHAIR["seat"]
            boxes.append((x - w / 2 - 0.08, y - l / 2 - 0.08, x + w / 2 + 0.08, y + l / 2 + 0.08))
        C = RM.COFFEE
        (cx, cy), (cw, cl) = C["center"], C["size"]
        boxes.append((cx - cw / 2, cy - cl / 2, cx + cw / 2, cy + cl / 2))
        boxes.append(tuple(RM.COUCH["footprint"]))
        S = RM.SHELF
        boxes.append((S["x0"], S["y0"], S["x1"], S["y1"]))
        # the rail itself (the tools, hanging on it or not, are walk obstacles of their own: Reference.loose_tools)
        boxes.append((RM.TOOL_RAIL["x0"], 0.0, RM.TOOL_RAIL["x1"], RM.TOOL_RAIL["depth"] + 0.02))
        pc, pr = RM.PLANT["foliage"]["center"], RM.PLANT["foliage"]["radius"]
        boxes.append((pc[0] - pr, pc[1] - pr, pc[0] + pr, pc[1] + pr))
        self.boxes = boxes
        self.radius = radius
        self.occ = self._inflate(boxes, radius)
        # (near the start and the goal only the body's own clearance applies, its hips' half width: the footprint
        # radius and a load's wider one would wall in a goal by a fixture; the stance checks see the real body)
        self.hard = self._inflate(boxes, min(radius, HARD_R))

    def _inflate(self, boxes, r):
        xs = (np.arange(self.nx) + 0.5) * GRID
        ys = (np.arange(self.ny) + 0.5) * GRID
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        occ = (X < r) | (Y < r) | (X > RM.ROOM_X - r) | (Y > RM.ROOM_Y - r)
        for x0, y0, x1, y1 in boxes:
            ddx = np.maximum(np.maximum(x0 - X, X - x1), 0.0)
            ddy = np.maximum(np.maximum(y0 - Y, Y - y1), 0.0)
            occ |= (ddx * ddx + ddy * ddy) < r * r
        return occ

    def cell(self, x, y):
        return (int(np.clip(x / GRID, 0, self.nx - 1)), int(np.clip(y / GRID, 0, self.ny - 1)))

    def free_line(self, a, b):
        n = int(max(2, math.hypot(b[0] - a[0], b[1] - a[1]) / (GRID / 2)))
        for t in np.linspace(0, 1, n):
            i, j = self.cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if self.occ[i, j]:
                return False
        return True

    def path(self, start, goal, extra=()):
        """Waypoints from start to goal (both (x, y)); start / goal cells inside the inflated obstacles are allowed
        (the robot starts or ends next to a fixture), the rest of the path stays clear. `extra`: boxes of what lies on
        the floor now (a tool set down), inflated like the rest."""
        s, g = self.cell(*start), self.cell(*goal)
        occ = self.occ.copy()
        hard = self.hard.copy()
        if extra:
            occ |= self._inflate(extra, self.radius) & ~self._inflate([], self.radius)
            hard |= self._inflate(extra, min(self.radius, HARD_R)) & ~self._inflate([], min(self.radius, HARD_R))
        # within 0.6 m of the start and the goal only the body's own clearance counts (and the start / goal cells
        # themselves are free: the robot starts or ends next to a fixture)
        I, J = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        ds, dg = np.hypot(I - s[0], J - s[1]), np.hypot(I - g[0], J - g[1])
        occ = np.where((ds <= 12) | (dg <= 12), hard, occ)
        occ[(ds <= 7) | (dg <= 7)] = False
        openq = [(0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        nb = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414),
              (-1, -1, 1.414)]
        while openq:
            _, c = heapq.heappop(openq)
            if c == g:
                break
            for di, dj, w in nb:
                n = (c[0] + di, c[1] + dj)
                if not (0 <= n[0] < self.nx and 0 <= n[1] < self.ny) or occ[n]:
                    continue
                nc = cost[c] + w
                if nc < cost.get(n, 1e18):
                    cost[n] = nc
                    came[n] = c
                    heapq.heappush(openq, (nc + math.hypot(g[0] - n[0], g[1] - n[1]), n))
        if g not in came:
            return None
        cells = []
        c = g
        while c is not None:
            cells.append(c)
            c = came[c]
        cells.reverse()
        pts = [((i + 0.5) * GRID, (j + 0.5) * GRID) for i, j in cells]
        pts[0], pts[-1] = tuple(start), tuple(goal)
        # shortcut: keep a waypoint only where the straight line to the next kept one is blocked
        out = [pts[0]]
        k = 0
        while k < len(pts) - 1:
            j = len(pts) - 1
            while j > k + 1 and not self._clear(pts[k], pts[j], occ):
                j -= 1
            out.append(pts[j])
            k = j
        return out[1:]

    def _clear(self, a, b, occ):
        n = int(max(2, math.hypot(b[0] - a[0], b[1] - a[1]) / (GRID / 2)))
        for t in np.linspace(0, 1, n):
            i, j = self.cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if occ[i, j]:
                return False
        return True


HOME_Q = {"left": [0.30, 0.25, 0.0, 1.10, 0.0, 0.0, 0.0], "right": [0.30, -0.25, 0.0, 1.10, 0.0, 0.0, 0.0]}


def rows(entries):
    """The joint rows of a planned path, in order (an entry is one row of 7 or several)."""
    for e in entries:
        e = np.asarray(e, float)
        if e.ndim == 1:
            yield e
        else:
            yield from e


def last(entry):
    """A planned entry's own joints (its last row)."""
    e = np.asarray(entry, float)
    return e if e.ndim == 1 else e[-1]


def _mirror(q):
    """A right-arm posture as the left arm's (the roll and yaw joints change sign)."""
    return [q[0], -q[1], -q[2], q[3], -q[4], q[5], -q[6]]


# natural reaching postures (right arm) that seed the IK when the straight approach line fails
_REACH_R = [[-0.30, -0.20, 0.30, 0.30, 0.80, 0.0, 0.0], [-0.30, 0.10, -0.20, -0.20, 1.50, 0.20, -0.80],
            [-0.60, -0.15, 0.0, 0.50, 0.0, 0.0, 0.0], [0.0, -0.30, 0.0, 0.80, -1.20, 0.0, 0.0],
            [-0.90, -0.20, 0.20, 0.90, 0.60, 0.0, 0.0]]
REACH_SEEDS = {"right": _REACH_R, "left": [_mirror(q) for q in _REACH_R]}


class NaturalIK:
    """The arm IK of the reference's plans: the cell's damped least squares (robot.ArmIK: the grasp site's position and
    rotation by the arm's 7 joints, the base, the waist and the legs as given), plus a posture term descended in the
    arm's null space, the redundancy the target leaves (the elbow's swivel about the shoulder-wrist line; with a
    position-only target the wrist as well). The term (terms) resolves the redundancy the way a person holds the arm:
    the swivel angle that puts the elbow down and a little out (the elbow on its circle about the shoulder-wrist line
    pulled towards that direction, always, not only past a threshold), the elbow not above the shoulder, not behind the
    body and not far out beside it, the wrist near its neutral (its sideways bend within 0.3 rad, its flexion within 0.55 rad, its roll within
    1 rad: the joint limits of the robot's wrist are far wider than a person's), the shoulder's twist within 1 rad, the
    elbow neither straightened past straight nor folded tight, every joint 0.35 rad off its limits. A target the arm
    can reach with the elbow up or down is so reached the way a person holds the arm, and posture_cost ranks grasps,
    stances and carries by the same measure. The posture step is capped per iteration, so a path solved waypoint by
    waypoint (each from the previous solution) drifts towards the natural posture smoothly instead of jumping."""

    LIMIT_MARGIN = 0.35        # rad: a joint closer than this to a limit costs
    ELBOW_RAISE = 0.02         # m: the elbow above the shoulder by more than this costs (reaching forward at shoulder
                               # height a person's elbow is level with the shoulder; above it the arm is raised)
    ELBOW_BACK = 0.05          # m: the elbow behind the shoulder (torso frame) beyond this costs
    ELBOW_OUT = 0.14           # m: the elbow out to the side of the shoulder beyond this costs
    # the elbow's natural direction about the shoulder-wrist line (torso frame: forward, out to the arm's side, up):
    # down and a little out and back; for a hand hanging straight below the shoulder, back (the forearm comes forward)
    SWIVEL_DIR = (-0.10, 0.35, -1.0)
    SWIVEL_DIR_HANG = (-1.0, 0.30, 0.0)
    # comfortable joint ranges (rad) beyond which a joint costs (a person's wrist and forearm, the robot's joints):
    # (joint index, half range, scale)
    COMFORT = ((5, 0.30, 0.25),     # wrist pitch: the hand bent sideways (radial / ulnar)
               (6, 0.55, 0.30),     # wrist yaw: the hand bent towards the palm or its back
               (4, 1.00, 0.40),     # wrist roll: the forearm turned
               (2, 1.00, 0.40))     # shoulder yaw: the upper arm twisted
    ELBOW_STRAIGHT = 1.45       # rad: the elbow joint past this is (nearly) straightened through (pi/2 is straight)
    ELBOW_FOLD = -0.70          # rad: below this the elbow is folded tight (0: a right angle)
    # (the swivel dominates the comfort terms: with the hand's pose given, a lower elbow often bends the wrist more, and
    # a raised or winged elbow reads as far more contorted than a bent wrist; the joint limits dominate both, so the
    # swivel never wrings a wrist against its stop)
    W = {"limit": 30.0, "elbow_high": 3.0, "elbow_back": 1.0, "elbow_out": 0.5, "swivel": 8.0, "comfort": 0.6,
         "elbow_straight": 1.5, "elbow_fold": 0.4, "twist": 0.04}
    # the ranking measure's constants (terms_rank)
    RANK_ELBOW_DROP, RANK_ELBOW_BACK, RANK_ELBOW_OUT = 0.08, 0.05, 0.12
    RANK_W = {"limit": 1.0, "elbow_high": 3.0, "elbow_back": 1.0, "elbow_out": 0.5, "twist": 0.08}

    def __init__(self, m, r):
        self.m, self.r = m, r
        self.dk = mujoco.MjData(m)
        self.jp = np.zeros((3, m.nv))
        self.jr = np.zeros((3, m.nv))
        self.je = np.zeros((3, m.nv))
        B = mujoco.mjtObj.mjOBJ_BODY
        self.elbow = {s: mujoco.mj_name2id(m, B, f"{r.name}/{s}_elbow_link") for s in ("left", "right")}
        self.shoulder = {s: mujoco.mj_name2id(m, B, f"{r.name}/{s}_shoulder_pitch_link") for s in ("left", "right")}
        self.wrist = {s: mujoco.mj_name2id(m, B, f"{r.name}/{s}_wrist_pitch_link") for s in ("left", "right")}
        self.wroll = {s: mujoco.mj_name2id(m, B, f"{r.name}/{s}_wrist_roll_link") for s in ("left", "right")}
        self.out = {"left": 1.0, "right": -1.0}

    def natural_elbow(self, side, Rt):
        """The elbow's natural place (torso frame, relative to the shoulder) on its circle about the shoulder-wrist
        line at the present wrist, and the circle's radius (m): (point, radius)."""
        dk = self.dk
        S = dk.xpos[self.shoulder[side]]
        E = Rt.T @ (dk.xpos[self.elbow[side]] - S)
        Wp = Rt.T @ (dk.xpos[self.wrist[side]] - S)
        d = float(np.linalg.norm(Wp))
        if d < 1e-6:
            return E, 0.0
        n = Wp / d
        a, b = float(np.linalg.norm(E)), float(np.linalg.norm(Wp - E))
        x = (a * a - b * b + d * d) / (2.0 * d)
        rho = math.sqrt(max(0.0, a * a - x * x))
        o = self.out[side]
        want = np.array([self.SWIVEL_DIR[0], o * self.SWIVEL_DIR[1], self.SWIVEL_DIR[2]])
        hang = np.array([self.SWIVEL_DIR_HANG[0], o * self.SWIVEL_DIR_HANG[1], self.SWIVEL_DIR_HANG[2]])
        w = want - (want @ n) * n
        w = w + 0.3 * (hang - (hang @ n) * n)
        nw = float(np.linalg.norm(w))
        if nw < 1e-6:
            return E, 0.0
        return x * n + rho * (w / nw), rho

    def terms(self, side, q):
        """The posture term at q (self.dk's kinematics and comPos set) as weighted residuals: (residuals, their
        Jacobian over the arm's 7 joints, {term: cost}); the cost is the sum of the squared residuals."""
        r, dk, m = self.r, self.dk, self.m
        lo, hi = r.arm_lo[side], r.arm_hi[side]
        mu = self.LIMIT_MARGIN
        res, rows_, parts = [], [], {}
        wl = math.sqrt(self.W["limit"]) / mu
        c = 0.0
        for j in range(7):
            a, b = mu - (q[j] - lo[j]), mu - (hi[j] - q[j])
            for v_, sg in ((a, -1.0), (b, 1.0)):
                if v_ > 0:
                    jr = np.zeros(7)
                    jr[j] = sg * wl
                    res.append(wl * v_)
                    rows_.append(jr)
                    c += (wl * v_) ** 2
        parts["limit"] = c
        Rt = dk.xmat[r.torso].reshape(3, 3)
        v = Rt.T @ (dk.xpos[self.elbow[side]] - dk.xpos[self.shoulder[side]])
        mujoco.mj_jacBody(m, dk, self.je, None, self.elbow[side])
        Jv = Rt.T @ self.je[:, r.arm_v[side]]
        for name, e, jrow, scale in (("elbow_high", v[2] - self.ELBOW_RAISE, Jv[2], 0.08),
                                     ("elbow_back", -v[0] - self.ELBOW_BACK, -Jv[0], 0.1),
                                     ("elbow_out", self.out[side] * v[1] - self.ELBOW_OUT, self.out[side] * Jv[1],
                                      0.1)):
            w = math.sqrt(self.W[name]) / scale
            parts[name] = (w * e) ** 2 if e > 0 else 0.0
            if e > 0:
                res.append(w * e)
                rows_.append(w * jrow)
        # the swivel: the elbow pulled towards its natural place on its circle (faded out as the arm straightens,
        # where the swivel no longer moves the elbow)
        Ep, rho = self.natural_elbow(side, Rt)
        fade = min(1.0, rho / 0.06)
        ws = math.sqrt(self.W["swivel"]) / 0.10 * fade
        dv = v - Ep
        parts["swivel"] = float(ws * ws * (dv @ dv))
        if ws > 0:
            for k in range(3):
                res.append(ws * dv[k])
                rows_.append(ws * Jv[k])
        cc = 0.0
        for j, half, scale in self.COMFORT:
            e = abs(q[j]) - half
            if e > 0:
                w = math.sqrt(self.W["comfort"]) / scale
                jr = np.zeros(7)
                jr[j] = w * (1.0 if q[j] > 0 else -1.0)
                res.append(w * e)
                rows_.append(jr)
                cc += (w * e) ** 2
        parts["comfort"] = cc
        for name, e, sg, scale in (("elbow_straight", q[3] - self.ELBOW_STRAIGHT, 1.0, 0.15),
                                   ("elbow_fold", self.ELBOW_FOLD - q[3], -1.0, 0.30)):
            w = math.sqrt(self.W[name]) / scale
            parts[name] = (w * e) ** 2 if e > 0 else 0.0
            if e > 0:
                jr = np.zeros(7)
                jr[3] = w * sg
                res.append(w * e)
                rows_.append(jr)
        wt = math.sqrt(self.W["twist"])
        parts["twist"] = wt * wt * float(q[2] ** 2 + q[4] ** 2 + q[6] ** 2)
        for j in (2, 4, 6):
            jr = np.zeros(7)
            jr[j] = wt
            res.append(wt * q[j])
            rows_.append(jr)
        return np.array(res), np.array(rows_).reshape(-1, 7), parts

    def terms_rank(self, side, q):
        """The ranking measure of a posture (posture_cost): the joints off their limits, the elbow at least 8 cm below
        the shoulder, not behind it or far out beside it, the twisting joints near their middle (residuals, Jacobian,
        {term: cost}). The planner's choices between grasps, carries and stances rank by it, the solver's redundancy
        resolution (terms) is finer."""
        r, dk, m = self.r, self.dk, self.m
        lo, hi = r.arm_lo[side], r.arm_hi[side]
        mu = self.LIMIT_MARGIN
        res, rows_, parts = [], [], {}
        wl = math.sqrt(self.RANK_W["limit"]) / mu
        c = 0.0
        for j in range(7):
            a, b = mu - (q[j] - lo[j]), mu - (hi[j] - q[j])
            for v_, sg in ((a, -1.0), (b, 1.0)):
                if v_ > 0:
                    jr = np.zeros(7)
                    jr[j] = sg * wl
                    res.append(wl * v_)
                    rows_.append(jr)
                    c += (wl * v_) ** 2
        parts["limit"] = c
        Rt = dk.xmat[r.torso].reshape(3, 3)
        v = Rt.T @ (dk.xpos[self.elbow[side]] - dk.xpos[self.shoulder[side]])
        mujoco.mj_jacBody(m, dk, self.je, None, self.elbow[side])
        Jv = Rt.T @ self.je[:, r.arm_v[side]]
        for name, e, jrow, scale in (("elbow_high", v[2] + self.RANK_ELBOW_DROP, Jv[2], self.RANK_ELBOW_DROP),
                                     ("elbow_back", -v[0] - self.RANK_ELBOW_BACK, -Jv[0], 0.1),
                                     ("elbow_out", self.out[side] * v[1] - self.RANK_ELBOW_OUT, self.out[side] * Jv[1],
                                      0.1)):
            w = math.sqrt(self.RANK_W[name]) / scale
            parts[name] = (w * e) ** 2 if e > 0 else 0.0
            if e > 0:
                res.append(w * e)
                rows_.append(w * jrow)
        wt = math.sqrt(self.RANK_W["twist"])
        parts["twist"] = wt * wt * float(q[2] ** 2 + q[4] ** 2)
        for j in (2, 4):
            jr = np.zeros(7)
            jr[j] = wt
            res.append(wt * q[j])
            rows_.append(jr)
        return np.array(res), np.array(rows_).reshape(-1, 7), parts

    # the human posture measure (round 5: posture_cost, by which every choice of the planner ranks): what a person's
    # arm does not do, as squared hinges past a comfortable range, the quantities of the review's posture measure
    # (tests/review/posture.md): (joint, half range, scale) of the wrist (sideways bend, flexion, the forearm's turn)
    H_WRIST = ((5, 0.30, 0.22), (6, 0.45, 0.22), (4, 1.05, 0.25))
    H_TWIST = (1.00, 0.25)              # rad: the shoulder's twist
    H_ELBOW_DROP = 0.09                 # m: the elbow at least this far below the shoulder (torso frame)
    H_SWIVEL = (30.0, 15.0)             # deg: the elbow off its natural direction about the shoulder-wrist line
    H_LIMIT = 0.25                      # rad: every joint this far off its limits

    def human(self, side, q):
        """{term: cost} of arm joints q (self.dk's kinematics set; 0 for a posture a person takes without effort, about
        1 at the edge of the comfortable range, 4 and more where the review's measure flags it)."""
        r, dk = self.r, self.dk
        q = np.asarray(q, float)
        parts = {}
        lo, hi = r.arm_lo[side], r.arm_hi[side]
        marg = np.minimum(q - lo, hi - q)
        parts["limit"] = float(np.sum((np.maximum(0.0, self.H_LIMIT - marg) / 0.10) ** 2))
        parts["wrist"] = float(sum((max(0.0, abs(q[j]) - h) / s) ** 2 for j, h, s in self.H_WRIST))
        parts["twist"] = (max(0.0, abs(q[2]) - self.H_TWIST[0]) / self.H_TWIST[1]) ** 2
        Rt = dk.xmat[r.torso].reshape(3, 3)
        S = Rt.T @ dk.xpos[self.shoulder[side]]
        E = Rt.T @ dk.xpos[self.elbow[side]]
        Wr = Rt.T @ dk.xpos[self.wroll[side]]
        v = E - S
        o = self.out[side]
        parts["elbow_high"] = (max(0.0, v[2] + self.H_ELBOW_DROP) / 0.04) ** 2
        parts["elbow_out"] = (max(0.0, o * v[1] - 0.13) / 0.05) ** 2
        parts["elbow_back"] = (max(0.0, -v[0] - 0.06) / 0.05) ** 2
        n = Wr - S
        dn = float(np.linalg.norm(n))
        sw = 0.0
        if dn > 1e-6:
            n = n / dn
            e = v - (v @ n) * n
            want = np.array([0.0, 0.3 * o, -1.0])
            w = want - (want @ n) * n
            if np.linalg.norm(w) < 0.2:
                want = np.array([0.0, o, -0.3])
                w = want - (want @ n) * n
            a_, b_ = float(np.linalg.norm(v)), float(np.linalg.norm(Wr - E))
            bend = 180.0 - math.degrees(math.acos(float(np.clip((a_ * a_ + b_ * b_ - dn * dn) / (2 * a_ * b_), -1, 1))))
            if bend > 20.0 and np.linalg.norm(e) > 1e-6:
                dev = math.degrees(math.acos(float(np.clip((e / np.linalg.norm(e)) @ (w / np.linalg.norm(w)), -1, 1))))
                sw = (max(0.0, dev - self.H_SWIVEL[0]) / self.H_SWIVEL[1]) ** 2 * min(1.0, (bend - 20.0) / 15.0)
        parts["swivel"] = sw
        parts["hyperext"] = (max(0.0, q[3] - 1.50) / 0.08) ** 2
        return parts

    def cost(self, qpos, side, q):
        """The posture cost of arm joints q (the rest of the robot as in qpos): the human measure (human)."""
        dk = self.dk
        dk.qpos[:] = qpos
        dk.qpos[self.r.arm_q[side]] = q
        mujoco.mj_kinematics(self.m, dk)
        return sum(self.human(side, np.asarray(q, float)).values())

    def cost_rank(self, qpos, side, q):
        """The round-3 ranking measure (terms_rank) of arm joints q."""
        dk = self.dk
        dk.qpos[:] = qpos
        dk.qpos[self.r.arm_q[side]] = q
        mujoco.mj_kinematics(self.m, dk)
        mujoco.mj_comPos(self.m, dk)
        return sum(self.terms_rank(side, np.asarray(q, float))[2].values())

    def solve(self, qpos, side, pos, quat, q_init, iters=30, damping=0.05, w_rot=0.35, null_step=0.04, null_total=None,
              w_home=None, natural=False):
        """Arm joints putting `side`'s grasp site at pos (world) with orientation quat (wxyz; None: position only),
        the posture term descended in the null space (at most null_step rad per iteration and, if given, null_total
        rad in all: a path's waypoint so changes the arm's posture only gradually). Returns (q, pos_err_m,
        rot_err_rad). (w_home: ignored; the posture term replaces the cell IK's pull towards the home posture.)"""
        m, r, dk = self.m, self.r, self.dk
        dk.qpos[:] = qpos
        q = np.array(q_init, dtype=float)
        lo, hi = r.arm_lo[side] + 1e-3, r.arm_hi[side] - 1e-3
        q = np.clip(q, lo, hi)
        sid = r.grasp_site[side]
        pos = np.asarray(pos, dtype=float)
        quat = None if quat is None else np.asarray(quat, dtype=float) / np.linalg.norm(quat)
        ep, er = np.zeros(3), np.zeros(3)
        budget = math.inf if null_total is None else float(null_total)
        for it in range(iters + 1):
            dk.qpos[r.arm_q[side]] = q
            mujoco.mj_kinematics(m, dk)
            ep = pos - dk.site_xpos[sid]
            if quat is not None:
                cur = np.zeros(4)
                mujoco.mju_mat2Quat(cur, dk.site_xmat[sid])
                er = _quat_err(quat, cur)
            if it == iters:
                break
            mujoco.mj_comPos(m, dk)
            mujoco.mj_jacSite(m, dk, self.jp, self.jr, sid)
            Jp = self.jp[:, r.arm_v[side]]
            if quat is not None:
                J0 = np.vstack([Jp, w_rot * self.jr[:, r.arm_v[side]]])
                e = np.concatenate([ep, w_rot * er])
            else:
                J0, e = Jp, ep
            # (the posture term only once the target is nearly reached: far from it the step is the cell IK's own,
            # so whatever the cell IK reaches this reaches as well)
            near = np.linalg.norm(ep) < 0.02 and np.linalg.norm(er) < 0.2
            r2, J2, _ = (self.terms if natural else self.terms_rank)(side, q)
            free = np.ones(7, bool)
            for _pass in range(3):
                # a joint at its limit that the step would push further is held (its column dropped) and the step
                # solved again with the others
                J = J0 * free
                JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
                Jpinv = J.T @ np.linalg.solve(JJt, np.eye(J.shape[0]))
                d1 = Jpinv @ e
                # the posture residuals as a second task in the target's null space (task priority, damped), through
                # the exact projector onto that null space (a damped one would let the posture step disturb the
                # target, and the two then settle millimetres off it)
                _, sv, Vt = np.linalg.svd(J)
                Vr = Vt[:int(np.sum(sv > 1e-3))]
                N = (np.eye(7) - Vr.T @ Vr) * free[:, None] * free[None, :]
                J2N = J2 @ N
                dn = N @ (J2N.T @ np.linalg.solve(J2N @ J2N.T + 0.1 ** 2 * np.eye(len(r2)), -r2 - J2 @ d1)) if near \
                    else np.zeros(7)
                sn = float(np.max(np.abs(dn))) if near else 1.0
                cap = min(null_step, budget)
                if near and sn > cap:
                    dn *= cap / sn if sn > 0 else 0.0
                dq = d1 + dn
                step = np.max(np.abs(dq))
                if step > 0.35:
                    dq *= 0.35 / step
                stuck = free & (((q <= lo + 1e-6) & (dq < 0)) | ((q >= hi - 1e-6) & (dq > 0)))
                if not stuck.any():
                    break
                free &= ~stuck
            if near:
                budget -= min(sn, cap)
            if near and np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3 and (sn < 1e-3 or budget <= 0.0):
                break
            q = np.clip(q + dq, lo, hi)
        return q, float(np.linalg.norm(ep)), float(np.linalg.norm(er))


class Reference:
    def __init__(self, cell, log=None):
        self.plan_stats = []          # (checks, wall s, found) per stance search
        self.c = cell
        self.m, self.d = cell.m, cell.d
        self.log = log or (lambda s: None)
        self.planner = Planner()
        self.carry_planner = Planner(radius=0.42)
        self.planners = {0.42: self.carry_planner}
        self.held = {"left": None, "right": None}
        self.grip_rel = {}
        self.notes = []
        self.stance = Stance(cell)
        self.stance.grip_rels = self.grip_rel        # (shared: the start check places what the hands hold)
        self.low_spot = None
        self.pushed = False
        self.yaw_captured = None
        self.dy_chosen = 0.0
        self.carry_lift = {}          # held item -> m its carry point is raised (go_put: lifted before leaning in)
        self.dest_hint = None         # (x, y) of where the robot is going next (carry_pose: the carry height)

    def run(self):
        """The whole job (run_job)."""
        return run_job(self)

    # ============================================================================================ primitives
    def step(self, n=1):
        return self.c.step(int(n))

    HIGH_NEAR = 1.2       # (round 5) m: a dish or the sponge raised for the counter only this near its stance (0.7 m:
    #                       a plate carried at the waist from the dining table's south side caught the chair's back as
    #                       the robot turned for the counter and slipped from the hand)

    def dest_counter(self):
        """Is the robot on its way to the kitchen counter (dest_hint)?"""
        if self.dest_hint is None:
            return False
        x, y = self.dest_hint[0], self.dest_hint[1]
        return x < 1.25 and 1.0 < y < 3.6

    def dest_high(self):
        """On its way to the kitchen counter (dest_hint) and near it (round 5: HIGH_NEAR)? A held dish or the sponge is
        then carried just clear of the counter (carry_pose), not at the waist. (Carried so all the way from the dining
        table, a glass rode on a straight arm at shoulder height for 25-30 s a trip.)"""
        if not self.dest_counter():
            return False
        if getattr(self, "_raised_goal", None) == (round(float(self.dest_hint[0]), 3), round(float(self.dest_hint[1]), 3)):
            return True                  # (raised on the way there: walk)
        b = self.base()
        return math.hypot(b[0] - self.dest_hint[0], b[1] - self.dest_hint[1]) < self.HIGH_NEAR

    def out_of_time(self, margin=5.0):
        return self.c.done or self.c.sim_time > self.c.sim_cap - margin

    def base(self):
        r = self.c.robot
        return np.array(self.d.qpos[r.base_q[:3]], float)

    def site(self, side):
        sid = self.c.robot.grasp_site[side]
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

    def pose(self, name):
        return self.c.pose(name)

    def _hold_cmd(self, cmd, held_tgt, h, pitch):
        """(Reference.posture's hold) the arm joints that keep each held hand at its room pose with the body at
        (h, pitch), added to the command."""
        st = self.stance
        b = self.base()
        for sd, (ps, Rs) in held_tgt.items():
            st.set_state((b[0], b[1], b[2]), h, pitch)
            qc = np.asarray(self.c.ctrl.arm_cmd[sd], float)
            q, ep, er = st.nik.solve(st.chk.qpos, sd, ps, _quat(Rs), qc, iters=15, null_total=0.04, natural=True)
            if ep < 0.01 and er < 0.05:
                cmd[f"{sd}_arm"] = {"q": [float(v) for v in q]}
        return cmd

    def posture(self, h, pitch, timeout=4.0, hold=()):
        """Stand height and waist pitch, waiting until the body is there (a waist still bending when the hand closes
        on an object would press it into what it lies on). `hold` (round 5): the hands of these sides that hold
        something keep their place in the room while the body bends or squats (as a person leans in over a counter
        with a plate in hand; with its arm's joints fixed the item would swing down with the torso)."""
        r, ctrl = self.c.robot, self.c.ctrl
        # a smooth (cosine) ramp below the controller's rates (its own ramps start and stop at full speed, which
        # shakes a long tool in the hand)
        h0, p0 = float(ctrl.stand_cmd), float(ctrl.waist_cmd[2])
        dh, dp = float(h) - h0, float(pitch) - p0
        T = max(math.pi / 2 * abs(dh) / (0.9 * 0.15), math.pi / 2 * abs(dp) / (0.9 * 0.8))
        held_tgt = {sd: self.site(sd) for sd in hold if self.held.get(sd)}
        if held_tgt and T > 0.0:
            # (the arm keeping the hand still moves its elbow and wrist faster than the waist bends: a slower bend,
            # so the arm's joints stay well inside their speed limits and the held item does not lag down into what
            # it is held over)
            T = max(2.5 * T, 0.6)
        n = int(math.ceil(T / (2 * TICK)))
        pressing = 0
        for k in range(1, n + 1):
            u = 0.5 - 0.5 * math.cos(math.pi * k / n)
            cmd = {"stand_height": h0 + dh * u, "waist": [0.0, 0.0, p0 + dp * u]}
            if held_tgt:
                self._hold_cmd(cmd, held_tgt, h0 + dh * u, p0 + dp * u)
            self.c.command(cmd)
            self.step(2)
            pressing = pressing + 1 if (dh < -0.005 or dp > 0.01) and self.body_push() > 90.0 else 0
            if pressing >= 2:
                # (lowering or bending into the room: back to where the ramp was a moment ago and stop there;
                # standing up is never stopped)
                ub = 0.5 - 0.5 * math.cos(math.pi * max(0, k - 4) / n)
                self.log("[ref] posture: the body presses on the room (%.0f N); stopping" % self.body_push())
                self.c.command({"stand_height": h0 + dh * ub, "waist": [0.0, 0.0, p0 + dp * ub]})
                self.step(15)
                return False
        cmd = {"stand_height": float(h), "waist": [0.0, 0.0, float(pitch)]}
        if held_tgt:
            self._hold_cmd(cmd, held_tgt, float(h), float(pitch))
        self.c.command(cmd)
        for _ in range(int(timeout / 0.05)):
            self.step(5)
            if self.c.done:
                return False
            wq = self.d.qpos[r.waist_q]
            hz = PELVIS_HEIGHT + float(self.d.qpos[r.base_q[3]])
            if abs(ctrl.stand - ctrl.stand_cmd) < 1e-4 and float(np.max(np.abs(ctrl.waist - ctrl.waist_cmd))) < 1e-4 \
                    and float(np.max(np.abs(wq - ctrl.waist_cmd))) < 0.01 and abs(hz - ctrl.stand_cmd) < 0.004:
                self.step(5)
                return True
        return False

    def stand(self, h, wait=True):
        self.c.command({"stand_height": float(h)})
        if wait:
            self.step(int(abs(h - self.c.ctrl.stand) / 0.15 / TICK) + 15)

    def waist(self, pitch=0.0, yaw=0.0, wait=True):
        self.c.command({"waist": [float(yaw), 0.0, float(pitch)]})
        if wait:
            self.step(int(max(abs(pitch - self.c.ctrl.waist[2]), abs(yaw - self.c.ctrl.waist[0])) / 0.8 / TICK) + 10)

    def goto(self, x, y, yaw, tol=0.01, timeout=40.0, yaw_tol=0.015):
        """Base goal (straight line), waiting until reached (the controller's approach is exponential: loose
        tolerances for the waypoints of a walk save most of its time)."""
        self.c.command({"base_goal": [float(x), float(y), float(yaw)]})
        self.pushed = False
        b_prev = self.base()
        for _ in range(int(timeout / 0.05)):
            self.step(5)
            b = self.base()
            if math.hypot(b[0] - x, b[1] - y) < tol and abs(wrap(b[2] - yaw)) < yaw_tol:
                break
            if self.c.done:
                break
            # (a body link pressing on the room: stop at once and back off 5 cm the way the base came, rather than
            # push on; the safety watch's limit is 150 N for 0.2 s; the walk gives up)
            if self.body_push() > 60.0:
                self.log("[ref] goto: the body presses on the room (%.0f N) at (%.2f, %.2f); backing off" % (
                    self.body_push(), b[0], b[1]))
                d = np.array([b_prev[0] - b[0], b_prev[1] - b[1]])
                n = float(np.linalg.norm(d))
                d = d / n * 0.05 if n > 1e-4 else -0.05 * np.array([math.cos(b[2]), math.sin(b[2])])
                self.c.command({"base_goal": [float(b[0] + d[0]), float(b[1] + d[1]), float(b[2])]})
                self.step(40)
                self.pushed = True
                break
            b_prev = b
        self.step(4)
        b = self.base()
        return math.hypot(b[0] - x, b[1] - y)

    def body_push(self):
        """The largest contact force between a robot link (not a hand) and the room's fixed geometry now (N)."""
        m, d = self.m, self.d
        st = self.stance
        worst = 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            ct = d.contact[i]
            g1, g2 = int(ct.geom1), int(ct.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if a in st.robot_geoms and a not in st.hand_geoms and (m.geom_contype[b] & st.world_bit):
                    mujoco.mj_contactForce(m, d, i, f6)
                    worst = max(worst, abs(float(f6[0])))
        return worst

    def walk(self, x, y, yaw, carrying=False):
        """Walk to (x, y, yaw) round the furniture: arms in first (a held item in front of the chest, a free hand at
        home), turn towards each waypoint, go there, turn to the final yaw."""
        b = self.base()
        self.dest_hint = (float(x), float(y))
        if getattr(self, "_final_walk", False):
            self._final_walk = False             # (the rest of the way after a raise for the counter: below)
        elif not getattr(self, "_mid_walk", False):
            self._raised_goal = None
        if math.hypot(x - b[0], y - b[1]) > 0.04 or abs(wrap(yaw - b[2])) > 0.05:
            self.stow()
            carrying = carrying or any(self.held.values())
            if self.back_out(carrying):
                self.stow()                  # (raised clear of what stood by it for the step back: carried again)
        b = self.base()
        planner = self.planner_for(carrying)
        # the last stretch: face the goal's heading 0.35 m before it, then step straight in (no turning beside a
        # fixture); a goal near where the robot stands is reached directly
        c_, s_ = math.cos(yaw), math.sin(yaw)
        pre = (x - 0.35 * c_, y - 0.35 * s_)
        if math.hypot(x - b[0], y - b[1]) < 0.45:
            pts = []
        else:
            pts = planner.path((b[0], b[1]), pre, extra=self.loose_tools())
            for radius in [r_ for r_ in (0.60, 0.52, 0.42) if r_ < planner.radius - 1e-6]:
                if pts is not None:
                    break
                # no way round with the load's clearance: a tighter one (the body clear, the load may brush by)
                self.log("[ref] walk: no path with %.2f m clearance to (%.2f, %.2f); trying %.2f" % (
                    planner.radius, pre[0], pre[1], radius))
                if radius not in self.planners:
                    self.planners[radius] = Planner(radius=radius)
                planner = self.planners[radius]
                pts = planner.path((b[0], b[1]), pre, extra=self.loose_tools())
            if pts is None:
                # wedged in by a fixture: step straight back and plan once more; never walk blind
                self.back_out(True)
                b = self.base()
                pts = self.planners.get(0.42, planner).path((b[0], b[1]), pre, extra=self.loose_tools())
            if pts is None:
                self.log("[ref] walk: no path to (%.2f, %.2f); not moving" % pre)
                return False
        raise_ = [s_ for s_ in ("left", "right") if self.held[s_] and (self.held[s_] == "sponge" or (
            self.held[s_] in O.LOOSE and O.CATEGORY[O.LOOSE[self.held[s_]][0]] == "dish"))]
        goal_key = (round(float(x), 3), round(float(y), 3))
        if pts and raise_ and not getattr(self, "_mid_walk", False) and self.dest_counter() and not self.dest_high() \
                and getattr(self, "_raised_goal", None) != goal_key:
            # (round 5) to the counter from afar with a dish or the sponge at the waist: walked so until HIGH_NEAR
            # short of the goal (along the planned way), raised there for the counter, then the rest
            legs = [np.array([b[0], b[1]], float)] + [np.asarray(q, float) for q in pts] + [np.asarray(pre, float)]
            rest = 0.0
            mid = None
            for k in range(len(legs) - 1, 0, -1):
                seg = legs[k] - legs[k - 1]
                L_ = float(np.linalg.norm(seg))
                if rest + L_ >= self.HIGH_NEAR - 0.1:
                    u = (self.HIGH_NEAR - 0.1 - rest) / max(L_, 1e-9)
                    mid = legs[k] - seg * u
                    head = math.atan2(seg[1], seg[0])
                    break
                rest += L_
            if mid is not None and float(np.linalg.norm(mid - legs[0])) > 0.3:
                self._mid_walk = True
                try:
                    got = self.walk(float(mid[0]), float(mid[1]), head, carrying=True)
                finally:
                    self._mid_walk = False
                self.dest_hint = (float(x), float(y))
                if got is False:
                    return False
                self.log("[ref] walk: %s raised for the counter %.1f m short of it" % (
                    ", ".join(self.held[s_] for s_ in raise_), self.HIGH_NEAR))
                # (once per walk: where the rest of the way is longer than HIGH_NEAR as the crow flies, the raise
                # here was planned again on every call, without end; dest_high holds for the rest of this walk)
                self._raised_goal = goal_key
                self._final_walk = True
                return self.walk(x, y, yaw, carrying=carrying)
        if not self.held_path_clear(pts + [pre], (x, y), yaw):
            self.log("[ref] walk: a held tool would touch the room on the way to (%.2f, %.2f); not moving" % (x, y))
            return False
        if pts and not self.held_path_clear(pts + [pre], (x, y), yaw, strict=True):
            # (round 5) turning in place, a held long tool would pass within a few centimetres of the room (the plain
            # check's tolerance; the dustpan hanging by the hip caught a dining-table leg as the robot turned beside
            # it and slipped out of the hand): a step straight back first, if the turns are clear from there
            b = self.base()
            c0, s0 = math.cos(b[2]), math.sin(b[2])
            for back in (0.15, 0.30):
                bx, by = b[0] - back * c0, b[1] - back * s0
                if self.planner.occ[self.planner.cell(bx, by)] or \
                        not self.held_path_clear([], (bx, by), b[2], line=True):
                    continue
                p2 = planner.path((bx, by), pre, extra=self.loose_tools())
                if p2 is None or not self.held_path_clear(p2 + [pre], (x, y), yaw, start=(bx, by, b[2]), strict=True):
                    continue
                self.log("[ref] walk: %.2f m straight back before turning with the held tool" % back)
                self.goto(bx, by, b[2], tol=0.03, yaw_tol=0.05)
                if self.pushed:
                    return False
                pts = p2
                break
        if pts and not self.held_path_clear(pts + [pre], (x, y), yaw, loose=True):
            # (round 5) an item carried at the waist that would meet the room on the way, mostly near the goal, where
            # the planner keeps only the body's own clearance (turning there towards the goal's heading, the item
            # ahead swings into what stands by it: the tools hanging by the bins): the last 0.8 m walked straight in
            # at the goal's heading, if that way is open
            via = (x - 0.80 * c_, y - 0.80 * s_)
            # (the straight way kept off the long tools as well, by the hips' half width, all the way in: checked
            # against the room's fixtures only, a knee walked into the dustpan hanging by the bins)
            r_ = min(planner.radius, HARD_R)
            tools_ = planner._inflate(self.loose_tools(), r_) & ~planner._inflate([], r_)
            if planner._clear(via, pre, planner.hard) and planner._clear(via, (x, y), tools_):
                p2 = planner.path((b[0], b[1]), via, extra=self.loose_tools())
                if p2 is not None and self.held_path_clear(p2 + [via, pre], (x, y), yaw, loose=True):
                    pts = p2 + [via]
                    self.log("[ref] walk: the last 0.8 m to (%.2f, %.2f) straight in (the held item)" % (x, y))
        for k_lift in range(3):
            # (round 5) an item carried at the waist that would still meet the room on the way (a chair's back, what
            # still stands on the table by the goal): carried 6 cm higher, then drawn in close
            if self.held_path_clear(pts + [pre], (x, y), yaw, loose=True):
                break
            for s_ in ("left", "right"):
                if self.held[s_] in O.LOOSE:
                    if k_lift < 2:
                        self.carry_lift[self.held[s_]] = self.carry_lift.get(self.held[s_], 0.0) + 0.06
                        self.log("[ref] walk: %s would touch the room on the way to (%.2f, %.2f); carried higher" % (
                            self.held[s_], x, y))
                        self.carry_pose((s_,))
                    else:
                        self.carry_lift.pop(self.held[s_], None)
                        self.log("[ref] walk: %s would touch the room on the way to (%.2f, %.2f); drawn in" % (
                            self.held[s_], x, y))
                        self.compact = True
                        try:
                            self.carry_pose((s_,))
                        finally:
                            self.compact = False
        replanned = False
        k = 0
        while k < len(pts):
            px, py = pts[k]
            b = self.base()
            if not replanned and k and not planner._clear((b[0], b[1]), (px, py), planner.occ):
                # (off the planned line: plan again from here, once)
                np_ = planner.path((b[0], b[1]), pre, extra=self.loose_tools())
                replanned = True
                if np_:
                    pts, k = np_, 0
                    continue
            dist = math.hypot(px - b[0], py - b[1])
            if dist > 0.25:
                head = math.atan2(py - b[1], px - b[0])
                self.goto(b[0], b[1], head, tol=0.05, yaw_tol=0.08)
                if not self.pushed:
                    self.goto(px, py, head, tol=0.05, yaw_tol=0.08)
            else:
                self.goto(px, py, self.base()[2], tol=0.04, yaw_tol=0.08)
            if self.c.done:
                return False
            if self.pushed:
                self.log("[ref] walk: gave up after pressing on the room on the way to (%.2f, %.2f)" % (x, y))
                return False
            k += 1
        b = self.base()
        if pts:
            self.goto(b[0], b[1], yaw, tol=0.04, yaw_tol=0.05)
            if self.pushed:
                self.log("[ref] walk: gave up after pressing on the room while turning")
                return False
        self.goto(x, y, yaw, tol=0.008, yaw_tol=0.02)
        if self.pushed:
            self.log("[ref] walk: gave up after pressing on the room on the last stretch")
            return False
        b = self.base()
        if math.hypot(b[0] - x, b[1] - y) > 0.03 or abs(wrap(b[2] - yaw)) > 0.05:
            self.log("[ref] walk: stopped at (%.2f, %.2f, %.2f), %.2f m short of (%.2f, %.2f)" % (
                b[0], b[1], b[2], math.hypot(b[0] - x, b[1] - y), x, y))
            # (a few centimetres short: the skill plans its arm from where the robot stands)
            return math.hypot(b[0] - x, b[1] - y) < 0.10 and abs(wrap(b[2] - yaw)) < 0.15
        return True

    def held_path_clear(self, pts, goal, yaw, loose=False, start=None, strict=False, line=False):
        """The held long tools (broom, dustpan) clear of the room along a walk: each turn in place towards the next
        leg, along the legs, the last leg stepped in at the goal's heading; the arms as commanded. (The walk planner
        counts only the body's own clearance near the start and the goal: a dustpan hanging on the table's side of the
        robot was caught on its edge and pulled out of the hand.) `loose` (round 5): the held loose items instead
        (carried at the waist, an item may meet a chair's back or what stands on a table near the goal). `start`: from
        there (x, y, yaw) rather than where the robot stands; `strict` (round 5): each turn in place also with the body
        3 cm off in each direction and 0.08 rad either way (the hanging tool sways as the robot steps round); `line`:
        the legs walked at the start's heading (a step straight back)."""
        if loose:
            tools = [s_ for s_ in ("left", "right") if self.held[s_] in O.LOOSE and s_ in self.grip_rel]
        else:
            tools = [s_ for s_ in ("left", "right") if self.held[s_] in ("broom", "dustpan") and s_ in self.grip_rel]
        if not tools:
            return True
        st = self.stance
        h, pitch = float(self.c.ctrl.stand_cmd), float(self.c.ctrl.waist_cmd[2])
        arms = {sd: np.asarray(self.c.ctrl.arm_cmd[sd], float) for sd in ("left", "right")}
        names = [self.held[s_] for s_ in tools]
        mine = set(names) | {n + "_head" for n in names}

        def ok_at(x, y, th):
            st.set_state((float(x), float(y), float(th)), h, pitch, arms)
            for s_ in tools:
                st.place_held(s_, self.held[s_], self.grip_rel[s_])
            return not [b_ for b_ in st.collisions(held=names) if b_[0] in mine]

        def turn_ok(x, y, th):
            if not ok_at(x, y, th):
                return False
            if strict:
                for dx, dy, dt in ((0.03, 0, 0), (-0.03, 0, 0), (0, 0.03, 0), (0, -0.03, 0), (0, 0, 0.08),
                                   (0, 0, -0.08)):
                    if not ok_at(x + dx, y + dy, th + dt):
                        return False
            return True
        b = self.base() if start is None else start
        cur, th = np.array([float(b[0]), float(b[1])]), float(b[2])
        legs = [np.asarray(q, float) for q in pts] + [np.asarray(goal, float)]
        for k, q in enumerate(legs):
            seg = q - cur
            L = float(np.linalg.norm(seg))
            if L < 0.02:
                continue
            th_new = th if line else math.atan2(seg[1], seg[0]) if k < len(legs) - 1 else float(yaw)
            dth = wrap(th_new - th)
            for f in np.linspace(0.0, 1.0, max(2, int(abs(dth) / 0.25) + 1)):
                if not turn_ok(cur[0], cur[1], th + dth * f):
                    return False
            th = th_new
            for f in np.linspace(0.0, 1.0, max(2, int(L / 0.06) + 1))[1:]:
                pt = cur + seg * f
                if not ok_at(pt[0], pt[1], th):
                    return False
            cur = q
        return True

    def loose_tools(self):
        """Boxes round the long tools wherever they are when not in a hand (hanging in their clips, standing or lying
        on the floor): the walk keeps off them."""
        out = []
        for tool in ("dustpan", "broom"):
            if tool in self.held.values():
                continue
            p, R = self.pose(tool)
            pts = [p]
            if tool == "dustpan":
                W, D = O.DUSTPAN["width"], O.DUSTPAN["depth"]
                pts = [p + R @ np.array([sx * W / 2, sy, 0.0]) for sx in (-1, 1) for sy in (0.0, D + 0.12)]
            else:
                f = next(f_ for f_ in self.c.meta["features"]["broom"] if f_[0] == "cyl")
                hx = O.BROOM["head"][0] / 2
                pts = [p + R @ np.asarray(f[1]), p + R @ np.asarray(f[2]), p + R[:, 0] * hx, p - R[:, 0] * hx]
            P = np.array(pts)[:, :2]
            lo, hi = P.min(axis=0) - 0.05, P.max(axis=0) + 0.05
            out.append((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))
        return out

    def arrived(self, x, y, yaw, tol=0.03):
        b = self.base()
        return math.hypot(b[0] - x, b[1] - y) < tol and abs(wrap(b[2] - yaw)) < 0.05

    def held_reach(self):
        """How far (horizontally) what the hands hold reaches from the base's centre now (its colliders' corners; a
        tool's head on its own body included)."""
        b = self.base()
        m, d = self.m, self.d
        reach = 0.0
        for n in self.held.values():
            if not n:
                continue
            root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            bodies = {k for k in range(m.nbody) if k == root or m.body_parentid[k] == root}
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) not in bodies or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                    continue
                R = d.geom_xmat[g].reshape(3, 3)
                sz = m.geom_size[g]
                t_ = int(m.geom_type[g])
                Gt = mujoco.mjtGeom
                if t_ in (Gt.mjGEOM_BOX, Gt.mjGEOM_ELLIPSOID):
                    ext = np.abs(R[:2, :]) @ sz[:3]
                elif t_ in (Gt.mjGEOM_CAPSULE, Gt.mjGEOM_CYLINDER):
                    ext = np.abs(R[:2, 2]) * sz[1] + sz[0]          # along its axis, plus its radius
                else:
                    ext = np.full(2, float(m.geom_rbound[g]))
                c = d.geom_xpos[g][:2] - b[:2]
                reach = max(reach, float(np.hypot(abs(c[0]) + ext[0], abs(c[1]) + ext[1])))
        return reach

    def planner_for(self, carrying):
        """The path planner for the robot's present load: what the hands hold, as far as it reaches from the body's
        centre (plus 5 cm), sweeps that circle when the robot turns."""
        if not carrying:
            return self.planner
        reach = max(0.42, round((self.held_reach() + 0.05) / 0.02) * 0.02)
        if reach not in self.planners:
            self.planners[reach] = Planner(radius=reach)
        return self.planners[reach]

    def clear_of_counter(self, reach, margin=0.05):
        """(round 5) Facing the kitchen counter, straight back until the point `reach` ahead of the robot is `margin`
        in front of the counter's top edge (an item lowered to the waist nearer would go in under the top's edge or
        below the sink's front wall). At most 0.4 m."""
        K = RM.KITCHEN
        b = self.base()
        c_, s_ = math.cos(b[2]), math.sin(b[2])
        ax, ay = b[0] + reach * c_, b[1] + reach * s_
        if c_ > -0.3 or not (K["y0"] - 0.1 < ay < K["y1"] + 0.1) or ax >= K["top_x1"] + margin:
            return
        d = min(0.4, (K["top_x1"] + margin - ax) / -c_)
        self.log("[ref] %.2f m back from the counter before lowering to the waist" % d)
        self.goto(b[0] - d * c_, b[1] - d * s_, b[2], tol=0.02, yaw_tol=0.05)

    def draw_in(self, side, dists=(0.15, 0.12, 0.08, 0.05)):
        """(round 5) The held item drawn straight back towards the body, level or 5 cm lower, the hand's rotation kept,
        by a checked straight line (to no nearer than 0.18 m in front of the body's middle): the steps back from a
        fixture then start from a bent arm, not one stretched out over the fixture; only if the arm then holds it more
        naturally (drawn in low over a table, the hand's rotation kept folds the elbow to its limit). True if drawn
        in."""
        b = self.base()
        p, R = self.site(side)
        fwd = np.array([math.cos(b[2]), math.sin(b[2]), 0.0])
        ahead = float((p - np.array([b[0], b[1], p[2]])) @ fwd)
        h, pitch = float(self.c.ctrl.stand_cmd), float(self.c.ctrl.waist_cmd[2])
        q_now = np.asarray(self.d.qpos[self.c.robot.arm_q[side]], float).copy()
        self.stance.set_state((b[0], b[1], b[2]), h, pitch)
        c_now = posture_cost(self, side, q_now)
        for dd, dz in [(dd_, dz_) for dd_ in dists for dz_ in (0.0, -0.05)]:
            if ahead - dd < 0.18:
                continue
            tgt = p - fwd * dd + np.array([0.0, 0.0, dz])
            # (start=False: what the item touches now, the soap pump's head it pressed, does not block the line;
            # (round 5) a little lower too: the arm folds down rather than up)
            ok, worst, qs, why = self.stance.check((b[0], b[1], b[2]), h, pitch, side, [(tgt, R)],
                                                   holding=self.held[side], grip_rel=self.grip_rel.get(side),
                                                   fallback=False, start=False)
            if ok:
                q_end = np.asarray(list(rows(qs))[-1], float)
                c_end = posture_cost(self, side, natural_resolve(self, side, tgt, R, q_end))
                if c_end < c_now - 0.1:
                    self.log("[ref] drawn in %.2f m before stepping back (posture %.2f -> %.2f)" % (dd, c_now, c_end))
                    self.run_path(side, qs)
                    return True
        return False

    def back_out(self, carrying):
        """Close to a fixture (inside the planner's inflated obstacles), step straight back until clear (at most
        0.6 m), so that turning does not sweep the arms or what they hold into it."""
        # (the body's own footprint decides: a held tool's wide clearance finds no free cell in a furnished room)
        planner = self.planner
        b = self.base()
        c_, s_ = math.cos(b[2]), math.sin(b[2])
        if not planner.occ[planner.cell(b[0], b[1])]:
            return
        occ_prev = True
        best = None
        for k in range(1, 13):
            x, y = b[0] - 0.05 * k * c_, b[1] - 0.05 * k * s_
            occ = bool(planner.occ[planner.cell(x, y)])
            if occ and not occ_prev:
                break                      # (backing into another fixture: no further than the free cells)
            occ_prev = occ
            if not occ:
                best = (x, y)
                if k >= 2:
                    break
        lifted = False
        if best is not None:
            for sd in ("left", "right"):
                if self.held[sd]:
                    lifted = self.lift_clear_of_neighbours(sd, float(math.hypot(best[0] - b[0], best[1] - b[1]))) \
                        or lifted
            self.goto(best[0], best[1], b[2], tol=0.02)
        return lifted

    def _loose_extent(self, name):
        """(xy radius about the body's origin, top's height) of a loose object as it lies now (its collision geoms'
        boxes, geom_aabb, turned into the room)."""
        m, d = self.m, self.d
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        c = d.xpos[bid]
        r, top = 0.0, float(c[2])
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != bid or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            Rg = d.geom_xmat[g].reshape(3, 3)
            ctr = d.geom_xpos[g] + Rg @ m.geom_aabb[g][:3]
            ext = np.abs(Rg) @ m.geom_aabb[g][3:]
            r = max(r, float(np.hypot(ctr[0] - c[0], ctr[1] - c[1]) + np.hypot(ext[0], ext[1])))
            top = max(top, float(ctr[2] + ext[2]))
        return r, top

    def lift_clear_of_neighbours(self, side, back, half=0.20, clear=0.03):
        """(round 5) Before a step straight back of `back` metres with this hand low over a surface: the hand (and what
        it holds) raised clear of the loose objects standing where it will sweep (the planner's checks leave the hands
        out, and a step back is no planned path: a plate drawn in just clear of the dining table knocked a glass
        standing by its edge off it as the robot stepped back; a can lifted at the coffee table dragged the bottle
        beside it off). The corridor: from 0.30 m ahead of the hand to `back` + 0.10 m behind it, `half` to either
        side; the hand's lowest point taken 6 cm below its grasp site, or the held item's bottom. Logged when raised."""
        if back < 0.01:
            return False
        b = self.base()
        p, R = self.site(side)
        fwd = np.array([math.cos(b[2]), math.sin(b[2]), 0.0])
        lat = np.array([-fwd[1], fwd[0], 0.0])
        held = self.held[side]
        low = float(p[2]) - 0.06
        if held:
            low = min(low, float(p[2]) - self.held_depth(side))
        need, who = 0.0, None
        for n in O.LOOSE:
            if n in self.held.values():
                continue
            q, _ = self.pose(n)
            if q[2] < 0.3:
                continue                      # (on the floor: not in the hand's way)
            rel = np.asarray(q, float) - np.asarray(p, float)
            along = -float(rel @ fwd)
            r_, top_ = self._loose_extent(n)
            if -0.30 <= along <= back + 0.10 and abs(float(rel @ lat)) <= half + r_ and top_ + clear - low > need:
                need, who = top_ + clear - low, n
        if need <= 0.005:
            return False
        h, pitch = float(self.c.ctrl.stand_cmd), float(self.c.ctrl.waist_cmd[2])
        for f in (1.0, 0.7):
            up = np.asarray(p, float) + np.array([0.0, 0.0, need * f])
            ok, worst, qs, why = self.stance.check((b[0], b[1], b[2]), h, pitch, side, [(up, R)], holding=held,
                                                   grip_rel=self.grip_rel.get(side) if held else None)
            if ok:
                self.log("[ref] %s hand %.2f m up clear of %s before stepping back" % (side, need * f, who))
                self.run_path(side, qs)
                return True
        self.log("[ref] %s hand: no way up clear of %s before stepping back (%s)" % (side, who, why))
        return False

    def arms(self, targets, timeout=6.0, tol=0.004, path="line", waist=None):
        """targets: {side: (pos, R)} in the room frame; waits until every grasp site is within tol (or timeout)."""
        cmd = {}
        for side, (p, R) in targets.items():
            cmd[f"{side}_arm"] = {"pose": [float(v) for v in p] + [float(v) for v in _quat(R)], "frame": "room",
                                  "path": path}
        if waist is not None:
            cmd["waist"] = [float(v) for v in waist]
        self.c.command(cmd)
        err = 1.0
        for _ in range(int(timeout / 0.05)):
            self.step(5)
            err = max(float(np.linalg.norm(self.site(s)[0] - np.asarray(p))) for s, (p, _) in targets.items())
            if err < tol or self.c.done:
                break
        self.step(5)
        return err

    def track(self, side, poses, ticks=4, iters=40):
        """Follow a sequence of grasp-site poses closely, in joint space (constrained motions: a lid, a lever, a
        sponge on the table): each pose's joints by a local solve, commanded as joint targets (R None: the position
        only, the wrist free)."""
        worst = 0.0
        q = np.asarray(self.c.ctrl.arm_cmd[side], float).copy()
        qs = []
        for p, R in poses:
            q, ep, er = self.stance.nik.solve(self.d.qpos, side, np.asarray(p, float), None if R is None else _quat(R),
                                              q, iters=iters, null_total=0.05, natural=True)
            worst = max(worst, ep)
            qs.append(q.copy())
        if qs:
            # a smooth joint path through the solutions, no faster than `ticks` ticks per pose
            span = sum(float(np.max(np.abs(qs[k] - (qs[k - 1] if k else self.c.ctrl.arm_cmd[side]))))
                       for k in range(len(qs)))
            rate = None
            if span > 1e-6:
                rate = min(0.5 if self.held[side] in ("broom", "dustpan") else 1.2, span / (len(qs) * ticks * TICK))
            self.run_path(side, qs, rate=rate)
        return worst

    def hand(self, side, close):
        self.c.command({f"{side}_hand": "close" if close else "open"})
        self.step(6)
        got = self.c.grips.holding(side)
        if close and got:
            p, R = self.site(side)
            po, Ro = self.pose_any(got)
            self.grip_rel[side] = (Ro.T @ (p - po), Ro.T @ R)
            self.held[side] = got
            self.stance.grip_rels = self.grip_rel
        if not close:
            self.carry_lift.pop(self.held[side], None)
            self.held[side] = None
            self.grip_rel.pop(side, None)
        return got

    def pose_any(self, name):
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def hand_for(self, side, p_obj, R_obj):
        """The grasp-site pose that puts the object held in `side` at (p_obj, R_obj)."""
        pg, Rg = self.grip_rel[side]
        return p_obj + R_obj @ pg, R_obj @ Rg

    def held_depth(self, side):
        """How far the held object reaches below the grasp site (m)."""
        name = self.held[side]
        if name is None:
            return 0.0
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        ps = self.site(side)[0]
        lows = [geom_lowest(self.m, self.d, g) for g in range(self.m.ngeom)
                if self.m.geom_bodyid[g] == b and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]
        return max(0.0, ps[2] - min(lows)) if lows else 0.05

    def carry_pose(self, sides=("left", "right")):
        """Held items up and in for walking, by a planned joint path (orientation kept): raised so the item clears
        the counter (0.95 m) by its depth below the hand, then drawn in towards the chest as far as reachable."""
        b = self.base()
        Rb = _Rz(b[2])
        fwd = Rb[:, 0]
        h = self.c.ctrl.stand
        pitch = float(self.c.ctrl.waist[2])
        for s_ in sides:
            p, R = self.site(s_)
            if self.held[s_] == "broom" and self.broom_upright(s_):
                continue
            if self.held[s_] == "dustpan" and self.pan_compact(s_):
                continue
            if self.held[s_] in ("broom", "dustpan"):
                # a long tool: the hand drawn in beside the body (the tool hanging there sweeps no wide arc when the
                # robot turns), its lowest point at least 12 cm off the floor
                lat = 0.24 if s_ == "left" else -0.24
                z = max(0.85, 0.22 + self.held_depth(s_))
                tgt = np.array([b[0], b[1], 0.0]) + Rb @ np.array([0.20, lat, 0.0])
                tgt[2] = z
                if float(np.linalg.norm(tgt - p)) < 0.03:
                    continue
                for k in (1.0, 0.7, 0.4):
                    goal = p + (tgt - p) * k
                    ok, worst, qs, why = self.stance.check((b[0], b[1], b[2]), h, pitch, s_, [(goal, R)],
                                                           holding=self.held[s_], grip_rel=self.grip_rel.get(s_))
                    if ok:
                        self.run_path(s_, qs)
                        break
                continue
            else:
                # (round 5) carried as a person carries it: in front of the waist with the forearm forward, at about
                # elbow height (the G1's elbow hangs 0.89 m up), clear of the dining table by the item's depth below
                # the hand; on the way to the kitchen counter (dest_hint) just clear of the counter instead (0.90 m),
                # so it is not swept into it as the robot steps up; go_put lifts it higher first if it would still
                # sweep down into what the robot leans over
                depth = self.held_depth(s_)
                lift = self.carry_lift.get(self.held[s_], 0.0)
                # (on the way to the kitchen counter, dest_hint, a dish or the sponge just clear of the counter: it is
                # put down there from a stance stepped into, go_put's low mode, held where it is while the robot
                # bends; books go onto the shelf so too but from the waist: the shelf's top is out of their way)
                high = self.dest_high() and self.held[s_] is not None and \
                    (self.held[s_] == "sponge" or O.CATEGORY[O.LOOSE[self.held[s_]][0]] == "dish")
                z_low = max(0.84, 0.79 + depth) + lift
                # (3 cm clear of what stands on the counter: the trays' rims, 0.92 m, as the robot steps in)
                z_high = max(0.96, 0.935 + depth) + lift
                if self.planner.occ[self.planner.cell(b[0], b[1])]:
                    # next to a fixture: the item first drawn in where the robot stands, just clear of the surface
                    # (the arm out of its working pose before the steps back), then a step straight back clear of the
                    # fixture (drawn in to the waist there it would sweep through what stands by it: the soap bottle
                    # beside the sponge's tray, the dishes still on the table)
                    # (round 5: just clear of the surface it stands by: by the dining table (0.75 m) the counter's
                    # height put a glass on a straight arm at shoulder height)
                    (dx_, dy_), (lx_, ly_) = RM.DINING["center"], RM.DINING["size"]
                    by_table = abs(b[0] - dx_) < lx_ / 2 + 0.6 and abs(b[1] - dy_) < ly_ / 2 + 0.6
                    z_fix = max(z_low, RM.DINING["top_z"] + depth + 0.05) if by_table else z_high
                    if high or not self.canonical_carry(s_, z_fix, low=by_table):
                        # (no carry pose there: the hand drawn straight back towards the body at its height first,
                        # so the robot does not step back with its arm stretched out over the fixture: the soap
                        # pump's head is at shoulder height)
                        self.draw_in(s_)
                    self.back_out(True)
                    b = self.base()
                if not high:
                    # (round 5) clear of the kitchen counter first: stepped back from the sink after a rinse, the
                    # sponge lowered to the waist 0.4 m ahead went in below the sink's front wall, caught on it as the
                    # robot stepped back and fell
                    self.clear_of_counter(0.45)
                    b = self.base()
                if self.canonical_carry(s_, z_high if high else z_low, low=not high):
                    continue
                if not high:
                    # (still over the counter's or the table's edge at the waist's height: 0.2 m further back first,
                    # where the floor is open)
                    b = self.base()
                    bx_, by_ = b[0] - 0.20 * math.cos(b[2]), b[1] - 0.20 * math.sin(b[2])
                    if not self.planner.occ[self.planner.cell(bx_, by_)]:
                        self.goto(bx_, by_, b[2], tol=0.03, yaw_tol=0.05)
                        if self.canonical_carry(s_, z_low, low=True):
                            continue
                if not high and self.canonical_carry(s_, z_high, low=False):
                    continue
                # (as round 4 carried it: chest height)
                z = max(1.00, 0.95 + depth + 0.05) + lift
                if self.canonical_carry(s_, z, low=False, legacy=True):
                    continue
            if p[2] >= z - 0.01:
                continue
            done = False
            self.log("[ref] carry %s: raised as held" % self.held[s_])
            for dz in (0.0, -0.03, -0.06):
                up = np.array([p[0], p[1], z + dz])
                ok, worst, qs, why = self.stance.check((b[0], b[1], b[2]), h, pitch, s_, [(up, R)],
                                                       holding=self.held[s_], grip_rel=self.grip_rel.get(s_))
                if ok:
                    self.run_path(s_, qs)
                    done = True
                    break
            if not done:
                # the hand (its rotation free) to a carry point in front of the chest, by a checked joint-space move
                lat = 0.18 if s_ == "left" else -0.18
                tgt = np.array([b[0], b[1], 0.0]) + Rb @ np.array([0.30, lat, 0.0])
                tgt[2] = z
                ok, worst, qs, why = self.stance.check((b[0], b[1], b[2]), h, pitch, s_, [(tgt, None)],
                                                       holding=self.held[s_], grip_rel=self.grip_rel.get(s_))
                if ok:
                    self.run_path(s_, qs)
                else:
                    self.log("[ref] carry %s: no way up (%s)" % (self.held[s_], why))

    def _tool_carry(self, side, tool, cands, z, dry=False):
        """A held long tool to a carry pose beside the hip: of the candidates [(target point, hand rotation, extra
        cost)], the one the arm holds most naturally (the human measure of the posture the executed path resolves),
        first up (if low), then there by a checked path (a held tool passes through the robot's own legs: only the
        room counts). (round 5: round 4 took the first of a few that the arm reached.) True if done or already so.
        `dry` (round 5): the best candidate's arm joints only (None if none is reachable), unmoved."""
        b = self.base()
        h, pitch = self.c.ctrl.stand, float(self.c.ctrl.waist[2])
        p, R = self.site(side)
        st, r = self.stance, self.c.robot
        q0 = self.d.qpos[r.arm_q[side]].copy()
        ranked = []
        for tgt, Rh, extra in cands:
            if not dry and float(np.linalg.norm(tgt - p)) < 0.02 and float(np.linalg.norm(Rh - R)) < 0.05:
                return True
            best = None
            for sq in (q0, np.array(NATURAL_CARRY_Q[side], float), np.array(HOME_Q[side], float)):
                st.set_state((b[0], b[1], b[2]), h, pitch)
                q, ep, er = st.solve(side, tgt, Rh, sq, iters=100, quick=30)
                if ep > 0.006 or er > 0.05:
                    continue
                q = natural_resolve(self, side, tgt, Rh, q)
                c = posture_cost(self, side, q) + extra
                if best is None or c < best[0]:
                    best = (c, q)
            if best is not None:
                ranked.append((best[0], tgt, Rh, best[1]))
        if dry:
            return min(ranked, key=lambda t_: t_[0])[3] if ranked else None
        ranked = [t_[:3] for t_ in ranked]
        why = None
        for c, tgt, Rh in sorted(ranked, key=lambda t_: t_[0])[:6]:
            up = p + np.array([0.0, 0.0, max(0.0, min(0.12, z - 0.06 - p[2]))])
            poses = ([(up, R)] if up[2] > p[2] + 0.01 else []) + [(tgt, Rh)]
            ok, worst, qs, why = st.check((b[0], b[1], b[2]), h, pitch, side, poses, holding=tool,
                                          grip_rel=self.grip_rel.get(side))
            if ok:
                self.log("[ref] %s carry: posture %.2f" % (tool, c))
                self.run_path(side, qs)
                return True
        self.log("[ref] %s carry: no way (%d candidates; %s)" % (tool, len(ranked), why))
        return False

    def broom_upright(self, side, dry=False):
        """The held broom beside the hip for walking, its head about 20 cm off the floor (a posture change at the next
        stance lowers it by up to 15 cm): upright (tilted forward its head would reach 0.7 m ahead of the robot) or
        (round 5) tilted 30-40 deg with its head trailing behind and out beside the leg, the hand then at the waist;
        turned about its handle so the arm holds it most naturally (round 5: the hand's approach forward, the wrist
        straight; the head fore and aft preferred, across it reaches out farther). False if no way is clear."""
        b = self.base()
        Rb = _Rz(b[2])
        pg, Rg = self.grip_rel[side]
        sg = 1.0 if side == "left" else -1.0
        z = max(0.90, float(pg[2]) + 0.08 + 0.20)       # the grip's height above the bristles, plus 20 cm
        cands = []
        for fwd, la in ((0.20, 0.26), (0.26, 0.22), (0.14, 0.28), (0.31, 0.21)):
            tgt = np.array([b[0], b[1], 0.0]) + Rb @ np.array([fwd, sg * la, 0.0])
            tgt[2] = z
            for k in range(12):
                psi = b[2] + k * math.pi / 6.0
                xb = np.array([math.cos(psi), math.sin(psi), 0.0])
                Rbr = np.stack([xb, np.cross(np.array([0.0, 0.0, 1.0]), xb), np.array([0.0, 0.0, 1.0])], axis=1)
                # (the head across the body reaches out by up to its half length)
                extra = 0.6 * abs(math.sin(k * math.pi / 6.0)) + 0.05 * abs(fwd - 0.24) / 0.06
                cands.append((tgt, Rbr @ Rg, extra))
                if getattr(self, "broom_stroke_start", False):
                    continue
                # (round 5) for a walk upright with its head lower, 10 or 14 cm off the floor, the hand at 0.96-1.0 m
                # (stroke() raises it again before the robot lowers and leans for a stroke): held with its head 20 cm
                # up the grip is at 1.06 m, about the G1's shoulder, and the elbow winged out to hold it there
                for zdrop in (0.10, 0.06):
                    tgt_l = tgt.copy()
                    tgt_l[2] = max(0.80, z - zdrop)
                    cands.append((tgt_l, Rbr @ Rg, extra + 0.1))
                # (round 5) or tilted 30-40 deg, the head trailing behind and out beside the leg, the hand at the
                # waist
                for th in (0.52, 0.70):
                    dh = Rb @ np.array([-1.0, 0.5 * sg, 0.0])
                    dh /= np.linalg.norm(dh)
                    u = np.array([0.0, 0.0, math.cos(th)]) - dh * math.sin(th)
                    xt = xb - (xb @ u) * u
                    if float(np.linalg.norm(xt)) < 1e-6:
                        continue
                    xt /= np.linalg.norm(xt)
                    Rt_ = np.stack([xt, np.cross(u, xt), u], axis=1)
                    tgt_t = tgt.copy()
                    tgt_t[2] = 0.30 + float((Rt_ @ pg)[2])
                    cands.append((tgt_t, Rt_ @ Rg, extra + 0.2))
        return self._tool_carry(side, "broom", cands, z, dry=dry)

    def pan_compact(self, side):
        """The held dustpan level beside the hip for walking, its floor about 15 cm off the floor (hanging out sideways
        it would reach 0.8 m from the robot's middle; a held tool passes through the robot's own legs), its lip
        towards whichever direction the arm holds it most naturally (round 5; round 4: backwards, else inwards,
        forwards, outwards, whichever the arm reached first; a pan lying across the body reaches out farther). False
        if no way is clear."""
        b = self.base()
        Rb = _Rz(b[2])
        pg, Rg = self.grip_rel[side]
        sgn = 1.0 if side == "left" else -1.0
        z = max(0.85, float(pg[2]) + 0.15)
        cands = []
        for fwd, la in ((0.18, 0.24), (0.24, 0.20), (0.12, 0.26), (0.30, 0.20), (0.34, 0.17)):
            tgt = np.array([b[0], b[1], 0.0]) + Rb @ np.array([fwd, sgn * la, 0.0])
            tgt[2] = z
            for k in range(8):
                ang = b[2] + math.pi + k * math.pi / 4.0           # the lip's direction: backwards first
                lip = np.array([math.cos(ang), math.sin(ang)])
                lat_out = abs(math.sin(k * math.pi / 4.0))
                cands.append((tgt, pan_rot(lip) @ Rg, 0.4 * lat_out + 0.05 * abs(fwd - 0.24) / 0.06))
        return self._tool_carry(side, "dustpan", cands, z)

    def canonical_carry(self, side, z, low=True, legacy=False, dry=False):
        """The held item to a carry point, level, in the most natural arm posture (posture_cost, the human measure, of
        the posture the executed path resolves: NaturalIK's natural resolution), by a checked joint-space move:
        whatever odd branch the grasp left the arm in, the next put starts from the same kind of posture. The points
        (round 5, carry_points): in front of the waist with the forearm forward at about the height z and a little
        above it; litter for a bin also beside the hip with the arm hanging; `legacy`: round 4's points in front of
        the chest (CARRY_POINTS). An item symmetric about its vertical axis may also turn about it (the hand then holds
        it at another angle) and litter is carried at any angle, so long as it stays clear of the robot's body. False
        when none is clear."""
        b = self.base()
        st = self.stance
        r = self.c.robot
        p, R = self.site(side)
        # the item level (a book flat, anything else upright): straightening up turned it with the torso
        name = self.held[side]
        if side in self.grip_rel and name in O.LOOSE:
            _, Ro = self.pose_any(name)
            if name in O.BOOKS:
                yb = np.array([0.0, 0.0, 1.0 if Ro[2, 1] > 0 else -1.0])
                xb = Ro[:, 0] - (Ro[:, 0] @ yb) * yb
                xb /= np.linalg.norm(xb)
                Rl = np.stack([xb, yb, np.cross(xb, yb)], axis=1)
            else:
                Rl = level(Ro)
            R = Rl @ self.grip_rel[side][1]
        free_rot = name in O.LOOSE and O.LOOSE[name][0] in FREE_CARRY
        sgn = 1.0 if side == "left" else -1.0
        h, pitch = self.c.ctrl.stand, float(self.c.ctrl.waist[2])
        st.set_state((b[0], b[1], b[2]), h, pitch)
        q0 = self.d.qpos[r.arm_q[side]].copy()
        home = np.array(HOME_Q[side])
        sym = name in O.LOOSE and O.LOOSE[name][0] in SYMMETRIC
        yaws = (0.0,) if free_rot else (0.0, -0.5, 0.5, -1.0, 1.0, -1.6, 1.6) if sym else (0.0,)
        seeds = [q0, np.array(NATURAL_CARRY_Q[side], float), home]
        sols = []
        z_hip = max(0.66, 0.50 + self.held_depth(side)) + self.carry_lift.get(name, 0.0)
        pts = CARRY_POINTS if legacy else carry_points(low, compact=getattr(self, "compact", False))
        for fwd, la, zp in pts:
            if zp < 0.0 and not hip_carry(name):
                continue
            t2 = np.array([b[0], b[1], 0.0]) + _Rz(b[2]) @ np.array([fwd, sgn * la, 0.0])
            t2[2] = (z + zp) if zp >= 0.0 else z_hip + (0.0 if zp == -1.0 else 0.06)
            # (litter beside the hip or drawn in close may hang at any angle: a can is litter)
            loose_rot = free_rot or ((zp < 0.0 or getattr(self, "compact", False)) and hip_carry(name))
            for dyaw in ((0.0,) if loose_rot else yaws):
                Rt = None if loose_rot else R if dyaw == 0.0 else _Rz(dyaw) @ R
                for sq in seeds:
                    st.set_state((b[0], b[1], b[2]), h, pitch)
                    q, ep, er = st.solve(side, t2, Rt, sq, iters=120, quick=40)
                    if ep > 0.01 or (Rt is not None and er > 0.08):
                        continue
                    q = natural_resolve(self, side, t2, Rt, q)
                    if any(float(np.max(np.abs(q - s_[1]))) < 0.05 for s_ in sols):
                        continue
                    if name and self.held_near_body(side, q, name):
                        continue
                    # (a person carries an item close in, the elbow bent: the arm stretched out far in front costs)
                    sols.append((posture_cost(self, side, q) + 0.1 * abs(dyaw) + 0.03 * float(np.max(np.abs(q - q0)))
                                 + 0.05 * abs(fwd - 0.33) / 0.04 + 0.5 * (max(0.0, float(q[3]) - 0.55) / 0.25) ** 2,
                                 q, (fwd, la, zp, dyaw)))
        if dry:
            # (round 5) the posture only, unmoved: what a put planned now for later, the item raised by then, starts
            # from
            return min(sols, key=lambda t: t[0])[1] if sols else None
        for k_, (c1, q1, how) in enumerate(sorted(sols, key=lambda t: t[0])[:8]):
            st.set_state((b[0], b[1], b[2]), h, pitch)
            st.chk.qpos[r.arm_q[side]] = q0
            path = None
            if k_ < 4:
                # (round 5) the hand's own straight line there (its waypoints solved, and executed, with the arm's
                # natural resolution: a straight joint move from an odd posture swings the elbow through odd ones),
                # else a joint move by a natural reach
                pc_, Rc_ = st.site_pose(side, q1)
                # (an item carried at any angle too to the planned posture's rotation: solved for the position
                # alone, the line's waypoints kept the wrist as the take had left it, bent)
                ok_, _, qs_, why_ = st.check((b[0], b[1], b[2]), h, pitch, side, [(pc_, Rc_)],
                                             holding=name, grip_rel=self.grip_rel.get(side), seed=q0, fallback=False)
                if ok_:
                    path = list(rows(qs_))
            if path is None:
                st.set_state((b[0], b[1], b[2]), h, pitch)
                st.chk.qpos[r.arm_q[side]] = q0
                path, why = st._joint_line(side, q0, q1, self.held[side], self.grip_rel.get(side), ())
            if path is not None:
                if float(np.max(np.abs(q1 - q0))) > 0.02:
                    self.log("[ref] carry %s: posture cost %.2f (point %.2f %.2f %s, turned %.1f)" % (
                        name, c1, how[0], how[1], "hip" if how[2] < 0 else
                        ("chest" if legacy else "front %.2f" % (z + how[2])), how[3]))
                self.run_path(side, path)
                return True
        self.log("[ref] carry %s: no natural carry (%d postures)" % (name, len(sols)))
        return False

    def held_near_body(self, side, q, name, clear=0.015):
        """With the arm at q (the stance model's state as set), would the held item come within `clear` of the
        robot's own body (its torso, arms and legs; the hands carry it)? The cell lets a held item pass through the
        robot (it never touches it), which the eye does not."""
        st, m = self.stance, self.stance.m
        st.chk.qpos[self.c.robot.arm_q[side]] = q
        st.place_held(side, name, self.grip_rel[side])
        mujoco.mj_kinematics(m, st.chk)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        mine = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid and (m.geom_contype[g] or
                                                                                m.geom_conaffinity[g])]
        body = getattr(self, "_body_geoms", None)
        if body is None:
            body = self._body_geoms = sorted(g for g in st.robot_geoms if g not in st.hand_geoms)
        ft = np.zeros(6)
        for g1 in mine:
            for g2 in body:
                if mujoco.mj_geomDistance(m, st.chk, g1, g2, clear, ft) < clear:
                    return True
        return False

    def sync_held(self):
        """What the hands really hold (a grip may have slipped): a lost object is forgotten and the hand opened."""
        for sd in ("left", "right"):
            if self.held[sd] and self.c.grips.holding(sd) != self.held[sd]:
                self.log("[ref] the %s hand lost %s (t %.1f)" % (sd, self.held[sd], self.c.sim_time))
                self.held[sd] = None
                self.grip_rel.pop(sd, None)
                self.c.command({f"{sd}_hand": "open"})
                self.step(5)

    def stow(self):
        """Arms in before the robot walks or turns: whatever a hand holds raised clear of what it was taken from or put
        on, then in front of the chest; a free arm at home; standing up straight."""
        self.sync_held()
        held = [sd for sd in ("left", "right") if self.held[sd]]
        if held and (float(self.c.ctrl.waist_cmd[2]) > 0.05 or self.c.ctrl.stand_cmd < 0.75):
            # upright first, so the raise below is in reach; (round 5) the hand holding an item keeps its place in the
            # room meanwhile, as far as the arm reaches (with the arm's joints fixed the item rose with the torso to
            # the face: the sponge taken from its tray leaning over the counter)
            self.posture(0.79, 0.0, hold=tuple(sd for sd in held if self.held[sd] not in ("broom", "dustpan")))
        if held:
            self.carry_pose(tuple(held))
        for sd in ("left", "right"):
            if not self.held[sd]:
                self.retreat(sd)
                self.home_safe(sd)
        self.posture(0.79, 0.0)
        self.wait_arms()

    def retreat(self, side, up_to=None):
        """A free hand that is out in front (or low) drawn back by a planned Cartesian path, clear of what it worked
        on: straight up to 1.0 m (at least 4 cm), then in towards the chest; the joint move home starts from there."""
        b = self.base()
        p, R = self.site(side)
        rel = _Rz(b[2]).T @ (p - np.array([b[0], b[1], 0.0]))
        if rel[0] < 0.30 and p[2] > 0.98:
            return True
        q = self.d.qpos[self.c.robot.arm_q[side]]
        if float(np.max(np.abs(q - np.array(HOME_Q[side])))) < 0.15:
            return True            # (the arm hangs at home: nothing to draw back; lifting it would only swing it about)
        h = self.c.ctrl.stand
        pitch = float(self.c.ctrl.waist[2])
        # (round 5: up to 1.0 m, but no higher than 8 cm below the shoulder: let go low with the knees bent, the
        # dustpan's handle, the hand rose to the face)
        bs = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{self.c.robot.name}/{side}_shoulder_pitch_link")
        z = max(p[2] + 0.04, min(1.0 if up_to is None else up_to, float(self.d.xpos[bs][2]) - 0.08))
        up = np.array([p[0], p[1], z])
        lat = 0.22 if side == "left" else -0.22
        inn = np.array([b[0], b[1], 0.0]) + _Rz(b[2]) @ np.array([0.26, lat, 0.0])
        inn[2] = z
        # the hand's rotation kept on the way up (a position-only solve may turn the wrist down into what is under
        # it); (round 5) drawn in towards the chest it turns to the arm's natural posture there (kept, the rotation of
        # a drop into a bin or of a put far over the counter folded the elbow to its limit and raised it over the
        # shoulder)
        st = self.stance
        q_in = None
        for sq in (np.array(NATURAL_CARRY_Q[side], float), np.asarray(q, float)):
            st.set_state((b[0], b[1], b[2]), h, pitch)
            q1_, ep, er = st.solve(side, inn, None, sq, iters=80, quick=30)
            if ep < 0.005:
                q_in = natural_resolve(self, side, inn, None, q1_)
                break
        if q_in is not None:
            # up by the planned line, then a checked joint move to the arm's natural posture there
            ok, worst, qs, why = self.stance.check((b[0], b[1], b[2]), h, pitch, side, [(up, R)])
            if ok:
                q_up = np.asarray(list(rows(qs))[-1], float)
                st.set_state((b[0], b[1], b[2]), h, pitch)
                st.chk.qpos[self.c.robot.arm_q[side]] = q_up
                path, why2 = st._joint_line(side, q_up, q_in, None, None, ())
                if path is not None:
                    self.run_path(side, qs, tol_end=0.03)
                    self.run_path(side, path, tol_end=0.03, refine=False)
                    return True
        cands = [[(up, R), (inn, R)], [(up, R)], [(p + np.array([0.0, 0.0, 0.03]), R)]]
        for poses in cands:
            ok, worst, qs, why = self.stance.check((b[0], b[1], b[2]), h, pitch, side, poses)
            if ok:
                self.run_path(side, qs, tol_end=0.03)
                return True
        self.log("[ref] retreat %s: no clear path (%s)" % (side, why))
        return False

    def home_safe(self, side):
        """The free arm home by a joint move checked in the stance model (the waist straightening with it); when that
        move would touch the room, the robot first steps straight back (up to 0.30 m) until it is clear."""
        r, d = self.c.robot, self.d
        home = np.array(HOME_Q[side])
        q0 = d.qpos[r.arm_q[side]].copy()
        if float(np.max(np.abs(q0 - home))) < 0.02:
            self.c.command({f"{side}_arm": {"q": [float(v) for v in home]}})
            return
        other = "right" if side == "left" else "left"
        h, p0 = self.c.ctrl.stand, float(self.c.ctrl.waist[2])
        held = [n for n in self.held.values() if n]
        st = self.stance

        def clear(base):
            # only contacts the move itself makes count (not what the other hand's object already touches); the
            # waist stays as it is during the move (it straightens afterwards, in stow)
            st.set_state(base, h, p0, {side: q0, other: d.qpos[r.arm_q[other]]})
            before = {(a_, b_) for a_, b_, _ in st.collisions(held=held)}
            for k in range(1, 11):
                u = k / 10.0
                st.set_state(base, h, p0, {side: q0 + (home - q0) * u, other: d.qpos[r.arm_q[other]]})
                if any((a_, b_) not in before for a_, b_, _ in st.collisions(held=held)):
                    return False
            return True

        b = self.base()
        c_, s_ = math.cos(b[2]), math.sin(b[2])
        found = False
        for k in range(0, 7):
            base = (b[0] - 0.05 * k * c_, b[1] - 0.05 * k * s_, b[2])
            if k and self.planner.occ[self.planner.cell(base[0], base[1])] and not self.stance_free(base, h, p0):
                break
            if clear(base):
                if k:
                    self.goto(base[0], base[1], base[2], tol=0.02)
                found = True
                break
        if not found and abs(p0) > 0.05:
            # straight up first (the arm rises and draws back with the torso), then the move from there
            self.posture(h, 0.0)
            p0 = 0.0
            q0 = d.qpos[r.arm_q[side]].copy()
            found = clear(tuple(self.base()))
        if not found:
            self.log("[ref] home %s: no clear move within 0.30 m back" % side)
        self.run_path(side, [home], tol_end=0.03, refine=False)

    def wait_arms(self, tol=0.02, timeout=4.0):
        """Until both arms are at their joint targets (and the waist at its)."""
        r = self.c.robot
        for _ in range(int(timeout / 0.05)):
            err = max(float(np.max(np.abs(self.d.qpos[r.arm_q[sd]] - self.c.ctrl.arm_cmd[sd])))
                      for sd in ("left", "right"))
            err = max(err, float(np.max(np.abs(self.d.qpos[r.waist_q] - self.c.ctrl.waist_cmd))))
            if err < tol or self.c.done:
                return True
            self.step(5)
        return False

    def stance_free(self, base, h, pitch=0.0, with_held=True):
        """The robot standing at `base` (arms as they are commanded) touches nothing of the room (`with_held`: nor does
        what the hands hold, as they hold it now)."""
        st = self.stance
        if not with_held:
            # (the arms at home: the arm that carries the item moves it up before the robot steps in)
            st.set_state(base, h, pitch, {sd: np.array(HOME_Q[sd], float) for sd in ("left", "right")})
            return not st.collisions(held=None)
        st.set_state(base, h, pitch, {sd: self.c.ctrl.arm_cmd[sd] for sd in ("left", "right")})
        return not st.collisions(held=[n for n in self.held.values() if n])

    def run_q(self, side, q, timeout=3.0, tol=0.012):
        """Joint targets for one arm, waiting until the arm is there."""
        self.c.command({f"{side}_arm": {"q": [float(v) for v in q]}})
        r = self.c.robot
        for _ in range(int(timeout / 0.05)):
            self.step(5)
            if float(np.max(np.abs(self.d.qpos[r.arm_q[side]] - np.asarray(q)))) < tol or self.c.done:
                break
        self.step(3)

    def run_qs(self, side, qs, timeout=3.0):
        for q in qs:
            self.run_q(side, q, timeout=timeout)

    def settle(self, side, ticks=12):
        """Hold the arm where it is (its present joints as the targets) until it is at rest."""
        r = self.c.robot
        self.c.command({f"{side}_arm": {"q": [float(v) for v in self.d.qpos[r.arm_q[side]]]}})
        self.step(ticks)

    def refine_path(self, side, qs, qcur):
        """The planned joint path with the arm's redundancy resolved as a person's (NaturalIK's posture term: the swivel
        that puts the elbow down and a little out, the wrist near its neutral, the shoulder untwisted), the hand's path
        unchanged: each planned configuration's grasp-site pose is reached again from the previous refined
        configuration, the posture drifting towards the natural one by at most 0.15 rad a waypoint (0.45 at the last),
        so the motion stays smooth. The hand, and what it holds, passes the same poses, so only the arm links can meet
        something new: a refined configuration that touches the room where the planned one did not, or a move between
        two that leaves the hand's path by more than the planned one did (+ 4 mm), keeps the planned path. Returns the
        refined rows or None."""
        st, r = self.stance, self.c.robot
        rws = [np.asarray(q, float) for q in rows(qs)]
        if not rws:
            return None
        b = self.base()
        other = "right" if side == "left" else "left"
        st.set_state((b[0], b[1], b[2]), float(self.c.ctrl.stand_cmd), float(self.c.ctrl.waist_cmd[2]),
                     {other: np.asarray(self.c.ctrl.arm_cmd[other], float)})
        prev = np.asarray(qcur, float)
        prev_l = prev.copy()
        prev_p = st.site_pose(side, prev)[0]
        out = []
        for k, ql in enumerate(rws):
            pl, Rl = st.site_pose(side, ql)
            q, ep, er = st.nik.solve(st.chk.qpos, side, pl, _quat(Rl), prev, iters=60,
                                     null_total=0.45 if k == len(rws) - 1 else 0.15, natural=True)
            if ep > 0.002 or er > 0.02:
                return None
            st.chk.qpos[r.arm_q[side]] = ql
            planned = {(a_, b_) for a_, b_, _ in st.collisions()}
            st.chk.qpos[r.arm_q[side]] = q
            if any((a_, b_) not in planned for a_, b_, _ in st.collisions()):
                return None
            mid = (prev_p + pl) / 2
            dev_l = float(np.linalg.norm(st.site_pose(side, (prev_l + ql) / 2)[0] - mid))
            dev_r = float(np.linalg.norm(st.site_pose(side, (prev + q) / 2)[0] - mid))
            if dev_r > max(0.006, dev_l + 0.004):
                return None
            out.append(q)
            prev, prev_l, prev_p = q, ql, pl
        return out

    def run_path(self, side, qs, tol_via=0.05, tol_end=0.01, timeout=2.0, rate=None, accel=None, refine=True):
        """Stream a planned joint path (entries of the planner: one row of joints, or several to pass through) along
        the joint-space polyline with a trapezoidal speed profile (at most `rate` rad/s, `accel` rad/s^2, in the
        joints' max norm): all joints move together, a target every 2 ticks. (Given a far joint target the controller
        moves every joint at its own top speed, so the joints arrive at different times and the hand leaves its path;
        a step start shakes a long tool out of the hand.) The last row is waited for to tol_end."""
        r = self.c.robot
        tool = self.held[side] in ("broom", "dustpan")
        rate = rate or (0.5 if tool else 1.2)
        accel = accel or (1.2 if tool else 5.0)
        qcur = np.asarray(self.c.ctrl.arm_cmd[side], float).copy()
        act = self.d.qpos[r.arm_q[side]]
        if float(np.max(np.abs(act - qcur))) > 0.05:
            qcur = act.copy()
        ref_rows = self.refine_path(side, qs, qcur) if refine else None
        pts = [qcur] + (ref_rows if ref_rows is not None else [np.asarray(q, float) for q in rows(qs)])
        seg = np.array([float(np.max(np.abs(pts[k + 1] - pts[k]))) for k in range(len(pts) - 1)])
        L = float(seg.sum())
        q = pts[-1]
        if L > 1e-6:
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            # trapezoid (or triangle) in arc length: accelerate, cruise, decelerate
            ta = rate / accel
            if accel * ta * ta > L:
                ta = math.sqrt(L / accel)
                vmax, tc = accel * ta, 0.0
            else:
                vmax, tc = rate, (L - accel * ta * ta) / rate
            T = 2 * ta + tc
            n = max(1, int(math.ceil(T / (2 * TICK))))
            hist = [qcur]
            pressing = 0
            for k in range(1, n + 1):
                t = T * k / n
                if t < ta:
                    sarc = 0.5 * accel * t * t
                elif t < ta + tc:
                    sarc = 0.5 * accel * ta * ta + vmax * (t - ta)
                else:
                    u = T - t
                    sarc = L - 0.5 * accel * u * u
                j = int(np.clip(np.searchsorted(cum, sarc, side="right") - 1, 0, len(seg) - 1))
                w = 0.0 if seg[j] < 1e-9 else (sarc - cum[j]) / seg[j]
                qk = pts[j] + (pts[j + 1] - pts[j]) * float(np.clip(w, 0.0, 1.0))
                self.c.command({f"{side}_arm": {"q": [float(v) for v in qk]}})
                self.step(2)
                hist.append(qk)
                if self.c.done:
                    return
                # (an arm link pressing on the room: back along the path and give the motion up, long before the
                # safety watch's 150 N for 0.2 s)
                pressing = pressing + 1 if self.body_push() > 90.0 else 0
                if pressing >= 2:
                    qb = hist[max(0, len(hist) - 8)]
                    self.log("[ref] arm %s: presses on the room (%.0f N); backing off" % (side, self.body_push()))
                    self.c.command({f"{side}_arm": {"q": [float(v) for v in qb]}})
                    self.step(15)
                    return
        self.c.command({f"{side}_arm": {"q": [float(v) for v in q]}})
        pressing = 0
        for _ in range(int(timeout / 0.02)):
            if float(np.max(np.abs(self.d.qpos[r.arm_q[side]] - q))) < tol_end or self.c.done:
                break
            self.step(2)
            pressing = pressing + 1 if self.body_push() > 90.0 else 0
            if pressing >= 2:
                self.log("[ref] arm %s: presses on the room at the end of its path (%.0f N); backing off" % (
                    side, self.body_push()))
                qb = np.asarray(pts[max(0, len(pts) - 2)], float) if len(pts) > 1 else q
                self.c.command({f"{side}_arm": {"q": [float(v) for v in qb]}})
                self.step(15)
                break

    def plan_go(self, side, poses, base0, heights, pitches, holding=None, offsets=None, low=False, what=""):
        """Find a stance for the poses (stance_search), walk there (for a low table: squat 0.15 m back first, then step
        in), take the posture. Returns the arm joints per pose, or None."""
        stance, qs = stance_search(self, side, poses, base0, heights, pitches, offsets=offsets, holding=holding)
        if stance is None:
            self.log("[ref] %s: no stance (%s)" % (what, qs))
            return None
        x, y, yaw, h, pitch = stance
        if low:
            c_, s_ = math.cos(yaw), math.sin(yaw)
            bx, by = x - 0.15 * c_, y - 0.15 * s_
            self.walk(bx, by, yaw, carrying=holding is not None)
            self.posture(h, pitch)
            self.goto(x, y, yaw, tol=0.006)
            self.low_spot = (bx, by, yaw)
        else:
            self.walk(x, y, yaw, carrying=holding is not None)
            self.posture(h, pitch)
            self.low_spot = None
        return qs

    def ik_err(self, side, p, R):
        q, ep, er = self.c.ctrl.ik.solve_global(self.d.qpos, side, np.asarray(p, float), _quat(R),
                                                self.c.ctrl.arm_cmd[side])
        return ep, er

    def best_frame(self, side, p, approach, pinch_options):
        """The first reachable hand frame among the pinch options (and their opposites)."""
        best = None
        for pin in pinch_options:
            for sgn in (1.0, -1.0):
                R = hand_frame(approach, sgn * np.asarray(pin, float))
                ep, er = self.ik_err(side, p, R)
                sc = ep + 0.1 * er
                if best is None or sc < best[0]:
                    best = (sc, R, ep, er)
        return best[1], best[2], best[3]

    # ============================================================================================ standing spots
    def release(self, side, settle=40):
        self.hand(side, False)
        self.step(settle)

    # ============================================================================================ spots
    def surface_of(self, p):
        """Which fixed surface an object at p rests on."""
        for name, ((x0, y0, x1, y1), h) in RM.surfaces().items():
            if x0 - 0.02 < p[0] < x1 + 0.02 and y0 - 0.02 < p[1] < y1 + 0.02 and abs(p[2] - h) < 0.12:
                return name
        return "floor"

    def leave_low(self):
        """Step back from a low table before standing up (the reverse of approach's two stages)."""
        if getattr(self, "low_spot", None) is not None:
            x, y, yaw = self.low_spot
            b = self.base()
            for sd in ("left", "right"):
                if self.held[sd]:
                    self.lift_clear_of_neighbours(sd, float(math.hypot(x - b[0], y - b[1])))
            self.goto(x, y, yaw, tol=0.02)
            self.low_spot = None

    # ============================================================================================ bins
    def lid_tab(self, slot, theta=None):
        """World position of a lid's tab at hinge angle theta (default: now) and the lid's rotation."""
        B = RM.BIN
        lt = B["lid_t"]
        L = B["x1"] - B["hinge_x"] + 0.01
        tab_local = np.array([L + B["tab_out"] - B["tab_r"], 0.0, lt / 2])
        yc = RM.BIN_SLOTS_Y[slot]
        hinge = np.array([B["hinge_x"], yc, B["rim_z"] + 0.002])
        if theta is None:
            theta = self.c.mech.lid_angle(slot)
        Rl = _Ry(-theta)          # rotation about -y by theta
        return hinge + Rl @ tab_local, Rl

    def lid(self, slot, open_=True, side="right"):
        """Open (past the hold-open spring) or close (below it: the lid falls shut) a bin's lid by its tab, the hand
        following the hinge's arc (a point grip: the tab turns in the hand)."""
        yc = RM.BIN_SLOTS_Y[slot]
        th0 = self.c.mech.lid_angle(slot)
        if open_ and th0 > RM.LID_HOLD + 0.05:
            return True
        if not open_ and th0 < 0.05:
            return True
        lat = 0.12 if side == "left" else -0.12
        bx, h, pitch = (0.64, 0.62, 0.35) if open_ else (0.55, 0.79, 0.30)
        self.stand(0.79, wait=False)
        self.waist(0.0, wait=False)
        tab, _ = self.lid_tab(slot)
        # (round 5: standing and leaning in; the closed lid's tab is 0.61 m up at the bin's front, open it is 0.88 m up
        # over the bin's back half)
        bx_r5 = 0.60
        th_goal = RM.LID_HOLD + math.radians(12) if open_ else RM.LID_HOLD - math.radians(22)
        end = self.lid_tab(slot, th_goal)[0]
        # (the tab in front of the hand's own shoulder: facing the wall, the robot's right is +y; round 4's nominal
        # stance had the tab on the other side of the body, reached across it)
        # (round 5) the tab taken from in front, the hand level or a little tilted, the pinch across the tab leaned
        # 30 deg towards it (the forearm then turned less than palm up or down): the frame whose grasp and arc's end
        # the arm reaches most naturally from the nominal stance, then the stance for it (ranked by the posture); by
        # the other hand if it is free and this one finds none (the bin by the south wall: the right hand's stance
        # would be against the wall)
        st = self.stance
        # (standing up, leaning a little: the tab and the arc's end both at about the forearm's height; squatting,
        # the arc's end is above the shoulder)
        hs = (0.79, 0.72)
        ps = (0.30, 0.45, 0.52, 0.15)
        chosen = None
        why_l = None
        n_frames = 0
        other_ = "right" if side == "left" else "left"
        for sd_ in [side] + ([other_] if not self.held[other_] else []):
            base0 = (bx_r5, yc + (0.18 if sd_ == "left" else -0.18), math.pi)
            if base0[1] < 0.35:
                # (the bin by the south wall: standing there the free hand, which the stance model leaves out,
                # met the wall at 245 N)
                continue
            frames = []
            for tilt in (60.0, 45.0, 30.0, 15.0, 0.0):
                a = tilted((-1.0, 0.0), tilt)
                nrm = np.cross(a, np.array([0.0, 1.0, 0.0]))
                nrm /= np.linalg.norm(nrm)
                for lean in (30.0, -30.0, 0.0):
                    pn = math.cos(math.radians(lean)) * nrm + math.sin(math.radians(lean)) * np.array([0.0, 1.0, 0.0])
                    for sgn in (1.0, -1.0):
                        Rh = hand_frame(a, sgn * pn)
                        st.set_state(base0, 0.79, 0.45)
                        best = None
                        for sq in (np.array(NATURAL_CARRY_Q[sd_], float), np.array(HOME_Q[sd_], float),
                                   np.array(REACH_SEEDS[sd_][2], float)):
                            q, ep, er = st.solve(sd_, tab, Rh, sq, iters=100, quick=30)
                            if ep > 0.004 or er > 0.04:
                                continue
                            q = natural_resolve(self, sd_, tab, Rh, q)
                            q2, ep2, _ = st.nik.solve(st.chk.qpos, sd_, end, None, q, iters=80, natural=True)
                            c = posture_cost(self, sd_, q) + (posture_cost(self, sd_, q2) if ep2 < 0.01 else 20.0)
                            if best is None or c < best:
                                best = c
                        if best is not None:
                            frames.append((best + 0.02 * tilt / 15.0, Rh, a))
            frames.sort(key=lambda f_: f_[0])
            n_frames += len(frames)
            for c, Rh, a in frames[:4]:
                pre = tab - a * 0.07 + np.array([0.0, 0.0, 0.03])
                stance, qs = stance_search(self, sd_, [(pre, Rh), (tab, Rh)], base0, hs, ps, rank=4)
                if stance is not None:
                    chosen = (stance, Rh, a, pre)
                    break
                why_l = qs
            if chosen is not None:
                if sd_ != side:
                    self.log("[ref] lid %d: by the %s hand" % (slot, sd_))
                side = sd_
                lat = 0.12 if side == "left" else -0.12
                break
        if chosen is None:
            self.log("[ref] lid %d: no standing stance (%d frames; last: %s); round 4's squat" % (
                slot, n_frames, why_l))
            # (round 4: the nominal stance, the cell's IK)
            if self.walk(bx, yc + lat * (-1.0), math.pi) is False:
                self.log("[ref] lid %d: the walk there failed" % slot)
                return False
            self.posture(h, pitch)
            a = tilted((-1.0, 0.0), 15.0)
            pin = np.cross(a, np.array([0.0, 1.0, 0.0]))
            R, ep, er = self.best_frame(side, tab, a, [pin])
            self.hand(side, False)
            self.arms({side: (tab + np.array([0.07, 0.0, 0.03]), R)}, timeout=3.0, tol=0.01)
            self.arms({side: (tab, R)}, timeout=2.5, tol=0.003)
        else:
            (x, y, yaw, h, pitch), Rh, a, pre = chosen
            if self.walk(x, y, yaw) is False:
                self.log("[ref] lid %d: the walk there failed" % slot)
                return False
            self.posture(h, pitch)
            b_ = self.base()
            self.hand(side, False)
            ok, worst, qs, why = st.check((b_[0], b_[1], b_[2]), h, pitch, side, [(pre, Rh), (tab, Rh)])
            if ok:
                self.run_path(side, qs)
                self.settle(side)
            else:
                self.arms({side: (pre, Rh)}, timeout=3.0, tol=0.01)
                self.arms({side: (tab, Rh)}, timeout=2.5, tol=0.003)
            ep = float(np.linalg.norm(self.site(side)[0] - tab))
        got = self.hand(side, True)
        if got != f"bin{slot}_lid":
            self.log("[ref] lid %d: closed on %s (reach %.3f)" % (slot, got, ep))
            self.hand(side, False)
            return False
        th_goal = RM.LID_HOLD + math.radians(12) if open_ else RM.LID_HOLD - math.radians(22)
        n = max(4, int(abs(th_goal - th0) / math.radians(1.0)))
        _, Rh = self.site(side)
        poses = [(self.lid_tab(slot, th0 + (th_goal - th0) * k / n)[0], None) for k in range(1, n + 1)]
        for k in range(0, n, 5):
            self.track(side, poses[k:k + 5], ticks=8, iters=6)
            if self.c.grips.holding(side) is None:
                self.log("[ref] lid %d: the grip slipped at %.2f rad" % (slot, self.c.mech.lid_angle(slot)))
                break
        self.step(25)
        self.release(side, settle=70)
        p, R = self.site(side)
        # (round 5) the free arm down at once: off the tab by 4 cm, then by a joint move to its natural posture in front
        # of the chest and home (drawn 12 cm back with the pinch's rotation kept and then up to 1.0 m, the open hand
        # rose beside the head, palm out; left there, it stayed up until the next move)
        self.retreat(side, up_to=float(p[2]) + 0.04)
        self.home_safe(side)
        self.step(60)
        th = self.c.mech.lid_angle(slot)
        ok = th > RM.LID_HOLD if open_ else th < 0.08
        if not ok:
            self.log("[ref] lid %d %s: angle %.2f" % (slot, "open" if open_ else "close", th))
        return ok

    def bin_slot(self, kind):
        return next(s for s, k in self.c.meta["bins"].items() if k == kind)

    def home_arm(self, side):
        """A free arm home: drawn back clear first, then a checked joint move."""
        self.retreat(side)
        self.home_safe(side)
        self.step(60)


# ================================================================================================ stance planning
class Stance:
    """Checks a robot stance (base pose, stand height, waist pitch) against a sequence of grasp-site poses on a
    scratch copy of the state: each pose solved by the cell's IK (from the previous solution), every configuration
    checked for penetration of the robot's links and the held object into the room."""

    MARGIN_ROBOT, MARGIN_HELD = 0.010, 0.008
    PATH_NULL = 0.10        # rad: the posture change a path's waypoint may make (NaturalIK null_total)

    def __init__(self, cell):
        from .robot import FLOOR, HELD, OBJ, ROBOT, WORLD, LegPosture
        self.c = cell
        # the check model: the cell's model with the robot's colliders inflated by MARGIN_ROBOT and the loose objects'
        # by MARGIN_HELD (a planned configuration keeps that much clearance; the executed one strays by millimetres)
        from .scene import build
        sp, _ = build(cell.layout, decor=False)       # the same model (deterministic), without the drawn decor
        G = mujoco.mjtGeom
        for g in sp.geoms:
            inflate = MARGIN = None
            if g.contype & ROBOT:
                MARGIN = self.MARGIN_ROBOT
            elif g.contype & OBJ:
                MARGIN = self.MARGIN_HELD
            if MARGIN is None:
                continue
            sz = list(g.size)
            if g.type in (G.mjGEOM_CAPSULE, G.mjGEOM_SPHERE, G.mjGEOM_CYLINDER):
                sz[0] += MARGIN
                if g.type == G.mjGEOM_CYLINDER:
                    sz[1] += MARGIN
            elif g.type == G.mjGEOM_BOX:
                sz = [v + MARGIN for v in sz]
            elif g.type == G.mjGEOM_ELLIPSOID:
                sz = [v + MARGIN for v in sz]
            g.size = sz
            _ = inflate
        self.m = sp.compile()
        self.chk = mujoco.MjData(self.m)
        self.legs = LegPosture(cell.m, cell.robot)
        r = cell.robot
        self.r = r
        m = cell.m
        self.robot_geoms = set(int(g) for g in r.geom_ids if m.geom_contype[g] & ROBOT)
        self.hand_geoms = set()
        for side in ("left", "right"):
            hb = set(int(b) for b in r.hand_bodies[side])
            self.hand_geoms |= {g for g in self.robot_geoms if int(m.geom_bodyid[g]) in hb}
        self.nik = NaturalIK(cell.m, r)      # the planner's IK (the cell's, with the posture term)
        self.world_bit, self.held_bit = WORLD, HELD
        self.obstacle_bits = WORLD | OBJ        # the room's fixed geometry and the loose objects lying about
        self.held_obstacle_bits = WORLD | OBJ | FLOOR      # what a held object may not touch either: the floor

    def set_state(self, base, h, pitch, arm_q=None):
        m, d, chk, r = self.m, self.c.d, self.chk, self.r
        # the grips' collision bits (a held object's HELD bit) live in the cell's model: mirror them
        n = self.m.ngeom
        self.m.geom_contype[:] = self.c.m.geom_contype[:n]
        self.m.geom_conaffinity[:] = self.c.m.geom_conaffinity[:n]
        chk.qpos[:] = d.qpos
        chk.qpos[r.base_q[:3]] = base
        chk.qpos[r.base_q[3]] = h - PELVIS_HEIGHT
        legs = self.legs.targets(h)
        for s in ("left", "right"):
            chk.qpos[r.leg_q[s]] = legs
        chk.qpos[r.waist_q] = [0.0, 0.0, pitch]
        if arm_q:
            for s, q in arm_q.items():
                chk.qpos[r.arm_q[s]] = q

    def place_held(self, side, name, grip_rel):
        """Put the held object where the hand's grasp site holds it (in the scratch state)."""
        m, chk = self.m, self.chk
        mujoco.mj_kinematics(m, chk)
        sid = self.r.grasp_site[side]
        ps, Rs = chk.site_xpos[sid], chk.site_xmat[sid].reshape(3, 3)
        pg, Rg = grip_rel
        Ro = Rs @ Rg.T
        po = ps - Ro @ pg
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
        a = m.jnt_qposadr[j]
        chk.qpos[a:a + 3] = po
        chk.qpos[a + 3:a + 7] = _quat(Ro)
        # the broom's head swivels about its long axis (the handle frame's x) to stay level as the handle tilts:
        # modelled so (at its own angle it would dip its edge into what it sweeps towards)
        js = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_swivel")
        if js >= 0:
            phi = math.atan2(-float(Ro[2, 1]), float(Ro[2, 2]))
            lo, hi = m.jnt_range[js]
            chk.qpos[m.jnt_qposadr[js]] = float(np.clip(phi, lo, hi))

    def collisions(self, held=None, tol=0.003, allow=()):
        """Penetrations of the robot's links (not the hands) and of the held objects into the room's fixed geometry
        (a held object may touch the geoms named in `allow`: the surface it works on)."""
        m, chk = self.m, self.chk
        mujoco.mj_kinematics(m, chk)
        mujoco.mj_collision(m, chk)
        held_bodies = set()
        if held:
            for n in held:
                held_bodies.add(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n))
            # and their child bodies (the broom's swivelling head)
            grow = True
            while grow:
                grow = False
                for bi in range(1, m.nbody):
                    if bi not in held_bodies and int(m.body_parentid[bi]) in held_bodies:
                        held_bodies.add(bi)
                        grow = True
        bad = []
        for i in range(chk.ncon):
            ct = chk.contact[i]
            if ct.dist > -tol:
                continue
            g1, g2 = int(ct.geom1), int(ct.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if int(m.geom_bodyid[a]) in held_bodies and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b) or "") \
                        in allow:
                    continue
                other_free = int(m.geom_bodyid[b]) not in held_bodies
                obstacle = bool(m.geom_contype[b] & self.obstacle_bits) and other_free
                held_obstacle = bool(m.geom_contype[b] & self.held_obstacle_bits) and other_free
                if (a in self.robot_geoms and a not in self.hand_geoms and obstacle) or \
                        (int(m.geom_bodyid[a]) in held_bodies and held_obstacle):
                    key = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[a]),
                           mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b) or "?")
                    if key in getattr(self, "_ignore", ()):
                        break
                    bad.append(key + (round(float(ct.dist), 4),))
                    break
        return bad

    def site_pose(self, side, q):
        """The grasp site's pose with the arm at q (in the scratch state as set)."""
        self.chk.qpos[self.r.arm_q[side]] = q
        mujoco.mj_kinematics(self.m, self.chk)
        sid = self.r.grasp_site[side]
        return self.chk.site_xpos[sid].copy(), self.chk.site_xmat[sid].reshape(3, 3).copy()

    def _follow(self, side, poses, q, holding, grip_rel, allow, iters, prev=None, strict=0):
        """Each pose solved by the cell's IK from the previous solution (from q first) and checked for collisions:
        (ok, worst position error, joints per pose solved, first failure). The first `strict` poses (a transit) may
        touch nothing, not even what `allow` names (the surface worked on, touched only at the poses themselves)."""
        out, worst = [], 0.0
        for k_pose, (p, R) in enumerate(poses):
            q0 = np.asarray(q, float).copy()
            q, ep, er = self.solve(side, p, R, q, iters, null_total=self.PATH_NULL)
            worst = max(worst, ep)
            if ep > 0.005 or (R is not None and er > 0.04):
                return False, worst, out, ("unreachable", round(ep, 4), round(er, 3))
            mids = []
            if prev is not None:
                mids = self._between(side, q0, prev[0], prev[1], q, p, R)
                if mids is None:
                    return False, worst, out, ("leaves the line", round(ep, 4), round(er, 3))
            for qq in mids + [q]:
                self.chk.qpos[self.r.arm_q[side]] = qq
                if holding:
                    self.place_held(side, holding, grip_rel)
                bad = self.collisions(held=[holding] if holding else None, allow=allow if k_pose >= strict else ())
                if bad:
                    return False, worst, out, bad[0]
            self.n_mid = getattr(self, "n_mid", 0) + len(mids)
            # one entry per pose: the joints to pass through on the way (rows), its own solution last
            out.append(np.vstack([np.asarray(m) for m in mids] + [q.copy()]) if mids else q.copy())
            prev = (np.asarray(p, float), R)
        return True, worst, out, None

    def solve(self, side, p, R, q, iters=60, quick=20, give_up=0.03, w_home=0.0, null_total=None):
        """The planner's IK (NaturalIK: the cell's damped least squares with the posture term in the arm's null space)
        in two stages: `quick` iterations, then (unless clearly out of reach) the rest. A failed solve is the planner's
        main cost. A waypoint of a path (null_total set) changes the posture only gradually, so the joint-space move
        from the previous waypoint's solution stays on the Cartesian line. (w_home: ignored.)"""
        ik = self.nik
        quat = None if R is None else _quat(R)
        p = np.asarray(p, float)
        q1, ep, er = ik.solve(self.chk.qpos, side, p, quat, q, iters=min(quick, iters), null_total=null_total)
        if not ((ep <= 0.005 and (R is None or er <= 0.04)) or ep > give_up or iters <= quick):
            q1, ep, er = ik.solve(self.chk.qpos, side, p, quat, q1, iters=iters - quick,
                                  null_total=None if null_total is None else 0.0)
        if (ep > 0.005 or (R is not None and er > 0.04)) and ep <= give_up and iters > quick:
            # (what the posture term cannot reach, the cell IK alone may: its answer, from the same start)
            q2, ep2, er2 = self.c.ctrl.ik.solve(self.chk.qpos, side, p, quat, q, iters=iters, w_home=0.0)
            if ep2 + 0.25 * er2 < ep + 0.25 * er:
                return q2, ep2, er2
        return q1, ep, er

    def _between(self, side, qa, pa, Ra, qb, pb, Rb, depth=0):
        """Joints to insert between two solved waypoints so that the joint-space moves keep the grasp site within
        4 mm (0.05 rad) of the straight Cartesian segment; [] when the plain move does, None when it cannot."""
        qm = (qa + qb) / 2
        pm_j, Rm_j = self.site_pose(side, qm)
        pm = (np.asarray(pa) + np.asarray(pb)) / 2
        dev = float(np.linalg.norm(pm_j - pm))
        rdev = 0.0
        Rm = None
        if Ra is not None and Rb is not None:
            dR = Rb @ Ra.T
            ang = math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)))
            ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            ax = ax / np.linalg.norm(ax) if np.linalg.norm(ax) > 1e-9 else np.array([0.0, 0.0, 1.0])
            Rm = _axis_angle(ax, ang / 2) @ Ra
            e = Rm_j @ Rm.T
            rdev = math.acos(float(np.clip((np.trace(e) - 1) / 2, -1.0, 1.0)))
        if dev < 0.004 and rdev < 0.03:
            return []
        if depth >= 4:
            return None
        q, ep, er = self.solve(side, pm, Rm, qm, iters=40, null_total=0.0)
        if ep > 0.005 or (Rm is not None and er > 0.04):
            return None
        left = self._between(side, qa, pa, Ra, q, pm, Rm, depth + 1)
        right = self._between(side, q, pm, Rm, qb, pb, Rb, depth + 1)
        if left is None or right is None:
            return None
        return left + [q] + right

    def _joint_line(self, side, q0, q1, holding, grip_rel, allow, step=0.08):
        """The straight joint-space move q0 -> q1 in steps of at most `step` rad, each checked: (joints, None) or
        (None, the first collision)."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = int(max(1, math.ceil(float(np.max(np.abs(q1 - q0))) / step)))
        out = []
        for k in range(1, n + 1):
            q = q0 + (q1 - q0) * (k / n)
            self.chk.qpos[self.r.arm_q[side]] = q
            if holding:
                self.place_held(side, holding, grip_rel)
            bad = self.collisions(held=[holding] if holding else None, allow=allow)
            if bad:
                return None, bad[0]
            out.append(q.copy())
        return out, None

    def check(self, *args, **kwargs):
        """See _check (the pre-existing contacts it ignores apply to that call only)."""
        self._ignore = set()
        try:
            return self._check(*args, **kwargs)
        finally:
            self._ignore = set()

    def _check(self, base, h, pitch, side, poses, holding=None, grip_rel=None, seed=None, iters=60, transit=True,
               allow=(), fallback=True, start="held", back=0.0, hold=False):
        """(ok, worst position error, [arm joints], first failure). With `transit`, the path starts at the arm's
        present joints and reaches the first pose either through Cartesian waypoints every 3 cm / 0.1 rad or, when
        that line is blocked or leaves the arm's reach, by a checked joint-space move to the first pose solved from
        natural reaching postures; those joints lead the returned list (self.n_via of them before the first pose's)."""
        r = self.r
        self.set_state(base, h, pitch)
        if seed is None:
            # where the arm is (it may still be on its way to its targets)
            act = self.c.d.qpos[r.arm_q[side]]
            cmd = self.c.ctrl.arm_cmd[side]
            seed = act if float(np.max(np.abs(act - cmd))) > 0.05 else cmd
            # (round 5: a put planned for an arm that will be in another posture by then, go_put's raise for the
            # counter on the way)
            if getattr(self, "seed_override", None) is not None and side == self.seed_override[0]:
                seed = self.seed_override[1]
        q0 = np.array(seed, float)
        poses = list(poses)
        self.n_via = 0
        if back > 0 and transit and poses and start == "full":
            return self._check_back(base, h, pitch, side, poses, holding, grip_rel, q0, iters, allow, back, hold=hold)
        if transit and poses and start:
            # the start itself: the arms as they are, at this stance, standing up straight (the robot walks in so)
            # and in the posture (it then bends with them); start="held" (a re-plan where the robot already stands)
            # checks only what the hands hold
            other = "right" if side == "left" else "left"
            # (a full check also covers the posture change in between: bending at a shelf with a held book in front
            # sweeps it down through what it then reaches over)
            postures = ((base, h, pitch),)
            if start == "full":
                postures = ((base, h, pitch), (base, 0.79, 0.0)) + tuple(
                    (base, 0.79 + (h - 0.79) * u, pitch * u) for u in (0.2, 0.4, 0.6, 0.8)
                    if abs(h - 0.79) > 0.01 or abs(pitch) > 0.05)
            hold_pose = None
            if hold and start == "full" and holding is not None:
                # (round 5) the hand keeps the place the carry gives it standing up straight while the robot bends
                # (Reference.posture's hold): the arm solved for it at each step of the bend, upright first
                self.set_state(base, 0.79, 0.0)
                hold_pose = self.site_pose(side, q0)
                postures = ((base, 0.79, 0.0),) + tuple(
                    (base, 0.79 + (h - 0.79) * u, pitch * u) for u in (0.25, 0.5, 0.75)
                    if abs(h - 0.79) > 0.01 or abs(pitch) > 0.05) + ((base, h, pitch),)
            q_prev = q0
            for bse, hh, pp in postures:
                self.set_state(bse, hh, pp)
                qa = q0
                if hold_pose is not None:
                    qa, ep_h, er_h = self.nik.solve(self.chk.qpos, side, hold_pose[0], _quat(hold_pose[1]), q_prev,
                                                    iters=40, null_total=0.3, natural=True)
                    if ep_h > 0.006 or er_h > 0.04:
                        return False, 0.0, [], ("hold unreachable", round(ep_h, 4), round(er_h, 3))
                    q_prev = qa
                self.chk.qpos[r.arm_q[side]] = qa
                self.chk.qpos[r.arm_q[other]] = self.c.d.qpos[r.arm_q[other]]
                held = []
                for sd in ("left", "right"):
                    hn = self.c.grips.holding(sd)
                    if hn and sd in getattr(self, "grip_rels", {}):
                        self.place_held(sd, hn, self.grip_rels[sd])
                        held.append(hn)
                bad = self.collisions(held=held or None, allow=allow)
                if start != "full":
                    # a re-plan where the robot stands: only what this hand holds (the other's is not moving)
                    mine = self.c.grips.holding(side)
                    bad = [b_ for b_ in bad if b_[0] == mine]
                if bad:
                    return False, 0.0, [], ("start collides",) + tuple(bad[0])
            if hold_pose is not None:
                q0 = np.asarray(q_prev, float).copy()           # (the transit starts from the arm so, bent)
            self.set_state(base, h, pitch)
        # contacts the robot already makes where it stands (a leg by the dustpan it just set down) do not block the
        # arm's path: only new ones count
        self._ignore = set()
        if start != "full":
            self.chk.qpos[r.arm_q[side]] = q0
            self._ignore = {(a_, b_) for a_, b_, _ in self.collisions(held=[holding] if holding else None)}
        if not (transit and poses):
            return self._follow(side, poses, q0, holding, grip_rel, allow, iters)
        # (0) a planner's preferred posture (prefer_seeds): a checked joint-space move to the first pose solved from
        # it, so the arm keeps that posture's branch
        for sq in (getattr(self, "prefer_seeds", None) or []):
            self.set_state(base, h, pitch)
            self.chk.qpos[r.arm_q[side]] = q0
            q1, ep, er = self.solve(side, poses[0][0], poses[0][1], np.array(sq, float), iters=150, quick=40)
            if ep > 0.004 or (poses[0][1] is not None and er > 0.03):
                continue
            self.chk.qpos[r.arm_q[side]] = q0
            path, why = self._joint_line(side, q0, q1, holding, grip_rel, ())
            if path is None:
                continue
            ok, worst2, out, why = self._follow(side, poses[1:], q1, holding, grip_rel, allow, iters,
                                                prev=(np.asarray(poses[0][0], float), poses[0][1]))
            if ok:
                self.n_via = len(path) - 1
                return True, max(worst2, ep), path + out, None
            self.set_state(base, h, pitch)
            self.chk.qpos[r.arm_q[side]] = q0
        # (1) the hand along a straight line (and a steady turn) from where it is to the first pose
        p0, R0 = self.site_pose(side, q0)
        p1, R1 = poses[0]
        R1x = R0 if R1 is None else R1
        dR = R1x @ R0.T
        ang = math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)))
        n = int(max(1, math.ceil(max(np.linalg.norm(np.asarray(p1) - p0) / 0.03, ang / 0.10))))
        axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-9 else np.array([0.0, 0.0, 1.0])
        via = []
        for k in range(1, n):
            u = k / n
            via.append((p0 + (np.asarray(p1) - p0) * u, (_axis_angle(axis, ang * u) @ R0) if R1 is not None else None))
        self.chk.qpos[r.arm_q[side]] = q0
        # (the transit to the first pose touches nothing: `allow` is for the poses themselves)
        ok, worst, out, why = self._follow(side, via + poses, q0, holding, grip_rel, allow, iters,
                                           prev=(p0, R0 if R1 is not None else None), strict=len(via))
        if ok:
            self.n_via = len(via)
            return ok, worst, out, why
        why_line = why
        if not fallback:
            return False, worst, [], why_line
        # (2) a checked joint-space move to the first pose, solved from natural reaching postures (round 5: the
        # natural carry's posture among them: the arm forward, the wrist straight)
        tried = []
        for sq in [q0, np.array(NATURAL_CARRY_Q[side], float)] + [np.array(v, float) for v in REACH_SEEDS[side]]:
            self.set_state(base, h, pitch)
            self.chk.qpos[r.arm_q[side]] = q0
            q1, ep, er = self.solve(side, p1, R1, sq, iters=150, quick=40)
            if ep > 0.004 or (R1 is not None and er > 0.03):
                continue
            if any(float(np.max(np.abs(q1 - t))) < 0.05 for t in tried):
                continue
            tried.append(q1)
            self.chk.qpos[r.arm_q[side]] = q0
            path, why = self._joint_line(side, q0, q1, holding, grip_rel, ())
            if path is None:
                continue
            ok, worst2, out, why = self._follow(side, poses[1:], q1, holding, grip_rel, allow, iters,
                                                prev=(np.asarray(p1, float), R1))
            if ok:
                self.n_via = len(path) - 1
                return True, max(worst2, ep), path + out, None
        return False, worst, [], why_line


def _stance_check_back(self, base, h, pitch, side, poses, holding, grip_rel, q0, iters, allow, back, hold=False):
    """A stance reached bent (go_put's low mode): the robot walks in upright to `back` metres short of it, bends there
    (a held item, carried in front of the chest, would otherwise sweep down through what the robot reaches over, and
    it tilts with the torso), moves the hand to the first pose's place relative to the body (that pose shifted back by
    `back`: the item level, in front of what it goes into) and steps in so. Every configuration of that is checked:
    (ok, worst position error, joints from the first pose on, first failure)."""
    r = self.r
    other = "right" if side == "left" else "left"
    c_, s_ = math.cos(base[2]), math.sin(base[2])
    bb = (base[0] - back * c_, base[1] - back * s_, base[2])

    def held_bad(bse, hh, pp, q):
        self.set_state(bse, hh, pp)
        self.chk.qpos[r.arm_q[side]] = q
        self.chk.qpos[r.arm_q[other]] = self.c.d.qpos[r.arm_q[other]]
        held = []
        for sd in ("left", "right"):
            hn = self.c.grips.holding(sd)
            if hn and sd in getattr(self, "grip_rels", {}):
                self.place_held(sd, hn, self.grip_rels[sd])
                held.append(hn)
        return self.collisions(held=held or None)

    # upright at the back-off point, then bending there (round 5, `hold`: the hand keeping its place meanwhile, as
    # Reference.posture holds it)
    hold_pose = None
    if hold and holding is not None:
        self.set_state(bb, 0.79, 0.0)
        hold_pose = self.site_pose(side, q0)
    qh = np.asarray(q0, float).copy()
    for hh, pp in [(0.79, 0.0)] + [(0.79 + (h - 0.79) * u, pitch * u) for u in (0.2, 0.4, 0.6, 0.8, 1.0)]:
        if hold_pose is not None:
            self.set_state(bb, hh, pp)
            qh, ep_h, er_h = self.nik.solve(self.chk.qpos, side, hold_pose[0], _quat(hold_pose[1]), qh, iters=40,
                                            null_total=0.3, natural=True)
            if ep_h > 0.006 or er_h > 0.04:
                return False, 0.0, [], ("hold unreachable", round(ep_h, 4), round(er_h, 3))
        bad = held_bad(bb, hh, pp, qh)
        if bad:
            return False, 0.0, [], ("start collides",) + tuple(bad[0])
    q0 = qh
    # bent there: the hand to the first pose's place relative to the body (a transit: it touches nothing)
    p1, R1 = poses[0]
    pre = (np.asarray(p1, float) - back * np.array([c_, s_, 0.0]), R1)
    ok, worst, out, why = self._check(bb, h, pitch, side, [pre], holding=holding, grip_rel=grip_rel, seed=q0,
                                      iters=iters, allow=(), fallback=True, start="held")
    if not ok:
        return False, worst, [], why
    q1 = np.asarray(last(out[-1]), float)
    # stepping in bent, the arm as it is
    for v in (0.25, 0.5, 0.75, 1.0):
        bad = held_bad((bb[0] + back * c_ * v, bb[1] + back * s_ * v, base[2]), h, pitch, q1)
        if bad:
            return False, worst, [], ("step-in collides",) + tuple(bad[0])
    # at the stance: the rest of the poses from the first
    self.set_state(base, h, pitch)
    self.chk.qpos[r.arm_q[side]] = q1
    self._ignore = set()
    ok2, worst2, out2, why2 = self._follow(side, poses[1:], q1, holding, grip_rel, allow, iters,
                                           prev=(np.asarray(p1, float), R1))
    if not ok2:
        return False, max(worst, worst2), [], why2
    self.n_via = 0
    return True, max(worst, worst2), [q1.copy()] + out2, None


Stance._check_back = _stance_check_back


def path_posture(ref, side, base, h, pitch, qs, poses):
    """How unnatural the arm is along a planned path (the human measure of the posture the executed path resolves at
    each pose: the worst plus a third of the mean), the robot at (base, h, pitch)."""
    st = ref.stance
    st.set_state(base, h, pitch)
    cs = []
    for entry, (p, R) in zip(qs[-len(poses):], poses):
        q = natural_resolve(ref, side, p, R, last(entry), iters=40)
        cs.append(posture_cost(ref, side, q))
    if not cs:
        return 0.0
    return max(cs) + sum(cs) / (3.0 * len(cs))


def stance_search(ref, side, poses, base0, heights, pitches, offsets=None, holding=None, yaw_offsets=(0.0,),
                  allow=(), passes=(False, True), back=0.0, rank=0, hold=False):
    """The first stance (x, y, yaw, h, pitch) near base0 from which every pose is reachable without collision, trying
    the offsets (forward, lateral metres in the base frame) in order: every stance with the straight approach line
    first, then again with the joint-space approach as well. With `rank` (round 5), of the first `rank` such stances
    of a pass the one whose path the arm takes most naturally (path_posture; the body chosen with the arm: where the
    robot stands, how far it leans, how low it stands). Returns (stance, joints) or (None, why)."""
    st = ref.stance
    grip = ref.grip_rel.get(side) if holding else None
    x0, y0, yaw0 = base0
    if offsets is None:
        offsets = [(0.0, 0.0), (-0.04, 0.0), (0.04, 0.0), (0.0, 0.04), (0.0, -0.04), (-0.08, 0.0), (0.0, 0.08),
                   (0.0, -0.08), (-0.04, 0.04), (-0.04, -0.04), (0.08, 0.0), (-0.12, 0.0)]
    why = None
    t0, n = time.time(), 0
    stances = []
    for h in heights:
        for pitch in pitches:
            for dyaw in yaw_offsets:
                yaw = yaw0 + dyaw
                c, s = math.cos(yaw), math.sin(yaw)
                for f, l in offsets:
                    base = (x0 + f * c - l * s, y0 + f * s + l * c, yaw)
                    # (stepped into with the item raised first, `back`: the item as carried now does not count there)
                    if ref.planner.occ[ref.planner.cell(base[0], base[1])] and \
                            not ref.stance_free(base, h, with_held=back <= 0.0):
                        continue
                    stances.append((base, h, pitch))
    for fallback in passes:
        found = []
        for base, h, pitch in stances:
            ok, worst, qs, why = st.check(base, h, pitch, side, poses, holding=holding, grip_rel=grip, allow=allow,
                                          fallback=fallback, start="full", back=back, hold=hold)
            n += 1
            if ok:
                if rank <= 1:
                    st.n_via_saved = st.n_via
                    ref.plan_stats.append((n, time.time() - t0, True))
                    return (base[0], base[1], base[2], h, pitch), qs
                # (round 5: and the legs: a deep squat costs, so of stances the arm takes alike the robot stands
                # taller; the human measure sees the arm only)
                c = path_posture(ref, side, base, h, pitch, qs, poses) + 0.03 * len(found) + \
                    1.0 * (max(0.0, 0.79 - h) / 0.1) ** 2
                found.append((c, (base[0], base[1], base[2], h, pitch), qs, st.n_via))
                if len(found) >= rank or c < 0.4:
                    break                   # (a stance the arm takes without effort needs no rival)
        if found:
            c, stance, qs, nv = min(found, key=lambda f_: f_[0])
            st.n_via_saved = nv
            ref.last_rank_cost = c
            ref.plan_stats.append((n, time.time() - t0, True))
            ref.log("[ref]   stance ranked: %d feasible, posture %.2f (pitch %.2f h %.2f)" % (len(found), c, stance[4],
                                                                                            stance[3]))
            return stance, qs
    ref.plan_stats.append((n, time.time() - t0, False))
    return None, why


# ================================================================================================ planned skills
def _nominal(ref, surf, p, side, flip=False):
    """(base0, heights, pitches, low) to work at an object on `surf` at p with `side`'s hand (`flip`: on the dining
    table, from the other long side than the nearer one)."""
    lat = 0.12 if side == "left" else -0.12
    if surf == "coffee_top":
        C = RM.COFFEE
        cy, cl = C["center"][1], C["size"][1]
        y = float(np.clip(p[1] - lat, cy - cl / 2 + C["leg_inset"] + 0.21, cy + cl / 2 - C["leg_inset"] - 0.21))
        return (C["center"][0] - C["size"][0] / 2 - 0.17, y, 0.0), (0.62, 0.55, 0.50), (0.52, 0.45, 0.35), True
    if surf == "dining_top":
        D = RM.DINING
        (dx, dy), (lx, ly) = D["center"], D["size"]
        south = p[1] < dy
        if name_of_book(ref, p) is not None:
            _, Rb = ref.pose(name_of_book(ref, p))
            south = float(-Rb[1, 0]) < 0.0           # the spine (body -x) faces south: work from the south side
        if flip:
            south = not south
        if south:
            base = (float(np.clip(p[0] + lat, dx - lx / 2 + 0.10, dx + lx / 2 - 0.10)), dy - ly / 2 - 0.20, math.pi / 2)
        else:
            base = (float(np.clip(p[0] - lat, dx - lx / 2 + 0.10, dx + lx / 2 - 0.10)), dy + ly / 2 + 0.20, -math.pi / 2)
        return base, (0.79,), (0.30, 0.45, 0.52, 0.15), False
    if surf in ("counter", "sink", "dish_tray"):
        return (RM.KITCHEN["top_x1"] + 0.16, p[1] + lat, math.pi), (0.79,), (0.30, 0.20, 0.45, 0.52), False
    return None


def name_of_book(ref, p):
    for k in O.BOOKS:
        if np.linalg.norm(ref.pose(k)[0] - p) < 1e-6:
            return k
    return None


def _grasp_frames(ref, name, side, base0):
    """Candidate (grasp point, approach, hand frame) for taking `name` from a robot standing at base0."""
    kind = O.LOOSE[name][0]
    p, R = ref.pose(name)
    to = np.array([p[0] - base0[0], p[1] - base0[1]])
    to /= max(1e-9, np.linalg.norm(to))
    flat = np.array([to[0], to[1], 0.0])
    lat = np.array([-to[1], to[0], 0.0])
    z = np.array([0.0, 0.0, 1.0])
    out = []
    if kind in ("can", "bottle", "glass"):
        # (round 5) from the side with the hand nearly level, at about the middle of its height, the way a person takes
        # a glass (the forearm forward, the wrist straight, the hand round it): from the robot's side and round the
        # object; then round 4's steeper grasps near the top
        hgt = {"can": 0.066, "bottle": 0.095, "glass": 0.062}[kind]
        g = p + R[:, 2] * hgt
        for tilt in (8.0, 20.0, 32.0):
            for daz in (0.0, -0.4, 0.4, -0.8, 0.8):
                d2 = _rot2(to, daz)
                a = tilted(d2, tilt)
                l2 = np.array([-d2[1], d2[0], 0.0])
                for sgn in (1.0, -1.0):
                    out.append((g, a, hand_frame(a, sgn * l2)))
        hgt = {"can": 0.075, "bottle": 0.10, "glass": 0.072}[kind]
        g = p + R[:, 2] * hgt
        for tilt in (62.0, 50.0, 40.0, 30.0):
            a = tilted(to, tilt)
            for sgn in (1.0, -1.0):
                out.append((g, a, hand_frame(a, sgn * lat)))
    elif kind in ("napkin", "apple"):
        # (round 5) a crumpled napkin or the core is taken any way round: the hand a little tilted from the robot's
        # side and round it first (the wrist straight), then round 4's steeper grasps from above
        for tilt in (28.0, 38.0):
            for daz in (0.0, -0.45, 0.45):
                d2 = _rot2(to, daz)
                a = tilted(d2, tilt)
                l2 = np.array([-d2[1], d2[0], 0.0])
                for pin in (l2, -l2):
                    out.append((p.copy(), a, hand_frame(a, pin)))
        for tilt in (55.0, 45.0, 65.0, 35.0):
            a = tilted(to, tilt)
            for pin in (lat, -lat):
                out.append((p.copy(), a, hand_frame(a, pin)))
    elif kind == "plate":
        # pinched at the rim (the pinch across the plate, within 35 deg of its axis), anywhere round the near half of
        # the rim; (round 5) the pinch also leaned 20 deg to the side (the forearm then turned less) and the approach
        # flatter, so the plate is carried level with the wrist straight
        rim = O.PLATE["r"] - 0.022
        for daz in (0.0, -0.35, 0.35):
            d2 = _rot2(to, daz)
            fl2 = np.array([d2[0], d2[1], 0.0])
            g = p - fl2 * rim + z * (O.PLATE["h"] / 2)
            # (and 38 deg: set down on the dish tray, a plate pinched with a flatter hand puts the wrist into the
            # counter's top or the tray's rim, 2 cm below the plate's rim)
            for tilt in (25.0, 18.0, 30.0, 38.0):
                a = tilted(d2, tilt)
                pin = z - (z @ a) * a
                pin /= np.linalg.norm(pin)
                side_v = np.cross(a, pin)
                for lean in (0.0, 20.0, -20.0):
                    if tilt > 26.0 and lean:
                        continue
                    pn = math.cos(math.radians(lean)) * pin + math.sin(math.radians(lean)) * side_v
                    for sgn in (1.0, -1.0):
                        out.append((g, a, hand_frame(a, sgn * pn)))
    if kind == "sponge":
        # (round 5) by the long edge facing the robot, pinched through its thickness (feature k = 2) with the hand
        # nearly level and the pinch leaned 25-30 deg to the side (within the grip's 35 deg of the thin axis): the
        # forearm then turned about 1 rad, not palm down, and wiping (the sponge's length across the stroke), rinsing
        # and carrying it need no bent wrist
        half = np.array(O.SPONGE["size"]) / 2
        zt = R[:, 2]
        edges = []
        for ax in (0, 1):
            for sgn in (-1.0, 1.0):
                nrm = R[:, ax] * sgn
                edges.append((float(nrm[:2] @ (-to)), ax, sgn))
        edges.sort(reverse=True)
        for _, ax, sgn in edges[:2]:
            g = p + R[:, ax] * sgn * (half[ax] - 0.020)
            inward = -R[:, ax] * sgn
            inward = np.array([inward[0], inward[1]])
            inward /= np.linalg.norm(inward)
            for tilt in (12.0, 20.0):
                a = tilted(inward, tilt)
                pv = zt - (zt @ a) * a
                pv /= np.linalg.norm(pv)
                sv = np.cross(a, pv)
                for lean in (28.0, -28.0, 22.0, -22.0):
                    pn = math.cos(math.radians(lean)) * pv + math.sin(math.radians(lean)) * sv
                    if abs(float(pn @ zt)) < 0.83:
                        continue
                    for s2 in (1.0, -1.0):
                        out.append((g, a, hand_frame(a, s2 * pn)))
    if kind == "sponge":
        # (round 5) across its width with the hand nearly level behind one end of its length, the fingers along its
        # sides (the pinch across its width, feature k = 1): wiping, rinsing and carrying it so need no bent wrist;
        # from either end (the one nearer the robot first)
        half = np.array(O.SPONGE["size"]) / 2
        ends = sorted((1.0, -1.0), key=lambda sx: float(R[:2, 0] @ to) * sx, reverse=True)
        for sx in ends:
            ax = R[:, 0] * sx
            ax2 = np.array([ax[0], ax[1]])
            if np.linalg.norm(ax2) < 1e-6:
                continue
            ax2 /= np.linalg.norm(ax2)
            g = p - ax * 0.012 + R[:, 2] * (half[2] - 0.006)
            for tilt in (18.0, 28.0, 38.0):
                a = tilted(ax2, tilt)
                wy = R[:, 1]
                pin = wy - (wy @ a) * a
                pin /= np.linalg.norm(pin)
                for s2 in (1.0, -1.0):
                    out.append((g, a, hand_frame(a, s2 * pin)))
    if kind == "sponge":
        # from above, pinched across its width (the pad stays free underneath): approach straight down or tilted
        # towards the robot
        half = np.array(O.SPONGE["size"]) / 2
        g = p + R[:, 2] * (half[2] - 0.012)
        wy = R[:, 1]
        for tilt in (62.0, 75.0, 90.0):
            a = tilted(to, tilt)
            pin = wy - (wy @ a) * a
            pin /= np.linalg.norm(pin)
            for s2 in (1.0, -1.0):
                out.append((g, a, hand_frame(a, s2 * pin)))
    if kind in ("book", "sponge"):
        if kind == "book":
            w, t, h = O.LOOSE[name][1]["size"]
            half, k = np.array([w / 2, t / 2, h / 2]), 1
        else:
            half, k = np.array(O.SPONGE["size"]) / 2, 2
        edges = []
        for ax in [i for i in range(3) if i != k]:
            for sgn in (-1.0, 1.0):
                nrm = R[:, ax] * sgn
                edges.append((float(nrm[:2] @ (-to)), ax, sgn))
        edges.sort(reverse=True)
        if kind == "book":
            edges = [(1.0, 0, -1.0)] + [e for e in edges if (e[1], e[2]) != (0, -1.0)]
        for _, ax, sgn in edges[:(4 if kind == "book" else 2)]:
            g = p + R[:, ax] * sgn * (half[ax] - 0.022)
            inward = -R[:, ax] * sgn
            inward[2] = 0.0
            inward /= np.linalg.norm(inward)
            for tilt in (30.0, 25.0, 33.0):
                a = tilted(inward[:2], tilt)
                thin = R[:, k]
                pin = thin - (thin @ a) * a
                for s2 in (1.0, -1.0):
                    out.append((g, a, hand_frame(a, s2 * pin)))
    return out


def carry_costs(ref, name, side, base0, frames):
    """Per grasp frame, the arm posture it leaves for carrying (round 5: canonical_carry's points in front of the waist
    and turns, from a robot standing upright at base0, the item level as it then is, at the height it will be carried
    at: dishes, books and the sponge to the counter or the shelf just clear of the counter, litter at the waist or
    beside the hip; the posture the executed path resolves, ranked by the human measure; 50 when none is reachable).
    Litter carried at any angle costs 0."""
    kind = O.LOOSE[name][0]
    if kind in FREE_CARRY:
        return [0.0] * len(frames)
    st = ref.stance
    po, Ro = ref.pose(name)
    if name in O.BOOKS:
        yb = np.array([0.0, 0.0, 1.0 if Ro[2, 1] > 0 else -1.0])
        xb = Ro[:, 0] - (Ro[:, 0] @ yb) * yb
        xb /= np.linalg.norm(xb)
        Rl = np.stack([xb, yb, np.cross(xb, yb)], axis=1)
    else:
        Rl = level(Ro)
    # the object's lowest point now (it rests on its surface): a grasp's depth below the hand
    m, d = ref.m, ref.d
    bo = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    lows = [geom_lowest(m, d, g_) for g_ in range(m.ngeom)
            if m.geom_bodyid[g_] == bo and (m.geom_contype[g_] or m.geom_conaffinity[g_])]
    z_low_obj = min(lows) if lows else float(po[2])
    high = kind == "sponge" or O.CATEGORY[kind] == "dish"
    x, y, yaw = base0
    sgn = 1.0 if side == "left" else -1.0
    yaws = (0.0, -0.6, 0.6, -1.3, 1.3) if kind in SYMMETRIC else (0.0,)
    seeds = [np.array(NATURAL_CARRY_Q[side], float)]
    costs = []
    for g, a, Rh in frames:
        depth = max(0.0, float(g[2]) - z_low_obj)
        zc = max(0.96, 0.935 + depth) if high else max(0.84, 0.79 + depth)
        pts = [(f, 0.15, zc + 0.015) for f in ((0.36, 0.42) if high else (0.32, 0.38))]
        if hip_carry(name):
            pts += [(0.12, 0.22, max(0.66, 0.50 + depth)), (0.20, 0.20, max(0.72, 0.56 + depth))]
        Rc = Rl @ (Ro.T @ Rh)
        cands = []
        for f, l, zz in pts:
            t = np.array([x, y, 0.0]) + _Rz(yaw) @ np.array([f, sgn * l, 0.0])
            t[2] = zz
            for dyaw in yaws:
                Rt = _Rz(dyaw) @ Rc
                for sq in seeds:
                    st.set_state((x, y, yaw), 0.79, 0.0)
                    q, ep, er = st.solve(side, t, Rt, sq, iters=80, quick=30)
                    if ep < 0.01 and er < 0.08:
                        cands.append((posture_cost(ref, side, q) + 0.1 * abs(dyaw), t, Rt, q, dyaw))
        best = 50.0
        for c0, t, Rt, q, dyaw in sorted(cands, key=lambda c_: c_[0])[:3]:
            st.set_state((x, y, yaw), 0.79, 0.0)
            qn = natural_resolve(ref, side, t, Rt, q)
            best = min(best, c0, posture_cost(ref, side, qn) + 0.1 * abs(dyaw))
        if O.CATEGORY[kind] == "dish":
            # (round 5) and the put it leaves at the dish tray (its nominal stance, a few leans and turns of the
            # dish): a grip that cannot set the dish down there is no grip for it
            best += 0.5 * tray_reach_cost(ref, name, side, Ro, Ro.T @ (np.asarray(g) - po), Ro.T @ Rh)
        costs.append(best)
    return costs


def tray_reach_cost(ref, name, side, Ro, pg, Rg):
    """The posture cost (posture_cost; 100 out of reach) of setting `name` held by the grip (pg, Rg) down on its dish
    tray spot from the tray's nominal stance: the best of a few leans and turns of the item about the vertical."""
    st = ref.stance
    spot = tray_spot(ref, name)
    tgt = spot + np.array([0.0, 0.0, 0.03])
    lat = 0.12 if side == "left" else -0.12
    base = (RM.KITCHEN["top_x1"] + 0.16, float(tgt[1]) + lat, math.pi)
    Rl = level(Ro)
    best = 100.0
    support = ("dish_tray",) + tuple(f"{k}_col" for k in O.DISHES if k != name and O.LOOSE[k][0] == "plate")
    arm_q = ref.c.robot.arm_q[side]
    for dyaw in (0.0, 0.8, -0.8, 1.6, -1.6, math.pi):
        Robj = _Rz(dyaw) @ Rl
        ph, Rh_ = tgt + Robj @ pg, Robj @ Rg
        for pitch in (0.15, 0.30, 0.45):
            st.set_state(base, 0.79, pitch)
            for sq in (np.array(NATURAL_CARRY_Q[side], float), np.array(REACH_SEEDS[side][2], float)):
                q, ep, er = st.solve(side, ph, Rh_, sq, iters=80, quick=30)
                if ep < 0.005 and er < 0.04:
                    q = natural_resolve(ref, side, ph, Rh_, q)
                    # (round 5) the arm and the dish clear of the counter and the tray's rim where it lets go (8 mm
                    # over its spot: the wrist lowest there): a plate pinched with the hand nearly level put the wrist
                    # into the counter's top, and the put then went in turned 1 rad with the arm stretched out
                    ql, ep2, er2 = st.solve(side, spot + np.array([0.0, 0.0, 0.008]) + Robj @ pg, Rh_, q, iters=40)
                    if ep2 > 0.005 or er2 > 0.04:
                        continue
                    st.chk.qpos[arm_q] = ql
                    st.place_held(side, name, (pg, Rg))
                    if st.collisions(held=[name], allow=support):
                        continue
                    best = min(best, posture_cost(ref, side, q))
        if best < 0.5:
            break
    return best


def carry_order(ref, name, side, base0, frames):
    """The grasp frames ordered by carry_costs (stable: equal costs keep the frames' own order)."""
    if O.LOOSE[name][0] in FREE_CARRY or len(frames) < 2:
        return frames
    costs = carry_costs(ref, name, side, base0, frames)
    order = sorted(range(len(frames)), key=lambda k: (round(costs[k], 1), k))
    return [frames[k] for k in order]


def posture_cost(ref, side, q):
    """How unnatural an arm posture is (0: fine), in the stance model's state as set: NaturalIK's ranking measure
    (terms_rank: joints near a limit, the elbow raised towards or above the shoulder, swung behind the body or out to
    the side, the twisting joints far from their middle), which the planner's choices rank by."""
    st = ref.stance
    return st.nik.cost(st.chk.qpos, side, np.asarray(q, float))


def sponge_work_costs(ref, side, frames):
    """Per grasp frame of the sponge, the arm posture its grip leaves for the work that follows (posture_cost, the
    human measure, of the posture the executed path resolves; 100 for a pose out of reach): a wiping stroke's near
    and far ends on the dining table from its south side (as wipe_side stands, the sponge level, the best of its turns
    about the vertical and two waist pitches), the rinse under the faucet, the press on the soap pump and the carry to
    the counter (round 5: each from a few stances, as the stance searches rank them; the rinse, the soap and the
    carry weighted 0.6, 0.3 and 0.3). A grasp is so chosen for the work as well as for the take."""
    st = ref.stance
    p, Ro = ref.pose("sponge")
    D = RM.DINING
    (dx, dy), (lx, ly) = D["center"], D["size"]
    zc = D["top_z"] + O.SPONGE["size"][2] / 2 - 0.001
    lat = 0.12 if side == "left" else -0.12
    sg = 1.0 if side == "left" else -1.0
    Rl = level(Ro)
    ox, oy, oz = RM.FAUCET["outlet"]
    kx = RM.KITCHEN["top_x1"]
    wipe_bases = [(dx + lat, dy - ly / 2 - 0.16, math.pi / 2)]           # (as wipe_side stands)
    wipe_pts = (np.array([dx, dy - ly / 2 + 0.10, zc]), np.array([dx, dy + 0.01, zc]))
    rinse_bases = [(kx + 0.17, oy + lat, math.pi)]
    rinse_pt = np.array([ox, oy, oz - RINSE_DROP])
    sx, sy = RM.SOAP["base"]
    soap_bases = [(kx + 0.17 + db, sy + lat, math.pi) for db in (0.0, 0.04)]
    soap_pt = np.array([sx, sy, RM.SOAP["neck_top"] + 0.010 + RM.SOAP["head_h"] + O.SPONGE["size"][2] / 2 + 0.03])
    carry_base = (2.0, 2.0, 0.0)
    carry_pts = [np.array([2.0 + f, 2.0 + sg * 0.15, z]) for f, z in ((0.36, 0.86), (0.40, 0.95))]
    yaws = (0.0, 0.5, -0.5, 1.0, -1.0, 1.57, -1.57, math.pi)
    seeds = [np.array(NATURAL_CARRY_Q[side], float), np.array(REACH_SEEDS[side][2], float)]

    def pose_cost(base, pitch, pobj, Robj, pg, Rg):
        st.set_state(base, 0.79, pitch)
        best = 100.0
        for sq in seeds:
            ph, Rh_ = pobj + Robj @ pg, Robj @ Rg
            q, ep, er = st.solve(side, ph, Rh_, sq, iters=100, quick=30)
            if ep < 0.005 and er < 0.04:
                best = min(best, posture_cost(ref, side, natural_resolve(ref, side, ph, Rh_, q)))
        return best

    def best_of(bases, pitches, pts, pg, Rg):
        c = 200.0
        for dyaw in yaws:
            Robj = _Rz(dyaw) @ Rl
            for base in bases:
                for pitch in pitches:
                    c = min(c, max(pose_cost(base, pitch, pt, Robj, pg, Rg) for pt in pts))
        return c

    out = []
    for g, a, Rh in frames:
        pg, Rg = Ro.T @ (np.asarray(g) - p), Ro.T @ Rh
        # (the wiping strokes' stances and leans as wipe_side plans them: a grip judged from another stance may not
        # reach the strokes from the ones the wiping uses; the stain's middle and both its ends along the table: a
        # grip that cannot reach every stroke cannot clean the table)
        wipe = best_of(wipe_bases, (0.45, 0.52), wipe_pts, pg, Rg)
        rinse = best_of(rinse_bases, (0.15, 0.30), (rinse_pt,), pg, Rg)
        soap = best_of(soap_bases, (0.0, 0.15), (soap_pt,), pg, Rg)
        carry = min(best_of([carry_base], (0.0,), (pt,), pg, Rg) for pt in carry_pts)
        out.append(wipe + 0.5 * rinse + 0.2 * min(soap, 20.0) + 0.3 * min(carry, 20.0))
    return out


def wipe_strokes_ok(ref, side, frame):
    """Can the sponge, held by `frame`'s grip, follow the first round's wiping strokes as degrease and wipe_side will
    plan them (the stain's ends and middle, from both long sides of the table, its near and far ends from the grease
    cells), from wipe_side's stances and leans (the path's waypoints solved one from the next, as the stance checks
    solve them; the best of a few turns of the sponge about the vertical)? The arm only, no collisions."""
    st = ref.stance
    g, a, Rh = frame
    p, Ro = ref.pose("sponge")
    pg, Rg = Ro.T @ (np.asarray(g) - p), Ro.T @ Rh
    D = RM.DINING
    (dx, dy), (lx, ly) = D["center"], D["size"]
    top = D["top_z"]
    hz = O.SPONGE["size"][2] / 2
    lat = 0.12 if side == "left" else -0.12
    zc = top + hz - 0.001
    Rl = level(Ro)
    cells = grease_cells(ref)
    if not len(cells):
        return True
    for from_south in (True, False):
        sel = cells[cells[:, 1] <= dy + 0.03] if from_south else cells[cells[:, 1] > dy - 0.03]
        if not len(sel):
            continue
        x0, x1 = float(sel[:, 0].min()), float(sel[:, 0].max())
        if from_south:
            y_near, y_far = float(sel[:, 1].min()) - 0.03, min(dy + 0.01, float(sel[:, 1].max()) + 0.03)
            yaw_r, y_base = math.pi / 2, dy - ly / 2 - 0.16
        else:
            y_near, y_far = float(sel[:, 1].max()) + 0.03, max(dy - 0.01, float(sel[:, 1].min()) - 0.03)
            yaw_r, y_base = -math.pi / 2, dy + ly / 2 + 0.16
        for x in sorted({x0, (x0 + x1) / 2, x1}):
            path = [np.array([x, y_near, top + hz + 0.04]), np.array([x, y_near, zc + 0.003])]
            n = max(2, int(abs(y_far - y_near) / 0.03))
            path += [np.array([x, y_near + (y_far - y_near) * i / n, zc]) for i in range(1, n + 1)]
            bx = float(np.clip(x + (lat if from_south else -lat), dx - lx / 2 + 0.02, dx + lx / 2 - 0.02))
            ok_x = False
            for dyaw in (0.0, 0.5, -0.5, 1.0, -1.0, math.pi):
                Robj = _Rz(dyaw) @ Rl
                hp = [(pt + Robj @ pg, Robj @ Rg) for pt in path]
                for pitch in (0.45, 0.52, 0.35):
                    for fwd, lt in ((0.0, 0.0), (0.03, 0.0), (0.05, 0.0), (-0.03, 0.0)):
                        c_, s_ = math.cos(yaw_r), math.sin(yaw_r)
                        base = (bx + fwd * c_ - lt * s_, y_base + fwd * s_ + lt * c_, yaw_r)
                        st.set_state(base, 0.79, pitch)
                        q, ep, er = st.solve(side, hp[0][0], hp[0][1], np.array(NATURAL_CARRY_Q[side], float),
                                             iters=100, quick=30)
                        if ep > 0.005 or er > 0.04:
                            continue
                        good = True
                        for ph, Rh_ in hp[1:]:
                            q, ep, er = st.solve(side, ph, Rh_, q, 60, null_total=st.PATH_NULL)
                            if ep > 0.005 or er > 0.04:
                                good = False
                                break
                        if good:
                            ok_x = True
                            break
                    if ok_x:
                        break
                if ok_x:
                    break
            if not ok_x:
                return False
    return True


def ranked_stances(ref, name, side, base0, frames, pitches=(0.30, 0.45, 0.20, 0.60), heights=(0.79, 0.70),
                   work=None, yaws=(0.0,)):
    """(cost, stance (x, y, yaw, h, pitch), frame, arm joints) for taking `name` by the grasp frames from stances
    beside base0 (the robot up to 24 cm to either side, so the arm can reach straight or across the body), feasible
    ones only, the most natural arm posture (posture_cost, plus the frame's `work` cost if given) first."""
    st, r = ref.stance, ref.c.robot
    work = work or [0.0] * len(frames)
    out = []
    for lat, dyaw in [(l_, y_) for y_ in yaws for l_ in (0.0, 0.08, -0.08, 0.16, -0.16)]:
        yaw = base0[2] + dyaw
        c_, s_ = math.cos(yaw), math.sin(yaw)
        for pitch in pitches:
            for h in heights:
                base = (base0[0] - lat * s_, base0[1] + lat * c_, yaw)
                st.set_state(base, h, pitch)
                for kf, (g, a, Rh) in enumerate(frames):
                    best = None
                    for sq in [np.array(ARM_HOME[side], float), np.array(NATURAL_CARRY_Q[side], float),
                               np.array(REACH_SEEDS[side][2], float)]:
                        q, ep, er = st.solve(side, g, Rh, sq, iters=120, quick=40, w_home=0.02)
                        if ep > 0.003 or er > 0.03:
                            continue
                        # (the posture the executed path resolves there)
                        q = natural_resolve(ref, side, g, Rh, q)
                        cost = posture_cost(ref, side, q)
                        if best is None or cost < best[0]:
                            best = (cost, q)
                    if best is None:
                        continue
                    st.chk.qpos[r.arm_q[side]] = best[1]
                    if st.collisions():
                        continue
                    # (round 5: and the legs: the shallower squat of postures the arm takes alike)
                    out.append((best[0] + work[kf] + 1.0 * (max(0.0, 0.79 - h) / 0.1) ** 2,
                                (base[0], base[1], base[2], h, pitch), (g, a, Rh), best[1]))
    out.sort(key=lambda t: t[0])
    return out


def _take_natural(ref, name, side, nom, lift, tries=6, top=6):
    """take() from the stance where the arm reaches the grasp in the most natural posture (ranked_stances) for the
    grasp that leaves the most natural carry (carry_costs; the sponge: the work that follows, sponge_work_costs),
    the `top` grasps by that: the stance search from each ranked stance in turn, the approach led by that posture (a
    joint-space move to it first, so the arm keeps its branch; the path from the present arm configuration may
    otherwise end with the elbow over the head). None when no ranked stance works (take() goes on as before)."""
    base0, heights, pitches, low = nom
    frames = _grasp_frames(ref, name, side, base0)
    if name == "sponge" and ref.surface_of(ref.pose(name)[0]) == "counter":
        work = sponge_work_costs(ref, side, frames)
        # (a grip that reached no stroke, degrease's regrasp: ruled out, and its turn by half a turn about the
        # sponge's vertical, the same grip of the symmetric sponge)
        bad = getattr(ref, "sponge_bad_grips", [])
        if bad:
            ps_, Rs_ = ref.pose(name)
            Rpi = _Rz(math.pi)
            for k, (g, a, Rh) in enumerate(frames):
                pg_, Rg_ = Rs_.T @ (np.asarray(g, float) - ps_), Rs_.T @ Rh
                for pgb, Rgb in bad:
                    for P_, R_ in ((pgb, Rgb), (Rpi @ pgb, Rpi @ Rgb)):
                        dR = R_ @ Rg_.T
                        ang = math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)))
                        if float(np.linalg.norm(P_ - pg_)) < 0.03 and ang < 0.35:
                            work[k] += 200.0
        # (the best few grips' wiping strokes checked as wipe_side will plan them, waypoint by waypoint: a grip whose
        # strokes the arm cannot follow cannot clean the table, however natural its postures at their ends)
        order = sorted(range(len(frames)), key=lambda k: (round(work[k], 2), k))
        n_ok = 0
        for k in order:
            if n_ok >= 3:
                break
            if wipe_strokes_ok(ref, side, frames[k]):
                n_ok += 1
            else:
                work[k] += 100.0
    else:
        work = carry_costs(ref, name, side, base0, frames)
    keep = sorted(range(len(frames)), key=lambda k: (round(work[k], 2), k))[:top]
    frames, work = [frames[k] for k in keep], [work[k] for k in keep]
    # (the sponge lies along its tray, across the robot's reach: the robot may stand turned a little towards it)
    yaws = (0.0, -0.35, 0.35) if name == "sponge" else (0.0,)
    ranked = ranked_stances(ref, name, side, base0, frames, pitches=pitches, heights=heights, work=work, yaws=yaws)
    for cost, (x, y, yaw, h, pitch), (g, a, Rh), q_nat in ranked[:tries]:
        pre = g - a * 0.09 + np.array([0.0, 0.0, 0.03])
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        poses = [(pre + (g - pre) * k / 3.0, Rh) for k in range(0, 4)]
        poses += [(g + np.array([0.0, 0.0, lift * k / 3.0]), Rh) for k in range(1, 4)]
        up = g + np.array([0.0, 0.0, lift])
        poses += [(up - fwd * 0.05 * k + np.array([0.0, 0.0, 0.02 * k]), Rh) for k in range(1, 3)]
        ref.stance.prefer_seeds = [q_nat]          # (also for _take_go's re-plan where the robot then stands)
        try:
            offs = [(0.0, 0.0), (-0.03, 0.0), (0.03, 0.0)]
            if low:
                # (at a low table: squatting where it stands the robot's hips may meet the table top: a little back)
                offs += [(-0.06, 0.0), (-0.09, 0.0), (-0.12, 0.0)]
            stance, sol = stance_search(ref, side, poses, (x, y, yaw), (h,), (pitch,), offsets=offs, passes=(True,))
            if stance is not None:
                ref.log("[ref] take %s: natural posture (cost %.3f), stance (%.2f, %.2f, %.2f) h %.2f pitch %.2f"
                        % (name, cost, stance[0], stance[1], stance[2], stance[3], stance[4]))
                return _take_go(ref, name, side, sol, poses, g, stance, low)
        finally:
            ref.stance.prefer_seeds = None
        ref.take_why = sol
    ref.log("[ref] take %s: no natural posture (%d ranked stances; last: %s)" % (name, len(ranked),
                                                                               getattr(ref, "take_why", None)))
    return None


def take(ref, name, side, lift=0.12):
    """Pick `name` up with `side`'s hand from where it lies, by a planned stance. Returns True when held."""
    p, _ = ref.pose(name)
    surf = ref.surface_of(p)
    nom = _nominal(ref, surf, p, side)
    if nom is None:
        ref.log("[ref] take %s: it lies on the %s, out of reach" % (name, surf))
        return False
    ref.dest_hint = (float(nom[0][0]), float(nom[0][1]))
    ref.stow()                                   # plan from the configuration the walk will leave the arm in
    if ref.held[side]:
        ref.log("[ref] take %s: the %s hand still holds %s" % (name, side, ref.held[side]))
        return False
    qs, why_last = None, None
    if name == "sponge" and surf == "counter" and not getattr(ref, "round4_sponge_grips", False):
        # the sponge from its tray: from the stance of the most natural arm posture for the grasp and for the work
        # that follows it (the wiping and the rinsing); the nominal stance in front of the tray leaves the arm
        # reaching over the head
        got = _take_natural(ref, name, side, nom, lift, top=12)
        if got is not None:
            return got
    elif O.LOOSE[name][0] != "book":
        # (round 5) every other item likewise: the grasp that leaves the most natural carry, from the stance of the
        # most natural reach for it
        got = _take_natural(ref, name, side, nom, lift)
        if got is not None:
            return got
    # from the nearer long side of the dining table first, then (a glass or a can far in) from the other one
    noms = [nom]
    if surf == "dining_top" and O.LOOSE[name][0] != "book":
        noms.append(_nominal(ref, surf, p, side, flip=True))
    for base0, heights, pitches, low in noms:
        qs = _take_plan(ref, name, side, base0, heights, pitches, lift)
        if qs is not None:
            qs, poses_used, g, stance = qs
            break
        why_last = ref.take_why
    if qs is None:
        ref.log("[ref] take %s: no stance for any grasp (last: %s)" % (name, why_last))
        return False
    return _take_go(ref, name, side, qs, poses_used, g, stance, low)


def _take_plan(ref, name, side, base0, heights, pitches, lift):
    """(joints, hand poses, grasp point, stance) of the first grasp of `name` reachable from near base0, or None
    (the last failure in ref.take_why)."""
    qs, why_last = None, None
    frames = _grasp_frames(ref, name, side, base0)
    if name == "sponge" and getattr(ref, "round4_sponge_grips", False):
        # (round 4's families: by the edge facing the robot, tilted 25-33 deg, then from above, as round 4 took it)
        frames = frames[-12:] + frames[-18:-12]
    else:
        frames = carry_order(ref, name, side, base0, frames)
    for passes in ((False,), (True,)):               # every grasp by the straight approach first
        for g, a, Rh in frames:
            pre = g - a * 0.09 + np.array([0.0, 0.0, 0.03])
            fwd = np.array([math.cos(base0[2]), math.sin(base0[2]), 0.0])
            poses = [(pre + (g - pre) * k / 3.0, Rh) for k in range(0, 4)]          # pre .. grasp (index 3)
            poses += [(g + np.array([0.0, 0.0, lift * k / 3.0]), Rh) for k in range(1, 4)]
            up = g + np.array([0.0, 0.0, lift])
            poses += [(up - fwd * 0.05 * k + np.array([0.0, 0.0, 0.02 * k]), Rh) for k in range(1, 3)]
            yo = (0.0, 0.4, -0.4, 0.8, -0.8) if O.LOOSE[name][0] == "book" else (0.0,)
            stance, sol = stance_search(ref, side, poses, base0, heights, pitches, yaw_offsets=yo, passes=passes)
            why_last = sol
            if stance is not None:
                qs = sol
                poses_used = poses
                break
        if qs is not None:
            break
    ref.take_why = why_last
    if qs is None:
        return None
    return qs, poses_used, g, stance


def _take_go(ref, name, side, qs, poses_used, g, stance, low):
    """Walk to the planned stance and take `name` (take())."""
    x, y, yaw, h, pitch = stance
    ref.hand(side, False)
    if low:
        c_, s_ = math.cos(yaw), math.sin(yaw)
        if ref.walk(x - 0.15 * c_, y - 0.15 * s_, yaw) is False:
            ref.log("[ref] take %s: the walk there failed" % name)
            return False
        ref.posture(h, pitch)
        ref.goto(x, y, yaw, tol=0.006)
        ref.low_spot = (x - 0.15 * c_, y - 0.15 * s_, yaw)
    else:
        if ref.walk(x, y, yaw) is False:
            ref.log("[ref] take %s: the walk there failed" % name)
            return False
        ref.low_spot = None
        ref.posture(h, pitch)
    if not ref.arrived(x, y, yaw):
        ref.log("[ref] take %s: the stance was not reached" % name)
        ref.leave_low()
        return False
    # the path again, from the arm and the base as they now are
    b = ref.base()
    ok, worst, qs2, why = ref.stance.check((b[0], b[1], b[2]), h, pitch, side, poses_used)
    if ok:
        qs = qs2
        ref.stance.n_via_saved = ref.stance.n_via
    nv = ref.stance.n_via_saved
    ref.run_path(side, qs[:nv + 4])
    ref.settle(side)
    err = float(np.linalg.norm(ref.site(side)[0] - g))
    got = ref.hand(side, True)
    if got != name:
        ref.log("[ref] take %s: closed on %s (hand %.3f m off)" % (name, got, err))
        if got is not None:
            ref.hand(side, False)
        ref.run_q(side, last(qs[0]), timeout=2.0)
        ref.leave_low()
        return False
    ref.run_path(side, qs[nv + 4:nv + 7], tol_via=0.012)      # off the surface: close to the vertical line
    # (round 5) not then drawn back 10 cm towards the body with the hand's rotation kept (qs[nv + 7:], still planned:
    # the stance search keeps room for it): drawn in so while the robot still bent over the table, the elbow folded to
    # its limit and the shoulder swung back, and the arm stayed so while the robot stepped back and straightened up;
    # the carry pose follows from the lift's posture
    ref.leave_low()
    return True


SYMMETRIC = ("can", "bottle", "glass", "plate", "napkin", "apple", "book", "sponge")   # (any way round is fine)
FREE_CARRY = ("napkin", "apple")      # carried at any angle (litter: the hand holds it as it comes)
# canonical_carry's carry points (forward, out to the hand's side, height; height 0: the carry height of the item):
# in front of the chest (in front of the shoulder, a little nearer the middle, a little farther out), and for litter
# on its way to a bin (hip_carry) also beside the hip with the arm hanging, its height clear of a low table by the
# item's depth (a can held from above dangles from the hand there; carried at the chest it puts the elbow up)
CARRY_POINTS = ((0.30, 0.20, 0.0), (0.27, 0.14, 0.0), (0.33, 0.25, 0.0), (0.10, 0.24, -1.0), (0.16, 0.24, -1.0))
# (round 5) the natural carry: the upper arm hanging a little forward, the elbow bent about a right angle, the forearm
# and the hand forward and level, the wrist straight (the G1's elbow then hangs 0.89 m up and the grasp site is 0.35 m
# in front of the shoulder): a seed of the carry solves
NATURAL_CARRY_Q = {"right": [-0.10, -0.05, 0.0, 0.05, 0.0, 0.0, 0.0], "left": [-0.10, 0.05, 0.0, 0.05, 0.0, 0.0, 0.0]}


def carry_points(low, compact=False):
    """canonical_carry's points (forward, out to the hand's side, height above the carry height; a negative height:
    beside the hip, at the hip height (-1) or 6 cm above it (-2)): in front of the waist, the forearm forward; `low`:
    the carry height is the item's own (in the open), else the counter's (a little farther out, the forearm then
    rising a little); `compact`: drawn in close to the body (a walk past a fixture)."""
    if compact:
        return tuple((f, l, dz) for dz in (0.0, 0.05) for f in (0.22, 0.26) for l in (0.14, 0.20))
    if low:
        pts = [(f, l, dz) for dz in (0.0, 0.05) for f in (0.30, 0.35, 0.40) for l in (0.12, 0.18)]
    else:
        pts = [(f, l, dz) for dz in (0.0, 0.03) for f in (0.32, 0.37, 0.42) for l in (0.12, 0.18)]
    return tuple(pts) + ((0.10, 0.22, -1.0), (0.17, 0.22, -1.0), (0.22, 0.20, -2.0))


def natural_resolve(ref, side, p, R, q, iters=60):
    """The arm configuration the executed path resolves at the hand pose (p, R) (NaturalIK's natural redundancy
    resolution from q, as Reference.refine_path solves it), or q if that does not converge: the posture a choice of
    the planner is ranked by."""
    st = ref.stance
    q2, ep, er = st.nik.solve(st.chk.qpos, side, np.asarray(p, float), None if R is None else _quat(R), q,
                              iters=iters, natural=True)
    return q2 if ep < 0.003 and (R is None or er < 0.02) else np.asarray(q, float)


def hip_carry(name):
    """May `name` be carried beside the hip? Litter for the bins (the bins' rims are low); dishes, books and the
    sponge go onto the counter, the shelf and the table and are carried in front of the chest."""
    return name in O.LOOSE and O.CATEGORY[O.LOOSE[name][0]] in ("recycling", "general")


def free_yaw(ref, name, R, yaw_stance, yaw_ref=None):
    """An object symmetric about its vertical axis may arrive turned: its target rotation (captured while the robot
    faced yaw_ref, default: now) turned by the robot's change of heading, so the hand keeps its pose relative to the
    robot."""
    if O.LOOSE[name][0] not in SYMMETRIC:
        return R
    yaw_ref = ref.base()[2] if yaw_ref is None else yaw_ref
    return _Rz(wrap(yaw_stance - yaw_ref)) @ R


def _turned(ref, name, R, dy, yaw_stance, absolute):
    """A target rotation turned by dy about the vertical; unless `absolute`, also by the robot's change of heading
    (free_yaw) for a symmetric object."""
    return _Rz(dy) @ (R if absolute else free_yaw(ref, name, R, yaw_stance, ref.yaw_captured))


def plan_put(ref, side, name, targets, base0, heights, pitches, yaw_offsets=(0.0,), allow=(), offsets=None,
             absolute=False, back=0.0, rank=4, hold=True):
    """(stance, joints, hand poses) for carrying the held object through `targets`; a symmetric object may arrive
    turned about its vertical axis (several turns are tried; with `absolute`, only small turns of the rotations as
    given)."""
    hold = hold and name not in ("broom", "dustpan")   # (a long tool hangs from the hand: it moves with the body)
    if absolute:
        dyaws = (0.0, 0.3, -0.3, 0.6, -0.6)
    else:
        dyaws = (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.4, -2.4, math.pi) if O.LOOSE[name][0] in SYMMETRIC else (0.0,)
    why = None
    for passes in ((False,), (True,)):               # every turn by the straight approach first
        found = []
        for dy in dyaws:
            tg = [(p, _turned(ref, name, R, dy, base0[2], absolute)) for p, R in targets]
            poses = [ref.hand_for(side, p, R) for p, R in tg]
            ref.last_rank_cost = 0.0
            stance, qs = stance_search(ref, side, poses, base0, heights, pitches, holding=name,
                                       yaw_offsets=yaw_offsets, allow=allow, offsets=offsets, passes=passes, back=back,
                                       rank=rank, hold=hold)
            if stance is not None:
                # (round 5) of the first few turns that work, the one the arm takes most naturally
                found.append((ref.last_rank_cost + 0.05 * abs(dy), dy, stance, qs, poses, ref.stance.n_via_saved))
                if rank <= 1 or len(found) >= 3 or found[-1][0] < 0.6:
                    break
            else:
                why = qs
        if found:
            c, dy, stance, qs, poses, nv = min(found, key=lambda f_: f_[0])
            ref.dy_chosen = dy
            ref.stance.n_via_saved = nv
            if rank > 1:
                ref.log("[ref] plan %s: posture %.2f (turned %.1f, pitch %.2f h %.2f)" % (name, c, dy, stance[4],
                                                                                        stance[3]))
            return stance, qs, poses
    return None, why, None


def go_put(ref, side, name, targets, base0, heights, pitches, what, yaw_offsets=(0.0,), absolute=False, low=False,
           allow=(), back=0.15, offsets=None, rank=4):
    """Plan a stance for carrying the held object through `targets` (from the carrying configuration, taken first),
    walk there (`low`: take the posture `back` metres short of it, then step in), take the posture, then plan the
    arm's path again from the arm as it now is. Returns (joint path, hand poses) or None."""
    ref.yaw_captured = ref.base()[2]          # the targets' rotations were taken now (the caller's pose())
    ref.dest_hint = (float(base0[0]), float(base0[1]))
    ref.stow()
    if ref.held[side] != name:
        ref.log("[ref] %s %s: not in the %s hand" % (what, name, side))
        return None
    q_plan = None
    if low and ref.dest_counter() and not ref.dest_high() and (name == "sponge" or (
            name in O.LOOSE and O.CATEGORY[O.LOOSE[name][0]] == "dish")):
        # (round 5) carried at the waist from afar and raised for the counter only on the way (walk): the put
        # planned from the raised posture
        z_hi = max(0.96, 0.935 + ref.held_depth(side)) + ref.carry_lift.get(name, 0.0)
        q_plan = ref.canonical_carry(side, z_hi, low=False, dry=True)
    ref.stance.seed_override = (side, np.asarray(q_plan, float)) if q_plan is not None else None
    try:
        stance, qs, poses = plan_put(ref, side, name, targets, base0, heights, pitches, yaw_offsets,
                                     absolute=absolute, allow=allow, back=back if low else 0.0, offsets=offsets,
                                     rank=rank)
    finally:
        ref.stance.seed_override = None
    for dz in (0.08, 0.16):
        # carried in front of the chest, the item would sweep down into what the robot leans over (a plate's far
        # rim into the counter): lifted higher first, as a person lifts a plate over the counter
        if stance is not None or not (isinstance(qs, tuple) and len(qs) > 1 and qs[0] == "start collides"
                                      and qs[1] == name):
            break
        ref.carry_lift[name] = dz
        ref.carry_pose((side,))
        stance, qs, poses = plan_put(ref, side, name, targets, base0, heights, pitches, yaw_offsets,
                                     absolute=absolute, allow=allow, back=back if low else 0.0, offsets=offsets)
    if stance is None:
        ref.log("[ref] %s %s: no stance (%s)" % (what, name, qs))
        return None
    x, y, yaw, h, pitch = stance
    if low:
        c_, s_ = math.cos(yaw), math.sin(yaw)
        if ref.walk(x - back * c_, y - back * s_, yaw, carrying=True) is False:
            ref.log("[ref] %s %s: the walk there failed" % (what, name))
            return None
        ref.posture(h, pitch, hold=(side,) if name not in ("broom", "dustpan") else ())
        ref.low_spot = (x - back * c_, y - back * s_, yaw)
        # bent where it stands: the hand to the first target's place relative to the body, then step in so
        b = ref.base()
        ref.sync_held()
        if ref.held[side] != name:
            ref.log("[ref] %s %s: lost on the way" % (what, name))
            return None
        p1, R1 = poses[0]
        pre = (np.asarray(p1, float) - np.array([x - b[0], y - b[1], 0.0]), R1)
        ok, worst, qs1, why = ref.stance.check((b[0], b[1], b[2]), h, pitch, side, [pre], holding=name,
                                               grip_rel=ref.grip_rel.get(side))
        if not ok:
            ref.log("[ref] %s %s: no way to the pre-pose (%s)" % (what, name, why))
            return None
        ref.run_path(side, qs1)
        ref.goto(x, y, yaw, tol=0.006)
    else:
        if ref.walk(x, y, yaw, carrying=True) is False:
            ref.log("[ref] %s %s: the walk there failed" % (what, name))
            return None
        ref.low_spot = None
        ref.posture(h, pitch, hold=(side,) if name not in ("broom", "dustpan") else ())
    b = ref.base()
    ref.sync_held()
    if ref.held[side] != name:
        ref.log("[ref] %s %s: lost on the way" % (what, name))
        return None
    dy0 = ref.dy_chosen
    why = None
    # (a target at the edge of the arm's reach may be a few millimetres out once the robot stands there for real: then
    # 2 and 4 cm closer; the held item, carried in front, may touch what the robot stepped up to once it really stands
    # there and has bent: then it is lifted 4.5 cm first, once)
    lifted, closer = False, 0
    while True:
        for dy in ((dy0, dy0 + 0.3, dy0 - 0.3, dy0 + 0.6, dy0 - 0.6) if O.LOOSE[name][0] in SYMMETRIC else (0.0,)):
            tg = [(p, _turned(ref, name, R, dy, b[2], absolute)) for p, R in targets]
            poses = [ref.hand_for(side, p, R) for p, R in tg]
            ok, worst, qs, why = ref.stance.check((b[0], b[1], b[2]), h, pitch, side, poses, holding=name,
                                                  grip_rel=ref.grip_rel.get(side), allow=allow)
            if ok:
                return qs, poses
        if isinstance(why, tuple) and len(why) > 1 and why[0] == "start collides" and why[1] == name and not lifted:
            p_, R_ = ref.site(side)
            ref.track(side, [(p_ + np.array([0.0, 0.0, 0.015 * k]), R_) for k in range(1, 4)], ticks=6)
            ref.log("[ref] %s %s: lifted clear of what it touched at the stance" % (what, name))
            lifted = True
            continue
        if isinstance(why, tuple) and why and why[0] == "unreachable" and closer < 2:
            closer += 1
            ref.goto(x + 0.02 * closer * math.cos(yaw), y + 0.02 * closer * math.sin(yaw), yaw, tol=0.005)
            b = ref.base()
            continue
        break
    ref.log("[ref] %s %s: no path from the stance (%s)" % (what, name, why))
    return None


def put(ref, side, name, targets, base0, heights, pitches, release_settle=40, what="put", yaw_offsets=(0.0,),
        absolute=False, low=False, allow=(), back=0.15, rank=4, offsets=None):
    """Carry the object held in `side` through the object poses `targets` [(p, R)] from a planned stance, let go at
    the last one, back the hand off (`allow`: what it may touch, the surface it is set on). Returns True if the hand
    got there."""
    got = go_put(ref, side, name, targets, base0, heights, pitches, what, yaw_offsets, absolute=absolute, low=low,
                 allow=allow, back=back, rank=rank, offsets=offsets)
    if got is None:
        return False
    qs, poses = got
    # the planned joint path: from the carrying arm through Cartesian waypoints to the targets
    ref.run_path(side, qs)
    hp, hR = poses[-1]
    err = float(np.linalg.norm(ref.site(side)[0] - hp))
    if err > 0.02:
        ref.log("[ref] %s %s: the hand stopped %.3f m short" % (what, name, err))
    ref.release(side, settle=release_settle)
    if len(qs) > 1:
        # the approach retraced (in low mode all of it: back out of the compartment to in front of it)
        n_back = len(targets) if low else 4
        back_path = list(rows(qs[max(0, len(qs) - n_back):]))[::-1][1:]
        ref.run_path(side, back_path)
    if low:
        ref.retreat(side)
        ref.leave_low()
    return err < 0.02


def bin_put(ref, side, name, kind):
    """Drop the held litter into its bin. (round 5) Litter may go in at any angle: of the item as carried and the item
    tipped forward about the robot's side-to-side axis by 0.5 and 0.9 rad (the hand then reaches down into the
    opening with the wrist straight, not level out over it), the one whose drop the arm reaches most naturally."""
    slot = ref.bin_slot(kind)
    yc = RM.BIN_SLOTS_Y[slot]
    lat = 0.12 if side == "left" else -0.12
    # (let go 12 cm over the rim of the opening's near half, the robot stepping up close and bending down to it)
    tgt = np.array([0.23, yc, RM.BIN["rim_z"] + 0.12])
    base0 = (0.52, yc + lat, math.pi)
    heights, pitches = (0.79, 0.72, 0.66), (0.30, 0.45, 0.15, 0.55)
    ref.dest_hint = (float(base0[0]), float(base0[1]))
    ref.stow()                                   # (carried first: the drop's rotations from the item as carried)
    po, Ro = ref.pose(name)
    best = None
    if ref.held[side] == name and side in ref.grip_rel:
        # the item as it will be relative to the robot facing the wall at the bin (it keeps its pose relative to the
        # body), then tipped about the robot's side-to-side axis (y there); each tip judged by the drop's posture
        # alone (its IK from a few stances near the nominal one, the item's turn free), the full plan made by put()
        Ro_s = _Rz(wrap(base0[2] - ref.base()[2])) @ Ro
        ax = np.array([0.0, 1.0, 0.0])
        st = ref.stance
        pg, Rg = ref.grip_rel[side]
        for tip in (0.0, 0.5, 0.9):
            Rt = _axis_angle(ax, -tip) @ Ro_s
            c_tip = 100.0
            for dyaw in (0.0, 0.6, -0.6, 1.2, -1.2, math.pi):
                Rtt = _Rz(dyaw) @ Rt
                ph, Rh_ = tgt + Rtt @ pg, Rtt @ Rg
                for dx in (0.0, -0.05, 0.05):
                    for h_, pitch in ((0.79, 0.30), (0.79, 0.45), (0.72, 0.45)):
                        st.set_state((base0[0] + dx, base0[1], base0[2]), h_, pitch)
                        for sq in (np.array(NATURAL_CARRY_Q[side], float), np.array(HOME_Q[side], float)):
                            q, ep, er = st.solve(side, ph, Rh_, sq, iters=80, quick=30)
                            if ep < 0.005 and er < 0.04:
                                c_tip = min(c_tip, posture_cost(ref, side, natural_resolve(ref, side, ph, Rh_, q))
                                            + 0.1 * abs(dyaw))
                if c_tip < 0.3:
                    break
            c = c_tip + 0.3 * tip
            if best is None or c < best[0]:
                best = (c, [(tgt + np.array([0.0, 0.0, 0.06]), Rt), (tgt, Rt)], tip)
            if best[0] < 0.6:
                break
    targets = best[1] if best is not None else [(tgt + np.array([0.0, 0.0, 0.06]), Ro), (tgt, Ro)]
    if best is not None and best[2] > 0.0:
        ref.log("[ref] bin %s: tipped %.1f rad (posture %.2f)" % (name, best[2], best[0]))
    if best is not None and best[2] == 0.0:
        targets = [(tgt + np.array([0.0, 0.0, 0.06]), Ro), (tgt, Ro)]
    # (from 0.25 m short, bent there, the item brought over the opening there and stepped in: carried at the waist
    # straight in, it met the open lid's front edge, which stands at the waist's height over the bin)
    return put(ref, side, name, targets, base0, heights, pitches, release_settle=50, what="bin",
               absolute=best is not None and best[2] > 0.0, low=True, back=0.25)


def tray_spot(ref, name):
    """Where a dish goes on the dish tray (its frame: the bottom's middle): the plates stacked on the tray's west
    half, the glasses upright on its east half."""
    D = RM.DISH_TRAY
    (cx, cy), (w, l) = D["center"], D["size"]
    z0 = RM.KITCHEN["top_z"] + D["base_t"]
    kind = O.LOOSE[name][0]
    if kind == "plate":
        n = 0
        for k in O.DISHES:
            if k != name and O.LOOSE[k][0] == "plate":
                pk, _ = ref.c.pose(k)
                if abs(pk[0] - (cx - 0.045)) < 0.06 and abs(pk[1] - cy) < 0.08 and pk[2] < z0 + 0.06:
                    n += 1
        return np.array([cx - 0.045, cy, z0 + 0.001 + 0.022 * n])
    k = 0 if name.endswith("1") else 1
    return np.array([cx + 0.15, cy - 0.075 + 0.15 * k, z0 + 0.001])


def tray_spots(ref, name):
    """tray_spot first; for a glass then the tray's east half's other spot (if no glass stands there) and the top of
    the plate stack (a dish on a dish on the tray counts)."""
    out = [tray_spot(ref, name)]
    if O.LOOSE[name][0] != "glass":
        return out
    D = RM.DISH_TRAY
    (cx, cy) = D["center"]
    z0 = RM.KITCHEN["top_z"] + D["base_t"]
    k = 0 if name.endswith("1") else 1
    other = np.array([cx + 0.15, cy - 0.075 + 0.15 * (1 - k), z0 + 0.001])
    if all(np.linalg.norm(ref.c.pose(g)[0][:2] - other[:2]) > 0.08 for g in O.DISHES if g != name):
        out.append(other)
    top = z0
    for pl in O.DISHES:
        if O.LOOSE[pl][0] == "plate":
            pk, _ = ref.c.pose(pl)
            if abs(pk[0] - (cx - 0.045)) < 0.06 and abs(pk[1] - cy) < 0.08 and pk[2] < z0 + 0.06:
                top = max(top, pk[2] + O.PLATE["h"])
    if top > z0:
        out.append(np.array([cx - 0.045, cy, top + 0.001]))
    return out


def tray_put(ref, side, name):
    po, Ro = ref.pose(name)
    # (round 5) set down level: taken just after the take, while the robot still bends over the table, the dish is
    # tilted with the torso (8 deg at a 0.15 rad lean), and the put planned with that tilt dipped the plate's rim
    Ro = level(Ro)
    lat = 0.12 if side == "left" else -0.12
    support = ("dish_tray",) + tuple(f"{k}_col" for k in O.DISHES if k != name and O.LOOSE[k][0] == "plate")
    ok = False
    # the planned spot, then (a glass) the others
    for n, tgt in enumerate(tray_spots(ref, name)):
        ref.sync_held()
        if ref.held[side] != name:
            break
        if n:
            ref.log("[ref] tray %s: another spot (%.3f, %.3f, %.3f)" % (name, *tgt))
            ref.stow()
        above = tgt + np.array([0.0, 0.0, 0.07])
        targets = [(above, Ro), (tgt + np.array([0.0, 0.0, 0.03]), Ro), (tgt + np.array([0.0, 0.0, 0.008]), Ro)]
        # (round 5: carried at the waist, the dish is raised over the counter 0.25 m short of the stance, then the
        # robot steps in: as a person lifts a plate over the counter's edge)
        # (bent `back` short of the stance, held just clear of the counter where it is, then stepped in)
        back = 0.30 if O.LOOSE[name][0] == "plate" else 0.25
        ok = put(ref, side, name, targets, (RM.KITCHEN["top_x1"] + 0.16, tgt[1] + lat, math.pi), (0.79,),
                 (0.15, 0.30, 0.0, 0.45), release_settle=40, what="tray", allow=support, low=True, back=back, rank=8,
                 yaw_offsets=(0.0, 0.3, -0.3))
        if ok:
            break
    return ok


def shelf_put(ref, side, name):
    """Lay the held book flat in the target compartment (the top compartment of the bookshelf's west bay, east of the
    books there), on its board or on the other book already there: slid in from the front, spine (held) towards the
    robot, its middle 5 cm inside the shelf's front edge so the hand stays outside the compartment."""
    T = RM.BOOK_TARGET
    w, t, h = O.LOOSE[name][1]["size"]
    po, Ro = ref.pose(name)
    # lying flat (its thickness, body y, vertical), turned as it is held now: the planner keeps that turn relative to
    # the robot (the hand's natural pose for this grasp) and tries others (any turn fits the compartment)
    zs = 1.0 if Ro[2, 1] > 0 else -1.0
    yb = np.array([0.0, 0.0, zs])
    xb = Ro[:, 0] - (Ro[:, 0] @ yb) * yb
    xb /= np.linalg.norm(xb)
    Rf = np.stack([xb, yb, np.cross(xb, yb)], axis=1)
    base_z = T["shelf_z"] + t / 2 + 0.003
    for k in O.BOOKS:
        if k == name:
            continue
        pk, _ = ref.c.pose(k)
        if T["x0"] - 0.03 < pk[0] < T["x1"] + 0.03 and T["y0"] - 0.05 < pk[1] < T["y1"] + 0.05 and \
                pk[2] < T["shelf_z"] + 0.06:
            base_z = max(base_z, pk[2] + O.LOOSE[k][1]["size"][1] / 2 + t / 2 + 0.003)
    xc = (T["x0"] + T["x1"]) / 2 + 0.02
    y_in = (T["y0"] + T["y1"]) / 2 - 0.03            # (its middle well inside, whichever way it lies)
    y_out = T["y0"] - 0.13                            # in front of the shelf (the book's far edge 5 cm out)
    z_mid = T["shelf_z"] + T["clear_h"] * 0.42        # in at about the compartment's middle height
    lat = 0.12 if side == "left" else -0.12
    # in front of the opening, level with its middle (the robot steps in so, bent), then in
    targets = [(np.array([xc, y_out, z_mid]), Rf)]
    for k in range(1, 7):
        targets.append((np.array([xc, y_out + (y_in - y_out) * k / 6, z_mid + (base_z + 0.012 - z_mid) * k / 6]), Rf))
    targets.append((np.array([xc, y_in, base_z + 0.004]), Rf))
    # (facing north the right hand is on the robot's east side: the base stands `lat` along x from the target)
    # (bent at the stance itself the book, held in front of the chest, would sweep down through the shelf's top: the
    # robot bends 0.25 m short of it and steps in so, the book then in front of the opening)
    return put(ref, side, name, targets, (xc + lat, RM.SHELF["y0"] - 0.40, math.pi / 2), (0.79, 0.72),
               (0.30, 0.15, 0.45), release_settle=50, what="shelf",
               allow=("shelf_board2",) + tuple(f"{k}_col" for k in O.BOOKS if k != name), low=True, back=0.25,
               rank=8, yaw_offsets=(0.0, 0.3, -0.3),
               # (round 5: the farther stances first: standing close to the waist-high shelf the elbow was pulled far
               # behind the back to set the book in)
               offsets=[(-0.08, 0.0), (-0.12, 0.0), (-0.04, 0.0), (-0.08, 0.04), (-0.08, -0.04), (0.0, 0.0),
                        (-0.16, 0.0), (0.04, 0.0)])


def put_hold(ref, side, name, targets, base0, heights, pitches, what="hold"):
    """Like put, without letting go."""
    got = go_put(ref, side, name, targets, base0, heights, pitches, what)
    if got is None:
        return False
    ref.run_path(side, got[0])
    return True


# ================================================================================================ wiping
def sponge_path(ref, side, obj_poses, base0, heights=(0.79,), pitches=(0.30, 0.15, 0.45, 0.52),
                yaw_offsets=(0.0, 0.3, -0.3), what="sponge", walk=True, allow=(), offsets=None, run_upto=None,
                low=False, back=0.25):
    """Move the held sponge through object poses (its rotation free about the vertical): a planned stance (walked
    to; `low`, round 5: bent `back` metres short of it and the sponge brought to the first pose's place relative to the
    body there, then stepped in, as go_put does: carried at the waist it would sweep through the counter), then the
    planned joint path. Returns the stance or None; with run_upto = k only the path up to pose k is run and (stance,
    the remaining hand poses) is returned."""
    got = go_put(ref, side, "sponge", obj_poses, base0, heights, pitches, what, yaw_offsets, low=low, allow=allow,
                 back=back, offsets=offsets)
    if got is None:
        return None
    qs2, hp = got
    stance = tuple(float(v) for v in ref.base()) + (float(ref.c.ctrl.stand_cmd), float(ref.c.ctrl.waist_cmd[2]))
    if run_upto is None:
        ref.run_path(side, qs2)
        return stance
    nv = ref.stance.n_via
    ref.run_path(side, qs2[:nv + run_upto + 1])
    return stance, hp[run_upto + 1:]


def faucet_on(ref, on=True, side="left"):
    """Lift the lever to run the water (0.32 rad) or press it down: a point grip at its end, the hand following the
    lever's arc by position-only steps."""
    F = RM.FAUCET
    th0 = ref.c.mech.lever_angle()
    goal = 0.32 if on else 0.0
    if (on and th0 > F["on_above"] + 0.08) or (not on and th0 < F["on_above"] - 0.08):
        return True
    px, py, pz = F["pivot"]
    L = F["lever_len"] - 0.03

    def tip(th):
        return np.array([px + L * math.cos(th), py, pz + L * math.sin(th)])

    g = tip(th0)
    lat = 0.12 if side == "left" else -0.12
    base0 = (RM.KITCHEN["top_x1"] + 0.17, py + lat, math.pi)
    ref.dest_hint = (float(base0[0]), float(base0[1]))
    ref.stow()
    best = None
    found = []
    # (round 5) the lever's end taken from in front along the lever, the hand nearly level and the pinch across it
    # (the forearm forward, the wrist straight), the robot leaning in over the counter; then round 4's grasps from
    # above; of the first few that work, the one the arm takes most naturally (the stance search ranks its stances)
    for tilt in (0.0, 12.0, 55.0, 40.0, 70.0, 25.0):
        if found and tilt in (55.0, 40.0, 70.0, 25.0):
            break
        a = tilted((-1.0, 0.0), tilt)
        for pin in (np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])):
            Rh = hand_frame(a, pin)
            pre = g - a * 0.07 + np.array([0.0, 0.0, 0.03])
            arc = [(tip(th0 + (goal - th0) * k / 8.0), None) for k in range(1, 9)]
            ref.last_rank_cost = 0.0
            stance, qs = stance_search(ref, side, [(pre, Rh), (g, Rh)] + arc, base0, (0.79,),
                                       (0.30, 0.45, 0.15), yaw_offsets=(0.0, 0.3, -0.3), rank=4)
            if stance is not None:
                found.append((ref.last_rank_cost + 0.01 * tilt / 10.0, (stance, Rh, pre)))
                if len(found) >= 3:
                    break
        if len(found) >= 3:
            break
    if found:
        best = min(found, key=lambda f_: f_[0])[1]
    if best is None:
        ref.log("[ref] faucet: no stance")
        return False
    (x, y, yaw, h, pitch), Rh, pre = best
    if ref.walk(x, y, yaw, carrying=any(ref.held.values())) is False:
        ref.log("[ref] faucet: the walk there failed")
        return False
    ref.posture(h, pitch)
    b = ref.base()
    ok, worst, qs, why = ref.stance.check((b[0], b[1], b[2]), h, pitch, side, [(pre, Rh), (g, Rh)])
    if not ok:
        ref.log("[ref] faucet: no path from the stance (%s)" % (why,))
        return False
    ref.hand(side, False)
    ref.run_path(side, qs)
    ref.settle(side)
    got = ref.hand(side, True)
    if got != "faucet_lever":
        ref.log("[ref] faucet: closed on %s" % got)
        ref.hand(side, False)
        return False
    n = max(3, int(abs(goal - th0) / 0.02))
    ref.track(side, [(tip(th0 + (goal - th0) * k / n), None) for k in range(1, n + 1)], ticks=8, iters=40)
    ref.step(15)
    ref.release(side, settle=15)
    p, R = ref.site(side)
    ref.track(side, [(p + np.array([0.02 * k, 0.0, 0.015 * k]), None) for k in range(1, 5)], ticks=6, iters=6)
    ref.home_arm(side)
    ok = ref.c.mech.water_on() == on
    if not ok:
        ref.log("[ref] faucet %s: lever at %.2f" % ("on" if on else "off", ref.c.mech.lever_angle()))
    return ok


# (round 5) the sponge rinsed 7 cm under the spout's mouth (the cell rinses it anywhere in the stream between the basin's
# bottom and the mouth): held just over the counter's height the arm reaches it with the forearm forward (15 cm under
# the mouth, inside the basin, the hand reached down over the sink's front with the wrist bent)
RINSE_DROP = 0.07


def rinse(ref, side="right", seconds=2.6):
    """The sponge (in `side`) in the running water."""
    if not ref.c.mech.water_on():
        faucet_on(ref, True, side="left" if side == "right" else "right")
    ox, oy, oz = RM.FAUCET["outlet"]
    Ro = level(ref.pose("sponge")[1])
    lat = 0.12 if side == "left" else -0.12
    path = [(np.array([ox + 0.06, oy, oz - RINSE_DROP + 0.02]), Ro), (np.array([ox, oy, oz - RINSE_DROP]), Ro)]
    st = sponge_path(ref, side, path, (RM.KITCHEN["top_x1"] + 0.17, oy + lat, math.pi), what="rinse", low=True,
                     pitches=(0.0, 0.15, 0.30, 0.45))
    if st is None:
        return False
    l0 = ref.c.process.sponge.load
    ref.step(int(seconds / TICK))
    p, R = ref.site(side)
    ref.track(side, [(p + np.array([0.02 * k, 0.0, 0.02 * k]), R) for k in range(1, 5)], ticks=6)
    ref.log("[ref] rinse: load %.2f -> %.2f, soap %.2f" % (l0, ref.c.process.sponge.load, ref.c.process.sponge.soap))
    return True


def soap(ref, side="right"):
    """Press the held sponge down on the soap pump's head (16 mm): a dose of soap into it."""
    S = RM.SOAP
    sx, sy = S["base"]
    head_top = ref.pose_any("soap_pump")[0][2] + 0.010 + S["head_h"]
    hz = O.SPONGE["size"][2] / 2
    Ro = level(ref.pose("sponge")[1])
    lat = 0.12 if side == "left" else -0.12
    c0 = np.array([sx, sy, head_top + hz])
    # (round 5: from 3.5 cm above the head, not 6: the pump stands about shoulder high for the G1)
    path = [(c0 + np.array([0.0, 0.0, 0.035]), Ro), (c0 + np.array([0.0, 0.0, 0.004]), Ro)]
    doses0 = ref.c.process.sponge.doses             # (the approach itself may already press the head)
    st = sponge_path(ref, side, path, (RM.KITCHEN["top_x1"] + 0.17, sy + lat, math.pi), what="soap",
                     allow=("soap_pump_col", "soap_bottle_col"), low=True, pitches=(0.0, 0.15, 0.30))
    if st is None:
        return False
    p, R = ref.site(side)
    # down in 2 mm steps until the pump strokes (at 12 mm of its 20 mm travel; pressed further the sponge's edges meet
    # the bottle and the grip gives), at most 22 mm below the approach
    strokes0 = ref.c.mech.strokes
    depth = 0.0
    while depth < 0.022 and ref.c.mech.strokes == strokes0 and not ref.c.done:
        depth += 0.002
        ref.track(side, [(p + np.array([0.0, 0.0, -depth]), R)], ticks=8)
        ref.step(4)
        if ref.c.grips.holding(side) != "sponge" or ref.c.grips.tactile(side) > 18.0:
            break                      # (pressing on something else than the head: stop before the grip gives)
    ref.step(10)
    ref.track(side, [(p + np.array([0.0, 0.0, -depth + (depth + 0.05) * k / 5]), R) for k in range(1, 6)], ticks=6)
    ok = ref.c.process.sponge.doses > doses0
    if not ok:
        ref.log("[ref] soap: no dose (pump strokes %d, missed %d)" % (ref.c.mech.strokes, ref.c.process.doses_missed))
    return ok


def grease_cells(ref, thresh=0.03):
    g = ref.c.process.grease
    return g.xy[g.g > thresh]


def press_follow(ref, side, hand_poses, target_N=12.0, gain=0.00025, max_dz=0.005):
    """Slide the held sponge through hand poses on the table, its height corrected after each by the measured
    contact force (a stiff arm pressing a rigid sponge by position alone pushes far harder than the grip holds)."""
    dz = 0.0
    q = np.asarray(ref.c.ctrl.arm_cmd[side], float)
    for p, R in hand_poses:
        q, ep, er = ref.stance.nik.solve(ref.d.qpos, side, np.asarray(p, float) + np.array([0.0, 0.0, dz]),
                                         None if R is None else _quat(R), q, iters=30, null_total=0.03)
        ref.run_path(side, [q], tol_end=0.02, timeout=0.2)
        if ref.c.done or ref.c.grips.holding(side) != "sponge":
            return False
        surf, N = ref.c.process.sponge_contact()
        N = N if surf == "dining_top" else 0.0
        dz = float(np.clip(dz + gain * (N - target_N), -max_dz, max_dz))
    return True


def wipe_side(ref, side, xs, y_near, y_far, from_south, press=0.001, target_N=12.0):
    """In-and-out strokes (along y) at the given x positions from one long side of the dining table, the sponge
    pressed flat, the base stepping sideways between strokes. Stops early when the sponge loads up."""
    D = RM.DINING
    (dx, dy), (lx, ly) = D["center"], D["size"]
    top = D["top_z"]
    yaw_r = math.pi / 2 if from_south else -math.pi / 2
    y_base = dy - ly / 2 - 0.16 if from_south else dy + ly / 2 + 0.16
    lat = 0.12 if side == "left" else -0.12
    hz = O.SPONGE["size"][2] / 2
    sgn = 1.0 if from_south else -1.0
    done = 0
    failed_in_row = 0
    for k, x in enumerate(xs):
        ref.sync_held()
        if ref.held[side] != "sponge" and not take(ref, "sponge", side):
            return done, "lost"
        Ro = level(ref.pose("sponge")[1])
        zc = top + hz - press
        path = [(np.array([x, y_near, top + hz + 0.04]), Ro), (np.array([x, y_near, zc + 0.003]), Ro)]
        n = max(2, int(abs(y_far - y_near) / 0.015))
        path += [(np.array([x, y_near + (y_far - y_near) * i / n, zc]), Ro) for i in range(1, n + 1)]
        path += [(np.array([x, y_far + (y_near - y_far) * i / n, zc]), Ro) for i in range(1, n + 1)]
        path += [(np.array([x, y_near, top + hz + 0.04]), Ro)]
        bx = float(np.clip(x + (lat if from_south else -lat), dx - lx / 2 + 0.02, dx + lx / 2 - 0.02))
        res = None
        if k > 0:
            # the next stroke from where the robot stands if the arm reaches it (lifted, over, down): no stowing, no
            # new stance
            b = ref.base()
            hp = [ref.hand_for(side, p_, R_) for p_, R_ in path]
            ok, worst, qs2, why = ref.stance.check((b[0], b[1], b[2]), float(ref.c.ctrl.stand_cmd),
                                                   float(ref.c.ctrl.waist_cmd[2]), side, hp, holding="sponge",
                                                   grip_rel=ref.grip_rel.get(side), allow=("dining_top",),
                                                   fallback=False)
            if ok and path_posture(ref, side, (b[0], b[1], b[2]), float(ref.c.ctrl.stand_cmd),
                                   float(ref.c.ctrl.waist_cmd[2]), qs2, hp) > 2.0:
                ok = False              # (round 5: reached from here only with an unnatural arm: a stance of its own)
            if ok:
                ref.run_path(side, qs2[:ref.stance.n_via + 2])
                res = (None, hp[2:])
        if res is None:
            res = sponge_path(ref, side, path, (bx, y_base, yaw_r), heights=(0.79,), pitches=(0.45, 0.52, 0.35),
                              yaw_offsets=(0.0, 0.2, -0.2), what="wipe x %.2f" % x, allow=("dining_top",),
                              offsets=[(0.0, 0.0), (0.03, 0.0), (0.05, 0.0), (-0.03, 0.0), (0.03, 0.04),
                                       (0.03, -0.04)],
                              run_upto=1)
        if res is None:
            # (round 5: none of those reaches the stroke: a step nearer the table)
            res = sponge_path(ref, side, path, (bx, y_base, yaw_r), heights=(0.79,), pitches=(0.45, 0.52, 0.35),
                              yaw_offsets=(0.0, 0.2, -0.2), what="wipe x %.2f (nearer)" % x, allow=("dining_top",),
                              offsets=[(0.08, 0.0), (0.10, 0.0), (0.08, 0.04), (0.08, -0.04)], run_upto=1)
        if res is None:
            failed_in_row += 1
            if failed_in_row >= 2 and done == 0:
                return done, "unreachable"       # (round 5: no stroke of this side reached: not every x again)
            continue
        failed_in_row = 0
        _, rest = res
        press_follow(ref, side, rest[:-1], target_N=target_N)
        ref.track(side, rest[-1:], ticks=8)
        done += 1
        sp = ref.c.process.sponge
        # (a loaded sponge puts grease back as it slides, more than it takes once little is left: rinse early)
        if sp.load > 0.40 * CL.LOAD_MAX or sp.soap < 0.05:
            return done, "loaded" if sp.load > 0.40 * CL.LOAD_MAX else "no soap"
    return done, "ok"


def table_crumbs(ref):
    cr = ref.c.process.crumbs
    sel = np.nonzero((cr.state == CL.FREE) & (cr.on == "dining_top"))[0]
    return sel, cr.P[sel, :2]


def crumb_lanes(pts, lane=0.06):
    """Lanes (x, members) of crumbs along x, each at most `lane` wide."""
    order = np.argsort(pts[:, 0])
    lanes, cur = [], []
    for k in order:
        if cur and pts[k, 0] - pts[cur[0], 0] > lane:
            lanes.append(cur)
            cur = []
        cur.append(int(k))
    if cur:
        lanes.append(cur)
    return [(float((pts[m, 0].min() + pts[m, 0].max()) / 2), m) for m in lanes]


def wipe_crumbs_off(ref, side="right", target_N=4.0, passes=4):
    """The crumbs left on the dining table pushed off its nearer long edge onto the floor, where the sweep takes them:
    per lane of crumbs one stroke towards the edge, the sponge set down beyond the lane's crumbs (towards the middle),
    drawn lightly pressed until its middle is 1 cm inside the edge (the crumbs it pushes ride at its leading face,
    4-6 cm ahead of its middle, over the edge and fall), then lifted. A crumb the sponge lands next to may be knocked
    aside: up to `passes` passes, each over the crumbs where they now are. Returns the number of strokes."""
    D = RM.DINING
    (dx, dy), (lx, ly) = D["center"], D["size"]
    top = D["top_z"]
    hz = O.SPONGE["size"][2] / 2
    lat = 0.12 if side == "left" else -0.12
    done = 0
    for pas in range(passes):
        n0 = len(table_crumbs(ref)[0])
        if not n0 or ref.out_of_time(600):
            break
        for from_south in (True, False):
            sel, pts = table_crumbs(ref)
            if not len(sel) or ref.out_of_time(600):
                break
            mine = pts[pts[:, 1] <= dy] if from_south else pts[pts[:, 1] > dy]
            if not len(mine):
                continue
            yaw_r = math.pi / 2 if from_south else -math.pi / 2
            y_base = dy - ly / 2 - 0.16 if from_south else dy + ly / 2 + 0.16
            y_end = dy - ly / 2 + 0.01 if from_south else dy + ly / 2 - 0.01
            first = True
            for x, members in crumb_lanes(mine):
                ref.sync_held()
                if ref.held[side] != "sponge":
                    return done
                x = float(np.clip(x, dx - lx / 2 + 0.06, dx + lx / 2 - 0.06))
                ys = mine[members, 1]
                # (beyond the lane's crumbs, clear of them where the sponge lands, but no further than just past the
                # middle: the reach from the long side)
                # (and at least 15 cm from the edge: the light press needs a few centimetres to take hold)
                if from_south:
                    y0 = min(max(float(ys.max()) + 0.07, y_end + 0.15), dy + 0.015)
                else:
                    y0 = max(min(float(ys.min()) - 0.07, y_end - 0.15), dy - 0.015)
                Ro = level(ref.pose("sponge")[1])
                zc = top + hz - 0.001
                path = [(np.array([x, y0, top + hz + 0.04]), Ro), (np.array([x, y0, zc + 0.003]), Ro)]
                n = max(2, int(abs(y_end - y0) / 0.015))
                path += [(np.array([x, y0 + (y_end - y0) * k / n, zc]), Ro) for k in range(1, n + 1)]
                path += [(np.array([x, y_end, top + hz + 0.04]), Ro)]
                bx = float(np.clip(x + (lat if from_south else -lat), dx - lx / 2 + 0.02, dx + lx / 2 - 0.02))
                res = None
                if not first:
                    # from where the robot stands if the arm reaches (no stowing, no new stance)
                    b = ref.base()
                    hp = [ref.hand_for(side, p_, R_) for p_, R_ in path]
                    ok, worst, qs2, why = ref.stance.check((b[0], b[1], b[2]), float(ref.c.ctrl.stand_cmd),
                                                           float(ref.c.ctrl.waist_cmd[2]), side, hp, holding="sponge",
                                                           grip_rel=ref.grip_rel.get(side), allow=("dining_top",),
                                                           fallback=False)
                    if ok and path_posture(ref, side, (b[0], b[1], b[2]), float(ref.c.ctrl.stand_cmd),
                                           float(ref.c.ctrl.waist_cmd[2]), qs2, hp) > 2.0:
                        ok = False      # (round 5: reached from here only with an unnatural arm: a stance of its own)
                    if ok:
                        ref.run_path(side, qs2[:ref.stance.n_via + 2])
                        res = (None, hp[2:])
                if res is None:
                    # (round 5: farther back first, the stance ranked by the path's posture: 16 cm from the edge
                    # and leaning, the shoulder is over the edge, and the hand drawing the sponge to it pulled the
                    # elbow far behind the back)
                    res = sponge_path(ref, side, path, (bx, y_base, yaw_r), heights=(0.79,), pitches=(0.45, 0.52, 0.35),
                                      yaw_offsets=(0.0, 0.2, -0.2), what="crumbs x %.2f" % x, allow=("dining_top",),
                                      offsets=[(-0.06, 0.0), (0.0, 0.0), (-0.10, 0.0), (0.03, 0.0), (0.05, 0.0),
                                               (-0.03, 0.0), (0.03, 0.04), (0.03, -0.04)],
                                      run_upto=1)
                if res is None:
                    continue
                first = False
                _, rest = res
                press_follow(ref, side, rest[:-1], target_N=target_N)
                ref.track(side, rest[-1:], ticks=8)
                # (round 5) then back out over the table and a little higher: at the edge the hand is in front of the
                # hips at the table's height, the elbow folded and the wrist bent, and the robot straightened up for
                # the next lane's stance with the arm held so
                p_, R_ = ref.site(side)
                sg = 1.0 if from_south else -1.0
                ref.track(side, [(p_ + np.array([0.0, sg * 0.06 * k, 0.015 * k]), R_) for k in (1, 2)], ticks=8)
                done += 1
        n1 = len(table_crumbs(ref)[0])
        ref.log("[ref] crumbs off the table, pass %d: %d -> %d left on it (t %.0f)" % (pas, n0, n1, ref.c.sim_time))
        if n1 >= n0:
            break
    sel, pts = table_crumbs(ref)
    ref.log("[ref] crumbs off the table: %d strokes, %d crumbs left on it (t %.0f)" % (done, len(sel), ref.c.sim_time))
    return done


def degrease(ref, side="right", max_rounds=5):
    """Take the sponge, run the water, soap it, wipe the stain clean from both long sides of the table (rinsing and
    re-soaping as it loads up), rinse it, put it back in its tray, stop the water."""
    # (round 5) the water on first, both hands free (as a person does at a sink; with the sponge in one hand the other
    # reached the lever across the counter while the held sponge dipped towards it as the robot leant in)
    faucet_on(ref, True, side="left" if side == "right" else "right")
    if not take(ref, "sponge", side):
        faucet_on(ref, False, side="left" if side == "right" else "right")
        return False
    soap(ref, side)
    D = RM.DINING
    (dx, dy), (lx, ly) = D["center"], D["size"]
    for rnd in range(max_rounds):
        cells = grease_cells(ref)
        if len(cells) == 0 or ref.c.process.grease.removed_fraction() > 0.985:
            break
        strokes_round = 0
        for from_south in (True, False):
            sel = cells[cells[:, 1] <= dy + 0.03] if from_south else cells[cells[:, 1] > dy - 0.03]
            if len(sel) == 0:
                continue
            sp = ref.c.process.sponge
            if sp.load > 0.22 * CL.LOAD_MAX or sp.soap < 0.15:
                rinse(ref, side)
                soap(ref, side)
            x0, x1 = float(sel[:, 0].min()), float(sel[:, 0].max())
            xs = list(np.arange(x0, x1 + 0.001, 0.05)) or [x0]
            if from_south:
                y_near, y_far = float(sel[:, 1].min()) - 0.03, min(dy + 0.01, float(sel[:, 1].max()) + 0.03)
            else:
                y_near, y_far = float(sel[:, 1].max()) + 0.03, max(dy - 0.01, float(sel[:, 1].min()) - 0.03)
            done, why = wipe_side(ref, side, xs, y_near, y_far, from_south)
            ref.log("[ref] wipe round %d %s: %d of %d strokes, %s, removed %.3f, load %.1f, soap %.2f" % (
                rnd, "south" if from_south else "north", done, len(xs), why,
                ref.c.process.grease.removed_fraction(), ref.c.process.sponge.load, ref.c.process.sponge.soap))
            strokes_round += done
            if why == "lost":
                break
        if strokes_round == 0:
            n_rg = getattr(ref, "sponge_regrasps", 0)
            if n_rg < 2 and rnd == n_rg:
                # (round 5) the grip reaches no stroke from the table's sides: the sponge back into its tray and
                # taken again, first by the next of round 5's grips (this one, and its turn by half a turn, ruled
                # out), then by round 4's (by an edge or from above), which reach every stroke
                ref.log("[ref] degrease: no stroke reached with this grip; the sponge taken again (%s)" % (
                    "round 5's next grip" if n_rg == 0 else "round 4's grips"))
                ref.sponge_regrasps = n_rg + 1
                ref.sync_held()
                if side in ref.grip_rel:
                    ref.sponge_bad_grips = getattr(ref, "sponge_bad_grips", []) + [ref.grip_rel[side]]
                T_ = RM.SPONGE_TRAY
                tgt_ = np.array([T_["center"][0], T_["center"][1],
                                 RM.KITCHEN["top_z"] + 0.002 + O.SPONGE["size"][2] / 2 + 0.004])
                Ro_ = level(ref.pose("sponge")[1])
                put(ref, side, "sponge", [(tgt_ + np.array([0.0, 0.0, 0.05]), Ro_), (tgt_, Ro_)],
                    (RM.KITCHEN["top_x1"] + 0.17, T_["center"][1] + (0.12 if side == "left" else -0.12), math.pi),
                    (0.79,), (0.30, 0.45, 0.15), what="sponge-home", yaw_offsets=(0.0, 0.3, -0.3),
                    allow=("tray_bottom", "tray_rim_0", "tray_rim_1", "tray_rim_2", "tray_rim_3"), low=True,
                    back=0.25)
                ref.sync_held()
                if ref.held[side] is None:
                    ref.round4_sponge_grips = n_rg >= 1
                    try:
                        got = take(ref, "sponge", side)
                    finally:
                        ref.round4_sponge_grips = False
                    if got:
                        continue
            ref.log("[ref] degrease: no stroke reached in round %d; stopping the wiping" % rnd)
            break
        ref.sync_held()
        if ref.held[side] != "sponge":
            ref.log("[ref] degrease: the sponge is lost (t %.0f)" % ref.c.sim_time)
            break
    if ref.held[side] != "sponge":
        faucet_on(ref, False, side="left" if side == "right" else "right")
        return False
    rinse(ref, side, seconds=3.0)
    if WIPE_CRUMBS and wipe_crumbs_off(ref, side):
        ref.sync_held()
        if ref.held[side] == "sponge":
            rinse(ref, side, seconds=1.5)
    T = RM.SPONGE_TRAY
    (tx, ty) = T["center"]
    tgt = np.array([tx, ty, RM.KITCHEN["top_z"] + 0.002 + O.SPONGE["size"][2] / 2 + 0.004])
    Ro = level(ref.pose("sponge")[1])
    ok = put(ref, side, "sponge", [(tgt + np.array([0.0, 0.0, 0.05]), Ro), (tgt, Ro)],
             (RM.KITCHEN["top_x1"] + 0.17, ty + (0.12 if side == "left" else -0.12), math.pi), (0.79,),
             (0.30, 0.45, 0.15), what="sponge-home", yaw_offsets=(0.0, 0.3, -0.3),
             allow=("tray_bottom", "tray_rim_0", "tray_rim_1", "tray_rim_2", "tray_rim_3"), low=True, back=0.25)
    faucet_on(ref, False, side="left" if side == "right" else "right")
    return ok


# ================================================================================================ sweeping
def tool_take(ref, tool, side, grip_s, from_lip=False):
    """Take a tool (broom / dustpan) by its handle, `grip_s` metres along the handle feature from its lower end, from
    its clip or from where it stands, with a horizontal approach. A dustpan standing on the floor is taken from behind
    it, or with `from_lip` from in front of its lip, reaching over the pan (held so, its lip faces the robot: the way
    it is tipped into a bin by the wall, its handle then leaning towards the robot rather than into the wall)."""
    p, R = ref.pose(tool)
    f = next(f for f in ref.c.meta["features"][tool] if f[0] == "cyl")
    A, C = p + R @ np.asarray(f[1]), p + R @ np.asarray(f[2])
    u = (C - A) / np.linalg.norm(C - A)
    g = A + u * float(np.clip(grip_s, 0.04, np.linalg.norm(C - A) - 0.02))
    lat = 0.12 if side == "left" else -0.12
    if ref.c.clips.engaged.get(tool):
        yaw = -math.pi / 2                       # facing the south wall
        base0 = (g[0] + (0.12 if side == "right" else -0.12), g[1] + 0.40, yaw)
    elif tool == "dustpan" and from_lip:
        # standing on the floor, from in front of its lip, facing the pan
        ld = -R[:2, 1] / max(1e-9, float(np.linalg.norm(R[:2, 1])))
        yaw = math.atan2(-ld[1], -ld[0])
        # (beside the handle as usual, else with the handle in front of the middle or on the hand's far side: by a
        # table the usual place may put the other shoulder into it)
        base0 = [(p[0] + k * ld[0] + lt * math.sin(yaw), p[1] + k * ld[1] - lt * math.cos(yaw), yaw)
                 for lt in (lat, 0.0, -lat) for k in (0.30, 0.22, 0.40)]
    elif tool == "dustpan":
        # standing on the floor: from behind it, facing the way its lip faces (where it was set down from)
        ld = -R[:2, 1] / max(1e-9, float(np.linalg.norm(R[:2, 1])))
        yaw = math.atan2(ld[1], ld[0])
        base0 = (p[0] - 0.55 * ld[0] + lat * math.sin(yaw), p[1] - 0.55 * ld[1] - lat * math.cos(yaw), yaw)
    else:
        # standing on the floor: from the side the robot is on
        b = ref.base()
        yaw = math.atan2(g[1] - b[1], g[0] - b[0])
        base0 = (g[0] - 0.40 * math.cos(yaw) + lat * math.sin(yaw), g[1] - 0.40 * math.sin(yaw) - lat * math.cos(yaw),
                 yaw)
    ref.stow()
    best = None
    why = None
    found = []
    for b0 in (base0 if isinstance(base0, list) else [base0]):
        for tilt in (0.0, 15.0, -10.0, 30.0):
            to = np.array([math.cos(b0[2]), math.sin(b0[2])])
            a = tilted(to, tilt)
            a = a - (a @ u) * u
            a /= np.linalg.norm(a)
            for sgn in (1.0, -1.0):
                Rh = hand_frame(a, sgn * np.cross(u, a))
                pre = g - a * 0.10
                poses = [(pre, Rh), (g - a * 0.05, Rh), (g, Rh)]
                ref.last_rank_cost = 0.0
                stance, qs = stance_search(ref, side, poses, b0, (0.79, 0.70, 0.62, 0.55), (0.25, 0.10, 0.40, 0.52),
                                           yaw_offsets=(0.0, 0.3, -0.3), rank=3)
                if stance is not None:
                    # (round 5) the grip whose take the arm makes most naturally (both turns of the hand round the
                    # handle: one of them leaves the arm turned out for the carry and the strokes), of the first few
                    found.append((ref.last_rank_cost + 0.02 * abs(tilt) / 15.0, (stance, Rh, poses)))
                else:
                    why = qs
            if len(found) >= 3:
                break
        if found:
            break
    if found:
        best = min(found, key=lambda f_: f_[0])[1]
    if best is None:
        ref.log("[ref] take %s: no stance (%s)" % (tool, why))
        return False
    (x, y, yaw, h, pitch), Rh, poses = best
    ref.hand(side, False)
    if ref.walk(x, y, yaw) is False:
        ref.log("[ref] take %s: the walk there failed" % tool)
        return False
    ref.posture(h, pitch)
    b = ref.base()
    ok, worst, qs, why = ref.stance.check((b[0], b[1], b[2]), h, pitch, side, poses)
    if not ok:
        ref.log("[ref] take %s: no path from the stance (%s)" % (tool, why))
        return False
    ref.run_path(side, qs)
    ref.settle(side)
    got = ref.hand(side, True)
    if got != tool:
        ref.log("[ref] take %s: closed on %s" % (tool, got))
        if got:
            ref.hand(side, False)
        return False
    # out of the clip: 2 cm away from the wall, then up 5 cm (slowly: a long tool); the base then steps back
    p, R = ref.site(side)
    fwd = np.array([math.cos(b[2]), math.sin(b[2]), 0.0])
    path = [(p - fwd * 0.01 * k, R) for k in range(1, 3)]
    path += [(path[-1][0] + np.array([0.0, 0.0, 0.01 * k]), R) for k in range(1, 6)]
    ok, worst, qs, why = ref.stance.check((b[0], b[1], b[2]), h, pitch, side, path, holding=tool,
                                          grip_rel=ref.grip_rel.get(side), transit=False,
                                          allow=("floor", "tool_rail", "coffee_top"))
    if ok:
        ref.run_path(side, qs)
    else:
        ref.log("[ref] take %s: no way out of the clip (%s)" % (tool, why))
        ref.track(side, [(p + np.array([0.0, 0.0, 0.01 * k]), R) for k in range(1, 5)], ticks=10)
    # straight back 0.35 m before the tool is turned into its carrying pose (next to the wall or the pan's spot it
    # would swing into them)
    b = ref.base()
    ref.goto(b[0] - 0.35 * math.cos(b[2]), b[1] - 0.35 * math.sin(b[2]), b[2], tol=0.03, yaw_tol=0.05)
    return True


def pan_rot(lip_dir):
    """The dustpan's rotation standing on the floor with its lip facing lip_dir (a horizontal unit vector)."""
    y = -np.array([lip_dir[0], lip_dir[1], 0.0])
    y /= np.linalg.norm(y)
    z = np.array([0.0, 0.0, 1.0])
    return np.stack([np.cross(y, z), y, z], axis=1)


def set_dustpan(ref, side, P, lip_dir, surface=("floor", 0.0), from_lip=False):
    """Stand the held dustpan on the floor (or a table: surface = (its geom, its height)), its lip's middle at P,
    facing lip_dir; let go. The robot stands behind the pan's back, or with `from_lip` in front of its lip (the pan
    held with its lip towards the robot)."""
    R = pan_rot(lip_dir)
    p0 = np.array([P[0], P[1], surface[1] + 0.0015])
    targets = [(p0 + np.array([0.0, 0.0, 0.10]), R), (p0 + np.array([0.0, 0.0, 0.03]), R), (p0, R)]
    lat = 0.12 if side == "left" else -0.12
    if from_lip:
        # the robot in front of the lip, facing the pan (a few distances)
        yaw = math.atan2(-lip_dir[1], -lip_dir[0])
        ok = False
        for lt, k in [(lt, k) for lt in (lat, 0.0, -lat) for k in (0.30, 0.22, 0.40)]:
            base0 = (P[0] + k * lip_dir[0] + lt * math.sin(yaw), P[1] + k * lip_dir[1] - lt * math.cos(yaw), yaw)
            ok = put(ref, side, "dustpan", targets, base0, (0.79, 0.70, 0.62, 0.55), (0.30, 0.15, 0.45, 0.52),
                     release_settle=40, what="set-dustpan", yaw_offsets=(0.0, 0.3, -0.3), allow=(surface[0],))
            ref.sync_held()
            if ref.held[side] != "dustpan":
                break
        if ref.held[side] == "dustpan":
            return False
        ref.retreat(side)
        b = ref.base()
        ref.goto(b[0] - 0.35 * math.cos(b[2]), b[1] - 0.35 * math.sin(b[2]), b[2], tol=0.03, yaw_tol=0.05)
        ref.stow()
        return ok and (ref.c.process.crumbs.pan_down() if surface[0] == "floor" else True)
    else:
        # the robot behind the pan's back, facing the lip's direction
        yaw = math.atan2(lip_dir[1], lip_dir[0])
        base0 = (P[0] - 0.55 * math.cos(yaw) + lat * math.sin(yaw), P[1] - 0.55 * math.sin(yaw) - lat * math.cos(yaw),
                 yaw)
    ok = put(ref, side, "dustpan", targets, base0, (0.79, 0.70, 0.62, 0.55), (0.30, 0.15, 0.45, 0.52),
             release_settle=40, what="set-dustpan", yaw_offsets=(0.0, 0.3, -0.3, 0.6, -0.6), allow=(surface[0],))
    # the hand up off the handle, then straight back from the pan before anything else (the arm swung home next to
    # it knocks it over by its handle; turning next to it, the legs would sweep it along), then the arms home
    ref.retreat(side)
    b = ref.base()
    ref.goto(b[0] - 0.35 * math.cos(b[2]), b[1] - 0.35 * math.sin(b[2]), b[2], tol=0.03, yaw_tol=0.05)
    ref.stow()
    return ok and (ref.c.process.crumbs.pan_down() if surface[0] == "floor" else ref.held[side] is None)


def broom_rot(u, tilt, flip=False):
    """The broom handle's rotation for a stroke along u (horizontal unit vector): the head's long axis across the
    stroke, the handle leaning back towards the robot by `tilt`."""
    u = np.array([u[0], u[1], 0.0])
    u /= np.linalg.norm(u)
    zw = np.array([0.0, 0.0, 1.0])
    xb = np.cross(zw, u)
    if flip:
        xb = -xb
    zb = math.cos(tilt) * zw - math.sin(tilt) * u
    yb = np.cross(zb, xb)
    return np.stack([xb, yb, zb], axis=1)


def broom_pose(H, u, tilt, flip=False, lift=0.0):
    """(origin, rotation) of the broom with its bristles' bottom middle at H (on the floor) for a stroke along u."""
    # the bristles 12 mm above the floor: within the 20 mm the cell sweeps at (cell
    # manual), never pressed (the floor's push on the head, 0.4 m from the hand along the tilted handle, and its
    # friction a metre below the hand twist the broom out of the grip)
    return np.array([H[0], H[1], O.BROOM["head"][2] + O.BROOM["bristle"][2] + 0.012 + lift]), broom_rot(u, tilt, flip)


def stroke(ref, side, A, B, tilt=0.42):
    """One sweep of the held broom: its bristles set down at A, pushed along the floor to B by walking the base
    while the hand holds still relative to the body, lifted."""
    u = np.array([B[0] - A[0], B[1] - A[1]])
    L = float(np.linalg.norm(u))
    ref.sync_held()
    if L < 0.02 or ref.held[side] != "broom":
        return False
    u /= L
    yaw = math.atan2(u[1], u[0])
    lat = 0.12 if side == "left" else -0.12
    # (round 5) planned from the broom as it is raised at the stance before the stroke (below), not as carried for the
    # walk (lower: bent at the stance so, its head would meet the floor)
    ref.broom_stroke_start = True
    try:
        q_raised = ref.broom_upright(side, dry=True)
    finally:
        ref.broom_stroke_start = False
    ref.stance.seed_override = (side, np.asarray(q_raised, float)) if q_raised is not None else None
    try:
        for flip in (False, True):
            hov = broom_pose(A, u, tilt, flip, lift=0.05)
            down = broom_pose(A, u, tilt, flip)
            targets = [hov, down]
            for ahead in (0.66, 0.58, 0.74):
                # the hand (and the handle above the head) on the stroke's line: the base `lat` off it, to the other
                # side
                base0 = (A[0] - ahead * u[0] + lat * u[1], A[1] - ahead * u[1] - lat * u[0], yaw)
                ref.yaw_captured = ref.base()[2]
                stance, qs, poses = plan_put(ref, side, "broom", targets, base0, (0.79, 0.72), (0.15, 0.30, 0.0),
                                             yaw_offsets=(0.0, 0.15, -0.15),
                                             offsets=[(0.0, 0.0), (-0.04, 0.0), (0.04, 0.0), (0.0, 0.04),
                                                      (0.0, -0.04)])
                if stance is not None:
                    break
            if stance is not None:
                break
    finally:
        ref.stance.seed_override = None
    if stance is None:
        ref.log("[ref] stroke: no stance for (%.2f, %.2f) -> (%.2f, %.2f)" % (A[0], A[1], B[0], B[1]))
        return False
    x, y, yaw_s, h, pitch = stance
    if ref.walk(x, y, yaw_s, carrying=True) is False:
        ref.log("[ref] stroke: the walk there failed")
        return False
    # (round 5) carried for the walk with its head 10-14 cm up (broom_upright), the broom is raised back to 20 cm here
    # before the robot lowers and leans for the stroke (the posture change lowers it by up to 15 cm), so the stroke
    # starts as round 4's did
    ref.broom_stroke_start = True
    try:
        ref.carry_pose((side,))
    finally:
        ref.broom_stroke_start = False
    ref.posture(h, pitch)
    b = ref.base()
    tg = [(p, _Rz(ref.dy_chosen) @ free_yaw(ref, "broom", R, b[2], ref.yaw_captured)) for p, R in targets]
    hp = [ref.hand_for(side, p, R) for p, R in tg]
    ok, worst, qs, why = ref.stance.check((b[0], b[1], b[2]), h, pitch, side, hp, holding="broom",
                                          grip_rel=ref.grip_rel.get(side))
    if not ok:
        ref.log("[ref] stroke: no path from the stance (%s)" % (why,))
        return False
    # the walk along the stroke with the arms as they will be (this arm down, the other as it is): every link of the
    # robot and the broom clear of the room (the broom's bristles may touch the floor)
    st = ref.stance
    other = "right" if side == "left" else "left"
    q_down = last(qs[-1])
    for t_ in np.linspace(0.0, 1.0, 7):
        bt = (b[0] + u[0] * L * t_, b[1] + u[1] * L * t_, b[2])
        st.set_state(bt, h, pitch, {side: q_down, other: ref.d.qpos[ref.c.robot.arm_q[other]]})
        st.place_held(side, "broom", ref.grip_rel[side])
        # (the head brushes over the pan's lip and floor plate at the end, as it should; at most grazes its walls)
        bad = st.collisions(held=["broom"], allow=("floor", "dustpan_floor_col", "dustpan_lip"), tol=0.008)
        if bad:
            ref.log("[ref] stroke: the walk along it would touch %s" % (bad[0],))
            return False
    ref.run_path(side, qs)
    # the arm stays at its planned joints (re-commanding the joints it sagged to under the broom's load would sag it
    # again, the head then pressing on the floor: that load, 0.4 m ahead of the hand, twists the broom in the grip);
    # the hand moves with the base, which walks along the stroke slowly (a sweeping pace); the bristles are held about
    # 14 mm off the floor (the cell sweeps within 20 mm; the pan's floor plate at its lip is 4 mm thick) by trimming the
    # stand height to their measured height
    bg = mujoco.mj_name2id(ref.m, mujoco.mjtObj.mjOBJ_GEOM, "broom_bristle_col")
    trim = {"h": float(ref.c.ctrl.stand_cmd)}

    def hover(n_ticks, target=0.014):
        for _ in range(max(1, n_ticks // 2)):
            e = target - geom_lowest(ref.m, ref.d, bg)
            trim["h"] = float(np.clip(trim["h"] + np.clip(0.6 * e, -0.002, 0.002), h - 0.05, min(0.80, h + 0.05)))
            ref.c.command({"stand_height": trim["h"]})
            ref.step(2)
            if ref.c.done:
                return

    hover(40)
    b0 = ref.base()
    rel = _Rz(b0[2]).T @ np.array([u[0], u[1], 0.0])
    v = 0.18
    ref.c.command({"base_velocity": [float(v * rel[0]), float(v * rel[1]), 0.0]})
    slow = False
    for _ in range(int(3.0 * L / v / (2 * TICK)) + 80):          # until the base has gone the stroke's length
        hover(2)
        b = ref.base()
        gone = (b[0] - b0[0]) * u[0] + (b[1] - b0[1]) * u[1]
        if gone >= L - 0.008 or ref.c.done:
            break
        if not slow and L - gone < 0.06:
            # the last few centimetres slower (stopped from full pace the base runs on about 2 cm: the head into
            # the pan, against its side walls; much slower, a wobble of the head reverses its push and the crumbs
            # jump to its back edge)
            slow = True
            ref.c.command({"base_velocity": [float(0.10 * rel[0]), float(0.10 * rel[1]), 0.0]})
    ref.c.command({"base_velocity": [0.0, 0.0, 0.0]})
    hover(20)
    # the bristles up out of the sweeping band (20 mm) first: drawn back low, they would pull the crumbs out again
    hover(40, target=0.035)
    # then the head drawn back off the pan's lip before lifting it (lifted where it stopped, it catches on the pan)
    ref.c.command({"base_velocity": [float(-0.1 * rel[0]), float(-0.1 * rel[1]), 0.0]})
    hover(70, target=0.035)
    ref.c.command({"base_velocity": [0.0, 0.0, 0.0]})
    hover(20, target=0.04)
    # lift, slowly
    p, R = ref.site(side)
    ref.track(side, [(p + np.array([0.0, 0.0, 0.015 * k]), R) for k in range(1, 5)], ticks=15)
    return True


def floor_crumbs(ref):
    cr = ref.c.process.crumbs
    sel = np.nonzero((cr.state == CL.FREE) & (cr.on == "floor"))[0]
    return sel, cr.P[sel, :2]


def clusters_of(points, radius=0.32):
    """Greedy clustering: seeds at the densest points, members within `radius`."""
    left = list(range(len(points)))
    out = []
    while left:
        P = points[left]
        dens = [(np.linalg.norm(P - P[i], axis=1) < radius).sum() for i in range(len(P))]
        i = int(np.argmax(dens))
        members = [left[j] for j in range(len(P)) if np.linalg.norm(P[j] - P[i]) < radius]
        out.append(members)
        left = [k for k in left if k not in members]
    return out


def chain_groups(points, link=0.25):
    """Groups of points chained by neighbours within `link` (single linkage): a line of crumbs along a table edge is
    one group."""
    n = len(points)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        d = np.linalg.norm(points - points[i], axis=1)
        for j in np.nonzero(d < link)[0]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: -len(g))


def follow_stroke(ref, P, lip_dir, reach=1.1, half=0.32):
    """A further stroke into the pan where it stands (its lip's middle at P, facing lip_dir) for the floor crumbs left
    in front of it (within `reach` of the lip and `half` to either side of its axis): (A, B) for the 0.2 m lane with
    the most of them, converging on the lip's middle, or None."""
    sel, pts = floor_crumbs(ref)
    if not len(sel):
        return None
    v = -np.asarray(lip_dir, float)
    v /= np.linalg.norm(v)
    w = np.array([-v[1], v[0]])
    rel = pts - np.asarray(P, float)[None, :]
    along, lat = rel @ v, rel @ w
    cand = (along < -0.03) & (along > -reach) & (np.abs(lat) < half)
    best = None
    for lo in np.arange(-0.10, 0.1001, 0.05):
        near = cand & (np.abs(lat - lo) < 0.10)
        if near.any() and (best is None or near.sum() > best[0]):
            best = (int(near.sum()), lo, near)
    if best is None:
        return None
    n, lo, near = best
    A = np.asarray(P, float) + v * (float(along[near].min()) - 0.08) + w * lo
    B = np.asarray(P, float) + v * 0.012 + w * lo
    if not path_clear(ref, A, B):
        return None
    # only in open floor: the robot walking behind the broom (0.70 m back, the right hand's side) keeps the walking
    # planner's full margin from the furniture (a follow-up stroke squeezed in by a chair knocked the broom out of the
    # hand)
    pl = getattr(ref, "_open_planner", None) or Planner(radius=0.38)
    ref._open_planner = pl
    u = (B - A) / max(1e-9, float(np.linalg.norm(B - A)))
    nrm = np.array([-u[1], u[0]]) * 0.12
    if any(pl.occ[pl.cell(*(A + (B - A) * t - u * 0.70 + nrm))] for t in np.linspace(-0.2, 1.0, 7)):
        return None
    return A, B


def path_clear(ref, A, B, half=0.17, step=0.03):
    """The broom head's swath from A to B (half width) touches nothing standing on the floor."""
    obs = ref.c.process.crumbs.obstacles
    n = int(max(2, np.linalg.norm(np.asarray(B) - np.asarray(A)) / step))
    for t in np.linspace(0, 1, n):
        x, y = A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t
        if x < half or y < half or x > RM.ROOM_X - half or y > RM.ROOM_Y - half:
            return False
        if np.any((obs[:, 0] - half < x) & (x < obs[:, 2] + half) & (obs[:, 1] - half < y) & (y < obs[:, 3] + half)):
            return False
    return True


def under_furniture(P, v, margin=0.04):
    """Whether the dustpan standing with its lip's middle at P (its pan towards v) would be under a table top, a chair
    seat, the counter or the shelf (its handle rises 0.8 m)."""
    w = np.array([-v[1], v[0]])
    pts = [P + v * a + w * b for a in (0.0, 0.12, 0.24) for b in (-0.15, 0.0, 0.15)]
    for name, ((x0, y0, x1, y1), h) in RM.surfaces().items():
        if h < 0.3:
            continue
        for q in pts:
            if x0 - margin < q[0] < x1 + margin and y0 - margin < q[1] < y1 + margin:
                return True
    return False


def stand_ok(ref, xy, yaw):
    """The robot can stand at xy facing yaw: clear of the occupancy map's margin, or else its body (arms at home,
    nothing held: what it holds now will be posed otherwise there) touches nothing."""
    if not ref.planner.occ[ref.planner.cell(float(xy[0]), float(xy[1]))]:
        return True
    st = ref.stance
    st.set_state((float(xy[0]), float(xy[1]), float(yaw)), 0.79, 0.0, {sd: ARM_HOME[sd] for sd in ("left", "right")})
    return not st.collisions(held=None)


def plan_cluster(ref, pts, broom_side="right"):
    """(pan lip point P, lip direction, strokes [(A, B)]) for a group of floor crumbs: the pan's direction that sweeps
    the most of them (then the least stroke length), each lane's stroke kept only where the broom's swath and the
    robot's walk behind it are open floor."""
    c = pts.mean(axis=0)
    best = None
    pl = Planner(radius=0.26) if not hasattr(ref, "_lane_planner") else ref._lane_planner
    ref._lane_planner = pl
    for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        v = np.array([math.cos(ang), math.sin(ang)])
        far = float(np.max(pts @ v - c @ v))
        # the pan's footprint must be open floor, and the robot must have a place to set the pan down from: in front of
        # its lip facing it, or behind it facing the lip (the occupancy map's margin, else the robot's own body); the
        # pan 12 cm beyond the group, or further out (past a table leg)
        P = None
        dbg = getattr(ref, "plan_debug", None)
        for gap in (0.12, 0.30, 0.50):
            Pc = c + v * (far + gap)
            if not path_clear(ref, Pc, Pc + v * 0.30, half=0.20) or under_furniture(Pc, v):
                if dbg is not None:
                    dbg.append("%.0f/%.2f: pan" % (math.degrees(ang), gap))
                continue
            if not (stand_ok(ref, Pc - v * 0.30, math.atan2(v[1], v[0])) or
                    stand_ok(ref, Pc + v * 0.85, math.atan2(-v[1], -v[0]))):
                if dbg is not None:
                    dbg.append("%.0f/%.2f: stand" % (math.degrees(ang), gap))
                continue
            P = Pc
            break
        if P is None:
            continue
        # lanes across v
        w = np.array([-v[1], v[0]])
        off = (pts - c) @ w
        lanes = np.arange(off.min() - 0.02, off.max() + 0.021, 0.22) if off.max() - off.min() > 0.20 else [off.mean()]
        strokes, covered = [], 0
        for lo in lanes:
            near = np.abs(off - lo) < 0.15
            if not near.any():
                continue
            start_along = float(np.min((pts[near] - P) @ v)) - 0.08          # behind the farthest crumb of the lane
            A = P + v * start_along + w * (lo - float((P - c) @ w))
            # (the head's middle 1.2 cm past the lip: its face leaves the crumbs 2.9 cm ahead of its middle, 4 cm into
            # the pan, its front edge 4 cm in, short of the side walls (6.2 cm in); the lane within 4 cm of the pan's
            # middle: the head is as long as the pan is wide)
            B = P + v * 0.012 + w * float(np.clip(lo - float((P - c) @ w), -0.04, 0.04))
            ud = (B - A) / max(1e-9, float(np.linalg.norm(B - A)))
            if not path_clear(ref, A, B):
                if dbg is not None:
                    dbg.append("%.0f: lane swath" % math.degrees(ang))
                continue
            # (a lane converging on the pan at more than 7 deg turns the head against the pan's lip: its end runs
            # into a side wall; those crumbs are left for a pan placement of their own)
            if float(ud @ v) < math.cos(math.radians(7.0)):
                if dbg is not None:
                    dbg.append("%.0f: lane angle" % math.degrees(ang))
                continue
            # the robot walks behind the broom (about 0.7 m back along the stroke, beside the line): its own path must
            # be open floor
            u = (B - A) / max(1e-9, float(np.linalg.norm(B - A)))
            # (the right hand's stance is 0.12 m to the left of the line, the left's to the right: stroke(); it checks the robot's
            # whole body along it, this only keeps the lanes the base can walk)
            nrm = np.array([-u[1], u[0]]) * (0.12 if broom_side == "right" else -0.12)
            # (the occupancy map's cells next to its margin, else the robot's own body there)
            yaw_u = math.atan2(u[1], u[0])
            free = True
            for t in np.linspace(0.0, 1.0, 6):
                q = A + (B - A) * t - u * 0.70 + nrm
                if pl.occ[pl.cell(*q)] and not stand_ok(ref, q, yaw_u):
                    free = False
                    break
            if not free and dbg is not None:
                dbg.append("%.0f: lane walk" % math.degrees(ang))
            if not free:
                continue
            strokes.append((A, B))
            covered += int(near.sum())
        if not strokes:
            continue
        cost = (-covered, sum(np.linalg.norm(B - A) for A, B in strokes))
        if best is None or cost < best[0]:
            best = (cost, P, -v, strokes)
    return best


def sweep_floor(ref, max_groups=6, reserve=150.0, min_group=4, follow=1):
    """The floor round the dining set swept into the dustpan, the dustpan emptied into the general-waste bin, the tools
    back in their clips. One long tool in the hands at a time (the other one's head would hang where the robot then
    reaches): per group of crumbs (chained neighbours, the largest left first, `min_group` or more) the pan is set
    down by it, the broom taken from its clip (right hand), the planned strokes swept and up to `follow` more for
    what is left in front of the pan, the broom hung back, the pan taken again from its lip's side by whichever hand
    reaches it (held so it is emptied into the bin by the wall, and set down by the next group from that side).
    `reserve`: simulated seconds kept for the dump and the clips after the last group."""
    failed = []                     # centres of groups that could not be planned or served
    done = 0
    for _ in range(max_groups + 3):
        if done >= max_groups or ref.out_of_time(reserve + 240):
            break
        sel, pts = floor_crumbs(ref)
        if len(sel) < min_group:
            break
        groups = [g for g in chain_groups(pts) if len(g) >= min_group]
        groups = [g for g in groups if all(np.linalg.norm(pts[g].mean(axis=0) - f) > 0.15 for f in failed)]
        if not groups:
            break
        gp = pts[groups[0]]
        if getattr(ref, "plan_debug", None) is not None:
            ref.plan_debug = []
        plan = plan_cluster(ref, gp, "right")
        if plan is None:
            failed.append(gp.mean(axis=0))
            ref.log("[ref] sweep: no plan for a group of %d at (%.2f, %.2f)" % (len(gp), *gp.mean(axis=0)))
            if getattr(ref, "plan_debug", None):
                ref.log("[ref]   points %s; checks: %s" % (np.round(gp, 2).tolist(), ref.plan_debug))
            continue
        _, P, lip_dir, strokes = plan
        ref.log("[ref] sweep: a group of %d at (%.2f, %.2f): the pan's lip at (%.2f, %.2f) facing (%.2f, %.2f), %d "
                "strokes" % (len(gp), *gp.mean(axis=0), P[0], P[1], lip_dir[0], lip_dir[1], len(strokes)))
        ref.sync_held()
        holder = next((s_ for s_ in ("left", "right") if ref.held[s_] == "dustpan"), None)
        if holder is None:
            if not tool_take(ref, "dustpan", "left", PAN_GRIP_S):
                ref.log("[ref] sweep: the dustpan cannot be taken")
                break
            holder = "left"
            ok = set_dustpan(ref, holder, P, lip_dir, from_lip=True)
            ref.sync_held()
            if ref.held[holder] == "dustpan":
                ok = set_dustpan(ref, holder, P, lip_dir)
        else:
            ok = set_dustpan(ref, holder, P, lip_dir, from_lip=True)
            ref.sync_held()
            if ref.held[holder] == "dustpan":
                ok = set_dustpan(ref, holder, P, lip_dir)
        ref.sync_held()
        if any(ref.held[s_] == "dustpan" for s_ in ("left", "right")):
            failed.append(gp.mean(axis=0))
            ref.log("[ref] sweep: the pan could not be set down there")
            continue
        if not ok:
            ref.log("[ref] sweep: the pan is not standing where planned; sweeping anyway")
        # the strokes aimed at the pan where it now stands (a set-down lands a few centimetres and degrees off the plan)
        pp, Rpp = ref.pose("dustpan")
        ld_act = -np.asarray(Rpp[:2, 1], float)
        if np.linalg.norm(ld_act) > 0.5:
            ld_act /= np.linalg.norm(ld_act)
            v_old, v_act = -np.asarray(lip_dir, float), -ld_act
            w_old, w_act = np.array([-v_old[1], v_old[0]]), np.array([-v_act[1], v_act[0]])
            P_act = np.asarray(pp[:2], float)
            strokes = [(A_, P_act + v_act * 0.012 + w_act * float(np.clip((B_ - P) @ w_old, -0.04, 0.04)))
                       for A_, B_ in strokes]
            P, lip_dir = P_act, ld_act
        if ref.held["right"] != "broom" and not tool_take(ref, "broom", "right", BROOM_GRIP_S):
            ref.log("[ref] sweep: the broom cannot be taken")
            break
        n_st = 0
        for A, B in strokes:
            if ref.out_of_time(reserve + 90):
                break
            n_st += bool(stroke(ref, "right", A, B))
        for _k in range(follow):
            if ref.out_of_time(reserve + 90):
                break
            fu = follow_stroke(ref, P, lip_dir)
            if fu is None:
                break
            n_st += bool(stroke(ref, "right", *fu))
        n_pan = int((ref.c.process.crumbs.state == CL.PAN).sum())
        ref.log("[ref] sweep: %d strokes, %d crumbs in the pan (t %.0f)" % (n_st, n_pan, ref.c.sim_time))
        if n_st == 0:
            failed.append(gp.mean(axis=0))          # (not again: none of its strokes could be swept)
        ref.sync_held()
        if ref.held["right"] == "broom":
            tool_home(ref, "right", "broom")
        done += 1
        taken = False
        for s_ in ("left", "right"):
            ref.sync_held()
            if ref.held[s_] is None and tool_take(ref, "dustpan", s_, PAN_DUMP_S, from_lip=True):
                taken = True
                break
        if not taken:
            ref.log("[ref] sweep: the dustpan cannot be taken again")
            break
    ref.sync_held()
    for s_ in ("left", "right"):
        if ref.held[s_] == "broom":
            tool_home(ref, s_, "broom")
    for s_ in ("left", "right"):
        if ref.held[s_] == "dustpan":
            if int((ref.c.process.crumbs.state == CL.PAN).sum()) > 0:
                dump_pan(ref, s_)
            tool_home(ref, s_, "dustpan")
    return done > 0


def pan_regrasp_low(ref, side):
    """Set the held dustpan down on the coffee table (lip east, its back towards the robot standing west of it) and
    take it again low on its handle."""
    C = RM.COFFEE
    (cx, cy), (cw, cl) = C["center"], C["size"]
    ok = False
    # (its top is at knee height: the robot stands back from it and reaches over; from the south first, the side
    # facing the room, then the west and the north)
    for lip_dir, P in (((0.0, 1.0), (cx, cy - cl / 2 + 0.31)), ((1.0, 0.0), (cx - cw / 2 + 0.31, cy)),
                       ((0.0, -1.0), (cx, cy + cl / 2 - 0.31))):
        ref.sync_held()
        if ref.held[side] != "dustpan":
            return False
        ok = set_dustpan(ref, side, np.array(P), np.array(lip_dir), surface=("coffee_top", C["top_z"]))
        if ok:
            break
    if not ok:
        ref.log("[ref] pan regrasp: could not set it on the coffee table")
        return False
    ok = tool_take(ref, "dustpan", side, PAN_LOW_S)
    if not ok:
        ref.log("[ref] pan regrasp: could not take it again")
    return ok


PAN_GRIP_S = 0.58        # the dustpan held this far up its handle (from the handle feature's lower end) for the floor
#                          (round 5: 4 cm higher, the set-down standing ranked 140 on the human measure, the hand then
#                          so close in at the hip that the wrist bent to its limit, and the pan could not be set down
#                          again after a take from behind: kept)
PAN_LOW_S = 0.30         # ... and this low (the coffee-table regrasp; unused)
PAN_DUMP_S = 0.40        # ... and this far, taken from the lip's side off the floor, to empty it into a bin and hang it
                         # up (the tipped pan puts the hand 1.1-1.2 m up; held at the high grip it would be 1.3 m)
BROOM_GRIP_S = 0.60      # the broom (low enough on its handle to carry it upright beside the hip)


def dump_pan(ref, side, tips=(0.80, 0.92, 1.02, 1.10)):
    """Empty the held dustpan into the general-waste bin (its lid opened first if need be). The pan, held with its lip
    towards the robot (taken from the lip's side: tool_take from_lip), is brought in tipped 46 deg (the crumbs stay in
    below 55 deg) with its lip over the room-side part of the opening, then tipped about the lip until the crumbs pour
    out over it (past 55 deg) and further. Tipped, the handle leans over the lip: towards the robot here, so the hand
    stays in front of the robot (with the lip towards the wall it would be inside the wall)."""
    slot = ref.bin_slot("general")
    if ref.c.mech.lid_angle(slot) < RM.LID_HOLD:
        other = "right" if side == "left" else "left"
        ref.sync_held()
        if ref.held[other] is None:
            ref.lid(slot, True, side=other)
    yc = RM.BIN_SLOTS_Y[slot]
    B = RM.BIN
    Rp = pan_rot(np.array([1.0, 0.0]))
    lo = Rp[:, 0]
    # the sense of the turn about the pan's x axis that lowers the lip (brings the back wall up)
    Wb = np.array([0.0, O.DUSTPAN["depth"], O.DUSTPAN["wall_h"]])
    sgn = 1.0 if float((_axis_angle(lo, 0.5) @ Rp @ Wb)[2]) > float((Rp @ Wb)[2]) else -1.0
    ok = False
    for attempt in range(2):
        if attempt:
            # (round 5) no way there from where the robot stands (beside the dining table the pan hanging by the hip
            # may not turn there: the walk refuses to move): a step straight back, then the plans again
            b = ref.base()
            c0, s0 = math.cos(b[2]), math.sin(b[2])
            stepped = False
            for back in (0.30, 0.15):
                bx, by = b[0] - back * c0, b[1] - back * s0
                if not ref.planner.occ[ref.planner.cell(bx, by)] and \
                        ref.held_path_clear([], (bx, by), b[2], line=True):
                    ref.log("[ref] dump: %.2f m straight back first" % back)
                    ref.goto(bx, by, b[2], tol=0.03, yaw_tol=0.05)
                    stepped = not ref.pushed
                    break
            if not stepped:
                break
        for dx, dz in ((0.05, 0.025), (0.07, 0.04), (0.04, 0.05)):
            ref.sync_held()
            if ref.held[side] != "dustpan":
                break                       # (lost on the way: nothing left to plan with)
            lip = np.array([B["x1"] - dx, yc, B["rim_z"] + dz])
            targets = [(lip + np.array([0.12, 0.0, 0.0]), _axis_angle(lo, sgn * tips[0]) @ Rp)]
            targets += [(lip, _axis_angle(lo, sgn * a) @ Rp) for a in tips]
            # the robot facing the wall, the hand about 0.40 m ahead of it (the tipped pan puts the hand beyond the
            # lip)
            hp, _ = ref.hand_for(side, *targets[-1])
            base0 = (float(hp[0]) + 0.40, float(hp[1]) + (0.15 if side == "left" else -0.15), math.pi)
            ok = put_hold(ref, side, "dustpan", targets, base0, (0.79, 0.72, 0.65), (0.20, 0.35, 0.05, 0.45),
                          what="dump")
            if ok:
                break
        if ok or ref.held[side] != "dustpan":
            break
    ref.step(80)
    n_bin = int((ref.c.process.crumbs.state == CL.BIN_GENERAL).sum())
    ref.log("[ref] dump: %s, %d crumbs in the general bin, %d still in the pan" % (
        ok, n_bin, int((ref.c.process.crumbs.state == CL.PAN).sum())))
    if ok:
        # back out: the pan up and towards the robot, off the bin
        p, R = ref.site(side)
        ref.track(side, [(p + np.array([0.02 * k, 0.0, 0.015 * k]), R) for k in range(1, 6)], ticks=8)
        b = ref.base()
        ref.goto(b[0] + 0.25, b[1], b[2], tol=0.03, yaw_tol=0.05)
    return ok


def tool_home(ref, side, tool):
    """Hang the held tool back in its clip (its handle vertical through the clip's jaws), let go."""
    cx, cy, cz = RM.CLIPS[tool]
    pos, quat = LAYOUT.clip_pose(tool)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    targets = [(pos + np.array([0.0, 0.10, 0.03]), R), (pos + np.array([0.0, 0.04, 0.01]), R), (pos, R)]
    lat = 0.12 if side == "left" else -0.12
    ok = put(ref, side, tool, targets, (cx + (0.12 if side == "right" else -0.12), cy + 0.42, -math.pi / 2),
             (0.79, 0.70, 0.62, 0.55), (0.25, 0.10, 0.40, 0.52), release_settle=40, what="clip " + tool,
             allow=("tool_rail",),
             yaw_offsets=(0.0, 0.3, -0.3))
    ok = ok and ref.c.clips.engaged[tool]
    if not ok:
        ref.log("[ref] clip %s: not engaged" % tool)
    # (round 5) then straight back from the wall, 0.25 m: turning where it stood, beside the tools hanging there, the
    # robot pressed on them and the walk to the bins' lids gave up (the lids were left open)
    b = ref.base()
    bx, by = b[0] - 0.25 * math.cos(b[2]), b[1] - 0.25 * math.sin(b[2])
    if not ref.planner.occ[ref.planner.cell(bx, by)] or by > b[1]:
        ref.goto(bx, by, b[2], tol=0.03, yaw_tol=0.05)
    return ok


# ================================================================================================ the whole job
def clear_item(ref, name):
    """Take one loose object where it lies and put it where it belongs (right hand first, then the left)."""
    kind = O.LOOSE[name][0]
    for side in ("right", "left"):
        if ref.held[side]:
            continue
        if not take(ref, name, side):
            continue
        cat = O.CATEGORY[kind]
        if cat in ("recycling", "general"):
            ok = bin_put(ref, side, name, cat)
        elif cat == "dish":
            ok = tray_put(ref, side, name)
        elif cat == "book":
            ok = shelf_put(ref, side, name)
        else:
            ok = False
        if ref.held[side]:
            # could not put it away: set it down where it came from is not planned; let go over the floor
            ref.log("[ref] %s: still held after its put; letting go" % name)
            ref.hand(side, False)
            ref.step(40)
        return ok
    return False


# The floor: the table's crumbs are pushed off its long edges with the sponge after the degreasing (WIPE_CRUMBS), the
# floor's swept into the set-down dustpan group by group (sweep_floor), and the pan, held with its lip towards the
# robot, is tipped into the general-waste bin by the wall (dump_pan: held the other way round, the tipped pan's
# handle would put the hand inside the wall).
SWEEP = True
WIPE_CRUMBS = True


def run_job(ref):
    """The whole cleaning job (the module docstring's order)."""
    t0 = ref.c.sim_time
    log = ref.log
    for slot in (0, 1):
        if ref.out_of_time(120):
            break
        ref.lid(slot, True, side="right")
    order = ["can_green", "bottle", "napkin_3", "apple_core", "plate_1", "plate_2", "glass_1", "glass_2", "can_red",
             "napkin_1", "napkin_2", "book_red", "book_blue"]
    for name in order:
        if ref.out_of_time(400):
            break
        ok = clear_item(ref, name)
        log("[ref] %s: %s (t %.0f)" % (name, "done" if ok else "FAILED", ref.c.sim_time))
    if not ref.out_of_time(400):
        ok = degrease(ref, "right")
        log("[ref] degrease: %s, removed %.3f (t %.0f)" % (ok, ref.c.process.grease.removed_fraction(), ref.c.sim_time))
    if SWEEP and not ref.out_of_time(200):
        ok = sweep_floor(ref)
        st = ref.c.process.crumbs.counts()
        log("[ref] sweep: %s, crumbs %s (t %.0f)" % (ok, st, ref.c.sim_time))
    for slot in (0, 1):
        if ref.out_of_time(30):
            break
        ref.lid(slot, False, side="right")
    log("[ref] job over at t %.0f (%.0f s)" % (ref.c.sim_time, ref.c.sim_time - t0))
    return True
