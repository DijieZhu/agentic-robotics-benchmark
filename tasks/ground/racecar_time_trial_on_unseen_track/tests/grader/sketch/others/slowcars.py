"""Slow cars of the avoidance test.

    add_slow_cars(spec, world)                                  # into the MjSpec of the variant `avoid`
    slow = SlowCars(model, data, world, grip_at, seed)          # once per model
    at the start of the dynamic phase:  slow.enter()
    before every mj_step:               slow.pre_step()        (after the ego car's pre_step)
    slow.states() -> [{"id", "pose", "speed_mps", "on_track"}]

A slow car is another team's car: the owner's car (the drive model) with the 0.12 x 0.12 x 0.20 m tail box standing
on its rear deck (`tail_box_centre_m`, measured on the owner's model) and the box's mass. Its LiDAR geom is the whole
box; its collision geom is the box above `tail_box.col_bottom_z_m` (0.08 m), so that the nose of a car behind meets
the rear of the hull, not the lower edge of the box that overhangs it. It is driven by the grader
from privileged state along its path (`truth.avoidance.slow_cars[].path`, the trajectory of its reference point) at
`speed_factor` x the path's reference speed, slows when anything is within `gap_keep_m` ahead of its front inside its
lane, never yields and never defends.

Two ways to build it (`slow_car_mode`, chosen by gate G3):
  dynamic  the drive model's car (`add_car(..., tail_box=True)`) with its own `CarRuntime`; the follower sets its steering and
           speed targets once per tick (pure pursuit on the path, the speed law below); the drive model does the rest.
  mocap    the same bodies and geoms welded to one mocap body (no joints, no actuators); the follower moves it along
           its path (the path replayed in station, time-warped by the same speed law with a first-order speed
           response), level, at the drive model's resting height. It pushes whatever it touches and is never pushed.

Staging: until `enter()` every geom of a slow car is in group 5 with contact bits 0 / 0, so it cannot
be seen, scanned or touched. A dynamic car rests at its staging pose on its explicit tyre x floor pairs (a contact
pair ignores contype / conaffinity); a mocap car hangs there.

Entry: each car goes to the path point at the centreline station `entry_s_m`, moved forward along its path in
`staff.place_step_m` steps while the ego car's reference point is less than `entry_min_gap_m` away along the track or
less than 2 m away in a straight line, another car is less than `entry_min_gap_m` away along the track, or the car
would stand within 0.1 m of a box. It enters at rest.

Speed law, once per tick (gap = station distance from the car's front to the nearest point of anything inside its lane,
the lane = the path +- (half car width + `band_margin_m`); v_lead = that thing's speed along the path, clipped to
[0, v_cruise]; a_plan = `plan_decel_frac` x grip x g):
    v_cruise = speed_factor x v_ref(s + anticipation)
    v_keep   = clip(v_lead + (v_cruise - v_lead) (gap - g_mid) / (gap_keep - g_mid), 0, v_cruise),  0 below stop_gap
    v_safe   = v_lead + sqrt(2 a_plan max(0, gap - stop_gap))
    v_cmd    = min(v_cruise, v_keep, v_safe)  (no gap: v_cruise), rate-limited to +accel_cmd_frac and
               -decel_cmd_frac x grip x g per second (no limit below stop_gap + 0.1 m)
with g_mid = (stop_gap + gap_keep) / 2; the gap in both terms is first shortened by the closing speed x
`response_lag_s` (the car's speed response lags its command). Behind a slower car it settles about g_mid behind it, in
front of a standing obstacle it stops between stop_gap and g_mid before it. "Anything": the ego car, the other slow car (tail box included), the boxes in their present
pose, and the duct pieces whose world-file pose is near the path, at their present pose: read from the piece's LiDAR
geom `duct/<D>/<k>/lidar` (the flat-ended cylinder exactly at the piece pose), which every duct mode keeps
at the piece's present pose (the duct model's active_window moves the frozen layer's LiDAR geom with the slot that carries the
duct; its frozen `col` geom stays at the pre-activation pose with contact off). Version 1.0 looked up the bodies
`duct/<D>/<k>`, which the default active_window mode does not have, so this branch was empty.

Staff (dynamic only): a car that has rolled over (`reset_tilt_rad`), is more than `reset_offpath_m` off its path, or
stands still for `stuck_s` while its command is above `stuck_cmd_mps` is held at rest for `hold_s` and then put back
on its path at the nearest station with `place_clear_m` of free space around it (forward in `place_step_m` steps).
Reported in `stats()` as `resets`; the G2 gate requires none.

Randomness: none. The `seed` argument (the cell passes the venue's `others` stream) is kept for a later use and never drawn.
"""

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sketch.car import CarRuntime, add_car, car_xml
from sketch.car import load_params as load_drive_params
from sketch.examples import check
from sketch.examples import constants as K

from .boxes import shell_inertia
from .params import load_params
from .path import G, REF_X, Path, car_extent, outline, transform, wrap

MODE_TEXT = "others_mode"


def _yaw_quat(yaw):
    return [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]


def _quat_yaw(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tail_col_box(tail_box_centre, col_bottom_z):
    """(centre, half size) of the tail box's collision geom: the LiDAR box cut below `col_bottom_z`."""
    c = [float(v) for v in tail_box_centre]
    h = [0.5 * float(v) for v in K.TAIL_BOX_SIZE_M]
    top = c[2] + h[2]
    bottom = max(c[2] - h[2], float(col_bottom_z))
    return [c[0], c[1], 0.5 * (top + bottom)], [h[0], h[1], 0.5 * (top - bottom)]


def mocap_car_xml(tail_box_centre, drive_params=None, col_bottom_z=None):
    """The drive model's car with a tail box as one rigid mocap body: joints, tendons, equalities, actuators and excludes removed,
    every body welded to the chassis, the chassis a mocap body."""
    root = ET.fromstring(car_xml(drive_params, tail_box=True, tail_box_centre=tail_box_centre))
    if col_bottom_z is not None:
        pos, half = tail_col_box(tail_box_centre, col_bottom_z)
        for g in root.iter("geom"):
            if g.get("name") == "tail_box/col":
                g.set("pos", " ".join("%.10g" % v for v in pos))
                g.set("size", " ".join("%.10g" % v for v in half))
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag in ("joint", "freejoint", "tendon", "equality", "actuator", "contact"):
                parent.remove(child)
    chassis = root.find("worldbody/body")
    chassis.set("mocap", "true")
    chassis.attrib.pop("sleep", None)
    return ET.tostring(root, encoding="unicode")


def rest_height(drive_params=None):
    """The drive model's resting height of the car frame C over the floor (measured, default tyre model)."""
    p = drive_params or load_drive_params()
    rest = p.get("measured", {}).get("rest_height_m") or {}
    if isinstance(rest, dict):
        return float(rest.get(p["tyre_model"]) or 0.0)
    return float(rest)


def add_slow_cars(spec, world, mode=None, params=None, drive_params=None, tyre_model=None):
    """The slow cars of `truth.avoidance.slow_cars` at their staging poses, prefixes `slow<i>/`.
    `mode` 'dynamic' or 'mocap' (default `slow_car_mode`); the mode is written into the model as text
    `slow<i>/others_mode`."""
    p = load_params(params)
    mode = mode or p["slow_car_mode"]
    if mode not in ("dynamic", "mocap"):
        raise ValueError("mode must be 'dynamic' or 'mocap'")
    tb = [float(v) for v in p["tail_box_centre_m"]]
    zb = float(p["tail_box"]["col_bottom_z_m"])
    dp = drive_params or load_drive_params()
    for i, sc in enumerate(world["truth"]["avoidance"]["slow_cars"]):
        pre = K.slow_prefix(i)
        x, y, yaw = (float(v) for v in sc["staging_pose"])
        if mode == "dynamic":
            add_car(spec, prefix=pre, pose=(x, y, yaw), tail_box=True, params=dp, tyre_model=tyre_model,
                    tail_box_centre=tb)
            pos, half = tail_col_box(tb, zb)
            g = spec.geom(pre + "tail_box/col")
            g.pos, g.size = pos, half
            m = float(p["tail_box"]["mass_kg"])
            if m > 0.0:
                ch = spec.body(pre + "chassis")
                b = ch.add_body(name=pre + "tail_box/mass", pos=tb)
                b.mass = m
                b.inertia = shell_inertia(K.TAIL_BOX_SIZE_M, m)
                b.ipos = [0.0, 0.0, 0.0]
                b.explicitinertial = True
        else:
            child = mujoco.MjSpec.from_string(mocap_car_xml(tb, dp, zb))
            frame = spec.worldbody.add_frame(pos=[x, y, rest_height(dp)], quat=_yaw_quat(yaw))
            frame.attach_body(child.worldbody.first_body(), pre, "")
        spec.add_text(name=pre + MODE_TEXT, data=mode)


def _text(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, name)
    if i < 0:
        return None
    a, n = int(model.text_adr[i]), int(model.text_size[i])
    return bytes(model.text_data[a:a + n - 1]).decode("ascii")


class _Car(object):
    """Run-time state of one slow car."""


class SlowCars(object):
    """The slow cars of one model (see the module text). Signature of the contract."""

    def __init__(self, model, data, world, grip_at=None, seed=None, params=None, ego_prefix=K.EGO_PREFIX):
        p = load_params(params)
        self.p = p
        self.fp = p["follower"]
        self.st = p["staff"]
        self.model, self.data, self.world = model, data, world
        self.seed = seed
        self.dt = float(model.opt.timestep)
        self.N = int(round(K.TICK_S / self.dt))
        self.grip_at = grip_at if grip_at is not None else check.FloorMap(world).grip_at
        self.track = check.Track(world)
        self.entry_gap = float(p["entry_min_gap_m"])
        self.k = 0
        self.force_plan = False
        dp = load_drive_params()
        self.rest_z = rest_height(dp)
        self.lock = float(dp["steering"]["lock_rad"])
        self.b = K.WHEELBASE_M
        sp = self.fp["point_spacing_m"]
        self.tb = [float(v) for v in p["tail_box_centre_m"]]
        # the ego car (any car under ego_prefix with a free joint `root`)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ego_prefix + "root")
        self.ego = None if jid < 0 else (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
        x0, x1, hw = car_extent(None)
        self.ego_outline = outline(x0, x1, hw, sp)
        # boxes: free joints and the 12 edges of each box as points (any attitude)
        self.boxes = []
        for bx in world["truth"]["avoidance"]["boxes"]:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, K.box_body(bx["id"]) + "/root")
            if j >= 0:
                self.boxes.append((int(model.jnt_qposadr[j]), _box_edge_points(bx["size_m"], sp)))
        # cars
        self.cars = []
        x0, x1, hw = car_extent(self.tb)
        self.car_outline = outline(x0, x1, hw, sp)
        self.band = hw + float(self.fp["band_margin_m"])
        for i, sc in enumerate(world["truth"]["avoidance"]["slow_cars"]):
            self.cars.append(self._bind(i, sc, grip_at))
        self.paths = [c.path for c in self.cars]
        self.ducts = self._near_ducts()
        for c in self.cars:
            self._stage(c)
        mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------ binding and staging
    def _bind(self, i, sc, grip_at):
        m = self.model
        c = _Car()
        c.i, c.id, c.cfg, c.prefix = i, sc["id"], sc, K.slow_prefix(i)
        c.path = Path(sc["path"]["samples"])
        c.factor = float(sc["speed_factor"])
        c.gap_keep = float(sc["gap_keep_m"])
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c.prefix + "chassis")
        if cid < 0:
            raise KeyError("slow car %r is not in the model (add_slow_cars, variant 'avoid')" % c.prefix)
        c.body = cid
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, c.prefix + "root")
        mode = _text(m, c.prefix + MODE_TEXT)
        if jid >= 0:
            c.mode = "dynamic"
            c.qa, c.va = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
            c.rt = CarRuntime(m, self.data, prefix=c.prefix, grip_at=grip_at if grip_at is not None else self.grip_at)
        elif int(m.body_mocapid[cid]) >= 0:
            c.mode = "mocap"
            c.mid = int(m.body_mocapid[cid])
            c.rt = None
        else:
            raise ValueError("slow car %r has neither a free joint nor a mocap chassis" % c.prefix)
        if mode is not None and mode != c.mode:
            raise ValueError("slow car %r: model text says %r, the bodies say %r" % (c.prefix, mode, c.mode))
        c.geoms = np.array([g for g in range(m.ngeom) if int(m.body_rootid[int(m.geom_bodyid[g])]) == cid], dtype=int)
        c.saved = (m.geom_group[c.geoms].copy(), m.geom_contype[c.geoms].copy(), m.geom_conaffinity[c.geoms].copy())
        c.on = False
        c.s = 0.0
        c.v = 0.0               # mocap: present speed along the path
        c.v_cmd = 0.0
        c.steer = 0.0
        c.held_until = None
        c.stuck_t = 0.0
        c.resets = 0
        c.entry = None
        c.dist = 0.0
        c.max_lat = 0.0
        c.pose = list(sc["staging_pose"])
        c.speed = 0.0
        c.gap = None
        return c

    def _stage(self, c):
        m = self.model
        m.geom_group[c.geoms] = K.GROUP_HIDDEN
        m.geom_contype[c.geoms] = 0
        m.geom_conaffinity[c.geoms] = 0
        c.on = False
        pose = [float(v) for v in c.cfg["staging_pose"]]
        if c.mode == "dynamic":
            c.rt.place(pose)
            c.rt.set_targets(0.0, 0.0)
        else:
            self._write_mocap(c, pose)
        c.pose = pose

    def _unstage(self, c):
        m = self.model
        grp, ct, ca = c.saved
        m.geom_group[c.geoms] = grp
        m.geom_contype[c.geoms] = ct
        m.geom_conaffinity[c.geoms] = ca

    def _near_ducts(self):
        """Duct pieces whose world-file pose is near a path: (LiDAR geom id, half length) pairs. A model without
        movable ducts (pushable=False: no `ducts/mode` text, or `static`) has none to watch."""
        out = []
        if check.model_text(self.model, "ducts/mode") in (None, "static"):
            return out
        near = float(self.fp["duct_near_m"]) + K.DUCT_RADIUS_M + self.band
        pts = np.concatenate([c.path.p for c in self.cars]) if self.cars else np.zeros((0, 2))
        for d in self.world["truth"]["ducts"]:
            for k, piece in enumerate(d["pieces"]):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, K.duct_geom(d["id"], k, "lidar"))
                if gid < 0 or not len(pts):
                    continue
                if float(np.min(np.hypot(pts[:, 0] - piece[0], pts[:, 1] - piece[1]))) <= near + 0.5 * d["piece_length_m"]:
                    out.append((gid, 0.5 * float(d["piece_length_m"])))
        return out

    # ------------------------------------------------------------------ poses
    def _pose_of(self, c):
        if c.mode == "dynamic":
            q = self.data.qpos[c.qa:c.qa + 7]
            return [float(q[0]), float(q[1]), _quat_yaw(q[3:7])]
        return list(c.pose)

    def _speed_of(self, c):
        if c.mode == "dynamic":
            v = self.data.qvel[c.va:c.va + 2]
            yaw = self._pose_of(c)[2]
            return float(math.cos(yaw) * v[0] + math.sin(yaw) * v[1])
        return float(c.v)

    def _tilt_cos(self, c):
        q = self.data.qpos[c.qa + 3:c.qa + 7]
        w, x, y, z = (float(v) for v in q)
        return 1.0 - 2.0 * (x * x + y * y)                 # z axis of the chassis . world z

    def _ego_state(self):
        if self.ego is None:
            return None
        qa, va = self.ego
        q = self.data.qpos[qa:qa + 7]
        return [float(q[0]), float(q[1]), _quat_yaw(q[3:7])], (float(self.data.qvel[va]), float(self.data.qvel[va + 1]))

    def _pose_at(self, path, s):
        x, y, h, _ = path.at(s)
        return [x - REF_X * math.cos(h), y - REF_X * math.sin(h), h]

    def _write_mocap(self, c, pose):
        self.data.mocap_pos[c.mid] = [pose[0], pose[1], self.rest_z]
        self.data.mocap_quat[c.mid] = _yaw_quat(pose[2])

    # ------------------------------------------------------------------ entry
    def enter(self, ego_pose=None):
        """Start of the dynamic phase: every slow car onto its path (see the module text). `ego_pose` overrides the
        ego car's pose read from the model."""
        es = self._ego_state()
        ego = list(ego_pose) if ego_pose is not None else (es[0] if es is not None else None)
        ego_ref = None if ego is None else (ego[0] + REF_X * math.cos(ego[2]), ego[1] + REF_X * math.sin(ego[2]))
        s_ego = None if ego_ref is None else self.track.project(*ego_ref)[0]
        placed = []
        step = float(self.st["place_step_m"])
        for c in self.cars:
            ex, ey = self.track.at(float(c.cfg["entry_s_m"]))[:2]
            s0 = c.path.project(ex, ey)[0]
            chosen, fallback = None, True
            for k in range(int(c.path.length / step) + 1):
                s = (s0 + k * step) % c.path.length
                if self._entry_ok(c, s, ego_ref, s_ego, placed):
                    chosen, fallback = s, False
                    break
            if chosen is None:
                chosen = s0
            self._place_on_path(c, chosen)
            c.entry = {"s_m": round(chosen, 6), "shift_m": round((chosen - s0) % c.path.length, 6), "fallback": fallback}
            placed.append(c)
        self.force_plan = True
        mujoco.mj_forward(self.model, self.data)

    def _entry_ok(self, c, s, ego_ref, s_ego, placed):
        x, y, _, _ = c.path.at(s)
        sc = self.track.project(x, y)[0]
        L = self.track.length
        if ego_ref is not None:
            gap = abs(((sc - s_ego) + 0.5 * L) % L - 0.5 * L)
            if gap < self.entry_gap or math.hypot(x - ego_ref[0], y - ego_ref[1]) < 2.0:
                return False
        for o in placed:
            ox, oy, _, _ = o.path.at(o.s)
            so = self.track.project(ox, oy)[0]
            if abs(((sc - so) + 0.5 * L) % L - 0.5 * L) < self.entry_gap:
                return False
        pose = self._pose_at(c.path, s)
        pts = transform(self.car_outline, pose)
        for qa, edge in self.boxes:
            bp = self._box_points(qa, edge)
            if _min_dist(pts, bp) < 0.1:
                return False
        return True

    def _place_on_path(self, c, s):
        self._unstage(c)
        pose = self._pose_at(c.path, s)
        c.s, c.v, c.v_cmd, c.steer = s, 0.0, 0.0, 0.0
        c.held_until, c.stuck_t = None, 0.0
        if c.mode == "dynamic":
            c.rt.place(pose)
            c.rt.set_targets(0.0, 0.0)
        else:
            self._write_mocap(c, pose)
        c.pose = pose
        c.on = True

    # ------------------------------------------------------------------ one physics step
    def pre_step(self):
        plan = self.force_plan or (self.k % self.N == 0)
        self.k += 1
        self.force_plan = False
        for c in self.cars:
            if not c.on:
                continue
            if c.mode == "dynamic":
                if plan:
                    self._plan_dynamic(c)
                c.rt.pre_step()
            else:
                if plan:
                    self._plan_mocap(c)
                self._move_mocap(c)

    @property
    def t(self):
        return self.k * self.dt

    # ------------------------------------------------------------------ the follower
    def _observe(self, c):
        pose = self._pose_of(c)
        rx, ry = pose[0] + REF_X * math.cos(pose[2]), pose[1] + REF_X * math.sin(pose[2])
        s, lat, d2 = c.path.project(rx, ry, s_hint=c.s, back_m=1.0, ahead_m=2.0)
        if d2 > 0.36:                                  # far from where it was: search the whole path
            s, lat, d2 = c.path.project(rx, ry)
        ds = (s - c.s) % c.path.length
        if ds < 0.5 * c.path.length:
            c.dist += ds
        c.s = s
        c.pose = pose
        c.speed = self._speed_of(c)
        c.max_lat = max(c.max_lat, abs(lat))
        return pose, s, lat

    def _grip(self, x, y):
        return float(self.grip_at(x, y))

    def _box_points(self, qa, edge):
        q = self.data.qpos[qa:qa + 7]
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, np.asarray(q[3:7], dtype=float))
        w = np.asarray(q[:3], dtype=float) + np.einsum("ij,kj->ik", edge, R.reshape(3, 3))     # BLAS-free
        return w[:, :2]

    def _obstacles(self, c, s_front, look):
        """Points (k, 2), inflation radii (k,) and lead speeds (k,) of everything that might be in c's lane ahead."""
        pts, rad, vel = [], [], []
        cx, cy = c.pose[0], c.pose[1]
        reach = look + 1.0
        es = self._ego_state()
        if es is not None:
            pose, v = es
            if math.hypot(pose[0] - cx, pose[1] - cy) < reach + 0.5:
                p = transform(self.ego_outline, pose)
                pts.append(p)
                rad.append(np.zeros(len(p)))
                vel.append(np.repeat([v], len(p), axis=0))
        for o in self.cars:
            if o is c or not o.on:
                continue
            if math.hypot(o.pose[0] - cx, o.pose[1] - cy) < reach + 0.6:
                p = transform(self.car_outline, o.pose)
                v = (o.speed * math.cos(o.pose[2]), o.speed * math.sin(o.pose[2]))
                pts.append(p)
                rad.append(np.zeros(len(p)))
                vel.append(np.repeat([v], len(p), axis=0))
        for qa, edge in self.boxes:
            q = self.data.qpos[qa:qa + 2]
            if math.hypot(float(q[0]) - cx, float(q[1]) - cy) < reach + 0.5:
                p = self._box_points(qa, edge)
                pts.append(p)
                rad.append(np.zeros(len(p)))
                vel.append(np.zeros((len(p), 2)))
        for gid, half in self.ducts:
            x, y = float(self.data.geom_xpos[gid][0]), float(self.data.geom_xpos[gid][1])
            if math.hypot(x - cx, y - cy) < reach + half + K.DUCT_RADIUS_M:
                ax = self.data.geom_xmat[gid].reshape(3, 3)[:2, 2]             # a cylinder's axis is its z axis
                p = np.array([[x + u * ax[0], y + u * ax[1]] for u in np.linspace(-half, half, 5)])
                pts.append(p)
                rad.append(np.full(len(p), K.DUCT_RADIUS_M))
                vel.append(np.zeros((len(p), 2)))
        if not pts:
            return None
        return np.concatenate(pts), np.concatenate(rad), np.concatenate(vel)

    def _gap(self, c, s, v_now, mu):
        """(gap, lead speed) of the nearest thing in c's lane ahead within the detection length, or (None, None)."""
        s_front = s + (K.X_FRONT_M - REF_X)
        a_plan = float(self.fp["plan_decel_frac"]) * mu * G
        look = c.gap_keep + v_now * v_now / (2.0 * max(a_plan, 0.1)) + float(self.fp["look_extra_m"])
        obs = self._obstacles(c, s_front, look)
        if obs is None:
            return None, None
        pts, rad, vel = obs
        rel, lat, d2 = c.path.project_many(pts, s_front, look, back_m=0.5)
        inside = (d2 <= (self.band + rad) ** 2) & (rel >= -0.02) & (rel <= look)
        if not inside.any():
            return None, None
        g = rel[inside] - rad[inside]
        k = int(np.argmin(g))
        sl = float((s_front + rel[inside][k]) % c.path.length)
        _, _, h, _ = c.path.at(sl)
        vl = vel[inside][k]
        return max(0.0, float(g[k])), float(vl[0] * math.cos(h) + vl[1] * math.sin(h))

    def _speed_cmd(self, c, s, mu):
        fp = self.fp
        v_ref = c.path.speed(s + float(fp["anticipation_m"]))
        v_cruise = c.factor * v_ref
        v_now = max(c.speed, 0.0)
        gap, v_lead = self._gap(c, s, max(v_now, v_cruise), mu)
        c.gap = gap
        v = v_cruise
        stop, keep = float(fp["stop_gap_m"]), c.gap_keep
        if gap is not None:
            vl = min(max(v_lead, 0.0), v_cruise)
            gap = gap - max(v_now - vl, 0.0) * float(fp["response_lag_s"])     # what the gap will be after the lag
            g_mid = 0.5 * (stop + keep)
            if gap <= stop:
                v_keep = 0.0
            else:
                v_keep = min(max(vl + (v_cruise - vl) * (gap - g_mid) / (keep - g_mid), 0.0), v_cruise)
            a_plan = float(fp["plan_decel_frac"]) * mu * G
            v_safe = vl + math.sqrt(2.0 * a_plan * max(0.0, gap - stop))
            v = min(v_cruise, v_keep, v_safe)
        tick = self.N * self.dt
        up = float(fp["accel_cmd_frac"]) * mu * G * tick
        dn = float(fp["decel_cmd_frac"]) * mu * G * tick
        emergency = gap is not None and gap < stop + 0.1
        lo = 0.0 if emergency else c.v_cmd - dn
        c.v_cmd = max(0.0, min(c.v_cmd + up, max(v, lo)))
        return c.v_cmd

    def _plan_dynamic(self, c):
        pose, s, lat = self._observe(c)
        st = self.st
        if c.held_until is not None:
            if self.t >= c.held_until:
                self._reset(c, s)
            else:
                c.rt.set_targets(0.0, 0.0)
            return
        mu = self._grip(pose[0], pose[1])
        v_cmd = self._speed_cmd(c, s, mu)
        # staff: rolled over, off the path, or stuck while asked to drive
        tick = self.N * self.dt
        if c.speed < float(st["stuck_speed_mps"]) and v_cmd > float(st["stuck_cmd_mps"]):
            c.stuck_t += tick
        else:
            c.stuck_t = 0.0
        if (self._tilt_cos(c) < math.cos(float(st["reset_tilt_rad"])) or abs(lat) > float(st["reset_offpath_m"])
                or c.stuck_t >= float(st["stuck_s"])):
            c.held_until = self.t + float(st["hold_s"])
            c.v_cmd = 0.0
            c.rt.set_targets(0.0, 0.0)
            return
        # pure pursuit of the path by the rear axle (the car frame's origin)
        k0, k1, lmin, lmax = (float(v) for v in self.fp["lookahead_m"])
        ld = min(max(k0 + k1 * max(c.speed, 0.0), lmin), lmax)
        s_rear = (s - REF_X) % c.path.length
        tx, ty, _ = c.path.target_at_distance(pose[0], pose[1], s_rear, ld)
        dx, dy = tx - pose[0], ty - pose[1]
        alpha = wrap(math.atan2(dy, dx) - pose[2])
        steer = math.atan2(2.0 * self.b * math.sin(alpha), ld)
        c.steer = min(max(steer, -self.lock), self.lock)
        c.rt.set_targets(c.steer, v_cmd)

    def _reset(self, c, s):
        """The staff put a held car back on its path: nearest station with free space, at rest."""
        step = float(self.st["place_step_m"])
        clear = float(self.st["place_clear_m"])
        es = self._ego_state()
        chosen = s
        for k in range(int(c.path.length / step) + 1):
            cand = (s + k * step) % c.path.length
            pts = transform(self.car_outline, self._pose_at(c.path, cand))
            ok = True
            if es is not None and _min_dist(pts, transform(self.ego_outline, es[0])) < clear:
                ok = False
            for o in self.cars:
                if ok and o is not c and o.on and _min_dist(pts, transform(self.car_outline, o.pose)) < clear:
                    ok = False
            for qa, edge in self.boxes:
                if ok and _min_dist(pts, self._box_points(qa, edge)) < 0.1:
                    ok = False
            if ok:
                chosen = cand
                break
        self._place_on_path(c, chosen)
        c.resets += 1
        self.force_plan = True

    def _plan_mocap(self, c):
        x, y, _, _ = c.path.at(c.s)
        c.pose = self._pose_at(c.path, c.s)
        c.speed = c.v
        mu = self._grip(x, y)
        c.mu = mu
        self._speed_cmd(c, c.s, mu)

    def _move_mocap(self, c):
        dt = self.dt
        mu = getattr(c, "mu", 1.0)
        tau = float(self.fp["mocap_tau_s"])
        dv = (c.v_cmd - c.v) * dt / tau
        up = float(self.fp["accel_cmd_frac"]) * mu * G * dt
        dn = float(self.fp["decel_cmd_frac"]) * mu * G * dt
        if c.gap is not None and c.gap < float(self.fp["stop_gap_m"]) + 0.1:
            dn = mu * G * dt
        c.v = max(0.0, c.v + min(max(dv, -dn), up))
        ds = c.v * dt
        c.s = (c.s + ds) % c.path.length
        c.dist += ds
        pose = self._pose_at(c.path, c.s)
        c.pose = pose
        self._write_mocap(c, pose)

    # ------------------------------------------------------------------ read-outs
    def states(self):
        out = []
        for c in self.cars:
            pose = self._pose_of(c) if c.on else [float(v) for v in c.cfg["staging_pose"]]
            out.append({"id": c.id, "pose": pose, "speed_mps": self._speed_of(c) if c.on else 0.0, "on_track": bool(c.on)})
        return out

    def stats(self):
        """Per car: mode, entry, resets, distance driven along the path, largest lateral error (reported only)."""
        return [{"id": c.id, "mode": c.mode, "entry": c.entry, "resets": c.resets, "distance_m": round(c.dist, 4),
                 "max_lateral_error_m": round(c.max_lat, 4)} for c in self.cars]


def _box_edge_points(size, spacing):
    """Points on the 12 edges of a box of the given size (local frame, centred)."""
    hx, hy, hz = (0.5 * float(v) for v in size)
    corners = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    pts = []
    for a in range(8):
        for b in range(a + 1, 8):
            diff = np.abs(corners[a] - corners[b])
            if int((diff > 1e-12).sum()) != 1:
                continue
            n = max(1, int(math.ceil(float(diff.max()) / spacing)))
            for k in range(n + 1):
                pts.append(corners[a] + (corners[b] - corners[a]) * k / n)
    return np.unique(np.round(np.asarray(pts), 9), axis=0)


def _min_dist(a, b):
    d = a[:, None, :] - b[None, :, :]
    return float(np.sqrt((d * d).sum(axis=2).min()))
