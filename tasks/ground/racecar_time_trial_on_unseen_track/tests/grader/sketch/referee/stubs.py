"""Stand-ins for the drive model, the sensors, the venue generator, the ducts and the obstacles, on the contract's greybox model, coded against the signatures of the contract.

They let the referee and the wire run end to end before the drive model, the sensors, the venue exporter, the pushable ducts and the boxes and slow cars exist. `providers.resolve()` replaces each of them by
the real package as soon as it can be imported. None of them is a model of anything:

  KinematicCar      a kinematic bicycle at the rear axle (the car frame C) written into the stub car's free joint
                    before every physics step: servo rate limit, first-order speed loop, no tyres, no slip; it passes
                    through walls (contacts are still reported, which is what the referee needs)
  IdealSensors      sub-scans cast at the contract's instants with mj_multiRay, no noise, no material range, no floor
                    rule; IMU from finite differences of the site velocity; no bias
  StaticDuctTracker the greybox's static ducts never move
  StubBoxTracker    the DuctTracker interface for the free boxes (tracked points: the four bottom corners)
  StubSlowCars      kinematic slow cars on their path at speed_factor x the reference speed, slowing for the ego car
"""

import math

import mujoco
import numpy as np

from sketch.examples import check
from sketch.examples import constants as K
from sketch.examples import greybox

from .config import load_params

G = 9.81


def _yaw_quat(yaw):
    return [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]


def _quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _id(model, kind, name):
    return mujoco.mj_name2id(model, kind, name)


# ----------------------------------------------------------------------------------------------- model pieces
def venue_spec(world, variant="track", pushable=False):
    """The venue generator's `build_spec` stand-in: the contract's greybox venue (floor, zones, tape, timing line, clutter, static ducts)."""
    return mujoco.MjSpec.from_string(greybox.build_xml(world, variant="track", cars=None))


def add_car(spec, prefix="car/", pose=(0.0, 0.0, 0.0), tail_box=False):
    """The drive model's `add_car` stand-in: the contract's stub car attached at a planar pose under `prefix`."""
    xml = "<mujoco><worldbody>%s</worldbody></mujoco>" % "\n".join(greybox.stub_car("", [0.0, 0.0, 0.0], tail_box=tail_box))
    child = mujoco.MjSpec.from_string(xml)
    frame = spec.worldbody.add_frame(pos=[float(pose[0]), float(pose[1]), 0.0], quat=_yaw_quat(float(pose[2])))
    spec.attach(child, prefix=prefix, frame=frame)


def add_boxes(spec, world):
    """The obstacle model's `add_boxes` stand-in: the boxes as free bodies (greybox geometry)."""
    for bx in world["truth"]["avoidance"]["boxes"]:
        lx, ly, lz = bx["size_m"]
        body = spec.worldbody.add_body(name=K.box_body(bx["id"]), pos=[bx["pose"][0], bx["pose"][1], 0.5 * lz],
                                       quat=_yaw_quat(bx["pose"][2]))
        body.add_freejoint(name=K.box_body(bx["id"]) + "/root")
        size = [0.5 * lx, 0.5 * ly, 0.5 * lz]
        ct, ca = K.CONTACT["box"]
        body.add_geom(name=K.box_geom(bx["id"], "col"), type=mujoco.mjtGeom.mjGEOM_BOX, size=size, mass=bx["mass_kg"],
                      group=K.GROUP_COLLISION, contype=ct, conaffinity=ca, rgba=[0.92, 0.92, 0.9, 1])
        body.add_geom(name=K.box_geom(bx["id"], "lidar"), type=mujoco.mjtGeom.mjGEOM_BOX, size=size, mass=0.0,
                      group=K.GROUP_LIDAR, contype=0, conaffinity=0, rgba=[0.92, 0.92, 0.9, 1])


def add_slow_cars(spec, world):
    """The obstacle model's `add_slow_cars` stand-in: stub cars with a tail box at their staging poses."""
    for i, sc in enumerate(world["truth"]["avoidance"]["slow_cars"]):
        add_car(spec, K.slow_prefix(i), sc["staging_pose"], tail_box=True)


# ----------------------------------------------------------------------------------------------- the car at run time
class KinematicCar(object):
    """`CarRuntime` stand-in (see the module text). Signature of the contract."""

    def __init__(self, model, data, prefix="car/", grip_at=None, params=None):
        p = (params or load_params())["stub"]
        self.model, self.data, self.prefix, self.grip_at = model, data, prefix, grip_at
        self.tau = float(p["speed_tau_s"])
        self.a_max = float(p["accel_max_mps2"])
        self.dt = float(model.opt.timestep)
        jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "root")
        self.qa, self.va = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        sj = _id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "steer")
        self.sq, self.sv = int(model.jnt_qposadr[sj]), int(model.jnt_dofadr[sj])
        self.spin = [int(model.jnt_dofadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + w + "/spin")]) for w in K.WHEELS]
        cid = _id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "chassis")
        for b in range(model.nbody):                     # the stub car does not feel gravity (no suspension to rest on)
            if b == cid or int(model.body_rootid[b]) == cid:
                model.body_gravcomp[b] = 1.0
        q = data.qpos[self.qa:self.qa + 7]
        self.x, self.y, self.yaw = float(q[0]), float(q[1]), _quat_yaw(q[3:7])
        self.v = 0.0
        self.delta = 0.0
        self.steer_t, self.speed_t = 0.0, 0.0
        self.roll = 0.0                                  # tests may tilt the chassis (rollover scenarios)
        self.script = None                               # tests: callable(t) -> (x, y, yaw, v) drives the car exactly
        self.t_script = 0.0                              # until the next place() (the staff's hand)

    def set_targets(self, steer_rad, speed_mps):
        self.steer_t, self.speed_t = float(steer_rad), float(speed_mps)

    def pre_step(self):
        dt = self.dt
        x0, y0, yaw0 = self.x, self.y, self.yaw
        if self.script is not None:
            self.t_script += dt
            self.x, self.y, self.yaw, self.v = [float(v) for v in self.script(self.t_script)]
            dyaw = math.atan2(math.sin(self.yaw - yaw0), math.cos(self.yaw - yaw0))
            self._write(x0, y0, yaw0, (self.x - x0) / dt, (self.y - y0) / dt, dyaw / dt)
            return
        ds = max(-K.STEER_RATE_MAX_RADPS * dt, min(K.STEER_RATE_MAX_RADPS * dt, self.steer_t - self.delta))
        self.delta += ds
        a = (self.speed_t - self.v) / self.tau
        a = max(-self.a_max, min(self.a_max, a))
        self.v += a * dt
        w = self.v * math.tan(self.delta) / K.WHEELBASE_M
        if abs(w) > 1e-9:
            r = self.v / w
            self.x += r * (math.sin(yaw0 + w * dt) - math.sin(yaw0))
            self.y -= r * (math.cos(yaw0 + w * dt) - math.cos(yaw0))
        else:
            self.x += self.v * dt * math.cos(yaw0)
            self.y += self.v * dt * math.sin(yaw0)
        self.yaw = math.atan2(math.sin(yaw0 + w * dt), math.cos(yaw0 + w * dt))
        # write the pose of the start of the step and the chord velocity: mj_step then carries the free joint to the
        # kinematic pose of the end of the step (no gravity on the stub car; contacts perturb it for one step only)
        self._write(x0, y0, yaw0, (self.x - x0) / dt, (self.y - y0) / dt, w)

    def _write(self, x, y, yaw, vx, vy, w):
        d = self.data
        cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
        cr, sr = math.cos(0.5 * self.roll), math.sin(0.5 * self.roll)
        d.qpos[self.qa:self.qa + 7] = [x, y, 0.0, cy * cr, cy * sr, sy * sr, sy * cr]     # yaw, then roll about x
        d.qvel[self.va:self.va + 6] = [vx, vy, 0.0, 0.0, w * math.sin(self.roll), w * math.cos(self.roll)]
        d.qpos[self.sq] = self.delta
        d.qvel[self.sv] = 0.0
        for dof in self.spin:
            d.qvel[dof] = self.v / K.TYRE_RADIUS_M

    def wheel_speed(self):
        return self.v

    def steer_echo(self):
        return self.delta

    def place(self, pose):
        self.script = None
        self.x, self.y, self.yaw = float(pose[0]), float(pose[1]), float(pose[2])
        self.v, self.delta = 0.0, 0.0
        self.steer_t, self.speed_t = 0.0, 0.0
        self.roll = 0.0
        d = self.data
        d.qpos[self.qa:self.qa + 7] = [self.x, self.y, 0.0] + _yaw_quat(self.yaw)
        d.qvel[self.va:self.va + 6] = 0.0
        d.qpos[self.sq] = 0.0
        d.qvel[self.sv] = 0.0
        for dof in self.spin:
            d.qvel[dof] = 0.0
        mujoco.mj_forward(self.model, d)


# ----------------------------------------------------------------------------------------------- sensors
class IdealSensors(object):
    """`SensorSuite` stand-in (see the module text). `cast=False` returns no-return scans (fast tests)."""

    def __init__(self, model, data, world, car, prefix="car/", seed=None, cast=True):
        self.model, self.data, self.car, self.cast = model, data, car, cast
        self.N = int(round(K.TICK_S / float(model.opt.timestep)))
        self.sub_at = {K.subscan_step(k, self.N): k for k in range(K.SUBSCANS_PER_TICK)}
        self.imu_at = {K.imu_step(j, self.N): j for j in range(K.IMU_SAMPLES_PER_TICK)}
        self.sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + "lidar")
        self.iid = _id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + "imu")
        self.cid = _id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "chassis")
        ang = K.ANGLE_MIN_RAD + K.ANGLE_INC_RAD * np.arange(K.N_BEAMS)
        self.local = np.stack([np.cos(ang), np.sin(ang), np.zeros(K.N_BEAMS)])
        self.mask = np.array(K.LIDAR_GROUPMASK, np.uint8)
        self.dist = np.full(K.N_BEAMS, -1.0)
        self.gid = np.full(K.N_BEAMS, -1, np.int32)
        self.v6 = np.zeros(6)
        self.prev_v = None
        self.acc = np.zeros(6)
        self.nacc = 0

    def begin_tick(self):
        self.dist[:] = -1.0
        self.imu = []

    def _vel(self):
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.iid, self.v6, 0)
        return self.v6.copy()                            # [angular, linear] in the world frame

    def after_step(self, i):
        d = self.data
        if self.cast and i in self.sub_at:
            lo, hi = K.SUBSCAN_SLICES[self.sub_at[i]]
            n = hi - lo
            vec = np.ascontiguousarray((d.site_xmat[self.sid].reshape(3, 3) @ self.local[:, lo:hi]).T).reshape(-1)
            gid = np.full(n, -1, np.int32)
            dist = np.full(n, -1.0)
            mujoco.mj_multiRay(self.model, d, d.site_xpos[self.sid].copy(), vec, self.mask, 1, self.cid, gid, dist,
                               None, n, K.RANGE_MAX_M)
            self.dist[lo:hi] = dist
        v = self._vel()
        R = d.site_xmat[self.iid].reshape(3, 3)
        if self.prev_v is None:
            self.prev_v = v
        a_w = (v[3:] - self.prev_v[3:]) / float(self.model.opt.timestep)
        self.prev_v = v
        f = R.T @ (a_w + np.array([0.0, 0.0, G]))
        w = R.T @ v[:3]
        self.acc += np.concatenate([f, w])
        self.nacc += 1
        if i in self.imu_at:
            m = self.acc / max(1, self.nacc)
            self.imu.append([round(float(x), 3) for x in m[:3]] + [round(float(x), 4) for x in m[3:]])
            self.acc[:] = 0.0
            self.nacc = 0

    def end_tick(self):
        dd = self.dist
        ok = (dd >= K.RANGE_MIN_M) & (dd <= K.RANGE_MAX_M)
        scan = np.where(ok, np.rint(dd * 1000.0), K.NO_RETURN).astype(np.int64).tolist()
        return {"scan": scan, "imu": self.imu, "wheel_speed": round(float(self.car.wheel_speed()), 3),
                "steer_echo": round(float(self.car.steer_echo()), 4)}

    def reset_motion(self):
        """Forget the last velocity (after the car was placed by hand)."""
        self.prev_v = None


# ----------------------------------------------------------------------------------------------- ducts
class StaticDuctTracker(object):
    """`DuctTracker` stand-in for static ducts: nothing ever moves."""

    def __init__(self, model, data, world, seed=None, putback_err_m=None):
        self.world_poses = np.array([p for d in world["truth"]["ducts"] for p in d["pieces"]], dtype=float).reshape(-1, 3)
        self.ids = [(d["id"], k) for d in world["truth"]["ducts"] for k in range(len(d["pieces"]))]

    def update(self):
        return 0.0

    def displaced(self, threshold_m):
        return []

    def put_back(self, pieces=None):
        return None

    def poses(self):
        return self.world_poses.copy()

    def reference_poses(self):
        return self.world_poses.copy()

    def set_poses(self, poses, reference=None):
        return None

    def total_displacement(self):
        return 0.0


# ----------------------------------------------------------------------------------------------- boxes and slow cars
class StubBoxTracker(object):
    """`BoxTracker` stand-in: displacement = largest horizontal distance of a bottom corner from where it was last set
    down; put back to the world-file pose plus an error of at most staff.putback_err_m."""

    def __init__(self, model, data, world, seed=None, putback_err_m=0.03):
        self.model, self.data = model, data
        self.boxes = world["truth"]["avoidance"]["boxes"]
        self.ids = [b["id"] for b in self.boxes]
        self.err = float(putback_err_m)
        self.rng = np.random.default_rng(world["truth"]["seeds"]["putback"] if seed is None else seed)
        self.adr = []
        for b in self.boxes:
            jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, K.box_body(b["id"]) + "/root")
            self.adr.append((int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])))
        self.world = np.array([b["pose"] for b in self.boxes], dtype=float).reshape(-1, 3)
        self.ref = self.world.copy()
        self._ref_corners = [self._corners(p, b) for p, b in zip(self.ref, self.boxes)]
        self.last = np.zeros(len(self.boxes))

    @staticmethod
    def _corners(pose, box):
        x, y, yaw = pose
        lx, ly = 0.5 * box["size_m"][0], 0.5 * box["size_m"][1]
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([(x + c * a - s * b, y + s * a + c * b) for a, b in ((lx, ly), (-lx, ly), (-lx, -ly), (lx, -ly))])

    def _corners_now(self, i):
        """Floor projections of the four bottom corners of box i now."""
        qa = self.adr[i][0]
        q = self.data.qpos
        pos = np.array(q[qa:qa + 3])
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, np.array(q[qa + 3:qa + 7]))
        R = R.reshape(3, 3)
        b = self.boxes[i]
        lx, ly, lz = 0.5 * b["size_m"][0], 0.5 * b["size_m"][1], 0.5 * b["size_m"][2]
        local = np.array([(lx, ly, -lz), (-lx, ly, -lz), (-lx, -ly, -lz), (lx, -ly, -lz)])
        return (pos + local @ R.T)[:, :2]

    def pose(self, i):
        qa = self.adr[i][0]
        q = self.data.qpos
        return [float(q[qa]), float(q[qa + 1]), _quat_yaw(q[qa + 3:qa + 7])]

    def update(self):
        for i in range(len(self.boxes)):
            self.last[i] = float(np.linalg.norm(self._corners_now(i) - self._ref_corners[i], axis=1).max())
        return float(self.last.max()) if len(self.last) else 0.0

    def displaced(self, threshold_m):
        return [self.ids[i] for i in range(len(self.boxes)) if self.last[i] >= threshold_m]

    def put_back(self, boxes=None):
        todo = self.ids if boxes is None else list(boxes)
        for bid in self.ids:
            if bid not in todo:
                continue
            i = self.ids.index(bid)
            r = self.err * math.sqrt(float(self.rng.uniform(0.0, 1.0)))
            a = float(self.rng.uniform(-math.pi, math.pi))
            x, y, yaw = self.world[i]
            pose = [x + r * math.cos(a), y + r * math.sin(a), yaw]
            self._set(i, pose)
            self.ref[i] = pose
            self._ref_corners[i] = self._corners(pose, self.boxes[i])
            self.last[i] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _set(self, i, pose):
        qa, va = self.adr[i]
        self.data.qpos[qa:qa + 7] = [pose[0], pose[1], 0.5 * self.boxes[i]["size_m"][2]] + _yaw_quat(pose[2])
        self.data.qvel[va:va + 6] = 0.0

    def poses(self):
        return np.array([self.pose(i) for i in range(len(self.boxes))]).reshape(-1, 3)

    def reference_poses(self):
        return self.ref.copy()

    def set_poses(self, poses, reference=None):
        for i, p in enumerate(poses):
            self._set(i, p)
        if reference is not None:
            self.ref = np.array(reference, dtype=float).reshape(-1, 3)
            self._ref_corners = [self._corners(p, b) for p, b in zip(self.ref, self.boxes)]
        mujoco.mj_forward(self.model, self.data)

    def total_displacement(self):
        return float(sum(np.linalg.norm(np.asarray(self.pose(i)[:2]) - self.world[i, :2]) for i in range(len(self.boxes))))


class StubSlowCars(object):
    """`SlowCars` stand-in: kinematic cars on their closed paths (see the module text). It reads the ego car's
    reference point from the model (site `car/ref`), as the obstacle model's placement rule must, because `enter()` takes no
    arguments in the contract."""

    def __init__(self, model, data, world, grip_at=None, seed=None, params=None):
        p = (params or load_params())["stub"]
        self.model, self.data = model, data
        self.dt = float(model.opt.timestep)
        self.acc, self.brake = float(p["slow_car_accel_mps2"]), float(p["slow_car_brake_mps2"])
        self.entry_gap = float(p["slow_car_entry_min_gap_m"])
        self.ego_sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, K.EGO_PREFIX + "ref")
        track = check.Track(world)
        self.cars = []
        for i, sc in enumerate(world["truth"]["avoidance"]["slow_cars"]):
            pre = K.slow_prefix(i)
            jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, pre + "root")
            path = np.asarray(sc["path"]["samples"], dtype=float)
            seg = np.roll(path[:, :2], -1, axis=0) - path[:, :2]
            seglen = np.linalg.norm(seg, axis=1)
            cid = _id(model, mujoco.mjtObj.mjOBJ_BODY, pre + "chassis")
            geoms = [g for g in range(model.ngeom) if int(model.body_rootid[int(model.geom_bodyid[g])]) == cid]
            for b in range(model.nbody):
                if int(model.body_rootid[b]) == cid:
                    model.body_gravcomp[b] = 1.0
            ex, ey = track.at(sc["entry_s_m"])[:2]
            self.cars.append({"id": sc["id"], "cfg": sc, "qa": int(model.jnt_qposadr[jid]), "va": int(model.jnt_dofadr[jid]),
                              "path": path, "seg": seg, "seglen": seglen, "cum": np.concatenate([[0.0], np.cumsum(seglen)[:-1]]),
                              "L": float(seglen.sum()), "geoms": geoms, "entry_xy": (ex, ey),
                              "saved": [(int(model.geom_group[g]), int(model.geom_contype[g]), int(model.geom_conaffinity[g])) for g in geoms],
                              "on": False, "s": 0.0, "v": 0.0})
            self._stage(self.cars[-1])

    def _ego(self):
        p = self.data.site_xpos[self.ego_sid]
        return float(p[0]), float(p[1])

    def _stage(self, c):
        for g in c["geoms"]:
            self.model.geom_group[g] = K.GROUP_HIDDEN
            self.model.geom_contype[g] = 0
            self.model.geom_conaffinity[g] = 0
        x, y, yaw = c["cfg"]["staging_pose"]
        self.data.qpos[c["qa"]:c["qa"] + 7] = [x, y, 0.0] + _yaw_quat(yaw)
        self.data.qvel[c["va"]:c["va"] + 6] = 0.0
        c["on"] = False

    def _at(self, c, s):
        s = s % c["L"]
        i = int(np.searchsorted(c["cum"], s, side="right") - 1)
        u = (s - c["cum"][i]) / c["seglen"][i]
        j = (i + 1) % len(c["path"])
        p = c["path"][i, :2] + u * c["seg"][i]
        yaw = math.atan2(c["seg"][i][1], c["seg"][i][0])
        vref = (1.0 - u) * c["path"][i, 2] + u * c["path"][j, 2]
        return float(p[0]), float(p[1]), yaw, float(vref)

    def _station(self, c, x, y):
        q = np.array([x, y])
        u = np.clip(((q - c["path"][:, :2]) * c["seg"]).sum(axis=1) / c["seglen"] ** 2, 0.0, 1.0)
        dist = np.linalg.norm(c["path"][:, :2] + c["seg"] * u[:, None] - q, axis=1)
        i = int(np.argmin(dist))
        return float(c["cum"][i] + u[i] * c["seglen"][i]), float(dist[i])

    def enter(self):
        """Start of the dynamic phase: restore the geoms and put each car on its path at the point nearest to the
        centreline at entry_s_m, moved forward in 1 m steps while the ego car is nearer than the entry gap."""
        ego = self._ego()
        for c in self.cars:
            for g, (grp, ct, ca) in zip(c["geoms"], c["saved"]):
                self.model.geom_group[g], self.model.geom_contype[g], self.model.geom_conaffinity[g] = grp, ct, ca
            s, _ = self._station(c, *c["entry_xy"])
            se, de = self._station(c, *ego)
            for _ in range(int(c["L"])):
                gap = abs(((s - se) + 0.5 * c["L"]) % c["L"] - 0.5 * c["L"])
                if gap >= self.entry_gap or de > 1.5:
                    break
                s += 1.0
            c["s"], c["v"], c["on"] = s % c["L"], 0.0, True
            self._write(c, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def pre_step(self):
        ego = None
        for c in self.cars:
            if not c["on"]:
                continue
            x, y, yaw, vref = self._at(c, c["s"])
            target = c["cfg"]["speed_factor"] * vref
            if ego is None:
                ego = self._ego()
            se, de = self._station(c, *ego)
            if de < 0.5 and (se - c["s"]) % c["L"] < c["cfg"]["gap_keep_m"] + K.ENVELOPE_M[0]:
                target = 0.0
            dv = target - c["v"]
            c["v"] += max(-self.brake * self.dt, min(self.acc * self.dt, dv))
            c["s"] = (c["s"] + c["v"] * self.dt) % c["L"]
            self._write(c, c["v"])

    def _write(self, c, v):
        x, y, yaw, _ = self._at(c, c["s"])
        # the path is that of the reference point; write the pose of the car frame at the start of the step and the
        # velocity that carries it to the end
        x0, y0 = x - K.REF_POINT_C[0] * math.cos(yaw), y - K.REF_POINT_C[0] * math.sin(yaw)
        self.data.qpos[c["qa"]:c["qa"] + 7] = [x0 - v * self.dt * math.cos(yaw), y0 - v * self.dt * math.sin(yaw), 0.0] + _yaw_quat(yaw)
        self.data.qvel[c["va"]:c["va"] + 6] = [v * math.cos(yaw), v * math.sin(yaw), 0.0, 0.0, 0.0, 0.0]

    def states(self):
        out = []
        for c in self.cars:
            q = self.data.qpos[c["qa"]:c["qa"] + 7]
            out.append({"id": c["id"], "pose": [float(q[0]), float(q[1]), _quat_yaw(q[3:7])], "speed_mps": float(c["v"]),
                        "on_track": bool(c["on"])})
        return out
