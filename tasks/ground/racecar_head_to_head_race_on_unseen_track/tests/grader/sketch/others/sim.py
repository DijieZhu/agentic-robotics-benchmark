"""A harness for the obstacle model's gates and tests: the `avoid` model of a world, stepped in the cell's order.

    sim = AvoidSim(world, mode="dynamic")           # venue (the venue generator's build_spec if present, else the contract's greybox) + ego car +
                                                    # boxes + slow cars, composed in one MjSpec
    sim.slow.enter(); sim.run_ticks(3600, ego_cmd=lambda sim: (steer, speed))
    sim.contacts                                    # per slow car: steps with a wall / box / ego contact, deepest one

It is not the referee (the cell owns that); it only reproduces the order of calls of one physics step:
ego car pre_step, slow cars pre_step, mj_step, mj_kinematics + mj_comPos + mj_comVel (the cell's refresh), trackers
(ducts, then boxes), then the sensors.

`pushable=True` builds the venue with pushable ducts (the venue generator's build_spec, the duct model's default active_window) and
binds a DuctTracker, as the cell does; the default stays the obstacle model's static-duct stand-in.
"""

import importlib
import math

import mujoco
import numpy as np

from sketch.car import CarRuntime, add_car
from sketch.examples import check
from sketch.examples import constants as K
from sketch.examples import greybox

from .boxes import BoxTracker, add_boxes
from .params import load_params
from .slowcars import SlowCars, add_slow_cars

CAT_NONE, CAT_DUCT, CAT_BOX, CAT_EGO, CAT_FLOOR = 0, 1, 2, 3, 4
CAT_SLOW = 10


def venue_spec(world, variant="avoid", pushable=False):
    """The venue generator's `build_spec` when the package exists, else the contract's greybox venue (static ducts). Returns (spec, source)."""
    try:
        m = importlib.import_module("sketch.venue")
    except ImportError as e:
        if getattr(e, "name", None) not in ("sketch.venue", "sketch"):
            raise
        m = None
    if m is not None and hasattr(m, "build_spec"):
        return m.build_spec(world, variant=variant, pushable=bool(pushable)), "S3"
    return mujoco.MjSpec.from_string(greybox.build_xml(world, variant="track", cars=None)), "greybox"


def grip_lookup(world):
    try:
        from sketch.car.floor import FloorGrip
        return FloorGrip(world).grip_at
    except ImportError:                                       # pragma: no cover
        return check.FloorMap(world).grip_at


def build_avoid_model(world, mode=None, ego_pose=None, ego=True, params=None, timestep=None, tyre_model=None,
                      pushable=False):
    spec, source = venue_spec(world, pushable=pushable)
    pose = world["truth"]["start_pose"]["pose"] if ego_pose is None else ego_pose
    if ego:
        add_car(spec, prefix=K.EGO_PREFIX, pose=tuple(pose), tyre_model=tyre_model)
    add_boxes(spec, world, params=params)
    add_slow_cars(spec, world, mode=mode, params=params, tyre_model=tyre_model)
    model = spec.compile()
    model.opt.timestep = float(timestep or K.PHYSICS_DT_S)
    return model, source


class AvoidSim(object):

    def __init__(self, world, mode=None, ego_pose=None, ego=True, params=None, timestep=None, refresh=True,
                 sensors=False, sensor_seed=None, tyre_model=None, monitor=True, pushable=False):
        self.world = world
        self.monitor = bool(monitor)
        self.params = load_params(params)
        self.mode = mode or self.params["slow_car_mode"]
        self.model, self.venue_source = build_avoid_model(world, self.mode, ego_pose, ego, self.params, timestep, tyre_model,
                                                          pushable=pushable)
        m = self.model
        self.data = mujoco.MjData(m)
        d = self.data
        mujoco.mj_forward(m, d)
        self.dt = float(m.opt.timestep)
        self.N = int(round(K.TICK_S / self.dt))
        self.refresh = refresh
        self.grip_at = grip_lookup(world)
        self.ego = None
        if ego:
            self.ego = CarRuntime(m, d, prefix=K.EGO_PREFIX, grip_at=self.grip_at)
            self.ego.place(world["truth"]["start_pose"]["pose"] if ego_pose is None else ego_pose)
        self.boxes = BoxTracker(m, d, world, params=self.params)
        self.ducts = None
        self.max_duct_disp = 0.0
        if pushable:
            from sketch.ducts import DuctTracker
            self.ducts = DuctTracker(m, d, world, seed=np.random.default_rng([world["truth"]["seeds"]["putback"], 4]))
        self.slow = SlowCars(m, d, world, grip_at=self.grip_at, params=self.params)
        self.sensors = None
        if sensors:
            from sketch.sensors import SensorSuite
            self.sensors = SensorSuite(m, d, world, self.ego, prefix=K.EGO_PREFIX,
                                       seed=world["truth"]["seeds"]["sensors"] if sensor_seed is None else sensor_seed)
        self._categories()
        self.steps = 0
        self.max_box_disp = 0.0
        self.contacts = [{"wall_steps": 0, "box_steps": 0, "ego_steps": 0, "other_slow_steps": 0, "deepest_m": 0.0,
                          "first_wall_t": None} for _ in self.slow.cars]
        self.ego_object_steps = 0
        self.warnings0 = self._warnings()

    def _categories(self):
        m = self.model
        cat = np.zeros(m.ngeom, dtype=np.int16)
        ego_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, K.EGO_PREFIX + "chassis")
        slow_roots = {c.body: c.i for c in self.slow.cars}
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            rule = K.geom_rule(name)
            root = int(m.body_rootid[int(m.geom_bodyid[g])])
            if rule is not None and rule[1] == "duct":
                cat[g] = CAT_DUCT
            elif rule is not None and rule[1] == "box":
                cat[g] = CAT_BOX
            elif rule is not None and rule[1] == "floor":
                cat[g] = CAT_FLOOR
            elif root in slow_roots:
                cat[g] = CAT_SLOW + slow_roots[root]
            elif root == ego_root and ego_root >= 0:
                cat[g] = CAT_EGO
        self.cat = cat
        # pairs worth a look: a slow car with anything but the floor, the ego car with a duct, a box or a slow car
        top = CAT_SLOW + max(1, len(self.slow.cars))
        tab = np.zeros((top, top), dtype=bool)
        for a in range(top):
            for b in range(top):
                if (a >= CAT_SLOW and b != CAT_FLOOR) or (b >= CAT_SLOW and a != CAT_FLOOR):
                    tab[a, b] = True
                if (a == CAT_EGO and b in (CAT_DUCT, CAT_BOX)) or (b == CAT_EGO and a in (CAT_DUCT, CAT_BOX)):
                    tab[a, b] = True
        self.pair_tab = tab

    def _warnings(self):
        return [int(self.data.warning[k].number) for k in range(len(self.data.warning))]

    def new_warnings(self):
        now = self._warnings()
        return {str(mujoco.mjtWarning(k)): now[k] - self.warnings0[k] for k in range(len(now)) if now[k] != self.warnings0[k]}

    # ------------------------------------------------------------------ stepping
    def physics_step(self):
        m, d = self.model, self.data
        if self.ego is not None:
            self.ego.pre_step()
        self.slow.pre_step()
        mujoco.mj_step(m, d)
        if self.refresh:
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_comVel(m, d)
        self.steps += 1
        if self.ducts is not None:
            self.max_duct_disp = max(self.max_duct_disp, float(self.ducts.update()))
        self.max_box_disp = max(self.max_box_disp, self.boxes.update())
        self._monitor()

    def _monitor(self):
        if not self.monitor:
            return
        d = self.data
        n = d.ncon
        if not n:
            return
        g = d.contact.geom[:n]
        ca, cb = self.cat[g[:, 0]], self.cat[g[:, 1]]
        if not self.pair_tab[ca, cb].any():
            return
        dist = d.contact.dist[:n]
        touching = dist <= 0.0
        egohit = touching & (((ca == CAT_EGO) & ((cb == CAT_DUCT) | (cb == CAT_BOX) | (cb >= CAT_SLOW)))
                             | ((cb == CAT_EGO) & ((ca == CAT_DUCT) | (ca == CAT_BOX) | (ca >= CAT_SLOW))))
        if egohit.any():
            self.ego_object_steps += 1
        for c in self.slow.cars:
            mine = CAT_SLOW + c.i
            sel = touching & ((ca == mine) | (cb == mine))
            if not sel.any():
                continue
            other = np.where(ca[sel] == mine, cb[sel], ca[sel])
            rec = self.contacts[c.i]
            if (other == CAT_DUCT).any():
                rec["wall_steps"] += 1
                if rec["first_wall_t"] is None:
                    rec["first_wall_t"] = round(self.steps * self.dt, 4)
            if (other == CAT_BOX).any():
                rec["box_steps"] += 1
            if (other == CAT_EGO).any():
                rec["ego_steps"] += 1
            if ((other >= CAT_SLOW) & (other != mine)).any():
                rec["other_slow_steps"] += 1
            nonfloor = other != CAT_FLOOR
            if nonfloor.any():
                rec["deepest_m"] = max(rec["deepest_m"], float(-dist[sel][nonfloor].min()))

    def tick(self, ego_cmd=None):
        if self.ego is not None and ego_cmd is not None:
            self.ego.set_targets(*ego_cmd(self))
        if self.sensors is not None:
            self.sensors.begin_tick()
        for i in range(1, self.N + 1):
            self.physics_step()
            if self.sensors is not None:
                self.sensors.after_step(i)
        return self.sensors.end_tick() if self.sensors is not None else None

    def run_ticks(self, n, ego_cmd=None, every=None, on_tick=None):
        for k in range(n):
            obs = self.tick(ego_cmd)
            if on_tick is not None:
                on_tick(self, k, obs)

    # ------------------------------------------------------------------ helpers
    def park_ego(self, pose=None):
        """Put the ego car outside the bounds at rest (it cannot meet the slow cars)."""
        b = self.world["truth"]["bounds"]
        self.ego.place(pose or [b["xmax"] + 3.0, b["ymax"] + 3.0, 0.0])
        self.ego.set_targets(0.0, 0.0)

    def state_digest_parts(self):
        d = self.data
        return [d.qpos.tobytes(), d.qvel.tobytes(), d.mocap_pos.tobytes(), d.mocap_quat.tobytes()]


class PathDriver(object):
    """A privileged pure-pursuit driver for the ego car along a path (rows [x, y, speed]) at `factor` x its speed
    column; for gates and tests only (it reads the true pose). With `lane_offset` the path is moved sideways."""

    def __init__(self, path_samples, factor=1.0, lookahead=(0.5, 0.25, 0.5, 1.4), speed=None):
        from .path import Path
        self.path = Path(path_samples)
        self.factor = float(factor)
        self.la = lookahead
        self.speed = speed
        self.s = None

    def __call__(self, sim):
        from .path import REF_X, wrap
        st = sim.ego.state()
        x, y, yaw = st["pose"]
        rx, ry = x + REF_X * math.cos(yaw), y + REF_X * math.sin(yaw)
        if self.s is None:
            self.s = self.path.project(rx, ry)[0]
        else:
            self.s = self.path.project(rx, ry, s_hint=self.s)[0]
        v = max(st["speed"], 0.0)
        k0, k1, lmin, lmax = self.la
        ld = min(max(k0 + k1 * v, lmin), lmax)
        tx, ty, _ = self.path.target_at_distance(x, y, (self.s - REF_X) % self.path.length, ld)
        alpha = wrap(math.atan2(ty - y, tx - x) - yaw)
        steer = math.atan2(2.0 * K.WHEELBASE_M * math.sin(alpha), ld)
        vt = self.speed(sim, self.s) if callable(self.speed) else self.factor * self.path.speed(self.s + 0.3)
        return steer, vt
