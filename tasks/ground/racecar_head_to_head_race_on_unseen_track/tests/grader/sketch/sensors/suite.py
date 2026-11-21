"""`SensorSuite`: the observation of one tick in wire form.

    suite = SensorSuite(model, data, world, car)          # once per venue; seed = truth.seeds.sensors
    for every tick:
        suite.begin_tick()
        for i in 1 .. N:  car.pre_step(); ...; mujoco.mj_step(model, data); suite.after_step(i); trackers ...
        obs = suite.end_tick()                           # {"scan", "imu", "wheel_speed", "steer_echo"}

`after_step(i)` must be called after every physics step of the tick, i = 1 .. N in order (N = 0.025 s / timestep, a
multiple of 10): it casts sub-scan k after step (2k + 1) N / 10 and completes IMU sample j after step 2 (j + 1) N / 10.
A model rebuilt for another session of the same venue is attached with `rebind(model, data, car)`, which keeps the
random streams and the IMU bias (the contract: one stream per venue, drawn in simulation order). A car placed by the
staff between ticks needs nothing (begin_tick reads the state afresh); a placement between two physics steps of a tick
must be followed by `notify_placed()`, or the IMU sees the jump as an acceleration.

Random streams: numpy.random.default_rng([seed, k]) with k = 0 LiDAR,
1 IMU, 2 wheel speed, for an integer `seed` (default truth.seeds.sensors). Each draws a fixed amount per tick, so one
sensor's draws never shift another's. A numpy Generator passed as `seed` (as the cell does) is split into the three
streams by `Generator.spawn(3)`, which is deterministic in the order suites are built from it.
"""

import mujoco
import numpy as np

from sketch.examples import constants as K

from .frames import ChassisState
from .imu import Imu
from .lidar import Lidar
from .motor import MotorSpeed, SteerEcho
from .params import load_params

STREAM_LIDAR, STREAM_IMU, STREAM_WHEEL = 0, 1, 2


def steps_per_tick(model):
    """Physics steps per tick of a model: 0.025 s / timestep, which must be an integer multiple of 10."""
    n = round(K.TICK_S / float(model.opt.timestep))
    if n <= 0 or n % 10 or abs(n * float(model.opt.timestep) - K.TICK_S) > 1e-12:
        raise ValueError("timestep %r does not give a multiple of 10 physics steps per 25 ms tick" % float(model.opt.timestep))
    return n


class SensorSuite(object):

    def __init__(self, model, data, world, car=None, prefix=K.EGO_PREFIX, seed=None, params=None, floor_rule=True,
                 verify=True):
        self.world = world
        self.prefix = prefix
        self.params = params if isinstance(params, dict) and "schema" in params else load_params(params)
        if isinstance(seed, np.random.Generator):
            self.seed = None
            self.rng_lidar, self.rng_imu, self.rng_wheel = seed.spawn(3)
        else:
            self.seed = int(world["truth"]["seeds"]["sensors"] if seed is None else seed)
            self.rng_lidar = np.random.default_rng([self.seed, STREAM_LIDAR])
            self.rng_imu = np.random.default_rng([self.seed, STREAM_IMU])
            self.rng_wheel = np.random.default_rng([self.seed, STREAM_WHEEL])
        self.model, self.data, self.car = model, data, car
        self.n_steps = steps_per_tick(model)
        self.chassis = ChassisState(model, prefix)
        if verify:
            self.chassis.verify(model)
        self.lidar = Lidar(model, world, self.params, prefix=prefix, floor_rule=floor_rule)
        self.motor = MotorSpeed(model, prefix=prefix, car=car, params=self.params, rng=self.rng_wheel)
        self.steer = SteerEcho(model, prefix=prefix, car=car)
        self.imu = Imu(model, self.chassis, self.params, self.rng_imu, self.n_steps,
                       speed_fn=lambda: self.motor.true_value(self.data))
        self._schedule()
        self.i = None

    def _schedule(self):
        self.subscan_at = {K.subscan_step(k, self.n_steps): k for k in range(K.SUBSCANS_PER_TICK)}

    def rebind(self, model, data, car=None):
        """Attach a new model of the same venue (another session variant); random streams and IMU bias carry on."""
        if self.i is not None:
            raise RuntimeError("rebind inside a tick")
        self.model, self.data = model, data
        if car is not None:
            self.car = car
        self.n_steps = steps_per_tick(model)
        self.chassis = ChassisState(model, self.prefix)
        self.chassis.verify(model)
        self.lidar.bind(model, self.world)
        self.motor.bind(model, self.car)
        self.steer.bind(model, self.car)
        self.imu.chassis = self.chassis
        self.imu.steps_per_tick = self.n_steps
        self.imu.window = 2 * (self.n_steps // 10)
        self.imu.bind(model)
        self._schedule()

    def begin_tick(self):
        if self.i is not None:
            raise RuntimeError("begin_tick inside a tick (end_tick was not called)")
        self.i = 0
        self.lidar.begin()
        self.imu.begin(self.data)

    def notify_placed(self):
        """The car was placed (teleported) between two physics steps of this tick: restart the IMU's differencing."""
        if self.i is not None:
            rot = self.chassis.rotation(self.data)
            self.imu.v_prev = self.chassis.site_velocity(self.data, self.imu.pos_c, rot)

    def after_step(self, i):
        if self.i is None:
            raise RuntimeError("after_step outside a tick (begin_tick was not called)")
        if i != self.i + 1 or i > self.n_steps:
            raise RuntimeError("after_step(%d): expected %d of 1 .. %d" % (i, self.i + 1, self.n_steps))
        self.i = i
        k = self.subscan_at.get(i)
        if k is not None:
            self.lidar.cast(k, self.data)
        self.imu.after_step(i, self.data)

    def end_tick(self):
        if self.i != self.n_steps:
            raise RuntimeError("end_tick after %s of %d physics steps" % (self.i, self.n_steps))
        obs = {"scan": self.lidar.finish(self.rng_lidar), "imu": self.imu.finish(),
               "wheel_speed": self.motor.read(self.data), "steer_echo": self.steer.read(self.data)}
        self.i = None
        return obs

    def tick(self, step_fn):
        """Convenience for tests and tools: one tick with `step_fn()` doing one physics step (pre_step + mj_step)."""
        self.begin_tick()
        for i in range(1, self.n_steps + 1):
            step_fn()
            self.after_step(i)
        return self.end_tick()


def default_step(model, data, car=None):
    """A step function for `SensorSuite.tick`: car.pre_step() if there is a runtime, then mj_step."""
    def step():
        if car is not None:
            car.pre_step()
        mujoco.mj_step(model, data)
    return step
