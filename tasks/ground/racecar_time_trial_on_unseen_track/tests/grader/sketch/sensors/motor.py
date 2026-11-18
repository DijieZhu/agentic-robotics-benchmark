"""Motor speed (`wheel_speed`) and the steering echo (`steer_echo`) of the wire.

wheel_speed: the motor speed converted to road speed, motor rad/s x 0.03290 / 8.3474, m/s, signed; one number for the
shaft-driven four-wheel drive. It comes from the drivetrain, not from the ground: it reads high when the wheels spin
and low (0) when they lock. Source, in this order:
  1. `car.wheel_speed()` of the drive model's `CarRuntime` (the noise-free motor speed as road speed; the drive model owns the drivetrain);
  2. without a runtime (the contract's stub car, tests): the drivetrain kinematics of a rigid propeller shaft with two open
     differentials: motor speed / final drive = carrier speed = the mean of the four wheel spin rates, so
     wheel_speed = tyre radius x mean(spin rate of fl, fr, rl, rr), each spin joint's sign taken so that rolling
     forward is positive.
It is read at the end of the tick, gets Gaussian noise `wheel_speed.noise_mps` and is rounded
to 0.001 m/s. One normal is drawn per tick.

steer_echo: `car.steer_echo()` (the servo's present target after the command delay and the rate limit), rounded to
0.0001 rad, no noise. Without a runtime: the position of joint `<prefix>steer` (an approximation for tests only).
"""

import mujoco
import numpy as np

from sketch.examples import constants as K


class MotorSpeed(object):

    def __init__(self, model, prefix=K.EGO_PREFIX, car=None, params=None, rng=None):
        self.prefix = prefix
        self.car = car
        self.noise = float(params["wheel_speed"]["noise_mps"]) if params else 0.0
        self.rng = rng
        self.source = "car" if car is not None and callable(getattr(car, "wheel_speed", None)) else "spin_joints"
        self.bind(model)

    def bind(self, model, car=None):
        if car is not None:
            self.car = car
            self.source = "car" if callable(getattr(car, "wheel_speed", None)) else "spin_joints"
        self.dofs, self.signs = [], []
        scratch = mujoco.MjData(model)
        mujoco.mj_kinematics(model, scratch)
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.prefix + "chassis")
        y_axis = scratch.xmat[cid].reshape(3, 3)[:, 1]
        for w in K.WHEELS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "%s%s/spin" % (self.prefix, w))
            if jid < 0:
                if self.source == "spin_joints":
                    raise ValueError("car %r: joint %s/spin is missing" % (self.prefix, w))
                continue
            if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError("car %r: joint %s/spin is not a hinge" % (self.prefix, w))
            s = float(np.einsum("i,i->", scratch.xaxis[jid], y_axis))
            if abs(s) < 0.9:
                raise ValueError("car %r: joint %s/spin does not turn about the car's y axis" % (self.prefix, w))
            self.dofs.append(int(model.jnt_dofadr[jid]))
            self.signs.append(1.0 if s > 0 else -1.0)
        self.dofs = np.array(self.dofs, dtype=int)
        self.signs = np.array(self.signs)

    def spin_road_speed(self, data):
        """Tyre radius x the mean signed spin rate of the four wheels (the drivetrain's shaft speed as road speed)."""
        return float(K.TYRE_RADIUS_M * np.mean(self.signs * data.qvel[self.dofs]))

    def true_value(self, data):
        if self.source == "car":
            return float(self.car.wheel_speed())
        return self.spin_road_speed(data)

    def read(self, data):
        v = self.true_value(data)
        n = float(self.rng.standard_normal()) if self.rng is not None else 0.0
        return round(v + self.noise * n, 3) + 0.0


class SteerEcho(object):

    def __init__(self, model, prefix=K.EGO_PREFIX, car=None):
        self.prefix = prefix
        self.bind(model, car)

    def bind(self, model, car=None):
        self.car = car
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.prefix + "steer")
        self.qa = int(model.jnt_qposadr[jid]) if jid >= 0 else None
        if self.car is None and self.qa is None:
            raise ValueError("car %r: no CarRuntime and no joint 'steer'" % self.prefix)

    def read(self, data):
        if self.car is not None and callable(getattr(self.car, "steer_echo", None)):
            v = float(self.car.steer_echo())
        else:
            v = float(data.qpos[self.qa])
        return round(v, 4) + 0.0
