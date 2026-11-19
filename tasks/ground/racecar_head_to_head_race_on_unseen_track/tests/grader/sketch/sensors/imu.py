"""The ego car's IMU on the motor controller board.

Five samples per tick (200 Hz). Sample j is complete right after physics step 2 (j + 1) u and is the mean, over the
2u physics steps of its 5 ms, of the true specific force and angular rate at the site `<prefix>imu`, in the IMU frame
(axes of the site; parallel to C by contract), plus bias and noise:

  true specific force of step k   f_k = R_mid^T ((v_k - v_(k-1)) / h - g)   v = world velocity of the site point,
                                                                             g = gravity, R_mid = rotation of the site
                                                                             half a step back from state k
                                  (R_mid^T x = R_k^T x + (h / 2) w_k x R_k^T x to first order: the velocity
                                  difference belongs to the middle of the step; without this a car turning at
                                  5 rad/s with 8 m/s^2 of lateral force would read 0.05 m/s^2 along x)
  true angular rate of step k     w_k = the chassis angular velocity in the site frame
  sample                          mean over the window + bias + noise, clipped to the sensor range, rounded to
                                  0.001 m/s^2 and 0.0001 rad/s (the wire's precision)

Because MuJoCo integrates velocities with one acceleration per step, the window mean of (v_k - v_(k-1)) / h is exactly
the window mean of the point's acceleration; everything is computed from the free joint's qpos / qvel (frames.py), so
it is the state after each step and needs no extra MuJoCo call.

Errors: a constant bias per axis drawn once when the sensor is built (std `accel_bias_mps2`, `gyro_bias_radps`) that
then walks (`*_bias_walk_*_per_sqrt_s`); white noise per sample with std sqrt(white^2 + (vibration x |drivetrain
speed|)^2) (the vibration of the drivetrain and floor that a rigid simulation lacks). Random numbers per tick: the
bias draws at construction, then 30 normals (noise) and 30 normals (bias walk) per tick, whatever the motion.
"""

import math

import numpy as np

SAMPLE_S = 0.005


class Imu(object):

    def __init__(self, model, chassis, params, rng, steps_per_tick, speed_fn=None):
        p = params["imu"]
        self.chassis = chassis
        self.rng = rng
        self.speed_fn = speed_fn or (lambda: 0.0)
        self.white = np.array([p["accel_noise_mps2"]] * 3 + [p["gyro_noise_radps"]] * 3, dtype=float)
        self.vib = np.array([p["accel_vibration_mps2_per_mps"]] * 3 + [p["gyro_vibration_radps_per_mps"]] * 3, dtype=float)
        self.walk = math.sqrt(SAMPLE_S) * np.array([p["accel_bias_walk_mps2_per_sqrt_s"]] * 3
                                                   + [p["gyro_bias_walk_radps_per_sqrt_s"]] * 3, dtype=float)
        self.limit = np.array([p["accel_range_mps2"]] * 3 + [p["gyro_range_radps"]] * 3, dtype=float)
        self.bias = rng.standard_normal(6) * np.array([p["accel_bias_mps2"]] * 3 + [p["gyro_bias_radps"]] * 3, dtype=float)
        self.steps_per_tick = int(steps_per_tick)
        self.window = 2 * (self.steps_per_tick // 10)
        self.true = np.zeros((5, 6))
        self.speed = np.zeros(5)
        self.bind(model)

    def bind(self, model):
        self.h = float(model.opt.timestep)
        self.gravity = np.array(model.opt.gravity, dtype=float)
        _, self.pos_c, self.rot_c = self.chassis.site_offset(model, "imu")

    def begin(self, data):
        rot = self.chassis.rotation(data)
        self.v_prev = self.chassis.site_velocity(data, self.pos_c, rot)
        self.acc = np.zeros(6)
        self.n = 0
        self.done = 0

    def after_step(self, i, data):
        rot = self.chassis.rotation(data)
        v = self.chassis.site_velocity(data, self.pos_c, rot)
        a = (v - self.v_prev) / self.h
        self.v_prev = v
        rs = np.einsum("ij,jk->ik", rot, self.rot_c)            # BLAS-free
        f = np.einsum("ji,j->i", rs, a - self.gravity)
        w = np.einsum("ji,j->i", self.rot_c, self.chassis.angular_velocity_local(data))
        hh = 0.5 * self.h
        self.acc[:3] += f + hh * np.array([w[1] * f[2] - w[2] * f[1], w[2] * f[0] - w[0] * f[2], w[0] * f[1] - w[1] * f[0]])
        self.acc[3:] += w
        self.n += 1
        if i % self.window == 0:
            j = i // self.window - 1
            self.true[j] = self.acc / self.n
            self.speed[j] = abs(float(self.speed_fn()))
            self.acc[:] = 0.0
            self.n = 0
            self.done += 1

    def finish(self):
        """The wire's `imu` of this tick: 5 rows [ax, ay, az, gx, gy, gz], oldest first."""
        if self.done != 5:
            raise RuntimeError("IMU: %d of 5 samples are complete this tick" % self.done)
        noise = self.rng.standard_normal((5, 6))
        walk = self.rng.standard_normal((5, 6))
        rows = []
        for j in range(5):
            self.bias += self.walk * walk[j]
            sigma = np.sqrt(self.white ** 2 + (self.vib * self.speed[j]) ** 2)
            s = np.clip(self.true[j] + self.bias + sigma * noise[j], -self.limit, self.limit)
            rows.append([round(float(x), 3) + 0.0 for x in s[:3]] + [round(float(x), 4) + 0.0 for x in s[3:]])
        return rows
