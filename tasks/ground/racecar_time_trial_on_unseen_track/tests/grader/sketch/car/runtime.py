"""Run-time side of the drive model: `CarRuntime`.

    car = CarRuntime(model, data, prefix="car/", grip_at=floor.grip_at)
    car.place(pose)                       # at a planar pose, at rest, loops reset
    every tick:  car.set_targets(steer_rad, speed_mps)        (the referee applies the command delay)
    every physics step:  car.pre_step(); mujoco.mj_step(model, data)
    car.wheel_speed(), car.steer_echo()   # the wire's read-backs, unrounded and noise free (the sensor model rounds and adds noise)

`pre_step` does, in this order:
1. servo: the present target moves toward the steering target at no more than `steer_rate_max_radps` (this is
   `steer_echo`); the actuator `servo` follows it through a first-order lag (MuJoCo filter dynamics, `steering.lag_s`)
   and a position loop with a stall torque;
2. speed loop: PI on the motor speed (= final drive x the mean spin rate of the four wheels) -> current command,
   clipped by the controller's current limits and by what the battery voltage can drive against the back-EMF; the
   motor torque Kt I is the ctrl of the actuator `motor` (no-load loss: tendon frictionloss);
3. grip: `grip_at(x, y)` at each wheel ground point -> friction of that wheel's tyre x floor contact pairs; for the
   tyre model `slip_angle` times the fall-off factor of the wheel's lateral sliding angle, and the slip pivots'
   stiffness C_S N e from each wheel's normal load (suspension force + unsprung weight, 10 ms filter);
4. centre coupling: damping of the tendon `centre` = C_k N_series r^2 / max(|forward speed|, 0.5 m/s);
5. air drag on the chassis (`xfrc_applied` of the chassis body, which only this runtime writes).
The model arrays this runtime writes every step (its own rows only): `pair_friction` of its pairs,
`jnt_stiffness` of its slip pivots, `tendon_damping` of its `centre`, `ctrl` of its two actuators, `xfrc_applied` of
its chassis.

Determinism: no randomness, no clock; everything is a function of `model`, `data` and the calls.
"""

import math

import mujoco
import numpy as np

from sketch.examples import constants as K

from .params import load_params, params_digest


def _jid(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if i < 0:
        raise KeyError("joint %r is not in the model" % name)
    return i


def _model_text(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, name)
    if i < 0:
        return None
    a, n = int(model.text_adr[i]), int(model.text_size[i])
    return bytes(model.text_data[a:a + n - 1]).decode("ascii")


class CarRuntime(object):
    """The drive model of one car at run time. `grip_at(x, y) -> float` is the floor lookup; without
    it every wheel gets `grip` (default: `tyre.default_grip` of the parameters)."""

    def __init__(self, model, data, prefix=K.EGO_PREFIX, grip_at=None, params=None, grip=None, centre_coupling=True):
        self.model, self.data, self.prefix = model, data, prefix
        p = params or load_params()
        stored = _model_text(model, prefix + "drive_params_sha256")
        if stored is not None and stored != params_digest(p):
            raise ValueError("car %r was built from other drive parameters than the runtime's" % prefix)
        self.params = p
        self.grip_at = grip_at
        self.grip_const = float(p["tyre"]["default_grip"] if grip is None else grip)
        geo, mot, dd = p["derived_geometry"], p["derived_motor"], p["derived_drivetrain"]
        self.r = float(geo["tyre_radius_m"])
        self.gr = float(geo["final_drive_ratio"])
        self.lock = float(p["steering"]["lock_rad"])
        self.rate = float(p["steer_rate_max_radps"])
        self.v_min, self.v_max = float(p["speed_cmd_min_mps"]), float(p["speed_cmd_max_mps"])
        self.kt = float(mot["kt_nm_per_a"])
        self.ke = float(mot["ke_v_per_radps"])
        self.r_circ = float(mot["circuit_resistance_ohm"])
        self.v_batt = float(p["battery"]["voltage_v"])
        self.i_lim = float(p["controller"]["current_limit_a"])
        self.i_brake = float(p["controller"]["brake_current_limit_a"])
        self.kp = float(dd["speed_kp_a_per_radps"])
        self.ki = float(dd["speed_ki_a_per_rad"])
        rs = p["resistance"]
        self.k_drag = 0.5 * float(rs["air_density_kgpm3"]) * float(rs["drag_area_m2"])
        self.dt = float(model.opt.timestep)
        # ids and addresses
        self.chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "chassis")
        if self.chassis < 0:
            raise KeyError("car %r is not in the model" % prefix)
        j = _jid(model, prefix + "root")
        self.q_root, self.v_root = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        self.spin_dof = np.array([int(model.jnt_dofadr[_jid(model, "%s%s/spin" % (prefix, w))]) for w in K.WHEELS])
        self.a_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + "motor")
        self.a_servo = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + "servo")
        if self.a_motor < 0 or self.a_servo < 0:
            raise KeyError("car %r: actuators motor / servo missing" % prefix)
        self.act_servo = int(model.actuator_actadr[self.a_servo])
        self.t_centre = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, prefix + "centre")
        if self.t_centre < 0:
            raise KeyError("car %r: tendon centre missing" % prefix)
        self.centre_on = bool(centre_coupling)
        self.c_coef = float(dd["centre_damping_coef_nm_per_rad"])
        self.c_vlo = float(dd["centre_low_speed_mps"])
        self.wheel_body = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "%s%s/wheel" % (prefix, w))
                                    for w in K.WHEELS])
        self.wheel_c = np.array([geo["wheel_centres_c"][w][:2] for w in K.WHEELS], dtype=float)
        # every joint of the car but the root: reset by place()
        own = []
        for jj in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jj) or ""
            if name.startswith(prefix) and name != prefix + "root":
                own.append(jj)
        self.own_q = np.array([int(model.jnt_qposadr[jj]) for jj in own], dtype=int)
        self.own_v = np.array([int(model.jnt_dofadr[jj]) for jj in own], dtype=int)
        # contact pairs of each wheel (tyre x floor geoms)
        self.pairs = []
        for w in K.WHEELS:
            tag = "%s%s/grip/" % (prefix, w)
            self.pairs.append(np.array([i for i in range(model.npair)
                                        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, i) or "").startswith(tag)],
                                       dtype=int))
        if any(len(x) == 0 for x in self.pairs):
            raise ValueError("car %r has no tyre x floor contact pairs (add the floor before the car, or call "
                             "build_car.add_floor_pairs)" % prefix)
        self.pair_all = np.concatenate(self.pairs)
        self.pair_wheel = np.concatenate([np.full(len(x), k, dtype=int) for k, x in enumerate(self.pairs)])
        # tyre model
        self.slip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "fl/slip") >= 0
        sl = p["tyre"]["slip_angle"]
        self.slide_ratio = float(sl["slide_ratio"])
        self.fall = float(sl["fall_angle_rad"])
        self.v_lo = float(sl["low_speed_mps"])
        if self.slip:
            # load-proportional cornering stiffness: k_pivot = C_S x N(t) x e, N(t) from the suspension's force plus
            # the unsprung weight (the wheel's own vertical inertia is neglected)
            g = -float(model.opt.gravity[2])
            e = float(sl["pivot_offset_m"])
            self.slip_jnt = np.array([_jid(model, "%s%s/slip" % (prefix, w)) for w in K.WHEELS])
            sj = [_jid(model, "%s%s/susp" % (prefix, w)) for w in K.WHEELS]
            self.susp_q = np.array([int(model.jnt_qposadr[j]) for j in sj])
            self.susp_v = np.array([int(model.jnt_dofadr[j]) for j in sj])
            self.susp_k = np.array([float(model.jnt_stiffness[j]) for j in sj])
            self.susp_ref = np.array([float(model.qpos_spring[model.jnt_qposadr[j]]) for j in sj])
            self.susp_c = np.array([float(model.dof_damping[model.jnt_dofadr[j]]) for j in sj])
            corners = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "%s%s/corner" % (prefix, w)) for w in K.WHEELS]
            self.unsprung_w = np.array([float(model.body_subtreemass[b]) * g for b in corners])
            cs = sl["cornering_stiffness_per_rad"]
            self.k_per_n = np.array([float(cs["front" if w[0] == "f" else "rear"]) * e for w in K.WHEELS])
            n0 = p["derived_static_loads"]["normal_n"]
            self.n_floor = np.array([0.05 * float(n0[w]) for w in K.WHEELS])
            self.n_static = np.array([float(n0[w]) for w in K.WHEELS])
            self.normal_est = self.n_static.copy()
            self.n_alpha = self.dt / (float(sl["normal_filter_s"]) + self.dt)        # first-order filter per step
        rest = p["measured"].get("rest_height_m") or {}
        self.rest_z = float(rest.get("slip_angle" if self.slip else "cone") or 0.0) if isinstance(rest, dict) else float(rest)
        self.grip_now = np.full(4, self.grip_const)
        self.mu_now = np.full(4, self.grip_const)
        self.sliding_angle = np.zeros(4)
        self._fall = np.ones(4)
        self._reset_loops()

    # ------------------------------------------------------------------------------------------------ commands
    def _reset_loops(self):
        self.steer_target = 0.0
        self.speed_target = 0.0
        self.echo = 0.0
        self.integ = 0.0
        self.current = 0.0

    def set_targets(self, steer_rad, speed_mps):
        """Targets of the steering angle (rad, left positive) and of the forward speed (m/s); clipped to the car's
        command limits."""
        self.steer_target = min(max(float(steer_rad), -self.lock), self.lock)
        self.speed_target = min(max(float(speed_mps), self.v_min), self.v_max)

    # ------------------------------------------------------------------------------------------------ one step
    def motor_speed(self):
        """Motor shaft speed, rad/s (final drive x the mean spin rate of the four wheels)."""
        return self.gr * 0.25 * float(self.data.qvel[self.spin_dof].sum())

    def pre_step(self):
        d = self.data
        dt = self.dt
        # 1. servo: rate limit of the present target
        step = self.rate * dt
        self.echo += min(max(self.steer_target - self.echo, -step), step)
        d.ctrl[self.a_servo] = self.echo
        # 2. speed loop -> current -> motor torque
        w_m = self.motor_speed()
        err = self.speed_target * self.gr / self.r - w_m
        i_cmd = self.kp * err + self.integ
        back = self.ke * w_m
        i_hi = min(self.i_lim, (self.v_batt - back) / self.r_circ)
        i_lo = max(-self.i_brake, (-self.v_batt - back) / self.r_circ)
        i = min(max(i_cmd, i_lo), i_hi)
        if not ((i_cmd > i_hi and err > 0.0) or (i_cmd < i_lo and err < 0.0)):
            self.integ = min(max(self.integ + self.ki * err * dt, -self.i_lim), self.i_lim)
        self.current = i
        d.ctrl[self.a_motor] = self.kt * i
        # 3. grip under each wheel
        q = d.qpos
        qa = self.q_root
        x, y = float(q[qa]), float(q[qa + 1])
        qw, qx, qy, qz = float(q[qa + 3]), float(q[qa + 4]), float(q[qa + 5]), float(q[qa + 6])
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        c, s = math.cos(yaw), math.sin(yaw)
        if self.grip_at is not None:
            for k in range(4):
                wx, wy = self.wheel_c[k]
                self.grip_now[k] = float(self.grip_at(x + c * wx - s * wy, y + s * wx + c * wy))
        mu = self.grip_now
        if self.slip:
            mu = mu * self._fall_off()
            n = (self.susp_k * (q[self.susp_q] - self.susp_ref) + self.susp_c * d.qvel[self.susp_v] + self.unsprung_w)
            self.normal_est = self.normal_est + self.n_alpha * (np.maximum(n, self.n_floor) - self.normal_est)
            self.model.jnt_stiffness[self.slip_jnt] = self.k_per_n * self.normal_est
        self.mu_now = mu
        self.model.pair_friction[self.pair_all, :2] = mu[self.pair_wheel][:, None]
        # 4. centre coupling: damping of the tendon `centre` = C_k N_series r^2 / max(|forward speed|, v_lo)
        va = self.v_root
        vx, vy = float(d.qvel[va]), float(d.qvel[va + 1])
        v_fwd = c * vx + s * vy
        self.model.tendon_damping[self.t_centre] = (self.c_coef / max(abs(v_fwd), self.c_vlo)) if self.centre_on else 0.0
        # 5. air drag on the chassis
        sp = math.hypot(vx, vy)
        xf = d.xfrc_applied[self.chassis]
        xf[0] = -self.k_drag * sp * vx
        xf[1] = -self.k_drag * sp * vy

    def _fall_off(self):
        """Fall-off factor of the pair friction for the tyre model `slip_angle`: 1 while the contact rolls, falling to
        `slide_ratio` as the lateral sliding angle of the contact point grows to `fall_angle_rad` (smoothstep)."""
        d = self.data
        b = self.wheel_body
        c0x, c0y, c0z = d.subtree_com[self.chassis].tolist()
        cvs = d.cvel[b].tolist()                      # [omega; v at subtree_com of the root], world frame
        ps = d.xpos[b].tolist()
        xms = d.xmat[b].tolist()
        r, v_lo, fall, drop = self.r, self.v_lo, self.fall, 1.0 - self.slide_ratio
        out = self._fall
        for k in range(4):
            wx, wy, wz, vx, vy, _ = cvs[k]
            px, py, pz = ps[k]
            rx, ry, rz = px - c0x, py - c0y, pz - c0z
            cx = vx + wy * rz - wz * ry               # velocity of the wheel centre (x, y)
            cy = vy + wz * rx - wx * rz
            kx = cx - r * wy                          # of the contact point, r below the centre: + omega x (0, 0, -r)
            ky = cy + r * wx
            ax, ay = xms[k][1], xms[k][4]             # the wheel's y axis (axle), x and y components
            n = math.sqrt(ax * ax + ay * ay) + 1e-12
            ax, ay = ax / n, ay / n
            alpha = math.atan2(abs(kx * ax + ky * ay), max(abs(cx * ay - cy * ax), v_lo))
            self.sliding_angle[k] = alpha
            u = min(alpha / fall, 1.0)
            out[k] = 1.0 - drop * u * u * (3.0 - 2.0 * u)
        return out

    # ------------------------------------------------------------------------------------------------ read-backs
    def wheel_speed(self):
        """Motor speed as road speed: motor rad/s x 0.03290 / 8.3474, m/s, signed (high when the wheels spin, low
        when they lock)."""
        return self.motor_speed() * self.r / self.gr

    def steer_echo(self):
        """The servo's present target: the steering target after the rate limit, rad."""
        return self.echo

    # ------------------------------------------------------------------------------------------------ placement
    def place(self, pose):
        """Put the car at the planar pose [x, y, yaw] of its frame C, at rest on the floor (at its resting height),
        steering straight, suspension at its design position, loops and filters reset; then mj_forward."""
        m, d = self.model, self.data
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        qa, va = self.q_root, self.v_root
        d.qpos[qa:qa + 7] = [x, y, self.rest_z, math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
        d.qvel[va:va + 6] = 0.0
        d.qpos[self.own_q] = 0.0
        d.qvel[self.own_v] = 0.0
        d.act[self.act_servo] = 0.0
        d.ctrl[self.a_servo] = 0.0
        d.ctrl[self.a_motor] = 0.0
        d.xfrc_applied[self.chassis] = 0.0
        self._reset_loops()
        if self.slip:
            self.normal_est = self.n_static.copy()
            m.jnt_stiffness[self.slip_jnt] = self.k_per_n * self.normal_est
        mujoco.mj_forward(m, d)

    # ------------------------------------------------------------------------------------------------ diagnostics
    def ground_points(self):
        """Floor projections of the four wheel centres, world [x, y], order fl fr rl rr."""
        return np.array(self.data.xpos[self.wheel_body][:, :2])

    def state(self):
        """Privileged state of the car for tests and tools (never on the wire)."""
        d = self.data
        qa, va = self.q_root, self.v_root
        q = d.qpos[qa:qa + 7]
        yaw = math.atan2(2.0 * (q[3] * q[6] + q[4] * q[5]), 1.0 - 2.0 * (q[5] * q[5] + q[6] * q[6]))
        vx, vy = float(d.qvel[va]), float(d.qvel[va + 1])
        c, s = math.cos(yaw), math.sin(yaw)
        return {"pose": [float(q[0]), float(q[1]), yaw], "z": float(q[2]),
                "vel_c": [c * vx + s * vy, -s * vx + c * vy], "yaw_rate": float(d.qvel[va + 5]),
                "speed": math.hypot(vx, vy), "wheel_speed": self.wheel_speed(), "current_a": self.current,
                "grip": self.grip_now.tolist(), "mu": np.asarray(self.mu_now).tolist(),
                "sliding_angle": np.asarray(self.sliding_angle).tolist()}
