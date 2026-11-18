"""UnderwaterSim: MuJoCo model + hydro layer + thrusters + battery + current, bit-for-bit deterministic.

    vb  = build_vehicle(floor_depth=20.0)          # reference mission vehicle (digest), nominal coefficient set A
    sim = UnderwaterSim(vb, current={"type": "uniform", "v": [0.3, 0, 0]}, seed=7)   # or UnderwaterSim() = default build
    sim.reset(pos=(0, 0, -5))
    for _ in range(500):
        sim.step(u)                      # u: normalised thruster commands, shape (8,), in [-1, 1]
    st = sim.state()

Step order (one physics step dt = model.opt.timestep):
  mj_step1 -> hydro forces (fresh kinematics) -> current limiter -> thruster lag + inflow loss ->
  thruster wrench -> external wrenches -> exact residual added mass (one extra acceleration pass)
  -> mj_step2 -> battery update (current = thrusters + hotel load).
Randomness: everything stochastic draws from Generators spawned from np.random.SeedSequence(seed)
(child 0: sim.rng for callers, child 1: current field).  No global RNG, no wall clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco

from .frames import quat_to_mat, quat_mul, quat_conj, mat_to_euler
from .params import BatterySpec, ThrusterModelSpec, Water
from .dynamics.hydro import HydroLayer, read_uwkit_meta
from .dynamics.thrusters import ThrusterBank, ThrusterCurve
from .dynamics.battery import Battery
from .dynamics.current import make_current, CurrentField


@dataclass
class VehicleState:
    t: float
    pos_w: np.ndarray          # body-frame origin in W [m]
    quat_wb: np.ndarray        # (w, x, y, z), B -> W
    rpy: np.ndarray            # roll, pitch, yaw [rad]
    vel_b: np.ndarray          # velocity of the body origin over ground, body axes [m/s]
    omega_b: np.ndarray        # angular velocity, body axes [rad/s]
    depth: float               # depth of the body origin [m]
    com_w: np.ndarray
    vel_com_b: np.ndarray
    water_vel_b: np.ndarray    # current at the CoM, body axes, as seen by the last physics step [m/s]
    thrust: np.ndarray         # per-thruster force along its site +x [N]
    rpm: np.ndarray
    u_cmd: np.ndarray
    u_eff: np.ndarray
    thruster_current: np.ndarray
    battery: dict = field(default_factory=dict)
    current_scale: float = 1.0  # supply-current limiter scale applied to the commands


class UnderwaterSim:
    def __init__(self, model=None, water: Optional[Water] = None, current=None, seed: int = 0,
                 scheme: str = "pc", unsteady_current: bool = True, battery_soc0: float = 1.0,
                 current_limit: bool = True, curve: Optional[ThrusterCurve] = None):
        from .models.vehicle import VehicleBuild, build_vehicle
        if model is None:
            model = build_vehicle()
        if isinstance(model, VehicleBuild):
            model = model.compile()
        elif isinstance(model, mujoco.MjSpec):
            model = model.compile()
        elif isinstance(model, str):
            model = mujoco.MjModel.from_xml_string(model)
        self.model = model
        self.data = mujoco.MjData(model)
        self.meta = read_uwkit_meta(model)
        self.water = water or Water(rho=float(self.meta["rho_build"]))
        seq = np.random.SeedSequence(int(seed))
        ch = seq.spawn(4)
        self.seed = int(seed)
        self.rng = np.random.Generator(np.random.PCG64(ch[0]))
        self.current: CurrentField = make_current(current, rng=np.random.Generator(np.random.PCG64(ch[1])))
        if self.current.needs_rng:
            self.current.bind_rng(np.random.Generator(np.random.PCG64(ch[2])))
        self.hydro = HydroLayer(model, self.data, self.water, self.current, scheme=scheme,
                                unsteady_current=unsteady_current)
        self.hull = model.body(self.meta["vehicle_body"]).id
        self.ih = self.hydro.names.index(self.meta["vehicle_body"])
        self.hull_com_b = np.array(self.hydro.params[self.ih].rigid_com)
        names = self.meta["thrusters"]
        sids = [model.site(n).id for n in names]
        for s in sids:
            if model.site_bodyid[s] != self.hull:
                raise ValueError("thruster sites must be on the hull body")
        self.thr_names = names
        self.thr_pos = np.array([model.site_pos[s] for s in sids])
        self.thr_dir = np.array([quat_to_mat(model.site_quat[s])[:, 0] for s in sids])
        self.thr_arm = self.thr_pos - self.hull_com_b
        self._thr_wcoef = np.cross(self.thr_arm, self.thr_dir)          # (n, 3): (r x d)
        self._B = np.vstack([self.thr_dir.T, self._thr_wcoef.T])        # 6 x n wrench about the CoM
        self.thrusters = ThrusterBank(len(names), ThrusterModelSpec(**self.meta["thruster_model"]), curve)
        bs = dict(self.meta["battery"])
        for k in ("ocv_soc", "ocv_cell_V"):
            bs[k] = tuple(bs[k])
        self.battery = Battery(BatterySpec(**bs), soc0=battery_soc0)
        self.hotel = {"base": float(self.meta["hotel_W"]["base"])}
        self._hotel_W = float(sum(self.hotel.values()))
        self.current_limit = bool(current_limit)
        self.current_scale = 1.0
        self.u = np.zeros(len(names))
        self._ext = []
        self._ext_b = {}
        self._soc0 = battery_soc0
        self.free_jnt = int(model.body_jntadr[self.hull])
        self.qadr = int(model.jnt_qposadr[self.free_jnt])
        self.vadr = int(model.jnt_dofadr[self.free_jnt])
        self._act = {model.actuator(i).name: i for i in range(model.nu)}
        # servo targets slewed by the firmware (camera tilt, gripper): (actuator id, target, rate [unit/s])
        self._slew = {}
        g = self.meta.get("gripper") or {}
        if g.get("type") == "hinge_pair":
            self._slew["gripper"] = [self._act["gripper"], 0.0, g["theta_open_rad"] / g["close_open_s"]]
        ct = self.meta.get("camera_tilt") or {}
        if "cam_tilt" in self._act:
            self._slew["cam_tilt"] = [self._act["cam_tilt"], 0.0, float(ct.get("rate_rad_s", 1.0472))]
        self._step_callbacks = []
        self.lights = None                   # lamp levels 0..1 per lamp once set_lights() was called
        mujoco.mj_forward(model, self.data)

    # ------------------------------------------------------------------ setup
    @property
    def dt(self):
        return float(self.model.opt.timestep)

    @property
    def time(self):
        return float(self.data.time)

    def reset(self, pos=None, quat=None, vel_w=None, omega_b=None, soc=None):
        m, d = self.model, self.data
        for obj in list(getattr(self, "_grasp_saved", {})):     # undo grasp collision masks
            self.detach(obj)
        mujoco.mj_resetData(m, d)
        if pos is not None:
            d.qpos[self.qadr:self.qadr + 3] = pos
        if quat is not None:
            q = np.asarray(quat, float)
            d.qpos[self.qadr + 3:self.qadr + 7] = q / np.linalg.norm(q)
        if vel_w is not None:
            d.qvel[self.vadr:self.vadr + 3] = vel_w
        if omega_b is not None:
            d.qvel[self.vadr + 3:self.vadr + 6] = omega_b
        self.thrusters.reset()
        self.battery.reset(self._soc0 if soc is None else soc)
        self.u[:] = 0.0
        self.current_scale = 1.0
        for sl in self._slew.values():
            sl[1] = 0.0
        self.hydro._vc_prev = None
        self.hydro.last = {}
        for i in self.hydro.full:
            self.hydro._acc_est[i] = np.zeros(3)
            self.hydro._vr_prev[i] = None
        mujoco.mj_forward(m, d)

    def set_body_pose(self, body, pos, quat=None):
        """Place a free body (e.g. an object) and refresh kinematics."""
        b = self.model.body(body).id
        j = self.model.body_jntadr[b]
        a = self.model.jnt_qposadr[j]
        self.data.qpos[a:a + 3] = pos
        if quat is not None:
            self.data.qpos[a + 3:a + 7] = quat
        v = self.model.jnt_dofadr[j]
        self.data.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_hotel(self, name, watts):
        self.hotel[name] = float(watts)
        self._hotel_W = float(sum(self.hotel.values()))

    def set_lights(self, levels, n_lamps: int = 4):
        """Lamp levels 0..1 (one value for all lamps, or one per lamp).  Stores them in `self.lights` (read by the
        visual-twin recorder) and sets the hotel load 'lights' = sum(levels) x the lamp's full power
        (meta hotel_W 'light_each_full', 15 W).  The camera model's lamp level is separate (UnderwaterCamera
        .set_lamp_level; the reference vehicle has one common dimming channel)."""
        lv = np.clip(np.asarray(levels, float).reshape(-1), 0.0, 1.0)
        if lv.size == 1:
            lv = np.full(int(n_lamps), float(lv[0]))
        self.lights = lv.copy()
        each = float((self.meta.get("hotel_W") or {}).get("light_each_full", 15.0))
        self.set_hotel("lights", each * float(lv.sum()))
        return self.lights

    def add_step_callback(self, fn):
        """fn(sim) is called after every physics step (also inside step(n > 1)); it must not change the physics
        state if bit-identical trajectories with / without it are wanted (a recorder, a logger).  Returns fn."""
        self._step_callbacks.append(fn)
        return fn

    def remove_step_callback(self, fn):
        self._step_callbacks.remove(fn)

    def _i_bound(self, V):
        return self.thrusters.curve.current_slope_bound(V)

    def set_thrusters(self, u):
        self.u = np.clip(np.asarray(u, float).reshape(-1), -1.0, 1.0)

    def set_gripper(self, opening, instant=False):
        """Command the gripper: opening in [0, 1] (0 = closed, 1 = fully open).  Reference vehicle: the servo target
        moves at the official open / close speed (full travel in close_open_s); instant=True jumps (tests)."""
        g = self.meta.get("gripper")
        if not g:
            raise ValueError("vehicle has no gripper")
        x = float(np.clip(opening, 0, 1))
        if g.get("type") == "hinge_pair":
            sl = self._slew["gripper"]
            sl[1] = x * g["theta_open_rad"]
            if instant:
                self.data.ctrl[sl[0]] = sl[1]
            return
        q = x * g["open_half_gap_m"]
        for n in ("grip_l", "grip_r"):
            self.data.ctrl[self._act[n]] = q

    def gripper_opening(self):
        """Distance between the jaw tips [m] (reference vehicle), from the jaw tip sites."""
        m, d = self.model, self.data
        a, b = m.site("jaw_upper_tip").id, m.site("jaw_lower_tip").id
        return float(np.linalg.norm(d.site_xpos[a] - d.site_xpos[b]))

    def set_camera_tilt(self, angle, instant=False):
        """Camera tilt [rad], optical axis UP positive (decision U8), slewed at the servo rate."""
        sl = self._slew["cam_tilt"]
        lo, hi = self.model.actuator_ctrlrange[sl[0]]
        sl[1] = float(np.clip(angle, lo, hi))
        if instant:
            self.data.ctrl[sl[0]] = sl[1]

    def camera_tilt(self):
        """Measured tilt [rad] (up positive)."""
        j = self.model.joint("cam_tilt").id
        return float(self.data.qpos[self.model.jnt_qposadr[j]])

    def apply_wrench(self, body, force_w, torque_w=(0.0, 0.0, 0.0), point_b=None):
        """Persistent external wrench (world axes) on a body; point_b = application point in the body
        frame (default: CoM).  Cleared by clear_wrenches()."""
        self._ext.append((self.model.body(body).id, np.asarray(force_w, float), np.asarray(torque_w, float),
                          None if point_b is None else np.asarray(point_b, float)))

    def set_body_wrench(self, body, wrench_b):
        """Persistent wrench [Fx Fy Fz Mx My Mz] in the body's own axes, applied at its CoM (rotates
        with the body; e.g. a tether pull or a test force).  Pass None to remove it."""
        b = self.model.body(body).id
        if wrench_b is None:
            self._ext_b.pop(b, None)
        else:
            self._ext_b[b] = np.asarray(wrench_b, float).reshape(6).copy()

    def clear_wrenches(self):
        self._ext = []
        self._ext_b = {}

    # ------------------------------------------------------------------ grasp (weld)
    def attach(self, obj, to=None):
        """Rigidly attach a free object to the hull at its current relative pose (weld equality).
        While attached, the object's geoms stop colliding with the vehicle (they keep colliding with
        the environment and other objects)."""
        from .models.vehicle import GRASPED_CONTYPE, GRASPED_CONAFFINITY
        m, d = self.model, self.data
        eq = m.equality(f"grasp_{obj}").id
        b1, b2 = m.eq_obj1id[eq], m.eq_obj2id[eq]
        mujoco.mj_kinematics(m, d)
        R1 = d.xmat[b1].reshape(3, 3)
        rel_p = R1.T @ (d.xpos[b2] - d.xpos[b1])
        rel_q = quat_mul(quat_conj(d.xquat[b1]), d.xquat[b2])
        m.eq_data[eq, 0:3] = 0.0
        m.eq_data[eq, 3:6] = rel_p
        m.eq_data[eq, 6:10] = rel_q
        m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1
        b = m.body(obj).id
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        self._grasp_saved = getattr(self, "_grasp_saved", {})
        self._grasp_saved[obj] = [(g, int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in geoms]
        for g in geoms:
            m.geom_contype[g] = GRASPED_CONTYPE
            m.geom_conaffinity[g] = GRASPED_CONAFFINITY

    def detach(self, obj):
        m = self.model
        self.data.eq_active[m.equality(f"grasp_{obj}").id] = 0
        for g, ct, ca in getattr(self, "_grasp_saved", {}).pop(obj, []):
            m.geom_contype[g] = ct
            m.geom_conaffinity[g] = ca

    # ------------------------------------------------------------------ stepping
    def _limit(self, u, V):
        """Supply-current limiter (firmware-style): scale all commands so that the steady current of
        thrusters + hotel load stays within the battery's allowed current."""
        I_allow = self.battery.allowed_current()
        I_hotel = self._hotel_W / max(V, 1.0)
        # cheap exact bound: steady current <= I_max(V) * |u| on this curve (checked in the tests)
        if self._i_bound(V) * float(np.abs(u).sum()) + I_hotel <= I_allow:
            self.current_scale = 1.0
            return u
        I_req = float(self.thrusters.steady_current(u, V).sum()) + I_hotel
        if I_req <= I_allow:
            self.current_scale = 1.0
            return u
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if float(self.thrusters.steady_current(mid * u, V).sum()) + I_hotel <= I_allow:
                lo = mid
            else:
                hi = mid
        self.current_scale = lo
        return lo * u

    def step(self, u=None, n=1):
        if u is not None:
            self.set_thrusters(u)
        cbs = getattr(self, "_step_callbacks", ())
        for _ in range(int(n)):
            self._step_once()
            for fn in cbs:
                fn(self)

    def _step_once(self):
        m, d = self.model, self.data
        dt = m.opt.timestep
        for aid, target, rate in self._slew.values():
            c = d.ctrl[aid]
            if c != target:
                step = rate * dt
                d.ctrl[aid] = c + min(max(target - c, -step), step)
        mujoco.mj_step1(m, d)
        d.xfrc_applied[:] = 0.0
        nur = self.hydro.apply(dt, d.time)
        V = self.battery.voltage
        u = self._limit(self.u, V) if self.current_limit else self.u
        vr, w = nur[self.ih, :3], nur[self.ih, 3:]
        u_ax = self.thr_dir @ vr + self._thr_wcoef @ w     # d . (v_r + w x r) = d.v_r + (r x d).w
        T = self.thrusters.update(u, V, dt, u_ax)
        R = self.hydro.last["R"][self.ih]
        wrench = self._B @ T
        d.xfrc_applied[self.hull, :3] += R @ wrench[:3]
        d.xfrc_applied[self.hull, 3:] += R @ wrench[3:]
        for b, F, M, pb in self._ext:
            d.xfrc_applied[b, :3] += F
            tq = M
            if pb is not None:
                Rb = d.xmat[b].reshape(3, 3)
                tq = M + np.cross(d.xpos[b] + Rb @ pb - d.xipos[b], F)
            d.xfrc_applied[b, 3:] += tq
        for b, wb in self._ext_b.items():
            Rb = d.xmat[b].reshape(3, 3)
            d.xfrc_applied[b, :3] += Rb @ wb[:3]
            d.xfrc_applied[b, 3:] += Rb @ wb[3:]
        self.hydro.correct()
        mujoco.mj_step2(m, d)
        I = float(self.thrusters.current.sum()) + self._hotel_W / max(V, 1.0)
        self.battery.step(I, dt)

    # ------------------------------------------------------------------ readout
    def state(self) -> VehicleState:
        d = self.data
        p = d.qpos[self.qadr:self.qadr + 3].copy()
        q = d.qpos[self.qadr + 3:self.qadr + 7].copy()
        R = quat_to_mat(q)
        v_o_w = d.qvel[self.vadr:self.vadr + 3]
        w_b = d.qvel[self.vadr + 3:self.vadr + 6].copy()
        v_b = R.T @ v_o_w
        com = p + R @ self.hull_com_b
        v_com_b = v_b + np.cross(w_b, self.hull_com_b)
        L = self.hydro.last
        vc_b = L["vc_b"][self.ih].copy() if L else R.T @ self.current(com, d.time)[0]
        return VehicleState(t=float(d.time), pos_w=p, quat_wb=q, rpy=mat_to_euler(R), vel_b=v_b,
                            omega_b=w_b, depth=float(-p[2]), com_w=com, vel_com_b=v_com_b,
                            water_vel_b=vc_b, thrust=self.thrusters.thrust.copy(),
                            rpm=self.thrusters.rpm.copy(), u_cmd=self.thrusters.u_cmd.copy(),
                            u_eff=self.thrusters.u_eff.copy(), thruster_current=self.thrusters.current.copy(),
                            battery=self.battery.status(), current_scale=self.current_scale)

    def body_pose(self, body):
        b = self.model.body(body).id
        j = self.model.body_jntadr[b]
        if j >= 0 and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = self.model.jnt_qposadr[j]
            return self.data.qpos[a:a + 3].copy(), self.data.qpos[a + 3:a + 7].copy()
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[b].copy(), self.data.xquat[b].copy()

    def energy(self):
        """Mechanical energy at the current state: KE (incl. added mass) + restoring potential +
        the elastic energy 0.5 kp (q - ctrl)^2 of position servos (gripper, camera tilt)."""
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        E = self.hydro.energy()
        for i in range(m.nu):
            if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE:
                kp = -float(m.actuator_biasprm[i, 1])
                q = float(d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]])
                E += 0.5 * kp * (q - float(d.ctrl[i])) ** 2
        return E

    def truth_nav(self, rng: Optional[np.random.Generator] = None, noise: Optional[dict] = None):
        """Navigation estimate built from the true state (+ optional white noise) - FOR TESTS ONLY.
        Real tasks feed the autopilot from the sensor / nav lane (uwkit.nav)."""
        from .control.autopilot import NavEstimate
        st = self.state()
        q, w, depth, v, p = st.quat_wb.copy(), st.omega_b.copy(), st.depth, st.vel_b.copy(), st.pos_w.copy()
        if rng is not None and noise:
            if noise.get("att"):
                dq = np.concatenate([[1.0], 0.5 * rng.normal(0, noise["att"], 3)])
                q = quat_mul(q, dq / np.linalg.norm(dq))
            w = w + rng.normal(0, noise.get("gyro", 0.0), 3)
            depth = depth + rng.normal(0, noise.get("depth", 0.0))
            v = v + rng.normal(0, noise.get("vel", 0.0), 3)
            p = p + rng.normal(0, noise.get("pos", 0.0), 3)
        return NavEstimate(t=st.t, quat_wb=q, omega_b=w, depth=float(depth), vel_b=v, pos_w=p)
