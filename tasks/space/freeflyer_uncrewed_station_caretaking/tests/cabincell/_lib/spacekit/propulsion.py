"""Fan propulsion of the reference cabin free-flyer: plant model and force allocation, ported from the
reference firmware (plant = its simulation model of the two propulsion modules; allocator = its force
allocation module), plus a nonnegative-least-squares allocator as an alternative.

Plant, per propulsion module (``FanModule``, stepped at the firmware rate 62.5 Hz):
  * impeller: commanded speed level -> rate limit (200 rpm/s) -> PID -> DC motor (back-EMF, resistance,
    viscous friction) -> speed;
  * six nozzles, each with two gear-coupled flaps driven by a servo (PID on the output angle, DC motor,
    gearbox, backlash box); open area A = (h - L cos(theta_min + theta)) * 2 * w;
  * plenum: dp = omega^2 D^2 rho c(A_tot) with c() the firmware's 334-point table of the total open
    area A_tot = leak + sum Cd_i A_i;
  * nozzle thrust u_i = 2 dp Cd_i^2 A_i * k, k = 1.25 (the firmware plant's thrust_error_scale);
  * body loads: force -sum u_i e_i (e_i = exhaust direction), torque of the nozzle forces, impeller motor
    reaction and the gyroscopic term of the impeller momentum.
  * impeller inertia = the firmware's 0.001 kg m^2 in the speed response, the reaction / gyroscopic
    torques and the coast after a power cut; the part-level 2.0e-4 kg m^2 is only recorded in the values file.
Power cut (``FanModule.set_power(False)``, v0.3; not part of the firmware plant, which has no unpowered state): the
unpowered fan drive leaves the motor open-circuit (no current, no braking), the impeller coasts against its viscous
friction and -- ``coast_aero_load`` -- the air it still pumps (dp Q / (eta omega)); the unpowered nozzle servos stop
driving and the flaps stay where they were, so the open nozzles keep producing thrust while the fan runs down
(``coast_down``).
Allocator (``FlightAllocator``): pseudo-inverse of the 6 x 12 nozzle matrix about the configured CoM,
negative thrusts removed by adding the most negative value of each 4-nozzle null group to the group,
plenum pressure from the 316-point thrust table, area -> flap angle -> servo PWM (0..255), impeller
command = round(0.792095 * level speed).  Saturation handling = only what the firmware does: each flap
angle is clamped to its range, nothing is rescaled (see ``NNLSAllocator`` for a direction-preserving
alternative).
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._fast import ScalarTable, cross, cross_rows
from .allocation import nnls

__all__ = [
    "load_firmware_reference", "PropulsionParams", "table_lookup", "nozzle_area", "flap_angle_for_area",
    "plenum_pressure", "FanModule", "FanPropulsion", "FlightAllocator", "NNLSAllocator", "nozzle_matrix",
    "static_thrust", "PropulsionWrench", "impeller_power", "intake_flow", "max_along", "wrench_envelope", "coast_down",
]

_DATA_DIR = Path(__file__).resolve().parent / "data"


def load_firmware_reference(path: str | Path | None = None) -> dict:
    """Return (a deep copy of) the firmware reference constants (data/firmware_reference.json)."""
    p = Path(path) if path is not None else _DATA_DIR / "firmware_reference.json"
    return copy.deepcopy(json.loads(p.read_text()))


@dataclass
class PropulsionParams:
    """Propulsion constants (plant or firmware copy).  Arrays are float64, body frame, SI units."""

    control_period: float
    nozzle_position: np.ndarray          # (12, 3) m, modules 0 (nozzles 0-5) and 1 (6-11)
    nozzle_exhaust_dir: np.ndarray       # (12, 3) unit; air leaves along it, force is opposite
    nozzle_width: np.ndarray             # (6,) m, same nozzle widths on both modules
    discharge_coeff: np.ndarray          # (2, 6)
    leak_area: np.ndarray                # (2,) m^2 effective open area with all flaps closed
    flaps_per_nozzle: int
    flap_max_open: float                 # rad, theoretical max flap angle
    flap_min_open: float                 # rad, closed flap angle
    flap_length: float                   # m
    intake_height: float                 # m
    flap_gear_ratio: float
    servo_k: float
    servo_r: float
    servo_friction: float
    servo_inertia: float
    servo_gear_ratio: float
    servo_pid: tuple
    servo_backlash_half_width: float
    servo_pwm_max: int
    imp_speed_k: float
    imp_r: float
    imp_torque_k: float
    imp_friction: float
    imp_inertia: float
    imp_diameter: float
    imp_cmd_per_rad_s: float
    imp_levels: np.ndarray               # (3,) rad/s for speed levels 1, 2, 3
    imp_ramp: float                      # rad/s^2
    imp_pid: tuple
    imp_axis: np.ndarray                 # (2, 3)
    battery_voltage: float
    air_density: float
    thrust_error_scale: float
    area_table_x: np.ndarray
    area_table_y: np.ndarray
    thrust_table_x: np.ndarray
    thrust_table_y: np.ndarray
    null_groups: tuple = field(default=((0, 1, 6, 7), (2, 3, 8, 9), (4, 5, 10, 11)))

    @classmethod
    def from_dict(cls, p: dict) -> "PropulsionParams":
        """Build from the 'propulsion' block of the firmware reference JSON."""
        s, i = p["servo"], p["impeller"]
        return cls(
            control_period=float(p["control_period_s"]),
            nozzle_position=np.asarray(p["nozzle_position_m"], float),
            nozzle_exhaust_dir=np.asarray(p["nozzle_exhaust_dir"], float),
            nozzle_width=np.asarray(p["nozzle_width_m"], float),
            discharge_coeff=np.asarray(p["discharge_coeff"], float),
            leak_area=np.asarray(p["leak_area_m2"], float),
            flaps_per_nozzle=int(p["flaps_per_nozzle"]),
            flap_max_open=float(p["flap_max_open_angle_rad"]),
            flap_min_open=float(p["flap_min_open_angle_rad"]),
            flap_length=float(p["flap_length_m"]),
            intake_height=float(p["intake_height_m"]),
            flap_gear_ratio=float(p["flap_gear_ratio"]),
            servo_k=float(s["motor_k_nm_per_a"]), servo_r=float(s["motor_r_ohm"]),
            servo_friction=float(s["viscous_friction_nms"]), servo_inertia=float(s["gearbox_inertia_kgm2"]),
            servo_gear_ratio=float(s["gear_ratio"]), servo_pid=tuple(float(v) for v in s["pid_kp_ki_kd_max_min"]),
            servo_backlash_half_width=float(s["backlash_half_width_rad"]), servo_pwm_max=int(s["pwm_max"]),
            imp_speed_k=float(i["speed_k_rad_s_per_v"]), imp_r=float(i["motor_r_ohm"]),
            imp_torque_k=float(i["torque_k_nm_per_a"]), imp_friction=float(i["viscous_friction_nms"]),
            imp_inertia=float(i["inertia_kgm2"]), imp_diameter=float(i["diameter_m"]),
            imp_cmd_per_rad_s=float(i["cmd_per_rad_s"]), imp_levels=np.asarray(i["levels_rad_s"], float),
            imp_ramp=float(i["ramp_rad_s2"]), imp_pid=tuple(float(v) for v in i["pid_kp_ki_kd_max_min"]),
            imp_axis=np.asarray(i["axis"], float), battery_voltage=float(i["sim_battery_voltage_v"]),
            air_density=float(p["air_density_kg_m3"]), thrust_error_scale=float(p["thrust_error_scale"]),
            area_table_x=np.asarray(p["plenum_area_table"]["total_area_m2"], float),
            area_table_y=np.asarray(p["plenum_area_table"]["pressure_coeff"], float),
            thrust_table_x=np.asarray(p["allocator_thrust_table"]["thrust_per_speed2"], float),
            thrust_table_y=np.asarray(p["allocator_thrust_table"]["pressure_coeff"], float),
            null_groups=tuple(tuple(g) for g in p["allocator_null_groups"]),
        )

    @classmethod
    def reference(cls) -> "PropulsionParams":
        """The reference free-flyer's firmware values."""
        return cls.from_dict(load_firmware_reference()["propulsion"])

    def copy(self) -> "PropulsionParams":
        """Deep copy (arrays copied)."""
        return copy.deepcopy(self)

    def area_table(self) -> ScalarTable:
        """Scalar (fast) version of the plenum area table (cached per table array; replace, do not edit in place)."""
        key = (id(self.area_table_x), id(self.area_table_y))
        c = self.__dict__.get("_area_tab")
        if c is None or c[0] != key:
            c = (key, ScalarTable(self.area_table_x, self.area_table_y))
            self.__dict__["_area_tab"] = c
        return c[1]

    def thrust_table(self) -> ScalarTable:
        """Scalar (fast) version of the allocator thrust table."""
        key = (id(self.thrust_table_x), id(self.thrust_table_y))
        c = self.__dict__.get("_thrust_tab")
        if c is None or c[0] != key:
            c = (key, ScalarTable(self.thrust_table_x, self.thrust_table_y))
            self.__dict__["_thrust_tab"] = c
        return c[1]

    @property
    def max_flap_travel(self) -> float:
        """Flap travel from closed to theoretical max [rad] (64.2 deg)."""
        return self.flap_max_open - self.flap_min_open

    def width_of(self, k: int) -> float:
        """Width [m] of nozzle k (0..11)."""
        return float(self.nozzle_width[k % 6])

    def max_area(self) -> np.ndarray:
        """(12,) open area of each nozzle at full flap travel [m^2]."""
        w = np.concatenate([self.nozzle_width, self.nozzle_width])
        return nozzle_area(np.full(12, self.max_flap_travel), w, self)


def table_lookup(breakpoints, values, x):
    """Firmware table lookup: clamp outside [bp0, bpN], else binary search + linear interpolation.

    Vectorised; identical to the firmware for every x except x == bp[-1] exactly, where the firmware reads
    one element past the table (undefined) and this returns values[-1].
    """
    bp = np.asarray(breakpoints, float)
    y = np.asarray(values, float)
    x = np.asarray(x, float)
    n = len(bp)
    k = np.clip(np.searchsorted(bp, x, side="right") - 1, 0, n - 2)
    r = (x - bp[k]) / (bp[k + 1] - bp[k])
    out = (1.0 - r) * y[k] + r * y[k + 1]
    out = np.where(x < bp[0], y[0], out)
    out = np.where(x > bp[-1], y[-1], out)
    return out if out.ndim else float(out)


def nozzle_area(theta, width, p: PropulsionParams):
    """Open area [m^2] of nozzles with flap angle theta (beyond the closed angle) and width w."""
    th = np.asarray(theta, float)
    return (p.intake_height - np.cos(p.flap_min_open + th) * p.flap_length) * p.flaps_per_nozzle * np.asarray(width, float)


def flap_angle_for_area(area, width, p: PropulsionParams):
    """Inverse of nozzle_area with the firmware's clamping to [0, max travel] (NaN -> fully open)."""
    area = np.asarray(area, float)
    op = np.full(np.broadcast_shapes(area.shape, np.shape(width)), np.nan)
    np.divide(p.intake_height - (1.0 / p.flaps_per_nozzle) * np.divide(area, width, out=np.full_like(op, np.inf),
                                                                       where=np.isfinite(area)),
              p.flap_length, out=op)
    lo, hi = math.cos(p.flap_max_open), math.cos(p.flap_min_open)
    op = np.where(np.isnan(op) | (op < lo), lo, op)
    op = np.where(op > hi, hi, op)
    return np.arccos(op) - p.flap_min_open


def plenum_pressure(omega, total_area, p: PropulsionParams):
    """Plenum overpressure [Pa] at impeller speed omega [rad/s] and total effective open area [m^2]."""
    if np.ndim(total_area) == 0 and np.ndim(omega) == 0:
        return float(omega) ** 2 * p.area_table()(float(total_area)) * p.imp_diameter ** 2 * p.air_density
    c = table_lookup(p.area_table_x, p.area_table_y, total_area)
    return np.asarray(omega, float) ** 2 * c * p.imp_diameter ** 2 * p.air_density


def intake_flow(omega, total_area, p: PropulsionParams):
    """Volume flow [m^3/s] through a module (= intake flow) from the orifice relation Q = A_tot sqrt(2 dp / rho)."""
    dp = plenum_pressure(omega, total_area, p)
    return np.asarray(total_area, float) * np.sqrt(2.0 * np.maximum(dp, 0.0) / p.air_density)


class _FlightPID:
    """The firmware plant's PID (derivative through a first-order state, as in its source)."""

    def __init__(self, kp, ki, kd, omax, omin, dt):
        self.kp, self.ki, self.kd, self.omax, self.omin, self.dt = kp, ki, kd, omax, omin, dt
        self.last = 0.0
        self.integral = 0.0

    def run(self, x):
        d = self.kd * x - self.last
        out = self.kp * x + self.ki * self.integral + d
        self.last += d * self.dt
        self.integral += x * self.dt
        return min(max(out, self.omin), self.omax)


class FanModule:
    """One propulsion module (impeller + plenum + 6 nozzles), a step-for-step port of the firmware plant.

    index: 0 (+y side, nozzles 0-5) or 1 (-y side, nozzles 6-11).  com: point about which the nozzle
    torques are taken (the firmware plant uses the body origin, com = 0).  gyro: 'physical' applies
    -omega x h (rigid body with an internal rotor) or 'firmware' applies +omega x h as the firmware plant
    does; the two impellers counter-rotate, so the sum is zero in both conventions when their speeds are
    equal.  aero_load: add the aerodynamic load torque on the impeller (False = firmware plant, which sets
    it to zero); fan_efficiency is used only when aero_load is True.
    """

    def __init__(self, p: PropulsionParams, index: int, com=(0.0, 0.0, 0.0), gyro: str = "physical",
                 thrust_scale: float | None = None, aero_load: bool = False, fan_efficiency: float = 0.52,
                 coast_aero_load: bool = True):
        self.p = p
        self.powered = True                  # propulsion channel power (fan drive, servos, module board)
        self.coast_aero_load = bool(coast_aero_load)
        self.index = int(index)
        sl = slice(6 * index, 6 * index + 6)
        self.offsets = p.nozzle_position[sl].copy()
        self.orient = p.nozzle_exhaust_dir[sl].copy()
        self.width = p.nozzle_width.copy()
        self.cd = p.discharge_coeff[index].copy()
        self.leak = float(p.leak_area[index])
        self.axis = p.imp_axis[index].copy()
        self.thrust_scale = p.thrust_error_scale if thrust_scale is None else float(thrust_scale)
        if gyro not in ("physical", "firmware"):
            raise ValueError(gyro)
        self.gyro_sign = -1.0 if gyro == "physical" else 1.0
        self.aero_load = aero_load
        self.fan_efficiency = fan_efficiency
        dt = p.control_period
        self.servo_pids = [_FlightPID(*p.servo_pid, dt) for _ in range(6)]
        self.speed_pid = _FlightPID(*p.imp_pid, dt)
        self.prev_omega = np.zeros(6)
        self.servo_thetas = np.zeros(6)
        self.backlash = np.zeros(6)
        self.prev_speed_rate = 0.0
        self.impeller_speed = 0.0
        self.last_speed = 0.0
        self.impeller_current = 0.0
        self.impeller_voltage = 0.0
        self.motor_torque = 0.0
        self.servo_current = np.zeros(6)
        self.servo_voltage = np.zeros(6)
        self.nozzle_theta = np.zeros(6)
        self.area = np.zeros(6)
        self.thrust = np.zeros(6)
        self.dp = 0.0
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.set_com(com)

    def set_com(self, com) -> None:
        """Set the point about which nozzle torques are computed (body frame)."""
        arm = self.offsets - np.asarray(com, float)[None, :]
        self.thrust2torque = -np.cross(arm, self.orient).T     # (3, 6)
        self.thrust2force = -self.orient.T                       # (3, 6)

    def set_power(self, on: bool) -> None:
        """Power the propulsion channel of this module (off: fan coasts open-circuit, servos unpowered, flaps hold).
        Re-powering restarts the speed loop from the measured speed (ramp state = speed, integrators cleared)."""
        on = bool(on)
        if on and not self.powered:
            self.prev_speed_rate = self.impeller_speed
            self.speed_pid.integral = 0.0
            self.speed_pid.last = 0.0
            for pid in self.servo_pids:
                pid.integral, pid.last = 0.0, 0.0
        self.powered = on

    # -- the three sub-models, in the firmware's order -------------------------------------------------
    def _servo(self, servo_cmd):
        p = self.p
        if not self.powered:                 # unpowered servo: no drive, open circuit; the motor coasts on friction
            self.servo_voltage = np.zeros(6)
            self.servo_current = np.zeros(6)
            st, bl, hw = self.servo_thetas, self.backlash, p.servo_backlash_half_width
            lo = st < bl - hw
            hi = st > bl + hw
            bl[lo] = st[lo] + hw
            bl[hi & ~lo] = st[hi & ~lo] - hw
            st += p.control_period * self.prev_omega
            np.clip(st, 0.0, p.flap_gear_ratio * p.max_flap_travel / p.servo_gear_ratio, out=st)
            self.prev_omega = self.prev_omega + (1.0 / p.servo_inertia) * p.control_period * (-p.servo_friction * self.prev_omega)
            self.nozzle_theta = bl * p.servo_gear_ratio / p.flap_gear_ratio
            self.area = nozzle_area(self.nozzle_theta, self.width, p)
            return
        max_theta = p.flap_gear_ratio * p.max_flap_travel
        pwm2angle = max_theta / 255.0
        hw = p.servo_backlash_half_width
        st, bl = self.servo_thetas, self.backlash
        lo = st < bl - hw
        hi = st > bl + hw
        bl[lo] = st[lo] + hw
        bl[hi & ~lo] = st[hi & ~lo] - hw
        err = pwm2angle * np.asarray(servo_cmd, float) - bl * p.servo_gear_ratio
        v = np.array([self.servo_pids[i].run(err[i]) for i in range(6)])
        self.servo_voltage = v
        self.servo_current = (v - p.servo_k * self.prev_omega) / p.servo_r
        st += p.control_period * self.prev_omega
        np.clip(st, 0.0, max_theta / p.servo_gear_ratio, out=st)
        tq = p.servo_k * self.servo_current - p.servo_friction * self.prev_omega
        self.prev_omega = self.prev_omega + (1.0 / p.servo_inertia) * p.control_period * tq
        self.nozzle_theta = bl * p.servo_gear_ratio / p.flap_gear_ratio
        self.area = nozzle_area(self.nozzle_theta, self.width, p)

    def _impeller(self, speed_cmd, voltage):
        p = self.p
        if not self.powered:                 # open-circuit coast: friction (+ the air the fan still moves)
            tq = -p.imp_friction * self.impeller_speed
            if self.coast_aero_load:
                tq -= self.aero_torque(self.impeller_speed)
            self.impeller_voltage = self.impeller_current = 0.0
            self.motor_torque = tq
            self.last_speed = self.impeller_speed
            self.impeller_speed = max(0.0, self.impeller_speed + tq / p.imp_inertia * p.control_period)
            return
        max_change = p.imp_ramp * p.control_period
        speed = speed_cmd / p.imp_cmd_per_rad_s
        speed = min(max(speed, self.prev_speed_rate - max_change), self.prev_speed_rate + max_change)
        self.prev_speed_rate = speed
        v = self.speed_pid.run(speed - self.impeller_speed)
        v = max(-voltage, min(voltage, v))
        self.impeller_voltage = v
        cur = (v - self.impeller_speed / p.imp_speed_k) / p.imp_r
        tq = cur * p.imp_torque_k - p.imp_friction * self.impeller_speed
        if self.aero_load:
            tq -= self.aero_torque(self.impeller_speed)
        self.impeller_current = cur
        self.motor_torque = tq
        self.last_speed = self.impeller_speed
        self.impeller_speed += tq / p.imp_inertia * p.control_period

    def total_area(self) -> float:
        """Effective open area leak + sum Cd_i A_i [m^2] of the current flap state."""
        return self.leak + float(np.sum(self.cd * self.area))

    def aero_torque(self, omega) -> float:
        """Aerodynamic load torque [N m] at speed omega for the current open area (fan efficiency model)."""
        if omega <= 0:
            return 0.0
        a = self.total_area()
        dp = float(plenum_pressure(omega, a, self.p))
        q = a * math.sqrt(2.0 * max(dp, 0.0) / self.p.air_density)
        return dp * q / (self.fan_efficiency * omega)

    def _aero(self):
        a = self.total_area()
        self.dp = float(plenum_pressure(self.impeller_speed, a, self.p))
        self.thrust = 2.0 * self.dp * self.cd * self.cd * self.area * self.thrust_scale

    def _body(self, omega_body):
        h = self.axis * self.p.imp_inertia * self.impeller_speed
        self.force = self.thrust2force @ self.thrust
        self.torque = (self.axis * -self.motor_torque + self.gyro_sign * cross(omega_body, h)
                       + self.thrust2torque @ self.thrust)

    def step(self, impeller_cmd: int, servo_cmd, omega_body=(0.0, 0.0, 0.0), voltage: float | None = None) -> None:
        """Advance one control period with an impeller command (0..255 count) and 6 servo commands (0..255)."""
        v = self.p.battery_voltage if voltage is None else float(voltage)
        self._servo(servo_cmd)
        self._impeller(int(impeller_cmd), v)
        self._aero()
        self._body(np.asarray(omega_body, float))

    @property
    def momentum(self) -> np.ndarray:
        """Impeller angular momentum [N m s], body frame."""
        return self.axis * self.p.imp_inertia * self.impeller_speed

    @property
    def electrical_power(self) -> float:
        """Impeller motor electrical input V * I [W] (positive when drawing power)."""
        return self.impeller_voltage * self.impeller_current


class FanPropulsion:
    """Both propulsion modules; nozzle order 0-11 as in the firmware tables.

    step(impeller_cmds (2,), servo_cmds (12,), omega_body, voltage) at the control period.  Outputs:
    ``thrust`` (12,), ``force`` / ``torque`` (body frame, torque about ``com``), ``momentum`` (3,).
    """

    def __init__(self, p: PropulsionParams | None = None, com=(0.0, 0.0, 0.0), **module_kw):
        self.p = p if p is not None else PropulsionParams.reference()
        self.modules = [FanModule(self.p, 0, com, **module_kw), FanModule(self.p, 1, com, **module_kw)]
        self.version = 0          # incremented at every step (lets wrench sources cache their body loads)

    def set_com(self, com) -> None:
        """Set the torque reference point of both modules (body frame)."""
        for m in self.modules:
            m.set_com(com)

    def set_power(self, on: bool, modules=(0, 1)) -> None:
        """Power / cut the propulsion channel of the given modules (``FanModule.set_power``)."""
        for i in modules:
            self.modules[int(i)].set_power(on)

    @property
    def powered(self) -> list:
        """Power state of the two propulsion channels."""
        return [m.powered for m in self.modules]

    def step(self, impeller_cmds, servo_cmds, omega_body=(0.0, 0.0, 0.0), voltage: float | None = None) -> None:
        """Advance both modules by one control period."""
        sc = np.asarray(servo_cmds, float)
        for i, m in enumerate(self.modules):
            m.step(int(impeller_cmds[i]), sc[6 * i:6 * i + 6], omega_body, voltage)
        self.version += 1

    @property
    def thrust(self) -> np.ndarray:
        """(12,) nozzle thrusts [N]."""
        return np.concatenate([m.thrust for m in self.modules])

    @property
    def force(self) -> np.ndarray:
        """Body-frame force [N]."""
        return self.modules[0].force + self.modules[1].force

    @property
    def torque(self) -> np.ndarray:
        """Body-frame torque [N m] about the reference point (incl. impeller reaction and gyroscopic terms)."""
        return self.modules[0].torque + self.modules[1].torque

    @property
    def momentum(self) -> np.ndarray:
        """Total impeller angular momentum [N m s], body frame."""
        return self.modules[0].momentum + self.modules[1].momentum

    @property
    def impeller_speed(self) -> np.ndarray:
        """(2,) impeller speeds [rad/s]."""
        return np.array([m.impeller_speed for m in self.modules])

    @property
    def electrical_power(self) -> float:
        """Total impeller motor electrical power [W]."""
        return sum(m.electrical_power for m in self.modules)

    @property
    def area(self) -> np.ndarray:
        """(12,) nozzle open areas [m^2]."""
        return np.concatenate([m.area for m in self.modules])

    def nozzle_forces(self) -> np.ndarray:
        """(12, 3) per-nozzle force vectors, body frame."""
        return -self.thrust[:, None] * self.p.nozzle_exhaust_dir

    def is_ready(self, impeller_cmds, tol: float = 0.1) -> bool:
        """Firmware plant readiness: |commanded speed - impeller speed| < tol [rad/s] on both modules."""
        return all(abs(c / self.p.imp_cmd_per_rad_s - m.impeller_speed) < tol
                   for c, m in zip(impeller_cmds, self.modules))


def nozzle_matrix(p: PropulsionParams, com=(0.0, 0.0, 0.0)) -> np.ndarray:
    """(6, 12) map from nozzle thrust magnitudes to the body wrench [F; tau] about com."""
    arm = p.nozzle_position - np.asarray(com, float)[None, :]
    d = -p.nozzle_exhaust_dir
    return np.vstack([d.T, np.cross(arm, d).T])


def _pinv(M):
    """Moore-Penrose pseudo-inverse with the firmware's tolerance (singular values > double epsilon)."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    sinv = np.where(s > np.finfo(float).eps, 1.0 / s, 0.0)
    return (Vt.T * sinv) @ U.T


class FlightAllocator:
    """Port of the firmware force allocation (wrench -> impeller commands + 12 servo PWM commands).

    allocate(force_body, torque_body, speed_level) -> dict with
      'impeller_cmd' (2,) int, 'servo_pwm' (12,) float, 'servo_cmd' (12,) int (truncated as the
      firmware's driver message does), 'thrust' (12,) allocated nozzle thrusts, 'theta' (12,) flap angles,
      'dp' (2,) predicted plenum pressures, 'area' (12,) commanded areas.
    """

    def __init__(self, p: PropulsionParams | None = None, com=(0.0, 0.0, 0.0)):
        self.p = p if p is not None else PropulsionParams.reference()
        self.update_com(com)

    def update_com(self, com) -> None:
        """Rebuild the allocation matrices for a new configured centre of mass (body frame)."""
        self.com = np.asarray(com, float).copy()
        self.B = nozzle_matrix(self.p, self.com)
        self.Binv = _pinv(self.B)

    def thrusts(self, force_body, torque_body) -> np.ndarray:
        """(12,) nonnegative nozzle thrusts: pseudo-inverse, then the per-axis null-group shift."""
        u = self.Binv @ np.concatenate([np.asarray(force_body, float), np.asarray(torque_body, float)])
        for g in self.p.null_groups:
            g = list(g)
            mn = min(0.0, float(np.min(u[g])))
            if mn < 0:
                u[g] -= mn
        return u

    def plenum_dp(self, omega, cd, thrusts) -> float:
        """Firmware plenum pressure estimate from the thrust table (sum u_i / Cd_i over omega^2)."""
        total = float(np.sum(np.asarray(thrusts) / cd))
        c = self.p.thrust_table()(total / (omega * omega))
        return omega * omega * self.p.imp_diameter ** 2 * self.p.air_density * c

    def module_command(self, index: int, speed_level: int, thrusts):
        """Impeller command, servo PWM, flap angles, areas and dp for one module."""
        p = self.p
        if speed_level < 0 or speed_level > 3:
            raise ValueError(f"invalid speed level {speed_level}")
        cd = p.discharge_coeff[index]
        omega, dp, cmd = 0.0, 0.0, 0
        if speed_level != 0:
            omega = float(p.imp_levels[speed_level - 1])
            cmd = int(p.imp_cmd_per_rad_s * omega + 0.5) & 0xFF
            dp = self.plenum_dp(omega, cd, thrusts)
        npd = 2.0 * cd * cd * dp
        t = np.asarray(thrusts, float)
        if dp > 0:
            area = t / npd
        else:                                   # fans off: x / 0 -> inf (x > 0) or nan (x == 0), as the firmware
            area = np.where(t > 0, np.inf, np.nan)
        theta = flap_angle_for_area(area, p.nozzle_width, p)
        pwm = float(p.servo_pwm_max) / p.max_flap_travel * theta
        return cmd, pwm, theta, area, dp

    def allocate(self, force_body, torque_body, speed_level: int) -> dict:
        """Full firmware allocation step (see class docstring)."""
        u = self.thrusts(force_body, torque_body)
        c0, pwm0, th0, a0, dp0 = self.module_command(0, speed_level, u[:6])
        c1, pwm1, th1, a1, dp1 = self.module_command(1, speed_level, u[6:])
        pwm = np.concatenate([pwm0, pwm1])
        return dict(impeller_cmd=np.array([c0, c1]), servo_pwm=pwm,
                    servo_cmd=np.clip(np.trunc(pwm), 0, 255).astype(int), thrust=u,
                    theta=np.concatenate([th0, th1]), area=np.concatenate([a0, a1]), dp=np.array([dp0, dp1]))


class NNLSAllocator(FlightAllocator):
    """Alternative allocator: nonnegative least squares (Lawson-Hanson, numpy) + small Tikhonov term, then
    uniform down-scaling of the whole wrench when a module's total thrust capacity or a nozzle's open
    area would be exceeded (direction-preserving saturation).  Same output dict as FlightAllocator plus
    'scale' (<= 1)."""

    def __init__(self, p: PropulsionParams | None = None, com=(0.0, 0.0, 0.0), reg: float = 1e-6,
                 margin: float = 0.98):
        super().__init__(p, com)
        self.reg = reg
        self.margin = margin

    def thrusts(self, force_body, torque_body) -> np.ndarray:
        """(12,) minimum-residual nonnegative thrusts for the wrench (unscaled)."""
        w = np.concatenate([np.asarray(force_body, float), np.asarray(torque_body, float)])
        A = np.vstack([self.B, math.sqrt(self.reg) * np.eye(12)])
        u, _ = nnls(A, np.concatenate([w, np.zeros(12)]))
        return u

    def _feasible_scale(self, u, speed_level) -> float:
        """Largest s <= 1 such that s*u respects the thrust table range and every nozzle's max area."""
        p = self.p
        if speed_level == 0:
            return 0.0
        omega = float(p.imp_levels[speed_level - 1])
        amax = p.max_area()
        lo, hi = 0.0, 1.0

        def ok(s):
            for i in (0, 1):
                ui = s * u[6 * i:6 * i + 6]
                cd = p.discharge_coeff[i]
                tot = float(np.sum(ui / cd)) / (omega * omega)
                if tot > p.thrust_table_x[-1] * self.margin:
                    return False
                dp = self.plenum_dp(omega, cd, ui)
                with np.errstate(divide="ignore", invalid="ignore"):
                    a = ui / (2.0 * cd * cd * dp)
                if np.any(a > amax[6 * i:6 * i + 6] * self.margin):
                    return False
            return True
        if ok(1.0):
            return 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo

    def allocate(self, force_body, torque_body, speed_level: int) -> dict:
        """NNLS allocation with direction-preserving saturation (see class docstring)."""
        u = self.thrusts(force_body, torque_body)
        s = self._feasible_scale(u, speed_level)
        u = s * u
        c0, pwm0, th0, a0, dp0 = self.module_command(0, speed_level, u[:6])
        c1, pwm1, th1, a1, dp1 = self.module_command(1, speed_level, u[6:])
        pwm = np.concatenate([pwm0, pwm1])
        return dict(impeller_cmd=np.array([c0, c1]), servo_pwm=pwm,
                    servo_cmd=np.clip(np.trunc(pwm), 0, 255).astype(int), thrust=u,
                    theta=np.concatenate([th0, th1]), area=np.concatenate([a0, a1]), dp=np.array([dp0, dp1]),
                    scale=s)


def static_thrust(p: PropulsionParams, servo_pwm, impeller_speed, thrust_scale: float | None = None):
    """Steady-state plant thrusts (12,) for servo PWM commands held long enough and impeller speeds (2,).

    Uses the servo's commanded angle (the servo loop has integral action, so its steady state is the
    command up to the backlash, < 0.01 deg of flap) and the firmware plenum table.  Returns
    (thrust (12,), dp (2,), area (12,)).
    """
    k = p.thrust_error_scale if thrust_scale is None else thrust_scale
    theta = np.asarray(servo_pwm, float) * p.max_flap_travel / float(p.servo_pwm_max)
    w = np.concatenate([p.nozzle_width, p.nozzle_width])
    area = nozzle_area(theta, w, p)
    thrust, dps = np.zeros(12), np.zeros(2)
    for i in (0, 1):
        sl = slice(6 * i, 6 * i + 6)
        cd = p.discharge_coeff[i]
        tot = p.leak_area[i] + float(np.sum(cd * area[sl]))
        dps[i] = float(plenum_pressure(impeller_speed[i], tot, p))
        thrust[sl] = 2.0 * dps[i] * cd * cd * area[sl] * k
    return thrust, dps, area


def impeller_power(p: PropulsionParams, omega, total_area, fan_efficiency: float = 0.52, index: int = 0) -> dict:
    """Steady-state electrical power of one impeller motor at speed omega [rad/s] and open area [m^2].

    Load torque = viscous friction + aerodynamic torque (air power dp * Q / fan_efficiency); current
    I = tau / k_t, voltage V = omega / k_s + I R.  Returns dict(power_w, current_a, voltage_v,
    air_power_w, shaft_power_w, copper_loss_w, flow_m3s, dp_pa).
    """
    dp = float(plenum_pressure(omega, total_area, p))
    q = float(total_area) * math.sqrt(2.0 * max(dp, 0.0) / p.air_density)
    p_air = dp * q
    tau = p.imp_friction * omega + (p_air / (fan_efficiency * omega) if omega > 0 else 0.0)
    cur = tau / p.imp_torque_k
    volt = omega / p.imp_speed_k + cur * p.imp_r
    return dict(power_w=volt * cur, current_a=cur, voltage_v=volt, air_power_w=p_air, shaft_power_w=tau * omega,
                copper_loss_w=cur * cur * p.imp_r, flow_m3s=q, dp_pa=dp)


class PropulsionWrench:
    """Wrench source applying the propulsion plant's loads to the robot body in MuJoCo.

    Every nozzle force acts at its nozzle position; the torque is taken about the body's actual CoM in the
    model (``model.body_ipos``), so a CoM offset produces the correct moment.  The fan motor reaction and the
    gyroscopic torque of the fan momentum (with the body's current rate) are added.  The plant is stepped
    separately (FanPropulsion.step at the control rate); this source replays its latest outputs (zero-order
    hold between control ticks; the body-frame loads are cached per plant step).
    nozzle_pos_body: (12, 3) nozzle positions in the MuJoCo body frame (default: the plant's table,
    i.e. MuJoCo body frame = firmware body frame).  intake_coeff (default 0 = firmware plant): optional
    reaction of the air drawn into each fan intake, -k rho Q^2 / A_in along the module's fan axis; a steady sink
    in still or uniform air carries no momentum to infinity, so the physical value is ~0 (see the notes).
    extra_force_body: constant body-frame force (default 0), e.g. the avionics cooling-air exhaust.
    """

    def __init__(self, body_id: int, propulsion: FanPropulsion, nozzle_pos_body=None, intake_coeff: float = 0.0,
                 intake_area: float | None = None, extra_force_body=(0.0, 0.0, 0.0)):
        self.body = int(body_id)
        self.prop = propulsion
        self.pos = (propulsion.p.nozzle_position if nozzle_pos_body is None
                    else np.asarray(nozzle_pos_body, float))
        self.enabled = True
        # intake reaction (off by default, as in the firmware plant): F = -k rho Q^2 / A_in along the intake's
        # outward axis (the fan axis of each module); k = 1 is the naive momentum flux of the drawn-in air
        self.intake_coeff = float(intake_coeff)
        self.intake_area = (math.pi * (propulsion.p.imp_diameter / 2) ** 2 if intake_area is None else float(intake_area))
        # constant body-frame force, e.g. the avionics cooling-air exhaust (published bound: < 5 % of the thrust)
        self.extra_force_body = np.asarray(extra_force_body, float)
        self._version = None
        self._Fb = np.zeros(3)
        self._Tb = np.zeros(3)
        self._h = np.zeros(3)
        self._sign = propulsion.modules[0].gyro_sign

    def _refresh(self, model):
        f_noz = self.prop.nozzle_forces()                        # (12, 3) body frame
        self._Fb = f_noz.sum(0) + self.extra_force_body
        tau = cross_rows(self.pos - model.body_ipos[self.body][None, :], f_noz).sum(0)
        for m in self.prop.modules:                               # fan motor reaction
            tau = tau + m.axis * -m.motor_torque
        if self.intake_coeff:
            p = self.prop.p
            for m in self.prop.modules:
                q = float(intake_flow(m.impeller_speed, m.total_area(), p))
                self._Fb = self._Fb - self.intake_coeff * p.air_density * q * q / self.intake_area * m.axis
        self._Tb = tau
        self._h = self.prop.momentum
        self._version = self.prop.version

    def body_wrench(self, model, data):
        """(force, torque) in world axes at the body CoM."""
        if self._version != self.prop.version:
            self._refresh(model)
        b = self.body
        R = data.xmat[b].reshape(3, 3)
        w_body = R.T @ data.cvel[b, 0:3]
        tau = self._Tb + self._sign * cross(w_body, self._h)
        return R @ self._Fb, R @ tau

    def accumulate(self, model, data, W) -> None:
        """Add the propulsion wrench into W."""
        if not self.enabled:
            return
        F, T = self.body_wrench(model, data)
        W[self.body, 0:3] += F
        W[self.body, 3:6] += T


def max_along(direction, allocator, speed_level: int, plant: PropulsionParams | None = None, kind: str = "force",
              rel_tol: float = 0.01, lever: float = 0.1, s_max: float | None = None, thrust_scale: float | None = None,
              n_grid: int = 60, iterations: int = 8, abs_tol: float | None = None):
    """Largest pure force (kind='force', N) or pure torque (N m) along ``direction`` that the allocator + static
    plant can hold, as a closed loop would: for a target s * e the command is corrected ``iterations`` times by
    the delivered-minus-target error (like the controller's feedback), and the target counts as feasible if
    the remaining error (torque errors divided by ``lever`` [m] for forces, force errors multiplied by it for
    torques) is <= rel_tol * s + abs_tol (abs_tol covers the 8-bit servo command resolution: default 2 mN or
    0.2 mN m).  s is scanned on a grid and refined by bisection at the first failure.
    Returns (feasible magnitude, delivered wrench (6,), final command (6,))."""
    p = allocator.p if plant is None else plant
    e = np.asarray(direction, float)
    e = e / np.linalg.norm(e)
    B = nozzle_matrix(p, allocator.com)
    hi = (2.0 if kind == "force" else 0.3) if s_max is None else s_max
    W = np.array([1, 1, 1, 1 / lever, 1 / lever, 1 / lever]) if kind == "force" else \
        np.array([lever, lever, lever, 1, 1, 1])
    atol = (2e-3 if kind == "force" else 2e-4) if abs_tol is None else abs_tol

    def delivered(w):
        al = allocator.allocate(w[:3], w[3:], speed_level)
        speeds = al["impeller_cmd"] / p.imp_cmd_per_rad_s
        th, _, _ = static_thrust(p, al["servo_cmd"], speeds, thrust_scale)
        return B @ th

    def feasible(s):
        t = np.zeros(6)
        if kind == "force":
            t[:3] = s * e
        else:
            t[3:] = s * e
        c = t.copy()
        for _ in range(iterations):
            a = delivered(c)
            c = c + (t - a)
        a = delivered(c)
        return float(np.linalg.norm(W * (a - t))) <= rel_tol * s + atol, a, c

    best = (0.0, np.zeros(6), np.zeros(6))
    prev_ok, fail = 0.0, None
    for s in np.linspace(hi / n_grid, hi, n_grid):
        good, a, c = feasible(s)
        if good:
            prev_ok, best = s, (s, a, c)
        else:
            fail = s
            break
    if fail is not None:
        lo_s, hi_s = prev_ok, fail
        for _ in range(14):
            mid = 0.5 * (lo_s + hi_s)
            good, a, c = feasible(mid)
            if good:
                lo_s, best = mid, (mid, a, c)
            else:
                hi_s = mid
    return best


def wrench_envelope(allocator, speed_level: int, mass: float, n_dirs: int = 64, plant: PropulsionParams | None = None,
                    rel_tol: float = 0.02, thrust_scale: float | None = None) -> dict:
    """Pure-force acceleration envelope over ``n_dirs`` Fibonacci-sphere directions plus the six signed axes,
    and the pure torque per signed axis.  Returns dict(axis_force (6,), axis_torque (6,), accel_min, accel_max,
    accel_mean, dirs (n, 3), accel (n,))."""
    k = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - 2 * k / n_dirs)
    th = np.pi * (1 + 5 ** 0.5) * k
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    acc = np.array([max_along(d, allocator, speed_level, plant, "force", rel_tol, thrust_scale=thrust_scale, n_grid=30)[0]
                    / mass for d in dirs])
    axes = [np.eye(3)[i] * s for i in range(3) for s in (1, -1)]
    fa = np.array([max_along(a, allocator, speed_level, plant, "force", rel_tol, thrust_scale=thrust_scale)[0] for a in axes])
    ta = np.array([max_along(a, allocator, speed_level, plant, "torque", rel_tol, thrust_scale=thrust_scale)[0] for a in axes])
    return dict(axis_force=fa, axis_torque=ta, accel_min=float(min(acc.min(), fa.min() / mass)),
                accel_max=float(max(acc.max(), fa.max() / mass)), accel_mean=float(acc.mean()), dirs=dirs, accel=acc)


def coast_down(p: PropulsionParams | None = None, level: int = 3, servo_pwm=None, mass: float = 10.449993,
               duration: float = 120.0, aero_load: bool = True, modules=(0, 1), settle_s: float = 20.0,
               thrust_scale: float | None = None) -> dict:
    """Thrust after a propulsion power cut (v0.3): both fans at speed ``level`` with the servo commands ``servo_pwm``
    (12,) held until steady, then the channels of ``modules`` lose power at t = 0: the fans coast (open circuit,
    friction and -- ``aero_load`` -- the air they still pump), the flaps stay where they were.  Returns dict(t (n,),
    thrust (n, 12), force_body (n, 3) [N], impeller_speed (n, 2), impulse_body (3,) [N s], dv_body (3,) [m/s] for a
    robot of ``mass`` (no drag, no rotation), t_half_thrust [s] (|F| below half its value at the cut), and the force
    and speeds at the cut).  The impeller inertia is the plant's (the firmware's 0.001 kg m^2)."""
    p = PropulsionParams.reference() if p is None else p
    fp = FanPropulsion(p, thrust_scale=thrust_scale, coast_aero_load=aero_load)
    cmd = int(p.imp_cmd_per_rad_s * p.imp_levels[level - 1] + 0.5) if level else 0
    sc = np.zeros(12) if servo_pwm is None else np.asarray(servo_pwm, float)
    w0 = cmd / p.imp_cmd_per_rad_s
    for m in fp.modules:                     # settled speed loop at the level (as FreeFlyerRig.spin_up)
        m.impeller_speed = m.last_speed = m.prev_speed_rate = w0
        v_ss = w0 / p.imp_speed_k + (p.imp_friction * w0 / p.imp_torque_k) * p.imp_r
        m.speed_pid.integral = v_ss / m.speed_pid.ki if m.speed_pid.ki else 0.0
    for _ in range(int(round(settle_s / p.control_period))):
        fp.step([cmd, cmd], sc)
    F0 = fp.force.copy()
    fp.set_power(False, modules)
    n = int(round(duration / p.control_period))
    t = np.arange(1, n + 1) * p.control_period
    T = np.zeros((n, 12))
    Fb = np.zeros((n, 3))
    W = np.zeros((n, 2))
    for k in range(n):
        fp.step([cmd, cmd], sc)
        T[k] = fp.thrust
        Fb[k] = fp.force
        W[k] = fp.impeller_speed
    imp = Fb.sum(axis=0) * p.control_period
    mag = np.linalg.norm(Fb, axis=1)
    f0 = float(np.linalg.norm(F0))
    below = np.nonzero(mag < 0.5 * f0)[0]
    return dict(t=t, thrust=T, force_body=Fb, impeller_speed=W, impulse_body=imp, dv_body=imp / float(mass),
                t_half_thrust=float(t[below[0]]) if len(below) and f0 > 0 else None, force_at_cut=F0,
                speed_at_cut=w0, residual_force_end=Fb[-1].copy())

