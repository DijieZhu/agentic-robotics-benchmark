"""Micro metal gearmotor (Pololu #5192, HPCB 6V 150.58:1, 12 CPR encoder) winding the line on a 6.35 mm spool.

Quasi-static brushed-DC model through the vendor's two points (6 V: no-load 220 RPM / 150 mA, stall 2.0 kg*cm /
1.5 A), scaled linearly with the applied voltage V = duty * V_bat (PWM averaging):
    tau(omega) = tau_s * V/V_r - b_m * omega,  b_m = tau_s / omega_nl   (output shaft)
    current    = (V - k_e * omega) / R,        R = V_r / i_s, k_e = (V_r - i_nl R) / omega_nl
The spool speed follows from the balance tau(omega) = T * r (line tension T, spool radius r): the line speed and
the stall force come out of the physics, not out of a speed setpoint. At duty 0 the TB6612 short-brakes the motor;
the gear train is taken as holding up to the vendor's rated instantaneous torque (ASSUMPTION motor_hold_when_idle).
Positive omega winds the line in (shortens the cable).

Sensitivity knobs (defaults = the declared ASSUMPTIONs; their measured sensitivities in params.py):
  spool_r  effective winding radius [m] (ASSUMPTION spool_layering: the drum radius 3.175 mm)
  tau_m    mechanical time constant [s] of a first-order lag of the spool speed (ASSUMPTION motor_model: 0 = no
           rotor / gearbox inertia)
  hold     static holding torque [N*m] of the braked or stalled gear train (ASSUMPTION motor_hold_when_idle: the
           vendor's 2 kg*cm instantaneous limit; 0 = the short-brake damping b_m only)
"""
import math

from . import params as prm


class GearMotor:
    def __init__(self, v_bat=None, spool_r=None, tau_m=0.0, hold=None):
        self.Vr = prm.value("motor_rated_voltage")
        self.w_nl = prm.value("motor_no_load_speed")
        self.tau_s = prm.value("motor_stall_torque")
        self.i_nl = prm.value("motor_no_load_current")
        self.i_s = prm.value("motor_stall_current")
        self.tau_hold = prm.value("motor_torque_limit") if hold is None else float(hold)
        self.R = self.Vr / self.i_s
        self.k_e = (self.Vr - self.i_nl * self.R) / self.w_nl
        self.b_m = self.tau_s / self.w_nl
        self.v_bat = prm.value("battery_voltage") if v_bat is None else v_bat
        self.r = prm.value("spool_radius") if spool_r is None else spool_r
        self.tau_m = float(tau_m or 0.0)
        self.cpr_out = prm.value("encoder_cpr") * prm.value("motor_gear_ratio")
        self.theta = 0.0         # output-shaft angle [rad], + = wound in
        self.omega = 0.0
        self.energy_elec = 0.0   # J drawn from the motor pack
        self.work_mech = 0.0     # J of output-shaft work (tau * omega)
        self.peak_current = 0.0
        self.stall_time = 0.0

    def speed(self, duty, tension):
        """Quasi-static spool speed [rad/s] for a duty in [-1, 1] and a line tension [N] (>= 0)."""
        load = max(tension, 0.0) * self.r
        if duty == 0.0:
            if load <= self.tau_hold:
                return 0.0
            return -(load - self.tau_hold) / self.b_m
        drive = self.tau_s * duty * self.v_bat / self.Vr
        if drive > 0.0:
            if load <= drive:
                return (drive - load) / self.b_m
            if load <= drive + self.tau_hold:
                return 0.0                                     # stalled, held by the gear train
            return -(load - drive - self.tau_hold) / self.b_m
        return (drive - load) / self.b_m                       # paying out (load assists)

    def step(self, duty, tension, dt):
        """Advance by dt; returns the change of the cable rest length [m] (negative = wound in)."""
        w_ss = self.speed(duty, tension)
        if self.tau_m > 0.0:
            w = self.omega + (w_ss - self.omega) * min(1.0, dt / self.tau_m)
        else:
            w = w_ss
        self.omega = w
        self.theta += w * dt
        if duty != 0.0:
            V = duty * self.v_bat
            i = (V - self.k_e * w) / self.R
            p = V * i
            if p > 0.0:
                self.energy_elec += p * dt
            self.peak_current = max(self.peak_current, abs(i))
            if w == 0.0:
                self.stall_time += dt
        self.work_mech += max(tension, 0.0) * self.r * w * dt
        return -self.r * w * dt

    def encoder_counts(self):
        return int(math.floor(self.theta / (2 * math.pi) * self.cpr_out))

    def no_load_line_speed(self, duty=None):
        d = prm.value("pwm_max_fraction") if duty is None else duty
        return self.w_nl * d * self.v_bat / self.Vr * self.r

    def stall_line_force(self, duty=None):
        d = prm.value("pwm_max_fraction") if duty is None else duty
        return self.tau_s * d * self.v_bat / self.Vr / self.r

    def state(self):
        return (self.theta, self.omega, self.energy_elec, self.work_mech, self.peak_current, self.stall_time)

    def set_state(self, st):
        self.theta, self.omega, self.energy_elec, self.work_mech, self.peak_current, self.stall_time = st
