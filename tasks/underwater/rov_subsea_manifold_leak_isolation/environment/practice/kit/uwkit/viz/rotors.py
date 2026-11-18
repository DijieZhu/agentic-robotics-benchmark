"""Rotor data (from rotor_split.json) and the rotor-speed / rotor-angle models used by the visual twin.

Rotor speed: the simulator's own per-thruster speed state (UnderwaterSim.thrusters.rpm: the official bollard table's rpm
column at the battery voltage, blended between table voltages, signed by the PWM side, zero inside the deadband, following
the first-order command lag).  For logs that only carry commands or thrust, rpm_from_command / rpm_from_thrust evaluate the
same table (steady state).

Spin direction: omega = spin_sign_forward[k] * rpm * 2 pi / 60 about axis_B (axis_B = site +x = force on the body under
forward thrust = towards the nose cone).  spin_sign_forward = -1 for the CW propeller (clockwise seen from the nose cone),
+1 for the CCW propeller; checked against the blade geometry in build_rotor_split.py.

Display angle (frame-rate aliasing):
  'true'    : angle = integral of the real omega over simulation time (at 30 fps a 3300 rpm rotor turns 660 deg per
              frame -> stroboscopic / wagon-wheel effect; use with motion blur)
  'limited' : angle = integral of a display speed that keeps the direction and the order of the speeds but never exceeds
              max_deg_per_frame (default 40 deg per frame: a 3-blade rotor looks the same every 120 deg, so the apparent
              direction flips above 60 deg per frame; 40 deg keeps a 1.5x margin)
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROTOR_SPLIT = os.path.join(HERE, "rotor_split.json")
RPM_TO_RAD_S = 2.0 * math.pi / 60.0


def load_rotor_split(path: Optional[str] = None) -> dict:
    with open(path or ROTOR_SPLIT) as f:
        return json.load(f)


@dataclass
class Rotor:
    site: str
    item: str
    propeller: str               # 'CW' | 'CCW'
    prototype: str               # 'cw' | 'ccw'
    axis_B: np.ndarray           # unit, = site +x
    centre_B: np.ndarray         # propeller centre on the axis [m]
    T_B: np.ndarray              # 4x4 prototype -> B
    spin_sign_forward: int       # sign of omega . axis_B under forward thrust
    rear_lip_B: np.ndarray       # duct exit for forward thrust
    front_lip_B: np.ndarray      # duct exit for reverse thrust
    duct_inner_radius: float
    tip_radius: float
    hub_radius: float


class RotorSet:
    """The rotors of one vehicle configuration, in site order thr_1 .. thr_n."""

    def __init__(self, config: str = "mission", path: Optional[str] = None, sites=None):
        J = load_rotor_split(path)
        self.split = J
        rows = J["thrusters"][config]
        if sites is not None:
            by = {r["site"]: r for r in rows}
            rows = [by[s] for s in sites]
        self.config = config
        self.rotors = []
        for r in rows:
            P = J["prototypes"][r["prototype"]]
            self.rotors.append(Rotor(site=r["site"], item=r["item"], propeller=r["propeller"], prototype=r["prototype"],
                                     axis_B=np.asarray(r["axis_B"], float) / np.linalg.norm(r["axis_B"]),
                                     centre_B=np.asarray(r["centre_B_m"], float),
                                     T_B=np.asarray(r["T_B"], float), spin_sign_forward=int(r["spin_sign_forward"]),
                                     rear_lip_B=np.asarray(r["rear_lip_B_m"], float),
                                     front_lip_B=np.asarray(r["front_lip_B_m"], float),
                                     duct_inner_radius=float(P["duct_inner_radius_m"]),
                                     tip_radius=float(P["tip_radius_m"]), hub_radius=float(P["hub_radius_m"])))
        self.n = len(self.rotors)
        self.spin = np.array([r.spin_sign_forward for r in self.rotors], float)
        self.sites = [r.site for r in self.rotors]

    def prototype(self, key):
        return self.split["prototypes"][key]

    # ------------------------------------------------------------------ speed models
    def omega(self, rpm) -> np.ndarray:
        """Real angular speed about axis_B [rad/s] (signed) for signed rotor speeds [rpm]."""
        return self.spin * np.asarray(rpm, float) * RPM_TO_RAD_S

    def visible_sense(self, rpm, eps=1e-9) -> np.ndarray:
        """+1 = counter-clockwise seen from the nose cone (omega along +axis_B), -1 = clockwise, 0 = stopped."""
        return np.sign(self.spin * np.where(np.abs(rpm) > eps, rpm, 0.0))


def rpm_from_command(u, V=14.8, curve=None, pwm_center=1500.0, pwm_half_span=400.0) -> np.ndarray:
    """Steady signed rotor speed [rpm] for normalised commands u in [-1, 1] at supply voltage V (official table)."""
    if curve is None:
        from ..dynamics.thrusters import ThrusterCurve
        curve = ThrusterCurve()
    pwm = pwm_center + np.clip(np.asarray(u, float), -1, 1) * pwm_half_span
    return np.asarray(curve.steady(pwm, V)[1], float)


def rpm_from_thrust(T, V=14.8, curve=None) -> np.ndarray:
    """Steady signed rotor speed [rpm] that produces bollard thrust T [N] at V (inverse of the official table)."""
    if curve is None:
        from ..dynamics.thrusters import ThrusterCurve
        curve = ThrusterCurve()
    return np.asarray(curve.steady(curve.inverse(np.asarray(T, float), V), V)[1], float)


def rpm_reference(V=16.8, curve=None) -> float:
    """max |rpm| of the official table at V (default: a full 4S battery) - the display-speed normaliser."""
    if curve is None:
        from ..dynamics.thrusters import ThrusterCurve
        curve = ThrusterCurve()
    return float(np.abs(curve.column(V)[1]).max())


@dataclass
class DisplaySpeed:
    """Display-speed map for the 'limited' mode: omega_disp = spin * sign(rpm) * cap * min(1, (|rpm| / rpm_ref)^gamma),
    cap = max_deg_per_frame * fps.  gamma = 1 keeps the speed ratios exactly (below the cap)."""
    fps: float = 30.0
    max_deg_per_frame: float = 40.0
    rpm_ref: float = 3800.0
    gamma: float = 1.0

    @property
    def cap_rad_s(self):
        return math.radians(self.max_deg_per_frame) * self.fps

    def omega(self, rpm, spin) -> np.ndarray:
        rpm = np.asarray(rpm, float)
        x = np.minimum(np.abs(rpm) / self.rpm_ref, 1.0) ** self.gamma
        return np.asarray(spin, float) * np.sign(rpm) * self.cap_rad_s * x


def integrate_angles(rpm_rows, dt, spin, display: Optional[DisplaySpeed] = None, phi0=None) -> np.ndarray:
    """Rotor angles at every log row: phi[i] = phi0 + sum_{j = 1..i} omega(rpm[j]) dt, where row j holds the rotor speed
    of the physics step that ended at row j (the Recorder convention, so display=None reproduces phi_true exactly).
    display=None: real speed; else the display map.  Returns (n_rows, n) [rad], unwrapped."""
    R = np.asarray(rpm_rows, float)
    w = (spin * R * RPM_TO_RAD_S) if display is None else display.omega(R, spin)
    phi = np.zeros_like(R)
    if phi0 is not None:
        phi[0] = phi0
    if len(R) > 1:
        phi[1:] = phi[0] + np.cumsum(w[1:] * dt, axis=0)
    return phi
