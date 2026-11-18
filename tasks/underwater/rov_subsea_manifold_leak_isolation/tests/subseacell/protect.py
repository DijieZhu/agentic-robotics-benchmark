"""Protective behaviours of the vehicle firmware (task 07, cell side; DESIGN section 1.4 "保护行为", package P04).

Three rules, all executed (phase 1 only counted the collision stops):

  collision stop   contact force on the vehicle body (every vehicle body except the two gripper jaws; a grasped object
                   is a separate body whose contacts with the vehicle are masked) > collision_force_N, or its impulse
                   over the last impulse_window_s > collision_impulse_Ns
                   -> thrusters off for stop_s, then depth + attitude hold for hold_s, count + 1;
                   the max_stops-th stop disables the vehicle and ends the episode (no stop / hold sequence).
  out of bounds    horizontal distance of the vehicle origin from the charted manifold centre > oob_horizontal_m, or
                   height of the vehicle origin above the mudline > oob_altitude_m, continuously for oob_persist_s
                   -> surfacing, episode ends.
  undervoltage     battery terminal voltage < undervoltage_V continuously for undervoltage_persist_s
                   -> surfacing, episode ends.

Measurement.  The contact force is F = |sum of the contact forces on the counted bodies| (MuJoCo contacts of the step,
mj_contactForce); the impulse is the sum of F * dt over the last round(impulse_window_s / dt) physics steps (the
counting of gates G1 / G2 / G6).  A stop starts on a rising edge of (F > force or impulse > impulse limit).  Rising
edges during the thrusters-off part of a stop belong to the same collision (the vehicle is not driving) and are logged
but not counted; a rising edge during the hold part starts a new stop.  The out-of-bounds rule uses the true vehicle
position (the cage's tracking, not the program's navigation estimate, which the program can reset); the
undervoltage rule uses the battery's terminal voltage.  All rules are checked once per physics step (tick()).

Phases (phase(t)): None | "stop" | "hold" | "surfacing" | "disabled".  The firmware (firmware.py) acts on them;
`ended` / `end_reason` tell the cell to end the episode ("disabled", "out_of_bounds", "undervoltage").

    P = Protection(sim.model, sim.data, ProtectConfig.from_values(values), site=SiteFrame(centre_xy, mudline_z))
    every physics step, after mj_step:   new_events = P.tick(sim.time, sim.battery.voltage)

No number of the reference vehicle is written here; the thresholds are DESIGN values (spec/task_values.json
`protect.*`, `budget.protective_stops`), handed in through ProtectConfig.from_values() by the cell's config.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional, Sequence

import mujoco
import numpy as np

PHASES = (None, "stop", "hold", "surfacing", "disabled")
END_REASONS = ("disabled", "out_of_bounds", "undervoltage")

# task_values.json key -> ProtectConfig field
VALUE_KEYS = {
    "protect.collision.force_N": "collision_force_N",
    "protect.collision.impulse_Ns": "collision_impulse_Ns",
    "protect.collision.impulse_window_s": "impulse_window_s",
    "protect.collision.stop_s": "stop_s",
    "protect.collision.hold_s": "hold_s",
    "budget.protective_stops": "max_stops",
    "protect.out_of_bounds.horizontal_m": "oob_horizontal_m",
    "protect.out_of_bounds.altitude_m": "oob_altitude_m",
    "protect.out_of_bounds.persist_s": "oob_persist_s",
    "protect.undervoltage.V": "undervoltage_V",
    "protect.undervoltage.persist_s": "undervoltage_persist_s",
}


@dataclass(frozen=True)
class ProtectConfig:
    """Thresholds of the protective behaviours (defaults = DESIGN v1.2 section 1.4; tests check them against
    spec/task_values.json)."""
    collision_force_N: float = 150.0
    collision_impulse_Ns: float = 12.0
    impulse_window_s: float = 0.1
    stop_s: float = 2.0
    hold_s: float = 8.0
    max_stops: int = 6
    oob_horizontal_m: float = 60.0
    oob_altitude_m: float = 15.0
    oob_persist_s: float = 10.0
    undervoltage_V: float = 12.0
    undervoltage_persist_s: float = 5.0

    @classmethod
    def from_values(cls, values: dict) -> "ProtectConfig":
        """From a flat {task_values key: value} mapping (the cell's grader_config); missing keys keep the default,
        unknown protect.* keys are an error."""
        kw = {}
        for k, v in values.items():
            if k in VALUE_KEYS:
                f = VALUE_KEYS[k]
                kw[f] = int(v) if f == "max_stops" else float(v)
            elif k.startswith("protect."):
                raise KeyError(f"unknown protection key {k!r}")
        return cls(**kw)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class SiteFrame:
    """Where the out-of-bounds rule is measured: the charted (nominal) manifold centre in world x, y [m] and the
    mudline height z [m].  None for facilities without a geofence (the training tank)."""
    centre_xy: tuple
    mudline_z: float


class Protection:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: Optional[ProtectConfig] = None,
                 site: Optional[SiteFrame] = None, vehicle_root: str = "hull",
                 excluded_bodies: Sequence[str] = ("jaw_upper", "jaw_lower")):
        self.m, self.d = model, data
        self.cfg = config or ProtectConfig()
        self.site = site
        root = model.body(vehicle_root).id
        self.root = int(model.body_rootid[root])
        self.hull = root
        excl = {model.body(n).id for n in excluded_bodies if _has_body(model, n)}
        counted = np.zeros(model.nbody, bool)
        for b in range(model.nbody):
            if int(model.body_rootid[b]) == self.root and b not in excl:
                counted[b] = True
        self._counted = counted
        self.counted_bodies = tuple(model.body(b).name for b in np.nonzero(counted)[0])
        self._geom_counted = counted[model.geom_bodyid]
        dt = float(model.opt.timestep)
        self.dt = dt
        self.n_win = max(1, int(round(self.cfg.impulse_window_s / dt)))
        self._f6 = np.zeros(6)
        self.reset()

    # ------------------------------------------------------------------------------------------------ state
    def reset(self) -> None:
        self.win = np.zeros(self.n_win)
        self.wi = 0
        self.count = 0
        self.events = []                       # grader-side log (t, kind, ...); never shown to the program
        self._on = False
        self._phase = None
        self._t_phase = None                   # start of the current stop (t0 of the "stop" + "hold" sequence)
        self.ended = False
        self.end_reason = None
        self.t_end = None
        self._t_oob = None                     # start of the current out-of-bounds stretch
        self._t_uv = None                      # start of the current undervoltage stretch
        self.F = 0.0
        self.impulse = 0.0
        self.F_peak = 0.0
        self.impulse_peak = 0.0
        self.new_stop = False                  # set by tick() when a stop / disable started in this step

    # ------------------------------------------------------------------------------------------------ queries
    def phase(self, t: float):
        """None | 'stop' | 'hold' | 'surfacing' | 'disabled' at time t."""
        ph = self._phase
        if ph in ("stop", "hold"):
            el = t - self._t_phase
            if el < self.cfg.stop_s - 1e-9:
                return "stop"
            if el < self.cfg.stop_s + self.cfg.hold_s - 1e-9:
                return "hold"
            return None
        return ph

    def remaining_s(self, t: float) -> float:
        """Time left in the stop + hold sequence (0 when none is active; inf when the vehicle is disabled or
        surfacing)."""
        ph = self.phase(t)
        if ph in ("stop", "hold"):
            return max(0.0, self._t_phase + self.cfg.stop_s + self.cfg.hold_s - t)
        if ph in ("surfacing", "disabled"):
            return math.inf
        return 0.0

    def active(self, t: float) -> bool:
        return self.phase(t) is not None

    def t_stop_started(self):
        return self._t_phase

    # ------------------------------------------------------------------------------------------------ per step
    def contact_force(self) -> tuple:
        """(|net contact force on the counted vehicle bodies| [N], name of the largest other geom or None)."""
        m, d = self.m, self.d
        n = int(d.ncon)
        if n == 0:
            return 0.0, None
        con = d.contact
        g = con.geom[:n]
        c1 = self._geom_counted[g[:, 0]]
        c2 = self._geom_counted[g[:, 1]]
        sel = np.nonzero(c1 != c2)[0]
        if sel.size == 0:
            return 0.0, None
        frames = con.frame[:n]
        F = np.zeros(3)
        big, other = -1.0, None
        f6 = self._f6
        for i in sel:
            i = int(i)
            mujoco.mj_contactForce(m, d, i, f6)
            fw = frames[i].reshape(3, 3).T @ f6[:3]           # force on geom2 from geom1, world axes
            on1 = bool(c1[i])
            f_on = -fw if on1 else fw
            F += f_on
            mag = float(f_on @ f_on)
            if mag > big:
                big, other = mag, int(g[i, 1] if on1 else g[i, 0])
        name = None
        if other is not None:
            name = m.geom(other).name or f"geom{other}"
        return float(math.sqrt(float(F @ F))), name

    def tick(self, t: float, battery_voltage: float) -> list:
        """Run every rule once (call after each physics step).  Returns the events started in this step."""
        self.new_stop = False
        if self.ended:
            return []
        cfg = self.cfg
        out = []
        F, other = self.contact_force()
        self.F = F
        self.win[self.wi] = F * self.dt
        self.wi = (self.wi + 1) % self.n_win
        imp = float(self.win.sum())
        self.impulse = imp
        self.F_peak = max(self.F_peak, F)
        self.impulse_peak = max(self.impulse_peak, imp)
        on = F > cfg.collision_force_N or imp > cfg.collision_impulse_Ns
        rising = on and not self._on
        self._on = on
        if rising:
            ph = self.phase(t)
            if ph == "stop":
                ev = dict(t=t, kind="collision_during_stop", F_N=F, impulse_Ns=imp, other=other, count=self.count)
                self.events.append(ev)
            else:
                self.count += 1
                if self.count >= cfg.max_stops:
                    ev = dict(t=t, kind="vehicle_disabled", F_N=F, impulse_Ns=imp, other=other, count=self.count)
                    self._phase, self._t_phase = "disabled", t
                    self._end(t, "disabled")
                else:
                    ev = dict(t=t, kind="protective_stop", F_N=F, impulse_Ns=imp, other=other, count=self.count)
                    self._phase, self._t_phase = "stop", t
                self.events.append(ev)
                out.append(ev)
                self.new_stop = True
        if self._phase in ("stop", "hold") and self.phase(t) is None:
            self._phase, self._t_phase = None, None
        if self.ended:
            return out
        # ---------------- out of bounds (true vehicle origin)
        if self.site is not None:
            p = self.d.xpos[self.hull]
            cx, cy = self.site.centre_xy
            horiz = math.hypot(float(p[0]) - cx, float(p[1]) - cy)
            alt = float(p[2]) - float(self.site.mudline_z)
            if horiz > cfg.oob_horizontal_m or alt > cfg.oob_altitude_m:
                if self._t_oob is None:
                    self._t_oob = t
                elif t - self._t_oob >= cfg.oob_persist_s - 1e-9:
                    ev = dict(t=t, kind="out_of_bounds", horizontal_m=horiz, altitude_m=alt, since=self._t_oob)
                    self.events.append(ev)
                    out.append(ev)
                    self._phase, self._t_phase = "surfacing", t
                    self._end(t, "out_of_bounds")
                    return out
            else:
                self._t_oob = None
        # ---------------- undervoltage (terminal voltage)
        V = float(battery_voltage)
        if V < cfg.undervoltage_V:
            if self._t_uv is None:
                self._t_uv = t
            elif t - self._t_uv >= cfg.undervoltage_persist_s - 1e-9:
                ev = dict(t=t, kind="undervoltage", V=V, since=self._t_uv)
                self.events.append(ev)
                out.append(ev)
                self._phase, self._t_phase = "surfacing", t
                self._end(t, "undervoltage")
        else:
            self._t_uv = None
        return out

    def _end(self, t, reason):
        self.ended = True
        self.end_reason = reason
        self.t_end = t

    def summary(self) -> dict:
        return dict(count=self.count, ended=self.ended, end_reason=self.end_reason, t_end=self.t_end,
                    F_peak_N=self.F_peak, impulse_peak_Ns=self.impulse_peak, events=list(self.events))


def _has_body(m, name) -> bool:
    try:
        m.body(name)
        return True
    except KeyError:
        return False
