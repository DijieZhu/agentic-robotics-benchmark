"""Simulation log for replay rendering: a read-only Recorder on an UnderwaterSim and the SimLog container.

    rec = Recorder(sim)                      # every physics step (default); records row 0 now
    for k in range(n):
        ...; sim.step(); rec.step()          # after EVERY physics step (integrates the true rotor angle)
        rec.set_label("forward"); rec.set_lights([1, 1, 1, 1])
    log = rec.finish(); log.save("run.npz"); log = SimLog.load("run.npz")

The recorder never writes to the simulator (no MjData / model / RNG access other than reading arrays), so a run with the
recorder is bit-for-bit the run without it (tested in uwkit_tests/viz/test_physics_unchanged.py).

Row fields (uniform rate log_hz; row i at t[i]):
  t, qpos (full physics qpos), qvel6 (hull free-joint qvel: world linear velocity, body angular velocity),
  thrust [N], rpm (signed, sim state), u_cmd, u_eff, phi_true (true rotor angle about axis_B at t[i], integrated at the
  physics step from the rpm held during each step), water_w (current at the hull CoM, world axes, as seen by the last
  physics step), battery_V, lights (0..1 per lamp), label (index into meta['labels'])
meta: config, log_hz, dt, sites, spin_sign_forward, joints {name: (type, qposadr)}, hull joint, labels, version.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco

from .rotors import RotorSet, RPM_TO_RAD_S

LOG_VERSION = 1
ARRAYS = ("t", "qpos", "qvel6", "thrust", "rpm", "u_cmd", "u_eff", "phi_true", "water_w", "battery_V", "lights", "label")


@dataclass
class SimLog:
    meta: dict
    t: np.ndarray
    qpos: np.ndarray
    qvel6: np.ndarray
    thrust: np.ndarray
    rpm: np.ndarray
    u_cmd: np.ndarray
    u_eff: np.ndarray
    phi_true: np.ndarray
    water_w: np.ndarray
    battery_V: np.ndarray
    lights: np.ndarray
    label: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.t)

    @property
    def log_hz(self):
        return float(self.meta["log_hz"])

    def labels_at(self, i):
        return self.meta["labels"][int(self.label[i])]

    def rows_for_fps(self, fps):
        """Row index of every video frame at fps: an exact stride if log_hz is a multiple of fps, else the nearest row
        (at the default log rate = physics rate 500 Hz the frame time is off by <= 1 ms)."""
        k = self.log_hz / float(fps)
        if abs(k - round(k)) < 1e-9:
            return np.arange(0, self.n, int(round(k)))
        t_end = (self.n - 1) / self.log_hz
        return np.unique(np.minimum(np.round(np.arange(0.0, t_end + 1e-9, 1.0 / fps) * self.log_hz).astype(int),
                                    self.n - 1))

    def save(self, path):
        arrs = {k: getattr(self, k) for k in ARRAYS}
        for k, v in self.extra.items():
            arrs["extra_" + k] = np.asarray(v)
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta)), **arrs)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        kw = {k: z[k] for k in ARRAYS}
        extra = {k[6:]: z[k] for k in z.files if k.startswith("extra_")}
        return cls(meta=meta, extra=extra, **kw)

    def slice(self, i0, i1):
        kw = {k: getattr(self, k)[i0:i1] for k in ARRAYS}
        return SimLog(meta=dict(self.meta), extra={k: v[i0:i1] for k, v in self.extra.items()}, **kw)


def joint_table(model) -> dict:
    """{joint name: (type, qposadr, nq)} of every named joint (free 7, ball 4, slide / hinge 1)."""
    nqs = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4,
           int(mujoco.mjtJoint.mjJNT_SLIDE): 1, int(mujoco.mjtJoint.mjJNT_HINGE): 1}
    out = {}
    for j in range(model.njnt):
        nm = model.joint(j).name
        if nm:
            t = int(model.jnt_type[j])
            out[nm] = [t, int(model.jnt_qposadr[j]), nqs[t]]          # list: survives the JSON round trip
    return out


class Recorder:
    """Read-only logger of an UnderwaterSim (see module docstring)."""

    def __init__(self, sim, log_hz: Optional[float] = None, n_lights: int = 4, rotors: Optional[RotorSet] = None,
                 config: Optional[str] = None, attach: bool = False):
        """attach=True registers the recorder as a step callback of the simulator (UnderwaterSim.add_step_callback):
        every physics step is recorded, also inside sim.step(n > 1); do not call rec.step() yourself then.
        Lamp levels: rec.set_lights(...) if called, otherwise the simulator's own sim.lights (sim.set_lights)."""
        self.sim = sim
        dt = sim.dt
        log_hz = 1.0 / dt if log_hz is None else float(log_hz)      # default: every physics step
        k = 1.0 / (log_hz * dt)
        if abs(k - round(k)) > 1e-6:
            raise ValueError(f"log_hz {log_hz} must divide the physics rate {1 / dt}")
        self.k = int(round(k))
        self.log_hz = float(log_hz)
        cfg = config or ((sim.meta.get("reference") or {}).get("config") or "mission")
        self.rotors = rotors or RotorSet(cfg, sites=list(sim.thr_names))
        if self.rotors.sites != list(sim.thr_names):
            raise ValueError("rotor set and simulator thrusters differ")
        self.phi = np.zeros(self.rotors.n)
        self.lights = np.zeros(n_lights)
        self._lights_explicit = False
        self.labels = [""]
        self._label = 0
        self.rows = {k: [] for k in ARRAYS}
        self._count = 0
        self.meta = dict(version=LOG_VERSION, config=cfg, log_hz=self.log_hz, dt=dt, sites=list(sim.thr_names),
                         spin_sign_forward=[int(s) for s in self.rotors.spin], joints=joint_table(sim.model),
                         hull_joint=sim.model.joint(int(sim.free_jnt)).name, n_lights=n_lights, labels=self.labels)
        self._record()
        self._attached = None
        if attach:
            self._attached = sim.add_step_callback(lambda s: self.step())

    def detach(self):
        """Stop recording through the step callback (attach=True)."""
        if self._attached is not None:
            self.sim.remove_step_callback(self._attached)
            self._attached = None

    def set_label(self, text: str):
        if text not in self.labels:
            self.labels.append(text)
        self._label = self.labels.index(text)

    def set_lights(self, levels):
        self.lights = np.clip(np.asarray(levels, float).reshape(-1), 0.0, 1.0).copy()
        self._lights_explicit = True

    def step(self, n: int = 1):
        """Call after every physics step (or after sim.step(n=n) with the same n): integrates the true rotor angle."""
        w = self.rotors.spin * self.sim.thrusters.rpm * RPM_TO_RAD_S
        for _ in range(int(n)):
            self.phi = self.phi + w * self.sim.dt
            self._count += 1
            if self._count % self.k == 0:
                self._record()

    def _record(self):
        s, d = self.sim, self.sim.data
        th = s.thrusters
        L = s.hydro.last
        wat = np.array(L["vc_w"][s.ih], float) if L and "vc_w" in L else np.zeros(3)
        R = self.rows
        R["t"].append(float(d.time))
        R["qpos"].append(np.array(d.qpos, float))
        R["qvel6"].append(np.array(d.qvel[s.vadr:s.vadr + 6], float))
        R["thrust"].append(np.array(th.thrust, float))
        R["rpm"].append(np.array(th.rpm, float))
        R["u_cmd"].append(np.array(th.u_cmd, float))
        R["u_eff"].append(np.array(th.u_eff, float))
        R["phi_true"].append(self.phi.copy())
        R["water_w"].append(wat)
        R["battery_V"].append(float(s.battery.voltage))
        sl = getattr(s, "lights", None)
        if not getattr(self, "_lights_explicit", False) and sl is not None and len(sl) == len(self.lights):
            self.lights = np.asarray(sl, float).copy()
        R["lights"].append(self.lights.copy())
        R["label"].append(self._label)

    def finish(self) -> SimLog:
        kw = {k: np.asarray(v) for k, v in self.rows.items()}
        kw["label"] = kw["label"].astype(np.int32)
        meta = dict(self.meta)
        meta["labels"] = list(self.labels)
        return SimLog(meta=meta, **kw)
