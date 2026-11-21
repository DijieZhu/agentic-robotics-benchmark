"""Visual twin of the reference build v1 for people-facing renders (v0.3): the display-only joints of the model --
two impellers and, per nozzle, two flaps, the drive gear and the idler -- and the status lights of the full-detail
visual set, driven from the firmware / plant state.

The twin never writes into the physics data: ``sample(rig)`` reads the plant (impeller speeds, the flap angles the
servos actually reached in response to the allocator's commands) and the executive state into a small vector that a
recorder stores; ``apply(model, data, sample, phase)`` writes that state into a RENDER copy (qpos of the display
joints, and -- render model only -- the rgba of the light geoms).  So the physics state is bit-identical with and
without the twin.

Mapping (reference build v1): plant nozzle k (firmware order; 0-5 = module 0 = +y = starboard, site ``nozzle_1_{k+1}``)
-> joints ``pm_s_n{k+1}_{flap_a,flap_b,drive,idler}``, 6-11 -> ``pm_p_n{k-5}_*``; flap joints = the plant's flap
angle beyond closed, drive gear = 0.5 x (the 12T / 24T servo gear), idler = 0.6 x; impellers ``pm_s_impeller`` /
``pm_p_impeller``.  The impeller spin is drawn slowed down: ``display_rev_s`` revolutions per second of video at the
reference speed (2800 rpm), proportional to the true speed, because 40 rev/s cannot be shown at video frame rates
(a staged visualisation; the README of a render must say so).  Status lights (full-detail set only, a staged
convention of this project, not the real light code): propulsion-module light arcs brighten with the fan speed;
the status LEDs show the executive state (moving amber, holding white, docked / charging green, off dim).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

__all__ = ["VisualTwin", "STATE_CODES"]

STATE_CODES = {"idle": 0, "preparing": 1, "moving": 2, "stopping": 3, "stopped": 4, "test": 5}
_DOCKED, _PERCHED = 8, 16


class VisualTwin:
    """Display-joint and light driver (see module doc).  model: a model that contains the reference build v1
    (names with ``prefix``); missing joints / lights are skipped (``n_joints``, ``n_lights``)."""

    def __init__(self, model, prefix: str = "", display_rev_s: float = 1.5, ref_speed_rad_s: float = 2800 * 2 * math.pi / 60):
        self.prefix = prefix
        self.rev_s = float(display_rev_s)
        self.w_ref = float(ref_speed_rad_s)
        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)
        self.imp = [jid("pm_s_impeller"), jid("pm_p_impeller")]
        self.noz = []
        for k in range(12):
            side, i = ("s", k + 1) if k < 6 else ("p", k - 5)
            self.noz.append([(jid(f"pm_{side}_n{i}_{part}"), c) for part, c in
                             (("flap_a", 1.0), ("flap_b", 1.0), ("drive", 0.5), ("idler", 0.6))])
        self.imp_adr = [int(model.jnt_qposadr[j]) if j >= 0 else -1 for j in self.imp]
        self.noz_adr = [[(int(model.jnt_qposadr[j]) if j >= 0 else -1, c, (float(model.jnt_range[j][0]), float(model.jnt_range[j][1])) if j >= 0 else (0, 0))
                         for j, c in row] for row in self.noz]
        self.n_joints = sum(a >= 0 for a in self.imp_adr) + sum(a >= 0 for row in self.noz_adr for a, _, _ in row)
        gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        self.arcs = [g for g in range(model.ngeom) if "_led_arc" in gname(g)]
        self.status = [g for g in range(model.ngeom) if gname(g).startswith(prefix + "v_status_leds") and gname(g).endswith("__c1")]
        self.bars = [g for g in range(model.ngeom) if gname(g).startswith(prefix + "v_light_bar") and gname(g).endswith("__c1")]
        self.n_lights = len(self.arcs) + len(self.status) + len(self.bars)
        self._rgba0 = {g: model.geom_rgba[g].copy() for g in self.arcs + self.status + self.bars}

    # -- recording side (reads only) ---------------------------------------------------------------------------------
    @staticmethod
    def sample(rig, layer=None) -> np.ndarray:
        """State vector of one instant: [impeller speed x2 (rad/s), flap angle x12 (rad), state code (+8 docked,
        +16 perched), fans powered x2].  Reads the plant / executive; writes nothing."""
        th = np.concatenate([m.nozzle_theta for m in rig.plant.modules])
        code = STATE_CODES.get(getattr(rig.firmware, "state", "idle"), 0)
        if layer is not None:
            code += _DOCKED * bool(getattr(layer, "docked", False)) + _PERCHED * bool(getattr(layer, "perched", False))
        pw = [float(m.powered) for m in rig.plant.modules]
        return np.concatenate([rig.plant.impeller_speed, th, [float(code)], pw])

    SIZE = 17

    def advance_phase(self, phase, sample, dt_video: float) -> np.ndarray:
        """Impeller display phases after dt_video seconds of video (slowed spin, see module doc)."""
        w = np.asarray(sample[:2], float)
        return np.asarray(phase, float) + 2 * math.pi * self.rev_s * (w / self.w_ref) * float(dt_video)

    # -- render side (writes a render copy only) ---------------------------------------------------------------------
    def apply(self, model, data, sample, phase=(0.0, 0.0), lights: bool = True) -> None:
        """Write the display joints (qpos of ``data``: a render copy) and, if ``lights``, the light rgba of ``model``
        (a render model) from a sample; call mj_kinematics / mj_forward on ``data`` afterwards."""
        s = np.asarray(sample, float)
        for k, a in enumerate(self.imp_adr):
            if a >= 0:
                data.qpos[a] = math.fmod(float(phase[k]), 2 * math.pi)
        for k, row in enumerate(self.noz_adr):
            th = float(s[2 + k])
            for a, c, (lo, hi) in row:
                if a >= 0:
                    data.qpos[a] = min(max(c * th, lo), hi)
        if lights and self.n_lights:
            self._lights(model, s)

    def _lights(self, model, s) -> None:
        w = float(np.clip(np.mean(np.abs(s[:2])) / self.w_ref, 0.0, 1.0))
        code = int(s[14])
        docked, perched, st = bool(code & _DOCKED), bool(code & _PERCHED), code & 7
        for g in self.arcs:
            model.geom_rgba[g] = [0.10 + 0.25 * w, 0.20 + 0.70 * w, 0.25 + 0.75 * w, 1.0]
        col = ((0.15, 0.95, 0.30) if docked else (0.95, 0.65, 0.10) if st == STATE_CODES["moving"] else
               (0.90, 0.90, 0.90) if st in (STATE_CODES["stopping"], STATE_CODES["stopped"]) or perched else (0.20, 0.25, 0.22))
        for g in self.status:
            model.geom_rgba[g] = [*col, 1.0]
        for g in self.bars:
            model.geom_rgba[g] = [0.30 + 0.5 * w, 0.55 + 0.4 * w, 0.95, 0.85]

    def reset_lights(self, model) -> None:
        """Restore the light geoms' colours."""
        for g, c in self._rgba0.items():
            model.geom_rgba[g] = c
