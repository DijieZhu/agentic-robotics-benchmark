"""Rule-based grasp constraint for items held in the hand (the pattern of the benchmark's
pump-skid task).

One inactive weld (body1 = the palm body, body2 = world placeholder) is retargeted to an item when the capture
conditions hold while the gripper closes:
  * the item feature's axis (handle bar direction) within ``angle_deg`` of the finger hinge axis (palm z);
  * the feature's centre line inside the capture box of the palm frame: x in [x_lo, x_hi] (depth between the palm
    face and the fingertips), |y| <= y_half;
the weld keeps the CURRENT relative pose (no snapping).  It breaks when its load exceeds ``slip_force_N`` or
``slip_torque_Nm`` for ``slip_ticks`` consecutive checks (constraint rows: translational norm; rotational norm /
``WELD_ROT_ROWS`` as in the pump-skid cell).  While held, the finger geoms do not collide (they would fight the
weld) and, with a ``TendonGripper`` given, the fingers' closing limits are latched at their contact angles so they
do not swing through the handle.  Use for items in the hand only: perching on a handrail is pure contact (the rule
is rigid up to its threshold, while the real grip turns about the rail from 0.09 N m).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .rotations import quat_conj, quat_mul

__all__ = ["WELD_ROT_ROWS", "GraspRule"]

WELD_ROT_ROWS = 2.0


class GraspRule:
    """See module doc.  palm_body / eq_name / finger_geoms by name or id; gripper: optional TendonGripper."""

    def __init__(self, model, data, palm_body="arm_distal", finger_geoms=(), eq_name="grasp_weld", gripper=None,
                 x_lo: float = -0.115, x_hi: float = -0.040, y_half: float = 0.020, z_plane: float = -0.0497,
                 angle_deg: float = 20.0, slip_force_N: float = 6.4, slip_torque_Nm: float = 0.5, slip_ticks: int = 3):
        self.m, self.d = model, data
        self.eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
        if self.eq < 0:
            raise ValueError(f"weld {eq_name} not in the model (spacekit.gripper.rig_spec adds it)")
        self.palm = palm_body if isinstance(palm_body, (int, np.integer)) else mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, palm_body)
        self.fg = [g if isinstance(g, (int, np.integer)) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                   for g in finger_geoms]
        self.gripper = gripper
        self.box = (x_lo, x_hi, y_half, z_plane)
        self.cosmax = float(np.cos(np.deg2rad(angle_deg)))
        self.slip = (float(slip_force_N), float(slip_torque_Nm), int(slip_ticks))
        self.count = 0
        self.held = None
        self.saved_bits = None
        self.events = []
        self.last_load = (0.0, 0.0)

    def palm_frame(self):
        """(position, rotation) of the palm body."""
        d = self.d
        return d.xpos[self.palm].copy(), d.xmat[self.palm].reshape(3, 3).copy()

    def check_capture(self, center, axis, half_len: float = 0.25) -> bool:
        """True when the feature (centre, axis in world) satisfies the capture conditions."""
        p, R = self.palm_frame()
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        if abs(float(a @ R[:, 2])) < self.cosmax:
            return False
        x_lo, x_hi, y_half, zp = self.box
        q = p + R[:, 2] * zp
        t = float((q - np.asarray(center, float)) @ a)
        if abs(t) > half_len:
            return False
        c = np.asarray(center, float) + a * t
        v = R.T @ (c - p)
        return bool((x_lo <= v[0] <= x_hi) and (abs(v[1]) <= y_half))

    def attach(self, body2: int) -> None:
        """Activate the weld to body2 at the current relative pose."""
        m, d = self.m, self.d
        b1, b2 = self.palm, int(body2)
        R1 = d.xmat[b1].reshape(3, 3)
        pos = R1.T @ (d.xpos[b2] - d.xpos[b1])
        quat = quat_mul(quat_conj(d.xquat[b1]), d.xquat[b2])
        m.eq_obj2id[self.eq] = b2
        m.eq_data[self.eq, 0:3] = 0.0
        m.eq_data[self.eq, 3:6] = pos
        m.eq_data[self.eq, 6:10] = quat / np.linalg.norm(quat)
        m.eq_data[self.eq, 10] = 1.0
        d.eq_active[self.eq] = 1
        self.saved_bits = [(g, int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in self.fg]
        for g in self.fg:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
        if self.gripper is not None:
            self.gripper.latch(d)
        self.held = b2
        self.count = 0
        self.events.append((float(d.time), "attach", b2))

    def release(self, why: str = "release") -> None:
        """Deactivate the weld (restores finger collisions and tendon control)."""
        m, d = self.m, self.d
        d.eq_active[self.eq] = 0
        if self.saved_bits:
            for g, ct, ca in self.saved_bits:
                m.geom_contype[g] = ct
                m.geom_conaffinity[g] = ca
        self.saved_bits = None
        if self.gripper is not None:
            self.gripper.unlatch()
        self.events.append((float(d.time), why, self.held))
        self.held = None

    def load(self):
        """(force norm [N], torque norm [N m]) carried by the weld, None when inactive."""
        d = self.d
        if not d.eq_active[self.eq]:
            return None
        rows = [i for i in range(d.nefc) if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[i] == self.eq]
        if not rows:
            return 0.0, 0.0
        f = np.asarray(d.efc_force[rows], float)
        self.last_load = (float(np.linalg.norm(f[:3])), float(np.linalg.norm(f[3:6])) / WELD_ROT_ROWS)
        return self.last_load

    def check_slip(self) -> bool:
        """Release when the load exceeded the limits for slip_ticks consecutive calls; True if it broke now."""
        ld = self.load()
        if ld is None:
            return False
        F, T = ld
        fmax, tmax, ticks = self.slip
        if F > fmax or T > tmax:
            self.count += 1
            if self.count >= ticks:
                self.release("slip")
                return True
        else:
            self.count = 0
        return False
