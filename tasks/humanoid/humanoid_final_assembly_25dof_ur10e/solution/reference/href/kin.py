"""The arm's kinematics on the agent's side: the given robot models (the UR10e and the wide gripper, the cell library's
own attachment code arm.assembly_rov.gripper_wide.add_ur_wide) compiled alone, the closed-form UR inverse kinematics of
the cell library (arm.assembly_quad.ur_ik.URKin) on that model. Calibrated against the cell: the forward kinematics of
the reported joint state must give the reported tcp pose (1e-6 m, 1e-6 rad), else the program stops."""
import math
import os

import numpy as np

from . import tf
from .errors import ReferenceFailure


class ArmKin(object):
    def __init__(self, info, models_dir=None):
        import mujoco
        from arm.assembly_rov import gripper_wide as GW
        from arm.assembly_rov import ur_robot as UR
        from arm.assembly_quad.ur_ik import URKin
        self.mj = mujoco
        spec = mujoco.MjSpec()
        spec.compiler.degree = False
        rb = info["robot"]
        mdir = models_dir or os.environ.get("ARB_ROBOT_MODELS") or UR.ROBOT_MODELS
        self.names = GW.add_ur_wide(spec, {"model": "ur10e", "pos": [float(v) for v in rb["base_pos"]], "yaw_deg": 0.0,
                                           "models_dir": mdir})
        self.m = spec.compile()
        self.kin = URKin(model=self.m, prefix=self.names["prefix"], tcp_site=self.names["tcp_site"])
        lo, hi = self.kin.lower, self.kin.upper
        want = np.array(rb["joint_range"], float)
        if np.max(np.abs(np.column_stack([lo, hi]) - want)) > 1e-4:
            raise ReferenceFailure("start", "the arm model's joint ranges differ from describe()'s",
                                   {"model": np.column_stack([lo, hi]).tolist(), "describe": want.tolist()})
        self.lower, self.upper = lo, hi
        self.d = mujoco.MjData(self.m)

    def fk(self, q):
        p, R = self.kin.fk(np.asarray(q, float))
        return p, R

    def fk_T(self, q):
        p, R = self.fk(q)
        return tf.T_of(p, R=R)

    def calibrate(self, st, tol=1e-6):
        q = np.array(st["robot"]["q"], float)
        p, R = self.fk(q)
        Rr = tf.quat_to_mat(st["robot"]["tcp_quat"])
        dp = float(np.linalg.norm(p - np.array(st["robot"]["tcp_pos"], float)))
        da = tf.angle(R.T @ Rr)
        return dp, da, (dp <= tol and da <= tol)

    def ik(self, T, q_near, margin=0.05):
        """the closed-form solution nearest q_near (every joint inside its range by `margin`), or None"""
        q, ok = self.kin.solve_one(T[:3, 3], T[:3, :3], np.asarray(q_near, float), margin=margin)
        if not ok:
            return None
        Tq = self.fk_T(q)
        if np.linalg.norm(Tq[:3, 3] - T[:3, 3]) > 1e-6 or tf.angle(Tq[:3, :3].T @ T[:3, :3]) > 1e-6:
            return None
        return q

    def ik_branch(self, T, posture, centre=None, margin=0.05):
        """the closed-form solution on the branch of `posture` (the smallest wrapped distance of joints 2-5 to it: the
        shoulder, elbow and wrist branches of the cell's home posture), with the pan and the last wrist joint taken at
        their alias nearest `centre` (default the posture itself), every joint inside its range by `margin`; or None"""
        posture = np.asarray(posture, float)
        centre = posture if centre is None else np.asarray(centre, float)
        best = None
        for q in self.kin.ik(T, q6_ref=float(centre[5])):
            w = np.array([tf.wrap(float(a - b)) for a, b in zip(q[1:5], posture[1:5])])
            cost = float(np.sum(w ** 2))
            qq = np.array(q, float)
            ok = True
            for i in range(6):
                cands = [qq[i] + k * 2.0 * math.pi for k in (-2, -1, 0, 1, 2)]
                cands = [c for c in cands if self.lower[i] + margin <= c <= self.upper[i] - margin]
                if not cands:
                    ok = False
                    break
                ref = centre[i] if i in (0, 5) else posture[i]
                qq[i] = min(cands, key=lambda c: abs(c - ref))
            if ok and (best is None or cost < best[0]):
                best = (cost, qq)
        if best is None:
            return None
        Tq = self.fk_T(best[1])
        if np.linalg.norm(Tq[:3, 3] - T[:3, 3]) > 1e-6 or tf.angle(Tq[:3, :3].T @ T[:3, :3]) > 1e-6:
            return None
        return best[1]

    def path_points(self, q0, q1, offsets, n=48):
        """world positions along the joint-space segment q0 -> q1 (linear in q) of points given in the tcp frame"""
        out = []
        for i in range(n + 1):
            s = i / float(n)
            T = self.fk_T(q0 + s * (q1 - q0))
            out.append([T[:3, :3] @ o + T[:3, 3] for o in offsets])
        return np.array(out)                     # (n+1, k, 3)
