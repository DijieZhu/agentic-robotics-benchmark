"""Analytic inverse kinematics of the UR arm on a PRIVATE model copy (oracle / planner helper), with the PandaKin API.

The UR geometry has a closed form (Hawkins 2013; ported from the maintainers' kinematics study): joints 2, 3, 4 are
parallel, so q1 follows from the wrist point's horizontal position and the lateral offset d4 (two shoulder branches),
q5 from the tool axis (two wrist branches), q6 from the tool's x / y axes, and q2, q3, q4 from a planar 2R arm (two
elbow branches): up to 8 solutions, each checked by forward kinematics (the closed form is verified, not trusted).
The parameters are read from the compiled model's body offsets (UR10e: d1 0.181, a2 0.613, a3 0.571,
d4 0.176 - 0.137 + 0.135 = 0.174, d5 0.120, d6 = the flange site's distance 0.11655; UR5e 0.163 / 0.425 / 0.392 /
0.134 / 0.100 / 0.0996), so the IK works on whatever UR `ur_robot.add_ur` attached. The fixed transforms between the
Menagerie frames and the DH frames (base: X, flange -> tcp: Y) and the joint offsets (the Menagerie base is turned 180
deg about z: q1 is offset by pi in the DH chain, or X = Rz(pi)) are calibrated against MuJoCo's FK of the model at
construction and the residual kept in `cal_err` (4e-16 m for the attached UR10e and UR5e).

Verified on the attached model with the tcp at the 2F-85 pinch point (the maintainers' robot smoke test, keys a / ur5e):
300 random postures in the joint ranges -> MuJoCo FK -> every analytic solution -> MuJoCo FK: within 3e-13 m and
7e-13 rad, the original posture among the solutions 300 / 300, 2-8 real solutions per pose (mean 7.3); det J equals
the singularity product below to 1e-13; 0.7-1.4 ms per `ik()` call on a loaded host. At the cell's eight work poses
all 8 branches lie inside the ranges and 4-8 of them are free of self-collision.

API (the PandaKin interface the reference program and planner use):
* `fk(q)` -> (p, R) of the tcp site by MuJoCo on the copy; `fk_T(q)` -> 4x4 by the closed-form chain;
* `ik(T)` -> every analytic solution (up to 8), angles wrapped to (-pi, pi], unsorted, not range-checked;
* `solve_one(p, R, q0, q_ref, margin=0.08)` -> (q, ok): the branch and the +-2 pi aliases nearest q0 (a weak pull to
  q_ref), every joint `margin` inside its range;
* `candidates(T, q_now)` -> [(score, q, margin)]: every branch inside the joint ranges (the alias nearest q_now; all
  aliases with aliases=True), scored as PandaKin (margin - 0.15 x joint distance - 0.3 x |delta q1|), best first;
* `solve(T, q_now, T_after=None)` -> (q, margin): the best candidate from which `T_after` (e.g. the descended pose) is
  reachable on the same branch with `min_margin_after` and no joint moving more than 1.2 rad;
* `posture_contacts(q)`: the interpenetrating contacts of the robot at q on the copy (the closed form ignores
  collisions; several branches fold the arm into itself); `candidates(..., clear=True)` / `solve(..., clear=True)` keep
  only interpenetration-free branches;
* `margin(q)`: the smallest of the joint-limit distance (rad) and three dimensionless singularity distances (~ rad
  near the singularity): wrist |sin q5|, elbow |sin q3|, shoulder |a2 c2 + a3 c23 + d5 s234| / (a2 + a3) (the wrist
  point's distance from the shoulder's singular plane, over the arm length; det J = a2 a3 s3 s5 (a2 c2 + a3 c23 +
  d5 s234)). With the UR's +-2 pi ranges the joint limits rarely bind; the singularities do. `margins(q)` gives each.
Joint ranges are the model's (UR10e and UR5e: +-2 pi, the elbow +-pi, Menagerie).
"""
import itertools
import math
import os

import numpy as np

__all__ = ["URKin", "JOINTS"]

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
TWO_PI = 2.0 * math.pi
FK_TOL = 1e-6          # m / rad: a closed-form solution is kept when its FK matches the target this well
DH_ALPHA = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]


def _dh(a, alpha, d, th):
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(th), math.sin(th)
    return np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(ax)
    if n < 1e-12:
        w, v = np.linalg.eigh(R)
        ax = v[:, int(np.argmin(np.abs(w - 1.0)))]
        return ax / np.linalg.norm(ax) * ang
    return ax / n * ang


def _wrap(v):
    return math.atan2(math.sin(v), math.cos(v))


class URKin(object):
    def __init__(self, xml_path=None, model=None, prefix="ur/", tcp_site=None):
        import mujoco
        self.mujoco = mujoco
        if model is not None:
            self.m = model
        else:
            xml_path = xml_path or os.path.join(os.environ.get("ARB_GIVEN_DIR", "/home/user/given"),
                                                "assembly_cell.xml")
            self.m = mujoco.MjModel.from_xml_path(xml_path)
        m = self.m
        self.prefix = prefix
        self.d = mujoco.MjData(m)
        self.jids = [m.joint(prefix + j).id for j in JOINTS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jids])
        self.vadr = np.array([m.jnt_dofadr[j] for j in self.jids])
        self.tcp = m.site(tcp_site or (prefix + "tcp")).id
        self.lower = np.array([m.jnt_range[j][0] for j in self.jids])
        self.upper = np.array([m.jnt_range[j][1] for j in self.jids])
        bp = {n: np.array(m.body_pos[m.body(prefix + n).id], float) for n in
              ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")}
        d6 = 0.1
        try:
            fs = m.site(prefix + "flange_ft").id
            mujoco.mj_kinematics(m, self.d)
            w3 = m.body(prefix + "wrist_3_link").id
            d6 = float(self.d.xmat[w3].reshape(3, 3).T @ (self.d.site_xpos[fs] - self.d.xpos[w3]) @ [0, 1.0, 0])
        except KeyError:
            pass
        # UR DH convention: a2, a3 negative (the arm at q = 0 stretches along -x of the DH base)
        self.d1 = float(bp["shoulder_link"][2])
        self.a2 = -abs(float(bp["forearm_link"][2]))
        self.a3 = -abs(float(bp["wrist_1_link"][2]))
        self.d4 = float(bp["upper_arm_link"][1] + bp["forearm_link"][1] + bp["wrist_2_link"][1])
        self.d5 = float(bp["wrist_3_link"][2])
        self.d6 = d6
        mujoco.mj_kinematics(m, self.d)
        b = m.body(prefix + "base").id
        self.B = np.eye(4)
        self.B[:3, :3] = self.d.xmat[b].reshape(3, 3)
        self.B[:3, 3] = self.d.xpos[b]
        self.Bi = _inv(self.B)
        self._calibrate()

    # ---------------------------------------------------------------- kinematics
    def fk_dh(self, th):
        """DH base -> DH flange for DH joint angles th"""
        a = [0.0, self.a2, self.a3, 0.0, 0.0, 0.0]
        d = [self.d1, 0.0, 0.0, self.d4, self.d5, self.d6]
        T = np.eye(4)
        for i in range(6):
            T = T @ _dh(a[i], DH_ALPHA[i], d[i], th[i])
        return T

    def fk_T(self, q):
        """world pose (4x4) of the tcp site by the closed-form chain"""
        return self.B @ self.X @ self.fk_dh(np.asarray(q, float) + self.off) @ self.Y

    def _model_T(self, q):
        self.d.qpos[self.qadr] = q
        self.mujoco.mj_kinematics(self.m, self.d)
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[self.tcp].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[self.tcp]
        return T

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        self.mujoco.mj_kinematics(self.m, self.d)
        self.mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.tcp].copy(), self.d.site_xmat[self.tcp].reshape(3, 3).copy()

    def _calibrate(self):
        """joint offsets and the fixed transforms X (base body -> DH base), Y (DH flange -> tcp) such that
        T_tcp(q) = B X fk_dh(q + off) Y; the expected Menagerie relations are tried first, then a search over offsets
        in multiples of pi/2 (as the study did)"""
        rng = np.random.default_rng(0)
        Qs = [rng.uniform(-3.0, 3.0, 6) for _ in range(6)]
        Tm = [self.Bi @ self._model_T(q) for q in Qs]
        T0 = self.Bi @ self._model_T(np.zeros(6))
        Rz = np.diag([-1.0, -1.0, 1.0, 1.0])

        def fit(X, off):
            off = np.asarray(off, float)
            Y = _inv(X @ self.fk_dh(off)) @ T0
            err = max(float(np.abs(T - X @ self.fk_dh(q + off) @ Y).max()) for q, T in zip(Qs, Tm))
            return err, off, X, Y

        best = None
        for X, off in ((np.eye(4), [math.pi, 0, 0, 0, 0, 0]), (Rz, [0] * 6), (np.eye(4), [0] * 6),
                       (Rz, [math.pi, 0, 0, 0, 0, 0])):
            r = fit(X, off)
            if best is None or r[0] < best[0]:
                best = r
            if r[0] < 1e-9:
                break
        if best[0] > 1e-9:
            for X in (np.eye(4), Rz):
                for off in itertools.product([0.0, math.pi / 2, -math.pi / 2, math.pi], repeat=6):
                    r = fit(X, off)
                    if r[0] < best[0]:
                        best = r
                    if r[0] < 1e-9:
                        break
                if best[0] < 1e-9:
                    break
        self.cal_err, self.off, self.X, self.Y = best
        self.Xi, self.Yi = _inv(self.X), _inv(self.Y)

    # ---------------------------------------------------------------- closed-form IK
    def ik(self, T_world_tcp, q6_ref=0.0):
        """every analytic solution (model joint angles wrapped to (-pi, pi]) for a world tcp pose, FK-checked. At the
        wrist singularity (sin q5 = 0) q4 and q6 are coupled: q6 is set to `q6_ref`."""
        T_world_tcp = np.asarray(T_world_tcp, float)
        Tdh = self.Xi @ self.Bi @ T_world_tcp @ self.Yi            # DH base -> DH flange
        d1, a2, a3, d4, d5, d6 = self.d1, self.a2, self.a3, self.d4, self.d5, self.d6
        p05 = Tdh @ np.array([0.0, 0.0, -d6, 1.0])
        px, py = p05[0], p05[1]
        Rxy = math.hypot(px, py)
        if Rxy < abs(d4) - 1e-12:
            return []
        psi = math.atan2(py, px)
        phi = math.acos(max(-1.0, min(1.0, d4 / Rxy)))
        X0, Y0, p06 = Tdh[:3, 0], Tdh[:3, 1], Tdh[:3, 3]
        raw = []
        for s1 in (1, -1):
            t1 = psi + s1 * phi + math.pi / 2
            c1, sn1 = math.cos(t1), math.sin(t1)
            c5 = (p06[0] * sn1 - p06[1] * c1 - d4) / d6
            if abs(c5) > 1.0 + 1e-9:
                continue
            c5 = max(-1.0, min(1.0, c5))
            for s5 in (1, -1):
                t5 = s5 * math.acos(c5)
                s5v = math.sin(t5)
                if abs(s5v) < 1e-9:
                    t6 = float(q6_ref) + self.off[5]
                else:
                    t6 = math.atan2((-Y0[0] * sn1 + Y0[1] * c1) / s5v, (X0[0] * sn1 - X0[1] * c1) / s5v)
                T14 = _inv(_dh(0.0, math.pi / 2, d1, t1)) @ Tdh @ _inv(_dh(0.0, 0.0, d6, t6)) @ \
                    _inv(_dh(0.0, -math.pi / 2, d5, t5))
                T13 = T14 @ _inv(_dh(0.0, math.pi / 2, d4, 0.0))
                px3, py3 = T13[0, 3], T13[1, 3]
                L2 = px3 * px3 + py3 * py3
                c3 = (L2 - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
                if abs(c3) > 1.0 + 1e-9:
                    continue
                c3 = max(-1.0, min(1.0, c3))
                for s3 in (1, -1):
                    t3 = s3 * math.acos(c3)
                    t2 = math.atan2(py3, px3) - math.atan2(a3 * math.sin(t3), a2 + a3 * math.cos(t3))
                    T34 = _inv(_dh(a2, 0.0, 0.0, t2) @ _dh(a3, 0.0, 0.0, t3)) @ T14
                    t4 = math.atan2(T34[1, 0], T34[0, 0])
                    raw.append([t1, t2, t3, t4, t5, t6])
        out = []
        for th in raw:
            q = np.array([_wrap(th[i] - self.off[i]) for i in range(6)])
            T = self.fk_T(q)
            if np.abs(T[:3, 3] - T_world_tcp[:3, 3]).max() > FK_TOL or \
                    np.abs(T[:3, :3] - T_world_tcp[:3, :3]).max() > FK_TOL:
                continue
            if any(np.max(np.abs(np.array([_wrap(v) for v in (q - o)]))) < 1e-9 for o in out):
                continue                                           # the s3 = 0 / s5 = 0 doubles
            out.append(q)
        return out

    # ---------------------------------------------------------------- branches inside the ranges
    def _alias(self, v, ref, lo, hi):
        best = None
        for k in (-2, -1, 0, 1, 2):
            x = v + k * TWO_PI
            if lo - 1e-12 <= x <= hi + 1e-12 and (best is None or abs(x - ref) < abs(best - ref)):
                best = x
        return best

    def nearest(self, q, q_ref, margin=0.0):
        """q with every joint moved by multiples of 2 pi nearest q_ref inside [lower + margin, upper - margin]; None
        when some joint has no such alias"""
        out = np.zeros(6)
        for i in range(6):
            x = self._alias(float(q[i]), float(q_ref[i]), self.lower[i] + margin, self.upper[i] - margin)
            if x is None:
                return None
            out[i] = x
        return out

    def all_aliases(self, q, margin=0.0):
        per = []
        for i in range(6):
            xs = [q[i] + k * TWO_PI for k in (-2, -1, 0, 1, 2)]
            per.append([x for x in xs if self.lower[i] + margin - 1e-12 <= x <= self.upper[i] - margin + 1e-12])
        return [np.array(c) for c in itertools.product(*per)]

    def solve_one(self, p_t, R_t, q0, q_ref=None, iters=None, w_ref=0.1, margin=0.08):
        """(q, ok): the analytic solution for the tcp pose (p_t, R_t) nearest q0 (plus w_ref x the squared distance
        to q_ref), every joint `margin` inside its range. `iters` is accepted for the PandaKin signature (unused)."""
        q0 = np.asarray(q0, float)
        q_ref = q0 if q_ref is None else np.asarray(q_ref, float)
        T = np.eye(4)
        T[:3, :3] = np.asarray(R_t, float)
        T[:3, 3] = np.asarray(p_t, float)
        best = None
        for q in self.ik(T, q6_ref=float(q0[5])):
            qw = self.nearest(q, q0, margin)
            if qw is None:
                continue
            cost = float(np.sum((qw - q0) ** 2) + w_ref * np.sum((qw - q_ref) ** 2))
            if best is None or cost < best[0]:
                best = (cost, qw)
        if best is None:
            return np.clip(q0, self.lower + margin, self.upper - margin), False
        return best[1], True

    def candidates(self, T_tcp, q_now, extra_seeds=(), aliases=False, clear=False):
        """[(score, q, margin)] for every branch inside the joint ranges, best first (PandaKin's score). `extra_seeds`
        is accepted for the PandaKin signature (the closed form needs no seeds). clear=True drops the branches whose
        posture interpenetrates (the robot with itself or with anything in the copy: `posture_contacts`)."""
        q_now = np.asarray(q_now, float)
        out = []
        for q in self.ik(T_tcp, q6_ref=float(q_now[5])):
            reps = self.all_aliases(q) if aliases else [self.nearest(q, q_now)]
            for qw in reps:
                if qw is None or any(np.max(np.abs(qw - o[1])) < 1e-6 for o in out):
                    continue
                if clear and self.posture_contacts(qw):
                    continue
                mrg = self.margin(qw)
                dist = float(np.linalg.norm(qw - q_now))
                out.append((mrg - 0.15 * dist - 0.3 * abs(qw[0] - q_now[0]), qw, mrg))
        out.sort(key=lambda o: -o[0])
        return out

    def solve(self, T_tcp, q_now, extra_seeds=(), T_after=None, min_margin_after=0.12, clear=False):
        """best configuration for the tcp pose; with `T_after` (a pose reached from there by a short Cartesian move)
        the branch must reach it too, keeping `min_margin_after` and moving no joint more than 1.2 rad; clear=True:
        both postures free of interpenetration"""
        for _score, q, mrg in self.candidates(T_tcp, q_now, extra_seeds, clear=clear):
            if T_after is not None:
                q2, ok = self.solve_one(T_after[:3, 3], T_after[:3, :3], q, q, margin=0.0)
                if not ok:
                    continue
                if self.margin(q2) < min_margin_after or np.max(np.abs(q2 - q)) > 1.2:
                    continue
                if clear and self.posture_contacts(q2):
                    continue
            return q, mrg
        return None, 0.0

    # ---------------------------------------------------------------- posture collisions on the copy
    def posture_contacts(self, q, margin=0.0, ignore_bodies=()):
        """[(geom1, geom2, distance)] of the contacts deeper than `margin` that involve a robot geom (prefix) at q
        (the copy's other joints as they are): self-collision (the UR folds into itself on several of the 8 branches:
        forearm / wrist_1 into the shoulder, the gripper into the forearm) and robot-environment."""
        m, d = self.m, self.d
        if not hasattr(self, "_robot_geoms"):
            self._robot_geoms = set(g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(self.prefix))
        skip = set(m.body(b).id for b in ignore_bodies)
        d.qpos[self.qadr] = q
        self.mujoco.mj_kinematics(m, d)
        self.mujoco.mj_collision(m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if c.dist >= -margin or not (g1 in self._robot_geoms or g2 in self._robot_geoms):
                continue
            if m.geom_bodyid[g1] in skip or m.geom_bodyid[g2] in skip:
                continue
            out.append((m.geom(g1).name or m.body(m.geom_bodyid[g1]).name,
                        m.geom(g2).name or m.body(m.geom_bodyid[g2]).name, float(c.dist)))
        return out

    # ---------------------------------------------------------------- margins
    def margins(self, q):
        q = np.asarray(q, float)
        th = q + self.off
        c2, c23, s234 = math.cos(th[1]), math.cos(th[1] + th[2]), math.sin(th[1] + th[2] + th[3])
        return {"limits_rad": float(np.min(np.minimum(q - self.lower, self.upper - q))),
                "wrist": abs(math.sin(th[4])),
                "elbow": abs(math.sin(th[2])),
                "shoulder": abs(self.a2 * c2 + self.a3 * c23 + self.d5 * s234) / (abs(self.a2) + abs(self.a3))}

    def margin(self, q):
        return float(min(self.margins(q).values()))

    def pose_error(self, q, T):
        """(position error m, rotation error rad) of the MuJoCo FK of q against a 4x4 target; the angle from
        ||R1 - R2||_F = 2 sqrt(2) sin(angle / 2), exact down to 1e-15 (acos of the trace stops at ~1e-8)"""
        p, R = self.fk(q)
        dR = float(np.linalg.norm(np.asarray(T)[:3, :3] - R))
        return float(np.linalg.norm(p - T[:3, 3])), 2.0 * math.asin(min(1.0, dR / (2.0 * math.sqrt(2.0))))
