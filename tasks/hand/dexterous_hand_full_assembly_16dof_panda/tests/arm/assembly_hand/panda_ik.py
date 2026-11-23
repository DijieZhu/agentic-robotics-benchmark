"""Posture-aware inverse kinematics of the Panda on the anchor's PRIVATE cell model (oracle / anchor helper).

Information tier: the model is the episode's physics model rebuilt by the anchor from its own copy of the GT kit and the
privileged layout (`arm.assembly_info.cell_info.episode_model`); the original six-servo helper read the given cell XML.

The grader's Cartesian mode is a per-tick differential IK: a large hand yaw is spread over joints 1, 3, 5
and 7 and can end at a joint limit. For large reorientations the oracle therefore solves the target
configuration itself (damped least squares with a null-space pull towards a reference posture and the
joint ranges) and commands it in joint mode; the kinematic model is the agent-visible
`assembly_cell.xml` (ARB_GIVEN_DIR, default /home/user/given). Only the Panda joints and the `tcp` site
are used.
"""

import math
import os

import numpy as np

JOINTS = ["panda/joint%d" % i for i in range(1, 8)]
# comfortable top-down postures over the fixture / trays (seeds for the IK, null-space references)
REFERENCE_POSTURES = [
    [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, -0.785],
    [0.0, 0.3, 0.0, -1.9, 0.0, 2.2, -0.785],
    [0.0, -0.4, 0.0, -2.3, 0.0, 1.9, -0.785],
    [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785],
    [0.0, 0.3, 0.0, -1.9, 0.0, 2.2, 0.785],
]


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


class PandaKin(object):
    def __init__(self, xml_path=None, model=None):
        import mujoco
        self.mujoco = mujoco
        if model is not None:
            self.m = model
        else:
            xml_path = xml_path or os.path.join(os.environ.get("ARB_GIVEN_DIR", "/home/user/given"), "assembly_cell.xml")
            self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.jids = [self.m.joint(j).id for j in JOINTS]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in self.jids])
        self.vadr = np.array([self.m.jnt_dofadr[j] for j in self.jids])
        self.tcp = self.m.site("tcp").id
        self.lower = np.array([self.m.jnt_range[j][0] for j in self.jids])
        self.upper = np.array([self.m.jnt_range[j][1] for j in self.jids])

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        self.mujoco.mj_kinematics(self.m, self.d)
        self.mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.tcp].copy(), self.d.site_xmat[self.tcp].reshape(3, 3).copy()

    def _jac(self):
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        self.mujoco.mj_jacSite(self.m, self.d, jp, jr, self.tcp)
        return np.vstack([jp[:, self.vadr], jr[:, self.vadr]])

    def solve_one(self, p_t, R_t, q0, q_ref, iters=250, w_ref=0.08, margin=0.08):
        lo, hi = self.lower + margin, self.upper - margin
        q = np.clip(np.asarray(q0, float).copy(), lo, hi)
        for _ in range(iters):
            p, R = self.fk(q)
            e = np.concatenate([p_t - p, _rotvec(R_t @ R.T)])
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-3:
                return q, True
            J = self._jac()
            dq = J.T @ np.linalg.solve(J @ J.T + 0.005 * np.eye(6), e)
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (w_ref * (np.asarray(q_ref) - q))
            q = np.clip(q + 0.6 * dq, lo, hi)
        p, R = self.fk(q)
        e = np.concatenate([p_t - p, _rotvec(R_t @ R.T)])
        return q, bool(np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < 1e-2)

    def candidates(self, T_tcp, q_now, extra_seeds=()):
        """Distinct IK solutions for the tcp pose from several seeds, each scored by joint-limit margin minus
        the joint-space distance from `q_now` (large joint-1 swings penalised). Sorted best first."""
        seeds = [np.asarray(q_now, float)] + [np.asarray(s, float) for s in extra_seeds] + [np.asarray(r, float) for r in REFERENCE_POSTURES]
        out = []
        for s in seeds:
            for ref in ([s] + [np.asarray(r, float) for r in REFERENCE_POSTURES[:3]]):
                q, ok = self.solve_one(T_tcp[:3, 3], T_tcp[:3, :3], s, ref)
                if not ok:
                    continue
                if any(np.max(np.abs(q - o[1])) < 0.05 for o in out):
                    continue
                mrg = float(np.min(np.minimum(q - self.lower, self.upper - q)))
                dist = float(np.linalg.norm(q - np.asarray(q_now, float)))
                score = mrg - 0.15 * dist - 0.3 * abs(q[0] - q_now[0])
                out.append((score, q, mrg))
        out.sort(key=lambda o: -o[0])
        return out

    def solve(self, T_tcp, q_now, extra_seeds=(), T_after=None, min_margin_after=0.12):
        """Best configuration for the tcp pose. With `T_after` (a pose the hand will reach from there by a short
        Cartesian move, e.g. the descended pose) the branch must also keep `min_margin_after` of joint range at
        that pose, so the descent does not run into a joint limit half way."""
        for score, q, mrg in self.candidates(T_tcp, q_now, extra_seeds):
            if T_after is not None:
                q2, ok = self.solve_one(T_after[:3, 3], T_after[:3, :3], q, q, margin=0.0)
                if not ok:
                    continue
                mrg2 = float(np.min(np.minimum(q2 - self.lower, self.upper - q2)))
                if mrg2 < min_margin_after or np.max(np.abs(q2 - q)) > 1.2:
                    continue
            return q, mrg
        return None, 0.0

    def margin(self, q):
        return float(np.min(np.minimum(np.asarray(q) - self.lower, self.upper - np.asarray(q))))

    # ---------------------------------------------------------------- collision-aware path checks on the private model
    def set_parts(self, parts):
        """Place the loose bodies of the private model at the poses `parts` reports ({name: {pos, quat}}; bodies the model
        does not have, or that are not free, are skipped) -- the standing compound and the loose servo become obstacles."""
        for name, p in parts.items():
            try:
                j = self.m.joint("free_" + name)
            except KeyError:
                continue
            adr = self.m.jnt_qposadr[j.id]
            self.d.qpos[adr:adr + 3] = p["pos"]
            self.d.qpos[adr + 3:adr + 7] = p["quat"]

    def _subtree(self, name):
        m = self.m
        try:
            root = m.body(name).id
        except KeyError:
            return set()
        out = set()
        for b in range(m.nbody):
            x = b
            while x > 0:
                if x == root:
                    out.add(b)
                    break
                x = m.body_parentid[x]
        return out

    def path_clear(self, q0, q1, exclude_bodies=(), n=30, margin_m=0.0, held=None):
        """True when along the joint-space interpolation q0 -> q1 (n postures, the private model's contact detection) no
        Panda geom touches any non-Panda geom except the `exclude_bodies` subtrees (the held part), and -- with
        `held=(part name, T_tcp_part)` -- the held part, carried rigidly by the tcp, touches nothing but the hand (the
        0.46 m compound swept through the trays on a long turn while the arm itself was clear). Returns (clear, worst
        penetration m, (body, other body)); `margin_m` applies to the arm, the held part is checked at zero margin."""
        m = self.m
        panda = set(b for b in range(m.nbody) if m.body(b).name.startswith("panda/"))
        skip = set()
        for name in exclude_bodies:
            skip |= self._subtree(name)
        held_bodies, held_adr = set(), None
        if held is not None:
            hname, T_tcp_part = held
            held_bodies = self._subtree(hname)
            try:
                held_adr = m.jnt_qposadr[m.joint("free_" + hname).id]
            except KeyError:
                held_adr = None
            T_tcp_part = np.asarray(T_tcp_part, float)
            quat = np.zeros(4)
        worst, pair = 0.0, None
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        for s in np.linspace(0.0, 1.0, int(n)):
            q = q0 + s * (q1 - q0)
            self.d.qpos[self.qadr] = q
            if held_adr is not None:
                p, R = self.fk(q)
                Rp = R @ T_tcp_part[:3, :3]
                self.d.qpos[held_adr:held_adr + 3] = p + R @ T_tcp_part[:3, 3]
                self.mujoco.mju_mat2Quat(quat, Rp.reshape(9))
                self.d.qpos[held_adr + 3:held_adr + 7] = quat
            self.mujoco.mj_forward(self.m, self.d)
            for i in range(self.d.ncon):
                c = self.d.contact[i]
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                in1, in2 = b1 in panda, b2 in panda
                if in1 != in2:                                  # arm vs something
                    other = b2 if in1 else b1
                    if other in skip:
                        continue
                    if m.body(other).name in ("world",) and m.geom(c.geom2 if in1 else c.geom1).name == "floor":
                        continue
                    pen = -float(c.dist) + margin_m
                    if pen > worst:
                        worst, pair = pen, (m.body(b1 if in1 else b2).name, m.body(other).name)
                elif held_adr is not None and not in1:          # neither is the arm: the held part vs the rest
                    h1, h2 = b1 in held_bodies, b2 in held_bodies
                    if h1 == h2:
                        continue
                    other = b2 if h1 else b1
                    pen = -float(c.dist)
                    if pen > worst:
                        worst, pair = pen, (m.body(b1 if h1 else b2).name + " (held)", m.body(other).name)
        return worst <= 0.0, worst, pair

    def path_min_z(self, q0, q1, n=20):
        """Lowest tcp height along the joint-space interpolation (a dip means the held part sweeps low)."""
        zs = []
        for s in np.linspace(0, 1, n):
            p, _ = self.fk(np.asarray(q0) + s * (np.asarray(q1) - np.asarray(q0)))
            zs.append(p[2])
        return float(min(zs))
