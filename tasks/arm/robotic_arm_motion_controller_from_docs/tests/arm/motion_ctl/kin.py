"""Kinematics, IK, clearance and planning on a MuJoCo model of a unit in a cell (grader side: the job generator, the
oracle, the metrics). Works on a clearance-probe model (`unit.unit_mjcf(..., probe=True)`): kinematics and collision
only, never stepped.

    K = Kin(probe_model, spec)                 # spec: the unit (its compliance sets the static deflection)
    K.fk(q)                                    # q = 6 design angles of the output hinges -> (tip position, R)
    K.ik(pos, approach, jaw_axis=None, ...)    # -> (q, errors) (multi-start Levenberg-Marquardt, joint limits)
    K.clearance(q)                             # -> (min robot-cell distance capped at the probe's margin, self contact)
    K.plan(q_from, q_to, clear_m)              # RRT-Connect + shortcut in the five arm joints -> waypoint list or None

The TOOL POINT is the TCP + R_tcp @ tool_off (tool_off = (0, 0, tool_len) by default; the needle tip: (seat x, 0, 45 mm)). The static
deflection of the series compliance (gravity load / stiffness, at the friction band's centre) is added to the slack
hinges when `deflect` is on, so a pose is the arm as it would rest there.
"""

import math

import numpy as np

from . import unit as U

ARM = 5
W_POS = 1.0
W_APP = 0.003          # 1 deg of approach weighs like 0.05 mm of position: position first, then the direction
W_JAW = 0.002
W_REG = 2e-5           # continuation IK (paths, straight lines): a weak pull towards the previous solution, only to
                       # keep the free wrist roll where it was (it must not trade the approach for continuity)


class Kin(object):
    def __init__(self, model, spec=None, tool_len=0.0, exempt=(), deflect=True, limit_margin_rad=math.radians(2.0), tool_off=None):
        import mujoco
        self.mj = mujoco
        self.m = model
        self.d = mujoco.MjData(model)
        m = model
        J = U.JOINTS
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in J]
        self.qadr = np.array([m.jnt_qposadr[i] for i in jid])
        self.dadr = np.array([m.jnt_dofadr[i] for i in jid])
        self.lo = np.array([m.jnt_range[i][0] for i in jid]) + math.radians(1.0) + limit_margin_rad   # unit.MECH_MARGIN back
        self.hi = np.array([m.jnt_range[i][1] for i in jid]) - math.radians(1.0) - limit_margin_rad
        sj = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j + "_slack") for j in U.ARM_JOINTS]
        self.slack_qadr = np.array([m.jnt_qposadr[i] for i in sj]) if min(sj) >= 0 else None
        self.slack_dadr = np.array([m.jnt_dofadr[i] for i in sj]) if min(sj) >= 0 else None
        self.k = (np.array([float(spec["D"]["compliance_nm_per_rad"][j]) for j in U.ARM_JOINTS]) if spec else None)
        self.deflect = bool(deflect and self.slack_qadr is not None and self.k is not None)
        self.tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.body = int(m.site_bodyid[self.tcp])
        self.tool_len = float(tool_len)
        self.tool_off = np.asarray(tool_off if tool_off is not None else (0.0, 0.0, self.tool_len), float)
        self.robot_geom = np.array([m.geom_bodyid[g] != 0 for g in range(m.ngeom)])
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self.geom_name = names
        self.exempt = set(exempt)            # (robot geom id, cell geom id): distances that do not count
        self.geom_object = {}
        for g, n in enumerate(names):
            if n.startswith("cell_"):
                self.geom_object[g] = n
        self.object_of = None                # set by set_cell_objects
        self.n_check = 0

    def set_cell_objects(self, geom_to_object):
        self.object_of = dict(geom_to_object)

    # ------------------------------------------------------------------------------------------------- kinematics
    def _set(self, q):
        q = np.asarray(q, float)
        self.d.qpos[:] = 0.0
        self.d.qpos[self.qadr[:len(q)]] = q
        if self.deflect:
            self.mj.mj_kinematics(self.m, self.d)
            self.mj.mj_comPos(self.m, self.d)
            self.mj.mj_rne(self.m, self.d, 0, self.d.qfrc_bias)
            g = -self.d.qfrc_bias[self.slack_dadr]
            self.d.qpos[self.slack_qadr] = np.clip(g / self.k, -U.SLACK_RANGE_RAD, U.SLACK_RANGE_RAD)
        self.mj.mj_kinematics(self.m, self.d)

    def fk(self, q):
        self._set(q)
        p = self.d.site_xpos[self.tcp].copy()
        R = self.d.site_xmat[self.tcp].reshape(3, 3).copy()
        return p + R @ self.tool_off, R

    def jac(self, q):
        """(Jp 3x6, Jr 3x6) at the tool point, columns = the six output hinges (rigid: the slack is held)."""
        self._set(q)
        self.mj.mj_comPos(self.m, self.d)
        p = self.d.site_xpos[self.tcp] + self.d.site_xmat[self.tcp].reshape(3, 3) @ self.tool_off
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        self.mj.mj_jac(self.m, self.d, jp, jr, p, self.body)
        return jp[:, self.dadr], jr[:, self.dadr]

    def conditioning(self, q, L=0.1):
        jp, jr = self.jac(q)
        pid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, "wrist_pitch")
        ax = self.d.xaxis[pid] / np.linalg.norm(self.d.xaxis[pid])
        Jt = np.vstack([jp[:, :4], L * (ax @ jr[:, :4])[None, :]])
        return float(np.linalg.svd(Jt, compute_uv=False)[-1])

    # ---------------------------------------------------------------------------------------------------------- IK
    def _plane_geometry(self):
        """Zero-pose anchors of the pitch chain in the arm's plane (y forward, z up; at yaw 0, rigid): S (shoulder
        pitch axis), E (elbow), W (wrist pitch), T (tool point); the tool direction at zero is +z."""
        if getattr(self, "_geom", None) is None:
            d, mj = self.d, self.mj
            with self.rigid():
                self._set(np.zeros(6))
            names = ["shoulder_pitch", "elbow_pitch", "wrist_pitch"]
            pts = []
            for n in names:
                j = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, n)
                pts.append(d.xanchor[j].copy())
            p = d.site_xpos[self.tcp] + d.site_xmat[self.tcp].reshape(3, 3) @ self.tool_off
            pts.append(p)
            P2 = [np.array([x[1], x[2]]) for x in pts]
            self._geom = {"S": P2[0], "SE": P2[1] - P2[0], "EW": P2[2] - P2[1], "WT": P2[3] - P2[2],
                          "lat": float(pts[3][0])}                # the tool point's lateral offset (exact at roll 0)
        return self._geom

    @staticmethod
    def _R2(phi):
        c, s_ = math.cos(phi), math.sin(phi)
        return np.array([[c, -s_], [s_, c]])

    def seeds(self, pos, app):
        """Up to two geometric seeds (the elbow branches) for pos / app, from the zero-pose geometry of the pitch chain;
        LM refines them (offsets, tilts and the deflection are left to it)."""
        g = self._plane_geometry()
        pos = np.asarray(pos, float)
        app = np.asarray(app, float)
        rho = math.hypot(pos[0], pos[1])
        lat = g["lat"]
        Y = math.sqrt(max(rho ** 2 - lat ** 2, 1e-12))
        yaw = math.atan2(-pos[0], pos[1]) + math.asin(max(-1.0, min(1.0, lat / max(rho, 1e-9))))
        h = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        a2 = np.array([float(np.dot(app, h)), float(app[2])])
        a2 /= max(1e-12, np.linalg.norm(a2))
        phi3 = math.atan2(-a2[0], a2[1])
        T = np.array([Y, pos[2]])
        W = T - self._R2(phi3) @ g["WT"]
        v = W - g["S"]
        l1, l2 = np.linalg.norm(g["SE"]), np.linalg.norm(g["EW"])
        d1 = math.atan2(-g["SE"][0], g["SE"][1])            # zero-pose link directions from vertical (rotation about +x)
        d2 = math.atan2(-g["EW"][0], g["EW"][1])
        r = np.linalg.norm(v)
        out = []
        c2 = (r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
        if abs(c2) > 1.0:
            c2 = max(-1.0, min(1.0, c2))
        base = math.atan2(-v[0], v[1])
        for sgn in (1.0, -1.0):
            k = sgn * math.acos(c2)                          # relative bend
            psi1 = base - math.atan2(l2 * math.sin(k), l1 + l2 * math.cos(k))
            psi2 = psi1 + k
            s_ = psi1 - d1
            e = (psi2 - d2) - s_
            w = phi3 - (psi2 - d2)
            q = np.array([yaw, s_, e, w, 0.0, 0.15])
            q[:4] = (q[:4] + math.pi) % (2 * math.pi) - math.pi
            out.append(q)
        return out

    def _resid(self, q, pos, app, jaw):
        p, R = self.fk(q)
        r = [W_POS * (p - pos)]
        r.append(W_APP * np.cross(R[:, 2], app))
        if jaw is not None:
            r.append(W_JAW * np.array([np.dot(R[:, 0], np.cross(app, jaw)), 0.0, 0.0]))
        return np.concatenate(r)

    def errors(self, q, pos, app, jaw=None):
        p, R = self.fk(q)
        e = {"pos_m": float(np.linalg.norm(p - pos)),
             "app_deg": math.degrees(math.acos(float(np.clip(np.dot(R[:, 2], app), -1, 1))))}
        if jaw is not None:
            jw = jaw - np.dot(jaw, R[:, 2]) * R[:, 2]
            jw /= max(1e-9, np.linalg.norm(jw))
            e["jaw_deg"] = math.degrees(math.acos(float(np.clip(abs(np.dot(R[:, 0], jw)), -1, 1))))
        return e

    def _resid_jac(self, q, pos, app, jaw):
        """Residual and its Jacobian over the five arm joints (analytic: mj_jac at the tool point)."""
        jp, jr = self.jac(q)
        p = self.d.site_xpos[self.tcp] + self.d.site_xmat[self.tcp].reshape(3, 3) @ self.tool_off
        R = self.d.site_xmat[self.tcp].reshape(3, 3)
        a = R[:, 2]
        r = [W_POS * (p - pos), W_APP * np.cross(a, app)]
        # d(a x app)/dq = (w x a) x app with w = jr[:, i]
        da = np.cross(jr[:, :ARM].T, a).T                      # 3 x 5: d a / d q_i
        J = [W_POS * jp[:, :ARM], W_APP * np.cross(da.T, app).T]
        if jaw is not None:
            x = R[:, 0]
            n = np.cross(app, jaw)
            r.append(W_JAW * np.array([np.dot(x, n)]))
            dx = np.cross(jr[:, :ARM].T, x).T
            J.append(W_JAW * (n @ dx)[None, :])
        return np.concatenate(r), np.vstack(J)

    def ik(self, pos, app, jaw=None, q0=None, starts=8, iters=80, rng=None, jaw_q=0.15, tol_m=2e-6, q_ref=None, w_reg=None):
        """Levenberg-Marquardt over the five arm joints from q0 and random starts (box limits); returns (q6, errors) of
        the best start. The approach error is minimised after the position (W_POS >> W_APP), so an approach the arm
        cannot take ends at the reachable direction closest to it."""
        pos = np.asarray(pos, float)
        app = np.asarray(app, float) / np.linalg.norm(app)
        jaw = None if jaw is None else np.asarray(jaw, float) / np.linalg.norm(jaw)
        rng = rng or np.random.default_rng(0)
        cands = []
        if q0 is not None:
            cands.append(np.asarray(q0, float)[:ARM].copy())
        for sq in self.seeds(pos, app):
            cands.append(sq[:ARM].copy())
        az = math.atan2(-pos[0], pos[1])
        for k in range(starts):
            q = rng.uniform(self.lo[:ARM], self.hi[:ARM])
            if k % 2 == 0:
                q[0] = np.clip(az + rng.normal(0, 0.1), self.lo[0], self.hi[0])
            cands.append(q)
        best = None
        for q in cands:
            qq = np.concatenate([np.clip(q, self.lo[:ARM], self.hi[:ARM]), [jaw_q]])
            lam = 1e-2

            def rj(qx):
                r_, J_ = self._resid_jac(qx, pos, app, jaw)
                if q_ref is not None:
                    wr = W_REG if w_reg is None else w_reg
                    r_ = np.concatenate([r_, wr * (qx[:ARM] - np.asarray(q_ref, float)[:ARM])])
                    J_ = np.vstack([J_, wr * np.eye(ARM)])
                return r_, J_
            r, Jm = rj(qq)
            f = float(r @ r)
            for _ in range(iters):
                H = Jm.T @ Jm
                dq = -np.linalg.solve(H + lam * (np.diag(np.diag(H)) + 1e-9 * np.eye(ARM)), Jm.T @ r)
                qn = qq.copy()
                qn[:ARM] = np.clip(qq[:ARM] + dq, self.lo[:ARM], self.hi[:ARM])
                rn, Jn = rj(qn)
                fn = float(rn @ rn)
                if fn < f:
                    qq, r, Jm, f = qn, rn, Jn, fn
                    lam = max(lam / 4, 1e-9)
                    if np.linalg.norm(dq) < 1e-9:
                        break
                else:
                    lam *= 5
                    if lam > 1e8:
                        break
            if best is None or f < best[1]:
                best = (qq, f)
            if best[1] < 1e-14:
                break
        q = best[0]
        return q, self.errors(q, pos, app, jaw)

    # -------------------------------------------------------------------------------------------------- clearance
    EXEMPT_MIN_M = 0.003                 # an exempt pair (the job's support) must still keep this distance in plans

    def clearance(self, q, cap=None):
        """(min robot-to-cell distance over the non-exempt pairs (capped at the probe margin), self_contact bool, pair);
        the smallest distance over the exempt pairs is left in self.last_exempt."""
        self._set(q)
        self.mj.mj_collision(self.m, self.d)
        self.n_check += 1
        best, pair, selfc = (cap if cap is not None else U.CLEAR_DETECT_M), None, False
        self.last_exempt = U.CLEAR_DETECT_M
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = self.robot_geom[g1], self.robot_geom[g2]
            if r1 and r2:
                if float(c.dist) < -1e-5:
                    selfc = True
                continue
            if not (r1 or r2):
                continue
            robg, cellg = (g1, g2) if r1 else (g2, g1)
            if (robg, cellg) in self.exempt:
                self.last_exempt = min(self.last_exempt, float(c.dist))
                continue
            obj = (self.object_of or {}).get(cellg, self.geom_name[cellg])
            if float(c.dist) < best:
                best, pair = float(c.dist), (self.geom_name[robg], obj)
        return best, selfc, pair

    def rigid(self):
        """A context in which the static deflection is off (planning checks)."""
        k = self

        class _C(object):
            def __enter__(self):
                self.old = k.deflect
                k.deflect = False

            def __exit__(self, *a):
                k.deflect = self.old
        return _C()

    cond_min = 0.0                       # the planner's conditioning floor (set by the job generator)

    def valid(self, q, clear_m):
        if np.any(q[:ARM] < self.lo[:ARM]) or np.any(q[:ARM] > self.hi[:ARM]):
            return False
        c, selfc, _ = self.clearance(q)
        if selfc or c < clear_m or self.last_exempt < self.EXEMPT_MIN_M:
            return False
        return self.cond_min <= 0.0 or self.conditioning(q) >= self.cond_min

    # ---------------------------------------------------------------------------------------------------- planning
    def edge_ok(self, qa, qb, clear_m, step=math.radians(1.5)):
        n = max(1, int(math.ceil(np.max(np.abs(qb[:ARM] - qa[:ARM])) / step)))
        for k in range(1, n + 1):
            if not self.valid(qa + (qb - qa) * k / n, clear_m):
                return False
        return True

    def plan(self, qa, qb, clear_m, rng=None, max_nodes=3000, step=math.radians(8.0), check_step=math.radians(1.5)):
        """RRT-Connect in the five arm joints (the jaw keeps qa's angle); clearance >= clear_m along every edge at
        `check_step`. Returns [qa, ..., qb] (shortcut) or None."""
        rng = rng or np.random.default_rng(1)
        qa = np.asarray(qa, float).copy()
        qb = np.asarray(qb, float).copy()
        qb[5] = qa[5]
        if not (self.valid(qa, clear_m * 0.0) and self.valid(qb, clear_m)):
            pass
        if self.edge_ok(qa, qb, clear_m, check_step):
            return [qa, qb]
        Ta, Tb = [(qa, -1)], [(qb, -1)]

        def nearest(T, q):
            return int(np.argmin([np.sum((t[0][:ARM] - q[:ARM]) ** 2) for t in T]))

        def extend(T, q):
            i = nearest(T, q)
            qn = T[i][0]
            d = q[:ARM] - qn[:ARM]
            dist = np.linalg.norm(d)
            if dist < 1e-9:
                return None
            qs = qn.copy()
            qs[:ARM] = qn[:ARM] + d * min(1.0, step / dist)
            if self.edge_ok(qn, qs, clear_m, check_step):
                T.append((qs, i))
                return len(T) - 1
            return None

        def path(T, i):
            out = []
            while i >= 0:
                out.append(T[i][0])
                i = T[i][1]
            return out

        for it in range(max_nodes):
            qr = qa.copy()
            qr[:ARM] = rng.uniform(self.lo[:ARM], self.hi[:ARM])
            ia = extend(Ta, qr)
            if ia is not None:
                qnew = Ta[ia][0]
                # connect Tb towards qnew
                while True:
                    ib = extend(Tb, qnew)
                    if ib is None:
                        break
                    if np.linalg.norm(Tb[ib][0][:ARM] - qnew[:ARM]) < 1e-6:
                        pa = path(Ta, ia)[::-1]
                        pb = path(Tb, ib)
                        full = pa + pb[1:]
                        if Ta[0][0] is not qa:
                            full = full[::-1]
                        return self.shortcut(full, clear_m, rng, check_step)
            Ta, Tb = Tb, Ta
        return None

    def shortcut(self, P, clear_m, rng, check_step, rounds=60):
        P = [np.asarray(p, float) for p in P]
        if np.linalg.norm(P[0][:ARM] - P[-1][:ARM]) > 0 and len(P) > 2:
            for _ in range(rounds):
                if len(P) <= 2:
                    break
                i, j = sorted(rng.choice(len(P), 2, replace=False))
                if j - i < 2:
                    continue
                if self.edge_ok(P[i], P[j], clear_m, check_step):
                    P = P[:i + 1] + P[j:]
        return P
