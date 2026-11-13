"""Appliance tending, P2: the privileged reference's primitives in the P1 environment.

A scripted controller that reads the simulator's truth (control geometry, joint axes, joint values)
and acts only through `ApplianceEnv.step` actions, the same 8 numbers a submission sends:
`[dx, dy, dz, drx, dry, drz, grip, cams]`. Nothing is set directly after reset. It ports the
RoboCasa pilot's primitives (`../appliance_primitives_pilot.py`) to this environment and adds the
reverse directions.

    drv = Driver(env)
    drv.knob("front_left", "on")       # or "off"; "temp" for the temperature knob (preheat)
    drv.button("start_button")         # or "stop_button"
    drv.door("open")                   # or "close"

Every primitive ends with the hand withdrawn and returns a dict of measurements; failures raise
PrimitiveError with the stage that failed. Camera pose estimation (the reference's only perception)
is a later P2 step; here the unit's pose is read from the simulator.
"""

import math

import mujoco
import numpy as np

import appliance_env as E

IK_STARTS = np.random.default_rng(20261113).uniform(0.15, 0.85, (6, 7))   # fixed restarts for the IK check
PRE = 0.12                     # m: pre-grasp stand-off along the approach
KNOB_ON_TARGET = 0.60          # rad: commanded past RoboCasa's 0.35 on-threshold, for margin (pilot)
KNOB_OFF_TARGET = 0.05         # rad: back towards the joint's lower limit
TEMP_TARGET = 0.32             # normalized: past PreheatOven's 0.25, with margin (P2: 0.40 left too large a turn)
HOLD_OK = 20                   # control steps the outcome must hold with the hand clear (design §4)
CLEAR = 0.10                   # m: hand clearance from the control while it holds


class PrimitiveError(Exception):
    def __init__(self, stage, msg):
        super().__init__("%s: %s" % (stage, msg))
        self.stage = stage


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rotvec(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-12:
        return np.zeros(3)
    return 2.0 * math.atan2(s, q[0]) * q[1:] / s


class Driver(object):
    MAX_STEPS = 3000

    def __init__(self, env, trace=None):
        self.env = env
        self.m, self.d = env.model, env.data
        self.steps = 0
        self.stage = "init"
        self.trace = trace
        self.pad_z, self.finger_axis, self.tip_reach = self._gripper_geometry()
        self._own = set(g for g in range(self.m.ngeom) if self.m.body(self.m.geom_bodyid[g]).name.startswith("r_"))
        self._allow, self._check = set(), set()
        self.why = []                   # why planned poses were rejected (diagnostics)
        self.via, self._via = [], []

    # ---- truth ----------------------------------------------------------------------
    def site_pos(self):
        return self.d.site_xpos[self.env.site].copy()

    def site_R(self):
        return self.d.site_xmat[self.env.site].reshape(3, 3).copy()

    def geom(self, name):
        g = self.m.geom(name).id
        return self.d.geom_xpos[g].copy(), self.d.geom_xmat[g].reshape(3, 3).copy(), self.m.geom_size[g].copy(), g

    def joint(self, name):
        j = self.m.joint(name)
        return (unit(self.d.xaxis[j.id]), self.d.xanchor[j.id].copy(), int(j.qposadr[0]),
                float(j.range[0]), float(j.range[1]))

    def qpos(self, adr):
        return float(self.d.qpos[adr])

    def touching(self, gid):
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (c.geom1 == gid and c.geom2 in self.env.gripper_geoms) or \
               (c.geom2 == gid and c.geom1 in self.env.gripper_geoms):
                return True
        return False

    def robot_contacts(self):
        own = set(g for g in range(self.m.ngeom) if self.m.body(self.m.geom_bodyid[g]).name.startswith("r_"))
        out = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            a, b = c.geom1 in own, c.geom2 in own
            if a != b:
                r, o = (c.geom1, c.geom2) if a else (c.geom2, c.geom1)
                out.add("%s|%s" % (self.m.body(self.m.geom_bodyid[r]).name, self.m.geom(o).name or
                                   self.m.body(self.m.geom_bodyid[o]).name))
        return sorted(out)

    def _gripper_geometry(self):
        """Where the finger pads sit along the grip site's z (m) and which site axis the fingers close along."""
        R, p = self.site_R(), self.site_pos()
        pads = [g for g in range(self.m.ngeom) if (self.m.geom(g).name or "").endswith("pad_collision")]
        local = [R.T @ (self.d.geom_xpos[g] - p) for g in pads]
        spread = local[0] - local[1]
        spread[2] = 0.0
        axis = np.zeros(3)
        axis[int(np.argmax(np.abs(spread)))] = 1.0
        tip = 0.0                       # how far the finger collision geometry reaches past the site along z
        for g in range(self.m.ngeom):
            if "finger" not in self.m.body(self.m.geom_bodyid[g]).name:
                continue
            if not (self.m.geom_contype[g] or self.m.geom_conaffinity[g]):
                continue
            Rg = self.d.geom_xmat[g].reshape(3, 3)
            c, h = self.m.geom_aabb[g, :3], self.m.geom_aabb[g, 3:]
            for sg in np.ndindex(2, 2, 2):
                w = self.d.geom_xpos[g] + Rg @ (c + h * (2 * np.array(sg) - 1))
                tip = max(tip, float((R.T @ (w - p))[2]))
        return float(np.mean([v[2] for v in local])), axis, tip

    def facing(self, approach, finger_dir, sign=None):
        """Rotation with the site's z along `approach` and the finger closing axis along `finger_dir`;
        of the two signs of finger_dir, the one nearer the current orientation."""
        z = unit(approach)
        best = None
        for sgn in ((1.0, -1.0) if sign is None else (sign,)):
            f = unit(sgn * (np.asarray(finger_dir, float) - np.dot(finger_dir, z) * z))
            if self.finger_axis[1] > 0.5:
                y = f
                x = np.cross(y, z)
            else:
                x = f
                y = np.cross(z, x)
            R = np.column_stack([x, y, z])
            ang = np.linalg.norm(rotvec(R @ self.site_R().T))
            if best is None or ang < best[0]:
                best = (ang, R)
        return best[1]

    def feasible(self, poses, allow=(), check=None, width=0.08):
        """Full-pose IK (from the current joints, null space towards the home pose) through `poses`
        [(position, R), ...] in order. Returns the smallest joint-limit margin (rad) over the poses, or
        -1 if one is not reached within 5 mm and 3 degrees."""
        m, env = self.m, self.env
        jid = [m.joint("r_joint%d" % i).id for i in range(1, 8)]
        lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        # Only from the current joints: the operational-space controller follows the same local path, so
        # a solution found from another start (another IK branch) is one the arm may not be able to reach.
        self._allow = set(allow)
        self._check = set(range(len(poses))) if check is None else set(check)
        self.via = []
        dd = mujoco.MjData(m)
        dd.qpos[:] = self.d.qpos
        dd.qpos[env.gq[0]], dd.qpos[env.gq[1]] = width / 2, -width / 2
        mg = self._chain(dd, poses, lo, hi)
        if mg > 0.1:
            return mg
        # Another arm configuration: solve the first pose from fixed starts, then check that the arm can get
        # there by following the joint-space interpolation's Cartesian poses (what `via` hands the executor).
        best = (mg, [])
        q_now = self.d.qpos[env.jq].copy()
        for r in IK_STARTS:
            dq_ = mujoco.MjData(m)
            dq_.qpos[:] = self.d.qpos
            dq_.qpos[env.gq[0]], dq_.qpos[env.gq[1]] = width / 2, -width / 2
            dq_.qpos[env.jq] = lo + (hi - lo) * r
            self._solve(dq_, poses[0][0], poses[0][1], lo, hi, iters=600)
            q_goal = dq_.qpos[env.jq].copy()
            via, ok = [], True
            dv = mujoco.MjData(m)
            dv.qpos[:] = dq_.qpos
            for f in np.linspace(0.1, 1.0, 10):
                dv.qpos[env.jq] = q_now + (q_goal - q_now) * f
                if self.collides(dv, set()):
                    ok = False
                    break
                via.append((dv.site_xpos[env.site].copy(), dv.site_xmat[env.site].reshape(3, 3).copy()))
            if not ok:
                continue
            dd = mujoco.MjData(m)
            dd.qpos[:] = self.d.qpos
            dd.qpos[env.gq[0]], dd.qpos[env.gq[1]] = width / 2, -width / 2
            for pos, Rt in via[:-1]:                    # follow the path locally, as the controller would
                self._solve(dd, pos, Rt, lo, hi, iters=150)
            mg = self._chain(dd, poses, lo, hi, lead=False)
            if mg > best[0]:
                best = (mg, via[:-1])
            if mg > 0.15:
                break
        self.via = best[1]
        return best[0]

    def body_geoms(self, gid):
        b = self.m.geom_bodyid[gid]
        return set(g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b)

    def _solve(self, dd, pos, Rt, lo, hi, iters=400):
        """Damped least-squares full-pose IK in dd, null space towards the home pose."""
        m, env = self.m, self.env
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, dd)
            mujoco.mj_comPos(m, dd)
            ep = pos - dd.site_xpos[env.site]
            er = rotvec(Rt @ dd.site_xmat[env.site].reshape(3, 3).T)
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3:
                break
            mujoco.mj_jacSite(m, dd, jp, jr, env.site)
            J = np.vstack([jp[:, env.jv], 0.3 * jr[:, env.jv]])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
            dq = Jp @ np.concatenate([ep, 0.3 * er])
            dq += (np.eye(7) - Jp @ J) @ (0.05 * (env.q_null - dd.qpos[env.jq]))
            dd.qpos[env.jq] = np.clip(dd.qpos[env.jq] + np.clip(dq, -0.1, 0.1), lo, hi)

    def collides(self, dd, allow):
        """Robot geoms in contact (penetration over 1 mm) with anything but `allow` geoms, at dd's pose."""
        mujoco.mj_fwdPosition(self.m, dd)
        own = self._own
        for i in range(dd.ncon):
            c = dd.contact[i]
            a, b = c.geom1 in own, c.geom2 in own
            if a == b or c.dist > -0.001:
                continue
            other = c.geom2 if a else c.geom1
            if other in allow:
                continue
            mine = c.geom1 if a else c.geom2
            return "%s/%s" % (self.m.body(self.m.geom_bodyid[mine]).name,
                              self.m.geom(other).name or self.m.body(self.m.geom_bodyid[other]).name)
        return None

    def _chain(self, dd, poses, lo, hi, lead=True):
        m, env = self.m, self.env
        margin = 10.0
        # reach the first pose through intermediate poses, as the controller's small steps do
        mujoco.mj_kinematics(m, dd)
        p0, R0 = dd.site_xpos[env.site].copy(), dd.site_xmat[env.site].reshape(3, 3).copy()
        pos1, R1 = poses[0]
        rv = rotvec(R1 @ R0.T)
        for f in ((0.25, 0.5, 0.75) if lead else ()):
            self._solve(dd, p0 + (pos1 - p0) * f, E.axis_angle_mat(rv * f) @ R0, lo, hi)
        for k, (pos, Rt) in enumerate(poses):
            self._solve(dd, pos, Rt, lo, hi)
            mujoco.mj_kinematics(m, dd)
            pe = np.linalg.norm(pos - dd.site_xpos[env.site])
            ae = np.linalg.norm(rotvec(Rt @ dd.site_xmat[env.site].reshape(3, 3).T))
            if pe > 0.005 or ae > math.radians(3):
                self.why.append("pose%d ik %.3fm %.0fdeg" % (k, pe, math.degrees(ae)))
                return -1.0
            hit = self.collides(dd, self._allow) if k in self._check else None
            if hit:
                self.why.append("pose%d hits %s" % (k, hit))
                return -1.0
            q = dd.qpos[env.jq]
            margin = min(margin, float(np.min(np.minimum(q - lo, hi - q))))
        return margin

    # ---- motion ---------------------------------------------------------------------
    def tick(self, target, R, grip, speed=0.03, turn=0.2, cams=0):
        if self.steps >= self.MAX_STEPS:
            raise PrimitiveError(self.stage, "step budget exhausted")
        if hasattr(self.env, "command"):          # over the wire: the action is computed per query (Shadow)
            obs = self.env.command(np.asarray(target, float), np.asarray(R, float), grip, speed, turn, cams)
            self.steps += 1
            return obs
        dp = np.asarray(target, float) - self.site_pos()
        n = np.linalg.norm(dp)
        if n > speed:
            dp *= speed / n
        dr = rotvec(np.asarray(R) @ self.site_R().T)
        n = np.linalg.norm(dr)
        if n > turn:
            dr *= turn / n
        obs, _done, _info = self.env.step(list(dp) + list(dr) + [grip, cams])
        self.steps += 1
        if self.trace is not None:
            self.trace.append({"t": self.env.t, "stage": self.stage, "site": np.round(self.site_pos(), 4).tolist()})
        return obs

    def follow_via(self, via, grip):
        for k, (pos, Rt) in enumerate(via):
            self.move("via%d" % k, pos, Rt, grip, max_steps=120, tol=0.03, ang_tol=0.15)

    def move(self, stage, target, R, grip, max_steps=300, tol=0.006, ang_tol=0.08, stop=None, **kw):
        """Servo the grip site to `target`, orientation `R`. Near the target an integral term removes
        the operational-space controller's steady-state offset (gravity and null-space pull)."""
        self.stage = stage
        target = np.asarray(target, float)
        bias = np.zeros(3)
        ok = 0
        for _ in range(max_steps):
            e = target - self.site_pos()
            if np.linalg.norm(e) < 0.03:
                bias = bias + 0.2 * e
                n = np.linalg.norm(bias)
                if n > 0.03:
                    bias *= 0.03 / n
            self.tick(target + bias, R, grip, **kw)
            if stop is not None and stop():
                return True
            err = np.linalg.norm(np.asarray(target) - self.site_pos())
            ang = np.linalg.norm(rotvec(np.asarray(R) @ self.site_R().T))
            ok = ok + 1 if (err < tol and ang < ang_tol) else 0
            if ok >= 3:
                return stop is None
        if stop is not None:
            return False
        raise PrimitiveError(stage, "did not reach target: %.3f m off; contacts %s" % (
            np.linalg.norm(np.asarray(target) - self.site_pos()), self.robot_contacts()))

    def hold(self, stage, steps, grip, R=None, target=None):
        self.stage = stage
        target = self.site_pos() if target is None else target
        R = self.site_R() if R is None else R
        for _ in range(steps):
            self.tick(target, R, grip)

    def confirm(self, stage, predicate, control_pos):
        """The outcome must hold HOLD_OK steps with the hand at least CLEAR from the control."""
        self.stage = stage
        held = 0
        for _ in range(HOLD_OK + 10):
            self.tick(self.site_pos(), self.site_R(), -1.0)
            far = np.linalg.norm(self.site_pos() - control_pos) >= CLEAR
            held = held + 1 if (predicate() and far) else 0
            if held >= HOLD_OK:
                return
        raise PrimitiveError(stage, "outcome did not hold %d steps with the hand clear (state %s)" % (
            HOLD_OK, self.env.unit_state()))

    def retreat(self, stage, direction, dist, R, grip=-1.0):
        self.move(stage, self.site_pos() + unit(direction) * dist, R, grip, tol=0.015, ang_tol=0.3)

    # ---- primitives -----------------------------------------------------------------
    def _knob_grasp(self, axis, anchor, outward, finger0, is_bar, tip_point, twist, open_width):
        """Choose an approach (along the knob axis, or tilted) and a finger roll whose pre-grasp, grasp and
        twisted poses are all reachable, with the largest joint-limit margin."""
        down = np.array([0.0, 0.0, -1.0])
        perp = down - np.dot(down, outward) * outward
        if np.linalg.norm(perp) < 1e-6:           # a vertical knob: tilt towards the robot instead
            perp = np.array([-1.0, 0.0, 0.0])
        perp = unit(perp)
        side = unit(np.cross(outward, perp))
        approaches = [("axis", -outward)]
        for deg in (20, 35):
            approaches.append(("tilt%d" % deg, unit(-outward * math.cos(math.radians(deg)) + perp * math.sin(math.radians(deg)))))
        for sg in (1, -1):
            approaches.append(("side%+d" % (20 * sg), unit(-outward * math.cos(math.radians(20)) + sg * side * math.sin(math.radians(20)))))
        rolls = (0, 90) if is_bar else (0, 45, 90, 135)
        best = (None, None, -1.0, None)
        for name, app in approaches:
            for roll, sgn in [(r, sg) for r in rolls for sg in (1.0, -1.0)]:
                f = E.axis_angle_mat(axis * math.radians(roll)) @ finger0
                R = self.facing(app, f, sign=sgn)
                if is_bar and roll == 90:
                    continue
                grasp = tip_point - R[:, 2] * self.tip_reach
                # a tilted hand puts one fingertip deeper than the other: keep the deepest one off the plane
                fa = R @ self.finger_axis
                tips = [grasp + R[:, 2] * self.tip_reach + sg * fa * (open_width / 2 + 0.01) for sg in (1, -1)]
                shift = float(np.dot(tip_point, outward)) - min(float(np.dot(t, outward)) for t in tips)
                if shift > 0.006:
                    continue
                grasp = grasp + outward * max(shift, 0.0)
                poses = [(grasp - R[:, 2] * PRE, R), (grasp, R)]
                Rot = E.axis_angle_mat(axis * twist)
                poses.append((anchor + Rot @ (grasp - anchor), Rot @ R))
                mg = self.feasible(poses, allow=self._knob_allow, width=open_width)
                if mg > best[2]:
                    best = (R, grasp, mg, "%s roll%d%s" % (name, roll, "" if sgn > 0 else " flip"))
                    self._via = list(self.via)
            if best[2] > 0.15:
                break
        return best

    def _knob_geometry(self, loc, axis, anchor):
        """Where and how to pinch the knob as it stands now: outward direction, finger direction, whether it
        is a bar, the fingertip point and the part's width across the fingers."""
        centre, gR, gsize, gid = self.geom("u_knob_%s_main" % loc)
        s_ = float(np.dot(centre - anchor, axis))
        if abs(s_) > 0.002:
            outward = axis * math.copysign(1.0, s_)
        else:
            home = self.env.layout["robot_base"] + E.HOME_SITE
            outward = axis if np.dot(home - anchor, axis) > 0 else -axis
        # the fingers close across a bar knob's thinnest in-plane dimension; a round knob any way
        is_bar = self.m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
        if is_bar:
            inplane = [i for i in range(3) if abs(np.dot(gR[:, i], axis)) < 0.5]
            finger0 = gR[:, min(inplane, key=lambda i: gsize[i])]
            half = sum(abs(np.dot(gR[:, i], outward)) * gsize[i] for i in range(3))
            width = 2 * min(gsize[i] for i in inplane)
        else:
            finger0 = unit(np.cross(axis, [0.0, 0.0, 1.0])) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            c = abs(np.dot(gR[:, 2], outward))
            half = c * gsize[1] + math.sqrt(max(0.0, 1 - c * c)) * gsize[0]
            width = 2 * gsize[0]
        # fingertips stop 3 mm out from the knob's inner end or the outer face of its base, whichever is
        # farther out (the base disc may stand proud of the bar's inner end)
        inner = float(np.dot(centre, outward)) - half
        body = self.m.geom_bodyid[gid]
        for g in range(self.m.ngeom):
            if g == gid or self.m.geom_bodyid[g] != body or not (self.m.geom_contype[g] or self.m.geom_conaffinity[g]):
                continue
            Rg, sz = self.d.geom_xmat[g].reshape(3, 3), self.m.geom_size[g]
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                c_ = abs(np.dot(Rg[:, 2], outward))
                h_ = c_ * sz[1] + math.sqrt(max(0.0, 1 - c_ * c_)) * sz[0]
                r_ = sz[0]
            else:
                h_ = sum(abs(np.dot(Rg[:, i], outward)) * sz[i] for i in range(3))
                r_ = max(sz)
            v = self.d.geom_xpos[g] - anchor
            radial = np.linalg.norm(v - np.dot(v, axis) * axis)
            if r_ > 0.6 * max(gsize[:2]) and radial < 0.01:          # a base disc under the knob
                inner = max(inner, float(np.dot(self.d.geom_xpos[g], outward)) + h_)
        tip_point = centre + outward * (inner + 0.003 - float(np.dot(centre, outward)))
        return centre, gid, outward, finger0, is_bar, tip_point, width

    def knob(self, loc, want, max_grasps=3):
        """Turn a burner knob on/off, or the temperature knob ('temp') to preheat ('on') or zero ('off').
        A turn the wrist cannot make in one grasp is made in up to `max_grasps` grasps."""
        jname = "u_knob_%s_joint" % loc
        axis, anchor, adr, lo, hi = self.joint(jname)
        out = {"control": jname, "start_angle": round(self.qpos(adr), 4), "grasps": []}
        start = self.qpos(adr)
        positive = hi > abs(lo)
        sign = 1.0 if positive else -1.0
        direction = sign if want == "on" else -np.sign(start or sign)
        if loc == "temp":
            is_done = (lambda: E.normalized_joint(self.qpos(adr), lo, hi) >= TEMP_TARGET - 0.02) if want == "on" \
                else (lambda: abs(self.qpos(adr)) <= KNOB_OFF_TARGET + 0.02)
            final = (lambda: self.env.unit_state()["preheated"]) if want == "on" \
                else (lambda: not self.env.unit_state()["preheated"])
            goal = TEMP_TARGET * (hi - lo) * sign if want == "on" else 0.0
        else:
            is_done = (lambda: abs(self.qpos(adr)) >= KNOB_ON_TARGET) if want == "on" \
                else (lambda: abs(self.qpos(adr)) <= KNOB_OFF_TARGET + 0.02)
            final = (lambda: self.env.unit.burner_on(self.d, loc)) if want == "on" \
                else (lambda: not self.env.unit.burner_on(self.d, loc))
            goal = KNOB_ON_TARGET * sign if want == "on" else 0.0
        for k in range(max_grasps):
            if is_done():
                break
            centre, gid, outward, finger0, is_bar, tip_point, width = self._knob_geometry(loc, axis, anchor)
            self._knob_allow = self.body_geoms(gid)
            remaining = abs(goal - self.qpos(adr)) + 0.1
            plan = None
            for chunk in (remaining, 1.0, 0.6):
                if chunk > remaining:
                    continue
                R, grasp, margin, how = self._knob_grasp(axis, anchor, outward, finger0, is_bar, tip_point,
                                                         direction * chunk, width + 0.025)
                if R is not None and margin > 0:
                    plan = (R, grasp, margin, how, chunk)
                    break
            if plan is None:
                raise PrimitiveError("plan", "no feasible grasp pose (grasp %d)" % (k + 1))
            R, grasp, margin, how, chunk = plan
            g_open = float(np.clip(1.0 - (width + 0.025) / 0.04, -1.0, 0.6))   # open a little wider than the part
            out["grasps"].append({"grasp": how, "ik_margin": round(margin, 3), "chunk": round(chunk, 2)})
            self.follow_via(self._via, g_open)
            self.move("approach_knob", grasp - R[:, 2] * PRE, R, g_open, max_steps=400, tol=0.008, ang_tol=0.08)
            self.move("to_knob", grasp, R, g_open, speed=0.01, tol=0.004)
            self.hold("pinch", 15, 1.0, R, grasp)
            if not self.touching(gid):
                raise PrimitiveError("pinch", "fingers do not touch the knob")
            # turn: rotate the hand about the knob axis towards the goal, at most `chunk`
            self.stage = "twist"
            q0 = self.qpos(adr)
            partial = chunk < remaining - 1e-9            # the last grasp turns until the goal is reached
            cmd = 0.0
            for _ in range(500):
                if is_done() or (partial and abs(self.qpos(adr) - q0) >= chunk - 0.1):
                    break
                cmd = min(cmd + 0.03, abs(self.qpos(adr) - q0) + 0.35, chunk + 0.2)
                Rot = E.axis_angle_mat(direction * cmd * axis)
                self.tick(anchor + Rot @ (grasp - anchor), Rot @ R, 1.0, turn=0.2)
            else:
                raise PrimitiveError("twist", "knob at %.3f rad" % self.qpos(adr))
            Rnow = self.site_R()
            self.hold("release", 12, -1.0, Rnow, self.site_pos())
            self.retreat("withdraw", -Rnow[:, 2], 0.14 if is_done() else 0.08, Rnow)
        if not is_done():
            raise PrimitiveError("twist", "knob at %.3f rad after %d grasps" % (self.qpos(adr), max_grasps))
        self.confirm("confirm", final, self.geom("u_knob_%s_main" % loc)[0])
        out["final_angle"] = round(self.qpos(adr), 4)
        return out

    # ---- buttons ----------------------------------------------------------------------
    def button(self, name):
        """Press the microwave's start or stop button with closed fingertips, then withdraw."""
        centre, gR, gsize, gid = self.geom("u_" + name)
        normal = gR[:, int(np.argmin(gsize))]
        home = self.env.layout["robot_base"] + E.HOME_SITE
        if np.dot(home - centre, normal) < 0:
            normal = -normal                              # out of the panel, towards the robot
        want_on = name == "start_button"
        flipped = lambda: self.env.unit.turned_on == want_on  # noqa: E731
        out = {"control": name}
        down = np.array([0.0, 0.0, -1.0])
        perp = unit(down - np.dot(down, normal) * normal)
        best = None
        for deg in (35, 20, 0, 50):                       # a level hand meets the counter first (P1)
            app = unit(-normal * math.cos(math.radians(deg)) + perp * math.sin(math.radians(deg)))
            for fname, fdir in (("across", np.cross(app, perp)), ("upright", perp)):
                R = self.facing(app, fdir)                # the hand is ~20 cm wide along the finger axis
                touch = centre - R[:, 2] * self.tip_reach
                mg = self.feasible([(touch - R[:, 2] * PRE, R), (touch - R[:, 2] * 0.002, R),
                                    (touch + R[:, 2] * 0.012, R)], allow={gid}, check=(0, 1), width=0.0)
                if best is None or mg > best[0]:
                    best = (mg, R, touch, "%d %s" % (deg, fname))
                    self._via = list(self.via)
            if best[0] > 0.15:
                break
        mg, R, touch, deg = best
        if mg < 0:
            raise PrimitiveError("plan", "no feasible press pose")
        out.update(pose=deg, ik_margin=round(mg, 3))
        self.follow_via(self._via, 1.0)
        self.move("approach_button", touch - R[:, 2] * PRE, R, 1.0, max_steps=400, tol=0.008, ang_tol=0.08)
        if not self.move("press", touch + R[:, 2] * 0.015, R, 1.0, speed=0.008, max_steps=150, stop=flipped):
            raise PrimitiveError("press", "the button did not switch (touching %s)" % self.touching(gid))
        self.hold("dwell", 3, 1.0, R, self.site_pos())
        self.move("withdraw", touch - R[:, 2] * 0.20, R, 1.0, tol=0.02, ang_tol=0.3)
        far = lambda: self.env.unit_state()[name + "_far"]  # noqa: E731
        self.confirm("confirm", lambda: flipped() and far(), centre)
        out["turned_on"] = self.env.unit.turned_on
        return out

    # ---- doors ------------------------------------------------------------------------
    def door_frame(self):
        qa, lo, hi, _dof = self.env.unit.door
        jname = self.m.joint(int(self.m.jnt_qposadr.tolist().index(qa))).name
        axis, anchor, adr, lo, hi = self.joint(jname)
        handle, hR, hsize, hgid = self.geom("u_door_handle_main")
        open_dir = -1.0 if lo < 0 else 1.0               # the joint value moves this way as the door opens
        return jname, axis, anchor, adr, lo, hi, open_dir, handle, hR, hsize, hgid

    def door(self, want):
        if self.env.unit.kind == "microwave":
            return self._side_door(want)
        return self._drop_door(want)

    def _arc(self, axis, anchor, open_dir, dtheta):
        return E.axis_angle_mat(axis * open_dir * dtheta)

    def door_geoms(self, adr):
        """Every geom of the door: the subtree of the door joint's body."""
        j = int(np.where(self.m.jnt_qposadr == adr)[0][0])
        root = self.m.jnt_bodyid[j]
        return set(g for g in range(self.m.ngeom) if self.env._under(self.m, self.m.geom_bodyid[g], root))

    def _side_door(self, want):
        """Microwave. Open: pinch the handle and pull the door part-way round the hinge, then push it the
        rest of the way with the side of the closed fingers from inside the gap (a held handle cannot be
        followed all the way round: the open door's handle ends up beside the robot's base). Close: push
        the door's outer face back round with closed fingertips."""
        jname, axis, anchor, adr, lo, hi, open_dir, handle, hR, hsize, hgid = self.door_frame()
        frac = lambda: E.normalized_joint(self.qpos(adr), lo, hi)  # noqa: E731
        opened = lambda: abs(self.qpos(adr) - (hi if open_dir < 0 else lo))  # noqa: E731
        span = hi - lo
        doorg = self.door_geoms(adr)
        out = {"control": jname, "start_fraction": round(frac(), 4)}
        dmain, dR, dsize, dgid = self.geom("u_door_main")
        n = dR[:, int(np.argmin(dsize))]
        if np.dot(handle - dmain, n) < 0:
            n = -n                                        # the door's outer normal, at its current angle
        v = handle - anchor
        v = v - np.dot(v, axis) * axis
        r_edge = float(np.linalg.norm(v))
        along = v / r_edge                                # hinge -> free edge, in the door plane
        z_mid = float(np.dot(handle - anchor, axis))
        if want == "open":
            thin = int(np.argmin(hsize))
            lat = [i for i in range(3) if i != thin and abs(np.dot(hR[:, i], axis)) < 0.5][0]
            tip_point = handle - n * (hsize[thin] + 0.004)  # fingertips just behind the handle bar
            th_a = 0.45
            best = None
            for deg in (20, 40, 0, -20):
                app = unit(E.axis_angle_mat(axis * math.radians(deg)) @ -n)
                for roll in (0,):                         # fingers close across the handle's width
                    R = self.facing(app, E.axis_angle_mat(app * math.radians(roll)) @ hR[:, lat])
                    grasp = tip_point - R[:, 2] * self.tip_reach
                    poses = [(grasp - R[:, 2] * PRE, R), (grasp, R)]
                    for th in (0.15, 0.3, th_a):
                        Rot = self._arc(axis, anchor, open_dir, th)
                        poses.append((anchor + Rot @ (grasp - anchor), Rot @ R))
                    mg = self.feasible(poses, allow=doorg, check=(0, 1), width=2 * hsize[lat] + 0.025)
                    if best is None or mg > best[0]:
                        best = (mg, R, grasp, "yaw%d roll%d" % (deg, roll))
                        self._via = list(self.via)
                if best[0] > 0.15:
                    break
            mg, R, grasp, how = best
            if mg < 0:
                raise PrimitiveError("plan", "no feasible handle grasp and pull")
            width = 2 * hsize[lat]
            g_open = float(np.clip(1.0 - (width + 0.025) / 0.04, -1.0, 0.6))
            out.update(pull=how, ik_margin=round(mg, 3))
            self.follow_via(self._via, g_open)
            self.move("approach_handle", grasp - R[:, 2] * PRE, R, g_open, max_steps=400, tol=0.008, ang_tol=0.08)
            self.move("to_handle", grasp, R, g_open, speed=0.01, tol=0.005)
            self.hold("grip", 15, 1.0, R, grasp)
            if not self.touching(hgid):
                raise PrimitiveError("grip", "fingers do not touch the handle")
            self.stage = "pull"
            th = 0.0
            for _ in range(400):
                if opened() >= th_a:
                    break
                th = min(th + 0.01, opened() + 0.12, th_a + 0.05)
                Rot = self._arc(axis, anchor, open_dir, th)
                self.tick(anchor + Rot @ (grasp - anchor), Rot @ R, 1.0, speed=0.03, turn=0.2)
            else:
                raise PrimitiveError("pull", "door at %.3f rad" % opened())
            Rn = self.site_R()
            self.hold("release", 10, -1.0, Rn, self.site_pos())
            self.move("ease_off", self.site_pos() - Rn[:, 2] * 0.03, Rn, -1.0, tol=0.01, ang_tol=0.2, max_steps=60)
            # push: closed fingers pointing down in the gap behind the door, the hand's narrow side along the push
            th0 = opened()
            r_push = 0.55 * r_edge
            thick = float(np.min(dsize))
            base_pt = anchor + axis * z_mid + along * r_push - n * (thick + 0.03)   # behind the inner face (closed frame)
            down = np.array([0.0, 0.0, -1.0])
            best = None
            for tilt in (0, 20, -20):
                app = unit(E.axis_angle_mat(along * math.radians(tilt)) @ down)
                Rc = self.facing(app, along)
                site_c = base_pt - Rc[:, 2] * (self.tip_reach - 0.03)   # finger sides at the door, tips 3 cm lower
                poses = []
                Rot0 = self._arc(axis, anchor, open_dir, th0)
                p0 = anchor + Rot0 @ (site_c - anchor)
                poses.append((p0 + np.array([0, 0, 0.12]), Rot0 @ Rc))
                poses.append((p0, Rot0 @ Rc))
                for th in np.linspace(th0, 0.97 * span, 4)[1:]:
                    Rot = self._arc(axis, anchor, open_dir, th)
                    poses.append((anchor + Rot @ (site_c - anchor), Rot @ Rc))
                mg = self.feasible(poses, allow=doorg, check=(0, 1), width=0.0)
                if best is None or mg > best[0]:
                    best = (mg, Rc, site_c, tilt)
                    self._via = list(self.via)
                if mg > 0.15:
                    break
            mg, Rc, site_c, tilt = best
            if mg < 0:
                raise PrimitiveError("plan", "no feasible push from inside (door at %.2f rad)" % th0)
            out.update(push_tilt=tilt, push_margin=round(mg, 3), pulled_to=round(th0, 3))
            Rot0 = self._arc(axis, anchor, open_dir, th0)
            p0 = anchor + Rot0 @ (site_c - anchor)
            self.follow_via(self._via, 1.0)
            self.move("over_gap", p0 + np.array([0, 0, 0.12]), Rot0 @ Rc, 1.0, max_steps=300, tol=0.01, ang_tol=0.1)
            self.move("into_gap", p0, Rot0 @ Rc, 1.0, speed=0.01, tol=0.008, ang_tol=0.1)
            self.stage = "push_open"
            th = th0
            for _ in range(600):
                if frac() >= 0.97:
                    break
                th = min(th + 0.012, opened() + 0.10, 0.99 * span)
                Rot = self._arc(axis, anchor, open_dir, th)
                self.tick(anchor + Rot @ (site_c - anchor), Rot @ Rc, 1.0, speed=0.03, turn=0.2)
            else:
                raise PrimitiveError("push_open", "door at fraction %.3f" % frac())
            Rn = self.site_R()
            self.move("lift_out", self.site_pos() + np.array([0, 0, 0.15]), Rn, 1.0, tol=0.02, ang_tol=0.3)
            self.confirm("confirm", lambda: self.env.unit_state()["door_open"], self.geom("u_door_handle_main")[0])
        else:
            # closing: closed fingers pointing down beside the door's outer face, the finger sides push it round
            cur = opened()
            thick = float(np.min(dsize))
            Rcur = self._arc(axis, anchor, open_dir, cur)
            n0, along0 = Rcur.T @ n, Rcur.T @ along      # the door's frame when closed
            down = np.array([0.0, 0.0, -1.0])
            best = None
            for frac_r in (0.55, 0.45, 0.7):
                base_pt = anchor + axis * z_mid + along0 * (frac_r * r_edge) + n0 * (thick + 0.03)
                for tilt in (0, 20, -20):
                    app = unit(E.axis_angle_mat(along0 * math.radians(tilt)) @ down)
                    Rc = self.facing(app, along0)
                    site_c = base_pt - Rc[:, 2] * (self.tip_reach - 0.03)
                    p0 = anchor + Rcur @ (site_c - anchor)
                    poses = [(p0 + np.array([0, 0, 0.12]), Rcur @ Rc), (p0, Rcur @ Rc)]
                    for th in np.linspace(cur, 0.0, 4)[1:]:
                        Rot = self._arc(axis, anchor, open_dir, th)
                        poses.append((anchor + Rot @ (site_c - anchor), Rot @ Rc))
                    mg = self.feasible(poses, allow=doorg, check=(0, 1), width=0.0)
                    if best is None or mg > best[0]:
                        best = (mg, Rc, site_c, "r%.2f tilt%d" % (frac_r, tilt))
                        self._via = list(self.via)
                    if mg > 0.15:
                        break
                if best[0] > 0.15:
                    break
            mg, Rc, site_c, how = best
            if mg < 0:
                raise PrimitiveError("plan", "no feasible push path")
            out.update(ik_margin=round(mg, 3), push=how)
            p0 = anchor + Rcur @ (site_c - anchor)
            self.follow_via(self._via, 1.0)
            self.move("over_door", p0 + np.array([0, 0, 0.12]), Rcur @ Rc, 1.0, max_steps=400, tol=0.01, ang_tol=0.1)
            self.move("beside_door", p0, Rcur @ Rc, 1.0, speed=0.01, tol=0.008, ang_tol=0.1)
            self.stage = "push_round"
            th = cur
            for _ in range(700):
                if self.env.unit_state()["door_closed"]:
                    break
                th = max(th - 0.012, opened() - 0.10, -0.04)
                Rot = self._arc(axis, anchor, open_dir, th)
                self.tick(anchor + Rot @ (site_c - anchor), Rot @ Rc, 1.0, speed=0.03, turn=0.2)
            else:
                raise PrimitiveError("push_round", "door at fraction %.3f" % frac())
            Rn = self.site_R()
            self.move("lift_out", self.site_pos() + np.array([0, 0, 0.15]), Rn, 1.0, tol=0.02, ang_tol=0.3)
            self.confirm("confirm", lambda: self.env.unit_state()["door_closed"], anchor + axis * z_mid + along * (0.55 * r_edge))
        out["fraction"] = round(frac(), 4)
        return out

    def _drop_door(self, want):
        """Oven or range: drag the handle out from above with closed fingertips until the door's weight
        takes over (pilot primitive); closing lifts the handle back up from below."""
        jname, axis, anchor, adr, lo, hi, open_dir, handle, hR, hsize, hgid = self.door_frame()
        frac = lambda: E.normalized_joint(self.qpos(adr), lo, hi)  # noqa: E731
        out = {"control": jname, "start_fraction": round(frac(), 4)}
        if want != "open":
            raise PrimitiveError("plan", "closing a drop-down door is not implemented (left out of v1)")
        dmain = self.geom("u_door_main")[0]
        outward = handle - dmain
        outward = unit(outward - np.dot(outward, axis) * axis - np.array([0, 0, outward[2]]))
        bar_r = float(min(x for x in hsize if x > 0))
        best = None
        for lean in (25, 40, 10):
            app = unit(np.array([0.0, 0.0, -1.0]) * math.cos(math.radians(lean)) - outward * math.sin(math.radians(lean)))
            R = self.facing(app, axis)
            for along in (0.0, -0.08, 0.08):
                tip = handle + axis * along - outward * (0.65 * bar_r) + np.array([0, 0, 0.55 * bar_r])
                site = tip - R[:, 2] * self.tip_reach
                poses = [(site - R[:, 2] * 0.10, R), (site, R)]
                Rot = self._arc(axis, anchor, open_dir, 0.2)
                poses.append((anchor + Rot @ (site - anchor), R))
                mg = self.feasible(poses, allow=self.body_geoms(hgid), check=(0,), width=0.0)
                if best is None or mg > best[0]:
                    best = (mg, R, site, lean, along)
                    self._via = list(self.via)
            if best[0] > 0.15:
                break
        mg, R, site, lean, along = best
        if mg < 0:
            raise PrimitiveError("plan", "no feasible drag pose")
        out.update(lean_deg=lean, along=along, ik_margin=round(mg, 3))
        self.follow_via(self._via, 1.0)
        self.move("over_handle", site - R[:, 2] * 0.10, R, 1.0, max_steps=400, tol=0.01, ang_tol=0.1)
        if not self.move("touch_handle", site + R[:, 2] * 0.01, R, 1.0, speed=0.01, max_steps=100,
                         stop=lambda: self.touching(hgid)):
            raise PrimitiveError("touch_handle", "fingertips do not reach the handle")
        self.stage = "drag"
        th = 0.0
        for _ in range(400):
            if abs(self.qpos(adr) - lo) >= 0.18:
                break
            th = min(th + 0.004, abs(self.qpos(adr) - lo) + 0.05)
            Rot = self._arc(axis, anchor, open_dir, th)
            self.tick(anchor + Rot @ (site - anchor), R, 1.0, speed=0.02)
        else:
            raise PrimitiveError("drag", "door at %.3f rad" % abs(self.qpos(adr) - lo))
        self.move("clear", self.site_pos() + np.array([0, 0, 0.20]), R, 1.0, tol=0.03, ang_tol=0.5)
        self.stage = "fall"
        for _ in range(200):
            self.tick(self.site_pos(), R, 1.0)
            if frac() >= 0.95:
                break
        self.confirm("confirm", lambda: self.env.unit_state()["door_open"], handle)
        out["fraction"] = round(frac(), 4)
        return out


# ----------------------------------------------------------------------------------------------
# Over the wire: the reference as a submission (solution/), driving its own copy of the scene
# ----------------------------------------------------------------------------------------------
class EpisodeOver(Exception):
    """The grader moved on (reset or close) while a primitive was running."""

    def __init__(self, msg):
        super().__init__(msg.get("type"))
        self.msg = msg


def step_law(obs, target, R, grip, speed, turn, cams):
    """The action for one query: Driver.tick's law, from the QUERIED observation's gripper pose, so a
    repeated query of the same step (the grader's probe) gets an answer consistent with what it shows."""
    ee = np.asarray(obs["ee_pos"], float)
    Ree = np.zeros(9)
    mujoco.mju_quat2Mat(Ree, np.asarray(obs["ee_quat"], float))
    dp = target - ee
    n = np.linalg.norm(dp)
    if n > speed:
        dp *= speed / n
    dr = rotvec(R @ Ree.reshape(3, 3).T)
    n = np.linalg.norm(dr)
    if n > turn:
        dr *= turn / n
    return [round(float(x), 6) for x in list(dp) + list(dr)] + [float(grip), int(cams)]


class Shadow(object):
    """The controller's own copy of the scene, kept in step with the observations: the robot's joints and
    gripper from each observation, the unit's pose and control states from what the controller was told
    (the oracle's token truth) or estimated (a baseline). Looks like an ApplianceEnv to `Driver`, and
    `command` speaks the stdio wire (`read`, `write` are line callables)."""

    def __init__(self, spec, assets, read, write, mount=None):
        self.spec, self.assets, self.mount = spec, assets, mount
        self.read, self.write = read, write
        self.env = None
        self.pending = None

    def __getattr__(self, name):
        return getattr(self.__dict__["env"], name)

    def start(self, msg, jitter):
        self.env = E.ApplianceEnv(self.spec, self.assets, render=False, mount=self.mount, jitter_scale=(0, 0, 0))
        self.env.reset(0, jitter=jitter)
        self.pending = msg
        self.sync(msg["obs"])

    def sync(self, obs):
        env = self.env
        m, d = env.model, env.data
        d.qpos[env.jq] = obs["q"]
        d.qvel[env.jv] = obs["dq"]
        w = float(obs["gripper"]) / 2
        d.qpos[env.gq[0]], d.qpos[env.gq[1]] = w, -w
        truth = obs.get("truth")
        if truth:
            for name, v in truth["joints"].items():
                d.qpos[m.joint("u_" + name).qposadr[0]] = v
            env.unit.turned_on = bool(truth["turned_on"])
        mujoco.mj_forward(m, d)
        if not truth and getattr(self, "hook", None) is not None:
            self.hook(self, obs)            # a controller without the truth keeps the unit's state itself
        env.t = int(obs["t"])

    @property
    def obs(self):
        return self.pending["obs"]

    def command(self, target, R, grip, speed, turn, cams):
        """Answer the pending step (and any repeat of it) with the law; return the next step's observation."""
        t0 = self.pending["t"]
        self.write({"action": step_law(self.pending["obs"], target, R, grip, speed, turn, cams)})
        while True:
            msg = self.read()
            if msg.get("type") != "act":
                raise EpisodeOver(msg)
            if int(msg["t"]) == t0:
                self.write({"action": step_law(msg["obs"], target, R, grip, speed, turn, cams)})
                continue
            self.pending = msg
            self.sync(msg["obs"])
            return msg["obs"]

