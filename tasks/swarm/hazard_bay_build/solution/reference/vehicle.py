"""The reference ground vehicle's program. Its kinematics come from the scene it was built in (`scene.usda` beside
it): the joints' frames read with pxr, forward kinematics, a damped least-squares IK on the grip point (with the
tool's axis held, where it matters).

Two programs:

- `Privileged`, the reference's own: with the true state in each observation (`privileged`, which the grader adds
  for the reference's token only), it does the whole job on the day of the run: the valve pushed shut round the
  lever's true hub (ending as soon as the lever reads shut, backing off along the arc), then every part off the tables
  and into the crate (a stand at a table's edge per part, chosen so the arm reaches it from there and from a little
  beside it, the vehicle stays outside the burning pan's ring and has room to turn at its approach; a path on a grid
  of the bay, kept a little further from the furniture where there is room, driven backward when the way lies behind,
  the goal itself approached by turning to it and driving straight at it, the last stretch to a stand driven along
  the stand's line; a pick from above with the suction tool at the top centre of the part's true box, whether it
  stands or lies, planned from where the vehicle really stands and solved again on the way down whenever the base has
  moved (the arm's reaction pushes a skid-steer's base, and its wheels slip); the part then lifted and carried in the
  pose it was taken in (the grip link's turn about the tool's axis held with the last joint: the IK leaves it free,
  and with the tool leaning it would tip the part); a drop planned on the part's box as the arm will hold it over its
  spot (the grader holds the part rigidly in the grip link's frame), the part level again as it lay on its table and
  squared with the crate's walls, where that box fits inside the walls and overlaps the parts already in least (the
  crate filled from its corners), let go 3 cm over the rim and only once the part is over the crate (from too far
  off, the vehicle drives to the crate's stand again first)),
  then it stops where it stands with its arm folded (after the last part, at the crate: it does not back out).
  While the pan burns and no side of the crate lies outside its ring, the vehicle holds its part and waits.
  It turns on the spot only where there is room: the skid-steer turns about a point some way off its root body, so a
  spin walks the vehicle. It keeps clear of the bucket wherever it is (hanging or set down), of the stretches the
  hoist says the bucket is about to cross and of the flying robots that fly low (over the set-down bucket); a part
  that no stand reaches is given up only once the hoist has settled (until then the bucket may be what is in the way).
- `Baseline`, the program without that block: it closes the valve on its odometry and parks (the part routine is
  kept but switched off: on raw wheel odometry the vehicle drifts by tens of centimetres over the minutes the parts
  take and runs into the tables; doing the station without the true state needs localisation against what the
  cameras see, which is the agent's job).
"""
import math
import os
import sys

import numpy as np

# the bay, as the reference built it (world frame)
BAY_PARK = (-2.80, 2.20)
LEVER_STEM = np.array([-1.20, 3.20, 0.95])  # the valve's stem; the lever's grip is 0.22 m up it, closing swings it to +x
GRIP_ABOVE_STEM = 0.1955  # [m] where the cameras report the lever's grip, above the stem, with the lever open
LOOK_HEADING = 60.0  # [deg] the heading at the valve that puts the lever in the mast camera's view (it looks 62 deg right)
VALVE_STAND = (-1.20, 2.62)
CRATE = np.array([0.10, 1.50])
CRATE_TURN, CRATE_STAND = (0.10, 0.55), (0.10, 0.92)
WEST_LANE, SOUTH_LANE = -2.35, -0.55  # the routes round Table A (west of it) and along the tables' south side
TABLE_A_ITEMS = ("cracker_box", "mustard_bottle", "sugar_box")
TABLE_B_ITEMS = ("soup_can", "power_drill", "mug")
SPEED, TURN = 0.6, 1.2  # [m/s] cruise, [rad/s] the largest turn rate
ARM_RATE = 0.9  # [rad/s] the largest joint rate the arm targets move at
CLEAR_STATION = False  # (see the docstring)


def log(*a):
    print("[vehicle]", *a, file=sys.stderr, flush=True)
    if os.environ.get("ARB_REFERENCE_LOG"):  # (a maintainer's record of a run: the grader discards stderr)
        try:
            with open(os.environ["ARB_REFERENCE_LOG"], "a") as f:
                print("[vehicle]", *a, file=f)
        except OSError:
            pass


# --------------------------------------------------------------------------------------------- kinematics
def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tf(p, q):
    T = np.eye(4)
    T[:3, :3] = quat_mat(q)
    T[:3, 3] = p
    return T


def axis_rot(axis, a):
    k = np.asarray(axis, dtype=float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    return T


class Kinematics:
    """The vehicle's chain from its root body to the grip link, read from the scene's joints."""

    def __init__(self, scene, root_body, grip_link):
        from pxr import Usd, UsdPhysics
        st = Usd.Stage.Open(scene)
        edges = []
        for p in Usd.PrimRange(st.GetPrimAtPath("/World/Fleet/UGV"), Usd.TraverseInstanceProxies()):
            if not (p.IsA(UsdPhysics.Joint) and p.IsActive()):
                continue
            j = UsdPhysics.Joint(p)
            b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
            b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
            if not b0 or not b1:
                continue

            def q4(v):
                return (v.GetReal(), *v.GetImaginary()) if v is not None else (1.0, 0.0, 0.0, 0.0)
            J0 = tf(j.GetLocalPos0Attr().Get() or (0, 0, 0), q4(j.GetLocalRot0Attr().Get()))
            J1 = tf(j.GetLocalPos1Attr().Get() or (0, 0, 0), q4(j.GetLocalRot1Attr().Get()))
            rev = p.IsA(UsdPhysics.RevoluteJoint)
            axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[p.GetAttribute("physics:axis").Get() or "X"] if rev else None
            edges.append(dict(name=p.GetName(), b0=b0[0], b1=b1[0], J0=J0, J1=J1, axis=axis))
        bodies = sorted({e["b0"] for e in edges} | {e["b1"] for e in edges})  # (sorted: the same pick in every process)
        # the root: the articulation root's body (an arm asset may carry a body of the same name)
        arts = [p.GetPath().pathString for p in Usd.PrimRange(st.GetPrimAtPath("/World/Fleet/UGV"), Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        root = next((b for b in bodies if b in arts), None) or next(b for b in sorted(bodies, key=len) if b.endswith("/" + root_body))
        grip = next(b for b in bodies if b.endswith("/" + grip_link))
        # the path root -> grip through the joint graph
        prev = {root: None}
        frontier = [root]
        while frontier and grip not in prev:
            nxt = []
            for b in frontier:
                for e in edges:
                    for a, c, fwd in ((e["b0"], e["b1"], True), (e["b1"], e["b0"], False)):
                        if a == b and c not in prev:
                            prev[c] = (b, e, fwd)
                            nxt.append(c)
            frontier = nxt
        chain = []
        b = grip
        while prev[b] is not None:
            a, e, fwd = prev[b]
            chain.append((e, fwd))
            b = a
        self.chain = chain[::-1]
        for e, _ in self.chain:
            e["J0i"], e["J1i"] = np.linalg.inv(e["J0"]), np.linalg.inv(e["J1"])
        self.joints = [e["name"] for e, _ in self.chain if e["axis"] is not None]
        # the wheels: revolute joints off the root that are not on the arm's chain
        self.wheels = {}
        for e in edges:
            if e["axis"] is not None and e["b0"] == root and e["name"] not in self.joints:
                T = e["J0"]
                axis_root = T[:3, :3] @ np.array(e["axis"], dtype=float)
                self.wheels[e["name"]] = dict(pos=T[:3, 3].copy(), axis=axis_root)
        p, w = self.points({})
        self.wrist_len = float(np.linalg.norm(p - w))  # (the grip point lies on the last joint's axis: a constant)

    def frames(self, q):
        """The grip link's frame in the root's frame and, for every revolute joint on the chain, its origin and its
        (signed) axis, for joint positions q (a dict by name)."""
        T = np.eye(4)
        joints = []
        for e, fwd in self.chain:
            a = q.get(e["name"], 0.0) if e["axis"] is not None else 0.0
            if fwd:
                Tj = T @ e["J0"]
                if e["axis"] is not None:
                    joints.append((Tj[:3, 3].copy(), Tj[:3, :3] @ np.asarray(e["axis"], dtype=float)))
                    T = Tj @ axis_rot(e["axis"], a) @ e["J1i"]
                else:
                    T = Tj @ e["J1i"]
            else:
                Tj = T @ e["J1"]
                if e["axis"] is not None:
                    joints.append((Tj[:3, 3].copy(), -(Tj[:3, :3] @ np.asarray(e["axis"], dtype=float))))
                    T = Tj @ np.linalg.inv(axis_rot(e["axis"], a)) @ e["J0i"]
                else:
                    T = Tj @ e["J0i"]
        return T, joints

    def points(self, q):
        """(the grip point, the last revolute joint's origin: the wrist) in the root's frame."""
        T, js = self.frames(q)
        return T[:3, 3].copy(), js[-1][0] if js else T[:3, 3].copy()

    def solve(self, target, q0, limits, wrist_dir=None, posture=None, iters=120, tol=0.002, wtol=0.006):
        """Damped least squares on the grip point and, with `wrist_dir` (a unit vector in the root's frame), on the
        wrist, held `wrist_len` from the grip point along that direction (the tool's axis); a light pull toward
        `posture` in the null space. Returns (q, grip residual, wrist residual)."""
        names = self.joints
        q = np.array([q0.get(n, 0.0) for n in names], dtype=float)
        lo = np.array([(limits.get(n) or [None, None])[0] if (limits.get(n) or [None, None])[0] is not None else -2 * math.pi for n in names])
        hi = np.array([(limits.get(n) or [None, None])[1] if (limits.get(n) or [None, None])[1] is not None else 2 * math.pi for n in names])
        lo, hi = np.maximum(lo, -2 * math.pi) + 1e-3, np.minimum(hi, 2 * math.pi) - 1e-3
        q = np.clip(q, lo, hi)
        pq = np.array([posture.get(n, 0.0) for n in names]) if posture else None
        target = np.asarray(target, dtype=float)
        wt = target + self.wrist_len * np.asarray(wrist_dir, dtype=float) if wrist_dir is not None else None
        best = (q.copy(), 1e9, 1e9)
        for _ in range(iters):
            T, js = self.frames(dict(zip(names, q)))
            p = T[:3, 3]
            e = target - p
            err = float(np.linalg.norm(e))
            werr = 0.0
            J = np.array([np.cross(ax, p - o) for o, ax in js]).T
            if wt is not None:
                w = js[-1][0]
                ew = wt - w
                werr = float(np.linalg.norm(ew))
                Jw = np.array([np.cross(ax, w - o) for o, ax in js]).T
                J = np.vstack([J, Jw])
                e = np.concatenate([e, ew])
            if err + 0.5 * werr < best[1] + 0.5 * best[2]:
                best = (q.copy(), err, werr)
            if err < tol and werr < wtol:
                break
            lam = 0.03
            JJ = J @ J.T + lam * lam * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJ, e)
            if pq is not None:
                N = np.eye(len(names)) - J.T @ np.linalg.solve(JJ, J)
                dq = dq + 0.15 * N @ (pq - q)
            dq = np.clip(dq, -0.2, 0.2)
            q = np.clip(q + dq, lo, hi)
        return dict(zip(names, best[0])), best[1], best[2]

    def fk(self, q):
        """The grip point in the root's frame for joint positions q (a dict by name)."""
        T = np.eye(4)
        for e, fwd in self.chain:
            a = q.get(e["name"], 0.0) if e["axis"] is not None else 0.0
            R = axis_rot(e["axis"], a) if e["axis"] is not None else np.eye(4)
            if fwd:
                T = T @ e["J0"] @ R @ np.linalg.inv(e["J1"])
            else:
                T = T @ e["J1"] @ np.linalg.inv(R) @ np.linalg.inv(e["J0"])
        return T[:3, 3].copy()

    def ik(self, target, q0, limits, iters=150, posture=None):
        """Damped least squares on the grip point (position only), within the joints' limits; a light pull toward
        `posture` in the null space. Returns (q, residual)."""
        names = self.joints
        q = np.array([q0.get(n, 0.0) for n in names], dtype=float)
        lo = np.array([limits.get(n, [-1e3, 1e3])[0] if limits.get(n, [None])[0] is not None else -1e3 for n in names])
        hi = np.array([limits.get(n, [-1e3, 1e3])[1] if limits.get(n, [None, None])[1] is not None else 1e3 for n in names])
        lo, hi = np.maximum(lo, -2 * math.pi), np.minimum(hi, 2 * math.pi)
        pq = np.array([posture.get(n, 0.0) for n in names]) if posture else None
        best = (q.copy(), 1e9)
        for _ in range(iters):
            qd = dict(zip(names, q))
            p = self.fk(qd)
            e = np.asarray(target) - p
            err = float(np.linalg.norm(e))
            if err < best[1]:
                best = (q.copy(), err)
            if err < 0.003:
                break
            J = np.zeros((3, len(names)))
            for i in range(len(names)):
                qd2 = dict(qd)
                qd2[names[i]] += 1e-4
                J[:, i] = (self.fk(qd2) - p) / 1e-4
            lam = 0.05
            JJ = J @ J.T + lam * lam * np.eye(3)
            dq = J.T @ np.linalg.solve(JJ, e)
            if pq is not None:
                N = np.eye(len(names)) - J.T @ np.linalg.solve(JJ, J)
                dq = dq + 0.1 * N @ (pq - q)
            dq = np.clip(dq, -0.25, 0.25)
            q = np.clip(q + dq, lo, hi)
        return dict(zip(names, best[0])), best[1]


_KINEMATICS = {}


def kinematics(scene, root_body, grip_link):
    """The vehicle's chain read from the scene, once per process (the controller reads it before it says READY: with
    the machine loaded, pxr and the stage can take longer than a control step may)."""
    key = (scene, root_body, grip_link)
    if key not in _KINEMATICS:
        _KINEMATICS[key] = Kinematics(scene, root_body, grip_link)
    return _KINEMATICS[key]


# --------------------------------------------------------------------------------------------- the program
class Baseline:
    """The vehicle's baseline program (no true state). `state` is a plain dict (the controller deep-copies it for
    repeated queries)."""

    def __init__(self):
        self.kin = None
        self.mis = None

    def setup(self, mis):
        self.mis = mis
        u = mis.get("ugv")
        if not u:
            return
        scene = os.path.join(os.getcwd(), "scene.usda")
        try:
            self.kin = kinematics(scene, "base_link", u["grip_link"])
            log("arm joints", self.kin.joints, "wheels", {k: [round(float(x), 3) for x in v["pos"]] for k, v in self.kin.wheels.items()})
        except Exception as exc:  # noqa: BLE001
            log("no kinematics:", exc)
            self.kin = None
        self.limits = u.get("limits") or {}
        if self.kin and self.kin.wheels:
            ws = list(self.kin.wheels.values())
            self.track = float(np.mean([abs(w["pos"][1]) for w in ws])) * 2
            self.radius = 0.098

    def init_state(self):
        return dict(step=0, steps=self.plan(), sub=None, t_step=0.0, q_tgt=None, hold=None, done=False, lever_done_t=None,
                    log=[], grip=False, items_left=list(TABLE_A_ITEMS + TABLE_B_ITEMS))

    def plan(self):
        """The task list: (kind, args)."""
        st = []
        # the valve: east along the bay's lane, north to the lever, push it closed
        st += [("go", (BAY_PARK[0], 2.20, None)), ("go", (VALVE_STAND[0], 2.20, None)), ("face", 0.0),
               ("go", (VALVE_STAND[0], VALVE_STAND[1], 0.0)), ("face", LOOK_HEADING), ("look_lever", None), ("face", 0.0),
               ("lever", None), ("arm_home", None),
               ("go", (VALVE_STAND[0], 2.20, 0.0)), ("mark", "valve")]
        if not CLEAR_STATION:  # back to the bay and stay there
            st += [("go", (BAY_PARK[0], 2.20, None)), ("face", 0.0), ("arm_home", None), ("mark", "done")]
            return st
        # Table A: round its west end to the south lane; each part: in from the south, pick, out, east, north to the crate
        st += [("go", (WEST_LANE, 2.20, None)), ("go", (WEST_LANE, SOUTH_LANE, None))]
        for name in TABLE_A_ITEMS:
            st += [("part", name)]
        st += [("wait_hoist", None)]
        for name in TABLE_B_ITEMS:
            st += [("part", name)]
        st += [("arm_home", None), ("mark", "done")]
        return st

    # -- helpers ----------------------------------------------------------------------------------------------
    def wheels(self, v, w):
        """Wheel speed targets for a forward speed v [m/s] and a turn rate w [rad/s] (skid steering)."""
        out = {}
        for name, wh in self.kin.wheels.items():
            side = 1.0 if wh["pos"][1] > 0 else -1.0  # (left: +y in the root's frame)
            sign = 1.0 if wh["axis"][1] > 0 else -1.0
            out[name] = sign * (v - side * w * self.track * 0.75) / self.radius
        return out

    @staticmethod
    def pose(obs):
        x, y, yaw = obs["ugv"]["pose"]
        return np.array([x, y]), math.radians(yaw)

    def to_root(self, obs, p_world):
        """A world point in the root's frame (the root's +x is the vehicle's forward: heading + 90 degrees)."""
        xy, yaw = self.pose(obs)
        a = yaw + math.pi / 2
        d = np.asarray(p_world[:2]) - xy
        c, s = math.cos(-a), math.sin(-a)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], p_world[2] - self.root_z])

    def drive_to(self, obs, x, y, yaw_end):
        """One step toward (x, y) and then the heading yaw_end (degrees, None: as arrived). Returns (cmd, done)."""
        xy, yaw = self.pose(obs)
        d = np.array([x, y]) - xy
        dist = float(np.linalg.norm(d))
        if dist > 0.04:
            want = math.atan2(d[1], d[0]) - math.pi / 2
            back = False
            err = (want - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(err) > math.radians(100) and dist < 0.8:  # (short moves straight back: reverse instead of turning)
                back = True
                err = (want + math.pi - yaw + math.pi) % (2 * math.pi) - math.pi
            w = float(np.clip(2.0 * err, -TURN, TURN))
            if abs(err) > math.radians(25):  # (turn in place, never slower than skid steering can turn)
                w = math.copysign(max(abs(w), 0.5), err)
                v = 0.0
            else:  # (drive and steer)
                v = min(SPEED, 0.9 * dist + 0.08) * math.cos(err)
            return self.wheels(-v if back else v, w), False
        if yaw_end is not None:
            err = (math.radians(yaw_end) - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(err) > math.radians(3.0):
                w = float(np.clip(2.0 * err, -TURN, TURN))
                return self.wheels(0.0, math.copysign(max(abs(w), 0.35), err)), False
        return self.wheels(0.0, 0.0), True

    def arm_toward(self, s, obs, q_goal, rate=ARM_RATE):
        """Move the arm's targets toward q_goal at a bounded rate; True when the joints are there."""
        dt = self.mis["dt"]
        joints = obs["ugv"]["joints"]
        if s["q_tgt"] is None:
            s["q_tgt"] = {n: joints[n][0] for n in self.kin.joints}
        for n in self.kin.joints:
            d = q_goal[n] - s["q_tgt"][n]
            s["q_tgt"][n] += float(np.clip(d, -rate * dt, rate * dt))
        # there: the targets have arrived at the goal and the joints have followed them
        there = all(abs(q_goal[n] - s["q_tgt"][n]) < 1e-6 and abs(q_goal[n] - joints[n][0]) < 0.03 for n in self.kin.joints)
        return there

    # -- one step -------------------------------------------------------------------------------------------------
    def act(self, s, obs, hoist_phase):
        if self.kin is None or "ugv" not in obs:
            return {}
        g = obs["ugv"]
        dt = self.mis["dt"]
        if s["step"] == 0 and s["t_step"] == 0.0 and s.get("home") is None:
            s["home"] = {n: g["joints"][n][0] for n in self.kin.joints}
            self.root_z = self.mis["ugv"]["start"][2]
            p = self.kin.fk(s["home"])
            log("fk check: grip point", [round(float(v), 3) for v in p], "observed", g["grip_point"])
        if not hasattr(self, "root_z"):
            self.root_z = self.mis["ugv"]["start"][2]
        s["t_step"] += dt
        cmd = {"vel": {}, "pos": {}, "grip": s["grip"]}
        if s["step"] >= len(s["steps"]) or g.get("disabled"):
            cmd["vel"] = self.wheels(0.0, 0.0)
            if s["q_tgt"]:
                cmd["pos"] = dict(s["q_tgt"])
            return cmd
        kind, arg = s["steps"][s["step"]]
        done = False
        wheels = self.wheels(0.0, 0.0)
        if kind == "go":
            wheels, done = self.drive_to(obs, *arg)
        elif kind == "face":
            wheels, done = self.drive_to(obs, *self.pose(obs)[0], arg)
        elif kind == "mark":
            done = True
        elif kind == "arm_home":
            done = self.arm_toward(s, obs, s["home"]) or s["t_step"] > 12.0
        elif kind == "wait_hoist":
            done = hoist_phase in ("RECOVER",) or s["t_step"] > 150.0
        elif kind == "look_lever":  # the lever's grip as the camera sees it, kept in the odometry's frame
            seen = [d["rel"] for d in obs["ugv"].get("sees") or [] if d["what"] == "lever"]
            if seen:
                xy = self.pose(obs)[0]
                g = np.array([xy[0] + seen[0][0], xy[1] + seen[0][1], self.root_z + seen[0][2]])
                s.setdefault("lever_seen", []).append(g.tolist())
            done = len(s.get("lever_seen", [])) >= 30 or s["t_step"] > 3.0
            if done:
                if s.get("lever_seen"):
                    g = np.mean(np.array(s["lever_seen"]), axis=0)
                    s["stem"] = [float(g[0]), float(g[1]), float(g[2] - GRIP_ABOVE_STEM)]
                    log(f"lever seen: stem at {np.round(s['stem'], 3).tolist()} (odometry frame)")
                else:
                    log("lever not seen: pushing where the map says it is")
        elif kind == "lever":
            done = self.lever(s, obs)
            if done and s.get("lever_done_t") is None:
                s["lever_done_t"] = obs["time"]
                log(f"lever pushed round at t={obs['time']:.1f}")
        elif kind == "part":
            wheels, done = self.part(s, obs, arg)
        if s["q_tgt"]:
            cmd["pos"] = dict(s["q_tgt"])
        cmd["vel"] = wheels
        cmd["grip"] = s["grip"]
        if done:
            xy, yaw = self.pose(obs)
            log(f"t={obs['time']:.1f} done {kind} {arg} at ({xy[0]:.2f}, {xy[1]:.2f}) heading {math.degrees(yaw):.0f}")
            s["step"] += 1
            s["t_step"] = 0.0
            s["sub"] = None
        return cmd

    def lever(self, s, obs):
        """Push the lever round its quarter turn: the grip point along an arc 0.17 m from the stem, from just on the
        lever's -x side to past its closed stop."""
        if s["sub"] is None:
            s["sub"] = dict(k=0, q=None)
        sub = s["sub"]
        n_way = 12
        if sub["k"] > n_way + 2:
            return True
        a = math.radians(-22.0 + (100.0 + 22.0) * min(sub["k"], n_way) / n_way)
        r = 0.24  # (at the lever's grip end: the open and closed stops sit close round the stem)
        stem = np.array(s["stem"]) if s.get("stem") else LEVER_STEM
        p = stem + np.array([r * math.sin(a), 0.0, r * math.cos(a)])
        if sub["k"] == 0:
            p = p + np.array([-0.04, -0.12, 0.0])  # (first in front of it, then into the lever's plane)
        if sub["k"] == n_way + 1:  # (then straight back out of the lever's plane, away from the line ...)
            p = stem + np.array([r * math.sin(a), -0.25, r * math.cos(a)])
        if sub["k"] == n_way + 2:  # (... and up, before the arm folds)
            p = stem + np.array([0.0, -0.30, 0.35])
        if sub["q"] is None:
            q0 = s["q_tgt"] or {n: obs["ugv"]["joints"][n][0] for n in self.kin.joints}
            sub["q"], res = self.kin.ik(self.to_root(obs, p), q0, self.limits)
            if res > 0.03:
                log(f"lever waypoint {sub['k']}: IK residual {res:.3f} m")
        there = self.arm_toward(s, obs, sub["q"], rate=0.6)
        if there or s["t_step"] > 6.0:
            log(f"t={obs['time']:.1f} lever waypoint {sub['k']} {'reached' if there else 'timed out'}; lever seen at "
                f"{next((d['rel'] for d in obs['ugv'].get('sees') or [] if d['what'] == 'lever'), None)}")
            sub["k"] += 1
            sub["q"] = None
            s["t_step"] = 0.0
        return False

    def part(self, s, obs, name):
        """One part from its table to the crate: in from the south lane, pick it from above, back out, east along the
        lane, north to the crate, drop it in, back to the lane."""
        if s["sub"] is None:
            items = self.mis.get("items") or []
            if name not in items:
                return self.wheels(0.0, 0.0), True
            pos = None
            for d in obs["ugv"].get("sees") or []:
                if d["what"] == f"item:{name}":
                    pos = self.pose(obs)[0].tolist() + [0.0]
                    pos = np.array([pos[0] + d["rel"][0], pos[1] + d["rel"][1], self.root_z + d["rel"][2]])
            if pos is None:
                pos = np.array(ITEMS_AT.get(name, (0.0, 0.0, 0.8)))
            table_south = 0.40 if name in TABLE_A_ITEMS else 0.13
            top = pos[2] + ITEM_HALF_H.get(name, 0.08)
            s["sub"] = dict(k=0, item=pos.tolist(), top=top, x=float(pos[0]), stand_y=table_south - 0.32, q=None, t=0.0)
        sub = s["sub"]
        x, sy = sub["x"], sub["stand_y"]
        legs = [("go", (x, SOUTH_LANE, 90.0 - 90.0)), ("go", (x, sy, 0.0)),
                ("arm", "above"), ("arm", "touch"), ("grip", True), ("arm", "lift"), ("arm", "carry"),
                ("go", (x, SOUTH_LANE, None)), ("go", (CRATE_TURN[0], SOUTH_LANE, None)), ("go", (CRATE_TURN[0], CRATE_TURN[1], 0.0)),
                ("go", (CRATE_STAND[0], CRATE_STAND[1], 0.0)), ("arm", "crate"), ("grip", False), ("arm", "carry"),
                ("go", (CRATE_TURN[0], CRATE_TURN[1], None)), ("go", (CRATE_TURN[0], SOUTH_LANE, None))]
        if sub["k"] >= len(legs):
            return self.wheels(0.0, 0.0), True
        kind, arg = legs[sub["k"]]
        sub["tl"] = sub.get("tl", 0.0) + self.mis["dt"]
        wheels, done = self.wheels(0.0, 0.0), False
        if kind == "go":
            wheels, done = self.drive_to(obs, *arg)
        elif kind == "grip":
            s["grip"] = arg
            sub["t"] += self.mis["dt"]
            done = sub["t"] > 0.5
            if arg and obs["ugv"].get("holding") is None and 0.4 < sub["t"] <= 0.4 + self.mis["dt"]:
                xy, yaw = self.pose(obs)
                gp = np.array(obs["ugv"]["grip_point"])
                a = yaw + math.pi / 2
                gw = np.array([xy[0] + math.cos(a) * gp[0] - math.sin(a) * gp[1], xy[1] + math.sin(a) * gp[0] + math.cos(a) * gp[1], self.root_z + gp[2]])
                log(f"{name}: nothing held after the touch; tool at {np.round(gw, 3).tolist()}, part top at "
                    f"{np.round([sub['item'][0], sub['item'][1], sub['top']], 3).tolist()}")
        elif kind == "arm":
            if sub["q"] is None:
                item = np.array(sub["item"])
                if arg == "above":
                    p = np.array([item[0], item[1], sub["top"] + 0.12])
                elif arg == "touch":
                    p = np.array([item[0], item[1], sub["top"] + 0.01])
                elif arg == "lift":
                    p = np.array([item[0], item[1], sub["top"] + 0.30])
                elif arg == "carry":
                    xy, yaw = self.pose(obs)
                    fwd = np.array([-math.sin(yaw), math.cos(yaw)])
                    p = np.array([xy[0] + 0.30 * fwd[0], xy[1] + 0.30 * fwd[1], 1.15])
                else:  # crate
                    p = np.array([CRATE[0], CRATE[1] - 0.05, 0.62])
                q0 = s["q_tgt"] or {n: obs["ugv"]["joints"][n][0] for n in self.kin.joints}
                sub["q"], res = self.kin.ik(self.to_root(obs, p), q0, self.limits, posture=s["home"])
                if res > 0.03:
                    log(f"{name} {arg}: IK residual {res:.3f} m")
            done = self.arm_toward(s, obs, sub["q"]) or s["t_step"] > 8.0
            if done:
                sub["q"] = None
        if done:
            log(f"t={obs['time']:.1f} {name} leg {sub['k']} {kind} {arg if kind != 'go' else tuple(round(v, 2) if v is not None else None for v in arg)} "
                f"done in {sub['tl']:.1f} s")
            sub["k"] += 1
            sub["t"] = 0.0
            sub["tl"] = 0.0
            s["t_step"] = 0.0
        return wheels, False


# where the reference put the parts (its own scene: the program may know what it built), their half heights
ITEMS_AT = {"cracker_box": (-1.62, 0.62, 0.856), "mustard_bottle": (-1.18, 0.80, 0.847), "sugar_box": (-0.90, 0.58, 0.837),
            "soup_can": (1.30, 0.50, 0.501), "power_drill": (1.62, 0.66, 0.546), "mug": (1.42, 0.34, 0.493)}
ITEM_HALF_H = {"cracker_box": 0.107, "mustard_bottle": 0.096, "sugar_box": 0.088, "soup_can": 0.051, "power_drill": 0.093, "mug": 0.059}


# ============================================================================================= the privileged program
# the real bay as the reference built it (the reference's twin is the real bay; what moves per day comes from the block)
TANK_RECT = ((-1.90, -3.10), (0.80, 0.60), 0.0)
LINE_RECTS = [((0.0, 3.36), (7.40, 0.20), 0.0),  # the steam line: pipe, supports and feet, the valve's branch
              ((-0.60, 3.26), (0.08, 0.12), 0.0)]  # the gauge's stub
CRATE_RIBS = 0.025  # [m] the handle ribs stand out of the crate's long sides
FOOT = (0.51, 0.43)  # [m] the vehicle's footprint (length along its heading, width), the arm tucked above it
INFLATE = 0.45  # [m] the planner keeps the vehicle's centre this far from everything (its half-diagonal, and room for a
                # spin, which walks the root body: the skid-steer turns about a point some way off it)
INFLATE_SOFT = 0.55  # [m] where the bay leaves the room, the paths keep this far from everything (the skid-steer swings some way
                     # off its path as it turns; the tighter INFLATE only where there is no other way)
PAN_SIZE = (0.60, 0.60)  # [m] the fire pan's footprint (square, its sides along the bay's axes)
RING = 2.30  # [m] the vehicle's centre keeps out of this circle round the burning pan (the rule: 2 m)
BUCKET_ROOM = 1.0  # [m] and out of this circle round the bucket, hanging or set down (it hangs about a metre up; the mast is higher)
STAND = 0.37  # [m] the vehicle's centre outside a table's (or the crate's) edge when it works there
APPROACH = 0.28  # [m] further out: where it turns to face the edge before it drives in straight
APPROACH_MORE = (0.0, 0.12, 0.27)  # [m] where there is no room to turn there, this much further out along the same line
GRID = (-4.0, 4.0, -0.70, 3.10, 0.05)  # the planner's grid: x0, x1, y0, y1, cell [m]
TUCK = (0.05, 0.0, 0.78)  # [m] the tool's point, the arm tucked above the deck (root frame); holding a part: CARRY
CARRY = (0.10, 0.0, 0.86)
V_DRIVE, V_SLOW, W_TURN = 0.7, 0.3, 0.8  # [m/s] cruise, the creep before a spin; [rad/s] turn rate (a skid-steer's spin walks)
V_STRAIGHT = 0.4  # [m/s] the straight stretch in to a stand and out again
LINE_LOOK, LINE_STEER = 0.5, math.radians(14.0)  # the straight stretch's pursuit: look-ahead [m], the largest steer off its line
                                                 # (moderate: a skid-steer that steers hard at a crawl only turns on the spot)
VALVE_AT, VALVE_APPROACH = (-1.15, 2.62), (-1.15, 2.05)  # [m] the valve's stand and the approach before it. Coming from the
                                                 # parking bay in the west, the vehicle turns north at the approach and the
                                                 # spin walks it 5-11 cm west, which the straight stretch takes up only in
                                                 # part: the stand lies east of the stem by about what is left (pushed from
                                                 # 11 cm or more west of the stem, the arm has caught the lever as it flew
                                                 # shut and pulled the vehicle into the valve)
WHEEL_SLEW = 0.12  # [rad/s per control step] the wheels' speed targets change gently (7.2 rad/s^2: about 0.7 m/s^2 at the rim)
ARM_FAST, ARM_SLOW = 1.0, 0.5  # [rad/s] joint target rates: free moves, moves near the parts and the lever
LEVER_R, LEVER_STEP = 0.24, math.radians(4.0)  # the push's radius round the hub; its arc step
SLOT_MARGIN = 0.05  # [m] a dropped part's box, as it will hang at its spot, stays this far inside the crate's outer box (the
                    # check before the release wants 0.035: the rest is for the base's slip and the arm's tracking)
TRANSIT, RELEASE = 0.10, 0.03  # [m] the held part's bottom over the crate's rim (or the parts under it): on its way over to its
                               # spot, and where it is let go
UAV_ROOM = 0.62  # [m] a flying robot's airframe reaches this far from its centre (its rotors; the landing gear below)
UAV_LOW = 2.25  # [m] a flying robot whose centre is lower than this (the set-down, a pouring hook) can meet the vehicle (its gear
                # hangs 0.47 m below its centre, the vehicle's mast stands 1.64 m): the vehicle keeps clear of it as of a table
SPIN_ROOM = 0.58  # [m] the vehicle spins on the spot only this far from everything (its half-diagonal and the spin's walk)
PRE_ABOVE, TOUCH = 0.16, 0.005  # [m] above a part's top: where the arm comes in; where the tool touches (the suction holds at 4 cm)
TURN_DRIFT, TURN_MAX = 0.6, 8.0  # [m] a turn at the goal that has carried the vehicle this far (more than it coasts after
                                 # arriving), or has lasted this long [s] (a wheel slipping: the vehicle crawls off in an arc),
                                 # is stopped and the goal driven to again
NEAR_GOAL, NEAR_MAX = 0.35, 6.0  # [m] within this of its goal the vehicle turns to it and drives straight at it; [s] after this
                                 # long within 0.5 m of it (hunting), within 0.25 m is taken as there
BAD_STAND_S = 40.0  # [s] how long a stand the arm missed a part from is left alone
PICK_REDO, CREEP = 0.10, 0.06  # [m] a pick out of reach from further than this off its stand: out and in again first; else the
                               # vehicle creeps this much closer to the table and tries again (once each)
STAND_OFF = 0.25  # [m] further than this from the crate's stand, the vehicle drives there again before it reaches over the crate
STAND_SLIP, FRAGILE = 0.12, 2.5  # [m] how far beside its stand the vehicle may come to rest; the cost [m of path] of a stand
                                # whose pick the arm does not reach from there
JUMP = 0.25  # [rad] a joint's change between two waypoints that `line_path(smooth=True)` takes for a leap to another branch
LIFTS, LEAN_TOL = (0.30, 0.22, PRE_ABOVE), 8.0  # [m] how high a part is lifted off its table: the first of these at which the arm
                                               # holds the tool's lean within LEAN_TOL [deg] (the last is where it came in)
REPLAN_M, REPLAN_RAD = 0.006, math.radians(0.4)  # the base's motion that makes a pick's or a drop's way down solved again


def rect_dist(px, py, c, size, yaw):
    """Distance from points to an oriented rectangle (0 inside)."""
    ca, sa = math.cos(yaw), math.sin(yaw)
    dx, dy = px - c[0], py - c[1]
    u, v = dx * ca + dy * sa, -dx * sa + dy * ca
    ox, oy = np.maximum(np.abs(u) - size[0] / 2, 0.0), np.maximum(np.abs(v) - size[1] / 2, 0.0)
    return np.hypot(ox, oy)


def seg_dist(px, py, a, b):
    """Distance from points to the segment a-b."""
    ax, ay = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - ax, float(b[1]) - ay
    ll = dx * dx + dy * dy
    f = np.clip(((px - ax) * dx + (py - ay) * dy) / ll, 0.0, 1.0) if ll > 1e-12 else 0.0
    return np.hypot(px - (ax + f * dx), py - (ay + f * dy))


def rot_between(a, b):
    """The smallest rotation that turns the direction a into the direction b."""
    a = np.asarray(a, dtype=float) / (np.linalg.norm(a) + 1e-12)
    b = np.asarray(b, dtype=float) / (np.linalg.norm(b) + 1e-12)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:  # (opposite: half a turn about any axis square to a)
        p = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0.0, 1.0, 0.0])
        p = p / np.linalg.norm(p)
        return 2.0 * np.outer(p, p) - np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rect_corners(c, size, yaw):
    ca, sa = math.cos(yaw), math.sin(yaw)
    out = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        u, v = sx * size[0] / 2, sy * size[1] / 2
        out.append((c[0] + u * ca - v * sa, c[1] + u * sa + v * ca))
    return np.array(out)


def rects_overlap(a, b, margin=0.0):
    """Do two oriented rectangles (corner arrays) come closer than `margin`? (separating axes)"""
    for poly in (a, b):
        for k in range(4):
            e = poly[(k + 1) % 4] - poly[k]
            n = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-12)
            pa, pb = a @ n, b @ n
            if pb.min() - pa.max() >= margin or pa.min() - pb.max() >= margin:
                return False
    return True


def heading_of(d):
    """The heading (0 facing +y, counter-clockwise) that faces the direction d."""
    return math.atan2(d[1], d[0]) - math.pi / 2


def wrap_a(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Planner:
    """A grid of the bay: the static obstacles (tables, crate, tank, steam line) and the day's (the pan, the burning
    pan's ring, the bucket wherever it is and the stretches the hoist is about to carry it over), inflated; A* and a
    string-pulled path."""

    def __init__(self, tables, crate):
        x0, x1, y0, y1, r = GRID
        self.r, self.x0, self.y0 = r, x0, y0
        self.nx, self.ny = int(round((x1 - x0) / r)) + 1, int(round((y1 - y0) / r)) + 1
        xs, ys = x0 + r * np.arange(self.nx), y0 + r * np.arange(self.ny)
        self.X, self.Y = np.meshgrid(xs, ys, indexing="ij")
        self.rects = [((t["center"][0], t["center"][1]), (t["size"][0], t["size"][1]), math.radians(t["yaw_deg"])) for t in tables]
        lo, hi = np.array(crate[0]), np.array(crate[1])
        self.crate = (((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2), (hi[0] - lo[0], hi[1] - lo[1] + 2 * CRATE_RIBS), 0.0)
        self.rects += [self.crate, TANK_RECT] + LINE_RECTS
        occ = np.zeros((self.nx, self.ny), dtype=bool)
        soft = np.zeros((self.nx, self.ny), dtype=bool)
        for c, size, yaw in self.rects:
            d = rect_dist(self.X, self.Y, c, size, yaw)
            occ |= d < INFLATE
            soft |= d < INFLATE_SOFT
        self.static, self.static_soft = occ, soft

    def clearance(self, p, pan=None):
        """How far the point is from the nearest obstacle (the tables, the crate, the tank, the steam line, the pan)."""
        d = min(float(rect_dist(np.array([p[0]]), np.array([p[1]]), c, size, yaw)[0]) for c, size, yaw in self.rects)
        if pan is not None:
            d = min(d, float(rect_dist(np.array([p[0]]), np.array([p[1]]), pan, PAN_SIZE, 0.0)[0]))
        return d

    def occupancy(self, pan, fire, bucket, keep=(), discs=()):
        """`keep`: capsules (x0, y0, x1, y1, room) the vehicle's centre keeps out of (where the bucket is about to pass);
        `discs`: circles (x, y, r) it keeps out of (the flying robots that fly low)."""
        occ = self.static | (rect_dist(self.X, self.Y, pan, PAN_SIZE, 0.0) < INFLATE)
        if fire:
            occ |= np.hypot(self.X - pan[0], self.Y - pan[1]) < RING
        if bucket is not None:
            occ |= np.hypot(self.X - bucket[0], self.Y - bucket[1]) < BUCKET_ROOM
        for x0, y0, x1, y1, room in keep or ():
            occ |= seg_dist(self.X, self.Y, (x0, y0), (x1, y1)) < room
        for x, y, r in discs or ():
            occ |= np.hypot(self.X - x, self.Y - y) < r
        return occ

    def cell(self, p):
        return (int(round((p[0] - self.x0) / self.r)), int(round((p[1] - self.y0) / self.r)))

    def inside(self, c):
        return 0 <= c[0] < self.nx and 0 <= c[1] < self.ny

    def free(self, occ, p):
        c = self.cell(p)
        return self.inside(c) and not occ[c]

    def line_free(self, occ, a, b):
        n = max(2, int(math.dist(a, b) / (self.r / 2)) + 1)
        for k in range(n + 1):
            f = k / n
            if not self.free(occ, (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))):
                return False
        return True

    def nearest_free(self, occ, p, rmax=0.6):
        c0 = self.cell(p)
        best = None
        k = int(rmax / self.r)
        for i in range(c0[0] - k, c0[0] + k + 1):
            for j in range(c0[1] - k, c0[1] + k + 1):
                if self.inside((i, j)) and not occ[i, j]:
                    d = (i - c0[0]) ** 2 + (j - c0[1]) ** 2
                    if best is None or d < best[0]:
                        best = (d, (i, j))
        return None if best is None else best[1]

    def plan(self, occ, a, b, pan=None):
        """A path from a to b (both points), as a list of points, or None. With the pan given, the path keeps
        INFLATE_SOFT from the bay's furniture and the pan where it can (the cells `occ` holds taken all the same)."""
        if pan is not None:
            soft = occ | self.static_soft | (rect_dist(self.X, self.Y, pan, PAN_SIZE, 0.0) < INFLATE_SOFT)
            path = self.plan(soft, a, b)
            if path is not None:
                return path
        import heapq
        sa = self.cell(a) if self.free(occ, a) else self.nearest_free(occ, a)
        sb = self.cell(b) if self.free(occ, b) else None
        if sa is None or sb is None:
            return None
        h = lambda c: math.hypot(c[0] - sb[0], c[1] - sb[1])  # noqa: E731
        openq, g, prev = [(h(sa), 0.0, sa)], {sa: 0.0}, {sa: None}
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        while openq:
            _, gc, c = heapq.heappop(openq)
            if c == sb:
                break
            if gc > g.get(c, 1e18) + 1e-9:
                continue
            for dx, dy, w in steps:
                nb = (c[0] + dx, c[1] + dy)
                if not self.inside(nb) or occ[nb]:
                    continue
                ng = gc + w
                if ng < g.get(nb, 1e18):
                    g[nb], prev[nb] = ng, c
                    heapq.heappush(openq, (ng + h(nb), ng, nb))
        if sb not in prev:
            return None
        cells = []
        c = sb
        while c is not None:
            cells.append(c)
            c = prev[c]
        pts = [(self.x0 + i * self.r, self.y0 + j * self.r) for i, j in reversed(cells)]
        pts[0], pts[-1] = tuple(a[:2]), tuple(b[:2])
        out = [pts[0]]  # (string pulling)
        k = 0
        while k < len(pts) - 1:
            j = len(pts) - 1
            while j > k + 1 and not self.line_free(occ, pts[k], pts[j]):
                j -= 1
            out.append(pts[j])
            k = j
        return [list(p) for p in out]

    def length(self, path):
        return sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1)) if path else 1e9


def held_part(pv):
    """The part the suction tool holds, by its wire name, from the true state (privileged.parts.*.held), or None: the
    observation's own `ugv.vacuum` says only that it holds one."""
    return next((n for n, q in pv["parts"].items() if q["held"]), None)


class Privileged:
    """The reference's program on the true state (obs['privileged']). `s` is a plain dict (the controller deep-copies
    it for repeated queries); every step is a dict in s['steps'] that keeps its own progress. The held part's name
    comes from the true state (`held_part`), at the same instant the observation's `ugv.vacuum` reports."""

    def __init__(self, kin, mis, wheels):
        self.kin, self.mis, self.wheels = kin, mis, wheels
        self.limits = (mis.get("ugv") or {}).get("limits") or {}
        self.dt = mis["dt"]
        self.planner = None
        self.hoist = {}  # (what the hoist's controller says this step: its phase, where the bucket is about to pass, settled)

    def init_state(self):
        return dict(phase="start", steps=[], q_tgt=None, grip=False, home=None, lever_done_t=None, done=False,
                    tries={}, gave_up=[], t=0.0, ik_seed=None, log_t=-1.0)

    # -- the true state -------------------------------------------------------------------------------------------
    def setup_world(self, pv):
        if self.planner is None:
            self.planner = Planner(pv["bay"]["tables"], pv["bay"]["crate"])

    @staticmethod
    def pose(pv):
        u = pv["ugv"]
        return np.array(u["pos"][:2]), math.radians(u["heading_deg"]), np.array(u["pos"]), quat_mat(u["quat"])

    def to_root(self, pv, p_world):
        _, _, p, R = self.pose(pv)
        return R.T @ (np.asarray(p_world, dtype=float) - p)

    def dir_root(self, pv, d_world):
        return self.pose(pv)[3].T @ np.asarray(d_world, dtype=float)

    @staticmethod
    def low_uavs(pv):
        """The flying robots in the air low enough to meet the vehicle (over the set-down bucket, a pouring robot):
        their centres (x, y). (On their pads they sit far south of every route.)"""
        out = []
        for u in (pv.get("uav") or {}).values():
            p = u.get("pos") or [0.0, 0.0, 0.0]
            if 0.75 < float(p[2]) < UAV_LOW:
                out.append((float(p[0]), float(p[1])))
        return out

    def occ(self, pv):
        """The grid's cells taken now: the bucket kept clear wherever it is (hanging, carried or set down), the
        stretches the hoist is about to carry it over, and the flying robots that fly low."""
        return self.planner.occupancy(pv["pan"], pv["fire"], pv["bucket"]["pos"][:2], self.hoist.get("keep") or (),
                                      [(x, y, UAV_ROOM + INFLATE) for x, y in self.low_uavs(pv)])

    def joints(self, obs):
        return {n: obs["ugv"]["joints"][n][0] for n in self.kin.joints}

    # -- arm paths -----------------------------------------------------------------------------------------------
    def reach(self, s, pv, obs, p, wrist):
        """The arm's best configuration for the grip point at p (world) with the tool's axis `wrist`, from several
        starting guesses (the current targets, the start, the arm straight up): (q, grip residual, wrist residual)."""
        pr, wr = self.to_root(pv, p), self.dir_root(pv, wrist)
        seeds = [dict(s["q_tgt"] or self.joints(obs)), dict(s["home"]), {n: 0.0 for n in self.kin.joints}]
        best = None
        for q0 in seeds:
            q, res, wres = self.kin.solve(pr, q0, self.limits, wrist_dir=wr, posture=s["home"], iters=200)
            if best is None or res + 0.5 * wres < best[1] + 0.5 * best[2]:
                best = (q, res, wres)
        return best

    def joint_line(self, q0, q1, n=8):
        """Joint-space waypoints from q0 to q1."""
        return [{k: q0[k] + (f / n) * (q1[k] - q0[k]) for k in self.kin.joints} for f in range(1, n + 1)]

    def line_path(self, s, pv, obs, p1, wrist1, p0=None, wrist0=None, step=0.03, root=None, smooth=False):
        """Joint waypoints that carry the grip point along a straight line (world) to p1, the tool's axis turning from
        wrist0 to wrist1 (world unit vectors, the wrist's side of the grip point). `root` = (p, R) of a planned pose.
        `smooth`: a waypoint whose solution leaps from the last one (the posture's pull can swing the arm over to
        another branch of its solutions, and on the way the tool turns and tips what it holds) is solved again without
        that pull: the least joint motion."""
        if root is None:
            to_r = lambda x: self.to_root(pv, x)  # noqa: E731
            dir_r = lambda d: self.dir_root(pv, d)  # noqa: E731
        else:
            rp, rR = root
            to_r = lambda x: rR.T @ (np.asarray(x, dtype=float) - rp)  # noqa: E731
            dir_r = lambda d: rR.T @ np.asarray(d, dtype=float)  # noqa: E731
        q = dict(s["q_tgt"] or self.joints(obs))
        if p0 is None:
            tool_r, wr = self.kin.points(q)
            p0r = tool_r
            w0r = (wr - tool_r) / (np.linalg.norm(wr - tool_r) + 1e-9)
        else:
            p0r, w0r = to_r(p0), dir_r(wrist0 if wrist0 is not None else wrist1)
        p1r, w1r = to_r(p1), dir_r(wrist1)
        n = max(1, int(math.ceil(float(np.linalg.norm(p1r - p0r)) / step)))
        out, worst = [], 0.0
        for k in range(1, n + 1):
            f = k / n
            w = (1 - f) * w0r + f * w1r
            w = w / (np.linalg.norm(w) + 1e-9)
            q1, res, wres = self.kin.solve(p0r + f * (p1r - p0r), q, self.limits, wrist_dir=w, posture=s["home"], iters=60)
            if smooth and max(abs(q1[n] - q[n]) for n in self.kin.joints[:-1]) > JUMP:
                q2, res2, wres2 = self.kin.solve(p0r + f * (p1r - p0r), q, self.limits, wrist_dir=w, posture=None, iters=60)
                if (max(abs(q2[n] - q[n]) for n in self.kin.joints[:-1]) < max(abs(q1[n] - q[n]) for n in self.kin.joints[:-1])
                        and res2 < max(res, 0.004) + 0.002 and wres2 < max(wres, 0.006) + 0.004):
                    q1, res, wres = q2, res2, wres2
            q = q1
            worst = max(worst, res)
            out.append(dict(q))
        return out, worst

    def joint_path(self, s, obs, q_goal):
        return [dict(q_goal)]

    # -- the program ---------------------------------------------------------------------------------------------
    def act(self, s, obs, hoist):
        pv = obs["privileged"]
        self.setup_world(pv)
        self.hoist = hoist if isinstance(hoist, dict) else {}
        s["t"] = obs["time"]
        if s["home"] is None:
            s["home"] = self.joints(obs)
            s["q_tgt"] = dict(s["home"])
        if pv["ugv"].get("lost"):
            return dict(vel=self.wheels(0.0, 0.0), pos=dict(s["q_tgt"]), grip=False)
        if not s["steps"]:
            self.plan_next(s, pv, obs, hoist)
        if os.environ.get("ARB_REFERENCE_LOG") and obs["time"] - s["log_t"] >= 2.0:
            s["log_t"] = obs["time"]
            xy, psi, _, _ = self.pose(pv)
            st0 = s["steps"][0] if s["steps"] else {}
            log(f"status t={obs['time']:.1f} {st0.get('kind')} {st0.get('why', '')} phase={st0.get('phase')} at ({xy[0]:.2f}, {xy[1]:.2f}) "
                f"heading {math.degrees(psi):.1f} lever {pv['lever']['deg']:.1f} holding {held_part(pv)} grip {s['grip']}")
        wheels = self.wheels(0.0, 0.0)
        for _ in range(4):  # (a step that is done at once lets the next one act in the same control step)
            if not s["steps"]:
                break
            st = s["steps"][0]
            st.setdefault("t0", obs["time"])
            wheels, done = self.run_step(st, s, pv, obs, hoist)
            if not done and obs["time"] - st["t0"] > st.get("timeout", 30.0):
                log(f"t={obs['time']:.1f} step {st['kind']} timed out ({st.get('why', '')})")
                done = True
                if st.get("critical"):
                    s["steps"] = []
                    s.setdefault("aborts", 0)
                    s["aborts"] += 1
                    break
            if not done:
                break
            log(f"t={obs['time']:.1f} {st['kind']} {st.get('why', '')} done in {obs['time'] - st['t0']:.1f} s")
            s["steps"].pop(0)
            wheels = self.wheels(0.0, 0.0)
        return dict(vel=self.slew(s, wheels), pos=dict(s["q_tgt"]), grip=bool(s["grip"]))

    @staticmethod
    def slew(s, wheels):
        """The wheels' speed targets, changed by at most WHEEL_SLEW per control step. The base's wheel drives hold
        their targets almost rigidly (damping 1e7): a target that jumps (a spin straightened out, a stop) jerks the
        chassis on its wheels, and the arm's aim with it."""
        last = s.get("wheel_cmd") or {}
        out = {}
        for n, v in wheels.items():
            v0 = float(last.get(n, 0.0))
            out[n] = v0 + float(np.clip(float(v) - v0, -WHEEL_SLEW, WHEEL_SLEW))
        s["wheel_cmd"] = out
        return dict(out)

    def plan_next(self, s, pv, obs, hoist):
        if s["phase"] == "start":
            s["phase"] = "valve"
            s["steps"] = self.valve_steps(s, pv, obs)
            return
        if s["phase"] == "valve":
            if pv["lever"]["deg"] < 85.0 and s["tries"].get("valve", 0) < 2:  # (pushed but not shut: again)
                s["tries"]["valve"] = s["tries"].get("valve", 0) + 1
                s["steps"] = self.valve_steps(s, pv, obs, depth=0.02 * s["tries"]["valve"])
                return
            s["phase"] = "parts"
        if s["phase"] == "parts":
            steps = self.part_steps(s, pv, obs)
            if steps is None:  # (nothing reachable now: the burning pan's ring, the bucket; again in a moment)
                s["steps"] = [dict(kind="wait", seconds=0.5)]
                return
            if steps:
                s["steps"] = steps
                return
            s["phase"] = "park"  # (where it stands, the arm tucked: any drive now only delays the end)
            s["steps"] = [dict(kind="arm_line", to="tuck", speed=ARM_FAST, timeout=15.0), dict(kind="stop", timeout=1e9)]
            return
        if s["phase"] == "park":
            s["phase"] = "done"
            s["done"] = True
        s["steps"] = [dict(kind="stop", timeout=1e9)]

    # -- the valve ------------------------------------------------------------------------------------------------
    def valve_steps(self, s, pv, obs, depth=0.0):
        # (on the white square, facing the lever; turned out in front of it, far enough back for the straight stretch in
        # to take up what the turn there left beside the line: the push wants the base within a few centimetres)
        stand, approach = VALVE_AT, VALVE_APPROACH
        return [dict(kind="arm_line", to="tuck", speed=ARM_FAST, timeout=12.0, why="tuck"),
                dict(kind="drive", goal=approach, heading=0.0, timeout=40.0, why="valve approach"),
                dict(kind="straight", goal=stand, heading=0.0, timeout=15.0, why="valve stand"),
                dict(kind="lever", depth=depth, timeout=60.0, why="lever"),
                dict(kind="arm_line", to="tuck", speed=ARM_FAST, timeout=12.0, why="tuck after the lever")]

    def lever_at(self, pv, a, depth, out=0.0):
        """A point of the push's arc: `a` round the lever's true hub from straight up (toward the closed stop), in the
        lever's plane (`depth` behind it), `out` in front of it."""
        hub = np.array(pv["lever"]["hub"])
        return hub + LEVER_R * np.array([math.sin(a), 0.0, math.cos(a)]) + np.array([0.0, depth - out, 0.0])

    def run_lever(self, st, s, pv, obs):
        """In front of the lever, into its plane behind the lever (20 degrees back from where it stands), round the
        true hub toward the closed stop; as soon as the lever reads shut, back along the arc and out of its plane (the
        arm never presses a lever that is at its stop)."""
        into = np.array([0.0, -1.0, 0.0])  # (the wrist on the vehicle's side: the tool points into the lever's plane)
        depth = st["depth"]
        if "path" not in st:
            log(f"t={obs['time']:.1f} lever: the base stands {float(self.pose(pv)[0][0] - LEVER_STEM[0]):+.3f} m east of the stem")
            a0 = math.radians(pv["lever"]["deg"] - 20.0)
            pts = [self.lever_at(pv, a0, depth, 0.25), self.lever_at(pv, a0, depth, 0.10), self.lever_at(pv, a0, depth)]
            a = a0
            while a < math.radians(95.0):
                a = min(math.radians(95.0), a + LEVER_STEP)
                pts.append(self.lever_at(pv, a, depth))
            q1, worst, _ = self.reach(s, pv, obs, pts[0], into)  # (in front of the lever: the best of several solutions,
            seg = self.joint_line(dict(s["q_tgt"]), q1)          # reached in joint space, clear of the valve)
            for k in range(1, len(pts)):
                more, w2 = self.line_path(dict(s, q_tgt=seg[-1]), pv, obs, pts[k], into, p0=pts[k - 1], wrist0=into, step=0.02)
                seg = seg + more
                worst = max(worst, w2)
            st.update(path=seg, k=0, held=0.0, phase="push", base0=[float(v) for v in self.pose(pv)[0]])
            log(f"t={obs['time']:.1f} lever from {pv['lever']['deg']:.1f} deg: {len(seg)} waypoints, IK residual {worst:.3f} m")
        if st["phase"] == "push":
            done = self.follow(st, s, obs, ARM_SLOW, tol=0.04)
            st["held"] = st["held"] + self.dt if pv["lever"]["deg"] >= 88.0 else 0.0
            if st["held"] >= 0.3 or done:
                if pv["lever"]["deg"] >= 85.0 and s["lever_done_t"] is None:
                    s["lever_done_t"] = obs["time"]
                    log(f"t={obs['time']:.1f} the lever is shut ({pv['lever']['deg']:.1f} deg)")
                tool = self.tool_world(s, pv, obs)
                hub = np.array(pv["lever"]["hub"])
                a_tool = math.atan2(tool[0] - hub[0], tool[2] - hub[2])
                back = self.lever_at(pv, a_tool - math.radians(15.0), depth)
                b1, w1 = self.line_path(s, pv, obs, back, into, step=0.02)
                b2, w2 = self.line_path(dict(s, q_tgt=b1[-1]), pv, obs, self.lever_at(pv, a_tool - math.radians(15.0), depth, 0.25), into,
                                        p0=back, wrist0=into, step=0.03)
                st.update(path=b1 + b2, k=0, phase="retreat")
                log(f"t={obs['time']:.1f} lever: the base moved {math.dist(self.pose(pv)[0], st['base0']):.3f} m during the push")
            return False
        done = self.follow(st, s, obs, ARM_FAST, tol=0.05)
        if done:
            log(f"t={obs['time']:.1f} lever: the base moved {math.dist(self.pose(pv)[0], st['base0']):.3f} m by the end of the retreat")
        return done

    # -- the parts -----------------------------------------------------------------------------------------------
    def pick_point(self, part):
        return np.array([part["center"][0], part["center"][1], part["hi"][2]])

    def clean(self, pv, name):
        """No other part's box within reach of the suction on the way down to this part's top."""
        p = self.pick_point(pv["parts"][name])
        for m, q in pv["parts"].items():
            if m == name or q["held"]:
                continue
            lo, hi = np.array(q["lo"]), np.array(q["hi"])
            for z in (p[2] + 0.015, p[2] + 0.04, p[2] + 0.07):
                x = np.array([p[0], p[1], z])
                if float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - x, x - hi)))) < 0.05:
                    return False
        return True

    def where(self, pv, name):
        part = pv["parts"][name]
        c = part["center"]
        lo, hi = pv["bay"]["crate"]
        if lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1] and c[2] <= hi[2] + 0.10:
            return "crate"
        for t in pv["bay"]["tables"]:
            if rect_dist(np.array([c[0]]), np.array([c[1]]), t["center"], t["size"], math.radians(t["yaw_deg"]))[0] < 0.01 and c[2] > t["height"]:
                return t["name"]
        return "elsewhere"

    def stands(self, rect, target_xy):
        """Stands outside each edge of an oriented rectangle, at the target's projection (kept 0.15 m inside the
        edge's ends), facing the edge: (stand, approach, heading)."""
        c, size, yaw = rect
        ax, ay = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
        d = np.asarray(target_xy, dtype=float) - np.asarray(c, dtype=float)
        out = []
        for n, t, hn, ht in ((ay, ax, size[1] / 2, size[0] / 2), (-ay, ax, size[1] / 2, size[0] / 2),
                             (ax, ay, size[0] / 2, size[1] / 2), (-ax, ay, size[0] / 2, size[1] / 2)):
            u = float(np.clip(d @ t, -(ht - 0.15), ht - 0.15)) if ht > 0.15 else 0.0
            base = np.asarray(c) + n * hn + t * u
            stand = base + n * STAND
            out.append((stand, stand + n * APPROACH, heading_of(-n), float((d @ n) - hn) * -1.0))
        return out

    def stand_ok(self, pv, occ, stand, approach, heading):
        """Whether the vehicle may work from this stand now: the approach it turns at (the given one, else one further
        out along the stand's line: the last stretch is driven straight in) or None. Outside the burning pan's ring,
        clear of the tables, the crate and the rest, 0.70 m from the pan, with room to turn at the approach (a pan
        moved up to the table's corner leaves no room there, and some further out)."""
        x, y = stand
        pan = pv["pan"]
        if pv["fire"] and math.hypot(x - pan[0], y - pan[1]) < RING:
            return None
        foot = rect_corners(stand, (FOOT[1] + 0.04, FOOT[0] + 0.04), heading)  # (its width across, its length along the heading)
        for c, size, yaw in self.planner.rects:
            if rects_overlap(foot, rect_corners(c, size, yaw)):
                return None
        if float(rect_dist(np.array([x]), np.array([y]), pan, PAN_SIZE, 0.0)[0]) <= 0.40:
            return None
        if any(math.hypot(x - ux, y - uy) < UAV_ROOM + 0.5 * math.hypot(FOOT[0] + 0.04, FOOT[1] + 0.04) for ux, uy in self.low_uavs(pv)):
            return None
        out = (np.asarray(approach, dtype=float) - np.asarray(stand, dtype=float)) / APPROACH
        for extra in APPROACH_MORE:
            a = np.asarray(approach, dtype=float) + extra * out
            if pv["fire"] and math.hypot(a[0] - pan[0], a[1] - pan[1]) < RING:
                continue
            if self.planner.free(occ, a) and self.planner.clearance(a, pan) >= SPIN_ROOM:
                return a  # (the vehicle turns to face the edge there: it needs the room)
        return None

    def stand_root(self, stand, heading):
        """The root's pose (position, rotation) with the vehicle at a stand: level, its forward (+x) along the heading."""
        z = self.mis["ugv"]["start"][2]
        a = heading + math.pi / 2
        R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        return np.array([stand[0], stand[1], z]), R

    @staticmethod
    def wrist_options(stand, target):
        """The tool's axis for a pick (the wrist's side of the grip point): straight up, else leaning toward the
        vehicle (the arm then reaches further)."""
        u = np.array([stand[0] - target[0], stand[1] - target[1], 0.0])
        u = u / (np.linalg.norm(u) + 1e-9)
        out = []
        for lean in (0.0, 25.0, 40.0):
            d = np.array([0.0, 0.0, 1.0]) * math.cos(math.radians(lean)) + u * math.sin(math.radians(lean))
            out.append(d / np.linalg.norm(d))
        return out

    def reach_ok(self, s, pv, obs, stand, heading, pts, wrists):
        """The first tool axis with which the arm reaches every point from the stand (None: none does)."""
        rp, rR = self.stand_root(stand, heading)
        for wrist in wrists:
            q, ok = dict(s["home"]), True
            for p in pts:
                q, res, wres = self.kin.solve(rR.T @ (np.asarray(p) - rp), q, self.limits, wrist_dir=rR.T @ np.asarray(wrist), posture=s["home"], iters=150)
                if res > 0.008 or wres > 0.02:
                    ok = False
                    break
            if ok:
                return [float(x) for x in wrist]
        return None

    @staticmethod
    def stand_bad(s, obs, name, stand):
        """A stand from which the arm missed this part: left alone for a while (the next trip there may end closer),
        and for good once it has failed three times (a part whose only stand it is would otherwise be given up)."""
        key = [name] + [round(float(v), 2) for v in stand]
        marks = [b[-1] for b in s.get("bad_stands", []) if b[:3] == key]
        return len(marks) >= 3 or any(obs["time"] - t < BAD_STAND_S for t in marks)

    def part_steps(self, s, pv, obs):
        """The next part: Table A's first, then Table B's; the cheapest reachable stand outside the burning pan's
        ring and clear of the bucket. None: parts left, none reachable now; []: none left. What no stand reaches is
        given up only once the fire is out and the hoist has settled (until then the ring or the bucket may be what
        is in the way)."""
        left = [n for n in pv["parts"] if self.where(pv, n) not in ("crate", "elsewhere") and n not in s["gave_up"]]
        if not left:
            return []
        occ = self.occ(pv)
        xy = self.pose(pv)[0]
        tables = {t["name"]: t for t in pv["bay"]["tables"]}
        best = None
        for name in sorted(left, key=lambda n: (self.where(pv, n) != "Table_A", not self.clean(pv, n))):
            tname = self.where(pv, name)
            t = tables[tname]
            rect = (t["center"], t["size"], math.radians(t["yaw_deg"]))
            top = self.pick_point(pv["parts"][name])
            for stand, appr, hd, depth in self.stands(rect, top[:2]):
                appr = self.stand_ok(pv, occ, stand, appr, hd)
                if appr is None or self.stand_bad(s, obs, name, stand):
                    continue
                path = self.planner.plan(occ, xy, appr, pv["pan"])
                if path is None:
                    continue
                cost = self.planner.length(path) + 2.0 * max(0.0, depth) + (0.0 if self.clean(pv, name) else 5.0) + (0.0 if tname == "Table_A" else 3.0)
                if best is not None and cost >= best[0]:
                    continue
                pts = [top + [0, 0, PRE_ABOVE], top + [0, 0, 0.10], top + [0, 0, TOUCH]]
                wrist = self.reach_ok(s, pv, obs, stand, hd, pts, self.wrist_options(stand, top))
                if wrist is None:
                    continue
                # (the vehicle comes to rest up to a dozen centimetres beside its stand: a pick that the arm reaches from
                # the stand itself only is a pick it may miss, and then the trip there is lost)
                side = np.array([math.cos(hd), math.sin(hd)])
                if any(self.reach_ok(s, pv, obs, stand + k * STAND_SLIP * side, hd, pts, [np.array(wrist)] + self.wrist_options(stand, top)) is None
                       for k in (1.0, -1.0)):
                    cost += FRAGILE
                    if best is not None and cost >= best[0]:
                        continue
                best = (cost, name, stand, appr, hd, wrist)
            if best is not None and self.where(pv, best[1]) == "Table_A" and tname != "Table_A":
                break
        if best is None:
            cooling = any(obs["time"] - b[-1] < BAD_STAND_S for b in s.get("bad_stands", []))  # (a missed stand may be tried again soon)
            s["unreached"] = s.get("unreached", 0) + (0 if pv["fire"] or not self.hoist.get("settled", True) or cooling else 1)
            if s["unreached"] >= 6:  # (no ring and no bucket to wait for, for 3 s: what no stand reaches is given up)
                for n in left:
                    s["gave_up"].append(n)
                    log(f"t={obs['time']:.1f} {n}: no stand reaches it; given up")
                return self.part_steps(s, pv, obs)
            return None
        s["unreached"] = 0
        _, name, stand, appr, hd, wrist = best
        log(f"t={obs['time']:.1f} next part {name} on {self.where(pv, name)} from ({stand[0]:.2f}, {stand[1]:.2f}) heading {math.degrees(hd):.0f}, "
            f"tool leaning {math.degrees(math.acos(max(-1.0, min(1.0, wrist[2])))):.0f} deg")
        return [dict(kind="arm_line", to="tuck", speed=ARM_FAST, timeout=12.0, why="tuck"),
                dict(kind="drive", goal=list(appr), heading=hd, timeout=60.0, why=f"to {name}"),
                dict(kind="straight", goal=list(stand), heading=hd, timeout=15.0, why="in"),
                dict(kind="pick", part=name, wrist=wrist, stand=list(stand), timeout=40.0, why=f"pick {name}"),
                dict(kind="straight", goal=list(appr), heading=hd, timeout=15.0, why="out"),
                dict(kind="crate", part=name, timeout=240.0, why=f"crate {name}")]

    def run_pick(self, st, s, pv, obs):
        name = st["part"]
        part = pv["parts"][name]
        down = np.array(st.get("wrist") or [0.0, 0.0, 1.0])
        if "phase" not in st:
            top = self.pick_point(part)
            xy, psi, _, _ = self.pose(pv)
            wrist = self.reach_ok(s, pv, obs, xy, psi, [top + [0, 0, PRE_ABOVE], top + [0, 0, 0.10], top + [0, 0, TOUCH]],
                                  [np.array(st.get("wrist") or [0.0, 0.0, 1.0])] + self.wrist_options(xy, top))  # (from where it stands)
            if wrist is None:  # (out of the arm's reach from here)
                off = math.dist(xy, st["stand"])
                log(f"t={obs['time']:.1f} {name}: out of reach from ({xy[0]:.2f}, {xy[1]:.2f}), {off:.2f} m off the stand")
                again = {k: v for k, v in st.items() if k != "t0"}
                if off > PICK_REDO and not st.get("redone"):
                    # (far off: the turn at the approach walked it there; out, to the approach and in again, from close by)
                    again["redone"] = True
                    appr = list(s["steps"][1]["goal"]) if len(s["steps"]) > 1 and s["steps"][1]["kind"] == "straight" else None
                    if appr is not None:
                        hd = float(s["steps"][1]["heading"])
                        s["steps"] = [st, dict(kind="straight", goal=appr, heading=hd, timeout=15.0, why="out to come in again"),
                                      dict(kind="drive", goal=appr, heading=hd, timeout=40.0, why=f"to {name} again"),
                                      dict(kind="straight", goal=list(st["stand"]), heading=hd, timeout=15.0, why="in again"), again] + s["steps"][1:]
                        return True
                if not st.get("crept"):
                    # (a little short of it: a few centimetres closer to the table, along the heading, and try again)
                    again["crept"] = True
                    s["steps"] = [st, dict(kind="straight", goal=(xy + CREEP * np.array([-math.sin(psi), math.cos(psi)])).tolist(), heading=psi,
                                           timeout=8.0, why="closer"), again] + s["steps"][1:]
                    return True
                # (this stand is no good for this part)
                s.setdefault("bad_stands", []).append([name] + [round(float(v), 2) for v in st["stand"]] + [float(obs["time"])])
                s["steps"] = s["steps"][:1] + [x for x in s["steps"][1:] if x["kind"] != "crate"]
                return True
            st["wrist"] = wrist
            down = np.array(wrist)
            a, w1 = self.line_path(s, pv, obs, top + [0, 0, PRE_ABOVE], down, step=0.04)
            b, w2 = self.line_path(dict(s, q_tgt=a[-1]), pv, obs, top + [0, 0, 0.10], down, p0=top + [0, 0, PRE_ABOVE], step=0.03)
            c, w3 = self.line_path(dict(s, q_tgt=b[-1]), pv, obs, top + [0, 0, TOUCH], down, p0=top + [0, 0, 0.10], step=0.015)
            st.update(phase="down", path=a + b + c, k=0, n_grip=len(a + b), n_fast=len(a), top=top.tolist(), plan_pose=self.pose_key(pv))
            if max(w1, w2, w3) > 0.015:
                log(f"t={obs['time']:.1f} {name}: IK residual {max(w1, w2, w3):.3f} m on the way down")
        if st["phase"] == "down":
            self.track_pick(st, s, pv, obs)
            if st["k"] >= st["n_grip"] - 2:
                s["grip"] = True
            done = self.follow(st, s, obs, ARM_FAST if st["k"] < st.get("n_fast", 0) else ARM_SLOW, tol=0.006)
            if st["k"] >= len(st["path"]) - 1:
                st.setdefault("t_last", obs["time"])
                done = done or obs["time"] - st["t_last"] > 1.5
            if held_part(pv) is not None or done:
                st["phase"] = "held" if held_part(pv) is not None else "missed"
                st["t_held"] = obs["time"]
                if st["phase"] == "held":  # (the grip link's frame as it took the part, and the part's own axis that stands up)
                    st["G_hold"] = self.grip_frame(pv, self.joints(obs))[1].tolist()
                    pR = quat_mat(part["quat"]) if part.get("quat") is not None else np.eye(3)
                    k = int(np.argmax(np.abs(pR[2, :])))
                    s.setdefault("rest", {})[name] = [k, 1.0 if pR[2, k] > 0 else -1.0]
                tool = self.tool_world(s, pv, obs)
                log(f"t={obs['time']:.1f} {name}: {st['phase']} ({held_part(pv)}) {obs['time'] - st['t0']:.1f} s into the pick; "
                    f"tool at {np.round(tool, 3).tolist()}, part top {np.round(st['top'], 3).tolist()}, box {part['lo']} {part['hi']}")
            return False
        if st["phase"] == "missed":
            s["grip"] = False
            log(f"t={obs['time']:.1f} {name}: nothing held after the touch")
            s["tries"][name] = s["tries"].get(name, 0) + 1
            if s["tries"][name] >= 2:
                s["gave_up"].append(name)
            st["phase"] = "up"
            top = np.array(st["top"])
            st["path"], w = self.line_path(s, pv, obs, top + [0, 0, 0.30], down, step=0.03)
            st["k"] = 0
            st["fail"] = True
            st.pop("t_last", None)
        if st["phase"] == "held":
            # (straight up, as high as the arm holds the tool's lean: the part keeps the pose it was taken in, turned
            # neither about the tool's axis nor, beyond the arm's reach, with the tool)
            top = np.array(st["top"])
            st["sync"] = True
            for lift in LIFTS:
                up, w = self.line_path(s, pv, obs, top + [0, 0, lift], down, step=0.03, smooth=True)
                dev = self.lean_off(pv, up, down)
                if dev <= LEAN_TOL:
                    break
            if lift < LIFTS[0] or dev > LEAN_TOL:
                log(f"t={obs['time']:.1f} {name}: lifted {lift:.2f} m (the tool's lean held within {dev:.0f} deg)")
            st.update(phase="up", path=self.hold_turn(pv, up, st.get("G_hold")), k=0)
        if st["phase"] == "up":
            if self.follow(st, s, obs, ARM_SLOW, tol=0.04):
                if st.get("fail"):
                    s["steps"] = s["steps"][:1] + [x for x in s["steps"][1:] if x["kind"] != "crate"]
                st["phase"] = "carry"
                path, w = self.line_path(s, pv, obs, self.carry_point(pv), down, step=0.04, smooth=st.get("G_hold") is not None)
                st["path"] = self.hold_turn(pv, path, st.get("G_hold"))
                st["k"] = 0
            return False
        if st["phase"] == "carry":
            return self.follow(st, s, obs, ARM_FAST, tol=0.05)
        return False

    def lean_off(self, pv, path, wrist):
        """The largest angle [deg] between the tool's axis along the joint waypoints and the axis `wrist` (world)."""
        _, _, _, R = self.pose(pv)
        w = np.asarray(wrist, dtype=float) / (np.linalg.norm(wrist) + 1e-12)
        worst = 0.0
        for q in path:
            tool_r, wr = self.kin.points(q)
            a = R @ (wr - tool_r)
            worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, float(a @ w) / (np.linalg.norm(a) + 1e-12))))))
        return worst

    def hold_turn(self, pv, path, G):
        """The waypoints with the last joint turned so the grip link keeps the frame G (world, as it took the part)
        about the tool's axis: the IK leaves that turn free, and with the tool leaning it would tip the part."""
        if G is None:
            return path
        G = np.asarray(G, dtype=float)
        return [self.turn_tool(pv, q, G)[0] for q in path]

    @staticmethod
    def pose_key(pv):
        u = pv["ugv"]
        return [float(u["pos"][0]), float(u["pos"][1]), math.radians(u["heading_deg"])]

    def moved(self, st, pv, obs):
        """Has the vehicle's base moved since the arm's path was planned (st['plan_pose'])? The arm's reaction pushes
        the skid-steer's base and its wheels slip; a path planned from where it stood then misses by as much as it
        moved. (At most every 0.2 s.)"""
        x0, y0, a0 = st["plan_pose"]
        u = pv["ugv"]
        if math.hypot(u["pos"][0] - x0, u["pos"][1] - y0) < REPLAN_M and abs(wrap_a(math.radians(u["heading_deg"]) - a0)) < REPLAN_RAD:
            return False
        return obs["time"] - st.get("t_replan", -9.0) >= 0.2

    def track_pick(self, st, s, pv, obs):
        """The rest of the way down to the part, solved again from where the arm's targets are on the vehicle's true
        pose now, when the base has moved (closed loop on the true pose, not on the pose at the start of the pick)."""
        if not self.moved(st, pv, obs):
            return
        top, down, k = np.array(st["top"]), np.array(st["wrist"]), st["k"]
        targets = [(top + [0, 0, PRE_ABOVE], 0.04), (top + [0, 0, 0.10], 0.03), (top + [0, 0, TOUCH], 0.015)]
        seg = 0 if k < st["n_fast"] else (1 if k < st["n_grip"] else 2)
        path, ends = [], {}
        for i in range(seg, 3):
            p1, step = targets[i]
            if not path:
                pts, _ = self.line_path(s, pv, obs, p1, down, step=step)
            else:
                pts, _ = self.line_path(dict(s, q_tgt=path[-1]), pv, obs, p1, down, p0=targets[i - 1][0], step=step)
            path += pts
            ends[i] = len(path)
        st.update(path=path, k=0, n_fast=ends.get(0, 0), n_grip=ends.get(1, 0), plan_pose=self.pose_key(pv), t_replan=obs["time"])
        st.pop("t_last", None)
        st["replans"] = st.get("replans", 0) + 1

    def track_lower(self, st, s, pv, obs):
        """The crate's way down, solved again on the true pose when the base has moved (as `track_pick`): the rest of
        the way over to the part's spot and down to where it is let go."""
        if not self.moved(st, pv, obs):
            return
        drop = st.get("drop")
        if drop is not None:
            st["path"], st["n_over"] = self.drop_path(s, pv, obs, drop, over=st["k"] < st.get("n_over", 0))
        else:
            st["path"], _ = self.line_path(s, pv, obs, np.array(st["p1"]), np.array([0.0, 0.0, 1.0]), step=0.03)
        st.update(k=0, plan_pose=self.pose_key(pv), t_replan=obs["time"])
        st["replans"] = st.get("replans", 0) + 1

    def carry_point(self, pv, which=CARRY):
        _, _, p, R = self.pose(pv)
        return p + R @ np.asarray(which, dtype=float)

    # -- the drop: planned on the part's box as it will hang over its spot (the grader holds it rigidly in the grip link's
    # frame, so its pose at the drop follows from the arm's)
    def grip_frame(self, pv, q):
        """The grip link's frame in the world for joint positions q on the vehicle's true pose (origin, rotation), and
        the last joint's (signed) axis there: it turns the tool about its own axis, the grip point on it."""
        T, js = self.kin.frames(q)
        _, _, p, R = self.pose(pv)
        return p + R @ T[:3, 3], R @ T[:3, :3], R @ np.asarray(js[-1][1], dtype=float)

    def held_shape(self, pv, obs, name):
        """The held part as the grip link's frame carries it: its box's corners and its orientation in that frame, the
        tool's axis there, and the frame's orientation now (world). None without the part's own box."""
        part = pv["parts"].get(name) or {}
        if part.get("own_lo") is None or part.get("quat") is None or part.get("pos") is None:
            return None
        q = self.joints(obs)
        gp, gR, _ = self.grip_frame(pv, q)
        tool_r, wrist_r = self.kin.points(q)
        a = self.pose(pv)[3] @ (wrist_r - tool_r)
        pR = quat_mat(part["quat"])
        lo, hi = np.array(part["own_lo"], dtype=float), np.array(part["own_hi"], dtype=float)
        cs = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        world = np.array(part["pos"], dtype=float) + cs @ pR.T
        return dict(corners=(world - gp) @ gR, R_rel=gR.T @ pR, axis=gR.T @ (a / (np.linalg.norm(a) + 1e-12)), gR=gR)

    @staticmethod
    def drop_options(shape, rest=None):
        """The ways the held part may go down into the crate, in order: (how, the grip link's orientation in the world,
        the tool's axis, the box's corners from the tool point in world axes). Level first: the part's own axis nearest
        the vertical made vertical, so it lands as it stood or lay on its table (the tool then leans as it did at the
        pick), turned about the vertical so its sides square with the crate's walls (the four quarter turns, the
        smallest first). Then the tool upright and the part as it hangs from it, turned to its smallest footprint."""
        R_rel, c_g, a_g, gR = shape["R_rel"], shape["corners"], shape["axis"], shape["gR"]
        up = np.array([0.0, 0.0, 1.0])
        pR = gR @ R_rel
        if rest is not None:  # (the own axis that stood up on the table, as the pick found it)
            k, sign = int(rest[0]), float(rest[1])
        else:
            k = int(np.argmax(np.abs(pR[2, :])))
            sign = 1.0 if pR[2, k] > 0 else -1.0
        L = rot_between(pR[:, k] * sign, up) @ pR
        j = (k + 1) % 3
        phi = math.atan2(L[1, j], L[0, j])
        level = []
        for m in range(4):
            psi = wrap_a(-phi + m * math.pi / 2)
            G = rot_z(psi) @ L @ R_rel.T
            level.append((abs(psi), "level", G, G @ a_g, c_g @ G.T))
        out = [o[1:] for o in sorted(level, key=lambda o: o[0])]
        U = rot_between(gR @ a_g, up) @ gR
        best = None
        for d in range(-90, 90, 5):
            G = rot_z(math.radians(d)) @ U
            cw = c_g @ G.T
            ext = cw[:, :2].max(axis=0) - cw[:, :2].min(axis=0)
            key = (round(float(ext[0] * ext[1]), 4), abs(d))
            if best is None or key < best[0]:
                best = (key, G, cw)
        out.append(("upright", best[1], up, best[2]))
        return out

    def turn_tool(self, pv, q, G):
        """q with its last joint turned so the grip link's frame comes round to G about the tool's axis (the grip point
        and the tool's axis stay where they are)."""
        _, gR, u = self.grip_frame(pv, q)
        u = u / (np.linalg.norm(u) + 1e-12)

        def flat(v):
            v = np.asarray(v, dtype=float) - (np.asarray(v, dtype=float) @ u) * u
            return v / (np.linalg.norm(v) + 1e-12)
        x0, x1 = flat(gR[:, 0]), flat(np.asarray(G)[:, 0])
        th = math.atan2(float(np.cross(x0, x1) @ u), float(x0 @ x1))
        out = dict(q)
        out[self.kin.joints[-1]] = out[self.kin.joints[-1]] + th
        return out, th

    def drop_path(self, s, pv, obs, drop, over=True):
        """Joint waypoints to the drop: over to the spot at the transit height, the tool turning to the drop's axis
        and the part to its turn on the way (`over`), then straight down to where the part is let go; and how many of
        them are the way over."""
        G, ax = np.array(drop["G"]), np.array(drop["ax"])
        path, start, p0 = [], s, None
        if over:  # (the grip link's frame turned from where it is now to the drop's, evenly along the way)
            a, _ = self.line_path(s, pv, obs, np.array(drop["p_over"]), ax, step=0.03, smooth=True)
            G0 = self.grip_frame(pv, dict(s["q_tgt"] or self.joints(obs)))[1]
            D = G @ G0.T
            ang = math.acos(max(-1.0, min(1.0, (float(np.trace(D)) - 1.0) / 2.0)))
            w = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
            w = w / np.linalg.norm(w) if np.linalg.norm(w) > 1e-9 else np.array([0.0, 0.0, 1.0])
            a = [self.turn_tool(pv, q, axis_rot(w, ang * (i + 1) / len(a))[:3, :3] @ G0)[0] for i, q in enumerate(a)]
            path, start, p0 = a, dict(s, q_tgt=a[-1]), np.array(drop["p_over"])
        b, _ = self.line_path(start, pv, obs, np.array(drop["p_rel"]), ax, p0=p0, wrist0=ax if p0 is not None else None, step=0.02, smooth=True)
        b = [self.turn_tool(pv, q, G)[0] for q in b]
        return path + b, len(path)

    def crate_spots(self, pv, d_lo, d_hi, held, xy):
        """Where the held part may go down into the crate: tool positions (x, y) at which its box (`d_lo`, `d_hi`: its
        corners from the tool point, as it will hang there) lies inside the crate's walls with SLOT_MARGIN to spare,
        on a grid over the room it has. The ones whose box overlaps the parts already in the crate least come first;
        then those whose box rests against the most walls and parts (the crate fills from its corners: a part lying
        on its broad side is broad, and six fit side by side only packed); then the nearest to the vehicle. Each with
        its overlap and its contacts."""
        lo, hi = np.array(pv["bay"]["crate"][0]), np.array(pv["bay"]["crate"][1])
        a, b = lo[:2] + SLOT_MARGIN - d_lo, hi[:2] - SLOT_MARGIN - d_hi
        axes = [np.linspace(a[k], b[k], n) if a[k] < b[k] else np.array([(a[k] + b[k]) / 2]) for k, n in ((0, 13), (1, 9))]
        others = [(np.array(q["lo"][:2]) - 0.01, np.array(q["hi"][:2]) + 0.01) for m, q in pv["parts"].items()
                  if m != held and not q.get("held") and self.where(pv, m) == "crate"]
        out = []
        for x in axes[0]:
            for y in axes[1]:
                plo, phi = np.array([x, y]) + d_lo, np.array([x, y]) + d_hi
                over = 0.0
                # (against a wall: at either end of the room it has along an axis)
                touch = sum(int(a[k] < b[k] and abs((x, y)[k] - a[k]) < 1e-6) + int(a[k] < b[k] and abs((x, y)[k] - b[k]) < 1e-6) for k in range(2))
                for olo, ohi in others:
                    w = np.minimum(phi, ohi) - np.maximum(plo, olo)
                    over += float(max(0.0, w[0]) * max(0.0, w[1]))
                    for k in range(2):  # (against a part: within 3 cm of its box along one axis, beside it along the other)
                        j = 1 - k
                        if w[j] > 0.0 and (0.0 <= olo[k] - phi[k] < 0.03 or 0.0 <= plo[k] - ohi[k] < 0.03):
                            touch += 1
                out.append((round(over, 4), -touch, math.dist((x, y), xy), np.array([x, y])))
        out.sort(key=lambda r: r[:3])
        return [(r[0], -r[1], r[3]) for r in out]

    def floor_under(self, pv, held, plo, phi):
        """The height the held part's bottom must clear over its footprint (plo, phi): the crate's rim, or the parts
        already in the crate under it."""
        z = float(pv["bay"]["crate"][1][2])
        for m, q in pv["parts"].items():
            if m == held or q.get("held") or self.where(pv, m) != "crate":
                continue
            if np.all(np.minimum(phi, np.array(q["hi"][:2]) + 0.01) > np.maximum(plo, np.array(q["lo"][:2]) - 0.01)):
                z = max(z, float(q["hi"][2]))
        return z

    def fk_box(self, pv, q, shape):
        """The held part's box (lo, hi in x, y) with the arm at joint positions q (forward kinematics)."""
        gp, gR, _ = self.grip_frame(pv, q)
        cw = gp + shape["corners"] @ gR.T
        return cw[:, :2].min(axis=0), cw[:, :2].max(axis=0)

    @staticmethod
    def into_crate(pv, blo, bhi, room):
        """The least horizontal move that brings a box (lo, hi in x, y) inside the crate's walls (the check before a
        release wants 0.035 m inside its outer box) with `room` to spare; a box too broad for that is centred."""
        lo, hi = np.array(pv["bay"]["crate"][0][:2]) + 0.035, np.array(pv["bay"]["crate"][1][:2]) - 0.035
        shift = np.zeros(2)
        for k in range(2):
            if bhi[k] - blo[k] > hi[k] - lo[k] - 2 * room:
                shift[k] = (lo[k] + hi[k]) / 2 - (blo[k] + bhi[k]) / 2
            elif blo[k] < lo[k] + room:
                shift[k] = lo[k] + room - blo[k]
            elif bhi[k] > hi[k] - room:
                shift[k] = hi[k] - room - bhi[k]
        return shift

    def plan_drop(self, s, pv, obs, held):
        """The drop: the best spot over the ways the part may go down (the least overlap with the parts already in,
        then the most walls and parts touched, then level before upright), among the spots the arm reaches from where
        the vehicle stands. None without the part's shape or a reachable spot."""
        shape = self.held_shape(pv, obs, held)
        if shape is None:
            return None
        xy, psi, _, _ = self.pose(pv)
        best, tried = None, []
        for rank, (how, G, ax, cw) in enumerate(self.drop_options(shape, (s.get("rest") or {}).get(held))):
            lean = math.degrees(math.acos(max(-1.0, min(1.0, float(ax[2])))))
            if lean > 65.0:  # (the tool would lean too far)
                tried.append(f"{how} leaning {lean:.0f}")
                continue
            d_lo, d_hi = cw[:, :2].min(axis=0), cw[:, :2].max(axis=0)
            below = -float(cw[:, 2].min())
            n = 0
            for over, touch, spot in self.crate_spots(pv, d_lo, d_hi, held, xy)[:4]:
                if best is not None and (over, -touch, rank) >= best[0]:
                    break
                n += 1
                z = self.floor_under(pv, held, spot + d_lo, spot + d_hi)
                p_over = np.array([spot[0], spot[1], z + TRANSIT + below])
                p_rel = np.array([spot[0], spot[1], z + RELEASE + below])
                if self.reach_ok(s, pv, obs, xy, psi, [p_over, p_rel], [ax]) is not None:
                    best = ((over, -touch, rank), dict(how=how, G=G.tolist(), ax=[float(v) for v in ax], p_over=p_over.tolist(),
                                                       p_rel=p_rel.tolist(), box=[(spot + d_lo).tolist(), (spot + d_hi).tolist()]))
                    break
            tried.append(f"{how} leaning {lean:.0f}: {n} spots{' ok' if best is not None and best[0][2] == rank else ''}")
            if best is not None and best[0][0] == 0.0 and best[0][1] <= -2 and how == "level":
                break  # (a level spot that overlaps no part and rests against two walls or parts: none better)
        pR = shape["gR"] @ shape["R_rel"]
        a_now = shape["gR"] @ shape["axis"]
        log(f"t={obs['time']:.1f} {held}: the drop's ways tried (the part tilted {math.degrees(math.acos(min(1.0, float(np.max(np.abs(pR[2, :])))))):.0f} deg, "
            f"the tool leaning {math.degrees(math.acos(max(-1.0, min(1.0, float(a_now[2]))))):.0f} deg): {'; '.join(tried)}")
        return None if best is None else best[1]

    def run_crate(self, st, s, pv, obs):
        """To the crate's nearest open side (none outside the burning pan's ring: the vehicle holds the part and waits,
        the step's clock stopped), the part down where it fits, let go, check it landed inside; back out, except after
        the last part."""
        held = held_part(pv)
        if "phase" not in st:
            if held is None:
                return True
            lo, hi = np.array(pv["bay"]["crate"][0]), np.array(pv["bay"]["crate"][1])
            c = (lo + hi) / 2
            occ = self.occ(pv)
            xy = self.pose(pv)[0]
            best = None
            for stand, appr, hd, _ in self.stands(self.planner.crate, c[:2]):
                appr = self.stand_ok(pv, occ, stand, appr, hd)
                if appr is None:
                    continue
                path = self.planner.plan(occ, xy, appr, pv["pan"])
                if path is None:
                    continue
                cost = self.planner.length(path) + 0.5 * math.dist(stand, c[:2])
                if best is None or cost < best[0]:
                    best = (cost, stand, appr, hd)
            if best is None:
                if not st.get("waits"):
                    log(f"t={obs['time']:.1f} crate: no side open (the burning pan's ring, the bucket): holding {held}, waiting")
                st["waits"] = st.get("waits", 0) + 1
                st["t0"] = obs["time"]  # (the step's time limit counts from when a side opens)
                return False
            _, stand, appr, hd = best
            log(f"t={obs['time']:.1f} crate: from ({stand[0]:.2f}, {stand[1]:.2f}) heading {math.degrees(hd):.0f}")
            st.update(phase="drive", stand=list(stand), appr=list(appr), hd=hd,
                      sub=[dict(kind="drive", goal=list(appr), heading=hd, timeout=60.0), dict(kind="straight", goal=list(stand), heading=hd, timeout=15.0)])
        if st["phase"] == "drive":
            sub = st["sub"][0]
            sub.setdefault("t0", obs["time"])
            wh, done = self.run_step(sub, s, pv, obs, None)
            if done or obs["time"] - sub["t0"] > sub["timeout"]:
                log(f"t={obs['time']:.1f} crate: {sub['kind']} {'done' if done else 'timed out'} in {obs['time'] - sub['t0']:.1f} s")
                st["sub"].pop(0)
                if not st["sub"]:
                    st["phase"] = "place"
            st["_wheels"] = wh
            return False
        if st["phase"] == "place":
            part = pv["parts"].get(held) if held else None
            if part is None:
                st["phase"] = "check"
                st["t_rel"] = obs["time"]
                return False
            off = math.dist(self.pose(pv)[0], st["stand"])
            if off > STAND_OFF and st.get("redrives", 0) < 2:  # (the drive or the straight stretch did not get it there: again)
                st["redrives"] = st.get("redrives", 0) + 1
                log(f"t={obs['time']:.1f} crate: {off:.2f} m off its stand, holding {held}: driving there again")
                st.update(phase="drive", sub=[dict(kind="drive", goal=list(st["appr"]), heading=st["hd"], timeout=60.0),
                                              dict(kind="straight", goal=list(st["stand"]), heading=st["hd"], timeout=15.0)])
                return False
            drop = self.plan_drop(s, pv, obs, held)
            if drop is not None:
                # (the part's box where the planned arm really puts it: the IK holds the tool's axis only so well, and
                # a part hanging from a leaning tool swings out with any error; moved in until it fits)
                shape = self.held_shape(pv, obs, held)
                for _ in range(3):
                    path, n_over = self.drop_path(s, pv, obs, drop)
                    blo, bhi = self.fk_box(pv, path[-1], shape)
                    shift = self.into_crate(pv, blo, bhi, 0.012)
                    if not np.any(shift):
                        break
                    drop["p_over"] = (np.array(drop["p_over"]) + [shift[0], shift[1], 0.0]).tolist()
                    drop["p_rel"] = (np.array(drop["p_rel"]) + [shift[0], shift[1], 0.0]).tolist()
                log(f"t={obs['time']:.1f} {held}: down into the crate {drop['how']} at ({drop['p_rel'][0]:.2f}, {drop['p_rel'][1]:.2f}), "
                    f"its box as planned {np.round(drop['box'][0], 3).tolist()} {np.round(drop['box'][1], 3).tolist()}, "
                    f"as the arm puts it {np.round(blo, 3).tolist()} {np.round(bhi, 3).tolist()}")
                st.update(path=path, n_over=n_over, k=0, phase="lower", part=held, drop=drop, sync=True, plan_pose=self.pose_key(pv))
                return False
            # (no shape of the part, or no spot the arm reaches with it: the tool upright, the part's box as it hangs now)
            tool = self.tool_world(s, pv, obs)
            below = float(tool[2] - part["lo"][2])
            hi = np.array(pv["bay"]["crate"][1])
            xy, psi, _, _ = self.pose(pv)
            spots = self.crate_spots(pv, np.array(part["lo"][:2]) - tool[:2], np.array(part["hi"][:2]) - tool[:2], held, xy)
            p1 = np.array([spots[0][2][0], spots[0][2][1], hi[2] + 0.10 + below])
            for _, _, spot in spots[:8]:
                p = np.array([spot[0], spot[1], hi[2] + 0.10 + below])
                if self.reach_ok(s, pv, obs, xy, psi, [p], [np.array([0.0, 0.0, 1.0])]) is not None:
                    p1 = p
                    break
            log(f"t={obs['time']:.1f} {held}: down into the crate at ({p1[0]:.2f}, {p1[1]:.2f}), its box "
                f"{np.round(np.array(part['hi'][:2]) - np.array(part['lo'][:2]), 3).tolist()} across")
            st["path"], w = self.line_path(s, pv, obs, p1, np.array([0.0, 0.0, 1.0]), step=0.03)
            st.update(k=0, phase="lower", part=held, p1=p1.tolist(), plan_pose=self.pose_key(pv))
            return False
        if st["phase"] == "lower":
            if not st.get("aims"):  # (the aim-again moves below re-plan from where the part is: only the first way down)
                self.track_lower(st, s, pv, obs)
            fast = st.get("drop") is not None and not st.get("aims") and st["k"] < st.get("n_over", 0) - 2
            there = self.follow(st, s, obs, ARM_FAST if fast else ARM_SLOW, tol=0.03)  # (over to the spot high above the crate: fast)
            if there and obs["time"] - st.get("t_aim", -9.0) >= 0.6:  # (an aim-again move: the arm has had the time to make it)
                part = pv["parts"].get(st["part"]) if st.get("part") else None
                lo, hi = np.array(pv["bay"]["crate"][0]), np.array(pv["bay"]["crate"][1])
                inner_lo, inner_hi = lo[:2] + 0.035, hi[:2] - 0.035
                inside = part is None or (np.all(np.array(part["lo"][:2]) >= inner_lo) and np.all(np.array(part["hi"][:2]) <= inner_hi))
                over = part is None or (np.all(np.array(part["center"][:2]) >= lo[:2] + 0.06) and np.all(np.array(part["center"][:2]) <= hi[:2] - 0.06))
                if not inside and st.get("aims", 0) >= 3 and not over and st.get("redrives", 0) < 2:
                    # (out of the arm's reach from where the vehicle stands: never let it go there; the part back up, to the
                    # stand again, the drop planned anew)
                    st["redrives"] = st.get("redrives", 0) + 1
                    log(f"t={obs['time']:.1f} {st['part']}: not over the crate from here; back up and to the stand again")
                    path, w = self.line_path(s, pv, obs, self.carry_point(pv), np.array([0.0, 0.0, 1.0]), step=0.04)
                    st.update(phase="lift", path=path, k=0, aims=0, drop=None)
                    return False
                if not inside and st.get("aims", 0) < 3:  # (the vehicle slid while the arm reached: aim again from here)
                    st["aims"] = st.get("aims", 0) + 1
                    # (the least move that brings the part's box, as it hangs now, inside the walls with a little room)
                    shift = self.into_crate(pv, np.array(part["lo"][:2]), np.array(part["hi"][:2]), 0.008)
                    tool = self.tool_world(s, pv, obs)
                    log(f"t={obs['time']:.1f} {st['part']} not over the crate yet: aiming again ({shift[0]:+.3f}, {shift[1]:+.3f})")
                    target = tool + np.array([shift[0], shift[1], 0.0])
                    drop = st.get("drop")
                    if drop is not None:
                        path, w = self.line_path(s, pv, obs, target, np.array(drop["ax"]), step=0.02)
                        st["path"] = [self.turn_tool(pv, q, np.array(drop["G"]))[0] for q in path]
                    else:
                        st["path"], w = self.line_path(s, pv, obs, target, np.array([0.0, 0.0, 1.0]), step=0.02)
                    st["k"], st["t_aim"] = 0, obs["time"]
                    return False
                s["grip"] = False
                st["phase"], st["t_rel"] = "check", obs["time"]
            return False
        if st["phase"] == "lift":  # (the part back to the carry pose, then the drive to the stand again)
            if self.follow(st, s, obs, ARM_FAST, tol=0.05):
                st.update(phase="drive", sub=[dict(kind="drive", goal=list(st["appr"]), heading=st["hd"], timeout=60.0),
                                              dict(kind="straight", goal=list(st["stand"]), heading=st["hd"], timeout=15.0)])
            return False
        if st["phase"] == "check":  # (let go: up and away to the tucked arm at once; where the part landed is read once it has fallen)
            name = st.get("part")
            # (the last part: the vehicle stays where it stands, the arm folds over the crate; any drive now only
            # delays the end, which wants the vehicle still)
            st["last"] = not [n for n in pv["parts"] if n != name and self.where(pv, n) not in ("crate", "elsewhere") and n not in s["gave_up"]]
            st["phase"] = "away"
            up_dir = np.array([0.0, 0.0, 1.0])
            p = self.tool_world(s, pv, obs) + np.array([0.0, 0.0, 0.12])
            up, w = self.line_path(s, pv, obs, p, up_dir, step=0.03)
            tuck, w = self.line_path(dict(s, q_tgt=up[-1]), pv, obs, self.carry_point(pv, TUCK), up_dir, p0=p, wrist0=up_dir, step=0.05)
            st["path"], st["k"] = up + tuck, 0
            return False
        if st["phase"] == "away":
            if self.follow(st, s, obs, ARM_FAST, tol=0.05) and obs["time"] - st["t_rel"] >= 0.9:
                name = st.get("part")
                ok = name is not None and self.where(pv, name) == "crate"
                log(f"t={obs['time']:.1f} {name} {'in the crate' if ok else 'missed the crate: ' + self.where(pv, name) if name else 'gone'}")
                if st.get("last"):
                    return True
                st["phase"] = "back"
                st["sub"] = [dict(kind="straight", goal=st["appr"], heading=st["hd"], timeout=15.0)]
            return False
        if st["phase"] == "back":
            sub = st["sub"][0]
            sub.setdefault("t0", obs["time"])
            wh, done = self.run_step(sub, s, pv, obs, None)
            st["_wheels"] = wh
            return bool(done or obs["time"] - sub["t0"] > sub["timeout"])
        return False

    def tool_world(self, s, pv, obs):
        tool_r, _ = self.kin.points(self.joints(obs))
        _, _, p, R = self.pose(pv)
        return p + R @ tool_r

    # -- steps ---------------------------------------------------------------------------------------------------
    def follow(self, st, s, obs, rate, tol=0.03):
        """Move the arm's targets along st['path'] (joint waypoints) at `rate`; True once at the end and there."""
        path = st["path"]
        if not path:
            return True
        k = min(st.get("k", 0), len(path) - 1)
        goal = path[k]
        step = rate * self.dt
        q = s["q_tgt"]
        if st.get("sync"):  # (all the joints arrive together: the tool keeps to the path between its waypoints)
            m = max(abs(goal[n] - q[n]) for n in self.kin.joints)
            f = 1.0 if m <= step else step / m
            for n in self.kin.joints:
                q[n] = goal[n] if f >= 1.0 else q[n] + f * (goal[n] - q[n])
        else:
            for n in self.kin.joints:
                d = goal[n] - q[n]
                q[n] += float(np.clip(d, -step, step))
        arrived = all(abs(goal[n] - q[n]) < 1e-6 for n in self.kin.joints)
        joints = obs["ugv"]["joints"]
        if arrived and k < len(path) - 1:
            if all(abs(goal[n] - joints[n][0]) < 0.12 for n in self.kin.joints):
                st["k"] = k + 1
            return False
        return arrived and k == len(path) - 1 and all(abs(goal[n] - joints[n][0]) < tol for n in self.kin.joints)

    def run_step(self, st, s, pv, obs, hoist):
        """One control step of a step: (wheels, done)."""
        k = st["kind"]
        stop = self.wheels(0.0, 0.0)
        if k == "stop":
            return stop, False
        if k == "wait":
            return stop, obs["time"] - st["t0"] >= st["seconds"]
        if k == "arm_line":
            if "path" not in st:
                target = CARRY if (st["to"] == "tuck" and held_part(pv)) else TUCK
                st["path"], w = self.line_path(s, pv, obs, self.carry_point(pv, target), np.array([0.0, 0.0, 1.0]), step=0.05)
                st["k"] = 0
            return stop, self.follow(st, s, obs, st.get("speed", ARM_FAST), tol=0.06)
        if k == "drive":
            return self.drive(st, s, pv, obs)
        if k == "straight":
            return self.straight(st, pv)
        if k == "lever":
            return stop, self.run_lever(st, s, pv, obs)
        if k == "pick":
            return stop, self.run_pick(st, s, pv, obs)
        if k == "crate":
            done = self.run_crate(st, s, pv, obs)
            return st.pop("_wheels", stop), done
        return stop, True

    def spin(self, pv, err, st=None):
        """Turn on the spot toward `err` where there is room; where there is not, first straight on or back, whichever
        way gains room (the spin walks the root body), at most 0.2 m in a step."""
        xy, psi, _, _ = self.pose(pv)
        room = self.planner.clearance(xy, pv["pan"])
        if room < SPIN_ROOM and (st is None or st.get("crept", 0.0) < 0.2):
            fwd = np.array([-math.sin(psi), math.cos(psi)])
            ahead, behind = self.planner.clearance(xy + 0.1 * fwd, pv["pan"]), self.planner.clearance(xy - 0.1 * fwd, pv["pan"])
            if st is not None:
                st["crept"] = st.get("crept", 0.0) + V_SLOW * self.dt
            return self.wheels(V_SLOW if ahead >= behind else -V_SLOW, 0.0)
        w = float(np.clip(2.0 * err, -W_TURN, W_TURN))
        return self.wheels(0.0, math.copysign(max(abs(w), 0.3), err))

    def clear_of_bucket(self, pv, pts):
        """Is the polyline `pts` clear of the bucket, of the stretches the hoist is about to carry it over and of the
        flying robots that fly low, as they are now? (The static obstacles do not move: the planner keeps the path
        clear of them.)"""
        b = pv["bucket"]["pos"]
        keep = self.hoist.get("keep") or ()
        uavs = self.low_uavs(pv)
        for a, c in zip(pts[:-1], pts[1:]):
            n = max(2, int(math.dist(a, c) / (self.planner.r / 2)) + 1)
            for k in range(n + 1):
                x, y = a[0] + k / n * (c[0] - a[0]), a[1] + k / n * (c[1] - a[1])
                if math.hypot(x - b[0], y - b[1]) < BUCKET_ROOM:
                    return False
                if any(float(seg_dist(x, y, (x0, y0), (x1, y1))) < room for x0, y0, x1, y1, room in keep):
                    return False
                if any(math.hypot(x - ux, y - uy) < UAV_ROOM + INFLATE for ux, uy in uavs):
                    return False
        return True

    def drive(self, st, s, pv, obs):
        xy, psi, _, _ = self.pose(pv)
        goal = np.array(st["goal"], dtype=float)
        if st.get("arrived"):
            if st.get("heading") is None:
                return self.wheels(0.0, 0.0), True
            st.setdefault("turn_at", [float(xy[0]), float(xy[1])])
            st["turn_t"] = st.get("turn_t", 0.0) + self.dt
            if math.dist(xy, st["turn_at"]) > TURN_DRIFT or st["turn_t"] > TURN_MAX:
                # (a turn on the spot that carries the vehicle off in an arc, a wheel slipping: not there any more, so
                # stop and drive to the goal again from where it got to, twice; then end the drive where it is, the
                # step after it checks how far off it stands. Turning on would wander anywhere, a burning pan's ring
                # included)
                log(f"t={obs['time']:.1f} drive: the turn has carried it {math.dist(xy, st['turn_at']):.2f} m in {st['turn_t']:.1f} s"
                    f"{': to the goal again' if st.get('turn_redo', 0) < 2 else ': stopped there'}")
                if st.get("turn_redo", 0) >= 2:
                    return self.wheels(0.0, 0.0), True
                st["turn_redo"] = st.get("turn_redo", 0) + 1
                for k in ("arrived", "turn_at", "turn_t", "near_t", "near_back", "crept", "path"):
                    st.pop(k, None)
                return self.wheels(0.0, 0.0), False
            return self.turn(pv, st["heading"], st)
        if "path" in st and obs["time"] - st.get("t_check", -9.0) >= 0.5:  # (the bucket moves: a path it now crosses goes)
            st["t_check"] = obs["time"]
            if not self.clear_of_bucket(pv, [list(xy)] + [list(p) for p in st["path"][1:]]):
                st["t_plan"] = -9.0
        if "path" not in st or obs["time"] - st.get("t_plan", -9) > 2.0:
            occ = self.occ(pv)
            path = self.planner.plan(occ, xy, goal, pv["pan"])
            st["t_plan"] = obs["time"]
            if path is None:
                st.pop("path", None)
                return self.wheels(0.0, 0.0), False
            st["path"] = path
        path = [np.array(p) for p in st["path"]]
        fwd = np.array([-math.sin(psi), math.cos(psi)])
        dg = float(np.linalg.norm(goal - xy))
        if dg < st.get("tol", 0.10):  # (close enough: a straight drive in, or the arm's IK on the true pose, takes up the rest)
            st["arrived"] = True
            return self.wheels(0.0, 0.0), st.get("heading") is None
        along = float((goal - xy) @ fwd)
        if dg < 0.30 and abs(float((goal - xy) @ np.array([-fwd[1], fwd[0]]))) < 0.08:
            # near and nearly in line: straight on along the heading until level with it (a spin on the spot would move
            # the root: the skid-steer's turning centre is not the root body's origin); further off, the pursuit below
            if abs(along) < 0.02:
                st["arrived"] = True
                return self.wheels(0.0, 0.0), st.get("heading") is None
            return self.wheels(math.copysign(min(V_SLOW, 0.8 * abs(along) + 0.03), along), 0.0), False
        st["near_t"] = st.get("near_t", 0.0) + self.dt if dg < 0.5 else 0.0
        if st["near_t"] > NEAR_MAX and dg < 0.25:  # (it has hunted round the goal this long: near enough, the straight stretch takes up the rest)
            log(f"t={obs['time']:.1f} drive: {dg:.2f} m from its goal after {st['near_t']:.0f} s near it: taken as there")
            st["arrived"] = True
            return self.wheels(0.0, 0.0), st.get("heading") is None
        if dg < NEAR_GOAL:
            # near the goal: face it (or turn the back to it, whichever is less of a turn), then straight at it; the
            # pursuit, its look-ahead on the goal itself, would circle a goal it passes beside at its turning radius
            to_goal = heading_of(goal - xy)
            e_fwd, e_back = wrap_a(to_goal - psi), wrap_a(to_goal + math.pi - psi)
            back = st.get("near_back")
            if back is None or abs(e_back if back else e_fwd) > math.radians(100.0):  # (kept once chosen: no dithering abeam of it)
                back = abs(e_back) < abs(e_fwd)
                st["near_back"] = back
            e = e_back if back else e_fwd
            if abs(e) > math.radians(15.0):
                return self.spin(pv, e, st), False
            v = min(V_SLOW, 0.8 * dg + 0.04) * math.cos(e)
            return self.wheels(-v if back else v, float(np.clip(2.0 * e, -0.5, 0.5))), False
        # pure pursuit: the point on the path 0.35 m ahead of the nearest one
        best, bi, bf = 1e9, 0, 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            ab = b - a
            f = float(np.clip((xy - a) @ ab / (ab @ ab + 1e-12), 0.0, 1.0))
            d = float(np.linalg.norm(a + f * ab - xy))
            if d < best:
                best, bi, bf = d, i, f
        look, i, f = 0.35, bi, bf
        pt = path[-1]
        while i < len(path) - 1:
            a, b = path[i], path[i + 1]
            seg = float(np.linalg.norm(b - a))
            rest = (1.0 - f) * seg
            if rest >= look:
                pt = a + (f + look / max(seg, 1e-9)) * (b - a)
                break
            look -= rest
            i, f = i + 1, 0.0
        dist_goal = float(np.linalg.norm(goal - xy))
        err = wrap_a(heading_of(pt - xy) - psi)
        back = abs(err) > math.radians(110.0) and self.planner.length([list(xy)] + [list(p) for p in path[bi + 1:]]) < 2.5
        if back:  # (the way lies behind and is short: drive it backward instead of turning round)
            err = wrap_a(err - math.pi)
        if abs(err) > math.radians(35.0):  # (turn in place, where there is room)
            return self.spin(pv, err, st), False
        v = min(V_DRIVE, 0.8 * dist_goal + 0.05) * math.cos(err) ** 2
        w = float(np.clip(2.5 * err, -W_TURN, W_TURN))
        return self.wheels(-v if back else v, w), False

    def turn(self, pv, heading, st=None):
        """Turn in place to the heading (skid steering needs a firm turn: never slower than 0.3 rad/s), where there is
        room."""
        psi = self.pose(pv)[1]
        err = wrap_a(heading - psi)
        if abs(err) < math.radians(2.0):
            return self.wheels(0.0, 0.0), abs(pv["ugv"]["yaw_rate"]) < 5.0
        return self.spin(pv, err, st), False

    def straight(self, st, pv):
        """Straight to the goal, forward or back, along the line through the goal at the step's heading: the vehicle
        steers gently back toward that line as it goes (pure pursuit LINE_LOOK ahead, at most LINE_STEER off the
        heading), taking up part of what the drive's arrival left beside it (a skid-steer cannot step sideways)."""
        xy, psi, _, _ = self.pose(pv)
        goal = np.array(st["goal"], dtype=float)
        hd = st["heading"]
        f = np.array([-math.sin(hd), math.cos(hd)])  # (the line's direction: the heading's forward)
        n = np.array([-f[1], f[0]])
        along = float((goal - xy) @ f)
        if abs(along) < 0.012:
            return self.wheels(0.0, 0.0), abs(pv["ugv"]["vel"][0]) + abs(pv["ugv"]["vel"][1]) < 0.03
        if abs(wrap_a(hd - psi)) > math.radians(25.0):  # (far off the line's heading, a drive along it would arc off: turn first,
            st.setdefault("spin_at", [float(xy[0]), float(xy[1])])  # unless the turn carries it off: then stop, the next step looks)
            if math.dist(xy, st["spin_at"]) > TURN_DRIFT:
                return self.wheels(0.0, 0.0), True
            return self.spin(pv, wrap_a(hd - psi), st), False
        lat = float((xy - goal) @ n)  # (> 0: left of the line)
        v = math.copysign(min(V_STRAIGHT, 1.2 * abs(along) + 0.02), along)
        steer = -math.copysign(1.0, v) * float(np.clip(math.atan2(lat, LINE_LOOK), -LINE_STEER, LINE_STEER))
        err = wrap_a(hd + steer - psi)
        return self.wheels(v, float(np.clip(2.0 * err, -0.5, 0.5))), False


# ============================================================================================= the facade
class Vehicle:
    """The vehicle's program: `Privileged` with the true state, `Baseline` without."""

    def __init__(self):
        self.base = Baseline()
        self.prog = None
        self.privileged = False

    def setup(self, mis, privileged=False):
        self.base.setup(mis)
        self.privileged = bool(privileged) and self.base.kin is not None
        if self.privileged:
            self.prog = Privileged(self.base.kin, mis, self.base.wheels)

    @property
    def mis(self):
        return self.base.mis

    def init_state(self):
        return self.prog.init_state() if self.privileged else self.base.init_state()

    def act(self, s, obs, hoist):
        """`hoist`: what the hoist's controller says (its phase; with the true state, where the bucket is about to pass
        and whether the hoist has settled); the baseline takes the phase only."""
        if self.privileged and "privileged" in obs and "ugv" in obs and obs["privileged"].get("ugv"):
            return self.prog.act(s, obs, hoist)
        if self.privileged:
            return {}
        return self.base.act(s, obs, hoist.get("phase") if isinstance(hoist, dict) else hoist)
