"""Cabinet door hanging: the privileged scripted two-arm reference (reads the simulator's truth).

`Reference(ep).run()` drives an episode object whose `step(action)` applies one wire action and whose `env.w` is a
`door_sim.World` holding the true state (over the wire: wire_world.ShadowEpisode, synced from `truth` every step).

First the old door comes off. The cup seals on its front face high up (so the arm passes over the new door in its
rack), holds it coaxial with the carcass knuckles and 0.5 mm up off them (which unloads its pins), and the left arm
pulls the two old pins (the top one from above, the bottom one from the front, below the top hinge), travelling over
the new door, and drops them on the table behind the stands. The cup slides the old door out of the knuckles, lifts
it 3 cm, carries it sideways to the parking rack, lowers it into the channel and lets go.

Then the right arm's vacuum cup takes the new door from the rack by its front face (mid-height, 8 cm from the leaf's centre
toward the free edge, so the cup arm keeps clear of the pin arm), lifts it out of the rack, carries it forward past
the rack and across, lines it up in front of the carcass, slides its knuckles in between the carcass knuckles at
2 mm per step and holds the door on the hung pose (servoing the door's true pose, with an integral term against the
load). The left arm, its hand pitched back 22 deg about the fingers' closing axis (which keeps the Panda's wrist off
the top hinge), takes a pin of the unit's class by its grip post and drives it into the bottom hinge (approached
from the front, below the top hinge, its tip brought onto the axis slowly), then a second one into the top hinge
from above. The drive pushes at most 5 mm per step and, when the pin stalls, backs off and searches in a spiral
around the axis. Then the cup lets go, both arms back off, and the reference holds still until the episode ends.
"""

import math

import mujoco
import numpy as np

import door_sim as S

R_DOWN = np.column_stack([[0, 1, 0], [1, 0, 0], [0, 0, -1]]).astype(float)   # gripper down, fingers along y
R_CUP = np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]]).astype(float)    # cup facing the door frame's +x
PIN_PITCH = math.radians(-22.0)
R_PIN = np.array([[math.cos(PIN_PITCH), 0, math.sin(PIN_PITCH)], [0, 1, 0],
                  [-math.sin(PIN_PITCH), 0, math.cos(PIN_PITCH)]]) @ R_DOWN
HIGH = S.TABLE_Z + 0.55                # the left hand's travel height over the new door (its top at TABLE_Z + 0.45)
OLD_GRASP_Z = S.DOOR_H - 0.04         # the old door's grasp height above its bottom edge
GRASP_OFF = 0.08                     # the cup's grasp point: this far from the leaf's centre toward the free edge
HOLD = {"l": [0.0] * 6 + [-1.0], "r": [0.0] * 6 + [-1.0], "cams": 0}


def rotvec(Rm):
    """The rotation vector of rotation matrix Rm."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(Rm, float).reshape(9))
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def rot_err(R_des, R):
    return rotvec(R_des @ R.T)


class Reference:
    def __init__(self, ep):
        self.ep = ep
        self.w = ep.env.w
        self.d = self.w.d
        self.integ = {"l": np.zeros(3), "r": np.zeros(3)}
        self.integ_r = {"l": np.zeros(3), "r": np.zeros(3)}
        self.events = []
        self.t = 0
        self.R_hand = self.w.cR @ R_PIN                # the pin hand square to the carcass

    # --- low-level -------------------------------------------------------------------------------
    def servo(self, s, p_des, R_des, grip, ki=0.0, ilim=0.03):
        """A Cartesian step toward (p_des, R_des); with ki, an integral term on both (against a held load)."""
        arm = self.w.arms[s]
        e = p_des - arm.pos()
        er = rot_err(R_des, arm.R())
        if ki:
            self.integ[s] = np.clip(self.integ[s] + ki * e, -ilim, ilim)
            self.integ_r[s] = np.clip(self.integ_r[s] + ki * er, -0.15, 0.15)
        return [float(x) for x in e + self.integ[s]] + [float(x) for x in er + self.integ_r[s]] + [float(grip)]

    def servo_capped(self, s, p_des, R_des, grip, vmax, ki=0.0):
        """servo toward p_des, the step capped at vmax (m)."""
        p = self.w.arms[s].pos()
        e = p_des - p
        n = np.linalg.norm(e)
        return self.servo(s, p + (e * (vmax / n) if n > vmax else e), R_des, grip, ki=ki)

    def run_until(self, cmd_fn, done_fn, max_steps, name):
        for _ in range(max_steps):
            act = cmd_fn()
            act["cams"] = 0
            self.ep.step(act)
            self.t += 1
            if done_fn():
                self.events.append((name, self.t))
                return True
        self.events.append((name + "_timeout", self.t))
        return False

    # --- the door --------------------------------------------------------------------------------
    def face_point(self, name="door", z=S.DOOR_H / 2):
        pd, Rd = self.w.door_pose(name)
        return pd + Rd @ np.array([-S.DOOR_T / 2, -S.EDGE_GAP - S.DOOR_W / 2 - GRASP_OFF, z])

    def hold_cmd(self, p_des, R_des, vmax=None, name="door"):
        """Move / hold the door frame toward (p_des, R_des); vmax (m per step) limits the approach speed."""
        pd, Rd = self.w.door_pose(name)
        if vmax is not None:
            e = p_des - pd
            n = np.linalg.norm(e)
            if n > vmax:
                p_des = pd + e * (vmax / n)
        arm = self.w.arms["r"]
        Rerr = R_des @ Rd.T
        return self.servo("r", p_des + Rerr @ (arm.pos() - pd), Rerr @ arm.R(), 1.0, ki=0.15)

    def door_err(self, p_des, R_des, name="door"):
        pd, Rd = self.w.door_pose(name)
        return float(np.linalg.norm(pd - p_des)), float(np.linalg.norm(rot_err(R_des, Rd)))

    def park_l(self):
        return self.servo("l", np.array([0.35, 0.40, 1.20]), R_DOWN, -1.0)

    # --- the whole job ---------------------------------------------------------------------------
    def run(self):
        self.job()
        while True:                                   # hold still until the grader ends the episode
            self.ep.step({k: (list(v) if isinstance(v, list) else v) for k, v in HOLD.items()})

    def job(self):
        if not self.take_down():
            return False
        self.home()
        return self.hang()

    def home(self):
        """The left arm back to the start posture by joint steps (it resets the arm's redundancy after the pin pulls'
        long reaches; the right arm stays where it is: its start posture is beyond the new door)."""
        w, d = self.w, self.d
        Q = [float(x) for x in S.HOME_Q]
        self.run_until(lambda: {"l": Q + [-1.0], "r": self.servo("r", w.arms["r"].pos(), w.arms["r"].R(), -1.0)},
                       lambda: np.max(np.abs(d.qpos[w.arms["l"].jq] - S.HOME_Q)) < 0.02, 120, "home")
        self.integ["l"][:] = 0
        self.integ_r["l"][:] = 0

    # --- the old door: pull its two pins, carry it to the parking rack -------------------------------
    def take_down(self):
        w, d = self.w, self.d
        Rold = lambda: w.door_pose("old")[1] @ R_CUP
        fp = lambda: self.face_point("old", OLD_GRASP_Z)        # high on the leaf: the arm passes over the new door
        pre = fp() + w.door_pose("old")[1] @ np.array([-0.06, 0, 0])
        ok = self.run_until(lambda: {"r": self.servo("r", pre, Rold(), -1.0), "l": self.park_l()},
                            lambda: np.linalg.norm(w.arms["r"].pos() - pre) < 0.004, 150, "old_pre")
        ok &= self.run_until(lambda: {"r": self.servo("r", fp() + w.door_pose("old")[1] @
                                                      np.array([0.003, 0, 0]), Rold(), 1.0), "l": self.park_l()},
                             lambda: w.held == "old", 60, "old_seal")
        if not ok:
            return False
        # unload the pins: hold the door coaxial with the carcass knuckles, 0.5 mm up off its knuckles' faces
        p_old, R_old = w.cpos + w.cR @ np.array([0, 0, 0.0005]), w.cR
        hold = lambda: self.hold_cmd(p_old, R_old, name="old")
        self.run_until(lambda: {"r": hold(), "l": self.park_l()},
                             lambda: self.door_err(p_old, R_old, "old")[0] < 0.0003 and
                             self.door_err(p_old, R_old, "old")[1] < 0.002, 100, "old_centre")
        for k, h in ((w.npin - 1, 1), (w.npin - 2, 0)):
            ok &= self.pull(k, h, hold)
        Rc = w.cR
        park = S.PARK + np.array(w.episode["park"])
        out = p_old + Rc @ np.array([-0.045, 0, 0])
        z = p_old[2] + 0.03
        I3 = np.eye(3)
        for name, tgt, R, tol, vmax in (("old_out", out, R_old, 0.003, 0.004),
                                        ("old_up", np.array([out[0], out[1], z]), R_old, 0.003, 0.004),
                                        ("old_across", np.array([out[0], park[1], z]), I3, 0.004, 0.006),
                                        ("old_over", np.array([park[0], park[1], z]), I3, 0.001, 0.01)):
            ok &= self.run_until(lambda tgt=tgt, R=R, vmax=vmax: {"r": self.hold_cmd(tgt, R, vmax, name="old"),
                                                                  "l": self.park_l()},
                                 lambda tgt=tgt, R=R, tol=tol: self.door_err(tgt, R, "old")[0] < tol and
                                 self.door_err(tgt, R, "old")[1] < 0.01, 400, name)
            if w.held != "old":
                self.events.append(("old_dropped", self.t))
                return False
        # down into the channel: the door's x and y are held within 1.5 mm before each 2 mm step down
        z_end = S.TABLE_Z + 0.003

        def down():
            pd, _R = w.door_pose("old")
            ok_xy = np.hypot(pd[0] - park[0], pd[1] - park[1]) < 0.0015
            zt = max(z_end, pd[2] - 0.002) if ok_xy else pd[2]
            return {"r": self.hold_cmd(np.array([park[0], park[1], zt]), I3, name="old"), "l": self.park_l()}
        self.run_until(down, lambda: w.door_pose("old")[0][2] < z_end + 0.002, 400, "old_down")
        if w.held != "old":
            self.events.append(("old_dropped", self.t))
            return False
        back = w.arms["r"].pos() + np.array([-0.10, 0, 0.03])
        self.run_until(lambda: {"r": self.servo("r", back, w.arms["r"].R(), -1.0), "l": self.park_l()},
                       lambda: np.linalg.norm(w.arms["r"].pos() - back) < 0.01, 80, "old_release")
        self.integ["r"][:] = 0
        self.integ_r["r"][:] = 0
        return True             # the old door is parked; the steps' timeouts are in the events

    def pull(self, k, h, hold):
        """Left arm: take old pin k by its post (the top hinge's from above, the bottom hinge's from the front), pull
        it up out of hinge h, and drop it on the table behind the stands."""
        w, d = self.w, self.d
        post = lambda: d.xpos[w.pins[k]] + d.xmat[w.pins[k]].reshape(3, 3) @ S.POST + np.array([0, 0, 0.012])
        Rc = w.cR
        self.integ["l"][:] = 0
        self.integ_r["l"][:] = 0
        p0 = post()
        f = p0 + Rc @ np.array([-0.06, 0, 0])
        via = [p0 + np.array([0, 0, 0.10])] if h == 1 else [np.array([f[0], f[1], HIGH]), f + np.array([0, 0, 0.03]),
                                                            f]
        ok = True
        for i, tgt in enumerate(via):
            ok &= self.run_until(lambda tgt=tgt: {"l": self.servo("l", tgt, self.R_hand, -1.0, ki=0.1), "r": hold()},
                                 lambda tgt=tgt: np.linalg.norm(w.arms["l"].pos() - tgt) < 0.004
                                 and np.linalg.norm(rot_err(self.R_hand, w.arms["l"].R())) < 0.02, 200,
                                 f"pull{k}_via{i}")
        ok &= self.run_until(lambda: {"l": self.servo("l", post(), self.R_hand, -1.0, ki=0.1), "r": hold()},
                             lambda: np.linalg.norm(w.arms["l"].pos() - post()) < 0.002, 100, f"pull{k}_down")
        n = [0]

        def grip_done():
            n[0] += 1
            return n[0] > 12
        at = post()             # a loose pin rides up as the fingers close on it: grip where the post was found
        self.run_until(lambda: {"l": self.servo("l", at, self.R_hand, 1.0, ki=0.1), "r": hold()}, grip_done, 20,
                       f"pull{k}_grip")
        top = w.from_carcass([0, 0, w.tops[h]])
        up = w.arms["l"].pos() + np.array([0, 0, S.SHAFT + 0.02])
        ok &= self.run_until(lambda: {"l": self.servo_capped("l", up, self.R_hand, 1.0, 0.004, ki=0.05), "r": hold()},
                             lambda: self.tip(k)[2] > top[2] + 0.008, 150, f"pull{k}_up")
        out = up + Rc @ np.array([-0.08, 0, 0])
        drop = np.array([0.30, 0.45 if h else 0.52, S.TABLE_Z + 0.20])    # behind the stands, clear of every rack
        for i, tgt in enumerate((out, np.array([out[0], out[1], max(out[2], HIGH)]),
                                 np.array([drop[0], drop[1], HIGH]), drop)):
            ok &= self.run_until(lambda tgt=tgt: {"l": self.servo_capped("l", tgt, self.R_hand, 1.0, 0.01), "r": hold()},
                                 lambda tgt=tgt: np.linalg.norm(w.arms["l"].pos() - tgt) < 0.006, 200,
                                 f"pull{k}_away{i}")
        self.run_until(lambda: {"l": self.servo("l", w.arms["l"].pos(), self.R_hand, -1.0), "r": hold()},
                       lambda: w.arms["l"].gripper() > 0.075, 30, f"pull{k}_open")
        return ok

    # --- the new door ------------------------------------------------------------------------------
    def hang(self):
        w = self.w
        Rcup = lambda: w.door_pose()[1] @ R_CUP                      # square to the door's face as it stands
        pre = self.face_point() + w.door_pose()[1] @ np.array([-0.06, 0, 0])
        ok = self.run_until(lambda: {"r": self.servo("r", pre, Rcup(), -1.0), "l": self.park_l()},
                            lambda: np.linalg.norm(w.arms["r"].pos() - pre) < 0.006, 150, "cup_pre")
        ok &= self.run_until(lambda: {"r": self.servo("r", self.face_point() + w.door_pose()[1] @ np.array([0.003, 0, 0]),
                                                      Rcup(), 1.0), "l": self.park_l()},
                             lambda: w.held == "door", 60, "seal")
        if not ok:
            return False
        p0, _ = w.door_pose()
        Rc = w.cR
        hung = w.cpos.copy()
        front = w.from_carcass([-0.035, 0, 0])
        # out of the rack, forward past the rack, across, down in front of the opening, in
        for name, tgt, tol, vmax in (("lift", p0 + np.array([0, 0, 0.07]), 0.004, None),
                                     ("forward", np.array([front[0], p0[1], p0[2] + 0.07]), 0.004, None),
                                     ("shift", np.array([front[0], front[1], p0[2] + 0.07]), 0.004, None),
                                     ("front", front, 0.0005, 0.01),
                                     ("slide_in", hung, 0.0006, 0.002)):
            ok &= self.run_until(lambda tgt=tgt, vmax=vmax: {"r": self.hold_cmd(tgt, Rc, vmax), "l": self.park_l()},
                                 lambda tgt=tgt, tol=tol: self.door_err(tgt, Rc)[0] < tol and
                                 self.door_err(tgt, Rc)[1] < 0.005, 400, name)
            if w.held != "door":
                self.events.append(("dropped", self.t))
                return False
        good = [k for k in range(w.npin) if w.pin_class[k] == w.unit["pin_class"]]
        # the bottom hinge first (from the front, below the top hinge), then the top hinge from above: the other
        # order puts the hand on the seated top pin's post
        for k, h in ((good[0], 0), (good[1], 1)):
            ok &= self.pin(k, h, hung, Rc)
        back_r = w.arms["r"].pos() + Rc @ np.array([-0.16, 0, 0])
        self.run_until(lambda: {"r": self.servo("r", back_r, Rcup(), -1.0), "l": self.park_l()},
                       lambda: np.linalg.norm(w.arms["r"].pos() - back_r) < 0.01, 80, "release")
        return ok

    def hold(self, hung, Rc):
        return self.hold_cmd(hung, Rc)

    def tip(self, k):
        return self.d.xpos[self.w.pins[k]] + self.d.xmat[self.w.pins[k]].reshape(3, 3) @ np.array([0, 0, -S.SHAFT])

    def pin_error(self, k, pin_des):
        """The error that moves the pin's head to pin_des: horizontal from its tip (a tilted pin's tip is what meets
        the bore), vertical from its head."""
        e = pin_des - self.d.xpos[self.w.pins[k]]
        et = (pin_des - np.array([0, 0, S.SHAFT])) - self.tip(k)
        return np.array([et[0], et[1], e[2]])

    def pin_cmd(self, k, pin_des, grip, R, ki=0.0, vmax=None):
        """Left arm: move so the pin goes to pin_des (the pin rides in the fingers); vmax (m per step) caps the step."""
        arm = self.w.arms["l"]
        e = self.pin_error(k, pin_des)
        n = np.linalg.norm(e)
        if vmax is not None and n > vmax:
            e = e * (vmax / n)
        return self.servo("l", arm.pos() + e, R, grip, ki=ki)

    def pin(self, k, h, hung, Rc):
        w, d = self.w, self.d
        R_take = R_PIN                                               # the pins stand square to the table
        post = lambda: d.xpos[w.pins[k]] + d.xmat[w.pins[k]].reshape(3, 3) @ S.POST + np.array([0, 0, 0.012])
        hold = lambda: self.hold(hung, Rc)
        above = post() + np.array([0, 0, 0.10])
        self.integ["l"][:] = 0
        self.integ_r["l"][:] = 0
        ok = self.run_until(lambda: {"l": self.servo("l", above, R_take, -1.0, ki=0.1), "r": hold()},
                            lambda: np.linalg.norm(w.arms["l"].pos() - above) < 0.004
                            and np.linalg.norm(rot_err(R_take, w.arms["l"].R())) < 0.02, 150, f"pin{k}_above")
        ok &= self.run_until(lambda: {"l": self.servo("l", post(), R_take, -1.0, ki=0.1), "r": hold()},
                             lambda: np.linalg.norm(w.arms["l"].pos() - post()) < 0.002, 100, f"pin{k}_down")
        n = [0]

        def grip_done():
            n[0] += 1
            return n[0] > 12
        self.run_until(lambda: {"l": self.servo("l", post(), R_take, 1.0, ki=0.1), "r": hold()}, grip_done, 20,
                       f"pin{k}_grip")
        top = w.from_carcass([0, 0, w.tops[h]])
        out = Rc @ np.array([-0.04, 0, 0])                            # in front of the axis
        p_lift = d.xpos[w.pins[k]] + np.array([0, 0, S.SHAFT + 0.03])
        z_start = top[2] + S.SHAFT + 0.012
        if h == 1:
            route = [p_lift, np.array([top[0], top[1], top[2] + S.SHAFT + 0.05]), np.array([top[0], top[1], z_start])]
        else:
            route = [p_lift, np.array([top[0] + out[0], top[1] + out[1], p_lift[2]]),
                     np.array([top[0] + out[0], top[1] + out[1], z_start]), np.array([top[0], top[1], z_start])]
        for i, tgt in enumerate(route):
            # the start of the drive and (bottom hinge) the point in front of it: within 1 mm
            tol = 0.001 if i >= len(route) - 2 and i > 0 else 0.003
            vmax = 0.003 if i == len(route) - 1 else None             # onto the axis slowly
            ok &= self.run_until(lambda tgt=tgt, vmax=vmax: {"l": self.pin_cmd(k, tgt, 1.0, self.R_hand, ki=0.1,
                                                                                vmax=vmax), "r": hold()},
                                 lambda tgt=tgt, tol=tol: np.linalg.norm(self.pin_error(k, tgt)) < tol, 250,
                                 f"pin{k}_r{i}")
        # drive: push about 5 mm per step (the detent needs about 6 N); when stalled, back off 1 mm and search in a
        # spiral around the axis
        stall, z_best, spiral = 0, 9.0, 0.0
        self.integ["l"][:] = 0

        def drive():
            nonlocal stall, z_best, spiral
            z = d.xpos[w.pins[k]][2]
            if z < z_best - 0.0005:
                z_best, stall = z, 0
            else:
                stall += 1
            if stall > 8:
                spiral += 0.6
            r = min(0.0004 * spiral, 0.004)
            tgt = np.array([top[0] + r * math.cos(spiral), top[1] + r * math.sin(spiral), top[2] - 0.004])
            dp = self.pin_error(k, tgt)
            dp[2] = 0.001 if stall % 15 == 14 else max(dp[2], -0.005)
            return {"l": self.servo("l", w.arms["l"].pos() + dp, self.R_hand, 1.0), "r": hold()}
        ok &= self.run_until(drive, lambda: w.snapped[k], 800, f"pin{k}_drive")
        self.run_until(lambda: {"l": self.servo("l", w.arms["l"].pos(), self.R_hand, -1.0), "r": hold()},
                       lambda: w.arms["l"].gripper() > 0.075, 30, f"pin{k}_open")
        back = w.arms["l"].pos() + Rc @ np.array([-0.06, 0, 0.02])
        self.run_until(lambda: {"l": self.servo("l", back, self.R_hand, -1.0), "r": hold()},
                       lambda: np.linalg.norm(w.arms["l"].pos() - back) < 0.01, 60, f"pin{k}_back")
        return ok
