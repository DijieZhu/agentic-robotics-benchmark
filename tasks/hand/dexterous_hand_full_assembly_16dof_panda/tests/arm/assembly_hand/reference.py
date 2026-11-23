"""The Panda assembles the hand's digits on the raised fixture: the privileged reference program (no observation).
After the tensegrity cell's reference (the bench plan's grasp first, every IK solution at the pick, the place checks
from the outside in with the fingers closed to the grasp's width, the planned carries, the closed-loop insertion on
the unit's live pose, the slip check after the lift) and the SO-101 cell's (the arm guard: stop, back off and retry
when a product joint turns unplanned).

Privileged state only: the units' exact poses and joint angles, the slots' design poses (the palm base as it is now),
the Panda's own kinematics (`panda_ik.PandaKin`), the cell's reach checker (`reach.Checker`) and a planner on a
copy of the cell (`motion.Planner`). Nothing but the Panda, its stock flat pads and the physics moves a unit: no jig,
no position lock, no collision switched off, no unit set anywhere by the program.

Scope: the nine units in the build's order: the index finger, the ring finger, the middle finger between
its parked neighbours, the thumb (it requires the index finger), then the five palm parts from above: the electronics
stack, the upper shell, the dorsal cover, the power hub and the press-fit cap (each requires the one before).

Per digit:
  grasps   the cell's grasp table (grasp_table.json, hidden side; both signs of each row's closing axis), in its
           order: the first that passes the pick check where the unit rests (reach.Checker: IK at the grasp and 10 /
           30 / 60 mm back along the approach with the hand open to span + 6 mm, the arm 4 mm, the fingers 2 mm from
           every other body and 0.5 mm from the gripped unit) and the place check at its mate (the base on its path
           from the pre-insertion pose to home in 2 mm steps, from the outside in, on one IK branch, the fingers
           closed to the grasp's width keeping 1 mm (the pre-insertion pose 2 mm) from every other body, the links
           3 mm; for the outer fingers the swing to their park angle as well)
  pick     planned free move to 60 mm back along the approach, in along it at 0.05 m/s, closed to span - 12 mm
           (15 N per finger), 1.5 s; lift 60 mm at 0.05 m/s; the unit must stay in the hand (3 mm / 4 deg)
  carry    planned (the held unit carried rigidly at its joint angles, 4 mm), time-parametrised with a trapezoid on
           the tcp's path at the carry limits measured in the cell's grasp trials: a finger 0.25 m/s and 3 m/s2, the thumb 0.2 m/s and
           2 m/s2; the unit re-measured in the hand after the carry (it shifts 0.1-0.6 mm in the pads)
  finger   (a) taken by its pitch yoke (L2) at the chain's drawn ab/ad angle, dorsal side up; the base aligned 50 mm
           out along its slide (closed loop on its live pose); slid home at 10 mm/s at stiffness 0.5, every tick the
           hand's command recomputed from the base's live pose in the hand (the push along the slide through the
           grasp, whose centre lies on the pitch axis); held against the bay wall until the cell fastens it; then,
           still held, turned about its ab/ad axis to its park angle (the index -45 deg, the ring +45 deg, just
           inside the shell stops at -+45.69) or straightened (the middle, 0 deg), closed loop on the joint's angle;
           its tip on the rest deck; let go; the hand straight up
  thumb    (b) taken by its corner block (T2) with the hand level, from the wrist side; its base aligned 40 mm out
           along the wall's normal; brought in along the normal to the wall, its yoke on the fixture's pad; held
           until the cell fastens it (unheld it tips about the pad and lifts out of its pocket); turned about its side axis to 0 deg;
           let go; the hand withdrawn towards the wrist side
  guard    the base more than 4 mm behind the command along the insertion for 0.3 s, or a product joint of the held
           unit turning more than 1.5 deg from where it stood at the start of the insertion (the SO-101 arm guard):
           back off 15 mm, re-align, retry (twice) at a lower stiffness
  plate    (c)-(g) a palm part: the stack, the top, the hub and the cap: of the table's grasps whose closing axis
           passes within 2 mm of its centre of mass (held off it, a part hinges about the pads as they open and lands
           tilted), the one that reaches its release pose lowest, else the lowest of the others; the cover: the table's
           first grasp that reaches its release pose (the release heights tried per kind, lowest first: the
           stack 13-18 mm above home, the top 14-20 mm, the hub 1-3 mm, the cap 3.5-5 mm; the cover 0 mm: carried
           home); carried to 30 mm above that pose at the flat parts' limits
           (0.15 m/s, 1.5 m/s2), aligned (closed loop), a checked line down at 10 mm/s closed loop on its live pose,
           let go there (span + 4 mm) and the hand straight up; the part settles in its own guides and the cell
           centres it once it lies; if it is not fastened within 2 s, and always for the cap (a press fit), it is
           pushed home with the closed fingertips (a push pose found on its collision pieces: the footprint on its
           top as it lies, the fingers 1 mm from everything else at the touch and at the end of the push), at
           5 mm/s and stiffness 0.3, down to contact whatever the part's height, then until it is 0.25 mm above home,
           the watch reads 12 N or the push has gone 2 mm past it; then the command set to where the arm stands, held
           until fastened, the hand straight up. The cover is carried home and pressed onto its seat (the hand's
           command kept 0.3 mm below home) until it is fastened
A unit the arm cannot place is put back into its nest if it is in the hand (never let go in the air), else let go
where it rests (`placed_by: nobody`); no other fallback. A contact monitor logs every
contact above 0.5 N of the Panda with anything but the unit it holds or grasps (`unplanned_contacts`), and of the unit
in work with anything but its carrier (the palm base), the hand, its nest (while it is picked or put back) and the
fixture's supports (the rest deck, the thumb's pad) (`unit_contacts`); every touch of the Panda, whatever its force, on
anything but the unit in work (`robot_touches_any_force`). With the server (`server=`), every tick's readings of
the watch's four rules R1-R4 are logged (`flog`): the reference's force record that fixes the rules' numbers.
"""
import math
import os
import time

import numpy as np

from . import reach as RC
from .coupling import _inv, quat_wxyz, rotation_angle

GRIP_SQUEEZE_M = 0.012          # 2500 N/m x 12 mm / 2 = 15 N per finger (the precedent; the cell's grasp trials)
OPEN_EXTRA_M = 0.006            # the hand open to span + 6 mm on the approach (the reach check's opening)
RELEASE_EXTRA_M = 0.004         # ... and to span + 4 mm to let a fastened unit go (2 mm a side; at + 6 mm the opening
                                # fingers came within 1.3 mm of the parked ring finger's yoke as the middle one was let go)
PREGRASP_M = 0.06
LIFT_M = 0.06
PRE_M = {"FM": 0.050, "TM": 0.040}   # the pre-insertion pose: a finger 50 mm back along its slide, the thumb 40 mm
                                     # (a palm part: its release height + PLATE_ABOVE_M above home)
PRE_UP_M = 0.006                # ... reached from this far above it when the held unit cannot keep 4 mm there (a finger's
                                # chain stands 1.8-3 mm above the rest deck 50 mm out: the carry ends above, a line down)
INSERT_SPEED_M_S = 0.010
INSERT_STIFFNESS = 0.5
# per kind: the arm's stiffness on the way in, and how high above its floor the base is carried in and held (m). The
# finger: 0.5 (its guide holds it across; the arm gives). The thumb has no guide and its yoke passes 0.19 mm over the
# fixture's pad: at stiffness 0.5 the carried thumb sat 0.2-0.3 mm low and its yoke slid on the pad at up to 6.4 N
# (development runs), so it goes in at full stiffness 0.2 mm above its floor (the lock admits 0.5 mm) and drops onto the
# pad when the hand lets go
INSERT = {"FM": {"stiffness": INSERT_STIFFNESS, "lift_m": 0.0}, "TM": {"stiffness": 1.0, "lift_m": 0.0002}}
CARRY = {"FM": (0.25, 3.0), "TM": (0.20, 2.0)}
DEFAULT_CARRY = (0.15, 1.5)     # the flat parts (the cell's grasp trials)
FREE_MOVE = (0.5, 2.0)          # a free move of the empty hand: tcp speed and acceleration
APPROACH_SPEED_M_S = 0.05
SWING_DEG_S = 10.0
HOLD_S = 0.6                    # a digit held still at home this long and until the cell fastens it
LOCK_WAIT_S = 4.0
STALL_M = 0.004
STALL_TICKS = 30
GUARD_DEG = 1.5
RETRIES = 2
HELD_TOL_M, HELD_TOL_RAD = 0.003, math.radians(4.0)
R_EFF_M = 0.12                  # the held unit's reach about the tcp (a turn of the hand counts as R_EFF x angle of path)
DEFAULT_STEPS = (("seat_FI", "finger_module_1"), ("seat_FR", "finger_module_2"), ("seat_FM", "finger_module_3"),
                 ("seat_TM", "thumb_module"), ("seat_BS", "electronics_stack"), ("seat_PT", "upper_shell"),
                 ("seat_PC", "dorsal_cover"), ("seat_HB", "power_hub"), ("seat_CC", "press_cap"))
PLATES = ("BS", "PT", "PC", "HB", "CC")
# the release heights tried per kind (m above home, lowest first): the design's and the cell's (measured on the pieces:
# the stack 13-14 mm, the top 12 mm with the long-edge grasp (lower, the left fingertip comes within 1 mm of the
# middle finger's ab/ad link on the line down), the hub 1 mm, the cap 3.5 mm); the cover is carried home
RELEASE_M = {"BS": (0.013, 0.014, 0.015, 0.0155, 0.016, 0.018), "PT": (0.012, 0.013, 0.014, 0.016, 0.018, 0.020), "PC": (0.0,),
             "HB": (0.001, 0.0015, 0.002, 0.003), "CC": (0.0035, 0.004, 0.005)}
COM_OFFSET_M = 0.002            # a palm part's grasps whose closing axis passes this near its centre of mass come first ...
COM_FIRST = ("BS", "HB", "CC", "PT")  # ... for these kinds (the top: the table's row 28, its long-edge grasp 2.5 mm higher,
                                # through the centre of mass, released 14 mm above home; held lower on its curved edges it
                                # pivoted on one pad as the hand opened, 14-18 deg)
PLATE_ABOVE_M = 0.030           # the carry ends this far above the release pose; a checked line down
RELEASE_FINGER_CLEAR_M = {"BS": 0.0009}  # the fingertips' clearance at the release pose itself, where it is not the
                                # line's 1 mm: the stack's planned grasp (through its centre of mass) holds its left
                                # fingertip 0.93 mm above the top of the bay wall beside the stack at its release pose
PLATE_DESCENT_M_S = 0.010
PLATE_HOLD_S = 0.3              # held still at the release pose before the hand opens
PLATE_WAIT_S = 2.0              # after the release: this long for the lock before a push
ALWAYS_PUSH = ("CC",)           # the press fit cannot go home by itself (3 N of fit friction against its 0.1 N)
PUSH_SPEED_M_S = 0.005
PUSH_STIFFNESS = 0.3
PUSH_TO_M = 0.00025             # pushed until the part is this far above home ...
PUSH_PAST_M = 0.002             # ... or the fingertips have gone this far below where the part's home would put them
PUSH_MAX_N = 12.0               # ... or the watch reads this much on the part (R2), half the rule's number
PUSH_FOOT_M = (0.0085, 0.0143)  # the closed fingertips' footprint (half sizes along the tcp's x and y) ...
PUSH_TIP_M = 0.0089             # ... and how far the tips stand beyond the tcp (measured on the cell's Panda model)
PUSH_TRUE_TOL_M = 0.00005       # under the footprint the part's TRUE top may stand at most this far above its collision
                                # top (the drawn meshes' own deflection), else the pushing tips would sit inside it
SEAT_PRESS_M = 0.0003           # the cover, carried home: once it lies within the lock's window the hand's command is set
                                # this far below home and kept, so that the cover is pressed onto its seat (the cell fastens
                                # a seated part only while its seat carries it; held in the pads above the seat it is not)
PARK_DEG = {"seat_FI": -45.0, "seat_FR": 45.0, "seat_FM": 0.0}
THUMB_SIDE_DEG = 0.0
ENGAGED = ("ENGAGED", "HOLDING", "LOCKED", "LOCKED_OFF")
FASTENED = ("LOCKED", "LOCKED_OFF")


def _T(p, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _rot_about(axis, ang):
    from scipy.spatial.transform import Rotation
    a = np.asarray(axis, float)
    return Rotation.from_rotvec(a / np.linalg.norm(a) * float(ang)).as_matrix()


def _turn_about(T, point, axis, ang):
    """T turned by ang about the world line through point along axis"""
    R = _rot_about(axis, ang)
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = point + R @ (T[:3, 3] - point)
    return out


def trapezoid(length, v, a):
    """(total time, s(t)) of a trapezoid speed profile over `length` with top speed v and acceleration a"""
    length = max(0.0, float(length))
    if length <= 1e-12:
        return 0.0, (lambda t: 0.0)
    ta = v / a
    if a * ta * ta > length:
        ta = math.sqrt(length / a)
        vpk, tc = a * ta, 0.0
    else:
        vpk, tc = v, (length - a * ta * ta) / v
    T = 2 * ta + tc

    def s_of(t):
        if t <= 0:
            return 0.0
        if t < ta:
            return 0.5 * a * t * t
        if t < ta + tc:
            return 0.5 * a * ta * ta + vpk * (t - ta)
        if t < T:
            return length - 0.5 * a * (T - t) ** 2
        return length
    return T, s_of


class Assembler(object):
    def __init__(self, cell, plan=None, log=print, server=None, monitor=True, wall_cap_s=None, snap=None):
        from .motion import Planner
        from .panda_ik import PandaKin
        self.cell, self.m, self.d, self.ct, self.sm = cell, cell.m, cell.d, cell.ctrl, cell.sm
        self.kit, self.log, self.srv = cell.kit, log, server
        self.plan = dict(plan or {})
        self.pl = Planner(cell, seed=int(self.plan.get("planner_seed", 0)))
        self.kin = PandaKin(model=self.pl.m)
        self.ck = RC.Checker(cell)
        self.max_width = float(self.ct.max_width)
        self.q_ready = np.array(cell.cfg["robot_home_q"], float)
        self.plan_grasp = {st["unit"]: st.get("grasp") for st in self.plan.get("steps", []) if st.get("grasp")}
        self._maps = {}                 # kind -> the top height map of its collision pieces (unit frame), for the push
        self.plan_push = {st["unit"]: st.get("push") for st in self.plan.get("steps", []) if st.get("push")}
        self.park = dict(PARK_DEG, **(self.plan.get("park_deg") or {}))
        self.thumb_side = float(self.plan.get("thumb_side_deg", THUMB_SIDE_DEG))
        self.work = {"unit": None, "slot": None, "phase": "start", "held": None, "carriers": set()}
        self.contacts, self.unit_contacts = [], []
        self.robot_touch = {}           # (phase, slot, body, body) -> largest normal force of any Panda touch (N)
        self.panda = set(b for b in range(self.m.nbody) if self.m.body(b).name.startswith("panda/"))
        self.t0_wall = time.time()
        self.wall_cap = None if wall_cap_s is None else float(wall_cap_s)
        self.snap = snap
        self.ik_fail = 0
        self.release_m = None
        self.paths = []                 # (slot, phase, waypoints) of every planned move: 2 = the straight segment
        self.flog = {"t": [], "slot": [], "phase": [], "R1_N": [], "R1_on": [], "R2_N": [], "R2a_N": [], "R2b_N": [], "R2_pair": [],
                     "R3_Nm": [], "R3_joint": [], "R4_m_s": [], "pad_N": []}
        self.jlog = []                  # (t, slot, phase, {joint: deg}) of the unit in work, every 5 ticks
        self.support = {}               # (slot, phase, unit geom's link, other geom) -> largest normal force (N): the unit in
                                        # work on the fixture's supports and the palm base while it goes in
        self.support_names = ("fixture_deck", "fixture_thumb_pad")
        if monitor:
            cell.hooks.append(self._monitor)
        if server is not None:
            server.reading_hooks.append(self._force_hook)

    # ------------------------------------------------------------------ state
    def tcp_T(self):
        p, R = self.ct.tcp_actual()
        return _T(p, R)

    def unit_T(self, u):
        return self.cell.unit_T(u)

    def G_act(self, u):
        """the unit's root link in the tcp frame, now"""
        return _inv(self.tcp_T()) @ self.unit_T(u)

    def joints_deg(self, u):
        return {jn: math.degrees(v) for jn, v in self.cell.joint_q_all()[u].items()}

    def out_of_time(self):
        return self.wall_cap is not None and time.time() - self.t0_wall > self.wall_cap

    def _monitor(self):
        """every tick: contacts above 0.5 N of the Panda with anything but the unit in work (held or being grasped),
        and of the unit in work with anything but its carrier (the palm base), the hand, its nest's tray while it is
        picked and the fixture's supports"""
        m, d = self.m, self.d
        w = self.work
        u = w.get("unit")
        ubs = set(self.cell.bodies_of[u]) if u else set()
        base = m.body(self.cell.base_body).id
        f6 = np.zeros(6)
        cell = self.cell
        if u and cell.n_tick % 5 == 0:
            self.jlog.append((round(float(d.time), 2), w.get("slot"), w.get("phase"), {k: round(v, 3) for k, v in self.joints_deg(u).items()}))
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            r1, r2 = b1 in self.panda, b2 in self.panda
            if r1 and r2:
                continue
            kind = None
            if r1 or r2:
                other = b2 if r1 else b1
                if other in ubs:
                    continue
                kind = "robot"
            elif b1 in ubs or b2 in ubs:
                if b1 in ubs and b2 in ubs:
                    continue
                other = b2 if b1 in ubs else b1
                og = c.geom2 if b1 in ubs else c.geom1
                nm = m.body(other).name
                gnm = m.geom(og).name or ""
                if other == base or other in w.get("carriers", ()) or any(gnm.startswith(p) for p in self.support_names):
                    if w["phase"] in ("insert", "hold", "swing", "side", "release", "align", "settle", "push"):
                        cell.mj.mj_contactForce(m, d, i, f6)
                        ug = c.geom1 if b1 in ubs else c.geom2
                        k = (w.get("slot"), w["phase"], m.body(int(m.geom_bodyid[ug])).name.split("__")[-1],
                             "palm_base" if other == base else (cell.body_unit.get(other) or gnm))
                        self.support[k] = max(self.support.get(k, 0.0), round(float(abs(f6[0])), 3))
                    continue
                if w["phase"] in ("plan", "to-pick", "pick", "close", "lift", "put back", "let go") and nm in ("digit_tray", "parts_tray"):
                    continue
                kind = "unit"
            else:
                continue
            cell.mj.mj_contactForce(m, d, i, f6)
            if kind == "robot":
                # every touch of the Panda on anything but the unit in work, whatever its force (the gate's count
                # uses the precedent's 0.5 N; this list shows whether there was any contact at all)
                kt = (w["phase"], w.get("slot"), m.body(b1).name, m.body(b2).name)
                self.robot_touch[kt] = max(self.robot_touch.get(kt, 0.0), round(float(abs(f6[0])), 4))
            if abs(f6[0]) < 0.5:
                continue
            n1, n2 = m.body(b1).name, m.body(b2).name
            lst = self.contacts if kind == "robot" else self.unit_contacts
            key = (w["phase"], w.get("slot"), n1, n2)
            if lst and lst[-1]["key"] == key:
                lst[-1]["force_N"] = max(lst[-1]["force_N"], round(float(abs(f6[0])), 2))
                lst[-1]["t1"] = round(float(d.time), 2)
                continue
            lst.append({"key": key, "t": round(float(d.time), 2), "t1": round(float(d.time), 2), "phase": w["phase"], "slot": w.get("slot"),
                        "bodies": [n1, n2], "geoms": [m.geom(c.geom1).name, m.geom(c.geom2).name], "force_N": round(float(abs(f6[0])), 2)})

    def _force_hook(self, srv):
        r = srv.readings
        f = self.flog
        f["t"].append(round(float(srv.data.time), 3))
        f["slot"].append(self.work.get("slot") or "-")
        f["phase"].append(self.work.get("phase") or "-")
        for k in ("R1_N", "R2_N", "R2a_N", "R2b_N", "R3_Nm", "R4_m_s", "pad_N"):
            f[k].append(float(r.get(k, 0.0)))
        p = r.get("R1_pair")
        f["R1_on"].append(p[1] if p else None)
        p = r.get("R2_pair")
        f["R2_pair"].append(list(p) if p else None)
        f["R3_joint"].append(r.get("R3_joint"))

    # ------------------------------------------------------------------ motion primitives (joint mode, IK per tick)
    def tick(self, n=1):
        for _ in range(n):
            self.cell.tick()

    def settle(self, tol=0.002, max_ticks=300):
        ct = self.ct
        for _ in range(max_ticks):
            self.cell.tick()
            if np.max(np.abs(ct.q() - ct.q_cmd)) < tol and np.max(np.abs(ct.dq())) < 0.02:
                return True
        return False

    def ik(self, T, q_seed, margin=0.01):
        return self.kin.solve_one(T[:3, 3], T[:3, :3], q_seed, q_seed, margin=margin)

    def follow(self, path, v_max, a_max, settle=True):
        """a planned joint path, densified (0.01 rad), time-parametrised by a trapezoid on the tcp's path length (plus
        R_EFF x the turn) at v_max / a_max and the joints' 1.4 rad/s, commanded tick by tick in joint mode"""
        ct = self.ct
        ct.set_mode("joint")
        dense = [np.asarray(path[0], float)]
        for qb in path[1:]:
            qa = dense[-1]
            qb = np.asarray(qb, float)
            n = max(1, int(math.ceil(float(np.max(np.abs(qb - qa))) / 0.01)))
            for k in range(1, n + 1):
                dense.append(qa + (qb - qa) * (k / n))
        Ts = [self.pl.tcp_T(q) for q in dense]
        s = [0.0]
        dqds = 0.0
        for i in range(1, len(dense)):
            ds = float(np.linalg.norm(Ts[i][:3, 3] - Ts[i - 1][:3, 3])) + R_EFF_M * rotation_angle(Ts[i - 1][:3, :3].T @ Ts[i][:3, :3])
            s.append(s[-1] + ds)
            if ds > 1e-9:
                dqds = max(dqds, float(np.max(np.abs(dense[i] - dense[i - 1]))) / ds)
        s = np.array(s)
        v = float(v_max)
        if dqds > 0:
            v = min(v, 1.4 / dqds)
        a = float(a_max)
        Ttot, s_of = trapezoid(s[-1], v, a)
        n = max(1, int(math.ceil(Ttot / self.cell.tick_s)))
        for k in range(1, n + 1):
            sk = s_of(k * self.cell.tick_s) if k < n else s[-1]
            i = int(np.searchsorted(s, sk, side="right")) - 1
            i = max(0, min(i, len(dense) - 2))
            f = (sk - s[i]) / (s[i + 1] - s[i]) if s[i + 1] > s[i] else 1.0
            q = dense[i] + (dense[i + 1] - dense[i]) * min(1.0, max(0.0, f))
            ct.command_joint(q)
            self.cell.tick()
        ct.command_joint(dense[-1])
        if settle:
            self.settle()
        return True

    def go(self, q, held=None, allow=(), width=None, v=None, a=None):
        """planned free-space move to q (the held unit carried as held now); via the ready posture if no direct path"""
        vm, am = (v, a) if v is not None else FREE_MOVE
        self.pl.sync(held=held, allow=allow, width=width)
        q = np.asarray(q, float)
        q0 = self.ct.q_cmd.copy()
        path = self.pl.plan(q0, q)
        if path is None and np.max(np.abs(q - self.q_ready)) > 1e-3:
            p1 = self.pl.plan(q0, self.q_ready)
            if p1 is not None:
                p2 = self.pl.plan(self.q_ready, q)
                if p2 is not None:
                    path = p1 + p2[1:]
        if path is None:
            return False
        self.paths.append((self.work.get("slot"), self.work.get("phase"), len(path)))
        return self.follow(path, vm, am)

    def line_poses(self, T0, T1, step_m=0.002, step_rad=math.radians(1.0)):
        from scipy.spatial.transform import Rotation, Slerp
        n = max(1, int(math.ceil(max(np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / step_m, rotation_angle(T0[:3, :3].T @ T1[:3, :3]) / step_rad))))
        sl = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))
        out = []
        for k in range(0, n + 1):
            s_ = k / n
            out.append(_T((1 - s_) * T0[:3, 3] + s_ * T1[:3, 3], sl([s_]).as_matrix()[0]))
        return out

    def poses_qs(self, poses, q0, jump=0.3):
        qs, q = [], np.asarray(q0, float)
        for T in poses:
            q2, ok = self.ik(T, q, margin=0.0)
            if not ok or np.max(np.abs(q2 - q)) > jump:
                return None
            qs.append(q2)
            q = q2
        return qs

    def move_poses(self, poses, v, a=None, ang_speed=math.radians(20.0), settle_ticks=10):
        """a tcp path through the given poses (a line or an arc), time-parametrised by a trapezoid (or constant speed)
        on its length plus R_EFF x its turn, IK per tick from the last commanded posture, joint mode"""
        ct = self.ct
        ct.set_mode("joint")
        L = [0.0]
        for i in range(1, len(poses)):
            L.append(L[-1] + float(np.linalg.norm(poses[i][:3, 3] - poses[i - 1][:3, 3]))
                     + R_EFF_M * rotation_angle(poses[i - 1][:3, :3].T @ poses[i][:3, :3]))
        L = np.array(L)
        ang = sum(rotation_angle(poses[i - 1][:3, :3].T @ poses[i][:3, :3]) for i in range(1, len(poses)))
        vv = min(float(v), (L[-1] / max(1e-9, ang / ang_speed)) if ang > 1e-9 else float(v))
        if a:
            Ttot, s_of = trapezoid(L[-1], vv, float(a))
        else:
            Ttot = L[-1] / vv if vv > 0 else 0.0

            def s_of(t):
                return min(L[-1], vv * t)
        n = max(1, int(math.ceil(Ttot / self.cell.tick_s)))
        from scipy.spatial.transform import Rotation, Slerp
        q = ct.q_cmd.copy()
        for k in range(1, n + 1):
            sk = s_of(k * self.cell.tick_s) if k < n else L[-1]
            i = int(np.searchsorted(L, sk, side="right")) - 1
            i = max(0, min(i, len(poses) - 2))
            f = (sk - L[i]) / (L[i + 1] - L[i]) if L[i + 1] > L[i] else 1.0
            f = min(1.0, max(0.0, f))
            R = Slerp([0, 1], Rotation.from_matrix([poses[i][:3, :3], poses[i + 1][:3, :3]]))([f]).as_matrix()[0]
            T = _T((1 - f) * poses[i][:3, 3] + f * poses[i + 1][:3, 3], R)
            q2, ok = self.ik(T, q)
            if ok:
                q = q2
                ct.command_joint(q)
            else:
                self.ik_fail += 1
            self.cell.tick()
        self.tick(settle_ticks)
        return True

    def line(self, T1, v=APPROACH_SPEED_M_S, a=0.5, held=None, allow=(), width=None, check=True, finger_clear=0.002, link_clear=0.004,
             held_clear=None, settle_ticks=10):
        """a checked Cartesian line from the actual tcp to T1 (the reach check's margins: the arm and the hand 4 mm,
        the fingers 2 mm; a free move plans with 8 mm)"""
        T0 = self.tcp_T()
        poses = self.line_poses(T0, T1)
        if check:
            qs = self.poses_qs(poses[::2] + [poses[-1]], self.ct.q_cmd)
            if qs is None:
                return False, "line IK"
            self.pl.sync(held=held, allow=allow, width=width)
            kw = {}
            if finger_clear is not None:
                kw["finger_clear"] = finger_clear
            if link_clear is not None:
                kw["link_clear"] = link_clear
            if held_clear is not None:
                kw["held_clear"] = held_clear
            if not self.pl.line_free(qs, **kw):
                return False, "line blocked (%s)" % self.pl.why
        self.move_poses(poses, v, a, settle_ticks=settle_ticks)
        return True, ""

    def grip(self, width, settle_ticks=60):
        self.ct.command_width(float(min(self.max_width, max(0.0, width))))
        last = None
        for i in range(400):
            self.cell.tick()
            w = self.ct.width()
            if i > 20 and abs(self.ct.width_goal - self.ct.width_cmd) < 1e-6 and last is not None and abs(w - last) < 2e-5:
                break
            last = w
        self.tick(settle_ticks)
        return self.ct.width()

    # ------------------------------------------------------------------ targets
    def slot(self, sid):
        return self.sm.by_key[sid]

    def home_T(self, sid):
        """the unit's root link at its design pose (world), the palm base as it is now"""
        return self.cell.palm_T() @ self.slot(sid)["Tdes"]

    def ins_world(self, sid):
        return self.cell.palm_T()[:3, :3] @ self.slot(sid)["ins"]

    def path_pose(self, sid, back):
        """the unit's root link `back` (m) out of home along its path (against the insertion)"""
        T = self.home_T(sid).copy()
        T[:3, 3] -= self.ins_world(sid) * back
        return T

    def kind(self, u):
        return self.kit.kind_of(u)

    # ------------------------------------------------------------------ grasp candidates and checks
    def candidates(self, u):
        """[(row index, sign, tcp pose world, span m)] of the grasp table on the unit as it rests now, the bench
        plan's grasp first"""
        kind = self.kind(u)
        tab = RC.table()["kinds"][kind]
        out = []
        hint = self.plan_grasp.get(u)
        rows = list(enumerate(tab["grasps"]))
        if hint is not None:
            rows = [r for r in rows if r[0] == int(hint["row"])] + [r for r in rows if r[0] != int(hint["row"])]
        for i, g in rows:
            c, uu, a = RC.grasp_world(self.cell, u, g)
            for sg, T in zip((1, -1), RC.tcp_frames(c, uu, a)):
                if hint is not None and i == int(hint["row"]) and sg != int(hint.get("sign", sg)):
                    continue
                out.append((i, sg, T, float(g["span_mm"]) * 1e-3))
        return out

    def pick_check(self, u, T, span):
        kind = self.kind(u)
        self.ck.sync()
        ok, q, why = self.ck.grasp_ok(T, span, back_mm=(10.0, 30.0, 60.0), grip_bodies=set(self.cell.bodies_of[u]),
                                      open_extra_m=RC.open_extra_m(kind))
        return (q if ok else None), why

    def place_check(self, sid, u, G, span, q_seed, step=0.002, finger_path=0.001, finger_pre=0.002, link=0.003):
        """(q_pre, None) when the hand, holding the unit's root link at G (its pose in the tcp frame), reaches the
        pre-insertion pose and every pose of the path to home on one IK branch with the fingers closed to the grasp's
        width (fingers 1 mm, at the pre-insertion pose 2 mm; links 3 mm from everything but the unit), every branch
        at the pre-insertion pose tried; else (None, why)"""
        kind = self.kind(u)
        if kind in PLATES:
            return self.plate_place_check(sid, u, G, span, q_seed, RELEASE_M[kind][0], step=step, finger_path=finger_path,
                                          finger_pre=finger_pre, link=link)
        pre = PRE_M[kind]
        T_pre = self.path_pose(sid, pre) @ _inv(G)
        sols = self.kin.candidates(T_pre, q_seed)
        if not sols:
            return None, "pre-insertion IK"
        xs = list(np.arange(pre, -1e-9, -step))
        if xs[-1] > 1e-9:
            xs.append(0.0)
        why = ""
        for _sc, q0, _mrg in sols[:6]:
            q, okb = q0, True
            for i, back in enumerate(xs):
                T_b = self.path_pose(sid, back)
                if i == 0:
                    qx, ok = q0, True
                else:
                    qx, ok = self.ik(T_b @ _inv(G), q, margin=0.0)
                    ok = ok and np.max(np.abs(qx - q)) < 0.3
                if not ok:
                    why, okb = "IK at %.0f mm" % (back * 1e3), False
                    break
                self.pl.sync(allow=(u,), poses={u: T_b}, width=span)
                hit = self.pl.hits(qx, finger_clear=finger_path if i else finger_pre, link_clear=link)
                if hit:
                    why, okb = "hits %s at %.0f mm" % (sorted(set((h[0], h[1]) for h in hit))[:3], back * 1e3), False
                    break
                q = qx
            if not okb:
                continue
            if kind == "FM" and sid in self.park:
                ok, why2 = self.swing_check(sid, u, G, q, span, finger=finger_path, link=link)
                if not ok:
                    why = "swing: " + why2
                    continue
            return q0, None
        return None, why or "insertion"

    def swing_check(self, sid, u, G, q_home, span, step_deg=3.0, finger=0.001, link=0.003):
        """the hand holding the fastened finger at home turned about its ab/ad axis to the park angle"""
        T_home = self.path_pose(sid, 0.0)
        j = self.cell.joint_ids[u]["abad"]
        anchor_u, axis_u = self._joint_axis_in_unit(u, "abad")
        p = T_home[:3, :3] @ anchor_u + T_home[:3, 3]
        ax = T_home[:3, :3] @ axis_u
        a0 = float(self.d.qpos[self.m.jnt_qposadr[j]])
        a1 = math.radians(self.park[sid])
        n = max(1, int(math.ceil(abs(math.degrees(a1 - a0)) / step_deg)))
        T_tcp0 = T_home @ _inv(G)
        q = q_home
        for k in range(1, n + 1):
            ang = (a1 - a0) * k / n
            T = _turn_about(T_tcp0, p, ax, ang)
            q2, ok = self.ik(T, q, margin=0.0)
            if not ok or np.max(np.abs(q2 - q)) > 0.3:
                return False, "IK at %.0f deg" % math.degrees(a0 + ang)
            self.pl.sync(allow=(u,), poses={u: T_home}, joints={u: {"abad": a0 + ang}}, width=span)
            hit = self.pl.hits(q2, finger_clear=finger, link_clear=link)
            if hit:
                return False, "hits %s at %.0f deg" % (sorted(set((h[0], h[1]) for h in hit))[:3], math.degrees(a0 + ang))
            q = q2
        return True, ""

    def _joint_axis_in_unit(self, u, jn):
        """(anchor, axis) of a unit's joint in its root link's frame (the cell's model, now)"""
        j = self.cell.joint_ids[u][jn]
        T_root = self.unit_T(u)
        Ri = T_root[:3, :3].T
        return Ri @ (self.d.xanchor[j] - T_root[:3, 3]), Ri @ self.d.xaxis[j]

    def feasible_grasp(self, sid, u):
        """the first grasp (the plan's first) that passes the pick check where the unit rests now and the place check
        at its mate: (row, sign, T_grasp, span, q_pick, q_pre, G) or (None, rejections); a palm part: of its release
        heights the lowest at which a candidate passes (the plan's height first), the candidates in the table's order
        (self.release_m holds the height taken)"""
        T_u = self.unit_T(u)
        rejected = {}
        if self.kind(u) in PLATES:
            return self._feasible_plate(sid, u, T_u)
        for i, sg, T, span in self.candidates(u):
            q_pick, why = self.pick_check(u, T, span)
            if q_pick is None:
                rejected["%d%+d" % (i, sg)] = "pick: %s" % (why,)
                continue
            G = _inv(T) @ T_u
            q_pre, why = self.place_check(sid, u, G, span, q_pick)
            if q_pre is None:
                rejected["%d%+d" % (i, sg)] = "place: %s" % why
                continue
            return (i, sg, T, span, q_pick, q_pre, G), rejected
        return None, rejected

    # ------------------------------------------------------------------ one step
    def place_unit(self, sid, u):
        sm = self.sm
        rec = {"slot": sid, "unit": u, "t0_s": round(float(self.d.time), 2), "placed_by": "nobody", "attempts": []}
        self.work.update({"unit": u, "slot": sid, "phase": "plan", "held": None, "carriers": self._carrier_bodies(sid)})
        t_wall = time.time()
        for round_ in range(2):
            if self.out_of_time():
                rec["why"] = "wall-clock cap"
                break
            feas, rejected = self.feasible_grasp(sid, u)
            if feas is None:
                rec["why"] = "no feasible grasp (%s)" % (sorted(set(str(v) for v in rejected.values()))[:4],)
                self.log("  %s %s: %s" % (sid, u, rec["why"]))
                break
            i, sg, T_g, span, q_pick, q_pre, G_pred = feas
            att = {"round": round_, "grasp_row": i, "sign": sg, "span_mm": round(span * 1e3, 2), "rejected_before": len(rejected),
                   "rejected": dict(list(rejected.items())[:12]), "plan_s": round(time.time() - t_wall, 1)}
            if self.kind(u) in PLATES:
                att["release_m"] = float(self.release_m)
            rec["attempts"].append(att)
            self.log("  %s %s: grasp row %d sign %+d (%d rejected), plan %.0f s" % (sid, u, i, sg, len(rejected), att["plan_s"]))
            ok, why = self.pick(u, T_g, span, q_pick, att)
            if not ok:
                att["result"] = why
                self.log("  %s %s: pick failed (%s)" % (sid, u, why))
                if att.get("lifted"):
                    # in the hand above its nest (it slipped): put back, then taken again (never let go in the air)
                    att["put_back"] = self.put_back(u, span)
                    if att["put_back"] != "put back":
                        rec["why"] = "held: could not be put back (%s)" % att["put_back"]
                        break
                else:
                    self.let_go(span)
                continue
            if self.snap:
                self.snap("%s_picked" % sid, u)
            ok, why = self.insert(sid, u, span, att, q_branch=q_pre)
            att["result"] = why
            if ok:
                rec["placed_by"] = "panda"
                break
            self.log("  %s %s: insertion failed (%s)" % (sid, u, why))
            if "put_back" in att:
                if att["put_back"] != "put back":
                    rec["why"] = "held: could not be put back (%s)" % att["put_back"]
            elif self.work["held"] is not None:
                self.let_go(span)              # it rests in its pocket: let go there
            break
        st = sm.rec[sid]
        rec.update({"state": sm.state[sid], "fastened": st["fastened"], "achieved": st["achieved"], "pos_err_mm": st["pos_err_mm"],
                    "ang_err_deg": st["ang_err_deg"], "why_off": st["why_off"], "capture": st.get("capture"),
                    "time_to_lock_s": st.get("time_to_lock_s"), "t_lock_s": st.get("t_s"), "blocked_by": st.get("blocked_by"),
                    "joints_after_deg": {k: round(v, 3) for k, v in self.joints_deg(u).items()},
                    "t1_s": round(float(self.d.time), 2), "wall_s": round(time.time() - t_wall, 1)})
        self.log("  %-8s %-16s -> %s (%s) sim %.1f s wall %.0f s" % (sid, u, sm.state[sid], rec["attempts"][-1].get("result") if rec["attempts"] else "-",
                                                                   rec["t1_s"] - rec["t0_s"], rec["wall_s"]))
        return rec

    def let_go(self, span):
        """a unit the arm could not place stays where it is: open, back out along the approach, ready posture"""
        self.work.update({"phase": "let go", "held": None})
        self.grip(min(self.max_width, span + OPEN_EXTRA_M + 0.01), settle_ticks=10)
        T_out = self.tcp_T()
        T_out[:3, 3] -= T_out[:3, 2] * 0.06
        self.line(T_out, check=False)
        if not self.go(self.q_ready):
            self.ct.command_joint(self.q_ready)
            self.settle(max_ticks=400)

    def pick(self, u, T_g, span, q_pick, att):
        wo = min(self.max_width, span + OPEN_EXTRA_M)
        self.work.update({"phase": "to-pick", "held": None})
        self.grip(wo, settle_ticks=5)
        T_pg = T_g.copy()
        T_pg[:3, 3] -= T_g[:3, 2] * PREGRASP_M
        q_pg, ok = self.ik(T_pg, q_pick, margin=0.0)
        if not ok:
            return False, "pre-grasp IK"
        if not self.go(q_pg, width=wo):
            return False, "no path to the pre-grasp (%s)" % self.pl.why
        self.work["phase"] = "pick"
        ok, why = self.line(T_g, v=APPROACH_SPEED_M_S, allow=(u,), width=wo, check=True, settle_ticks=20)
        if not ok:
            return False, "approach: " + why
        self.work["phase"] = "close"
        self.ct.command_width(max(0.0, span - GRIP_SQUEEZE_M))
        self.tick(150)
        G0 = self.G_act(u)
        q0 = self.joints_deg(u)
        self.work.update({"phase": "lift", "held": u})
        T_up = self.tcp_T()
        T_up[2, 3] += LIFT_M
        ok, why = self.line(T_up, v=APPROACH_SPEED_M_S, held=(u, G0), check=True)
        if not ok:
            self.line(T_up, v=APPROACH_SPEED_M_S, check=False)
        self.tick(50)
        self.lifted = {"unit": u, "q": self.ct.q_cmd.copy(), "T_tcp": self.tcp_T()}
        G1 = self.G_act(u)
        dp = float(np.linalg.norm(G1[:3, 3] - G0[:3, 3]))
        da = rotation_angle(G1[:3, :3].T @ G0[:3, :3])
        q1 = self.joints_deg(u)
        att["lift"] = {"slip_mm": round(dp * 1e3, 3), "slip_deg": round(math.degrees(da), 3),
                       "joint_turn_deg": {k: round(q1[k] - q0[k], 3) for k in q0}, "width_mm": round(self.ct.width() * 1e3, 2),
                       "held": self.srv.held if self.srv is not None else None}
        att["lifted"] = True
        if dp > HELD_TOL_M or da > HELD_TOL_RAD or self.ct.width() < 0.002:
            return False, "the unit moved in the hand (%.1f mm, %.1f deg)" % (dp * 1e3, math.degrees(da))
        return True, ""

    def put_back(self, u, span):
        """a held unit that could not be placed goes back where it was picked (planned to the lifted posture, a line
        down by the lift, let go, the hand up); never let go in the air"""
        lf = getattr(self, "lifted", None)
        if lf is None or lf["unit"] != u:
            return "nothing to put back"
        self.work["phase"] = "put back"
        G = self.G_act(u)
        v, a = CARRY.get(self.kind(u), DEFAULT_CARRY)
        if not self.go(lf["q"], held=(u, G), width=span, v=v, a=a):
            return "no path back (%s)" % self.pl.why
        T_dn = self.tcp_T()
        T_dn[2, 3] -= LIFT_M - 0.001
        self.line(T_dn, v=0.03, check=False)
        self.work.update({"phase": "let go", "held": None})
        self.grip(min(self.max_width, span + OPEN_EXTRA_M), settle_ticks=20)
        T_up = self.tcp_T()
        T_up[:3, 3] -= T_up[:3, 2] * PREGRASP_M
        self.line(T_up, v=0.05, check=False)
        return "put back"

    def align(self, sid, u, back, iters=5, tol_m=0.0002, tol_rad=math.radians(0.2), lift=0.0):
        """closed loop at a pose on the insertion path: the root link `back` short of home (`lift` m higher), its pose
        error read from the privileged state and corrected through the hand; returns the remaining (mm, deg)"""
        T_t = self.path_pose(sid, back)
        T_t[2, 3] += lift
        for _ in range(iters):
            E = _inv(T_t) @ self.unit_T(u)
            e_p, e_a = float(np.linalg.norm(E[:3, 3])), rotation_angle(E[:3, :3])
            if e_p < tol_m and e_a < tol_rad:
                break
            T_cmd = T_t @ _inv(self.G_act(u))
            self.move_poses(self.line_poses(self.tcp_T(), T_cmd), 0.01, None, ang_speed=math.radians(5.0), settle_ticks=30)
        E = _inv(T_t) @ self.unit_T(u)
        return round(float(np.linalg.norm(E[:3, 3])) * 1e3, 3), round(math.degrees(rotation_angle(E[:3, :3])), 3)

    def _lag(self, sid, u, back):
        """how far the root link is behind the command along the insertion (m)"""
        T_u = self.path_pose(sid, back)
        return float((T_u[:3, 3] - self.unit_T(u)[:3, 3]) @ self.ins_world(sid))

    def _cmd_unit_at(self, sid, u, back, q, lift=0.0):
        """command the hand so that the root link, as held now, goes to its path pose `back` out (`lift` m higher);
        IK from q"""
        T_t = self.path_pose(sid, back)
        T_t[2, 3] += lift
        T_cmd = T_t @ _inv(self.G_act(u))
        q2, ok = self.ik(T_cmd, q)
        if ok:
            self.ct.command_joint(q2)
            return q2
        self.ik_fail += 1
        return q

    def insert(self, sid, u, span, att, q_branch=None):
        """carry to the pre-insertion pose, align, in along the path closed loop on the live pose, hold until fastened,
        swing / straighten (fingers) or turn the side joint (thumb), let go, back out"""
        cell, ct, sm = self.cell, self.ct, self.sm
        kind = self.kind(u)
        if kind in PLATES:
            return self.place_plate(sid, u, span, att, q_branch=q_branch)
        pre = PRE_M[kind]
        G = self.G_act(u)
        T_t = self.path_pose(sid, pre) @ _inv(G)
        sols = self.kin.candidates(T_t, ct.q_cmd)
        if q_branch is not None:
            q_v, ok_v = self.ik(T_t, q_branch, margin=0.0)
            if ok_v:
                sols = [(0.0, q_v, 0.0)] + [s for s in sols if np.max(np.abs(s[1] - q_v)) > 1e-3]
        self.work["phase"] = "carry"
        v, a = CARRY.get(kind, DEFAULT_CARRY)
        moved, via = False, None
        for _sc, q_pre, _m in sols[:6]:
            if self.go(q_pre, held=(u, G), width=span, v=v, a=a):
                moved = True
                break
        if not moved:
            # the goal itself inside the held unit's 4 mm (a support under the chain): the carry ends PRE_UP_M above
            # it, then a line down (the held unit no deeper than 0.5 mm into the clearance, the robot's margins)
            T_hi = T_t.copy()
            T_hi[2, 3] += PRE_UP_M
            q_hi, ok = self.ik(T_hi, sols[0][1] if sols else ct.q_cmd, margin=0.0)
            if ok and self.go(q_hi, held=(u, G), width=span, v=v, a=a):
                ok2, why2 = self.line(T_t, v=0.01, a=0.2, held=(u, G), width=span, finger_clear=0.001, link_clear=0.003,
                                      held_clear=0.0005)
                via = "from %.0f mm above" % (PRE_UP_M * 1e3) if ok2 else None
                moved = ok2
                if not ok2:
                    self.pl.why = "descent: " + why2
        if not moved:
            att["put_back"] = self.put_back(u, span)
            return False, "no path to the pre-insertion (%s)" % self.pl.why
        att["pre_via"] = via
        self.tick(30)
        G_after = self.G_act(u)
        att["carry_shift"] = {"mm": round(float(np.linalg.norm(G_after[:3, 3] - G[:3, 3])) * 1e3, 3),
                              "deg": round(math.degrees(rotation_angle(G[:3, :3].T @ G_after[:3, :3])), 3)}
        self.work["phase"] = "align"
        ins = INSERT.get(kind, INSERT["FM"])
        att["align_at_pre"] = self.align(sid, u, pre, lift=float(ins["lift_m"]))
        if self.snap:
            self.snap("%s_pre" % sid, u)
        self.work["phase"] = "insert"
        ins = INSERT.get(kind, INSERT["FM"])
        stiff, lift = float(ins["stiffness"]), float(ins["lift_m"])
        back = pre
        end = 0.0
        q = ct.q_cmd.copy()
        for attempt in range(RETRIES + 1):
            ct.set_stiffness(stiff)
            ct.set_mode("joint")
            j0 = self.joints_deg(u)
            stall, guard = 0, None
            for _ in range(int((back - end + 0.02) / (INSERT_SPEED_M_S * cell.tick_s)) + 100):
                back = max(end, back - INSERT_SPEED_M_S * cell.tick_s)
                q = self._cmd_unit_at(sid, u, back, q, lift)
                cell.tick()
                lag = self._lag(sid, u, back)
                stall = stall + 1 if lag > STALL_M else 0
                jn = self.joints_deg(u)
                turned = {k: jn[k] - j0[k] for k in jn if abs(jn[k] - j0[k]) > GUARD_DEG}
                if turned:
                    guard = {k: round(v_, 2) for k, v_ in turned.items()}
                if stall > STALL_TICKS or guard or back <= end:
                    break
            if stall <= STALL_TICKS and not guard:
                break
            att.setdefault("stalls", []).append({"back_mm": round(back * 1e3, 1), "lag_stall": stall > STALL_TICKS, "guard": guard})
            self.log("  %s: stall / guard at %.1f mm (%s)" % (sid, back * 1e3, guard))
            back = min(pre, back + 0.015)
            self.work["phase"] = "back off"
            for _ in range(int(0.015 / (INSERT_SPEED_M_S * cell.tick_s))):
                q = self._cmd_unit_at(sid, u, back, q, lift)
                cell.tick()
            self.work["phase"] = "align"
            self.align(sid, u, back, lift=lift)
            q = ct.q_cmd.copy()
            self.work["phase"] = "insert"
            stiff = max(0.2, stiff - 0.15)
        if sm.state[sid] not in ENGAGED:
            ct.set_stiffness(1.0)
            att["state_at_home"] = sm.state[sid]
            att["put_back"] = self.put_back(u, span)
            return False, "not engaged (state %s, stalls %s)" % (sm.state[sid], att.get("stalls"))
        att["capture"] = sm.rec[sid].get("capture")
        # held still at home, closed loop, until fastened
        self.work["phase"] = "hold"
        t_hold = float(self.d.time)
        for _ in range(int((HOLD_S + LOCK_WAIT_S) / cell.tick_s)):
            q = self._cmd_unit_at(sid, u, end, q, lift)
            cell.tick()
            if sm.state[sid] in FASTENED and float(self.d.time) - t_hold >= HOLD_S:
                break
        att["state_at_hold_end"] = sm.state[sid]
        att["hold_s"] = round(float(self.d.time) - t_hold, 2)
        if self.snap:
            self.snap("%s_home" % sid, u)
        # fastened: the lock welds the base at its design pose (the thumb's 0.2 mm down onto its floor); the hand stops
        # pulling towards its own target, its command set to where the arm stands now, then full stiffness
        ct.set_mode("joint")
        ct.command_joint(ct.q())
        self.tick(5)
        ct.command_joint(ct.q())
        ct.set_stiffness(1.0)
        self.tick(20)
        if sm.state[sid] not in FASTENED:
            return False, "not fastened while held (%s, blocked by %s)" % (sm.state[sid], sm.rec[sid].get("blocked_by"))
        # still held: the finger turned about its ab/ad axis to its park angle (or straightened); the thumb's corner
        # block turned about its side axis
        if kind == "FM" and sid in self.park:
            self.work["phase"] = "swing"
            att["swing"] = self.swing(sid, u, "abad", math.radians(self.park[sid]))
        elif kind == "TM" and self.thumb_side is not None:
            self.work["phase"] = "side"
            att["side"] = self.swing(sid, u, "side", math.radians(self.thumb_side))
        # let go
        self.work.update({"phase": "release", "held": None})
        self.grip(min(self.max_width, span + RELEASE_EXTRA_M), settle_ticks=20)
        self.work["phase"] = "retreat"
        T_out = self.tcp_T()
        if kind == "FM":
            T_out[2, 3] += 0.06
        else:
            T_out[:3, 3] -= T_out[:3, 2] * 0.06
        ok, why = self.line(T_out, v=APPROACH_SPEED_M_S, allow=(u,), width=span + RELEASE_EXTRA_M, check=True)
        if not ok:
            att["retreat_unchecked"] = why
            self.line(T_out, v=APPROACH_SPEED_M_S, check=False)
        self.work.update({"phase": "wait", "unit": None})
        self.tick(30)
        return sm.state[sid] == "LOCKED", sm.state[sid]

    def swing(self, sid, u, jn, target, iters=3, tol_deg=0.1):
        """the held unit's joint `jn` turned to `target` (rad) by turning the hand about the joint's axis (world, now),
        at SWING_DEG_S, closed loop on the joint's angle"""
        j = self.cell.joint_ids[u][jn]
        a_start = float(self.d.qpos[self.m.jnt_qposadr[j]])
        for _ in range(iters):
            a0 = float(self.d.qpos[self.m.jnt_qposadr[j]])
            dang = target - a0
            if abs(math.degrees(dang)) < tol_deg:
                break
            p, ax = self.d.xanchor[j].copy(), self.d.xaxis[j].copy()
            T0 = self.tcp_T()
            n = max(2, int(math.ceil(abs(math.degrees(dang)) / 1.0)))
            poses = [_turn_about(T0, p, ax, dang * k / n) for k in range(n + 1)]
            # the path length move_poses measures per radian of turn: the tcp's arc plus R_EFF; the turn's speed and
            # acceleration (SWING_DEG_S, twice that per second) in those units
            per_rad = float(np.linalg.norm(np.cross(ax, T0[:3, 3] - p))) + R_EFF_M
            w = math.radians(SWING_DEG_S)
            self.move_poses(poses, per_rad * w, per_rad * 2.0 * w, ang_speed=w, settle_ticks=30)
        a1 = float(self.d.qpos[self.m.jnt_qposadr[j]])
        return {"joint": jn, "from_deg": round(math.degrees(a_start), 3), "to_deg": round(math.degrees(a1), 3),
                "target_deg": round(math.degrees(target), 3)}

    # ------------------------------------------------------------------ the palm parts
    def _carrier_bodies(self, sid):
        """the bodies the unit in work may lie on at its mate: the palm base and the mate's carrier units"""
        out = {self.m.body(self.cell.base_body).id}
        for cu in self.sm._carrier_units(self.slot(sid)):
            out |= set(self.cell.bodies_of.get(cu, []))
        return out

    def _above_home(self, sid, u):
        """how far the unit's root link stands above its design pose along the insertion (m)"""
        return float(-((self.unit_T(u)[:3, 3] - self.home_T(sid)[:3, 3]) @ self.ins_world(sid)))

    def _com_offset(self, u, row):
        """how far (m, in plan) the unit's centre of mass lies from the closing axis of a grasp-table row"""
        kind = self.kind(u)
        g = RC.table()["kinds"][kind]["grasps"][int(row)]
        com = np.asarray(self.m.body_ipos[self.cell.bodies_of[u][0]], float) * 1e3
        p, uu = np.asarray(g["centre_mm"], float), np.asarray(g["u"], float)
        v, w = (com - p)[:2], uu[:2] / max(1e-12, float(np.linalg.norm(uu[:2])))
        return abs(float(v[0] * w[1] - v[1] * w[0])) * 1e-3

    def _feasible_plate(self, sid, u, T_u):
        """a palm part's grasp and release height. The candidates whose closing axis passes within COM_OFFSET_M of the
        part's centre of mass come first (nearest first): held off it, a part hinges about the pads as they open and
        lands tilted (development runs: the stack held 12 mm off it tilted 3.7 deg, the top held 5 mm off it 14 deg);
        then the others in the table's order (the plan's grasp first of all). Each group: for each release height of
        the kind, lowest first, its candidates; the pick check computed once per candidate; the first that passes the
        place check at that height"""
        kind = self.kind(u)
        hs = list(RELEASE_M[kind])
        hint = self.plan_grasp.get(u)
        if hint is not None and hint.get("release_m") is not None:
            h0 = float(hint["release_m"])
            hs = [h0] + [h for h in hs if abs(h - h0) > 1e-9]
        rejected, picks = {}, {}
        allc = self.candidates(u)
        if hint is not None or kind not in COM_FIRST:
            groups = [allc]
        else:
            near = sorted([c for c in allc if self._com_offset(u, c[0]) <= COM_OFFSET_M], key=lambda c: self._com_offset(u, c[0]))
            groups = [near, [c for c in allc if c not in near]]
        for cands in groups:
            res = self._feasible_plate_in(sid, u, T_u, hs, cands, picks, rejected)
            if res is not None:
                return res, rejected
        return None, rejected

    def _feasible_plate_in(self, sid, u, T_u, hs, cands, picks, rejected):
        for h in hs:
            for i, sg, T, span in cands:
                key = "%d%+d" % (i, sg)
                if key not in picks:
                    picks[key] = self.pick_check(u, T, span)
                q_pick, why = picks[key]
                if q_pick is None:
                    rejected[key] = "pick: %s" % (why,)
                    continue
                G = _inv(T) @ T_u
                q_pre, why = self.plate_place_check(sid, u, G, span, q_pick, h)
                if q_pre is None:
                    rejected["%s@%.1f" % (key, h * 1e3)] = "place: %s" % why
                    continue
                self.release_m = h
                return (i, sg, T, span, q_pick, q_pre, G)
        return None

    def plate_place_check(self, sid, u, G, span, q_seed, h, step=0.002, finger_path=0.001, finger_pre=0.002, link=0.003,
                          finger_open=0.0005):
        """(q_top, None) when the hand, holding the part's root link at G, reaches the pose PLATE_ABOVE_M above its
        release pose (h above home) and every pose of the line down to it on one IK branch with the fingers closed to
        the grasp's width (1 mm from every other body, at the top 2 mm; the links 3 mm), lets it go there (the fingers
        open to span + 4 mm keep 0.5 mm from every other body) and backs straight up 60 mm (the fingers open, 1 mm;
        the links 3 mm); every branch at the top tried; else (None, why)"""
        pre = h + PLATE_ABOVE_M
        T_pre = self.path_pose(sid, pre) @ _inv(G)
        sols = self.kin.candidates(T_pre, q_seed)
        if not sols:
            return None, "top IK"
        xs = list(np.arange(pre, h - 1e-9, -step))
        if xs[-1] > h + 1e-9:
            xs.append(h)
        why = ""
        w_open = min(self.max_width, span + RELEASE_EXTRA_M)
        f_rel = RELEASE_FINGER_CLEAR_M.get(self.kind(u), finger_path)
        for _sc, q0, _mrg in sols[:6]:
            q, okb = q0, True
            for i, back in enumerate(xs):
                T_b = self.path_pose(sid, back)
                if i == 0:
                    qx, ok = q0, True
                else:
                    qx, ok = self.ik(T_b @ _inv(G), q, margin=0.0)
                    ok = ok and np.max(np.abs(qx - q)) < 0.3
                if not ok:
                    why, okb = "IK at %.1f mm" % (back * 1e3), False
                    break
                self.pl.sync(allow=(u,), poses={u: T_b}, width=span)
                fc = finger_pre if i == 0 else (f_rel if i == len(xs) - 1 else finger_path)
                hit = self.pl.hits(qx, finger_clear=fc, link_clear=link)
                if hit:
                    why, okb = "hits %s at %.1f mm" % (sorted(set((hh[0], hh[1]) for hh in hit))[:3], back * 1e3), False
                    break
                q = qx
            if not okb:
                continue
            T_r = self.path_pose(sid, h)
            self.pl.sync(allow=(u,), poses={u: T_r}, width=w_open)
            hit = self.pl.hits(q, finger_clear=finger_open, link_clear=link)
            if hit:
                why = "opening hits %s" % (sorted(set((hh[0], hh[1]) for hh in hit))[:3],)
                continue
            T_tcp = T_r @ _inv(G)
            ups, qq, oku = [q], q, True
            for k in range(1, 31):
                T_up = T_tcp.copy()
                T_up[2, 3] += 0.002 * k
                q2, ok = self.ik(T_up, qq, margin=0.0)
                if not ok or np.max(np.abs(q2 - qq)) > 0.3:
                    oku = False
                    break
                ups.append(q2)
                qq = q2
            if not oku:
                why = "retreat IK"
                continue
            if not self.pl.line_free(ups, finger_clear=0.001, link_clear=link):
                why = "retreat: " + self.pl.why
                continue
            return q0, None
        return None, why or "release"

    def place_plate(self, sid, u, span, att, q_branch=None):
        """carried to PLATE_ABOVE_M above its release pose, aligned, a line down closed loop on its live pose,
        held still, let go (the cover, released at home: held until the cell fastens it), the hand straight up; the
        part settles in its own guides and the cell centres it; the lock waited for; pushed home if it is not
        fastened by then (the cap always)"""
        cell, ct, sm = self.cell, self.ct, self.sm
        kind = self.kind(u)
        h = float(att.get("release_m", RELEASE_M[kind][0]))
        pre = h + PLATE_ABOVE_M
        G = self.G_act(u)
        T_t = self.path_pose(sid, pre) @ _inv(G)
        sols = self.kin.candidates(T_t, ct.q_cmd)
        if q_branch is not None:
            q_v, ok_v = self.ik(T_t, q_branch, margin=0.0)
            if ok_v:
                sols = [(0.0, q_v, 0.0)] + [s for s in sols if np.max(np.abs(s[1] - q_v)) > 1e-3]
        self.work["phase"] = "carry"
        v, a = CARRY.get(kind, DEFAULT_CARRY)
        moved = False
        for _sc, q_pre, _m in sols[:6]:
            if self.go(q_pre, held=(u, G), width=span, v=v, a=a):
                moved = True
                break
        if not moved:
            att["put_back"] = self.put_back(u, span)
            return False, "no path to the release pose (%s)" % self.pl.why
        self.tick(30)
        G_after = self.G_act(u)
        att["carry_shift"] = {"mm": round(float(np.linalg.norm(G_after[:3, 3] - G[:3, 3])) * 1e3, 3),
                              "deg": round(math.degrees(rotation_angle(G[:3, :3].T @ G_after[:3, :3])), 3)}
        self.work["phase"] = "align"
        att["align_at_pre"] = self.align(sid, u, pre)
        if self.snap:
            self.snap("%s_pre" % sid, u)
        # the line down, closed loop on the part's live pose (the cover, carried home: the arm at half stiffness)
        self.work["phase"] = "insert"
        ct.set_stiffness(1.0 if h > 0 else INSERT_STIFFNESS)
        ct.set_mode("joint")
        back, q, stall = pre, ct.q_cmd.copy(), 0
        for _ in range(int((pre - h + 0.02) / (PLATE_DESCENT_M_S * cell.tick_s)) + 100):
            back = max(h, back - PLATE_DESCENT_M_S * cell.tick_s)
            q = self._cmd_unit_at(sid, u, back, q)
            cell.tick()
            stall = stall + 1 if self._lag(sid, u, back) > STALL_M else 0
            if stall > STALL_TICKS or back <= h:
                break
        stalled = stall > STALL_TICKS
        if stalled:
            att["descent_stall_mm"] = round(back * 1e3, 2)
        att["at_release"] = {"above_home_mm": round(self._above_home(sid, u) * 1e3, 3), "state": sm.state[sid]}
        # held still there (the cover until fastened: once it lies within the lock's 0.5 mm the hand's command stays
        # where the arm stands, so that the cover comes to rest on its seat instead of following the closed loop's
        # corrections; held by the closed loop it kept HOLDING for 4.3 s without 50 ticks at rest in a development run)
        self.work["phase"] = "hold"
        t_hold = float(self.d.time)
        frozen, lagging = stalled, 0
        if stalled:
            # the part sits on something above its release pose: no closed loop pushing it there
            ct.command_joint(ct.q())
        for _ in range(int((PLATE_HOLD_S + (LOCK_WAIT_S if h <= 0 else 0.0)) / cell.tick_s)):
            if h <= 0 and not frozen and float(self.d.time) - t_hold >= PLATE_HOLD_S and self._above_home(sid, u) <= 0.0004:
                # pressed onto its seat: one command SEAT_PRESS_M below home, kept (no closed loop, so it comes to rest)
                q = self._cmd_unit_at(sid, u, -SEAT_PRESS_M, q)
                frozen = True
                att["seat_press_mm"] = round(SEAT_PRESS_M * 1e3, 3)
            if not frozen:
                q = self._cmd_unit_at(sid, u, h, q)
            cell.tick()
            lagging = lagging + 1 if (not frozen and self._lag(sid, u, h) > 0.0015) else 0
            if lagging > STALL_TICKS:
                # held 1.5 mm or more above where the hand wants it for 0.3 s: something is in the way; stop pressing
                ct.command_joint(ct.q())
                frozen = True
                att["hold_lag_stop"] = round(self._above_home(sid, u) * 1e3, 2)
            if h <= 0 and sm.state[sid] in FASTENED and float(self.d.time) - t_hold >= PLATE_HOLD_S:
                break
        att["hold_frozen"] = frozen
        att["hold_s"] = round(float(self.d.time) - t_hold, 2)
        att["state_at_hold_end"] = sm.state[sid]
        ct.set_mode("joint")
        ct.command_joint(ct.q())
        self.tick(5)
        ct.command_joint(ct.q())
        ct.set_stiffness(1.0)
        self.tick(10)
        if self.snap:
            self.snap("%s_release" % sid, u)
        # let go, the hand straight up
        self.work.update({"phase": "release", "held": None})
        t_rel = float(self.d.time)
        at_open = {}

        def _at_open():
            # the first tick of the opening: the part's height there (the end of the descent is `at_release`)
            if not at_open:
                at_open.update({"t_s": round(float(self.d.time), 3), "above_home_mm": round(self._above_home(sid, u) * 1e3, 3),
                                "state": sm.state[sid]})
        cell.hooks.append(_at_open)
        try:
            self.grip(min(self.max_width, span + RELEASE_EXTRA_M), settle_ticks=20)
        finally:
            cell.hooks.remove(_at_open)
        att["at_open"] = dict(at_open)
        self.work["phase"] = "retreat"
        T_out = self.tcp_T()
        T_out[2, 3] += 0.06
        ok, why = self.line(T_out, v=APPROACH_SPEED_M_S, allow=(u,), width=span + RELEASE_EXTRA_M, check=True)
        if not ok:
            att["retreat_unchecked"] = why
            self.line(T_out, v=APPROACH_SPEED_M_S, check=False)
        # the part settles in its guides; the cell fastens it once it lies at rest on its seat
        self.work["phase"] = "settle"
        wait_s = 0.3 if kind in ALWAYS_PUSH else PLATE_WAIT_S
        while sm.state[sid] not in FASTENED and float(self.d.time) - t_rel < wait_s:
            cell.tick()
        att["after_wait"] = {"state": sm.state[sid], "above_home_mm": round(self._above_home(sid, u) * 1e3, 3),
                             "t_after_release_s": round(float(self.d.time) - t_rel, 2)}
        if sm.state[sid] not in FASTENED:
            att["push"] = self.push(sid, u)
        st = sm.rec[sid]
        att["release_to_lock_s"] = round(float(st["t_s"]) - t_rel, 3) if st["fastened"] else None
        if self.snap:
            self.snap("%s_home" % sid, u)
        self.work.update({"phase": "wait", "unit": None})
        self.tick(30)
        return sm.state[sid] == "LOCKED", sm.state[sid]

    def _top_map(self, kind):
        """(xs, ys, H): the top height (unit frame, m) of the kind's collision pieces over 1 mm columns (NaN: none),
        by downward rays on a model of its pieces alone"""
        if kind in self._maps:
            return self._maps[kind]
        import mujoco
        spec = mujoco.MjSpec()
        b = spec.worldbody.add_body(name="u")
        hulls = self.kit.collision_hulls(kind)
        for i, (_l, V, _s) in enumerate(hulls):
            ms = spec.add_mesh(name="h%d" % i)
            ms.uservert = (np.asarray(V, float) * 1e-3).flatten().tolist()
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="h%d" % i)
        mm = spec.compile()
        dd = mujoco.MjData(mm)
        mujoco.mj_forward(mm, dd)
        P = np.concatenate([np.asarray(V, float) for _l, V, _s in hulls]) * 1e-3
        lo, hi = P.min(0), P.max(0)
        xs = np.arange(lo[0], hi[0] + 1e-9, 0.001)
        ys = np.arange(lo[1], hi[1] + 1e-9, 0.001)
        H = np.full((len(ys), len(xs)), np.nan)
        z0 = float(hi[2]) + 0.02
        gid = np.zeros(1, np.int32)
        dn = np.array([0.0, 0.0, -1.0])
        for r, y in enumerate(ys):
            for c_, x in enumerate(xs):
                dist = mujoco.mj_ray(mm, dd, np.array([x, y, z0]), dn, None, 1, -1, gid)
                if dist >= 0:
                    H[r, c_] = z0 - dist
        self._maps[kind] = (xs, ys, H)
        return self._maps[kind]

    def _true_top_map(self, kind):
        """the top height (unit frame, m) of the kind's TRUE surface over the columns of `_top_map` (NaN: none): the
        hidden kit's drawn meshes of its root link (the authors' solids tessellated within 0.05 mm), each triangle's
        plane sampled at the column centres it covers, the highest kept"""
        key = "true:" + kind
        if key in self._maps:
            return self._maps[key]
        from .kit import read_ply
        xs, ys, _H = self._top_map(kind)
        Ht = np.full((len(ys), len(xs)), np.nan)
        for v in self.kit.kinds[kind].get("visual", []):
            if v.get("link", "body") != "body":
                continue
            V, F = read_ply(os.path.join(self.kit.dir, v["file"]))
            V = np.asarray(V, float) * 1e-3
            for f in np.asarray(F, np.int64):
                a, b, c_ = V[f[0]], V[f[1]], V[f[2]]
                den = (b[1] - c_[1]) * (a[0] - c_[0]) + (c_[0] - b[0]) * (a[1] - c_[1])
                if abs(den) < 1e-18:
                    continue
                lo = np.minimum(np.minimum(a, b), c_)
                hi = np.maximum(np.maximum(a, b), c_)
                i0, i1 = int(np.ceil((lo[0] - xs[0]) / 0.001 - 1e-9)), int(np.floor((hi[0] - xs[0]) / 0.001 + 1e-9))
                j0, j1 = int(np.ceil((lo[1] - ys[0]) / 0.001 - 1e-9)), int(np.floor((hi[1] - ys[0]) / 0.001 + 1e-9))
                i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1, len(xs) - 1), min(j1, len(ys) - 1)
                if i1 < i0 or j1 < j0:
                    continue
                X, Y = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1])
                l1 = ((b[1] - c_[1]) * (X - c_[0]) + (c_[0] - b[0]) * (Y - c_[1])) / den
                l2 = ((c_[1] - a[1]) * (X - c_[0]) + (a[0] - c_[0]) * (Y - c_[1])) / den
                l3 = 1.0 - l1 - l2
                inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
                if not inside.any():
                    continue
                Z = np.where(inside, l1 * a[2] + l2 * b[2] + l3 * c_[2], np.nan)
                blk = Ht[j0:j1 + 1, i0:i1 + 1]
                Ht[j0:j1 + 1, i0:i1 + 1] = np.fmax(blk, Z)
        self._maps[key] = (xs, ys, Ht)
        return self._maps[key]

    def push_search(self, sid, u, step=0.002, finger_clear=0.001, finger_end=0.0005, link_clear=0.003, n_exact=400,
                    min_support_mm2=25):
        """a pose of the closed fingertips straight down on the part's top as it lies now: candidates on a 2 mm grid
        of its unit frame and four yaws of the footprint (each with the hand both ways), kept where the footprint
        rests on at least 25 mm2 of the top of its collision pieces within 0.5 mm of their highest point there and
        where the part's TRUE top (its drawn meshes) stands nowhere under the footprint more than PUSH_TRUE_TOL_M above
        that point (the tips then rest on the true surface, not inside it; the touch at the higher of the two), ordered
        by the distance from the part's centre of mass in plan (the bench plan's push pose first, if it passes the same
        test); the first whose touch pose has an IK solution with the fingers 1 mm from every other body and the links
        3 mm, and whose pose at the push's end (the part 0.25 mm above home, the hand lowered by as much) keeps the
        fingers 0.5 mm from every other body: (T_touch, q_touch, info) or None"""
        kind = self.kind(u)
        xs, ys, H = self._top_map(kind)
        _xs, _ys, Ht = self._true_top_map(kind)
        T_u = self.unit_T(u)
        com = np.asarray(self.m.body_ipos[self.cell.bodies_of[u][0]], float)
        fx, fy = PUSH_FOOT_M
        ga = np.arange(-fx, fx + 1e-9, 0.001)
        gb = np.arange(-fy, fy + 1e-9, 0.001)
        A, B = np.meshgrid(ga, gb)
        A, B = A.ravel(), B.ravel()
        diag = self.push_diag = {"candidates": 0, "true_above": 0, "ik": 0, "touch": {}, "end": {}, "end_ik": 0}

        def foot(yaw, x0, y0):
            """(touch top, support mm2, collision top, true top) of the footprint at yaw about (x0, y0), or None"""
            cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            # the tcp's y (closing axis) along (cy, sy) of the unit frame, its x = y x z with z down: (-sy, cy)
            px = -sy * A + cy * B
            py = cy * A + sy * B
            ix = np.rint((x0 + px - xs[0]) / 0.001).astype(int)
            iy = np.rint((y0 + py - ys[0]) / 0.001).astype(int)
            ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
            hh = np.full(len(ix), np.nan)
            hh[ok] = H[iy[ok], ix[ok]]
            if np.all(np.isnan(hh)):
                return None
            top = float(np.nanmax(hh))
            sup = int(np.sum(hh >= top - 0.0005))
            if sup < min_support_mm2:
                return None
            ht = Ht[iy[ok], ix[ok]]
            ttop = float(np.nanmax(ht)) if np.any(~np.isnan(ht)) else -1.0
            if ttop > top + PUSH_TRUE_TOL_M:
                diag["true_above"] += 1
                return None
            return max(top, ttop), sup, top, ttop

        cands = []
        for yaw in (0.0, 45.0, 90.0, 135.0):
            for x0 in np.arange(xs[0] + 0.002, xs[-1] - 0.002, step):
                for y0 in np.arange(ys[0] + 0.002, ys[-1] - 0.002, step):
                    f_ = foot(yaw, float(x0), float(y0))
                    if f_ is None:
                        continue
                    cands.append((float(math.hypot(x0 - com[0], y0 - com[1])), yaw, float(x0), float(y0)) + f_)
        cands.sort()
        hint = self.plan_push.get(u)
        if hint is not None:
            # the bench plan's push pose first (the yaw of its footprint, the hand either way), on the same test
            hy = float(hint["yaw_deg"]) % 180.0
            hx, hyy = float(hint["push_point_mm"][0]) * 1e-3, float(hint["push_point_mm"][1]) * 1e-3
            f_ = foot(hy, hx, hyy)
            diag["plan_push_pose"] = "taken" if f_ is not None else "refused by the footprint test"
            if f_ is not None:
                cands = [(0.0, hy, hx, hyy) + f_] + cands
        diag["candidates"] = len(cands)
        above0 = self._above_home(sid, u)
        drop = max(0.0, above0 - PUSH_TO_M)
        T_end_u = self.home_T(sid).copy()
        T_end_u[:3, 3] -= self.ins_world(sid) * PUSH_TO_M
        tried = 0
        for sc, yaw, x0, y0, top, sup, ctop, ttop in cands:
            if tried >= n_exact:
                break
            tried += 1
            for flip in (0.0, 180.0):
                a = math.radians(yaw + flip)
                yl = np.array([math.cos(a), math.sin(a), 0.0])
                yw = T_u[:3, :3] @ yl
                yw[2] = 0.0
                yw /= np.linalg.norm(yw)
                zw = np.array([0.0, 0.0, -1.0])
                xw = np.cross(yw, zw)
                Tt = np.eye(4)
                Tt[:3, :3] = np.column_stack([xw, yw, zw])
                Tt[:3, 3] = T_u[:3, :3] @ np.array([x0, y0, top]) + T_u[:3, 3] + np.array([0.0, 0.0, PUSH_TIP_M])
                q, ok = self.ik(Tt, self.ct.q_cmd, margin=0.0)
                if not ok:
                    sols = self.kin.candidates(Tt, self.ct.q_cmd)
                    if not sols:
                        diag["ik"] += 1
                        continue
                    q = sols[0][1]
                self.pl.sync(allow=(u,), width=0.0)
                hit = self.pl.hits(q, finger_clear=finger_clear, link_clear=link_clear)
                if hit:
                    for h_ in set((h_[0], h_[1]) for h_ in hit):
                        diag["touch"]["%s|%s" % h_] = diag["touch"].get("%s|%s" % h_, 0) + 1
                    continue
                Te = Tt.copy()
                Te[2, 3] -= drop
                qe, oke = self.ik(Te, q, margin=0.0)
                if not oke:
                    diag["end_ik"] += 1
                    continue
                self.pl.sync(allow=(u,), poses={u: T_end_u}, width=0.0)
                hit = self.pl.hits(qe, finger_clear=finger_end, link_clear=link_clear)
                if hit:
                    for h_ in set((h_[0], h_[1]) for h_ in hit):
                        diag["end"]["%s|%s" % h_] = diag["end"].get("%s|%s" % h_, 0) + 1
                    continue
                info = {"push_point_mm": [round(x0 * 1e3, 1), round(y0 * 1e3, 1)], "yaw_deg": yaw + flip, "support_mm2": sup,
                        "from_com_mm": round(sc * 1e3, 1), "top_mm": round(top * 1e3, 2), "collision_top_mm": round(ctop * 1e3, 3),
                        "true_top_mm": round(ttop * 1e3, 3), "candidates": len(cands), "tried": tried,
                        "true_above_refused": diag["true_above"], "plan_push_pose": diag.get("plan_push_pose")}
                return Tt, q, info
        return None

    def push(self, sid, u):
        """the part pushed home with the closed fingertips: a push pose (push_search), a planned move 30 mm above it,
        a checked line to 3 mm above it, then down at PUSH_SPEED_M_S at stiffness PUSH_STIFFNESS until the part is
        PUSH_TO_M above home, the tips PUSH_PAST_M past where its home would put them, or the watch reads PUSH_MAX_N;
        the command set to where the arm stands; held until fastened; the hand straight up"""
        cell, ct, sm = self.cell, self.ct, self.sm
        rec = {"t0_s": round(float(self.d.time), 2), "above_home_mm_before": round(self._above_home(sid, u) * 1e3, 3)}
        self.work["phase"] = "to-push"
        t_w = time.time()
        cand = self.push_search(sid, u)
        rec["search_wall_s"] = round(time.time() - t_w, 1)
        if cand is None:
            rec["result"] = "no push pose"
            return rec
        T_touch, q_touch, info = cand
        rec.update(info)
        self.grip(0.0, settle_ticks=10)
        T_hi = T_touch.copy()
        T_hi[2, 3] += 0.03
        q_hi, ok = self.ik(T_hi, q_touch, margin=0.0)
        if not ok or not self.go(q_hi, width=0.0):
            rec["result"] = "no path to the push pose (%s)" % self.pl.why
            return rec
        T_near = T_touch.copy()
        T_near[2, 3] += 0.003
        ok, why = self.line(T_near, v=0.02, allow=(u,), width=0.0, check=True)
        if not ok:
            rec["result"] = "approach: " + why
            return rec
        self.work["phase"] = "push"
        ct.set_stiffness(PUSH_STIFFNESS)
        ct.set_mode("joint")
        z_floor = T_touch[2, 3] - self._above_home(sid, u) - PUSH_PAST_M
        Tc = T_near.copy()
        q = ct.q_cmd.copy()
        stop, r2max = None, 0.0
        rec["first_contact"] = None
        for _ in range(int((0.003 + 0.03) / (PUSH_SPEED_M_S * cell.tick_s))):
            Tc[2, 3] = max(z_floor, Tc[2, 3] - PUSH_SPEED_M_S * cell.tick_s)
            q2, ok = self.ik(Tc, q)
            if ok:
                q = q2
                ct.command_joint(q)
            else:
                self.ik_fail += 1
            cell.tick()
            r2 = float(self.srv.readings.get("R2_N", 0.0)) if self.srv is not None else 0.0
            r2max = max(r2max, r2)
            if rec["first_contact"] is None and self._tips_on(u):
                rec["first_contact"] = {"t_s": round(float(self.d.time), 3), "above_home_mm": round(self._above_home(sid, u) * 1e3, 3),
                                        "state": sm.state[sid]}
            # the push runs to contact whatever the part's height; from there it stops at home, on force or past
            if rec["first_contact"] is not None and self._above_home(sid, u) <= PUSH_TO_M:
                stop = "home"
                break
            if r2 > PUSH_MAX_N:
                stop = "force"
                break
            if Tc[2, 3] <= z_floor + 1e-9 and self._lag_z(Tc) < 0.0005:
                stop = "past"
                break
        rec["stop"] = stop or "time"
        rec["above_home_mm_at_stop"] = round(self._above_home(sid, u) * 1e3, 3)
        rec["R2_max_N"] = round(r2max, 3)
        ct.command_joint(ct.q())
        self.tick(5)
        ct.command_joint(ct.q())
        t_h = float(self.d.time)
        while sm.state[sid] not in FASTENED and float(self.d.time) - t_h < LOCK_WAIT_S:
            cell.tick()
        rec["state"] = sm.state[sid]
        rec["wait_s"] = round(float(self.d.time) - t_h, 2)
        ct.set_stiffness(1.0)
        self.tick(10)
        self.work["phase"] = "retreat"
        T_out = self.tcp_T()
        T_out[2, 3] += 0.06
        ok, why = self.line(T_out, v=APPROACH_SPEED_M_S, allow=(u,), width=0.0, check=True)
        if not ok:
            rec["retreat_unchecked"] = why
            self.line(T_out, v=APPROACH_SPEED_M_S, check=False)
        rec["result"] = sm.state[sid]
        return rec

    def _tips_on(self, u):
        """whether a finger body of the Panda touches the unit now (a contact with a positive normal force)"""
        m, d = self.m, self.d
        ub = set(self.cell.bodies_of[u])
        fb = {m.body("panda/left_finger").id, m.body("panda/right_finger").id}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in ub and b2 in fb) or (b2 in ub and b1 in fb):
                self.cell.mj.mj_contactForce(m, d, i, f6)
                if f6[0] > 0.0:
                    return True
        return False

    def _lag_z(self, Tc):
        """how far the actual tcp stands above the commanded one (m)"""
        return float(self.tcp_T()[2, 3] - Tc[2, 3])


def steps_of(plan, scope_slots):
    """[(slot, unit)] in the assembly order: the bench plan's, else the default one"""
    if plan and plan.get("steps"):
        return [(st["slot"], st["unit"]) for st in plan["steps"]]
    return [(s, u) for s, u in DEFAULT_STEPS if s in scope_slots]


def program(cell, plan=None, log=print, server=None, only=None, wall_cap_s=None, snap=None, end_ticks=300, keep=None):
    """run the privileged assembler on `cell` (a HandCell with the Panda; its tick routed through the server's watch
    when the server built it); returns the report (`keep`, a dict, receives the Assembler: its force log)"""
    t0 = time.time()
    asm = Assembler(cell, plan=plan, log=log, server=server, wall_cap_s=wall_cap_s, snap=snap)
    steps = []
    asm.ct.command_joint(asm.q_ready)
    asm.settle(max_ticks=200)
    for sid, u in steps_of(plan, set(x["key"] for x in cell.sm.slots)):
        if only and sid not in only:
            continue
        if asm.out_of_time():
            steps.append({"slot": sid, "unit": u, "placed_by": "nobody", "why": "wall-clock cap"})
            continue
        x_ = cell.sm.by_key[sid]
        missing = [q for q in x_["requires"] if not cell.sm.rec[cell.sm.by_id[q]["key"]]["fastened"]]
        if missing:
            # what this unit is screwed onto is not fastened: it stays in its nest (a unit pushed onto an unfastened
            # carrier only loads it)
            log("step %s <- %s: not attempted (requires %s, not fastened)" % (sid, u, missing))
            steps.append({"slot": sid, "unit": u, "placed_by": "nobody", "why": "requires %s not fastened" % missing,
                          "state": cell.sm.state[sid], "attempts": []})
            continue
        log("step %s <- %s" % (sid, u))
        steps.append(asm.place_unit(sid, u))
    asm.work.update({"phase": "end", "unit": None, "slot": None, "held": None})
    if not asm.go(asm.q_ready):
        asm.ct.command_joint(asm.q_ready)
        asm.settle(max_ticks=400)
    asm.tick(end_ticks)
    rep = {"score": round(cell.sm.score(), 4), "steps": steps, "events": [(round(t, 3), e) for t, e in cell.sm.events],
           "states": dict(cell.sm.state),
           "unplanned_contacts": [{k: v for k, v in c.items() if k != "key"} for c in asm.contacts],
           "unit_contacts": [{k: v for k, v in c.items() if k != "key"} for c in asm.unit_contacts],
           "support_contacts_N": [list(k) + [v] for k, v in sorted(asm.support.items())],
           "robot_touches_any_force": [list(k) + [v] for k, v in sorted(asm.robot_touch.items())],
           "sim_s": round(float(cell.d.time), 2), "wall_s": round(time.time() - t0, 1),
           "planner_checks": int(asm.pl.n_checks), "ik_failures": int(asm.ik_fail),
           "planned_moves": [list(p) for p in asm.paths]}
    log("score %.3f, %d / %d slots, sim %.0f s, wall %.0f s, unplanned robot contacts %d, unit contacts %d" % (
        rep["score"], sum(1 for x in cell.sm.rec.values() if x["achieved"]), len(cell.sm.rec), rep["sim_s"], rep["wall_s"],
        len(rep["unplanned_contacts"]), len(rep["unit_contacts"])))
    if keep is not None:
        keep["assembler"] = asm
    return rep
