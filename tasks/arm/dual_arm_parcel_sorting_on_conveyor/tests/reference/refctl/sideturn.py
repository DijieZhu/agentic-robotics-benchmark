"""Quarter turn of a parcel about the vertical on the running belt (the
"door swing"), for a label on the face away from the robot (+x) or towards
it (-x).

Physics. A parcel stopped against a point contact on the near-side end of
its downstream face is a door hinged at that point: the belt keeps sliding
under its base and drags it downstream with mu_belt m g through its centre of
mass, which lies on the far side of the hinge, so the parcel swings
counter-clockwise (seen from above: its far end goes downstream). Every
corner of a parcel turning about a hinge on its near side moves away from the
near lip, so the swing never presses the parcel into the lip (a clockwise
swing would need its hinge on the far side and sweeps the near rear corner
into the lip). The hinge holds the parcel against the belt through its pad
friction only while the face is within the pad's friction angle of square to
the belt (tan(swing) <= mu_pad = 1.8, about 61 degrees); beyond that the
parcel slides off the hinge and rides on, turned by some 55-65 degrees. The
roll's catch downstream (the left palm, square to the belt) then squares it
to the nearest quarter turn: 90 degrees.

Turn: +x -> +y (then one end-over-end roll brings the label up),
      -x -> -y (the label is then on the upstream face).

The hinge is the right palm: tool pointing upstream (-y) and tilted down,
pad line across the belt, its far pad at the downstream face 2 cm from the
near corner, 3-4 cm above the belt; the near pad and the gripper body stay
inside the near-lip line, so the swinging parcel never meets them. The stop
is a speed-matched soft catch, as the roll's own catch. While the parcel
swings, the left arm moves to the roll's catch wait pose; when the parcel has
left the hinge the right palm lifts clear and parks.

Privileged reference: reads the parcel's pose, size and mass from the
simulator's truth; writes only joint position targets and gripper commands
(through the roll package's executor: inverse-dynamics feedforward, a rate
clamp of the command at 5 rad/s so the cell's 6 rad/s limiter never engages).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .endover import PAD_TIP, PAD_X, EndOver, RollParams
from .execu import SL
from .geom import label_dir, quat_mat
from .kin import rot_axis_angle
from .motion import DT, joint_move
from .paths import PlanFail, palm_R

G = 9.81


def frame_zx(z, x_hint):
    z = np.asarray(z, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, float)
    x = x - z * float(x @ z)
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


@dataclass
class SideParams:
    method: str = "hinge"            # "hinge" (chosen): a pad at the downstream face is the pivot; "press": a finger on the top (rejected)
                                     # "hinge": a finger / pad at the downstream face is the pivot
    press_e1: float = 0.016          # press point this far in from the near edge of the top (m)
    press_e2: float = 0.016          # ... and from the front edge (m)
    press_f: float = 6.0             # N press (down), base
    press_f_mg: float = 1.5          # + this x m g (the pad's friction 1.8 N must hold 0.9 (m g + N))
    press_y_alt: tuple = (-0.50, -0.48, -0.46, -0.44, -0.42, -0.40, -0.37, -0.34, -0.31, -0.28)
    press_clear: float = 0.02        # finger above the top while it tracks the parcel (m)
    press_over: float = 0.003        # target this far under the top once pressing (m)
    press_t_acc: float = 0.15
    press_t_desc: float = 0.15
    press_t_ramp: float = 0.10
    press_t_dec: float = 0.30
    press_release_deg: float = 88.0  # lift the finger at this swing (minus the rate lead)
    press_lead_s: float = 0.04       # release lead: swing rate x this
    press_timeout_s: float = 10.0
    press_lift_s: float = 0.15       # fast first lift off the top
    press_lift_dz: float = 0.04
    edge_e: float = 0.012            # edge hinge: pad centre this far from the near corner along the top-front edge
    edge_over: float = 0.002         # edge hinge: pad target pressed in along the tool axis (m)
    edge_k: float = 0.6              # edge hinge: downward / upstream force ratio of the feedforward
    edge_f_margin: float = 1.15      # edge hinge: feedforward = margin x the holding force
    release_edge_deg: float = 88.0   # edge hinge: lift at this swing (minus the rate lead)
    release_lead_s: float = 0.04
    handover: str = "square"         # "square": square catch placed for the squared centre, then justify; "flush"
    catch_mode: str = "roll"         # "roll": the roll's catch pose; "low": the low-face catch (pads side by side, tilted)
    catch_x_hi_tilted: float = 0.39  # tilted catch palm: reach across the belt (m)
    square_pred: str = "corner"      # where the parcel's centre will be once squared: "corner" | "sim"
    square_margin: float = -1.0      # > 0: near pad at least this far on the near side of the centre (m); < 0: off
    long_margin: float = -1.0        # > 0: for parcels at least long_min long along the belt at the catch (the widest
                                     # +x cartons after the turn) the catch palm's centre stays this far inboard of the
                                     # parcel's centre: with the palm at or outboard of the centre the belt's drag turns
                                     # a long parcel back instead of squaring it (catch yaw -17..-33 deg, the gate lets
                                     # it go: 5 of 17 cartons wider than 0.28 m in e10); < 0: off
    long_min: float = 0.27
    glide_t_acc: float = 0.15        # glide re-pose: s to belt speed
    glide_t_dec: float = 0.30        # ... s of braking at the end
    glide_t_min: float = 0.30        # ... at least this long at belt speed (the pose change)
    glide_v_frac: float = 1.0
    own_t_acc: float = 0.10          # own catch: s to the parcel's speed
    own_t_dec: float = 0.24          # ... s of braking
    own_vmax: float = 4.5            # rad/s of the move to its wait pose
    own_amax: float = 30.0           # rad/s^2 of that move (short moves are acceleration-bound: at 8 a move of
                                     # 0.36 rad takes 0.52 s, which makes the low catch "too late" on fast belts)
    release_time_need: float = 1.05  # +x: the hinge lets go once the parcel's front would reach the roll's catch line
                                     # (0.15 m) in less than this (s): the low catch still has time (0: off)
    catch_early: int = 1             # +x: the left arm starts its move to the low catch's wait pose at the hinge's
                                     # release, during the hinge's exit
    plus_x_handover: str = "low2"    # +x after the turn: "low2" (low catch, two-palm slide, re-pose,
                                     # staging, gate, then "caught"); "free": the right arm parks and the turned
                                     # parcel rides on, released, for the scheduler's own catch and roll
    catch_follow_x: int = 0          # own catch: the palm follows the parcel's centre across the belt while it squares
    catch_follow_rate: float = 0.15  # m/s of that following
    j2_tilt_deg: float = 30.0        # two-palm justify: right palm tilt (down)
    j2_z: float = 0.03               # ... pad height on the upstream face (m)
    j2_over: float = 0.002           # ... target this far into the face (m)
    j2_x_back: float = 0.0           # (unused)
    j2_balance: float = 0.0          # ... 1: balance the two palms' yaw moments; 0: right palm at the centre
    repose_mode: str = "squeezed"    # "squeezed" (chosen): re-pose in place while the right palm squeezes; "glide": while gliding with the parcel
    repose_s: float = 0.6
    repose_bias: float = 0.0         # the re-posed palm this far beyond the parcel's centre across the belt (m)
    repose_x_hi: float = 0.33        # ... at most this far out (the roll's catch reach)
    repose_psi_deg: float = 30.0     # >= 0: the re-posed left palm's pad-line angle (0: pads side by side across the
                                     # belt, a +-4.7 cm span against the creep of an off-centre parcel); < 0: the roll's
                                     # own catch geometry (psi 45 deg, +-3.3 cm)
    j2_f: float = 6.0                # ... squeeze, N
    j2_f_mg: float = 0.5             # ... + this x m g
    j2_f_cap: float = 110.0          # ... squeeze lowered while the summed force on the parcel is above this (N)
    j2_gain: float = 1.0             # ... commanded slide / wanted slide
    self_margin: float = 0.004       # an arm's clearance from the column, the head and the other arm in every plan (m; pairs already that close at the start excepted)
    escape_checked: int = 1          # the right palm's escapes planned and checked (0: a straight-up line, unchecked)
    justify_passes: int = 1
    justify_gain: float = 1.35       # palm travel / wanted parcel travel (the parcel lags the palm)
    justify_press_mg: float = 1.1    # palm feedforward into the face while sliding (x m g)
    justify_tol: float = 0.008
    prepose_dx: float = 0.06         # low catch pre-pose: expected outward drift of the turned parcel (m)
    prepose_low: int = 1             # the left arm's pre-pose during the turn: the handover's low catch
    justify_gap: float = 0.006       # square-and-justify: near face this far inside the near belt edge (m)
    justify_speed: float = 0.06      # m/s of the sideways slide
    justify_f_mg: float = 0.25       # sideways feedforward (x m g) while sliding
    square_rate_deg_s: float = 25.0
    hinge: str = "outward"           # "outward" (chosen): the right palm pointing upstream, tilted down; "vertical": a finger pointing down;
                                     # "outward": the right palm pointing upstream, tilted down
    pin_tilt_deg: float = 35.0       # outward hinge: tool tilted down from pointing upstream
    pin_e: float = 0.008             # hinge finger centre this far from the near corner (m)
    pin_e_out: float = 0.020         # outward hinge: pad contact this far from the corner (m)
    hinge_closed: int = 1            # outward hinge with the fingers closed (1, chosen): the closed fingertips are the pivot (no open finger reaches back over the lip or around the tall rail's end); 0: the open far pad; 2: open where its plan is clear, else closed (measured: more parcels damaged, a rail touch left)
    pin_e_closed: float = 0.020      # ... closed fingertips' centre this far from the corner (m)
    close_s: float = 0.36            # ... s to close the fingers before the plan
    pin_yaw_deg: float = 0.0         # ... tool yawed towards +x by this (the gripper body on the robot's side of the hinge)
    pin_h: float = 0.012             # vertical hinge: TCP height above the belt (fingertips 1.5 cm up)
    pin_h_out: float = 0.035         # outward hinge: pad tip height above the belt (m)
    pin_y_alt: tuple = (-0.40, -0.38, -0.36, -0.34, -0.32, -0.30, -0.28, -0.26)
    rail_y_min: float = -0.405       # the gripper body stays downstream of the tall near rail
    rail_y_min_out: float = -0.415   # outward hinge: wait-pose TCP y lower bound (same reason)
    pin_gap: float = -0.001          # hinge tip target relative to the face (< 0: pressed in)
    pin_t_acc: float = 0.15          # s to reach the parcel's speed
    pin_t_dec: float = 0.30          # s of braking to rest
    pin_f: float = 2.0               # N feedforward of the hinge (upstream)
    pin_f_mg: float = 0.8            # + this x m g
    pin_vmax: float = 3.6            # rad/s of the approach move
    pin_amax: float = 8.0            # rad/s^2 of the approach move
    pin_fast_v: float = 0.26         # belts faster than this (m/s): the approach moves at pin_vmax_fast / pin_amax_fast
    pin_vmax_fast: float = 4.4
    pin_amax_fast: float = 14.0
    release_deg: float = 0.0         # > 0: the hinge retracts at this swing angle
    swing_timeout_s: float = 9.0
    lost_ticks: int = 12             # no hinge contact this long (past free_min_deg): it has left
    moving_ticks: int = 5            # ... or riding at > 0.6 x belt speed this long
    free_min_deg: float = 40.0
    left_prepose: int = 1            # 1: the left arm moves to the roll's catch wait pose
    lift_dz: float = 0.12            # hinge lift after the release (m)
    lift_s: float = 0.18             # s of the lift
    exit_dx: float = 0.025           # hinge exit: first this far towards the robot (m)
    exit_dy: float = 0.04            # ... and this far upstream (m)
    exit_s: float = 0.14
    rail_y_exit: float = -0.405      # the hinge's exit stays downstream of this (the tall near rail)
    rail_end_clear: float = -1.0     # > -1: the hinge's wait pose keeps the gripper this far downstream of the tall rail's end (m); off: the true-distance checks decide
    approach_margin: float = 0.012   # station clearance of the hinge's approach move (true distance, m)
    descent_margin: float = 0.006    # ... of its descent to the stop pose (m)
    minus_x_mode: str = "tip"        # label facing the robot: "tip" (chosen: tip over the far bottom edge, refside.tipover) | "decline" (pass untouched) | "turn" (to -y, square, released)
    tip: dict | None = None          # overrides of the tip's parameters (refside.tipover.TipParams)
    gate: int = 1                    # +x: no roll after a handover out of square or off the palm (the parcel is let go)
    gate_err_deg: float = 12.0       # ... yaw error above this (8 deg cut 7 readable of 157)
    gate_off: float = 0.10           # ... parcel centre this far beyond the palm across the belt (m)
    stage_right: int = 1             # +x handover: the right palm ends at the roll's staging pose (eo.staged) instead of parking
    stage_min_h: float = 0.10        # ... for parcels at least this tall (flatter ones: parked; staging cost them 6 of 35)
    ik_guard: float = 0.01           # +x hand-over: the two-palm slide and the re-pose stop where a palm's IK misses its
                                     # target by more than this (m); without it the IK's answer is played regardless (on a fast
                                     # belt the left palm drifted 17 cm downstream and the re-pose then pressed it onto the
                                     # belt: a protective stop); 0: off
    front_max: float = 0.17          # +x: no re-pose / no roll when the parcel's front is beyond this (m; the roll's
                                     # catch pose reaches 0.155): the parcel is let go
    stage_mode: str = "pre"          # "top": back off the face, then the staging pose above the parcel's top
                                     # (else the roll's stop pre-pose); "pre": straight from the face to the roll's own
                                     # stop pre-pose (5 cm upstream of the face, 7 cm up): the roll's drop from there is a
                                     # null move, so the parcel is held by one palm for about a second less (it creeps)
    plus_x_aspect: int = 1           # +x: decline parcels the roll cannot take after the turn (W/H above plus_x_max_aspect)
    plus_x_max_aspect: float = 3.0   # ... width across the belt at arrival / height
    squeeze_guard_N: float = 110.0   # summed robot force on the parcel above which the hinge lets go (kept below
                                     # the 300 N damage limit;
                                     # it needs 2 ticks above it after 20 deg of swing and fired once in 300 swings)
    hinge_hold: float = 0.0          # > 0: during the swing the hinge's commanded TCP is corrected by this x (planned -
                                     # measured fingertip position, horizontal) each tick (integral): the arm yields under
                                     # the swing's load and the pivot walks outward (0: off)
    hinge_hold_max: float = 0.03     # ... the correction stays within this (m)
    release_detect: str = "point"    # the hinge's "parcel has left" test: "point": the parcel's material
                                     # point at the fingertips moves downstream at > 0.6 x belt speed; "com":
                                     # the centre of mass does (on slow belts the swing itself moves the centre that
                                     # fast: the hinge lifted at 40-50 degrees of swing)
    follow_deg: float = 0.0          # > 0: past this swing the hinge pad follows the face (following hinge)
    follow_end_deg: float = 86.0     # ... and lets go here (minus the rate lead)
    above_dz: float = 0.10           # approach: above the wait pose by this (m)
    descend_s: float = 0.35          # s of the descent to the wait pose
    v_park: float = 3.2


class BackgroundArm:
    """Drives one arm along precomputed joint values during another
    routine's ticks: a per-tick hook of the executor that writes the arm's
    next planned value into the executor's planned state. A routine that
    builds its rows from the planned state then carries the motion; a
    routine playing precomputed rows holds the arm at its last value (the
    hook waits). As soon as a tick commands the arm to anything else (another
    routine drives it), the hook ends at once and writes nothing more. It
    also ends with its rows."""

    def __init__(self, ex, side, rows):
        self.ex = ex
        self.sl = SL[side]
        self.rows = [np.asarray(r, float) for r in rows]
        self.i = 0
        self.waited = 0
        self.foreign = False
        self.done = not self.rows
        self.prev = np.asarray(ex.q[self.sl], float).copy()
        if self.rows:
            ex.q[self.sl] = self.rows[0]

    def _end(self, ex):
        self.done = True
        if self in ex.hooks:
            ex.hooks.remove(self)

    def __call__(self, ex):
        if self.done:
            return
        cur = ex.q[self.sl]
        if np.abs(cur - self.rows[self.i]).max() < 1e-9:
            self.prev = self.rows[self.i]
            self.i += 1
            if self.i >= len(self.rows):
                self._end(ex)
                return
        elif np.abs(cur - self.prev).max() < 1e-9:
            self.waited += 1                 # precomputed rows: hold, wait
        else:
            self.foreign = True              # someone else drives the arm
            self._end(ex)
            return
        ex.q[self.sl] = self.rows[self.i]

    def finish(self, eo):
        """Play the rest in the foreground (the other arm holding)."""
        while not self.done:
            eo.tick(self.ex.q.copy())


class SideTurn:
    """Quarter turn (counter-clockwise seen from above) on the belt, by the
    right palm as a hinge. Wraps the roll package's EndOver (its executor,
    kinematics and truth helpers)."""

    def __init__(self, rc, eo: EndOver | None = None, params: SideParams | None = None,
                 roll_params: RollParams | None = None, log=None):
        self.rc = rc
        self.cell = rc.cell
        self.eo = eo or EndOver(rc, roll_params or RollParams(), log=log)
        self.ex = rc.ex
        self.kin = rc.kin
        self.p = params or SideParams()
        self.belt = rc.belt
        self.log = log or (lambda *a: None)
        self.trace = []
        self.bg = None

    # ------------------------------------------------------------------
    def box(self, slot):
        c = self.cell
        q = c._slot_qadr[slot]
        pos = c.data.qpos[q:q + 3].copy()
        R = quat_mat(c.data.qpos[q + 3:q + 7])
        p = c.slots[slot].parcel
        return pos, R, np.asarray(p.size, float), float(p.mass)

    def yaw_of(self, R):
        """Yaw of the parcel's body x axis (rad)."""
        return math.atan2(R[1, 0], R[0, 0])

    def front_face(self, slot):
        """(near-front bottom corner, unit direction along the face towards
        +x, outward normal) of the face whose normal is closest to +y."""
        pos, R, size, _ = self.box(slot)
        h = size / 2
        best = None
        for i in range(3):
            for s in (1.0, -1.0):
                n = s * R[:, i]
                if best is None or n[1] > best[0][1]:
                    best = (n, i, s)
        n, i, s = best
        # the two in-plane axes of that face: the most horizontal one spans
        # the face across the belt
        others = [j for j in range(3) if j != i]
        j = max(others, key=lambda j: abs(R[0, j]))
        k = [o for o in others if o != j][0]
        u = R[:, j] * (1.0 if R[0, j] >= 0 else -1.0)
        c = pos + n * h[i] - u * h[j] - R[:, k] * (h[k] if R[2, k] > 0 else -h[k])
        return c, u, n

    _LIBS = {}

    def q_clear(self, side, q7, margin=0.004):
        """True if the arm at joints q7 (the other arm as commanded) is clear
        of the station by margin and free of contact with the robot."""
        row = self.ex.q.copy()
        row[SL[side]] = q7
        return self.path_clear(row[None], side, margin=margin, every=1)

    def ik_near(self, side, pos, R, q_from, k=14, clear=True):
        """IK solution of a pose closest (weighted joint distance) to q_from:
        the shortest joint move there; with clear, solutions that touch the
        robot or come near the station are taken only when there is no other.
        None if unreachable."""
        from .kin import SeedLib, ik_global
        kin = self.kin[side]
        q_from = np.asarray(q_from, float)
        best = None
        q, ok, ep, er = kin.ik(pos, R=R, q0=q_from, iters=300, margin=0.02)
        if ok:
            best = q
        if side not in SideTurn._LIBS:
            SideTurn._LIBS[side] = SeedLib(kin, n=6000, seed=11 if side == "Left" else 12)
        mid = (kin.lo + kin.hi) / 2
        span = (kin.hi - kin.lo) / 2

        def cost(qq):
            c_ = float(np.max(np.abs(qq - q_from))) + 0.2 * float(np.sum(((qq - mid) / span) ** 8))
            if clear and not self.q_clear(side, qq):
                c_ += 10.0
            return c_
        q2 = ik_global(kin, SideTurn._LIBS[side], pos, R=R, k=k, extra_seeds=[q_from],
                       margin=0.02, cost=cost)
        if q2 is not None and (best is None or cost(q2) < cost(best)):
            best = q2
        return best

    def hinge_frame(self):
        if self.p.hinge == "vertical":
            return frame_zx((0.0, 0.0, -1.0), (1.0, 0.0, 0.0))
        t = math.radians(self.p.pin_tilt_deg)
        b = math.radians(self.p.pin_yaw_deg) if self.hinge_closed() else 0.0
        # closed hinge: the tool also yawed by b towards +x, so that the
        # gripper body behind the fingertips lies on the robot's side of the
        # hinge, out of the way of the face that swings about it
        return frame_zx((math.sin(b) * math.cos(t), -math.cos(b) * math.cos(t), -math.sin(t)),
                        (math.cos(b), math.sin(b), 0.0))

    # closed fingers: the two pads side by side at +-4 mm about the TCP x
    # axis, their tip faces 10.7 mm ahead of the TCP along the tool axis
    TIP_CLOSED = 0.0107

    def hinge_closed(self):
        if self.p.hinge != "outward":
            return False
        if self.p.hinge_closed == 2:          # auto: as chosen for this job
            return bool(getattr(self, "_closed_now", False))
        return bool(self.p.hinge_closed)

    def tip_ahead(self):
        """Distance of the hinge's contact faces ahead of the TCP along the
        tool axis (negative: behind it)."""
        return self.TIP_CLOSED if self.hinge_closed() else -PAD_TIP

    def hinge_tcp(self, contact, R):
        """TCP for a hinge contact point. Vertical hinge: the far finger's
        upstream side (its pad is 22 mm wide along the TCP y axis, which
        points upstream) on the face; outward hinge: the far pad's tip (open)
        or the closed fingertips' centre (closed)."""
        if self.p.hinge == "vertical":
            return contact - PAD_X * R[:, 0] - 0.011 * R[:, 1]
        if self.hinge_closed():
            return contact - self.TIP_CLOSED * R[:, 2]
        return contact - PAD_X * R[:, 0] + PAD_TIP * R[:, 2]

    def hinge_contact(self, c, u, y_f, H=None):
        p = self.p
        if p.hinge == "edge":
            # the far pad's tip face centre on the top-front edge, pressed in
            z = self.hinge_frame()[:, 2]
            return np.array([c[0] + p.edge_e * u[0], y_f, self.belt + H]) + p.edge_over * z
        if p.hinge == "vertical":
            return np.array([c[0] + p.pin_e * u[0], y_f + p.pin_gap, self.belt + p.pin_h])
        e = p.pin_e_closed if self.hinge_closed() else p.pin_e_out
        return np.array([c[0] + e * u[0], y_f + p.pin_gap, self.belt + p.pin_h_out])

    def plan_hinge(self, slot):
        """Stop pose of the hinge (TCP, R, joint solutions of the stop, the
        wait pose and the pose above it, and the joint move there) at the most
        upstream face position of the alternatives whose approach is in time
        and clear of the station."""
        p = self.p
        c, u, n = self.front_face(slot)
        R = self.hinge_frame()
        q14 = self.ex.q.copy()
        qR0 = q14[SL["Right"]].copy()
        v = max(self.rc.belt_speed(), 0.02)
        vmx, amx = (p.pin_vmax, p.pin_amax) if v <= p.pin_fast_v else (p.pin_vmax_fast, p.pin_amax_fast)
        T_acc = p.pin_t_acc
        why = []
        H_top = self.top_corner(slot)[4]
        for y_f in sorted(p.pin_y_alt):
            contact = self.hinge_contact(c, u, y_f, H_top)
            tcp = self.hinge_tcp(contact, R)
            wait = tcp.copy()
            wait[1] -= v * (p.pin_t_acc + p.pin_t_dec) / 2
            if p.hinge == "vertical" and wait[1] - 0.045 < p.rail_y_min:
                why.append(f"{y_f}: rail")
                continue
            if p.hinge in ("outward", "edge") and wait[1] < p.rail_y_min_out - 0.03:
                why.append(f"{y_f}: rail")
                continue
            # the open fingers must not straddle the end of the tall near
            # rail upstream of the work window: the whole gripper (its pad
            # tips lead the TCP upstream by about 6 mm) waits downstream of
            # the rail's end with a clearance
            y_rail_end = -float(self.cell.cfg.station.work_window_half)
            if p.hinge in ("outward", "edge") and wait[1] - 0.006 < y_rail_end + p.rail_end_clear:
                why.append(f"{y_f}: rail end")
                continue
            cands = []
            for dz in (p.above_dz, 0.06, 0.03):
                above = wait + np.array([0.0, 0.0, dz])
                for qa in (self.ik_near("Right", above, R, qR0), self.eo.glob_R("Right", above, R, None)):
                    if qa is not None:
                        cands.append((dz, above, qa))
            for dz, above, qa in cands:
                # continuous descent to the wait pose and on to the stop; the
                # whole descent clear of the station (the tall near rail
                # upstream of the work window, the lip) and of the robot
                q = qa.copy()
                ok = True
                rows_d = []
                for s_ in np.linspace(0, 1, 9)[1:]:
                    q, ok_, ep = self.eo.ik_R("Right", above + (wait - above) * s_, R, q, iters=80)
                    ok = ok and ok_
                    rows_d.append(q.copy())
                qw = q.copy()
                for s_ in np.linspace(0, 1, 7)[1:]:
                    q, ok_, ep = self.eo.ik_R("Right", wait + (tcp - wait) * s_, R, q, iters=80)
                    ok = ok and ok_
                    rows_d.append(q.copy())
                if not ok:
                    continue
                Qd = np.repeat(q14[None], len(rows_d), axis=0)
                Qd[:, SL["Right"]] = np.array(rows_d)
                if not self.path_clear(Qd, "Right", margin=p.descent_margin, every=1):
                    why.append(f"{y_f}: descent")
                    continue
                qg = q14.copy()
                qg[SL["Right"]] = qa
                Qm, _, _ = joint_move(q14, qg, vmax=vmx, tmin=0.6, amax=amx)
                if self.eo.path_hits_station(Qm, margin=0.012, every=1) or \
                        not self.path_clear(Qm, "Right", margin=p.approach_margin, every=1):
                    # through a high waypoint above the wait pose, the tool
                    # already turned
                    Qm = None
                    for hz in (0.12, 0.20, 0.30):
                        hp = above + np.array([-0.03, 0.0, hz])
                        qv = self.ik_near("Right", hp, R, qR0)
                        if qv is None:
                            continue
                        qg1 = q14.copy()
                        qg1[SL["Right"]] = qv
                        Q1, _, _ = joint_move(q14, qg1, vmax=vmx, tmin=0.5, amax=amx)
                        # then a straight line down to the pose above the wait
                        # pose (the descent's own start: no joint-space arc
                        # near the rail and the lip)
                        rows2 = self._cart_rows("Right", above, R, max(0.25, hz / 0.6), q0=qv)
                        if rows2 is None:
                            Q2, _, _ = joint_move(Q1[-1], qg, vmax=vmx, tmin=0.3, amax=amx)
                        else:
                            Q2 = np.repeat(Q1[-1][None], len(rows2), axis=0)
                            Q2[:, SL["Right"]] = np.array(rows2)
                        Qc = np.vstack([Q1, Q2])
                        if not self.eo.path_hits_station(Qc, margin=0.012, every=1) and \
                                self.path_clear(Qc, "Right", margin=p.approach_margin, every=1):
                            if rows2 is not None and np.abs(np.array(rows2[-1]) - qa).max() > 0.05:
                                # another arm posture at the pose above: its own
                                # descent must be clear too
                                qd = np.array(rows2[-1])
                                rows_e = []
                                ok_e = True
                                for s_ in np.linspace(0, 1, 9)[1:]:
                                    qd, ok_, ep = self.eo.ik_R("Right", above + (wait - above) * s_, R, qd, iters=80)
                                    ok_e = ok_e and ok_
                                    rows_e.append(qd.copy())
                                qw_e = qd.copy()
                                for s_ in np.linspace(0, 1, 7)[1:]:
                                    qd, ok_, ep = self.eo.ik_R("Right", wait + (tcp - wait) * s_, R, qd, iters=80)
                                    ok_e = ok_e and ok_
                                    rows_e.append(qd.copy())
                                Qe = np.repeat(q14[None], len(rows_e), axis=0)
                                Qe[:, SL["Right"]] = np.array(rows_e)
                                if not ok_e or not self.path_clear(Qe, "Right", margin=p.descent_margin, every=1):
                                    continue
                                q, qw = qd, qw_e
                            Qm = Qc
                            break
                    if Qm is None:
                        why.append(f"{y_f}: path")
                        continue
                y_meet = wait[1] + v * T_acc / 2 + self.tip_ahead() * R[1, 2] + p.pin_gap
                t_start = (y_meet - c[1]) / v - T_acc
                if t_start < len(Qm) * DT + p.descend_s * (dz / p.above_dz) + 0.1:
                    why.append(f"{y_f}: late")
                    continue
                return dict(y_face=y_f, contact=contact, tcp=tcp, R=R, q=q, wait=wait, qw=qw,
                            above=above, qa=qa, Qm=Qm, descend_s=p.descend_s * (dz / p.above_dz),
                            y_meet=y_meet)
            why.append(f"{y_f}: ik")
        raise PlanFail("hinge: no stop pose (" + ", ".join(why) + ")")

    # ------------------------------------------------------------------
    def left_wait_pose(self, slot, swing_deg=90.0):
        """Predicted roll catch wait pose of the left palm for the parcel
        after the quarter turn (the roll's own catch formulas)."""
        from .endover import PAD_X as PX
        rp = self.eo.p
        pos, R, size, mass = self.box(slot)
        from .geom import aligned_dims
        q = self.cell.data.qpos[self.cell._slot_qadr[slot] + 3:self.cell._slot_qadr[slot] + 7]
        W, L, H = aligned_dims(size, q)
        W2, L2 = L, W
        v = max(self.rc.belt_speed(), 0.02)
        y_face = float(np.clip(rp.catch_y_face, -0.15 + L2, 0.155))
        psi_deg = rp.catch_psi_deg
        for cand in (rp.catch_psi_deg, 30.0, 20.0, 10.0, 0.0):
            dz_c = PX * math.sin(math.radians(cand))
            if rp.catch_lo + 2 * dz_c + 0.012 <= H:
                psi_deg = cand
                break
        psi_c = math.radians(psi_deg)
        tilt_c = math.radians(rp.catch_tilt_deg)
        c, u, n = self.front_face(slot)
        x_c2 = c[0] + self.p.pin_e + W2 / 2 * 0 + L / 2   # parcel centre x after the turn
        x_hi = 0.33
        low = self.p.catch_mode == "low" or self.p.prepose_low
        if low:
            # the handover's low catch (tilted palm, pads side by side)
            m0 = self.p.catch_mode
            self.p.catch_mode = "low"
            try:
                psi_c, tilt_c, z_rel = self.catch_geom(H, False)
            finally:
                self.p.catch_mode = m0
            x_c2 += self.p.prepose_dx
            x_hi = self.p.catch_x_hi_tilted
            if self.p.repose_mode == "glide":
                y_face = self.y_face_low(v)
        Rl = palm_R("Left", psi_c, 0.0, tilt_c)
        x_lo = max(0.22, 0.150 + PX * math.cos(psi_c) + rp.lip_clear)
        x_s = float(np.clip(x_c2, x_lo, x_hi))
        dz = PX * abs(math.sin(psi_c))
        z_lo = max(rp.catch_zmin, rp.catch_lo + dz)
        z_s = self.belt + float(np.clip(rp.catch_mid_frac * H, z_lo, max(z_lo, H - dz - 0.012)))
        if low:
            z_s = self.belt + z_rel
        pad_off = -PAD_TIP * Rl[:, 2]
        stop = np.array([x_s, y_face + rp.catch_gap, z_s]) - pad_off
        wait = stop.copy()
        if low:
            wait[1] = stop[1] - v * (self.p.own_t_acc + self.p.own_t_dec) / 2
        else:
            wait[1] = stop[1] - v * (rp.catch_t_acc + rp.catch_t_dec) / 2
        qw = self.ik_near("Left", wait, Rl, self.ex.q[SL["Left"]])
        if qw is None:
            qw = self.eo.glob_R("Left", wait, Rl, self.ex.q[SL["Left"]])
        return qw

    # ------------------------------------------------------------------
    def _tick(self, row, forces=None):
        """eo.tick with the scheduled left-arm motion (self._ql: rows of 14,
        consumed one per tick) applied to the left arm's joints."""
        ql = getattr(self, "_ql", None)
        if ql is not None and len(ql):
            # rows planned earlier for both arms would put the left arm back:
            # past its schedule's end it holds the schedule's last value
            row = np.array(row, float).copy()
            row[SL["Left"]] = ql[min(self._ql_k, len(ql) - 1)][SL["Left"]]
            self._ql_k += 1
        self.eo.tick(row, forces)

    def set_grip(self, side, cmd):
        g = list(self.eo.grip)
        g[0 if side == "Left" else 1] = float(cmd)
        self.eo.grip = tuple(g)

    def _hold(self, seconds, forces=None):
        for _ in range(max(1, int(round(seconds / DT)))):
            self._tick(self.ex.q.copy(), forces)

    # ------------------------------------------------------------------
    def _side_geoms(self):
        """{geom id: side} of the robot's collision geoms ("Left" / "Right"
        by the arm subtree they belong to; the column and head: none)."""
        if getattr(self, "_gside", None) is None:
            import mujoco
            m = self.cell.model
            roots = {"Left": m.body("Left_Shoulder_Base_Link").id,
                     "Right": m.body("Right_Shoulder_Base_Link").id}
            gs = {}
            for g in self.cell.robot_geoms:
                b = int(m.geom_bodyid[int(g)])
                while b > 0:
                    hit = [s_ for s_, r in roots.items() if r == b]
                    if hit:
                        gs[int(g)] = hit[0]
                        break
                    b = int(m.body_parentid[b])
            self._gside = gs
        return self._gside

    def contact_side(self, slot, side):
        """Summed normal force of one arm's contacts on the parcel."""
        import mujoco
        gs = self._side_geoms()
        c = self.cell
        m, d = c.model, c.data
        f6 = np.zeros(6)
        f = 0.0
        for k in range(d.ncon):
            con = d.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if gs.get(a) == side and c.parcel_geom_slot[b] == slot:
                    if con.exclude == 0 and con.efc_address >= 0:
                        mujoco.mj_contactForce(m, d, k, f6)
                        f += abs(float(f6[0]))
        return f

    def contact_hinge(self, slot):
        return self.contact_side(slot, "Right")

    def contact_any(self, slot):
        f = 0.0
        for c in self.cell.truth.contacts(kinds=("robot-parcel",)):
            if c.slot == slot:
                f += c.force_normal
        return f

    def turn(self, slot):
        """The quarter turn. Entry: the parcel approaching on the belt,
        upstream of y = -0.5, justified (near face at the near belt edge
        within 1 cm, yaw a multiple of 90 deg within +-6 deg), label on its
        world +x or -x face; both arms at the park posture. Exit: the parcel
        riding downstream on the belt, released, turned counter-clockwise by
        about 55-65 deg (the roll's catch squares it to 90); the right arm
        parked; the left arm at (or moving to) the roll's catch wait pose.
        Returns a dict (ok, swing_deg, reason, timing, forces)."""
        p = self.p
        eo = self.eo
        self.bg = None
        out = dict(ok=False, reason="", t0=self.cell.time)
        if not self.cell.slots[slot].active:
            out["reason"] = "parcel gone"
            return out
        pos0, R0, size, mass = self.box(slot)
        mg = mass * G
        yaw0 = self.yaw_of(R0)
        out["label0"] = label_dir(self.cell.truth.parcel_state(slot).label_normal)[0]
        plan = None
        self._closed_now = False
        if p.hinge_closed == 2 and p.hinge == "outward":
            # auto: the open hinge (its far pad the pivot, the gripper body
            # on the robot's side of the hinge) where its approach and stop
            # pose are clear; else the closed fingertips
            try:
                plan = self.plan_hinge(slot)
                out["hinge_kind"] = "open"
            except PlanFail as e:
                out["open_fail"] = str(e)[:200]
                self._closed_now = True
        if plan is None:
            if self.hinge_closed():
                # close the right gripper first (the plan's checks then see
                # the closed fingers)
                self.set_grip("Right", 1.0)
                self._hold(p.close_s)
                out["hinge_kind"] = "closed"
            try:
                plan = self.plan_hinge(slot)
            except PlanFail as e:
                out["reason"] = str(e)
                self.set_grip("Right", 0.0)
                return out
        out["pin_y"] = plan["y_face"]
        v = max(self.rc.belt_speed(), 0.02)
        Fpin = p.pin_f + p.pin_f_mg * mg
        Fz = 0.0
        if p.hinge == "edge":
            Fpin = p.edge_f_margin * 0.9 * mg / max(0.2, 1.0 - 0.9 * p.edge_k)
            Fz = p.edge_k * Fpin
        out["hinge_force_plan"] = [round(Fpin, 1), round(Fz, 1)]
        # 0. the left arm moves to the catch's wait pose from the start
        #    (scheduled, one row per tick, alongside the right arm's work)
        self._ql, self._ql_k = None, 0
        if p.left_prepose:
            try:
                qL_goal = self.left_wait_pose(slot)
            except Exception:  # noqa: BLE001
                qL_goal = None
            if qL_goal is not None:
                qg = self.ex.q.copy()
                qg[SL["Left"]] = qL_goal
                self._ql, _, _ = joint_move(self.ex.q, qg, vmax=2.4, tmin=1.0)
        # 1. approach: right arm to a pose above the wait pose (joint move,
        #    checked against the station), then straight down to it
        Qm = plan["Qm"]
        T_acc, T_d = p.pin_t_acc, p.pin_t_dec
        y_meet = plan["y_meet"]
        eo.phase = "side_approach"
        for row in Qm:
            self._tick(row)
        self._move_right_cart(plan["wait"], plan["R"], plan["descend_s"])
        c, u, n = self.front_face(slot)
        c, u, n = self.front_face(slot)
        t_left = (y_meet - c[1]) / v - T_acc
        if t_left > DT:
            self._hold(t_left)
        # 2. speed-matched stop
        eo.phase = "side_stop"
        kR = self.kin["Right"]
        qq = self.ex.q[SL["Right"]].copy()
        n1, n2 = int(round(T_acc / DT)), int(round(T_d / DT))
        yy = plan["wait"][1]
        R = plan["R"]
        for i in range(n1 + n2):
            t = (i + 1) * DT
            vp = v * t / T_acc if i < n1 else v * max(0.0, 1 - (t - T_acc) / T_d)
            yy += vp * DT
            tgt = plan["tcp"].copy()
            tgt[1] = min(yy, plan["tcp"][1])
            qn, ok, ep = eo.ik_R("Right", tgt, R, qq, iters=60)
            qq = qn
            row = self.ex.q.copy()
            row[SL["Right"]] = qn
            lvl = min(1.0, max(0.0, (i - n1 + 2) / 10.0))
            self._tick(row, {"Right": np.array([0.0, -Fpin * lvl, -Fz * lvl])})
        out["t_stop"] = self.cell.time
        # 3. swing: hold the hinge (the left arm goes on to its wait pose)
        eo.phase = "side_swing"
        QL = None
        t_sw = self.cell.time
        pm0_, Rm0_ = self.cell.tcp_pose("Right")
        hold_tip = np.asarray(pm0_, float) + self.tip_ahead() * np.asarray(Rm0_, float)[:, 2]
        hold_tcp = self.kin["Right"].fk(self.ex.q[SL["Right"]])[0].copy()
        q_hold = self.ex.q[SL["Right"]].copy()
        corr = np.zeros(3)
        lost = 0
        moving = 0
        n_guard = 0
        follow_pt = None
        swing = 0.0
        swing_max = 0.0
        f_max = 0.0
        k = 0
        released = False
        while self.cell.time - t_sw < p.swing_timeout_s and not self.cell.done:
            if not self.cell.slots[slot].active:
                out["reason"] = "parcel gone"
                break
            pos, Rb, _, _ = self.box(slot)
            swing = math.degrees(math.atan2(math.sin(self.yaw_of(Rb) - yaw0),
                                            math.cos(self.yaw_of(Rb) - yaw0)))
            swing_max = max(swing_max, swing)
            fh = self.contact_hinge(slot)
            f_max = max(f_max, fh)
            lost = lost + 1 if fh < 0.05 else 0
            n_guard = n_guard + 1 if fh > p.squeeze_guard_N else 0
            if n_guard >= 2 and swing > 20.0:
                out["guard"] = round(fh, 1)
                break
            ps_ = self.cell.truth.parcel_state(slot)
            vy = float(ps_.linvel[1])
            if p.release_detect == "point":
                ph_, Rh_ = self.cell.tcp_pose("Right")
                p_tip = np.asarray(ph_, float) + self.tip_ahead() * np.asarray(Rh_, float)[:, 2]
                v_pt = np.asarray(ps_.linvel, float) + np.cross(np.asarray(ps_.angvel, float), p_tip - pos)
                vy = float(v_pt[1])
            moving = moving + 1 if vy > 0.6 * v else 0
            self.trace.append((round(self.cell.time, 2), round(swing, 1), round(fh, 1),
                               round(float(pos[1]), 3), round(vy, 3)))
            if swing > p.free_min_deg and (moving >= p.moving_ticks or lost >= p.lost_ticks):
                released = True
                break
            if p.release_time_need > 0 and swing > p.free_min_deg and out["label0"] == "+x":
                fy = self._front_max_y(slot)
                if (0.15 - fy) / v < p.release_time_need:
                    out["release_by_time"] = round(swing, 1)
                    break
            if p.release_deg > 0 and swing >= p.release_deg:
                break
            if p.hinge == "edge":
                w_z = float(self.cell.truth.parcel_state(slot).angvel[2])
                if swing + math.degrees(w_z) * p.release_lead_s >= p.release_edge_deg:
                    break
            row = self.ex.q.copy()
            k += 1
            F = None if lost else {"Right": np.array([0.0, -Fpin, -Fz])}
            if p.hinge_hold > 0 and not lost and p.follow_deg <= 0:
                pm_, Rm_ = self.cell.tcp_pose("Right")
                tip_m = np.asarray(pm_, float) + self.tip_ahead() * np.asarray(Rm_, float)[:, 2]
                e_ = hold_tip - tip_m
                e_[2] = 0.0
                corr = np.clip(corr + p.hinge_hold * e_, -p.hinge_hold_max, p.hinge_hold_max)
                qn_, ok_, ep_ = eo.ik_R("Right", hold_tcp + corr, R, q_hold, iters=40)
                stp_ = np.abs(qn_ - q_hold).max()
                if stp_ > eo.p.step_max:
                    qn_ = q_hold + (qn_ - q_hold) * (eo.p.step_max / stp_)
                q_hold = qn_
                row[SL["Right"]] = q_hold
            if p.follow_deg > 0 and swing >= p.follow_deg:
                # following hinge: past the pad's friction angle the parcel
                # slides along the pad; the pad goes with the face's material
                # point it touches, pressing along the face's inward normal,
                # so the parcel keeps turning about it (the belt's moment)
                if follow_pt is None:
                    tcp_h, R_h = self.kin["Right"].fk(self.ex.q[SL["Right"]])
                    tip = tcp_h + PAD_X * R_h[:, 0] - PAD_TIP * R_h[:, 2]
                    follow_pt = Rb.T @ (tip - pos)
                    follow_R = R_h
                    qf = self.ex.q[SL["Right"]].copy()
                tip_now = pos + Rb @ follow_pt
                tgt = tip_now - PAD_X * follow_R[:, 0] + PAD_TIP * follow_R[:, 2]
                qn, ok_, ep_ = eo.ik_R("Right", tgt, follow_R, qf, iters=40)
                stp = np.abs(qn - qf).max()
                if stp > eo.p.step_max:
                    qn = qf + (qn - qf) * (eo.p.step_max / stp)
                qf = qn
                row[SL["Right"]] = qf
                nf_, _, _ = self.lead_face(slot) if False else (None, None, None)
                ang = math.radians(swing)
                n_in = np.array([math.sin(ang), -math.cos(ang), 0.0])
                F = None if lost else {"Right": n_in * Fpin}
                w_z = float(self.cell.truth.parcel_state(slot).angvel[2])
                if swing + math.degrees(w_z) * p.release_lead_s >= p.follow_end_deg:
                    break
            self._tick(row, F)
        out["swing_at_release"] = round(swing, 1)
        out["swing_max"] = round(swing_max, 1)
        out["hinge_force_max"] = round(f_max, 1)
        out["t_release"] = self.cell.time
        out["slid_off"] = released
        if p.catch_early and self.cell.slots[slot].active and out["label0"] == "+x" and p.plus_x_handover != "free":
            # the left arm sets off for the low catch's wait pose now (rows
            # consumed by _tick during the hinge's exit below); the catch then
            # plans its remaining short move from wherever the arm is
            try:
                qw_ = self.low_catch_wait_q(slot)
            except Exception:  # noqa: BLE001
                qw_ = None
            if qw_ is not None:
                qg_ = self.ex.q.copy()
                qg_[SL["Left"]] = qw_
                self._ql, _, _ = joint_move(self.ex.q, qg_, vmax=p.own_vmax, tmin=0.25, amax=p.own_amax)
                self._ql_k = 0
                out["catch_early"] = round(float(np.abs(qg_ - self.ex.q).max()), 3)
        # 4. the hinge lifts clear (straight up, a little towards the robot);
        #    the right arm then waits there, above the belt and upstream of
        #    the parcel. The left arm's scheduled move continues in the
        #    catch that follows (the remaining rows are dropped here: the
        #    catch plans its own short move from wherever the arm is).
        eo.phase = "side_exit"
        pr, Rr_ = self.cell.tcp_pose("Right")
        # first away from the parcel (upstream and towards the robot: the
        # parcel has swung downstream past the hinge, beside its finger),
        # then up
        # (upstream only as far as the tall near rail upstream of the work
        # window allows: the pads stay downstream of y = rail_y_exit)
        if self.hinge_closed():
            # closed fingertips: nothing beside the parcel; a little upstream
            # (the fingertips stay 12 mm downstream of the tall rail's end)
            # and up, no move towards the robot (the rail's line)
            y_min = -float(self.cell.cfg.station.work_window_half) + 0.012 + 0.015
            dy_ex = float(np.clip(pr[1] - y_min, 0.0, 0.02))
            self._move_right_cart(pr + np.array([0.0, -dy_ex, 0.025]), Rr_, p.exit_s)
        else:
            dy_ex = float(np.clip(pr[1] - p.rail_y_exit, 0.0, p.exit_dy))
            self._move_right_cart(pr + np.array([-p.exit_dx, -dy_ex, 0.015]), Rr_, p.exit_s)
        pr, Rr_ = self.cell.tcp_pose("Right")
        # lift clear of the parcel's top (tall parcels) as well
        top = self.belt + self.top_corner(slot)[4] if self.cell.slots[slot].active else self.belt
        dz_l = max(p.lift_dz, top + 0.05 - float(pr[2]))
        self._move_right_cart(pr + np.array([0.0, 0.0, dz_l]), Rr_, max(p.lift_s, dz_l / 0.7))
        self.set_grip("Right", 0.0)
        self._ql = None
        pos, Rb, _, _ = self.box(slot) if self.cell.slots[slot].active else (None, None, None, None)
        if Rb is not None:
            sw = math.degrees(math.atan2(math.sin(self.yaw_of(Rb) - yaw0), math.cos(self.yaw_of(Rb) - yaw0)))
            out["swing_end"] = round(sw, 1)
            out["label_after"] = label_dir(self.cell.truth.parcel_state(slot).label_normal)[0]
        out["t_end"] = self.cell.time
        out["ok"] = bool(Rb is not None and 45.0 < out.get("swing_end", 0.0) < 135.0)
        if not out["ok"] and not out["reason"]:
            out["reason"] = f"swing {out.get('swing_end')}"
        return out

    # ==================================================================
    # press pivot: a finger pressed on the top face near the near-front
    # corner; its friction holds the parcel against the belt in any
    # direction, the belt swings the parcel about it (no slide-off); the
    # finger lifts at 90 degrees
    # ==================================================================
    def top_corner(self, slot):
        """(near-front top corner, across-belt unit vector u, front normal n,
        up unit vector, height H)."""
        pos, R, size, _ = self.box(slot)
        c, u, n = self.front_face(slot)
        k = int(np.argmax(np.abs(R[2, :])))
        up = R[:, k] * (1.0 if R[2, k] > 0 else -1.0)
        H = float(size[k])
        return c + up * H, u, n, up, H

    def press_point(self, slot):
        p = self.p
        ct, u, n, up, H = self.top_corner(slot)
        return ct + p.press_e1 * u - p.press_e2 * n, H

    def press_frame(self):
        return frame_zx((0.0, 0.0, -1.0), (1.0, 0.0, 0.0))

    def press_tcp(self, P):
        """TCP with the far finger's pad tip at P (vertical tool, pad line
        across the belt: the near finger is in the air beyond the near
        edge)."""
        R = self.press_frame()
        return P - PAD_X * R[:, 0] + PAD_TIP * R[:, 2]

    def plan_press(self, slot):
        p = self.p
        P0, H = self.press_point(slot)
        R = self.press_frame()
        q14 = self.ex.q.copy()
        qR0 = q14[SL["Right"]].copy()
        v = max(self.rc.belt_speed(), 0.02)
        T_acc, T_trk, T_dec = p.press_t_acc, p.press_t_desc + p.press_t_ramp, p.press_t_dec
        why = []
        for y_l in sorted(p.press_y_alt):
            Pl = np.array([P0[0], y_l, P0[2]])
            tcp_l = self.press_tcp(Pl) + np.array([0.0, 0.0, -p.press_over])
            y_w = y_l - v * (T_acc / 2 + T_trk + T_dec / 2)
            wait = self.press_tcp(np.array([P0[0], y_w, P0[2]])) + np.array([0.0, 0.0, p.press_clear])
            if wait[1] < -0.60:
                why.append(f"{y_l}: wait far")
                continue
            qw = self.ik_near("Right", wait, R, qR0)
            if qw is None:
                why.append(f"{y_l}: ik wait")
                continue
            # the whole tracked path must be reachable (continuous IK)
            q = qw.copy()
            ok = True
            for s_ in np.linspace(0, 1, 13)[1:]:
                tgt = wait + (tcp_l - wait) * s_
                q, ok_, ep = self.eo.ik_R("Right", tgt, R, q, iters=80)
                ok = ok and ok_
            if not ok:
                why.append(f"{y_l}: ik path")
                continue
            # above the wait pose first (the joint move ends high), then down
            above = wait + np.array([0.0, 0.0, 0.08])
            qa = self.ik_near("Right", above, R, qR0)
            if qa is None:
                why.append(f"{y_l}: ik above")
                continue
            qg = q14.copy()
            qg[SL["Right"]] = qa
            Qm, _, _ = joint_move(q14, qg, vmax=p.pin_vmax, tmin=0.6)
            # the finger starts when the press point is T_acc/2 x v short of
            # the wait pose's y
            t_start = (y_w - P0[1]) / v - T_acc / 2
            if t_start < len(Qm) * DT + 0.30 + 0.1:
                why.append(f"{y_l}: late")
                continue
            if self.eo.path_hits_station(Qm, margin=0.01, every=2) or \
                    not self.path_clear(Qm, "Right", margin=0.008):
                why.append(f"{y_l}: path")
                continue
            return dict(y_land=y_l, wait=wait, qw=qw, above=above, qa=qa, Qm=Qm, R=R, H=H,
                        tcp_land=tcp_l, y_w=y_w)
        raise PlanFail("press: no landing (" + ", ".join(why) + ")")

    def turn_press(self, slot):
        """Quarter turn counter-clockwise about a pressed finger. Entry and
        exit as turn() (the parcel leaves square to the belt, its new near
        face within about 2 cm of the near belt edge)."""
        p = self.p
        eo = self.eo
        out = dict(ok=False, reason="", t0=self.cell.time, method="press")
        if not self.cell.slots[slot].active:
            out["reason"] = "parcel gone"
            return out
        pos0, R0, size, mass = self.box(slot)
        mg = mass * G
        yaw0 = self.yaw_of(R0)
        out["label0"] = label_dir(self.cell.truth.parcel_state(slot).label_normal)[0]
        try:
            plan = self.plan_press(slot)
        except PlanFail as e:
            out["reason"] = str(e)
            return out
        out["y_land"] = plan["y_land"]
        v = max(self.rc.belt_speed(), 0.02)
        R = plan["R"]
        N = p.press_f + p.press_f_mg * mg
        drag = 0.9 * (mg + N)
        # 1. approach: high above the wait pose, then down to it
        eo.phase = "press_approach"
        for row in plan["Qm"]:
            eo.tick(row)
        self._move_right_cart(plan["wait"], R, 0.30)
        # 2. wait for the parcel, then a speed-matched landing on its top
        P, H = self.press_point(slot)
        T_acc = p.press_t_acc
        t_left = (plan["y_w"] - P[1]) / v - T_acc / 2
        if t_left > DT:
            eo.hold(t_left)
        eo.phase = "press_land"
        q = self.ex.q[SL["Right"]].copy()
        n_acc = int(round(T_acc / DT))
        n_desc = int(round(p.press_t_desc / DT))
        n_ramp = int(round(p.press_t_ramp / DT))
        n_dec = int(round(p.press_t_dec / DT))
        y_f = plan["wait"][1]
        vy = 0.0
        total = n_acc + n_desc + n_ramp + n_dec
        z_top = None
        for i in range(total):
            if not self.cell.slots[slot].active:
                out["reason"] = "parcel gone"
                return out
            Pn, H = self.press_point(slot)
            if i < n_acc:
                vy = v * (i + 1) / n_acc
                dz = p.press_clear
            elif i < n_acc + n_desc:
                vy = v
                a = (i - n_acc + 1) / n_desc
                dz = p.press_clear * (1 - a) - p.press_over * a
            elif i < n_acc + n_desc + n_ramp:
                vy = v
                dz = -p.press_over
            else:
                vy = v * max(0.0, 1 - (i - n_acc - n_desc - n_ramp + 1) / n_dec)
                dz = -p.press_over
            y_f += vy * DT
            tgt = self.press_tcp(np.array([Pn[0], y_f, Pn[2]])) + np.array([0.0, 0.0, dz])
            if i >= n_acc + n_desc:
                # x and z follow the parcel's top while pressing
                tgt[0] = self.press_tcp(Pn)[0]
            q, ok, ep = eo.ik_R("Right", tgt, R, q, iters=60)
            row = self.ex.q.copy()
            row[SL["Right"]] = q
            lvl = 0.0
            if i >= n_acc + n_desc:
                lvl = min(1.0, (i - n_acc - n_desc + 1) / max(1, n_ramp))
            fy = -drag * lvl if i >= n_acc + n_desc + n_ramp else 0.0
            eo.tick(row, {"Right": np.array([0.0, fy * 0.8, -N * lvl])})
        out["t_press"] = self.cell.time
        # 3. hold the pivot: the belt swings the parcel
        eo.phase = "press_swing"
        qL_goal = None
        if p.left_prepose:
            try:
                qL_goal = self.left_wait_pose_press(slot)
            except Exception:  # noqa: BLE001
                qL_goal = None
        QL = None
        if qL_goal is not None:
            qg = self.ex.q.copy()
            qg[SL["Left"]] = qL_goal
            QL, _, _ = joint_move(self.ex.q, qg, vmax=2.0, tmin=1.0)
        t_sw = self.cell.time
        k = 0
        swing = 0.0
        w_z = 0.0
        f_max = 0.0
        hold_tcp = self.cell.tcp_pose("Right")[0].copy()
        q_hold = self.ex.q[SL["Right"]].copy()
        released = False
        while self.cell.time - t_sw < p.press_timeout_s and not self.cell.done:
            if not self.cell.slots[slot].active:
                out["reason"] = "parcel gone"
                break
            pos, Rb, _, _ = self.box(slot)
            swing = math.degrees(math.atan2(math.sin(self.yaw_of(Rb) - yaw0),
                                            math.cos(self.yaw_of(Rb) - yaw0)))
            w_z = float(self.cell.truth.parcel_state(slot).angvel[2])
            fh = self.contact_hinge(slot)
            f_max = max(f_max, fh)
            self.trace.append((round(self.cell.time, 2), round(swing, 1), round(fh, 1),
                               round(float(pos[0]), 3), round(float(pos[1]), 3), round(w_z, 2)))
            if swing + math.degrees(w_z) * p.press_lead_s >= p.press_release_deg:
                released = True
                break
            row = self.ex.q.copy()
            if QL is not None and k < len(QL):
                row[SL["Left"]] = QL[k][SL["Left"]]
            k += 1
            eo.tick(row, {"Right": np.array([0.0, -0.8 * drag, -N])})
        out["swing_at_release"] = round(swing, 1)
        out["w_at_release"] = round(math.degrees(w_z), 1)
        out["press_force_max"] = round(f_max, 1)
        out["t_release"] = self.cell.time
        out["released"] = released
        if not released and not out["reason"]:
            out["reason"] = f"press: no swing to {p.press_release_deg} ({swing:.0f})"
        # 4. lift: fast off the top, then higher; the left arm finishes its
        #    move to the catch's wait pose
        eo.phase = "press_exit"
        pr, Rr_ = self.cell.tcp_pose("Right")
        self._move_right_cart(pr + np.array([0.0, 0.0, p.press_lift_dz]), Rr_, p.press_lift_s, QL, k)
        k += int(math.ceil(p.press_lift_s / DT))
        pr, Rr_ = self.cell.tcp_pose("Right")
        self._move_right_cart(pr + np.array([-0.02, 0.0, 0.06]), Rr_, 0.25, QL, k)
        k += int(math.ceil(0.25 / DT))
        if QL is not None and k < len(QL):
            for r_ in QL[k:]:
                row = self.ex.q.copy()
                row[SL["Left"]] = r_[SL["Left"]]
                eo.tick(row)
        if self.cell.slots[slot].active:
            pos, Rb, _, _ = self.box(slot)
            sw = math.degrees(math.atan2(math.sin(self.yaw_of(Rb) - yaw0), math.cos(self.yaw_of(Rb) - yaw0)))
            out["swing_end"] = round(sw, 1)
            out["x_end"] = round(float(pos[0]), 3)
            out["label_after"] = label_dir(self.cell.truth.parcel_state(slot).label_normal)[0]
        out["t_end"] = self.cell.time
        out["ok"] = bool(released and 60.0 < out.get("swing_end", 0.0) < 120.0)
        if not out["ok"] and not out["reason"]:
            out["reason"] = f"swing {out.get('swing_end')}"
        return out

    def left_wait_pose_press(self, slot):
        """The roll's catch wait pose for the parcel after a press turn (its
        centre then lies at the pivot's x plus half its former length)."""
        from .endover import PAD_X as PX
        from .geom import aligned_dims
        rp = self.eo.p
        pos, R, size, mass = self.box(slot)
        q = self.cell.data.qpos[self.cell._slot_qadr[slot] + 3:self.cell._slot_qadr[slot] + 7]
        W, L, H = aligned_dims(size, q)
        W2, L2 = L, W
        v = max(self.rc.belt_speed(), 0.02)
        y_face = float(np.clip(rp.catch_y_face, -0.15 + L2, 0.155))
        psi_deg = rp.catch_psi_deg
        for cand in (rp.catch_psi_deg, 30.0, 20.0, 10.0, 0.0):
            dz_c = PX * math.sin(math.radians(cand))
            if rp.catch_lo + 2 * dz_c + 0.012 <= H:
                psi_deg = cand
                break
        psi_c = math.radians(psi_deg)
        Rl = palm_R("Left", psi_c, 0.0, math.radians(rp.catch_tilt_deg))
        ct, u, n, up, H_ = self.top_corner(slot)
        x_c2 = ct[0] + self.p.press_e1 - self.p.press_e2 + L / 2
        x_lo = max(0.22, 0.150 + PX * math.cos(psi_c) + rp.lip_clear)
        x_s = float(np.clip(x_c2, x_lo, 0.33))
        dz = PX * abs(math.sin(psi_c))
        z_lo = max(rp.catch_zmin, rp.catch_lo + dz)
        z_s = self.belt + float(np.clip(rp.catch_mid_frac * H, z_lo, max(z_lo, H - dz - 0.012)))
        pad_off = -PAD_TIP * Rl[:, 2]
        stop = np.array([x_s, y_face + rp.catch_gap, z_s]) - pad_off
        wait = stop.copy()
        wait[1] = stop[1] - v * (rp.catch_t_acc + rp.catch_t_dec) / 2
        return self.eo.glob_R("Left", wait, Rl, self.ex.q[SL["Left"]])

    # ==================================================================
    # flush catch + square-and-justify: the left palm takes the turned
    # parcel's leading face flush (palm yawed to it), at the roll's catch
    # line; then, the parcel held against it by the belt (the belt slips
    # under the parcel, so the parcel moves on it like on a viscous floor),
    # the palm turns it square and slides it to the near belt edge, and ends
    # in the roll's own catch pose (eo.catch_pose): the "caught" state
    # ==================================================================
    def lead_face(self, slot):
        """(outward normal n, body axis index, half size along it) of the
        face whose normal is closest to +y."""
        pos, R, size, _ = self.box(slot)
        best = None
        for i in range(3):
            for sg in (1.0, -1.0):
                n = sg * R[:, i]
                if best is None or n[1] > best[0][1]:
                    best = (n, i, sg)
        n, i, sg = best
        return n, i, float(size[i]) / 2

    def catch_geom(self, H, low_stop=False):
        """(psi, tilt, z_s above the belt) of the roll's catch for height H
        (the roll package's formulas; low_stop: the reverse roll's low stop;
        p.catch_mode "low": the low-face catch, pads side by side across the
        belt, tool tilted down, for every height)."""
        rp = self.eo.p
        if not low_stop and self.p.catch_mode == "low":
            tilt = math.radians(getattr(rp, "low_tilt_deg", 30.0))
            zmin = getattr(rp, "low_zmin", 0.018)
            return 0.0, tilt, float(np.clip(0.5 * H, zmin, max(zmin, H - 0.012)))
        if low_stop:
            return 0.0, math.radians(getattr(rp, "rd_stop_tilt_deg", 30.0)), \
                float(min(getattr(rp, "rd_stop_z", 0.022), max(0.012, H - 0.015)))
        psi_deg = rp.catch_psi_deg
        for cand in (rp.catch_psi_deg, 30.0, 20.0, 10.0, 0.0):
            if rp.catch_lo + 2 * PAD_X * math.sin(math.radians(cand)) + 0.012 <= H:
                psi_deg = cand
                break
        psi = math.radians(psi_deg)
        dz = PAD_X * abs(math.sin(psi))
        z_lo = max(rp.catch_zmin, rp.catch_lo + dz)
        z_s = float(np.clip(rp.catch_mid_frac * H, z_lo, max(z_lo, H - dz - 0.012)))
        return psi, math.radians(rp.catch_tilt_deg), z_s

    def catch_flush(self, slot, low_stop=False, timeout=14.0):
        """Flush catch of a turned parcel and square-and-justify. Entry: the
        parcel riding on the belt upstream of the catch line, turned (its
        leading face within about 40 deg of square), the left arm free.
        Exit: the parcel at rest against the left palm in the roll's catch
        pose (front face at the catch line, square to the belt within about
        2 deg, near face at the near belt edge within about 1 cm), held;
        eo.catch_pose set (the roll's own format). Raises PlanFail."""
        p = self.p
        eo = self.eo
        rp = eo.p
        pos, R, size, mass = self.box(slot)
        mg = mass * G
        v = max(self.rc.belt_speed(), 0.02)
        n, ia, h_n = self.lead_face(slot)
        nh = np.array([n[0], n[1], 0.0])
        nh /= np.linalg.norm(nh)
        alpha = math.atan2(-nh[0], nh[1])            # palm yaw (tool z = -n)
        ct, u, nf, up, H = self.top_corner(slot)
        psi, tilt, z_s = self.catch_geom(H, low_stop)
        # the parcel's extent across the belt once square: the horizontal
        # body axis other than ia
        k_up = int(np.argmax(np.abs(R[2, :])))
        jb = [j for j in range(3) if j not in (ia, k_up)][0]
        W2 = float(size[jb])
        L2 = 2 * h_n
        y_face = float(np.clip(rp.catch_y_face, -0.15 + L2, 0.155))
        com_y_final = y_face - L2 / 2
        # contact: the pads' midpoint on the leading face at the projection
        # of the centre, at z_s; reached when the centre is at com_y_final
        Rl = palm_R("Left", psi, alpha, tilt)
        pad_off = -PAD_TIP * Rl[:, 2]
        c_rel = nh * h_n                                # centre -> face centre (horizontal)
        stop_c = np.array([pos[0], com_y_final, 0.0]) + c_rel
        stop_c[2] = self.belt + z_s
        stop = stop_c + rp.catch_gap * nh - pad_off
        T_acc, T_d = rp.catch_t_acc, rp.catch_t_dec
        wait = stop.copy()
        wait[1] -= v * (T_acc + T_d) / 2
        qw = self.ik_near("Left", wait, Rl, self.ex.q[SL["Left"]])
        if qw is None and tilt < math.radians(25):
            tilt = math.radians(20.0)
            Rl = palm_R("Left", psi, alpha, tilt)
            pad_off = -PAD_TIP * Rl[:, 2]
            stop = stop_c + rp.catch_gap * nh - pad_off
            wait = stop.copy()
            wait[1] -= v * (T_acc + T_d) / 2
            qw = self.ik_near("Left", wait, Rl, self.ex.q[SL["Left"]])
        if qw is None:
            raise PlanFail("flush catch: wait pose unreachable")
        q_goal = self.ex.q.copy()
        q_goal[SL["Left"]] = qw
        Qm, _, _ = joint_move(self.ex.q, q_goal, vmax=rp.catch_vmax, tmin=0.4)
        # the face centre's y moves with the parcel at belt speed
        def face_y():
            pos_, R_, _, _ = self.box(slot)
            n_, ia_, h_ = self.lead_face(slot)
            return float(pos_[1] + n_[1] * h_)
        y_c_stop = stop_c[1]
        y_meet = wait[1] + pad_off[1] + v * T_acc / 2 - rp.catch_gap * nh[1]
        y_meet_face = y_c_stop - (stop[1] - wait[1]) + v * T_acc / 2
        t_start = (y_meet_face - face_y()) / v - T_acc
        if t_start < len(Qm) * DT + 0.05:
            raise PlanFail(f"flush catch: too late ({t_start:.2f} s)")
        eo.phase = "flush_approach"
        for row in Qm:
            eo.tick(row)
        t_left = (y_meet_face - face_y()) / v - T_acc
        if t_left > DT:
            eo.hold(t_left)
        eo.phase = "flush_stop"
        qq = self.ex.q[SL["Left"]].copy()
        n1, n2 = int(round(T_acc / DT)), int(round(T_d / DT))
        yy = wait[1]
        F_in = -nh * 0.8 * mg
        for i in range(n1 + n2):
            t = (i + 1) * DT
            vp = v * t / T_acc if i < n1 else v * max(0.0, 1 - (t - T_acc) / T_d)
            yy += vp * DT
            tgt = stop.copy()
            tgt[1] = min(yy, stop[1])
            qn, ok, ep = eo.ik_R("Left", tgt, Rl, qq, iters=60)
            qq = qn
            row = self.ex.q.copy()
            row[SL["Left"]] = qn
            lvl = min(1.0, max(0.0, (i - n1 + 2) / 10.0))
            eo.tick(row, {"Left": F_in * lvl})
        hist = []

        def stopped():
            if not self.cell.slots[slot].active:
                return True
            hist.append(float(self.box(slot)[0][1]))
            return len(hist) >= 10 and (max(hist[-10:]) - min(hist[-10:])) < 0.003
        eo.hold(timeout, forces={"Left": F_in}, check=stopped)
        if not self.cell.slots[slot].active or len(hist) < 10 or \
                (max(hist[-10:]) - min(hist[-10:])) > 0.004:
            raise PlanFail("flush catch: not stopped")
        # ---- square and justify: the parcel's pose is driven from its
        #      present pose to the target (square, centre x on the near-edge
        #      line + half its width, centre y unchanged), the palm rigidly
        #      with it
        eo.phase = "square"
        pos0, R0, _, _ = self.box(slot)
        yaw_box0 = self.yaw_of(R0)
        n0, _, _ = self.lead_face(slot)
        err0 = math.atan2(-n0[0], n0[1])               # rotation still to do = -err0? (CCW positive)
        # target yaw: rotate by -err0 so that the leading face's normal is +y
        d_yaw = -err0
        x_t = 0.16 + p.justify_gap + W2 / 2
        x_t = float(np.clip(x_t, 0.0, 0.33))
        com0 = pos0.copy()
        tcp0, Rt0 = self.cell.tcp_pose("Left")
        # palm pose in the parcel's (yaw-only) frame
        def rz(a):
            c_, s_ = math.cos(a), math.sin(a)
            return np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
        rel_p = rz(-yaw_box0) @ (self.kin["Left"].fk(self.ex.q[SL["Left"]])[0] - com0)
        rel_R = rz(-yaw_box0) @ self.kin["Left"].fk(self.ex.q[SL["Left"]])[1]
        n_sq = max(10, int(math.ceil(max(abs(d_yaw) / math.radians(p.square_rate_deg_s),
                                         abs(x_t - com0[0]) / p.justify_speed, 0.4) / DT)))
        qq = self.ex.q[SL["Left"]].copy()
        for i in range(1, n_sq + 1):
            s_ = i / n_sq
            s = s_ * s_ * (3 - 2 * s_)
            yaw_d = yaw_box0 + d_yaw * s
            com_d = com0 + np.array([(x_t - com0[0]) * s, 0.0, 0.0])
            tgt = com_d + rz(yaw_d) @ rel_p
            Rd = rz(yaw_d) @ rel_R
            qn, ok, ep = eo.ik_R("Left", tgt, Rd, qq, iters=60)
            stp = np.abs(qn - qq).max()
            if stp > rp.step_max:
                qn = qq + (qn - qq) * (rp.step_max / stp)
            qq = qn
            row = self.ex.q.copy()
            row[SL["Left"]] = qn
            n_now = rz(yaw_d) @ rz(-yaw_box0) @ n0
            lat = np.array([np.sign(x_t - com0[0]) * p.justify_f_mg * mg, 0.0, 0.0]) if i < n_sq else np.zeros(3)
            eo.tick(row, {"Left": -np.array([n_now[0], n_now[1], 0.0]) * 0.8 * mg + lat})
            if not self.cell.slots[slot].active:
                raise PlanFail("square: parcel gone")
        eo.hold(0.3, forces={"Left": np.array([0.0, -0.8 * mg, 0.0])})
        # the roll's catch pose (its own format)
        pos1, R1, _, _ = self.box(slot)
        n1_, _, _ = self.lead_face(slot)
        err1 = math.degrees(math.atan2(-n1_[0], n1_[1]))
        tcp1, Rt1 = self.cell.tcp_pose("Left")
        Rl_sq = palm_R("Left", psi, 0.0, tilt)
        eo.catch_pose = dict(x=float(tcp1[0]), z=self.belt + z_s, R=Rl_sq, tilt=tilt, psi=psi,
                             pad_off=-PAD_TIP * Rl_sq[:, 2], q=self.ex.q[SL["Left"]].copy())
        if low_stop:
            eo._low_catch = True
        else:
            eo._low_catch = False
        return dict(err_deg=round(err1, 1), x_com=round(float(pos1[0]), 3),
                    near_face=round(float(pos1[0] - W2 / 2), 3), y_face=round(float(pos1[1] + L2 / 2), 3),
                    alpha_deg=round(math.degrees(alpha), 1), t=round(self.cell.time, 2))

    # ==================================================================
    # square catch with drift compensation + justify (the roll's caught
    # state for a turned parcel)
    # ==================================================================
    def predict_square(self, slot, px, y_line, pad_half):
        """2-D prediction of a square catch: the parcel rides into a palm
        line at y_line spanning px +- pad_half and turns flat against it
        about its first contact point. Returns (centre x after squaring,
        rotation, contact x)."""
        pos, R, size, _ = self.box(slot)
        k_up = int(np.argmax(np.abs(R[2, :])))
        hor = [j for j in range(3) if j != k_up]
        h = size / 2
        corners = []
        for sa in (1, -1):
            for sb in (1, -1):
                c = pos + sa * h[hor[0]] * R[:, hor[0]] + sb * h[hor[1]] * R[:, hor[1]]
                corners.append(np.array([c[0], c[1]]))
        corners = np.array(corners)
        com = np.array([pos[0], pos[1]])
        # the leading corner and the parcel's travel distance to the line
        lo, hi = px - pad_half, px + pad_half
        best = None
        # candidate contact points: corners inside the span, and the front
        # edges' crossings of x = lo / hi
        cand = [c for c in corners if lo <= c[0] <= hi]
        order = [0, 1, 3, 2]
        for a_, b_ in zip(order, order[1:] + order[:1]):
            A, B = corners[a_], corners[b_]
            for xe in (lo, hi):
                if (A[0] - xe) * (B[0] - xe) < 0:
                    t = (xe - A[0]) / (B[0] - A[0])
                    cand.append(A + t * (B - A))
        if not cand:
            return float(com[0]), 0.0, float(px)
        c0 = max(cand, key=lambda c: c[1])
        # rotation to square: the leading face's yaw error, in the sense the
        # belt drag (through the centre) drives about the contact
        n, ia, hn = self.lead_face(slot)
        err = math.atan2(-n[0], n[1])         # palm yaw that would be flush
        d = -err                               # rotation that squares it
        if (com[0] - c0[0]) * d < 0 and abs(com[0] - c0[0]) > 0.01:
            # the drag turns it the other way: it squares to the other face
            d = math.copysign(math.pi / 2 - abs(d), com[0] - c0[0])
        cr, sr = math.cos(d), math.sin(d)
        r = com - c0
        com_f = c0 + np.array([cr * r[0] - sr * r[1], sr * r[0] + cr * r[1]])
        return float(com_f[0]), float(d), float(c0[0])

    def leading_corner_x(self, slot):
        """(x of the leading face's front-most corner, the leading face's
        length across) of a riding parcel."""
        pos, R, size, _ = self.box(slot)
        n, ia, hn = self.lead_face(slot)
        k_up = int(np.argmax(np.abs(R[2, :])))
        jb = [j for j in range(3) if j not in (ia, k_up)][0]
        h = size / 2
        centre = pos + n * hn
        best = None
        for sg in (1.0, -1.0):
            c = centre + sg * h[jb] * R[:, jb]
            if best is None or c[1] > best[1]:
                best = c
        return float(best[0]), float(size[jb])

    def _front_max_y(self, slot):
        """Largest y of the parcel's footprint (its most downstream point)."""
        pos, R, size, _ = self.box(slot)
        h = np.asarray(size, float) / 2
        best = -1e9
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                for sz in (1.0, -1.0):
                    c = pos + R @ (h * np.array([sx, sy, sz]))
                    best = max(best, float(c[1]))
        return best

    def low_catch_wait_q(self, slot):
        """Joint solution (left arm) of the low catch's wait pose for the
        parcel's present state (catch_at's own formulas, catch_mode "low",
        the roll's catch line); None if unreachable."""
        m0 = self.p.catch_mode
        self.p.catch_mode = "low"
        try:
            rp = self.eo.p
            pos, R, size, mass = self.box(slot)
            v = max(self.rc.belt_speed(), 0.02)
            from .geom import aligned_dims
            q4 = self.cell.data.qpos[self.cell._slot_qadr[slot] + 3:self.cell._slot_qadr[slot] + 7]
            W, L, H = aligned_dims(size, q4)
            y_face = float(np.clip(rp.catch_y_face, -0.15 + L, 0.155))
            psi_c, tilt, z_rel = self.catch_geom(H, False)
            Rl = palm_R("Left", psi_c, 0.0, tilt)
            x_lo = max(0.22, 0.150 + PAD_X * math.cos(psi_c) + rp.lip_clear)
            x_hi = 0.33 if tilt < math.radians(10) else self.p.catch_x_hi_tilted
            xc_lead, l_face = self.leading_corner_x(slot)
            px = xc_lead + l_face / 2
            px = min(px, float(pos[0]) + PAD_X * abs(math.cos(psi_c)) - self.p.square_margin)
            if self.p.long_margin > 0 and L >= self.p.long_min:
                px = min(px, float(pos[0]) - self.p.long_margin)
            x_s = float(np.clip(px, x_lo, x_hi))
            pad_off = -PAD_TIP * Rl[:, 2]
            stop = np.array([x_s, y_face + rp.catch_gap, self.belt + z_rel]) - pad_off
            wait = stop.copy()
            wait[1] = stop[1] - v * (self.p.own_t_acc + self.p.own_t_dec) / 2
            qw = self.ik_near("Left", wait, Rl, self.ex.q[SL["Left"]])
            if qw is None:
                qw = self.eo.glob_R("Left", wait, Rl, self.ex.q[SL["Left"]])
            return qw
        finally:
            self.p.catch_mode = m0

    def catch_at(self, slot, x_s=None, y_face=None, low_stop=False, timeout=14.0, park_right=True):
        """The roll's speed-matched catch (left palm, square to the belt) with
        the palm placed at x_s (default: the centre the parcel will have
        once squared against the palm), at the roll's catch line. Sets
        eo.catch_pose (the roll's format). Raises PlanFail."""
        eo = self.eo
        rp = eo.p
        pos, R, size, mass = self.box(slot)
        mg = mass * G
        v = max(self.rc.belt_speed(), 0.02)
        from .geom import aligned_dims
        q4 = self.cell.data.qpos[self.cell._slot_qadr[slot] + 3:self.cell._slot_qadr[slot] + 7]
        W, L, H = aligned_dims(size, q4)
        if y_face is None:
            y_face = float(np.clip(rp.catch_y_face, -0.15 + L, 0.155))
        psi_c, tilt, z_rel = self.catch_geom(H, low_stop)
        Rl = palm_R("Left", psi_c, 0.0, tilt)
        x_lo = max(0.22, 0.150 + PAD_X * math.cos(psi_c) + rp.lip_clear)
        x_hi = 0.33 if tilt < math.radians(10) else self.p.catch_x_hi_tilted
        if x_s is None:
            if self.p.square_pred == "corner":
                # the parcel squares against the palm line about a point of
                # its leading face: once flush, that face runs from (about)
                # its leading corner's x over the face's length, so its centre
                # lies half a face length from the leading corner
                xc_lead, l_face = self.leading_corner_x(slot)
                px = xc_lead + l_face / 2
                # the first contact (the near pad) must lie on the near side
                # of the centre, or the belt's moment turns the parcel back
                # instead of squaring it
                px = min(px, float(pos[0]) + PAD_X * abs(math.cos(psi_c)) - self.p.square_margin)
                if self.p.long_margin > 0 and L >= self.p.long_min:
                    px = min(px, float(pos[0]) - self.p.long_margin)
                px = float(np.clip(px, x_lo, x_hi))
            else:
                pad_half = PAD_X * abs(math.cos(psi_c)) + 0.011
                px = float(np.clip(pos[0], x_lo, x_hi))
                for _ in range(4):
                    cx, d, c0x = self.predict_square(slot, px, y_face, pad_half)
                    px = float(np.clip(cx, x_lo, x_hi))
            x_s = px
        z_s = self.belt + z_rel
        pad_off = -PAD_TIP * Rl[:, 2]
        stop = np.array([x_s, y_face + rp.catch_gap, z_s]) - pad_off
        T_acc, T_d = self.p.own_t_acc, self.p.own_t_dec
        y_w = stop[1] - v * (T_acc + T_d) / 2
        wait = stop.copy()
        wait[1] = y_w
        qw = self.ik_near("Left", wait, Rl, self.ex.q[SL["Left"]])
        if qw is None:
            qw = eo.glob_R("Left", wait, Rl, self.ex.q[SL["Left"]])
        if qw is None:
            raise PlanFail("catch: wait pose unreachable")
        q_goal = self.ex.q.copy()
        q_goal[SL["Left"]] = qw
        Qm, _, _ = joint_move(self.ex.q, q_goal, vmax=self.p.own_vmax, tmin=0.25, amax=self.p.own_amax)

        def front_y():
            pos_, R_, s_, _ = self.box(slot)
            k_ = int(np.argmax(np.abs(R_[2, :])))
            ext = sum(abs(R_[1, j]) * s_[j] / 2 for j in range(3) if j != k_)
            return float(pos_[1] + ext)
        t_start = (y_w + v * T_acc / 2 - front_y()) / v - T_acc
        self.last_catch_timing = dict(t_start=round(t_start, 3), t_move=round(len(Qm) * DT, 3),
                                      y_front=round(front_y(), 3), y_w=round(float(y_w), 3),
                                      dq=round(float(np.abs(q_goal - self.ex.q).max()), 3))
        if t_start < len(Qm) * DT + 0.05:
            raise PlanFail(f"catch: too late ({t_start:.2f} s, move {len(Qm) * DT:.2f} s, front {front_y():.3f})")
        eo.phase = "catch"
        for row in Qm:
            eo.tick(row)
        t_left = (y_w + v * T_acc / 2 - front_y()) / v - T_acc
        if t_left > DT:
            eo.hold(t_left)
        qq = self.ex.q[SL["Left"]].copy()
        n1, n2 = int(round(T_acc / DT)), int(round(T_d / DT))
        yy = y_w
        xx = float(stop[0])
        follow = self.p.catch_follow_x and not low_stop
        dxmax = self.p.catch_follow_rate * DT

        def follow_x(xx_):
            # the palm stays centred on the parcel's centre (across the belt)
            # while it squares against the palm
            if not follow or not self.cell.slots[slot].active:
                return xx_
            cx = float(np.clip(self.box(slot)[0][0], x_lo, x_hi)) - pad_off[0]
            return xx_ + float(np.clip(cx - xx_, -dxmax, dxmax))
        for i in range(n1 + n2):
            t = (i + 1) * DT
            vp = v * t / T_acc if i < n1 else v * max(0.0, 1 - (t - T_acc) / T_d)
            yy += vp * DT
            if i >= n1:
                xx = follow_x(xx)
            tgt = stop.copy()
            tgt[0] = xx
            tgt[1] = min(yy, stop[1])
            qn, ok, ep = eo.ik_R("Left", tgt, Rl, qq, iters=60)
            qq = qn
            row = self.ex.q.copy()
            row[SL["Left"]] = qn
            lvl = min(1.0, max(0.0, (i - n1 + 2) / 10.0))
            eo.tick(row, {"Left": np.array([0.0, -0.8 * mg * lvl, 0.0])})
        hist = []

        def stopped():
            if not self.cell.slots[slot].active:
                return True
            hist.append(float(self.box(slot)[0][1]))
            return len(hist) >= 12 and (max(hist[-12:]) - min(hist[-12:])) < 0.003
        n_hold = int(round(timeout / DT))
        for i in range(n_hold):
            xx = follow_x(xx)
            tgt = stop.copy()
            tgt[0] = xx
            qn, ok, ep = eo.ik_R("Left", tgt, Rl, qq, iters=40)
            qq = qn
            row = self.ex.q.copy()
            row[SL["Left"]] = qn
            eo.tick(row, {"Left": np.array([0.0, -0.8 * mg, 0.0])})
            if self.cell.done or stopped():
                break
        if not self.cell.slots[slot].active or len(hist) < 12 or \
                (max(hist[-12:]) - min(hist[-12:])) > 0.004:
            raise PlanFail("catch failed")
        x_s = float(xx + pad_off[0])
        eo.catch_pose = dict(x=x_s, z=z_s, R=Rl, tilt=tilt, psi=psi_c, pad_off=pad_off,
                             q=self.ex.q[SL["Left"]].copy())
        eo._low_catch = bool(low_stop)
        out = dict(x_s=round(x_s, 3), y_face=round(y_face, 3))
        if park_right:
            # the parcel is held by the left palm: the right arm parks now
            # (the roll's stop then starts from the park posture, as usual)
            self._hold_left = {"Left": np.array([0.0, -0.8 * mg, 0.0])}
            self._park_right_holding(slot, mg)
        pos1, R1, _, _ = self.box(slot)
        n1_, _, _ = self.lead_face(slot)
        out["err_deg"] = round(math.degrees(math.atan2(-n1_[0], n1_[1])), 1)
        out["x_com"] = round(float(pos1[0]), 3)
        return out

    def handover(self, slot, low_stop=False, justify=2, glide=True):
        """After the turn: catch the turned parcel at the roll's catch line
        in the roll's own catch pose, square to the belt and justified to the
        near edge, held (the "caught" state; eo.catch_pose set).
        Preferred: the low catch (pads side by side, tilted: it reaches the
        drifted parcel and squares it stably) upstream of the line, the
        two-palm sideways slide to the near edge, and the glide re-pose into
        the roll's pose at the line. If the low catch is too late (fast belts),
        the roll's own catch pose at the line and the slide. Raises PlanFail."""
        out = {}
        v = max(self.rc.belt_speed(), 0.02)
        m0 = self.p.catch_mode
        try:
            if glide:
                try:
                    self.p.catch_mode = "low"
                    yfl = self.y_face_low(v) if self.p.repose_mode == "glide" else None
                    out["catch"] = self.catch_at(slot, y_face=yfl)
                    out["mode"] = "low+glide"
                except PlanFail as e:
                    if "too late" not in str(e):
                        raise
                    out["low_fail"] = str(e)
                    out["low_timing"] = dict(getattr(self, "last_catch_timing", {}) or {})
                    glide = False
            if not glide:
                self.p.catch_mode = "roll"
                out["catch"] = self.catch_at(slot, y_face=None, low_stop=low_stop)
                out["mode"] = "direct"
        finally:
            self.p.catch_mode = m0
        self._repose_in_place = None
        if justify == 2 and self.p.repose_mode == "squeezed" and out.get("mode") != "direct":
            self._repose_in_place = dict(low_stop=low_stop)
            glide = False
        if justify == 2:
            out["justify"] = self.justify2(slot)
        elif justify:
            out["justify"] = self.justify(slot)
        self._repose_in_place = None
        if glide:
            out["glide"] = self.repose_glide(slot, low_stop=low_stop)
        pos1, R1, size1, _ = self.box(slot)
        from .geom import aligned_dims
        q4 = self.cell.data.qpos[self.cell._slot_qadr[slot] + 3:self.cell._slot_qadr[slot] + 7]
        W1, L1, H1 = aligned_dims(size1, q4)
        n1_, _, _ = self.lead_face(slot)
        out["final"] = dict(near_face=round(float(pos1[0] - W1 / 2), 3),
                            err_deg=round(math.degrees(math.atan2(-n1_[0], n1_[1])), 1),
                            y_face=round(float(pos1[1] + L1 / 2), 3),
                            palm_x=round(float(self.cell.tcp_pose("Left")[0][0]), 3),
                            com_x=round(float(pos1[0]), 3))
        return out

    def justify2(self, slot):
        """Two-palm sideways slide of the caught parcel to the near edge: the
        right palm (tool pointing downstream, tilted down, pads low) presses
        the upstream face and squeezes the parcel against the left palm; both
        palms move across the belt together (the belt slips under the
        parcel); then the right palm backs off and parks. The left palm keeps
        the parcel caught. Updates eo.catch_pose."""
        p = self.p
        eo = self.eo
        rp = eo.p
        pos, R, size, mass = self.box(slot)
        mg = mass * G
        from .geom import aligned_dims
        q4 = self.cell.data.qpos[self.cell._slot_qadr[slot] + 3:self.cell._slot_qadr[slot] + 7]
        W, L, H = aligned_dims(size, q4)
        x_near = float(pos[0] - W / 2)
        target = 0.16 + p.justify_gap
        dx = target - x_near
        out = dict(x_near0=round(x_near, 3), dx=round(dx, 3))
        if abs(dx) < 0.006:
            out["moved"] = 0.0
            out["x_near1"] = round(x_near, 3)
            return out
        FL = np.array([0.0, -0.8 * mg, 0.0])
        ts = math.radians(p.j2_tilt_deg)
        Rr = palm_R("Right", 0.0, 0.0, ts)
        y_u = float(pos[1] - L / 2)
        # the right palm's place across the belt balances the yaw moments of
        # the two palm forces about the parcel's centre (the left palm may be
        # off the centre after the squaring): x_r - x_c = (x_l - x_c) (Fs + Fd) / Fs
        Fs_ = p.j2_f + p.j2_f_mg * mg
        Fd_ = 0.9 * mg
        x_l = float(self.cell.tcp_pose("Left")[0][0])
        x_r = float(pos[0]) + (x_l - float(pos[0])) * (Fs_ + Fd_) / Fs_ * p.j2_balance
        x_r = float(np.clip(x_r, 0.215, 0.38))
        contact = np.array([x_r, y_u + p.j2_over, self.belt + p.j2_z])
        tcp_r = contact + PAD_TIP * Rr[:, 2]
        pre = tcp_r + np.array([0.0, -0.05, 0.05])
        qpre = self.ik_near("Right", pre, Rr, self.ex.q[SL["Right"]])
        if qpre is None:
            qpre = eo.glob_R("Right", pre, Rr, None)
        if qpre is None:
            raise PlanFail("justify2: right pre-pose unreachable")
        qg = self.ex.q.copy()
        qg[SL["Right"]] = qpre
        Q, _, _ = joint_move(self.ex.q, qg, vmax=2.6, tmin=0.6)
        if not self.path_clear(Q, "Right", margin=0.006):
            raise PlanFail("justify2: right approach path blocked")
        eo.phase = "justify2"
        for row in Q:
            r_ = row.copy()
            r_[SL["Left"]] = self.ex.q[SL["Left"]]
            eo.tick(r_, {"Left": FL})
        self._move_right_cart(tcp_r, Rr, 0.4)
        Fs = p.j2_f + p.j2_f_mg * mg
        nr = 12
        for i in range(nr):
            a = (i + 1) / nr
            eo.tick(self.ex.q.copy(), {"Left": FL - np.array([0.0, Fs * a, 0.0]),
                                       "Right": np.array([0.0, Fs * a, 0.0])})
        # slide both palms together
        tl0 = self.kin["Left"].fk(self.ex.q[SL["Left"]])[0]
        tr0 = self.kin["Right"].fk(self.ex.q[SL["Right"]])[0]
        RL = self.kin["Left"].fk(self.ex.q[SL["Left"]])[1]
        RR = self.kin["Right"].fk(self.ex.q[SL["Right"]])[1]
        n = max(8, int(math.ceil(abs(dx) / p.justify_speed / DT)))
        qL = self.ex.q[SL["Left"]].copy()
        qR = self.ex.q[SL["Right"]].copy()
        lat = math.copysign(p.justify_f_mg * mg / 2, dx)
        # the commanded slide overshoots the wanted one by j2_gain (the
        # parcel lags the palms and creeps back a little afterwards); the
        # palms' near pads keep clear of the near lip
        dxc = dx * p.j2_gain
        x_r_min = 0.16 + PAD_X + 0.011 + 0.012
        dx_r = max(dxc, x_r_min - float(tr0[0])) if dxc < 0 else dxc
        dx_l = max(dxc, x_r_min - float(tl0[0])) if dxc < 0 else dxc
        slide_halt = None
        for i in range(1, n + 1):
            s_ = i / n
            s = s_ * s_ * (3 - 2 * s_)
            off = np.array([dx_l * s, 0.0, 0.0])
            qn, ok, ep = eo.ik_R("Left", tl0 + off, RL, qL, iters=50)
            if p.ik_guard > 0 and float(ep) > p.ik_guard:
                slide_halt = ("left", round(float(ep), 4), i)
                break
            stp = np.abs(qn - qL).max()
            if stp > rp.step_max:
                qn = qL + (qn - qL) * (rp.step_max / stp)
            qL = qn
            # squeeze governor: the summed force on the parcel stays moderate
            ftot = self.contact_side(slot, "Left") + self.contact_side(slot, "Right")
            if ftot > p.j2_f_cap:
                Fs = max(2.0, Fs * 0.85)
            qn, ok, ep = eo.ik_R("Right", tr0 + np.array([dx_r * s, 0.0, 0.0]), RR, qR, iters=50)
            if p.ik_guard > 0 and float(ep) > p.ik_guard:
                slide_halt = ("right", round(float(ep), 4), i)
                break
            stp = np.abs(qn - qR).max()
            if stp > rp.step_max:
                qn = qR + (qn - qR) * (rp.step_max / stp)
            qR = qn
            row = self.ex.q.copy()
            row[SL["Left"]] = qL
            row[SL["Right"]] = qR
            eo.tick(row, {"Left": FL - np.array([-lat, Fs, 0.0]), "Right": np.array([lat, Fs, 0.0])})
            if not self.cell.slots[slot].active:
                raise PlanFail("justify2: parcel gone")
        out["slide_halt"] = slide_halt
        if getattr(self, "_repose_in_place", None) is not None:
            # re-pose the left palm into the roll's catch pose while the right
            # palm keeps squeezing (it holds the parcel against the twist)
            pos_r, R_r, size_r, _ = self.box(slot)
            q4_r = self.cell.data.qpos[self.cell._slot_qadr[slot] + 3:self.cell._slot_qadr[slot] + 7]
            W_r, L_r, H_r = aligned_dims(size_r, q4_r)
            if float(pos_r[1] + L_r / 2) > p.front_max:
                out["repose_skipped"] = round(float(pos_r[1] + L_r / 2), 3)
                self._repose_in_place = None
            else:
                self._repose_squeezed(slot, FL, Fs)
        # release the right palm: ramp down, back off, up, park
        eo.phase = "j2_release"
        for i in range(10):
            a = 1 - (i + 1) / 10
            eo.tick(self.ex.q.copy(), {"Left": FL - np.array([0.0, Fs * a, 0.0]),
                                       "Right": np.array([0.0, Fs * a, 0.0])})
        # back off the face (checked line; else the escape alone)
        fast_stage = self.p.stage_mode == "pre" and self.p.stage_right and hasattr(eo.p, "stage_clear") \
            and hasattr(eo.p, "stop_tilt_deg") and H >= self.p.stage_min_h
        pr_, Rr_ = self.kin["Right"].fk(self.ex.q[SL["Right"]])
        rows = None if fast_stage else self._cart_rows("Right", pr_ + np.array([0.0, -0.04, 0.03]), Rr_, 0.3)
        if rows is not None:
            Qb = np.repeat(self.ex.q.copy()[None], len(rows), axis=0)
            Qb[:, SL["Right"]] = np.array(rows)
            if self.path_clear(Qb, "Right", margin=0.008, every=1):
                self._play_right(Qb, {"Left": FL})
        staged = False
        if self.p.stage_right and hasattr(eo.p, "stage_clear") and hasattr(eo.p, "stop_tilt_deg") \
                and H >= self.p.stage_min_h:
            # the right palm waits where the roll will set its stop (the
            # roll's own staging pose above the upstream face): a parcel held
            # by one palm only creeps, so the stop follows at once
            staged = self._stage_right(slot, FL)
        if not staged:
            self._park_right_holding(slot, mg)
        out["staged"] = bool(staged)
        eo.hold(0.2 if not staged else 0.04, forces={"Left": FL})
        pos1, R1, _, _ = self.box(slot)
        cp = dict(eo.catch_pose)
        tl1 = self.cell.tcp_pose("Left")[0]
        cp["x"] = float(tl1[0])
        cp["q"] = self.ex.q[SL["Left"]].copy()
        eo.catch_pose = cp
        out["x_near1"] = round(float(pos1[0] - W / 2), 3)
        out["palm_off"] = round(float(pos1[0] - tl1[0]), 3)
        out["escape"] = list(getattr(self, "escape_log", []) or [])
        eo.phase = "caught"
        return out

    def _repose_squeezed(self, slot, FL, Fs):
        """Left palm from its present pose to the roll's catch pose (at the
        parcel's centre across the belt, the catch height), turning about its
        TCP, while the right palm squeezes the parcel."""
        eo = self.eo
        rp = eo.p
        low_stop = self._repose_in_place.get("low_stop", False)
        pos, R, size, mass = self.box(slot)
        from .geom import aligned_dims
        q4 = self.cell.data.qpos[self.cell._slot_qadr[slot] + 3:self.cell._slot_qadr[slot] + 7]
        W, L, H = aligned_dims(size, q4)
        if low_stop:
            psi_c, tilt_c, z_rel = self.catch_geom(H, True)
        else:
            psi_c, tilt_c, z_rel = self.catch_geom_std(H)
            if self.p.repose_psi_deg >= 0:
                rp_ = self.eo.p
                psi_c = math.radians(self.p.repose_psi_deg)
                dz_ = PAD_X * abs(math.sin(psi_c))
                z_lo_ = max(rp_.catch_zmin, rp_.catch_lo + dz_)
                z_rel = float(np.clip(rp_.catch_mid_frac * H, z_lo_, max(z_lo_, H - dz_ - 0.012)))
        R1 = palm_R("Left", psi_c, 0.0, tilt_c)
        pad_off1 = -PAD_TIP * R1[:, 2]
        tcp0, R0 = self.kin["Left"].fk(self.ex.q[SL["Left"]])
        x_lo = max(0.22, 0.150 + PAD_X * math.cos(psi_c) + rp.lip_clear)
        x_s = float(np.clip(pos[0] + self.p.repose_bias, x_lo, self.p.repose_x_hi))
        y_face = float(pos[1] + L / 2)
        tgt1 = np.array([x_s, y_face + rp.catch_gap, self.belt + z_rel]) - pad_off1
        from .kin import rot_err
        rv = rot_err(R0, R1)
        ang = float(np.linalg.norm(rv))
        n = max(10, int(math.ceil(self.p.repose_s / DT)))
        qq = self.ex.q[SL["Left"]].copy()
        eo.phase = "repose"
        for i in range(1, n + 1):
            s_ = i / n
            s = s_ * s_ * (3 - 2 * s_)
            Rt = R0 if ang < 1e-9 else rot_axis_angle(rv / ang, ang * s) @ R0
            tgt = tcp0 + (tgt1 - tcp0) * s
            qn, ok, ep = eo.ik_R("Left", tgt, Rt, qq, iters=60)
            if self.p.ik_guard > 0 and float(ep) > self.p.ik_guard:
                self._repose_halt = round(float(ep), 4)
                break
            stp = np.abs(qn - qq).max()
            if stp > rp.step_max:
                qn = qq + (qn - qq) * (rp.step_max / stp)
            qq = qn
            row = self.ex.q.copy()
            row[SL["Left"]] = qn
            ftot = self.contact_side(slot, "Left") + self.contact_side(slot, "Right")
            if ftot > self.p.j2_f_cap:
                Fs = max(2.0, Fs * 0.85)
            eo.tick(row, {"Left": FL - np.array([0.0, Fs, 0.0]), "Right": np.array([0.0, Fs, 0.0])})
            if not self.cell.slots[slot].active:
                raise PlanFail("repose: parcel gone")
        eo.catch_pose = dict(x=x_s, z=self.belt + z_rel, R=R1, tilt=tilt_c, psi=psi_c,
                             pad_off=pad_off1, q=self.ex.q[SL["Left"]].copy())
        eo._low_catch = bool(low_stop)

    def _stage_right(self, slot, FL):
        """Right palm to the roll's staging pose (the stop orientation, above
        and upstream of the upstream face; the roll then drops it straight to
        its stop pre-pose): a checked straight line up, then across. Sets
        eo.staged. False if no clear path."""
        eo = self.eo
        rp = eo.p
        from .geom import aligned_dims
        pos, R, size, mass = self.box(slot)
        q4 = self.cell.data.qpos[self.cell._slot_qadr[slot] + 3:self.cell._slot_qadr[slot] + 7]
        W, L, H = aligned_dims(size, q4)
        ts = math.radians(float(rp.stop_tilt_deg))
        Rr = palm_R("Right", 0.0, 0.0, ts)
        x_st = float(np.clip(pos[0], 0.215, 0.34))
        y_u = float(pos[1] - L / 2)
        stgs = [np.array([x_st, y_u - 0.05, self.belt + H + float(rp.stage_clear)]),
                # else the roll's own stop pre-pose (5 cm upstream of the
                # face, 7 cm up): the roll's drop from there is a null move
                np.array([x_st, y_u - 0.05, self.belt + 0.07])]
        if self.p.stage_mode == "pre":
            stgs = stgs[::-1]
        for stg in stgs:
            if self._stage_to(slot, FL, stg, Rr, y_u, direct=(self.p.stage_mode == "pre")):
                return True
        return False

    def _stage_to(self, slot, FL, stg, Rr, y_u, direct=False):
        eo = self.eo
        pr_, Rr0 = self.kin["Right"].fk(self.ex.q[SL["Right"]])
        mid = np.array([pr_[0], min(pr_[1], y_u - 0.03), max(stg[2], pr_[2])])
        q14 = self.ex.q.copy()
        Q = None
        if direct:
            # one straight line from the face to the stop pre-pose (checked)
            r_d = self._cart_rows("Right", stg, Rr, 0.35)
            if r_d is not None:
                Q = np.repeat(q14[None], len(r_d), axis=0)
                Q[:, SL["Right"]] = np.array(r_d)
                if self.path_clear(Q, "Right", margin=0.008, every=1):
                    eo.phase = "stage"
                    self._play_right(Q, {"Left": FL})
                    eo.staged = dict(pos=stg, R=Rr)
                    return True
                Q = None
        r1 = self._cart_rows("Right", mid, Rr0, max(0.3, (mid[2] - pr_[2]) / 0.3))
        r2 = self._cart_rows("Right", stg, Rr, 0.4, q0=np.array(r1[-1])) if r1 is not None else None
        if r1 is not None and r2 is not None:
            Q = np.repeat(q14[None], len(r1) + len(r2), axis=0)
            Q[:, SL["Right"]] = np.array(list(r1) + list(r2))
            if not self.path_clear(Q, "Right", margin=0.008, every=1):
                Q = None
        if Q is None:
            # up the straight line if it is clear, then a joint move across
            q_up = None
            if r1 is not None:
                Q1 = np.repeat(q14[None], len(r1), axis=0)
                Q1[:, SL["Right"]] = np.array(r1)
                if self.path_clear(Q1, "Right", margin=0.008, every=1):
                    q_up = Q1
            q_from = q_up[-1] if q_up is not None else q14
            for qs in (self.ik_near("Right", stg, Rr, q_from[SL["Right"]]),
                       eo.glob_R("Right", stg, Rr, q_from[SL["Right"]])):
                if qs is None:
                    continue
                g = q_from.copy()
                g[SL["Right"]] = qs
                Qj, _, _ = joint_move(q_from, g, vmax=2.4, tmin=0.5)
                Qc = Qj if q_up is None else np.vstack([q_up, Qj])
                if self.path_clear(Qc, "Right", margin=0.008, every=1):
                    Q = Qc
                    break
        if Q is None:
            return False
        eo.phase = "stage"
        self._play_right(Q, {"Left": FL})
        eo.staged = dict(pos=stg, R=Rr)
        return True

    def _park_right_holding(self, slot, mg):
        """Right arm to its park joints while the left palm holds the caught
        parcel (left feedforward kept)."""
        F = {"Left": np.array([0.0, -0.8 * mg, 0.0])}
        self._escape_right(F)

    def _cart_rows(self, side, tgt, R, T, q0=None, jump=0.10, tol=0.004):
        """Joint rows of a straight TCP line from the commanded pose to tgt
        (orientation R, smooth profile); None if the IK loses the line
        (position error above tol, or a joint step above jump rad)."""
        eo = self.eo
        kin = self.kin[side]
        q = (self.ex.q[SL[side]] if q0 is None else q0).copy()
        p0, _ = kin.fk(q)
        n = max(2, int(math.ceil(T / DT)))
        rows = []
        for i in range(1, n + 1):
            s_ = i / n
            s = s_ * s_ * (3 - 2 * s_)
            qn, ok, ep = eo.ik_R(side, p0 + (np.asarray(tgt, float) - p0) * s, R, q, iters=80)
            if (not ok) or ep > tol or np.abs(qn - q).max() > jump:
                return None
            q = qn
            rows.append(q.copy())
        return rows

    def _play_right(self, Q, F=None):
        for row in Q:
            r_ = np.array(row, float).copy()
            r_[SL["Left"]] = self.ex.q[SL["Left"]]
            self.eo.tick(r_, F)

    def _escape_right(self, F=None, dz_min=0.22, back_y=-0.02):
        """The right palm away from the parcel: up to belt + dz_min (and a
        little upstream), then its park joints. Every leg is checked against
        the station (true distances) and the robot before it is played:
        a straight TCP line, else a joint move to the pose above; to park a
        joint move, else through a high waypoint on the robot side."""
        rc = self.rc
        kin = self.kin["Right"]
        self.escape_log = log = []
        self.eo.phase = "escape"
        q = self.ex.q[SL["Right"]].copy()
        p0, R0 = kin.fk(q)
        q14 = self.ex.q.copy()
        leg = None
        Rd = frame_zx((0.0, 0.0, -1.0), R0[:, 0])
        alts = ((dz_min, back_y), (0.15, back_y), (0.15, -0.06), (0.10, -0.04)) if self.p.escape_checked else ()
        up = np.array([p0[0], p0[1], max(p0[2], self.belt + dz_min)])
        for dz_, by_ in alts:
            up = np.array([p0[0], p0[1] + by_, max(p0[2] + 0.04, self.belt + dz_)])
            rows = self._cart_rows("Right", up, R0, max(0.3, (up[2] - p0[2]) / 0.3))
            if rows is not None:
                Q = np.repeat(q14[None], len(rows), axis=0)
                Q[:, SL["Right"]] = np.array(rows)
                wl = []
                if self.path_clear(Q, "Right", margin=0.008, every=1, why=wl):
                    leg = Q
                    log.append(("line", dz_))
                    break
                log.append(("line-blocked", wl[:1]))
            for Rt, tag in ((R0, "joint"), (Rd, "joint-down")):
                for qa in (self.ik_near("Right", up, Rt, q), self.eo.glob_R("Right", up, Rt, q)):
                    if qa is None:
                        continue
                    qg = q14.copy()
                    qg[SL["Right"]] = qa
                    Q, _, _ = joint_move(q14, qg, vmax=2.4, tmin=0.5)
                    wl = []
                    if self.path_clear(Q, "Right", margin=0.008, every=1, why=wl):
                        leg = Q
                        log.append((tag, dz_))
                        break
                    log.append((tag + "-blocked", wl[:1]))
                if leg is not None:
                    break
            if leg is not None:
                break
        if leg is None:
            log.append("unchecked")
            rows = []
            qq = q.copy()
            n = max(2, int(math.ceil(max(0.3, (up[2] - p0[2]) / 0.3) / DT)))
            for i in range(1, n + 1):
                s_ = i / n
                s = s_ * s_ * (3 - 2 * s_)
                qn, ok, ep = self.eo.ik_R("Right", p0 + (up - p0) * s, R0, qq)
                stp = np.abs(qn - qq).max()
                if stp > self.eo.p.step_max:
                    qn = qq + (qn - qq) * (self.eo.p.step_max / stp)
                qq = qn
                rows.append(qq.copy())
            leg = np.repeat(q14[None], len(rows), axis=0)
            leg[:, SL["Right"]] = np.array(rows)
        self._play_right(leg, F)
        q14 = self.ex.q.copy()
        qg = q14.copy()
        qg[SL["Right"]] = rc.q_ready[SL["Right"]]
        Q, _, _ = joint_move(q14, qg, vmax=self.p.v_park, tmin=0.6)
        if self.p.escape_checked and not self.path_clear(Q, "Right", margin=0.01, every=1):
            pk, Rk = kin.fk(rc.q_ready[SL["Right"]])
            pc, Rc = kin.fk(self.ex.q[SL["Right"]])
            done = False
            for mid in (np.array([0.12, pc[1], max(pc[2], pk[2])]),
                        np.array([0.10, -0.25, pk[2] + 0.05])):
                qm = self.ik_near("Right", mid, Rk, self.ex.q[SL["Right"]])
                if qm is None:
                    continue
                g1 = q14.copy()
                g1[SL["Right"]] = qm
                Q1, _, _ = joint_move(q14, g1, vmax=self.p.v_park, tmin=0.5)
                Q2, _, _ = joint_move(Q1[-1], qg, vmax=self.p.v_park, tmin=0.5)
                Qc = np.vstack([Q1, Q2])
                if self.path_clear(Qc, "Right", margin=0.01, every=1):
                    Q = Qc
                    done = True
                    log.append("park-via")
                    break
            if not done:
                log.append("park-unchecked")
        else:
            log.append("park")
        self._play_right(Q, F)

    def justify(self, slot):
        out = None
        for k in range(max(1, self.p.justify_passes)):
            r = self._justify_pass(slot)
            out = r if out is None else dict(out, **{f"pass{k}": r})
            out["x_near1"] = r.get("x_near1", r.get("x_near0"))
            out["palm_off"] = r.get("palm_off", out.get("palm_off"))
            if abs(r.get("x_near1", r.get("x_near0", 0.0)) - (0.16 + self.p.justify_gap)) < self.p.justify_tol \
                    or r.get("moved") == 0.0:
                break
        return out

    def _justify_pass(self, slot):
        """Slide the caught parcel along the belt's width to the near edge
        (its near face justify_gap inside it): the left palm, pressed on the
        downstream face by the belt, moves across the belt and drags the
        parcel by pad friction while the belt slips under it. Updates
        eo.catch_pose. Returns the record."""
        p = self.p
        eo = self.eo
        rp = eo.p
        pos, R, size, mass = self.box(slot)
        mg = mass * G
        from .geom import aligned_dims
        q4 = self.cell.data.qpos[self.cell._slot_qadr[slot] + 3:self.cell._slot_qadr[slot] + 7]
        W, L, H = aligned_dims(size, q4)
        x_near = float(pos[0] - W / 2)
        target = 0.16 + p.justify_gap
        dx = (target - x_near) * p.justify_gain
        out = dict(x_near0=round(x_near, 3), dx=round(dx, 3))
        if abs(dx) < 0.004:
            out["moved"] = 0.0
            return out
        cp = eo.catch_pose
        Rl = cp["R"]
        tcp0 = self.kin["Left"].fk(self.ex.q[SL["Left"]])[0]
        n = max(8, int(math.ceil(abs(dx) / p.justify_speed / DT)))
        qq = self.ex.q[SL["Left"]].copy()
        eo.phase = "justify"
        for i in range(1, n + 1):
            s_ = i / n
            s = s_ * s_ * (3 - 2 * s_)
            tgt = tcp0 + np.array([dx * s, 0.0, 0.0])
            qn, ok, ep = eo.ik_R("Left", tgt, Rl, qq, iters=60)
            stp = np.abs(qn - qq).max()
            if stp > rp.step_max:
                qn = qq + (qn - qq) * (rp.step_max / stp)
            qq = qn
            row = self.ex.q.copy()
            row[SL["Left"]] = qn
            eo.tick(row, {"Left": np.array([math.copysign(p.justify_f_mg * mg, dx), -p.justify_press_mg * mg, 0.0])})
            if not self.cell.slots[slot].active:
                raise PlanFail("justify: parcel gone")
        eo.hold(0.3, forces={"Left": np.array([0.0, -0.8 * mg, 0.0])})
        pos1, R1, _, _ = self.box(slot)
        cp = dict(cp)
        cp["x"] = float(cp["x"] + dx)
        cp["q"] = self.ex.q[SL["Left"]].copy()
        eo.catch_pose = cp
        out["x_near1"] = round(float(pos1[0] - W / 2), 3)
        out["palm_off"] = round(float(pos1[0] - self.cell.tcp_pose("Left")[0][0]), 3)
        return out

    def repose_glide(self, slot, low_stop=False):
        """From a caught state in another palm pose (the low catch after
        squaring and justifying) to the roll's standard catch pose at the
        roll's catch line, without twisting the parcel: the palm accelerates
        with the parcel to belt speed (the belt no longer presses it), turns
        and moves to the standard pose while gliding with it, then brakes to
        rest at the catch line (a speed-matched catch). Sets eo.catch_pose."""
        eo = self.eo
        rp = eo.p
        p = self.p
        pos, R, size, mass = self.box(slot)
        mg = mass * G
        v = max(self.rc.belt_speed(), 0.02)
        from .geom import aligned_dims
        q4 = self.cell.data.qpos[self.cell._slot_qadr[slot] + 3:self.cell._slot_qadr[slot] + 7]
        W, L, H = aligned_dims(size, q4)
        if low_stop:
            psi_c, tilt_c, z_rel = self.catch_geom(H, True)
        else:
            psi_c, tilt_c, z_rel = self.catch_geom_std(H)
            if self.p.repose_psi_deg >= 0:
                rp_ = self.eo.p
                psi_c = math.radians(self.p.repose_psi_deg)
                dz_ = PAD_X * abs(math.sin(psi_c))
                z_lo_ = max(rp_.catch_zmin, rp_.catch_lo + dz_)
                z_rel = float(np.clip(rp_.catch_mid_frac * H, z_lo_, max(z_lo_, H - dz_ - 0.012)))
        R1 = palm_R("Left", psi_c, 0.0, tilt_c)
        pad_off1 = -PAD_TIP * R1[:, 2]
        tcp0, R0 = self.kin["Left"].fk(self.ex.q[SL["Left"]])
        y_face_now = float(pos[1] + L / 2)
        x_c = float(pos[0])
        x_lo = max(0.22, 0.150 + PAD_X * math.cos(psi_c) + rp.lip_clear)
        if low_stop:
            x_lo = max(0.22, 0.150 + PAD_X + rp.lip_clear)
        x_s = float(np.clip(x_c, x_lo, 0.33))
        y_face1 = float(np.clip(rp.catch_y_face, -0.15 + L, 0.155))
        stop1 = np.array([x_s, y_face1 + rp.catch_gap, self.belt + z_rel]) - pad_off1
        n_acc = int(round(p.glide_t_acc / DT))
        n_dec = int(round(p.glide_t_dec / DT))
        # the glide distance at belt speed fills the rest of the way
        d_total = stop1[1] - tcp0[1]
        d_acc = v * p.glide_t_acc / 2
        d_dec = v * p.glide_t_dec / 2
        d_glide = d_total - d_acc - d_dec
        if d_glide < v * p.glide_t_min:
            raise PlanFail(f"glide: too short ({d_total:.3f} m)")
        n_gl = max(1, int(round(d_glide / v / DT)))
        from .kin import rot_err
        rv = rot_err(R0, R1)
        ang = float(np.linalg.norm(rv))
        qq = self.ex.q[SL["Left"]].copy()
        eo.phase = "glide"
        y = float(tcp0[1])
        N = n_acc + n_gl + n_dec
        for i in range(N):
            if i < n_acc:
                vp = v * (i + 1) / n_acc
            elif i < n_acc + n_gl:
                vp = v
            else:
                vp = v * max(0.0, 1 - (i - n_acc - n_gl + 1) / n_dec)
            y += vp * DT
            # pose blend during the glide phase only
            s_ = min(1.0, max(0.0, (i - n_acc + 1) / max(1, n_gl)))
            s = s_ * s_ * (3 - 2 * s_)
            Rt = R0 if ang < 1e-9 else rot_axis_angle(rv / ang, ang * s) @ R0
            tgt = np.array([tcp0[0] + (stop1[0] - tcp0[0]) * s, min(y, stop1[1]),
                            tcp0[2] + (stop1[2] - tcp0[2]) * s])
            qn, ok, ep = eo.ik_R("Left", tgt, Rt, qq, iters=60)
            stp = np.abs(qn - qq).max()
            if stp > rp.step_max:
                qn = qq + (qn - qq) * (rp.step_max / stp)
            qq = qn
            row = self.ex.q.copy()
            row[SL["Left"]] = qn
            lvl = 1.0 if i >= n_acc + n_gl else (0.3 if i < n_acc else 0.0)
            eo.tick(row, {"Left": np.array([0.0, -0.8 * mg * lvl, 0.0])})
            if not self.cell.slots[slot].active:
                raise PlanFail("glide: parcel gone")
        hist = []

        def stopped():
            if not self.cell.slots[slot].active:
                return True
            hist.append(float(self.box(slot)[0][1]))
            return len(hist) >= 12 and (max(hist[-12:]) - min(hist[-12:])) < 0.003
        eo.hold(6.0, forces={"Left": np.array([0.0, -0.8 * mg, 0.0])}, check=stopped)
        if not self.cell.slots[slot].active or len(hist) < 12 or \
                (max(hist[-12:]) - min(hist[-12:])) > 0.004:
            raise PlanFail("glide: not stopped")
        eo.catch_pose = dict(x=x_s, z=self.belt + z_rel, R=R1, tilt=tilt_c, psi=psi_c,
                             pad_off=pad_off1, q=self.ex.q[SL["Left"]].copy())
        eo._low_catch = bool(low_stop)
        pos2, R2, _, _ = self.box(slot)
        n2, _, _ = self.lead_face(slot)
        return dict(x_com=round(float(pos2[0]), 3), near_face=round(float(pos2[0] - W / 2), 3),
                    err_deg=round(math.degrees(math.atan2(-n2[0], n2[1])), 1),
                    y_face=round(float(pos2[1] + L / 2), 3), glide_m=round(d_total, 3))

    def y_face_low(self, v):
        """Downstream-face position of the low catch: upstream of the roll's
        catch line by the glide's distance."""
        p = self.p
        return 0.15 - v * (p.glide_t_acc / 2 + p.glide_t_min + p.glide_t_dec / 2) - 0.01

    def catch_geom_std(self, H):
        """The roll's own catch geometry (whatever catch_mode is set)."""
        m0 = self.p.catch_mode
        self.p.catch_mode = "roll"
        try:
            return self.catch_geom(H, False)
        finally:
            self.p.catch_mode = m0

    CHECK_MARGIN = 0.03              # contact margin of the robot geoms in the check model (m)
    SELF_MARGIN = 0.008              # clearance of an arm from the column, the head and the other arm in plans (m)

    def _check_model(self):
        """A private copy of the cell's model whose robot geoms carry a
        contact margin, so that the collision pass reports every robot -
        station pair closer than CHECK_MARGIN with its true distance (the
        cell's own geoms have margin 0: its contacts show penetration only)."""
        import copy
        import mujoco
        c = self.cell
        key = id(c.model)
        cache = SideTurn._CHECK
        if cache.get("key") != key:
            m2 = copy.deepcopy(c.model)
            for g in c.robot_geoms:
                m2.geom_margin[int(g)] = self.CHECK_MARGIN
            cache.clear()
            cache.update(key=key, m=m2, d=mujoco.MjData(m2))
        return cache["m"], cache["d"]

    _CHECK = {}

    def path_clear(self, Q, side="Right", margin=0.008, every=2, why=None):
        """True if the joint path Q (rows of 14) keeps the given arm's
        collision geoms at least margin away from the station (structure and
        the moving belt surface: true distances, through the check model) and
        free of contact with the rest of the robot (the column and head, the
        other arm: no penetration). Parcels are not checked (planned
        contacts)."""
        import mujoco
        c = self.cell
        m, d = self._check_model()
        gs = self._side_geoms()
        robot = c._geom_is_robot
        d.qpos[:] = c.data.qpos
        d.qvel[:] = 0
        qadr = self.ex.dyn.qadr
        # robot - robot pairs (the column and head, the other arm) keep
        # SELF_MARGIN, except pairs already that close at the path's start
        # (links that always sit next to the column)
        persistent = set()
        d.qpos[qadr] = Q[0]
        mujoco.mj_fwdPosition(m, d)
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if robot[g1] and robot[g2] and con.dist < self.p.self_margin:
                persistent.add((min(g1, g2), max(g1, g2)))
        for k in list(range(0, len(Q), every)) + [len(Q) - 1]:
            d.qpos[qadr] = Q[k]
            mujoco.mj_fwdPosition(m, d)
            for i in range(d.ncon):
                con = d.contact[i]
                g1, g2 = int(con.geom1), int(con.geom2)
                s1, s2 = gs.get(g1), gs.get(g2)
                if s1 != side and s2 != side:
                    continue
                other = g2 if s1 == side else g1
                if gs.get(other) == side:
                    continue                      # within the arm
                if c.parcel_geom_slot[other] >= 0:
                    continue                      # parcels: planned contacts
                if robot[other]:
                    lim = 0.0 if (min(g1, g2), max(g1, g2)) in persistent else self.p.self_margin
                else:
                    lim = margin
                if con.dist < lim:
                    if why is not None:
                        why.append((k, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1),
                                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2),
                                    round(float(con.dist), 4), [round(float(v), 3) for v in con.pos],
                                    len(Q)))
                    return False
        return True

    def exit_traj(self, slot):
        """Joint rows (14) of the right arm's exit: straight up (and a little
        towards the robot) from the hinge, then a joint move to the park
        joints; the left arm's joints are those of the moment (the hook only
        drives the right arm)."""
        p = self.p
        kin = self.kin["Right"]
        q = self.ex.q[SL["Right"]].copy()
        p0, R0 = kin.fk(q)
        tgt = p0 + np.array([-0.02, -0.02, p.lift_dz])
        n = max(2, int(math.ceil(p.lift_s / DT)))
        rows = []
        for i in range(1, n + 1):
            s_ = i / n
            s = s_ * s_ * s_ * (10 - 15 * s_ + 6 * s_ * s_)
            qn, ok, ep = self.eo.ik_R("Right", p0 + (tgt - p0) * s, R0, q)
            stp = np.abs(qn - q).max()
            if stp > self.eo.p.step_max:
                qn = q + (qn - q) * (self.eo.p.step_max / stp)
            q = qn
            rows.append(q.copy())
        q14 = self.ex.q.copy()
        q14[SL["Right"]] = q
        qg = q14.copy()
        qg[SL["Right"]] = self.rc.q_ready[SL["Right"]]
        Q, _, _ = joint_move(q14, qg, vmax=p.v_park, tmin=0.6)
        if self.eo.path_hits_station(Q, margin=0.01, every=2):
            # through a high intermediate pose
            pk, Rk = kin.fk(self.rc.q_ready[SL["Right"]])
            mid = np.array([max(0.10, tgt[0] - 0.05), tgt[1], max(tgt[2], pk[2])])
            qm = self.eo.glob_R("Right", mid, R0, q)
            if qm is not None:
                qg2 = q14.copy()
                qg2[SL["Right"]] = qm
                Q1, _, _ = joint_move(q14, qg2, vmax=p.v_park, tmin=0.4)
                Q2, _, _ = joint_move(Q1[-1], qg, vmax=p.v_park, tmin=0.6)
                Q = np.vstack([Q1, Q2])
        rows += [r[SL["Right"]].copy() for r in Q]
        return rows

    def _move_right_cart(self, tgt, R, T, QL=None, k0=0):
        eo = self.eo
        kin = self.kin["Right"]
        q = self.ex.q[SL["Right"]].copy()
        p0, _ = kin.fk(q)
        n = max(2, int(math.ceil(T / DT)))
        for i in range(1, n + 1):
            s_ = i / n
            s = s_ * s_ * s_ * (10 - 15 * s_ + 6 * s_ * s_)
            pt = p0 + (np.asarray(tgt) - p0) * s
            qn, ok, ep = eo.ik_R("Right", pt, R, q)
            stp = np.abs(qn - q).max()
            if stp > eo.p.step_max:
                qn = q + (qn - q) * (eo.p.step_max / stp)
            q = qn
            row = self.ex.q.copy()
            row[SL["Right"]] = q
            if QL is not None and k0 + i < len(QL):
                row[SL["Left"]] = QL[k0 + i][SL["Left"]]
            self._tick(row)

    def _park_right(self, slot, QL=None, k0=0):
        """Joint move of the right arm to its park joints (the left arm keeps
        its own motion), delayed while it would sweep the parcel."""
        eo = self.eo
        rc = self.rc
        qg = self.ex.q.copy()
        qg[SL["Right"]] = rc.q_ready[SL["Right"]]
        Q, _, _ = joint_move(self.ex.q, qg, vmax=self.p.v_park, tmin=0.6)
        if eo.path_hits_station(Q):
            pk, Rk = self.kin["Right"].fk(rc.q_ready[SL["Right"]])
            p_, R_ = self.cell.tcp_pose("Right")
            up = np.array([p_[0], p_[1], max(p_[2], pk[2])])
            if up[2] > p_[2] + 1e-3:
                self._move_right_cart(up, R_, max(0.3, (up[2] - p_[2]) / 0.3), QL, k0)
            self._move_right_cart(pk, Rk, 0.8, QL, k0)
            qg = self.ex.q.copy()
            qg[SL["Right"]] = rc.q_ready[SL["Right"]]
            Q, _, _ = joint_move(self.ex.q, qg, vmax=self.p.v_park, tmin=0.4)
        for i, row in enumerate(Q):
            r_ = row.copy()
            if QL is not None and k0 + i < len(QL):
                r_[SL["Left"]] = QL[k0 + i][SL["Left"]]
            else:
                r_[SL["Left"]] = self.ex.q[SL["Left"]]
            eo.tick(r_)

    def run_chain(self, slot):
        """Development chain for +x labels with the roll package:
        quarter turn, flush catch with square-and-justify
        (the roll's caught state), then the roll's own attempts from the
        caught state exactly as in EndOver.run (eo.roll_up with its retry
        rule), release, park."""
        eo = self.eo
        rp = eo.p
        out = dict(turn=None, catch=None, roll=None, ok=False, reason="")
        t = self.turn_press(slot) if self.p.method == "press" else self.turn(slot)
        out["turn"] = t
        if not t.get("ok"):
            out["reason"] = "turn: " + str(t.get("reason"))
            eo.back_off(slot=slot if self.cell.slots[slot].active else None)
            eo.park(slot if self.cell.slots[slot].active else None)
            return out
        try:
            if self.p.handover == "flush":
                out["catch"] = self.catch_flush(slot)
            else:
                out["catch"] = self.catch_at(slot)
                out["justify"] = self.justify(slot)
        except PlanFail as e:
            out["reason"] = "catch: " + str(e)
            eo.back_off(slot=slot if self.cell.slots[slot].active else None)
            eo.park(slot if self.cell.slots[slot].active else None)
            return out
        lab0, _ = eo.label_now(slot)
        want = "up" if lab0 == "+y" else ("down" if lab0 == "-y" else "?")
        roll = dict(label0=lab0, want=want, attempts=[], rolls=0, t0=self.cell.time)
        base = dict(push_fn=rp.push_fn, lead_deg=rp.lead_deg)
        for k in range(rp.max_rolls):
            if not self.cell.slots[slot].active:
                break
            try:
                r = eo.roll_up(slot, follow=bool(rp.follow))
            except PlanFail as e:
                roll["reason"] = f"roll plan: {e}"
                break
            roll["rolls"] += 1
            lab, err = eo.label_now(slot)
            roll["attempts"].append(dict(committed=r.committed, tilt=round(r.tilt_max_deg, 1),
                                         label=lab, err=round(err, 1), held=r.held,
                                         t=round(self.cell.time, 2)))
            if lab in ("up", "down", "+x", "-x", "gone"):
                break
            if r.held:
                rp.push_fn = base["push_fn"] * (1.5 + 0.5 * k)
                rp.lead_deg = base["lead_deg"] + 4.0 * (k + 1)
                continue
            if not self.cell.slots[slot].active:
                break
            try:
                eo.recatch(slot)
            except PlanFail as e:
                roll["reason"] = f"recatch: {e}"
                break
        rp.push_fn, rp.lead_deg = base["push_fn"], base["lead_deg"]
        eo.phase = "exit"
        eo.back_off(slot=slot if self.cell.slots[slot].active else None)
        eo.park(slot if self.cell.slots[slot].active else None)
        roll["t_end"] = self.cell.time
        lab, err = eo.label_now(slot)
        roll["label_after"], roll["label_err"] = lab, round(err, 1)
        out["roll"] = roll
        out["ok"] = lab == want and err < 25.0
        if not out["ok"]:
            out["reason"] = roll.get("reason") or f"label {lab} ({err:.0f} deg)"
        return out

    # ------------------------------------------------------------------
    def turn_only(self, slot):
        """Development: the turn alone; the parcel then rides on (the arms
        park), its yaw recorded 1.5 s after the turn."""
        t = self.turn_press(slot) if self.p.method == "press" else self.turn(slot)
        self.eo.hold(1.5)
        if self.cell.slots[slot].active:
            pos, Rb, _, _ = self.box(slot)
            t["yaw_after"] = round(math.degrees(self.yaw_of(Rb)), 1)
            t["x_after"] = round(float(pos[0]), 3)
        self.eo.back_off(slot=slot)
        self.eo.park(slot)
        return t

    def run(self, slot):
        """The whole job for a side-face label: quarter turn, then the
        roll package's end-over-end chain (EndOver.run). Entry as turn();
        exit as EndOver.run (the parcel resting on the belt, released, riding
        downstream; both arms parked). +x labels end label up (one roll);
        -x labels end on the upstream face... and one roll turns it down."""
        out = dict(turn=None, roll=None)
        t = self.turn_press(slot) if self.p.method == "press" else self.turn(slot)
        out["turn"] = t
        if not t.get("ok"):
            self.eo.back_off(slot=slot)
            self.eo.park(slot)
            out["ok"] = False
            out["reason"] = "turn: " + str(t.get("reason"))
            return out
        r = self.eo.run(slot)
        out["roll"] = r
        out["ok"] = bool(r.get("ok"))
        out["reason"] = r.get("reason", "")
        return out


# ======================================================================
# entry point
# ======================================================================
def release_caught(st, slot):
    """Let a caught parcel go without a kick: the left palm runs with it at
    belt speed (the belt no longer presses it into the palm), lifts away while
    running, then both arms park (the park move waits while it would sweep
    the parcel)."""
    eo = st.eo
    rp = eo.p
    if not st.cell.slots[slot].active:
        eo.park(None)
        return
    v = max(st.rc.belt_speed(), 0.02)
    tcp0, R0 = st.kin["Left"].fk(st.ex.q[SL["Left"]])
    qq = st.ex.q[SL["Left"]].copy()
    n_acc, n_lift = 8, 14
    y = float(tcp0[1])
    z = float(tcp0[2])
    for i in range(n_acc + n_lift):
        vp = v * min(1.0, (i + 1) / n_acc) * 1.05
        y += vp * DT
        if i < n_acc:
            z += 0.012 / n_acc              # off the belt from the start (a yawed parcel presses the palm)
        else:
            z += 0.09 / n_lift
        tgt = np.array([tcp0[0] - (0.0 if i < n_acc else 0.02 * (i - n_acc + 1) / n_lift), y, z])
        qn, ok, ep = eo.ik_R("Left", tgt, R0, qq, iters=40)
        stp = np.abs(qn - qq).max()
        if stp > rp.step_max:
            qn = qq + (qn - qq) * (rp.step_max / stp)
        qq = qn
        row = st.ex.q.copy()
        row[SL["Left"]] = qn
        eo.tick(row)
    eo.back_off(slot=slot if st.cell.slots[slot].active else None)
    eo.park(slot if st.cell.slots[slot].active else None)


def side_job(rc, eo, slot, params: SideParams | None = None, release_minus_y=True, log=None):
    """The side-face job: a parcel whose label faces away from the robot
    (+x) or towards it (-x) at the job's start.

    Entry: the parcel on the belt upstream of y = -0.5 m (the earlier the
    better: the right arm needs about 1.5 s to reach its hinge pose), as the
    infeed delivers it (near face at the near belt edge within 1 cm, yaw a
    multiple of 90 deg within +-6 deg); both arms at the park posture (the
    cell's start posture); rc / eo: the controller's RefController and its
    end-over-end roll object (EndOver of the roll package; its executor,
    kinematics, truth helpers and catch / roll parameters are used).

    What it does: the quarter turn (counter-clockwise seen from above, the
    right palm as a hinge at the near end of the downstream face, the belt
    turning the parcel), then the handover: the left palm catches the turned
    parcel at the roll's catch line, squares it, slides it with the right
    palm's help to the near belt edge and takes the roll's own catch pose.

    Exit (returned "state"):
      "caught"  (label now on +y): the parcel at rest against the left palm
                in the roll's catch pose at the roll's catch line, square to
                the belt, justified, held by the belt's push; eo.catch_pose
                set; the right arm parked. The caller goes straight on with
                the roll from the caught state (the roll package's
                roll_up / the scheduler's roll_chain(caught=True)).
      "riding"  the parcel released and riding downstream, both arms parked;
                with label "-y" (a label that faced the robot, now on the
                upstream face) when release_minus_y, else whatever the turn
                left (job given up: the parcel untouched or released).
      "gone"    the parcel left the belt or the scan line during the job.
    Returns a dict: state, label (after the turn), turn record, handover
    record, released (bool), reason."""
    from .geom import label_dir
    st = SideTurn(rc, eo, params or SideParams(), log=log)
    out = dict(state="riding", released=True, reason="", label=None)
    cell = rc.cell
    lab0 = label_dir(cell.truth.parcel_state(slot).label_normal)[0] if cell.slots[slot].active else "gone"
    out["label0"] = lab0
    if lab0 == "-x" and st.p.minus_x_mode == "tip":
        # a label facing the robot: tip the parcel over its far bottom edge
        # (the label comes up); refside.tipover
        from .tipover import TipOver
        from .tipover import TipParams
        tp_ = TipOver(rc, eo, st.p, tip=TipParams(**(st.p.tip or {})), log=log)
        r = tp_.tip(slot)
        out["tip"] = r
        out["label"] = r.get("label_after")
        out["reason"] = r.get("reason", "")
        out["state"] = "riding" if cell.slots[slot].active else "gone"
        out["released"] = True
        return out
    if lab0 == "+x" and st.p.plus_x_aspect:
        # the roll after the turn needs the parcel's new length along the belt
        # (its width across the belt now) over its height inside the roll's
        # reach: flatter parcels pass untouched (measured on 600 parcels:
        # W/H 2.6-3.0 8/15 readable, 0 stops; above 3.0 1/6
        # readable, 1 protective stop, 2 damaged)
        from .geom import aligned_dims
        ps_ = cell.truth.parcel_state(slot)
        W_, L_, H_ = (float(v_) for v_ in aligned_dims(ps_.size, ps_.quat))
        lim = st.p.plus_x_max_aspect
        if W_ / max(H_, 1e-3) > lim:
            out["reason"] = f"declined (+x: too flat to roll after the turn, W/H {W_ / H_:.2f} > {lim})"
            return out
    if lab0 == "-x" and st.p.minus_x_mode == "decline":
        # a label facing the robot turns onto the upstream face (-y), which
        # no chain of the station finishes today: the parcel passes untouched
        out["reason"] = "declined (-x: no -y chain)"
        return out
    t = st.turn(slot)
    st.set_grip("Right", 0.0)
    out["turn"] = t
    if not cell.slots[slot].active:
        out["state"] = "gone"
        eo.park(None)
        return out
    lab = label_dir(cell.truth.parcel_state(slot).label_normal)[0]
    out["label"] = lab
    if not t.get("ok") or lab not in ("+y", "-y"):
        out["reason"] = "turn: " + str(t.get("reason") or lab)
        eo.back_off(slot=slot)
        eo.park(slot)
        return out
    if lab == "+y" and st.p.plus_x_handover == "free":
        # the turned parcel rides on, released; the right arm parks; the
        # scheduler's own catch and roll take it from here
        st._escape_right(None)
        out["state"] = "riding" if cell.slots[slot].active else "gone"
        out["released"] = True
        out["handover"] = {"mode": "free"}
        return out
    try:
        out["handover"] = st.handover(slot, low_stop=False, justify=2, glide=True)
    except Exception as e:  # noqa: BLE001  (PlanFail of the roll package in use)
        out["reason"] = f"handover: {type(e).__name__}: {e}"
        if cell.slots[slot].active:
            release_caught(st, slot)
        else:
            eo.park(None)
        out["state"] = "riding" if cell.slots[slot].active else "gone"
        return out
    if not cell.slots[slot].active:
        out["state"] = "gone"
        return out
    if lab == "-y" and release_minus_y:
        release_caught(st, slot)
        out["state"] = "riding" if cell.slots[slot].active else "gone"
        out["released"] = True
        return out
    # quality gate: a handover left out of square (or very far off the
    # palm's centre) is not rolled: of 600 measured, rolls from |yaw| > 12
    # deg never succeeded and two of them ended in protective stops; the
    # parcel is let go instead
    fin = (out.get("handover") or {}).get("final") or {}
    off = float(fin.get("com_x", 0.0)) - float(fin.get("palm_x", 0.0))
    jst = (out.get("handover") or {}).get("justify") or {}
    bad_front = float(fin.get("y_face", 0.0)) > st.p.front_max or jst.get("repose_skipped") is not None \
        or getattr(st, "_repose_halt", None) is not None
    if st.p.gate and (abs(float(fin.get("err_deg", 0.0))) > st.p.gate_err_deg or off > st.p.gate_off or bad_front):
        out["reason"] = f"handover gate (yaw {fin.get('err_deg')} deg, off {off:.3f} m, front {fin.get('y_face')}): let go"
        eo.staged = None
        release_caught(st, slot)
        out["state"] = "riding" if cell.slots[slot].active else "gone"
        out["released"] = True
        return out
    out["state"] = "caught"
    out["released"] = False
    return out


def side_module(ctl, slot):
    """Hook for the main controller's scheduler (options["side_module"]):
    ctl.rc, ctl.eo are its RefController and EndOver."""
    return side_job(ctl.rc, ctl.eo, slot)
