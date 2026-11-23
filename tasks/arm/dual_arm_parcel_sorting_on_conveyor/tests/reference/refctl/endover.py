"""End-over-end roll of a parcel about the belt-transverse axis (world x).

For a parcel whose label is on its upstream (-y) or downstream (+y) face. A
rotation about the squeeze line of the two-palm squeeze (world y) cannot turn
such a label up; it needs a quarter turn about world x ("end over end").

The primitive (roll U: top toward upstream about the upstream bottom edge;
label +y -> up, -y -> down, the 180-degree squeeze flip then completes it):

  catch   the left (downstream) palm stops the parcel on the running belt:
          speed-matched soft catch, palm square to the belt, tool horizontal,
          pads across the belt (pad roll 45 deg, less for low faces so that
          both pads sit on the face), centred on the centre of mass.
  stop    the right (upstream) palm, tool tilted 30 deg down, sets a low stop
          at the upstream bottom edge (pad tips 10-30 mm above the belt) with a
          feedforward force; while the parcel tilts its target turns about
          the pivot with the face and its tool ducks (tilts down 1.5 x the
          tilt) under the leaning face; it backs out at min(balance - 8, 35)
          deg.
  push    the left palm's pad contact point on the downstream face is driven
          along the arc about the pivot (target leads the measured tilt by
          <= 5 deg at <= 40 deg/s, J^T F feedforward sized from the gravity
          moment); the pad stays on the parcel's initial centre line (a pad
          that follows the yawing face drags it further: unstable). Out of
          reach, the pad holds and the belt drag on the pivot keeps tilting
          the parcel; no progress for 0.8 s -> abort: the parcel is lowered
          back and held, the arm returns to its catch configuration, retry.
  brake   past the balance the pad holds still along the face (fixed in the
          world in the face's plane), pressed 6 N + 0.3 m g: the face slides
          under it and the pad friction slows the fall.
  release at 88 deg the palms let go; the parcel lands on its former
          upstream face (former downstream face on top) and rides on; both
          palms lift clear and park (paths checked against the station).

Rotation sign: theta about world +x, positive = top toward upstream (-y).

Every command passes a rate clamp (5 rad/s): the cell's 6 rad/s slew limiter
never engages. Privileged reference: reads the parcel's pose, size, mass and
centre of mass from the simulator's truth; writes only joint targets and
gripper commands (no state writes, no welds, no external forces).

Entry point: EndOver(rc, RollParams()).run(slot) -- see README_ENDS.txt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .execu import SL
from .geom import quat_mat
from .kin import rot_axis_angle
from .motion import DT, joint_move
from .paths import Q_PALM_NOM, SG, PlanFail, palm_R

G = 9.81
SIDES = ("Left", "Right")
PAD_TIP = 0.0025        # pad tips lie this far before the TCP along the tool axis
PAD_X = 0.0466          # pad centres at +-this along the TCP x axis (open)


def rot_x(a):
    return rot_axis_angle((1.0, 0.0, 0.0), a)


@dataclass
class RollParams:
    catch_tilt_deg: float = 0.0      # left palm tilt during the catch
    catch_psi_deg: float = 45.0      # pad line at 45 deg: two contacts 6.6 cm apart across the belt
    catch_psi_low_deg: float = 30.0  # ... 30 deg for low parcels (both pads on the face)
    catch_yaw_follow: float = 0.0    # catch palm yaw = this x the parcel's yaw misalignment (0: square to the belt)
    catch_vmax: float = 2.8          # rad/s of the approach to the catch's wait pose
    lip_clear: float = 0.035         # m from the lip's inner face to the near pad centre
    belt_clear: float = 0.012        # m the pusher's wrist / gripper body keeps from the belt (0 = off)
    stroke: int = 0                  # 1: after the landing the pad strokes the new top downstream
    stroke_min_deg: float = 80.0
    stroke_dv: float = 0.25          # m/s stroke speed above the belt speed
    stroke_s: float = 0.3
    stroke_press: float = 0.004      # m the stroke target lies under the top
    stroke_fn: float = 3.0
    stroke_fn_mg: float = 0.3
    stop_vmax: float = 2.2           # rad/s of the stop's approach to its pre-pose
    stop_tmin: float = 0.7           # s minimum of that move
    stop_out_s: float = 0.4          # s of the drop to the stop's outer pose
    stop_in_s: float = 0.35          # s of the press-in
    catch_mid_frac: float = 0.5      # pad-midpoint height of the catch / parcel height
    catch_gap: float = 0.002         # m of the catch pad target downstream of the face (< 0: pressed in)
    catch_t_acc: float = 0.15        # s the palm takes to reach the parcel's speed
    catch_t_dec: float = 0.30        # s of the braking to rest
    stage_stop: int = 0              # 1: the right palm waits above the stop place during the catch
    stage_clear: float = 0.07        # m of the staging pose above the parcel's top
    stage_drop_s: float = 0.5        # s of the drop from the staging pose to the stop pre-pose
    psi_h_split: float = 0.11        # parcel height below which the low pad roll is used
    catch_zfrac: float = 0.40        # catch pad height / parcel height
    catch_zmin: float = 0.05         # lowest pad-midpoint height above the belt
    catch_lo: float = 0.040          # lower pad centre this far above the belt at least (its coupler clears)
    catch_y_face: float = 0.15       # downstream face position at the catch (m)
    stop_tilt_deg: float = 30.0      # right palm (stop) tilt
    stop_z: float = 0.018            # stop TCP height above the belt (m)
    stop_overpress: float = 0.0      # m into the upstream face (force by feedforward)
    stop_force: float = 6.0          # N feedforward (+y) of the stop, base
    stop_force_mg: float = 0.4       # + this times m g
    push_below_top: float = 0.018    # pad centre below the top edge (m)
    push_overpress: float = 0.0      # m into the downstream face (force by feedforward)
    yaw_gain: float = 0.0            # m of push-point shift per rad of parcel yaw (off)
    yaw_dx_max: float = 0.03
    push_fn: float = 20.0            # N normal force (into the face), base
    push_fn_mg: float = 1.0          # + this times m g
    push_ft_gain: float = 1.4        # tangential force = gain * gravity moment / r
    lead_deg: float = 5.0           # target lead over the measured tilt
    rate_deg_s: float = 40.0         # max rate of the target tilt
    commit_margin_deg: float = 10.0
    stop_retreat_before_deg: float = 8.0    # stop leaves at th0 - this
    stop_retreat_max_deg: float = 35.0      # ... and not later than this tilt
    stop_turn: float = 1.5                  # stop palm tilts down (ducks under the leaning face) by this times the tilt
    stop_lift: float = 0.010                # m extra stop height reached at 30 deg tilt
    stop_fz: float = 3.0                    # N upward feedforward of the stop (carries its load)
    stop_fz_mg: float = 0.3
    follow_end_deg: float = 88.0     # follow-through: lower the parcel to this tilt (follow=True)
    follow_rate_deg_s: float = 35.0  # target rate past the balance
    push_fn_late: float = 5.0        # N normal force once the stop has left
    push_fn_late_mg: float = 0.4     # + this times m g
    lag_max_deg: float = 12.0        # (unused)
    free_after_deg: float = 0.0      # past balance + this: no lead, no tangential force (brake from here)
    fall_fn: float = 2.0             # N normal force of the accompanying pad during the fall
    moment_margin: float = 0.25      # driving moment = (1 + this) x gravity moment
    rate_gain: float = 0.6           # force scale reduction per rad/s of tilt rate above plan
    fn_share: float = 0.5            # share of the driving moment from the normal force
    mu_eff: float = 0.9              # friction used to size the normal force (pads: 1.8)
    fn_min: float = 6.0              # N minimum normal force (contact kept)
    fn_min_mg: float = 0.0           # + this times m g
    ret_freeze_ticks: int = 18       # target tilt frozen while the stop backs out
    hold_ticks: int = 40             # out of reach: hold this long without tilt progress, then abort
    slow_deg: float = 0.0            # slow the arc over this many degrees before the balance (0 = off)
    guard: int = 0                   # 1: after its stop role the right palm stands guard upstream of the landing
    brake: int = 1                   # 1: past the balance the pad holds still on the face (friction brake)
    brake_fn: float = 6.0            # N normal force of the braking pad
    brake_fn_mg: float = 0.3         # + this times m g
    brake_fn_v: float = 2.0          # + this x belt speed (m/s) x m g (stronger brake on fast belts)
    brake_slide_max: float = 0.45    # stop braking when the pad has slid this times H along the face
    brake_move: int = 0              # 1: the braking point moves with the belt
    brake_rate_deg_s: float = 50.0   # brake 2: the pad's arc rate past the balance
    brake_lag_deg: float = 10.0      # brake 2: max lag of the pad behind the parcel
    guard_tilt_deg: float = 30.0
    guard_zfrac: float = 0.55        # guard pad height / the parcel's new height (its old length)
    guard_gap: float = 0.02          # m between the guard's pads and the final upstream face
    slow_frac: float = 0.3           # ... down to this fraction of the planned rate
    first_timeout_s: float = 0.0     # test hook (videos): cut the first attempt short (0 = off)
    tool_cap_deg: float = 90.0       # cap of the pusher tool tilt
    push_psi_deg: float = 45.0       # pad line of the pusher (= catch: no turn in contact)
    psi_steps: tuple = (0, 2, -2, 4, -4)   # allowed pad-roll change per tick (deg) when reach needs it
    psi_goal_deg: float = -1.0       # >= 0: during the arc the pad roll moves toward this (deg)
    psi_rate_deg: float = 1.0        # ... at most this many deg per tick
    push_zfrac: float = 0.55         # push pad height / parcel height (if the palm turns)
    turn_s: float = 0.6              # time of the catch -> push turn
    step_max: float = 0.05           # rad per tick of the planned joints
    timeout_s: float = 6.0
    u_rate_max: float = 5.0          # rad/s clamp of the servo command (limiter at 6)
    stop_psi_deg: float = 0.0        # the stop palm's roll about its tool axis (180 = turned over)
    v_park: float = 2.4              # rad/s free moves
    max_rolls: int = 3               # bounded retries of the roll
    follow: int = 1                  # 1: keep the pad on through the fall (lead only), 0: let go at the commit
    mode: int = 1                    # 1: arc to the commit (roll_up), 2: lift and hold (roll_up_b)
    hand_deg: float = 24.0           # mode 2: hand-over tilt (stop leaves, belt couple finishes)
    push_tilt_max_deg: float = 12.0  # mode 2: max tool tilt of the lifting palm
    push_tilt0_deg: float = 0.0      # mode 2: tool tilt at the start of the lift (flush)
    push_zmin: float = 0.05          # mode 2: lowest push pad height (housing clears the belt)
    push_below_frac: float = 0.2     # mode 2: push pad below the top edge by this times H
    hold_fn: float = 4.0             # mode 2: normal force of the holding palm
    stall_s: float = 0.4             # mode 2: nudge the held contact if the tilt stalls this long
    p2_follow: int = 1               # mode 2: after the hand-over keep driving the arc (1) or hold (0)
    # the task's cell
    retreat_x: float = 0.17          # the stop's retreat goes toward the robot only to this x
    column_clear: float = 0.015      # m the pusher's arm keeps from the robot's own column / head
    fence: int = 1                   # command fence (see _command_fence)
    fence_struct_mm: float = 2.0     # ... robot geoms against the station structure (not the belt surface)
    fence_arm_mm: float = 20.0       # ... one arm's wrist / gripper against the other's (20 mm; at 4-10 mm the grippers still touched through servo lag)
    arm_gap_min: float = 0.015       # second arc: the pusher holds when its wrist / gripper would come this close to the other arm's (m)
    letgo_safe: int = 1              # the pusher's let-go after the arc keeps the arm clear of the column
    letgo_clear: float = 0.004       # ... the plain let-go is kept unless its planned way comes closer than this (m)
    jump_check: int = 1              # a pusher pose far from the present posture (another IK branch) is taken
                                     # only if the joint-space way there keeps the column / belt clearances (else: none)
    jump_tol: float = 0.25           # ... "far" = a joint more than this away (rad)
    recatch_y_max: float = 0.155     # farthest downstream face line of a re-catch (0.30 with a tilted palm)
    park_cartesian_near: int = 0     # 1: with a parcel near, park along up then across (rejected)
    guard_mm: float = 8.0            # station guard: non-finger parts kept this far from the station
    guard_gain: float = 2.0
    retreat_y_min: float = 0.17      # ... and keeps its TCP at y <= -this (clear of the column)
    retreat_adapt: int = 0           # (option): while the stop would rise into a parcel that follows it (gap < retreat_adapt_gap), it backs further upstream instead
    retreat_adapt_gap: float = 0.008
    retreat_adapt_step: float = 0.006     # m per tick
    retreat_adapt_max: float = 0.12       # m at most
    light_mass_kg: float = 0.0       # (option, 0 = off): parcels lighter than this get the light-parcel stop retreat
    stop_retreat_max_light_deg: float = 50.0   # ... the stop leaves not later than this tilt (instead of stop_retreat_max_deg)
    retreat_clear_light: int = 0     # ... and its retreat runs low out of the landing zone first (retreat_clear for light parcels only)
    retreat_clear: int = 0           # the stop's retreat first runs low and upstream out of the whole landing zone (H + margin) before it rises
    retreat_clear_margin: float = 0.025   # ... this far beyond the landing zone (m)
    retreat_clear_step: float = 0.008     # ... m per tick of that run (0.4 m/s)
    sq_soft: float = 110.0           # N summed robot force on the parcel above which the palms yield
    arc2_sq_soft: float = 90.0       # ... during the second arc of the half turn
    sq_gain: float = 1.0 / 5000.0    # m of retreat per N above sq_soft, per tick
    sq_max_back: float = 0.08        # m largest retreat offset
    sq_decay: float = 0.85           # per-tick decay of the retreat offset once below sq_soft
    sq_along_normal: int = 0         # 1: yield along the contacts' normal (else along the tool axis)
    sq_hard: float = 1e9             # N: above this also an emergency back-off along the contact normal (off: it disrupted the push)
    sq_em_step: float = 0.012        # m per tick of the emergency back-off
    sq_em_max: float = 0.06
    settle_mid_s: float = 0.25       # intermediate rolls: settle this long after the landing
    settle_mid_w: float = 0.8        # ... landing = angular speed below this (rad/s), on the belt
    hold_slide: float = -1.0         # >= 0: the pusher pushes the parcel back beyond this downstream drift (m)
    hold_slide_max: float = 0.03     # ... by at most this much (m)
    arc2_lead_deg: float = 6.0       # continued half turn: target lead of the second arc
    arc2_rate_deg_s: float = 45.0    # ... its rate
    arc2_end_deg: float = 170.0      # ... end of the driven second arc (total rotation)
    arc2_timeout_s: float = 6.0
    arc1_stall_s: float = 0.0        # > 0: arc-1 stall boost after this long without tilt progress
    arc1_stall_lead_max_deg: float = 8.0
    arc2_stall_s: float = 0.5        # no tilt progress this long -> the lead grows
    arc2_lead_max_m: float = 0.07
    arc2_past_m: float = 0.005       # past the second balance by this (COM upstream of the pivot)
    arc2_follow_deg: float = 0.0     # past the second balance the pad stays on and follows the fall this far (deg; 0: let go at once)
    arc2_follow_fn: float = 4.0      # ... with this normal force (N) + arc2_follow_fn_mg x m g and no tangential push
    arc2_follow_fn_mg: float = 0.2
    arc2_fn: float = 15.0            # N normal force of the pad on the new top face
    arc2_fn_mg: float = 0.8          # + this times m g
    arc2_press: float = 0.003        # m the second arc's pad target lies inside the face
    arc2_lead_m: float = 0.03        # m the second arc's pad target leads the contact along the face
    low_face_h: float = 0.0          # parcels lower than this get the low-face catch (also when no pad roll fits)
    low_tilt_deg: float = 30.0       # tool tilt (down) of the low-face catch
    low_zmin: float = 0.018          # m lowest pad height of the low-face catch
    # reverse roll ("roll D", rolld.py): the top towards downstream about the
    # downstream bottom edge; the left palm is a low stop, the right palm pushes
    rd_stop_tilt_deg: float = 30.0   # stop tool tilt (down)
    rd_stop_z: float = 0.022         # m stop pad height above the belt
    rd_stop_force: float = 4.0       # N feedforward of the stop (-y), base
    rd_stop_force_mg: float = 0.9    # + this times m g (the belt drag)
    rd_push_below: float = 0.015     # m pusher pads below the top edge (at least)
    rd_push_below_frac: float = 0.15 # ... or this times H
    rd_push_fn: float = 12.0         # N normal force of the pusher, base
    rd_push_fn_mg: float = 0.6       # + this times m g
    rd_lead_deg: float = 5.0
    rd_rate_deg_s: float = 40.0
    rd_commit_deg: float = 10.0      # committed at the balance + this
    rd_stop_leave_deg: float = 30.0  # the stop leaves at min(balance - 8, this)
    rd_end_deg: float = 86.0         # let go at this tilt
    rd_brake_fn: float = 5.0         # N past the balance (brake), + 0.3 m g
    rd_timeout_s: float = 6.0
    rd_tool_cap_deg: float = 95.0
    rd_sq_soft: float = 125.0        # squeeze governor limit during roll D
    rd_follow: int = 1               # 1: the stop's pads follow the turning face (about the pivot)
    rd_free_deg: float = 12.0        # the pusher keeps driving to the balance + this, then brakes
    rd_leave_dy: float = 0.12        # m the stop moves downstream when it leaves
    rd_press: float = 0.006          # m the pusher's target lies inside the face (before the balance)
    rd_duck: float = 0.0             # > 0: the stop's tool also turns down with the face (tilt += this x tilt)
    rd_leave_before_deg: float = 8.0 # the stop leaves at balance - this (or rd_stop_leave_deg)
    rd_psi_list: tuple = (90.0, 60.0, 45.0, 30.0, 0.0)   # pusher pad roll candidates (deg)


@dataclass
class RollResult:
    ok: bool = False
    reason: str = ""
    t_start: float = 0.0
    t_end: float = 0.0
    tilt_max_deg: float = 0.0
    committed: bool = False
    held: bool = False
    label_after: str = ""
    label_err_deg: float = 180.0
    peaks: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


class JobAbort(Exception):
    """Raised by the tick when the scheduler's abort check fires (a job is
    ended early: the parcel behind is too close)."""


class EndOver:
    """Wraps a RefController (executor, kinematics, truth helpers)."""

    def __init__(self, rc, params: RollParams | None = None, log=None):
        self.rc = rc
        self.cell = rc.cell
        self.ex = rc.ex
        self.kin = rc.kin
        self.p = params or RollParams()
        self.belt = rc.belt
        self.log = log or (lambda *a: None)
        self.grip = (0.0, 0.0)
        self.peak = dict(stop_force=0.0, push_force=0.0, belt_tip_force=0.0,
                         struct_force=0.0, belt_tip_ticks=0, struct_ticks=0,
                         struct_geom="",
                         # wrist-camera housings: peak force and
                         # ticks of any contact, by what they touched
                         housing_parcel_force=0.0, housing_parcel_ticks=0,
                         housing_other_force=0.0, housing_other_ticks=0,
                         housing_geom="", housing_phase="")
        self.peak_cmd_speed = 0.0
        self._qd_f = np.zeros(14)
        self.trace = []
        self.trace_contacts = False
        self._R0 = np.eye(3)

    # ------------------------------------------------------------------
    # truth helpers
    # ------------------------------------------------------------------
    def body(self, slot):
        c = self.cell
        if not c.slots[slot].active:
            raise PlanFail("parcel gone (fell off or left the belt)")
        q = c._slot_qadr[slot]
        pos = c.data.qpos[q:q + 3].copy()
        R = quat_mat(c.data.qpos[q + 3:q + 7])
        p = c.slots[slot].parcel
        return pos, R, np.asarray(p.size, float), np.asarray(p.com_offset, float), float(p.mass)

    def tilt_x(self, R, R0):
        Rr = R @ R0.T
        return math.atan2(Rr[2, 1], Rr[1, 1])

    def _is_finger_geom(self, name):
        """Unnamed gripper geoms are reported as 'geom<id>': finger links
        (driver, coupler, spring link, follower) count as fingertip parts."""
        import mujoco
        if not hasattr(self, "_fg_cache"):
            self._fg_cache = {}
        if name not in self._fg_cache:
            ok = False
            try:
                gid = int(name[4:])
                bn = mujoco.mj_id2name(self.cell.model, mujoco.mjtObj.mjOBJ_BODY,
                                       int(self.cell.model.geom_bodyid[gid])) or ""
                ok = any(t in bn for t in ("driver", "coupler", "spring_link", "follower", "_pad"))
            except ValueError:
                ok = False
            self._fg_cache[name] = ok
        return self._fg_cache[name]

    def contacts_step(self, slot=None):
        """Per-tick force bookkeeping: robot-belt-hardware and
        robot-parcel forces."""
        tr = self.cell.truth
        f_stop = f_push = 0.0
        n_acc = {"Left": np.zeros(3), "Right": np.zeros(3)}
        tip = struct = 0.0
        hp = ho = 0.0
        for c in tr.contacts(kinds=("robot-parcel", "robot-belt")):
            if "wrist_cam_housing" in c.robot_geom:
                if c.kind == "robot-parcel":
                    hp = max(hp, c.force_normal)
                else:
                    ho = max(ho, c.force_normal)
                pk_ = self.peak
                if max(hp, ho) >= max(pk_["housing_parcel_force"], pk_["housing_other_force"]):
                    pk_["housing_geom"] = f"{c.robot_geom}/{c.other_geom}"
                    pk_["housing_phase"] = getattr(self, "phase", "")
            if c.kind == "robot-parcel":
                if c.robot_geom.startswith("R_") or c.robot_geom.startswith("Right"):
                    f_stop += c.force_normal
                    n_acc["Right"] = n_acc["Right"] + c.force_normal * c.normal
                else:
                    f_push += c.force_normal
                    n_acc["Left"] = n_acc["Left"] + c.force_normal * c.normal
            else:
                through_belt = False
                if c.other_geom == "belt_shell_col":
                    # the shell's top lies 2 mm under the moving slab: a pad
                    # touching it inside the slab's width has pressed through
                    # the belt surface -> book it as fingertip-on-belt
                    st_ = self.cell.station
                    through_belt = (abs(float(c.pos[0]) - st_.belt_center_x) < st_.half_width
                                    and float(c.pos[2]) > st_.belt_height - 0.006)
                rb = c.robot_geom
                finger_part = ("_pad" in rb) or rb.startswith(("geom",)) and self._is_finger_geom(rb)
                if (c.other_geom == "belt_surface" or through_belt) and not finger_part:
                    # an arm link or the gripper body on the belt: structure class
                    if c.force_normal > struct:
                        struct = c.force_normal
                        if struct > self.peak["struct_force"]:
                            self.peak["struct_geom"] = f"{c.robot_geom}/{c.other_geom}"
                            self.peak["struct_pos"] = [round(float(v), 3) for v in c.pos]
                            self.peak["struct_t"] = round(self.cell.time, 2)
                elif c.other_geom == "belt_surface" or through_belt:
                    if c.force_normal > tip:
                        tip = c.force_normal
                        if tip > self.peak["belt_tip_force"]:
                            self.peak["belt_tip_geom"] = c.robot_geom
                else:
                    if c.force_normal > struct:
                        struct = c.force_normal
                        if struct > self.peak["struct_force"]:
                            self.peak["struct_geom"] = f"{c.robot_geom}/{c.other_geom}"
                            self.peak["struct_pos"] = [round(float(v), 3) for v in c.pos]
                            self.peak["struct_t"] = round(self.cell.time, 2)
        pk = self.peak
        self._f_last = (f_push, f_stop)
        self._n_last = {k: (v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-9 else None)
                        for k, v in n_acc.items()}
        pk["stop_force"] = max(pk["stop_force"], f_stop)
        pk["push_force"] = max(pk["push_force"], f_push)
        if tip > pk["belt_tip_force"]:
            pk["belt_tip_phase"] = getattr(self, "phase", "")
        if struct > pk["struct_force"]:
            pk["struct_phase"] = getattr(self, "phase", "")
        pk["belt_tip_force"] = max(pk["belt_tip_force"], tip)
        pk["struct_force"] = max(pk["struct_force"], struct)
        pk["belt_tip_ticks"] += int(tip > 0)
        pk["struct_ticks"] += int(struct > 0)
        pk["housing_parcel_force"] = max(pk["housing_parcel_force"], hp)
        pk["housing_other_force"] = max(pk["housing_other_force"], ho)
        pk["housing_parcel_ticks"] += int(hp > 0)
        pk["housing_other_ticks"] += int(ho > 0)
        return f_stop, f_push

    # ------------------------------------------------------------------
    # low-level motion
    # ------------------------------------------------------------------
    def _command_fence(self, q14):
        """Command fence: before a tick's targets go out, each arm's
        planned posture is checked: its wrist and gripper geoms against the
        station structure (rails, lip, conveyor body, hardware boxes; not the
        belt surface) and against the other arm's wrist / gripper. An arm
        whose target would come closer than fence_struct_mm / fence_arm_mm
        AND closer than it is now keeps its previous target this tick (true
        distances: the cell's zero contact margin hides near misses from
        contact lists). A reactive guard cannot stop a target that runs deep
        into a rail; this keeps such a target from being sent."""
        import mujoco
        c = self.cell
        m = c.model
        if not hasattr(self, "_fc"):
            wrist = {s_: [] for s_ in SIDES}
            for g in (int(x) for x in c.robot_moving_geoms):
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
                for s_ in SIDES:
                    if bn.startswith(s_[0] + "_") or bn.startswith(s_ + "_Wrist") or bn.startswith(s_ + "_Elbow"):
                        wrist[s_].append(g)
            struct = sorted(int(h) for h in c.structure_geoms)
            self._fc = (wrist, struct, mujoco.MjData(m), mujoco.MjData(m), np.zeros(6))
            self.peak["fence_holds"] = 0
        wrist, struct, dq_, dn_, ft = self._fc
        moved = {s_: bool(np.abs(q14[SL[s_]] - self.ex.q[SL[s_]]).max() > 1e-7) for s_ in SIDES}
        if not (moved["Left"] or moved["Right"]):
            return q14
        ths = self.p.fence_struct_mm / 1000.0
        tha = self.p.fence_arm_mm / 1000.0
        qadr = self.ex.dyn.qadr
        dq_.qpos[:] = c.data.qpos
        dq_.qpos[qadr] = q14
        mujoco.mj_kinematics(m, dq_)
        now = [False]

        def cur():
            if not now[0]:
                dn_.qpos[:] = c.data.qpos
                mujoco.mj_kinematics(m, dn_)
                now[0] = True
            return dn_

        def smin(d, s_):
            best = 1.0
            for g in wrist[s_]:
                for h in struct:
                    best = min(best, mujoco.mj_geomDistance(m, d, g, h, ths + 0.005, ft))
            return best

        def amin(d):
            best = 1.0
            for g1 in wrist["Left"]:
                for g2 in wrist["Right"]:
                    best = min(best, mujoco.mj_geomDistance(m, d, g1, g2, tha + 0.005, ft))
            return best
        out = q14
        for s_ in SIDES:
            if not moved[s_]:
                continue
            a = smin(dq_, s_)
            if a < ths and a < smin(cur(), s_) - 5e-4:
                if out is q14:
                    out = q14.copy()
                out[SL[s_]] = self.ex.q[SL[s_]]
                self.peak["fence_holds"] += 1
        if out is not q14:
            dq_.qpos[qadr] = out
            mujoco.mj_kinematics(m, dq_)
        a = amin(dq_)
        if a < tha and a < amin(cur()) - 5e-4:
            if out is q14:
                out = q14.copy()
            out[:] = self.ex.q
            self.peak["fence_holds"] += 1
        return out

    def _station_guard(self, q14):
        """Station guard (guard_mm > 0): for each
        arm, the closest pair of its non-finger collision geoms (links,
        gripper body) and the station's collision geoms (belt surface, rails,
        lip, conveyor body, station hardware) in the current state; if closer
        than guard_mm, the arm's joint target moves that point away from the
        station by guard_gain x the deficit (point Jacobian, damped least
        squares). Finger parts touching the belt are left to the primitive."""
        import mujoco
        c = self.cell
        m, d = c.model, c.data
        if not hasattr(self, "_sg"):
            from parcelcell.robot import JOINT_NAMES
            sel = {s_: [] for s_ in SIDES}
            fsel = {s_: [] for s_ in SIDES}
            for g in (int(x) for x in c.robot_moving_geoms):
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
                finger = any(t in bn for t in ("driver", "coupler", "spring_link", "follower", "_pad"))
                for s_ in SIDES:
                    if nm.startswith(s_) or bn.startswith(s_) or bn.startswith(s_[0] + "_"):
                        (fsel if finger else sel)[s_].append(g)
            self._sg_fingers = fsel
            dof = {s_: np.array([int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES[k])])
                                 for k in range(14)][SL[s_]]) for s_ in SIDES}
            hw = sorted(set(int(h) for h in c.structure_geoms) | {int(c._belt_surface)})
            # the fingers too, against the rails / lip / conveyor body /
            # hardware (not the belt surface, which the primitives touch)
            self._sg_struct = sorted(int(h) for h in c.structure_geoms)
            self._sg = (sel, hw, dof, np.zeros(6), np.zeros((3, m.nv)))
            self.peak["guard_ticks"] = 0
            self.peak["guard_min_mm"] = 1e3
        sel, hw, dof, ft, jacp = self._sg
        thr = self.p.guard_mm / 1000.0
        out = None
        for s_ in SIDES:
            best = None
            for g in sel[s_]:
                for h in hw:
                    dist = mujoco.mj_geomDistance(m, d, g, h, thr + 0.01, ft)
                    if dist < thr and (best is None or dist < best[0]):
                        best = (dist, ft.copy(), g)
            for g in self._sg_fingers[s_]:
                for h in self._sg_struct:
                    dist = mujoco.mj_geomDistance(m, d, g, h, thr + 0.01, ft)
                    if dist < thr and (best is None or dist < best[0]):
                        best = (dist, ft.copy(), g)
            if best is None:
                continue
            dist, fromto, g = best
            self.peak["guard_ticks"] += 1
            self.peak["guard_min_mm"] = min(self.peak["guard_min_mm"], round(1000 * dist, 1))
            n = fromto[:3] - fromto[3:]
            nn = float(np.linalg.norm(n))
            n = n / nn if (dist > 1e-4 and nn > 1e-6) else np.array([0.0, 0.0, 1.0])
            jacp[:] = 0.0
            mujoco.mj_jac(m, d, jacp, None, fromto[:3], int(m.geom_bodyid[g]))
            J = jacp[:, dof[s_]]
            dx = n * (thr - dist) * self.p.guard_gain
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), dx)
            dq = np.clip(dq, -0.02, 0.02)
            if out is None:
                out = q14.copy()
            out[SL[s_]] += dq
        return q14 if out is None else out

    def tick(self, q14, forces=None, grip=None):
        """One 20 ms tick: inverse-dynamics feedforward through the position
        servos (as Executor.tick) plus a rate clamp on the command so that
        the cell's slew limiter (6 rad/s) never engages."""
        ex = self.ex
        q14 = np.asarray(q14, float)
        # a protective stop of the cell's safety watch: wait until it clears,
        # take the robot's state over and end the job (the scheduler releases
        # and parks)
        w_ = self.cell
        watch = getattr(getattr(w_, "command_gate", None), "__self__", None)
        if watch is not None and getattr(watch, "stopped", False):
            while watch.stopped and not w_.done:
                w_.step(w_._q_applied.copy(), w_._grip_cmd.copy())
            self.resync()
            self.n_stops_seen = getattr(self, "n_stops_seen", 0) + 1
            if not getattr(self, "_in_exit", False):
                raise JobAbort("protective stop")
        chk = getattr(self, "abort_check", None)
        if chk is not None and chk():
            self.abort_check = None
            raise JobAbort("the parcel behind is too close")
        if self.p.fence:
            q14 = self._command_fence(q14)
        if self.p.guard_mm > 0:
            q14 = self._station_guard(q14)
        # squeeze governor: when the summed robot force on the
        # parcel (previous tick, truth) exceeds sq_soft, each palm in contact
        # yields along its tool axis in proportion to its share and its
        # feedforward force is scaled down; the planner's own state (ex.q)
        # is not changed by it
        fL, fR = getattr(self, "_f_last", (0.0, 0.0))
        ftot = fL + fR
        off = getattr(self, "_sq_back", None)
        if off is None:
            off = self._sq_back = {"Left": 0.0, "Right": 0.0}
        p_ = self.p
        scale = {"Left": 1.0, "Right": 1.0}
        for s, fs in (("Left", fL), ("Right", fR)):
            if ftot > p_.sq_soft and fs > 1.0:
                # integrating yield: keeps backing off while the force stays
                # high (a parcel driven into a held palm by the belt follows
                # a fixed retreat)
                off[s] = min(p_.sq_max_back, off[s] + p_.sq_gain * (ftot - p_.sq_soft) * fs / ftot)
                scale[s] = p_.sq_soft / ftot
            elif ftot < 0.8 * p_.sq_soft:
                off[s] = off[s] * p_.sq_decay
                em = getattr(self, "_sq_em", None)
                if em is not None:
                    em[s] = em[s] * p_.sq_decay
            if off[s] < 1e-5:
                off[s] = 0.0
        q_cmd = q14
        em0 = getattr(self, "_sq_em", {"Left": 0.0, "Right": 0.0})
        if off["Left"] > 0 or off["Right"] > 0 or em0.get("Left", 0) > 0 or em0.get("Right", 0) > 0:
            q_cmd = q14.copy()
            for s in SIDES:
                if off[s] <= 0 and em0.get(s, 0.0) <= 0 and not (ftot > p_.sq_hard):
                    continue
                k = self.kin[s]
                q7 = q14[SL[s]]
                Jp, _ = k.jac(q7)
                _, Rt = k.fk(q7)
                # yield along the contacts' force-weighted normal (from the
                # parcel to the robot) when known, else back along the tool
                n_c = (getattr(self, "_n_last", None) or {}).get(s)
                d_ = n_c if (n_c is not None and p_.sq_along_normal) else -Rt[:, 2]
                dx = d_ * off[s]
                # emergency (a jam: the cell's damage rule counts 3 ticks above
                # its squeeze limit): also back off along the contact normal
                if ftot > p_.sq_hard and n_c is not None:
                    em = getattr(self, "_sq_em", {"Left": 0.0, "Right": 0.0})
                    em[s] = min(p_.sq_em_max, em[s] + p_.sq_em_step)
                    self._sq_em = em
                    self.peak["sq_emergency_ticks"] = self.peak.get("sq_emergency_ticks", 0) + 1
                em_ = getattr(self, "_sq_em", {}).get(s, 0.0)
                if em_ > 0 and n_c is not None:
                    dx = dx + n_c * em_
                dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + 1e-4 * np.eye(3), dx)
                q_cmd[SL[s]] = q7 + dq
            self.peak["sq_back_max"] = max(self.peak.get("sq_back_max", 0.0), max(off.values()))
        tau_x = None
        for s in SIDES:
            F = None if forces is None else forces.get(s)
            ex.force[s] = None
            if F is None:
                continue
            J = ex.dyn.jac_point(q_cmd, None, site=f"{s}_tcp")
            t = J.T @ (np.asarray(F, float) * scale[s])
            tau_x = t if tau_x is None else tau_x + t
        qd_raw = (q14 - ex.q) / DT
        self._qd_f = 0.5 * self._qd_f + 0.5 * qd_raw
        u = ex.dyn.command(q_cmd, self._qd_f, None, tau_x)
        if ex._u_prev is not None:
            lim = self.p.u_rate_max * DT
            du = np.clip(u - ex._u_prev, -lim, lim)
            u = ex._u_prev + du
            self.peak_cmd_speed = max(self.peak_cmd_speed, float(np.abs(du).max() / DT))
        ex._u_prev = u
        ex.q = q14.copy()
        ex.qd = self._qd_f.copy()
        g = self.grip if grip is None else grip
        ex.grip = np.asarray(g, float)
        self.cell.step(u, g)
        js = self.cell.joint_state()
        ex.peak_qd = np.maximum(ex.peak_qd, np.abs(js.qd))
        ex.peak_tau = np.maximum(ex.peak_tau, np.abs(js.tau))
        ex.ticks += 1
        for h in ex.hooks:
            h(ex)
        self.rc._monitor()
        self.contacts_step()

    def resync(self):
        """Planned state from the measured posture and the applied command
        (after a protective stop or a hand-over between executors)."""
        js = self.cell.joint_state()
        self.ex.q = js.q.copy()
        self.ex.qd = np.zeros(14)
        self.ex._u_prev = js.q_target.copy()
        self._qd_f = np.zeros(14)
        self._sq_back = {"Left": 0.0, "Right": 0.0}
        self._sq_em = {"Left": 0.0, "Right": 0.0}

    def ik_axis(self, side, pos, axis, q0, iters=60):
        q, ok, ep, er = self.kin[side].ik(pos, axis=axis, q0=q0, iters=iters,
                                          margin=0.005, null_gain=0.0)
        return q, ok, ep

    def ik_R(self, side, pos, R, q0, iters=80, rot_tol=0.02, pos_tol=2e-3):
        q, ok, ep, er = self.kin[side].ik(pos, R=R, q0=q0, iters=iters,
                                          margin=0.005, null_gain=0.0,
                                          rot_tol=rot_tol, pos_tol=pos_tol)
        return q, ok, ep

    _LIBS = {}

    def glob_R(self, side, pos, R, q0=None):
        k = self.kin[side]
        for sd in ([q0] if q0 is not None else []) + [Q_PALM_NOM[side]]:
            q, ok, ep, er = k.ik(pos, R=R, q0=sd, q_rest=Q_PALM_NOM[side],
                                 iters=300, margin=0.01)
            if ok:
                return q
        from .kin import SeedLib, ik_global
        if side not in EndOver._LIBS:
            EndOver._LIBS[side] = SeedLib(k, n=5000, seed=7 if side == "Left" else 8)
        return ik_global(k, EndOver._LIBS[side], pos, R=R, k=10,
                         q_rest=Q_PALM_NOM[side])

    def move_cart(self, side, p1, R1=None, axis1=None, T=0.6, forces=None,
                  hold_other=True):
        """Straight-line move of one arm's TCP with a smooth profile
        (closed IK per tick, continuous)."""
        k = self.kin[side]
        q = self.ex.q[SL[side]].copy()
        p0, R0 = k.fk(q)
        n = max(2, int(math.ceil(T / DT)))
        for i in range(1, n + 1):
            u = i / n
            s = u * u * u * (10 - 15 * u + 6 * u * u)
            p = p0 + (np.asarray(p1) - p0) * s
            if R1 is not None:
                from .kin import rot_err
                v = rot_err(R0, R1)
                ang = float(np.linalg.norm(v))
                R = R0 if ang < 1e-9 else rot_axis_angle(v / ang, ang * s) @ R0
                qn, ok, ep = self.ik_R(side, p, R, q)
            elif axis1 is not None:
                a0 = R0[:, 2]
                a = a0 + (np.asarray(axis1) - a0) * s
                qn, ok, ep = self.ik_axis(side, p, a / np.linalg.norm(a), q)
            else:
                qn, ok, ep = self.ik_axis(side, p, R0[:, 2], q)
            step = np.abs(qn - q).max()
            if step > self.p.step_max:
                qn = q + (qn - q) * (self.p.step_max / step)
            q = qn
            row = self.ex.q.copy()
            row[SL[side]] = q
            f = forces(i) if callable(forces) else forces
            self.tick(row, f)
            if self.cell.done:
                return False
        return True

    def plan_cart(self, side, p1, R1, T):
        """Joint rows of move_cart's straight-line move (planned, not run)."""
        from .kin import rot_err
        q = self.ex.q[SL[side]].copy()
        p0, R0 = self.kin[side].fk(q)
        n = max(2, int(math.ceil(T / DT)))
        v = rot_err(R0, R1)
        ang = float(np.linalg.norm(v))
        rows = []
        for i in range(1, n + 1):
            u = i / n
            s = u * u * u * (10 - 15 * u + 6 * u * u)
            p = p0 + (np.asarray(p1) - p0) * s
            R = R0 if ang < 1e-9 else rot_axis_angle(v / ang, ang * s) @ R0
            qn, ok, ep = self.ik_R(side, p, R, q)
            step = np.abs(qn - q).max()
            if step > self.p.step_max:
                qn = q + (qn - q) * (self.p.step_max / step)
            q = qn
            rows.append(q.copy())
        return rows

    def move_cart_safe(self, side, targets, R1, T, forces=None, clear=None):
        """move_cart to the first of the targets whose planned way keeps the
        arm at least `clear` (default letgo_clear) from the robot's own
        column / head; if none does, the way with the largest clearance."""
        clear = self.p.letgo_clear if clear is None else clear
        best = None
        for j, p1 in enumerate(targets):
            rows = self.plan_cart(side, p1, R1, T)
            g = min(self.arm_column_gap(side, q) for q in rows[1::2] + [rows[-1]])
            if j == 0:
                self.peak["letgo_gap_first_mm"] = min(self.peak.get("letgo_gap_first_mm", 1000.0), 1000.0 * float(g))
            if g >= clear:
                best = (g, p1, j)
                break
            if best is None or g > best[0]:
                best = (g, p1, j)
        self.peak["letgo_gap_min"] = min(self.peak.get("letgo_gap_min", 1.0), float(best[0]))
        if best[2] > 0:
            self.peak["letgo_alt"] = self.peak.get("letgo_alt", 0) + 1
        # the chosen move is run as move_cart itself (closed IK per tick)
        return self.move_cart(side, best[1], R1=R1, T=T, forces=forces)

    def joint_goal(self, q14, vmax=2.5, tmin=0.4, forces=None):
        Q, Qd, _ = joint_move(self.ex.q, q14, vmax=vmax, tmin=tmin)
        for i in range(len(Q)):
            f = forces(i) if callable(forces) else forces
            self.tick(Q[i], f)
            if self.cell.done:
                return False
        return True

    def hold(self, seconds, forces=None, check=None):
        n = max(1, int(round(seconds / DT)))
        for i in range(n):
            f = forces(i) if callable(forces) else forces
            self.tick(self.ex.q.copy(), f)
            if self.cell.done or (check is not None and check()):
                return i + 1
        return n

    # ------------------------------------------------------------------
    # geometry of the parcel now (world-aligned dims)
    # ------------------------------------------------------------------
    def dims(self, slot):
        from .geom import aligned_dims, world_dims
        pos, R, size, com, mass = self.body(slot)
        q = self.cell.data.qpos[self.cell._slot_qadr[slot] + 3:self.cell._slot_qadr[slot] + 7]
        d = aligned_dims(size, q)
        _, yaw_err = world_dims(size, q)
        return pos, R, d, yaw_err, mass

    # ------------------------------------------------------------------
    # catch: the left palm stops the parcel (speed-matched soft catch)
    # ------------------------------------------------------------------
    def catch(self, slot, y_face=None, timeout=14.0, low_stop=False, tilt_deg=None):
        p = self.p
        pos, R, (W, L, H), yaw_err, mass = self.dims(slot)
        v = max(self.rc.belt_speed(), 0.02)
        mg = mass * G
        if y_face is None:
            # as far downstream as the left palm's stop reach allows; the
            # stop (right palm) then sits at y_u = y_face - L >= -0.15
            y_face = float(np.clip(p.catch_y_face, -0.15 + L, 0.155))
        tilt = math.radians(p.catch_tilt_deg)
        # largest pad roll whose two pads both fit on the face with the lower
        # pad's lowest point >= 15 mm above the belt and the upper one 12 mm
        # below the top edge
        psi_deg = p.catch_psi_deg
        low = False
        for cand in (p.catch_psi_deg, 30.0, 20.0, 10.0, 0.0):
            dz_c = PAD_X * math.sin(math.radians(cand))
            if p.catch_lo + 2 * dz_c + 0.012 <= H:
                psi_deg = cand
                break
        else:
            low = True
        if H < p.low_face_h:
            low = True
        tilted = tilt_deg is not None and tilt_deg > 0.5
        if low:
            # a low face (flat parcel): pads side by side across the
            # belt (pad roll 0), the tool tilted down so that the pad tips
            # sit low on the face while the gripper base clears the belt
            psi_deg = 0.0
            tilt = math.radians(p.low_tilt_deg)
        self._low_catch = low
        if tilted and not low:
            # a catch further downstream than the horizontal palm reaches: the
            # tool tilted down (the pusher then keeps at least this tilt)
            tilt = math.radians(float(tilt_deg))
        psi_c = math.radians(psi_deg)
        # the stop palm square to the belt motion (yaw 0): an oblique palm
        # deflects the parcel sideways; a square one squares the parcel
        Rl = palm_R("Left", psi_c, yaw_err * self.p.catch_yaw_follow, tilt)
        com_w = pos + R @ np.asarray(self.cell.slots[slot].parcel.com_offset, float)
        # the near finger (and its coupler link) keeps clear of the lip
        x_lo = max(0.22, 0.150 + PAD_X * math.cos(psi_c) + p.lip_clear)
        x_s = float(np.clip(com_w[0], x_lo, 0.33))
        dz = PAD_X * abs(math.sin(psi_c))
        z_lo = max(p.catch_zmin, p.catch_lo + dz)
        z_s = self.belt + float(np.clip(p.catch_mid_frac * H, z_lo, max(z_lo, H - dz - 0.012)))
        if low:
            z_s = self.belt + float(np.clip(0.5 * H, p.low_zmin, max(p.low_zmin, H - 0.012)))
        if low_stop:
            # the stop of a reverse roll: pads side by side across the belt,
            # low at the downstream face's bottom edge, tool tilted down
            psi_c = 0.0
            tilt = math.radians(p.rd_stop_tilt_deg)
            Rl = palm_R("Left", 0.0, 0.0, tilt)
            x_lo = max(0.22, 0.150 + PAD_X + p.lip_clear)
            x_s = float(np.clip(com_w[0], x_lo, 0.33))
            z_s = self.belt + float(min(p.rd_stop_z, max(0.012, H - 0.015)))
            self._low_catch = low = True
        # TCP at the pads' midpoint, pad tips on the face at (x_s, y_face, z_s)
        pad_off = -PAD_TIP * Rl[:, 2]
        stop = np.array([x_s, y_face + self.p.catch_gap, z_s]) - pad_off
        T_acc, T_d = self.p.catch_t_acc, self.p.catch_t_dec
        y_w = stop[1] - v * (T_acc + T_d) / 2
        wait = stop.copy()
        wait[1] = y_w
        qw = self.glob_R("Left", wait, Rl, self.ex.q[SL["Left"]])
        if qw is None:
            raise PlanFail("catch: wait pose unreachable")
        q_goal = self.ex.q.copy()
        q_goal[SL["Left"]] = qw
        self.staged = None
        if p.stage_stop:
            # the right palm waits above the parcel's path, over the place
            # where the upstream face will stop, so that the stop can be set
            # right after the catch (a parcel held by one palm only drifts)
            ts_ = math.radians(p.stop_tilt_deg)
            Rr_ = palm_R("Right", 0.0, 0.0, ts_)
            x_st_ = float(np.clip(pos[0], 0.215, 0.34))
            y_u_ = y_face - L
            stg = np.array([x_st_, y_u_ - 0.05, self.belt + H + p.stage_clear])
            qs_ = self.glob_R("Right", stg, Rr_, self.ex.q[SL["Right"]])
            if qs_ is not None:
                q_goal[SL["Right"]] = qs_
                self.staged = dict(pos=stg, R=Rr_)
        Qm, _, _ = joint_move(self.ex.q, q_goal, vmax=self.p.catch_vmax, tmin=0.5)
        y_front = pos[1] + L / 2
        t_start = (y_w + v * T_acc / 2 - y_front) / v - T_acc
        if t_start < len(Qm) * DT + 0.05:
            raise PlanFail(f"catch: too late ({t_start:.2f} s)")
        # streams: the approach must not sweep another parcel riding
        # the belt (predicted at belt speed); wait while it would, as long as
        # the catch's timing allows
        n_wait = 0
        while self.rc.path_collides(Qm, margin=0.015, exclude=(slot,)):
            pos, R, (W, L, H), yaw_err, mass = self.dims(slot)
            t_start = (y_w + v * T_acc / 2 - (pos[1] + L / 2)) / v - T_acc
            if t_start < len(Qm) * DT + 0.25 or self.cell.done:
                raise PlanFail("catch: approach blocked by another parcel")
            self.hold(0.1)
            n_wait += 1
        self.catch_waited = n_wait
        for row in Qm:
            self.tick(row)
        pos, R, (W, L, H), yaw_err, mass = self.dims(slot)
        t_left = (y_w + v * T_acc / 2 - (pos[1] + L / 2)) / v - T_acc
        if t_left > DT:
            self.hold(t_left)
        kL = self.kin["Left"]
        qq = self.ex.q[SL["Left"]].copy()
        n1, n2 = int(round(T_acc / DT)), int(round(T_d / DT))
        yy = y_w
        for i in range(n1 + n2):
            t = (i + 1) * DT
            vp = v * t / T_acc if i < n1 else v * max(0.0, 1 - (t - T_acc) / T_d)
            yy += vp * DT
            tgt = stop.copy()
            tgt[1] = min(yy, stop[1])
            qn, ok, ep = self.ik_R("Left", tgt, Rl, qq, iters=60)
            qq = qn
            row = self.ex.q.copy()
            row[SL["Left"]] = qn
            lvl = min(1.0, max(0.0, (i - n1 + 2) / 10.0))
            self.tick(row, {"Left": np.array([0.0, -0.8 * mg * lvl, 0.0])})
        hist = []

        def stopped():
            if not self.cell.slots[slot].active:
                return True
            hist.append(float(self.body(slot)[0][1]))
            return len(hist) >= 10 and (max(hist[-10:]) - min(hist[-10:])) < 0.003
        self.hold(timeout, forces={"Left": np.array([0.0, -0.8 * mg, 0.0])}, check=stopped)
        if not self.cell.slots[slot].active or len(hist) < 10 or \
                (max(hist[-10:]) - min(hist[-10:])) > 0.004:
            raise PlanFail("catch failed")
        self.catch_pose = dict(x=x_s, z=z_s, R=Rl, tilt=tilt, psi=psi_c, pad_off=pad_off,
                               q=self.ex.q[SL["Left"]].copy(), low=bool(low or tilted))
        return True

    # ------------------------------------------------------------------
    # roll U: top toward upstream about the upstream bottom edge
    # ------------------------------------------------------------------
    def trace_row(self, slot, phase, th_t=None):
        self.phase = phase
        if not self.cell.slots[slot].active:
            return
        ps = self.cell.truth.parcel_state(slot)
        pos, R, *_ = self.body(slot)
        th = self.tilt_x(R, self._R0)
        Rr = R @ self._R0.T
        roll_y = math.degrees(math.atan2(-Rr[2, 0], Rr[2, 2]))
        yaw = math.degrees(math.atan2(Rr[1, 0], Rr[0, 0]))
        f_st = f_pu = 0.0
        tip = 0.0
        for c in self.cell.truth.contacts(kinds=("robot-parcel", "robot-belt")):
            if c.kind == "robot-parcel":
                if c.robot_geom.startswith("R_"):
                    f_st += c.force_normal
                else:
                    f_pu += c.force_normal
            elif c.other_geom == "belt_surface":
                tip = max(tip, c.force_normal)
        cdet = []
        if getattr(self, "trace_contacts", False):
            for c in self.cell.truth.contacts(kinds=("robot-parcel",)):
                cdet.append((c.robot_geom, round(float(c.pos[0]), 3), round(float(c.pos[1]), 3),
                             round(float(c.pos[2]), 3), round(c.force_normal, 1)))
        tl = self.cell.tcp_pose("Left")[1]
        self.trace.append(dict(t=round(self.cell.time, 3), ph=phase, cd=cdet,
                               lx=[round(float(v), 2) for v in tl[:, 0]],
                               lz=[round(float(v), 2) for v in tl[:, 2]],
                               th=round(math.degrees(th), 1),
                               tht=None if th_t is None else round(math.degrees(th_t), 1),
                               ry=round(roll_y, 1), yaw=round(yaw, 1),
                               y=round(float(pos[1]), 3), z=round(float(pos[2]), 3),
                               fst=round(f_st, 1), fpu=round(f_pu, 1), tip=round(tip, 1),
                               w=round(float(np.linalg.norm(ps.angvel)), 2)))

    def pusher_pose(self, c, n_in, th_tool, psi_pref, yaw, q0, cap):
        """Left palm pose with its lower pad (left_pad1) tip at the contact
        point c, tool axis along the face's inward normal (tilt capped),
        pad line rotated by psi about the tool axis (90 deg = pads stacked
        along the face, one central contact). Tries psi near psi_pref."""
        p = self.p
        tt = min(th_tool, cap)
        cp_ = getattr(self, "catch_pose", None)
        if cp_ is not None and cp_.get("low"):
            tt = max(tt, float(cp_["tilt"]))
        best = None
        for dpsi in self.p.psi_steps:
            psi = psi_pref + math.radians(dpsi)
            R = palm_R("Left", psi, yaw, tt)
            tcp = c + R[:, 2] * (PAD_TIP + p.push_overpress)
            q, ok, ep = self.ik_R("Left", tcp, R, q0, iters=60)
            if ok and p.belt_clear > 0 and self.arm_belt_gap("Left", q) < p.belt_clear:
                ok = False          # wrist / gripper body would touch the belt
            if ok and p.column_clear > 0 and self.arm_column_gap("Left", q) < p.column_clear:
                ok = False          # the arm would touch the robot's own column / head
            if ok and np.abs(q - q0).max() < 0.25:
                return q, psi, R, tcp
            if best is None and ok:
                if p.jump_check and np.abs(q - q0).max() >= p.jump_tol and not self._jump_clear("Left", q0, q):
                    continue        # the way to another IK branch would sweep the arm into the column / belt
                best = (q, psi, R, tcp)
        return (None, psi_pref, None, None) if best is None else best

    def _jump_clear(self, side, q0, q1, n=6):
        """The straight joint-space way q0 -> q1 keeps the arm clear of the
        robot's column / head and of the belt (the pusher's clearances)."""
        p = self.p
        for u in np.linspace(0.0, 1.0, n + 1)[1:]:
            qi = q0 + (q1 - q0) * u
            if p.column_clear > 0 and self.arm_column_gap(side, qi) < p.column_clear:
                return False
            if p.belt_clear > 0 and self.arm_belt_gap(side, qi) < p.belt_clear:
                return False
        return True

    def arm_column_gap(self, side, q7):
        """Smallest distance from the arm's links and gripper (all its
        collision geoms) to the robot's own column / head boxes for joint
        values q7 (private kinematics data)."""
        import mujoco
        k = self.kin[side]
        m = self.cell.model
        if not hasattr(self, "_col_geoms"):
            self._col_geoms = [g for g in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                                           for n in ("base_col_head", "base_col_column")) if g >= 0]
            self._arm_geoms_all = {}
            self._ft2 = np.zeros(6)
        if side not in self._arm_geoms_all:
            pre = side[0] + "_"
            sel = []
            for g in self.cell.robot_moving_geoms:
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[int(g)])) or ""
                if "Shoulder_Pitch" in bn:
                    continue        # mounted on the column (always touching its box)
                if bn.startswith(side) or bn.startswith(pre):
                    sel.append(int(g))
            self._arm_geoms_all[side] = sel
        k.fk(q7)
        best = 1.0
        for g in self._arm_geoms_all[side]:
            for c in self._col_geoms:
                best = min(best, mujoco.mj_geomDistance(m, k.d, g, c, 0.05, self._ft2))
        return best

    def arm_belt_gap(self, side, q7):
        """Smallest distance from the arm's links and gripper body (not the
        finger pads and links) to the belt surface for joint values q7
        (private kinematics data)."""
        import mujoco
        k = self.kin[side]
        m = self.cell.model
        if not hasattr(self, "_gap_geoms"):
            self._gap_geoms = {}
            self._belt_gid = self.cell._belt_surface
            self._ft = np.zeros(6)
        if side not in self._gap_geoms:
            pre = side[0] + "_"
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in self.cell.robot_moving_geoms]
            bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[int(g)])) or ""
                      for g in self.cell.robot_moving_geoms]
            sel = []
            for g, nm, bn in zip(self.cell.robot_moving_geoms, names, bodies):
                finger = any(t in bn for t in ("driver", "coupler", "spring_link", "follower", "_pad"))
                if (nm.startswith(side) or bn.startswith(side) or bn.startswith(pre)) and not finger:
                    sel.append(int(g))
            self._gap_geoms[side] = sel
        k.fk(q7)
        best = 1.0
        for g in self._gap_geoms[side]:
            dist = mujoco.mj_geomDistance(m, k.d, g, self._belt_gid, 0.05, self._ft)
            best = min(best, dist)
        return best

    def _guard_path(self, p0, R0s, piv_w0, x_st, H, L, z_st, yaw_err, qR):
        """Waypoints (per tick) from the stop to the landing guard: back out
        from under the leaning face along the belt, turn to the guard tilt,
        rise, then forward to 2 cm from where the parcel's upstream face
        will stand. None if the guard pose is out of reach."""
        from .kin import rot_err
        p = self.p
        tg = math.radians(p.guard_tilt_deg)
        Rg = palm_R("Right", 0.0, yaw_err, tg)
        y_face = piv_w0[1] - H                     # final upstream face
        z_g = self.belt + float(np.clip(p.guard_zfrac * L, 0.06, 0.16))
        g_pos = np.array([x_st, y_face - p.guard_gap - PAD_TIP * Rg[1, 2], z_g])
        qg = self.glob_R("Right", g_pos, Rg, qR)
        if qg is None:
            return None
        pts = []
        def seg(a, Ra, b, Rb, n):
            v = rot_err(Ra, Rb)
            ang = float(np.linalg.norm(v))
            for i in range(1, n + 1):
                u = i / n
                s_ = u * u * (3 - 2 * u)
                R_ = Ra if ang < 1e-9 else rot_axis_angle(v / ang, ang * s_) @ Ra
                pts.append((a + (b - a) * s_, R_))
        P1 = p0 + np.array([0.0, -0.03, 0.004])
        P2 = np.array([x_st, min(P1[1], y_face - p.guard_gap - 0.04), max(P1[2], z_st + 0.03)])
        P3 = np.array([x_st, P2[1], z_g])
        seg(p0, R0s, P1, R0s, 6)
        seg(P1, R0s, P2, Rg, 12)
        seg(P2, Rg, P3, Rg, 10)
        seg(P3, Rg, g_pos, Rg, 8)
        return pts

    def _stop_retreat_point(self, p0, i, H, z_st):
        """Retreat path of the stop: upstream along the belt at the stop
        height (out from under the leaning upstream face, out of the landing
        zone y_u - H .. y_u), then up."""
        # toward the robot only as far as x = retreat_x, and upstream to
        # |y| >= retreat_y_min: the gripper keeps clear of the robot's own
        # column / head (a box |x| < 0.095, |y| < 0.093 up to z 1.33)
        y3 = min(p0[1] - 0.06, -self.p.retreat_y_min) - (p0[1] - 0.06)
        segs = [(9, np.array([0.0, -0.05, 0.003])),          # out from under the face
                (11, np.array([0.0, -0.01, 0.10])),          # up, well above the lip
                (18, np.array([min(0.0, self.p.retreat_x - p0[0]), y3, 0.02])),  # toward the robot, upstream
                (12, np.array([0.0, 0.0, 0.10]))]            # up
        if self.p.retreat_clear or (self.p.retreat_clear_light and getattr(self, "_light_roll", False)):
            # a light or fast-turning parcel runs ahead of the plan and
            # its top-upstream edge sweeps upstream and down to y_u - H: the
            # stop first runs low out of the whole landing zone, then rises
            d1 = max(0.05, float(H) + self.p.retreat_clear_margin)
            n1 = max(9, int(math.ceil(d1 / self.p.retreat_clear_step)))
            y3 = min(p0[1] - d1 - 0.01, -self.p.retreat_y_min) - (p0[1] - d1 - 0.01)
            segs = [(n1, np.array([0.0, -d1, 0.003])),
                    (11, np.array([0.0, -0.01, 0.10])),
                    (18, np.array([min(0.0, self.p.retreat_x - p0[0]), y3, 0.02])),
                    (12, np.array([0.0, 0.0, 0.10]))]
        q = p0.copy()
        k = i
        for n, dlt in segs:
            if k <= n:
                u = k / n
                return q + dlt * (u * u * (3 - 2 * u))
            q = q + dlt
            k -= n
        return q

    def roll_up(self, slot, follow=True, final=True):
        p = self.p
        res = RollResult(t_start=self.cell.time)
        self.trace = []
        pos0, R0, (W, L, H), yaw_err, mass = self.dims(slot)
        self._R0 = R0
        com_b = np.asarray(self.cell.slots[slot].parcel.com_offset, float)
        mg = mass * G
        z_bot = pos0[2] - H / 2
        x_c = float(pos0[0])
        y_u, y_d = pos0[1] - L / 2, pos0[1] + L / 2
        piv_w0 = np.array([x_c, y_u, z_bot])
        e_b = R0.T @ (piv_w0 - pos0)
        cp = self.catch_pose
        pl_, Rl_ = self.cell.tcp_pose("Left")
        pad_w = pl_ - Rl_[:, 2] * PAD_TIP
        c_w0 = np.array([float(pl_[0]), y_d, float(pad_w[2])])
        c_b = R0.T @ (c_w0 - pos0)
        com_w0 = pos0 + R0 @ com_b
        d = com_w0 - piv_w0
        th0 = math.atan2(d[1], d[2])
        th_commit = th0 + math.radians(p.commit_margin_deg)
        res.log.append(dict(W=W, L=L, H=H, th0=math.degrees(th0)))
        self.log(f"roll_up: W {W:.3f} L {L:.3f} H {H:.3f} m {mass:.2f} th0 {math.degrees(th0):.1f}")
        f_catch = {"Left": np.array([0.0, -0.8 * mg, 0.0])}

        # ---- 1. the stop: right palm low at the upstream face
        ts = math.radians(p.stop_tilt_deg)
        psi_st = math.radians(p.stop_psi_deg)
        Rr = palm_R("Right", psi_st, yaw_err, ts)
        ax_r = Rr[:, 2]
        x_st = float(np.clip(x_c, 0.215, 0.34))
        z_st = self.belt + p.stop_z
        qh = None
        yaw_st = yaw_err
        # pre-pose: upstream of the face, a few cm up; alternatives if out of
        # reach (a retry may find the parcel yawed or shifted)
        for dy, dz, yw in ((0.05, 0.07, 1.0), (0.04, 0.09, 1.0), (0.06, 0.06, 1.0),
                           (0.05, 0.07, 0.0), (0.04, 0.10, 0.0)):
            Rr_try = palm_R("Right", psi_st, yaw_err * yw, ts)
            pre = np.array([x_st, y_u - dy, self.belt + dz])
            qh = self.glob_R("Right", pre, Rr_try, self.ex.q[SL["Right"]])
            if qh is not None:
                qg_ = self.ex.q.copy()
                qg_[SL["Right"]] = qh
                Qt_, _, _ = joint_move(self.ex.q, qg_, vmax=p.stop_vmax, tmin=p.stop_tmin)
                if self.arms_clash(Qt_):
                    qh = None           # the grippers would touch on the way
                    continue
                Rr = Rr_try
                ax_r = Rr[:, 2]
                yaw_st = yaw_err * yw
                break
        if qh is None:
            raise PlanFail("roll: stop pre-pose unreachable")
        q_goal = self.ex.q.copy()
        q_goal[SL["Right"]] = qh
        if getattr(self, "staged", None) is not None:
            # from the staging pose straight down behind the parcel
            self.move_cart("Right", pre, R1=Rr, T=p.stage_drop_s, forces=f_catch)
            self.staged = None
        else:
            self.joint_goal(q_goal, vmax=p.stop_vmax, tmin=p.stop_tmin, forces=f_catch)
        st_out = np.array([x_st, y_u - 0.03, z_st + 0.003])
        self.move_cart("Right", st_out, R1=Rr, T=p.stop_out_s, forces=f_catch)
        st_in = np.array([x_st, y_u + PAD_TIP * ax_r[1] + p.stop_overpress, z_st])
        Fs = p.stop_force + p.stop_force_mg * mg

        def f_stop_ramp(i, n=15):
            a = min(1.0, i / n)
            return {"Left": f_catch["Left"], "Right": np.array([0.0, Fs * a, 0.0])}
        self.move_cart("Right", st_in, R1=Rr, T=p.stop_in_s, forces=f_stop_ramp)
        self.trace_row(slot, "stop_set")

        # ---- 2. the pusher: the catching palm; if its pad roll differs from
        #         the push pad roll, it turns about the lower pad's contact
        Fn0 = p.push_fn + p.push_fn_mg * mg
        cap = math.radians(p.tool_cap_deg)
        n_in = np.array([0.0, -1.0, 0.0])
        psi = cp["psi"]
        nf = 15

        def f_up(i):
            a = min(1.0, i / nf)
            return {"Left": n_in * (0.8 * mg * (1 - a) + Fn0 * a),
                    "Right": np.array([0.0, Fs, 0.0])}
        if abs(psi - cp["psi"]) > 1e-3:
            z_p = z_bot + max(p.catch_zmin, min(H - 0.02, p.push_zfrac * H))
            c_w0 = np.array([x_c, y_d, z_p])
            c_b = R0.T @ (c_w0 - pos0)
            qL0 = self.ex.q[SL["Left"]].copy()
            q_push, psi, Rp, tcp_push = self.pusher_pose(c_w0, n_in, 0.0, psi, yaw_err, qL0, cap)
            if q_push is None:
                raise PlanFail("roll: push pose unreachable")
            self.move_cart("Left", tcp_push, R1=Rp, T=p.turn_s, forces=f_up)
        else:
            self.hold(nf * DT, forces=f_up)
        self.trace_row(slot, "push_ready")

        # ---- 3. the arc (closed loop on the tilt), then follow-through
        th_t = self.tilt_x(self.body(slot)[1], R0)
        t0 = self.cell.time
        qL = self.ex.q[SL["Left"]].copy()
        qR = self.ex.q[SL["Right"]].copy()
        lead = math.radians(p.lead_deg)
        rate = math.radians(p.rate_deg_s) * DT
        ret_max = p.stop_retreat_max_deg
        light = p.light_mass_kg > 0 and mass < p.light_mass_kg
        if light:
            # (option): a light parcel slides upstream under the push
            # once the stop has left; the stop stays to a larger tilt
            ret_max = p.stop_retreat_max_light_deg
        self._light_roll = light
        th_ret = min(th0 - math.radians(p.stop_retreat_before_deg), math.radians(ret_max))
        self._stop_left_at = None
        th_end = math.radians(p.follow_end_deg) if follow else th_commit
        stop_on = True
        ret_i = 0
        ret_extra = 0.0
        ret_from = None
        guard_path = None
        brake_P = None
        brake_th = None
        n_fail = 0
        th_hold_best = 0.0
        hist1 = []
        boost1 = 0.0
        while self.cell.time - t0 < p.timeout_s and not self.cell.done:
            if not self.cell.slots[slot].active:
                res.reason = "parcel gone"
                break
            pos, R, *_ = self.body(slot)
            th_m = self.tilt_x(R, R0)
            res.tilt_max_deg = max(res.tilt_max_deg, math.degrees(th_m))
            if th_m >= th_commit:
                res.committed = True
            if th_m >= th_end - math.radians(1.0) or (not follow and res.committed):
                break
            past = th_m > th0
            # approach the balance slowly: the rate falls to slow_frac of the
            # plan over the last slow_deg before the balance tilt
            slow = float(np.clip((th0 - th_m) / max(math.radians(p.slow_deg), 1e-6), p.slow_frac, 1.0)) \
                if p.slow_deg > 0 else 1.0
            r_lim = rate * slow if not past else math.radians(p.follow_rate_deg_s) * DT
            if not stop_on and ret_i < p.ret_freeze_ticks:
                r_lim = 0.0
            # stall boost: no tilt progress before the balance for
            # stall_s -> the lead grows (up to stall_lead_max_deg more)
            hist1.append(th_m)
            ks_ = max(1, int(round(p.arc1_stall_s / DT)))
            if p.arc1_stall_s > 0 and not past and len(hist1) > ks_ and \
                    hist1[-1] - hist1[-1 - ks_] < math.radians(1.0):
                boost1 = min(math.radians(p.arc1_stall_lead_max_deg), boost1 + math.radians(0.25))
            lead_b = lead + boost1
            th_t = max(th_t, min(th_t + r_lim, th_m + lead_b, th_end))
            # never behind the parcel (a lagging target would leave the face);
            # past the balance no lead: the pad only accompanies the fall
            dth = float(np.clip(th_t - th_m, 0.0, lead + boost1))
            if th_m > th0 + math.radians(p.free_after_deg):
                dth = 0.0
            piv = pos + R @ e_b
            Rr_ = R @ R0.T
            yaw_now = math.atan2(Rr_[1, 0], Rr_[0, 0])
            dx_y = float(np.clip(p.yaw_gain * yaw_now, -p.yaw_dx_max, p.yaw_dx_max))
            c_now = pos + R @ c_b + np.array([dx_y, 0.0, 0.0])
            n_out_now = R @ R0.T @ np.array([0.0, 1.0, 0.0])
            Rd = rot_x(dth)
            c_tgt = piv + Rd @ (c_now - piv)
            # the pad holds the parcel's place along the belt (the
            # belt drags the pivot downstream once the face tilts): beyond
            # hold_slide of downstream drift the target is shifted back
            # upstream by the excess (force capped by the squeeze governor)
            if p.hold_slide >= 0 and not (p.brake and th_m > th0 + math.radians(p.free_after_deg)):
                exc = float(piv[1] - piv_w0[1]) - p.hold_slide
                if exc > 0:
                    c_tgt = c_tgt - np.array([0.0, min(exc, p.hold_slide_max), 0.0])
            n_in_t = -(Rd @ n_out_now)
            braking = p.brake and th_m > th0 + math.radians(p.free_after_deg)
            if braking and p.brake == 2:
                # moving brake: the pad follows the face along the arc at a
                # lag (bounded) behind the parcel's tilt; projected onto the
                # face it slides back along it, the pad friction slows the
                # fall all the way down instead of for ~10 deg
                if brake_th is None:
                    brake_th = th_m
                brake_th = max(brake_th + math.radians(p.brake_rate_deg_s) * DT,
                               th_m - math.radians(p.brake_lag_deg))
                brake_th = min(brake_th, th_m)
                Rb = rot_x(brake_th - th_m)
                cb = piv + Rb @ (c_now - piv)
                c_tgt = cb - float((cb - c_now) @ n_out_now) * n_out_now
            elif braking:
                # past the balance: the pad holds its place along the face
                # (fixed in the world in the face's plane) and stays pressed
                # on it; the face slides under the pad and the pad friction
                # brakes the fall
                if brake_P is None:
                    brake_P = self.cell.tcp_pose("Left")[0] - \
                        self.cell.tcp_pose("Left")[1][:, 2] * (PAD_TIP + p.push_overpress)
                elif p.brake_move:
                    # the braking point rides with the belt: the parcel falls
                    # in the belt's frame and lands without slip (no kick)
                    brake_P = brake_P + np.array([0.0, self.rc.belt_speed() * DT, 0.0])
                d_n = float((brake_P - c_now) @ n_out_now)
                c_tgt = brake_P - d_n * n_out_now
                # the pad slides along the face toward its end: stop braking
                # near the edge (the face is H long, the contact started at c_b)
                if float(np.linalg.norm(c_tgt - c_now)) > p.brake_slide_max * H:
                    c_tgt = c_now.copy()
            # the pad does not follow the parcel's yaw: its x stays on the
            # initial centre line (a pad that followed the yawing face would
            # drag it further by friction: unstable), its axis stays in the
            # y-z plane
            c_tgt[0] = c_w0[0]
            n_in_t[0] = 0.0
            n_in_t /= max(np.linalg.norm(n_in_t), 1e-9)
            th_tool = math.atan2(-n_in_t[2], -n_in_t[1])
            if p.psi_goal_deg >= 0.0:
                # pad roll moves slowly from the catch's toward psi_goal
                # (both pads on the face; a slow turn twists little)
                g_ = math.radians(p.psi_goal_deg)
                psi = psi + float(np.clip(g_ - psi, -math.radians(p.psi_rate_deg), math.radians(p.psi_rate_deg)))
            qn, psi_n, Rp, tcp_t = self.pusher_pose(c_tgt, n_in_t, th_tool, psi, yaw_err, qL, cap)
            if qn is None:
                # out of reach: hold the last pose; the belt drags the pivot
                # under the held contact and may finish the roll by itself.
                # Give up only when the tilt has stopped increasing.
                n_fail += 1
                qn = qL
                if th_m > th_hold_best + math.radians(0.5):
                    th_hold_best = th_m
                    n_fail = 0
                if n_fail > p.hold_ticks:
                    res.reason = "push IK"
                    break
            else:
                psi = psi_n
                th_hold_best = th_m
            step = np.abs(qn - qL).max()
            if step > p.step_max:
                qn = qL + (qn - qL) * (p.step_max / step)
            qL = qn
            r = c_now - piv
            rr = math.hypot(r[1], r[2])
            tan = np.array([0.0, -r[2], r[1]]) / max(rr, 1e-6)
            dg = (pos + R @ com_b) - piv
            Mg = mg * dg[1]                      # restoring moment about +x
            # moment budget: the normal force's lever (contact height over the
            # pivot in the parcel frame) is h_n; the driving moment of normal
            # + tangential force is held near (1 + margin) of the gravity
            # moment and scaled down when the tilt rate exceeds the plan
            h_n = max(abs(float((c_now - piv) @ (R @ R0.T @ np.array([0.0, 0.0, 1.0])))), 0.02)
            ps_ = self.cell.truth.parcel_state(slot)
            w_x = float(ps_.angvel[0])
            w_des = math.radians(p.rate_deg_s) * (slow if not past else 1.0)
            kscale = float(np.clip(1.0 - p.rate_gain * (w_x - w_des), 0.0, 1.3))
            M_need = max(0.0, Mg) * (1.0 + p.moment_margin) * kscale
            Fn_need = M_need * p.fn_share / h_n
            Fn_fric = M_need * (1.0 - p.fn_share) / max(rr, 1e-3) / p.mu_eff
            fn_lo = p.fn_min + p.fn_min_mg * mg
            Fn = float(np.clip(max(Fn_need, Fn_fric) + fn_lo, fn_lo, Fn0))
            Ft = max(0.0, M_need - min(Fn, Fn_need) * h_n) / max(rr, 1e-3)
            if not stop_on:
                # no upstream shove once the stop has left
                Fn = min(Fn, p.push_fn_late + p.push_fn_late_mg * mg)
            if th_m > th0 + math.radians(p.free_after_deg):
                Ft = 0.0
                Fn = p.fall_fn if not p.brake else \
                    p.brake_fn + (p.brake_fn_mg + p.brake_fn_v * self.rc.belt_speed()) * mg
            F = n_in_t * Fn + tan * Ft
            # the stop: holds the initial pivot, yields to the rotation only
            if stop_on and th_m >= th_ret:
                stop_on = False
                ret_from, R_from = self.cell.tcp_pose("Right")
                guard_path = self._guard_path(ret_from, R_from, piv_w0, x_st, H, L, z_st,
                                              yaw_err, qR) if p.guard else None
            if stop_on:
                # the stop's pads stay on the bottom of the leaning upstream
                # face; with stop_turn the palm also turns with the face (its
                # fingers stay clear of the face), as far as reachable
                th_s = th_m * p.stop_turn
                Rr_t = palm_R("Right", psi_st, yaw_st, ts + th_s) if p.stop_turn else Rr
                s_tgt = piv_w0 + rot_x(th_m) @ (st_in - piv_w0)
                s_tgt[2] = max(s_tgt[2], z_st + p.stop_lift * min(1.0, th_m / math.radians(30)))
                qrn, okr, epr = self.ik_R("Right", s_tgt, Rr_t, qR, iters=60,
                                          rot_tol=0.004, pos_tol=1e-3)
                if okr and p.belt_clear > 0 and self.arm_belt_gap("Right", qrn) < 0.5 * p.belt_clear:
                    okr = False         # the ducked gripper body would touch the belt
                if p.stop_turn and not okr:
                    qrn, okr, epr = self.ik_R("Right", s_tgt, Rr, qR, iters=60,
                                              rot_tol=0.004, pos_tol=1e-3)
                FR = rot_x(th_m) @ np.array([0.0, Fs, 0.0]) if p.stop_turn else np.array([0.0, Fs, 0.0])
                FR = FR + np.array([0.0, 0.0, p.stop_fz + p.stop_fz_mg * mg])
            else:
                ret_i += 1
                if guard_path is not None:
                    # the stop becomes a landing guard upstream of the
                    # parcel's final upstream face (stops an over-rotation)
                    k_ = min(ret_i, len(guard_path)) - 1
                    s_tgt, Rg_ = guard_path[k_]
                    qrn, okr, epr = self.ik_R("Right", s_tgt, Rg_, qR, iters=40)
                else:
                    if p.retreat_adapt and 9 <= ret_i <= 21 and ret_extra < p.retreat_adapt_max and \
                            self.stop_parcel_gap(slot) < p.retreat_adapt_gap:
                        # the parcel follows the stop upstream (a
                        # light parcel is pushed back off its pivot): the
                        # stop does not rise into it but backs further
                        # upstream at its height until it is clear
                        ret_i -= 1
                        ret_extra += p.retreat_adapt_step
                        self.peak["retreat_adapt_ticks"] = self.peak.get("retreat_adapt_ticks", 0) + 1
                    s_tgt = self._stop_retreat_point(ret_from, ret_i, H, z_st) + np.array([0.0, -ret_extra, 0.0])
                    qrn, okr, epr = self.ik_R("Right", s_tgt, Rr, qR, iters=40)
                FR = None
            stp = np.abs(qrn - qR).max()
            if stp > p.step_max:
                qrn = qR + (qrn - qR) * (p.step_max / stp)
            qR = qrn
            row = self.ex.q.copy()
            row[SL["Left"]] = qL
            row[SL["Right"]] = qR
            self.tick(row, {"Left": F, "Right": FR})
            self.trace_row(slot, "arc" if not past else "follow", th_t)
        res.log.append(dict(t=self.cell.time, phase="arc_end", tilt=res.tilt_max_deg,
                            committed=res.committed))
        if not res.committed and self.cell.slots[slot].active and res.reason in ("", "push IK"):
            # ---- abort: lower the parcel back onto its base with the pad on
            # the face, then hold it caught (left palm) for a retry
            res.reason = "no commit"
            if stop_on:
                pr, Rr_ = self.cell.tcp_pose("Right")
                self.move_cart("Right", pr - ax_r * 0.04 + np.array([0.0, 0.0, 0.02]), R1=Rr_, T=0.3,
                               forces={"Left": n_in * (0.8 * mg)})
            for _ in range(int(0.8 / DT)):
                if not self.cell.slots[slot].active:
                    break
                pos, R, *_ = self.body(slot)
                c_now = pos + R @ c_b
                n_out_now = R @ R0.T @ np.array([0.0, 1.0, 0.0])
                th_tool = math.atan2(n_out_now[2], n_out_now[1])
                qn, psi_n, Rp, tcp_t = self.pusher_pose(c_now - n_out_now * 0.002, -n_out_now,
                                                        th_tool, psi, yaw_err, qL, cap)
                if qn is not None:
                    st_ = np.abs(qn - qL).max()
                    if st_ > p.step_max:
                        qn = qL + (qn - qL) * (p.step_max / st_)
                    qL = qn
                row = self.ex.q.copy()
                row[SL["Left"]] = qL
                self.tick(row, {"Left": -n_out_now * (0.8 * mg)})
            if self.cell.slots[slot].active:
                # back to the catch configuration itself (same IK branch as
                # the first attempt: the retry's arc stays reachable)
                cp = self.catch_pose
                q_goal = self.ex.q.copy()
                q_goal[SL["Left"]] = cp["q"]
                Qb_, _, _ = joint_move(self.ex.q, q_goal, vmax=1.2, tmin=0.6)
                if self.path_hits_station(Qb_, margin=0.012, every=2):
                    # the way back would bring the wrist to the rail:
                    # no retry from here (the scheduler releases and parks)
                    res.held = False
                    res.reason = "no commit (no safe way back to the catch pose)"
                else:
                    self.joint_goal(q_goal, vmax=1.2, tmin=0.6,
                                    forces={"Left": np.array([0.0, -0.8 * mg, 0.0])})
                    self.hold(0.3, forces={"Left": np.array([0.0, -0.8 * mg, 0.0])})
                    res.held = True
            res.t_end = self.cell.time
            res.label_after, res.label_err_deg = self.label_now(slot)
            res.peaks = dict(self.peak)
            return res
        # ---- 3b. the continued half turn: the parcel stands on its
        #          former upstream face with the pad on its new top (the
        #          former downstream face); the pad drives a second arc about
        #          the new upstream bottom edge (the former top-upstream edge)
        #          while the belt drags that edge downstream, until the
        #          parcel is past its second balance; then it falls onto its
        #          former top face (half a turn in all)
        if getattr(self, "_half_turn", False) and res.committed and self.cell.slots[slot].active \
                and not self.cell.done:
            if stop_on:
                pr, Rr_ = self.cell.tcp_pose("Right")
                self.move_cart("Right", pr - ax_r * 0.06 + np.array([0.0, 0.0, 0.03]), R1=Rr_, T=0.3)
                stop_on = False
            self._arc2(slot, res, pos0, R0, e_b, c_b, com_b, mg, H, L, x_c, y_u, z_bot, c_w0,
                       psi, yaw_err, cap)
        # ---- 4. let go
        pr, Rr_ = self.cell.tcp_pose("Right")
        if stop_on:
            self.move_cart("Right", pr - ax_r * 0.06 + np.array([0.0, 0.0, 0.03]), R1=Rr_, T=0.35)
        pl, Rl_ = self.cell.tcp_pose("Left")
        n_back = -Rl_[:, 2]
        dy_belt = self.rc.belt_speed() * 0.35 if (p.brake and p.brake_move) else 0.0
        th_end_now = self.tilt_x(self.body(slot)[1], R0) if self.cell.slots[slot].active else 0.0
        if p.stroke and res.committed and th_end_now > math.radians(p.stroke_min_deg):
            # the parcel is landing on its former upstream face: the pad,
            # pressed lightly on the new top, strokes it downstream (its
            # friction keeps the top from running on upstream: over-rotation)
            v_s = self.rc.belt_speed() + p.stroke_dv
            n_s = max(3, int(p.stroke_s / DT))
            qL = self.ex.q[SL["Left"]].copy()
            Rl_s = Rl_
            for k_ in range(1, n_s + 1):
                tgt = pl + np.array([0.0, v_s * DT * k_, -p.stroke_press])
                qn_, ok_, ep_ = self.ik_R("Left", tgt, Rl_s, qL, iters=40)
                st_ = np.abs(qn_ - qL).max()
                if st_ > p.step_max:
                    qn_ = qL + (qn_ - qL) * (p.step_max / st_)
                qL = qn_
                row = self.ex.q.copy()
                row[SL["Left"]] = qL
                self.tick(row, {"Left": np.array([0.0, 0.0, -(p.stroke_fn + p.stroke_fn_mg * mg)])})
                if not self.cell.slots[slot].active:
                    break
            pl, Rl_ = self.cell.tcp_pose("Left")
            dy_belt = self.rc.belt_speed() * 0.35
            self.move_cart("Left", pl + np.array([0, dy_belt, 0.06]), R1=Rl_, T=0.35)
        elif p.letgo_safe:
            # the let-go is planned first and taken only along a
            # way that keeps the arm clear of the robot's own column / head
            # (a long parcel on a slow belt is pushed near the column; the
            # plain let-go swung the left shoulder into the column head)
            # alternatives in order: the plain let-go; the same, also away from
            # the robot (+x); straight up; back along the tool axis; last
            # (it drags the face downstream) up and downstream
            self.move_cart_safe("Left", [pl + n_back * 0.03 + np.array([0, dy_belt, 0.05]),
                                         pl + n_back * 0.03 + np.array([0.03, dy_belt, 0.05]),
                                         pl + np.array([0.0, 0.0, 0.06]),
                                         pl + n_back * 0.045 + np.array([0.0, 0.0, 0.03]),
                                         pl + np.array([0.0, 0.04, 0.05])],
                                R1=Rl_, T=0.35)
        else:
            self.move_cart("Left", pl + n_back * 0.03 + np.array([0, dy_belt, 0.05]),
                           R1=Rl_, T=0.35)
        # ---- 5. settle (an intermediate roll of a chain: only until the
        #         parcel has landed, the re-catch follows at once)
        calm = []

        def settled():
            if not self.cell.slots[slot].active:
                return True
            ps = self.cell.truth.parcel_state(slot)
            self.trace_row(slot, "settle")
            w_ = float(np.linalg.norm(ps.angvel))
            if not final:
                calm.append(w_ < self.p.settle_mid_w and ps.on_belt)
                k_ = max(1, int(round(self.p.settle_mid_s / DT)))
                return len(calm) >= k_ and all(calm[-k_:])
            return w_ < 0.15 and ps.on_belt and \
                self.cell.time - res.t_start > 1.0
        self.hold(2.5, check=settled)
        res.t_end = self.cell.time
        if self.cell.slots[slot].active:
            from .geom import label_dir
            ps = self.cell.truth.parcel_state(slot)
            lab, err = label_dir(ps.label_normal)
            res.label_after, res.label_err_deg = lab, err
        res.peaks = dict(self.peak)
        return res

    def _arc2(self, slot, res, pos0, R0, e_b, c_b, com_b, mg, H, L, x_c, y_u, z_bot, c_w0,
              psi, yaw_err, cap):
        """Second quarter of a continued half turn (see roll_up, 3b)."""
        p = self.p
        sq0 = p.sq_soft
        p.sq_soft = p.arc2_sq_soft
        try:
            return self._arc2_run(slot, res, pos0, R0, e_b, c_b, com_b, mg, H, L, x_c, y_u, z_bot,
                                  c_w0, psi, yaw_err, cap)
        finally:
            p.sq_soft = sq0

    def _arc2_run(self, slot, res, pos0, R0, e_b, c_b, com_b, mg, H, L, x_c, y_u, z_bot, c_w0,
                  psi, yaw_err, cap):
        p = self.p
        self.phase = "arc2"
        e_b2 = R0.T @ (np.array([x_c, y_u, z_bot + H]) - pos0)     # former top-upstream edge
        lead = math.radians(p.arc2_lead_deg)
        rate = math.radians(p.arc2_rate_deg_s) * DT
        th_end2 = math.radians(p.arc2_end_deg)
        qL = self.ex.q[SL["Left"]].copy()
        t0 = self.cell.time
        pos, R, *_ = self.body(slot)
        th_t = self.tilt_x(R, R0)
        best = th_t
        n_fail = 0
        # the contact point of the second arc: where the pad tip is now,
        # projected onto the face (the brake let the face slide under it)
        pl_, Rl_ = self.cell.tcp_pose("Left")
        tip = pl_ - Rl_[:, 2] * PAD_TIP
        c_now0 = pos + R @ c_b
        n_out0 = R @ R0.T @ np.array([0.0, 1.0, 0.0])
        c_start = tip - float((tip - c_now0) @ n_out0) * n_out0
        c_b = R.T @ (c_start - pos)
        res.log.append(dict(t=t0, phase="arc2_start", tilt=math.degrees(th_t)))
        hist2 = []
        boost = 0.0
        th_past = None
        n_close = 0
        while self.cell.time - t0 < p.arc2_timeout_s and not self.cell.done:
            if not self.cell.slots[slot].active:
                break
            pos, R, *_ = self.body(slot)
            Rr_ = R @ R0.T
            th_m = math.atan2(Rr_[2, 1], Rr_[1, 1])
            if th_m < -math.pi / 2:
                th_m += 2 * math.pi
            res.tilt_max_deg = max(res.tilt_max_deg, math.degrees(th_m))
            if th_m >= th_end2:
                break
            piv2 = pos + R @ e_b2
            c_now = pos + R @ c_b
            n_out = R @ R0.T @ np.array([0.0, 1.0, 0.0])
            com_w = pos + R @ com_b
            dg2 = com_w - piv2
            if dg2[1] < -p.arc2_past_m:
                if th_past is None:
                    th_past = th_m
                    res.log.append(dict(t=self.cell.time, phase="arc2_past", tilt=math.degrees(th_m)))
                if p.arc2_follow_deg <= 0.0 or th_m >= th_past + math.radians(p.arc2_follow_deg):
                    break                   # past the second balance: falls on by itself
            # the pad leads the contact point along the face towards upstream
            # (its friction drags the top upstream while the belt drags the
            # base downstream: the parcel tips over its upstream bottom edge)
            # and stays pressed into the face
            n_in = -n_out.copy()
            n_in[0] = 0.0
            n_in /= max(np.linalg.norm(n_in), 1e-9)
            t_up = np.array([0.0, n_in[2], -n_in[1]])         # face tangent, upstream for a top face
            if t_up[1] > 0:
                t_up = -t_up
            # stall boost: no tilt progress for arc2_stall_s -> lead more
            hist2.append(th_m)
            k_ = max(1, int(round(p.arc2_stall_s / DT)))
            if len(hist2) > k_ and hist2[-1] - hist2[-1 - k_] < math.radians(1.0):
                boost = min(p.arc2_lead_max_m - p.arc2_lead_m, boost + 0.0005)
            lead_m = p.arc2_lead_m + boost
            c_tgt = c_now + t_up * lead_m + n_in * p.arc2_press
            c_tgt[0] = c_w0[0]
            th_tool = math.atan2(-n_in[2], -n_in[1])
            qn, psi_n, Rp, tcp_t = self.pusher_pose(c_tgt, n_in, th_tool, psi, yaw_err, qL, cap)
            if qn is None:
                n_fail += 1
                qn = qL
                if th_m > best + math.radians(0.5):
                    best = th_m
                    n_fail = 0
                if n_fail > p.hold_ticks:
                    res.log.append(dict(t=self.cell.time, phase="arc2_ik", tilt=math.degrees(th_m)))
                    break
            else:
                psi = psi_n
                best = max(best, th_m)
            step = np.abs(qn - qL).max()
            if step > p.step_max:
                qn = qL + (qn - qL) * (p.step_max / step)
            if p.arm_gap_min > 0:
                row_ = self.ex.q.copy()
                row_[SL["Left"]] = qn
                if self.grip_gap(row_) < p.arm_gap_min:
                    # the pusher would come too close to the other
                    # arm: hold (and end the arc if that lasts)
                    qn = qL
                    n_close += 1
                    self.peak["arc2_arm_holds"] = self.peak.get("arc2_arm_holds", 0) + 1
                    if n_close > p.hold_ticks // 2:
                        res.log.append(dict(t=self.cell.time, phase="arc2_arm_gap", tilt=math.degrees(th_m)))
                        break
            qL = qn
            r = c_now - piv2
            h_t = max(abs(float(r[2])), 0.03)
            M_need = max(0.0, mg * float(dg2[1])) * (1.0 + p.moment_margin)
            Fn = p.arc2_fn + p.arc2_fn_mg * mg
            Ft = min(M_need / h_t, p.mu_eff * Fn)
            if th_past is not None:
                # following the fall: light contact, no push
                Fn = p.arc2_follow_fn + p.arc2_follow_fn_mg * mg
                Ft = 0.0
            F = n_in * Fn + t_up * Ft
            row = self.ex.q.copy()
            row[SL["Left"]] = qL
            self.tick(row, {"Left": F})
            self.trace_row(slot, "arc2", th_t)
        res.log.append(dict(t=self.cell.time, phase="arc2_end", tilt=res.tilt_max_deg))

    # ------------------------------------------------------------------
    # label, release, park
    # ------------------------------------------------------------------
    def label_now(self, slot):
        from .geom import label_dir
        if not self.cell.slots[slot].active:
            return "gone", 180.0
        ps = self.cell.truth.parcel_state(slot)
        return label_dir(ps.label_normal)

    def at_rest(self, slot):
        if not self.cell.slots[slot].active:
            return False
        ps = self.cell.truth.parcel_state(slot)
        return ps.on_belt and float(np.linalg.norm(ps.angvel)) < 0.2

    def back_off(self, T=0.4, slot=None):
        """Both palms away from the parcel without touching it: straight up
        until above the parcel's top (+ margin), then toward the robot to
        behind the belt's near edge."""
        top = self.belt + 0.10
        if slot is not None and self.cell.slots[slot].active:
            pos, R, (W, L, H), _, _ = self.dims(slot)
            top = pos[2] + H / 2
        for s in SIDES:
            p_, R_ = self.cell.tcp_pose(s)
            z_up = max(p_[2], top + 0.09, self.belt + 0.16)
            tgt = np.array([p_[0], p_[1], z_up])
            if z_up - p_[2] < 1e-3:
                continue
            # a straight Cartesian lift can be out of reach from a ducked
            # stop posture (the local IK then drifts toward the robot and
            # the lip): solve the lifted pose globally and take the joint
            # move when it keeps clear of the station, else the Cartesian one
            q_up = self.glob_R(s, tgt, R_, self.ex.q[SL[s]])
            done = False
            if q_up is not None:
                q_goal = self.ex.q.copy()
                q_goal[SL[s]] = q_up
                Qb, _, _ = joint_move(self.ex.q, q_goal, vmax=2.0, tmin=max(T, (z_up - p_[2]) / 0.25))
                # a lifted pose of another IK branch makes the joint
                # path swing the gripper sideways (it swept a parcel that a
                # roll had left standing off the belt): take the joint path
                # only if the tool stays near the straight lift and the path
                # keeps clear of every parcel (riding with the belt)
                touching = any(c.robot_geom is not None and self._side_of_geom(c.robot_geom) == s
                               for c in self.cell.truth.contacts(kinds=("robot-parcel",)))
                if (not self.path_hits_station(Qb) and self._joint_path_straight(s, Qb, p_, tgt, R_)
                        and (touching or not self.rc.path_collides(Qb, margin=0.01))):
                    for row in Qb:
                        self.tick(row)
                    done = True
            if not done:
                # a lift that keeps the tool's full orientation and
                # stops where the IK cannot follow (the axis-only lift drifted
                # towards the robot and into the near lip)
                self.safe_lift(s, tgt, R_, T=max(T, (z_up - p_[2]) / 0.25))

    def _side_of_geom(self, gname):
        n = str(gname)
        if n.startswith(("Left_", "L_")):
            return "Left"
        if n.startswith(("Right_", "R_")):
            return "Right"
        import mujoco
        try:
            gid = int(n[4:]) if n.startswith("geom") else mujoco.mj_name2id(self.cell.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            bn = self.cell.model.body(int(self.cell.model.geom_bodyid[gid])).name
        except Exception:
            return None
        return "Left" if bn.startswith(("Left_", "L_")) else ("Right" if bn.startswith(("Right_", "R_")) else None)

    def _joint_path_straight(self, side, Q, p0, p1, R0, dev=0.03, ang_max=0.35, every=2):
        """Does the joint path Q keep this side's tool within dev of the
        straight segment p0 -> p1 and within ang_max of the orientation R0?"""
        from .kin import rot_err
        k = self.kin[side]
        p0 = np.asarray(p0, float)
        d = np.asarray(p1, float) - p0
        dd = float(d @ d)
        for i in list(range(0, len(Q), every)) + [len(Q) - 1]:
            p, R = k.fk(Q[i][SL[side]])
            u = 0.0 if dd < 1e-12 else min(1.0, max(0.0, float((p - p0) @ d) / dd))
            if np.linalg.norm(p - (p0 + u * d)) > dev:
                return False
            if float(np.linalg.norm(rot_err(R0, R))) > ang_max:
                return False
        return True

    def separate(self, back=0.06, up=0.04):
        """Both palms back off along their own tool axes and up (after a
        protective stop the grippers may be touching each other)."""
        for s in ("Right", "Left"):
            p_, R_ = self.cell.tcp_pose(s)
            tgt = p_ - R_[:, 2] * back + np.array([0.0, 0.0, up])
            self.safe_lift(s, tgt, R_, T=0.4)

    def _lift_above_parcels(self, clear=0.09, reach_y=0.35):
        """Each palm straight up to clear the top of every parcel within
        reach_y of it along the belt (privileged poses), if it is lower."""
        from .geom import aligned_dims
        tops = []
        for ps in self.cell.truth.parcel_states():
            d = aligned_dims(ps.size, ps.quat)
            tops.append((float(ps.pos[1]), float(ps.pos[2] + d[2] / 2), float(d[1])))
        if not tops:
            return
        for s_ in SIDES:
            p_, R_ = self.cell.tcp_pose(s_)
            near = [t for y, t, L in tops if abs(y - p_[1]) < reach_y + L / 2]
            if not near:
                continue
            z_need = max(near) + clear
            if p_[2] < z_need - 0.005:
                self.safe_lift(s_, np.array([p_[0], p_[1], z_need]), R_, T=max(0.3, (z_need - p_[2]) / 0.3))

    def safe_lift(self, side, tgt, R, T=0.4):
        """Straight lift of one TCP to tgt with the orientation R held; every
        tick's IK must converge within 4 mm and keep 2 cm from the station
        (rails, lip, conveyor body), else the lift stops there."""
        k = self.kin[side]
        q = self.ex.q[SL[side]].copy()
        p0, _ = k.fk(q)
        n = max(2, int(math.ceil(T / DT)))
        for i in range(1, n + 1):
            u = i / n
            sm = u * u * u * (10 - 15 * u + 6 * u * u)
            p_ = p0 + (np.asarray(tgt) - p0) * sm
            qn, ok, ep = self.ik_R(side, p_, R, q, iters=60, pos_tol=4e-3)
            if not ok or ep > 0.004:
                break
            row = self.ex.q.copy()
            row[SL[side]] = qn
            if self.path_hits_station(row[None, :], margin=0.02, every=1):
                break
            step = np.abs(qn - q).max()
            if step > self.p.step_max:
                qn = q + (qn - q) * (self.p.step_max / step)
                row[SL[side]] = qn
            q = qn
            self.tick(row)
            if self.cell.done:
                return False
        return True

    def park(self, slot=None, wait_max=12.0):
        """Joint move to the park posture (the cell's start posture), delayed
        while it would sweep the parcel riding the belt."""
        rc = self.rc
        # a parcel that a roll left standing can be taller than the
        # park height; first lift each palm straight up above every parcel
        # near it (its top + clearance) before any sideways move
        self._lift_above_parcels()
        # if the joint move to the park would sweep a gripper through the
        # lip, the rails or the conveyor body, go first in Cartesian space to
        # the park's tool pose (same IK branch as now), then join the park
        # posture from there (a short move behind and above the belt)
        Q0, _, _ = joint_move(self.ex.q, rc.q_ready, vmax=self.p.v_park, tmin=0.6)
        near_parcel = False
        if self.p.park_cartesian_near:
            for ps in self.cell.truth.parcel_states():
                for s_ in SIDES:
                    if abs(float(ps.pos[1]) - float(self.cell.tcp_pose(s_)[0][1])) < 0.45:
                        near_parcel = True
        # (park_cartesian_near = 1, measured and rejected: the Cartesian route
        # across to the park pose crossed the path of a tall parcel riding
        # downstream; the local IK sagged and the pad tipped it)
        if self.path_hits_station(Q0) or near_parcel:
            for s_ in SIDES:
                pk, Rk = self.kin[s_].fk(rc.q_ready[SL[s_]])
                p_, R_ = self.cell.tcp_pose(s_)
                up = np.array([p_[0], p_[1], max(p_[2], pk[2])])
                if up[2] > p_[2] + 1e-3:
                    self.move_cart(s_, up, axis1=R_[:, 2], T=max(0.3, (up[2] - p_[2]) / 0.3))
                self.move_cart(s_, pk, R1=Rk, T=0.8)
        t_end = self.cell.time + wait_max
        while True:
            Q, _, _ = joint_move(self.ex.q, rc.q_ready, vmax=self.p.v_park, tmin=0.6)
            if not rc.path_collides(Q, margin=0.015) or self.cell.time > t_end or self.cell.done:
                break
            self.hold(0.2)
        for row in Q:
            self.tick(row)
            if self.cell.done:
                break
        return True

    def arms_clash(self, Q, margin=0.012, every=2):
        """Would the joint path Q (rows of 14) bring a left-arm collision geom
        within margin of a right-arm one (private data)?"""
        import mujoco
        c = self.cell
        m = c.model
        if not hasattr(self, "_cd_arm"):
            self._cd_arm = mujoco.MjData(m)
            side = {}
            for g in c.robot_geoms:
                bn = m.body(int(m.geom_bodyid[g])).name
                side[int(g)] = 1 if bn.startswith(("Left_", "L_")) else (2 if bn.startswith(("Right_", "R_")) else 0)
            self._arm_side = side
        d = self._cd_arm
        d.qpos[:] = c.data.qpos
        d.qvel[:] = 0
        for k in list(range(0, len(Q), every)) + [len(Q) - 1]:
            d.qpos[self.ex.dyn.qadr] = Q[k]
            mujoco.mj_fwdPosition(m, d)
            for i in range(d.ncon):
                con = d.contact[i]
                s1 = self._arm_side.get(int(con.geom1), 0)
                s2 = self._arm_side.get(int(con.geom2), 0)
                if s1 and s2 and s1 != s2 and con.dist < margin:
                    return True
        return False

    def stop_parcel_gap(self, slot):
        """True distance between the right (stop) arm's wrist / gripper geoms
        and the parcel's collision geoms in the present state."""
        import mujoco
        c = self.cell
        m = c.model
        if not hasattr(self, "_spg"):
            R_ = []
            for g in c.robot_moving_geoms:
                bn = m.body(int(m.geom_bodyid[int(g)])).name
                if bn.startswith("R_") or bn.startswith("Right_Wrist"):
                    R_.append(int(g))
            self._spg = (R_, np.zeros(6))
        R_, ft = self._spg
        pg = np.nonzero(np.asarray(c.parcel_geom_slot) == slot)[0]
        best = 0.05
        for g1 in R_:
            for g2 in pg:
                best = min(best, mujoco.mj_geomDistance(m, c.data, g1, int(g2), 0.05, ft))
        return best

    def grip_gap(self, q14):
        """Smallest true distance between the two arms' wrist / gripper
        collision geoms for the posture q14 (private data; mj_geomDistance,
        so the cell's zero contact margin does not hide near misses)."""
        import mujoco
        c = self.cell
        m = c.model
        if not hasattr(self, "_gg"):
            self._gg_d = mujoco.MjData(m)
            L_, R_ = [], []
            for g in c.robot_moving_geoms:
                bn = m.body(int(m.geom_bodyid[int(g)])).name
                if bn.startswith("L_") or bn.startswith("Left_Wrist"):
                    L_.append(int(g))
                elif bn.startswith("R_") or bn.startswith("Right_Wrist"):
                    R_.append(int(g))
            self._gg = (L_, R_)
            self._gg_ft = np.zeros(6)
        d = self._gg_d
        d.qpos[:] = c.data.qpos
        d.qpos[self.ex.dyn.qadr] = q14
        mujoco.mj_kinematics(m, d)
        best = 0.05
        for g1 in self._gg[0]:
            for g2 in self._gg[1]:
                best = min(best, mujoco.mj_geomDistance(m, d, g1, g2, 0.05, self._gg_ft))
        return best

    def path_hits_station(self, Q, margin=0.01, every=3):
        """Would the joint path Q (rows of 14) bring a robot collision geom
        within margin of the belt hardware (lip, rails, conveyor body)?"""
        import mujoco
        c = self.cell
        m = c.model
        if not hasattr(self, "_cd_st"):
            self._cd_st = mujoco.MjData(m)
            self._hw = set(int(g) for g in c.belt_geoms
                           if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) != "belt_surface")
        d = self._cd_st
        d.qpos[:] = c.data.qpos
        d.qvel[:] = 0
        mg_ = m.opt
        for k in range(0, len(Q), every):
            d.qpos[self.ex.dyn.qadr] = Q[k]
            mujoco.mj_fwdPosition(m, d)
            for i in range(d.ncon):
                con = d.contact[i]
                g1, g2 = int(con.geom1), int(con.geom2)
                r1, r2 = c._geom_is_robot[g1], c._geom_is_robot[g2]
                if r1 == r2:
                    continue
                og = g2 if r1 else g1
                if og in self._hw and con.dist < margin:
                    return True
        return False

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def run(self, slot, exit_mode="release"):
        """Bring a parcel whose label is on its upstream or downstream face
        to a state the chain needs.

        Entry: the parcel approaching on the belt (justified to the near
        side, yaw a multiple of 90 deg), upstream of the work window, arms at
        the park posture. Exit ("release"): the parcel resting on the belt,
        released (no robot contact), riding downstream; arms parked. The
        label is up for a downstream-face label (one roll), down for an
        upstream-face label (one roll; the 180-degree squeeze flip completes
        it). Returns a dict (ok, label after, timing, rolls, reasons)."""
        p = self.p
        out = dict(ok=False, reason="", exit="", rolls=0, t0=self.cell.time, attempts=[])
        lab0, _ = self.label_now(slot)
        out["label0"] = lab0
        if lab0 not in ("+y", "-y"):
            out["reason"] = f"not an end label ({lab0})"
            return out
        want = "up" if lab0 == "+y" else "down"
        out["want"] = want
        self.phase = "catch"
        try:
            self.catch(slot)
        except PlanFail as e:
            out["reason"] = f"catch: {e}"
            self.park(slot)
            return out
        out["t_catch"] = self.cell.time
        _, Rc_, _, yaw_c, _ = self.dims(slot)
        out["yaw_catch_deg"] = round(math.degrees(yaw_c), 1)
        base = dict(push_fn=p.push_fn, lead_deg=p.lead_deg, push_below_top=p.push_below_top)
        for k in range(p.max_rolls):
            if not self.cell.slots[slot].active:
                break
            t_save = p.timeout_s
            if k == 0 and p.first_timeout_s > 0:
                p.timeout_s = p.first_timeout_s      # test hook: interrupt the first attempt
            try:
                if p.mode == 2:
                    r = self.roll_up_b(slot)
                else:
                    r = self.roll_up(slot, follow=bool(p.follow))
            except PlanFail as e:
                p.timeout_s = t_save
                out["reason"] = f"roll plan: {e}"
                break
            p.timeout_s = t_save
            out["rolls"] += 1
            lab, err = self.label_now(slot)
            out["attempts"].append(dict(committed=r.committed, tilt=round(r.tilt_max_deg, 1),
                                        label=lab, err=round(err, 1), held=r.held,
                                        t=round(self.cell.time, 2)))
            if lab in ("up", "down", "+x", "-x", "gone"):
                break
            if r.held:
                # retry from the caught state: more push force, more lead
                p.push_fn = base["push_fn"] * (1.5 + 0.5 * k)
                p.lead_deg = base["lead_deg"] + 4.0 * (k + 1)
                continue
            # not turned: the parcel is on the belt with the label still on
            # an end face; re-catch and roll again
            if not self.cell.slots[slot].active:
                break
            try:
                self.recatch(slot)
            except PlanFail as e:
                out["reason"] = f"recatch: {e}"
                break
        p.push_fn, p.lead_deg = base["push_fn"], base["lead_deg"]
        out["t_roll_end"] = self.cell.time
        lab_r, err_r = self.label_now(slot)
        out["label_roll"], out["err_roll"] = lab_r, round(err_r, 1)
        self.phase = "exit"
        self.back_off(slot=slot)
        self.park(slot)
        out["t_end"] = self.cell.time
        uid = None
        if self.cell.slots[slot].active:
            lab, err = self.label_now(slot)
            out["at_rest"] = self.at_rest(slot)
        else:
            # the parcel rode on past the exit while the arms parked: its
            # state is the one recorded at the scan line
            cr = [c for c in self.cell.truth.crossings("scan")]
            if cr:
                c = cr[-1]
                lab = "up" if c["label_angle_deg"] < 25 else (
                    "down" if c["label_angle_deg"] > 155 else "side")
                err = c["label_angle_deg"] if lab == "up" else 180.0 - c["label_angle_deg"]
                out["at_rest"] = bool(c["on_belt"])
            else:
                lab, err = "gone", 180.0
                out["at_rest"] = False
        out["label_after"], out["label_err"] = lab, round(err, 1)
        out["exit"] = "released"
        out["ok"] = (lab == want and err < 25.0 and out["at_rest"])
        if not out["ok"] and not out["reason"]:
            out["reason"] = f"label {lab} ({err:.0f} deg)"
        return out

    def recatch(self, slot):
        """The parcel rides the belt after a roll that did not turn it: the
        left palm stops it again (fresh soft catch further downstream)."""
        pos, R, (W, L, H), yaw_err, mass = self.dims(slot)
        y_face = pos[1] + L / 2
        v = max(self.rc.belt_speed(), 0.02)
        # as far downstream as the catch allows (time for the palm to get
        # there), but never upstream of where the face will be in 1.2 s
        y_max0 = float(np.clip(self.p.catch_y_face, -0.15 + L, 0.155))
        far = float(self.p.recatch_y_max) > 0.155 + 1e-9
        y_max = max(y_max0, float(self.p.recatch_y_max)) if far else y_max0
        y_stop = max(min(y_max, y_face + v * 1.2 + 0.03), min(y_max, y_face + 0.03))
        if y_stop < y_face + 0.02:
            raise PlanFail(f"recatch: parcel already past the catch line (face y {y_face:.3f}, "
                           f"t {self.cell.time:.2f})")
        if not far:
            # the default (horizontal palm only)
            self.catch(slot, y_face=y_max if y_face + v * 1.0 < y_max else y_stop)
            return
        y_c = min(y_max, y_stop) if y_face + v * 1.0 >= min(y_max, 0.155) else min(y_max, 0.155)
        # beyond the horizontal palm's reach (0.155 m) the catch palm tilts down
        # (30 deg reaches a stop at 0.30 m)
        tilt = 0.0 if y_c <= 0.155 else float(np.interp(y_c, [0.155, 0.30], [10.0, 30.0]))
        self.catch(slot, y_face=y_c, tilt_deg=tilt)


# ======================================================================
# roll U, mode B: lift to the hand-over tilt with the stop, then the belt
# finishes the roll against the held pusher (the chosen primitive)
# ======================================================================
def _roll_up_b(self, slot):
    """Roll U, lift-and-hold variant.

    Phase 1 (0 -> th_hand): the right palm is a low stop at the upstream
    bottom edge, the left palm (pads across the belt, as in the catch)
    pushes its contact point on the downstream face along the arc about
    the pivot (target leads the measured tilt, J^T F feedforward: a normal
    force into the face, a tangential force from the gravity moment). The
    stop takes the excess push, so the pads' normal force is not limited
    by the belt drag and the friction lift is reliable.
    Phase 2 (th_hand -> commit): the stop backs off; the left palm HOLDS
    its contact fixed in the world. The belt drags the pivot edge
    downstream under the held contact: belt drag (at the base) and pad
    reaction (at height h) form a couple that turns the parcel on to its
    balance tilt and over. From about 20 deg of tilt this couple exceeds
    the gravity moment for every parcel of the population.
    Phase 3: the left palm backs off; the parcel falls onto its former
    upstream face and rides downstream."""
    p = self.p
    res = RollResult(t_start=self.cell.time)
    self.trace = []
    pos0, R0, (W, L, H), yaw_err, mass = self.dims(slot)
    self._R0 = R0
    com_b = np.asarray(self.cell.slots[slot].parcel.com_offset, float)
    mg = mass * G
    z_bot = pos0[2] - H / 2
    x_c = float(pos0[0])
    y_u, y_d = pos0[1] - L / 2, pos0[1] + L / 2
    piv_w0 = np.array([x_c, y_u, z_bot])
    e_b = R0.T @ (piv_w0 - pos0)
    pl_, Rl_ = self.cell.tcp_pose("Left")
    # contact point: midway between the two pads' tips (pads across the belt)
    c_w0 = np.array([x_c, y_d, float((pl_ - Rl_[:, 2] * PAD_TIP)[2])])
    c_b = R0.T @ (c_w0 - pos0)
    com_w0 = pos0 + R0 @ com_b
    d = com_w0 - piv_w0
    th0 = math.atan2(d[1], d[2])
    th_commit = th0 + math.radians(p.commit_margin_deg)
    th_hand = min(math.radians(p.hand_deg), th0 - math.radians(8.0))
    res.log.append(dict(W=W, L=L, H=H, th0=math.degrees(th0)))
    f_catch = {"Left": np.array([0.0, -0.8 * mg, 0.0])}

    # ---- 1. the stop
    ts = math.radians(p.stop_tilt_deg)
    Rr = palm_R("Right", 0.0, yaw_err, ts)
    ax_r = Rr[:, 2]
    x_st = float(np.clip(x_c, 0.215, 0.34))
    z_st = self.belt + p.stop_z
    pre = np.array([x_st, y_u - 0.05, self.belt + 0.07])
    qh = self.glob_R("Right", pre, Rr, self.ex.q[SL["Right"]])
    if qh is None:
        raise PlanFail("roll: stop pre-pose unreachable")
    q_goal = self.ex.q.copy()
    q_goal[SL["Right"]] = qh
    self.joint_goal(q_goal, vmax=2.2, tmin=0.7, forces=f_catch)
    st_out = np.array([x_st, y_u - 0.03, z_st + 0.003])
    self.move_cart("Right", st_out, R1=Rr, T=0.4, forces=f_catch)
    st_in = np.array([x_st, y_u + PAD_TIP * ax_r[1] + p.stop_overpress, z_st])
    Fs = p.stop_force + p.stop_force_mg * mg

    def f_stop_ramp(i, n=15):
        a = min(1.0, i / n)
        return {"Left": f_catch["Left"], "Right": np.array([0.0, Fs * a, 0.0])}
    self.move_cart("Right", st_in, R1=Rr, T=0.35, forces=f_stop_ramp)
    self.trace_row(slot, "stop_set")

    # ---- 2. phase 1: the left palm turns flush (pads across the belt) and
    #         slides up to the push height, then lifts along the arc
    Fn0 = p.push_fn + p.push_fn_mg * mg
    nf = 12
    n_in0 = np.array([0.0, -1.0, 0.0])
    cp = self.catch_pose
    tilt_c = math.radians(p.push_tilt0_deg)
    z_p = z_bot + min(H - 0.015, max(p.push_zmin, H - max(0.02, p.push_below_frac * H)))
    Rp0 = palm_R("Left", cp["psi"], yaw_err, tilt_c)
    c_w0 = np.array([x_c, y_d, z_p])
    c_b = R0.T @ (c_w0 - pos0)
    tcp0 = c_w0 + Rp0[:, 2] * (PAD_TIP + p.push_overpress)
    self.move_cart("Left", tcp0, R1=Rp0, T=0.45,
                   forces=lambda i: {"Left": n_in0 * (0.8 * mg + (Fn0 - 0.8 * mg) * min(1.0, i / nf)),
                                     "Right": np.array([0.0, Fs, 0.0])})
    self.trace_row(slot, "push_ready")
    th_t = self.tilt_x(self.body(slot)[1], R0)
    qL = self.ex.q[SL["Left"]].copy()
    qR = self.ex.q[SL["Right"]].copy()
    lead = math.radians(p.lead_deg)
    rate = math.radians(p.rate_deg_s) * DT
    t0 = self.cell.time
    phase = 1
    hold_tcp = None
    hold_R = None
    ret_i = 0
    ret_from = None
    stall_t = None
    th_best = 0.0
    while self.cell.time - t0 < p.timeout_s and not self.cell.done:
        if not self.cell.slots[slot].active:
            res.reason = "parcel gone"
            break
        pos, R, *_ = self.body(slot)
        th_m = self.tilt_x(R, R0)
        res.tilt_max_deg = max(res.tilt_max_deg, math.degrees(th_m))
        if th_m >= th_commit:
            res.committed = True
            break
        piv = pos + R @ e_b
        c_now = pos + R @ c_b
        if phase == 1 and th_m >= th_hand:
            phase = 2
            ret_from = self.cell.tcp_pose("Right")[0]
            hold_tcp, hold_R = self.cell.tcp_pose("Left")
            stall_t = self.cell.time
            th_best = th_m
        if phase == 1 or p.p2_follow:
            th_t = max(th_t, min(th_t + rate, th_m + lead))
            dth = float(np.clip(th_t - th_m, 0.0, lead))
            Rd = rot_x(dth)
            c_tgt = piv + Rd @ (c_now - piv)
            if phase == 2:
                # after the hand-over: lift only (the contact rises along the
                # arc, its y stays on the face): no upstream shove, the belt
                # drags the pivot under the contact
                c_tgt[1] = c_now[1]
            c_tgt[0] = c_w0[0]
            n_out_now = R @ R0.T @ np.array([0.0, 1.0, 0.0])
            n_in_t = -(Rd @ n_out_now)
            n_in_t[0] = 0.0
            n_in_t /= max(np.linalg.norm(n_in_t), 1e-9)
            # tool: catch orientation tilted further with the face (pads stay
            # across the belt); TCP so that the pad tips are on the face
            face_tilt = math.atan2(-n_in_t[2], -n_in_t[1])
            tt = max(tilt_c, min(face_tilt, math.radians(p.push_tilt_max_deg)))
            Rp = palm_R("Left", cp["psi"], yaw_err, tt)
            tcp_t = c_tgt + Rp[:, 2] * (PAD_TIP + p.push_overpress)
            qn, ok, ep = self.ik_R("Left", tcp_t, Rp, qL, iters=60)
            r = c_now - piv
            rr = math.hypot(r[1], r[2])
            tan = np.array([0.0, -r[2], r[1]]) / max(rr, 1e-6)
            dg = (pos + R @ com_b) - piv
            Ft = p.push_ft_gain * mg * dg[1] / max(rr, 1e-3)
            if th_m > th0 + math.radians(p.free_after_deg):
                Ft = 0.0
            if phase == 1:
                FL = n_in_t * Fn0 + tan * Ft
            else:
                # lift by friction: vertical force from the gravity moment
                # about the pivot over the lever of the contact, small normal
                lever = max(abs(c_now[1] - piv[1]), 0.03)
                F_up = p.push_ft_gain * max(0.0, mg * dg[1]) / lever
                FL = n_in_t * (p.hold_fn + p.push_fn_late_mg * mg) + np.array([0.0, 0.0, F_up])
            if phase == 1:
                # stop: holds the initial pivot, yields to the rotation only
                s_tgt = piv_w0 + rot_x(th_m) @ (st_in - piv_w0)
                s_tgt[2] = max(s_tgt[2], z_st)
                qrn, okr, epr = self.ik_R("Right", s_tgt, Rr, qR, iters=60, rot_tol=0.004, pos_tol=1e-3)
                FR = np.array([0.0, Fs, 0.0])
            else:
                ret_i += 1
                s_tgt = ret_from + self._stop_backoff(ret_i)
                qrn, okr, epr = self.ik_R("Right", s_tgt, Rr, qR, iters=40)
                FR = None
        else:
            # phase 2: the left palm holds its contact; the stop backs off
            if th_m > th_best + math.radians(0.5):
                th_best = th_m
                stall_t = self.cell.time
            tcp_t = hold_tcp.copy()
            if self.cell.time - stall_t > p.stall_s:
                # the belt couple stalled: nudge the contact along the arc
                r = c_now - piv
                tan = np.array([0.0, -r[2], r[1]])
                hold_tcp = hold_tcp + tan / max(np.linalg.norm(tan), 1e-6) * 0.004
                tcp_t = hold_tcp.copy()
            qn, ok, ep = self.ik_R("Left", tcp_t, hold_R, qL, iters=40)
            FL = hold_R[:, 2] * p.hold_fn
            ret_i += 1
            s_tgt = ret_from + self._stop_backoff(ret_i)
            qrn, okr, epr = self.ik_R("Right", s_tgt, Rr, qR, iters=40)
            FR = None
        step = np.abs(qn - qL).max()
        if step > p.step_max:
            qn = qL + (qn - qL) * (p.step_max / step)
        qL = qn
        stp = np.abs(qrn - qR).max()
        if stp > p.step_max:
            qrn = qR + (qrn - qR) * (p.step_max / stp)
        qR = qrn
        row = self.ex.q.copy()
        row[SL["Left"]] = qL
        row[SL["Right"]] = qR
        self.tick(row, {"Left": FL, "Right": FR})
        self.trace_row(slot, f"p{phase}", th_t)
    res.log.append(dict(t=self.cell.time, phase="arc_end", tilt=res.tilt_max_deg,
                        committed=res.committed))
    # ---- 3. let go: both palms back off, the parcel falls over
    pr, Rr_ = self.cell.tcp_pose("Right")
    if phase == 1:
        self.move_cart("Right", pr + self._stop_backoff(40), R1=Rr_, T=0.4)
    pl, Rl2 = self.cell.tcp_pose("Left")
    self.move_cart("Left", pl - Rl2[:, 2] * 0.015 + np.array([0, 0, 0.06]), R1=Rl2, T=0.35)

    def settled():
        if not self.cell.slots[slot].active:
            return True
        ps = self.cell.truth.parcel_state(slot)
        self.trace_row(slot, "settle")
        return float(np.linalg.norm(ps.angvel)) < 0.15 and ps.on_belt and \
            self.cell.time - res.t_start > 1.0
    self.hold(2.5, check=settled)
    res.t_end = self.cell.time
    if self.cell.slots[slot].active:
        res.label_after, res.label_err_deg = self.label_now(slot)
    res.peaks = dict(self.peak)
    return res


def _stop_backoff(self, i):
    """Offset of the stop's back-off path after i ticks: back upstream along
    the belt (out from under the leaning face), then up."""
    n1, n2 = 10, 14
    if i <= n1:
        u = i / n1
        return np.array([0.0, -0.035, 0.004]) * (u * u * (3 - 2 * u))
    u = min(1.0, (i - n1) / n2)
    return np.array([0.0, -0.035, 0.004]) + np.array([0.0, -0.02, 0.10]) * (u * u * (3 - 2 * u))


EndOver.roll_up_b = _roll_up_b
EndOver._stop_backoff = _stop_backoff
