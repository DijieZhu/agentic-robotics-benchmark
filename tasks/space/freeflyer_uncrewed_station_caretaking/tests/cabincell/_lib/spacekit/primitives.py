"""Firmware-level primitives on top of the v0.1 controller / executive (``control.Firmware`` inside
``robot.FreeFlyerRig``): undock and dock, fly-to with the zone validation, stop on localisation loss and resume,
perch / unperch on a handrail, the arm as pan / tilt, gripper open / close (no grasp feedback), grasping an item
feature, a carried-item report, battery charging on the dock, and the per-nozzle factory test.

Structure: ``FirmwareLayer`` wires the arm servo, the tendon gripper, the optional grasp rule, the dock magnets, the
station model and the zone map around a rig, registers three periodic tasks on the rig's deterministic scheduler
(arm / gripper every physics step, station fans and suction networks at 100 Hz, actions at the firmware rate right
after the firmware tick) and runs one ``Action`` at a time: ``run(action, timeout)`` steps the simulation until the
action returns a result.  Every action logs (t_start, t_end, name, result, details) in ``timeline``.
The sequences mirror the reference firmware's behaviours (states of the dock / perch behaviours; the final dock and
perch moves are holonomic and unvalidated, as the firmware sets them) [R firmware behaviors/dock, behaviors/perch].
Parameters that no source gives are marked [G] at their definition.  Names here are mechanics-level, not an
agent-facing API (each task chooses that).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .control import plan_trapezoid
from .gripper import GRASP_PROFILES, ArmServo, TendonGripper, grasp_profile, q_to_tilt, tilt_to_q
from .rotations import mat_to_quat, quat_angle, quat_conj, quat_mul, quat_to_mat

__all__ = ["DockMagnets", "FirmwareLayer", "Action", "Wait", "SetMode", "FlyTo", "FlyToFiducial", "Undock", "Dock", "ArmTo", "GripperCmd",
           "GraspFeature", "Perch", "Unperch", "Hold", "Call", "pose_for_palm", "carried_mass_properties",
           "NozzleTestMode", "nozzle_factory_test", "DOCK_DEFAULTS"]

DOCK_DEFAULTS = dict(
    approach_m=0.5,              # approach point off the berth  [R firmware geometry / dock config]
    initial_tolerance_m=1.0,     # must be this close to the approach point to begin  [R dock config]
    cone_half_angle_deg=6.0,     # approach cone about the berth axis at the berthing start (v1 dock sweep)  [V]
    att_tolerance_deg=20.0,      # attitude error allowed at the berthing start (v1 dock sweep)  [V]
    capture_m=0.04, release_m=0.05, docked_after_s=5.0,   # magnet capture / release / docked delay  [R simulator]
    hold_force_N=5.0,            # contact holding force (>= 4.5 N must not release, <= 22 N must release)  [R/G]
    magnet_g0_m=0.004,           # force falloff F = F_hold / (1 + (g / g0)^2)^2  [G]
    retract_s=8.0,               # magnets retracted while undocking  [R 'a few seconds']
    charge_w=72.0,               # 24 V x 3 A  [R dock power]
)


# ---------------------------------------------------------------------------------------------- dock
class DockMagnets:
    """Wrench source: the berth magnets pulling the robot onto the berth along the dock axis.

    berth_pose: (pos, quat) of the robot body when berthed (dock frame x = egress direction); the gap g is the
    robot origin's distance outward along x from that pose.  Force on the robot -F(g) x_dock at its CoM, F(g) =
    F_hold / (1 + (g / g0)^2)^2 for g < capture (and F_hold for g <= 0); lateral offset must be < ``lat_max``.
    ``retract(t, s)`` disables the magnets for s seconds.  Attributes: gap, lateral, force (last)."""

    def __init__(self, robot_body: int, berth_pose, hold_force: float = DOCK_DEFAULTS["hold_force_N"],
                 g0: float = DOCK_DEFAULTS["magnet_g0_m"], capture: float = DOCK_DEFAULTS["capture_m"], lat_max: float = 0.03):
        self.b = int(robot_body)
        self.p = np.asarray(berth_pose[0], float)
        self.R = quat_to_mat(np.asarray(berth_pose[1], float))
        self.x = self.R[:, 0]
        self.F0, self.g0, self.cap, self.lat_max = float(hold_force), float(g0), float(capture), float(lat_max)
        self.enabled = True
        self._retract_until = -1.0
        self.gap, self.lateral, self.force = np.inf, np.inf, 0.0

    def retract(self, t: float, seconds: float) -> None:
        """Retract the magnets until t + seconds."""
        self._retract_until = float(t) + float(seconds)

    def geometry(self, data):
        d = data.xpos[self.b] - self.p
        g = float(d @ self.x)
        lat = float(np.linalg.norm(d - g * self.x))
        return g, lat

    def accumulate(self, model, data, W) -> None:
        self.gap, self.lateral = self.geometry(data)
        self.force = 0.0
        if not self.enabled or data.time < self._retract_until:
            return
        g, lat = self.gap, self.lateral
        if g >= self.cap or lat > self.lat_max:
            return
        F = self.F0 if g <= 0 else self.F0 / (1.0 + (g / self.g0) ** 2) ** 2
        self.force = F
        W[self.b, 0:3] += -F * self.x


# ---------------------------------------------------------------------------------------------- helpers
def pose_for_palm(T_body_palm, palm_pos, palm_R):
    """Robot body pose (pos, quat) that puts the palm frame at (palm_pos, palm_R), the arm at the joint angles at
    which T_body_palm = (p, R) was measured."""
    pb, Rb = T_body_palm
    R = palm_R @ Rb.T
    p = palm_pos - R @ pb
    return p, mat_to_quat(R)


def carried_mass_properties(model, data, robot_body: int, extra_bodies=()):
    """(mass, CoM world, inertia about the CoM (world axes)) of the robot subtree plus extra bodies (a held item)."""
    root = model.body_rootid[robot_body]
    bodies = list(np.nonzero(model.body_rootid == root)[0]) + [int(b) for b in extra_bodies]
    M = float(sum(model.body_mass[b] for b in bodies))
    c = sum(model.body_mass[b] * data.xipos[b] for b in bodies) / M
    J = np.zeros((3, 3))
    for b in bodies:
        R = data.ximat[b].reshape(3, 3)
        r = data.xipos[b] - c
        J += R @ np.diag(model.body_inertia[b]) @ R.T + model.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
    return M, c, J


# ---------------------------------------------------------------------------------------------- actions
class Action:
    """One firmware-level action: start(fw) once, tick(fw, t) at the firmware rate -> None (running) or result."""

    name = "action"

    def start(self, fw) -> None:
        pass

    def tick(self, fw, t: float):
        return "done"

    def info(self) -> dict:
        return {}


class Wait(Action):
    """Wait a fixed time."""

    name = "wait"

    def __init__(self, seconds: float):
        self.s = float(seconds)

    def start(self, fw):
        self.t_end = fw.t + self.s

    def tick(self, fw, t):
        return "done" if t >= self.t_end - 1e-9 else None


class Call(Action):
    """Run a function once (e.g. a station request); result 'done'."""

    name = "call"

    def __init__(self, fn, name: str = "call"):
        self.fn, self.name = fn, name

    def start(self, fw):
        self.fn(fw)

    def tick(self, fw, t):
        return "done"


class SetMode(Action):
    """Change flight mode and wait until the fans reach the mode's speed (result 'ready' or 'speed_timeout')."""

    name = "set_mode"

    def __init__(self, mode: str):
        self.mode = mode

    def start(self, fw):
        fw.rig.set_mode(self.mode)
        self.t0 = fw.t

    def tick(self, fw, t):
        f = fw.rig.firmware
        if f.state != "preparing":
            return "speed_timeout" if f.result == "speed_timeout" else "ready"
        return None


class Hold(Action):
    """Stop and hold for a time."""

    name = "hold"

    def __init__(self, seconds: float):
        self.s = float(seconds)

    def start(self, fw):
        fw.rig.hold()
        self.t_end = fw.t + self.s

    def tick(self, fw, t):
        return "done" if t >= self.t_end else None


class FlyTo(Action):
    """Fly to (pos, quat) with the firmware planner and executive.

    mode: flight mode (None = current); faceforward: firmware face-forward planning; validate: check the planned
    segment against the zone map first (refuses with 'refused_keep_in' / 'refused_keep_out'); desired: lower planner
    limits; resume: on 'localization_lost' wait until the estimate is good again, then re-plan to the same goal
    (at most ``max_resumes`` times).  timeout_margin_s (v0.4; None = v0.3): the move returns 'timeout'
    (the executive is told to stop) when it has not reported 'stopped' this long after the end of its planned segment
    (counted from the latest plan; not while the fans are still changing speed).  Result: the executive's result
    ('success', 'tolerance_*', ...), a refusal, or 'timeout'.  arm_exempt (v0.4): validate with the stowed inflation
    even when the arm is deployed and the zone map has an arm-deployed inflation (``FirmwareLayer.check_segment``).
    """

    name = "fly_to"

    def __init__(self, pos, quat, mode: str | None = None, faceforward: bool = True, validate: bool = True,
                 desired: dict | None = None, resume: bool = True, max_resumes: int = 3, label: str = "",
                 timeout_margin_s: float | None = 30.0, arm_exempt: bool = False):
        self.pos, self.quat = np.asarray(pos, float), np.asarray(quat, float)
        self.arm_exempt = bool(arm_exempt)
        self.mode, self.ff, self.validate, self.desired = mode, faceforward, validate, desired
        self.resume, self.max_resumes = resume, max_resumes
        self.label = label
        self.resumes = 0
        self.violation = None
        self.phase = None
        self.timeout_margin = timeout_margin_s
        self.timed_out = None

    def _plan_and_go(self, fw):
        rig = fw.rig
        est = rig.est if rig.est is not None else rig.estimate()
        if self.validate and fw.zones is not None:
            seg = plan_trapezoid([(est.pos, est.quat), (self.pos, self.quat)], rig.firmware.mode, self.ff, t0=fw.t,
                                 desired=self.desired)
            res, k = fw.check_segment(seg, self.arm_exempt) if hasattr(fw, "check_segment") else fw.zones.check_segment(seg)
            if res != "ok":
                self.violation = dict(result=res, sample=k)
                return "refused_" + res
        rig.move_to(self.pos, self.quat, faceforward=self.ff, desired=self.desired)
        self.phase = "moving"
        return None

    def _timed_out(self, fw, t) -> bool:
        """no 'stopped' within timeout_margin of the planned end (segment times are live: the executive shifts
        them when a fan-speed change delays the start)."""
        f = fw.rig.firmware
        if self.timeout_margin is None or f.state == "preparing" or not f.segment:
            return False
        if t > f.segment[-1].t + self.timeout_margin:
            self.timed_out = dict(t=float(t), plan_end=float(f.segment[-1].t), state=f.state, executive_result=f.result)
            fw.rig.stop()
            return True
        return False

    def start(self, fw):
        self.t0 = fw.t
        if self.mode is not None and self.mode != fw.rig.firmware.mode.name:
            fw.rig.set_mode(self.mode)
        self.phase = "start"
        self._refused = None

    def tick(self, fw, t):
        f = fw.rig.firmware
        if self.phase == "start":
            if f.state == "preparing":
                return None
            r = self._plan_and_go(fw)
            if r is not None:
                fw.rig.hold()
                return r
            return None
        if self.phase == "moving":
            if f.result == "localization_lost" and self.resume and self.resumes < self.max_resumes:
                self.phase = "lost"
                return None
            if f.state == "stopped":
                return f.result
            return "timeout" if self._timed_out(fw, t) else None
        if self.phase == "lost":
            est = fw.rig.est
            if est is not None and est.confidence == 0:
                self.resumes += 1
                r = self._plan_and_go(fw)
                if r is not None:
                    fw.rig.hold()
                    return r
                return None
            return "timeout" if self._timed_out(fw, t) else None
        return None

    def info(self):
        return dict(goal=self.pos.tolist(), resumes=self.resumes, violation=self.violation, label=self.label,
                    timed_out=self.timed_out)


class FlyToFiducial(Action):
    """Fly to a pose next to a fiducial plate (a bag mouth's, the dock's target) on the fiducial-relative estimate.

    fiducial: dict(name, points (N, 3) world marker points, normal, cam_offset_body, cam_axis_body) as taken by
    ``dock.DockRelativeLocalizer.reference``; goal (pos, quat) in the world, computed by the caller from the
    fiducial's catalogued pose -- what the fiducial improves is the estimate the move is flown on (about 1 cm at
    0.6 m instead of the map's 5 cm).  settle_s: time on the fiducial before planning; keep: keep the reference
    after arriving (for a following insertion move); validate / mode / desired / faceforward as ``FlyTo``.  Without a
    fiducial-capable localiser (``reference`` method) it is a plain FlyTo.  Result: the executive's result;
    ``info()`` has the estimate source and the true position error at the end (for logs / tests)."""

    name = "fly_to_fiducial"

    def __init__(self, fiducial: dict, pos, quat, settle_s: float = 1.0, keep: bool = False, validate: bool = False,
                 mode: str | None = None, desired: dict | None = None, faceforward: bool = False, label: str = ""):
        self.fid, self.pos, self.quat = fiducial, np.asarray(pos, float), np.asarray(quat, float)
        self.settle, self.keep, self.label = float(settle_s), keep, label
        self.sub = FlyTo(self.pos, self.quat, mode=mode, faceforward=faceforward, validate=validate, desired=desired,
                         label=label)
        self.detail = {}

    def start(self, fw):
        loc = getattr(fw.rig, "localizer", None)
        self.loc = loc if (loc is not None and hasattr(loc, "reference")) else None
        if self.loc is not None:
            self.loc.reference(**self.fid)
        fw.rig.hold()
        self.t0 = fw.t
        self.state = "settling"

    def tick(self, fw, t):
        if self.state == "settling":
            if t - self.t0 < self.settle:
                return None
            self.sub.start(fw)
            self.state = "moving"
            return None
        r = self.sub.tick(fw, t)
        if r is None:
            return None
        b = fw.rig.adapter.body
        self.detail.update(source=getattr(self.loc, "last_source", "map") if self.loc is not None else "map",
                           true_pos_error_m=float(np.linalg.norm(fw.data.xpos[b] - self.pos)))
        if self.loc is not None and not self.keep:
            self.loc.release()
        return r

    def info(self):
        return dict(self.detail, label=self.label, fiducial=self.fid.get("name"))


class ArmTo(Action):
    """Move the arm to a firmware tilt / pan [deg] through the rate-limited servo; result 'done' or 'timeout', or
    'refused_holding' at once (the joints hold) while the layer's holding interlock is engaged (v0.4)."""

    name = "arm_to"

    def __init__(self, tilt_deg: float, pan_deg: float = 0.0, tol: float = 0.03, settle_s: float = 0.5):
        self.tilt, self.pan, self.tol, self.settle = tilt_deg, pan_deg, tol, settle_s
        self.refused = None

    def start(self, fw):
        cmd = getattr(fw, "arm_command", None)
        if cmd is not None:
            self.refused = cmd(tilt=tilt_to_q(self.tilt), pan=math.radians(self.pan))
        else:
            fw.arm.set(tilt=tilt_to_q(self.tilt), pan=math.radians(self.pan))
        self.t_ok = None

    def tick(self, fw, t):
        if self.refused is not None:
            return self.refused
        if fw.arm.done(fw.data, self.tol):
            if self.t_ok is None:
                self.t_ok = t
            if t - self.t_ok >= self.settle:
                return "done"
        return None

    def info(self):
        return dict(tilt_deg=self.tilt, pan_deg=self.pan)


class GripperCmd(Action):
    """Open or release-to-close (False) the tendon; returns 'done' after the stroke -- no grasp feedback (the
    finger encoders and the tendon load are readable through FirmwareLayer.gripper_telemetry).  amount: the opening
    commanded when opening (1 = fully open; a partial opening frees a handle without swinging the fingertips out --
    the v1 fingertips move 1 cm toward the item when fully open)."""

    name = "gripper"

    def __init__(self, open_: bool, extra_s: float = 0.3, amount: float = 1.0):
        self.open, self.extra, self.amount = bool(open_), float(extra_s), float(amount)

    def start(self, fw):
        if self.open and fw.grasp is not None and fw.grasp.held is not None:
            fw.grasp.release("open")
        fw.gripper.set(self.amount if self.open else 0.0)
        self.t_end = None

    def tick(self, fw, t):
        if not fw.gripper.moving:
            if self.t_end is None:
                self.t_end = t + self.extra
            if t >= self.t_end:
                return "done"
        return None

    def info(self):
        return dict(open=self.open, amount=self.amount if self.open else 0.0)


class Undock(Action):
    """Undocking (firmware states: switching to map localisation, waiting for spin-up, moving to the approach
    pose): fans to the mode's speed, magnets retracted, holonomic unvalidated move 0.5 m out along the dock axis.
    ``info()`` records how the robot left the berth: magnet pull when the move starts, the largest pull and gap
    after the retraction command, the time until the robot is 5 mm out."""

    name = "undock"

    def __init__(self, mode: str = "standard"):
        self.mode = mode
        self.detail = {}

    def start(self, fw):
        self.state = "switching_to_map_loc"
        self.t_state = fw.t
        fw.docked = False
        loc = getattr(fw.rig, "localizer", None)
        if loc is not None and hasattr(loc, "release"):
            loc.release()

    def tick(self, fw, t):
        f = fw.rig.firmware
        if self.state == "switching_to_map_loc":
            if t - self.t_state >= 0.5:
                fw.rig.set_mode(self.mode)
                self.state, self.t_state = "waiting_for_spin_up", t
            return None
        if self.state == "waiting_for_spin_up":
            if f.state != "preparing":
                if f.result == "speed_timeout":
                    return "speed_timeout"
                fw.magnets.retract(t, DOCK_DEFAULTS["retract_s"])
                p, q = fw.berth_pose
                goal = np.asarray(p) + quat_to_mat(q)[:, 0] * DOCK_DEFAULTS["approach_m"]
                fw.rig.move_to(goal, q, faceforward=False)
                self.detail.update(t_retract=t, magnet_N_at_retract=float(fw.magnets.force), max_magnet_N_after=0.0)
                self.state, self.t_state = "moving_to_approach_pose", t
            return None
        if self.state == "moving_to_approach_pose":
            F = float(fw.magnets.force)
            if t - self.t_state > 0.2:
                self.detail["max_magnet_N_after"] = max(self.detail["max_magnet_N_after"], F)
            if "t_out_5mm" not in self.detail and fw.magnets.gap > 0.005:
                self.detail["t_out_5mm"] = t - self.t_state
                self.detail["max_magnet_N_beyond_5mm"] = F
            if "t_out_5mm" in self.detail:
                self.detail["max_magnet_N_beyond_5mm"] = max(self.detail["max_magnet_N_beyond_5mm"], F)
            if f.state == "stopped":
                return "undocked" if f.result in ("success", "already_there") else f.result
        return None

    def info(self):
        return dict(self.detail)


class Dock(Action):
    """Docking, in the firmware's states: approach pose (nominal mode), dock-relative localisation, berthing move to
    the complete pose in the precision mode, the dock sees the robot (pins / contact), check attached (the move back
    to the approach pose must FAIL), spin down [R firmware behaviors/dock].  The firmware itself only requires the
    start within ``initial_tolerance`` of the approach point; this kit also enforces an approach cone at the start of
    the berthing move (measured: ``cone_half_angle_deg`` about the berth axis and ``att_tolerance_deg``, both from
    the dock-relative estimate): outside it the robot goes back to the approach pose first (``max_reapproach``).

    berth_speed: speed of the berthing move (the precision mode's 0.02 m/s limit; faster speeds fly it in the
    nominal mode with that desired speed); overshoot: the complete pose lies this far into the dock past the contact
    pose (the firmware's docked pose is 13.4 mm inside the contact pose).  go_to_approach=False starts the berthing
    move from where the robot is (sweeps); spin_down=False ends with 'attached'.  With the reference dock v1 the
    berth model is ``dock.BerthDock`` (``fw.magnets``); with ``DockMagnets`` the v0.2 capture rule is used.
    Results: 'docked' (or 'attached'), 'refused_outside_cone' (start beyond the initial tolerance or, with
    go_to_approach=False, outside the cone), 'eps_dock_failed' (the berthing move ended without contact),
    'attach_failed' (the check move succeeded: not held), or a failed move's result."""

    name = "dock"

    def __init__(self, berth_speed: float = 0.02, overshoot: float = 0.0134, mode: str = "standard",
                 go_to_approach: bool = True, spin_down: bool = True, enforce_cone: bool = True,
                 max_reapproach: int = 2, berth_mode: str = "fine", check_timeout_s: float = 30.0):
        self.v, self.over, self.mode = berth_speed, overshoot, mode
        self.go_to_approach, self.spin_down, self.enforce_cone = go_to_approach, spin_down, enforce_cone
        self.max_reapproach, self.berth_mode, self.check_timeout = int(max_reapproach), berth_mode, float(check_timeout_s)
        self.detail = {}

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _cone(self, fw, est):
        p, q = fw.berth_pose
        R = quat_to_mat(q)
        rel = est.pos - np.asarray(p)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(rel @ R[:, 0]) / max(np.linalg.norm(rel), 1e-9)))))
        att = math.degrees(quat_angle(est.quat, q))
        return ang, att, float(np.linalg.norm(est.pos - self.app))

    def _in_contact(self, fw, strict=True):
        mg = fw.magnets
        if hasattr(mg, "eps_contact"):
            return bool(mg.eps_contact) if strict else bool(mg.eps_contact or mg.force > 0.5)
        return mg.force > 0 and mg.gap < (0.005 if strict else 0.01)

    def _berth_move(self, fw):
        p, q = fw.berth_pose
        goal = np.asarray(p) - quat_to_mat(q)[:, 0] * self.over
        fine = fw.rig.firmware.modes.get(self.berth_mode)
        if fine is not None and self.v <= fine.hard_limit_vel + 1e-12:
            fw.rig.set_mode(self.berth_mode)
            fw.rig.move_to(goal, q, faceforward=False, desired=dict(vel=self.v))
        else:
            fw.rig.set_mode(self.mode)
            fw.rig.move_to(goal, q, faceforward=False, desired=dict(vel=self.v))

    def _localise_on_dock(self, fw):
        loc = getattr(fw.rig, "localizer", None)
        ref = getattr(fw, "dock_fiducial", None)
        if loc is not None and hasattr(loc, "reference") and ref is not None:
            loc.reference(**ref)
            self.detail["dock_relative_localisation"] = True

    # -- action ----------------------------------------------------------------------------------------------------
    def start(self, fw):
        p, q = fw.berth_pose
        R = quat_to_mat(q)
        self.app = np.asarray(p) + R[:, 0] * DOCK_DEFAULTS["approach_m"]
        est = fw.rig.est if fw.rig.est is not None else fw.rig.estimate()
        ang, att, dist_app = self._cone(fw, est)
        self.detail = dict(dist_to_approach=dist_app, cone_angle_deg=ang, att_err_deg=att, reapproaches=0)
        ok = dist_app <= DOCK_DEFAULTS["initial_tolerance_m"]
        if not self.go_to_approach and self.enforce_cone:
            ok = ok and ang <= DOCK_DEFAULTS["cone_half_angle_deg"] and att <= DOCK_DEFAULTS["att_tolerance_deg"]
        self.t_state = fw.t
        if not ok:
            self.state = "refused"
            return
        if self.go_to_approach:
            fw.rig.set_mode(self.mode)
            fw.rig.move_to(self.app, q, faceforward=False)
            self.state = "moving_to_approach_pose"
        else:
            self.state = "switching_to_dock_loc"
            self._localise_on_dock(fw)

    def tick(self, fw, t):
        if self.state == "refused":
            return "refused_outside_cone"
        f = fw.rig.firmware
        p, q = fw.berth_pose
        if self.state == "moving_to_approach_pose":
            if f.state == "stopped":
                if f.result not in ("success", "already_there"):
                    return f.result
                self.state, self.t_state = "switching_to_dock_loc", t
                self._localise_on_dock(fw)
            return None
        if self.state == "switching_to_dock_loc":
            if t - self.t_state < 1.0:
                return None
            est = fw.rig.est
            ang, att, _ = self._cone(fw, est)
            self.detail.setdefault("cone_at_berthing", []).append(dict(angle_deg=ang, att_deg=att))
            if (self.enforce_cone and (ang > DOCK_DEFAULTS["cone_half_angle_deg"] or att > DOCK_DEFAULTS["att_tolerance_deg"])
                    and self.go_to_approach and self.detail["reapproaches"] < self.max_reapproach):
                self.detail["reapproaches"] += 1          # outside the cone: back to the approach pose (dock-relative)
                fw.rig.set_mode(self.mode)
                fw.rig.move_to(self.app, q, faceforward=False)
                self.state = "moving_to_approach_pose"
                return None
            self._berth_move(fw)
            self.state, self.t_state = "moving_to_complete_pose", t
            return None
        if self.state == "moving_to_complete_pose":
            if self._in_contact(fw):
                self.detail["t_contact"] = t
                # check attached: back toward the approach pose in the precision mode; held = this move fails
                fw.rig.set_mode(self.berth_mode if self.berth_mode in f.modes else self.mode)
                fw.rig.move_to(self.app, q, faceforward=False)
                self.state, self.t_state = "checking_attached", t
                return None
            if f.state == "stopped" and t - self.t_state > 2.0:
                self.detail["result_move"] = f.result
                return "eps_dock_failed"
            return None
        if self.state == "checking_attached":
            if f.state != "stopped" and t - self.t_state < self.check_timeout:
                return None
            self.detail["check_result"] = f.result
            self.detail["check_gap_m"] = float(fw.magnets.gap)
            if f.result in ("success", "already_there") or not self._in_contact(fw, strict=False):
                return "attach_failed"
            fw.rig.idle()
            if not self.spin_down:
                fw.docked = True
                self.detail.update(gap_m=float(fw.magnets.gap), lateral_m=float(fw.magnets.lateral))
                return "attached"
            fw.rig.set_mode("off")
            self.state, self.t_state = "waiting_for_spin_down", t
            return None
        if self.state == "waiting_for_spin_down":
            if max(abs(w) for w in fw.rig.plant.impeller_speed) < 1.0 or t - self.t_state > 30.0:
                fw.docked = True
                self.detail.update(gap_m=float(fw.magnets.gap), lateral_m=float(fw.magnets.lateral))
                loc = getattr(fw.rig, "localizer", None)
                if loc is not None and hasattr(loc, "release"):
                    loc.release()
                return "docked"
        return None

    def info(self):
        return dict(self.detail)


class GraspFeature(Action):
    """Take an item by a handle (script-level primitive for the mission; the firmware has no such behaviour).

    feature(fw) -> (centre (3,), bar axis (3,), face normal toward the robot (3,), velocity (3,)) of the handle in
    world (a privileged script may use the true item state); the arm must be deployed.  Sequence: validated fly to a
    pre-grasp pose ``standoff`` m out along the face normal (skipped with standoff=None), open the gripper, straight
    holonomic unvalidated approach at ``approach_speed`` to the pose that puts the bar at palm depth ``depth``
    (re-targeted on the predicted feature every ``retarget_s``), release the tendon when the bar is inside the capture
    box (bar at palm x >= close_x, |y| <= 0.025), wait for the stroke, then optionally attach the grasp rule.
    close_on_the_move: keep approaching during the 1 s stroke (a free item is not pushed by the fingertips; use with
    close_x near the fingertip line, e.g. -0.115), then hold.  keep_attitude: during the approach track only the
    position for the robot's current attitude (a slowly rotating free item; the pre-grasp sets the attitude).
    lead_s: the pre-grasp aims at the feature predicted lead_s ahead (``"auto"``: the planned duration of the
    pre-grasp move).  intercept: every approach re-target aims at the feature predicted at the end of the planned
    approach move (the firmware plans rest-to-rest moves, so a drifting item is otherwise chased and never reached).
    align_depth: first hold the bar at this palm depth (outside the fingertips) until it is within align_tol of the
    finger plane laterally (or align_timeout_s passed), then go in; align_attitude: track the item's attitude while
    aligning (a rotating item), keep it while going in; box_z_tol: capture-box tolerance along the bar (a short
    handle needs the fingers between its end posts).
    Result 'closed' (no grasp feedback; ``info()`` has the true capture check for logs) or 'missed'.  Either result
    engages the layer's holding interlock when it is enabled and ``engage_interlock`` (v0.4; the perch
    behaviour does not engage it).
    """

    name = "grasp_feature"

    def __init__(self, feature, standoff: float | None = 0.25, approach_speed: float = 0.03, depth: float | None = None,
                 close_x: float | None = None, retarget_s: float = 1.0, timeout_s: float = 180.0, use_rule: bool = False,
                 rule_body=None, lead_s: "float | str" = 0.0, label: str = "", close_on_the_move: bool = False,
                 keep_attitude: bool = False, intercept: bool = False, align_depth: float | None = None,
                 align_tol: float = 0.012, align_timeout_s: float = 30.0, box_z_tol: float = 0.06,
                 align_attitude: bool = False, engage_interlock: bool = True):
        self.feature, self.standoff, self.v, self.depth = feature, standoff, approach_speed, depth
        self.engage_interlock = bool(engage_interlock)
        self.close_x, self.retarget, self.timeout, self.use_rule = close_x, retarget_s, timeout_s, use_rule
        self.rule_body, self.lead, self.label = rule_body, lead_s, label
        self.on_the_move = close_on_the_move
        self.keep_attitude = keep_attitude
        self.intercept = intercept
        self.align_depth, self.align_tol, self.align_timeout, self.box_z_tol = align_depth, align_tol, align_timeout_s, box_z_tol
        self._aligned = align_depth is None
        self.align_attitude = align_attitude
        self.detail = {}

    def _target(self, fw, extra: float = 0.0, lead: float = 0.0, depth: float | None = None):
        depth = self.depth if depth is None else depth
        c, a, n, v = self.feature(fw)
        c = np.asarray(c, float) + np.asarray(v, float) * lead
        a = np.asarray(a, float) / np.linalg.norm(a)
        n = np.asarray(n, float) / np.linalg.norm(n)
        n = n - (n @ a) * a
        n /= np.linalg.norm(n)
        # palm frame: z along the bar, x along the face normal (fingers point into the item along -x)
        z = a if (a @ fw.palm_z_pref(n)) >= 0 else -a
        x = n
        y = np.cross(z, x)
        Rp = np.column_stack([x, y, z])
        pp = c - Rp @ np.array([depth, 0.0, fw.z_plane]) + n * extra
        return pose_for_palm(fw.T_body_palm_deployed, pp, Rp)

    def _target_keep_attitude(self, fw, lead: float = 0.0, depth: float | None = None):
        """Robot position that puts the bar at the palm target with the robot's CURRENT attitude (the approach of a
        slowly rotating item then has no lateral error from the attitude lag of repeated re-plans)."""
        depth = self.depth if depth is None else depth
        c, a, n, v = self.feature(fw)
        c = np.asarray(c, float) + np.asarray(v, float) * lead
        b = fw.rig.adapter.body
        R = fw.data.xmat[b].reshape(3, 3)
        pb, Rb = fw.T_body_palm_deployed
        p = c - R @ (pb + Rb @ np.array([depth, 0.0, fw.z_plane]))
        return p, fw.data.xquat[b].copy()

    @staticmethod
    def _move_time(fw, p, q, vel=None) -> float:
        """Duration of the rest-to-rest move the firmware would plan from the current estimate to (p, q)."""
        est = fw.rig.est if fw.rig.est is not None else fw.rig.estimate()
        seg = plan_trapezoid([(est.pos, est.quat), (p, q)], fw.rig.firmware.mode, False, t0=0.0,
                             desired=None if vel is None else dict(vel=vel))
        return float(seg[-1].t - seg[0].t) if seg else 0.0

    def start(self, fw):
        self.t0 = fw.t
        prof = getattr(fw, "grasp_defaults", None) or GRASP_PROFILES["generic"]
        if self.depth is None:
            self.depth = prof["grasp_depth"]
        if self.close_x is None:
            self.close_x = prof["close_x"]
        self.state = "pre_grasp" if self.standoff is not None else "open"
        if self.standoff is not None:
            if self.lead == "auto":
                lead = 0.0
                for _ in range(2):                  # aim at the feature where it will be when the move ends
                    p, q = self._target(fw, self.standoff, lead)
                    lead = self._move_time(fw, p, q)
                self.detail["pre_grasp_lead_s"] = lead
            else:
                lead = float(self.lead)
            p, q = self._target(fw, self.standoff, lead)
            self.sub = FlyTo(p, q, faceforward=False, validate=True, label="pre_grasp")
            self.sub.start(fw)

    def _in_box(self, fw):
        c, a, n, v = self.feature(fw)
        pr, Rr = fw.palm_frame()
        vloc = Rr.T @ (np.asarray(c, float) - pr)
        return (self.close_x <= vloc[0] <= -0.035) and abs(vloc[1]) <= 0.025 and abs(vloc[2] - fw.z_plane) < self.box_z_tol, vloc

    def tick(self, fw, t):
        if t - self.t0 > self.timeout:
            return "timeout"
        if self.state == "pre_grasp":
            r = self.sub.tick(fw, t)
            if r is None:
                return None
            if r not in ("success", "already_there"):
                self.detail["pre_grasp"] = r
                return "pre_grasp_" + r
            self.state = "open"
        if self.state == "open":
            fw.gripper.set(1.0)
            self.state = "opening"
            return None
        if self.state == "opening":
            if not fw.gripper.moving:
                self.state, self.t_ret, self.t_app = "approach", -1.0, t
            return None
        if self.state == "approach":
            ok, vloc = self._in_box(fw)
            if ok:
                fw.gripper.set(0.0)
                if not self.on_the_move:
                    fw.rig.hold()
                self.state, self.t_close = "closing", t
                self.detail["bar_in_palm_at_close"] = vloc.tolist()
                return None
            if not self._aligned:
                lateral = max(abs(vloc[1]), abs(vloc[2] - fw.z_plane))
                if (vloc[0] >= self.align_depth - 0.02 and lateral <= self.align_tol) or t - self.t_app > self.align_timeout:
                    self._aligned, self.t_ret = True, -1.0
                    self.detail["aligned"] = dict(t=t - self.t_app, bar_in_palm=vloc.tolist(), lateral=float(lateral))
            if t >= self.t_ret:
                depth = self.depth if self._aligned else self.align_depth
                keep = self.keep_attitude and (self._aligned or not self.align_attitude)
                lead = 0.0 if self.lead == "auto" else float(self.lead)
                if self.intercept:
                    p, q = (self._target_keep_attitude(fw, 0.0, depth) if keep else self._target(fw, 0.0, 0.0, depth))
                    lead = self._move_time(fw, p, q, vel=self.v)
                p, q = self._target_keep_attitude(fw, lead, depth) if keep else self._target(fw, 0.0, lead, depth)
                fw.rig.move_to(p, q, faceforward=False, desired=dict(vel=self.v))
                self.t_ret = t + self.retarget
            return None
        if self.state == "closing":
            if self.on_the_move and fw.gripper.moving:
                return None
            if self.on_the_move and fw.gripper.c <= 0.0 and not getattr(self, "_held", False):
                fw.rig.hold()
                self._held = True
            if not fw.gripper.moving and t - self.t_close > 0.3:
                ok, vloc = self._in_box(fw)
                cage = (-0.135 <= vloc[0] <= -0.030) and abs(vloc[1]) <= 0.03
                self.detail.update(bar_in_palm_after=vloc.tolist(), caged_true=bool(cage))
                if self.use_rule and fw.grasp is not None and cage and self.rule_body is not None:
                    fw.grasp.attach(self.rule_body(fw))
                    self.detail["rule_attached"] = True
                if self.engage_interlock and getattr(fw, "holding_interlock", False):
                    fw.holding = True             # on the command state, not on a grasp sensor
                return "closed" if cage else "missed"
        return None

    def info(self):
        return dict(self.detail, label=self.label)


class Perch(Action):
    """Perch on a handrail by pure contact (firmware states: handrail localisation, approach pose, deploy arm,
    open gripper, complete pose, close gripper, check attached, spin down).

    rail: dict(pos, axis (bar direction), normal (away from the wall)).  depth / close_x: the grasp depth and the
    capture-box entry of the gripper profile (``gripper.grasp_profile``) unless given.  Check attached as the firmware
    does it: command the move back to the approach pose (``standoff`` out along the normal); the robot is attached if
    that move FAILS (the executive reports a tolerance error because the grasp holds it), not attached if it succeeds
    [R firmware behaviors/perch: PERCHING_CHECKING_ATTACHED, MOTION_FAILED -> spin down].  check='displacement' keeps
    the v0.2 check (back off ``check_back`` m for ``check_s`` s, attached if the robot moved < ``check_tol``).
    Result 'perched' or the failing step; ``info()`` has the cage play during the check (``check_displacement_m``)."""

    name = "perch"

    def __init__(self, rail: dict, standoff: float = 0.20, speed: float = 0.02, depth: float | None = None,
                 check_back: float = 0.03, check_s: float = 3.0, check_tol: float = 0.01, spin_down: bool = True,
                 close_x: float | None = None, check: str = "firmware", check_timeout_s: float = 60.0):
        self.rail, self.standoff, self.speed, self.depth = rail, standoff, speed, depth
        self.check_back, self.check_s, self.check_tol, self.spin_down = check_back, check_s, check_tol, spin_down
        self.close_x = close_x
        if check not in ("firmware", "displacement"):
            raise ValueError(check)
        self.check, self.check_timeout = check, float(check_timeout_s)
        self.detail = {}

    def _feature(self, fw):
        r = self.rail
        return np.asarray(r["pos"], float), np.asarray(r["axis"], float), np.asarray(r["normal"], float), np.zeros(3)

    def start(self, fw):
        prof = getattr(fw, "grasp_defaults", None) or GRASP_PROFILES["generic"]
        depth = prof["perch_depth"] if self.depth is None else self.depth
        close_x = prof["perch_close_x"] if self.close_x is None else self.close_x
        self.detail["depth"], self.detail["close_x"] = depth, close_x
        self.gf = GraspFeature(self._feature, standoff=self.standoff, approach_speed=self.speed, depth=depth,
                               close_x=close_x, label="perch", engage_interlock=False)
        self.state = "deploying_arm"
        self.sub = ArmTo(0.0, 0.0)
        self.sub.start(fw)

    def tick(self, fw, t):
        if self.state == "deploying_arm":
            if self.sub.tick(fw, t) is None:
                return None
            self.state = "grasping"
            self.gf.start(fw)
            return None
        if self.state == "grasping":
            r = self.gf.tick(fw, t)
            if r is None:
                return None
            self.detail["grasp"] = self.gf.info()
            if r != "closed":
                return "perch_" + r
            p0 = fw.data.xpos[fw.rig.adapter.body].copy()
            self.p0 = p0
            n = np.asarray(self.rail["normal"], float)
            est = fw.rig.est
            if self.check == "firmware":      # back to the approach pose: attached iff this move fails
                fw.rig.move_to(est.pos + n * self.standoff, est.quat, faceforward=False)
            else:
                fw.rig.move_to(est.pos + n * self.check_back, est.quat, faceforward=False, desired=dict(vel=0.02))
            self.state, self.t_state = "checking_attached", t
            self.detail["max_check_displacement_m"] = 0.0
            return None
        if self.state == "checking_attached":
            disp = float(np.linalg.norm(fw.data.xpos[fw.rig.adapter.body] - self.p0))
            self.detail["max_check_displacement_m"] = max(self.detail["max_check_displacement_m"], disp)
            if self.check == "firmware":
                f = fw.rig.firmware
                if f.state != "stopped" and t - self.t_state < self.check_timeout:
                    return None
                self.detail.update(check_displacement_m=disp, check_result=f.result)
                fw.rig.hold()
                if f.result in ("success", "already_there"):
                    return "not_attached"
            else:
                if t - self.t_state < self.check_s:
                    return None
                self.detail["check_displacement_m"] = disp
                fw.rig.hold()
                if disp > self.check_tol:
                    return "not_attached"
            if self.spin_down:
                fw.rig.idle()
                fw.rig.set_mode("off")
            self.state, self.t_state = "waiting_for_spin_down", t
            return None
        if self.state == "waiting_for_spin_down":
            if not self.spin_down or max(abs(w) for w in fw.rig.plant.impeller_speed) < 1.0 or t - self.t_state > 30:
                fw.perched = True
                return "perched"
        return None

    def info(self):
        return dict(self.detail)


class Unperch(Action):
    """Unperch (firmware states: map localisation, spin-up, open gripper, approach pose, stow arm)."""

    name = "unperch"

    def __init__(self, rail: dict, mode: str = "standard", back: float = 0.20, stow: bool = True):
        self.rail, self.mode, self.back, self.stow = rail, mode, back, stow

    def start(self, fw):
        fw.rig.set_mode(self.mode)
        self.state = "waiting_for_spin_up"

    def tick(self, fw, t):
        f = fw.rig.firmware
        if self.state == "waiting_for_spin_up":
            if f.state != "preparing":
                fw.rig.hold()
                fw.gripper.set(1.0)
                self.state = "opening_gripper"
            return None
        if self.state == "opening_gripper":
            if not fw.gripper.moving:
                est = fw.rig.est
                n = np.asarray(self.rail["normal"], float)
                fw.rig.move_to(est.pos + n * self.back, est.quat, faceforward=False)
                self.state = "moving_to_approach_pose"
            return None
        if self.state == "moving_to_approach_pose":
            if f.state == "stopped":
                fw.perched = False
                if not self.stow:
                    return "unperched"
                fw.gripper.set(0.0)
                self.sub = ArmTo(180.0, 0.0)
                self.sub.start(fw)
                self.state = "stowing_arm"
            return None
        if self.state == "stowing_arm":
            if self.sub.tick(fw, t) is not None:
                return "unperched"
        return None


# ---------------------------------------------------------------------------------------------- layer
class _InterlockedArm:
    """The layer's arm servo behind the holding interlock (v0.4): ``set`` is refused while the layer
    holds a captured item (the goal stays, so the joints hold); everything else is the wrapped ``ArmServo``."""

    def __init__(self, base, layer):
        self.base, self.layer = base, layer

    def set(self, tilt=None, pan=None):
        return self.layer.arm_command(tilt=tilt, pan=pan)

    def __getattr__(self, name):
        return getattr(self.base, name)


class FirmwareLayer:
    """Arm, gripper, grasp rule, dock magnets, station and zones around a FreeFlyerRig (see module doc).

    names: dict with joint / actuator / body names (``gripper.ARM_NAMES`` keys).  berth_pose: robot body pose on the
    berth.  station: StationModel or None; suction: SuctionFaces or None (its networks are re-solved at 100 Hz);
    zones: ZoneMap or None.  Telemetry rows every ``log_every`` firmware ticks go to ``log``.
    """

    def __init__(self, rig, names: dict, berth_pose=None, station=None, suction=None, zones=None, grasp_rule=None,
                 magnets=None, log_every: int = 6, charge_w: float = DOCK_DEFAULTS["charge_w"], recorder=None,
                 zone_band_exit: bool | str = True, fiducial_scope: str = "sticky", holding_interlock: bool = False):
        self.rig, self.model, self.data = rig, rig.model, rig.data
        self.zone_band_exit = zone_band_exit if zone_band_exit == "exit" else bool(zone_band_exit)
        if fiducial_scope not in ("sticky", "action"):
            raise ValueError(f"fiducial_scope {fiducial_scope!r} not in ('sticky', 'action')")
        self.fiducial_scope = fiducial_scope
        self.fiducial_releases = 0
        self.holding_interlock = bool(holding_interlock)
        self.holding = False
        self.interlock_refusals = 0
        m, d = self.model, self.data
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.names = names
        self.arm_joints = [jid(n) for n in names["joints"]]
        self.finger_joints = [jid(n) for n in names["fingers"]]
        self.palm = bid(names["palm_body"])
        self.arm = ArmServo(m, d, [aid(n) for n in names["actuators"]], self.arm_joints)
        if self.holding_interlock:
            self.arm = _InterlockedArm(self.arm, self)
        self.gripper = TendonGripper(m, self.finger_joints, open0=0.0)
        self.grasp = grasp_rule
        if self.grasp is not None:
            self.grasp.gripper = self.gripper
        self.station, self.suction, self.zones = station, suction, zones
        self.berth_pose = berth_pose
        self.magnets = magnets
        self.docked = False
        self.perched = False
        self.z_plane = -0.0497
        self.grasp_defaults = grasp_profile(m, names)
        self.T_body_palm_deployed = self._palm_in_body(tilt_deg=0.0, pan=0.0)
        self.action = None
        self.timeline = []
        self.log = []
        self.log_every = int(log_every)
        self.charge_w = float(charge_w)
        self.recorder = recorder
        self._ticks = 0
        dt = m.opt.timestep
        rig.sim.add_periodic(self._fast, dt, "arm_gripper")
        rig.sim.add_periodic(self._net, 5 * dt, "station")
        rig.sim.add_periodic(self._tick, rig.period, "actions")
        self.thrust_saturation_ticks = 0
        self.ticks = 0

    @property
    def t(self) -> float:
        return float(self.data.time)

    def _end_of_action(self, action, result) -> None:
        """Called once when an action returns a result or is abandoned on the caller's timeout: with
        fiducial_scope='action' (v0.4) the localiser's fiducial reference ends with the action."""
        if self.fiducial_scope == "action":
            loc = getattr(self.rig, "localizer", None)
            if loc is not None and hasattr(loc, "release") and getattr(loc, "ref", None) is not None:
                loc.release()
                self.fiducial_releases += 1

    def arm_command(self, tilt=None, pan=None) -> str | None:
        """Arm goal through the holding interlock (v0.4): 'refused_holding' (and the joints keep their
        goal) after a capture closed the gripper and until the gripper is commanded open; None when accepted."""
        if self.holding_interlock and self.holding:
            self.interlock_refusals += 1
            return "refused_holding"
        (self.arm.base if isinstance(self.arm, _InterlockedArm) else self.arm).set(tilt=tilt, pan=pan)
        return None

    def check_segment(self, seg, arm_exempt: bool = False) -> tuple:
        """Zone validation of a planned segment as the firmware does it for ``FlyTo``: ``zones.check_segment`` with
        the band rule when ``zone_band_exit`` (v0.4: a robot in the inflated band may leave it or
        move without getting closer -- ``ZoneMap.check_positions``; 'exit' = the departure rule only;
        False = v0.3), and with the arm-deployed inflation (v0.4) when the zone map has one
        (``ZoneMap(inflation_arm=...)``), the arm is deployed (``arm_deployed``) and the caller is not exempt
        (``FlyTo(arm_exempt=True)``).  ('ok', -1) without a zone map."""
        if self.zones is None:
            return "ok", -1
        arm = (not arm_exempt) and getattr(self.zones, "map_arm", None) is not None and self.arm_deployed()
        if arm:
            return self.zones.check_segment(seg, band_exit=self.zone_band_exit, arm=True)
        return self.zones.check_segment(seg, band_exit=self.zone_band_exit)

    def arm_deployed(self, tol_deg: float = 5.0) -> bool:
        """True when the arm's tilt joint (encoder) or its commanded tilt is more than tol_deg from the stowed tilt
        (180 deg): the robot is then larger than the 0.28 m radius of the stowed inflation."""
        stowed = tilt_to_q(180.0)
        q = float(self.data.qpos[self.model.jnt_qposadr[self.arm_joints[0]]])
        tol = math.radians(tol_deg)
        return abs(q - stowed) > tol or abs(float(self.arm.goal[0]) - stowed) > tol

    def _palm_in_body(self, tilt_deg: float, pan: float):
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.data.qpos
        d.qpos[m.jnt_qposadr[self.arm_joints[0]]] = tilt_to_q(tilt_deg)
        d.qpos[m.jnt_qposadr[self.arm_joints[1]]] = pan
        mujoco.mj_kinematics(m, d)
        b = self.rig.adapter.body
        Rb = d.xmat[b].reshape(3, 3)
        Rp = d.xmat[self.palm].reshape(3, 3)
        return Rb.T @ (d.xpos[self.palm] - d.xpos[b]), Rb.T @ Rp

    def palm_frame(self):
        """(position, rotation) of the palm body (true state)."""
        return self.data.xpos[self.palm].copy(), self.data.xmat[self.palm].reshape(3, 3).copy()

    def palm_z_pref(self, n):
        """Preferred sign of the palm z axis (keeps the robot's top toward the module's overhead when possible)."""
        return np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])

    def gripper_telemetry(self) -> dict:
        """What the robot itself can read: finger encoders, tendon position and load (motor current proxy)."""
        q = [float(self.data.qpos[self.model.jnt_qposadr[j]]) for j in self.finger_joints]
        return dict(finger_q=q, tendon=self.gripper.c, opening=self.gripper.opening(self.data),
                    tendon_load_Nm=self.gripper.tendon_load(self.model, self.data))

    # -- periodic tasks --------------------------------------------------------------------------------
    def _fast(self, sim):
        dt = self.model.opt.timestep
        if self.holding and self.gripper.cmd > 0.0:
            self.holding = False                  # the interlock ends when the gripper is commanded open
        self.arm.step(self.data, dt)
        self.gripper.step(dt)

    def _net(self, sim):
        dt = 5 * self.model.opt.timestep
        if self.station is not None:
            self.station.advance(dt, solve=False)
        if self.suction is not None:
            self.suction.update_networks()
        elif self.station is not None:
            for net in self.station.vent.cabin_fans.values():
                net.solve()
        if self.grasp is not None:
            self.grasp.check_slip()

    def _tick(self, sim):
        self.ticks += 1
        f = self.rig.firmware
        al = f.last
        if "servo_cmd" in al:
            sc = np.asarray(al["servo_cmd"])
            if np.any(sc >= 255) or np.any(np.asarray(al.get("theta", np.zeros(12))) >= self.rig.plant.p.max_flap_travel - 1e-9):
                self.thrust_saturation_ticks += 1
        if self.docked:
            self.rig.battery.energy_wh += self.charge_w * self.rig.period / 3600.0
            cap = self.rig.battery.n * self.rig.battery.cap
            self.rig.battery.energy_wh = min(self.rig.battery.energy_wh, cap)
        if self.action is not None:
            r = self.action.tick(self, self.t)
            if r is not None:
                self._result = r
                self._end_of_action(self.action, r)
                self.action = None
        self._ticks += 1
        if self.recorder is not None and self._ticks % self.log_every == 0:
            self.recorder(self)

    # -- running ---------------------------------------------------------------------------------------
    def run(self, action: Action, timeout: float = 600.0, label: str = "") -> str:
        """Start an action and step until it returns a result (or timeout); logs it in ``timeline``."""
        self._result = None
        t0 = self.t
        bat0 = self.rig.battery.soc
        action.start(self)
        self.action = action
        step = int(round(self.rig.period / self.model.opt.timestep))
        while self._result is None and self.t - t0 < timeout:
            self.rig.step(step)
        r = self._result if self._result is not None else "timeout"
        if self._result is None:
            self._end_of_action(action, r)
            self.action = None
        self.timeline.append(dict(t0=t0, t1=self.t, name=action.name, label=label or getattr(action, "label", ""),
                                  result=r, soc0=bat0, soc1=self.rig.battery.soc, info=action.info()))
        return r


# ---------------------------------------------------------------------------------------------- factory test
class NozzleTestMode:
    """Stand-in executive for the factory test: fans at a level, one nozzle's servo fully open at a time.
    Exposes the attributes FreeFlyerRig reads from a firmware (tick(), state, result, mode, last, speed_level)."""

    class _Mode:
        name = "factory_test"

    def __init__(self, plant_params, level: int = 3):
        self.p = plant_params
        self.level = int(level)
        self.open_nozzle = None
        self.state, self.result, self.mode = "test", None, self._Mode()
        self.last = {}

    def speed_level(self) -> int:
        return self.level

    def tick(self, t, est, impeller_speed=None):
        cmd = int(self.p.imp_cmd_per_rad_s * self.p.imp_levels[self.level - 1] + 0.5) & 0xFF if self.level else 0
        servo = np.zeros(12, int)
        if self.open_nozzle is not None:
            servo[self.open_nozzle] = int(self.p.servo_pwm_max)
        self.last = dict(impeller_cmd=np.array([cmd, cmd]), servo_cmd=servo, force=np.zeros(3), torque=np.zeros(3),
                         state=self.state, result=self.result, mode=self.mode.name)
        return self.last


def nozzle_factory_test(rig, carriage_dofs, fire_s: float = 1.5, gap_s: float = 2.0, level: int = 3,
                        settle_s: float = 3.0, nozzles=range(12)) -> list:
    """Fire each nozzle fully open for ``fire_s`` with the fans at ``level`` while the robot rides an air-bearing
    carriage (planar joints, ids of the x, y slides and the yaw hinge in ``carriage_dofs`` as dof addresses), log
    the response.  The rig's firmware is replaced by ``NozzleTestMode`` for the duration.  Returns one dict per
    nozzle: steady thrust (plant), impulse, carriage delta-v (x, y), delta yaw rate, peak plenum pressure and the
    direction check against the nozzle table."""
    m, d = rig.model, rig.data
    test = NozzleTestMode(rig.plant.p, level)
    saved = rig.firmware
    rig.firmware = test
    dt = m.opt.timestep
    step = int(round(rig.period / dt))
    out = []
    ix, iy, iz = carriage_dofs
    rig.run(settle_s)
    b = rig.adapter.body
    for k in nozzles:
        v0 = np.array([d.qvel[ix], d.qvel[iy]])
        w0 = float(d.qvel[iz])
        test.open_nozzle = int(k)
        t0 = d.time
        imp = 0.0
        thr = []
        dp = 0.0
        while d.time - t0 < fire_s - 1e-9:
            rig.step(step)
            thr.append(float(rig.plant.thrust[k]))
            imp += rig.plant.thrust[k] * rig.period
            dp = max(dp, float(max(mm.dp for mm in rig.plant.modules)))
        test.open_nozzle = None
        v1 = np.array([d.qvel[ix], d.qvel[iy]])
        w1 = float(d.qvel[iz])
        R = d.xmat[b].reshape(3, 3)
        fdir_body = -rig.plant.p.nozzle_exhaust_dir[k]
        fdir_w = R @ fdir_body
        dv = v1 - v0
        planar = abs(fdir_w[2]) < 0.5
        dir_ok = bool(float(dv @ fdir_w[:2]) > 0) if planar else None
        out.append(dict(nozzle=int(k), thrust_steady_N=float(np.mean(thr[len(thr) // 2:])), impulse_Ns=float(imp),
                        dv_xy=dv.tolist(), dwz=w1 - w0, plenum_dp_max_Pa=dp, force_dir_world=fdir_w.tolist(),
                        planar=planar, direction_ok=dir_ok))
        t1 = d.time
        while d.time - t1 < gap_s - 1e-9:
            rig.step(step)
    rig.firmware = saved
    return out
