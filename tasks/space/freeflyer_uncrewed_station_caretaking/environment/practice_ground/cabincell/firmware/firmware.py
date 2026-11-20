"""TaskFirmware: the robot calls of the caretaking task (the robot API) on top of the simulation
package's firmware layer.

The layer is ``spacekit.primitives.FirmwareLayer`` built with every 0.4.0 option of this task on: the zone map with
the arm-deployed inflation (``ZoneMap(inflation_arm=firmware.zone.inflation_arm_m)``), ``fiducial_scope="action"`` (a
fiducial-relative estimate ends with the action that set it), ``holding_interlock=True`` (after a capture until
gripper(open) every arm command is refused) and the band rule ``zone_band_exit=True`` (a segment that
starts inside the inflated band may leave it or move without getting closer).  The rules this module adds:

* fly_to: REFUSED_STATE while docked; LOCALIZATION when the confidence is not GOOD at the start; the zone check of the
  layer (arm-deployed map while the arm is deployed, no exemption); TIMEOUT at planned end + extra.
* Bounds of a move's plan at the interface (fly_to and fly_rel): a ``max_speed`` below ``MIN_MAX_SPEED_M_S`` is
  INVALID_ARGS; a goal outside the zone map's grid (more than ``GRID_MARGIN_M`` beyond it) is answered as the zone check
  answers it (REFUSED_ZONE, the end point is outside the keep-in volume) without planning the segment.  So the longest
  plan the firmware builds is the grid's diagonal at that speed, and the zone check's 10 Hz samples stay bounded.
* fly_rel: the plate must be visible (``localisation.fiducial_check``) at the call, else NO_FIDUCIAL; the zone check of
  ``zonecheck.ZoneRules.check_fly_rel`` (arm-deployed map outside the approach zone of the plate's own unit / dock);
  the fiducial-relative estimate is gated by the plate's visibility at the fiducial rate; the plate lost longer than
  the firmware's hold: the move stops, NO_FIDUCIAL; the estimate is back on the map when the call ends (scope 'action').
* arm_deploy() / arm(): REFUSED_STATE while docked; REFUSED_HOLDING under the interlock; REFUSED_ZONE when the target
  attitude is deployed and the robot's position is not free on the envelope map (``ZoneRules.check_envelope``);
  arm(): joint limits from spec arm.urdf_joints (INVALID_ARGS outside).  arm_stow(): interlock only.
* capture: ``capture.Capture`` (gate, lock rule, NO_BAR, LOST_BAR, STALLED, TIMEOUT, palm_dir); REFUSED_STATE while
  docked or when the arm is not at the grasp attitude; REFUSED_HOLDING under the interlock; REFUSED_BUDGET when the
  cell's budget hook says so (it is consumed only when the call starts).  From the close until the next motion call
  (fly_to, fly_rel, hold, stop, dock, undock, capture) the robot holds where it stopped in the inertial sense (the
  inertial hold of ``_relatch``: the map estimate's wander is not chased); DONE comes firmware close_settle_s after
  the close.
* dock(berth): REFUSED_STATE when docked; REFUSED_ARM unless both arm joints read stowed; the berthing cone
  (``spec.apply_dock_defaults``); REFUSED_CONE / FAILED; undock().
* lamp(): the lamp's power booked (``power.TaskBattery``) and its light in the localisation (``CabinRichness``).
* status() / pose(): the fields of the robot API, quantised as the sensors (``power.Telemetry``).
Blocking calls run the layer's actions; a ``clock`` (``left()`` = simulated seconds left in the site's budget) makes a
call that runs out of time answer TIME_UP.
"""
from __future__ import annotations

import math

import numpy as np

from spacekit import primitives as PR
from spacekit.control import plan_trapezoid
from spacekit.gripper import q_to_tilt, tilt_to_q
from spacekit.rotations import mat_to_quat, quat_conj, quat_mul, quat_normalize, quat_to_mat

from .capture import Capture

MODES = ("STANDARD", "AGILE", "HUSH", "FINE")
LAMPS = ("forward", "aft")
FLY_CODES = {"success": "SUCCESS", "already_there": "SUCCESS", "tolerance_pos": "TOLERANCE_PATH",
             "tolerance_pos_endpoint": "TOLERANCE_END", "tolerance_att": "TOLERANCE_ATT",
             "localization_lost": "LOCALIZATION", "overspeed_stop": "OVERSPEED_STOP", "overspeed_cutoff": "OVERSPEED_STOP",
             "refused_keep_in": "REFUSED_ZONE", "refused_keep_out": "REFUSED_ZONE", "timeout": "TIMEOUT",
             "speed_timeout": "TIMEOUT", "cancelled": "TIMEOUT", "no_fiducial": "NO_FIDUCIAL"}
MIN_MAX_SPEED_M_S = 0.005     # the lowest max_speed of fly_to / fly_rel: a quarter of the slowest flight mode's speed limit
GRID_MARGIN_M = 1e-6          # a goal farther than this outside the zone map's grid is refused before any planning
TILT_RANGE = (-30.0, 180.0)     # tilt joint limits of spec arm.urdf_joints (q -2.0944 .. 1.5708 rad), see arm_limits()
PAN_RANGE = (-90.0, 90.0)


def arm_limits(robot_values: dict) -> tuple:
    """((tilt_min, tilt_max), (pan_min, pan_max)) [deg] from spec arm.urdf_joints (firmware tilt, 0 = deployed aft)."""
    J = robot_values["arm.urdf_joints"]["value"]
    prox = next(j for j in J if j["name"].endswith("arm_proximal_joint"))
    dist = next(j for j in J if j["name"].endswith("arm_distal_joint"))
    t = sorted((q_to_tilt(prox["limit"]["lower"]), q_to_tilt(prox["limit"]["upper"])))
    p = sorted((math.degrees(dist["limit"]["lower"]), math.degrees(dist["limit"]["upper"])))
    return (round(t[0], 3), round(t[1], 3)), (round(p[0], 3), round(p[1], 3))


def _vec(x, n):
    try:
        a = np.asarray(x, float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        return None
    return a


class Clock:
    """Simulated-time budget of a site: ``left()`` seconds (None = unlimited)."""

    def __init__(self, data, t0: float, limit_s: float | None = None):
        self.data, self.t0, self.limit = data, float(t0), limit_s

    def left(self) -> float:
        if self.limit is None:
            return math.inf
        return self.t0 + self.limit - float(self.data.time)


class FlyRel(PR.Action):
    """fly_rel's motion: reference the plate, settle on it, validate the segment planned from the fiducial-relative
    estimate (``ZoneRules.check_fly_rel``), fly; plate lost longer than ``loss_hold_s`` -> 'no_fiducial'."""

    name = "fly_rel"

    def __init__(self, tf, plate: dict, pos_w, quat_w, desired=None, faceforward: bool = False):
        self.tf, self.plate = tf, plate
        self.pos, self.quat = np.asarray(pos_w, float), np.asarray(quat_w, float)
        self.desired, self.ff = desired, faceforward
        self.detail = dict(plate=plate["id"])
        self.sub = None

    def start(self, fw):
        P = self.plate
        self.loc = self.tf.fid_loc
        self.tf.stack_top.reference(name=P["id"], points=P["points"], normal=P["normal"], cam_offset_body=self.tf.awc_off,
                                    cam_axis_body=self.tf.awc_axis, skip=P["skip"])
        fw.rig.hold()
        self.t0 = fw.t
        self.state = "settling"

    def tick(self, fw, t):
        sp = self.tf.spec
        if self.state == "settling":
            if t - self.t0 < sp.rel_settle_s:
                return None
            if not self.loc.visible or not str(self.loc.last_source).startswith("fiducial"):
                self.detail["why"] = "plate not seen while settling"
                return "no_fiducial"
            est = fw.rig.est
            seg = plan_trapezoid([(est.pos, est.quat), (self.pos, self.quat)], fw.rig.firmware.mode, self.ff, t0=fw.t,
                                 desired=self.desired)
            r, k, det = self.tf.zones.check_fly_rel(seg, self.plate["owner"], fw.arm_deployed(sp.arm_deployed_tol_deg),
                                                    band_exit=fw.zone_band_exit)
            self.detail["zone"] = dict(result=r, sample=int(k), runs=det.get("runs"), inside=det.get("inside"))
            if r != "ok":
                fw.rig.hold()
                return "refused_" + r
            self.sub = PR.FlyTo(self.pos, self.quat, faceforward=self.ff, validate=False, desired=self.desired,
                                resume=False, timeout_margin_s=sp.move_timeout_extra_s, label="fly_rel")
            self.sub.start(fw)
            self.state = "moving"
            return None
        if self.loc.hidden_since is not None and t - self.loc.hidden_since > sp.loss_hold_s:
            fw.rig.stop()
            self.detail["why"] = f"plate lost for more than {sp.loss_hold_s} s"
            return "no_fiducial"
        return self.sub.tick(fw, t)

    def info(self):
        d = dict(self.detail)
        if self.sub is not None:
            d["move"] = self.sub.info()
        d["true_pos_error_m"] = float(np.linalg.norm(self.tf.data.xpos[self.tf.body] - self.pos))
        return d


class GoalOutsideGrid(PR.Action):
    """fly_to toward a goal outside the zone map's grid: answered as the zone check of ``PR.FlyTo`` answers it -- the end
    point lies outside the keep-in volume, so 'refused_keep_in' at the first firmware tick that is not spent preparing,
    the robot holding -- but without planning the segment (its length, and with it the plan's samples, is unbounded
    there)."""

    name = "fly_to"

    def __init__(self, pos, label: str = ""):
        self.pos = np.asarray(pos, float)
        self.label = label
        self.violation = dict(result="keep_in", sample=-1, outside_grid=True)

    def tick(self, fw, t):
        if fw.rig.firmware.state == "preparing":
            return None
        fw.rig.hold()
        return "refused_keep_in"

    def info(self):
        return dict(goal=self.pos.tolist(), resumes=0, violation=self.violation, label=self.label, timed_out=None)


class TaskFirmware:
    """The calls of the robot API (module doc).  Built by ``stack.build_stack``; the attributes are the stack's parts."""

    def __init__(self, rig, layer, spec, zones, detector, stack_top, fid_loc, battery, telemetry, plates: dict,
                 berths: dict | None = None, magnets: dict | None = None, richness=None, robot_values: dict | None = None,
                 awc: tuple | None = None, clock: Clock | None = None, capture_budget=None, t0: float | None = None):
        self.rig, self.layer, self.spec, self.zones = rig, layer, spec, zones
        self.model, self.data = rig.model, rig.data
        self.body = rig.adapter.body
        self.detector, self.stack_top, self.fid_loc = detector, stack_top, fid_loc
        self.battery, self.telemetry, self.richness = battery, telemetry, richness
        self.plates = dict(plates)
        self.berths = dict(berths or {})
        self.magnets = dict(magnets or {})
        self.berth = None
        self.awc_off, self.awc_axis = awc if awc is not None else (np.zeros(3), np.array([-1.0, 0.0, 0.0]))
        self.tilt_range, self.pan_range = arm_limits(robot_values) if robot_values else (TILT_RANGE, PAN_RANGE)
        self.t0 = float(self.data.time) if t0 is None else float(t0)
        self.clock = clock
        self.capture_budget = capture_budget
        self.last_result = None
        self.calls = []
        self.log_calls = True
        self.relatches = 0
        self._src = None
        self.inertial = None              # None | "pending" | (true pos, true quat) of the inertial hold after a capture
        rig.sim.add_periodic(self._relatch, rig.period, "firmware_relatch")

    # ---------------------------------------------------------------- hold point across a change of the estimate's source
    @staticmethod
    def _src_class(lo) -> str:
        s = str((lo or {}).get("source", "map"))
        return "RAIL" if s == "rail" else ("FIDUCIAL" if s.startswith("fiducial") else "MAP")

    def _relatch(self, sim) -> None:
        """Hold point of the controller's latched hold (one firmware tick behind the estimate).
        (1) When the estimate's source changes (map <-> fiducial-relative, rail -> map) the hold point is re-latched on
        the new estimate: the robot does not lurch by the difference of the two estimates.
        (2) The inertial hold after a capture closed (``inertial``, until the next motion call): the pose where the robot
        stopped is held in the inertial sense (as an IMU-propagated hold does), i.e. the hold point is that pose carried
        into the current estimate's frame every tick, so the map estimate's own wander is not chased: a robot clamped to
        a bar and a free one hold alike and both estimates wander as the localisation does.  Moves are unaffected."""
        c = self._src_class(getattr(self.rig, "loc_out", None))
        est = self.rig.est
        if est is not None:
            ctl = self.rig.firmware.ctl
            changed = self._src is not None and c != self._src
            if ctl.stopped_mode and self.inertial is not None:
                pt = np.asarray(self.data.xpos[self.body], float)
                qt = np.asarray(self.data.xquat[self.body], float)
                if isinstance(self.inertial, str):
                    self.inertial = (pt.copy(), qt.copy())
                p0, q0 = self.inertial
                ctl.prev_position = np.asarray(est.pos, float) + (p0 - pt)
                ctl.prev_att = quat_normalize(quat_mul(quat_mul(np.asarray(est.quat, float), quat_conj(qt)), q0))
            elif ctl.stopped_mode and changed:
                ctl.prev_position = np.asarray(est.pos, float).copy()
                ctl.prev_att = np.asarray(est.quat, float).copy()
                self.relatches += 1
        self._src = c

    def _inertial_on(self) -> None:
        self.inertial = "pending"

    def _motion_call(self) -> None:
        """A motion call ends the inertial hold of a capture (hold() / stop() station-keep on the estimate again)."""
        if self.inertial is not None and self.rig.est is not None:
            ctl = self.rig.firmware.ctl
            ctl.prev_position = np.asarray(self.rig.est.pos, float).copy()     # no lurch into the estimate's frame
            ctl.prev_att = np.asarray(self.rig.est.quat, float).copy()
        self.inertial = None

    # ---------------------------------------------------------------- helpers
    def t(self) -> float:
        return float(self.data.time) - self.t0

    def _left(self) -> float:
        return self.clock.left() if self.clock is not None else math.inf

    def _log(self, call: str, code: str, **kw):
        if self.log_calls:
            self.calls.append(dict(t=round(self.t(), 3), call=call, code=code, **kw))
        return code

    def _run(self, action, timeout: float, label: str) -> str:
        left = self._left()
        if left <= 0:
            return "TIME_UP"
        to = min(float(timeout), left)
        r = self.layer.run(action, timeout=max(to, 1e-3), label=label)
        if r == "timeout" and self.layer.timeline[-1]["t1"] - self.layer.timeline[-1]["t0"] >= to - 1e-9 and to >= left - 1e-9:
            return "TIME_UP"
        return r

    def _release_rail(self):
        if getattr(self.stack_top, "rail", False):
            self.stack_top.release_rail()

    def _est(self):
        return self.rig.est if self.rig.est is not None else self.rig.estimate()

    def arm_angles(self):
        """(tilt, pan) [deg] of the joint encoders (truth of the joints; status() quantises)."""
        q = [float(self.data.qpos[self.model.jnt_qposadr[j]]) for j in self.layer.arm_joints]
        return float(q_to_tilt(q[0])), math.degrees(q[1])

    def arm_stowed(self) -> bool:
        tl, pn = self.arm_angles()
        tol = self.spec.arm_deployed_tol_deg
        return abs(tl - 180.0) <= tol and abs(pn) <= tol

    def arm_at_grasp(self) -> bool:
        tl, pn = self.arm_angles()
        tol = self.spec.capture_deployed_tol_deg
        return abs(tl) <= tol and abs(pn) <= tol

    def _desired(self, max_speed):
        if max_speed is None:
            return None
        return dict(vel=min(float(max_speed), float(self.rig.firmware.mode.hard_limit_vel)))

    def goal_in_grid(self, pos) -> bool:
        """False when ``pos`` lies more than GRID_MARGIN_M outside the zone map's grid (every such goal is refused by
        the zone check: its cell is unknown)."""
        zm = getattr(self.zones, "map", None)
        if zm is None:
            return True
        p = np.asarray(pos, float)
        lo = np.asarray(zm.origin, float) - GRID_MARGIN_M
        hi = np.asarray(zm.origin, float) + np.asarray(zm.dim, float) * float(zm.res) + GRID_MARGIN_M
        return bool(np.all(p >= lo) and np.all(p <= hi))

    def _planned_s(self, pos, quat, ff, desired) -> float:
        e = self._est()
        seg = plan_trapezoid([(e.pos, e.quat), (pos, quat)], self.rig.firmware.mode, ff, desired=desired)
        return float(seg[-1].t - seg[0].t) if seg else 0.0

    def plate_pose(self, plate_id: str):
        """(centre, R columns x = long axis, y = z cross x, z = normal into the cabin) of a plate (actual pose)."""
        P = self.plates[plate_id]
        return np.asarray(P["center"], float), np.asarray(P["R"], float)

    def plate_check(self, plate_id: str) -> dict:
        from .localisation import fiducial_check
        P = self.plates[plate_id]
        G = self.fid_loc.gate
        return fiducial_check(self.data, G["cam"], G["tester"], P["points"], P["normal"], P["skip"], G["min_points"])

    # ---------------------------------------------------------------- query calls
    def status(self) -> dict:
        f = self.rig.firmware
        S = self.telemetry.snapshot()
        est = self._est()
        if self.layer.docked:
            motion = "DOCK"
        elif f.state == "idle" or est.confidence != 0:
            motion = "DRIFT"
        elif f.state in ("moving", "preparing", "stopping"):
            motion = "FLY"
        else:
            motion = "STOP"
        pct = S["battery_pct"]
        return dict(motion=motion, last_result=self.last_result, battery_pct=pct, low_battery=bool(pct < self.spec.low_flag_pct),
                    flight_mode=f.mode.name.upper(), fan_level=int(f.speed_level()), arm_tilt_deg=S["arm_tilt_deg"],
                    arm_pan_deg=S["arm_pan_deg"], tendon_pct=round(100.0 * float(self.layer.gripper.c), 1),
                    arm_current_a=S["arm_current_a"], imu=S["imu"], interlock=bool(self.layer.holding), t=round(self.t(), 4))

    def pose(self) -> dict:
        e = self._est()
        lo = getattr(self.rig, "loc_out", None) or {}
        cp = np.asarray(lo.get("cov_pos", np.eye(3) * 1e-8))
        ca = np.asarray(lo.get("cov_att", np.eye(3) * 1e-8))
        return dict(pos=[float(x) for x in e.pos], quat=[float(x) for x in e.quat],
                    cov_diag=[float(x) for x in np.concatenate([np.diag(cp), np.diag(ca)])],
                    confidence=("GOOD", "POOR", "LOST")[int(e.confidence)], loc_mode=self.stack_top.mode, t=round(self.t(), 4))

    # ---------------------------------------------------------------- simple commands
    def set_mode(self, mode) -> str:
        if not isinstance(mode, str) or mode.upper() not in MODES:
            return self._log("set_mode", "INVALID_ARGS")
        r = self._run(PR.SetMode(mode.lower()), 30.0, f"set_mode:{mode}")
        return self._log("set_mode", "TIME_UP" if r == "TIME_UP" else "DONE", mode=mode.upper())

    def stop(self) -> str:
        self._motion_call()
        self.rig.stop()
        return self._log("stop", "OK")

    def hold(self) -> str:
        self._motion_call()
        self.rig.hold()
        return self._log("hold", "OK")

    def lamp(self, which, on) -> str:
        if which not in LAMPS or not isinstance(on, bool):
            return self._log("lamp", "INVALID_ARGS")
        self.battery.set_lamp(which, on)
        if self.richness is not None:
            self.richness.set_lamp(which, on)
        return self._log("lamp", "OK", which=which, on=on)

    def set_lighting(self, module: str, state: str) -> None:
        """The station's lighting state of a module changed (the station service calls this; 'off' = failed circuit)."""
        if self.richness is not None:
            self.richness.set_lighting(module, state)

    # ---------------------------------------------------------------- motion
    def fly_to(self, pos, quat, face_forward=True, max_speed=None) -> str:
        p, q = _vec(pos, 3), _vec(quat, 4)
        if p is None or q is None or np.linalg.norm(q) < 1e-9 or not isinstance(face_forward, bool) or \
                (max_speed is not None and (not isinstance(max_speed, (int, float)) or not math.isfinite(max_speed) or max_speed < MIN_MAX_SPEED_M_S)):
            return self._log("fly_to", "INVALID_ARGS")
        if self.layer.docked:
            return self._set_last(self._log("fly_to", "REFUSED_STATE"))
        self._motion_call()
        self._release_rail()
        if self._est().confidence != 0:
            return self._set_last(self._log("fly_to", "LOCALIZATION"))
        q = quat_normalize(q)
        if not self.goal_in_grid(p):
            r = self._run(GoalOutsideGrid(p, label="fly_to"), self.spec.move_timeout_extra_s + 30.0, "fly_to")
            code = "TIME_UP" if r == "TIME_UP" else FLY_CODES.get(r, "TIMEOUT")
            return self._set_last(self._log("fly_to", code, raw=r))
        des = self._desired(max_speed)
        to = self._planned_s(p, q, face_forward, des) + self.spec.move_timeout_extra_s + 30.0
        act = PR.FlyTo(p, q, faceforward=face_forward, validate=True, desired=des, resume=False,
                       timeout_margin_s=self.spec.move_timeout_extra_s, label="fly_to")
        r = self._run(act, to, "fly_to")
        code = "TIME_UP" if r == "TIME_UP" else FLY_CODES.get(r, "TIMEOUT")
        return self._set_last(self._log("fly_to", code, raw=r))

    def fly_rel(self, fiducial_id, pos, quat, max_speed=None) -> str:
        p, q = _vec(pos, 3), _vec(quat, 4)
        if not isinstance(fiducial_id, str) or fiducial_id not in self.plates or p is None or q is None or np.linalg.norm(q) < 1e-9 \
                or (max_speed is not None and (not isinstance(max_speed, (int, float)) or not math.isfinite(max_speed) or max_speed < MIN_MAX_SPEED_M_S)):
            return self._log("fly_rel", "INVALID_ARGS")
        if self.layer.docked:
            return self._set_last(self._log("fly_rel", "REFUSED_STATE"))
        self._motion_call()
        self._release_rail()
        vis = self.plate_check(fiducial_id)
        if not vis["visible"]:
            return self._set_last(self._log("fly_rel", "NO_FIDUCIAL", why="not visible at the call", check=vis))
        c, Rp = self.plate_pose(fiducial_id)
        pw = c + Rp @ p
        qw = mat_to_quat(Rp @ quat_to_mat(quat_normalize(q)))
        if not self.goal_in_grid(pw):
            return self._set_last(self._log("fly_rel", "REFUSED_ZONE", raw="refused_keep_in", sample=-1))
        des = self._desired(max_speed)
        plate = self.plates[fiducial_id]
        e = self._est()
        seg = plan_trapezoid([(e.pos, e.quat), (pw, qw)], self.rig.firmware.mode, False, t0=float(self.data.time), desired=des)
        r0, k0, det0 = self.zones.check_fly_rel(seg, plate["owner"], self.layer.arm_deployed(self.spec.arm_deployed_tol_deg),
                                                band_exit=self.layer.zone_band_exit)
        if r0 != "ok":
            return self._set_last(self._log("fly_rel", "REFUSED_ZONE", raw="refused_" + r0, sample=int(k0)))
        to = self._planned_s(pw, qw, False, des) + self.spec.rel_settle_s + self.spec.move_timeout_extra_s + 30.0
        act = FlyRel(self, plate, pw, qw, desired=des)
        r = self._run(act, to, "fly_rel")
        code = "TIME_UP" if r == "TIME_UP" else FLY_CODES.get(r, "TIMEOUT")
        return self._set_last(self._log("fly_rel", code, raw=r, plate=fiducial_id))

    def _set_last(self, code: str) -> str:
        if code not in ("INVALID_ARGS",):
            self.last_result = code
        return code

    # ---------------------------------------------------------------- arm and gripper
    def _arm_goal(self, call: str, tilt: float, pan: float, wait: bool, envelope: bool) -> str:
        if self.layer.holding_interlock and self.layer.holding:
            self.layer.interlock_refusals += 1
            return self._log(call, "REFUSED_HOLDING")
        if envelope and abs(tilt - 180.0) > self.spec.arm_deployed_tol_deg:
            e = self._est()
            z = self.zones.check_envelope(e.pos)
            if z != "ok":
                return self._log(call, "REFUSED_ZONE", raw=z)
        r = self.layer.arm_command(tilt=tilt_to_q(float(tilt)), pan=math.radians(float(pan)))
        if r is not None:                         # the layer's interlock (same rule)
            return self._log(call, "REFUSED_HOLDING")
        if not wait:
            return self._log(call, "STARTED", tilt=tilt, pan=pan)
        r = self._run(PR.ArmTo(float(tilt), float(pan)), 120.0, call)
        return self._log(call, "TIME_UP" if r == "TIME_UP" else "DONE", tilt=tilt, pan=pan)

    def arm_deploy(self, wait=False) -> str:
        if not isinstance(wait, bool):
            return self._log("arm_deploy", "INVALID_ARGS")
        if self.layer.docked:
            return self._log("arm_deploy", "REFUSED_STATE")
        return self._arm_goal("arm_deploy", 0.0, 0.0, wait, envelope=True)

    def arm_stow(self, wait=False) -> str:
        if not isinstance(wait, bool):
            return self._log("arm_stow", "INVALID_ARGS")
        return self._arm_goal("arm_stow", 180.0, 0.0, wait, envelope=False)

    def arm(self, tilt_deg, pan_deg, wait=False) -> str:
        ok = all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (tilt_deg, pan_deg))
        if not ok or not isinstance(wait, bool) or not (self.tilt_range[0] <= tilt_deg <= self.tilt_range[1]) or \
                not (self.pan_range[0] <= pan_deg <= self.pan_range[1]):
            return self._log("arm", "INVALID_ARGS")
        if self.layer.docked:
            return self._log("arm", "REFUSED_STATE")
        return self._arm_goal("arm", float(tilt_deg), float(pan_deg), wait, envelope=True)

    def gripper(self, open) -> dict:
        if not isinstance(open, bool):
            return dict(code=self._log("gripper", "INVALID_ARGS"))
        r = self._run(PR.GripperCmd(open), 10.0, "gripper:" + ("open" if open else "close"))
        code = "TIME_UP" if r == "TIME_UP" else "DONE"
        self._log("gripper", code, open=open)
        return dict(code=code, tendon_pct=round(100.0 * float(self.layer.gripper.c), 1))

    # ---------------------------------------------------------------- capture
    def capture(self, bar_pos, bar_axis, palm_dir=None) -> str:
        c, a = _vec(bar_pos, 3), _vec(bar_axis, 3)
        pd = None if palm_dir is None else _vec(palm_dir, 3)
        if c is None or a is None or np.linalg.norm(a) < 1e-9 or (palm_dir is not None and (pd is None or np.linalg.norm(pd) < 1e-9)):
            return self._log("capture", "INVALID_ARGS")
        if self.layer.docked or not self.arm_at_grasp():
            return self._set_last(self._log("capture", "REFUSED_STATE"))
        if self.layer.holding_interlock and self.layer.holding:
            return self._set_last(self._log("capture", "REFUSED_HOLDING"))
        if self.capture_budget is not None and not self.capture_budget():
            return self._set_last(self._log("capture", "REFUSED_BUDGET"))
        self._motion_call()
        self._release_rail()
        cap = Capture(self.detector, self.stack_top, self.spec, c, a, pd, label="capture", on_close=self._inertial_on)
        r = self._run(cap, self.spec.capture_timeout_s + 10.0, "capture")
        self.last_capture = cap
        if r == "TIME_UP":
            code = "TIME_UP"
        else:
            code = r if r in ("DONE", "NO_BAR", "LOST_BAR", "STALLED", "TIMEOUT") else "TIMEOUT"
        info = cap.info()
        return self._set_last(self._log("capture", code, locked=info.get("locked"), end=info.get("end"),
                                        lock_t=info.get("lock_t")))

    # ---------------------------------------------------------------- dock
    def set_berth(self, berth: int) -> None:
        """Point the layer's berth pose and dock model at a berth (1 or 2)."""
        self.berth = int(berth)
        self.layer.berth_pose = self.berths[self.berth]
        self.layer.magnets = self.magnets[self.berth]

    def dock(self, berth) -> str:
        if berth not in (1, 2) or isinstance(berth, bool) or berth not in self.berths:
            return self._log("dock", "INVALID_ARGS")
        if self.layer.docked:
            return self._set_last(self._log("dock", "REFUSED_STATE"))
        if not self.arm_stowed():
            return self._set_last(self._log("dock", "REFUSED_ARM"))
        self._motion_call()
        self._release_rail()
        self.set_berth(berth)
        r = self._run(PR.Dock(berth_speed=self.spec.approach_speed_m_s, mode="standard", max_reapproach=2), 600.0, "dock")
        code = "TIME_UP" if r == "TIME_UP" else {"docked": "DOCKED", "attached": "DOCKED", "refused_outside_cone": "REFUSED_CONE"}.get(r, "FAILED")
        return self._set_last(self._log("dock", code, raw=r, berth=berth))

    def undock(self) -> str:
        if not self.layer.docked:
            return self._set_last(self._log("undock", "REFUSED_STATE"))
        self._motion_call()
        r = self._run(PR.Undock("standard"), 180.0, "undock")
        code = "TIME_UP" if r == "TIME_UP" else ("UNDOCKED" if r == "undocked" else "FAILED")
        return self._set_last(self._log("undock", code, raw=r))

    # ---------------------------------------------------------------- time
    def step(self, dt) -> str:
        if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            return "INVALID_ARGS"
        left = self._left()
        if left <= 0:
            return "TIME_UP"
        self.rig.run(min(float(dt), left))
        return "OK"
