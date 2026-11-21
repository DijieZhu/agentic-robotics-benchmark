"""capture(bar_pos, bar_axis, palm_dir) of the task firmware (robot_api.md, capture).

Sequence.  The hint (bar centre and axis in the body frame at the call) is mapped into the estimate frame with the
estimate at the call; the robot holds.  The bar detector (``bars.BarDetector``, firmware.capture.detector_hz) measures
every visible bar; in any cycle, a bar whose last firmware.capture.lock_mean_of measurements (all taken after the call)
average within firmware.capture.gate of the hint locks, the nearest in the gate's normalised metric winning.  Nothing
within firmware.capture.no_bar_s: NO_BAR (the firmware does not search).  After the lock the localiser switches to the
rail-relative mode; the robot aligns with the bar firmware.capture.align_out_m outside the fingertip line to within
firmware.capture.align_tol_m laterally (current mode, align speed), switches to firmware.capture.approach (FINE, 0.02
m/s) and goes in without turning; the tendon is released when the tracked bar reaches the closing depth of its kind
(firmware.capture.close_depth_mm: item bar / map handrail; the program cannot choose).  From the close the robot holds
where it stopped in the inertial sense (``on_close``: the task firmware's inertial hold -- the map estimate's wander
is not chased, so a robot clamped to a bar and a free one hold alike) and the action returns DONE firmware
close_settle_s later (the closing transient stays inside the blocking call); DONE says the gripper closed, nothing
about holding; the holding interlock engages (spacekit FirmwareLayer).
palm_dir (body frame) picks which of the two cage rolls 180 deg apart is used: the palm +z (along the bar) toward
palm_dir; None = the roll that needs the smaller turn from the attitude at the lock.
Ends: LOST_BAR -- the locked bar unseen for firmware lost_bar_s before the alignment (after it the bar is in the
camera's blind zone and the approach is dead reckoning); STALLED -- after the lock, the fingertips have not come
closer to the closing point by firmware progress_m for firmware.capture.stalled_s; TIMEOUT -- not closed
firmware.capture.timeout_s after the call.  The final approach is not zone-checked.
"""
from __future__ import annotations

import math

import numpy as np

from spacekit.primitives import Action, GraspFeature, pose_for_palm
from spacekit.rotations import quat_angle, quat_to_mat

from .bars import line_angle_deg, unit

FINGERTIP_X = -0.115            # palm-frame x of the fingertip line (spacekit GraspFeature doc)
CAGE_X = (-0.135, -0.030)       # true cage check: palm x, |y| <= 0.030, |z - z_plane| <= 0.030
CAGE_Y = 0.030
CAGE_Z = 0.030


class _EstGrasp(GraspFeature):
    """GraspFeature whose capture-box test uses the firmware's estimate (rail-relative pose + arm encoders) and the
    tracked bar; ``fine_last``: switch to the approach mode once aligned; ``zsign``: the cage roll (palm z = zsign *
    bar axis, fixed at the lock)."""

    fine_last = False
    _switching = False
    zsign = 1.0
    att_tol_deg = None
    approach_mode = "fine"
    approach_speed = 0.02

    def _target(self, fw, extra: float = 0.0, lead: float = 0.0, depth: float | None = None):
        depth = self.depth if depth is None else depth
        c, a, n, v = self.feature(fw)
        c = np.asarray(c, float) + np.asarray(v, float) * lead
        a = unit(a)
        n = unit(np.asarray(n, float) - (np.asarray(n, float) @ a) * a)
        z = self.zsign * a
        x = n
        y = np.cross(z, x)
        Rp = np.column_stack([x, y, z])
        pp = c - Rp @ np.array([depth, 0.0, fw.z_plane]) + n * extra
        return pose_for_palm(fw.T_body_palm_deployed, pp, Rp)

    def tick(self, fw, t):
        if self._switching:                      # hold while the executive changes the fan speed level
            if fw.rig.firmware.state == "preparing":
                return None if t - self.t0 <= self.timeout else "timeout"
            self._switching = False
            self.v = min(self.v, self.approach_speed)
            self.t_ret = -1.0
            self.detail["fine_ready_t"] = t - self.t0
        was = self._aligned
        r = super().tick(fw, t)
        if (r is None and self._aligned and not was and self.att_tol_deg is not None and self.state == "approach"
                and t - self.t_app <= self.align_timeout):
            # aligned in position but not yet at the chosen cage roll: keep turning (full-attitude re-targets)
            _, q_t = self._target(fw, 0.0, 0.0, self.align_depth)
            err = math.degrees(quat_angle(fw.rig.est.quat, q_t))
            if err > self.att_tol_deg:
                self._aligned = False
                self.t_ret = -1.0
                self.detail["align_att_wait"] = self.detail.get("align_att_wait", 0) + 1
        if r is None and self.fine_last and not was and self._aligned and fw.rig.firmware.mode.name != self.approach_mode:
            fw.rig.hold()
            fw.rig.set_mode(self.approach_mode)
            self._switching = True
            self.detail["fine_from_t"] = t - self.t0
        return r

    def bar_in_palm_est(self, fw):
        c, a, n, v = self.feature(fw)
        b = fw.rig.adapter.body
        d = fw.data
        Rb = d.xmat[b].reshape(3, 3)
        pb = Rb.T @ (d.xpos[fw.palm] - d.xpos[b])           # palm in body from the arm encoders
        Rpb = Rb.T @ d.xmat[fw.palm].reshape(3, 3)
        est = fw.rig.est
        Re = quat_to_mat(est.quat)
        pr = np.asarray(est.pos) + Re @ pb
        Rr = Re @ Rpb
        return Rr.T @ (np.asarray(c, float) - pr)

    def _in_box(self, fw):
        vloc = self.bar_in_palm_est(fw)
        return (self.close_x <= vloc[0] <= -0.035) and abs(vloc[1]) <= 0.025 and abs(vloc[2] - fw.z_plane) < self.box_z_tol, vloc


def true_cage(fw, bar) -> tuple:
    """Truth check (for logs, tests and the bench, never for the firmware): the bar centre inside the closed cage."""
    c, _, _ = bar.pose(fw.data)
    pr, Rr = fw.palm_frame()
    v = Rr.T @ (c - pr)
    return bool(CAGE_X[0] <= v[0] <= CAGE_X[1] and abs(v[1]) <= CAGE_Y and abs(v[2] - fw.z_plane) <= CAGE_Z), v


class Capture(Action):
    """The capture action (module doc).  det: bars.BarDetector; rail: localisation.RailLocalizer; spec: FirmwareSpec;
    hint_c_body / hint_a_body: the hint (body frame at the call); palm_dir_body: None or a body-frame direction."""

    name = "capture"

    def __init__(self, det, rail, spec, hint_c_body, hint_a_body, palm_dir_body=None, label: str = "", on_close=None):
        self.det, self.rail, self.spec = det, rail, spec
        self.on_close = on_close
        self.hc, self.ha = np.asarray(hint_c_body, float), unit(hint_a_body)
        self.palm_dir_b = None if palm_dir_body is None else unit(palm_dir_body)
        self.label = label
        self.gate_pos, self.gate_ang = spec.gate_pos_m, spec.gate_axis_deg
        self.detail = dict(gate=dict(position_m=self.gate_pos, axis_deg=self.gate_ang))
        self.result_code = None

    # ---------------------------------------------------------------- helpers
    def _est(self, fw):
        e = fw.rig.est if fw.rig.est is not None else fw.rig.estimate()
        return np.asarray(e.pos, float), np.asarray(e.quat, float)

    def start(self, fw):
        self.t0 = fw.t
        pe, qe = self._est(fw)
        Re = quat_to_mat(qe)
        self.hint_w = pe + Re @ self.hc
        self.hint_aw = Re @ self.ha
        self.palm_dir_w = None if self.palm_dir_b is None else Re @ self.palm_dir_b
        self.det.reset()
        self.state = "detect"
        self.next_det = fw.t
        self.locked = None
        self.gf = None
        self.e_best = None
        self.t_progress = None
        fw.rig.hold()

    def _feature(self, fw):
        c, a, n = self.det.track(self.locked)
        return c, a, unit(np.asarray(n, float) - (np.asarray(n, float) @ a) * a), np.zeros(3)

    def _gate_check(self, meas_bars):
        best = None
        k = self.spec.lock_mean_of
        for i in meas_bars:
            tr = self.det.tracks[i]
            if len(tr.c) < k:
                continue
            c, a = tr.mean_last(k)
            dp = float(np.linalg.norm(c - self.hint_w))
            da = line_angle_deg(a, self.hint_aw)
            if dp <= self.gate_pos and da <= self.gate_ang:
                score = (dp / self.gate_pos) ** 2 + (da / self.gate_ang) ** 2
                if best is None or score < best[0]:
                    best = (score, i, dp, da)
        return best

    def _choose_roll(self, fw) -> float:
        """+1 / -1: palm z = sign * tracked bar axis."""
        c, a, n, _ = self._feature(fw)
        if self.palm_dir_w is not None:
            return 1.0 if float(a @ self.palm_dir_w) >= 0 else -1.0
        _, qe = self._est(fw)
        best = None
        for s in (1.0, -1.0):
            self.gf.zsign = s
            _, q = self.gf._target(fw, 0.0, 0.0, self.gf.align_depth)
            ang = quat_angle(qe, q)
            if best is None or ang < best[0]:
                best = (ang, s)
        return best[1]

    def _start_approach(self, fw):
        bar = self.det.bars[self.locked]
        if self.rail is not None:
            self.rail.engage(fw.model, fw.data, fw.rig.adapter.body)
        sp = self.spec
        depth = -(sp.close_depth_rail_m if bar.depth_rule == "rail" else sp.close_depth_item_m)
        self.close_x = depth - 0.006            # release the tendon 6 mm before the closing depth (104 / 98 mm)
        v_align = min(sp.align_speed_m_s, float(fw.rig.firmware.mode.hard_limit_vel))   # called in a slower mode
        self.gf = _EstGrasp(self._feature, standoff=None, approach_speed=v_align, depth=depth, close_x=self.close_x,
                            retarget_s=2.0, timeout_s=sp.capture_timeout_s, keep_attitude=True,
                            align_depth=FINGERTIP_X - sp.align_out_m, align_tol=sp.align_tol_m, align_attitude=True,
                            align_timeout_s=40.0, label=self.label)
        self.gf.approach_mode, self.gf.approach_speed = sp.approach_mode, sp.approach_speed_m_s
        self.gf.att_tol_deg = sp.align_att_tol_deg
        self.gf.fine_last = True
        self.gf.zsign = self._choose_roll(fw)
        self.gf.start(fw)
        self.gf.t0 = self.t0                     # the TIMEOUT clock runs from the call
        self.detail["roll_sign"] = self.gf.zsign
        self.state = "approach"
        self.e_best = None
        self._progress(fw, fw.t)                 # the STALLED clock starts at the lock
        self.detail["e_at_lock_m"] = self.e_best

    def _progress(self, fw, t) -> float:
        """Distance of the tracked bar to the closing point in the palm frame (estimate); resets the STALLED clock when
        it improves by at least firmware progress_m."""
        v = self.gf.bar_in_palm_est(fw)
        e = float(np.linalg.norm(v - np.array([self.close_x, 0.0, fw.z_plane])))
        if self.e_best is None or e < self.e_best - self.spec.progress_m:
            self.e_best = e
            self.t_progress = t
        return e

    def _end(self, fw, code: str, why: str):
        self.detail["end"] = why
        self.result_code = code
        # the rail-relative estimate ends with the action (as fly_rel's fiducial-relative one, the 'action' scope);
        # then hold at the map estimate: no jump of the hold point, no push against what the fingers closed on, and
        # the next motion is planned from the estimate pose() reports
        if self.rail is not None and getattr(self.rail, "rail", False):
            self.rail.release_rail()
            self.detail["rail_released"] = True
        fw.rig.hold()                            # (the task firmware re-latches the hold point when the source changes)
        return code

    # ---------------------------------------------------------------- tick
    def tick(self, fw, t):
        sp = self.spec
        if self.state == "settle":                # closed: brake (compliant) until the settle time is over
            if t - self.t_close >= sp.close_settle_s - 1e-9:
                return self._end(fw, "DONE", "done")
            return None
        if t - self.t0 > sp.capture_timeout_s:
            fw.rig.hold()
            return self._end(fw, "TIMEOUT", "timeout")
        if t >= self.next_det - 1e-9:
            self.next_det += 1.0 / sp.detector_hz
            pe, qe = self._est(fw)
            meas = self.det.update(fw.data, t, pe, qe)
            self.detail["n_det_cycles"] = self.detail.get("n_det_cycles", 0) + 1
            if self.state == "detect":
                vis = [m_["bar"] for m_ in meas]
                self.detail.setdefault("seen", set()).update(vis)
                best = self._gate_check(vis)
                if best is not None:
                    _, i, dp, da = best
                    self.locked = i
                    self.detail.update(locked=self.det.bars[i].name, locked_kind=self.det.bars[i].kind, lock_t=t - self.t0,
                                       lock_dp_m=dp, lock_da_deg=da)
                    self._start_approach(fw)
                    return None
        if self.state == "detect":
            if t - self.t0 > sp.no_bar_s:
                self.detail["seen"] = sorted(self.det.bars[i].name for i in self.detail.get("seen", set()))
                return self._end(fw, "NO_BAR", "no_bar")
            return None
        tr = self.det.tracks[self.locked]
        if not self.gf._aligned and t - tr.t_last > sp.lost_bar_s:
            fw.rig.hold()
            return self._end(fw, "LOST_BAR", "lost_bar")
        if self.gf.state == "approach":
            self._progress(fw, t)
            if t - self.t_progress > sp.stalled_s:
                fw.rig.hold()
                self.detail["stalled_after_s"] = t - self.t_progress
                return self._end(fw, "STALLED", "stalled")
        r = self.gf.tick(fw, t)
        if r is None:
            return None
        self.detail["grasp"] = self.gf.info()
        self.detail["last_seen_before_close_s"] = t - tr.t_last
        if r in ("closed", "missed"):
            fw.rig.hold()                         # brake where the fingers closed
            if self.on_close is not None:
                self.on_close()                   # ... and hold inertially until the next motion call
            self.state = "settle"
            self.t_close = t
            self.detail["closed_t"] = t - self.t0
            return None
        fw.rig.hold()
        return self._end(fw, "TIMEOUT", r)

    def info(self):
        d = dict(self.detail)
        if isinstance(d.get("seen"), set):
            d["seen"] = sorted(self.det.bars[i].name for i in d["seen"])
        return d
