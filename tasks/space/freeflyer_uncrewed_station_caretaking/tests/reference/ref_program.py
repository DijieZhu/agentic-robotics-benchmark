"""ref_program.py -- the token-privileged reference program of freeflyer_uncrewed_station_caretaking (the
privileged reference).

It moves only through the public calls (ref_robot.Robot on the cell's own call boundary: every move is a validated
fly_to / fly_rel of the task firmware, no zone check is ever bypassed) and decides on what the operator sees: the log
and the configuration record (ref_log: the stepped unit, the onset, the route), the map (ref_nav), the station
telemetry (the return pressure for the fan stop and the removal), its own pose estimate and covariance (the lamp).
The five privileged reads come from ref_truth.Truth only (P1 true pose, P2 grasp features, P3 gripper,
P4 held item attitude, P5 on the face after a placement).

Fixed order: read the log and the configuration record -> one frame of every camera -> the night
modules' lights -> undock -> L2 -> H2 -> L1: survey every inlet (forward imaging frame; at the blocked inlet the
report first), the module's pouches onto its unit's pouch pads, the blocker after them (S: capture the carry strap,
fan off, return pressure < released_below_dp_pa, back off, fan on at once; P: capture the handle loop with the palm
+z toward the loop's end, peel along the inlet normal) onto the module's holding screen -> arm stowed -> L2 -> dock.

The five deposit rules: (1) the deposit puts the ITEM's centre on the face centre (the
held item's pose in the palm, P4), not the bar; (2) after the release the open hand first slides along the bar axis
(pouch handles: the fingers leave the handle) and then backs off; (3) after every placement the item is checked (P5)
and placed again, at most reference.retries.placement times, only when it is still between the fingers or stuck on
the face (a floating item is not chased; K counts the single attempt); (4) the gripper closes only when the finger
cage is empty (P3); (5) a refused pre-grasp or pre-deposit pose is replaced by another one (distance, cage roll,
plate side, pad), the item is never let go in place.  Simplified for spacekit 0.4.0: no periodic re-centring, no own
time-outs (the firmware's TIMEOUT); moves retried <= reference.retries.move; capture / removal / placement <= their
retries; the removal probe moves > reference.probe_backoff_min_m.
"""
from __future__ import annotations

import math

import numpy as np

import ref_geom as RG
import ref_log as RL
import ref_nav as RN
from ref_robot import CellError

MODULE_ORDER = RN.MODULE_ORDER
INLET_ORDER = dict(l2=("ri_l2_3", "ri_l2_2", "ri_l2_1"), h2=("ri_h2_2", "ri_h2_1"), l1=("ri_l1_1", "ri_l1_2", "ri_l1_3"))
CAMERAS = ("FIC", "FWC", "AWC", "FDC", "ADC")
TOL_CODES = ("TOLERANCE_PATH", "TOLERANCE_END", "TOLERANCE_ATT")
# the reference's own recipe values that the runtime site_spec does not carry (all [G] unless noted):
DEFAULTS = dict(
    place_method="press",            # 'press' (reference.place_press) | 'flyby' (reference.place_flyby)
    pregrasp_dists=(0.40, 0.45, 0.35, 0.50, 0.55),   # world.pregrasp.distance_m sampled; ~0.40-0.45 m puts the
                                     # aft depth camera 0.52-0.57 m from the bar (reference.hint_range_m)
    pouch_pull_m=0.45,               # direct pull of a pouch (18 / 18)
    peel_pull_m=0.50,                # P route: the loop pulled along the inlet normal
    pull_s=40.0,                     # a held pull re-issued for at most this long
    s_pull_s=16.0,                   # S route: the pull after the stop
    fan_wait_s=20.0,                 # S route: at most this long for the return pressure to fall
    transit_out_m=0.60,              # deposit transit point out along the face normal
    pre_out_m=0.30,                  # fly_rel pre-deposit point
    clear_m=dict(pouch=0.10, sheet=0.0),   # rule 2: slide of the open hand along the bar axis (pouch 0.10 m);
                                     # a sheet's bar runs along the chain and the screen recipe backs off at once
    clear_speed=0.03,
    pouch_insert_speed=0.03,         # reference.place_press speed; 0.06 left the pouch short of the pad
    press_settle=dict(pouch=True, sheet=False),   # a second fly_rel to the press goal after the insert (pouches):
                                     # fly_rel ends inside its end tolerance and the hold after it is on the map, so the
                                     # pouch can stop short of the pad; a sheet is pressed
                                     # once (one insert, the dwell)
    reseat_backoff_m=0.10,           # a press that did not seat: back off this far (on the plate) and press again -- a
                                     # sheet partly on the screen stays close instead of being pulled out of the cage
    pouch_align="clamped",           # ref_geom.plan_pouch
    pouch_clamp_deg=35.0,
    square_max_deg=35.0,             # the held item's top normal off the face normal above this: wait before the insert
    square_wait_n=2,                 # ... at most this many 2 s waits
    removed_fraction=0.6,            # a removal counts when the robot got this share of the pull out, still holding
    peel_mode="FINE",                # the P route pulls in FINE: the strongest sustained pull (FINE 0.231 N,
                                     # AGILE 0.204, STANDARD 0.158; the generator's loop condition is 0.22 N, 40 s)
    peel_s=60.0,                     # the peel pull re-issued while it makes progress, at most this long
    max_presses=5,                   # presses per item over every face / attitude
    deposit="aligned",               # 'aligned': a closed loop on the held item's deviation from the
                                     # face plane (P4) before every press, never a press above align.dev_press; 'legacy'
                                     # = a square wait and a re-press from 0.10 m
    align=dict(dev_go=dict(sheet=22.0, sheet_fan_stop=60.0, pouch=20.0),     # go for the insert when the item's face is this close to the
                                     # face plane [deg]
               dev_press=dict(sheet=55.0, sheet_fan_stop=60.0, pouch=30.0),   # press-flattening: sheets pressed up to these
               turn_max=dict(sheet=40.0, pouch=35.0),   # the palm turned about the bar by the item's in-hand turn,
                                     # clamped (the body clears the rack face up to ~45 deg; 35 deg for pouches)
               tries=3, settle_s=2.0,   # reads (and turns) at the pre-deposit point, the wait before each read
               turn_mode="HUSH",     # the turn in HUSH (1 deg/s): the item follows the slow palm [V]
               near_m=dict(sheet=0.08, pouch=0.06),     # insert leg 1 stops this far off the press point: the last
                                     # check before the press
               backoffs=2,           # leg 1 found the item turned again: back to the pre-deposit point and align again
               max_turns=4,          # palm turns per item (each 20-50 s in HUSH)
               mode="full",          # 'full': the palm turned by the smallest rotation that lays the item's face in the
                                     # face plane (twist about the bar and tilt of the bar in the cage); 'twist': about
                                     # the bar only
               touch_fraction=0.05, more_s=3.0,    # press check: touching the face -> keep pressing more_s, release
               leg1_speed=dict(sheet=None, pouch=None),  # leg 1 speed (None: the press speed)
               where="pre", transit_settle_s=3.0),   # where the palm turns: out at the transit point (0.6 m beyond
                                     # the pre-deposit point), where the cabin air is the background: a slow turn at the
                                     # pre-deposit point (0.14 m/s rack-face sweep) let strap-held sheets swing over the
                                     # bar [V]; 'pre' = turn at the pre-deposit point
    sheet_z_order=dict(fan_stop=("near_b", "far_b", None), peel=("-b", "+b", None)),   # bar directions on the screen:
                                     # +-b keeps the screen's plates (+-b) on the overhang side, in view;
                                     # None = the bar kept in its in-plane direction.  P route -b first:
                                     # the chain hangs off the loop toward +b, downstream of the air that
                                     # sweeps the rack faces from the overhead supply to the deck-side returns
                                     # (station handbook: top supply, side return; 0.16-0.19 m/s along +b within 0.2 m
                                     # of every screen) instead of upstream into it
    sheet_ok_fraction=0.50,          # own check after a placement (P5): a sheet fixed = sealed >= half its footprint
    pouch_ok_fraction=0.80,          # a pouch placed = >= 80 % of its own footprint sealed on the pad; a pouch
                                     # that holds on the pad cannot be pulled off again (0.16-0.23 N against ~1 N)
    regrasp_max_fraction=0.25,       # an item stuck on the face is grasped again only when it seals less than this
                                     # share of its footprint (more holds harder than the robot pulls)
    lamp_sigma_m=0.06,               # forward lamp on when the reported position sigma exceeds this (lit maps: 0.03)
    return_reserve_s=dict(l2=200.0, h2=300.0, l1=420.0),   # time kept for the way home and the dock (160 / 260 / 380)
    op_cost_s=dict(pouch=190.0, blocker=280.0),
    battery_reserve_pct=3.0,         # kept above the charge the way home needs
    swing=dict(sheet_peel=False, sheet_fan_stop=False, pouch=False, min_deg=12.0, mode="HUSH"),
                                     # before the transit with a held item: the rotation that brings the grasped bar to
                                     # its deposit direction made in place in HUSH (1 deg/s), the rest of the turn (about
                                     # the bar) in the transit -- a fast swing of the bar's direction flings a loop-held
                                     # sheet's 0.3 m chain round and wedges the bar askew in the cage (P route: bar tilt
                                     # at the pre-deposit point 0.06-0.6 of the cage axis in the failed placements)
                                     # [V]
    survey="items",                  # 'items': survey stops only at the inlets that hold an item (the
                                     # blocked inlet's frame is the report's evidence; P2 says where the items are), and
                                     # no visit to a module without an item; 'all' = a stop at every inlet of every
                                     # module (the time goes to the closed-loop placements, the hidden-set rule margins)
    battery_wh=196.0,                # the 2 h basis at 98 W [G]
    power_w=98.0,
    dock_berth=1,
    arm_overlap=True,                # the arm's own motion overlapped with a move instead of waited
                                     # for: after the module's last placement (PLACED, the cage empty, the hand free)
                                     # the stow starts when the move to the centreline starts, so the corridor waits
                                     # only for the rest of it; the module's first capture flies its centreline leg
                                     # while the deploy started at the module's arrival finishes, the arm is at the
                                     # grasp attitude before the leg toward the wall.  The zone rule is the same as
                                     # before: the firmware and the program's prediction take a moving arm as deployed
                                     # (FirmwareLayer.arm_deployed: the encoder or the goal off the stowed tilt).
)


class Abort(Exception):
    """The run has to end now (TIME_UP or the cell closed the run)."""


class Program:
    """The reference (module doc).  robot: ref_robot.Robot; truth: ref_truth.Truth; params: DEFAULTS overrides."""

    def __init__(self, robot, truth, params: dict | None = None):
        self.r, self.T = robot, truth
        self.P = dict(DEFAULTS, **(params or {}))
        V = truth.runtime_values()
        self.V = V
        self.retries = dict(V["reference.retries"])
        self.press = dict(V["reference.place_press"])
        self.flyby = dict(V["reference.place_flyby"])
        self.probe_min = float(V["reference.probe_backoff_min_m"])
        self.s_backoff = float(V["reference.s_tier_backoff_m"])
        self.dp_go = float(V["station.fan.released_below_dp_pa"])
        self.sim_limit = float(V["cell.budget.sim_time_s"])
        self.map = RN.StationMap(V)
        self.zones = RN.zone_predictor()
        self.geom = truth.palm_geometry()
        self.notes, self.ops = [], []
        self.cur = "l2"
        self.lamp = False
        self.reported = False
        self.frames = {}
        self.pads_used = {m: [] for m in MODULE_ORDER}
        self.presses = {}
        self._dep_plan = {}
        self._turns = {}
        self._flipped = {}
        self.t_end_ops = None
        self._last_in_module = False     # the op running now is its module's last planned op (arm_overlap)
        self._placed_now = False         # stow(): the placement that _after_place follows ended PLACED (arm_overlap)
        self._deploying = False          # an arm_deploy(wait=False) was started and the arm is not at grasp yet

    # ================================================================== small helpers
    def note(self, kind: str, **kw):
        kw.update(kind=kind, t=self.t())
        self.notes.append(kw)
        return kw

    def t(self) -> float:
        return float(self.r.status()["t"])

    def left(self) -> float:
        return self.sim_limit - self.t()

    def _act(self, res: dict) -> str:
        code = res.get("code") if isinstance(res, dict) else None
        if code == "TIME_UP":
            raise Abort("time up")
        return code

    def pose(self):
        e = self.r.pose()
        return np.asarray(e["pos"], float), np.asarray(e["quat"], float), e

    def sigma(self) -> float:
        e = self.r.pose()
        return math.sqrt(max(float(v) for v in e["cov_diag"][:3]))

    def wait(self, s: float):
        k = float(s)
        while k > 1e-6:
            dt = min(5.0, k)
            self._act(self.r.step(dt))
            k -= dt

    def arm_deployed(self) -> bool:
        st = self.r.status()
        return abs(float(st["arm_tilt_deg"]) - 180.0) > 5.0 or abs(float(st["arm_pan_deg"])) > 5.0

    def arm_at_grasp(self) -> bool:
        st = self.r.status()
        return abs(float(st["arm_tilt_deg"])) <= 2.5 and abs(float(st["arm_pan_deg"])) <= 2.5

    # ================================================================== motion
    def fly(self, p, q, label: str, ff: bool = False, max_speed=None, retries: int | None = None) -> str:
        """A validated fly_to, re-issued on a tolerance end / time-out (<= reference.retries.move); LOCALIZATION: the
        lamp and a short wait first."""
        n = int(self.retries["move"]) if retries is None else int(retries)
        code = None
        for k in range(n + 1):
            code = self._act(self.r.fly_to(p, q, face_forward=ff, max_speed=max_speed))
            if code == "SUCCESS" or code == "REFUSED_ZONE":
                break
            if code == "LOCALIZATION":
                self.lamp_check(force=True)
                self.wait(5.0)
        self.ops.append(dict(op="fly", label=label, code=code, t=self.t()))
        return code

    def lamp_check(self, force: bool = False):
        """The forward lamp on when the reported position sigma is above lamp_sigma_m (a dark module: the station's
        light circuit failed, or night)."""
        if self.lamp:
            return
        s = self.sigma()
        if force or s > float(self.P["lamp_sigma_m"]):
            self._act(self.r.lamp("forward", True))
            self.lamp = True
            self.note("lamp_on", sigma=s, module=self.cur)
            self.wait(3.0)

    def lamp_off(self):
        if self.lamp:
            self._act(self.r.lamp("forward", False))
            self.lamp = False
            self.note("lamp_off", module=self.cur)

    def exit_band(self, n_dir, label="exit_band", extra: float = 0.05) -> str:
        """After work next to a wall: straight out along n_dir to the nearest point the zone map holds free with the
        arm's current state (the band rule accepts a segment that leaves the band)."""
        arm = self.arm_deployed()
        p, q, _ = self.pose()
        n_dir = RG.unit(n_dir)
        for d in np.arange(0.10, 1.31, 0.05):
            g = p + n_dir * (float(d) + extra)
            if self.zones.point(g, arm) == "ok" and self.zones.segment(p, q, g, q, arm) == "ok":
                return self.fly(g, q, label)
        return "NO_FREE_POINT"

    def to_centreline(self, label="to_centreline") -> str:
        p, q, _ = self.pose()
        m = self.map.module_of_point(p)
        c = self.map.centreline_point(m, p)
        if float(np.linalg.norm(c - p)) < 0.10:
            return "SUCCESS"
        arm = self.arm_deployed()
        if self.zones.segment(p, q, c, q, arm) != "ok":
            self.exit_band(c - p, label=label + "_band")
            p, q, _ = self.pose()
        return self.fly(c, q, label)

    def stow_arm(self, wait=True):
        """arm_stow; the hand is freed first if the interlock is still on (nothing is carried through a hatch)."""
        if self.T.holding() or self.r.status().get("interlock"):
            self.release_if_holding(None, "stow_arm")
        self._deploying = False
        return self._act(self.r.arm_stow(wait=wait))

    def deploy_arm(self, wait=False) -> str:
        r = self._act(self.r.arm_deploy(wait=wait))
        if r == "REFUSED_ZONE":
            self.to_centreline("deploy_recentre")
            r = self._act(self.r.arm_deploy(wait=wait))
        self._deploying = (r == "STARTED")
        self.note("arm_deploy", code=r)
        return r

    def ensure_grasp_attitude(self) -> bool:
        if self.arm_at_grasp():
            self._deploying = False
            return True
        r = self._act(self.r.arm_deploy(wait=True))
        if r == "REFUSED_ZONE":
            return False
        ok = self.arm_at_grasp()
        if ok:
            self._deploying = False
        return ok

    def go_module(self, m: str):
        """Corridor legs to module m with the arm stowed (the 0.50 m arm band would refuse the hatches)."""
        if m == self.cur:
            return
        self.to_centreline("leave_" + self.cur)
        self.stow_arm(wait=True)
        self.lamp_off()
        for p, q in self.map.transit_legs(self.cur, m):
            code = self.fly(p, q, f"transit_{self.cur}_{m}", ff=True)
            if code == "REFUSED_ZONE":
                self.to_centreline("transit_recentre")
                self.fly(p, q, f"transit_{self.cur}_{m}_again", ff=False)
        self.cur = m
        self.lamp_check()

    # ================================================================== the run
    def run(self) -> dict:
        try:
            self._run()
        except Abort as e:
            self.note("abort", why=str(e))
        except CellError as e:
            self.note("cell_error", code=e.code, detail=e.detail)
        return dict(notes=self.notes, ops=self.ops, reads=dict(self.T.reads), refusals=dict(self.T.refusals),
                    diagnosis=getattr(self, "diag", None), frames=self.frames, params=dict(self.P, retries=self.retries))

    def _run(self):
        r = self.r
        M = r.mission()
        self.mission = M
        self.diag = RL.diagnose(M)
        self.note("diagnosis", **{k: v for k, v in self.diag.items() if k != "lights"})
        # one frame of every camera (validation exercises every sensor path)
        for cam in CAMERAS:
            f = r.camera(cam)
            self.frames.setdefault("first", {})[cam] = f.get("frame_id")
        # the night modules' lights (configuration record): request the general lighting
        for mod, st in sorted((M.get("config_record") or {}).get("lights", {}).items()):
            if st != "general":
                res = r.station_request("lights", mod, "general")
                self.note("lights", module=mod, code=res.get("code"))
        self._act(r.undock())
        qL2 = RN.look_quat((0.0, -1.0, 0.0))
        if self.fly(RN.HOME0, qL2, "egress") == "REFUSED_ZONE":
            self.exit_band(RN.rot_of(self.map.dock["quat_wxyz"])[:, 0], label="egress_band")
            self.fly(RN.HOME0, qL2, "egress_again")
        self.cur = "l2"
        self.lamp_check()
        self.items = self.locate_items()
        self.note("items", **{k: v for k, v in self.items.items() if k != "features"})
        for m in MODULE_ORDER:
            if self.P.get("survey") == "items" and not self._module_has_items(m):
                self.note("skip_module", module=m, why="no item")
                continue
            self.go_module(m)
            self.deploy_arm(wait=False)
            self.survey(m)
            pids = list(self.items["pouches_by_module"].get(m, []))
            has_blocker = self.items.get("blocker_module") == m
            for k, pid in enumerate(pids):
                if not self.time_ok("pouch"):
                    self.note("skip_pouch", item=pid, why="time")
                    continue
                self._last_in_module = (k == len(pids) - 1) and not has_blocker
                self.pouch_op(pid, m)
                self._last_in_module = False
            if has_blocker and self.time_ok("blocker"):
                self._last_in_module = True
                self.blocker_op(m)
                self._last_in_module = False
            self.to_centreline("module_done_" + m)
        self.home_and_dock()

    def time_ok(self, op: str) -> bool:
        res = float(self.P["return_reserve_s"][self.cur]) + float(self.P["op_cost_s"][op])
        if self.left() < res:
            return False
        st = self.r.status()
        need = float(self.P["battery_reserve_pct"]) + 15.0 + 100.0 * float(self.P["power_w"]) * res / 3600.0 / float(self.P["battery_wh"])
        return float(st["battery_pct"]) >= need

    # ================================================================== items (P2)
    def locate_items(self) -> dict:
        """Which item lies on which inlet (P2: the grasp features' true poses, what the survey frames show)."""
        feats = self.T.features()
        on = {}
        for f in feats:
            pose = self.T.feature(f["name"])
            if pose is None:
                continue
            iid = self.map.nearest_inlet(pose[0], max_m=0.30)
            if iid is not None:
                on.setdefault(f["item"], iid)
        pouches = {}
        for item, iid in on.items():
            if item == "blocker":
                continue
            m = self.map.inlet(iid)["module"]
            pouches.setdefault(m, []).append((INLET_ORDER[m].index(iid), item))
        out = dict(on=on, pouches_by_module={m: [x for _, x in sorted(v)] for m, v in pouches.items()},
                   blocker_inlet=on.get("blocker"), blocker_module=(self.map.inlet(on["blocker"])["module"] if "blocker" in on else None),
                   features=[f["name"] for f in feats])
        return out

    # ================================================================== survey and report
    def _module_has_items(self, m: str) -> bool:
        return self.items.get("blocker_module") == m or bool(self.items["pouches_by_module"].get(m))

    def survey(self, m: str):
        with_items = set(self.items.get("on", {}).values())
        for iid in INLET_ORDER[m]:
            if self.P.get("survey") == "items" and iid not in with_items:
                continue
            p, q, _ = self.map.survey_pose(iid)
            code = self.fly(p, q, f"survey_{iid}")
            self.lamp_check()
            self.wait(1.0)
            fr = self.r.camera("FIC")
            self.frames[iid] = fr.get("frame_id")
            self.note("survey", inlet=iid, code=code, frame=fr.get("frame_id"))
            if iid == self.items.get("blocker_inlet") and not self.reported:
                self.report(iid, fr.get("frame_id"))

    def report(self, iid: str, frame_id):
        ref = self.T.register_ref("blocker")
        onset = self.diag.get("onset_min")
        if onset is None or ref is None or frame_id is None:
            self.note("report_skipped", inlet=iid, onset=onset, ref=ref)
            return
        res = self.r.report(self.map.labels[iid], ref, float(onset), int(frame_id))
        self.reported = res.get("code") == "RECORDED"
        self.note("report", inlet=iid, code=res.get("code"), item_ref=ref, onset=onset, frame=frame_id)

    # ================================================================== capture (P1 + P2)
    def capture_feature(self, name: str, palm_dir_fn=None, label: str = "") -> str:
        """Fly to an accepted pre-grasp pose of the feature (from its module's centreline; a refused one is replaced:
        distance, cage roll) and capture it with the hint from the true poses (P1, P2).  Returns the capture code or
        the reason it was not tried."""
        feat = self.T.feature(name)
        if feat is None:
            return "NOT_ATTACHED"
        c, a, n = feat
        m = self.map.module_of_point(c)
        cands = RG.pregrasp_candidates(self.geom, c, a, n, self.P["pregrasp_dists"])
        pd0 = palm_dir_fn(c, a, n) if palm_dir_fn else None
        if pd0 is not None:
            # the cage roll the capture will take (palm +z toward palm_dir) already at the pre-grasp pose: no half
            # turn about the bar after the lock, next to the wall
            cands = [cd for cd in cands if float(cd["palm_z"] @ pd0) > 0] + [cd for cd in cands if float(cd["palm_z"] @ pd0) <= 0]
        # arm_overlap: a deploy started at the module's arrival that is still running finishes during the centreline
        # leg (the zone rule takes a moving arm as deployed); the arm is at the grasp attitude before the leg to the wall
        defer = bool(self.P.get("arm_overlap")) and self._deploying and not self.arm_at_grasp()
        if not defer and not self.ensure_grasp_attitude():
            self.to_centreline(label + "_recentre_arm")
            if not self.ensure_grasp_attitude():
                return "ARM_REFUSED"
        p0, q0, _ = self.pose()
        ranked = []
        for k, cd in enumerate(cands):
            cl = self.map.centreline_point(m, c)
            s1 = self.zones.segment(p0, q0, cl, cd["quat"], True)
            s2 = self.zones.segment(cl, cd["quat"], cd["pos"], cd["quat"], True)
            ranked.append((0 if (s1 == "ok" and s2 == "ok") else 1, k, cd, cl))
        ranked.sort(key=lambda x: (x[0], x[1]))
        tried = []
        for pri, k, cd, cl in ranked[:4]:
            p, q, _ = self.pose()
            if float(np.linalg.norm(p - cl)) > 0.15 or RN.angle_deg(q, cd["quat"]) > 10.0:
                c1 = self.fly(cl, cd["quat"], label + "_pregrasp_centreline")
                if c1 == "REFUSED_ZONE":
                    self.to_centreline(label + "_recentre")
                    c1 = self.fly(cl, cd["quat"], label + "_pregrasp_centreline2")
            if defer:
                defer = False
                self.note("arm_deploy_overlap", feature=name, at_grasp=self.arm_at_grasp())
                if not self.ensure_grasp_attitude():
                    self.to_centreline(label + "_recentre_arm")
                    if not self.ensure_grasp_attitude():
                        return "ARM_REFUSED"
            c2 = self.fly(cd["pos"], cd["quat"], label + "_pregrasp")
            tried.append(dict(dist=cd["dist"], zsign=cd["zsign"], predicted=pri, code=c2))
            if c2 == "SUCCESS" or c2 in TOL_CODES:
                break
        else:
            self.note("pregrasp_refused", feature=name, tried=tried)
            return "NO_PREGRASP"
        if not self.ensure_grasp_attitude():
            return "ARM_REFUSED"
        self.wait(2.0)
        feat = self.T.feature(name)
        if feat is None:
            return "NOT_ATTACHED"
        c, a, n = feat
        pos, R = self.T.robot_pose()
        pdw = palm_dir_fn(c, a, n) if palm_dir_fn else None
        cb, ab, pdb = RG.hint(pos, R, c, a, pdw)
        res = self.r.capture(cb, ab, pdb)
        code = self._act(res)
        st = self.r.status()
        if st.get("flight_mode") != "STANDARD":
            self._act(self.r.set_mode("STANDARD"))
        self.ops.append(dict(op="capture", feature=name, code=code, pregrasp=tried, t=self.t()))
        return code

    def pull(self, n_dir, dist: float, T: float, label: str, progress: bool = False) -> float:
        """Back off along n_dir: fly_to a fixed goal, re-issued while the firmware ends it on a tolerance (the item
        still holds), at most T s; progress: stop early after 40 s when the last try gained < 1 cm.
        Returns the distance the estimate moved."""
        p0, q0, _ = self.pose()
        goal = p0 + RG.unit(n_dir) * float(dist)
        t0 = self.t()
        codes = []
        last = 0.0
        while True:
            code = self._act(self.r.fly_to(goal, q0, face_forward=False))
            codes.append(code)
            if code == "SUCCESS" or code not in TOL_CODES + ("TIMEOUT",) or self.t() - t0 >= T:
                break
            now = float(np.linalg.norm(self.pose()[0] - p0))
            if progress and self.t() - t0 >= 40.0 and now - last < 0.01:
                break
            last = now
        p1, _, _ = self.pose()
        moved = float(np.linalg.norm(p1 - p0))
        self.ops.append(dict(op="pull", label=label, codes=codes, moved=moved, t=self.t()))
        return moved

    def release_hand(self, n_dir, label: str):
        """Open and back away (an item that came along stays behind: never carried off)."""
        self._act(self.r.gripper(True))
        self.wait(1.0)
        self.exit_band(n_dir, label=label)

    def close_hand_if_empty(self, n_dir) -> bool:
        """Rule 4: close the gripper only when no bar is between the fingers (P3); otherwise back off further."""
        for k in range(3):
            if not self.T.bar_between_fingers():
                self._act(self.r.gripper(False))
                return True
            self.note("cage_not_empty", attempt=k)
            p, q, _ = self.pose()
            self.fly(p + RG.unit(n_dir) * 0.15, q, "cage_clear", max_speed=0.05)
        return False

    # ================================================================== pouches
    def pouch_op(self, pid: str, m: str):
        name = f"{pid}:handle"
        iid = self.items["on"].get(pid)
        _, n_in, _ = self.map.inlet_frame(iid)
        removed = False
        for k in range(int(self.retries["capture"])):
            code = self.capture_feature(name, label=f"{pid}_{k}")
            if code in ("NOT_ATTACHED", "NO_PREGRASP", "ARM_REFUSED"):
                break
            if code != "DONE":
                self.exit_band(n_in, label=f"{pid}_capture_backoff")
                continue
            moved = self.pull(n_in, self.P["pouch_pull_m"], self.P["pull_s"], f"pull_{pid}")
            self.wait(2.0)
            if self.T.holding(pid) and moved >= max(self.probe_min, float(self.P["removed_fraction"]) * float(self.P["pouch_pull_m"])):
                removed = True
                break
            self.release_hand(n_in, f"{pid}_release_failed")
            self.close_hand_if_empty(n_in)
        self.ops.append(dict(op="pouch_removal", item=pid, removed=removed, t=self.t()))
        if not removed:
            return False
        self.exit_band(n_in, label=f"{pid}_out")
        self.to_centreline(f"{pid}_centreline")
        pads = [f for f in ("P1", "P2") if f not in self.pads_used[m]]
        ok = self.stow(pid, m, pads, kind="pouch")
        return ok

    # ================================================================== the blocker
    def blocker_op(self, m: str):
        iid = self.items["blocker_inlet"]
        if not self.reported:
            p, q, _ = self.map.survey_pose(iid)
            self.fly(p, q, "survey_again")
            self.wait(1.0)
            fr = self.r.camera("FIC")
            self.report(iid, fr.get("frame_id"))
        _, n_in, _ = self.map.inlet_frame(iid)
        route = self.diag.get("route") or "peel"
        removed = self.fan_stop_removal(m, n_in) if route == "fan_stop" else self.peel_removal(m, n_in)
        self.ops.append(dict(op="blocker_removal", route=route, removed=removed, t=self.t()))
        if not removed:
            if self.T.holding() or self.r.status().get("interlock"):
                self.release_hand(n_in, "blocker_release_failed")
                self.close_hand_if_empty(n_in)
            return False
        self.exit_band(n_in, label="blocker_out")
        self.to_centreline("blocker_centreline")
        return self.stow("blocker", m, ["S"], kind="sheet")

    def _pd_away(self, c, a, n):
        """P route: the palm +z toward the loop's end of the sheet, away from the sheet's centre: the
        chain hangs off the hand on the far side (P2: the strap's pose gives the sheet's middle)."""
        strap = self.T.feature("blocker:strap")
        mid = strap[0] if strap is not None else c
        d = np.asarray(c, float) - np.asarray(mid, float)
        d = d - (d @ n) * n
        if float(np.linalg.norm(d)) < 1e-6:
            return None
        return RG.unit(d)

    def peel_removal(self, m: str, n_in) -> bool:
        for k in range(int(self.retries["removal"])):
            code = self.capture_feature("blocker:loop", palm_dir_fn=self._pd_away, label=f"loop_{k}")
            if code in ("NOT_ATTACHED", "NO_PREGRASP", "ARM_REFUSED"):
                return False
            if code != "DONE":
                self.exit_band(n_in, label="loop_capture_backoff")
                continue
            self.wait(3.0)
            if self.P.get("peel_mode") and self.r.status().get("flight_mode") != self.P["peel_mode"]:
                self._act(self.r.set_mode(self.P["peel_mode"]))
            moved = self.pull(n_in, self.P["peel_pull_m"], self.P["peel_s"], "peel", progress=True)
            if self.r.status().get("flight_mode") != "STANDARD":
                self._act(self.r.set_mode("STANDARD"))
            self.wait(3.0)
            held = self.T.holding("blocker")
            self.note("peel", attempt=k, moved=moved, held=held)
            if held and moved >= max(self.probe_min, float(self.P["removed_fraction"]) * float(self.P["peel_pull_m"])):
                return True
            if self.T.holding():
                self.release_hand(n_in, "peel_release")
                self.close_hand_if_empty(n_in)
        return False

    def _dp(self, m: str) -> float:
        ch = self.r.station_telemetry()["channels"]
        return 0.5 * (float(ch[f"cau.{m.upper()}.return_dp_a"]) + float(ch[f"cau.{m.upper()}.return_dp_b"]))

    def fan_stop_removal(self, m: str, n_in) -> bool:
        """S route: grasp first, then the stop; back off when the return pressure is below dp_go; fan on at once."""
        M = m.upper()
        for k in range(2):
            code = None
            for j in range(int(self.retries["capture"])):
                code = self.capture_feature("blocker:strap", label=f"strap_{k}_{j}")
                if code in ("NOT_ATTACHED", "NO_PREGRASP", "ARM_REFUSED"):
                    return False
                if code == "DONE":
                    break
                self.exit_band(n_in, label="strap_capture_backoff")
            if code != "DONE":
                return False
            self.wait(2.0)
            res = self.r.station_request("cau_fan", M, "off")
            self.note("fan_off", unit=M, code=res.get("code"))
            if res.get("code") != "ACCEPTED":
                return False
            t_req = self.t()
            while self.t() - t_req < float(self.P["fan_wait_s"]) and self._dp(m) >= self.dp_go:
                self.wait(1.0)
            moved = self.pull(n_in, self.s_backoff, self.P["s_pull_s"], "pull_after_stop")
            res = self.r.station_request("cau_fan", M, "on")
            self.note("fan_on", unit=M, code=res.get("code"), stop_s=self.t() - t_req, moved=moved)
            self.wait(2.0)
            held = self.T.holding("blocker")
            if held and moved >= max(self.probe_min, float(self.P["removed_fraction"]) * self.s_backoff):
                return True
            if self.T.holding():
                self.release_hand(n_in, "strap_release")
                self.close_hand_if_empty(n_in)
            if not self.time_ok("blocker"):
                return False
            self.wait(125.0)            # the next stop request only 120 s after the restart
        return False

    # ================================================================== stowage (P4 plan, P5 check)
    def _plates(self, m: str, face: str, palm_z) -> list:
        M = m.upper()
        U = self.map.unit_part(m)
        if face == "S":
            up = float(np.asarray(palm_z, float) @ U.b)
            first = ["F1", "F2"] if up >= 0 else ["F2", "F1"]
            order = first + ["F3", "F4"]
        elif face == "P1":
            order = ["F3", "F1", "F2", "F4"]
        else:
            order = ["F4", "F1", "F2", "F3"]
        return [f"SB-{M}-{x}" for x in order]

    def _plan(self, item: str, kind: str, m: str, face: str, z_choice=None, standoff=None, turn=None):
        held = self.T.held_pose(item)
        if held is None:
            return None
        F = self.map.face(m, face)
        so = -float(self.press["press_mm"]) / 1000.0 if standoff is None else float(standoff)
        if kind == "pouch":
            return RG.plan_pouch(self.geom, F, held, plate_side=self.V["world.stowage_unit.pad_plate"]["selected"], standoff=so,
                                 align=self.P["pouch_align"], clamp_deg=float(self.P["pouch_clamp_deg"]), turn_deg=turn)
        _, q, _ = self.pose()
        R_palm_world = RN.rot_of(q) @ self.geom["Rb"]
        return RG.plan_sheet(self.geom, F, held, R_palm_world, standoff=so, z_choice=z_choice,
                             turn_deg=(0.0 if turn is None else float(turn)))

    def _plan_full(self, item: str, kind: str, m: str, face: str, z_choice, max_deg: float):
        """The squared plan of the item (palm square to the face, no turn) turned by ref_geom.plan_aligned so that the
        item's face lies in the face plane at its pose in the hand now (P4), the turn clamped to max_deg."""
        base = self._plan(item, kind, m, face, z_choice, turn=0.0)
        if base is None:
            return None
        held = self.T.held_pose(item)
        if held is None:
            return None
        so = -float(self.press["press_mm"]) / 1000.0
        return RG.plan_aligned(self.geom, self.map.face(m, face), held, kind, base, so, max_deg)

    def _athr(self, name: str, kind: str) -> float:
        """An align threshold (P["align"][name]): a number, or per kind {'sheet' | 'pouch'} with optional per-route
        'sheet_fan_stop' / 'sheet_peel' entries."""
        v = self.P["align"][name]
        if not isinstance(v, dict):
            return float(v)
        if kind == "sheet":
            k = f"sheet_{self.diag.get('route') or 'peel'}"
            if k in v:
                return float(v[k])
        return float(v[kind])

    def _held_dev(self, item: str, kind: str, plan):
        """(deviation of the item's sealing face from the face plane for ``plan``'s palm attitude [deg], the item's in-hand
        angle about the bar [deg]) from the held pose (P4); (None, None) when the item is not in the hand."""
        held = self.T.held_pose(item)
        if held is None:
            return None, None
        z, phi = RG.held_item_normal(held, kind)
        if float(z[0]) < 0.0:
            # the item has swung over to the palm side of the bar (its handle side away from the palm): its face can
            # be parallel to the face plane but it cannot reach the face -- never pressed, as if edge-on (a palm turn
            # does not bring it back: _align does not turn for it)
            self._flipped[item] = True
            return 90.0, phi
        self._flipped[item] = False
        return RG.item_normal_dev_deg(plan["R_palm"] @ z, plan["n"]), phi

    def _set_mode(self, mode: str):
        if self.r.status().get("flight_mode") != mode:
            self._act(self.r.set_mode(mode))

    def _align(self, item: str, kind: str, m: str, face: str, z_choice, plates, plan, label: str, at: str = "pre"):
        """At the pre-deposit point: read the item's deviation for the plan (P4); above align.dev_go (or the item on the
        palm side of the bar), turn the palm (align.mode 'full': the smallest rotation that lays the item's face in the
        face plane; 'twist': about the bar by the in-hand angle), clamped to align.turn_max and scaled down until the
        moves to the near and press poses keep the zone rule (the program's own prediction on the as-surveyed map,
        stowed inflation inside the unit's approach zone), in align.turn_mode (HUSH: the item follows a slow palm), and
        read again: <= align.tries reads, <= align.max_turns turns per item.  Returns the plan to insert with (None: the
        item left the hand)."""
        A = self.P["align"]
        pre = float(self.P["pre_out_m"])
        near = float(A["near_m"][kind])
        turn = float(plan.get("turn_deg") or 0.0)
        tmax = float(A["turn_max"][kind])
        if at == "transit":
            # out at the transit point (the cabin air there is the background, not the rack-face sweep): the turn is a
            # validated fly_to (arm band) that keeps the robot's offset from the plan's deposit pose
            p_here, _, _ = self.pose()
        for a in range(int(A["tries"])):
            self.wait(float(A["settle_s"]))
            dev, phi = self._held_dev(item, kind, plan)
            self.note("align_read", item=item, face=face, attempt=a, dev_deg=dev, inhand_deg=phi, turn_deg=turn, label=label)
            if dev is None:
                return None
            if dev <= self._athr("dev_go", kind):
                return plan
            if self._turns.get(item, 0) >= int(A.get("max_turns", 4)) or self._flipped.get(item):
                return plan
            pt = None
            modes = ("full", "twist") if A.get("mode") == "full" else ("twist",)
            for md in modes:                               # the full turn, else (zone) the twist about the bar alone
                for sc in (1.0, 0.7, 0.45):
                    if md == "full":
                        c = self._plan_full(item, kind, m, face, z_choice, tmax * sc)
                    else:
                        c = self._plan(item, kind, m, face, z_choice, turn=float(np.clip(phi * sc, -tmax, tmax)))
                    if c is None:
                        return None
                    if at == "transit":
                        okz = self._transit_turn_ok(plan, c, p_here, pre, near)
                    else:
                        okz = self._approach_ok(plan, c, pre, near)
                    if okz and float(c.get("item_normal_dev_deg", 90.0)) < float(dev) - 3.0:
                        pt = c
                        break
                if pt is not None:
                    break
            if pt is None or RN.angle_deg(pt["q_dep"], plan["q_dep"]) < 3.0:
                return plan
            want = float(pt.get("aligned_deg") if pt.get("aligned_deg") is not None else pt.get("turn_deg") or 0.0)
            self._set_mode(str(A["turn_mode"]))
            if at == "transit":
                code = self.fly(p_here + (pt["p_dep"] - plan["p_dep"]), pt["q_dep"], f"{item}_align_turn", retries=0)
                plate = None
            else:
                code, plate, _ = self.fly_rel_any(plates, pt["p_dep"] + pt["n"] * pre, pt["q_dep"])
            self._set_mode("STANDARD")
            self._turns[item] = self._turns.get(item, 0) + 1
            self.note("align_turn", item=item, face=face, attempt=a, want_deg=want, code=code, plate=plate, at=at)
            if code in ("NO_FIDUCIAL", "REFUSED_ZONE"):
                # the turned pose is not reachable (plate hidden / zone): back to the last accepted plan
                if at == "transit":
                    self.fly(p_here, plan["q_dep"], f"{item}_align_back", retries=0)
                else:
                    self.fly_rel_any(plates, plan["p_dep"] + plan["n"] * pre, plan["q_dep"])
                continue
            if at == "transit":
                p_here = p_here + (pt["p_dep"] - plan["p_dep"])
            plan, turn = pt, want
        return plan

    def _swing(self, item: str, kind: str, plan) -> None:
        """Turn the grasped bar to its deposit direction in place, slowly (P["swing"]): the smallest rotation of the body
        that takes the palm z (the bar) now onto the plan's palm z, in swing.mode, from the robot's own estimate; the
        remaining turn about the bar is left to the transit."""
        S = self.P.get("swing") or {}
        key = kind if kind == "pouch" else f"sheet_{self.diag.get('route') or 'peel'}"
        if not S.get(key):
            return
        p, q, _ = self.pose()
        R = RN.rot_of(q)
        a_now = (R @ self.geom["Rb"])[:, 2]
        a_dep = RG.unit(plan["palm_z"])
        c = float(np.clip(a_now @ a_dep, -1.0, 1.0))
        ang = math.acos(c)
        if math.degrees(ang) < float(S.get("min_deg", 12.0)):
            return
        ax = np.cross(a_now, a_dep)
        if float(np.linalg.norm(ax)) < 1e-6:                  # a half turn: about the face normal of the plan
            ax = np.asarray(plan["n"], float)
        Q = RG._axis_angle(RG.unit(ax), ang)
        q_sw = RN.quat_of(Q @ R)
        if self.zones.segment(p, q, p, q_sw, self.arm_deployed()) != "ok":
            return
        self._set_mode(str(S.get("mode", "HUSH")))
        code = self.fly(p, q_sw, f"{item}_swing", retries=0)
        self._set_mode("STANDARD")
        self.note("swing", item=item, deg=math.degrees(ang), code=code)

    def _transit_turn_ok(self, cur, new, p_here, pre: float, near: float) -> bool:
        """The zone prediction of a turn at the transit point (a fly_to with the arm band) and of the approach of the
        turned plan from there (pre-deposit, near and press poses, fly_rel inside the unit's zone)."""
        p1 = p_here + (new["p_dep"] - cur["p_dep"])
        if self.zones.segment(p_here, cur["q_dep"], p1, new["q_dep"], True) != "ok":
            return False
        if self.zones.segment(p1, new["q_dep"], new["p_dep"] + new["n"] * pre, new["q_dep"], False) != "ok":
            return False
        return self._approach_ok(new, new, pre, near)

    def _approach_ok(self, cur, new, pre: float, near: float) -> bool:
        """The zone prediction (as-surveyed map, the stowed inflation that holds inside a unit's approach zone, the band
        rule) of the turn to the new pre-deposit pose and of its insert legs (near point, press point)."""
        a = (cur["p_dep"] + cur["n"] * pre, cur["q_dep"])
        b = (new["p_dep"] + new["n"] * pre, new["q_dep"])
        c = (new["p_dep"] + new["n"] * near, new["q_dep"])
        d = (new["p_dep"], new["q_dep"])
        for (p0, q0), (p1, q1) in ((a, b), (b, c), (c, d)):
            if self.zones.segment(p0, q0, p1, q1, False) != "ok":
                return False
        return True

    def _transit_goals(self, plan, m: str) -> list:
        U = self.map.unit_part(m)
        x, b, a = plan["n"], U.b, U.a
        offs = [(dn, db, da) for dn in (0.0, 0.1, 0.2, -0.1) for db in (0.0, -0.1, 0.1, -0.2, 0.2) for da in (0.0,)]
        offs.sort(key=lambda o: (round(math.hypot(o[0], o[1]), 6), o[0] < 0))
        base = plan["p_dep"] + x * float(self.P["transit_out_m"])
        return [base + x * dn + b * db + a * da for dn, db, da in offs]

    def fly_rel_any(self, plates, pos_w, quat_w, max_speed=None, m=None) -> tuple:
        tried = []
        for pid in plates:
            pp, qq = self.map.to_plate(pid, pos_w, quat_w)
            code = self._act(self.r.fly_rel(pid, pp, qq, max_speed=max_speed))
            tried.append((pid, code))
            if code != "NO_FIDUCIAL":
                return code, pid, tried
        return "NO_FIDUCIAL", None, tried

    def stow(self, item: str, m: str, faces: list, kind: str) -> bool:
        """Carry the held item to a face of module m's unit and place it (module doc; rules 1-5)."""
        z_choices = self._sheet_z_order(m) if kind == "sheet" else [None]
        for face in faces:
            for zc in z_choices:
                res = self._place_on(item, m, face, kind, zc)
                self.ops.append(dict(op="stow", item=item, face=face, z_choice=zc, result=res, t=self.t()))
                if res in ("PLACED", "LOST", "GAVE_UP", "NOT_HELD"):
                    if res == "PLACED" and kind == "pouch":
                        self.pads_used[m].append(face)
                    if res in ("PLACED", "LOST", "GAVE_UP"):
                        self._placed_now = (res == "PLACED")
                        self._after_place(m, face)
                        self._placed_now = False
                    else:
                        self.release_if_holding(item, "not_held")
                    return res == "PLACED"
                if self.presses.get(item, 0) >= int(self.P["max_presses"]):
                    break
                # 'REFUSED' / 'NO_FIDUCIAL': try the next attitude / face (never let go in place); the hand is still
                # closed on the item, back on the transit side of the unit
        self.note("stow_failed", item=item, faces=faces)
        self.release_if_holding(item, "stow_failed")
        return False

    def release_if_holding(self, item, why: str):
        """Last resort when no face takes the item: let it go here (the hand must be free: the interlock blocks the
        arm, every later capture and the dock), back off, close only with an empty cage."""
        st = self.r.status()
        if self.T.holding() or st.get("interlock"):
            p, q, _ = self.pose()
            m = self.map.module_of_point(p)
            c = self.map.centreline_point(m, p)
            self._act(self.r.gripper(True))
            self.note("release_last_resort", item=item, why=why)
            self.wait(1.0)
            self.exit_band(c - p, label="release_last_resort_out")
            self.close_hand_if_empty(c - p)

    def _sheet_z_order(self, m: str) -> list:
        order = list(self.P["sheet_z_order"][(self.diag.get("route") or "peel")])
        if "near_b" in order or "far_b" in order:
            _, q, _ = self.pose()
            zb = float((RN.rot_of(q) @ self.geom["Rb"])[:, 2] @ self.map.unit_part(m).b)
            near, far = ("+b", "-b") if zb >= 0 else ("-b", "+b")
            order = [near if z == "near_b" else far if z == "far_b" else z for z in order]
        out = []
        for z in order:
            if z not in out:
                out.append(z)
        return out

    def _after_place(self, m: str, face: str):
        F = self.map.face(m, face)
        if self.r.status().get("interlock"):
            self._act(self.r.gripper(True))              # the hand is open after a placement, whatever happened
            self.wait(1.0)
        empty = self.close_hand_if_empty(F.n)
        self.exit_band(F.n, label=f"after_{face}")
        if self.P.get("arm_overlap") and self._placed_now and empty and self._last_in_module:
            # the module's last item is on its face and the hand is closed on an empty cage: the stow the corridor
            # needs starts now and runs during the move to the centreline (the move's zone check is unchanged: a
            # moving arm counts as deployed); go_module's blocking stow then waits only for the rest
            if not self.r.status().get("interlock") and not self.T.holding():
                code = self._act(self.r.arm_stow(wait=False))
                self._deploying = False
                self.note("arm_stow_overlap", module=m, face=face, code=code)
        self.to_centreline(f"after_{face}_centreline")

    def _place_on(self, item: str, m: str, face: str, kind: str, z_choice) -> str:
        plan = self._plan(item, kind, m, face, z_choice)
        if plan is None:
            return "NOT_HELD"
        plates = self._plates(m, face, plan["palm_z"])
        F = self.map.face(m, face)
        self.note("plan", item=item, face=face, z_choice=z_choice, dep_gap_m=float((plan["p_dep"] - F.c) @ F.n),
                  palm_z_b=float(RG.unit(plan["palm_z"]) @ F.b), p_dep=[round(float(v), 4) for v in plan["p_dep"]])
        self._swing(item, kind, plan)
        # transit: a validated fly_to to the first transit point the firmware accepts (a refused one is replaced)
        code = "NO_ROUTE"
        p0, q0, _ = self.pose()
        goals = self._transit_goals(plan, m)
        ranked = sorted(range(len(goals)), key=lambda k: (0 if self.zones.segment(p0, q0, goals[k], plan["q_dep"], True) == "ok" else 1, k))
        for k in ranked[:4]:
            code = self.fly(goals[k], plan["q_dep"], f"{item}_transit_{face}")
            if code == "SUCCESS" or code in TOL_CODES:
                break
            if code == "REFUSED_ZONE":
                self.to_centreline(f"{item}_transit_recentre")
        if code not in ("SUCCESS",) + TOL_CODES:
            return "REFUSED"
        if self.P.get("deposit") == "aligned" and (self.P["align"].get("where") or "pre") == "transit":
            self.wait(float(self.P["align"].get("transit_settle_s", 3.0)))
            plan = self._align(item, kind, m, face, z_choice, plates, plan, label="transit", at="transit")
            if plan is None:
                return "NOT_HELD"
        # pre-deposit on the plate (fiducial-relative): a hidden plate -> another plate of the unit
        p_pre = plan["p_dep"] + plan["n"] * float(self.P["pre_out_m"])
        code, plate, tried = self.fly_rel_any(plates, p_pre, plan["q_dep"])
        self.note("pre_deposit", item=item, face=face, code=code, plate=plate, tried=tried, z_choice=z_choice)
        if code == "NO_FIDUCIAL":
            return "NO_FIDUCIAL"
        if code == "REFUSED_ZONE":
            return "REFUSED"
        self._dep_plan[item] = plan                         # the attitude the robot holds at the pre-deposit point
        for attempt in range(int(self.retries["placement"])):
            res = self._deposit(item, m, face, kind, z_choice, plates, attempt)
            if res in ("PLACED", "LOST", "GAVE_UP"):
                return res
            if res in ("NO_FIDUCIAL", "REFUSED", "NOT_ALIGNED"):
                return "NO_FIDUCIAL" if res == "NO_FIDUCIAL" else "REFUSED"
        return "GAVE_UP"

    def _deposit_aligned(self, item, m, face, kind, z_choice, plates, attempt) -> str:
        """One placement from the pre-deposit point with the closed loop on the held item's deviation (P4): align (the
        palm turned about the bar in HUSH until the item's face is within align.dev_go of the face plane), insert leg 1
        to align.near_m off the press point, read again -- above align.dev_press back to the pre-deposit point and align
        again (<= align.backoffs), never a press above it ('NOT_ALIGNED': the caller tries the next attitude / face) --
        then leg 2, the press, the press check (P5): seated, or touching the face -> keep pressing align.more_s more and
        release (the suction seals it); not touching -> back to the pre-deposit point ('AGAIN'); then the open hand
        slides off the bar (pouches), backs off, and the P5 placement check (PLACED / LOST / the re-grasp)."""
        A = self.P["align"]
        F = self.map.face(m, face)
        fid = f"SB-{m.upper()}-{face}"
        pre = float(self.P["pre_out_m"])
        near = float(A["near_m"][kind])
        speed = float(self.press["speed_m_s"]) if kind == "sheet" else float(self.P["pouch_insert_speed"])
        speed1 = float((A.get("leg1_speed") or {}).get(kind) or speed)      # leg 1 (to the near point)
        last = self._dep_plan.get(item)
        # the plan of the attitude the robot holds (the same palm turn), on the item's pose in the hand now
        plan = self._plan(item, kind, m, face, z_choice, turn=(None if last is None else float(last.get("turn_deg") or 0.0)))
        if plan is None:
            return self._slipped(item, kind, F, fid, face, attempt)
        if attempt > 0:                                     # back at the pre-deposit point of that plan
            self.fly_rel_any(plates, plan["p_dep"] + plan["n"] * pre, plan["q_dep"], max_speed=0.05)
        dev = None
        where = A.get("where") or "pre"
        for b in range(int(A["backoffs"]) + 1):
            if where == "transit" and b > 0:
                # back out to the transit point (fly_to: the band rule lets the robot leave the face), align there,
                # come in again (translation)
                p, q, _ = self.pose()
                tgt = plan["p_dep"] + plan["n"] * float(self.P["transit_out_m"])
                self.fly(tgt, plan["q_dep"], f"{item}_back_to_transit", retries=1)
                self.wait(float(A.get("transit_settle_s", 3.0)))
                plan = self._align(item, kind, m, face, z_choice, plates, plan, label=f"attempt{attempt}_b{b}", at="transit")
                if plan is None:
                    return self._slipped(item, kind, F, fid, face, attempt)
                code, _, _ = self.fly_rel_any(plates, plan["p_dep"] + plan["n"] * pre, plan["q_dep"])
                if code == "NO_FIDUCIAL":
                    return "NO_FIDUCIAL"
            elif where != "transit":
                plan = self._align(item, kind, m, face, z_choice, plates, plan, label=f"attempt{attempt}_b{b}")
                if plan is None:
                    return self._slipped(item, kind, F, fid, face, attempt)
            self._dep_plan[item] = plan
            code, plate, _ = self.fly_rel_any(plates, plan["p_dep"] + plan["n"] * near, plan["q_dep"], max_speed=speed1)
            dev, phi = self._held_dev(item, kind, plan)
            self.note("near_check", item=item, face=face, attempt=attempt, backoff=b, code=code, dev_deg=dev, inhand_deg=phi,
                      turn_deg=plan.get("turn_deg"))
            if dev is None:
                return self._slipped(item, kind, F, fid, face, attempt)
            if code == "NO_FIDUCIAL":
                return "NO_FIDUCIAL"
            if code not in ("SUCCESS",) + TOL_CODES:
                return "REFUSED"
            if dev <= self._athr("dev_press", kind):
                break
            if where != "transit":
                self.fly_rel_any(plates, plan["p_dep"] + plan["n"] * pre, plan["q_dep"], max_speed=0.05)
        else:
            self.note("not_aligned", item=item, face=face, attempt=attempt, dev_deg=dev, z_choice=z_choice)
            return "NOT_ALIGNED"
        method = self.P["place_method"] if kind == "sheet" else "press"
        if method == "flyby":
            # fly-by (the API form of the second placement method): open at the release gap, push on with the open fingers at the fly-by speed
            gap = float(self.flyby["release_gap_m"])
            fspeed = float(self.flyby["speed_m_s"])
            code, plate, _ = self.fly_rel_any(plates, plan["p_dep"] + plan["n"] * gap, plan["q_dep"], max_speed=speed)
            self.note("insert", item=item, face=face, code=code, plate=plate, method="flyby", attempt=attempt, speed=fspeed,
                      item_dev_deg=dev, turn_deg=plan.get("turn_deg"), aligned_deg=plan.get("aligned_deg"))
            if code == "NO_FIDUCIAL":
                return "NO_FIDUCIAL"
            if code not in ("SUCCESS",) + TOL_CODES:
                return "REFUSED"
            self.presses[item] = self.presses.get(item, 0) + 1
            self._act(self.r.gripper(True))
            self.fly_rel_any(plates, plan["p_dep"] + plan["n"] * 0.02, plan["q_dep"], max_speed=fspeed)
            return self._release_and_check(item, m, face, kind, z_choice, plates, plan, attempt, F, fid)
        # leg 2 and the press
        code, plate, _ = self.fly_rel_any(plates, plan["p_dep"], plan["q_dep"], max_speed=speed)
        self.note("insert", item=item, face=face, code=code, plate=plate, method="press", attempt=attempt, speed=speed,
                  dep_gap_m=float((plan["p_dep"] - F.c) @ F.n), palm_z_b=float(RG.unit(plan["palm_z"]) @ F.b),
                  item_dev_deg=dev, turn_deg=plan.get("turn_deg"), aligned_deg=plan.get("aligned_deg"))
        if code == "NO_FIDUCIAL":
            return "NO_FIDUCIAL"
        if code not in ("SUCCESS",) + TOL_CODES:
            return "REFUSED"
        self.presses[item] = self.presses.get(item, 0) + 1
        if (self.P.get("press_settle") or {}).get(kind):
            c2, _, _ = self.fly_rel_any(plates, plan["p_dep"], plan["q_dep"], max_speed=speed)
            self.ops.append(dict(op="fly", label=f"{item}_press_settle", code=c2, t=self.t()))
        self.wait(float(self.press["dwell_s"]))
        seat = self.T.on_face(item, fid)                   # P5 at the press, the item still in the hand
        own = 0.0 if seat is None else float(seat.get("own_fraction", 0.0))
        seated = self._seated(seat, kind)
        if not seated and own >= float(A.get("touch_fraction", 0.05)):
            self.wait(float(A.get("more_s", 3.0)))         # touching the face: keep pressing, the suction seals it
            seat = self.T.on_face(item, fid)
            own = 0.0 if seat is None else float(seat.get("own_fraction", 0.0))
            seated = self._seated(seat, kind)
        self.note("press_check", item=item, face=face, attempt=attempt, seated=seated, own_fraction=own)
        if own < float(A.get("touch_fraction", 0.05)) and attempt + 1 < int(self.retries["placement"]) \
                and self.presses[item] < int(self.P["max_presses"]):
            # not on the face at all: keep the item, back to the pre-deposit point, align again
            self.fly_rel_any(plates, plan["p_dep"] + plan["n"] * pre, plan["q_dep"], max_speed=float(self.press["backoff_speed_m_s"]))
            return "AGAIN"
        self._act(self.r.gripper(True))
        self.wait(1.0)
        return self._release_and_check(item, m, face, kind, z_choice, plates, plan, attempt, F, fid)

    def _slipped(self, item, kind, F, fid, face, attempt) -> str:
        """The item left the hand before the press: free the hand, back off, then the P5 check."""
        if self.r.status().get("interlock"):
            self._act(self.r.gripper(True))
        p, q, _ = self.pose()
        self.fly(p + F.n * float(self.press["backoff_m"]), q, f"{item}_backoff_slipped", max_speed=float(self.press["backoff_speed_m_s"]), retries=1)
        self.wait(2.0)
        chk = self.T.on_face(item, fid)
        ok = self._placed_ok(item, kind, chk)
        self.note("placed_check", item=item, face=face, attempt=attempt, ok=ok, slipped=True,
                  check=(None if chk is None else {k: v for k, v in chk.items() if k != "other"}))
        return "PLACED" if ok else ("LOST" if chk is None else "GAVE_UP")

    def _release_and_check(self, item, m, face, kind, z_choice, plates, plan, attempt, F, fid) -> str:
        """After the hand opened at the face: rule 2, the open hand slides off the bar (pouches), backs off; the P5 check;
        still between the fingers or stuck on the face -> re-grasped for another attempt (the reference's block)."""
        cl = float(self.P["clear_m"][kind])
        if cl > 0:
            g = plan["p_dep"] + RG.unit(plan["palm_z"]) * cl
            code, _, _ = self.fly_rel_any(plates, g, plan["q_dep"], max_speed=float(self.P["clear_speed"]))
            self.ops.append(dict(op="fly", label=f"{item}_clear_fingers", code=code, t=self.t()))
            if code not in ("SUCCESS",) + TOL_CODES:
                p, q, _ = self.pose()
                self.fly(p + RG.unit(plan["palm_z"]) * cl, q, f"{item}_clear_fingers_map", max_speed=float(self.P["clear_speed"]), retries=0)
        g = plan["p_dep"] + (RG.unit(plan["palm_z"]) * cl if cl > 0 else 0.0) + F.n * float(self.press["backoff_m"])
        code, _, _ = self.fly_rel_any(plates, g, plan["q_dep"], max_speed=float(self.press["backoff_speed_m_s"]))
        self.ops.append(dict(op="fly", label=f"{item}_backoff", code=code, t=self.t()))
        if code not in ("SUCCESS",) + TOL_CODES:
            p, q, _ = self.pose()
            self.fly(p + F.n * float(self.press["backoff_m"]), q, f"{item}_backoff_map", max_speed=float(self.press["backoff_speed_m_s"]), retries=1)
        self.wait(2.0)
        chk = self.T.on_face(item, fid)
        ok = self._placed_ok(item, kind, chk)
        self.note("placed_check", item=item, face=face, attempt=attempt, ok=ok,
                  check=(None if chk is None else {k: v for k, v in chk.items() if k != "other"}))
        if ok:
            return "PLACED"
        if chk is None:
            return "LOST"                                   # floats: not chased (K counts the single attempt)
        if attempt + 1 >= int(self.retries["placement"]):
            return "GAVE_UP"
        return self._regrasp(item, m, face, kind, z_choice, plates, chk, F)

    def _regrasp(self, item, m, face, kind, z_choice, plates, chk, F) -> str:
        """The re-grasp after a placement that left the item between the fingers or stuck on the face (a seal below
        regrasp_max_fraction); back on the pre-deposit side: 'AGAIN' (the next attempt aligns again) or 'GAVE_UP'."""
        pre = float(self.P["pre_out_m"])
        if chk.get("between_fingers") or chk.get("held"):
            self._act(self.r.gripper(False))
            self.wait(1.5)
            if not self.T.holding(item):
                return "GAVE_UP"
            p_pre = self._plan(item, kind, m, face, z_choice)
            if p_pre is None:
                return "GAVE_UP"
            code, _, _ = self.fly_rel_any(plates, p_pre["p_dep"] + p_pre["n"] * pre, p_pre["q_dep"])
            return "AGAIN" if code not in ("NO_FIDUCIAL", "REFUSED_ZONE") else "GAVE_UP"
        if (chk.get("sealed_m2", 0.0) > 0.0 or chk.get("other")) and \
                float(chk.get("own_fraction", 0.0)) < float(self.P["regrasp_max_fraction"]):
            bar = "blocker:loop" if kind == "sheet" and self.diag.get("route") == "peel" else ("blocker:strap" if kind == "sheet" else f"{item}:handle")
            code = self.capture_feature(bar, palm_dir_fn=(self._pd_away if bar == "blocker:loop" else None), label=f"{item}_regrasp")
            if code != "DONE":
                self.exit_band(F.n, label=f"{item}_regrasp_backoff")
                return "GAVE_UP"
            moved = self.pull(F.n, 0.20, 20.0, f"{item}_regrasp_pull")
            if not self.T.holding(item):
                return "GAVE_UP"
            p_pre = self._plan(item, kind, m, face, z_choice)
            if p_pre is None:
                return "GAVE_UP"
            code, _, _ = self.fly_rel_any(plates, p_pre["p_dep"] + p_pre["n"] * pre, p_pre["q_dep"])
            self.note("regrasp", item=item, moved=moved, code=code)
            return "AGAIN" if code not in ("NO_FIDUCIAL", "REFUSED_ZONE") else "GAVE_UP"
        if chk.get("sealed_m2", 0.0) > 0.0 or chk.get("other"):
            return "GAVE_UP"                                # stuck on a face too firmly to pull off: left as it is
        return "LOST"

    def _deposit(self, item, m, face, kind, z_choice, plates, attempt) -> str:
        """One placement attempt from the pre-deposit point: the closed-loop
        deposit (default, reference.deposit 'aligned') or the legacy one ('legacy')."""
        if self.P.get("deposit") == "aligned":
            return self._deposit_aligned(item, m, face, kind, z_choice, plates, attempt)
        return self._deposit_legacy(item, m, face, kind, z_choice, plates, attempt)

    def _deposit_legacy(self, item, m, face, kind, z_choice, plates, attempt) -> str:
        """The legacy insert-release-back-off from the pre-deposit point, then the P5 check: PLACED, AGAIN (still between the
        fingers or stuck on the face: re-grasped, ready for another attempt), LOST (floats: not chased), GAVE_UP."""
        F = self.map.face(m, face)
        fid = f"SB-{m.upper()}-{face}"
        method = self.P["place_method"] if kind == "sheet" else "press"
        gap = float(self.flyby["release_gap_m"])
        so = gap if method == "flyby" else None

        def plan_now():
            return self._plan(item, kind, m, face, z_choice, standoff=so)
        plan = plan_now()
        if plan is None:
            # the item left the hand (after a press that did not seat): free the hand, back off, then the P5 check
            if self.r.status().get("interlock"):
                self._act(self.r.gripper(True))
            p, q, _ = self.pose()
            self.fly(p + F.n * float(self.press["backoff_m"]), q, f"{item}_backoff_slipped", max_speed=float(self.press["backoff_speed_m_s"]), retries=1)
            self.wait(2.0)
            chk = self.T.on_face(item, fid)
            ok = self._placed_ok(item, kind, chk)
            self.note("placed_check", item=item, face=face, attempt=attempt, ok=ok, slipped=True,
                      check=(None if chk is None else {k: v for k, v in chk.items() if k != "other"}))
            return "PLACED" if ok else ("LOST" if chk is None else "GAVE_UP")
        # the item turns freely about the grasped bar: wait for it to face the face before the last approach (P4)
        waits = 0
        while plan["item_normal_dev_deg"] > float(self.P["square_max_deg"]) and waits < int(self.P["square_wait_n"]):
            self.wait(2.0)
            waits += 1
            plan = plan_now()
            if plan is None:
                return "GAVE_UP"
        if method == "press":
            speed = float(self.press["speed_m_s"]) if kind == "sheet" else float(self.P["pouch_insert_speed"])
        else:
            speed = float(self.flyby["speed_m_s"])
        code, plate, tried = self.fly_rel_any(plates, plan["p_dep"], plan["q_dep"], max_speed=speed)
        self.note("insert", item=item, face=face, code=code, plate=plate, method=method, attempt=attempt, speed=speed,
                  dep_gap_m=float((plan["p_dep"] - F.c) @ F.n), palm_z_b=float(RG.unit(plan["palm_z"]) @ F.b),
                  item_dev_deg=plan["item_normal_dev_deg"], square_waits=waits)
        if code == "NO_FIDUCIAL":
            return "NO_FIDUCIAL"
        if code not in ("SUCCESS",) + TOL_CODES:
            return "REFUSED"                                # never let go short of the face (rule 5)
        self.presses[item] = self.presses.get(item, 0) + 1
        if method == "press":
            if (self.P.get("press_settle") or {}).get(kind):
                c2, _, _ = self.fly_rel_any(plates, plan["p_dep"], plan["q_dep"], max_speed=speed)
                self.ops.append(dict(op="fly", label=f"{item}_press_settle", code=c2, t=self.t()))
            self.wait(float(self.press["dwell_s"]))
            seat = self.T.on_face(item, fid)               # P5 at the press, the item still in the hand
            seated = self._seated(seat, kind)
            self.note("press_check", item=item, face=face, attempt=attempt, seated=seated,
                      own_fraction=(None if seat is None else seat.get("own_fraction")))
            if not seated and attempt + 1 < int(self.retries["placement"]) and self.presses[item] < int(self.P["max_presses"]):
                # not flat on the face: keep it, back off a little on the plate, plan and press again
                back = plan["p_dep"] + plan["n"] * float(self.P["reseat_backoff_m"])
                self.fly_rel_any(plates, back, plan["q_dep"], max_speed=speed)
                return "AGAIN"
            self._act(self.r.gripper(True))
            self.wait(1.0)
        else:
            # fly-by (API form): the hand opens at the release gap and pushes on with the open fingers, so the item
            # leaves the hand moving toward the face at the fly-by speed (a blocking move cannot open mid-flight)
            self._act(self.r.gripper(True))
            push = plan["p_dep"] - plan["n"] * max(0.0, gap - 0.02)
            self.fly_rel_any(plates, push, plan["q_dep"], max_speed=speed)
        # rule 2: the open hand leaves the bar along its axis first (pouch handle), then backs off
        cl = float(self.P["clear_m"][kind])
        if cl > 0:
            g = plan["p_dep"] + RG.unit(plan["palm_z"]) * cl
            code, _, _ = self.fly_rel_any(plates, g, plan["q_dep"], max_speed=float(self.P["clear_speed"]))
            self.ops.append(dict(op="fly", label=f"{item}_clear_fingers", code=code, t=self.t()))
            if code not in ("SUCCESS",) + TOL_CODES:
                p, q, _ = self.pose()
                self.fly(p + RG.unit(plan["palm_z"]) * cl, q, f"{item}_clear_fingers_map", max_speed=float(self.P["clear_speed"]), retries=0)
        g = plan["p_dep"] + (RG.unit(plan["palm_z"]) * cl if cl > 0 else 0.0) + F.n * float(self.press["backoff_m"])
        code, _, _ = self.fly_rel_any(plates, g, plan["q_dep"], max_speed=float(self.press["backoff_speed_m_s"]))
        self.ops.append(dict(op="fly", label=f"{item}_backoff", code=code, t=self.t()))
        if code not in ("SUCCESS",) + TOL_CODES:
            p, q, _ = self.pose()
            self.fly(p + F.n * float(self.press["backoff_m"]), q, f"{item}_backoff_map", max_speed=float(self.press["backoff_speed_m_s"]), retries=1)
        self.wait(2.0)
        chk = self.T.on_face(item, f"SB-{m.upper()}-{face}")
        ok = self._placed_ok(item, kind, chk)
        self.note("placed_check", item=item, face=face, attempt=attempt, ok=ok,
                  check=(None if chk is None else {k: v for k, v in chk.items() if k != "other"}))
        if ok:
            return "PLACED"
        if chk is None:
            return "LOST"                                   # floats: not chased (K counts the single attempt)
        if attempt + 1 >= int(self.retries["placement"]):
            return "GAVE_UP"
        # still between the fingers, or stuck on the face: grasp it again and place it once more
        if chk.get("between_fingers") or chk.get("held"):
            self._act(self.r.gripper(False))
            self.wait(1.5)
            if not self.T.holding(item):
                return "GAVE_UP"
            p_pre = self._plan(item, kind, m, face, z_choice)
            if p_pre is None:
                return "GAVE_UP"
            code, _, _ = self.fly_rel_any(plates, p_pre["p_dep"] + p_pre["n"] * float(self.P["pre_out_m"]), p_pre["q_dep"])
            return "AGAIN" if code not in ("NO_FIDUCIAL", "REFUSED_ZONE") else "GAVE_UP"
        if (chk.get("sealed_m2", 0.0) > 0.0 or chk.get("other")) and \
                float(chk.get("own_fraction", 0.0)) < float(self.P["regrasp_max_fraction"]):
            bar = "blocker:loop" if kind == "sheet" and self.diag.get("route") == "peel" else ("blocker:strap" if kind == "sheet" else f"{item}:handle")
            code = self.capture_feature(bar, palm_dir_fn=(self._pd_away if bar == "blocker:loop" else None), label=f"{item}_regrasp")
            if code != "DONE":
                self.exit_band(F.n, label=f"{item}_regrasp_backoff")
                return "GAVE_UP"
            moved = self.pull(F.n, 0.20, 20.0, f"{item}_regrasp_pull")
            if not self.T.holding(item):
                return "GAVE_UP"
            p_pre = self._plan(item, kind, m, face, z_choice)
            if p_pre is None:
                return "GAVE_UP"
            code, _, _ = self.fly_rel_any(plates, p_pre["p_dep"] + p_pre["n"] * float(self.P["pre_out_m"]), p_pre["q_dep"])
            self.note("regrasp", item=item, moved=moved, code=code)
            return "AGAIN" if code not in ("NO_FIDUCIAL", "REFUSED_ZONE") else "GAVE_UP"
        if chk.get("sealed_m2", 0.0) > 0.0 or chk.get("other"):
            return "GAVE_UP"                                # stuck on a face too firmly to pull off: left as it is
        return "LOST"

    def _seated(self, chk, kind: str) -> bool:
        frac = float(self.P["sheet_ok_fraction"] if kind == "sheet" else self.P["pouch_ok_fraction"])
        return chk is not None and float(chk.get("own_fraction", 0.0)) >= frac

    def _placed_ok(self, item: str, kind: str, chk) -> bool:
        if chk is None or chk.get("held") or chk.get("between_fingers"):
            return False
        frac = float(self.P["sheet_ok_fraction"] if kind == "sheet" else self.P["pouch_ok_fraction"])
        return float(chk.get("own_fraction", 0.0)) >= frac

    # ================================================================== home
    def _free_arm(self, label: str):
        """The dock refused the arm (its joints are not at the stowed position, e.g. an item that got away is caught in
        the hand): open, swing the arm out and back (the firmware's own moves), back off from whatever is between the
        fingers, close only with an empty cage (rule 4)."""
        self._act(self.r.gripper(True))
        self.wait(1.0)
        d = self._act(self.r.arm_deploy(wait=True))
        p, q, _ = self.pose()
        c = self.map.centreline_point(self.map.module_of_point(p), p)
        self.close_hand_if_empty(c - p if float(np.linalg.norm(c - p)) > 0.05 else np.array([0.0, 0.0, 1.0]))
        st = self._act(self.r.arm_stow(wait=True))
        self.note("free_arm", label=label, deploy=d, stow=st)

    def home_and_dock(self):
        r = self.r
        if self.T.holding() or r.status().get("interlock"):
            self.note("holding_at_home")
            self.release_if_holding(None, "home")
        self.to_centreline("home_centreline")
        self.stow_arm(wait=False)
        self.go_module("l2")
        qL2 = RN.look_quat((0.0, -1.0, 0.0))
        self.fly(RN.HOME0, qL2, "home_0", ff=True)
        berth = int(self.P["dock_berth"])
        start, bq = self.map.approach_start(berth)
        self.fly(RN.HOME0, bq, "home_turn")
        self.fly(start, bq, "home_1")
        self.stow_arm(wait=True)
        code = self._act(r.dock(berth))
        n = 0
        while code != "DOCKED" and n < 2:
            n += 1
            if code == "REFUSED_ARM":
                self._free_arm(f"dock{n}")
            self.fly(start, bq, f"dock_reapproach{n}")
            self.stow_arm(wait=True)
            code = self._act(r.dock(berth))
        self.note("dock", code=code, berth=berth, tries=n + 1)
        self.lamp_off()


def run_reference(server, params: dict | None = None) -> dict:
    """The reference inside the cell (``server``: the CabinCellServer of the run)."""
    from ref_robot import Robot
    from ref_truth import Truth
    robot = Robot(server.handle)
    truth = Truth(server)
    prog = Program(robot, truth, params)
    out = prog.run()
    out["calls"] = robot.n_calls
    return out
