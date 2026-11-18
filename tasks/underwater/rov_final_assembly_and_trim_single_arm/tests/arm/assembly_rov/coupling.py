"""Coupling state machine of the final-assembly-and-trim cell: one state per SLOT of the kit (`kit["slots"]`), each slot
taking whichever of its candidate units reaches it (four thrusters of two part types for the four vertical-thruster
places, two guards for the two sides, twelve ballast blocks for the thirteen ballast sites).

Source (copied, then extended; grader side, never shipped to the agent):
  docs/plan/driving/tasks/racecar_full_assembly_tenth_scale_ur10e/tests/arm/assembly_car/coupling.py
  sha256 96b5a87620c8aaf6bcd8ca306cb168178837d20881e459b65518e7f375ec13b0
  the capture rest gate follows the phase-1 prototypes of this task (read-only, not imported):
  -dev/proto/bigparts/bp_cell.py `GatedCoupling` (sha256 72923f67b67fba18c211b8d303abb9d0244c4b798db40cd96220ba5d98a8c74a)
  -dev/proto/ballast/bal_cell.py `GatedCoupling` (sha256 73cbb15cc0e464fdd91239976e5df20ec0423665fda76a7c357f3fa0e3883693)
Unchanged from the source: the keyed / seat predicates, the soft weld that pulls a captured unit home at <= pull_speed
and pull_rate, rest for `hold_ticks` -> LOCKED, release outside release_factor x the window, scrapping (release first,
then off every candidate list; void mates), premounted slots, the sub-assembly rule and the wheel-stop machinery (no
slot of this kit has a stop; kept so that the source's behaviour is reproduced on any kit).

The four additions of this cell (DESIGN 1.2, rulings L13 / L19; each one has a switch, all on by default; `PRECEDENT`
= all four off reproduces the source machine on any kit, which the dev test checks pose by pose):
  1. handedness (`handedness`): a slot may name the part type it needs (`design_part`; the vertical-thruster places:
     one propeller hand per diagonal) and every candidate carries its part `type`. A candidate of another type is
     captured, pulled home and fastened like the right one (it fits), but LOCKED_OFF: it counts 0 and holds the place.
     `design_ok` = (the key orientation counts) and (the part type is the slot's design part).
  2. unscored sites (`unscored_sites`): slots with `scored: false` (the ballast sites) are left out of the structural
     weight and score; `record()` keeps them (with `scored`, the `site` record and the unit on it) so that the grader's
     trim reads the ballast that is LOCKED on its sites (`trim_sites()`); for such a slot `achieved` = a unit is
     fastened on it in a counting orientation and neither it nor its carrier is scrapped.
  3. capture rest gate (`capture_gate`): a keyed candidate is captured only once it has been inside the capture
     window AND at rest relative to its carrier on `still_ticks` consecutive ticks: EVERY POINT of the unit slower
     than `still_mm_s` relative to the carrier and the unit turning slower than `still_deg_s` relative to it (one
     counter per (slot, candidate); L19: 50 mm/s, 30 deg/s, 10 ticks). Without it a thruster let go above its window
     slides down the side plate through the window and a guard dropped onto its brackets is caught at the instant it
     lands (G6: two false locks; halving the windows did not help). "Every point" (ruling P09-1 (b) of group Q, a
     supplement to L19; `gate_point="extent"`, the default) is read at the farthest points of the unit: the corners of
     the local bounding boxes of all the geoms of the unit's body (`extent_points`), each one's velocity relative to
     the carrier's material point at the same place (the rigid-body velocities of the two bodies, `mj_objectVelocity`
     at their centres of mass, world orientation). A guard rocked down into its window by an opening pad at ~28 deg/s
     and ~30 mm/s at its centre of mass moves its ends at ~0.1 m/s and is not captured (record P2_P09 2-3: 2 / 5
     false locks with the centre-of-mass reading in G6's wider windows, 0 / 5 with this one; held placements
     unchanged). `gate_point="com"` is the earlier reading (the two centres of mass only), kept for the regression's
     comparison. The window predicates themselves (`_inside`) stay pure.
  4. per-slot past-home bound (`slot_past`): every slot must give its own `capture.past_mm` (the enclosure rests on
     its saddles 1.49 mm past its CAD home, so its bound is 2.0 mm; the others 1.0 mm). The source already reads
     `capture.past_mm` per slot but silently falls back to one global default (1 mm) when a slot has none, which for
     the enclosure means "never captured once it lies on the saddles"; with the switch on a missing or negative value
     is a kit error. Off = the source's fallback.
Also new, outside the four: the weld constraint forces (soft and hard welds) are reported on their own
(`weld_forces()`, `metrics()["weld_*"]`; the cell's H2 counts contact forces only, ruling L19), and `capture_table()`
carries the columns the cell manual prints for this cell (self-standing / hold before release, still before capture,
physical key, orientation count, whether the part type is checked) -- it is the single source of the manual's table
and holds no design pose, part type, unit name or hand.

Slot kinds (frames 4 x 4 in metres, z along the insertion; conventions in the dev folder's kit/frames.md):
  keyed  the part's frame (`frame_in_part`) against the carrier's (`frame_in_carrier`): the axes within the tilt, the
         axis lines within the lateral window, the part's frame home along the axis (from `capture.depth_mm` short of
         home to `capture.past_mm` past it), the turn about the axis within the clock window of a key orientation
         (k x pitch_deg). `any_key` slots (the ballast sites) count in every orientation, the others only in
         `design_k`; pitch 360 = one orientation only. Released once the key's tip is `key_release_clear_m` clear of
         the carrier's face (the key height `kit["carve"]["keys"][slot]["len_mm"]`; virtual in this cell: no
         physical key, `"virtual": true`).
  seat   the part's seat frame within the window of its design pose on the carrier (unused by this kit).

States: WAITING (a required slot not achieved) -> FREE -> ENGAGED (captured: a soft weld holds the part on the feature
and pulls it home) -> HOLDING (lock predicate holds and the unit rests relative to the carrier, `hold_ticks` ticks) ->
LOCKED (the weld turned rigid) or LOCKED_OFF (fastened off the design orientation or part type: counts 0). VOID:
fastened but its unit or carrier scrapped, or never fastened and its carrier scrapped. PREMOUNTED: fastened by the cell
at the start, not scored. A captured or fastened unit leaves every other slot's candidate list.

Kit fields read here: kit["slots"][*] = {id, kind, carrier, requires, weight, scored, tol_pos_mm, tol_ang_deg,
axial_tol_mm, frame_in_carrier (keyed), pitch_deg, any_key, design_k, design_part, insertion_carrier (seat),
capture{lateral_mm, tilt_deg, clock_deg, key_frac, depth_mm, past_mm, still_mm_s, still_deg_s, still_ticks},
self_standing, hold_before_release_s, site{station, name_03, hole_B_m, conflicts_with}, stop{...}, designed_unit,
candidates[{unit, part, type, T_part_in_carrier, frame_in_part, stop_geoms}]}; kit["carve"]["keys"][slot]{len_mm,
virtual}; kit["capture_gate"]{rel_speed_mm_s, rel_rate_deg_s, ticks} (cross-checked against the slots' still_*);
kit["loose"]; kit["bodies"][body]["unit"]. The weld of (slot, unit) is the equality `weld_name(slot, unit)`.
The cell calls `tick(sim_time)` after every control tick (after mj_forward) and `scrap(unit, sim_time, reason)` from its
watch (and for every loose unit on the vehicle when a turn of the positioner starts).
"""

import math

import numpy as np

STATES = ("WAITING", "FREE", "ENGAGED", "HOLDING", "LOCKED", "LOCKED_OFF", "VOID", "PREMOUNTED")
SCRAPPED = "<scrapped>"          # self.used[unit] of a scrapped unit: on no candidate list any more
EXTENSIONS = ("handedness", "unscored_sites", "capture_gate", "slot_past")
PRECEDENT = {k: False for k in EXTENSIONS}       # all four additions off: the source machine
GATE_POINTS = ("extent", "com")                  # the rest gate's speed: every point of the unit (default) / its centre of mass


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def rotation_angle(R):
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))


def quat_wxyz(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def body_pose(d, b):
    T = np.eye(4)
    T[:3, :3] = d.xmat[b].reshape(3, 3)
    T[:3, 3] = d.xpos[b]
    return T


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def weld_name(slot_id, unit):
    return "weld_%s__%s" % (slot_id, unit)


def _names(spec, unit):
    if not spec:
        return []
    if isinstance(spec, str):
        spec = [spec]
    return [str(n).format(unit=unit) for n in spec]


def _finite(v, what):
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("kit: %s is not a finite number" % what)
    return v


class CouplingStateMachine(object):
    def __init__(self, model, data, kit, hold_ticks=50, rel_speed_max=0.005, rest_ang_speed_max=math.radians(2.0),
                 capture_lateral_m=0.002, capture_ang_rad=math.radians(6.0), capture_key_frac=0.25,
                 capture_seat_depth_m=0.008, capture_seat_lateral_m=0.003, capture_past_m=0.001, release_factor=2.0,
                 key_release_clear_m=0.002, soft_solref=(0.03, 1.0), pull_speed_m_s=0.010,
                 pull_rate_rad_s=math.radians(10.0), stop_clear_m=1e-4, count_subassembly=True, premounted=(),
                 seat_settle_speed_m_s=0.05, seat_settle_rate_rad_s=math.radians(30.0),
                 handedness=True, unscored_sites=True, capture_gate=True, slot_past=True,
                 tick_s=0.01, weld_window_ticks=20, track_weld_force=True, gate_point="extent"):
        """The capture_* arguments are the defaults of a slot without its own `capture` entry (the source's global
        window); this kit gives every slot its own. seat_settle_*: a captured seat's pull starts once the part moves
        slower than this against its carrier. handedness / unscored_sites / capture_gate / slot_past: the four
        additions (module doc). tick_s: the control tick (only to print the rest gate's time, `capture.still_s`).
        weld_window_ticks: the window of the weld forces' sustained maximum (20 ticks = 0.2 s, H2's window).
        gate_point: where the rest gate reads the speed, "extent" (every point of the unit: its geoms' bounding-box
        corners; the product rule) or "com" (the centre of mass; the earlier reading)."""
        if gate_point not in GATE_POINTS:
            raise ValueError("gate_point %r (one of %s)" % (gate_point, ", ".join(GATE_POINTS)))
        self.gate_point = gate_point
        self._extent = {}
        self.ext = {"handedness": bool(handedness), "unscored_sites": bool(unscored_sites),
                    "capture_gate": bool(capture_gate), "slot_past": bool(slot_past)}
        self.tick_s = float(tick_s)
        self.seat_settle_v, self.seat_settle_w = float(seat_settle_speed_m_s), float(seat_settle_rate_rad_s)
        self.m, self.d, self.kit = model, data, kit
        self.hold_ticks, self.rel_speed_max, self.rest_ang = int(hold_ticks), float(rel_speed_max), float(rest_ang_speed_max)
        self.release_factor = float(release_factor)
        self.key_release_clear = float(key_release_clear_m)
        self.soft_solref = [float(v) for v in soft_solref]
        # the capture pulls the part home at a finite rate (the screws being driven), not in one jump: a hand that still
        # holds the part (or its pads opening) would otherwise pivot it
        self.pull_v, self.pull_w = float(pull_speed_m_s), float(pull_rate_rad_s)
        self.stop_clear = float(stop_clear_m)
        self.count_subassembly = bool(count_subassembly)
        self.collision_bits = None
        self.pull = {}
        self.loose = list(kit.get("loose") or [])
        self.loose_set = set(self.loose)
        self.body_unit = {bn: bb.get("unit") for bn, bb in (kit.get("bodies") or {}).items()}
        keys = (kit.get("carve") or {}).get("keys") or {}
        kit_gate = kit.get("capture_gate") if self.ext["capture_gate"] else None
        f = self.release_factor
        self.slots = []
        for s in kit["slots"]:
            sid = s["id"]
            x = dict(s)
            x["requires"] = list(s.get("requires") or [])
            x["tol_pos"] = s["tol_pos_mm"] * 1e-3
            x["tol_ang"] = math.radians(s["tol_ang_deg"])
            x["axial_tol"] = s.get("axial_tol_mm", 0.0) * 1e-3
            x["pitch"] = math.radians(s.get("pitch_deg", 360.0))
            x["bc"] = model.body(s["carrier"]).id
            x["Fc"] = np.array(s.get("frame_in_carrier", np.eye(4)), float)
            x["carrier_unit"] = self._unit_of(s["carrier"])
            cap = dict(s.get("capture") or {})
            seat = s["kind"] == "seat"
            x["cap_lat"] = float(cap.get("lateral_mm", 1e3 * (capture_seat_lateral_m if seat else capture_lateral_m))) * 1e-3
            x["cap_tilt"] = math.radians(float(cap.get("tilt_deg", math.degrees(capture_ang_rad))))
            # (4) the past-home bound: the slot's own, required (or the source's fallback to the global default)
            if self.ext["slot_past"]:
                if "past_mm" not in cap:
                    raise ValueError("kit: slot %s has no capture.past_mm (every slot gives its own)" % sid)
                past_mm = _finite(cap["past_mm"], "slot %s capture.past_mm" % sid)
                if past_mm < 0.0:
                    raise ValueError("kit: slot %s capture.past_mm %.3f < 0" % (sid, past_mm))
                x["cap_past"] = past_mm * 1e-3
            else:
                x["cap_past"] = float(cap.get("past_mm", 1e3 * capture_past_m)) * 1e-3
            if seat:
                x["cap_depth"] = float(cap.get("depth_mm", 1e3 * capture_seat_depth_m)) * 1e-3
                x["rel_depth"] = x["cap_depth"] * f
                ins = s.get("insertion_carrier") or s.get("insertion_robot_design") or s.get("insertion_robot")
                if ins is None:
                    raise ValueError("kit: seat slot %s has no insertion_carrier" % sid)
                ins = np.array(ins, float)
                x["ins_c"] = ins / np.linalg.norm(ins)
            elif s["kind"] == "keyed":
                if sid not in keys:
                    raise ValueError("kit: keyed slot %s has no carve.keys entry (its key height)" % sid)
                x["key_len"] = float(keys[sid]["len_mm"]) * 1e-3
                x["key_virtual"] = bool(keys[sid].get("virtual", False))
                x["key_frac"] = float(cap.get("key_frac", capture_key_frac))
                x["cap_clock"] = math.radians(float(cap.get("clock_deg", math.degrees(capture_ang_rad))))
                x["cap_depth"] = float(cap["depth_mm"]) * 1e-3 if "depth_mm" in cap else x["key_len"] * (1.0 - x["key_frac"])
                # a captured key is released once its tip is key_release_clear_m clear of the carrier's face
                x["rel_depth"] = max(x["key_len"] + self.key_release_clear, x["cap_depth"])
            else:
                raise ValueError("kit: slot %s has unknown kind %r" % (sid, s["kind"]))
            # (3) the capture rest gate of a keyed slot: the slot's capture.still_* (the kit-level capture_gate must agree)
            x["gate"] = None
            if self.ext["capture_gate"] and s["kind"] == "keyed":
                x["gate"] = self._gate_of(sid, cap, kit_gate)
            # (2) unscored slots
            x["scored"] = bool(s.get("scored", True)) if self.ext["unscored_sites"] else True
            x["site"] = dict(s["site"]) if (s.get("site") is not None and self.ext["unscored_sites"]) else None
            # (1) the part type a slot needs
            x["design_part"] = s.get("design_part") if self.ext["handedness"] else None
            # documents only (the manual's table): stands by itself / held before release
            x["self_standing"] = s.get("self_standing")
            x["hold_s"] = s.get("hold_before_release_s")
            if x["self_standing"] is False:
                hs = x["hold_s"]
                if hs is None or not (_finite(hs, "slot %s hold_before_release_s" % sid) > 0.0):
                    raise ValueError("kit: slot %s does not stand by itself but has no hold_before_release_s > 0" % sid)
            # the capture (release) region lies inside a ball about the carrier's slot point: |offset|^2 = across^2 + along^2
            x["reach"] = math.hypot(x["cap_lat"], max(x["cap_depth"], x["cap_past"])) + 5e-4
            x["rel_reach"] = math.hypot(x["cap_lat"] * f, max(x["rel_depth"], x["cap_past"] * f)) + 5e-4
            stop = s.get("stop") or {}
            x["has_stop"] = bool(stop) or any(c.get("stop_geoms") for c in s["candidates"])
            x["cands"] = []
            for c in s["candidates"]:
                Tpc = np.array(c["T_part_in_carrier"], float)
                Fp = np.array(c.get("frame_in_part", np.eye(4)), float)
                geoms = _names(c.get("stop_geoms") or stop.get("geom"), c["unit"])
                if x["design_part"] is not None and not c.get("type"):
                    raise ValueError("kit: slot %s needs part %s but candidate %s has no type" % (sid, x["design_part"], c["unit"]))
                x["cands"].append({"unit": c["unit"], "type": c.get("type"), "bp": model.body(c["part"]).id, "Tpc": Tpc, "Fp": Fp,
                                   "a_seat": (Tpc @ Fp)[:3, 3].copy(), "eq": model.equality(weld_name(sid, c["unit"])).id,
                                   "stop": [model.geom(n).id for n in geoms]})
            if x["design_part"] is not None and not any(c["type"] == x["design_part"] for c in x["cands"]):
                raise ValueError("kit: slot %s needs part %s and no candidate is of that type" % (sid, x["design_part"]))
            self.slots.append(x)
        self.by_id = {x["id"]: x for x in self.slots}
        self.hard_solref = {c["eq"]: model.eq_solref[c["eq"]].copy() for x in self.slots for c in x["cands"]}
        # every (slot, candidate) pair's two slot points (carrier side, part side) in their bodies: `_pair_dist`
        # measures all of them in one pass per tick (the cheap test before `evaluate`)
        bc, bp, pa, pb = [], [], [], []
        for x in self.slots:
            for c in x["cands"]:
                c["pair"] = len(bc)
                bc.append(x["bc"])
                bp.append(c["bp"])
                pa.append(c["a_seat"] if x["kind"] == "seat" else x["Fc"][:3, 3])
                pb.append(c["Fp"][:3, 3])
        self._pbc, self._pbp = np.array(bc, int), np.array(bp, int)
        self._pa, self._pb = np.array(pa, float).reshape(-1, 3), np.array(pb, float).reshape(-1, 3)
        self._dist = None
        # the wheel stops (none in this kit): their compiled collision bits, and every carrier geom a restored stop
        # must be clear of
        self.stop_bits = {}
        obstacles = set()
        stop_geoms = set()
        for x in self.slots:
            for c in x["cands"]:
                for g in c["stop"]:
                    stop_geoms.add(g)
                    self.stop_bits[g] = (int(model.geom_contype[g]), int(model.geom_conaffinity[g]))
            if x["has_stop"]:
                for g in range(model.ngeom):
                    if int(model.geom_bodyid[g]) == x["bc"] and (model.geom_contype[g] or model.geom_conaffinity[g]):
                        obstacles.add(g)
        self.stop_obstacles = sorted(obstacles - stop_geoms)
        self.stop_off = set()           # stop geoms whose collision this machine switched off
        self.stop_pending = set()       # ... released, waiting to be clear of every hub
        self.state = {x["id"]: "WAITING" for x in self.slots}
        self.rec = {x["id"]: {"id": x["id"], "kind": x["kind"], "weight": x["weight"], "unit": None, "fastened": False, "design_ok": None,
                              "achieved": False, "void": None, "t_s": None, "pos_err_mm": None, "ang_err_deg": None,
                              "axial_err_mm": None, "key_k": None,
                              "scored": x["scored"], "unit_type": None, "hand_ok": None, "key_ok": None} for x in self.slots}
        for x in self.slots:
            if x["site"] is not None:
                self.rec[x["id"]]["site"] = dict(x["site"])
        self.hold = {x["id"]: 0 for x in self.slots}
        self.bound = {}                 # slot -> candidate index (captured)
        self.used = {}                  # unit -> slot that fastened or holds it, or SCRAPPED
        self.scrapped = {}              # unit -> {"t_s", "reason"}
        self.premounted = []            # slot ids fastened by the cell at the start (not scored)
        self.events = []
        # (3) the rest gate's counters: (slot, candidate index) -> (consecutive ticks inside and still, tick index)
        self._still = {}
        self._tick_n = 0
        # the weld constraint forces: per slot, the last `weld_window_ticks` ticks (ring), maxima
        self.track_weld_force = bool(track_weld_force)
        self._wwin = max(1, int(weld_window_ticks))
        self._eq_all = np.array([c["eq"] for x in self.slots for c in x["cands"]], int)
        self._eq_where = [(i, ci) for i, x in enumerate(self.slots) for ci in range(len(x["cands"]))]
        self._wbuf = np.zeros((len(self.slots), self._wwin))
        self._wk = 0
        self._wseen = 0
        self.weld_stats = {x["id"]: {"F_max_N": 0.0, "F_sustained_max_N": 0.0, "T_max_Nm": 0.0,
                                     "soft_F_max_N": 0.0, "hard_F_max_N": 0.0} for x in self.slots}
        self._weld_now = {}
        self.total_w = self._total_weight()
        for sid in premounted:
            self.premount(sid)

    # ------------------------------------------------------------------ structure
    def _gate_of(self, sid, cap, kit_gate):
        """(speed m/s, rate rad/s, ticks) of a keyed slot's rest gate: the slot's capture.still_*; a kit-level
        capture_gate must say the same (two copies of one rule must not drift); one of the two must be there"""
        own = all(k in cap for k in ("still_mm_s", "still_deg_s", "still_ticks"))
        if any(k in cap for k in ("still_mm_s", "still_deg_s", "still_ticks")) and not own:
            raise ValueError("kit: slot %s gives only part of capture.still_mm_s / still_deg_s / still_ticks" % sid)
        g = None
        if own:
            g = (_finite(cap["still_mm_s"], "slot %s still_mm_s" % sid), _finite(cap["still_deg_s"], "slot %s still_deg_s" % sid),
                 int(cap["still_ticks"]))
        if kit_gate is not None:
            kg = (_finite(kit_gate["rel_speed_mm_s"], "capture_gate.rel_speed_mm_s"),
                  _finite(kit_gate["rel_rate_deg_s"], "capture_gate.rel_rate_deg_s"), int(kit_gate["ticks"]))
            if g is None:
                g = kg
            elif abs(g[0] - kg[0]) > 1e-9 or abs(g[1] - kg[1]) > 1e-9 or g[2] != kg[2]:
                raise ValueError("kit: slot %s capture.still_* %s differ from the kit's capture_gate %s" % (sid, g, kg))
        if g is None:
            raise ValueError("kit: keyed slot %s has no capture.still_* and the kit no capture_gate (rest gate on)" % sid)
        if not (g[0] > 0.0 and g[1] > 0.0 and g[2] >= 1):
            raise ValueError("kit: slot %s rest gate %s is not positive" % (sid, g))
        return (g[0] * 1e-3, math.radians(g[1]), int(g[2]))

    def _total_weight(self):
        return sum(x["weight"] for x in self.slots if x["scored"] and x["id"] not in self.premounted)

    def _unit_of(self, body_name):
        u = self.body_unit.get(body_name)
        if u is None:
            u = body_name if body_name in self.loose_set else "base"
        return u

    def _fastened_slot_of(self, unit):
        for x in self.slots:
            st = self.rec[x["id"]]
            if st["fastened"] and st["unit"] == unit:
                return x["id"]
        return None

    def _scrapped_in_chain(self, x):
        """the first scrapped unit on the slot's carrier chain (its carrier's unit, the unit that one is fastened on,
        ... down to the base), or None"""
        u, seen = x["carrier_unit"], set()
        while u in self.loose_set and u not in seen:
            seen.add(u)
            if u in self.scrapped:
                return u
            s2 = self._fastened_slot_of(u)
            if s2 is None:
                return None
            u = self.by_id[s2]["carrier_unit"]
        return None

    def _chain_fastened(self, x):
        """every loose unit on the slot's carrier chain is fastened (counting, or premounted) down to the base"""
        u, seen = x["carrier_unit"], set()
        while u in self.loose_set:
            if u in seen:
                return False
            seen.add(u)
            s2 = self._fastened_slot_of(u)
            if s2 is None or not (s2 in self.premounted or self._ok_with_chain(s2)):
                return False
            u = self.by_id[s2]["carrier_unit"]
        return True

    # ------------------------------------------------------------------ score
    def _mate_ok(self, sid):
        """fastened in a counting orientation (and part type), not void, not premounted"""
        st = self.rec[sid]
        return not (sid in self.premounted or not st["fastened"] or not st["design_ok"] or st["void"])

    def _ok_with_chain(self, sid):
        return self._mate_ok(sid) and (self.count_subassembly or self._chain_fastened(self.by_id[sid]))

    def counted(self, sid):
        """the slot's mate counts in the structural score: fastened in a counting orientation and part type, not void,
        not premounted, a scored slot (and, with count_subassembly off, its carrier chain fastened to the base)"""
        return bool(self.by_id[sid]["scored"]) and self._ok_with_chain(sid)

    def site_holds(self, sid):
        """an unscored site holds a unit (fastened, counting orientation, not void): what the trim reads"""
        return (not self.by_id[sid]["scored"]) and self._ok_with_chain(sid)

    def score(self):
        """the structural score: counted weight / total weight of the scored slots"""
        return sum(x["weight"] for x in self.slots if self.counted(x["id"])) / self.total_w if self.total_w > 0 else 0.0

    def weights(self):
        """(counted weight, total weight) of the scored slots (the grader adds the trim to both)"""
        return float(sum(x["weight"] for x in self.slots if self.counted(x["id"]))), float(self.total_w)

    def _refresh(self):
        for x in self.slots:
            sid = x["id"]
            self.rec[sid]["achieved"] = self.counted(sid) if x["scored"] else self.site_holds(sid)

    def _vel(self, b):
        import mujoco
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
        return v

    # ------------------------------------------------------------------ predicates
    def _pair_dist(self):
        """the distance between the two slot points of every (slot, candidate) pair (m), from the current kinematics"""
        if not len(self._pbc):
            return np.zeros(0)
        xpos = np.asarray(self.d.xpos)
        xmat = np.asarray(self.d.xmat).reshape(-1, 3, 3)
        a = np.einsum("nij,nj->ni", xmat[self._pbc], self._pa) + xpos[self._pbc]
        b = np.einsum("nij,nj->ni", xmat[self._pbp], self._pb) + xpos[self._pbp]
        return np.sqrt(((b - a) ** 2).sum(axis=1))

    def _near(self, x, c, release=False):
        """necessary condition of `_inside`: the part's slot point within the window's reach of the carrier's (from
        this tick's `_pair_dist`, or measured now outside a tick)"""
        dist = self._dist if self._dist is not None else self._pair_dist()
        return float(dist[c["pair"]]) <= (x["rel_reach"] if release else x["reach"])

    def evaluate(self, x, c):
        d = self.d
        Tc, Tp = body_pose(d, x["bc"]), body_pose(d, c["bp"])
        hand_ok = x["design_part"] is None or c["type"] == x["design_part"]
        if x["kind"] == "seat":
            # at the seat's frame on the part (`frame_in_part`), not at the body's origin: a tilt of the part must not
            # count as a shift of a far-away frame origin
            A, B = Tc @ c["Tpc"] @ c["Fp"], Tp @ c["Fp"]
            E = _inv(A) @ B
            dw = B[:3, 3] - A[:3, 3]
            ins_w = Tc[:3, :3] @ x["ins_c"]
            axial = float(dw @ ins_w)                        # < 0: short of home along the insertion
            r = {"pos_err": float(np.linalg.norm(E[:3, 3])), "ang_err": rotation_angle(E[:3, :3]), "axial": axial,
                 "lateral": float(np.linalg.norm(dw - axial * ins_w))}
            r["holds"] = r["pos_err"] <= x["tol_pos"] and r["ang_err"] <= x["tol_ang"]
            r["key_ok"] = True
        else:
            A, B = Tc @ x["Fc"], Tp @ c["Fp"]
            za = A[:3, 2]
            ang = math.acos(max(-1.0, min(1.0, float(za @ B[:3, 2]))))
            dv = B[:3, 3] - A[:3, 3]
            axial = float(dv @ za)                           # the part's frame past the carrier's: 0 at home, < 0 short
            line = float(np.linalg.norm(dv - axial * za))
            qe = quat_wxyz((_inv(A) @ B)[:3, :3])
            clock = wrap(2.0 * math.atan2(float(qe[3]), float(qe[0])))
            k = int(round(clock / x["pitch"]))
            res = wrap(clock - k * x["pitch"])
            nk = int(round(2 * math.pi / x["pitch"]))
            r = {"pos_err": line, "ang_err": ang, "axial": axial, "clock": clock, "k": k % nk, "clock_res": res,
                 "holds": ang <= x["tol_ang"] and line <= x["tol_pos"] and abs(axial) <= x["axial_tol"] and abs(res) <= x["tol_ang"]}
            r["key_ok"] = bool(x.get("any_key")) or (k - int(x.get("design_k", 0))) % nk == 0
        # (1) the orientation key and the part type
        r["hand_ok"] = bool(hand_ok)
        r["design_ok"] = bool(r["key_ok"] and hand_ok) if x["kind"] == "keyed" else bool(hand_ok)
        if r["holds"]:
            vc, vp = self._vel(x["bc"]), self._vel(c["bp"])
            r["rel_speed"] = float(np.linalg.norm(vp[3:] - vc[3:]))
            r["rel_ang_speed"] = float(np.linalg.norm(vp[:3] - vc[:3]))
        return r

    def _inside(self, x, r, release=False):
        """inside the slot's capture window; `release` = the window a captured unit has to leave to be released
        (release_factor x the window; a key: its tip key_release_clear_m clear of the carrier's face). Pure: the rest
        gate is applied in `_tick`, not here."""
        f = self.release_factor if release else 1.0
        lo = -(x["rel_depth"] if release else x["cap_depth"])
        if not (lo <= r["axial"] <= x["cap_past"] * f and r["ang_err"] <= x["cap_tilt"] * f):
            return False
        if x["kind"] == "seat":
            return r["lateral"] <= x["cap_lat"] * f
        return r["pos_err"] <= x["cap_lat"] * f and abs(r["clock_res"]) <= x["cap_clock"] * f

    def extent_points(self, b):
        """(n x 3) the farthest points of body b in its own frame (cached): the 8 corners of the local bounding box
        (`geom_aabb`) of every geom of the body, whatever its group or collision bits (the visual meshes are the
        unit's real shape, the collision geoms lie inside them). A body without any geom (a test stand-in): the
        corners of the box with its mass and principal inertia about its centre of mass."""
        P = self._extent.get(b)
        if P is not None:
            return P
        import mujoco
        m = self.m
        signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
        pts = []
        R = np.zeros(9)
        for g in np.nonzero(np.asarray(m.geom_bodyid) == b)[0]:
            a = np.asarray(m.geom_aabb[g], float)
            mujoco.mju_quat2Mat(R, np.asarray(m.geom_quat[g], float))
            pts.append((a[:3] + signs * a[3:]) @ R.reshape(3, 3).T + np.asarray(m.geom_pos[g], float))
        if pts:
            P = np.concatenate(pts)
            src = "geoms"
        else:
            I = np.asarray(m.body_inertia[b], float)
            mass = float(m.body_mass[b])
            h = np.sqrt(np.maximum(0.0, 1.5 / mass * (I.sum() - 2.0 * I))) if mass > 0 else np.zeros(3)
            mujoco.mju_quat2Mat(R, np.asarray(m.body_iquat[b], float))
            P = (signs * h) @ R.reshape(3, 3).T + np.asarray(m.body_ipos[b], float)
            src = "inertia box"
        self._extent[b] = P
        self.extent_source = dict(getattr(self, "extent_source", {}), **{m.body(int(b)).name: src})
        return P

    def gate_speeds(self, x, c):
        """(speed m/s, turn rate rad/s) of candidate c relative to slot x's carrier as the rest gate reads them:
        gate_point "extent" = the largest speed relative to the carrier over the unit's extent points (every point
        of the unit), "com" = the centre of mass's; the turn rate = |w_unit - w_carrier| either way"""
        vc, vp = self._vel(x["bc"]), self._vel(c["bp"])
        w = float(np.linalg.norm(vp[:3] - vc[:3]))
        if self.gate_point == "com":
            return float(np.linalg.norm(vp[3:] - vc[3:])), w
        d = self.d
        bp, bc = c["bp"], x["bc"]
        P = np.asarray(d.xpos[bp]) + self.extent_points(bp) @ np.asarray(d.xmat[bp]).reshape(3, 3).T
        v_p = vp[3:] + np.cross(vp[:3], P - np.asarray(d.xipos[bp]))
        v_c = vc[3:] + np.cross(vc[:3], P - np.asarray(d.xipos[bc]))
        return float(np.sqrt(((v_p - v_c) ** 2).sum(axis=1)).max()), w

    def _gate(self, x, key, c, inside):
        """(3) the rest gate of one (slot, candidate): count the consecutive ticks on which the candidate is inside the
        capture window and at rest relative to the carrier (`gate_speeds` below the gate's two bounds; a tick without
        an evaluation, or outside, or too fast, starts the count again); True once the count reaches the gate's ticks"""
        v_max, w_max, n_need = x["gate"]
        n, last = self._still.get(key, (0, -2))
        if last != self._tick_n - 1:
            n = 0
        ok = False
        if inside:
            v, w = self.gate_speeds(x, c)
            ok = v < v_max and w < w_max
        n = n + 1 if ok else 0
        self._still[key] = (n, self._tick_n)
        return n >= n_need

    def gate_counts(self):
        """{(slot, unit): consecutive still ticks inside the window} of the counters updated on the last tick"""
        out = {}
        for (sid, i), (n, t) in self._still.items():
            if t == self._tick_n:
                out[(sid, self.by_id[sid]["cands"][i]["unit"])] = n
        return out

    # ------------------------------------------------------------------ welds
    def _target(self, x, c, r):
        """the unit's pose in the carrier's frame: a seat at its design, a key home on the nearest key orientation"""
        if x["kind"] == "seat":
            return c["Tpc"]
        return x["Fc"] @ _rotz(r["k"] * x["pitch"]) @ _inv(c["Fp"])

    def _pull_target(self, x, c, r, sim_time):
        """the soft weld's target: from the pose the part had at its capture towards home at <= pull_speed at the weld
        point and <= pull_rate in angle. None: a seat that is still settling (no weld yet)"""
        from scipy.spatial.transform import Rotation, Slerp
        sid = x["id"]
        home = self._target(x, c, r)
        if sid not in self.pull or self.pull[sid]["unit"] != c["unit"]:
            T0 = _inv(body_pose(self.d, x["bc"])) @ body_pose(self.d, c["bp"])        # the part in the carrier now
            a = c["Fp"][:3, 3]
            if x["kind"] == "seat":
                # a seat's pull starts once the part has settled on its carrier, from a pose not past home along the
                # insertion (the source cell: a weld engaged on a part still falling onto its bay asked kN for a tick)
                vc, vp = self._vel(x["bc"]), self._vel(c["bp"])
                if (float(np.linalg.norm(vp[3:] - vc[3:])) > self.seat_settle_v or
                        float(np.linalg.norm(vp[:3] - vc[:3])) > self.seat_settle_w):
                    return None
                past = float(((T0[:3, :3] @ a + T0[:3, 3]) - (home[:3, :3] @ a + home[:3, 3])) @ x["ins_c"])
                if past > 0.0:
                    T0[:3, 3] -= past * x["ins_c"]
            dist = float(np.linalg.norm((T0[:3, :3] @ a + T0[:3, 3]) - (home[:3, :3] @ a + home[:3, 3])))
            ang = rotation_angle(T0[:3, :3].T @ home[:3, :3])
            dur = max(dist / self.pull_v, ang / self.pull_w, 1e-3)
            self.pull[sid] = {"unit": c["unit"], "T0": T0, "t0": float(sim_time), "dur": dur}
        p = self.pull[sid]
        s_ = min(1.0, (float(sim_time) - p["t0"]) / p["dur"])
        if s_ >= 1.0:
            return home
        T0 = p["T0"]
        a = c["Fp"][:3, 3]
        R = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], home[:3, :3]]))([s_]).as_matrix()[0]
        # the weld point moves on the straight line, the orientation slerps
        pa = (1 - s_) * (T0[:3, :3] @ a + T0[:3, 3]) + s_ * (home[:3, :3] @ a + home[:3, 3])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pa - R @ a
        return T

    def _weld(self, c, rel, soft):
        e = c["eq"]
        # the weld point at the slot's frame on the part (the key's root, the seat), not at the part's body origin: a soft
        # weld then pulls the feature home about the feature. MuJoCo at run time: body2's point = data[0:3] (body2
        # frame), body1's point = data[3:6] as given (body1 frame), so data[3:6] is that point's home in the carrier's
        # frame
        a2 = c["Fp"][:3, 3]
        self.m.eq_data[e, 0:3] = a2
        self.m.eq_data[e, 3:6] = rel[:3, 3] + rel[:3, :3] @ a2
        self.m.eq_data[e, 6:10] = quat_wxyz(rel[:3, :3])
        self.m.eq_data[e, 10] = 1.0
        self.m.eq_solref[e] = self.soft_solref if soft else self.hard_solref[e]
        self.d.eq_active[e] = 1

    def _release(self, x, ci, sim_time=None, why=None):
        self.pull.pop(x["id"], None)
        c = x["cands"][ci]
        self.d.eq_active[c["eq"]] = 0
        self.m.eq_solref[c["eq"]] = self.hard_solref[c["eq"]]
        self.bound.pop(x["id"], None)
        self.hold[x["id"]] = 0
        if self.used.get(c["unit"]) == x["id"]:
            self.used.pop(c["unit"], None)
        if why is not None:
            self.events.append((sim_time, "released from capture %s (%s): %s" % (x["id"], c["unit"], why)))
        self._stop_release(c, sim_time)

    def weld_forces(self):
        """{slot: {"unit", "soft", "F_N", "T_Nm"}} of every active weld of this machine, from the constraint forces of
        the last physics computation (efc_force; 0 for a weld switched on since). Not contact: H2 does not count it."""
        import mujoco
        out = {}
        d = self.d
        act = np.nonzero(np.asarray(d.eq_active)[self._eq_all] != 0)[0]
        if not len(act):
            return out
        n = int(d.nefc)
        rows = {}
        if n:
            sel = np.nonzero(np.asarray(d.efc_type[:n]) == mujoco.mjtConstraint.mjCNSTR_EQUALITY)[0]
            ids = np.asarray(d.efc_id[:n])[sel]
            for r_, e_ in zip(sel.tolist(), ids.tolist()):
                rows.setdefault(int(e_), []).append(r_)
        for j in act.tolist():
            i, ci = self._eq_where[j]
            x = self.slots[i]
            e = int(self._eq_all[j])
            rr = rows.get(e, [])
            fv = np.asarray(d.efc_force[rr], float) if rr else np.zeros(0)
            F = float(np.linalg.norm(fv[:3])) if len(fv) else 0.0
            T = float(np.linalg.norm(fv[3:6])) if len(fv) >= 6 else 0.0
            soft = bool(np.allclose(self.m.eq_solref[e], self.soft_solref)) and not self.rec[x["id"]]["fastened"]
            out[x["id"]] = {"unit": x["cands"][ci]["unit"], "soft": soft, "F_N": F, "T_Nm": T}
        return out

    def _track_welds(self):
        now = self.weld_forces()
        self._weld_now = now
        col = np.zeros(len(self.slots))
        for i, x in enumerate(self.slots):
            w = now.get(x["id"])
            if w is None:
                continue
            col[i] = w["F_N"]
            ws = self.weld_stats[x["id"]]
            ws["F_max_N"] = max(ws["F_max_N"], w["F_N"])
            ws["T_max_Nm"] = max(ws["T_max_Nm"], w["T_Nm"])
            k = "soft_F_max_N" if w["soft"] else "hard_F_max_N"
            ws[k] = max(ws[k], w["F_N"])
        self._wbuf[:, self._wk] = col
        self._wk = (self._wk + 1) % self._wwin
        self._wseen += 1
        if self._wseen >= self._wwin:
            sus = self._wbuf.min(axis=1)
            for i, x in enumerate(self.slots):
                if sus[i] > self.weld_stats[x["id"]]["F_sustained_max_N"]:
                    self.weld_stats[x["id"]]["F_sustained_max_N"] = float(sus[i])

    # ------------------------------------------------------------------ the wheel stop (no slot of this kit has one)
    def _bits(self, g):
        if self.collision_bits is not None and g in self.collision_bits:
            ct, ca = self.collision_bits[g]
            return int(ct), int(ca)
        return self.stop_bits[g]

    def _stop_set_off(self, c, sim_time=None):
        for g in c["stop"]:
            self.stop_pending.discard(g)
            if g not in self.stop_off:
                self.stop_off.add(g)
                self.m.geom_contype[g] = 0
                self.m.geom_conaffinity[g] = 0
                self.events.append((sim_time, "stop off (%s)" % c["unit"]))

    def _stop_release(self, c, sim_time=None):
        for g in c["stop"]:
            if g in self.stop_off:
                self.stop_pending.add(g)
        self._restore_stops(sim_time)

    def _stop_is_clear(self, g):
        import mujoco
        dmax = self.stop_clear + 1e-3
        for h in self.stop_obstacles:
            if mujoco.mj_geomDistance(self.m, self.d, g, h, dmax, None) <= self.stop_clear:
                return False
        return True

    def _restore_stops(self, sim_time=None):
        """a released unit's stop gets its collision back only once it is clear of every carrier geom"""
        for g in sorted(self.stop_pending):
            if self._stop_is_clear(g):
                ct, ca = self._bits(g)
                self.m.geom_contype[g], self.m.geom_conaffinity[g] = ct, ca
                self.stop_pending.discard(g)
                self.stop_off.discard(g)
                self.events.append((sim_time, "stop on (%s)" % self.m.geom(g).name))

    def stop_state(self):
        """{stop geom name: "on" | "off" | "pending"}"""
        out = {}
        for g in self.stop_bits:
            out[self.m.geom(g).name] = "pending" if g in self.stop_pending else ("off" if g in self.stop_off else "on")
        return out

    # ------------------------------------------------------------------ scrapping
    def scrap(self, unit, sim_time=None, reason="scrapped"):
        """take a loose unit out of the cell's coupling for good (off the bench, faster than the part-speed rule, on the
        vehicle unfastened when a turn of the positioner starts): release its capture first, then take it off every
        candidate list; void the mates it or a unit it carries make. A fastened unit stays welded where it is.
        Returns False if it was scrapped before."""
        if unit in self.scrapped:
            return False
        if unit not in self.loose_set:
            raise ValueError("scrap: %s is not a loose unit" % unit)
        for x in self.slots:
            ci = self.bound.get(x["id"])
            if ci is not None and x["cands"][ci]["unit"] == unit:
                self._release(x, ci, sim_time, "scrapped")
        self.used[unit] = SCRAPPED
        self.scrapped[unit] = {"t_s": None if sim_time is None else float(sim_time), "reason": str(reason)}
        self.events.append((sim_time, "SCRAPPED %s (%s)" % (unit, reason)))
        self._void_pass(sim_time)
        return True

    def _void_pass(self, sim_time=None):
        for x in self.slots:
            sid = x["id"]
            st = self.rec[sid]
            if sid in self.premounted or st["void"]:
                continue
            why = None
            if st["fastened"] and st["unit"] in self.scrapped:
                why = "unit %s scrapped" % st["unit"]
            else:
                cu = self._scrapped_in_chain(x)
                if cu is not None:
                    why = "carrier %s scrapped" % cu
            if why is None:
                continue
            ci = self.bound.get(sid)
            if ci is not None:
                self._release(x, ci, sim_time, why)
            st["void"] = why
            self.state[sid] = "VOID"
            self.hold[sid] = 0
            self.events.append((sim_time, "VOID %s: %s" % (sid, why)))
        self._refresh()

    # ------------------------------------------------------------------ premounted slots
    def premount(self, sid, unit=None, place=True, sim_time=0.0):
        """fasten `unit` (default: the slot's designed unit, else its first candidate of the slot's design part) on its
        slot at its design pose, rigidly, before the episode (a gate's fallback); the slot leaves the total weight.
        `place` moves a free unit there first."""
        import mujoco
        x = self.by_id[sid]
        if unit is None:
            unit = x.get("designed_unit")
            if unit is None and x["design_part"] is not None:
                unit = [c for c in x["cands"] if c["type"] == x["design_part"]][0]["unit"]
            if unit is None:
                unit = x["cands"][0]["unit"]
        c = [c for c in x["cands"] if c["unit"] == unit][0]
        if unit in self.used or self.rec[sid]["fastened"] or sid in self.bound:
            raise ValueError("premount: %s or %s is taken" % (sid, unit))
        if place:
            m, d = self.m, self.d
            mujoco.mj_kinematics(m, d)
            T = body_pose(d, x["bc"]) @ c["Tpc"]
            j = int(m.body_jntadr[c["bp"]])
            if j < 0 or m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError("premount: %s has no free joint to place" % unit)
            q, v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            d.qpos[q:q + 3] = T[:3, 3]
            d.qpos[q + 3:q + 7] = quat_wxyz(T[:3, :3])
            d.qvel[v:v + 6] = 0.0
            mujoco.mj_kinematics(m, d)
        self._weld(c, c["Tpc"], soft=False)
        self._stop_set_off(c, sim_time)
        hand_ok = x["design_part"] is None or c["type"] == x["design_part"]
        self.rec[sid].update({"fastened": True, "design_ok": bool(hand_ok), "unit": unit, "t_s": float(sim_time), "pos_err_mm": 0.0,
                              "ang_err_deg": 0.0, "axial_err_mm": 0.0, "unit_type": c["type"], "hand_ok": bool(hand_ok),
                              "key_ok": True})
        self.used[unit] = sid
        self.premounted.append(sid)
        self.state[sid] = "PREMOUNTED"
        self.total_w = self._total_weight()
        self.events.append((sim_time, "PREMOUNTED %s (%s)" % (sid, unit)))
        self._refresh()

    # ------------------------------------------------------------------ per control tick
    def tick(self, sim_time):
        self._tick_n += 1
        if self.stop_pending:
            self._restore_stops(sim_time)
        self._dist = self._pair_dist()
        try:
            self._tick(sim_time)
        finally:
            self._dist = None
        self._refresh()
        if self.track_weld_force:
            self._track_welds()

    def _tick(self, sim_time):
        for x in self.slots:
            sid = x["id"]
            st = self.rec[sid]
            if sid in self.premounted:
                self.state[sid] = "PREMOUNTED"
                continue
            if st["fastened"]:
                self.state[sid] = "VOID" if st["void"] else ("LOCKED" if st["design_ok"] else "LOCKED_OFF")
                continue
            if st["void"]:
                self.state[sid] = "VOID"
                continue
            if not all(q in self.premounted or self.counted(q) or self.site_holds(q) for q in x["requires"]):
                if sid in self.bound:
                    self._release(x, self.bound[sid], sim_time, "a required slot is not achieved")
                self.state[sid] = "WAITING"
                self.hold[sid] = 0
                continue
            ci = self.bound.get(sid)
            r = None
            if ci is not None:
                c = x["cands"][ci]
                r = self.evaluate(x, c) if self._near(x, c, release=True) else None
                if r is None or not self._inside(x, r, release=True):
                    self._release(x, ci, sim_time, "left the window")
                    ci, r = None, None
            if ci is None:
                best = None
                for i, c in enumerate(x["cands"]):
                    if c["unit"] in self.used or not self._near(x, c):
                        self._still.pop((sid, i), None)
                        continue
                    rr = self.evaluate(x, c)
                    inside = self._inside(x, rr)
                    if x["gate"] is not None:
                        inside = self._gate(x, (sid, i), c, inside)
                    if inside and (best is None or rr["pos_err"] < best[1]["pos_err"]):
                        best = (i, rr)
                if best is None:
                    self.state[sid] = "FREE"
                    self.hold[sid] = 0
                    continue
                ci, r = best
                c = x["cands"][ci]
                self.bound[sid] = ci
                self.used[c["unit"]] = sid
                for i in range(len(x["cands"])):
                    self._still.pop((sid, i), None)
                self.events.append((sim_time, "ENGAGED %s (%s)" % (sid, c["unit"])))
                self._stop_set_off(c, sim_time)
            c = x["cands"][ci]
            T_pull = self._pull_target(x, c, r, sim_time)
            if T_pull is None:                  # a captured seat still settling: no weld yet
                self.d.eq_active[c["eq"]] = 0
                self.state[sid] = "ENGAGED"
                self.hold[sid] = 0
                continue
            self._weld(c, T_pull, soft=True)
            still = r["holds"] and r.get("rel_speed", 1.0) < self.rel_speed_max and r.get("rel_ang_speed", 1.0) < self.rest_ang
            if not still:
                self.state[sid] = "ENGAGED"
                self.hold[sid] = 0
                continue
            self.hold[sid] += 1
            self.state[sid] = "HOLDING"
            if self.hold[sid] >= self.hold_ticks:
                self._lock(x, ci, r, sim_time)

    def _lock(self, x, ci, r, sim_time):
        sid = x["id"]
        self.pull.pop(sid, None)
        c = x["cands"][ci]
        st = self.rec[sid]
        st.update({"fastened": True, "t_s": float(sim_time), "unit": c["unit"], "pos_err_mm": r["pos_err"] * 1e3,
                   "ang_err_deg": math.degrees(r["ang_err"]), "axial_err_mm": r.get("axial", 0.0) * 1e3,
                   "design_ok": bool(r["design_ok"]), "unit_type": c["type"], "hand_ok": bool(r["hand_ok"]),
                   "key_ok": bool(r["key_ok"])})
        if x["kind"] == "keyed":
            st["key_k"] = int(r["k"])
        self._weld(c, self._target(x, c, r), soft=False)
        self.bound.pop(sid, None)
        self.used[c["unit"]] = sid
        self.state[sid] = "LOCKED" if st["design_ok"] else "LOCKED_OFF"
        if st["design_ok"]:
            msg = "LOCKED %s (%s)" % (sid, c["unit"])
        else:
            msg = "LOCKED_OFF %s (%s, key orientation %d, design %d)" % (sid, c["unit"], r["k"], x.get("design_k", 0)) \
                if x["kind"] == "keyed" else "LOCKED_OFF %s (%s)" % (sid, c["unit"])
            if not r["hand_ok"]:
                msg += "; part %s, design part %s" % (c["type"], x["design_part"])
        self.events.append((sim_time, msg))

    # ------------------------------------------------------------------ reports
    def record(self):
        """one entry per slot (premounted slots are left out: they are not in the total weight); `achieved` = the mate
        counts now (a scored slot: in the structural score; an unscored site: it holds a fastened unit, for the trim);
        unscored sites carry `scored: false` and their `site` record"""
        self._refresh()
        out = []
        for x in self.slots:
            if x["id"] in self.premounted:
                continue
            r = dict(self.rec[x["id"]])
            r["state"] = self.state[x["id"]]
            out.append(r)
        return out

    def trim_sites(self):
        """the unscored sites that hold a fastened unit now (what the grader's trim reads): slot, unit, the site record
        (station, name_03, hole_B_m, conflicts_with), the key orientation, the time it was fastened"""
        self._refresh()
        out = []
        for x in self.slots:
            if x["scored"] or not self.site_holds(x["id"]):
                continue
            st = self.rec[x["id"]]
            e = {"slot": x["id"], "unit": st["unit"], "unit_type": st["unit_type"], "key_k": st["key_k"], "t_s": st["t_s"]}
            e.update(dict(x["site"] or {}))
            out.append(e)
        return out

    def site_conflicts(self):
        """pairs of unscored sites that both hold a unit although the kit says they exclude each other (cannot happen
        with solid blocks; a sanity check of the cell)"""
        holding = set(e["slot"] for e in self.trim_sites())
        out = set()
        for sid in holding:
            for o in (self.by_id[sid]["site"] or {}).get("conflicts_with", []) or []:
                if o in holding:
                    out.add(tuple(sorted((sid, o))))
        return sorted(out)

    def metrics(self):
        rec = [self.rec[x["id"]] for x in self.slots if x["id"] not in self.premounted and x["scored"]]
        out = {"locked": sum(1 for r in rec if r["fastened"] and r["design_ok"] and not r["void"]),
               "locked_off": sum(1 for r in rec if r["fastened"] and r["design_ok"] is False),
               "void_unit_scrapped": sum(1 for r in rec if (r["void"] or "").startswith("unit ")),
               "void_carrier_scrapped": sum(1 for r in rec if (r["void"] or "").startswith("carrier ")),
               "scrapped_units": len(self.scrapped), "premounted": len(self.premounted),
               "weight_total": float(self.total_w), "weight_counted": float(sum(x["weight"] for x in self.slots if self.counted(x["id"])))}
        sites = [x for x in self.slots if not x["scored"]]
        out.update({
            "locked_off_part_type": sum(1 for r in rec if r["fastened"] and r["design_ok"] is False and r["hand_ok"] is False),
            "locked_off_key": sum(1 for r in rec if r["fastened"] and r["design_ok"] is False and r["key_ok"] is False),
            "sites_total": len(sites),
            "sites_holding": sum(1 for x in sites if self.site_holds(x["id"])),
            "sites_void": sum(1 for x in sites if self.rec[x["id"]]["void"]),
            "site_conflicts": len(self.site_conflicts()),
            "weld_F_max_N": float(max([w["F_max_N"] for w in self.weld_stats.values()] or [0.0])),
            "weld_F_sustained_max_N": float(max([w["F_sustained_max_N"] for w in self.weld_stats.values()] or [0.0])),
            "weld_soft_F_max_N": float(max([w["soft_F_max_N"] for w in self.weld_stats.values()] or [0.0])),
            "weld_T_max_Nm": float(max([w["T_max_Nm"] for w in self.weld_stats.values()] or [0.0]))})
        return out

    def capture_table(self):
        """the slots' windows in mm / deg for the cell's documents (the single source of the manual's capture table):
        capture, release, lock, rest, pull, and per slot whether the unit stands by itself or must be held still
        before release (and for how long), whether it must rest before it is captured (the rest gate), whether the
        key is physical, how many orientations the place takes and whether the part type is checked. No design pose,
        unit name, part type or hand is in it."""
        out = []
        f = self.release_factor
        for x in self.slots:
            e = {"slot": x["id"], "kind": x["kind"], "carrier": x["carrier"],
                 "weight": x["weight"], "premounted": x["id"] in self.premounted,
                 "capture": {"lateral_mm": x["cap_lat"] * 1e3, "from_home_mm": x["cap_depth"] * 1e3,
                             "past_home_mm": x["cap_past"] * 1e3, "tilt_deg": math.degrees(x["cap_tilt"])},
                 "release": {"lateral_mm": x["cap_lat"] * f * 1e3, "from_home_mm": x["rel_depth"] * 1e3,
                             "tilt_deg": math.degrees(x["cap_tilt"] * f)},
                 "lock": {"pos_mm": x["tol_pos"] * 1e3, "ang_deg": math.degrees(x["tol_ang"]), "hold_ticks": self.hold_ticks,
                          "rel_speed_mm_s": self.rel_speed_max * 1e3, "rel_ang_speed_deg_s": math.degrees(self.rest_ang)},
                 "pull": {"speed_mm_s": self.pull_v * 1e3, "rate_deg_s": math.degrees(self.pull_w)}}
            if x["kind"] == "seat":
                e["pull"].update({"starts_below_mm_s": self.seat_settle_v * 1e3,
                                  "starts_below_deg_s": math.degrees(self.seat_settle_w)})
            if x["kind"] == "keyed":
                e["capture"]["clock_deg"] = math.degrees(x["cap_clock"])
                e["capture"]["key_frac"] = x["key_frac"]
                e["release"]["clock_deg"] = math.degrees(x["cap_clock"] * f)
                e["lock"]["axial_mm"] = x["axial_tol"] * 1e3
                e["key_height_mm"] = x["key_len"] * 1e3
                e["orientations"] = {"pitch_deg": math.degrees(x["pitch"]), "any_key": bool(x.get("any_key")),
                                     "design_k": int(x.get("design_k", 0))}
            if x["has_stop"]:
                e["stop"] = dict(x.get("stop") or {})
            # this cell's columns
            e["scored"] = bool(x["scored"])
            e["site"] = x["site"] is not None
            e["self_standing"] = x["self_standing"]
            e["hold_before_release_s"] = None if x["self_standing"] is not False else float(x["hold_s"])
            g = x["gate"]
            e["still_before_capture"] = g is not None
            e["capture"].update({"still_mm_s": None if g is None else g[0] * 1e3,
                                 "still_deg_s": None if g is None else math.degrees(g[1]),
                                 "still_ticks": None if g is None else int(g[2]),
                                 "still_s": None if g is None else int(g[2]) * self.tick_s,
                                 "still_speed_of": None if g is None else (
                                     "every point of the unit" if self.gate_point == "extent" else "the unit's centre of mass")})
            if x["kind"] == "keyed":
                e["physical_key"] = not x["key_virtual"]
                e["orientations"]["count"] = int(round(2 * math.pi / x["pitch"]))
                e["orientations"]["part_type_checked"] = x["design_part"] is not None
            out.append(e)
        return out
