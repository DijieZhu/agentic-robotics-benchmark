"""Coupling state machine of the race-car assembly cell: one state per SLOT of the kit (`kit.json` "slots"), each slot
taking whichever of its candidate units reaches it (the four wheels are rigid copies of one part: any wheel on any hub).

Ported from the quadruped cell's coupling (`arm/assembly_quad/coupling.py`): the same predicates, soft weld, pull home,
rest and lock. Changed for this cell:
  * a capture window per slot (`slot["capture"]`: lateral, from home, tilt and clock separately; the quadruped had one
    global window whose tilt and clock shared one angle);
  * scrapping (`scrap(unit)`): a loose unit off the bench or faster than the part-speed rule (the cell's watch decides
    and calls it) is first released from the capture it is in, then taken off every candidate list for good; a mate
    whose unit, or a loose unit in its carrier chain, is scrapped counts nothing (the deck falling off the bench voids
    the compute unit and the LiDAR fastened on it), and a slot whose carrier is scrapped can no longer capture;
  * the wheel stop (`slot["stop"]`): a wheel's collision has a shallow round seat and, deeper, a stop that rests on the
    hub's hex end face at any turn; the stop's collision is switched off when the wheel is captured (its turn within
    the clock window of a hex orientation) and switched back on after a release only once it is clear of every hub;
  * no joint limits at fastening (the quadruped's hip ab/ad range);
  * premounted slots (`premount`: a gate's fallback, the grader config's scope.premounted; they leave the total
    weight) and the sub-assembly rule (`count_subassembly`, default on: a mate on a loose carrier counts wherever the
    carrier is; off = it counts only once its carrier chain is fastened to the base).

Slot kinds (frames 4 x 4 in metres, z along the insertion):
  keyed  the part's frame (`frame_in_part`) against the carrier's (`frame_in_carrier`): the axes within the tilt, the
         axis lines within the lateral window, the part's frame home along the axis, the turn about the axis within
         the clock window of a key orientation (k x pitch_deg). `any_key` slots (the wheels on the hubs' hexes: 60 deg)
         count in every orientation; the others only in `design_k` (the LiDAR's 90 deg pitch: its connector to the
         rear; turned by 90 / 180 / 270 deg it is fastened LOCKED_OFF and counts 0); pitch 360 = one orientation only.
         The key's height (`kit["carve"]["keys"][slot]["len_mm"]`) sets the capture depth: captured once at least
         `key_frac` of it is in (from home = (1 - key_frac) x height, unless `capture.depth_mm` says otherwise),
         released once its tip is `key_release_clear_m` clear of the carrier's face.
  seat   the part's seat frame (`frame_in_part`) within the window of its design pose on the carrier
         (`T_part_in_carrier`); `insertion_carrier` (carrier frame) splits the offset into along / across.

States: WAITING (a required slot not achieved) -> FREE -> ENGAGED (a candidate inside its slot's capture window: a soft
weld then holds it on the feature and pulls it home at <= pull_speed and pull_rate: the screws being driven, not a
jump; a seat's weld starts once the part has settled against its carrier, from a pose not past home) -> HOLDING (the lock predicate holds and the unit rests relative to the carrier, `hold_ticks` ticks) -> LOCKED
(the weld turned rigid) or LOCKED_OFF (keyed off the design orientation: welded, counts 0). VOID: fastened but its unit
or carrier scrapped, or never fastened and its carrier scrapped (no capture any more). PREMOUNTED: fastened by the cell
at the start, not scored. A captured or fastened unit leaves every other slot's candidate list; a captured unit leaving
`release_factor` x its window is released (it is a candidate again).

Kit fields read here: kit["slots"][*] = {id, kind, carrier, requires, weight, tol_pos_mm, tol_ang_deg, axial_tol_mm,
frame_in_carrier (keyed), pitch_deg, any_key, design_k, insertion_carrier (seat), capture{lateral_mm, tilt_deg,
clock_deg, key_frac, depth_mm, past_mm}, stop{geom, seat_depth_mm}, designed_unit, candidates[{unit, part,
T_part_in_carrier, frame_in_part, stop_geoms}]}; kit["carve"]["keys"][slot]["len_mm"]; kit["loose"];
kit["bodies"][body]["unit"] (the unit a body belongs to; the base's bodies say "base"). A stop's geom names may carry
"{unit}" (one stop per wheel: "stop_{unit}"); the weld of (slot, unit) is the equality `weld_name(slot, unit)`.
The cell calls `tick(sim_time)` after every control tick (after mj_forward) and `scrap(unit, sim_time, reason)` from its
watch; it may set `collision_bits` (geom id -> (contype, conaffinity), its contact clusters' table) so that a restored
stop gets the cluster bits of the moment.
"""

import math

import numpy as np

STATES = ("WAITING", "FREE", "ENGAGED", "HOLDING", "LOCKED", "LOCKED_OFF", "VOID", "PREMOUNTED")
SCRAPPED = "<scrapped>"          # self.used[unit] of a scrapped unit: on no candidate list any more


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


class CouplingStateMachine(object):
    def __init__(self, model, data, kit, hold_ticks=50, rel_speed_max=0.005, rest_ang_speed_max=math.radians(2.0),
                 capture_lateral_m=0.002, capture_ang_rad=math.radians(6.0), capture_key_frac=0.25,
                 capture_seat_depth_m=0.008, capture_seat_lateral_m=0.003, capture_past_m=0.001, release_factor=2.0,
                 key_release_clear_m=0.002, soft_solref=(0.03, 1.0), pull_speed_m_s=0.010,
                 pull_rate_rad_s=math.radians(10.0), stop_clear_m=1e-4, count_subassembly=True, premounted=(),
                 seat_settle_speed_m_s=0.05, seat_settle_rate_rad_s=math.radians(30.0)):
        """The capture_* arguments are the defaults of a slot without its own `capture` entry (the quadruped's global
        window); this kit gives every slot its own. seat_settle_*: a captured seat's pull starts once the part moves
        slower than this against its carrier (`_pull_target`)."""
        self.seat_settle_v, self.seat_settle_w = float(seat_settle_speed_m_s), float(seat_settle_rate_rad_s)
        self.m, self.d, self.kit = model, data, kit
        self.hold_ticks, self.rel_speed_max, self.rest_ang = int(hold_ticks), float(rel_speed_max), float(rest_ang_speed_max)
        self.release_factor = float(release_factor)
        self.key_release_clear = float(key_release_clear_m)
        self.soft_solref = [float(v) for v in soft_solref]
        # the capture pulls the part home at a finite rate (the screws being driven), not in one jump: a hand that still
        # holds the part (or its pads opening) would otherwise pivot it (quadruped: a support turned 6.6 deg between the
        # pads as the soft weld yanked its bore 6.5 mm home in 0.1 s, and jammed)
        self.pull_v, self.pull_w = float(pull_speed_m_s), float(pull_rate_rad_s)
        self.stop_clear = float(stop_clear_m)
        self.count_subassembly = bool(count_subassembly)
        self.collision_bits = None
        self.pull = {}
        self.loose = list(kit.get("loose") or [])
        self.loose_set = set(self.loose)
        self.body_unit = {bn: bb.get("unit") for bn, bb in (kit.get("bodies") or {}).items()}
        keys = (kit.get("carve") or {}).get("keys") or {}
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
                x["key_frac"] = float(cap.get("key_frac", capture_key_frac))
                x["cap_clock"] = math.radians(float(cap.get("clock_deg", math.degrees(capture_ang_rad))))
                x["cap_depth"] = float(cap["depth_mm"]) * 1e-3 if "depth_mm" in cap else x["key_len"] * (1.0 - x["key_frac"])
                # a captured key is released once its tip is key_release_clear_m clear of the carrier's face
                x["rel_depth"] = max(x["key_len"] + self.key_release_clear, x["cap_depth"])
            else:
                raise ValueError("kit: slot %s has unknown kind %r" % (sid, s["kind"]))
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
                x["cands"].append({"unit": c["unit"], "bp": model.body(c["part"]).id, "Tpc": Tpc, "Fp": Fp,
                                   "a_seat": (Tpc @ Fp)[:3, 3].copy(), "eq": model.equality(weld_name(sid, c["unit"])).id,
                                   "stop": [model.geom(n).id for n in geoms]})
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
        # the wheel stops: their compiled collision bits, and every hub geom a restored stop must be clear of
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
                              "axial_err_mm": None, "key_k": None} for x in self.slots}
        self.hold = {x["id"]: 0 for x in self.slots}
        self.bound = {}                 # slot -> candidate index (captured)
        self.used = {}                  # unit -> slot that fastened or holds it, or SCRAPPED
        self.scrapped = {}              # unit -> {"t_s", "reason"}
        self.premounted = []            # slot ids fastened by the cell at the start (not scored)
        self.events = []
        self.total_w = sum(x["weight"] for x in self.slots)
        for sid in premounted:
            self.premount(sid)

    # ------------------------------------------------------------------ structure
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
        """every loose unit on the slot's carrier chain is fastened (counted, or premounted) down to the base"""
        u, seen = x["carrier_unit"], set()
        while u in self.loose_set:
            if u in seen:
                return False
            seen.add(u)
            s2 = self._fastened_slot_of(u)
            if s2 is None or not (s2 in self.premounted or self.counted(s2)):
                return False
            u = self.by_id[s2]["carrier_unit"]
        return True

    # ------------------------------------------------------------------ score
    def counted(self, sid):
        """the slot's mate counts: fastened in a counting orientation, not void, not premounted (and, with
        count_subassembly off, its carrier chain fastened to the base)"""
        st = self.rec[sid]
        if sid in self.premounted or not st["fastened"] or not st["design_ok"] or st["void"]:
            return False
        return self.count_subassembly or self._chain_fastened(self.by_id[sid])

    def score(self):
        return sum(x["weight"] for x in self.slots if self.counted(x["id"])) / self.total_w if self.total_w > 0 else 0.0

    def _refresh(self):
        for x in self.slots:
            self.rec[x["id"]]["achieved"] = self.counted(x["id"])

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
            r["design_ok"] = True
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
            r["design_ok"] = bool(x.get("any_key")) or (k - int(x.get("design_k", 0))) % nk == 0
        if r["holds"]:
            vc, vp = self._vel(x["bc"]), self._vel(c["bp"])
            r["rel_speed"] = float(np.linalg.norm(vp[3:] - vc[3:]))
            r["rel_ang_speed"] = float(np.linalg.norm(vp[:3] - vc[:3]))
        return r

    def _inside(self, x, r, release=False):
        """inside the slot's capture window; `release` = the window a captured unit has to leave to be released
        (release_factor x the window; a key: its tip key_release_clear_m clear of the carrier's face)"""
        f = self.release_factor if release else 1.0
        lo = -(x["rel_depth"] if release else x["cap_depth"])
        if not (lo <= r["axial"] <= x["cap_past"] * f and r["ang_err"] <= x["cap_tilt"] * f):
            return False
        if x["kind"] == "seat":
            return r["lateral"] <= x["cap_lat"] * f
        return r["pos_err"] <= x["cap_lat"] * f and abs(r["clock_res"]) <= x["cap_clock"] * f

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
                # insertion: a weld engaged on a battery still falling onto its bay (0.3-0.4 m/s, 600-800 deg/s at the
                # capture tick) and the landing's contacts ask nearly hard, conflicting accelerations of the same body
                # (6.5-9.6 kN for one tick in the prototype cell, 17-32 N without the capture)
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
        # frame (the quadruped's check: zero residual at the target, none from a turn about it; in this cell a wheel
        # and a battery were pulled home through it as well)
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

    # ------------------------------------------------------------------ the wheel stop
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
        """a released wheel's stop gets its collision back only once it is clear of every hub (switched on while it
        overlaps a hex, it would throw the wheel off)"""
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
        """take a loose unit out of the cell's coupling for good (off the bench, or faster than the part-speed rule):
        release its capture first, then take it off every candidate list; void the mates it or a unit it carries make.
        A fastened unit stays welded where it is (a deck falling with its compute unit takes it along). Returns False
        if it was scrapped before."""
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
        """fasten `unit` (default: the slot's designed unit) on its slot at its design pose, rigidly, before the
        episode (a gate's fallback); the slot leaves the total weight. `place` moves a free unit there first."""
        import mujoco
        x = self.by_id[sid]
        if unit is None:
            unit = x.get("designed_unit") or x["cands"][0]["unit"]
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
        self.rec[sid].update({"fastened": True, "design_ok": True, "unit": unit, "t_s": float(sim_time), "pos_err_mm": 0.0,
                              "ang_err_deg": 0.0, "axial_err_mm": 0.0})
        self.used[unit] = sid
        self.premounted.append(sid)
        self.state[sid] = "PREMOUNTED"
        self.total_w = sum(y["weight"] for y in self.slots if y["id"] not in self.premounted)
        self.events.append((sim_time, "PREMOUNTED %s (%s)" % (sid, unit)))
        self._refresh()

    # ------------------------------------------------------------------ per control tick
    def tick(self, sim_time):
        if self.stop_pending:
            self._restore_stops(sim_time)
        self._dist = self._pair_dist()
        try:
            self._tick(sim_time)
        finally:
            self._dist = None
        self._refresh()

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
            if not all(q in self.premounted or self.counted(q) for q in x["requires"]):
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
                        continue
                    rr = self.evaluate(x, c)
                    if self._inside(x, rr) and (best is None or rr["pos_err"] < best[1]["pos_err"]):
                        best = (i, rr)
                if best is None:
                    self.state[sid] = "FREE"
                    self.hold[sid] = 0
                    continue
                ci, r = best
                c = x["cands"][ci]
                self.bound[sid] = ci
                self.used[c["unit"]] = sid
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
                   "design_ok": bool(r["design_ok"])})
        if x["kind"] == "keyed":
            st["key_k"] = int(r["k"])
        self._weld(c, self._target(x, c, r), soft=False)
        self.bound.pop(sid, None)
        self.used[c["unit"]] = sid
        self.state[sid] = "LOCKED" if st["design_ok"] else "LOCKED_OFF"
        self.events.append((sim_time, ("LOCKED %s (%s)" % (sid, c["unit"])) if st["design_ok"] else
                            "LOCKED_OFF %s (%s, key orientation %d, design %d)" % (sid, c["unit"], r["k"], x.get("design_k", 0))))

    # ------------------------------------------------------------------ reports
    def record(self):
        """one entry per scored slot (premounted slots are left out: they are not in the total weight); `achieved`
        = the mate counts now"""
        self._refresh()
        out = []
        for x in self.slots:
            if x["id"] in self.premounted:
                continue
            r = dict(self.rec[x["id"]])
            r["state"] = self.state[x["id"]]
            out.append(r)
        return out

    def metrics(self):
        rec = [self.rec[x["id"]] for x in self.slots if x["id"] not in self.premounted]
        return {"locked": sum(1 for r in rec if r["fastened"] and r["design_ok"] and not r["void"]),
                "locked_off": sum(1 for r in rec if r["fastened"] and r["design_ok"] is False),
                "void_unit_scrapped": sum(1 for r in rec if (r["void"] or "").startswith("unit ")),
                "void_carrier_scrapped": sum(1 for r in rec if (r["void"] or "").startswith("carrier ")),
                "scrapped_units": len(self.scrapped), "premounted": len(self.premounted),
                "weight_total": float(self.total_w), "weight_counted": float(sum(x["weight"] for x in self.slots if self.counted(x["id"])))}

    def capture_table(self):
        """the slots' windows in mm / deg (for the cell's documents): capture, release, lock, rest, pull"""
        out = []
        for x in self.slots:
            f = self.release_factor
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
            out.append(e)
        return out
