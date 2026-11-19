"""The heat cell: a venue day of a closed practice and two heats of four cars.

    venue = HeatVenue(world, config, providers=prov, priv_token_sha256=sha)   # config: config.load_config()
    venue.handle(method, params) ...                                          # the wire's calls, as for the shared cell
    venue.result()                                                            # racecar-venue-result-h2h/1

With any configuration that has no heats the class is the shared cell (same replies, same result record).

A heat is a session whose model holds the field: the venue and the car plus the heat's three field cars of the
world file (`truth.head_to_head.heats[h].field`, slot order, model prefixes slow0/ to slow2/), driven by the shared
driver of other cars as it is. `open_session` puts the four cars at rest on their grid slots (`truth.head_to_head.grid`:
the field cars on their own lines, the car in the last slot); the field drives from the first physics step. The heat
ends at the first of: a car completes N laps (`venue_sheet.heat_laps`; never when the sheet has none), the session's
length, the car's third warning, the void rule, `close_session`.

After every physics step of a heat the referee takes, in this order: the staff's timeline of a stopped car (set down
at 5 s, released at 20 s); the field cars (reference points, laps, progress, their own staff's holds and put-backs,
their contacts with walls and with each other); the contacts of the car (track objects, tape lines, field cars); its
laps and progress; the overtake ledger; the contact bouts with the field cars (the samples of the push, see below);
then, unless the heat ends at this step or the car is stopped, the standing-still pick-up and the four crash kinds of
the track. The count of the standing-still rule is not kept while the car is stopped, and it starts again from zero at
every physics step at which a field car standing still has its reference point within `heat.standing_car_ahead_m`
ahead of the car's along the track. A crash of the track and a car crash at the car's fault each stop the car with
one warning; whatever completes while the car is stopped with a warning adds none. A pick-up takes no warning away:
while the car is stopped for a stop without a warning (a pick-up on request or for standing still) and the staff have
not set it down yet, the four crash kinds of the track are judged at every physics step as when the car drives, and a
contact bout whose first contact is earlier than the set-down and that is judged a car crash at the car's fault while
the car is stopped for that pick-up counts as well; either turns that stop into the stop of this crash or car crash,
with its warning given at the judgement (the stop keeps its start, so the car is set down and released as before).
The car is set down clear of the band of every field line, and after a car crash at its fault at least
`heat.place_behind_m` behind that field car's reference point along the track, never ahead of its last clean position.

The push of a contact bout is measured against a copy of the heat without the car (the replay module): at the end of
the referee's work on the step at which a new bout starts, the simulation is copied as it stands, the car is taken out
of the copy, and the copy runs for `heat.car_crash_window_s`; bouts that start at the same step share one copy. The
heat's own samples of the field cars are taken every `heat.car_crash_sample_s` from that step on; the bout is a car
crash at the first sample at which its push reaches `heat.car_crash_push_m`, or when that field car's staff reset is
triggered within the bout and `heat.car_crash_window_s` after it. Nothing of a copy returns into the heat.
The fault of a bout is judged at its first contact by `rules.fault_of`.

What the referee keeps is never on the wire: the public replies carry only the car's own `lap`, `stopped`,
`released`, `warning` (with its count) and `session_end` (with its reason).
"""

import copy
import math

import mujoco
import numpy as np

from sketch.examples import check
from sketch.examples import constants as K
from sketch.others.path import Path
from sketch.referee import config as shared_config
from sketch.referee import venue as shared_venue
from sketch.referee.rules import LapClock, Touches

from h2h.shared.session_table import ConfiguredVenue

from . import config as HC
from . import replay as RP
from . import rules as R
from .staff import BandRelocator

EPS = shared_venue.EPS
REF_X = K.REF_POINT_C[0]
TRACK_CRASHES = ("displacement", "open_wall", "rollover", "out_of_bounds")
PICKUPS = ("pickup", "standing")
CODE_NONE, CODE_CAR, CODE_OBJECT = 0, 1, 2          # geom owners in a heat model; field car k is 3 + k


def _no_boxes(*args, **kwargs):
    return None


class _SimInputs(object):
    """What the shared simulation object reads of a venue, with the heat's view of the world."""

    def __init__(self, venue, world, prov):
        self.world = world
        self.prov = prov
        self.cfg = venue.cfg
        self.floor = venue.floor
        self.rng = venue.rng


class _SessionInputs(object):
    """What the shared session object reads of a venue, with a configuration of its own."""

    def __init__(self, venue, cfg):
        self.cfg = cfg
        self.params = venue.params
        self.lines = venue.lines


def heat_view(world, h):
    """The world as the shared builder, driver and duct model of a heat see it: the avoidance set replaced by heat
    h's field (no boxes). Never written to a file."""
    view = dict(world)
    view["truth"] = dict(world["truth"])
    view["truth"]["avoidance"] = {"boxes": [], "slow_cars": list(world["truth"]["head_to_head"]["heats"][h]["field"])}
    return view


def check_heat_world(world, n_heats):
    """Errors of the parts of a world that a heat reads."""
    err = []
    t = world.get("truth", {})
    hh = t.get("head_to_head")
    if not isinstance(hh, dict):
        return ["truth.head_to_head is missing"]
    laps = world.get("venue_sheet", {}).get("heat_laps", "missing")
    if laps == "missing" or not (laps is None or (isinstance(laps, int) and not isinstance(laps, bool) and laps >= 1)):
        err.append("venue_sheet.heat_laps must be a positive integer or null")
    heats = hh.get("heats")
    slots = (hh.get("grid") or {}).get("slots")
    if not isinstance(heats, list) or len(heats) < n_heats or not isinstance(slots, list) or len(slots) < n_heats:
        return err + ["truth.head_to_head needs %d heats and %d sets of grid slots" % (n_heats, n_heats)]
    for h in range(n_heats):
        field = heats[h].get("field")
        if not isinstance(field, list) or len(field) != 3:
            err.append("heat %d: the field needs 3 cars" % (h + 1))
            continue
        if [int(f.get("slot", 0)) for f in field] != [1, 2, 3]:
            err.append("heat %d: the field is not in slot order 1, 2, 3" % (h + 1))
        if len(slots[h]) != 4 or [int(s.get("slot", 0)) for s in slots[h]] != [1, 2, 3, 4]:
            err.append("heat %d: the grid needs slots 1 to 4 in order" % (h + 1))
        ids = [f.get("id") for f in field]
        if len(set(ids)) != 3 or HC.CAR_ID in ids:
            err.append("heat %d: field ids must be distinct and not %r" % (h + 1, HC.CAR_ID))
    return err


class _FieldCar(object):
    """The referee's state of one field car in a heat."""

    def __init__(self, k, rec, line, n_lines, length, s_line, s0, merge_s, D0):
        self.k = k
        self.id = rec["id"]
        self.slot = int(rec["slot"])
        self.line = line
        self.lap = LapClock(n_lines - 1)
        self.standing = R.Standing(length, s_line, s0, D0=D0)
        self.ref = None
        self.ref_prev = None
        self.speed = 0.0
        self.resets_seen = 0
        self.placed = False
        self.trigger = False
        self.done = False
        self.walls = Touches(merge_s)
        self.others = Touches(merge_s)


class HeatState(object):
    """The referee's state of one open heat (built when the heat opens)."""

    def __init__(self, venue, ses, h):
        cfg = venue.cfg
        hc = cfg["heat"]
        self.h = h
        self.hc = hc
        self.N = venue.world["venue_sheet"].get("heat_laps")
        self.L = venue.track.length
        self.dt = venue.dt
        self.s_line = venue.lines[0].s_m
        hh = venue.world["truth"]["head_to_head"]
        recs = hh["heats"][h]["field"]
        self.slots = hh["grid"]["slots"][h]
        merge = float(venue.params["touch_merge_s"])
        sim = venue.sim
        ref = sim.ref_xy()
        self.grid_ref = ref
        s_car = venue.track.project1(*ref)[0]
        self.car = R.Standing(self.L, self.s_line, s_car)
        self.field = []
        for k, rec in enumerate(recs):
            pose = sim.slow.states()[k]["pose"]
            rx, ry = pose[0] + REF_X * math.cos(pose[2]), pose[1] + REF_X * math.sin(pose[2])
            s0 = venue.track.project1(rx, ry)[0]
            # counted from the timing line through the car: on a grid within half a lap behind the line this is the
            # field car's own distance from the line
            D0 = self.car.D + R.signed_gap(s0, s_car, self.L)
            fc = _FieldCar(k, rec, Path(rec["path"]["samples"]), len(venue.lines), self.L, self.s_line, s0, merge, D0)
            fc.ref = fc.ref_prev = (rx, ry)
            fc.resets_seen = int(sim.slow.cars[k].resets)
            self.field.append(fc)
        self.ids = [fc.id for fc in self.field]
        self.by_id = {fc.id: fc for fc in self.field}
        self.ledger = R.Ledger(self.ids, self.L, hc["half_car_length_m"], hc["clean_before_s"], hc["clean_after_s"])
        self.contacts = R.Contacts(self.ids, hc["contact_merge_s"], hc["car_crash_push_m"], hc["car_crash_window_s"])
        self.handling = R.Handling(self.ids)
        self.still = R.StandingStill(self.dt, hc["standing_s"], hc["standing_speed_mps"])
        self.left_grid = False
        self.stops = []
        self.t_last = 0.0
        self.ended_by = None
        self.warn_end = int(cfg["staff"]["warnings_end_session"])
        # the copies without the car: one sample every `every` physics steps, `n` samples after the first one
        self.sample_s, self.window_s = float(hc["car_crash_sample_s"]), float(hc["car_crash_window_s"])
        self.every = int(round(self.sample_s / self.dt))
        self.copy_steps = int(round(self.window_s / self.dt))
        self.n = self.copy_steps // self.every
        self.i = 0                      # physics steps of the heat so far
        self.probes = []                # copies whose window is still being sampled in the heat
        self.pending = []               # new bouts of this step that wait for their copy

    # ------------------------------------------------------------------ the field after a physics step
    def observe_field(self, t, venue, sim):
        """Positions, staff state, laps and progress of the field cars; returns their N-th laps at this step."""
        ends = []
        dt = self.dt
        for fc, c, st in zip(self.field, sim.slow.cars, sim.slow.states()):
            x, y, yaw = st["pose"]
            rx, ry = x + REF_X * math.cos(yaw), y + REF_X * math.sin(yaw)
            resets = int(c.resets)
            fc.placed = resets != fc.resets_seen
            fc.resets_seen = resets
            fc.trigger = self.handling.step(t, fc.id, c.held_until is not None)
            inside, s, _ = venue.track.inside1(rx, ry)
            fc.standing.move(s, inside, placed=fc.placed)
            p0 = fc.ref_prev
            hits = []
            for ln in venue.lines:
                cr = ln.crossing(p0, (rx, ry))
                if cr is not None:
                    hits.append((cr[1], ln.index, ln, cr[0]))
            for frac, _, ln, direction in sorted(hits, key=lambda q: (q[0], q[1])):
                if ln.index < 0:
                    fc.lap.timing(t - dt + frac * dt, direction)
                    if direction > 0:
                        fc.standing.crossed_forward(frac, fc.lap.count)
                        if self.N is not None and fc.lap.count == self.N and not fc.done:
                            fc.done = True
                            ends.append((t - dt + frac * dt, fc.slot, fc.id))
                else:
                    fc.lap.checkpoint(ln.index, direction)
            fc.speed = math.hypot(rx - p0[0], ry - p0[1]) / dt
            fc.ref_prev = fc.ref = (rx, ry)
        return ends

    def field_touches(self, t, walls, pairs):
        for fc in self.field:
            fc.walls.step(t, fc.k in walls)
            fc.others.step(t, any(fc.k in p for p in pairs))

    # ------------------------------------------------------------------ the rules
    def ledger_step(self, t, on_track, stopped, placed_car):
        lam = {fc.id: self.car.D - fc.standing.D for fc in self.field}
        return self.ledger.step(t, lam, on_track, stopped, placed_car, {fc.id: fc.placed for fc in self.field})

    def fault(self, j, t0):
        """The fault of a bout that starts at t0 (`rules.fault_of`): the opponent's when the car leads j on the track
        by more than half a car length, completed no overtake of j within `heat.fault_clear_ahead_s` before t0, and
        either had overtaken j before or is being lapped by it; else the car's."""
        return R.fault_of(self.ledger, j, t0, self.hc["fault_clear_ahead_s"])

    def bouts_contact(self, t, touched, replays=True):
        opened = self.pending.append if replays else None
        for k in sorted(touched):
            fc = self.field[k]
            self.contacts.contact(fc.id, t, lambda j=fc.id: self.fault(j, t), opened)

    def open_probe(self, venue, bouts=()):
        """Copy the heat as it stands now (after the referee's work on the last physics step), take the car out, run
        the copy for the window, and register the heat's own sampling of the same window; `bouts` are measured
        against it. Returns the probe."""
        lines = [fc.line for fc in self.field]
        parts = RP.cell_parts(venue.sim, bool(venue.params.get("refresh_kinematics", True)))
        alone, cut, wall = RP.run_copy(parts, lines, self.copy_steps, self.every, keep=venue.replay_keep())
        probe = RP.Probe(self.i, lines, [fc.ref for fc in self.field], [fc.resets_seen for fc in self.field], alone, cut,
                         self.every, self.n, self.sample_s, self.window_s, ids=self.ids)
        for b in bouts:
            b.probe, b.k = probe, self.ids.index(b.car)
        self.probes.append(probe)
        venue.replay_walls.append(wall)
        venue.timing["replays"] = venue.timing.get("replays", 0) + 1
        venue.timing["replay_s"] = venue.timing.get("replay_s", 0.0) + wall
        return probe

    def bouts_measure(self, t):
        judged = []
        i = self.i
        if self.probes:
            refs, resets = [fc.ref for fc in self.field], [fc.resets_seen for fc in self.field]
            for pr in self.probes:
                if pr.due(i):
                    pr.observe(i, refs, resets)
            self.probes = [pr for pr in self.probes if not pr.over(i)]

        def sample(b):
            return None if b.probe is None else b.probe.latest_push(b.k, i)
        for fc in self.field:
            if not self.contacts.watching(fc.id, t):
                continue
            judged += self.contacts.measure(fc.id, t, fc.trigger, sample)
        self.contacts.settle(t)
        self.ledger.settle(t, self.contacts, self.handling)
        return judged

    def standing_car_ahead(self, s_car):
        """True when a field car standing still has its reference point within the distance of the rule ahead."""
        reach = self.hc["standing_car_ahead_m"]
        for fc in self.field:
            if fc.speed < self.hc["standing_speed_mps"]:
                g = R.signed_gap(fc.standing.s, s_car, self.L)
                if -EPS <= g <= reach + EPS:
                    return True
        return False

    # ------------------------------------------------------------------ the record
    def block(self, ses):
        """The heat's record block, the windows and bouts settled at the last step the referee took."""
        t_end = self.t_last
        ledger = copy.deepcopy(self.ledger)
        contacts = self.contacts                  # read only: an open bout is recorded as a touch (settled as final)
        ledger.settle(t_end, contacts, self.handling, t_end=t_end)
        K_ = int(ses.warnings)
        car_p = self.car.progress
        field, C, beaten = [], 0, 0
        for fc in self.field:
            clean, contact, losses = ledger.balance(fc.id)
            bal = clean - losses
            prog = fc.standing.progress
            bt = bal >= 1 and prog < car_p
            C += bal
            beaten += int(bt)
            field.append({"id": fc.id, "slot": fc.slot, "laps": fc.lap.count, "progress": round(prog, 6),
                          "resets": fc.resets_seen, "clean_overtakes": clean, "contact_overtakes": contact,
                          "losses": losses, "balance": bal, "beaten": bool(bt), "wall_touches": fc.walls.episodes,
                          "field_touches": fc.others.episodes})
        laps = ses.lap.count
        short = int(self.hc["classified_laps_short"])
        classified = None if self.N is None else bool(laps >= self.N - short and K_ < self.warn_end)
        kinds = [s["kind"] for s in self.stops]
        end = ses.end or "aborted"
        return {"laps_target": self.N, "end_reason": end, "end_t_s": round(t_end, 6),
                "ended_by": self.ended_by if end == "laps" else None,
                "car": {"laps": laps, "progress": round(car_p, 6), "classified": classified, "warnings": K_,
                        "crashes_track": sum(1 for k in kinds if k in TRACK_CRASHES),
                        "crashes_car": kinds.count("car_crash"), "pickups_requested": kinds.count("pickup"),
                        "pickups_standing": kinds.count("standing"),
                        "placement_fallbacks": sum(1 for s in self.stops if s["placement"] == "fallback")},
                "field": field, "overtakes": ledger.records(), "contacts": [b.record() for b in contacts.all],
                "stops": [{"t_s": round(s["t"], 6), "kind": s["kind"], "warned": s["warned"], "car": s["car"],
                           "placement": s["placement"]} for s in self.stops],
                "C": C, "beaten": beaten, "K": K_}

    def truth(self):
        out = []
        for fc in self.field:
            out.append({"id": fc.id, "laps": fc.lap.count, "progress": fc.standing.progress})
        return out


class HeatSession(shared_venue._Session):
    """The referee's state of one open session; `heat` holds the heat's state (None in practice)."""

    def __init__(self, venue, index):
        cfg = venue.cfg
        if "avoid" in cfg.get("scoring", {}):
            super().__init__(venue, index)
        else:
            h = cfg["heat"]
            sc = dict(cfg["scoring"])
            sc["avoid"] = {"standstill_s": h["standing_s"], "standstill_speed_mps": h["standing_speed_mps"]}
            c2 = dict(cfg)
            c2["scoring"] = sc
            super().__init__(_SessionInputs(venue, c2), index)
        self.heat = None

    def record(self):
        r = super().record()
        if self.heat is not None:
            r["heat"] = self.heat.block(self)
        return r


class HeatVenue(ConfiguredVenue):
    """The cell of one venue of the head-to-head day (see the module text)."""

    SESSION_CLASS = HeatSession
    VARIANTS = {"avoid": "avoid", "heat1": "heat", "heat2": "heat"}

    def __init__(self, world, config=None, replays=True, **kw):
        """`replays` False (maintainer use only) runs no copy without the car: every push is then 0 and the record
        has no replay block, so it fails the record checker."""
        cfg = config if config is not None else HC.load_config()
        super().__init__(world, cfg, **kw)
        self.heats = HC.heat_sessions(self.cfg)
        self.replays = bool(replays)
        self.replay_walls = []          # wall-clock seconds of every copy run (never in the record)
        self._keep = {}
        if self.heats:
            err = check_heat_world(self.world, len(self.heats))
            if err:
                raise ValueError("world %s: %s" % (self.world.get("world_id"), "; ".join(err)))
            hc = self.cfg["heat"]
            self.relocator = BandRelocator(self.track, self.lines, self.cfg, self.params,
                                           self.world["truth"]["seeds"]["relocation"], hc["lane_half_width_m"],
                                           hc["lane_clearance_m"])

    # ------------------------------------------------------------------ helpers
    def _heat(self):
        ses = self.ses
        return getattr(ses, "heat", None) if ses is not None else None

    def replay_keep(self):
        """Objects a copy of the open heat shares instead of copying (they are only read): the world and the heat's
        view of it with their top-level parts, the floor, the track, the parameters."""
        sim = self.sim
        key = id(sim)
        if key not in self._keep:
            objs = [self.world, self.world.get("truth"), self.floor, self.track, self.params, self.cfg, self.lines]
            view = getattr(sim.slow, "world", None)
            if isinstance(view, dict):
                objs += [view, view.get("truth")]
                objs += [v for v in (view.get("truth") or {}).values() if isinstance(v, (dict, list))]
            objs += [v for v in self.world["truth"].values() if isinstance(v, (dict, list))]
            self._keep[key] = tuple(o for o in objs if o is not None)
        return self._keep[key]

    def open_probe(self):
        """Maintainer entry (not reachable from the wire): copy the open heat now, without the car, and sample the
        heat itself over the same window. The heat feeds the probe as it goes on; `probe.over(heat.i)` tells when
        its window is complete, `probe.result()` gives per field car both lists of samples and the push."""
        H = self._heat()
        if H is None or self.ses.end_pending is not None or self.state != "open":
            raise ValueError("no heat is running")
        return H.open_probe(self)

    def _build_heat_sim(self, name, sensors=None):
        h = self.heats.index(name)
        prov = copy.copy(self.prov)
        prov.BoxTracker = _no_boxes
        sim = shared_venue._Sim(_SimInputs(self, heat_view(self.world, h), prov), "avoid", sensors=sensors)
        sim.variant = "heat"
        sim.heat_index = h
        m = sim.model
        oid = mujoco.mjtObj
        code = np.zeros(m.ngeom, dtype=np.int64)
        code[sim.ego] = CODE_CAR
        roots = [mujoco.mj_name2id(m, oid.mjOBJ_BODY, K.slow_prefix(k) + "chassis") for k in range(3)]
        for g in range(m.ngeom):                 # by name: a staged field car has its contact bits cleared until placed
            if code[g]:
                continue
            rule = K.geom_rule(mujoco.mj_id2name(m, oid.mjOBJ_GEOM, g) or "")
            if rule is None:
                continue
            root = int(m.body_rootid[int(m.geom_bodyid[g])])
            if rule[1] in ("duct", "box"):
                code[g] = CODE_OBJECT
            elif rule[1] == "car" and root in roots:
                code[g] = 3 + roots.index(root)
        sim.code = code
        return sim

    def _heat_contacts(self, sim):
        """(the car touches a track object, field cars the car touches, field cars touching a track object, pairs of
        field cars touching each other) at the last physics step; contacts inside a margin do not count."""
        d = sim.data
        n = d.ncon
        if not n:
            return False, set(), set(), set()
        g = d.contact.geom
        live = d.contact.dist <= 0.0
        if not live.any():
            return False, set(), set(), set()
        ca, cb = sim.code[g[live, 0]], sim.code[g[live, 1]]
        lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
        pairs = set(zip(lo.tolist(), hi.tolist()))
        obj, touched, walls, field = False, set(), set(), set()
        for a, b in pairs:
            if a == CODE_CAR and b == CODE_OBJECT:
                obj = True
            elif a == CODE_CAR and b >= 3:
                touched.add(b - 3)
            elif a == CODE_OBJECT and b >= 3:
                walls.add(b - 3)
            elif a >= 3 and b >= 3 and a != b:
                field.add((a - 3, b - 3))
        return obj, touched, walls, field

    # ------------------------------------------------------------------ sessions
    def _enter_variant(self, variant):
        if variant != "heat":
            return super()._enter_variant(variant)
        name = self.session_names[self.next_idx - 1]
        key = "heat:" + name
        if key in self.sims:
            self.sim = self.sims[key]
            return
        old = self.sim
        sim = self._build_heat_sim(name, sensors=old.sensors if old is not None else None)
        if old is not None:
            ref = old.ducts.reference_poses() if hasattr(old.ducts, "reference_poses") else None
            sim.ducts.set_poses(old.ducts.poses(), reference=ref)
            mujoco.mj_forward(sim.model, sim.data)
        self.sims[key] = sim
        self.sim = sim

    def open_session(self, name):
        reply = super().open_session(name)
        if "error" in reply or name not in self.heats:
            return reply
        ses, sim = self.ses, self.sim
        h = sim.heat_index
        slots = self.world["truth"]["head_to_head"]["grid"]["slots"][h]
        sim.car.place(slots[3]["pose"])
        sim.car.set_targets(0.0, 0.0)
        for k, c in enumerate(sim.slow.cars):
            x, y, yaw = (float(v) for v in slots[k]["pose"])
            s = c.path.project(x + REF_X * math.cos(yaw), y + REF_X * math.sin(yaw))[0]
            sim.slow._place_on_path(c, s)
        sim.slow.force_plan = True
        if hasattr(sim.sensors, "reset_motion"):
            sim.sensors.reset_motion()
        mujoco.mj_forward(sim.model, sim.data)
        ses.ref_prev = sim.ref_xy()
        s, d = self.track.project1(*ses.ref_prev)[:2]
        ses.clean = (s, d)
        ses.heat = HeatState(self, ses, h)
        self.relocator.set_bands([fc.line for fc in ses.heat.field])
        return reply

    # ------------------------------------------------------------------ the referee after one physics step
    def _referee_step(self, t, dmax, events):
        if self._heat() is None:
            return super()._referee_step(t, dmax, events)
        return self._heat_step(t, dmax, events)

    def _heat_step(self, t, dmax, events):
        """The referee after one physics step of a heat; the copies of the bouts that started at this step are run
        once the step's work is done (the state the heat goes on from)."""
        H = self.ses.heat
        H.i += 1
        out = self._heat_referee(t, dmax, events)
        if H.pending:
            bouts, H.pending = H.pending, []
            H.open_probe(self, bouts)
        return out

    def _heat_referee(self, t, dmax, events):
        ses, sim, H = self.ses, self.sim, self.ses.heat
        if dmax > ses.max_disp:
            ses.max_disp = dmax
        if ses.end_pending is not None:
            ses.ref_prev = sim.ref_xy()
            return
        H.t_last = t
        was_stopped = ses.stopped
        placed_car = False
        if ses.stopped:
            if not ses.relocated and t >= ses.t_stop + self.relocate_at_s - EPS:
                self._relocate(t)
                placed_car = True
            elif t >= ses.t_stop + self.hold_s - EPS:
                if not ses.relocated:
                    self._relocate(t)
                    placed_car = True
                ses.stopped = False
                events.append({"type": "released"})
                ses.log.append((t, "released", None))
        ends = H.observe_field(t, self, sim)
        obj, touched, walls, pairs = self._heat_contacts(sim)
        H.field_touches(t, walls, pairs)
        ref = sim.ref_xy()
        on_ref, s_ref, d_ref = self.track.inside1(*ref)
        H.car.move(s_ref, on_ref, placed=placed_car)
        if not H.left_grid and math.hypot(ref[0] - H.grid_ref[0], ref[1] - H.grid_ref[1]) >= H.hc["standing_left_grid_m"] - EPS:
            H.left_grid = True
        if was_stopped:
            ses.ref_prev = ref
            H.ledger_step(t, on_ref, True, placed_car)
            H.bouts_contact(t, touched, self.replays)
            if ends:
                return self._end_on_laps(ends)
            for b in H.bouts_measure(t):
                self._car_crash(t, b, events)
            if self._picking_up():
                # a pick-up takes no warning away: until the staff set the car down, its crashes are judged as it drives
                wheels = sim.data.xpos[sim.wheel_bids, :2].tolist()
                if obj:
                    ses.last_obj_t = t
                inside = [self.track.inside1(x, y)[0] for x, y in wheels]
                crash = self._track_crash(t, dmax, sim, self._beyond(ref, wheels), inside)
                if crash is not None:
                    self._pickup_becomes(t, crash, events)
            return
        d = sim.data
        wheels = d.xpos[sim.wheel_bids, :2].tolist()
        beyond = self._beyond(ref, wheels)
        touching = obj or beyond > 0 or bool(touched)
        if obj:
            ses.last_obj_t = t
        if ses.touches.step(t, touching):
            ses.n_touch += 1
            ses.lap.touch()
            ses.log.append((t, "touch", "object" if obj else ("car" if touched else "tape")))
        p0 = ses.ref_prev
        hits = []
        for ln in self.lines:
            c = ln.crossing(p0, ref)
            if c is not None:
                hits.append((c[1], ln.index, ln, c[0]))
        for frac, _, ln, direction in sorted(hits, key=lambda q: (q[0], q[1])):
            if ln.index < 0:
                ev = ses.lap.timing(t - self.dt + frac * self.dt, direction, ses.touches.active(t))
                if ev is not None:
                    events.append(ev)
                if direction > 0:
                    H.car.crossed_forward(frac, ses.lap.count)
                    if H.N is not None and ses.lap.count == H.N and ev is not None and ev["lap_ms"] is not None:
                        ends.append((t - self.dt + frac * self.dt, 4, HC.CAR_ID))
            else:
                ses.lap.checkpoint(ln.index, direction)
        v = math.hypot(ref[0] - p0[0], ref[1] - p0[1]) / self.dt
        ses.ref_prev = ref
        warn, still = ses.watch.step(v, ses.phases[ses.phase_idx]["speed_limit_mps"])
        if still:
            ses.lap.standstill()
        new = H.ledger_step(t, on_ref, False, False)
        net = sum(1 if e["kind"] == "overtake" else -1 for e in new)
        if net:
            ses.lap.overtake(net)
        H.bouts_contact(t, touched, self.replays)
        if ends:
            return self._end_on_laps(ends)
        for b in H.bouts_measure(t):
            self._car_crash(t, b, events)
        if ses.stopped or ses.end_pending is not None:
            return
        if H.standing_car_ahead(s_ref):
            H.still.reset()                 # the count starts again once no standing car is within reach ahead
        elif H.still.step(v) and H.left_grid:
            self._stop(t, "standing", events)
            return
        inside = [self.track.inside1(x, y)[0] for x, y in wheels]
        crash = self._track_crash(t, dmax, sim, beyond, inside)
        if crash is not None:
            self._stop(t, crash, events)
        elif all(inside) and not touching:
            ses.clean = (s_ref, d_ref)

    def _beyond(self, ref, wheels):
        """The most wheels of the car beyond one open wall's tape near its reference point."""
        beyond = 0
        for tape in self.tapes:
            if tape.near(ref):
                beyond = max(beyond, tape.count(wheels))
        return beyond

    def _track_crash(self, t, dmax, sim, beyond, inside):
        """The crash of the four kinds of the track that the last physics step makes (None when there is none): a
        displaced track object attributed to the car, the wheels beyond an open wall, a rollover, the whole car outside
        the track. A displaced object found while the car has touched no track object for a moment is noted as not
        the car's and not judged again."""
        ses = self.ses
        crash = None
        if dmax >= self.thr:
            newly = set(("duct",) + tuple(p) for p in sim.ducts.displaced(self.thr))
            newly -= ses.known_displaced
            if newly:
                if t - ses.last_obj_t <= float(self.params["displacement_attribution_s"]) + EPS:
                    crash = "displacement"
                else:
                    ses.known_displaced |= newly
                    ses.log.append((t, "displaced_unattributed", sorted(map(str, newly))))
        if crash is None and beyond >= self.wheels_over:
            crash = "open_wall"
        if crash is None and float(sim.data.xmat[sim.cid][8]) < self.cos_roll:
            crash = "rollover"
        if crash is None and not any(inside):
            crash = "out_of_bounds"
        return crash

    def _picking_up(self):
        """True while the car is stopped for a stop without a warning (a pick-up) and the staff have not set it down."""
        ses, H = self.ses, self.ses.heat
        return (ses.stopped and not ses.relocated and ses.end_pending is None and bool(H.stops)
                and not H.stops[-1]["warned"])

    def _pickup_becomes(self, t, kind, events, opponent=None):
        """The stop of a pick-up becomes the stop of the crash or car crash judged now: its kind, its warning (given now
        and counted like any other: the third ends the heat), the field car it names; it keeps its start, so the car
        is set down and released as before. The session's counts follow the stops of the record; a whole car outside
        the track counts for the venue's out-of-bounds rule as at any other time."""
        ses, H = self.ses, self.ses.heat
        st = H.stops[-1]
        st["kind"], st["warned"] = kind, True
        if opponent is not None:
            st["car"] = opponent
        ses.pickups -= 1                    # the session's counts follow the stops of the record
        ses.crashes += 1
        if opponent is not None:
            ses.log.append((t, "stop_is_car_crash", opponent))
        else:
            ses.log.append((t, "stop_is_crash", kind))
        if kind == "out_of_bounds":
            ses.out_of_bounds += 1
            self.out_of_bounds += 1
            if self.out_of_bounds >= int(self.cfg["staff"]["out_of_bounds_void_at"]):
                ses.end_pending = "void"
        ses.warnings += 1
        events.append({"type": "warning", "count": ses.warnings})
        ses.log.append((t, "warning", ses.warnings))
        if ses.warnings >= H.warn_end and ses.end_pending is None:
            ses.end_pending = "warnings"

    def _end_on_laps(self, ends):
        ses, H = self.ses, self.ses.heat
        t_c, _, who = min(ends)
        H.ended_by = who
        if ses.end_pending is None:
            ses.end_pending = "laps"

    def _car_crash(self, t, bout, events):
        """A bout judged a car crash. At the field car's fault: nothing for the car. At the car's fault: while the car
        drives, a stop with one warning. While the car is stopped for a stop without a warning (a pick-up), and the
        bout's first contact is earlier than the staff set the car down (before the pick-up or during it), that stop
        becomes the stop of this car crash (`_pickup_becomes`): its kind, its warning (given now), the field car it
        names; it keeps its start, so the car is set down and released as before. While the car is stopped in any other
        case nothing more is given. In every case in which the car is stopped and the staff have not set it down yet,
        it is set down behind that field car too."""
        ses, H = self.ses, self.ses.heat
        if bout.fault != "car":
            return
        if not ses.stopped:
            self._stop(t, "car_crash", events, opponent=bout.car)
            bout.warned = True
            return
        st = H.stops[-1] if H.stops else None
        if st is None:
            return
        if not st["warned"] and (st["t_down"] is None or bout.t0 < st["t_down"] - EPS):
            self._pickup_becomes(t, "car_crash", events, opponent=bout.car)
            bout.warned = True
        if not ses.relocated:
            st["behind"].add(bout.car)

    # ------------------------------------------------------------------ the staff
    def _stop(self, t, kind, events, opponent=None):
        H = self._heat()
        if H is None:
            return super()._stop(t, kind, events)
        ses = self.ses
        super()._stop(t, "pickup" if kind in PICKUPS else kind, events)
        warned = kind not in PICKUPS
        H.stops.append({"t": t, "kind": kind, "warned": warned, "car": opponent, "placement": None, "t_down": None,
                        "behind": {opponent} if opponent is not None else set()})
        H.still.reset()
        if warned:
            ses.warnings += 1
            events.append({"type": "warning", "count": ses.warnings})
            ses.log.append((t, "warning", ses.warnings))
            if ses.warnings >= H.warn_end and ses.end_pending is None:
                ses.end_pending = "warnings"

    def _relocate(self, t):
        H = self._heat()
        if H is None:
            return super()._relocate(t)
        ses, sim = self.ses, self.sim
        st = H.stops[-1] if H.stops else None
        s_target, d_side = ses.clean
        # the shared search may move the target forward by its along-track error: start that much further back
        behind = float(H.hc["place_behind_m"]) + float(self.cfg["staff"]["relocate_pos_err_m"])
        for j in sorted(st["behind"]) if st else ():
            cand = (H.by_id[j].standing.s - behind) % self.track.length
            if R.signed_gap(cand, s_target, self.track.length) < 0.0:
                s_target = cand
        pose, info = self.relocator.place_clear(s_target, d_side, self._obstacles(sim))
        if st is not None:
            st["placement"] = info["placement"]
            st["t_down"] = t
        sim.car.place(pose)
        sim.car.set_targets(0.0, 0.0)
        self._put_back(sim)
        ses.known_displaced = set()
        mujoco.mj_forward(sim.model, sim.data)
        if hasattr(sim.sensors, "reset_motion"):
            sim.sensors.reset_motion()
        ses.touches.reset_gap()
        ses.watch.reset()
        ses.overtakes.forget()
        H.still.reset()
        ses.relocated = True
        ses.log.append((t, "relocated", info))

    # ------------------------------------------------------------------ records
    def truth_state(self):
        out = super().truth_state()
        H = self._heat()
        field = []
        if H is not None:
            for st, fc, c in zip(self.sim.slow.states(), H.field, self.sim.slow.cars):
                field.append({"id": fc.id, "pose": [float(v) for v in st["pose"]], "speed_mps": float(st["speed_mps"]),
                              "on_track": bool(st["on_track"]), "held": c.held_until is not None, "laps": fc.lap.count,
                              "progress": fc.standing.progress})
        if self.heats:
            out["field"] = field
        return out

    def result(self):
        if not self.heats:
            return super().result()
        sessions = [s.record() for s in self.sessions]
        sim = self.sim
        res = {"schema": HC.RESULT_SCHEMA, "world_id": self.world["world_id"], "layout_digest": self.world["layout_digest"],
               "config_sha256": shared_config.config_sha256(self.cfg),
               "end_reason": self.end_reason or ("finish" if self.over else "eof"), "step_called": self.step_called,
               "sessions": sessions,
               "stats": {"touches": sum(s["touches"] for s in sessions), "crashes": sum(s["crashes"] for s in sessions),
                         "wall_displacement_m": round(float(sim.ducts.total_displacement()), 6) if sim else 0.0,
                         "laps_by_session": {s["name"]: len(s["laps"]) for s in sessions},
                         "program_s": round(self.program_total_s, 6), "warnings": sum(s["warnings"] for s in sessions),
                         "forbidden_calls": self.counts["forbidden_calls"], "bad_calls": self.counts["bad_calls"]}}
        return res


__all__ = ["HeatVenue", "HeatSession", "HeatState", "heat_view", "check_heat_world"]
