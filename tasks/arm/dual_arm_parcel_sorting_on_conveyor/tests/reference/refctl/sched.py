"""Shift scheduler of the reference: one job at a time; each job is decided
step by step from the truth.

Label directions are the world directions of the label normal (``geom``):
up, down, +y (downstream face), -y (upstream face), +x (away from the
robot), -x (towards the robot). The workhorse is the end-over-end roll
("roll U", ``endover``): a quarter turn about the belt-transverse axis, top
towards upstream,

    down -> +y -> up -> -y -> down.

Methods per label (at the job's start, from the truth):
  up           nothing (the arms stay parked; the parcel passes untouched)
  +y           one roll U
  down         the continued half turn: roll U, and without letting go a
               second arc about the new upstream bottom edge (``endover``
               3b), so the parcel lands on its former top
  -y           ``options["minus_y"]``: "roll" (one roll U: the label goes
               down) or "pass"
  +x / -x      the side-face module (``options["side_module"]``), else the
               parcel passes untouched

Streams: one job at a time. A job starts on the most downstream parcel that
needs work once every parcel downstream of it has cleared the work zone; a
job is admitted only if the parcel behind it reaches the work zone after the
job's expected end (else the parcel is let through untouched), and a running
job is ended early (clean give-up: release, park) if the parcel behind comes
closer to the work zone than ``abort_gap_m``.
"""

from __future__ import annotations

import time

import numpy as np

from .controller import RefController
from .endover import EndOver, JobAbort, RollParams
from .geom import aligned_dims, label_dir
from .paths import PlanFail

ROLL_CYCLE = ("down", "+y", "up", "-y")


def rolls_needed(lab: str) -> int | None:
    if lab not in ROLL_CYCLE:
        return None
    return (ROLL_CYCLE.index("up") - ROLL_CYCLE.index(lab)) % 4


DEFAULTS = dict(
    max_rolls=4,              # roll attempts per job
    brake_intermediate=1,     # brake past the balance also on rolls that are not the last one
    half_turn=True,           # label down: the continued half turn
    minus_y="pass",           # label -y: "pass" | "roll" (one roll U: the label then down) | "rolld" (reverse roll, experimental)
    early_finish=False,       # evaluation: end the shift once every parcel is judged
    side_module=None,         # callable(ctl, slot) -> dict for labels on +-x (None: the package's side-face module, see side)
    side="module",            # labels on +-x: "module" (the side-face module: sideturn.side_job) | "pass" (untouched)
    side_params=None,         # dict of sideturn.SideParams overrides for the side-face module
    mailers="pass",           # soft mailers: outside the station's duty: they pass untouched
    deadline_abort=True,      # end a job early (clean give-up) when the next parcel that needs work would
                              # otherwise be missed: its job must start before its own deadline
    t_exit_s=3.0,             # ... time the exit (back-off + park) needs
    side_lead={"+x": (-0.26, 1.7)},   # side jobs that start upstream: front y <= a - b x v at the job's start
    catch_line=0.15,          # downstream face at the catch (m)
    zone_clear_y=0.45,        # a parcel downstream clears the work zone (the left arm's sweep) when its upstream face is past this
    zone_up_margin=0.12,      # the right palm works up to this far upstream of the job's upstream face
    abort_gap_m=0.04,         # end a job early if the parcel behind is closer than this to the zone
    t_job={"+y": 9.0, "down": 11.0, "-y": 9.0, "+x": 12.0, "-x": 12.0},   # expected job time (s), catch to clear
    down_squeeze="flat",      # label down: "flat" (squeeze flip for parcels lower than flat_h, else the half turn) | "always" | "never"
    flat_h=0.09,
    half_min_h=0.07,          # label down without a squeeze plan: the half turn only from this height
    held_retries=1,           # retries of a roll that did not commit (then a clean give-up)
    max_aspect=2.6,           # roll / half turn only if (length along the belt) / height <= this
    min_roll_h={"+y": 0.09, "down": 0.07},   # ... and the height >= this (lower parcels jam the pusher's upper finger)
    max_aspect_tall=3.0,      # parcels with H >= 0.10: roll / half turn up to this L / H
    soft_squeeze_only=True,   # soft mailers: the squeeze flip where it plans, else they pass
    squeeze_ab_by_kind={"soft": (16.0, 8.0), "rigid": (28.0, 12.0)},   # squeeze per palm a + b m (N)
    sq_soft_by_label={"+y": 110.0, "down": 90.0},   # squeeze governor limit (N) of the first arc per label
    log=False,
)


def _arrival_label(parcel):
    """World label direction of a parcel at its arrival (label face turned
    by the arrival yaw)."""
    import math
    n = ((0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))[int(parcel.label_face)]
    c, s_ = math.cos(parcel.yaw), math.sin(parcel.yaw)
    return label_dir(np.array([c * n[0] - s_ * n[1], s_ * n[0] + c * n[1], n[2]]))[0]


def default_side_module(ctl, slot):
    """The side-face job of this package (sideturn.side_job) with the
    options' SideParams overrides (options["side_params"])."""
    from .sideturn import SideParams, side_job
    sp = SideParams()
    for k, v in (ctl.o.get("side_params") or {}).items():
        setattr(sp, k, v)
    return side_job(ctl.rc, ctl.eo, slot, sp)


def _small(r, depth=0):
    """A JSON-friendly digest of a method's record (scalars, short lists,
    nested dicts two levels deep; long traces left out)."""
    if not isinstance(r, dict):
        return r
    out = {}
    for k, v in r.items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, dict) and depth < 2:
            out[k] = _small(v, depth + 1)
        elif isinstance(v, (list, tuple)) and len(v) <= 8 and all(
                isinstance(x, (bool, int, float, str)) or x is None for x in v):
            out[k] = list(v)
    return out


class ShiftController:
    def __init__(self, world, server=None, options=None):
        self.w = world
        self.server = server
        self.o = dict(DEFAULTS)
        self.o.update(options or {})
        self.rc = RefController(world)
        # the cell's start posture is the park posture
        self.rc.q_ready = np.asarray(world.home, float).copy()
        prm = RollParams()
        for k, v in (self.o.get("roll") or {}).items():
            setattr(prm, k, type(getattr(prm, k))(v))
        self.prm = prm
        self.eo = EndOver(self.rc, prm, log=(print if self.o.get("log") else None))
        self.eo.abort_check = None
        if self.o.get("side", "module") == "pass":
            self.o["side_module"] = None
        elif self.o.get("side_module") is None:
            self.o["side_module"] = default_side_module
        self.jobs = []
        self.n_errors = 0
        self.n_deadline_aborts = 0
        self.status = {}          # uid -> pass_up | pass_side | job | let_through | missed
        self.t_wall0 = time.time()
        self.n_aborts = 0

    # ------------------------------------------------------------------
    def log(self, *a):
        if self.o.get("log"):
            print(f"[{self.w.time:7.2f}]", *a)

    def label_of(self, slot):
        ps = self.w.truth.parcel_state(slot)
        return label_dir(ps.label_normal)

    def label_now(self, slot):
        if not self.w.slots[slot].active:
            return ("gone", 180.0)
        return self.label_of(slot)

    @staticmethod
    def extent_y(ps):
        d = aligned_dims(ps.size, ps.quat)
        return float(ps.pos[1] - d[1] / 2), float(ps.pos[1] + d[1] / 2)

    def _all_judged(self):
        w = self.w
        if w._next_arrival < len(w.shift.arrivals):
            return False
        recs = self.server.record.parcels if self.server is not None else []
        return all(r["scan"] is not None or r["fall_t"] is not None for r in recs)

    # ------------------------------------------------------------------
    def run(self):
        w = self.w
        while not w.done:
            if self.o.get("early_finish") and self.server is not None and self._all_judged():
                w.end("finished")
                break
            slot = self.pick()
            if slot is None:
                self.eo.hold(0.1)
                continue
            self.do_job(slot)

    def behind_of(self, y_back):
        """Downstream face y of the nearest parcel upstream of y_back: an
        active parcel on the belt, else the next due arrival (it lands at
        entry_y at its arrival time and rides at belt speed)."""
        w = self.w
        v = max(self.rc.belt_speed(), 1e-3)
        best = None
        for ps in w.truth.parcel_states():
            yb, yf = self.extent_y(ps)
            if yf < y_back - 1e-3 and (best is None or yf > best):
                best = yf
        if best is not None:
            return best
        arr = w.shift.arrivals
        k = w._next_arrival
        if k < len(arr):
            a = arr[k]
            return w.station.entry_y + 0.5 * a.parcel.diag - v * max(0.0, a.t - w.time)
        return None

    def pick(self):
        w = self.w
        v = max(self.rc.belt_speed(), 1e-3)
        cands = sorted(w.truth.parcel_states(), key=lambda ps: -ps.pos[1])
        for ps in cands:
            uid = ps.uid
            if uid in self.status:
                continue
            lab, _ = label_dir(ps.label_normal)
            yb, yf = self.extent_y(ps)
            if ps.kind == "soft" and self.o.get("mailers", "pass") == "pass":
                # a soft mailer is outside the station's duty: it passes untouched
                self.status[uid] = "pass_mailer"
                continue
            if lab == "up":
                self.status[uid] = "pass_up"
                continue
            if lab in ("+x", "-x") and self.o.get("side_module") is None:
                self.status[uid] = "pass_side"
                continue
            if lab == "-y" and (self.o.get("minus_y") == "pass" or
                                (self.o.get("minus_y") == "side" and self.o.get("side_module") is None)):
                self.status[uid] = "pass_minus_y"
                continue
            # can it still be caught? (the catch needs about 1.2 s of lead)
            if yf > self.o["catch_line"] - v * 1.2 - 0.02:
                self.status[uid] = "missed"
                self.jobs.append(dict(uid=uid, arrival=w.slots[ps.slot].arrival, label0=lab,
                                      result="missed (too far downstream at the start)",
                                      t0=round(w.time, 3)))
                continue
            # every parcel downstream must have cleared the work zone
            for o in cands:
                if o.uid == uid:
                    continue
                ob, of = self.extent_y(o)
                if of > yf and ob < self.o["zone_clear_y"]:
                    return None
            # admission: the parcel behind must reach the zone after the job's end
            t_to_catch = max(0.0, (self.o["catch_line"] - yf) / v)
            t_exp = float(self.o["t_job"].get(lab, 10.0))
            y_zone = self.o["catch_line"] - (yf - yb) - self.o["zone_up_margin"]
            yf2 = self.behind_of(yb)
            if yf2 is not None:
                t_behind = (y_zone - yf2) / v
                if t_behind < t_to_catch + t_exp:
                    self.status[uid] = "let_through"
                    self.jobs.append(dict(uid=uid, arrival=w.slots[ps.slot].arrival, label0=lab,
                                          result="let through (the parcel behind is too close)",
                                          t0=round(w.time, 3), t_behind=round(t_behind, 2),
                                          t_need=round(t_to_catch + t_exp, 2)))
                    continue
            return ps.slot
        return None

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _works(self, kind, lab):
        """Would the scheduler start a job on a parcel of this kind and label?"""
        if kind == "soft" and self.o.get("mailers", "pass") == "pass":
            return False
        if lab == "up":
            return False
        if lab in ("+x", "-x"):
            return self.o.get("side_module") is not None
        if lab == "-y":
            return self.o.get("minus_y") not in ("pass", None) and not (
                self.o.get("minus_y") == "side" and self.o.get("side_module") is None)
        return True

    def _start_line(self, lab, v):
        """Front y beyond which a job on this label can no longer start."""
        a_b = (self.o.get("side_lead") or {}).get(lab)
        if a_b is not None:
            return float(a_b[0]) - float(a_b[1]) * v
        return self.o["catch_line"] - v * 1.2 - 0.02

    def next_work(self, y_back):
        """(front y, label) of the nearest parcel upstream of y_back that
        needs a job: on the belt, else the next due arrivals (front at the
        entry at its arrival time, riding at belt speed); None if none."""
        w = self.w
        v = max(self.rc.belt_speed(), 1e-3)
        best = None
        for ps in w.truth.parcel_states():
            if ps.uid in self.status:
                continue
            yb, yf = self.extent_y(ps)
            if yf < y_back - 1e-3:
                lab, _ = label_dir(ps.label_normal)
                if self._works(ps.kind, lab) and (best is None or yf > best[0]):
                    best = (yf, lab)
        if best is not None:
            return best
        arr = w.shift.arrivals
        for k in range(w._next_arrival, len(arr)):
            a = arr[k]
            lab = _arrival_label(a.parcel)
            if self._works(a.parcel.kind, lab):
                return (w.station.entry_y + 0.5 * a.parcel.diag - v * max(0.0, a.t - w.time), lab)
        return None

    def _arm_abort(self, slot, y_zone):
        """Abort check during a job: the parcel behind closer than abort_gap_m
        to the work zone's upstream end; or (deadline_abort) the next parcel
        that needs work would miss the start of its own job unless this one
        ends now (not while a roll's arc is running)."""
        w = self.w
        margin = self.o["zone_up_margin"]
        gap = self.o["abort_gap_m"]
        t_exit = float(self.o.get("t_exit_s", 3.0))
        dl = bool(self.o.get("deadline_abort", True))

        def check():
            if not w.slots[slot].active:
                return False
            ps = w.truth.parcel_state(slot)
            yb, _ = self.extent_y(ps)
            yf2 = self.behind_of(yb)
            if yf2 is not None and yf2 > min(y_zone, yb - margin) - gap:
                return True
            if dl and not getattr(self.eo, "_rolling", False):
                nw = self.next_work(yb)
                if nw is not None:
                    v = max(self.rc.belt_speed(), 1e-3)
                    t_left = (self._start_line(nw[1], v) - nw[0]) / v
                    if t_left < t_exit:
                        self.n_deadline_aborts += 1
                        return True
            return False
        self.eo.abort_check = check

    def do_job(self, slot):
        w = self.w
        s = w.slots[slot]
        uid = s.uid
        self.status[uid] = "job"
        lab0, err0 = self.label_of(slot)
        rec = dict(uid=uid, arrival=s.arrival, kind=s.parcel.kind, size=list(s.parcel.size),
                   mass=s.parcel.mass, label0=lab0, t0=round(w.time, 3), steps=[], result="",
                   _stops0=int(getattr(self.eo, "n_stops_seen", 0)))
        self.jobs.append(rec)
        ps = w.truth.parcel_state(slot)
        yb, yf = self.extent_y(ps)
        y_zone = self.o["catch_line"] - (yf - yb) - self.o["zone_up_margin"]
        self._arm_abort(slot, y_zone)
        try:
            if lab0 == "down" and self._use_squeeze(slot):
                if self.squeeze_job(slot, rec):
                    return
                # no squeeze plan (nothing moved): the continued half turn,
                # unless the parcel is too flat for it or soft (then it passes)
                if self._height(slot) < float(self.o.get("half_min_h", 0.0)) or \
                        (s.parcel.kind == "soft" and self.o.get("soft_squeeze_only", True)):
                    rec["result"] = "pass (no squeeze plan, too flat for the half turn)"
                    rec["t1"] = round(w.time, 3)
                    rec["label_end"] = self.label_now(slot)[0]
                    return
            if lab0 == "-y" and self.o.get("minus_y") == "rolld":
                self.rolld_job(slot, rec)
                return
            if lab0 == "-y" and self.o.get("minus_y") == "roll_squeeze":
                self.ym_job(slot, rec)
                return
            if lab0 == "-y" and self.o.get("minus_y") == "half_then_roll":
                self.ym_half_job(slot, rec)
                return
            if lab0 in ("+y", "down") and not self._box_like(slot, lab0):
                # too flat to roll (the parcel would jam between the pad and
                # the belt): it passes untouched
                rec["result"] = "pass (too flat to roll: L/H %.2f, H %.3f)" % (self._aspect(slot),
                                                                              self._height(slot))
                rec["t1"] = round(w.time, 3)
                rec["label_end"] = self.label_now(slot)[0]
                return
            if lab0 in ("+x", "-x") or (lab0 == "-y" and self.o.get("minus_y") == "side"):
                r = self.o["side_module"](self, slot)
                rec["side"] = _small(r)
                lab1, _ = self.label_now(slot)
                if lab1 not in ROLL_CYCLE or lab1 == "up" or (r or {}).get("state") == "gone":
                    rec["result"] = "side:" + str(lab1)
                    self.finish_job(slot, rec, released=bool((r or {}).get("released", True)))
                    return
                self.roll_chain(slot, rec, caught=(r or {}).get("state") == "caught")
            else:
                self.roll_chain(slot, rec)
        except JobAbort as e:
            self.n_aborts += 1
            rec["result"] = f"aborted ({e})"
            rec["aborted"] = True
            self.eo.abort_check = None
            self.finish_job(slot, rec)
        except Exception as e:  # noqa: BLE001  (a fault of a method: end the job cleanly, go on)
            if w.done or type(e).__name__ == "ShiftOver":
                raise
            self.n_errors += 1
            import traceback
            rec["result"] = f"error ({type(e).__name__}: {e})"
            rec["error"] = traceback.format_exc()[-1500:]
            self.eo.abort_check = None
            try:
                self.eo.resync()
            except Exception:  # noqa: BLE001
                pass
            try:
                self.finish_job(slot, rec)
            except Exception as e2:  # noqa: BLE001
                if w.done or type(e2).__name__ == "ShiftOver":
                    raise
                rec["exit_note"] = f"exit after error: {type(e2).__name__}: {e2}"
        finally:
            self.eo.abort_check = None

    # ------------------------------------------------------------------
    def _height(self, slot):
        ps = self.w.truth.parcel_state(slot)
        return float(aligned_dims(ps.size, ps.quat)[2])

    def _aspect(self, slot):
        """Length along the belt over height: the roll's balance tilt is
        atan(L / H); a flat parcel (large L / H) cannot be lifted to it."""
        ps = self.w.truth.parcel_state(slot)
        d = aligned_dims(ps.size, ps.quat)
        return float(d[1] / max(d[2], 1e-3))

    def _use_squeeze(self, slot):
        mode = self.o.get("down_squeeze", "flat")
        if mode == "never":
            return False
        if mode == "always":
            return True
        if self.w.slots[slot].parcel.kind == "soft" and self.o.get("soft_squeeze_only", True):
            return True
        return self._height(slot) < float(self.o.get("flat_h", 0.09)) or \
            self._aspect(slot) > float(self.o.get("max_aspect", 99.0))

    def _box_like(self, slot, lab=None):
        h_min = self.o.get("min_roll_h", 0.0)
        if isinstance(h_min, dict):
            h_min = h_min.get(lab, 0.0)
        h = self._height(slot)
        a_max = float(self.o.get("max_aspect", 99.0))
        if h >= 0.10:
            a_max = max(a_max, float(self.o.get("max_aspect_tall", a_max)))
        return h >= float(h_min) and self._aspect(slot) <= a_max

    def _sync_exec(self):
        """Hand the robot between the two executors (roll and squeeze):
        the planned state is taken from the measured posture and the last
        applied command."""
        js = self.w.joint_state()
        ex = self.eo.ex
        ex.q = js.q.copy()
        ex.qd = np.zeros(14)
        ex._u_prev = js.q_target.copy()
        self.eo._qd_f = np.zeros(14)
        if getattr(self, "core", None) is not None:
            c = self.core.ex
            c.q = js.q.copy()
            c.qd = np.zeros(14)
            c.u_prev = js.q_target.copy()

    def squeeze_job(self, slot, rec):
        """Label down on a flat parcel: the planned two-palm squeeze flip
        where it plans; else the parcel passes."""
        if getattr(self, "core", None) is None:
            from . import squeeze as _sq
            from .squeeze import Core
            if "sq_max_drop" in self.o:
                _sq.MAX_DROP = float(self.o["sq_max_drop"])
            if "sq_soft_drop" in self.o:
                _sq.SOFT_DROP = float(self.o["sq_soft_drop"])
            if "squeeze_ab" in self.o:
                _sq.SQUEEZE_A, _sq.SQUEEZE_B = (float(v) for v in self.o["squeeze_ab"])
            self.core = Core(self.w)
            self.core.ex.hooks.append(lambda ex: self._squeeze_tick())
        # squeeze per palm: soft mailers 16 + 8 m N (a lighter squeeze
        # lifted them better and damaged fewer), cartons 28 + 12 m N
        from . import squeeze as _sq
        ab = (self.o.get("squeeze_ab_by_kind") or {}).get(self.w.slots[slot].parcel.kind)
        if ab is not None:
            _sq.SQUEEZE_A, _sq.SQUEEZE_B = (float(v) for v in ab)
        self._sync_exec()
        try:
            res = self.core.run_parcel(slot)
        finally:
            self._sync_exec()
            self.eo._sq_back = {"Left": 0.0, "Right": 0.0}
        rec["squeeze"] = {k: res.get(k) for k in ("result", "plan_why", "prm", "plan_s", "cls")
                          if k in res}
        if res.get("result") in ("no_plan", "pass", "pass_y", "skip") and \
                self.o.get("down_squeeze_fallback", True):
            return False
        rec["t1"] = round(self.w.time, 3)
        rec["label_end"] = self.label_now(slot)[0]
        rec["result"] = "up" if rec["label_end"] == "up" else "squeeze:" + str(res.get("result"))
        return True

    def _squeeze_tick(self):
        chk = self.eo.abort_check
        if chk is not None and chk():
            self.eo.abort_check = None
            raise JobAbort("the parcel behind is too close")

    def rolld_job(self, slot, rec):
        """Label on the upstream face: the reverse roll (rolld), with bounded
        retries from the held state."""
        from .rolld import roll_down
        w, eo = self.w, self.eo
        try:
            eo.phase = "catch"
            eo.catch(slot, low_stop=True)
        except PlanFail as e:
            rec["result"] = f"catch: {e}"
            self.finish_job(slot, rec, released=True)
            return
        rec["t_catch"] = round(w.time, 3)
        p = self.prm
        base = dict(rd_push_fn=p.rd_push_fn, rd_lead_deg=p.rd_lead_deg)
        reason = ""
        try:
            for k in range(int(self.o.get("rd_tries", 3))):
                if not w.slots[slot].active or w.done:
                    break
                lab, _ = self.label_of(slot)
                if lab != "-y":
                    break
                try:
                    r = roll_down(eo, slot)
                except PlanFail as e:
                    reason = f"roll D plan: {e}"
                    break
                lab2 = self.label_now(slot)
                rec["steps"].append(dict(lab=lab, after=lab2[0], err=round(lab2[1], 1), rolld=True,
                                         committed=r.committed, held=r.held,
                                         tilt=round(r.tilt_max_deg, 1), t=round(w.time, 2)))
                if not r.held:
                    break
                p.rd_push_fn = base["rd_push_fn"] * (1.5 + 0.5 * k)
                p.rd_lead_deg = base["rd_lead_deg"] + 4.0 * (k + 1)
        finally:
            p.rd_push_fn, p.rd_lead_deg = base["rd_push_fn"], base["rd_lead_deg"]
        rec["t_rolls"] = round(w.time, 3)
        self.finish_job(slot, rec)
        rec["result"] = "up" if rec.get("label_end") == "up" else (reason or f"label {rec.get('label_end')}")

    def ym_job(self, slot, rec):
        """Label on the upstream face: one roll U as far upstream as the stop
        reaches (the label goes underneath, the parcel stands on its former
        upstream face), a short lift of the palms, then at once the squeeze
        flip of the standing parcel where it plans (else it passes)."""
        w, eo, p = self.w, self.eo, self.prm
        s = w.slots[slot]
        if not self._box_like(slot, "+y"):
            rec["result"] = "pass (too flat to roll)"
            rec["t1"] = round(w.time, 3)
            rec["label_end"] = self.label_now(slot)[0]
            return
        try:
            eo.phase = "catch"
            _, _, (W_, L_, H_), _, _ = eo.dims(slot)
            y_face = max(float(self.o.get("ym_stop_y", -0.15)) + L_ + 0.005, -0.10)
            eo.catch(slot, y_face=y_face)
        except PlanFail as e:
            rec["result"] = f"catch: {e}"
            self.finish_job(slot, rec, released=True)
            return
        rec["t_catch"] = round(w.time, 3)
        sq0 = p.sq_soft
        p.sq_soft = float((self.o.get("sq_soft_by_label") or {}).get("+y", sq0))
        try:
            r = eo.roll_up(slot, follow=True, final=False)
        except PlanFail as e:
            rec["result"] = f"roll plan: {e}"
            p.sq_soft = sq0
            self.finish_job(slot, rec)
            return
        p.sq_soft = sq0
        lab2 = self.label_now(slot)
        rec["steps"].append(dict(lab="-y", after=lab2[0], err=round(lab2[1], 1),
                                 committed=r.committed, held=r.held, tilt=round(r.tilt_max_deg, 1),
                                 t=round(w.time, 2)))
        if lab2[0] != "down" or not w.slots[slot].active:
            self.finish_job(slot, rec)
            rec["result"] = "up" if rec.get("label_end") == "up" else f"label {rec.get('label_end')}"
            return
        # lift both palms clear of the parcel (no full park: the flip must
        # catch the parcel before it rides out of its window)
        eo.phase = "lift"
        for sd in ("Left", "Right"):
            p_, R_ = w.tcp_pose(sd)
            eo.safe_lift(sd, p_ + np.array([0.0, 0.0, 0.06]), R_, T=0.25)
        sub = dict(uid=s.uid, arrival=s.arrival, kind=s.parcel.kind, label0="down",
                   t0=round(w.time, 3), steps=[], result="")
        planned = self.squeeze_job(slot, sub)
        rec["second"] = sub if planned else dict(sub, result="no squeeze plan")
        if not planned:
            self.finish_job(slot, rec)
        rec["label_end"] = self.label_now(slot)[0]
        rec["t1"] = round(w.time, 3)
        rec["result"] = "up" if rec["label_end"] == "up" else "roll+squeeze: " + str(
            (rec["second"].get("squeeze") or {}).get("result"))

    def ym_half_job(self, slot, rec):
        """Label on the upstream face: the continued half turn caught as far
        upstream as the stop reaches (-y -> down -> +y: the parcel lies flat
        again, label on its downstream face, upstream of where a roll leaves
        it), a fresh catch, then one roll U (+y -> up)."""
        w, eo, p = self.w, self.eo, self.prm
        if not self._box_like(slot, "+y"):
            rec["result"] = "pass (too flat to roll)"
            rec["t1"] = round(w.time, 3)
            rec["label_end"] = self.label_now(slot)[0]
            return
        sq0 = p.sq_soft
        try:
            eo.phase = "catch"
            _, _, (W_, L_, H_), _, _ = eo.dims(slot)
            y_face = max(float(self.o.get("ym_stop_y", -0.15)) + L_ + 0.005, -0.10)
            eo.catch(slot, y_face=y_face)
            rec["t_catch"] = round(w.time, 3)
            p.sq_soft = float((self.o.get("sq_soft_by_label") or {}).get("down", sq0))
            eo._half_turn = True
            try:
                r = eo.roll_up(slot, follow=True, final=False)
            finally:
                eo._half_turn = False
            lab2 = self.label_now(slot)
            rec["steps"].append(dict(lab="-y", after=lab2[0], err=round(lab2[1], 1), half=True,
                                     committed=r.committed, held=r.held,
                                     tilt=round(r.tilt_max_deg, 1), t=round(w.time, 2)))
            if lab2[0] == "+y" and w.slots[slot].active:
                eo.recatch(slot)
                p.sq_soft = float((self.o.get("sq_soft_by_label") or {}).get("+y", sq0))
                r2 = eo.roll_up(slot, follow=True, final=True)
                lab3 = self.label_now(slot)
                rec["steps"].append(dict(lab="+y", after=lab3[0], err=round(lab3[1], 1),
                                         committed=r2.committed, held=r2.held,
                                         tilt=round(r2.tilt_max_deg, 1), t=round(w.time, 2)))
            reason = ""
        except PlanFail as e:
            reason = str(e)
        finally:
            p.sq_soft = sq0
        self.finish_job(slot, rec)
        rec["result"] = "up" if rec.get("label_end") == "up" else (reason or f"label {rec.get('label_end')}")

    def finish_job(self, slot, rec, released=False):
        """Clean end of a job: palms off the parcel, arms parked."""
        w, eo = self.w, self.eo
        eo.abort_check = None
        eo.phase = "exit"
        eo._in_exit = True
        try:
            if getattr(eo, "n_stops_seen", 0) > rec.get("_stops0", 0):
                # after a protective stop: first separate the palms (each one
                # backs off along its own tool axis and up), then the usual exit
                try:
                    eo.separate()
                except (PlanFail, JobAbort) as e:
                    rec["exit_note"] = f"separate: {e}"
            try:
                if not released:
                    eo.back_off(slot=slot if w.slots[slot].active else None)
            except (PlanFail, JobAbort) as e:
                rec["exit_note"] = f"back_off: {e}"
            try:
                eo.park(slot if w.slots[slot].active else None)
            except (PlanFail, JobAbort) as e:
                rec["exit_note"] = rec.get("exit_note", "") + f" park: {e}"
        finally:
            eo._in_exit = False
        rec["t1"] = round(w.time, 3)
        rec["label_end"] = self.label_now(slot)[0]

    def roll_chain(self, slot, rec, caught=False):
        w, eo, p = self.w, self.eo, self.prm
        base = dict(push_fn=p.push_fn, lead_deg=p.lead_deg, brake=p.brake, sq_soft=p.sq_soft)
        if not caught:
            try:
                eo.phase = "catch"
                y_face = None
                lab_c, _ = self.label_of(slot)
                if lab_c == "-y" and self.o.get("minus_y") == "roll_squeeze":
                    # the first roll as far upstream as the stop reaches: the
                    # parcel slides downstream during the roll and must land
                    # where the squeeze flip can still catch it
                    _, _, (W_, L_, H_), _, _ = eo.dims(slot)
                    y_face = max(float(self.o.get("ym_stop_y", -0.15)) + L_ + 0.005, -0.10)
                eo.catch(slot, y_face=y_face)
            except PlanFail as e:
                rec["result"] = f"catch: {e}"
                self.finish_job(slot, rec, released=True)
                return
        rec["t_catch"] = round(w.time, 3)
        n = 0
        retries = 0
        reason = ""
        try:
            while n < self.o["max_rolls"] and w.slots[slot].active and not w.done:
                lab, err = self.label_of(slot)
                need = rolls_needed(lab)
                if need is None or need == 0:
                    break
                if lab == "-y" and n >= 1:
                    reason = "label -y again"
                    break
                half = bool(need == 2 and self.o.get("half_turn"))
                last = need == 1 or half
                p.brake = base["brake"] if (last or self.o["brake_intermediate"]) else 0
                eo._half_turn = half
                p.sq_soft = float((self.o.get("sq_soft_by_label") or {}).get(lab, base["sq_soft"]))
                try:
                    eo._rolling = True
                    r = eo.roll_up(slot, follow=True, final=last)
                except PlanFail as e:
                    reason = f"roll plan: {e}"
                    break
                finally:
                    eo._half_turn = False
                    eo._rolling = False
                n += 1
                lab2 = self.label_now(slot)
                face = None
                if w.slots[slot].active:
                    _, _, (W_, L_, H_), _, _ = eo.dims(slot)
                    face = round(float(w.truth.parcel_state(slot).pos[1] + L_ / 2), 3)
                rec["steps"].append(dict(lab=lab, after=lab2[0], err=round(lab2[1], 1),
                                         committed=r.committed, held=r.held, half=half,
                                         tilt=round(r.tilt_max_deg, 1), t=round(w.time, 2), face=face))
                if lab2[0] in ("up", "+x", "-x", "gone"):
                    break
                if lab == "-y" and self.o.get("minus_y") in ("roll", "roll_squeeze"):
                    break
                if r.held:
                    # a roll that did not commit (the parcel lowered back and
                    # held): a bounded retry with more push, or a clean give-up
                    retries += 1
                    n_ret = int(self.o.get("held_retries", 0))
                    if self._aspect(slot) > float(self.o.get("max_aspect", 99.0)):
                        n_ret = 0     # long boxes: no retry (its arc ran the pusher's fingers onto the rail)
                    if retries > n_ret:
                        reason = "no commit"
                        break
                    p.push_fn = base["push_fn"] * (1.5 + 0.5 * min(retries, 3))
                    p.lead_deg = base["lead_deg"] + 4.0 * min(retries, 3)
                    continue
                p.push_fn, p.lead_deg = base["push_fn"], base["lead_deg"]
                try:
                    eo.recatch(slot)
                except PlanFail as e:
                    reason = f"recatch: {e}"
                    break
        finally:
            p.push_fn, p.lead_deg, p.brake = base["push_fn"], base["lead_deg"], base["brake"]
            p.sq_soft = base["sq_soft"]
        rec["t_rolls"] = round(w.time, 3)
        rec["n_rolls"] = n
        self.finish_job(slot, rec)
        rec["result"] = "up" if rec.get("label_end") == "up" else (reason or f"label {rec.get('label_end')}")

    # ------------------------------------------------------------------
    def summary(self):
        w = self.w
        return dict(jobs=self.jobs, status={str(k): v for k, v in self.status.items()},
                    aborts=self.n_aborts, errors=self.n_errors, deadline_aborts=self.n_deadline_aborts,
                    wall_s=round(time.time() - self.t_wall0, 2),
                    peak_cmd_speed=round(float(self.eo.peak_cmd_speed), 3),
                    roll_peaks={k: (round(v, 2) if isinstance(v, float) else v)
                                for k, v in self.eo.peak.items()},
                    slew_clip_ticks=int(w.n_slew_clip_ticks))
