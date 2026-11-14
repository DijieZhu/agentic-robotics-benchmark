"""The reference program: the operating procedure (given/sop.md) carried out by the privileged skills, deciding only on
what a field operator at the site can see.

It runs inside the cell it drives (it is the oracle, not a camera program): it takes the devices' poses from the cell
instead of finding them and reads the instruments' displays without a camera. It drives the robot through the cell's
own calls, so every rate limit, the grip rule and the safety watch apply to it. Its decisions use only what an operator
can observe (`Eyes`): the instruments' displays (PI-101, PI-103, PDI-111, FI-105, LG-101, LG-201, AI-201), the lamps and
the horn, the sight glasses VS-102 / VS-112, the drums' visible levels and the valves' positions; and facts of the series
that hold on every site and are documented or measured on any practice site (the tanks' volume per percent of their
level gauges, the analyser's lag, the alarm set points, the consumer's usual draw, typical valve and dosing gains as first
guesses). It never reads the process's hidden state: not the consumer's draw, the pump's curve, the hypochlorite's
strength, the metering pump's capacity, the supply pressure, the fault flags or the event schedule. What it does not
know it estimates from the trends (the draw from LG-201 against FI-105, the dosing gain from AI-201's response) or finds
by trying (the flow-control valve's setting against FI-105). `run(server)` is called by the server's privileged
`reference_run`; it returns when the shutdown is done or the episode ends.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skidcell.episode import log_round  # noqa: E402  (the log sheet's two rounds: sop.md 2.3)
from skidcell.layout import CONC_BAND, SHIFT_END_S, STARTUP_DEADLINE_S  # noqa: E402
from skidcell.process import TANKS  # noqa: E402  (the tanks' geometry: a series fact, the same on every site)
from skills import Skills, Stop  # noqa: E402

LEVEL_TARGET = 65.0
CONC_TARGET = 2.2
FILL_LEVEL = 62.0              # % of T-201 reached 10 s before the window; from 15 s before it the transfer runs at the usual draw
PRE_DRAW_S = STARTUP_DEADLINE_S - 15.0
LOG_UNITS = {"PI-101": "bar", "PI-103": "bar", "PDI-111": "bar", "FI-105": "m3/h", "LG-201": "%", "AI-201": "mg/L"}
PDI_ACT_BAR = 0.45             # switch filters a little before the alarm when the trend is clear
T101_TARGET = 60.0             # % of T-101 held with the make-up valve once its float valve no longer refills it
DRUM_LOW = 0.04                # change drums in drum A's last 4 % (sop AOP-4: below 5 % counts)
# series facts and first guesses (an operator's datasheet values or what practice sites show)
T201_M3_PER_PCT = TANKS["T-201"]["area_m2"] * TANKS["T-201"]["height_m"] / 100.0
T101_M3_PER_PCT = TANKS["T-101"]["area_m2"] * TANKS["T-101"]["height_m"] / 100.0
AI_LAG_S = 20.0                # sop.md 2.2
USUAL_DRAW_M3H = 2.3           # the consumer's draw when the window opens, as practice sites show it (about 2.0-2.6)
DOSE_GAIN_GUESS = 9400.0       # mg/h of hypochlorite at a fully open V-301 on a typical site: the first setting only
V105_GAIN_GUESS = 5.0          # m3/h per unit opening of V-105 near its working point: the first step only
V106_GAIN_GUESS = 4.5          # m3/h per unit opening of V-106: the first step only
FLOW_TOL = 0.08                # m3/h: close enough when trying V-105 against FI-105


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


class Eyes(object):
    """What an operator can see at the site, read perfectly; the reference decides on nothing else."""

    def __init__(self, server):
        self._proc = server.proc
        self._scene = server.scene

    def value(self, tag):
        """An instrument's display in its base unit (bar, m3/h, %, mg/L)."""
        return float(self._proc.values()[tag])

    def lamp(self, name):
        """RUN-101, FAULT-101, RUN-301 or ALM (the common alarm)."""
        return bool(self._proc.lamps()[name])

    def horn(self):
        return bool(self._proc.horn)

    def sight(self, tag):
        """VS-102 / VS-112: True once liquid stands in the sight glass."""
        return bool(self._proc.sights()[tag])

    def drum(self, tag):
        """T-301 (drum A) / T-302 (drum B): the level seen through the drum's wall, as a fraction of the drum."""
        return float(self._proc.drum_levels()[tag])

    def drum_lined_up(self):
        """The drum V-303's lever points at: "A" or "B"."""
        return self._proc.selected_drum()

    def opening(self, tag):
        """A valve's position (from its handle and stem), 0 closed .. 1 open."""
        return float(self._scene.opening(tag))


class Trend(object):
    """A short history of readings, for slopes."""

    def __init__(self, keep_s=300.0):
        self.rows = []
        self.keep_s = keep_s

    def add(self, t, row):
        if self.rows and t - self.rows[-1][0] < 0.5:
            self.rows[-1] = (t, row)
            return
        self.rows.append((t, row))
        while self.rows and t - self.rows[0][0] > self.keep_s:
            self.rows.pop(0)

    def window(self, span, since=-1e9):
        if not self.rows:
            return []
        t_end = self.rows[-1][0]
        return [(t, r) for t, r in self.rows if t >= max(since, t_end - span)]

    def slope(self, key, span, since=-1e9, min_span=None):
        """Least-squares slope of `key` (per second) over the last `span` seconds; None without enough data."""
        w = self.window(span, since)
        need = 0.5 * span if min_span is None else min_span
        if len(w) < 3 or w[-1][0] - w[0][0] < need:
            return None
        ts = [t for t, _ in w]
        vs = [r[key] for _, r in w]
        tm, vm = sum(ts) / len(ts), sum(vs) / len(vs)
        den = sum((t - tm) ** 2 for t in ts)
        if den <= 0.0:
            return None
        return sum((t - tm) * (v - vm) for t, v in zip(ts, vs)) / den

    def mean(self, key, span, since=-1e9):
        w = self.window(span, since)
        return sum(r[key] for _, r in w) / len(w) if w else None


class Reference(object):
    def __init__(self, server):
        self.S = server
        self.eyes = Eyes(server)
        self.sk = Skills(server, log=server.log)
        self.log = server.log
        self.trend = Trend()
        self.notes = []
        self.logged = set()
        self.last_flow_adjust = -1e9
        self.last_dose_adjust = -1e9
        self.last_makeup_adjust = -1e9
        self.flow_changed_t = -1e9       # V-105 last moved (or the flow came back): AI-201's response restarts there
        self.dose_changed_t = -1e9       # V-301 last moved, or the dosing restarted
        self.dose_cut_t = -1e9           # V-301 last closed down by a third, or the dosing restarted: AI-201 may fall
        self.flow_gain = V105_GAIN_GUESS
        self.makeup_gain = V106_GAIN_GUESS
        self.dose_gain = DOSE_GAIN_GUESS
        self.dose_gain_measured = False
        self.draw = USUAL_DRAW_M3H
        self.filter_done = False
        self.supply_lost = False
        self.makeup_on = False
        self.drum_changed = False
        self.dosing_suspect_t = None
        self.started = False
        self.level0_101 = None
        self.ST = server.layout["site"]["tags"]

    def note(self, s):
        self.notes.append((round(self.sk.t(), 1), s))
        self.log("[ref] %7.1f %s" % (self.sk.t(), s))

    # ------------------------------------------------------------------ what the operator sees
    def observe(self):
        e = self.eyes
        t = self.sk.t()
        row = {"L201": e.value("LG-201"), "L101": e.value("LG-101"), "Q": e.value("FI-105"), "cd": e.value("AI-201"),
               "s": e.opening("V-301"), "run101": e.lamp("RUN-101"), "run301": e.lamp("RUN-301")}
        if self.level0_101 is None:
            self.level0_101 = row["L101"]          # the float valve holds T-101 here while nothing is drawn
        self.trend.add(t, row)
        return row

    def conc_now(self):
        """T-201's chlorine now: AI-201's display plus its lag times its slope (the analyser lags the tank by about 20 s)."""
        cd = self.eyes.value("AI-201")
        sl = self.trend.slope("cd", 15.0, min_span=6.0)
        return cd + AI_LAG_S * sl if sl is not None else cd

    def draw_estimate(self):
        """The consumer's draw from T-201's balance over the last 40 s: what came in (FI-105) minus what T-201 kept (LG-201)."""
        w = self.trend.window(40.0, since=STARTUP_DEADLINE_S + 2.0)
        if len(w) < 3 or w[-1][0] - w[0][0] < 15.0:
            return self.draw
        came_in = sum(0.5 * (ra["Q"] + rb["Q"]) * (tb - ta) for (ta, ra), (tb, rb) in zip(w[:-1], w[1:]))   # m3/h x s
        kept = (w[-1][1]["L201"] - w[0][1]["L201"]) * T201_M3_PER_PCT * 3600.0                                 # m3/h x s
        self.draw = _clip((came_in - kept) / (w[-1][0] - w[0][0]), 0.0, 5.0)
        return self.draw

    # ------------------------------------------------------------------ procedure steps
    def desired_flow(self):
        t = self.sk.t()
        lev = self.eyes.value("LG-201")
        if t < STARTUP_DEADLINE_S:
            if t >= PRE_DRAW_S - 0.5:
                return USUAL_DRAW_M3H                     # the draw starts at 600 s
            vol = max(0.0, (FILL_LEVEL - lev) * T201_M3_PER_PCT)              # m3 still to fill
            left = max(30.0, STARTUP_DEADLINE_S - 10.0 - t)
            return _clip(vol / left * 3600.0, 0.5, 3.4)
        return _clip(self.draw_estimate() + 0.08 * (LEVEL_TARGET - lev), 0.8, 3.5)

    def set_flow(self, force=False):
        """Set the transfer with V-105 by trying against FI-105 (a globe valve: the flow follows its opening); at most a
        few tries per call, so the rounds go on (the next call carries on)."""
        if not self.eyes.lamp("RUN-101"):
            return
        want = self.desired_flow()
        q = self.eyes.value("FI-105")
        if not force and abs(q - want) < 0.12:
            return
        u = self.eyes.opening("V-105")
        self.note("V-105 for %.2f m3/h (FI-105 %.2f, V-105 at %.3f)" % (want, q, u))
        for i in range(3 if force else 2):
            if abs(q - want) < FLOW_TOL:
                break
            if i == 0 and u > 0.03 and q > 0.2:
                u_new = _clip(u * want / q, 0.02, 1.0)          # the flow roughly in proportion to the opening
            else:
                u_new = _clip(u + (want - q) / self.flow_gain, 0.02, 1.0)
            if abs(u_new - u) < 0.004:
                break
            self.sk.set_opening("V-105", u_new)
            self.sk.step(20)
            q_new = self.eyes.value("FI-105")
            if abs(u_new - u) > 0.01 and abs(q_new - q) > 0.03:
                g = (q_new - q) / (u_new - u)
                if g > 0.3:
                    self.flow_gain = _clip(g, 0.5, 40.0)
            u, q = u_new, q_new
        self.last_flow_adjust = self.sk.t()
        self.flow_changed_t = self.sk.t()

    def estimate_dose_gain(self):
        """The dosing gain (mg/h at a fully open V-301) from AI-201's response once V-301 and the transfer are steady."""
        t = self.sk.t()
        e = self.eyes
        s = e.opening("V-301")
        if (s < 0.03 or not e.lamp("RUN-301") or self.dosing_suspect_t is not None or t - self.dose_changed_t < 60.0
                or t - self.flow_changed_t < 30.0 or e.drum(self.drum_tag()) < 0.02):
            return
        sl = self.trend.slope("cd", 15.0, since=max(self.dose_changed_t, self.flow_changed_t), min_span=9.0)
        if sl is None:
            return
        q = e.value("FI-105")
        c = e.value("AI-201") + AI_LAG_S * sl
        vol_l = max(20.0, e.value("LG-201") * T201_M3_PER_PCT * 1000.0)
        if q > 0.3 and sl < -0.3 * c * q * 1000.0 / (vol_l * 3600.0):
            return                                           # falling fast: perhaps the dosing is failing, learn nothing
        k = (vol_l * 3600.0 * sl + c * q * 1000.0) / s
        if k <= 500.0:
            return
        if not self.dose_gain_measured:
            self.dose_gain, self.dose_gain_measured = k, True
        elif 0.6 * self.dose_gain <= k <= 1.6 * self.dose_gain:
            self.dose_gain = 0.5 * (self.dose_gain + k)

    def drum_tag(self):
        return "T-301" if self.eyes.drum_lined_up() == "A" else "T-302"

    def set_dose(self, force=False):
        if not self.eyes.lamp("RUN-301"):
            return
        q = max(self.eyes.value("FI-105"), 0.5)
        c = self.conc_now()
        t = self.sk.t()
        if t < PRE_DRAW_S - 60.0:
            # start-up: the hypochlorite that brings T-201 to the target by the window, plus what the transfer washes out
            vol_l = max(20.0, self.eyes.value("LG-201") * T201_M3_PER_PCT * 1000.0)
            left = max(60.0, PRE_DRAW_S - t)
            mass = max(0.0, CONC_TARGET - c) * vol_l * 3600.0 / left + CONC_TARGET * q * 1000.0        # mg/h
            s = _clip(mass / self.dose_gain, 0.03, 1.0)
        else:
            target = _clip(CONC_TARGET + 0.6 * (CONC_TARGET - c), 1.6, 3.0)
            s = _clip(target * q * 1000.0 / self.dose_gain, 0.03, 1.0)
        if not force and abs(self.eyes.opening("V-301") - s) < 0.03:
            return
        self.note("V-301 -> %.3f (AI-201 %.2f, FI-105 %.2f, gain %.0f)" % (s, self.eyes.value("AI-201"), q, self.dose_gain))
        cut = s < 0.7 * self.eyes.opening("V-301")          # a cut this deep makes AI-201 fall for a while
        self.sk.set_opening("V-301", s)
        self.last_dose_adjust = self.sk.t()
        self.dose_changed_t = self.sk.t()
        if cut:
            self.dose_cut_t = self.sk.t()

    def startup(self):
        sk, e = self.sk, self.eyes
        self.observe()
        for tag in ("V-102", "V-112", "V-302", "V-103", "V-105", "V-106", "V-107"):
            if e.opening(tag) > 0.02:
                self.note("line-up: close %s" % tag)
                sk.set_opening(tag, 0.0)
        for tag in ("V-111", "V-303"):
            if e.opening(tag) > 0.05:
                self.note("line-up: %s to A" % tag)
                sk.set_device(tag, 0.0)
        for tag in ("V-101", "V-104"):
            if e.opening(tag) < 0.98:
                self.note("line-up: open %s" % tag)
                sk.set_opening(tag, 1.0)
        # an operator checks the line-up before the start (sop.md 1.1): a valve a stroke did not bring home is set again
        for _ in range(2):
            off = [t for t in ("V-102", "V-112", "V-302", "V-103", "V-105", "V-106", "V-107") if e.opening(t) > 0.03]
            off += [t for t in ("V-111", "V-303") if e.opening(t) > 0.08]
            off += [t for t in ("V-101", "V-104") if e.opening(t) < 0.95]
            if not off:
                break
            for tag in off:
                self.note("line-up check: %s at %.3f, set again" % (tag, e.opening(tag)))
                if tag in ("V-111", "V-303"):
                    sk.set_device(tag, 0.0)
                else:
                    sk.set_opening(tag, 1.0 if tag in ("V-101", "V-104") else 0.0)
        self.note("prime: vent the casing through V-102 until VS-102 shows liquid")
        sk.set_opening("V-102", 1.0)
        while not e.sight("VS-102"):
            sk.step(20)
        sk.set_opening("V-102", 0.0)
        self.note("start P-101")
        sk.set_selector("HAND")
        sk.press("START-101")
        if not e.lamp("RUN-101"):
            sk.press("START-101")
        self.started = True
        self.note("open V-103")
        sk.set_opening("V-103", 1.0)
        self.set_flow(force=True)
        self.note("start P-301")
        sk.press("START-301")
        self.set_dose(force=True)
        while sk.t() < PRE_DRAW_S:
            self.supervise(startup=True)
            sk.step(100)
        self.set_flow(force=True)

    def log_all(self):
        """The log sheet's round now (sop.md 2.3: each instrument once in each of the two rounds, P-101 running)."""
        t = self.sk.t()
        rnd = log_round(t)
        if rnd is None or not self.eyes.lamp("RUN-101"):
            return
        for tag, unit in LOG_UNITS.items():
            if (rnd, tag) in self.logged:
                continue
            self.S.log_reading(self.ST[tag], round(self.eyes.value(tag), 3), unit)
            self.logged.add((rnd, tag))

    def aop_filter(self):
        sk, e = self.sk, self.eyes
        on_b = e.opening("V-111") > 0.5 * math.pi          # the lever points at the chamber in service
        self.note("AOP-1: PDI-111 %.2f bar: %s, reset" % (e.value("PDI-111"), "change back to chamber A (full)" if on_b else
                                                           "vent chamber B, change over"))
        if not on_b and not e.sight("VS-112"):
            sk.set_opening("V-112", 1.0)
            while not e.sight("VS-112"):
                sk.step(20)
            sk.set_opening("V-112", 0.0)
        sk.set_device("V-111", 0.0 if on_b else math.pi)
        self.flow_changed_t = sk.t()             # the clean chamber gives the pump its flow back
        sk.press("RESET")
        self.filter_done = True

    def aop_airlock(self):
        sk, e = self.sk, self.eyes
        self.note("AOP-3: vent the dosing pump's head (stop, V-302 open 9 s, start)")
        sk.press("STOP-301")
        sk.set_opening("V-302", 1.0)
        sk.wait(9.0)
        sk.set_opening("V-302", 0.0)
        for _ in range(2):
            if not e.lamp("RUN-301"):
                sk.press("START-301")
        self.dose_changed_t = self.dose_cut_t = sk.t()
        self.dosing_suspect_t = None

    def aop_drum(self):
        sk, e = self.sk, self.eyes
        dry = e.drum("T-301") <= 0.002
        self.note("AOP-4: drum A at %.1f %%: dosing suction to drum B%s" % (100.0 * e.drum("T-301"), " (dry: vent too)" if dry else ""))
        sk.set_device("V-303", math.pi)
        self.drum_changed = True
        self.dose_changed_t = self.dose_cut_t = sk.t()
        if dry and e.lamp("RUN-301"):
            self.aop_airlock()
        self.dosing_suspect_t = None

    def dosing_stopped(self):
        """AI-201 falling while P-301 runs, faster than the transfer alone would explain (sop section 3)."""
        e = self.eyes
        t = self.sk.t()
        if (not e.lamp("RUN-301") or e.opening("V-301") < 0.03 or t - self.dose_cut_t < 60.0
                or t < STARTUP_DEADLINE_S - 45.0):
            self.dosing_suspect_t = None
            return False
        sl = self.trend.slope("cd", 15.0, min_span=6.0)
        if sl is None:
            return False
        q = e.value("FI-105")
        c = e.value("AI-201") + AI_LAG_S * sl
        vol_l = max(20.0, e.value("LG-201") * T201_M3_PER_PCT * 1000.0)
        washout = c * q * 1000.0 / (vol_l * 3600.0)        # mg/L per s with no dosing at all
        if washout <= 1e-5 or sl > -0.6 * washout:
            self.dosing_suspect_t = None
            return False
        if self.dosing_suspect_t is None:
            self.dosing_suspect_t = t
            return False
        return t - self.dosing_suspect_t >= 9.0

    def supply_check(self):
        """T-101's float valve no longer refills it: LG-101 well below where the float holds it and still falling, or its
        low alarm."""
        e = self.eyes
        if self.supply_lost or not self.started:
            return self.supply_lost
        lev = e.value("LG-101")
        sl = self.trend.slope("L101", 20.0, min_span=9.0)
        if (lev < 30.0 and e.lamp("ALM")) or (lev < self.level0_101 - 10.0 and sl is not None and sl < -0.03):
            self.supply_lost = True
        return self.supply_lost

    def aop_supply(self, force=False):
        sk, e = self.sk, self.eyes
        lev = e.value("LG-101")
        q = e.value("FI-105")
        x_now = e.opening("V-106")
        sl = self.trend.slope("L101", 30.0, since=self.last_makeup_adjust + 3.0, min_span=12.0)
        if x_now > 0.05 and sl is not None:
            inflow = q + sl * T101_M3_PER_PCT * 3600.0           # what V-106 passes now (T-101's balance)
            if inflow > 0.1:
                self.makeup_gain = _clip(inflow / x_now, 0.5, 20.0)
        want = max(0.0, q + (T101_TARGET - lev) * T101_M3_PER_PCT * 3600.0 / 300.0)
        x = _clip(want / self.makeup_gain, 0.0, 1.0)
        if not force and abs(x_now - x) < 0.05:
            return
        self.note("AOP-5: LG-101 %.0f %% (float valve not refilling): make-up V-106 -> %.2f" % (lev, x))
        sk.set_opening("V-106", x)
        self.makeup_on = True
        self.last_makeup_adjust = sk.t()

    def aop_trip(self):
        sk, e = self.sk, self.eyes
        self.note("AOP-6: P-101 stopped (FAULT lamp %s): reset, restart against a closed discharge" % e.lamp("FAULT-101"))
        x106 = e.opening("V-106")
        if x106 > 0.05:
            # nothing leaves T-101 while P-101 stands: keep the make-up from filling it past 85 % in the next 4 minutes
            x_max = max(0.0, 85.0 - e.value("LG-101")) * T101_M3_PER_PCT * 3600.0 / (240.0 * self.makeup_gain)
            if x106 > x_max + 0.02:
                self.note("AOP-6: LG-101 %.0f %%: make-up V-106 %.2f -> %.2f while P-101 is restarted" % (e.value("LG-101"), x106, x_max))
                sk.set_opening("V-106", x_max)
                self.last_makeup_adjust = sk.t()
        if e.lamp("FAULT-101"):
            sk.press("RESET")
        sk.set_opening("V-103", 0.0)
        sk.set_selector("HAND")
        for _ in range(2):
            if not e.lamp("RUN-101"):
                sk.press("START-101")
        for _ in range(2):
            if not e.lamp("RUN-301"):
                sk.press("START-301")
        self.dose_changed_t = self.dose_cut_t = sk.t()
        sk.set_opening("V-103", 1.0)
        self.set_flow(force=True)

    def before(self, seconds):
        """True while more than `seconds` of the shift remain: a procedure that would end after the shift earns nothing
        and only holds up the shutdown."""
        return self.sk.t() < SHIFT_END_S - seconds

    def supervise(self, startup=False):
        e = self.eyes
        self.observe()
        if not startup:
            self.log_all()
        if not e.lamp("RUN-101") and self.started and STARTUP_DEADLINE_S <= self.sk.t() < SHIFT_END_S:
            self.aop_trip()
        if not self.filter_done and e.value("PDI-111") > PDI_ACT_BAR and self.before(30.0):
            self.aop_filter()
        if self.supply_check() and (
                (not self.makeup_on and self.before(5.0))
                or (self.makeup_on and self.before(60.0) and self.sk.t() - self.last_makeup_adjust > 40.0)):
            self.aop_supply(force=not self.makeup_on)
        if not self.drum_changed and e.drum_lined_up() == "A" and e.drum("T-301") < DRUM_LOW and self.before(100.0):
            self.aop_drum()
        if self.before(100.0) and self.dosing_stopped():
            if e.drum(self.drum_tag()) < 0.05 and e.drum_lined_up() == "A" and not self.drum_changed:
                self.aop_drum()
            else:
                self.aop_airlock()
        if e.horn() and self.before(5.0):
            self.sk.press("RESET")
        if not startup and not self.before(20.0):
            return                                           # the shutdown comes next
        self.observe()
        self.estimate_dose_gain()
        t = self.sk.t()
        if t - self.last_flow_adjust > 25.0:
            self.set_flow()
        if t - self.last_dose_adjust > 40.0 and (startup or self.before(20.0)):
            c = self.conc_now()
            if startup or c < CONC_BAND[0] + 0.25 or c > CONC_BAND[1] - 0.25:
                self.set_dose()

    def operate(self):
        while self.sk.t() < SHIFT_END_S:
            self.supervise()
            self.log_all()
            self.sk.step(int(min(500, max(1, round((SHIFT_END_S - self.sk.t()) / 0.01)), 300)))

    def shutdown(self):
        sk, e = self.sk, self.eyes
        self.note("shutdown")
        sk.press("STOP-301")
        sk.set_opening("V-103", 0.0)
        sk.press("STOP-101")
        if e.lamp("RUN-101"):
            sk.press("STOP-101")
        if e.opening("V-106") > 0.02:
            self.note("shutdown: close the make-up valve V-106")
            sk.set_opening("V-106", 0.0)
        sk.set_opening("V-101", 0.0)
        sk.set_selector("OFF")
        sk.home()

    def report(self):
        return {"notes": self.notes[-60:], "logged": sorted("%s@%d" % (tag, r) for r, tag in self.logged), "t_end": self.sk.t()}


def run(server):
    ref = Reference(server)
    try:
        ref.startup()
        ref.operate()
        ref.shutdown()
    except Stop:
        ref.note("episode over")
    return ref.report()
