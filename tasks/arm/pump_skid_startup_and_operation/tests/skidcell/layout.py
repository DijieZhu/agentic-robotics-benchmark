"""The skid of one episode, drawn from its seed (cell internals, run by the grader only; the hidden episodes' seeds are
pinned in tests/hidden/seeds.json).

Every skid of the series has the same process (the P&ID, `pid.png`) and is described with the series tags (V-101,
P-101, ...); each site builds and numbers its skid its own way. What a seed draws:
* the site: its area number and the site tag of every device (`site["tags"]`: series tag -> site tag), which is what
  the tag plates, the control station's labels and the equipment list show;
* the arrangement: the line runs left to right (supply tank on the left) or mirrored; the flow meter and the flow control
  valve in either order; the control station's two pump groups either way round; every device's position along the
  skid is jittered by a few centimetres; the suction and discharge pipe heights vary;
* how each valve is built: the suction and discharge isolation valves are gate valves with a handwheel or ball valves
  with a lever, the minimum-flow valve a ball valve or a small gate valve, the vents small ball valves or needle valves;
  handwheel sizes, spokes, turns, frictions, lever lengths, colours;
* the gauges' types (range and unit) at PI-101, PI-103, PDI-111 and FI-105 (textures.GAUGE_KINDS);
* the shutdown state the previous shift left: two or three valves are not where the operating procedure expects them
  (v0.6: the minimum-flow return V-104 stands open after a shutdown by the SOP, which leaves it open, and a V-104 left
  out of place is shut; before v0.6 every site started with V-104 shut and a V-104 "out of place" stood open);
* the process parameters (pump curve, tank levels, residual chlorine, the consumer's demand, filter fouling, the
  dosing pump, the drums, the strainer chambers' fouling) and the operating window's events (always a change of the
  consumer's demand, plus three faults out of: a debris slug on the strainer, a stuck float valve on the supply tank,
  an air lock in the dosing pump, the duty dosing drum running dry, a trip of the transfer pump) with their hidden
  times.

Every length is in metres; the floor is z = 0 and the skid's front plane y = 0 (cell_manual.md).
"""

import math

import numpy as np

from .textures import GAUGE_KINDS

__all__ = ["draw_layout", "X_TOTAL", "RAIL", "WINDOW", "SHIFT_END_S", "STARTUP_DEADLINE_S", "LEVEL_BAND", "CONC_BAND",
           "nominal_layout", "SERVICES", "STARTUP_POSITION", "FAULT_TYPES", "LOG_SERIES"]

X_TOTAL = 5.0                     # the mirror maps x -> X_TOTAL - x
RAIL = {"y": -0.70, "x_lo": 0.30, "x_hi": 4.70, "speed_m_s": 0.30, "lift_lo": 0.05, "lift_hi": 0.60, "lift_speed_m_s": 0.15,
        "lift_start": 0.30}
STARTUP_DEADLINE_S = 600.0        # the operating window opens here, ready or not
SHIFT_END_S = 1500.0              # the shift ends: shut the skid down
WINDOW = (STARTUP_DEADLINE_S, SHIFT_END_S)
LEVEL_BAND = (55.0, 75.0)         # % of LG-201 to hold through the window
CONC_BAND = (1.8, 2.6)            # mg/L of AI-201 to hold through the window
FAULT_TYPES = ("filter_slug", "supply_loss", "dosing_airlock", "drum_empty", "pump_trip")
EVENT_GAP_S = 120.0               # the window's events are at least this far apart (v0.5; 150 s before)
LOG_SERIES = ("PI-101", "PI-103", "PDI-111", "FI-105", "LG-201", "AI-201")

WHEEL_RGBA = {"red": (0.72, 0.09, 0.07, 1), "black": (0.12, 0.12, 0.13, 1), "blue": (0.10, 0.25, 0.62, 1)}
GRIP_RGBA = {"red": (0.85, 0.12, 0.08, 1), "yellow": (0.95, 0.78, 0.10, 1), "blue": (0.12, 0.30, 0.75, 1)}

# what each series tag is (the equipment list's "service" column) and the valves' start-up position
SERVICES = {
    "V-101": "P-101 suction isolation valve",
    "V-102": "P-101 casing vent valve",
    "V-103": "P-101 discharge isolation valve",
    "V-104": "minimum-flow return valve (to T-101)",
    "V-105": "product flow control valve",
    "V-106": "T-101 make-up valve (bypass of the float valve)",
    "V-107": "V-105 bypass valve",
    "V-111": "strainer changeover valve (chamber A / chamber B)",
    "V-112": "strainer chamber B vent valve",
    "V-301": "dosing rate needle valve",
    "V-302": "P-301 head vent valve",
    "V-303": "dosing suction selector (drum A / drum B)",
    "P-101": "transfer pump",
    "P-301": "chlorine dosing pump",
    "T-101": "supply tank",
    "T-201": "product tank",
    "T-301": "dosing drum A (duty)",
    "T-302": "dosing drum B (standby)",
    "F-101": "duplex suction strainer",
    "PI-101": "P-101 suction pressure gauge",
    "PI-103": "P-101 discharge pressure gauge",
    "PDI-111": "strainer differential pressure gauge",
    "FI-105": "product flow meter",
    "LG-101": "T-101 level gauge",
    "LG-201": "T-201 level gauge",
    "AI-201": "residual chlorine analyser (T-201)",
    "LCS": "local control station",
}
STARTUP_POSITION = {"V-101": "open", "V-102": "closed", "V-103": "closed", "V-104": "open", "V-105": "closed",
                    "V-106": "closed", "V-107": "closed", "V-111": "A", "V-112": "closed", "V-301": "set the dosing rate",
                    "V-302": "closed", "V-303": "A"}
TAG_CLASS = {"P-101": "P", "P-301": "P", "T-101": "TK", "T-201": "TK", "T-301": "TK", "T-302": "TK", "F-101": "F",
             "PI-101": "PI", "PI-103": "PI", "PDI-111": "PDI", "FI-105": "FI", "LG-101": "LI", "LG-201": "LI", "AI-201": "AI"}


def nominal_layout():
    """The arrangement before jitter and mirroring (the line runs left to right; the flow meter before the control
    valve). x positions only; heights follow the pipe heights drawn per episode."""
    return {
        "LCS": 0.42, "T-101": 0.42, "V-106": 0.88, "LG-101": 1.12, "F-101A": 1.40, "F-101B": 1.64, "V-101": 1.92,
        "PI-101": 2.09, "P-101": 2.34, "V-103": 2.62, "V-104": 2.88, "FI-105": 3.22, "V-105": 3.56, "AI-201": 3.84,
        "V-301": 3.78, "P-301": 3.92, "T-302": 4.02, "V-303": 4.14, "T-301": 4.27, "LG-201": 4.40, "T-201": 4.66,
    }


def _u(rng, a, b):
    return float(rng.uniform(a, b))


def _site_tags(rng):
    area = int(rng.integers(11, 90))
    series = ["V-101", "V-102", "V-103", "V-104", "V-105", "V-106", "V-107", "V-111", "V-112", "V-301", "V-302", "V-303",
              "P-101", "P-301", "T-101", "T-201", "T-301", "T-302", "F-101", "PI-101", "PI-103", "PDI-111", "FI-105",
              "LG-101", "LG-201", "AI-201"]
    nums = [int(n) for n in rng.permutation(np.arange(1, 90))[:len(series)]]
    tags = {}
    for s, n in zip(series, nums):
        cls = "HV" if s.startswith("V-") else TAG_CLASS[s]
        tags[s] = "%s-%d%02d" % (cls, area, n)
    tags["LCS"] = "LCS-%d" % area
    return {"area": area, "name": "SITE %d" % area, "tags": tags}


def draw_layout(seed, cfg=None):
    """The episode's skid: a plain dict (JSON-able) that physics.Scene builds and process.Process runs."""
    cfg = cfg or {}
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFFFFFFFFFF, 0x5C1D2])
    site = _site_tags(rng)
    ST = site["tags"]
    mirror = bool(rng.random() < 0.5)
    meter_first = bool(rng.random() < 0.5)          # FI-105 upstream of V-105 (else V-105 first)
    panel_p101_left = bool(rng.random() < 0.5)      # the P-101 group on the station's left half (as the operator sees it)
    nom = nominal_layout()
    if not meter_first:
        nom["FI-105"], nom["V-105"] = nom["V-105"], nom["FI-105"]
    x = {}
    for k, v in nom.items():
        x[k] = v + _u(rng, -0.03, 0.03)
    for k in ("T-101", "T-201", "T-301", "T-302"):
        x[k] = nom[k] + _u(rng, -0.01, 0.01)
    # keep the coupled items together
    x["LCS"] = x["T-101"] + _u(rng, -0.02, 0.02)
    x["V-112"] = x["F-101B"]
    x["VS-112"] = x["F-101B"] + 0.07
    x["V-111"] = 0.5 * (x["F-101A"] + x["F-101B"])
    x["PDI-111"] = x["V-111"]
    x["V-102"] = x["P-101"] - 0.075
    x["VS-102"] = x["P-101"] - 0.15
    x["PI-103"] = x["P-101"] - 0.12
    x["V-302"] = x["P-301"] + 0.02
    x["V-301"] = x["P-301"] - 0.14
    x["V-107"] = x["V-105"]
    z_suc = _u(rng, 0.78, 0.86)
    z_dis = _u(rng, 1.04, 1.12)
    z_mk = _u(rng, 1.10, 1.18)

    def mx(v):
        return X_TOTAL - v if mirror else v

    def mdir(a):
        """A direction in the face plane, mirrored left-right with the layout."""
        return (math.pi - a) % (2 * math.pi) if mirror else a

    side = -1.0 if mirror else 1.0            # +1: downstream is +x
    colours = list(WHEEL_RGBA)
    wheel_colour = colours[int(rng.integers(len(colours)))]
    grip_colour = list(GRIP_RGBA)[int(rng.integers(len(GRIP_RGBA)))]

    def wheel(kind, small=False):
        spokes = int(rng.choice([3, 4, 5]))
        if small:
            d = float(rng.choice([0.12, 0.14]))
            turns = float(rng.choice([1.5, 2.0]))
        else:
            d = float(rng.choice([0.16, 0.18, 0.20]))
            turns = float(rng.choice([2.0, 2.5, 3.0])) if kind == "gate" else float(rng.choice([3.0, 4.0]))
        return {"builder": "handwheel", "wheel_d": d, "spokes": spokes, "turns": turns, "friction": _u(rng, 0.35, 0.7),
                "damping": 0.4, "wheel_phase": _u(rng, 0.0, 2 * math.pi / spokes), "wheel_rgba": WHEEL_RGBA[wheel_colour], "kind": kind}

    def ball(length, closed_dir, rng_rad=math.pi / 2, pipe_axis="x", small=False, friction=(1.0, 1.6), tag_offset=(0.0, -0.075)):
        return {"builder": "lever", "lever_len": float(length), "closed_dir": float(closed_dir), "range_rad": float(rng_rad),
                "stem_len": 0.075 if small else 0.09, "friction": _u(rng, *friction), "grip_rgba": GRIP_RGBA[grip_colour],
                "pipe_axis": pipe_axis, "small": bool(small), "tag_offset": tuple(tag_offset)}

    def needle(turns, pipe_axis="z", friction=(0.08, 0.15), tag_offset=(0.0, -0.06), small=True):
        return {"builder": "knob", "turns": float(turns), "stem_len": 0.08 if small else 0.10, "friction": _u(rng, *friction),
                "pipe_axis": pipe_axis, "knob_r": 0.016 if small else 0.019, "tag_offset": tuple(tag_offset)}

    styles = {}
    dev = {}
    # isolation valves: gate valve with a handwheel, or ball valve with a lever (closed = lever up, across the pipe;
    # it opens counter-clockwise to lie along the pipe)
    for tag, z in (("V-101", z_suc), ("V-103", z_dis)):
        if rng.random() < 0.5:
            d = dict(wheel("gate"), stem_len=0.15, tag_side=1)
            styles[tag] = "handwheel"
        else:
            d = ball(_u(rng, 0.16, 0.20), math.pi / 2, pipe_axis="x", tag_offset=(side * 0.0, -0.085))
            styles[tag] = "lever"
        d["pos"] = [mx(x[tag]), 0.0, z]
        dev[tag] = d
    dev["V-105"] = dict(wheel("globe"), stem_len=0.15, tag_side=1, pos=[mx(x["V-105"]), 0.0, z_dis])
    styles["V-105"] = "handwheel"
    dev["V-107"] = dict(wheel("globe", small=True), stem_len=0.11, tag_side=-1, pos=[mx(x["V-107"]), 0.0, z_dis - 0.28])
    styles["V-107"] = "handwheel"
    dev["V-106"] = dict(wheel("globe", small=True), stem_len=0.12, tag_side=1, pos=[mx(x["V-106"]), 0.0, z_mk])
    styles["V-106"] = "handwheel"
    # the minimum-flow valve on its vertical drop: a ball valve (closed across, opens downwards along the pipe) or a
    # small gate valve
    if rng.random() < 0.5:
        d = ball(_u(rng, 0.12, 0.15), mdir(math.pi), pipe_axis="z", tag_offset=(side * 0.085, 0.0))
        styles["V-104"] = "lever"
    else:
        d = dict(wheel("gate", small=True), stem_len=0.12, tag_side=-1, pipe_axis="z")
        styles["V-104"] = "handwheel"
    d["pos"] = [mx(x["V-104"]), 0.0, 0.86]
    dev["V-104"] = d
    dev["V-111"] = dict(ball(0.13, mdir(math.pi), rng_rad=math.pi, pipe_axis="x", friction=(1.2, 1.8), tag_offset=(0.0, 0.19)),
                        pos=[mx(x["V-111"]), 0.0, 0.80], stem_len=0.11)
    styles["V-111"] = "lever"
    # vents: a small ball valve (closed across the vertical pipe, opens upwards) or a needle valve
    for tag, xx, zz in (("V-102", x["V-102"], 0.82), ("V-112", x["V-112"], 1.00), ("V-302", x["V-302"], 0.82)):
        if rng.random() < 0.5:
            d = ball(0.08, mdir(0.0), pipe_axis="z", small=True, friction=(0.6, 1.0), tag_offset=(side * -0.075, -0.035))
            styles[tag] = "lever"
        else:
            d = needle(float(rng.choice([1.5, 2.0])), pipe_axis="z", tag_offset=(side * -0.07, -0.03))
            styles[tag] = "knob"
        d["pos"] = [mx(xx), 0.0, zz]
        dev[tag] = d
    dev["V-301"] = dict(needle(float(rng.choice([2.0, 3.0])), pipe_axis="x", small=False), pos=[mx(x["V-301"]), 0.0, 0.88])
    styles["V-301"] = "knob"
    dev["V-303"] = dict(ball(0.08, mdir(0.0), rng_rad=math.pi, pipe_axis="x", small=True, friction=(0.6, 1.0),
                             tag_offset=(0.0, 0.07)), pos=[mx(x["V-303"]), 0.0, 0.92],
                        labels={"A": mdir(0.0), "B": mdir(math.pi)})
    styles["V-303"] = "lever"
    # gauges and instruments
    kinds = {"PI-101": str(rng.choice(["c_m1_3_bar", "c_m100_300_kpa"])),
             "PI-103": str(rng.choice(["p_0_6_bar", "p_0_10_bar", "p_0_100_psi", "p_0_600_kpa"])),
             "PDI-111": str(rng.choice(["d_0_1_bar", "d_0_100_kpa"])),
             "FI-105": str(rng.choice(["f_05_5_m3h", "f_10_80_lpm"]))}
    dev["PI-101"] = {"builder": "dial", "pos": [mx(x["PI-101"]), -0.07, 1.06], "kind": kinds["PI-101"], "dial_d": 0.10,
                     "riser_to": [mx(x["PI-101"]), 0.0, z_suc], "title": "SUCTION"}
    dev["PI-103"] = {"builder": "dial", "pos": [mx(x["PI-103"]), -0.07, z_dis + 0.21], "kind": kinds["PI-103"], "dial_d": 0.10,
                     "riser_to": [mx(x["P-101"]), 0.0, z_dis + 0.21], "title": "DISCHARGE"}
    dev["PDI-111"] = {"builder": "dial", "pos": [mx(x["PDI-111"]), -0.07, 1.22], "kind": kinds["PDI-111"], "dial_d": 0.09,
                      "riser_to": [mx(x["PDI-111"]), 0.0, 0.88], "title": "FILTER DP"}
    dev["FI-105"] = {"builder": "rotameter", "pos": [mx(x["FI-105"]), 0.0, z_dis + 0.02], "kind": kinds["FI-105"], "height": 0.36}
    dev["LG-101"] = {"builder": "level", "pos": [mx(x["LG-101"]), -0.04, 0.32], "height": 1.10}
    dev["LG-201"] = {"builder": "level", "pos": [mx(x["LG-201"]), -0.04, 0.32], "height": 1.10}
    dev["VS-102"] = {"builder": "sight", "pos": [mx(x["VS-102"]), -0.02, 0.90]}
    dev["VS-112"] = {"builder": "sight", "pos": [mx(x["VS-112"]), -0.03, 1.05]}
    dev["AI-201"] = {"builder": "display", "pos": [mx(x["AI-201"]), -0.06, z_dis + 0.30]}
    dev["LCS"] = {"builder": "panel", "centre": [mx(x["LCS"]), -0.13, 1.08], "p101_left": panel_p101_left}
    dev["T-101"] = {"builder": "tank", "centre_xy": [mx(x["T-101"]), 0.58], "radius": 0.30, "z0": 0.12, "height": 1.78}
    dev["T-201"] = {"builder": "tank", "centre_xy": [mx(x["T-201"]), 0.58], "radius": 0.28, "z0": 0.12, "height": 1.78}
    dev["T-301"] = {"builder": "drum", "centre_xy": [mx(x["T-301"]), 0.40], "radius": 0.12, "z0": 0.02, "height": 0.55}
    dev["T-302"] = {"builder": "drum", "centre_xy": [mx(x["T-302"]), 0.40], "radius": 0.12, "z0": 0.02, "height": 0.55}
    dev["P-101"] = {"builder": "pump", "casing": [mx(x["P-101"]), 0.02, 0.44], "direction": side}
    dev["P-301"] = {"builder": "dosing_pump", "pos": [mx(x["P-301"]), 0.0, 0.30], "direction": side}
    dev["F-101"] = {"builder": "filter", "a": [mx(x["F-101A"]), 0.04], "b": [mx(x["F-101B"]), 0.04], "z0": 0.30, "z1": 0.90}
    # the shutdown state the previous shift left (fractions of full travel; V-111 / V-303: the lever angle, 0 = A). The
    # SOP's shutdown (section 4) leaves the minimum-flow return V-104 open, so it stands open (v0.6)
    initial = {"V-101": 0.0, "V-102": 0.0, "V-103": 0.0, "V-104": 1.0, "V-105": 0.0, "V-106": 0.0, "V-107": 0.0, "V-111": 0.0,
               "V-112": 0.0, "V-301": _u(rng, 0.25, 0.75), "V-302": 0.0, "V-303": 0.0, "HS-101": "OFF"}
    wrong_pool = ["V-103", "V-105", "V-102", "V-112", "V-104", "V-106", "V-107", "V-303"]
    n_wrong = int(rng.choice([2, 3]))
    wrong = [str(w) for w in rng.choice(wrong_pool, size=n_wrong, replace=False)]
    for wtag in wrong:
        if wtag == "V-303":
            initial[wtag] = math.pi
        elif wtag == "V-104":
            # v0.6: out of place is SHUT (its line-up position is open); a handwheel keeps its draw, so every other
            # random draw of the site is the one before v0.6
            if styles.get(wtag) not in ("lever", "knob"):
                _u(rng, 0.25, 0.6)
            initial[wtag] = 0.0
        elif styles.get(wtag) in ("lever", "knob") and wtag in ("V-102", "V-112"):
            initial[wtag] = 1.0
        else:
            initial[wtag] = _u(rng, 0.25, 0.6)
    # process parameters
    proc = {
        "H0_m": _u(rng, 34.0, 40.0), "Qmax_m3h": 6.0,
        "h101_frac": _u(rng, 0.55, 0.80), "h201_frac": _u(rng, 0.36, 0.50), "c201_mgL": _u(rng, 1.10, 1.65),
        "demand_m3h": _u(rng, 1.9, 2.6), "foul_rate": _u(rng, 0.004, 0.010), "dose_cap_Lh": _u(rng, 5.5, 7.2),
        "stock_mgL": _u(rng, 1300.0, 1700.0), "ai_lag_s": 20.0, "supply_dp_bar": _u(rng, 1.8, 2.2),
        "drumA_frac": _u(rng, 0.45, 0.85), "drumB_frac": _u(rng, 0.75, 0.95), "stockB_factor": _u(rng, 0.92, 1.08),
        "foulA0": _u(rng, 0.10, 0.40), "foulB0": _u(rng, 0.10, 0.30),
    }
    # the operating window's events (v0.5): the demand step always, plus three of the five faults; at least
    # EVENT_GAP_S apart, and each drawn early enough that its judgement (AOP-2 360 s, AOP-3 / -4 / -6 300 s, AOP-1 240 s
    # after the alarm) falls inside the window: a schedule that does not fit is drawn again (the same rng:
    # deterministic per seed)
    windows = {"filter_slug": (630.0, 1050.0), "supply_loss": (620.0, 980.0), "drum_empty": (640.0, 1200.0),
               "pump_trip": (640.0, 1200.0), "dosing_airlock": (630.0, 1200.0), "demand_step": (640.0, 1140.0)}
    n_faults = 3
    faults = [str(f) for f in rng.choice(FAULT_TYPES, size=n_faults, replace=False)]
    # the faults whose consequences take longest to show get their times first
    order = {"supply_loss": 0, "filter_slug": 1, "drum_empty": 2, "pump_trip": 3, "dosing_airlock": 4}
    ev_kinds = sorted(faults, key=lambda f: order[f]) + ["demand_step"]
    # rejection sampling of the whole schedule (one uniform draw per event in its window; the same rng, so the draw is
    # deterministic per seed)
    times = None
    for _attempt in range(20000):
        cand = [_u(rng, *windows[f]) for f in ev_kinds]
        srt = sorted(cand)
        if any(b - a < EVENT_GAP_S for a, b in zip(srt, srt[1:])):
            continue
        # a trip of P-101 inside the 360 s that judge the demand step would judge the trip instead (no operator
        # could hold the level through both): such a schedule is drawn again
        t_d = cand[ev_kinds.index("demand_step")]
        if any(f == "pump_trip" and t_d < t <= t_d + 360.0 for f, t in zip(ev_kinds, cand)):
            continue
        # with both dosing faults on one shift the air lock comes after drum A ran dry: an air lock while drum A is in
        # its last few percent would look like AOP-4 on every instrument (sop.md section 3 tells them apart by the drum)
        if "drum_empty" in ev_kinds and "dosing_airlock" in ev_kinds and \
                cand[ev_kinds.index("dosing_airlock")] < cand[ev_kinds.index("drum_empty")]:
            continue
        times = cand
        break
    if times is None:
        raise RuntimeError("no event schedule fits (seed %d)" % int(seed))
    events = []
    for f, t in zip(ev_kinds, times):
        if f == "filter_slug":
            events.append({"type": f, "t": t, "jump": _u(rng, 2.6, 3.4), "rate": _u(rng, 0.016, 0.024)})
        elif f == "drum_empty":
            proc["drumA_frac"] = _u(rng, 0.10, 0.22)
            events.append({"type": f, "t": t})
        elif f == "demand_step":
            d0 = proc["demand_m3h"]
            step = _u(rng, 0.6, 1.0) * (1 if rng.random() < 0.5 else -1)
            room = {1: 3.1 - d0, -1: d0 - 1.4}             # the new draw stays in 1.4-3.1 m3/h
            sgn = 1 if step > 0 else -1
            if room[sgn] < abs(step):
                sgn = -sgn if room[-sgn] >= abs(step) else max(room, key=room.get)
            step = sgn * min(abs(step), room[sgn])
            events.append({"type": f, "t": t, "delta_m3h": step})
        else:
            events.append({"type": f, "t": t})
    events.sort(key=lambda e: e["t"])
    light = _u(rng, 0.80, 1.20)
    equipment = []
    for s in ("V-101", "V-102", "V-103", "V-104", "V-105", "V-106", "V-107", "V-111", "V-112", "V-301", "V-302", "V-303",
              "P-101", "P-301", "T-101", "T-201", "T-301", "T-302", "F-101", "PI-101", "PI-103", "PDI-111", "FI-105", "LG-101",
              "LG-201", "AI-201", "LCS"):
        e = {"series_tag": s, "site_tag": ST[s], "service": SERVICES[s]}
        if s in STARTUP_POSITION:
            e["startup_position"] = STARTUP_POSITION[s]
        equipment.append(e)
    site["equipment"] = equipment
    site["log_sheet"] = [ST[s] for s in LOG_SERIES]
    return {"seed": int(seed), "mirror": mirror, "side": side, "meter_first": meter_first, "panel_p101_left": panel_p101_left,
            "z_suc": z_suc, "z_dis": z_dis, "z_mk": z_mk, "devices": dev, "styles": styles, "gauge_kinds": kinds,
            "initial": initial, "wrong_at_start": wrong, "process": proc, "events": events, "faults": faults, "light": light,
            "rail": dict(RAIL), "x_total": X_TOTAL, "site": site}


def gauge_reading(kind, value_base):
    """The reading of a gauge of `kind` for a process value in the base unit (bar, m3/h, %)."""
    k = GAUGE_KINDS[kind]
    return float(value_base) / float(k["scale"])
