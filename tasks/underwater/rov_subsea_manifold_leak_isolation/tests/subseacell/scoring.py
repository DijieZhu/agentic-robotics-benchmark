"""Station scoring of task 07 (grader side only; DESIGN v1.2 sections 3.2-3.5; decisions J2, M13; package P10).

The score of one station is a function of three grader-side inputs and nothing else (no value the program reports):

    site       the hidden site package (subseacell.site.Site, its files, bytes or a path): as-found valve angles,
               the leak branch, the panel / structure frames, the marker zones, the laydown area, the ports
    log        the run's event log (subseacell.events.EventLog.export() / EventLog.load(dir) / load_log(dir)):
               the 10 Hz series, the camera frames with the raw quantities of the "imaged" predicate, the world's
               discrete events (grasp, stab, protect, process, ...)
    terminal   the world's terminal state (capture_terminal(world), taken when the run ends, before the world is
               closed): final valve angles, stab state, what the jaws hold, marker / stand / vehicle centres of mass,
               speeds and the stand's collision geometry in world coordinates

This module never imports the world, the server, the physics engine or the sensor kit (tests/test_scoring.py checks it
statically and at run time); `capture_terminal` reads a world object by attribute only.  All weights are integer
hundredths, so a station score is an exact multiple of 0.01 and the six-station mean is an exact fraction.

Predicates (DESIGN 3.2; thresholds from the flat task values `score.*`, `valve.*`, `budget.protective_stops`):
  valve closed / open     hinge angle >= 80 deg closed, <= 10 deg open; the process treats < 80 deg as conducting
  critical error (CE)     a valve outside the isolation set whose conduction differs from its as-found conduction
                          for >= 0.2 s at any time, also if it is turned back later.  Read on the process
                          connectivity sequence (DESIGN 3.2): the log's rows, each row's state holding until the next
                          row (the cell updates the process every 0.1 s); duration = time of the first row back - time
                          of the first changed row, so two changed 0.1 s rows are a CE and one is not.  A change still
                          present at the last row persists and counts.
  misoperation            a valve outside the isolation set whose final angle differs from as-found by > 10 deg
  isolation set           the leak branch's {BV, PV, TV} that conduct as found (1-3 valves)
  imaged                  per frame and tag plate, all of: the code's four outer corners in the image with a 20 px
                          margin, slant range camera -> code centre <= 1.3 m, shortest projected edge of the code's
                          8 x 8 cells >= 4 px, line of sight within 60 deg of the plate normal, lamp level >= 0.3 and
                          the code centre within half the lamp beam of at least one lamp, >= 20 of the code's 5 x 5
                          sample points unoccluded.  Re-evaluated here from the raw quantities the cell recorded at
                          the frame (subseacell.events.ImagedJudge); where the cell's own verdict differs only
                          because a quantity was rounded for the log at its threshold, the cell's verdict stands.
  survey complete (V)     >= 15 of the 18 valve positions imaged in some frame
  at rest                 the last 5 s of the run (rows of the series + the terminal state): object speed < 1 cm/s,
                          vehicle < 2 cm/s
  stand grasp event       a grasp event of one of the stand's eyes (src grasp, ev grasp, kind eye)
  vehicle disabled        the protective-stop count reached budget.protective_stops (6)
Items (DESIGN 3.3; weights score.weights, sum 1.00):
  M1 0.03  marker at rest, touching the roof grating, not held, over the last 5 s
  M2 0.12  M1 and its centre of mass horizontally within the leak branch's marker zone
  H  0.12  V, no CE, vehicle not disabled
  I1 0.08  at least one isolation-set valve closed at the end
  I2 0.22  every isolation-set valve closed at the end
  I3 0.08  I2 and no misoperation
  D1 0.07  the stand's nearest point >= 1.2 m from the panel face at the end, and a stand grasp event happened
  D2 0.05  D1, the stand at rest and its centre of mass horizontally inside the laydown area
  S1 0.04  the stab left its initial port and ends Locked in the leak branch's test port or a parking port
  S2 0.14  the stab ends Locked in the leak branch's test port
  R  0.05  V, the vehicle inside the cage and at rest over the last 5 s
  A CE sets the items in score.ce_zeroes (I1, I2, I3, S1, S2, H) to 0; M, D and R are not affected.  Station score
  = sum of the items' weights, floor 0; no cross-station penalty.  Aggregate (DESIGN 3.5):
  intervention_score = contract_ok x mean of the station scores; reward = clamp(raw / anchor / full_at, 0, 1).

    from subseacell import scoring as SC
    rules = SC.Rules.from_values(values)                 # flat task values (budget.load_task_values)
    term = SC.capture_terminal(server.world)             # cell side, at the end of the run
    res = SC.score_station(site_pkg, server.event_log().export(), term, rules)
    res["items"], res["points"], res["score"], res["details"]
    SC.aggregate([res, ...], contract_ok=True, anchor=1.0, full_at=0.95)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import numpy as np

SCHEMA = "subseacell-score/1"
LOG_SCHEMA = "subseacell-events/1"
TERMINAL_SCHEMA = "subseacell-terminal/1"
SITE_TRUTH_SCHEMA = "subseacell-site-truth/1"
ITEMS = ("M1", "M2", "H", "I1", "I2", "I3", "D1", "D2", "S1", "S2", "R")
VALVE_KINDS = ("BV", "PV", "TV")
N_BRANCHES = 6
VALVE_TAGS = tuple(f"{k}{b}" for k in VALVE_KINDS for b in range(1, N_BRANCHES + 1))
STAB_LOCKED = "Locked"
EPS_T = 1e-9                     # time comparisons (s)
EPS_X = 1e-12                    # length / angle comparisons
# the stand's lifting eyes are not collision geometry; for the nearest-point test each eye counts as a sphere that
# contains its ring (radius 0.045 m, tube 3.5 mm) and the top of its post (6 mm): subseacell.objects.FRAME_DEFAULTS
EYE_BOUND_R = 0.051
# rounding of the predicate's raw quantities in the log (subseacell.events.ImagedJudge.evaluate)
LOG_ROUNDING = dict(margin_px=1e-3, range_m=1e-5, angle_deg=1e-3, cell_px=1e-4, lamp_deg=1e-3)

# Every decision of this module, one line each (the documents P22 / P23 map their sentences to these ids).
CONDITIONS = (
    ("valve.closed", "a valve is closed when its hinge angle is >= valve.closed_min_deg (80 deg)"),
    ("valve.open", "a valve is open when its hinge angle is <= valve.open_max_deg (10 deg); below 80 deg it conducts"),
    ("isolation.set", "the isolation set is the leak branch's BV, PV and TV that conduct as found"),
    ("ce.definition", "critical error: a valve outside the isolation set changes its conduction against as-found "
                      "for >= score.ce_debounce_s (0.2 s) of the process state sequence (updated every 0.1 s), at any "
                      "time, also if turned back; still changed at the end counts"),
    ("ce.zeroes", "a critical error sets score.ce_zeroes (I1, I2, I3, S1, S2, H) to 0 for that station"),
    ("misoperation", "a valve outside the isolation set ends more than score.misoperation_deg (10 deg) from as-found"),
    ("imaged.corners", "the code's four outer corners lie in the image with score.imaged.edge_margin_px (20 px) margin"),
    ("imaged.range", "slant range camera -> code centre <= score.imaged.slant_max_m (1.3 m)"),
    ("imaged.cells", "the shortest projected edge of the code's 8 x 8 cells >= score.imaged.cell_min_px (4 px)"),
    ("imaged.angle", "line of sight within score.imaged.angle_max_deg (60 deg) of the plate normal"),
    ("imaged.light", "lamp level >= score.imaged.light_level_min (0.3) and the code centre inside half the lamp beam "
                     "of at least one lamp"),
    ("imaged.occlusion", ">= 20 of the code's 5 x 5 sample points are unoccluded (jaws, rail, stand, held objects "
                         "occlude; fouling does not)"),
    ("survey", "the survey is complete when >= score.survey_min_imaged (15) of the 18 valve positions were imaged"),
    ("rest.object", "an object is at rest when its speed < score.rest_object_speed_m_s (0.01 m/s) over the last "
                    "score.rest_window_s (5 s)"),
    ("rest.vehicle", "the vehicle is at rest when its speed < score.rest_vehicle_speed_m_s (0.02 m/s) over the last 5 s"),
    ("disabled", "the vehicle is disabled at budget.protective_stops (6) protective stops"),
    ("item.M1", "M1: the marker rests on the roof grating (touching it, not held) over the last 5 s"),
    ("item.M2", "M2: M1 and the marker's centre of mass horizontally inside the leak branch's marker zone"),
    ("item.H", "H: survey complete, no critical error, vehicle not disabled"),
    ("item.I1", "I1: at least one isolation-set valve closed at the end"),
    ("item.I2", "I2: every isolation-set valve closed at the end"),
    ("item.I3", "I3: I2 and no misoperation"),
    ("item.D1", "D1: the stand's nearest point >= score.debris_clear_distance_m (1.2 m) from the panel face at the "
                "end, and the vehicle held one of its eyes at some time (score.debris_needs_grasp_event)"),
    ("item.D2", "D2: D1, the stand at rest and its centre of mass inside the laydown area"),
    ("item.S1", "S1: the stab left its initial port and ends Locked in the leak branch's test port or a parking port"),
    ("item.S2", "S2: the stab ends Locked in the leak branch's test port"),
    ("item.R", "R: survey complete, the vehicle inside the cage and at rest over the last 5 s"),
    ("station", "station score = sum of the weights score.weights of the items earned (no negative points)"),
    ("aggregate", "intervention_score = contract_ok x mean station score; reward = clamp(raw / anchor / full_at, 0, 1)"),
)


class ScoringError(ValueError):
    """A malformed or inconsistent input (the cell's fault, not the program's)."""


# ===================================================================================================== rules
def _centi(x, name) -> int:
    c = round(float(x) * 100.0)
    if abs(c - float(x) * 100.0) > 1e-6:
        raise ScoringError(f"{name}: {x} is not a whole number of hundredths")
    if c < 0:
        raise ScoringError(f"{name}: negative weight")
    return int(c)


@dataclass(frozen=True)
class Rules:
    weights_c: dict                       # item -> integer hundredths (sum 100)
    ce_zeroes: tuple
    closed_min_deg: float = 80.0
    open_max_deg: float = 10.0
    misoperation_deg: float = 10.0
    ce_debounce_s: float = 0.2
    rest_window_s: float = 5.0
    rest_object_m_s: float = 0.01
    rest_vehicle_m_s: float = 0.02
    survey_min: int = 15
    valve_positions: int = 18
    debris_clear_m: float = 1.2
    debris_needs_grasp: bool = True
    max_stops: int = 6
    candidate_branches: int = 6
    imaged: dict = field(default_factory=dict)

    @classmethod
    def from_values(cls, values: dict, weights: Optional[dict] = None, variant: str = "full") -> "Rules":
        """From the flat task values.  variant "no_debris" = DESIGN 2.3.3 fallback (5): the debris items dropped and
        score.fallback_no_debris_weights applied (only if P02's fallback ladder reaches step 5)."""
        w = dict(values["score.weights"] if weights is None else weights)
        if variant == "no_debris":
            w.update(values["score.fallback_no_debris_weights"])
            w["D1"] = w["D2"] = 0.0
        elif variant != "full":
            raise ScoringError(f"unknown rules variant {variant!r}")
        if set(w) != set(ITEMS):
            raise ScoringError(f"weights must name exactly the 11 items, got {sorted(w)}")
        wc = {k: _centi(w[k], f"weight {k}") for k in ITEMS}
        if sum(wc.values()) != 100:
            raise ScoringError(f"weights sum to {sum(wc.values())}/100, want 1.00")
        cz = tuple(values["score.ce_zeroes"])
        if not set(cz) <= set(ITEMS):
            raise ScoringError("score.ce_zeroes names an unknown item")
        s = values["score.imaged.samples"]
        img = dict(margin_px=float(values["score.imaged.edge_margin_px"]),
                   max_range_m=float(values["score.imaged.slant_max_m"]),
                   min_cell_px=float(values["score.imaged.cell_min_px"]),
                   max_angle_deg=float(values["score.imaged.angle_max_deg"]),
                   min_lamp=float(values["score.imaged.light_level_min"]),
                   grid=[int(s["grid"][0]), int(s["grid"][1])], min_visible=int(s["unoccluded_min"]))
        return cls(weights_c=wc, ce_zeroes=cz, closed_min_deg=float(values["valve.closed_min_deg"]),
                   open_max_deg=float(values["valve.open_max_deg"]), misoperation_deg=float(values["score.misoperation_deg"]),
                   ce_debounce_s=float(values["score.ce_debounce_s"]), rest_window_s=float(values["score.rest_window_s"]),
                   rest_object_m_s=float(values["score.rest_object_speed_m_s"]),
                   rest_vehicle_m_s=float(values["score.rest_vehicle_speed_m_s"]),
                   survey_min=int(values["score.survey_min_imaged"]), valve_positions=int(values["score.valve_positions"]),
                   debris_clear_m=float(values["score.debris_clear_distance_m"]),
                   debris_needs_grasp=bool(values["score.debris_needs_grasp_event"]),
                   max_stops=int(values["budget.protective_stops"]),
                   candidate_branches=int(values.get("score.candidate_branches", N_BRANCHES)), imaged=img)

    @classmethod
    def default(cls) -> "Rules":
        """From $SUBSEACELL_TASK_VALUES or the development layout (subseacell.budget.default_values_path)."""
        from . import budget as _BU
        return cls.from_values(_BU.load_task_values(_BU.default_values_path()))

    def weight(self, item) -> Fraction:
        return Fraction(self.weights_c[item], 100)

    def as_dict(self) -> dict:
        return dict(weights_c=dict(self.weights_c), ce_zeroes=list(self.ce_zeroes), closed_min_deg=self.closed_min_deg,
                    open_max_deg=self.open_max_deg, misoperation_deg=self.misoperation_deg,
                    ce_debounce_s=self.ce_debounce_s, rest_window_s=self.rest_window_s,
                    rest_object_m_s=self.rest_object_m_s, rest_vehicle_m_s=self.rest_vehicle_m_s,
                    survey_min=self.survey_min, valve_positions=self.valve_positions, debris_clear_m=self.debris_clear_m,
                    debris_needs_grasp=self.debris_needs_grasp, max_stops=self.max_stops,
                    candidate_branches=self.candidate_branches, imaged=dict(self.imaged))


# ===================================================================================================== site truth
def _site_parts(site):
    """(data, manifold, skid) of a site given as subseacell.site.Site, {name: bytes} files, package bytes or a path."""
    if hasattr(site, "data") and hasattr(site, "manifold") and hasattr(site, "skid"):
        return site.data, site.manifold, site.skid
    if isinstance(site, dict) and {"data", "manifold", "skid"} <= set(site):
        return site["data"], site["manifold"], site["skid"]
    from . import site as _SITE                    # package parsing only (stdlib + numpy)
    if isinstance(site, dict):
        s = _SITE.Site.from_files(site)
    else:
        s = _SITE.load(os.fspath(site) if not isinstance(site, (bytes, bytearray)) else bytes(site))
    return s.data, s.manifold, s.skid


def _rz(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def site_truth(site, rules: Optional[Rules] = None) -> dict:
    """What scoring needs from the hidden site, JSON-able (world frame W; panel frame P: X along the face, Y up, Z out
    of the face, origin on the face plane at the top of the mudmat)."""
    rules = rules or Rules.default()
    d, man, skid = _site_parts(site)
    valves = {v["tag"]: dict(theta0_deg=float(v["theta0_deg"]), branch=int(v["branch"]), kind=str(v["kind"]),
                             key=str(v["key"])) for v in d["valves"]}
    if set(valves) != set(VALVE_TAGS):
        raise ScoringError("site: the valve tags must be BV1..TV6")
    leak = int(d["leak_branch"])
    cond0 = {t: valves[t]["theta0_deg"] < rules.closed_min_deg for t in VALVE_TAGS}
    iso = [f"{k}{leak}" for k in VALVE_KINDS if cond0[f"{k}{leak}"]]
    if not iso:
        raise ScoringError("site: the leak branch has no open valve (DESIGN 2.1: every branch has one)")
    o_s = np.asarray(d["pose_o"], float)
    R_s = _rz(float(d["pose_psi"]))
    pf = d["panel_frame_S"]
    R_sp = np.stack([np.asarray(pf["X"], float), np.asarray(pf["Y"], float), np.asarray(pf["Z"], float)], 1)
    P_o = R_s @ np.asarray(pf["origin"], float) + o_s
    P_R = R_s @ R_sp
    zones = [z for z in man.get("marker_zones", ()) if int(z["branch"]) == leak]
    if len(zones) != 1:
        raise ScoringError("site: the manifold description has no single marker zone for the leak branch")
    z = zones[0]
    zone_c = R_s @ np.asarray(z["centre"], float) + o_s
    ports = {p["name"]: dict(kind=str(p["kind"]), branch=p.get("branch")) for p in d["ports"]}
    tps = [n for n, p in ports.items() if p["kind"] == "test" and p["branch"] is not None and int(p["branch"]) == leak]
    if len(tps) != 1:
        raise ScoringError("site: no single test port on the leak branch")
    parks = sorted(n for n, p in ports.items() if p["kind"] == "park")
    lay = skid["laydown"]
    return dict(schema=SITE_TRUTH_SCHEMA, label=d.get("label"), valves=valves, leak_branch=leak,
                as_found_conducting=cond0, isolation_set=iso, leak_test_port=tps[0], park_ports=parks,
                initial_port=str(d["stab"].get("initial_port", "PARK-A")),
                zone=dict(centre_w=zone_c.tolist(), radius_m=float(z["radius"])),
                panel=dict(origin_w=P_o.tolist(), R_w=P_R.tolist()),
                laydown=dict(x=[float(lay["x"][0]), float(lay["x"][1])], z=[float(lay["z"][0]), float(lay["z"][1])]))


# ===================================================================================================== terminal state
def capture_terminal(world) -> dict:
    """The world's terminal state for scoring (cell side; call when the run has ended, before the world is closed).
    Reads the world object by attribute only (no import of it)."""
    m, d = world.m, world.d
    tr = world.truth()

    def com(name):
        return [float(x) for x in d.xipos[m.body(name).id]]
    db = m.body("debris").id
    x, R = np.asarray(d.xpos[db], float), np.asarray(d.xmat[db], float).reshape(3, 3)
    fs = world.debris_spec
    members = [[(x + R @ np.asarray(a, float)).tolist(), (x + R @ np.asarray(b, float)).tolist()] for a, b, _ in fs.members]
    eyes = {}
    for name, e in getattr(world, "eyes", {}).items():
        p_b = e.get("point_b", e.get("p_b"))
        if p_b is not None:
            eyes[name] = (x + R @ np.asarray(p_b, float)).tolist()
    held = tr["gripper"].get("held")
    return _jsonable(dict(
        schema=TERMINAL_SCHEMA, t=float(tr["t"]), ended=bool(tr["ended"]), end_reason=tr.get("end_reason"),
        valves={k: float(v) for k, v in tr["valves"].items()},
        stab=dict(state=tr["stab"]["state"], port=tr["stab"].get("port"), left_initial=bool(tr["stab"].get("left_initial"))),
        gripper=dict(state=tr["gripper"]["state"], held=list(held) if held is not None else None),
        marker=dict(com_w=com("marker"), pos_w=list(tr["marker"]["pos"]), speed=float(tr["marker"]["speed"])),
        debris=dict(com_w=com("debris"), pos_w=x.tolist(), R_w=R.tolist(), speed=float(tr["debris"]["speed"]),
                    tube_radius_m=float(fs.r), members_w=members, eyes_w=eyes, eye_bound_m=EYE_BOUND_R),
        vehicle=dict(pos_w=list(tr["vehicle"]["pos"]), speed=float(tr["vehicle"]["speed"]),
                     in_cage=bool(tr["vehicle"]["in_cage"])),
        protect=dict(count=int(tr["protect"]["count"]), ended=bool(tr["protect"]["ended"]))))


# ===================================================================================================== helpers
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, Fraction):
        return f"{x.numerator}/{x.denominator}"
    return x


def canonical(obj) -> bytes:
    return (json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            + "\n").encode()


def load_log(folder: str) -> dict:
    """events.json + series.npz written by subseacell.events.EventLog.save (same as EventLog.load)."""
    with open(os.path.join(folder, "events.json"), "rb") as f:
        doc = json.loads(f.read().decode("utf-8"))
    with np.load(os.path.join(folder, "series.npz")) as z:
        doc["series_f"], doc["series_i"] = np.array(z["series_f"]), np.array(z["series_i"])
    return doc


def log_digest(doc: dict) -> str:
    """sha256 of a log document, computed like subseacell.events.EventLog.digest."""
    h = hashlib.sha256()
    h.update(canonical(dict(meta=doc["meta"], events=doc["events"], frames=doc["frames"])))
    h.update(np.ascontiguousarray(np.asarray(doc["series_f"], np.float64), "<f8").tobytes())
    h.update(np.ascontiguousarray(np.asarray(doc["series_i"], np.int64), "<i8").tobytes())
    return h.hexdigest()


class _Series:
    def __init__(self, doc):
        meta = doc.get("meta") or {}
        if meta.get("schema") != LOG_SCHEMA:
            raise ScoringError(f"log schema {meta.get('schema')!r}, want {LOG_SCHEMA!r}")
        self.meta = meta
        self.F = np.asarray(doc["series_f"], np.float64)
        self.I = np.asarray(doc["series_i"], np.int64)
        cf, ci = list(meta["columns_f"]), list(meta["columns_i"])
        if self.F.ndim != 2 or self.F.shape[1] != len(cf) or self.I.ndim != 2 or self.I.shape[1] != len(ci) \
                or self.F.shape[0] != self.I.shape[0] or self.F.shape[0] < 1:
            raise ScoringError("log series: shape does not match its columns")
        self.cf = {n: k for k, n in enumerate(cf)}
        self.ci = {n: k for k, n in enumerate(ci)}
        self.t = self.F[:, self.cf["t"]]
        if not np.all(np.isfinite(self.t)) or np.any(np.diff(self.t) <= 0.0):
            raise ScoringError("log series: time must be finite and strictly increasing")
        names = list(meta["valve_names"])
        if sorted(names) != sorted(VALVE_TAGS):
            raise ScoringError("log: valve_names must be the 18 tags")
        self.valve_names = names
        for n in names:
            if f"valve_{n}_deg" not in self.cf:
                raise ScoringError(f"log series: no column for {n}")
        A = np.stack([self.F[:, self.cf[f"valve_{n}_deg"]] for n in names], 1)
        if not np.all(np.isfinite(A)):
            raise ScoringError("log series: non-finite valve angle")
        self.A = A

    def f(self, name):
        return self.F[:, self.cf[name]]

    def i(self, name):
        return self.I[:, self.ci[name]]


# ===================================================================================================== critical errors
def excursions(t, a, closed_min_deg, cond0):
    """Maximal time intervals in which the valve's conduction differs from cond0 (True = conducting as found), on
    the rows (t, a) with each row's state holding until the next row: [(time of the first changed row, time of the
    first row back, or None if still changed at the last row)]."""
    t = np.asarray(t, float)
    a = np.asarray(a, float)
    changed = (a < closed_min_deg) != bool(cond0)
    if not changed.any():
        return []
    idx = np.flatnonzero(changed)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    n = len(t)
    return [(float(t[i]), None if j == n - 1 else float(t[j + 1])) for i, j in zip(starts, ends)]


def critical_errors(series_t, angles, valve_names, st: dict, rules: Rules) -> list:
    """CE list (one entry per valve outside the isolation set: its first qualifying excursion)."""
    iso = set(st["isolation_set"])
    out = []
    for k, tag in enumerate(valve_names):
        if tag in iso:
            continue
        c0 = bool(st["as_found_conducting"][tag])
        for ts, te in excursions(series_t, angles[:, k], rules.closed_min_deg, c0):
            dur = None if te is None else te - ts
            if te is None or dur >= rules.ce_debounce_s - EPS_T:
                out.append(dict(tag=tag, t_start=round(ts, 6), t_end=None if te is None else round(te, 6),
                                duration_s=None if dur is None else round(dur, 6), persists_at_end=te is None,
                                kind="closed a conducting valve" if c0 else "opened a closed valve"))
                break
    return out


# ===================================================================================================== imaged
def _cond_table(rec, lamp, thr):
    """[(name, passes, value, threshold, rounding unit or None)] of one plate record."""
    vis = rec.get("visible")
    return [
        ("in_front", bool(rec.get("in_front")), None, None, None),
        ("corners", float(rec["margin_px"]) >= thr["margin_px"], float(rec["margin_px"]), thr["margin_px"],
         LOG_ROUNDING["margin_px"]),
        ("range", float(rec["range_m"]) <= thr["max_range_m"], float(rec["range_m"]), thr["max_range_m"],
         LOG_ROUNDING["range_m"]),
        ("cells", float(rec["cell_px"]) >= thr["min_cell_px"], float(rec["cell_px"]), thr["min_cell_px"],
         LOG_ROUNDING["cell_px"]),
        ("angle", float(rec["angle_deg"]) <= thr["max_angle_deg"], float(rec["angle_deg"]), thr["max_angle_deg"],
         LOG_ROUNDING["angle_deg"]),
        ("lamp_level", float(lamp) >= thr["min_lamp"], float(lamp), thr["min_lamp"], None),
        ("lamp_cone", float(rec["lamp_deg"]) <= thr["lamp_half_angle_deg"], float(rec["lamp_deg"]),
         thr["lamp_half_angle_deg"], LOG_ROUNDING["lamp_deg"]),
        ("occlusion", vis is not None and int(vis) >= thr["min_visible"], vis, thr["min_visible"], None),
    ]


def plate_verdict(rec: dict, lamp: float, thr: dict) -> tuple:
    """(holds, how) for one plate record of a frame.  how = "recomputed" | "rounding" (the cell's verdict kept: only
    rounded quantities at their thresholds disagree) | "inconsistent" (the record's verdict differs beyond rounding;
    the recomputed one is used)."""
    tab = _cond_table(rec, lamp, thr)
    ok = all(c[1] for c in tab)
    recorded = rec.get("imaged")
    if recorded is None or bool(recorded) == ok:
        return ok, "recomputed"
    amb = [c for c in tab if c[4] is not None and abs(c[2] - c[3]) <= 0.5 * c[4] + EPS_X]
    if recorded:
        flip_ok = all(c[1] or c in amb for c in tab)
    else:
        flip_ok = len(amb) > 0
    return (bool(recorded), "rounding") if flip_ok else (ok, "inconsistent")


def predicate_thresholds(meta: dict, rules: Rules, n_frames: Optional[int] = None) -> dict:
    """The predicate's thresholds: the scoring rules + the lamps' half beam angle the cell recorded; the cell's other
    recorded thresholds must equal the rules.  A log without a single camera frame (n_frames == 0) needs none: no
    plate was imaged, the rules' thresholds stand and the lamps' angle is not used (P24a, decision P18-4: a program
    that takes no frame scores what it did, never a grader error)."""
    thr = dict(rules.imaged)
    mt = meta.get("thresholds")
    if not mt and n_frames == 0:
        thr["lamp_half_angle_deg"] = None
        return thr
    if not mt:
        raise ScoringError("log: no predicate thresholds (the cell ran without its judge)")
    for k in ("margin_px", "max_range_m", "min_cell_px", "max_angle_deg", "min_lamp", "min_visible"):
        if abs(float(mt[k]) - float(thr[k])) > EPS_X:
            raise ScoringError(f"log: the cell's predicate threshold {k} = {mt[k]} differs from the rules ({thr[k]})")
    if list(mt.get("grid", thr["grid"])) != list(thr["grid"]):
        raise ScoringError("log: the cell's occlusion sample grid differs from the rules")
    half = float(mt["lamp_half_angle_deg"])
    if not 0.0 < half < 90.0:
        raise ScoringError("log: lamp half beam angle out of range")
    thr["lamp_half_angle_deg"] = half
    return thr


def frame_verdicts(doc: dict, rules: Rules) -> list:
    """[(frame index, {label: (holds, how)})] for every frame of the log."""
    thr = predicate_thresholds(doc["meta"], rules, n_frames=len(doc.get("frames") or ()))
    out = []
    for k, fr in enumerate(doc.get("frames") or ()):
        lamp = float(fr["lamp"])
        out.append((int(fr.get("i", k)), {lab: plate_verdict(rec, lamp, thr) for lab, rec in (fr.get("plates") or {}).items()}))
    return out


# ===================================================================================================== geometry
def debris_face_distance(term_debris: dict, st: dict) -> float:
    """Distance of the stand's nearest point to the panel face plane (P: Z = 0), from the terminal geometry: the
    collision tubes are capsules (radius tube_radius_m) on the member segments; each eye a sphere of eye_bound_m."""
    o = np.asarray(st["panel"]["origin_w"], float)
    n = np.asarray(st["panel"]["R_w"], float)[:, 2]
    r = float(term_debris["tube_radius_m"])
    best = math.inf
    for a, b in term_debris["members_w"]:
        best = min(best, float(n @ (np.asarray(a, float) - o)) - r, float(n @ (np.asarray(b, float) - o)) - r)
    rb = float(term_debris.get("eye_bound_m", EYE_BOUND_R))
    for c in (term_debris.get("eyes_w") or {}).values():
        best = min(best, float(n @ (np.asarray(c, float) - o)) - rb)
    if not math.isfinite(best):
        raise ScoringError("terminal: no stand geometry")
    return best


def in_laydown(com_w, st: dict) -> tuple:
    """(inside, (X, Z)) of a point's horizontal projection in the panel frame vs the laydown rectangle."""
    o = np.asarray(st["panel"]["origin_w"], float)
    R = np.asarray(st["panel"]["R_w"], float)
    q = R.T @ (np.asarray(com_w, float) - o)
    (x0, x1), (z0, z1) = st["laydown"]["x"], st["laydown"]["z"]
    return bool(x0 <= q[0] <= x1 and z0 <= q[2] <= z1), (float(q[0]), float(q[2]))


# ===================================================================================================== the station
def score_station(site, log: dict, terminal: dict, rules: Optional[Rules] = None) -> dict:
    rules = rules or Rules.default()
    st = site if isinstance(site, dict) and site.get("schema") == SITE_TRUTH_SCHEMA else site_truth(site, rules)
    if not isinstance(terminal, dict) or terminal.get("schema") != TERMINAL_SCHEMA:
        raise ScoringError("terminal state missing or of the wrong schema")
    S = _Series(log)
    meta = S.meta
    t = S.t
    t_end = float(terminal["t"])
    if t_end < float(t[-1]) - EPS_T:
        raise ScoringError("terminal time before the last row of the series")
    checks = dict(mask_mismatch_rows=0, imaged_rounding=0, imaged_inconsistent=0, ce_without_process_event=0)
    # ---------------------------------------------------------------- valves: CE, final state, misoperation
    angles = S.A
    if "conducting_mask" in S.ci:
        bits = np.array([1 << k for k in range(len(S.valve_names))], np.int64)
        mask = ((angles < rules.closed_min_deg) * bits).sum(1)
        checks["mask_mismatch_rows"] = int(np.count_nonzero(mask != S.i("conducting_mask")))
    ce = critical_errors(t, angles, S.valve_names, st, rules)
    proc = {e.get("tag") for e in log.get("events") or () if e.get("src") == "process" and e.get("ev") == "conduction"}
    checks["ce_without_process_event"] = sum(1 for c in ce if c["tag"] not in proc)
    fin = dict(terminal["valves"])
    if set(fin) != set(VALVE_TAGS):
        raise ScoringError("terminal: valves must be the 18 tags")
    iso = list(st["isolation_set"])
    closed = {k: float(fin[k]) >= rules.closed_min_deg for k in VALVE_TAGS}
    misop = sorted(k for k in VALVE_TAGS if k not in iso
                   and abs(float(fin[k]) - st["valves"][k]["theta0_deg"]) > rules.misoperation_deg + EPS_X)
    # ---------------------------------------------------------------- survey
    thr = predicate_thresholds(meta, rules, n_frames=len(log.get("frames") or ()))
    imaged_first = {}
    for k, fr in enumerate(log.get("frames") or ()):
        lamp = float(fr["lamp"])
        for lab, rec in (fr.get("plates") or {}).items():
            ok, how = plate_verdict(rec, lamp, thr)
            if how == "rounding":
                checks["imaged_rounding"] += 1
            elif how == "inconsistent":
                checks["imaged_inconsistent"] += 1
            if ok and lab not in imaged_first:
                imaged_first[lab] = int(fr.get("i", k))
    valves_imaged = sorted(k for k in imaged_first if k in VALVE_TAGS)
    V = len(valves_imaged) >= rules.survey_min
    # ---------------------------------------------------------------- the end window
    W = rules.rest_window_s
    full = t_end >= W - EPS_T and float(t[0]) <= t_end - W + EPS_T
    tail = t >= t_end - W - EPS_T
    tail_rows = int(np.count_nonzero(tail))

    def still(col, thr_v, term_speed):
        if not full:
            return False
        v = S.f(col)[tail]
        return bool(np.all(v < thr_v) and float(term_speed) < thr_v)
    held_names = list(meta.get("held_names") or [])
    i_marker = held_names.index("marker") if "marker" in held_names else -2
    held_marker_tail = bool(np.any(S.i("held_name")[tail] == i_marker))
    th = terminal["gripper"].get("held")
    held_marker_end = th is not None and len(th) == 2 and th[1] == "marker"
    # ---------------------------------------------------------------- protect
    max_count = int(max(int(S.i("protect_count").max()), int(terminal["protect"]["count"])))
    disabled = (max_count >= rules.max_stops
                or any(e.get("src") == "protect" and e.get("ev") == "vehicle_disabled" for e in log.get("events") or ())
                or terminal.get("end_reason") == "disabled")
    # ---------------------------------------------------------------- items (before the CE rule)
    mk = terminal["marker"]
    mk_still = still("marker_speed", rules.rest_object_m_s, mk["speed"])
    mk_roof = bool(full and np.all(S.i("marker_on_roof")[tail] == 1))
    M1 = bool(mk_still and mk_roof and not held_marker_tail and not held_marker_end)
    zc = np.asarray(st["zone"]["centre_w"], float)
    zone_d = float(math.hypot(mk["com_w"][0] - zc[0], mk["com_w"][1] - zc[1]))
    M2 = bool(M1 and zone_d <= st["zone"]["radius_m"] + EPS_X)
    H = bool(V and not disabled)
    I1 = any(closed[k] for k in iso)
    I2 = all(closed[k] for k in iso)
    I3 = bool(I2 and not misop)
    de = terminal["debris"]
    face_d = debris_face_distance(de, st)
    grasp_eyes = [e for e in log.get("events") or () if e.get("src") == "grasp" and e.get("ev") == "grasp"
                  and e.get("kind") == "eye" and (e.get("body") == "debris" or str(e.get("eye", "")).startswith("debris"))]
    D1 = bool(face_d >= rules.debris_clear_m - EPS_X and (grasp_eyes or not rules.debris_needs_grasp))
    deb_still = still("debris_speed", rules.rest_object_m_s, de["speed"])
    lay_in, lay_xz = in_laydown(de["com_w"], st)
    D2 = bool(D1 and deb_still and lay_in)
    sb = terminal["stab"]
    left = bool(sb.get("left_initial")) or bool(np.any(S.i("stab_left_initial") == 1))
    locked = sb.get("state") == STAB_LOCKED
    S1 = bool(left and locked and sb.get("port") in set([st["leak_test_port"]] + list(st["park_ports"])))
    S2 = bool(locked and sb.get("port") == st["leak_test_port"])
    veh = terminal["vehicle"]
    veh_in = bool(full and np.all(S.i("in_cage")[tail] == 1) and veh["in_cage"])
    veh_still = still("hull_speed", rules.rest_vehicle_m_s, veh["speed"])
    R_ = bool(V and veh_in and veh_still)
    raw = dict(M1=M1, M2=M2, H=H, I1=I1, I2=I2, I3=I3, D1=D1, D2=D2, S1=S1, S2=S2, R=R_)
    items = dict(raw)
    if ce:
        for k in rules.ce_zeroes:
            items[k] = False
    points = sum(rules.weights_c[k] for k in ITEMS if items[k])
    points = max(0, points)
    frac = Fraction(points, 100)
    details = dict(
        site=st.get("label"), t_end=t_end, rows=int(len(t)), tail_rows=tail_rows, end_window_full=bool(full),
        isolation_set=iso, leak_branch=st["leak_branch"], critical_errors=ce, misoperated=misop,
        final_angles={k: round(float(fin[k]), 4) for k in VALVE_TAGS}, closed=[k for k in VALVE_TAGS if closed[k]],
        n_imaged=len(valves_imaged), valves_imaged=valves_imaged,
        ports_imaged=sorted(k for k in imaged_first if k not in VALVE_TAGS), survey_complete=bool(V),
        frames=len(log.get("frames") or ()), disabled=bool(disabled), protective_stops=max_count,
        marker=dict(at_rest=mk_still, on_roof=mk_roof, held=bool(held_marker_tail or held_marker_end),
                    zone_distance_m=round(zone_d, 4), zone_radius_m=st["zone"]["radius_m"]),
        debris=dict(face_distance_m=round(face_d, 4), grasp_events=len(grasp_eyes),
                    grasped_eyes=sorted({str(e.get("eye")) for e in grasp_eyes}), at_rest=deb_still,
                    in_laydown=lay_in, laydown_xz_m=[round(lay_xz[0], 4), round(lay_xz[1], 4)]),
        stab=dict(state=sb.get("state"), port=sb.get("port"), left_initial=left, leak_test_port=st["leak_test_port"]),
        vehicle=dict(in_cage=veh_in, at_rest=veh_still, speed_end=round(float(veh["speed"]), 5)),
        items_before_ce=raw, checks=checks)
    return dict(schema=SCHEMA, items=items, points=int(points), score=points / 100.0,
                score_fraction=f"{frac.numerator}/{frac.denominator}", details=_jsonable(details),
                log_digest=log_digest(log), rules=rules.as_dict())


def station_fraction(res: dict) -> Fraction:
    return Fraction(int(res["points"]), 100)


def aggregate(stations, contract_ok: bool = True, anchor: float = 1.0, full_at: float = 0.95) -> dict:
    """DESIGN 3.5: intervention_score = contract_ok x mean station score; reward = clamp(raw / anchor / full_at, 0, 1).
    Exact rational arithmetic; floats only at the end."""
    pts = [station_fraction(r) for r in stations]
    if not pts:
        raise ScoringError("no stations")
    mean = sum(pts, Fraction(0)) / len(pts)
    raw = mean if contract_ok else Fraction(0)
    a, f = Fraction(repr(float(anchor))), Fraction(repr(float(full_at)))      # decimal values (0.95 = 19/20)
    if a <= 0 or f <= 0:
        raise ScoringError("anchor and full_at must be positive")
    reward = min(Fraction(1), max(Fraction(0), raw / a / f))
    return dict(intervention_score=float(raw), reward=float(reward), contract_ok=bool(contract_ok),
                station_scores=[float(p) for p in pts], mean_fraction=f"{mean.numerator}/{mean.denominator}",
                anchor=float(anchor), full_at=float(full_at))


def random_branch_increment_analytic(rules: Rules) -> Fraction:
    """DESIGN 3.4: expected gain of 'close one of the n candidate branches at random' over 'touch no valve' for a
    program that has done the survey: (I + S) / n - H (n - 1) / n."""
    n = rules.candidate_branches
    IS = sum(rules.weight(k) for k in ("I1", "I2", "I3", "S1", "S2"))
    return IS / n - rules.weight("H") * (n - 1) / n
