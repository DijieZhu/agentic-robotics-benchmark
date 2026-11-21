"""telemetry_synth.py -- the station-side records of one site of freeflyer_uncrewed_station_caretaking: the 72 h
telemetry log, the configuration record and the advisory log (the robot API and
the site package format), the draw of a site's telemetry section under the fixed arrangement of the hidden set, and the state of
every sensor at the sortie (read by station_service.py, so the live channels continue the log).

synth_log / advisory: the decoy sizing and the re-synthesis until the advisory record agrees with the cell; every
number comes from the runtime copy of site_spec.json (data/site_spec.json, checked against its sha256 file); the
advisory limit is the unit's registered clean pressure x station.advisory.factor; the slow drift is drawn so the advisory points where the hidden-set rule says, with the expected 10 min mean of
every unit >= n_sigma x sigma_pa away from its limit (decoy: below before its drift starts, above at its end); the
filter loading is a change of r_face in the network, not a ramp added in Pa; the log has the full channel
set of the robot API; a one-sensor offset step is not drawn (the telemetry dictionary does not list one).

Model (all noise-free values are network values, spacekit.suction.FanNetwork, the cell's own construction):
  * a cabin air unit = n parallel inlets of world.inlet.opening_m (K_face world.inlet.k_face, its r_face), design flow
    n x station.ventilation.inlet_flow_cfm x flow_scale, fan_scale 'design_point': a fresh network per r_face, as
    spacekit.station.StationVentilation builds it from station_service.vent_config(site);
  * covers: the blocker (truth.cover_m2, measured in the scene) from telemetry.onset_min; each pouch (telemetry
    .pouch_cover_m2[id], else columns x the area of one sub-point column) from telemetry.pouch_onsets_min[id]; extra
    telemetry.temporary_covers [{inlet, cover_m2, t_on_min, t_off_min}] (the rehearsal sample's crew occlusion);
  * slow drift (telemetry.drift): 'filter_loading' = r_face rising linearly from t0_min to the site's r_face at the
    sortie (both sensors move); 'sensor_drift' = one sensor reading rising linearly from t0_min and on through the
    mission; rise_pa per station.decoy.slow_drift.over_h; telemetry.filter_change {unit, t_min, drop_pa}: the medium
    is changed (r_face lower afterwards; a negative step of both sensors);
  * a log row t_min = k is the mean of the minute [k, k+1) of the level + the sensor's offset (U(+-1/2 accuracy)) +
    its random walk + white noise of 1 accuracy class (sigma station.telemetry.dp_accuracy_pa); the sortie is minute
    station.telemetry.log.hours x 60 = mission second 0 (the robot API's log_axis);
  * advisory: the 10 min trailing mean of the two sensors' average over the unit's advisory_limit_pa raises an entry
    {unit, t_min (last row of the window), mean_pa, limit_pa}; it re-arms when the mean falls below the limit minus
    one accuracy class (hysteresis);
  * the draw keeps the true level of every unit inside the handbook's pressure range of its build
    (datasheet.inlet.dp_range_pa) over the whole log, and every unit's expected 10 min mean >= n_sigma x sigma_pa from
    its limit on the side the hidden-set rule gives the cell (the decoy of a wrong-advisory cell: below before its drift, above
    at the end); the realised advisory log must then say the same (else the draw is repeated);
  * the other channels: fan speed (% of design, the unit's speed controller holds 100 % while running), fan current
    (supply channel: other loads + fan air power / efficiency, on SYNTH['bus_v']), CO2 (one analyser sampling the
    modules in turn every station.telemetry.co2.period_min; rows repeat the last sample; no source in an uncrewed
    station, so the level is flat), cabin temperature and dew point (flat + slow walk).
Constants that site_spec.json does not hold are in SYNTH below (estimates [G]).

API (site = station_site/2 dict; spec = runtime values dict, default the checked runtime copy):
  synth(site, spec=None) -> log dict (columns, data, advisory_log, expected, truth, sortie)
  log_csv(log) -> str; parse_log_csv(text) -> {col: list}; mission_log(log) -> what mission()['log'] returns
  config_record(site), advisory_log(log), mission_members(site) -> {member name: bytes} of the site package
  draw_telemetry(site, rng, spec=None) -> (telemetry, units_patch, info) or raises Reject (the generator redraws)
  check_telemetry(site, spec=None) -> {ok, problems, units: margins}  (the site generator and the selection re-check a written site)
  blocked_unit_margin(site, spec=None) -> margins of the blocked unit (before the noise is drawn)
  unit_state(n, r_face, flow_scale, covers, spec) -> (Pc, Q, P_air) of one network configuration
  sortie_state(site, spec=None, log=None) -> sensor states at the sortie (station_service)
  l3_sample_site(seed, spec=None) -> a rehearsal-module (L3) site for the given telemetry sample
Standard library + numpy + the vendored spacekit (``_lib``).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(HERE, "_lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
from spacekit.suction import CFM, FanNetwork, Inlet, fan_dp  # noqa: E402
from spacekit.station import HATCH_OPENING  # noqa: E402

VERSION = "caretaking.telemetry_synth/1"
RUNTIME_PATH = os.path.join(HERE, "data", "site_spec.json")
RUNTIME_SHA_PATH = os.path.join(HERE, "data", "site_spec.sha256")
MODULE_ORDER = ("l2", "h2", "l1", "l3")
BUILD_LABEL = {"pad": "pad-type", "element": "element-type"}

# Constants the runtime spec does not hold ([G]: estimates of this package, among them the sensor model); they live
# here, once.
SYNTH = {
    "dp_resolution_pa": 0.1,          # logged resolution of the pressure channels
    "offset_frac": 0.5,               # fixed sensor offset U(+-offset_frac x accuracy class)
    "dp_walk_pa": 0.5,                # random walk of each pressure sensor: standard deviation over the 72 h log
    "drift_t0_jitter_min": 300.0,     # a slow drift starts over_h + U(0, 5 h) before the sortie
    "loading_share": 0.6,             # share of filter loading among the slow drifts of 'no wrong advisory' cells
    "pouch_onset_guard_min": 20.0,    # pouches land in [0, log end - 20 min)
    "speed_resolution_pct": 0.1,
    "bus_v": 28.0,                    # supply channel voltage of the fan current [G]
    "fan_efficiency": 0.40,           # fan + motor efficiency (air power / electrical power) [G]
    "other_load_w": (15.0, 40.0),     # other loads on the fan's supply channel [G]
    "current_resolution_a": 0.01,
    "co2_mmhg": (0.6, 1.6),           # flat CO2 level of an uncrewed module [G]
    "co2_resolution_mmhg": 0.01,
    "temp_c": (21.0, 25.0),           # cabin temperature [G]
    "dew_c": (5.0, 10.0),             # dew point [G]
    "temp_walk_c": 0.2,               # slow walk of temperature and dew point over the 72 h log [G]
    "temp_noise_c": 0.05,
    "temp_resolution_c": 0.1,
    "survey_scatter": 0.2,            # old hatch survey: flow x U(1 +- scatter) / hatch area [G]
    "r_grid": 5,                      # r_face nodes of a configuration whose r_face changes with time (degree-4 interpolation)
    "max_tries": 40,                  # telemetry draws per site before Reject
}
LOG_FORMAT = {"return_dp_a": 1, "return_dp_b": 1, "fan_speed": 1, "fan_current": 2, "co2": 2, "temp": 1, "dew_point": 1}


class Reject(Exception):
    """The site cannot carry its cell's telemetry arrangement (the generator draws again); ``cause`` names why."""

    def __init__(self, cause: str, info: dict | None = None):
        super().__init__(cause)
        self.cause = cause
        self.info = info or {}


# ------------------------------------------------------------------------------------------------ the contract
_SPEC = {}


def load_spec(path: str = RUNTIME_PATH, sha_path: str = RUNTIME_SHA_PATH) -> dict:
    """The runtime copy's values, after its sha256 self-check (raises on a mismatch).  Cached per path."""
    if path in _SPEC:
        return _SPEC[path]
    data = open(path, "rb").read()
    want = open(sha_path, "rb").read().decode("ascii").split()[0]
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise RuntimeError(f"site_spec.json sha256 {got} != {want} ({sha_path})")
    doc = json.loads(data.decode("ascii"))
    if doc.get("schema") != "caretaking.site_spec_runtime/1":
        raise RuntimeError(f"{path}: not the runtime copy (schema {doc.get('schema')!r})")
    vals = dict(doc["values"])
    vals["_sha256"] = got
    _SPEC[path] = vals
    return vals


def _spec(spec):
    return load_spec() if spec is None else spec


def n_min(spec=None) -> int:
    s = _spec(spec)
    lg = s["station.telemetry.log"]
    return int(round(lg["hours"] * 60 / lg["period_min"]))


def margin_pa(spec=None) -> float:
    m = _spec(spec)["station.advisory.margin_sigma"]
    return float(m["n_sigma"]) * float(m["sigma_pa"])


def upper(m: str) -> str:
    return m.upper()


def unit_id(m: str) -> str:
    return f"cau_{m}"


def unit_label(m: str) -> str:
    return f"CAU-{m.upper()}"


def modules_of(site) -> list:
    return [m for m in MODULE_ORDER if unit_id(m) in site["cabin_air_units"]]


def inlets_of(m: str, spec=None) -> list:
    """Inlet ids of a module: world.inlets.per_module (L2 / H2 / L1) or rehearsal.l3.inlets (L3)."""
    s = _spec(spec)
    n = int(s["rehearsal.l3"]["inlets"]) if m == "l3" else int(s["world.inlets.per_module"][m.upper()])
    return [f"ri_{m}_{i}" for i in range(1, n + 1)]


def module_of_inlet(inlet: str) -> str:
    return inlet.split("_")[1]


def column_area_m2(spec=None) -> float:
    p = _spec(spec)["world.pouch.placement"]
    return float(p["covered_m2"][0]) / float(p["columns"][0])


def r_range(build: str, n_inlets: int, spec=None) -> tuple:
    r = _spec(spec)["world.inlet.r_face"]
    if build == "element":
        return tuple(r["element"])
    return tuple(r["pad_two_inlet_unit"] if n_inlets == 2 else r["pad_three_inlet_unit"])


def log_channels(modules) -> list:
    """Columns of the log after t_min (the robot API's telemetry_channels with a log rate), per module in site order."""
    out = []
    for m in modules:
        M = m.upper()
        out += [f"cau.{M}.return_dp_a", f"cau.{M}.return_dp_b", f"cau.{M}.fan_speed", f"cau.{M}.fan_current",
                f"co2.{M}", f"cabin.{M}.temp", f"cabin.{M}.dew_point"]
    return out


def _fmt_kind(col: str) -> int:
    parts = col.split(".")
    return LOG_FORMAT["co2" if parts[0] == "co2" else parts[-1]]


# ------------------------------------------------------------------------------------------------ the network
_NET = {}


def unit_state(n: int, r_face: float, flow_scale: float, covers=(), spec=None) -> tuple:
    """(Pc [Pa], Q [m^3/s], fan air power [W]) of a unit with ``covers`` ((inlet index, sealed m^2), ...): a fresh
    FanNetwork per r_face (the cell's construction).  Memoised."""
    s = _spec(spec)
    cov = tuple(sorted((int(i), round(float(a), 10)) for i, a in covers if a > 0))
    key = (int(n), round(float(r_face), 9), round(float(flow_scale), 9), cov, s["_sha256"] if "_sha256" in s else "")
    hit = _NET.get(key)
    if hit is not None:
        return hit
    w, h = s["world.inlet.opening_m"]
    q1 = float(s["station.ventilation.inlet_flow_cfm"]) * float(flow_scale) * CFM
    inls = [Inlet(f"g{i}", (float(i), 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (w / 2, h / 2), flow=q1) for i in range(n)]
    net = FanNetwork(inls, build=dict(K_face=float(s["world.inlet.k_face"]), r_face=float(r_face)), Q_design=n * q1,
                     fan_scale="design_point")
    for i, a in cov:
        net.set_blocked(f"g{i}", a)
    net.solve()
    out = (float(net.Pc), float(net.Q), float(net.Q) * fan_dp(float(net.Q), float(net.omega), float(net.size)))
    if len(_NET) > 200000:
        _NET.clear()
    _NET[key] = out
    return out


def clean_pc(n, r_face, flow_scale, spec=None) -> float:
    return unit_state(n, r_face, flow_scale, (), spec)[0]


def registered(clean_true: float, build: str, spec=None) -> tuple:
    """(clean_reference_pa, advisory_limit_pa) as the configuration record carries them: the clean pressure measured
    at the unit's last service and factor x that pressure, each rounded to 0.1 Pa (the limit from the unrounded
    pressure: 36.14 Pa -> 36.1 / 50.6 Pa)."""
    f = float(_spec(spec)["station.advisory.factor"][build])
    return round(float(clean_true), 1), round(f * float(clean_true), 1)


def r_for_clean(n, pc_target, flow_scale, r_ref, spec=None) -> float:
    """r_face whose unobstructed Pc is pc_target (the clean Pc of a fresh network is linear in r_face)."""
    p1 = clean_pc(n, r_ref, flow_scale, spec)
    p0 = clean_pc(n, r_ref - 10.0, flow_scale, spec)
    b = (p1 - p0) / 10.0
    return float(r_ref + (pc_target - p1) / b)


# ------------------------------------------------------------------------------------------------ covers and events
def _covers(site, spec=None) -> list:
    """[(item id, inlet, area m^2, t_on min, t_off min or None)] of the site (see module doc)."""
    s = _spec(spec)
    tel = site.get("telemetry") or {}
    N = n_min(s)
    out = []
    B = site.get("blocker")
    if B:
        area = (site.get("truth") or {}).get("cover_m2")
        if area is None:
            raise ValueError("site truth.cover_m2 (the blocker's sealed area measured in the scene) is required")
        t_on = tel.get("onset_min")
        if t_on is None:
            raise ValueError("telemetry.onset_min is required for a site with a blocker")
        out.append(("blocker", B["inlet"], float(area), float(t_on), None))
    col = column_area_m2(s)
    pcov = tel.get("pouch_cover_m2") or {}
    pon = tel.get("pouch_onsets_min") or {}
    for P in site.get("small_items") or []:
        pid = P["id"]
        if pid not in pon:
            raise ValueError(f"telemetry.pouch_onsets_min has no {pid}")
        area = float(pcov[pid]) if pid in pcov else float(P["columns"]) * col
        out.append((pid, P["inlet"], area, float(pon[pid]), None))
    for j, c in enumerate(tel.get("temporary_covers") or []):
        out.append((f"temporary_{j + 1}", c["inlet"], float(c["cover_m2"]), float(c["t_on_min"]),
                    None if c.get("t_off_min") is None else float(c["t_off_min"])))
    for it in out:
        if it[3] >= N:
            raise ValueError(f"{it[0]} lands at {it[3]} min, after the log ({N} min)")
    return out


def _r_profile(site, m, spec=None):
    """r_face(t) of a unit on the log axis: (function of an array of minutes, list of kink times, r values span)."""
    s = _spec(spec)
    N = float(n_min(s))
    u = site["cabin_air_units"][unit_id(m)]
    r_end = float(u["r_face"])
    fs = float(u.get("flow_scale", 1.0))
    n = len(inlets_of(m, s))
    tel = site.get("telemetry") or {}
    dr = tel.get("drift")
    fc = tel.get("filter_change")
    load = dr is not None and dr.get("unit") == unit_id(m) and dr.get("kind") == "filter_loading"
    chg = fc is not None and fc.get("unit") == unit_id(m)
    if load and chg:
        raise ValueError("filter loading and a filter change on the same unit are not modelled together")
    if load:
        over_min = float(s["station.decoy.slow_drift"]["over_h"]) * 60.0
        t0 = float(dr["t0_min"])
        rise = float(dr["rise_pa"]) * (N - t0) / over_min
        r0 = r_for_clean(n, clean_pc(n, r_end, fs, s) - rise, fs, r_end, s)

        def f(t):
            t = np.asarray(t, float)
            return np.where(t <= t0, r0, r0 + (r_end - r0) * (t - t0) / (N - t0))
        return f, [t0], (r0, r_end)
    if chg:
        t_fc = float(fc["t_min"])
        r_pre = r_for_clean(n, clean_pc(n, r_end, fs, s) + float(fc["drop_pa"]), fs, r_end, s)

        def f(t):
            t = np.asarray(t, float)
            return np.where(t < t_fc, r_pre, r_end)
        return f, [t_fc], (r_end, r_pre)

    def f(t):
        return np.full(np.shape(np.asarray(t, float)), r_end)
    return f, [], (r_end, r_end)


def _unit_levels(site, m, spec=None) -> dict:
    """Noise-free minute means of one unit over the log: Pc, Q, fan air power; the events with their steps."""
    s = _spec(spec)
    N = n_min(s)
    u = site["cabin_air_units"][unit_id(m)]
    fs = float(u.get("flow_scale", 1.0))
    ins = inlets_of(m, s)
    n = len(ins)
    covers = [c for c in _covers(site, s) if module_of_inlet(c[1]) == m]
    rf, kinks, (r_lo, r_hi) = _r_profile(site, m, s)
    varying = abs(r_hi - r_lo) > 1e-12
    grid = np.linspace(min(r_lo, r_hi), max(r_lo, r_hi), int(SYNTH["r_grid"])) if varying else None
    if varying:     # interpolation through the nodes (Chebyshev-free: 5 equidistant nodes on a near-linear function)
        mid, half = 0.5 * (r_lo + r_hi), 0.5 * abs(r_hi - r_lo)
        xg = (grid - mid) / half
    gcache = {}

    def cfg_at(t):
        return tuple(sorted((ins.index(c[1]), c[2]) for c in covers if c[3] <= t and (c[4] is None or t < c[4])))

    def values(cfg, r):
        r = np.atleast_1d(np.asarray(r, float))
        if not varying:
            v = unit_state(n, float(r[0]), fs, cfg, s)
            return tuple(np.full(r.shape, x) for x in v)
        if cfg not in gcache:
            G = np.array([unit_state(n, float(x), fs, cfg, s) for x in grid])
            gcache[cfg] = [np.polyfit(xg, G[:, j], len(grid) - 1) for j in range(3)]
        x = (r - mid) / half
        return tuple(np.polyval(c, x) for c in gcache[cfg])

    cuts = sorted(set([0.0, float(N)] + [c[3] for c in covers if 0.0 < c[3] < N] +
                      [c[4] for c in covers if c[4] is not None and 0.0 < c[4] < N] + [k for k in kinks if 0.0 < k < N]))
    lev = np.zeros((3, N))
    for a, b in zip(cuts[:-1], cuts[1:]):
        cfg = cfg_at(0.5 * (a + b))
        k0, k1 = int(math.floor(a)), int(math.ceil(b))           # minutes touched by [a, b)
        ks = np.arange(k0, k1)
        lo = np.maximum(ks, a)
        hi = np.minimum(ks + 1.0, b)
        wgt = hi - lo
        keep = wgt > 0
        ks, lo, hi, wgt = ks[keep], lo[keep], hi[keep], wgt[keep]
        vals = values(cfg, rf(0.5 * (lo + hi)))
        for j in range(3):
            np.add.at(lev[j], ks, wgt * vals[j])
    events = []
    for c in covers:
        for t, sign in ((c[3], 1), (c[4], -1)):
            if t is None or not (0.0 < t < N):
                continue
            r = float(rf(t))
            before = values(cfg_at(t - 1e-9), r)[0][0]
            after = values(cfg_at(t + 1e-9), r)[0][0]
            events.append(dict(t_min=float(t), item=c[0], inlet=c[1], kind="on" if sign > 0 else "off", step_pa=float(after - before),
                               area_m2=float(c[2]), level_before_pa=float(before), clean_pa=float(values((), r)[0][0])))
    fc = (site.get("telemetry") or {}).get("filter_change")
    if fc is not None and fc.get("unit") == unit_id(m):
        t = float(fc["t_min"])
        cfg = cfg_at(t)
        before = values(cfg, rf(t - 1e-9))[0][0]
        after = values(cfg, rf(t + 1e-9))[0][0]
        events.append(dict(t_min=t, item="filter_change", inlet=None, kind="filter_change", step_pa=float(after - before)))
    events.sort(key=lambda e: (e["t_min"], e["item"]))
    end_cfg = cfg_at(float(N))
    pc_end, q_end, pa_end = (float(x[0]) for x in values(end_cfg, float(rf(float(N)))))
    return dict(level=lev[0], q=lev[1], p_air=lev[2], events=events, r_span=(float(r_lo), float(r_hi)),
                end=dict(pc=pc_end, q=q_end, p_air=pa_end, covers={ins[i]: a for i, a in end_cfg}),
                clean_end=clean_pc(n, float(u["r_face"]), fs, s), n_inlets=n, flow_scale=fs)


# ------------------------------------------------------------------------------------------------ sensors
def _rng(seed: int, *key) -> np.random.Generator:
    return np.random.default_rng([int(seed)] + [int(k) for k in key])


def _unit_index(m: str) -> int:
    return MODULE_ORDER.index(m)


def _round(x, nd):
    return np.round(np.asarray(x, float), nd) + 0.0


def _dp_streams(site, m, spec=None) -> dict:
    """Fixed offsets and random walks of the two pressure sensors (noise_seed stream 1), white noise (2, 3)."""
    s = _spec(spec)
    N = n_min(s)
    acc = float(s["station.telemetry.dp_accuracy_pa"])
    seed = int(site["telemetry"]["noise_seed"])
    i = _unit_index(m)
    g = _rng(seed, 1000 + i, 1)
    off = g.uniform(-SYNTH["offset_frac"] * acc, SYNTH["offset_frac"] * acc, 2)
    walk = np.cumsum(g.normal(0.0, SYNTH["dp_walk_pa"] / math.sqrt(N), (2, N)), axis=1)
    white = np.stack([_rng(seed, 1000 + i, 2).normal(0.0, acc, N), _rng(seed, 1000 + i, 3).normal(0.0, acc, N)])
    return dict(off=off, walk=walk, white=white)


def _drift_terms(site, m, spec=None):
    """Per-sensor additive drift on the log rows (sensor_drift only) and its slope [Pa/min] for the live channels."""
    s = _spec(spec)
    N = n_min(s)
    dr = (site.get("telemetry") or {}).get("drift")
    z = np.zeros((2, N))
    if not dr or dr.get("unit") != unit_id(m) or dr.get("kind") != "sensor_drift":
        return z, None
    slope = float(dr["rise_pa"]) / (float(s["station.decoy.slow_drift"]["over_h"]) * 60.0)
    t0 = float(dr["t0_min"])
    mid = np.arange(N) + 0.5
    ramp = np.clip(mid - t0, 0.0, None) * slope
    sens = dr.get("sensor", "a")
    for j, name in enumerate("ab"):
        if sens == name or sens == "both":
            z[j] = ramp
    return z, dict(sensor=sens, slope_pa_per_min=slope, t0_min=t0)


def _other_streams(site, m, modules, spec=None) -> dict:
    s = _spec(spec)
    N = n_min(s)
    seed = int(site["telemetry"]["noise_seed"])
    i = _unit_index(m)
    fan_acc = float(s["station.telemetry.fan_accuracy_pct"])
    g = _rng(seed, 1000 + i, 4)
    sp_off = g.uniform(-SYNTH["offset_frac"], SYNTH["offset_frac"]) * fan_acc / 100.0
    cur_off = g.uniform(-SYNTH["offset_frac"], SYNTH["offset_frac"]) * fan_acc / 100.0
    other_w = g.uniform(*SYNTH["other_load_w"])
    sp_white = _rng(seed, 1000 + i, 5).normal(0.0, fan_acc, N)
    cur_white = _rng(seed, 1000 + i, 6).normal(0.0, fan_acc / 100.0, N)
    co2 = s["station.telemetry.co2"]
    gc = _rng(seed, 1000 + i, 7)
    c_level = gc.uniform(*SYNTH["co2_mmhg"])
    period = int(co2["period_min"])
    phase = int(round(modules.index(m) * period / max(1, len(modules)))) % period
    samples = c_level * (1.0 + gc.normal(0.0, float(co2["accuracy_pct"]) / 100.0, N // period + 2))
    gt = _rng(seed, 1000 + i, 8)
    t_base = gt.uniform(*SYNTH["temp_c"])
    d_base = gt.uniform(*SYNTH["dew_c"])
    t_walk = np.cumsum(gt.normal(0.0, SYNTH["temp_walk_c"] / math.sqrt(N), N))
    d_walk = np.cumsum(gt.normal(0.0, SYNTH["temp_walk_c"] / math.sqrt(N), N))
    t_white = gt.normal(0.0, SYNTH["temp_noise_c"], N)
    d_white = gt.normal(0.0, SYNTH["temp_noise_c"], N)
    return dict(sp_off=sp_off, cur_off=cur_off, other_w=other_w, sp_white=sp_white, cur_white=cur_white, c_level=c_level,
                period=period, phase=phase, c_samples=samples, t_base=t_base, d_base=d_base, t_walk=t_walk, d_walk=d_walk,
                t_white=t_white, d_white=d_white)


def fan_current_a(p_air_w, other_w, off_frac=0.0):
    """Supply channel current [A]: other loads + fan air power / efficiency, on the bus voltage (noise-free)."""
    return (np.asarray(other_w, float) + np.asarray(p_air_w, float) / SYNTH["fan_efficiency"]) / SYNTH["bus_v"] * (1.0 + off_frac)


def _co2_series(o, N):
    """Held analyser samples: rows repeat the latest sample; sample j at minute phase + j x period (j = 0 is the
    sample before the log's first one)."""
    k = np.arange(N)
    j = np.floor((k - o["phase"]) / o["period"]).astype(int) + 1
    j = np.clip(j, 0, len(o["c_samples"]) - 1)
    return o["c_samples"][j]


def _rolling_mean(x, w):
    c = np.convolve(np.asarray(x, float), np.ones(w) / w, mode="valid")
    out = np.full(len(x), np.nan)
    out[w - 1:] = c
    return out


# ------------------------------------------------------------------------------------------------ advisory monitor
def advisory_entries(dp_a, dp_b, limit_pa, label, window, hysteresis, t0=0, latched=False):
    """Advisories of one unit from its two pressure channels (see module doc): (entries, latched at the end)."""
    s = 0.5 * (np.asarray(dp_a, float) + np.asarray(dp_b, float))
    m = _rolling_mean(s, window)
    out = []
    for k in range(window - 1, len(s)):
        v = float(m[k])
        if not latched and v > limit_pa:
            latched = True
            out.append(dict(unit=label, t_min=int(t0 + k), mean_pa=round(v, 2), limit_pa=float(limit_pa)))
        elif latched and v < limit_pa - hysteresis:
            latched = False
    return out, latched


# ------------------------------------------------------------------------------------------------ the log
def synth(site, spec=None) -> dict:
    """The site's 72 h log and everything derived from it (see module doc).  Deterministic in the site."""
    s = _spec(spec)
    N = n_min(s)
    mods = modules_of(site)
    acc = float(s["station.telemetry.dp_accuracy_pa"])
    win = int(s["station.advisory.mean_window_min"])
    res = SYNTH["dp_resolution_pa"]
    nd = int(round(-math.log10(res)))
    data = {"t_min": np.arange(N)}
    expected, truth, sortie, adv = {}, {}, {}, []
    for m in mods:
        M = m.upper()
        u = site["cabin_air_units"][unit_id(m)]
        L = _unit_levels(site, m, s)
        dp = _dp_streams(site, m, s)
        drift, drift_live = _drift_terms(site, m, s)
        exp = np.stack([L["level"] + dp["off"][j] + dp["walk"][j] + drift[j] for j in range(2)])
        rows = _round(exp + dp["white"], nd)
        o = _other_streams(site, m, mods, s)
        sp = _round(100.0 * (1.0 + o["sp_off"]) + o["sp_white"], 1)
        cur0 = fan_current_a(L["p_air"], o["other_w"], o["cur_off"])
        cur = _round(cur0 * (1.0 + o["cur_white"]), 2)
        co2 = _round(_co2_series(o, N), 2)
        tc = _round(o["t_base"] + o["t_walk"] + o["t_white"], 1)
        dc = _round(o["d_base"] + o["d_walk"] + o["d_white"], 1)
        data.update({f"cau.{M}.return_dp_a": rows[0], f"cau.{M}.return_dp_b": rows[1], f"cau.{M}.fan_speed": sp,
                     f"cau.{M}.fan_current": cur, f"co2.{M}": co2, f"cabin.{M}.temp": tc, f"cabin.{M}.dew_point": dc})
        limit = float(u["advisory_limit_pa"])
        ent, latched = advisory_entries(rows[0], rows[1], limit, unit_label(m), win, acc)
        adv += ent
        E = _rolling_mean(0.5 * (exp[0] + exp[1]), win)
        expected[m] = dict(E=E, limit=limit, exp=exp)
        truth[m] = dict(level=L["level"], q=L["q"], p_air=L["p_air"], events=L["events"], r_span=L["r_span"], end=L["end"],
                        clean_end=L["clean_end"], n_inlets=L["n_inlets"])
        k_last = N - 1
        last_sample = int(k_last - ((k_last - o["phase"]) % o["period"]))
        sortie[m] = dict(dp_bias=[float(dp["off"][j] + dp["walk"][j, -1]) for j in range(2)], drift=drift_live,
                         speed_off=float(o["sp_off"]), current_off=float(o["cur_off"]), other_w=float(o["other_w"]),
                         co2=dict(level=float(o["c_level"]), period=o["period"], phase=o["phase"], held=float(co2[-1]),
                                  last_sample_min=last_sample),
                         temp=float(o["t_base"] + o["t_walk"][-1]), dew=float(o["d_base"] + o["d_walk"][-1]),
                         tail=[float(x) for x in (0.5 * (rows[0][-(win - 1):] + rows[1][-(win - 1):]))], latched=bool(latched),
                         end=L["end"], n_inlets=L["n_inlets"], flow_scale=float(u.get("flow_scale", 1.0)), r_face=float(u["r_face"]),
                         limit=limit)
    adv.sort(key=lambda e: (e["t_min"], e["unit"]))
    cols = ["t_min"] + log_channels(mods)
    return dict(schema=VERSION, site_id=site.get("site_id"), n_min=N, modules=mods, columns=cols, data=data,
                advisory_log=adv, expected=expected, truth=truth, sortie=dict(units=sortie, imd=_imd(site, s)),
                spec_sha256=s.get("_sha256"))


def _imd(site, spec=None) -> dict:
    """Static hatch survey channels imd.<A>-<B>.last_survey from the site's inter-module links."""
    s = _spec(spec)
    seed = int(site["telemetry"]["noise_seed"])
    out = {}
    age = s["station.telemetry.imd_survey_age_days"]
    for j, (lk, v) in enumerate(sorted((site.get("inter_module") or {}).items())):
        a, b = lk.split("_")[1:3]
        pair = sorted((a, b), key=lambda x: MODULE_ORDER.index(x) if x in MODULE_ORDER else 99)
        g = _rng(seed, 2000 + j, 1)
        speed = float(v["flow_cfm"]) * CFM / HATCH_OPENING ** 2 * g.uniform(1 - SYNTH["survey_scatter"], 1 + SYNTH["survey_scatter"])
        out[f"imd.{pair[0].upper()}-{pair[1].upper()}.last_survey"] = dict(speed_m_s=round(speed, 3),
                                                                           age_days=int(g.integers(int(age[0]), int(age[1]) + 1)))
    return out


def log_csv(log: dict) -> str:
    """The log as the site package's mission/log.csv: header row, one row per minute, '\\n' line ends, fixed decimals
    per channel kind (LOG_FORMAT)."""
    cols = log["columns"]
    fmts = [None] + [_fmt_kind(c) for c in cols[1:]]
    data = log["data"]
    colstr = [[str(int(v)) for v in np.asarray(data["t_min"]).tolist()]]
    for c, nd in zip(cols[1:], fmts[1:]):
        f = f"%.{nd}f"
        # the channels are rounded to their decimals and normalised (+ 0.0), so no '-0.0' can appear
        colstr.append([f % v for v in (np.round(np.asarray(data[c], float), nd) + 0.0).tolist()])
    lines = [",".join(cols)] + [",".join(r) for r in zip(*colstr)]
    return "\n".join(lines) + "\n"


def parse_log_csv(text: str) -> dict:
    """{column: list} of a log.csv (t_min as int, the rest as float)."""
    lines = text.rstrip("\n").split("\n")
    cols = lines[0].split(",")
    vals = [ln.split(",") for ln in lines[1:]]
    out = {c: [] for c in cols}
    for r in vals:
        if len(r) != len(cols):
            raise ValueError("ragged log.csv row")
        out[cols[0]].append(int(r[0]))
        for c, x in zip(cols[1:], r[1:]):
            out[c].append(float(x))
    return out


def mission_log(log_or_text) -> dict:
    """mission()['log']: {t_min: [...], <channel>: [...]} -- parsed from the CSV text, so both carry the same numbers."""
    text = log_or_text if isinstance(log_or_text, str) else log_csv(log_or_text)
    return parse_log_csv(text)


def config_record(site) -> dict:
    """mission/config_record.json (the robot API's mission.config_record)."""
    units = {}
    for m in modules_of(site):
        u = site["cabin_air_units"][unit_id(m)]
        units[unit_label(m)] = dict(build=BUILD_LABEL[u["build"]], clean_reference_pa=float(u["clean_reference_pa"]),
                                    advisory_limit_pa=float(u["advisory_limit_pa"]), fan_stop=u["fan_stop"],
                                    inhibit_reason=u.get("inhibit_reason"))
    return dict(units=units, lights=commanded_lighting(site))


def commanded_lighting(site) -> dict:
    """Commanded lighting per module at t = 0 ({M: general | night}): a failed light circuit ('off') reports the
    station's night mode (the lights channel reports the commanded state only)."""
    return {m.upper(): ("night" if v == "off" else v) for m, v in sorted((site.get("lighting") or {}).items(),
                                                                          key=lambda kv: MODULE_ORDER.index(kv[0]))}


def advisory_log(log) -> list:
    return [dict(e) for e in log["advisory_log"]]


def _dumps(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def mission_members(site, spec=None, log=None) -> dict:
    """{name: bytes} of the three station members of the site package (the site package format)."""
    lg = synth(site, spec) if log is None else log
    return {"mission/config_record.json": _dumps(config_record(site)), "mission/log.csv": log_csv(lg).encode("ascii"),
            "mission/advisory_log.json": _dumps(advisory_log(lg))}


def verify_members(site, members: dict, spec=None) -> list:
    """Problems when a package's station members differ from a fresh synthesis of its site.json (cell start check)."""
    fresh = mission_members(site, spec)
    return [f"{k} differs from the synthesis of site.json" for k in sorted(fresh) if members.get(k) != fresh[k]]


# ------------------------------------------------------------------------------------------------ arrangement checks
def _windows(N, win, onset):
    k = np.arange(N)
    if onset is None:
        return k >= win - 1, np.zeros(N, bool)
    pre = (k >= win - 1) & (k + 1 <= onset)
    post = (k - (win - 1)) >= math.ceil(onset)
    return pre, post


def unit_margins(log, site, spec=None) -> dict:
    """Per unit the expected 10 min mean (no white noise) against the limit, in Pa and in sigma_pa: max over the log,
    max before / min after the onset (blocked unit), max before the drift start (decoy)."""
    s = _spec(spec)
    sig = float(s["station.advisory.margin_sigma"]["sigma_pa"])
    win = int(s["station.advisory.mean_window_min"])
    N = log["n_min"]
    tel = site.get("telemetry") or {}
    onset = tel.get("onset_min")
    blk = module_of_inlet(site["blocker"]["inlet"]) if site.get("blocker") else None
    dr = tel.get("drift")
    out = {}
    for m, ex in log["expected"].items():
        E, lim = ex["E"], ex["limit"]
        valid = np.isfinite(E)
        d = dict(limit_pa=lim, max_pa=float(np.nanmax(E)), max_margin_sigma=float((lim - np.nanmax(E)) / sig))
        if m == blk and onset is not None:
            pre, post = _windows(N, win, onset)
            d["pre_max_pa"] = float(np.max(E[pre & valid])) if np.any(pre & valid) else None
            d["post_min_pa"] = float(np.min(E[post & valid])) if np.any(post & valid) else None
            d["pre_margin_sigma"] = None if d["pre_max_pa"] is None else float((lim - d["pre_max_pa"]) / sig)
            d["post_margin_sigma"] = None if d["post_min_pa"] is None else float((d["post_min_pa"] - lim) / sig)
        if dr and dr.get("unit") == unit_id(m):
            k = np.arange(N)
            before = valid & (k + 1 <= float(dr["t0_min"]))
            d["before_drift_max_pa"] = float(np.max(E[before])) if np.any(before) else None
            d["before_drift_margin_sigma"] = None if d["before_drift_max_pa"] is None else float((lim - d["before_drift_max_pa"]) / sig)
            d["end_margin_sigma"] = float((E[-1] - lim) / sig)
        out[m] = d
    return out


def _cell_row(site, spec=None):
    c = site.get("cell")
    if not c:
        return None
    return _spec(spec)[f"layout.cell.{c['module'].upper()}x{c['tier']}"]


def _side(d, n_sig) -> str:
    """'below' / 'over' / 'band' of the blocked unit from its margins."""
    if d["max_margin_sigma"] >= n_sig:
        return "below"
    if d.get("pre_margin_sigma") is not None and d.get("post_margin_sigma") is not None and \
            d["pre_margin_sigma"] >= n_sig and d["post_margin_sigma"] >= n_sig:
        return "over"
    return "band"


def check_telemetry(site, spec=None, log=None) -> dict:
    """The arrangement of the hidden-set rule on a written site: expected margins (every unit >= n_sigma from its limit, the
    decoy below before its drift and above at the end, the blocked unit on the cell's side) and the realised advisory
    log (who is in it).  {ok, problems, side, units, advisory_units, points_elsewhere, blocked_unit_over_limit}."""
    s = _spec(spec)
    n_sig = float(s["station.advisory.margin_sigma"]["n_sigma"])
    lg = synth(site, s) if log is None else log
    U = unit_margins(lg, site, s)
    row = _cell_row(site, s)
    tel = site.get("telemetry") or {}
    blk = module_of_inlet(site["blocker"]["inlet"]) if site.get("blocker") else None
    dr = tel.get("drift")
    decoy = dr["unit"][4:] if dr else None
    wrong = (row or {}).get("advisory") == "wrong_unit"
    probs = []
    side = None
    if wrong and decoy is None:
        probs.append("a wrong-advisory cell needs a decoy")
    if blk is not None:
        side = _side(U[blk], n_sig)
        want = row.get("blocked_unit_limit", "below") if row is not None else "either"
        if side == "band":
            probs.append("blocked unit inside the +-n sigma band of its limit")
        elif want == "below" and side != "below":
            probs.append("blocked unit over its limit in a 'below' cell")
    for m, d in U.items():
        if m == blk:
            continue
        if wrong and m == decoy:
            if d.get("before_drift_margin_sigma") is None or d["before_drift_margin_sigma"] < n_sig:
                probs.append(f"decoy {m} not below its limit before its drift starts")
            if d["end_margin_sigma"] < n_sig:
                probs.append(f"decoy {m} not over its limit at the end")
        elif d["max_margin_sigma"] < n_sig:
            probs.append(f"unit {m} within n sigma of (or over) its limit")
    if wrong and (dr is None or dr.get("kind") != "filter_loading"):
        probs.append("a wrong-advisory cell needs a filter-loading decoy")
    rng_dp = s["datasheet.inlet.dp_range_pa"]
    for m in lg["modules"]:                      # the handbook's pressure range of the build holds for the true level
        b = site["cabin_air_units"][unit_id(m)]["build"]
        lv = lg["truth"][m]["level"]
        if float(np.min(lv)) < float(rng_dp[b][0]) or float(np.max(lv)) > float(rng_dp[b][1]):
            probs.append(f"unit {m} level outside the datasheet range of its build")
    units_adv = [e["unit"] for e in lg["advisory_log"]]
    blk_lab = unit_label(blk) if blk else None
    elsewhere = any(u != blk_lab for u in units_adv)
    on_blk = [e for e in lg["advisory_log"] if e["unit"] == blk_lab]
    blk_over = bool(on_blk)
    if wrong and decoy is not None:
        if not units_adv or units_adv[0] != unit_label(decoy):
            probs.append("realised advisory log: the first entry is not the decoy")
        if any(u not in (unit_label(decoy),) for u in units_adv):
            probs.append("realised advisory log: an entry on a unit other than the decoy")
    elif row is not None:
        if elsewhere:
            probs.append("realised advisory log: an entry on a unit that is not blocked")
    if blk is not None:
        if side == "over" and not (on_blk and on_blk[0]["t_min"] >= tel.get("onset_min", 0)):
            probs.append("realised advisory log: the over-limit blocked unit has no advisory after the onset")
        if side == "below" and blk_over:
            probs.append("realised advisory log: an advisory on the blocked unit of a 'below' outcome")
    exp = tel.get("advisory_expected")
    if exp is not None and row is not None:
        if bool(exp.get("points_elsewhere")) != elsewhere:
            probs.append("telemetry.advisory_expected.points_elsewhere disagrees with the log")
        if bool(exp.get("blocked_unit_over_limit")) != blk_over:
            probs.append("telemetry.advisory_expected.blocked_unit_over_limit disagrees with the log")
    return dict(ok=not probs, problems=probs, side=side, units=U, advisory_units=units_adv, points_elsewhere=elsewhere,
                blocked_unit_over_limit=blk_over)


def blocked_unit_margin(site, spec=None) -> dict:
    """Margins of the blocked unit before any noise is drawn: the end level (blocker + every pouch of the unit) and
    the level just before the onset (pouches that landed earlier excluded: all pouches of the unit, the worst case),
    against round(factor x clean); in Pa and sigma.  The site generator calls it before drawing the telemetry."""
    s = _spec(spec)
    sig = float(s["station.advisory.margin_sigma"]["sigma_pa"])
    B = site["blocker"]
    m = module_of_inlet(B["inlet"])
    u = site["cabin_air_units"][unit_id(m)]
    ins = inlets_of(m, s)
    n = len(ins)
    fs = float(u.get("flow_scale", 1.0))
    clean, lim = registered(clean_pc(n, u["r_face"], fs, s), u["build"], s)
    col = column_area_m2(s)
    pcov = ((site.get("telemetry") or {}).get("pouch_cover_m2") or {})
    pouch = [(ins.index(P["inlet"]), float(pcov.get(P["id"], P["columns"] * col))) for P in site.get("small_items") or []
             if module_of_inlet(P["inlet"]) == m]
    area = float(site["truth"]["cover_m2"])
    end = unit_state(n, u["r_face"], fs, [(ins.index(B["inlet"]), area)] + pouch, s)[0]
    pre_worst = unit_state(n, u["r_face"], fs, pouch, s)[0]
    return dict(clean_pa=clean, limit_pa=lim, end_pa=end, pre_pa=pre_worst, end_margin_sigma=(lim - end) / sig,
                pre_margin_sigma=(lim - pre_worst) / sig, ratio=end / clean_pc(n, u["r_face"], fs, s))


# ------------------------------------------------------------------------------------------------ the draw
def _r2(x, nd=3):
    return float(round(float(x), nd))


def draw_telemetry(site, rng, spec=None, max_tries=None) -> tuple:
    """Draw the telemetry section of a site and its units' clean references and advisory limits (see module doc and
    the hidden-set rule).  ``site`` needs cabin_air_units (build, r_face, fan_stop, inhibit_reason, flow_scale), blocker,
    small_items, truth.cover_m2 and cell (optionally telemetry.pouch_cover_m2, the pouches' sealed areas measured in
    the scene: carried into the drawn section); it is not modified.  Returns (telemetry, units_patch, info); raises Reject
    when no draw in max_tries meets the arrangement (info counts the causes)."""
    s = _spec(spec)
    N = n_min(s)
    row = _cell_row(site, s)
    fac = s["station.advisory.factor"]
    sd = s["station.decoy.slow_drift"]
    over_min = float(sd["over_h"]) * 60.0
    oh = s["world.onset_h_before"]
    M = margin_pa(s)
    dp_lo = {b: float(v[0]) for b, v in s["datasheet.inlet.dp_range_pa"].items()}
    dp_hi = {b: float(v[1]) for b, v in s["datasheet.inlet.dp_range_pa"].items()}
    mods = modules_of(site)
    blk = module_of_inlet(site["blocker"]["inlet"]) if site.get("blocker") else None
    wrong = bool(row) and row.get("advisory") == "wrong_unit"
    pads = [m for m in mods if site["cabin_air_units"][unit_id(m)]["build"] == "pad" and m != blk]
    if wrong and not pads:
        raise Reject("no_pad_unit_for_decoy")
    n_sig = float(s["station.advisory.margin_sigma"]["n_sigma"])
    if blk is not None:     # the network level alone decides whether the site can be on its cell's side (no noise yet)
        bm = blocked_unit_margin(site, s)
        want = row.get("blocked_unit_limit", "below") if row is not None else "either"
        below = bm["end_margin_sigma"] >= n_sig
        over = bm["end_margin_sigma"] <= -n_sig and bm["pre_margin_sigma"] >= n_sig
        if not (below or over):
            raise Reject("blocked_band", dict(margin=bm))
        if want == "below" and not below:
            raise Reject("blocked_over", dict(margin=bm))
    rng_dp = s["datasheet.inlet.dp_range_pa"]
    for m in mods:                               # units without a slow drift keep their level: check it before any draw
        u = site["cabin_air_units"][unit_id(m)]
        ins = inlets_of(m, s)
        col = column_area_m2(s)
        pcov = ((site.get("telemetry") or {}).get("pouch_cover_m2") or {})
        cov = [(ins.index(P["inlet"]), float(pcov.get(P["id"], P["columns"] * col))) for P in site.get("small_items") or []
               if module_of_inlet(P["inlet"]) == m]
        if site.get("blocker") and module_of_inlet(site["blocker"]["inlet"]) == m:
            cov.append((ins.index(site["blocker"]["inlet"]), float(site["truth"]["cover_m2"])))
        lo_l = clean_pc(len(ins), u["r_face"], u.get("flow_scale", 1.0), s)
        hi_l = unit_state(len(ins), u["r_face"], u.get("flow_scale", 1.0), cov, s)[0]
        if lo_l < float(rng_dp[u["build"]][0]) or hi_l > float(rng_dp[u["build"]][1]):
            raise Reject("level_outside_datasheet_range", dict(unit=m, clean_pa=lo_l, with_items_pa=hi_l))
    pc_in = ((site.get("telemetry") or {}).get("pouch_cover_m2") or None)
    causes = {}
    tries = int(max_tries or SYNTH["max_tries"])
    base = copy.deepcopy(site)
    for attempt in range(1, tries + 1):
        tel = dict(log_minutes=N, synth_version=VERSION, filter_change=None, temporary_covers=[])
        if pc_in:                                # the pouches' sealed areas measured in the scene, kept
            tel["pouch_cover_m2"] = dict(pc_in)
        tel["noise_seed"] = int(rng.integers(0, 2 ** 31 - 1))
        tel["onset_min"] = _r2(N - rng.uniform(float(oh[0]) * 60.0, float(oh[1]) * 60.0), 2) if blk else None
        tel["pouch_onsets_min"] = {P["id"]: _r2(rng.uniform(0.0, N - SYNTH["pouch_onset_guard_min"]), 2)
                                   for P in base.get("small_items") or []}
        tel["drift"] = None
        if row is not None and pads:
            dm = pads[int(rng.integers(len(pads)))]
            kind = "filter_loading" if (wrong or rng.uniform() < SYNTH["loading_share"]) else "sensor_drift"
            sensor = "both" if kind == "filter_loading" else "ab"[int(rng.integers(2))]
            rise = _r2(rng.uniform(*sd["rise_pa"]))
            t0 = _r2(N - over_min - rng.uniform(0.0, SYNTH["drift_t0_jitter_min"]), 2)
            tel["drift"] = dict(unit=unit_id(dm), kind=kind, sensor=sensor, rise_pa=rise, t0_min=t0)
        if tel["drift"] is not None and tel["drift"]["kind"] == "filter_loading":
            pre = _decoy_precheck(base, tel["drift"], wrong, s)
            if pre is not None:
                causes[pre] = causes.get(pre, 0) + 1
                continue
        trial = copy.deepcopy(base)
        trial["telemetry"] = dict(tel, advisory_expected=None)          # filled from the checked log below
        patch = {}
        for m in mods:
            u = trial["cabin_air_units"][unit_id(m)]
            n = len(inlets_of(m, s))
            c, lim = registered(clean_pc(n, u["r_face"], u.get("flow_scale", 1.0), s), u["build"], s)
            patch[unit_id(m)] = dict(clean_reference_pa=c, advisory_limit_pa=lim)
        dr = tel["drift"]
        loading = dr is not None and dr["kind"] == "filter_loading"
        if loading:   # the decoy's limit is placed after its expected readings are known (a dummy limit first)
            patch[dr["unit"]] = dict(clean_reference_pa=1.0, advisory_limit_pa=1.0)
        for k, v in patch.items():
            trial["cabin_air_units"][k].update(v)
        try:
            lg = synth(trial, s)
        except ValueError as exc:
            causes["synth_error:" + str(exc)[:60]] = causes.get("synth_error:" + str(exc)[:60], 0) + 1
            continue
        if loading:
            dm = dr["unit"][4:]
            u = trial["cabin_air_units"][dr["unit"]]
            n = len(inlets_of(dm, s))
            fs = float(u.get("flow_scale", 1.0))
            f = float(fac[u["build"]])
            E = lg["expected"][dm]["E"]
            k = np.arange(N)
            before = np.isfinite(E) & (k + 1 <= float(dr["t0_min"]))
            r0, _ = lg["truth"][dm]["r_span"]
            pc_start = clean_pc(n, r0, fs, s)
            if pc_start < dp_lo["pad"] or float(np.nanmax(lg["truth"][dm]["level"])) > dp_hi["pad"]:
                causes["decoy_outside_datasheet_range"] = causes.get("decoy_outside_datasheet_range", 0) + 1
                continue
            lo = f * dp_lo["pad"]
            hi = f * pc_start                                        # serviced before the drift: clean <= start level
            if wrong:
                lo = max(lo, float(np.max(E[before])) + M)
                hi = min(hi, float(E[-1]) - M)
            else:
                lo = max(lo, float(np.nanmax(E)) + M)
            if not hi > lo + 0.4:                                     # room for the 0.15 Pa guards below
                causes["decoy_infeasible"] = causes.get("decoy_infeasible", 0) + 1
                continue
            lim_raw = float(rng.uniform(lo + 0.15, hi - 0.15))       # rounding to 0.1 Pa moves it <= 0.05 Pa
            clean, lim = registered(lim_raw / f, u["build"], s)
            patch[dr["unit"]] = dict(clean_reference_pa=clean, advisory_limit_pa=lim)
            trial["cabin_air_units"][dr["unit"]].update(patch[dr["unit"]])
            lg = synth(trial, s)
        chk = check_telemetry(trial, s, lg)
        if not chk["ok"]:
            for p in chk["problems"]:
                key = {"blocked unit inside the +-n sigma band of its limit": "blocked_band_with_sensor_offsets",
                       "blocked unit over its limit in a 'below' cell": "blocked_over_with_sensor_offsets"}.get(p, p)
                causes[key] = causes.get(key, 0) + 1
            continue
        tel["advisory_expected"] = dict(points_elsewhere=chk["points_elsewhere"], blocked_unit_over_limit=chk["blocked_unit_over_limit"])
        info = dict(tries=attempt, causes=causes, side=chk["side"], units=chk["units"], advisory_log=lg["advisory_log"],
                    csv_sha256=hashlib.sha256(log_csv(lg).encode("ascii")).hexdigest())
        return tel, patch, info
    raise Reject("telemetry_draw_failed", dict(tries=tries, causes=causes))


def _decoy_precheck(site, dr, wrong, spec=None):
    """Cheap feasibility of a filter-loading decoy before the log is synthesized (network values only, with 1.5 Pa of
    slack for the sensor offsets and walks): None when it may work, else the cause."""
    s = _spec(spec)
    N = float(n_min(s))
    m = dr["unit"][4:]
    u = site["cabin_air_units"][dr["unit"]]
    ins = inlets_of(m, s)
    n = len(ins)
    fs = float(u.get("flow_scale", 1.0))
    f = float(s["station.advisory.factor"][u["build"]])
    lo_dp, hi_dp = (float(x) for x in s["datasheet.inlet.dp_range_pa"]["pad"])
    M = margin_pa(s)
    rise = float(dr["rise_pa"]) * (N - float(dr["t0_min"])) / (float(s["station.decoy.slow_drift"]["over_h"]) * 60.0)
    pc_end = clean_pc(n, u["r_face"], fs, s)
    pc_start = pc_end - rise
    col = column_area_m2(s)
    pcov = ((site.get("telemetry") or {}).get("pouch_cover_m2") or {})
    cov = [(ins.index(P["inlet"]), float(pcov.get(P["id"], P["columns"] * col))) for P in site.get("small_items") or []
           if module_of_inlet(P["inlet"]) == m]
    with_items = unit_state(n, u["r_face"], fs, cov, s)[0]
    if pc_start < lo_dp or with_items > hi_dp:
        return "decoy_outside_datasheet_range"
    hi = f * pc_start
    lo = f * lo_dp
    if wrong:
        hi = min(hi, with_items - M + 1.5)
        lo = max(lo, pc_start + M - 1.5)
    else:
        lo = max(lo, with_items + M - 1.5)
    return None if hi > lo + 0.4 else "decoy_infeasible"


# ------------------------------------------------------------------------------------------------ live continuation
def sortie_state(site, spec=None, log=None) -> dict:
    """What station_service needs to continue every channel at mission second 0 (see synth()['sortie'])."""
    lg = synth(site, spec) if log is None else log
    return dict(lg["sortie"], n_min=lg["n_min"], advisory_log=[dict(e) for e in lg["advisory_log"]], modules=lg["modules"])


# ------------------------------------------------------------------------------------------------ the rehearsal sample
def l3_sample_site(seed: int, spec=None, build: str = "pad") -> dict:
    """A rehearsal-module site for the given 72 h telemetry sample (the given ~/given/telemetry_sample/l3_72h.csv is
    written from synth()): one unit CAU-L3 (2 inlets), no blocker, no pouches; one known crew occlusion of an inlet (a
    temporary cover of the blocker's sealed area range, 1-4 h) and one filter change (a negative step, the
    pad-type medium only), at least 6 h apart.  ``seed`` must not be the seed of any other site."""
    s = _spec(spec)
    N = n_min(s)
    g = _rng(seed, 3000)
    n = len(inlets_of("l3", s))
    lo, hi = r_range(build, n, s)
    r = _r2(g.uniform(lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo)))
    fs = _r2(1.0 + float(g.uniform(-1.0, 1.0)) * float(s["world.airflow"]["cau_flow_rel"]), 4)
    c, lim = registered(clean_pc(n, r, fs, s), build, s)
    cov_lo, cov_hi = s["world.blocker.placement"]["covered_m2"]
    t_on = float(g.uniform(600.0, N - 600.0))
    dur = float(g.uniform(60.0, 240.0))
    fc = None
    if build == "pad":
        while True:
            t_fc = float(g.uniform(300.0, N - 300.0))
            if abs(t_fc - t_on) >= 360.0 and abs(t_fc - (t_on + dur)) >= 360.0:
                break
        fc = dict(unit="cau_l3", t_min=_r2(t_fc, 2), drop_pa=_r2(g.uniform(4.0, 8.0)))
    site = dict(schema="station_site/2", site_id=f"l3_sample_{int(seed)}", scene="practice", kind="practice", seed=int(seed),
                cell=None, cabin_air_units={"cau_l3": dict(build=build, r_face=r, fan_stop="allowed", inhibit_reason=None,
                                                           clean_reference_pa=c, advisory_limit_pa=lim, flow_scale=fs)},
                inter_module={}, blocker=None, small_items=[], lighting={"l3": "general"},
                telemetry=dict(log_minutes=N, onset_min=None, drift=None, pouch_onsets_min={}, noise_seed=int(g.integers(0, 2 ** 31 - 1)),
                               advisory_expected=dict(points_elsewhere=False, blocked_unit_over_limit=False), synth_version=VERSION,
                               filter_change=fc, temporary_covers=[dict(inlet="ri_l3_1", cover_m2=_r2(g.uniform(cov_lo, cov_hi), 4),
                                                                        t_on_min=_r2(t_on, 2), t_off_min=_r2(t_on + dur, 2))]),
                truth=dict(cover_m2=None))
    return site
