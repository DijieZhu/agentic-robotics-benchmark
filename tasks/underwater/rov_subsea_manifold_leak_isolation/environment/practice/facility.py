"""The practice facilities' worlds: the training tank and the sister site (task 07, DESIGN v1.2 section 1.6; rulings
JUDG 4.5, K4, U1; package P20).  Service side only (the practice service's account, mode 700): never shipped to the
agent's account.

A facility package is a deterministic tar.xz like a site package (subseacell.site.pack: regular files, sorted,
canonical headers) of schema "subseacell-practice/1", built by DEV/gen/practice_sites.py from the facility descriptions
of DEV/gen/drawings/facilities.py (package P13: the drawings and sample frames come from the same descriptions):
    facility.json     kind (tank | sister), label (manifold-leak/practice/...), the description (static primitives,
                      panel pose, tags, board plates, stands, cage, laydown), the valves and ports, the session-draw
                      ranges, the work-order template, the vehicle's water-interaction coefficients (the task world's
                      one draw, decision U1: the same numbers as the scored stations')
    panel_skid.json   the panel segment (gen/panel_cad.py, fewer columns / rows / ports)
    textures/*.png    tag plates, board plates, the cage plate (gen/tag_textures.py)

The guard (`guard_package`, `guard_path`) refuses anything that is not such a package: a scored-site package (schema
subseacell-site/1), a label outside manifold-leak/practice/, any JSON member that mentions a scored set's labels
(manifold-leak/v1, manifold-leak/dev), a path outside the service's sites folder, a symbolic link, a path through a
"hidden" folder, a package whose sha256 differs from sites.json.  `world_factory` refuses every object that is not a
FacilitySite of a known kind, so a scored site cannot be built here whatever the caller does.

A session = FacilitySite.session(seed, setup): every per-session draw (tank: the operator's settings; sister: tide,
water, plume, clutter, leak pressure and time constant, gusts, sensor noise) is a pure function of (package, seed,
setup), and so is the world.

Worlds (TankWorld, SisterWorld) are subclasses of subseacell.world.SiteWorld: the same vehicle (build, hydrodynamics
coefficients, firmware, protective behaviours, sensors, grasp rules, hot stab, stands, cage, plume couplings, render twin,
acoustic model), the same time stepping (SiteWorld.advance), the same frames and pings; only the structure, the panel
segment, the process and the current differ.  Amendment 01 (DECISIONS_P2_BEH.md 11): the cage with its two reflector
poles is in the acoustic model (decision CAL-2, subseacell.world.cage_acoustic), and the tank's roof-grating coupon and
the sister site's roof grating are modelled like a scored site's roof (decision CAL-3): bars without contacts, seen by
the DVL, the camera and the sonar, and a proxy plate over the grating that carries the contacts and is what the DVL
tracks (the package's primitive of role roof_proxy, gen/practice_sites.py).  The tank adds `instruments()`: the tank's instruments (tracking-system
pose of the vehicle at 5 Hz in the tank frame, the six handle angle encoders, the hot-stab receptacles' status lamps,
the flow generator's and the bubbler's set points) - physical readings of a real test tank, never a score.  The sister
site has no instruments (the server answers instruments() with forbidden there).  Amendment 02 (package P38, decision
AM2-8): the sister site's facility.json may carry a static-scenery block (scenery_layout/1, placed by the scored sites'
rules, gen/practice_sites.py); the world adds it after every other part, in static zone bodies, to its three models
exactly as subseacell.world.SiteWorld does (_add_scenery, _scenery_classes, _scenery_acoustic); the tank has none, and a
package without the block gives the world it gave before.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time

import numpy as np
import mujoco

import uwkit
from uwkit.dynamics import current as CUR
from uwkit.params import HydroCoeffs
from uwkit.sensors.suite import SensorSuite, SensorWorld, SuiteConfig
from uwkit.sensors.acoustics import WaterColumn
from uwkit.sensors.sonar import ScanningSonarConfig
from uwkit.sensors import water as UWAT

from subseacell import site as SITE
from subseacell import world as WD
from subseacell import vehicle as VH
from subseacell import devices as DV
from subseacell import grasp as GR
from subseacell import stab as ST
from subseacell import objects as OB
from subseacell import protect as PT
from subseacell import firmware as FW
from subseacell import flow as FL
from subseacell import scenery as SCN
from subseacell import plume as PL
from subseacell import sensing as SE
from subseacell import twin as TW

SCHEMA = "subseacell-practice/1"
FACILITY_JSON = "facility.json"
LABEL_ROOT = "manifold-leak/practice/"
KINDS = ("tank", "sister")
SCORED_LABEL_MARKS = ("manifold-leak/v1", "manifold-leak/dev")      # the scored sets' label roots (P12 / P26)
SCORED_SCHEMAS = (SITE.SCHEMA,)                                     # subseacell-site/1
REPLY_WORDS = ("score", "ratio", "reward")                          # never in any reply (DESIGN 5, the acceptance)
BINARY_KEYS = ("image_png_b64", "intensity_u8_b64")                 # base64 sensor data (not text)
TRACK_RATE_HZ = 5.0
TRACK_POS_DEC = 4                   # tracking system: 0.1 mm
TRACK_QUAT_DEC = 6
ENCODER_DEC = 2                     # handle angle encoders: 0.01 deg
TRACK_MAX = 9000                    # samples kept (30 min at 5 Hz; a tank run is at most 600 s)
ACCEPT_DEPTH_M = 1.0e-6
ROOF_PROXY_ROLE = "roof_proxy"      # gen/practice_sites.py: the grating's proxy plate (amendment 01, decision CAL-3)


def _static_kw(pr) -> dict:
    """MjSpec geom keywords of a facility primitive: colliding ones with the cell's surface friction, the others
    without contacts.  The roof grating's proxy plate (amendment 01, decision CAL-3) is built like the scored sites'
    (subseacell.world.SiteWorld._build_manifold, decision P18-3): group 0 and opaque, so that the DVL's rays meet it
    (MuJoCo's ray casts skip a geom without material whose alpha is 0, and the package gives it alpha 0), the roof's
    friction, contacts on; the render twin deletes it (twin_scene) and the acoustic model skips it (not acoustic)."""
    kw = dict(name=pr["name"], rgba=list(pr["rgba"]), friction=list(OB.SURFACE_FRICTION))
    if pr["role"] == ROOF_PROXY_ROLE:
        kw.update(group=TW.GROUP_SCENE, friction=list(OB.ROOF_FRICTION), rgba=list(WD.ROOF_PROXY_RGBA))
    if not pr["physics"]:
        kw.update(contype=0, conaffinity=0)
    return kw


class GuardError(SITE.SitePackageError):
    """A package / path the practice service must not load."""


def canonical(obj) -> bytes:
    return SITE.dumps(obj)


def _jsonable(x):
    return WD._jsonable(x)


def hydro_digest(coeffs: dict) -> str:
    """sha256 of the vehicle's water-interaction coefficients (the numbers, not the set's name)."""
    d = {k: [float(v) for v in coeffs[k]] for k in ("added_mass", "damping_linear", "damping_quadratic")}
    d["rho_ref"] = float(coeffs["rho_ref"])
    return hashlib.sha256(canonical(d)).hexdigest()


def world_hydro_digest(world) -> dict:
    """The coefficients a built world's vehicle really uses (its build meta): sha256 of the coefficient set and of the
    hull's hydro record (what the simulator's hydro layer reads)."""
    meta = world.vb.meta
    hull = meta["hydro_bodies"]["hull"]
    return dict(coeffs=hydro_digest(meta["hydro_coeffs"]),
                hull_record=hashlib.sha256(canonical(_jsonable(hull))).hexdigest())


# ==================================================================================================== guards
def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def guard_path(path, sites_dir) -> str:
    """The real path of a facility package if it may be loaded: a regular file (not a symbolic link) directly inside
    the service's sites folder, no path component named like a hidden set.  Raises GuardError otherwise."""
    p = os.fspath(path)
    base = os.path.realpath(os.fspath(sites_dir))
    full = p if os.path.isabs(p) else os.path.join(base, p)
    if os.path.islink(full):
        raise GuardError("a symbolic link is not a practice package")
    real = os.path.realpath(full)
    if os.path.dirname(real) != base:
        raise GuardError("a practice package must sit in the practice sites folder")
    parts = [x.lower() for x in re.split(r"[\\/]+", p)] + [x.lower() for x in real.split(os.sep)]
    if any("hidden" in x for x in parts) or ".." in p.replace("\\", "/").split("/"):
        raise GuardError("path refused")
    if not os.path.isfile(real):
        raise GuardError("not a regular file")
    return real


def guard_package(blob: bytes, sha256_hex=None) -> dict:
    """Unpack and check a facility package (see the module doc); returns the members.  Raises GuardError."""
    if sha256_hex is not None and hashlib.sha256(blob).hexdigest() != str(sha256_hex).lower():
        raise GuardError("package sha256 differs from the service's list")
    try:
        files = SITE.unpack(blob, strict=True)
    except SITE.SitePackageError as e:
        raise GuardError(f"not a canonical package: {e}") from None
    if SITE.SITE_JSON in files:
        raise GuardError("a scored-site package (site.json) is refused")
    if FACILITY_JSON not in files:
        raise GuardError("no facility.json")
    for name, b in files.items():
        if name.endswith(".json"):
            try:
                obj = SITE.loads(b)
            except ValueError:
                raise GuardError(f"{name}: not JSON") from None
            for s in _strings(obj):
                if any(m in s for m in SCORED_LABEL_MARKS):
                    raise GuardError(f"{name}: mentions a scored set")
        elif not name.startswith("textures/") or not name.endswith(".png"):
            raise GuardError(f"unexpected member {name!r}")
    data = SITE.loads(files[FACILITY_JSON])
    if data.get("schema") != SCHEMA or data.get("schema") in SCORED_SCHEMAS:
        raise GuardError(f"schema {data.get('schema')!r} is not a practice facility")
    label = str(data.get("label", ""))
    if not label.startswith(LABEL_ROOT) or any(m in label for m in SCORED_LABEL_MARKS):
        raise GuardError(f"label {label!r} refused")
    if data.get("kind") not in KINDS:
        raise GuardError("unknown facility kind")
    if data.get("panel_skid") not in files:
        raise GuardError("panel skid member missing")
    return files


def server_site_guard(site_obj) -> bool:
    """The server's site_guard (subseacell.server, practice reset path): only facility sessions pass."""
    return isinstance(site_obj, FacilitySite) and site_obj.kind in KINDS and str(site_obj.label).startswith(LABEL_ROOT)


# ==================================================================================================== replies
def reply_words(obj, words=REPLY_WORDS) -> list:
    """Words of `words` found (case-insensitive) in the keys and text values of a reply (base64 sensor data skipped)."""
    hits = []

    def walk(o, key=None):
        if isinstance(o, dict):
            for k, v in o.items():
                ks = str(k).lower()
                hits.extend(f"key {k}" for w in words if w in ks)
                if str(k) in BINARY_KEYS:
                    continue
                walk(v, str(k))
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, key)
        elif isinstance(o, str):
            s = o.lower()
            hits.extend(f"text in {key}: {w}" for w in words if w in s)
    walk(obj)
    return hits


# ==================================================================================================== the package
class FacilitySite(SITE.Site):
    """A facility package (or one session of it).  `data` = facility.json; `manifold` = {"primitives": the static
    structure}; `skid` = the panel segment.  dict() is what the server and the event log read (valves, ports, tags,
    stab, label, the work order under "mission")."""

    def __init__(self, data: dict, skid: dict, files: dict, blob: bytes = None):
        d = dict(data)
        d.setdefault("pose_o", d["frame_w"]["origin"])
        d.setdefault("pose_psi", d["frame_w"]["psi"])
        super().__init__(d, {"primitives": d["description"]["primitives"]}, skid, files)
        self.kind = d["kind"]
        self.label = d["label"]
        self.blob = blob
        self.session_seed = None
        self.setup = None
        self.params = None

    @classmethod
    def from_package(cls, blob: bytes, sha256_hex=None) -> "FacilitySite":
        files = guard_package(blob, sha256_hex)
        data = SITE.loads(files[FACILITY_JSON])
        return cls(data, SITE.loads(files[data["panel_skid"]]), files, blob)

    @classmethod
    def load(cls, path, sites_dir, sha256_hex=None) -> "FacilitySite":
        real = guard_path(path, sites_dir)
        with open(real, "rb") as f:
            return cls.from_package(f.read(), sha256_hex)

    def to_package(self) -> bytes:
        return self.blob if self.blob is not None else SITE.pack(self.files)

    def dict(self) -> dict:
        return copy.deepcopy(self.data)

    def problems(self) -> list:
        return []

    def hydro_coeffs(self) -> HydroCoeffs:
        h = self.data["hydro"]["coeffs"]
        return HydroCoeffs(name="task_world_draw", added_mass=np.asarray(h["added_mass"], float),
                           damping_linear=np.asarray(h["damping_linear"], float),
                           damping_quadratic=np.asarray(h["damping_quadratic"], float), rho_ref=float(h["rho_ref"]))

    # ------------------------------------------------------------------------------------------ sessions
    def session(self, seed: int, setup=None) -> "FacilitySite":
        """This facility for one practice run: setup checked (tank; SetupError on a bad value), every per-session draw
        made from `seed`, the work order written (data["mission"])."""
        s = FacilitySite(copy.deepcopy(self.data), self.skid, self.files, self.blob)
        s.session_seed = int(seed)
        if self.kind == "tank":
            s.setup = normalise_tank_setup(self.data, setup)
            s.params = _tank_params(s.data, s.setup, int(seed))
            s.data["stab"] = dict(initial_port=s.setup["stab_port"])
        else:
            if setup not in (None, {}):
                raise SetupError("the sister site takes no setup")
            s.setup = {}
            s.params = _sister_params(s.data, int(seed))
        s.data["sensor_seed"] = int(seed) & 0x7FFFFFFFFFFFFFFF
        s.data["mission"] = _work_order(s)
        return s


# ==================================================================================================== tank set-up
class SetupError(ValueError):
    """A tank set-up the operator cannot run (the detail names the setting)."""


def _num(x, name, lo, hi):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise SetupError(f"{name} must be a number")
    v = float(x)
    if not lo - 1e-12 <= v <= hi + 1e-12:
        raise SetupError(f"{name} must be in [{lo:g}, {hi:g}]")
    return min(max(v, lo), hi)


def normalise_tank_setup(data: dict, setup) -> dict:
    """The tank operator's settings for one run (defaults where not given): see data["setup_schema"]."""
    sch = data["setup_schema"]
    setup = {} if setup is None else setup
    if not isinstance(setup, dict):
        raise SetupError("setup must be an object")
    if set(setup) - set(sch["keys"]):
        # the detail never echoes the caller's text
        raise SetupError("unknown setting; the settings are %s" % ", ".join(sch["keys"]))
    out = {}
    lo, hi = sch["current_m_s"]
    cur = setup.get("current_m_s", 0.0)
    if isinstance(cur, (list, tuple)):
        if not 1 <= len(cur) <= int(sch["current_steps_max"]):
            raise SetupError("current_m_s: a number or 1 .. %d [t_s, m_s] steps" % int(sch["current_steps_max"]))
        steps, t_prev = [], -1.0
        for st in cur:
            if not isinstance(st, (list, tuple)) or len(st) != 2:
                raise SetupError("current_m_s: each step is [t_s, m_s]")
            t = _num(st[0], "current_m_s step time", 0.0, float(data["max_sim_s"]))
            if t <= t_prev:
                raise SetupError("current_m_s: step times must increase")
            steps.append([round(t, 3), round(_num(st[1], "current_m_s", lo, hi), 4)])
            t_prev = t
        if steps[0][0] != 0.0:
            steps.insert(0, [0.0, 0.0])
        out["current_m_s"] = steps
    else:
        out["current_m_s"] = [[0.0, round(_num(cur, "current_m_s", lo, hi), 4)]]
    q = setup.get("bubbler_L_min", 0.0)
    q = _num(q, "bubbler_L_min", 0.0, sch["bubbler_L_min"][1])
    if 0.0 < q < sch["bubbler_L_min"][0]:
        raise SetupError("bubbler_L_min: 0 (off) or %g .. %g" % tuple(sch["bubbler_L_min"]))
    out["bubbler_L_min"] = round(q, 4)
    tb = setup.get("turbidity", sch["turbidity_default"])
    if tb not in sch["turbidity"]:
        raise SetupError("turbidity must be one of %s" % ", ".join(sch["turbidity"]))
    out["turbidity"] = tb
    out["test_port_bar"] = round(_num(setup.get("test_port_bar", 0.0), "test_port_bar", *sch["test_port_bar"]), 3)
    hv = setup.get("handles", {})
    if not isinstance(hv, dict) or set(hv) - set(sch["handles"]) or any(v not in ("open", "closed") for v in hv.values()):
        raise SetupError("handles: {tag: 'open' | 'closed'} for the tags %s" % ", ".join(sch["handles"]))
    out["handles"] = {t: hv.get(t, "open") for t in sch["handles"]}
    d1 = setup.get("d1_stand", sch["d1_stand"][0])
    if d1 not in sch["d1_stand"]:
        raise SetupError("d1_stand must be one of %s" % ", ".join(sch["d1_stand"]))
    out["d1_stand"] = d1
    sp = setup.get("stab_port", sch["stab_port"][0])
    if sp not in sch["stab_port"]:
        raise SetupError("stab_port must be one of %s" % ", ".join(sch["stab_port"]))
    out["stab_port"] = sp
    return out


def _sub(seed: int, tag: str, bits=63) -> int:
    return int.from_bytes(hashlib.sha256(f"{int(seed)}/{tag}".encode()).digest()[:8], "little") & ((1 << bits) - 1)


def _tank_params(data: dict, setup: dict, seed: int) -> dict:
    D = data["description"]["params"]
    c550 = dict(zip(data["setup_schema"]["turbidity"], D["turbidity_c550"]))[setup["turbidity"]]
    q = setup["bubbler_L_min"]
    lo, hi = D["bubbler"]["gas_L_min"]
    rate = 0.0 if q <= 0.0 else 100.0 + (min(max(q, lo), hi) - lo) * 200.0 / (hi - lo)
    return dict(c550=float(c550), current_steps=[list(s) for s in setup["current_m_s"]],
                ramp_m_s2=float(data["tank"]["current_ramp_m_s2"]), gust_frac=float(D["flow"]["gust_frac"]),
                gust_tau_s=float(D["flow"]["gust_tau_s"]), gm_seed=_sub(seed, "gust", 31),
                plume=dict(rate=float(rate), w_b=float(D["bubbler"]["rise_m_s"]), seed=_sub(seed, "plume", 47)),
                port_bar=float(setup["test_port_bar"]), handles=dict(setup["handles"]), d1_stand=setup["d1_stand"],
                stab_port=setup["stab_port"])


def _sister_params(data: dict, seed: int) -> dict:
    """One draw of the sister site's conditions (the ranges in data["draw"]: tide, gusts, in-structure factor, water,
    seep, clutter, the spool's pressure and decay)."""
    r = data["draw"]
    rng = np.random.default_rng(_sub(seed, "conditions"))
    u = lambda key: float(rng.uniform(*r[key]))  # noqa: E731
    flow = dict(U2=u("tide_peak_m_s"), theta=float(rng.uniform(0.0, 2 * math.pi)), t_slack=u("tide_slack_s"),
                minor_ratio=u("tide_minor_ratio"), f_inside=u("inside_factor"), gm_coef=u("gust_frac"),
                gm_tau=u("gust_tau_s"), gm_seed=_sub(seed, "gust", 31))
    c550 = u("water_c550")
    load = SE.particle_load(c550)
    n_clump = int(min(3, int(rng.integers(1, 3)) + (1 if load > 2.0 else 0)))
    if c550 < 0.55 and rng.uniform() < 0.3:
        n_clump = int(rng.integers(0, 2))
    zroof = float(data["frame_w"]["origin"][2]) + float(data["structure"]["roof_z"])
    clutter = dict(n=n_clump, radius=rng.uniform(0.10, 0.25, n_clump).tolist(),
                   ts_db=rng.uniform(-38.0, -28.0, n_clump).tolist(), z_band=[zroof + 0.8, zroof + 7.0],
                   domain=30.0, seed=_sub(seed, "clutter", 47))
    plume = dict(rate=u("seep_per_s"), w_b=u("rise_m_s"), seed=_sub(seed, "plume", 47))
    proc = dict(p0_bar=u("p0_bar"), tau_s=u("tau_s"))
    # the work order's forecast: slack +- the forecast error, direction +- its error, the rate class of the tide
    se, de = float(r["forecast_slack_err_s"]), math.radians(float(r["forecast_direction_err_deg"]))
    f_slack = flow["t_slack"] + float(rng.uniform(-se, se))
    f_dir = (flow["theta"] + float(rng.uniform(-de, de))) % (2 * math.pi)
    lo, hi = r["tide_peak_m_s"]
    k = min(2, int(3 * (flow["U2"] - lo) / (hi - lo)))
    forecast = dict(slack_time_s=round(f_slack, 1), slack_err_s=se, flood_direction_rad=round(f_dir, 4),
                    direction_err_rad=round(de, 4), rate_class=("neap", "mean", "spring")[k])
    return dict(flow=flow, c550=c550, clutter=clutter, plume=plume, process=proc, forecast=forecast)


def _work_order(s: FacilitySite) -> dict:
    """The run's work order (mission(); the server adds the budgets).  Keys of subseacell.server.MISSION_KEYS only."""
    d = s.data
    wo = copy.deepcopy(d["work_order"])
    if s.kind == "tank":
        p = s.setup
        wo["state_table"] = dict(p["handles"])
        wo["notes"] = dict(wo.get("notes", {}), run_settings=dict(
            current_m_s=[list(x) for x in p["current_m_s"]], bubbler_L_min=p["bubbler_L_min"],
            turbidity=p["turbidity"], test_port_bar=p["test_port_bar"], d1_stand=p["d1_stand"],
            stab_port=p["stab_port"]))
        if p["d1_stand"] != "panel":
            wo["debris_report"] = d["tank"]["debris_report_bay"]
    else:
        wo["tide_forecast"] = dict(s.params["forecast"])
    return _jsonable(wo)


# ==================================================================================================== processes
class TankProcess:
    """The training panel has no process: the handles turn, nothing flows.  The test port is fed by the tank's pressure
    unit at the run's set pressure (constant); the bubbler runs at its set gas flow whatever the handles do."""

    def __init__(self, port_bar: float, dt: float = 0.1):
        self.p = float(port_bar)
        self.dt = float(dt)
        self.k = 0

    def step_to(self, t, angles_deg=None, conducting=None):
        self.k = max(self.k, int(math.floor(float(t) / self.dt + 1e-9)))

    def rate_scale_step(self, j):
        return 1.0

    def rate_scale(self, t):
        return 1.0

    def leak_pressure(self):
        return self.p

    def segment_pressure(self, branch):
        return self.p if int(branch) == 1 else 0.0

    def leak_isolated(self):
        return False


class PletProcess:
    """The sister site's spool: pipeline --BV1-- spool (the leaking flange, TP-1) --PV1-- header to the export line.
    Both ends are pressure sources; the spool holds p0 while either valve conducts (angle < closed_min_deg) and decays
    p <- p exp(-dt / tau) once both are closed (the seep drains it).  Bubble emission scale = p / p0, read on the 0.1 s
    grid by index (the plume's contract, as subseacell.process.Process)."""

    TAGS = ("BV1", "PV1")

    def __init__(self, p0_bar: float, tau_s: float, dt: float = 0.1, closed_min_deg: float = 80.0):
        self.p0, self.tau, self.dt = float(p0_bar), float(tau_s), float(dt)
        self.decay = math.exp(-self.dt / self.tau)
        self.closed_min_deg = float(closed_min_deg)
        self.k = 0
        self._p = [self.p0]
        self._conn = [True]

    def step_to(self, t, angles_deg=None, conducting=None):
        if conducting is None:
            conducting = {k: float(angles_deg[k]) < self.closed_min_deg for k in self.TAGS}
        conn = bool(conducting["BV1"] or conducting["PV1"])
        kt = int(math.floor(float(t) / self.dt + 1e-9))
        while self.k < kt:
            self._p.append(self.p0 if conn else self._p[-1] * self.decay)
            self._conn.append(conn)
            self.k += 1

    def rate_scale_step(self, j):
        if j < 0:
            return 1.0
        if j > self.k:
            raise RuntimeError("process grid point not computed yet")
        return self._p[j] / self.p0

    def rate_scale(self, t):
        return self.rate_scale_step(int(math.floor(float(t) / self.dt + 1e-9)))

    def leak_pressure(self):
        return float(self._p[self.k])

    def segment_pressure(self, branch):
        return float(self._p[self.k]) if int(branch) == 1 else 0.0

    def leak_isolated(self):
        return not self._conn[self.k]


# ==================================================================================================== currents
class ScheduleTide(CUR.Tidal):
    """The tank's flow generator: a horizontal current along +x of the tank whose set speed follows the run's steps
    [[t_s, m_s], ...], each change slewed at `ramp` m/s per second (the pumps' acceleration)."""

    def __init__(self, steps, ramp_m_s2, direction=(1.0, 0.0)):
        super().__init__(0.0, direction=direction, period_s=FL.TIDE_PERIOD_S, t_peak_s=0.0, minor_ratio=0.0)
        self.steps = [(float(t), float(v)) for t, v in steps]
        self.ramp = float(ramp_m_s2)
        self.vmax = max(v for _, v in self.steps)

    def _eval(self, seg, t):
        t_s, v_s, v_t = seg
        dv = v_t - v_s
        if dv == 0.0 or self.ramp <= 0.0 or t >= t_s + abs(dv) / self.ramp:
            return v_t
        return v_s + math.copysign(self.ramp * max(0.0, t - t_s), dv)

    def speed(self, t):
        """Speed at time t: the run starts at the first step's speed; at each later step the pumps slew from the speed
        they have then towards the new set speed."""
        t = float(t)
        seg = (0.0, self.steps[0][1], self.steps[0][1])
        for ti, vi in self.steps[1:]:
            if t < ti:
                break
            seg = (ti, self._eval(seg, ti), vi)
        return self._eval(seg, t)

    def setpoint(self, t):
        sp = self.steps[0][1]
        for t0, v0 in self.steps:
            if float(t) >= t0:
                sp = v0
        return sp

    def horizontal(self, t):
        return self.speed(t) * self.e


class TankFlow(FL.StationFlow):
    """The tank current: ScheduleTide + the cell's residual turbulence (Gauss-Markov, std = gust_frac x the largest
    set speed + 0.015 m/s), the cell's 1/7 profile normalised at 2 m above the floor (P13's tank_flow), no structure
    reduction.  Same fast paths as the stations' flow (subseacell.flow.StationFlow)."""

    def __init__(self, steps, ramp_m_s2, gust_frac, gust_tau_s, gm_seed, depth_m):
        tide = ScheduleTide(steps, ramp_m_s2)
        fs = FL.FlowSpec(U2=tide.vmax, theta=0.0, t_slack=-FL.TIDE_PERIOD_S / 4.0, minor_ratio=0.0, f_inside=1.0,
                         gm_coef=float(gust_frac), gm_tau=float(gust_tau_s), gm_seed=int(gm_seed), t_meas=0.0)
        env = FL.StructureEnvelope(lx=0.0, ly=0.0, roof_z=0.0, mudmat_thk=0.0, ramp_m=0.3)
        super().__init__(fs, FL.SitePose([0.0, 0.0, -float(depth_m)], 0.0), float(depth_m), env)
        self.tide = tide
        self.free = CUR.BoundaryLayer(CUR.Sum([self.tide, self.gust]), self.seabed_depth, delta=FL.BL_DELTA_M,
                                      exponent=FL.BL_EXPONENT, z_ref=FL.Z_REF_M)
        self.field = CUR.Scaled(self.free, self.sfac)
        self.mean_field = CUR.Scaled(CUR.BoundaryLayer(self.tide, self.seabed_depth, delta=FL.BL_DELTA_M,
                                                       exponent=FL.BL_EXPONENT, z_ref=FL.Z_REF_M), self.sfac)

    def v2m(self, t):
        return self.tide.horizontal(float(t))


# ==================================================================================================== stands
class StandSleep(WD.DebrisSleep):
    """subseacell.world.DebrisSleep for one of several stands: its own Morison model, held eyes of its own body only."""

    def __init__(self, world, body, hydro):
        self.hydro = hydro
        super().__init__(world, body=body)
        self.eye_names = {n for n, e in world.eyes.items() if e["body"] == body}
        self.stab_bid = world.m.body("stab").id

    def near(self, extra=0.0):
        """DebrisSleep.near, except that a hot stab latched in a receptacle and not held does not count: it cannot
        move, so it cannot touch the stand (the training tank's D1 stands in front of the parked stab)."""
        d = self.d
        c = d.xpos[self.bid]
        stab_static = self.W.stab.state == "Locked" and not (self.W.gripper.held is not None
                                                              and self.W.gripper.held[1] == "stab")
        for b, r in self.others:
            if b == self.stab_bid and stab_static:
                continue
            p = d.xpos[b]
            dx, dy, dz = float(p[0] - c[0]), float(p[1] - c[1]), float(p[2] - c[2])
            lim = r + self.r_self + self.margin + extra
            if dx * dx + dy * dy + dz * dz < lim * lim:
                return True
        return False

    def wake(self):
        m, d = self.m, self.d
        for g, (ct, ca) in zip(self.gids, self.bits):
            m.geom_contype[g] = ct
            m.geom_conaffinity[g] = ca
        m.body_gravcomp[self.bid] = self.gravcomp0
        d.qvel[self.va:self.va + 6] = 0.0
        self.asleep = False
        self.t_rest = 0.0
        self.n_wake += 1
        self.hydro.reset()
        self.W.events.append(dict(t=float(d.time), src="world", ev="debris_wake", body=self.W.m.body(self.bid).name))

    def post_step(self, dt):
        if self.asleep:
            return
        g = self.W.gripper.held
        if g is not None and g[0] == "eye" and g[1] in self.eye_names:
            self.t_rest = 0.0
            return
        d = self.d
        v = d.qvel[self.va:self.va + 6]
        if float(v[:3] @ v[:3]) < self.v_rest ** 2 and float(v[3:] @ v[3:]) < self.w_rest ** 2 \
                and not self.near(self.hyst):
            self.t_rest += dt
            if self.t_rest >= self.rest_s - 1e-9:
                self.sleep()
        else:
            self.t_rest = 0.0


class Sleepers:
    """Every stand's sleeper behind SiteWorld.advance's single `debris_sleep` slot (none at the sister site)."""

    def __init__(self, items=()):
        self.items = list(items)

    def pre_step(self):
        for s in self.items:
            s.pre_step()

    def post_step(self, dt):
        for s in self.items:
            s.post_step(dt)

    def restart(self):
        for s in self.items:
            s.restart()

    @property
    def asleep(self):
        return all(s.asleep for s in self.items)


class _Gate:
    def __init__(self, fh, sleeper_ref):
        self.fh, self.ref = fh, sleeper_ref

    def apply(self, t):
        s = self.ref()
        if s is None or not s.asleep:
            self.fh.apply(t)


# ==================================================================================================== the worlds
class FacilityWorld(WD.SiteWorld):
    """A practice facility as the cell's three models (physics, render twin, acoustic model) with the scored cell's
    vehicle, firmware, sensors and devices.  Build order and runtime = subseacell.world.SiteWorld's."""

    kind = None

    def __init__(self, site: FacilitySite, *, config=None, camera_sizes=((640, 480), (320, 240)), cameras=True,
                 mesh_root=None, timing=False, sensor_seed=None):
        if not isinstance(site, FacilitySite) or site.params is None or site.kind != self.kind:
            raise GuardError("not a session of this practice facility")
        import gc
        gc.collect()
        c_build0, w_build0 = time.process_time(), time.perf_counter()
        self.build = {}
        last = [c_build0, w_build0]

        def mark(name):
            c, w = time.process_time(), time.perf_counter()
            self.build[name] = dict(cpu_s=c - last[0], wall_s=w - last[1])
            last[0], last[1] = c, w
        self.site = site
        self.sd = site.dict()
        sd = self.sd
        self.P = site.params
        self.cfg = dict(config or {})
        self.timing = bool(timing)
        self.T = WD.Timer()
        self.mesh_root = mesh_root
        self.sensor_seed = int(sd["sensor_seed"] if sensor_seed is None else sensor_seed)
        D = sd["description"]
        # ------------------------------------------------------------------ vehicle (the task world's coefficients)
        hydro = site.hydro_coeffs()
        self.hydro_info = dict(digest=hydro_digest(sd["hydro"]["coeffs"]))
        self.seabed_depth = float(sd["depth_m"])
        self.z_bed = -self.seabed_depth
        cg = sd["cage"]
        vb = VH.build_vehicle(hydro, floor_depth=self.seabed_depth,
                              pos=(float(cg["pos_w"][0]), float(cg["pos_w"][1]), self.z_bed + 1.0))
        self.vb = vb
        s = vb.spec
        floor = [g for g in s.geoms if g.name == "seabed"]
        if floor and sd.get("floor_rgba"):
            floor[0].rgba = list(sd["floor_rgba"])
        mark("vehicle_spec")
        # ------------------------------------------------------------------ static structure
        self.pose = site.pose
        self.o_w = np.asarray(sd["frame_w"]["origin"], float)
        self.manifold_geoms = self._build_static(s)
        # ------------------------------------------------------------------ panel segment
        pf = sd["panel_frame_w"]
        self.P_o = np.asarray(pf["origin"], float)
        self.P_R = np.stack([np.asarray(pf["X"], float), np.asarray(pf["Y"], float), np.asarray(pf["Z"], float)], 1)
        vspecs = {v["key"]: DV.ValveSpec(v["tag"], float(v["breakaway_Nm"]), float(v["running_Nm"]),
                                         self._theta0(v)) for v in sd["valves"]}
        self.panel = DV.build_panel(s, site.skid, self.P_o, self.P_R, valves=vspecs,
                                    ports={p["key"]: p["name"] for p in sd["ports"]}, contact_rules=OB.SURFACE_RULES)
        self.valve_tags = {v["tag"]: v for v in sd["valves"]}
        for inf in self.panel["valves"].values():
            s.add_exclude(bodyname1="world", bodyname2=inf["body"])
        self.stab_info = ST.add_stab(s, "stab", pos=list(self.P_o + self.P_R[:, 2] * 0.3), vb=vb)
        self.receptacles = ST.add_receptacles(s, self.panel, "stab")
        self.marker_handle = DV.add_marker(vb, "marker")
        # ------------------------------------------------------------------ stands
        self.stands = []
        for st in self._stand_list():
            fs = OB.make_spec(st["kind"])
            info = OB.add_frame(s, st["body"], fs, list(st["pos_w"]), list(OB.frame_quat(st["yaw"])),
                                visual_eyes=False)
            self.stands.append(dict(body=st["body"], kind=st["kind"], spec=fs, info=info))
        if not any(st["body"] == "debris" for st in self.stands):
            # the event log reads a body named "debris": a facility without a stand carries an empty static one
            s.worldbody.add_body(name="debris", pos=list(self.o_w - np.array([0.0, 0.0, 0.5])))
        if self.stands:
            self.debris_spec = self.stands[0]["spec"]
            self.debris_info = self.stands[0]["info"]
        # ------------------------------------------------------------------ cage
        self.cage = OB.add_cage(s, "cage", cg["pos_w"], float(cg["yaw"]))
        self.handles = dict(self.panel["handles"])
        self.handles["marker"] = self.marker_handle
        self.handles["stab"] = self.stab_info["handle"]
        self.eyes = {}
        for st in self.stands:
            self.eyes.update(OB.frame_eyes(st["info"]))
        GR.add_constraints(s, self.handles, self.eyes)
        tg = [g for g in s.bodies if g.name == "cage"][0]
        for g in [g for g in tg.geoms if g.name == "cage_tag_plate"]:
            s.delete(g)
        # ------------------------------------------------------------------ static scenery (amendment 02, P38: the sister
        # site's facility.json["scenery"]; after every other part, as subseacell.world.SiteWorld does; none in the tank)
        self.scenery = None
        if self.sd.get("scenery") is not None:
            self._add_scenery(s, self.sd["scenery"])
        mark("scene_spec")
        # ------------------------------------------------------------------ current and simulator
        self.flow = self._make_flow()
        self.sim = uwkit.UnderwaterSim(vb, current=WD.FastSiteField(self.flow), seed=_sub(self.sensor_seed, "sim", 31),
                                       battery_soc0=float(sd.get("battery_soc0", 0.9)))
        hw = vb.meta.get("hotel_W") or {}
        for k in ("sonar", "dvl"):
            if k in hw:
                self.sim.set_hotel(k, float(hw[k]))
        m, d = self.sim.model, self.sim.data
        self.m, self.d = m, d
        if self.scenery is not None:
            self.scenery["compiled"] = SCN.check_compiled(m, self.scenery)
        mark("compile_sim")
        # ------------------------------------------------------------------ devices and rules
        self.events = []
        self.bank = WD.FastValveBank(m, d, list(self.panel["valves"].values()))
        self.gripper = WD.FastGripper(self.sim, self.handles, self.eyes, on_event=self._on("grasp"))
        self.process = self._make_process()
        self.port_branch = {p["name"]: p.get("branch") for p in sd["ports"]}
        self.stab = WD.FastStabMachine(m, d, self.stab_info, self.receptacles, pressure_bar=self._port_pressure,
                                       gripper=self.gripper, on_event=self._on("stab"))
        self.stand_hydros = [OB.FrameHydro(m, d, st["body"], st["spec"], self.sim.current) for st in self.stands]
        if self.stand_hydros:
            self.debris_hydro = self.stand_hydros[0]
        self.debris_sleep = None
        self._sleepers = []
        OB.install_hydro(self.sim, [_Gate(h, (lambda i=i: self._sleepers[i] if i < len(self._sleepers) else None))
                                    for i, h in enumerate(self.stand_hydros)])
        chart = sd["work_order"]["chart"]
        centre = chart.get("origin_xy") or list(self.o_w[:2])
        self.protect = PT.Protection(m, d, PT.ProtectConfig.from_values(self.cfg),
                                     site=PT.SiteFrame(centre_xy=(float(centre[0]), float(centre[1])), mudline_z=self.z_bed))
        # ------------------------------------------------------------------ navigation sensors
        self.column = WaterColumn(temperature_c=float(sd["water"]["temperature_c"]),
                                  salinity_psu=float(sd["water"]["salinity_psu"]))
        self.water_optics = SE.site_water(float(self.P["c550"]))
        self.acoustic_classes = self._acoustic_classes()
        scfg = SuiteConfig.for_vehicle(m, with_camera=False, scanning_sonar=None, usbl=None)
        rng = np.random.default_rng([self.sensor_seed & 0x7FFFFFFFFFFFFFFF, 1])
        self.suite = SensorSuite(m, d, SensorWorld(water=self.water_optics, ambient=UWAT.DARK, column=self.column,
                                                   material_overrides=self.acoustic_classes), rng, config=scfg)
        self.fw = FW.Firmware(self.sim, self.suite, vb.vehicle, config=FW.FirmwareConfig.from_values(self.cfg),
                              protect=self.protect)
        self.k_ctrl = max(1, int(round(self.fw.dt / self.sim.dt)))
        self.k_slow = max(1, int(round(WD.SLOW_DT_S / self.sim.dt)))
        mark("devices_firmware")
        # ------------------------------------------------------------------ plume (spin-up 150 s before the run)
        self.plume = self._make_plume()
        self.plume.attach_process(self.process)
        self.plume.advance_to(0.0)
        mark("plume_spinup")
        # ------------------------------------------------------------------ acoustic model + sonar
        self.acoustic = TW.AcousticModel(m, self.acoustic_scene())
        self.sonar = SE.PlumeSonar(self.acoustic.model, self.acoustic.materials(),
                                   np.random.default_rng([self.sensor_seed & 0x7FFFFFFFFFFFFFFF, 2]), plume=self.plume,
                                   acoustic=SE.AcousticSpec.for_water(float(self.P["c550"])), water=self.column,
                                   config=ScanningSonarConfig(), geomgroup=None, exclude_body=self.acoustic.mount_id,
                                   seabed_depth_m=self.seabed_depth)
        mark("acoustic_model")
        # ------------------------------------------------------------------ render twin + cameras
        self.twin = TW.RenderTwin(s, m, self.twin_scene())
        mark("render_twin")
        self.cams = {}
        self.camera_sizes = [tuple(int(v) for v in sz) for sz in camera_sizes]
        if cameras:
            from dataclasses import replace
            base = SuiteConfig.for_vehicle(m).camera
            for i, (w, h) in enumerate(self.camera_sizes):
                ccfg = replace(base, width=w, height=h, show_geomgroups=(TW.GROUP_FOUL,))
                self.cams[(w, h)] = SE.PlumeCamera(self.twin.model, self.water_optics,
                                                   np.random.default_rng([self.sensor_seed & 0x7FFFFFFFFFFFFFFF, 3, w, h]),
                                                   ambient=UWAT.DARK, config=ccfg, plume=self.plume,
                                                   bubbles=SE.BubbleOptics())
        mark("renderers")
        # ------------------------------------------------------------------ start pose and reset
        self._start = OB.cage_start_pose(self.cage, OB.vehicle_lowest_z(m, d))
        self._sleepers = [StandSleep(self, st["body"], h) for st, h in zip(self.stands, self.stand_hydros)]
        self.debris_sleep = Sleepers(self._sleepers)
        self.reset()
        mark("reset")
        self.build["total"] = dict(cpu_s=time.process_time() - c_build0, wall_s=time.perf_counter() - w_build0)

    # ====================================================================================== facility specifics
    def _theta0(self, v):
        return 0.0

    def _stand_list(self):
        return []

    def _make_flow(self):
        raise NotImplementedError

    def _make_process(self):
        raise NotImplementedError

    def _make_plume(self):
        raise NotImplementedError

    def _build_static(self, s):
        """The facility's static primitives (description frame + frame_w origin): colliding ones with the cell's
        surface friction, the others without contacts.  Returns {role: [geom names]} of the colliding ones."""
        out = {}
        wb = s.worldbody
        o = self.o_w
        for pr in self.sd["description"]["primitives"]:
            if pr.get("drop"):
                continue
            kw = _static_kw(pr)
            if pr["kind"] == "box":
                A = np.eye(3) if pr.get("axes") is None else np.asarray(pr["axes"], float).T
                wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(pr["half"]), pos=list(o + np.asarray(pr["centre"])),
                            quat=list(SITE.quat_of(A)), **kw)
            else:
                gt = mujoco.mjtGeom.mjGEOM_CAPSULE if pr["kind"] == "capsule" else mujoco.mjtGeom.mjGEOM_CYLINDER
                wb.add_geom(type=gt, size=[pr["radius"], 0, 0],
                            fromto=list(o + np.asarray(pr["a"])) + list(o + np.asarray(pr["b"])), **kw)
            if pr["physics"]:
                out.setdefault(pr["role"], []).append(pr["name"])
        return out

    def _acoustic_classes(self) -> dict:
        cls = {}
        for pr in self.sd["description"]["primitives"]:
            if pr.get("acls") and not pr.get("drop"):
                cls[pr["name"]] = pr["acls"]
            elif pr["role"] == ROOF_PROXY_ROLE and not pr.get("drop"):
                cls[pr["name"]] = WD.ROOF_PROXY_ACLS
        for role, names in self.panel["geoms"].items():
            for n in names:
                cls[n] = "steel"
        cls.update(self.cage.get("acoustic_overrides", {}))
        if self.sd.get("floor_acls"):
            cls["seabed"] = self.sd["floor_acls"]
        return self._scenery_classes(cls)                  # amendment 02: the scenery's classes (none without it)

    def acoustic_scene(self) -> dict:
        m = self.m
        skip = set(self.panel["geoms"].get("port_receptacle", ()))
        skip.update(pr["name"] for pr in self.sd["description"]["primitives"] if not pr.get("acoustic", True))
        static = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != 0:
                continue
            n = m.geom(g).name
            if not n or n in skip or int(m.geom_group[g]) == TW.GROUP_PROXY:
                continue
            static.append(n)
        cyl = []
        sk_ports = {p["key"]: p for p in self.site.skid["ports"]}
        for name, p in self.panel["ports"].items():
            r = 0.5 * float(sk_ports[p["key"]]["outer_diameter_mouth"])
            cyl.append(dict(name=f"port_{name}_steel", a=list(p["mouth_w"]), b=list(p["seat_w"]), radius=r, acls="steel"))
        moving = [dict(body=inf["body"], acls="steel") for inf in self.panel["valves"].values()]
        moving += [dict(body="stab", acls="steel"), dict(body="marker", acls="plastic")]
        moving += [dict(body=st["body"], acls="steel") for st in self.stands]
        cage_mv, cage_cls = WD.cage_acoustic(m, self.cage)           # the cage and its reflector poles (CAL-2)
        moving.append(cage_mv)
        classes = dict(self.acoustic_classes)
        classes.update(cage_cls)
        # amendment 02: the scenery (primitives as static geoms, rocks as static meshes; its names out of the body-0 list)
        return self._scenery_acoustic(dict(static_geoms=static, cylinders=cyl, moving=moving, classes=classes,
                                           sonar_site="sonar_head"))

    def twin_scene(self) -> dict:
        sd, site = self.sd, self.site
        oP, RP = self.P_o, self.P_R
        plates = []
        tags = {(t["of"], t["key"]): t for t in sd["tags"]}
        for t in site.skid["tags"]:
            st = tags[(t["of"], t["key"])]
            X, Y, _ = t["centre"]
            plates.append(dict(name=f"{t['of']}_{t['key']}", centre_w=list(oP + RP @ np.array([X, Y, 0.0])),
                               R_w=RP.tolist(), size=list(t["size"]), thickness=float(t["thickness"]),
                               image=site.texture(st["texture"]), fouling=None, code_id=st["code_id"],
                               text=st.get("text", "")))
        for pl in sd.get("board_plates", ()):
            n = np.asarray(pl["normal"], float)
            up = np.array([0.0, 0.0, 1.0])
            Rb = np.stack([np.cross(up, n), up, n], 1)
            plates.append(dict(name=pl["name"], centre_w=list(np.asarray(pl["centre"], float) + self.o_w), R_w=Rb.tolist(),
                               size=list(pl["size"]), thickness=0.003, image=site.texture(pl["texture"]), fouling=None,
                               code_id=pl["code_id"], text=""))
        cgt = self.cage["tag"]
        n = np.asarray(cgt["normal_w"], float)
        up = np.array([0.0, 0.0, 1.0])
        Rc = np.stack([np.cross(up, n), up, n], 1)
        th = float(self.cage["params"]["tag"]["thickness"])
        plates.append(dict(name="cage", centre_w=list(np.asarray(cgt["centre_w"], float) - n * th), R_w=Rc.tolist(),
                           size=list(cgt["size_m"]), thickness=th, image=site.texture(sd["cage"]["texture"]),
                           fouling=None, code_id=sd["cage"]["code_id"], text=""))
        paddles = []
        for inf in self.panel["valves"].values():
            h = inf["handle"]
            paddles.append(dict(body=h["body"], geom=h["paddle_geom"], centre_b=h["centre_b"], R_bp=h["R_bp"],
                                size=h["paddle_m"]))
        h = self.stab_info["handle"]
        paddles.append(dict(body=h["body"], geom=h["paddle_geom"], centre_b=h["centre_b"], R_bp=h["R_bp"], size=h["paddle_m"]))
        F = OB.FRAME_DEFAULTS
        rr = F["eye_ring_r"]
        eyes = []
        for st in self.stands:
            fs = st["spec"]
            for en, c in fs.eyes_b.items():
                c = np.asarray(c, float)
                base = np.array([c[0], fs.W / 2, fs.r]) if en == "drag" else np.array([c[0], c[1], fs.H - fs.r])
                eyes.append(dict(name=f"{st['body']}_eye_{en}", body=st["body"], centre_b=c.tolist(), u_b=[0.0, 1.0, 0.0],
                                 v_b=[0.0, 0.0, 1.0], ring_r=rr,
                                 post=[base.tolist(), (c - np.array([0.0, 0.0, rr])).tolist()]))
        segs = []
        for k, (a, b) in enumerate(sd.get("paint_lines", ())):
            segs.append(dict(name=f"laydown_paint_{k}", a=list(np.asarray(a, float) + self.o_w),
                             b=list(np.asarray(b, float) + self.o_w), radius=WD.LAYDOWN_LINE_R, rgba=TW.LAYDOWN_RGBA))
        proxies = [pr["name"] for pr in sd["description"]["primitives"]
                   if pr["role"] == ROOF_PROXY_ROLE and not pr.get("drop")]           # the DVL's plate: never drawn
        return dict(plates=plates, paddles=paddles, delete=proxies, hide=[], eyes=eyes, segments=segs, boxes=[],
                    mesh_root=self.mesh_root)

    # ====================================================================================== reset
    def reset(self):
        m, d, sim = self.m, self.d, self.sim
        self.events = []
        pos, quat = self._start
        sim.reset(pos=pos, quat=quat)
        self.bank.reset()
        mujoco.mj_forward(m, d)
        self.gripper.reset()
        self.stab.reset_locked(self.sd["stab"].get("initial_port", "PARK-A"))
        for h in self.stand_hydros:
            h.reset()
        self.gripper.attach_initial("marker")
        self.protect.reset()
        if self.debris_sleep is not None:
            self.debris_sleep.restart()
        self._n_protect_logged = 0
        mujoco.mj_forward(m, d)
        self.k = 0
        self.lights = 0.0
        self.sim.set_lights(0.0)
        self.hooks = []
        self.frame_hooks = []
        self.ended = False
        self.end_reason = None
        self.qhash = hashlib.sha256()
        self.n_frames = {sz: 0 for sz in self.camera_sizes}
        self.n_pings = 0
        self.frame_log = []
        self.valve_names = [v["tag"] for v in sorted(self.sd["valves"], key=lambda v: (v["kind"], v["branch"]))]
        self._vidx = [self.bank.names.index(n) for n in self.valve_names]
        a0 = self.bank.angles_deg()
        self.angles0 = {n: a0[n] for n in self.valve_names}
        self.angle_hist = [[a0[n] for n in self.valve_names]]
        self._cond = {n: a0[n] < 80.0 for n in self.valve_names}
        self.cpu_steps = 0.0
        self.T = WD.Timer()
        self.n_steps = 0
        self.fw.ingest(self.suite.step(sim.time))
        self._reset_extra()

    def _reset_extra(self):
        pass

    # ====================================================================================== record side
    def truth(self) -> dict:
        """The facility's true state (the service's own run record; nothing of it is sent to the program)."""
        d = self.d
        st = self.sim.state()
        out = dict(t=float(self.sim.time), ended=self.ended, end_reason=self.end_reason,
                   valves={n: float(a) for n, a in zip(self.valve_names, self.angle_hist[-1])},
                   spool_bar=float(self.process.leak_pressure()), stab=self.stab.public(),
                   gripper=dict(state=self.gripper.state, held=self.gripper.held),
                   marker=dict(pos=self.body_pose("marker")[0].tolist()),
                   vehicle=dict(pos=st.pos_w.tolist(), quat=st.quat_wb.tolist(), in_cage=OB.in_cage(self.cage, st.pos_w)),
                   battery=dict(soc=float(self.sim.battery.status()["soc"]), V=float(self.sim.battery.voltage)),
                   protect=dict(count=int(self.protect.count), ended=bool(self.protect.ended)),
                   stands={s["body"]: self.body_pose(s["body"])[0].tolist() for s in self.stands},
                   plume_particles=int(self.plume.n), pings=int(self.n_pings),
                   frames={f"{w}x{h}": n for (w, h), n in self.n_frames.items()})
        return _jsonable(out)


class TankWorld(FacilityWorld):
    """The training tank (12 x 8 x 4 m of sea water, frame T: origin on the floor at the centre, W = T shifted down by the
    water depth): the training panel segment (3 x 2 handles BV1-BV3 / PV1-PV3, PARK-A with the hot stab, TP-1 on the
    pressure unit), the bubbler under the roof-grating coupon, the flow generator, the code board, the D1 and D4
    stands, the cage, the laydown paint, the tracking system."""

    kind = "tank"

    def _theta0(self, v):
        return 90.0 if self.P["handles"].get(v["tag"]) == "closed" else 0.0

    def _stand_list(self):
        out = []
        for st in self.sd["stands"]:
            pos = st["pos_by_setup"][self.P["d1_stand"]] if "pos_by_setup" in st else st["pos_T"]
            yaw = st["yaw_by_setup"][self.P["d1_stand"]] if "yaw_by_setup" in st else st["yaw"]
            out.append(dict(body=st["body"], kind=st["kind"], pos_w=list(np.asarray(pos, float) + self.o_w), yaw=float(yaw)))
        return out

    def _make_flow(self):
        P = self.P
        return TankFlow(P["current_steps"], P["ramp_m_s2"], P["gust_frac"], P["gust_tau_s"], P["gm_seed"],
                        self.seabed_depth)

    def _make_process(self):
        return TankProcess(self.P["port_bar"])

    def _make_plume(self):
        p = self.P["plume"]
        src = np.asarray(self.sd["bubbler_T"], float) + self.o_w
        ps = PL.PlumeSpec(src_w=tuple(float(x) for x in src), rate=float(p["rate"]), w_b=float(p["w_b"]),
                          seed=int(p["seed"]), cull_dz=float(self.seabed_depth - self.sd["bubbler_T"][2]))
        return PL.Plume(ps, self.flow, None, allow_out_of_range=True)

    def _reset_extra(self):
        self.track = []
        self._track_from = 0
        self._n_slow = 0
        self._sample_instruments()

    def _slow_tick(self):
        super()._slow_tick()
        self._n_slow += 1
        if self._n_slow % int(round(1.0 / (TRACK_RATE_HZ * WD.SLOW_DT_S))) == 0:
            self._sample_instruments()

    def _sample_instruments(self):
        if len(self.track) >= TRACK_MAX:
            return
        d = self.d
        hb = self.m.body("hull").id
        p = d.xpos[hb] - self.o_w
        q = d.xquat[hb]
        ang = self.bank.angles_deg()
        self.track.append([round(float(self.sim.time), 3)] + [round(float(x), TRACK_POS_DEC) + 0.0 for x in p]
                          + [round(float(x), TRACK_QUAT_DEC) + 0.0 for x in q]
                          + [round(float(ang[n]), ENCODER_DEC) + 0.0 for n in self.valve_names])

    def lamps(self) -> dict:
        """The hot-stab receptacles' status lamps: empty / engaged (stab in, not latched) / latched."""
        st = self.stab.public()
        out = {}
        for p in self.sd["ports"]:
            n = p["name"]
            if st.get("port") != n:
                out[n] = "empty"
            else:
                out[n] = "latched" if st["state"] == "Locked" else "engaged"
        return out

    def setpoints(self, t=None) -> dict:
        t = float(self.sim.time) if t is None else float(t)
        P = self.P
        return dict(current_set_m_s=round(float(self.flow.tide.setpoint(t)), 4), current_heading_T="+x",
                    bubbler_L_min=float(self.site.setup["bubbler_L_min"]), turbidity=self.site.setup["turbidity"],
                    c550_per_m=float(P["c550"]), test_port_bar=float(P["port_bar"]))

    def instruments(self) -> dict:
        """The tank's instrument readings (the practice call instruments()): the tracking system's samples of the
        vehicle's body frame in the tank frame since the previous call (5 Hz: t, x, y, z, qw, qx, qy, qz, then the six
        handle encoders in deg), the encoders and lamps now, and the set points."""
        new = self.track[self._track_from:]
        first = self._track_from
        self._track_from = len(self.track)
        ang = self.bank.angles_deg()
        return dict(t=round(float(self.sim.time), 6), frame="tank: origin on the floor at the tank centre, x along "
                    "the flow, z up", rate_hz=TRACK_RATE_HZ, columns=self.track_columns(), first_index=first,
                    samples=[list(r) for r in new], encoders_deg={n: round(float(ang[n]), ENCODER_DEC) + 0.0
                                                                  for n in self.valve_names},
                    lamps=self.lamps(), setpoints=self.setpoints())

    def track_columns(self):
        return ["t", "x", "y", "z", "qw", "qx", "qy", "qz"] + [f"{n}_deg" for n in self.valve_names]

    def instrument_log(self) -> dict:
        """The whole run's instrument record (sent with the end of the run and kept in the tank's records)."""
        return dict(frame="tank", rate_hz=TRACK_RATE_HZ, columns=self.track_columns(),
                    samples=[list(r) for r in self.track], lamps=self.lamps(), setpoints=self.setpoints(),
                    run_settings=copy.deepcopy(self.site.setup))


class SisterWorld(FacilityWorld):
    """The sister site: a pipeline end termination (PLET) on open sea bed at 48 m, its two-valve panel (BV1 on the
    pipeline side, PV1 on the export side, TP-1 on the spool between them, PARK-A with the hot stab), one fixed seep at
    the spool flange, the cage 20 m off.  Tide, water, seep, clutter and the spool's pressure are drawn for every run."""

    kind = "sister"

    def _theta0(self, v):
        return 0.0

    def _make_flow(self):
        f = self.P["flow"]
        fs = FL.FlowSpec(U2=f["U2"], theta=f["theta"], t_slack=f["t_slack"], minor_ratio=f["minor_ratio"],
                         f_inside=f["f_inside"], gm_coef=f["gm_coef"], gm_tau=f["gm_tau"], gm_seed=int(f["gm_seed"]))
        st = self.sd["structure"]
        env = FL.StructureEnvelope(lx=float(st["lx"]), ly=float(st["ly"]), roof_z=float(st["roof_z"]),
                                   mudmat_thk=float(st["mudmat_thk"]), ramp_m=float(st["ramp_m"]))
        return FL.StationFlow(fs, FL.SitePose(self.sd["pose_o"], self.sd["pose_psi"]), self.seabed_depth, env)

    def _make_process(self):
        p = self.P["process"]
        return PletProcess(p["p0_bar"], p["tau_s"])

    def _make_plume(self):
        p = self.P["plume"]
        src = self.pose.to_w(np.asarray(self.sd["leak_S"], float))
        ps = PL.PlumeSpec(src_w=tuple(float(x) for x in src), rate=float(p["rate"]), w_b=float(p["w_b"]),
                          seed=int(p["seed"]))
        c = self.P["clutter"]
        cs = PL.ClutterSpec(n=int(c["n"]), radius=tuple(c["radius"]), ts_db=tuple(c["ts_db"]), z_band=tuple(c["z_band"]),
                            domain=float(c["domain"]), seed=int(c["seed"])) if int(c["n"]) > 0 else None
        return PL.Plume(ps, self.flow, cs)

    def _build_static(self, s):
        # the PLET description's frame is S: place it with the structure pose (origin on the mudmat top)
        out = {}
        wb = s.worldbody
        R, o = self.pose.R, self.pose.o
        for pr in self.sd["description"]["primitives"]:
            if pr.get("drop"):
                continue
            kw = _static_kw(pr)
            if pr["role"] == "mudmat":
                kw.update(solref=list(OB.DEBRIS_CONTACT["solref"]), solimp=list(OB.DEBRIS_CONTACT["solimp"]))
            if pr["kind"] == "box":
                A = np.eye(3) if pr.get("axes") is None else np.asarray(pr["axes"], float).T
                wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(pr["half"]), pos=list(self.pose.to_w(pr["centre"])),
                            quat=list(SITE.quat_of(R @ A)), **kw)
            else:
                gt = mujoco.mjtGeom.mjGEOM_CAPSULE if pr["kind"] == "capsule" else mujoco.mjtGeom.mjGEOM_CYLINDER
                wb.add_geom(type=gt, size=[pr["radius"], 0, 0],
                            fromto=list(self.pose.to_w(pr["a"])) + list(self.pose.to_w(pr["b"])), **kw)
            if pr["physics"]:
                out.setdefault(pr["role"], []).append(pr["name"])
        _ = o
        return out


FACTORIES = {"tank": TankWorld, "sister": SisterWorld}


def world_factory(site, values, seed, render):
    """subseacell.server's world_factory: a facility session -> its world; anything else is refused."""
    if not isinstance(site, FacilitySite) or site.kind not in FACTORIES or site.params is None:
        raise GuardError("the practice service builds practice facility sessions only")
    if not str(site.label).startswith(LABEL_ROOT):
        raise GuardError("label refused")
    sizes = tuple(tuple(int(v) for v in s) for s in (values or {}).get("api.camera_sizes", ((640, 480), (320, 240))))
    return FACTORIES[site.kind](site, config=values, cameras=bool(render), sensor_seed=seed, camera_sizes=sizes)
