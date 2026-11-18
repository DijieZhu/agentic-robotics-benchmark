"""Parameter containers and loaders (vehicle spec, hydrodynamic coefficient sets, water, ranges).

Units are SI throughout.  Body frame B = FLU with origin at the CAD origin (see frames.py).
The reference vehicle's numbers come from the digest (spec_values.json + manifest, see refdata.py and
models/vehicle.py: build_vehicle / reference_vehicle_spec).  models/vehicle_placeholder.json is the pre-digest
box vehicle (kept for the analytic tests and the old-vs-new comparison).
Hydrodynamic coefficient sets and the hidden-world ranges are grader / hidden-world data and live in
uwkit/hidden/ (never ship that directory to an agent sandbox); they are addressed by file stem, set A / set B
= the two endpoints listed in hidden/hidden_world_ranges.json.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

G = 9.80665               # standard gravity, used by MuJoCo models built by uwkit (= spec_values gravity) [m/s^2]
RHO_FRESH = 1000.0        # reference density of the literature coefficient sets [kg/m^3]

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_HERE, "models")
HIDDEN_DIR = os.environ.get("UWKIT_HIDDEN_DIR") or os.path.join(_HERE, "hidden")
SPEC_VALUES_DEFAULT = os.path.normpath(os.path.join(_HERE, "..", "spec_values.json"))


# --------------------------------------------------------------------------- dataclasses
@dataclass
class ThrusterSpec:
    name: str
    pos: np.ndarray          # nozzle centre in B [m]
    dir: np.ndarray          # unit vector in B: force on the vehicle for a POSITIVE command

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float).reshape(3)
        d = np.asarray(self.dir, float).reshape(3)
        self.dir = d / np.linalg.norm(d)


@dataclass
class ThrusterModelSpec:
    pwm_center_us: float = 1500.0
    pwm_half_span_us: float = 400.0
    tau_up_s: float = 0.04             # effective-command lag while spinning up
    tau_down_s: float = 0.06           # ... while spinning down / reversing
    inflow_loss: bool = True           # advance-ratio thrust loss on/off
    geometric_pitch_m: float = 0.0742  # P_g: thrust -> 0 when axial inflow = ratio * P_g * n
    inflow_zero_ratio: float = 1.0
    inflow_factor_max: float = 1.25    # cap for J < 0 (vehicle moving against the jet)


@dataclass
class BatterySpec:
    series_cells: int = 4
    capacity_Ah: float = 18.0
    r0_ohm: float = 0.020
    r1_ohm: float = 0.010
    tau1_s: float = 30.0
    ocv_soc: tuple = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)
    ocv_cell_V: tuple = (3.00, 3.20, 3.33, 3.47, 3.55, 3.61, 3.67, 3.74, 3.81, 3.89, 3.98, 4.06, 4.18)
    i_continuous_A: float = 60.0
    i_burst_A: float = 132.0
    burst_s: float = 10.0
    v_cutoff_V: float = 12.0
    v_recover_V: float = 12.5


@dataclass
class HydroCoeffs:
    """Diagonal Fossen coefficients of the main hull about its CG (positive magnitudes)."""
    name: str
    added_mass: np.ndarray
    damping_linear: np.ndarray
    damping_quadratic: np.ndarray
    rho_ref: float = RHO_FRESH

    def __post_init__(self):
        for k in ("added_mass", "damping_linear", "damping_quadratic"):
            v = np.asarray(getattr(self, k), float).reshape(6)
            if np.any(v < 0):
                raise ValueError(f"{k} must be non-negative magnitudes, got {v}")
            setattr(self, k, v)

    def to_dict(self):
        return dict(name=self.name, added_mass=self.added_mass.tolist(),
                    damping_linear=self.damping_linear.tolist(),
                    damping_quadratic=self.damping_quadratic.tolist(), rho_ref=self.rho_ref)


@dataclass
class VehicleSpec:
    name: str
    mass_kg: float
    com_m: np.ndarray
    inertia_com_kgm2: np.ndarray
    volume_m3: float
    cob_m: np.ndarray
    half_height_m: float
    thrusters: list
    thruster_model: ThrusterModelSpec
    collision_boxes: list
    sites: dict
    camera_tilt: dict
    gripper: dict
    battery: BatterySpec
    hotel_W: dict
    site_quats: dict = field(default_factory=dict)      # site name -> (w, x, y, z) in B (reference model)
    sources: dict = field(default_factory=dict)

    def __post_init__(self):
        self.com_m = np.asarray(self.com_m, float).reshape(3)
        self.cob_m = np.asarray(self.cob_m, float).reshape(3)
        I = np.asarray(self.inertia_com_kgm2, float)
        self.inertia_com_kgm2 = I.reshape(3, 3) if I.size == 9 else np.diag(I.reshape(3))

    @property
    def n_thrusters(self):
        return len(self.thrusters)

    def thruster_positions(self):
        return np.array([t.pos for t in self.thrusters])

    def thruster_dirs(self):
        return np.array([t.dir for t in self.thrusters])

    def allocation_matrix(self, ref_point=None):
        """6 x n map from per-thruster force [N] to the body wrench [F; M] about ref_point (default CG)."""
        r0 = self.com_m if ref_point is None else np.asarray(ref_point, float)
        P, D = self.thruster_positions(), self.thruster_dirs()
        return np.vstack([D.T, np.cross(P - r0, D).T])

    def weight_N(self, g=G):
        return self.mass_kg * g

    def buoyancy_N(self, rho=RHO_FRESH, g=G):
        return rho * self.volume_m3 * g


@dataclass
class Water:
    """Water properties.  rho may vary with depth via a (depth_m, rho) table."""
    rho: float = 1000.0
    g: float = G
    rho_profile: Optional[tuple] = None    # ((d0, d1, ...), (rho0, rho1, ...)), depth >= 0 down

    def rho_at_depth(self, depth):
        if self.rho_profile is None:
            return np.full(np.shape(depth), self.rho) if np.ndim(depth) else self.rho
        d, r = self.rho_profile
        return np.interp(depth, d, r)


# --------------------------------------------------------------------------- loaders
def _read_json(path):
    with open(path) as f:
        return json.load(f)


def hydro_sets() -> tuple:
    """Names (file stems) of the two literature coefficient sets: (set A, set B) = the hidden-ranges endpoints."""
    return tuple(load_hidden_world_ranges()["endpoints"])


def default_hydro() -> "HydroCoeffs":
    """Nominal coefficient set used when a model is built without an explicit set (set A)."""
    return load_hydro(hydro_sets()[0])


def load_hydro(name_or_path=None) -> HydroCoeffs:
    """Load a hydrodynamic coefficient set by name (hidden/hydro_<name>.json), 'A' / 'B', or a path."""
    if name_or_path is None or name_or_path in ("A", "B"):
        name_or_path = hydro_sets()[0 if name_or_path in (None, "A") else 1]
    path = name_or_path if os.path.sep in str(name_or_path) or str(name_or_path).endswith(".json") \
        else os.path.join(HIDDEN_DIR, f"hydro_{name_or_path}.json")
    d = _read_json(path)
    return HydroCoeffs(name=d.get("name", os.path.basename(path)), added_mass=d["added_mass"],
                       damping_linear=d["damping_linear"], damping_quadratic=d["damping_quadratic"],
                       rho_ref=float(d.get("rho_ref", RHO_FRESH)))


def load_hidden_world_ranges(path=None):
    return _read_json(path or os.path.join(HIDDEN_DIR, "hidden_world_ranges.json"))


def blend_hydro(lam_A, lam_D, a=None, b=None, name="blend") -> HydroCoeffs:
    """Coefficients between set a (lam = 0) and set b (lam = 1); quadratic damping in log space."""
    a = a or load_hydro("A")
    b = b or load_hydro("B")
    A = (1 - lam_A) * a.added_mass + lam_A * b.added_mass
    Dl = (1 - lam_D) * a.damping_linear + lam_D * b.damping_linear
    qa, qb = np.maximum(a.damping_quadratic, 1e-12), np.maximum(b.damping_quadratic, 1e-12)
    Dq = np.exp((1 - lam_D) * np.log(qa) + lam_D * np.log(qb))
    return HydroCoeffs(name=name, added_mass=A, damping_linear=Dl, damping_quadratic=Dq,
                       rho_ref=0.5 * (a.rho_ref + b.rho_ref))


def sample_hydro(rng: np.random.Generator, ranges=None) -> tuple:
    """One hidden-world draw following hidden_world_ranges.json.  Returns (HydroCoeffs, draw dict)."""
    r = ranges or load_hidden_world_ranges()
    lam_A, lam_D = float(rng.uniform()), float(rng.uniform())
    h = blend_hydro(lam_A, lam_D, name="hidden_draw")
    jit = {}
    for key in ("added_mass", "damping_linear", "damping_quadratic"):
        j = float(r[key]["jitter"])
        f = np.exp(rng.uniform(-j, j, size=6))
        setattr(h, key, getattr(h, key) * f)
        jit[key] = f.tolist()
    return h, dict(lam_A=lam_A, lam_D=lam_D, jitter=jit, coeffs=h.to_dict())


def _merge(base: dict, override: dict, sources: dict, origin: str):
    for k, v in override.items():
        if k in base and k not in ("name", "status", "frame"):
            base[k] = copy.deepcopy(v)
            sources[k] = origin


def _spec_values_vehicle_section(sv: dict):
    """The digest lane is free to choose its layout; we accept the uwkit keys either in a dedicated
    section ('uwkit_vehicle' preferred, or 'vehicle') or at top level."""
    for key in ("uwkit_vehicle", "vehicle", "reference_vehicle"):
        if isinstance(sv.get(key), dict):
            return sv[key], key
    return sv, "<top level>"


def load_vehicle_spec(config="placeholder", spec_values=None) -> VehicleSpec:
    """Legacy loader: models/vehicle_<config>.json, optionally overridden by a dict / file with the uwkit keys
    (section 'uwkit_vehicle', 'vehicle' or top level).  The reference vehicle is built from the digest by
    uwkit.models.vehicle.reference_vehicle_spec(); this loader only serves the placeholder vehicle.

    spec_values: None (file only), a dict, or a path.
    """
    if config == "default":
        config = "placeholder"
    base = _read_json(os.path.join(MODELS_DIR, f"vehicle_{config}.json"))
    sources = {k: f"models/vehicle_{config}.json" for k in base}
    sv = None
    if isinstance(spec_values, dict):
        sv, origin = spec_values, "<dict>"
    elif spec_values and spec_values != "auto":
        sv, origin = _read_json(spec_values), str(spec_values)
    if sv is not None:
        sec, where = _spec_values_vehicle_section(sv)
        _merge(base, sec, sources, f"{origin}:{where}")
    thr = [ThrusterSpec(name=t["name"], pos=t["pos_m"], dir=t["dir"]) for t in base["thrusters"]]
    return VehicleSpec(
        name=base.get("name", config), mass_kg=float(base["mass_kg"]), com_m=base["com_m"],
        inertia_com_kgm2=base["inertia_com_kgm2"], volume_m3=float(base["volume_m3"]),
        cob_m=base["cob_m"], half_height_m=float(base["half_height_m"]), thrusters=thr,
        thruster_model=ThrusterModelSpec(**base["thruster_model"]),
        collision_boxes=base["collision_boxes"], sites=base["sites"], camera_tilt=base["camera_tilt"],
        gripper=base["gripper"], battery=BatterySpec(**{k: (tuple(v) if isinstance(v, list) else v)
                                                        for k, v in base["battery"].items()}),
        hotel_W=base["hotel_W"], sources=sources)


def literature_test_vehicle(hydro=None, z_bg=None, buoyancy_ratio=1.0, com_at_origin=True,
                            inertia=None, mass=None, thrusters_from: Optional[VehicleSpec] = None):
    """Single-box vehicle matching a literature parameter set (used by the analytic tests).

    The CoM sits at the body origin (com_at_origin=True) like the R4 spike; thruster geometry is
    copied from the reference mission vehicle (shifted so that it keeps its position relative to the CoM)."""
    hydro = hydro or hydro_sets()[0]
    h = load_hydro(hydro)
    meta = _read_json(os.path.join(HIDDEN_DIR, f"hydro_{hydro}.json"))["identified_vehicle"]
    m = float(mass if mass is not None else meta["mass_kg"])
    I = np.diag(inertia if inertia is not None else meta["inertia_diag_kgm2"])
    zbg = float(z_bg if z_bg is not None else meta["z_bg_m"])
    if thrusters_from is None:
        from .models.vehicle import reference_vehicle_spec
        thrusters_from = reference_vehicle_spec("mission")[0]
    base = thrusters_from
    shift = base.com_m if com_at_origin else np.zeros(3)
    thr = [ThrusterSpec(t.name, t.pos - shift, t.dir) for t in base.thrusters]
    com = np.zeros(3) if com_at_origin else base.com_m.copy()
    spec = VehicleSpec(
        name=f"literature_{h.name}", mass_kg=m, com_m=com, inertia_com_kgm2=I,
        volume_m3=buoyancy_ratio * m / RHO_FRESH, cob_m=com + np.array([0.0, 0.0, zbg]),
        half_height_m=0.127, thrusters=thr, thruster_model=copy.deepcopy(base.thruster_model),
        collision_boxes=[{"name": "frame", "pos_m": list(com), "half_m": [0.2285, 0.2875, 0.127]}],
        sites=dict(base.sites), camera_tilt={}, gripper={}, battery=copy.deepcopy(base.battery),
        hotel_W=dict(base.hotel_W), sources={"*": f"literature_test_vehicle({h.name})"})
    return spec, h


def wet_weight_N(mass_kg, volume_m3, rho=RHO_FRESH, g=G):
    return (mass_kg - rho * volume_m3) * g


def roll_period_estimate(spec: VehicleSpec, hydro: HydroCoeffs, rho=RHO_FRESH, g=G):
    """Small-angle undamped roll / pitch periods [s] from W*BG (fully submerged)."""
    bg = float(spec.cob_m[2] - spec.com_m[2])
    k = rho * spec.volume_m3 * g * bg
    if k <= 0:
        return math.inf, math.inf
    Ix = spec.inertia_com_kgm2[0, 0] + hydro.added_mass[3]
    Iy = spec.inertia_com_kgm2[1, 1] + hydro.added_mass[4]
    return 2 * math.pi * math.sqrt(Ix / k), 2 * math.pi * math.sqrt(Iy / k)
