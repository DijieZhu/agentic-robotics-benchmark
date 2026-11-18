"""Static hydrostatics of a rigid assembly (pure numpy; no MuJoCo).  Shared by the design / assembly graders.

Input: a table of parts, each with mass m_i, centroid c_i, displaced volume V_i and centre of buoyancy b_i (body frame,
metres; default b_i = c_i), optionally an inertia about its own centroid.  Output (for a water density rho):

  mass  M = sum m_i                     displacement  V = sum V_i          buoyancy  rho V      net buoyancy  rho V - M  [kg]
  CG    = sum m_i c_i / M               CB = sum V_i b_i / V               BG = z_CB - z_CG (CB above CG: > 0)
  inertia about the CG = sum (I_i + m_i (|d_i|^2 E - d_i d_i^T)),  d_i = c_i - CG       (dry; no added mass)
  restoring stiffness  W * BG  [N m / rad]  (W = M g)

Static attitude.  All forces of a submerged rigid body at rest are vertical in the world (weight at the CG, buoyancy at
the CB, and any vertical reaction).  With f_k the upward force at body point p_k (sum f_k = 0), the moment is
(R m) x z_W with m = sum f_k p_k, so the equilibrium attitude R makes R m parallel to +z_W (stable) or -z_W (unstable).
For yaw 0 and R = Ry(pitch) Rx(roll) (the project's ZYX convention) this has the closed form

    roll  = atan2(m_y, m_z)               pitch = atan2(-m_x, hypot(m_y, m_z))                 (exact, any angle)

Free floating with W = B: m = B (CB - CG).  Free floating with W != B the vehicle accelerates vertically; the inertial
force acts at the CG, so the attitude is the same as for W = B (m = B (CB - CG)).  A vertical reaction at a known point
(tether, seabed contact, a held object) enters as one more (f, p) pair.  The digest reports the small-angle form
pitch = atan((x_G - x_B) / BG), roll = -atan((y_G - y_B) / BG) (FLU: + pitch = nose down, + roll = port side up); the two
agree to O((dy / BG)^2) (1e-5 relative for the reference vehicle).  Large angles need nothing special while the
vehicle is fully submerged (the CB is fixed in the body); near the surface the CB moves with the waterline and this
module does not apply (the simulator's surface model is a separate, cruder ramp).

Depth-hold trim (task 03's grader form, decision M9): with the four vertical thrusters cancelling the net buoyancy
dB = B - W at their centroid p_T, the static attitude follows from d = B (r_B - p_T) - W (r_G - p_T):
    theta = atan2(-d_x, d_z)   (+ = nose down)        phi = atan2(d_y, d_z)   (+ = port side up)
(static_trim(rows, rho, p_T_mm) in the reported millimetre form, PartTable.depth_hold_trim(rho, p_T_m) in SI).  It is
NOT the free-floating attitude (equilibrium(): the inertial force at the CG); for the reference vehicle the two differ
by a few tenths of a degree.  The roll of both forms is the same function of the moment vector; the pitch here is the
slope angle atan2(-d_x, d_z) where equilibrium() returns the Tait-Bryan angle atan2(-m_x, hypot(m_y, m_z)).
rho_teos10(T, SP, p) gives the density the trim is evaluated in (TEOS-10; needs the 'gsw' package).

Construction helpers: parts_from_manifest(config) builds the part table of a configuration straight from the digest
manifest (optionally with another ballast layout and battery position), lumped_from_spec(config) gives the full-precision
digest totals as one part; PartTable.add / remove / add_ballast / remove_ballast / add_buoyancy / add_payload / move_battery
change the table.

digest v3: a ballast UNIT is a weight + its thread-forming screw (fitted and removed together; uwkit.ballast_unit is the unit);
parts_from_manifest merges the two manifest items of a unit into one Part (category 'ballast'). The battery's axial position in
the 3-inch tube is a trim setting (forward-only travel, spec 'battery_axial_travel'); every digest state carries its offset.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

G_STD = 9.80665


# ------------------------------------------------------------------------------------------------------- parts
@dataclass
class Part:
    name: str
    mass_kg: float
    com_m: np.ndarray
    displacement_m3: float = 0.0
    cb_m: Optional[np.ndarray] = None            # default: the mass centroid
    inertia_com_kgm2: Optional[np.ndarray] = None
    category: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mass_kg = float(self.mass_kg)
        self.displacement_m3 = float(self.displacement_m3)
        self.com_m = np.asarray(self.com_m, float).reshape(3)
        self.cb_m = self.com_m.copy() if self.cb_m is None else np.asarray(self.cb_m, float).reshape(3)
        I = np.zeros((3, 3)) if self.inertia_com_kgm2 is None else np.asarray(self.inertia_com_kgm2, float)
        self.inertia_com_kgm2 = I.reshape(3, 3) if I.size == 9 else np.diag(I.reshape(3))


@dataclass
class Hydrostatics:
    rho: float
    g: float
    mass_kg: float
    displacement_m3: float
    buoyancy_kg: float
    net_buoyancy_kg: float                       # rho V - M (positive = floats)
    cg_m: np.ndarray
    cb_m: np.ndarray
    BG_m: float                                  # z_CB - z_CG
    inertia_cg_kgm2: np.ndarray
    pitch_trim_deg: float                        # digest small-angle form (FLU: + = nose down)
    roll_trim_deg: float                         # (+ = port side up)
    pitch_eq_deg: float                          # exact free-floating equilibrium (yaw 0, ZYX)
    roll_eq_deg: float
    stable: bool                                 # CB above CG
    restoring_N_m_per_rad: float                 # W * BG

    @property
    def net_buoyancy_N(self) -> float:
        return self.net_buoyancy_kg * self.g

    @property
    def weight_N(self) -> float:
        return self.mass_kg * self.g

    def as_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        out["net_buoyancy_N"] = self.net_buoyancy_N
        return out


def equilibrium_attitude(m_vec) -> tuple:
    """(roll, pitch) [rad] that turn the body vector m_vec to +z_W (yaw 0, R = Ry(pitch) Rx(roll)); exact."""
    mx, my, mz = (float(x) for x in np.asarray(m_vec, float).reshape(3))
    return math.atan2(my, mz), math.atan2(-mx, math.hypot(my, mz))


def trim_small_angle(cg, cb) -> tuple:
    """Digest convention: (pitch, roll) [deg] = (atan((xG - xB)/BG), -atan((yG - yB)/BG))."""
    cg, cb = np.asarray(cg, float), np.asarray(cb, float)
    bg = float(cb[2] - cg[2])
    if bg <= 0:
        return float("nan"), float("nan")
    return math.degrees(math.atan2(cg[0] - cb[0], bg)), -math.degrees(math.atan2(cg[1] - cb[1], bg))


def rotation_rp(roll, pitch) -> np.ndarray:
    """R = Ry(pitch) Rx(roll) (yaw 0)."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return Ry @ Rx


class PartTable:
    """An editable list of parts with hydrostatic totals."""

    def __init__(self, parts: Iterable[Part] = (), name: str = ""):
        self.parts = list(parts)
        self.name = name

    def __len__(self):
        return len(self.parts)

    def __iter__(self):
        return iter(self.parts)

    def copy(self) -> "PartTable":
        return PartTable(copy.deepcopy(self.parts), self.name)

    def names(self) -> list:
        return [p.name for p in self.parts]

    def get(self, name) -> Part:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(name)

    # ---------------------------------------------------------------- edit
    def add(self, part: Optional[Part] = None, **kw) -> Part:
        p = part if part is not None else Part(**kw)
        self.parts.append(p)
        return p

    def remove(self, which) -> list:
        """Remove parts by name, list of names or predicate(part) -> bool.  Returns the removed parts."""
        if callable(which):
            pred = which
        elif isinstance(which, str):
            pred = lambda p: p.name == which          # noqa: E731
        else:
            s = set(which)
            pred = lambda p: p.name in s              # noqa: E731
        out = [p for p in self.parts if pred(p)]
        if not out:
            raise KeyError(f"no part matches {which!r}")
        self.parts = [p for p in self.parts if not pred(p)]
        return out

    def select(self, pred: Callable[[Part], bool]) -> "PartTable":
        return PartTable([p for p in self.parts if pred(p)], self.name)

    def add_ballast(self, hole_xy_m, unit: Optional[dict] = None, name: Optional[str] = None) -> Part:
        """One ballast weight centred on a bottom-plate hole (x, y in the body frame).  unit: dict with mass_kg,
        displacement_m3, com_z_B_m, cb_z_B_m, inertia_about_com_B_kgm2 (default: the digest's ballast unit)."""
        u = unit or ballast_unit()
        x, y = (float(v) for v in hole_xy_m[:2])
        z_off = float(hole_xy_m[2]) if len(hole_xy_m) > 2 else 0.0          # optional plate offset (skid plate)
        cr = u.get("com_rel_hole_m", [0.0, 0.0]); br = u.get("cb_rel_hole_m", cr)
        nm = name or (f"ballast@({x * 1e3:.2f},{y * 1e3:.2f})" + (f"/{z_off * 1e3:+.1f}" if z_off else ""))
        return self.add(Part(nm, u["mass_kg"], [x + cr[0], y + cr[1], u["com_z_B_m"] + z_off], u["displacement_m3"],
                             [x + br[0], y + br[1], u["cb_z_B_m"] + z_off], u.get("inertia_about_com_B_kgm2"), "ballast",
                             dict(hole_m=[x, y] + ([z_off] if z_off else []), plate_dz_m=z_off)))

    def remove_ballast(self, hole_xy_m, tol_m=1e-4, unit: Optional[dict] = None) -> Part:
        """Remove the ballast unit on a hole (x, y[, dz]); dz = plate offset (default 0 = the vehicle bottom plate;
        mission_skid: -0.144 m = the skid plate).  Units on the same (x, y) of another plate are not touched."""
        x, y = (float(v) for v in hole_xy_m[:2])
        z_off = float(hole_xy_m[2]) if len(hole_xy_m) > 2 else 0.0
        for p in self.parts:
            if p.category == "ballast" and abs(p.com_m[0] - x) < tol_m and abs(p.com_m[1] - y) < tol_m \
                    and abs(self._plate_offset(p, unit) - z_off) < 0.02:
                self.parts.remove(p)
                return p
        raise KeyError(f"no ballast unit at ({x}, {y}" + (f", {z_off}" if z_off else "") + ")")

    @staticmethod
    def _plate_offset(p: Part, unit: Optional[dict] = None) -> float:
        if "plate_dz_m" in p.meta:
            return float(p.meta["plate_dz_m"])
        if "hole_m" in p.meta and len(p.meta["hole_m"]) > 2:
            return float(p.meta["hole_m"][2])
        try:
            zu = float((unit or ballast_unit())["com_z_B_m"])
        except Exception:                                # no digest at hand: only one plate can be meant
            return 0.0
        dz = float(p.com_m[2]) - zu
        return 0.0 if abs(dz) < 0.02 else dz

    def battery(self) -> Part:
        """The battery pack part (manifest id 'battery' or '<config>.battery')."""
        for p in self.parts:
            if p.name == "battery" or p.name.endswith(".battery"):
                return p
        raise KeyError("no battery part")

    def move_battery(self, dx_m: float) -> Part:
        """Slide the battery pack along the tube axis (+x) by dx_m (mass properties only; the tube envelope displaces the water)."""
        p = self.battery()
        d = np.array([float(dx_m), 0.0, 0.0])
        p.com_m = p.com_m + d
        p.cb_m = p.cb_m + d
        return p

    def ballast_holes(self, with_plate: bool = False) -> list:
        """Hole centres of the ballast units [(x, y)], or [(x, y, dz)] with the plate offset (with_plate=True)."""
        if not with_plate:
            return [(float(p.com_m[0]), float(p.com_m[1])) for p in self.parts if p.category == "ballast"]
        return [(float(p.com_m[0]), float(p.com_m[1]), self._plate_offset(p)) for p in self.parts if p.category == "ballast"]

    def add_buoyancy(self, name, volume_m3, density_kg_m3, centroid_m, inertia_com_kgm2=None) -> Part:
        """A buoyancy block (e.g. syntactic foam): mass = density x volume, displacement = volume."""
        return self.add(Part(name, density_kg_m3 * volume_m3, centroid_m, volume_m3, centroid_m, inertia_com_kgm2,
                             "buoyancy"))

    def add_payload(self, name, mass_kg, volume_m3, centroid_m, cb_m=None, inertia_com_kgm2=None) -> Part:
        return self.add(Part(name, mass_kg, centroid_m, volume_m3, cb_m, inertia_com_kgm2, "payload"))

    # ---------------------------------------------------------------- totals
    def totals(self, rho: float, g: float = G_STD) -> Hydrostatics:
        return compute(self.parts, rho, g)

    def depth_hold_trim(self, rho: float, p_T_m, g: float = G_STD) -> tuple:
        """(dB_N, theta_deg, phi_deg) with the vertical thrusters holding depth at their centroid p_T_m (body frame, m);
        see the module docstring and static_trim()."""
        return static_trim_si(self.parts, rho, p_T_m, g)

    def equilibrium(self, rho: float, reactions: Sequence = (), g: float = G_STD) -> tuple:
        """Exact static (roll, pitch) [rad] with extra vertical forces [(f_up_N, point_B), ...].  If the forces do
        not sum to zero the remainder is an inertial force at the CG (free vertical acceleration)."""
        h = self.totals(rho, g)
        W, B = h.mass_kg * g, rho * h.displacement_m3 * g
        pairs = [(B, h.cb_m), (-W, h.cg_m)] + [(float(f), np.asarray(p, float)) for f, p in reactions]
        rest = -sum(f for f, _ in pairs)
        pairs.append((rest, h.cg_m))
        mvec = sum(f * (p - h.cg_m) for f, p in pairs)
        return equilibrium_attitude(mvec)


def static_trim(rows, rho: float, p_T_mm, g: float = G_STD) -> tuple:
    """Depth-hold static trim (decision M9; task 03 record G3 section 6), millimetre form.

    rows: iterable of (mass_kg, disp_m3, cg_mm[3], cb_mm[3]) in the vehicle frame B (FLU); p_T_mm: centroid of the vertical
    thrusters (mission: vertical_thruster_centroid() = (12.7, 0, -8.7) mm); returns (dB_N, theta_deg, phi_deg) with
        d  = sum_i [rho g V_i (b_i - p_T) - g m_i (c_i - p_T)],   dB = rho g sum V - g sum m
        theta = atan2(-d_x, d_z) (+ bow down),   phi = atan2(d_y, d_z) (+ port up)."""
    M = V = 0.0
    mG = np.zeros(3)
    vB = np.zeros(3)
    for m, v, cg, cb in rows:
        M += m
        V += v
        mG += m * np.asarray(cg, float)
        vB += v * np.asarray(cb, float)
    W = M * g
    B = rho * V * g
    pT = np.asarray(p_T_mm, float)
    d = rho * g * vB - g * mG - (B - W) * pT
    return B - W, math.degrees(math.atan2(-d[0], d[2])), math.degrees(math.atan2(d[1], d[2]))


def static_trim_si(parts: Iterable[Part], rho: float, p_T_m, g: float = G_STD) -> tuple:
    """static_trim() on Part objects (metres); p_T_m in metres.  Returns (dB_N, theta_deg, phi_deg)."""
    rows = [(p.mass_kg, p.displacement_m3, p.com_m * 1e3, p.cb_m * 1e3) for p in parts]
    return static_trim(rows, rho, np.asarray(p_T_m, float) * 1e3, g)


def rho_teos10(T_C: float, SP: float, p_dbar: float = 1.0) -> float:
    """In-situ density [kg/m^3] of sea water (TEOS-10, 'gsw' package): SA_from_SP(SP, p, 0, 0) -> CT_from_t -> rho.
    SP = practical salinity (0 = fresh), T_C = in-situ temperature, p_dbar = sea pressure (1 dbar ~ 1 m)."""
    try:
        import gsw
    except ImportError as e:                                   # pragma: no cover
        raise ImportError("rho_teos10 needs the 'gsw' package (TEOS-10); pip install gsw") from e
    SA = gsw.SA_from_SP(SP, p_dbar, 0.0, 0.0)
    return float(gsw.rho(SA, gsw.CT_from_t(SA, T_C, p_dbar), p_dbar))


def vertical_thruster_centroid(config: str = "mission", spec_path: Optional[str] = None,
                               sites=("thr_5", "thr_6", "thr_7", "thr_8")) -> np.ndarray:
    """Centroid [m, body frame] of the vertical thrusters of a digest configuration (the p_T of the depth-hold trim)."""
    from .refdata import spec
    t = {x["site"]: x for x in spec(f"{config}.thrusters", spec_path)}
    return np.mean([np.asarray(t[s]["pos_m"], float) for s in sites], axis=0)


def compute(parts: Sequence[Part], rho: float, g: float = G_STD) -> Hydrostatics:
    P = list(parts)
    m = np.array([p.mass_kg for p in P])
    V = np.array([p.displacement_m3 for p in P])
    C = np.array([p.com_m for p in P])
    Bc = np.array([p.cb_m for p in P])
    M, Vt = float(m.sum()), float(V.sum())
    if M <= 0:
        raise ValueError("total mass must be positive")
    cg = (m[:, None] * C).sum(0) / M
    cb = (V[:, None] * Bc).sum(0) / Vt if Vt > 0 else cg.copy()
    I = np.zeros((3, 3))
    E = np.eye(3)
    for p in P:
        d = p.com_m - cg
        I += p.inertia_com_kgm2 + p.mass_kg * (float(d @ d) * E - np.outer(d, d))
    bg = float(cb[2] - cg[2])
    pt, rl = trim_small_angle(cg, cb)
    req, peq = equilibrium_attitude(cb - cg)
    return Hydrostatics(rho=float(rho), g=float(g), mass_kg=M, displacement_m3=Vt, buoyancy_kg=float(rho * Vt),
                        net_buoyancy_kg=float(rho * Vt - M), cg_m=cg, cb_m=cb, BG_m=bg, inertia_cg_kgm2=I,
                        pitch_trim_deg=pt, roll_trim_deg=rl, pitch_eq_deg=math.degrees(peq),
                        roll_eq_deg=math.degrees(req), stable=bg > 0, restoring_N_m_per_rad=M * g * bg)


def righting_moment(h: Hydrostatics, angle_rad, axis="roll") -> np.ndarray:
    """Restoring moment [N m] about the CG when the vehicle is held at `angle` (roll or pitch, the other angle at its
    equilibrium value) while fully submerged: B (R (CB - CG)) x z_W, component about the rotation axis.  For the W = B
    vehicle this is -W BG' sin(angle - angle_eq) with BG' = |CB - CG| projected on the plane of rotation."""
    a = np.atleast_1d(np.asarray(angle_rad, float))
    B = h.rho * h.displacement_m3 * h.g
    r = h.cb_m - h.cg_m
    out = []
    for x in a:
        if axis == "roll":
            R = rotation_rp(x, math.radians(h.pitch_eq_deg))
            k = 0
        else:
            R = rotation_rp(math.radians(h.roll_eq_deg), x)
            k = 1
        Mw = np.cross(R @ r, np.array([0.0, 0.0, B]))
        out.append(float((R.T @ Mw)[k]))
    return np.array(out)


def weights_for_neutral(h: Hydrostatics, unit: Optional[dict] = None) -> float:
    """Number of ballast weights (fractional) that brings the net buoyancy to zero in this water."""
    u = unit or ballast_unit()
    wet = u["mass_kg"] - h.rho * u["displacement_m3"]
    return h.net_buoyancy_kg / wet


# ------------------------------------------------------------------------------------------------ from the digest
def ballast_unit(spec_path: Optional[str] = None) -> dict:
    from .refdata import spec
    return dict(spec("uwkit.ballast_unit", spec_path))


def _is_sea(water) -> bool:
    return water in ("sea", "sea_SP35", "sea_SP35_20C")


def ballast_layout(config="mission", water="fresh_20C", spec_path: Optional[str] = None) -> list:
    """Hole centres [(x, y) m] of a ballast layout: the instanced (fresh-water) layout of a configuration, or the
    mission sea-water layout (decisions R5 / L2: the fresh layout + added units). mission_skid holes on the skid's bottom
    plate carry the plate offset as a third element (x, y, dz)."""
    from .refdata import spec
    if _is_sea(water):
        if config != "mission":
            raise ValueError("the digest gives a sea-water ballast layout for 'mission' only")
        holes = spec("uwkit.mission.ballast_sea_SP35", spec_path)["holes_mm"]
        return [(h[0] * 1e-3, h[1] * 1e-3) for h in holes]
    b = spec(f"{config}.ballast", spec_path)
    if "holes_mm" not in b:
        return None                                    # std: the vendor default layout as instanced
    if config == "mission_skid":
        st = {s["station"]: s for s in spec("mission_skid.ballast_stations", spec_path)}
        dz = spec("mission_skid.datum", spec_path)["plates_apart_m"]
        return [(st[k]["x_B_m"], st[k]["y_B_m"]) + ((-dz,) if st[k]["plate"] == "K" else ()) for k in b["stations"]]
    return [(h[0] * 1e-3, h[1] * 1e-3) for h in b["holes_mm"]]


def battery_offset(config="mission", water="fresh_20C", spec_path: Optional[str] = None) -> float:
    """Axial battery position [m, +x from the centred position = the rear stop] of a digest state: the instanced
    (fresh-water) trim of a configuration or the mission sea-water trim."""
    from .refdata import spec
    if _is_sea(water):
        if config != "mission":
            raise ValueError("the digest gives a sea-water trim for 'mission' only")
        return float(spec("uwkit.mission.ballast_sea_SP35", spec_path).get("battery_offset_m", 0.0))
    return float(spec(f"{config}.battery", spec_path, default={"offset_m": 0.0})["offset_m"])


def battery_travel(spec_path: Optional[str] = None) -> tuple:
    """(rear stop, front stop) of the battery offset [m]."""
    from .refdata import spec
    t = spec("battery_axial_travel", spec_path)
    return float(t["rear_stop_offset_m"]), float(t["front_stop_offset_m"])


def _merge(name, parts, category="ballast", meta=None) -> Part:
    M = sum(p.mass_kg for p in parts); V = sum(p.displacement_m3 for p in parts)
    c = sum(p.mass_kg * p.com_m for p in parts) / M
    b = sum(p.displacement_m3 * p.cb_m for p in parts) / V if V > 0 else c
    I = np.zeros((3, 3))
    for p in parts:
        d = p.com_m - c
        I += p.inertia_com_kgm2 + p.mass_kg * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return Part(name, M, c, V, b, I, category, dict(meta or {}))


def parts_from_manifest(config="mission", ballast=None, manifest: Optional[dict] = None,
                        spec_path: Optional[str] = None, include_zero=False, battery_offset_m: Optional[float] = None) -> PartTable:
    """Part table of a configuration from the digest manifest (instances + derived items: sealed envelopes,
    calibration lumps, removed blanks); each ballast unit (weight + screw, same `ballast_station`) is one Part.
    ballast: None (as instanced), 'sea_SP35' (mission sea layout and its battery position), or a list of hole centres
    [(x, y) m] that replaces the instanced units.  battery_offset_m: absolute battery offset [m] (None: the digest value of the
    chosen state; a custom hole list keeps the instanced battery)."""
    from .refdata import manifest_items
    items = manifest_items(config, manifest)
    parts, units = [], {}
    for it in items:
        m, V = float(it["mass_kg"]), float(it["displacement_m3"])
        if not include_zero and m == 0.0 and V == 0.0:
            continue
        p = Part(it["id"], m, it["com_B_m"], V, it["cb_B_m"], it.get("inertia_about_com_B_kgm2"),
                 it["category"], dict(generic_name=it.get("generic_name"), location=it.get("location")))
        if it.get("ballast_station"):
            units.setdefault(it["ballast_station"], []).append(p)
        else:
            parts.append(p)
    for st, ps in units.items():
        ps.sort(key=lambda p: p.category != "ballast")                # the weight first: its id names the unit
        parts.append(_merge(ps[0].name, ps, "ballast", dict(station=st, members=[p.name for p in ps])))
    t = PartTable(parts, f"{config} (manifest)")
    d_inst = battery_offset(config, "fresh_20C", spec_path)
    if ballast is not None:
        holes = ballast_layout(config, "sea_SP35", spec_path) if isinstance(ballast, str) else ballast
        t.remove(lambda p: p.category == "ballast")
        for h in holes:
            t.add_ballast(h, ballast_unit(spec_path))
        if battery_offset_m is None and isinstance(ballast, str):
            battery_offset_m = battery_offset(config, "sea_SP35", spec_path)
    if battery_offset_m is not None and abs(battery_offset_m - d_inst) > 0.0:
        t.move_battery(battery_offset_m - d_inst)
    return t


def lumped_from_spec(config="mission", water="fresh_20C", spec_path: Optional[str] = None) -> Part:
    """The digest's full-precision totals of a configuration (instanced ballast) as a single part."""
    from .refdata import spec
    s = spec(f"uwkit.{config}.{water}", spec_path)
    return Part(f"{config}_total", s["mass_kg"], s["cg_B_m"], s["displacement_m3"], s["cb_B_m"],
                s["inertia_about_cg_B_kgm2"], "vehicle")
