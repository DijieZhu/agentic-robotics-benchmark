"""evaluation_basis -- the published evaluation procedure of tender specification TS-01 rev C.

This module is the exact, executable form of the calculations that TS-01 Part A (site duties and policy), Part C
(evaluation) and Part D (factory acceptance) refer to.  The same file is used by whoever evaluates the tenders.

It contains formulas, the Part A policy constants (POLICY), the closed dictionary of facility quantities of Part D
(FACILITY_QUANTITIES) and the equation of state of pure water.  It contains no site data, no catalogue page values
and no facts about any tendered unit: every site condition, every page value and every fact about a unit is an
argument.  It is not a checker: without the site conditions and the facts of a unit it computes nothing.

Conventions
  * SI units with the unit in the name: _kg, _l (litre), _n (newton), _m, _mm, _deg, _c (degC), _v, _a, _w, _wh,
    _ohm, _mohm, _bar, _min, _mps (m/s), _m3s (m3/s), _pct, _frac (0..1), _msw (metres of sea water), _kgf (as
    printed on thruster pages), _rpm, _nm (N m), _kg_m3 (kg/m3).
  * A margin is signed: >= 0 meets the need, < 0 does not.  Ratios carry their own name (see site_values).
  * No default values.  Every key of a record and every argument must be given; a missing key raises KeyError and
    an unknown enumeration value raises ValueError.  Extra keys in a record are ignored.
  * Deterministic and machine-independent: elementary operations are IEEE double (+, -, x, /, sqrt, element-wise
    numpy arithmetic and comparisons); the transcendental functions (sin, cos, tan, asin and x ** y) are evaluated
    to 56 significant digits in decimal arithmetic and then rounded once to double, so each result is the correctly
    rounded value and does not depend on the machine or its maths library; table look-ups use the written-out
    linear interpolation interp(); the one reduction (the energy integral) is math.fsum.  No file, network or
    clock access.
  * The policy constants are read from POLICY at call time.

Records (plain dicts; the key lists are THRUSTER_RECORD_KEYS, PACK_RECORD_KEYS, VEHICLE_RECORD_KEYS,
SITE_RECORD_KEYS)

  thruster page (full-throttle table by bus voltage, as printed on the thruster datasheet):
    supply_v            list of the bus-voltage columns [V], increasing
    thrust_kgf          full-throttle forward thrust of one thruster in each column [kgf]
    current_a           full-throttle current of one thruster in each column [A]
    speed_rpm           full-throttle shaft speed in each column [rpm]
    reverse_ratio       reverse / forward thrust at the same throttle [1] (one constant per page)
    thrust_power_exp    part-throttle law: T = T_full(V) x (I / I_full(V)) ** thrust_power_exp [1]
    prop_pitch_m        geometric propeller pitch [m] (inflow loss)
    power_at_1kgf_w     electrical power of one thruster producing 1 kgf in the 12 V column [W];
                        power at thrust t [kgf] = power_at_1kgf_w x t ** (1 / thrust_power_exp)

  battery pack page (one cell lot; temperature tables on the printed nodes):
    capacity_ah         nominal capacity [Ah]
    capacity_factor     delivered capacity of the lot relative to nominal at the reference temperature [1]
    retention_temp_c    temperature nodes of the capacity-retention table [degC], increasing
    retention_frac      capacity retention at those nodes [1]
    r_ss_23c_ohm        steady-state DC resistance of the pack (after 60 s at load) at 23 degC [Ohm]
    r_factor_temp_c     temperature nodes of the resistance-factor table [degC], increasing
    r_factor            steady-state resistance at the node / resistance at 23 degC [1]
    continuous_current_a  continuous current rating of the pack [A]
    ocv_soc_frac        state-of-charge nodes of the open-circuit-voltage table [1], increasing
    ocv_v               open-circuit pack voltage at those nodes, at ocv_ref_temp_c [V]
    ocv_ref_temp_c      temperature of the OCV table [degC]
    ocv_drift_v_per_c   drop of the pack OCV per degC below ocv_ref_temp_c [V/degC]

  vehicle (one unit in one state: as tendered, as delivered, or as on a test day):
    thruster                    thruster page record of the four horizontal thrusters
    front_thruster_angle_deg    angle between a front horizontal thruster axis and the vehicle x axis [deg]
    rear_thruster_angle_deg     angle between a rear horizontal thruster axis and the vehicle x axis [deg]
    thrust_system_coeff         system coefficient of the axis thrust of this unit in this state [1]
    surge_drag_lin_n_per_mps    surge drag, linear coefficient, without tether [N s/m]
    surge_drag_quad_n_per_mps2  surge drag, quadratic coefficient, without tether [N s2/m2]
    pack                        battery pack page record of the pack fitted
    current_preset_a            factory battery-current preset (firmware limit) [A]
    hotel_load_w                hotel electrical load with the lights at 100 % [W]
    lights_load_w               part of hotel_load_w drawn by the lights at 100 % [W]
    camera_port                 "flat" or "dome"
    camera_hfov_air_deg         horizontal field of view of camera + lens in air [deg]
    pressure_part_ratings_msw   rated depth of every pressure part as built [msw]
    air_mass_kg                 air mass [kg]
    displaced_volume_l          displaced volume [L]
    tank_water_density_kg_m3    density of the water of the trim tank the unit was weighed in [kg/m3]
    ballast_piece_mass_kg       air mass of one ballast piece [kg]
    ballast_piece_volume_l      displaced volume of one ballast piece [L]
    removable_ballast_pieces    removable ballast pieces fitted (0 when the ballast is not removable)
    spare_ballast_pieces        loose spare ballast pieces supplied with the unit
    bg_m                        vertical distance from the centre of buoyancy down to the centre of gravity [m]
    static_pitch_trim_deg       static pitch trim, nose down positive [deg]
    active_pitch_moment_nm      pitch moment the thrusters can add at the 12 V column [N m]
    jaw_reach_m                 forward distance from the centre of gravity to the gripper jaw tip [m]
    tether_length_m             tether length [m]
    tether_diameter_mm          tether diameter [mm]
    wet_part_min_temps_c        lowest operating water temperature of every wet part [degC]
    link_test_length_m          tether length over which the data link is tested [m]
    sonar_min_range_m           minimum range of the imaging sonar [m]
    dvl_min_altitude_m          minimum bottom-lock altitude of the velocity log [m]
    gripper_opening_mm          gripper jaw opening [mm]
    depth_sensor_range_bar      full-scale pressure of the depth sensor [bar]

  site (the conditions of the inspection, each as defined by TS-01 Part A):
    water_temp_c        bottom water temperature of the inspection-month cast at the intake station [degC]
                        (not rounded; density and both battery rows use it)
    working_depth_m     design water level minus the lowest point of the inspection scope [m]
    launch_path_m       tether length from the payout counter zero along the payout route to the lowest point of
                        the inspection scope [m]
    face_reach_m        dam-face reach on one side [m]
    design_flow_m3s     design discharge, rounded up to 0.5 m3/s (policy_design_flow_m3s) [m3/s]
    rack_width_m        width of the trash rack [m]
    rack_height_m       height of the trash rack scanned [m]
    open_height_m       height of the rack open to the flow [m]
    secchi_m            Secchi statistic, rounded down to 0.1 m (policy_secchi_m) [m]
    min_water_temp_c    lowest water temperature of the policy, rounded down to 0.1 degC (policy_min_water_temp_c)
    crane_swl_kg        safe working load of the Owner's lifting appliance [kg]
    rigging_kg          mass of the lifting gear [kg]
    survey_altitude_m   survey altitude above the bottom [m]
    sample_max_mm       largest sample to be retrieved [mm]

Needs and functions (rows returns all twelve margins; site_values returns the eight `site` keys of evaluation.json)
  N-01 site trim         n01_trim_reachable (Part C) / n01_as_delivered (Part D); margin [kg] to the nearest edge
                         of the net-buoyancy window [buoy_lo, buoy_hi] x air mass in the site water
  N-02 depth rating      n02_depth_margin_m
  N-03 station keeping   n03_station_thrust: astern thrust at the end of the dive, facing the rack against the
                         design flow, / (vehicle drag + tether drag over the open height) - thrust_factor
  N-04 dive energy       n04_energy_reserve: (usable - needed) / usable for one dive
  N-05 tether reach      n05_tether_margin_m
  N-06 pitch with sample n06_pitch_authority
  N-07 wet temperature   n07_wet_temperature_margin_c
  N-08 launch mass       n08_launch_mass_margin_kg
  N-09 .. N-12           n09_near_structure_margin_m, n10_link_test_margin_m, n11_gripper_opening_margin_mm,
                         n12_depth_range_margin_m (pass / fail forms: k4_near_structure .. k7_depth_range)

Fixed forms
  * Thruster page: linear interpolation in the voltage columns, clamped at the end columns.  Part throttle follows
    the page's thrust-power law.  Reverse-running thrusters give reverse_ratio x the forward thrust.  The shaft
    speed scales with the square root of the thrust.  Inflow loss: 1 - v / (prop_pitch_m x rpm / 60), floored at 0,
    with v the whole on-coming flow speed (not multiplied by the cosine of the thruster angle).
  * Battery: OCV(SoC) from the table, minus the temperature drift; capacity = capacity_ah x capacity_factor x
    retention(T); steady-state resistance R(T) = r_ss_23c_ohm x r_factor(T).  Temperature tables are interpolated
    linearly between the printed nodes and clamped outside them.  The battery current is the smallest of the
    continuous rating, the factory preset and the current that pulls the bus down to v_min; ties go to the first of
    these three in that order.
  * Density: pure water, no salinity correction; the site density is taken at the working depth with the pressure
    as 1 dbar per metre.
  * The three policy roundings are exact decimal roundings of the value first rounded to 0.000001.
  * interp(x, xp, fp): xp increasing; x below xp[0] gives fp[0], x at or above xp[-1] gives fp[-1], x on a node
    gives that node's value, otherwise slope x (x - xp[j]) + fp[j] with slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j]).
"""
from __future__ import annotations

import bisect
import math
import numbers
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Context, Decimal, localcontext

import numpy as np

G = 9.80665                      # standard gravity [m/s2]
BASIS_VERSION = "evaluation_basis 1.0 (TS-01 rev C)"

# ---------------------------------------------------------------- TS-01 Part A policy constants
POLICY = {
    "depth_factor": 1.25,           # rated depth >= depth_factor x working depth (N-02)
    "msw_rho": 1025.0,              # density behind the 'msw' unit of depth ratings [kg/m3]
    "thrust_factor": 1.25,          # available thrust at the rack >= thrust_factor x drag (N-03)
    "reserve": 0.2,                 # state of charge left at the end of the dive (N-03, N-04)
    "v_min": 12.0,                  # bus voltage floor [V] (N-03, N-04)
    "slack": 1.1,                   # tether length >= slack x path (N-05)
    "standoff_per_secchi": 0.4,     # inspection standoff = standoff_per_secchi x Secchi statistic
    "overlap": 0.2,                 # overlap of neighbouring video strips
    "buoy_lo": -0.0005,                 # net-buoyancy window in the site water, lower edge, fraction of the air mass
    "buoy_hi": 0.0095,                # net-buoyancy window in the site water, upper edge, fraction of the air mass
    "pitch_lim_deg": 5.0,           # pitch from horizontal with the sample held (N-06)
    "sample_kg": 1.5,               # sample weight in water, expressed in kg (N-06)
    "cd_tether": 1.2,               # in-service drag coefficient of the tether
    "contingency_s": 120.0,          # contingency time at the rack [s]
    "scan_mps": 0.05,               # strip scan speed [m/s]
    "strip_turn_s": 30.0,           # time per strip change [s]
    "transit_mps": 0.5,             # transit speed, vertical and along the dam face [m/s]
    "transit_vert_extra_w": 60.0,   # extra electrical power while transiting vertically [W]
    "transit_face_extra_w": 30.0,   # extra electrical power while transiting along the dam face [W]
    "rack_extra_w": 6.0,            # extra electrical load at the rack [W]
    "eta_dist": 0.97,               # power-distribution efficiency applied to the usable pack energy
    "n_flat_port": 1.333,           # refractive index used for flat camera ports
}

# ---------------------------------------------------------------- TS-01 Part D facility quantities
# quantity -> (unit, condition names); every condition must be given; comparators are fixed by TS-01 Part D
FACILITY_QUANTITIES = {
    "air_mass_kg": ("kg", ()),
    "tank_net_buoyancy_n": ("N", ()),
    "displaced_volume_l": ("L", ()),
    "bollard_thrust_n": ("N", ("axis", "supply_v", "limiter")),
    "bollard_bus_current_a": ("A", ("axis", "supply_v", "limiter")),
    "battery_delivered_energy_wh": ("Wh", ("load_w", "chamber_c", "cutoff_v")),
    "pack_dcir_mohm": ("mOhm", ("chamber_c",)),
    "hotel_power_w": ("W", ("lights_pct",)),
    "proof_pressure_bar": ("bar", ("hold_min",)),
    "tow_drag_n": ("N", ("speed_mps",)),
    "tether_length_m": ("m", ()),
    "tether_diameter_mm": ("mm", ()),
    "camera_hfov_deg": ("deg", ("medium",)),
    "sonar_min_range_m": ("m", ()),
    "gripper_opening_mm": ("mm", ()),
}
AXES = ("surge_ahead", "surge_astern")
LIMITER = ("on", "off")
MEDIUM = ("air", "tank_water")
CAMERA_PORTS = ("flat", "dome")

THRUSTER_RECORD_KEYS = ("supply_v", "thrust_kgf", "current_a", "speed_rpm", "reverse_ratio", "thrust_power_exp",
                        "prop_pitch_m", "power_at_1kgf_w")
PACK_RECORD_KEYS = ("capacity_ah", "capacity_factor", "retention_temp_c", "retention_frac", "r_ss_23c_ohm",
                    "r_factor_temp_c", "r_factor", "continuous_current_a", "ocv_soc_frac", "ocv_v", "ocv_ref_temp_c",
                    "ocv_drift_v_per_c")
VEHICLE_RECORD_KEYS = ("thruster", "front_thruster_angle_deg", "rear_thruster_angle_deg", "thrust_system_coeff",
                       "surge_drag_lin_n_per_mps", "surge_drag_quad_n_per_mps2", "pack", "current_preset_a",
                       "hotel_load_w", "lights_load_w", "camera_port", "camera_hfov_air_deg",
                       "pressure_part_ratings_msw", "air_mass_kg", "displaced_volume_l", "tank_water_density_kg_m3",
                       "ballast_piece_mass_kg", "ballast_piece_volume_l", "removable_ballast_pieces",
                       "spare_ballast_pieces", "bg_m", "static_pitch_trim_deg", "active_pitch_moment_nm",
                       "jaw_reach_m", "tether_length_m", "tether_diameter_mm", "wet_part_min_temps_c",
                       "link_test_length_m", "sonar_min_range_m", "dvl_min_altitude_m", "gripper_opening_mm",
                       "depth_sensor_range_bar")
SITE_RECORD_KEYS = ("water_temp_c", "working_depth_m", "launch_path_m", "face_reach_m", "design_flow_m3s",
                    "rack_width_m", "rack_height_m", "open_height_m", "secchi_m", "min_water_temp_c", "crane_swl_kg",
                    "rigging_kg", "survey_altitude_m", "sample_max_mm")
NEEDS = ("N-01", "N-02", "N-03", "N-04", "N-05", "N-06", "N-07", "N-08", "N-09", "N-10", "N-11", "N-12")


def _choice(value, allowed, what):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{what} must be one of {allowed}, got {value!r}")
    return value


def _number(value, what):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return float(value)


# ---------------------------------------------------------------- numeric kernel (machine-independent)
_DCTX = Context(prec=56)                 # decimal working precision of the transcendental functions
_DEPS = Decimal(10) ** -58               # series / iteration stop
_KCACHE = {}


def _kcache(name, fn, args):
    key = (name,) + tuple(a.hex() for a in args)          # bit-exact key: keeps the sign of zero
    v = _KCACHE.get(key)
    if v is None:
        if len(_KCACHE) > 200000:
            _KCACHE.clear()
        v = _KCACHE[key] = fn(*args)
    return v


def _sin_cos_d(x):
    x2 = x * x
    s, term, i = x, x, 1
    while abs(term) > _DEPS:
        term = -term * x2 / ((2 * i) * (2 * i + 1))
        s += term
        i += 1
    c, term, i = Decimal(1), Decimal(1), 1
    while abs(term) > _DEPS:
        term = -term * x2 / ((2 * i - 1) * (2 * i))
        c += term
        i += 1
    return s, c


def _trig(kind, x):
    with localcontext(_DCTX):
        if kind == "asin":
            z = Decimal(x)
            y = z
            for _ in range(200):
                s, c = _sin_cos_d(y)
                y_new = y - (s - z) / c
                done = abs(y_new - y) <= _DEPS
                y = y_new
                if done:
                    break
            return float(y)
        s, c = _sin_cos_d(Decimal(x))
        return float(s if kind == "sin" else c if kind == "cos" else s / c)


def _sin(x):
    """Correctly rounded sin of the double x (|x| <= 8)."""
    x = float(x)
    if not abs(x) <= 8.0:
        raise ValueError(f"sin argument out of range: {x!r}")
    return _kcache("sin", lambda a: _trig("sin", a), (x,))


def _cos(x):
    """Correctly rounded cos of the double x (|x| <= 8)."""
    x = float(x)
    if not abs(x) <= 8.0:
        raise ValueError(f"cos argument out of range: {x!r}")
    return _kcache("cos", lambda a: _trig("cos", a), (x,))


def _tan(x):
    """Correctly rounded tan of the double x (|x| <= 1.5)."""
    x = float(x)
    if not abs(x) <= 1.5:
        raise ValueError(f"tan argument out of range: {x!r}")
    return _kcache("tan", lambda a: _trig("tan", a), (x,))


def _asin(x):
    """Correctly rounded asin of the double x (|x| <= 0.99)."""
    x = float(x)
    if not abs(x) <= 0.99:
        raise ValueError(f"asin argument out of range: {x!r}")
    return _kcache("asin", lambda a: _trig("asin", a), (x,))


def _pow_d(x, y):
    with localcontext(_DCTX):
        return float((Decimal(y) * Decimal(x).ln()).exp())


def _pow(x, y):
    """Correctly rounded x ** y for finite x >= 0 and finite y (0 ** y = 0 for y > 0)."""
    x, y = float(x), float(y)
    if not (math.isfinite(x) and math.isfinite(y) and x >= 0.0) or (x == 0.0 and y <= 0.0):
        raise ValueError(f"power out of domain: {x!r} ** {y!r}")
    if x == 0.0:
        return 0.0
    if x == 1.0 or y == 0.0:
        return 1.0
    return _kcache("pow", _pow_d, (x, y))


def _nodes(xp, fp):
    xp = [float(a) for a in xp]
    fp = [float(a) for a in fp]
    if len(xp) < 2 or len(xp) != len(fp) or any(not b > a for a, b in zip(xp, xp[1:])) \
            or not all(math.isfinite(a) for a in xp + fp):
        raise ValueError("a table needs >= 2 strictly increasing finite nodes and as many finite values")
    return xp, fp


def interp(x, xp, fp):
    """Linear interpolation of the table (xp, fp) at one number x (the formula of the module docstring)."""
    x = _number(x, "x")
    xp, fp = _nodes(xp, fp)
    n = len(xp)
    if x < xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]
    j = bisect.bisect_right(xp, x) - 1
    if xp[j] == x:
        return fp[j]
    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    return slope * (x - xp[j]) + fp[j]


def _interp_array(xs, xp, fp):
    """interp() element-wise on a float array (separate IEEE operations, the same formula)."""
    xp, fp = _nodes(xp, fp)
    xa, fa = np.asarray(xp, dtype=float), np.asarray(fp, dtype=float)
    n = len(xp)
    j = np.clip(np.searchsorted(xa, xs, side="right") - 1, 0, n - 2)
    slope = (fa[j + 1] - fa[j]) / (xa[j + 1] - xa[j])
    y = slope * (xs - xa[j]) + fa[j]
    y = np.where(xs == xa[j], fa[j], y)
    y = np.where(xs >= xa[n - 1], fa[n - 1], y)
    return np.where(xs < xa[0], fa[0], y)


# ---------------------------------------------------------------- pure-water equation of state
def rho(t_c, p_dbar):
    """Density of pure water [kg/m3] at in-situ temperature t_c [degC, ITS-90] and sea pressure p_dbar [dbar].

    The 1980 international equation of state of seawater at zero salinity (one-atmosphere density and secant bulk
    modulus of pure water), temperature on the 1968 scale (t68 = 1.00024 t90).  No salinity term: conductivity
    records are not used.  Element-wise on arrays."""
    t = 1.00024 * np.asarray(t_c, dtype=float)
    p = np.asarray(p_dbar, dtype=float) / 10.0                      # bar
    rw = (((((6.536332e-9 * t - 1.120083e-6) * t + 1.001685e-4) * t - 9.095290e-3) * t + 6.793952e-2) * t
          + 999.842594)
    kw = (((-5.155288e-5 * t + 1.360477e-2) * t - 2.327105) * t + 148.4206) * t + 19652.21
    aw = ((-5.77905e-7 * t + 1.16092e-4) * t + 1.43713e-3) * t + 3.239908
    bw = (5.2787e-8 * t - 6.12293e-6) * t + 8.50935e-5
    k = kw + (aw + bw * p) * p
    out = rw / (1.0 - p / k)
    return float(out) if np.ndim(out) == 0 else out


def site_density(water_temp_c, working_depth_m):
    """Site water density [kg/m3]: rho at the working depth, the pressure taken as 1 dbar per metre of depth."""
    return rho(water_temp_c, working_depth_m)


# ---------------------------------------------------------------- policy statistics: the three roundings of Part A
def _round_to_step(x, step, rounding, what):
    x = _number(x, what)
    with localcontext(Context(prec=60)):
        d = Decimal(repr(x)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
        s = Decimal(step)
        return float((d / s).to_integral_value(rounding=rounding) * s)


def policy_design_flow_m3s(max_plant_flow_m3s):
    """Design discharge: the policy maximum of the hourly plant discharge, rounded UP to 0.5 m3/s."""
    return _round_to_step(max_plant_flow_m3s, "0.5", ROUND_CEILING, "max_plant_flow_m3s")


def policy_secchi_m(lower_quartile_m):
    """Secchi statistic: the policy lower quartile of the Secchi readings, rounded DOWN to 0.1 m."""
    return _round_to_step(lower_quartile_m, "0.1", ROUND_FLOOR, "lower_quartile_m")


def policy_min_water_temp_c(lowest_c):
    """Lowest water temperature of the policy, rounded DOWN to 0.1 degC."""
    return _round_to_step(lowest_c, "0.1", ROUND_FLOOR, "lowest_c")


def rack_flow_speed_mps(design_flow_m3s, rack_width_m, open_height_m):
    """Design flow speed in front of the rack [m/s] = design discharge / (rack width x open height)."""
    return design_flow_m3s / (rack_width_m * open_height_m)


def standoff_m(secchi_m):
    """Inspection standoff [m] = standoff_per_secchi x Secchi statistic."""
    return POLICY["standoff_per_secchi"] * secchi_m


# ---------------------------------------------------------------- thrusters
def thr_full(thruster, bus_v):
    """Full-throttle forward thrust [kgf], current [A] and shaft speed [rpm] of one thruster at bus voltage bus_v:
    linear interpolation in the page's voltage columns, clamped at the end columns."""
    vg = thruster["supply_v"]
    return (interp(bus_v, vg, thruster["thrust_kgf"]), interp(bus_v, vg, thruster["current_a"]),
            interp(bus_v, vg, thruster["speed_rpm"]))


def axis_factor(axis, vehicle):
    """Thrust along an axis per unit full-forward thrust of one thruster [1].

    surge_astern: the front pair runs forward, the rear pair reversed: 2 cos(front) + 2 reverse_ratio cos(rear).
    surge_ahead: the mirror pairing: 2 reverse_ratio cos(front) + 2 cos(rear)."""
    _choice(axis, AXES, "axis")
    k = vehicle["thruster"]["reverse_ratio"]
    c_front = _cos(math.radians(vehicle["front_thruster_angle_deg"]))
    c_rear = _cos(math.radians(vehicle["rear_thruster_angle_deg"]))
    if axis == "surge_astern":
        return 2.0 * c_front + 2.0 * k * c_rear
    return 2.0 * k * c_front + 2.0 * c_rear


def thrust_at_current(vehicle, axis, bus_v, battery_a):
    """Static axis thrust [N] and the shaft speed [rpm] of a forward-running thruster when the battery delivers
    battery_a [A] at bus voltage bus_v [V]; battery_a None means full throttle (no current budget).

    The hotel current hotel_load_w / bus_v comes first; the rest is shared equally by the four horizontal
    thrusters; per-thruster thrust = T_full(V) x (I / I_full(V)) ** thrust_power_exp, capped at full throttle;
    axis thrust = thrust_system_coeff x axis_factor x per-thruster thrust x G."""
    thr = vehicle["thruster"]
    tf, af, nf = thr_full(thr, bus_v)
    frac = 1.0 if battery_a is None else min(1.0, max(0.0, (battery_a - vehicle["hotel_load_w"] / bus_v) / 4) / af)
    t1 = tf * _pow(frac, thr["thrust_power_exp"])
    return vehicle["thrust_system_coeff"] * axis_factor(axis, vehicle) * t1 * G, nf * math.sqrt(t1 / tf)


def inflow_factor(thruster, rpm, flow_mps):
    """Inflow (advance-ratio) loss factor: 1 - flow_mps / (prop_pitch_m x rpm / 60), floored at 0; flow_mps is the
    whole on-coming flow speed."""
    return max(0.0, 1.0 - flow_mps / (thruster["prop_pitch_m"] * rpm / 60.0))


# ---------------------------------------------------------------- battery
def pack_ocv_v(pack, soc, temp_c):
    """Open-circuit pack voltage [V] at state of charge soc and temperature temp_c."""
    return interp(soc, pack["ocv_soc_frac"], pack["ocv_v"]) - pack["ocv_drift_v_per_c"] * (pack["ocv_ref_temp_c"] - temp_c)


def pack_retention(pack, temp_c):
    """Capacity retention of the cell lot at temp_c [1] (table, linear between nodes, clamped)."""
    return interp(temp_c, pack["retention_temp_c"], pack["retention_frac"])


def pack_r_ss_ohm(pack, temp_c):
    """Steady-state pack resistance at temp_c [Ohm] (after 60 s at load) = r_ss_23c_ohm x r_factor(temp_c)."""
    return pack["r_ss_23c_ohm"] * interp(temp_c, pack["r_factor_temp_c"], pack["r_factor"])


def pack_energy_wh(pack, load_w, temp_c, cutoff_v, soc_end):
    """THE battery function: energy [Wh] delivered at constant power load_w [W] from full charge at temp_c [degC]
    until the terminal voltage falls below cutoff_v [V] or the state of charge reaches soc_end.

    Terminal voltage at constant power: V = (E + sqrt(E**2 - 4 P R)) / 2 with E = pack_ocv_v(SoC, T) and
    R = pack_r_ss_ohm(T).  The charge axis (capacity_ah x capacity_factor x retention(T)) is cut into 400 equal
    steps from SoC 1 to soc_end; V is evaluated at the step ends; the energy is the trapezoid sum over the steps whose
    two ends both have V >= cutoff_v, plus the part of the first failing step up to the cutoff crossing, located by
    linear interpolation of V between its two ends.  A point with E**2 < 4 P R (no operating point) has V = 0.
    V(SoC = 1) < cutoff_v gives 0 Wh."""
    n_grid = 400
    c_ah = pack["capacity_ah"] * pack["capacity_factor"] * pack_retention(pack, temp_c)
    r = pack_r_ss_ohm(pack, temp_c)
    i = np.arange(n_grid + 1, dtype=float)
    s = 1.0 + (soc_end - 1.0) * (i / n_grid)
    s[-1] = soc_end
    e0 = _interp_array(s, pack["ocv_soc_frac"], pack["ocv_v"]) - pack["ocv_drift_v_per_c"] * (pack["ocv_ref_temp_c"] - temp_c)
    disc = e0 * e0 - 4.0 * load_w * r
    v = np.where(disc > 0, (e0 + np.sqrt(np.maximum(disc, 0.0))) / 2.0, 0.0)
    ok = v >= cutoff_v
    n = len(v) if bool(ok.all()) else int(np.argmin(ok))
    if n < 1:
        return 0.0
    terms = (0.5 * (v[1:n] + v[:n - 1])).tolist()
    if n < len(v):                                   # partial step n-1 -> n up to the crossing (v[n-1] >= cutoff > v[n])
        va, vb = float(v[n - 1]), float(v[n])
        frac = (va - cutoff_v) / (va - vb)
        terms.append(frac * 0.5 * (va + cutoff_v))
    return math.fsum(terms) * c_ah * (1.0 - soc_end) / n_grid


def end_of_dive_bus(vehicle, water_temp_c):
    """Bus at the end of the dive under a full astern demand: (bus voltage [V], battery current [A], binding limit).

    At SoC = reserve and the water temperature: I = min(continuous rating, factory preset, (OCV - v_min) / R) -- ties
    go to the first of 'continuous', 'preset', 'bus_floor' -- and, if smaller, the full-throttle demand of the four
    horizontal thrusters plus the hotel current ('full_throttle', evaluated at the resulting bus voltage, six
    passes).  Bus voltage = OCV - I R."""
    pack, thr = vehicle["pack"], vehicle["thruster"]
    e = pack_ocv_v(pack, POLICY["reserve"], water_temp_c)
    r = pack_r_ss_ohm(pack, water_temp_c)
    lim = {"continuous": pack["continuous_current_a"], "preset": vehicle["current_preset_a"],
           "bus_floor": max(0.0, (e - POLICY["v_min"]) / r)}
    which = min(lim, key=lim.get)
    i = lim[which]
    for _ in range(6):
        v = e - i * r
        i_full = 4.0 * thr_full(thr, v)[1] + vehicle["hotel_load_w"] / v
        if i_full < i:
            i, which = i_full, "full_throttle"
    return e - i * r, i, which


# ---------------------------------------------------------------- drag, holding power, camera
def drag_n(vehicle, flow_mps, water_density_kg_m3, open_height_m):
    """Drag at the rack [N]: vehicle surge drag (linear + quadratic) + in-service tether drag over the open height
    (0.5 rho cd_tether d v**2 per metre)."""
    v = flow_mps
    return (vehicle["surge_drag_lin_n_per_mps"] * v + vehicle["surge_drag_quad_n_per_mps2"] * v * v
            + 0.5 * water_density_kg_m3 * POLICY["cd_tether"] * (vehicle["tether_diameter_mm"] / 1000.0) * v * v * open_height_m)


def hold_power_w(vehicle, drag_force_n, flow_mps):
    """Electrical power [W] of the four horizontal thrusters holding drag_force_n against flow_mps (12 V column of
    the thruster page): fixed point t = drag / (coeff x axis_factor(surge_astern) x G x max(0.05, inflow factor(t)))
    from t = 0.5 kgf, at most 60 passes; power = 4 x power_at_1kgf_w x t ** (1 / thrust_power_exp)."""
    thr = vehicle["thruster"]
    tf, _, nf = thr_full(thr, 12.0)
    k = vehicle["thrust_system_coeff"] * axis_factor("surge_astern", vehicle) * G
    t1 = 0.5
    for _ in range(60):
        n = nf * math.sqrt(t1 / tf)
        infl = max(0.05, 1.0 - flow_mps / (thr["prop_pitch_m"] * n / 60.0))
        t_new = drag_force_n / (k * infl)
        if t_new == t1:
            break
        t1 = t_new
    return 4.0 * thr["power_at_1kgf_w"] * _pow(t1, 1.0 / thr["thrust_power_exp"])


def water_hfov_deg(camera_port, hfov_air_deg):
    """Horizontal field of view in water [deg]: a dome port keeps the air value; a flat port refracts:
    2 asin(sin(hfov_air / 2) / n_flat_port)."""
    _choice(camera_port, CAMERA_PORTS, "camera_port")
    hf = math.radians(hfov_air_deg)
    return math.degrees(2.0 * _asin(_sin(hf / 2.0) / POLICY["n_flat_port"])) if camera_port == "flat" else hfov_air_deg


# ---------------------------------------------------------------- the twelve needs
def n01_net_buoyancy_kg(air_mass_kg, displaced_volume_l, water_density_kg_m3):
    """Net buoyancy [kg] (positive = floats) of (air mass, displaced volume) in water of the given density."""
    return water_density_kg_m3 * displaced_volume_l / 1000.0 - air_mass_kg


def n01_trim_reachable(vehicle, water_density_kg_m3):
    """N-01, Part C rule: the best window margin [kg] over the ballast states the unit's own kit reaches: j removable
    pieces taken out (j = 0 .. removable_ballast_pieces) or -j spares added (j = -spare_ballast_pieces .. -1).
    Window: [buoy_lo, buoy_hi] x air mass of that state; margin = distance to the nearer edge, inside positive."""
    b = n01_net_buoyancy_kg(vehicle["air_mass_kg"], vehicle["displaced_volume_l"], water_density_kg_m3)
    bal_m = vehicle["ballast_piece_mass_kg"]
    step = bal_m - vehicle["ballast_piece_volume_l"] * water_density_kg_m3 / 1000.0
    best = -math.inf
    for j in range(-vehicle["spare_ballast_pieces"], vehicle["removable_ballast_pieces"] + 1):
        m = vehicle["air_mass_kg"] - j * bal_m
        best = max(best, min(b + j * step - POLICY["buoy_lo"] * m, POLICY["buoy_hi"] * m - b - j * step))
    return best


def n01_as_delivered(air_mass_kg, displaced_volume_l, water_density_kg_m3):
    """N-01, Part D rule: window margin [kg] of the unit as ballasted at the works (no trimming at the site)."""
    b = n01_net_buoyancy_kg(air_mass_kg, displaced_volume_l, water_density_kg_m3)
    return min(b - POLICY["buoy_lo"] * air_mass_kg, POLICY["buoy_hi"] * air_mass_kg - b)


def n02_depth_margin_m(pressure_part_ratings_msw, working_depth_m, water_density_kg_m3):
    """N-02 [m]: the weakest pressure part as built, in metres of the site water (rating x msw_rho / rho), minus
    depth_factor x working depth."""
    return min(pressure_part_ratings_msw) * POLICY["msw_rho"] / water_density_kg_m3 - POLICY["depth_factor"] * working_depth_m


def _n03(vehicle, site):
    rho_site = site_density(site["water_temp_c"], site["working_depth_m"])
    v = rack_flow_speed_mps(site["design_flow_m3s"], site["rack_width_m"], site["open_height_m"])
    vb, itot, which = end_of_dive_bus(vehicle, site["water_temp_c"])
    t0, n = thrust_at_current(vehicle, "surge_astern", vb, itot)
    t_av = t0 * inflow_factor(vehicle["thruster"], n, v)
    d = drag_n(vehicle, v, rho_site, site["open_height_m"])
    return t_av / d - POLICY["thrust_factor"], {
        "end_of_dive_bus_v": vb, "end_of_dive_battery_a": itot, "binding_limit": which,
        "available_thrust_n": t_av, "drag_n": d}


def n03_station_thrust(vehicle, site):
    """N-03 [1]: available astern thrust at the end of the dive (end_of_dive_bus at the water temperature, then the
    inflow loss at the design flow speed) / drag at the rack (drag_n at the site density and the open height)
    - thrust_factor."""
    return _n03(vehicle, site)[0]


def _n04(vehicle, site):
    rho_site = site_density(site["water_temp_c"], site["working_depth_m"])
    hfov_w = water_hfov_deg(vehicle["camera_port"], vehicle["camera_hfov_air_deg"])
    hw = math.radians(hfov_w)
    swath = 2.0 * standoff_m(site["secchi_m"]) * _tan(hw / 2.0)
    n_strip = math.ceil(site["rack_width_m"] / (swath * (1.0 - POLICY["overlap"])))
    t_rack = n_strip * (site["rack_height_m"] / POLICY["scan_mps"] + POLICY["strip_turn_s"]) + POLICY["contingency_s"]
    v = rack_flow_speed_mps(site["design_flow_m3s"], site["rack_width_m"], site["open_height_m"])
    hotel = vehicle["hotel_load_w"]
    p_rack = hotel + hold_power_w(vehicle, drag_n(vehicle, v, rho_site, site["open_height_m"]), v) + POLICY["rack_extra_w"]
    e = (p_rack * t_rack
         + (hotel + POLICY["transit_vert_extra_w"]) * 2.0 * site["working_depth_m"] / POLICY["transit_mps"]
         + (hotel + POLICY["transit_face_extra_w"]) * 2.0 * site["face_reach_m"] / POLICY["transit_mps"]) / 3600.0
    usable = pack_energy_wh(vehicle["pack"], p_rack, site["water_temp_c"], POLICY["v_min"], POLICY["reserve"]) * POLICY["eta_dist"]
    detail = {"water_hfov_deg": hfov_w, "strips": n_strip, "rack_time_s": t_rack, "rack_power_w": p_rack,
              "dive_energy_wh": e, "usable_energy_wh": usable}
    if usable <= 0.0:
        return -math.inf, detail
    return (usable - e) / usable, detail


def n04_energy_reserve(vehicle, site):
    """N-04 [1]: (usable - needed) / usable for one dive (-inf when nothing is usable).

    Strips: swath = 2 standoff tan(water hfov / 2); strips = ceil(rack width / (swath x (1 - overlap))).
    Time at the rack = strips x (rack height / scan_mps + strip_turn_s) + contingency_s, at power hotel +
    hold_power_w(drag at the rack) + rack_extra_w.  Transits: down and up the working depth at transit_mps with
    transit_vert_extra_w, and twice the dam-face reach with transit_face_extra_w.  Usable = pack_energy_wh(pack,
    rack power, water temperature, v_min, reserve) x eta_dist."""
    return _n04(vehicle, site)[0]


def n05_tether_margin_m(tether_len_m, launch_path_m, face_reach_m):
    """N-05 [m]: tether length - slack x (launch_path_m + face_reach_m); launch_path_m = the tether length from the
    payout counter zero along the payout route to the lowest point of the inspection scope."""
    return tether_len_m - POLICY["slack"] * (launch_path_m + face_reach_m)


def n06_pitch_authority(vehicle):
    """N-06 [1]: (W BG sin(pitch_lim - static trim) + active pitch moment) / (sample x G x jaw reach) - 1, with
    W = air mass x G, the static trim nose down positive and pitch_lim_deg from horizontal."""
    cap = (vehicle["air_mass_kg"] * G * vehicle["bg_m"]
           * _sin(math.radians(POLICY["pitch_lim_deg"] - vehicle["static_pitch_trim_deg"]))
           + vehicle["active_pitch_moment_nm"])
    return cap / (POLICY["sample_kg"] * G * vehicle["jaw_reach_m"]) - 1.0


def n07_wet_temperature_margin_c(min_water_temp_c, wet_part_min_temps_c):
    """N-07 [degC]: lowest water temperature of the policy minus the highest lower operating limit of the wet parts."""
    return min_water_temp_c - max(wet_part_min_temps_c)


def n08_launch_mass_margin_kg(crane_swl_kg, air_mass_kg, rigging_kg):
    """N-08 [kg]: safe working load of the lifting appliance - air mass of the unit - lifting gear."""
    return crane_swl_kg - air_mass_kg - rigging_kg


def n09_near_structure_margin_m(sonar_min_range_m, standoff_m, dvl_min_altitude_m, survey_altitude_m):
    """N-09 [m]: min(standoff - sonar minimum range, survey altitude - velocity-log minimum altitude); standoff_m is
    the inspection standoff (the function standoff_m of the Secchi statistic)."""
    return min(standoff_m - sonar_min_range_m, survey_altitude_m - dvl_min_altitude_m)


def n10_link_test_margin_m(link_test_length_m, tether_len_m):
    """N-10 [m]: tested data-link length - tether length."""
    return link_test_length_m - tether_len_m


def n11_gripper_opening_margin_mm(gripper_opening_mm, sample_max_mm):
    """N-11 [mm]: jaw opening - largest sample."""
    return gripper_opening_mm - sample_max_mm


def n12_depth_range_margin_m(depth_sensor_range_bar, working_depth_m, water_density_kg_m3):
    """N-12 [m]: depth-sensor full scale in metres of the site water - working depth."""
    return depth_sensor_range_bar * 1e5 / (water_density_kg_m3 * G) - working_depth_m


def k4_near_structure(sonar_min_range_m, standoff_m, dvl_min_altitude_m, survey_altitude_m):
    """N-09 pass / fail (margin >= 0)."""
    return bool(n09_near_structure_margin_m(sonar_min_range_m, standoff_m, dvl_min_altitude_m, survey_altitude_m) >= 0.0)


def k5_link_test(link_test_length_m, tether_len_m):
    """N-10 pass / fail (margin >= 0)."""
    return bool(n10_link_test_margin_m(link_test_length_m, tether_len_m) >= 0.0)


def k6_gripper_opening(gripper_opening_mm, sample_max_mm):
    """N-11 pass / fail (margin >= 0)."""
    return bool(n11_gripper_opening_margin_mm(gripper_opening_mm, sample_max_mm) >= 0.0)


def k7_depth_range(depth_sensor_range_bar, working_depth_m, water_density_kg_m3):
    """N-12 pass / fail (margin >= 0)."""
    return bool(n12_depth_range_margin_m(depth_sensor_range_bar, working_depth_m, water_density_kg_m3) >= 0.0)


def rows(vehicle, site):
    """All twelve margins of one unit (one configuration in one state) on one site record: {need id: margin}.
    N-01 is the Part C rule n01_trim_reachable."""
    rho_site = site_density(site["water_temp_c"], site["working_depth_m"])
    depth = site["working_depth_m"]
    return {
        "N-01": n01_trim_reachable(vehicle, rho_site),
        "N-02": n02_depth_margin_m(vehicle["pressure_part_ratings_msw"], depth, rho_site),
        "N-03": n03_station_thrust(vehicle, site),
        "N-04": n04_energy_reserve(vehicle, site),
        "N-05": n05_tether_margin_m(vehicle["tether_length_m"], site["launch_path_m"], site["face_reach_m"]),
        "N-06": n06_pitch_authority(vehicle),
        "N-07": n07_wet_temperature_margin_c(site["min_water_temp_c"], vehicle["wet_part_min_temps_c"]),
        "N-08": n08_launch_mass_margin_kg(site["crane_swl_kg"], vehicle["air_mass_kg"], site["rigging_kg"]),
        "N-09": n09_near_structure_margin_m(vehicle["sonar_min_range_m"], standoff_m(site["secchi_m"]),
                                            vehicle["dvl_min_altitude_m"], site["survey_altitude_m"]),
        "N-10": n10_link_test_margin_m(vehicle["link_test_length_m"], vehicle["tether_length_m"]),
        "N-11": n11_gripper_opening_margin_mm(vehicle["gripper_opening_mm"], site["sample_max_mm"]),
        "N-12": n12_depth_range_margin_m(vehicle["depth_sensor_range_bar"], depth, rho_site),
    }


def site_values(vehicle, site):
    """The eight `site` values of evaluation.json for one unit on one site record.

    as_tendered_net_buoyancy_n = net buoyancy of the unit as given (no trimming) in the site water x G [N];
    station_thrust_ratio = N-03 margin + thrust_factor; pitch_authority_ratio = N-06 margin + 1; the other five are
    the margins of N-02, N-04, N-05, N-07 and N-08."""
    m = rows(vehicle, site)
    rho_site = site_density(site["water_temp_c"], site["working_depth_m"])
    return {
        "as_tendered_net_buoyancy_n": n01_net_buoyancy_kg(vehicle["air_mass_kg"], vehicle["displaced_volume_l"], rho_site) * G,
        "depth_rating_margin_m": m["N-02"],
        "station_thrust_ratio": m["N-03"] + POLICY["thrust_factor"],
        "dive_energy_reserve_frac": m["N-04"],
        "tether_reach_margin_m": m["N-05"],
        "pitch_authority_ratio": m["N-06"] + 1.0,
        "wet_temperature_margin_c": m["N-07"],
        "launch_mass_margin_kg": m["N-08"],
    }


def row_details(vehicle, site):
    """Intermediate values of N-03 and N-04 (and the site quantities they use), for checking a calculation."""
    _, d3 = _n03(vehicle, site)
    _, d4 = _n04(vehicle, site)
    out = {"water_density_kg_m3": site_density(site["water_temp_c"], site["working_depth_m"]),
           "flow_speed_mps": rack_flow_speed_mps(site["design_flow_m3s"], site["rack_width_m"], site["open_height_m"]),
           "standoff_m": standoff_m(site["secchi_m"]),
           "tether_need_m": POLICY["slack"] * (site["launch_path_m"] + site["face_reach_m"])}
    out.update(d3)
    out.update(d4)
    return out


# ---------------------------------------------------------------- facility quantities (supplier's works, TS-01 Part D)
def tank_net_buoyancy_n(air_mass_kg, displaced_volume_l, tank_water_density_kg_m3):
    """Net buoyancy in the supplier's trim tank [N] (positive = floats)."""
    return n01_net_buoyancy_kg(air_mass_kg, displaced_volume_l, tank_water_density_kg_m3) * G


def bench_bollard_thrust_n(axis, supply_v, limiter, vehicle):
    """THE bench-bollard function (factory bollard reports, the witness bench, the facility quantity) [N].
    The bench supply feeds the battery interface at supply_v; limiter 'on' = as delivered (the firmware battery-current
    limit = the factory preset); 'off' = full throttle on the four horizontal thrusters, no current budget."""
    _choice(axis, AXES, "axis")
    _choice(limiter, LIMITER, "limiter")
    battery_a = vehicle["current_preset_a"] if limiter == "on" else None
    return thrust_at_current(vehicle, axis, supply_v, battery_a)[0]


def bollard_bus_current_a(axis, supply_v, limiter, vehicle):
    """Bench supply current during the bollard test [A]: full-throttle current of the four horizontal thrusters +
    hotel current, limited to the factory preset when the limiter is on."""
    _choice(axis, AXES, "axis")
    _choice(limiter, LIMITER, "limiter")
    full = 4 * thr_full(vehicle["thruster"], supply_v)[1] + vehicle["hotel_load_w"] / supply_v
    return min(full, vehicle["current_preset_a"]) if limiter == "on" else full


def battery_delivered_energy_wh(pack, load_w, chamber_c, cutoff_v):
    """Cycler discharge in the climatic chamber [Wh]: from full charge at constant power load_w to cutoff_v."""
    return pack_energy_wh(pack, load_w, chamber_c, cutoff_v, 0.0)


def pack_dcir_mohm(pack, chamber_c):
    """Steady-state DC internal resistance of the pack at chamber_c [mOhm] (after 60 s at load)."""
    return pack_r_ss_ohm(pack, chamber_c) * 1000.0


def hotel_power_w(vehicle, lights_pct):
    """Hotel electrical load with the lights at lights_pct [W] = hotel_load_w - lights_load_w x (1 - lights_pct / 100)."""
    return vehicle["hotel_load_w"] - vehicle["lights_load_w"] * (1.0 - lights_pct / 100.0)


def proof_pressure_bar(vehicle, hold_min):
    """Proof pressure the weakest pressure part holds [bar] (the hold time does not change it)."""
    return min(vehicle["pressure_part_ratings_msw"]) * POLICY["msw_rho"] * G / 1e5


def tow_drag_n(vehicle, speed_mps):
    """Tow-tank surge drag of the unit without tether [N]."""
    return vehicle["surge_drag_lin_n_per_mps"] * speed_mps + vehicle["surge_drag_quad_n_per_mps2"] * speed_mps * speed_mps


def camera_hfov_deg(vehicle, medium):
    """Horizontal field of view with air or tank water in front of the port [deg]."""
    _choice(medium, MEDIUM, "medium")
    if medium == "air":
        return vehicle["camera_hfov_air_deg"]
    return water_hfov_deg(vehicle["camera_port"], vehicle["camera_hfov_air_deg"])


def facility_value(quantity, conditions, vehicle):
    """Value of a facility quantity of FACILITY_QUANTITIES for a unit at the supplier's works.

    conditions: a dict whose keys are exactly the quantity's condition names (no defaults); numeric conditions are
    finite numbers, enumerated ones take the listed values (axis: AXES, limiter: LIMITER, medium: MEDIUM)."""
    if quantity not in FACILITY_QUANTITIES:
        raise ValueError(f"unknown facility quantity {quantity!r}")
    names = FACILITY_QUANTITIES[quantity][1]
    if not isinstance(conditions, dict) or set(conditions) != set(names):
        raise ValueError(f"{quantity}: conditions must have exactly the keys {names}, got {conditions!r}")
    c = {}
    for k in names:
        if k == "axis":
            c[k] = _choice(conditions[k], AXES, k)
        elif k == "limiter":
            c[k] = _choice(conditions[k], LIMITER, k)
        elif k == "medium":
            c[k] = _choice(conditions[k], MEDIUM, k)
        else:
            c[k] = _number(conditions[k], k)
    if quantity == "air_mass_kg":
        return vehicle["air_mass_kg"]
    if quantity == "displaced_volume_l":
        return vehicle["displaced_volume_l"]
    if quantity == "tank_net_buoyancy_n":
        return tank_net_buoyancy_n(vehicle["air_mass_kg"], vehicle["displaced_volume_l"], vehicle["tank_water_density_kg_m3"])
    if quantity == "bollard_thrust_n":
        return bench_bollard_thrust_n(c["axis"], c["supply_v"], c["limiter"], vehicle)
    if quantity == "bollard_bus_current_a":
        return bollard_bus_current_a(c["axis"], c["supply_v"], c["limiter"], vehicle)
    if quantity == "battery_delivered_energy_wh":
        return battery_delivered_energy_wh(vehicle["pack"], c["load_w"], c["chamber_c"], c["cutoff_v"])
    if quantity == "pack_dcir_mohm":
        return pack_dcir_mohm(vehicle["pack"], c["chamber_c"])
    if quantity == "hotel_power_w":
        return hotel_power_w(vehicle, c["lights_pct"])
    if quantity == "proof_pressure_bar":
        return proof_pressure_bar(vehicle, c["hold_min"])
    if quantity == "tow_drag_n":
        return tow_drag_n(vehicle, c["speed_mps"])
    if quantity == "tether_length_m":
        return vehicle["tether_length_m"]
    if quantity == "tether_diameter_mm":
        return vehicle["tether_diameter_mm"]
    if quantity == "camera_hfov_deg":
        return camera_hfov_deg(vehicle, c["medium"])
    if quantity == "sonar_min_range_m":
        return vehicle["sonar_min_range_m"]
    return vehicle["gripper_opening_mm"]
