"""Static trim of the delivered vehicle: the grader form of the trim item.

Truth = ``tests/trim_table.json`` (the numbers of the printed trim sheet): every row rounded (mass 1 g, the ballast row
0.1 g; displacement 0.1 cm3; CG / CB 0.1 mm), plus this vehicle's per-unit deviation (dm, dx, dy) applied to the
upstream rows taken together (mass and CG x, y; displacement and CB as tabled).  The delivered state is the upstream
rows as this vehicle is, the rows of this station and of the next station as tabled, and one ballast unit for every hole
that carries a LOCKED ballast block at finalize.  In water of density rho:

    W = g sum m_i          B = rho g sum V_i          dB = B - W                 (net buoyancy, + = floats up)
    d = B (r_B - p_T) - W (r_G - p_T)                  (p_T = centroid of the four vertical thrusters: they hold depth)
    theta = atan2(-d_x, d_z)  (+ = bow down)           phi = atan2(d_y, d_z)  (+ = port side up)
    trim = 0.4 [lo <= dB <= hi] + 0.3 [.. and |theta| <= A] + 0.3 [.. and |phi| <= A]

This is the depth-hold attitude, not the free-floating one (do not substitute a free-floating equilibrium formula: on
the batch-median vehicle with its standard plan the two differ by about 0.2 deg).  rho comes from the work order's water
temperature and practical salinity by TEOS-10 at 1 dbar (``rho_teos10``, self-contained below: SA from SP with the
absolute-salinity anomaly ratio of the TEOS-10 atlas at 0 E 0 N, CT from in-situ t, density from the 75-term
specific-volume polynomial; transcribed from the TEOS-10 GSW C library 3.6.23 and checked against it to 1e-9 kg/m3).

Dependencies: the standard library and numpy.  Nothing else of the task or of the development tree is imported.

Signatures (the static-trim function reported to the shared vehicle package):
    rho_teos10(T_C, SP, p_dbar=1.0) -> float                              kg/m3
    static_trim(rows, rho, p_T_mm, g=9.80665) -> (dB_N, theta_deg, phi_deg)
        rows: iterable of (mass_kg, disp_m3, cg_mm[3], cb_mm[3]) in the vehicle frame (FLU, mm)
Grader entry points:
    t = TrimTable()                                        # tests/trim_table.json
    r = grade(t, deviation, locked_holes, T_C=..., SP=...) # or rho=...
    r['trim'], r['dB_N'], r['theta_deg'], r['phi_deg']
Study helpers (vectorised over every legal ballast layout): legal_layouts(t), LayoutSpace(t), evaluate_layouts(...).
"""
from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np

SCHEMA = "rov-trim-table/1"
G_STD = 9.80665                       # standard gravity, m/s2 (the table carries its own value; this is only the default)
TABLE_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "trim_table.json")

# ------------------------------------------------------------------------------------------------------------------
# TEOS-10 density (in-situ density from in-situ temperature and practical salinity)
# ------------------------------------------------------------------------------------------------------------------
_SSO = 35.16504                       # standard ocean reference salinity, g/kg
_UPS = _SSO / 35.0                    # g/kg per unit of practical salinity
_SFAC = 0.0248826675584615            # 1 / (40 _UPS)
_OFFSET = 5.971840214030754e-1        # deltaS = 24, offset = deltaS * _SFAC
_CP0 = 3991.86795711963               # J/(kg K)
_T0 = 273.15                          # K
_SAAR_1DBAR_0E_0N = 1.60321800309889e-05   # absolute-salinity anomaly ratio of the TEOS-10 atlas at 1 dbar, 0 E, 0 N
_P_DBAR_TABULATED = 1.0

_V = dict(
    v000=1.0769995862e-3, v001=-6.0799143809e-5, v002=9.9856169219e-6, v003=-1.1309361437e-6, v004=1.0531153080e-7,
    v005=-1.2647261286e-8, v006=1.9613503930e-9, v010=-3.1038981976e-4, v011=2.4262468747e-5, v012=-5.8484432984e-7,
    v013=3.6310188515e-7, v014=-1.1147125423e-7, v020=6.6928067038e-4, v021=-3.4792460974e-5, v022=-4.8122251597e-6,
    v023=1.6746303780e-8, v030=-8.5047933937e-4, v031=3.7470777305e-5, v032=4.9263106998e-6, v040=5.8086069943e-4,
    v041=-1.7322218612e-5, v042=-1.7811974727e-6, v050=-2.1092370507e-4, v051=3.0927427253e-6, v060=3.1932457305e-5,
    v100=-1.5649734675e-5, v101=1.8505765429e-5, v102=-1.1736386731e-6, v103=-3.6527006553e-7, v104=3.1454099902e-7,
    v110=3.5009599764e-5, v111=-9.5677088156e-6, v112=-5.5699154557e-6, v113=-2.7295696237e-7, v120=-4.3592678561e-5,
    v121=1.1100834765e-5, v122=5.4620748834e-6, v130=3.4532461828e-5, v131=-9.8447117844e-6, v132=-1.3544185627e-6,
    v140=-1.1959409788e-5, v141=2.5909225260e-6, v150=1.3864594581e-6, v200=2.7762106484e-5, v201=-1.1716606853e-5,
    v202=2.1305028740e-6, v203=2.8695905159e-7, v210=-3.7435842344e-5, v211=-2.3678308361e-7, v212=3.9137387080e-7,
    v220=3.5907822760e-5, v221=2.9283346295e-6, v222=-6.5731104067e-7, v230=-1.8698584187e-5, v231=-4.8826139200e-7,
    v240=3.8595339244e-6, v300=-1.6521159259e-5, v301=7.9279656173e-6, v302=-4.6132540037e-7, v310=2.4141479483e-5,
    v311=-3.4558773655e-6, v312=7.7618888092e-9, v320=-1.4353633048e-5, v321=3.1655306078e-7, v330=2.2863324556e-6,
    v400=6.9111322702e-6, v401=-3.4102187482e-6, v402=-6.3352916514e-8, v410=-8.7595873154e-6, v411=1.2956717783e-6,
    v420=4.3703680598e-6, v500=-8.0539615540e-7, v501=5.0736766814e-7, v510=-3.3052758900e-7, v600=2.0543094268e-7)


def _entropy_part(sa, t, p):
    """entropy minus the terms that are a function of SA only (GSW gsw_entropy_part)"""
    x2 = _SFAC * sa
    x = math.sqrt(x2)
    y = t * 0.025
    z = p * 1e-4
    g03 = (z * (-270.983805184062 +
                z * (776.153611613101 + z * (-196.51255088122 +
                                             (28.9796526294175 - 2.13290083518327 * z) * z))) +
           y * (-24715.571866078 + z * (2910.0729080936 +
                                         z * (-1513.116771538718 + z * (546.959324647056 +
                                                                        z * (-111.1208127634436 + 8.68841343834394 * z)))) +
                y * (2210.2236124548363 + z * (-2017.52334943521 +
                                               z * (1498.081172457456 + z * (-718.6359919632359 +
                                                                             (146.4037555781616 - 4.9892131862671505 * z) * z))) +
                     y * (-592.743745734632 + z * (1591.873781627888 +
                                                   z * (-1207.261522487504 + (608.785486935364 -
                                                                              105.4993508931208 * z) * z)) +
                          y * (290.12956292128547 + z * (-973.091553087975 +
                                                         z * (602.603274510125 + z * (-276.361526170076 +
                                                                                      32.40953340386105 * z))) +
                               y * (-113.90630790850321 + y * (21.35571525415769 -
                                                               67.41756835751434 * z) +
                                    z * (381.06836198507096 + z * (-133.7383902842754 +
                                                                   49.023632509086724 * z))))))))
    g08 = x2 * (z * (729.116529735046 +
                     z * (-343.956902961561 + z * (124.687671116248 +
                                                   z * (-31.656964386073 + 7.04658803315449 * z)))) +
                x * (x * (y * (-137.1145018408982 + y * (148.10030845687618 +
                                                         y * (-68.5590309679152 + 12.4848504784754 * y))) -
                          22.6683558512829 * z) + z * (-175.292041186547 +
                                                       (83.1923927801819 - 29.483064349429 * z) * z) +
                     y * (-86.1329351956084 + z * (766.116132004952 +
                                                   z * (-108.3834525034224 + 51.2796974779828 * z)) +
                          y * (-30.0682112585625 - 1380.9597954037708 * z +
                               y * (3.50240264723578 + 938.26075044542 * z)))) +
                y * (1760.062705994408 + y * (-675.802947790203 +
                                              y * (365.7041791005036 + y * (-108.30162043765552 +
                                                                            12.78101825083098 * y) +
                                                   z * (-1190.914967948748 + (298.904564555024 -
                                                                              145.9491676006352 * z) * z)) +
                                              z * (2082.7344423998043 + z * (-614.668925894709 +
                                                                             (340.685093521782 - 33.3848202979239 * z) * z))) +
                     z * (-1721.528607567954 + z * (674.819060538734 +
                                                    z * (-356.629112415276 + (88.4080716616 -
                                                                              15.84003094423364 * z) * z)))))
    return -(g03 + g08) * 0.025


def _entropy_part_zerop(sa, pt0):
    """entropy part at the sea surface (GSW gsw_entropy_part_zerop)"""
    x2 = _SFAC * sa
    x = math.sqrt(x2)
    y = pt0 * 0.025
    g03 = y * (-24715.571866078 + y * (2210.2236124548363 +
                                       y * (-592.743745734632 + y * (290.12956292128547 +
                                                                     y * (-113.90630790850321 + y * 21.35571525415769)))))
    g08 = x2 * (x * (x * (y * (-137.1145018408982 + y * (148.10030845687618 +
                                                         y * (-68.5590309679152 + 12.4848504784754 * y)))) +
                     y * (-86.1329351956084 + y * (-30.0682112585625 +
                                                   y * 3.50240264723578))) +
                y * (1760.062705994408 + y * (-675.802947790203 +
                                              y * (365.7041791005036 + y * (-108.30162043765552 +
                                                                            12.78101825083098 * y)))))
    return -(g03 + g08) * 0.025


def _gibbs_pt0_pt0(sa, pt0):
    """second temperature derivative of the Gibbs function at (sa, pt0, 0) (GSW gsw_gibbs_pt0_pt0)"""
    x2 = _SFAC * sa
    x = math.sqrt(x2)
    y = pt0 * 0.025
    g03 = (-24715.571866078 +
           y * (4420.4472249096725 +
                y * (-1778.231237203896 +
                     y * (1160.5182516851419 +
                          y * (-569.531539542516 + y * 128.13429152494615)))))
    g08 = x2 * (1760.062705994408 + x * (-86.1329351956084 +
                                         x * (-137.1145018408982 + y * (296.20061691375236 +
                                                                        y * (-205.67709290374563 + 49.9394019139016 * y))) +
                                         y * (-60.136422517125 + y * 10.50720794170734)) +
                y * (-1351.605895580406 + y * (1097.1125373015109 +
                                               y * (-433.20648175062206 + 63.905091254154904 * y))))
    return (g03 + g08) * 0.000625


def _pt0_from_t(sa, t, p):
    """potential temperature referred to 0 dbar (GSW gsw_pt0_from_t: polynomial start + 2 modified Newton steps)"""
    s1 = sa / _UPS
    pt0 = t + p * (8.65483913395442e-6 -
                   s1 * 1.41636299744881e-6 -
                   p * 7.38286467135737e-9 +
                   t * (-8.38241357039698e-6 +
                        s1 * 2.83933368585534e-8 +
                        t * 1.77803965218656e-8 +
                        p * 1.71155619208233e-10))
    dentropy_dt = _CP0 / ((_T0 + pt0) * (1.0 - 0.05 * (1.0 - sa / _SSO)))
    true_entropy_part = _entropy_part(sa, t, p)
    for _ in range(2):
        pt0_old = pt0
        dentropy = _entropy_part_zerop(sa, pt0_old) - true_entropy_part
        pt0 = pt0_old - dentropy / dentropy_dt
        pt0m = 0.5 * (pt0 + pt0_old)
        dentropy_dt = -_gibbs_pt0_pt0(sa, pt0m)
        pt0 = pt0_old - dentropy / dentropy_dt
    return pt0


def _ct_from_pt(sa, pt):
    """Conservative Temperature from potential temperature (GSW gsw_ct_from_pt)"""
    x2 = _SFAC * sa
    x = math.sqrt(x2)
    y = pt * 0.025
    pot_enthalpy = (61.01362420681071 + y * (168776.46138048015 +
                                              y * (-2735.2785605119625 + y * (2574.2164453821433 +
                                                                              y * (-1536.6644434977543 + y * (545.7340497931629 +
                                                                                                              (-50.91091728474331 - 18.30489878927802 * y) * y))))) +
                    x2 * (268.5520265845071 + y * (-12019.028203559312 +
                                                    y * (3734.858026725145 + y * (-2046.7671145057618 +
                                                                                  y * (465.28655623826234 + (-0.6370820302376359 -
                                                                                                             10.650848542359153 * y) * y)))) +
                          x * (937.2099110620707 + y * (588.1802812170108 +
                                                        y * (248.39476522971285 + (-3.871557904936333 -
                                                                                   2.6268019854268356 * y) * y)) +
                               x * (-1687.914374187449 + x * (246.9598888781377 +
                                                              x * (123.59576582457964 - 48.5891069025409 * x)) +
                                    y * (936.3206544460336 +
                                         y * (-942.7827304544439 + y * (369.4389437509002 +
                                                                        (-33.83664947895248 - 9.987880382780322 * y) * y)))))))
    return pot_enthalpy / _CP0


def _specvol(sa, ct, p):
    """specific volume, 75-term polynomial in (SA, CT, p) (GSW gsw_specvol)"""
    v = _V
    xs = math.sqrt(_SFAC * sa + _OFFSET)
    ys = ct * 0.025
    z = p * 1e-4
    return (v['v000']
            + xs * (v['v010'] + xs * (v['v020'] + xs * (v['v030'] + xs * (v['v040'] + xs * (v['v050']
            + v['v060'] * xs))))) + ys * (v['v100'] + xs * (v['v110'] + xs * (v['v120'] + xs * (v['v130'] + xs * (v['v140']
            + v['v150'] * xs)))) + ys * (v['v200'] + xs * (v['v210'] + xs * (v['v220'] + xs * (v['v230'] + v['v240'] * xs)))
            + ys * (v['v300'] + xs * (v['v310'] + xs * (v['v320'] + v['v330'] * xs)) + ys * (v['v400'] + xs * (v['v410']
            + v['v420'] * xs) + ys * (v['v500'] + v['v510'] * xs + v['v600'] * ys))))) + z * (v['v001'] + xs * (v['v011']
            + xs * (v['v021'] + xs * (v['v031'] + xs * (v['v041'] + v['v051'] * xs)))) + ys * (v['v101'] + xs * (v['v111']
            + xs * (v['v121'] + xs * (v['v131'] + v['v141'] * xs))) + ys * (v['v201'] + xs * (v['v211'] + xs * (v['v221']
            + v['v231'] * xs)) + ys * (v['v301'] + xs * (v['v311'] + v['v321'] * xs) + ys * (v['v401'] + v['v411'] * xs
            + v['v501'] * ys)))) + z * (v['v002'] + xs * (v['v012'] + xs * (v['v022'] + xs * (v['v032'] + v['v042'] * xs)))
            + ys * (v['v102'] + xs * (v['v112'] + xs * (v['v122'] + v['v132'] * xs)) + ys * (v['v202'] + xs * (v['v212']
            + v['v222'] * xs) + ys * (v['v302'] + v['v312'] * xs + v['v402'] * ys))) + z * (v['v003'] + xs * (v['v013']
            + v['v023'] * xs) + ys * (v['v103'] + v['v113'] * xs + v['v203'] * ys) + z * (v['v004'] + v['v014'] * xs + v['v104'] * ys
            + z * (v['v005'] + v['v006'] * z))))))


def absolute_salinity(SP, p_dbar=1.0):
    """Absolute Salinity [g/kg] from practical salinity at 0 E 0 N (the anomaly ratio is tabulated for 1 dbar only)"""
    if p_dbar != _P_DBAR_TABULATED:
        raise ValueError("the anomaly ratio is tabulated for p = 1 dbar only (the work order's water is at the surface)")
    return _UPS * float(SP) * (1.0 + _SAAR_1DBAR_0E_0N)


def rho_teos10(T_C, SP, p_dbar=1.0):
    """in-situ density [kg/m3] of the delivery-site water: TEOS-10, SA_from_SP(SP, p, 0 E, 0 N) -> CT_from_t -> rho"""
    T_C = float(T_C)
    SP = float(SP)
    if not (math.isfinite(T_C) and math.isfinite(SP)) or not (-2.0 <= T_C <= 40.0) or not (0.0 <= SP <= 42.0):
        raise ValueError("water outside the range of the density function: T %r degC, SP %r" % (T_C, SP))
    sa = absolute_salinity(SP, p_dbar)
    ct = _ct_from_pt(sa, _pt0_from_t(sa, T_C, float(p_dbar)))
    return 1.0 / _specvol(sa, ct, float(p_dbar))


# ------------------------------------------------------------------------------------------------------------------
# static trim
# ------------------------------------------------------------------------------------------------------------------
def static_trim(rows, rho, p_T_mm, g=G_STD):
    """rows: iterable of (mass_kg, disp_m3, cg_mm[3], cb_mm[3]) -> (dB_N, theta_deg, phi_deg) at depth hold about p_T."""
    M = 0.0
    V = 0.0
    mG = [0.0, 0.0, 0.0]
    vB = [0.0, 0.0, 0.0]
    for m, v, cg, cb in rows:
        m = float(m)
        v = float(v)
        M += m
        V += v
        for k in range(3):
            mG[k] += m * float(cg[k])
            vB[k] += v * float(cb[k])
    W = M * g
    B = rho * V * g
    dB = B - W
    d = [rho * g * vB[k] - g * mG[k] - dB * float(p_T_mm[k]) for k in range(3)]
    return dB, math.degrees(math.atan2(-d[0], d[2])), math.degrees(math.atan2(d[1], d[2]))


def _as_deviation(deviation):
    if deviation is None:
        return 0.0, 0.0, 0.0
    if isinstance(deviation, dict):
        return float(deviation["dm_kg"]), float(deviation["dx_mm"]), float(deviation["dy_mm"])
    dm, dx, dy = deviation
    return float(dm), float(dx), float(dy)


class TrimTable:
    """tests/trim_table.json, read once; all lengths in mm, masses in kg, volumes in m3."""

    def __init__(self, path=None, data=None):
        if data is None:
            with open(path or TABLE_DEFAULT) as f:
                data = json.load(f)
        if data.get("schema") != SCHEMA:
            raise ValueError("not a %s table: %r" % (SCHEMA, data.get("schema")))
        self.raw = data
        self.g = float(data["g"])
        self.p_T_mm = tuple(float(c) for c in data["p_T_mm"])
        acc = data["acceptance"]
        self.window = dict(net_buoyancy_N=tuple(float(x) for x in acc["net_buoyancy_N"]),
                           pitch_abs_deg=float(acc["pitch_abs_deg"]), roll_abs_deg=float(acc["roll_abs_deg"]))
        sh = acc["score_shares"]
        self.shares = (float(sh["net_buoyancy"]), float(sh["pitch"]), float(sh["roll"]))
        self.rows = []
        for r in data["rows"]:
            self.rows.append((r["row"], r["cls"], r["mass_g"] / 1000.0, r["disp_cm3"] * 1e-6,
                              tuple(float(c) for c in r["cg_mm"]), tuple(float(c) for c in r["cb_mm"])))
        self.groups = {}
        for cls in ("base", "station", "next"):
            sel = [r for r in self.rows if r[1] == cls]
            if not sel:
                raise ValueError("no %s rows" % cls)
            M = sum(r[2] for r in sel)
            V = sum(r[3] for r in sel)
            mG = tuple(sum(r[2] * r[4][k] for r in sel) for k in range(3))
            vB = tuple(sum(r[3] * r[5][k] for r in sel) for k in range(3))
            self.groups[cls] = (M, V, mG, vB)
        M0, _, mG0, _ = self.groups["base"]
        self.base_mass_kg = M0
        self.base_cg_mm = tuple(c / M0 for c in mG0)
        bu = data["ballast_unit"]
        self.unit_mass_kg = bu["mass_g"] / 1000.0
        self.unit_disp_m3 = bu["disp_cm3"] * 1e-6
        self.unit_zg_mm = float(bu["cg_mm"][2])
        self.unit_zb_mm = float(bu["cb_mm"][2])
        self.holes = {k: (float(v[0]), float(v[1])) for k, v in data["holes_mm"].items()}
        self.names = list(data["holes_mm"])
        self.blocked = tuple(data["blocked"])
        self.usable = [n for n in self.names if n not in self.blocked]
        cf = data["conflict_mm"]
        self.conflict_dx = float(cf["dx_below"])
        self.conflict_dy = float(cf["dy_below"])

    # ---- the delivered state ---------------------------------------------------------------------------------
    def conflicts(self, a, b):
        (xa, ya), (xb, yb) = self.holes[a], self.holes[b]
        return abs(xa - xb) < self.conflict_dx and abs(ya - yb) < self.conflict_dy

    def base_row(self, deviation=None):
        """the upstream rows taken together as this vehicle: (mass, disp, CG, CB) with the deviation on mass and CG x, y"""
        dm, dx, dy = _as_deviation(deviation)
        M0, V0, _, vB0 = self.groups["base"]
        M = M0 + dm
        cg = (self.base_cg_mm[0] + dx, self.base_cg_mm[1] + dy, self.base_cg_mm[2])
        return (M, V0, cg, tuple(c / V0 for c in vB0))

    def ballast_row(self, hole):
        x, y = self.holes[hole]
        return (self.unit_mass_kg, self.unit_disp_m3, (x, y, self.unit_zg_mm), (x, y, self.unit_zb_mm))

    def delivered_rows(self, deviation=None, holes=(), include=("base", "station", "next")):
        rows = []
        if "base" in include:
            rows.append(self.base_row(deviation))
        for _, cls, m, v, cg, cb in self.rows:
            if cls in include and cls != "base":
                rows.append((m, v, cg, cb))
        for h in holes:
            rows.append(self.ballast_row(h))
        return rows


def trim_score(dB, theta, phi, window, shares=(0.4, 0.3, 0.3)):
    """0, 0.4, 0.7 or 1.0 with the shares 0.4 / 0.3 / 0.3; every bound included"""
    lo, hi = window["net_buoyancy_N"]
    ok = lo <= dB <= hi
    p = ok and abs(theta) <= window["pitch_abs_deg"]
    r = ok and abs(phi) <= window["roll_abs_deg"]
    return round(shares[0] * ok + shares[1] * p + shares[2] * r, 12)


def check_holes(table, holes):
    """the LOCKED ballast holes as the coupling reports them: known, usable, each once"""
    holes = list(holes)
    bad = [h for h in holes if h not in table.holes]
    if bad:
        raise ValueError("unknown ballast holes %r" % bad)
    blk = [h for h in holes if h in table.blocked]
    if blk:
        raise ValueError("blocked holes cannot carry a fastened ballast block: %r" % blk)
    if len(set(holes)) != len(holes):
        raise ValueError("a hole is listed twice: %r" % holes)
    return holes


def grade(table, deviation, locked_holes, rho=None, T_C=None, SP=None, window=None):
    """trim of one episode -> dict(trim, dB_N, theta_deg, phi_deg, rho_kg_m3, n_ballast, holes, in_window, conflicting_pairs)"""
    holes = check_holes(table, locked_holes)
    if rho is None:
        if T_C is None or SP is None:
            raise ValueError("give rho or the work order's T_C and SP")
        rho = rho_teos10(T_C, SP)
    rho = float(rho)
    w = window or table.window
    dB, th, ph = static_trim(table.delivered_rows(deviation, holes), rho, table.p_T_mm, table.g)
    s = trim_score(dB, th, ph, w, table.shares)
    lo, hi = w["net_buoyancy_N"]
    pairs = [(a, b) for a, b in itertools.combinations(holes, 2) if table.conflicts(a, b)]
    return dict(trim=s, dB_N=dB, theta_deg=th, phi_deg=ph, rho_kg_m3=rho, n_ballast=len(holes),
                holes=[h for h in table.names if h in holes],
                in_window=dict(net_buoyancy=bool(lo <= dB <= hi), pitch=bool(abs(th) <= w["pitch_abs_deg"]),
                               roll=bool(abs(ph) <= w["roll_abs_deg"])),
                conflicting_pairs=pairs)


# ------------------------------------------------------------------------------------------------------------------
# every legal layout (studies, selection of the hidden set); the grader itself only needs grade()
# ------------------------------------------------------------------------------------------------------------------
def legal_layouts(table, blocked=None):
    """canonical order: number of weights ascending, then itertools.combinations over the usable holes in the
    table's hole-map order; the empty layout first."""
    blocked = set(table.blocked if blocked is None else blocked)
    ok = [n for n in table.names if n not in blocked]
    out = []
    for k in range(len(ok) + 1):
        for c in itertools.combinations(ok, k):
            if not any(table.conflicts(a, b) for a, b in itertools.combinations(c, 2)):
                out.append(c)
    return out


class LayoutSpace:
    def __init__(self, table, blocked=None):
        self.t = table
        self.L = legal_layouts(table, blocked)
        XY = np.array([table.holes[n] for n in table.names], float)
        SM = np.zeros((len(self.L), len(table.names)), bool)
        col = {n: i for i, n in enumerate(table.names)}
        for i, c in enumerate(self.L):
            SM[i, [col[n] for n in c]] = True
        self.NN = SM.sum(1).astype(float)
        self.SX = SM @ XY[:, 0]
        self.SY = SM @ XY[:, 1]
        self.index = {frozenset(c): i for i, c in enumerate(self.L)}

    def idx(self, holes):
        return self.index[frozenset(holes)]


def evaluate_layouts(table, space, deviation, rho, include=("base", "station", "next")):
    """(dB, theta, phi) arrays over space.L for this vehicle and water (same algebra as static_trim, vectorised)"""
    g = table.g
    pT = table.p_T_mm
    M = V = 0.0
    mG = [0.0, 0.0, 0.0]
    vB = [0.0, 0.0, 0.0]
    for m, v, cg, cb in table.delivered_rows(deviation, (), include):
        M += m
        V += v
        for k in range(3):
            mG[k] += m * cg[k]
            vB[k] += v * cb[k]
    bm, bv, zg, zb = table.unit_mass_kg, table.unit_disp_m3, table.unit_zg_mm, table.unit_zb_mm
    NN, SX, SY = space.NN, space.SX, space.SY
    dB = rho * (V + NN * bv) * g - (M + NN * bm) * g
    dx = rho * g * (vB[0] + bv * SX) - g * (mG[0] + bm * SX) - dB * pT[0]
    dy = rho * g * (vB[1] + bv * SY) - g * (mG[1] + bm * SY) - dB * pT[1]
    dz = rho * g * (vB[2] + bv * NN * zb) - g * (mG[2] + bm * NN * zg) - dB * pT[2]
    return dB, np.degrees(np.arctan2(-dx, dz)), np.degrees(np.arctan2(dy, dz))


def score_layouts(dB, th, ph, window, shares=(0.4, 0.3, 0.3)):
    lo, hi = window["net_buoyancy_N"]
    ok = (dB >= lo) & (dB <= hi)
    p = ok & (np.abs(th) <= window["pitch_abs_deg"])
    r = ok & (np.abs(ph) <= window["roll_abs_deg"])
    return np.round(shares[0] * ok + shares[1] * p + shares[2] * r, 12)


if __name__ == "__main__":
    t = TrimTable()
    sp = LayoutSpace(t)
    rho = rho_teos10(20.0, 0.0)
    print("layouts", len(sp.L), "most weights", int(sp.NN.max()), "blocked", t.blocked)
    print("rho(20 C, SP 0) = %.6f kg/m3" % rho)
    print("nominal vehicle, no ballast:", grade(t, None, [], rho=rho))
    std = t.raw["standard_pattern"]
    dr = t.raw["per_unit_deviation"]
    med = tuple(0.5 * (dr[k][0] + dr[k][1]) for k in ("mass_kg", "cg_x_mm", "cg_y_mm"))
    print("standard pattern on the batch-median vehicle:",
          grade(t, med, std["holes"], T_C=std["water"]["temperature_C"], SP=std["water"]["salinity"]))
