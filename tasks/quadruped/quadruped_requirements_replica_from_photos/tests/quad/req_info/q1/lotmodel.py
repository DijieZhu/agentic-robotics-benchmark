"""Capability of the delivered lot's weakest expected unit (B-14 g, B-10) under a given lot description and a given
catalogue (public code; every number comes from the inputs).

For a module of an inspected vendor, a unit's ratio to its datasheet values at the module voltage V_s is
    peak torque  f_peak (1 + s z1),   no-load speed  f_speed (1 + s z2),   continuous torque  f_cont (1 + s z3)
with the delivered lot's factors f_*, spread s and droop d; a lot's factors apply to every SKU of its vendor. The
weakest expected unit has z = -z_quantile on every factor, and its torque-speed characteristic is
w(tau) = w0 (1 - u)(1 - d u) with u = tau / tau_s, so the torque available at the speed w is tau_s u*(|w| / w0, d) with
u* the root in [0, 1] of (1 - u)(1 - d u) = x. A vendor without an inspection report counts at untested_cap x its
typical peak torque, no-load speed and continuous torque, on the straight line (d = 0).

A pack's capacity counts at g_lot = g_mean (1 - z_quantile g_spread) of its vendor's delivered cell lot.
"""
import math

import numpy as np

from . import layer as L


def droop_fraction(x, d):
    """u* in [0, 1] with (1 - u)(1 - d u) = x, for the speed fraction x clipped to [0, 1]; d <= 1e-12: 1 - x.

    The smaller root (b - sqrt(disc)) / (2 d) with b = 1 + d and disc = b^2 - 4 d (1 - x) = (1 - d)^2 + 4 d x is
    evaluated as 2 (1 - x) / (b + sqrt(disc)) with the second form of disc: the same number without cancellation."""
    x = min(max(float(x), 0), 1)
    d = float(d)
    if d <= 1e-12:
        return 1 - x
    disc = (1 - d) * (1 - d) + 4 * d * x
    return max(0, min(1, 2 * (1 - x) / (1 + d + math.sqrt(disc))))


def droop_fraction_array(x, d):
    """Vectorised droop_fraction."""
    x = np.clip(np.asarray(x, float), 0, 1)
    d = float(d)
    if d <= 1e-12:
        return 1 - x
    disc = (1 - d) * (1 - d) + 4 * d * x
    return np.clip(2 * (1 - x) / (1 + d + np.sqrt(disc)), 0, 1)


def capability(pid, family, cat, lots, layer):
    """{"tau_s", "w0", "tau_c", "droop", "i_pk", "J_out", "R_ph", "mass_kg", "v_s", "rule", "vendor"} of the delivered
    lot's weakest expected unit of module `pid` on a pack of `family`."""
    b = layer["brief"]
    v_s = L.module_voltage(layer, family)
    tau_pg, w_pg, i_pg = cat.page_at(pid, v_s)
    m = cat.get(pid)
    vendor = m["vendor"]
    cont = float(m["continuous_torque_nm"])
    if cat.inspected(vendor):
        mv = lots["modules"][vendor]
        lot = mv["lots"][mv["delivered"]]
        q = 1 - float(b["z_quantile"]) * float(lot["spread"])
        tau_s = tau_pg * float(lot["f_peak"]) * q
        w0 = w_pg * float(lot["f_speed"]) * q
        tau_c = cont * float(lot["f_cont"]) * q
        droop = float(lot["droop"])
        rule = "lot"
    else:
        c = float(b["untested_cap"])
        tau_s, w0, tau_c, droop = c * tau_pg, c * w_pg, c * cont, 0
        rule = "untested_cap"
    return {"tau_s": tau_s, "w0": w0, "tau_c": tau_c, "droop": droop, "i_pk": i_pg,
            "J_out": float(m["output_inertia_kg_m2"]), "R_ph": float(m["phase_resistance_ohm"]),
            "mass_kg": float(m["mass_kg"]), "v_s": v_s, "rule": rule, "vendor": vendor}


def pack_factor(pid, cat, lots, layer):
    """g_lot = g_mean (1 - z_quantile g_spread) of the delivered cell lot of the pack's vendor."""
    vendor = cat.get(pid)["vendor"]
    cells = lots["cells"][vendor]
    lot = cells["lots"][cells["delivered"]]
    return float(lot["g_mean"]) * (1 - float(layer["brief"]["z_quantile"]) * float(lot["g_spread"]))
