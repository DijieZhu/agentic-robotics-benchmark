"""Parameters of the pushable duct walls: load, check, derive per piece.

`duct_params.json` beside this file is the one parameter set. Its frozen keys are `mass_per_m_kg`,
`floor_friction`, `bend_stiffness_nm_per_rad`, `bend_damping_nms_per_rad`, `mode`, `duct_duct_contact`; the rest
are this package's own (constraint softness, sleep, window, put-back). `piece_constants(c)` turns them into the
numbers of one piece of length c: mass, inertia of a thin-walled tube, the three friction limits and the joint
equality that carries the bending spring.

Standard library and numpy only; no MuJoCo here.
"""

import copy
import hashlib
import json
import os

from sketch.examples import constants as K

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "duct_params.json")
G = 9.81
MODES = ("sleep", "active_window", "awake")
FROZEN_KEYS = ("mass_per_m_kg", "floor_friction", "bend_stiffness_nm_per_rad", "bend_damping_nms_per_rad", "mode",
               "duct_duct_contact")


def load_params(path=None):
    """The duct parameters as a dict (a fresh copy on every call)."""
    with open(path or PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def params_digest(params):
    """sha256 of the canonical text of a parameter set (written into every model built from it)."""
    text = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def check_params(params):
    """Errors of a parameter set (an empty list when it is usable)."""
    err = []
    for k in FROZEN_KEYS:
        if k not in params:
            err.append("frozen key %r is missing" % k)
    if err:
        return err
    if params["mode"] not in MODES:
        err.append("mode must be one of %s" % (MODES,))
    for k in ("mass_per_m_kg", "floor_friction"):
        if not float(params[k]) > 0.0:
            err.append("%s must be positive" % k)
    for k in ("bend_stiffness_nm_per_rad", "bend_damping_nms_per_rad"):
        if float(params[k]) < 0.0:
            err.append("%s must not be negative" % k)
    if not isinstance(params["duct_duct_contact"], bool):
        err.append("duct_duct_contact must be true or false")
    d = float(params["bend"]["solimp_d"])
    if not 0.0 < d < 1.0:
        err.append("bend.solimp_d must be in (0, 1)")
    w = params["window"]
    if int(w["slots"]) < 1:
        err.append("window.slots must be at least 1")
    if w.get("slot_sleep"):
        # measured: with the sleep flag on, MuJoCo 3.14 keeps the constraint rows of the
        # previous sleep state until the next sleep event, so a slot whose pins and friction are switched on at
        # activation runs unconstrained for several steps and then jumps
        err.append("window.slot_sleep = true is not supported (MuJoCo 3.14 ignores run-time eq_active / "
                   "frictionloss changes while the sleep flag is on)")
    if float(w["margin_off_m"]) <= float(w["margin_on_m"]):
        err.append("window.margin_off_m must exceed margin_on_m (hysteresis)")
    if not 0.0 < float(params["putback"]["max_err_m"]) <= 0.10:
        err.append("putback.max_err_m must be in (0, 0.10]")
    return err


def with_overrides(params=None, **kw):
    """A copy of the parameters with top-level keys replaced (`mode="sleep"`, `mass_per_m_kg=0.25`, ...) and
    nested ones given as `section__key` (`window__slots=4`)."""
    p = copy.deepcopy(params if params is not None else load_params())
    for key, value in kw.items():
        if "__" in key:
            sec, sub = key.split("__", 1)
            p[sec][sub] = value
        else:
            if key not in p:
                raise KeyError("no parameter %r" % key)
            p[key] = value
    err = check_params(p)
    if err:
        raise ValueError("; ".join(err))
    return p


def piece_constants(params, c):
    """The numbers of one duct piece of length c (m), in its own frame (x along the axis, z up, origin at the
    centre): mass of a thin-walled tube of the duct's radius, its inertia, the frictionloss of the slides along and
    across the axis and of the yaw hinge (a uniform rod turning about its centre: mu m g c / 4), and the bending
    spring toward the laid shape for the hinge between two such pieces (stiffness scaled with 1 / c)."""
    r = K.DUCT_RADIUS_M
    m = float(params["mass_per_m_kg"]) * c
    i_axis = m * r * r
    i_perp = m * (0.5 * r * r + c * c / 12.0)
    mu = float(params["floor_friction"])
    c_nom = float(params["nominal_piece_length_m"])
    return {"length_m": c, "mass_kg": m, "inertia_kgm2": [i_axis, i_perp, i_perp],
            "frictionloss": [mu * m * G, mu * m * G, mu * m * G * c / 4.0],
            "bend_stiffness_nm_per_rad": float(params["bend_stiffness_nm_per_rad"]) * c_nom / c,
            "bend_damping_nms_per_rad": float(params["bend_damping_nms_per_rad"]) * c_nom / c}


def bend_solref(params, izz_a, izz_b, stiffness, damping):
    """The negative solref (-k, -b) of a joint equality yaw_b = yaw_a whose quasi-static torque is `stiffness` x the
    relative yaw and `damping` x its rate. MuJoCo: K_eff = k d / R with k = -solref0 / d^2, R = (1 - d) / d x
    (1 / Izz_a + 1 / Izz_b) at constant impedance d (dmin = dmax), so -solref0 = stiffness (1 - d) A and, with
    b = -solref1 / d, -solref1 = damping (1 - d) A (A = 1 / Izz_a + 1 / Izz_b, the dofs' invweight0)."""
    d = float(params["bend"]["solimp_d"])
    a = 1.0 / izz_a + 1.0 / izz_b
    # MuJoCo reads solref as (-stiffness, -damping) only when solref0 < 0: a zero stiffness becomes a tiny one
    return [min(-stiffness * (1.0 - d) * a, -1e-12), -damping * (1.0 - d) * a]


def bend_solimp(params):
    d = float(params["bend"]["solimp_d"])
    return [d, d, float(params["bend"]["width_rad"]), 0.5, 2.0]
