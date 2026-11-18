"""Typed readers of a quad-layer/1 (public code; no defaults: a missing key raises KeyError naming it).

A layer is {"format": "quad-layer/1", "brief": {...}, "lots": {...}}: the brief block carries every number of the
customer brief and the build log that the acceptance procedure reads (SI units; the mission time in seconds).
"""
import re

from . import schemas as S
from .quadmodel import ProcedureError

DUTY_KEYS = ("stance_height_m", "crouch_height_m", "standup_time_s", "step_length_m", "swing_height_m", "shift_time_s",
             "swing_time_s", "sway_fraction", "dt_s", "gravity_m_s2", "proof_factor")
ELECTRICAL_KEYS = ("cell_end_voltage_v", "harness_drop_v", "tray_current_a", "driver_efficiency", "copper_loss_factor",
                   "usable_fraction", "mission_time_s")
POLICY_KEYS = ("untested_cap", "z_quantile", "battery_families", "ceiling_usd", "eco_usd")
PAYLOAD_KEYS = ("payload_top_kg", "payload_bottom_kg")


def _pick(layer, keys):
    b = layer["brief"]
    return {k: b[k] for k in keys}


def brief(layer):
    """The brief block's keys that the procedure and the criteria read, values unchanged."""
    return _pick(layer, S.BRIEF_KEYS)


def duty(layer):
    return _pick(layer, DUTY_KEYS)


def electrical(layer):
    return _pick(layer, ELECTRICAL_KEYS)


def policy(layer):
    return _pick(layer, POLICY_KEYS)


def payload(layer):
    return _pick(layer, PAYLOAD_KEYS)


def family_cells(family):
    """Cells in series of a battery family name "<n>S" (ProcedureError for any other name)."""
    if not isinstance(family, str) or not re.fullmatch(S.FAMILY_PATTERN, family):   # whole string: no trailing newline
        raise ProcedureError("battery family %r is not of the form <cells>S" % (family,))
    return int(family[:-1])


def module_voltage(layer, family):
    """V_s = cells x the end-of-mission cell voltage - the harness drop (B-14 g)."""
    b = layer["brief"]
    return family_cells(family) * float(b["cell_end_voltage_v"]) - float(b["harness_drop_v"])
