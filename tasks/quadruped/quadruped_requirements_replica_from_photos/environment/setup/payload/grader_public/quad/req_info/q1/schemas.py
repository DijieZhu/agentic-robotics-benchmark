"""Vocabularies, the unit table and the JSON schemas of the quadruped requirements task (public code; no data numbers).

The deliverables are `requirements.yaml` (req-quad/1), `bom.json` (bom-quad/1) and `kinematic_sketch.json`
(sketch-quad/1). The public procedure also reads a replica model (quad-model/1), a layer (quad-layer/1), a catalogue
(quad-catalogue/1) and a lot description (quad-lots/1); their schemas ship here too.

    to_si(value, unit)          -> (value_si, si_unit), or None for an unknown unit
    validate(doc, which)        -> sorted list of "path: message" problems ([] = valid)
    write_schemas(out_dir)      -> {name: path} of the seven "<name>.schema.json" files
"""
import json
import os

import jsonschema

LEGS = ("fl", "fr", "rl", "rr")
LEG_SIGNS = {"fl": (1, 1), "fr": (1, -1), "rl": (-1, 1), "rr": (-1, -1)}      # (sx, sy) of each leg in the trunk frame
JOINT_TYPES = ("hip_roll", "hip_pitch", "knee")
JOINTS = tuple(leg + "_" + t for leg in LEGS for t in JOINT_TYPES)          # 12, leg-major
ITEMS = JOINTS + ("battery", "drivers", "hub", "budget")                     # the 16 acceptance items
LINES = ("hip_roll", "hip_pitch", "knee", "battery", "drivers", "hub")       # BOM lines (the budget is no line)
LINKS = ("shoulder", "thigh", "shank")
BODIES = ("trunk",) + tuple(leg + "_" + link for leg in LEGS for link in LINKS)   # the 13 bodies of the chain
GEOMETRY_KEYS = ("pitch_axis_spacing", "roll_axis_spacing", "pitch_axis_drop", "leg_offset", "hip_spacer",
                 "thigh_length", "shank_length", "foot_radius")
HOUSING_CLASSES = ("R52", "R38", "R62")
PACK_BAYS = ("nose_box", "belly_bay")
KNEE_BRANCHES = ("><", "<>", ">>", "<<")
ROLES = ("actuator", "battery", "driver", "hub")
COMPARATORS = (">=", "<=", "==", ">", "<")
FAMILY_PATTERN = r"^[1-9][0-9]*S$"

UNIT_TO_SI = {
    "kg": ("kg", 1.0), "g": ("kg", 0.001),
    "m": ("m", 1.0), "mm": ("m", 0.001), "cm": ("m", 0.01),
    "rad": ("rad", 1.0), "deg": ("rad", 0.017453292519943295),
    "s": ("s", 1.0), "ms": ("s", 0.001), "min": ("s", 60.0), "h": ("s", 3600.0),
    "N.m": ("N.m", 1.0), "N m": ("N.m", 1.0), "Nm": ("N.m", 1.0),
    "kg.cm": ("N.m", 0.0980665), "oz.in": ("N.m", 0.00706155181), "lbf.in": ("N.m", 0.11298482902761668),
    "rad/s": ("rad/s", 1.0), "rpm": ("rad/s", 0.10471975511965977), "deg/s": ("rad/s", 0.017453292519943295),
    "A": ("A", 1.0), "mA": ("A", 0.001),
    "Ah": ("Ah", 1.0), "A.h": ("Ah", 1.0), "mAh": ("Ah", 0.001),
    "V": ("V", 1.0),
    "N": ("N", 1.0),
    "USD": ("USD", 1.0),
    "count": ("count", 1.0),
    "fraction": ("fraction", 1.0), "%": ("fraction", 0.01),
}

DERIVED_QUANTITIES = (
    "proof_peak_torque:hip_roll", "proof_peak_torque:hip_pitch", "proof_peak_torque:knee",
    "duty_rms_torque:hip_roll", "duty_rms_torque:hip_pitch", "duty_rms_torque:knee",
    "standup_peak_torque:knee",
    "joint_peak_speed:hip_pitch", "joint_peak_speed:knee",
    "supply_peak_current", "mission_charge",
    "drive_peak_current:knee", "drive_rms_current:knee",
    "robot_mass", "loaded_com_x", "peak_normal_force")

QUANTITY_SI_UNIT = {
    # brief items (24)
    "payload_mass_max": "kg", "payload_mass_min": "kg", "proof_load_factor": "fraction", "stance_height": "m",
    "crouch_height": "m", "standup_time": "s", "step_length": "m", "swing_height": "m", "shift_time": "s",
    "swing_time": "s", "sway_fraction": "fraction", "mission_time": "s", "usable_fraction": "fraction",
    "cell_end_voltage": "V", "harness_drop": "V", "tray_current": "A", "driver_efficiency": "fraction",
    "unit_count": "count", "actuator_count": "count", "revolute_dof": "count", "pack_count": "count",
    "total_cost": "USD", "eco_cost": "USD", "untested_vendor_derating": "fraction",
    # photo items (3)
    "payload_box_length": "m", "payload_box_width": "m", "payload_box_height": "m",
    # derived items (16)
    "proof_peak_torque:hip_roll": "N.m", "proof_peak_torque:hip_pitch": "N.m", "proof_peak_torque:knee": "N.m",
    "duty_rms_torque:hip_roll": "N.m", "duty_rms_torque:hip_pitch": "N.m", "duty_rms_torque:knee": "N.m",
    "standup_peak_torque:knee": "N.m",
    "joint_peak_speed:hip_pitch": "rad/s", "joint_peak_speed:knee": "rad/s",
    "supply_peak_current": "A", "mission_charge": "Ah",
    "drive_peak_current:knee": "A", "drive_rms_current:knee": "A",
    "robot_mass": "kg", "loaded_com_x": "m", "peak_normal_force": "N",
    # vocabulary only, never graded (2)
    "joint_peak_speed:hip_roll": "rad/s", "loaded_com_z": "m",
}

SOURCE_PATTERN = r"^(B|BL)-[0-9]{2}$|^PH-[0-9]{2}$|^DS-[A-Za-z0-9-]+$|^LR(-[0-9]{4}-[0-9]{4})?$|^derived$"

_DRAFT = "https://json-schema.org/draft/2020-12/schema"

KNEE_BRANCH_TEXT = ("knee_branch gives the knee configuration of B-05 in two characters, the first for the front legs "
                    "(fl, fr) and the second for the rear legs (rl, rr): '>' puts the knee axis behind the line from "
                    "the leg's hip_pitch axis to its foot point (toward -x of the trunk frame of Appendix D), '<' puts "
                    "it ahead of that line (toward +x), so the two characters picture the legs as seen from the left "
                    "(+y) side with the nose on the left.")


def to_si(value, unit):
    """(float(value) * factor, si_unit) for a unit of UNIT_TO_SI, else None."""
    if unit not in UNIT_TO_SI:
        return None
    si, factor = UNIT_TO_SI[unit]
    return float(value) * factor, si


# ---------------------------------------------------------------------------------------------- building blocks
def _num(**kw):
    out = {"type": "number"}
    out.update(kw)
    return out


def _pos():
    return _num(exclusiveMinimum=0)


def _vec(n, item=None):
    return {"type": "array", "items": item or _num(), "minItems": n, "maxItems": n}


def _range():
    return _vec(2)


def _obj(props, required=None, extra=False):
    return {"type": "object", "required": list(props if required is None else required), "properties": props,
            "additionalProperties": extra}


def _per_type(item):
    return _obj({t: item for t in JOINT_TYPES})


# ---------------------------------------------------------------------------------------------- deliverables
def requirements_schema():
    item = _obj({"id": {"type": "string", "pattern": "^REQ-[A-Za-z0-9_.-]+$"},
                 "quantity": {"type": "string", "enum": sorted(QUANTITY_SI_UNIT)},
                 "comparator": {"type": "string", "enum": list(COMPARATORS)},
                 "bound": _num(),
                 "unit": {"type": "string", "enum": sorted(UNIT_TO_SI)},
                 "source_id": {"type": "string", "pattern": SOURCE_PATTERN},
                 "source_clause": {"type": "string"},
                 "derivation": {"type": "string"}},
                required=["id", "quantity", "comparator", "bound", "unit", "source_id"], extra=True)
    return {"$schema": _DRAFT, "title": "req-quad/1 requirements sheet",
            "description": ("Traced requirements of the replica fleet: one item per quantity that the brief, the build "
                            "log, the photos or the documents fix, plus the derived items of brief B-16 (source_id "
                            "'derived' with a derivation note). Bounds in the listed units."),
            "type": "object", "required": ["requirements"],
            "properties": {"project": {"type": "string"}, "units": {"type": "object"}, "notes": {"type": "string"},
                           "requirements": {"type": "array", "minItems": 1, "items": item}},
            "additionalProperties": True}


def bom_schema():
    line = _obj({"part_id": {"type": "string"}, "role": {"type": "string", "enum": list(ROLES)},
                 "count": {"type": "integer", "minimum": 1},
                 "joint_type": {"type": "string", "enum": list(JOINT_TYPES)}, "note": {"type": "string"}},
                required=["part_id", "role", "count"], extra=True)
    return {"$schema": _DRAFT, "title": "bom-quad/1 bill of materials (per replica unit)",
            "description": ("Three actuator lines (one module SKU per joint type, count 4 each), one battery line, one "
                            "driver line (count = 12 / channels per board) and one hub line."),
            "type": "object", "required": ["battery_family", "parts"],
            "properties": {"format": {"const": "bom-quad/1"},
                           "battery_family": {"type": "string", "pattern": FAMILY_PATTERN},
                           "parts": {"type": "array", "minItems": 1, "items": line},
                           "total_cost_usd": _num(), "notes": {"type": "string"}},
            "additionalProperties": True}


def sketch_schema():
    nt = _obj({"nominal": _num(), "tol": _num(minimum=0)}, extra=True)
    return {"$schema": _DRAFT, "title": "sketch-quad/1 kinematic sketch of the replica",
            "description": ("The Appendix-D geometry (m, each with a nominal value and a tolerance), the spacer legs, the "
                            "stops (rad, Appendix-D conventions), the housing classes, the pack bay and the BOM's "
                            "actuator line per joint type."),
            "type": "object",
            "required": ["units", "geometry", "spacer_legs", "stops", "housing_class", "pack_bay", "modules"],
            "properties": {
                "format": {"const": "sketch-quad/1"},
                "units": _obj({"length": {"const": "m"}, "angle": {"const": "rad"}}),
                "geometry": {"type": "object", "required": list(GEOMETRY_KEYS),
                             "properties": {k: nt for k in GEOMETRY_KEYS}, "additionalProperties": True},
                "spacer_legs": {"type": "array", "items": {"type": "string", "enum": list(LEGS)},
                                "description": "distinct leg names (a G0 rule checks that no leg repeats)"},
                "stops": _per_type(_range()),
                "housing_class": _per_type({"type": "string", "enum": list(HOUSING_CLASSES)}),
                "pack_bay": {"type": "string", "enum": list(PACK_BAYS)},
                "modules": _per_type({"type": "string"})},
            "additionalProperties": True}


# ---------------------------------------------------------------------------------------------- inputs of the procedure
def catalogue_schema():
    point = _obj({"voltage_v": _num(), "peak_torque_nm": _num(), "no_load_speed_rad_s": _num(),
                  "peak_current_a": _num()}, extra=True)
    module = _obj({"part_id": {"type": "string"}, "vendor": {"type": "string"},
                   "housing_class": {"type": "string", "enum": list(HOUSING_CLASSES)},
                   "points": {"type": "array", "minItems": 1, "maxItems": 2, "items": point},
                   "continuous_torque_nm": _num(), "phase_resistance_ohm": _num(), "output_inertia_kg_m2": _num(),
                   "mass_kg": _num(), "voltage_range_v": _range(), "price_usd": _num(),
                   "lead_time": {"type": "string"}, "title": {"type": "string"}, "headline": {"type": "string"}},
                  required=["part_id", "vendor", "housing_class", "points", "continuous_torque_nm",
                            "phase_resistance_ohm", "output_inertia_kg_m2", "mass_kg", "voltage_range_v", "price_usd"],
                  extra=True)
    pack = _obj({"part_id": {"type": "string"}, "vendor": {"type": "string"}, "cells": {"type": "integer", "minimum": 1},
                 "capacity_ah": _num(), "continuous_current_a": _num(), "mass_kg": _num(), "outline_m": _vec(3),
                 "cell_voltage_v": _obj({"nominal": _num(), "full": _num(), "cutoff": _num()}, extra=True),
                 "price_usd": _num()}, extra=True)
    driver = _obj({"part_id": {"type": "string"}, "vendor": {"type": "string"},
                   "channels": {"type": "integer", "minimum": 1}, "peak_current_a": _num(),
                   "continuous_current_a": _num(), "input_range_v": _range(), "efficiency": _num(),
                   "price_usd": _num()},
                  required=["part_id", "vendor", "channels", "peak_current_a", "continuous_current_a", "input_range_v",
                            "price_usd"], extra=True)
    hub = _obj({"part_id": {"type": "string"}, "vendor": {"type": "string"}, "ports": {"type": "integer", "minimum": 1},
                "input_range_v": _range(), "price_usd": _num()},
               required=["part_id", "vendor", "ports", "price_usd"], extra=True)
    housing = _obj({"can_diameter_m": _num(), "can_length_m": _num(), "bolt_circle_m": _num(),
                    "bolts": {"type": "integer"}, "bolt": {"type": "string"}}, required=[], extra=True)
    erratum = _obj({"sheet": {"type": "string"}, "item": {"type": "string"}, "part_id": {"type": "string"},
                    "field": {"type": "string"}, "as_printed": {}, "correct": {}}, required=[], extra=True)
    return {"$schema": _DRAFT, "title": "quad-catalogue/1 vendor parts and the quote (SI units)",
            "description": ("The datasheet pages (with the vendors' erratum sheets applied) and the quote in SI units: "
                            "torques N.m, speeds rad/s, currents A, voltages V, resistance ohm, inertia kg.m2 (output "
                            "side), masses kg, lengths m, capacities Ah, prices USD. Each list in the quote's order. A "
                            "vendor is 'inspected' when an incoming-inspection report exists for its parts."),
            "type": "object", "required": ["format", "vendors", "modules", "packs", "drivers", "hubs"],
            "properties": {
                "format": {"const": "quad-catalogue/1"},
                "vendors": {"type": "object", "additionalProperties": _obj(
                    {"name": {"type": "string"}, "inspected": {"type": "boolean"}}, required=["inspected"], extra=True)},
                "housing_classes": {"type": "object", "propertyNames": {"enum": list(HOUSING_CLASSES)},
                                    "additionalProperties": housing},
                "modules": {"type": "array", "items": module},
                "packs": {"type": "array", "items": pack},
                "drivers": {"type": "array", "items": driver},
                "hubs": {"type": "array", "items": hub},
                "errata": {"type": "array", "items": erratum}},
            "additionalProperties": True}


def quad_model_schema():
    body = {"type": "string", "enum": list(BODIES)}
    geometry = _obj({"pitch_axis_spacing": _pos(), "roll_axis_spacing": _pos(), "pitch_axis_drop": _num(),
                     "leg_offset": _num(minimum=0), "hip_spacer": _num(minimum=0),
                     "spacer_legs": {"type": "array", "items": {"type": "string", "enum": list(LEGS)},
                                     "uniqueItems": True},
                     "thigh_length": _pos(), "shank_length": _pos(), "foot_radius": _pos(),
                     "knee_branch": {"type": "string", "enum": list(KNEE_BRANCHES), "description": KNEE_BRANCH_TEXT}})
    seat = _obj({"carrier": body, "pos_m": _vec(3), "quat": _vec(4)})
    inertial = _obj({"name": {"type": "string", "pattern": "^[a-z0-9_]+$"}, "attach": body, "mass_kg": _pos(),
                     "com_m": _vec(3), "inertia_kg_m2": _vec(3, _vec(3)), "note": {"type": "string"}},
                    required=["name", "attach", "mass_kg", "com_m", "inertia_kg_m2"])
    return {"$schema": _DRAFT, "title": "quad-model/1 replica model",
            "description": ("The replica model that quadmodel.py turns into the dynamics model of acceptance procedure "
                            "B-14: the Appendix-D geometry (m), the stops (rad, Appendix-D conventions), the housing "
                            "class of each joint type with its can box (seat frame, z = can axis), the 12 module seats, "
                            "the pack bay, the pod mount with its CoM marks, and every printed part and fixed component "
                            "as a rigid inertial (mass kg, CoM m and inertia kg.m2 about the CoM, in the frame of the "
                            "body it is attached to)."),
            "type": "object",
            "required": ["format", "name", "units", "geometry", "stops", "housing_class", "class_box_m", "seats",
                         "pack_bay", "pod", "rigid_inertials"],
            "properties": {
                "format": {"const": "quad-model/1"},
                "name": {"type": "string"},
                "units": {"const": {"length": "m", "angle": "rad", "mass": "kg"}},
                "geometry": geometry,
                "stops": _per_type(_range()),
                "housing_class": _per_type({"type": "string", "enum": list(HOUSING_CLASSES)}),
                "class_box_m": {"type": "object", "minProperties": 1, "propertyNames": {"enum": list(HOUSING_CLASSES)},
                                "additionalProperties": _vec(3, _pos())},
                "seats": _obj({j: seat for j in JOINTS}),
                "pack_bay": _obj({"name": {"type": "string", "enum": list(PACK_BAYS)}, "centre_m": _vec(3),
                                  "inner_m": _vec(3, _pos())}),
                "pod": _obj({"mount_centre_m": _vec(3), "box_m": _vec(3, _pos()), "mark_offset_m": _vec(2)}),
                "rigid_inertials": {"type": "array", "items": inertial},
                "notes": {"type": "string"}},
            "additionalProperties": True}


_BRIEF_INT = ("unit_count", "actuator_count", "dof", "pack_count")
_BRIEF_NUM = ("payload_top_kg", "payload_bottom_kg", "proof_factor", "stance_height_m", "crouch_height_m",
              "standup_time_s", "step_length_m", "swing_height_m", "shift_time_s", "swing_time_s", "sway_fraction",
              "dt_s", "gravity_m_s2", "mission_time_s", "usable_fraction", "cell_end_voltage_v", "harness_drop_v",
              "tray_current_a", "driver_efficiency", "copper_loss_factor", "untested_cap", "z_quantile",
              "ceiling_usd", "eco_usd")
BRIEF_KEYS = _BRIEF_INT + ("battery_families",) + _BRIEF_NUM


def layer_schema():
    props = {k: {"type": "integer", "minimum": 0} for k in _BRIEF_INT}
    props.update({k: _num() for k in _BRIEF_NUM})
    props["battery_families"] = {"type": "array", "minItems": 1,
                                 "items": {"type": "string", "pattern": FAMILY_PATTERN}, "uniqueItems": True}
    ref = {"type": "object", "additionalProperties": _obj({"lot_id": {"type": "string"}, "report": {"type": "string"}},
                                                         extra=True)}
    return {"$schema": _DRAFT, "title": "quad-layer/1 brief constants and requirements",
            "description": ("The numbers of the customer brief and the build log that the acceptance procedure reads "
                            "(SI units; mission_time_s in seconds), plus the delivered lots named in the quote."),
            "type": "object", "required": ["format", "brief"],
            "properties": {"format": {"const": "quad-layer/1"},
                           "brief": {"type": "object", "required": list(BRIEF_KEYS), "properties": props,
                                     "additionalProperties": True},
                           "lots": {"type": "object", "properties": {"modules": ref, "cells": ref},
                                    "additionalProperties": True}},
            "additionalProperties": True}


def lots_schema():
    module_lot = _obj({"f_peak": _num(), "f_speed": _num(), "f_cont": _num(), "spread": _num(), "droop": _num()},
                      extra=True)
    cell_lot = _obj({"g_mean": _num(), "g_spread": _num()}, extra=True)
    module_vendor = _obj({"delivered": {"type": "string"},
                          "lots": {"type": "object", "minProperties": 1, "additionalProperties": module_lot}},
                         extra=True)
    cell_vendor = _obj({"delivered": {"type": "string"},
                        "lots": {"type": "object", "minProperties": 1, "additionalProperties": cell_lot}}, extra=True)
    return {"$schema": _DRAFT, "title": "quad-lots/1 delivered lots (capability factors)",
            "description": ("Per inspected module vendor: the delivered lot id and each lot's ratios to the datasheet "
                            "(f_peak, f_speed, f_cont), its unit spread and its droop; a lot's factors apply to every "
                            "SKU of the vendor (B-14 g). Per pack vendor: the delivered cell lot and each lot's mean "
                            "and spread of the measured-to-rated capacity ratio."),
            "type": "object", "required": ["format", "modules", "cells"],
            "properties": {"format": {"const": "quad-lots/1"},
                           "modules": {"type": "object", "additionalProperties": module_vendor},
                           "cells": {"type": "object", "additionalProperties": cell_vendor}},
            "additionalProperties": True}


SCHEMAS = {"requirements": requirements_schema, "bom": bom_schema, "kinematic_sketch": sketch_schema,
           "catalogue": catalogue_schema, "quad_model": quad_model_schema, "layer": layer_schema,
           "lots": lots_schema}


def write_schemas(out_dir):
    """Write the seven schemas as "<name>.schema.json" (indent 2, sorted keys, trailing newline)."""
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for name, fn in SCHEMAS.items():
        path = os.path.join(out_dir, "%s.schema.json" % name)
        with open(path, "w") as fh:
            json.dump(fn(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        out[name] = path
    return out


def validate(doc, which):
    """jsonschema problems of `doc` against schema `which` as "path: message", sorted by path."""
    v = jsonschema.Draft202012Validator(SCHEMAS[which]())
    probs = []
    for e in v.iter_errors(doc):
        path = "/".join(str(x) for x in e.absolute_path) or "(root)"
        probs.append((path, e.message))
    return ["%s: %s" % pm for pm in sorted(probs)]
