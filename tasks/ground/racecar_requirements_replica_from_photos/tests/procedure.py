"""Public acceptance procedure of the replica specification (rc-procedure/2; Python 3.12, numpy).

A requirement means: computed by this file, under the conditions the requirement writes out, the quantity of a replica
satisfies the bound. A replica is the reference unit with the parts of a bill of materials (BOM: one SKU code per
purchase line; the purchase lines are the slots of the catalogue, SLOT_ORDER) fitted in place of the parts the
reference unit was found with, and with the pod of the car record on board.

This file holds no number that describes a car, a part, a maker, a site or a use. All of those are inputs:

- the catalogue (format rc-catalogue/2): for every SKU its page values, its pack price, its lot and its fit and
  connector classes; for the chassis kit, which is on every car, its price and the classes it takes;
- the car record of the reference unit (format rc-car/1): its masses, heights, steering geometry and gear data as it
  was found, the parts it was found with, and the pod. CAR_FIELDS lists the fields; Appendix D of the brief defines
  each of them;
- the context {lot_table, cycles, scenarios}: the state of every lot, the cycles (closed tracks) and the scenarios
  (a cycle, a schedule and an environment) that the conditions of a requirement name.

What this file does hold: the closed dictionary of quantities (QUANTITIES: name, unit, condition keys), the rules of
the condition language, the physics, and the constants of the reference driver (stack_notes.md prints the same
values). quantities.md explains every formula in prose; where the two differ, this file decides. Where a comment or
a docstring points to a part of quantities.md, it gives the heading of that part in double quotes.

Units. An input field ends in its unit: _mm, _g, _deg, _s, _v, _a, _ohm, _w, _hz, _usd, _c (degrees Celsius) and so
on. A quantity is returned in the unit of its entry in QUANTITIES (a headroom whose name ends in _c is a temperature
difference in K). Heights are heights above the ground the tyres stand on; x runs forward from the rear axle.

    cat = load_catalogue(json.load(open("my_catalogue.json")))
    car = load_car(json.load(open("my_car.json")))
    prep = prepare(cat, car, {"lot_table": {...}, "cycles": {...}, "scenarios": {...}})
    bom = {"motor": "...", "pinion": "...", ..., "steering_servo": "..."}     # one SKU code per purchase line
    q = quantity("mass_kg", bom, cat, prep, {"config": {"body_shell": False}})   # one car, one quantity
    sat(q, "<=", bound)                                                        # is the requirement satisfied?
    for idx in iter_cars(cat):                                                 # every combination, in blocks
        vals = evaluate_block(idx, cat, prep, [("fits", {}), ("lap_time_s", {"scenario": "..."})])

The block functions compute every car of a block with the same element-wise operations, so a block result equals the
single-car result of the same car. A value that cannot be reached (a speed above the top speed, a servo at its stall
torque) comes back infinite, and sat() counts a value that is not finite as not satisfied.
"""
import itertools
import json
import math
import re
from decimal import ROUND_HALF_UP, Context, Decimal

import numpy as np

PROCEDURE_VERSION = "rc-procedure/2"
CATALOGUE_SCHEMA = "rc-catalogue/2"
CAR_SCHEMA = "rc-car/1"

# ------------------------------------------------------------------------------------------------ messages
# Every text that an error of this file shows (a ValueError, or an entry of the list validate_context returns).
# The placeholders are filled in the order they appear. A leading %s names the place of the problem: one of the
# where_* texts, a slot (followed by an SKU code), or the format of the record.
MESSAGES = {
    # units and ranges
    "unknown_unit": "unknown unit %r: UNIT_FACTORS lists the units; the temperatures are degC, K and degF",
    "cannot_convert": "cannot convert %s (a unit of %s) to %s (a unit of %s)",
    "range": "%s%s, %s%s",
    # where a problem is
    "where_request": "request %s",
    "where_conditions": "%s conditions",
    "where_cycle": "cycle %s",
    "where_segment": "%s segment %d",
    "where_scenario": "scenario %s",
    "where_environment": "%s.environment",
    "where_zone": "%s.zone",
    "where_lot": "lot_table %s",
    # the condition language
    "config_keys": "%s: config must have exactly body_shell",
    "config_shell": "%s: config.body_shell must be true or false",
    "target_keys": "%s: target must have exactly class, width_m",
    "target_class": "%s: target.class must be one of %s",
    "target_width": "%s: target.width_m must be a number in %s",
    "zone_keys": "%s: zone must be null or have exactly speed_cap_mps, appear_m, surface, target",
    "zone_number": "%s: zone.%s must be a number in %s",
    "zone_surface": "%s: zone.surface must be one of %s",
    "cycle_keys": "%s: must have exactly segments",
    "cycle_segments": "%s: segments must be a list of 1..%d segments",
    "segment_kind": "%s: kind must be straight or arc",
    "segment_keys": "%s: a %s segment has exactly %s",
    "segment_number": "%s: %s must be a number in %s",
    "segment_surface": "%s: surface must be one of %s",
    "cycle_length": "%s: total length %.6g m exceeds %g m",
    "schedule_keys": "%s: schedule must have exactly %s",
    "schedule_blocks": "%s: schedule.blocks must be a list of 1..%d blocks",
    "block_keys": "%s: block %d must have exactly drive_s, pace, rest_s",
    "block_number": "%s: block %d %s must be a number in %s",
    "schedule_integer": "%s: schedule.%s must be an integer in %s",
    "schedule_swap": "%s: schedule.swap_after_blocks must be null or an integer in %s",
    "schedule_expanded": "%s: %d expanded blocks (blocks x repeat) exceed %d",
    "schedule_paces": "%s: more than %d distinct pace values",
    "schedule_time": "%s: total time %.6g s exceeds %g s",
    "environment_keys": "%s: environment must have exactly ambient_c, config, zone",
    "environment_ambient": "%s: environment.ambient_c must be a number in %s",
    "lot_id": "lot_table: %r is not a lot id (LOT-C, LOT-T or LOT-L followed by one digit)",
    "cell_lot_keys": "%s: a cell lot has exactly capacity_factor, resistance_factor",
    "lot_number": "%s: %s must be a number in %s",
    "lot_keys": "%s: a %s lot has exactly %s: {%s}",
    "lot_field_number": "%s: %s.%s must be a number in %s",
    "conditions_keys": "%s: conditions of %s must have exactly {%s}",
    "conditions_surface": "%s: surface must be one of %s",
    "conditions_scenario": "%s: scenario %r is not a scenario of the context",
    "conditions_zone": "%s: zone_stop_margin_m needs a scenario with a zone",
    "conditions_number": "%s: %s must be a number in %s",
    "conditions_json": "%s: the conditions must be a mapping of finite numbers, strings, booleans, null and "
                       "mappings of those",
    # the context
    "context_keys": "the context must be a mapping with lot_table, cycles and scenarios",
    "context_lot_table": "the context must be a mapping with a lot_table",
    "lot_table_type": "lot_table must be a mapping",
    "lot_table_ids": "lot_table must have exactly the lot ids %s",
    "lot_table_missing": "lot_table has no entry for lot %s",
    "cycles_type": "cycles must be a mapping of at most %d cycles",
    "cycle_id": "cycle id %r does not match %s",
    "scenarios_type": "scenarios must be a mapping of at most %d scenarios",
    "scenario_id": "scenario id %r does not match %s",
    "scenario_keys": "%s: must have exactly cycle, schedule, environment",
    "scenario_cycle": "%s: cycle %r is not a cycle of the context",
    "cycle_missing": "cycle %r is not in the context",
    "unknown_quantity": "quantity %r is not in the dictionary",
    "no_shell": "%s: a configuration with body_shell true needs the body shell of the catalogue (platform "
                "body_shell_mass_g and body_shell_thermal_factor)",
    "no_formula": "quantity %r is in the dictionary but has no formula",
    "not_prepared": "evaluate_block needs the object that prepare(cat, car, context) returns",
    "requests": "requests must be a list of (quantity, conditions) pairs",
    # the catalogue
    "cat_schema": "not an %s object: the catalogue must be a mapping whose schema is that string",
    "cat_slot_order": "slot_order must be %s",
    "cat_slots": "slots must have exactly the keys of slot_order",
    "cat_no_sku": "slot %s has no SKU: it must be a list of at least one SKU record",
    "cat_record": "slot %s: every SKU record must be a mapping",
    "cat_codes": "slot %s: SKU codes missing or not unique (every record needs its own sku)",
    "cat_two_slots": "an SKU code appears in two slots",
    "rec_field": "%s %s: %s missing or invalid",
    "rec_qty": "%s %s: qty_per_car must be %d",
    "rec_lot": "%s %s: lot missing or not a lot id",
    "rec_no_lot": "%s %s: lot must be null (only the SKUs of battery, lidar and tyres carry a lot)",
    "rec_fit": "%s %s: fit_class missing",
    "rec_connector": "%s %s: connector_class missing",
    "rec_values": "%s %s: values missing",
    "rec_value": "%s %s: values.%s missing or invalid",
    "rec_ocv": "%s %s: ocv_soc must increase within [0, 1] and ocv_cell_v must not decrease, same length",
    "platform_missing": "platform record missing: the catalogue needs the mapping platform",
    "platform_field": "platform.%s missing or invalid",
    "platform_fit": "platform.fit must have exactly motor_mount, battery_bay, deck_socket, wheel_hub (class codes) "
                    "and pinion_teeth (a list of tooth counts)",
    "platform_shell": "platform: body_shell_mass_g and body_shell_thermal_factor come together or not at all",
    "platform_kit": "platform.kit_sku %s is also an SKU of a purchase line",
    "lots_missing": "lots missing: the catalogue needs the mapping lots",
    "lots_ids": "lots must list exactly the lot ids the SKUs carry",
    "lot_kind": "lot %s: kind must be %s",
    "lot_slot": "lot %s (a %s lot) is carried by an SKU of another slot",
    "lot_skus": "lot %s: skus must list exactly the SKUs that carry it",
    "bom_lines": "a BOM names exactly one SKU per purchase line of %s",
    "bom_sku": "%r is not an SKU of the line %s",
    # the car record
    "car_schema": "not an %s record: the car record must be a mapping whose schema is that string",
    "car_missing": "%s: fields missing: %s",
    "car_unknown": "%s: unknown fields: %s",
    "car_pos": "%s: %s must be a positive number",
    "car_nonneg": "%s: %s must be a number >= 0",
    "car_num": "%s: %s must be a finite number",
    "car_posint": "%s: %s must be an integer >= 1",
    "car_lock": "%s: %s must be a number above 0 and below 90",
    "car_bom": "%s: %s must map exactly the lines %s to SKU codes",
    "car_lines": "%s: %s must map exactly the lines %s to finite numbers",
    "car_as_found": "%s: as_found_bom names %r, which is not an SKU of the line %s",
    # decisions
    "comparator": "comparator must be <=, >= or ==",
    "margin_comparator": "a margin needs the comparator <= or >=",
    # lots
    "samples_number": "samples must be finite numbers",
    "samples_list": "samples must be a non-empty list of numbers",
    "lot_range": "%s = %s is outside its range %s",
    "tyre_samples": "tyre samples need pull_n for every ground class %s",
    "lidar_samples": "lidar samples need range_m for every target class %s",
    "lot_kind_name": "lot kind must be one of %s",
    "page_tyre_lot": "tyre lot %s: its SKUs print different mu_page values",
    "page_lidar_lot": "lidar lot %s: its SKUs print different ranges",
}

# ------------------------------------------------------------------------------------------------ enumerations
# The purchase lines (slots) of the catalogue. A car is one SKU per slot; this is the order of the columns of an
# index row and the order in which iter_cars counts.
SLOT_ORDER = ("motor", "pinion", "motor_controller", "battery", "lidar", "deck_post", "compute", "power_board",
              "tyres", "steering_servo")
AS_FOUND_SLOTS = tuple(s for s in SLOT_ORDER if s != "steering_servo")    # the lines of the car record's as_found_bom
SURFACES = ("S1", "S2", "P", "M")                                     # ground classes: the values of a surface key
TARGET_CLASSES = ("white", "grey", "black_duct", "dark_fabric")       # classes of a target the LiDAR has to detect
COMPARATORS = ("<=", ">=", "==")                                      # comparators of a requirement: see sat()
SEGMENT_KINDS = ("straight", "arc")                                   # kinds of a segment of a cycle
LOT_KINDS = ("cell", "tyre", "lidar")                                 # kinds of a lot
LOT_KIND_BY_LETTER = {"C": "cell", "T": "tyre", "L": "lidar"}         # the letter in a lot id gives the kind
LOT_SLOT = {"cell": "battery", "tyre": "tyres", "lidar": "lidar"}     # the slot whose SKUs carry the lots of a kind

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$"                     # ids of cycles and of scenarios
LOT_ID_PATTERN = r"^LOT-(C|T|L)[0-9]$"                                # lot ids
SKU_PATTERN = r"^[A-Z]{3}-[0-9A-HJ-NP-Z]{4}$"                         # the form of an SKU code

# ------------------------------------------------------------------------------------------------ language limits
# Limits of a context (quantities.md, "Decision function and limits").
MAX_SCENARIOS = 6               # scenarios of a context
MAX_CYCLES = 6                  # cycles of a context
MAX_SEGMENTS = 64               # segments of a cycle
MAX_CYCLE_LENGTH_M = 120.0      # total length of a cycle, m
MAX_BLOCKS = 16                 # expanded blocks of a schedule: len(blocks) x repeat
MAX_DISTINCT_PACES = 4          # distinct pace values of a schedule
MAX_SCHEDULE_S = 4 * 3600.0     # total drive + rest time of a schedule: four hours, in s
MAX_ABS_INT = 2 ** 53           # largest magnitude of an integer that is accepted as a number

# ------------------------------------------------------------------------------------------------ reference driver
# The constants of the physics. stack_notes.md prints the same values; quantities.md, "The reference driver and the
# constants", says where each one enters.
N_NODES = 400                       # a cycle is resampled to this many nodes, uniform in arc length
GRAVITY_MPS2 = 9.81                 # gravitational acceleration, m/s^2
LATERAL_UTILISATION = 0.85          # share of the pace-scaled friction the driver uses in a bend
BRAKE_UTILISATION = 0.8             # share of the full braking limit the driver plans a stop with
ACTUATION_LATENCY_S = 0.02          # command to wheel, in s; added to the scan period and the compute latency
MIN_BEAM_HITS = 3                   # beams that must fall on a target for it to count as detected
DRIVETRAIN_EFFICIENCY = 0.85        # motor shaft to tyre contact, both directions of power flow
ROLLING_RESISTANCE = 0.015          # rolling-resistance coefficient, the same on every ground class
DRAG_N_PER_MPS2 = 0.012             # aerodynamic drag force / speed squared (N per (m/s)^2)
PROFILE_CELL_V = 1.2                # per-cell voltage of the reference bus that speed profiles are solved at
RAIL_CONVERTER_EFFICIENCY = 0.9     # power board: input power = rail loads / this + its quiescent power
LINKAGE_EFFICIENCY = 0.80           # steering linkage: share of the servo's torque that reaches the two king pins

# ------------------------------------------------------------------------------------------------ dictionary
# The closed list of quantities a requirement can be written on. For each: the unit its value is returned in ("1" =
# a pure number) and its signature, the keys that the conditions of a request must have, exactly those ({} when the
# signature is empty). The condition keys: config = {body_shell: true or false}; surface = a ground class; target =
# {class, width_m}; scenario = the id of a scenario of the context; cell_v, current_a, v0_mps, v1_mps = numbers in
# the ranges of CONDITION_RANGES. quantities.md, "The dictionary", has the same list with every formula.
_A, _B, _C, _D, _E, _F = "A", "B", "C", "D", "E", "F"
QUANTITIES = {
    # A: the replica as built on the reference unit (quantities.md, "The replica" and "Fit")
    # mass of the replica with the pod, and with the body shell when config.body_shell is true; kg
    "mass_kg": {"group": _A, "unit": "kg", "signature": ("config",)},
    # static load on the front axle of the replica with the pod; N
    "front_axle_load_n": {"group": _A, "unit": "N", "signature": ()},
    # height of the scan plane of the LiDAR above the ground; mm
    "scan_plane_height_mm": {"group": _A, "unit": "mm", "signature": ()},
    # height of the lowest point of the replica above the ground, wheels and tyres left out of account; mm
    "ground_clearance_mm": {"group": _A, "unit": "mm", "signature": ()},
    # height of the highest point of the replica above the ground (the pod and the body shell do not enter); mm
    "envelope_height_mm": {"group": _A, "unit": "mm", "signature": ()},
    # 1 when the car can be built (its parts fit by class and the pack is no longer than the battery bay), else 0
    "fits": {"group": _A, "unit": "1", "signature": ()},
    # B: values of the parts (quantities.md, "The dictionary")
    # nominal voltage of the battery pack: cells x nominal voltage of a cell; V
    "battery_nominal_v": {"group": _B, "unit": "V", "signature": ()},
    # torque of the motor at the motor current current_a, its no-load current taken off; N.m
    "motor_torque_nm": {"group": _B, "unit": "N.m", "signature": ("current_a",)},
    # scans per second of the LiDAR; Hz
    "lidar_scan_rate_hz": {"group": _B, "unit": "Hz", "signature": ()},
    # angle between neighbouring beams of the LiDAR; deg
    "lidar_angular_step_deg": {"group": _B, "unit": "deg", "signature": ()},
    # price of one car at the quoted prices: the parts of every line plus the fixed cost of the chassis kit; USD
    "unit_cost_usd": {"group": _B, "unit": "USD", "signature": ()},
    # the largest latency the page of the compute module states; s
    "compute_latency_s": {"group": _B, "unit": "s", "signature": ()},
    # distance at which the LiDAR detects a target of the given class and width (its lot's range on that class, or
    # where MIN_BEAM_HITS beams still fall on the width, whichever is shorter); m
    "detection_range_m": {"group": _B, "unit": "m", "signature": ("target",)},
    # C: electrical compatibility (quantities.md, "The dictionary")
    # the lower of the maximum input voltages of the motor controller and of the power board, minus the pack voltage
    # at cell_v volts per cell; V
    "overvoltage_headroom_v": {"group": _C, "unit": "V", "signature": ("cell_v",)},
    # no-load speed of the motor at the pack voltage of cell_v volts per cell, over the maximum speed of the motor; 1
    "motor_overspeed_ratio": {"group": _C, "unit": "1", "signature": ("cell_v",)},
    # the highest ratio of load to rating among the supply rails that the LiDAR and the compute module use; 1
    "rail_load_ratio": {"group": _C, "unit": "1", "signature": ()},
    # D: driving on flat ground (quantities.md, "The reference driver and the constants")
    # distance of an emergency stop from the speed v0_mps on a ground class, the reaction latency included; m
    "stop_distance_m": {"group": _D, "unit": "m", "signature": ("surface", "v0_mps", "config")},
    # top speed at the pack voltage of cell_v volts per cell; m/s
    "top_speed_mps": {"group": _D, "unit": "m/s", "signature": ("cell_v", "config")},
    # time from rest to the speed v1_mps on a ground class at cell_v volts per cell; infinite when v1_mps is not
    # below the top speed; s
    "accel_time_s": {"group": _D, "unit": "s", "signature": ("surface", "v1_mps", "cell_v", "config")},
    # E: one scenario of the context (quantities.md, "The speed profile (one lap)" and "Over a schedule")
    # time of one lap of the scenario's cycle at the highest pace of its schedule; s
    "lap_time_s": {"group": _E, "unit": "s", "signature": ("scenario",)},
    # distance left when the car has stopped for the target of the scenario's zone (negative: it stops too late);
    # the scenario must have a zone; m
    "zone_stop_margin_m": {"group": _E, "unit": "m", "signature": ("scenario",)},
    # the lowest state of charge with which a pack ends the schedule, as a share of its capacity (negative: the
    # pack falls short by that much); 1
    "schedule_energy_reserve_frac": {"group": _E, "unit": "1", "signature": ("scenario",)},
    # the lowest bus voltage of the schedule, minus the minimum input voltage of the power board; V
    "schedule_bus_margin_v": {"group": _E, "unit": "V", "signature": ("scenario",)},
    # winding temperature limit of the motor, minus the peak winding temperature of the schedule; K
    "schedule_winding_headroom_c": {"group": _E, "unit": "K", "signature": ("scenario",)},
    # temperature limit of the motor controller, minus its peak temperature of the schedule; K
    "schedule_controller_headroom_c": {"group": _E, "unit": "K", "signature": ("scenario",)},
    # F: steering at standstill on a ground class (quantities.md, "Steering at standstill")
    # torque the steering servo must give to turn both front wheels of the standing replica; N.m
    "steering_torque_nm": {"group": _F, "unit": "N.m", "signature": ("surface",)},
    # that torque over the stall torque of the servo; 1
    "steering_torque_ratio": {"group": _F, "unit": "1", "signature": ("surface",)},
    # time from full lock to full lock against that torque; infinite when the torque reaches the stall torque; s
    "lock_time_s": {"group": _F, "unit": "s", "signature": ("surface",)},
}
SCENARIO_QUANTITIES = tuple(q for q, d in QUANTITIES.items() if d["group"] == _E)      # the six of group E
STEERING_QUANTITIES = tuple(q for q, d in QUANTITIES.items() if d["group"] == _F)      # the three of group F
QUANTITY_UNITS = {q: d["unit"] for q, d in QUANTITIES.items()}                         # quantity -> unit

# Ranges of the numbers of the condition language and of a context: key -> (low, high, low_open, high_open); an open
# end excludes the end value itself (quantities.md, "Conditions", "Cycles", "Scenarios and schedules").
CONDITION_RANGES = {
    "width_m": (0.0, 2.0, True, False),             # width of a target, m
    "speed_cap_mps": (0.0, 10.0, True, False),      # speed limit of a zone, m/s
    "appear_m": (0.0, 30.0, True, False),           # distance ahead at which the target of a zone appears, m
    "cell_v": (0.8, 1.6, False, False),             # voltage per cell of the pack, V
    "current_a": (0.0, 200.0, False, False),        # motor current, A
    "v0_mps": (0.0, 15.0, True, False),             # speed at which an emergency stop begins, m/s
    "v1_mps": (0.0, 15.0, True, False),             # speed at which an acceleration from rest ends, m/s
    "ambient_c": (-10.0, 50.0, False, False),       # ambient temperature of an environment, degrees Celsius
    "length_m": (0.0, 120.0, True, False),          # length of a straight, m
    "seg_width_m": (0.3, 10.0, True, False),        # lane width of a segment (the segment's key is width_m), m
    "sight_m": (0.0, 50.0, True, False),            # how far ahead the driver sees on a segment, m
    "radius_m": (0.2, 50.0, False, False),          # radius of an arc, m
    "angle_deg": (0.0, 360.0, True, False),         # angle of an arc, deg
    "drive_s": (0.0, 7200.0, True, False),          # driving time of a block, s
    "pace": (0.0, 1.0, True, False),                # pace of a block: scales the friction used and the top speed
    "rest_s": (0.0, 7200.0, False, False),          # standing time after the drive of a block, s
    "repeat": (1, 16, False, False),                # how many times the list of blocks is driven
    "packs": (1, 4, False, False),                  # battery packs a schedule may use
    "swap_after_blocks": (1, 16, False, False),     # expanded blocks driven on one pack before the next is fitted
}
# The fields of a lot-table entry per kind of lot, and the range of each value (quantities.md, "Lot table").
LOT_FIELDS = {"cell": ("capacity_factor", "resistance_factor"), "tyre": ("mu",), "lidar": ("range_m",)}
LOT_RANGES = {"capacity_factor": (0.5, 1.1, False, False), "resistance_factor": (0.9, 2.0, False, False),
              "mu": (0.2, 1.5, False, False), "range_m": (0.1, 30.0, False, False)}

# ------------------------------------------------------------------------------------------------ units
# convert() is there for a page that prints a value in another unit than the one its field is named for; the
# formulas of this file use the fields in their own units only. The six numbers below are exact by the definition of
# the unit (stack_notes.md lists them); the unit table is built from them.
INCH_M = 0.0254                     # one inch in metres
FOOT_M = 0.3048                     # one foot in metres
MILE_M = 1609.344                   # one mile in metres
POUND_KG = 0.45359237               # one pound in kilograms
STANDARD_GRAVITY_MPS2 = 9.80665     # one kilogram-force in newtons (for units only: the driver uses GRAVITY_MPS2)
CELSIUS_ZERO_K = 273.15             # zero degrees Celsius in kelvin
# A unit as a page may print it -> (dimension, factor to the base unit of the dimension). The temperatures are not
# in this table: _TEMPERATURES gives each as (scale, offset) to degrees Celsius, and convert() applies both.
UNIT_FACTORS = {
    "m": ("length", 1.0), "mm": ("length", 1e-3), "cm": ("length", 1e-2), "in": ("length", INCH_M),
    "ft": ("length", FOOT_M),
    "kg": ("mass", 1.0), "g": ("mass", 1e-3), "oz": ("mass", POUND_KG / 16.0), "lb": ("mass", POUND_KG),
    "s": ("time", 1.0), "ms": ("time", 1e-3), "us": ("time", 1e-6), "min": ("time", 60.0), "h": ("time", 3600.0),
    "Hz": ("rate", 1.0), "rps": ("rate", 1.0), "rpm": ("rate", 1.0 / 60.0), "krpm": ("rate", 1000.0 / 60.0),
    "rad/s": ("rate", 1.0 / (2.0 * math.pi)),
    "deg": ("angle", 1.0), "rad": ("angle", 180.0 / math.pi), "mrad": ("angle", 0.18 / math.pi),
    "arcmin": ("angle", 1.0 / 60.0),
    "V": ("voltage", 1.0), "mV": ("voltage", 1e-3),
    "A": ("current", 1.0), "mA": ("current", 1e-3),
    "Ohm": ("resistance", 1.0), "mOhm": ("resistance", 1e-3),
    "W": ("power", 1.0), "mW": ("power", 1e-3),
    "Ah": ("charge", 1.0), "mAh": ("charge", 1e-3), "C": ("charge", 1.0 / 3600.0),
    "N": ("force", 1.0), "kgf": ("force", STANDARD_GRAVITY_MPS2),
    "N.m": ("torque", 1.0), "mN.m": ("torque", 1e-3), "kgf.cm": ("torque", STANDARD_GRAVITY_MPS2 / 100.0),
    "oz.in": ("torque", POUND_KG / 16.0 * STANDARD_GRAVITY_MPS2 * INCH_M),
    "N.m/A": ("torque_constant", 1.0), "mN.m/A": ("torque_constant", 1e-3),
    "oz.in/A": ("torque_constant", POUND_KG / 16.0 * STANDARD_GRAVITY_MPS2 * INCH_M),
    "rpm/V": ("speed_constant", 1.0), "krpm/V": ("speed_constant", 1000.0),
    "K/W": ("thermal_resistance", 1.0), "degC/W": ("thermal_resistance", 1.0),
    "J/K": ("heat_capacity", 1.0), "J/degC": ("heat_capacity", 1.0),
    "m/s": ("speed", 1.0), "km/h": ("speed", 1.0 / 3.6), "mph": ("speed", MILE_M / 3600.0),
    "1": ("ratio", 1.0), "%": ("ratio", 0.01),
    "USD": ("money", 1.0),
}
_TEMPERATURES = {"degC": (1.0, 0.0), "K": (1.0, -CELSIUS_ZERO_K), "degF": (5.0 / 9.0, -32.0 * 5.0 / 9.0)}


def _dimension(unit):
    """The dimension of a unit: "temperature" for degC, K and degF, else the dimension UNIT_FACTORS gives."""
    if unit in _TEMPERATURES:
        return "temperature"
    if unit in UNIT_FACTORS:
        return UNIT_FACTORS[unit][0]
    raise ValueError(MESSAGES["unknown_unit"] % (unit,))


def convert(value, from_unit, to_unit):
    """The number `value`, given in from_unit, expressed in to_unit, for instance convert(x, "kgf.cm", "N.m"). Both
    units must be of the same dimension; the temperatures degC, K and degF convert with their offsets. ValueError
    for an unknown unit or for two different dimensions."""
    df, dt = _dimension(from_unit), _dimension(to_unit)
    if df != dt:
        raise ValueError(MESSAGES["cannot_convert"] % (from_unit, df, to_unit, dt))
    if df == "temperature":
        a, b = _TEMPERATURES[from_unit]
        c, d = _TEMPERATURES[to_unit]
        return ((float(value) * a + b) - d) / c
    return float(value) * UNIT_FACTORS[from_unit][1] / UNIT_FACTORS[to_unit][1]


# ================================================================================================ helpers
def _is_num(x):
    """True for a finite number: an int or a float, not a bool; an int of magnitude at most MAX_ABS_INT."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and (
        not isinstance(x, int) or abs(x) <= MAX_ABS_INT)


def _is_int(x):
    """True for an int (not a bool) of magnitude at most MAX_ABS_INT."""
    return isinstance(x, int) and not isinstance(x, bool) and abs(x) <= MAX_ABS_INT


def _in_range(x, key):
    """True when x is a number inside the range of key (CONDITION_RANGES, else LOT_RANGES)."""
    lo, hi, lo_open, hi_open = CONDITION_RANGES.get(key) or LOT_RANGES[key]
    if not _is_num(x):
        return False
    if (x <= lo) if lo_open else (x < lo):
        return False
    if (x >= hi) if hi_open else (x > hi):
        return False
    return True


def _range_text(key):
    """The range of key as an interval for a message: a round bracket at an open end, a square one at a closed end."""
    lo, hi, lo_open, hi_open = CONDITION_RANGES.get(key) or LOT_RANGES[key]
    return MESSAGES["range"] % ("(" if lo_open else "[", lo, hi, ")" if hi_open else "]")


def canonical(obj):
    """Canonical JSON text of a JSON-able object (sorted keys, compact): equal objects give equal text. A number
    that is not finite is refused (ValueError)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


# ================================================================================================ language rules
# The structure of conditions, cycles, scenarios and lot-table entries (quantities.md, "Conditions", "Cycles",
# "Scenarios and schedules", "Lot table"). A _check_* function appends every problem of one object to the list p, as
# a text of MESSAGES; `where` is the place the text begins with. No function here judges engineering sense.
def _check_config(c, where, p):
    """A configuration is {body_shell: true or false}: whether the body shell is fitted."""
    if not isinstance(c, dict) or set(c) != {"body_shell"}:
        p.append(MESSAGES["config_keys"] % where)
        return
    if not isinstance(c["body_shell"], bool):
        p.append(MESSAGES["config_shell"] % where)


def _check_target(t, where, p):
    """A target is {class: one of TARGET_CLASSES, width_m}: what the LiDAR has to detect."""
    if not isinstance(t, dict) or set(t) != {"class", "width_m"}:
        p.append(MESSAGES["target_keys"] % where)
        return
    if t["class"] not in TARGET_CLASSES:
        p.append(MESSAGES["target_class"] % (where, ", ".join(TARGET_CLASSES)))
    if not _in_range(t["width_m"], "width_m"):
        p.append(MESSAGES["target_width"] % (where, _range_text("width_m")))


def _check_zone(z, where, p):
    """A zone is null or {speed_cap_mps, appear_m, surface, target}: the one emergency-stop check of a scenario (the
    car at speed_cap_mps, or at its top speed if that is lower, on that ground class, when the target appears
    appear_m ahead). It enters zone_stop_margin_m and nothing else."""
    if z is None:
        return
    if not isinstance(z, dict) or set(z) != {"speed_cap_mps", "appear_m", "surface", "target"}:
        p.append(MESSAGES["zone_keys"] % where)
        return
    for k in ("speed_cap_mps", "appear_m"):
        if not _in_range(z[k], k):
            p.append(MESSAGES["zone_number"] % (where, k, _range_text(k)))
    if z["surface"] not in SURFACES:
        p.append(MESSAGES["zone_surface"] % (where, ", ".join(SURFACES)))
    _check_target(z["target"], MESSAGES["where_zone"] % where, p)


def _segment_length(s):
    """Length of a segment in m: length_m of a straight, radius_m x angle_deg (in radians) of an arc."""
    if s["kind"] == "straight":
        return float(s["length_m"])
    return float(s["radius_m"]) * float(s["angle_deg"]) * math.pi / 180.0


def _check_cycle(cid, cyc, p):
    """A cycle is {segments: [...]}, driven as a closed loop. A segment is {kind: straight, length_m, width_m,
    surface, sight_m} or {kind: arc, radius_m, angle_deg, width_m, surface, sight_m}: width_m the lane width, surface
    a ground class, sight_m how far ahead the driver sees. The number of segments and the total length are limited
    (MAX_SEGMENTS, MAX_CYCLE_LENGTH_M)."""
    where = MESSAGES["where_cycle"] % (cid,)
    if not isinstance(cyc, dict) or set(cyc) != {"segments"}:
        p.append(MESSAGES["cycle_keys"] % where)
        return
    segs = cyc["segments"]
    if not isinstance(segs, list) or not 1 <= len(segs) <= MAX_SEGMENTS:
        p.append(MESSAGES["cycle_segments"] % (where, MAX_SEGMENTS))
        return
    ok = True
    for i, s in enumerate(segs):
        w = MESSAGES["where_segment"] % (where, i)
        if not isinstance(s, dict) or s.get("kind") not in SEGMENT_KINDS:
            p.append(MESSAGES["segment_kind"] % w)
            ok = False
            continue
        keys = {"kind", "width_m", "surface", "sight_m"} | ({"length_m"} if s["kind"] == "straight"
                                                             else {"radius_m", "angle_deg"})
        if set(s) != keys:
            p.append(MESSAGES["segment_keys"] % (w, s["kind"], ", ".join(sorted(keys))))
            ok = False
            continue
        for k, rk in (("length_m", "length_m"), ("radius_m", "radius_m"), ("angle_deg", "angle_deg"),
                      ("width_m", "seg_width_m"), ("sight_m", "sight_m")):
            if k in s and not _in_range(s[k], rk):
                p.append(MESSAGES["segment_number"] % (w, k, _range_text(rk)))
                ok = False
        if s["surface"] not in SURFACES:
            p.append(MESSAGES["segment_surface"] % (w, ", ".join(SURFACES)))
            ok = False
    if ok:
        total = math.fsum(_segment_length(s) for s in segs)
        if total > MAX_CYCLE_LENGTH_M:
            p.append(MESSAGES["cycle_length"] % (where, total, MAX_CYCLE_LENGTH_M))


def expand_blocks(schedule):
    """The blocks of a schedule in driving order: the list blocks, repeat times over (the expanded blocks)."""
    return list(schedule["blocks"]) * int(schedule["repeat"])


def _check_schedule(sch, where, p):
    """A schedule is {blocks: [{drive_s, pace, rest_s}, ...], repeat, packs, swap_after_blocks}: in each block the
    car laps the cycle for drive_s seconds at the block's pace, then stands for rest_s seconds. swap_after_blocks is
    null (every block on the first pack) or the number of expanded blocks after which the next pack is fitted. The
    expanded blocks, the distinct paces and the total time are limited (MAX_BLOCKS, MAX_DISTINCT_PACES,
    MAX_SCHEDULE_S)."""
    keys = {"blocks", "repeat", "packs", "swap_after_blocks"}
    if not isinstance(sch, dict) or set(sch) != keys:
        p.append(MESSAGES["schedule_keys"] % (where, ", ".join(sorted(keys))))
        return
    blocks = sch["blocks"]
    ok = True
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= MAX_BLOCKS:
        p.append(MESSAGES["schedule_blocks"] % (where, MAX_BLOCKS))
        ok = False
    else:
        for i, b in enumerate(blocks):
            if not isinstance(b, dict) or set(b) != {"drive_s", "pace", "rest_s"}:
                p.append(MESSAGES["block_keys"] % (where, i))
                ok = False
                continue
            for k in ("drive_s", "pace", "rest_s"):
                if not _in_range(b[k], k):
                    p.append(MESSAGES["block_number"] % (where, i, k, _range_text(k)))
                    ok = False
    for k in ("repeat", "packs"):
        if not _is_int(sch[k]) or not _in_range(sch[k], k):
            p.append(MESSAGES["schedule_integer"] % (where, k, _range_text(k)))
            ok = False
    s = sch["swap_after_blocks"]
    if s is not None and (not _is_int(s) or not _in_range(s, "swap_after_blocks")):
        p.append(MESSAGES["schedule_swap"] % (where, _range_text("swap_after_blocks")))
    if not ok:
        return
    exp = expand_blocks(sch)
    if len(exp) > MAX_BLOCKS:
        p.append(MESSAGES["schedule_expanded"] % (where, len(exp), MAX_BLOCKS))
    if len({float(b["pace"]) for b in exp}) > MAX_DISTINCT_PACES:
        p.append(MESSAGES["schedule_paces"] % (where, MAX_DISTINCT_PACES))
    total = math.fsum(float(b["drive_s"]) + float(b["rest_s"]) for b in exp)
    if total > MAX_SCHEDULE_S:
        p.append(MESSAGES["schedule_time"] % (where, total, MAX_SCHEDULE_S))


def _check_environment(env, where, p):
    """An environment is {ambient_c, config, zone}: the ambient temperature in degrees Celsius, the configuration of
    the car and the zone (or null) of a scenario."""
    if not isinstance(env, dict) or set(env) != {"ambient_c", "config", "zone"}:
        p.append(MESSAGES["environment_keys"] % where)
        return
    if not _in_range(env["ambient_c"], "ambient_c"):
        p.append(MESSAGES["environment_ambient"] % (where, _range_text("ambient_c")))
    _check_config(env["config"], MESSAGES["where_environment"] % where, p)
    _check_zone(env["zone"], MESSAGES["where_environment"] % where, p)


def _scenario_problems(sid, sc, cycles):
    """The problems of one scenario {cycle, schedule, environment} as a list of texts: cycle must be the id of a
    cycle in `cycles`; the schedule and the environment are checked as above."""
    p = []
    where = MESSAGES["where_scenario"] % (sid,)
    if not isinstance(sc, dict) or set(sc) != {"cycle", "schedule", "environment"}:
        p.append(MESSAGES["scenario_keys"] % where)
        return p
    if not isinstance(sc["cycle"], str) or sc["cycle"] not in cycles:
        p.append(MESSAGES["scenario_cycle"] % (where, sc["cycle"]))
    _check_schedule(sc["schedule"], where, p)
    _check_environment(sc["environment"], where, p)
    return p


def _check_lot(lid, val, p):
    """An entry of the lot table. The letter in the lot id gives the kind: a cell lot is {capacity_factor,
    resistance_factor}, a tyre lot {mu: {ground class: friction}} with all four ground classes, a LiDAR lot
    {range_m: {target class: range in m}} with all four target classes; every value inside LOT_RANGES."""
    m = re.match(LOT_ID_PATTERN, lid) if isinstance(lid, str) else None
    if not m:
        p.append(MESSAGES["lot_id"] % (lid,))
        return
    kind = LOT_KIND_BY_LETTER[m.group(1)]
    where = MESSAGES["where_lot"] % lid
    if kind == "cell":
        if not isinstance(val, dict) or set(val) != {"capacity_factor", "resistance_factor"}:
            p.append(MESSAGES["cell_lot_keys"] % where)
            return
        for k in ("capacity_factor", "resistance_factor"):
            if not _in_range(val[k], k):
                p.append(MESSAGES["lot_number"] % (where, k, _range_text(k)))
        return
    field, keys = ("mu", SURFACES) if kind == "tyre" else ("range_m", TARGET_CLASSES)
    if not isinstance(val, dict) or set(val) != {field} or not isinstance(val[field], dict) \
            or set(val[field]) != set(keys):
        p.append(MESSAGES["lot_keys"] % (where, kind, field, ", ".join(keys)))
        return
    for k in keys:
        if not _in_range(val[field][k], field):
            p.append(MESSAGES["lot_field_number"] % (where, field, k, _range_text(field)))


def _check_conditions(rid, q, cond, scenarios, p):
    """The conditions of a request for the quantity q: a mapping with exactly the keys of the quantity's signature,
    each value of its kind and inside its range; a scenario key must be the id of a scenario of the context, and
    for zone_stop_margin_m of one that has a zone."""
    sig = QUANTITIES[q]["signature"]
    if not isinstance(cond, dict) or set(cond) != set(sig):
        p.append(MESSAGES["conditions_keys"] % (rid, q, ", ".join(sig)))
        return
    where = MESSAGES["where_conditions"] % rid
    for k in sig:
        v = cond[k]
        if k == "config":
            _check_config(v, where, p)
        elif k == "target":
            _check_target(v, where, p)
        elif k == "surface":
            if v not in SURFACES:
                p.append(MESSAGES["conditions_surface"] % (where, ", ".join(SURFACES)))
        elif k == "scenario":
            if not isinstance(v, str) or v not in scenarios:
                p.append(MESSAGES["conditions_scenario"] % (where, v))
            elif q == "zone_stop_margin_m" and isinstance(scenarios[v], dict) and isinstance(
                    scenarios[v].get("environment"), dict) and scenarios[v]["environment"].get("zone") is None:
                p.append(MESSAGES["conditions_zone"] % where)
        elif not _in_range(v, k):
            p.append(MESSAGES["conditions_number"] % (where, k, _range_text(k)))


def validate_context(context, lot_ids=None):
    """The problems of a context {lot_table, cycles, scenarios} as a list of texts; an empty list means that the
    structure is valid. With lot_ids (for instance the lot_ids of a loaded catalogue) the lot table must have
    exactly those keys. Only the structure is checked: nothing here says whether the numbers make engineering
    sense. quantities.md, "Lot table", "Cycles" and "Scenarios and schedules", describes the three parts."""
    if not isinstance(context, dict) or not {"lot_table", "cycles", "scenarios"} <= set(context):
        return [MESSAGES["context_keys"]]
    p = []
    lt, cycles, scenarios = context["lot_table"], context["cycles"], context["scenarios"]
    if not isinstance(lt, dict):
        p.append(MESSAGES["lot_table_type"])
    else:
        if lot_ids is not None and set(lt) != set(lot_ids):
            p.append(MESSAGES["lot_table_ids"] % ", ".join(sorted(lot_ids)))
        for lid in sorted(lt, key=str):
            _check_lot(lid, lt[lid], p)
    if not isinstance(cycles, dict) or len(cycles) > MAX_CYCLES:
        p.append(MESSAGES["cycles_type"] % MAX_CYCLES)
        cycles = {}
    for cid in cycles:
        if not isinstance(cid, str) or not re.match(ID_PATTERN, cid):
            p.append(MESSAGES["cycle_id"] % (cid, ID_PATTERN))
        _check_cycle(cid, cycles[cid], p)
    if not isinstance(scenarios, dict) or len(scenarios) > MAX_SCENARIOS:
        p.append(MESSAGES["scenarios_type"] % MAX_SCENARIOS)
        scenarios = {}
    for sid, sc in scenarios.items():
        if not isinstance(sid, str) or not re.match(ID_PATTERN, sid):
            p.append(MESSAGES["scenario_id"] % (sid, ID_PATTERN))
        p.extend(_scenario_problems(sid, sc, cycles))
    return p


# ================================================================================================ catalogue
# The page fields (the `values` of an SKU record) that the procedure reads, per slot, each in the unit that ends
# its name; where a page prints another unit, convert first. More fields may be present and are ignored. The kinds:
# pos = number above 0, nonneg = number >= 0, num = finite number, posint = integer >= 1, vec3 = list of three
# numbers, str = non-empty string, list = list of two or more numbers, rails = {rail name: {voltage_v, rating_w}}.
# Fields whose name does not say it all: rth_wc / rth_ca = thermal resistance winding to case / case to ambient;
# cth_w / cth_c = heat capacity of the winding / of the case; ocv_soc and ocv_cell_v = the open-circuit voltage of
# one cell against the state of charge; box_mm = [length, width, overall height]; optical_offset_mm = height of the
# scan plane above the underside of the LiDAR's base; length_mm of a battery = overall length of the pack, end caps
# included; od_mm = outside diameter of the tyre; no_load_s_per_60deg = time the unloaded servo takes to turn its
# output shaft through 60 degrees; case_mm = [length, width, height] of the servo's case. A mass_g is the mass of
# one piece.
# range_white_m, range_dark_m and mu_page are the values a page prints: the quantities take range and friction from
# the lot table, and page_lot_table() makes a lot table of these page values. quantities.md, "The catalogue".
SLOT_FIELDS = {
    "motor": {"mass_g": "pos", "kv_rpm_per_v": "pos", "kt_nm_per_a": "pos", "resistance_ohm": "pos",
              "no_load_current_a": "nonneg", "max_speed_rpm": "pos", "winding_limit_c": "num",
              "rth_wc_k_per_w": "pos", "rth_ca_k_per_w": "pos", "cth_w_j_per_k": "pos", "cth_c_j_per_k": "pos"},
    "pinion": {"teeth": "posint", "mass_g": "pos"},
    "motor_controller": {"mass_g": "pos", "input_abs_max_v": "pos", "current_limit_a": "pos",
                         "on_resistance_ohm": "nonneg", "rth_k_per_w": "pos", "cth_j_per_k": "pos",
                         "temp_limit_c": "num", "quiescent_w": "nonneg"},
    "battery": {"mass_g": "pos", "length_mm": "pos", "cells": "posint", "cell_nominal_v": "pos", "capacity_ah": "pos",
                "cell_resistance_ohm": "nonneg", "ocv_soc": "list", "ocv_cell_v": "list"},
    "lidar": {"mass_g": "pos", "box_mm": "vec3", "optical_offset_mm": "num", "range_white_m": "pos",
              "range_dark_m": "pos", "scan_rate_hz": "pos", "angular_step_deg": "pos", "power_w": "nonneg",
              "supply_rail": "str"},
    "deck_post": {"length_mm": "pos", "mass_g": "pos"},
    "compute": {"mass_g": "pos", "box_mm": "vec3", "latency_max_s": "nonneg", "power_w": "nonneg",
                "supply_rail": "str"},
    "power_board": {"mass_g": "pos", "input_min_v": "nonneg", "input_max_v": "pos", "rails": "rails",
                    "quiescent_w": "nonneg"},
    "tyres": {"mass_g": "pos", "od_mm": "pos", "width_mm": "pos", "mu_page": "pos"},
    "steering_servo": {"stall_torque_nm": "pos", "no_load_s_per_60deg": "pos", "mass_g": "pos", "case_mm": "vec3"},
}
LOT_SLOTS = ("battery", "lidar", "tyres")        # the SKUs of these slots carry a lot id; elsewhere lot is null
SETS_OF_FOUR = ("deck_post", "tyres")            # these lines fit four parts per car, every other line one
# slot -> the key of platform.fit that holds the fit class the chassis kit takes in that place
FIT_SLOTS = {"motor": "motor_mount", "battery": "battery_bay", "deck_post": "deck_socket", "tyres": "wheel_hub"}
# pairs of slots whose two parts must have the same connector_class
CONNECTOR_PAIRS = (("battery", "power_board"), ("motor_controller", "motor"))
# The platform record describes the chassis kit, which is on every car: kit_sku its code, fixed_cost_usd its price
# per car, fit its fit classes and the pinion tooth counts that its motor mount reaches.
PLATFORM_FIELDS = {"kit_sku": "str", "fixed_cost_usd": "nonneg"}
# The body shell, in the platform record only when a configuration fits it: its mass, and the factor on the thermal
# resistances to ambient of the motor and of the motor controller while it is fitted.
SHELL_FIELDS = {"body_shell_mass_g": "nonneg", "body_shell_thermal_factor": "pos"}


def _field_ok(v, kind):
    """True when v is a value of one of the kinds pos, nonneg, num, posint, vec3 or str (see SLOT_FIELDS); the
    kinds list and rails are checked where a record is checked."""
    if kind == "pos":
        return _is_num(v) and v > 0
    if kind == "nonneg":
        return _is_num(v) and v >= 0
    if kind == "num":
        return _is_num(v)
    if kind == "posint":
        return _is_int(v) and v >= 1
    if kind == "vec3":
        return isinstance(v, list) and len(v) == 3 and all(_is_num(x) for x in v)
    if kind == "str":
        return isinstance(v, str) and len(v) > 0
    return True


class Catalogue:
    """A checked rc-catalogue/2 object: {schema, slot_order, slots: {slot: [SKU record, ...]}, platform, lots: {lot
    id: {kind, skus}}}. An SKU record has sku, qty_per_car, pack_size, pack_price_usd, lot, values (the page fields
    of SLOT_FIELDS) and, where its slot needs them, fit_class and connector_class. ValueError with the first problem
    found. Within a slot the SKUs are sorted by code; the position in that order is the SKU index of iter_cars.

    The slots are held as records[slot], codes[slot], index[slot] (SKU code -> SKU index) and f[slot] (one array
    per numeric page field, by SKU index, plus qty and unit_price = pack_price_usd / pack_size). sizes are the
    numbers of SKUs per slot in SLOT_ORDER, n_combinations their product; platform, lots and lot_ids hold the rest.
    quantities.md, "The catalogue"."""

    def __init__(self, obj):
        if not isinstance(obj, dict) or obj.get("schema") != CATALOGUE_SCHEMA:
            raise ValueError(MESSAGES["cat_schema"] % CATALOGUE_SCHEMA)
        order = obj.get("slot_order")
        if not isinstance(order, list) or tuple(order) != SLOT_ORDER:
            raise ValueError(MESSAGES["cat_slot_order"] % (SLOT_ORDER,))
        slots = obj.get("slots")
        if not isinstance(slots, dict) or set(slots) != set(SLOT_ORDER):
            raise ValueError(MESSAGES["cat_slots"])
        self.records, self.codes, self.index = {}, {}, {}
        for slot in SLOT_ORDER:
            recs = slots[slot]
            if not isinstance(recs, list) or not recs:
                raise ValueError(MESSAGES["cat_no_sku"] % slot)
            if not all(isinstance(r, dict) for r in recs):
                raise ValueError(MESSAGES["cat_record"] % slot)
            recs = sorted(recs, key=lambda r: str(r.get("sku")))
            codes = [r.get("sku") for r in recs]
            if len(set(codes)) != len(codes) or not all(isinstance(c, str) for c in codes):
                raise ValueError(MESSAGES["cat_codes"] % slot)
            for r in recs:
                self._check_record(slot, r)
            self.records[slot], self.codes[slot] = recs, codes
            self.index[slot] = {c: i for i, c in enumerate(codes)}
        all_codes = [c for s in SLOT_ORDER for c in self.codes[s]]
        if len(set(all_codes)) != len(all_codes):
            raise ValueError(MESSAGES["cat_two_slots"])
        self.sizes = tuple(len(self.codes[s]) for s in SLOT_ORDER)
        self.n_combinations = int(np.prod(self.sizes, dtype=np.int64))
        self._load_platform(obj.get("platform"), set(all_codes))
        self._load_lots(obj.get("lots"))
        self._build_arrays()

    @staticmethod
    def _check_record(slot, r):
        """One SKU record of a slot: the fields every record has, its lot, its classes, the page fields of the slot."""
        sku = r.get("sku")
        for k, kind in (("qty_per_car", "posint"), ("pack_size", "posint"), ("pack_price_usd", "pos")):
            if not _field_ok(r.get(k), kind):
                raise ValueError(MESSAGES["rec_field"] % (slot, sku, k))
        qty = 4 if slot in SETS_OF_FOUR else 1
        if r["qty_per_car"] != qty:
            raise ValueError(MESSAGES["rec_qty"] % (slot, sku, qty))
        if slot in LOT_SLOTS:
            if not (isinstance(r.get("lot"), str) and re.match(LOT_ID_PATTERN, r["lot"])):
                raise ValueError(MESSAGES["rec_lot"] % (slot, sku))
        elif r.get("lot") is not None:
            raise ValueError(MESSAGES["rec_no_lot"] % (slot, sku))
        if slot in FIT_SLOTS and not isinstance(r.get("fit_class"), str):
            raise ValueError(MESSAGES["rec_fit"] % (slot, sku))
        if any(slot in pair for pair in CONNECTOR_PAIRS) and not isinstance(r.get("connector_class"), str):
            raise ValueError(MESSAGES["rec_connector"] % (slot, sku))
        vals = r.get("values")
        if not isinstance(vals, dict):
            raise ValueError(MESSAGES["rec_values"] % (slot, sku))
        for f, kind in SLOT_FIELDS[slot].items():
            v = vals.get(f)
            if kind == "list":
                ok = isinstance(v, list) and len(v) >= 2 and all(_is_num(x) for x in v)
            elif kind == "rails":
                ok = isinstance(v, dict) and len(v) >= 1 and all(
                    isinstance(k, str) and isinstance(x, dict) and _field_ok(x.get("rating_w"), "pos")
                    and _field_ok(x.get("voltage_v"), "pos") for k, x in v.items())
            else:
                ok = _field_ok(v, kind)
            if not ok:
                raise ValueError(MESSAGES["rec_value"] % (slot, sku, f))
        if slot == "battery":
            s, v = vals["ocv_soc"], vals["ocv_cell_v"]
            if len(s) != len(v) or any(b <= a for a, b in zip(s, s[1:])) or s[0] < 0 or s[-1] > 1 \
                    or any(b < a for a, b in zip(v, v[1:])):
                raise ValueError(MESSAGES["rec_ocv"] % (slot, sku))

    def _load_platform(self, pf, codes):
        """The platform record: kit_sku (not a code of a purchase line), fixed_cost_usd, fit, and the two body shell
        fields, both or neither."""
        if not isinstance(pf, dict):
            raise ValueError(MESSAGES["platform_missing"])
        for f, kind in PLATFORM_FIELDS.items():
            if not _field_ok(pf.get(f), kind):
                raise ValueError(MESSAGES["platform_field"] % f)
        if pf["kit_sku"] in codes:
            raise ValueError(MESSAGES["platform_kit"] % pf["kit_sku"])
        fit = pf.get("fit")
        keys = tuple(FIT_SLOTS.values())
        if not (isinstance(fit, dict) and set(fit) == set(keys) | {"pinion_teeth"}
                and all(isinstance(fit[k], str) for k in keys) and isinstance(fit["pinion_teeth"], list)
                and all(_is_int(t) and t >= 1 for t in fit["pinion_teeth"])):
            raise ValueError(MESSAGES["platform_fit"])
        given = [f for f in SHELL_FIELDS if f in pf]
        if given and len(given) != len(SHELL_FIELDS):
            raise ValueError(MESSAGES["platform_shell"])
        for f in given:
            if not _field_ok(pf[f], SHELL_FIELDS[f]):
                raise ValueError(MESSAGES["platform_field"] % f)
        self.has_shell = bool(given)
        self.platform = pf

    def _load_lots(self, lots):
        """lots must hold exactly the lot ids that the SKUs carry, each with its kind and the list of its SKUs."""
        if not isinstance(lots, dict):
            raise ValueError(MESSAGES["lots_missing"])
        used = {}
        for slot in LOT_SLOTS:
            for r in self.records[slot]:
                used.setdefault(r["lot"], []).append((slot, r["sku"]))
        if set(lots) != set(used):
            raise ValueError(MESSAGES["lots_ids"])
        for lid, rec in lots.items():
            kind = LOT_KIND_BY_LETTER[re.match(LOT_ID_PATTERN, lid).group(1)]
            if not isinstance(rec, dict) or rec.get("kind") != kind:
                raise ValueError(MESSAGES["lot_kind"] % (lid, kind))
            if any(slot != LOT_SLOT[kind] for slot, _ in used[lid]):
                raise ValueError(MESSAGES["lot_slot"] % (lid, kind))
            skus = rec.get("skus")
            if not isinstance(skus, list) or sorted(map(str, skus)) != sorted(s for _, s in used[lid]):
                raise ValueError(MESSAGES["lot_skus"] % lid)
        self.lots = lots
        self.lot_ids = tuple(sorted(lots))

    def _build_arrays(self):
        """The arrays that the block functions index by SKU index: the page fields of every slot, the rail ratings of
        the power boards, and the tables of what fits what."""
        self.f = {}
        for slot in SLOT_ORDER:
            d = {}
            for fld, kind in SLOT_FIELDS[slot].items():
                vals = [r["values"][fld] for r in self.records[slot]]
                if kind in ("pos", "nonneg", "num", "posint"):
                    d[fld] = np.array(vals, dtype=np.float64)
                elif kind == "vec3":
                    d[fld] = np.array(vals, dtype=np.float64).reshape(-1, 3)
            recs = self.records[slot]
            d["qty"] = np.array([r["qty_per_car"] for r in recs], dtype=np.float64)
            d["unit_price"] = np.array([r["pack_price_usd"] / r["pack_size"] for r in recs], dtype=np.float64)
            self.f[slot] = d
        # supply rails: one column per rail name; rail_rating[power board, rail] in W, 0 where a board lacks the rail
        names = sorted({k for r in self.records["power_board"] for k in r["values"]["rails"]}
                       | {r["values"]["supply_rail"] for s in ("lidar", "compute") for r in self.records[s]})
        rid = {n: i for i, n in enumerate(names)}
        rating = np.zeros((len(self.records["power_board"]), len(names)))
        for i, r in enumerate(self.records["power_board"]):
            for k, x in r["values"]["rails"].items():
                rating[i, rid[k]] = float(x["rating_w"])
        self.rail_names = tuple(names)
        self.rail_rating = rating
        for s in ("lidar", "compute"):
            self.f[s]["rail_id"] = np.array([rid[r["values"]["supply_rail"]] for r in self.records[s]], dtype=np.intp)
        # fit_ok[slot][SKU index]: the SKU is of the fit class the chassis kit takes; for the pinion: its tooth count
        # is one the motor mount reaches. conn_ok[(a, b)][SKU of a, SKU of b]: the two share a connector class.
        # rail_ok[slot][power board, SKU]: the board has the supply rail of that LiDAR or compute module.
        fit = self.platform["fit"]
        self.fit_ok = {}
        for slot, key in FIT_SLOTS.items():
            self.fit_ok[slot] = np.array([r["fit_class"] == fit[key] for r in self.records[slot]], dtype=bool)
        self.fit_ok["pinion"] = np.array([int(r["values"]["teeth"]) in fit["pinion_teeth"]
                                          for r in self.records["pinion"]], dtype=bool)
        self.conn_ok = {}
        for a, b in CONNECTOR_PAIRS:
            self.conn_ok[(a, b)] = np.array([[ra["connector_class"] == rb["connector_class"]
                                              for rb in self.records[b]] for ra in self.records[a]], dtype=bool)
        self.rail_ok = {s: rating[:, self.f[s]["rail_id"]] > 0 for s in ("lidar", "compute")}   # (boards, SKUs)
        self.ocv = [(np.array(r["values"]["ocv_soc"], dtype=np.float64),
                     np.array(r["values"]["ocv_cell_v"], dtype=np.float64)) for r in self.records["battery"]]
        self.lot_of = {s: [r["lot"] for r in self.records[s]] for s in LOT_SLOTS}

    def car_index(self, bom):
        """The index row of a BOM: the SKU indices of bom = {slot: SKU code} in SLOT_ORDER, an int16 array with one
        entry per slot. ValueError when bom does not name exactly one SKU of every slot."""
        if not isinstance(bom, dict) or set(bom) != set(SLOT_ORDER):
            raise ValueError(MESSAGES["bom_lines"] % (SLOT_ORDER,))
        out = []
        for s in SLOT_ORDER:
            if not isinstance(bom[s], str) or bom[s] not in self.index[s]:
                raise ValueError(MESSAGES["bom_sku"] % (bom[s], s))
            out.append(self.index[s][bom[s]])
        return np.array(out, dtype=np.int16)

    def car_of(self, row):
        """The BOM {slot: SKU code} of one index row (the inverse of car_index)."""
        return {s: self.codes[s][int(row[i])] for i, s in enumerate(SLOT_ORDER)}


def load_catalogue(obj):
    """A Catalogue from an rc-catalogue/2 object, for instance a parsed JSON file (an existing Catalogue is returned
    as it is). The values in it are the numbers the procedure computes with: a page that has an erratum is entered
    at its corrected value. ValueError when the object is not a valid catalogue."""
    return obj if isinstance(obj, Catalogue) else Catalogue(obj)


# ================================================================================================ the car record
# The fields of rc-car/1, the car record of the reference unit, and the kind of value each must hold (pos = number
# above 0, nonneg = number >= 0, num = finite number, posint = integer >= 1, lock = number above 0 and below 90; bom
# and lines are the two mappings). Every field is required and no other field is allowed. A height is a height above
# the ground the tyres stand on, taken on the reference unit as it was found: on its own tyres and deck posts,
# without the pod. x runs forward from the rear axle. The comments below are short: Appendix D of the brief gives
# the exact definition of every field, and quantities.md, "The car record", says which formulas a field enters.
CAR_FIELDS = {
    "schema": "schema",                     # the string rc-car/1
    "as_found_mass_g": "pos",               # mass of the reference unit as weighed, the whole car on the scale; g
    "as_found_front_axle_g": "pos",         # reading of the scale under its two front wheels, the car level; g
    "as_found_bom": "bom",                  # {line: SKU code} of the parts it was found with, all lines but the servo
    "as_found_servo_mass_g": "pos",         # mass of the steering servo it was found with (not a catalogue item); g
    "wheelbase_mm": "pos",                  # distance along x between the front and the rear wheel centres; mm
    "line_x_mm": "lines",                   # {line: x at which the mass of the line's part acts}, every line; mm
    "lidar_seat_height_mm": "num",          # height of the face on which the base of the LiDAR stands; mm
    "lowest_point_height_mm": "num",        # height of the lowest point, wheels and tyres left out of account; mm
    "chassis_top_height_mm": "num",         # highest point of what the chassis carries and the deck does not; mm
    "deck_load_top_height_mm": "num",       # highest point of what the deck carries, the LiDAR left out; mm
    "battery_bay_length_mm": "pos",         # usable length of the battery bay along x; mm
    "steering_lock_deg": "lock",            # angle a front wheel turns from straight ahead to full lock; deg
    "kingpin_ground_offset_mm": "num",      # king-pin offset of a front wheel at the ground; mm
    "steering_arm_mm": "pos",               # steering arm of a front knuckle: king-pin axis to tie-rod joint; mm
    "bellcrank_output_arm_mm": "pos",       # pivot of a steering bellcrank to its tie-rod joint; mm
    "bellcrank_input_arm_mm": "pos",        # pivot of a steering bellcrank to its drag-link joint; mm
    "servo_saver_arm_mm": "pos",            # axis of the servo's output shaft to the servo-link joint; mm
    "spur_teeth": "posint",                 # teeth of the spur gear that the motor pinion drives
    "gearbox_ratio": "pos",                 # bevel stage of a gearbox: ring gear teeth / bevel pinion teeth; 1
    "pod_mass_g": "nonneg",                 # mass of the pod; g
    "pod_x_mm": "num",                      # x of the centre of the pod in its place on the car; mm
}
_CAR_KIND_MESSAGE = {"pos": "car_pos", "nonneg": "car_nonneg", "num": "car_num"}


def load_car(obj):
    """A checked copy of an rc-car/1 record (CAR_FIELDS): every field present, no other field, each value of its
    kind. Numbers come back as float, spur_teeth as int, the two mappings as new dicts. ValueError with the first
    problem otherwise. Whether the codes of as_found_bom are SKUs of the catalogue is checked by prepare()."""
    if not isinstance(obj, dict) or obj.get("schema") != CAR_SCHEMA:
        raise ValueError(MESSAGES["car_schema"] % CAR_SCHEMA)
    missing = [k for k in CAR_FIELDS if k not in obj]
    if missing:
        raise ValueError(MESSAGES["car_missing"] % (CAR_SCHEMA, ", ".join(missing)))
    extra = sorted(str(k) for k in obj if k not in CAR_FIELDS)
    if extra:
        raise ValueError(MESSAGES["car_unknown"] % (CAR_SCHEMA, ", ".join(extra)))
    out = {"schema": CAR_SCHEMA}
    for k, kind in CAR_FIELDS.items():
        v = obj[k]
        if kind == "schema":
            continue
        if kind == "bom":
            if not isinstance(v, dict) or set(v) != set(AS_FOUND_SLOTS) or not all(isinstance(x, str)
                                                                                    for x in v.values()):
                raise ValueError(MESSAGES["car_bom"] % (CAR_SCHEMA, k, ", ".join(AS_FOUND_SLOTS)))
            out[k] = {s: v[s] for s in AS_FOUND_SLOTS}
        elif kind == "lines":
            if not isinstance(v, dict) or set(v) != set(SLOT_ORDER) or not all(_is_num(x) for x in v.values()):
                raise ValueError(MESSAGES["car_lines"] % (CAR_SCHEMA, k, ", ".join(SLOT_ORDER)))
            out[k] = {s: float(v[s]) for s in SLOT_ORDER}
        elif kind == "posint":
            if not _field_ok(v, "posint"):
                raise ValueError(MESSAGES["car_posint"] % (CAR_SCHEMA, k))
            out[k] = int(v)
        elif kind == "lock":
            if not (_is_num(v) and 0 < v < 90):
                raise ValueError(MESSAGES["car_lock"] % (CAR_SCHEMA, k))
            out[k] = float(v)
        else:
            if not _field_ok(v, kind):
                raise ValueError(MESSAGES[_CAR_KIND_MESSAGE[kind]] % (CAR_SCHEMA, k))
            out[k] = float(v)
    return out


# ================================================================================================ enumeration
def iter_cars(cat, block=20000):
    """Every car of the catalogue, all combinations of one SKU per slot, as blocks of index rows. A block is an
    int16 array of shape (n, number of slots), n <= block; a row holds the SKU index of each slot in SLOT_ORDER.
    The rows come in itertools.product order over SLOT_ORDER (the last slot changes fastest)."""
    cat = load_catalogue(cat)
    sizes = np.array(cat.sizes, dtype=np.int64)
    strides = np.array([int(np.prod(sizes[i + 1:], dtype=np.int64)) for i in range(len(sizes))], dtype=np.int64)
    total = cat.n_combinations
    for s in range(0, total, block):
        k = np.arange(s, min(s + block, total), dtype=np.int64)
        yield ((k[:, None] // strides[None, :]) % sizes[None, :]).astype(np.int16)


def _col(idx, slot):
    """(n,) the SKU indices of one slot in a block of index rows."""
    return np.asarray(idx)[:, SLOT_ORDER.index(slot)].astype(np.intp)


def buildable_block(idx, cat):
    """(n,) bool: the parts of each car fit by class. The motor, the battery pack, the deck posts and the tyres are
    of the fit classes of the chassis kit; the pinion has a tooth count that the motor mount reaches; the battery
    pack and the power board share a connector class, and so do the motor controller and the motor; the power board
    has the supply rails of the LiDAR and of the compute module. The length of the battery bay is not part of this:
    the quantity fits adds it. Limits of voltage, current and temperature are not part of it either: they have
    quantities of their own. quantities.md, "Fit"."""
    cat = load_catalogue(cat)
    idx = np.asarray(idx)
    ok = np.ones(idx.shape[0], dtype=bool)
    for slot in ("motor", "battery", "deck_post", "tyres", "pinion"):
        ok &= cat.fit_ok[slot][_col(idx, slot)]
    for a, b in CONNECTOR_PAIRS:
        ok &= cat.conn_ok[(a, b)][_col(idx, a), _col(idx, b)]
    pb = _col(idx, "power_board")
    for s in ("lidar", "compute"):
        ok &= cat.rail_ok[s][pb, _col(idx, s)]
    return ok


def buildable(bom, cat):
    """True when bom = {slot: SKU code} fits by class (buildable_block of one car)."""
    cat = load_catalogue(cat)
    return bool(buildable_block(cat.car_index(bom)[None, :], cat)[0])


def price_block(idx, cat):
    """(n,) price of one car in USD at the prices of the catalogue: the sum over the slots of qty_per_car x
    pack_price_usd / pack_size, plus fixed_cost_usd of the platform record (the chassis kit)."""
    cat = load_catalogue(cat)
    idx = np.asarray(idx)
    total = np.zeros(idx.shape[0])
    for slot in SLOT_ORDER:
        f = cat.f[slot]
        up = f["qty"] * f["unit_price"]
        total = total + up[_col(idx, slot)]
    return total + float(cat.platform["fixed_cost_usd"])


def sat(q, comparator, bound):
    """The decision function: is `q comparator bound` satisfied? For '>=' and '<=': d (q - b) >= -1e-9 max(|b|, 1),
    with d = +1 for '>=' and -1 for '<=' (the small term leaves room for rounding in the arithmetic). For '==',
    which is meant for a quantity that is 1 or 0 such as fits: q equals b. A q that is not finite is not satisfied.
    q is a number or an array; the result is a bool, or an array of bools of the same shape. quantities.md,
    "Decision function and limits"."""
    q = np.asarray(q, dtype=np.float64)
    b = float(bound)
    if comparator == "==":
        with np.errstate(invalid="ignore"):
            return np.isfinite(q) & (q == b)
    d = 1.0 if comparator == ">=" else -1.0 if comparator == "<=" else None
    if d is None:
        raise ValueError(MESSAGES["comparator"])
    with np.errstate(invalid="ignore"):
        return np.isfinite(q) & (d * (q - b) >= -1e-9 * max(abs(b), 1.0))


def signed_margin(q, comparator, bound, scale):
    """How far q lies on the satisfied side of the bound b, in units of scale: d (q - b) / scale, with d = +1 for
    '>=' and -1 for '<='. Positive: satisfied with that much room; negative: missed by that much."""
    if comparator not in ("<=", ">="):
        raise ValueError(MESSAGES["margin_comparator"])
    d = 1.0 if comparator == ">=" else -1.0
    return d * (np.asarray(q, dtype=np.float64) - float(bound)) / float(scale)


# ================================================================================================ lots
def _decimal(x):
    """The printed digits of a sample as a Decimal (a float is read as its shortest decimal text)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(MESSAGES["samples_number"])
    return Decimal(repr(x)) if isinstance(x, float) else Decimal(x)


def _samples(xs):
    """The printed samples of a non-empty list, as Decimals."""
    if not isinstance(xs, list) or not xs:
        raise ValueError(MESSAGES["samples_list"])
    return [_decimal(x) for x in xs]


def lot_state(kind, samples):
    """The lot-table entry of a lot from the samples that its lot report prints, in exact decimal arithmetic on the
    printed digits (so the result is what a hand calculation from the page gives). kind and samples:

    - "cell", {nominal_capacity_ah, nominal_cell_resistance_ohm, capacity_ah: [...], cell_resistance_ohm: [...]}:
      capacity_factor = mean capacity / nominal capacity, resistance_factor = mean cell resistance / nominal cell
      resistance;
    - "tyre", {normal_load_n, pull_n: {ground class: [...]}}: mu[class] = mean pull on that class / normal load;
    - "lidar", {range_m: {target class: [...]}}: range_m[class] = the smallest sample of that class.

    Each value is rounded to 2 decimals, halves away from zero. ValueError when a value is outside its range
    (LOT_RANGES). quantities.md, "Lot table"."""
    ctx = Context(prec=40, rounding=ROUND_HALF_UP)

    def mean(xs):
        """The exact mean of the printed samples."""
        d = _samples(xs)
        return ctx.divide(sum(d, Decimal(0)), Decimal(len(d)))

    def ranged(v, key):
        """v rounded to 2 decimals, halves away from zero; ValueError when that is outside the range of key."""
        v = float(v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        if not _in_range(v, key):
            raise ValueError(MESSAGES["lot_range"] % (key, v, _range_text(key)))
        return v

    if kind == "cell":
        return {"capacity_factor": ranged(ctx.divide(mean(samples["capacity_ah"]),
                                                     _decimal(samples["nominal_capacity_ah"])), "capacity_factor"),
                "resistance_factor": ranged(ctx.divide(mean(samples["cell_resistance_ohm"]),
                                                       _decimal(samples["nominal_cell_resistance_ohm"])),
                                            "resistance_factor")}
    if kind == "tyre":
        pulls = samples["pull_n"]
        if not isinstance(pulls, dict) or set(pulls) != set(SURFACES):
            raise ValueError(MESSAGES["tyre_samples"] % (SURFACES,))
        load = _decimal(samples["normal_load_n"])
        return {"mu": {s: ranged(ctx.divide(mean(pulls[s]), load), "mu") for s in SURFACES}}
    if kind == "lidar":
        rng = samples["range_m"]
        if not isinstance(rng, dict) or set(rng) != set(TARGET_CLASSES):
            raise ValueError(MESSAGES["lidar_samples"] % (TARGET_CLASSES,))
        return {"range_m": {c: ranged(min(_samples(rng[c])), "range_m") for c in TARGET_CLASSES}}
    raise ValueError(MESSAGES["lot_kind_name"] % (LOT_KINDS,))


def page_lot_table(cat):
    """The lot table that reads every lot of the catalogue at its page values: both cell factors 1.0; mu = mu_page
    on every ground class; range_m = range_white_m for the class white and range_dark_m for the other target
    classes. ValueError when the SKUs of one tyre lot or of one LiDAR lot print different page values."""
    cat = load_catalogue(cat)
    out = {}
    for lid in cat.lot_ids:
        kind = cat.lots[lid]["kind"]
        if kind == "cell":
            out[lid] = {"capacity_factor": 1.0, "resistance_factor": 1.0}
            continue
        recs = [r for r in cat.records[LOT_SLOT[kind]] if r["lot"] == lid]
        if kind == "tyre":
            mus = sorted({float(r["values"]["mu_page"]) for r in recs})
            if len(mus) != 1:
                raise ValueError(MESSAGES["page_tyre_lot"] % lid)
            out[lid] = {"mu": {s: mus[0] for s in SURFACES}}
        else:
            pairs = sorted({(float(r["values"]["range_white_m"]), float(r["values"]["range_dark_m"])) for r in recs})
            if len(pairs) != 1:
                raise ValueError(MESSAGES["page_lidar_lot"] % lid)
            w, d = pairs[0]
            out[lid] = {"range_m": {c: (w if c == "white" else d) for c in TARGET_CLASSES}}
    return out


# ================================================================================================ cycles
def _merge_segments(segs):
    """Consecutive segments equal in kind, radius, width, surface and sight merged into one (lengths or angles
    added), so that an equivalent split of a segment leaves the resampled nodes unchanged."""
    out = []
    for s in segs:
        s = dict(s)
        if out:
            t = out[-1]
            same = (t["kind"] == s["kind"] and float(t["width_m"]) == float(s["width_m"])
                    and t["surface"] == s["surface"] and float(t["sight_m"]) == float(s["sight_m"])
                    and (s["kind"] == "straight" or float(t["radius_m"]) == float(s["radius_m"])))
            if same:
                if s["kind"] == "straight":
                    t["length_m"] = float(t["length_m"]) + float(s["length_m"])
                else:
                    t["angle_deg"] = float(t["angle_deg"]) + float(s["angle_deg"])
                continue
        out.append(s)
    return out


class Nodes:
    """A cycle resampled to N_NODES nodes, uniform in arc length (ds = length / N_NODES, in m). Node i manages the
    interval [i ds, (i + 1) ds]: its curvature (kappa, in 1/m) is the largest curvature of the segments that
    overlap the interval (a straight has 0, an arc 1 / radius), its width and sight (in m) the smallest, its ground
    classes all of them (mask, one bit per class of SURFACES: the friction at the node is the smallest of the car's
    friction on those classes). A segment boundary that falls on an interval boundary (within 1e-9 of the cycle
    length) does not count as an overlap. The car drives the cycle as a closed loop: after the last node it
    continues with the first. quantities.md, "Cycles"."""

    def __init__(self, cycle):
        segs = _merge_segments(cycle["segments"])
        lengths = [_segment_length(s) for s in segs]
        ends = np.array(list(itertools.accumulate(lengths)))
        starts = np.concatenate([[0.0], ends[:-1]])
        total = float(ends[-1])
        n = N_NODES
        ds = total / n
        eps = 1e-9 * total
        seg_k = np.array([0.0 if s["kind"] == "straight" else 1.0 / float(s["radius_m"]) for s in segs])
        seg_w = np.array([float(s["width_m"]) for s in segs])
        seg_s = np.array([float(s["sight_m"]) for s in segs])
        seg_m = np.array([1 << SURFACES.index(s["surface"]) for s in segs], dtype=np.int64)
        kappa, width, sight = np.zeros(n), np.zeros(n), np.zeros(n)
        mask = np.zeros(n, dtype=np.int64)
        for i in range(n):
            a = i * ds
            b = total if i == n - 1 else (i + 1) * ds
            sel = (ends > a + eps) & (starts < b - eps)
            kappa[i] = seg_k[sel].max()
            width[i] = seg_w[sel].min()
            sight[i] = seg_s[sel].min()
            mask[i] = int(np.bitwise_or.reduce(seg_m[sel]))
        self.length_m, self.ds, self.n = total, ds, n
        self.kappa, self.width, self.sight, self.mask = kappa, width, sight, mask
        keys = list(zip(kappa.tolist(), mask.tolist(), sight.tolist()))
        kinds = sorted(set(keys))
        pos = {k: j for j, k in enumerate(kinds)}
        self.kinds = kinds                                   # the distinct (curvature, mask, sight) of the nodes
        self.kind_of = np.array([pos[k] for k in keys], dtype=np.intp)


def resample_cycle(cycle):
    """The Nodes object of a cycle {segments: [...]} (see Nodes)."""
    return Nodes(cycle)


# ================================================================================================ the replica
class _Cars:
    """The replicas of one block of index rows, built on one prepared car record under its lot table: one array of
    shape (n,) per parameter. A replica is the reference unit with the parts of the BOM in place of the parts it was
    found with, its as-found parts (quantities.md, "The replica"):

    - mass_g (g) = as_found_mass_g - the page masses of the as-found parts (for the steering servo the record's
      as_found_servo_mass_g) + the page masses of the BOM's parts + pod_mass_g; a line of four parts counts four
      times;
    - front_n, the static load on the front axle (N) = GRAVITY_MPS2 x (as_found_front_axle_g x wheelbase_mm + the
      sum over the ten lines of (mass of the BOM's parts - mass of the as-found parts) x line_x_mm + pod_mass_g x
      pod_x_mm) / wheelbase_mm / 1000;
    - ride (mm) = (outside diameter of the BOM's tyre - outside diameter of the as-found tyre) / 2 and rise (mm) =
      length of the BOM's deck post - length of the as-found post: every height of the car record moves up by the
      ride, the deck and everything it carries also by the rise, one for one.

    The drivetrain and the electrics (quantities.md, "The reference driver and the constants"): r the wheel radius
    (m); G the overall reduction from the motor shaft to the wheels; kt (N.m/A), kv (rpm/V) and ke (V s/rad) of the
    motor; Rm, Ron and Rp the resistances of the motor, of the motor controller and of the pack under its lot (ohm),
    Rtot their sum; I0 the no-load current of the motor and Ilim the current limit of the controller (A); cap_as the
    charge of the pack under its lot (A s); kGd and kGb the wheel force per ampere of motor current when driving and
    when braking (N/A); keGr the back-EMF per unit of speed (V s/m); Vref the reference bus voltage (V); mu and
    range_by_class the friction and the LiDAR range of the car's lots, one column per ground class and per target
    class; t_lat the reaction latency (s); p_hotel and i_hotel the power and the current of the electronics that are
    always on (W, A).
    """

    def __init__(self, idx, prep):
        cat, car, lot_table = prep.cat, prep.car, prep.lot_table
        idx = np.asarray(idx)
        n = idx.shape[0]
        c = {s: _col(idx, s) for s in SLOT_ORDER}
        F = cat.f

        def g(slot, fld):
            """(n,) the page field fld of the SKU that each car has in the slot."""
            return F[slot][fld][c[slot]]

        self.n, self.cat, self.c, self.prep = n, cat, c, prep
        # heights: those of the record belong to the reference unit on its as-found tyres and deck posts (mm)
        self.od_mm = g("tyres", "od_mm")
        self.ride = (self.od_mm - prep.ref_od_mm) / 2.0
        self.rise = g("deck_post", "length_mm") - prep.ref_post_mm
        # mass (g): the weighed reference unit less the page masses of its as-found parts, plus the page masses of
        # this car's parts, plus the pod
        m = np.zeros(n)
        for slot in SLOT_ORDER:
            m = m + F[slot]["qty"][c[slot]] * F[slot]["mass_g"][c[slot]]
        self.mass_g = (prep.base_mass_g + m) + prep.pod_mass_g
        # static load on the front axle (N): the as-found front-axle reading, plus every line's mass change and the
        # pod at their x, over the wheelbase
        s = np.zeros(n)
        for slot in SLOT_ORDER:
            s = s + prep.moment[slot][c[slot]]
        wb = car["wheelbase_mm"]
        self.front_n = GRAVITY_MPS2 * ((car["as_found_front_axle_g"] * wb + s)
                                       + car["pod_mass_g"] * car["pod_x_mm"]) / wb / 1000.0
        # drivetrain and electrics
        self.r = self.od_mm / 2000.0
        self.G = car["spur_teeth"] / g("pinion", "teeth") * car["gearbox_ratio"]
        self.kt = g("motor", "kt_nm_per_a")
        self.kv = g("motor", "kv_rpm_per_v")
        self.ke = 60.0 / (2.0 * math.pi * self.kv)
        self.Rm = g("motor", "resistance_ohm")
        self.I0 = g("motor", "no_load_current_a")
        self.Ilim = g("motor_controller", "current_limit_a")
        self.Ron = g("motor_controller", "on_resistance_ohm")
        self.cells = g("battery", "cells")
        ba_lot = [lot_table[x] for x in cat.lot_of["battery"]]
        capf = np.array([float(x["capacity_factor"]) for x in ba_lot])[c["battery"]]
        resf = np.array([float(x["resistance_factor"]) for x in ba_lot])[c["battery"]]
        self.Rp = self.cells * g("battery", "cell_resistance_ohm") * resf
        self.cap_as = g("battery", "capacity_ah") * capf * 3600.0
        self.kGd = self.kt * self.G * DRIVETRAIN_EFFICIENCY / self.r       # wheel force per motor amp, driving
        self.kGb = self.kt * self.G / (DRIVETRAIN_EFFICIENCY * self.r)     # wheel force per motor amp, braking
        self.keGr = self.ke * self.G / self.r                              # back-EMF per m/s
        self.Rtot = self.Rm + self.Ron + self.Rp
        self.Rmc = self.Rm + self.Ron
        self.Vref = self.cells * PROFILE_CELL_V
        ty_lot = [lot_table[x] for x in cat.lot_of["tyres"]]
        self.mu = np.array([[float(x["mu"][s]) for s in SURFACES] for x in ty_lot])[c["tyres"]]  # (n, ground class)
        li_lot = [lot_table[x] for x in cat.lot_of["lidar"]]
        self.range_by_class = np.array([[float(x["range_m"][k]) for k in TARGET_CLASSES]
                                        for x in li_lot])[c["lidar"]]                             # (n, target class)
        self.step_deg = g("lidar", "angular_step_deg")
        self.scan_hz = g("lidar", "scan_rate_hz")
        self.t_lat = 1.0 / self.scan_hz + g("compute", "latency_max_s") + ACTUATION_LATENCY_S
        self.p_hotel = ((g("lidar", "power_w") + g("compute", "power_w")) / RAIL_CONVERTER_EFFICIENCY
                        + g("power_board", "quiescent_w") + g("motor_controller", "quiescent_w"))
        self.i_hotel = self.p_hotel / self.Vref
        self.g = g

    def config_mass(self, cfg):
        """(n,) mass in kg under a configuration: mass_g, plus body_shell_mass_g of the platform record when
        body_shell is true."""
        m = self.mass_g
        if cfg["body_shell"]:
            m = m + float(self.cat.platform["body_shell_mass_g"])
        return m / 1000.0

    def ocv(self, soc):
        """(n,) open-circuit voltage of the pack in V at the state of charge soc: cells x the cell voltage of the
        page's table (ocv_soc, ocv_cell_v), interpolated linearly and held at its ends."""
        out = np.empty(self.n)
        ba = self.c["battery"]
        for k in np.unique(ba).tolist():
            sel = ba == k
            s_tab, v_tab = self.cat.ocv[k]
            out[sel] = np.interp(soc[sel], s_tab, v_tab)
        return self.cells * out

    def steering(self, surface):
        """The three steering quantities when both front wheels of the standing car are turned on a ground class, as
        (steering_torque_nm, steering_torque_ratio, lock_time_s), each (n,). Per wheel the torque about the king pin
        is M = mu N sqrt(B^2 / 8 + e^2), in N.mm: mu the friction of the tyre's lot on that ground class, N half the
        front-axle load in newtons, B the width of the tyre in mm (width_mm of its page), e kingpin_ground_offset_mm.
        At the servo T = 2 M R / LINKAGE_EFFICIENCY, in N.m after division by 1000, with the linkage ratio R =
        (servo_saver_arm_mm / bellcrank_input_arm_mm) x (bellcrank_output_arm_mm / steering_arm_mm). The torque
        ratio is T over stall_torque_nm of the servo. From lock to lock the servo turns through 2 x
        steering_lock_deg / R degrees, at its no-load speed (60 / no_load_s_per_60deg, in deg/s) times (1 - torque
        ratio): the lock time is that angle over that speed, and it is infinite when T reaches the stall torque.
        quantities.md, "Steering at standstill"."""
        car, ratio = self.prep.car, self.prep.ratio
        mu = self.mu[:, SURFACES.index(surface)]
        b = self.g("tyres", "width_mm")
        e = car["kingpin_ground_offset_mm"]
        wheel = mu * (self.front_n / 2.0) * np.sqrt(b * b / 8.0 + e * e)
        torque = 2.0 * wheel * ratio / LINKAGE_EFFICIENCY / 1000.0
        u = torque / self.g("steering_servo", "stall_torque_nm")
        w0 = 60.0 / self.g("steering_servo", "no_load_s_per_60deg")             # deg/s
        travel = 2.0 * car["steering_lock_deg"] / ratio                          # deg of the servo, lock to lock
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            time = np.where(u < 1.0, travel / (w0 * (1.0 - u)), np.inf)
        return torque, u, time


# ================================================================================================ drive physics
# Driving on flat ground (quantities.md, "The reference driver and the constants"). In these functions cars is a
# _Cars object, m the mass in kg, V the bus voltage in V and mu the friction on the ground class in question.
def _regimes(cars, m, V):
    """The net tractive force (N) on flat ground as two regimes of the speed v (m/s): limited by the controller's
    current, f1(v) = a1 - kd v^2, and limited by the bus voltage, f2(v) = a2 - b2 v - kd v^2, with kd =
    DRAG_N_PER_MPS2; the force is min(f1, f2). Returns (a1, a2, b2, vx), vx being the speed above which the current
    limit no longer binds."""
    fres0 = ROLLING_RESISTANCE * m * GRAVITY_MPS2
    a1 = cars.kGd * (cars.Ilim - cars.I0) - fres0
    a2 = cars.kGd * (V / cars.Rtot - cars.I0) - fres0
    b2 = cars.kGd * cars.keGr / cars.Rtot
    vx = np.maximum((V - cars.Ilim * cars.Rtot) / cars.keGr, 0.0)
    return a1, a2, b2, vx


def _root_where(a1, a2, b2, vx, shift):
    """The speed (m/s) at which the net tractive force has fallen to `shift` (N); 0 where the force is below `shift`
    already at rest."""
    kd = DRAG_N_PER_MPS2
    c1 = a1 - shift
    c2 = a2 - shift
    with np.errstate(invalid="ignore", divide="ignore"):
        v1 = np.sqrt(np.maximum(c1, 0.0) / kd)
        v2 = np.where(c2 > 0, 2.0 * c2 / (b2 + np.sqrt(b2 * b2 + 4.0 * kd * np.maximum(c2, 0.0))), 0.0)
    f_vx = c1 - kd * vx * vx
    return np.where(f_vx > 0, v2, v1)


def _top_speed(cars, m, V):
    """Top speed (m/s) at the mass m and the bus voltage V: the speed at which the net tractive force is zero."""
    a1, a2, b2, vx = _regimes(cars, m, V)
    return _root_where(a1, a2, b2, vx, 0.0)


def _accel_time(cars, m, V, mu, v1):
    """Time (s) from rest to the speed v1 on flat ground with dv/dt = min(mu g, net tractive force / m), in closed
    form per regime; infinite when v1 is not below the top speed."""
    kd = DRAG_N_PER_MPS2
    a1, a2, b2, vx = _regimes(cars, m, V)
    ftr = mu * GRAVITY_MPS2 * m
    vtop = _root_where(a1, a2, b2, vx, 0.0)
    vt = _root_where(a1, a2, b2, vx, ftr)
    lo = np.minimum(vt, vx)
    hi = np.maximum(vt, vx)
    bounds = [np.zeros_like(v1), np.minimum(lo, v1), np.minimum(hi, v1), v1 + np.zeros_like(lo)]
    total = np.zeros_like(lo)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        d1 = np.sqrt(4.0 * kd * np.maximum(a1, 1e-300))
        r1 = np.sqrt(np.maximum(a1, 0.0) / kd)
        disc2 = b2 * b2 + 4.0 * kd * a2
        sq2 = np.sqrt(np.maximum(disc2, 1e-300))
        rp = (-b2 + sq2) / (2.0 * kd)
        rm = (-b2 - sq2) / (2.0 * kd)
        for va, vb in zip(bounds[:-1], bounds[1:]):
            vm = 0.5 * (va + vb)
            f = np.minimum(a1 - kd * vm * vm, a2 - b2 * vm - kd * vm * vm)
            t_tr = (vb - va) * m / ftr
            t_cur = m / d1 * np.log((vb + r1) * (r1 - va) / ((r1 - vb) * (va + r1)))
            t_vol = m / sq2 * np.log((vb - rm) * (rp - va) / ((rp - vb) * (va - rm)))
            t = np.where(f >= ftr, t_tr, np.where(vm <= vx, t_cur, t_vol))
            total = total + np.where(vb > va, t, 0.0)
    return np.where(v1 < vtop, total, np.inf)


def _stop_decel(cars, m, mu):
    """The deceleration (m/s^2) with which the driver plans an emergency stop: BRAKE_UTILISATION x min(mu g, (motor
    braking force at the controller's current limit + rolling resistance) / m)."""
    fb = cars.kGb * cars.Ilim + ROLLING_RESISTANCE * m * GRAVITY_MPS2
    return BRAKE_UTILISATION * np.minimum(mu * GRAVITY_MPS2, fb / m)


def _detection_range(cars, target):
    """(n,) distance (m) at which the LiDAR detects a target {class, width_m}: the range of its lot on that class, or
    the distance at which MIN_BEAM_HITS neighbouring beams still fall on the width, whichever is shorter."""
    k = TARGET_CLASSES.index(target["class"])
    beams = float(target["width_m"]) / (MIN_BEAM_HITS * cars.step_deg * math.pi / 180.0)
    return np.minimum(cars.range_by_class[:, k], beams)


def _mu_of_mask(cars, mask):
    """(n,) friction on a set of ground classes (mask: one bit per class of SURFACES): the smallest of the car's
    friction on them."""
    bits = [b for b in range(len(SURFACES)) if mask >> b & 1]
    out = cars.mu[:, bits[0]]
    for b in bits[1:]:
        out = np.minimum(out, cars.mu[:, b])
    return out


# ================================================================================================ lap profile
# The lap of the reference driver (quantities.md, "The speed profile (one lap)"). Two shortcuts that do not change
# any result: cars with equal profile parameters are computed once (_DEDUP), and the second time around the loop a
# pass stops at the first node at which no car changes (_EARLY_STOP). _STATS counts the work done: the laps solved,
# the cars in them and the node steps.
_DEDUP = True
_EARLY_STOP = True
_STATS = {"profiles": 0, "cars": 0, "steps": 0}

# the parameters a lap depends on: cars that are equal in all of them drive the same lap
_PROFILE_FIELDS = ("m", "kGd", "kGb", "keGr", "Rtot", "Rmc", "I0", "Ilim", "Vref", "t_lat", "mu0", "mu1", "mu2",
                   "mu3")


def _profile(cars, nodes, pace, m):
    """One lap of the reference driver at `pace` on the nodes of a cycle, for cars of the mass m (kg), at the
    reference bus voltage. Returns (lap time in s, battery charge drawn for traction per lap in A s, integral of the
    motor current squared over the lap in A^2 s, peak battery current for traction in A), each (n,). Cars with equal
    parameters are computed once."""
    cols = [m, cars.kGd, cars.kGb, cars.keGr, cars.Rtot, cars.Rmc, cars.I0, cars.Ilim, cars.Vref, cars.t_lat,
            cars.mu[:, 0], cars.mu[:, 1], cars.mu[:, 2], cars.mu[:, 3]]
    P = np.column_stack(cols)
    if not _DEDUP:
        par = {k: np.ascontiguousarray(P[:, j]) for j, k in enumerate(_PROFILE_FIELDS)}
        return _profile_core(par, nodes, float(pace))
    uniq, inv = np.unique(P, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    par = {k: np.ascontiguousarray(uniq[:, j]) for j, k in enumerate(_PROFILE_FIELDS)}
    out = _profile_core(par, nodes, float(pace))
    return tuple(x[inv] for x in out)


def _profile_core(par, nodes, pace):
    """The lap of _profile for cars given by their parameters par (one array per name of _PROFILE_FIELDS). The
    squared speed at every node starts at its limit: pace x top speed, in a bend the lateral limit, and the speed
    from which an emergency stop fits into the sight distance. A forward pass lowers it to what the car can
    accelerate to from the node before, a backward pass to what it can brake from before the node after; each pass
    goes twice around the loop. The lap is then integrated between consecutive nodes at constant acceleration."""
    g = GRAVITY_MPS2
    kd = DRAG_N_PER_MPS2
    m = par["m"]
    n = m.shape[0]
    N, ds = nodes.n, nodes.ds
    fres0 = ROLLING_RESISTANCE * m * g
    kdm = kd / m
    A1 = (par["kGd"] * (par["Ilim"] - par["I0"]) - fres0) / m
    A2 = (par["kGd"] * (par["Vref"] / par["Rtot"] - par["I0"]) - fres0) / m
    B2 = par["kGd"] * par["keGr"] / (par["Rtot"] * m)
    BK = (par["kGb"] * par["Ilim"] + fres0) / m
    mu = np.column_stack([par["mu0"], par["mu1"], par["mu2"], par["mu3"]])
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        vx = np.maximum((par["Vref"] - par["Ilim"] * par["Rtot"]) / par["keGr"], 0.0)
        c1, c2, b2 = A1 * m, A2 * m, B2 * m
        v1 = np.sqrt(np.maximum(c1, 0.0) / kd)
        v2 = np.where(c2 > 0, 2.0 * c2 / (b2 + np.sqrt(b2 * b2 + 4.0 * kd * np.maximum(c2, 0.0))), 0.0)
        vtop = np.where(c1 - kd * vx * vx > 0, v2, v1)
        ucap = (pace * vtop) ** 2
        t_lat = par["t_lat"]
        kinds = nodes.kinds
        UL = np.empty((len(kinds), n))
        AMX = np.empty((len(kinds), n))
        AMX2 = np.empty((len(kinds), n))
        KAP = np.array([k[0] for k in kinds])
        for j, (kap, mask, sight) in enumerate(kinds):
            bits = [b for b in range(len(SURFACES)) if mask >> b & 1]
            mu_n = mu[:, bits].min(axis=1)
            amax = pace * mu_n * g
            ab = BRAKE_UTILISATION * np.minimum(mu_n * g, BK)
            vs = ab * (np.sqrt(t_lat * t_lat + 2.0 * sight / ab) - t_lat)
            ul = np.minimum(ucap, vs * vs)
            if kap > 0:
                ul = np.minimum(ul, LATERAL_UTILISATION * amax / kap)
            UL[j], AMX[j], AMX2[j] = ul, amax, amax * amax
        kind_of = nodes.kind_of
        U = UL[kind_of]                                       # (N, n) squared speed at the nodes
        two_ds = 2.0 * ds
        tmp = np.empty(n)
        un = np.empty(n)

        def accel(i):
            """Into un: the largest squared speed the cars can reach at the next node from node i (acceleration
            limited by the friction ellipse at the pace and by the motor at the bus voltage)."""
            ui = U[i]
            k = kind_of[i]
            np.sqrt(ui, out=tmp)
            np.multiply(B2, tmp, out=tmp)
            np.subtract(A2, tmp, out=tmp)
            np.minimum(A1, tmp, out=tmp)
            np.subtract(tmp, kdm * ui, out=tmp)
            if KAP[k] > 0:
                lat = ui * KAP[k]
                np.minimum(tmp, np.sqrt(np.maximum(AMX2[k] - lat * lat, 0.0)), out=tmp)
            else:
                np.minimum(tmp, AMX[k], out=tmp)
            np.maximum(tmp, 0.0, out=tmp)
            np.multiply(tmp, two_ds, out=un)
            np.add(un, ui, out=un)

        def decel(j):
            """Into un: the largest squared speed at the node before from which the cars can still brake down to
            the squared speed of node j (the deceleration is evaluated at node j)."""
            uj = U[j]
            k = kind_of[j]
            np.multiply(kdm, uj, out=tmp)
            np.add(tmp, BK, out=tmp)
            if KAP[k] > 0:
                lat = uj * KAP[k]
                np.minimum(tmp, np.sqrt(np.maximum(AMX2[k] - lat * lat, 0.0)), out=tmp)
            else:
                np.minimum(tmp, AMX[k], out=tmp)
            np.multiply(tmp, two_ds, out=un)
            np.add(un, uj, out=un)

        # forward pass around the loop twice; the second lap stops at the first node no car changes at
        steps = 0
        for lap in range(2):
            for i in range(N):
                j = i + 1 if i + 1 < N else 0
                accel(i)
                steps += 1
                if lap == 1 and _EARLY_STOP and bool(np.all(un >= U[j])):
                    break
                np.minimum(U[j], un, out=U[j])
        # backward pass, same rule
        for lap in range(2):
            for j in range(N - 1, -1, -1):
                i = j - 1 if j > 0 else N - 1
                decel(j)
                steps += 1
                if lap == 1 and _EARLY_STOP and bool(np.all(un >= U[i])):
                    break
                np.minimum(U[i], un, out=U[i])
        _STATS["profiles"] += 1
        _STATS["cars"] += n
        _STATS["steps"] += steps
        # integrate the lap between consecutive nodes (constant acceleration on each step)
        V = np.sqrt(U)
        T = np.zeros(n)
        Q = np.zeros(n)
        S2 = np.zeros(n)
        IPK = np.zeros(n)
        inv_kGd = 1.0 / par["kGd"]
        inv_kGb = 1.0 / par["kGb"]
        inv_V = 1.0 / par["Vref"]
        keGr, Rmc, I0, Ilim = par["keGr"], par["Rmc"], par["I0"], par["Ilim"]
        inv_2ds = 1.0 / two_ds
        for i in range(N):
            j = i + 1 if i + 1 < N else 0
            s = V[i] + V[j]
            dt = two_ds / s
            vb = 0.5 * s
            F = m * ((U[j] - U[i]) * inv_2ds) + fres0 + kd * vb * vb
            drive = F > 0
            im = np.where(drive, F * inv_kGd + I0, -F * inv_kGb)
            np.minimum(im, Ilim, out=im)
            ib = np.where(drive, im * (keGr * vb + im * Rmc) * inv_V, 0.0)
            T += dt
            Q += ib * dt
            S2 += im * im * dt
            np.maximum(IPK, ib, out=IPK)
    return T, Q, S2, IPK


# ================================================================================================ thermal
def _two_node(yw, yc, P, Cw, Cc, Rwc, Rca, t):
    """The two-node thermal model of the motor in closed form: from the winding and case temperatures above ambient
    yw and yc (K), after t seconds of constant heat P (W) in the winding, with the heat capacities Cw and Cc (J/K)
    and the thermal resistances Rwc, winding to case, and Rca, case to ambient (K/W). Returns (winding at the end,
    case at the end, the highest winding temperature on the way), all above ambient. quantities.md, "Over a
    schedule"."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        a11 = -1.0 / (Cw * Rwc)
        a12 = 1.0 / (Cw * Rwc)
        a21 = 1.0 / (Cc * Rwc)
        a22 = -1.0 / (Cc * Rwc) - 1.0 / (Cc * Rca)
        wss = P * (Rwc + Rca)
        css = P * Rca
        sq = np.sqrt((a11 - a22) ** 2 + 4.0 * a12 * a21)
        tr = a11 + a22
        l1 = 0.5 * (tr + sq)
        l2 = 0.5 * (tr - sq)
        dw = yw - wss
        dc = yc - css
        c1 = ((a11 - l2) * dw + a12 * dc) / sq
        c2 = -((a11 - l1) * dw + a12 * dc) / sq
        d1 = (a21 * dw + (a22 - l2) * dc) / sq
        d2 = -(a21 * dw + (a22 - l1) * dc) / sq
        e1 = np.exp(l1 * t)
        e2 = np.exp(l2 * t)
        w_end = wss + c1 * e1 + c2 * e2
        c_end = css + d1 * e1 + d2 * e2
        ratio = (-c2 * l2) / (c1 * l1)
        ts = np.log(ratio) / sq
        ok = np.isfinite(ts) & (ratio > 0) & (ts > 0) & (ts < t)
        w_star = wss + c1 * np.exp(l1 * ts) + c2 * np.exp(l2 * ts)
        peak = np.maximum(np.maximum(yw, w_end), np.where(ok, w_star, -np.inf))
    return w_end, c_end, peak


def _one_node(y, P, R, C, t):
    """The one-node thermal model of the motor controller: the temperature above ambient (K) that follows y after t
    seconds of constant heat P (W), with the thermal resistance R (K/W) and the heat capacity C (J/K)."""
    yss = P * R
    return yss + (y - yss) * np.exp(-t / (R * C))


# ================================================================================================ scenarios
def _schedule(block, sc, prof, m):
    """The four schedule quantities of the scenario sc for the cars of a block, as {name: (n,) array}. prof maps
    each pace of the schedule to the lap of that pace (see _profile); m is the mass in kg. Block after block: the
    drive takes charge from the pack in use (traction plus the electronics that are always on), the bus voltage is
    taken at the end of the drive under the peak current of the lap, and the winding, case and controller
    temperatures above ambient are carried on through drive and rest. A pack that the schedule moves on from keeps
    the state of charge it has at that moment. With the body shell fitted the thermal resistances to ambient are
    multiplied by body_shell_thermal_factor. quantities.md, "Over a schedule"."""
    cars = block.cars
    g = cars.g
    sch = sc["schedule"]
    env = sc["environment"]
    shell = bool(env["config"]["body_shell"])
    tf = float(cars.cat.platform["body_shell_thermal_factor"]) if shell else 1.0
    Cw, Cc = g("motor", "cth_w_j_per_k"), g("motor", "cth_c_j_per_k")
    Rwc, Rca = g("motor", "rth_wc_k_per_w"), g("motor", "rth_ca_k_per_w") * tf
    Rc, Ccap = g("motor_controller", "rth_k_per_w") * tf, g("motor_controller", "cth_j_per_k")
    qc = g("motor_controller", "quiescent_w")
    ih = cars.i_hotel
    swap, packs = sch["swap_after_blocks"], int(sch["packs"])
    n = cars.n
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        soc = np.ones(n)
        finals = []
        cur = 0
        vmin = np.full(n, np.inf)
        yw = np.zeros(n)
        yc = np.zeros(n)
        yk = np.zeros(n)
        wpk = np.zeros(n)
        kpk = np.zeros(n)
        for k, b in enumerate(expand_blocks(sch)):
            pk = 0 if swap is None else min(k // int(swap), packs - 1)
            if pk != cur:
                finals.append(soc)
                soc = np.ones(n)
                cur = pk
            T, Q, S2, IPK = prof[float(b["pace"])]
            drive, rest = float(b["drive_s"]), float(b["rest_s"])
            soc = soc - (Q / T + ih) * drive / cars.cap_as
            vmin = np.minimum(vmin, cars.ocv(soc) - (IPK + ih) * cars.Rp)
            ms = S2 / T
            yw, yc, peak = _two_node(yw, yc, ms * cars.Rm, Cw, Cc, Rwc, Rca, drive)
            wpk = np.maximum(wpk, peak)
            yk = _one_node(yk, ms * cars.Ron + qc, Rc, Ccap, drive)
            kpk = np.maximum(kpk, yk)
            if rest > 0:
                yw, yc, _ = _two_node(yw, yc, 0.0, Cw, Cc, Rwc, Rca, rest)
                yk = _one_node(yk, qc, Rc, Ccap, rest)
                kpk = np.maximum(kpk, yk)
                soc = soc - ih * rest / cars.cap_as
        finals.append(soc)
        reserve = finals[0]
        for f in finals[1:]:
            reserve = np.minimum(reserve, f)
        ta = float(env["ambient_c"])
        return {"schedule_energy_reserve_frac": reserve,
                "schedule_bus_margin_v": vmin - g("power_board", "input_min_v"),
                "schedule_winding_headroom_c": g("motor", "winding_limit_c") - (ta + wpk),
                "schedule_controller_headroom_c": g("motor_controller", "temp_limit_c") - (ta + kpk)}


# ================================================================================================ evaluation
class Prepared:
    """A catalogue, the car record of the reference unit and a context {lot_table, cycles, scenarios}, checked
    against each other once, for repeated evaluation (see prepare). The lot table must have a valid entry for every
    lot of the catalogue, and every code of the record's as_found_bom must be an SKU of its line. A cycle or a
    scenario is checked when a request first uses it.

    Worked out once from the record and the catalogue: as_found (slot -> SKU index of the as-found part), ref_od_mm
    and ref_post_mm (outside diameter of the as-found tyre, length of the as-found deck post), base_mass_g
    (as_found_mass_g less the page masses of the as-found parts and less as_found_servo_mass_g), moment[slot] (for
    every SKU of the slot: the mass of its parts on a car less that of the as-found parts, times line_x_mm of the
    slot; g mm) and ratio (the linkage ratio of the steering, see _Cars.steering)."""

    def __init__(self, cat, car, context):
        self.cat = cat = load_catalogue(cat)
        self.car = car = load_car(car)
        if isinstance(context, Prepared):
            context = context.context
        if not isinstance(context, dict) or not isinstance(context.get("lot_table"), dict):
            raise ValueError(MESSAGES["context_lot_table"])
        self.context = context
        lt = context["lot_table"]
        p = []
        for lid in cat.lot_ids:
            if lid not in lt:
                p.append(MESSAGES["lot_table_missing"] % lid)
            else:
                _check_lot(lid, lt[lid], p)
        if p:
            raise ValueError("; ".join(p))
        self.lot_table = lt
        self.cycles = context.get("cycles") or {}
        self.scenarios = context.get("scenarios") or {}
        self._nodes = {}
        self._ok = set()
        # the as-found parts of the reference unit, their page masses and the line moments of every SKU
        self.as_found = {}
        for s in AS_FOUND_SLOTS:
            code = car["as_found_bom"][s]
            if code not in cat.index[s]:
                raise ValueError(MESSAGES["car_as_found"] % (CAR_SCHEMA, code, s))
            self.as_found[s] = cat.index[s][code]
        F = cat.f
        self.ref_od_mm = float(F["tyres"]["od_mm"][self.as_found["tyres"]])
        self.ref_post_mm = float(F["deck_post"]["length_mm"][self.as_found["deck_post"]])
        x = car["line_x_mm"]
        total = 0.0
        self.moment = {}
        for s in SLOT_ORDER:
            qm = F[s]["qty"] * F[s]["mass_g"]
            asf = float(qm[self.as_found[s]]) if s in self.as_found else car["as_found_servo_mass_g"]
            total = total + asf
            self.moment[s] = (qm - asf) * x[s]
        self.base_mass_g = car["as_found_mass_g"] - total
        self.pod_mass_g = car["pod_mass_g"]
        self.ratio = (car["servo_saver_arm_mm"] / car["bellcrank_input_arm_mm"]) * (
            car["bellcrank_output_arm_mm"] / car["steering_arm_mm"])

    def nodes(self, cid):
        """The Nodes of the cycle cid of the context, resampled once and kept. ValueError when the context has no
        such cycle or the cycle is not valid."""
        if cid not in self._nodes:
            p = []
            if cid not in self.cycles:
                raise ValueError(MESSAGES["cycle_missing"] % (cid,))
            _check_cycle(cid, self.cycles[cid], p)
            if p:
                raise ValueError("; ".join(p))
            self._nodes[cid] = Nodes(self.cycles[cid])
        return self._nodes[cid]

    def check(self, name, cond):
        """Check one request: the name is a quantity of QUANTITIES, the conditions have exactly the keys of its
        signature and valid values, and a scenario they name is valid together with its cycle. ValueError with the
        problems found. A request that has passed is not checked again."""
        where = MESSAGES["where_request"] % (name,)
        try:
            key = (name, canonical(cond))
        except (TypeError, ValueError):
            raise ValueError(MESSAGES["conditions_json"] % where) from None
        if key in self._ok:
            return
        if name not in QUANTITIES:
            raise ValueError(MESSAGES["unknown_quantity"] % (name,))
        p = []
        _check_conditions(where, name, cond, self.scenarios, p)
        configs = [cond["config"]] if not p and "config" in cond else []
        if not p and name in SCENARIO_QUANTITIES:
            sid = cond["scenario"]
            p.extend(_scenario_problems(sid, self.scenarios[sid], self.cycles))
            if not p:
                self.nodes(self.scenarios[sid]["cycle"])
                configs.append(self.scenarios[sid]["environment"]["config"])
        if not p and any(c["body_shell"] for c in configs) and not self.cat.has_shell:
            p.append(MESSAGES["no_shell"] % where)
        if p:
            raise ValueError("; ".join(p))
        self._ok.add(key)


class _Block:
    """One block of cars under a Prepared object. Masses, laps, scenario results and steering results are worked out
    when a request first needs them and shared by the requests that follow."""

    def __init__(self, prep, idx):
        self.prep = prep
        self.idx = np.asarray(idx)
        self.cars = _Cars(self.idx, prep)
        self._mass, self._prof, self._scen, self._steer = {}, {}, {}, {}

    def mass(self, cfg):
        """(n,) mass in kg under the configuration cfg."""
        key = canonical(cfg)
        if key not in self._mass:
            self._mass[key] = self.cars.config_mass(cfg)
        return self._mass[key]

    def profile(self, cid, pace, cfg):
        """The lap of the cycle cid at a pace under the configuration cfg (see _profile)."""
        key = (cid, float(pace), canonical(cfg))
        if key not in self._prof:
            m = self.mass(cfg)
            self._prof[key] = _profile(self.cars, self.prep.nodes(cid), pace, m)
        return self._prof[key]

    def steering(self, surface):
        """The three steering quantities on a ground class (see _Cars.steering)."""
        if surface not in self._steer:
            self._steer[surface] = self.cars.steering(surface)
        return self._steer[surface]

    def scenario(self, sid):
        """{name: (n,) array} of the six scenario quantities on the scenario sid. The mass is that of the scenario's
        configuration; lap_time_s is the lap at the highest pace of the schedule; zone_stop_margin_m = min(appear_m,
        detection range of the zone's target) - the stopping distance from min(speed_cap_mps, top speed at the
        reference bus voltage) on the zone's ground class, and NaN when the scenario has no zone."""
        if sid in self._scen:
            return self._scen[sid]
        sc = self.prep.scenarios[sid]
        cars = self.cars
        env = sc["environment"]
        cfg = env["config"]
        m = self.mass(cfg)
        paces = sorted({float(b["pace"]) for b in expand_blocks(sc["schedule"])})
        prof = {p: self.profile(sc["cycle"], p, cfg) for p in paces}
        out = {"lap_time_s": prof[paces[-1]][0]}
        out.update(_schedule(self, sc, prof, m))
        z = env["zone"]
        if z is None:
            out["zone_stop_margin_m"] = np.full(cars.n, np.nan)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                vz = np.minimum(float(z["speed_cap_mps"]), _top_speed(cars, m, cars.Vref))
                ab = _stop_decel(cars, m, cars.mu[:, SURFACES.index(z["surface"])])
                seen = np.minimum(float(z["appear_m"]), _detection_range(cars, z["target"]))
                out["zone_stop_margin_m"] = seen - (vz * cars.t_lat + vz * vz / (2.0 * ab))
        self._scen[sid] = out
        return out

    def value(self, name, cond):
        """(n,) values of one quantity of QUANTITIES under its conditions, in the unit of its entry. The heights, in
        mm, with ride and rise as in _Cars: scan plane = lidar_seat_height_mm + ride + rise + optical_offset_mm of
        the LiDAR; ground clearance = lowest_point_height_mm + ride; envelope height = the highest of
        chassis_top_height_mm + ride, deck_load_top_height_mm + ride + rise and lidar_seat_height_mm + ride + rise +
        the height of the LiDAR's housing (the third number of its box_mm). fits = 1 when the parts fit by class
        (buildable_block) and length_mm of the pack is no more than battery_bay_length_mm, else 0. Every other
        quantity is one line below or a function above. quantities.md, "The dictionary"."""
        self.prep.check(name, cond)
        cars = self.cars
        g = cars.g
        car = self.prep.car
        if name in SCENARIO_QUANTITIES:
            return self.scenario(cond["scenario"])[name].copy()
        if name in STEERING_QUANTITIES:
            return self.steering(cond["surface"])[STEERING_QUANTITIES.index(name)].copy()
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            if name == "mass_kg":
                return self.mass(cond["config"]).copy()
            if name == "front_axle_load_n":
                return cars.front_n.copy()
            if name == "scan_plane_height_mm":
                return car["lidar_seat_height_mm"] + cars.ride + cars.rise + g("lidar", "optical_offset_mm")
            if name == "ground_clearance_mm":
                return car["lowest_point_height_mm"] + cars.ride
            if name == "envelope_height_mm":
                chassis = car["chassis_top_height_mm"] + cars.ride
                deck = car["deck_load_top_height_mm"] + cars.ride + cars.rise
                lidar = car["lidar_seat_height_mm"] + cars.ride + cars.rise + g("lidar", "box_mm")[:, 2]
                return np.maximum(np.maximum(chassis, deck), lidar)
            if name == "fits":
                bay = car["battery_bay_length_mm"]
                ok = buildable_block(self.idx, self.prep.cat) & (
                    g("battery", "length_mm") - bay <= 1e-9 * max(abs(bay), 1.0))
                return ok.astype(np.float64)
            if name == "battery_nominal_v":
                return cars.cells * g("battery", "cell_nominal_v")
            if name == "motor_torque_nm":
                return cars.kt * np.maximum(float(cond["current_a"]) - cars.I0, 0.0)
            if name == "lidar_scan_rate_hz":
                return cars.scan_hz.copy()
            if name == "lidar_angular_step_deg":
                return cars.step_deg.copy()
            if name == "unit_cost_usd":
                return price_block(self.idx, self.prep.cat)
            if name == "compute_latency_s":
                return g("compute", "latency_max_s")
            if name == "detection_range_m":
                return _detection_range(cars, cond["target"])
            if name == "overvoltage_headroom_v":
                return np.minimum(g("motor_controller", "input_abs_max_v"), g("power_board", "input_max_v")) \
                    - cars.cells * float(cond["cell_v"])
            if name == "motor_overspeed_ratio":
                return cars.cells * float(cond["cell_v"]) * cars.kv / g("motor", "max_speed_rpm")
            if name == "rail_load_ratio":
                return self._rail_ratio()
            mu = cars.mu[:, SURFACES.index(cond["surface"])] if "surface" in cond else None
            m = self.mass(cond["config"])
            if name == "stop_distance_m":
                v0 = float(cond["v0_mps"])
                return v0 * cars.t_lat + v0 * v0 / (2.0 * _stop_decel(cars, m, mu))
            if name == "top_speed_mps":
                return _top_speed(cars, m, cars.cells * float(cond["cell_v"]))
            if name == "accel_time_s":
                v1 = np.full(cars.n, float(cond["v1_mps"]))
                return _accel_time(cars, m, cars.cells * float(cond["cell_v"]), mu, v1)
        raise ValueError(MESSAGES["no_formula"] % (name,))               # not reached for a name of QUANTITIES

    def _rail_ratio(self):
        """(n,) the higher of load / rating on the supply rail of the LiDAR and on that of the compute module; the
        two loads add up when both parts are on the same rail."""
        cars, cat = self.cars, self.prep.cat
        rl = cat.f["lidar"]["rail_id"][cars.c["lidar"]]
        rc = cat.f["compute"]["rail_id"][cars.c["compute"]]
        pl = cars.g("lidar", "power_w")
        pc = cars.g("compute", "power_w")
        same = rl == rc
        rat = cat.rail_rating[cars.c["power_board"]]
        rows = np.arange(cars.n)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_l = (pl + np.where(same, pc, 0.0)) / rat[rows, rl]
            r_c = (pc + np.where(same, pl, 0.0)) / rat[rows, rc]
        return np.maximum(r_l, r_c)


def prepare(cat, car, context):
    """A Prepared object: the catalogue (an rc-catalogue/2 object or a Catalogue), the car record of the reference
    unit (an rc-car/1 record) and the context {lot_table, cycles, scenarios}, checked against each other once. It is
    what quantity(), evaluate_block() and scenario_quantities() take as `prepared`. ValueError when the three do
    not go together."""
    return Prepared(cat, car, context)


def evaluate_block(idx, cat, prepared, requests):
    """The values of several quantities on one block of cars. idx is a block of index rows (from iter_cars, or
    Catalogue.car_index for one BOM); requests = [(quantity name, conditions), ...]. Returns one (n,) float64 array
    per request, in the unit of the quantity. Laps, scenario results and the steering on a ground class are worked
    out once and shared by the requests. ValueError for an unknown quantity or for conditions that are not valid."""
    cat = load_catalogue(cat)
    if not isinstance(prepared, Prepared):
        raise ValueError(MESSAGES["not_prepared"])
    if prepared.cat is not cat:
        prepared = Prepared(cat, prepared.car, prepared.context)
    if not isinstance(requests, (list, tuple)) or not all(isinstance(r, (list, tuple)) and len(r) == 2
                                                          for r in requests):
        raise ValueError(MESSAGES["requests"])
    blk = _Block(prepared, np.asarray(idx))
    return [blk.value(name, cond) for name, cond in requests]


def quantity_block(name, idx, cat, prepared, conditions):
    """(n,) float64 values of one quantity on a block of index rows, in the unit of the quantity. A value that is
    not finite counts as not satisfied in sat()."""
    return evaluate_block(idx, cat, prepared, [(name, conditions)])[0]


def quantity(name, bom, cat, prepared, conditions):
    """The value (a float, in the unit of the quantity) of one quantity for the replica with bom = {slot: SKU code},
    under the conditions given: a mapping with exactly the keys of the quantity's signature in QUANTITIES."""
    cat = load_catalogue(cat)
    return float(quantity_block(name, cat.car_index(bom)[None, :], cat, prepared, conditions)[0])


def scenario_quantities(bom, cat, prepared, scenario):
    """{name: value} of the six scenario quantities (SCENARIO_QUANTITIES) of the replica with bom = {slot: SKU code}
    on the scenario with the id `scenario`. For a scenario without a zone zone_stop_margin_m is NaN."""
    cat = load_catalogue(cat)
    if not isinstance(prepared, Prepared):
        raise ValueError(MESSAGES["not_prepared"])
    if prepared.cat is not cat:
        prepared = Prepared(cat, prepared.car, prepared.context)
    blk = _Block(prepared, cat.car_index(bom)[None, :])
    prepared.check("lap_time_s", {"scenario": scenario})
    res = blk.scenario(scenario)
    return {k: float(res[k][0]) for k in SCENARIO_QUANTITIES}
