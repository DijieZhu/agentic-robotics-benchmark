# Quantities and the acceptance procedure

`procedure.py` is the exact, public acceptance procedure of the order. A pass item of Appendix A of the brief
means: *computed by `procedure.py` for a replica, under the conditions the item writes out, the quantity satisfies
the bound.* This page explains every formula in prose; where this page and the code differ, the code decides.

A replica is the reference unit with the parts of a bill of materials (BOM: one SKU per purchase line) fitted in
place of the parts the reference unit was found with, and with the pod on board.

The procedure holds no number that describes a particular car, part, maker, site or use. All of those are
inputs:

- the **catalogue** (format `rc-catalogue/2`, section 2): you build it yourself from the datasheets and their
  errata and from the quote;
- the **car record** of the reference unit (format `rc-car/1`, section 3): you fill it in yourself from the
  photographs and the build log;
- the **context** (section 4): the lot table, the cycles and the scenarios that the conditions of a quantity
  name. `acceptance.json` gives the cycles and the scenarios of the acceptance; the lot table follows from the
  brief (B-12).

Nothing has a default. A condition that a quantity needs is a key of its conditions, and every key must be
given.

## 1. Using the code

```python
import json, procedure as P
cat = P.load_catalogue(json.load(open("my_catalogue.json")))
car = P.load_car(json.load(open("my_car.json")))
acc = json.load(open("acceptance.json"))
context = {"lot_table": my_lot_table, "cycles": acc["context"]["cycles"], "scenarios": acc["context"]["scenarios"]}
print(P.validate_context(context, lot_ids=cat.lot_ids))        # [] = structurally valid
prep = P.prepare(cat, car, context)
bom = {"motor": "...", "pinion": "...", "motor_controller": "...", "battery": "...", "lidar": "...",
       "deck_post": "...", "compute": "...", "power_board": "...", "tyres": "...", "steering_servo": "..."}
P.quantity("mass_kg", bom, cat, prep, {"config": {"body_shell": False}})
P.quantity("fits", bom, cat, prep, {})
P.scenario_quantities(bom, cat, prep, "<scenario id>")         # the six scenario quantities at once
for idx in P.iter_cars(cat):                                   # every combination, int16 blocks of SKU indices
    vals = P.evaluate_block(idx, cat, prep, [("fits", {}), ("lap_time_s", {"scenario": "..."})])
    price = P.price_block(idx, cat)
```

`P.sat(q, comparator, bound)` is the decision function (section 12). `P.lot_state(kind, samples)` turns the
samples of a lot report into the fields of a lot-table entry (section 4.1). `P.page_lot_table(cat)` is the lot
table that reads every lot at its page values. `P.convert(x, from_unit, to_unit)` converts between units
(`P.UNIT_FACTORS` lists them). `P.buildable(bom, cat)` is the fit by class alone (section 7). A value that
cannot be reached comes back infinite, and `sat` counts a value that is not finite as not satisfied.

## 2. The catalogue (`rc-catalogue/2`)

```json
{
 "schema": "rc-catalogue/2",
 "slot_order": ["motor", "pinion", "motor_controller", "battery", "lidar", "deck_post", "compute",
                "power_board", "tyres", "steering_servo"],
 "slots": {"<slot>": ["<SKU record>", "..."]},
 "platform": "<platform record>",
 "lots": {
  "<lot id>": {"kind": "<cell, tyre or lidar>", "skus": ["<SKU code>", "..."]}
 }
}
```

A slot is a purchase line. A car is one SKU per slot. Within a slot the procedure sorts the SKUs by code; that
position is the SKU index of `iter_cars`. All values are in the canonical unit that ends the field name (`_mm`,
`_m`, `_g`, `_s`, `_hz`, `_deg`, `_v`, `_a`, `_ohm`, `_w`, `_c` = degrees Celsius, `_k_per_w`, `_j_per_k`, `_rpm`,
`_rpm_per_v`, `_nm_per_a`, `_nm`, `_ah`, `_usd`). A page may print a value in the maker's own unit: convert it
first. Where a page has an erratum, the catalogue holds the corrected value.

**SKU record.** `sku` (the code), `qty_per_car` (4 for `deck_post` and `tyres`, otherwise 1), `pack_size`,
`pack_price_usd` (the price of one pack, as quoted), `lot` (a lot id in the slots `battery`, `lidar` and `tyres`;
`null` in the others), `fit_class` (in the slots `motor`, `battery`, `deck_post` and `tyres`), `connector_class`
(in the slots `battery`, `power_board`, `motor_controller` and `motor`), and `values`, the page fields below.
Other fields are allowed and ignored.

| slot | page fields the procedure reads |
|---|---|
| `motor` | `mass_g`, `kv_rpm_per_v`, `kt_nm_per_a`, `resistance_ohm`, `no_load_current_a`, `max_speed_rpm`, `winding_limit_c`, `rth_wc_k_per_w` (winding to case), `rth_ca_k_per_w` (case to ambient), `cth_w_j_per_k` (winding), `cth_c_j_per_k` (case) |
| `pinion` | `teeth`, `mass_g` |
| `motor_controller` | `mass_g`, `input_abs_max_v`, `current_limit_a`, `on_resistance_ohm`, `rth_k_per_w`, `cth_j_per_k`, `temp_limit_c`, `quiescent_w` |
| `battery` | `mass_g`, `length_mm` (overall length of the pack, end caps included), `cells`, `cell_nominal_v`, `capacity_ah`, `cell_resistance_ohm`, `ocv_soc` and `ocv_cell_v` (the open-circuit voltage per cell against state of charge: same length, `ocv_soc` increasing within [0, 1], `ocv_cell_v` not decreasing) |
| `lidar` | `mass_g`, `box_mm` [length, width, overall height of the housing], `optical_offset_mm` (height of the scan plane above the underside of the LiDAR's base), `range_white_m`, `range_dark_m` (the page's two range rows), `scan_rate_hz`, `angular_step_deg`, `power_w`, `supply_rail` |
| `deck_post` | `length_mm`, `mass_g` (one piece) |
| `compute` | `mass_g`, `box_mm` [length, width, height], `latency_max_s` (the page's maximum latency), `power_w`, `supply_rail` |
| `power_board` | `mass_g`, `input_min_v`, `input_max_v`, `rails` (for each rail name: `voltage_v`, `rating_w`), `quiescent_w` |
| `tyres` | `mass_g` (one wheel with its tyre), `od_mm` (outside diameter), `width_mm`, `mu_page` (the page's friction) |
| `steering_servo` | `stall_torque_nm`, `no_load_s_per_60deg` (time the unloaded servo takes to turn its output shaft through 60 degrees), `mass_g`, `case_mm` [length, width, height of the case] |

**Platform record.** It describes what is on every car and is never chosen:

| field | meaning |
|---|---|
| `kit_sku` | the code of the rolling chassis kit (BL-02) |
| `fixed_cost_usd` | the fixed items of every car as the quote prices them, added to every car's price |
| `fit` | {`motor_mount`, `battery_bay`, `deck_socket`, `wheel_hub`: the fit classes the kit takes in these four places; `pinion_teeth`: [the pinion tooth counts the motor mount reaches]}, from the data page of the kit |
| `body_shell_mass_g`, `body_shell_thermal_factor` | the body shell of the brief (B-06): its mass, and the factor on the thermal resistances to ambient of the motor and of the motor controller while it is fitted. Both or neither; a configuration with the shell needs both |

**Lots.** `lots` lists every lot id that an SKU carries, with its kind and its SKUs. A lot id is `LOT-C<digit>`
(a cell lot, carried by battery packs), `LOT-T<digit>` (a tyre lot) or `LOT-L<digit>` (a LiDAR lot).

## 3. The car record (`rc-car/1`)

The car record describes the reference unit as it was found: on the tyres and deck posts it was found with,
wheels straight ahead, without the pod. Two more fields, `pod_mass_g` and `pod_x_mm`, add the pod that every
replica carries. Appendix D of the brief defines every field. Every field is required and no other field is
allowed. Heights are heights above the ground the tyres stand on; x runs forward from the rear axle.

| field | where it enters |
|---|---|
| `schema` | the string `rc-car/1` |
| `as_found_mass_g`, `as_found_bom`, `as_found_servo_mass_g` | the mass of a replica (section 5) |
| `as_found_front_axle_g`, `wheelbase_mm`, `line_x_mm` | the front-axle load (section 5) |
| `pod_mass_g`, `pod_x_mm` | the mass and the front-axle load (section 5) |
| `lidar_seat_height_mm` | scan plane and envelope height (section 5) |
| `lowest_point_height_mm` | ground clearance (section 5) |
| `chassis_top_height_mm`, `deck_load_top_height_mm` | envelope height (section 5) |
| `battery_bay_length_mm` | `fits` (section 7) |
| `steering_lock_deg`, `kingpin_ground_offset_mm`, `steering_arm_mm`, `bellcrank_output_arm_mm`, `bellcrank_input_arm_mm`, `servo_saver_arm_mm` | the steering quantities (section 6) |
| `spur_teeth`, `gearbox_ratio` | the overall reduction of the drivetrain (section 8) |

`as_found_bom` maps each purchase line except `steering_servo` to the SKU the reference unit was found with; each
code must be an SKU of its line in the catalogue. The tyre and the deck post it names are the reference for the
heights of the record. `line_x_mm` maps each of the ten purchase lines to an x.

## 4. The context: lot table, cycles, scenarios

A context is `{lot_table, cycles, scenarios}`. `validate_context` checks its structure and nothing else.

### 4.1 Lot table

`lot_table` has exactly one entry per lot id of the catalogue, each with exactly the fields of its kind:

| kind | fields (range) | effect |
|---|---|---|
| cell | `capacity_factor` [0.5, 1.1], `resistance_factor` [0.9, 2.0] | battery `capacity_ah` x `capacity_factor`; `cell_resistance_ohm` x `resistance_factor` |
| tyre | `mu`: {`S1`, `S2`, `P`, `M`}, each [0.2, 1.5] | the tyre's friction on each ground class (the page's `mu_page` is not used) |
| LiDAR | `range_m`: {`white`, `grey`, `black_duct`, `dark_fabric`}, each [0.1, 30] | the LiDAR's range on each target class (the page rows are not used) |

Reading a lot at its page values means factors 1.0, `mu` = `mu_page` on every class, `range_m` = `range_white_m`
for `white` and `range_dark_m` for the other classes (`page_lot_table`). The quantities take the friction of a
tyre, the range of a LiDAR and the capacity and resistance factors of a pack from the lot table of the context
and from nowhere else. A lot's entry holds for every SKU of the lot.

`lot_state(kind, samples)` gives the lot-table fields of a lot from the samples of a lot report:

- cell: `capacity_factor` = mean capacity / nominal capacity and `resistance_factor` = mean cell resistance /
  nominal cell resistance; `samples` = {`nominal_capacity_ah`, `nominal_cell_resistance_ohm`, `capacity_ah`: [...],
  `cell_resistance_ohm`: [...]};
- tyre: `mu[class]` = mean pull on that class / normal load; `samples` = {`normal_load_n`, `pull_n`}, with
  `pull_n` = {class: [...]} for all four ground classes;
- LiDAR: `range_m[class]` = the smallest sample of that class; `samples` = {`range_m`}, with `range_m` =
  {class: [...]} for all four target classes.

The arithmetic is exact decimal arithmetic on the printed digits (what a hand calculation from the page gives), and
each value is rounded to 2 decimals, halves away from zero. A value outside its range in the table above is an
error.

### 4.2 Conditions

| key | value |
|---|---|
| `config` | {`body_shell`: true / false}: whether the car wears the body shell |
| `target` | {`class`: `white` / `grey` / `black_duct` / `dark_fabric`, `width_m` (0, 2]} |
| `surface` | a ground class: `S1`, `S2`, `P`, `M` |
| `cell_v` | per-cell voltage [0.8, 1.6] |
| `current_a` | motor current [0, 200] |
| `v0_mps`, `v1_mps` | speed (0, 15] |
| `scenario` | a scenario id of the context |

The pod is not a condition: it is part of the car record and rides on every replica.

### 4.3 Cycles

A cycle is `{segments: [1..64 segments]}`, driven as a closed loop (after the last segment the car continues with
the first). A segment is `{kind: straight, length_m (0, 120], width_m (0.3, 10], surface, sight_m (0, 50]}` or
`{kind: arc, radius_m [0.2, 50], angle_deg (0, 360], width_m, surface, sight_m}`. `width_m` is the lane width;
`sight_m` is how far ahead the driver can see along the lane on that segment; an arc's length is radius x angle,
and an arc has no left or right: only its radius and its angle enter. Lengths and radii are those of the lane's
centre line. A cycle is at most 120 m long; a context has at most 6 cycles.

**Resampling.** Consecutive segments that are equal in kind, radius, width, surface and sight are merged first.
The cycle is then cut into 400 equal intervals of length `ds` = length / 400. Node *i* manages interval *i*: its
curvature is the largest curvature (0 for a straight, 1 / radius for an arc) of the segments that overlap the
interval, its width and sight the smallest, its ground classes all of them (the friction at the node is the
smallest of the car's friction on those classes). A segment boundary that falls on an interval boundary (within
1e-9 of the cycle length) is not an overlap. So a short bend is never skipped and splitting a segment into equal
pieces changes nothing.

### 4.4 Scenarios and schedules

A scenario is `{cycle, schedule, environment}`; a context has at most 6. An environment is `{ambient_c [-10, 50],
config, zone}`.

`zone` is `null` or `{speed_cap_mps (0, 10], appear_m (0, 30], surface, target}`. A zone is one emergency-stop
check, not a place on the cycle: the car travels at `speed_cap_mps`, or at its top speed if that is lower, on the
zone's ground class, when a target of the given class and width appears `appear_m` ahead. The zone enters
`zone_stop_margin_m` (section 11) and nothing else: not the lap, not the energy, not the temperatures.

A schedule is `{blocks: [{drive_s (0, 7200], pace (0, 1], rest_s [0, 7200]}, 1..16], repeat [1, 16], packs [1, 4],
swap_after_blocks [1, 16] or null}`. The car drives `blocks` in order, `repeat` times (the expanded list has at
most 16 blocks, at most 4 distinct `pace` values, and drive plus rest at most 14400 s). In a block the car laps the
cycle continuously for `drive_s` seconds at the block's pace (fractions of a lap count), then stands `rest_s`
seconds with the motor stopped and the electronics on. `pace` scales the friction the driver uses and the top
speed of the lap (section 9); it does not scale the sight speed, and it does not enter the zone check.

Battery packs: with `swap_after_blocks` = *n*, expanded block *k* (counting from 0) runs on pack
min(floor(*k* / *n*), `packs` − 1), each pack fully charged when first used; with `null`, every block runs on the
first pack. Nothing is recharged inside a schedule.

## 5. The replica

The record gives the reference unit; a BOM changes it in three ways. In the formulas a line of four pieces
(`deck_post`, `tyres`) counts four times its page mass.

**Mass.** The mass of a replica, in g, is `as_found_mass_g`, less the page masses of the as-found parts of
`as_found_bom`, less `as_found_servo_mass_g`, plus the page masses of the parts of the BOM, plus `pod_mass_g`.
`mass_kg` is this mass, plus `body_shell_mass_g` when the configuration has `body_shell` true, divided by 1000.
In every formula below *m* is the mass of the car in the configuration in question.

**Front-axle load.** `front_axle_load_n` = `GRAVITY_MPS2` x (`as_found_front_axle_g` x `wheelbase_mm` + Σ over the
ten lines of (mass of the BOM's parts − mass of the as-found parts) x `line_x_mm` of the line + `pod_mass_g` x
`pod_x_mm`) / `wheelbase_mm` / 1000. For the line `steering_servo` the mass of the as-found part is
`as_found_servo_mass_g`. The body shell does not enter.

**Ride and rise.** *ride* = (tyre `od_mm` of the BOM − tyre `od_mm` of the as-found tyre) / 2. *rise* = deck post
`length_mm` of the BOM − deck post `length_mm` of the as-found post. Every height of the record moves up by the
ride. The deck plate and everything it carries move up by the rise as well, one for one (BL-07).

| quantity | value |
|---|---|
| `scan_plane_height_mm` | `lidar_seat_height_mm` + *ride* + *rise* + `optical_offset_mm` of the LiDAR |
| `ground_clearance_mm` | `lowest_point_height_mm` + *ride* |
| `envelope_height_mm` | the highest of: `chassis_top_height_mm` + *ride*; `deck_load_top_height_mm` + *ride* + *rise*; `lidar_seat_height_mm` + *ride* + *rise* + the overall height of the LiDAR's housing (`box_mm[2]`). The pod and the body shell do not count |

## 6. Steering at standstill

The steering quantities describe a standing car whose two front wheels are turned on a ground class, the
`surface` of the conditions.

- **At a wheel.** The torque about the king pin is *M* = μ *N* √(*B*² / 8 + *e*²), in N mm: μ the friction of
  the tyre's lot on the ground class, *N* half of `front_axle_load_n`, *B* the tyre's `width_mm`, *e* the
  `kingpin_ground_offset_mm` of the record.
- **Linkage ratio.** *R* = (`servo_saver_arm_mm` / `bellcrank_input_arm_mm`) x (`bellcrank_output_arm_mm` /
  `steering_arm_mm`): the angle a front wheel turns through for one unit of angle of the servo.
- **At the servo.** `steering_torque_nm` = 2 *M* *R* / `LINKAGE_EFFICIENCY` / 1000. `steering_torque_ratio` is
  this torque divided by the servo's `stall_torque_nm`.
- **Lock time.** From lock to lock the servo turns through 2 x `steering_lock_deg` / *R* degrees. Its speed
  under load is its no-load speed, 60 / `no_load_s_per_60deg` degrees per second, times (1 − `steering_torque_ratio`).
  `lock_time_s` is that angle divided by that speed; it is infinite when the torque reaches the stall torque.

## 7. Fit

`buildable` is the fit by class: the motor, battery, deck post and tyre `fit_class` equal the kit's `fit` entries
(`motor_mount`, `battery_bay`, `deck_socket`, `wheel_hub`); the pinion's `teeth` is in the kit's `pinion_teeth`;
battery and power board share a `connector_class`, and so do motor controller and motor; the LiDAR's and the
compute's `supply_rail` are rails of the power board.

The quantity `fits` is 1 when the car is buildable in this sense and, in addition, the `length_mm` of the battery
pack is no more than `battery_bay_length_mm` of the car record; otherwise it is 0. Over-voltage, over-current and
over-heating are not part of the fit: they have quantities of their own.

## 8. The reference driver and the constants

| constant | value | meaning |
|---|---|---|
| `N_NODES` | 400 | nodes per cycle |
| `GRAVITY_MPS2` | 9.81 | gravitational acceleration *g*, m/s² |
| `LATERAL_UTILISATION` | 0.85 | share of the pace-scaled friction used in a bend |
| `BRAKE_UTILISATION` | 0.8 | share of the full braking limit an emergency stop is planned with |
| `ACTUATION_LATENCY_S` | 0.02 | command to wheel, s |
| `MIN_BEAM_HITS` | 3 | beams that must fall on a target |
| `DRIVETRAIN_EFFICIENCY` | 0.85 | motor shaft to tyre, both directions |
| `ROLLING_RESISTANCE` | 0.015 | rolling-resistance coefficient, the same on every ground class |
| `DRAG_N_PER_MPS2` | 0.012 | drag force / speed squared, N per (m/s)² |
| `PROFILE_CELL_V` | 1.2 | per-cell voltage of the reference bus that speed profiles are solved at |
| `RAIL_CONVERTER_EFFICIENCY` | 0.9 | power board: input power = rail loads / this + quiescent power |
| `LINKAGE_EFFICIENCY` | 0.8 | steering linkage: share of the servo's torque that reaches the two king pins |

`stack_notes.md` prints the same twelve values. In the formulas below *m* is the mass of the car in the
configuration and *g* is `GRAVITY_MPS2`.

**Drivetrain.** Overall ratio *G* = `spur_teeth` / pinion `teeth` x `gearbox_ratio` (`spur_teeth` and
`gearbox_ratio` of the car record); wheel radius *r* = tyre `od_mm` / 2000 m; back-EMF constant *ke* = 60 /
(2π `kv_rpm_per_v`) V s/rad; *kt* = `kt_nm_per_a`. Wheel force per motor ampere: *kGd* = *kt G* η / *r* driving,
*kGb* = *kt G* / (η *r*) braking (η = `DRIVETRAIN_EFFICIENCY`). Pack resistance *Rp* = `cells` x
`cell_resistance_ohm` x `resistance_factor`; circuit resistance *R* = motor `resistance_ohm` + controller
`on_resistance_ohm` + *Rp*. Motor current at speed *v* and bus voltage *V*: *I(v)* = min(`current_limit_a`,
(*V* − *ke G v / r*) / *R*). Net tractive force on flat ground: *f(v)* = *kGd* (*I(v)* − `no_load_current_a`) −
`ROLLING_RESISTANCE` *m g* − `DRAG_N_PER_MPS2` *v²*.

**Top speed** at bus voltage *V*: the speed where *f(v)* = 0 (closed form; 0 when *f(0)* ≤ 0).

**Emergency stop.** Latency *t* = 1 / `scan_rate_hz` + compute `latency_max_s` + `ACTUATION_LATENCY_S`. Planned
deceleration on a ground of friction μ: *a_b* = `BRAKE_UTILISATION` x min(μ *g*, (*kGb* `current_limit_a` +
`ROLLING_RESISTANCE` *m g*) / *m*). Stopping distance from speed *v*: *v t* + *v²* / (2 *a_b*).

**Detection range** of a target {class, width}: min(lot `range_m[class]`, width / (`MIN_BEAM_HITS` x
`angular_step_deg` in radians)).

## 9. The speed profile (one lap)

One lap is solved for a cycle, a pace *p* and a configuration (mass *m*), at the reference bus voltage
*V* = `cells` x `PROFILE_CELL_V`. The lap does not change with the state of charge. At node *i* with curvature κ,
friction μ and sight *s*:

- friction budget *A* = *p* μ *g*;
- the speed *v* is limited by the smallest of three limits: *p* x top speed at *V*; in a bend, *v²* ≤
  `LATERAL_UTILISATION` *A* / κ; the sight speed, from which an emergency stop fits in *s* (latency *t* and *a_b*
  as in section 8, with the full friction μ, not scaled by the pace).

Forward pass: from node *i* to *i* + 1 the squared speed rises at most by 2 *a* `ds`, *a* = max(0, min(√(*A²* −
(*v²* κ)²), *f(v)* / *m*)) evaluated at node *i* (friction ellipse at the pace, motor at the bus). Backward pass:
from node *i* + 1 back to *i* by 2 *a* `ds`, *a* = min(√(*A²* − (*v²* κ)²), (*kGb* `current_limit_a` +
`ROLLING_RESISTANCE` *m g* + `DRAG_N_PER_MPS2` *v²*) / *m*) evaluated at node *i* + 1. Each pass goes twice around
the loop (the second time it may stop at the first node where nothing changes; the result is the same).

Between consecutive nodes the acceleration is constant: d*t* = 2 `ds` / (*v_i* + *v_i+1*), mean speed
*v̄* = (*v_i* + *v_i+1*) / 2, force *F* = *m* (*v_i+1²* − *v_i²*) / (2 `ds`) + `ROLLING_RESISTANCE` *m g* +
`DRAG_N_PER_MPS2` *v̄²*. Driving (*F* > 0): motor current *I* = *F* / *kGd* + `no_load_current_a`; braking:
*I* = −*F* / *kGb*; both capped at `current_limit_a`. Traction battery current (driving only; braking returns
nothing to the pack) = *I* (*ke G v̄ / r* + *I* (motor `resistance_ohm` + `on_resistance_ohm`)) / *V*. The lap
gives: lap time *T*, traction charge *Q* = Σ battery current d*t*, the integral of the motor current squared
*S* = Σ *I²* d*t*, and the peak traction battery current *I_pk*.

Hotel load (always on, also at rest): *P_h* = (LiDAR `power_w` + compute `power_w`) / `RAIL_CONVERTER_EFFICIENCY`
+ power board `quiescent_w` + controller `quiescent_w`; current *I_h* = *P_h* / *V*.

## 10. Over a schedule

Each expanded block uses the lap of its pace. Per block, on its pack (charge *C* = `capacity_ah` x
`capacity_factor` x 3600 A s, state of charge starting at 1 for a fresh pack):

- drive: SoC falls by (*Q / T* + *I_h*) `drive_s` / *C*; then the bus voltage at the end of the drive, under the
  lap's peak current, is `cells` x OCV(SoC) − (*I_pk* + *I_h*) *Rp* (OCV per cell interpolated linearly in the
  page table and held at its ends);
- rest: SoC falls by *I_h* `rest_s` / *C*.

The lowest bus voltage of a schedule is the smallest of these end-of-drive voltages over all its blocks. A pack
ends the schedule with the state of charge it has when the schedule leaves it for the next pack, or when the
schedule is over. The state of charge is not held at 0: a negative value says by how much the pack falls short.

Thermal models, starting at the ambient temperature and running through all blocks. Motor: two nodes (winding,
case), winding-to-case resistance `rth_wc_k_per_w`, case-to-ambient `rth_ca_k_per_w` (x
`body_shell_thermal_factor` when `body_shell` is true), capacities `cth_w_j_per_k` and `cth_c_j_per_k`; heat in
the winding = (*S / T*) x motor `resistance_ohm` while driving, 0 at rest; solved in closed form per block, the
peak includes a maximum inside a block. Controller: one node, resistance `rth_k_per_w` (x the shell factor),
capacity `cth_j_per_k`, heat (*S / T*) x `on_resistance_ohm` + `quiescent_w` while driving and `quiescent_w` at
rest.

## 11. The dictionary

28 quantities; `unit` is exactly the string shown; the conditions of a quantity are exactly the keys shown (`{}`
where the column shows a dash).

| quantity | unit | conditions | value |
|---|---|---|---|
| `mass_kg` | `kg` | `config` | the mass of the replica in the configuration (section 5) |
| `front_axle_load_n` | `N` | — | the static load on the front axle (section 5) |
| `scan_plane_height_mm` | `mm` | — | height of the LiDAR scan plane (section 5) |
| `ground_clearance_mm` | `mm` | — | height of the lowest point, wheels and tyres left out of account (section 5) |
| `envelope_height_mm` | `mm` | — | height of the highest point (section 5) |
| `fits` | `1` | — | 1 when the car can be built, else 0 (section 7) |
| `battery_nominal_v` | `V` | — | `cells` x `cell_nominal_v` |
| `motor_torque_nm` | `N.m` | `current_a` | *kt* x max(`current_a` − `no_load_current_a`, 0) |
| `lidar_scan_rate_hz` | `Hz` | — | `scan_rate_hz` |
| `lidar_angular_step_deg` | `deg` | — | `angular_step_deg` |
| `unit_cost_usd` | `USD` | — | Σ slots `qty_per_car` x `pack_price_usd` / `pack_size` + `fixed_cost_usd`: the price of a car at the quoted prices |
| `compute_latency_s` | `s` | — | `latency_max_s` |
| `detection_range_m` | `m` | `target` | detection range (section 8) |
| `overvoltage_headroom_v` | `V` | `cell_v` | min(controller `input_abs_max_v`, power board `input_max_v`) − `cells` x `cell_v` |
| `motor_overspeed_ratio` | `1` | `cell_v` | `cells` x `cell_v` x `kv_rpm_per_v` / `max_speed_rpm` |
| `rail_load_ratio` | `1` | — | largest over the LiDAR's and the compute's supply rails of (loads on that rail) / its `rating_w` |
| `stop_distance_m` | `m` | `surface`, `v0_mps`, `config` | emergency stopping distance from `v0_mps` (section 8) |
| `top_speed_mps` | `m/s` | `cell_v`, `config` | top speed at *V* = `cells` x `cell_v` (section 8) |
| `accel_time_s` | `s` | `surface`, `v1_mps`, `cell_v`, `config` | time from rest to `v1_mps` with d*v*/d*t* = min(μ *g*, *f(v)* / *m*) at *V* = `cells` x `cell_v` (closed form); infinite at or above the top speed |
| `lap_time_s` | `s` | `scenario` | lap time *T* at the scenario's highest pace |
| `zone_stop_margin_m` | `m` | `scenario` | min(`appear_m`, detection range of the zone's target) − stopping distance from *v_z* on the zone's surface, *v_z* = min(`speed_cap_mps`, top speed at *V*); needs a scenario with a zone |
| `schedule_energy_reserve_frac` | `1` | `scenario` | the lowest state of charge any pack ends the schedule with |
| `schedule_bus_margin_v` | `V` | `scenario` | lowest bus voltage over the schedule − power board `input_min_v` |
| `schedule_winding_headroom_c` | `K` | `scenario` | motor `winding_limit_c` − peak winding temperature over the schedule |
| `schedule_controller_headroom_c` | `K` | `scenario` | controller `temp_limit_c` − peak controller temperature over the schedule |
| `steering_torque_nm` | `N.m` | `surface` | torque the servo must give at standstill (section 6) |
| `steering_torque_ratio` | `1` | `surface` | that torque over the servo's stall torque (section 6) |
| `lock_time_s` | `s` | `surface` | time from full lock to full lock (section 6) |

The six quantities from `lap_time_s` to `schedule_controller_headroom_c` are the scenario quantities
(`P.SCENARIO_QUANTITIES`). They use the scenario's environment: its configuration for the mass, its ambient
temperature, its body shell and its zone. In the zone check *V* is the reference bus voltage of section 9.

## 12. Decision function and limits

`sat(q, comparator, bound)`: for `>=` and `<=`, with *d* = +1 for `>=` and -1 for `<=`, satisfied when
*d* (*q* − bound) ≥ -1e-9 x max(|bound|, 1). For `==`, which is meant for a quantity that is 1 or 0 such as
`fits`: satisfied when *q* equals the bound. A value that is not finite is not satisfied. The same function
decides every pass item of the acceptance.

Limits of a context: at most 6 cycles and 6 scenarios; at most 64 segments and 120 m per cycle; at most 16
expanded blocks, 4 distinct paces and 14400 s per schedule; ids of cycles and scenarios match
`^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$`; integers at most 2^53 in magnitude.
