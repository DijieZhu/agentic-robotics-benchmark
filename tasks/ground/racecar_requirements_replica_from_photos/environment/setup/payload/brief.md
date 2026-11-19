# Brief: six replicas of the reference unit

From the club, to the requirements engineer of the lab. This brief says what we order, what is new in the way we
will use the cars, how a car is accepted, and which terms bind the order. Its items are numbered B-01 to B-16. It
has two appendices, each named by its initial: Appendix A, the acceptance, and Appendix D, the definitions.

## What we order

**B-01. The order.** We are a model racing club. We order six cars, the replicas, of the tenth-scale autonomous
race car that your lab built and has shown: the reference unit. All replicas are built to one configuration, so
one bill of materials serves all of them.

**B-02. A replica.** A replica is the reference unit built again. It has the same rolling chassis kit (BL-02), the
same deck plate, and the same layout of the autonomy layer: the LiDAR, the compute module, the motor controller
and the power distribution board sit on the deck where they sit on the reference unit. Replicate the car as the
photographs show it. Every height of this order is a height above the ground on which the tyres of the reference
unit stand; Appendix D says so in full.

**B-03. What is bought anew.** For every replica the parts of ten purchase lines are bought anew, each from the
quote: the motor, its pinion, the motor controller, the battery pack, the LiDAR, the four deck posts, the compute
module, the power distribution board, the four wheels with their tyres, and the steering servo. You choose one
quoted item for each line. Nothing else on a replica is yours to choose.

**B-04. Our pod.** Every replica carries our sensor pod on the front edge of its deck. The reference unit never
drove with it. Before the car left, your colleagues weighed our pod and tried it in its place on the reference
unit; the photographs show both (BL-10). The pod is ours: it is not an item of the order and it has no price.

## What is new in our use

**B-05. The league.** Our cars race in the autonomous class of the regional indoor league; the reference unit was
never entered. The league limits the mass of a car, the nominal voltage of its battery pack, the torque of its
motor and the angular step of its LiDAR, and it sets a pace line on its reference track. So that the cars see one
another, it also holds the scan plane of a car's LiDAR inside a band of heights above the floor. Appendix A gives
these limits.

**B-06. Our open days.** On an open day the cars run sessions for visitors on the loop in our hall, wearing our
body shell. Our programme is longer than anything the reference unit has run. A car gets through the whole
programme on one battery pack: the pack does not run flat, the bus voltage holds, and the motor does not
overheat on the hottest afternoon we plan for. A car also stops in time for a visitor who steps on the crossing.
Appendix A gives the programme, the afternoon and the crossing. The body shell is counted with
110.00 g; while a car wears it, the thermal resistances to ambient of the motor and of the motor
controller are 1.35 times their page values.

**B-07. Cable covers.** Cable covers lie across the floor of our hall, and the cars drive over them. Appendix A
gives the ground clearance a car needs for that.

**B-08. Steering on the grid.** Before every start a car stands on the rubber matting of the grid, and its stack
swings the front wheels from lock to lock to centre the steering. The servo must do this with torque in hand and
without delay. Appendix A limits the share of its stall torque that the standing car asks of the servo, and the
time from lock to lock.

**B-09. The foam bumper.** If a front wheel of a replica touches the foam bumper at full lock, we trim the foam
bumper ourselves; the trimming is not part of this order.

## How a car is accepted

**B-10. Acceptance.** A replica is acceptable when it passes every pass item of Appendix A. The appendix is exact
and it is public: it is the whole acceptance, and nothing else is judged. Each pass item is one quantity of the
public procedure `procedure.py`, computed for the replica under the conditions the item writes out and compared
with the bound of the item by the decision function of the procedure. The replica is the reference unit, as its
car record describes it (Appendix D), with the items of the bill of materials fitted in place of the parts the
reference unit was found with (BL-04), and with our pod on board. Besides the limits of B-05 to B-08 the appendix
asks that the car can be built at all and that its electrics go together: no part sees more than its maximum
input voltage from a pack fresh off the charger, and no supply rail is loaded above its rating.

**B-11. The catalogue.** Only items of the quote (`catalogue/quote.md`) can be ordered. The makers' datasheets
(`catalogue/datasheets/`) describe them. Where a maker has issued an erratum (`catalogue/errata/`), the corrected
value is the value of the page. The rolling chassis kit and the deck kit are the same on every replica; the quote
lists them as the fixed items of every car.

**B-12. Arrival lots.** Battery packs, tyres and LiDARs come to the importer in arrival lots, and a lot does not
always deliver what the maker's page prints. The quote names the arrival lot of every such item. For each lot on
the quote we take the following values, and no others:

- a lot that has an arrival report (`catalogue/lot_reports/`): the lot values that the public function
  `lot_state` of `procedure.py` gives for the samples printed for that lot in its report, taken together with the
  nominal values or the normal load that the same report prints;
- a lot that has no arrival report: the policy values of this table.

| kind of lot | policy values of a lot without an arrival report |
|---|---|
| cells (battery packs) | capacity factor 0.95; resistance factor 2.00 |
| tyres | on every ground class, the friction printed on the tyre's page times 0.85 |
| LiDARs | on every target class, the range printed on the LiDAR's page times 0.80: the white-target range for the class `white`, the other range row for the remaining classes |

These values are the lot table of the acceptance. A report covers the lot it names, and with it every item that
the quote delivers from that lot.

## Price and changes

**B-13. The price of a car.** The price of a car is the price of its parts at the quoted prices: for each of the
ten purchase lines the pieces a car needs, each at the pack price divided by the pack size, plus the fixed items of
every car. Of the acceptable cars we order the cheapest.

**B-14. Changes after the order.** Once the order is placed, every change of the bill of materials is an
engineering change. It costs a fixed fee of 45 USD for each purchase line whose item is replaced by another.

## What you deliver

**B-15. The three files.** You deliver the requirements sheet of the order, the bill of materials, and a
kinematic sketch of the reference unit as it was photographed. `submission_contract.md` gives their form. The
quantities of the sketch and the fields of the car record have the definitions of Appendix D, and no others.

**B-16. Kit radio gear.** The radio receiver and the kit speed controller that come with the rolling chassis kit
stay on a replica, as they are on the reference unit, so that a car can be driven back by hand. They are not items
to choose. On the track the motor is driven by the motor controller on the deck.

## Appendix A. Acceptance

A replica is acceptable when it passes every one of the 17 pass items of table A.1 (B-10). A pass item reads:
the quantity, computed by `procedure.py` for the replica under the conditions of the item, satisfies
`sat(quantity, comparator, bound)`. `quantities.md` describes every quantity and the decision function `sat`.

### A.1 Pass items

| id | what is judged | quantity | conditions | comparator | bound | unit | brief item |
|---|---|---|---|---|---|---|---|
| `K3` | Mass of the car with the pod. | `mass_kg` | `{"config": {"body_shell": false}}` | `<=` | 2.21 | `kg` | B-05 |
| `K4` | Nominal voltage of the battery pack. | `battery_nominal_v` | `{}` | `<=` | 8.0 | `V` | B-05 |
| `K5` | Motor torque at the current the row states. | `motor_torque_nm` | `{"current_a": 30.0}` | `<=` | 0.127 | `N.m` | B-05 |
| `K7` | Angular step of the LiDAR. | `lidar_angular_step_deg` | `{}` | `>=` | 0.22 | `deg` | B-05 |
| `K9` | Lower of the maximum input voltages of the motor controller and of the power board, minus the pack voltage at the cell voltage the row states. | `overvoltage_headroom_v` | `{"cell_v": 1.45}` | `>=` | 0.0 | `V` | B-10 |
| `K11` | Highest ratio of load to rating among the supply rails in use. | `rail_load_ratio` | `{}` | `<=` | 1.00 | `1` | B-10 |
| `W1` | Lap time on the reference track. | `lap_time_s` | `{"scenario": "SC-league"}` | `<=` | 7.21 | `s` | B-05 |
| `W2` | Distance left when the car has stopped for the target of the stop zone. | `zone_stop_margin_m` | `{"scenario": "SC-openday"}` | `>=` | 0.00 | `m` | B-06 |
| `W3` | Lowest state of charge with which a pack ends the session programme. | `schedule_energy_reserve_frac` | `{"scenario": "SC-openday"}` | `>=` | 0.21 | `1` | B-06 |
| `W4` | Lowest bus voltage of the session programme, minus the minimum input voltage of the power board. | `schedule_bus_margin_v` | `{"scenario": "SC-openday"}` | `>=` | 0.00 | `V` | B-06 |
| `W5` | Winding temperature limit of the motor, minus the peak winding temperature of the session programme. | `schedule_winding_headroom_c` | `{"scenario": "SC-openday"}` | `>=` | 0.0 | `K` | B-06 |
| `A1` | Height of the LiDAR scan plane: lower limit of the band. | `scan_plane_height_mm` | `{}` | `>=` | 116.1 | `mm` | B-05 |
| `A2` | Height of the LiDAR scan plane: upper limit of the band. | `scan_plane_height_mm` | `{}` | `<=` | 122.1 | `mm` | B-05 |
| `A3` | Ground clearance. | `ground_clearance_mm` | `{}` | `>=` | 11.4 | `mm` | B-07 |
| `A4` | Steering torque needed at standstill, over the stall torque of the servo. | `steering_torque_ratio` | `{"surface": "M"}` | `<=` | 0.58 | `1` | B-08 |
| `A5` | Time from full lock to full lock. | `lock_time_s` | `{"surface": "M"}` | `<=` | 0.40 | `s` | B-08 |
| `FIT` | The car can be built: the motor, the battery pack, the deck posts and the tyres are of the fit classes of the chassis kit; the pinion has a tooth count that the motor mount reaches; the battery pack and the power board share a connector class, and so do the motor controller and the motor; the power board has the supply rails of the LiDAR and of the compute module; and the battery pack is no longer than the usable length of the battery bay. The value is 1 when all of this holds. | `fits` | `{}` | `==` | 1 | `1` | B-10 |

The column *brief item* names the item of this brief that a row belongs to; a scenario of A.2 names its item
after its title. Where a value of this appendix needs a source, cite that item.

### A.2 The context of the conditions

The conditions name ground classes, a configuration and scenarios. They are computed in one context of the
procedure: the lot table of B-12, the cycles of A.3 and the scenarios below. A configuration says whether
the car wears the body shell of B-06.

| ground class | ground |
|---|---|
| `S1` | sealed sport floor |
| `S2` | bare sealed concrete |
| `P` | painted floor |
| `M` | rubber matting |

| target class | target |
|---|---|
| `white` | a white card |
| `grey` | a grey card |
| `black_duct` | the black duct of the barriers |
| `dark_fabric` | dark fabric |

**`SC-league`: a qualifying run of the league (B-05).**

- cycle: `CY-league`
- environment: `ambient_c` = 22; `config` = `{"body_shell": false}`
- stop zone: `zone` = `null`: none
- schedule: `repeat` = 1, `packs` = 1, `swap_after_blocks` = `null`, and the blocks of this table

| block | `drive_s` | `pace` | `rest_s` |
|---|---|---|---|
| 1 | 300 | 1 | 0 |

**`SC-openday`: an open day in our hall (B-06).**

- cycle: `CY-hall`
- environment: `ambient_c` = 33; `config` = `{"body_shell": true}`
- stop zone: `speed_cap_mps` = 3, `appear_m` = 1.65, `surface` = `P`, `target` = `{"class": "dark_fabric", "width_m": 0.3}`
- schedule: `repeat` = 3, `packs` = 1, `swap_after_blocks` = `null`, and the blocks of this table

| block | `drive_s` | `pace` | `rest_s` |
|---|---|---|---|
| 1 | 750 | 0.6 | 240 |

### A.3 The tracks

Each track is a closed lane, given segment by segment in driving order; after the last segment the lane continues
with the first. Lengths and radii are those of the centre line of the lane. Sight is how far ahead along the lane
a driver can see on the segment.

**`CY-league`: the reference track of the league.**

| segment | `kind` | `length_m` | `radius_m` | `angle_deg` | `width_m` | `surface` | `sight_m` |
|---|---|---|---|---|---|---|---|
| 1 | `straight` | 9.937 | - | - | 2 | `S1` | 9 |
| 2 | `arc` | - | 1.109 | 180 | 1.6 | `S1` | 3.2 |
| 3 | `straight` | 4.5 | - | - | 2 | `S1` | 4.5 |
| 4 | `arc` | - | 1.7 | 25 | 1.8 | `S1` | 3 |
| 5 | `arc` | - | 1.7 | 25 | 1.8 | `S1` | 3 |
| 6 | `straight` | 4 | - | - | 2 | `S1` | 4 |
| 7 | `arc` | - | 0.95 | 180 | 1.6 | `S1` | 2.6 |

**`CY-hall`: the loop in our hall.**

| segment | `kind` | `length_m` | `radius_m` | `angle_deg` | `width_m` | `surface` | `sight_m` |
|---|---|---|---|---|---|---|---|
| 1 | `straight` | 3 | - | - | 1.2 | `S2` | 8 |
| 2 | `straight` | 2.4 | - | - | 1.2 | `P` | 7.5 |
| 3 | `straight` | 5.1 | - | - | 1.2 | `S2` | 5.1 |
| 4 | `arc` | - | 0.811 | 180 | 1.1 | `S2` | 2.4 |
| 5 | `straight` | 5 | - | - | 1.2 | `S2` | 5 |
| 6 | `arc` | - | 1.5 | 30 | 1 | `S2` | 2.6 |
| 7 | `arc` | - | 1.5 | 30 | 1 | `S2` | 2.6 |
| 8 | `straight` | 4 | - | - | 1.2 | `S2` | 4 |
| 9 | `arc` | - | 0.61 | 180 | 0.8 | `S2` | 1.8 |

### A.4 `acceptance.json`

`acceptance.json` holds this appendix in machine-readable form: `pass_items` (the rows of A.1, each with its
`id`, `quantity`, `conditions`, `comparator`, `bound`, `unit` and `brief_item`), and `context` with the `cycles` and the `scenarios`
exactly as `procedure.py` takes them. Two values are `null` in the file, because this brief does not print them:
`context.lot_table`, which follows from B-12 and the lot reports, and `pod.mass_g`, which is the reading of the
scale in the photograph of the pod.

## Appendix D. Definitions

Every height is a height above the ground plane on which the tyres of the reference unit stand, with the unit as it was photographed: on the tyres and deck posts it was found with, its wheels straight ahead, without the pod. The car frame has its origin on that ground plane, under the midpoint of the rear axle (midway between the two rear wheel centres). x points forward, y to the left, z up. A wheel centre is the point of the wheel's axis midway across the width of its tyre. Lengths are in millimetres and angles in degrees. Where a quantity exists on both sides of the car, report the mean of the left and the right value.

### D.1 The quantities of the kinematic sketch

**Heights.**

| id | unit | definition |
|---|---|---|
| `deck_top_height_mm` | `mm` | Height of the flat upper face of the deck plate. |
| `deck_thickness_mm` | `mm` | Thickness of the deck plate: the distance between its upper and its lower face. |
| `lowest_point_height_mm` | `mm` | Height of the lowest point of the car, the four wheels and their tyres left out of account. |
| `highest_point_height_mm` | `mm` | Height of the highest point of the car as photographed. |
| `lidar_top_height_mm` | `mm` | Height of the top of the LiDAR's housing. |
| `antenna_top_height_mm` | `mm` | Height of the tips of the two antennas (the higher tip if they differ). |
| `post_lower_end_height_mm` | `mm` | Height of the lower end faces of the four deck posts. |
| `rear_body_post_top_height_mm` | `mm` | Height of the top ends of the two rear body posts. |

**Layout.**

| id | unit | definition |
|---|---|---|
| `deck_length_mm` | `mm` | Length of the deck plate along x: from its rearmost to its foremost edge. |
| `deck_width_mm` | `mm` | Width of the deck plate along y: across its two outermost side edges. |
| `deck_front_edge_x_mm` | `mm` | x of the foremost edge of the deck plate. |
| `lidar_x_mm` | `mm` | x of the centre of the LiDAR in plan view: midway between the foremost and the rearmost point of its housing, the connector at its rear included. |
| `compute_x_mm` | `mm` | x of the centre of the compute module in plan view: midway between its foremost and its rearmost point. |
| `motor_controller_x_mm` | `mm` | x of the centre of the motor controller board in plan view: midway between its front and its rear edge. |
| `post_spacing_x_mm` | `mm` | Distance along x between the axes of the front pair and of the rear pair of deck posts. |
| `post_spacing_y_mm` | `mm` | Distance along y between the axes of the left pair and of the right pair of deck posts. |
| `battery_bay_length_mm` | `mm` | Usable length of the battery bay: the distance along x between the surface of the chassis tub that bounds the bay behind the pack and the surface of the chassis tub that bounds it ahead of the pack, both taken on the centre line of the pack as fitted, at half the height of the pack. |

**Steering.**

| id | unit | definition |
|---|---|---|
| `kingpin_ground_offset_mm` | `mm` | King-pin offset at the ground of a front wheel. The king-pin axis is the line through the centres of the upper and the lower ball joint of the steering knuckle. The offset is the distance from the point where that axis meets the ground to the point of the ground under the wheel centre, measured along y. |
| `caster_angle_deg` | `deg` | Caster angle: the angle between the king-pin axis and the vertical in side view (projected on the x-z plane), positive when the upper ball joint lies behind the lower one. |
| `steering_arm_mm` | `mm` | Steering arm of a front knuckle: the distance in plan view (projected on the x-y plane) from the king-pin axis, taken at the height of the wheel centre, to the centre of the joint that connects the knuckle to its tie rod. |
| `bellcrank_output_arm_mm` | `mm` | Output arm of a steering bellcrank: the distance in plan view from the pivot axis of the bellcrank to the centre of its joint with the tie rod. |
| `bellcrank_input_arm_mm` | `mm` | Input arm of a steering bellcrank: the distance in plan view from the pivot axis of the bellcrank to the centre of its joint with the drag link that ties the two bellcranks together. |
| `servo_saver_arm_mm` | `mm` | Arm of the servo saver: the distance, at right angles to the axis of the servo's output shaft, from that axis to the centre of the joint where the servo link attaches to the servo saver. |
| `toe_in_per_side_deg` | `deg` | Toe-in of a front wheel with the steering straight ahead: the angle in plan view between the centre plane of the wheel and the x axis, positive when the front of the wheel points towards the centre line of the car. |

**Chassis.**

| id | unit | definition |
|---|---|---|
| `wheelbase_mm` | `mm` | Distance along x between the front and the rear wheel centres. |
| `track_front_mm` | `mm` | Front track: the distance along y between the two front wheel centres. |
| `track_rear_mm` | `mm` | Rear track: the distance along y between the two rear wheel centres. |
| `overall_length_mm` | `mm` | Overall length of the car as photographed, along x: from its rearmost to its foremost point. |
| `overall_width_mm` | `mm` | Overall width of the car as photographed, along y: across its outermost points. |

**Gears.**

| id | unit | definition |
|---|---|---|
| `pinion_teeth` | `count` | Number of teeth of the motor pinion fitted to the reference unit. |
| `spur_teeth` | `count` | Number of teeth of the spur gear that the motor pinion drives. |
| `bevel_pinion_teeth` | `count` | Number of teeth of the bevel drive pinion of the front gearbox. |
| `ring_gear_teeth` | `count` | Number of teeth of the ring gear of the front differential, the gear that this bevel pinion drives. |

### D.2 The fields of the car record

The car record (`rc-car/1`) is the description of the reference unit that `procedure.py` computes with. You fill
in your own; you do not deliver it. Every field is required.

| field | type | unit | definition |
|---|---|---|---|
| `schema` | string | - | The literal string rc-car/1. |
| `as_found_mass_g` | number | `g` | Mass of the reference unit as weighed: the reading of the scale with the whole car on it. |
| `as_found_front_axle_g` | number | `g` | Reading of the scale under the two front wheels of the reference unit, its rear wheels standing on a block level with the scale's platform. |
| `as_found_bom` | mapping | - | The SKU fitted to the reference unit on each of the nine catalogue lines other than the steering servo: a mapping from line to SKU code. |
| `as_found_servo_mass_g` | number | `g` | Mass of the steering servo fitted to the reference unit. It is not an item of the catalogue. |
| `wheelbase_mm` | number | `mm` | The sketch quantity of the same name. |
| `line_x_mm` | mapping | `mm` | For each of the ten purchase lines, the x at which the mass of the line's part acts on the reference unit: the centre of the part as fitted, and for the four wheels and the four deck posts the centre of the set. |
| `lidar_seat_height_mm` | number | `mm` | Height of the underside of the base of the LiDAR: the face it stands on. |
| `lowest_point_height_mm` | number | `mm` | The sketch quantity of the same name. |
| `chassis_top_height_mm` | number | `mm` | Height of the highest point of everything that is carried by the chassis and not by the deck. |
| `deck_load_top_height_mm` | number | `mm` | Height of the highest point of everything that is carried by the deck, the LiDAR left out of account. |
| `battery_bay_length_mm` | number | `mm` | The sketch quantity of the same name. |
| `steering_lock_deg` | number | `deg` | Steering lock: the angle through which a front wheel turns from straight ahead to full lock. |
| `kingpin_ground_offset_mm` | number | `mm` | The sketch quantity of the same name. |
| `steering_arm_mm` | number | `mm` | The sketch quantity of the same name. |
| `bellcrank_output_arm_mm` | number | `mm` | The sketch quantity of the same name. |
| `bellcrank_input_arm_mm` | number | `mm` | The sketch quantity of the same name. |
| `servo_saver_arm_mm` | number | `mm` | The sketch quantity of the same name. |
| `spur_teeth` | integer | `count` | The sketch quantity of the same name. |
| `gearbox_ratio` | number | `1` | Reduction ratio of the bevel stage of a gearbox: the teeth of the ring gear divided by the teeth of the bevel pinion. |
| `pod_mass_g` | number | `g` | Mass of the customer's sensor pod: the reading of the scale with the pod on it. |
| `pod_x_mm` | number | `mm` | x of the centre of the pod when it sits in its place on the front edge of the deck. |

`as_found_bom` has one key for each purchase line except `steering_servo`; `line_x_mm` has one key for each of
the ten purchase lines.
