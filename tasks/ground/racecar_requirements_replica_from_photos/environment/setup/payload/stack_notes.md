# Notes on the driving stack

These notes describe how the driving stack of the cars drives and steers, as far as the acceptance has to know
it. The public procedure `procedure.py` implements the same model, and `quantities.md` has every formula. Where
these notes and the code differ, the code decides.

## 1. Pace

The stack drives at a pace, a coefficient between 0 and 1: the `pace` of a schedule block. At 1 the driver drives
at full pace. The coefficient scales the grip that the driver is willing to use, in bends, under power and on the
brakes, and it scales the top speed that the driver lets the car reach. Of the grip scaled in this way the driver
uses the share `LATERAL_UTILISATION` = 0.85 in a bend. The pace does not change the speed limit that the driver's
sight sets (section 3), and it does not enter the emergency-stop check of a zone.

## 2. Sessions and standing

The stack runs a car in blocks. In a block the car laps the loop at the block's pace until the session timer runs
out, wherever the car then is on the loop, and after that it stands for the block's standing time.

While a car stands its motor is stopped and its electronics stay on: the LiDAR, the compute module, the power
distribution board and the motor controller go on drawing power. The LiDAR and the compute module are fed through
the supply rails of the power distribution board, whose converters work at the efficiency
`RAIL_CONVERTER_EFFICIENCY` = 0.9. A schedule of blocks holds no charging: a pack is fully charged when the
schedule first uses it, and after that it is only drawn down.

## 3. Reaction latency and braking

Between the moment an obstacle comes into view and the moment the brakes bite, a car travels on at its speed for
the reaction latency. The reaction latency is a scan period of the LiDAR (the reciprocal of its scan rate), plus
the maximum latency that the compute module's page prints, plus the actuation latency from command to wheel,
`ACTUATION_LATENCY_S` = 0.02 s.

The driver plans a stop with the share `BRAKE_UTILISATION` = 0.8 of the full braking limit. The full limit is
the smaller of what the tyres hold on the ground and what the motor can brake. The driver never goes faster than
the speed from which such a stop fits into the distance it can see ahead along the lane. This sight speed is
worked out with the full grip of the ground, whatever the pace.

## 4. Detection

The stack accepts a target when at least `MIN_BEAM_HITS` = 3 beams of a scan fall on it. The detection range of
a target is therefore the smaller of the LiDAR's range on a target of that class and the distance at which the
width of the target still spans that many angular steps.

## 5. Lap model

The procedure resamples a loop to `N_NODES` = 400 nodes, evenly spaced along the lane, and solves a lap at the
reference bus voltage, which is `PROFILE_CELL_V` = 1.2 V for every cell of the pack. The lap does not change as
the pack empties. Gravitational acceleration is `GRAVITY_MPS2` = 9.81 m/s².

## 6. Drivetrain and road load

Between the motor shaft and the tyres the drivetrain passes on the share `DRIVETRAIN_EFFICIENCY` = 0.85 of
the motor's torque when the car is driven, and it takes the same loss when the motor brakes. Rolling resistance
is `ROLLING_RESISTANCE` = 0.015 times the weight of the car, on every ground class. Air drag is the speed
squared times `DRAG_N_PER_MPS2` = 0.012 N per (m/s)².

## 7. Steering at standstill

Before a start the stack swings the front wheels of the standing car from one lock to the other. A tyre that is
turned on the spot scrubs over its contact patch, and the patch is dragged around the king pin as well; both
resist the servo. The servo works on the two wheels through the servo saver, the two
bellcranks and the tie rods. This linkage passes on the share `LINKAGE_EFFICIENCY` = 0.8 of the servo's
torque to the two king pins. Under load a servo turns more slowly than its page's no-load figure, in proportion
to the share of its stall torque that is left over. `quantities.md` (section 6) has the formulas.

## 8. Constants of the model

The model has these constants and no others. Each value below is the value in `procedure.py`.

| constant | value | meaning |
|---|---|---|
| `N_NODES` | 400 | nodes that a loop is resampled to |
| `GRAVITY_MPS2` | 9.81 | gravitational acceleration, m/s² |
| `LATERAL_UTILISATION` | 0.85 | share of the pace-scaled grip that the driver uses in a bend |
| `BRAKE_UTILISATION` | 0.8 | share of the full braking limit that a stop is planned with |
| `ACTUATION_LATENCY_S` | 0.02 | latency from command to wheel, s |
| `MIN_BEAM_HITS` | 3 | beams that must fall on a target |
| `DRIVETRAIN_EFFICIENCY` | 0.85 | motor shaft to tyre, driving and braking |
| `ROLLING_RESISTANCE` | 0.015 | rolling-resistance coefficient, the same on every ground class |
| `DRAG_N_PER_MPS2` | 0.012 | drag force divided by the speed squared, N per (m/s)² |
| `PROFILE_CELL_V` | 1.2 | voltage per cell of the reference bus that a lap is solved at, V |
| `RAIL_CONVERTER_EFFICIENCY` | 0.9 | efficiency of the supply-rail converters of the power distribution board |
| `LINKAGE_EFFICIENCY` | 0.8 | share of the servo's torque that the steering linkage passes on to the two king pins |

## 9. Exact unit definitions

A data page may print a value in a unit other than the one its catalogue field is named for. `procedure.convert`
converts between the units of `procedure.UNIT_FACTORS`. Its table is built from six numbers that are exact by the
definition of the unit:

| constant | value | meaning |
|---|---|---|
| `INCH_M` | 0.0254 | one inch in metres |
| `FOOT_M` | 0.3048 | one foot in metres |
| `MILE_M` | 1609.344 | one mile in metres |
| `POUND_KG` | 0.45359237 | one pound in kilograms |
| `STANDARD_GRAVITY_MPS2` | 9.80665 | one kilogram-force in newtons |
| `CELSIUS_ZERO_K` | 273.15 | zero degrees Celsius in kelvin |

`STANDARD_GRAVITY_MPS2` serves the units only (kgf, kgf.cm, oz.in). The driver and the steering model use
`GRAVITY_MPS2`.
