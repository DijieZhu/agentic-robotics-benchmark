# DA-6 desktop arm: unit calibration record

Document DA6-CAL, revision A. Format `calibration-record/1`.

Every DA-6 is shipped with the calibration record of its six servos, taken at final assembly. The record relates the
servos' tick readings to the design angles of the kinematics sheet
([../robot/kinematics_sheet.md](../robot/kinematics_sheet.md)) on that particular unit. The motion controller receives
the record of the unit it drives at runtime, in `setup` (`unit["calibration"]`;
[../controller/controller_interface.md](../controller/controller_interface.md), section 4). The file
[example_calibration_record.json](example_calibration_record.json) is the record of the demonstration unit MU-DEMO1.

## 1. Why every unit has its own record

The servo horns are pressed onto the output splines at assembly. The spline has many teeth and the tooth on which a
horn ends up varies from unit to unit, so the reading of a servo at a given design angle differs between units by
several degrees. The direction of each servo follows from its mounting; on the DA-6 servo 4 (`wrist_pitch`) is mounted
reversed.

## 2. How the record is taken

1. The arm is mounted on the assembly jig, which holds every joint at its design zero: the zero pose of the kinematics
   sheet (the arm straight up, the jaw at its reference angle with the pads parallel).
2. Each servo's present position is read and stored as `zero_ticks` (whole ticks).
3. The design-angle limits of the kinematics sheet are converted through the new zero (section 5), rounded inwards to
   whole ticks, written into the servo's EEPROM angle-limit registers and copied into the record as `min_ticks` and
   `max_ticks`.

Uncertainty: the jig locates each arm joint within ±0.5° and the jaw within ±1.0° of its design zero, so the design
angle at which a servo actually reads `zero_ticks` lies within ±0.5° (arm joints) or ±1.0° (jaw) of 0, plus the
rounding of `zero_ticks` to whole ticks (±0.044°). The record states nothing else about the unit: the tolerances of
its geometry, masses and servo dynamics are given as ranges in the drawings ([../robot/drawings/](../robot/drawings/)
and the kinematics sheet, section 7), the mass table ([../robot/mass_properties.md](../robot/mass_properties.md)) and
the servo datasheet ([../servo/servo_datasheet.md](../servo/servo_datasheet.md)).

## 3. Format

A JSON object:

| key | type | content |
|---|---|---|
| format | string | `"calibration-record/1"` |
| serial | string | the unit's serial number (the same as `unit["serial"]` in setup) |
| convention | string | the conversion of section 4, in words |
| uncertainty_deg | object | `{"arm joints": 0.5, "jaw": 1.0}`: the uncertainty of the zero (section 2) |
| servos | object | one entry per joint name (`base_yaw`, `shoulder_pitch`, `elbow_pitch`, `wrist_pitch`, `wrist_roll`, `jaw`) |

Each entry of `servos`:

| key | type | content |
|---|---|---|
| id | int | the servo's bus id, 1 to 6 (the order of every bus array) |
| direction | +1 or -1 | +1: the ticks increase with the design angle; -1: they decrease (the servo is mounted reversed) |
| zero_ticks | int | the servo's reading at design angle 0 |
| min_ticks | int | the servo's EEPROM min angle limit (ticks) |
| max_ticks | int | the servo's EEPROM max angle limit (ticks) |

Every tick value of a DA-6 record lies in 0 to 4095, like every reading of the unit (servo datasheet, section 6).

## 4. Conversions

```
design angle   q     = direction x (ticks - zero_ticks) x 2 pi / 4096          [rad]
goal           ticks = round(zero_ticks + direction x q x 4096 / (2 pi))      [whole ticks]
speed          dq/dt = direction x speed x 2 pi / 4096                       [rad/s]
load           tau   = direction x load / 1000 x 2.94 N m                     [N m along +q]
```

The load conversion uses the typical reading scale; the scale of an individual servo's reading lies within 0.9-1.1 and
the reading carries a noise of about 1 % of 2.94 N m (servo datasheet). A design angle derived from a reading carries
the uncertainty of section 2 besides the quantisation of one tick (0.0879°).

## 5. The EEPROM angle limits

`min_ticks` and `max_ticks` are the design-angle limits [q_lo, q_hi] of the kinematics sheet written through the
record and rounded inwards:

```
direction +1:  min_ticks = ceil(zero_ticks + q_lo x 4096 / (2 pi)),  max_ticks = floor(zero_ticks + q_hi x 4096 / (2 pi))
direction -1:  min_ticks = ceil(zero_ticks - q_hi x 4096 / (2 pi)),  max_ticks = floor(zero_ticks - q_lo x 4096 / (2 pi))
```

Converted back through the record they lie inside the sheet's limits, within one tick of them. The servo clamps every
goal to them (bus interface manual, section 4). Because the zero of the record has the uncertainty of section 2, the
design angle at which a joint actually stops at a limit lies within ±0.5° (jaw ±1.0°) of the sheet's limit, plus the
rounding.

## 6. The home pose

The operator's homing tool brings the arm to its home pose (the design angles of the acceptance specification,
[../acceptance/acceptance_spec.md](../acceptance/acceptance_spec.md), section 1.1) through the unit's record: each
goal is the conversion of section 4, limited to [min_ticks, max_ticks]. The readings at the home pose therefore
convert back through the record to the home pose's design angles (to a tick or two), while the arm's actual angles
differ from them by the record's error (section 2) and the servos' elastic deflection.

## 7. Example: the record of MU-DEMO1

```json
{
 "format": "calibration-record/1",
 "serial": "MU-DEMO1",
 "convention": "q = direction x (ticks - zero_ticks) x 2 pi / 4096 (q = design angle of the kinematics sheet)",
 "uncertainty_deg": {
  "arm joints": 0.5,
  "jaw": 1.0
 },
 "servos": {
  "base_yaw": {
   "id": 1,
   "direction": 1,
   "zero_ticks": 1714,
   "min_ticks": 517,
   "max_ticks": 2933
  },
  "shoulder_pitch": {
   "id": 2,
   "direction": 1,
   "zero_ticks": 1844,
   "min_ticks": 744,
   "max_ticks": 2801
  },
  "elbow_pitch": {
   "id": 3,
   "direction": 1,
   "zero_ticks": 2535,
   "min_ticks": 1373,
   "max_ticks": 2744
  },
  "wrist_pitch": {
   "id": 4,
   "direction": -1,
   "zero_ticks": 1892,
   "min_ticks": 945,
   "max_ticks": 2827
  },
  "wrist_roll": {
   "id": 5,
   "direction": 1,
   "zero_ticks": 2168,
   "min_ticks": 379,
   "max_ticks": 3660
  },
  "jaw": {
   "id": 6,
   "direction": 1,
   "zero_ticks": 1687,
   "min_ticks": 1629,
   "max_ticks": 2758
  }
 }
}
```

| id | joint | direction | zero_ticks | min_ticks | max_ticks | limits through the record (°) | design-angle limits, kinematics sheet (°) |
|---|---|---|---|---|---|---|---|
| 1 | base_yaw | +1 | 1714 | 517 | 2933 | -105.21 to 107.14 | -105.29 to 107.14 |
| 2 | shoulder_pitch | +1 | 1844 | 744 | 2801 | -96.68 to 84.11 | -96.71 to 84.17 |
| 3 | elbow_pitch | +1 | 2535 | 1373 | 2744 | -102.13 to 18.37 | -102.15 to 18.44 |
| 4 | wrist_pitch | -1 | 1892 | 945 | 2827 | -82.18 to 83.23 | -82.24 to 83.25 |
| 5 | wrist_roll | +1 | 2168 | 379 | 3660 | -157.24 to 131.13 | -157.25 to 131.21 |
| 6 | jaw | +1 | 1687 | 1629 | 2758 | -5.10 to 94.13 | -5.12 to 94.19 |

Through this record the homing tool commands the home pose of the shelf cell (jaw 8.59°) as the goals `[1714, 2072,
1590, 2707, 2168, 1785]` (servos 1 to 6). Example conversions: [../servo/bus_interface.md](../servo/bus_interface.md),
section 6.
