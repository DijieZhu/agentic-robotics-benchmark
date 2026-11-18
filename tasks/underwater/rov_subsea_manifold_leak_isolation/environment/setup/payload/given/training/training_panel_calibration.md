# Training tank: calibration sheet

Calibration data of the training tank and of its training panel segment: the handle torques, the settings a run can
make, and what the instruments record. The arrangement is in the tank drawing.

## 1. Water

- Sea water of salinity 35 at 10 degrees Celsius. Density for weights and buoyancy: 1024.8 kg/m^3, the density for
  which the vehicle manual gives the vehicle's trim state.
- Turbidity is set per run to one of three levels, given as the beam attenuation coefficient at 550 nm:

| Level | Beam attenuation coefficient [1/m] |
|---|---|
| `clear` | 0.40 |
| `medium` | 0.80 |
| `murky` | 1.20 |

## 2. Training panel handles

Running and breakaway torque of each handle, measured at the stem. The heaviest handle is at the upper limit of the
field's torque class (panel standard, section 3).

| Handle | Running torque [N m] | Breakaway torque [N m] |
|---|---|---|
| BV1 | 2.0 | 3.0 |
| BV2 | 3.5 | 5.5 |
| BV3 | 5.0 | 8.0 |
| PV1 | 2.5 | 4.0 |
| PV2 | 4.0 | 6.0 |
| PV3 | 3.0 | 4.5 |

The detent near the stops is that of the standard. Each handle can be set open or closed for the start of a run.

## 3. Test port and hot stab

- The pressure behind test port TP-1 is held by a pressure unit at the value set for the run, between 0 bar and 60
  bar. It does not bleed down. Mating force, seal and release follow the panel standard, section 5.4.
- The hot stab starts latched in PARK-A or, if set so, in TP-1.
- Each receptacle has a status lamp with three states: `empty`, `engaged` (a stab is in the guide) and `latched`.

## 4. Flow generator

- Set speed between 0.0 m/s and 0.5 m/s along +x, at the centre of the tank 2 m above the floor.
- A run gives either one set speed or a list of up to 8 steps (time in the run, set speed). The generator moves to a
  new set speed at 0.02 m/s^2.

## 5. Bubbler

- Gas flow off, or between 0.5 L/min and 2.9 L/min at the depth of the diffuser.
- The bubbles are of the size of the field's seeps (leak detection note). Their rise speed in this water is not listed
  here; a run can measure it.

## 6. Settings of a run

| Setting | Option of `training/run_session.py` | Values | Default |
|---|---|---|---|
| `current_m_s` | `--current`, `--current-steps` | 0.0 to 0.5 m/s, or up to 8 steps | 0 |
| `bubbler_L_min` | `--bubbler` | 0 (off), or 0.5 to 2.9 L/min | 0 (off) |
| `turbidity` | `--turbidity` | `clear`, `medium`, `murky` | `clear` |
| `test_port_bar` | `--port-bar` | 0 to 60 bar | 0 |
| `handles` | `--handle TAG=STATE` | `open` or `closed` for any of BV1, BV2, BV3, PV1, PV2, PV3 | all open |
| `d1_stand` | `--d1` | `panel`, `bay` | `panel` |
| `stab_port` | `--stab-port` | `PARK-A`, `TP-1` | `PARK-A` |

A setting that is left out takes its default. A value outside its range is refused and uses up no run.

## 7. What the instruments record

- The tracking system gives the pose of the vehicle 5 times a second: the position of the origin of the vehicle's body
  frame in the tank frame T and the orientation of the body frame as a unit quaternion (w, x, y, z).
- The encoders give the angle of every handle of the training panel from its open stop, in degrees.
- One sample of the record has these columns:

| Column | Meaning | Unit |
|---|---|---|
| `t` | time of the sample in the run | s |
| `x` | position of the body-frame origin in T | m |
| `y` |  | m |
| `z` |  | m |
| `qw` | orientation of the body frame in T, unit quaternion | - |
| `qx` |  | - |
| `qy` |  | - |
| `qz` |  | - |
| `BV1_deg` | encoder angle of handle BV1 from its open stop | deg |
| `BV2_deg` | encoder angle of handle BV2 from its open stop | deg |
| `BV3_deg` | encoder angle of handle BV3 from its open stop | deg |
| `PV1_deg` | encoder angle of handle PV1 from its open stop | deg |
| `PV2_deg` | encoder angle of handle PV2 from its open stop | deg |
| `PV3_deg` | encoder angle of handle PV3 from its open stop | deg |

- With the samples the record gives the present encoder angles, the state of the two receptacle lamps, and the set
  points in force: current set speed and direction, bubbler gas flow, turbidity level with its coefficient, and test
  port pressure.
- `instruments()` returns the samples since the previous call; the record of the whole run is returned when the run
  ends and is written by `training/run_session.py` into the run's file.
- The record holds measurements only.