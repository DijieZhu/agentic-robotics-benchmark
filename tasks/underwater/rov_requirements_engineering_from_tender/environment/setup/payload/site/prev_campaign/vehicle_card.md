# Vehicle data card: rented inspection vehicle

Filled in by the vehicle crew for the Owner's file of the previous inspection campaign. The vehicle was rented for
that campaign. It is not a vehicle of the Owner.

## 1. Vehicle

- Inspection-class ROV with an open frame and six thrusters: four horizontal thrusters, vectored at
  35 deg to the long axis of the vehicle, and two vertical thrusters.
- A thruster command is a pulse width in microseconds; 1500 us is stop. Thrusters 1 to 4 are the
  horizontal thrusters, thrusters 5 and 6 the vertical ones. For a pull astern, thrusters 1 and 2 run in reverse and
  thrusters 3 and 4 run forward.
- One thruster at bollard, by command and bus voltage. A negative thrust is a reverse thrust. Between the two
  voltages, interpolate.

| Command (us) | Thrust at 12 V (kgf) | Current at 12 V (A) | Thrust at 16 V (kgf) | Current at 16 V (A) |
|---|---|---|---|---|
| 1100 | -3.40 | 18.3 | -4.76 | 25.6 |
| 1150 | -2.70 | 12.9 | -3.77 | 18.0 |
| 1200 | -2.05 | 8.5 | -2.88 | 11.9 |
| 1250 | -1.48 | 5.2 | -2.07 | 7.3 |
| 1300 | -0.98 | 2.8 | -1.37 | 3.9 |
| 1350 | -0.56 | 1.2 | -0.79 | 1.7 |
| 1400 | -0.24 | 0.3 | -0.33 | 0.5 |
| 1450 | -0.03 | 0.0 | -0.04 | 0.0 |
| 1500 | 0.00 | 0.0 | 0.00 | 0.0 |
| 1550 | 0.04 | 0.0 | 0.06 | 0.0 |
| 1600 | 0.30 | 0.3 | 0.41 | 0.5 |
| 1650 | 0.70 | 1.2 | 0.98 | 1.7 |
| 1700 | 1.22 | 2.8 | 1.71 | 3.9 |
| 1750 | 1.85 | 5.2 | 2.59 | 7.3 |
| 1800 | 2.57 | 8.5 | 3.59 | 11.9 |
| 1850 | 3.37 | 12.9 | 4.72 | 18.0 |
| 1900 | 4.25 | 18.3 | 5.95 | 25.6 |

- Surge drag of the vehicle without its tether, from the maker's tow test:

| Speed (m/s) | Drag (N) |
|---|---|
| 0.25 | 5.0 |
| 0.50 | 15.5 |
| 0.75 | 31.5 |
| 1.00 | 53.0 |

- Hotel load with the lights off: 58 W.

## 2. Launch and recovery

- Launch and recovery: from the instrument davit on the tower deck.

## 3. Tether and payout counter

- Tether: 9.0 mm in diameter.
- Tether counter: zeroed with the vehicle-end mark at the guard-rail fairlead beside the reel. From there it counts
  the tether paid out, in metres.
- Deck lead: from the fairlead along the outside of the guard rail, round the corner, over the guide sheave under
  the davit head, then down. 10.0 m of tether lie between the fairlead and deck level at the davit.

## 4. Lights

- Lights: 36 W together at 100 %.

## 5. Sensors and telemetry columns

- Depth: from a pressure sensor. The display and the telemetry compute the depth with a fluid density of
  1025 kg/m3, the sea-water setting.
- The telemetry files hold one row for each second of a dive.

| Column | Unit | Content |
|---|---|---|
| `utc` | - | time stamp, UTC |
| `depth_m` | m | depth as displayed |
| `heading_deg`, `pitch_deg`, `roll_deg` | deg | attitude |
| `bus_v` | V | bus voltage |
| `bus_a` | A | bus current: hotel load, lights and thrusters together |
| `pwm1` to `pwm6` | us | commands of thrusters 1 to 6 |
| `lights_pct` | % | light setting |
| `tether_out_m` | m | reading of the tether counter |
| `enclosure_t_c` | degC | temperature inside the electronics enclosure |
| `mode` | - | control mode |
