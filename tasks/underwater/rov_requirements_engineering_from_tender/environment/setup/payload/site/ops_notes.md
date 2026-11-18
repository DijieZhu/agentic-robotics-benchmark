# Operating notes: intake works

Kept by the plant's operations group for staff and contractors who work at the intake.

## Reservoir gauge

- The zero of the reservoir gauge is at EL 1420.00 m.
- The stage file, `reservoir_stage_daily.csv`, holds one gauge reading for each day. A reading is a height above
  the gauge zero. It is not an elevation.

## Clocks and time stamps

- The plant clock runs on local time, and every permit to work is written in local time. Local time is
  UTC-7 all year. Daylight saving time is not observed.
- Exports from the control system are stamped in UTC. The hourly discharge of the units,
  `unit_discharge_hourly.csv`, is such an export. Each stamp marks the start of its hour.

## Generating units

- The plant has two generating units. They are numbered 1 and 2 from the left bank.
- The discharge file has one column for each unit.

## Lifting gear

- The lifting set kept for the instrument davit (sling, shackle and swivel) weighs 3.0 kg.
