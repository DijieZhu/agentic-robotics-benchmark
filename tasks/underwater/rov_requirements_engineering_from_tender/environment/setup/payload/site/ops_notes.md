# Operating notes: intake works

Kept by the plant's operations group for staff and contractors who work at the intake.

## Reservoir gauge

- The zero of the reservoir gauge is at EL 1425.00 m.
- The stage file, `reservoir_stage_daily.csv`, holds one gauge reading for each day. A reading is a height above
  the gauge zero. It is not an elevation.

## Weather station

- The plant's weather station stands on the dam crest. Its daily values are kept in the stage file beside the
  gauge reading of the same day, in four columns:
  - `air_min_c` and `air_max_c`: the lowest and the highest air temperature of the day, in degC;
  - `precip_mm`: the precipitation of the day, in mm of water;
  - `wind_mean_ms`: the mean wind speed of the day, in m/s.
- A weather day runs from midnight to midnight, local time.

## Clocks and time stamps

- The plant clock runs on local time, and every permit to work is written in local time. Local time is
  UTC-7 all year. Daylight saving time is not observed.
- Exports from the control system are stamped in UTC. The hourly discharge of the units,
  `unit_discharge_hourly.csv`, is such an export. Each stamp marks the start of its hour.

## Generating units

- The plant has two generating units. They are numbered 1 and 2 from the left bank.
- The discharge file has one column for each unit.

## Access and the top of the tower

- The intake tower is reached on foot over the footbridge from the dam crest. Cars and vans stay on the crest.
- There is no room or other enclosed space on the tower; the gate hoist stands on a gantry.
- A socket for contractors' equipment stands at the foot of the instrument davit: 230 V, 16 A. It is live
  whenever a permit to work is open at the intake.
- The deck lights are fed from the panel of the gate hoist and are switched from the control room.
- Battery packs of contractors' equipment are charged in the powerhouse workshop, not on the tower.
- Contractors' tools and spares are kept in the store beside the powerhouse workshop between permits.

## Lifting gear

- The lifting set kept for the instrument davit (sling, shackle and swivel) weighs 3.0 kg.

## Seasonal routine

- Forebay: the draw of the units keeps the water at the tower open in every month of the year.
- Spring: before the refill the gauge well is flushed and the staff gauge is checked against the survey mark.
