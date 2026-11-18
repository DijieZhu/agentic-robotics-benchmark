# Station scope: what is fitted where

This sheet says what is fitted to the vehicle upstream of this station, at this station and at the next one. The trim
sheet (`trim_sheet.png`) carries the same split in its column "state", item by item, and the photographs of the
finished vehicle show it, the wiring harness and the tether termination excepted.

## Upstream: the base vehicle

The vehicle arrives at this station as the base vehicle (`~/given/parts/base_vehicle.stl`), clamped in the
positioner's fork:

* the frame: the bottom plate, the two side plates and the centre plates, with the two tube cradles for the
  electronics enclosure and the clamps of the battery enclosure;
* the battery enclosure with its battery pack;
* the 4 horizontal ducted thrusters;
* the vehicle's own gripper with its mount (the work gripper of the drawings and the trim sheet);
* the Doppler velocity log with its bracket (the velocity log of the trim sheet);
* the 4 lower guard brackets.

Nothing of this is touched at this station.

## This station

* ballast blocks on the underside of the bottom plate, as the vehicle's trim needs them for the water on its work
  order. The station's box holds 12 blocks;
* the electronics enclosure, on the two tube cradles;
* the 4 vertical ducted thrusters, on the outer faces of the side plates;
* the 2 thruster guards, on the lower guard brackets.

The station drives the screws itself: when a unit is in place, the cell fastens it (`~/given/cell/cell_manual.md`,
"Fastening"). The trim sheet tables the screws of this station's units: those of the enclosure, of the guards and of
the ballast blocks in the unit's own row, the mounting screws of the vertical thrusters in a row of their own.

## Next station

Not fitted here, and not in the cell:

* the 4 corner fairing shells with their 4 buoyancy foam blocks;
* the 2 round buoyancy foam blocks with the 2 centre fairing caps;
* the scanning imaging sonar with its frame mount;
* the 4 lights;
* the 4 upper guard brackets, which clamp the guards from above;
* the wiring harness and the tether termination.

The vehicle leaves the factory with all of this on it. The acceptance of its trim is for that state: the items of the
next station enter with their values of the trim sheet.

## The work order

Every vehicle comes with a work order. It names the unit and the water at the site the vehicle is delivered to: the
water's temperature in degrees Celsius and its practical salinity. `work_order_example.json` shows the form. The cell
gives the work order of the vehicle on the fork as `describe()["work_order"]`. The values of the example belong to no
graded run.
