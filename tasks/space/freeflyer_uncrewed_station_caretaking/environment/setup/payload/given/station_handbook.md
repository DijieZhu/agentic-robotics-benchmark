# Station handbook: cabin air, flight rules, stowage units, dock and map

This handbook is the station's side of a sortie: how the cabin air of the modules is moved and what holds a loose item
on a return inlet, the rules the station sets for a robot that works in it alone, the lights, the stowage units, the
dock, the zones of the map, and how old the map is. The robot is described in `robot_manual.md`, its calls in
`robot_api.md`, the station's telemetry in `telemetry_dictionary.md`, the lost-item register in
`lost_item_register.md`. The map itself is in `station_map/`.

Lengths are in metres, pressures in pascals (Pa), forces in newtons (N), air flows in cubic feet per minute (cfm), as
the station's records give them.

## 1. The station and its modules

A sortie takes place in 3 modules of the station's laboratory segment, joined by open hatches:

| module | name | open hatch to | return inlets | cabin air unit | stowage unit |
|---|---|---|---|---|---|
| `L2` | laboratory module L2 (the dock is here) | `H2` | 3 | `CAU-L2` | `SB-L2` |
| `H2` | hub H2 | `L2` and `L1` | 2 | `CAU-H2` | `SB-H2` |
| `L1` | laboratory module L1 | `H2` | 3 | `CAU-L1` | `SB-L1` |

The other hatches of these modules are closed during a sortie, the hatch to laboratory module `L3` among them. `L3` is
the rehearsal facility (`rehearsal_guide.md`); it is no part of a sortie.

The station is uncrewed during a sortie: nobody moves, stows or services anything, and nothing on board gives off CO2.

Positions are given in the station frame of the map: x forward, y to starboard, z toward the deck. The words of
direction in this handbook and on the walls (forward, aft, starboard, port, deck, overhead) are directions of that
frame. `station_map/landmarks.json` gives each module's axis, its walls and the rack location codes.

## 2. The cabin air system

### 2.1 Architecture

Each module has its own cabin air unit, `CAU-<module>`: a fan that blows air into the module through supply diffusers
in the overhead corners and draws it back through the module's return inlets (`figures/air_loop.png`). The return
inlets of a unit are in parallel on the unit's return duct: `L2` has 3, `H2` has 2 and `L1` has 3, 8 in all, with the
ids `RI-<module>-<n>`. Each return inlet is designed for 40 cfm. A unit's actual flow lies within 10 % of the design
flow of its inlets together. About 130 cfm of air passes between neighbouring modules through each open hatch.

A unit has 2 pressure sensors on its return duct. They measure the unit's **return pressure**: the pressure difference
that the fan draws across the unit's return inlets (channels `return_dp_a` and `return_dp_b`). The return pressure is
one value for the whole unit: every return inlet of a unit sees the same return pressure, and no channel of the
telemetry tells the inlets of a unit apart.

### 2.2 The return inlet and its builds

Every return inlet is the same part (`figures/return_inlet.png`): a frame of 0.46 × 0.18 m on the wall, 30 mm deep,
with an opening of 0.40 × 0.12 m (0.048 m²) flush with the frame's front. 7 slats run across the opening (open-area
ratio 0.55); behind them lie a debris screen (20 mesh) and the filter medium. A label plate on the frame carries the
inlet's id. `station_map/fixtures.json` gives each inlet's centre, its normal into the cabin and the directions of the
opening's long and short side.

There are 2 builds, which differ in the medium:

| build | medium behind the debris screen | return pressure of a unit of this build |
|---|---|---|
| `pad-type` | a pre-filter pad | 20 to 65 Pa |
| `element-type` | an inlet filter element | 140 to 240 Pa |

All return inlets of a cabin air unit have the same build. Which build a unit has is in the sortie's configuration
record (`mission()["config_record"]`), not in the map. Inside the range of its build a unit's return pressure depends
on the state of its medium and on its inlets: it rises as the medium loads with dust, and it rises when something
covers part of an opening.

### 2.3 Air in front of a return inlet

The mean air speed through an inlet's opening is about 0.4 m/s. 10 cm in front of the opening it is 0.15 to 0.25 m/s;
from 20 to 30 cm on it has fallen to the cabin's background of about 0.1 m/s. The return inlets of this station draw
harder than a crew's comfort would ask for: they are sized to collect loose debris, and what drifts loose in a module
ends up on a return inlet.

### 2.4 What holds an item on a return inlet

An item that lies on an opening is held there by the pressure difference across it:

    holding force ≈ return pressure × area of the opening that the item covers        (N = Pa × m²)

The return pressure is the unit's, as its sensors read it at that moment. The covered area is the part of the 0.048 m²
opening that the item lies over. An item on an opening also raises the unit's return pressure, because the rest of the
unit's open area has to pass the unit's flow.

### 2.5 Stopping a cabin air unit's fan

`station_request("cau_fan", module, "off")` stops the fan of the module's cabin air unit; the rules for it are in
section 3.1. The fan runs down with a time constant of about 3 s. The unit's return pressure falls with it toward
zero, and so does the suction of every return inlet of the unit; the unit's supply into the module ceases. `"on"`
restarts the fan, which runs up in the same way. The live telemetry shows the run-down and the run-up
(`telemetry_dictionary.md`).

What a stop does to the other items on the inlets of the same unit:

- in a stop of up to 20 s, an item that nothing touches stays where it is;
- in a stop of 40 s or longer, a pouch that hangs over the edge of an inlet's frame can tilt and lose its hold, and
  some drift away.

The screen blowers of the stowage units belong to no cabin air unit: a fan stop does not change what a stowage unit
holds (section 5.1).

## 3. Flight rules and station rules

The station is run by its automation while it is uncrewed. The rules of this section are the station's; the automation
enforces the limits on fan stops itself and keeps a record of every violation.

### 3.1 Fan stops

Fire detection relies on the ventilation to carry smoke to the detectors, so a cabin air unit may be stopped only
briefly and rarely:

| rule | limit | what the station does at the limit |
|---|---|---|
| stops of a unit in a sortie | at most 2 | a further stop request is answered `REJECTED_QUOTA` |
| length of a stop | at most 60 s | the station restarts the fan itself 60 s after the stop began; the reply to an accepted stop names that moment (`auto_restart_at`, mission seconds) |
| interval between stops of a unit | at least 120 s from the unit's last restart, whoever restarted it | an earlier stop request is answered `REJECTED_INTERVAL` |

The robot may ask for the restart (`"on"`) at any time. The station books a **violation of its flight rules** for each
of these:

- a stop request to a unit that has used its 2 stops (`third_stop_request`);
- a stop request inside the interval (`stop_within_gap`);
- a stop that lasts until the station's own restart (`overrun_auto_restart`).

Not violations: a stop request to a unit whose stops are inhibited, `"off"` for a fan that is already stopped
(`REJECTED_STATE`), and `"on"` for a fan that is running (`ACCEPTED`, nothing changes).

**Inhibited units.** The configuration record can mark the stops of a unit as inhibited (`fan_stop`: `"inhibited"`),
with the reason in `inhibit_reason`: "smoke detector check in progress" or "equipment cooling requirement". Every stop
request to such a unit is answered `REJECTED_INHIBITED`, for the whole sortie; its fan keeps running.

### 3.2 Docking

- **Charge.** The robot docks with at least 15 % of charge. The charge that counts is the one at the moment of
  docking, at every docking of the sortie: charging on the berth afterwards does not make up for an arrival below it.
- **Nothing rides onto the dock.** When the robot docks, and while it is docked, no lost item is in its gripper or
  caught between the robot and the dock.

### 3.3 Stowing

- The holding screen of a stowage unit takes sheet-like items only.
- A pouch pad takes pouches only; a pad is made for 1 pouch.

A pouch on a holding screen and a sheet-like item on a pouch pad are not stowed. Which classes of the lost-item
register are sheet-like and which are pouches is in `lost_item_register.md`.

### 3.4 Keep-out

The robot stays out of the keep-out boxes of the map (section 7).

## 4. Lights

The cabin lights of each module have 2 modes: `general` and `night` (dimmed). The configuration record gives the mode
each module is commanded to at the start of the sortie (`lights`). `station_request("lights", module, mode)` switches
a module's lights. The station takes at most 6 such requests in a sortie and answers `REJECTED_QUOTA` after that; this
is no violation of the flight rules.

**A light circuit can have failed.** Its module is then dark, whatever mode is commanded. The station accepts a
request for such a module like any other, and the channel `lights.<module>` reports the commanded mode: the telemetry
says what was commanded, never whether the lights are on.

What the light means for the robot's localisation and for its frames is in `robot_manual.md` (localisation, colour
frames). The map was surveyed with every module in general mode.

## 5. The stowage unit

### 5.1 Faces and blower

Each module has a stowage unit, `SB-<module>`, mounted on a wall (`figures/stowage_unit.png`). A unit has 3 suction
faces in a common plane, 30 mm proud of the wall:

| face | id | opening |
|---|---|---|
| holding screen | `SB-<module>-S` | 0.36 × 0.36 m (0.1296 m²), in a square frame of 0.42 m |
| pouch pad | `SB-<module>-P1` | 0.20 × 0.26 m (0.052 m²), its centre 0.35 m from the screen's centre |
| pouch pad | `SB-<module>-P2` | the same, on the other side of the screen |

The unit's outline on the wall is 0.96 × 0.564 m. `station_map/fixtures.json` gives the centre of each face, the
unit's normal into the cabin, the direction from pad `-P1` to pad `-P2` (`pad_axis`) and the direction across it
(`cross_axis`).

The faces are perforated. Behind them the unit's own **screen blower** draws cabin air through all of them. The blower
is always on, belongs to no cabin air unit, takes no request and is not affected by a fan stop. The air speed through
an open face is about 0.015 m/s. A face holds what lies flat on it by suction: a sheet-like item on the holding screen
with about 4 to 5 N, a pouch on a pouch pad with about 1 N.

### 5.2 Fiducial plates

A unit carries 4 fiducial plates of 0.10 × 0.06 m, which `fly_rel` flies on: `-F1` beside the holding screen on its
deck side, `-F2` beside the holding screen on its overhead side, `-F3` beside pouch pad `-P1` and `-F4` beside pouch
pad `-P2`, each on the deck side of its pad. A plate's frame has its origin at the plate's centre, x along the plate's
long side, z along its normal into the cabin and y = z × x. `station_map/fixtures.json` lists every plate with its
centre and axes at the unit's registered position.

### 5.3 Approach zone

In front of each unit lies its **approach zone**: a box over the unit's outline grown by 0.2 m on every side, reaching
from the wall 1.0 m into the cabin. The map gives it as an oriented box (`approach_zone`: centre, axes, half sizes).
What the zone changes for a `fly_rel` with the arm deployed is in `robot_manual.md` (the zone check). The dock has an
approach zone of the same kind.

### 5.4 Where a unit is

The map shows each unit where it was registered at the survey. A unit can have been moved since, by up to 0.3 m along
its wall; its faces, its plates and its approach zone move with it. `fly_rel` flies on a plate where the plate really
is.

## 6. The dock

The dock is in `L2` and has 2 berths. `station_map/fixtures.json` gives, for each berth, the approach point in front
of it, the attitude there and the axis along which the robot leaves, and for the dock its fiducial plate `DOCK-F1` and
its approach zone. `mission()["berth"]` is the berth the robot starts on. What the robot does at the dock (leaving,
berthing, charging) is in `robot_manual.md`; the station's rules for docking are in section 3.2.

## 7. Keep-in and keep-out zones

`station_map/zones.json` holds 12 keep-in boxes and 4 keep-out boxes, axis-aligned in the station frame.

- **Keep-in**: the volume that the survey found free for flight. The flyable volume is the union of the keep-in boxes
  without the keep-out boxes.
- **Keep-out**: volumes around station equipment that the robot must not enter (all of them are in `L2`).

The robot's firmware checks its moves against these boxes with a margin of its own (`robot_manual.md`, the zone
check). For the station a box is a box: the robot is inside a keep-out box while its body origin is inside the box as
the map gives it.

## 8. The map and its age

`station_map/` is the as-surveyed map: `zones.json`; `fixtures.json` (the return inlets, the cabin air units, 101 crew
handrails, the stowage units, the dock); `landmarks.json` (the modules, the open hatches, the rack location codes, the
signs printed on the walls); and `views/` (a plan, an elevation and a picture of each wall, indexed in
`views/index.json`). The survey was made 41 days before the sortie, with the lights of every module in general mode,
no loose item in the modules and the robot away. Positions are survey values to 0.01 m.

What the map does not know:

- where the stowage units are now (section 5.4);
- how the modules are lit now (section 4);
- where the lost items are (`lost_item_register.md`).

The fixed hardware (the return inlets, the crew handrails, the dock, the walls with their signs) is where the map
shows it. Each of the map's files has a part `rehearsal_module` with the same content for `L3`.

## 9. Station requests at a glance

`station_request(kind, target, value)` returns at once with the station's answer; what a request does shows in the
telemetry and in the frames. `target` is a module, `L2`, `H2` or `L1`.

| kind | value | answers |
|---|---|---|
| `cau_fan` | `off` | `ACCEPTED` with `auto_restart_at`; else, checked in this order: `REJECTED_INHIBITED`, `REJECTED_STATE`, `REJECTED_QUOTA`, `REJECTED_INTERVAL` |
| `cau_fan` | `on` | `ACCEPTED` (a stopped fan restarts; a running fan is left as it is) |
| `lights` | `general` or `night` | `ACCEPTED`; `REJECTED_QUOTA` after 6 requests |

| answer | meaning |
|---|---|
| `ACCEPTED` | the station took the request |
| `REJECTED_INHIBITED` | the unit's stops are inhibited (configuration record); no violation (section 3.1) |
| `REJECTED_STATE` | `off` for a fan that is already stopped; no violation |
| `REJECTED_QUOTA` | a stop request after the unit's 2 stops (a violation), or a lights request after 6 (no violation) |
| `REJECTED_INTERVAL` | a stop request less than 120 s after the unit's last restart (a violation) |
| `INVALID_ARGS` | an unknown kind, target or value; nothing happened |
| `TIME_UP` | the site's simulated time is used up (`robot_api.md`) |
