# Facility notes for the inspection team (plant owner's briefing)

These notes go with the documents the robot receives for a building (`robot.mission()`): the general arrangement
drawing, the radio link plan, the work order with the round sheet's data, and the modification record. They say what
those documents are, where they can be trusted and where not. They hold for every auxiliary building of the plant
that gets a robotic round. Coordinates are in metres from the airlock datum (the site frame of the robot manual).

## 1. The buildings

Each auxiliary building is a single-storey block of about 30 m by 21 m behind an airlock: corridors with rooms on
both sides (pump rooms, valve galleries, tank rooms, electrical rooms, stores, a waste store, plant rooms), and a pipe
chase that runs between rooms and is entered through low hatches. The buildings are in care and surveillance: cooling
water, drains, fire water, ventilation and the electrical supplies still run, so the gauges read and the indicator
lamps are lit.

Rounds were walked until 2009. Then dose rates rose and entry was stopped; the last round sheet is that year's. The
process has been run differently since, so a value recorded in 2009 says nothing about the value today.

## 2. The drawing

The general arrangement drawing shows the building as it was at its last revision (`drawing.revision_date`): rooms,
walls, doors, the larger equipment, the column grid with its marker plates and the radiological zoning. Its note says
so: modifications after the revision date are not shown. The radio link's base station and repeaters are on the link
plan (`link`), a document of its own.

<!-- BEGIN GENERATED drawing_values -->
| field | values in the training hall's and the practice buildings' documents |
|---|---|
| `rooms[].type` | `airlock`, `change_room`, `corridor`, `decon_room`, `electrical_room`, `hall`, `hvac_room`, `office`, `pipe_chase`, `pump_room`, `sample_lab`, `store`, `tank_room`, `valve_gallery`, `waste_store`, `workshop` |
| `walls[].kind` (thickness, m) | `outer` (0.3); `shield` (0.8); `wall` (0.25) |
| `doors[].kind` | `door`, `double_door`, `fire_door`, `hatch` |
| `doors[].note` | "STAIR 2 (no access)"; "airlock outer door"; "low access hatch, clear height ... m" |
| `equipment[].kind` | `ahu`, `check_source`, `deployment_pad`, `divider`, `drum_store`, `heat_exchanger`, `mcc`, `pallets`, `panel`, `partition`, `pipe_rack`, `plate_board`, `platform`, `post`, `pump_set`, `rack`, `shield`, `shield_wall`, `switchgear`, `tank`, `vessel` |
| `zoning.zones[].zone_class` | `controlled`, `high_radiation`, `supervised` |

These lists hold what occurs in the documents given here. Another building's drawing can use a kind of room, door or equipment that is not among them; its entry carries the same fields.
<!-- END GENERATED drawing_values -->

- **Sheets.** The drawing is on four sheets, two by two over the building's outline, cut at its mid-lines
  (`drawing.sheets[]`: number, rectangle, `missing`). The archive's copy of a building's drawing does not always hold
  all four. Where a sheet is missing, its `missing` is true and the drawing's notes (`drawing.notes`) hold the line
  `SHEET n OF 4 MISSING FROM THE ARCHIVE COPY`.
  The sheet at the airlock, the one with the first stretch of corridor C01, is in every copy.
- **What a missing sheet takes away.** A room is on the sheet that holds the centre of its rectangle. The rooms of a
  missing sheet are not in `rooms[]`, and nothing of them is drawn: no equipment, no zone, no grid marker plate, no
  door between two such rooms, no wall that only they share. A corridor that runs onto the missing sheet is drawn up
  to a cut within 1 m of the sheet's edge: its entry gives the drawn part and carries `continues: true`. Beyond the
  cut it is drawn no more than the missing rooms are.
- **What stays.** `building.outline` is the outline of the whole building; the airlock stands in front of it
  (`building.airlock`). The outer walls stand on the outline on every sheet, their inner face 0.125 m inside it. They
  have three doors and no other opening: the door from the airlock into corridor C01; the stair door at the low-x
  end of corridor C03, which is kept shut; and the door of the annex (section 3). Where the sheet with that end of
  C03 is missing, the stair door is not drawn, like everything else there; it is shut all the same. A wall that a
  missing room shares with a drawn room, or with the drawn part of a corridor, is drawn in full, with its doors: such
  a door's entry names the missing room by its number (`doors[].rooms`). The link plan shows the base station and
  both repeaters wherever they hang. The work order, the round sheet and the modification record are whole: the round
  sheet records the location of an instrument or a survey mark on a missing sheet as it does every other one (room
  number and name, position, height, facing), and an entry of the record can name a room of the missing sheet.
- **Room numbers.** The airlock is room 100. The corridors are C01, the leg the airlock opens into; C03, the leg
  parallel to it; and C02, the cross leg that joins their far ends. The rooms are numbered in rows, each row from the
  airlock's end of the building (low x) to the far end, without a gap: 101, 102, ... along the outer side of C01;
  121, 122, ... between C01 and C03; 141, 142, ... along the outer side of C03; 161, 162, ... in the block beyond
  C02, with rising y. A number that is missing among the drawn rooms of a row is a room of a missing sheet.
- **Room plates.** Every room of the drawing's sheets except the corridors, those of a missing sheet included, has a
  plate with its number and name in the plant lettering (catalogue, section 6). It hangs outside the room, on the
  wall beside its main door: the plate's centre is 1.6 m above the floor and 0.28 m from the edge of the door's
  opening. The annex (section 3) has no plate.
- **Doors.** A door's entry gives the middle of its opening, its width, the two rooms it joins, how its leaf is hung,
  and its state on the drawing (`drawn_state`): `open as drawn` for every interior door, `exterior, kept shut` for the
  airlock's outer door and the stair door. Those two lead out of the building and stay shut: they are no way for the
  robot, and a map must not treat them as openings. Interior doors were left standing open. One of them was sealed
  later (section 3). A few others may stand ajar, the leaf part-way in the opening, so that the robot may not get
  through; none of them is the only way to any part of the building. The leaves are heavy: the robot cannot move
  one. A `hatch` is a low opening into the pipe chase; its note gives its clear height.
<!-- BEGIN GENERATED ajar -->
- Doors ajar in the practice buildings: `t3` two; `t1`, `t2` none.
<!-- END GENERATED ajar -->
- **Equipment** is drawn as outlines with a label and a kind. What stands in a room besides is not drawn: storage
  racks, pallets with drums or boxes, pipes, benches, cabinets. Such things also stand along walls, in the airlock,
  and just inside doors: something in the way behind an open door does not make it a sealed door (section 3).
- The round sheet's locations were copied from this drawing's instrument index when the drawing was whole.

**Where a sheet is missing** the robot has nothing drawn to match and no grid marker plate to read. It has its
odometry; what its own sensors have seen (a wall it has scanned can be matched again); the outer walls, whose line is
known from the outline; the walls of the neighbouring sheets, which are drawn; the room plates; the recorded position
of an instrument that is still at its place, once its tag plate is read; and the survey mark's disc on the floor.

## 3. What was changed after the last revision

Ordinary plant modifications were made between the drawing's last revision and 2009 and never drawn, because the
operators knew the building:

- **Doors were sealed.** A sealed door is shut, with steel straps across the leaf and the notice "DOOR SEALED - NO
  ACCESS"; a door without them is not sealed. The room behind it stayed in use and is reached another way: through an
  opening made in another wall or through the pipe chase.
- **Wall openings were made**: plain openings without a door, where a new way was wanted.
- **Temporary shielding was erected**: in some buildings a wall of shield blocks stands across a corridor and closes
  it. It carries the sign "TEMPORARY SHIELDING - NO ENTRY". The way on leads round it.
- **Instruments were moved** to another wall of their room. The old position was then used for another instrument of
  the same kind, with its own tag.
- **Instruments were relocated** to a neighbouring room or to the corridor wall outside their room. The old mounting
  plate was left empty.
- **An annex was built on** (below).
- **A link repeater went out of service.** The link plan shows two repeaters, both in service; one of them no longer
  relays. A repeater that works shows a green lamp on its housing.
- **The process changed.** Activity collected where there was none at the time of the zoning survey: each building
  holds one radiation source that no document lists.

Each building's documents carry the owner's statements about them (`facility_facts`). In the practice buildings they
read:

<!-- BEGIN GENERATED facts -->
> 1. A listed instrument is in the room the round sheet gives, in a neighbouring room, or on the corridor wall outside that room.
> 2. An instrument without a recorded location is in the one room that is on no sheet of the drawing: an annex built against the outside of an outer-wall room and entered through a door that is not drawn.
> 3. About a third of the listed instruments are not at their drawn place or have no recorded location.
> 4. One drawn interior door of the building is sealed and cannot be passed. A few other doors may stand ajar; none of them is the only way to any part of the building.
> 5. One sheet of the drawing is missing from the archive copy; nothing of its rooms is drawn. A corridor that runs onto it is drawn up to a cut within 1 m of the sheet edge, and its entry says that it continues. Besides the rooms of the missing sheet, one room is on no sheet: the annex.
> 6. One unlisted source exists somewhere in the building, in an area the drawing shows as supervised; the zoning was not updated after it appeared.
> 7. Grid marker plates hang where the drawing shows them; about three in four of them still exist. Where a sheet is missing, no plate is listed and none hangs.
> 8. Both link repeaters are shown in service; their state was not checked. The link plan shows them wherever they hang.
> 9. Some of the listed instruments are on the missing sheet; the round sheet gives their recorded location as for every other instrument.
> 10. The building's outer walls stand on the outline on every sheet, their inner face 0.125 m inside it.
> 11. Every room of the drawing's sheets but the corridors, those of the missing sheet included, has its number on the plate beside its main door.
> 12. Every listed instrument carries a tag plate with its tag in the plant lettering (one to three letters, a hyphen, two to five digits, at most eight characters); no two plates of the building carry the same tag; an instrument is the listed one only if its plate says so.
<!-- END GENERATED facts -->

In more words:

- **Where a listed instrument can be.** In the room the round sheet gives; or in a room that shares a wall with that
  room; or on the corridor wall just outside that room: on the corridor's side of one of that room's own walls, within
  the room's extent along the wall (0.3 m beyond its corners at most). Nowhere else. All of this holds on a missing
  sheet as on a drawn one: the recorded room can be a missing room, and so can the neighbouring room an instrument
  was relocated to.
- **What the recorded location is.** The round sheet's location of an instrument (room, position, height and
  `facing_deg`, the direction a face looks) is that of the recorded place: the place where the instrument hung when
  the drawing was made. An instrument that is still there is where the round sheet records it and faces as recorded,
  to the sheet's rounding (catalogue, section 8). For an instrument that was moved or relocated, position, height
  and facing are the old place's: the recorded facing is the direction a face looks at the recorded place (where a
  moved instrument's successor hangs today), not the direction the listed instrument looks now. That instrument
  hangs on another wall, which can face any other way, and no document says which. An instrument without a recorded
  location has no recorded facing either.
- **The annex.** Besides the rooms of a missing sheet, exactly one room of a building is on no sheet at all: an annex
  built against the outside of a room that lies on the building's outer wall. Apart from the airlock it is the
  only room outside `building.outline`. It is entered from that room, through a door in the outer wall that no sheet shows. An
  instrument whose location the round sheet does not record is in the annex.
- **How many.** About a third of the listed instruments are not where the round sheet records them, counting the one
  without a recorded location. One drawn interior door is sealed and cannot be passed. A shielding wall or a further
  wall opening was added in some buildings, never more than two such changes in one. Where a sheet is missing, some
  of the listed instruments are on it.
- **What you can count on.** Every listed instrument carries a tag plate with its tag (section 5), and the round
  sheet's data of the instrument (scale and unit, case size, window length, number and colours of lamps, surface type
  and component of a temperature point) are the instrument's own, also for one that was moved; its recorded location
  is not (above). The survey mark is at
  its recorded place; where its sheet is drawn, it lies in a zone drawn as controlled. The rooms that hold listed
  instruments and the survey mark have their lighting on. The base station and the repeaters are where the link plan
  shows them. The sealed door is a door the drawing shows, and the unlisted source lies in an area the drawing shows
  as supervised (section 4).
- **Other instruments.** A building holds several times as many instruments as the round lists. Some of them carry
  tag plates of their own, most carry none. An instrument at a recorded position is the listed one only if its tag
  plate says so.

**What the robot meets on the floor.** Door sills, hoses and pipes laid across floors, steel covers of cable
trenches, ramped cable protectors, wet patches, low pipe racks, pipe bridges and pipe cross-overs, narrow gaps between
equipment, the low hatches of the pipe chase, and the bund of the tank room. The drawing shows the doors and the
hatches and none of the rest. Such things lie at doors, and also inside rooms and across corridors, away from
anything drawn: a hose, a floor pipe or a trench cover can run across a room from wall to wall, and a low pipe bridge
can stand across a room outside the pipe chase, with the back of the room behind it. In every building the corridor
is crossed, between the airlock and the rest of the building, by a fire-door sill or a hose that is too high for
plain walking.

**The bund.** Every building has one tank room, and a spill kerb (the bund) runs across it from wall to wall: the
tanks stand behind it. There are two kinds of bund in the plant. In some buildings the kerb is 5 to 6 cm high, within
what the robot's `high_step` mode is rated for; in the others it is 11 to 18 cm high, which no gait mode steps over.
One ramp crosses the kerb in either kind: 0.9 m wide, with raised side edges 0.12 m high, no steeper than 15°. Where
the kerb is high, that ramp is the only way in and the only way out. Neither the kerb nor the ramp is on the
drawing. A listed instrument or the survey mark can stand behind the bund.

## 4. Radiological zoning

The zoning on the drawing is the survey of `zoning.survey_date`. Each zone is an area with a class:

<!-- BEGIN GENERATED zone_classes -->
| `zone_class` | upper bound in `zoning.classes` (`zoning.unit`: Gy/h) | dose rate (Gy/h) | dosimeter (counts per second) |
|---|---|---|---|
| `supervised` | 0.1675 | up to 0.1675 | up to 7.5 |
| `controlled` | 1.675 | up to 1.675 | up to 75.4 |
| `high_radiation` | null (no upper bound) | above 1.675 | above 75.4 |
<!-- END GENERATED zone_classes -->

The zoning shows the sources that were known then; the work order lists them by room (`work_order.documented_sources`:
the waste store's drums and the process equipment that holds activity). For those sources it still holds: on the
floor the robot can reach, an area drawn `supervised` or `controlled` stays within its class. It was never updated for
the unlisted source, which lies in an area the drawing shows as `supervised`; the dose rate around it is higher than
that class allows: a dosimeter that reads well above its zone's class is the sign of it.

No zone is drawn for the rooms of a missing sheet. A listed source can be in one of them: the work order still names
its room, and nothing else warns of it. There the dosimeter, not the drawing, is what tells the dose rate ahead
(site rules, section 1).

## 5. Tags and tag plates

<!-- BEGIN GENERATED tag_letters -->
| tag letters | instrument | round-sheet `type` |
|---|---|---|
| `PI-` | dial gauge | `gauge` |
| `TI-` | dial thermometer (a dial gauge with a °C scale) | `gauge` |
| `LG-` | tubular level gauge | `level` |
| `HV-` | quarter-turn valve with a lever | `valve_lever` |
| `V-` | rising-stem gate valve with a handwheel | `valve_handwheel` |
| `LCP-` | local control panel: a row of indicator lamps | `lamps` |
| `TS-` | temperature point: a surface whose temperature is taken with the thermal camera | `thermal` |
| `SM-` | survey mark on the floor | `dose_point` |

The item of the unlisted source has the tag `X` in the work order (`work_order.items[]`); the round sheet's line for it (`round_sheet.items[]`) is tagged `UNLISTED`; it has no tag plate, and `report_source` takes no tag.
<!-- END GENERATED tag_letters -->

- A tag is one to three capital letters, a hyphen and two to five digits, at most eight characters (`PI-1033`,
  `V-1217`, `LCP-1454`).
- The digits of an instrument's tag begin with the number of the room the instrument was installed in when the
  drawing was made (a corridor `C01` counts as `01`): `PI-1033` was installed in room 103. The tag did not change
  when an instrument was moved. The tag of an instrument without a recorded location also begins with a room number,
  but not with the number of the room it is in. A survey mark's number says nothing about its room.
- Every tagged instrument has a tag plate next to it: a signal yellow plate with a black border line and the tag
  engraved in black, in the plant lettering, and nothing else on it. The catalogue (section 6) gives the plate, its
  border line and every glyph of the lettering in numbers, and for each type of instrument where its plate hangs
  (section 8). The plate carries no code: the tag is read from its letters.
- No two plates of a building carry the same tag, but tags that differ in a single character are common: the
  instruments of a room were numbered one after another.
- The plant keeps signal yellow, above the floor, for tag plates. The only other things that carry it are hazard
  stripes, yellow and black (on low beams and hatch heads, on sills, kerbs and cable protectors), and the radiation
  warning signs of the waste store, at its door and inside it.
- Other signs on the walls carry words or pictograms (door notices, fire and first-aid signs, hazard warnings); none
  of them carries a tag.
- A survey mark has no plate on a wall: it is a painted disc on the floor, lettered with its tag in the same
  lettering (round procedure, section 4).

## 6. The column grid and its marker plates

The buildings stand on a column grid (`grid`): numbered lines across x, lettered lines across y. When the buildings
were surveyed, marker plates were fixed to walls at known points of this grid, at the height the drawing gives for
each. Unlike the tag plates, these carry a machine-readable marker. The drawing lists every plate with its marker
id, its position, its height and the direction it faces (`grid_markers`). Position and height are those of the
plate's centre. The plate is 349.8 mm wide and 145.2 mm high, and its 110 mm marker stands at the left end: the
marker's centre lies 102 mm to the left of the plate's centre, as seen facing the plate (catalogue, section 6). About
three in four of the plates still hang, each exactly where the drawing has it; the others are gone. No plate hangs
anywhere the drawing does not show one: there is none in the annex, and none where a sheet is missing.

## 7. The modification record

Some of the changes of section 3 were entered in the plant's modification record, most were not: about four in ten
over the plant's buildings. The record of a single building can hold several of its changes or none, so a change
that is not in the record may still have been made. An entry gives the year, the kind of change and one room,
nothing more; the room can be one of a missing sheet. Every record also holds a few entries about routine work. No
entry is false. The kinds of entry, and the room each names:

<!-- BEGIN GENERATED modlog -->
| `kind` of an entry | what it records |
|---|---|
| "door sealed" | a door was sealed; the entry names the room behind it |
| "instrument moved" | an instrument was moved to another wall of its room; the entry names that room |
| "instrument relocated" | an instrument was relocated; the entry names the room it was taken from, the room the round sheet records it in |
| "annex added" | the annex was built on; the entry names the room it was built against |
| "process change, not re-surveyed" | the process in the named room was changed after the zoning survey, and its dose rates were not measured again |
| "link repeater out of service" | a link repeater stopped relaying; the entry names the corridor it hangs in |
| "wall opening made" | an opening was made in a wall; the entry names one of the two rooms it joins |
| "temporary shielding erected" | a temporary shielding wall was put up across a corridor; the entry names the corridor |
| "lighting replaced", "floor repainted", "signage renewed", "fire extinguishers replaced", "cable tray added", "pump motor rewound", "breaker replaced", "filters replaced", "handrail repaired", "drain grating renewed" | routine work that changed nothing a robot would notice; the entry names the room it was done in |
<!-- END GENERATED modlog -->
