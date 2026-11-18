# Venues of the league: what to expect

The league races in halls it rents for a few days. Each venue is built on the day, and no drawing of its track is
published: what a team knows before its car first drives there is the venue sheet (rules, clause S-02) and this
guide. The guide collects what the venues of past seasons had in common. It gives kinds and ranges; where
something stands at a particular venue, how many there are of it and what the floor is like on the day are things
the car has to find out on the track. Photographs of earlier venues are in `photos/` (section 8); none of them
shows a venue of the coming round.

## 1. The venue sheet

The sheet a program receives at a venue (`describe()` in `../interface/car_api.md`) has these fields.

| field | meaning | how far it can be trusted |
|---|---|---|
| `venue_label` | a name for the venue | a label, nothing more |
| `floor.kind` | `painted_concrete`, `sealed_concrete`, `polished_concrete` or `exhibition_carpet` (section 2) | the kind of the hall floor, as it is; it says nothing about local patches (section 3) |
| `floor.side_pull` | the organiser's side-pull reading, `peak_force_n` and `test_mass_kg`; `null` if the organiser took none (section 2) | a hint: another car, other tyres, one spot |
| `walls.kinds` | the kinds of wall that may stand at the venue: `orange_ribbed`, `yellow_ribbed`, `black_plain` (section 4) | every kind in use is listed; a listed kind need not be in use |
| `walls.height_m` | 0.33 | exact |
| `track.min_width_m` | no part of the track is narrower than this | a true lower bound, given in steps of a tenth of a metre: the narrowest place is at most 0.1 m wider |
| `track.max_footprint_m` | the track with its walls fits in a rectangle of this size, [long side, short side] | a true upper bound in whole metres: at past venues each side of the rectangle was 0.2 m to 1.2 m longer than the track needed |
| `features` | the kinds of feature present at the venue, from the list below | exact as to kinds: a listed kind is present at least once, a kind that is not listed is absent; no places and no counts |

| feature | meaning |
|---|---|
| `duct_gaps` | some consecutive ducts do not meet: there are gaps in the walls (section 4) |
| `open_wall` | there is at least one open wall (section 4; rules, clause S-04) |
| `floor_patch_dust`, `floor_patch_metal_cover`, `floor_patch_carpet_tile` | the floor has local patches of that kind (section 3) |
| `black_ducts` | some ducts are plain black (section 4) |
| `narrow_section` | somewhere the track is narrower than 1.2 m |
| `shared_wall` | two sections of the track share one row of ducts (section 4) |
| `outside_clutter` | things stand beside the track (section 6) |

## 2. Floors and grip

The floor of a venue is the floor of the hall as the league finds it: flat, of one kind throughout.

| `floor.kind` | what it is | to the scanning sensor |
|---|---|---|
| `painted_concrete` | concrete under a coat of floor paint | hard and shiny |
| `sealed_concrete` | concrete with a clear sealer | hard and shiny |
| `polished_concrete` | ground and polished concrete | hard and shiny |
| `exhibition_carpet` | thin exhibition carpet laid over the hall floor | matt |

**Grip.** In the league's papers the grip of a floor is a number for one car on one floor: the deceleration of
that car under full braking on that floor, divided by 9.81 m/s^2. On the floors of past venues the grip of this
car's tyres lay between 0.45 and 0.95. It was lowest on painted and on sealed concrete, in between on polished
concrete and highest on exhibition carpet; floors of the same kind differed from hall to hall. Every one of them
gripped less than the floor of the team's lab, where the same tyres reach 1.00 (`../lab_logs/README.md`). These
numbers are for the floor of the hall itself: on a local patch (section 3) the grip can lie below the lowest of
them.

**Side-pull reading.** Organisers like to give a number for their floor. They pull their own reference car
sideways over the floor at one spot and note the largest force before it slides: `peak_force_n`, with the mass of
that car, `test_mass_kg`. The friction this implies is force / (mass x 9.81). It is a hint and no more: it was
taken with another car on other tyres, at one spot, and it is a peak. It is not the grip of this car on the track.
At past venues this car's grip on the floor of the hall was about 0.85 to 1.15 times the friction the reading
implies.

## 3. Local patches

At every venue so far the floor was not the same everywhere. In a few places it differed from the rest. At least
one such place lay in a bend, on the line the cars drive; the others lay anywhere on the track.

| kind | what it is | grip of this car's tyres | what else a car notices |
|---|---|---|---|
| dust | fine dust or grit where the hall was not swept | less than on the floor around it | nothing: it has no edge |
| metal cover | a steel cover plate of a service duct, flush with the floor, running across the track | less than on the floor around it | nothing: it lies flush, without an edge |
| carpet tile | a piece of carpet laid on the floor | more than on a hard floor around it, less than on exhibition carpet around it | an edge of up to 7 mm, which the car feels when it drives over it |

A patch lies in the floor or on it, below the scan plane of the car's sensor: with the car level, no beam meets
the floor and the sensor shows nothing of a patch. When the scan plane tilts and beams meet the floor, the floor
rule of `../car/datasheet.md` (section 5.1) holds for the surface they meet: a carpet tile answers like carpet,
dust and a metal cover like a hard floor. Photographs: `photo_09_carpet_tile.jpg`, `photo_10_metal_cover.jpg`.

## 4. Walls

- **Ducts.** Every wall is a row of flexible air ducts lying on the floor: 0.33 m in diameter, up to 7.62 m long
  each. A duct is light. It is not fastened to the floor or to its neighbour, and it bends where it is pushed
  (`photo_02_wall_low.jpg`).
- **Kinds.** `orange_ribbed` and `yellow_ribbed` ducts are bright, orange or yellow, with black ribs; the scanning
  sensor measures them over its whole range. `black_plain` ducts are plain black; the sensor measures them only up
  to about 4 m (`../car/datasheet.md`; `photo_06_black_ducts.jpg`).
- **Gaps.** Consecutive ducts of a wall are laid end to end. Where they do not meet there is a gap of up to 0.15 m.
  Through a gap the sensor sees what lies behind the wall (`photo_03_duct_gap.jpg`).
- **Open walls.** An open wall is an opening of 0.6 m to 1.2 m in a wall, always narrower than the track at that
  place. Its boundary is the tape line of clause S-04. Behind an opening lies open floor, never another part of the
  track: the floor of the hall behind an outer wall, the infield behind an inner wall. Things may stand on that
  floor (section 6; `photo_04_open_wall.jpg`).
- **Shared walls.** Where two sections of the track pass each other at less than 0.5 m, they are divided by a
  single row of ducts: a wall with track on both of its sides. At past venues such a row had a free end, and the
  track turned round that end from one side of the row to the other (`photo_07_shared_wall.jpg`).

## 5. The track

| quantity | past venues |
|---|---|
| shape | one closed loop, driven in one direction, clockwise at some venues and counter-clockwise at others; no crossings, no forks, no ramps |
| length of the centreline | 60 m to 110 m |
| width | 1.0 m to 2.2 m; most of the length is at least 1.5 m wide |
| tightest bend | the radius of the centreline is nowhere below 0.9 m |
| footprint | the track with its walls fitted in 30 m x 20 m at every venue; the venue sheet gives the bound of the day |

Photographs: a whole track in `photo_01_hall_overview.jpg`, a start position with its timing line in
`photo_05_start.jpg`, a narrow section in `photo_08_narrow_section.jpg`.

## 6. Beside the track

Halls are not empty. Pillars of the hall, legs of tables, crates, roll-up banners and things of that size stand
outside the walls, and also in the infield (the floor the loop encloses: teams put their tables there), never on
the track and never closer than 0.5 m to a wall. They are no part of the track, but the scanning sensor sees them
through gaps and open walls. Some of them are dark and are measured only at short range, like black ducts
(`photo_01_hall_overview.jpg`, `photo_03_duct_gap.jpg`).

## 7. The avoidance test

For the avoidance test the organiser puts cardboard boxes on the track and later sends slow cars round it (rules,
clauses S-22 to S-24). Where the boxes stand, how many slow cars there are and how they drive differs from venue to
venue (`photo_11_boxes.jpg`, `photo_12_slow_car.jpg`).

## 8. Photographs

The files of `photos/` show venues of earlier rounds, all of them on hard floors. They are there to show what the
things named in this guide look like; no place, count or distance in them says anything about another venue.

| file | what it shows |
|---|---|
| `photo_01_hall_overview.jpg` | a whole track from high up: the closed loop of orange and yellow ducts on the hall floor, tables, crates and pillars beside it and in the infield |
| `photo_02_wall_low.jpg` | a wall seen along its length from just above the floor: ribbed ducts laid end to end, the shiny floor, the car beside the wall for scale |
| `photo_03_duct_gap.jpg` | a gap between the ends of two ducts, seen from the track at about the height of the car's sensor, with what stands behind the wall showing through |
| `photo_04_open_wall.jpg` | an open wall: the two duct ends, the tape line on the floor between their track-side edges, open floor behind it |
| `photo_05_start.jpg` | the car on the start position with the timing line across the track ahead of it |
| `photo_06_black_ducts.jpg` | plain black ducts in a wall, next to orange ribbed ones |
| `photo_07_shared_wall.jpg` | a shared wall from above: one row of ducts between two sections of the track, its free end and the turn round it |
| `photo_08_narrow_section.jpg` | a narrow section of the track, with the car in it for scale |
| `photo_09_carpet_tile.jpg` | a carpet tile on a hard floor, lying across the track, with its edge |
| `photo_10_metal_cover.jpg` | a steel cover plate running across the track, flush with the floor |
| `photo_11_boxes.jpg` | the avoidance test: a cardboard box standing on the track, the car passing it |
| `photo_12_slow_car.jpg` | a slow car seen from behind, the marker box standing upright at its rear |
