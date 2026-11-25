# The calibration site

The calibration site is the practice ground's surveyed site (`cal` in `station_api.md`, section 10). The team built
it to try the robot and the camera against things of known size: every dimension of it is on this page, as data in
`site_plan.json` and as a drawing in `site_plan.png`. The page states dimensions only. It does not say what the robot
can pass, cross or reach: that is what the site is for.

All coordinates are in the site frame (x east, y north, z up, origin at the south-west floor corner), in metres. The
floor is level concrete with the friction coefficient of the handbook's measurements, and the light is the same in
every episode. The camera hangs where `mission()` says, like the camera of any site. Its picture shows the floor from
the south wall to about 4 m north of it; the north part of this site is outside the picture.

<!-- BEGIN GENERATED plan -->
Site rectangle 4.4 × 5.8 m; perimeter wall 0.20 m thick and 0.90 m high, standing outside the rectangle; simulated time limit 1400 s.

**The ridge** runs across the whole width from y = 1.60 to y = 2.15 m. From west to east:

| piece | kind | x from (m) | x to (m) | width (m) | clear height (m) | along y |
|---|---|---|---|---|---|---|
| block 1 | rubble block, 0.700 m high | 0.000 | 0.150 | 0.150 | — | solid |
| **opening A** | open gap | 0.150 | 0.407 | 0.257 | open to the sky | through, 0.55 m long |
| block 2 | rubble block, 0.700 m high | 0.407 | 0.547 | 0.140 | — | solid |
| **opening B** | open gap | 0.547 | 1.007 | 0.459 | open to the sky | through, 0.55 m long |
| block 3 | rubble block, 0.700 m high | 1.007 | 1.147 | 0.140 | — | solid |
| **opening C** | open gap | 1.147 | 1.706 | 0.559 | open to the sky | through, 0.55 m long |
| block 4 | rubble block, 0.700 m high | 1.706 | 1.846 | 0.140 | — | solid |
| block 5 | support, 0.387 m high | 1.846 | 2.026 | 0.180 | — | solid |
| **opening D** | covered slot | 2.026 | 2.585 | 0.559 | 0.150 | through, 0.55 m long |
| block 6 | support, 0.387 m high | 2.585 | 2.765 | 0.180 | — | solid |
| **opening E** | covered mouth with a back wall | 2.765 | 3.325 | 0.559 | 0.387 | a back wall 0.30 m inside the south face |
| block 7 | support, 0.387 m high | 3.325 | 3.505 | 0.180 | — | solid |
| **opening F** | covered passage | 3.505 | 4.064 | 0.559 | 0.387 | through, 0.55 m long, into the roofed room |
| block 8 | support, 0.387 m high | 4.064 | 4.244 | 0.180 | — | solid |
| block 9 | rubble block, 0.700 m high | 4.244 | 4.400 | 0.156 | — | solid |

The covers of the covered openings are 0.10 m thick.

**The roofed room**: interior x 2.80 to 4.40 m, y 2.15 to 3.00 m (1.60 × 0.85 m), clear height 0.407 m; the side wall on its west side, the site's east wall on its east side, the ridge on its south side, the solid east of the roofed passage on its north side from x = 3.65 m to the east wall. It is entered through opening F (covered passage) on its south side; a doorway 0.85 m wide at the west end of its north side leads into the roofed passage.

**The roofed passage**: interior x 2.80 to 3.65 m, y 3.00 to 4.40 m (0.85 m wide, 1.40 m long), clear height 0.407 m; the side wall on its west side, a solid on its east side (x 3.65 to 4.40 m, y 3.00 to 4.40 m, up to the roof's underside). It leads from the roofed room north into the roofed chamber. In it a rubble block stands on the floor against the west wall, up to the roof's underside: x 2.800 to 3.091 m, y 3.575 to 3.825 m (0.291 m across the passage, 0.250 m along it); the clear width beside it, on its east side, is 0.559 m.

**The roofed chamber**: interior x 2.80 to 4.40 m, y 4.40 to 5.80 m (1.60 × 1.40 m), clear height 0.407 m; the side wall on its west side, the site's east wall on its east side, the site's back wall on its north side; on its south side the roofed passage opens into it from x = 2.80 to 3.65 m, and the solid closes the rest.

**The side wall**: x 2.62 to 2.80 m, y 2.15 to 5.80 m (0.18 m thick), up to the roof's underside. **One roof**, 0.12 m thick, covers the roofed room, the roofed passage and the roofed chamber: x 2.62 to 4.40 m, y 2.15 to 5.80 m.

**The free-standing wall**: x 0.55 to 2.05 m, y 3.50 to 3.65 m (1.50 m long), 0.60 m high.

**Ground strips** (flush with the floor):

| strip | x from (m) | x to (m) | y from (m) | y to (m) |
|---|---|---|---|---|
| gravel strip | 0.15 | 1.25 | 2.40 | 3.00 |
| dust strip | 3.75 | 4.35 | 0.30 | 1.20 |

**The flat floor**: x 0.70 to 3.70 m, y 0.05 to 1.55 m, free of anything that stands. Painted on it (paint only): lines 12 mm wide along x = 0.7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.7 m and along y = 0.5, 1.0, 1.5 m, and four crosses (arms of 0.10 m, bars 20 mm wide) centred at (1.20, 0.55), (3.20, 0.55), (1.20, 1.30), (3.20, 1.30).

**Pad and search points** (`mission()` gives the same):

| | x (m) | y (m) | where |
|---|---|---|---|
| pad centre | 2.20 | 0.45 | on the flat floor; retrieval radius 0.60 m |
| search point 1 | 0.95 | 1.05 | flat floor |
| search point 2 | 1.75 | 2.70 | north of the ridge, open floor |
| search point 3 | 3.75 | 5.15 | roofed chamber |

**Crates**:

| colour | x from (m) | x to (m) | y from (m) | y to (m) | height (m) |
|---|---|---|---|---|---|
| red | 0.15 | 0.45 | 0.20 | 0.48 | 0.30 |
| blue | 4.12 | 4.40 | 0.00 | 0.30 | 0.36 |

**Heaps** on the open floor (for each heap the bounding rectangle of its outline, where it stands more than 1 mm above the floor, and the height of its highest point, measured to 1 mm):

| heap | kind | x from (m) | x to (m) | y from (m) | y to (m) | highest point (m) | stands |
|---|---|---|---|---|---|---|---|
| heap 1 | rubble heap | 3.791 | 4.120 | 0.001 | 0.294 | 0.272 | against the site's south wall |
| heap 2 | mound of fill | 1.051 | 1.554 | 3.178 | 3.500 | 0.197 | against the free-standing wall's south face |

**Plastic chips** (flat, lying on the floor, in the end caps' colours):

| colour | shape | size (mm) | x (m) | y (m) |
|---|---|---|---|---|
| red | disc | 40 | 1.55 | 0.95 |
| green | square | 50 | 2.85 | 1.15 |
| blue | disc | 55 | 0.95 | 0.40 |
| red | square | 35 | 1.25 | 3.70 |
| green | disc | 45 | 0.40 | 2.25 |
| blue | square | 60 | 2.60 | 0.85 |
<!-- END GENERATED plan -->

## Notes

- The three open gaps A, B, C differ only in width. The covered openings D, E, F have the same width as gap C and
  differ in clear height and in what lies behind their south face.
- Opening E and opening F look alike from the south: the same width, the same clear height, the same cover. E ends at
  its back wall; F leads through into the roofed room.
- The camera does not see into the roofed room, the roofed passage or the roofed chamber. Search point 3 of this site
  lies in the chamber, 0.65 m from the site's east wall and 0.65 m from its north wall.
- In the roofed passage a block stands against the west wall, from the floor to the roof. Beside it the passage is as
  wide as openings C to F. The camera does not see it.
- An episode on this site runs like an episode on any site: the robot is set down at the pad, the station's calls and
  budgets are the same, and reports are taken. Its debrief gives the true positions of the robot's six nodes twice a
  second, under the roofs too, and the true displacement of every primitive that ran (`station_api.md`, section 10);
  it gives no credit.
- The gravel strip and the dust strip are the two kinds of loose ground of the sites.
- North of y = 4.0 m and west of the roofed part (x 0 to 2.62 m, y 4.0 to 5.8 m) the floor is open, level concrete
  with nothing on it; the camera shows only its southern edge.
- The rubble heap and the mound of fill on the open floor are the two kinds of heaps of the sites.
