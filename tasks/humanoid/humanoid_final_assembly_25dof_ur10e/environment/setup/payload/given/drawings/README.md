# The dimension drawings

Seven sheets, `drawing_1.png` to `drawing_7.png`. Each is a landscape sheet of 4200
by 2970 pixels at 10 pixels per millimetre of paper. The lettering is about
1.6 mm high on paper, about 16 pixels: read a sheet from crops at full size. A whole sheet shown reduced does not
show its smallest text.

The values on the sheets are exact. They are given only as pictures: no file lists them.

| sheet | title | what it is for |
|---|---|---|
| `drawing_1.png` | Hip interface | the hip output flange and the leg's bracket: pilot, faces, hole patterns and their angles, on both sides; where the flanges lie on the base |
| `drawing_2.png` | Shoulder socket | the shoulder socket and the arm's ring: diameters, depths, hole patterns and their angles, on both sides; where the sockets lie on the base; the arm as delivered |
| `drawing_3.png` | Neck interface | the neck output and the head's bracket: boss, bore, lip, hole patterns and their angles; where the neck seat lies on the base |
| `drawing_4.png` | Stand and clamping | the base on its stand and the stand in the cell at the nominal layout; the stand's datum board |
| `drawing_5.png` | Racks and nests | the cell from above at the nominal layout: every datum board, and each module's seat centre and attitude on its fixture relative to the fixture's board |
| `drawing_6.png` | Finished product, general arrangement | the finished robot from the front and from the right, and the table of the delivered joint angles |
| `drawing_7.png` | Shoulder socket: detail G of section E-E | the corner of the shoulder socket's section where ring, collar, bore and housing meet, enlarged |

How the sheets are drawn. Every view stands on its own: its title names the side it is seen from and its scale, and a
triad in the view shows the axes. Lengths are in millimetres, angles in degrees. The datums A, B and C are defined on
the sheets that use them, and a datum symbol says whether it stands on a face or on an axis. A signed value is the
coordinate of the named point less that of the origin, along the positive direction of the triad; each sheet with
angles states which way a positive angle turns in each of its views. Sheets 4 and 5 show
the nominal layout of the cell; how far each fixture is placed away from it per episode is in the cell manual,
section 10. Sheet 4 defines a datum board's corner and axes. Sheet 5 gives
each module's place and attitude on its fixture relative to the fixture's board, defines the order and the direction
of the turns of a module's axes, and carries a worked example. Sheet 6 carries the table of the joint
angles the modules are delivered with, in the zero and sign of each joint in its module's file.

The legends, in short. A thin chain line is a centre line, an axis or a named reference line; a chain circle is a
pitch circle; a wavy line is a break. In a section, line hatching is a part whose model is a solid, cut by the plane,
with touching parts hatched at different angles; cross-hatching is a bought unit whose model is a surface only, which
is not cut and whose inside is not shown, with white inside it where there is air on the plane. Heavy short lines in
a section are mating faces drawn from the parts' face data at their true size; a dashed short line is such a face
behind the plane. Threaded holes are drawn at the diameter of their model. The interface screws of the
five mates are not drawn on any sheet.

One-sided diameter lines. Sheets 3 and 7 dimension some diameters with a line that runs
square to the axis from a single face of the feature towards the axis, with its only arrowhead on that face; the value
beside it is the whole diameter. On sheet 3 such a line runs a little past the axis, and the arrowhead of
the bracket bore's line stands on the outer side of its face, in the bracket's material, and points at the face. On
sheet 7 the axis lies below the detail.
