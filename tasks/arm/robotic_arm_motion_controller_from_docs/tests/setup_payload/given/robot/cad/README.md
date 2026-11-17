# DA-6 desktop arm - link CAD files

One binary STL per link: the link's printed parts and its servo body merged into one file. Units: **metres**. Each file is in its **link CAD frame**: parallel to the base frame at the zero pose, with the origin on the axis of the joint that carries the link (kinematics sheet, section 6, for the origins at the zero pose and the chain formula that places a link at any configuration). The parts are closed, outward-oriented shells; the parts of one link are not merged into a single solid and may touch or overlap where they are assembled. The files show the nominal outer shape of the parts (no fasteners, no cables). Contact and clearance in the cells are judged on the arm's collision envelope, published in [collision/](collision/README.md), not on these shapes (section below).

| file | link | triangles | x range [mm] | y range [mm] | z range [mm] | parts |
|---|---|---|---|---|---|---|
| `base.stl` | base | 37582 | -55.46 .. 55.46 | -66.00 .. 25.80 | 0.00 .. 78.02 | printed part P01, printed part P02, servo body |
| `shoulder.stl` | shoulder | 30010 | -28.70 .. 26.38 | -12.20 .. 50.60 | -46.20 .. 71.69 | printed part P03, printed part P04, servo body |
| `upper_arm.stl` | upper arm | 26472 | -53.80 .. 13.50 | -45.45 .. 12.21 | -11.99 .. 138.42 | printed part P05, servo body |
| `forearm.stl` | forearm | 34576 | -52.40 .. 12.00 | -18.16 .. 19.59 | -11.99 .. 160.08 | printed part P06, printed part P07, servo body |
| `wrist.stl` | wrist | 36800 | -53.30 .. 9.00 | -20.00 .. 15.66 | -12.00 .. 73.49 | printed part P08, servo body |
| `gripper.stl` | gripper | 41444 | -24.00 .. 28.00 | -30.40 .. 35.20 | -1.00 .. 100.64 | printed part P09, servo body |
| `jaw.stl` | jaw | 30060 | -5.10 .. 42.90 | -10.00 .. 12.30 | -10.00 .. 78.22 | printed part P10 |

Use with the link transforms of the kinematics sheet: a vertex p of `<link>.stl` is at T(B, link)(q) * p in the base frame (p in metres; the sheet's lengths are in mm).

## Collision envelope

The collision envelope, on which contact and clearance at acceptance are judged (acceptance specification, [../../acceptance/acceptance_spec.md](../../acceptance/acceptance_spec.md)), is published in [collision/](collision/README.md): per link a set of convex pieces in the same link CAD frames, metres (`collision/<link>.stl`, index `collision/pieces.json`). It approximates each link: the printed parts by a convex decomposition, each servo body by a box of 45.80 x 25.20 x 38.50 mm around it. It is the same on every unit and moves with the link frames. It reaches beyond the CAD surface where its convex pieces bridge the recesses, holes and gaps of the parts and at the corners of the servo boxes (the servo bodies are rounded there); the printed parts' surfaces reach beyond it by a few tenths of a millimetre at most; each servo body lies inside its box, 0.2 mm inside each of the box's faces at its extremes. Largest distances, to 0.1 mm:

| link | convex solids | envelope beyond the CAD surface [mm] | printed parts beyond the envelope [mm] | servo body beyond the envelope [mm] |
|---|---|---|---|---|
| base | 33 | 8.9 | 0.1 | 0.0 |
| shoulder | 33 | 5.7 | 0.6 | 0.0 |
| upper arm | 17 | 5.7 | 0.2 | 0.0 |
| forearm | 33 | 5.7 | 0.4 | 0.0 |
| wrist | 17 | 5.1 | 0.3 | 0.0 |
| gripper | 17 | 5.7 | 0.4 | 0.0 |
| jaw | 16 | 2.9 | 0.3 | - |

The base stands on the table: its envelope counts only against the moving links (contact of the arm with itself), not against the table and the cell objects.

