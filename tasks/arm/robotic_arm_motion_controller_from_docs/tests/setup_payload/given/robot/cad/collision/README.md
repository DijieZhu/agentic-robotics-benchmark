# DA-6 desktop arm - collision envelope

Contact and clearance at acceptance are judged on this envelope (acceptance specification, [../../../acceptance/acceptance_spec.md](../../../acceptance/acceptance_spec.md)), not on the visual CAD shells of [../](../README.md).

## Files

One binary STL per link, `<link>.stl`, units **metres**, in the link's **CAD frame** (the frame of the visual CAD file `../<link>.stl`: parallel to the base frame at the zero pose, origin on the axis of the joint that carries the link; kinematics sheet, section 6). A file holds the link's convex pieces one after the other, each a closed, outward-oriented convex shell with its own triangles; the pieces of one link may overlap. Each piece is a solid; the link's envelope is the union of its pieces.

| file | link | pieces | convex hulls | boxes | triangles | x range [mm] | y range [mm] | z range [mm] |
|---|---|---|---|---|---|---|---|---|
| `base.stl` | base | 33 | 32 | 1 | 10224 | -56.07 .. 56.07 | -66.00 .. 26.41 | -0.61 .. 78.63 |
| `shoulder.stl` | shoulder | 33 | 32 | 1 | 8778 | -28.96 .. 26.68 | -12.20 .. 50.90 | -46.20 .. 71.89 |
| `upper_arm.stl` | upper arm | 17 | 16 | 1 | 3936 | -53.80 .. 13.50 | -45.46 .. 13.20 | -11.93 .. 139.41 |
| `forearm.stl` | forearm | 33 | 32 | 1 | 7386 | -52.40 .. 12.00 | -18.16 .. 19.59 | -12.00 .. 160.39 |
| `wrist.stl` | wrist | 17 | 16 | 1 | 3790 | -53.50 .. 9.00 | -20.00 .. 15.66 | -11.94 .. 73.49 |
| `gripper.stl` | gripper | 17 | 16 | 1 | 3512 | -24.00 .. 28.00 | -30.60 .. 35.20 | -1.00 .. 100.64 |
| `jaw.stl` | jaw | 16 | 16 | 0 | 2646 | -5.10 .. 42.90 | -10.00 .. 12.30 | -9.98 .. 78.22 |

`pieces.json` is the index: for every link its file and its pieces in file order, each with `name`, `kind` (`hull` or `box`), `first_triangle` and `triangles` (the piece's triangles in the file); a box also as a primitive: `centre_m`, `half_m` and `axes` (the box's x, y and z directions in the link frame). Splitting a file by these ranges gives the pieces. Piece names: `P<nn>_<kk>` is convex piece kk of printed part P<nn> (the part numbers of the CAD files), `S<k>` the box around the body of servo k (45.80 x 25.20 x 38.50 mm).

Placing: a vertex p of `<link>.stl` is at T(B, link)(q) * p in the base frame (kinematics sheet, section 6; p in metres), as for the CAD files. The envelope is the same on every unit and moves with the link frames.

## What is checked against what

- **Contact**: any penetration of the envelope of a moving link (shoulder to jaw), or of a clamped tool, into the table or a cell object; or of the envelopes of two links into each other, except adjacent links: base - shoulder, shoulder - upper arm, upper arm - forearm, forearm - wrist, wrist - gripper, gripper - jaw are not checked against each other.
- **Clearance**: the smallest distance between the envelope of the moving links (and of a clamped tool) and the table and the cell objects (the acceptance specification states which supports of a job do not count).
- The base stands on the table: its envelope counts only against the moving links (upper arm to jaw, and a clamped tool), not against the table and the cell objects.
- **Clamped tools** (TCP frame; kinematics sheet, section 5.3; gripper and needle drawings): a gripped part is a box of its half sizes at its seat (its -x face on the fixed pad face, x_TCP = -7.8410 mm); the dispensing needle's holder is a box of 15.68 x 12.00 x 18.00 mm centred at (-0.0010, 0.0000, -3.0000) mm; the needle is a capsule of radius 1.2 mm whose axis runs from (-0.0010, 0.0000, 6.0) to (-0.0010, 0.0000, 43.8) mm, so that its rounded end reaches the tip at z_TCP = 45.0 mm. They move with the gripper; they are not checked against the gripper, the jaw and the wrist, and are checked against every other link, the table and the cell objects. The same values are in `pieces.json` (`clamped_tools`, metres).
- How the envelope differs from the visual CAD: [../README.md](../README.md), section Collision envelope.

