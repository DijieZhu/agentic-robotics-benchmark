# Fabrication data of the main board

`fab_data.json` describes the bare main board the way its fabrication outputs do: the outline of the board, where each
part is placed, every pad of every placement, and which pads the copper of the board joins. It was compiled from the
fabrication outputs of the board. It carries no names from the schematic: no part names or values, no package names,
no pin names, no net names, and none of the text printed on the board. The main boards of the struts are alike.

## Units and coordinates

- Lengths are in millimetres (`units`); angles are in degrees.
- Positions are board coordinates in a top view of the board: the origin is the lower left corner of the outline, x
  runs to the right and y upward. Rotations are counter-clockwise in that view (`coordinates`).

## `outline`

`loops` is a list of closed polygons, each a list of `[x, y]` corners. The first loop is the edge of the board; every
further loop is a cut-out through the board.

## `placements`

One entry for each place where a part is mounted on the board (its land pattern with its pads).

| key | meaning |
|---|---|
| `id` | the designator of the placement, `X<nn>`. Placements are numbered in the order of the x coordinate, then the y coordinate, of their origin; the number says nothing about what is placed there |
| `x`, `y` | origin of the placement (the reference point of its land pattern) |
| `rotation_deg` | rotation of the land pattern about its origin |
| `side` | the side of the board on which the part sits: `top` or `bottom` |
| `pads` | the pads of the placement, in the order of their numbers |
| `holes` | holes without a pad that belong to the placement: `x`, `y`, and `drill` = the diameter of the hole |

A pad:

| key | meaning |
|---|---|
| `pad` | the number of the pad within its placement, as its land pattern numbers the pads. A pad of the board is written `X<nn>.<pad>` |
| `x`, `y` | centre of the pad |
| `kind` | `through_hole`: a hole through the board with copper around it; `surface_mount`: copper on one side of the board only |
| `drill`, `diameter`, `shape` | through-hole pad: the diameter of the hole, the diameter of the copper around it, the shape of that copper |
| `size_x`, `size_y`, `copper` | surface-mount pad: the size of its copper along the x and the y axis of the board, and the side of the board the copper is on |

## `connectivity`

One entry for each net, that is, each set of pads that the copper of the board joins: `net` is its name, `N<nnn>`, and
`pads` lists its pads. Nets are numbered in the order of their lowest pad (by placement, then by pad number); the name
carries no meaning. A pad that is listed under no net is joined to no other pad.

The file does not say what is mounted at a placement or what a net carries.
