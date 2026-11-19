# Sources of racecar_cad_design_from_photos

This note says where the task's car and its other material come from. The agent never sees this folder; everything the
agent sees uses generic names and invented part numbers only.

## The car

The car of this task is one model: the benchmark owner's own CAD and scene model of a tenth-scale four-wheel-drive
autonomous race car. The owner supplied the model for the benchmark and has agreed to its public distribution in the
forms this task uses. The model is used as it is: no hidden variant, no redesign, no rebuilt geometry, no stand-in for
any component.

What the task derives from the model:

| Derived material | Where | How |
|---|---|---|
| The true solids of the 23 made parts | `reference/parts/` (and, byte-identical, `../solution/reference/parts/`) | Each solid of the model at the pose it has in the model. Parts that the model holds as meshes only are sewn from the mesh itself, triangle by triangle. |
| The given components (the rolling base assembly and 15 component types) | the payload: `base/`, `purchased/` | The model's own solids, each type written to one STEP file in a documented frame. The compute module and the antennas keep the uniform scale the model gives them. |
| The photographs | the payload: `photos/` | Renders of the model, complete, taken apart stage by stage, and part by part on a cutting mat. |
| The documents (`base/README.md`, datasheets, drawings, `parts_list.md`) | the payload | Written for this task from the model's measured geometry; none is a copy of a maker's document. |

Two treatments are applied to derived copies; the owner's model files themselves are never modified:

- Lettering is removed from five given solids: in the rolling base assembly, a maker's name raised on the battery's
  hold-down bar, a type designation engraved into the speed controller's case and two maker names raised on the
  servo saver; in the compute module, a maker's logo and wordmark engraved into the top plate of its fan sink and a
  maker's name raised on the underside of one connector. Raised letters are cut flush with the face they stand on,
  engraved ones are filled flush with it; the removed or filled volume is the lettering alone, and the rest of each
  solid keeps its geometry. No given file carries a name, and none of this lettering can be read in any photograph.
  Marks that name no maker or product stay as modelled: mould, cavity, revision, material and date marks, moulded part
  numbers, certification, recycling and origin marks, polarity, arrow and pin marks, the numbers on the tyres, tooth
  counts, single key letters and labels such as ON / OFF. The mould numbers and date marks on the floor of the tub are
  flattened in the rendered mesh only.
- The made plate and the printed part exist in the model as meshes whose hole walls are polygons. Each such wall is
  put back to the conic section it approximates (an ellipse or a circle fitted to the polygon, at most 0.02 mm from
  the mesh on the plate and 0.06 mm on the printed part), so that the grader's hole detector reads the holes. Nothing
  else of those parts is changed.

In the photographs only: the two circuit boards, which carry product photographs as textures in the model, are
rendered in a plain colour, and so is the block under one of them.

Screws, nuts, washers, the ball studs with their sockets, the wiring with its plugs and the rear wing frame are in the
model and in the photographs, but in neither the given files nor the reference. The raised print layer on the
underside of the compute module's carrier board, which carries a maker's name and logo, is left out of the given
file, the photographs and the reference alike.

## Invented names

The part numbers in the agent's files (those beginning `GLH-`, `EST-`, `WNL-`, `PRD-`, `RTN-` and `SP-`) and the
components' generic names are invented for the benchmark. No platform, product or maker name appears in any file the agent sees,
in the STEP headers, in the photographs or on the given solids.

## Third-party software

- FreeCAD 1.1.4 (LGPL-2.0-or-later, https://github.com/FreeCAD/FreeCAD) is installed unmodified in the task image
  from its release AppImage, pinned by sha256 in `environment/Dockerfile`.
- The FreeCAD MCP server neka-nat/freecad-mcp at commit `78bf0230a03324b56c9a0b7f55150fb5be2a0d42` (MIT,
  https://github.com/neka-nat/freecad-mcp) is vendored unmodified in `environment/vendor/freecad-mcp/` with its
  `LICENSE`.
- The grader's Python packages are those of the benchmark's shared base image (`environment/requirements.txt`).

## Background of the photographs

The photographs are rendered with Blender Cycles on a bench in the lab room of the project's licensed art assets (the
room bundle `room_lab_v4r3`), as in the sibling CAD-from-photos tasks. The world light is the HDRI `schadowplatz_4k`
and the bench top carries the scanned material `oak_veneer_01`; both are public CC0 material and are also used by the
sibling tasks. None of these assets is shipped with the task: only the rendered photographs are.

## Where the files are

The payload (photographs, `base/`, `purchased/`, `parts_list.md`) is served by the benchmark's asset dataset under
`tasks/ground/racecar_cad_design_from_photos/payload/`, at the commit pinned in `environment/assets.sha256`. The
hidden truth is in this folder and in `../solution/reference/` and is never part of the payload. The public source
files of the model are listed by digest in `public_blocklist.json`; a submission that contains one of them scores 0.
