# Sources of freeflyer_cad_design_from_photos

This note says where the task's robot and its hidden standard come from. It lives on the grader's side (`tests/`),
which the agent never sees. Everything the agent sees (the instruction, the photographs, the tags) uses the
benchmark's neutral vocabulary and never names the real robot, its maker or its station. Real names appear only in
this note, in the byte copies of the maker's own files under `reference/maker/`, in their pins
(`reference/maker/pins.json`, `public_blocklist.json`) and where the grader's files quote the makers' text as written
(file names and mesh paths in `reference/placement.json` and `makerstd/placement.py`).

## The robot and the standard

The task's robot, "the cabin free-flyer", is NASA's Astrobee free flyer of the International Space Station, in its
default flight configuration: the perching arm in the top-aft payload bay, the arm stowed. Its makers published the
robot's appearance meshes and its robot description. **The hidden standard is those files, exactly as published.**
The project made no product geometry: no model, no completed internals, no variant, no ground truth of its own.

| Stored under `reference/maker/` | Upstream file | Role |
|---|---|---|
| `meshes/body.dae.xz` | `astrobee_freeflyer/meshes/body.dae` | tag P01 (core module) |
| `meshes/pmc.dae.xz` | `astrobee_freeflyer/meshes/pmc.dae` | tags P02, P03 (propulsion modules) |
| `meshes/pmc_bumper.dae` | `astrobee_freeflyer/meshes/pmc_bumper.dae` | tags P04, P05 (corner-bumper sets) |
| `meshes/base_link.dae` | `astrobee_freeflyer/meshes/base_link.dae` | tag P06 |
| `meshes/arm_proximal_link.dae.xz` | `astrobee_freeflyer/meshes/arm_proximal_link.dae` | tag P07 |
| `meshes/arm_distal_link.dae` | `astrobee_freeflyer/meshes/arm_distal_link.dae` | tag P08 |
| `meshes/gripper_left_proximal_link.dae` | `astrobee_freeflyer/meshes/gripper_left_proximal_link.dae` | tag P09 |
| `meshes/gripper_left_distal_link.dae` | `astrobee_freeflyer/meshes/gripper_left_distal_link.dae` | tag P10 |
| `meshes/gripper_right_proximal_link.dae` | `astrobee_freeflyer/meshes/gripper_right_proximal_link.dae` | tag P11 |
| `meshes/gripper_right_distal_link.dae` | `astrobee_freeflyer/meshes/gripper_right_distal_link.dae` | tag P12 |
| `description/model.urdf.xacro` | `description/description/urdf/model.urdf.xacro` | the body link's visuals and their placements; the bay of the arm |
| `description/macro_perching_arm.urdf.xacro` | `description/description/urdf/macro_perching_arm.urdf.xacro` | the arm's links, joint origins, axes, limits and its start values |
| `context/gazebo_model_plugin_perching_arm.cc` | `simulation/src/gazebo_model_plugin_perching_arm/gazebo_model_plugin_perching_arm.cc` | read as text only: how the description's gripper start value maps to the four finger joints |
| `context/astrobee.launch` | `astrobee/launch/astrobee.launch` | read as text only: the default bay arguments and the description's namespace argument |

- The meshes come from `nasa/astrobee_media` (https://github.com/nasa/astrobee_media) at commit
  `1cb0620121099cc848f5a3db480464188dc91ea3`; the description, plugin and launch files from `nasa/astrobee`
  (https://github.com/nasa/astrobee) at commit `bf43a42d5f89bf0679e51ab25e7a9b7800313e9c`.
- The ten meshes hold 11,753,330 bytes and 87,608 triangles (104,853 once the second propulsion module and the
  second corner-bumper set are placed). Every stored file is pinned by its byte count and the sha256 of the maker's
  own bytes in `reference/maker/pins.json`; the grader's reader (`makerstd/`) checks every pin before it reads a file.
- `reference/placement.json` is derived from the four texts by `makerstd/placement.py` and is not the standard: the
  grader derives it again at every verify and stops with a task error when any item differs.

### Licences, as the makers wrote them

- `nasa/astrobee`: Apache License, Version 2.0 (its `LICENSE`; the description and plugin files say "The Astrobee
  platform is licensed under the Apache License, Version 2.0").
- `nasa/astrobee_media`: two statements, both recorded here as written, without a further judgement of ours. The
  repository's `readme.md`, section "Usage Guidelines": "All multimedia is covered by NASAs Media Usage Guidelines
  (https://www.nasa.gov/multimedia/guidelines/index.html)". The package file `astrobee_freeflyer/package.xml`:
  `<license>Apache License, Version 2.0</license>`.

## Used as given, oddities included

The triangles, node matrices, unit and materials of the meshes and the numbers of the description are used as
written. These oddities of the makers' files are kept, not corrected:

- The arm base's mesh does not touch the core module: about 14 mm lie between them; it touches the two propulsion
  modules.
- The starboard propulsion module, its corner-bumper set and its skin are turned by `rpy="0 3.14 3.14"`, not by an
  exact half turn (the surface moves by up to about 0.57 mm against an exact half turn); the bay of the arm is turned by
  `3.1415`. Both are used as written.
- The corner-bumper file holds four stray pieces of 1, 1, 3 and 6 triangles besides its four bumper shells.
- The core module's node `Astrobee_005` repeats one battery shell in place: its 111 vertices coincide with vertices
  of the top port battery pack of node `Astrobee_006`.
- The arm's proximal start value `1.57079632679` lies 6.3e-6 rad above the joint's upper limit `1.57079` as written.
- Each of the two proximal finger pieces (`gripper_left_proximal_link.dae`, `gripper_right_proximal_link.dae`) holds
  10 triangles of zero area whose three vertex indices differ but two of whose corners lie at the same position.
  They stay in the standard (they have no area, so no score depends on them).

## What the project made (each item, openly)

1. The photographs: renders of the makers' triangles, normals, texture coordinates, materials and textures as given.
2. The bench props and the room: cutting mats, the checker card, the steel rule, the paper tags, support blocks and
   the laboratory room setting.
3. The scoring samples: which points of the makers' surfaces count as seen in the photographs. The cameras decide it;
   every sample lies on a maker triangle (a triangle index and two barycentric coordinates); the samples are frozen
   before the frontier episode.
4. The rules and tolerances of the grading.
5. The reference submission's 0.3 mm skins: around the makers' open surface patches, 0.15 mm to either side along the
   vertex normals, only to show that a document of real solids reaches the full score. The grader never reads the
   reference submission.

## Processing of the makers' files that leaves the standard unchanged

Allowed conversions only: the three large meshes are stored xz-compressed (pinned on the uncompressed bytes); the
reader turns metres into millimetres and applies the meshes' node matrices and the description's placements. Three
further steps touch the makers' triangles; none changes the graded standard:

1. **The photographs drop triangles that repeat a vertex index** before rendering. Per file there are none:
   0 in each of the ten meshes of the standard and 0 in the default skin `pmc_skin_.dae` (any such triangle would have
   an area below 1e-12 mm²). Only the photographs could be affected. The zero-area triangles with distinct indices
   listed above are not of this kind and are not dropped.
2. **The reference submission welds vertices at 1e-3 mm**; a triangle that collapses in the weld does not become a
   face. The counts per file are in `solution/reference/oracle.json`. Only the reference submission is affected.
3. **The teardown photographs rest each part on a stable face** of the convex hull of its surface, taking the area
   centroid of the makers' surface in place of a centre of mass (the files carry no mass or density). Only the pose
   in which a part lies on the bench is affected.

## Real and neutral names

| Real | In the agent's files |
|---|---|
| NASA's Astrobee free flyer | the cabin free-flyer |
| `body.dae` (the central core) | core module (P01) |
| `pmc.dae` (propulsion module, port and starboard) | propulsion module (P02 at the port place, P03 at the starboard place) |
| `pmc_bumper.dae` (four corner bumpers of one propulsion module) | corner-bumper set (P04, P05) |
| `base_link.dae`, `arm_proximal_link.dae`, `arm_distal_link.dae` (the perching arm) | arm piece (P06, P07, P08) |
| `gripper_left_proximal_link.dae`, `gripper_left_distal_link.dae`, `gripper_right_proximal_link.dae`, `gripper_right_distal_link.dae` | finger piece (P09, P10, P11, P12) |
| `pmc_skin_.dae` (the default side skin) | side skin (photographs only, not a tag) |
| top aft payload bay | top-aft bay |
| International Space Station | the station |
| the Gazebo plugin of the perching arm | not named |

## Software in the image

The FreeCAD MCP server in `environment/vendor/freecad-mcp/` is `neka-nat/freecad-mcp` at commit
`78bf0230a03324b56c9a0b7f55150fb5be2a0d42` (MIT; its own `LICENSE` and `VENDORED.md` are kept there). FreeCAD 1.1.4
(LGPL-2.1-or-later) is installed at image build from the upstream release AppImage, pinned by sha256 in
`environment/Dockerfile`.
