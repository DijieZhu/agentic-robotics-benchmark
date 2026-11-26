# Twin contract

This is the interface between your twin and the grader. Every number here is the one the grader's code reads, apart
from the one measured time quoted beside rule Z7; the parameters are tabled at the end. `tools/check_twin.py` runs the
same code for the gates and the probe.

## Files

The submission is the folder `/home/user/submission/`. It holds one flight folder, `flight_a/`, with:

- `scene.xml`: an MJCF scene of static collision geometry, UTF-8, and the mesh files it names (binary STL or OBJ). The
  grader reads a mesh file at its `file` path relative to the flight folder or, when `<compiler meshdir>` is set, at
  the `meshdir` path joined with the `file` path, relative to the flight folder (empty and `.` components are dropped
  in the join; the name rules for both paths are in the section on the allowed MJCF). The file must be a regular file
  reached without a symbolic link and must lie inside the flight folder. Files inside a folder named `__pycache__` are
  not copied for grading: a mesh file there is missing;
- `cameras.json`:

```
{"format": "flight-twin-cameras/1",
 "frames": {"f_NNNN": {"position": [x, y, z], "quaternion_wxyz": [w, x, y, z]}, ...}}
```

  one entry for every given frame of the flight, named as in `flights/flight_a/frames.csv` (1068 entries), and no
  other entry; no frame and no key of the file written twice. This is rule L: a file that lacks a given frame, holds
  an entry for a name that is no given frame of the flight, or writes a frame or a key twice fails rule Z2.

## The scored moments

The grader scores 24 moments of the flight and does not say which. They are given frames of the flight (the grader's
key frames), and no given file names them. `cameras.json` declares the wide camera's pose at every given frame of the
flight (1068 poses) because the grader reads the scored moments' poses from it and does not say which they are. At a
scored moment the grader reads your entry as the declared pose and applies rules Z3 to Z6 and the scoring below to it.
Every other entry is checked for form only, by the same per-entry rules as rule Z4 (an object with a `position` and a
`quaternion_wxyz`): an entry without a valid pose at a moment that is not scored changes no score, and the same entry
at a scored moment is rule Z4.

## Frames, units and poses

- Lengths are metres. The twin's frame is yours to choose: the grader neither aligns nor scales it.
- The wide camera frame: origin at the camera centre, +z along the optical axis, +x towards the image's right, +y
  towards the image's bottom. A declared pose is the wide camera's pose in your twin's frame: `position` is the camera
  centre; `quaternion_wxyz` (order w, x, y, z) turns camera-frame vectors into twin-frame vectors. It is normalised
  before use; a quaternion and its negative are the same rotation.
- The sensor files write quaternions in the order x, y, z, w and say so in their keys (`quaternion_xyzw`).
- The depth camera frame is the frame of the ray table `sensors/depth_rays.npy` (rows x columns x unit ray, sha256
  `1d8d9dbd3531e66cbc6e57bbe0eaa0dfedbf715df0e6e2b2f355ae50d22649c2`; pixels without a ray are NaN).
  `sensors/wide_to_depth.json` gives R and t with X_depth = R X_wide + t. The grader places the depth camera of a key
  frame at T_twin_depth = T_twin_wide x inverse([R | t]), with exactly this transform.
- The on-board estimate (`onboard_estimate.csv`) gives the body's position and attitude in the robot's map frame
  (`qx`, `qy`, `qz`, `qw`: the attitude quaternion, read as the rotation that turns body vectors into map-frame
  vectors); the wide camera's pose by the estimate is that body pose times the camera mounting of
  `sensors/camera_sheet.json` (camera centre in the body frame, and the rotation that turns camera vectors into body
  vectors).

## Time

`t_s` in `frames.csv` is a frame's recorded stamp, not the instant of its exposure. The only measured relation the
task states: a wide frame's exposure lies 0.067 s before its recorded stamp, measured against the depth camera's
stamps. Wherever the grader matches the on-board estimate to a frame, it reads the estimate at the frame's recorded
stamp minus 0.067 s: position linearly and attitude along the shortest arc between the two samples around that instant
(outside the estimate's recorded span, the nearest end sample).

The given files' columns, read by name after the lines that start with `#` (comment lines that state each file's time
base): `frames.csv` `frame`, `t_s`; `onboard_estimate.csv` `t_s`, `x_m`, `y_m`, `z_m`, `qx`, `qy`, `qz`, `qw`,
`confidence`, `map_feature_count`, `optical_flow_feature_count`, `cov_diag_1`, `cov_diag_2`, `cov_diag_3`,
`cov_diag_4`, `cov_diag_5`, `cov_diag_6`, `cov_diag_7`, `cov_diag_8`, `cov_diag_9`, `cov_diag_10`, `cov_diag_11`,
`cov_diag_12`, `cov_diag_13`, `cov_diag_14`, `cov_diag_15` (the grader reads `t_s` to `qw`; the others are the
estimator's health fields as recorded); `imu.csv` `t_s`, `wx_rad_s`, `wy_rad_s`, `wz_rad_s`, `ax_m_s2`, `ay_m_s2`,
`az_m_s2`.

## How the grader probes your twin

- **The collision geometry.** Every geom in `scene.xml` is collision geometry, whatever its contype, conaffinity,
  group, colour or material: before any ray is cast, every geom is rebuilt opaque, in group zero and without material,
  and the gate's report says so. A mesh is cast as the convex hull of its vertices, which is the shape the simulator
  collides with: a concave mesh is a solid block.
- **A scored moment's depth frame.** From the depth camera's centre, along each ray of the table turned into your
  twin's frame, the grader takes the distance to the first surface of your twin (primitives as they are, meshes as
  their hulls), with the simulator's ray call and a cutoff of 1000 m. A ray that meets nothing gives no range.
- **The wide camera's model** (used by the check tool's overlay; the grader casts the depth camera's rays only): x_u =
  X / Z, y_u = Y / Z, r_u = sqrt(x_u^2 + y_u^2), r_d = atan(2 r_u tan(w / 2)) / w, u = fx x_u r_d / r_u + cx, v = fy
  y_u r_d / r_u + cy, with the values of `sensors/camera_sheet.json`; pixel coordinates have integer values at pixel
  centres, the centre of the top-left pixel being the origin.

## The hazard check

The check runs on one depth frame (ranges along the table's rays, in the depth camera frame):

1. points: the pixels with a range r, 0.2 m <= r < 4 m; point = r x ray;
2. occupied cells: cubes of 0.08 m edge, a point's cell being floor(point / 0.08 m) on each axis, the grid's origin at
   the depth camera's centre;
3. inflation: every occupied cell plus every integer cell offset o with |o| x 0.08 m <= 0.3 m (251 offsets; the radius
   is the robot's 0.28 m plus 0.02 m);
4. a straight move: the robot's centre starts at (0.03462969950432764, -0.08609609681043709, -0.11139868174201827) m
   (the body origin in the depth camera frame) and goes along a unit direction u, sampled every 0.02 m from the start;
   a sample collides when it, or one of its 26 neighbours on a grid of 0.02 m around it, falls in an inflated cell;
5. a move of length L is clear when no sample up to L (inclusive) collides; otherwise it is blocked, and its stop
   distance is that of its first colliding sample.

The move family: u = (sin az cos el, sin el, cos az cos el) in the depth camera frame, for the elevations -12, -6, 0,
6, 12 degrees (outer loop) and the azimuths -20, -15, -10, -5, 0, 5, 10, 15, 20 degrees (inner loop), 45 directions; a
clear move has one of the lengths 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 3.5 m, a blocked one is checked to 3.5 m. Which moves
of the family are scored is not published.

The grader runs the check on your twin's depth frame of a key frame at the pixels where the real depth frame has a
return and nothing is masked as moving or as another free-flyer; the real outcome of each scored move comes from the
same check on the returns of the real depth frame, leaving out weak returns and returns that more depth frames of the
flight see through than see. Which returns are left out is not published.

## Allowed MJCF

Elements and their attributes (anything else fails rule Z2):

- `<mujoco>` inside the root: model
- `<compiler>` inside `<mujoco>`: angle, meshdir
- `<asset>` inside `<mujoco>`: no attributes
- `<mesh>` inside `<asset>`: file, inertia, name, scale
- `<material>` inside `<asset>`: name, rgba
- `<default>` inside `<mujoco>` or `<default>`: class
- `<worldbody>` inside `<mujoco>`: no attributes
- `<body>` inside `<worldbody>` or `<body>`: axisangle, childclass, euler, name, pos, quat, xyaxes, zaxis
- `<geom>` inside `<worldbody>` or `<body>` or `<default>`: name and class (not inside `<default>`) and the geom
  attributes listed below

Geom attributes: `axisangle`, `conaffinity`, `condim`, `contype`, `density`, `euler`, `fitscale`, `friction`,
`fromto`, `gap`, `group`, `margin`, `mass`, `material`, `mesh`, `pos`, `priority`, `quat`, `rgba`, `shellinertia`,
`size`, `solimp`, `solmix`, `solref`, `type`, `xyaxes`, `zaxis`. Geom types: `box`, `sphere`, `capsule`, `cylinder`,
`ellipsoid`, `mesh`; refused: `hfield`, `plane`, `sdf` (`planes are not allowed (use a thin box for a wall or a
deck)`). Refused elements, among others: `<actuator>`, `<attach>`, `<camera>`, `<composite>`, `<contact>`, `<custom>`,
`<deformable>`, `<equality>`, `<extension>`, `<flexcomp>`, `<frame>`, `<freejoint>`, `<hfield>`, `<include>`,
`<inertial>`, `<joint>`, `<keyframe>`, `<light>`, `<option>`, `<plugin>`, `<replicate>`, `<sensor>`, `<site>`,
`<size>`, `<statistic>`, `<tendon>`, `<visual>`.

- The text of `scene.xml` must not hold the letters `xmlns` anywhere, in any letter case: not as an XML namespace and
  not inside a name, a value or a comment. The gate refuses such a scene with `XML namespaces (xmlns) are not
  allowed`.
- `scene.xml` holds no NUL byte; an encoding declaration, if there is one, names UTF-8 or ASCII; the text holds
  neither `<!DOCTYPE` nor `<!ENTITY` in any letter case, and no `<?` apart from the XML declaration at its start (no
  processing instruction). An element holds no text, only attributes and child elements, and the scene holds at most
  one `<compiler>`.
- The elements of `scene.xml` nest at most 400 levels deep, the root element (`<mujoco>`) being level 1 and each
  element one level below its parent; a scene nested deeper fails rule Z2.
- `<compiler angle>` is `degree` (the simulator's default when the attribute is absent) or `radian`; euler and
  axis-angle values are read in that unit. `<mesh inertia>` is one of `convex`, `exact`, `legacy`, `shell`. A `<geom>`
  inside a `<default>` takes the geom attributes only, without `name` or `class`.
- Every `<mesh>` names a `file`, and the file ends in `.stl` or `.obj` (in any letter case). A `file` path and a
  `meshdir` path: relative (not starting with `/` or `~`), not empty, at most 240 characters, only the letters A to Z
  and a to z, the digits 0 to 9, space and _ . + - ( ) / (so no backslash, no `:` and no control character), with no
  empty component and no `..` component; `meshdir` may end in one `/`.
- A mesh file is malformed when a binary STL file is shorter than its header or its size does not match the triangle
  count in its header (an ASCII STL file is not read), when it holds no triangle, when an OBJ file holds no face or no
  vertex, or when an OBJ file's `mtllib` names anything but a plain file name (no `/`, backslash or `:`, not `.` or
  `..`, no leading `~`).
- The compiled scene holds only finite numbers.
- A material that carries only `name`, `rgba` is accepted and ignored. A texture is refused with the gate's message
  `textures are not allowed: a material may carry an rgba colour only, and it is ignored (every geom is cast opaque
  and without material)`.
- A mesh whose vertices all lie in one plane cannot be compiled: `mesh <name>: all its vertices lie in one plane, so
  it has no convex hull and the scene cannot be compiled (give a surface patch a thickness, or use a thin box)`. An
  STL file with more triangles than the simulator reads: `mesh file <file>: <n> triangles in one STL file; the
  simulator reads at most 200000 triangles from an STL file, so the scene cannot be compiled (write a larger mesh as
  an OBJ file)`.
- A compile warning of the simulator is reported in the gate's lines and does not fail the gate; a compile error fails
  it.
- Every geom's extent must lie within 100 m of the origin on each axis.

Caps (each one MB is 1048576 bytes): per flight at most 2000 geoms, 2000 bodies, 8000 mesh assets, 20000 XML elements
in `scene.xml`, nested at most 400 levels deep (the root element being level 1), 500000 mesh triangles (each mesh
asset's triangles counted once for every geom that names the asset, by its own `mesh` attribute or through a default
class and whatever the geom's type, and once if no geom names it; summed over the assets, so a file named by two
assets counts for each of them) and 26 MB of mesh files, at most 200000 triangles in one STL file; in the whole
submission at most 5000 files and folders and 64 MB.

## Rules that set a score to zero

The whole submission:

- **Z1** (the integrity gate G1): the twin score is 0 when an entry of the submission was written after the session
  ended; when the submission folder itself is a symbolic link, is not a folder, lies outside `/home/user`, or was
  replaced or changed after the session ended; when it holds more files and folders than 5000 or more than 64 MB; or
  when it holds a symbolic link or a special file anywhere (an entry that cannot be inspected counts as one). A
  submission folder that does not exist is not rule Z1: each flight then has no `scene.xml` and fails rule Z2.

The flight:

- **Z2** (the validity gate, `flight-twin-gate/1`): the flight scores 0 when `scene.xml` is missing, cannot be read,
  is not a regular file at the flight folder's root, is not valid XML, holds the letters `xmlns`, nests its elements
  more than 400 levels deep, breaks a rule of the section on the allowed MJCF (the allow list, the text rules, the
  name rules), does not compile or is over a cap; when a mesh file is missing, cannot be read or is malformed; when
  `cameras.json` is missing, cannot be read, is not a regular file at the flight folder's root, is not JSON of the
  format above, or lacks its `frames` object; when `cameras.json` declares no entry for a given frame of the flight,
  holds an entry for a name that is no given frame of the flight, names a frame more than once or writes one of its
  own keys more than once (rule L); or when the simulator refuses to rebuild the compiled geometry for the rays.
- **Z3** (one place): the one-flight fit below finds fewer than 12 inliers among the scored moments with a valid pose:
  the flight scores 0.
- **Z7** (the grader's limits on the ray scene): the flight scores 0 when the grader cannot build your twin's ray
  scene inside 120 s of its wall clock, cannot cast the flight's rays into it inside 300 s of its wall clock, or
  cannot do either inside an address space of 6144 MB. The ray scene is built once for the flight, from the compiled
  twin after the validity gate; the casting takes the key frames in order, applies rules Z4, Z6 and Z5 to each and
  casts the depth camera's rays into your twin. Both run in one process of the grader, whose address space, the
  grader's own code and data included, is held to that limit. The grader's and the tool's messages call this limit the
  worker's address space. When rule Z7 stops the building and rule Z3 would apply as well, the grader reports rule Z7.
  No other time or memory limit of the grader sets a score to zero. On the grader's machine, the whole verification of
  the maintainers' own twin of this flight took about 44 s, the building of its ray scene and its casting included.
  `tools/check_twin.py` builds the ray scene and casts in a process of its own under the same address-space limit and
  reports rule Z7 when either does not fit inside it. It decides nothing by time: it prints how long its own build
  took beside the build's limit, and how long its own cast of one frame's rays took and what the flight's scored
  moments would take at that rate beside the casting's limit. That estimate rests on that one frame alone (the first
  frame of the tool's spread with a valid pose), and frames differ in what they see. The session's machine is not the
  grader's, and the tool's own code and data are not the grader's, so the times and the room left for your twin
  differ.

A scored moment (it scores 0 on every scored pixel and every move of that moment, clear moves included, so that rule P
counts it as lost when it has a scored move), checked in this order:

- **Z4**: no valid pose is declared at the scored moment (an entry that is not an object; a `position` that is not a
  list of exactly three numbers or a `quaternion_wxyz` that is not a list of exactly four numbers, true and false not
  counting as numbers; a number that is not finite; or a quaternion whose norm lies outside [0.9, 1.1]);
- **Z6**: the frame is an outlier of the one-flight fit;
- **Z5**: the wide camera's or the depth camera's centre lies inside the solid of a geom (a primitive's closed form, a
  mesh's convex hull; a point on the surface counts as inside). This is tested before any ray is cast: a ray that
  starts inside a solid would see the far side of the solid.

The one-flight fit (rules Z3 and Z6), on the scored moments with a valid pose (no other entry enters it): the declared
wide-camera poses (p_k, R_k) against the estimate's wide-camera poses (p'_k, R'_k) read as in the section on time.
Starting with every frame as an inlier: R_g = the rotation nearest to the sum over the inliers of R'_k R_k^T; with R_g
fixed, the least-squares scale s and translation t of p'_k = s R_g p_k + t over the inliers, s clamped to [0.8, 1.25]
(s is one when the squared distances of the inliers' turned centres R_g p_k from their mean sum to less than 1e-06
m^2); every frame's residuals e_k = |s R_g p_k + t - p'_k| and a_k = the angle between R_g R_k and R'_k; a frame is an
inlier when e_k <= 0.5 m and a_k <= 15 degrees. This is repeated until the inlier set no longer changes, at most 10
times.

Within the scoring (not gates, but each can make a term 0):

- **S1**: a scored pixel whose ray meets nothing in your twin scores 0, as does a range at or beyond the band's zero
  end; a clear move that your twin's frame blocks within its length scores 0; a blocked move that your twin's frame
  finds clear over 3.5 m scores 0, as does one whose stop distance differs by 0.4 m or more.
- **S2**: the structure enters the depth term only multiplied by the content.
- **S3**: J = max(0, aC + aBC - 1): no hazard credit unless clear moves and moves blocked by content are both told
  apart.

No other rule sets a score, a flight, a key frame, a pixel or a move to zero. A failure of the grader itself makes the
verification void; it is never scored: its own error, a crash of one of its processes, a time-out or an exhausted
memory in any part of the grading other than rule Z7's two steps, and the verification's own deadline wherever it
falls, inside those two steps included.

## The depth term

The scored pixels of a key frame are content pixels and structure pixels, as `flight_brief.md` defines them. For a
scored pixel with measured range r and your twin's range q along the same ray, with b = max(0.06 m, 0.04 r):

c = clip((3 b - |q - r|) / (2 b), 0, 1)

so c is 1 within b, falls linearly and is 0 at 3 b(r) and beyond; a ray that meets nothing gives c = 0, and so does
every pixel of a zeroed moment. Dc_k is the mean of c over scored moment k's content pixels, for the moments with at
least 1500 of them; Dc is the mean of Dc_k over those moments, each moment weighing the same. Ds is formed alike from
the structure pixels. D is the product of Dc and Ds (rule S):

D = Dc x Ds

## The hazard term

The moves come in three classes: clear, blocked by content, blocked by structure. A blocked move is blocked by
structure when the given module shell alone, placed by the maintainers' one best rigid fit (as for the depth term's
classes), stops the move at about the same distance, and blocked by content when the shell alone would let it pass or
would stop it clearly later; blocked moves in between are not scored. A blocked move may come with a shorter clear
move along the same direction that ends well before its stop.

On your twin's depth frame of the move's scored moment, the check of the section above, once:

- a clear move scores 1 when your twin's frame is clear along it over its length, else 0;
- a blocked move scores 0 when your twin's frame is clear along it over 3.5 m; otherwise, with d the difference
  between your stop distance and the real one, it scores 1 when d <= 0.16 m, 0 when d >= 0.4 m, and linearly between
  (stop distances are whole numbers of path samples, so the grader compares sample counts, 8 and 20);
- every move of a zeroed moment scores 0.

With aC, aBC and aBS the mean scores of the three classes:

J = max(0, aC + aBC - 1),  H_moves = J x (0.5 + 0.5 aBS)

The scored moments are also judged one by one (rule P): one wrong obstacle verdict at a scored moment loses that
moment. A scored moment that has at least one scored move is lost when one of its scored moves scores 0: a clear move
that your twin's frame blocks within its length, a blocked move that your twin's frame finds clear over 3.5 m, a
blocked move whose stop differs by 0.4 m or more, or any move of a moment that rule Z4, Z5 or Z6 zeroes. A move with
partial credit does not lose its moment. With K the number of scored moments that have at least one scored move and
K_lost the number of them that are lost:

H_poses = 1 - K_lost / K,  H = min(H_moves, H_poses)

For example, take a scored moment with nine scored moves. If your twin's frame blocks one of its clear moves within
the move's length, that move scores 0 and the moment is lost, whatever its eight other moves score. If instead the
moment's only flaw is one blocked move whose stop lies 0.2 m from the real one (10 path samples, between 8 and 20),
that move scores about 0.83, its partial credit, and the moment is kept.

## The flight's score

The flight scores H x D. The twin score is G1 x G0 x that score, with G1 the integrity gate (rule Z1) and G0 the
validity gate (rule Z2), each 1 or 0; a flight that fails rule Z3 or rule Z7 scores 0. The reward is the twin score;
full credit is a twin score of 1.

## Parameters

Every number the gates and the scoring read:

| parameter | value | meaning |
|---|---|---|
| `key_frames.per_flight` | 24 | scored moments (key frames) per flight; which ones is not told |
| `key_frames.time_offset_wide_minus_depth_stamp` | 0.067 | s: the estimate is read at a frame's recorded stamp minus this |
| `wide_camera.body_mount_translation` | [0.1177, -0.0422, -0.0826] | m: the wide camera's centre in the body frame |
| `wide_camera.body_mount_quaternion_xyzw` | [0.5, 0.5, 0.5, 0.5] | x, y, z, w: turns wide-camera vectors into body vectors |
| `two_camera_transform.translation` | [0.0764645596864683, -0.0021427546911120254, 0.005471206683257068] | m: t of X_depth = R X_wide + t |
| `two_camera_transform.quaternion_xyzw` | [-0.006069374726179163, 0.008046021087535497, 0.9999290115421315, 0.006355793535658162] | x, y, z, w of R |
| `depth_camera.image_size` | [224, 171] | columns, rows of the depth frame and of the ray table |
| `depth_camera.rays` | 38304 | pixels of the ray table |
| `depth_camera.never_returning_corner_pixels` | 1304 | pixels without a ray (NaN in the table) |
| `depth.scored_range_min` | 0.2 | m: shortest scored measured range |
| `depth.scored_range_max` | 5 | m: longest scored measured range |
| `depth.band_full_abs` | 0.06 | m: b(r) = max(this, the next row x r) |
| `depth.band_full_rel` | 0.04 | share of r in b(r) |
| `depth.band_zero_factor` | 3 | credit 0 at this many b(r) and beyond |
| `depth.min_class_pixels_per_frame` | 1500 | scored pixels of a class for a frame to count in that class |
| `hazard.voxel` | 0.08 | m: edge of the occupancy grid's cells |
| `hazard.min_range` | 0.2 | m: shortest range that makes a point |
| `hazard.max_range` | 4 | m: a point needs a range below this |
| `hazard.robot_radius` | 0.28 | m: the robot's radius in the check |
| `hazard.collision_distance` | 0.02 | m: added to the radius |
| `hazard.inflation_radius` | 0.3 | m: radius of the inflation (the two rows above, summed) |
| `hazard.inflation_offsets` | 251 | integer cell offsets of the inflation |
| `hazard.path_sample` | 0.02 | m: step along a move and of the neighbour grid |
| `hazard.move_start` | [0.03462969950432764, -0.08609609681043709, -0.11139868174201827] | m: the robot's centre in the depth camera frame, where every move starts |
| `hazard.directions_azimuth` | [-20, -15, -10, -5, 0, 5, 10, 15, 20] | deg: azimuths of the move family |
| `hazard.directions_elevation` | [-12, -6, 0, 6, 12] | deg: elevations of the move family |
| `hazard.lengths` | [0.6, 0.8, 1, 1.5, 2, 2.5, 3, 3.5] | m: lengths a clear move may have |
| `hazard.max_length` | 3.5 | m: longest move; blocked moves are checked to this length |
| `hazard.stop_band_full` | 0.16 | m: stop difference with full credit |
| `hazard.stop_band_zero` | 0.4 | m: stop difference with no credit |
| `hazard.structure_floor` | 0.5 | f in H_moves = J x (f + (1 - f) aBS) |
| `score.hazard_aggregation` | min_of_moves_and_poses | the hazard term in force (rule P) |
| `score.depth_aggregation` | product | the depth term in force (rule S) |
| `score.flight_aggregation` | product | the flight score in force |
| `gate.one_flight.position_threshold` | 0.5 | m: inlier line of the one-flight fit (position) |
| `gate.one_flight.angle_threshold` | 15 | deg: inlier line of the one-flight fit (angle) |
| `gate.one_flight.scale_clamp` | [0.8, 1.25] | the fit's scale is clamped to this interval |
| `gate.one_flight.scale_min_spread` | 1e-06 | m^2: below this spread of the declared centres the scale is 1 |
| `gate.one_flight.min_inliers` | 12 | fewer inliers: rule Z3 |
| `gate.one_flight.max_iterations` | 10 | fits at most |
| `gate.quaternion_norm_range` | [0.9, 1.1] | a declared quaternion's norm must lie in this interval (rule Z4) |
| `gate.coordinate_bound` | 100 | m: every geom's extent within this of the origin on each axis |
| `gate.ray_cutoff` | 1000 | m: the cutoff handed to the simulator's ray call |
| `runtime.scene_time_limit_s` | 120 | s of the grader's wall clock for building one flight's ray scene (rule Z7) |
| `runtime.rays_time_limit_s` | 300 | s of the grader's wall clock for casting one flight's rays (rule Z7) |
| `runtime.scene_memory_limit_mb` | 6144 | MB of address space of the grader's process that builds a flight's ray scene and casts its rays (rule Z7) |
| `caps.files` | 5000 | files and folders in the whole submission |
| `caps.megabytes` | 64 | MB in the whole submission |
| `contract.megabyte_bytes` | 1048576 | bytes in one MB of every size cap |
| `caps.geoms_per_flight` | 2000 | geoms per flight |
| `caps.bodies_per_flight` | 2000 | bodies per flight |
| `caps.mesh_assets_per_flight` | 8000 | mesh assets per flight |
| `caps.xml_elements_per_flight` | 20000 | XML elements of scene.xml |
| `caps.nesting_levels_per_flight` | 400 | levels of nested elements in scene.xml, the root element being level 1 |
| `caps.mesh_faces_per_flight` | 500000 | triangles per flight: each mesh asset's once for every geom that names it, once if no geom does, summed |
| `caps.mesh_megabytes_per_flight` | 26 | MB of mesh files per flight |
| `caps.stl_faces_per_file` | 200000 | triangles in one STL file |

## Practical notes (not rules)

- A whole module modelled as one mesh is one convex solid: the cameras are inside it (rule Z5).
  `module_shell/shell.obj` used as one mesh is such a solid too.
- A transparent geom is still hit by the rays. MJCF angles are degrees unless `<compiler angle="radian">`.
- If you scale or move your twin, move the declared camera poses with it. The poses are the wide camera's, not the
  body's.
- `python3 tools/check_twin.py` checks rule L on your `cameras.json` and prints, for every entry, the outcome of rules
  Z4, Z5 and Z6 (each applies where the frame is a scored moment), the one-flight fit of all your entries with a valid
  pose and the frames whose residuals are largest (the grader fits the scored moments only), the rules L, P and S in
  force, and the times of its own build of the ray scene and of its own cast of one frame's rays beside rule Z7's
  limits; `--depth`, `--hazard` and `--overlay` show your twin's depth frame (with the cast's time), its hazard check
  and its depth edges over a wide frame, each with the build's time, at any given frame you name, or without a frame
  at a spread of 24 frames evenly spaced over the flight. The spread is not the set of scored moments: the tool does
  not know which moments are scored, never says it, and scores nothing. When your twin's ray scene does not fit inside
  rule Z7's address-space limit, `--depth`, `--hazard` and `--overlay` stop with exit code 2 and print rule Z7's
  reason instead of a time.
