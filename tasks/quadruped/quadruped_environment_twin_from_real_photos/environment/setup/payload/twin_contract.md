# Twin contract

This is the technical contract of the twin: what you deliver, the rules the files must follow, the standard
simulation and the probes the twin is run through, the robot that walks it, and how it is assessed. Every number
below is the number the code uses (`twin_gate.py`, `twin_sim.py` with `twin_sim_config.json`, `twin_camera.py`,
`twin_slice.py`, the robot's `walk_probe.py`, and the grader's configuration). `site_brief.md` explains what the
twin is for, what counts as a unit, which gates slide and what the surveyed area is.

## 1. What you deliver

`/home/user/submission/` holds:

| path | content |
|---|---|
| `scene.xml` | the twin: one MuJoCo MJCF file (MuJoCo 3.16.0), no includes |
| `objects.json` | which bodies are the units, with their class labels, and which bodies are the gates, with their slide joints (section 6) |
| `assets/` (any layout) | the mesh files and textures that `scene.xml` references, by relative path |

- Only `scene.xml`, `objects.json` and the files `scene.xml` references are read. Every other file is ignored, but
  it still counts toward the folder caps, so keep photos, reconstructions, point clouds, renders, notes and scratch
  work outside `submission/`.
- Folder caps: at most 5000 entries (files and folders) and 300 MB in total.
- File caps: `scene.xml` at most 16 MB, `objects.json` at most 1 MB, each mesh
  file at most 50 MB, each PNG at most 64 MB and 4096 px a side;
  at most 5000 referenced files, 8000 meshes and
  500 textures; at most 100000 XML elements, nested at most
  64 deep.
- Formats: meshes `.stl` (binary), `.obj` or `.msh`; textures `.png` only. A referenced path is relative (to the
  submission folder, or through `meshdir` / `texturedir` / `assetdir`), has no `..`, no symbolic link on the way,
  and only the characters `A-Z a-z 0-9 _ . + - ( ) /` and spaces. An OBJ's `mtllib` must be a plain file name.
- `scene.xml` is UTF-8, without DOCTYPE or entity declarations, namespaces or processing instructions (XML comments
  are fine).

## 2. Frames and units

- World: z up, metres, kilograms, seconds. Gravity points along -z (the standard simulation sets it, section 8).
- The initial state `qpos0` is the surveyed state: every unit as written in `scene.xml`, every gate where the
  photos show it. There are no keyframes and no joint `ref`, so a gate's slide joint is at zero in that state.
- Where the world origin is and which way x points are yours to choose: the twin is placed on the site through its
  photo cameras (sections 3 and 10). Its vertical is not free (your +z must be the bay's up direction), and neither
  is its scale: the robot, the lift and every tolerance of the simulation take one unit of length of your twin as
  one metre (section 10 says where the fitted scale is used instead).

## 3. The photo cameras

For every photo `site_NNN.jpg` that you registered, `scene.xml` contains a camera named `photo_NNN` (the photo's
three digits: `photo_007` for `site_007.jpg`) with your estimate of where that photo was taken from, relative to
your twin, and of its intrinsics:

- `mode="fixed"`, on the world body or on a static body (a body without joints whose parents have none);
- `resolution="6208 4135"`, the size of every photo as stored (a
  photo shot with the camera on its side is stored turned, and its camera is turned with it);
- a perspective camera with intrinsics in one of MuJoCo's two forms:
  - **sensor form** (any focal lengths and principal point): `sensorsize` (any positive size), `focalpixel="fx fy"`
    and `principalpixel` (or the length forms `focal` / `principal`; the pixel forms win when both are given);
  - **fovy form**: `fovy` only, which means fx = fy and a centred principal point;
- fx and fy between 1000 and 20000 px, and the principal point inside the image.

At least 12 valid photo cameras are required, of the 44 photos. Each photo
camera is checked on its own: one that breaks a rule above is reported
as invalid and left out, and the gate fails only when fewer than 12 valid ones remain. A
camera whose name starts with `photo_` without being the `photo_NNN` of a given photo fails the gate; cameras with
other names are ignored (a note).

MuJoCo's convention differs from OpenCV's in two places, and both are easy to get wrong. In OpenCV terms (pixel
(u, v) = (column, row), (0, 0) the centre of the top-left pixel, camera axes x right, y down, z forward), with W x H
the resolution:

```
sensor form:  focalpixel     = (fx, fy)
              principalpixel = ((W - 1)/2 - cx, (H - 1)/2 - cy)      # the image centre MINUS the principal point
fovy form:    fy = (H/2) / tan(fovy/2),  fx = fy,  (cx, cy) = ((W - 1)/2, (H - 1)/2)
camera frame: R_cv = cam_xmat @ diag(1, -1, -1),  p = cam_xpos       # MuJoCo cameras look along their -z, y up
              x_cam = R_cv^T (x_world - p)
```

`twin_camera.py` implements this and nothing else: `encode(K, W, H)` gives the attributes for an OpenCV K,
`camera_xml(name, K, W, H, R_cv, p)` a whole `<camera>` element, `decode(model, cam)` reads K back from a compiled
model, and `camera_frame`, `project`, `backproject` and `pixel_rays` do the geometry.

`python3 check_twin.py` prints one line per photo camera: the decoded fx, fy, cx, cy; the angle between the
image's up direction and your world +z (close to zero for a level landscape shot, close to a right angle for a
shot with the camera on its side); the elevation of the optical axis; and the shares of your twin's collision
geometry (the corners and centres of the geoms' bounding boxes) that lie in front of the camera and inside its
image. A camera that sees none of your twin, or whose image-up points down, almost always has the wrong convention.

Your photo cameras are your estimate of where the photos were taken from, relative to your twin. The assessment
places your twin on the site through them (section 10), so they must be the cameras of the twin you submit: in its
frame and at its scale. A twin that was moved, turned, levelled or rescaled after its cameras were written, without
moving the cameras with it, is placed wrongly or not at all.

## 4. Units

A unit is a loose thing that a person, a pallet truck or a forklift moves as one piece, with at least two of its
three extents 0.3 m or more, standing in the surveyed area and not inside a cage or a rack;
vehicles and trailers are units too (`site_brief.md` has the rule in full, with what is one unit and what is none).
Every unit is:

- a top-level body (a child of `<worldbody>`, directly or through `<frame>` elements, never inside another body)
  with a `<freejoint/>` (or a joint of type free), and no other joint; it may have child bodies without joints
  (they move with it);
- listed in `objects.json` with its class label from `class_names.json`;
- given collision geometry that follows the **solid convention**: the unit's solid is the union of its collision
  geoms, and it is the filled outer envelope of each visually distinct part of the unit, with the open space
  between and under the parts left open:
  - a pallet with collars, a crate, a closed box: one filled box from its underside to its top rim (not a base and
    four thin walls);
  - an open container (a bin, a tub, a skip without a lid): its outer shell, filled to the rim;
  - a flat pallet: its filled bounding slab (the gaps between the boards and under the deck are filled);
  - a thing on wheels, castors or legs (a cart, a trailer, a vehicle, a table): its deck or body as one or more
    pieces, and each wheel, leg, post or drawbar as its own piece; the space under the deck or the body stays open,
    because the robot may pass there;
  - a stack or a load of long goods: its envelope, overhangs included.

  Between 1 and 64 collision geoms per unit. A collision geom is any geom whose
  `contype` or `conaffinity` is not 0. Use primitives (box, sphere, capsule, cylinder, ellipsoid) or meshes; **a
  mesh geom collides as the convex hull of its vertices**, so model a shape that is not convex as several convex
  pieces. Planes cannot be part of a unit. A unit's collision meshes together have at most
  50000 vertices;
- optionally given visual geoms: `contype="0" conaffinity="0"` and no mass (`density="0"` or `mass="0"`). They are
  drawn by renders and play no part anywhere in the assessment;
- physically plausible:
  - density (the unit's mass divided by the volume of the union of its collision geoms) between
    10 and 3000 kg/m^3;
  - sliding friction of every collision geom between 0.2 and 1.5 (the standard
    simulation then uses its own value, section 8);
  - inertia consistent with the shape: the smallest principal moment at least 0.0001 x mass x L^2
    (L the unit's largest extent), the largest at most mass x d_max^2 (d_max the largest distance from the centre of
    mass to the collision geometry), with a 1 % tolerance;
  - the centre of mass inside the convex hull of the collision geometry; the unit no larger than
    8 m.

A unit that breaks one of the plausibility rules, the geom-count or vertex rule, or has a visual geom with mass, is
**excluded**: it is removed from the twin before any simulation and any walk (it can neither support nor disturb
anything, and the robot does not meet it) and scores 0 in every criterion, while it still counts as submitted. The
same happens to a unit that the settle check throws out (section 8). The rest of the twin is assessed as usual.
`check_twin.py` lists each unit's status with the reasons.

A label that is not in `class_names.json` is kept, but never counts as correct.

## 5. The environment and the gates

- Everything that is not a unit or a gate is environment (`site_brief.md` says what belongs to it): geoms of static
  bodies, that is of the world body or of bodies without joints. Their collision geoms (contype or conaffinity not
  0) are what the assessment and the robot meet; visual geoms are ignored. Planes are allowed in static bodies
  only, and a plane is infinite.
- **No height fields**: the gate refuses them. A floor is planes, boxes or convex meshes; a floor that is not level
  is a few tilted slabs.
- A wall of wire mesh is a wall, and a mesh gate is a solid leaf: model each as a thin solid panel. Like every
  mesh geom, a wall, a room or a rack given as ONE mesh collides as its convex hull, which is a solid block
  (section 12).
- Chains, tape barriers, cables and hoses get **no collision geometry**: leave them out, or make them visual geoms
  (`contype="0" conaffinity="0"`). The robot's rounds pass under a hanging chain; a chain modelled as a solid would
  stop the robot where the site lets it through.
- A **gate** is one body with exactly one slide joint (`<joint type="slide" axis="..."/>`), hanging from a static
  body, with collision geometry (no plane) and no further joint inside it; it is listed in `objects.json` with the
  name of that joint. The gate body holds the moving leaf only: the rail, the posts and the wall it slides along are
  environment. There is no other joint in the twin than the gates' slides and the units' free joints, and every
  slide joint belongs to a listed gate.
- The slide's axis is the gate's direction of travel along its rail. "Opening" is the direction along the axis in
  which the rail continues beyond the leaf in the photos; the assessment takes it from the real gate. Directions and
  positions of a gate are measured at the centre of the leaf's collision geometry (the middle of the box around its
  collision geoms), never at the body's origin, which may sit anywhere.
- **Travel rule.** The slide's limits (`range`, limited) are the real gate's travel, and they must be plausible for
  a gate of its size. Three conditions, with the surveyed position (`qpos0`, the joint's zero) inside the range:
  1. the total travel (the range's length) is between 0.5 and
     1 times the leaf's width (the extent of the gate's collision geoms along the
     slide, at `qpos0`);
  2. the gate can open at least 0.4 m beyond the surveyed position in the opening
     direction;
  3. it can travel at most 0.1 m beyond the surveyed position against the opening
     direction (the closing direction), plus whatever the real gate itself can travel that way. A gate that the
     photos show shut has nowhere to go in the closing direction, so a range that reaches equally far both ways from
     the surveyed position fails whichever way the gate opens.

  A gate that breaks this rule scores 0 in C5's gate term, without failing the twin; `check_twin.py` prints the check
  for both directions of the axis, with the travel left on the opening and on the closing side of each.
- In the standard simulation the gates are held at the surveyed position and never collide with the environment;
  units collide with everything. In the walks the gates are fixed there.

## 6. `objects.json`

```json
{"format": "site-twin-objects/1",
 "items": [{"body": "<body name>", "label": "<class name>"}, ...],
 "gates": [{"body": "<body name>", "joint": "<its slide joint>"}, ...]}
```

- Every top-level body with a free joint must be listed under `items` (the units), and every listed body must be
  one; a body appears once.
- Between 0 and 150 items. `gates` must be present (an empty list when you model no
  gate); a gate body cannot also be an item. Other keys are ignored (with a note).

## 7. Allowed MJCF

The gate reads `scene.xml` as XML before MuJoCo sees it. Only these elements, each under the parents and with the
attributes listed, are accepted; anything else is rejected with a reason.

| element | allowed under | attributes |
|---|---|---|
| `<mujoco>` | (root) | model |
| `<compiler>` | `<mujoco>` | angle, assetdir, autolimits, discardvisual, eulerseq, fitaabb, inertiafromgeom, inertiagrouprange, meshdir, texturedir, usethread |
| `<option>` | `<mujoco>` | ccd_iterations, ccd_tolerance, cone, gravity, impratio, integrator, iterations, jacobian, ls_iterations, ls_tolerance, noslip_iterations, noslip_tolerance, sdf_initpoints, sdf_iterations, solver, timestep, tolerance |
| `<flag>` | `<option>` | actuation, autoreset, clampctrl, constraint, contact, damper, diagexact, energy, equality, eulerdamp, filterparent, frictionloss, fwdinv, gravity, invdiscrete, ipc, island, limit, midphase, multiccd, nativeccd, refsafe, sensor, spring, warmstart |
| `<statistic>` | `<mujoco>` | center, extent |
| `<visual>` | `<mujoco>` | (none) |
| `<global>` | `<visual>` | azimuth, bvactive, cameraid, elevation, ellipsoidinertia, fovy, glow, ipd, linewidth, offheight, offwidth, orthographic, realtime |
| `<quality>` | `<visual>` | numquads, numslices, numstacks, offsamples, shadowsize |
| `<headlight>` | `<visual>` | active, ambient, diffuse, specular |
| `<map>` | `<visual>` | actuatortendon, alpha, fogend, fogstart, force, haze, shadowclip, shadowscale, stiffness, stiffnessrot, torque, zfar, znear |
| `<scale>` | `<visual>` | actuatorlength, actuatorwidth, camera, com, connect, constraint, contactheight, contactwidth, forcewidth, framelength, framewidth, frustum, jointlength, jointwidth, light, selectpoint, slidercrank |
| `<rgba>` | `<visual>` | actuator, actuatornegative, actuatorpositive, bv, bvactive, camera, com, connect, constraint, contactforce, contactfriction, contactgap, contactpoint, contacttorque, crankbroken, fog, force, frustum, haze, inertia, joint, light, rangefinder, selectpoint, slidercrank |
| `<default>` | `<mujoco>`, `<default>` | class |
| `<asset>` | `<mujoco>` | (none) |
| `<mesh>` | `<asset>`, `<default>` | class, face, file, inertia, name, normal, refpos, refquat, scale, smoothnormal, texcoord, vertex |
| `<texture>` | `<asset>` | builtin, colorspace, file, gridlayout, gridsize, height, hflip, mark, markrgb, name, nchannel, random, rgb1, rgb2, type, vflip, width |
| `<material>` | `<asset>`, `<default>` | class, emission, metallic, name, reflectance, rgba, roughness, shininess, specular, texrepeat, texture, texuniform |
| `<layer>` | `<material>` | role, texture |
| `<worldbody>` | `<mujoco>` | (none) |
| `<body>` | `<worldbody>`, `<body>`, `<frame>` | axisangle, childclass, euler, name, pos, quat, xyaxes, zaxis |
| `<frame>` | `<worldbody>`, `<body>`, `<frame>` | axisangle, childclass, euler, name, pos, quat, xyaxes, zaxis |
| `<inertial>` | `<body>` | axisangle, diaginertia, euler, fullinertia, mass, pos, quat, xyaxes, zaxis |
| `<joint>` | `<body>`, `<default>` | axis, class, group, limited, name, pos, range, type (type free or slide only) |
| `<freejoint>` | `<body>` | group, name |
| `<geom>` | `<worldbody>`, `<body>`, `<frame>`, `<default>` | axisangle, class, conaffinity, condim, contype, density, euler, fitscale, friction, fromto, gap, group, margin, mass, material, mesh, name, pos, priority, quat, rgba, shellinertia, size, solimp, solmix, solref, type, xyaxes, zaxis (type box, capsule, cylinder, ellipsoid, mesh, plane, sphere) |
| `<site>` | `<worldbody>`, `<body>`, `<frame>`, `<default>` | axisangle, class, euler, fromto, group, material, name, pos, quat, rgba, size, type, xyaxes, zaxis |
| `<camera>` | `<worldbody>`, `<body>`, `<frame>`, `<default>` | axisangle, class, euler, focal, focalpixel, fovy, ipd, mode, name, pos, principal, principalpixel, projection, quat, resolution, sensorsize, target, xyaxes, zaxis |
| `<light>` | `<worldbody>`, `<body>`, `<frame>`, `<default>` | active, ambient, attenuation, bulbradius, castshadow, class, cutoff, diffuse, dir, directional, exponent, intensity, mode, name, pos, range, softness, specular, target, type |

Inside `<default>`, an element carries default attributes only (no names, files or targets).

Rejected elements include `<actuator>`, `<attach>`, `<composite>`, `<connect>`, `<contact>`, `<custom>`, `<deformable>`, `<equality>`, `<exclude>`, `<extension>`, `<fixed>`, `<flex>`, `<flexcomp>`, `<hfield>`, `<include>`, `<instance>`, `<key>`, `<keyframe>`, `<lengthrange>`, `<model>`, `<numeric>`, `<pair>`, `<plugin>`, `<replicate>`, `<sensor>`, `<size>`, `<skin>`, `<spatial>`, `<tendon>`, `<text>`, `<tuple>`, `<weld>`. An attribute that is not in the table is rejected too: among
others joint springs, damping, armature, friction loss and `ref`, `mocap`, `gravcomp`, `sleep`, contact overrides,
fluid and wind settings, `hfield`, `maxhullvert`, `strippath`, `settotalmass`, `boundmass`, `boundinertia`,
`balanceinertia`, cube-face texture files and compiler outputs.

Then the gate compiles a copy of the twin with textures removed: **any MuJoCo compile warning fails the twin**.
On the compiled model it checks:

- no equality, tendon, actuator, sensor, flex, skin, plugin, mocap body, contact pair or exclude, keyframe, height
  field or custom data;
- at most 2000 bodies, 20000 geoms, 6000 collision geoms (the
  environment included), and 3 million mesh faces and 3 million mesh vertices in
  total; a mesh used by a collision geom has at most 20000 vertices, and the
  convex hulls of all collision meshes together at most 400000 vertices;
- **column depth**: on a grid of 0.25 m squares over the floor plan, at most
  128 collision geoms' bounding boxes reach over any one square (planes are not counted). The
  walks keep looking for the surface under the robot along vertical lines, and thousands of thin layers stacked
  over one spot would make them arbitrarily slow; an ordinary twin stays far below;
- finite numbers; no body further than 200 m from its parent, no geom further than that
  from its body, no geom size above 100 m, no mesh vertex further than that from its mesh's
  centre;
- no joint stiffness, damping, armature, friction loss or margin, no gravity compensation, adhesion, surface
  velocity or fluid forces;
- the photo cameras (section 3), the units (section 4), the gates and the joints (section 5).

The gate also writes notes that never fail the twin: an invalid photo camera; a gate whose travel rule is not met
in either direction, or not in the direction that the twin's own rail suggests, or whose opening direction cannot
be read from the twin's rail at all (section 11); and the **hull note**: a static collision geom whose collision
shape (a mesh's convex hull, a primitive, a plane's half-space) contains a photo camera centre, or contains another
body's centre of mass by more than 1 cm. Such a geom is almost always a room or a large
object exported as one mesh.

## 8. The standard simulation

Every twin is simulated the same way, by `twin_sim.py` with `twin_sim_config.json` (which you have). Whatever
`<option>`, contact parameters, friction or contact bits the twin sets, the simulation first replaces them:

- `mjOption` is reset to MuJoCo's defaults, then: timestep 0.002 s, integrator
  `implicitfast`, `elliptic` friction cone, impratio 10, solver
  `Newton` with 100 iterations (50 line-search iterations) and
  tolerance 1e-10, gravity (0, 0, -9.81) m/s^2, autoreset off;
- every collision geom: condim 3, solref (0.005, 1), solimp (0.95, 0.99, 0.001, 0.5, 2), margin and gap
  0, friction (0.6, 0.005, 0.0001) (sliding, torsional, rolling) whatever the twin says;
- contact bits: units collide with units, the environment and the gates; a gate collides with units only (it
  slides along its rail, never against the building);
- every force that does not come from a contact is removed (joint springs and damping, friction loss, armature,
  gravity compensation, surface velocities, adhesion, fluid forces);
- driven bodies are kinematic: their pose and velocity are written before every step and a large armature
  (10000000) keeps contacts from pushing them. Every gate is held at its surveyed position
  throughout. Masses and inertias are kept as delivered.

**The settle.** From `qpos0`, first a relaxation of 1 s: the physics is stepped and every
velocity is set to zero after each step, so a small penetration is pushed out slowly instead of launching the unit,
and a small gap closes without an impact. Then 3 s of free dynamics. A unit's motion is
measured at the centroid of its collision geometry (geom centres weighted by the volume of their bounding boxes),
its rotation as the angle of its rotation since the start; both are sampled at every step of both phases against
the pose you submitted, so what the relaxation moves counts as motion. A unit is **static** when its displacement
stays within 10 mm and its rotation within 2 deg over the whole settle.

**Support.** A unit b is supported by a (another unit, an environment geom, or a gate) when the contacts between
them push b upward: contact normals within 60 deg of vertical (up-component at least
0.5), and a summed upward force averaged over the last 0.5 s of the
settle of at least 0.1 of b's weight.

**Blow-up check (part of G0).** The same relaxation, then 2 s of free dynamics (the gates
held, every unit free). The twin blows up on a non-finite state, on MuJoCo reporting bad accelerations, on a gain of
mechanical energy, KE(t) > PE(0) - PE(t) + e with e = (total unit mass) x (2 m/s)^2 / 2
(what releasing a deep interpenetration produces), or on a unit whose centre of mass moves faster than the free-fall
bound g t + 2 m/s (t counted from the end of the relaxation). The units that break the
speed bound are **excluded** (section 4) and the check runs once more without them. G0 fails only on a non-finite
state, on a blow-up that no unit's speed accounts for, or when the second run blows up as well. Ordinary falls,
slides and topples are not G0 failures: C4 charges them.

**The lift probe** of unit i, from the settled state: i is lifted 100 mm straight up (world +z)
along a minimum-jerk profile over 1 s, then held for 1 s, without turning; the
gates stay held, every other unit is free. Every other unit's outcome is read at the end, from its displacement d
and rotation since the probe started:

- **carried**: it follows the lifted unit: |d - d_lifted| within 30 mm, rotation below
  10 deg and a rise of at least 60 mm;
- **stayed**: |d| below 10 mm and rotation below 5 deg;
- **disturbed**: not following, and |d| at least 20 mm or rotation at least
  10 deg;
- **intermediate**: anything else (it matches no outcome).

**Gates** are never driven. A gate is assessed on where its leaf is, on its slide axis and on its travel rule
(section 5), all read from the twin as submitted.

`python3 check_twin.py --settle` runs the blow-up check and the settle on your twin and `--probes` the lift probes
and the gates' travel rule, with the same code. Lengths there are in your twin's units; the assessment converts
displacements to true units first (section 10).

## 9. The robot and the walk probes

`robot/` holds the robot of the rounds: a small four-legged robot with a fixed walking controller
(`robot/robot_card.md`: its size and mass, how a run works, what it gets through and what stops it). The assessment
walks it with exactly these files. `robot/walk_probe.py` drives it along a **route**, a list of floor points with a
gait per segment (`walk` or `crouch`), and reports how the run ended (reached, stopped, fell) and its **progress**,
the share of the route's length that the robot covered.

A scored walk is set up as follows (`check_twin.py --walk` applies the same conditions to routes of your own):

- **The twin.** The twin as submitted (`qpos0`, not the settled state), without the excluded units. Every unit is
  held where your `scene.xml` puts it (its free joint is removed) and every gate is fixed at its surveyed position
  (its slide joint is removed): the robot can neither push a unit nor open a gate, so a walk depends on geometry
  only.
- **Contacts.** Every collision geom gets the same contact parameters, whatever your scene says: it collides with
  the robot, sliding friction 0.8, MuJoCo's default contact softness (solref
  (0.02, 1)), no margin, no gap. Visual geoms stay inert.
  The harness replaces `<option>` with its own simulation settings (robot card).
- **The route.** The hidden routes are given on the site and mapped into your twin by the inverse of the alignment
  (section 10): their points keep their place on the site and their lengths follow your twin's scale, while the
  robot keeps its size. A twin built too small or too large therefore gives the robot less or more room than the
  site does.
- **The start.** The expected height of the start surface is that of the true surface under the first point,
  mapped into your twin. A ray along your -z, from 0.3 m above that height to
  0.3 m below it, finds the surface the robot starts on: the top of the first solid below the
  ray's origin. If the origin
  lies inside a solid, or no surface lies in that window, the route is not reached (progress 0). The harness then
  stands the robot on that surface; if something occupies the standing robot's space there, or the surface it finds
  under the robot leaves the window, the run is not reached (progress 0).
- **The starts.** Each route is walked 4 times: from the nominal start, and from starts
  shifted by up to 2 cm along and across the first segment, turned by up to
  2 deg and begun at another phase of the gait (the same seeded list for every
  submission).
- **The outcome.** A run **reaches** when the harness reports that the robot arrived at the last point (robot card)
  or its progress is at least 0.95; any other end (stopped, fell, or a twin that breaks the
  harness) is **not reached**, and its progress is what is compared. The outcome is decided in simulated time: the
  harness's own time limit for the route. No outcome comes from the wall clock: if the verifier cannot finish a
  walk within its (generous) guard, the verification is void and repeated; it is never scored as not reached.

**The hidden routes** are routes a round would take: across open floor, between units, pillars and walls, under
decks and overhanging loads, up to walls and cage fronts. At least 5 of them are
**open** (on site the robot gets through) and at least 5 **blocked**. What decides
a route is a passage's width, an overhead clearance or a solid thing standing on the route, never a step, a kerb or
a slope, and it keeps a margin from the sizes at which the robot's outcome turns (robot card): a deciding passage
is at most 0.26 m or at least 0.36 m wide,
a deciding clearance for walking at most 0.29 m or at least
0.35 m high, for crouching at most
0.23 m or at least 0.28 m.
A route is kept only when the true twin's outcome is the same under every variant of the robustness rule
(section 10) and the photos determine what decides it.

## 10. How the twin is assessed

```
twin_score = G1 x G0 x (0.15 C1 + 0.3 C2 + 0.15 C3 + 0.14 C4 + 0.06 C5 + 0.15 C6 + 0.05 C7)
```

Each criterion is in [0, 1]. A **band(e, full, zero)** gives 1 for an error e at or below `full`, 0 at or above
`zero`, and is linear between. C3, C5, C6 and C7 are multiplied by the inventory C1: a bay without its units earns
nothing for its building, its gates, its walks or its scale.

**G1, submission integrity.** The folder is the one your session wrote: nothing in it may be written after the
session ends, it must be a real folder (not a symbolic link, not replaced or renamed into place afterwards), and it
must stay within the folder caps (section 1). Processes left running are stopped when the assessment starts.
Otherwise G1 = 0.

**G0, validity.** The gate of sections 1-7 (`twin_gate.py`, the same code `check_twin.py` runs) passes, the settle
does not blow up (section 8), and your photo cameras fit the true ones (step 1 of the alignment). Otherwise G0 = 0.
An excluded unit does not fail G0 (section 4).

**The true twin.** The assessment compares your twin with a true twin of the bay, built to this same contract (the
same solid convention, the same simulation, the same robot) from a survey that you do not have. Three things about
it matter to you:

- Parts of it are flagged *not determined*: what the photos do not show from at least two positions or the survey
  did not measure (hidden backs and undersides, completed from the usual shape of the class). They are never held
  against you.
- Its units rest in the surveyed state. The few units of the real bay that cannot rest as surveyed in the standard
  simulation (for example a thing that leans on something the survey could not determine) are held kinematically
  (section 8) in the true twin, and so are your units matched to them: they are never lifted, and whether they stay
  at rest and what they rest on is not scored.
- Besides its counted units it carries entries that are *not counted*: small or soft clutter, things the survey saw
  too little of, and things whose footprint centre lies outside the surveyed area. A unit of yours that is matched
  to such an entry is removed from every count: it is neither credit nor charge. The counted units are paired
  first: an entry that is not counted is offered only to those of your units that no counted unit took.

**Alignment.** Your twin's frame is free. The alignment finds the similarity that takes it onto the site,
x_site = s* R x_twin + t, in three steps. Camera i of yours has the centre p_i and the axes R_cv,i (section 3); the
true camera of the same photo has the centre P_i and the axes T_i.

1. **Camera fit (part of G0).** For every three of your valid photo cameras whose true centres span a triangle of
   at least 0.5 m^2, the similarity that takes their centres onto the true
   ones is computed. The one that brings the most of your cameras within 0.15 m of
   their true centres is kept and refitted on those inliers until the set no longer changes. G0 needs at least
   8 inliers, an RMS distance of the inliers of at most
   0.08 m, and a median angle between your camera axes, mapped, and the true ones
   of at most 3 deg. Photo numbers mixed up, a mirrored
   reconstruction, cameras written with the wrong axis convention, or cameras from one standpoint only (no triangle
   to fit) fail here.
2. **Rotation from the orientations.** R becomes the rotation that best takes the axes of your inlier cameras onto
   the true ones; scale and translation are refitted to the inlier centres with that rotation.
3. **Refinement on your building.** The sample rays of C3's depth term (below: rays of the true cameras that end on
   the true building) are mapped into your twin and cast against your environment's collision geoms. Each hit, mapped
   back, has a distance to the true surface point along the true surface normal. A similarity update (scale,
   rotation, translation) that minimises these distances (Huber loss at 3 cm,
   distances above 15 cm ignored, at most
   30 iterations, the rays cast anew each time) is accepted only inside a trust
   region around the map of step 2: a rotation of at most 0.5 deg, a change of
   scale of at most 1 %, a displacement of at most
   0.15 m in the middle of the camera positions. If the update leaves that
   region, or fewer than 30 % of the rays end within
   6 cm of the true surfaces, the map of step 2 is used.

So the twin is placed by its cameras and then settled onto the site by its own building: small camera errors do
not displace a carefully built hall, and the units are judged relative to the building they stand in. s* is the
scale of the final map and up_tilt the angle between your +z, mapped, and the site's vertical.

Geometry is compared on the site, after the map: the units' solids (C1, C2), the building's surfaces (C3's depth)
and the gates' leaves and axes (C5). Whatever the robot or the simulation does runs in your own twin, in its own
units, where the robot's size, the lift height and the tolerances of section 8 are absolute: the stand / crouch /
blocked map, the support under a unit, the settle, the lift probes and the walks. Displacements and gaps measured
there are multiplied by s* before a band is applied. The scale itself is charged once, in C7, but a twin at the
wrong scale also gives the robot the wrong room.

**Matching.** Each of your units is a solid: the union of its collision geoms (a mesh as its convex hull), mapped
onto the site. Its outer surface is sampled about every 1 cm (coarser for very
large units; faces buried inside another piece of the same unit are not surface). The pair measure F of a true unit
T and your unit S is the F-score of the two surfaces at 10 cm: precision = the share
of S's samples within that distance of T's surface, recall = the share of T's samples within that distance of S's
surface, F = 2 P R / (P + R). T's samples that are not determined are left out of the recall, and your samples
whose nearest point of T is not determined are left out of the precision, so a sensible guess of a hidden back
costs nothing. A solid larger than the true one loses precision, a smaller or displaced one both. Units are paired
by Hungarian assignment that maximises F over the pairs with F of at least 0.25; labels play
no part. The counted true units are assigned first; the entries that are not counted are then assigned among your
units that are left, so clutter lying on a counted unit cannot take your model of that unit. N_true is the number of
counted true units; n_sub is the number of units in your `objects.json`
(excluded ones included), less those matched to an entry that is not counted and those outside the surveyed area.

**Areas.** The surveyed area is split into the parts named in `site_brief.md`, and each part counts the same: C1,
C2 and C4 are computed per part and averaged over the parts. A true unit belongs to the part that holds the centre
of its footprint. Your unit belongs to the part of the true unit it is paired with; an unpaired one to the part in
which the centre of its own footprint lies after the alignment, and when that lies outside the surveyed area (or in
a spot of it that the survey did not determine) it is not counted at all. Within a part, N_true and n_sub count
that part's units. So a twin that models one part well and leaves the others out earns that part's share only.

**C1 inventory (0.15).** Per part, 2 TP / (N_true + n_sub). A pair is a true positive when your label is
the true class, or another class that the truth accepts for that unit (where two classes of the list describe it
about equally well), and its shape is recognisable: F_reg of at least 0.5 (F_reg is defined
under C2). A rough box standing in for a unit is matched, but it is not a true positive. C1 is the mean over the
parts.

**C2 pose and shape (0.3).** Per pair, q = b_pose x b_shape x b_vert: all three must hold at once.

- b_pose = band(e_pose, 3, 10 cm). A restricted
  registration moves the true unit's determined surface onto your unit's surface: a turn about the vertical through
  the centre of the true footprint box (the smallest rectangle around the true unit in plan) and a translation in
  three dimensions, at most 30 deg and 0.5
  m. e_pose is the largest horizontal displacement, under that motion, of the four corners of the true footprint
  box (a corner further than 2 m from the box's centre is taken on its ray at
  2 m, so a turn costs a long vehicle no more than it costs a
  4 m unit), taken at the best element of the unit's symmetry: none, a half turn, quarter turns (a square footprint), or
  any turn (a round footprint, where the horizontal translation alone counts). One rule from a sign to a truck: it
  charges the place and the turn together, and a common height offset not at all;
- b_shape is linear in F_reg, 0 at 0.5 and 1 at 0.9. F_reg is the
  F-score of the two surfaces at 3 cm after that registration has been applied, so it
  measures the shape alone (undetermined parts treated as in the matching);
- b_vert = band(e_vert, 2, 6 cm): a unit must rest on
  something. In your own twin, on vertical columns under the unit, e_vert is the gap or the penetration between the
  unit's lowest point and the highest top of any other shape under it (environment, another unit, a gate) that does
  not reach more than 6 cm above that lowest point; with nothing under the unit,
  or the unit sunk deeper than that, b_vert is 0. b_vert follows the support chain: a unit that stands on another
  unit gets at most that unit's b_vert, and the chain must end on the environment, so a stack on a floating unit
  floats too.

Per part, the sum of q / max(N_true, n_sub); C2 is the mean over the parts. q_i is also the weight of unit i in C4
and C5 (0 for an unmatched true unit), so physics credit goes only to units that are where they should be.

**C3 site geometry in the robot's terms (0.15).** C3 = C1 x (0.5 x slice +
0.3 x depth + 0.2 x up x min(1, depth / 0.5)). A level
frame earns its up term only as far as there is a building in it.

- **slice**: where the robot can stand, where it can only crouch and where it is blocked, computed by
  `twin_slice.py` on your twin as submitted (units where you put them, gates at their surveyed position, excluded
  units removed; in your twin's own units) and compared with the same map of the true twin. On
  5 cm cells, with the reference level z_f = the true floor's height at the cell,
  mapped into your twin (the floor only: a dock, a kerb or a raised walkway is an obstacle standing on it):
  - a cell's clearance is the height above z_f of the lowest point, over the cell's square, of the collision
    geometry that reaches above z_f + 0.06 m (what stays below that level is stepped
    over, what is higher than 1 m does not count). It is computed exactly from the
    shapes, not sampled: a thin wall is found in every cell it crosses;
  - a cell is supported when some shape's top over it lies within 0.06 m of z_f (your
    floor is where the true floor is);
  - O_low = the cells with a clearance of at most 0.26 m, or without support; O_high the
    same with 0.33 m;
  - both sets are grown by the robot's half-width plus margin, 0.17 m (a cell joins the
    grown set when the disc of that radius around its centre touches a cell of the set);
  - the cell is **blocked** in grown O_low, else **crouch** in grown O_high, else **stand**.

  A cell is scored when it lies in the surveyed area, at least 2 photos see it and the
  survey determined it, its true class stays the same under the robustness variants, and it is decisive: within
  0.6 m of a boundary between classes in the true map or in yours, or with a true
  clearance below 0.6 m. A class is scored when the true map, by its own
  boundaries, has at least 200 such cells of it. With K scored classes and BA the
  mean over them of the recall (the share of a class's scored cells to which your map gives that class), slice =
  max(0, (BA - 1/K) / (1 - 1/K)): a map of one class (an empty floor, a solid block) scores 0. At this site two
  classes are scored, stand and blocked (too few cells are crouch-only), so K = 2, and a cell of yours that says
  crouch where the true map says stand or blocked counts against that class.
- **depth**: sample rays of the true cameras that end on the true floor, walls, pillars, dock and kerbs (at most
  12 m away, meeting the surface at 10 deg or more,
  not higher than 2.2 m, where the survey confirms the surface) are mapped into your twin and cast
  against your environment's collision geoms. A hit, mapped back, scores band(its distance to the true surface point
  along the true normal, 2, 6 cm); a miss scores 0.
  depth is the mean, over the kinds of surface that have samples, of each kind's mean, so that the floor cannot
  stand in for the walls. At this site the samples lie on the floor, the walls, the pillars and the kerbs (none on
  the dock), and doors are not one of these kinds;
- **up**: band(up_tilt, 0.5, 3 deg).

**C4 static physics (0.14).** From the settle (section 8) of your twin, with your units that are matched
to held true units held: per part, C4 = 0.5 x static share + 0.5 x support
F1, and C4 is the mean over the parts.

- static share of a part = the sum over its true units that rest (counted, not held, static in every robustness
  variant) of q_i x [your matched unit is static, its displacement taken in true units] / max(number of such true
  units, number of your units in the part that are not matched to a true unit left out of this sum);
- support F1 of a part: the support edges (section 8) that the true twin has in every robustness variant, onto a
  counted resting unit from another such unit or from the floor, the dock or a kerb, are required. Your edges are
  mapped onto true names: a unit through its pair; an environment supporter by WHERE your unit rests on it: the
  force-weighted contact point, mapped onto the site, takes the kind of the highest true environment surface within
  8 cm of its height (the floor, the dock, a kerb, another part, or nothing), and the
  forces that map onto the same true supporter are added before the threshold of section 8 is applied (a floor of
  many tiles is one floor). A reproduced edge counts with the weight min(q_a, q_b), an environment part having
  q = 1, and F1 = 2 x the sum of these weights / (number of required edges + number of your edges). An edge belongs
  to the part of the unit it carries. Left out on both sides: edges that involve a held or an uncounted unit or a
  true unit that moves in some variant, and edges that the true twin has in some variants only or onto another kind
  of environment part.

**C5 interaction (0.06).** At this site C5 = C1 x gate; in general C5 = C1 x (0.4 x lift
+ 0.6 x gate).

- **gate**: each true gate is matched with your gate whose leaf centre, mapped onto the site, lies within
  0.5 m of its own (nearest first, one to one). A matched gate scores band(angle between the
  two slide axes, taken as lines, 3, 12 deg) when the
  travel rule (section 5) holds in the real gate's opening direction, the opening and closing lengths taken in true
  metres and the closing allowance including the real gate's own travel that way, and 0 when it does not. gate = the sum of these scores / max(number of true gates, number of your gates that are
  matched or whose leaf centre lies in the surveyed area).
- **lift**: the lift probe (section 8) of each unit on the truth's probe list is run on your unit paired with it,
  from your settled state, by the same height in true units. A probe whose true outcome set (the other units that
  end up carried or disturbed) is not empty scores q_i x the Jaccard index between that set and yours (a unit counts
  as the same in both when its outcome is the same; your unmatched units that move count against you). A probe whose
  true set is empty only charges false movers: 1 when any other unit of yours ends up carried or disturbed (two
  units merged into one body, a unit resting on its neighbour). lift = max(0, the mean over the non-empty probes -
  1 x the mean charge over the empty ones). The lift term is scored only when the
  true twin has at least 3 non-empty probes that hold under every robustness
  variant. **This site has fewer, so the lift term is not scored here**: C5 is the gate term, and the weights of
  the formula above (C4 0.14, C5 0.06) are the ones that apply. What rests on what is still
  assessed, through C2's support chain and C4's support edges.

**C6 walk predictivity (0.15).** The robot walks every hidden route in your twin under the conditions of
section 9. For a run on a route that is open in the true twin: 1 when your run reaches, else 0. On a blocked route:
0 when your run reaches, else band(|your progress - the true progress|, 0.05,
0.2): the robot must be stopped where the site stops it. A route scores the mean over
its starts. A route that units decide (it passes between, beside or under them, or runs into one) scores 0 unless
your twin has every one of those units: each must be paired, in the matching above, with one of your units that
takes part in the walk (an excluded unit does not). A route's outcome is a statement about a thing, and a twin
without that thing has said nothing about it; how well the unit is modelled is not asked here, only whether the
robot's run ends as on the site. Routes decided by the building, the gates or open floor carry no such condition.
BA = (the mean over the open routes + the mean over the blocked routes) / 2, and C6 = C1 x
max(0, 2 BA - 1): an empty hall, where every route is reached, and a solid block, where none is, both score 0. When
C1 is 0 the walks are not run.

**C7 scale (0.05).** C7 = C1 x band(|s* - 1|, 0.03, 0.12).

**The robustness rule.** Nothing is scored that depends on what the photos cannot tell. A static unit, a support
edge, a lift outcome, a map cell or a route counts only when it is the same in every variant of the true twin: unit
masses x0.5 and x2 (all units, and per-unit mixes of the two), friction
0.4, 0.8 and 1 (nominal 0.6), timestep 1 ms (nominal 2 ms), lift speed x0.5 and x2, unit
poses moved by up to 1 cm and turned by up to 1 deg, support heights
opened or closed by 5 mm, and every unit replaced by one box; for the map of C3, the floor
level moved by 2 cm, all geometry shifted by
4 cm, the robot's radius changed by the same; for the walks, at least
12 runs with starts moved by up to 5 cm and
3 deg, other gait phases, floor friction x0.92 to
x1.08 and units moved by up to 1 cm, in which a
blocked route must also stop the robot within 0.05 of the same progress. A true
lift outcome that is intermediate in any variant is dropped.

**Not assessed:** unit masses (beyond the plausibility rule), friction values, contact softness, a gate's exact
travel (beyond the travel rule), whether wheels turn, doors of vehicles, chains, tape barriers, cables and hoses as
things of their own (they get no collision geometry, section 5), what is stored inside the cages and the rack, the
environment above 2.2 m (the ceiling, lamps, pipes), small unevenness of the floor (the true
twin's floor is one level surface across the surveyed area), textures and colours, visual geoms, and anything the
photos do not show from at least two positions.

**Limits.** Each step of the assessment runs under a time limit and a memory limit that are sized on twins at the
caps of section 7. A step that does not finish on your twin scores 0 for what it feeds: G0 for the gate, the
blow-up check and the camera fit, every criterion for the matching, its own criterion otherwise. The walks are the
exception stated in section 9: they are never cut short into a score. The assessment is deterministic: the same
submission gets the same score on every run (it runs on x86-64-v3 processors, with the floating-point code paths
of its numerical libraries pinned).

## 11. Tools in your home folder

| file | what it does |
|---|---|
| `check_twin.py` | the gate (G0's first part) with every reason and note, a line per photo camera, the gates and the units; on request the blow-up check and the settle, the lift probes, the stand / crouch / blocked map, walks of the robot along your own routes, and outlines or renders through a photo camera. It reads no truth and scores nothing |
| `twin_gate.py` | the gate itself, as a module (`gate(folder)`, `settle_check(result)`, `physics_model(result)`) or a command |
| `twin_sim.py`, `twin_sim_config.json` | the standard simulation, the settle, the support rule, the lift probe and the travel rule (`standardize`, `relax`, `settle`, `lift_probe`, `run_probes`, `gate_rule`) |
| `twin_camera.py` | the camera conventions of section 3 |
| `twin_slice.py` | the stand / crouch / blocked map of section 10 (`map_twin`, `render_png`), as a module or a command |
| `twin_walk.py` | the walk conditions of section 9, the code the assessment applies them with; `check_twin.py --walk` drives it |
| `robot/` | the robot of section 9: `robot_card.md`, `walk_probe.py` (the route runner, as a module or a command), `locomotion/` (the controller) and `robot/model/` (the model and its meshes) |

`check_twin.py` takes the submission folder as its first argument (default `/home/user/submission`; write it
before the options, since `--probes` and `--hold` take a list of names) and these options:

```
python3 check_twin.py                                    # the gate: PASS / FAIL, reasons, notes, cameras, gates, units
python3 check_twin.py --settle                           # + the blow-up check and the standard settle
python3 check_twin.py --probes                           # + every lift probe from the settled state (implies --settle)
python3 check_twin.py --probes unit_a unit_b             # only these units' lift probes
python3 check_twin.py --settle --hold unit_c             # hold a unit still, as the assessment holds some (section 10)
python3 check_twin.py --slice slice.png                  # the stand / crouch / blocked map of your twin
python3 check_twin.py --slice slice.png --slice-floor-z Z --slice-bounds X0 Y0 X1 Y1
python3 check_twin.py --walk routes.json --walk-variants K --procs N
python3 check_twin.py --overlay site_007 overlay.png     # outlines of your collision geoms over the photo
python3 check_twin.py --render site_007 render.png --blend W
python3 check_twin.py --json                             # everything as one JSON document
```

- **The gate** (always): every reason for a FAIL and every note; per photo camera the line of section 3; the column
  depth; per gate its axis, range, surveyed position and leaf width, the travel rule for both directions of the
  axis with the travel left on the opening and on the closing side of each, and the direction that your own twin's
  rail suggests (static geoms within 0.25 m of the height of the leaf's top edge and
  within 0.25 m of its plane: the side on which they reach farther beyond the leaf;
  a hint only); per unit its label, status, mass, density, friction, pieces and extent, with the reasons when it is
  excluded. When your twin has no such rail, or one that reaches equally far both ways, the note says that the
  opening direction cannot be read from your twin's rail; the assessment does not use your rail anyway, so what
  counts are the two lines "travel rule if it opens toward +axis / -axis", of which the one in the real gate's
  opening direction must say OK. These lines have no truth to go by, so they judge the closing side as for a gate
  that the photos show shut. The gate cannot run the camera fit, which needs the true cameras.
- `--settle`: the blow-up check with the units it excludes, then the settle on the twin without them: per unit the
  initial penetration and into what, the displacement in the relaxation and in the dynamic phase, the largest
  displacement and rotation and the static flag; then the support edges, by your own body and geom names. The
  report says UNSTABLE when the state goes non-finite, MuJoCo warns, or a unit moves more than
  300 mm or faster than 3 m/s within the first
  2 s of the dynamic phase (a flag for you; the assessment uses the rules of
  section 8).
- `--probes [UNIT ...]`: for each lifted unit, the other units that end up carried, disturbed or intermediate, and
  the gates' travel rule both ways, with the closing-side travel of each direction. `--hold UNIT ...` holds units
  kinematically in the settle and the probes.
- `--slice OUT.png`: the map as a picture with the area of each class. Without the truth it takes as reference
  level the lowest top of your own static geometry over each cell, which is your floor where your floor runs under
  everything; a cell without static geometry under it, or with only the top of something tall, gets no class.
  `--slice-floor-z Z` sets one level everywhere, `--slice-bounds` the rectangle to map. The assessment uses the true
  floor's height (section 10).
- `--walk ROUTES.json`: the robot walks your routes in your twin under the conditions of section 9. The file is a
  list of routes (or `{"routes": [...]}`), each `{"id": "r1", "waypoints": [[x0, y0], [x1, y1], ...], "modes":
  ["walk", ...]}` in your twin's frame, with one mode (`walk` or `crouch`) per segment and optionally `"start_z"`
  (the floor height at the first point, default 0; it stands in for the true floor height of the start rule) and
  `"start_yaw"` (radians; default along the first segment). `--walk-variants K` walks each route from K starts
  (default: the nominal start only; the assessment uses 4), `--procs N` uses N processes. It
  prints per route how many runs reached and the mean progress, and for every run that did not reach how it ended,
  its progress and which of your geoms the robot's body touched. A run that takes longer than the assessment's own
  generous wall-clock guard (many times the walk's own duration) stops with a time error instead of inventing an
  outcome, as in the assessment: run it again when CPUs are free.
- `--overlay site_NNN OUT.png`: your collision geoms seen through `photo_NNN` (MuJoCo's CPU renderer), drawn as
  outlines over the photo, at most 1552 px wide: units tinted and named, gates orange, every
  environment geom outlined in white; it lists the units that are not in this view. `--render site_NNN OUT.png`
  renders your twin with its own materials through that camera; `--blend W` mixes the photo in (W between 0 and 1);
  `--max-width PX` makes either picture smaller.
- `--photos DIR`, `--robot DIR`, `--vocab FILE`: other places for the photos, the robot folder and the class list
  (default: beside `check_twin.py`). `--json` prints one JSON document instead of the text.
- Exit code 0 when the gate passes (and, with `--settle` or `--probes`, the blow-up check passes), 1 otherwise, 2
  for a usage error.

## 12. Practical notes

What the authors of the gate and the assessment saw on test twins, and what follows from the rules above. None
of it is a further rule.

- **A mesh collides as its convex hull.** Most units are a few boxes; a shape that is not convex is several convex
  pieces (CoACD is installed). A U-shaped or L-shaped mesh in one geom has its hollow filled. A room, a rack or a
  cage exported as ONE mesh is a solid block with the bay inside it: the gate's hull note names such a geom, the
  settle finds the units inside a solid, and the robot finds no floor to start on.
- **Thin things are boxes.** A collision mesh needs a volume: a flat sheet of triangles has no convex hull and
  fails the compile. A board, a sheet, a wall panel or a mesh leaf is a thin box.
- **Every geom collides and has mass unless you say otherwise** (MuJoCo's defaults). A geom that is only for looks
  needs `contype="0" conaffinity="0"`, and inside a unit also `density="0"`, or the unit is excluded.
- **Give every unit an explicit mass** (on its geoms, or in an `<inertial>`). Any mass that puts the density
  inside the range of section 4 passes, and its true value is not assessed. With MuJoCo's default density,
  pieces that overlap count their overlap twice, which can push a unit of many overlapping pieces out of the range.
- **Rest every unit on what it stands on**: touching it, neither sunk nor hovering. The relaxation pushes a unit
  out of a floor or a neighbour that it is sunk into, and closes a small gap, without launching it; but what it
  moves counts as motion, a gap or a sink beyond the bands of b_vert costs C2, and a unit left hovering falls in the
  settle. A deep overlap between two units, or a unit inside the environment's solid, can be released so fast that
  the unit is excluded. `--settle` shows all of this per unit.
- **Level and scale the twin, cameras included.** A reconstruction's frame is arbitrary. Turn and scale the whole
  twin together with its cameras until +z is the bay's vertical and lengths are metres; then write the cameras from
  the same transformed poses. With the photos' world-to-camera pose x_cam = R x_world + t (what a reconstruction
  gives), the camera centre is p = -R^T t and R_cv = R^T; `twin_camera.camera_xml` takes these. In the camera
  lines of `check_twin.py`, a camera with none of your twin in its image or with its image-up pointing down has the
  wrong convention.
- **The alignment uses where your cameras are and how they are turned.** The intrinsics you declare must be valid
  (section 3) and are used by `--overlay` and `--render`; they do not enter the placement.
- **Size is absolute for the robot.** A passage, an underside or a start that is a few centimetres off can change a
  walk; the robot card says at which widths and heights its outcome turns. Use `--slice` and `--walk` on your own
  twin to see what the robot meets, and compare with what the photos show.
- **Planes are infinite.** A floor plane also lies under everything outside the bay, and a plane used as a wall
  cuts the world in two. Boxes are usually the better choice.
- **Few, simple geoms.** One slab is a floor; thousands of tiles or layers buy nothing, slow every check down, and
  stacked over one spot they break the column-depth cap.
- **Threads.** The machine reports more processors than you may use. Give every tool an explicit thread count of at
  most 8 (pycolmap's `num_threads`, the dense tools' `--max-threads`, `OMP_NUM_THREADS`), or it starts
  dozens of threads and runs slower. `check_twin.py --walk` uses as many processes as `--procs` says.
- **Cost of the checks.** The gate takes seconds, also on a twin at the caps; the settle and a lift probe take
  seconds on an ordinary twin; a walk takes about as long as the walk itself lasts, per process, so a batch of
  routes belongs in the background with its output in a file.
- **A unit that matches nothing counts as submitted** in the part where it stands (unless its footprint centre lies
  outside the surveyed area): model what the unit rule names, and leave doubtful clutter out rather than in.

A skeleton of the structure (numbers left out):

```xml
<mujoco model="bay">
  <compiler angle="radian" meshdir="assets"/>
  <asset>
    <mesh name="unit_b_piece_a" file="unit_b_piece_a.stl"/>
  </asset>
  <worldbody>
    <!-- environment: static geoms (floor, walls, pillars, dock, kerbs, cage walls, the gates' rails, ...) -->
    <geom name="floor" type="box" size="..." pos="..."/>
    <geom name="wall_a" type="box" size="..." pos="..." quat="..."/>
    <geom name="gate_a_rail" type="box" size="..." pos="..."/>
    <!-- one camera per registered photo (twin_camera.camera_xml writes the element) -->
    <camera name="photo_007" pos="..." quat="..." resolution="6208 4135"
            sensorsize="..." focalpixel="..." principalpixel="..."/>
    <!-- a unit: a top-level body with a free joint and convex collision pieces -->
    <body name="unit_a" pos="..." quat="...">
      <freejoint/>
      <geom type="box" size="..." mass="..."/>
    </body>
    <body name="unit_b" pos="...">
      <freejoint/>
      <geom type="mesh" mesh="unit_b_piece_a" mass="..."/>
      <geom type="box" size="..." pos="..." mass="..."/>
    </body>
    <!-- a gate: one body, one limited slide joint along its rail, zero at the surveyed position -->
    <body name="gate_a" pos="...">
      <joint name="gate_a_slide" type="slide" axis="..." range="... ..."/>
      <geom type="box" size="..."/>
    </body>
  </worldbody>
</mujoco>
```

```json
{"format": "site-twin-objects/1",
 "items": [{"body": "unit_a", "label": "<class name>"}, {"body": "unit_b", "label": "<class name>"}],
 "gates": [{"body": "gate_a", "joint": "gate_a_slide"}]}
```
