# Twin contract

This is the technical contract of the twin: what you deliver, the rules the files must follow, the standard
simulation and the probes the twin is run through, and how it is assessed. Every number below is the number the
code uses (`twin_gate.py`, `twin_sim.py` with `twin_sim_config.json`, `twin_camera.py`, and the grader's
configuration). `site_brief.md` explains what the twin is for.

## 1. What you deliver

`/home/user/submission/` holds:

| path | content |
|---|---|
| `scene.xml` | the twin: one MuJoCo MJCF file (MuJoCo 3.16), no includes |
| `objects.json` | which bodies are the items, with their product labels, and which body is the drawer (section 6) |
| `assets/` (any layout) | the mesh files and textures that `scene.xml` references, by relative path |

- Only `scene.xml`, `objects.json` and the files `scene.xml` references are read. Every other file is ignored, but
  it still counts toward the folder caps, so keep renders, notes, caches and scratch work outside `submission/`.
- Folder caps: at most 5000 entries (files and folders) and 300 MB in total.
- File caps: `scene.xml` at most 5 MB, `objects.json` at most 1 MB, each mesh
  file at most 50 MB, each PNG at most 64 MB and 4096 px a side;
  at most 2000 referenced files, 2000 meshes and
  500 textures; at most 20000 XML elements, nested at most
  64 deep.
- Formats: meshes `.stl` (binary), `.obj` or `.msh`; textures `.png` only. A referenced path is relative (to the
  submission folder, or through `meshdir` / `texturedir` / `assetdir`), has no `..`, no symbolic link on the way,
  and only the characters `A-Z a-z 0-9 _ . + - ( ) /` and spaces. An OBJ's `mtllib` must be a plain file name.

## 2. Frames and units

- World: z up, metres, kilograms, seconds. Gravity points along -z (the standard simulation sets it, section 8).
- The initial state `qpos0` is the photographed state: every body as written in `scene.xml`, the drawer at the
  opening it has in the photo. There are no keyframes and no joint `ref`.
- Where the world origin is and which way x points are yours to choose: the twin is assessed in the frame of its own
  `photo_cam` (section 9).

## 3. The camera `photo_cam`

`scene.xml` must contain a camera named `photo_cam`:

- `mode="fixed"`, on the world body or on a static body (one without joints);
- `resolution="1920 1080"`, the photo's size;
- intrinsics in one of MuJoCo's two forms:
  - **sensor form** (any focal lengths and principal point): `sensorsize` (any positive size), `focalpixel="fx fy"`
    and `principalpixel` (or the length forms `focal` / `principal`; the pixel forms win when both are given);
  - **fovy form**: `fovy` only, which means fx = fy and a centred principal point;
- fx and fy between 500 and 5000 px, and the principal point inside the image.

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
model, and `camera_frame`, `project`, `backproject` and `pixel_rays` do the geometry. `python3 check_twin.py`
prints the decoded fx, fy, cx, cy, the camera pose and the world's up direction in camera coordinates.

Your `photo_cam` is your estimate of where the photo was taken from, relative to your twin. The assessment puts your
twin into the true camera frame through it (section 9), so a twin that lines up with the photo through `photo_cam`
lines up with the truth.

## 4. Items

Every loose grocery toy is an item, and every item is:

- a top-level body (a child of `<worldbody>`, directly or through `<frame>` elements) with a `<freejoint/>` (or a
  joint of type free), and no other joint; it may have child bodies without joints (they move with it);
- listed in `objects.json` with its product label from `product_names.json`;
- given collision geometry: between 1 and 64 collision geoms. A collision geom is
  any geom whose `contype` or `conaffinity` is not 0. Use primitives (box, sphere, capsule, cylinder, ellipsoid) or
  meshes; **a mesh geom collides as the convex hull of its vertices**, so model a concave shape as several convex
  pieces. Planes cannot be part of an item. A collision mesh has at most 20000 vertices,
  and an item's collision meshes together at most 50000;
- optionally given visual geoms: `contype="0" conaffinity="0"` and no mass (`density="0"` or `mass="0"`). They are
  drawn by renders and play no part anywhere in the assessment;
- physically plausible:
  - density (the item's mass divided by the volume of the union of its collision geoms) between
    10 and 3000 kg/m^3;
  - sliding friction of every collision geom between 0.1 and 1.5 (the standard
    simulation then uses its own value, section 8);
  - inertia consistent with the shape: the smallest principal moment at least 0.0001 x mass x L^2
    (L the item's largest extent), the largest at most mass x d_max^2 (d_max the largest distance from the centre of
    mass to the collision geometry), with a 1 % tolerance;
  - the centre of mass inside the convex hull of the collision geometry; the item no larger than
    2 m.

An item that breaks one of the plausibility rules, or its contact-bit or geom-count rules, is **excluded**: it is
removed from the twin before any simulation (it can neither support nor disturb anything) and scores 0 in every
criterion, while it still counts as submitted. The rest of the twin is assessed as usual. `check_twin.py` lists
each item's status with the reasons.

A label that is not in `product_names.json` is kept, but never counts as correct.

## 5. The environment and the drawer

- The furniture, the wall and the floor are geoms of static bodies: the world body, or bodies without joints. Their
  collision geoms (contype or conaffinity not 0) are what the environment criterion looks at; visual geoms are
  ignored. Planes are allowed in static bodies only.
- The drawer is **one body with exactly one slide joint** (`<joint type="slide" axis="..."/>`), hanging from a
  static body. Bodies inside it carry no joints and belong to the drawer. There is no other joint in the twin than
  the drawer's slide and the items' free joints.
- The slide's axis is the drawer's direction of travel. "Out" is the direction along the axis that moves the drawer
  toward `photo_cam`, measured from the centre of the drawer's collision geometry (the drawer body's origin may sit
  anywhere); `check_twin.py` prints it.
- **Travel.** The slide's limits (`range`, limited) are the real drawer's travel, and they must be plausible for a
  drawer of its size: the photographed opening (`qpos0`) lies inside the range; the total travel (the range's
  length) is between 0.5 and 1 times the
  drawer's depth (the extent of the drawer's collision geoms along its slide, at `qpos0`); and the drawer can come
  out at least 50 mm beyond the photographed opening. A drawer that breaks this rule
  scores 0 in C5's drawer term, without failing the twin; `check_twin.py` prints the check. The standard simulation
  drives the drawer itself and does not use the limits.
- In the standard simulation the drawer never collides with the furniture; items collide with everything.

## 6. `objects.json`

```json
{"format": "twin-objects/1",
 "items": [{"body": "<body name>", "label": "<product name>"}, ...],
 "drawer": "<body name>"}
```

- Every top-level body with a free joint must be listed, and every listed body must be one; a body appears once.
- Between 1 and 60 items. Other keys are ignored (with a note).

## 7. Allowed MJCF

The gate reads `scene.xml` as XML before MuJoCo sees it. Only these elements, each under the parents and with the
attributes listed, are accepted; anything else is rejected with a reason (no DOCTYPE or entities, no namespaces).

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
others joint springs, damping, armature and friction loss, `mocap`, `gravcomp`, `sleep`, contact overrides, fluid
and wind settings, `maxhullvert`, `strippath`, cube-face texture files and compiler outputs.

Then the gate compiles a copy of the twin with textures removed: **any MuJoCo compile warning fails the twin**.
On the compiled model it checks: no equality, tendon, actuator, sensor, flex, skin, plugin, mocap body, contact pair
or exclude, keyframe, height field or custom data; at most 120 bodies, 3000 geoms,
1500 collision geoms, and 3 million mesh faces and 3 million mesh
vertices in total; finite numbers, no
body further than 100 m from its parent and no geom size above 100 m;
no joint stiffness, damping, armature, friction loss or margin, no gravity compensation, adhesion, surface velocity or
fluid forces; the camera (section 3), the items (section 4) and the drawer (section 5).

## 8. The standard simulation

Every twin is simulated the same way, by `twin_sim.py` with `twin_sim_config.json` (which you have). Whatever
`<option>`, contact parameters, friction or contact bits the twin sets, the simulation first replaces them:

- `mjOption` is reset to MuJoCo's defaults, then: timestep 0.002 s, integrator
  `implicitfast`, `elliptic` friction cone, impratio 10, solver
  `Newton` with 100 iterations (50 line-search iterations) and
  tolerance 1e-10, gravity (0, 0, -9.81) m/s^2, autoreset off;
- every collision geom: condim 3, solref (0.005, 1), solimp (0.95, 0.99, 0.001, 0.5, 2), margin and gap
  0, friction (0.6, 0.005, 0.0001) (sliding, torsional, rolling) whatever the twin says;
- contact bits: items collide with items, furniture and the drawer; the drawer does not collide with the
  furniture;
- every force that does not come from a contact is removed (joint springs and damping, friction loss, armature,
  gravity compensation, surface velocities, adhesion, fluid forces);
- driven bodies are kinematic: their pose and velocity are written before every step and a large armature
  (10000) keeps contacts from pushing them. The drawer is held at its `qpos0` opening
  except while a probe drives it. Masses and inertias are kept as delivered.

**The settle.** 3 s from `qpos0`. An item's motion is measured at the centroid of its collision
geometry (geom centres weighted by the volume of their bounding boxes), its rotation as the angle of its rotation
since the start; both are sampled at every step. An item is **static** when its displacement stays within
3 mm and its rotation within 2 deg over the whole settle.

**Support.** An item b is supported by a (another item, a furniture geom, or the drawer) when the contacts between
them push b upward: contact normals within 60 deg of vertical (up-component at least
0.5), and a summed upward force averaged over the last 0.5 s of the
settle of at least 0.1 of b's weight.

**Blow-up check (part of G0).** The standard simulation runs for 2 s from `qpos0` (the drawer held, every item free). G0 fails only on a numerical blow-up: a non-finite state or MuJoCo reporting bad accelerations; a gain of mechanical energy, KE(t) > PE(0) - PE(t) + e with e = 0.5 x (total item mass) x (2 m/s)^2 (what releasing a deep interpenetration produces); or an item whose centre of mass moves faster than the free-fall bound g t + 2 m/s. Ordinary falls, slides and topples are not G0 failures: C4 and C5 charge them.

**The probes**, each from the settled state:

- **Lift probe** of item i: i is lifted 40 mm straight up (world +z) along a minimum-jerk profile
  over 0.8 s, then held for 0.7 s, without turning. Every other item's outcome is
  read at the end, from its displacement d and rotation since the probe started.
- **Drawer probe**: the drawer is pulled out 50 mm along its slide (toward `photo_cam`),
  minimum-jerk over 1 s, then held 0.7 s. Only the pull-out is part of the
  protocol.
- Outcomes of an item:
  - **carried** (lift) / **rides** (drawer): it follows the driven body: |d - d_driven| within
    15 mm (lift) or 10 mm (drawer), rotation below
    10 deg, and for a lift a rise of at least 24 mm;
  - **stayed**: |d| below 5 mm and rotation below 5 deg;
  - **disturbed**: not following, and |d| at least 10 mm or rotation at least
    10 deg;
  - **intermediate**: anything else (it matches no outcome).

`python3 check_twin.py --settle` runs the settle on your twin and `--probes` the probes, with the same code. Lengths
there are in your twin's units; the assessment converts your twin to true units first (section 9).

## 9. How the twin is assessed

```
twin_score = G1 x G0 x (0.15 C1 + 0.25 C2 + 0.1 C3 + 0.2 C4 + 0.25 C5 + 0.05 C6)
```

Each criterion is in [0, 1]. A **band(e, full, zero)** gives 1 for an error e at or below `full`, 0 at or above
`zero`, and is linear between.

**G1, submission integrity.** The folder is the one your session wrote: nothing in it may be written after the
session ends, it must be a real folder (not a symbolic link, not replaced or renamed into place afterwards), and it
must stay within the folder caps (section 1). Processes left running are stopped when the assessment starts.
Otherwise G1 = 0.

**G0, validity.** The gate of sections 1-7 (`twin_gate.py`, the same code `check_twin.py` runs) passes, and the
settle does not blow up (section 8). Otherwise G0 = 0. An excluded item does not fail G0 (section 4).

**The true twin.** The assessment compares your twin with a true twin of the photographed scene, built to this same
contract and run through the same simulation. Its items rest in the photographed state; the few items of the real
scene that cannot rest exactly as photographed in the standard simulation are held kinematically (section 8) in the
true twin, and so are your items matched to them; they are never lifted, and outcomes and support edges involving
them are not scored. A held item that rests in the drawer moves with the drawer in the drawer probe.

**Alignment.** Your twin is mapped into the true camera frame, in three steps:

1. into your own camera frame, x_cam = R_cv^T (x_world - p), from your `photo_cam`;
2. R_pp: the rotation that takes the optical axis onto the true ray through the pixel you declared as the principal
   point (cx, cy), clamped at 2 deg. So a principal point that is off does not cost
   you a rotation of the whole scene, while your focal lengths still set where things project;
3. one global scale s* about the camera centre, fitted on matched items: s0 = the median ratio of true to submitted
   item distances from the camera over pairs whose bearings agree within 3 deg; then,
   repeatedly, match on the capped IoU at s, drop pairs whose centroids are more than 30 mm
   apart, and refit s by least squares, until the pairs no longer change (at most
   4 refits). With fewer than 3 pairs, s* = s0 (1 without a
   bearing pair).

Everything below is measured in the true camera frame, with your twin scaled by s*.

**Matching.** Each of your items is a solid: the union of its collision geoms (a mesh as its convex hull). The
**capped IoU** of a true item T and your item S is their 3-D IoU on a grid of 3 mm cells
(1 mm for true items thinner than 40 mm), times a cap that charges inflation:
the product over three axes of min(1, true extent / your extent), with the axes those of your solid after its
rotation fit (C2), so a solid larger than the truth never buys overlap, while a rotation error is not charged as
inflation. Items are paired by one Hungarian assignment on the capped IoU, labels playing no part; pairs below
0.25 are dropped. N_true is the number of true items, n_sub the number of items in your
`objects.json` (excluded ones included).

**Work areas.** The job covers three places: the console top, the open drawer, and the shelves under the top. Each
counts the same: C1, C2, C4 and the lift term of C5 are computed per work area and averaged over the three areas. A
true item belongs to the area its support chain ends in (it rests there, directly or on other items). Your item
belongs to the area of the true item it is paired with; an unpaired item (excluded ones included) belongs to the
area of the true item whose centroid is nearest to its own (your solid's volume centroid, or the body's centre of
mass for an item without a solid). Within an area, N_true and n_sub count that area's true and submitted items, and
the sums run over that area. So a twin that models one place well and leaves the others out scores about a third of
what it would score for the whole scene.

**C1 inventory (0.15).** Per area, 2 TP / (N_true + n_sub), TP the pairs whose label equals the
true label; C1 is the mean over the areas.

**C2 pose and shape (0.25).** Per pair, q = b_pos x b_rot x b_iou: all three must hold at once,
as a grasp needs the place, the turn and the shape (an item right in two of them and far off in the third earns
nothing):

- b_pos = band(distance between the centroids of the two solids, 4, 15 mm);
- b_rot = band(rotation angle, 3, 15 deg). The angle is that of a
  point-to-plane ICP rotation of your solid's surface onto the true surface about the common centroid, from the
  identity and limited to 30 deg, with a uniform scale fitted alongside so that a size error is
  not read as a rotation. Turns about an axis the true shape is nearly symmetric about are not charged (an axis
  counts as symmetric when the true solid turned 30 deg about it keeps an IoU of at least
  0.9 with itself, as a can about its axis);
- b_iou = linear in the capped IoU, 0 at 0.4 and 1 at 0.85.

Per area, sum of q / max(N_true, n_sub); C2 is the mean over the areas. q_i is also the weight of item i in C4 and C5
(0 for an unmatched true item), so physics credit goes only to items that are where they should be.

**C3 environment (0.1).** C3 = C1 x (0.75 x depth share +
0.25 x vertical). The environment counts as far as the inventory on it: furniture without the
items earns nothing.

- Depth share: through the true camera, a ray per pixel (on a pixel grid with a step of 2) is cast into
  your aligned twin against your environment's collision geoms (everything that is not an item: furniture, wall,
  floor, drawer, at `qpos0`; a mesh as its convex hull). A pixel counts when the hit lies within
  15 mm of the true surface (distance along the true surface normal); a miss does not count. Only
  pixels where the true environment is reliably known are used: the console, the drawer, the wall and the floor
  seen in the photo, away from the items. The depth is taken at the better of s* and your own scale (1). The share
  is the mean over the environment parts the reference distinguishes (the top, its front edge, the wall and window,
  the drawer's floor, the drawer's walls and front, the shelves, the floor), each part's share of its own pixels,
  so that a large surface cannot stand in for the others.
- Vertical: band(angle between your world's +z, mapped into the true camera frame, and the true up,
  2, 8 deg).

**C4 static physics (0.2).** C4 = 0.5 x static share +
0.5 x support F1, both from the settle (section 8) of your twin, with the static rule applied in
true units, each taken per area and averaged over the areas:

- static share of an area = sum over its true items that rest (not held) of q_i x [your matched item is static] /
  max(number of such true items, number of your items in the area not matched to a held true item);
- support F1 of an area: the support edges (section 8) that are present in every robustness variant of the true twin
  (see below) are required; each counts with the weight min(q_a, q_b), furniture and the drawer having q = 1. Your
  edges are mapped onto true items through the pairs, and furniture supporters onto the true surface they touch. An
  edge belongs to the area of the item it carries. Precision, recall and F1 are taken over these weights.

**C5 interaction (0.25).** C5 = 0.7 x lift + 0.3 x drawer,
from the probes (section 8) of both twins, each from its own settled state, outcomes read in true units.

- Lift: the lift probe of true item i is matched with the lift probe of your item paired with i. A probe whose true
  outcome set (the other items that end up carried or disturbed) is not empty is scored by the Jaccard index between
  that set and yours (an item counts as the same in both when its outcome class is equal; your unmatched items that
  move count against you), times q_i; lift is the mean over these probes. Probes whose true outcome set is empty
  only charge false movers: each of your items that ends up carried or disturbed in such a probe, while its true counterpart stays, is a false mover, and the lift term loses 1 x (false movers / items that could have moved) per such probe, averaged over these probes. The lift term is taken per area (over the probes of the area's
  true items), floored at 0, and averaged over the areas.
- Drawer: the Jaccard index of the sets of items that ride along on the pull-out, times
  band(angle between your slide axis and the true one, 4, 15 deg), times
  1 if the drawer's travel rule (section 5) holds, else 0.

**The ambiguity rule.** An outcome or a support edge is scored only when it is the same in every robustness variant of
the true twin: item masses x0.5 and x2 (all items, and random per-item mixes of the two),
friction 0.4 and 0.8 (nominal 0.6), timestep 1 ms (nominal 2 ms), probe speed x0.5 and x2,
item poses jittered by up to ±2 mm per axis and ±1.5 deg, support surfaces moved
by ±3 mm, and item shapes replaced by fitted primitives. A true outcome that is
intermediate in any variant is dropped. So nothing that depends on unknowable details (exact masses, friction,
contact softness) is scored.

**C6 scale (0.05).** C6 = C1 x band(|s* - 1|, 0.1, 0.3) when at
least 5 pairs are matched, else 0.

**Not scored:** item masses (beyond the plausibility rule), friction values, contact softness, the drawer's exact
travel (beyond the travel rule), textures and colours, visual geoms, and anything the photo cannot show.

## 10. Tools in your home folder

| file | what it does |
|---|---|
| `check_twin.py` | the gate (G0) with every reason, the decoded `photo_cam`, per-item mass, density, friction and pieces; `--settle`, `--probes`, `--overlay out.png` (segmentation outlines through `photo_cam` on the photo, MuJoCo on the CPU), `--json`. It reads no truth and scores nothing |
| `twin_gate.py` | the gate itself, as a module (`gate(folder)`) or a command |
| `twin_sim.py`, `twin_sim_config.json` | the standard simulation, the settle, the support rule and the probes (`standardize`, `settle`, `lift_probe`, `drawer_probe`, `run_probes`) |
| `twin_camera.py` | the camera conventions of section 3 |
| `render_twin.py`, `ovrtx_render/` | path-traced renders of a scene through `photo_cam` with OVRTX on the GPU, optionally blended with the photo (`ovrtx_render/README.md`) |
