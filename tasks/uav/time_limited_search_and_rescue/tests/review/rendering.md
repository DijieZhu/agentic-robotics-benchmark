# The cameras' renderer and what it shows

Since 2026-11-22 the site holds the furnished building of the task's approved pictures, and the aircraft's two
cameras are drawn by MuJoCo 3.16's Filament renderer on Mesa's lavapipe (Vulkan on the CPU). This note records what
is in the scene, what a picture shows, what was measured, and how to repeat the measurements.

## The scene (`tests/runtime/`, byte copies in `environment/setup/runtime/`)

- `rescue_scene.py`: the structure, the markings, the fixtures, as before.
- `rescue_dressing.py`: what stands in a site besides its structure, from the site alone: cartons, drawer chests,
  tables, chairs, barrels, buckets along door-less walls of every room, stock in the staging row, a gurney and
  furniture in the receiving room on the open station's side, posters and wall cabinets, ceiling lamps, paper on
  the floors, the yard. The rule (stated in the instruction): nothing within 0.72 m of a lane, nothing taller than
  0.80 m within 1.25 m of a turning point. Lanes: the lines joining two rooms' centres through their doorway, the
  staging row's centre line, the ways to the inlet, the outlet, the pocket, both receiving sides, the tool stands
  and the launch point, the litter's length and the way from it to its room's centre; wells are points on both
  storeys. Turning points: room centres, the pocket's centre, both receiving pads.
- Every standing thing is one box in the contact check (obstacle kind `prop`) and its model's meshes in the
  cameras. The models are the render bundle's library (`look/library.json`, one OBJ set and its pictures per model).
- The litter: a basket litter with a clothed casualty under a blanket and a lifting receiver. The receiver's ring is
  the site's flange (radius 0.041 or 0.093 m, the task's height and colour), closed at the bottom, on a post from
  the bed; two hoops outside the collet's reach carry a guide ring with a green band. No part is in the collet's way
  at any yaw. The flange can be seen from above through the guide ring (bore 0.28 m): in the down camera it is about
  22 against 10 pixels across from 1.0 m above the litter and 15 against 6.5 from 1.5 m.
- Contact pieces: the litter's, the casualty's and the aircraft's contact pieces are the task's former geoms of
  those bodies (board, shell, mannequin boxes, fitting; the airframe meshes, mast, jaws). They are not drawn; the
  drawn models lie inside them and carry no mass and no contact (`look_*` geoms). The collet meets none of them:
  the latch is the task's rule on position, speed and the flange's radius, as before. A hidden cylinder `flange`
  keeps that rule's radius.
- `rescue_marks.py`: the paints. The model's signal colours are drawn as paint (toned: saturation and value
  reduced, then greyed by 0.22 and dusted); a door apron's wear pattern follows from the site's seed and the colour.

## What a picture shows (`rescue_render.py`, `rescue_look.json`)

- Every drawn geom of the model where the simulation has it. Walls, floors, ceilings, skirting and door frames carry
  the approved maps; doors and windows have timber and metal frames by the renderer's rule (a frame's boards, a head
  at 2.10 m, wall above it; a window's opening with frame and bars in the wall's thickness).
- Windows of the outer hull, the receiving room's glazed door and window: openings in the drawn walls through which
  the yard is seen; the walls stay solid in the contact check. No glass is drawn.
- Light: one image light (the same in every compass direction), ambient occlusion, and daylight computed at the
  site's build: for a grid of points (1 cm) on floors, floor markings and walls a ray towards the sun (travel
  azimuth -160 degrees, elevation 25 degrees) that leaves through a hull window without meeting a wall, a slab, a
  panel or a standing thing's box; sunlit rectangles are drawn 0.6 mm over the surface with the surface's own
  picture and a glowing copy. The yard is drawn at daylight exposure. Panels inside a hoist well are darker by a
  fixed factor. No lamp lights anything; no shadow map.
- Not in the aircraft's own cameras: its rotors' blur (the cameras sit clear of it). Not in the scene: motes in
  the air, people.

## A picture is a function of the site and the state

- One worker process draws one site; every Filament object is made once, in one order; temporal dithering is off;
  images are drawn at twice their size and averaged in linear light with integer arithmetic.
- Everything is drawn with one material (one atlas), and everything that stands still is one mesh and one draw
  call: the order of two draw calls changes from process to process, and with it which of two surfaces a sample on
  their common edge shows. A body that moves is one mesh, moved as a whole. No mesh is uploaded after the build.
- Coplanar faces that overlap are separated by 0.37 mm per level (the face alone).
- The daylight grid and the apron pictures use elementwise arithmetic and table lookups only.
- `tests/review/camera_digests.py` repeats the check (made-up states over every room cell on four headings; in
  order, in reverse, in fresh processes; `--dump` and `--expect` compare two hosts).

Measured on the final runtime (2026-11-22), task image, 2 CPUs / 4 GB, two machines of different CPU families
(AMD Zen 3 and Zen 5): development sites 8009 and 8018, 164 states each, both cameras (656 images). On each machine:
0 images differ when drawn again in reverse order, 0 in three (two) fresh processes. Between the machines: 0 of 656
differ. The privileged pilot's flights on the eight graded sites through the grader repeat the earlier action
counts (443, 478, 509, 484, 508, 503, 541, 487).

## What the pictures cost

Task image, 2 CPUs / 4 GB, on a machine that was busy with other work (load 22 on 32 cores):

| | median | 90th percentile |
|---|---|---|
| the two pictures of a state | 0.234 s | 0.262 s |
| an answer to an action (physics, pictures, PNG) | 0.247 s | 0.270 s |

The former renderer drew a pair in 0.057 s. A site's first answer takes about 10 s (the model with its 106 meshes,
the worker, the daylight grid). The worker process holds 1.2 GB, the service beside it about 0.4 GB. A flight of
4,400 actions needs about 1,100 s of the environment's time; the site's wall limit is 2,100 s and the verifier's
timeout 8 x 2,100 + 1,200 s. The privileged pilot's eight flights through the grader took 1,345 s in all.

## Information parity

Per group of geoms (by the former scene's segmentation of the same states of a recorded development flight, 144
states and three made-up ones in the turning pocket) the mean colour in the former pictures and in the new ones;
per pair that a pipeline has to tell apart the CIELAB distance. The new pictures count everything that now hides a
group (furniture, the casualty, the yard seen through windows). The signal colours are paint now, so the distances
are smaller than between display colours; every pair stays apart:

| pair | former | new |
|---|---|---|
| purple apron / lower floor | 108 | 61 |
| blue apron / lower floor | 91 | 54 |
| amber apron / lower floor | 90 | 35 |
| amber apron / upper floor | 97 | 25 |
| white apron / lower floor | 55 | 38 |
| white apron / upper floor | 54 | 21 |
| sign / lower floor | 17 | 18 |
| sign / upper floor | 14 | 32 |
| bay rail / upper floor | 92 | 33 |
| bay rail / toe line | 48 | 24 |
| toe line / lower floor | 108 | 49 |
| receiving mat / lower floor | 90 | 43 |
| pad / receiving mat | 153 | 63 |
| amber wall / wall | 74 | 32 |
| cargo door / wall | 82 | 59 |
| cargo wall band / wall | 83 | 52 |
| throat panel / wall | 23 | 17 |
| shutter / wall | 68 | 46 |
| shutter / well frame | 19 | 17 |
| arrow / its housing | 87 | 69 |
| arrow housing / wall | 54 | 50 |
| well frame / upper floor | 61 | 37 |
| skirting / lower floor | 24 | 15 |

Between backgrounds: wall / lower floor 21 (21), wall / upper floor 10 (18), wall / ceiling 7 (39). A trial with
the rotors' blur drawn opaque in the forward camera hid the band over the cargo door (52 fell to 4) and was dropped:
the blur is in neither camera.
