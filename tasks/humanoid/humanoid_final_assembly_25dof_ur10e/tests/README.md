# humanoid_final_assembly_25dof_ur10e: grading notes for maintainers

Grader-side; never staged into the agent's home.

## What is here

- `test.sh` builds the root-owned copy of the given files and runs `verify.py` as root.
- `verify.py`: the lockdown at verify start (the agent account's processes stopped, the submission snapshotted), the
  check of the given files, the hidden kit, the network guard, the contract gate, the token of the reference, the
  episodes on layouts chosen anew at random, the score.
- `grader_config.json`: the grader's keys. The cell's configuration is `arm/assembly_humanoid/cell_config.json`; the grader's overrides are
  applied on top of it.
- `arm/`: the cell's code, with the files it uses of the packages vendored from the benchmark's earlier assembly
  cells, byte for byte; the files the grader never loads are left out, and the builder proves on every build that the
  grader reaches the same files as with them.
- `render_seam/twin_worker.py`: the render worker of the cell's cameras. It renders the GIVEN render twin through the GIVEN
  render layer with a hidden body map, so that the agent's layer and the grader's frames are the same pictures.
- `hidden/oracle_token.sha256`: the digest of the reference program's token.
- The hidden kit (the collision pieces, the picture meshes, the mating graph, the fixtures, the frozen numbers) is not
  in the repository: the grader unpacks it at verify start from the asset dataset, pinned by digest and file count.

## The score

3 episodes, each on a layout chosen at random from operating-system entropy; the reference runs on the nominal
layout. Per episode: the weight of the mates fastened at their design bolt position, their modules not scrapped,
over 7.0 (a leg 1.5, an arm 1.5, the head 1.0; all five mates count), kept
at 30 % with a counted safety violation. The reward is the mean over the episodes times the contract
gate. An untouched sandbox scores zero; the reference scores full.

Safety numbers: arm contact 150 N (any link of the arm or of its gripper; the gripper's pads are
exempt on a module only); part force 150 N for
20 ticks; part speed 3 m/s; floor contact scraps the module. Each number is
the larger of its proposal and, up to its cap (250 N, 300 N,
4 m/s), 1.5 times the reference's largest reading rounded up.

## The reference

The reference program holds a token and reads the true poses through two privileged calls
(`get_state_privileged()`, `describe_privileged()`); it moves the arm only through the public calls, takes a frame of each
camera at the start, and is graded by the same server and monitor as any program, with the same simulated-time cap and a wall-clock cap of its own (4826 s against 8074 s). Without the token the
privileged calls answer as forbidden calls and are counted. It plans in closed form on the true poses with fixed
waypoints: no sampling planner and no dependence on wall-clock time. Its order is leg_R, leg_L, arm_L, arm_R, head. Its complete
episodes of record: 5 of 5 mates fastened at the design bolt position in every episode, no
safety event, no unplanned contact of the arm, 365.99 to 369.40 s of simulated time. The force and
torque sensor's bias and noise are chosen anew at random in every episode, the reference's episodes on the nominal
layout included; the reference's runs agree in their physics and differ in the sensor's readings.

## Budgets

The simulated-time cap is 3.5 times the reference's simulated time, rounded up, and is never reduced
to fit a time-out. The wall cap is that cap times the wall seconds per simulated second of the slowest
complete program measured in the task image, times 1.1, plus the frame allowance times the cost of a frame, the first
frame and the build. The verifier's time-out is the episodes times build and wall cap, plus the fixed parts, times
1.30; it may exceed the usual figure of the benchmark's tasks. Today's values: simulated cap
1293 s by the rule, wall cap
8074 s, a simulated second 4.2 s of wall clock, a frame 4.4 s, the first frame 51 s, verifier time-out
31687 s, the agent's session 21600 s. Of these inputs the pace was
measured in the task image on the shared, busy host: the slowest episode of a complete program that fastens the arms
first, from the quality control, in place of the reference's scaled pace; the cost of a frame is scaled, not measured:
the quiet number of record times 1.2, because the shared host never came to a low load during the
budget measurement; the first frame and the build were measured in the task image; the simulated cap, the frame
allowance and the episodes come from the rule and the configuration.

## Checks of the grading that ran on a host only

The grader review's rows on long outputs, on a failing render worker and on loopback traffic ran on a host on the fixed
build, not in the task image; regression tests of the cell, which are not part of this task folder, keep
them. A flood of 48 MiB on the wire ended
the episode as a protocol error after 0.08 s with the grader 4.9 MiB larger; a flood of
standard error left a log of 5242968 bytes; a program that prints 24 MiB while it is
imported makes the grader grow by less than 8 MiB; a render worker that exits, one that answers
with a missing frame file and one that never answers ended the run as a task error after 9.6 s,
10.6 s and 910 s; 5000 loopback datagrams of another process (the
host's counter rose by 5016) beside a sleeping program left the network gate passed. The row of a
program that asks for memory until it is stopped was not run. Not checked before publication; read only from a kept
scoring sandbox by the audit after the published episode: that the program's account cannot read the stage's hidden
folders and files; the size of each agent log against the cap of 5 MB per episode; whether
`import_check.log` exists.

## Known limitations

- The cell fastens a module that is inside its lock window and slower than 5 mm/s and
  2 degrees per second for 0.5 s, also while it still moves: a lock is not a proof of rest. The lock window allows a seat gap
  of up to 0.5 mm.
- Collision pieces can stand outside the true surfaces (the manual prints the sizes per body). Beside the brace's
  tines pieces of the torso stop its turn earlier than the true surface would.
- The true surfaces can stand outside the collision pieces (the manual prints the sizes per kind of module). In the
  reference's own episodes the right sole's true surface and its station's toe wall overlap by
  0.56 to 0.63 mm and the left sole's by 0.66 to 0.72 mm while their collision
  pieces do not touch; at the toe the soles' true surface stands up to 0.78 mm outside their pieces. As the
  right leg leaves its station, the right sole's true surface also reaches into the station's side lip, by
  0.23 mm on the frames sampled from a recorded run of the reference and 0.044 mm on denser
  samples of the same run.
- At the head's grasp place countersunk screws that are in the mesh files and have no collision piece lie in the
  planes of the pad faces and stand up to 1.3 mm off the nearest piece: in a frame a pad can appear sunk
  up to that depth into a screw head.
- Some bought parts that have no collision piece, actuators of the arms and the legs, show on the outside of their
  modules, within the manual's seen-from-outside distances of a piece. Rays that meet the mesh files and no piece on
  their whole line are 0.02 to 0.07 percent of a module's rays, each at a part's outline.
- Fasteners inside a module are in the mesh files and do not collide. In the reference's own episodes such a fastener
  may pass into another body by at most 0.5 mm; bodies that both carry collision pieces by at most the
  sum of their recorded deviations plus 0.1 mm; overlaps between neighbouring bodies of a module at
  its joints are the makers' CAD and are listed, not judged: in the carry they reach 7.97 mm.
- The cameras' frames are pinhole images and the wrist camera is rendered at its nominal mount and field: the cell
  applies no lens distortion and no tolerance of the wrist camera's mount or lens.
- The camera layer: a broken outline on turned plate edges, the plates' contact shade missing, a few nearly empty
  grey codes in every frame, weak holes on the left shoulder socket in colour (the depth frame resolves them), screw
  heads as bright as their faces, facets on coarse meshes. Frames of an unchanged state differ slightly.
- The drawings: in section F-F the groove at the root of the neck unit's boss is drawn partly closed, and gaps and
  grooves of bought units narrower than the closing distance of a sheet's legend are drawn closed; some diameters of
  section F-F and of detail G are one-sided lines; nothing beyond the plane is drawn inside the openings of bought
  units; threads are not drawn.
- The photographs: in the photograph of the right shoulder socket the left arm's shoulder end shows deep in the bore
  (the caption says so); in the photograph of the held leg the gap is a narrow sliver beside the stand's upright; the
  photograph of the head in its nest has no datum board in view.
- The data pages: the models' output discs of types D and E carry no hole pattern, although the real units are bolted
  there; the pages and the texts say 'where the model has one'.
- The interface screws (41) are not in the kit, not drawn and not photographed.
- The right shoulder's shell holes are turned 5.53 degrees against the threaded holes in the
  makers' CAD; the design bolt position follows the threaded holes.
- In the task image the folder ~/.cache/uv of the agent's account belongs to root, so the venv tool uv fails there with
  its default cache; uv works with a cache folder of the agent's own, and no text of the task asks for uv.

## The agent-read texts

Every agent-read sentence is rendered from a template; every number in it is bound to a file of this folder's
build inputs. When the builder runs the lint of the texts, the build fails on a number without a binding, on a call or
key that the cell's code does not have, and on a name of the leak lists.
