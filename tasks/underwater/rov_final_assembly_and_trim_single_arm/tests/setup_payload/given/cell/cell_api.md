# Final-assembly station: cell API (`cell`), version rov-assembly-cell/1.0.0

Your program `/home/user/submission/assemble.py` exposes `solve(cell) -> None`. The grader runs it once per hidden
episode in a fresh process under a dedicated unprivileged account, from a per-run directory holding a root-owned
READ-ONLY copy of your submission folder (regular files only: symbolic links and `__pycache__` are not copied), with
an empty writable `$HOME` / `$TMPDIR`, a minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL` set for
off-screen rendering, `PYTHONPATH` = a read-only copy of `~/given/lib`, `ARB_GIVEN_DIR` = a read-only copy of
`~/given`, BLAS threads 2) and the launcher `python3 -B -E -s -c <bootstrap> <lib> assemble` (the bootstrap puts
`<lib>` on `sys.path` and calls `arm.assembly_rov.cell_client_ext.main`, which imports `assemble` from the current
directory), or `bash run.sh` when you ship one. The process cannot open an Internet socket: creating one fails. `cell`
is a proxy (`arm.assembly_rov.cell_client_ext.StationCell`: the library's general camera-cell proxy
`arm.assembly_info.client.InfoCell` with `fixture_index` and the `force` argument of `command` added): every method is
one JSON-lines request on the process's original stdout, answered on stdin by the grader's cell server; `print()` goes
to stderr. A reply that is nothing but an error (`forbidden`, `bad_params`, `camera_budget`, an unknown camera) raises
`CellError` in the proxy; an exception that `solve` does not catch ends the episode with what it achieved.

Units on the API: metres, radians, newtons, quaternions `wxyz`; the cell frame = on the bench top at the arm's base
axis, z up (see `cell_manual.md`). The given part files and the drawings are in millimetres.

| call | returns | notes |
|---|---|---|
| `describe()` | `robot` (model, base pose and axes, joint names / ranges, home posture, torque limits, tcp site and offset, tcp axes, gripper with its width and force ranges, wrist housing, force-sensor note), `control` (tick, rate limits, stiffness range, modes, the handling procedure for a held unit), `fixture` (the index positions, the start position, the duration of a turn, the turns of a run, the turn column's radius, the load cells' order, noise, zero offset and the time after which a reading is valid), `fastening` (the rules, every place's windows with the rest before capture, lock tolerance and weight, the total weight), `work_order` (`unit`, `delivery_site` with `water_temperature_C` and `salinity_psu`), `cameras` (name -> resolution, rgb / depth, nominal horizontal field of view, mount description), `camera_budget_frames`, `datums` (the documented boards; the vehicle's pose on the positioner at the two index positions; the place of the ballast box; the nominal places of the thruster tray and the racks), `items` (what is in the cell, as text), `budgets`, `safety` (the limits and the rules as text), `task`, `units`, `cell_version` | static for the episode; no unit poses, no fixture poses beyond the nominal ones, nothing about propellers, nothing about what is captured, fastened or scrapped, nothing about the trim. `work_order` is this run's work order (`~/given/docs/work_order_example.json` shows its form); it differs between episodes |
| `get_state()` | `sim_time`, `done`, `robot {q[6], dq[6], cmd_q[6], gripper_width, tcp_pos[3], tcp_quat[4], wrench[6], mode, stiffness_scale}`, `fixture {index, moving, load_N[3]}` | joint angles, the tcp pose and the gripper width are exact; `wrench` (force N, torque N m, at the tool flange's centre in the flange frame) carries white noise (about 0.3 N / 0.02 N m rms per reading) and an unknown per-episode bias (up to about 1 N / 0.05 N m), and includes the gripper's own weight: tare it. `fixture.index` is `keel_up` or `deck_up`: the index position the positioner is in or, during a turn, the one it turns to; `fixture.moving` is true during a turn; `fixture.load_N` = the three load cells A, B, C in newtons, tared, compression positive, one new sample per tick with white noise of 0.4 N rms and a per-episode zero offset of up to 0.02 N per cell, not converted to the vehicle's frame (`cell_manual.md`, "Weighing", says when a reading is valid) |
| `set_control(mode, stiffness_scale)` | ok | `mode` in `joint`, `cartesian`; `stiffness_scale` in [0.2, 1.0] scales the six joint position gains (the damping with its square root); pass both: `set_control("cartesian")` alone resets the scale to 1.0. Entering Cartesian mode takes its goal from the COMMANDED configuration |
| `command(q=[6])` or `command(tcp_pos=[3], tcp_quat=[4])`, optional `width`, optional `force` | ok / error | targets are held until the next command; joint targets are clamped to the ranges and rate-limited to 1.5 rad/s; Cartesian targets go through the cell's damped-least-squares differential IK on the `ur/tcp` site (at most 0.5 m/s and 1.5 rad/s, and the joint rate limit; no null-space control: it slows near the arm's singularities); there is no limit on acceleration (`cell_manual.md`, "The gripper", has the station's limits for carrying a unit, which the cell does not enforce); the gripper `width` (m, 0.0074 .. 0.1336) ramps at 0.10 m/s (an opening command releases at once); `force` (N per pad, 10 .. 125; 100 until it is set) is the grip force with which the pads close on a unit and stays until the next `force`; a target of the wrong length or otherwise malformed answers `bad_params`; a `q` target in Cartesian mode, a tcp target in joint mode and any command after the episode is over answer `{"ok": false, "error": ...}` and change nothing |
| `step(n_ticks)` | `sim_time`, `done`, `events[]` | advances the cell by n x 10 ms; the arm's controller holds the commanded posture against gravity all the while; nothing moves outside `step`, a turn of the positioner included; at most 500 ticks per call: a larger n is cut to that many, and the reply's `sim_time` tells how far the cell went; events: only `safety:<what>` (a violation that zeroes the episode) and `episode_over`; nothing reports a capture, a fastening, a grasp, a dropped or a scrapped unit |
| `fixture_index(target)` | `{"ok": true}` or `{"refused": "<why>"}` | `target` is `keel_up` or `deck_up`. Starts the positioner's turn to that index position; the turn then takes 6 s of `step`. When the turn starts, every unit on the vehicle that is not fastened is scrapped (`cell_manual.md`, "Turning procedure"). Refused, with nothing done and no turn used: `robot_in_column` (a part of the robot is inside the turn column), `moving` (a turn is under way), `no_turns_left` (the run's 4 turns are used). A call for the index position the positioner is already in answers ok, does nothing and uses no turn. A refusal is not a violation and not an illegal call |
| `get_camera(name, depth=True)` | `camera`, `frame_index`, `width`, `height`, `format` (`png`), `rgb_png_b64`, and with depth `sensor_depth_u16_mm_b64`, `depth_unit_m`, `depth_invalid_value` | one frame of `top` or `wrist` (both RGB-D) rendered at the current state of the cell: the PNG colour image and the depth sensor's 16-bit PNG (depth along the optical axis, in steps of `depth_unit_m`, registered to the colour pixels, `depth_invalid_value` = no reading; the transparent tube and dome of the electronics enclosure give no reading, nor does the tube of the battery enclosure); no intrinsics, no pose, no exact depth; at most 600 frames per episode over all cameras, then `{"error": "camera_budget"}`; rendering takes about 0.5-2.4 s of wall clock per frame (an episode's first frame about 3-20 s more: the renderer's session opens then) and counts against the episode's wall-clock cap |
| `finish()` | ok | ends the episode at once (a unit still being pulled home does not fasten); the grader then reads which places the cell has fastened and which ballast sites hold a fastened block |

`arm.assembly_info.client.decode_frame(reply)` returns `(rgb, depth)`: `rgb` the colour image as a uint8 array (height
x width x three channels), `depth` a float32 array (height x width, metres, NaN = no reading).

The arm and the gripper are the given models built the way the cell builds them:
`arm.assembly_rov.gripper_wide.add_ur_wide` (on an `mujoco.MjSpec`: the arm, the gripper on the real flange face, the
tcp, the camera housing) and `arm.assembly_rov.gripper_wide.WideController` (the cell's controller: the same
`set_control` and `command` as above in your own simulation; there the grip force is set with its `command_force`).
The gripper's measured tables are in `arm.assembly_rov.gripper_wide`.

Anything else answers `{"error": "forbidden"}`, counts as an illegal call and does nothing. In particular
`get_state_privileged()`, `describe_privileged()` and `reference_assemble()` (exact unit poses, the places' states,
the grader's reference program), `reset(seed)` and `truth()` are not available to your program.

## Budgets and aborts

1200 s of simulated time and 2400 s of wall clock per episode (rendering included); 120 s of silence aborts the
episode: the time from the program's start, or from the cell's reply to a request, until the program's next request
(the cell's own time inside a request does not count; any request resets it, `get_state()` being the cheapest); a
CPU-time backstop of 4 x the wall cap applies per process; every process of the run's account is killed when the
episode ends; a program that breaks the wire (crash, malformed line, exit) ends ITS episode with what it achieved so
far and the other episodes still run. Whatever ends an episode (`finish`, the end of `solve`, a cap, the silence
limit, a broken wire), it is scored with what the cell has fastened at that moment.
