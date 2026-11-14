# Assembly cell API (`cell`), version quadruped-assembly-cell/1.0.0

Your program `/home/user/submission/assemble.py` exposes `solve(cell) -> None`. The grader runs it once per hidden
episode in a fresh process under a dedicated unprivileged account, from a per-run directory holding a root-owned
READ-ONLY copy of your submission folder (regular files only: symbolic links and `__pycache__` are not copied), with an
empty writable `$HOME` / `$TMPDIR`, a minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL=osmesa`,
`PYTHONPATH` = a read-only copy of `~/given/lib`, `ARB_GIVEN_DIR` = a read-only copy of `~/given`, BLAS threads 2) and the
launcher `python3 -B -E -s -c <bootstrap> <lib> assemble` (the bootstrap puts `<lib>` on `sys.path` and calls
`arm.assembly_info.client.main`, which imports `assemble` from the current directory), or `bash run.sh` when you ship one.
`cell` is a proxy (`arm.assembly_info.client.InfoCell`): every method is one JSON-lines request on the process's original
stdout, answered on stdin by the grader's physics server; `print()` goes to stderr.

Units on the API: metres, radians, quaternions `wxyz`; the cell frame = on the bench top at the arm's base axis, z up
(see `cell_manual.md`). The given unit files are in millimetres.

| call | returns | notes |
|---|---|---|
| `describe()` | `robot` (model, base pose and axes, joint names / ranges, home posture, torque limits, tcp site and offset, tcp axes, gripper with its force model, wrist housing, force-sensor note), `control` (tick, rate limits, stiffness range, modes), `fastening` (the rules and their numbers, the key heights, the weights), `cameras` (name -> resolution, rgb / depth, nominal horizontal field of view, mount description), `camera_budget_frames`, `datums` (the documented boards and the stand's nominal position, as in `cell_manual.md`), `items` (what is on the bench, as text), `budgets`, `safety` | static for the episode; no unit poses, no slots, no stand pose |
| `get_state()` | `sim_time`, `done`, `robot {q[6], dq[6], cmd_q[6], gripper_width, tcp_pos[3], tcp_quat[4], wrench[6], mode, stiffness_scale}` | joint angles, the tcp pose and the gripper width are exact; `wrench` (force N, torque N m, at the tool flange's centre in the flange frame) carries white noise (~0.3 N / 0.02 N m rms per reading) and an unknown per-episode bias (up to ~1 N / 0.05 N m), and includes the gripper's own weight: tare it |
| `set_control(mode, stiffness_scale)` | ok | `mode` in `joint`, `cartesian`; `stiffness_scale` in [0.2, 1.0] scales the six joint position gains (the damping with its square root); pass both: `set_control("cartesian")` alone resets the scale to 1.0. Entering Cartesian mode takes its goal from the COMMANDED configuration |
| `command(q=[6])` or `command(tcp_pos=[3], tcp_quat=[4])`, optional `width` | ok / error | targets are held until the next command; joint targets are clamped to the ranges and rate-limited to 1.5 rad/s; Cartesian targets go through the cell's damped-least-squares differential IK on the `ur/tcp` site (<= 0.5 m/s, 1.5 rad/s, the joint rate limit; no null-space control: it slows near the arm's singularities); the gripper `width` (m, 0..0.085) ramps at 0.10 m/s (an opening command releases at once); grip force as `describe()["robot"]["gripper"]["force_model"]`; a 7-vector or a malformed target answers `bad_params` |
| `step(n_ticks)` | `sim_time`, `done`, `events[]` | advances n x 10 ms of physics (1 kHz inside, the arm's computed-torque control recomputed every 1 ms: the arm holds its posture against gravity); nothing moves outside `step`; n <= 500 per call; events: only `safety:<what>` and `episode_over` (no "fastened", "grasped" or "dropped" events) |
| `get_camera(name, depth=True)` | `camera`, `frame_index`, `width`, `height`, `format` (`png`), `rgb_png_b64`, and with depth `sensor_depth_u16_mm_b64`, `depth_unit_m` (0.001), `depth_invalid_value` (0) | one frame of `top` or `wrist` (both RGB-D) rendered at the current physics state: the PNG colour image and the depth sensor's 16-bit PNG (depth along the optical axis, millimetres, registered to the colour pixels, 0 = no reading); no intrinsics, no pose, no exact depth; at most 600 frames per episode over all cameras, then `{"error": "camera_budget"}`; rendering takes about 0.5-1.5 s of wall clock per frame (an episode's first frame about 25-30 s more: the renderer's session opens then) and counts against the episode's wall-clock cap |
| `finish()` | ok | ends the episode at once (a unit still being pulled home does not fasten); the grader then reads which slots the cell has fastened |

`arm.assembly_info.client.decode_frame(reply)` returns `(rgb uint8 H x W x 3, depth float32 H x W metres, NaN = no reading)`.

The arm and the gripper are the given models built the way the cell builds them: `arm.assembly_quad.ur_robot.add_ur`
(on an `mujoco.MjSpec`; the gripper on the real flange face, the pinch-point tcp, the camera housing) and
`arm.assembly_quad.ur_controller.URController` (the cell's controller: the same API as above in your own simulation).

Anything else answers `{"error": "forbidden"}`, counts as an illegal call and does nothing. In particular
`get_state_privileged()`, `describe_privileged()` and `reference_assemble()` (exact unit poses, slot states, the
grader's reference program), `reset(seed)` and `truth()` are not available to your program.

## Budgets and aborts

1200 s of simulated time and 2400 s of wall clock per episode (rendering included); 120 s of silence aborts the episode:
the time from the program's start, or from the cell's reply to a request, until the program's next request (the cell's
own time inside a request does not count; any request resets it, `get_state()` being the cheapest); a CPU-time backstop
of 4 x the wall cap applies per process; every process of the run's account is killed when the episode ends; a program
that breaks the wire (crash, malformed line, exit) ends ITS episode with what it achieved so far and the other episodes
still run.
