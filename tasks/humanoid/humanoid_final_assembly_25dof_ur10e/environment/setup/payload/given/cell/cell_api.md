# Assembly cell API

Your program `/home/user/submission/assemble.py` exposes `solve(cell)`. The grader runs it once per hidden episode in a fresh process
under a dedicated unprivileged account, from a per-run directory that holds a read-only copy of your submission folder
(regular files only), with an empty writable home and temporary folder, a minimal environment (the search path, the
home and temporary folders, the locale, the off-screen rendering setting, the Python path set to a read-only copy of
`~/given/lib/`, and `ARB_GIVEN_DIR` set to a read-only copy of `~/given`) and the launcher `python3 -B -E -s -c <bootstrap> <lib> assemble`, or
`bash run.sh` when you ship one. The launcher imports `assemble.py` from the current directory, calls the entry
function with the cell's proxy and then ends the episode. The process cannot open an Internet socket.

The proxy is `arm.assembly_humanoid.client.HumanoidCell`: every method is a single JSON-lines request on the process's original standard output,
answered on its standard input by the grader's physics server. Whatever the program prints, and whatever native
libraries write to that descriptor, goes to standard error and never reaches the wire. Of the standard error the
grader keeps up to 5 MB per episode in its own log and drops what comes after; nothing on standard
error counts for the score.

Units on the API: metres, radians, newtons; quaternions w x y z; the cell frame of `~/given/cell/cell_manual.md`, section
2. The mesh files are in millimetres.

| call | returns | notes |
|---|---|---|
| `describe()` | `cell_version`, `units`, `robot`, `control`, `cameras`, `fastening`, `items`, `datums`, `budgets`, `safety`, `scoring`, `observation` | the station's public description: the arm, the controller, the gripper, the cameras' nominal mounts and the overhead camera's tolerances, the fastening table with what was measured, where each unit is presented and how far its fixture is placed per episode, the datum boards, the budgets, the safety numbers, the weights. The same in every episode: no module pose, no fixture pose of the episode, no seat pose, no bolt position, nothing about what is fastened |
| `get_state()` | `sim_time`, `done`, `robot`, `robot.q`, `robot.dq`, `robot.cmd_q`, `robot.gripper_width`, `robot.force_cmd`, `robot.tcp_pos`, `robot.tcp_quat`, `robot.wrench`, `robot.mode`, `robot.stiffness_scale` | the arm's joint angles and speeds, the commanded configuration, the gripper's opening and force setting, the pose reached by the tool point, the control mode: exact. The wrench (force in N, torque in N m, at the flange in the flange frame) carries white noise and a bias chosen at random per episode, and includes the weight of the gripper and of what it holds |
| `set_control(mode='joint', stiffness_scale=1.0)` | ok, the mode, the stiffness scale | the mode is `joint` or `cartesian`; the stiffness scale lies in 0.2 to 1.0; a `set_control()` call with the mode alone sets the scale back to 1.0 |
| `command(q=None, tcp_pos=None, tcp_quat=None, width=None, force=None)` | `{"ok": true}`, or `{"ok": false, "error": ...}` | a joint target (6 angles inside the joint ranges) in joint mode, or a tool pose (position and quaternion together) in Cartesian mode; optional the gripper's opening in metres (closed to 0.1336) and the force per pad in newtons (10 to 125). Targets are held until the next command. A value outside its range is refused, not clipped, and nothing of that call is applied ("Error replies" below) |
| `step(n_ticks=1)` | `sim_time`, `done`, `events` | advances the physics by the given number of control ticks of 10 ms; nothing moves outside this call; more than 500 ticks per call are cut to 500. The events are only safety events and the end of the episode; nothing reports a lock, a candidate, a dropped or a scrapped module |
| `get_camera(name='top', depth=True)` | `ok`, `camera`, `frame_index`, `width`, `height`, `format`, `rgb_png_b64`, `sensor_depth_u16_mm_b64`, `depth_unit_m`, `depth_invalid_value` | a frame of `top` or `wrist` at the current physics state: the colour image and, with depth, the 16-bit depth image, both as PNG images carried as text; no intrinsics, no pose. At most 450 frames per episode over both cameras; a call beyond them is an error reply with the text `camera_budget`. A frame that cannot be rendered is an error reply with the text `camera_unavailable` and still counts |
| `finish()` | ok, the simulated time | ends the episode at once; the grader then reads which mates the cell has fastened |

`arm.assembly_humanoid.client.decode_frame` turns a camera reply into the colour image (height by width by the colour channels, unsigned
bytes) and the depth image in metres (not-a-number where there is no reading).

Events of `step()`: `safety:robot_contact:<robot body>:<other body>:<N>N`, `safety:part_force:<module>:<body pressed on>:<N>N`, `safety:part_speed:<module>:<v>m/s`, `safety:floor_contact:<module>:<module body>`, `episode_over`. A safety event carries the rule, the bodies and the force or speed; the
episode goes on.

## Error replies

A call other than the seven calls of the table is an error reply with the text `forbidden`; it is
counted as an illegal call and does nothing. A call with a wrong parameter (a wrong name, type or count, a value that is not finite or lies outside its
range) is an error reply whose text starts with `bad_params`; it changes nothing.

Through the given proxy an error reply does not come back as a value: the call raises `arm.assembly_humanoid.client.CellError` with the
error's text. A program that catches it goes on, and so does the episode. An error that the program does not catch
leaves the entry function, and the given launcher ends the episode at once with what was achieved.

`command()` has replies that are returned, not raised: `{"ok": false, "error": ...}` with the text
`episode_over` after the episode has ended, `q target needs joint mode` for a joint target in Cartesian mode and
`tcp target needs cartesian mode` for a tool pose in joint mode. Nothing of such a call is applied.

## Budgets and aborts

Per episode: 1293 s of simulated time, at which the episode ends; 8074 s of wall
clock, rendering included; 120 s of silence, counted from the program's start or from the cell's
reply to a request until the program's next request (the cell's own time inside a request does not count, and any
request resets it). The wall cap is built on about 4.2 s of wall clock per simulated second
on a machine of the grading machine's size, and on the allowance for a frame that the last section gives: the wall
clock can end an episode before the simulated time does. Each process of the program may use at most
32296 s of CPU time per episode (four times the wall cap); a process that passes it is
killed, which for the program's own process is a crash. A program that breaks the wire (a crash, a malformed line, a
line longer than 4194304 characters, an exit) ends its own episode with what it achieved; the other
episodes still run. The line limit is the configuration's `max_request_bytes`: the given proxy writes its lines in
ASCII, one byte per character.

## A short exchange

```python
def solve(cell):
    info = cell.describe()                      # static: the same in every episode
    cell.set_control("cartesian", 1.0)
    state = cell.get_state()
    position, quaternion = state["robot"]["tcp_pos"], state["robot"]["tcp_quat"]
    reply = cell.get_camera("top", depth=True)  # the first frame of an episode opens the renderer
    cell.command(tcp_pos=position, tcp_quat=quaternion, width=0.1336, force=100)
    out = cell.step(50)                         # {"sim_time": 0.5, "done": False, "events": []}
    cell.finish()
```

## Rendering your own frames

MuJoCo's own renderer gives a different picture than the cell's cameras: no room, its own lights, flat colours. Image
processing tuned on its frames may behave differently on the cell's frames. The given render layer is the way to see
what the cell's cameras would see: it is the cell cameras' own renderer, room and look.

`arm.assembly_humanoid.agent_render` holds it. `load_twin()` reads the render twin of the cell from the scenery folder;
`open_view(twin)` opens a view of it; the view's render call takes a state, a camera and the depth switch and returns
the colour image and the depth image in metres. The twin holds every body of the cell that has a mesh, each as a free
body: the bodies of the base and of the modules, each in its unit frame; every fixture, in the cell frame at its
nominal place, so that its pose is its motion away from that place; and the bodies of the arm and of the gripper. A
state is a mapping from the name of a body's free joint, which `pose_joint(body)` gives, to the 7 numbers x, y, z, qw,
qx, qy, qz in the cell frame; a body that is not placed keeps its last pose. A camera is a mapping with a name, the image size, the focal lengths and the principal point in
pixels, and its pose in the cell frame with the camera's x to the right, y down and z forward. A camera named
`wrist` is exposed as the cell's wrist camera, any other as the overhead camera. The layer applies no lens
distortion and no sensor noise, and neither does the cell. The cell rounds its depth image to steps of
1 mm (`~/given/cell/cell_manual.md`, section 9); the layer's depth image is not rounded and is
0 where the view meets no surface.

The scenery folder is `~/render/` in your session. In the graded run the home folder is empty and the scenery is
the folder next to the copy of the given files; `render_dir()` finds it in both.

What it costs. A view holds about 5.2 GB of memory while it is open; open a single view per process.
For a frame of the cell's cameras the wall cap allows about 4.4 s, and about 51 s more
for the first frame of an episode, when the renderer's session opens; expect a frame of the overhead camera's size in a
view of your own, and the first frame of that view, to cost about as much. In the graded run your program shares the
machine (8 CPUs,
24 GB) with the cell's physics and the cell's own camera session, and the time counts against the
episode's wall clock.
