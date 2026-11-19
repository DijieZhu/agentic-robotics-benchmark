# Assembly cell API (`cell`), version quadruped-assembly-cell/1.0.0

Your program `/home/user/submission/assemble.py` exposes `solve(cell) -> None`. The grader runs it once per hidden
episode in a fresh process under a dedicated unprivileged account, from a per-run directory holding a root-owned
READ-ONLY copy of your submission folder (regular files only: symbolic links and `__pycache__` are not copied), with an
empty writable `$HOME` / `$TMPDIR`, a minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL=osmesa`,
`PYTHONPATH` = a read-only copy of `~/given/lib`, `ARB_GIVEN_DIR` = a read-only copy of `~/given`, BLAS threads 2) and the
launcher `python3 -B -E -s -c <bootstrap> <lib> assemble` (the bootstrap puts `<lib>` on `sys.path` and calls
`arm.assembly_info.client.main`, which imports `assemble` from the current directory), or `bash run.sh` when you ship one.
`cell` is a proxy (`arm.assembly_info.client.InfoCell`): every method is one JSON-lines request on the process's original
stdout, answered on stdin by the grader's physics server; `print()` goes to stderr, and so does everything else written
to file descriptor 1 (native libraries' output too, such as the Filament renderer's start-up lines): none of it reaches
the wire.

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

## Rendering your own frames

MuJoCo's own renderer (`mujoco.Renderer`, or the render layer's `"mujoco"` backend) draws a different picture than the
cell's cameras (no hall, MuJoCo's own lights, flat colours, no reflections), so image processing tuned on its frames
may behave differently on the cell's frames; the given render layer is the way to see what the cell's cameras would see.

The given render layer renders like the cell's cameras: MuJoCo's Filament renderer (physically based; on the CPU through
Vulkan, Mesa's lavapipe) in the factory hall of `~/render` (its baked lightmaps, its lamps, its HDR image light, the
scanned oak bench top), with the cell's look (`arm/assembly_quad/agent_look.json` and `agent_render.make_look()`: light
units, exposure, the filmic display transform, physically based terms per material class). Depth comes from MuJoCo's
renderer on the same geometry: metres along the optical axis, registered with the colour pixels, 0 where nothing is
hit. Cameras are exact pinholes (`render.spec.CameraSpec`: `fx`, `fy`, `cx`, `cy` in pixels, `world_T_cam` the pose in
the cell frame with OpenCV axes: x right, y down, z forward); lens distortion, noise, blur, vignetting, colour response
and the depth-sensor model are the render layer's photo pipeline (`render.spec.PhotoSpec`), the cell's own code. Two
ways in (put `~/given/lib` on `sys.path`):

* `arm.assembly_quad.agent_render.open_session(spec, exposure_factor=1.0)`: the cell cameras' own render session for
  your twin of this cell (`spec`: your `mujoco.MjSpec` in the cell frame, copied). Name the bench top geom `table_top`
  with material `mat_bench` for the oak top; geoms named `floor`, `backdrop_*`, `bench_leg_*`, `bench_rail_*`,
  `bench_shelf` and every geom in a group above 2 are left out; the robot table's steel frame, the control box with its
  pendant, the floor tape and the hall are added, and the hall's lamps replace your scene's lights. `exposure_factor`
  stands for the hall's brightness (the cell's own varies by up to about 20 % between runs). The session follows the
  render layer's interface: `sync_from(model, data)` copies your physics state by joint name,
  `render_raw_views([cam], ("rgb", "depth"))` gives the pinhole images, `render([cam], photo=..., seed=k,
  outputs=("rgb", "sensor_depth"))` runs the photo pipeline.
* `render.filament.view.open_view(spec_or_mjcf_path, room=..., look=..., surface=..., bench_xy=(0.45, 0.0),
  exposure_factor=1.0)`: any scene of yours, rendered exactly as given: none of the cell's props are added (no robot
  table frame, control box or floor tape), your scene's own floor and walls stay unless `drop_prefixes` names them, and
  only geoms in a group above 2 are left out; with `room`, the hall is added and its lamps replace your scene's lights
  (unless `keep_lights`). A twin of this cell goes through `agent_render.open_session` instead. `view.render(state,
  camera, width=None, height=None, depth=False)` returns RGB uint8 H x W x 3 (with `depth=True` also depth float32
  H x W); `state` is an `MjData` (joints copied by name), a `(model, data)` pair, a dict `{joint name: value}` or None;
  `camera` is a camera name of the model (with `width`, `height`), a `CameraSpec` or a dict `{"width", "height", "fx",
  "fy", "cx", "cy", "world_T_cam"}` (`"K"` (a 3 x 3 matrix) may stand for the four intrinsics). `look` is the cell's
  look (`agent_render.make_look()`), a look file of your own (JSON: the fields of `render.filament.backend.Look`;
  `"materials"` a list of `[geom glob, material glob, {"roughness", "metallic"}]` rules, the first match wins) or None.
  For the hall's brightness pass `exposure_factor=...` to `open_view` and leave `make_look()` at its default (a view
  would render `make_look(f)` with a wrong display gain). `view.session` is a render-layer session for the photo
  pipeline.

```python
import os, sys
sys.path.insert(0, os.path.expanduser("~/given/lib"))
from arm.assembly_quad import agent_render
from render.filament.view import open_view
from render.spec import CameraSpec

cam = CameraSpec.look_at("my_top", eye=(0.6, -0.3, 1.2), target=(0.5, 0.0, 0.0), width=1280, height=960, hfov_deg=70.0)
session = agent_render.open_session(spec)            # spec: your MjSpec of the cell
session.sync_from(model, data)                       # model, data: your physics
raw = session.render_raw_views([cam], ("rgb", "depth"))[0]    # raw.rgb, raw.depth
session.close()                                      # one session per process at a time

# the scenery: ~/render in your session; in the graded run `~` is an empty folder and it is next to $ARB_GIVEN_DIR
scenery = os.path.join(os.path.dirname(os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))), "render")
view = open_view(part_spec, room=os.path.join(scenery, "room"), surface=os.path.join(scenery, "oak"),
                 look=agent_render.make_look(), bench_xy=(0.45, 0.0))   # part_spec: a scene of yours, not the cell twin
rgb, depth = view.render(part_data, cam, depth=True)                     # part_data: its MjData
rgb_w = view.render({"my_hinge": 0.5}, "my_camera", 640, 480)            # a joint and a camera of part_spec
view.close()
```

What it costs here (8 CPUs, a twin of the whole cell): `open_session` takes about 13-15 s (up to about 30 s on a busy
machine), the display transform's lookup table (about 2 s) included, and the open session holds about 6-7 GB. Its first
frame takes about 10-15 s: that is the first frame after Filament's context is made, which the first session or view of
a process pays, and again the next one after the last was closed; the first frame of each further image size takes a
few seconds more (the photo pipeline renders a slightly larger image per camera; its first `top` picture took 5-20 s
here). Then a 1280 x 960 frame takes about 1.2-1.5 s with depth (about 2 s through the photo pipeline) and a 640 x 480
frame about 0.6-0.8 s (0.8-0.9 s); `open_view` takes about 5-20 s on the same scene and renders as fast. Until you give
a state (`sync_from`, `set_state`, or `state` in `view.render`), a session or view shows your twin's own pose (`qpos0`:
the bodies where your spec puts them). One Filament session per process at a time: close it before opening the next (a
second one is refused). Closing the last one also releases Filament's context and its memory once nothing refers to the
closed session (about 0.3-1.4 GB stays with the process; each open/close cycle keeps about 0.2-0.7 GB more; keep one
session open for the whole program), so the next session's or view's first frame takes the 10-15 s again. Your program
may use the layer during an episode too (the session finds the hall next to `$ARB_GIVEN_DIR`; `~` is an empty folder
there, so give `open_view` the folders next to `$ARB_GIVEN_DIR`, as the example does); it then has 2 render threads
(about 1.5-1.9 s / 2.0-2.4 s and 0.7-0.9 s / 0.8-1.0 s per frame, its first frame about 10-14 s), and the time counts
against the episode's wall clock. In the graded run your program shares the verifier's memory (24 GB) with the cell's
own camera session, which opens at your first `get_camera` call: measured in this cell, a program holding a twin session
took about 5 GB and the cell's own session about 4-7 GB (the cell's process reached about 14 GB by the third episode).
MuJoCo's classic renderer stays available: the render layer's MuJoCo backend (`render.backends.get_backend("mujoco")`)
and `mujoco.Renderer`.
