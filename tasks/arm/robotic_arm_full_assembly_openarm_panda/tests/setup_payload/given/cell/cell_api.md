# Assembly cell API (`cell`), version openarm-pair-assembly-cell/1.0.0

Your program `/home/user/submission/assemble.py` exposes `solve(cell) -> None`. The grader runs it once per hidden
episode in a fresh process under a dedicated unprivileged account, from a per-run directory holding a root-owned
READ-ONLY copy of your submission folder (regular files only: symbolic links and `__pycache__` are not copied), with an
empty writable `$HOME` / `$TMPDIR`, a minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL=osmesa`,
`PYTHONPATH` = a read-only copy of `~/given/lib`, `ARB_GIVEN_DIR` = a read-only copy of `~/given`, BLAS threads 2) and the
launcher `python3 -B -E -s -c <bootstrap> <lib> assemble` (the bootstrap puts `<lib>` on `sys.path` and calls
`arm.assembly_openarm.client.main`, which imports `assemble` from the current directory), or `bash run.sh` when you
ship one. `cell` is a proxy (`arm.assembly_openarm.client.PairCell`): every method is one JSON-lines request on the
process's original stdout, answered on stdin by the grader's physics server; `print()` goes to stderr.

Units on the API: metres, radians, quaternions `wxyz`; the cell frame = the bench-top centre, z up (see
`cell_manual.md`). The given files are in metres.

The cell has two robots, `"left"` (the Panda at y = -0.476 m, beside the left arm) and `"right"` (y = +0.476 m). Every
control call names its robot; `step` advances both.

| call | returns | notes |
|---|---|---|
| `describe()` | `robots` {`left`, `right`: model, base pose, joint names / ranges, home posture, tcp offset, gripper, force-sensor note}, `control` (tick, rate limits, stiffness range, modes), `cameras` (name -> resolution, rgb / depth, nominal horizontal field of view, mount description), `camera_budget_frames`, `bench`, `budgets`, `safety` | static for the episode; no part poses, no mates, no gallows pose |
| `get_state()` | `sim_time`, `done`, `robots` {`left`, `right`: `q[7], dq[7], cmd_q[7], gripper_width, tcp_pos[3], tcp_quat[4], wrench[6], mode, stiffness_scale`} | joint angles, the tcp pose and the gripper width are exact; `wrench` (force N, torque N m, at the hand flange in the hand frame) carries white noise (~0.3 N / 0.02 N m rms per reading) and an unknown per-episode bias (up to ~1 N / 0.05 N m), and includes the hand's own weight: tare it |
| `set_control(robot, mode, stiffness_scale)` | ok | `mode` in `joint`, `cartesian`; `stiffness_scale` in [0.2, 1.0] scales the seven joint position gains of that robot |
| `command(robot, q=[7])` or `command(robot, tcp_pos=[3], tcp_quat=[4])`, optional `width` | ok / error | targets are held until the next command; joint targets are clamped to the ranges and rate-limited to 1.5 rad/s; Cartesian targets go through the cell's damped-least-squares differential IK on the `tcp` site (<= 0.5 m/s, 1.5 rad/s, joint rate limit; the controller is `arm.assembly_cell.controller.PandaController` in the given library); the gripper `width` (m, 0..0.08) ramps at 0.05 m/s; grip force = 2500 N/m x (object width - target width) / 4 per finger, <= 70 N per finger |
| `step(n_ticks)` | `sim_time`, `done`, `events[]` | advances n x 10 ms of physics (1 kHz inside) for both robots; nothing moves outside `step`; n <= 500 per call; events: only `safety:<what>` and `episode_over` (no "fastened", "grasped" or "dropped" events) |
| `get_camera(name, depth=True)` | `camera`, `frame_index`, `width`, `height`, `format` (`png`), `rgb_png_b64`, and with depth `sensor_depth_u16_mm_b64`, `depth_unit_m` (0.001), `depth_invalid_value` (0) | one frame of `top`, `wrist_left` or `wrist_right` (all RGB-D) rendered at the current physics state: the PNG colour image and the depth sensor's 16-bit PNG (depth along the optical axis, millimetres, registered to the colour pixels, 0 = no reading); no intrinsics, no pose, no exact depth; at most 900 frames per episode over all cameras, then `{"error": "camera_budget"}`; a frame takes about 0.5-1.5 s of wall clock (the first one of an episode about 25 s more: the renderer starts) and counts against the episode's wall-clock cap |
| `finish()` | ok | ends the episode; the grader then reads which mates the cell has fastened |

`arm.assembly_openarm.client.decode_frame(reply)` returns `(rgb uint8 H x W x 3, depth float32 H x W metres, NaN = no
reading)`.

Anything else answers `{"error": "forbidden"}`, counts as an illegal call and does nothing. In particular
`get_state_privileged()`, `describe_privileged()` and `reference_assemble()` are not available to your program.

## Budgets and aborts

1800 s of simulated time and 3000 s of wall clock per episode (rendering included); 120 s of silence between two requests
aborts the episode; a CPU-time backstop of 4 x the wall cap applies per process; every process of the run's account is
killed when the episode ends; a program that breaks the wire (crash, malformed line, exit) ends ITS episode with what it
achieved so far and the other episodes still run.

## Rendering your own frames

MuJoCo's own renderer (`mujoco.Renderer`, or the render layer's `"mujoco"` backend) draws a different picture than the
cell's cameras (no hall, MuJoCo's own lights, flat colours, no reflections), so image processing tuned on its frames
may behave differently on the cell's frames; the given render layer is the way to see what the cell's cameras would see.

The given render layer renders like the cell's cameras: MuJoCo's Filament renderer (physically based; on the CPU through
Vulkan, Mesa's lavapipe; MuJoCo 3.17's Filament module) in the factory hall of `~/render` (its baked lightmaps, its
lamps, its HDR image light, the scanned oak bench top), with the cell's look (`arm/assembly_openarm/agent_look.json`
and `agent_render.make_look()`: light units, exposure, the filmic display transform, physically based terms per
material class). Depth comes from MuJoCo's renderer on the same geometry: metres along the optical axis, registered
with the colour pixels, 0 where nothing is hit. Cameras are exact pinholes (`render.spec.CameraSpec`: `fx`, `fy`, `cx`,
`cy` in pixels, `world_T_cam` the pose in the cell frame with OpenCV axes: x right, y down, z forward); lens
distortion, noise, blur, vignetting, colour response and the depth-sensor model are the render layer's photo pipeline
(`render.spec.PhotoSpec`), the cell's own code. Two ways in (put `~/given/lib` on `sys.path`):

* `arm.assembly_openarm.agent_render.open_session(spec, exposure_factor=1.0)`: the cell cameras' own render session
  for your twin of this cell (`spec`: your `mujoco.MjSpec` in the cell frame, copied). Name the bench top geom
  `table_top` with material `mat_bench` for the oak top; geoms named `floor`, `backdrop_*`, `bench_leg_*`,
  `bench_apron_*`, `bench_shelf` and every geom in a group above 2 are left out; the bench's steel frame and the hall
  are added, and the hall's lamps replace your scene's lights. The look's material classes go by material name
  (`mat_alu`, `mat_steel`, `mat_dark`, `mat_rotor`, `mat_mark`, `mat_riser`, the Pandas' `pl/*` and `pr/*`,
  `mat_bench`, the boards' `*_board__*` and `hand_plate*`; anything else gets a plain default). The datum boards are
  part of your twin: `render.scale_refs.checkerboard(name, cols=..., rows=..., square_mm=..., margin_mm=...,
  thickness_mm=...)` makes a board of the kind `cell_manual.md` describes, `render.scale_refs.add_to_spec(spec, ref)`
  puts it on the world body (its geoms and materials named `<name>__*`) and `render.scale_refs.parts(ref)` gives its
  pieces for a board on a moving body (a hand plate). `exposure_factor` stands for the hall's brightness (the cell's
  own varies by up to about 20 % between runs). The session follows the render layer's interface:
  `sync_from(model, data)` copies your physics state by joint name, `render_raw_views([cam], ("rgb", "depth"))` gives
  the pinhole images, `render([cam], photo=..., seed=k, outputs=("rgb", "sensor_depth"))` runs the photo pipeline.
* `render.filament.view.open_view(spec_or_mjcf_path, room=..., look=..., surface=..., bench_xy=(0.0, 0.0),
  exposure_factor=1.0)`: any scene of yours, rendered exactly as you give it. The hall, its lamps and its image light
  are added (the lamps replace your scene's lights unless `keep_lights=True`); nothing of the cell is (no bench frame,
  no props), and your own geoms stay, floor and walls included, unless `drop_prefixes` names them (only geoms in groups
  above `max_group`, 2, are left out). A twin of the cell goes through `agent_render.open_session` instead.
  `view.render(state, camera, width=None, height=None, depth=False)` returns RGB uint8 H x W x 3 (with `depth=True` also
  depth float32 H x W); `state` is an `MjData` (joints copied by name), a `(model, data)` pair, a dict `{joint name:
  value}` or None; `camera` is a camera name of the model (with `width`, `height`), a `CameraSpec` or a dict `{"width",
  "height", "fx", "fy", "cx", "cy", "world_T_cam"}` (`"K"` (a 3 x 3 matrix) may stand for the four intrinsics). `look`
  is the cell's look (`agent_render.make_look()`), a look file of your own (JSON: the fields of
  `render.filament.backend.Look`; `"materials"` a list of `[geom glob, material glob, {"roughness", "metallic"}]` rules,
  the first match wins) or None; `surface` a folder of scanned maps (`~/render/oak`) for the look's bench top geom. For
  the hall's brightness pass `exposure_factor=...` to `open_view` and leave `make_look()` at its default (a view would
  render `make_look(f)` with a wrong display gain). `view.session` is a render-layer session for the photo pipeline.

```python
import os, sys
sys.path.insert(0, os.path.expanduser("~/given/lib"))
from arm.assembly_openarm import agent_render
from render.filament.view import open_view
from render.spec import CameraSpec

cam = CameraSpec.look_at("my_top", eye=(0.0, -0.3, 1.5), target=(0.0, 0.0, 0.0), width=1280, height=960, hfov_deg=70.0)
session = agent_render.open_session(spec)            # spec: your MjSpec of the cell
session.sync_from(model, data)                       # model, data: your physics
raw = session.render_raw_views([cam], ("rgb", "depth"))[0]    # raw.rgb, raw.depth
session.close()                                      # one session per process at a time

view = open_view(my_spec, room=os.path.expanduser("~/render/room"), look=agent_render.make_look(),
                 surface=os.path.expanduser("~/render/oak"))   # my_spec: any scene of yours, rendered as it is
rgb, depth = view.render(my_data, cam, depth=True)
rgb_w = view.render({"my_joint": 0.5}, "my_camera", 640, 480)
view.close()
```

What it costs here (8 CPUs, a twin of the whole cell): `open_session` takes about 12-30 s (up to about 45 s on a busy
machine), the display transform's lookup table (about 2 s) included, and the open session holds about 5.5-6.5 GB. Its
first frame takes about 7-11 s (up to about 17 s on a busy machine): that is the first frame after Filament's context is
made, which the first session or view of a process pays, and again the next one after the last was closed; the first
frame of another image size takes about 1-1.5 s more (the photo pipeline renders a slightly larger image per camera; its
first `top` picture took about 4-19 s here). Then a 1280 x 960 frame takes about 0.45-0.65 s with depth (about 1.1-1.5 s
through the photo pipeline, 2.5-4 s on a busy machine) and a 640 x 480 frame about 0.25-0.35 s (0.35-0.55 s);
`open_view` takes about 5-40 s on the same scene and renders as fast. Until you give a state (`sync_from`, `set_state`,
or `state` in `view.render`), a session or view shows your twin's own pose (`qpos0`: the bodies where your spec puts
them). One Filament session per process at a time: close it before opening the next (a second one is refused). Closing
the last one also releases Filament's context and its memory once nothing refers to the closed session (about 0.5-1.7 GB
stays with the process), so the next session's or view's first frame takes the 7-11 s again. Your program may use the
layer during an episode too (it finds the hall next to `$ARB_GIVEN_DIR`); it then has 2 render threads (about 0.8-1.05 s
/ 1.3-3.7 s and 0.35-0.5 s / 0.45-0.65 s per frame, its first frame about 7-11 s), and the time counts against the
episode's wall clock. MuJoCo's classic renderer stays available: the render layer's MuJoCo backend
(`render.backends.get_backend("mujoco")`) and `mujoco.Renderer`.
