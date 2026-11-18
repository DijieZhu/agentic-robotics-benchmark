# Assembly cell API (`cell`), version racecar-assembly-cell/1.0.0

Your program `/home/user/submission/assemble.py` exposes `solve(cell) -> None`. The grader runs it once per hidden
episode in a fresh process under a dedicated unprivileged account, from a per-run directory holding a root-owned
READ-ONLY copy of your submission folder (regular files only: symbolic links and `__pycache__` are not copied), with
an empty writable `$HOME` / `$TMPDIR`, a minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL` set for
off-screen rendering, `PYTHONPATH` = a read-only copy of `~/given/lib`, `ARB_GIVEN_DIR` = a read-only copy of
`~/given`, BLAS threads 2) and the launcher `python3 -B -E -s -c <bootstrap> <lib> assemble` (the bootstrap puts
`<lib>` on `sys.path` and calls `arm.assembly_info.client.main`, which imports `assemble` from the current directory),
or `bash run.sh` when you ship one. The process cannot open an Internet socket: creating one fails. `cell` is a proxy
(`arm.assembly_info.client.InfoCell`): every method is one JSON-lines request on the process's original stdout,
answered on stdin by the grader's physics server; `print()` goes to stderr.

Units on the API: metres, radians, quaternions `wxyz`; the cell frame = on the bench top at the arm's base axis, z up
(see `cell_manual.md`). The given part files are in millimetres.

| call | returns | notes |
|---|---|---|
| `describe()` | `robot` (model, base pose and axes, joint names / ranges, home posture, torque limits, tcp site and offset, tcp axes, gripper with its force model, wrist housing, force-sensor note), `control` (tick, rate limits, stiffness range, modes), `fastening` (the rules, every place's windows, lock tolerance and weight, the total weight), `cameras` (name -> resolution, rgb / depth, nominal horizontal field of view, mount description), `camera_budget_frames`, `datums` (the documented boards; the nominal places of the work stand, the deck shelf and the parts tray), `items` (what is in the cell, as text), `budgets`, `safety` | static for the episode; no unit poses, no hub angles, no fixture poses, nothing about what is fastened |
| `get_state()` | `sim_time`, `done`, `robot {q[6], dq[6], cmd_q[6], gripper_width, tcp_pos[3], tcp_quat[4], wrench[6], mode, stiffness_scale}` | joint angles, the tcp pose and the gripper width are exact; `wrench` (force N, torque N m, at the tool flange's centre in the flange frame) carries white noise (about 0.3 N / 0.02 N m rms per reading) and an unknown per-episode bias (up to about 1 N / 0.05 N m), and includes the gripper's own weight: tare it |
| `set_control(mode, stiffness_scale)` | ok | `mode` in `joint`, `cartesian`; `stiffness_scale` in [0.2, 1.0] scales the six joint position gains (the damping with its square root); pass both: `set_control("cartesian")` alone resets the scale to 1.0. Entering Cartesian mode takes its goal from the COMMANDED configuration |
| `command(q=[6])` or `command(tcp_pos=[3], tcp_quat=[4])`, optional `width` | ok / error | targets are held until the next command; joint targets are clamped to the ranges and rate-limited to 1.5 rad/s; Cartesian targets go through the cell's damped-least-squares differential IK on the `ur/tcp` site (at most 0.5 m/s and 1.5 rad/s, and the joint rate limit; no null-space control: it slows near the arm's singularities); the gripper `width` (m, 0 .. 0.085) ramps at 0.10 m/s (an opening command releases at once); grip force as `describe()["robot"]["gripper"]["force_model"]`; a target of the wrong length or otherwise malformed answers `bad_params` |
| `step(n_ticks)` | `sim_time`, `done`, `events[]` | advances n x 10 ms of physics (1000 Hz inside, the arm's computed-torque control recomputed every physics step: the arm holds its posture against gravity); nothing moves outside `step`; n <= 500 per call; events: only `safety:<what>` (a safety violation: the episode goes on, but keeps only 30 % of its score) and `episode_over`; nothing reports a capture, a fastening, a grasp, a dropped or a scrapped unit |
| `get_camera(name, depth=True)` | `camera`, `frame_index`, `width`, `height`, `format` (`png`), `rgb_png_b64`, and with depth `sensor_depth_u16_mm_b64`, `depth_unit_m`, `depth_invalid_value` | one frame of `top` or `wrist` (both RGB-D) rendered at the current physics state: the PNG colour image and the depth sensor's 16-bit PNG (depth along the optical axis, in steps of `depth_unit_m`, registered to the colour pixels, `depth_invalid_value` = no reading); no intrinsics, no pose, no exact depth; at most 450 frames per episode over all cameras, then `{"error": "camera_budget"}`; rendering takes about 0.4-1.2 s of wall clock per frame (an episode's first frame about 14-15 s more: the renderer's session opens then) and counts against the episode's wall-clock cap |
| `finish()` | ok | ends the episode at once (a unit still being pulled home does not fasten); the grader then reads which places the cell has fastened |

`arm.assembly_info.client.decode_frame(reply)` returns `(rgb, depth)`: `rgb` the colour image as a uint8 array (height
x width x three channels), `depth` a float32 array (height x width, metres, NaN = no reading).

The arm and the gripper are the given models built the way the cell builds them: `arm.assembly_quad.ur_robot.add_ur`
(on an `mujoco.MjSpec`; the gripper on the real flange face, the pinch-point tcp, the camera housing) and
`arm.assembly_quad.ur_controller.URController` (the cell's controller: the same API as above in your own simulation).
The package is named after the cell it was first built for; arm, gripper and controller are the same here.

Anything else answers `{"error": "forbidden"}`, counts as an illegal call and does nothing. In particular
`get_state_privileged()`, `describe_privileged()` and `reference_assemble()` (exact unit poses, the places' states,
the grader's reference program), `reset(seed)` and `truth()` are not available to your program.

## Budgets and aborts

650 s of simulated time and 3100 s of wall clock per episode (rendering included); 120 s of silence aborts the
episode: the time from the program's start, or from the cell's reply to a request, until the program's next request
(the cell's own time inside a request does not count; any request resets it, `get_state()` being the cheapest); a
CPU-time backstop of 4 x the wall cap applies per process; every process of the run's account is killed when the
episode ends; a program that breaks the wire (crash, malformed line, exit) ends ITS episode with what it achieved so
far and the other episodes still run.

## Rendering your own frames

MuJoCo's own renderer (`mujoco.Renderer`, or the render layer's `"mujoco"` backend) draws a different picture than the
cell's cameras (no hall, MuJoCo's own lights, flat colours, no reflections), so image processing tuned on its frames
may behave differently on the cell's frames; the given render layer is the way to see what the cell's cameras would
see.

The given render layer renders like the cell's cameras: MuJoCo's Filament renderer (physically based; on the CPU
through Vulkan, Mesa's lavapipe) in the factory hall of `~/render` (its baked lightmaps, its lamps, its HDR image
light, the scanned oak bench top), with the cell's look (`arm/assembly_car/agent_look.json` and
`agent_render.make_look()`: light units, exposure, the filmic display transform, physically based terms per material
class). Depth comes from MuJoCo's renderer on the same geometry: metres along the optical axis, registered with the
colour pixels, 0 where nothing is hit. Cameras are exact pinholes (`render.spec.CameraSpec`: `fx`, `fy`, `cx`, `cy` in
pixels, `world_T_cam` the pose in the cell frame with OpenCV axes: x right, y down, z forward); lens distortion,
noise, blur, vignetting, colour response and the depth-sensor model are the render layer's photo pipeline
(`render.spec.PhotoSpec`), the cell's own code. Two ways in (put `~/given/lib` on `sys.path`):

* `arm.assembly_car.agent_render.open_session(spec, exposure_factor=1.0, kit=None)`: the cell cameras' own render
  session for your twin of this cell (`spec`: your `mujoco.MjSpec` in the cell frame, copied). Name the bench top geom
  `table_top` with material `mat_bench` for the oak top; geoms named `floor`, `backdrop_*`, `bench_leg_*`,
  `bench_rail_*`, `bench_shelf` and every geom in a group above two are left out; the robot table's steel frame, the
  control box with its pendant, the floor tape and the hall are added, and the hall's lamps replace your scene's
  lights. The look's material classes are named in `agent_render.make_look` (the arm's, the fixtures' `mat_stand`,
  `mat_shelf`, `mat_tray`, the boards', the props'); materials named `mat_vis_<key>` take metallic and roughness from
  a material table `kit["materials"][<key>]`: by default the one in `~/given/kit.json` (`agent_render.given_kit()`:
  the car model's materials by key, with the terms the cell's cameras render them with), else the `kit` you pass
  (`kit={}`: none); a material without an entry gets the look's default terms. `exposure_factor` stands for the hall's
  brightness (the cell's own varies by up to about 20 % between runs). The session follows the render layer's
  interface: `sync_from(model, data)` copies your physics state by joint name,
  `render_raw_views([cam], ("rgb", "depth"))` gives the pinhole images,
  `render([cam], photo=..., seed=k, outputs=("rgb", "sensor_depth"))` runs the photo pipeline.
* `render.filament.view.open_view(spec_or_mjcf_path, room=..., look=..., surface=..., bench_xy=..., exposure_factor=1.0)`:
  any scene of yours, rendered exactly as given: none of the cell's props are added (no robot table frame, control box
  or floor tape), your scene's own floor and walls stay unless `drop_prefixes` names them, and only geoms in a group
  above two are left out; with `room`, the hall is added and its lamps replace your scene's lights (unless
  `keep_lights`). A twin of this cell goes through `agent_render.open_session` instead.
  `view.render(state, camera, width=None, height=None, depth=False)` returns an RGB uint8 array (H x W, three
  channels; with `depth=True` also depth float32 H x W); `state` is an `MjData` (joints copied by name), a
  `(model, data)` pair, a dict `{joint name: value}` or None; `camera` is a camera name of the model (with `width`,
  `height`), a `CameraSpec` or a dict `{"width", "height", "fx", "fy", "cx", "cy", "world_T_cam"}` (`"K"` (a
  three-by-three matrix) may stand for the four intrinsics). `look` is the cell's look (`agent_render.make_look()`), a
  look file of your own (JSON: the fields of `render.filament.backend.Look`; `"materials"` a list of
  `[geom glob, material glob, {"roughness", "metallic"}]` rules, the first match wins) or None. For the hall's
  brightness pass `exposure_factor=...` to `open_view` and leave `make_look()` at its default (a view would render
  `make_look(f)` with a wrong display gain). `view.session` is a render-layer session for the photo pipeline.

A twin of the car from the part files: the part files carry no materials; `~/given/kit.json` says which material each
of their triangles has and what each material looks like (appearance only). `part_materials` lists for each part file
its runs `[first triangle, number of triangles, material key]` in the file's triangle order, together covering the
file; `materials[<key>]` holds the material's `rgba` (its base colour) next to its `metallic` and `roughness`, and a
key with a `texture` (the printed boards of the deck) also names a `texture_mesh`, an OBJ file in
`~/given/appearance/` with the same triangles as that key's run, with texture coordinates, in the part file's frame
and millimetres. Split each part mesh into its runs (the runs of one key may share one mesh), give each a material
named `mat_vis_<key>` with that `rgba` (for a textured key take its `texture_mesh` instead of the run, and its image
as the material's colour map: a two-dimensional texture in sRGB, the RGB role), scale the meshes from millimetres to
metres, set their inertia to `shell` (the surfaces are open; make the geoms massless, without contacts, in group two)
and pose each part file on its body (`roller.stl` is in the car's own frame, its wheel hubs at the drawn angle).
`agent_render.open_session` then renders the car as the cell's cameras do.

An example, with `~/given/lib` on `sys.path`, `spec` your `MjSpec` of the cell, `model` / `data` your physics and
`cam` a `CameraSpec` (`CameraSpec.look_at(name, eye=..., target=..., width=..., height=..., hfov_deg=...)`):
`from arm.assembly_car import agent_render`; `session = agent_render.open_session(spec)`;
`session.sync_from(model, data)`; `raw = session.render_raw_views([cam], ("rgb", "depth"))[0]` (`raw.rgb`,
`raw.depth`); `session.close()`; then, for a scene of yours that is not the cell twin (`part_spec`, its `part_data`),
`from render.filament.view import open_view`;
`view = open_view(part_spec, room=os.path.expanduser("~/render/room"), surface=os.path.expanduser("~/render/oak"), look=agent_render.make_look(), bench_xy=agent_render.DEFAULT_BENCH["centre_xy"])`;
`rgb, depth = view.render(part_data, cam, depth=True)`;
`rgb_w = view.render({"my_hinge": angle}, "my_camera", width, height)` (a joint and a camera of `part_spec`);
`view.close()`.

What it costs here (eight CPUs, a twin of the whole cell): `open_session` takes about 4-11 s, the display
transform's lookup table (about 2 s) included, and the open session holds about 4 GB on top of your twin. Its
first frame takes about 7-10 s: that is the first frame after Filament's context is made, which the first session
or view of a process pays, and again the next one after the last was closed; the first frame of another image size
takes about 0.8 s more. Then a 1280 x 960 frame takes about 0.7 s with depth (1.3 s through the photo
pipeline) and a 640 x 480 frame about 0.4 s (0.5 s); `open_view` takes about 4 s on the same scene and
renders as fast. Until you give a state (`sync_from`, `set_state`, or `state` in `view.render`), a session or view
shows your twin's own pose (its initial joint positions: the bodies where your spec puts them). One Filament session
per process at a time: close it before opening the next (a second one is refused). Closing the last one also releases
Filament's context and its memory once nothing refers to the closed session (about 0.7 GB stays with the process, up
to about 1.7 GB after several opens), so the next session's or view's first frame takes that long again. Your
program may use the layer during an episode too (it finds the hall next to `$ARB_GIVEN_DIR`); it then has two render
threads (about 1.1 s / 2.1 s and 0.5 s / 0.7 s per frame, its first frame about 10 s), and the time counts
against the episode's wall clock. MuJoCo's classic renderer stays available: the render layer's MuJoCo backend
(`render.backends.get_backend("mujoco")`) and `mujoco.Renderer`.
