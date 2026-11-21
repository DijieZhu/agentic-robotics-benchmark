# Assembly cell API (`cell`), version strut-assembly-cell/1.0.0

Your program `/home/user/submission/assemble.py` exposes `solve(cell) -> None`. The grader runs it once per hidden
episode in a fresh process under a dedicated unprivileged account, from a per-run directory holding a root-owned
READ-ONLY copy of your submission folder (regular files only: symbolic links and `__pycache__` are not copied), with an
empty writable `$HOME` / `$TMPDIR`, a minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL=osmesa`,
`PYTHONPATH` = a read-only copy of `~/given/lib`, `ARB_GIVEN_DIR` = a read-only copy of `~/given`, BLAS threads 2) and the
launcher `python3 -B -E -s -c <bootstrap> <lib> assemble` (the bootstrap puts `<lib>` on `sys.path` and calls
`arm.assembly_info.client.main`, which imports `assemble` from the current directory), or `bash run.sh` when you ship one.
`cell` is a proxy (`arm.assembly_info.client.InfoCell`): every method is one JSON-lines request on the process's original
stdout, answered on stdin by the grader's physics server; `print()` goes to stderr.

Units on the API: metres, radians, quaternions `wxyz`; the cell frame = the bench-top centre, z up (see
`cell_manual.md`). The given part files are in millimetres. MuJoCo reads at most 200 000 faces from one STL file;
`parts/M0_strut_core.stl`, `parts/M1a_motor_module.stl` and `parts/M1b_motor_module.stl` have more, so a twin loads
each of them as several meshes or from a converted file.

| call | returns | notes |
|---|---|---|
| `describe()` | `cell_version`, `units`, `task` (text), `robot` (model, base pose, joint names / ranges, home posture `home_q`, tcp offset, gripper, force-sensor note), `control` (tick, rate limits, stiffness range, modes), `fastening` (the capture rules as text), `cameras` (name -> resolution, rgb / depth, nominal horizontal field of view, mount description), `camera_budget_frames`, `datums` (the bench board, the hand plate, the cradle board, the cradle's plate, seat spacing, rod height and nominal place), `items` (what is on the bench, as text), `trays` (the tray frame; per tray the plate size, every nest's name, kind and (x, y) in the tray frame, the tray's nominal place and turn), `budgets`, `safety` (the four limits and rules); every key: Replies, key by key (below) | the same in every episode (only `robot.home_q` follows the arm: it is the joint set-point `cmd_q` at the time of the call, the home posture until the arm first moves); no module poses, no mates, no tray or cradle pose, no cover colours; the reply itself: `describe_example.json`. `cell_manual.md` is the complete statement of the rules |
| `get_state()` | `sim_time`, `done`, `robot {q[7], dq[7], cmd_q[7], gripper_width, tcp_pos[3], tcp_quat[4], wrench[6], mode, stiffness_scale}` | joint angles, the tcp pose and the gripper width are exact; `wrench` (force N, torque N m, at the hand flange in the hand frame) carries white noise (~0.3 N / 0.02 N m rms per reading) and an unknown per-episode bias (up to ~1 N / 0.05 N m), and includes the hand's own weight: tare it |
| `set_control(mode, stiffness_scale)` | ok | `mode` in `joint`, `cartesian`; `stiffness_scale` in [0.2, 1.0] (a value outside is clamped) scales the seven joint position gains; an episode starts in `joint` mode at scale 1.0 |
| `command(q=[7])` or `command(tcp_pos=[3], tcp_quat=[4])`, optional `width` | ok / error | a joint target needs `joint` mode and a Cartesian target `cartesian` mode (otherwise `{"ok": false, "error": ...}`); `width` alone is accepted; targets are held until the next command; joint targets are clamped to the ranges and rate-limited to 1.5 rad/s; Cartesian targets go through the cell's damped-least-squares differential IK on the `tcp` site (<= 0.5 m/s, 1.5 rad/s, joint rate limit); the gripper `width` (m, 0..0.08) ramps at 0.05 m/s; grip force = 2500 N/m x (object width - target width) / 2 per finger, <= 60 N |
| `step(n_ticks)` | `sim_time`, `done`, `events[]` | advances n x 10 ms of physics (1 kHz inside); nothing moves outside `step`; n <= 500 per call (a larger n is cut to 500); events: only `safety:<what>` (`robot_contact`, `part_force`, `rod_load`, `part_speed`, each followed by the cell's names of the bodies and the reading) and `episode_over` (no "fastened", "grasped" or "dropped" events) |
| `get_camera(name, depth=True)` | `ok`, `camera`, `frame_index`, `width`, `height`, `format` (`png`), `rgb_png_b64`, and with depth `sensor_depth_u16_mm_b64`, `depth_unit_m` (0.001), `depth_invalid_value` (0) | one frame of `top` or `wrist` (both RGB-D) rendered at the current physics state: the PNG colour image and the depth sensor's 16-bit PNG (depth along the optical axis, millimetres, registered to the colour pixels, 0 = no reading); no intrinsics, no pose, no exact depth; at most 600 frames per episode over all cameras, then `{"error": "camera_budget"}`. Rendering counts against the episode's wall-clock cap: measured in the grader's container on 8 CPUs a `top` frame with depth takes about 1.7-1.9 s and a `wrist` frame with depth about 1.2-1.4 s (medians; single frames up to about 2.1 s), a frame without depth about 0.2-0.6 s less, and the first frame of an episode about 15-20 s (the render session is built) |
| `finish()` | ok | ends the episode; the grader then reads which mates the cell has fastened |

`arm.assembly_info.client.decode_frame(reply)` returns `(rgb uint8 H x W x 3, depth float32 H x W metres, NaN = no reading)`
(`depth` is `None` for a frame taken without depth).

Anything else answers `{"error": "forbidden"}`, counts as an illegal call and does nothing. In particular
`get_state_privileged()`, `describe_privileged()` and `reference_assemble()` (exact module poses, mate status, the mating
graph, the grader's reference program), `reset(seed)` and `truth()` are not available to your program.

## Replies, key by key

On the wire every reply is one JSON line `{"id": <the request's id>, "result": <reply>}`, and the proxy returns the
`result`. A reply `{"id": ..., "error": <text>}` makes the proxy raise `arm.assembly_cell.client.CellError(<text>)`
instead (the texts: `error` below). A refused `command` or `step` is an ordinary reply with `ok` false and an `error`
text. After the simulated-time cap, `step` answers `done` true with the event `episode_over` and `command` answers `ok`
false with the error `episode_over`. After `finish()`, the wall-clock cap, a silence or some 50 `step` calls past the
simulated-time cap the cell answers no more calls: the next one raises `CellError("the cell closed the connection")`.

`describe_example.json` next to this file is the `describe()` reply as the cell gives it at the start of an episode
(keys sorted, indented). It is the same in every episode: the cradle's and the trays' places in it are their nominal
places, and where they lie in an episode is in no reply and has to be measured; the nests' places on their trays
(`x_in_tray_m`, `y_in_tray_m`) hold in every episode.

The schema has one line per key, `key: type; unit, frame and meaning`, indented by two spaces per level of nesting;
`a, b:` are two keys of the same type; `float[n]` is a list of n numbers and `float[n][m]` a list of n such lists;
`str[]` and `dict[]` are lists of any length (the lines under a `dict[]` are the keys of each element); `key?` is a key
that only some replies carry. Positions and orientations are in the cell frame where no other frame is named.

```text
every reply on the wire ->
  id: int; the request's id
  result?: dict; the reply of the call (one of those below): what the proxy returns
  error?: str; instead of result, and the proxy raises CellError with it: `forbidden` (a call not in the table above), `bad_params: <why>` (a parameter the call cannot take, such as an unknown mode or a q that is not seven numbers), `unknown camera (cameras: top, wrist)` (also for the proxy's default name overview), `camera_budget` or `camera_unavailable` (the renderer could not make the frame)

describe() ->
  cell_version: str; this cell's version
  units: dict; the units of the API
    angle, length, quaternion: str; rad, m, wxyz
  task: str; the task in words
  robot: dict
    model: str; the arm and its given model
    base_pos: float[3]; m, the robot base
    base_axes: str; the base's axes against the cell frame
    joints: str[7]; the joint names, in the order of every q
    joint_range: float[7][2]; rad, the lower and the upper limit of each joint
    home_q: float[7]; rad, the joint set-point at the time of the call: the home posture until the arm first moves
    tcp_site: str; the tool point's name in the given model
    tcp_offset_in_hand_m: float[3]; m, the tool point in the hand frame
    gripper: dict
      max_width_m: float; m, the widest opening
      rate_m_s: float; m/s, the rate of the width command
      pad_size_m: float[2]; m, the face of a fingertip pad
      pads: str; which pads are fitted
      force_model: str; the grip force in words
    gravity_compensated: bool
    ft_sensor: str; the force / torque sensor in words
  control: dict
    tick_s: float; s, one control tick (what step counts)
    joint_rate_limit_rad_s: float; rad/s, the joint mode's rate limit
    cartesian_speed_limit_m_s: float; m/s, the Cartesian mode's speed limit of the tool point
    cartesian_angular_limit_rad_s: float; rad/s, the Cartesian mode's angular limit
    stiffness_scale_range: float[2]; the range of set_control's stiffness_scale
    modes: str[2]; the control modes
  fastening: dict
    rod, lid: str; the capture and fastening rules in words (onto a rod; a lid onto its bay)
  cameras: dict; one entry per camera
    top, wrist: dict
      resolution: int[2]; px, width and height
      rgb, depth: bool; the camera gives colour / depth
      nominal_hfov_deg: int; deg, the nominal horizontal field of view
      mount: str; the approximate mount in words
  camera_budget_frames: int; frames per episode over all cameras
  datums: dict; the surveyed datums (cell_manual.md, Datum boards)
    frame: str; the cell frame in words
    bench_board: dict
      squares: int[2]; columns and rows
      square_mm, margin_mm, thickness_mm: float; mm
      inner_corner_cell_m: str; the inner corners' formula, cell frame, m
      black_corner_square: str; where its black corner square is
      lies_flat_on_bench: bool
    hand_plate: dict
      squares: int[2]; columns and rows
      square_mm, margin_mm, thickness_mm: float; mm
      inner_corner_hand_m: str; the inner corners' formula, hand frame, m
      black_corner_square: str; where its black corner square is
      mounting: str; where it is fixed on the hand
      mounting_tolerance: str; how well it is fixed
    cradle_board: dict
      squares: int[2]; columns and rows
      square_mm, margin_mm, thickness_mm: float; mm
      mounting: str; where it is fixed on the cradle
      note: str; that it moves with the cradle
    cradle: dict
      plate_m: float[3]; m, the plate's size along x, y and z
      seat_spacing_m: float; m, from one seat to the next along x
      rod_height_m: float; m, the rods' axis above the bench
      nominal_middle_seat_m: float[2]; m, the nominal (cx, cy), not the episode's
      tolerance_mm: float; mm, the cradle's placement range in x and in y
  items: str[2]; what is on the bench, in words
  trays: dict
    tray_frame: str; the tray frame in words
    trays: dict; one entry per tray
      tray_l, tray_r: dict
        plate_m: float[3]; m, the plate's size along the tray's x, y and z
        nominal_place: str; the tray's nominal centre and turn in words, not the episode's
        nests: dict[]; every nest of the tray
          nest: str; the nest's name
          kind: str; `motor`, `cap` or `lid`
          x_in_tray_m, y_in_tray_m: float; m, the nest's place in its tray's frame (the same in every episode)
  budgets: dict
    sim_time_cap_s, wall_clock_cap_s, silence_abort_s: float; s
    max_ticks_per_step, camera_frames: int
  safety: dict
    panda_force_N: float; N, rule R1
    part_force_N: float; N, rule R2
    part_force_window_s: float; s, rule R2
    rod_lateral_force_N: float; N, rule R3
    rod_window_s: float; s, rule R3
    part_speed_m_s: float; m/s, rule R4
    rules: str[4]; the four rules in words

get_state() ->
  sim_time: float; s, the episode's simulated time
  done: bool; true once the episode is over
  robot: dict
    q: float[7]; rad, the joint angles (exact)
    dq: float[7]; rad/s, the joint velocities
    cmd_q: float[7]; rad, the joints' position set-point: the rate-limited command that the position actuators track
    gripper_width: float; m, the opening between the fingers (exact)
    tcp_pos: float[3]; m, the tool point (exact)
    tcp_quat: float[4]; wxyz, the orientation of the hand frame (exact)
    wrench: float[6]; N then N m, force x, y, z and torque x, y, z at the flange, hand frame (noisy, biased, with the hand's weight)
    mode: str; `joint` or `cartesian`
    stiffness_scale: float; the scale in force

set_control(mode, stiffness_scale) ->
  ok: bool; true
  mode: str; the mode now in force
  stiffness_scale: float; the scale now in force (after clamping)

command(q or tcp_pos with tcp_quat, width) ->
  ok: bool; false when a target was refused
  error?: str; only with ok false: `q target needs joint mode`, `tcp target needs cartesian mode`, `tcp_pos and tcp_quat go together` or `episode_over`

step(n_ticks) ->
  sim_time: float; s, the simulated time after the step
  done: bool; true once the episode is over
  events: str[]; what happened during the step: `safety:robot_contact:panda:<body>:<force>N` (R1), `safety:part_force:<module>:<body>:<force>N` (R2), `safety:rod_load:<seat>:<end>:<force>N` (R3), `safety:part_speed:<module>:<speed>m/s` (R4), `episode_over`

step(n_ticks), refused ->
  ok: bool; false
  error: str; `n_ticks must be >= 1`

get_camera(name, depth=True) ->
  ok: bool; true
  camera: str; the camera's name
  frame_index: int; the frame's number in the episode, over all cameras, counted from zero
  width, height: int; px
  format: str; `png`
  rgb_png_b64: str; the colour image, a PNG file in base64
  sensor_depth_u16_mm_b64?: str; only with depth: the depth image, a PNG file in base64 (see the table above)
  depth_unit_m?: float; only with depth: m per unit of the depth image
  depth_invalid_value?: int; only with depth: the value of a pixel without a reading

finish() ->
  ok: bool; true
  sim_time: float; s, the simulated time at the end
```

## Picturing a model of your own

The given library holds the renderer of the cell's cameras: MuJoCo's Filament module (on the CPU) in the baked
laboratory of the render folder (`~/render`; in a graded episode `render` next to `$ARB_GIVEN_DIR`) with the cameras'
look. Two entries use it; one of them can be open in a process at a time (opening a second raises `RuntimeError` until
the first is closed).

| call | returns | notes |
|---|---|---|
| `arm.assembly_strut.agent_render.open_session(spec, exposure_factor=1.0, render_dir=None, content_files=(), lod="full", live_lamps=6, shadow_lamps=2)` | a render session (`render.backends.base.RenderSession`) | the code the cell renders its frames with, on a twin of yours: `spec` is a `mujoco.MjSpec` of the cell in the cell frame (copied, not changed); left out of it are geoms named `floor`, `backdrop_*`, `bench_leg_*`, `bench_apron_*`, `bench_shelf` and geoms in groups above 2, added are the steel workbench frame under the bench top and the laboratory, and your scene's lights are replaced by the room's (the `live_lamps` lamps nearest the bench light it, the `shadow_lamps` nearest of them cast shadows); a geom `table_top` of material `mat_table` gets the bench top's laminate and materials named as in `agent_render.make_look` the cameras' terms; `exposure_factor`: the room's brightness, nominal 1.0; `render_dir`: by default the render folder |
| `session.sync_from(model, data)`, `session.render_raw_views([cam], ("rgb", "depth"))`, `session.render([cam], photo=agent_render.photo_spec("top"), seed=k, outputs=("rgb", "sensor_depth"))`, `session.close()` | the session's state, `[RawView]` with `rgb` and `depth` and, with `"segmentation"` among the outputs, `seg` (int32 H x W: at each pixel the id in `session.model` of the geom seen there, -1 = nothing hit; `session.labels[id]` gives its `geom` and `body` names), a `render.spec.RenderResult` whose lists hold one entry per camera: `rgb` (uint8 H x W x 3) and, with `"sensor_depth"` among the outputs, `sensor_depth` (float32 H x W metres, 0 = no reading) | `sync_from` copies your physics state by joint name; `cam` is a `render.spec.CameraSpec` (`fx`, `fy`, `cx`, `cy`, `world_T_cam` = its pose in the cell frame with OpenCV axes, optional lens distortion), and `render.filament.view.camera_spec(name, session.model, session.data, width, height)` gives one for a camera of your twin; `render_raw_views`: the ideal pinhole picture (uint8, no lens distortion) and MuJoCo's exact depth of the same geometry, the laboratory's furniture included (float32 metres, 0 = nothing hit); `render` runs the render layer's pipeline: the camera's lens distortion (`k1`, `k2`, `p1`, `p2`) and, with `photo=agent_render.photo_spec(camera="top", unit_seed=None)`, the cameras' photo and depth-sensor models, the frame's image noise drawn from `seed` (nonzero; `render.spec.new_seed()` gives a fresh one) and a camera unit of nominal response for `unit_seed=None` (an integer draws a unit's colour response and fixed depth errors within the cell's spreads) |
| `render.filament.view.open_view(model, room=None, look=None, *, surface=None, bench_xy=(0.0, 0.0), max_live_lamps=6, lamp_range_m=40.0, keep_lights=False, exposure_factor=1.0, max_group=2, ...)` | a `FilamentView` | any MuJoCo scene of yours (an `MjSpec`, copied, or the path of an MJCF file) rendered exactly as given in the room `room` (`room` in the render folder) with the look `look` (a look file such as `arm/assembly_strut/agent_look.json`, a dict or a `Look`): its own floors and walls stay unless `drop_prefixes` names them and nothing of the cell is added; geoms in groups above `max_group` are left out and its lights are replaced by the room's (the `max_live_lamps` lamps nearest `bench_xy` light it, those within 1.6 m of it cast shadows); `view.render(state=None, camera=None, width=None, height=None, depth=False)` gives rgb (uint8) or with `depth=True` (rgb, depth float32 metres, 0 = nothing hit); `view.close()` |

A submitted program can use both entries during an episode: what the renderer prints when it starts goes to the
program's standard error, not to the wire.

A picture of a twin made with `open_session` and the nominal values differs from a frame of the cell at the same state
and camera in the room's brightness of the run (within about 20 % of `exposure_factor=1.0`), in the camera unit's
response (gain and tint within a few per cent, the depth unit's fixed errors), in the frame's image noise and in what
the twin does not know about the cell (the true layout); one made with `open_view` differs besides in the scenery the
scene brings or lacks and in its shadows. Measured in the task image on 8 CPUs with a twin of the whole cell made from
the given module files (about 3.4 million triangles) and the Panda: `open_session` takes about 6 s and holds about
6 GB; the first picture takes about 11 s, then a `top` picture (1280 x 960) with depth about 1.8 s and a `wrist`
picture (640 x 480) with depth about 1.3 s (medians); `open_view` of the same twin in the laboratory about 3-4 s for a
`top` picture (all six lamps cast shadows there). Closing a session or a view releases the renderer (Filament's
context) and most of its memory once nothing refers to the closed one (about 1.2 GB stays with the program after the
first one, about 3 GB after seven in a row), so the next session's or view's first picture takes about 11 s again.
In a graded episode your program shares the verifier's memory (24 GB) with the cell's own camera session, which opens
at your first `get_camera` call: measured in this cell, a program holding a twin session took about 8 GB and the cell's
own session about 6-8 GB (the cell's process reached about 10 GB by the third episode).
A program that needs pictures at several moments keeps one session or view open for all of them. Two pictures of one
state, camera and seed made in two processes are not always equal byte for byte either: a small share of their pixels
(measured: at most 0.2 % of the values) can differ by a few grey levels, mostly 1 or 2; the same holds for two frames
of the cell at one state in two runs. A picture also depends on how many pictures the session has made before it.

```python
import os
import mujoco
from PIL import Image
from arm.assembly_strut import agent_render
from render.filament.view import camera_spec
from render.spec import CameraSpec

spec = mujoco.MjSpec.from_file("twin.xml")                      # your twin of the cell, in the cell frame
model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
session = agent_render.open_session(spec)
session.sync_from(model, data)
top = CameraSpec.look_at("top", (0.0, 0.0, 1.25), (0.0, 0.0, 0.0), width=1280, height=960, hfov_deg=70)
wrist = camera_spec("wrist", session.model, session.data, 640, 480)    # a camera named wrist in your twin
for name, cam in (("top", top), ("wrist", wrist)):
    raw = session.render_raw_views([cam], ("rgb", "depth"))[0]
    # to HOME: a graded episode's working directory is read-only
    Image.fromarray(raw.rgb).save(os.path.join(os.path.expanduser("~"), name + ".png"))
session.close()
```

## Budgets and aborts

1200 s of simulated time and 2400 s of wall clock per episode (rendering included: 600 `top` frames with depth would
take about 1000-1150 s of it); 120 s of silence between two requests aborts the episode; a CPU-time backstop of 4 x the wall cap
applies per process; every process of the run's account is killed when the episode ends; a program that breaks the wire
(crash, malformed line, exit) ends ITS episode with what it achieved so far and the other episodes still run.
