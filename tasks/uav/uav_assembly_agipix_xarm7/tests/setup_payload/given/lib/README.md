# Rendering your own frames

MuJoCo's own renderer (`mujoco.Renderer`, or the render layer's `"mujoco"` backend) draws a different picture than the
cell's cameras (no workshop hall, MuJoCo's default lights, flat colours, no reflections), so image processing tuned on
its frames may behave differently on the cell's frames; this library is the way to see what the cell's cameras would
see.

The cell's cameras render RGB with MuJoCo's Filament renderer (physically based; on the CPU through Vulkan, Mesa's
lavapipe) in the workshop hall of `render/` (its baked lightmaps, its lamps, its HDR image light, the scanned oak bench
top), with the cell's look (light units, exposure, the AgX display transform, physically based terms per material
class). Depth comes from MuJoCo's renderer on the same model: metres along the optical axis, registered with the colour
pixels, 0 where nothing is hit. This folder holds that renderer. Put it on the import path in your session
(`sys.path.insert(0, os.path.expanduser("~/given/lib"))`); in the graded run the runner puts it after your
`submission/` folder, and it finds the kit metadata and the hall itself (beside its `given/` folder), so import it from
where it is given rather than copying it. Two ways in:

* `cell_render.open_session(spec, parts=None)`: the cell cameras' own render session for your twin of the cell (`spec`:
  your `mujoco.MjSpec` or MJCF file in the world frame of `given/cell.md`, copied). It leaves out the geoms in groups
  above 2 and every plane geom, gives the box geom named `table` (the bench top) the oak maps, and adds the bench's
  steel frame and the hall, whose lamps replace your scene's lights. Geoms named `visual_<label>_<i>` (and
  `visual_<label>_rotor_<i>` for a motor's rotor) get the kit's roughness / metallic of visual region `i`
  (`given/kit/kit.json`, `visual_parts`); `parts` maps each label to its kit part kind, and by default a label that is a
  kind, or a kind followed by `_<number>`, is that kind; the fixture box `fixture`, the staging stands `stand_*` and the
  xArm's materials have rules of their own (`cell_render.make_look`). `sync_from(model, data)` copies your physics state
  by joint name. `capture("top")` / `capture("wrist")` returns `{"rgb", "depth_m"}` (1280 x 960 / 640 x 480) the cell's
  way: the camera of that name in the model you last synced, its mount (`cam_pos`, `cam_quat`) and vertical field of
  view (`cam_fovy`) read at every capture, square pixels, the principal point at the image centre; so a calibration
  estimate written into your model's camera is used by the next capture. The RGB bytes are what the cell sends;
  `cell_render.sensor_depth(depth_m, rng)` adds what the cell's depth sensor does (0.1-4 m, about 3 % missing pixels,
  millimetres). `render_raw_views([cam], ("rgb", "depth"))` renders any `render.spec.CameraSpec`.
* `render.filament.view.open_view(spec_or_mjcf_path, room=..., look=..., surface=..., bench_xy=(0.0, 0.0))`: any scene
  of yours, rendered as given: nothing of the cell is added (no bench frame; the oak maps only on a geom `table` with
  the material `render_table_oak`, under `cell_render.make_look()`), only geoms in a group above 2 are left out; with
  `room`, the hall is added and its lamps replace your scene's lights (unless `keep_lights`). Take `room` (and
  `surface`, the oak maps) from `cell_render.render_dir()`: `os.path.join(cell_render.render_dir(), "room")` (and
  `"oak"`). That is the scenery folder in your session and in the graded run alike; there `HOME` is not your home, so a
  path built from `~` does not resolve. `view.render(state, camera, width=None, height=None, depth=False)` returns RGB
  uint8 H x W x 3 (with `depth=True` also depth float32 H x W); `state` is an `MjData` (joints copied by name), a
  `(model, data)` pair, a dict `{joint name: value}` or None; `camera` is a camera name of the model (with `width`,
  `height`), a `CameraSpec` or a dict `{"width", "height", "fx", "fy", "cx", "cy", "world_T_cam"}` (`"K"` may stand for
  the four intrinsics). `look` is the cell's look (`cell_render.make_look()`), a look file of your own (JSON: the fields
  of `render.filament.backend.Look`; `"materials"` a list of `[geom glob, material glob, {"roughness", "metallic"}]`
  rules, the first match wins) or None.

Cameras are exact pinholes (`render.spec.CameraSpec`: `fx`, `fy`, `cx`, `cy` in pixels, `world_T_cam` the pose in the
world frame with OpenCV axes: x right, y down, z forward).

```python
import os, sys
sys.path.insert(0, os.path.expanduser("~/given/lib"))
import cell_render
from render.filament.view import open_view
from render.spec import CameraSpec

session = cell_render.open_session(spec)              # spec: your MjSpec of the cell
session.sync_from(model, data)                        # model, data: your physics
frame = session.capture("top")                        # frame["rgb"], frame["depth_m"]
cam = CameraSpec.look_at("my_view", eye=(0.3, -0.5, 0.6), target=(0.08, 0, 0.18), width=640, height=480, hfov_deg=60.0)
raw = session.render_raw_views([cam], ("rgb", "depth"))[0]    # raw.rgb, raw.depth
session.close()                                       # one session per process at a time

view = open_view(part_spec, room=os.path.join(cell_render.render_dir(), "room"), look=cell_render.make_look())
rgb, depth = view.render(part_data, cam, depth=True)  # part_spec / part_data: a scene of yours and its MjData
view.close()
```

What it costs here (8 CPUs, a twin of the whole cell): `open_session` takes about 12 s, the display transform's lookup
table (about 2 s) included, and the open session holds about 5 GB (the process about 6.5 GB). In the graded run your
program shares the sandbox's 24 GB with the cell's own camera session in the grader's process: here that process held
about 6.5 GB in the first workcell and about 9.5 GB in the third, and a program rendering through a session of a twin of
the whole cell held up to about 7 GB, about 16 GB together. A session's first frame takes about 10-11 s: that is the
first frame after Filament's context is made, which the first session or view of a process pays, and again the next one
after the last was closed; the first frame of the other image size takes about 2.5 s. Then a 1280 x 960 frame takes
about 0.7-0.9 s with depth and a 640 x 480 frame about 0.4-0.6 s; `open_view` takes about 11-12 s on the same scene and
renders a little slower (about 0.8-1.1 s and 0.45-0.7 s: its depth comes from the render layer's MuJoCo session). Until
you give a state (`sync_from`, `set_state`, or `state` in `view.render`), a session or view shows your twin's own pose
(`qpos0`: the bodies where your spec puts them). One Filament session per process at a time: close it before opening the
next (a second one is refused). Closing the last one also releases Filament's context and most of its memory once
nothing refers to the closed session, so the next session's or view's first frame takes the 10-11 s again. What stays
adds up: with `LP_NUM_THREADS` unset, as in the graded run, the process keeps about 0.5 GB after the first close and
about 0.2-1.2 GB more after each further open / close cycle, varying from run to run (about 1.8-2.8 GB after the fourth
close here), while a session kept open renders frame after frame without growing. So open one session (or view) for the
whole program and sync it to each new state, rather than one per frame or per step. With `LP_NUM_THREADS=8` set before
the first session the growth slowed here (about 1.1-1.8 GB kept after four cycles). During an episode Mesa's lavapipe
picks its own number of render threads (the times above hold there too; set `LP_NUM_THREADS` before the first session to
choose), and rendering counts against the episode's wall clock.
