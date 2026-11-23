# Station cell API (`cell`), version parcelcell-3.0

Your program `/home/user/submission/sort.py` exposes `solve(cell) -> None`. The grader runs it once per hidden shift in
a fresh process (how it is run: `instruction.md`). `cell` is a proxy (`parcelcell.client.Cell`): every method is one
JSON-lines request on the process's original standard output, answered on its standard input by the grader's cell.
`print()` goes to the standard error. A reply carrying an error raises `parcelcell.client.CellError` (with `.code` and
`.detail`). When `solve` returns or raises, the launcher calls `finish()`: the shift ends there (`cell_manual.md`
section 8).

Units: metres, radians, seconds, newtons; the world frame of `cell_manual.md` section 1. The robot's joint positions and
speeds are exact on the wire; its torque readings carry the drives' sensor noise (`cell_manual.md` section 4). **Nothing
about a parcel is**: no pose, size, kind, mass, label face, count or event, and no score. You see the parcels through
the cameras only: there is no line signal (no belt speed, no photo-eye).

## Calls

| call | returns | notes |
|---|---|---|
| `describe()` | `cell_version`, `units`, `tick_s` (0.02), `physics_step_s`; `robot` {`joints` (14 names, the order of every vector), `joint_range`, `torque_limit_Nm`, `command_speed_limit_rad_s` (6), `command`, `start_posture`, `flange_sites`, `T_flange_tcp` {`left`, `right`: 4x4}, `tcp`, `tool_pose` {`frame`, `command`, `solver`, `gravity_correction`, `reached_m`, `reached_deg`}}; `gripper` {`command_range`, `max_opening_m`, `opening_m`, `object_detected`, `free_opening_m`, `free_opening_at_command`}; `station` {`belt` {`top_z`, `centre_x`, `width`, `direction`, `frame_half_length`, `work_window_half_length`}, `lines` {`entry_y` (the set-down range's upstream end), `scan_y`, `exit_y`}, `hardware` {name: {`centre`, `half_size`}}, `floor_z`}; `cameras` {name: {`width`, `height`, `fx`, `fy`, `cx`, `cy`, `K`, `fovy_deg`, `depth_range_m`, `convention`, `exposure_period_s`, `frame_age_s`, `mount`, and `T_world_cam` (fixed cameras) or `flange_site` and `T_flange_cam` (wrist cameras): the installation drawing's poses, `installation` {`position_m`, `angle_deg`, `frame`, `focal_length_rel`, `note`} (the installation's tolerances)}}; `sensor_models`; `shift` {`duration_s`, `end`, `arrivals`, `list`, `due`, `belt` (the shift's rules in words), `indexing_zone_y`, `due_margin_s`}; `budgets` {`program_time_s`, `silence_s`, `frames`, `frames_pooled`, `max_ticks_per_step`, `after_over`}; `safety` {`structure_N`, `self_N`, `belt_N`, `press_ticks`, `stop_lock_s`, `damage_N`, `max_stops`, `squeeze_N`, `squeeze_ticks`, `damage_look` (what a damaged parcel looks like in the cameras)}; `score` {`label_tilt_deg`, `max_angular_speed_rad_s`, `max_rel_speed_m_s`, `rest_window_s`, `rest_clearance_max_m`, `readable` (the scan line's rule in words), `duty` {`kinds`, `label_positions`}, `classes`, `done`, `lost`, `credit`, `shift_score`} | static for the shift; free (no simulated time, no frame) |
| `get_state()` | `t` (s), `q`[14], `dq`[14], `tau`[14] (the drives' motor torque readings, N m, with the sensor's noise), `q_target`[14] (the applied target, after the speed limit), `tool` {`left`, `right`: {`pos` (x, y, z), `quat` (w, x, y, z)}} (each tool centre point's measured pose in the world frame: forward kinematics of the measured joints), `tool_target` {`left`, `right`: `null` (the arm is at the joint level) or {`pos`, `quat`, `reached`, `out_of_reach`, `error_m`, `error_deg`} (its tool-pose target and how far the controller's solution is from it, `cell_manual.md` section 4)}, `gripper` {`command`[2], `opening_m`[2] (m), `object_detected`[2] (true: the fingers stopped short of the command's free opening and stand still, as the real gripper reports)} (left, right), `stop` {`active`, `until`, `count`, `locked_out`, `link`}, `frames_left`, `program_time_left` (s), `shift_over`, `over_reason` (why the shift ended: `shift_end`, `finished`, `program_time`, `stalled`, `protocol`, `crashed`, `physics_fault` or `internal_error`, `cell_manual.md` section 8; `null` while it runs) | free: no simulated time, no frame |
| `step(q_target=None, grip=None, n=1, tool_pose=None)` | the state (as `get_state()`) after the ticks, plus `ticks` (how many ran) and `refused` (`null`, `"protective_stop"` or `"locked_out"`) | runs n ticks of 20 ms (at most 50 per call; a larger n is cut) holding the targets: `q_target` = 14 joint position targets (rad, `robot.joints` order), or {`left`: 7 values, `right`: 7 values} for one arm or both, or `null` to keep the previous ones; `tool_pose` = {`left`: {`pos`: [x, y, z], `quat`: [w, x, y, z]}, `right`: ...} (either arm, or `null` for an arm), a target pose of that arm's tool centre point in the world frame that the controller follows by solving the arm's joints every tick (`cell_manual.md` section 4), or `null` to keep; an arm cannot get joint targets and a tool pose in the same call; a target beyond reach is approached as far as the arm goes, and the hand may then be displaced from it in any direction by several centimetres (a parcel held between the hands included: `cell_manual.md` section 4), which `tool_target.out_of_reach` and `error_m` report at every step; `grip` = [left, right] in 0 (open) .. 1 (closed) or `null` to keep. Each tick the applied target moves at most 6 rad/s x 0.02 s toward the commanded one; targets are clipped to `joint_range`, grips to [0, 1]. During a protective stop or a lock-out a call that carries `q_target`, `tool_pose` or `grip` is refused: none is taken, and the n ticks still run with the frozen targets. Once the shift is over: `ticks: 0`, nothing moves |
| `get_camera(name, depth=False)` | `camera`, `frame` (the exposure's tick index), `stamp` (the exposure time, s), `width`, `height`, `format` (`png`), `rgb_png_b64`; with `depth=true` also `depth_png_b64` (16-bit, millimetres), `depth_unit_m` (0.001), `depth_invalid` (0); for a wrist camera `T_world_cam` (4x4, OpenCV axes, at the exposure: the drawing's mount on the flange, `cell_manual.md` section 6); `repeat` (true when this frame was delivered before: free), `frames_left` | `name`: `infeed`, `head`, `wrist_left`, `wrist_right`. The newest delivered frame (`cell_manual.md` section 6): exposed every 40 ms, delivered 60 ms later. A new frame costs one of the shift's pooled frames; a repeat, or adding depth to a delivered frame, costs nothing. Takes no simulated time and no program time |
| `finish()` | `ok`, `t` | ends the shift now; parcels not yet done earn nothing (`cell_manual.md` section 8) |

`parcelcell.client.decode_frame(reply)` returns `(rgb, depth)`: `rgb` uint8 H x W x 3, `depth` float32 H x W in metres
with NaN where there is no reading (or `None` without depth). `decode_rgb` and `decode_depth` return one each.

## Errors

| error | when | what changes |
|---|---|---|
| `bad_params` (with `detail`) | an argument fails its check: numbers must be JSON numbers (not booleans), finite and at most 1e+06 in magnitude; `n` a whole number of at least 1; vectors of exactly 14 or 2 numbers, or `null`; per-arm objects with the keys `left` and `right` only, 7 joint targets an arm; a tool pose with `pos` (x, y, z) and `quat` (w, x, y, z, of unit length within 0.001); joint targets and a tool pose for the same arm in one call; `depth` true or false; a camera name (a string of at most 64 characters); no unknown argument names | nothing: no tick runs, no frame is charged |
| `no_frames_left` | the shift's frame budget is used up | nothing; the shift goes on |
| `no_frame_yet` | no frame delivered yet (does not happen in a shift: the cameras already run during the settle before t = 0) | nothing |
| `camera_unavailable` | the cell could not render the frame | nothing, the frame is not charged; a cell failure (below) |
| `forbidden` | a call that is not in the list above | nothing; counted as an illegal call |
| `internal_error` | the cell failed while answering | inside `step` or `finish` it ends the shift; a cell failure (below) |

**Cell failures.** When the cell failed in a shift (an internal error, an unstable physics state, a frame it could not
render), the grader runs that shift once more from scratch, with a fresh cell and a fresh process of your program, and
the second run counts.

## Budgets (per shift, `describe()["budgets"]`)

The shift's duration of simulated time (the clock advances only inside `step`); the shift's duration + 60 s of **program
time** (the shift's wall clock minus the cell's own time inside `step` and `get_camera`), 4 x that in CPU seconds; at
most 20 s of program time from a reply to your next request; 2 frames per second of the shift's duration, pooled over
the four cameras; at most 50 ticks per `step`; a request line of at most 4 MiB. After the shift is over the cell answers
at most 50 steps, 200 requests and 60 s more. From the program's start to its first request: at most 120 s (the start
window), inside the program-time budget.

## A worked example

```python
from parcelcell.client import decode_frame

def solve(cell):
    info = cell.describe()
    q0 = info["robot"]["start_posture"]                  # 14 values, info["robot"]["joints"] order
    K = info["cameras"]["infeed"]["K"]
    T = info["cameras"]["infeed"]["T_world_cam"]         # camera to world, OpenCV axes
    st = cell.get_state()
    print("shift", info["shift"]["duration_s"], "s; left tool at", st["tool"]["left"]["pos"])
    f = cell.get_camera("infeed", depth=True)            # newest delivered frame
    rgb, depth = decode_frame(f)                         # (480, 640, 3) uint8, (480, 640) float32 m
    print("frame exposed at", f["stamp"], "s; frames left", f["frames_left"])
    q = list(q0)
    q[3] -= 0.3                                          # left elbow: the cell ramps it at the speed limit
    st = cell.step(q_target=q, grip=[0.0, 0.5], n=10)    # 10 ticks = 0.2 s
    print(st["t"], st["q_target"][3], st["gripper"]["opening_m"], st["refused"])
    up = dict(st["tool"]["left"])
    up["pos"] = [up["pos"][0], up["pos"][1], up["pos"][2] + 0.05]   # the left tool 5 cm up, same orientation
    st = cell.step(tool_pose={"left": up}, n=25)         # the controller solves the left arm's joints each tick
    print(st["tool_target"]["left"]["reached"], st["tool_target"]["left"]["error_m"], st["tool"]["left"]["pos"])
    while not st["shift_over"]:
        st = cell.step(n=50)                             # hold the targets, 1 s per call
```

## Commissioning (`cell_manual.md` section 12; development only)

The commissioning service answers the same calls on the development shifts, plus `reset(site)`, the first call of a
connection: it starts the connection's one shift on development shift `site` (1 to 8).
`parcelcell.client.connect_commissioning(site)` opens a connection and calls it. Errors: `bad_params` (no such shift),
`no_shift` (a call before `reset`), `one_shift_per_connection`, `commissioning_window_used` (every shift of the window
is used), `busy` (as many shifts open as the service takes at a time: the connection's first request gets it and the
connection closes; no shift is used). When the service closes a shift for any reason but `finished`, it sends `{"id":
null, "event": "shift_over", "reason": ...}` and closes the connection (the client raises `CellError`). The window's
count: `/run/parcelcell/commission_status.json`. Nothing else differs from the graded cell: no truth, no score, no
debrief.
