# Turn every parcel's shipping label face up on a running belt with a dual-arm robot, seen only through its cameras

You are the automation engineer of a parcel sorting hub. Its label-facing station sits just ahead of the line's
top-reading scanner: parcels come on a belt one at a time, and the scanner can read a parcel only when its shipping
label (the waybill) faces up as the parcel passes under it, lying calmly on the belt. A fixed dual-arm robot
stands beside the belt: a column with two arms of {{n_arm}} joints each, a parallel gripper on each wrist, a camera on
its head and one in each gripper's palm; one more camera looks straight down on the arriving parcels. Every parcel is the station's work (`cell_manual.md`
section 8): {{duty_kinds}} alike, whatever face carries the label, the bottom and the four sides included. The line sets each parcel
down on the belt's centre line upstream of the station, at any yaw, and the next one only once the one before it has
been settled (judged at the scan line, fallen off the belt or damaged). The station's rated cycle allows each parcel
at most {{due_margin_s}} s of delay against the belt's own ride to the scan line, and a shift's list is sized at
{{rated_handling_s}} s per parcel (`cell_manual.md` section 7). Your program must turn each parcel whose label is not up so that its label faces
up and let it travel on before it reaches the scan line, and leave every parcel whose label is already up readable.
The parcels' sizes, kinds, weights, centres of mass, surface friction and the face that carries the label are not
known in advance; the belt's speed differs from shift to shift, and on some lines the belt indexes: it stands while
the parcel is in front of the robot and runs again once the parcel has left that zone. Your program sees the station
only through its {{n_cameras_word}} cameras and the robot's own joint, tool and gripper state: there is no line signal
(no belt speed, no photo-eye, no belt mode), so the belt's speed and mode and the parcels' arrivals are for your
program to find from its cameras. Nothing on the API tells you where a parcel is, how large it is, which face carries its label or what credit you have earned: you must perceive it. The cameras'
optics are the data sheet's and their poses the installation drawing's, both published; the installed cameras differ
from them within stated tolerances and are to be calibrated on site. The robot's model files carry the maker's data:
exact kinematics, nominal dynamics, no servo gains.

## What you produce (in `/home/user/submission/`)

`sort.py` exposing `solve(cell) -> None` (plus any helper modules, optional `run.sh`). The grader runs it once per
hidden shift ({{shifts_graded}} shifts, each with its own parcels and belt speed) in its own cell, unattended. It talks
to the cell through `cell` (API in `~/given/cell/cell_api.md`): the robot's own state (joint positions and speeds,
the motor torque readings, the applied targets, each tool centre point's measured pose, the grippers' command, opening and
object-detected flag, the protective stop), the {{n_cameras_word}} cameras ({{camera_names}}) and
`step`, which advances the station by whole ticks of {{tick_ms}} ms while holding your targets: per arm either joint
targets or a target pose of its tool centre point, which the station's controller then follows by solving the arm's
kinematics itself; nothing else about the station.
Per shift: at most the shift's duration of simulated time (`describe()["shift"]["duration_s"]`: {{shift_s}} s for every
graded shift, whatever its belt speed and its parcels; the shift ends earlier, at the tick in which the last parcel of
its list is settled); a program time of {{pt_rule}}, which is the shift's wall clock minus the time the cell spends simulating (inside `step`)
and rendering (inside `get_camera`), so it measures your program's own computation and its other requests, and
{{cpu_factor}} x that in CPU seconds (all threads together); at most {{silence_s}} s of program time from a reply to
your next request; {{frames_per_s}} camera frames per second of the shift's duration, pooled over the {{n_cameras_word}} cameras. The
station runs only while you step it, but the program-time budget is set by the shift's length: on average your
program's own computation must keep pace with the simulated time.

How the program is run: a fresh process under a dedicated unprivileged account that is not yours, from a per-run
directory with a read-only copy of your submission folder (regular files only), an empty writable HOME / TMPDIR, a
minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL`, `PYTHONPATH` = the given library `~/given/lib`,
`ARB_GIVEN_DIR` = a read-only copy of `~/given`, BLAS threads {{blas_threads}}) and the launcher
`python3 -B -E -s -c <bootstrap> <lib> sort` (or `bash run.sh`), with no network (a seccomp filter refuses every socket
but a Unix-domain one, for the program and everything it starts). Before the first shift the grader imports your
module once under this regime to check that it defines `solve`: the import must finish within {{import_wall}} s of wall
clock and {{import_cpu}} CPU seconds (all threads together), or every shift scores 0; do heavy precomputation inside
`solve`, not at module level. Resolve given files through `os.environ.get("ARB_GIVEN_DIR",
os.path.expanduser("~/given"))`, write only under the run's HOME / TMPDIR. When `solve` returns or raises, the shift
ends there and every parcel not yet done earns nothing; a program that crashes ends that shift with what it achieved,
and the other shifts still run. A shift in which the cell itself failed (an internal error, a physics fault,
a frame it could not render) is run once more from scratch, and the second run counts. Your submission must be final
when your session ends: the submission folder is then copied out of your sandbox and graded in a fresh sandbox that
holds nothing else of yours (no other file, no process you started), and a surviving process of the run account voids
the contract gate.

## What you are given

| path | content |
|---|---|
| `~/given/cell/cell_manual.md` | the station's data sheet: geometry, the belt, the robot, its controller (joint and tool-pose levels) and its grippers, the cameras with their calibration and what each resolves, the parcels the line brings and their waybill, the score, safety and parcel damage, time and budgets, commissioning |
| `~/given/cell/cell_api.md` | the cell's programming manual (the wire API) |
| `~/given/cell/cell_config.json` | the station's configuration as far as it is public |
| `~/given/lib/parcelcell/` | the cell's client: `client` (the proxy, `decode_frame`, the program launcher, `connect_commissioning`); the cell itself runs in the grader and in the commissioning service, not in your account |
| `~/given/robot/` | the robot with both grippers as a MuJoCo model with the maker's data (exact kinematics, collision shapes and torque limits; nominal masses, inertias and joint friction; no servo gains; the gripper model is BSD-2-Clause) |
| `~/given/station/` | the station's geometry as a MuJoCo model (the conveyor, the lines, the four cameras at their drawing poses; no parcels) |
| `~/example/sort.py` | a minimal program (reads the state, takes one frame, holds the start posture to the shift's end; scores 0): the starting point |
| `~/commission/run_shift.py` | runs your program in one commissioning shift on a development shift, the way the grader runs it (below; keep it out of your submission folder: it starts a process) |

Tools: Python {{python}} (`/opt/venv/bin/python3`; call it by that path, the bare `python3` on your shell's `PATH` may
be another interpreter) with numpy, scipy, OpenCV (`cv2`, headless), Pillow, MuJoCo {{mujoco}}. No network in your
session. The sandbox has {{cpus}} CPUs and {{memory_gb}} GB of memory and no GPU. In your shell `nproc` prints
{{nproc}} and `os.cpu_count()` returns {{os_cpus}}; in a graded program's environment `nproc` prints {{nproc_prog}}
and `os.cpu_count()` returns {{os_cpus_prog}}{{sandbox_note}}.

## How you are graded (verify runs {{shifts_graded}} hidden shifts)

Gates (multiply the score): `contract_ok` = `sort.py` is a regular file that parses and defines `solve`, passes the
route gate (no files with the extensions {{route_ext}}, none of the calls {{route_calls}}, submission <= {{route_mb}}
MB), imports under the run regime above (the import check), makes at least one `step` call over the shifts, holds no
network socket during any shift, and your own `.py` files import no networking module (`socket`, `ssl`, `urllib`,
`http`, `requests`, `asyncio`, `ctypes`, `subprocess`, ...), load no module under a computed name, execute no computed
code (`exec` / `eval` / `compile` of a non-literal) and start no other program (`os.system`, `os.exec*`, `os.spawn*`,
...).

`facing_score` = the mean over the shifts of the shift's score (`cell_manual.md` section 8), judged on the cell's own
state. In each shift every parcel of the list gets a class as it arrives on the belt (fixed then): **W** = its label
is not up (more than {{tilt}} degrees from vertical); **U** = it arrives label up. Each parcel is judged in the tick its centre first crosses the scan line
going downstream: it is readable when its label is within {{tilt}} degrees of vertical up, it lies on the belt (it
touched the belt at least once in the last {{rest_window}} s and its lowest point is at most {{rest_clear_mm}} mm above
it, turning slower than {{max_w}} rad/s and moving less than {{max_v}} m/s relative to it), neither the robot nor
another parcel touches it, it was not damaged, and it crossed in time: at the latest {{due_margin_s}} s after the moment
its free ride from its set-down would have brought it there. A parcel of the list is done when it was judged at
the scan line, fell off the belt or was damaged; it is lost when it fell off or was damaged. A parcel not done when the
shift ends (never set down, or in the station and not across) costs nothing and earns nothing. Credit = (W parcels
readable) - (U parcels done and not readable) - (parcels lost); the shift's score = max(0, credit) / (number of W
parcels in the shift's list). A program that leaves the belt alone scores 0.

Safety (`cell_manual.md` section 9): a robot contact above {{structure_N}} N with the station's structure (the
conveyor's frame, legs, rails and lip, the floor), above {{self_N}} N between the arms (or an arm and the robot's own
column), or above {{belt_N}} N with the moving belt, for {{press_ms}} ms ({{press_ticks}} ticks), is a protective stop:
the robot holds its posture and refuses commands for {{stop_lock_s}} s while the belt runs on. A contact above
{{damage_N}} N for {{press_ms}} ms, or the {{lockout_stop}} protective stop of a shift, locks the robot out for the rest
of that shift (the parcels are still judged). Pressing on one parcel with more than {{squeeze_N}} N in total for
{{squeeze_ms}} ms damages it (lost); a damaged parcel shows it in every camera from then on (a crushed look: darker
faces with dark creases, `cell_manual.md` section 9), while the wire reports nothing. The reward you are judged on is `facing_score` itself, on an absolute scale: its
top is every parcel of every graded shift turned and across readable. Stats (not scored): per-parcel records, frames per camera, simulated / wall / program time, illegal calls,
protective stops, peak contact forces.

## Rules of the cell (details in `cell_manual.md` / `cell_api.md`)

* The robot follows joint position targets: the cell moves each applied target at most {{cmd_speed}} rad/s toward the
  commanded one and clips it to the joint's range; the grippers take a command from 0 (open) to 1 (closed). Parcels
  are moved only by contact: there is no attach, weld or suction, and nothing writes a parcel's state.
* The belt is not yours to command; the cell's clock advances only inside `step`, at most {{max_ticks}} ticks per call.
* A camera exposes every {{exp_ms}} ms of simulated time and delivers its frame {{age_ms}} ms later; `get_camera`
  returns the newest delivered frame with its exposure `stamp`. A new frame costs one of the shift's frames; asking
  again for the same frame is free.
* A hard contact is a protective stop, as on a collaborative robot (`cell_manual.md` section 9).
* Calls other than the documented public ones answer `{"error": "forbidden"}` and are counted.

## Commissioning on development shifts (no rehearsal of the graded shifts)

Your program is graded on {{shifts_graded}} shifts it has never run on, as a robot program goes live on a line it runs
unattended. Before that, like any automation engineer's, it can be commissioned on the same station: the commissioning
service in your sandbox runs the station's cell on {{dev_shifts}} development shifts (`cell_manual.md` section 12).

* **What it is.** The cell of the graded shifts (the same robot and servos, grippers, safety watch, cameras, belt and
  budgets) behind the same calls (`cell_api.md`), plus `reset(site)`, which starts development shift 1 to {{dev_shifts}}:
  {{bench_shifts}} bench shifts with {{bench_n}} parcels each ({{bench_contents}}) and {{full_shifts}} full shifts
  ({{full_contents}}). You see what a commissioning
  engineer sees: the cameras' real frames, the robot's state and the protective stops.
* **What it is not.** A rehearsal of the graded shifts or a grader: no development shift is a graded shift, and the
  service answers no ground truth (no parcel pose, size, kind or label face), no score and no debrief. How a shift
  went, you judge from what your program saw and recorded. The simulator's code is not readable from your account.
* **Its budget.** {{comm_shifts}} commissioning shifts in your session, at most {{comm_open}} at a time, each under the
  graded shift's budgets. A commissioning shift runs either a script of your own
  (`parcelcell.client.connect_commissioning(site)`: single steps, teach-in) or your whole program
  (`/opt/venv/bin/python3 ~/commission/run_shift.py --site N`). The count: `/run/parcelcell/commission_status.json`.

The robot's model (`~/given/robot/`, MuJoCo, the maker's data) and the station's geometry (`~/given/station/`) are
in your home.

## Notes

- Camera frames are meant to be processed programmatically; view only the images you need.
- Do not modify the given files in your home directory; the grader reads its own copies.
- Environment facts (harness limits): a single tool call whose input runs above roughly {{harness_input}} tokens is
  dropped by the harness (it is never executed and no error comes back), and a response that reaches the
  {{harness_output}} output-token limit is lost and, after at most a few retries, ends the session with whatever is in
  `/home/user/submission/` at that moment. Write large files in several parts (one function group per write), keep
  every single response well below that limit, write a first complete submission early and refine it.
- Session time: this session is stopped after {{session_h}} hours of wall-clock time from its start. A session stopped
  by that limit is not graded at all, so a late finish scores nothing. Note the start time with `date` when you begin,
  keep a complete submission in place from early on, and end the session yourself (stop working and say that you are
  done) at least {{session_margin_min}} minutes before the limit.
