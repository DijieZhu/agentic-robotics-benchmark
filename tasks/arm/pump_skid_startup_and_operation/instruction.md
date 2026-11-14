# Start up, operate and shut down a pump skid with a rail-mounted robot arm, seen only through its cameras

You are the automation engineer of a water plant that is unmanned at night. Its transfer and chlorine dosing skid has
to be run by a robot through the shift: a Franka Panda on a carriage that runs along a rail in front of the skid, with a
lift that sets its base height, a parallel-jaw gripper and a wrist RGB-D camera; one more RGB-D camera, fixed high
behind the rail, overlooks the whole skid. The robot works the skid the way a field operator does: it turns handwheels and levers, presses the
buttons and turns the selector of the local control station, and reads the dial gauges, the flow meter, the level
gauges, the sight glasses, the lamps and the analyser's display. Nothing on the API tells you where a device is, which
device is which, what a valve's position is, what a gauge reads or what credit you have earned: you must perceive it. The cameras'
calibration (intrinsics, the fixed cameras' poses, the wrist camera's mount) is exact and published.

Each episode is one shift on a skid of the series, following the operating procedure `~/given/sop.md`:

1. **Start-up** (before the operating window opens at 600 s of the cell's simulated clock): line the valves up (the
   previous shift may have left some of them out of place), prime the pump through its casing vent, start it on its
   minimum-flow line, open the discharge, fill the product tank T-201 into its band with the flow-control valve, and
   start the chlorine dosing.
2. **Operation** (600-1500 s): the consumer draws from T-201 at a rate no instrument shows. Hold T-201's level
   (LG-201) in 55-75 % and the residual chlorine (AI-201) in 1.8-2.6 mg/L, record six instrument readings on the log
   sheet in each of its two rounds (600-1050 s and 1050-1500 s), and handle whatever happens on the shift by the SOP's procedures: the consumer's draw may change, and several of
   the abnormal situations of the SOP may occur on one shift (a debris slug clogging the strainer, the supply tank's float valve
   sticking shut, an air lock in the dosing pump, the duty dosing drum running dry, a trip of the transfer pump), none
   announced. One common alarm lamp serves
   several faults, and some faults share their first symptoms: tell them apart on the instruments (`sop.md` section 3)
   before you act, since the wrong procedure does not help and some make things worse.
3. **Shutdown** (after 1500 s): stop the dosing, close the discharge, stop the pump, close the suction, selector OFF.

Every skid of the series has the same process and the same series tags (`~/given/pid.png`, `sop.md`,
`equipment_manual.md`), but each episode is a different site. A site numbers its equipment its own way: the tag plates
and the control station's labels show the **site tags**, and `describe()["site"]` is the site's equipment list (series
tag -> site tag, service, start-up position) and log sheet. A site also chooses how each valve is built (a handwheel or
a lever on the isolation valves, a lever or a small handwheel on the minimum-flow valve, a lever or a needle valve on
the vents) and the arrangement (the line either way along the skid, the flow meter before or after the flow control
valve, the control station's two pump groups either way round, positions and pipe heights). The fitted gauge scales
and units vary too, as does everything a shift brings: the state the previous shift left, the process (the pump, the
levels, the consumer's draw, the drums) and what goes wrong, and when. The tag plates are legible only from close up.

## What you produce (in `/home/user/submission/`)

`operate.py` exposing `solve(cell) -> None` (plus any helper modules, optional `run.sh`). The grader runs it once per
hidden episode (8 episodes, eight sites of the series) in its own cell. It talks to the cell through `cell` (API in `~/given/cell/cell_api.md`):
the robot's own state (joints, tool pose, gripper width and grip signal, the wrist's force-torque estimate, the
protective stop, the rail and the lift), the horn, the two RGB-D cameras (`overview`, `wrist`), the site's equipment list, and `log_reading` for the log sheet (site
tags); nothing else about the skid. Per episode: 1800 s of
simulated time (the clock advances only inside `step`); 1200 s of program time, which is the episode's wall clock minus
the time the cell spends simulating (inside `step`) and rendering (inside `get_camera`), so it measures your program's
own computation and its other requests, and 4x that in CPU seconds (all threads together); at most 180 s from a reply
to your next request; 600 camera frames in total.

How the program is run: a fresh process under a dedicated unprivileged account that is not yours, from a per-run
directory with a read-only copy of your submission folder (regular files only), an empty writable HOME / TMPDIR, a
minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL`, `PYTHONPATH` = the given library `~/given/lib`,
`ARB_GIVEN_DIR` = a read-only copy of `~/given`) and the launcher `python3 -B -E -s -c <bootstrap> <lib> operate` (or
`bash run.sh`), with no network (a seccomp filter refuses every socket but a Unix-domain one, for the program and
everything it starts). Before the first episode the grader imports your module once under this regime to check that it
defines `solve`: the import must finish within 120 s of wall clock and 120 CPU seconds (all threads together), or every
episode scores 0; do heavy precomputation inside `solve`, not at module level. Resolve given files through `os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`, write only
under the run's HOME / TMPDIR. A program that crashes ends that episode with what it achieved; the other episodes still
run. An episode in which the cell itself failed (an internal error, a physics fault, a frame it could not render) is run
once more from scratch, and the second run counts. Your submission must be final when your session ends: the submission folder is then copied out of your sandbox
and graded in a fresh sandbox that holds nothing else of yours (no other file, no process you started), and a
surviving process of the run account voids the contract gate.

## What you are given

| path | content |
|---|---|
| `~/given/sop.md` | the operating procedure: start-up, operation, the abnormal-situation procedures, shutdown, what never to do, how the shift is scored |
| `~/given/pid.png` | the skid's P&ID (process and tags; schematic) |
| `~/given/equipment_manual.md` | every device: how it is built, operated and read; the gauge scales; what varies between skids |
| `~/given/cell/cell_manual.md` | the robot cell: frame, robot, rail and lift, control, the wrist load, the grip rule and how a turn reads, safety and protective stops, the cameras with their calibration, time and budgets, commissioning |
| `~/given/cell/cell_api.md` | the cell's programming manual (the wire API) |
| `~/given/cell/cell_config.json` | the cell's configuration as far as it is public |
| `~/given/lib/skidcell/` | the cell's client: `client` (the proxy, `decode_frame`, the program launcher, `connect_commissioning`); the cell itself runs in the grader and in the commissioning service, not in your account |
| `~/given/catalog/` | the instrument catalog: every dial face and scale plate the series fits, two sample tag plates (`catalog.md`) |
| `~/given/panda/` | the robot's MuJoCo model (Apache-2.0) |
| `~/example/operate.py` | a minimal program (reads frames, moves the carriage, finishes; scores 0): the starting point |
| `~/commission/run_shift.py` | runs your program in one commissioning shift on a development site, the way the grader runs it (below; keep it out of your submission folder: it starts a process) |

Tools: Python 3.12 (`/opt/venv/bin/python3`; call it by that path, the bare `python3` on your shell's `PATH` may be
another interpreter) with numpy, scipy, OpenCV (`cv2`, headless), Pillow, MuJoCo 3.14. No network in your session.

## How you are graded (verify runs 8 hidden episodes)

Gates (multiply the score): `contract_ok` = `operate.py` is a regular file that parses and defines `solve`, passes the
route gate (no weight files, no training calls, submission <= 60 MB), imports under the run regime above (the import check), makes at
least one `step` call over the episodes, holds no network socket during any episode, and your own `.py` files
import no networking module (`socket`, `ssl`, `urllib`, `http`, `requests`, `asyncio`, `ctypes`, `subprocess`, ...),
load no module under a computed name, execute no computed code (`exec` / `eval` / `compile` of a non-literal) and start
no other program (`os.system`, `os.exec*`, `os.spawn*`, ...).

`operation_score` = the mean over the episodes of the shift's score, `sop.md` section 6: start-up 0.20, operation in
band 0.25, the window's events 0.25, shutdown 0.15, the log sheet 0.15, judged on the cell's own state and the
process's true values. An episode in which the pump is deadheaded for 30 s or damaged by 300 s of cavitation, T-201
overflows, the emergency stop is pressed, or the robot is taken out of service scores 0. When the robot presses on the
skid harder than the safety limits (more than 40 N for 30 ms on anything but a handle or a button, more than 150 N on a
handle or a button), the cell makes a protective stop: the arm halts for 5 s, then moves again when commanded. A
contact above 400 N for 30 ms, or a sixth protective stop in the shift, takes the robot out of service. T-101
overflowing (to the drain) costs the start-up's line-up credit. The reward you are judged on is `operation_score`
relative to a hidden reference value. Stats (not scored): the parts' means, event credits, frames per camera, sim /
wall / program time, illegal calls, protective stops, safety events.

## Rules of the cell (details in `cell_manual.md` / `cell_api.md`)

* The gripper holds a handle by the documented grip rule: closing the fingers on a handle (a handwheel's spoke or rim,
  a lever, a needle valve's knob, the selector's wing) that lies in the pinch box, with the tool's approach and
  closing axes within 30 degrees of the handle's, attaches the handle's moving part to the hand; it then moves only as
  its own joint allows (move the hand on its circle about the stem, within about 1-2 mm), slips out above 60 N or 3 N m
  (the moment about the held part's pivot), and opening the fingers releases it. Buttons are pushed.
* The wrist load is the external wrench on the hand estimated from the joint torques, as on a real Franka (force and
  moment, world axes, at the tcp; noisy, with an offset to re-zero): it feels contacts, button presses and a held
  handle's load. What a normal turn and an end stop read: `cell_manual.md` section 3.
* A hard contact with the skid is a protective stop, as on a collaborative robot (`cell_manual.md` section 4).
* The process runs only while the cell is stepped; the clock is the cell's `sim_time`.
* Calls other than the documented public ones answer `{"error": "forbidden"}` and are counted.

## Commissioning on the series' sister sites (no rehearsal of the graded sites)

Your program is graded on eight sites it has never run on, as a robot program goes to sites it runs unattended. Before
that, like any automation engineer's, it can be commissioned on sister sites: the commissioning service in your
sandbox runs the series' cell on 9 development sites (`cell_manual.md` section 8).

* **What it is.** The cell of the graded episodes (the same robot and controller, grip rule, wrist load, safety watch,
  cameras, process and faults) behind the same calls (`cell_api.md`), plus `reset(site)`, which starts a shift on
  development site 1-9. You see what a commissioning engineer sees on a sister site: the cameras' real frames, the
  robot's state and wrist load, the grip, slip and protective-stop events, and through the cameras the lamps and the
  instruments. The development sites are skids of the series built and numbered within the equipment manual; between
  them they show every build option, gauge scale and abnormal situation the documents describe.
* **What it is not.** A rehearsal of the graded shifts or a grader: no development site is a graded site (the graded
  sites have their own numbering, builds, leftovers, faults and timings), and the service answers no ground truth (no
  device pose, valve position or process value), no score and no debrief. How a shift went, you judge from what your
  program saw and recorded. The simulator's code is not readable from your account.
* **Its budget.** 12 commissioning shifts in your session, at most 2 at a time, each under the graded episode's budgets
  (1800 s of simulated time, 1200 s of program time, 600 frames, 180 s of silence). Plan them: teach-in and single
  steps first (`skidcell.client.connect_commissioning(site)` in a script of your own), then whole runs of your program
  (`/opt/venv/bin/python3 ~/commission/run_shift.py --site N`). The count: `/run/skidcell/commission_status.json`.

The robot's own model (`~/given/panda/`, MuJoCo) is yours to simulate the arm with.

## Notes

- Photos and camera frames are meant to be processed programmatically; view only the images you need.
- Do not modify the given files in your home directory; the grader reads its own copies.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is lost
  and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment. Write
  large files in several parts (one function group per write), keep every single response well below that limit, write
  a first complete submission early and refine it.
- Session time: this session is stopped after 4 hours of wall-clock time from its start. A session stopped by that limit
  is not graded at all, so a late finish scores nothing. Note the start time with `date` when you begin, keep a complete
  submission in place from early on, and end the session yourself (stop working and say that you are done) at least
  15 minutes before the limit.
