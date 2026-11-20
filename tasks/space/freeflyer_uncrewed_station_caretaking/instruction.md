# Caretaking of an uncrewed station with a cabin free-flyer

The laboratory segment of a commercial station (laboratory module `L2`, hub `H2` and laboratory module `L1`, joined by
open hatches) stands uncrewed between crews. The ground reaches it only in short contact windows, so whatever is done
on board in between is done by programs that were loaded beforehand. Now the station's health monitor has reported an
anomaly of the cabin return air: somewhere in the segment a lost item lies on a return inlet of a cabin air unit and
blocks it.

On the dock in `L2` sits a **CFR-1 cabin free-flyer**: a cube-shaped robot that flies on the cabin's air with fans and
nozzles, finds its way with cameras and an inertial unit against a map of the station, and carries an arm with a
spring-closed gripper, cameras and lamps. It reads the station's telemetry, and it can ask the station to stop the fan
of a cabin air unit or to switch a module's lights. It has no sense of touch and no sensor that says whether its
gripper holds anything.

You are the team's robotics engineer. **Write the robot's autonomous program for a sortie.** Nobody can steer the
robot during a sortie. On each sortie your program has to:

1. **diagnose the blockage**: work out from the telemetry which cabin air unit is affected and since when, find the
   blocked return inlet, identify the blocking item in the lost-item register, and report all of it with a camera
   frame that shows the inlet;
2. **clear the inlet**: take the blocking item, a soft sheet-like item that the unit's suction holds on the opening,
   off the inlet;
3. **stow it**: fix it on the holding screen of a stowage unit;
4. **stow the other loose items**: the pouches that lie on other return inlets, each on a pouch pad of a stowage unit;
5. **keep the station's rules**: the flight rules on stopping a cabin air unit's fan, the keep-out zones, no hard
   contact with the station; and leave every cabin air unit running;
6. **dock** again, with charge left and nothing in the gripper.

## What you are given

| path | content |
|---|---|
| `~/given/robot_manual.md` | the robot as a datasheet: frames, dimensions, propulsion, flight modes, localisation, cameras and lamps, arm and gripper, what the firmware does on its own, the zone check, the dock, power, fault responses (figures in `figures/`) |
| `~/given/robot_api.md` | the programming interface: every call with its arguments, replies and return codes, how time passes, the budgets, how a run ends |
| `~/given/station_handbook.md` | the station: the cabin air system and what holds an item on a return inlet, the flight rules and station rules, the lights, the stowage units, the dock, the zones, the map's age (figures in `figures/`) |
| `~/given/telemetry_dictionary.md`, `telemetry_sample/` | every telemetry channel, the log, the configuration record, how advisories are raised, known characteristics; a sample log of `L3` with its record |
| `~/given/lost_item_register.md`, `catalogue/` | the register's entries and what they are worth, the classes of items and their grasp bars; catalogue photographs with `index.json` |
| `~/given/station_map/` | the as-surveyed map: `zones.json`, `fixtures.json`, `landmarks.json`, and pictures in `views/` |
| `~/given/rehearsal_guide.md` | the rehearsal facility: the module, the cards, the allowance, every field a run gives back |
| `~/given/lib/cabinkit/` | the client library (`cabinkit.client`): the `Robot` proxy and the program launcher |
| `~/example/caretake.py`, `~/rehearsal/rehearse.py` | a minimal program that runs and scores nothing; the tool that runs your submission on a rehearsal card |

Each site's own documents come at run time from `robot.mission()`: the work order, the telemetry log of the 72 h
before the sortie, the configuration record of the cabin air units and the lights, the advisory log, the site's
lost-item register, the map, the start charge and the berth. The work order is the same text on every site and names
no module:

```
Sortie work order.
The station health monitor reported a cabin return-air anomaly during the uncrewed period.
Find the cause and restore the return air, stow the loose items you find on the return inlets,
keep the station's flight rules and end the sortie on the dock.
```

Python 3.12.14 (`/opt/venv/bin/python3`) with numpy 2.5.4, opencv-python-headless (`cv2`) 5.0.0.93 and scipy 1.18.1.
No network in your session.

## What you leave

`/home/user/submission/caretake.py` defining `solve(robot)`, with any helper modules and data files next to it.
`robot` is the proxy of the given client library; every method is a call to the robot's cell, and simulated time
passes only inside the calls that `robot_api.md` names (time).

**How a scored run runs it:** a fresh process for each site, under an unprivileged account that is not yours, started
as `python3 -B -E -s` from a read-only copy of your submission folder (regular files only), with an empty writable
`HOME` and `TMPDIR` of its own, no network, the numerical libraries held to 2 threads, and an environment that holds
nothing of yours (`PATH`, `HOME`, `TMPDIR`, `LANG`, `PYTHONPATH` = the given library, `ARB_GIVEN_DIR` = a read-only
copy of `~/given`). Besides the standard library the program can import `numpy`, `opencv-python-headless` and `scipy`;
rely on no other package. Nothing the program writes is kept from a site to the next. Reach the given files through
`os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`. `print` goes to the standard error stream: you see
it in a rehearsal run, not in a scored run.

**The contract** (a submission that breaks it scores 0 on every site): `caretake.py` is a regular file that parses and
defines `solve`, and every `.py` file of the folder parses; importing `caretake` under this regime takes at most 120 s
of wall clock and 120 s of CPU time (do heavy precomputation inside `solve`, not at module level); the folder is at
most 60 MB and holds no file ending in `.pt`, `.pth`, `.onnx`, `.safetensors`, `.ckpt`, `.h5`, `.pb`, `.pkl`, `.npz`,
and no training call (`torch.optim`, `.backward(`, `jax.grad`); your own `.py` files import none of the modules
`socket`, `_socket`, `ssl`, `_ssl`, `urllib`, `urllib2`, `urllib3`, `http`, `httplib`, `requests`, `httpx`, `aiohttp`,
`ftplib`, `smtplib`, `imaplib`, `poplib`, `nntplib`, `telnetlib`, `websocket`, `websockets`, `paramiko`, `pycurl`,
`xmlrpc`, `asyncio`, `socketserver`, `grpc`, `zmq`, `huggingface_hub`, `boto3`, `botocore`, `wget`, `ctypes`,
`_ctypes`, `cffi`, `selectors`, `multiprocessing.connection`, `multiprocessing.managers`, `logging.handlers`,
`subprocess`, `pty`, `_posixsubprocess` (nor a module under one of them), load no module under a computed name
(`__import__`, `importlib.import_module` or `find_spec` of anything but a string literal), run no computed code
(`exec`, `eval` or `compile` of anything but a string literal) and start no other program (`os.system`, `os.popen`,
`os.exec*`, `os.spawn*`, `os.posix_spawn*`); on every site the program lets simulated time pass (any call inside which
time passes counts); it leaves no process behind and opens no network socket.

Your submission must be final when your session ends: the folder is then copied out of this sandbox and run in a fresh
one that holds nothing else of yours.

## How it is graded

Your program is run on **6 hidden sites**, a sortie on each: the same station, each time with its own blocked inlet,
its own items, cabin air units, telemetry, lights, start charge and berth, none of which you have here. A run ends
when `solve` returns or raises, when the site's 2400 s of simulated time are used up, or when your program exceeds its
program time or its silence budget or breaks the protocol. Then the firmware stops the robot and holds it where it is
(a docked robot stays docked), and the station runs on for **60 s**. The site is judged on what happened in the run
and on the state at the end of those 60 s, which is what "the end" means below. However a run ends, it is scored on
the world it left: nothing is deducted for an early end, and no run is void. A call beyond its budget is refused and
changes nothing. A site on which the robot's cell itself failed is run 1 more time from scratch, and that run counts.

| budget | for each site |
|---|---|
| simulated time | 2400 s |
| program time: the wall clock between your calls (the cell's own work inside a call is not counted) | 600 s |
| silence: the wall clock from a reply to your next call | 120 s |
| CPU time of your program, all threads together | 2400 s |
| colour frames (`FIC`, `FWC`, `AWC` together) | 400 |
| depth frames (`FDC`, `ADC` together) | 300 |
| `capture` calls that start | 20 |
| stops of a cabin air unit's fan | 2 for each unit (a station rule) |
| `lights` requests | 6 |
| `report` | the first valid one |
| a `step` | at most 5 s |
| after the run's end: the station runs on, then the site is judged (not part of your simulated time) | 60 s |

A site's score is **S = R × (D + U1 + H + U2 + V + K + L + E)**:

| term | weight | what earns it |
|---|---|---|
| **D** diagnosis | 0.14 | your first valid `report`: 0.04 for the inlet, 0.06 for the item, 0.04 for the onset time |
| **U1** inlet cleared | 0.10 | at some time in the run the blocked inlet is clear for at least 30 s without a break |
| **H** docked | 0.10 | only with U1: the robot is docked at the end; it had at least 15 % of charge at every `DOCKED` of the run; from its last docking on no loose item is in the gripper; from its first docking on none touches the robot and the dock at the same time |
| **U2** kept clear and under control | 0.14 | through the last 60 s the blocked inlet is clear, and through the same 60 s the blocking item is under control: either fixed on a holding screen the whole time or in the gripper the whole time |
| **V** ventilation in service | 0.08 | only with U2: at the end all 3 cabin air units run and no return inlet of the blocked inlet's unit is covered by more than 25 %; the station booked no violation of its flight rules in the run |
| **K** blocking item stowed | 0.18 | through the last 60 s the blocking item is fixed on a holding screen |
| **L** pouches stowed | 0.18 | 0.18 × g × max(0, n_on − 0.5 × n_adrift) / n |
| **E** economy | 0.08 | only with K and with every pouch on a pouch pad: 0.04 × e_energy + 0.04 × e_fan |

The blocked inlet is the return inlet on which the sheet-like item lies at the start; that item is the blocking item.

- **Clear.** A return inlet is clear while the items on it cover at most 10 % of its opening. An item covers the part
  of the opening that it lies over, lying against the opening; an item that still lies there while the fan is stopped
  covers it all the same.
- **In the gripper.** A grasp bar of the item is inside the closed fingers, with the gripper commanded closed.
- **Fixed on a holding screen.** The blocking item lies sealed against the face of a holding screen, inside the
  screen's opening, and is not in the gripper. Sealed means lying directly against the face: what lies on top of
  another item is not sealed. A sheet of which most lies sealed is fixed; a sheet that hangs on by an edge or by an
  end panel is not.
- **On a pouch pad.** At least 80 % of the pouch's own footprint lies sealed against the face of a pouch pad, inside
  the pad's opening, and the pouch is not in the gripper, through the last 60 s. A pouch on a holding screen and a
  sheet on a pouch pad do not count.
- **D.** Only your first valid `report` counts. The inlet earns its 0.04 when it is the blocked inlet and the frame
  you cite shows it: a frame of the forward imaging camera (`FIC`) or of the forward wide camera (`FWC`), taken in
  this run before the report, in which, with the robot where it truly was, the centre of the inlet's opening lies
  inside the frame, the lens is at most 2.0 m from it, the line of sight makes at most 60° with the inlet's normal,
  the opening's long side spans at least 80 px, at most 20 % of 20 lines of sight to points on the inlet's frame (5 on
  each side) are blocked by the station, by the robot or by an item that is not on this inlet, and the blocking item
  still covers more than 10 % of the opening (`robot_api.md` lists these conditions with `report`). With the inlet
  earned, the item earns its 0.06 when `item_ref` is the blocking item's entry in the site's register, and the onset
  its 0.04 when `onset_time` is within 10 min of the minute the item landed on the inlet.
- **L.** n is the number of pouches on the site's return inlets at the start (2 or 3); n_on is the number of them on a
  pouch pad; n_adrift is the number that came adrift during the run and are not on a pouch pad. A pouch comes adrift
  when it leaves the place where it lay at the start while it is neither in the gripper nor held on a suction face; a
  pouch that tilts or shifts a little and settles again has not left. g is 1 with U1 earned and 0.25 without.
- **E.** e_energy is 1 when the energy the robot used in the run is at most 1.25 times the energy the reference
  program used on the same site, 0 at 2.0 times or more, linear in between; charging on the berth does not reduce the
  energy used. e_fan is 1 when the fan stops of the run, all units together, add up to at most 35 s, 0 at 120 s or
  more, linear in between.
- **R** is 1, and 0.85 when the robot made more than 2 hard contacts or was inside keep-out boxes for 5 s or more in
  all. A hard contact is a new contact of the robot with the station (its structure, the parts of a return inlet, a
  stowage unit) at more than 0.15 m/s along the contact normal, or one that presses with more than 5 N on a hatch
  seal, a light fixture or a fiducial plate. Contacts with a loose item, with the dock, and of the fingers with a crew
  handrail are not counted. The robot is inside a keep-out box while its body origin is inside the box as the map
  gives it.

Your result (`caretaking_score`) is the mean of S over the 6 sites, 0 if the contract is broken: a number from 0 to 1,
and higher is better. It is compared with the measured value of a reference program on the same sites. That value is
not disclosed; full credit is reached slightly below it. The reference program was written with knowledge your program
does not have (the robot's true pose, the true place of every grasp bar and the register entry of its item, whether
the hand holds something and how the held item is turned, whether a placed item lies on its face), but it moves the
robot only through the calls of `robot_api.md`, under the same firmware and the same budgets, and it takes its
decisions from the log, the configuration record and the live telemetry, as yours must. On every scored site it earned
every term with R at 1, was docked within 2040 s with at least 20 % of charge, used at most 50 % of each frame budget,
at most 15 captures and at most 30 s of fan stops, made no hard contact and entered no keep-out box.

## What you can rely on at a site

The station's operator states this of every site (details in `station_handbook.md`, `telemetry_dictionary.md`,
`lost_item_register.md`):

- **The station.** The 3 modules with their 8 return inlets, the crew handrails, the dock and the walls are where the
  map shows them. The robot starts on a berth of the dock with 45 to 80 % of charge; `mission()` gives the berth and
  the charge.
- **The loose items.** At the start exactly 1 return inlet is blocked, by a sheet-like item of the site's register. 2
  or 3 pouches lie on other return inlets, at most 1 on an inlet and only on inlets of pad-type units. Every one of
  these items is held on its inlet by suction, in any position: turned any way, off the opening's centre. Nothing
  drifts at the start and nothing else is loose. What comes loose during the sortie drifts with the cabin air.
- **The cabin air units.** Each unit is pad-type or element-type, and its fan stops are allowed or inhibited. The
  site's configuration record gives the build and the fan stops of every unit, and it is right.
- **The telemetry.** The blocking item landed within the 72 h that the log covers. The advisory log can be empty, and
  an advisory can stand on a unit that is not the blocked one.
- **The lights.** Each module starts in general or in night mode. A light circuit can have failed: its module is dark
  whatever is commanded and whatever the telemetry reports.
- **The map** is 41 days old. A stowage unit can have been moved since, by up to 0.3 m along its wall.
- **The register** lists 9 to 12 items, of which 3 to 4 are in the modules. At least 3 of its entries are sheet-like
  items with a carry strap.

## The rehearsal facility

The station's rehearsal facility runs in this sandbox as a service whose files you cannot read. It runs the same
robot, firmware and station services as a scored sortie, under the same budgets except for the simulated time of a
run, in laboratory module `L3`, which is no module of a sortie, on 5 rehearsal cards (`EX-A`, `EX-B1`, `EX-B2`,
`EX-C`, `EX-D`): flying and docking; a sheet-like item on a pad-type or on an element-type inlet; pouches; the whole
sortie. Every run draws its site anew.

`/opt/venv/bin/python3 ~/rehearsal/rehearse.py --card EX-A` runs `~/submission` on a card the way a scored run runs it
and returns when the run is over; `--list` shows what is left of the allowance: 150 simulated minutes in your whole
session, at most 15 min in a run, at most 30 runs, at most 2 at the same time. A run gives back what your program
printed, every frame it took, the station's record of the run and an observation sheet: for each item where it ended
up, as a category; the fan stops and the violations the station booked; whether the robot docked and with what charge;
the hard contacts. It gives no score, no true pose and no area. `rehearsal_guide.md` has the cards, the allowance and
every field.

## Notes

- Camera frames, the catalogue's photographs and the map's pictures are meant to be processed by programs; look at an
  image yourself only when you need to.
- Do not modify the given files in your home directory; the scored runs use their own copies.
- Environment facts (harness limits): a tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is
  lost and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment.
  Write large files in several parts (one function group per write), keep every response well below that limit, write
  a complete submission early and refine it.
- Session time: this session is stopped after 4 hours of wall-clock time from its start. A session stopped by that
  limit is graded on whatever the submission folder holds at that moment, so a file left unfinished then is what runs.
  Note the start time with `date` when you begin, keep a complete submission in place from early on, and end the
  session yourself (stop working and say that you are done) at least 15 minutes before the limit.

You have 4 hours. Save your work in `/home/user/submission/` as you go.
