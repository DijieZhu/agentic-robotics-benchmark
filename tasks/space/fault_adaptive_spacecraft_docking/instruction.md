Build an actively identifying, fault-adaptive controller for a planar spacecraft
rendezvous and leave it runnable at:

    /home/user/submission/run.sh

This is not a direct three-axis control problem. The spacecraft has six unidirectional
body-frame thrusters. Each action slot is wired to a different nozzle in every episode;
the initial wiring is an unknown permutation. Nozzle direction, moment arm, and
effectiveness also have continuous installation error. Every nozzle couples translation
and yaw. During flight, a fault-tolerant rotor bus silently installs an unknown cyclic
ordering of the six nozzles, with a hidden starting offset and a hidden direction.
The ordering is an arbitrary permutation, not numerical nozzle order. After every
subsequent command the rotor advances with an action-dependent one- or two-slot stride,
and two physical nozzles retain different fractions of their authority. The two degraded
nozzles are not the only nonlinear effect: after each bus event every physical nozzle
also follows its own hidden thrust curve, its thrust fraction about its valve state to
a power between 0.65 and 1.60, and the set of powers is replaced at the re-key. A
single-amplitude probe therefore cannot separate degradation scale from curvature. After
each bus event, each electrical action slot also has its own valve that follows the
commanded duty with a hidden first-order lag; all six valves close when an event occurs,
and a valve follows its electrical slot while the rotor changes which physical nozzle that
slot drives. The two degraded nozzles form a hysteretic protection relay, armed when each
bus event begins. A duty above 0.78 on the more severely degraded nozzle while the relay
is armed reverses the rotor direction after that command and disarms the relay; a duty
below 0.12 on the moderately degraded nozzle arms it again. Within one command the arm
condition is checked first, so a command that does both arms the relay and then fires it.
The cycle can repeat. After every post-event command, the rotor advances one slot in its
current direction, or two slots when the sum of all six commanded duties exceeds 1.35.
Later in the same episode, the bus silently re-keys: it installs an independent cyclic
ordering, starting offset, direction, degraded pair, and degradation scales, and arms the
relay again. At that same instant the target reverses its slew with a new angular speed,
the redundant IMU mux silently selects a new orthogonal mounting of its linear channels
plus an independent yaw-channel polarity, and the avionics restart. You must identify the
plant from motion, allocate a desired wrench under nonnegative duty-cycle constraints,
detect both change points, reject the stale model, infer each rotor online, and track the
action-dependent plant.

The target body slowly tumbles, so the desired position, velocity, yaw, and yaw rate move.
You see the target only through a docking camera: a monochrome image, delayed by one to
three steps and absent for bursts, from which you must recover where the target and its
port are. A star tracker reports the craft's own yaw, and a current locally mounted IMU
remains available, but its coordinate wiring must also be calibrated. Direct spacecraft
velocity and yaw rate are not observed. A strong solution therefore needs camera-based
relative navigation, active but safe excitation, delayed-state estimation, fault
isolation, constrained control allocation, and collision-aware guidance.

Episode randomization is fixed to these ranges:

- initial target angular speed: 0.060 to 0.125 rad/s in either direction;
- at the second bus event the target reverses direction with a fresh speed of
  0.020 to 0.045 rad/s;
- throughout the episode the docking boom oscillates radially about 1.14 m with a hidden
  amplitude from 0.10 to 0.16 m, angular frequency from 0.18 to 0.28 rad/s, and
  initial phase in [-pi, pi]; its motion is continuous across both bus events;
- the target's boom telemetry reports its radius about 1.14 m with an unknown
  polarity, so the reported value is either `r` or `2*1.14-r`; the re-key independently
  selects either polarity, with no mux flag;
- fixed camera latency: 1 to 3 simulator steps, shown by each frame's capture stamp;
- nominal per-nozzle effectiveness multiplier: 0.70 to 1.28;
- nozzle installation angle error: at most 0.18 rad;
- nozzle yaw moment-arm multiplier: 0.82 to 1.18;
- each of the two post-fault rotors uses an independently shuffled cyclic ordering,
  a starting offset of one to five slots, and either direction;
- after the event its routing advances one nozzle per step, or two when total duty
  exceeds 1.35, in its current direction;
- after the first event one physical nozzle retains 35% to 50% authority and a
  distinct second nozzle retains 55% to 70% authority;
- after the second event a second pair, drawn independently of the first (it may share a
  nozzle with it), retains 45% to 60% and 65% to 80%;
- after each event every physical nozzle has an independent thrust-curve power from
  0.65 to 1.60; commanded duty still controls fuel use, stride, and relay logic,
  while realized thrust follows the valves and the curve;
- after the first event each electrical slot's valve has a time constant of 60 to
  90 ms, rising with the slot's index; the re-key replaces them independently with 80 to
  125 ms, again rising with the index, and closes all valves;
- the relay is armed when each event begins; a duty above 0.78 on the more severely
  degraded nozzle while it is armed reverses the direction after that command and
  disarms it, and a duty below 0.12 on the moderately degraded nozzle arms it again
  (checked first within a command); the cycle repeats;
- IMU linear axes have an unknown rotation in [-pi, pi] and may be mirrored;
- the IMU yaw-acceleration channel has unknown polarity;
- at the re-key, the already hidden IMU frame is further rotated by a
  nonzero 0.43 or 0.79 rad in either direction and may be mirrored, while yaw
  polarity independently stays or flips;
- the first bus remapping and degradation event occurs between steps 58 and 103;
- the independent re-key arms between steps 108 and 128, at least 25 steps after
  the first event. Once armed, it fires on the first simulator step whose true
  port-position error is within a hidden 1.18 to 1.30 m threshold; a hidden
  deadline between steps 148 and 165 forces it if the approach has not entered
  that region. The observation carries no explicit arm, proximity, or event flag;
- two camera outages last 7 to 18 steps, plus sparse single-frame dropouts;
- the re-key itself restarts the avionics: neither a camera frame nor a star-tracker
  reading arrives for the next 29 to 41 observations. Thus the target reversal, rotor
  replacement, and IMU remount must initially be detected and survived on the IMU alone;
  the restart does not add steps to the 400-step episode.

`/home/user/given/model.md` is the plant's datasheet: structure, nominal values and
tolerances, every range, and the docking test and quality exactly. It is not an
executable model: the grader's equations, update order and noise processes are not
published, as for a real vehicle. `/home/user/given/camera.md` describes the sensors
(where they sit, their outputs, noise, timing and outages), `/home/user/given/target.md`
the target's shape, surfaces, fiducials and light, and `/home/user/given/frames/` holds
sample camera frames of example episodes with their true geometry.

The episode lasts at most 400 steps at 0.12 simulated seconds per step. This bounded
horizon is deliberate: difficulty should come from identification and closed-loop
reasoning, not from a long rollout. There is no network access while you work, and none
while your controller is graded.

## Observation and action

At each step the grader supplies one JSON object, its keys in this order:

    {"imu": [a, b, c], "fuel": f, "episode_fraction": k/400, "boom_encoder": r,
     "star_tracker_yaw": y, "frame": {"stamp": s, "data": "<base64>"}}

- `frame` is the docking camera's image captured at step count `stamp` (1 to 3 steps
  before `t`, or 0 early in the episode): 640 x 480 pixels, 8-bit monochrome, row by row
  from the top, base64-encoded. It is `null` when no fresh frame arrives. The camera sits
  on the docking face below the probe and looks along the docking axis; its intrinsics,
  mount and noise are in `given/camera.md`.
- `star_tracker_yaw` is the craft's current yaw in the inertial frame, with small noise;
  it is `null` during the avionics restart after the re-key.
- `imu` holds the current IMU reading: the wrench realized during the preceding simulator
  step, including constant bias and sensor noise, in its own hidden mounting
  coordinates. Its linear reading equals an unknown 2D rotation applied after an optional
  axis reflection; the yaw-acceleration sign is independently unknown. The camera and the
  star tracker stay in the documented inertial geometry, so these conventions are
  observable from active motion.
- `boom_encoder` is the target's current boom-radius telemetry with its unknown polarity.
  It does not report radial velocity or its active polarity. Its polarity can be
  calibrated against the boom's position in time-aligned camera frames; after the silent
  re-key, both polarity hypotheses must be handled until the camera returns.
- `fuel` is the propellant left and `episode_fraction` the mission clock.

Return six normalized duty cycles `[u0, u1, u2, u3, u4, u5]`. Each is clipped to
[0, 1]. Thrusters are unidirectional: negative commands are not available.
`given/model.md` states the nominal six nozzle geometries and every randomization range,
but each episode's values are hidden: the initial wiring, the continuous column errors,
the bias, both rotor orderings with their offsets and directions, both degraded pairs and
their scales, both sets of curve powers, all twelve valve lags, the first event's step,
the re-key's arming step, proximity threshold and deadline, the target's slew speed after
the re-key, both IMU mountings, both boom-encoder polarities, the outage schedule, the
camera latency, the boom's amplitude, frequency and phase, and the swirl parity; the
start state shows only through the sensors. The bus supplies no explicit event flag.

Terminal approach also invalidates a static far-field inverse model. Each physical
nozzle's exhaust points opposite its translational force vector. When the spacecraft's
surface gap from the circular target falls below 0.65 m, target-body plume impingement
redirects exhaust aimed at the target: at the port a nozzle loses between 29% of its
thrust (exhaust pointing away, the shared docking-port shroud wake) and 82% (aimed
straight at the target), and the lost thrust reappears as a transverse force, 0.9 to 1.2
times the loss, whose sign is held by an independent passive vortex cell per physical
nozzle. The nominal signs alternate `[+, -, +, -, +, -]` over physical nozzles 0 through
5; one hidden episode-wide parity bit either preserves or reverses all six. The cell
follows the physical nozzle through rotor remapping, not the electrical action slot. Each
cell is bistable and action-dependent: it flips after an accumulated near-field firing
equal to 0.24 to 0.38 s of full-duty, fully aimed firing at the port (more for higher
nozzle indices), keeps the excess, and acts from the next command on; simultaneous
nozzle firings do not identify which cell changed without maintaining plant hypotheses.
The effect changes continuously with range and attitude; it does not change fuel burn,
valve state, relay logic, or rotor stride.

The circular target body has radius 0.82 m and the docking port's mean radius is 1.14 m.
Entering the body or leaving the radius-6.2 m operating region terminates the episode with zero
quality. Successful docking requires position error < 0.085 m, translational velocity
error < 0.060 m/s, yaw error < 0.020 rad, and yaw-rate error < 0.015 rad/s for thirty
consecutive steps after the re-key has occurred. In addition, the relay must have
reversed the rotor at least once after each bus event (it starts armed, so the first
trigger counts); merely capping duty below the trigger and bypassing the documented
fault mode does not qualify. Reaching the port earlier
does not terminate the episode; the controller must survive the re-key and recover.

## Metric

The hardware-invariant headline metric is `mean_docking_quality`, higher is better.
A failed but safe approach can earn at most 0.03 from progress. A successful episode
earns 0.94 plus small terms for remaining fuel and completion in fewer simulation steps,
up to 1.0. Thus robust docking dominates cosmetic efficiency improvements.

The hidden grader runs 48 hidden episodes, drawn once from the ranges above and kept
hidden: every submission meets the same 48. It averages their quality. An episode that
aborts (a missed deadline, a protocol error, a malformed action, or the reply budget
spent) scores 0 and still counts in the mean. Your score compares that mean with a
reference controller's measured value on the same episodes. Full credit is reached just
below the reference's level; the verdict also records the plain ratio to the reference,
capped at 1.5. The reference value itself is not disclosed. The grader also checks that
your actions depend on the observation they were given: if fewer than 35% of the
repeated-`t` answer pairs (see the wire contract) differ by more than 1e-4, the score is
0 regardless of the metric.

## Wire contract

Running `bash run.sh` in the submission directory must start a long-lived process:

```
                                  <- READY
{"type": "reset"}                 -> {"ok": true}
{"type": "act", "t": 0, "obs": {...}} -> {"action": [u0,u1,u2,u3,u4,u5]}
{"type": "close"}                 -> {"ok": true}, then exit
```

Each message is one line; an `act` line carries its frame and is about 410 KB long.
Print only protocol messages to stdout; diagnostics belong on stderr. Print READY within
120 seconds and answer each request within 30 seconds (reading it is part of your time).
Each step is queried twice with probability
0.06, with the same `t` and in a hidden order: once with the true observation and once
with a copy whose camera frame shows the craft displaced by up to 0.28 m along each axis
and turned by up to 0.18 rad at the frame's capture, and whose star-tracker yaw is turned
by the same angle; the other fields are unchanged, and a request without a frame or a
star-tracker reading has a copy without it. Only the reply to the true observation is
applied. Always calculate the action from the observation in that request; repeated
queries do not advance the simulation.

If an episode aborts on a missed deadline, a crash or a protocol error, the grader stops
the process (and everything it started) and runs `bash run.sh` again before the next
episode; it must print READY again within 120 seconds. If it cannot, or after three such
restarts, every remaining episode aborts. A malformed action scores its episode 0 without
a restart. So `run.sh` has to work when started a second time: no fixed ports, PID files
or locks left behind.

Grading happens in a fresh sandbox that holds only `/home/user/submission/`:
`/home/user/given/` and anything else you leave elsewhere in your home are not present
then, so put everything `run.sh` needs inside `/home/user/submission/`.
`/opt/venv/bin/python3` with numpy, OpenCV (`cv2`), Pillow and MuJoCo is part of the
image and is still available. The controller runs as you from that folder and may read
and write inside it. Its environment holds only
`PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin`, `HOME=/home/user` and `LANG=C.UTF-8`.
Pipes and Unix-domain sockets work; no other socket can be opened, not even an internet
socket to localhost.

Your controller's replies have 30 minutes in all: the time from the moment the grader
starts writing each request to the moment its reply arrives, summed over the whole
grading run (at most 48 x 400 steps plus repeated queries, about 20,400 requests), may not
exceed 1800 s; answer in well under 0.09 s per request on average. Once the sum passes
1800 s, the episode in progress and every later one abort. The grader's own time
(simulating and rendering the next frame) is not counted, nor is work your controller
finishes before the grader starts writing the next request; such work shares the
machine's two CPUs with the grader's renderer.

## Rehearsing

Nothing in your home simulates the plant, renders the camera or grades a controller.
Rehearse with a simulator and a harness of your own, built from the datasheet and the
sensor documents in `/home/user/given/`. Python 3.12 with numpy, OpenCV, Pillow and MuJoCo
is `/opt/venv/bin/python3`; call it by that path in `run.sh`.
