# The docking plant: datasheet

This is the plant's datasheet: the vehicle and its thrusters, the bus and its two silent events, the target and its
docking boom, the plume near the target, the start of an episode, and the docking test, the end of an episode and its
quality. It gives the structure, nominal values with their tolerances and every range an episode can take; the docking
test and the quality are stated exactly. It is not an executable model: the grader's equations, its update order and
its noise processes are not published, as for a real vehicle. The sensors are in `camera.md` (the docking camera, the
star tracker, the IMU, the boom's encoder telemetry, the fuel gauge), the target's appearance and fiducials in
`target.md`, sample camera frames with their true geometry in `frames/`.

Nothing in your home simulates the plant, renders the camera or grades a controller. Each graded episode's values stay
hidden; only their ranges are given here.

## Frames and units

- Metres, seconds, radians. The plant runs in steps of 0.12 s; an episode has at most 400 steps. A command is held for
  its step.
- The **inertial frame** is centred on the target's spin axis, z up; the motion is planar, in the plane z = 0 that
  holds the craft's centre and the boom's axis.
- The craft's **body frame**: x forward, y left, z up. The docking probe sits on the -x face, so the craft docks
  moving backwards: at the port its +x axis points away from the target. Its yaw `theta` is the angle of the body's +x
  axis in the inertial frame.
- The target is a drum of radius 0.82 m about the inertial z axis. The docking port sits on a boom that comes out of
  the drum's side, at a mean radius of 1.14 m from the axis. The operating region is the disc of radius 6.2 m.

## The craft and its thrusters

- A free-flying rigid body in the plane: six unidirectional cold-gas nozzles push it in its body frame. A slight
  residual damping (part of the dynamics, not seen by the IMU) takes 1.5-2.0 % of the translational velocity and
  3.0-3.6 % of the yaw rate per second.
- Each action entry is a duty cycle, clipped to [0, 1], held for one step.
- The six nominal nozzle geometries, as the body-frame acceleration a nozzle gives at full thrust, far from the target
  (`ax`, `ay` in m/s², yaw acceleration in rad/s²):

| geometry | ax | ay | yaw |
|---|---|---|---|
| 0 | 0.235 | 0 | 0.132 |
| 1 | 0.235 | 0 | -0.132 |
| 2 | -0.235 | 0 | -0.132 |
| 3 | -0.235 | 0 | 0.132 |
| 4 | 0 | 0.205 | 0.148 |
| 5 | 0 | -0.205 | -0.148 |

- **Installation**, per physical nozzle: effectiveness 0.70-1.28 of nominal; pointing error up to 0.18 rad either way
  (the force turns; the yaw term keeps its nominal sign); yaw moment arm 0.82-1.18 of nominal.
- **Wiring at the start**: action slot `i` drives physical nozzle `i` with its whole authority, linearly and without
  memory, and the six physical nozzles carry the six nominal geometries in any of the 720 assignments.
- **Bias**: a constant body-frame acceleration for the whole episode, up to 0.010 m/s² on each linear axis and up to
  0.007 rad/s² in yaw.
- **Propellant**: the gauge reads 1.0 full. A nozzle at full duty for one step uses 0.00034 of a full tank, so 400
  steps at full duty on all six use 0.816. Consumption follows the commanded duty.

## The bus and its two events

From the first event on, a fault-tolerant bus sits between the action slots and the nozzles.

- **Rotor.** The bus routes the six slots to the six physical nozzles through a cyclic ordering of the nozzles (an
  arbitrary permutation, not nozzle order) and a rotor position. At each event the bus installs a new ordering, a
  starting offset of 1 to 5 places and a direction (either). After every command the rotor advances in its current
  direction by one place, or by two when the commanded duties of that command sum to more than 1.35.
- **Degraded pair.** Two distinct physical nozzles keep only part of their authority: after the first event the
  severe one keeps 35-50 % and the moderate one 55-70 %; after the re-key a second pair, drawn independently of the
  first (it may share one or both nozzles with it), keeps 45-60 % and 65-80 %.
- **Protection relay.** It is armed when each event begins. A command whose duty on the slot that drives the moderate
  nozzle is below 0.12 arms it; a command whose duty on the slot that drives the severe nozzle is above 0.78 while it
  is armed reverses the rotor's direction after that command and disarms it. Within one command the arm test comes
  first, so a command that does both arms the relay and fires it. The cycle can repeat.
- **Valves.** After each event every electrical slot has a valve that follows its commanded duty with a first-order
  lag: time constants of 60-90 ms after the first event and 80-125 ms after the re-key, rising with the slot's index;
  every valve is reset to closed when an event fires. The valve belongs to the slot, not to the nozzle the slot
  currently drives.
- **Thrust curve.** After each event a nozzle's thrust fraction is its valve's state raised to a power, one exponent
  per physical nozzle, 0.65-1.60, a new set at each event, times its degradation. Before the first event thrust is
  linear in duty.
- The relay, the rotor's stride and the propellant use the commanded duties; only the thrust sees the valves and the
  curve.
- **The first event** fires in a step from 58 to 103 (the step that applies your reply to the request with that `t`,
  before its command acts).
- **The re-key** is armed from a step between 108 and 128, and at least 25 steps after the first event. Once armed, it
  fires in the first step that starts with the craft within a hidden 1.18-1.30 m of the port, or at a hidden deadline
  between steps 148 and 165 if the approach has not come that close by then. Its ordering differs from the first
  event's. In the same step the target's slew reverses (a new speed of 0.020-0.045 rad/s, of the opposite sign) and the
  boom's encoder may change its polarity. From the observation that follows, the IMU's linear channels are remounted
  (turned a further 0.43 or 0.79 rad either way, possibly mirrored), its yaw channel may change its polarity, and the
  avionics restart: the camera and the star tracker deliver nothing for the next 29 to 41 observations.
- Nothing in the observation flags an event.

## The target, the boom and the port

- The target's phase `phi` (the port's bearing from the target's centre) turns at the slew rate: 0.060-0.125 rad/s in
  either direction until the re-key, then 0.020-0.045 rad/s with the opposite sign.
- The boom breathes radially: the port's radius is `r_port = 1.14 + A sin(psi)`, with an amplitude `A` of 0.10-0.16 m
  and `psi` advancing at 0.18-0.28 rad/s from a phase anywhere in [-pi, pi). The boom co-rotates with the target, and
  its motion is continuous across both events.
- The **goal**, the port's state that docking is measured against:
  - position `g = r_port (cos phi, sin phi)`;
  - velocity `v_g = A omega_b cos(psi) (cos phi, sin phi) + slew (-g_y, g_x)`, with `omega_b` the boom's rate;
  - yaw `phi` (the body's +x axis along the port's bearing, pointing away from the target) and yaw rate `slew`.

## The plume near the target

Close to the target, exhaust aimed at it is redirected. A nozzle's exhaust leaves opposite its force.

- **Where.** The effect starts at a surface gap of 0.65 m (the craft's distance from the drum's surface) and is full
  at the port's radius; it changes continuously with range and attitude.
- **How much.** A nozzle loses a fraction of its thrust and of its torque: at the port, 29 % with its exhaust pointing
  away from the target (the docking shroud's wake) up to 82 % aimed straight at it, more the more it points at it.
- **Swirl.** The lost thrust reappears as a transverse force, perpendicular to the nozzle's thrust, of 0.9-1.2 times
  the lost part; its sign is held by a bistable vortex cell, one per physical nozzle. The cells start in an alternating
  pattern (nozzles 0, 2 and 4 one way, 1, 3 and 5 the other) whose overall sign is hidden. A cell belongs to its
  physical nozzle and follows it through the rotor.
- **Flips.** A cell flips its sign once its accumulated near-field firing reaches what 0.24-0.38 s of full-duty firing
  aimed straight at the target at the port's radius would give (more for higher nozzle indices; it takes longer at
  lower duty, weaker aim or farther out); it keeps the excess and starts again. A flip acts from the next command on.
  The cells start empty, and neither event resets them.
- The plume changes neither the propellant, the valves, the relay nor the stride.

## The start of an episode

- The target's phase starts anywhere in [-pi, pi).
- The craft starts 2.85-3.65 m from the target's centre, at a bearing within 1.45 rad either way of the port's.
- Its yaw lies within 1.1 rad either way of the port's bearing, its velocity is up to 0.10 m/s on each inertial axis
  and its yaw rate up to 0.11 rad/s.

## Docking, the end and the quality

After every step, with the new state and the new port:

- the errors are: position `|p - g|`, velocity `|v - v_g|`, yaw `|wrap(theta - phi)|` (the angle difference wrapped
  to [-pi, pi)), yaw rate `|omega - slew|`;
- the step counts as docked when the re-key has fired, the first rotor has reversed at least once and the second
  rotor has reversed at least once (the relay), and the four errors are below 0.085 m, 0.060 m/s, 0.020 rad and
  0.015 rad/s;
- the hold counts the consecutive docked steps; a step that is not docked sets it back to 0;
- the best error is the smallest position error so far, counting from the error at the start (the initial error).

The episode ends after the step in which the hold reaches 30 (docked), in which `|p| < 0.82` (a collision with the
target) or `|p| > 6.2` (an escape from the operating region), or after the 400th step. (An empty tank would end it
too; it cannot empty in 400 steps.) Its quality, the number the grader averages over the episodes:

- docked: `0.94 + 0.04 * fuel + 0.02 * (1 - k / 400)`, with the fuel and the step count `k` at the end;
- a collision or an escape: 0;
- otherwise: `0.03 * clamp((initial error - best error) / initial error, 0, 1)`.

Reaching the port before the re-key does not end the episode: the controller has to ride out the re-key and dock
afterwards.

## An episode on the wire

The wire contract is in the task instruction. One `act` request as it comes over the wire, on one line (an example
episode, not a graded one; the frame's 409,600 characters of base64 are cut to their first and last 16 here):

```
{"type": "act", "t": 37, "obs": {"imu": [0.030785404003960556, -0.008646980485457896, 0.01192248203795915], "fuel": 0.988677999999999, "episode_fraction": 0.0925, "boom_encoder": 1.0598546174619217, "star_tracker_yaw": -0.8636987662789832, "frame": {"stamp": 34, "data": "AAABAQABAgEAAAAA...OTo7NzQxMCwtLSkn"}}}
```
