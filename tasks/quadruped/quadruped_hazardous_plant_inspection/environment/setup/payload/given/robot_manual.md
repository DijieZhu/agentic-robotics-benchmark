# Q12 inspection quadruped: operation and integration manual

The Q12 is a small twelve-actuator quadruped fitted for inspection rounds in buildings people cannot enter: an on-board
gait, a pan-tilt sensor head (navigation camera with depth, telephoto inspection camera, thermal camera, lamp), a
planar lidar on its back, a dosimeter, leg odometry and a radio link to its team. This manual gives what an integrator
needs to write its autonomous mode: frames, dimensions, what each gait mode crosses and how often, the protective
behaviours, every sensor as a datasheet, the link, and what the inspection team accepts as a record photo. The
programming interface is in `robot_api.md`. The measured tables of section 4 are also in `robot.describe()`, as data
(`modes.tables`).

Lengths in metres, angles in radians unless degrees are written, times in seconds of the robot's own (simulated)
clock, dose rates in Gy/h, doses in Gy, temperatures in °C.

## 1. Frames

- **Site frame**: origin at the centre of the deployment pad in the airlock, on the floor; x into the building, y to
  the left, z up. The drawing's coordinates, the odometry and the position of a source report are in this frame.
- **Body frame**: origin at the centre of the body, x forward, y to the left, z up. Yaw is positive counter-clockwise
  seen from above, pitch is positive nose down, roll is positive with the left side up.
- **Camera frames** (optical): x to the right in the image, y down, z forward. The cameras are pinholes with square
  pixels and no distortion. Every frame reply carries the camera's intrinsic matrix `K` and `T_body_cam`, the 4 × 4
  pose of the camera in the body frame at the instant of the frame (it includes the head's pan and tilt).
- **Lidar frame**: the body frame's axes, moved up to the scan plane (`T_body_lidar`).
- **From the body to the site**: the body's pose in the site frame is `T_site_body = Trans(x, y, h) · Rz(yaw) ·
  Ry(pitch) · Rx(roll)`, with x, y, yaw from the odometry, h the body height and roll, pitch from the IMU
  (`get_state()`); a camera's pose in the site frame is `T_site_body · T_body_cam`. It is as good as the odometry.

## 2. The robot

<!-- BEGIN GENERATED dimensions -->
| item | value |
|---|---|
| mass | 3.15 kg |
| length × width | 0.55 × 0.33 m |
| height when standing at the default body height (0.24 m) | 0.32 m to the top of the lidar, which is 0.081 m above the body centre |
| lidar scan plane | 0.066 m above the body centre |
| head | pan axis at (0.165, 0, 0.043) in the body frame; tilt axis 0.012 m above that point; at pan 0 the cameras are 0.18 m ahead of the body centre |
| head lamp | on the head, along the cameras' axis; lights about 3 m |
<!-- END GENERATED dimensions -->

The legs are named FL, FR, HL, HR (front / hind, left / right); each has a hip abduction joint, a hip flexion joint
and a knee. The gait is the robot's firmware: your program gives it a velocity command in the body frame, a mode, a
body height and pitch and the head angles; it steps and balances on its own. It walks blind: it feels obstacles with
its feet and does not look at them. Choosing the mode, the approach and the speed for what lies ahead is your
program's part.

## 3. Gait modes

<!-- BEGIN GENERATED modes -->
| mode | forward / back, up to (m/s) | sideways, up to (m/s) | turn, up to (rad/s) | body height that can be set (m) | default body height (m) | meant for |
|---|---|---|---|---|---|---|
| `stand` | - | - | - | 0.15 to 0.26, pitch -20° to +20° | 0.24 | standing still: aiming the head, reading instruments, counting at a survey mark |
| `walk` | 0.45 | 0.20 | 1.0 | 0.24 to 0.26 | 0.24 | level floor, low thresholds, ramps |
| `high_step` | 0.25 | 0.10 | 0.6 | 0.24 to 0.27 | 0.255 | door sills, hoses, pipes on the floor, kerbs |
| `crouch_walk` | 0.20 | 0.10 | 0.6 | 0.17 to 0.19 | 0.18 | under low pipe racks and through low hatches |
| `sit` | - | - | - | the body rests on the floor (body centre about 0.03 to 0.04 m up) | - | resting on the floor |
<!-- END GENERATED modes -->

- A command above a mode's limits is cut to the limit. Commands hold until you change them.
- `set_mode` puts the body at the mode's default height; `set_body(height=...)` moves it within the mode's range (a
  value outside is cut to the range). A change of mode takes effect at once, also while walking; the body reaches
  the new mode's height within about half a second.
- A moving command in `stand` or `sit` switches the robot to `walk` (from `sit` it stands up first). A zero command
  leaves it standing in the walking mode it is in; `stand` is the mode in which body height and pitch can be set
  freely.
- In `stand` the body moves to a new height at about 0.08 m/s and to a new pitch at about 20° per second. Sitting
  down takes about 2 s, standing up from `sit` about 2.5 s. From standing, a walk reaches its commanded speed within
  about half a second.

Where the cameras are, by mode and body height (the head at tilt 0):

<!-- BEGIN GENERATED posture -->
| mode | body height set (m) | body centre above the floor (m) | cameras above the floor (m) |
|---|---|---|---|
| `stand` | 0.15 | 0.149 | 0.204 |
| `stand` | 0.18 | 0.179 | 0.234 |
| `stand` | 0.22 | 0.219 | 0.274 |
| `stand` | 0.26 | 0.259 | 0.314 |
| `stand` | default, pitch -20° | 0.239 | 0.347 |
| `stand` | default, pitch +20° | 0.239 | 0.235 |
| `walk` | 0.24 | 0.242 | 0.299 |
| `walk` | 0.26 | 0.259 | 0.316 |
| `high_step` | 0.24 | 0.243 | 0.302 |
| `high_step` | 0.255 | 0.257 | 0.316 |
| `high_step` | 0.27 | 0.270 | 0.328 |
| `crouch_walk` | 0.17 | 0.181 | 0.236 |
| `crouch_walk` | 0.18 | 0.190 | 0.245 |
| `crouch_walk` | 0.19 | 0.199 | 0.254 |
| `sit` | default | 0.040 | 0.096 |
<!-- END GENERATED posture -->

The rows of the walking modes were taken while the robot walked. Standing still in `high_step` or `crouch_walk` the
body is 6 to 8 mm higher than the table says and the cameras about 1 cm. In `sit` the body goes on settling for a few
seconds, to about 0.03 m, with the cameras about 0.09 m above the floor.

Speed on level floor against the command (means over the runs):

<!-- BEGIN GENERATED speed -->
| mode | command: vx (m/s), vy (m/s), yaw rate (rad/s) | measured vx (m/s) | measured vy (m/s) | measured yaw rate (rad/s) |
|---|---|---|---|---|
| `walk` | 0.10, 0.00, 0.00 | 0.086 | -0.001 | 0.007 |
| `walk` | 0.20, 0.00, 0.00 | 0.189 | -0.002 | 0.001 |
| `walk` | 0.30, 0.00, 0.00 | 0.295 | -0.003 | -0.004 |
| `walk` | 0.40, 0.00, 0.00 | 0.397 | -0.004 | -0.006 |
| `walk` | 0.45, 0.00, 0.00 | 0.445 | -0.002 | -0.008 |
| `walk` | -0.30, 0.00, 0.00 | -0.309 | 0.000 | 0.012 |
| `walk` | 0.00, 0.10, 0.00 | 0.001 | 0.089 | -0.012 |
| `walk` | 0.00, 0.20, 0.00 | -0.002 | 0.198 | -0.021 |
| `walk` | 0.00, 0.00, 0.50 | -0.004 | -0.004 | 0.474 |
| `walk` | 0.00, 0.00, 1.00 | -0.003 | -0.005 | 0.987 |
| `walk` | 0.30, 0.00, 0.50 | 0.303 | -0.005 | 0.482 |
| `high_step` | 0.10, 0.00, 0.00 | 0.077 | 0.005 | 0.006 |
| `high_step` | 0.20, 0.00, 0.00 | 0.189 | 0.001 | 0.000 |
| `high_step` | 0.25, 0.00, 0.00 | 0.241 | -0.003 | -0.001 |
| `high_step` | 0.00, 0.10, 0.00 | 0.015 | 0.077 | 0.007 |
| `high_step` | 0.00, 0.00, 0.60 | -0.001 | -0.005 | 0.573 |
| `crouch_walk` | 0.10, 0.00, 0.00 | 0.092 | 0.005 | 0.012 |
| `crouch_walk` | 0.15, 0.00, 0.00 | 0.146 | 0.003 | 0.011 |
| `crouch_walk` | 0.20, 0.00, 0.00 | 0.197 | 0.001 | 0.007 |
| `crouch_walk` | 0.00, 0.10, 0.00 | -0.005 | 0.081 | -0.015 |
| `crouch_walk` | 0.00, 0.00, 0.60 | -0.011 | 0.012 | 0.571 |
<!-- END GENERATED speed -->

Stopping from a steady walk after a zero command:

<!-- BEGIN GENERATED stopping -->
| mode | from (m/s) | distance to standstill (m): mean (range) | time (s): mean (range) |
|---|---|---|---|
| `walk` | 0.30 | 0.014 (0.007 to 0.022) | 0.20 (0.14 to 0.36) |
| `walk` | 0.45 | 0.040 (0.030 to 0.048) | 0.23 (0.18 to 0.34) |
| `high_step` | 0.25 | 0.022 (-0.004 to 0.053) | 0.19 (0.16 to 0.26) |
| `crouch_walk` | 0.20 | 0.028 (0.015 to 0.036) | 0.16 (0.10 to 0.22) |
<!-- END GENERATED stopping -->

Walking on level floor with nothing in the way, under commands that keep changing:

<!-- BEGIN GENERATED level_floor -->
| mode | commands | floor | runs | minutes | distance (m) | `blocked` | `protective_sit` | falls |
|---|---|---|---|---|---|---|---|---|
| `walk` | changing commands: mostly forward, some backward, with sideways and turning parts and stops; a new command every 2 to 4 s | dry | 30 | 30 | 423.4 | 0 | 0 | 0 |
| `high_step` | changing commands: mostly forward, some backward, with sideways and turning parts and stops; a new command every 2 to 4 s | dry | 30 | 30 | 225.2 | 0 | 0 | 0 |
| `crouch_walk` | changing commands: mostly forward, some backward, with sideways and turning parts and stops; a new command every 2 to 4 s | dry | 30 | 30 | 179.5 | 0 | 0 | 0 |
| `walk` | sideways, turning on the spot, both together, slow creeping, and turning and walking in alternation every 0.2 to 1.0 s as a path follower does; a new command every 2 to 5 s | friction 0.90 | 30 | 30 | - | 0 | 0 | 0 |
| `walk` | sideways, turning on the spot, both together, slow creeping, and turning and walking in alternation every 0.2 to 1.0 s as a path follower does; a new command every 2 to 5 s | friction 0.20 | 10 | 10 | - | 0 | 0 | 0 |
| `high_step` | sideways, turning on the spot, both together, slow creeping, and turning and walking in alternation every 0.2 to 1.0 s as a path follower does; a new command every 2 to 5 s | friction 0.90 | 30 | 30 | - | 0 | 0 | 0 |
| `high_step` | sideways, turning on the spot, both together, slow creeping, and turning and walking in alternation every 0.2 to 1.0 s as a path follower does; a new command every 2 to 5 s | friction 0.20 | 10 | 10 | - | 0 | 0 | 0 |
| `crouch_walk` | sideways, turning on the spot, both together, slow creeping, and turning and walking in alternation every 0.2 to 1.0 s as a path follower does; a new command every 2 to 5 s | friction 0.90 | 30 | 30 | - | 0 | 0 | 0 |
| `crouch_walk` | sideways, turning on the spot, both together, slow creeping, and turning and walking in alternation every 0.2 to 1.0 s as a path follower does; a new command every 2 to 5 s | friction 0.20 | 10 | 10 | - | 0 | 0 | 0 |
<!-- END GENERATED level_floor -->

## 4. What each mode crosses

Every row below is a set of randomised trials with this robot and this firmware. "Crossed" counts the trials in which
the robot got over or under the element within the tries allowed; a try ends when the gait reports `blocked`.
"Stopped without a report" counts trials in which the robot neither crossed nor reported `blocked` nor fell before
the trial's time ran out.

<!-- BEGIN GENERATED how_measured -->
How the trials were run: the robot starts 0.7 to 0.9 m before the element, up to 0.12 m to the side of the lane's middle and turned by up to 10° from its path. The commanded speed is drawn from the range the row gives. After a `blocked` the same command is given again after 1 s. Floors are dry (friction 0.8 to 0.9) unless a row says otherwise. **Approach** is how far the element's edge is turned from square to the robot's path (for a ramp over a kerb: the path against the ramp's axis). **Steering** is what the trial's driver held with the robot's own IMU and odometry: "heading and line" = the path's heading and the starting line, turning at no more than 0.4 rad/s and sidestepping at no more than 0.08 m/s; "heading only" = the heading alone, turning at no more than 0.3 rad/s, no sidestepping. A rate is written k/n: k of n trials.
<!-- END GENERATED how_measured -->

The trials sent their retries 1 s after a `blocked`. Through the programming interface a retry can be sent 5 s after
it at the earliest (the hold of section 5), from a standstill; section 5 gives what that changes.

### 4.1 Obstacles on the floor and low overheads

<!-- BEGIN GENERATED obstacles -->
**Door sill (square edges, 0.12 m deep)**

| mode | height (m) | approach | steering | commanded speed (m/s) | tries allowed | trials | crossed | crossed on the first try | `blocked` on the first try | stopped without a report | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `walk` | 0.025 | up to 20° | heading and line | 0.15 to 0.45 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| `walk` | 0.03 | up to 20° | heading and line | 0.15 to 0.45 | 3 | 660 | 660/660 | 638/660 | 22/660 | 0/660 | 0/660 |
| `walk` | 0.035 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 23/60 | 23/60 | 37/60 | 0/60 | 0/60 |
| `walk` | 0.04 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.045 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.0495 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 300 | 0/300 | 0/300 | 300/300 | 0/300 | 0/300 |
| `walk` | 0.05 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.055 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.06 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `high_step` | 0.045 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 35/35 | 0/35 | 0/35 | 0/35 |
| `high_step` | 0.045 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 20/25 | 5/25 | 0/25 | 0/25 |
| `high_step` | 0.05 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 35/35 | 0/35 | 0/35 | 0/35 |
| `high_step` | 0.05 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 19/25 | 6/25 | 0/25 | 0/25 |
| `high_step` | 0.055 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 31/35 | 4/35 | 0/35 | 0/35 |
| `high_step` | 0.055 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 15/25 | 10/25 | 0/25 | 0/25 |
| `high_step` | 0.0565 | within 10° | heading and line | 0.10 to 0.25 | 3 | 450 | 448/450 | 416/450 | 34/450 | 0/450 | 0/450 |
| `high_step` | 0.0565 | within 10° | heading only | 0.10 to 0.25 | 3 | 300 | 299/300 | 283/300 | 17/300 | 0/300 | 0/300 |
| `high_step` | 0.0565 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 150 | 140/150 | 80/150 | 70/150 | 0/150 | 0/150 |
| `high_step` | 0.0565 | 10° to 20° | heading only | 0.10 to 0.25 | 3 | 150 | 140/150 | 93/150 | 57/150 | 0/150 | 0/150 |
| `high_step` | 0.058 | within 10° | heading and line | 0.10 to 0.25 | 3 | 299 | 296/299 | 268/299 | 31/299 | 0/299 | 0/299 |
| `high_step` | 0.058 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 301 | 277/301 | 160/301 | 141/301 | 0/301 | 0/301 |
| `high_step` | 0.06 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 30/35 | 5/35 | 0/35 | 0/35 |
| `high_step` | 0.06 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 24/25 | 16/25 | 9/25 | 0/25 | 0/25 |
| `high_step` | 0.1 | within 10° | heading and line | 0.10 to 0.25 | 3 | 25 | 0/25 | 0/25 | 25/25 | 0/25 | 0/25 |
| `high_step` | 0.1 | 10° to 20° | heading and line | 0.11 to 0.25 | 3 | 15 | 0/15 | 0/15 | 15/15 | 0/15 | 0/15 |
| `crouch_walk` | 0.025 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 60 | 60/60 | 48/60 | 12/60 | 0/60 | 0/60 |
| `crouch_walk` | 0.045 | up to 20° | heading and line | 0.08 to 0.20 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `crouch_walk` | 0.05 | up to 20° | heading and line | 0.08 to 0.20 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |

**Kerb or bund edge (square edges, 0.20 m deep)**

| mode | height (m) | approach | steering | commanded speed (m/s) | tries allowed | trials | crossed | crossed on the first try | `blocked` on the first try | stopped without a report | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `walk` | 0.025 | up to 20° | heading and line | 0.15 to 0.45 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| `walk` | 0.03 | up to 20° | heading and line | 0.15 to 0.45 | 3 | 60 | 60/60 | 59/60 | 1/60 | 0/60 | 0/60 |
| `walk` | 0.035 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 29/60 | 29/60 | 31/60 | 0/60 | 0/60 |
| `walk` | 0.04 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.045 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.0495 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 300 | 0/300 | 0/300 | 300/300 | 0/300 | 0/300 |
| `walk` | 0.05 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.055 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.06 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.12 | up to 20° | heading and line | 0.15 to 0.45 | 3 | 40 | 0/40 | 0/40 | 40/40 | 0/40 | 0/40 |
| `walk` | 0.18 | up to 20° | heading and line | 0.15 to 0.45 | 3 | 40 | 0/40 | 0/40 | 40/40 | 0/40 | 0/40 |
| `high_step` | 0.045 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 35/35 | 0/35 | 0/35 | 0/35 |
| `high_step` | 0.045 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 21/25 | 4/25 | 0/25 | 0/25 |
| `high_step` | 0.05 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 35/35 | 0/35 | 0/35 | 0/35 |
| `high_step` | 0.05 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 24/25 | 20/25 | 5/25 | 0/25 | 0/25 |
| `high_step` | 0.055 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 32/35 | 3/35 | 0/35 | 0/35 |
| `high_step` | 0.055 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 18/25 | 7/25 | 0/25 | 0/25 |
| `high_step` | 0.0565 | within 10° | heading and line | 0.10 to 0.25 | 3 | 449 | 448/449 | 412/449 | 37/449 | 0/449 | 0/449 |
| `high_step` | 0.0565 | within 10° | heading only | 0.10 to 0.25 | 3 | 300 | 299/300 | 284/300 | 16/300 | 0/300 | 0/300 |
| `high_step` | 0.0565 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 151 | 146/151 | 95/151 | 56/151 | 0/151 | 0/151 |
| `high_step` | 0.0565 | 10° to 20° | heading only | 0.10 to 0.25 | 3 | 150 | 142/150 | 100/150 | 50/150 | 0/150 | 0/150 |
| `high_step` | 0.058 | within 10° | heading and line | 0.10 to 0.25 | 3 | 299 | 297/299 | 274/299 | 25/299 | 0/299 | 0/299 |
| `high_step` | 0.058 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 301 | 283/301 | 188/301 | 113/301 | 0/301 | 0/301 |
| `high_step` | 0.06 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 34/35 | 30/35 | 5/35 | 0/35 | 0/35 |
| `high_step` | 0.06 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 23/25 | 19/25 | 6/25 | 0/25 | 0/25 |
| `high_step` | 0.1 | within 10° | heading and line | 0.10 to 0.25 | 3 | 25 | 0/25 | 0/25 | 25/25 | 0/25 | 0/25 |
| `high_step` | 0.1 | 10° to 20° | heading and line | 0.11 to 0.25 | 3 | 15 | 0/15 | 0/15 | 15/15 | 0/15 | 0/15 |
| `high_step` | 0.12 | within 10° | heading and line | 0.10 to 0.25 | 3 | 25 | 0/25 | 0/25 | 25/25 | 0/25 | 0/25 |
| `high_step` | 0.12 | 10° to 20° | heading and line | 0.11 to 0.25 | 3 | 15 | 0/15 | 0/15 | 15/15 | 0/15 | 0/15 |
| `high_step` | 0.15 | within 10° | heading and line | 0.10 to 0.25 | 3 | 25 | 0/25 | 0/25 | 25/25 | 0/25 | 0/25 |
| `high_step` | 0.15 | 10° to 20° | heading and line | 0.11 to 0.25 | 3 | 15 | 0/15 | 0/15 | 15/15 | 0/15 | 0/15 |
| `high_step` | 0.18 | within 10° | heading and line | 0.10 to 0.25 | 3 | 25 | 0/25 | 0/25 | 25/25 | 0/25 | 0/25 |
| `high_step` | 0.18 | 10° to 20° | heading and line | 0.11 to 0.25 | 3 | 15 | 0/15 | 0/15 | 15/15 | 0/15 | 0/15 |
| `crouch_walk` | 0.12 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 40 | 0/40 | 0/40 | 40/40 | 0/40 | 0/40 |

**Hose lying on the floor (round)**

| mode | height (m) | approach | steering | commanded speed (m/s) | tries allowed | trials | crossed | crossed on the first try | `blocked` on the first try | stopped without a report | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `walk` | 0.025 | up to 20° | heading and line | 0.15 to 0.45 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| `walk` | 0.03 | up to 20° | heading and line | 0.15 to 0.45 | 3 | 660 | 660/660 | 654/660 | 6/660 | 0/660 | 0/660 |
| `walk` | 0.04 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 14/60 | 14/60 | 46/60 | 0/60 | 0/60 |
| `walk` | 0.045 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 3/60 | 3/60 | 57/60 | 0/60 | 0/60 |
| `walk` | 0.0495 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 300 | 0/300 | 0/300 | 300/300 | 0/300 | 0/300 |
| `walk` | 0.05 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 1/60 | 1/60 | 59/60 | 0/60 | 0/60 |
| `walk` | 0.055 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.06 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `high_step` | 0.045 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 35/35 | 0/35 | 0/35 | 0/35 |
| `high_step` | 0.045 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 24/25 | 1/25 | 0/25 | 0/25 |
| `high_step` | 0.05 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 35/35 | 0/35 | 0/35 | 0/35 |
| `high_step` | 0.05 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 23/25 | 2/25 | 0/25 | 0/25 |
| `high_step` | 0.055 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 35/35 | 0/35 | 0/35 | 0/35 |
| `high_step` | 0.055 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 21/25 | 4/25 | 0/25 | 0/25 |
| `high_step` | 0.0565 | within 10° | heading and line | 0.10 to 0.25 | 3 | 150 | 150/150 | 139/150 | 11/150 | 0/150 | 0/150 |
| `high_step` | 0.0565 | within 10° | heading only | 0.10 to 0.25 | 3 | 150 | 150/150 | 143/150 | 7/150 | 0/150 | 0/150 |
| `high_step` | 0.0565 | 10° to 20° | heading only | 0.10 to 0.25 | 3 | 150 | 149/150 | 127/150 | 23/150 | 0/150 | 0/150 |
| `high_step` | 0.058 | within 10° | heading and line | 0.10 to 0.25 | 3 | 299 | 299/299 | 283/299 | 16/299 | 0/299 | 0/299 |
| `high_step` | 0.058 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 301 | 299/301 | 220/301 | 81/301 | 0/301 | 0/301 |
| `high_step` | 0.06 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 34/35 | 31/35 | 4/35 | 0/35 | 0/35 |
| `high_step` | 0.06 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 14/25 | 11/25 | 0/25 | 0/25 |
| `crouch_walk` | 0.025 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 60 | 60/60 | 57/60 | 3/60 | 0/60 | 0/60 |
| `crouch_walk` | 0.045 | up to 20° | heading and line | 0.08 to 0.20 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `crouch_walk` | 0.0495 | up to 20° | heading and line | 0.08 to 0.20 | 1 | 300 | 0/300 | 0/300 | 300/300 | 0/300 | 0/300 |
| `crouch_walk` | 0.05 | up to 20° | heading and line | 0.08 to 0.20 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |

**Pipe lying on the floor (round)**

| mode | height (m) | approach | steering | commanded speed (m/s) | tries allowed | trials | crossed | crossed on the first try | `blocked` on the first try | stopped without a report | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `walk` | 0.025 | up to 20° | heading and line | 0.15 to 0.45 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| `walk` | 0.03 | up to 20° | heading and line | 0.15 to 0.45 | 3 | 660 | 660/660 | 658/660 | 2/660 | 0/660 | 0/660 |
| `walk` | 0.04 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 11/60 | 11/60 | 49/60 | 0/60 | 0/60 |
| `walk` | 0.045 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 4/60 | 4/60 | 56/60 | 0/60 | 0/60 |
| `walk` | 0.05 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 360 | 3/360 | 3/360 | 357/360 | 0/360 | 0/360 |
| `walk` | 0.055 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.06 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `high_step` | 0.045 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 35/35 | 0/35 | 0/35 | 0/35 |
| `high_step` | 0.045 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 23/25 | 2/25 | 0/25 | 0/25 |
| `high_step` | 0.05 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 35/35 | 0/35 | 0/35 | 0/35 |
| `high_step` | 0.05 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 23/25 | 2/25 | 0/25 | 0/25 |
| `high_step` | 0.055 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 35/35 | 0/35 | 0/35 | 0/35 |
| `high_step` | 0.055 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 21/25 | 4/25 | 0/25 | 0/25 |
| `high_step` | 0.0599 | within 10° | heading and line | 0.10 to 0.25 | 3 | 150 | 150/150 | 138/150 | 12/150 | 0/150 | 0/150 |
| `high_step` | 0.0599 | within 10° | heading only | 0.10 to 0.25 | 3 | 150 | 150/150 | 135/150 | 15/150 | 0/150 | 0/150 |
| `high_step` | 0.0599 | 10° to 20° | heading only | 0.10 to 0.25 | 3 | 150 | 150/150 | 116/150 | 34/150 | 0/150 | 0/150 |
| `high_step` | 0.06 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 34/35 | 1/35 | 0/35 | 0/35 |
| `high_step` | 0.06 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 19/25 | 6/25 | 0/25 | 0/25 |
| `high_step` | 0.062 | within 10° | heading and line | 0.10 to 0.25 | 3 | 299 | 298/299 | 259/299 | 40/299 | 0/299 | 0/299 |
| `high_step` | 0.062 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 301 | 296/301 | 198/301 | 103/301 | 0/301 | 0/301 |
| `high_step` | 0.12 | within 10° | heading and line | 0.10 to 0.25 | 3 | 25 | 0/25 | 0/25 | 25/25 | 1/25 | 0/25 |
| `high_step` | 0.12 | 10° to 20° | heading and line | 0.11 to 0.25 | 3 | 15 | 0/15 | 0/15 | 15/15 | 1/15 | 0/15 |
| `high_step` | 0.15 | within 10° | heading and line | 0.10 to 0.25 | 3 | 25 | 0/25 | 0/25 | 25/25 | 1/25 | 0/25 |
| `high_step` | 0.15 | 10° to 20° | heading and line | 0.11 to 0.25 | 3 | 15 | 0/15 | 0/15 | 15/15 | 0/15 | 0/15 |
| `crouch_walk` | 0.05 | up to 20° | heading and line | 0.08 to 0.20 | 1 | 300 | 0/300 | 0/300 | 300/300 | 0/300 | 0/300 |

**Ramped cable protector (25 degree sides, 0.10 m flat top)**

| mode | height (m) | approach | steering | commanded speed (m/s) | tries allowed | trials | crossed | crossed on the first try | `blocked` on the first try | stopped without a report | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `walk` | 0.025 | up to 20° | heading and line | 0.15 to 0.45 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| `walk` | 0.03 | up to 20° | heading and line | 0.15 to 0.45 | 3 | 660 | 660/660 | 660/660 | 0/660 | 0/660 | 0/660 |
| `walk` | 0.04 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 17/60 | 17/60 | 43/60 | 0/60 | 0/60 |
| `walk` | 0.05 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 15/60 | 15/60 | 45/60 | 0/60 | 0/60 |
| `walk` | 0.06 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 10/60 | 10/60 | 50/60 | 0/60 | 0/60 |
| `high_step` | 0.05 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 35/35 | 0/35 | 0/35 | 0/35 |
| `high_step` | 0.05 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 25/25 | 0/25 | 0/25 | 0/25 |
| `high_step` | 0.06 | within 10° | heading and line | 0.10 to 0.25 | 3 | 35 | 35/35 | 35/35 | 0/35 | 0/35 | 0/35 |
| `high_step` | 0.06 | 10° to 20° | heading and line | 0.10 to 0.25 | 3 | 25 | 25/25 | 25/25 | 0/25 | 0/25 | 0/25 |

**Low overhead: pipe cross-over (two pipes of 0.14 m diameter)**

| mode | clearance under it (m) | approach | steering | commanded speed (m/s) | tries allowed | trials | crossed | crossed on the first try | `blocked` on the first try | stopped without a report | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `walk` | 0.28 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.29 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 40 | 0/40 | 0/40 | 40/40 | 0/40 | 0/40 |
| `walk` | 0.3 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 40 | 0/40 | 0/40 | 40/40 | 0/40 | 0/40 |
| `walk` | 0.31 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 41/60 | 41/60 | 19/60 | 0/60 | 0/60 |
| `walk` | 0.33 | up to 20° | heading and line | 0.15 to 0.45 | 1 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| `high_step` | 0.28 | up to 20° | heading and line | 0.10 to 0.25 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `high_step` | 0.3 | up to 20° | heading and line | 0.10 to 0.25 | 1 | 40 | 0/40 | 0/40 | 40/40 | 0/40 | 0/40 |
| `high_step` | 0.31 | up to 20° | heading and line | 0.10 to 0.25 | 1 | 40 | 0/40 | 0/40 | 40/40 | 0/40 | 0/40 |
| `high_step` | 0.33 | up to 20° | heading and line | 0.10 to 0.25 | 1 | 60 | 25/60 | 25/60 | 35/60 | 0/60 | 0/60 |
| `crouch_walk` | 0.26 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 40 | 20/40 | 8/40 | 32/40 | 0/40 | 0/40 |
| `crouch_walk` | 0.27 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| `crouch_walk` | 0.28 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| `crouch_walk` | 0.283 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 600 | 600/600 | 600/600 | 0/600 | 0/600 | 0/600 |
| `crouch_walk`, body set to 0.18 m | 0.283 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| `crouch_walk`, body set to 0.185 m | 0.283 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| `crouch_walk`, body set to 0.19 m | 0.283 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 340 | 340/340 | 340/340 | 0/340 | 0/340 | 0/340 |
| `crouch_walk` | 0.29 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| `crouch_walk` | 0.31 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| `crouch_walk` | 0.33 | up to 20° | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |

**Low overhead: hatch and pipe rack 6 m long (cross beams every 1.5 m), a 2.5 cm conduit on the floor**

| mode | clearance under it (m) | approach | steering | commanded speed (m/s) | tries allowed | trials | crossed | crossed on the first try | `blocked` on the first try | stopped without a report | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `walk` | 0.3 | square | heading and line | 0.15 to 0.45 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `walk` | 0.31 | square | heading and line | 0.15 to 0.45 | 1 | 40 | 0/40 | 0/40 | 40/40 | 0/40 | 0/40 |
| `walk` | 0.32 | square | heading and line | 0.15 to 0.45 | 1 | 40 | 32/40 | 32/40 | 8/40 | 0/40 | 0/40 |
| `walk` | 0.33 | square | heading and line | 0.15 to 0.45 | 1 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| `high_step`, body set to 0.24 m | 0.3 | square | heading and line | 0.10 to 0.25 | 1 | 40 | 0/40 | 0/40 | 40/40 | 0/40 | 0/40 |
| `high_step` | 0.31 | square | heading and line | 0.10 to 0.25 | 1 | 40 | 0/40 | 0/40 | 40/40 | 0/40 | 0/40 |
| `high_step` | 0.33 | square | heading and line | 0.10 to 0.25 | 1 | 60 | 0/60 | 0/60 | 60/60 | 0/60 | 0/60 |
| `crouch_walk` | 0.27 | square | heading and line | 0.08 to 0.20 | 3 | 40 | 30/40 | 22/40 | 18/40 | 0/40 | 0/40 |
| `crouch_walk` | 0.28 | square | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| `crouch_walk` | 0.283 | square | heading and line | 0.08 to 0.20 | 3 | 600 | 600/600 | 582/600 | 18/600 | 0/600 | 0/600 |
| `crouch_walk`, body set to 0.18 m | 0.283 | square | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| `crouch_walk`, body set to 0.185 m | 0.283 | square | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| `crouch_walk`, body set to 0.19 m | 0.283 | square | heading and line | 0.08 to 0.20 | 3 | 340 | 340/340 | 331/340 | 9/340 | 0/340 | 0/340 |
| `crouch_walk` | 0.29 | square | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| `crouch_walk` | 0.3 | square | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| `crouch_walk` | 0.31 | square | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| `crouch_walk` | 0.33 | square | heading and line | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
<!-- END GENERATED obstacles -->

### 4.2 Ramps, side slopes and wet patches

<!-- BEGIN GENERATED slopes -->
| what | mode | slope (deg) | floor | approach | steering | commanded speed (m/s) | tries allowed | trials | crossed | crossed on the first try | `blocked` on the first try | stopped without a report | fell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ramp over a kerb 0.05 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 8 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.45 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| ramp over a kerb 0.05 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 10 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.45 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| ramp over a kerb 0.05 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 12 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.45 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| ramp over a kerb 0.05 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 14 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.45 | 3 | 40 | 40/40 | 38/40 | 2/40 | 0/40 | 0/40 |
| ramp over a kerb 0.12 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 8 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.45 | 3 | 40 | 40/40 | 38/40 | 2/40 | 0/40 | 0/40 |
| ramp over a kerb 0.12 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 10 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.45 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| ramp over a kerb 0.12 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 12 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.45 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| ramp over a kerb 0.12 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 14 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.45 | 3 | 40 | 40/40 | 38/40 | 2/40 | 0/40 | 0/40 |
| ramp over a kerb 0.18 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 8 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.45 | 3 | 40 | 36/40 | 32/40 | 8/40 | 0/40 | 0/40 |
| ramp over a kerb 0.18 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 10 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.45 | 3 | 40 | 40/40 | 38/40 | 2/40 | 0/40 | 0/40 |
| ramp over a kerb 0.18 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 12 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.45 | 3 | 40 | 40/40 | 39/40 | 1/40 | 0/40 | 0/40 |
| ramp over a kerb 0.18 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 14 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.30 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| ramp over a kerb 0.18 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 14 | dry | within 10° of the ramp's axis | heading and line | 0.15 to 0.45 | 3 | 640 | 640/640 | 597/640 | 43/640 | 0/640 | 0/640 |
| ramp over a kerb 0.18 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 14 | dry | within 10° of the ramp's axis | heading only | 0.15 to 0.45 | 3 | 200 | 198/200 | 178/200 | 22/200 | 0/200 | 0/200 |
| ramp over a kerb 0.18 m high (up, 0.20 m flat, down; 0.90 m wide, raised side edges) | `walk` | 14 | dry | within 10° of the ramp's axis | heading and line | 0.30 to 0.45 | 3 | 60 | 60/60 | 53/60 | 7/60 | 0/60 | 0/60 |
| ramp, walking down | `walk` | 10 | dry | - | - | 0.15 to 0.45 | 3 | 100 | 100/100 | 100/100 | 0/100 | 0/100 | 0/100 |
| ramp, walking down | `walk` | 15 | dry | - | - | 0.15 to 0.45 | 3 | 700 | 700/700 | 700/700 | 0/700 | 0/700 | 0/700 |
| ramp, walking down | `walk` | 20 | dry | - | - | 0.15 to 0.45 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| ramp, walking down | `high_step` | 15 | dry | - | - | 0.10 to 0.25 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| ramp, walking down | `crouch_walk` | 15 | dry | - | - | 0.08 to 0.20 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| ramp, walking up | `walk` | 10 | dry | - | - | 0.15 to 0.45 | 3 | 100 | 100/100 | 100/100 | 0/100 | 0/100 | 0/100 |
| ramp, walking up | `walk` | 15 | dry | - | - | 0.15 to 0.30 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| ramp, walking up | `walk` | 15 | dry | - | - | 0.15 to 0.45 | 3 | 100 | 99/100 | 82/100 | 18/100 | 0/100 | 0/100 |
| ramp, walking up | `walk` | 15 | friction 0.25 throughout | - | - | 0.15 to 0.45 | 1 | 30 | 0/30 | 0/30 | 30/30 | 0/30 | 0/30 |
| ramp, walking up | `walk` | 15 | dry | - | - | 0.30 to 0.45 | 3 | 60 | 58/60 | 40/60 | 20/60 | 0/60 | 0/60 |
| ramp, walking up | `walk` | 20 | dry | - | - | 0.15 to 0.45 | 3 | 60 | 26/60 | 26/60 | 34/60 | 0/60 | 0/60 |
| ramp, walking up | `walk` | 20 | friction 0.35 throughout | - | - | 0.15 to 0.45 | 1 | 30 | 0/30 | 0/30 | 30/30 | 0/30 | 0/30 |
| ramp, walking up | `high_step` | 15 | dry | - | - | 0.10 to 0.25 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| ramp, walking up | `crouch_walk` | 15 | dry | - | - | 0.08 to 0.20 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| side slope, walking across it (along the contour) | `walk` | 10 | dry | - | - | 0.15 to 0.45 | 3 | 100 | 100/100 | 100/100 | 0/100 | 0/100 | 0/100 |
| side slope, walking across it (along the contour) | `walk` | 15 | dry | - | - | 0.15 to 0.45 | 3 | 700 | 700/700 | 700/700 | 0/700 | 0/700 | 0/700 |
| side slope, walking across it (along the contour) | `walk` | 20 | dry | - | - | 0.15 to 0.45 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| side slope, walking across it (along the contour) | `high_step` | 15 | dry | - | - | 0.10 to 0.25 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| side slope, walking across it (along the contour) | `crouch_walk` | 15 | dry | - | - | 0.08 to 0.20 | 3 | 60 | 60/60 | 60/60 | 0/60 | 0/60 | 0/60 |
| wet patch 1.0 m across on level floor, friction 0.12 | `walk` | 0 | dry around the patch | - | - | 0.15 to 0.45 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| wet patch 1.0 m across on level floor, friction 0.15 | `walk` | 0 | dry around the patch | - | - | 0.15 to 0.45 | 1 | 30 | 30/30 | 30/30 | 0/30 | 0/30 | 0/30 |
| wet patch 1.0 m across on level floor, friction 0.20 | `walk` | 0 | dry around the patch | - | - | 0.15 to 0.45 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| wet patch 1.0 m across on level floor, friction 0.20 | `high_step` | 0 | dry around the patch | - | - | 0.10 to 0.25 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| wet patch 1.0 m across on level floor, friction 0.20 | `crouch_walk` | 0 | dry around the patch | - | - | 0.08 to 0.20 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |
| wet patch 1.0 m across on level floor, friction 0.25 | `walk` | 0 | dry around the patch | - | - | 0.15 to 0.45 | 1 | 30 | 30/30 | 30/30 | 0/30 | 0/30 | 0/30 |
| wet patch 1.0 m across on level floor, friction 0.30 | `walk` | 0 | dry around the patch | - | - | 0.15 to 0.45 | 3 | 40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 |

On a level floor of friction 0.20 throughout, `walk` at 0.15 to 0.45 m/s walked 15 s without a stop or a fall in 40/40 trials (`blocked` 0/40, fell 0/40; 1.9 to 6.3 m walked).
<!-- END GENERATED slopes -->

### 4.3 Narrow places

<!-- BEGIN GENERATED spaces -->
**Walking straight through a passage between two walls (one try)**

| mode | clear width (m) | done | `blocked` | fell | touched a wall |
|---|---|---|---|---|---|
| `walk` | 0.36 | 7/10 | 3/10 | 0/10 | 10/10 |
| `walk` | 0.40 | 10/10 | 0/10 | 0/10 | 2/10 |
| `walk` | 0.45 | 10/10 | 0/10 | 0/10 | 0/10 |
| `walk` | 0.50 | 10/10 | 0/10 | 0/10 | 0/10 |
| `high_step` | 0.36 | 9/10 | 1/10 | 0/10 | 10/10 |
| `high_step` | 0.40 | 10/10 | 0/10 | 0/10 | 7/10 |
| `high_step` | 0.45 | 10/10 | 0/10 | 0/10 | 1/10 |
| `high_step` | 0.50 | 10/10 | 0/10 | 0/10 | 0/10 |
| `crouch_walk` | 0.36 | 5/10 | 5/10 | 0/10 | 10/10 |
| `crouch_walk` | 0.40 | 10/10 | 0/10 | 0/10 | 2/10 |
| `crouch_walk` | 0.45 | 10/10 | 0/10 | 0/10 | 0/10 |
| `crouch_walk` | 0.50 | 10/10 | 0/10 | 0/10 | 0/10 |

**Turning 180° on the spot between two walls**

| mode | clear width (m) | tries allowed | done | `blocked` | neither within 20 s | fell | touched a wall | turned before the first `blocked` (deg): mean (range) |
|---|---|---|---|---|---|---|---|---|
| `walk` | 0.50 | 1 | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 | - |
| `walk` | 0.60 | 1 | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 | - |
| `walk` | 0.70 | 1 | 10/10 | 0/10 | 0/10 | 0/10 | 10/10 | - |
| `walk` | 0.80 | 1 | 10/10 | 0/10 | 0/10 | 0/10 | 5/10 | - |
| `walk` | 0.90 | 1 | 10/10 | 0/10 | 0/10 | 0/10 | 0/10 | - |
| `walk` | 0.50 | 3 | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 | 26 (25 to 27) |
| `walk` | 0.60 | 3 | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 | 51 (50 to 52) |
| `walk` | 0.70 | 3 | 10/10 | 0/10 | 0/10 | 0/10 | 10/10 | - |
| `high_step` | 0.50 | 1 | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 | - |
| `high_step` | 0.60 | 1 | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 | - |
| `high_step` | 0.70 | 1 | 10/10 | 0/10 | 0/10 | 0/10 | 10/10 | - |
| `high_step` | 0.80 | 1 | 10/10 | 0/10 | 0/10 | 0/10 | 1/10 | - |
| `high_step` | 0.90 | 1 | 10/10 | 0/10 | 0/10 | 0/10 | 0/10 | - |
| `high_step` | 0.50 | 3 | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 | 26 (25 to 28) |
| `high_step` | 0.60 | 3 | 5/10 | 5/10 | 0/10 | 0/10 | 10/10 | 58 (56 to 60) |
| `high_step` | 0.70 | 3 | 10/10 | 0/10 | 0/10 | 0/10 | 10/10 | - |
| `crouch_walk` | 0.50 | 1 | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 | - |
| `crouch_walk` | 0.60 | 1 | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 | - |
| `crouch_walk` | 0.70 | 1 | 10/10 | 0/10 | 0/10 | 0/10 | 10/10 | - |
| `crouch_walk` | 0.80 | 1 | 10/10 | 0/10 | 0/10 | 0/10 | 4/10 | - |
| `crouch_walk` | 0.90 | 1 | 10/10 | 0/10 | 0/10 | 0/10 | 0/10 | - |
| `crouch_walk` | 0.50 | 3 | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 | 24 (24 to 25) |
| `crouch_walk` | 0.60 | 3 | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 | 41 (34 to 46) |
| `crouch_walk` | 0.70 | 3 | 10/10 | 0/10 | 0/10 | 0/10 | 10/10 | - |
<!-- END GENERATED spaces -->

### 4.4 What the tables mean for a program

1. **Low thresholds.** Sills, kerbs, hoses, floor pipes and cable protectors up to 3 cm high are crossed by `walk` in
   every trial, nearly always on the first try. `crouch_walk` crossed the 2.5 cm sills and hoses it was tried on in
   every trial, but needed a second try at up to one sill in five.
2. **Steps above that.** `walk` and `crouch_walk` have a step guard: when a foot comes down clearly higher than its
   last foothold they stop and report `blocked` rather than climb. From 4 cm on, `walk` was stopped at every sill and
   kerb. Round hoses and floor pipes of 4 to 5 cm, and ramped cable protectors, it still got over in some trials:
   nothing to plan with. Take such elements in `high_step`.
3. **`high_step` and the angle of approach.** Within 10° of square, `high_step` crossed sills, kerbs and hoses from
   4.5 to 6 cm and floor pipes from 4.5 to 6.2 cm, the largest sizes it is rated for, in at least 34 of 35 trials
   within three tries, and at most one first try in seven ended in `blocked`. Turned 10° to 20°, up to half of the first tries at the top of the range ended in
   `blocked`, and up to one trial in twelve was not crossed within three tries. Sills, kerbs and pipes of 10 cm and
   more stop every mode: go round them or find a ramp.
4. **A `blocked` at a rated size is a delay, not a wall.** At a sill, hose, pipe or kerb of a size the mode is
   rated for (item 3 for `high_step`), retry up to twice in the same mode (send the walking command again when the
   hold of section 5 is over) before you change the plan. Every `blocked` costs the 5 s of that hold, so a program
   that finds out what lies ahead by walking into it pays for each find. At a kerb of 10 cm or more and at a ramp's
   side edge do not retry: item 8.
5. **Approach.** Come square to the element, in the middle of the way, and hold heading and line while crossing, as
   the trials' driver did: nothing was measured beyond the approach conditions stated at the top of this section.
6. **Low overheads.** `crouch_walk` passed every clearance of 0.28 m and more, also with its body at the top of its
   range. `walk` and `high_step` were stopped under every clearance of 0.30 m and less; `walk` passed 0.33 m in every
   trial, `high_step` did not pass the 0.33 m hatch at all.
7. **Ramps.** Ramps of 10° are taken at any speed. Up a 15° ramp, walk at 0.30 m/s or slower: faster, the step guard
   takes the foot of the ramp for a step in a third of first tries. Up a 20° ramp `walk` was stopped in more than half
   of the trials. Ramps over kerbs with slopes of 10° to 14° were taken in at least 198 of 200 trials within three
   tries, also with the heading held alone.
8. **Bund kerbs can trap the robot.** A kerb of 10 cm or more stops every mode. Pressed against such a kerb, or
   against a ramp's raised side edge, with all feet on the floor, the robot reports `blocked` every few seconds;
   stop and walk straight back, and it is free (60 of 60 trials). But a robot that keeps pushing or turning there
   can get a leg caught on the kerb or in the corner where a ramp's side edge meets the kerb, or come to lie across
   the kerb's top. Then no command got it free in our trials (0 of 85: backing off in every mode, sitting and
   standing, turning, sidestepping, body height and pitch, forward). Cross a bund only on its ramp, square to the
   ramp's foot and on its centre line; do not walk along a kerb or turn on the spot touching it; after a `blocked`
   at such a kerb or at a ramp's side edge, stop and back straight off before anything else.
9. **Narrow places.** Every walking mode passes straight through 0.40 m of clear width. To turn on the spot every
   mode needs 0.70 m, brushing the walls, and 0.90 m to turn without touching them; in 0.60 m no mode turned round at
   the first try.
10. **Wet patches** of 1 m were crossed in every trial in every mode.
11. **Two obstacles without a table of their own.** A cable trench cover is a steel plate strip 0.20 m deep with
    square edges, laid across a floor: for the gait it is a kerb of its height, and the kerb table applies. A pipe
    bridge is a short pipe rack across a room, pipes in layers on cross beams: it is a low overhead of the clearance
    under its beams, and the rack table applies. The training hall has a lane with each.

In none of the trials of this section did the robot fall.

## 5. Protective behaviours

<!-- BEGIN GENERATED protective -->
| `status` in `get_state()` | what happened | how it clears |
|---|---|---|
| `blocked` | the gait met something it is not crossing in this mode, or made no headway: it stopped, backed off a few centimetres and stands; for 5 s it takes no walking command (the hold, below) | after the 5 s: send a walking command (the same one as before is a retry); a change of mode alone does not move the robot |
| `protective_sit` | roll or pitch went beyond 35°: the body is lowered | after 5 s, once the tilt is below 25°, a walking command or `stand` makes it rise |
| `fallen` | roll or pitch went beyond 57° | does not clear: the robot cannot right itself |
| robot lost (event `robot_lost`) | tilted beyond 70° for 2 s | the mission ends |
<!-- END GENERATED protective -->

- **`blocked`.** The gait reports it when, in `walk` or `crouch_walk`, a foot lands on something higher than the mode
  is rated for (the step guard), when the swinging feet keep striking something, or when the robot makes too little
  headway (below). It then backs off a few centimetres, stands, and drops the velocity command:
  `get_state()["status"]` is `blocked`, and the `step` reply in which it happened carries an event `blocked`. Nothing
  moves until your program sends a walking command again, which it can do once the hold is over (the same command is
  a retry; the status returns to `ok` as the robot starts walking). A change of mode alone does not make it walk.

<!-- BEGIN GENERATED hold -->
- **The hold.** For 5 s of simulated time from every `blocked` report the robot takes no walking command. `walk` answers `{"ok": false, "held": true, "hold_until": t}`, with t the simulated time the hold ends, and changes nothing; `get_state()["status"]` reads `blocked` for the whole time, also after a change of mode. The back-off happens inside these 5 s. The walking command is dropped at the report: `set_mode` and `set_body` take effect at once, with the robot standing, and when the 5 s are over nothing moves until your program sends a walking command (the status then reads `ok` if the mode was changed, `blocked` if not, until the robot walks). The time passes only in `step`, as always; no event marks the end of a hold (`describe()["blocked_hold_s"]` is its length).
- **What a `blocked` costs.** Measured through this interface in the training hall: on level floor, under commands redrawn every 2 to 4 s, the gait reported no `blocked` in 60 minutes in each walking mode. A robot that walks into an element in the wrong mode, changes the mode at the report and walks on when it may is across 4.15 s later than it would be without the hold (the median of 326 crossings; 3.95 to 4.45 s): 11.93 to 15.95 s after the approach began in `high_step`, 20.83 s under a 0.29 m overhead in `crouch_walk`. A retry after the 5 s starts from a standstill close to the element. At an edge turned 20° at the top of the rated size (a 6 cm floor pipe) that is about as sure as the quick retry of section 4's trials: 127 of 130 crossed within three tries, against 128 of 130 with a retry 1 s after the report.
<!-- END GENERATED hold -->

<!-- BEGIN GENERATED headway -->
- **What is watched.** While the robot walks, the gait compares its headway with the command over the last 1.5 s (`high_step`: 2.5 s), one kind of command at a time: the forward or backward command when it has stayed above 0.05 m/s with one sign for that whole time; otherwise a sideways command above 0.05 m/s or a turning command above 0.10 rad/s that has stayed on one side for that whole time. If the robot made less than a quarter (`high_step`: 15 %) of the commanded distance or rotation, whether its feet held or slipped, it reports `blocked`.
- **When the watch starts, and what it does not see.** The watch starts 2 s (`high_step`: 3 s) after the robot starts walking, and starts again after every stop (a zero command), every `blocked` and every change of a command's sign. It does not watch commands below those limits, a command that changes sign or falls below its limit more often than that, a program that switches between turning on the spot and walking faster than that, or the sideways and turning parts of a command while a forward command is held. In those cases the robot can tread on the spot with status `ok`. A change of command on free floor (speeding up, reversing, another mode) is not answered with `blocked`.
<!-- END GENERATED headway -->

- **Check headway yourself as well.** The odometry counts the legs' steps and goes on counting when the feet slip or
  tread on the spot: caught at a kerb, the robot typically counted 1.3 to 3.7 m of odometry distance in 20 s while its
  body moved less than 5 cm and the odometry's position wandered about 0.2 m. Judge headway from the lidar against
  the drawing or an earlier sweep, or from the odometry's net displacement over 10 to 20 s, never from its distance;
  a tilt of 4° to 8° that stays while the floor is level is a further sign.

What the gait reported in trials in which the robot was held (the time runs from the start of the command):

<!-- BEGIN GENERATED stall -->
| what the robot was told to do | mode | floor friction | trials | `blocked` | stopped without a report | fell | time to the `blocked` (s): mean (range) | otherwise |
|---|---|---|---|---|---|---|---|---|
| commanded sideways into a wall beside the robot | `walk` | 0.90 | 10 | 10/10 | 0/10 | 0/10 | 2.5 (2.3 to 2.6) | - |
| commanded sideways into a wall beside the robot | `walk` | 0.20 | 10 | 10/10 | 0/10 | 0/10 | 2.4 (1.9 to 2.7) | - |
| commanded sideways into a wall beside the robot | `high_step` | 0.90 | 10 | 10/10 | 0/10 | 0/10 | 4.6 (3.7 to 5.2) | - |
| commanded sideways into a wall beside the robot | `high_step` | 0.20 | 10 | 10/10 | 0/10 | 0/10 | 4.2 (3.3 to 5.7) | - |
| commanded sideways into a wall beside the robot | `crouch_walk` | 0.90 | 10 | 10/10 | 0/10 | 0/10 | 2.9 (2.3 to 3.5) | - |
| commanded sideways into a wall beside the robot | `crouch_walk` | 0.20 | 10 | 10/10 | 0/10 | 0/10 | 3.1 (2.1 to 4.0) | - |
| facing into a room corner, commanded to turn in place | `walk` | 0.90 | 10 | 2/10 | 0/10 | 0/10 | 2.9 (2.7 to 3.1) | turned out of the corner 8/10 |
| facing into a room corner, commanded to turn in place | `walk` | 0.20 | 10 | 0/10 | 0/10 | 0/10 | - | turned out of the corner 10/10 |
| facing into a room corner, commanded to turn in place | `high_step` | 0.90 | 10 | 0/10 | 0/10 | 0/10 | - | turned out of the corner 10/10 |
| facing into a room corner, commanded to turn in place | `high_step` | 0.20 | 10 | 0/10 | 0/10 | 0/10 | - | turned out of the corner 10/10 |
| facing into a room corner, commanded to turn in place | `crouch_walk` | 0.90 | 10 | 3/10 | 0/10 | 0/10 | 2.4 (1.0 to 3.5) | turned out of the corner 7/10 |
| facing into a room corner, commanded to turn in place | `crouch_walk` | 0.20 | 10 | 0/10 | 0/10 | 0/10 | - | turned out of the corner 10/10 |
| walking sideways into a 4.95 cm sill | `walk` | 0.90 | 40 | 40/40 | 0/40 | 0/40 | 1.8 (1.1 to 3.2) | crossed 0/40 |
| walking sideways into a 4.95 cm sill | `crouch_walk` | 0.90 | 20 | 20/20 | 0/20 | 0/20 | 3.8 (2.1 to 8.0) | crossed 0/20 |

| what the robot was told to do | mode | floor friction | trials | longest stretch without headway or `blocked` (s): mean (range) | `blocked` per minute: mean (range) |
|---|---|---|---|---|---|
| facing into a room corner, a path follower alternating turning in place and walking towards a goal behind the wall (retrying after each report) | `walk` | 0.90 | 10 | 1.9 (1.5 to 2.6) | 20 (19 to 23) |
| facing into a room corner, a path follower alternating turning in place and walking towards a goal behind the wall (retrying after each report) | `high_step` | 0.90 | 10 | 3.0 (2.4 to 3.5) | 14 |
| facing into a room corner, a path follower alternating turning in place and walking towards a goal behind the wall (retrying after each report) | `crouch_walk` | 0.90 | 10 | 2.1 (1.3 to 2.5) | 25 (22 to 27) |
<!-- END GENERATED stall -->

- **`protective_sit`.** When roll or pitch passes the protective limit, the gait lowers the body and waits; an event
  `protective_sit` is in the `step` reply. See the table for how it clears.
- **`fallen` and lost.** The robot cannot right itself. A robot that stays overturned is lost and its mission ends
  (event `robot_lost`); what it had delivered until then stays delivered.

## 6. Body and head

- `set_body(height, pitch)`: body height within the mode's range; pitch only takes effect in `stand` (see the table of
  section 3 for its limits). Pitching the body adds to the head's tilt range; the table of section 3 shows how far
  the cameras rise and sink.
- `set_head(pan, tilt)`: the limits are in section 7.1. The reply gives the targets after cutting them to the limits.
  The head reaches a new target within about 0.3 s: step that long before you take a frame. `get_state()["head"]`
  gives the measured angles.
- The head lamp is a spot along the cameras' axis (its reach: section 2). The rooms that hold listed instruments are
  lit; the lamp is for unlit corners.

## 7. Sensors

### 7.1 Cameras

<!-- BEGIN GENERATED cameras -->
| name | what | image (px) | field of view, horizontal × vertical | focal length (px) | principal point (px) | lens on the head: metres from the tilt axis |
|---|---|---|---|---|---|---|
| `nav` | navigation camera: colour and depth (0.15 to 8 m) | 640 × 480 | 90° × 73.7° | 320.0 | (319.5, 239.5) | 0.0155 ahead, on the axis |
| `inspect` | inspection camera: colour only, telephoto | 1280 × 960 | 18° × 13.5° | 4040.8 | (639.5, 479.5) | 0.0155 ahead, 0.022 to the right |
| `thermal` | thermal camera: radiometric | 160 × 120 | 50° × 38.6° | 171.6 | (79.5, 59.5) | 0.0155 ahead, 0.022 to the left |

Depth accuracy of the navigation camera (typical error of one pixel, one standard deviation); about 1 % of the pixels of a frame carry no reading:

| distance (m) | noise added to a pixel (mm) | error measured on a flat wall (mm) |
|---|---|---|
| 0.5 | 2.5 | 3.8 |
| 1 | 4 | 4.7 |
| 2 | 10 | 10.5 |
| 4 | 34 | 34.9 |
| 8 | 130 | not measured |

Exposure is automatic: the gain applied to a colour frame (`gain` in the reply) is 1 where a room's lights reach (1.0 to 1.4 at the places instruments are read from) and at most 12 in the dark, where the image gets noisy.

The thermal camera's noise is about 0.1 K per pixel. A thermal pixel that sees no surface at all reads 15 °C.

Head: pan -1.75 to +1.75 rad (+ = to the left), tilt -1 to +0.8 rad (+ = nose down).
<!-- END GENERATED cameras -->

The three cameras sit side by side on the head and look the same way; the pose of each in the body frame comes with
every frame. Depth is the distance along the optical axis (not along the ray). Where the depth camera has no reading
(nearer or farther than its range, or a dropped pixel) the decoded depth is NaN.

**The thermal camera is radiometric.** Each pixel is an apparent temperature in °C: the temperature a perfect emitter
would need to send the same radiation. A real surface at temperature T with emissivity ε, in a room at T_room, shows
T_app with `T_app^4 = ε · T^4 + (1 − ε) · T_room^4` (temperatures in kelvin). So the surface temperature is
`T = ((T_app^4 − (1 − ε) · T_room^4) / ε)^(1/4)`. The emissivity of each surface type is in the plant's catalogue. The
room's temperature is the same throughout a building and can be read off the same thermal frame: walls, floor and
everything unheated show it to within a few tenths of a kelvin, whatever their emissivity. On a hot plate the
correction is several kelvin: a reading reported without it is too low.

A pixel shows the mean of what lies in its footprint. The rim of a hot face, and a face less than about three pixels
across, therefore read lower than the face itself: take the reading inside the face, away from its edges.

The thermal camera's field is wide. From a few metres a frame of a temperature point also holds whatever else is
warm near it, and a neighbouring heated box can be hotter than the tagged one: the hottest spot of a frame is not
necessarily the tagged component's face. Where that face lies relative to its tag plate is in the catalogue,
section 8.

### 7.2 Lidar

<!-- BEGIN GENERATED lidar -->
| item | value |
|---|---|
| beams per sweep | 720, one every 0.5°, beam k at k × 0.5° counter-clockwise from the body's x axis |
| range | 0.15 to 12 m; nothing nearer or farther gives a return (NaN) |
| range accuracy (one standard deviation) | about 1.1 cm at 1 m, 1.6 cm at 12 m |
| missing returns | about 1 % of the beams of a sweep, at random |
<!-- END GENERATED lidar -->

- The scan plane is horizontal in the body frame and moves with the body: it tilts when the body pitches or rolls, and
  its height above the floor is the body height plus the lidar's height in the body (section 2).
- **Blind sector.** The sensor head stands in the scan plane. With the head at pan 0, the beams within about 15° of
  straight ahead return nothing. Panning the head moves and narrows the shadow, but not away: at a pan of 1 rad
  about 11° each side of straight ahead stay hidden, at full pan about 7° each side. While the legs step, a raised
  knee can hide a few more degrees to one side for a sweep. What lies straight ahead is the depth camera's to see.
- **What the lidar registers.** Walls, door leaves, shield blocks, equipment, instruments and marker plates, where
  they cross the scan plane. It does not register what the gait modes are for: door sills, hoses, pipes on the floor,
  trench covers, cable protectors, kerbs and ramps lie below the plane, and the low overheads (the heads of low
  hatches, the beams and pipes of low racks, pipe bridges and pipe cross-overs) are passed through by the scan even
  where they hang into the plane; the next item says what little of them a sweep shows. Find and measure floor
  obstacles and low overheads with the depth camera. What stands low on a floor is no different: a pallet (0.14 m
  high) lies below the plane, and of a storage rack the plane meets the uprights and what stands on the shelves, not
  the lowest shelf. In a depth image the edge of a pallet or of a low shelf looks like a kerb; the robot gets neither
  onto nor under them in any mode.
<!-- BEGIN GENERATED lidar_overheads -->
- **Low overheads in a sweep.** The heads of the low hatches, the pipe chase's rack, pipe bridges and pipe cross-overs begin 0.28 to 0.30 m above the floor. The scan plane of a robot that walks upright is 0.31 to 0.34 m up (`walk`, `high_step`), so they hang into it, and the scan passes through them all the same: the head of a hatch, the cross beams, the supports under the pipes and the pipes return nothing. What a sweep does show of a pipe bridge is its hangers: four posts 4 cm square that stand against the room's two walls, one at each end of the bridge's two cross beams, and run from the beams up to 2.4 m. On some bridges it also shows the rims of a few pipe flanges. Of the chase's rack it shows the same kind of hangers at the chase's walls. A low hatch therefore shows as an opening in the wall, and a pipe bridge as open floor from wall to wall. In `crouch_walk` the plane is 0.25 to 0.26 m up, below all of them, and shows none of this. In the training hall the plain low racks (lanes `L27` to `L31`) are built with cross beams that a sweep does return where its plane reaches them; the hall's pipe bridge (lane `L32`) has no hangers, and nothing of it returns.
<!-- END GENERATED lidar_overheads -->
<!-- BEGIN GENERATED lidar_furniture -->
- **Furniture in a sweep.** Furniture and stored goods return like walls where they cross the scan plane: a bench, a cart or a pallet's load is a line of returns, and most such things are on no drawing (facility notes, section 2).
<!-- END GENERATED lidar_furniture -->
- A sweep is taken at one instant. Sweeps have no frame id and cannot be cited in a report.

### 7.3 Dosimeter

<!-- BEGIN GENERATED dosimeter -->
| item | value |
|---|---|
| sensitivity | 45 counts per second for 1 Gy/h |
| counting | the detector counts in steps of 0.1 s; `get_state()` returns the counts since the previous `get_state()` and the time they were collected over |
| dose rate from a count | counts ÷ (45 × seconds), in Gy/h; a count of N has a statistical spread of √N |
| accumulated dose | `dose_gy`: the dosimeter's own sum, from its counts |
| rating of the sensor head | 2 Gy accumulated: the mission ends when the dose received reaches it |
<!-- END GENERATED dosimeter -->

The dosimeter sits at the body centre. Its reading is a count, so a short reading is a noisy one: 100 counts are good
to about 10 %, 400 counts to about 5 %. Stand still and count longer where the number matters. Each `get_state()`
call empties the counter, so the counts of successive calls can simply be added up. An unlisted source in a building
of this kind measures between 20 and 60 Gy/h at 1 m; the dose rate falls with the square of the distance and is lower
behind walls, shield blocks and equipment. The distance is the one to the source itself, which sits 0.45 to 0.85 m
above the floor: the dosimeter of a robot that stands 1 m from it along the floor is a little farther away, and
reads between three quarters of the 1 m figure and nearly all of it.

### 7.4 IMU, joints, feet

`get_state()` gives roll and pitch (IMU), the body's turn rates, the twelve joint angles (legs in the order FL, FR,
HL, HR; each hip abduction, hip flexion, knee), the four foot contact switches, the body height as the firmware
estimates it from the legs, and the measured head angles.

### 7.5 Odometry

<!-- BEGIN GENERATED odometry -->
| item | value |
|---|---|
| start | at the drawing's datum pose (`mission()["datum"]`); the robot is set down within 5 cm and 2° of it, and the odometry does not know that offset |
| drift on level floor | about 2 % of the distance travelled; the heading drifts by up to about 1° per minute |
| on ramps and slopes | much more: typically 5 to 15 % of the distance walked on them, at worst about a third |
| when the feet slip or tread on the spot | the odometry goes on counting distance (section 5) |
<!-- END GENERATED odometry -->

The odometry is computed from the legs and the IMU. It does not notice when the feet slip or tread on the spot, and
it knows nothing of the building: correct it against what the drawing gets right (walls seen by the lidar), the grid
marker plates and the pad. Furniture that is not on the drawing can stand along a drawn wall at the lidar's height: a
line of returns is not always the drawn wall. Where a sheet of the drawing is missing there is nothing drawn to
correct it against: section 9.

## 8. The radio link

<!-- BEGIN GENERATED link -->
| item | value |
|---|---|
| base station (airlock) | reaches 12 m; its aerial is 1.9 m above the floor, at the place the link plan gives (`link`) |
| repeater, when it works | reaches 9 m; it hangs 2 m above the floor, at the place the link plan gives |
| condition | a free straight line from the top of the robot's back (the lidar) to the node's aerial, within the node's reach: walls, door leaves, shield blocks and equipment block it |
| checked | every 0.5 s of simulated time, and once more at the instant the mission ends |
<!-- END GENERATED link -->

- A report made while the link is up is delivered at once. A report made while it is down is queued on board and
  delivered when the link next comes up; `step` replies carry the events `link_up`, `link_down` and `report_sent`.
  Reports still queued when the mission ends are lost.
- `get_state()["link"]` tells whether the link was up at its last check and, when it is, a signal strength that falls
  with the distance to the node (it does not warn of a wall ahead).
- The reach figures are upper limits. What decides in a building is the line of sight: a node covers what its aerial
  can see. In every building the link is up everywhere within 1.0 m of the pad's centre and on nearly all of the
  airlock's floor; through the open inner door the base station also sees a strip of the corridor in front of the
  airlock, a few metres long. A repeater sees the corridor stretch it hangs in and a little of the rooms whose open
  doors face it.

## 9. Localisation aids

<!-- BEGIN GENERATED markers -->
| plate | marker | plate | what the drawing lists |
|---|---|---|---|
| grid marker plate | AprilTag 36h11 (`DICT_APRILTAG_36h11`), 110 mm side | 349.8 × 145.2 mm | `grid_markers[].marker_id`, with the surveyed position, height and facing of the plate's centre; the marker's centre lies 102 mm to the left of it, as seen facing the plate |
<!-- END GENERATED markers -->

- **The pad** is a painted square in the airlock, drawn in `mission()["equipment"]`; its centre is the site frame's
  origin and its pose is `mission()["datum"]`.
- **Grid marker plates** hang on walls at surveyed positions on the building's column grid. The drawing lists each
  plate's marker id, the position and height of the plate's centre and the direction it faces. The marker itself
  stands 102 mm to the left of that centre, as seen facing the plate. A marker's four corners in an image give the
  camera's pose relative to the marker; seen near head-on, a single marker fits two poses almost equally well, as a
  tag plate does (below): check the one you take against the direction the drawing lists for the plate
  (`facing_deg`). The markers' dictionary is in OpenCV as `cv2.aruco.DICT_APRILTAG_36h11`. No plate hangs where a
  sheet of the drawing is missing.
- **Tag plates** hang next to every tagged instrument, parallel to its face; where exactly is in the catalogue,
  section 8. A tag plate carries the instrument's tag in the plant lettering and no marker: the catalogue, section
  6, gives the plate, its border line and every glyph in numbers. The plates are signal yellow, a colour the plant
  keeps for them above the floor (facility notes, section 5): the navigation camera shows a plate as a small yellow
  patch, and its depth image tells how large the patch really is. Under the plant's lighting a plate shows darker
  in a frame than the catalogue's paint colour, with the same hue.
- **Room plates** hang beside the main door of every room but the corridors, 1.6 m up, with the room's number and
  name in the same lettering on a white plate (facility notes, section 2). The annex has none.

<!-- BEGIN GENERATED plate_view -->
|  | inspection camera | navigation camera |
|---|---|---|
| one frame covers | 18° × 13.5°: 0.63 × 0.48 m at 2 m, 1.58 × 1.19 m at 5 m | 90° × 73.7°: 4.00 × 3.00 m at 2 m |
| a tag plate (190.8 × 79.2 mm) seen head-on from d metres | 771 / d × 320 / d px | 61 / d × 25 / d px (31 × 13 px at 2 m, 12 × 5 px at 5 m) |
| capitals of a tag (25 mm high) | 101.0 / d px | 8.0 / d px |
| character pitch of a tag (20 mm), the plate seen a degrees off its normal | 80.8 × cos a / d px | 6.4 × cos a / d px |
| capitals of a room number (45 mm) | 181.8 / d px | 14.4 / d px |

The lettering of a tag plate is legible in an inspection frame when its capitals are at least 14 px high and its characters at least 8 px apart, up to 60° off the plate's normal: out to 7.2 m head-on, 7.1 m at 45° and 5.0 m at 60°. Smaller or more oblique, neighbouring glyphs can no longer be told apart. The navigation camera does not resolve the lettering (a capital is 4 px high at 2 m): it shows where a plate is, not what it says.
<!-- END GENERATED plate_view -->

- **A plate's pose.** A plate is flat and its size is known, so the four corners of its border line in a frame give
  its position and the direction it faces relative to the camera, and with the offsets of the catalogue's section 8
  the instrument's reference point. Two poses fit four corners of a small, distant plate almost equally well when
  it is seen nearly head-on: the true one and its mirror image about the line of sight. What tells them apart is
  the direction the plate is known to face. The round sheet's facing is the facing of the recorded place (facility
  notes, section 3): it settles the pose of a plate that hangs there, at the recorded position, and of no other. A
  listed instrument that was moved or relocated hangs somewhere else and faces as its wall there does, and for one
  without a recorded location no facing is recorded: for their plates, and for any plate whose place is not a
  recorded one, take the direction from the wall behind the plate, as the lidar or the depth image shows it.

<!-- BEGIN GENERATED plate_pose -->
Measured with the inspection camera on plates near enough to be read: up to 3 m the centre of a plate comes out to 2.2 mm at the median and 10 mm for 95 % of the plates; it gets worse with distance and with the angle off the plate's normal, to 5.8 cm at the median and 8.3 cm for 95 % at 5 to 6 m and 55° to 65°. The direction the plate faces comes out to 1.3° at the median and 4.6° for 95 % of the plates up to 6 m, worst head-on. With the facing known to 2.5°, as the round sheet records it for an instrument that hangs where it is recorded, the mirror pose was taken for none of 8049 plates. With a facing 5° off, for 7 of them, all beyond 5 m and within 20° of head-on. With a facing 10° off, for 101, from 2 m out. With no facing at all, for 169, from 4 m out, within 40° of head-on.
<!-- END GENERATED plate_pose -->

- **Where a sheet of the drawing is missing**, nothing is drawn to match and no grid marker plate hangs. There the
  robot knows its position from its odometry and from what its own sensors have seen: a wall it has scanned can be
  matched again. The building's outer walls stand on `building.outline` on every sheet, their inner face 0.125 m
  inside it, and the walls of the neighbouring sheets are drawn. The rooms of a missing sheet have their room plates
  as the others have (above): the number is beside the main door. An instrument that is still at its recorded place
  is where the round sheet says, to the sheet's rounding.

## 10. What the team accepts as a record photo

A reading is only as good as the photo that proves it: with every instrument report the team wants the frame the
value was read from. The inspection team accepts a frame for an instrument when all of this holds:

<!-- BEGIN GENERATED evidence -->
| instrument (round-sheet `type`) | frames that count | reference point | feature, and its smallest size in the cited frame | largest angle off the face | must lie in the frame | camera to the reference point | the size condition holds out to | full credit within |
|---|---|---|---|---|---|---|---|---|
| dial gauge (`gauge`) | inspection camera | the centre of the dial | dial diameter (0.9 × the case diameter) ≥ 77 px | 45° | at least 90 % of the dial's rim | 0.8 to 5.0 m | 63 mm case: 2.97 m; 100 mm case: 4.72 m; 160 mm case: 5.0 m (the distance limit) | 2 % of the scale's span |
| level gauge (`level`) | inspection camera | the middle of the glass tube and the scale plate taken together | length of the glass window (round sheet: `window_m`) ≥ 400 px | 30° | the whole scale plate and the glass tube | 2.5 to 5.0 m | 0.40 m window: 4.04 m; 0.44 m window: 4.44 m | 5 percentage points |
| quarter-turn valve (`valve_lever`) | inspection camera | the lever's pivot on the valve stem | lever length (0.18 m) ≥ 80 px | 45° | the whole lever and the whole tag plate | 1.5 to 5.0 m | 5.0 m (the distance limit) | 5° |
| rising-stem valve (`valve_handwheel`) | inspection camera | the centre of the indicator plate | indicator stroke, 0 % to 100 % (0.10 m) ≥ 115 px | 45° | the whole indicator plate and the pointer | 1.5 to 5.0 m | 3.51 m | 5 percentage points |
| lamp panel (`lamps`) | inspection camera | the centre of the panel's front | lens diameter (23 mm) ≥ 9 px | 55° | the whole front of the panel and every lens | 1.5 to 5.0 m | 5.0 m (the distance limit) | every lamp right |
| temperature point (`thermal`) | thermal camera | the centre of the hot face | smaller side of the hot face's outline in the image × √(share of the face in view) ≥ 6 px | 60° | the whole hot face | 0.4 to 5.0 m | 5.0 m (the distance limit) (face 0.34 × 0.28 m, fully in view) | 3 °C |

- **Reference point**: the point of the instrument that distance, angle and size are measured to. Where it lies relative to the position the work order gives, and to the tag plate, is in the catalogue, section 8 (a rising-stem valve's is 0.42 m above the work order's position).
- **Size in the cited frame** = the frame's focal length in pixels (`K[0][0]`: 4040.8 for the inspection camera, 171.6 for the thermal camera) × the feature's real size ÷ the distance from the camera to the reference point. For a temperature point it is measured in the image instead: the smaller side of the rectangle around the hot face plate's corners (the part inside the image), times the square root of the share of the face that is in view.
- **Angle off the face**: between the face's outward normal and the line from the reference point to the camera (sideways and vertical parts together).
- **In the frame**: at least 1 px inside the image border. For a dial, 24 points on its rim are tested.
- **In view**: at least 90 % of the feature must be seen from the camera with nothing in front of it; the robot's own body and legs count as in front, and so does a pipe, an upright or a neighbouring object (temperature point: at least 5 % of the hot face, with the size rule above). The share is counted on points spread over these parts: dial gauge: the dial; level gauge: the scale plate and the front of the glass tube over the window; quarter-turn valve: the lever and the tag plate; rising-stem valve: the indicator plate and the pointer; lamp panel: the lenses and the panel's front; temperature point: the hot face.
- **Camera to the reference point**: the straight distance between them; at most 5.0 m for every type.
<!-- END GENERATED evidence -->

Cite up to three frames with a report (`frame_ids`); one accepted frame is enough. A frame must have been taken in
the same mission, before the report is made. Frames of the navigation camera and depth-only frames are never accepted.

## 11. Time and budgets

Time passes only while your program calls `step`: a sensor call returns what the sensor sees at the current instant.

<!-- BEGIN GENERATED budgets -->
| budget | per mission |
|---|---|
| simulated time | 1350 s (training hall: 900 s) |
| program time: the wall clock minus the robot cell's own work inside `step` and the four sensor calls | 1200 s |
| silence: from the start, or from a reply, to your next request | 180 s |
| CPU time of your program, all threads together | 4800 s |
| navigation camera frames with colour (`get_camera("nav")`, with or without depth) | 300 |
| inspection camera frames (`get_camera("inspect")`) | 240 |
| depth-only frames (`get_depth()`) | 1500 |
| thermal frames (`get_thermal()`) | 150 |
| lidar sweeps (`get_lidar()`) | 4000 |
| reports | instrument 40, survey mark 6, source 3; at most 3 cited frames per report |
<!-- END GENERATED budgets -->

A sensor call beyond its budget is refused with the error `budget` and nothing is taken. What the calls cost in wall
clock:

<!-- BEGIN GENERATED costs -->
| what | wall clock with one mission running | with two missions at a time |
|---|---|---|
| walking or standing (any `step`) | about 0.44 s per simulated second | 0.45 to 0.46 s per simulated second each |
| an inspection frame | 0.46 to 0.55 s | 0.43 to 0.56 s |
| a navigation frame with depth | 0.29 to 0.47 s | 0.28 to 0.47 s |
| a navigation frame, colour only | 0.16 to 0.18 s | - |
| a depth-only frame | 0.09 to 0.12 s | - |
| a thermal frame | 0.05 to 0.11 s | - |
| a lidar sweep | under 0.03 s | - |
| the first frame of each camera in a mission | 0.1 to 1.5 s | - |
| a mission that uses all 1350 s of simulated time | about 10 min for the simulation alone | - |
| a training-hall mission that uses all 900 s | about 7 min for the simulation alone | - |

Measured in this image through `~/training/run_mission.py`. The sandbox has 8 CPUs. `nproc` prints 8 in your shell and 2 inside a practice or scored run of your program, where the numerical libraries are held to 2 threads (`nproc` reports that setting); Python's `os.cpu_count()` prints the cores of the host machine in both places, more than the sandbox can use. Your program and the robot cell share the sandbox's 16 GB of memory, in a practice run as in a scored mission; should your program's processes use it up in a scored mission, the kernel stops one of them.
<!-- END GENERATED costs -->

## 12. The practice ground

The team's practice ground runs the same robot, firmware, sensors and rules as a real mission:

<!-- BEGIN GENERATED practice -->
| number | name | what | simulated time limit |
|---|---|---|---|
| 1 | `hall` | the training hall | 900 s |
| 2 | `t1` | a practice building | 1350 s |
| 3 | `t2` | a practice building | 1350 s |
| 4 | `t3` | a practice building | 1350 s |

Allowance: 30 practice missions in your whole session, at most 2 running at the same time.
<!-- END GENERATED practice -->

The training hall is the place to measure this robot yourself: every lane of `training/hall/README.md` holds one
obstacle of known size, the instrument wall carries every instrument type at three heights with known values, a
plate board and two posts carry tag plates of known text at known places, and the hall's debrief adds the robot's
true pose twice a second, against which the odometry can be calibrated. How to start a practice mission and what its
debrief holds is in `robot_api.md`, section 10.
