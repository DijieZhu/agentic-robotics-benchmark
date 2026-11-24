# Calibrate a robot's camera, LiDAR and IMU from its driving logs, without a reference

A wheeled warehouse robot (a Clearpath Husky with a sensor mast) carries a spinning LiDAR, an RGB camera and an IMU. After transport and a repair, the camera and IMU brackets were refitted and the camera's lens was re-focused. Nothing about the camera is calibrated any more: not its mount (the bracket may even sit on another face of the mast), not its focal length or distortion, not its rolling shutter's readout time. The IMU board's orientation is not documented. The camera and the LiDAR each run on their own clock, offset from the robot's clock by up to half a second and drifting. Nobody can build a calibration target room for this robot. What you do have are ordinary driving logs: several sequences per robot, recorded in the warehouse it works in while forklifts and carts drive around. A few stretches of these logs are corrupted by sensor or driver faults, and on some robots the camera bracket gets knocked once during the session.

Write a **reusable calibration program**. Given one robot's logs, it must recover the camera's intrinsics, distortion and readout time, the camera-LiDAR and IMU-LiDAR extrinsics, both sensor clocks and the IMU biases, and it must report the faulty stretches and a bracket knock if there was one. After you finish, your program is run on six robots it has never seen. Its output is checked against their true parameters on drives of each robot that your program never sees.

**There is no score feedback in your session**: you have three practice robots' logs, but not their true parameters. Like an engineer at a real robot, you judge your calibration by its own consistency (LiDAR points landing on the right things in the images, integrated IMU motion agreeing with the LiDAR's, the sequences agreeing with each other).

You have 4 hours (see "Session time").

## What you submit

A program in `/home/user/submission/` that runs as

```
bash run.sh --input <rig_dir> --output <out_dir>
```

and writes two files into `<out_dir>` (which exists and is empty): `calibration.yaml` and `fault_report.json`, described below. Put everything the program needs beside `run.sh`. Nothing outside `/home/user/submission/` is kept, and there is no submit step: whatever is there when your session ends is graded.

## The rig folder (input)

Three practice rigs in exactly the format of the hidden ones are in `/opt/rigcal/practice/rig_a`, `rig_b` and `rig_c` (read-only, about 100 MB each). They come without truth.

```
<rig_dir>/
├── manifest.json        sensors, the camera's NOMINAL data, LiDAR beam angles, T_body_lidar, the sequence names, whether there is wheel odometry
├── rough_prior.yaml     CAD values of T_lidar_camera and T_lidar_imu, with their error bounds
└── seq_00/, seq_01/, seq_02/
    ├── camera/timestamps.csv    frame,stamp_s       (frames/000000.jpg ... : 768x512 RGB JPEG, 10 Hz on the camera's clock)
    ├── camera/frames/*.jpg
    ├── lidar/timestamps.csv     scan,stamp_s        (scans/000000.npz ... : 10 Hz on the LiDAR's clock)
    ├── lidar/scans/*.npz        range_mm (16x1800 uint16, 0 = no return), reflectivity (16x1800 uint8)
    ├── imu.csv                  stamp_s,gx,gy,gz,ax,ay,az    (200 Hz, rad/s and m/s^2)
    └── wheel_odom.csv           stamp_s,v_mps,yaw_rate_rps   (50 Hz; ONLY on rigs whose manifest says wheel_odom.present)
```

- **Sequences.** Each is 24 to 28 s long. Each starts and ends with the robot standing still for about 2 to 3 s, and some also stop in the middle. They are one to two minutes apart within one recording session of about twelve minutes; the session also holds a drive before the first given sequence and one after the last, which you do not get (the held-out drives). `manifest.json` lists the given sequences with their approximate start times. The robot drives on a flat floor; its suspension rolls and pitches the body by a few degrees when it moves, and the body vibrates while it drives.
- **The site.** A warehouse hall (60 x 40 m) with painted floor lanes, doors, poles and partitions, and goods standing around the robot's working area: crates, cartons, pallets, tool cabinets, carts, racks, trucks. Four to seven vehicles (forklifts, reach trucks, pallet trucks, carts, hand trucks) drive straight runs back and forth through the area during every sequence, at 0.5 to 2 m/s. All rigs work in this hall; the working area and what stands and drives in it differ from rig to rig.
- **LiDAR** (a Velodyne VLP-16). Ring `i` points at elevation `manifest.lidar.elevation_deg[i]` (16 rings, -15 to +15 degrees, 2 degrees apart). Column `j` points at azimuth `360*j/1800` degrees, counter-clockwise from +x. A return is `range * (cos(el) cos(az), cos(el) sin(az), sin(el))` in the LiDAR frame **at the moment the column was taken**: a sweep takes 0.1 s, its stamp is the time of column 0, and column `j` is taken `j/1800 * 0.1` s later (on the LiDAR's clock). The driver does **not** motion-compensate, so a sweep taken while the robot moves is skewed. Ranges have 2 cm of noise, rising to 3 cm at 60 m. Returns nearer than 0.5 m are dropped (the mast's legs blank two sectors), as are weak, grazing and a few random returns; at depth edges some returns lie between the two surfaces. Reflectivity is the surface material's near-infrared reflectance, weaker at grazing incidence. It is a property of the material, not of its visible colour or print.
- **Camera** (a FLIR Blackfly S BFS-U3-63S4C, rolling shutter, 4x4 binned, 6 mm lens). A pinhole with Brown-Conrady distortion `[k1, k2, p1, p2, k3]` (OpenCV's order). **Its intrinsics are not calibrated**: the manifest gives the nominal focal length (625 px) and principal point (384, 256); the true focal length is within 6 % of nominal (fx and fy within 0.3 % of each other), the principal point within 20 px, `k1` in -0.32..-0.08, `k2` in -0.02..0.14, `p1` and `p2` within 0.0015, `k3` within 0.02. **Rolling shutter**: the rows are read top to bottom; reading a frame takes between 0.010 and 0.030 s, constant for a rig and not calibrated; a frame's stamp is the middle of its readout, so row `v` is taken `(v - 255.5) / 512 * readout` from the stamp. Exposure 2 ms per row. Frames carry vignetting, sensor noise and JPEG compression. The robot itself is not drawn in the camera's view.
- **IMU** (a Bosch BMI088). The gyro measures angular velocity plus a bias plus noise (about 0.0035 rad/s per sample). The accelerometer measures specific force plus a bias plus noise (about 0.024 m/s^2 per sample); at rest it reads +9.80665 m/s^2 along "up". Both are in the IMU's own frame. Each axis has a scale error of up to 0.1 % and the axes are coupled by up to 0.1 %; the biases wander slowly (far less than the tolerances below over a session). The IMU and the wheel odometry are stamped by the robot's reference clock.
- **Wheel odometry** (not on every rig: `manifest.wheel_odom.present`). The body's forward speed and yaw rate from the wheel encoders. The speed has up to 3 % of scale error and reads a few percent high while the wheels slip (accelerating, turning). The robot is skid-steered: the yaw rate reads only 0.80 to 1.00 of the true rate, a factor that is constant for a rig.

## Frames and conventions

- World: z up, gravity (0, 0, -9.80665) m/s^2. Body: x forward, y left, z up, origin on the floor. LiDAR: x forward, y left, z up. `T_body_lidar` in the manifest is exact.
- Camera: the OpenCV optical frame (x right, y down, z forward).
- `T_a_b` is the 4x4 pose of frame `b` in frame `a`: `p_a = R_a_b p_b + t_a_b`, with the last row `[0, 0, 0, 1]`. Metres and radians throughout.
- **Clocks.** At reference time `t` (seconds since the start of the session, the clock in `imu.csv`) a sensor's clock with offset `o` (s) and drift `d` (ppm) reads `s = t + o + d * 1e-6 * t`. The camera and the LiDAR each have their own `o` and `d`, and each runs free on its own clock: its frames are 0.1 s apart **on that clock** (plus a trigger jitter of 0.3 ms for the camera and 0.05 ms for the LiDAR, and 0.2 ms of stamping jitter), whatever the drift. A sensor's log of a sequence starts a few tenths of a second to a second after the sequence starts and ends as long before it ends, differently for each sensor and sequence. On every rig, `|o|` is at most 0.5 s for each of the two sensors, and `|d|` is at most 200 ppm for the camera and 100 ppm for the LiDAR.
- **The camera's bracket.** `rough_prior.yaml` gives the CAD mount. The bracket may have been refitted on another face of the mast: the true mount is the CAD mount, or the CAD mount turned by a quarter turn (either way) or a half turn about the LiDAR's z axis, and then off by up to 10 degrees and 8 cm.
- **The IMU.** Its position prior is good to 4 cm. Its orientation is not documented (the rotation in `rough_prior.yaml` is a placeholder): the board sits in one of the 24 axis-aligned orientations, off by up to 5 degrees. Gyro biases are up to 0.015 rad/s per axis and accelerometer biases up to 0.15 m/s^2 per axis.
- **A knock.** On some rigs the camera bracket is knocked once, during a given sequence while the robot drives: from one frame to the next the camera's mount changes by 1.5 to 4 degrees and a few millimetres and stays there. The held-out drive before the given sequences has the mount from before the knock, the one after has the mount from after it.

## calibration.yaml (output)

```yaml
T_lidar_camera: [[r00, r01, r02, tx], [r10, r11, r12, ty], [r20, r21, r22, tz], [0, 0, 0, 1]]   # camera in the LiDAR frame, at the END of the session
T_lidar_imu:    [[...], [...], [...], [0, 0, 0, 1]]                                             # IMU in the LiDAR frame
camera:
  fx: 625.0
  fy: 625.0
  cx: 384.0
  cy: 256.0
  distortion: [k1, k2, p1, p2, k3]
  readout_s: 0.02                    # rolling shutter: first row to last row
time:
  camera: {offset_s: 0.0, drift_ppm: 0.0}
  lidar:  {offset_s: 0.0, drift_ppm: 0.0}
imu:
  gyro_bias_rps:   [bx, by, bz]     # in the IMU frame; measured = true + bias (the mean bias over the given sequences)
  accel_bias_mps2: [bx, by, bz]
changes:                             # optional; only if the bracket was knocked (at most one entry)
  - type: camera_bracket_shift
    T_lidar_camera_before: [[...], [...], [...], [0, 0, 0, 1]]   # the camera's mount before the knock
```

Every key but `changes` is required. Each rotation block must be orthonormal with determinant +1, to within 1e-3; it is re-orthonormalised before use. `~/tools/prior_baseline.py` writes a valid example from the rough prior and the nominal camera, and `~/tools/check_output.py <out_dir>` says whether two files are well-formed (it knows nothing about their values).

## fault_report.json (output)

```json
{"faults": [{"sequence": "seq_01", "sensor": "camera", "type": "camera_frozen", "start_s": 123.45, "end_s": 125.10}]}
```

Each rig's logs contain one to three fault intervals, at most one of each type, and on some rigs a knock. Each interval lasts 1.5 to 3.5 s while the robot is moving, and two can fall in the same sequence. The types and their sensors:

| type | sensor | what happens |
|---|---|---|
| `camera_frozen` | camera | the driver re-sends the last good image (byte-identical files) with fresh stamps |
| `camera_time_jump` | camera | the frames' stamps are shifted by a constant 35 to 80 ms (either sign) |
| `lidar_degraded` | lidar | a large share of returns are replaced by spurious short, dim returns |
| `imu_axis_swap` | imu | two axes of both the gyro and the accelerometer are exchanged |
| `odom_glitch` | wheel_odom | the forward speed reads about zero or jumps by a large offset (only on rigs with wheel odometry) |
| `camera_bracket_shift` | camera | the knock: `start_s` = `end_s` = the stamp of the first frame taken with the new mount; a report within 0.5 s of it matches |

`start_s` and `end_s` are stamps from the affected sensor's own log: the camera's or LiDAR's `timestamps.csv`, or `stamp_s` in `imu.csv` or `wheel_odom.csv`. They are the first and last affected samples. At most 50 entries.

## How it is graded

The six hidden rigs are a fixed set (some without wheel odometry, some with a knock). Your program is run once per rig, **all six at the same time** on an 8-CPU machine with 32 GB of memory. Each run has the working directory `/home/user/submission/`, runs as your user at the lowest scheduling priority, and has **no network** (AF_UNIX sockets only). Its limits:

- **1800 seconds of its own processor time**: user plus system time of all its processes together. Waiting, and whatever else the machine is doing, does not count. A run that uses several cores uses its allowance that much faster.
- a wall-clock stop at 7200 s, which a run inside its processor-time allowance does not reach on a machine that is not overloaded;
- a **6144 MB address-space limit** per process.

A run that reaches a limit is killed, and the rig is scored on whatever files exist by then. NumPy, SciPy and OpenCV are pinned to one thread and to one set of kernels (`OMP_NUM_THREADS=1`, `OPENBLAS_CORETYPE=Haswell` and related settings, defaults of the image's Python; do not change them), so plan on one core per run, and a program computes the same on every machine.

Each rig earns a credit in [0, 1]:

```
credit = 0.10 * faults + 0.90 * (c_projection * c_motion * c_time) ^ (1/3)
```

The three factors are measured on the two held-out drives (one before the given sequences, one after), and each needs several parameters right together; their geometric mean is zero when one of them is zero, so a part left at the prior is not made up for by the others. Each error `e` becomes a credit `c(e) = clamp(log(zero/e) / log(zero/full), 0, 1)`: 1 at or below `full`, 0 at or above `zero`, log-linear in between.

| factor | measured as | full | zero |
|---|---|---|---|
| `c_projection` | `projection_px`: median distance, in the frame's own pixels, between where held-out LiDAR points land in held-out camera frames taken while the robot moves, through your intrinsics, distortion, readout time, `T_lidar_camera` and both clocks, and where they really appear; the worse of the two held-out drives | 2 px | 4 px |
| `c_motion` | mean of `c(rpe_rot_deg)` and `c(rpe_trans_m)`: the mean error of the LiDAR's motion over 1 s windows of the held-out drives, predicted by integrating the held-out IMU samples with your `T_lidar_imu` and biases | 0.2 deg, 0.02 m | 0.4 deg, 0.06 m |
| `c_time` | `time_rms_s`: RMS error of the reference times your clock models give for the held-out camera and LiDAR stamps (the worse of the two sensors) | 3 ms | 9 ms |
| `faults` | F1 of your reported intervals; a report matches a true fault with the same sequence and sensor and interval IoU >= 0.3 (each interval padded by 0.05 s; a knock: within 0.5 s), counting 1 with the right type and 0.5 with a wrong one | | |

More precisely:

- **Projection.** Each LiDAR point of a held-out sweep is placed in the world with the LiDAR's true pose at the reference time your LiDAR clock gives for its column. It is then projected into the frame: the camera's pose is the LiDAR's true pose composed with your `T_lidar_camera`, at the time your camera clock gives for the frame's stamp plus the row's delay under your readout time (the row the point falls in), through your focal lengths, principal point and distortion. The result is compared with the pixel under the true parameters. Points behind your camera count as 1000 px. On the held-out drive BEFORE the given sequences the mount is `changes[0].T_lidar_camera_before` if you report a change, else `T_lidar_camera`; on a rig with a knock the true mount there is the one from before it.
- **Motion.** Each window starts from the true LiDAR pose and velocity. Its IMU pose is the LiDAR pose composed with your `T_lidar_imu`. It integrates your bias-corrected gyro and accelerometer samples at 200 Hz, then maps the result back to a LiDAR pose through your `T_lidar_imu`. (The IMU's small scale errors are part of the data; the true parameters leave about 0.05 deg and 0.01 m.)
- **Held-out drives.** One is recorded before the first given sequence and one after the last, so the clock models are used outside the time span of your data on both sides.
- **Missing or malformed files.** A missing or unparsable `calibration.yaml` earns 0 on the three factors for that rig. A missing or malformed `fault_report.json` earns 0 on the fault term.

The graded number is the mean credit over the six rigs. The true parameters score 1.0 on every rig: each full point is a small multiple of what a careful calibration of these logs can reach.

## Tools

- `~/tools/rig_io.py` has readers for the format, `~/tools/overlay.py` draws a sweep over a camera frame for a given calibration, `~/tools/prior_baseline.py` writes the output format from the prior and the nominal camera, and `~/tools/check_output.py <out_dir>` checks that an output folder's two files are well-formed.
- `bash ~/session_time.sh` prints the minutes of your session used and left.
- There is no practice grader and there are no true parameters in your sandbox.

## Environment

Python 3.12 (`python3`, `/opt/venv`) with numpy 2.5.4, scipy 1.18.1, opencv-python-headless 5.0.0.93, PyYAML 6.0.2 and matplotlib 3.11.2. Your program is graded on exactly this interpreter and these packages, and nothing else can be installed. Your sandbox has 8 CPUs, 16 GB of memory, no GPU and no network. (`nproc` prints 1 there because `OMP_NUM_THREADS=1` is set; the CPUs are 8.)

## Session time

Your session is stopped 4 hours after it started, and what is in `/home/user/submission/` at that moment is graded.

- **How to read the time.** `bash /home/user/session_time.sh` prints the minutes used and the minutes left, from a time stamp written when your sandbox was set up (a few minutes before your session started, so you have slightly more than it says, never less). Run it whenever you plan your next step; do not estimate the time from memory or from how much work you have done.
- **Use the time.** Stopping early earns nothing: the score is the program's, not the session's, and nothing in the session tells you that the program is good enough. Keep a working `run.sh` in the submission folder from early on and replace it as you improve it, so that the stop never finds the folder empty or half-edited.
- **Budget your runs.** A program near its processor-time limit takes half an hour per rig; three practice rigs fit side by side on your 8 CPUs.
