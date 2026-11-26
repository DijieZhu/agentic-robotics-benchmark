# Collision twin of a cabin from a recorded flight

You are the engineer who rehearses flights of the cabin free-flyer, a small cube-shaped robot that flies inside a
pressurised module. Before a flight is rehearsed in simulation, the module has to exist as a collision model in the
state the robot found it in: walls and racks, and everything that was stowed, strapped or parked in the module during
the recorded session.

From the recorded camera frames of one real flight of the robot (`flight_a`), build a **static collision twin** of the
module as found (an MJCF scene), and declare the pose of the robot's forward wide camera at every one of the flight's
1068 given frames.

## What you are given (in `/home/user/`)

- `flight_brief.md`: the flight, the module, the robot and its sensors, what the twin is for, what counts as the
  environment and what is not scored. Read it first.
- `twin_contract.md`: the exact interface: files, frames and units, the time convention, how the grader probes your
  twin, the allowed MJCF, the caps, every rule that can set a score to zero, and the formulas.
- `flights/flight_a/`: the flight's 1068 forward wide-camera frames (`frames/f_NNNN.png`, grey, one byte per pixel, in
  recording order), `frames.csv` (their names and recorded stamps), `imu.csv` (the inertial record) and
  `onboard_estimate.csv` (the robot's own estimate of its pose, with the estimator's health fields).
- `sensors/`: `camera_sheet.json` (the wide camera's model and mounting, the measured time relation, the depth
  camera's image, the inertial unit's mounting), `depth_rays.npy` (one unit ray per pixel of the forward depth
  camera), `wide_to_depth.json` (where the depth camera sits relative to the wide camera) and `probe_card.json` (the
  hazard check's parameters and its family of straight moves).
- `module_shell/`: `shell.obj` (the maker's mesh of the bare module, geometry only, in the robot's map frame) and
  `zone_record.json` (the maker's dated record of the module's keep-out and keep-in boxes, as given).
- `tools/check_twin.py` with `tools/twinprobe/`: the grader's own gate, ray, hazard and fit code. It checks your twin
  the way the grader's gates do and shows your twin's depth and hazard view at any given frame; it reads no truth,
  scores nothing and does not know which moments are scored.

The depth camera's record of the flight is **not** in the package: it is what your twin is scored against.

## What you deliver

`/home/user/submission/flight_a/`: `scene.xml` (with any mesh files it names) and `cameras.json` (format
`flight-twin-cameras/1`). `cameras.json` declares the wide camera's pose at every given frame of the flight (1068
poses): one entry for each frame of `frames.csv` and no other entry (rule L). The twin's frame is yours to choose; the
grader neither aligns nor scales it. The submission folder as it stands when the session ends is what the grader
reads.

## How it is scored

The grader scores 24 moments of the flight and does not say which. No given file names them. At each scored moment the
grader takes your declared pose of that frame, places the forward depth camera (by the given two-camera transform),
casts its rays into your twin and compares with the depth camera's real record at that instant. Your other poses are
checked for form only.

- the **depth term** D: per scored pixel, the range your twin gives against the measured range, within a band that
  grows with range; the module's as-found content (stowage and other things in front of the bare module) and its
  structure (the module itself) are scored apart as Dc and Ds, and D is the product of Dc and Ds (rule S);
- the **hazard term** H: the robot's own hazard check of straight moves, run on your twin's depth frame, must find
  clear what is clear and blocked where it is blocked, at the right distance. One wrong obstacle verdict at a scored
  moment loses that moment (rule P): H is the lower of the moves' own score and the share of scored moments kept.

The flight scores H x D, between 0 and 1. The twin score is that flight's score (or 0 under rule Z1 below), and it is
the reward.

Rules that can set a score to zero (each is defined exactly in `twin_contract.md`):

- **Z1** the integrity gate: anything in the submission written after the session ended, more files or bytes than the
  caps, or a symbolic link or special file: the whole score is 0;
- **Z2** the validity gate: a missing or invalid `scene.xml` or `cameras.json`, a `cameras.json` that lacks a given
  frame or holds another name, an element off the allow list, a scene that does not compile or is over a cap: the
  flight is 0;
- **Z3** fewer than 12 of the scored moments' poses agree with the on-board estimate as one place (a fit of rotation,
  translation and scale): the flight is 0;
- **Z4** no valid pose at a scored moment, **Z5** the wide or the depth camera's centre inside a solid of your twin,
  **Z6** a pose that does not agree with the rest of the flight: that scored moment scores 0 on all its pixels and
  moves;
- **Z7** the grader cannot build the flight's ray scene inside 120 s or cast its rays inside 300 s of its wall clock,
  or either does not fit inside 6144 MB of address space: the flight is 0 (`tools/check_twin.py` builds and casts
  under the same memory limit and prints its own times beside these limits);
- **S1** a scored pixel whose ray meets nothing in your twin, or whose range is off by the band's zero end or more,
  scores 0, and so does a move that your twin blocks where it is clear, finds clear where it is blocked, or stops off
  by the stop band's zero end or more; **S2** the structure counts only multiplied by the content; **S3** the hazard
  term is 0 unless clear moves and moves blocked by content are told apart.

## Session

8 CPUs (use 8 threads), 32 GB of memory, no GPU, no network, 6 hours. Installed: mujoco 3.16.0, numpy, scipy, opencv
(headless), scikit-image, pillow, trimesh, manifold3d, rtree, shapely, coacd, open3d (CPU), pycolmap (CPU), the
OpenMVS CPU tools, MuJoCo's offscreen renderer (OSMesa). Run `python3 /home/user/tools/check_twin.py` on your
submission before you finish.
