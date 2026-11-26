# Cloth material identification from video

A Franka Panda robot with a flat fabric clamp handles pieces of fabric on a table: it drags them across the table and
lets go, pulls them out over the table's front edge, lifts them by a corner and shakes them. The robot plans every fabric-handling motion in a cloth simulator, `cphys`. A plan works on the real fabric only if the
material the simulator is given makes it predict what the real fabric does. You have **RGB video** of the robot
handling ten fabrics, **A** to **J**. For each of them, choose the simulator material the robot should plan with. Your
materials are scored by what the robot then achieves on the real fabrics in two handling skills, when it plans with
them.

You have 8 hours of wall time (see "Session time"), 4 CPU cores, 8 GB of memory (processes that exceed it are killed),
no GPU and no network. Python 3.12 is in `/opt/venv` (NumPy, SciPy, OpenCV (headless, with FFmpeg), Pillow,
scikit-image, MuJoCo 3.16.0, and the robosuite 1.5.2 asset files for the robot model); the `ffmpeg` command is
installed. MuJoCo offscreen rendering works out of the box (`MUJOCO_GL=osmesa`, CPU).

## The fabrics

Each fabric is a 0.30 x 0.15 m rectangle. They are firm sheet fabrics of the kind used for bags, covers, upholstery
and workwear (canvas, denim, felt, coated cloth and the like): stiff enough that a few centimetres of unsupported
fabric hold some shape, each with its own weight, stiffness and surface. A fabric's look (its weave and colour) says
nothing about its material: the looks were dealt to the fabrics without regard to it, and every fabric is drawn
equally thick.

## What you get (`/home/user`)

```
data/cameras.json                      three fixed, calibrated cameras (cam0, cam1, cam2)
data/fabrics/<ID>/fabric.json          ID in A ... J and P: size, simulation grid, mass (kg), number of clips
data/fabrics/<ID>/clipNN/cam0.mp4      RGB video, 640x480, 25 fps, H.264 (also cam1.mp4, cam2.mp4)
data/fabrics/<ID>/clipNN/clip.json     clip kind, fps, number of frames
data/fabrics/<ID>/clipNN/clamp_500hz.npy   commanded clamp pose and jaw state (M, 8), one row per 2 ms physics step
data/fabrics/<ID>/clipNN/arm_q.npy     the arm's 7 joint angles per video frame (N, 7)
cphys.py, ctasks.py                    the robot's cloth simulator and the scored skills' code, as the grader runs them
carm.py                                the robot's visual model (for example to render robot masks)
cio.py                                 I/O helpers (video loading, cameras, projection, writing and checking the submission)
session_time.sh                        prints the minutes of your session used and left
submission/                            write params.json here
```

- **Cameras.** `cameras.json` gives, per camera, the OpenCV pinhole model: pixel ~ K (R_w2c X + t_w2c), no lens
  distortion. World frame: the table top is the plane z = 0 for x <= 0.30 (its front edge is the line x = 0.30), z up,
  metres. The calibration is exact.
- **Videos.** Rendered images of a workshop: the bench (the table), the robot on its pedestal and the fabric; the
  robot occludes parts of the scene.
  They are degraded like a real camera: motion blur, optical blur, vignetting, white balance and exposure differences
  between cameras, sensor noise and video compression. Frame k of a clip shows the fabric when the clamp is at row
  20 k of `clamp_500hz.npy` (time k/25 s). A fabric's face (up when it lies flat) and its back look different.
- **Clips.** Each fabric has 7 clips. At the start of every clip the fabric lies flat and at rest on the table, at a
  pose that is not given (it is in the first frames). Kinds:
  - `drag_release` (three clips): the clamp takes the short edge that leads, lifts it 10 to 20 mm, drags the fabric
    along the table towards the table's front edge and lets go while moving, once at each of three speeds between
    0.35 and 1.05 m/s;
  - `pull_over_edge` (two clips): the clamp takes the short edge nearest the table's front edge, lifts it a little,
    pulls the fabric out until that edge is 0.04 to 0.12 m beyond the table's edge, waits and lets go;
  - `lift_hang` (grasp a corner, lift, hang, lower partway) and `shake` (lift by a short edge, three quick vertical
    strokes, hold), one each.

  The clamp's commanded motion is exact; the jaws hold whatever part of the fabric lies between them when they close,
  and the robot does not place its grasp exactly on the fabric's corner or edge (it is off by up to 2 cm along the
  edge, 4 mm inward and 5 degrees).
- **The only information about the materials is in the images.** The size and the mass of each fabric are given (a
  ruler and a scale reading); nothing else about a material is.
- **Fabric P** is one more fabric of the same kind with the same seven clips. It is not scored and comes with no
  answer: use it to build and try your method.

## The robot's simulator and the real fabrics

The videos show the **real** fabrics, and the scored handling is executed on the real fabrics. `cphys.py` is the
robot's planning simulator: **an approximation of a real fabric**, coarser than the real thing in space and time and
without everything that acts on a real sheet. No material makes it reproduce a real fabric exactly. The answer wanted
for a fabric is therefore not a physical measurement: it is **the `cphys` material with which plans made in `cphys`
succeed on the real fabric** in the two scored skills.

`cphys.py`: MuJoCo 3.16.0, timestep 2 ms, the `discrete` integrator, the CG solver, elliptic friction cones. A fabric
is simulated as a MuJoCo `flexcomp` grid of 21 x 11 vertices (2D shell, every vertex a point mass, the total mass
given), with MuJoCo's shell elasticity, Rayleigh damping and one friction coefficient for all of its contacts; it does
not collide with itself. The table has zero friction of its own, so the fabric's coefficient decides every contact. The clamp holds the vertices inside its jaws with stiff equality constraints and never touches
anything. `cphys.Sim` builds this world from **your** material and replays a clamp program, so you can simulate any
clip with your own material. The table, clamp, robot, cameras and clamp programs are the same as in the videos.
One simulation of a clip or of a scored handling takes about 5 to 15 s on one core (a pull over the edge is the
longest); planning one test takes up to ten of them.

## Deliverable

`/home/user/submission/params.json` (`cio.write_params` writes it; `python3 cio.py` checks its format):

```json
{"fabrics": {
  "A": {"young": ..., "poisson": ..., "thickness": ..., "damping": ..., "mu": ...},
  "B": {...}, ..., "J": {...}}}
```

- The five values are MuJoCo's flex material parameters exactly as `cphys.scene_xml` uses them: `young` (Young's
  modulus, Pa), `poisson` (Poisson's ratio), `thickness` (the shell thickness that, with `young` and `poisson`, sets the
  stretch, shear and bending stiffness; m), `damping` (Rayleigh damping, s) and `mu` (sliding friction). Different
  values that behave the same in `cphys` are equally good: only the robot's results on the real fabric are scored.
- Rules: numbers only; each value is clamped to its admissible range (`cphys.RANGES`). A fabric with a missing or
  non-numeric value scores 0; a missing, unreadable or non-conforming file, a symlink or a special file scores 0.
- The format check tells you whether the file will be read. Nothing in this sandbox tells you how good a material
  is: there is no score feedback before the session ends.

## Scoring

For each fabric A to J the grader runs eight hidden tests, four of each skill below (poses, lifts, release distances,
pulling speeds and targets are hidden). Every test starts with the fabric flat at rest at a pose the robot knows; the
robot chooses one control value by planning with `ctasks.plan` in `cphys` made of **your** material, so that the
predicted outcome hits the test's target, and then executes that value once on the **real** fabric
(`ctasks.run_test`).

| Skill | What the robot does | Control value it plans | Outcome that must hit the target | Tolerance |
|---|---|---|---|---|
| slide | takes the short edge that leads, lifts it 15 mm, drags the fabric along the table towards the table's front edge and lets go while moving | the drag speed, 0.3 to 1.1 m/s | how far the fabric's leading point stops beyond the release point (targets between 35 and 110 mm) | 3 mm |
| overhang | takes the short edge nearest the table's front edge, lifts it 12 to 20 mm, pulls the fabric along its long axis out over the edge, waits 0.4 s and lets go | how far that edge is pulled beyond the table's edge, 0.03 to 0.13 m | the mean height of that edge once the fabric has settled (targets between 12 and 80 mm below the table top) | 6 mm |

- A test scores q = clip(2 - error / tolerance, 0, 1): full marks within the tolerance, nothing at twice the
  tolerance. Every target is one the real fabric reaches for some control value in the range.
- **A fabric's score is the product of its two skill means** (the mean q of its four slide tests times the mean q of
  its four overhang tests): a material is worth planning with only if its plans work in both skills. The metric
  `material_transfer_score` is the mean over the ten fabrics, and the reward is the metric (1.0 when every test is
  within its tolerance).
- The mass, gravity, the clamp programs and the physics settings always come from the grader; nothing you write is
  executed.

## Session time

`bash /home/user/session_time.sh` prints how many minutes of the 8 hours are used and left. The session ends at the
limit without warning, and whatever `submission/params.json` holds then is graded: write a complete file early and
improve it.

## Numerics

Every `/opt/venv` Python computes the same here and in the grading run, on whatever machine: the image fixes NumPy's
CPU-specific loops (`NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`), its linear-algebra kernels
(`OPENBLAS_CORETYPE=Haswell`, `OPENBLAS_L2_SIZE=256`), one thread for them and OpenCV's IPP (`OPENCV_IPP=avx2`), as
defaults that apply when the variables are not set.
