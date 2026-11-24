# How the rigs' logs were generated (verify never runs this)

The practice and hidden rigs are generated ONCE, offline, and stored: the practice rigs' logs as one archive of the
task's assets (fetched at image build), the hidden rigs' logs as one archive fetched at verify (`tests/verify.py`),
the hidden rigs' truth in `tests/hidden/`. This folder is the generator, kept for review and for regenerating a rig.

| file | what |
|---|---|
| `rigsim.py` | a rig's parameters from its seed; motion, clocks, IMU, odometry, faults; the LiDAR sweep model; the camera's lens, noise and JPEG; the log writers; the held-out data |
| `rigsite.py` | the site: the hall, the props placed per rig, the robot's body; ray casting (Embree) on their triangles; occupancy and prop placement; the near-infrared reflectance table |
| `assets.py` | triangles and material names of a USD file (the files the renderers draw) |
| `bl_render.py` | Blender (EEVEE, headless): the pinhole source image of every camera frame from the same USD files |
| `gen_rigs.py` | one rig end to end |

```
<python 3.12: numpy 2.5.4, scipy 1.18.1, opencv-python-headless 5.0.0.93, pyyaml, embreex, usd-core> -I -B gen_rigs.py \
    --set practice --index 1 --name rig_b --odom 0 --knock 0 --hall <site>/hall.usdc --props <site>/props \
    --husky <site>/husky --people <dir> --cache <dir> --out <root> [--blender blender]        # Blender 5.3
```

## One scene, two renderers

- **Site.** A warehouse hall (60 x 40 x 11 m; 24 materials with their maps: polished concrete floor with painted
  lanes, panel walls, roller doors, poles, movable partitions, cable trays, ducts, lamps) and 34 kinds of warehouse
  and factory props at their own size. Per rig a working zone of 24-34 x 16-24 m is drawn and 30-45 small and 3-6
  large props are placed in it by rejection sampling, each at least 0.9 m clear of the rig's five drives; every
  kind is used once before any repeats. In every drive 4-7 vehicles (forklift, reach truck, pallet truck, cart,
  hand truck, pallet jack) run straight legs back and forth at 0.5-2 m/s, clear of the props by 0.4 m and of the
  robot by 1.2 m. (Walking people are not in this version: `rigsite.sample_movers`.)
- **Camera frames** are rasterised by Blender EEVEE from those USD files, movers at the frame's time; the
  demonstration pictures are path traced (OVRTX) from the same files. The frame is drawn as a pinhole image at
  1.5 x the focal length at the middle of its readout. The logged frame is that image re-projected row by row
  with the camera's motion over the readout (rolling shutter; depth by ray casting on the same triangles), through
  the rig's lens distortion, then vignetting, exposure, sensor noise and JPEG (`rigsim.rolling_shutter_maps`,
  `rigsim.finish_frame`).
- **LiDAR** rays are cast on the triangles of the same USD files, every column from the LiDAR's pose at its own
  time, on the movers at that time, plus the robot's own body (Clearpath Husky A200 meshes with the 510 mm sensor
  arch). Reflectivity comes from a table of near-infrared reflectance per material class (`rigsite.NIR`), not from
  the colour maps.
- **Generation inputs** (kept by the maintainers, not published; the task does not need them to run): the hall
  and the props are conversions to USD (geometry, UsdPreviewSurface materials, colour / normal / ORM maps) of a
  licensed warehouse environment pack and a licensed warehouse-and-factory props pack, made with the
  maintainers' Unity-package converter; `site_inputs.json` beside this file lists every scene file with its
  sha256 and triangle count.

## Sensors (datasheet figures used)

| sensor | model | given | unknown, to be calibrated |
|---|---|---|---|
| camera | FLIR Blackfly S BFS-U3-63S4C (Sony IMX178, rolling shutter), 4x4 binned to 768 x 512, 6 mm lens, 10 Hz free-running | nominal focal length 625 px (6 mm on 9.6 um binned pixels), nominal principal point; exposure 2 ms; vignetting 25 % at the corner; noise 1.2-2.0 levels; JPEG quality 92 | focal length within 6 % of nominal (re-focused), fx / fy within 0.3 %, principal point within 20 px, k1 -0.32..-0.08, k2 -0.02..0.14, p1, p2 within 0.0015, k3 within 0.02; readout time 10-30 ms |
| LiDAR | Velodyne VLP-16, 600 rpm | 16 beams, -15..+15 deg; 1800 columns a sweep (0.2 deg), each column at its own time; minimum range 0.5 m; range noise 2 cm rising to 3 cm at 60 m; a return is lost under the level of a 10 % target at 60 m, at grazing incidence and for 0.5 % at random; mixed returns at depth edges | - |
| IMU | Bosch BMI088, 200 Hz | noise 0.0035 rad/s and 0.024 m/s^2 per sample; scale and cross-axis within 0.1 %; bias walk 2e-5 rad/s and 2e-4 m/s^2 per sqrt(s) | gyro bias within 0.015 rad/s, accelerometer bias within 0.15 m/s^2; the board's orientation (any of 24 axis-aligned, off by up to 5 deg) |
| wheel odometry | Husky A200 encoders, 50 Hz; absent on some rigs | speed scale within 3 %; yaw-rate factor 0.80-1.00 (skid steer); slip up to 4 % accelerating and 3 % turning | - |

Mounts: the camera's true mount is the CAD mount in one of four bracket states (as designed, a quarter turn either
way, a half turn about the LiDAR's z axis), then off by 2-10 deg and 2-8 cm. On some rigs the bracket is knocked
once during the second given sequence (1.5-4 deg, 2-8 mm).

## Time

The camera and the LiDAR run free on their own clocks for the whole session: frame n happens at sensor time
`phase + n * 0.1 s + trigger jitter` and is stamped with that time plus a stamping jitter; the reference time of the
event is `(s - offset) / (1 + drift)`. Offsets are within 0.5 s, drifts within 200 ppm (camera) and 100 ppm (LiDAR).
A sensor's log of a sequence starts 0.2-1.0 s after the sequence starts and ends as long before its end, drawn per
sensor and sequence. So neither the stamps' spacing nor where a log starts says anything about offset or drift
(`tests/review/stamp_leak.py` measures that on every rig). A session is five drives: a held-out one, the three
given sequences, a held-out one.

## Rigs

- Seed of rig i of a set: the first four 32-bit words of
  `sha256("ale-robotics/multisensor_rig_calibration_from_logs/v3/<set>/<i>")` (`rigsim.rig_seed`), set `practice`
  (rig_a, rig_b, rig_c) or `hidden` (rig_00 .. rig_05). A rig that fails the acceptance check (the generator's own
  placement checks; every scored measure pinned inside its full point by `tests/review/ident.py`) is replaced by
  index `<i>.1`. No index was rejected in this version (practice 0, 1, 2; hidden 0 .. 5).
- Wheel odometry: practice rig_a and rig_c, hidden rigs 0, 2, 4. A knock: practice rig_c, hidden rigs 1, 3, 4.
- The practice rigs are published as logs only; their truth exists but is not shipped.
- Everything but the camera frames is a pure function of the seed and the asset files (.npz members carry fixed
  dates). A second rendering of a sequence repeats the JPEG files but for about 1.5 % of them, which differ by at
  most 4 levels in under 100 pixels. The stored logs are the task: the practice archive and the hidden archive are
  pinned by sha256, and every hidden file again by `tests/hidden/<rig>/sha256.json`.
