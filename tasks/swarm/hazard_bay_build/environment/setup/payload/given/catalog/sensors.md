# Sensors

Cameras are pinhole: focal length and sensor apertures in mm, image size in pixels. A camera counts only as a
`UsdGeom.Camera` prim with `arb:sensor_model` naming it and exactly these intrinsics (within 2 %); its mass counts toward the
robot's. A lidar is not a camera: declare it on any prim with `arb:sensor_model`; its mass counts. Every sensor that
counts streams its frames in the bay mission (`control.md`, The sensors: up to four cameras and one lidar a robot), as
the second table says (`catalog.json`: each model's `stream`).

| model | type | resolution | focal [mm] | aperture h × v [mm] | range [m] | vertical field [°] | mass [kg] |
|---|---|---|---|---|---|---|---|
| `CAM-RGB-W` | rgb | 1920 × 1200 | 4.0 | 11.26 × 7.04 | - | - | 0.060 |
| `CAM-RGB-M` | rgb | 1920 × 1200 | 8.0 | 11.26 × 7.04 | - | - | 0.065 |
| `CAM-RGB-T` | rgb | 1920 × 1200 | 16.0 | 11.26 × 7.04 | - | - | 0.090 |
| `CAM-RGB-S` | rgb | 640 × 480 | 3.6 | 4.80 × 3.60 | - | - | 0.012 |
| `THERM-A` | thermal | 160 × 120 | 1.7 | 1.92 × 1.44 | - | - | 0.015 |
| `THERM-B` | thermal | 640 × 512 | 13.0 | 7.68 × 6.14 | - | - | 0.110 |
| `THERM-C` | thermal | 640 × 512 | 9.0 | 7.68 × 6.14 | - | - | 0.095 |
| `TOF-D` | depth | 320 × 240 | 2.0 | 2.88 × 2.16 | - | - | 0.040 |
| `STEREO-X` | depth | 1280 × 720 | 2.1 | 5.00 × 2.80 | - | - | 0.075 |
| `LIDAR-2D` | lidar | - | - | - | 25 | 0 | 0.160 |
| `LIDAR-3D` | lidar | - | - | - | 30 | 90 | 0.350 |

What each model streams in the bay mission (every frame row-major, little-endian; a lidar's beam (row c, column k)
points along (cos e cos a, cos e sin a, sin e) in its prim's frame, a = k · 360° / columns counter-clockwise about +Z
from +X, e from the first row's elevation to the last's in equal steps):

| model | frame | format | noise | range [m] |
|---|---|---|---|---|
| `CAM-RGB-W`, `CAM-RGB-M`, `CAM-RGB-T` | 1920 × 1200 | uint8 RGB (sRGB) | none added | - |
| `CAM-RGB-S` | 640 × 480 | uint8 RGB (sRGB) | none added | - |
| `THERM-A` | 160 × 120 | float32 K | 0.10 K | - |
| `THERM-B`, `THERM-C` | 640 × 512 | float32 K | 0.05 K | - |
| `TOF-D` | 320 × 240 | float32 m, z-depth, 0 = no return | 4 mm + 0.5 % of z | 0.15 - 4.0 |
| `STEREO-X` | 1280 × 720 | float32 m, z-depth, 0 = no return | 1.55 mm · z² (z in m) | 0.3 - 10 |
| `LIDAR-2D` | 1 × 1440 beams, elevation 0° | float32 m, 0 = no return | 0.02 m | 0.15 - 25 |
| `LIDAR-3D` | 32 × 1024 beams, elevations +45° to −45° | float32 m, 0 = no return | 0.02 m | 0.3 - 30 |
