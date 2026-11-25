# Search camera: data sheet

The search camera is one fixed colour-and-depth camera at the head of the team's mast. It hangs about five metres
above the floor at the south side of a site and looks down, tilted towards the north, so that the site's floor is in
the picture from the south wall to about 4 m north of it, over the site's whole width. A site reaches further north
than that: its north part lies under the collapsed slab and is outside the picture. It is the only camera there is:
the robot has none. `station.get_frame()` returns its newest frame (`station_api.md`, section 4.1). The values below
are the rated values of this unit.

<!-- BEGIN GENERATED camera -->
| item | value |
|---|---|
| image | 640 × 480 px; colour 8 bit per channel; depth 16 bit, aligned to the colour image pixel for pixel |
| intrinsics `K` = `[fx, fy, cx, cy]` | 607.686, 607.987, 326.550, 250.662 px; a pinhole without lens distortion |
| pixel convention | pixel (u, v) with integer u and v is the centre of column u, row v; u grows to the right, v downwards |
| frame rate | 7 exposures per second |
| depth | the distance along the optical axis (the camera-frame z, not the length of the ray), in counts of 0.25 mm (4000 counts per metre); 0 = no reading |
| depth range | 0.25 to 9.0 m; nearer or farther reads 0 |
| depth noise | Gaussian, independent per pixel and frame, standard deviation 2.5 mm + 1.6 mm per metre of depth beyond 1 m |
| depth edges | where two neighbouring pixels (left-right or up-down) differ by more than 0.05 m in depth, the farther one reads 0 |
| colour | no noise is added to the colour image |
| extrinsic calibration | `T_world_camera` of `mission()` is the surveyed pose: within 5 mm in position and 0.2° in orientation of the camera's true pose |
<!-- END GENERATED camera -->

## Frames and projection

- **Camera frame**: x to the right of the picture (+u), y down the picture (+v), z forward along the optical axis.
- **`T_world_camera`** (4 × 4, in `mission()["camera"]`) maps a point of the camera frame to the site frame:
  `P_site = R · P_cam + t`, with `R` its upper-left 3 × 3 block and `t` its last column.
- **Projection.** A site point `P` has camera coordinates `(X, Y, Z) = Rᵀ · (P − t)` and appears at
  `u = fx · X / Z + cx`, `v = fy · Y / Z + cy`.
- **Back-projection.** A pixel `(u, v)` with depth reading `d` counts lies at `Z = d · depth_unit_m`,
  `X = (u − cx) · Z / fx`, `Y = (v − cy) · Z / fy` in the camera frame; `T_world_camera` takes it to the site frame.
- **In the picture** north is up and east is to the right, apart from the camera's small rotation about its axis.

## What to expect in a frame

- The colour image and the depth image are taken at the same instant and are aligned pixel for pixel.
- Depth is missing (0) at depth edges, as the table says: around the outline of everything that stands up steeply
  from the floor, the robot included. Where a surface slopes down to the floor, as the flank of a mound of fill does,
  depth is missing at a few pixels at most.
- The station's renderer adds no noise, blur or lens distortion to the colour image; what varies from site to site is
  the light (`site_briefing.md`).
- The pose in `mission()` is a calibration, not the truth: a site point computed from a pixel carries the calibration's
  error in addition to the depth noise. The error is constant during an episode.
