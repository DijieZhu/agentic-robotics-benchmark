# The chaser's sensors

What the craft's sensors deliver in each `act` request's `obs` (the task instruction's wire contract), where they sit
and how good they are. Like a real chaser docking with a cooperative target, the craft fixes its own attitude with a
star tracker and an inertial unit, and measures the target with a docking camera that looks along the docking axis
at the fiducials the target carries for it (`target.md`). The plant itself is in `model.md`; sample frames with their
true geometry are in `frames/` (`frames/README.md`).

| field | sensor | what | when it is missing |
|---|---|---|---|
| `frame` | docking camera (DOCK) | a 640 x 480 monochrome image, captured 1-3 steps earlier, with its capture step | `null` during outages (below) |
| `star_tracker_yaw` | star tracker | the craft's yaw now | `null` during the re-key's avionics restart |
| `imu` | inertial unit | three channels: the body acceleration of the last step, in the IMU's own mounting | never |
| `boom_encoder` | the target's telemetry | the boom's extension, with an unknown polarity | never |
| `fuel` | propellant gauge | the propellant left, 1.0 full | never |
| `episode_fraction` | mission clock | `t / 400` | never |

## The docking camera (DOCK)

**Mount.** On the docking face (the body's -x face), the lens centre at (-0.082, 0, -0.115) m in the body frame:
centred across the face and 0.115 m below the probe's axis. The boresight is the docking axis, body -x. Image right
is body +y and image down is body -z, so the image's top is up (+z). The craft's own body is outside the field.

**Optics.** A pinhole without distortion and a global shutter: 640 x 480 pixels, `fx = fy = 320` px, principal point
`(cx, cy) = (319.5, 239.5)`, with pixel centres at integer coordinates (`u` to the right from the left column, `v`
down from the top row). The field is 90.0 x 73.74 degrees, 3.1 mrad a pixel at the centre. A point with camera
coordinates `(X, Y, Z)` (`Z` along the boresight, `X` to the image's right, `Y` to the image's bottom) is imaged at
`u = cx + fx X / Z`, `v = cy + fy Y / Z`.

**Response.** Monochrome and linear in scene radiance (the luma `0.2126 R + 0.7152 G + 0.0722 B` of the scene's
colour), 8 bits, at a fixed exposure (EV100 15: f/16, 1/125 s, ISO 100; no automatic exposure or gain). A white
diffuser of albedo 0.8 facing the sun reads about 198 DN; the target's LEDs saturate (255) wherever they are seen; the
black sky reads 0 before noise. A pixel saturated in all colours reads 255.

**Noise.** Per pixel and per frame, independent and Gaussian: `sigma = sqrt(0.8**2 + DN / 39)` DN (read noise
0.8 DN; 39 electrons per DN; full well 10,000 electrons), after which the value is rounded to the nearest integer and
clipped to 0-255. Two frames with the same capture step carry the same noise.

**Timing.** One frame per step. A frame captured at step count `s` (its `stamp`) shows the state that the request with
`t = s` describes. It is delivered with a fixed latency of 1, 2 or 3 steps (one value for the whole episode): the
request with `t` carries the frame stamped `max(0, t - latency)`, so for `t <= latency` the frame of stamp 0 repeats.
`t - stamp` is the frame's age.

**Outages.** `frame` is `null` when no fresh frame arrives: during two outages of 7 to 18 steps (0.8 to 2.2 s) each,
at varying times in the first 21 s of an episode; for single frames dropped at random, a few percent of them; and
during the avionics restart that follows the re-key (`model.md`): the 29 to 41 observations after it. Missing
frames are not delivered later.

**Format.**

```json
"frame": {"stamp": 34, "data": "<409,600 characters of base64>"}
```

`data` decodes to 307,200 bytes, the image row by row, the top row first:

```python
import base64, numpy as np
img = np.frombuffer(base64.b64decode(obs["frame"]["data"]), np.uint8).reshape(480, 640)   # img[v, u]
```

**What it sees.** The target, its fiducials and their shadows (`target.md`); the Earth below, its horizon about
20 degrees under the boresight (115 pixels below the image centre) when the craft is level; the black sky; the sun
(a saturated disc) when it is in the field. No thruster plumes are visible.

## The star tracker

On the top face (+z), looking at the zenith (the sun is 54 degrees from its boresight). `star_tracker_yaw` is the
craft's yaw `theta` in the inertial frame now (no latency), wrapped to [-pi, pi), with independent Gaussian noise of
2.0e-4 rad per reading. It is `null` during the avionics restart that follows the re-key, the same observations in
which the camera delivers nothing; at every other request it is there.

## The inertial unit (IMU)

`imu` holds three channels, current: two linear channels and a yaw-acceleration channel. They report the body
acceleration that thrust and the bias produced during the step just taken (not the residual damping), in the IMU's own
mounting: the linear pair is the body-frame `(ax, ay)`, possibly mirrored, then turned by an unknown angle (any
angle); the yaw channel's sign is unknown. Noise, independent per reading: at most 0.006 m/s² on each linear axis of
the IMU's own frame (after the re-key's remount, below, the noise turns with the reading) and at most 0.004 rad/s² on
the yaw channel. The request with `t = 0` follows no step: its IMU reads noise alone.

At the re-key the redundant IMU mux silently selects a new mounting: from the observation that follows the re-key's
step, the linear pair is turned a further 0.43 or 0.79 rad either way and possibly mirrored, and the yaw channel's sign
independently stays or flips. Nothing marks the change.

## The boom's encoder (the target's telemetry)

The target's boom controller reports the boom's extension over the proximity link, as cooperative targets share
their state with visiting vehicles. `boom_encoder` is the port's radius now, without latency or noise, but with an
unknown polarity: either `r_port` itself or `2 * 1.14 - r_port`. The polarity may change at the re-key, from the
observation that follows the re-key's step. It does not report the boom's rate.

## The propellant gauge and the clock

`fuel` is the propellant left (1.0 full), exactly. `episode_fraction` is `t / 400`.

## A repeated request (the probe copy)

A step is sometimes asked twice with the same `t` (the task instruction's wire contract): once with the true
observation and once with a copy whose camera frame shows the craft displaced by up to 0.28 m along each inertial axis
and turned by up to 0.18 rad at the frame's capture, and whose star-tracker yaw is turned by the same angle; every
other field is the true one. A request without a frame (or without a star-tracker reading) has a copy without one too.

## The numbers, for programs

```json
{
 "image": {
  "width": 640,
  "height": 480,
  "channels": 1,
  "bits": 8,
  "response": "linear",
  "row_order": "top row first"
 },
 "intrinsics": {
  "fx": 320.0,
  "fy": 320.0,
  "cx": 319.5,
  "cy": 239.5,
  "distortion": null,
  "pixel_centres": "integer (u right, v down from the top-left pixel)"
 },
 "mount_body_m": [
  -0.082,
  0.0,
  -0.115
 ],
 "boresight_body": [
  -1.0,
  0.0,
  0.0
 ],
 "image_right_body": [
  0.0,
  1.0,
  0.0
 ],
 "image_down_body": [
  0.0,
  0.0,
  -1.0
 ],
 "noise": {
  "read_dn": 0.8,
  "electrons_per_dn": 39.0,
  "full_well_electrons": 10000.0,
  "sigma_dn": "sqrt(read_dn**2 + DN / electrons_per_dn)",
  "rounding": "nearest, clipped to 0..255"
 },
 "luma": [
  0.2126,
  0.7152,
  0.0722
 ],
 "star_tracker": {
  "sigma_rad": 0.0002,
  "output": "yaw in the inertial frame, wrapped to [-pi, pi)"
 }
}
```
