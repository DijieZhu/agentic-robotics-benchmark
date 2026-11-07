# The target: interface document

What the docking camera sees of the target: its shape, its surfaces, the docking boom, the fiducials it carries for
visiting vehicles, and the light. Positions are given in the **port frame**: x radially out along the port's bearing
`phi` (the target's phase, `model.md`), y along increasing `phi`, z up, origin on the target's spin axis in the plane of
the craft's motion. The camera is in `camera.md`; sample frames with their true geometry are in `frames/`.

## The drum

- A drum about the inertial z axis, radius 0.82 m (the plant's collision radius): its side wall spans z = -0.67 to
  +0.67 m. Above the top deck stand a platform and an antenna mast (to z = 1.08 m); below the bottom deck a mast
  reaches z = -1.27 m.
- Surfaces, as on a real spacecraft: the side wall is covered with silicon solar cells under cover glass (dark
  navy-violet, specular, a fine grid of cells), divided by white beta-cloth bands and a belt of gold multi-layer
  insulation (crinkled aluminised Kapton) round the middle; the decks are gold insulation; there are small black,
  grey and aluminium fittings.
- The docking well faces the port's bearing at z = 0: a blanket of silver insulation (crinkled) about 0.33 m wide and
  0.27 m tall (z = -0.160 to 0.108) round a bare aluminium collar (outer radius 0.075 m) and the well's black opening
  (radius 0.064 m), from which the boom comes out.

## The boom and the drogue

- The boom is a carbon-fibre tube (radius 0.018 m) along the port frame's x axis at y = z = 0. Its end carries the
  drogue: an aluminium cone opening outward (rim radius 0.058 m), whose apex, the bottom of the cone, sits at
  `x = r_port - 0.1299` m; the rim's outer face is 0.037 m above the apex. The docking probe of a docked craft rests in
  the cone. The boom's length follows the port's radius `r_port` (`model.md`), which the boom's encoder reports
  (`camera.md`).

## The fiducials

The LEDs are white discs 20 mm across, 2 mm proud of their mounts, bright enough to saturate the docking camera
wherever their face is seen; everything they are mounted on is black.

- **The docking target (DT)**, fixed to the drum below the port, at the camera's height, the way the camera sits
  below the probe (a docked craft sees it straight ahead at the hold):
  - a black backplate 74 mm wide (y = -0.037 to 0.037) and 74 mm tall (z = -0.152 to -0.078), its front face at
    x = 0.8325 m;
  - four flush LEDs in a 50 mm square, their faces at x = 0.8345 m, centred at (y, z) = (+0.025, -0.140),
    (-0.025, -0.140), (+0.025, -0.090) and (-0.025, -0.090) m;
  - the standoff LED S, its face at x = 0.8645 m (0.030 m in front of the others), centred at (0, -0.115) m on a black
    post of radius 0.004 m: seen off the docking axis it shifts against the square of flush LEDs, which shows the
    camera's direction to the target.
- **The boom-tip target (BT)**: three LEDs on black tabs at the drogue's rim, their faces 0.039 m above the drogue's
  apex (at x = `r_port - 0.0909` m), centred at (y, z) = (+0.072, 0), (-0.072, 0) and (0, -0.072) m. They move with
  the boom; close to the hold they leave the camera's field (the low one last).

## The light

- **The sun**, fixed in the inertial frame: its direction (from the target towards the sun) is (0.6198, 0.5200,
  0.5878), azimuth 40 degrees from +x towards +y and elevation 36 degrees; about 118,000 lux. The target, the boom and
  the craft cast shadows; the side of the drum away from the sun is lit only by the Earth.
- **The Earth** is 400 km below: its sunlit disc (ocean, cloud, land, a hazy limb) fills the view below the horizon,
  and its light (Earthshine, a few thousand lux from below) is what shows the drum's shaded side.
- **The sky** is black; the stars are too faint for the camera's exposure.

## The numbers, for programs

```json
{
 "frame": "port frame: x radially out along the port's bearing, y along increasing target phase, z up",
 "drum_radius_m": 0.82,
 "led_disc_diameter_m": 0.02,
 "docking_target": {
  "backplate_m": {
   "x_front": 0.8325,
   "y": [
    -0.037,
    0.037
   ],
   "z": [
    -0.152,
    -0.078
   ]
  },
  "leds_m": {
   "DT1": [
    0.8345,
    0.025,
    -0.14
   ],
   "DT2": [
    0.8345,
    -0.025,
    -0.14
   ],
   "DT3": [
    0.8345,
    0.025,
    -0.09
   ],
   "DT4": [
    0.8345,
    -0.025,
    -0.09
   ],
   "DTS": [
    0.8645,
    0.0,
    -0.115
   ]
  },
  "standoff_post_m": {
   "radius": 0.004,
   "x": [
    0.8325,
    0.8625
   ]
  }
 },
 "boom_tip_target": {
  "drogue_apex_radius_m": "port_radius - 0.1299",
  "leds_above_apex_m": {
   "BT1": [
    0.039,
    0.072,
    0.0
   ],
   "BT2": [
    0.039,
    -0.072,
    0.0
   ],
   "BT3": [
    0.039,
    0.0,
    -0.072
   ]
  }
 }
}
```
