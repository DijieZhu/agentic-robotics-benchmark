"""The bench of worklist v1 (probe 8): the incoming rack, the cold rack, the rack camera.

- Incoming rack: one row of ten positions A1-A10 along +y, 45 mm apart, a wire rack (its 50 mm deep rings drawn
  near-transparent, so a tube's liquid shows; RACK_CLEARANCE around the tube); the rack sits within RACK_SHIFT of
  its nominal place per episode.
- Cold rack: localize_probe's 3 x 3 orange rack (45 mm pitch, rows A-C along +x, columns 1-3 along +y), 50 mm
  deep, within COLD_SHIFT of its nominal place per episode; beside the robot, out of the rack camera's view (its
  orange behind a tube read as liquid).
- Rack camera: a fixed camera on the robot's side of the incoming rack, level with the tubes and a little above,
  looking across the row (no tube in front of another), for reading the fills.
- Tubes: translucent, with a red liquid column (visual), capped blue.
"""

from lab_sim import (  # noqa: F401
                     COLD_DEPTH,
                     COLD_PITCH,
                     COLD_SHIFT,
                     COLD_XY,
                     N_SLOTS,
                     RACK_CAM,
                     RACK_CLEARANCE,
                     RACK_PITCH,
                     RACK_RGBA,
                     RACK_SHIFT,
                     RACK_X,
                     RACK_Y0,
                     cold_names,
                     layout,
                     rack_cam_K,
                     rack_mid_z,
                     slot_names,
)
from lab_sim import build_bench as build  # noqa: F401
