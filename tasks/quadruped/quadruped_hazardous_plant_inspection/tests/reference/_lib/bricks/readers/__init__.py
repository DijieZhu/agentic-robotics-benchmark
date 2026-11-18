"""The reference readers: classical image algorithms that read each instrument type from a frame and the public
catalogue hint. They are the privileged reference program's tools: they never ship to the agent and
never enter the cell's vendored library or the practice ground. The target modules
(bricks.targets.*) keep the old reader names, which resolve lazily to these modules; importing a target module does
not import this package or bricks.vision.

  bricks.readers.gauge   read(rgb, hint)
  bricks.readers.level   read(rgb, hint)                      v3: catalogue scale check
  bricks.readers.valve   read_lever(rgb, hint), read_gate(rgb, hint), read(rgb, hint)   v3 for read_gate
  bricks.readers.lamps   read(rgb, hint)
The tag reader stays in bricks.targets.tag (OpenCV's ArUco detector and PnP; no catalogue logic).
"""
