"""bricks: reusable building blocks for the quadruped plant-inspection MuJoCo cell.

Sub-packages (import what you need; importing has no side effects):
  bricks.common          poses, quaternions, MjSpec helpers (materials, PNG textures, meshes), TargetNames
  bricks.vision          classical image helpers shared by the reference readers
  bricks.targets.*       gauge, valve, level, lamps, tag, thermal_source
  bricks.sensors.*       camera (+ head rig), thermal, lidar, radiation
"""

__version__ = "0.1.0"
