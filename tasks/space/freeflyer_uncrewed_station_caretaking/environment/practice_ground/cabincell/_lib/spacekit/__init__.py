"""spacekit: space-specific physics and firmware for MuJoCo 3.14 scenes of a cabin free-flyer.

Runtime dependencies: mujoco 3.14.0 and numpy only.  Deterministic (no wall clock, every random draw from
an explicit numpy Generator), no module-level mutable state.  See API.md for every public function.

Modules
  frames      rotating station frame (Coriolis, centrifugal, tidal, rotational terms), inertial variant
  stepping    mj_step1 -> wrenches -> mj_step2, leapfrog start, RK4 passive callback, slow-force cache
  propulsion  fan propulsion plant and force allocation of the reference free-flyer, NNLS alternative
  allocation  generic thruster matrix and Lawson-Hanson NNLS
  cabin       cabin air: drag, velocity fields, nozzle exhaust jets, grille suction
  control     flight controller, flight modes, trapezoidal planner, firmware executive
  sensors     IMU, localisation model, ray-cast depth, cameras, battery, joint and contact sensing
  render      optional mujoco.Renderer wrapper (the core never renders)
  robot       reference-robot adapter: spec values -> simulated free-flyer
  rotations   quaternion helpers (w, x, y, z)
  -- v0.2 (cabin mechanics, station, firmware-level primitives) --
  aero        drag on loose items (box / cylinder / sphere sample points) and the nozzle-jet field
  suction     lumped duct-and-fan network of return inlets, inlet builds, fan spin-down, item face suction
  gripper     spring-closed / tendon-opened gripper, arm servo, deploy / stow, MjSpec rigging
  grasp_rule  rule-based grasp constraint for items held in the hand
  fixtures    MJCF fragments: handrail, items with handles, soft item, stowage bag (spring flap or split flaps)
  station     ventilation field from the feature catalogue, per-module fans, two-zone CO2 balance
  zones       keep-in / keep-out validation (port of the firmware validator)
  primitives  undock / dock, fly-to with validation, loss and resume, perch, grasp / catch, factory nozzle test
  cameras     optional: robot cameras at the configured intrinsics (one OpenGL context), ray-cast depth
  -- v0.3 (the part-level reference build v1) --
  dock        berth of the reference dock v1: magnets, pogo pins, retraction, lance / cup guide geometry, the
              dock-relative (fiducial) localisation
  visual_twin display-only joints (impellers, nozzle flaps) and status lights driven from the plant into renders
  -- v0.3.1 (reference build v1.2) --
  refbuild    which adapter corrections a reference-build model already contains (its 'refbuild' text), so that
              the adapters do not apply them twice
  -- v0.4 --
  scenes      scene composition: the world body's static collision geometry grouped into cell bodies
"""
from __future__ import annotations

__version__ = "0.4.0"

from . import allocation, cabin, control, frames, propulsion, robot, rotations, sensors, stepping  # noqa: F401
from . import aero, fixtures, grasp_rule, gripper, primitives, station, suction, zones  # noqa: F401
from . import dock, refbuild, visual_twin  # noqa: F401
from . import scenes  # noqa: F401

# render and cameras are optional (need an OpenGL backend): import spacekit.render / spacekit.cameras explicitly.
