"""Cell core of the void-search task (07): the site in the simulation base, the primitive executor, the truth.

    world.py     site package (site.json + site.xml of the W2 generator) -> ysim model; per-site floor friction;
                 the set-down (rest shape 0.05 m above the pad, seeded position error and orientation, settle,
                 the landing face must be one of the six rollable faces)
    executor.py  run_primitive(name, bottom_nodes) / set_targets / stop / status / step: tick-based, the simulated
                 time advances only in step(dt); the symmetry map is taken from the PASSED bottom triple
    truth.py     node positions, planar centroid and heading, bottom triple, occupancy and opening classes from
                 the site's boxes, contacts with the world, rod and tendon penetration into boxes

The robot is the simulation base's model, used as it is (read-only import of ysim; no file of it is changed):
three copies of the authors' strut mesh, the authors' masses and tendon net. Nothing here models, redraws or
completes any part of the robot.
"""
CELLCORE_VERSION = "cellcore 0.1.0"
