"""site_gen: procedural grey-box industrial plant generator for the quadruped inspection task.

Entry points:
  site_gen.generator.generate(seed, out_dir, cfg=None)   -> site folder (site.json, prior.json, scene.xml, maps/)
  site_gen.compose.compose(site_dir, robot_xml=None)     -> (MjSpec, MjModel, info) with the robot on the pad
  site_gen.compose.load_site(site_dir)                   -> Site (truth queries, traversability, dose, link)
"""

__version__ = "0.1.0"
