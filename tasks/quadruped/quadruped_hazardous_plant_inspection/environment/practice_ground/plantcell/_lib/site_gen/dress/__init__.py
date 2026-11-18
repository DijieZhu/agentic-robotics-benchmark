"""site_gen.dress: the look of a stored site on MuJoCo's classic renderer (version 2).

Run-time half, like site_gen.loader: nothing here imports another site_gen module, only numpy, mujoco and the
standard library. The dressing is a pure function of the site package (its scene and its site.json, the site's seed)
and of the look bundle; it is applied by the loader (site_gen.loader.compose, after the robot is attached), so a
stored package stays as it is (no dressing record), the same package and bundle always give the same frames, and
every geom of the site and of the robot keeps its id: the dressing only appends (textures, materials, meshes, one
body per decor item and the skins' bodies, all after the robot) and re-materials non-colliding or decorative geoms.

    from site_gen.dress import dress_spec, settings
    rep = dress_spec(spec, site)          # spec: an MjSpec (site, or site + robot); site: site.json as a dict
    rep["checks"]                         # what was checked (DressError when a check fails)

Switch (environment): SITE_GEN_DRESS=0 off, =1 required (DressError without a bundle), unset or 'auto' = on when the
bundle is found. Bundle: SITE_GEN_DRESS_BUNDLE (a directory holding manifest.json), else /opt/task-assets/<BUNDLE>.

What it does:
  * skins: visual meshes 1 mm in front of every face of the walls, partitions, lintels and shield walls (the
    room's two-tone paint with its dado, block or concrete, metric texture scale, the dado's height fixed in metres),
    on the floors (tiles, epoxy or concrete by room class; the decorative floor slab is hidden, the skin lies at its
    top) and under the ceilings; subdivided so the classic renderer's per-vertex lighting is smooth;
  * pack materials on equipment, pipes (service colours), doors, steel, by the generator's material name;
  * pack meshes for decorative items (the ceiling battens) and new decor on walls above 1.0 m or at the ceiling,
    clear of every instrument, plate, sign and marker by 0.3 m and outside their reading regions' sight lines;
  * nothing on instruments, tag plates, room plates, grid markers, hazard markings, the pad or any generated sign.
"""

from .checks import verify_compiled, yellow_mask, yellow_offenders  # noqa: F401
from .core import (BUNDLE, DRESS_VERSION, DressError, bundle_dir, dress_spec, enabled, maybe_dress,  # noqa: F401
                   settings)
