"""site_gen.loader: the run-time half of the site package.

Everything a cell needs at run time to turn a stored site into a MuJoCo world, and nothing of the generator's
authoring half (layout, furnishing, deviations, ladder rules, proofs, maps): nothing under site_gen.loader imports
another site_gen module, and site_gen/__init__.py imports nothing.

    from site_gen.loader import open_site, compose, apply_truth_states, scene_option, lights_for_view, Site
    pkg = open_site("h01.tar.xz")        # or a generated site folder; checks every member against the manifest
    pkg.site, pkg.prior                   # site.json (ground truth), prior.json (the documents)
    spec, model, info = compose(pkg, robot_xml=..., start_pose=(x, y, yaw))
    truth = Site(pkg)                     # privileged queries: plan, modes_at, readable masks, link, dose

Package format: site_gen/loader/package.py.
"""

from .compose import DEFAULT_ROBOT_XML, compile_scene, compose, standin_robot  # noqa: F401
from .dose import dose_rate  # noqa: F401
from .package import (FORMAT, SHARED_TEXTURES, PackageError, SitePackage, open_site,  # noqa: F401
                      shared_dir_default, write_package)
from .site import Site, load_prior, load_site  # noqa: F401
from .states import apply_truth_states, instrument_records  # noqa: F401
from .travmap import MODES, Grid, TravMap  # noqa: F401
from .view import lights_for_view, scene_option  # noqa: F401
