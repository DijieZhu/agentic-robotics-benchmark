"""Compatibility layer: the first compose / load_site / load_prior, now served by site_gen.loader (the run-time
half). New code imports site_gen.loader directly.

    from site_gen.compose import compose, load_site, load_prior
    spec, model, info = compose(site_dir_or_archive)       # robot on the pad, final options, compiled
    site = load_site(site_dir_or_archive)                  # truth queries (privileged)

Geom groups: 0 structure / equipment (collision), 1 decor, 2 ceiling and overhead decor, 3 traversal elements
(collision). Render with scene_option() so the traversal elements are drawn.
"""

from __future__ import annotations

from .loader.compose import DEFAULT_ROBOT_XML, compose  # noqa: F401
from .loader.compose import standin_robot as _standin
from .loader.site import Site, load_prior, load_site  # noqa: F401
from .loader.travmap import MODES  # noqa: F401
from .loader.view import lights_for_view, scene_option  # noqa: F401


def standin_robot(cfg=None):
    """A free box of the robot's footprint (3 kg), for tests without the robot's MJCF."""
    fp = None
    if cfg is not None:
        try:
            fp = tuple(cfg["robot"]["footprint"])
        except (KeyError, TypeError):
            fp = None
    return _standin(fp)
