"""sketch.venue: the seeded venue generator and the venue's exports.

    from sketch.venue import generate, build_spec, Track, FloorMap, top_view
    world = generate(seed_hex, grip_tier, layout_class)      # a world file (dict); passes check.check_world
    spec = build_spec(world, variant="track", pushable=True)  # floor, zones, markings, clutter, ducts (the duct model's add_ducts)
    top_view(world, "venue.png")                              # greybox top view
    export_all(world, out_dir)                                # world JSON, venue sheet, truth, digest, MJCF, PNG

`gen_venue.py` is the pipeline, `layout.py` the shape and widths, `walls.py` the walls and ducts, `export.py` the
model and the files, `checks.py` the gates' measures (drivability, statistics), `gates.py` the batch gates and the
candidate pool, `integration.py` the runs in the cell and with the reference program.
"""

from sketch.examples.check import FloorMap, Track

from .gen_venue import (GenerationError, PARAMS_PATH, dumps, generate, generate_with_info, load_params, params_digest,
                        seed_words, world_id, write_world)


def build_spec(world, variant="track", pushable=True, **kw):
    from .export import build_spec as _b
    return _b(world, variant=variant, pushable=pushable, **kw)


def top_view(world, path, **kw):
    from .export import top_view as _t
    return _t(world, path, **kw)


def export_all(world, out_dir, **kw):
    from .export import export_all as _e
    return _e(world, out_dir, **kw)


__all__ = ["generate", "generate_with_info", "build_spec", "Track", "FloorMap", "top_view", "export_all",
           "write_world", "dumps", "load_params", "params_digest", "seed_words", "world_id", "GenerationError",
           "PARAMS_PATH"]
