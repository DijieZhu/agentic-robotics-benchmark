"""Where ysim finds its inputs.

* Small data tables (parameters, measured strut geometry, the data copied from the authors' MJCF, decoded authors'
  tables, collision hull, validation results) live next to the package, under ``notes/sim_base_data/``.
  The physics model (``ModelConfig(visual=False)``) is built from these tables alone (ysim 0.3.0; checked by a
  test of the base that is not shipped).
* Large derived meshes (visual only) live outside the package in an asset directory: ``$YSIM_ASSETS`` if set, else
  the default ``SCRATCH/work/ysim_assets``. They are produced by
  ``ysim.assets.build`` from the authors' strut mesh and checked against the hashes recorded in
  ``strut_geometry.json``.
* The authors' own source files (untrusted downloads) are read only by the development tools (``assets.build``)
  and by tests that re-check the copied data against them: ``$YSIM_SOURCES`` if set, else ``SCRATCH/src``. Nothing in
  the run-time path reads them.
"""
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
YALE_DIR = os.path.dirname(PKG_DIR)
DATA_DIR = os.path.join(YALE_DIR, "notes", "sim_base_data")
SCRATCH = "/tmp/ysim"
DEFAULT_ASSETS = os.path.join(SCRATCH, "work", "ysim_assets")
SOURCES = os.environ.get("YSIM_SOURCES", os.path.join(SCRATCH, "src"))
AUTHORS_PLY = os.path.join(SOURCES, "gh_shiyoung77_tensegrity_perception", "ros_branch_mesh",
                           "Will_strut_scaled_centered.ply")
AUTHORS_PLY_SHA256 = "8dc4caff1efa26468660f30d5e2d91f7d927d6b287480ecd3b06b351f88a4614"
# the authors' MJCF whose node and site coordinates are copied into authors_model.json (dev / test use only)
AUTHORS_MJCF = os.path.join(SOURCES, "gdrive_crgs", "extracted", "crgs_3_bar_dataset", "dataset_0",
                            "3bar_new_platform_all_cables.xml")


def assets_dir(override=None):
    """Asset directory: explicit override > $YSIM_ASSETS > scratch default."""
    if override:
        return override
    return os.environ.get("YSIM_ASSETS", DEFAULT_ASSETS)


def data_path(name):
    return os.path.join(DATA_DIR, name)


def tmp_dir():
    """Scratch directory for library caches (matplotlib config): $TMPDIR if set, else SCRATCH/work/tmp."""
    d = os.environ.get("TMPDIR") or os.path.join(SCRATCH, "work", "tmp")
    os.makedirs(d, exist_ok=True)
    return d
