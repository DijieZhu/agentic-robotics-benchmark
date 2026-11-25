"""voidcell: the simulated collapsed-warehouse site and the three-strut rolling robot of
tensegrity_collapsed_warehouse_void_search (the grader's cell; agents get only voidcell.client).

Importing the package puts the cell's vendored library on sys.path (_lib/: the frozen copy of the simulation base
`ysim` with its run-time data tables under _lib/notes/sim_base_data/ and its per-file lock _lib/ysim.lock.json; the
cell core `cellcore` (site loading, set-down, primitive executor, truth); the cell's own modules camera, telemetry,
budgets, mission, pose_source, protocol). The vendored base is the only `ysim` the cell may import: YSIM_PARENT is set
to _lib before anything imports cellcore, and the import is checked (`check_vendored`). The robot's visual meshes (a
decimated copy of the authors' strut mesh) are in data/meshes/ (MESHES.json); they never reach the agent.
"""

import os
import sys

CELL_VERSION = "0.1.0-dev"
__version__ = CELL_VERSION

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(HERE, "_lib")
DATA_DIR = os.path.join(HERE, "data")
MESH_DIR = os.path.join(DATA_DIR, "meshes")


def _setup():
    os.environ["YSIM_PARENT"] = LIB_DIR            # cellcore.world reads it at import: the vendored base only
    os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")
    if LIB_DIR not in sys.path:
        sys.path.insert(0, LIB_DIR)


def check_vendored():
    """Raise when the imported ysim / cellcore are not the vendored copies under _lib/."""
    import cellcore
    import ysim
    for mod in (ysim, cellcore):
        p = os.path.realpath(os.path.dirname(mod.__file__))
        if not p.startswith(os.path.realpath(LIB_DIR) + os.sep):
            raise RuntimeError("voidcell: %s imported from %s, not from the vendored _lib/" % (mod.__name__, p))
    return ysim.__version__


_setup()
