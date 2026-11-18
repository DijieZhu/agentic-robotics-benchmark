"""Where the cell's code and data live, in the two layouts the mission package runs in.

source tree (the maintainers' single source of the library code)
    this package at mission/; bricks at bricks/; site_gen at site/site_gen/; the gait firmware at
    robot_rl/ (rl_controller.py, rl_support/, policy/); the shared acceptance file acceptance_v1.yaml;
    the robot model built by the maintainers' robot builder into a scratch folder.
vendored (tests/plantcell/_lib, written by the maintainers' assembly tool)
    a LAYOUT.json in _lib says where everything is, relative to _lib: the import roots ("." and "gait"), the
    acceptance file and the robot model in plantcell/data/, the mission config in mission/config/.

Every source-tree location can still be overridden by an environment variable (MISSION_*), so that the same
code runs from the source tree and from a copy. In the vendored layout the environment is ignored: the cell is what
the task folder holds and nothing else.

This file names no machine and no folder outside the code tree (it is vendored into the task, which is public). The
development tools' defaults that point at a build machine's scratch folder (the scratch root, the showcase site) live
in the data file `dev_paths.json` next to this file, which the source-tree layout reads when it exists and which the
assemble tool never vendors; without it (a copy, the task's review copy) those defaults are None. The robot model
defaults to the source tree's own model of the gait (the one the assemble tool vendors).
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_LAYOUT_FILE = os.path.join(os.path.dirname(HERE), "LAYOUT.json")
VENDORED = os.path.isfile(_LAYOUT_FILE)

if VENDORED:
    LIB_DIR = os.path.dirname(HERE)
    with open(_LAYOUT_FILE) as _f:
        LAYOUT = json.load(_f)

    def _lib(rel):
        return None if rel is None else os.path.normpath(os.path.join(LIB_DIR, rel))

    SKETCH_DIR = LIB_DIR
    IMPORT_ROOTS = [_lib(p) for p in LAYOUT["import_roots"]]
    ROBOT_XML = _lib(LAYOUT["robot_xml"])
    ACCEPTANCE_YAML = _lib(LAYOUT["acceptance"])
    MANUAL_TABLES_JSON = _lib(LAYOUT.get("manual_tables"))
    CONFIG_DIR = _lib(LAYOUT.get("config_dir", "mission/config"))
    SHARED_ASSETS = _lib(LAYOUT.get("shared_assets"))      # the textures every site package shares
    CAPABILITY_JSON = None                  # the generator's bands never ship with the cell
    SCRATCH = None
    DEFAULT_SITE = None
    RUNS_DIR = None
    DEFAULT_CONTROLLER = "rl"
else:
    LAYOUT = None
    LIB_DIR = None
    SKETCH_DIR = os.environ.get("MISSION_SKETCH_DIR", os.path.dirname(HERE))
    IMPORT_ROOTS = [SKETCH_DIR, os.path.join(SKETCH_DIR, "site"), os.path.join(SKETCH_DIR, "robot_rl")]
    _DEV = {}
    _dev_file = os.environ.get("MISSION_DEV_PATHS", os.path.join(HERE, "dev_paths.json"))
    if os.path.isfile(_dev_file):
        with open(_dev_file) as _f:
            _DEV = json.load(_f)
    SCRATCH = os.environ.get("MISSION_SCRATCH", _DEV.get("scratch"))
    ROBOT_XML = os.environ.get("MISSION_ROBOT_XML", _DEV.get("robot_xml")
                               or os.path.join(SKETCH_DIR, "robot", "model", "robot.xml"))
    # the showcase site (the development tools' default --site): the ten-slot ladder site 110 (2026-11-13)
    DEFAULT_SITE = os.environ.get("MISSION_SITE", _DEV.get("default_site"))
    RUNS_DIR = os.path.join(SCRATCH, "runs") if SCRATCH else None
    CONFIG_DIR = os.path.join(HERE, "config")
    CAPABILITY_JSON = os.path.join(SKETCH_DIR, "capability.json")
    ACCEPTANCE_YAML = os.environ.get("MISSION_ACCEPTANCE", os.path.join(SKETCH_DIR, "acceptance_v1.yaml"))
    # the gait's measured mode table; describe() quotes it when it exists
    MANUAL_TABLES_JSON = os.environ.get("MISSION_MANUAL_TABLES",
                                        os.path.join(SKETCH_DIR, "robot_rl", "manual_tables.json"))
    DEFAULT_CONTROLLER = os.environ.get("MISSION_CONTROLLER", "rl")  # the learned gait is the on-board gait
    SHARED_ASSETS = os.environ.get("MISSION_SHARED_ASSETS") or None   # None: site_gen.loader's own default

# the classical controller is a development-only comparison (a maintainers' tool registers it);
# the cell knows only the on-board gait firmware
ROBOT_DIR_DEV = None if VENDORED else os.path.join(SKETCH_DIR, "robot")


def setup(rl: bool = True) -> None:
    """Put the import roots on sys.path (idempotent). `rl` is kept for older callers (always on)."""
    for p in reversed(IMPORT_ROOTS):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
