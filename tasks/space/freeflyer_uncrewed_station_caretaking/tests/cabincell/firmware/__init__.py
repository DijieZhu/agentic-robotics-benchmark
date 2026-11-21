"""Task firmware of the caretaking cell (hidden side only).

The robot calls of the robot API on top of the simulation package's firmware layer (spacekit 0.4.0
``primitives.FirmwareLayer`` with the arm-deployed zone inflation, the action-scoped fiducial estimate, the holding
interlock and the band rule all on).  Modules:

* ``spec``          numbers by key: the runtime site_spec (sha256-checked) and the reference robot's values; the berthing-cone setting
* ``zonecheck``     zone rules: fly_rel's approach-zone exemption, the arm envelope check, approach zones
* ``bars``          handles and crew handrails; the geometric bar detector of the aft depth camera
* ``capture``       capture(): gate, lock rule, NO_BAR / LOST_BAR / STALLED / TIMEOUT, palm_dir
* ``localisation``  lighting richness, the plates' visibility (occlusion) gate, the gated fiducial and rail modes; the
                    robot's occluder of every camera-visibility test (the maker's mesh as the frames render it)
* ``power``         battery on the 2 h basis with the lamps; status() telemetry sampled as the sensors
* ``firmware``      TaskFirmware: the calls and their codes
* ``stack``         the robot side of the cell's wiring (build_stack, stack_for_site)

The cell (tests/cabincell) imports this package with tests/cabincell/_lib first on sys.path.
"""
VERSION = "1.1.0"          # 1.1.0: the robot occluder of the camera-visibility tests is the maker's mesh

from .firmware import Clock, TaskFirmware  # noqa: E402,F401
from .spec import FirmwareSpec, default_spec  # noqa: E402,F401
from .stack import build_stack, stack_for_site  # noqa: E402,F401
