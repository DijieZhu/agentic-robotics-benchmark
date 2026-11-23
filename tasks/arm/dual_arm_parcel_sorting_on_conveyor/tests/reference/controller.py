"""The reference program (privileged): ``run(server)`` is called by the cell's
``reference_run``. It runs the refctl package beside this file (or one folder
up) on the cell's own command path, with the station's duty: cartons whose
label is underneath or on the downstream face are turned; cartons with the
label on the upstream face or on a side face, and every mailer, pass
untouched."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if os.path.isdir(os.path.join(_p, "refctl")) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

from refctl import run as _run  # noqa: E402

# the duty option (the package's side-face methods stay in it, switched off)
DUTY = {"side": "pass", "minus_y": "pass", "mailers": "pass"}


def run(server):
    return _run(server, DUTY)
