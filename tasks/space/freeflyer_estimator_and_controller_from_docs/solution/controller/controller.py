# Generated file (the task's solution): do not edit; regenerate it from the maintained source.
"""The sensor-only reference controller of the cabin free-flyer in the form of a submission (Controller with
setup(session), plan(job, frame) and step(frame)). Its estimator and controller are the maintained reference under
refpkg/ (generated from the maintained source, logic unchanged), flown in its registered form.

It reads, beyond the frame, the session paper and the values any controller has, the session's registered reads
(the unit's effective response per signed axis and fan level, the noise levels of the inertial unit and of the fixes,
the mass, the per-session probability of a grossly wrong fix). The grader hands them with the set-up's session, under
the key 'registered_reads', only to a controller folder that holds the access token; without them this controller does
not start. Nothing else is handed to it. The zone file the session paper names (relative to the payload's root) is
read from the copy of the payload's zone files in this folder (cabin/zone_files)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "refpkg")
RS_DIR = os.path.join(REF, "reference", "rsense")
for _p in (RS_DIR, REF):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from world import config as _wcfg  # noqa: E402

_wcfg.configure(spec=os.path.join(REF, 'numbers.json'))
import rsense  # noqa: E402

if os.path.abspath(rsense.PKG_ROOT) != os.path.abspath(REF):
    raise RuntimeError("the reference resolved its modules outside its own folder")

READS_KEY = 'registered_reads'
REGISTERED_READS = ('effective_response', 'noise', 'mass', 'p_gross')


def _frame(fr):
    for s in fr.get("inertial") or []:
        s["rate"] = np.asarray(s["rate"], float)
        s["accel"] = np.asarray(s["accel"], float)
    for f in fr.get("fixes") or []:
        if f.get("position") is not None:
            f["position"] = np.asarray(f["position"], float)
        if f.get("attitude") is not None:
            f["attitude"] = np.asarray(f["attitude"], float)
    return fr


class Controller(object):
    def __init__(self):
        self.r = rsense.RSense()

    def setup(self, session):
        reads = session.get(READS_KEY) if isinstance(session, dict) else None
        if not isinstance(reads, dict) or tuple(sorted(reads)) != tuple(sorted(REGISTERED_READS)):
            raise RuntimeError("this controller needs the session's registered reads")
        paper = {k: v for k, v in session.items() if k != READS_KEY}
        return self.r.setup(paper, reads)

    def plan(self, job, frame):
        return self.r.plan(job, _frame(frame))

    def step(self, frame):
        return self.r.step(_frame(frame))
