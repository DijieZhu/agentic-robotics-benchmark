"""refctl: the privileged reference controller of the dual-arm label-facing
station: one integrated controller (the end-over-end roll, the continued
half turn and the side-face methods behind one shift scheduler).

Entry point for the cell's privileged ``reference_run``:

    from refctl import run
    run(server)          # server = parcelcell.server.ParcelCellServer

It drives ``server.world`` (the per-shift world) through the simulator-level
API (``world.step(q14, grip2)``), so every tick passes the cell's command
gate (protective stops), its command speed limit, the safety watch and the
shift's record. It reads the truth (parcel poses, sizes, masses, kinds,
label normals, the belt speed) and writes only the 14 joint position targets
and the 2 gripper commands of each tick.
"""

from __future__ import annotations

__all__ = ["run"]


def run(server, options=None):
    """Run the whole shift; returns {"summary": {...}} for the cell's record.
    Options: ``server.refctl_options`` (a dict; see ``sched.DEFAULTS``), then
    ``options``."""
    from .sched import ShiftController
    opts = dict(getattr(server, "refctl_options", None) or {})
    opts.update(options or {})
    ctl = ShiftController(server.world, server=server, options=opts)
    ctl.run()
    return {"summary": ctl.summary()}
