"""The privileged reference program of quadruped_hazardous_plant_inspection, run inside the cell by `reference_run`
(the oracle's token only).

    run(server) -> {"summary": {...}, "log": [...], "crossings": [...]}

It drives the robot through the same client proxy and the same calls an agent's program uses (plantcell.client on
the cell's own `handle`, with the wire's JSON round trip, so every frame it reads went through the wire's encoding)
and adds the privileged Truth (mission.truth: the true pose, the true layout, the traversability map, each
instrument's pose and acceptance region, the unlisted source's position for planning the survey). It never gets a
value: every reported reading comes from its own inspect or thermal frames through the reference readers, the
survey mark's rate from its own dosimeter counts, the source's position from a fit of its own counts
(mission/reference.py).

The modules the reference needs beyond the cell's own (its navigation, the reference readers, Truth) are vendored by
the maintainers' assembly tool into _lib/ here, in their packages' layout; `_extend_packages` adds them to
the cell's already imported packages (mission, bricks), so nothing of them is in the cell's own folder (and none in
the practice ground's copy of the cell).
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "_lib")


def _extend_packages():
    """Make the reference's vendored modules importable inside the cell's packages."""
    if not os.path.isdir(LIB):
        return
    for name in sorted(os.listdir(LIB)):
        d = os.path.join(LIB, name)
        if not os.path.isdir(d):
            continue
        try:
            pkg = importlib.import_module(name)
        except ImportError:
            if LIB not in sys.path:
                sys.path.append(LIB)
            continue
        if d not in list(getattr(pkg, "__path__", [])):
            pkg.__path__.append(d)
        for dp, dns, fns in os.walk(d):
            for sub in dns:
                rel = os.path.relpath(os.path.join(dp, sub), LIB).replace(os.sep, ".")
                try:
                    sp = importlib.import_module(rel)
                except ImportError:
                    continue
                full = os.path.join(dp, sub)
                if full not in list(getattr(sp, "__path__", [])):
                    sp.__path__.append(full)


def run(server):
    _extend_packages()
    from plantcell.client import connect_handler
    from mission.reference import Reference
    from mission.truth import Truth
    robot = connect_handler(server.handle)
    ref = Reference(robot, Truth(server.world), out_dir=None)
    ref.run()
    w = server.world
    summary = {"sim_time_s": round(float(w.time), 2), "crossings": len(ref.crossings),
               "crossings_blocked": int(sum(int(c.get("blocked", 0)) for c in ref.crossings)),
               "reports": len(w.reports), "notes": len(ref.log),
               "halted": any(n.get("kind") == "halt" for n in ref.log),
               # version 2: every plate read, and the instruments whose plate was read as their tag
               "plate_reads": len(getattr(ref, "plate_reads", [])),
               "plates_read_as_tag": sorted(getattr(ref, "plate_ok", {})),
               "blocked_holds": int(getattr(w, "n_holds", 0))}
    # the whole log (version 2: admission row 2 checks in it that every instrument's plate was read as its tag before
    # its report; the verify stage keeps only the summary)
    return {"summary": summary, "log": ref.log, "crossings": ref.crossings,
            "plate_reads": list(getattr(ref, "plate_reads", []))}
