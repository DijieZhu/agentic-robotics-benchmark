"""The privileged reference program of freeflyer_uncrewed_station_caretaking, run inside the cell by `reference_run`
(the oracle token only; tests/cabincell/server.py ``CabinCellServer.reference_run`` loads this file and calls run).

    run(server) -> {"summary": {...}, "notes": [...], "ops": [...], "reads": {...}, ...}

It drives the robot through the cell's own call boundary (``server.handle``, with the wire's JSON round trip: every
call is budgeted, logged and recorded like an agent's program's) and reads the five privileged quantities through ref_truth.Truth only (the true pose, the grasp features, 'in the gripper', the held item's pose,
'on the face' after a placement).  The program is ref_program.Program (its module doc has the order and the deposit
changes).  ``REFERENCE_PARAMS`` (environment, JSON) overrides the recipe's own values in tests (e.g.
{"place_method": "flyby"}); the verifier sets none.

The modules live next to this file (ref_*.py, names that the cell's packages do not use); importing them writes no
bytecode.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _params():
    raw = os.environ.get("REFERENCE_PARAMS")
    if not raw:
        return None
    p = json.loads(raw)
    if not isinstance(p, dict):
        raise ValueError("REFERENCE_PARAMS must be a JSON object")
    return p


def run(server, params=None):
    sys.dont_write_bytecode = True
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import ref_program as RP
    out = RP.run_reference(server, params if params is not None else _params())
    c = server.cell
    notes = out.get("notes", [])
    summary = dict(sim_time_s=round(float(c.t_mission()), 2), calls=out.get("calls"), notes=len(notes),
                   reported=any(n.get("kind") == "report" and n.get("code") == "RECORDED" for n in notes),
                   docked=any(n.get("kind") == "dock" and n.get("code") == "DOCKED" for n in notes),
                   aborted=any(n.get("kind") in ("abort", "cell_error") for n in notes),
                   reads=out.get("reads"), refusals=out.get("refusals"))
    out["summary"] = summary
    return out
