"""s7ref: the privileged reference program of racecar_head_to_head_race_on_unseen_track.

Maintainer material: it reads the hidden world through the token-unlocked calls and is never given to an entrant.
Standard library and numpy only; it drives only through `car.step`. The planner (`planner`, `qp`), the tracker
(`control`) and the plane geometry (`geometry`) carry their parameters in `params.json`; the session driver (`heats`)
and the racecraft among the opponents (`racecraft`) carry theirs in `race_params.json`.

    from s7ref import run, load_params, load_race_params
    stats = run(car, token)            # every session of one venue
"""

import json
import os

from .heats import RaceReference, field_from_world

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "params.json")
RACE_PARAMS_PATH = os.path.join(HERE, "race_params.json")


def load_params(path=PARAMS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_race_params(path=RACE_PARAMS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(car, token, params=None, race_params=None, roles=None, field_of=None, log=None, trace=None):
    """Drive one venue (every session in the order of describe()) and return the program's own statistics."""
    ref = RaceReference(car, params or load_params(), race_params or load_race_params(), token, roles=roles,
                        field_of=field_of, log=log)
    if trace is not None:
        ref.trace = trace
    return ref.run()


__all__ = ["RaceReference", "field_from_world", "run", "load_params", "load_race_params", "PARAMS_PATH",
           "RACE_PARAMS_PATH"]
