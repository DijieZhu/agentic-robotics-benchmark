"""s7ref: the privileged reference program of `racecar_time_trial_on_unseen_track`.

Maintainer material: it reads the hidden world through the token-unlocked calls and is never given to an agent.
Standard library and numpy only; it drives only through `car.step`.

    from s7ref import run, load_params
    stats = run(car, token)            # all four sessions of one venue
"""

import json
import os

from .reference import Reference

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "params.json")


def load_params(path=PARAMS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(car, token, params=None, log=None):
    """Drive one venue (practice, tt1, tt2, avoid) and return the program's own statistics."""
    ref = Reference(car, params or load_params(), token, log=log)
    return ref.run()


__all__ = ["Reference", "run", "load_params", "PARAMS_PATH"]
