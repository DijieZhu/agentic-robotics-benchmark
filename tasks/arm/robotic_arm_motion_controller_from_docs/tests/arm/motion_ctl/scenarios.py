"""Fixed hidden acceptance scenarios (`motion-scenario/1`): one unit and its three cells with their generated jobs.

The task's hidden set (a task with random content gets a fixed hidden set that the oracle completes) is drawn once
by the maintainers' pre-registered selection (seeds
from os.urandom, the rule written down before anything runs, each candidate run once with the oracle, the first passes
kept) and kept in the task's `tests/hidden/scenarios/` (never agent-visible). At verify the grader runs these instead of
drawing new ones, so verification needs no job generation and its score is reproducible.

    save(path, seed, spec, cells)       cells = {kind: (cell, gen)}; writes the pickle with the code digest
    load(path)                          -> {"format", "seed", "spec", "cells": {kind: {"cell", "gen"}}, "code"}
    code_digest()                       sha256 of the modules a scenario's jobs depend on

A scenario is valid only for the code it was generated with (the plans and the true-view checks are the generator's):
`load` raises ScenarioStale when the digest differs (regenerate the set after any change of those modules).
"""

import hashlib
import os
import pickle

FORMAT = "motion-scenario/1"
CODE_FILES = ("cells.py", "unit.py", "kin.py", "jobs.py", "plant.py", "privileged.py")


class ScenarioStale(RuntimeError):
    pass


def code_digest():
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for fn in CODE_FILES:
        with open(os.path.join(here, fn), "rb") as f:
            h.update(fn.encode() + b"\0" + f.read())
    return h.hexdigest()


def save(path, seed, spec, cells):
    data = {"format": FORMAT, "seed": int(seed), "spec": spec, "code": code_digest(),
            "cells": {k: {"cell": c, "gen": g} for k, (c, g) in cells.items()}}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=4)
    os.replace(tmp, path)
    return data


def load(path, check_code=True):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if data.get("format") != FORMAT:
        raise ValueError("{}: not a {} file".format(path, FORMAT))
    if check_code and data.get("code") != code_digest():
        raise ScenarioStale("{} was generated with other code (digest {}..., now {}...)".format(
            path, str(data.get("code"))[:12], code_digest()[:12]))
    return data
