"""voidscore: the score of one void-search episode and the practice ground's debrief (hidden side).

    scoring      S = F x sum(w_i c_i) from the cell's truth record (the rule: tests/README.md, Scoring)
    debrief      the practice ground's debrief (given/station_api.md, The practice ground): calibration site and practice sites; never a scored episode
    acceptance.yaml   the single source of every threshold

Relative imports only: the package works as tests/voidscore (the grader) and as voidcell.voidscore (the practice
ground's copy). Needs PyYAML.
"""

from . import scoring  # noqa: F401

__all__ = ["scoring"]
VERSION = "1.0.0-dev"
