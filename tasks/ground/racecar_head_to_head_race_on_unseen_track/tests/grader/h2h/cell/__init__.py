"""The heat cell of racecar_head_to_head_race_on_unseen_track: the day, the heats, the referee of four cars.

    from h2h.cell import HeatVenue, load_config, check_result
    venue = HeatVenue(world, load_config(), providers=prov, priv_token_sha256=sha)

`config.py` reads and checks the grader configuration (`grader_config.json`, schema in `records.schema.json`);
`heat.py` is the cell; `rules.py` the heat's rules as state machines; `replay.py` the copy of a heat without the car
that the push of a contact bout is measured against; `staff.py` where the staff set a stopped car down; `record.py`
the checker of the venue result.
"""

from .config import check_config, heat_sessions, load_config
from .heat import HeatVenue
from .record import check_result, heat_numbers

__all__ = ["HeatVenue", "load_config", "check_config", "heat_sessions", "check_result", "heat_numbers"]
