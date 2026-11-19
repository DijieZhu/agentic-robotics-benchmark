"""sketch.others: what exists only in the avoidance test, boxes and slow cars.

    from sketch.others import add_boxes, add_slow_cars, BoxTracker, SlowCars      # the contract
    from sketch.others import place_avoidance, with_avoidance                       # the placement rule (for the venue generator)
    from sketch.others import avoid_schedule, check_schedule, load_params

`others_params.json` holds the parameters; `gates.py` the gates.
"""

from .boxes import BoxTracker, add_boxes, shell_inertia
from .params import PARAMS_PATH, load_params, params_digest
from .placement import PlacementError, place_avoidance, with_avoidance
from .schedule import avoid_schedule, check_schedule
from .slowcars import SlowCars, add_slow_cars, mocap_car_xml

__all__ = ["add_boxes", "add_slow_cars", "BoxTracker", "SlowCars", "place_avoidance", "with_avoidance", "PlacementError",
           "avoid_schedule", "check_schedule", "load_params", "params_digest", "PARAMS_PATH", "shell_inertia",
           "mocap_car_xml"]
