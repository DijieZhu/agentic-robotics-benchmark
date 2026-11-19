"""Drive model of the owner's car.

    from sketch.car import add_car, CarRuntime
    add_car(spec, prefix="car/", pose=(x, y, yaw), tail_box=False)      # into an MjSpec that holds the floor
    car = CarRuntime(model, data, prefix="car/", grip_at=floor.grip_at)

One parameter set (`drive_params.json`); two tyre models selected by its key `tyre_model` (`cone`, `slip_angle`).
"""

from .build_car import TYRE_MODELS, add_car, add_floor_pairs, apply_options, car_xml, standalone_xml
from .params import PARAMS_PATH, load_params, params_digest
from .runtime import CarRuntime

__all__ = ["add_car", "add_floor_pairs", "apply_options", "car_xml", "standalone_xml", "CarRuntime", "load_params",
           "params_digest", "PARAMS_PATH", "TYRE_MODELS"]
