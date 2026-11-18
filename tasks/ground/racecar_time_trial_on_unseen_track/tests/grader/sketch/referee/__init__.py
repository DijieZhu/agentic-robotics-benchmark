"""sketch.referee: the cell of one venue (sessions, laps, staff, events, budgets).

    from sketch.referee import Venue, load_config
    venue = Venue(world, load_config())      # the contract: describe / open_session / step / close_session /
                                             # request_pickup return the wire's result or {"error", "detail"}
"""

from .config import car_table, config_sha256, describe_result, load_config, load_params, program_caps, session_table
from .providers import Providers, build_model, resolve
from .venue import Venue

__all__ = ["Venue", "load_config", "load_params", "config_sha256", "describe_result", "session_table", "car_table",
           "program_caps", "Providers", "resolve", "build_model"]
