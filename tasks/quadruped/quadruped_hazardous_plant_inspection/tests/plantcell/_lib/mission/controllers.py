"""The on-board gait: the learned controller (robot_rl in the source tree, the robot's firmware in the task).

It exposes reset / set_command / set_head / set_body_pitch / step / get_state / truth. Its get_state() carries a
'status' key ('ok', 'blocked', 'protective_sit', 'fallen') and an 'events' list: it implements the protective
behaviours itself and the world passes its events on. A controller without them (the classical one,
a development-only comparison that a maintainers' tool registers here) gets the world's stand-in protections
(world.MissionWorld._protections).

The cell itself knows only the firmware: the classical controller is not part of the task.
"""

from __future__ import annotations

from . import paths

FACTORIES = {}


def register(kind: str, factory) -> None:
    """Make another controller selectable (development tools only): factory(model, data, prefix, seed) -> controller."""
    FACTORIES[str(kind).lower()] = factory


def kinds():
    return ("rl",) + tuple(sorted(k for k in FACTORIES if k != "rl"))


def make_controller(kind: str | None, model, data, prefix: str = "robot/", seed: int = 0):
    kind = (kind or paths.DEFAULT_CONTROLLER).lower()
    if kind == "rl":
        paths.setup()
        from rl_controller import RLQuadrupedController  # noqa: WPS433 (the gait firmware)
        return RLQuadrupedController(model, data, prefix=prefix, seed=seed)
    if kind in FACTORIES:
        return FACTORIES[kind](model, data, prefix, seed)
    raise ValueError("unknown controller %r (choose from %s)" % (kind, ", ".join(kinds())))


def has_own_protections(ctl) -> bool:
    try:
        return "status" in ctl.get_state()
    except Exception:  # noqa: BLE001
        return False
