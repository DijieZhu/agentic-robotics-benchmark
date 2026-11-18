"""drive_protocol - the wire and the episode loop of a supervised driving episode.

Shared byte-identically by the grading stage and the practice grader: the scenario kinds,
the observation the supervisor gets, the reply it sends and the episode loop. The simulator
lives behind `sim_client.SimClient`; this module never imports it. How an episode is scored
is not part of it.

An episode: an ego vehicle drives a route in a CARLA town under a nominal planner (CARLA's
stock BehaviorAgent, 'normal' profile). Somewhere along the route a scenario is staged.
Every control period (0.1 s) the supervisor receives the front camera image, the ego speed,
the next navigation target and command, and the nominal planner's control, and either lets
the nominal control through or replaces it. The episode ends when the route is completed,
on a collision with a pedestrian, when the vehicle has been stationary for too long, when it
leaves the route, or at the time limit.

stdlib only, 3.8-compatible syntax.
"""

import math

__all__ = [
    "KINDS", "ProtocolError", "EpisodeOutcome", "agent_observation", "make_specs",
    "parse_action", "probe_observation", "run_episode",
]

#: Scenario kinds. Each one is generated procedurally from a seed in a town.
KINDS = (
    "construction_zone",      # cones and a barrier close the ego lane
    "broken_down_vehicle",    # a stopped vehicle with hazard lights blocks the ego lane
    "occluded_pedestrian",    # a pedestrian steps out from behind a parked vehicle
    "red_light_runner",       # a cross-traffic vehicle runs its red light at a junction
    "oncoming_left_turn",     # an oncoming vehicle while the route turns left at a junction
)


class ProtocolError(Exception):
    """The supervisor's reply is not a valid action. Aborts the episode."""


def parse_action(raw):
    """[mode, throttle, steer, brake]: mode 0 lets the nominal control through (the other
    three numbers are ignored), mode 1 applies them, clipped to throttle [0, 1],
    steer [-1, 1], brake [0, 1]. A bare 0 / [0] means pass-through."""
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    if len(values) not in (1, 4):
        raise ProtocolError("action must hold 1 or 4 numbers, got {}".format(len(values)))
    numbers = []
    for value in values:
        if isinstance(value, bool):
            raise ProtocolError("booleans are not numbers here")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ProtocolError("non-numeric action entry {!r}".format(value)[:120])
        if not math.isfinite(number):
            raise ProtocolError("non-finite action entry")
        numbers.append(number)
    mode = numbers[0]
    if mode not in (0.0, 1.0):
        raise ProtocolError("mode must be 0 or 1, got {!r}".format(values[0]))
    if mode == 1.0 and len(numbers) != 4:
        raise ProtocolError("mode 1 needs [1, throttle, steer, brake]")
    if mode == 0.0:
        return 0, [0.0, 0.0, 0.0]
    throttle, steer, brake = numbers[1:]
    return 1, [min(1.0, max(0.0, throttle)), min(1.0, max(-1.0, steer)),
               min(1.0, max(0.0, brake))]


def make_specs(rng, towns, per_kind, kinds=KINDS):
    """Episode specs: every kind `per_kind` times, towns cycled, each with a fresh 62-bit
    seed from `rng` (anything with random(); seeds built from it)."""
    specs = []
    index = 0
    for kind in kinds:
        for _ in range(per_kind):
            seed = int(rng.random() * (1 << 31)) * (1 << 31) + int(rng.random() * (1 << 31))
            specs.append({"town": towns[index % len(towns)], "kind": kind, "seed": seed})
            index += 1
    # interleave kinds by town so each town is loaded once
    specs.sort(key=lambda s: towns.index(s["town"]))
    return specs


def agent_observation(state, t):
    """What the supervisor sees: the front camera, its speed, where the route goes next and
    what the nominal planner wants to do. The scoring block (`truth`) never reaches the
    wire; the ground-truth block (`privileged`) does only when the episode's spec asked the
    bridge for it (a practice aid; graded runs do not give it to your supervisor)."""
    obs = {
        "t": int(t),
        "image": state["frame"]["image"],
        "speed": round(float(state["ego"]["speed"]), 3),
        "target": [round(float(v), 2) for v in state["navigation"]["target"]],
        "command": state["navigation"]["command"],
        "nominal": [round(float(v), 3) for v in state["nominal"]["control"]],
    }
    if "privileged" in state:
        obs["privileged"] = state["privileged"]
    return obs


def probe_observation(obs, state):
    """The probe's perturbed copy: the bridge pre-renders the same frame with a road user
    pasted into the front camera ahead of the car; the rest is unchanged, except that a
    ground-truth block lists the pasted road user too."""
    perturbed = dict(obs)
    perturbed["image"] = state["probe"]["image"]
    if "privileged" in obs and state["probe"].get("actor"):
        privileged = dict(obs["privileged"])
        privileged["actors"] = list(privileged["actors"]) + [dict(state["probe"]["actor"], id=-1)]
        perturbed["privileged"] = privileged
    return perturbed


class EpisodeOutcome(object):
    """The simulator's own record of one episode (`truth`), or an aborted episode."""

    def __init__(self, truth, aborted=False, note=""):
        self.truth = dict(truth or {})
        self.aborted = bool(aborted)
        self.note = str(note)

    def record(self):
        t = self.truth
        return {"aborted": self.aborted,
                "termination": t.get("termination", self.note),
                "route_completion": round(float(t.get("route_completion", 0.0)), 3),
                "collisions": t.get("collisions", {}), "red_lights": t.get("red_lights", 0),
                "harsh_brakes": t.get("harsh_brakes", 0),
                "mean_speed": t.get("mean_speed", 0.0),
                "overrides": t.get("overrides", 0), "steps": t.get("steps", 0)}


def run_episode(sim, spec, ask, probe=None, max_steps=3000):
    """Drive one episode. `ask(t, obs)` returns the raw action; `probe(t, obs, state)`
    returns None (not probed) or the REAL raw action after a probed double query. Wire
    failures and ProtocolError propagate (the caller charges an aborted episode)."""
    state = sim.reset(spec)
    t = 0
    overrides = 0
    while not state["truth"]["done"]:
        obs = agent_observation(state, t)
        raw = probe(t, obs, state) if probe is not None else None
        if raw is None:
            raw = ask(t, obs)
        mode, control = parse_action(raw)
        overrides += mode
        state = sim.step(mode, control)
        t += 1
        if t > max_steps:
            raise RuntimeError("episode did not terminate; simulator state machine broken")
    truth = dict(state["truth"])
    truth["overrides"] = overrides
    truth["steps"] = t
    return EpisodeOutcome(truth)
