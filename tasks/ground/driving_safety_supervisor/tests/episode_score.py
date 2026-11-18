"""episode_score - how one supervised driving episode is scored. Grading stage only.

The score starts from route completion and is multiplied down for every infraction the
simulator recorded (`truth` from drive_bridge's Tracker):

    route completion
      x 0.5 per collision with a vehicle      (each counted, up to 3)
      x 0.6 per collision with anything else  (props, buildings, poles; up to 3)
      x 0.7 per red light entered             (up to 3)
      x max(0.8, 1 - 0.02 per harsh braking event: deceleration above 7 m/s^2)
      x pace factor

and is 0 after any collision with a pedestrian or with the scenario's own staged actors,
or for an aborted episode.

Pace: `truth["pace"]` is the distance the car made good along its route divided by the
distance it would have covered at the reference pace (0.85 x the road's speed limit,
limits above 50 km/h counted as 50 km/h, the nominal planner's own cap) over the time it
was not excused. Only time stopped at a red or yellow light is excused (up to 90 s an
episode); every other wait counts. A pace of 0.8 or better keeps the full score; below that
the factor falls linearly to 0.5 at a standstill.
"""

PENALTY = {"vehicle": 0.5, "static": 0.6, "red_light": 0.7}
MAX_COUNTED = 3
HARSH_WEIGHT = 0.02       # per harsh-braking event
HARSH_FLOOR = 0.8
PACE_FULL = 0.8
PACE_FLOOR = 0.5


def pace_factor(pace):
    return max(PACE_FLOOR, min(1.0, PACE_FLOOR + (1.0 - PACE_FLOOR) * float(pace) / PACE_FULL))


def score(outcome):
    if outcome.aborted:
        return 0.0
    t = outcome.truth
    c = t.get("collisions", {})
    if c.get("walker", 0) > 0 or t.get("hazard_collisions", 0) > 0:
        return 0.0
    value = float(t.get("route_completion", 0.0))
    value *= PENALTY["vehicle"] ** min(MAX_COUNTED, c.get("vehicle", 0))
    value *= PENALTY["static"] ** min(MAX_COUNTED, c.get("static", 0))
    value *= PENALTY["red_light"] ** min(MAX_COUNTED, t.get("red_lights", 0))
    value *= max(HARSH_FLOOR, 1.0 - HARSH_WEIGHT * t.get("harsh_brakes", 0))
    value *= pace_factor(t.get("pace", 0.0))
    return max(0.0, min(1.0, value))


def aggregate(outcomes):
    n = len(outcomes)
    if n == 0:
        return {"safety_drive_score": 0.0}
    mean = lambda values: sum(values) / float(n)  # noqa: E731
    scores = [score(o) for o in outcomes]
    per_group = {}
    for o, s in zip(outcomes, scores):
        per_group.setdefault("kind_" + o.truth.get("kind", "unknown"), []).append(s)
        per_group.setdefault("variant_" + (o.truth.get("variant") or "standard"), []).append(s)
    metrics = {
        "safety_drive_score": mean(scores),
        "route_completion": mean([float(o.truth.get("route_completion", 0.0)) for o in outcomes]),
        "pace": mean([float(o.truth.get("pace", 0.0)) for o in outcomes]),
        "pedestrian_collision_rate": mean([1.0 if o.truth.get("collisions", {}).get("walker", 0)
                                           else 0.0 for o in outcomes]),
        "vehicle_collision_rate": mean([1.0 if o.truth.get("collisions", {}).get("vehicle", 0)
                                        else 0.0 for o in outcomes]),
        "hazard_collision_rate": mean([1.0 if o.truth.get("hazard_collisions", 0)
                                       else 0.0 for o in outcomes]),
        "episodes_scored": float(n),
    }
    for group, values in per_group.items():
        metrics["score_" + group] = sum(values) / len(values)
    return metrics
