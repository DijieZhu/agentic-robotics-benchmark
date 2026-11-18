"""The schedule of the fourth session, the avoidance test.

    from sketch.others.schedule import avoid_schedule, check_schedule
    avoid_schedule(config)    # the timeline below, with the session's numbers taken from the grader configuration
    check_schedule(config)    # errors: the configuration disagrees with the design's schedule

The timeline (times from the opening of the session; the configuration's `sessions[avoid]` is the source of the
numbers, the cell owns it, the design fixes them):

    t = 0          model variant `avoid`: the ego car at the start pose, the three boxes at their
                   world-file poses (the BoxTracker's references), the slow cars staged hidden and without contact;
                   phase `static`
    t = 90 s       the referee sends the event `phase` {"name": "dynamic"} with the reply of the first tick that
                   starts at or after 90 s, and before that tick's physics calls SlowCars.enter(): the 1 or 2 slow
                   cars are put on their paths (entry rule in slowcars.py) at rest and start; the boxes stay
    t = 180 s      `session_end` {"reason": "time"}; the venue is over (the last session)

A lap counts for a phase only when it lies wholly inside it; a crash in either phase is
handled by the staff as in any session (displaced boxes are put back by BoxTracker.put_back).
"""

import math

DESIGN = {"session": "avoid", "duration_s": 180.0, "phases": [("static", 0.0), ("dynamic", 90.0)]}


def _avoid(config):
    for s in config["sessions"]:
        if s["name"] == "avoid":
            return s
    raise KeyError("the grader configuration has no session 'avoid'")


def avoid_schedule(config):
    """The avoidance session's timeline (see the module text) from the grader configuration."""
    ses = _avoid(config)
    tick = float(config["sim"]["tick_s"])
    phases = sorted(ses["phases"], key=lambda p: p["t0_s"])
    out = []
    for k, ph in enumerate(phases):
        t1 = phases[k + 1]["t0_s"] if k + 1 < len(phases) else ses["duration_s"]
        out.append({"name": ph["name"], "t0_s": float(ph["t0_s"]), "t1_s": float(t1),
                    "present": ["boxes"] + (["slow_cars"] if ph["name"] == "dynamic" else [])})
    dyn = next((p for p in out if p["name"] == "dynamic"), None)
    # the first tick n whose start n x tick is at or after t0 (the referee's test, with its 1e-9 tolerance)
    entry_tick = None if dyn is None else int(math.ceil(dyn["t0_s"] / tick - 1e-9))
    return {"session": "avoid", "duration_s": float(ses["duration_s"]), "desk_before_s": float(ses["desk_before_s"]),
            "phases": out, "slow_car_entry_t_s": None if dyn is None else dyn["t0_s"], "slow_car_entry_tick": entry_tick,
            "ticks": int(round(float(ses["duration_s"]) / tick))}


def check_schedule(config):
    """Errors of the configuration's avoidance session against the design's schedule (empty when they agree)."""
    err = []
    try:
        ses = _avoid(config)
    except KeyError as e:
        return [str(e)]
    if abs(float(ses["duration_s"]) - DESIGN["duration_s"]) > 1e-9:
        err.append("avoid lasts %s s, the design says %s" % (ses["duration_s"], DESIGN["duration_s"]))
    got = [(p["name"], float(p["t0_s"])) for p in sorted(ses["phases"], key=lambda p: p["t0_s"])]
    if got != DESIGN["phases"]:
        err.append("avoid phases %s, the design says %s" % (got, DESIGN["phases"]))
    for p in ses["phases"]:
        if p.get("speed_limit_mps") is not None:
            err.append("avoid phase %s has a speed limit; the test is at race speed" % p["name"])
    if not ses.get("scored", False):
        err.append("the avoidance session must be scored")
    names = [s["name"] for s in config["sessions"]]
    if names[-1:] != ["avoid"]:
        err.append("avoid must be the last session, the order is %s" % names)
    return err
