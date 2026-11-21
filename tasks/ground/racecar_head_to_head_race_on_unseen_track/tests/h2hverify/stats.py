"""What the verdict reports beside the score: the counts of a venue's record, by session (none of them is scored
other than through the rules the cell applied).

    venue_metrics(result) -> {name: number}     # finite numbers only, names like "practice_touches", "heat1_C"
    describe(result) -> str                     # one log line

Per session: touches, crashes, pick-ups, warnings, out-of-bounds, laps, the largest wall displacement, program and
desk time. Per heat, from its block: the lap target, the car's laps, progress, warnings `K`, track crashes, car
crashes, pick-ups on request and for standing still, placements without a place clear of the other cars' bands,
`classified`, the clean overtake balance `C`, the number of other cars beaten, the session time of the end. For the
venue: how far walls were pushed in all, calls outside the interface and malformed calls, program time.
"""

import math

__all__ = ["venue_metrics", "describe"]

SESSION_KEYS = ("touches", "crashes", "pickups", "warnings", "out_of_bounds", "max_displacement_m", "program_s", "desk_s")
HEAT_KEYS = ("laps_target", "end_t_s", "C", "beaten", "K")
CAR_KEYS = ("laps", "progress", "warnings", "crashes_track", "crashes_car", "pickups_requested", "pickups_standing",
            "placement_fallbacks", "classified")
STATS_KEYS = ("touches", "crashes", "warnings", "wall_displacement_m", "forbidden_calls", "bad_calls", "program_s")


def _put(out, key, v):
    if isinstance(v, bool):
        out[key] = 1.0 if v else 0.0
    elif isinstance(v, (int, float)) and math.isfinite(float(v)):
        out[key] = float(v)


def venue_metrics(result):
    out = {}
    for s in result.get("sessions") or []:
        name = s.get("name")
        if not isinstance(name, str):
            continue
        for k in SESSION_KEYS:
            _put(out, "%s_%s" % (name, k), s.get(k))
        _put(out, "%s_laps" % name, len(s.get("laps") or []))
        heat = s.get("heat")
        if isinstance(heat, dict):
            for k in HEAT_KEYS:
                _put(out, "%s_%s" % (name, k), heat.get(k))
            car = heat.get("car") or {}
            for k in CAR_KEYS:
                _put(out, "%s_car_%s" % (name, k), car.get(k))
            _put(out, "%s_contacts" % name, len(heat.get("contacts") or []))
            _put(out, "%s_overtake_events" % name, len(heat.get("overtakes") or []))
    for k in STATS_KEYS:
        _put(out, "venue_" + k, (result.get("stats") or {}).get(k))
    return out


def describe(result):
    parts = ["end %s" % result.get("end_reason")]
    for s in result.get("sessions") or []:
        h = s.get("heat")
        if isinstance(h, dict):
            car = h.get("car") or {}
            parts.append("%s: end %s laps %s/%s C %s beaten %s K %s classified %s" % (
                s.get("name"), s.get("end"), car.get("laps"), h.get("laps_target"), h.get("C"), h.get("beaten"),
                h.get("K"), car.get("classified")))
        else:
            parts.append("%s: end %s laps %d" % (s.get("name"), s.get("end"), len(s.get("laps") or [])))
    st = result.get("stats") or {}
    parts.append("touches %s crashes %s" % (st.get("touches"), st.get("crashes")))
    return "; ".join(parts)
