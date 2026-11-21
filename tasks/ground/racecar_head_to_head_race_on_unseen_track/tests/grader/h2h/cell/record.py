"""The venue result of the head-to-head day: its checker and the numbers a heat is scored on.

    check_result(result, config) -> [error, ...]     # schema, laps, and every heat's numbers against its lists
    heat_numbers(block, config) -> {"C", "beaten", "K", "classified"}   # recomputed from the lists of a heat block

A heat block carries `C`, `beaten`, `K` and the car's `classified`; `check_result` recomputes each from the block's
lists (the overtake events, the field cars' progress, the stops) and the configuration, and reports any difference.
The counts of the car and of the session follow the stops (a pick-up's stop that became the stop of a crash or a car
crash counts as that), and a stop that is still a pick-up has no car crash at the car's fault without a warning that
began from its start to its set-down (`staff.relocate_at_s` later): a pick-up takes no warning away.

Of every contact bout it checks the `replay` block against the configuration: the heat's and the copy's samples
start with the same pair, neither list is longer than the window's samples plus one, a list shorter than that says
why (`cut`, `alone_cut`) and a full one does not, the push recomputed from the two lists equals `push_m`, and `class`
is `crash` exactly when `push_m` reaches `heat.car_crash_push_m` or the field car's staff reset was triggered. It does
not check that `alone` is what the simulation gives without the car: that needs the simulation, and the cell's own
tests prove it (a copy of a heat that the car does not influence repeats the heat exactly).
"""

import json

from sketch.examples import check

from . import config as HC
from . import rules as RP

TRACK_CRASHES = ("displacement", "open_wall", "rollover", "out_of_bounds")
WALL_KEYS = ("program_s", "desk_s")
EPS = 1e-9
TIME_TOL = 1e-6                 # times in the record are rounded to 1e-6 s; physics steps are 2.5 ms apart


def contact_errors(name, b, config):
    """Errors of one contact bout's replay block, push and class against the configuration."""
    hc = config["heat"]
    where = "%s: the contact with %s at %s s" % (name, b.get("car"), b.get("t0_s"))
    rp = b.get("replay")
    if not isinstance(rp, dict):
        return [where + " has no replay block"]
    err = []
    if abs(rp["sample_s"] - hc["car_crash_sample_s"]) > EPS or abs(rp["window_s"] - hc["car_crash_window_s"]) > EPS:
        err.append(where + ": its sample period or window is not the configuration's")
    n = int(round(hc["car_crash_window_s"] / hc["car_crash_sample_s"])) + 1
    a, r = rp["actual"], rp["alone"]
    if len(a) > n or len(r) > n:
        err.append(where + ": more samples than the window holds")
    if (rp["cut"] is None) != (len(a) == n):
        err.append(where + ": the heat's samples are %d, cut is %r" % (len(a), rp["cut"]))
    if (rp["alone_cut"] is None) != (len(r) == n):
        err.append(where + ": the copy's samples are %d, alone_cut is %r" % (len(r), rp["alone_cut"]))
    try:
        push = RP.push_of(a, r)
    except ValueError as e:
        return err + ["%s: %s" % (where, e)]
    if round(push, RP.ROUND) != round(float(b["push_m"]), RP.ROUND):
        err.append(where + ": push_m is %s, the samples give %s" % (b["push_m"], push))
    crash = float(b["push_m"]) >= float(hc["car_crash_push_m"]) - EPS or bool(b["reset"])
    if (b["class"] == "crash") != crash:
        err.append(where + ": class is %s with push %s and reset %s" % (b["class"], b["push_m"], b["reset"]))
    return err


def heat_numbers(block, config):
    """C, beaten, K and classified of a heat block, recomputed from its lists."""
    car_p = block["car"]["progress"]
    C, beaten = 0, 0
    per = {}
    for f in block["field"]:
        evs = [e for e in block["overtakes"] if e["car"] == f["id"]]
        clean = sum(1 for e in evs if e["kind"] == "overtake" and e["clean"] is True)
        contact = sum(1 for e in evs if e["kind"] == "overtake" and e["clean"] is not True)
        losses = sum(1 for e in evs if e["kind"] == "loss")
        bal = clean - losses
        bt = bal >= 1 and f["progress"] < car_p
        per[f["id"]] = {"clean_overtakes": clean, "contact_overtakes": contact, "losses": losses, "balance": bal, "beaten": bt}
        C += bal
        beaten += int(bt)
    K_ = sum(1 for s in block["stops"] if s["warned"])
    N = block["laps_target"]
    short = int(config["heat"]["classified_laps_short"])
    end_at = int(config["staff"]["warnings_end_session"])
    classified = None if N is None else bool(block["car"]["laps"] >= N - short and K_ < end_at)
    return {"C": C, "beaten": beaten, "K": K_, "classified": classified, "field": per}


def _heat_errors(name, ses, block, config):
    err = []
    want = heat_numbers(block, config)
    for k in ("C", "beaten", "K"):
        if block[k] != want[k]:
            err.append("%s: %s is %s, the lists give %s" % (name, k, block[k], want[k]))
    if block["car"]["classified"] != want["classified"]:
        err.append("%s: classified is %s, the lists give %s" % (name, block["car"]["classified"], want["classified"]))
    if block["car"]["warnings"] != block["K"] or ses["warnings"] != block["K"]:
        err.append("%s: the car's warnings, the session's warnings and K differ" % name)
    for f in block["field"]:
        w = want["field"][f["id"]]
        for k, v in w.items():
            if f[k] != v:
                err.append("%s: field car %s %s is %s, the lists give %s" % (name, f["id"], k, f[k], v))
    kinds = [s["kind"] for s in block["stops"]]
    c = block["car"]
    counts = {"crashes_track": sum(1 for k in kinds if k in TRACK_CRASHES), "crashes_car": kinds.count("car_crash"),
              "pickups_requested": kinds.count("pickup"), "pickups_standing": kinds.count("standing"),
              "placement_fallbacks": sum(1 for s in block["stops"] if s["placement"] == "fallback")}
    for k, v in counts.items():
        if c[k] != v:
            err.append("%s: car %s is %s, the stops give %s" % (name, k, c[k], v))
    if ses["crashes"] != c["crashes_track"] + c["crashes_car"] or ses["pickups"] != c["pickups_requested"] + c["pickups_standing"]:
        err.append("%s: the session's crashes or pick-ups differ from the stops" % name)
    if ses["out_of_bounds"] != kinds.count("out_of_bounds"):
        err.append("%s: the session's out-of-bounds count differs from the stops" % name)
    down_s = float(config["staff"]["relocate_at_s"])
    for s in block["stops"]:
        if s["warned"] != (s["kind"] not in ("pickup", "standing")):
            err.append("%s: a stop of kind %s has warned %s" % (name, s["kind"], s["warned"]))
        if s["kind"] in ("pickup", "standing"):
            # a pick-up takes no warning away: a car crash at the car's fault begun before the staff set the car down
            # turns the pick-up's stop into its own
            for b in block["contacts"]:
                if (b["class"] == "crash" and b["fault"] == "car" and not b["warned"]
                        and s["t_s"] - TIME_TOL <= b["t0_s"] < s["t_s"] + down_s - TIME_TOL):
                    err.append("%s: a car crash at the car's fault with %s begun at %s s, in the pick-up of %s s before "
                               "the car was set down, is not warned" % (name, b["car"], b["t0_s"], s["t_s"]))
    warned_car = sum(1 for b in block["contacts"] if b["warned"])
    if warned_car != sum(1 for s in block["stops"] if s["kind"] == "car_crash"):
        err.append("%s: warned car crashes differ from the car-crash stops" % name)
    for b in block["contacts"]:
        if b["warned"] and not (b["class"] == "crash" and b["fault"] == "car"):
            err.append("%s: a contact with %s at %s is warned but is not a car crash at the car's fault" % (name, b["car"], b["t0_s"]))
        if b["t1_s"] < b["t0_s"]:
            err.append("%s: a contact ends before it starts" % name)
        err += contact_errors(name, b, config)
    if block["K"] > int(config["staff"]["warnings_end_session"]):
        err.append("%s: more warnings than end a heat" % name)
    if block["end_reason"] != ses["end"]:
        err.append("%s: the heat's end_reason differs from the session's end" % name)
    if (block["ended_by"] is not None) != (block["end_reason"] == "laps"):
        err.append("%s: ended_by is set exactly when the heat ended on laps" % name)
    if block["end_reason"] == "laps" and block["laps_target"] is not None:
        who = block["ended_by"]
        laps = block["car"]["laps"] if who == HC.CAR_ID else next((f["laps"] for f in block["field"] if f["id"] == who), None)
        if laps != block["laps_target"]:
            err.append("%s: the heat ended on laps but %s has %s laps, not N" % (name, who, laps))
    if block["car"]["laps"] != len(ses["laps"]):
        err.append("%s: the car's laps differ from the session's lap list" % name)
    ts = [e["t_s"] for e in block["overtakes"]]
    if ts != sorted(ts):
        err.append("%s: overtake events are not in time order" % name)
    return err


def check_result(result, config):
    """Errors of a venue result of the head-to-head day."""
    err = HC.record_errors("result", result)
    if err:
        return err
    names = [s["name"] for s in result["sessions"]]
    cfg_names = [s["name"] for s in config["sessions"]]
    if names != cfg_names[:len(names)]:
        err.append("sessions are not a prefix of %s" % (cfg_names,))
    heats = HC.heat_sessions(config)
    for s in result["sessions"]:
        for i, lap in enumerate(s["laps"]):
            if lap["n"] != i + 1:
                err.append("session %s: lap numbers do not count up from 1" % s["name"])
                break
            if lap["clean"] != ((not lap["interrupted"]) and lap["touches"] == 0):
                err.append("session %s lap %d: clean is not (not interrupted and no touch)" % (s["name"], lap["n"]))
            if lap["t_end_s"] > s["ticks"] * config["sim"]["tick_s"] + 1e-9 or lap["t_end_s"] - lap["lap_s"] < -1e-9:
                err.append("session %s lap %d lies outside the session" % (s["name"], lap["n"]))
            if i == 0 and lap["chain_break"]:
                err.append("session %s lap 1 has chain_break" % s["name"])
            elif i > 0 and not lap["chain_break"] and not check.laps_joined(s["laps"][i - 1]["t_end_s"], lap):
                err.append("session %s lap %d does not start where lap %d ended but has no chain_break"
                           % (s["name"], lap["n"], lap["n"] - 1))
        if s["name"] in heats:
            if "heat" not in s:
                err.append("session %s has no heat block" % s["name"])
            else:
                err += _heat_errors(s["name"], s, s["heat"], config)
        elif "heat" in s:
            err.append("session %s is not a heat but has a heat block" % s["name"])
        if s["end"] == "laps" and s["name"] not in heats:
            err.append("session %s ended on laps but is not a heat" % s["name"])
    if result["stats"]["laps_by_session"] != {s["name"]: len(s["laps"]) for s in result["sessions"]}:
        err.append("stats.laps_by_session differs from the lap lists")
    for k in ("touches", "crashes", "warnings"):
        if result["stats"][k] != sum(s[k] for s in result["sessions"]):
            err.append("stats.%s differs from the sessions" % k)
    return err


def strip_wall(obj):
    """A copy without the wall-clock fields (program and desk time)."""
    if isinstance(obj, dict):
        return {k: strip_wall(v) for k, v in obj.items() if k not in WALL_KEYS}
    if isinstance(obj, list):
        return [strip_wall(v) for v in obj]
    return obj


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


__all__ = ["check_result", "contact_errors", "heat_numbers", "strip_wall", "canonical"]
