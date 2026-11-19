"""The line between a touch and a car crash: contact bouts with an opponent, their push, and the rule that confirms
or moves the line.

    bt = BoutTracker()                                   # merge 1.0 s, push window 1.5 s after the last contact
    every physics step:  bt.step(t, ids_in_contact, offset_of, reset_of)
                         # offset_of(id) -> the opponent's signed lateral offset from its own line now
                         # reset_of(id)  -> True while the opponent is held or being put back by its staff
    bt.finish(t_end)                                     # closes the open bouts (window cut at the end)
    bt.bouts                                             # [{car, t0_s, t1_s, e0_m, push_m, reset, ...}]
    d = decide(u, [b["push_m"] for b in bt.bouts])       # {"outcome": "stands" | "moves" | "none", "line_m": ...}

    pushes = pushes_of_results(venue_results)            # the push of every contact bout of the heat referee's records
    d = decide(u, pushes)                                # the same rule on the pushes the heat referee measured

A bout: contacts of the car with one opponent less than `contact_merge_s` apart, [t0, t1]. The push of `BoutTracker`
is the first definition: the largest |e(t) - e(t0)| for t in [t0, t1 + `window_after_s`], e = the opponent's lateral
offset from its own line; it is kept as an independent cross-check of contact bouts.
The push the rule is applied to is the heat referee's: at the first contact of a bout the heat is copied without the
car, the copy runs for 1.5 s, and the push is the largest difference of the opponent's lateral offset between the heat
and the copy at the same place on its line (`contacts[].push_m` of a venue result, each contact with its `replay`
block). `decide` takes numbers, contact records or whole venue results; `pushes_of_results` reads every bout of every
heat of the results, none dropped, and refuses a contact without its `replay` block (a push measured the first way).

The rule (block `crash_line_rule` of the field parameters), with u = the undisturbed motion (the largest push measured
in the same way on the field running from the grid without the car, from copies taken during that run): the line stays at `start_m` when `start_m` >= 2 u and no push lies
in [`empty_band_low_m`, `empty_band_high_m`]; otherwise it moves to the middle, rounded to `round_to_m`, of the widest
interval inside `search_range_m` that holds no push, lies at or above 2 u and is at least `min_width_m` wide (the
lowest such interval when two are equally wide); if there is none, the rule sets no line.
"""

import math

from . import world as W


def _rule(params=None):
    return W.load_params(params)["crash_line_rule"]


class BoutTracker(object):
    def __init__(self, merge_s=None, after_s=None, params=None):
        r = _rule(params)
        self.merge = float(r["contact_merge_s"] if merge_s is None else merge_s)
        self.after = float(r["window_after_s"] if after_s is None else after_s)
        self.cur = {}                # opponent id -> the bout that can still merge contacts
        self.measuring = []          # bouts whose push window is open
        self.bouts = []              # every bout, in the order of their first contact

    def step(self, t, ids, offset_of, reset_of=None, info_of=None):
        for j in sorted(ids):
            b = self.cur.get(j)
            if b is not None and t - b["t1_s"] < self.merge:
                b["t1_s"] = t
                b["contact_steps"] += 1
                continue
            e0 = float(offset_of(j))
            b = {"car": j, "t0_s": t, "t1_s": t, "e0_m": e0, "push_m": 0.0, "push_at_s": t, "reset": False,
                 "contact_steps": 1, "final": False}
            if info_of is not None:
                b.update(info_of(j))
            self.cur[j] = b
            self.bouts.append(b)
            self.measuring.append(b)
        keep = []
        for b in self.measuring:
            if t > b["t1_s"] + self.after:
                b["final"] = True
                continue
            j = b["car"]
            if reset_of is not None and reset_of(j):
                b["reset"] = True
            p = abs(float(offset_of(j)) - b["e0_m"])
            if p > b["push_m"]:
                b["push_m"], b["push_at_s"] = p, t
            keep.append(b)
        self.measuring = keep

    def finish(self, t_end=None):
        for b in self.measuring:
            b["final"] = True
            b["cut_at_s"] = t_end
        self.measuring = []
        return self.bouts


def _contacts_of(result):
    """The contact records of every heat of a venue result (or of a record that holds one under `result`)."""
    if isinstance(result, dict) and "sessions" not in result and isinstance(result.get("result"), dict):
        result = result["result"]
    if not isinstance(result, dict) or not isinstance(result.get("sessions"), list):
        raise ValueError("not a venue result: no list of sessions")
    out = []
    for s in result["sessions"]:
        heat = s.get("heat") if isinstance(s, dict) else None
        if isinstance(heat, dict):
            out += list(heat.get("contacts") or [])
    return out


def pushes_of_results(results, require_replay=True):
    """The push (m) of every contact bout in every heat of the venue results, in order; with `require_replay` a
    contact without its `replay` block is an error."""
    if isinstance(results, dict):
        results = [results]
    out = []
    for r in results:
        for c in _contacts_of(r):
            if require_replay and "replay" not in c:
                raise ValueError("a contact of %s at %s s has no replay block" % (c.get("car"), c.get("t0_s")))
            out.append(float(c["push_m"]))
    return out


def push_values(pushes):
    """Pushes as floats: from numbers, contact records ({push_m, ...}) or venue results (their every heat bout)."""
    if isinstance(pushes, dict):
        pushes = [pushes]
    out = []
    for x in pushes:
        if isinstance(x, bool):
            raise ValueError("a push is a number of metres, not %r" % (x,))
        if isinstance(x, dict):
            if "push_m" in x:
                out.append(float(x["push_m"]))
            else:
                out += pushes_of_results([x])
        else:
            out.append(float(x))
    return out


def round_half_up(x, step):
    """x rounded to a multiple of step, halves up (a tolerance of 1e-9 steps absorbs the binary representation)."""
    return round(math.floor(x / step + 0.5 + 1e-9) * step, 10)


def decide(u, pushes, params=None):
    """The crash line by the rule (see the module text). `u` in metres; `pushes` = the push of every bout (numbers,
    contact records or venue results, `push_values`)."""
    r = _rule(params)
    start = float(r["start_m"])
    two_u = float(r["undisturbed_factor"]) * float(u)
    lo_b, hi_b = float(r["empty_band_low_m"]), float(r["empty_band_high_m"])
    p = sorted(push_values(pushes))
    in_band = [x for x in p if lo_b <= x <= hi_b]
    out = {"u_m": float(u), "two_u_m": two_u, "bouts": len(p), "enough_bouts": len(p) >= int(r["min_contact_bouts"]),
           "start_m": start, "pushes_in_band": in_band, "start_above_two_u": start >= two_u}
    if start >= two_u and not in_band:
        out.update({"outcome": "stands", "line_m": start, "interval_m": None})
        return out
    lo = max(float(r["search_range_m"][0]), two_u)
    hi = float(r["search_range_m"][1])
    inner = [x for x in p if lo < x < hi]
    edges = [lo] + inner + [hi] if lo < hi else []
    cands = [(a, b) for a, b in zip(edges, edges[1:]) if b - a >= float(r["min_width_m"]) - 1e-12]
    out["candidates_m"] = [[round(a, 6), round(b, 6)] for a, b in cands]
    if not cands:
        out.update({"outcome": "none", "line_m": None, "interval_m": None})
        return out
    best = cands[0]
    for c in cands[1:]:
        if c[1] - c[0] > best[1] - best[0] + 1e-12:
            best = c
    out.update({"outcome": "moves", "interval_m": [round(best[0], 6), round(best[1], 6)],
                "line_m": round_half_up(0.5 * (best[0] + best[1]), float(r["round_to_m"]))})
    return out
