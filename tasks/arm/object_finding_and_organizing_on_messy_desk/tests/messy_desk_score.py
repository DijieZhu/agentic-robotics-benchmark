"""messy_desk_score — the episode score, computed from the simulator's privileged state.

    S = O * R                      (scaled by steps/H when the grader cut the episode short:
                                    the controller exhausted its time budget)

Each request is a task (find + organize) judged when it closes: at its mark or its deadline. The
product makes the retrieval the task (2026-11-08): sorting the desk counts only as far as the
requests are served, and serving requests counts only as far as the desk stays in order. Until
then S was 1/2 * O + 1/2 * R, and a controller that only sorted the arrivals and never searched
for a requested object earned about 0.4 (a measured agent episode of 2026-11-08: O 0.81, R 0.08).

O, organization: the mean of snapshots O(t) taken at every step a request closes and at the
episode's last step. O(t): every object that was ever in the scene contributes v_o(t) - b_o(t),
where v is +1 placed in a right container, -1 placed in a wrong container, 0 on the table /
held / in flight (the requested object belongs to the red container only, and holding it
counts +1), and b_o(t) is the object's baseline: its value when it entered the scene, when
the world last moved it, or when a request named it. Summed over objects and divided by the
gain that was available, sum_o max(0, 1 - b_o(t)), then clipped to [0, 1]. An untouched scene
scores exactly 0. Removed objects keep their last value (container service, delivery) or -1
(lost off the table), also when their instance is reused.

R, request progress: points divided by twice the number of opened requests, including an ongoing one,
and floored at 0. The first hold of the requested target (both fingers on it, lifted clear of its
support) earns 1 point once; a verified delivery or correct absence declaration brings that request
to 2 points. Repeated holds earn nothing more. An expiry keeps any first-hold credit; a wrong mark
(delivered while the object is not placed in red, or absent while it is on the desk) costs
WRONG_MARK_COST on top, so a guessed mark has a negative expected value. No requests gives 0.

Stdlib-only; the verify stage's (the agent's sandbox has no copy since 2026-11-07).
"""

SUCCESS_EVENTS = ("request_delivered", "request_absent_correct")
FAILURE_EVENTS = ("request_claim_wrong", "request_absent_wrong", "request_expired")
WRONG_MARK_EVENTS = ("request_claim_wrong", "request_absent_wrong")
# A wrong mark costs a point; an expired request costs nothing. Without the cost a mark would be
# free to guess: one request in five names an absent object, so "no such object" on every request
# would earn 0.4 points per request for nothing. With it, a guess below 1/3 confidence loses
# (below 1/2 for a "delivered" mark once the first hold is banked).
WRONG_MARK_COST = 1.0


class EpisodeScorer(object):
    def __init__(self, horizon):
        self.horizon = float(horizon)
        self.steps = 0
        self.snapshots = []       # O(t) at every step a request closed
        self.ghosts = []          # (value, baseline) of removed incarnations whose instance was reused
        self.prev = {}            # id -> (present, removed, value, baseline) at the latest step
        self.request_points = {}  # request id -> 0, 1 (first hold) or 2 (success), minus WRONG_MARK_COST after a wrong mark
        self.held_requests = set()
        self.counts = {"requests_opened": 0, "requests_success": 0, "requests_failed": 0,
                       "requests_delivered": 0, "absent_correct": 0, "claims_wrong": 0, "absent_wrong": 0,
                       "requests_expired": 0, "first_holds": 0, "arrivals": 0, "containers_cleared": 0,
                       "disturbances": 0, "objects_lost": 0}

    def add_step(self, state):
        for o in state.get("objects", []):
            oid = o["id"]
            v, b = float(o.get("value", 0)), float(o.get("baseline", 0))
            was = self.prev.get(oid)
            if was is not None and was[1] and o.get("present") and not o.get("removed"):
                self.ghosts.append((was[2], was[3]))       # the removed incarnation keeps counting
            self.prev[oid] = (bool(o.get("present")), bool(o.get("removed")), v, b)
        if any(e.get("event", "") in SUCCESS_EVENTS or e.get("event", "") in FAILURE_EVENTS for e in state.get("events", [])):
            self.snapshots.append(self._snapshot())
        for e in state.get("events", []):
            kind = e.get("event", "")
            if kind == "request_opened" or kind == "request_first_hold" or kind in SUCCESS_EVENTS or kind in FAILURE_EVENTS:
                rid = e["request"]
                self.request_points.setdefault(rid, 0.0)
                if kind == "request_first_hold":
                    self.request_points[rid] = max(self.request_points[rid], 1.0)
                    if rid not in self.held_requests:
                        self.held_requests.add(rid)
                        self.counts["first_holds"] += 1
                elif kind in SUCCESS_EVENTS:
                    self.request_points[rid] = 2.0
                elif kind in WRONG_MARK_EVENTS:
                    self.request_points[rid] -= WRONG_MARK_COST
            if kind in SUCCESS_EVENTS:
                self.counts["requests_success"] += 1
            elif kind in FAILURE_EVENTS:
                self.counts["requests_failed"] += 1
            if kind == "request_delivered":
                self.counts["requests_delivered"] += 1
            elif kind == "request_absent_correct":
                self.counts["absent_correct"] += 1
            elif kind == "request_claim_wrong":
                self.counts["claims_wrong"] += 1
            elif kind == "request_absent_wrong":
                self.counts["absent_wrong"] += 1
            elif kind == "request_expired":
                self.counts["requests_expired"] += 1
            elif kind == "request_opened":
                self.counts["requests_opened"] += 1
            elif kind == "arrival":
                self.counts["arrivals"] += 1
            elif kind == "container_cleared":
                self.counts["containers_cleared"] += 1
            elif kind.startswith("disturb_"):
                self.counts["disturbances"] += 1
            elif kind == "fell_off_desk":
                self.counts["objects_lost"] += 1
        self.steps += 1

    def _final(self):
        """(sum of v - b, sum of max(0, 1 - b)) over every incarnation, at the latest step."""
        vb = [(v, b) for (_, _, v, b) in self.prev.values()] + self.ghosts
        return sum(v - b for v, b in vb), sum(max(0.0, 1.0 - b) for v, b in vb)

    def _snapshot(self):
        num, den = self._final()
        return min(1.0, max(0.0, num / den)) if den > 0 else 0.0

    def organization(self):
        """Mean of the request-close snapshots and the latest step (the episode's last step once
        it has ended)."""
        snaps = self.snapshots + [self._snapshot()]
        return sum(snaps) / len(snaps)

    def request_rate(self):
        closed = self.counts["requests_success"] + self.counts["requests_failed"]
        return self.counts["requests_success"] / float(closed) if closed else 0.0

    def request_progress(self):
        opened = len(self.request_points)
        return max(0.0, sum(self.request_points.values()) / (2.0 * opened)) if opened else 0.0

    def score_unscaled(self):
        """S = O * R, before the steps/H scaling of an episode the grader cut short."""
        return self.organization() * self.request_progress()

    def score(self):
        s = self.score_unscaled()
        if self.steps < self.horizon:                    # cut short by the grader: the steps never run earn nothing
            s *= self.steps / self.horizon
        return s

    def live(self):
        """Compact running view for the practice tools: `score` is what the episode would score if
        the grader cut it now (steps/H scaling included), `score_running` the unscaled O * R
        with the current state as the last snapshot, `organization_now` that snapshot alone."""
        closed = self.counts["requests_success"] + self.counts["requests_failed"]
        num, den = self._final()
        return {"score": round(self.score(), 5), "score_running": round(self.score_unscaled(), 5),
                "organization": round(self.organization(), 5), "request_rate": round(self.request_rate(), 5),
                "request_progress": round(self.request_progress(), 5),
                "org_num": round(num, 1), "org_den": round(den, 1),
                "requests_success": self.counts["requests_success"], "requests_closed": closed,
                "requests_open": self.counts["requests_opened"] - closed, "steps": self.steps, "horizon": int(self.horizon),
                "organization_now": round(self._snapshot(), 5), "snapshots": [round(x, 4) for x in self.snapshots]}

    def summary(self):
        d = dict(self.counts)
        num, den = self._final()
        d.update(score=self.score(), organization=self.organization(), request_rate=self.request_rate(),
                 request_progress=self.request_progress(),
                 org_num=num, org_den=den, steps=self.steps, snapshots=[round(x, 4) for x in self.snapshots])
        return d
