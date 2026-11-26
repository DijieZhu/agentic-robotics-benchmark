"""Example controller: valid replies on the wire and nothing else. It scores nothing, by construction.

What it does: setup answers nothing (no mass profile); every plan call is refused with the reason outside_zones
(one of refusal.reasons), so it flies no job; every step answers a zero force and torque with the event none. It
reads nothing of the frame or of the session paper, estimates nothing and controls nothing.

Why it scores nothing, in the acceptance specification's own terms. It accepts no feasible job, so the share of
feasible jobs accepted is nothing and the refusal term is nothing by score.refusals:
    J x (share of accepted feasible jobs completed), J = share of requests refused + share of feasible jobs accepted - 1, and 0 where J is below 0; the feasible jobs are those that count toward the score, the requests every request of the session; a refused feasible job and a flown request score 0 themselves
It flies no job, so no job is completed by score.completed and every kind's term is nothing (score.leg, score.slew,
score.hold, score.view).

Run it through the toy self-test from the controller folder:
    python3 selftest/selftest.py example_controller
"""


class Controller(object):
    def setup(self, session):
        return None

    def plan(self, job, frame):
        return {"refuse": "outside_zones"}

    def step(self, frame):
        return {"force": [0.0, 0.0, 0.0], "torque": [0.0, 0.0, 0.0], "event": "none"}
