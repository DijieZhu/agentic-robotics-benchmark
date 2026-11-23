"""Privileged control (flip nothing, touch nothing): in a privileged cell it reads the full truth every second
(get_state_privileged) and holds the start posture. Expected shift score 0, and the same record as program 1 apart
from the call counts: reading the truth changes nothing."""

WIRE_RUN = "pending: needs a privileged cell (the oracle token); in-process only until then"
PRIVILEGED = True


def solve(cell):
    info = cell.describe()
    st = cell.step(q_target=info["robot"]["start_posture"], grip=[0.0, 0.0], n=1)
    n = info["budgets"]["max_ticks_per_step"]
    seen = set()
    while not st["shift_over"]:
        truth = cell.get_state_privileged()
        for p in truth["parcels"]:
            seen.add(p["i"])
        st = cell.step(n=n)
