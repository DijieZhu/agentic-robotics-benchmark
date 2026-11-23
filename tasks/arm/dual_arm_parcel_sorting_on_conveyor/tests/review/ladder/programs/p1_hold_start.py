"""Degenerate program 1 (do nothing): holds the start posture with the grippers open to the shift's end. Expected
shift score 0, credit exactly 0 (no W parcel turned, no P parcel touched, nothing lost)."""

WIRE_RUN = "pending: run through the full graded path (verify) once the packaging exists"


def solve(cell):
    info = cell.describe()
    st = cell.step(q_target=info["robot"]["start_posture"], grip=[0.0, 0.0], n=1)
    n = info["budgets"]["max_ticks_per_step"]
    while not st["shift_over"]:
        st = cell.step(n=n)
