"""Degenerate program 4 (random targets): random joint targets around the start posture (up to 1 rad from it,
within the ranges), approached at a moderate pace (about 1 rad/s per joint, below the cell's limit), a new one every
1.5 s, with a random grip. Expected shift score 0: the arms wander through the work window, knock parcels about and
stop on contacts (protective stops), without locking out at once."""

import random

WIRE_RUN = "pending: run through the full graded path (verify) once the packaging exists"
SPREAD = 1.0          # rad around the start posture
PACE = 1.0            # rad/s per joint, commanded in small steps
HOLD = 1.5            # s per target


def solve(cell):
    info = cell.describe()
    rng = random.Random(20261119)
    lo = [r[0] for r in info["robot"]["joint_range"]]
    hi = [r[1] for r in info["robot"]["joint_range"]]
    q0 = list(info["robot"]["start_posture"])
    tick = info["tick_s"]
    st = cell.get_state()
    cur = list(st["q_target"])
    while not st["shift_over"]:
        goal = [min(b, max(a, c + rng.uniform(-SPREAD, SPREAD))) for a, b, c in zip(lo, hi, q0)]
        grip = [rng.random(), rng.random()]
        t_end = st["t"] + HOLD
        while not st["shift_over"] and st["t"] < t_end:
            step = PACE * 5 * tick
            cur = [c + max(-step, min(step, g - c)) for c, g in zip(cur, goal)]
            st = cell.step(q_target=cur, grip=grip, n=5)
            if st["refused"]:
                cur = list(st["q_target"])
            if st["stop"]["locked_out"]:
                break
        if st["stop"]["locked_out"]:
            break
    while not st["shift_over"]:
        st = cell.step(n=info["budgets"]["max_ticks_per_step"])
