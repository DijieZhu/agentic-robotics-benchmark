"""Shared helper of the degenerate programs (a helper module, as a submission may ship one)."""


def go(cell, q, grip=None, ticks=40):
    """Command q (and grip) and step ``ticks`` ticks, waiting out a protective stop (the command is sent again once
    the stop is over). Returns the last state."""
    st = cell.step(q_target=list(q), grip=grip, n=min(ticks, 50))
    left = ticks - st["ticks"]
    while not st["shift_over"] and (left > 0 or st["refused"]):
        if st["stop"]["locked_out"]:
            break
        if st["refused"] and not st["stop"]["active"]:
            st = cell.step(q_target=list(q), grip=grip, n=1)
        else:
            st = cell.step(n=min(max(left, 1), 50))
        left -= max(1, st["ticks"])
    return st
