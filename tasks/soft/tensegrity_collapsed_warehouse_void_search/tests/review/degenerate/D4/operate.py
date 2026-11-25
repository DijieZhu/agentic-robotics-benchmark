"""Degenerate submission D4 (the grader's QC): breaks the contract (imports socket); otherwise D1's program."""
import socket  # noqa: F401


def solve(station):
    t = 0.0
    while t < 25.0:
        t = float(station.step(5.0)["t"])
    for i in (1, 2, 3):
        station.report_point(i)
    station.finish()
