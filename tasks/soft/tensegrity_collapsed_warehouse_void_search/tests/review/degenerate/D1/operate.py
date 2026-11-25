"""Degenerate submission D1 (the grader's QC): wait 25 s, report the three points from the pad, finish."""


def solve(station):
    t = 0.0
    while t < 25.0:
        t = float(station.step(5.0)["t"])
    for i in (1, 2, 3):
        station.report_point(i)
    station.finish()
