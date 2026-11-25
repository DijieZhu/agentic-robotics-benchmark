"""Degenerate submission D2 (the grader's QC): never move; step until the time limit."""


def solve(station):
    while not station.step(10.0)["done"]:
        pass
