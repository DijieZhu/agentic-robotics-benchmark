"""Degenerate program 0: reads the station's facts and returns at once (the shift ends there). Expected shift
score 0: every parcel is lost (credit negative)."""

WIRE_RUN = "pending: run through the full graded path (verify) once the packaging exists"


def solve(cell):
    cell.describe()
    cell.get_state()
