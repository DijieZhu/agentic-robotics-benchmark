"""The reference submission (installed by solution/solve.sh together with the oracle token): one privileged call that
runs the reference program inside the cell (tests/reference/reference.py: the true pose and the site's true layout,
the robot's own primitive table, every command through the station's calls). The mission, one camera frame, the
telemetry (still empty at t = 0: no record before the first step) and the status come first, so every validation also
exercises the paths the agents' programs use."""


def solve(station):
    station.mission(); station.get_frame(); station.telemetry(); station.status()
    station._call("reference_run")       # (a call of the operators' cell only: refused without the oracle token)
