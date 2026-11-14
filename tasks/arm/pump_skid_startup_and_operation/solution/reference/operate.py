"""The reference submission (installed by solution/solve.sh together with the oracle token): one privileged call that runs
the scripted reference program inside the cell. Two camera frames are read first so every validation also exercises
the render path the agent's programs use."""


def solve(cell):
    cell.describe()
    cell.get_camera("overview")
    cell.get_camera("wrist")
    cell.reference_run()
