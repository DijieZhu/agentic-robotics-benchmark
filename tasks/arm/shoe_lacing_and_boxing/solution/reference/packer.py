"""The reference submission (installed by solution/solve.sh together with the oracle token): one privileged call that runs
the scripted reference controller inside the cell. A camera frame is read first so every validation also exercises the
render path the agent's programs use."""


def solve(cell):
    cell.describe()
    cell.get_camera("overhead")
    cell.get_camera("front")
    cell.reference_run()
