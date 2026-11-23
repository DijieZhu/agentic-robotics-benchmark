"""The reference submission (installed by solution/solve.sh together with the oracle token): one privileged call that
runs the reference program inside the cell. The station's facts and one frame of every camera are read first, so that
every validation also exercises the render path the agent's programs use."""

CAMERAS = ("infeed", "head", "wrist_left", "wrist_right")


def solve(cell):
    cell.describe()
    for name in CAMERAS:
        cell.get_camera(name, depth=True)
    cell.reference_run()
