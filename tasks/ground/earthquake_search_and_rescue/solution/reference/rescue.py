"""The reference submission (installed by solution/solve.sh together with the oracle token): one privileged call that
runs the reference program inside the cell (tests/reference/reference.py, with the site's plan). A camera frame, a
lidar sweep and a thermal frame are read first so every validation also exercises the sensor paths agents use."""


def solve(robot):
    robot.describe()
    robot.get_camera("mast")
    robot.get_lidar()
    robot.get_thermal()
    robot.reference_run()
