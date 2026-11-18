"""The reference submission (installed by solution/solve.sh together with the oracle token): one privileged call that
runs the reference program inside the cell (tests/reference/reference.py, with the true pose and the site's true
layout). One frame of every sensor is taken first, so every validation also exercises the sensor paths the agents'
programs use."""


def solve(robot):
    robot.describe()
    robot.mission()
    robot.get_state()
    robot.get_camera("nav", depth=True)
    robot.get_camera("inspect")
    robot.get_depth()
    robot.get_thermal()
    robot.get_lidar()
    robot._call("reference_run")         # (a call of the operators' cell only: not in the agents' client)
