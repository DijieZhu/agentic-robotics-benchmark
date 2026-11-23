"""A minimal search program that runs: it looks around once, drives a few metres straight ahead and comes back.
It finds nobody; it shows the calls. Copy it to ~/submission/rescue.py to try the wire
(/opt/venv/bin/python3 ~/training/run_mission.py --site lanes)."""

import math

from rubblecell.client import decode_frame, decode_lidar, decode_thermal


def solve(robot):
    info = robot.describe()
    print("sim cap", info["sim_cap_s"], "s; budgets", info["budgets"])
    st = robot.get_state()
    print("start pose (map frame)", st["pose"], "link", st["link"])
    pts = decode_lidar(robot.get_lidar())
    print("lidar points", len(pts), "nearest", float(min((p[0] ** 2 + p[1] ** 2) ** 0.5 for p in pts)) if len(pts) else None)
    rgb, depth = decode_frame(robot.get_camera("mast"))
    print("mast frame", rgb.shape, "depth valid", int((depth == depth).sum()))
    temp = decode_thermal(robot.get_thermal())
    print("thermal min/max C", float(temp.min()), float(temp.max()))
    robot.set_flippers(math.radians(10), math.radians(10))
    robot.drive(0.3, 0.0)
    robot.step(600)                      # 6 s
    robot.drive(-0.3, 0.0)
    robot.step(600)
    robot.drive(0.0, 0.0)
    robot.step(50)
    print("end pose", robot.get_state()["pose"])
    robot.finish()
