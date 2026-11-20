"""An example program that runs and scores 0: it leaves the dock, takes one frame and docks again.

Copy it to ~/submission/caretake.py as a starting point.  The cell imports this file and calls solve(robot);
robot is a cabinkit Robot (robot_api.md lists every call, its arguments, return codes and budgets).
"""


def solve(robot):
    m = robot.mission()
    berth = m["berth"]
    print("start charge", m["battery_pct"], "% on berth", berth)

    r = robot.undock()
    print("undock", r["code"], "at", r["t"], "s")

    f = robot.camera("FIC")
    print("frame", f["frame_id"], f["width"], "x", f["height"], "pixels" if "image" in f else "(no pixels)")

    robot.step(2.0)
    r = robot.dock(berth)
    print("dock", r["code"], "at", r["t"], "s")
