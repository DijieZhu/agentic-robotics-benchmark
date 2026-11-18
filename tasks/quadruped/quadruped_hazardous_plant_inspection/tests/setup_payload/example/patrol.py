"""A minimal patrol program that runs: it makes every call of the robot once, walks a metre into the airlock and back,
and reports nothing that counts (its only frames are taken on the pad, the instrument report cites a navigation
frame, and the survey mark and the source are reported without going there; it reads no tag plate and looks at no
obstacle). Copy it to ~/submission/patrol.py to try the wire:

    /opt/venv/bin/python3 ~/training/run_mission.py --list
    /opt/venv/bin/python3 ~/training/run_mission.py --site hall

print() goes to stderr (the launcher keeps stdout for the wire).
"""

import math


def solve(robot):
    info = robot.describe()                                   # the datasheet
    # drawing (a sheet with docs["drawing"]["sheets"][i]["missing"] true is not there), link plan, work order, round
    # sheet, modification record, catalogue with the plant lettering, budgets
    docs = robot.mission()
    print("robot:", info["robot"]["model"], "| sim time limit", info["time"]["sim_time_limit_s"], "s")
    print("work order:", [(it["tag"], it["type"]) for it in docs["work_order"]["items"]])
    st = robot.get_state()
    print("start: odometry", st["odometry"], "link", st["link"], "status", st["status"])

    robot.set_mode("stand")
    robot.set_body(height=0.24, pitch=0.0)
    robot.set_head(pan=0.0, tilt=0.1)                         # rad; + tilts the head nose down
    robot.set_lamp(True)
    robot.step(50)                                            # 50 x 10 ms

    nav = robot.get_camera("nav", depth=True)                 # RGB + depth (metres, NaN = no reading)
    ins = robot.get_camera("inspect")                         # 1280 x 960 telephoto
    dep = robot.get_depth()                                   # a depth-only nav frame
    th = robot.get_thermal()                                  # degrees C
    sw = robot.get_lidar()                                    # ranges (m, NaN = no return), angles (rad)
    print("nav", nav["image"].shape, "valid depth", int((nav["depth"] == nav["depth"]).sum()),
          "| inspect", ins["image"].shape, "| depth frame", dep["frame_id"],
          "| thermal max %.1f C" % float(th["image_c"].max()),
          "| lidar nearest %.2f m" % float(min((r for r in sw["ranges"] if r == r), default=float("nan"))))

    robot.set_lamp(False)
    robot.set_mode("walk")
    # forward 0.25 m/s. The reply is {"ok": True}; for 5 s after a `blocked` event it is {"ok": False, "held": True,
    # "hold_until": t} and the command is not taken (robot_api.md, section 5): a program that walks checks it
    robot.walk(0.25, 0.0, 0.0)
    robot.step(400)
    robot.walk(-0.25, 0.0, 0.0)                               # and back
    robot.step(400)
    robot.walk(0.0, 0.0, 0.0)
    robot.step(100)
    robot.set_mode("stand")
    robot.step(50)
    st = robot.get_state()
    print("back: odometry (%.2f, %.2f, %.1f deg)" % (st["odometry"]["x"], st["odometry"]["y"],
                                                     math.degrees(st["odometry"]["yaw"])))

    items = docs["work_order"]["items"]
    instruments = [it for it in items if it["type"] not in ("dose_point", "hotspot")]
    marks = [it for it in items if it["type"] == "dose_point"]
    # a value-free report citing a navigation frame (which is never evidence for an instrument): it scores 0
    print("report:", robot.report(instruments[0]["tag"], None, [nav["frame_id"]]))
    if marks:                                                 # far from the mark: no dwell there, so it scores 0
        print("report_mark:", robot.report_mark(marks[0]["tag"], 0.01))
    print("report_source:", robot.report_source(0.0, 0.0))   # never approached: it scores 0
    robot.step(100)
    print("finish:", robot.finish())
