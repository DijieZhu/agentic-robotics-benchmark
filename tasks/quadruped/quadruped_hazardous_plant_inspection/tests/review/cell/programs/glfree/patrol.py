"""A test program that renders nothing (no camera, depth or thermal frame): it walks 8 m into the building on its
odometry with a lidar sweep every second, turns round, walks back to the pad, counts the dose for 20 s, reports a
survey mark and an instrument (no frame), and finishes. Physics, the gait firmware, the lidar's ray casts, the
dosimeter, the link and the report queue only: its result can be compared bit for bit across machines whatever
their OpenGL."""

import math


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def solve(robot):
    docs = robot.mission()
    items = docs["work_order"]["items"]
    inst = [it for it in items if it["type"] not in ("dose_point", "hotspot")]
    marks = [it for it in items if it["type"] == "dose_point"]
    robot.set_mode("walk")
    t_next = 0.0
    for leg, (goal, back) in enumerate(((8.0, False), (0.15, True))):
        for _ in range(4000):
            st = robot.get_state()
            o = st["odometry"]
            if st["sim_time"] >= t_next:
                robot.get_lidar()
                t_next = st["sim_time"] + 1.0
            want = math.pi if back else 0.0
            e = wrap(want - o["yaw"])
            if abs(e) > 0.35:
                robot.walk(0.0, 0.0, 0.8 * (1 if e > 0 else -1))
            else:
                lat = (o["y"] - 0.0) * (1 if back else -1)
                robot.walk(0.3, 0.6 * lat, 1.2 * e)
            robot.step(10)
            if (not back and o["x"] >= goal) or (back and o["x"] <= goal):
                break
        robot.walk(0.0, 0.0, 0.0)
        robot.step(100)
        if leg == 0:
            robot.report(inst[0]["tag"], 1.0, [])          # out of coverage: queued
    robot.set_mode("stand")
    robot.step(100)
    robot.get_state()
    robot.step(2000)
    st = robot.get_state()
    rate = st["dosimeter"]["counts"] / max(st["dosimeter"]["dwell_s"], 1e-6) / 45.0
    if marks:
        robot.report_mark(marks[0]["tag"], rate)
    robot.step(100)
    robot.finish()
