"""The author's privileged oracle: it holds the token whose sha256 only the grader knows, so the graded
robot service answers its `truth` call (the site as built, the launch pose, the fittings' dimensions). It
shows that every graded site can be completed with the robot's own commands; it is not a participant
program and reads nothing a participant can."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/opt/visual_robotics/client")
from walk import Walk  # noqa: E402

APPROACH = .53


def route(t):
    """West to east through the open wide opening of every rack (the middle rack's is the one the pallet
    does not stand in), changing aisles in the open bays."""
    racks = t["racks"]
    deck_x = [f["x"] for f in t["fittings"] if f["kind"] == "deck"]
    is_deck = [any(r["near"]-.12 < x < r["far"]+.12 for x in deck_x) for r in racks]
    bays = []
    for i in range(len(racks)-1):
        far, near = racks[i]["far"], racks[i+1]["near"]
        fraction = .55 if is_deck[i] else .45 if is_deck[i+1] else .50
        bays.append(far+fraction*(near-far))
    path = []
    for i, r in enumerate(racks):
        y = sum(r["wide"])/2
        left = bays[i-1] if i else r["near"]-(.82 if is_deck[i] else APPROACH)
        right = bays[i] if i < len(bays) else r["far"]+(.82 if is_deck[i] else APPROACH)
        path += [(left, y), (right, y)]
    return path+[tuple(t["goal"])]


def mission(robot, token):
    obs = robot.observe()
    for step in ([.3, 0.], [.3, 0.], [-.3, 0.], [-.3, 0.]):       # the switch needs four scout moves
        obs = robot.act(step)
    obs = robot.advance()
    t = robot.request("truth", token=token)       # the site as the Go1 finds it, not as it was surveyed
    walk = Walk(route(t), t["fittings"])
    while not obs["done"]:
        # The Go1's own odometry drifts and its survey is out of date: the oracle steers on its true pose.
        x, y, yaw = robot.request("truth", token=token)["pose"]
        obs = robot.act(walk.action(np.array([x, y]), yaw, obs))


if __name__ == "__main__":
    from client import Client
    mission(Client(), open(os.path.join(HERE, "token.txt")).read().strip())
