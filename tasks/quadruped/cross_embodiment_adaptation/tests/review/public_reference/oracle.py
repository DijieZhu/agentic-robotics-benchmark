"""ALE author solution: align the two sensor frames before walking."""
import math

from client import Client
from registration import mast_scan, register, rot
from survey import scan
from walk import Walk


def deploy(robot, survey, align=register, posture="adaptive"):
    path = survey.route()
    chart = survey.landmarks()
    obs = robot.advance()
    local, offset = mast_scan(robot, obs)
    pose, diagnostics = align(local, chart, offset)
    R = rot(pose[2])
    terrain = survey.terrain()
    ducts = [item["xy"][0] for item in terrain if item["top_m"] > .30]
    walk = Walk(path, ducts, posture)
    while not obs["done"]:
        xy = pose[:2]+R @ obs["odom_xy_m"]
        yaw = pose[2]+obs["odom_yaw_rad"]
        obs = robot.act(walk.action(xy, math.atan2(math.sin(yaw), math.cos(yaw)), obs["ticks_remaining"]))
    return {"applied_pose": pose.tolist(), "observed_fittings": terrain, **diagnostics}


def main():
    robot = Client()
    deploy(robot, scan(robot))


if __name__ == "__main__":
    main()
