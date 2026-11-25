"""The gripper: its width and force from the cell library's measured tables (arm.assembly_rov.gripper_wide: the grasp
centre in the tcp frame at an opening, the squeeze at which the force register's force is reached, the largest
opening, the force range), and a grasp of a module as a tcp pose in the module's part body frame."""
import json
import os

import numpy as np

from . import tf

HERE = os.path.dirname(os.path.abspath(__file__))


def tables():
    from arm.assembly_rov import gripper_wide as GW
    return GW


def load_grasps(path=None):
    with open(path or os.path.join(HERE, "data", "grasps.json")) as f:
        return json.load(f)


class Grasp(object):
    """one row of the grasp table: closing axis, approach and centre (the midpoint of the two pad faces' middles) in
    the module's part body frame (m), the module's width across the pads there (m)"""

    def __init__(self, row):
        self.row = dict(row)
        self.a = np.array(row["approach"], float)
        self.a /= np.linalg.norm(self.a)
        c = np.array(row["closing_axis"], float)
        c = c - (c @ self.a) * self.a
        self.c = c / np.linalg.norm(c)
        self.centre = np.array(row["centre_in_part_m"], float)
        self.width = float(row["width_m"])

    def R_tcp_in_part(self):
        x = np.cross(self.c, self.a)
        x /= np.linalg.norm(x)
        return np.column_stack([x, self.c, self.a])            # tcp x across the pads, y = closing, z = approach

    def T_tcp_in_part(self, opening):
        """the tcp pose in the part body frame that puts the grasp centre of the hand at `opening` on the centre"""
        GW = tables()
        R = self.R_tcp_in_part()
        p = self.centre - R @ np.asarray(GW.pad_offset(opening), float)
        return tf.T_of(p, R=R)

    def closing_width(self, force_N, extra_m=0.001):
        """the commanded width that makes the force register's force at this part's width (the table's squeeze) plus
        `extra_m`"""
        GW = tables()
        return max(0.0, self.width - GW.squeeze_for_force(force_N, self.width) - extra_m)


def open_width(info, part_width, clear_m=0.025):
    """the open hand for an approach: the part's width + clear_m, at most the gripper's largest opening"""
    return min(float(info["robot"]["gripper"]["max_width_m"]), float(part_width) + float(clear_m))
