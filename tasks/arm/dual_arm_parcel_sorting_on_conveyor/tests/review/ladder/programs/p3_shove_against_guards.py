"""Degenerate program 3 (shove against the guards): the left hand, fingers closed and the tool down, sweeps across the
belt from the far side to beyond the near edge in the downstream half of the work window, lifts and returns high,
again and again, its fingertips about 9 cm and 6 cm above the belt surface on alternate sweeps: every parcel in its
way is shoved towards the near side. The station's guards (the near lip through the work window, the rails) keep a
shoved parcel on the belt, so it is crushed against the lip (damaged, lost) or knocked over. Expected shift score 0.

The waypoints were taught offline on the given robot and station models (for each height one continuous inverse
kinematics chain of the tool centre point along the sweep, the same arm configuration throughout; every straight
joint segment keeps more than 4 cm from the station and from the robot's own parts)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reach import go  # noqa: E402

WIRE_RUN = "pending: run through the full graded path (verify) once the packaging exists"
SWEEP_LOW = [
    [-2.1703, 0.1439, 2.5523, 0.8769, -1.4394, -1.2517, -1.1225, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # above the belt, towards the far side
    [-1.6925, 0.0266, 2.5999, 0.3590, -1.3675, -1.0670, -1.1783, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # low sweep 1 (far to near)
    [-1.8352, 0.2206, 2.2881, 0.7285, -1.7847, -1.3429, -0.4497, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # low sweep 2 (far to near)
    [-1.9053, 0.4102, 2.2155, 1.0315, -2.2313, -1.3238, 0.1199, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # low sweep 3 (far to near)
    [-2.0038, 0.5367, 2.3313, 1.3342, -2.5984, -1.1399, 0.4684, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # low sweep 4 (far to near)
    [-2.0874, 0.6490, 2.4013, 1.6257, -2.7700, -0.9434, 0.6662, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # low sweep 5 (far to near)
    [-2.1533, 0.7368, 2.4327, 1.8746, -2.7700, -0.8375, 0.7337, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # low sweep 6 (far to near)
    [-2.2109, 0.8185, 2.4691, 2.1027, -2.7700, -0.7355, 0.8056, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # low sweep 7 (far to near)
    [-2.2578, 0.8981, 2.5097, 2.3187, -2.7700, -0.6279, 0.8868, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # low sweep 8 (far to near)
    [-2.7747, 0.6896, 2.4524, 2.0498, -2.7700, -1.3043, 0.4015, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # up, beyond the near edge
]
SWEEP_HIGH = [
    [-2.1896, 0.1085, 2.6029, 0.8184, -1.3836, -1.2710, -1.2510, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # above the belt, towards the far side
    [-1.7274, 0.0031, 2.6442, 0.3058, -1.3074, -1.1117, -1.3139, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # high sweep 1 (far to near)
    [-1.8877, 0.1981, 2.2796, 0.6785, -1.7496, -1.4314, -0.5138, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # high sweep 2 (far to near)
    [-1.9785, 0.3911, 2.2405, 0.9869, -2.2952, -1.3941, 0.1241, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # high sweep 3 (far to near)
    [-2.1159, 0.4990, 2.4257, 1.3123, -2.7069, -1.1702, 0.4573, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # high sweep 4 (far to near)
    [-2.2185, 0.5886, 2.4812, 1.6029, -2.7700, -1.0271, 0.5530, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # high sweep 5 (far to near)
    [-2.3074, 0.6641, 2.5185, 1.8530, -2.7700, -0.9284, 0.6042, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # high sweep 6 (far to near)
    [-2.3905, 0.7315, 2.5579, 2.0802, -2.7700, -0.8416, 0.6572, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # high sweep 7 (far to near)
    [-2.4691, 0.7913, 2.5993, 2.2930, -2.7700, -0.7610, 0.7116, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # high sweep 8 (far to near)
    [-2.8911, 0.6037, 2.5562, 1.9913, -2.7700, -1.3219, 0.2797, 0.7842, -2.1067, -0.7271, 1.9460, 1.9143, 1.5400, 1.1825],    # up, beyond the near edge
]
TICKS = [30] + [4] * 8 + [12]          # per waypoint: reach the start slowly, sweep briskly, lift


def solve(cell):
    info = cell.describe()
    st = go(cell, SWEEP_LOW[0], grip=[1.0, 0.0], ticks=60)
    k = 0
    while not st["shift_over"] and not st["stop"]["locked_out"]:
        k += 1
        for q, n in zip(SWEEP_HIGH if k % 2 else SWEEP_LOW, TICKS):
            st = go(cell, q, grip=[1.0, 0.0], ticks=n)
            if st["shift_over"] or st["stop"]["locked_out"]:
                break
    while not st["shift_over"]:
        st = cell.step(n=info["budgets"]["max_ticks_per_step"])
