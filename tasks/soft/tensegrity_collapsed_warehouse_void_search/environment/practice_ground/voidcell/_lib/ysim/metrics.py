"""SE(2) pose and primitive displacement exactly as the authors compute their primitive table
(ChemStud24/tensegrity rss25 src/reformat_dictionary.py): planar centroid of the six end-cap centres;
heading = atan2 of the planar principal axis mean(nodes 0,2,4) - mean(nodes 1,3,5); displacement expressed in the
start frame (x = start principal axis, y = x turned +90 deg, z up); heading change positive counter-clockwise seen
from above. Verified: this formula reproduces the published table from the published primitive dict to 0 m / 0 deg
(notes/sim_base_data/authors_tables.json, rederive_check).
"""
import math

import numpy as np


def se2(nodes):
    p = np.asarray(nodes)[:, :2]
    ax = (p[0] + p[2] + p[4]) / 3.0 - (p[1] + p[3] + p[5]) / 3.0
    return p.mean(axis=0), math.atan2(ax[1], ax[0])


def relative(nodes0, nodes1):
    """(dx_local, dy_local, dtheta_deg, distance) from the start to the end node sets."""
    c0, th0 = se2(nodes0)
    c1, th1 = se2(nodes1)
    t = c1 - c0
    R0 = np.array([[math.cos(th0), -math.sin(th0)], [math.sin(th0), math.cos(th0)]])
    tl = R0.T @ t
    dth = math.degrees((th1 - th0 + math.pi) % (2 * math.pi) - math.pi)
    return float(tl[0]), float(tl[1]), dth, float(np.hypot(*tl))


def bottom3(nodes):
    """Three lowest end caps (run_tensegrity_Astar.py bottom3: argpartition of z)."""
    z = np.asarray(nodes)[:, 2]
    return tuple(sorted(int(i) for i in np.argsort(z, kind="stable")[:3]))
