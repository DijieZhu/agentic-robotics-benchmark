"""Isolated loading fixtures: instantaneous geometry and an empty physical flight."""
import argparse
import json
import math

from check_litter import OnDemandWorld
from check_rescue import Record
from oracle_run import Pilot, TRAVEL, transfer
from rescue_scene import POCKET_HALF_DEPTH
import mujoco
import numpy as np


def check(seed):
    w = OnDemandWorld("rescue", seed)
    try:
        cx, cy = w.layout.pocket
        offset, side, stagger = w.layout.loading
        margins = {}
        for name, center, axis, baseline in (
            ("throat_entry", (cx+offset, cy+POCKET_HALF_DEPTH), 1, cx),
            ("throat_exit", (cx+side*2.10, cy+side*stagger), 0, cy),
        ):
            blocks = w.solids[np.array(w.kinds) == name]
            def distance(degrees, lateral):
                angle = math.radians(degrees)+(math.pi/2 if axis == 0 else 0.)
                xy = np.asarray(center).copy()
                xy[1-axis] += lateral
                w.data.qpos[4:11] = (*xy, 1.1, math.cos(angle/2), 0., 0., math.sin(angle/2))
                mujoco.mj_forward(w.model, w.data)
                return min(mujoco.mj_geomDistance(w.model, w.data, w.board, g, 1., None) for g in blocks)
            aligned = distance(0., 0.)
            offset_pose = distance(0., baseline-center[1-axis])
            perturbed = min(distance(a, x) for a in (-10., 10.) for x in (-.07, .07))
            broadside = distance(90., 0.)
            assert aligned > .30 and perturbed > .09 and broadside < 0 and offset_pose < 0
            margins[name] = dict(aligned_m=aligned, ten_deg_seven_cm_m=perturbed,
                                 broadside_m=broadside, unshifted_m=offset_pose)
        # Centering alone is insufficient if the ends overhang the receiving mat.
        w.data.qpos[4:7] = (*w.layout.receiving_pad, .073)
        w.data.qpos[7:11] = (1., 0., 0., 0.)
        mujoco.mj_forward(w.model, w.data)
        assert w.mat_margin() < 0.
        w.data.qpos[7:11] = (math.cos(math.pi/4), 0., 0., math.sin(math.pi/4))
        mujoco.mj_forward(w.model, w.data)
        assert w.mat_margin() > .30
        # Fresh world below. The pose probes above are not episodes or task outcomes.
    finally:
        w.close()
    w = OnDemandWorld("rescue", seed)
    try:
        x, y = w.layout.xy((w.layout.outlet, 0))
        w.command[:] = (x, y, TRAVEL, 0.)
        w.data.qpos[:4] = w.data.ctrl[:] = w.command
        mujoco.mj_forward(w.model, w.data)
        pilot = Pilot(Record(w))
        pilot.litter_relative = math.pi/2
        transfer(pilot, hold_heading=True)
        assert not w.failure and side*(w.position[0]-cx) > 3.0, (w.failure, w.position)
        print(json.dumps({"seed": str(seed), "empty_fixed_heading_passed": True,
                          "actions": w.steps, "geometry": margins}), flush=True)
    finally:
        w.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", type=int, nargs="+")
    for seed in parser.parse_args().seeds:
        check(seed)
