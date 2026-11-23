"""Author controls for local visual odometry and interchangeable rescue fittings."""
import argparse
import base64
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(TASK/"environment/setup/runtime"), str(Path(__file__).resolve().parent)]
from oracle_run import run
from rescue import World


class Record:
    """The controller receives only the public envelope; truth is logged here."""
    def __init__(self, world, capture=False):
        self.world = world
        self.actions, self.frames, self.poses = [], {}, {}
        self.capture = capture

    def save(self, obs):
        step = obs["frame_id"]
        self.poses[step] = self.world.position.copy()
        if self.capture:
            self.frames[step] = obs["images"]
        return obs

    def observe(self):
        return self.save(self.world.observe())

    def act(self, action):
        self.actions.append(action.copy())
        return self.save(self.world.act(action))

    def export(self, step, path, camera):
        path.write_bytes(base64.b64decode(self.frames[step][camera]))


def check_known_returns(trace):
    """The replayed physical doorway trail must take the aircraft back to its bay."""
    found = next(i for i,e in enumerate(trace) if e["event"]=="sled_found")
    depart = next(i for i,e in enumerate(trace) if e["event"]=="depot_departure")
    back = next(i for i,e in enumerate(trace) if e["event"]=="sled_reacquired")
    bay = tuple(trace[found]["cell"])
    assert tuple(trace[back]["cell"])==bay
    def rooms(events):
        return [tuple(e["cell"]) for e in events if e["event"]=="station_fix" and len(e["cell"])==3]
    outbound = rooms(trace[found:depart])
    return_path = rooms(trace[depart:back])
    assert outbound[0]==bay and return_path==outbound[::-1],(outbound,return_path)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("seed",type=int)
    args=parser.parse_args()
    world=World("rescue",args.seed)
    try:
        assert world.metrics()["rescue_progress"]==0
        wrong="narrow" if world.layout.flange>.06 else "wide"
        pilot=run(Record(world),fixed_tool=wrong)
        assert world.fit_misses and not world.metrics()["grasped"]
        print(json.dumps(dict(untouched=0,seed=str(args.seed),fixed=wrong,**world.metrics())))
    finally:
        world.close()
