"""hello: drive one lap of the practice session on the LiDAR alone, then open and close the other sessions.

The cell's wire gate: a program in its own process that receives a `lap` event. It balances the free space
on the left and on the right of the car (20th percentile of the ranges in two sectors) at 1.5 m/s, below the mapping
phase's limit. It is not a racing program and is not given to agents.
"""

import math
import sys

from carkit import SessionLog
from carkit import scan as S

SPEED = 1.5
GAIN = 1.6


def sector(ranges, deg0, deg1):
    i0, i1 = S.beam_index(math.radians(deg0)), S.beam_index(math.radians(deg1))
    vals = sorted((v if v > 0 else 10000) for v in ranges[i0:i1 + 1])
    return vals[len(vals) // 5] / 1000.0


def solve(car):
    info = car.describe()
    ticks = info["sessions"][0]["ticks"]
    car.open_session("practice")
    log = SessionLog("practice")
    steer = 0.0
    for _ in range(ticks):
        r = car.step(steer, SPEED)
        log.feed(r)
        if log.laps or log.ended:
            break
        ranges = [int(v) for v in r["scan"]]
        left, right = sector(ranges, 30, 80), sector(ranges, -80, -30)
        steer = max(-0.4189, min(0.4189, GAIN * (left - right) / max(0.2, left + right)))
    print("hello: lap events %s at tick %d" % ([e for _, e in log.history if e["type"] == "lap"], log.tick), file=sys.stderr)
    car.close_session()
    for name in ("tt1", "tt2", "avoid"):
        car.open_session(name)
        car.close_session()
