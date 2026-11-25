"""Degenerate submission D3 (the grader's QC): primitives drawn with a fixed seed (name and bottom face uniform over
the published sets) until the time limit; no report."""
import random

NAMES = ("roll_100_100", "roll_120_120", "roll_140_140", "roll_100_120", "roll_120_100", "roll_100_140",
         "roll_140_100", "roll_120_140", "roll_140_120", "ccw", "cw")
FACES = ((0, 2, 5), (0, 3, 5), (1, 2, 4), (1, 2, 5), (0, 3, 4), (1, 3, 4))


def solve(station):
    rng = random.Random(20261117)
    done = False
    while not done:
        station.run_primitive(rng.choice(NAMES), list(rng.choice(FACES)))
        while True:
            r = station.step(0.5)
            done = bool(r["done"])
            if done or not station.status()["busy"]:
                break
