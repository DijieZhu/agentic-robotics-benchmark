"""Review: what the Go1's mast camera shows along the way (an information audit, no solver).

    /opt/venv/bin/python3 -B tests/review/observability.py --n 16          # development sites
    /opt/venv/bin/python3 -B tests/review/observability.py --hidden
    /opt/venv/bin/python3 -B tests/review/observability.py --reference --n 8    # the cue colours' references

The oracle walks each site on its true pose. Every 0.4 m of travel this script draws the eight mast detents itself
(outside the mission's exposure budget, nothing is sent to the controller) and counts what the pictures show of the
bollard caps, the pallet and the dock beacon. A cap counts only if the picture shows it: the cap's centre is
projected into the image, and within its projected disc (plus 2 px) there has to be a connected patch of at least 6
pixels whose CIELAB distance to the cap colour's reference is at most 25. The hall's racks, lamps and ducts are in
these pictures; a cap behind something, in the dark or washed out does not count. The pallet counts when at least
150 pixels inside its projected outline are within 25 of its reference.

It answers whether a program restricted to the stated sensors has something to re-localise on wherever its odometry
has drifted (at least three caps at every sample), sees where the pallet stands before it has to choose an opening,
and sees the dock's beacon before it docks. The references (REFERENCE) are medians over development sites
(--reference prints them: the median colour of the inner half of every projected cap of at least 3 px radius).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK/"tests/runtime"), str(TASK/"tests/setup_payload"), str(TASK/"solution/oracle"),
                str(TASK/"tests/review")]
import mujoco  # noqa: E402
import oracle  # noqa: E402
from articulated import BEACON, MAST_FOV, PALLET_Z, POST_R, POST_Z  # noqa: E402
from controls import Local  # noqa: E402

# CIELAB medians on development sites 1-4 (1,107 caps, 76 beacon views, 174 pallet views): --reference --n 4 --start 1
REFERENCE = {"cap": (20.6, -28.6, 25.2), "beacon": (46.4, 30.2, -73.8), "pallet": (25.7, 46.5, -11.0)}
NEAR, LEAST, STACK_PX = 25., 6, 150


def lab(rgb):
    c = np.asarray(rgb, float)/255.
    c = np.where(c <= .04045, c/12.92, ((c+.055)/1.055)**2.4)
    m = np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]])
    xyz = c @ m.T/np.array([.95047, 1., 1.08883])
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787*xyz+16/116)
    return np.stack([116*f[..., 1]-16, 500*(f[..., 0]-f[..., 1]), 200*(f[..., 1]-f[..., 2])], -1)


class Watcher(Local):
    def __init__(self, seed, reference=False):
        super().__init__(seed)
        self.samples, self.mark, self.reference = [], None, reference
        self.colours = {"cap": [], "beacon": [], "pallet": []}

    def project(self, cam, point, radius):
        dog = self.w.go1
        axes = dog.data.cam_xmat[cam].reshape(3, 3)
        d = (np.asarray(point, float)-dog.data.cam_xpos[cam]) @ axes
        depth = -d[2]
        if depth < .25:
            return None
        k = 128/(depth*math.tan(math.radians(MAST_FOV)/2))
        return 128+k*d[0], 128-k*d[1], k*radius

    def seen(self, colour, kind, at):
        if at is None:
            return False
        x, y, r = at
        yy, xx = np.mgrid[0:256, 0:256]
        disc = (xx-x)**2+(yy-y)**2 <= (r+2)**2
        if not disc.any():
            return False
        if self.reference and r >= 3:
            core = (xx-x)**2+(yy-y)**2 <= (r/2)**2
            if core.sum() >= 4 and x-r > 2 and x+r < 253 and y-r > 2 and y+r < 253:
                self.colours[kind].append(np.median(colour[core], 0))
        like = disc & (np.linalg.norm(colour-np.asarray(REFERENCE[kind]), axis=2) <= NEAR)
        labels, n = ndimage.label(like)
        return bool(n > 0 and max(ndimage.sum(like, labels, range(1, n+1))) >= LEAST)

    def act(self, action):
        obs = super().act(action)
        dog = self.w.go1
        if dog is not None and (self.mark is None or np.linalg.norm(dog.xy-self.mark) >= .40):
            self.mark = dog.xy.copy()
            posts = [p for p in self.w.posts if p is not None]
            beacon_at = (self.w.goal[0]+BEACON[0], self.w.goal[1]+BEACON[1], POST_Z)
            x0, x1, y0, y1 = self.w.pallet
            caps = stack = beacon = 0
            for detent in range(8):
                colour = lab(dog.picture(detent))
                cam = dog.mast_ids[detent]
                caps += sum(self.seen(colour, "cap", self.project(cam, (*p, POST_Z), POST_R)) for p in posts)
                beacon += int(self.seen(colour, "beacon", self.project(cam, beacon_at, POST_R)))
                corners = [self.project(cam, (x, y, z), 0.) for x in (x0, x1) for y in (y0, y1) for z in (.05, PALLET_Z)]
                if all(c is not None for c in corners):
                    cx, cy = np.array([c[:2] for c in corners]).T
                    box = np.zeros((256, 256), bool)
                    box[max(0, int(cy.min())):max(0, int(cy.max())+1), max(0, int(cx.min())):max(0, int(cx.max())+1)] = True
                    near = box & (np.linalg.norm(colour-np.asarray(REFERENCE["pallet"]), axis=2) <= NEAR)
                    stack += int(near.sum())
                    if self.reference and box.sum() >= 400:
                        self.colours["pallet"].append(np.median(colour[box], 0))
            self.samples.append({"x": float(dog.xy[0]), "caps": caps, "stack_px": stack, "beacon": beacon})
        return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--reference", action="store_true", help="development sites only: print the cue colours' medians")
    args = parser.parse_args()
    graded = json.loads((TASK/"tests/hidden/sites.json").read_text())["seeds"]
    draws = [int(x) for x in np.random.default_rng(20261120).integers(1, 2**62, size=args.start+args.n)]
    if args.hidden and args.reference:
        sys.exit("the references are taken from development sites")
    seeds = graded if args.hidden else [s for s in draws if s not in set(graded)][args.start:]
    worst = {"caps_min": 99, "stack_seen_before_middle_rack": 1, "beacon_seen_before_dock": 1}
    colours = {"cap": [], "beacon": [], "pallet": []}
    for i, seed in enumerate(seeds):
        robot = Watcher(seed, args.reference)
        oracle.mission(robot, "review")
        middle = robot.w.racks[1].near
        before = [s for s in robot.samples if s["x"] < middle-.45]
        last = [s for s in robot.samples if s["x"] > robot.w.racks[2].far+.3]
        row = {"site": i, "outcome": robot.w.go1.outcome, "samples": len(robot.samples),
               "caps_min": min(s["caps"] for s in robot.samples),
               "caps_median": int(np.median([s["caps"] for s in robot.samples])),
               "stack_seen_before_middle_rack": int(any(s["stack_px"] > STACK_PX for s in before)),
               "beacon_seen_before_dock": int(any(s["beacon"] for s in last))}
        print(json.dumps(row), flush=True)
        worst["caps_min"] = min(worst["caps_min"], row["caps_min"])
        for key in ("stack_seen_before_middle_rack", "beacon_seen_before_dock"):
            worst[key] = min(worst[key], row[key])
        for kind in colours:
            colours[kind].extend(robot.colours[kind])
        robot.w.close()
    if args.reference:
        print(json.dumps({"reference": {k: [round(float(v), 1) for v in np.median(c, 0)] if c else None for k, c in colours.items()},
                          "instances": {k: len(c) for k, c in colours.items()}}), flush=True)
    else:
        print(json.dumps({"sites": len(seeds), "worst": worst}), flush=True)


if __name__ == "__main__":
    main()
