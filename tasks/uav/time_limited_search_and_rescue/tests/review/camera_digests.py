"""Are the cameras' pictures a function of the site and the state alone?

Run inside the task image, as root (the render bundle is root's), from anywhere:

    /opt/venv/bin/python3 -I tests/review/camera_digests.py [--seeds 8009,8018] [--fresh 3] [--dump FILE] [--expect FILE]

For each seed (development seeds; never a graded site) a list of made-up states: the aircraft at the launch point and
over the centre of every room cell of both storeys, at flight height, on four headings, the litter and the collets
where the site puts them. One worker process (rescue_render.Cameras) draws them in order and then again in reverse
(an image must not depend on what was drawn before it); --fresh further worker processes draw them again (an image
must not depend on the process). Prints how many images differed in each comparison (all must be 0) and writes
the images' SHA-256 digests with --dump; --expect compares them with the digests of another run, for example another
host's.
"""
import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "runtime"))
import numpy as np  # noqa: E402
from rescue_render import Cameras  # noqa: E402
from rescue_scene import DECK, SIDE, START_Z, layout_for  # noqa: E402


def states(seed):
    """(name, qpos, mocap positions) of the made-up states of a site; qpos as rescue.World starts it."""
    import mujoco
    from rescue_scene import build
    layout = layout_for(seed >> 1)
    xml, decals, _obstacles, _shutters = build(layout)
    model = mujoco.MjModel.from_xml_string(xml, assets=decals)
    data = mujoco.MjData(model)
    x, y, yaw = layout.pod
    data.qpos[:4] = (*layout.launch, START_Z, 0.)
    data.qpos[4:11] = (x, y, layout.target[2]*DECK+.073, math.cos(yaw/2), 0., 0., math.sin(yaw/2))
    base, mocap = data.qpos.copy(), data.mocap_pos.copy()
    out = []
    for k in range(4):
        q = base.copy()
        q[3] = k*math.pi/2+.3
        out.append((f"launch_{k}", q, mocap))
    for level in (0, 1):
        for cx in range(SIDE):
            for cy in range(1, SIDE):
                for k in range(4):
                    q = base.copy()
                    q[0], q[1] = layout.xy((cx, cy))
                    q[2], q[3] = level*DECK+1.3, k*math.pi/2+.2*cx-.1*cy
                    out.append((f"cell_{level}_{cx}_{cy}_{k}", q, mocap))
    return out


def digests(seed, todo):
    cameras = Cameras(seed)
    out = {}
    for name, q, mocap in todo:
        down, forward = cameras.images(q, mocap)
        out[f"{seed}/{name}/down"] = hashlib.sha256(down.tobytes()).hexdigest()
        out[f"{seed}/{name}/forward"] = hashlib.sha256(forward.tobytes()).hexdigest()
    return cameras, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="8009,8018")
    parser.add_argument("--fresh", type=int, default=3)
    parser.add_argument("--dump")
    parser.add_argument("--expect")
    args = parser.parse_args()
    all_digests, report = {}, {}
    for seed in (int(s) for s in args.seeds.split(",")):
        if 39000000 <= seed < 39001000:
            raise SystemExit("development seeds only")
        todo = states(seed)
        cameras, first = digests(seed, todo)
        again = {}
        for name, q, mocap in reversed(todo):
            down, forward = cameras.images(q, mocap)
            again[f"{seed}/{name}/down"] = hashlib.sha256(down.tobytes()).hexdigest()
            again[f"{seed}/{name}/forward"] = hashlib.sha256(forward.tobytes()).hexdigest()
        cameras.close()
        fresh = []
        for i in range(args.fresh):
            other, got = digests(seed, todo if i % 2 else list(reversed(todo)))
            other.close()
            fresh.append(sum(first[k] != got[k] for k in first))
        report[seed] = {"images": len(first), "distinct": len(set(first.values())),
                        "redrawn_differ": sum(first[k] != again[k] for k in first), "fresh_processes_differ": fresh}
        all_digests.update(first)
    result = {"report": report, "cpu": next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                                             if line.startswith("model name")), "?")}
    if args.expect:
        other = json.load(open(args.expect))["digests"]
        result["other_run_differ"] = sum(all_digests[k] != other.get(k) for k in all_digests)
        result["other_run_cpu"] = json.load(open(args.expect)).get("cpu")
    print(json.dumps(result, indent=1))
    if args.dump:
        with open(args.dump, "w") as handle:
            json.dump({"digests": all_digests, **result}, handle, indent=1)
    bad = any(r["redrawn_differ"] or any(r["fresh_processes_differ"]) for r in report.values()) or result.get("other_run_differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
