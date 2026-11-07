"""Build the mission pool of a set of TravelUAV scenes. Run at image build time.

Inputs: AeroVLA's per-map object placement areas (`map_spawnarea_info.json`: each area is
a list whose items 9-11 are a placement point in the scene's NED frame, 12-15 the placement
rotation as w, x, y, z) and this task's target catalogue (`targets.json`). Output:

    {"maps": {"<map>": {"slots": [[x, y, z, qw, qx, qy, qz], ...],
                        "targets": [{"asset", "category", "description"}, ...]}}}

Missions are drawn from this at run time (`sar_protocol.assemble_missions`): a mission puts
target assets on slots of one map and gives the drone their descriptions.

stdlib only. Used by both images (the practice pool and the held-out pool).
"""

import argparse
import json
import math


def build(areas, catalogue, maps, min_spacing):
    pool = {"maps": {}}
    for name in maps:
        if name not in areas:
            raise SystemExit("no placement areas for map {}".format(name))
        slots = []
        for area in areas[name]:
            point = [float(v) for v in area[9:12]]
            if any(math.hypot(point[0] - s[0], point[1] - s[1]) < min_spacing for s in slots):
                continue            # two slots this close would let one object hide another
            slots.append(point + [float(v) for v in area[12:16]])
        targets = [{"asset": t["asset"], "category": t["category"], "description": t["description"]}
                   for t in catalogue["targets"] if t["maps"] == "all" or name in t["maps"]]
        pool["maps"][name] = {"slots": slots, "targets": targets}
    return pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn-areas", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--maps", nargs="+", required=True)
    parser.add_argument("--min-spacing", type=float, default=4.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.spawn_areas) as handle:
        areas = json.load(handle)
    with open(args.targets) as handle:
        catalogue = json.load(handle)
    pool = build(areas, catalogue, args.maps, args.min_spacing)
    for name, entry in pool["maps"].items():
        print("[pool] {}: {} slots, {} target kinds".format(
            name, len(entry["slots"]), len(entry["targets"])))
    with open(args.out, "w") as handle:
        json.dump(pool, handle)


if __name__ == "__main__":
    main()
