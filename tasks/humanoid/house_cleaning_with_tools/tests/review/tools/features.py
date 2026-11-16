"""Layout features of candidate scenes for the hollowing check (no cell, no render): features.py STAGE SEED [SEED...]"""
import json, math, sys
stage = sys.argv[1]
sys.path.insert(0, stage); sys.dont_write_bytecode = True
from house_cell import layout as L
from house_cell import room as RM
D = RM.DINING
(dx, dy), (lx, ly) = D["center"], D["size"]
out = {}
for s in sys.argv[2:]:
    lay = L.draw(int(s))
    ob = lay["objects"]
    def edge(name):
        p = ob[name][0]
        return round(min(p[1] - (dy - ly / 2), (dy + ly / 2) - p[1]), 3)
    g = lay["grease"]
    floor = [c for c in lay["crumbs"] if c[2] < 0.05]
    south = sum(1 for c in floor if c[1] < dy)
    out[s] = {"bins_recycling_slot": 0 if lay["bins"][0] == "recycling" else 1,
              "glass_edge_m": [edge("glass_1"), edge("glass_2")], "can_red_edge_m": edge("can_red"),
              "grease_total": round(sum(g["g0"]) * (g["cell"] / 0.025) ** 2, 1),
              "floor_crumbs_south": south, "floor_crumbs_north": len(floor) - south,
              "robot": [round(v, 3) for v in lay["robot"]]}
print(json.dumps(out, indent=1))
