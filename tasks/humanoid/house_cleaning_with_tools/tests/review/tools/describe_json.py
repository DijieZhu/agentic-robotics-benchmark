import json, os, sys
stage, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, stage); sys.dont_write_bytecode = True
from house_cell.server import CellServer
cfg = json.load(open(os.path.join(stage, "grader_config.json")))
a = CellServer(11, config=cfg, render=False).handle("describe", {})
b = CellServer(12345, config=cfg, render=False).handle("describe", {})
assert a == b, "describe() depends on the seed"
open(out, "w").write(json.dumps(a, indent=1, sort_keys=True) + "\n")
print("ok", len(json.dumps(a)))
