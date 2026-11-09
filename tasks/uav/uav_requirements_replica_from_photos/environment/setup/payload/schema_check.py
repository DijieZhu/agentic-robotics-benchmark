#!/usr/bin/env python3
import json,sys,math
from pathlib import Path
import yaml,jsonschema
root=Path(__file__).resolve().parent
for name in ("requirements.yaml","bom.json","kinematic_sketch.json"):
    p=Path(sys.argv[1])/name
    doc=yaml.safe_load(p.read_text()) if name.endswith("yaml") else json.loads(p.read_text())
    jsonschema.validate(doc,json.loads((root/"schemas"/(name.split(".")[0]+".schema.json")).read_text()))
    def finite(x):
        if isinstance(x,float) and not math.isfinite(x): raise ValueError("nonfinite number")
        if isinstance(x,dict):
            for v in x.values(): finite(v)
        if isinstance(x,list):
            for v in x: finite(v)
    finite(doc)
print("Structure OK; this does not evaluate engineering correctness.")
