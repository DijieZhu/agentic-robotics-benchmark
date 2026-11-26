"""One hidden test in its own process (run by verify.py): reads {"material", "fabric", "index"} on stdin, prints
{"q", "u", "predicted", "real"} as one JSON line. A crash of the simulator ends this process only."""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cgrading as CG  # noqa: E402


def main():
    job = json.load(sys.stdin)
    ref_f = CG.load_ref(os.path.join(HERE, "ref", "fabrics.json"))[job["fabric"]]
    q, info = CG.score_test(job["material"], ref_f, ref_f["tests"][job["index"]])
    out = {"q": q}
    for k in ("u", "predicted", "real"):
        v = info.get(k)
        out[k] = v if isinstance(v, (int, float)) else None
    if "error" in info:
        out["error"] = info["error"]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
