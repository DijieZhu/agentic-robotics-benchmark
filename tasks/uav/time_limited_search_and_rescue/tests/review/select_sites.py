"""Apply tests/review/selection.md: fly the privileged pilot on the candidates and write tests/hidden/.

Run inside the task image from the task folder:  /opt/venv/bin/python3 tests/review/select_sites.py [--extend N]
"""
import json
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK/"tests"/"runtime"), str(TASK/"solution"/"oracle")]
from rescue import World  # noqa: E402
import pilot as P  # noqa: E402

BASE, COUNT, RULE = 39000000, 96, 2


class InProcess:
    def __init__(self, world):
        self.world = world

    def observe(self):
        return self.world.observe()

    def act(self, action):
        return self.world.act(action)


def fly(seed):
    world = World("rescue", seed, render=False, truth=True)
    try:
        P.Pilot(InProcess(world)).fly()
    except P.Stop:
        pass
    metrics = world.metrics()
    flange_large = int(world.layout.flange > .06)
    world.close()
    return {"seed": seed, "stratum": [(seed >> 1) & 1, (seed >> 2) & 1, flange_large],
            "delivered": int(metrics["delivered"]), "actions": int(metrics["steps"])}


def main():
    count = COUNT*(1+int(sys.argv[sys.argv.index("--extend")+1])) if "--extend" in sys.argv else COUNT
    candidates = [fly(BASE+k) for k in range(count)]
    done = [c for c in candidates if c["delivered"]]
    strata = [[m, s, f] for m in (0, 1) for s in (0, 1) for f in (0, 1)]
    sites, sizes = [], []
    for i, stratum in enumerate(strata):
        members = sorted((c for c in done if c["stratum"] == stratum), key=lambda c: (c["actions"], c["seed"]))
        sizes.append(len(members))
        if not members:
            raise SystemExit(f"stratum {stratum} is empty: extend the range (--extend 1) as the rule says")
        sites.append(members[int((i+.5)/8*len(members))]["seed"])
    hidden = TASK/"tests"/"hidden"
    hidden.mkdir(exist_ok=True)
    (hidden/"candidates.json").write_text(json.dumps(candidates, indent=1)+"\n")
    (hidden/"sites.json").write_text(json.dumps({"seeds": sites}, indent=1)+"\n")
    (hidden/"selection.json").write_text(json.dumps({
        "rule": "tests/review/selection.md", "rule_version": RULE, "candidates": len(candidates),
        "reference_completed": len(done), "stratum_sizes": sizes, "sites": len(sites)}, indent=1)+"\n")
    print(json.dumps({"candidates": len(candidates), "completed": len(done), "strata": sizes}))


if __name__ == "__main__":
    main()
