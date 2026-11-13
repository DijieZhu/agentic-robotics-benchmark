"""Copy the canonical appliance-tending code and unit models into the task folder (P3).

The task's copies must be byte-identical to these sources (the dev sim, the lab, the grader and the
reference run the same environment file). Hidden units: the grader (tests/units), the lab
(environment/setup/ref/units) and the reference (solution/oracle/units); practice units: the agent's
home (environment/setup/payload/units). The selection is tests/hidden/selection.json's `selected`.

    python3 -I sync_task.py --assets ~/arb-data/robocasa-assets/extracted/fixtures
"""
import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tasks", "arm", "kitchen_appliance_tending"))
KIND = {"stovetops": "stovetop", "microwaves": "microwave", "ovens": "oven", "stoves": "stove"}
CODE = {
    "environment/setup/payload": ["appliance_env.py", "episode.py"],
    "environment/setup/lab": ["appliance_env.py", "episode.py"],
    "tests": ["appliance_env.py", "episode.py"],
    "solution/oracle": ["appliance_env.py", "reference.py", "oracle_main.py"],
}


def copy_units(assets, picks, dest_dir):
    units_dir = os.path.join(TASK, dest_dir, "units")
    if os.path.isdir(units_dir):
        shutil.rmtree(units_dir)
    table = {}
    for uid, model in picks:
        name = model.split("/")[1]
        shutil.copytree(os.path.join(assets, model), os.path.join(units_dir, name))
        table[uid] = {"model": model, "kind": KIND[model.split("/")[0]], "dir": "units/" + name}
    with open(os.path.join(TASK, dest_dir, "units.json"), "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    a = ap.parse_args()
    assets = os.path.expanduser(a.assets)
    for dest, names in CODE.items():
        os.makedirs(os.path.join(TASK, dest), exist_ok=True)
        for n in names:
            shutil.copyfile(os.path.join(HERE, n), os.path.join(TASK, dest, n))
    sel = json.load(open(os.path.join(TASK, "tests", "hidden", "selection.json")))["selected"]
    order = ("cooktop", "microwave", "oven")
    # `hidden` is a list per kind (two since 2026-11-13): u0-u2 are the first entries in kind order, u3-u5 the
    # second entries, so the original three keep their ids.
    per_kind = [sel[k]["hidden"] if isinstance(sel[k]["hidden"], list) else [sel[k]["hidden"]] for k in order]
    hidden = [("u%d" % (r * len(order) + i), models[r])
              for r in range(max(len(m) for m in per_kind)) for i, models in enumerate(per_kind) if r < len(models)]
    practice = [("p%d" % i, sel[k]["practice"]) for i, k in enumerate(order)]
    for dest in ("tests", "environment/setup/ref", "solution/oracle"):
        copy_units(assets, hidden, dest)
    copy_units(assets, practice, "environment/setup/payload")
    print("synced into", TASK)


if __name__ == "__main__":
    main()
