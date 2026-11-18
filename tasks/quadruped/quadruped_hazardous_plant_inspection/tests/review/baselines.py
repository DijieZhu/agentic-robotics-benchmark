"""The degenerate and document-trusting baselines as real submissions.

    python -I tests/review/baselines.py make NAME OUT_DIR [--task DIR] write the submission folder of one baseline (from
                                                                      the modules of DIR, default this task folder)
    MUJOCO_GL=osmesa python -I tests/review/baselines.py run --site SITE --out DIR [--names a,b,...] [--seed N]
                                                                      make each and run it over the wire (run_wire)

Baselines (mission/baselines.py; none has a privilege: the robot's public calls only, through plantcell.client):
  idle                         stands, then finishes                                        design target 0.00
  walk_out_and_back            walks 5 m into the building and back                         0.00
  last_logged_values           reports the round sheet's 2009 values from the pad, no frame 0.00
  constant_values              one constant per item from the pad, no frame                 0.00
  corridor_only                stays where the link is up; reads corridor instruments       at most 0.10
  trust_the_drawing            walk only, the drawing and the odometry                      about 0.10
  trust_the_drawing_all_modes  the same, trying high_step and crouch_walk where walk stops  (as measured)

A submission folder is patrol.py plus the modules the baseline imports (its navigation, the plant catalogue, the
reference readers it reads with), copied from the task's vendored libraries as listed in
review/_lib/baselines_closure.json. The instrument modules' lazy reader import (a computed import, which the grader's
import audit refuses) becomes a literal import in the submission's copy; `make` then checks the folder against the
grader's static gates (the route gate, cellkit.netaudit) and refuses to write a submission that would fail them, so a
baseline measured through the real verify stage scores its behaviour, never a contract failure.
"""

import argparse
import ast
import json
import os
import shutil
import sys

REVIEW = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(REVIEW)
TASK = os.path.dirname(TESTS)
NAMES = ("idle", "walk_out_and_back", "last_logged_values", "constant_values", "corridor_only", "trust_the_drawing",
         "trust_the_drawing_all_modes")
PATROL = '''"""Baseline submission `{name}` (tests/review/baselines.py): mission/baselines.py's `{name}`, unprivileged.
Its own log (what it decided and why) goes to stderr, the program log of the mission."""

import json
import sys

from mission import baselines


def solve(robot):
    log = baselines.{name}(robot)
    if log:
        for row in log:
            print("[baseline] " + json.dumps(row, default=str)[:400], file=sys.stderr)
'''


LAZY_READER_CALL = 'importlib.import_module(_READER_MODULE, __package__.rpartition(".")[0])'


def _literal_reader_import(path, modname):
    """The vendored instrument modules (bricks.targets.*) resolve their reader names lazily through
    importlib.import_module(_READER_MODULE, <package>): a computed import, which keeps the readers out of the cell's
    static import closure but which the grader's import audit refuses in a submission
    (cellkit.netaudit: `import_module(<computed name>)`). A program an agent writes imports by literal names, so the
    submission's copy of such a module imports the same reader module by its literal, absolute name. Returns the name,
    or None when the module has no lazy reader import."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "_READER_MODULE" not in src:
        return None
    rel = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_READER_MODULE" for t in node.targets) \
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            rel = node.value.value
    if rel is None or LAZY_READER_CALL not in src:
        raise SystemExit("%s: a lazy reader import of an unexpected form (update tests/review/baselines.py)" % path)
    base = modname.rpartition(".")[0].rpartition(".")[0]          # __package__.rpartition(".")[0] of the module
    level = len(rel) - len(rel.lstrip("."))
    parts = base.split(".")
    absname = ".".join(parts[:len(parts) - (level - 1)] + [rel.lstrip(".")]) if level else rel
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(LAZY_READER_CALL, "importlib.import_module(%r)" % absname))
    return absname


def admissible(out):
    """Problems the grader's static gates would find in the submission folder `out` (tests/verify.py: the route gate's
    forbidden calls and file types, the import audit of cellkit.netaudit); [] when it passes them."""
    sys.path.insert(0, TESTS)
    from cellkit import netaudit                      # noqa: E402 (tests/cellkit)
    with open(os.path.join(TESTS, "grader_config.json")) as f:
        rg = json.load(f).get("route_gate", {})
    probs = ["import audit: %s: %s" % hit for hit in netaudit.scan_imports(out)]
    for dp, _dns, fns in os.walk(out):
        for fn in fns:
            p = os.path.join(dp, fn)
            if fn.lower().endswith(tuple(rg.get("forbidden_extensions", []))):
                probs.append("route gate: weight file %s" % os.path.relpath(p, out))
            if fn.endswith((".py", ".sh")):
                text = open(p, encoding="utf-8", errors="replace").read()
                probs += ["route gate: %s uses %s" % (os.path.relpath(p, out), c) for c in rg.get("forbidden_calls", [])
                          if c in text]
    return probs


def make(name, out, task=None):
    """Write the submission folder of one baseline into `out` from the modules of `task` (default: this task folder)."""
    if name not in NAMES:
        raise SystemExit("unknown baseline %r (%s)" % (name, ", ".join(NAMES)))
    task = os.path.abspath(task or TASK)
    with open(os.path.join(task, "tests", "review", "_lib", "baselines_closure.json")) as f:
        where = json.load(f)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    pkgs = set()
    for mod, rel in sorted(where.items()):
        src = os.path.join(task, rel)
        parts = mod.split(".")
        is_pkg = os.path.basename(rel) == "__init__.py"
        dst = os.path.join(out, *(parts + ["__init__.py"] if is_pkg else parts[:-1] + [parts[-1] + ".py"]))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        if not is_pkg:
            _literal_reader_import(dst, mod)
        pkgs.update(".".join(parts[:k]) for k in range(1, len(parts)))
    for p in pkgs:                                   # every package level has its __init__ (copied above)
        init = os.path.join(out, *p.split("."), "__init__.py")
        if not os.path.isfile(init):
            raise SystemExit("package %s has no __init__ in the closure" % p)
    with open(os.path.join(out, "patrol.py"), "w") as f:
        f.write(PATROL.format(name=name))
    probs = admissible(out)
    if probs:
        raise SystemExit("the submission %s would fail the grader's static gates: %s" % (out, "; ".join(probs[:5])))
    return out


def run_all(site, out, names, seed=None):
    sys.path.insert(0, os.path.join(REVIEW, "cell"))
    sys.path.insert(0, TESTS)
    import run_wire                                   # noqa: E402 (tests/review/cell/run_wire.py)
    rows = []
    for n in names:
        sub = make(n, os.path.join(out, "submissions", n))
        s = run_wire.run(site, sub, os.path.join(out, n), privileged=False, seed=seed,
                         log=lambda m: print(m, file=sys.stderr, flush=True))
        row = {"baseline": n, "episode_score": s["episode_score"], "R": s["R"], "parts": s["parts"],
               "done_reason": s["done_reason"], "end_reason": s.get("end_reason"), "sim_time_s": s["sim_time_s"],
               "program_time_s": round(float(s["program_time_s"]), 1), "episode_wall_s": s.get("episode_wall_s"),
               "reports_made": s["reports_made"], "reports_delivered": s["reports_delivered"],
               "illegal_call_count": s["illegal_call_count"], "internal_errors": s["internal_errors"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(out, "baselines.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("name")
    m.add_argument("out")
    m.add_argument("--task", default=None, help="take the modules from this task folder (default: this one)")
    r = sub.add_parser("run")
    r.add_argument("--site", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--names", default=",".join(NAMES))
    r.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        print(make(a.name, a.out, a.task))
        return 0
    run_all(a.site, a.out, [n for n in a.names.split(",") if n], a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
