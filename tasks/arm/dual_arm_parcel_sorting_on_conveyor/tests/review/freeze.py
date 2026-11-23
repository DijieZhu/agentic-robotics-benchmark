"""The frozen generator and scoring configuration (a maintainer's tool; the grader never runs it). Standard library only.

    python3 -B tests/review/freeze.py check      # exit 1 when a frozen file, a version or a digest differs
    python3 -B tests/review/freeze.py write      # only when a new hidden set is selected and a new anchor measured

tests/frozen.json pins, by sha256:
  shift_and_score   the files that define a shift and its score: the generator (parcelcell/shiftgen.py), the shift
                    and parcel types (parcelcell/stream.py), the parcel classes, credit and shift score
                    (parcelcell/episode.py), the scan-line judgement (parcelcell/truth.py), the damage and stop rules
                    (parcelcell/safety.py), the configuration's loader and defaults (parcelcell/config.py) and the
                    configuration itself (grader_config.json: belt classes, population, arrival gap, fixed
                    durations, budgets, safety thresholds, scoring clauses, duty);
  cell              every file the hidden set's admission and the anchor depend on: the whole cell package
                    (parcelcell/, less the robot's and the gripper's mesh files, which the image fetches from the
                    task's pinned asset list), the reference program (reference/) and grader_config.json.
Each group's digest is the sha256 of its lines "<sha256>  <path>\\n" in path order. The versions recorded beside them
(the generators' GENERATOR_VERSION and LINE_GENERATOR_VERSION, PACKAGE_FORMAT, the cell's CELL_VERSION) must also be
unchanged.

  rated_set         the cell files the rated set (hidden/seeds.json, hidden/selection.json) was selected on (the
                    selection's own record, copied at ``write``) and the identity check that carries it to this
                    cell: the reference's records on the rated set's eight shifts, run on both, identical field by
                    field but the cell's version string (``identity``).

The line set (hidden/line/) is drawn by its registered rule on the frozen files of this manifest; the anchor
(anchor.json) was measured on the rated set. A change to a file of ``shift_and_score`` or ``cell`` needs the
identity check of the rated set again and, once the line set is drawn, a new draw.
"""

import ast
import hashlib
import json
import os
import sys

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(TESTS, "frozen.json")
FROZEN_VERSION = "frozen-2"
SHIFT_AND_SCORE = ("parcelcell/shiftgen.py", "parcelcell/stream.py", "parcelcell/episode.py", "parcelcell/truth.py",
                   "parcelcell/safety.py", "parcelcell/config.py", "grader_config.json")
CELL_ROOTS = ("parcelcell", "reference")
MESH_EXT = (".stl",)          # the fetched mesh files (lower-cased extension), outside the cell digest


def sha256_file(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def digest(files):
    return hashlib.sha256("".join("%s  %s\n" % (h, rel) for rel, h in sorted(files.items())).encode()).hexdigest()


def cell_files(tests=TESTS):
    out = {}
    for root in CELL_ROOTS:
        for dp, dn, fn in os.walk(os.path.join(tests, root)):
            dn[:] = sorted(d for d in dn if d != "__pycache__")
            for f in sorted(fn):
                if f.lower().endswith(MESH_EXT) or f.endswith(".pyc"):
                    continue
                p = os.path.join(dp, f)
                out[os.path.relpath(p, tests)] = sha256_file(p)
    out["grader_config.json"] = sha256_file(os.path.join(tests, "grader_config.json"))
    return out


def _constant(path, name):
    with open(path) as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def versions(tests=TESTS):
    gen = os.path.join(tests, "parcelcell", "shiftgen.py")
    return {"generator": _constant(gen, "GENERATOR_VERSION"), "line_generator": _constant(gen, "LINE_GENERATOR_VERSION"),
            "package_format": _constant(gen, "PACKAGE_FORMAT"),
            "cell": _constant(os.path.join(tests, "parcelcell", "__init__.py"), "CELL_VERSION")}


def build(tests=TESTS, identity=None):
    ss = {rel: sha256_file(os.path.join(tests, rel)) for rel in SHIFT_AND_SCORE}
    cf = cell_files(tests)
    out = {"frozen_version": FROZEN_VERSION, "versions": versions(tests),
           "shift_and_score": {"files": ss, "digest": digest(ss)},
           "cell": {"n_files": len(cf), "digest": digest(cf), "files": cf},
           "note": ("The generator and the scoring configuration are frozen (tests/review/freeze.py check): the line "
                    "set is drawn on these files; the rated set was selected on the files of rated_set and carried to "
                    "these by its identity check.")}
    sel_p = os.path.join(tests, "hidden", "selection.json")
    if os.path.isfile(sel_p):
        with open(sel_p) as fh:
            sel = json.load(fh)
        out["rated_set"] = {"selected_on_cell": dict(sel.get("frozen") or {}), "generator": sel.get("generator"),
                            "identity": identity}
    return out


def problems(tests=TESTS):
    with open(os.path.join(tests, "frozen.json")) as fh:
        want = json.load(fh)
    have = build(tests)
    out = []
    if want.get("frozen_version") != FROZEN_VERSION:
        out.append("frozen_version %r, the tool's %r" % (want.get("frozen_version"), FROZEN_VERSION))
    for k, v in have["versions"].items():
        if want["versions"].get(k) != v:
            out.append("version %s: frozen %r, now %r" % (k, want["versions"].get(k), v))
    rs = want.get("rated_set") or {}
    if have.get("rated_set") is not None and rs.get("selected_on_cell") != have["rated_set"]["selected_on_cell"]:
        out.append("rated_set: the recorded cell is not the rated selection's")
    if have.get("rated_set") is not None and not (rs.get("identity") or {}).get("identical"):
        out.append("rated_set: no identity record of the rated set on this cell")
    for group in ("shift_and_score", "cell"):
        w, h = want[group]["files"], have[group]["files"]
        for rel in sorted(set(w) | set(h)):
            if w.get(rel) != h.get(rel):
                out.append("%s: %s %s" % (group, rel, "added" if rel not in w else "missing" if rel not in h else "changed"))
        if want[group]["digest"] != digest(w):
            out.append("%s: the recorded digest does not match the recorded files" % group)
        if want[group]["digest"] != have[group]["digest"]:
            out.append("%s: digest %s, now %s" % (group, want[group]["digest"][:16], have[group]["digest"][:16]))
    return out


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd == "write":
        # write [identity json]: the identity record of the rated set on this cell ({"identical": true, ...})
        identity = None
        if len(argv) > 2:
            with open(argv[2]) as fh:
                identity = json.load(fh)
        elif os.path.isfile(MANIFEST):
            with open(MANIFEST) as fh:
                identity = (json.load(fh).get("rated_set") or {}).get("identity")
        with open(MANIFEST, "w") as fh:
            json.dump(build(identity=identity), fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("written %s" % MANIFEST)
        return 0
    bad = problems()
    for line in bad:
        print("FROZEN FILE CHANGED: " + line)
    print("frozen: %s" % ("OK" if not bad else "%d problem(s)" % len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
