"""The graded line set: its registered rule and its draw (a maintainer's tool; the grader never runs it).

    python3 -B tests/review/draw_line_set.py rule        # writes hidden/line/selection_rule.txt (before the draw)
    python3 -B tests/review/draw_line_set.py draw        # the draw: salt, packages, seeds.json, selection.json
    python3 -B tests/review/draw_line_set.py check       # the set on disk is the rule's draw for the recorded salt

The rule (``RULE``, written to ``hidden/line/selection_rule.txt`` and its sha256 recorded before the draw): eight
slots (``gen_shifts.LINE_SLOTS``: 1 slow running, 2 medium running, 3 fast running, 4 medium indexing, 5 fast
running, 6 slow indexing, 7 medium running, 8 fast running); one salt of 16 bytes from ``os.urandom``, drawn once at
the draw and recorded in ``hidden/line/selection.json``; slot s's shift is the frozen line generator's draw
(``parcelcell.shiftgen.draw_line_shift``) for the seed ``line_seed("line-hidden-<s>", 0, mix=salt)`` in the slot's
belt class and belt mode: the first draw, with no quota and no admission; nothing is rejected and nothing is run on
a shift before the set is frozen. Run it on the frozen files (``review/freeze.py check`` passes first).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import common  # noqa: E402
import gen_shifts  # noqa: E402

LINE_DIR = os.path.join(common.TESTS, "hidden", "line")
RULE_FILE = os.path.join(LINE_DIR, "selection_rule.txt")
LABEL = "line-hidden-%d"
RULE = (
    "THE REGISTERED RULE OF THE GRADED LINE SET (written before any draw). Eight shifts; slots: 1 slow (0.05-0.15 "
    "m/s) running, 2 medium (0.15-0.25) running, 3 fast (0.25-0.35) running, 4 medium indexing, 5 fast running, 6 "
    "slow indexing, 7 medium running, 8 fast running. One salt of 16 bytes from the system's entropy, drawn once at "
    "selection time and recorded in the hidden selection file; each slot's shift is the frozen line generator's draw "
    "for the seed hash(salt, slot label, 0): the first draw, with no quota and no admission; nothing is rejected and "
    "nothing is run on a candidate before the set is frozen. After the freeze, as information only: the reference's "
    "and two other programs' scores on the set.\n"
    "\n"
    "How it is carried out. The slots in order, each with its belt class and belt mode: 1 slow running, 2 medium "
    "running, 3 fast running, 4 medium indexing, 5 fast running, 6 slow indexing, 7 medium running, 8 fast running "
    "(review/gen_shifts.py LINE_SLOTS; the belt classes of grader_config.json block line: slow 0.05-0.15 m/s, medium "
    "0.15-0.25 m/s, fast 0.25-0.35 m/s, the speed drawn uniformly inside the class and rounded to 1 mm/s; indexing: the "
    "belt stands while the station's parcel has its centre in y -0.15..0.10). The salt: os.urandom(16), called once by "
    "review/draw_line_set.py draw, recorded as salt_hex in hidden/line/selection.json. Slot s's label is "
    "'line-hidden-<s>' and its seed parcelcell.shiftgen.line_seed('line-hidden-<s>', 0, cfg, mix=salt): the first 8 "
    "bytes, shifted right by one bit, of sha256 of '<task id>/line-hidden-<s>/<line shifts version>/0' followed by "
    "'/' and the salt. Slot s's shift is parcelcell.shiftgen.draw_line_shift(seed, cfg, profile='full', belt_class, "
    "indexing) on the files pinned by tests/frozen.json (review/freeze.py check passes before the draw), written as "
    "hidden/line/shifts/line_<ss>.json with its sha256 in hidden/line/seeds.json; review/draw_line_set.py check draws "
    "the set again from the recorded salt and compares the bytes. The rule file's sha256 is recorded in "
    "hidden/line/selection.json and in the maintainer's log before the draw.\n")


def sha256_file(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def draw_set(cfg, salt):
    from parcelcell.shiftgen import draw_line_shift, line_seed, package_sha256, to_package
    out = []
    for s, (bc, idx) in enumerate(gen_shifts.LINE_SLOTS, 1):
        seed = line_seed(LABEL % s, 0, cfg, mix=salt)
        sh, meta = draw_line_shift(seed, cfg, profile="full", belt_class=bc, indexing=idx, label=LABEL % s, k=0)
        pkg = to_package(sh, meta)
        out.append((s, seed, bc, idx, pkg, package_sha256(pkg), meta["features"]))
    return out


def cmd_rule(_a):
    os.makedirs(LINE_DIR, exist_ok=True)
    if os.path.exists(RULE_FILE):
        sys.exit("the rule file exists: a rule is written once, before its draw")
    with open(RULE_FILE, "w") as fh:
        fh.write(RULE)
    print("rule written", sha256_file(RULE_FILE))
    return 0


def cmd_draw(_a):
    from parcelcell.shiftgen import package_bytes
    if not os.path.isfile(RULE_FILE) or open(RULE_FILE).read() != RULE:
        sys.exit("write the rule first (draw_line_set.py rule)")
    cfg = common.load_cfg()
    salt = os.urandom(16)
    rows = draw_set(cfg, salt)
    sd = os.path.join(LINE_DIR, "shifts")
    os.makedirs(sd, exist_ok=True)
    for f in sorted(os.listdir(sd)):
        if f.startswith("line_") and f.endswith(".json"):
            os.remove(os.path.join(sd, f))                 # the placeholder's packages, by name
    shifts = []
    for s, seed, bc, idx, pkg, sha, feat in rows:
        rel = "shifts/line_%02d.json" % s
        with open(os.path.join(LINE_DIR, rel), "wb") as fh:
            fh.write(package_bytes(pkg))
        shifts.append({"tag": "line_%02d" % s, "package": rel, "sha256": sha, "seed": seed, "k": 0, "label": LABEL % s,
                       "slot": s, "belt_class": bc, "indexing": idx, "belt_speed": feat["belt_speed"],
                       "n_parcels": feat["n_parcels"], "classes": feat["classes"], "n_soft": feat["n_soft"],
                       "label_positions": feat["label_positions"]})
    common.dump_json(os.path.join(LINE_DIR, "seeds.json"),
                     {"placeholder": False, "note": "The graded line set: eight shifts drawn by the registered rule "
                                                    "(selection_rule.txt, selection.json).",
                      "seeds": [r[1] for r in rows], "shifts": shifts})
    common.dump_json(os.path.join(LINE_DIR, "selection.json"),
                     {"rule_sha256": sha256_file(RULE_FILE), "salt_hex": salt.hex(),
                      "salt_source": "os.urandom(16), drawn %s" % time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                      "frozen_cell_digest": json.load(open(os.path.join(common.TESTS, "frozen.json")))["cell"]["digest"],
                      "tool_sha256": sha256_file(os.path.abspath(__file__)),
                      "slots": [{k: v for k, v in sh.items() if k != "package"} for sh in shifts]})
    for sh in shifts:
        print(sh["tag"], sh["belt_class"], "indexing" if sh["indexing"] else "running", sh["belt_speed"],
              sh["n_parcels"], sh["classes"])
    return 0


def cmd_check(_a):
    cfg = common.load_cfg()
    sel = json.load(open(os.path.join(LINE_DIR, "selection.json")))
    seeds = json.load(open(os.path.join(LINE_DIR, "seeds.json")))
    rows = draw_set(cfg, bytes.fromhex(sel["salt_hex"]))
    bad = [s for (s, seed, bc, idx, pkg, sha, feat), pin in zip(rows, seeds["shifts"])
           if sha != pin["sha256"] or sha256_file(os.path.join(LINE_DIR, pin["package"])) != sha]
    ok = not bad and sel["rule_sha256"] == sha256_file(RULE_FILE) and len(rows) == len(seeds["shifts"])
    print("line set:", "the rule's draw" if ok else "DIFFERS (slots %s)" % bad)
    return 0 if ok else 1


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("rule", "draw", "check"))
    a = ap.parse_args(argv)
    return {"rule": cmd_rule, "draw": cmd_draw, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
