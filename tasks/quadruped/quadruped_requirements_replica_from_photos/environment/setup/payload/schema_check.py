#!/usr/bin/env python3
"""Structure check (gate G0) of a submission, without any ground truth: the three files parse, are mappings and
validate against schemas/; requirement ids are unique, every unit is a unit of its quantity and every bound is finite;
every BOM part is on the quote (price_list.md) with the quote's role; the BOM has exactly six lines (one actuator line
per joint type with one module per leg, one battery line and one hub line with count 1, one driver line); the sketch's
modules equal the BOM's actuator lines, every length, tolerance and stop is finite and the spacer legs are distinct leg
names.

Two G0 rules need inputs this script does not have: the battery family must be one of the brief's families, and the
driver line's count must be the boards a full set of channels needs. Give your own layer and catalogue transcription
(the formats of schemas/layer.schema.json and schemas/catalogue.schema.json) to check them too.

    python3 schema_check.py [SUBMISSION_DIR] [--layer LAYER.json] [--catalogue CATALOGUE.json]
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "grader_public"))


def quote_roles(path):
    """{part_id: role} from the quote's table rows (| part_id | role | ...)."""
    from quad.req_info.q1 import schemas as S
    roles = {}
    with open(path) as fh:
        for line in fh:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[1] in S.ROLES:
                roles[cells[0]] = cells[1]
    return roles


def main(argv=None):
    ap = argparse.ArgumentParser(description="gate G0 of a submission (no ground truth)")
    ap.add_argument("submission", nargs="?", default=os.path.join(HERE, "submission"))
    ap.add_argument("--layer", help="your quad-layer/1 (checks the battery family against its families)")
    ap.add_argument("--catalogue", help="your quad-catalogue/1 transcription (checks the driver count)")
    a = ap.parse_args(argv)
    from quad.req_info.q1 import grade as GR
    from quad.req_info.q1.catalog import Catalogue
    roles = quote_roles(os.path.join(HERE, "price_list.md"))
    docs, problems = GR.load_submission(a.submission, roles)
    not_checked = []
    if not problems and (a.layer or a.catalogue):
        cat = Catalogue(a.catalogue) if a.catalogue else None
        if a.layer:
            with open(a.layer) as fh:
                layer = json.load(fh)
            problems = GR.g0_rules(docs, roles, layer, cat)
        else:
            not_checked.append("the battery family (give --layer)")
            problems = GR._g0(docs, roles, None, cat)
    if not problems:
        if not a.layer:
            not_checked.append("the battery family (give --layer)")
        if not a.catalogue:
            not_checked.append("the driver count (give --catalogue)")
    if problems:
        print("G0 FAILS:")
        for p in problems:
            print("  -", p)
        return 1
    print("G0 ok: requirements.yaml (%d items), bom.json (%d lines), kinematic_sketch.json" % (
        len(docs["requirements"]["requirements"]), len(docs["bom"]["parts"])))
    for n in sorted(set(not_checked)):
        print("  not checked here:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
