"""Structure check of the three submission files (submission_contract.md).

    /opt/venv/bin/python3 schema_check.py                 # checks /home/user/submission/
    /opt/venv/bin/python3 schema_check.py SOME/FOLDER     # checks another folder

It checks that evaluation.json, spec_review.json and acceptance_schedules.json exist in the folder, are regular files
of at most 131072 bytes, are strict JSON (UTF-8 without a byte-order mark; no duplicate key; no NaN or Infinity; no
integer literal longer than Python converts; every number finite; no lone surrogate; an object at the top), match the
JSON schemas in schemas/ (an array longer than the longest one the schemas allow is refused before the schema
check), and follow the structural rules a schema cannot express: the options of an accepted bid and the for_options of
every schedule entry are options of that bid, spec_review.json labels exactly the clauses of TS-01 Part B, and every
clause of a schedule is a well-formed clause of the facility dictionary.

It looks at structure only. It does not check which bids have a schedule entry, it knows nothing of the site, of the
tendered units or of any delivery band, and it says nothing about any score. Exit status 0 when every rule holds,
1 otherwise (each problem is printed).
"""
import json
import math
import os
import stat
import sys

import jsonschema

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.join(HERE, "schemas")
DEFAULT_FOLDER = os.path.join(HERE, "submission")
FILES = ("evaluation.json", "spec_review.json", "acceptance_schedules.json")
MAX_BYTES = 131072
# the two priced options of each bid, as the price sheets print them
OPTION_IDS = json.loads('{"T-01": ["O-11", "O-12"], "T-02": ["O-21", "O-22"], "T-03": ["O-31", "O-32"], "T-04": ["O-41", "O-42"], "T-05": ["O-51", "O-52"], "T-06": ["O-61", "O-62"], "T-07": ["O-71", "O-72"], "T-08": ["O-81", "O-82"]}')


class Problem(ValueError):
    pass


def _short(x, n=120):
    s = repr(x)
    return s if len(s) <= n else s[:n] + "..."


def strict_json(data, name):
    if data[:3] == b"\xef\xbb\xbf":
        raise Problem(f"{name}: starts with a byte-order mark")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Problem(f"{name}: not UTF-8 ({exc.reason} at byte {exc.start})") from None
    limit = sys.get_int_max_str_digits()

    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise Problem(f"{name}: duplicate key {_short(k)}")
            out[k] = v
        return out

    def constant(word):
        raise Problem(f"{name}: {word} is not a finite number")

    def p_int(s):
        digits = len(s) - (1 if s[:1] == "-" else 0)
        if limit and digits > limit:
            raise Problem(f"{name}: an integer literal has more than {limit} digits")
        v = int(s)
        try:
            float(v)
        except OverflowError:
            raise Problem(f"{name}: an integer is too large for a float") from None
        return v

    def p_float(s):
        v = float(s)
        if not math.isfinite(v):
            raise Problem(f"{name}: a number is not finite once read")
        return v

    try:
        obj = json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_int=p_int, parse_float=p_float)
    except Problem:
        raise
    except RecursionError:
        raise Problem(f"{name}: nested too deeply") from None
    except ValueError as exc:
        raise Problem(f"{name}: not JSON ({' '.join(str(exc).split())[:200]})") from None
    todo = [obj]
    while todo:
        x = todo.pop()
        if isinstance(x, str):
            try:
                x.encode("utf-8")
            except UnicodeEncodeError:
                raise Problem(f"{name}: a string holds a lone surrogate") from None
        elif isinstance(x, dict):
            todo.extend(x.keys())
            todo.extend(x.values())
        elif isinstance(x, list):
            todo.extend(x)
    if not isinstance(obj, dict):
        raise Problem(f"{name}: the top level is not an object")
    return obj


def read_file(path):
    name = os.path.basename(path)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise Problem(f"{name}: missing") from None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise Problem(f"{name}: not a regular file")
    if st.st_size > MAX_BYTES:
        raise Problem(f"{name}: larger than {MAX_BYTES} bytes")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as fh:
        data = fh.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise Problem(f"{name}: larger than {MAX_BYTES} bytes")
    return data


def load_schemas():
    out, raw = {}, []
    for name in FILES:
        with open(os.path.join(SCHEMA_DIR, name[:-len(".json")] + ".schema.json"), encoding="utf-8") as fh:
            sch = json.load(fh)
        raw.append(sch)
        out[name] = jsonschema.Draft202012Validator(sch)
    return out, raw, array_limit(raw)


def array_limit(raw):
    """The largest maxItems of the schemas (None when an array of a schema has no maxItems)."""
    best, todo = 0, list(raw)
    while todo:
        x = todo.pop()
        if isinstance(x, dict):
            if x.get("type") == "array":
                if not isinstance(x.get("maxItems"), int):
                    return None
                best = max(best, x["maxItems"])
            todo.extend(x.values())
        elif isinstance(x, list):
            todo.extend(x)
    return best


def shape_gate(doc, name, limit):
    """No array anywhere in the document may be longer than the longest array the schemas allow."""
    if limit is None:
        return
    todo = [doc]
    while todo:
        x = todo.pop()
        if isinstance(x, list):
            if len(x) > limit:
                raise Problem(f"{name}: an array has {len(x)} items; the schemas allow at most {limit}")
            todo.extend(x)
        elif isinstance(x, dict):
            todo.extend(x.values())


def structure_rules(docs, raw):
    """The structural rules the schemas do not spell out."""
    probs = []
    bids = docs["evaluation.json"]["bids"]
    for b in sorted(bids):
        d = bids[b]
        if d["verdict"] == "accept":
            extra = sorted(set(d["options"]) - set(OPTION_IDS[b]))
            if extra:
                probs.append(f"evaluation.json: {b}: {extra} are not options of {b}")
    sc = docs["acceptance_schedules.json"]
    for b in sorted(sc):
        extra = sorted(set(sc[b]["for_options"]) - set(OPTION_IDS[b]))
        if extra:
            probs.append(f"acceptance_schedules.json: {b}: {extra} are not options of {b}")
    part_b = sorted(raw[1]["properties"]["clauses"]["properties"])
    if sorted(docs["spec_review.json"]["clauses"]) != part_b:
        probs.append("spec_review.json: the labels are not exactly the clauses of TS-01 Part B")
    for b in sorted(sc):
        for c in sc[b]["clauses"]:
            bound = c["bound"]
            if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(float(bound)):
                probs.append(f"acceptance_schedules.json: {b}: clause {c['id']}: the bound is not a finite number")
    return probs


def check(folder):
    """The list of problems of the submission folder (empty: the structure is right)."""
    if sorted(OPTION_IDS) != ["T-%02d" % i for i in range(1, 9)]:
        return ["schema_check.py: this copy has no option table (it is a template that was not built)"]
    schemas, raw, limit = load_schemas()
    docs, probs = {}, []
    for name in FILES:
        try:
            doc = strict_json(read_file(os.path.join(folder, name)), name)
            shape_gate(doc, name, limit)
        except Problem as exc:
            probs.append(str(exc))
            continue
        try:
            errs = sorted(schemas[name].iter_errors(doc), key=lambda e: ([str(x) for x in e.absolute_path], e.message))
        except RecursionError:
            probs.append(f"{name}: nested too deeply")
            continue
        if errs:
            for e in errs[:8]:
                where = "/".join(str(x) for x in e.absolute_path) or "(top)"
                probs.append(f"{name}: {where}: {' '.join(e.message.split())[:200]}")
            continue
        docs[name] = doc
    if probs:
        return probs
    return structure_rules(docs, raw)


def main(argv):
    folder = argv[1] if len(argv) > 1 else DEFAULT_FOLDER
    probs = check(folder)
    for p in probs:
        print("problem: " + p)
    if not probs:
        print("ok: the three files in %s have the required structure" % folder)
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
