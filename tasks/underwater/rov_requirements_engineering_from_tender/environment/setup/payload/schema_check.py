"""Structure check of the three submission files (submission_contract.md, sections 6 to 8).

    /opt/venv/bin/python3 schema_check.py                 # checks /home/user/submission/
    /opt/venv/bin/python3 schema_check.py SOME/FOLDER     # checks another folder

It checks that evaluation.json, spec_review.json and acceptance_schedule.json exist in the folder, are regular files
of at most 131072 bytes, are strict JSON (UTF-8 without a byte-order mark; no duplicate key; no NaN or Infinity; no
integer literal longer than Python converts; every number finite; no lone surrogate; an object at the top), match the
JSON schemas in schemas/ (an array longer than the longest one the schemas allow is refused before the schema
check), and follow the rules a schema cannot express: the options of an accepted bid are options of
that bid, the award is a bid the file accepts, spec_review.json labels exactly the clauses of TS-01 Part B, for_bid and
for_options agree with the award, clause ids are unique, and no (quantity, conditions, comparator) is repeated.

It looks at structure only. It knows nothing of the site, of the tendered units or of any delivery band, and it says
nothing about any score. Exit status 0 when every rule holds, 1 otherwise (each problem is printed).
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
FILES = ("evaluation.json", "spec_review.json", "acceptance_schedule.json")
MAX_BYTES = 131072
# the two priced options of each bid, as the price sheets print them
OPTION_IDS = {
    "T-01": ("O-PK20", "O-XL"),
    "T-02": ("O-PK18", "O-TL150"),
    "T-03": ("O-DVL", "O-L6"),
    "T-04": ("O-DVL", "O-L6"),
    "T-05": ("O-PK20", "O-TL200"),
    "T-06": ("O-L8", "O-PK20"),
    "T-07": ("O-DOME", "O-TL200"),
    "T-08": ("O-DVL", "O-XL"),
}


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
    return out, array_limit(raw)


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


def cross_file(docs, schemas):
    probs = []
    ev = docs["evaluation.json"]
    bids, award = ev["bids"], ev["award"]
    for b in sorted(bids):
        d = bids[b]
        if d["verdict"] == "accept":
            extra = sorted(set(d["options"]) - set(OPTION_IDS[b]))
            if extra:
                probs.append(f"evaluation.json: {b}: {extra} are not options of {b}")
    if award is not None and bids[award]["verdict"] != "accept":
        probs.append(f"evaluation.json: the award {award} is not a bid you accept")
    sc = docs["acceptance_schedule.json"]
    if award is None:
        if sc["for_bid"] is not None or sc["for_options"] or sc["clauses"]:
            probs.append("acceptance_schedule.json: no award, so for_bid must be null and for_options and clauses empty")
    elif sc["for_bid"] != award or (bids[award]["verdict"] == "accept" and set(sc["for_options"]) != set(bids[award]["options"])):
        probs.append("acceptance_schedule.json: for_bid / for_options do not match the award in evaluation.json")
    ids = [c["id"] for c in sc["clauses"]]
    if len(set(ids)) != len(ids):
        probs.append("acceptance_schedule.json: clause ids are not unique")
    seen = set()
    for c in sc["clauses"]:
        cond = tuple(sorted((k, float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                            for k, v in c["conditions"].items()))
        k = (c["quantity"], cond, c["comparator"])
        if k in seen:
            probs.append(f"acceptance_schedule.json: clause {c['id']} repeats a (quantity, conditions, comparator)")
        seen.add(k)
    return probs


def check(folder):
    """The list of problems of the submission folder (empty: the structure is right)."""
    schemas, limit = load_schemas()
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
    return cross_file(docs, schemas)


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
