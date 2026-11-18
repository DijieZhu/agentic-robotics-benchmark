#!/usr/bin/env python3
"""Structure check of a submission: gates G0 and G2, exactly as the grader runs them.

    python3 schema_check.py [SUBMISSION_DIR] [--json]

SUBMISSION_DIR defaults to ~/submission. Exit status: 0 when both gates hold, otherwise 1 (every problem is printed).
The check reads the three deliverables and the schemas in schemas/ next to this file, nothing else; it knows no
answer and scores nothing. A failed gate scores the whole submission 0.

G0 structure
  * requirements.yaml, bom.json and cable_net.json are present as regular files (a symbolic link, a directory or a
    device is refused), each at most MAX_FILE_BYTES bytes of UTF8 text;
  * each parses into a mapping: requirements.yaml as YAML, the other two as JSON;
  * a YAML file nests collections at most MAX_NESTING levels deep, holds at most MAX_NODES nodes with every alias
    expanded, has no alias that refers to itself, and writes no integer in more than MAX_INT_CHARS characters;
    a JSON file holds at most MAX_NODES values;
  * every number is finite and every string is valid text (no lone surrogate escape);
  * each document validates against its schema in schemas/ (names, units, comparators, source ids, the order
    size, the list lengths of the schemas).
G2 no hedging
  * cable_net.json: one record per motor_index, per sensor_index and per controller_index, one spare input per strut;
  * requirements.yaml: one entry per quantity and comparator ("=" and "==" are one comparator, so are "<=" and "<",
    and ">=" and ">"), one entry per derived quantity;
  * no more entries than the schemas allow in requirements.yaml items and derived and in bom.json lines.
Not errors: a record, field, entry or section left out; a lower and an upper limit for one quantity; one part on
several BOM lines; entries or lines that the evidence does not support (those are scored, not gated).
"""
import argparse
import json
import math
import os
import re
import stat
import sys

sys.dont_write_bytecode = True

import jsonschema  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.join(HERE, "schemas")

MAX_FILE_BYTES = 2_000_000
MAX_NODES = 100_000
MAX_NESTING = 64
MAX_INT_CHARS = 64
MAX_SCHEMA_ERRORS = 40
MAX_MESSAGE_CHARS = 240
MAX_LINE_CHARS = 4000
MAX_SHOWN_INT_BITS = 4096
READ_CHUNK = 65536

# (key, file name, schema file); requirements.yaml is YAML, the others JSON
FILES = (("requirements", "requirements.yaml", "requirements.schema.json"),
         ("bom", "bom.json", "bom.schema.json"),
         ("cable_net", "cable_net.json", "cable_net.schema.json"))
# top-level lists whose length cap (the schema's maxItems) is a G2 rule; every other list cap is a G0 rule
G2_CAPPED = (("requirements", "items"), ("requirements", "derived"), ("bom", "lines"))
# cable_net.json lists and the field that identifies a record (G2: unique)
INDEXED = (("cables", "motor_index"), ("sensors", "sensor_index"), ("controllers", "controller_index"),
           ("spare_inputs", "strut"))
COMPARATOR_CLASS = {"=": "=", "==": "=", "<=": "<=", "<": "<=", ">=": ">=", ">": ">="}

_LOADER = yaml.CSafeLoader if getattr(yaml, "__with_libyaml__", False) else yaml.SafeLoader
_YAML_INT_TAGS = frozenset(tag for tag, fn in yaml.SafeLoader.yaml_constructors.items()
                           if fn is yaml.constructor.SafeConstructor.construct_yaml_int)
_NODE_EVENTS = (yaml.ScalarEvent, yaml.AliasEvent, yaml.SequenceStartEvent, yaml.MappingStartEvent)
_OPEN_EVENTS = (yaml.SequenceStartEvent, yaml.MappingStartEvent)
_CLOSE_EVENTS = (yaml.SequenceEndEvent, yaml.MappingEndEvent)
_UNQUOTED_COMPARATOR = re.compile(r"^[ \t-]*comparator[ \t]*:[ \t]*[=<>]", re.MULTILINE)
_EXPONENT_NUMBER = re.compile(r"^[ \t]*[-+]?[\d_]*\.?[\d_]+[eE][-+]?\d+[ \t]*$", re.ASCII)


# ------------------------------------------------------------------------------------------------------------ text
def ascii_text(text, limit=MAX_MESSAGE_CHARS):
    """`text` as printable ASCII (anything else, a lone surrogate included, as a backslash escape), cut to `limit`."""
    out = str(text).encode("ascii", "backslashreplace").decode("ascii")
    out = " ".join(out.split())
    return out if len(out) <= limit else out[:limit - len(" ...")] + " ..."


def _shown(value):
    """A short, safe rendering of an agent value for a message."""
    if isinstance(value, str):
        return ascii_text(repr(value[:MAX_INT_CHARS]), MAX_MESSAGE_CHARS)
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, int):
        return ascii_text(repr(value)) if value.bit_length() <= MAX_SHOWN_INT_BITS else "a very long integer"
    if isinstance(value, float):
        return repr(value)
    return "a value of type %s" % type(value).__name__


def _problem(gate, name, message):
    return {"gate": gate, "file": name, "message": ascii_text(message)}


def _path_text(path):
    out = ""
    for p in path:
        out += "[%d]" % p if isinstance(p, int) else (".%s" % p if out else str(p))
    return out or "(top level)"


# ------------------------------------------------------------------------------------------------------------ read
def read_capped(path):
    """(bytes, None) for a regular file of at most MAX_FILE_BYTES bytes, else (None, why). Never follows a symbolic
    link, never blocks on a pipe, never reads more than MAX_FILE_BYTES + one chunk."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None, "is missing"
    except OSError as e:
        return None, "cannot be read (%s)" % (e.strerror or type(e).__name__)
    if stat.S_ISLNK(st.st_mode):
        return None, "is a symbolic link: submit a regular file"
    if not stat.S_ISREG(st.st_mode):
        return None, "is not a regular file"
    if st.st_size > MAX_FILE_BYTES:
        return None, "has %d bytes, more than the limit of %d" % (st.st_size, MAX_FILE_BYTES)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:
        return None, "cannot be opened (%s)" % (e.strerror or type(e).__name__)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None, "is not a regular file"
        if st.st_size > MAX_FILE_BYTES:
            return None, "has %d bytes, more than the limit of %d" % (st.st_size, MAX_FILE_BYTES)
        chunks, size = [], 0
        while size <= MAX_FILE_BYTES:
            block = os.read(fd, READ_CHUNK)
            if not block:
                break
            chunks.append(block)
            size += len(block)
        if size > MAX_FILE_BYTES:
            return None, "has more than %d bytes" % MAX_FILE_BYTES
        return b"".join(chunks), None
    except OSError as e:
        return None, "cannot be read (%s)" % (e.strerror or type(e).__name__)
    finally:
        os.close(fd)


def _decode(data):
    try:
        text = data.decode("utf8")
    except UnicodeDecodeError as e:
        raise ValueError("is not UTF8 text (invalid byte at offset %d)" % e.start)
    return text[len("\ufeff"):] if text.startswith("\ufeff") else text


# ------------------------------------------------------------------------------------------------------------ YAML
def _yaml_prescan(text):
    """Walk the YAML events once, before anything is built: refuse (ValueError) a text that nests collections more
    than MAX_NESTING levels deep or holds more than MAX_NODES nodes (aliases not yet expanded). A deeply nested text
    is slow to compose, so this pass stops at the first level too deep."""
    loader = _LOADER(text)
    try:
        depth = nodes = 0
        while loader.check_event():
            event = loader.get_event()
            if isinstance(event, _NODE_EVENTS):
                nodes += 1
                if nodes > MAX_NODES:
                    raise ValueError("holds more than %d YAML nodes" % MAX_NODES)
            if isinstance(event, _OPEN_EVENTS):
                depth += 1
                if depth > MAX_NESTING:
                    raise ValueError("nests collections more than %d levels deep" % MAX_NESTING)
            elif isinstance(event, _CLOSE_EVENTS):
                depth -= 1
    finally:
        loader.dispose()


def _yaml_children(node):
    if isinstance(node, yaml.SequenceNode):
        return node.value
    if isinstance(node, yaml.MappingNode):
        return [n for pair in node.value for n in pair]
    return []


def _yaml_expanded_nodes(root):
    """Node count of the composed graph `root` with every alias expanded (a merge key's mappings count where they are
    merged), found without expanding anything: each distinct node is visited once. ValueError when an alias refers to
    a node that contains it, when the count passes MAX_NODES, or when an integer scalar is written in more than
    MAX_INT_CHARS characters."""
    size, on_path = {}, set()
    todo = [(root, False)]
    while todo:
        node, done = todo.pop()
        key = id(node)
        if done:
            n = sum(size[id(c)] for c in _yaml_children(node)) + 1
            if n > MAX_NODES:
                raise ValueError("its YAML aliases expand it past %d nodes" % MAX_NODES)
            size[key] = n
            on_path.discard(key)
        elif key in size:
            continue
        elif key in on_path:
            raise ValueError("a YAML alias refers to a node that contains it (a cycle)")
        else:
            if isinstance(node, yaml.ScalarNode) and node.tag in _YAML_INT_TAGS and len(node.value) > MAX_INT_CHARS:
                raise ValueError("writes an integer in more than %d characters" % MAX_INT_CHARS)
            on_path.add(key)
            todo.append((node, True))
            todo.extend((c, False) for c in _yaml_children(node))
    return size[id(root)]


def load_yaml(text):
    """yaml.safe_load(text) within the bounds of _yaml_prescan and _yaml_expanded_nodes (ValueError beyond them);
    within them, a text loads exactly as yaml.safe_load loads it."""
    _yaml_prescan(text)
    loader = _LOADER(text)
    try:
        root = loader.get_single_node()
        if root is None:
            return None
        _yaml_expanded_nodes(root)
        return loader.construct_document(root)
    finally:
        loader.dispose()


# ------------------------------------------------------------------------------------------------------------ values
def _value_problems(doc):
    """Problems of the loaded document as a whole: more than MAX_NODES values, a number that is not finite or too large
    for a float, a string that is not valid text. At most one message per kind."""
    found, count, todo = {}, 0, [doc]
    while todo:
        v = todo.pop()
        count += 1
        if count > MAX_NODES:
            found["size"] = "holds more than %d values" % MAX_NODES
            break
        if isinstance(v, dict):
            for k, x in v.items():
                todo.append(k)
                todo.append(x)
        elif isinstance(v, list):
            todo.extend(v)
        elif isinstance(v, str):
            try:
                v.encode("utf8")
            except UnicodeEncodeError:
                found.setdefault("text", "a string contains a lone surrogate escape (%s): write valid text" % _shown(v))
        elif isinstance(v, bool) or v is None:
            continue
        elif isinstance(v, (int, float)):
            try:
                ok = math.isfinite(float(v))
            except OverflowError:
                ok = False
            if not ok:
                found.setdefault("number", "a number is not finite or too large (%s)" % _shown(v))
    return [found[k] for k in sorted(found)]


def read_submission(sub_dir):
    """({key: document}, [problem]): every deliverable read, decoded, parsed and checked as a whole (the first four G0
    items of the module text). A document with a problem is left out of the mapping."""
    docs, problems = {}, []
    for key, name, _schema in FILES:
        data, why = read_capped(os.path.join(sub_dir, name))
        if why:
            problems.append(_problem("G0", name, "%s %s" % (name, why)))
            continue
        try:
            text = _decode(data)
        except ValueError as e:
            problems.append(_problem("G0", name, "%s %s" % (name, e)))
            continue
        try:
            doc = load_yaml(text) if name.endswith(".yaml") else json.loads(text)
        except Exception as e:  # noqa: BLE001  (any refusal or parse error of an agent file is a G0 problem)
            why = str(e) if isinstance(e, ValueError) else "%s: %s" % (type(e).__name__, e)
            msg = "%s does not parse: %s" % (name, why)
            if name.endswith(".yaml") and _UNQUOTED_COMPARATOR.search(text):
                msg = "%s does not parse (write comparators in quotes, e.g. comparator: \"<=\"): %s" % (name, why)
            problems.append(_problem("G0", name, msg))
            continue
        if not isinstance(doc, dict):
            problems.append(_problem("G0", name, "%s is not a mapping at the top level" % name))
            continue
        found = _value_problems(doc)
        if found:
            problems += [_problem("G0", name, "%s %s" % (name, m)) for m in found]
            continue
        docs[key] = doc
    return docs, problems


# ------------------------------------------------------------------------------------------------------------ schema
_SCHEMAS = {}


def _validator(schema_dir, schema_file):
    key = (os.path.abspath(schema_dir), schema_file)
    if key not in _SCHEMAS:
        with open(os.path.join(schema_dir, schema_file), encoding="utf8") as fh:
            schema = json.load(fh)
        _SCHEMAS[key] = (schema, jsonschema.Draft202012Validator(schema))
    return _SCHEMAS[key]


def _trim(doc, schema, path, over):
    """A copy of `doc` in which every list that `schema` bounds (maxItems, reached through properties and items) is cut
    to its bound; each cut is appended to `over` as (path, length, bound). The schema check then never walks a long
    list, and the cut itself is the reported problem."""
    if not isinstance(schema, dict):
        return doc
    if isinstance(doc, dict) and isinstance(schema.get("properties"), dict):
        out = dict(doc)
        for k, sub in schema["properties"].items():
            if k in out:
                out[k] = _trim(out[k], sub, path + (k,), over)
        return out
    if isinstance(doc, list):
        cap = schema.get("maxItems")
        if isinstance(cap, int) and not isinstance(cap, bool) and len(doc) > cap:
            over.append((path, len(doc), cap))
            doc = doc[:cap]
        item = schema.get("items")
        return [_trim(x, item, path + (i,), over) for i, x in enumerate(doc)] if isinstance(item, dict) else list(doc)
    return doc


def _error_key(err):
    return ([(0, p) if isinstance(p, int) else (1, str(p)) for p in err.absolute_path], err.message)


def _schema_problems(key, name, doc, schema_dir):
    schema, validator = _validator(schema_dir, dict((k, s) for k, _n, s in FILES)[key])
    problems, over = [], []
    trimmed = _trim(doc, schema, (), over)
    for path, n, cap in over:
        gate = "G2" if len(path) == 1 and (key, path[0]) in G2_CAPPED else "G0"
        what = "entries" if gate == "G2" else "items"
        problems.append(_problem(gate, name, "%s: %s has %d %s, the limit is %d%s" % (
            name, _path_text(path), n, what, cap, " (entries after the limit were not checked)" if gate == "G2" else "")))
    try:
        errors = sorted(validator.iter_errors(trimmed), key=_error_key)
    except Exception as e:  # noqa: BLE001  (a value the schema check cannot even report on is a G0 problem)
        return problems + [_problem("G0", name, "%s: the schema check cannot report on it: %s" % (name, type(e).__name__))]
    for err in errors[:MAX_SCHEMA_ERRORS]:
        msg = "%s: %s: %s" % (name, _path_text(err.absolute_path), err.message)
        inst = err.instance
        if (isinstance(inst, str) and err.validator == "type" and _EXPONENT_NUMBER.match(inst)
                and err.absolute_path and err.absolute_path[-1] in ("bound", "value")):
            msg += " (write numbers in plain decimal notation, without an exponent)"
        problems.append(_problem("G0", name, msg))
    if len(errors) > MAX_SCHEMA_ERRORS:
        problems.append(_problem("G0", name, "%s: %d more schema errors not shown" % (name, len(errors) - MAX_SCHEMA_ERRORS)))
    return problems


# ------------------------------------------------------------------------------------------------------------ G2
def _repeats(values):
    """{value: count} of the values that occur more than once, in first-seen order."""
    seen, rep = {}, {}
    for v in values:
        if v in seen:
            rep[v] = rep.get(v, 1) + 1
        seen[v] = True
    return rep


def _g2_problems(docs):
    problems = []
    net = docs.get("cable_net")
    if isinstance(net, dict):
        for section, field in INDEXED:
            recs = net.get(section)
            if not isinstance(recs, list):
                continue
            vals = [r.get(field) for r in recs if isinstance(r, dict)]
            vals = [v for v in vals if isinstance(v, (int, str)) and not isinstance(v, bool)]
            for v, n in _repeats(vals).items():
                problems.append(_problem("G2", "cable_net.json", "cable_net.json: %s %s appears in %d records of %s" % (
                    field, _shown(v), n, section)))
    req = docs.get("requirements")
    if isinstance(req, dict):
        items = req.get("items")
        if isinstance(items, list):
            keys = []
            for it in items:
                if isinstance(it, dict) and isinstance(it.get("quantity"), str):
                    cc = COMPARATOR_CLASS.get(it.get("comparator")) if isinstance(it.get("comparator"), str) else None
                    if cc is not None:
                        keys.append((it["quantity"], cc))
            for (q, cc), n in _repeats(keys).items():
                problems.append(_problem("G2", "requirements.yaml", "requirements.yaml: quantity %s with comparator %s "
                                         "appears in %d entries of items" % (_shown(q), _shown(cc), n)))
        derived = req.get("derived")
        if isinstance(derived, list):
            qs = [d.get("quantity") for d in derived if isinstance(d, dict) and isinstance(d.get("quantity"), str)]
            for q, n in _repeats(qs).items():
                problems.append(_problem("G2", "requirements.yaml", "requirements.yaml: derived quantity %s appears in %d "
                                         "entries of derived" % (_shown(q), n)))
    return problems


def _notes(docs):
    """Remarks that are not errors: repeated ids (ids are labels; nothing is scored by id)."""
    notes = []
    for key, name, section, field in (("requirements", "requirements.yaml", "items", "id"),
                                      ("requirements", "requirements.yaml", "derived", "id"),
                                      ("bom", "bom.json", "lines", "line_id")):
        doc = docs.get(key)
        recs = doc.get(section) if isinstance(doc, dict) else None
        if isinstance(recs, list):
            ids = [r.get(field) for r in recs if isinstance(r, dict) and isinstance(r.get(field), str)]
            for v, n in _repeats(ids).items():
                notes.append(ascii_text("%s: %s %s is used by %d entries of %s (not an error; ids are labels, but an "
                                        "input that cites it is ambiguous)" % (name, field, _shown(v), n, section)))
    return notes


# ------------------------------------------------------------------------------------------------------------ entry
def check_documents(docs, schema_dir=SCHEMA_DIR):
    """(problems, notes) of loaded documents: the schema of each, then the G2 rules."""
    problems = []
    for key, name, _schema in FILES:
        if key in docs:
            problems += _schema_problems(key, name, docs[key], schema_dir)
    problems += _g2_problems(docs)
    return problems, _notes(docs)


def check_submission(sub_dir, schema_dir=SCHEMA_DIR):
    """{"ok", "g0", "g2", "problems", "notes", "docs"} of the submission folder `sub_dir`. Never raises on anything an
    agent can write; `docs` holds the documents that loaded (the grader scores them only when both gates hold)."""
    docs, problems = read_submission(sub_dir)
    more, notes = check_documents(docs, schema_dir)
    problems += more
    g0 = not any(p["gate"] == "G0" for p in problems)
    g2 = not any(p["gate"] == "G2" for p in problems)
    return {"ok": g0 and g2, "g0": g0, "g2": g2, "problems": problems, "notes": notes, "docs": docs}


def _summary(docs):
    req, bom, net = docs.get("requirements", {}), docs.get("bom", {}), docs.get("cable_net", {})
    parts = ["requirements.yaml: %d items, %d derived" % (len(req.get("items") or []), len(req.get("derived") or [])),
             "bom.json: %d lines" % len(bom.get("lines") or [])]
    net_parts = ["%d %s" % (len(net.get(s) or []), s) for s, _f in INDEXED]
    net_parts += ["%d geometry" % len(net.get("geometry") or {}), "%d counts" % len(net.get("counts") or {})]
    return "; ".join(parts + ["cable_net.json: " + ", ".join(net_parts)])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gates G0 and G2 of a submission (structure, no hedging).")
    ap.add_argument("submission", nargs="?", default=os.path.join(os.path.expanduser("~"), "submission"))
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    try:
        rep = check_submission(args.submission)
    except Exception as e:  # noqa: BLE001  (the check must report, never crash)
        rep = {"ok": False, "g0": False, "g2": True, "notes": [], "docs": {},
               "problems": [_problem("G0", "-", "the check cannot report on the submission: %s" % type(e).__name__)]}
    if args.json:
        out = {k: rep[k] for k in ("ok", "g0", "g2", "problems", "notes")}
        print(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=True))
        return 0 if rep["ok"] else 1
    print(ascii_text("schema_check: %s" % os.path.abspath(args.submission), MAX_LINE_CHARS))
    for p in rep["problems"]:
        print("%s %s" % (p["gate"], p["message"]))
    for n in rep["notes"]:
        print("note %s" % n)
    if rep["ok"]:
        print(ascii_text("PASS: G0 and G2 hold. " + _summary(rep["docs"]), MAX_LINE_CHARS))
        return 0
    n0 = sum(p["gate"] == "G0" for p in rep["problems"])
    n2 = sum(p["gate"] == "G2" for p in rep["problems"])
    print("FAIL: %d problem(s), G0 %d, G2 %d. A failed gate scores the whole submission 0." % (n0 + n2, n0, n2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
