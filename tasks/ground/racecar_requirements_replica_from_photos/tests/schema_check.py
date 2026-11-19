#!/usr/bin/env python3
"""Structure check of a submission: requirements.yaml, bom.json and kinematic_sketch.json.

    python3 schema_check.py [SUBMISSION_DIR] [--schemas SCHEMA_DIR]

For each of the three files it checks that the file exists, is a regular file of at most 256 kB, is UTF-8 text,
parses (YAML for requirements.yaml, strict JSON for the other two: no duplicate keys, no NaN or Infinity), holds
only finite numbers and no integer above 2^53 in magnitude, matches its schema in schemas/, and follows the one
rule that a schema does not state: every facet id appears at most once. submission_contract.md gives the rules in
words. The check reads the structure only: it looks at no value and says nothing about any score.

Every text this file prints is a value of MESSAGES. A leading %s is the name of the file the problem is in.
"""
import argparse
import json
import math
import os
import stat
import sys

MESSAGES = {
    "missing": "%s is missing from the submission folder",
    "inspect": "%s cannot be inspected: %s",
    "not_regular": "%s is not a regular file (a link, a folder or a special file is not accepted)",
    "too_large": "%s is larger than %d bytes",
    "read": "%s cannot be read: %s",
    "not_utf8": "%s is not UTF-8 text",
    "parse": "%s does not parse: %s",
    "empty": "%s is empty: it holds no document",
    "not_finite": "%s: a number is not finite",
    "big_int": "%s: an integer is larger than %d in magnitude",
    "schema_unusable": "%s cannot be checked against its schema: %s",
    "schema": "%s does not match its schema: %s",
    "schema_item": "at %s: %s",
    "duplicate_facet": "requirements.yaml: the facet id %r appears more than once",
    "yaml_nodes": "its YAML aliases expand it past %d nodes",
    "yaml_cycle": "its YAML aliases form a cycle",
    "yaml_long_int": "it writes an integer in more than %d characters",
    "yaml_key": "a mapping key is not a string (line %d)",
    "duplicate_key": "duplicate key %r in one mapping (line %d)",
    "json_duplicate_key": "duplicate key %r in one object",
    "json_constant": "%s is not a number that a submission may hold",
    "summary_bad": "%d structural problem(s).",
    "summary_ok": "Structure OK. This check reads the structure only: it says nothing about any value or score.",
    "cli_description": "Structure check of requirements.yaml, bom.json and kinematic_sketch.json.",
    "cli_submission": "the submission folder (default: /home/user/submission)",
    "cli_schemas": "the folder of the three schemas (default: schemas/ next to this file)",
}

FILES = ("requirements.yaml", "bom.json", "kinematic_sketch.json")
SCHEMA_FILES = {"requirements.yaml": "requirements.schema.json", "bom.json": "bom.schema.json",
                "kinematic_sketch.json": "kinematic_sketch.schema.json"}
MAX_FILE_BYTES = 256 * 1024
MAX_YAML_NODES = 20000
MAX_ABS_INT = 2 ** 53
SCHEMA_PROBLEMS_SHOWN = 3


class Problem(ValueError):
    """A structure problem of the submission. Its text names the file and the submission's own content, nothing
    else."""


def _one_line(text, limit=300):
    """text on one line, cut to at most limit characters."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


# ================================================================================================ reading
def read_text(path):
    """The text of one submission file: a regular file (never followed through a link) of at most MAX_FILE_BYTES
    bytes, decoded as UTF-8. Problem otherwise."""
    name = os.path.basename(path)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise Problem(MESSAGES["missing"] % name) from None
    except OSError as exc:
        raise Problem(MESSAGES["inspect"] % (name, _one_line(exc.strerror or exc, 80))) from None
    if not stat.S_ISREG(st.st_mode):
        raise Problem(MESSAGES["not_regular"] % name)
    if st.st_size > MAX_FILE_BYTES:
        raise Problem(MESSAGES["too_large"] % (name, MAX_FILE_BYTES))
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as fh:
            data = fh.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise Problem(MESSAGES["read"] % (name, _one_line(exc.strerror or exc, 80))) from None
    if len(data) > MAX_FILE_BYTES:
        raise Problem(MESSAGES["too_large"] % (name, MAX_FILE_BYTES))
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise Problem(MESSAGES["not_utf8"] % name) from None


def _yaml_children(node):
    """The child nodes of a YAML node: the items of a sequence, the keys and values of a mapping."""
    import yaml
    if isinstance(node, yaml.SequenceNode):
        return node.value
    if isinstance(node, yaml.MappingNode):
        return [n for pair in node.value for n in pair]
    return []


def _yaml_nodes(root, cap):
    """Node count of the YAML node graph with every alias expanded, found without expanding anything. ValueError
    when the aliases form a cycle, the count passes cap, or an integer scalar is longer than Python converts."""
    import yaml
    int_tags = frozenset(tag for tag, fn in yaml.SafeLoader.yaml_constructors.items()
                         if fn is yaml.constructor.SafeConstructor.construct_yaml_int)
    size, path = {}, set()
    digits = sys.get_int_max_str_digits()
    todo = [(root, False)]
    while todo:
        node, done = todo.pop()
        key = id(node)
        if done:
            n = 1 + sum(size[id(c)] for c in _yaml_children(node))
            if n > cap:
                raise ValueError(MESSAGES["yaml_nodes"] % cap)
            size[key] = n
            path.discard(key)
        elif key in size:
            continue
        elif key in path:
            raise ValueError(MESSAGES["yaml_cycle"])
        else:
            if isinstance(node, yaml.ScalarNode) and node.tag in int_tags and 0 < digits < len(node.value):
                raise ValueError(MESSAGES["yaml_long_int"] % digits)
            path.add(key)
            todo.append((node, True))
            todo.extend((c, False) for c in _yaml_children(node))
    return size[id(root)]


def load_yaml_text(text):
    """A YAML document as Python objects, as yaml.safe_load builds it, but refused (ValueError) when it has more
    than MAX_YAML_NODES nodes with its aliases expanded, aliases that form a cycle, an over-long integer, a
    duplicate mapping key or a key that is not a string. Aliases as such are accepted."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _mapping(loader, node, deep=False):
        out = {}
        for knode, vnode in node.value:
            k = loader.construct_object(knode, deep=deep)
            if not isinstance(k, str):
                raise ValueError(MESSAGES["yaml_key"] % (knode.start_mark.line + 1))
            if k in out:
                raise ValueError(MESSAGES["duplicate_key"] % (k, knode.start_mark.line + 1))
            out[k] = loader.construct_object(vnode, deep=deep)
        return out

    _Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
    loader = _Loader(text)
    try:
        root = loader.get_single_node()
        if root is None:
            return None
        _yaml_nodes(root, MAX_YAML_NODES)
        return loader.construct_document(root)
    finally:
        loader.dispose()


def load_json_text(text):
    """A JSON document as Python objects, as json.loads builds it, but refused (ValueError) when an object holds a
    key twice or the text writes NaN or Infinity."""
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(MESSAGES["json_duplicate_key"] % (k,))
            out[k] = v
        return out

    def const(name):
        raise ValueError(MESSAGES["json_constant"] % name)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=const)


def parse(name, text):
    """The document of one submission file; Problem when it does not parse."""
    try:
        doc = load_yaml_text(text) if name.endswith(".yaml") else load_json_text(text)
    except Exception as exc:  # noqa: BLE001 (any parse failure is a structure problem)
        raise Problem(MESSAGES["parse"] % (name, _one_line(exc))) from None
    if doc is None:
        raise Problem(MESSAGES["empty"] % name)
    return doc


def read_document(folder, name):
    """read_text + parse of one submission file of folder."""
    return parse(name, read_text(os.path.join(str(folder), name)))


# ================================================================================================ checking
def numbers_problem(name, doc):
    """None when every number in doc is finite and every integer at most MAX_ABS_INT in magnitude."""
    stack = [doc]
    while stack:
        v = stack.pop()
        if isinstance(v, bool):
            continue
        if isinstance(v, float):
            if not math.isfinite(v):
                return MESSAGES["not_finite"] % name
        elif isinstance(v, int):
            if abs(v) > MAX_ABS_INT:
                return MESSAGES["big_int"] % (name, MAX_ABS_INT)
        elif isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
    return None


def schema_problems(name, doc, schema):
    """Up to SCHEMA_PROBLEMS_SHOWN schema violations of doc as one problem text each (empty: it matches). Any
    exception of the schema check is itself a problem of the document."""
    import jsonschema
    try:
        errs = sorted(jsonschema.Draft202012Validator(schema).iter_errors(doc),
                      key=lambda e: [str(x) for x in e.path])
        items = [MESSAGES["schema_item"] % ("/".join(str(x) for x in e.path) or "(top)", _one_line(e.message, 200))
                 for e in errs[:SCHEMA_PROBLEMS_SHOWN]]
    except Exception as exc:  # noqa: BLE001
        return [MESSAGES["schema_unusable"] % (name, _one_line(exc))]
    return [MESSAGES["schema"] % (name, "; ".join(items))] if items else []


def rule_problems(name, doc):
    """The rule a schema does not state, on a document that matches its schema: a facet id at most once (the
    schemas use no uniqueItems, whose check is quadratic on a long list)."""
    out = []
    if name == "requirements.yaml":
        seen = set()
        for f in doc["facets"]:
            if f["id"] in seen:
                out.append(MESSAGES["duplicate_facet"] % (f["id"],))
            seen.add(f["id"])
    return out


def check_document(name, doc, schema):
    """Every structure problem of one parsed document (empty list: it is well formed)."""
    bad = numbers_problem(name, doc)
    if bad:
        return [bad]
    probs = schema_problems(name, doc, schema)
    if probs:
        return probs
    return rule_problems(name, doc)


def load_schemas(folder):
    """{file name: schema} from the three schema files of folder."""
    out = {}
    for name in FILES:
        with open(os.path.join(str(folder), SCHEMA_FILES[name]), encoding="utf-8") as fh:
            out[name] = json.load(fh)
    return out


def check(folder, schemas):
    """(problems, documents): every structure problem of the three files of folder, and the documents that are well
    formed."""
    problems, docs = [], {}
    for name in FILES:
        try:
            doc = read_document(folder, name)
        except Problem as exc:
            problems.append(str(exc))
            continue
        probs = check_document(name, doc, schemas[name])
        if probs:
            problems.extend(probs)
        else:
            docs[name] = doc
    return problems, docs


def main(argv):
    """Check the submission folder named on the command line, print every problem, and return 0 when there is none
    and 1 otherwise."""
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=MESSAGES["cli_description"])
    ap.add_argument("submission", nargs="?", default="/home/user/submission", help=MESSAGES["cli_submission"])
    ap.add_argument("--schemas", default=os.path.join(here, "schemas"), help=MESSAGES["cli_schemas"])
    a = ap.parse_args(argv)
    problems, _ = check(a.submission, load_schemas(a.schemas))
    for x in problems:
        print(x.encode("utf-8", "backslashreplace").decode("utf-8"))
    if problems:
        print(MESSAGES["summary_bad"] % len(problems))
        return 1
    print(MESSAGES["summary_ok"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
