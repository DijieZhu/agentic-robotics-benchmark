import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC_PATH = os.path.join(HERE, "params_public.json")
LAW_PATH = os.path.join(HERE, "params_law.json")
STATUS_CODES = ("PM", "PV", "MR", "MS", "AN", "DR", "OURS", "RECALLED", "LISTED")

_FIELD = re.compile(r"\{\{\s*([a-z][a-z0-9_]*:[A-Za-z0-9_.\-]+)\s*(?:\|\s*([^}|]*?)\s*)?\}\}")
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w])")


class DocsError(KeyError):
    ""


def is_entry(node):
    return isinstance(node, dict) and "value" in node and "status" in node


class Tables:

    def __init__(self, public=None, law=None, profiles=None):
        self.sha256 = {}
        self.files = {"public": public if public is not None else self._read("public", PUBLIC_PATH),
                      "law": law if law is not None else self._read("law", LAW_PATH)}
        for name, prof in (profiles or {}).items():
            if not re.fullmatch(r"[a-z0-9_]+", name):
                raise DocsError(f"profile name {name!r}: lower-case letters, digits and '_' only")
            if isinstance(prof, (str, os.PathLike)):
                prof = self._read("profile_" + name, os.fspath(prof))
            self.files["profile_" + name] = prof

    def _read(self, name, path):
        with open(path, "rb") as f:
            raw = f.read()
        self.sha256[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode("ascii"))

    def resolve(self, key):
        if ":" not in key:
            raise DocsError(f"{key}: a key is '<file>:<path>'")
        fname, path = key.split(":", 1)
        if fname not in self.files:
            raise DocsError(f"{key}: unknown file {fname}")
        node = self.files[fname]
        parts = path.split(".")
        i = 0
        while i < len(parts) and not is_entry(node):
            if not isinstance(node, dict) or parts[i] not in node:
                raise DocsError(key)
            node = node[parts[i]]
            i += 1
        if not is_entry(node):
            raise DocsError(f"{key}: no leaf entry")
        value = node["value"]
        for p in parts[i:]:
            if isinstance(value, list):
                if not re.fullmatch(r"\d+", p) or int(p) >= len(value):
                    raise DocsError(key)
                value = value[int(p)]
            elif isinstance(value, dict) and p in value:
                value = value[p]
            else:
                raise DocsError(key)
        return node, value

    def entry(self, key):
        return self.resolve(key)[0]

    def value(self, key):
        return self.resolve(key)[1]

    def number(self, key):
        v = self.value(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise DocsError(f"{key}: not a number ({v!r})")
        return v

    def status(self, key):
        return self.entry(key)["status"]

    def with_status(self, key, fmt=None):
        e, _ = self.resolve(key)
        return f"{fmt_number(self.number(key), fmt)} {e['unit']} ({e['status']})"

    def entries(self):
        out = []

        def walk(prefix, node):
            if is_entry(node):
                out.append((prefix, node))
            elif isinstance(node, dict):
                for k in sorted(node):
                    walk(f"{prefix}.{k}" if not prefix.endswith(":") else f"{prefix}{k}", node[k])
        for fname in self.files:
            walk(f"{fname}:", self.files[fname])
        return out

    def render(self, template):
        out, manifest, pos, last = [], [], 0, 0
        for m in _FIELD.finditer(template):
            out.append(template[last:m.start()])
            pos += m.start() - last
            key, fmt = m.group(1), (m.group(2) or None)
            e, _ = self.resolve(key)
            printed = fmt_number(self.number(key), fmt)
            manifest.append({"key": key, "fmt": fmt, "printed": printed, "status": e["status"], "unit": e["unit"],
                             "start": pos, "end": pos + len(printed)})
            out.append(printed)
            pos += len(printed)
            last = m.end()
        out.append(template[last:])
        return "".join(out), manifest

    def check_narrows(self, profile_key, law_key):
        def interval(key):
            v = self.value(key)
            if isinstance(v, bool):
                raise DocsError(f"{key}: not an interval")
            if isinstance(v, (int, float)):
                return float(v), float(v)
            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                           for x in v):
                return float(v[0]), float(v[1])
            raise DocsError(f"{key}: not an interval ({v!r})")
        p, w = interval(profile_key), interval(law_key)
        bad = []
        if p[0] > p[1]:
            bad.append({"key": profile_key, "problem": "the profile's interval is empty", "profile": p})
        if p[0] < w[0] or p[1] > w[1]:
            bad.append({"key": profile_key, "problem": "the profile widens the law", "profile": p, "law": w,
                        "law_key": law_key})
        return bad

    def check_print(self, text, manifest, strict=False, allow=()):
        bad = []
        covered = []
        for item in manifest:
            s, e = item["start"], item["end"]
            covered.append((s, e))
            if text[s:e] != item["printed"]:
                bad.append({"key": item["key"], "problem": "the page no longer holds the printed text at its place",
                            "page": text[s:e], "printed": item["printed"]})
                continue
            try:
                v = self.number(item["key"])
            except DocsError as exc:
                bad.append({"key": item["key"], "problem": f"the key names no number now: {exc}"})
                continue
            want = fmt_number(v, item["fmt"])
            if want != item["printed"]:
                bad.append({"key": item["key"], "problem": "the table's value prints differently", "page":
                            item["printed"], "table": want})
            elif not agrees(item["printed"], v):
                bad.append({"key": item["key"], "problem": "the printed decimal is not the table value rounded",
                            "page": item["printed"], "table": repr(v)})
        if strict:
            allowed = set(allow)
            for m in _NUMBER.finditer(text):
                inside = any(s <= m.start() and m.end() <= e for s, e in covered)
                if not inside and m.group(0) not in allowed:
                    bad.append({"key": None, "problem": "a number on the page that the manifest does not cover",
                                "page": m.group(0), "at": m.start()})
        return bad


def fmt_number(v, fmt=None):
    if fmt:
        return format(v, fmt)
    if isinstance(v, int):
        return str(v)
    return repr(float(v))


def agrees(printed, value):
    s = printed.strip().replace("+", "")
    try:
        p = float(s)
    except ValueError:
        return False
    mant = s.lower().split("e")[0]
    exp = int(s.lower().split("e")[1]) if "e" in s.lower() else 0
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    half = 0.5 * 10.0 ** (exp - decimals)
    return abs(p - float(value)) <= half * (1.0 + 1e-9) + 1e-300


_DEFAULT = None


def tables():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Tables()
    return _DEFAULT


def number(key):
    return tables().number(key)


def status(key):
    return tables().status(key)


def render(template):
    return tables().render(template)


def check_print(text, manifest, strict=False, allow=()):
    return tables().check_print(text, manifest, strict=strict, allow=allow)
