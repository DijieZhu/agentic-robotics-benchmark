"""The maker's robot description read as XML text, with the line of every element, attribute and text value.

Nothing is evaluated: no xacro processing, no include, no macro expansion by a xacro engine. The two description
files are well-formed XML; this module parses them with the standard library's expat and records where each value
is written, so that every number of the placement can be traced to a line of the pinned file. The values are
returned twice: the text exactly as written ("3.14", "3.1415") and the float it reads as.

read_model() and read_arm_macro() take out what the placement needs: the body link's visuals (origin, mesh), the
active bay call of the arm macro, and the arm's seven joints (origin, axis, limits), its links' visual origins and
its start values. Only the expression forms these files use are read: "$(arg NAME)" in a property, "${NAME ==
'LITERAL'}" in an xacro:if, "${NAME}" substitutions of the macro's parameters and of the prefix and ns properties;
anything else in a value the placement needs raises MakerStdError.

The same line-recording parser also reads the launch file (XML) for the default bay and namespace arguments.
"""
import re
import xml.parsers.expat

from .errors import MakerStdError

XACRO_NS = "http://www.ros.org/wiki/xacro"
MODEL_FILE = "description/model.urdf.xacro"
MACRO_FILE = "description/macro_perching_arm.urdf.xacro"
MESH_PREFIX = "package://astrobee_freeflyer/meshes/"
_ATTR_RX = re.compile(rb'([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*("[^"]*"|\'[^\']*\')')


class Element:
    """An XML element with line numbers (1-based) for itself, each attribute and its text."""

    __slots__ = ("tag", "attrs", "attr_lines", "line", "text", "text_line", "children", "parent")

    def __init__(self, tag, attrs, attr_lines, line, parent):
        self.tag = tag
        self.attrs = attrs
        self.attr_lines = attr_lines
        self.line = line
        self.text = ""
        self.text_line = None
        self.children = []
        self.parent = parent

    def get(self, name, default=None):
        return self.attrs.get(name, default)

    def findall(self, tag):
        return [c for c in self.children if c.tag == tag]

    def find(self, tag):
        for c in self.children:
            if c.tag == tag:
                return c
        return None

    def iter(self, tag=None):
        if tag is None or self.tag == tag:
            yield self
        for c in self.children:
            yield from c.iter(tag)


def _tag_end(data, start):
    """Index just past the '>' that closes the start tag beginning at data[start] ('<'), quotes respected."""
    q = None
    i = start + 1
    n = len(data)
    while i < n:
        c = data[i:i + 1]
        if q is not None:
            if c == q:
                q = None
        elif c in (b'"', b"'"):
            q = c
        elif c == b">":
            return i + 1
        i += 1
    raise MakerStdError("unterminated start tag at byte %d" % start)


def parse(data, name=""):
    """Root Element of an XML document given as bytes (namespaced tags read as '{uri}local')."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    line_starts = [0]
    for k, c in enumerate(data):
        if c == 10:
            line_starts.append(k + 1)

    def line_of(pos):
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    p = xml.parsers.expat.ParserCreate(namespace_separator="}")
    stack = []
    root = []

    def qual(t):
        return "{" + t if "}" in t else t

    def start(tag, attrs):
        pos = p.CurrentByteIndex
        end = _tag_end(data, pos)
        seg = data[pos:end]
        alines = {}
        for m in _ATTR_RX.finditer(seg):
            key = m.group(1).decode("utf-8")
            if ":" in key and not key.startswith("xmlns"):
                pre, loc = key.split(":", 1)
                key = loc if pre == "xml" else key
            alines.setdefault(key, line_of(pos + m.start()))
        clean = {}
        for k, v in attrs.items():
            kk = qual(k)
            clean[kk] = v
            if kk not in alines:
                loc = kk.split("}")[-1]
                for ak, al in list(alines.items()):
                    if ak.split(":")[-1] == loc:
                        alines[kk] = al
                        break
        el = Element(qual(tag), clean, alines, line_of(pos), stack[-1] if stack else None)
        if stack:
            stack[-1].children.append(el)
        else:
            root.append(el)
        stack.append(el)

    def end(tag):
        stack.pop()

    def chars(text):
        if not stack:
            return
        el = stack[-1]
        if text.strip() and el.text_line is None:
            el.text_line = line_of(p.CurrentByteIndex)
        el.text += text

    p.StartElementHandler = start
    p.EndElementHandler = end
    p.CharacterDataHandler = chars
    p.buffer_text = False
    try:
        p.Parse(data, True)
    except xml.parsers.expat.ExpatError as exc:
        raise MakerStdError("%s: not well-formed XML: %s" % (name, exc))
    if not root:
        raise MakerStdError("%s: empty document" % name)
    return root[0]


def value(el, attr):
    """{'text', 'line'} of an attribute as written."""
    if attr not in el.attrs:
        raise MakerStdError("<%s> at line %d has no attribute %r" % (el.tag, el.line, attr))
    return {"text": el.attrs[attr], "line": el.attr_lines.get(attr, el.line)}


def numbers(text):
    """The floats of a space-separated attribute text."""
    try:
        return [float(x) for x in text.split()]
    except ValueError:
        raise MakerStdError("not a list of numbers: %r" % (text,))


def xacro(local):
    return "{" + XACRO_NS + "}" + local


# ---------------------------------------------------------------------------------------------------- readers
_ARG_RX = re.compile(r"^\$\(arg (\w+)\)$")
_COND_RX = re.compile(r"^\$\{(\w+) == '([^']*)'\}$")
_SUB_RX = re.compile(r"\$\{(\w+)\}")


def _src(file, el, attr):
    v = value(el, attr)
    return {"file": file, "line": v["line"], "attr": attr, "text": v["text"]}


def _num(file, el, attr):
    d = _src(file, el, attr)
    d["value"] = numbers(d["text"])
    return d


def _scalar(file, el, attr):
    d = _num(file, el, attr)
    if len(d["value"]) != 1:
        raise MakerStdError("%s line %d: %s is not one number" % (file, d["line"], attr))
    d["value"] = d["value"][0]
    return d


def _subst(text, env, where):
    def rep(m):
        if m.group(1) not in env:
            raise MakerStdError("%s: unsupported expression ${%s}" % (where, m.group(1)))
        return env[m.group(1)]
    out = _SUB_RX.sub(rep, text)
    if "${" in out or "$(" in out:
        raise MakerStdError("%s: unsupported expression in %r" % (where, text))
    return out


def _mesh_file(url, where):
    if not url.startswith(MESH_PREFIX):
        raise MakerStdError("%s: mesh %r is not under %s" % (where, url, MESH_PREFIX))
    return url[len(MESH_PREFIX):]


# ---------------------------------------------------------------------------------------------------- model
def read_model(text, xargs, file=MODEL_FILE):
    """The body link's visuals, the properties and the arm bay calls of the robot description. xargs: the
    description's arguments {name: {"value", "evaluated"}} as the launch file passes them."""
    MODEL = file
    root = parse(text.encode("utf-8"), MODEL)
    props = {}
    for el in root.children:
        if el.tag == xacro("property"):
            v = el.attrs.get("value", "")
            m = _ARG_RX.match(v)
            if m:
                a = xargs.get(m.group(1))
                if a is None or not a["evaluated"]:
                    continue
                props[el.attrs["name"]] = {"value": a["value"], "line": el.attr_lines.get("value", el.line),
                                           "written": v}
            else:
                props[el.attrs["name"]] = {"value": v, "line": el.attr_lines.get("value", el.line), "written": v}

    def cond(el):
        v = el.attrs.get("value", "")
        m = _COND_RX.match(v)
        if not m or m.group(1) not in props:
            raise MakerStdError("%s line %d: unsupported condition %r" % (MODEL, el.line, v))
        return props[m.group(1)]["value"] == m.group(2)

    for el in root.children:   # property overrides inside top-level conditions (the prefix rule)
        if el.tag == xacro("if"):
            m = _COND_RX.match(el.attrs.get("value", ""))
            if m and m.group(1) in props and any(c.tag == xacro("property") for c in el.children):
                if cond(el):
                    for c in el.children:
                        if c.tag == xacro("property"):
                            props[c.attrs["name"]] = {"value": c.attrs.get("value", ""),
                                                      "line": c.attr_lines.get("value", c.line),
                                                      "written": c.attrs.get("value", ""),
                                                      "by_condition_line": el.line}
    env = {k: v["value"] for k, v in props.items()}
    body = [l for l in root.findall("link") if _subst(l.attrs.get("name", ""), {"prefix": env.get("prefix", "")}, MODEL) == "body"]
    if len(body) != 1:
        raise MakerStdError("%s: expected one body link" % MODEL)
    visuals = []
    for v in body[0].findall("visual"):
        o = v.find("origin")
        mesh = v.find("geometry").find("mesh")
        url = _subst(mesh.attrs["filename"], {"ns": env.get("ns", "")}, MODEL)
        visuals.append({"name": v.attrs["name"], "name_line": v.attr_lines.get("name", v.line),
                        "xyz": _num(MODEL, o, "xyz"), "rpy": _num(MODEL, o, "rpy"),
                        "mesh": dict(_src(MODEL, mesh, "filename"), resolved=url),
                        "maker_file": _mesh_file(url, MODEL)})
    calls = []
    for el in root.children:
        if el.tag == xacro("if"):
            arm = [c for c in el.children if c.tag == xacro("perching_arm")]
            if not arm:
                continue
            active = cond(el)
            for c in arm:
                calls.append({"active": active,
                              "condition": {"file": MODEL, "line": el.attr_lines.get("value", el.line), "attr": "value",
                                            "text": el.attrs["value"]},
                              "bay": _src(MODEL, c, "bay"), "xyz": _num(MODEL, c, "xyz"), "rpy": _num(MODEL, c, "rpy")})
    on = [c for c in calls if c["active"]]
    if len(on) != 1:
        raise MakerStdError("%s: %d active arm bays under the launch defaults, expected 1" % (MODEL, len(on)))
    carriage = [el for el in root.children if el.tag == xacro("if") and "world" in el.attrs.get("value", "")]
    return {"properties": props, "env": env, "visuals": visuals, "arm_calls": calls, "arm_call": on[0],
            "world_conditions": [{"file": MODEL, "line": el.attr_lines.get("value", el.line), "attr": "value",
                                  "text": el.attrs["value"]} for el in carriage]}


# ---------------------------------------------------------------------------------------------------- arm macro
def read_arm_macro(text, call, env, file=MACRO_FILE):
    """The arm macro's links, joints (origin, axis, limits) and start values, for one bay call of read_model."""
    MACRO = file
    root = parse(text.encode("utf-8"), MACRO)
    macros = [m for m in root.children if m.tag == xacro("macro") and m.attrs.get("name") == "perching_arm"]
    if len(macros) != 1:
        raise MakerStdError("%s: expected one perching_arm macro" % MACRO)
    mac = macros[0]
    params = mac.attrs.get("params", "").split()
    if params != ["bay", "xyz", "rpy"]:
        raise MakerStdError("%s: macro parameters %r not the expected bay xyz rpy" % (MACRO, params))
    names = {"prefix": env.get("prefix", ""), "bay": call["bay"]["text"]}
    links = []
    for l in mac.findall("link"):
        v = l.find("visual")
        o = v.find("origin")
        mesh = v.find("geometry").find("mesh")
        url = mesh.attrs["filename"]
        links.append({"name": _subst(l.attrs["name"], names, MACRO),
                      "name_src": _src(MACRO, l, "name"),
                      "visual_xyz": _num(MACRO, o, "xyz"), "visual_rpy": _num(MACRO, o, "rpy"),
                      "mesh": dict(_src(MACRO, mesh, "filename"), resolved=url), "maker_file": _mesh_file(url, MACRO)})
    joints = []
    for j in mac.findall("joint"):
        o = j.find("origin")
        d = {"name": _subst(j.attrs["name"], names, MACRO), "name_src": _src(MACRO, j, "name"),
             "type": _src(MACRO, j, "type"),
             "parent": _subst(j.find("parent").attrs["link"], names, MACRO),
             "child": _subst(j.find("child").attrs["link"], names, MACRO)}
        for key in ("xyz", "rpy"):
            raw = o.attrs[key]
            if raw == "${%s}" % key:
                d[key] = dict(call[key], written_in_macro=_src(MACRO, o, key))
            else:
                d[key] = _num(MACRO, o, key)
        ax = j.find("axis")
        d["axis"] = _num(MACRO, ax, "xyz") if ax is not None else None
        lim = j.find("limit")
        d["lower"] = _scalar(MACRO, lim, "lower") if lim is not None else None
        d["upper"] = _scalar(MACRO, lim, "upper") if lim is not None else None
        joints.append(d)
    plug = [p for p in mac.iter("plugin")]
    if len(plug) != 1:
        raise MakerStdError("%s: expected one plugin block" % MACRO)
    start = {}
    for key in ("proximal", "distal", "gripper"):
        el = plug[0].find(key)
        if el is None:
            raise MakerStdError("%s: plugin block without <%s>" % (MACRO, key))
        t = el.text.strip()
        start[key] = {"file": MACRO, "line": el.text_line or el.line, "element": key, "text": t, "value": float(t)}
    return {"links": links, "joints": joints, "start": start}
