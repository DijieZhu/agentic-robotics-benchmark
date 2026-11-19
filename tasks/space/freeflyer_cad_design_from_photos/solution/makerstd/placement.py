"""Derive the placement of the standard from the four pinned texts, and check reference/placement.json against it.

Inputs (pinned under the maker root, read as text, nothing executed):
    description/model.urdf.xacro               the body link's visuals, the bay call of the arm macro, properties
    description/macro_perching_arm.urdf.xacro  the arm's links, joints, limits and its start values (plugin block)
    context/gazebo_model_plugin_perching_arm.cc  how the description's "gripper" start value maps to the four
                                               finger joints (constants, ratio, joint value, which finger takes r)
    context/astrobee.launch                    the default bay arguments and the namespace argument of the
                                               description (which skin file the default robot wears)

The two description files are read by xacro_text.read_model / read_arm_macro (the expression forms they accept are
listed there); this module reads the launch file's arguments and the plugin's text, and derives the rest. Anything
it does not recognise in a value the placement needs raises MakerStdError.

Numbers are kept as written: every value read from a file is stored as {"file", "line", "attr" | "element" |
"code", "text", "value"}; derived quantities (the frames at configuration Z, the axis lines, the finger angles) are
computed by fk.py in Python floats. placement.json is a derived file, not the standard: verify() re-derives it from
the pinned texts and compares every item (numbers within 1e-12, everything else equal).
"""
import json
import os
import re

from . import fk
from . import pinned
from . import xacro_text as xt
from .errors import MakerStdError

SCHEMA = "ffcad.placement/1"
MODEL = "description/model.urdf.xacro"
MACRO = "description/macro_perching_arm.urdf.xacro"
PLUGIN = "context/gazebo_model_plugin_perching_arm.cc"
LAUNCH = "context/astrobee.launch"
SOURCES = (MODEL, MACRO, PLUGIN, LAUNCH)
TOLERANCE = 1e-12

# The task's tags: its labelling of the description's visuals and links (one tag per visual; the arm's links).
TAG_OF_ELEMENT = (
    ("P01", "visual", "body_visual"),
    ("P02", "visual", "pmc_port_visual"),
    ("P03", "visual", "pmc_stbd_visual"),
    ("P04", "visual", "pmc_port_bumper_visual"),
    ("P05", "visual", "pmc_stbd_bumper_visual"),
    ("P06", "link", "top_aft"),
    ("P07", "link", "top_aft_arm_proximal_link"),
    ("P08", "link", "top_aft_arm_distal_link"),
    ("P09", "link", "top_aft_gripper_left_proximal_link"),
    ("P10", "link", "top_aft_gripper_left_distal_link"),
    ("P11", "link", "top_aft_gripper_right_proximal_link"),
    ("P12", "link", "top_aft_gripper_right_distal_link"),
)
SKIN_VISUALS = ("pmc_port_skin_visual", "pmc_stbd_skin_visual")
FINGER_JOINTS = ("gripper_left_proximal_joint", "gripper_left_distal_joint", "gripper_right_proximal_joint",
                 "gripper_right_distal_joint")

# ---------------------------------------------------------------------------------------------------- launch
def read_launch(text):
    root = xt.parse(text.encode("utf-8"), LAUNCH)
    args = {}
    for a in root.iter("arg"):
        if a.parent is root and "default" in a.attrs:
            args[a.attrs["name"]] = {"file": LAUNCH, "line": a.attr_lines.get("default", a.line), "attr": "default",
                                     "text": a.attrs["default"]}
    cmd = [p for p in root.iter("param") if "model.urdf.xacro" in p.attrs.get("command", "")]
    if len(cmd) != 1:
        raise MakerStdError("%s: expected one description command, found %d" % (LAUNCH, len(cmd)))
    c = cmd[0]
    command = {"file": LAUNCH, "line": c.attr_lines["command"], "attr": "command", "text": c.attrs["command"]}
    xargs = {}
    for name, val in re.findall(r'(\w+):="([^"]*)"', c.attrs["command"]):
        def rep(m):
            a = args.get(m.group(1))
            if a is None:
                raise MakerStdError("%s: launch argument %r has no default" % (LAUNCH, m.group(1)))
            return a["text"]
        resolved = re.sub(r"\$\(arg (\w+)\)", rep, val)
        xargs[name] = {"written": val, "value": resolved, "evaluated": "$(" not in resolved}
    return {"args": args, "description_command": command, "xacro_args": xargs}


# ---------------------------------------------------------------------------------------------------- plugin text
def _lines(text):
    return text.split("\n")


def _find(lines, rx, file, what, start=0, stop=None):
    hits = []
    for k in range(start, len(lines) if stop is None else stop):
        m = re.match(rx, lines[k].strip())
        if m:
            hits.append((k, m))
    if len(hits) != 1:
        raise MakerStdError("%s: %s found %d times, expected once" % (file, what, len(hits)))
    k, m = hits[0]
    return k, m, {"file": file, "line": k + 1, "code": lines[k].strip()}


def _body(lines, header_rx, file, what):
    k, _m, src = _find(lines, header_rx, file, what)
    depth = 0
    opened = False
    for e in range(k, len(lines)):
        depth += lines[e].count("{") - lines[e].count("}")
        opened = opened or "{" in lines[e]
        if opened and depth == 0:
            return k, e, src
    raise MakerStdError("%s: %s has no end" % (file, what))


def read_plugin(text):
    L = _lines(text)
    _k, m, open_src = _find(L, r"^static constexpr double GRIPPER_OPEN\s*=\s*([-0-9.eE+]+);$", PLUGIN, "GRIPPER_OPEN")
    open_src["value"] = float(m.group(1))
    _k, m, closed_src = _find(L, r"^static constexpr double GRIPPER_CLOSED\s*=\s*([-0-9.eE+]+);$", PLUGIN,
                              "GRIPPER_CLOSED")
    closed_src["value"] = float(m.group(1))
    lk, le, load_src = _body(L, r"^void LoadCallback\(", PLUGIN, "LoadCallback")
    _k, _m, grip_default = _find(L, r"^double grip = GRIPPER_CLOSED;$", PLUGIN, "the default gripper", lk, le + 1)
    _k, _m, grip_read = _find(L, r'^grip = sdf->Get<double>\("gripper"\);$', PLUGIN, "the gripper start value", lk, le + 1)
    _k, _m, grip_apply = _find(L, r"^SetGripperPosition\(grip\);$", PLUGIN, "the start gripper call", lk, le + 1)
    _k, _m, prox_read = _find(L, r'^prox = sdf->Get<double>\("proximal"\);$', PLUGIN, "the proximal start value", lk, le + 1)
    _k, _m, dist_read = _find(L, r'^dist = sdf->Get<double>\("distal"\);$', PLUGIN, "the distal start value", lk, le + 1)
    _k, _m, prox_apply = _find(L, r'^model->GetJoint\(bay_\+"_arm_proximal_joint"\)->SetPosition\(0, prox\);$', PLUGIN,
                               "the proximal start call", lk, le + 1)
    _k, _m, dist_apply = _find(L, r'^model->GetJoint\(bay_\+"_arm_distal_joint"\)->SetPosition\(0, dist\);$', PLUGIN,
                               "the distal start call", lk, le + 1)
    jk, je, jp_src = _body(L, r"^void SetGripperJointPosition\(std::string const& name, double position\) \{$", PLUGIN,
                           "SetGripperJointPosition")
    _k, _m, value_src = _find(L, r"^double value = lower \+ position \* \(upper - lower\);$", PLUGIN,
                              "the joint value formula", jk, je + 1)
    _k, _m, setpos_src = _find(L, r"^joint->SetPosition\(0, value\);$", PLUGIN, "the joint value call", jk, je + 1)
    gk, ge, gp_src = _body(L, r"^void SetGripperPosition\(double position\) \{$", PLUGIN, "SetGripperPosition")
    _k, _m, ratio_src = _find(L, r"^double r = \(position - GRIPPER_CLOSED\) / \(GRIPPER_OPEN - GRIPPER_CLOSED\);$",
                              PLUGIN, "the ratio formula", gk, ge + 1)
    takes = {}
    for k in range(gk, ge + 1):
        m = re.match(r'^SetGripperJointPosition\(bay_\+"_(\w+)", (.+)\);$', L[k].strip())
        if m:
            expr = m.group(2).strip()
            if expr not in ("r", "1.0 - r"):
                raise MakerStdError("%s line %d: finger ratio %r is not r or 1.0 - r" % (PLUGIN, k + 1, expr))
            if m.group(1) in takes:
                raise MakerStdError("%s line %d: %s set twice" % (PLUGIN, k + 1, m.group(1)))
            takes[m.group(1)] = {"file": PLUGIN, "line": k + 1, "code": L[k].strip(), "takes": expr}
    if sorted(takes) != sorted(FINGER_JOINTS):
        raise MakerStdError("%s: SetGripperPosition sets %s, expected the four finger joints" % (PLUGIN, sorted(takes)))
    table = {}
    for k, line in enumerate(L):
        m = re.match(r"^o (gripper_\w+_joint)\s+([0-9.]+)\s*\|\s*([0-9.]+)$", line.strip())
        if m:
            table[m.group(1)] = {"file": PLUGIN, "line": k + 1, "code": line.strip(),
                                 "open": float(m.group(2)), "closed": float(m.group(3))}
    return {"open": open_src, "closed": closed_src, "load_callback": load_src, "grip_default": grip_default,
            "grip_read": grip_read, "grip_apply": grip_apply, "prox_read": prox_read, "dist_read": dist_read,
            "prox_apply": prox_apply, "dist_apply": dist_apply, "joint_position_function": jp_src,
            "joint_value": value_src, "joint_set": setpos_src, "gripper_position_function": gp_src,
            "ratio": ratio_src, "takes": takes, "comment_table": table}


def finger_angles(plugin, joints_by_suffix, position):
    """The four finger joint angles the plugin sets for a gripper position (its own arithmetic, in doubles)."""
    r = (position - plugin["closed"]["value"]) / (plugin["open"]["value"] - plugin["closed"]["value"])
    out = {}
    for suffix, t in plugin["takes"].items():
        ratio = r if t["takes"] == "r" else 1.0 - r
        j = joints_by_suffix[suffix]
        lower, upper = j["lower"]["value"], j["upper"]["value"]
        out[j["name"]] = lower + ratio * (upper - lower)
    return r, out


# ---------------------------------------------------------------------------------------------------- derive
def derive_from_texts(texts):
    """The placement content from the four texts {logical path: str}."""
    launch = read_launch(texts[LAUNCH])
    model = xt.read_model(texts[MODEL], launch["xacro_args"], MODEL)
    call = model["arm_call"]
    macro = xt.read_arm_macro(texts[MACRO], call, model["env"], MACRO)
    plugin = read_plugin(texts[PLUGIN])
    bay = call["bay"]["text"]
    by_suffix = {}
    for j in macro["joints"]:
        if j["name"].startswith(bay + "_"):
            by_suffix[j["name"][len(bay) + 1:]] = j
    for s in FINGER_JOINTS:
        if s not in by_suffix:
            raise MakerStdError("%s: no joint %s_%s" % (MACRO, bay, s))
    start = macro["start"]
    rZ, fingers_Z = finger_angles(plugin, by_suffix, start["gripper"]["value"])
    r_open, fingers_open = finger_angles(plugin, by_suffix, plugin["open"]["value"])
    qZ = {by_suffix["arm_proximal_joint"]["name"]: start["proximal"]["value"],
          by_suffix["arm_distal_joint"]["name"]: start["distal"]["value"]}
    qZ.update(fingers_Z)
    q_open = dict(qZ)
    q_open.update(fingers_open)
    kin = [{"name": j["name"], "type": j["type"]["text"], "parent": j["parent"], "child": j["child"],
            "xyz": j["xyz"]["value"], "rpy": j["rpy"]["value"], "axis": j["axis"]["value"] if j["axis"] else None}
           for j in macro["joints"]]
    frames_Z, axes_Z = fk.chain(kin, qZ)
    frames_open, _ax = fk.chain(kin, q_open)
    link_of = {l["name"]: l for l in macro["links"]}
    visual_of = {v["name"]: v for v in model["visuals"]}
    tag_of_link = {}
    tags = []
    for tag, kind, element in TAG_OF_ELEMENT:
        if kind == "visual":
            v = visual_of.get(element)
            if v is None:
                raise MakerStdError("%s: no visual %s" % (MODEL, element))
            T = fk.origin(v["xyz"]["value"], v["rpy"]["value"])
            tags.append({"tag": tag, "maker_file": v["maker_file"], "description_element": element, "link": "body",
                         "visual_name_line": {"file": MODEL, "line": v["name_line"], "attr": "name", "text": element},
                         "mesh": v["mesh"], "origin_xyz": v["xyz"], "origin_rpy": v["rpy"], "T_body_file_m": T})
        else:
            l = link_of.get(element)
            if l is None:
                raise MakerStdError("%s: no link %s" % (MACRO, element))
            T = fk.mul(frames_Z[element], fk.origin(l["visual_xyz"]["value"], l["visual_rpy"]["value"]))
            tag_of_link[element] = tag
            tags.append({"tag": tag, "maker_file": l["maker_file"], "description_element": "link " + element,
                         "link": element, "link_name": l["name_src"], "mesh": l["mesh"], "origin_xyz": l["visual_xyz"],
                         "origin_rpy": l["visual_rpy"], "T_body_file_m": T})
    tag_of_link["body"] = "P01"
    joints = []
    for j in macro["joints"]:
        d = {"name": j["name"], "name_src": j["name_src"], "type": j["type"], "parent_link": j["parent"],
             "child_link": j["child"], "tags": [tag_of_link[j["parent"]], tag_of_link[j["child"]]],
             "origin_xyz": j["xyz"], "origin_rpy": j["rpy"]}
        if j["type"]["text"] == "revolute":
            ax = axes_Z[j["name"]]
            d.update({"axis": j["axis"], "lower": j["lower"], "upper": j["upper"], "value_at_Z": qZ[j["name"]],
                      "axis_line_at_Z": {"point_m": ax["point"], "point_mm": [c * 1000.0 for c in ax["point"]],
                                         "dir": ax["dir"]}})
        joints.append(d)
    skins = []
    for name in SKIN_VISUALS:
        v = visual_of.get(name)
        if v is None:
            raise MakerStdError("%s: no visual %s" % (MODEL, name))
        skins.append({"description_element": name, "maker_file": v["maker_file"],
                      "visual_name_line": {"file": MODEL, "line": v["name_line"], "attr": "name", "text": name},
                      "mesh": v["mesh"], "origin_xyz": v["xyz"], "origin_rpy": v["rpy"],
                      "T_body_file_m": fk.origin(v["xyz"]["value"], v["rpy"]["value"])})
    finger_tags = {}
    for j in macro["joints"]:
        if j["name"] in fingers_open:
            l = link_of[j["child"]]
            finger_tags[tag_of_link[j["child"]]] = fk.mul(frames_open[j["child"]],
                                                         fk.origin(l["visual_xyz"]["value"], l["visual_rpy"]["value"]))
    ns_arg = launch["xacro_args"].get("ns", {})
    out = {
        "schema": SCHEMA,
        "rule": "Derived from the four pinned texts by makerstd/placement.py; the standard is those texts, not this "
                "file. The grader re-derives it at the start of every verify and stops with a task error when any "
                "item differs (numbers by more than 1e-12). Values read from a file carry file, line and the text "
                "as written; 3.14 and 3.1415 are kept as written.",
        "frame": "the description's body frame; metres and radians; T_body_file_m maps the maker file's "
                 "coordinates (metres, node matrices applied) into the body frame at configuration Z",
        "sources": list(SOURCES),
        "launch": {"top_aft": launch["args"].get("top_aft"), "bot_aft": launch["args"].get("bot_aft"),
                   "bot_front": launch["args"].get("bot_front"), "ns": launch["args"].get("ns"),
                   "description_command_line": launch["description_command"]["line"],
                   "xacro_ns_argument": {"file": LAUNCH, "line": launch["description_command"]["line"],
                                         "written": ns_arg.get("written"), "value": ns_arg.get("value")}},
        "bay": {"name": call["bay"], "condition": call["condition"], "xyz": call["xyz"], "rpy": call["rpy"],
                "inactive_calls": [{"condition": c["condition"], "bay": c["bay"]} for c in model["arm_calls"]
                                   if not c["active"]]},
        "configuration_Z": {
            "definition": "the description's own start values (plugin block of the arm macro) with the default bay "
                          "layout of the launch file; the gripper start value maps to the four finger joints by the "
                          "plugin's arithmetic (left fingers take r, right fingers 1 - r; value = lower + ratio x "
                          "(upper - lower))",
            "proximal": start["proximal"], "distal": start["distal"], "gripper": start["gripper"],
            "gripper_ratio_r": rZ, "joint_values": qZ},
        "gripper_mapping": {
            "open_constant": plugin["open"], "closed_constant": plugin["closed"], "ratio": plugin["ratio"],
            "joint_value": plugin["joint_value"], "joint_set": plugin["joint_set"], "fingers": plugin["takes"],
            "start": {"default": plugin["grip_default"], "read": plugin["grip_read"], "applied": plugin["grip_apply"],
                      "proximal_read": plugin["prox_read"], "distal_read": plugin["dist_read"],
                      "proximal_applied": plugin["prox_apply"], "distal_applied": plugin["dist_apply"]},
            "comment_table": plugin["comment_table"]},
        "gripper_open": {
            "use": "the photographs of the open gripper only; never graded",
            "position": plugin["open"]["value"], "gripper_ratio_r": r_open, "joint_values": fingers_open,
            "T_body_file_m": finger_tags},
        "tags": tags,
        "joints": joints,
        "links_at_Z": {k: v for k, v in frames_Z.items() if k != "body"},
        "not_in_the_standard": {
            "default_skin": {"use": "the assembled photographs and the occlusion of the scoring samples only; not a "
                                    "tag, never graded", "instances": skins,
                             "file_name_rule": "the description's mesh name pmc_skin${ns}.dae with ns from the launch "
                                               "file's description command"},
            "named_skins": ["pmc_skin_bumble.dae", "pmc_skin_honey.dae", "pmc_skin_queen.dae"],
            "carriage": {"maker_file": "carriage.dae", "use": "not used: included by the description only for the "
                                                              "ground-test world", "conditions": model["world_conditions"]}},
    }
    return out


def kinematics(pl):
    """The joints of a loaded placement as fk.chain inputs, and each arm tag's link and visual origin."""
    joints = [{"name": j["name"], "type": j["type"]["text"], "parent": j["parent_link"], "child": j["child_link"],
               "xyz": j["origin_xyz"]["value"], "rpy": j["origin_rpy"]["value"],
               "axis": j["axis"]["value"] if j.get("axis") else None} for j in pl["joints"]]
    visuals = {t["tag"]: (t["link"], t["origin_xyz"]["value"], t["origin_rpy"]["value"])
               for t in pl["tags"] if t["link"] != "body"}
    return joints, visuals


def tag_transforms(pl, q=None):
    """T_body_file_m of the twelve tags with the arm's joints at q ({joint name: angle}; joints not named keep their
    configuration Z value). For the photographs' other poses; the standard itself is the stored configuration Z."""
    joints, visuals = kinematics(pl)
    values = dict(pl["configuration_Z"]["joint_values"])
    unknown = set(q or {}) - set(values)
    values.update(q or {})
    if unknown:
        raise MakerStdError("no such joint: %s" % sorted(unknown))
    frames, _axes = fk.chain(joints, values)
    out = {t["tag"]: t["T_body_file_m"] for t in pl["tags"] if t["link"] == "body"}
    for tag, (link, xyz, rpy) in visuals.items():
        out[tag] = fk.mul(frames[link], fk.origin(xyz, rpy))
    return out


def read_texts(maker_root, pins=None, check_pins=True):
    texts = {}
    for logical in SOURCES:
        if check_pins:
            texts[logical] = pinned.read_text(maker_root, logical, pins)
        else:
            with open(os.path.join(maker_root, logical), "rb") as fh:
                texts[logical] = fh.read().decode("utf-8")
    return texts


def derive(maker_root, pins=None, check_pins=True):
    """The placement content derived from the pinned texts under maker_root (pins checked unless told not to)."""
    if check_pins and pins is None:
        pins = pinned.load_pins(maker_root)
    return derive_from_texts(read_texts(maker_root, pins, check_pins))


def dumps(placement):
    return json.dumps(placement, indent=1, ensure_ascii=True) + "\n"


def load(path):
    try:
        with open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise MakerStdError("cannot read the placement %s: %s" % (path, exc))


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare(derived, stored, tol=TOLERANCE, path="placement"):
    """Every item where the two differ: [(path, derived, stored)] (numbers by more than tol, else not equal)."""
    out = []
    if isinstance(derived, dict) and isinstance(stored, dict):
        for k in derived:
            if k not in stored:
                out.append(("%s.%s" % (path, k), derived[k], "<missing>"))
            else:
                out.extend(compare(derived[k], stored[k], tol, "%s.%s" % (path, k)))
        for k in stored:
            if k not in derived:
                out.append(("%s.%s" % (path, k), "<missing>", stored[k]))
    elif isinstance(derived, (list, tuple)) and isinstance(stored, (list, tuple)):
        if len(derived) != len(stored):
            out.append((path + ".length", len(derived), len(stored)))
        else:
            for i, (a, b) in enumerate(zip(derived, stored)):
                key = a.get("tag") or a.get("name") if isinstance(a, dict) else None
                out.extend(compare(a, b, tol, "%s[%s]" % (path, key if isinstance(key, str) else i)))
    elif _is_num(derived) and _is_num(stored):
        if not (abs(float(derived) - float(stored)) <= tol):
            out.append((path, derived, stored))
    elif derived != stored:
        out.append((path, derived, stored))
    return out


def verify(maker_root, placement_path, pins=None, tol=TOLERANCE):
    """Re-derive the placement from the pinned texts and compare it with the stored file. Returns the stored
    placement; raises MakerStdError naming every differing item."""
    stored = load(placement_path)
    derived = json.loads(dumps(derive(maker_root, pins)))
    diff = compare(derived, stored, tol)
    if diff:
        shown = "; ".join("%s: derived %r, stored %r" % (p, a, b) for p, a, b in diff[:12])
        raise MakerStdError("placement.json differs from the pinned texts in %d item(s): %s" % (len(diff), shown))
    return stored
