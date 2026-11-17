"""MJCF is data: the element / attribute whitelist every chain grader applies (rule F1).

An agent's
scene.xml or robot.xml is the physics the grader steps, so it may carry nothing that buys
credit outside the task: no `<option>` beyond `timestep` in [0.0005, 0.005] and
`gravity 0 0 -9.81`, no `<compiler>` overrides beyond the fixed header, no `<include>` outside
the submission, no `file=` outside the submission, no plugins / extensions, no `<equality>`
except a declared gripper coupling, no actuators outside the robot subtree, no mocap /
gravcomp bodies, friction / solref / solimp / margin inside bands.

    ok, violations, stats = check_mjcf(path, submission_root=..., robot_root="base")

stdlib only (xml.etree). DOCTYPE / ENTITY declarations are rejected before parsing.
"""

import os
import xml.etree.ElementTree as ET

__all__ = ["check_mjcf", "DEFAULT_POLICY", "Policy"]


class Policy(object):
    """The bands and allowed names; copy and edit for a task-specific variant."""

    def __init__(self, **overrides):
        self.elements = {
            "mujoco", "compiler", "option", "size", "visual", "statistic", "default", "asset", "worldbody", "body",
            "geom", "joint", "freejoint", "site", "camera", "light", "inertial", "actuator", "contact", "exclude",
            "equality", "keyframe", "key", "sensor", "include", "mesh", "texture", "material",
            # defaults may carry any of the above element names as children
            "position", "general", "motor", "velocity", "intvelocity", "damper", "cylinder", "muscle", "adhesion",
            "weld", "connect",
            # visual / rendering children
            "global", "quality", "headlight", "map", "scale", "rgba",
            # sensor children (harmless; read-only)
            "jointpos", "jointvel", "actuatorpos", "actuatorvel", "actuatorfrc", "framepos", "framequat", "framelinvel",
            "frameangvel", "touch", "force", "torque", "accelerometer", "gyro", "velocimeter", "subtreecom",
            "subtreelinvel", "subtreeangmom", "rangefinder", "clock", "user",
        }
        self.forbidden_elements = {"plugin", "extension", "instance", "custom", "numeric", "text", "tuple", "hfield",
                                   "skin", "deformable", "flex", "flexcomp", "composite", "replicate", "frame", "tendon",
                                   "fixed", "spatial", "pair", "distance", "flex", "attach"}
        self.option_attrs = {"timestep", "gravity"}
        self.timestep_range = (0.0005, 0.005)
        self.gravity = (0.0, 0.0, -9.81)
        self.compiler_attrs = {"angle", "meshdir", "texturedir", "autolimits", "inertiafromgeom", "balanceinertia",
                               "discardvisual", "eulerseq", "strippath", "coordinate", "assetdir", "usethread", "exactmeshinertia",
                               "inertiagrouprange"}
        self.compiler_angle = "radian"
        self.friction_bands = ((0.3, 1.5), (0.0, 0.05), (0.0, 0.01))       # sliding, torsional, rolling
        self.solref_bands = ((0.002, 0.05), (0.5, 2.0))                    # timeconst, dampratio (positive form)
        self.solimp_bands = ((0.5, 0.9999), (0.5, 0.9999), (0.0, 0.01))     # d0, dwidth, width
        self.margin_max = 0.001
        self.gap_max = 0.0
        self.condim = {"1", "3", "4", "6"}
        self.body_forbidden_attrs = {"mocap", "gravcomp"}
        self.geom_forbidden_attrs = {"fluidshape", "fluidcoef"}
        self.joint_nonnegative_attrs = ("armature", "damping", "frictionloss", "stiffness")
        self.actuator_transmissions = {"joint"}                             # site / tendon / body / cranksite forbidden
        self.actuator_gear = 1.0
        self.allowed_equality_names = set()                                # e.g. the twin-finger coupling joints
        self.robot_root = "base"
        self.max_bodies = 512
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise TypeError("unknown policy field {}".format(k))
            setattr(self, k, v)


DEFAULT_POLICY = Policy()


def _floats(text):
    try:
        return [float(v) for v in text.replace(",", " ").split()]
    except ValueError:
        return None


def _resolve(base_dir, sub, f):
    """Resolved path of a referenced file: symlinks followed (realpath), so a link that points outside the
    submission is judged by where it points, not by where it sits."""
    p = f if os.path.isabs(f) else os.path.join(base_dir, sub or "", f)
    return os.path.realpath(os.path.normpath(os.path.abspath(p)))


def _inside(path, root):
    root = os.path.realpath(os.path.normpath(os.path.abspath(root)))
    return path == root or path.startswith(root + os.sep)


def _body_names_under(body):
    return set(b.attrib.get("name") for b in body.iter("body") if b.attrib.get("name"))


def check_mjcf(path_or_xml, submission_root=None, policy=None, robot_root=None, _depth=0, _seen=None):
    """Return (ok, violations, stats). `submission_root` bounds every `file=` and `<include>`.

    `robot_root` (default policy.robot_root, "base"): the body whose subtree is the robot; when
    absent from the file (a robot.xml alone has it; a mug-only scene would not) actuator and
    equality checks fall back to "none allowed outside a robot".
    """
    policy = policy or DEFAULT_POLICY
    robot_root = robot_root or policy.robot_root
    violations = []
    stats = {"elements": 0, "bodies": 0, "geoms": 0, "actuators": 0, "includes": 0, "files": 0}
    is_file = os.path.isfile(str(path_or_xml))
    if is_file:
        base_dir = os.path.dirname(os.path.abspath(path_or_xml))
        with open(path_or_xml, "rb") as f:
            raw = f.read()
        submission_root = submission_root or base_dir
    else:
        base_dir = submission_root or os.getcwd()
        raw = path_or_xml.encode() if isinstance(path_or_xml, str) else path_or_xml
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        return False, ["DOCTYPE / ENTITY declarations are not allowed"], stats
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return False, ["XML parse error: {}".format(exc)], stats
    if root.tag != "mujoco":
        return False, ["root element is <{}>, not <mujoco>".format(root.tag)], stats

    comp = root.find("compiler")
    meshdir = comp.attrib.get("meshdir", "") if comp is not None else ""
    texdir = comp.attrib.get("texturedir", meshdir) if comp is not None else ""
    robot_body = None
    for b in root.iter("body"):
        if b.attrib.get("name") == robot_root:
            robot_body = b
            break
    robot_bodies = _body_names_under(robot_body) if robot_body is not None else set()
    robot_joints = set(j.attrib.get("name") for j in (robot_body.iter("joint") if robot_body is not None else []) if j.attrib.get("name"))

    for elem in root.iter():
        stats["elements"] += 1
        tag = elem.tag
        if tag in policy.forbidden_elements:
            violations.append("<{}> is forbidden".format(tag))
            continue
        if tag not in policy.elements:
            violations.append("<{}> is not in the whitelist".format(tag))
            continue
        if tag == "option":
            for k, v in elem.attrib.items():
                if k not in policy.option_attrs:
                    violations.append("<option {}=...> is not allowed (only timestep, gravity)".format(k))
                elif k == "timestep":
                    ts = _floats(v)
                    if not ts or not (policy.timestep_range[0] <= ts[0] <= policy.timestep_range[1]):
                        violations.append("<option timestep={}> outside {}".format(v, policy.timestep_range))
                elif k == "gravity":
                    g = _floats(v)
                    if not g or len(g) != 3 or any(abs(a - b) > 1e-9 for a, b in zip(g, policy.gravity)):
                        violations.append("<option gravity={}> must be {}".format(v, " ".join(str(x) for x in policy.gravity)))
            for child in elem:
                violations.append("<option><{}> is not allowed".format(child.tag))
        elif tag == "compiler":
            for k, v in elem.attrib.items():
                if k not in policy.compiler_attrs:
                    violations.append("<compiler {}=...> is not allowed".format(k))
                if k == "angle" and v != policy.compiler_angle:
                    violations.append("<compiler angle={}> must be {}".format(v, policy.compiler_angle))
                if k in ("meshdir", "texturedir", "assetdir"):
                    if os.path.isabs(v) or ".." in v.split(os.sep) or ".." in v.split("/"):
                        violations.append("<compiler {}={}> must be a relative path inside the submission".format(k, v))
        elif tag == "include":
            stats["includes"] += 1
            f = elem.attrib.get("file", "")
            p = _resolve(base_dir, "", f)
            if not submission_root or not _inside(p, submission_root):
                violations.append("<include file={}> resolves outside the submission".format(f))
            elif not os.path.isfile(p):
                violations.append("<include file={}> does not exist".format(f))
            elif _depth > 4 or (_seen and p in _seen):
                violations.append("<include> nesting too deep or cyclic at {}".format(f))
            else:
                seen = set(_seen or ()) | {p}
                ok_i, v_i, s_i = check_mjcf(p, submission_root, policy, robot_root, _depth + 1, seen)
                violations += ["{}: {}".format(os.path.basename(f), v) for v in v_i]
                for k in stats:
                    stats[k] += s_i.get(k, 0)
        elif tag == "body":
            stats["bodies"] += 1
            for k in policy.body_forbidden_attrs:
                if k in elem.attrib and elem.attrib[k] not in ("false", "0", "0.0"):
                    violations.append("<body name={} {}={}> is not allowed".format(elem.attrib.get("name"), k, elem.attrib[k]))
        elif tag == "geom":
            stats["geoms"] += 1
            _check_geom_attrs(elem, policy, violations, "geom {}".format(elem.attrib.get("name", "?")))
        elif tag == "joint":
            for k in policy.joint_nonnegative_attrs:
                if k in elem.attrib:
                    vals = _floats(elem.attrib[k])
                    if vals is None or vals[0] < 0:
                        violations.append("<joint name={} {}={}> must be >= 0".format(elem.attrib.get("name"), k, elem.attrib[k]))
            _check_solref_solimp(elem, policy, violations, "joint {}".format(elem.attrib.get("name", "?")), ("solreflimit", "solimplimit", "solreffriction", "solimpfriction"))
        elif tag == "default":
            pass   # children are checked as their own tags (geom / joint / ... under default)
        if tag in ("mesh", "texture", "skin"):
            f = elem.attrib.get("file")
            if f:
                stats["files"] += 1
                sub = texdir if tag == "texture" else meshdir
                p = _resolve(base_dir, sub, f)
                if os.path.isabs(f) or not submission_root or not _inside(p, submission_root):
                    violations.append("<{} file={}> resolves outside the submission".format(tag, f))
        if tag == "equality":
            for eq in elem:
                names = [eq.attrib.get(k) for k in ("body1", "body2", "joint1", "joint2", "site1", "site2") if eq.attrib.get(k)]
                if not names or not all(n in policy.allowed_equality_names for n in names):
                    violations.append("<equality><{} {}> is not a declared gripper coupling (no equality on task objects)".format(
                        eq.tag, " ".join("{}={}".format(k, v) for k, v in eq.attrib.items())))
        if tag == "contact":
            for ex in elem:
                if ex.tag == "exclude":
                    b1, b2 = ex.attrib.get("body1"), ex.attrib.get("body2")
                    if robot_bodies and not (b1 in robot_bodies and b2 in robot_bodies):
                        violations.append("<contact><exclude body1={} body2={}> involves a non-robot body".format(b1, b2))
                    elif not robot_bodies:
                        violations.append("<contact><exclude> without a robot subtree")
        if tag == "actuator":
            for a in elem:
                stats["actuators"] += 1
                trn = [k for k in ("joint", "site", "tendon", "body", "cranksite", "slidersite", "jointinparent") if k in a.attrib]
                if not trn or trn[0] not in policy.actuator_transmissions:
                    violations.append("actuator {} transmission {} is not allowed (joint only)".format(a.attrib.get("name"), trn))
                    continue
                j = a.attrib.get("joint")
                if j not in robot_joints:
                    violations.append("actuator {} drives joint {} outside the robot subtree".format(a.attrib.get("name"), j))
                gear = _floats(a.attrib.get("gear", "1"))
                if gear is None or abs(gear[0] - policy.actuator_gear) > 1e-9 or any(abs(g) > 1e-9 for g in gear[1:]):
                    violations.append("actuator {} gear={} must be {}".format(a.attrib.get("name"), a.attrib.get("gear"), policy.actuator_gear))
    if stats["bodies"] > policy.max_bodies:
        violations.append("{} bodies exceed the limit {}".format(stats["bodies"], policy.max_bodies))
    return (not violations), violations, stats


def _check_geom_attrs(elem, policy, violations, label):
    a = elem.attrib
    for k in policy.geom_forbidden_attrs:
        if k in a:
            violations.append("{}: {}= is not allowed".format(label, k))
    if "friction" in a:
        fr = _floats(a["friction"])
        if fr is None:
            violations.append("{}: friction unparsable".format(label))
        else:
            for v, (lo, hi), nm in zip(fr, policy.friction_bands, ("sliding", "torsional", "rolling")):
                if not (lo <= v <= hi):
                    violations.append("{}: {} friction {} outside [{}, {}]".format(label, nm, v, lo, hi))
    if "margin" in a:
        m = _floats(a["margin"])
        if m is None or m[0] > policy.margin_max or m[0] < 0:
            violations.append("{}: margin {} outside [0, {}]".format(label, a["margin"], policy.margin_max))
    if "gap" in a:
        g = _floats(a["gap"])
        if g is None or g[0] > policy.gap_max or g[0] < 0:
            violations.append("{}: gap {} not allowed".format(label, a["gap"]))
    if "condim" in a and a["condim"] not in policy.condim:
        violations.append("{}: condim {} not in {}".format(label, a["condim"], sorted(policy.condim)))
    if "priority" in a:
        pr = _floats(a["priority"])
        if pr is None or pr[0] != 0:
            violations.append("{}: priority {} not allowed (contact parameter override)".format(label, a["priority"]))
    _check_solref_solimp(elem, policy, violations, label, ("solref", "solimp", "solmix"))


def _check_solref_solimp(elem, policy, violations, label, keys):
    a = elem.attrib
    for k in keys:
        if k not in a:
            continue
        vals = _floats(a[k])
        if vals is None:
            violations.append("{}: {} unparsable".format(label, k))
            continue
        if k.startswith("solref"):
            if len(vals) < 2 or vals[0] <= 0 or vals[1] <= 0:
                violations.append("{}: {}={} must be the positive (timeconst, dampratio) form".format(label, k, a[k]))
            else:
                for v, (lo, hi), nm in zip(vals, policy.solref_bands, ("timeconst", "dampratio")):
                    if not (lo <= v <= hi):
                        violations.append("{}: {} {} {} outside [{}, {}]".format(label, k, nm, v, lo, hi))
        elif k.startswith("solimp"):
            for v, (lo, hi), nm in zip(vals[:3], policy.solimp_bands, ("d0", "dwidth", "width")):
                if not (lo <= v <= hi):
                    violations.append("{}: {} {} {} outside [{}, {}]".format(label, k, nm, v, lo, hi))
            if len(vals) >= 2 and vals[0] > vals[1]:
                violations.append("{}: {} d0 > dwidth".format(label, k))
        elif k == "solmix":
            if vals[0] < 0 or vals[0] > 1e3:
                violations.append("{}: solmix {} outside [0, 1000]".format(label, vals[0]))
